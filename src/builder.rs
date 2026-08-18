//! Offline compiler for the SPARROW filter.
//!
//! Given a pattern set and a byte-distribution model, this module chooses
//! (1) a sparse set of sampled positions inside the common anchor window and
//! (2) an assignment of patterns to the 8 SIMD buckets, by minimizing an
//! *exact* expected-verification-cost objective that models the PSHUFB
//! nibble cross-product closure. See `docs/DESIGN.md` for the theorems this
//! implementation realizes.

use crate::BuildError;

/// Maximum number of sampled positions (one shuffle pair + shift + AND each).
pub const MAX_K: usize = 4;
/// Number of buckets: one per bit of a byte lane.
pub const NUM_BUCKETS: usize = 8;
/// The anchor window is limited so that all shift distances fit in a single
/// `alignr` carry (<= 15 bytes).
pub const MAX_WINDOW: usize = 16;

/// Fixed per-candidate overhead (bit extraction, bounds checks) measured in
/// units of "one pattern memcmp", used by the cost objective.
const CANDIDATE_OVERHEAD: f64 = 2.0;

/// A compiled SPARROW matcher: filter tables + verification data.
pub struct Compiled {
    pub(crate) k: usize,
    /// Sampled positions, ascending. Offsets from the pattern start.
    pub(crate) positions: Vec<u8>,
    /// shifts[j] = positions[k-1] - positions[j] (the alignr carry distance).
    pub(crate) shifts: Vec<u8>,
    /// positions[k-1]: candidate bits are anchored at pattern_start + s_last.
    pub(crate) s_last: usize,
    /// Low/high nibble -> bucket-bitmap shuffle tables, one pair per position.
    pub(crate) tl: [[u8; 16]; MAX_K],
    pub(crate) th: [[u8; 16]; MAX_K],
    /// byte -> bucket-bitmap tables (the nibble closure, precomposed) for the
    /// scalar engine. Identical semantics to the SIMD path by construction.
    pub(crate) byte_tbl: Vec<[u8; 256]>,
    /// Pattern ids per bucket.
    pub(crate) buckets: Vec<Vec<u32>>,
    pub(crate) patterns: Vec<Box<[u8]>>,
    pub(crate) min_len: usize,
    /// The model-expected verification cost per haystack byte (the minimized
    /// objective). Exposed for inspection and ablation.
    pub(crate) expected_cost: f64,
    /// Model-expected candidate bits per haystack byte.
    pub(crate) expected_candidates: f64,
}

/// A small built-in mixed corpus (English prose + markup + code + digits)
/// used to estimate the background byte distribution when the caller does
/// not provide one. Laplace smoothing guarantees every byte value has
/// non-zero probability, so binary inputs are still modeled sanely.
const DEFAULT_CORPUS: &[u8] = br#"
The quick brown fox jumps over the lazy dog, and then the dog got up and
walked to the river to drink. It was a warm afternoon; the light came in
low over the water and everything smelled of cut grass. She said that the
report would be ready by Thursday, but the numbers from the eastern region
were still missing, and nobody could find the spreadsheet that had them.
GET /api/v1/users?page=2&limit=50 HTTP/1.1
Host: example.com
Content-Type: application/json; charset=utf-8
{"status": 200, "items": [1, 2, 3], "next": "/api/v1/users?page=3"}
fn main() { let total: u64 = (0..100).map(|x| x * x).sum(); println!("{}", total); }
for (int i = 0; i < n; i++) { sum += values[i]; }
<div class="container"><p id="intro">Hello, world!</p></div>
SELECT name, count(*) FROM orders WHERE created_at > '2025-01-01' GROUP BY name;
In the beginning there was only the sea, and the sky above it, and between
them a wind that moved without purpose. 0123456789 aeiou AEIOU etaoin shrdlu
the of and to in is was he for it with as his on be at by had not are but
"#;

/// Byte-distribution model: probabilities for each byte value and the
/// 16x16 (high-nibble x low-nibble) matrix view used by the optimizer.
struct Model {
    pi: [f64; 256],
    mat: [[f64; 16]; 16], // mat[hi][lo]
}

impl Model {
    fn from_corpus(corpus: &[u8]) -> Model {
        let mut cnt = [1u64; 256]; // Laplace smoothing
        for &b in corpus {
            cnt[b as usize] += 1;
        }
        let total: u64 = cnt.iter().sum();
        let mut pi = [0f64; 256];
        for i in 0..256 {
            pi[i] = cnt[i] as f64 / total as f64;
        }
        let mut mat = [[0f64; 16]; 16];
        for x in 0..256 {
            mat[x >> 4][x & 15] = pi[x];
        }
        Model { pi, mat }
    }
}

/// Incrementally-maintained state of one bucket during assignment: nibble
/// sets per sampled position, with exact closure probabilities.
#[derive(Clone)]
struct BucketState {
    n: usize,
    lo_cnt: Vec<[u32; 16]>,  // [k][nibble] membership counts (for removal)
    hi_cnt: Vec<[u32; 16]>,
    lo_set: Vec<u16>,        // [k] bitmask of low nibbles present
    hi_set: Vec<u16>,
    row_tot: Vec<[f64; 16]>, // row_tot[j][h] = sum_{l in LoSet_j} mat[h][l]
    col_tot: Vec<[f64; 16]>, // col_tot[j][l] = sum_{h in HiSet_j} mat[h][l]
    prob: Vec<f64>,          // closure probability per position
}

impl BucketState {
    fn new(k: usize) -> BucketState {
        BucketState {
            n: 0,
            lo_cnt: vec![[0; 16]; k],
            hi_cnt: vec![[0; 16]; k],
            lo_set: vec![0; k],
            hi_set: vec![0; k],
            row_tot: vec![[0.0; 16]; k],
            col_tot: vec![[0.0; 16]; k],
            prob: vec![0.0; k],
        }
    }

    /// Expected verification cost contributed by this bucket, per byte:
    /// (overhead + |bucket|) * prod_j Pr[closure class at position j].
    fn cost(&self) -> f64 {
        if self.n == 0 {
            return 0.0;
        }
        (CANDIDATE_OVERHEAD + self.n as f64) * self.candidate_prob()
    }

    fn candidate_prob(&self) -> f64 {
        self.prob.iter().product()
    }

    /// Cost this bucket would have after adding the pattern whose sampled
    /// bytes are `bytes` (no mutation).
    fn cost_with(&self, bytes: &[u8], model: &Model) -> f64 {
        let mut prod = 1.0;
        for (j, &b) in bytes.iter().enumerate() {
            let lo = (b & 15) as usize;
            let hi = (b >> 4) as usize;
            let lo_new = self.lo_set[j] & (1 << lo) == 0;
            let hi_new = self.hi_set[j] & (1 << hi) == 0;
            let mut p = self.prob[j];
            if lo_new {
                p += self.col_tot[j][lo];
            }
            if hi_new {
                p += self.row_tot[j][hi];
            }
            if lo_new && hi_new {
                p += model.mat[hi][lo];
            }
            prod *= p;
        }
        (CANDIDATE_OVERHEAD + (self.n + 1) as f64) * prod
    }

    fn add(&mut self, bytes: &[u8], model: &Model) {
        for (j, &b) in bytes.iter().enumerate() {
            let lo = (b & 15) as usize;
            let hi = (b >> 4) as usize;
            let lo_new = self.lo_set[j] & (1 << lo) == 0;
            let hi_new = self.hi_set[j] & (1 << hi) == 0;
            if lo_new {
                self.prob[j] += self.col_tot[j][lo];
            }
            if hi_new {
                self.prob[j] += self.row_tot[j][hi];
            }
            if lo_new && hi_new {
                self.prob[j] += model.mat[hi][lo];
            }
            if lo_new {
                self.lo_set[j] |= 1 << lo;
                for h in 0..16 {
                    self.row_tot[j][h] += model.mat[h][lo];
                }
            }
            if hi_new {
                self.hi_set[j] |= 1 << hi;
                for l in 0..16 {
                    self.col_tot[j][l] += model.mat[hi][l];
                }
            }
            self.lo_cnt[j][lo] += 1;
            self.hi_cnt[j][hi] += 1;
        }
        self.n += 1;
    }

    /// Rebuild from scratch for a member list (used after removals, where
    /// incremental downdating of the closure sums would be error-prone).
    fn rebuild(k: usize, members: &[u32], sampled: &[Vec<u8>], model: &Model) -> BucketState {
        let mut s = BucketState::new(k);
        for &id in members {
            s.add(&sampled[id as usize], model);
        }
        s
    }
}

/// Result of assigning all patterns to buckets for one position set.
struct Assignment {
    buckets: Vec<Vec<u32>>,
    states: Vec<BucketState>,
    total_cost: f64,
}

fn greedy_assign(sampled: &[Vec<u8>], k: usize, model: &Model) -> Assignment {
    let n = sampled.len();
    // Process common (high solo-probability) patterns first so they claim
    // buckets early; rare patterns fit almost anywhere with little damage.
    let mut order: Vec<usize> = (0..n).collect();
    let solo: Vec<f64> = sampled
        .iter()
        .map(|bs| bs.iter().map(|&b| model.pi[b as usize]).product::<f64>())
        .collect();
    order.sort_by(|&a, &b| solo[b].partial_cmp(&solo[a]).unwrap());

    let mut states: Vec<BucketState> = (0..NUM_BUCKETS).map(|_| BucketState::new(k)).collect();
    let mut buckets: Vec<Vec<u32>> = vec![Vec::new(); NUM_BUCKETS];
    for &i in &order {
        let mut best = 0usize;
        let mut best_delta = f64::INFINITY;
        for b in 0..NUM_BUCKETS {
            let delta = states[b].cost_with(&sampled[i], model) - states[b].cost();
            if delta < best_delta {
                best_delta = delta;
                best = b;
            }
        }
        states[best].add(&sampled[i], model);
        buckets[best].push(i as u32);
    }
    let total_cost = states.iter().map(|s| s.cost()).sum();
    Assignment { buckets, states, total_cost }
}

/// Local-search refinement: keep moving single patterns to strictly better
/// buckets until a fixpoint (or a safety cap). Each accepted move strictly
/// decreases the total cost, so this terminates (see DESIGN.md, Thm 4).
fn refine(a: &mut Assignment, sampled: &[Vec<u8>], k: usize, model: &Model) {
    const MAX_PASSES: usize = 32;
    const MIN_GAIN: f64 = 1e-15;
    for _ in 0..MAX_PASSES {
        let mut improved = false;
        for b0 in 0..NUM_BUCKETS {
            let mut idx = 0;
            while idx < a.buckets[b0].len() {
                let id = a.buckets[b0][idx];
                // Cost of b0 without this pattern.
                let members_without: Vec<u32> =
                    a.buckets[b0].iter().copied().filter(|&x| x != id).collect();
                let state_without = BucketState::rebuild(k, &members_without, sampled, model);
                let removal_gain = a.states[b0].cost() - state_without.cost();
                let mut best_b = b0;
                let mut best_gain = MIN_GAIN;
                for b in 0..NUM_BUCKETS {
                    if b == b0 {
                        continue;
                    }
                    let add_cost =
                        a.states[b].cost_with(&sampled[id as usize], model) - a.states[b].cost();
                    let gain = removal_gain - add_cost;
                    if gain > best_gain {
                        best_gain = gain;
                        best_b = b;
                    }
                }
                if best_b != b0 {
                    a.buckets[b0].retain(|&x| x != id);
                    a.states[b0] = state_without;
                    a.states[best_b].add(&sampled[id as usize], model);
                    a.buckets[best_b].push(id);
                    a.total_cost = a.states.iter().map(|s| s.cost()).sum();
                    improved = true;
                    // Do not advance idx: current slot now holds the next id.
                } else {
                    idx += 1;
                }
            }
        }
        if !improved {
            break;
        }
    }
}

/// Enumerate candidate sampled-position sets.
fn candidate_position_sets(
    patterns: &[Box<[u8]>],
    w: usize,
    max_k: usize,
    model: &Model,
    exhaustive: bool,
) -> Vec<Vec<u8>> {
    let kmax = max_k.min(w).min(MAX_K);
    let pool: Vec<u8> = if exhaustive {
        (0..w as u8).collect()
    } else {
        // Rank positions by expected rarity of the pattern bytes there
        // (lower total probability = more discriminating), keep the top 9.
        let mut scored: Vec<(f64, u8)> = (0..w)
            .map(|s| {
                let tot: f64 = patterns.iter().map(|p| model.pi[p[s] as usize]).sum();
                (tot, s as u8)
            })
            .collect();
        scored.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        scored.truncate(9);
        let mut v: Vec<u8> = scored.into_iter().map(|(_, s)| s).collect();
        v.sort_unstable();
        v
    };

    let mut sets: Vec<Vec<u8>> = Vec::new();
    // All subsets of the pool of size 1..=kmax.
    let m = pool.len();
    for mask in 1u32..(1u32 << m) {
        let size = mask.count_ones() as usize;
        if size > kmax {
            continue;
        }
        let mut s: Vec<u8> = Vec::with_capacity(size);
        for (i, &p) in pool.iter().enumerate() {
            if mask & (1 << i) != 0 {
                s.push(p);
            }
        }
        sets.push(s);
    }
    // Always include the Teddy-style contiguous prefix as a baseline so the
    // optimizer can never do worse (under the model) than the classic choice.
    let prefix: Vec<u8> = (0..kmax as u8).collect();
    if !sets.contains(&prefix) {
        sets.push(prefix);
    }
    sets
}

pub(crate) fn build(
    patterns: Vec<Box<[u8]>>,
    corpus: Option<&[u8]>,
    max_k: usize,
    forced_positions: Option<&[u8]>,
    exhaustive: bool,
) -> Result<Compiled, BuildError> {
    if patterns.is_empty() {
        return Err(BuildError::NoPatterns);
    }
    if patterns.iter().any(|p| p.is_empty()) {
        return Err(BuildError::EmptyPattern);
    }
    let max_k = max_k.clamp(1, MAX_K);
    let min_len = patterns.iter().map(|p| p.len()).min().unwrap();
    let w = min_len.min(MAX_WINDOW);
    let model = Model::from_corpus(corpus.unwrap_or(DEFAULT_CORPUS));

    let position_sets: Vec<Vec<u8>> = match forced_positions {
        Some(ps) => {
            let mut v: Vec<u8> = ps.to_vec();
            v.sort_unstable();
            v.dedup();
            if v.is_empty() || v.len() > MAX_K {
                return Err(BuildError::BadPositions);
            }
            if v.iter().any(|&s| (s as usize) >= w) {
                return Err(BuildError::BadPositions);
            }
            vec![v]
        }
        None => candidate_position_sets(&patterns, w, max_k, &model, exhaustive),
    };

    // Score every candidate set with a greedy assignment, then run the
    // (more expensive) local-search refinement on the best few — always
    // including the Teddy-style contiguous prefix, so the final choice is
    // never worse than the classic one under the model (Thm 5).
    let sample = |pos: &[u8]| -> Vec<Vec<u8>> {
        patterns.iter().map(|p| pos.iter().map(|&s| p[s as usize]).collect()).collect()
    };
    let mut scored: Vec<(f64, Vec<u8>)> = position_sets
        .iter()
        .map(|pos| {
            let a = greedy_assign(&sample(pos), pos.len(), &model);
            (a.total_cost, pos.clone())
        })
        .collect();
    scored.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    let kmax = max_k.min(w).min(MAX_K);
    let prefix: Vec<u8> = (0..kmax as u8).collect();
    let mut finalists: Vec<Vec<u8>> = scored.iter().take(3).map(|(_, p)| p.clone()).collect();
    if position_sets.contains(&prefix) && !finalists.contains(&prefix) {
        finalists.push(prefix);
    }
    let mut best: Option<(Vec<u8>, Vec<Vec<u8>>, Assignment)> = None;
    for pos in finalists {
        let sampled = sample(&pos);
        let mut a = greedy_assign(&sampled, pos.len(), &model);
        refine(&mut a, &sampled, pos.len(), &model);
        let better = match &best {
            None => true,
            Some((_, _, cur)) => a.total_cost < cur.total_cost,
        };
        if better {
            best = Some((pos, sampled, a));
        }
    }
    let (positions, sampled, assignment) = best.unwrap();
    let k = positions.len();

    // Materialize the shuffle tables.
    let s_last = *positions.last().unwrap() as usize;
    let shifts: Vec<u8> = positions.iter().map(|&s| (s_last as u8) - s).collect();
    let mut tl = [[0u8; 16]; MAX_K];
    let mut th = [[0u8; 16]; MAX_K];
    for (b, members) in assignment.buckets.iter().enumerate() {
        for &id in members {
            for (j, &byte) in sampled[id as usize].iter().enumerate() {
                tl[j][(byte & 15) as usize] |= 1 << b;
                th[j][(byte >> 4) as usize] |= 1 << b;
            }
        }
    }
    let mut byte_tbl = vec![[0u8; 256]; k];
    for j in 0..k {
        for x in 0..256usize {
            byte_tbl[j][x] = tl[j][x & 15] & th[j][x >> 4];
        }
    }
    let expected_candidates: f64 = assignment.states.iter().map(|s| s.candidate_prob()).sum();
    let expected_cost = assignment.total_cost;

    Ok(Compiled {
        k,
        positions,
        shifts,
        s_last,
        tl,
        th,
        byte_tbl,
        buckets: assignment.buckets,
        patterns,
        min_len,
        expected_cost,
        expected_candidates,
    })
}
