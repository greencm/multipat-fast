//! Offline compiler for the SPARROW filter.
//!
//! Given a pattern set and a byte-distribution model, this module chooses
//! (1) a sparse set of sampled positions inside the common anchor window,
//! (2) an assignment of patterns to 8 or 16 SIMD buckets, and (3) how many
//! positions and bucket planes to spend, by minimizing a total model cost =
//! expected verification work + SIMD scan work. The verification term is
//! exact under the model and includes the PSHUFB nibble cross-product
//! closure; finalist configurations are re-scored under a first-order
//! empirical scan of the corpus sample, which captures correlation no
//! closed-form byte model can. Patterns may be split into length cohorts
//! (one compiled filter each) when the model says splitting is cheaper.
//! See `docs/DESIGN.md` for the theorems this implementation realizes.

use crate::{BuildError, MatchOpts};

/// Maximum number of sampled positions (one shuffle pair per plane each).
pub const MAX_K: usize = 4;
/// Buckets per plane: one per bit of a byte lane.
pub const PLANE_BUCKETS: usize = 8;
/// 1 plane = 8 buckets, 2 planes = 16 buckets.
pub const MAX_PLANES: usize = 2;
/// The anchor window. The offset-load kernels start SIMD blocks at offset
/// 32, so any spread of positions within 32 bytes is loadable directly.
pub const MAX_WINDOW: usize = 32;

/// Fixed per-candidate overhead (mask store, bit extraction, bounds checks)
/// in units of "one guard probe / pattern comparison". Measured: a block
/// with any candidate pays a mispredicted branch + vector store + bit loop
/// on top of the per-entry probes, several times the cost of one probe.
const CANDIDATE_OVERHEAD: f64 = 6.0;
/// Model cost, per haystack byte, of one sampled position on one plane
/// (shuffle pair + AND amortized over a SIMD block), in the same units.
/// This is what lets the optimizer decide that a 5th filter term isn't
/// worth its throughput cost — or that a 2nd bucket plane is. Calibrated
/// against measured kernels: one filter term costs ~0.1 ns/byte while one
/// verification unit (guard probe + bookkeeping) costs ~10 ns, i.e. a term
/// is worth paying whenever it saves ~2e-3 expected comparisons per byte.
const SCAN_COST_PER_TERM: f64 = 0.002;
/// Bands smaller than this get merged into a neighboring length cohort.
const MIN_COHORT: usize = 8;

pub(crate) struct BuildOpts<'a> {
    pub corpus: Option<&'a [u8]>,
    pub max_k: usize,
    pub forced_positions: Option<&'a [u8]>,
    pub exhaustive: bool,
    pub corpus_score: bool,
    pub match_opts: MatchOpts,
}

/// One bucket entry: pattern id plus a precomputed guard probe — the
/// model-rarest pattern byte outside the sampled positions. Verification
/// checks the guard byte before running the full comparison.
#[derive(Clone, Copy)]
pub(crate) struct Entry {
    pub id: u32,
    pub guard_off: u32,
    pub guard_byte: u8,
}

/// A compiled filter for one length cohort.
pub struct Compiled {
    pub(crate) k: usize,
    pub(crate) planes: usize,
    /// Sampled positions, ascending. Offsets from the pattern start.
    pub(crate) positions: Vec<u8>,
    /// d[j] = positions[k-1] - positions[j]; kernels load hay at (block - d[j]).
    pub(crate) d: [usize; MAX_K],
    /// Candidate bits are anchored at pattern_start + s_last.
    pub(crate) s_last: usize,
    /// Low/high nibble -> bucket-bitmap shuffle tables per plane/position.
    pub(crate) tl: [[[u8; 16]; MAX_K]; MAX_PLANES],
    pub(crate) th: [[[u8; 16]; MAX_K]; MAX_PLANES],
    /// byte -> bucket-bitmap tables (nibble closure precomposed) for the
    /// scalar engine; identical semantics to the SIMD path by construction.
    pub(crate) byte_tbl: Vec<Vec<[u8; 256]>>, // [plane][j]
    /// Entries per bucket; bucket index = plane * 8 + bit.
    pub(crate) buckets: Vec<Vec<Entry>>,
    pub(crate) min_len: usize,
    /// Total model cost per haystack byte (the minimized objective:
    /// verification + scan terms, under the selection model).
    pub(crate) expected_cost: f64,
    /// Model-expected candidate bits per haystack byte (i.i.d. model).
    pub(crate) expected_candidates: f64,
}

/// A small built-in mixed corpus (English prose + markup + code + digits)
/// used to estimate the background byte distribution when the caller does
/// not provide one. Laplace smoothing guarantees every byte value (and
/// transition) has non-zero probability, so binary inputs stay modeled.
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

/// The byte class a pattern presents at one position: sets of low and high
/// nibbles. Exact bytes get singleton nibbles; ASCII case-insensitive
/// letters contribute both cases; a wildcard byte contributes everything.
/// The filter can only ever represent the nibble cross product, so the
/// class *is* its (Lo, Hi) pair and probability math on it is exact.
#[derive(Clone, Copy, PartialEq)]
pub(crate) struct Class {
    lo: u16,
    hi: u16,
}

pub(crate) fn class_of(byte: u8, opts: &MatchOpts) -> Class {
    if opts.wildcard == Some(byte) {
        return Class { lo: 0xFFFF, hi: 0xFFFF };
    }
    let mut lo = 1u16 << (byte & 15);
    let mut hi = 1u16 << (byte >> 4);
    if opts.case_insensitive && byte.is_ascii_alphabetic() {
        let other = byte ^ 0x20;
        lo |= 1 << (other & 15);
        hi |= 1 << (other >> 4);
    }
    Class { lo, hi }
}

/// i.i.d. byte-distribution model.
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

    /// Exact probability of the nibble-closure of a (Lo, Hi) class union.
    fn closure_prob(&self, lo: u16, hi: u16) -> f64 {
        let mut p = 0.0;
        for h in 0..16 {
            if hi & (1 << h) != 0 {
                let row = &self.mat[h];
                for l in 0..16 {
                    if lo & (1 << l) != 0 {
                        p += row[l];
                    }
                }
            }
        }
        p
    }

    /// Model probability that a haystack byte satisfies `byte_matches`
    /// against this pattern byte (used for guard selection).
    fn match_prob(&self, byte: u8, opts: &MatchOpts) -> f64 {
        if opts.wildcard == Some(byte) {
            return 1.0;
        }
        let mut p = self.pi[byte as usize];
        if opts.case_insensitive && byte.is_ascii_alphabetic() {
            p += self.pi[(byte ^ 0x20) as usize];
        }
        p
    }
}


/// Incrementally-maintained state of one bucket during assignment.
#[derive(Clone)]
struct BucketState {
    n: usize,
    k: usize,
    lo_set: [u16; MAX_K],
    hi_set: [u16; MAX_K],
    prob: [f64; MAX_K],
}

impl BucketState {
    fn new(k: usize) -> BucketState {
        BucketState { n: 0, k, lo_set: [0; MAX_K], hi_set: [0; MAX_K], prob: [0.0; MAX_K] }
    }

    fn candidate_prob(&self) -> f64 {
        self.prob[..self.k].iter().product()
    }

    /// Expected verification cost contributed by this bucket, per byte.
    fn cost(&self) -> f64 {
        if self.n == 0 {
            return 0.0;
        }
        (CANDIDATE_OVERHEAD + self.n as f64) * self.candidate_prob()
    }

    fn cost_with(&self, classes: &[Class], model: &Model) -> f64 {
        let mut prod = 1.0;
        for j in 0..self.k {
            prod *= model.closure_prob(self.lo_set[j] | classes[j].lo, self.hi_set[j] | classes[j].hi);
        }
        (CANDIDATE_OVERHEAD + (self.n + 1) as f64) * prod
    }

    fn add(&mut self, classes: &[Class], model: &Model) {
        for j in 0..self.k {
            self.lo_set[j] |= classes[j].lo;
            self.hi_set[j] |= classes[j].hi;
            self.prob[j] = model.closure_prob(self.lo_set[j], self.hi_set[j]);
        }
        self.n += 1;
    }

    fn rebuild(k: usize, members: &[u32], classes: &[Vec<Class>], model: &Model) -> BucketState {
        let mut s = BucketState::new(k);
        for &id in members {
            s.add(&classes[id as usize], model);
        }
        s
    }
}

struct Assignment {
    /// buckets[b] holds *local* pattern indices (0..n within the cohort).
    buckets: Vec<Vec<u32>>,
    states: Vec<BucketState>,
    verif_cost: f64,
}

fn greedy_assign(classes: &[Vec<Class>], k: usize, nb: usize, model: &Model) -> Assignment {
    let n = classes.len();
    // Process common (high solo-probability) patterns first so they claim
    // buckets early; rare patterns fit almost anywhere with little damage.
    let mut order: Vec<usize> = (0..n).collect();
    let solo: Vec<f64> = classes
        .iter()
        .map(|cs| cs.iter().map(|c| model.closure_prob(c.lo, c.hi)).product::<f64>())
        .collect();
    order.sort_by(|&a, &b| solo[b].partial_cmp(&solo[a]).unwrap());

    let mut states: Vec<BucketState> = (0..nb).map(|_| BucketState::new(k)).collect();
    let mut buckets: Vec<Vec<u32>> = vec![Vec::new(); nb];
    for &i in &order {
        let mut best = 0usize;
        let mut best_delta = f64::INFINITY;
        for b in 0..nb {
            let delta = states[b].cost_with(&classes[i], model) - states[b].cost();
            if delta < best_delta {
                best_delta = delta;
                best = b;
            }
        }
        states[best].add(&classes[i], model);
        buckets[best].push(i as u32);
    }
    let verif_cost = states.iter().map(|s| s.cost()).sum();
    Assignment { buckets, states, verif_cost }
}

/// Local-search refinement: keep moving single patterns to strictly better
/// buckets until a fixpoint (or a safety cap). Each accepted move strictly
/// decreases the cost, so this terminates (DESIGN.md, Thm 3).
fn refine(a: &mut Assignment, classes: &[Vec<Class>], k: usize, model: &Model) {
    const MAX_PASSES: usize = 32;
    const MIN_GAIN: f64 = 1e-15;
    let nb = a.buckets.len();
    for _ in 0..MAX_PASSES {
        let mut improved = false;
        for b0 in 0..nb {
            let mut idx = 0;
            while idx < a.buckets[b0].len() {
                let id = a.buckets[b0][idx];
                let members_without: Vec<u32> =
                    a.buckets[b0].iter().copied().filter(|&x| x != id).collect();
                let state_without = BucketState::rebuild(k, &members_without, classes, model);
                let removal_gain = a.states[b0].cost() - state_without.cost();
                let mut best_b = b0;
                let mut best_gain = MIN_GAIN;
                for b in 0..nb {
                    if b == b0 {
                        continue;
                    }
                    let add_cost =
                        a.states[b].cost_with(&classes[id as usize], model) - a.states[b].cost();
                    let gain = removal_gain - add_cost;
                    if gain > best_gain {
                        best_gain = gain;
                        best_b = b;
                    }
                }
                if best_b != b0 {
                    a.buckets[b0].retain(|&x| x != id);
                    a.states[b0] = state_without;
                    a.states[best_b].add(&classes[id as usize], model);
                    a.buckets[best_b].push(id);
                    a.verif_cost = a.states.iter().map(|s| s.cost()).sum();
                    improved = true;
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
    pattern_classes: &dyn Fn(usize, usize) -> Class,
    n: usize,
    w: usize,
    max_k: usize,
    model: &Model,
    exhaustive: bool,
) -> Vec<Vec<u8>> {
    let kmax = max_k.min(w).min(MAX_K);
    let pool: Vec<u8> = if exhaustive {
        (0..w as u8).collect()
    } else {
        // Rank positions by expected rarity of the classes they expose
        // (lower total probability = more discriminating), keep the top 9.
        // Positions where some pattern is a wildcard score badly and sink.
        let mut scored: Vec<(f64, u8)> = (0..w)
            .map(|s| {
                let tot: f64 = (0..n)
                    .map(|i| {
                        let c = pattern_classes(i, s);
                        model.closure_prob(c.lo, c.hi)
                    })
                    .sum();
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
    // final choice is never worse than the classic one under the model.
    let prefix: Vec<u8> = (0..kmax as u8).collect();
    if !sets.contains(&prefix) {
        sets.push(prefix);
    }
    sets
}

/// Total model cost of a refined assignment: verification + scan terms.
fn total_cost(verif: f64, k: usize, planes: usize) -> f64 {
    verif + SCAN_COST_PER_TERM * (k * planes) as f64
}

/// Re-score an assignment's verification cost *empirically*: run the exact
/// candidate filter (nibble closure included) over the corpus sample and
/// count per-bucket candidate bits. Unlike any closed-form byte model, this
/// captures arbitrary correlation in the data — e.g. that "G", "/" and "v"
/// of "GET /api/v1/" always co-occur at fixed offsets, which an i.i.d. (or
/// Markov) model underprices by orders of magnitude. Cost: O(|corpus| * k).
fn empirical_verif_cost(a: &Assignment, positions: &[u8], corpus: &[u8]) -> f64 {
    let k = positions.len();
    let nb = a.buckets.len();
    let s_last = *positions.last().unwrap() as usize;
    if corpus.len() <= s_last {
        return a.verif_cost; // sample too small to measure: keep the model
    }
    let planes = nb / PLANE_BUCKETS;
    // Materialize the same byte -> bucket-bitmap tables the engines use.
    let mut tbl = vec![vec![[0u8; 256]; k]; planes];
    for (b, st) in a.states.iter().enumerate() {
        if a.buckets[b].is_empty() {
            continue;
        }
        let (plane, bit) = (b / PLANE_BUCKETS, b % PLANE_BUCKETS);
        for j in 0..k {
            for x in 0..256usize {
                if st.lo_set[j] & (1 << (x & 15)) != 0 && st.hi_set[j] & (1 << (x >> 4)) != 0 {
                    tbl[plane][j][x] |= 1 << bit;
                }
            }
        }
    }
    let mut cnt = vec![0u64; nb];
    let mut d = [0usize; MAX_K];
    for j in 0..k {
        d[j] = s_last - positions[j] as usize;
    }
    for t in s_last..corpus.len() {
        for plane in 0..planes {
            let mut m = 0xFFu8;
            for j in 0..k {
                m &= tbl[plane][j][corpus[t - d[j]] as usize];
                if m == 0 {
                    break;
                }
            }
            while m != 0 {
                cnt[plane * PLANE_BUCKETS + m.trailing_zeros() as usize] += 1;
                m &= m - 1;
            }
        }
    }
    let anchors = (corpus.len() - s_last) as f64;
    (0..nb)
        .map(|b| (CANDIDATE_OVERHEAD + a.buckets[b].len() as f64) * cnt[b] as f64 / anchors)
        .sum()
}

/// Pick the guard probe for one pattern: the model-rarest byte at an offset
/// not already covered by the sampled positions (and not a wildcard).
fn choose_guard(p: &[u8], positions: &[u8], opts: &MatchOpts, model: &Model) -> (u32, u8) {
    let sampled: Vec<bool> = {
        let mut v = vec![false; p.len()];
        for &s in positions {
            if (s as usize) < p.len() {
                v[s as usize] = true;
            }
        }
        v
    };
    let mut best: Option<(f64, usize)> = None;
    for (off, &b) in p.iter().enumerate() {
        if sampled[off] || opts.wildcard == Some(b) {
            continue;
        }
        let prob = model.match_prob(b, opts);
        if best.map_or(true, |(bp, _)| prob < bp) {
            best = Some((prob, off));
        }
    }
    match best {
        Some((_, off)) => (off as u32, p[off]),
        // Every byte is sampled or wildcard: guard on byte 0 (a wildcard
        // guard byte always passes byte_matches, so this stays correct).
        None => (0, p[0]),
    }
}

/// Compile one cohort: the given pattern ids share this filter.
fn compile_cohort(
    ids: &[u32],
    patterns: &[Box<[u8]>],
    opts: &BuildOpts,
    model: &Model,
    corpus: &[u8],
) -> Result<Compiled, BuildError> {
    let n = ids.len();
    assert!(n > 0);
    let max_k = opts.max_k.clamp(1, MAX_K);
    let min_len = ids.iter().map(|&i| patterns[i as usize].len()).min().unwrap();
    let w = min_len.min(MAX_WINDOW);
    let class_at = |i: usize, s: usize| class_of(patterns[ids[i] as usize][s], &opts.match_opts);

    let position_sets: Vec<Vec<u8>> = match opts.forced_positions {
        Some(ps) => {
            let mut v: Vec<u8> = ps.to_vec();
            v.sort_unstable();
            v.dedup();
            if v.is_empty() || v.len() > MAX_K || v.iter().any(|&s| (s as usize) >= w) {
                return Err(BuildError::BadPositions);
            }
            vec![v]
        }
        None => candidate_position_sets(&class_at, n, w, max_k, model, opts.exhaustive),
    };

    // Stage 1: score every candidate set with a single-plane greedy
    // assignment; keep the best few plus the contiguous-prefix baseline.
    let classes_for = |pos: &[u8]| -> Vec<Vec<Class>> {
        (0..n).map(|i| pos.iter().map(|&s| class_at(i, s as usize)).collect()).collect()
    };
    let mut scored: Vec<(f64, Vec<u8>)> = position_sets
        .iter()
        .map(|pos| {
            let a = greedy_assign(&classes_for(pos), pos.len(), PLANE_BUCKETS, model);
            (total_cost(a.verif_cost, pos.len(), 1), pos.clone())
        })
        .collect();
    scored.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    let kmax = max_k.min(w).min(MAX_K);
    let prefix: Vec<u8> = (0..kmax as u8).collect();
    // Finalists: the best few overall, plus the best set of every size
    // (the i.i.d. model systematically underprices correlated positions,
    // so high-k sets must reach the empirical referee even when a cheap
    // low-k set looks better under independence), plus the prefix baseline.
    let mut finalists: Vec<Vec<u8>> = scored.iter().take(8).map(|(_, p)| p.clone()).collect();
    for size in 1..=kmax {
        if let Some((_, p)) = scored.iter().find(|(_, p)| p.len() == size) {
            if !finalists.contains(p) {
                finalists.push(p.clone());
            }
        }
    }
    if position_sets.contains(&prefix) && !finalists.contains(&prefix) {
        finalists.push(prefix);
    }

    // Stage 2: for each finalist and each plane count, refine and score
    // under the selection model (empirical corpus scan if enabled, i.i.d.
    // closed form otherwise).
    let plane_options: &[usize] = if n >= 32 { &[1, 2] } else { &[1] };
    let mut best: Option<(f64, Vec<u8>, usize, Assignment)> = None;
    for pos in &finalists {
        let classes = classes_for(pos);
        for &planes in plane_options {
            let mut a = greedy_assign(&classes, pos.len(), planes * PLANE_BUCKETS, model);
            refine(&mut a, &classes, pos.len(), model);
            let verif = if opts.corpus_score {
                empirical_verif_cost(&a, pos, corpus)
            } else {
                a.verif_cost
            };
            let score = total_cost(verif, pos.len(), planes);
            if best.as_ref().map_or(true, |(s, ..)| score < *s) {
                best = Some((score, pos.clone(), planes, a));
            }
        }
    }
    let (expected_cost, positions, planes, assignment) = best.unwrap();
    let k = positions.len();
    let classes = classes_for(&positions);

    // Materialize tables and buckets.
    let s_last = *positions.last().unwrap() as usize;
    let mut d = [0usize; MAX_K];
    for j in 0..k {
        d[j] = s_last - positions[j] as usize;
    }
    let mut tl = [[[0u8; 16]; MAX_K]; MAX_PLANES];
    let mut th = [[[0u8; 16]; MAX_K]; MAX_PLANES];
    let mut buckets: Vec<Vec<Entry>> = vec![Vec::new(); planes * PLANE_BUCKETS];
    for (b, members) in assignment.buckets.iter().enumerate() {
        let (plane, bit) = (b / PLANE_BUCKETS, b % PLANE_BUCKETS);
        for &local in members {
            let gid = ids[local as usize];
            for j in 0..k {
                let c = classes[local as usize][j];
                for nib in 0..16 {
                    if c.lo & (1 << nib) != 0 {
                        tl[plane][j][nib] |= 1 << bit;
                    }
                    if c.hi & (1 << nib) != 0 {
                        th[plane][j][nib] |= 1 << bit;
                    }
                }
            }
            let (guard_off, guard_byte) =
                choose_guard(&patterns[gid as usize], &positions, &opts.match_opts, model);
            buckets[b].push(Entry { id: gid, guard_off, guard_byte });
        }
    }
    let mut byte_tbl: Vec<Vec<[u8; 256]>> = vec![vec![[0u8; 256]; k]; planes];
    for plane in 0..planes {
        for j in 0..k {
            for x in 0..256usize {
                byte_tbl[plane][j][x] = tl[plane][j][x & 15] & th[plane][j][x >> 4];
            }
        }
    }
    let expected_candidates: f64 = assignment.states.iter().map(|s| {
        if s.n == 0 { 0.0 } else { s.candidate_prob() }
    }).sum();

    Ok(Compiled {
        k,
        planes,
        positions,
        d,
        s_last,
        tl,
        th,
        byte_tbl,
        buckets,
        min_len,
        expected_cost,
        expected_candidates,
    })
}

/// Top-level build: validate, fit models, and decide between a single
/// filter and a partition into length cohorts by comparing total model
/// cost (each cohort scans the haystack once, and its scan term is part of
/// its cost, so the comparison already charges for the extra passes).
pub(crate) fn build(
    patterns: &[Box<[u8]>],
    opts: &BuildOpts,
) -> Result<Vec<Compiled>, BuildError> {
    if patterns.is_empty() {
        return Err(BuildError::NoPatterns);
    }
    if patterns.iter().any(|p| p.is_empty()) {
        return Err(BuildError::EmptyPattern);
    }
    let corpus = opts.corpus.unwrap_or(DEFAULT_CORPUS);
    let model = Model::from_corpus(corpus);
    let all_ids: Vec<u32> = (0..patterns.len() as u32).collect();
    let single = compile_cohort(&all_ids, patterns, opts, &model, corpus)?;

    // Cohort splitting: only when lengths are diverse enough to matter and
    // the caller didn't pin positions (whose validity is window-relative).
    if opts.forced_positions.is_some() || patterns.len() < 2 * MIN_COHORT {
        return Ok(vec![single]);
    }
    const BANDS: &[(usize, usize)] = &[(1, 4), (4, 8), (8, 16), (16, 32), (32, usize::MAX)];
    let mut bands: Vec<Vec<u32>> = vec![Vec::new(); BANDS.len()];
    for (i, p) in patterns.iter().enumerate() {
        let band = BANDS.iter().position(|&(lo, hi)| p.len() >= lo && p.len() < hi).unwrap();
        bands[band].push(i as u32);
    }
    let mut cohorts: Vec<Vec<u32>> = Vec::new();
    let mut acc: Vec<u32> = Vec::new();
    for band in bands.into_iter().filter(|b| !b.is_empty()) {
        acc.extend(band);
        if acc.len() >= MIN_COHORT {
            cohorts.push(std::mem::take(&mut acc));
        }
    }
    if !acc.is_empty() {
        match cohorts.last_mut() {
            Some(last) => last.extend(acc),
            None => cohorts.push(acc),
        }
    }
    if cohorts.len() < 2 {
        return Ok(vec![single]);
    }
    let mut split: Vec<Compiled> = Vec::with_capacity(cohorts.len());
    for c in &cohorts {
        split.push(compile_cohort(c, patterns, opts, &model, corpus)?);
    }
    let split_cost: f64 = split.iter().map(|c| c.expected_cost).sum();
    if split_cost < single.expected_cost {
        Ok(split)
    } else {
        Ok(vec![single])
    }
}
