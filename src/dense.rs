//! The dense lane: bit-parallel multi-pattern Shift-Or for short, common
//! patterns where no sampled-position filter can win.
//!
//! SPARROW's filter pays per *candidate*; on match-dense sets (English stop
//! words over English text) candidates arrive every few bytes and a DFA,
//! which pays a flat price per byte, wins. Shift-Or also pays a flat price
//! per byte, but its per-byte critical path is `shift, and, or` on a
//! register (3 cycles) instead of the DFA's dependent table load (~4-5
//! cycles), and
//! its only memory access — `B[byte]` — is independent across bytes, so it
//! pipelines. To hide even that chain, the haystack is scanned as
//! `SEGS` interleaved segments with independent state.
//!
//! Capacity: every pattern byte costs one bit; a lane is 64 bits, and a
//! matcher holds up to `MAX_LANES` lanes. Anything bigger stays with the
//! sampled-position filter (the router decides by model cost).

use crate::{byte_matches, Match, MatchOpts};

/// Bits per lane.
pub const LANE_BITS: usize = 64;
/// Maximum lanes per dense matcher (= max total pattern bytes / 64).
pub const MAX_LANES: usize = 4;
/// Longest pattern the dense lane accepts. Shift-Or handles any length up
/// to the lane width, but long patterns are exactly where sparse sampling
/// shines, so they are never worth a lane.
pub const MAX_PATTERN_LEN: usize = 32;
/// Interleaved haystack segments scanned in lock-step.
const SEGS: usize = 4;
/// Haystack is scanned in chunks of this size (each split into `SEGS`
/// segments) so per-segment output runs stay small and cache-resident.
const CHUNK: usize = 64 * 1024;
/// A lock-step iteration in which some (segment, lane) state had an end
/// bit clear: the step and all eight states (index g*2 + l), recorded by
/// the scan loop and decoded afterwards.
#[derive(Clone, Copy)]
struct Event {
    step: u32,
    st: [u64; 8],
}
/// Events buffered before the scan loop pauses to drain them.
const EVENTS: usize = 256;

/// Model cost of one lane per haystack byte, in the builder's units (one
/// verification unit ≈ 10 ns). Measured on Apple M-series: the 4-segment
/// kernel scans at ≈0.35 ns/byte per lane (instruction-throughput bound:
/// ~3 loads + 4 ALU ops per segment-lane-step).
pub const LANE_COST: f64 = 0.035;
/// Model cost per reported match (event record + decode + push), same
/// units. Measured ≈8 ns. Sampled-position cohorts already pay for true
/// matches through their candidate term; the dense lane charges for them
/// explicitly, at the rate observed on the corpus sample.
pub const MATCH_COST: f64 = 0.8;

struct Lane {
    /// Shift-Or table: bit (off + j) is *clear* for byte `c` iff pattern
    /// byte `j` accepts `c`.
    table: Box<[u64; 256]>,
    /// Bit set at the last position of every pattern.
    ends: u64,
    /// Mask with every pattern's first-position bit *clear* (so the shift
    /// re-activates every start each step; bit 0 is free from the shift).
    keep: u64,
    /// Indexed by end-bit position: (pattern id, pattern length).
    end_info: Vec<(u32, u32)>,
}

pub struct DenseLane {
    lanes: Vec<Lane>,
    max_len: usize,
    min_len: usize,
    n_patterns: usize,
    /// Matches per haystack byte observed on the build corpus sample.
    match_rate: f64,
}

impl DenseLane {
    /// Pack `ids` into lanes (first-fit). `None` if they don't fit.
    pub(crate) fn compile(
        ids: &[u32],
        patterns: &[Box<[u8]>],
        opts: &MatchOpts,
        corpus: &[u8],
    ) -> Option<DenseLane> {
        if ids.is_empty() {
            return None;
        }
        // First-fit by length descending packs tighter.
        let mut order: Vec<u32> = ids.to_vec();
        order.sort_by_key(|&i| std::cmp::Reverse(patterns[i as usize].len()));
        let mut lanes: Vec<(Vec<u32>, usize)> = Vec::new(); // (ids, used bits)
        for &id in &order {
            let len = patterns[id as usize].len();
            if len > MAX_PATTERN_LEN || len > LANE_BITS {
                return None;
            }
            match lanes.iter_mut().find(|(_, used)| used + len <= LANE_BITS) {
                Some((v, used)) => {
                    v.push(id);
                    *used += len;
                }
                None => {
                    if lanes.len() == MAX_LANES {
                        return None;
                    }
                    lanes.push((vec![id], len));
                }
            }
        }
        let mut out = Vec::with_capacity(lanes.len());
        for (members, _) in lanes {
            let mut table = Box::new([u64::MAX; 256]);
            let mut ends = 0u64;
            let mut keep = u64::MAX;
            let mut end_info = vec![(0u32, 0u32); LANE_BITS];
            let mut off = 0usize;
            for &id in &members {
                let p = &patterns[id as usize];
                for (j, &pb) in p.iter().enumerate() {
                    let bit = 1u64 << (off + j);
                    for c in 0..256usize {
                        if byte_matches(c as u8, pb, opts) {
                            table[c] &= !bit;
                        }
                    }
                }
                let end = off + p.len() - 1;
                ends |= 1u64 << end;
                keep &= !(1u64 << off);
                end_info[end] = (id, p.len() as u32);
                off += p.len();
            }
            out.push(Lane { table, ends, keep, end_info });
        }
        let lens = ids.iter().map(|&i| patterns[i as usize].len());
        let mut d = DenseLane {
            lanes: out,
            max_len: lens.clone().max().unwrap(),
            min_len: lens.min().unwrap(),
            n_patterns: ids.len(),
            match_rate: 0.0,
        };
        if !corpus.is_empty() {
            let mut m = Vec::new();
            d.find_all(corpus, &mut m);
            d.match_rate = m.len() as f64 / corpus.len() as f64;
        }
        Some(d)
    }

    pub fn lane_count(&self) -> usize {
        self.lanes.len()
    }
    pub fn pattern_count(&self) -> usize {
        self.n_patterns
    }
    pub fn min_len(&self) -> usize {
        self.min_len
    }
    pub fn max_len(&self) -> usize {
        self.max_len
    }
    /// Model cost per haystack byte: scan plus corpus-observed match work.
    pub fn expected_cost(&self) -> f64 {
        LANE_COST * self.lanes.len() as f64 + MATCH_COST * self.match_rate
    }
    /// Matches per byte observed on the build corpus sample.
    pub fn corpus_match_rate(&self) -> f64 {
        self.match_rate
    }
    /// Bytes of table state touched per byte scanned.
    pub fn table_bytes(&self) -> usize {
        self.lanes.len() * 256 * 8
    }

    pub fn find_all(&self, hay: &[u8], out: &mut Vec<Match>) {
        if hay.len() < self.min_len {
            return;
        }
        // Per-segment run buffers, reused across chunks so they stay hot
        // and never page-fault; `out` grows once.
        let mut runs: [Vec<Match>; SEGS] = std::array::from_fn(|_| Vec::new());
        let mut off = 0usize;
        while off < hay.len() {
            let end = (off + CHUNK).min(hay.len());
            // Lanes are scanned two at a time: 2 lanes x 4 segments is what
            // fits the register file; a further pass re-reads the chunk
            // from L1. Each pass emits into the same per-segment runs, so
            // a run is the concatenation of one sorted sub-run per pass.
            let mut l = 0;
            while l < self.lanes.len() {
                if l + 2 <= self.lanes.len() {
                    self.scan_chunk2(hay, off, end, l, &mut runs);
                    l += 2;
                } else {
                    self.scan_chunk1(hay, off, end, l, &mut runs);
                    l += 1;
                }
            }
            // Each sub-run was emitted in end order; in start order every
            // element is displaced by at most the matches inside one
            // `max_len` window, so an insertion sort is linear. Sub-runs
            // of different passes interleave in position, so with several
            // passes use a real sort on the (small, cache-hot) run.
            let passes = self.lanes.len().div_ceil(2);
            for r in runs.iter_mut() {
                if passes == 1 {
                    insertion_sort(r);
                } else {
                    r.sort_unstable();
                }
                out.append(r);
            }
            off = end;
        }
    }

    /// Drain recorded hit events. Exactly one (segment, lane) pair usually
    /// hit; locate it with a bit scan rather than a branch per pair (which
    /// segment hit is unpredictable, and a mispredict costs as much as
    /// scanning ~15 bytes). Lives outside the scan loop so that loop stays
    /// call-free and its invariants stay in registers.
    #[inline(never)]
    fn drain_events(
        &self,
        ev: &[Event],
        lane0: usize,
        nl: usize,
        pos: &[usize; SEGS],
        report_from: &[usize; SEGS],
        runs: &mut [Vec<Match>; SEGS],
    ) {
        let ends = [self.lanes[lane0].ends, if nl == 2 { self.lanes[lane0 + 1].ends } else { 0 }];
        for e in ev {
            let mut which = 0u32;
            for k in 0..8 {
                which |= (((!e.st[k] & ends[k & 1]) != 0) as u32) << k;
            }
            while which != 0 {
                let k = which.trailing_zeros() as usize;
                which &= which - 1;
                let (g, l) = (k / 2, k % 2);
                let i = pos[g] + e.step as usize;
                if i >= report_from[g] {
                    self.emit(lane0 + l, e.st[k], i, &mut runs[g]);
                }
            }
        }
    }

    /// Report every pattern whose end bit is clear in `s` at haystack
    /// offset `i`.
    #[inline(always)]
    fn emit(&self, lane: usize, s: u64, i: usize, run: &mut Vec<Match>) {
        let ln = &self.lanes[lane];
        let mut hits = !s & ln.ends;
        loop {
            let bit = hits.trailing_zeros() as usize;
            hits &= hits - 1;
            let (id, len) = ln.end_info[bit];
            run.push(Match { start: i + 1 - len as usize, end: i + 1, pattern: id as usize });
            if hits == 0 {
                break;
            }
        }
    }

    /// Segment layout for one chunk `[lo, hi)`: segment g reports matches
    /// ending in [b_g, b_{g+1}) and starts scanning `max_len - 1` bytes
    /// earlier (with all-ones state) so it is exact by the time it reports;
    /// the prefix it needs may lie in the previous chunk.
    #[inline(always)]
    fn segments(&self, lo: usize, hi: usize) -> ([usize; SEGS], [usize; SEGS], [usize; SEGS], usize) {
        let warm = self.max_len - 1;
        let n = hi - lo;
        let mut pos = [0usize; SEGS];
        let mut stop = [0usize; SEGS];
        let mut report_from = [0usize; SEGS];
        for g in 0..SEGS {
            let b0 = lo + n * g / SEGS;
            pos[g] = b0.saturating_sub(warm);
            stop[g] = lo + n * (g + 1) / SEGS;
            report_from[g] = b0;
        }
        let common = (0..SEGS).map(|g| stop[g] - pos[g]).min().unwrap();
        (pos, stop, report_from, common)
    }

    /// Two-lane kernel. Every (segment, lane) state is a named local so it
    /// lives in a register; the only memory reads per step are 4 haystack
    /// bytes and 8 (L1-resident) table rows, none dependent on state.
    #[inline(never)]
    fn scan_chunk2(&self, hay: &[u8], lo: usize, hi: usize, l0: usize, runs: &mut [Vec<Match>; SEGS]) {
        let (pos, stop, report_from, common) = self.segments(lo, hi);
        let (la, lb) = (&self.lanes[l0], &self.lanes[l0 + 1]);
        let (ta, tb): (&[u64; 256], &[u64; 256]) = (&la.table, &lb.table);
        let (ka, kb) = (la.keep, lb.keep);
        let (ea, eb) = (la.ends, lb.ends);
        let base = hay.as_ptr();
        let (p0, p1, p2, p3) = unsafe {
            (base.add(pos[0]), base.add(pos[1]), base.add(pos[2]), base.add(pos[3]))
        };
        let (mut a0, mut a1, mut a2, mut a3) = (u64::MAX, u64::MAX, u64::MAX, u64::MAX);
        let (mut b0, mut b1, mut b2, mut b3) = (u64::MAX, u64::MAX, u64::MAX, u64::MAX);
        let mut ev = [Event { step: 0, st: [0; 8] }; EVENTS];
        let mut step = 0usize;
        while step < common {
            // Call-free inner loop: hits are recorded, not handled, so every
            // invariant stays in a register. Stops to drain when the event
            // buffer fills.
            let mut nev = 0usize;
            while step < common && nev < EVENTS {
                // SAFETY: pos[g] + step < stop[g] <= hay.len() for every g.
                let (c0, c1, c2, c3) = unsafe {
                    (*p0.add(step) as usize, *p1.add(step) as usize, *p2.add(step) as usize, *p3.add(step) as usize)
                };
                a0 = ((a0 << 1) & ka) | ta[c0];
                a1 = ((a1 << 1) & ka) | ta[c1];
                a2 = ((a2 << 1) & ka) | ta[c2];
                a3 = ((a3 << 1) & ka) | ta[c3];
                b0 = ((b0 << 1) & kb) | tb[c0];
                b1 = ((b1 << 1) & kb) | tb[c1];
                b2 = ((b2 << 1) & kb) | tb[c2];
                b3 = ((b3 << 1) & kb) | tb[c3];
                // An end bit is clear in the AND of all segments iff it is
                // clear in some segment.
                let any = (!(a0 & a1 & a2 & a3) & ea) | (!(b0 & b1 & b2 & b3) & eb);
                // Branch-free record: always write the slot, advance only on
                // a hit (an unpredictable branch here costs more than the
                // 72-byte store).
                // SAFETY: nev < EVENTS by the loop condition.
                unsafe {
                    *ev.get_unchecked_mut(nev) =
                        Event { step: step as u32, st: [a0, b0, a1, b1, a2, b2, a3, b3] };
                }
                nev += (any != 0) as usize;
                step += 1;
            }
            if nev != 0 {
                self.drain_events(&ev[..nev], l0, 2, &pos, &report_from, runs);
            }
        }
        // Remainders (segments differ in length by at most a byte or two).
        let mut st = [a0, b0, a1, b1, a2, b2, a3, b3];
        for g in 0..SEGS {
            for i in pos[g] + common..stop[g] {
                let c = hay[i] as usize;
                st[g * 2] = ((st[g * 2] << 1) & ka) | ta[c];
                st[g * 2 + 1] = ((st[g * 2 + 1] << 1) & kb) | tb[c];
                if i >= report_from[g] {
                    if !st[g * 2] & ea != 0 {
                        self.emit(l0, st[g * 2], i, &mut runs[g]);
                    }
                    if !st[g * 2 + 1] & eb != 0 {
                        self.emit(l0 + 1, st[g * 2 + 1], i, &mut runs[g]);
                    }
                }
            }
        }
    }

    /// One-lane kernel (see `scan_chunk2`).
    #[inline(never)]
    fn scan_chunk1(&self, hay: &[u8], lo: usize, hi: usize, l0: usize, runs: &mut [Vec<Match>; SEGS]) {
        let (pos, stop, report_from, common) = self.segments(lo, hi);
        let la = &self.lanes[l0];
        let ta: &[u64; 256] = &la.table;
        let (ka, ea) = (la.keep, la.ends);
        let base = hay.as_ptr();
        let (p0, p1, p2, p3) = unsafe {
            (base.add(pos[0]), base.add(pos[1]), base.add(pos[2]), base.add(pos[3]))
        };
        let (mut a0, mut a1, mut a2, mut a3) = (u64::MAX, u64::MAX, u64::MAX, u64::MAX);
        let mut ev = [Event { step: 0, st: [0; 8] }; EVENTS];
        let mut step = 0usize;
        while step < common {
            let mut nev = 0usize;
            while step < common && nev < EVENTS {
                // SAFETY: pos[g] + step < stop[g] <= hay.len() for every g.
                let (c0, c1, c2, c3) = unsafe {
                    (*p0.add(step) as usize, *p1.add(step) as usize, *p2.add(step) as usize, *p3.add(step) as usize)
                };
                a0 = ((a0 << 1) & ka) | ta[c0];
                a1 = ((a1 << 1) & ka) | ta[c1];
                a2 = ((a2 << 1) & ka) | ta[c2];
                a3 = ((a3 << 1) & ka) | ta[c3];
                let any = !(a0 & a1 & a2 & a3) & ea;
                // SAFETY: nev < EVENTS by the loop condition.
                unsafe {
                    *ev.get_unchecked_mut(nev) = Event {
                        step: step as u32,
                        st: [a0, u64::MAX, a1, u64::MAX, a2, u64::MAX, a3, u64::MAX],
                    };
                }
                nev += (any != 0) as usize;
                step += 1;
            }
            if nev != 0 {
                self.drain_events(&ev[..nev], l0, 1, &pos, &report_from, runs);
            }
        }
        let mut st = [a0, a1, a2, a3];
        for g in 0..SEGS {
            for i in pos[g] + common..stop[g] {
                let c = hay[i] as usize;
                st[g] = ((st[g] << 1) & ka) | ta[c];
                if i >= report_from[g] && !st[g] & ea != 0 {
                    self.emit(l0, st[g], i, &mut runs[g]);
                }
            }
        }
    }
}

/// Insertion sort for nearly-sorted runs (bounded displacement).
fn insertion_sort(v: &mut [Match]) {
    for i in 1..v.len() {
        let m = v[i];
        let mut j = i;
        while j > 0 && v[j - 1] > m {
            v[j] = v[j - 1];
            j -= 1;
        }
        v[j] = m;
    }
}
