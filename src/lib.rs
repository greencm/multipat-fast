//! # SPARROW — Sparse Position Adaptive Rejection Over Rolling Windows
//!
//! A SIMD multi-pattern matcher built around a new prefilter design:
//! instead of inspecting a *fixed, contiguous* run of pattern bytes (the
//! prefix in Teddy, the whole literal in Harry, the tail domain in
//! Hyperscan's FDR), SPARROW samples a **sparse, optimizer-chosen set of
//! positions** inside the patterns' common anchor window, and assigns
//! patterns to its SIMD buckets by **minimizing an exact expected-cost
//! objective** under a byte-distribution model that also models the PSHUFB
//! nibble cross-product closure exactly. The same objective arbitrates how
//! many positions to sample, whether to run 8 or 16 buckets, and whether to
//! split the pattern set into length cohorts.
//!
//! Guarantees (proved in `docs/DESIGN.md`):
//! * **No false negatives** — every occurrence of every pattern is reported.
//! * The verification term of the objective equals the **exact** expected
//!   verification cost per haystack byte under the i.i.d. byte model,
//!   nibble closure included (finalists are re-scored by an exact,
//!   correlation-aware empirical scan of the corpus sample by default).
//! * The bucket refinement terminates and returns a move-optimal partition.
//! * The chosen configuration is never worse, under the selection model,
//!   than the Teddy-style contiguous-prefix configuration.
//!
//! ```
//! use sparrow::Sparrow;
//! let m = Sparrow::new(["raven", "sparrow", "swallow"]).unwrap();
//! let hits = m.find_all(b"a sparrow and a swallow sat on a wire");
//! assert_eq!(hits.len(), 2);
//! assert_eq!(&b"a sparrow and a swallow sat on a wire"[hits[0].start..hits[0].end], b"sparrow");
//! ```

mod builder;
mod scalar;
#[cfg(target_arch = "x86_64")]
mod avx2;
#[cfg(target_arch = "x86_64")]
mod avx512;
#[cfg(target_arch = "aarch64")]
mod neon;
pub mod naive;
pub mod dense;

use dense::DenseLane;

use builder::{Compiled, Entry};

/// A single pattern occurrence. All overlapping occurrences of all patterns
/// are reported, sorted by `(start, end, pattern)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Match {
    /// Byte offset of the first matched byte.
    pub start: usize,
    /// One past the last matched byte (`start + pattern length`).
    pub end: usize,
    /// Index of the pattern (in the order given at build time).
    pub pattern: usize,
}

/// A pattern occurrence reported by [`StreamScanner`], with offsets global
/// to the whole stream.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct StreamMatch {
    pub start: u64,
    pub end: u64,
    pub pattern: usize,
}

/// Errors from building a [`Sparrow`] matcher.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BuildError {
    /// The pattern set was empty.
    NoPatterns,
    /// A pattern was the empty string.
    EmptyPattern,
    /// Explicitly forced sampled positions were empty, more than 4, or not
    /// strictly inside the common anchor window `[0, min(min_len, 32))`.
    BadPositions,
    /// A forced engine is not supported by the current CPU.
    EngineUnavailable,
}

impl core::fmt::Display for BuildError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            BuildError::NoPatterns => write!(f, "pattern set is empty"),
            BuildError::EmptyPattern => write!(f, "patterns must be non-empty"),
            BuildError::BadPositions => {
                write!(f, "forced positions must be 1..=4 offsets inside [0, min(min_len, 32))")
            }
            BuildError::EngineUnavailable => {
                write!(f, "the forced engine is not supported on this CPU")
            }
        }
    }
}

impl std::error::Error for BuildError {}

/// Which scan kernel to use.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Engine {
    Scalar,
    Avx2,
    Avx512,
    Neon,
}

/// Matching semantics shared by the filter compiler and the verifier.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct MatchOpts {
    pub case_insensitive: bool,
    pub wildcard: Option<u8>,
}

/// Configures and builds a [`Sparrow`] matcher.
#[derive(Debug, Clone)]
pub struct Builder {
    corpus: Option<Vec<u8>>,
    max_positions: Option<usize>,
    forced_positions: Option<Vec<u8>>,
    exhaustive: bool,
    corpus_score: bool,
    match_opts: MatchOpts,
    forced_engine: Option<Engine>,
    dense_lane: bool,
    force_dense: bool,
}

impl Default for Builder {
    fn default() -> Builder {
        Builder::new()
    }
}

impl Builder {
    pub fn new() -> Builder {
        Builder {
            corpus: None,
            max_positions: None,
            forced_positions: None,
            exhaustive: cfg!(feature = "exhaustive-search"),
            corpus_score: true,
            match_opts: MatchOpts::default(),
            forced_engine: None,
            dense_lane: true,
            force_dense: false,
        }
    }

    /// Provide a sample of the traffic the matcher will scan. The optimizer
    /// estimates its byte distribution from it (with smoothing) and scores
    /// finalist filter configurations by scanning it; the closer the sample
    /// is to the real
    /// workload, the closer the realized candidate rate is to the
    /// model-optimal one. Without a sample, a built-in mixed
    /// text/markup/code model is used.
    pub fn corpus_sample(mut self, sample: &[u8]) -> Builder {
        self.corpus = Some(sample.to_vec());
        self
    }

    /// Cap the number of sampled positions (1..=4). The optimizer may use
    /// fewer than the cap when the model says an extra filter term costs
    /// more scan time than the verification it saves.
    pub fn max_positions(mut self, k: usize) -> Builder {
        self.max_positions = Some(k);
        self
    }

    /// Force the sampled positions (offsets from each pattern's start),
    /// bypassing position optimization and cohort splitting. Bucket
    /// assignment is still optimized. Useful for ablation:
    /// `positions(&[0, 1, 2, 3])` gives a Teddy-style contiguous-prefix
    /// filter inside the same runtime.
    pub fn positions(mut self, positions: &[u8]) -> Builder {
        self.forced_positions = Some(positions.to_vec());
        self
    }

    /// Search every position subset of the anchor window instead of the
    /// pruned candidate pool. Exponentially more configurations; identical
    /// guarantees, occasionally a slightly better model cost.
    pub fn exhaustive_search(mut self, on: bool) -> Builder {
        self.exhaustive = on;
        self
    }

    /// Re-score finalist configurations by running the exact candidate
    /// filter over the corpus sample and counting real candidates (default:
    /// on). This captures byte correlations that no closed-form model can —
    /// shared prefixes, protocol framing, repeated structure. Turning it
    /// off selects purely by the closed-form i.i.d. objective.
    pub fn corpus_scoring(mut self, on: bool) -> Builder {
        self.corpus_score = on;
        self
    }

    /// Match ASCII letters case-insensitively. The filter widens each
    /// sampled class with the opposite-case nibbles (the cost model prices
    /// the widening), and verification compares case-insensitively.
    pub fn ascii_case_insensitive(mut self, on: bool) -> Builder {
        self.match_opts.case_insensitive = on;
        self
    }

    /// Treat this byte in *patterns* as a single-byte wildcard (`?`-glob).
    /// A wildcard position matches any haystack byte; the position
    /// optimizer naturally avoids sampling wildcard-heavy positions because
    /// their class probability is 1.
    pub fn wildcard_byte(mut self, byte: Option<u8>) -> Builder {
        self.match_opts.wildcard = byte;
        self
    }

    /// Force a specific scan kernel (mainly for tests and benchmarks).
    /// Errors at build time if the CPU lacks the required features.
    /// Enable the two-prong build (default on): patterns that no
    /// sampled-position filter can handle cheaply (short, common) are routed
    /// to a bit-parallel Shift-Or lane, the rest to sparse cohorts, when the
    /// model says the split is cheaper. Off = pure sampled-position matcher.
    pub fn dense_lane(mut self, on: bool) -> Builder {
        self.dense_lane = on;
        self
    }

    /// Route every pattern to the dense lane if it fits (testing / probing).
    #[doc(hidden)]
    pub fn force_dense(mut self, on: bool) -> Builder {
        self.force_dense = on;
        self
    }

    pub fn force_engine(mut self, engine: Engine) -> Builder {
        self.forced_engine = Some(engine);
        self
    }

    pub fn build<I, P>(self, patterns: I) -> Result<Sparrow, BuildError>
    where
        I: IntoIterator<Item = P>,
        P: AsRef<[u8]>,
    {
        let pats: Vec<Box<[u8]>> =
            patterns.into_iter().map(|p| p.as_ref().to_vec().into_boxed_slice()).collect();
        let opts = builder::BuildOpts {
            corpus: self.corpus.as_deref(),
            max_k: self.max_positions.unwrap_or(builder::MAX_K),
            forced_positions: self.forced_positions.as_deref(),
            exhaustive: self.exhaustive,
            corpus_score: self.corpus_score,
            match_opts: self.match_opts,
        };
        let (cohorts, dense) = builder::build_routed(&pats, &opts, self.dense_lane, self.force_dense)?;
        let engine = match self.forced_engine {
            Some(e) => {
                if !engine_available(e) {
                    return Err(BuildError::EngineUnavailable);
                }
                e
            }
            None => detect_engine(),
        };
        let max_len = pats.iter().map(|p| p.len()).max().unwrap();
        Ok(Sparrow { patterns: pats, cohorts, dense, engine, match_opts: self.match_opts, max_len })
    }
}

fn engine_available(e: Engine) -> bool {
    match e {
        Engine::Scalar => true,
        #[cfg(target_arch = "x86_64")]
        Engine::Avx2 => std::arch::is_x86_feature_detected!("avx2"),
        #[cfg(target_arch = "x86_64")]
        Engine::Avx512 => std::arch::is_x86_feature_detected!("avx512bw"),
        // NEON is baseline on aarch64.
        #[cfg(target_arch = "aarch64")]
        Engine::Neon => true,
        #[allow(unreachable_patterns)]
        _ => false,
    }
}

fn detect_engine() -> Engine {
    if engine_available(Engine::Avx512) {
        Engine::Avx512
    } else if engine_available(Engine::Avx2) {
        Engine::Avx2
    } else if engine_available(Engine::Neon) {
        Engine::Neon
    } else {
        Engine::Scalar
    }
}

/// A compiled multi-pattern matcher. Build once, search many haystacks.
pub struct Sparrow {
    patterns: Vec<Box<[u8]>>,
    cohorts: Vec<Compiled>,
    dense: Option<DenseLane>,
    engine: Engine,
    match_opts: MatchOpts,
    max_len: usize,
}

/// Everything a kernel needs to scan and verify for one cohort.
pub(crate) struct ScanCtx<'a> {
    pub c: &'a Compiled,
    pub patterns: &'a [Box<[u8]>],
    pub opts: MatchOpts,
}

impl Sparrow {
    /// Build with default settings. See [`Builder`] for knobs.
    pub fn new<I, P>(patterns: I) -> Result<Sparrow, BuildError>
    where
        I: IntoIterator<Item = P>,
        P: AsRef<[u8]>,
    {
        Builder::new().build(patterns)
    }

    pub fn builder() -> Builder {
        Builder::new()
    }

    /// Find all (overlapping) occurrences of all patterns, sorted by
    /// `(start, end, pattern)`.
    pub fn find_all(&self, haystack: &[u8]) -> Vec<Match> {
        let mut out = self.find_all_unsorted(haystack);
        // Every engine emits a sorted run, so the concatenation is a handful
        // of sorted runs; the stable sort detects and merges them in ~O(n)
        // (sort_unstable would pay a full n log n on match-dense inputs).
        out.sort();
        out
    }

    /// Find all (overlapping) occurrences of all patterns, in the order the
    /// engines produce them: each cohort and each dense-lane segment emits
    /// in increasing position, but the runs are concatenated, not merged.
    /// Use this when order doesn't matter — on match-dense inputs the sort
    /// in [`find_all`](Self::find_all) can cost as much as the scan.
    pub fn find_all_unsorted(&self, haystack: &[u8]) -> Vec<Match> {
        let mut out = Vec::new();
        for c in &self.cohorts {
            let ctx = ScanCtx { c, patterns: &self.patterns, opts: self.match_opts };
            self.scan_cohort(&ctx, haystack, &mut out);
        }
        if let Some(d) = &self.dense {
            d.find_all(haystack, &mut out);
        }
        out
    }

    fn scan_cohort(&self, ctx: &ScanCtx<'_>, hay: &[u8], out: &mut Vec<Match>) {
        #[cfg(target_arch = "x86_64")]
        match self.engine {
            Engine::Avx512 if hay.len() >= 96 => {
                // SAFETY: gated on runtime AVX-512BW detection at build.
                unsafe { avx512::find_all(ctx, hay, out) };
                return;
            }
            Engine::Avx2 if hay.len() >= 64 => {
                // SAFETY: gated on runtime AVX2 detection at build.
                unsafe { avx2::find_all(ctx, hay, out) };
                return;
            }
            _ => {}
        }
        #[cfg(target_arch = "aarch64")]
        if self.engine == Engine::Neon && hay.len() >= 48 {
            // SAFETY: NEON is baseline on aarch64.
            unsafe { neon::find_all(ctx, hay, out) };
            return;
        }
        scalar::find_in_range(ctx, hay, 0, hay.len(), out);
    }

    /// Leftmost, non-overlapping match semantics (like a lazy scan-and-skip
    /// or aho-corasick's leftmost-first): at each leftmost match start the
    /// earliest-built pattern wins, and scanning resumes past its end.
    pub fn find_leftmost_nonoverlapping(&self, haystack: &[u8]) -> Vec<Match> {
        let all = self.find_all(haystack);
        let mut out = Vec::new();
        let mut next_start = 0usize;
        let mut i = 0;
        while i < all.len() {
            if all[i].start < next_start {
                i += 1;
                continue;
            }
            // Among all matches at this leftmost start, the earliest-built
            // pattern wins (aho-corasick's leftmost-first tie-break).
            let s = all[i].start;
            let mut best = all[i];
            while i < all.len() && all[i].start == s {
                if all[i].pattern < best.pattern {
                    best = all[i];
                }
                i += 1;
            }
            next_start = best.end;
            out.push(best);
        }
        out
    }

    /// Incremental scanning over a stream of chunks. Matches that span
    /// chunk boundaries are found; offsets are global to the stream.
    pub fn stream(&self) -> StreamScanner<'_> {
        StreamScanner { m: self, tail: Vec::new(), consumed: 0 }
    }

    /// Force the portable scalar engine (testing / non-x86 reference).
    #[doc(hidden)]
    pub fn find_all_scalar(&self, haystack: &[u8]) -> Vec<Match> {
        let mut out = Vec::new();
        for c in &self.cohorts {
            let ctx = ScanCtx { c, patterns: &self.patterns, opts: self.match_opts };
            scalar::find_in_range(&ctx, haystack, 0, haystack.len(), &mut out);
        }
        if let Some(d) = &self.dense {
            d.find_all(haystack, &mut out);
        }
        out.sort_unstable();
        out
    }

    /// The engine selected at build time.
    pub fn engine(&self) -> Engine {
        self.engine
    }

    /// The dense (Shift-Or) lane, if the router gave it any patterns.
    pub fn dense_lane(&self) -> Option<&DenseLane> {
        self.dense.as_ref()
    }

    /// Number of compiled length cohorts (1 unless the cost model chose to
    /// split the pattern set by length; 0 if every pattern went to the
    /// dense lane).
    pub fn cohort_count(&self) -> usize {
        self.cohorts.len()
    }

    /// Sampled positions per cohort (offsets from pattern start, ascending).
    pub fn sampled_positions(&self) -> Vec<Vec<u8>> {
        self.cohorts.iter().map(|c| c.positions.clone()).collect()
    }

    /// Buckets per cohort (8 or 16 = one or two SIMD planes).
    pub fn bucket_counts(&self) -> Vec<usize> {
        self.cohorts.iter().map(|c| c.buckets.len()).collect()
    }

    /// Total model cost per haystack byte — the objective the optimizer
    /// minimized (expected pattern comparisons + scan terms), summed over
    /// cohorts.
    pub fn expected_cost(&self) -> f64 {
        self.cohorts.iter().map(|c| c.expected_cost).sum::<f64>()
            + self.dense.as_ref().map_or(0.0, |d| d.expected_cost())
    }

    /// Model-expected candidate bits per haystack byte under the i.i.d.
    /// model, summed over cohorts.
    pub fn expected_candidate_rate(&self) -> f64 {
        self.cohorts.iter().map(|c| c.expected_candidates).sum()
    }

    pub fn pattern_count(&self) -> usize {
        self.patterns.len()
    }

    /// Length of the shortest pattern.
    pub fn min_pattern_len(&self) -> usize {
        self.cohorts
            .iter()
            .map(|c| c.min_len)
            .chain(self.dense.as_ref().map(|d| d.min_len()))
            .min()
            .unwrap()
    }

    /// Heap/working-set footprint of the compiled matcher, by component.
    pub fn memory_usage(&self) -> MemoryUsage {
        let mut u = MemoryUsage::default();
        for c in &self.cohorts {
            // Nibble shuffle tables actually loaded by the SIMD kernels
            // (the [MAX_PLANES][MAX_K] arrays are statically sized; count
            // the entries the configuration uses).
            u.filter_tables += 2 * c.planes * c.k * 16;
            u.scalar_tables += c.byte_tbl.iter().map(|p| p.len() * 256).sum::<usize>();
            u.buckets += c
                .buckets
                .iter()
                .map(|b| b.len() * std::mem::size_of::<Entry>())
                .sum::<usize>();
        }
        if let Some(d) = &self.dense {
            u.scalar_tables += d.table_bytes();
        }
        u.patterns = self
            .patterns
            .iter()
            .map(|p| p.len() + std::mem::size_of::<Box<[u8]>>())
            .sum();
        u.total = u.filter_tables + u.scalar_tables + u.buckets + u.patterns;
        u
    }
}

/// Compiled-matcher footprint breakdown, in bytes. `filter_tables` is the
/// SIMD-hot state (register-resident during scans); `scalar_tables` backs
/// the prelude/tail path; `buckets` + `patterns` are touched only on
/// candidates/matches.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct MemoryUsage {
    pub filter_tables: usize,
    pub scalar_tables: usize,
    pub buckets: usize,
    pub patterns: usize,
    pub total: usize,
}

/// Incremental scanner returned by [`Sparrow::stream`]. Feed chunks with
/// [`push`](StreamScanner::push); each call returns the matches whose end
/// falls in the new chunk (so every match is reported exactly once, and
/// boundary-spanning matches are found). Keeps `max_pattern_len - 1` bytes
/// of history.
pub struct StreamScanner<'a> {
    m: &'a Sparrow,
    tail: Vec<u8>,
    /// Total stream bytes consumed before the current chunk.
    consumed: u64,
}

impl StreamScanner<'_> {
    pub fn push(&mut self, chunk: &[u8]) -> Vec<StreamMatch> {
        let tail_len = self.tail.len();
        let scan_owned;
        let scan: &[u8] = if tail_len == 0 {
            chunk
        } else {
            let mut b = std::mem::take(&mut self.tail);
            b.extend_from_slice(chunk);
            scan_owned = b;
            &scan_owned
        };
        let scan_base = self.consumed - tail_len as u64;
        let out: Vec<StreamMatch> = self
            .m
            .find_all(scan)
            .into_iter()
            .filter(|m| m.end > tail_len)
            .map(|m| StreamMatch {
                start: scan_base + m.start as u64,
                end: scan_base + m.end as u64,
                pattern: m.pattern,
            })
            .collect();
        let keep = (self.m.max_len - 1).min(scan.len());
        let new_tail = scan[scan.len() - keep..].to_vec();
        self.tail = new_tail;
        self.consumed += chunk.len() as u64;
        out
    }
}

/// True iff a haystack byte satisfies one pattern byte under the matcher's
/// semantics (exact, ASCII-case-insensitive, or wildcard).
#[inline(always)]
pub(crate) fn byte_matches(hay_b: u8, pat_b: u8, opts: &MatchOpts) -> bool {
    if opts.wildcard == Some(pat_b) {
        return true;
    }
    if hay_b == pat_b {
        return true;
    }
    opts.case_insensitive && hay_b.eq_ignore_ascii_case(&pat_b)
}

#[inline]
pub(crate) fn pattern_matches(window: &[u8], pat: &[u8], opts: &MatchOpts) -> bool {
    if opts.wildcard.is_none() && !opts.case_insensitive {
        return window == pat;
    }
    window.iter().zip(pat.iter()).all(|(&h, &p)| byte_matches(h, p, opts))
}

/// Exact verification of a candidate anchor for one bucket plane: check
/// every entry of every flagged bucket at the implied start position, guard
/// byte first. Shared by all engines.
#[inline]
pub(crate) fn verify_at(
    ctx: &ScanCtx<'_>,
    hay: &[u8],
    t: usize,
    mut mask: u8,
    plane: usize,
    out: &mut Vec<Match>,
) {
    // Window start implied by the anchor. Bits produced near the beginning
    // of the haystack can imply a start before offset 0; those are filter
    // artifacts, not matches.
    let Some(start) = t.checked_sub(ctx.c.s_last) else { return };
    let base = plane * builder::PLANE_BUCKETS;
    while mask != 0 {
        let b = mask.trailing_zeros() as usize;
        mask &= mask - 1;
        for &Entry { id, guard_off, guard_byte } in &ctx.c.buckets[base + b] {
            let p = &ctx.patterns[id as usize];
            let end = start + p.len();
            if end > hay.len() {
                continue;
            }
            // Guard probe: the model-rarest unsampled pattern byte. Rejects
            // most false candidates with one load instead of a full compare.
            if !byte_matches(hay[start + guard_off as usize], guard_byte, &ctx.opts) {
                continue;
            }
            if pattern_matches(&hay[start..end], p, &ctx.opts) {
                out.push(Match { start, end, pattern: id as usize });
            }
        }
    }
}
