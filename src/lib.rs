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
pub mod pattern;
#[cfg(feature = "prefilter")]
pub mod prefilter;

pub use pattern::{ByteSet, Pattern, PatternError};
use pattern::Pat;

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
    /// A class pattern failed to parse (see [`PatternError`]).
    BadPattern(PatternError),
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
            BuildError::BadPattern(e) => write!(f, "bad pattern: {}", e),
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
    timed_referee: bool,
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
            timed_referee: true,
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

    /// Decide dense-vs-sparse routing by *timing* the best model candidates
    /// on the corpus sample (default on). Off = model cost alone, which
    /// makes the routing choice deterministic across runs. Correctness is
    /// identical either way; only throughput can differ.
    pub fn timed_referee(mut self, on: bool) -> Builder {
        self.timed_referee = on;
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

    /// Build from byte-string patterns, under the builder's semantics
    /// (`ascii_case_insensitive`, `wildcard_byte`).
    pub fn build<I, P>(self, patterns: I) -> Result<Sparrow, BuildError>
    where
        I: IntoIterator<Item = P>,
        P: AsRef<[u8]>,
    {
        let o = self.match_opts;
        let pats: Vec<Pattern> = patterns
            .into_iter()
            .map(|p| {
                let mut pat = Pattern::new();
                for &b in p.as_ref() {
                    let set = if o.wildcard == Some(b) {
                        ByteSet::ANY
                    } else if o.case_insensitive {
                        ByteSet::byte(b).ascii_case_fold()
                    } else {
                        ByteSet::byte(b)
                    };
                    pat.push(set);
                }
                pat
            })
            .collect();
        self.build_patterns(pats)
    }

    /// Build from class-syntax strings (see [`Pattern::parse`]):
    /// `Builder::new().build_parsed(["GET /api/v\d/", "[Hh]ost: "])`.
    pub fn build_parsed<I, S>(self, patterns: I) -> Result<Sparrow, BuildError>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let mut pats = Vec::new();
        for p in patterns {
            pats.push(Pattern::parse(p.as_ref()).map_err(BuildError::BadPattern)?);
        }
        self.build_patterns(pats)
    }

    /// Build from byte-class [`Pattern`]s. `ascii_case_insensitive` still
    /// applies (each position is case-closed); `wildcard_byte` does not —
    /// use [`ByteSet::ANY`].
    pub fn build_patterns<I>(self, patterns: I) -> Result<Sparrow, BuildError>
    where
        I: IntoIterator<Item = Pattern>,
    {
        let pats: Vec<Pat> = patterns
            .into_iter()
            .map(|p| {
                let p = if self.match_opts.case_insensitive { p.ascii_case_fold() } else { p };
                Pat::from_pattern(&p)
            })
            .collect();
        let opts = builder::BuildOpts {
            corpus: self.corpus.as_deref(),
            max_k: self.max_positions.unwrap_or(builder::MAX_K),
            forced_positions: self.forced_positions.as_deref(),
            exhaustive: self.exhaustive,
            corpus_score: self.corpus_score,
        };
        let engine = match self.forced_engine {
            Some(e) => {
                if !engine_available(e) {
                    return Err(BuildError::EngineUnavailable);
                }
                e
            }
            None => detect_engine(),
        };
        let routing = builder::RoutingOpts {
            dense_enabled: self.dense_lane,
            force_dense: self.force_dense,
            timed: self.timed_referee,
            engine,
        };
        let (cohorts, dense, decision) = builder::build_routed(&pats, &opts, &routing)?;
        if pats.is_empty() {
            return Err(BuildError::NoPatterns);
        }
        if pats.iter().any(|p| p.len() == 0) {
            return Err(BuildError::EmptyPattern);
        }
        let max_len = pats.iter().map(|p| p.len()).max().unwrap();
        Ok(Sparrow { patterns: pats, cohorts, dense, engine, max_len, decision })
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
    patterns: Vec<Pat>,
    cohorts: Vec<Compiled>,
    dense: Option<DenseLane>,
    engine: Engine,
    max_len: usize,
    decision: RoutingDecision,
}

/// How the dense-vs-sparse routing was decided at build time.
#[derive(Debug, Clone, PartialEq)]
pub struct RoutingDecision {
    /// `true` if the timed referee ran and chose; `false` if the model
    /// cost alone decided (referee off, corpus too small, or only one
    /// candidate).
    pub timed: bool,
    /// Every candidate partition the model considered, best model cost
    /// first: (dense-lane pattern count, model cost/byte, measured ns/byte
    /// if timed). Index 0 of `chosen` refers into this list.
    pub candidates: Vec<RoutingCandidate>,
    /// Index into `candidates` of the configuration that was built.
    pub chosen: usize,
}

/// One routing candidate as seen by the referee.
#[derive(Debug, Clone, PartialEq)]
pub struct RoutingCandidate {
    /// Patterns routed to the dense lane (0 = sparse-only).
    pub dense_patterns: usize,
    /// Summed model cost per haystack byte.
    pub model_cost: f64,
    /// Best-of-N measured scan time on the tiled corpus, ns/byte; `None`
    /// if this candidate was not timed.
    pub measured_ns_per_byte: Option<f64>,
}

/// Everything a kernel needs to scan and verify for one cohort.
pub(crate) struct ScanCtx<'a> {
    pub c: &'a Compiled,
    pub patterns: &'a [Pat],
}

/// Where kernels deliver matches: a `Vec` (`find_all`), a counter
/// (`count_all`), or a user callback (`scan_with`). Monomorphised — no
/// dynamic dispatch on the emit path.
pub(crate) trait Sink {
    fn accept(&mut self, m: Match);
}

impl Sink for Vec<Match> {
    #[inline(always)]
    fn accept(&mut self, m: Match) {
        self.push(m);
    }
}

pub(crate) struct CountSink(pub usize);
impl Sink for CountSink {
    #[inline(always)]
    fn accept(&mut self, _m: Match) {
        self.0 += 1;
    }
}

pub(crate) struct FnSink<F: FnMut(Match)>(pub F);
impl<F: FnMut(Match)> Sink for FnSink<F> {
    #[inline(always)]
    fn accept(&mut self, m: Match) {
        (self.0)(m);
    }
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
        scan_all(
            &self.cohorts,
            self.dense.as_ref(),
            &self.patterns,
            self.engine,
            haystack,
            &mut out,
        );
        out
    }

    /// Invoke `f` for every (overlapping) occurrence, without
    /// materialising a `Vec`. Emission order is the engines' native order:
    /// each cohort emits in increasing position, and the dense lane emits
    /// per 64 KB chunk in interleaved-segment order — *not* globally
    /// sorted. Use [`find_all`](Self::find_all) when order matters.
    pub fn scan_with<F: FnMut(Match)>(&self, haystack: &[u8], f: F) {
        let mut sink = FnSink(f);
        scan_unordered(
            &self.cohorts,
            self.dense.as_ref(),
            &self.patterns,
            self.engine,
            haystack,
            &mut sink,
        );
    }

    /// Number of (overlapping) occurrences, skipping the output path
    /// entirely (no allocation, no ordering). On match-dense inputs this
    /// is measurably faster than `find_all(..).len()`.
    pub fn count_all(&self, haystack: &[u8]) -> usize {
        let mut sink = CountSink(0);
        scan_unordered(
            &self.cohorts,
            self.dense.as_ref(),
            &self.patterns,
            self.engine,
            haystack,
            &mut sink,
        );
        sink.0
    }

    /// How the dense/sparse split was decided at build time.
    pub fn routing_decision(&self) -> &RoutingDecision {
        &self.decision
    }
}

/// Scan `hay` with a full configuration (sparse cohorts + optional dense
/// lane). Shared by [`Sparrow::find_all_unsorted`] and the builder's timed
/// routing referee, so the referee measures exactly what will run.
pub(crate) fn scan_all(
    cohorts: &[Compiled],
    dense: Option<&DenseLane>,
    patterns: &[Pat],
    engine: Engine,
    hay: &[u8],
    out: &mut Vec<Match>,
) {
    for c in cohorts {
        let ctx = ScanCtx { c, patterns };
        scan_cohort(engine, &ctx, hay, out);
    }
    if let Some(d) = dense {
        d.find_all(hay, out);
    }
}

/// Sink-directed variant of [`scan_all`]: same kernels, but the dense lane
/// skips its per-segment run ordering and emits straight into the sink.
pub(crate) fn scan_unordered<S: Sink>(
    cohorts: &[Compiled],
    dense: Option<&DenseLane>,
    patterns: &[Pat],
    engine: Engine,
    hay: &[u8],
    sink: &mut S,
) {
    for c in cohorts {
        let ctx = ScanCtx { c, patterns };
        scan_cohort(engine, &ctx, hay, sink);
    }
    if let Some(d) = dense {
        d.scan_unordered(hay, sink);
    }
}

fn scan_cohort<S: Sink>(engine: Engine, ctx: &ScanCtx<'_>, hay: &[u8], out: &mut S) {
        #[cfg(target_arch = "x86_64")]
        match engine {
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
        if engine == Engine::Neon && hay.len() >= 48 {
            // SAFETY: NEON is baseline on aarch64.
            unsafe { neon::find_all(ctx, hay, out) };
            return;
        }
        scalar::find_in_range(ctx, hay, 0, hay.len(), out);
}

impl Sparrow {
    /// Leftmost, non-overlapping match semantics (like a lazy scan-and-skip
    /// or aho-corasick's leftmost-first): at each leftmost match start the
    /// earliest-built pattern wins, and scanning resumes past its end.
    ///
    /// Reference implementation: materializes every overlapping match and
    /// filters. [`find_leftmost`](Self::find_leftmost) is the fast path.
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

    /// Leftmost-first, non-overlapping matches (aho-corasick / regex
    /// `LeftmostFirst` semantics: at each leftmost start the earliest-built
    /// pattern wins; scanning resumes at its end). Same result as
    /// [`find_leftmost_nonoverlapping`](Self::find_leftmost_nonoverlapping)
    /// without materializing the overlapping matches: the haystack is
    /// scanned in windows, re-entering the kernels at the end of each
    /// accepted match, so on match-dense inputs most overlapping
    /// occurrences are never verified, let alone sorted.
    pub fn find_leftmost(&self, haystack: &[u8]) -> Vec<Match> {
        self.leftmost(haystack, false)
    }

    /// Leftmost-longest, non-overlapping matches (POSIX-style: at each
    /// leftmost start the longest pattern wins, ties to the earliest-built).
    pub fn find_leftmost_longest(&self, haystack: &[u8]) -> Vec<Match> {
        self.leftmost(haystack, true)
    }

    fn leftmost(&self, hay: &[u8], longest: bool) -> Vec<Match> {
        const MIN_WIN: usize = 8 * 1024;
        const MAX_WIN: usize = 64 * 1024;
        let n = hay.len();
        let mut out = Vec::new();
        if self.cohorts.is_empty() {
            // Everything is in the dense lane: its Shift-Or state resets
            // at a match end, so leftmost is native and skips the bytes
            // inside accepted matches.
            if let Some(d) = &self.dense {
                d.find_leftmost(hay, longest, &mut out);
            }
            return out;
        }
        let mut buf: Vec<Match> = Vec::new();
        let mut pos = 0usize;
        let mut win = 16 * 1024usize;
        while pos < n {
            // Accept starts in [pos, win_end); scan far enough that any
            // pattern starting there is fully visible.
            let win_end = (pos + win).min(n);
            let scan_end = (win_end + self.max_len - 1).min(n);
            buf.clear();
            scan_all(&self.cohorts, self.dense.as_ref(), &self.patterns, self.engine, &hay[pos..scan_end], &mut buf);
            let limit = win_end - pos;
            buf.retain(|m| m.start < limit);
            buf.sort_unstable();
            let mut next = 0usize; // window-relative
            let mut i = 0;
            while i < buf.len() {
                if buf[i].start < next {
                    i += 1;
                    continue;
                }
                let s = buf[i].start;
                let mut best = buf[i];
                while i < buf.len() && buf[i].start == s {
                    let m = buf[i];
                    let better = if longest {
                        m.end > best.end || (m.end == best.end && m.pattern < best.pattern)
                    } else {
                        m.pattern < best.pattern
                    };
                    if better {
                        best = m;
                    }
                    i += 1;
                }
                next = best.end;
                out.push(Match { start: pos + best.start, end: pos + best.end, pattern: best.pattern });
            }
            // Resume past the last accepted match, or at the window end.
            pos = (pos + next).max(win_end);
            // Adapt: dense windows waste verification on discarded
            // overlaps; sparse ones waste per-call prelude/tail work.
            if buf.len() > 4096 {
                win = (win / 2).max(MIN_WIN);
            } else if buf.len() < 64 {
                win = (win * 2).min(MAX_WIN);
            }
        }
        out
    }

    /// Incremental scanning over a stream of chunks. Matches that span
    /// chunk boundaries are found; offsets are global to the stream.
    pub fn stream(&self) -> StreamScanner<'_> {
        StreamScanner { m: self, tail: Vec::new(), stitch: Vec::new(), consumed: 0 }
    }

    /// Force the portable scalar engine (testing / non-x86 reference).
    #[doc(hidden)]
    pub fn find_all_scalar(&self, haystack: &[u8]) -> Vec<Match> {
        let mut out = Vec::new();
        for c in &self.cohorts {
            let ctx = ScanCtx { c, patterns: &self.patterns };
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

    /// Buckets per cohort (8, 16, or 32 = one, two, or four SIMD planes).
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
            u.buckets += c
                .hashed
                .iter()
                .flatten()
                .map(|h| h.entries.len() * std::mem::size_of::<Entry>() + 257 * 4)
                .sum::<usize>();
        }
        if let Some(d) = &self.dense {
            u.scalar_tables += d.table_bytes();
        }
        u.patterns = self
            .patterns
            .iter()
            .map(|p| p.len() * (1 + if p.exact { 0 } else { 32 }) + std::mem::size_of::<Pat>())
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
    /// Reusable stitch buffer (tail + the first `max_len - 1` bytes of the
    /// incoming chunk) for boundary-spanning matches.
    stitch: Vec<u8>,
    /// Total stream bytes consumed before the current chunk.
    consumed: u64,
}

impl StreamScanner<'_> {
    /// Scan the next chunk. The chunk itself is scanned in place — no
    /// concatenation or copy of the chunk — and only a small stitched
    /// buffer (`tail` + the first `max_len − 1` chunk bytes) is scanned
    /// for boundary-spanning matches:
    ///
    /// * matches ending in the chunk's first `max_len − 1` bytes are found
    ///   in the stitch (they may start in the tail; any that start inside
    ///   the chunk are also fully contained in the stitch),
    /// * matches ending later cannot reach back into the tail
    ///   (`start = end − len > (max_len − 1) − max_len`), so the direct
    ///   chunk scan finds them,
    ///
    /// and the two regions partition every match by end offset, so nothing
    /// is duplicated or lost.
    ///
    /// Trade-off: filter-fast matchers stream 2–4× faster than the old
    /// concatenate-and-rescan path (the copy dominated); match-dense
    /// matchers on packet-sized chunks pay ~10% for the second (tiny)
    /// boundary scan.
    pub fn push(&mut self, chunk: &[u8]) -> Vec<StreamMatch> {
        let max_len = self.m.max_len;
        let tail_len = self.tail.len();
        let mut out: Vec<StreamMatch> = Vec::new();
        if chunk.is_empty() {
            return out;
        }
        // Boundary region: ends in chunk[..take] (relative to the chunk).
        let take = if tail_len == 0 { 0 } else { (max_len - 1).min(chunk.len()) };
        let boundary: Vec<Match> = if take > 0 {
            self.stitch.clear();
            self.stitch.extend_from_slice(&self.tail);
            self.stitch.extend_from_slice(&chunk[..take]);
            self.m.find_all(&self.stitch)
        } else {
            Vec::new()
        };
        let inner = self.m.find_all(chunk);
        // Merge the two sorted groups (they interleave only within one
        // `max_len` window at the boundary) — no per-push sort.
        let sbase = self.consumed - tail_len as u64;
        let cbase = self.consumed;
        let mut bi = boundary
            .into_iter()
            // Ends inside the tail were reported by earlier pushes.
            .filter(|m| m.end > tail_len)
            .map(|m| StreamMatch {
                start: sbase + m.start as u64,
                end: sbase + m.end as u64,
                pattern: m.pattern,
            })
            .peekable();
        let mut ci = inner
            .into_iter()
            .filter(|m| m.end > take)
            .map(|m| StreamMatch {
                start: cbase + m.start as u64,
                end: cbase + m.end as u64,
                pattern: m.pattern,
            })
            .peekable();
        loop {
            match (bi.peek(), ci.peek()) {
                (Some(b), Some(c)) => {
                    if b <= c {
                        out.push(bi.next().unwrap());
                    } else {
                        out.push(ci.next().unwrap());
                    }
                }
                (Some(_), None) => out.extend(bi.by_ref()),
                (None, Some(_)) => out.extend(ci.by_ref()),
                (None, None) => break,
            }
        }
        // New tail: last `max_len - 1` stream bytes (fewer near the start).
        let keep = (max_len - 1).min(tail_len + chunk.len());
        if chunk.len() >= keep {
            self.tail.clear();
            self.tail.extend_from_slice(&chunk[chunk.len() - keep..]);
        } else {
            self.tail.drain(..tail_len + chunk.len() - keep);
            self.tail.extend_from_slice(chunk);
        }
        self.consumed += chunk.len() as u64;
        out
    }
}

/// Exact membership test of a window against a pattern (memcmp for exact
/// patterns, per-position set tests otherwise).
#[inline]
pub(crate) fn pattern_matches(window: &[u8], pat: &Pat) -> bool {
    pat.matches(window)
}

/// Exact verification of a candidate anchor for one bucket plane: check
/// every entry of every flagged bucket at the implied start position, guard
/// byte first. Shared by all engines.
#[inline]
pub(crate) fn verify_at<S: Sink>(
    ctx: &ScanCtx<'_>,
    hay: &[u8],
    t: usize,
    mut mask: u8,
    plane: usize,
    out: &mut S,
) {
    // Window start implied by the anchor. Bits produced near the beginning
    // of the haystack can imply a start before offset 0; those are filter
    // artifacts, not matches.
    let Some(start) = t.checked_sub(ctx.c.s_last) else { return };
    let base = plane * builder::PLANE_BUCKETS;
    // Fingerprint of the sampled haystack bytes at this anchor, computed
    // at most once per candidate and shared by every flagged bucket (the
    // sampled offsets are cohort-wide). Cheap: the filter just read those
    // k bytes, they are in cache.
    let mut fp: Option<u8> = None;
    while mask != 0 {
        let b = mask.trailing_zeros() as usize;
        mask &= mask - 1;
        if let Some(hb) = &ctx.c.hashed[base + b] {
            let f = *fp.get_or_insert_with(|| {
                builder::fingerprint(ctx.c.d[..ctx.c.k].iter().map(|&d| hay[t - d]))
            }) as usize;
            let (lo, hi) = (hb.offsets[f] as usize, hb.offsets[f + 1] as usize);
            for &Entry { id, guard_off } in &hb.entries[lo..hi] {
                check_entry(ctx, hay, start, id, guard_off, out);
            }
        }
        for &Entry { id, guard_off } in &ctx.c.buckets[base + b] {
            check_entry(ctx, hay, start, id, guard_off, out);
        }
    }
}

/// Verify one bucket entry at `start`: bounds, guard probe (the
/// model-rarest unsampled pattern position — rejects most false
/// candidates with one load), then the full comparison.
#[inline(always)]
fn check_entry<S: Sink>(
    ctx: &ScanCtx<'_>,
    hay: &[u8],
    start: usize,
    id: u32,
    guard_off: u32,
    out: &mut S,
) {
    let p = &ctx.patterns[id as usize];
    let end = start + p.len();
    if end > hay.len() {
        return;
    }
    let g = guard_off as usize;
    let hb = hay[start + g];
    if if p.exact { hb != p.bytes[g] } else { !p.sets[g].contains(hb) } {
        return;
    }
    if pattern_matches(&hay[start..end], p) {
        out.accept(Match { start, end, pattern: id as usize });
    }
}
