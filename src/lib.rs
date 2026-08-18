//! # SPARROW — Sparse Position Adaptive Rejection Over Rolling Windows
//!
//! A SIMD multi-pattern matcher built around a new prefilter design:
//! instead of inspecting a *fixed, contiguous* run of pattern bytes (the
//! prefix in Teddy, the whole literal in Harry, the tail domain in
//! Hyperscan's FDR), SPARROW samples a **sparse, optimizer-chosen set of
//! positions** inside the patterns' common anchor window, and assigns
//! patterns to its 8 SIMD buckets by **minimizing an exact expected
//! verification-cost objective** under a byte-distribution model that also
//! models the PSHUFB nibble cross-product closure exactly.
//!
//! Guarantees (proved in `docs/DESIGN.md`):
//! * **No false negatives** — every occurrence of every pattern is reported.
//! * The optimizer's objective equals the **exact** expected verification
//!   cost per haystack byte under the i.i.d. byte model, nibble closure
//!   included.
//! * The bucket refinement terminates and returns a move-optimal partition.
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
pub mod naive;

use builder::Compiled;

/// A single pattern occurrence. All overlapping occurrences of all patterns
/// are reported, sorted by `(start, pattern)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Match {
    /// Byte offset of the first matched byte.
    pub start: usize,
    /// One past the last matched byte (`start + pattern length`).
    pub end: usize,
    /// Index of the pattern (in the order given at build time).
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
    /// strictly inside the common anchor window `[0, min(min_len, 16))`.
    BadPositions,
}

impl core::fmt::Display for BuildError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            BuildError::NoPatterns => write!(f, "pattern set is empty"),
            BuildError::EmptyPattern => write!(f, "patterns must be non-empty"),
            BuildError::BadPositions => {
                write!(f, "forced positions must be 1..=4 offsets inside [0, min(min_len, 16))")
            }
        }
    }
}

impl std::error::Error for BuildError {}

/// Configures and builds a [`Sparrow`] matcher.
#[derive(Debug, Clone, Default)]
pub struct Builder {
    corpus: Option<Vec<u8>>,
    max_positions: Option<usize>,
    forced_positions: Option<Vec<u8>>,
    exhaustive: bool,
}

impl Builder {
    pub fn new() -> Builder {
        Builder {
            corpus: None,
            max_positions: None,
            forced_positions: None,
            exhaustive: cfg!(feature = "exhaustive-search"),
        }
    }

    /// Provide a sample of the traffic the matcher will scan. The optimizer
    /// estimates the background byte distribution from it (with Laplace
    /// smoothing); the closer the sample is to the real workload, the closer
    /// the realized candidate rate is to the model-optimal one. Without a
    /// sample, a built-in mixed text/markup/code model is used.
    pub fn corpus_sample(mut self, sample: &[u8]) -> Builder {
        self.corpus = Some(sample.to_vec());
        self
    }

    /// Cap the number of sampled positions (1..=4). More positions cost one
    /// extra shuffle-pair + shift + AND per 32 haystack bytes but reject
    /// more candidates. Default: 4 (or fewer if the shortest pattern is
    /// shorter).
    pub fn max_positions(mut self, k: usize) -> Builder {
        self.max_positions = Some(k);
        self
    }

    /// Force the sampled positions (offsets from each pattern's start),
    /// bypassing position optimization. Bucket assignment is still
    /// optimized. Useful for ablation: `positions(&[0, 1, 2, 3])` gives a
    /// Teddy-style contiguous-prefix filter inside the same runtime.
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

    pub fn build<I, P>(self, patterns: I) -> Result<Sparrow, BuildError>
    where
        I: IntoIterator<Item = P>,
        P: AsRef<[u8]>,
    {
        let pats: Vec<Box<[u8]>> =
            patterns.into_iter().map(|p| p.as_ref().to_vec().into_boxed_slice()).collect();
        let compiled = builder::build(
            pats,
            self.corpus.as_deref(),
            self.max_positions.unwrap_or(builder::MAX_K),
            self.forced_positions.as_deref(),
            self.exhaustive,
        )?;
        Ok(Sparrow {
            compiled,
            #[cfg(target_arch = "x86_64")]
            use_avx2: std::arch::is_x86_feature_detected!("avx2"),
        })
    }
}

/// A compiled multi-pattern matcher. Build once, search many haystacks.
pub struct Sparrow {
    compiled: Compiled,
    #[cfg(target_arch = "x86_64")]
    use_avx2: bool,
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
    /// `(start, pattern)`.
    pub fn find_all(&self, haystack: &[u8]) -> Vec<Match> {
        let mut out = Vec::new();
        #[cfg(target_arch = "x86_64")]
        if self.use_avx2 && haystack.len() >= 32 {
            // SAFETY: gated on runtime AVX2 detection.
            unsafe { avx2::find_all(&self.compiled, haystack, &mut out) };
            out.sort_unstable();
            return out;
        }
        scalar::find_in_range(&self.compiled, haystack, 0, haystack.len(), &mut out);
        out.sort_unstable();
        out
    }

    /// Force the portable scalar engine (testing / non-x86 reference).
    #[doc(hidden)]
    pub fn find_all_scalar(&self, haystack: &[u8]) -> Vec<Match> {
        let mut out = Vec::new();
        scalar::find_in_range(&self.compiled, haystack, 0, haystack.len(), &mut out);
        out.sort_unstable();
        out
    }

    /// The sampled positions the optimizer chose (offsets from pattern
    /// start, ascending).
    pub fn sampled_positions(&self) -> &[u8] {
        &self.compiled.positions
    }

    /// Model-expected verification cost per haystack byte — the objective
    /// the optimizer minimized (units: pattern comparisons, including a
    /// fixed per-candidate overhead).
    pub fn expected_cost(&self) -> f64 {
        self.compiled.expected_cost
    }

    /// Model-expected candidate bits per haystack byte (filter pass rate).
    pub fn expected_candidate_rate(&self) -> f64 {
        self.compiled.expected_candidates
    }

    pub fn pattern_count(&self) -> usize {
        self.compiled.patterns.len()
    }

    /// Length of the shortest pattern; the anchor window is its first
    /// `min(16, min_pattern_len)` bytes.
    pub fn min_pattern_len(&self) -> usize {
        self.compiled.min_len
    }
}

/// Exact verification of a candidate anchor: check every pattern of every
/// flagged bucket at the implied start position. Shared by both engines.
#[inline]
pub(crate) fn verify_at(c: &Compiled, hay: &[u8], t: usize, mut mask: u8, out: &mut Vec<Match>) {
    // Window start implied by the anchor. Bits produced near the beginning
    // of the haystack can imply a start before offset 0; those are filter
    // artifacts, not matches.
    let Some(start) = t.checked_sub(c.s_last) else { return };
    while mask != 0 {
        let b = mask.trailing_zeros() as usize;
        mask &= mask - 1;
        for &id in &c.buckets[b] {
            let p = &c.patterns[id as usize];
            let end = start + p.len();
            if end <= hay.len() && &hay[start..end] == &p[..] {
                out.push(Match { start, end, pattern: id as usize });
            }
        }
    }
}
