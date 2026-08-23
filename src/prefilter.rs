//! Regex literal prefilter (feature `prefilter`): extract each regex's
//! required *prefix literals* with `regex-syntax`, match the union of those
//! literals with SPARROW, and confirm each candidate with an anchored
//! `regex-automata` search starting at the literal. This is how Hyperscan
//! and `regex`'s own Teddy integration get their throughput; SPARROW's
//! shared-prefix advantage maps directly onto IDS-style rule sets whose
//! literals share long prefixes.
//!
//! Guarantee: **no false negatives.** Every match of a regex begins with
//! one of its extracted prefix literals (that is what "prefix literal set"
//! means in `regex-syntax`: a finite, complete set of possible match
//! prefixes), so every match start is a candidate. Regexes with no usable
//! finite prefix set (`.*foo`, `a*`, anything that can match empty) are
//! kept *unfiltered* and run over the whole haystack by `regex-automata`,
//! so the API is complete for every regex it accepts.
//!
//! Semantics of [`Prefilter::find_all`]: for each regex, exactly the
//! matches `find_iter` would report (leftmost-first, non-overlapping per
//! regex), sorted by `(start, end, regex)`.

use crate::{Builder, BuildError, Match, Sparrow};
use regex_automata::meta::Regex;
use regex_automata::{Anchored, Input};
use regex_syntax::hir::literal::{ExtractKind, Extractor};
use regex_syntax::ParserBuilder;

/// A regex match.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct RegexMatch {
    pub start: usize,
    pub end: usize,
    /// Index of the regex in the order given at build time.
    pub regex: usize,
}

/// A position at which `regex` *may* match (its required literal occurs).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Candidate {
    pub start: usize,
    pub regex: usize,
}

#[derive(Debug)]
pub enum PrefilterError {
    /// The regex failed to parse/compile (index, message).
    Regex(usize, String),
    /// Building the literal matcher failed.
    Build(BuildError),
    NoRegexes,
}

impl std::fmt::Display for PrefilterError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PrefilterError::Regex(i, m) => write!(f, "regex {}: {}", i, m),
            PrefilterError::Build(e) => write!(f, "literal matcher: {}", e),
            PrefilterError::NoRegexes => write!(f, "no regexes"),
        }
    }
}
impl std::error::Error for PrefilterError {}

/// A set of regexes with a SPARROW literal prefilter in front.
pub struct Prefilter {
    regexes: Vec<Regex>,
    /// Literal matcher over the distinct prefix literals of filtered
    /// regexes; `None` if no regex is filterable.
    lits: Option<Sparrow>,
    /// literal id -> regex ids requiring it.
    lit_regexes: Vec<Vec<u32>>,
    /// Regexes with no usable prefix literal set.
    unfiltered: Vec<usize>,
    literal_bytes: Vec<Vec<u8>>,
}

impl Prefilter {
    /// Build from regex patterns (byte-oriented: `(?-u)` semantics, so
    /// `.` and classes match arbitrary bytes). `corpus` tunes SPARROW's
    /// position optimizer like [`Builder::corpus_sample`].
    pub fn new<S: AsRef<str>>(patterns: &[S], corpus: Option<&[u8]>) -> Result<Prefilter, PrefilterError> {
        if patterns.is_empty() {
            return Err(PrefilterError::NoRegexes);
        }
        let mut regexes = Vec::with_capacity(patterns.len());
        let mut lit_index: std::collections::HashMap<Vec<u8>, usize> = Default::default();
        let mut literal_bytes: Vec<Vec<u8>> = Vec::new();
        let mut lit_regexes: Vec<Vec<u32>> = Vec::new();
        let mut unfiltered = Vec::new();
        for (i, p) in patterns.iter().enumerate() {
            let p = p.as_ref();
            let re = Regex::builder()
                .syntax(regex_automata::util::syntax::Config::new().utf8(false))
                .build(p)
                .map_err(|e| PrefilterError::Regex(i, e.to_string()))?;
            regexes.push(re);
            let hir = ParserBuilder::new()
                .utf8(false)
                .build()
                .parse(p)
                .map_err(|e| PrefilterError::Regex(i, e.to_string()))?;
            let seq = Extractor::new().kind(ExtractKind::Prefix).extract(&hir);
            let lits = match seq.literals() {
                Some(ls) if !ls.is_empty() && ls.iter().all(|l| !l.as_bytes().is_empty()) => ls,
                _ => {
                    unfiltered.push(i);
                    continue;
                }
            };
            for l in lits {
                let b = l.as_bytes().to_vec();
                let id = *lit_index.entry(b.clone()).or_insert_with(|| {
                    literal_bytes.push(b);
                    lit_regexes.push(Vec::new());
                    literal_bytes.len() - 1
                });
                if lit_regexes[id].last() != Some(&(i as u32)) {
                    lit_regexes[id].push(i as u32);
                }
            }
        }
        let lits = if literal_bytes.is_empty() {
            None
        } else {
            let mut b = Builder::new();
            if let Some(c) = corpus {
                b = b.corpus_sample(c);
            }
            Some(b.build(&literal_bytes).map_err(PrefilterError::Build)?)
        };
        Ok(Prefilter { regexes, lits, lit_regexes, unfiltered, literal_bytes })
    }

    pub fn regex_count(&self) -> usize {
        self.regexes.len()
    }
    /// Regexes that could not be given a literal prefilter.
    pub fn unfiltered(&self) -> &[usize] {
        &self.unfiltered
    }
    /// The distinct literals the prefilter scans for.
    pub fn literals(&self) -> &[Vec<u8>] {
        &self.literal_bytes
    }
    /// The literal matcher (for `routing_decision`, `memory_usage`, ...).
    pub fn literal_matcher(&self) -> Option<&Sparrow> {
        self.lits.as_ref()
    }

    /// Candidate `(start, regex)` pairs for filtered regexes, sorted and
    /// deduplicated. Unfiltered regexes are not represented (every
    /// position is a candidate for them).
    pub fn candidates(&self, hay: &[u8]) -> Vec<Candidate> {
        let mut out = Vec::new();
        if let Some(m) = &self.lits {
            for Match { start, pattern, .. } in m.find_all_unsorted(hay) {
                for &r in &self.lit_regexes[pattern] {
                    out.push(Candidate { start, regex: r as usize });
                }
            }
        }
        out.sort_unstable();
        out.dedup();
        out
    }

    /// All matches of all regexes (per regex: what `find_iter` reports),
    /// sorted by `(start, end, regex)`.
    pub fn find_all(&self, hay: &[u8]) -> Vec<RegexMatch> {
        let mut out = Vec::new();
        // Candidates come sorted by (start, regex); walk per regex with a
        // non-overlap cursor, exactly like find_iter's resume rule.
        let mut next: Vec<usize> = vec![0; self.regexes.len()];
        for c in self.candidates(hay) {
            if c.start < next[c.regex] {
                continue;
            }
            let input = Input::new(hay).span(c.start..hay.len()).anchored(Anchored::Yes);
            if let Some(m) = self.regexes[c.regex].find(input) {
                out.push(RegexMatch { start: m.start(), end: m.end(), regex: c.regex });
                next[c.regex] = if m.end() > m.start() { m.end() } else { m.end() + 1 };
            }
        }
        for &r in &self.unfiltered {
            for m in self.regexes[r].find_iter(hay) {
                out.push(RegexMatch { start: m.start(), end: m.end(), regex: r });
            }
        }
        out.sort_unstable();
        out
    }

    /// Does any regex match?
    pub fn is_match(&self, hay: &[u8]) -> bool {
        for c in self.candidates(hay) {
            let input = Input::new(hay).span(c.start..hay.len()).anchored(Anchored::Yes);
            if self.regexes[c.regex].is_match(input) {
                return true;
            }
        }
        self.unfiltered.iter().any(|&r| self.regexes[r].is_match(hay))
    }
}
