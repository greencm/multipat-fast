//! Brute-force reference matcher used by the test suite and benchmarks as
//! ground truth. O(n * m) — correct by inspection.

use crate::{Match, MatchOpts};

/// Every overlapping occurrence of every pattern, sorted by
/// `(start, end, pattern)`.
pub fn find_all<P: AsRef<[u8]>>(patterns: &[P], haystack: &[u8]) -> Vec<Match> {
    find_all_with(patterns, haystack, false, None)
}

/// Reference matcher with the same optional semantics as [`crate::Builder`]:
/// ASCII case-insensitivity and a single-byte wildcard in patterns.
pub fn find_all_with<P: AsRef<[u8]>>(
    patterns: &[P],
    haystack: &[u8],
    case_insensitive: bool,
    wildcard: Option<u8>,
) -> Vec<Match> {
    let opts = MatchOpts { case_insensitive, wildcard };
    let mut out = Vec::new();
    for (id, p) in patterns.iter().enumerate() {
        let p = p.as_ref();
        if p.is_empty() || p.len() > haystack.len() {
            continue;
        }
        for start in 0..=haystack.len() - p.len() {
            if crate::pattern_matches(&haystack[start..start + p.len()], p, &opts) {
                out.push(Match { start, end: start + p.len(), pattern: id });
            }
        }
    }
    out.sort_unstable();
    out
}
