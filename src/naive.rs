//! Brute-force reference matcher used by the test suite and benchmarks as
//! ground truth. O(n * m) — correct by inspection.

use crate::pattern::Pat;
use crate::{ByteSet, Match, Pattern};

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
    let pats: Vec<Pattern> = patterns
        .iter()
        .map(|p| {
            let mut pat = Pattern::new();
            for &b in p.as_ref() {
                pat.push(if wildcard == Some(b) {
                    ByteSet::ANY
                } else if case_insensitive {
                    ByteSet::byte(b).ascii_case_fold()
                } else {
                    ByteSet::byte(b)
                });
            }
            pat
        })
        .collect();
    find_all_patterns(&pats, haystack)
}

/// Reference matcher for byte-class [`Pattern`]s.
pub fn find_all_patterns(patterns: &[Pattern], haystack: &[u8]) -> Vec<Match> {
    let mut out = Vec::new();
    for (id, p) in patterns.iter().enumerate() {
        let p = Pat::from_pattern(p);
        if p.len() == 0 || p.len() > haystack.len() {
            continue;
        }
        for start in 0..=haystack.len() - p.len() {
            if p.matches(&haystack[start..start + p.len()]) {
                out.push(Match { start, end: start + p.len(), pattern: id });
            }
        }
    }
    out.sort_unstable();
    out
}
