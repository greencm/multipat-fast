//! Brute-force reference matcher used by the test suite and benchmarks as
//! ground truth. O(n * m) — correct by inspection.

use crate::Match;

/// Every overlapping occurrence of every pattern, sorted by `(start, pattern)`.
pub fn find_all<P: AsRef<[u8]>>(patterns: &[P], haystack: &[u8]) -> Vec<Match> {
    let mut out = Vec::new();
    for (id, p) in patterns.iter().enumerate() {
        let p = p.as_ref();
        if p.is_empty() || p.len() > haystack.len() {
            continue;
        }
        for start in 0..=haystack.len() - p.len() {
            if &haystack[start..start + p.len()] == p {
                out.push(Match { start, end: start + p.len(), pattern: id });
            }
        }
    }
    out.sort_unstable();
    out
}
