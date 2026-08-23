#![cfg(feature = "prefilter")]
//! The regex prefilter must report exactly what regex::bytes find_iter
//! reports, per regex — no false negatives from literal extraction, no
//! duplicates from overlapping candidate literals.

use sparrow::prefilter::{Prefilter, RegexMatch};

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

fn reference(patterns: &[&str], hay: &[u8]) -> Vec<RegexMatch> {
    let mut out = Vec::new();
    for (i, p) in patterns.iter().enumerate() {
        let re = regex::bytes::RegexBuilder::new(p).unicode(false).build().unwrap();
        for m in re.find_iter(hay) {
            out.push(RegexMatch { start: m.start(), end: m.end(), regex: i });
        }
    }
    out.sort_unstable();
    out
}

fn check(patterns: &[&str], hay: &[u8]) -> usize {
    let pf = Prefilter::new(patterns, Some(&hay[..hay.len().min(65536)])).unwrap();
    let got = pf.find_all(hay);
    let want = reference(patterns, hay);
    assert_eq!(got, want, "patterns={:?}", patterns);
    assert_eq!(pf.is_match(hay), !want.is_empty());
    want.len()
}

fn log_haystack(rng: &mut Rng, n: usize) -> Vec<u8> {
    let mut h = Vec::with_capacity(n + 64);
    let mut i = 0usize;
    while h.len() < n {
        let line = match rng.below(6) {
            0 => format!("GET /api/v1/zz{:03}?p={} HTTP/1.1\n", i % 1000, i % 7),
            1 => format!("GET /api/v1/users/{} HTTP/1.1\n", rng.below(100000)),
            2 => format!("POST /api/v2/items?id={:x}&q=a+b\n", rng.next() as u32),
            3 => format!("Host: svc-{}.internal\n", i % 13),
            4 => format!("User-Agent: curl/8.{}.{}\n", rng.below(10), rng.below(10)),
            _ => format!("X-Trace: {:08x}-{:04x}\n", rng.next() as u32, rng.next() as u16),
        };
        h.extend_from_slice(line.as_bytes());
        i += 1;
    }
    h.truncate(n);
    h
}

#[test]
fn prefilter_log_rules() {
    let mut rng = Rng(1);
    let hay = log_haystack(&mut rng, 400_000);
    let rules = [
        r"GET /api/v1/users/\d+",
        r"GET /api/v\d/zz0\d\d\?p=[0-3]",
        r"User-Agent: (?:curl|wget)/\d+\.\d+",
        r"Host: svc-1?\d\.internal",
        r"X-Trace: [0-9a-f]{8}-[0-9a-f]{4}",
        r"POST /api/v2/items\?id=[0-9a-f]+",
    ];
    let n = check(&rules, &hay);
    assert!(n > 1000, "expected plenty of matches, got {n}");
    let pf = Prefilter::new(&rules, None).unwrap();
    assert!(pf.unfiltered().is_empty(), "all these rules have prefix literals");
    assert!(pf.literals().len() >= rules.len());
}

#[test]
fn unfiltered_regexes_still_exact() {
    let mut rng = Rng(2);
    let hay = log_haystack(&mut rng, 100_000);
    // `.*` prefix and empty-matchable patterns cannot be prefiltered.
    let rules = [r".*internal", r"(?:curl)?/8", r"svc-\d+", r"z*9"];
    let pf = Prefilter::new(&rules, None).unwrap();
    assert!(!pf.unfiltered().is_empty());
    check(&rules, &hay);
}

#[test]
fn shared_prefix_rules_and_candidates() {
    let mut rng = Rng(3);
    let hay = log_haystack(&mut rng, 300_000);
    let rules: Vec<String> =
        (0..20).map(|i| format!(r"GET /api/v1/users/{}\d* HTTP", i)).collect();
    let refs: Vec<&str> = rules.iter().map(|s| s.as_str()).collect();
    check(&refs, &hay);
    let pf = Prefilter::new(&refs, Some(&hay[..65536])).unwrap();
    let cands = pf.candidates(&hay);
    let matches = pf.find_all(&hay).len();
    // The point of the prefilter: candidates are rare relative to bytes.
    assert!(cands.len() < hay.len() / 50, "{} candidates", cands.len());
    assert!(matches <= cands.len());
}

#[test]
fn anchors_and_alternations() {
    let hay = b"foo bar\nbaz foo qux\nfoo";
    check(&[r"^foo", r"foo$", r"(?m)^foo", r"(?m)foo$", r"ba[rz]", r"\bfoo\b"], hay);
}

#[test]
fn fuzz_small_regexes() {
    let mut rng = Rng(0xF17E);
    let pieces = [r"ab", r"a\d", r"[ab]c", r"a+b", r"(?:ab|cd)", r"a.c", r"x[0-9]{2}"];
    for _ in 0..80 {
        let np = 1 + rng.below(4);
        let rules: Vec<&str> = (0..np).map(|_| pieces[rng.below(pieces.len())]).collect();
        let n = 200 + rng.below(3000);
        let hay: Vec<u8> =
            (0..n).map(|_| b"abcdx0123 "[rng.below(10)]).collect();
        check(&rules, &hay);
    }
}
