//! Native leftmost scanning must equal the reference (materialize-and-
//! filter) implementation and aho-corasick's leftmost-first.

use aho_corasick::{AhoCorasick, MatchKind};
use sparrow::{Builder, Match, Pattern};

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

fn ac(patterns: &[Vec<u8>], hay: &[u8], kind: MatchKind) -> Vec<Match> {
    AhoCorasick::builder()
        .match_kind(kind)
        .build(patterns)
        .unwrap()
        .find_iter(hay)
        .map(|m| Match { start: m.start(), end: m.end(), pattern: m.pattern().as_usize() })
        .collect()
}

fn check(patterns: &[Vec<u8>], hay: &[u8]) -> usize {
    let corpus = &hay[..hay.len().min(65536)];
    for b in [
        Builder::new().corpus_sample(corpus).dense_lane(false),
        Builder::new().corpus_sample(corpus),
        Builder::new().corpus_sample(corpus).force_dense(true),
    ] {
        let m = b.build(patterns).unwrap();
        let reference = m.find_leftmost_nonoverlapping(hay);
        assert_eq!(m.find_leftmost(hay), reference, "leftmost-first vs reference");
        assert_eq!(reference, ac(patterns, hay, MatchKind::LeftmostFirst), "vs aho-corasick");
        let longest = m.find_leftmost_longest(hay);
        assert_eq!(longest, ac(patterns, hay, MatchKind::LeftmostLongest), "longest vs aho-corasick");
    }
    ac(patterns, hay, MatchKind::LeftmostFirst).len()
}

fn english(rng: &mut Rng, n: usize) -> Vec<u8> {
    let words = ["the", "and", "for", "theme", "therefore", "android", "thethe", "a", "an", "x", "hall", "over"];
    let mut h = Vec::with_capacity(n + 16);
    while h.len() < n {
        h.extend_from_slice(words[rng.below(words.len())].as_bytes());
        if rng.below(3) != 0 {
            h.push(b' ');
        }
    }
    h.truncate(n);
    h
}

fn pats(v: &[&str]) -> Vec<Vec<u8>> {
    v.iter().map(|s| s.as_bytes().to_vec()).collect()
}

#[test]
fn leftmost_dense_overlapping_words() {
    let mut rng = Rng(1);
    let hay = english(&mut rng, 600_000);
    // Order matters for leftmost-first: "the" before "theme" / "therefore".
    let p = pats(&["the", "theme", "therefore", "and", "android", "a", "an", "thethe"]);
    assert!(check(&p, &hay) > 10_000);
    let p2 = pats(&["therefore", "theme", "the", "android", "and", "an", "a"]);
    check(&p2, &hay);
}

#[test]
fn leftmost_sparse_and_empty() {
    let mut rng = Rng(2);
    let hay: Vec<u8> = (0..300_000).map(|_| rng.next() as u8).collect();
    let p: Vec<Vec<u8>> = (0..8).map(|_| (0..6).map(|_| rng.next() as u8).collect()).collect();
    check(&p, &hay);
    let mut planted = hay.clone();
    for i in (0..planted.len() - 8).step_by(9973) {
        let q = &p[i % p.len()];
        planted[i..i + q.len()].copy_from_slice(q);
    }
    assert!(check(&p, &planted) > 20);
    check(&p, b"");
    check(&p, &hay[..3]);
}

#[test]
fn leftmost_matches_spanning_window_edges() {
    // Long patterns relative to the minimum window (256) and matches that
    // straddle every power-of-two boundary a window could land on.
    let mut rng = Rng(3);
    let mut hay = english(&mut rng, 200_000);
    let long = b"0123456789abcdefghijklmnopqrstu"; // 31 bytes
    for b in (256..hay.len() - 64).step_by(256) {
        let off = rng.below(31);
        hay[b - off..b - off + long.len()].copy_from_slice(long);
    }
    let mut p = pats(&["the", "and", "a"]);
    p.push(long.to_vec());
    p.push(b"0123".to_vec());
    check(&p, &hay);
}

#[test]
fn leftmost_with_class_patterns() {
    let mut rng = Rng(4);
    let mut hay = english(&mut rng, 100_000);
    for i in (0..hay.len()).step_by(37) {
        hay[i] = b'0' + (i % 10) as u8;
    }
    let pats: Vec<Pattern> = vec![Pattern::parse(r"\d").unwrap(), Pattern::parse(r"\dthe").unwrap(), Pattern::bytes(b"the")];
    let m = Builder::new().build_patterns(pats).unwrap();
    assert_eq!(m.find_leftmost(&hay), m.find_leftmost_nonoverlapping(&hay));
}

#[test]
fn fuzz_leftmost_small_alphabet() {
    let mut rng = Rng(0x1EF7);
    for round in 0..200 {
        let alpha = 2 + rng.below(3);
        let np = 1 + rng.below(8);
        let p: Vec<Vec<u8>> = (0..np)
            .map(|_| (0..1 + rng.below(5)).map(|_| b'a' + rng.below(alpha) as u8).collect())
            .collect();
        let n = rng.below(if round % 10 == 0 { 50_000 } else { 600 });
        let hay: Vec<u8> = (0..n).map(|_| b'a' + rng.below(alpha) as u8).collect();
        check(&p, &hay);
    }
}
