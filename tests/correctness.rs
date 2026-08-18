//! Differential correctness tests: SPARROW (SIMD and scalar engines) must
//! report exactly the same match set as the brute-force reference on
//! planted-occurrence workloads, adversarial edges, and random fuzzing.

use sparrow::{naive, Builder, BuildError, Match, Sparrow};

/// splitmix64 — deterministic tiny PRNG, no dependencies.
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

fn check(patterns: &[Vec<u8>], hay: &[u8]) {
    let m = Sparrow::new(patterns).expect("build should succeed");
    let expect = naive::find_all(patterns, hay);
    let got_simd = m.find_all(hay);
    let got_scalar = m.find_all_scalar(hay);
    assert_eq!(got_scalar, expect, "scalar mismatch (patterns={:?})", patterns);
    assert_eq!(got_simd, expect, "simd mismatch (patterns={:?})", patterns);
}

#[test]
fn simple_smoke() {
    let pats: Vec<Vec<u8>> = ["sparrow", "swallow", "raven"]
        .iter()
        .map(|s| s.as_bytes().to_vec())
        .collect();
    let hay = b"a sparrow and a swallow; the raven watched the sparrow";
    let m = Sparrow::new(&pats).unwrap();
    let hits = m.find_all(hay);
    assert_eq!(
        hits,
        vec![
            Match { start: 2, end: 9, pattern: 0 },
            Match { start: 16, end: 23, pattern: 1 },
            Match { start: 29, end: 34, pattern: 2 },
            Match { start: 47, end: 54, pattern: 0 },
        ]
    );
}

#[test]
fn build_errors() {
    assert_eq!(Sparrow::new(Vec::<Vec<u8>>::new()).err(), Some(BuildError::NoPatterns));
    assert_eq!(Sparrow::new([b"".to_vec()]).err(), Some(BuildError::EmptyPattern));
    assert_eq!(
        Builder::new().positions(&[9]).build(["short"]).err(),
        Some(BuildError::BadPositions),
        "position beyond min pattern length must be rejected"
    );
    assert_eq!(
        Builder::new().positions(&[]).build(["short"]).err(),
        Some(BuildError::BadPositions)
    );
    assert_eq!(
        Builder::new().positions(&[0, 1, 2, 3, 4]).build(["abcdefgh"]).err(),
        Some(BuildError::BadPositions)
    );
}

#[test]
fn edges_and_boundaries() {
    let pats: Vec<Vec<u8>> = vec![b"ab".to_vec(), b"abcd".to_vec(), b"dab".to_vec()];
    // Match at offset 0, at the very end, overlapping, across the 32-byte
    // SIMD block boundary, and exactly-31/32/33-byte haystacks.
    check(&pats, b"ab");
    check(&pats, b"abcdab");
    check(&pats, b"xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxab"); // ends at 32
    check(&pats, b"xxxxxxxxxxxxxxxxxxxxxxxxxxxxxabcdxxxxxxx"); // spans boundary
    check(&pats, &vec![b'a'; 31]);
    check(&pats, &vec![b'a'; 32]);
    check(&pats, &vec![b'a'; 33]);
    check(&pats, b"");
    check(&pats, b"a"); // shorter than every pattern
}

#[test]
fn self_overlapping_and_repeats() {
    let pats: Vec<Vec<u8>> = vec![b"aaa".to_vec(), b"aaaa".to_vec()];
    check(&pats, &vec![b'a'; 100]);
    let pats: Vec<Vec<u8>> = vec![b"abab".to_vec(), b"baba".to_vec()];
    check(&pats, &b"abababababababababababababababababababab".to_vec());
}

#[test]
fn identical_and_prefix_sharing_patterns() {
    let pats: Vec<Vec<u8>> =
        vec![b"dup".to_vec(), b"dup".to_vec(), b"dupli".to_vec(), b"duplicate".to_vec()];
    check(&pats, b"a duplicate dup and another duplicate, dupli");
}

#[test]
fn binary_bytes() {
    let pats: Vec<Vec<u8>> = vec![
        vec![0x00, 0x00, 0x01],
        vec![0xFF, 0xFE, 0xFD],
        vec![0x00, 0xFF, 0x00, 0xFF],
    ];
    let mut hay = vec![0u8; 200];
    hay[50] = 0x01; // plant 00 00 01 at 48
    hay[100] = 0xFF;
    hay[101] = 0xFE;
    hay[102] = 0xFD;
    hay[150] = 0xFF;
    hay[152] = 0xFF;
    check(&pats, &hay);
}

#[test]
fn min_len_one() {
    let pats: Vec<Vec<u8>> = vec![b"e".to_vec(), b"needle".to_vec()];
    check(&pats, b"weekend at the beach: we need a needle, evidently");
}

#[test]
fn forced_positions_still_exact() {
    let pats: Vec<Vec<u8>> = vec![b"httpAlpha".to_vec(), b"httpOmega".to_vec()];
    let hay = b"httpX httpAlpha http httpOmega httpAlphahttpOmega";
    for pos in [&[0u8][..], &[0, 1], &[0, 1, 2, 3], &[4, 8], &[3, 5, 8]] {
        let m = Builder::new().positions(pos).build(&pats).unwrap();
        assert_eq!(m.find_all(hay), naive::find_all(&pats, hay), "positions {:?}", pos);
        assert_eq!(m.find_all_scalar(hay), naive::find_all(&pats, hay), "positions {:?}", pos);
    }
}

#[test]
fn optimizer_beats_or_ties_prefix_on_shared_prefix_sets() {
    // 16 URL-style patterns sharing the common prefix "GET /api/" —
    // a contiguous-prefix filter (Teddy-style) is nearly blind here, while
    // sparse sampling can pick discriminating later bytes.
    let pats: Vec<Vec<u8>> = (0..16)
        .map(|i| format!("GET /api/{:02}x", i * 7 % 100).into_bytes())
        .collect();
    let corpus = b"GET /api/ GET /api/ GET /api/ the quick brown fox 0123456789".to_vec();
    let optimized = Builder::new().corpus_sample(&corpus).build(&pats).unwrap();
    let prefix = Builder::new()
        .corpus_sample(&corpus)
        .positions(&[0, 1, 2, 3])
        .build(&pats)
        .unwrap();
    assert!(
        optimized.expected_cost() <= prefix.expected_cost() * 1.0001,
        "optimized cost {} should not exceed prefix cost {}",
        optimized.expected_cost(),
        prefix.expected_cost()
    );
    // And on this construction it should be strictly, substantially better.
    assert!(
        optimized.expected_cost() < prefix.expected_cost() * 0.5,
        "expected a large win: optimized {} vs prefix {}",
        optimized.expected_cost(),
        prefix.expected_cost()
    );
    assert_ne!(optimized.sampled_positions(), &[0, 1, 2, 3]);
}

#[test]
fn fuzz_small_alphabet_dense_overlaps() {
    // Alphabet {a,b,c}: many accidental and overlapping occurrences.
    let mut rng = Rng(0xC0FFEE);
    for round in 0..60 {
        let n_pat = 1 + rng.below(12);
        let patterns: Vec<Vec<u8>> = (0..n_pat)
            .map(|_| {
                let len = 1 + rng.below(6);
                (0..len).map(|_| b'a' + rng.below(3) as u8).collect()
            })
            .collect();
        let hay_len = rng.below(400);
        let hay: Vec<u8> = (0..hay_len).map(|_| b'a' + rng.below(3) as u8).collect();
        let m = Sparrow::new(&patterns)
            .unwrap_or_else(|e| panic!("round {}: build failed: {}", round, e));
        let expect = naive::find_all(&patterns, &hay);
        assert_eq!(m.find_all_scalar(&hay), expect, "round {} scalar", round);
        assert_eq!(m.find_all(&hay), expect, "round {} simd", round);
    }
}

#[test]
fn fuzz_planted_patterns_full_alphabet() {
    let mut rng = Rng(0xDEADBEEF);
    for round in 0..40 {
        let n_pat = 1 + rng.below(40);
        let patterns: Vec<Vec<u8>> = (0..n_pat)
            .map(|_| {
                let len = 2 + rng.below(15);
                (0..len).map(|_| rng.next() as u8).collect()
            })
            .collect();
        let mut hay: Vec<u8> = (0..2000 + rng.below(2000)).map(|_| rng.next() as u8).collect();
        // Plant ~20 occurrences at random offsets (may overlap each other).
        for _ in 0..20 {
            let p = &patterns[rng.below(patterns.len())];
            if p.len() <= hay.len() {
                let at = rng.below(hay.len() - p.len() + 1);
                hay[at..at + p.len()].copy_from_slice(p);
            }
        }
        let m = Sparrow::new(&patterns).unwrap();
        let expect = naive::find_all(&patterns, &hay);
        assert_eq!(m.find_all_scalar(&hay), expect, "round {} scalar", round);
        assert_eq!(m.find_all(&hay), expect, "round {} simd", round);
        assert!(!expect.is_empty(), "round {} should have planted matches", round);
    }
}

#[test]
fn agrees_with_aho_corasick_overlapping() {
    use aho_corasick::AhoCorasick;
    let mut rng = Rng(0xFEEDF00D);
    for _ in 0..20 {
        let n_pat = 1 + rng.below(20);
        let patterns: Vec<Vec<u8>> = (0..n_pat)
            .map(|_| {
                let len = 2 + rng.below(8);
                (0..len).map(|_| b'a' + rng.below(5) as u8).collect()
            })
            .collect();
        let hay: Vec<u8> = (0..3000).map(|_| b'a' + rng.below(5) as u8).collect();
        let ac = AhoCorasick::new(&patterns).unwrap();
        let mut expect: Vec<Match> = ac
            .find_overlapping_iter(&hay)
            .map(|m| Match { start: m.start(), end: m.end(), pattern: m.pattern().as_usize() })
            .collect();
        expect.sort_unstable();
        // aho-corasick dedupes nothing and neither do we, but duplicate
        // patterns get distinct ids in both, so the sets are comparable.
        let m = Sparrow::new(&patterns).unwrap();
        assert_eq!(m.find_all(&hay), expect);
    }
}
