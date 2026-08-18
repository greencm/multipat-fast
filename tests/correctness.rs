//! Differential correctness tests: every SPARROW engine (scalar, AVX2,
//! AVX-512) must report exactly the same match set as the brute-force
//! reference on planted-occurrence workloads, adversarial edges, random
//! fuzzing, and the optional matching semantics (case-insensitive,
//! wildcards, streaming, leftmost).

use sparrow::{naive, BuildError, Builder, Engine, Match, Sparrow, StreamMatch};

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

const ENGINES: &[Engine] = &[Engine::Scalar, Engine::Avx2, Engine::Avx512];

/// Build one matcher per available engine from the same builder config.
fn build_engines<P: AsRef<[u8]> + Clone>(base: &Builder, patterns: &[P]) -> Vec<(Engine, Sparrow)> {
    let mut v = Vec::new();
    for &e in ENGINES {
        match base.clone().force_engine(e).build(patterns.to_vec()) {
            Ok(m) => v.push((e, m)),
            Err(BuildError::EngineUnavailable) => continue,
            Err(err) => panic!("build failed for {:?}: {}", e, err),
        }
    }
    assert!(!v.is_empty());
    v
}

fn check_opts(patterns: &[Vec<u8>], hay: &[u8], ci: bool, wc: Option<u8>) {
    let expect = naive::find_all_with(patterns, hay, ci, wc);
    let base = Builder::new()
        .ascii_case_insensitive(ci)
        .wildcard_byte(wc)
        .corpus_scoring(true);
    for (engine, m) in build_engines(&base, patterns) {
        assert_eq!(m.find_all(hay), expect, "engine {:?} (patterns={:?})", engine, patterns);
    }
}

fn check(patterns: &[Vec<u8>], hay: &[u8]) {
    check_opts(patterns, hay, false, None);
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
    // Match at offset 0, at the very end, overlapping, across SIMD block
    // boundaries (32/64), and around the scalar-prelude/kernel thresholds.
    check(&pats, b"ab");
    check(&pats, b"abcdab");
    check(&pats, b"");
    check(&pats, b"a"); // shorter than every pattern
    for len in [31, 32, 33, 63, 64, 65, 95, 96, 97, 128] {
        let mut hay = vec![b'x'; len];
        // Plant "abcd" straddling each interesting boundary.
        for at in [0usize, 30, 31, 62, 63, 94, 95] {
            if at + 4 <= len {
                hay[at..at + 4].copy_from_slice(b"abcd");
            }
        }
        if len >= 2 {
            hay[len - 2..].copy_from_slice(b"ab");
        }
        check(&pats, &hay);
    }
    check(&pats, &vec![b'a'; 200]);
}

#[test]
fn self_overlapping_and_repeats() {
    let pats: Vec<Vec<u8>> = vec![b"aaa".to_vec(), b"aaaa".to_vec()];
    check(&pats, &vec![b'a'; 300]);
    let pats: Vec<Vec<u8>> = vec![b"abab".to_vec(), b"baba".to_vec()];
    check(&pats, &b"abab".repeat(40));
}

#[test]
fn identical_and_prefix_sharing_patterns() {
    let pats: Vec<Vec<u8>> =
        vec![b"dup".to_vec(), b"dup".to_vec(), b"dupli".to_vec(), b"duplicate".to_vec()];
    check(&pats, b"a duplicate dup and another duplicate, dupli xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx dup");
}

#[test]
fn binary_bytes() {
    let pats: Vec<Vec<u8>> = vec![
        vec![0x00, 0x00, 0x01],
        vec![0xFF, 0xFE, 0xFD],
        vec![0x00, 0xFF, 0x00, 0xFF],
    ];
    let mut hay = vec![0u8; 300];
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
    check(&pats, b"weekend at the beach: we need a needle, evidently -- and then some more text to cross 96 bytes eee");
}

#[test]
fn case_insensitive_matching() {
    let pats: Vec<Vec<u8>> = vec![b"NeEdLe".to_vec(), b"HAY".to_vec(), b"mix3d".to_vec()];
    let hay = b"a needle in the HAY, a NEEDLE in the hay, MIX3D and Mix3d and mIx3D xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx";
    check_opts(&pats, hay, true, None);
    // Sanity: case-sensitive build on the same input finds strictly fewer.
    let ci = naive::find_all_with(&pats, hay, true, None);
    let cs = naive::find_all_with(&pats, hay, false, None);
    assert!(ci.len() > cs.len());
}

#[test]
fn wildcard_matching() {
    let pats: Vec<Vec<u8>> = vec![
        b"a?c".to_vec(),
        b"?bc?".to_vec(),
        b"x??y".to_vec(),
        b"???".to_vec(), // all-wildcard: matches every 3-byte window
    ];
    let mut rng = Rng(0xABCD);
    for _ in 0..10 {
        let hay: Vec<u8> = (0..200).map(|_| b'a' + rng.below(4) as u8).collect();
        check_opts(&pats, &hay, false, Some(b'?'));
    }
    let hay = b"abc xbcy xzzy aXc xy";
    check_opts(&pats, hay, false, Some(b'?'));
}

#[test]
fn wildcard_and_case_insensitive_combined() {
    let pats: Vec<Vec<u8>> = vec![b"A?c".to_vec(), b"GET /?PI".to_vec()];
    let hay = b"aXC and get /aPI and GET /API abc AbC xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx";
    check_opts(&pats, hay, true, Some(b'?'));
}

#[test]
fn mixed_length_cohorts_correct() {
    // Diverse lengths so the cohort arbitration is exercised (whichever way
    // the model decides, the match set must be identical).
    let mut rng = Rng(0x1234);
    let mut pats: Vec<Vec<u8>> = Vec::new();
    for len in [2usize, 3, 5, 7, 9, 12, 20, 33, 48] {
        for _ in 0..4 {
            pats.push((0..len).map(|_| b'a' + rng.below(6) as u8).collect());
        }
    }
    for _ in 0..8 {
        let mut hay: Vec<u8> = (0..3000).map(|_| b'a' + rng.below(6) as u8).collect();
        for _ in 0..15 {
            let p = &pats[rng.below(pats.len())];
            let at = rng.below(hay.len() - p.len() + 1);
            hay[at..at + p.len()].copy_from_slice(p);
        }
        check(&pats, &hay);
    }
    let m = Sparrow::new(&pats).unwrap();
    assert!(m.cohort_count() >= 1);
}

#[test]
fn streaming_equals_whole_buffer() {
    let mut rng = Rng(0x57AE);
    let pats: Vec<Vec<u8>> =
        vec![b"stream".to_vec(), b"ream".to_vec(), b"boundary".to_vec(), b"aa".to_vec()];
    for _ in 0..20 {
        let mut hay: Vec<u8> = (0..2000).map(|_| b'a' + rng.below(8) as u8).collect();
        for _ in 0..25 {
            let p = &pats[rng.below(pats.len())];
            let at = rng.below(hay.len() - p.len() + 1);
            hay[at..at + p.len()].copy_from_slice(p);
        }
        let m = Sparrow::new(&pats).unwrap();
        let expect: Vec<StreamMatch> = m
            .find_all(&hay)
            .into_iter()
            .map(|x| StreamMatch { start: x.start as u64, end: x.end as u64, pattern: x.pattern })
            .collect();
        let mut got = Vec::new();
        let mut sc = m.stream();
        let mut off = 0usize;
        while off < hay.len() {
            // Adversarial chunk sizes, including empty and size-1 chunks.
            let sz = match rng.below(5) {
                0 => 0,
                1 => 1,
                2 => 1 + rng.below(7),
                _ => 1 + rng.below(200),
            };
            let end = (off + sz).min(hay.len());
            got.extend(sc.push(&hay[off..end]));
            off = end;
        }
        got.sort_unstable();
        assert_eq!(got, expect);
    }
}

#[test]
fn leftmost_nonoverlapping_matches_aho_corasick() {
    use aho_corasick::{AhoCorasick, MatchKind};
    let mut rng = Rng(0x1EF7);
    for _ in 0..25 {
        let n_pat = 1 + rng.below(12);
        let patterns: Vec<Vec<u8>> = (0..n_pat)
            .map(|_| {
                let len = 1 + rng.below(6);
                (0..len).map(|_| b'a' + rng.below(3) as u8).collect()
            })
            .collect();
        let hay: Vec<u8> = (0..1500).map(|_| b'a' + rng.below(3) as u8).collect();
        let ac = AhoCorasick::builder()
            .match_kind(MatchKind::LeftmostFirst)
            .build(&patterns)
            .unwrap();
        let expect: Vec<Match> = ac
            .find_iter(&hay)
            .map(|m| Match { start: m.start(), end: m.end(), pattern: m.pattern().as_usize() })
            .collect();
        let m = Sparrow::new(&patterns).unwrap();
        assert_eq!(m.find_leftmost_nonoverlapping(&hay), expect, "patterns={:?}", patterns);
    }
}

#[test]
fn forced_positions_still_exact() {
    let pats: Vec<Vec<u8>> = vec![b"httpAlpha".to_vec(), b"httpOmega".to_vec()];
    let hay = b"httpX httpAlpha http httpOmega httpAlphahttpOmega xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx";
    for pos in [&[0u8][..], &[0, 1], &[0, 1, 2, 3], &[4, 8], &[3, 5, 8]] {
        let base = Builder::new().positions(pos).corpus_scoring(true);
        let expect = naive::find_all(&pats, hay);
        for (engine, m) in build_engines(&base, &pats) {
            assert_eq!(m.find_all(hay), expect, "engine {:?} positions {:?}", engine, pos);
        }
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
    assert_ne!(optimized.sampled_positions()[0], vec![0, 1, 2, 3]);
}

#[test]
fn fuzz_small_alphabet_dense_overlaps() {
    // Alphabet {a,b,c}: many accidental and overlapping occurrences.
    let mut rng = Rng(0xC0FFEE);
    for _round in 0..40 {
        let n_pat = 1 + rng.below(12);
        let patterns: Vec<Vec<u8>> = (0..n_pat)
            .map(|_| {
                let len = 1 + rng.below(6);
                (0..len).map(|_| b'a' + rng.below(3) as u8).collect()
            })
            .collect();
        let hay_len = rng.below(500);
        let hay: Vec<u8> = (0..hay_len).map(|_| b'a' + rng.below(3) as u8).collect();
        check(&patterns, &hay);
    }
}

#[test]
fn fuzz_planted_patterns_full_alphabet() {
    let mut rng = Rng(0xDEADBEEF);
    for round in 0..25 {
        let n_pat = 1 + rng.below(40);
        let patterns: Vec<Vec<u8>> = (0..n_pat)
            .map(|_| {
                let len = 2 + rng.below(15);
                (0..len).map(|_| rng.next() as u8).collect()
            })
            .collect();
        let mut hay: Vec<u8> = (0..2000 + rng.below(2000)).map(|_| rng.next() as u8).collect();
        for _ in 0..20 {
            let p = &patterns[rng.below(patterns.len())];
            if p.len() <= hay.len() {
                let at = rng.below(hay.len() - p.len() + 1);
                hay[at..at + p.len()].copy_from_slice(p);
            }
        }
        let expect = naive::find_all(&patterns, &hay);
        assert!(!expect.is_empty(), "round {} should have planted matches", round);
        check(&patterns, &hay);
    }
}

#[test]
fn corpus_scored_build_is_correct() {
    // Most tests disable the Markov re-scoring for speed; make sure the
    // default (Markov on) build matches too.
    let pats: Vec<Vec<u8>> = ["orchard", "cluster", "harvest", "GET /api/v1/users"]
        .iter()
        .map(|s| s.as_bytes().to_vec())
        .collect();
    let hay = b"the orchard by the cluster: GET /api/v1/users harvest orchard xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx";
    let expect = naive::find_all(&pats, hay);
    let m = Sparrow::new(&pats).unwrap();
    assert_eq!(m.find_all(hay), expect);
    assert_eq!(m.find_all_scalar(hay), expect);
}

#[test]
fn large_pattern_set_two_planes() {
    // >= 32 patterns makes the 16-bucket (two-plane) mode eligible; whether
    // or not the model picks it, results must be identical. Force a build
    // where planes=2 is likely by using many common-ish patterns.
    let mut rng = Rng(0xB16B00B5);
    let words: Vec<Vec<u8>> = (0..80)
        .map(|i| {
            let len = 4 + (i % 5);
            (0..len).map(|_| b'a' + rng.below(10) as u8).collect()
        })
        .collect();
    let mut hay: Vec<u8> = (0..4000).map(|_| b'a' + rng.below(10) as u8).collect();
    for _ in 0..30 {
        let p = &words[rng.below(words.len())];
        let at = rng.below(hay.len() - p.len() + 1);
        hay[at..at + p.len()].copy_from_slice(p);
    }
    check(&words, &hay);
    let m = Builder::new().corpus_scoring(true).build(&words).unwrap();
    let buckets = m.bucket_counts();
    assert!(buckets.iter().all(|&b| b == 8 || b == 16));
}
