//! Differential tests for the dense (Shift-Or) lane and the two-prong
//! router: forced-dense matchers, routed matchers, and sparse-only matchers
//! must all report exactly the brute-force reference match set.

use sparrow::{naive, Builder, Match};

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

fn pats(v: &[&str]) -> Vec<Vec<u8>> {
    v.iter().map(|s| s.as_bytes().to_vec()).collect()
}

/// Check forced-dense, default-routed, and sparse-only builds against the
/// reference, under the given semantics.
fn check_opts(patterns: &[Vec<u8>], hay: &[u8], ci: bool, wc: Option<u8>) -> Vec<Match> {
    let expect = naive::find_all_with(patterns, hay, ci, wc);
    let base = || Builder::new().ascii_case_insensitive(ci).wildcard_byte(wc);
    let dense = base().force_dense(true).build(patterns).unwrap();
    assert!(dense.dense_lane().is_some(), "force_dense must engage for {:?}", patterns);
    assert_eq!(dense.cohort_count(), 0);
    assert_eq!(dense.find_all(hay), expect, "forced dense (patterns={:?})", patterns);
    let routed = base().corpus_sample(&hay[..hay.len().min(1 << 16)]).build(patterns).unwrap();
    assert_eq!(routed.find_all(hay), expect, "routed (patterns={:?})", patterns);
    let sparse = base().dense_lane(false).build(patterns).unwrap();
    assert!(sparse.dense_lane().is_none());
    assert_eq!(sparse.find_all(hay), expect, "sparse only (patterns={:?})", patterns);
    expect
}

fn check(patterns: &[Vec<u8>], hay: &[u8]) -> Vec<Match> {
    check_opts(patterns, hay, false, None)
}

fn english(rng: &mut Rng, n: usize) -> Vec<u8> {
    let words = [
        "the", "and", "for", "was", "hall", "rain", "wing", "none", "over", "rose", "orchard",
        "cluster", "harvest", "orchestra", "a", "an", "to", "of", "in", "it", "x", "zz", "theme",
        "therefore", "android", "thethe", "andand",
    ];
    let mut h = Vec::with_capacity(n + 16);
    while h.len() < n {
        h.extend_from_slice(words[rng.below(words.len())].as_bytes());
        if rng.below(4) != 0 {
            h.push(b' ');
        }
    }
    h.truncate(n);
    h
}

#[test]
fn dense_smoke() {
    let p = pats(&["the", "and", "for", "hall"]);
    let hay = b"the hall and the hall for all; thethe andand fora";
    let m = check(&p, hay);
    assert!(m.len() >= 8);
}

#[test]
fn dense_output_is_sorted_and_complete() {
    let mut rng = Rng(7);
    let hay = english(&mut rng, 300_000);
    let p = pats(&["the", "and", "a", "an", "thethe", "over", "rose", "x"]);
    let m = check(&p, &hay);
    assert!(m.windows(2).all(|w| w[0] < w[1]), "strictly sorted, no duplicates");
}

#[test]
fn dense_chunk_and_segment_boundaries() {
    // Plant a pattern straddling every 16 KB segment edge (every fourth one
    // is a 64 KB chunk edge), at a different offset across the edge each
    // time so every split position is exercised.
    let mut rng = Rng(11);
    let mut hay = english(&mut rng, 4 * 65536 + 1234);
    let p = pats(&["boundary", "edgecase!", "qq", "segmentsplit"]);
    let mut planted = 0;
    let mut b = 16384;
    let mut i = 0usize;
    while b + 16 < hay.len() {
        let pat = &p[i % p.len()];
        let start = b - (i % pat.len());
        hay[start..start + pat.len()].copy_from_slice(pat);
        planted += 1;
        b += 16384;
        i += 1;
    }
    let m = check(&p, &hay);
    assert!(m.len() >= planted && planted >= 16);
}

#[test]
fn dense_tiny_haystacks() {
    let p = pats(&["abc", "bcd", "a", "abcdefgh"]);
    for hay in [&b""[..], b"a", b"ab", b"abc", b"abcd", b"xabcdefgh", b"abcdefghabcdefgh"] {
        check(&p, hay);
    }
}

#[test]
fn dense_length_one_and_thirty_two() {
    let long = "0123456789abcdef0123456789abcdef"; // 32 bytes = one full lane
    let p = pats(&["z", long, "9a", "f0"]);
    let mut hay = Vec::new();
    for i in 0..2000 {
        hay.extend_from_slice(if i % 5 == 0 { long.as_bytes() } else { b"zzf09a" });
    }
    check(&p, &hay);
}

#[test]
fn dense_multi_lane_passes() {
    // > 128 pattern bytes forces 3-4 lanes = two passes per chunk, whose
    // sub-runs interleave in position and must still come out sorted.
    let mut rng = Rng(3);
    let hay = english(&mut rng, 200_000);
    let mut p: Vec<Vec<u8>> = Vec::new();
    // 40 patterns of length 4-6 drawn from the haystack: ~200 bytes.
    while p.len() < 40 {
        let s = rng.below(hay.len() - 8);
        let l = 4 + rng.below(3);
        let w = hay[s..s + l].to_vec();
        if !p.contains(&w) {
            p.push(w);
        }
    }
    let dense = Builder::new().force_dense(true).build(&p).unwrap();
    assert!(dense.dense_lane().unwrap().lane_count() >= 3);
    check(&p, &hay);
}

#[test]
fn dense_too_big_falls_back() {
    // 64 x 8 = 512 pattern bytes > 4 lanes: force_dense must not engage.
    let mut rng = Rng(5);
    let p: Vec<Vec<u8>> = (0..64).map(|_| (0..8).map(|_| rng.next() as u8).collect()).collect();
    let m = Builder::new().force_dense(true).build(&p).unwrap();
    assert!(m.dense_lane().is_none());
    assert!(m.cohort_count() >= 1);
    let p33 = vec![vec![b'a'; 33]];
    assert!(Builder::new().force_dense(true).build(&p33).unwrap().dense_lane().is_none());
}

#[test]
fn dense_case_insensitive_and_wildcard() {
    let mut rng = Rng(9);
    let mut hay = english(&mut rng, 100_000);
    for i in (0..hay.len()).step_by(97) {
        hay[i] = hay[i].to_ascii_uppercase();
    }
    let p = pats(&["the", "AND", "h?ll", "r?se", "?", "O?er"]);
    check_opts(&p, &hay, true, Some(b'?'));
    check_opts(&p, &hay, true, None);
    check_opts(&p, &hay, false, Some(b'?'));
}

#[test]
fn dense_streaming_matches_whole_buffer() {
    let mut rng = Rng(13);
    let hay = english(&mut rng, 150_000);
    let p = pats(&["the", "and", "over", "thethe", "a"]);
    let m = Builder::new().force_dense(true).build(&p).unwrap();
    let whole = m.find_all(&hay);
    let mut got = Vec::new();
    let mut s = m.stream();
    let mut off = 0;
    while off < hay.len() {
        let n = 1 + rng.below(5000);
        let end = (off + n).min(hay.len());
        got.extend(s.push(&hay[off..end]));
        off = end;
    }
    got.sort_unstable();
    let whole: Vec<(u64, u64, usize)> =
        whole.iter().map(|m| (m.start as u64, m.end as u64, m.pattern)).collect();
    let got: Vec<(u64, u64, usize)> = got.iter().map(|m| (m.start, m.end, m.pattern)).collect();
    assert_eq!(got, whole);
}

#[test]
fn router_splits_mixed_set_and_reports_it() {
    let mut rng = Rng(21);
    let hay = english(&mut rng, 1 << 20);
    let mut p = pats(&[
        "the", "and", "for", "was", "hall", "rain", "wing", "none", "over", "rose", "a", "an",
        "to", "of", "in", "it",
    ]);
    for i in 0..16 {
        p.push(format!("LONGSIGNATURE-{:02}-with-tail-bytes", i).into_bytes());
    }
    // Pin the routing to the model so the assertions below are deterministic.
    let m = Builder::new().corpus_sample(&hay[..65536]).timed_referee(false).build(&p).unwrap();
    assert!(!m.routing_decision().timed);
    let d = m.dense_lane().expect("short common words should route dense");
    assert_eq!(d.pattern_count(), 16);
    assert!(d.max_len() <= 4);
    assert_eq!(m.cohort_count(), 1, "long signatures stay sparse");
    assert!(m.min_pattern_len() == 1);
    assert_eq!(m.find_all(&hay), naive::find_all(&p, &hay));
    let u = m.memory_usage();
    assert!(u.scalar_tables >= d.lane_count() * 2048);
}

#[test]
fn fuzz_dense_small_alphabet() {
    let mut rng = Rng(0xD15E);
    for round in 0..200 {
        let alpha = 2 + rng.below(3);
        let np = 1 + rng.below(12);
        let p: Vec<Vec<u8>> = (0..np)
            .map(|_| (0..1 + rng.below(6)).map(|_| b'a' + rng.below(alpha) as u8).collect())
            .collect();
        let n = rng.below(if round % 10 == 0 { 70_000 } else { 400 });
        let hay: Vec<u8> = (0..n).map(|_| b'a' + rng.below(alpha) as u8).collect();
        check(&p, &hay);
    }
}

#[test]
fn timed_referee_reports_measurements_and_is_correct() {
    let mut rng = Rng(33);
    let hay = english(&mut rng, 1 << 20);
    let mut p = pats(&["the", "and", "for", "was", "hall", "rain", "wing", "none", "over", "rose"]);
    for i in 0..8 {
        p.push(format!("LONGSIGNATURE-{:02}-with-tail-bytes", i).into_bytes());
    }
    let m = Builder::new().corpus_sample(&hay[..65536]).build(&p).unwrap();
    let d = m.routing_decision();
    assert!(d.timed, "64 KB corpus with several candidates must be timed");
    assert!(d.candidates.len() >= 2);
    let timed: Vec<_> = d.candidates.iter().filter(|c| c.measured_ns_per_byte.is_some()).collect();
    assert!(timed.len() >= 2, "baseline plus at least one split must be timed");
    assert!(d.candidates.iter().any(|c| c.dense_patterns == 0 && c.measured_ns_per_byte.is_some()));
    assert!(d.candidates[d.chosen].measured_ns_per_byte.is_some());
    let best = timed.iter().map(|c| c.measured_ns_per_byte.unwrap()).fold(f64::INFINITY, f64::min);
    assert_eq!(d.candidates[d.chosen].measured_ns_per_byte.unwrap(), best);
    assert_eq!(m.find_all(&hay), naive::find_all(&p, &hay));

    // Too-small corpus: model decides, nothing is timed.
    let m2 = Builder::new().corpus_sample(&hay[..4096]).build(&p).unwrap();
    assert!(!m2.routing_decision().timed);
    assert!(m2.routing_decision().candidates.iter().all(|c| c.measured_ns_per_byte.is_none()));
    assert_eq!(m2.find_all(&hay), naive::find_all(&p, &hay));

    // Referee off: model decides.
    let m3 = Builder::new().corpus_sample(&hay[..65536]).timed_referee(false).build(&p).unwrap();
    assert!(!m3.routing_decision().timed);
    assert_eq!(m3.routing_decision().chosen, 0, "model ranking puts its best first");
}

#[test]
fn build_time_with_referee_is_bounded() {
    let mut rng = Rng(44);
    let hay = english(&mut rng, 1 << 18);
    let p = pats(&["the", "and", "for", "orchard", "cluster", "harvest", "orchestra", "over"]);
    let t = std::time::Instant::now();
    let _ = Builder::new().corpus_sample(&hay[..65536]).build(&p).unwrap();
    let dt = t.elapsed();
    assert!(dt.as_millis() < 500, "build took {:?}", dt);
}

#[test]
fn scan_with_and_count_match_find_all() {
    let mut rng = Rng(0xCAB);
    for round in 0..30 {
        let hay = english(&mut rng, if round % 6 == 0 { 200_000 } else { 700 });
        let np = 1 + rng.below(10);
        let p: Vec<Vec<u8>> = (0..np)
            .map(|_| {
                let s = rng.below(hay.len().saturating_sub(9).max(1));
                let l = 1 + rng.below(8);
                hay[s..(s + l).min(hay.len())].to_vec()
            })
            .filter(|w| !w.is_empty())
            .collect();
        if p.is_empty() {
            continue;
        }
        for m in [
            Builder::new().corpus_sample(&hay[..hay.len().min(1 << 16)]).build(&p).unwrap(),
            Builder::new().force_dense(true).build(&p).unwrap(),
            Builder::new().dense_lane(false).build(&p).unwrap(),
        ] {
            let expect = m.find_all(&hay);
            assert_eq!(m.count_all(&hay), expect.len());
            let mut got = Vec::new();
            m.scan_with(&hay, |mm| got.push(mm));
            got.sort_unstable();
            assert_eq!(got, expect);
        }
    }
}

#[test]
fn streaming_tiny_chunks_fuzz() {
    let mut rng = Rng(0x57F);
    for round in 0..40 {
        let n = 300 + rng.below(3000);
        let hay = english(&mut rng, n);
        let p = pats(&["the", "and", "thethe", "a", "over", "orchestra"]);
        let m = if round % 2 == 0 {
            Builder::new().force_dense(true).build(&p).unwrap()
        } else {
            Builder::new().dense_lane(false).build(&p).unwrap()
        };
        let whole: Vec<(u64, u64, usize)> = m
            .find_all(&hay)
            .iter()
            .map(|x| (x.start as u64, x.end as u64, x.pattern))
            .collect();
        let mut s = m.stream();
        let mut got = Vec::new();
        let mut off = 0;
        while off < hay.len() {
            // Chunk sizes 1..7 bytes: every boundary/stitch edge case.
            let n = 1 + rng.below(7);
            let end = (off + n).min(hay.len());
            for sm in s.push(&hay[off..end]) {
                got.push((sm.start, sm.end, sm.pattern));
            }
            off = end;
        }
        assert_eq!(got.len(), whole.len(), "round {round}");
        got.sort_unstable();
        assert_eq!(got, whole, "round {round}");
    }
}

#[test]
fn neon_pair_kernel_matches_scalar() {
    let mut rng = Rng(0x9E09);
    for round in 0..60 {
        let n = if round % 8 == 0 { 200_000 } else { 200 + rng.below(2000) };
        let hay = english(&mut rng, n);
        // 9..=20 patterns so two lanes engage; lengths 1..=12.
        let np = 9 + rng.below(12);
        let p: Vec<Vec<u8>> = (0..np)
            .map(|_| {
                let s = rng.below(hay.len().saturating_sub(13).max(1));
                let l = 1 + rng.below(12);
                hay[s..(s + l).min(hay.len())].to_vec()
            })
            .filter(|w| !w.is_empty())
            .collect();
        let fast = Builder::new().force_dense(true).build(&p).unwrap();
        let slow = Builder::new().force_dense(true).dense_scalar_kernel(true).build(&p).unwrap();
        if fast.dense_lane().is_none() {
            continue;
        }
        assert_eq!(fast.find_all(&hay), slow.find_all(&hay), "round {round}");
        assert_eq!(fast.count_all(&hay), slow.count_all(&hay), "round {round}");
    }
}
