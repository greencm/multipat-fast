//! Hashed-verification and large-set tests: bucket occupancy far above
//! HASH_MIN, exercising the fingerprint region, the non-exact scan tail,
//! and id remapping under two-prong routing.

use sparrow::{naive, Builder};

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

fn word(rng: &mut Rng, len: usize) -> Vec<u8> {
    (0..len).map(|_| b'a' + rng.below(6) as u8).collect()
}

/// 800 patterns over a 6-letter alphabet: every bucket holds dozens of
/// entries, so the fingerprint region carries the verification load.
#[test]
fn hashed_buckets_exact_patterns() {
    let mut rng = Rng(0x5CA1E);
    let mut pats: Vec<Vec<u8>> = Vec::new();
    while pats.len() < 800 {
        let l = 5 + rng.below(8); let w = word(&mut rng, l);
        if !pats.contains(&w) {
            pats.push(w);
        }
    }
    let mut hay: Vec<u8> = (0..300_000).map(|_| b'a' + rng.below(6) as u8).collect();
    for i in 0..500 {
        // plant occurrences, including back-to-back and overlapping-ish
        let p = &pats[rng.below(pats.len())];
        let at = rng.below(hay.len() - p.len());
        hay[at..at + p.len()].copy_from_slice(p);
        if i % 7 == 0 && at + 2 * p.len() < hay.len() {
            let q = p.clone();
            hay[at + p.len()..at + 2 * p.len()].copy_from_slice(&q);
        }
    }
    let expect = naive::find_all(&pats, &hay);
    let m = Builder::new().corpus_sample(&hay[..65536]).build(&pats).unwrap();
    assert_eq!(m.find_all(&hay), expect);
    let sp = Builder::new().dense_lane(false).build(&pats).unwrap();
    assert_eq!(sp.find_all(&hay), expect);
    // The point of the test: hashed regions must actually be in play.
    let u = sp.memory_usage();
    assert!(u.buckets > 800 * 8, "expected hashed offset tables in footprint");
}

/// Case-insensitive large set: sampled positions are case pairs, so every
/// entry lands in the scan tail — the fallback must stay exact.
#[test]
fn hashed_fallback_case_insensitive() {
    let mut rng = Rng(0xFACE);
    let mut pats: Vec<Vec<u8>> = Vec::new();
    while pats.len() < 200 {
        let l = 5 + rng.below(5); let w = word(&mut rng, l);
        if !pats.contains(&w) {
            pats.push(w);
        }
    }
    let mut hay: Vec<u8> = (0..120_000).map(|_| b'a' + rng.below(6) as u8).collect();
    for i in (0..hay.len()).step_by(3) {
        hay[i] = hay[i].to_ascii_uppercase();
    }
    let expect = naive::find_all_with(&pats, &hay, true, None);
    let m = Builder::new().ascii_case_insensitive(true).dense_lane(false).build(&pats).unwrap();
    assert_eq!(m.find_all(&hay), expect);
}

/// Two-prong routing with a big sparse remainder: hashed-entry ids must be
/// remapped to the full pattern index space.
#[test]
fn hashed_id_remap_under_routing() {
    let mut rng = Rng(0xBEEF);
    let mut pats: Vec<Vec<u8>> = vec![b"ab".to_vec(), b"ba".to_vec(), b"aa".to_vec()];
    while pats.len() < 400 {
        let l = 6 + rng.below(6); let w = word(&mut rng, l);
        if !pats.contains(&w) {
            pats.push(w);
        }
    }
    let hay: Vec<u8> = (0..200_000).map(|_| b'a' + rng.below(6) as u8).collect();
    let expect = naive::find_all(&pats, &hay);
    let m = Builder::new()
        .corpus_sample(&hay[..65536])
        .timed_referee(false)
        .build(&pats)
        .unwrap();
    assert_eq!(m.find_all(&hay), expect);
}

/// Streaming over a hashed-bucket build.
#[test]
fn hashed_streaming() {
    let mut rng = Rng(0x51EA);
    let mut pats: Vec<Vec<u8>> = Vec::new();
    while pats.len() < 300 {
        let l = 5 + rng.below(6); let w = word(&mut rng, l);
        if !pats.contains(&w) {
            pats.push(w);
        }
    }
    let hay: Vec<u8> = (0..150_000).map(|_| b'a' + rng.below(6) as u8).collect();
    let m = Builder::new().dense_lane(false).build(&pats).unwrap();
    let whole = m.find_all(&hay);
    let mut s = m.stream();
    let mut got = Vec::new();
    let mut off = 0;
    while off < hay.len() {
        let end = (off + 1 + rng.below(7000)).min(hay.len());
        got.extend(s.push(&hay[off..end]));
        off = end;
    }
    assert_eq!(got.len(), whole.len());
    for (g, w) in got.iter().zip(whole.iter()) {
        assert_eq!((g.start as usize, g.end as usize, g.pattern), (w.start, w.end, w.pattern));
    }
}
