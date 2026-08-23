//! Leftmost-first vs overlapping on a pattern set that overlaps itself
//! heavily ("a" / "an" / "and" / "android", "the" / "theme" / "therefore").
use aho_corasick::{AhoCorasick, AhoCorasickKind, MatchKind};
use sparrow::Builder;
use std::time::Instant;

fn best<F: FnMut() -> usize>(mut f: F) -> (f64, usize) {
    let mut b = f64::INFINITY;
    let mut n = 0;
    f();
    for _ in 0..5 {
        let t = Instant::now();
        n = f();
        b = b.min(t.elapsed().as_secs_f64());
    }
    (b, n)
}

fn main() {
    let words = ["the", "and", "for", "a", "an", "theme", "therefore", "android", "another", "forest", "hall", "over", "x", "it", "in", "into"];
    let mut seed = 7u64;
    let n = 16 << 20;
    let mut hay = Vec::with_capacity(n + 16);
    while hay.len() < n {
        seed = seed.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = seed;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z ^= z >> 31;
        hay.extend_from_slice(words[(z % words.len() as u64) as usize].as_bytes());
        if z % 3 != 0 {
            hay.push(b' ');
        }
    }
    hay.truncate(n);
    let gb = n as f64 / (1u64 << 30) as f64;
    let pats = ["the", "theme", "therefore", "a", "an", "and", "android", "another", "in", "into", "for", "forest"];
    let m = Builder::new().corpus_sample(&hay[..65536]).build(pats).unwrap();
    let d = m.routing_decision();
    println!("routing: dense patterns = {}", d.candidates[d.chosen].dense_patterns);
    let (t, k) = best(|| m.find_all(&hay).len());
    println!("find_all (overlapping)        {:6.3} GB/s  {k} matches", gb / t);
    let (t, k) = best(|| m.find_leftmost_nonoverlapping(&hay).len());
    println!("find_leftmost_nonoverlapping  {:6.3} GB/s  {k} matches", gb / t);
    let (t, k) = best(|| m.find_leftmost(&hay).len());
    println!("find_leftmost (native)        {:6.3} GB/s  {k} matches", gb / t);
    let (t, k) = best(|| m.find_leftmost_longest(&hay).len());
    println!("find_leftmost_longest         {:6.3} GB/s  {k} matches", gb / t);
    let ac = AhoCorasick::builder().kind(Some(AhoCorasickKind::DFA)).match_kind(MatchKind::LeftmostFirst).build(pats).unwrap();
    let (t, k) = best(|| ac.find_iter(&hay).count());
    println!("aho-corasick DFA leftmost     {:6.3} GB/s  {k} matches", gb / t);
    if let Some(pk) = aho_corasick::packed::Config::new().builder().extend(pats).build() {
        let (t, k) = best(|| pk.find_iter(&hay).count());
        println!("packed Teddy leftmost         {:6.3} GB/s  {k} matches", gb / t);
    }
}
