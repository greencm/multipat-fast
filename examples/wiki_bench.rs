//! Real-corpus benchmark: Simple English Wikipedia (XML dump slice).
//!
//! Pattern sets are derived deterministically from the corpus itself:
//! common words (match-dense), mid-frequency words, rare words, MediaWiki
//! markup literals with shared prefixes, and a mixed short+long set. The
//! haystack is the first 32 MB of the file; the corpus sample handed to
//! the builder is held out (taken from the second half, never scanned).
//!
//! Run: cargo run --release --example wiki_bench
//!
//! The data file is local-only (gitignored). Regenerate with:
//!   mkdir -p bench_data && cd bench_data
//!   curl -sL "https://dumps.wikimedia.org/simplewiki/latest/simplewiki-latest-pages-articles-multistream.xml.bz2" \
//!        -o simplewiki.xml.bz2 --range 0-50000000
//!   bzip2 -dc simplewiki.xml.bz2 2>/dev/null | head -c 64000000 > simplewiki-64MB.xml
//!   rm simplewiki.xml.bz2

use aho_corasick::{AhoCorasick, AhoCorasickKind};
use sparrow::Builder;
use std::collections::HashMap;
use std::time::Instant;

const MB: usize = 1 << 20;
const DATA: &str = "bench_data/simplewiki-64MB.xml";

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

fn time_best_of<F: FnMut() -> usize>(mut f: F) -> (f64, usize) {
    let mut best = f64::INFINITY;
    let mut count = 0;
    f(); // warmup
    for _ in 0..5 {
        let t = Instant::now();
        count = f();
        best = best.min(t.elapsed().as_secs_f64());
    }
    (best, count)
}

/// Distinct ASCII-alphabetic tokens of the slice with their counts, in a
/// deterministic (alphabetical) order.
fn word_counts(text: &[u8]) -> Vec<(Vec<u8>, u32)> {
    let mut map: HashMap<&[u8], u32> = HashMap::new();
    let mut i = 0;
    while i < text.len() {
        if text[i].is_ascii_alphabetic() {
            let s = i;
            while i < text.len() && text[i].is_ascii_alphabetic() {
                i += 1;
            }
            *map.entry(&text[s..i]).or_insert(0) += 1;
        } else {
            i += 1;
        }
    }
    let mut v: Vec<(Vec<u8>, u32)> = map.into_iter().map(|(w, c)| (w.to_vec(), c)).collect();
    v.sort_unstable();
    v
}

/// `n` items spread evenly over `pool` (which must be deterministic).
fn spread(pool: &[Vec<u8>], n: usize) -> Vec<Vec<u8>> {
    assert!(pool.len() >= n, "pool {} < wanted {}", pool.len(), n);
    (0..n).map(|i| pool[i * pool.len() / n].clone()).collect()
}

fn run_set(name: &str, patterns: &[Vec<u8>], hay: &[u8], corpus: &[u8], leftmost: bool) {
    println!("\n=== {} ===", name);
    println!(
        "patterns: {} (len {}..{}), haystack: {} MB",
        patterns.len(),
        patterns.iter().map(|p| p.len()).min().unwrap(),
        patterns.iter().map(|p| p.len()).max().unwrap(),
        hay.len() / MB
    );
    let sp = Builder::new().corpus_sample(corpus).build(patterns).unwrap();
    let sp_sparse = Builder::new().corpus_sample(corpus).dense_lane(false).build(patterns).unwrap();

    let d = sp.routing_decision();
    let chosen = &d.candidates[d.chosen];
    println!(
        "routing ({}): dense:{}{} of {} candidates | positions {:?}, cohorts {}",
        if d.timed { "timed referee" } else { "model" },
        chosen.dense_patterns,
        chosen.measured_ns_per_byte.map_or(String::new(), |ns| format!(" ({:.2} ns/B measured)", ns)),
        d.candidates.len(),
        sp.sampled_positions(),
        sp.cohort_count(),
    );
    if let Some((i, c)) = d
        .candidates
        .iter()
        .enumerate()
        .filter(|(_, c)| c.measured_ns_per_byte.is_some())
        .min_by(|a, b| a.1.measured_ns_per_byte.partial_cmp(&b.1.measured_ns_per_byte).unwrap())
    {
        if i != d.chosen {
            println!(
                "  NOTE: mis-route — fastest measured candidate was dense:{} at {:.2} ns/B",
                c.dense_patterns,
                c.measured_ns_per_byte.unwrap()
            );
        }
    }

    let ac_dfa =
        AhoCorasick::builder().kind(Some(AhoCorasickKind::DFA)).build(patterns).unwrap();
    let packed = aho_corasick::packed::Config::new().builder().extend(patterns).build();
    let gb = hay.len() as f64 / (1u64 << 30) as f64;

    let (t, n_sp) = time_best_of(|| sp.find_all(hay).len());
    println!("  sparrow                  {:8.3} GB/s   {} matches", gb / t, n_sp);
    let (t, n_so) = time_best_of(|| sp_sparse.find_all(hay).len());
    println!("  sparrow (sparse only)    {:8.3} GB/s   {} matches", gb / t, n_so);
    let (t, n_cnt) = time_best_of(|| sp.count_all(hay));
    println!("  sparrow count_all        {:8.3} GB/s   {} matches", gb / t, n_cnt);
    assert_eq!(n_sp, n_cnt, "count_all must agree");
    let (t, n_ac) = time_best_of(|| ac_dfa.find_overlapping_iter(hay).count());
    println!("  aho-corasick DFA (ovlp)  {:8.3} GB/s   {} matches", gb / t, n_ac);
    assert_eq!(n_sp, n_so, "two-prong and sparse-only must agree");
    assert_eq!(n_sp, n_ac, "sparrow must agree with AC overlapping");
    if let Some(ref pk) = packed {
        let (t, n_pk) = time_best_of(|| pk.find_iter(hay).count());
        println!("  aho-corasick Teddy(pkd)  {:8.3} GB/s   {} matches (leftmost)", gb / t, n_pk);
    }

    if leftmost {
        println!("  -- leftmost-first --");
        let (t, n_lm) = time_best_of(|| sp.find_leftmost(hay).len());
        println!("  sparrow find_leftmost    {:8.3} GB/s   {} matches", gb / t, n_lm);
        let ac_lf = AhoCorasick::builder()
            .kind(Some(AhoCorasickKind::DFA))
            .match_kind(aho_corasick::MatchKind::LeftmostFirst)
            .build(patterns)
            .unwrap();
        let (t, n_al) = time_best_of(|| ac_lf.find_iter(hay).count());
        println!("  aho-corasick DFA (lmf)   {:8.3} GB/s   {} matches", gb / t, n_al);
        assert_eq!(n_lm, n_al, "sparrow leftmost must agree with AC leftmost-first");
    }
}

fn main() {
    let text = std::fs::read(DATA).unwrap_or_else(|e| {
        eprintln!("could not read {DATA}: {e}");
        eprintln!("regenerate it with the commands in the header of examples/wiki_bench.rs");
        std::process::exit(1);
    });
    assert!(text.len() >= 48 * MB, "expected ~64 MB, got {} bytes", text.len());
    let half = text.len() / 2;
    let hay = &text[..half.min(32 * MB)];
    let held_out = &text[half..];
    let corpus = &held_out[..256 * 1024];
    println!(
        "corpus: {} ({} MB haystack, {} KB held-out builder sample)",
        DATA,
        hay.len() / MB,
        corpus.len() / 1024
    );

    // Word statistics from the held-out half only.
    let words = word_counts(held_out);
    let mut rng = Rng(0x5EED);

    // (a) 16 most common words of len >= 3 (match-dense).
    let mut by_freq: Vec<&(Vec<u8>, u32)> = words.iter().filter(|(w, _)| w.len() >= 3).collect();
    by_freq.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    let common: Vec<Vec<u8>> = by_freq.iter().take(16).map(|(w, _)| w.clone()).collect();

    // (b) 64 mid-frequency words, len 6-12.
    let mid_pool: Vec<Vec<u8>> = by_freq
        .iter()
        .filter(|(w, c)| (6..=12).contains(&w.len()) && (50..=2000).contains(c))
        .map(|(w, _)| w.clone())
        .collect();
    let mid = spread(&mid_pool, 64);

    // (c) 256 rare words, len 8-20 (seen once or twice in the held-out half).
    let rare_pool: Vec<Vec<u8>> = words
        .iter()
        .filter(|(w, c)| (8..=20).contains(&w.len()) && *c <= 2)
        .map(|(w, _)| w.clone())
        .collect();
    let rare = spread(&rare_pool, 256);

    // (d) MediaWiki markup literals sharing prefixes; keep those that occur.
    let markup_candidates: [&str; 30] = [
        "[[Category:", "[[File:", "[[Image:", "[[wikt:", "[[en:", "[[de:", "[[fr:", "[[es:",
        "{{Infobox", "{{cite web", "{{cite book", "{{cite news", "{{reflist", "{{DEFAULTSORT:",
        "{{Commons", "{{authority control", "<ref name=", "<ref>", "</ref>", "<references",
        "<comment>", "<contributor>", "<username>", "<timestamp>", "<title>", "<text bytes=",
        "== References ==", "==Other websites==", "&quot;", "&amp;",
    ];
    let contains = |needle: &[u8]| hay.windows(needle.len()).any(|w| w == needle);
    let markup: Vec<Vec<u8>> = markup_candidates
        .iter()
        .map(|s| s.as_bytes().to_vec())
        .filter(|p| contains(p))
        .take(24)
        .collect();
    assert!(markup.len() >= 16, "only {} markup literals present", markup.len());

    // (e) mixed: 16 common short + 32 rare long.
    let short: Vec<Vec<u8>> = by_freq
        .iter()
        .filter(|(w, _)| (3..=5).contains(&w.len()))
        .take(16)
        .map(|(w, _)| w.clone())
        .collect();
    let long_pool: Vec<Vec<u8>> =
        rare_pool.iter().filter(|w| w.len() >= 10).cloned().collect();
    let mut mixed = short;
    let mut long = spread(&long_pool, 32);
    // Shuffle deterministically so cohorts aren't an artifact of ordering.
    for i in (1..long.len()).rev() {
        long.swap(i, rng.below(i + 1));
    }
    mixed.extend(long);

    run_set("(a) 16 common words, match-dense", &common, hay, corpus, true);
    run_set("(b) 64 mid-frequency words, len 6-12", &mid, hay, corpus, false);
    run_set("(c) 256 rare words, len 8-20", &rare, hay, corpus, false);
    run_set("(d) 24 MediaWiki markup literals, shared prefixes", &markup, hay, corpus, true);
    run_set("(e) mixed: 16 common short + 32 rare long", &mixed, hay, corpus, false);
}
