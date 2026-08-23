//! Pattern-set scaling study on real text: throughput, compiled footprint,
//! and build time for SPARROW vs aho-corasick (DFA, contiguous NFA) and
//! packed Teddy while it accepts the set, at 64 .. 16 K patterns drawn
//! from Simple English Wikipedia.
//!
//! Needs bench_data/simplewiki-64MB.xml (gitignored). Regenerate with:
//!   curl -sL --range 0-50000000 \
//!     https://dumps.wikimedia.org/simplewiki/latest/simplewiki-latest-pages-articles-multistream.xml.bz2 \
//!     -o /tmp/w.bz2 && bzip2 -dc /tmp/w.bz2 | head -c 64000000 > bench_data/simplewiki-64MB.xml
//!
//! Run: cargo run --release --example scale_bench

use aho_corasick::{AhoCorasick, AhoCorasickKind};
use sparrow::Builder;
use std::collections::HashMap;
use std::time::Instant;

const MB: usize = 1 << 20;

fn time_best_of<F: FnMut() -> usize>(mut f: F, reps: usize) -> (f64, usize) {
    let mut best = f64::INFINITY;
    let mut n = 0;
    f();
    for _ in 0..reps {
        let t = Instant::now();
        n = f();
        best = best.min(t.elapsed().as_secs_f64());
    }
    (best, n)
}

fn main() {
    let path = "bench_data/simplewiki-64MB.xml";
    let data = std::fs::read(path).unwrap_or_else(|_| {
        eprintln!("missing {path} — see the header of examples/scale_bench.rs to regenerate");
        std::process::exit(1);
    });
    let hay = &data[..32 * MB];
    let held_out = &data[48 * MB..48 * MB + 256 * 1024];

    // Rare words (len 8..=20, frequency <= 3 in the haystack), deduped,
    // deterministic order by first occurrence.
    let mut freq: HashMap<&[u8], (usize, usize)> = HashMap::new(); // word -> (count, first_pos)
    let mut i = 0;
    while i < hay.len() {
        if hay[i].is_ascii_alphabetic() {
            let s = i;
            while i < hay.len() && hay[i].is_ascii_alphabetic() {
                i += 1;
            }
            let w = &hay[s..i];
            if (8..=20).contains(&w.len()) {
                let e = freq.entry(w).or_insert((0, s));
                e.0 += 1;
            }
        } else {
            i += 1;
        }
    }
    let mut rare: Vec<(&[u8], usize)> =
        freq.iter().filter(|(_, &(c, _))| c <= 3).map(|(&w, &(_, p))| (w, p)).collect();
    rare.sort_by_key(|&(_, p)| p);
    println!("rare word pool: {}", rare.len());

    let gb = hay.len() as f64 / (1u64 << 30) as f64;
    println!(
        "\n{:>7} | {:>28} | {:>24} | {:>21} | {:>8}",
        "n", "throughput GB/s", "footprint KB", "build ms", "sparrow"
    );
    println!(
        "{:>7} | {:>8} {:>9} {:>9} | {:>7} {:>7} {:>8} | {:>6} {:>6} {:>7} | {:>8}",
        "", "sparrow", "AC-DFA", "AC-NFA", "sparrow", "AC-DFA", "AC-NFA", "spw", "DFA", "NFA", "cand/B"
    );
    for &n in &[64usize, 256, 1024, 4096, 16384] {
        if n > rare.len() {
            println!("{n:>7} | (pool exhausted)");
            continue;
        }
        let pats: Vec<Vec<u8>> = rare.iter().take(n).map(|&(w, _)| w.to_vec()).collect();

        let t = Instant::now();
        let sp = Builder::new().corpus_sample(held_out).build(&pats).unwrap();
        let b_sp = t.elapsed().as_secs_f64() * 1e3;
        let t = Instant::now();
        let dfa = AhoCorasick::builder().kind(Some(AhoCorasickKind::DFA)).build(&pats).unwrap();
        let b_dfa = t.elapsed().as_secs_f64() * 1e3;
        let t = Instant::now();
        let nfa = AhoCorasick::builder()
            .kind(Some(AhoCorasickKind::ContiguousNFA))
            .build(&pats)
            .unwrap();
        let b_nfa = t.elapsed().as_secs_f64() * 1e3;
        let teddy = aho_corasick::packed::Config::new().builder().extend(&pats).build();

        let reps = if n >= 4096 { 2 } else { 3 };
        let (t_sp, n_sp) = time_best_of(|| sp.find_all(hay).len(), reps);
        let (t_dfa, n_dfa) = time_best_of(|| dfa.find_overlapping_iter(hay).count(), reps);
        let (t_nfa, n_nfa) = time_best_of(|| nfa.find_overlapping_iter(hay).count(), reps);
        assert_eq!(n_sp, n_dfa);
        assert_eq!(n_sp, n_nfa);

        println!(
            "{n:>7} | {:>8.3} {:>9.3} {:>9.3} | {:>7.0} {:>7.0} {:>8.0} | {:>6.0} {:>6.0} {:>7.0} | {:>8.4}",
            gb / t_sp,
            gb / t_dfa,
            gb / t_nfa,
            sp.memory_usage().total as f64 / 1024.0,
            dfa.memory_usage() as f64 / 1024.0,
            nfa.memory_usage() as f64 / 1024.0,
            b_sp,
            b_dfa,
            b_nfa,
            sp.expected_candidate_rate(),
        );
        println!(
            "        | cohorts {} buckets {:?} matches {} teddy {}",
            sp.cohort_count(),
            sp.bucket_counts(),
            n_sp,
            match &teddy {
                Some(pk) => {
                    let (t_pk, _) = time_best_of(|| pk.find_iter(hay).count(), reps);
                    format!("{:.3} GB/s", gb / t_pk)
                }
                None => "refused".to_string(),
            },
        );
    }
}
