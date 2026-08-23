//! Benchmark harness: SPARROW (AVX-512 / AVX2 kernels, plus the
//! prefix-positions ablation = Teddy-style contiguous sampling inside the
//! same runtime) vs aho-corasick's DFA (overlapping), packed Teddy, and a
//! textbook Wu-Manber (block-shift skipping, the classic "skip" competitor).
//!
//! Run: cargo run --release --example bench
//!
//! Semantics note: SPARROW and AhoCorasick(overlapping) report identical
//! match sets (asserted). packed::Teddy is leftmost-first, so its count can
//! differ; it is included as a throughput reference only.

use aho_corasick::{AhoCorasick, AhoCorasickKind};
use sparrow::{Builder, Engine};
use std::time::Instant;

/// Textbook Wu-Manber (1994): a SHIFT table indexed by the B-byte block that
/// ends the length-m window (m = shortest pattern), and HASH buckets of
/// candidates verified at shift-0 positions. Reports every (pattern, start)
/// pair, so its count is comparable to AC overlapping and SPARROW.
struct WuManber {
    m: usize,
    b: usize,
    shift: Vec<u16>,
    bucket: Vec<Vec<usize>>,
    patterns: Vec<Vec<u8>>,
}

impl WuManber {
    fn new(patterns: &[Vec<u8>]) -> Self {
        let m = patterns.iter().map(|p| p.len()).min().unwrap();
        let b = if m >= 3 { 3 } else { 2 };
        let mut shift = vec![(m - b + 1) as u16; 1 << 16];
        let mut bucket = vec![Vec::new(); 1 << 16];
        for (idx, p) in patterns.iter().enumerate() {
            // Only the first m bytes participate in the filter.
            for q in b..=m {
                let h = Self::hash(&p[q - b..q], b);
                let d = (m - q) as u16;
                if d < shift[h] {
                    shift[h] = d;
                }
                if d == 0 {
                    bucket[h].push(idx);
                }
            }
        }
        Self { m, b, shift, bucket, patterns: patterns.to_vec() }
    }

    #[inline(always)]
    fn hash(block: &[u8], b: usize) -> usize {
        if b == 2 {
            ((block[0] as usize) << 8) | block[1] as usize
        } else {
            let x = ((block[0] as u32) << 16) | ((block[1] as u32) << 8) | block[2] as u32;
            (x.wrapping_mul(0x9E37_79B1) >> 16) as usize
        }
    }

    fn count(&self, hay: &[u8]) -> usize {
        let (m, b) = (self.m, self.b);
        if hay.len() < m {
            return 0;
        }
        let mut n = 0;
        let mut i = 0;
        let end = hay.len() - m;
        while i <= end {
            let h = Self::hash(&hay[i + m - b..i + m], b);
            let s = self.shift[h] as usize;
            if s != 0 {
                i += s;
                continue;
            }
            for &idx in &self.bucket[h] {
                let p = &self.patterns[idx];
                if hay.len() - i >= p.len() && &hay[i..i + p.len()] == p.as_slice() {
                    n += 1;
                }
            }
            i += 1;
        }
        n
    }
}

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

const MB: usize = 1 << 20;
const HAY_SIZE: usize = 16 * MB;

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

fn run_workload(name: &str, patterns: &[Vec<u8>], hay: &[u8], corpus: &[u8]) {
    println!("\n=== {} ===", name);
    println!(
        "patterns: {} (len {}..{}), haystack: {} MB",
        patterns.len(),
        patterns.iter().map(|p| p.len()).min().unwrap(),
        patterns.iter().map(|p| p.len()).max().unwrap(),
        hay.len() / MB
    );

    let base = || Builder::new().corpus_sample(corpus);
    // One matcher per SIMD engine this CPU supports (plus scalar if none).
    let mut engines: Vec<(String, sparrow::Sparrow)> = Vec::new();
    for e in [Engine::Avx512, Engine::Avx2, Engine::Neon] {
        if let Ok(m) = base().force_engine(e).build(patterns) {
            engines.push((format!("{:?}", e), m));
        }
    }
    if engines.is_empty() {
        engines.push(("Scalar".to_string(), base().build(patterns).unwrap()));
    }
    let sp2 = &engines.last().unwrap().1;
    let sp_sparse_only = base().dense_lane(false).build(patterns).unwrap();
    let sp_prefix = {
        let w = patterns.iter().map(|p| p.len()).min().unwrap().min(4) as u8;
        let pos: Vec<u8> = (0..w).collect();
        base().positions(&pos).build(patterns).unwrap()
    };
    println!(
        "sparrow: positions {:?}, buckets {:?}, cohorts {}, model cost {:.3e}/byte | prefix ablation {:.3e}/byte",
        sp2.sampled_positions(),
        sp2.bucket_counts(),
        sp2.cohort_count(),
        sp2.expected_cost(),
        sp_prefix.expected_cost()
    );
    match sp2.dense_lane() {
        Some(d) => println!(
            "  dense lane: {} patterns (len {}..{}) in {} Shift-Or lane(s); sparse-only model cost {:.3e}/byte",
            d.pattern_count(), d.min_len(), d.max_len(), d.lane_count(), sp_sparse_only.expected_cost()
        ),
        None => println!("  dense lane: unused"),
    }
    let d = sp2.routing_decision();
    let cands: Vec<String> = d
        .candidates
        .iter()
        .enumerate()
        .map(|(i, c)| {
            format!(
                "{}{}dense:{} model {:.2e}{}",
                if i == d.chosen { "*" } else { "" },
                if i == d.chosen { "" } else { " " },
                c.dense_patterns,
                c.model_cost,
                c.measured_ns_per_byte.map_or(String::new(), |ns| format!(" meas {:.2}ns/B", ns))
            )
        })
        .collect();
    println!("  routing ({}): {}", if d.timed { "timed referee" } else { "model" }, cands.join(" | "));

    let ac_dfa = AhoCorasick::builder()
        .kind(Some(AhoCorasickKind::DFA))
        .build(patterns)
        .unwrap();
    let packed = aho_corasick::packed::Config::new().builder().extend(patterns).build();
    let wm = WuManber::new(patterns);

    let gb = hay.len() as f64 / (1u64 << 30) as f64;

    let mut n_sp: Option<usize> = None;
    for (label, m) in &engines {
        let (t, n) = time_best_of(|| m.find_all(hay).len());
        println!("  sparrow {:16} {:8.3} GB/s   {} matches", label, gb / t, n);
        if let Some(prev) = n_sp {
            assert_eq!(n, prev, "all sparrow engines must agree");
        }
        n_sp = Some(n);
    }
    let n_sp = n_sp.unwrap();
    if sp2.dense_lane().is_some() {
        let (t, n_so) = time_best_of(|| sp_sparse_only.find_all(hay).len());
        println!("  sparrow (sparse only)    {:8.3} GB/s   {} matches", gb / t, n_so);
        assert_eq!(n_sp, n_so, "two-prong and sparse-only must agree");
    }
    let (t, n_pre) = time_best_of(|| sp_prefix.find_all(hay).len());
    println!("  sparrow (prefix ablate)  {:8.3} GB/s   {} matches", gb / t, n_pre);
    let (t, n_cnt) = time_best_of(|| sp2.count_all(hay));
    println!("  sparrow count_all        {:8.3} GB/s   {} matches", gb / t, n_cnt);
    assert_eq!(n_sp, n_cnt, "count_all must agree with find_all");
    let (t, n_ac) = time_best_of(|| ac_dfa.find_overlapping_iter(hay).count());
    println!("  aho-corasick DFA (ovlp)  {:8.3} GB/s   {} matches", gb / t, n_ac);
    if let Some(ref pk) = packed {
        let (t, n_pk) = time_best_of(|| pk.find_iter(hay).count());
        println!("  aho-corasick Teddy(pkd)  {:8.3} GB/s   {} matches (leftmost)", gb / t, n_pk);
    } else {
        println!("  aho-corasick Teddy(pkd)  unavailable for this pattern set");
    }

    let (t, n_wm) = time_best_of(|| wm.count(hay));
    println!(
        "  wu-manber (B={}, m={:2}) {:8.3} GB/s   {} matches",
        wm.b, wm.m, gb / t, n_wm
    );

    assert_eq!(n_sp, n_pre, "both sparrow configs must agree");
    assert_eq!(n_sp, n_wm, "sparrow must agree with Wu-Manber");
    assert_eq!(n_sp, n_ac, "sparrow must agree with AC overlapping");

    // Leftmost-first semantics: apples-to-apples with packed Teddy and the
    // aho-corasick default (leftmost-first) DFA.
    println!("  -- leftmost-first --");
    let (t, n_lm) = time_best_of(|| sp2.find_leftmost(hay).len());
    println!("  sparrow find_leftmost    {:8.3} GB/s   {} matches", gb / t, n_lm);
    let (t, n_lr) = time_best_of(|| sp2.find_leftmost_nonoverlapping(hay).len());
    println!("  sparrow (all+filter ref) {:8.3} GB/s   {} matches", gb / t, n_lr);
    let ac_lf = AhoCorasick::builder()
        .kind(Some(AhoCorasickKind::DFA))
        .match_kind(aho_corasick::MatchKind::LeftmostFirst)
        .build(patterns)
        .unwrap();
    let (t, n_al) = time_best_of(|| ac_lf.find_iter(hay).count());
    println!("  aho-corasick DFA (lmf)   {:8.3} GB/s   {} matches", gb / t, n_al);
    if let Some(ref pk) = packed {
        let (t, n_pk) = time_best_of(|| pk.find_iter(hay).count());
        println!("  aho-corasick Teddy(pkd)  {:8.3} GB/s   {} matches", gb / t, n_pk);
        assert_eq!(n_lm, n_pk, "sparrow leftmost must agree with Teddy");
    }
    assert_eq!(n_lm, n_lr, "native leftmost must agree with reference");
    assert_eq!(n_lm, n_al, "sparrow leftmost must agree with AC leftmost-first");
}

fn english_haystack(rng: &mut Rng) -> Vec<u8> {
    let sentences: Vec<&str> = vec![
        "The committee reviewed the quarterly figures before lunch. ",
        "A light rain moved across the valley and settled over the orchard. ",
        "Engineers deployed the new build to the staging cluster overnight. ",
        "She wrote three letters and mailed none of them. ",
        "Prices rose modestly while inventories continued to shrink. ",
        "The museum's east wing reopened after a decade of restoration. ",
        "Local farmers reported an unusually early harvest this year. ",
        "The orchestra tuned quietly as the hall filled with visitors. ",
    ];
    let mut hay = Vec::with_capacity(HAY_SIZE + 128);
    while hay.len() < HAY_SIZE {
        hay.extend_from_slice(sentences[rng.below(sentences.len())].as_bytes());
    }
    hay.truncate(HAY_SIZE);
    hay
}

fn main() {
    let mut rng = Rng(0x5EED);

    // Workload 1: English words over English text (match-heavy).
    {
        let hay = english_haystack(&mut rng);
        let patterns: Vec<Vec<u8>> = [
            "orchard", "cluster", "harvest", "orchestra", "restoration", "committee",
            "quarterly", "staging", "inventories", "museum", "letters", "valley",
            "overnight", "modestly", "decade", "farmers",
        ]
        .iter()
        .map(|s| s.as_bytes().to_vec())
        .collect();
        run_workload("english words / english text", &patterns, &hay, &hay[..64 * 1024]);
    }

    // Workload 2: random 8-byte patterns over random bytes (rare hits).
    {
        let hay: Vec<u8> = (0..HAY_SIZE).map(|_| rng.next() as u8).collect();
        let patterns: Vec<Vec<u8>> =
            (0..64).map(|_| (0..8).map(|_| rng.next() as u8).collect()).collect();
        run_workload("64 random patterns / random bytes", &patterns, &hay, &hay[..64 * 1024]);
    }

    // Workload 3: adversarial shared prefix — the case sparse sampling is
    // built for. Every pattern starts with "GET /api/v1/", and the haystack
    // is full of near-miss "GET /api/v1/..." lines that never complete a
    // pattern. Prefix filters shortlist every line; sparse sampling doesn't.
    {
        let routes = [
            "users", "carts", "items", "batch", "token", "audit", "quota", "flags",
            "goals", "notes", "tasks", "roles", "keys0", "sync1", "jobs2", "docs3",
        ];
        let patterns: Vec<Vec<u8>> =
            routes.iter().map(|r| format!("GET /api/v1/{}", r).into_bytes()).collect();
        let mut hay = Vec::with_capacity(HAY_SIZE + 128);
        let mut i = 0usize;
        while hay.len() < HAY_SIZE {
            let line = format!(
                "GET /api/v1/zz{:03}?p={} HTTP/1.1\nHost: svc-{}.internal\n",
                i % 1000,
                i % 7,
                i % 13
            );
            hay.extend_from_slice(line.as_bytes());
            if i % 97 == 0 {
                hay.extend_from_slice(
                    format!("GET /api/v1/{} HTTP/1.1\n", routes[i / 97 % routes.len()]).as_bytes(),
                );
            }
            i += 1;
        }
        hay.truncate(HAY_SIZE);
        run_workload("shared-prefix API routes / near-miss log", &patterns, &hay, &hay[..64 * 1024]);
    }

    // Workload 4: mixed pattern lengths — 24 short common-looking words and
    // 24 long rare signatures. Exercises the length-cohort arbitration
    // (a single filter would be limited to the shortest pattern's window).
    {
        let hay = english_haystack(&mut rng);
        let mut patterns: Vec<Vec<u8>> = [
            "the", "and", "for", "was", "hall", "rain", "wing", "none", "over", "rose",
            "east", "year", "build", "light", "early", "three", "lunch", "wrote",
            "tuned", "moved", "hills", "vapor", "quill", "zesty",
        ]
        .iter()
        .map(|s| s.as_bytes().to_vec())
        .collect();
        for i in 0..24 {
            patterns.push(format!("signature-block-{:04x}-terminal-marker", i * 2654435761u64 % 65536).into_bytes());
        }
        run_workload("mixed short words + long signatures", &patterns, &hay, &hay[..64 * 1024]);
    }
}
