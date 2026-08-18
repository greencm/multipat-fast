//! Cache footprint and residency probe.
//!
//! Default mode reports (a) compiled footprints of SPARROW vs aho-corasick
//! engines for the benchmark workloads, and (b) a haystack-residency sweep:
//! throughput while the working set sits in L1 / L2 / L3 / DRAM. Because
//! SPARROW's filter state is a few hundred bytes (register-resident), its
//! scan should be residency-insensitive until DRAM bandwidth matters; an
//! automaton whose table misses L1 is latency-bound regardless of haystack
//! size.
//!
//! `cache_probe cg-sparrow|cg-prefix|cg-ac|cg-teddy` runs one engine once
//! over a 4 MiB haystack (for `valgrind --tool=cachegrind`).

use aho_corasick::{AhoCorasick, AhoCorasickKind};
use sparrow::Builder;
use std::time::Instant;

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }
}

/// Shared-prefix API-route workload (see examples/bench.rs).
fn route_workload(bytes: usize) -> (Vec<Vec<u8>>, Vec<u8>) {
    let routes = [
        "users", "carts", "items", "batch", "token", "audit", "quota", "flags",
        "goals", "notes", "tasks", "roles", "keys0", "sync1", "jobs2", "docs3",
    ];
    let patterns: Vec<Vec<u8>> =
        routes.iter().map(|r| format!("GET /api/v1/{}", r).into_bytes()).collect();
    let mut hay = Vec::with_capacity(bytes + 128);
    let mut i = 0usize;
    while hay.len() < bytes {
        hay.extend_from_slice(
            format!(
                "GET /api/v1/zz{:03}?p={} HTTP/1.1\nHost: svc-{}.internal\n",
                i % 1000,
                i % 7,
                i % 13
            )
            .as_bytes(),
        );
        if i % 97 == 0 {
            hay.extend_from_slice(
                format!("GET /api/v1/{} HTTP/1.1\n", routes[i / 97 % routes.len()]).as_bytes(),
            );
        }
        i += 1;
    }
    hay.truncate(bytes);
    (patterns, hay)
}

fn random_workload(bytes: usize) -> (Vec<Vec<u8>>, Vec<u8>) {
    let mut rng = Rng(0x5EED);
    let patterns: Vec<Vec<u8>> =
        (0..64).map(|_| (0..8).map(|_| rng.next() as u8).collect()).collect();
    let hay: Vec<u8> = (0..bytes).map(|_| rng.next() as u8).collect();
    (patterns, hay)
}

/// Scan `hay` repeatedly until ~`budget` bytes are processed; GB/s.
fn sweep<F: FnMut(&[u8]) -> usize>(hay: &[u8], budget: usize, mut scan: F) -> f64 {
    let reps = (budget / hay.len()).max(1);
    let mut sink = 0usize;
    sink += scan(hay); // warmup
    let t = Instant::now();
    for _ in 0..reps {
        sink += scan(hay);
    }
    let secs = t.elapsed().as_secs_f64();
    std::hint::black_box(sink);
    (reps * hay.len()) as f64 / (1u64 << 30) as f64 / secs
}

fn residency_sweep(name: &str, patterns: &[Vec<u8>], corpus: &[u8], make_hay: &dyn Fn(usize) -> Vec<u8>) {
    println!("\n--- residency sweep: {} ---", name);
    let sp = Builder::new().corpus_sample(corpus).build(patterns).unwrap();
    let ac = AhoCorasick::builder().kind(Some(AhoCorasickKind::DFA)).build(patterns).unwrap();
    println!(
        "{:>10}  {:>14}  {:>14}  (haystack resides in)",
        "haystack", "sparrow GB/s", "AC-DFA GB/s"
    );
    for (size, tier) in [
        (16 << 10, "L1d (32K)"),
        (192 << 10, "L2 (1M)"),
        (8 << 20, "L3 (33M)"),
        (64 << 20, "DRAM"),
    ] {
        let hay = make_hay(size);
        let g1 = sweep(&hay, 256 << 20, |h| sp.find_all(h).len());
        let g2 = sweep(&hay, 64 << 20, |h| ac.find_overlapping_iter(h).count());
        println!("{:>10}  {:>14.3}  {:>14.3}  {}", format!("{}K", size >> 10), g1, g2, tier);
    }
}

fn footprints() {
    println!("--- compiled footprints (bytes) ---");
    for (name, (patterns, hay)) in [
        ("16 shared-prefix routes", route_workload(64 << 10)),
        ("64 random len-8", random_workload(64 << 10)),
    ] {
        let sp = Builder::new().corpus_sample(&hay).build(&patterns).unwrap();
        let u = sp.memory_usage();
        let ac_dfa =
            AhoCorasick::builder().kind(Some(AhoCorasickKind::DFA)).build(&patterns).unwrap();
        let ac_nfa = AhoCorasick::builder()
            .kind(Some(AhoCorasickKind::ContiguousNFA))
            .build(&patterns)
            .unwrap();
        let teddy = aho_corasick::packed::Config::new().builder().extend(&patterns).build();
        println!("\n  {}:", name);
        println!(
            "    sparrow: total {:>8}  (SIMD tables {} | scalar tables {} | buckets {} | patterns {})",
            u.total, u.filter_tables, u.scalar_tables, u.buckets, u.patterns
        );
        println!("    aho-corasick DFA:        {:>8}", ac_dfa.memory_usage());
        println!("    aho-corasick contig-NFA: {:>8}", ac_nfa.memory_usage());
        if let Some(t) = teddy {
            println!("    packed Teddy:            {:>8}", t.memory_usage());
        }
    }
}

fn main() {
    let mode = std::env::args().nth(1).unwrap_or_default();
    if let Some(engine) = mode.strip_prefix("cg2-") {
        // Random workload (the AC DFA here is ~514 KB, 16x a 32K L1d).
        let (patterns, hay) = random_workload(4 << 20);
        let n = match engine {
            "sparrow" => {
                let m = Builder::new().corpus_sample(&hay[..64 << 10]).build(&patterns).unwrap();
                m.find_all(&hay).len()
            }
            "ac" => {
                let m = AhoCorasick::builder()
                    .kind(Some(AhoCorasickKind::DFA))
                    .build(&patterns)
                    .unwrap();
                m.find_overlapping_iter(&hay).count()
            }
            other => panic!("unknown engine {}", other),
        };
        println!("{} matches: {}", engine, n);
        return;
    }
    if let Some(engine) = mode.strip_prefix("cg-") {
        // Single-pass runs for cachegrind.
        let (patterns, hay) = route_workload(4 << 20);
        let n = match engine {
            "sparrow" => {
                let m = Builder::new().corpus_sample(&hay[..64 << 10]).build(&patterns).unwrap();
                m.find_all(&hay).len()
            }
            "prefix" => {
                let m = Builder::new()
                    .corpus_sample(&hay[..64 << 10])
                    .positions(&[0, 1, 2, 3])
                    .build(&patterns)
                    .unwrap();
                m.find_all(&hay).len()
            }
            "ac" => {
                let m = AhoCorasick::builder()
                    .kind(Some(AhoCorasickKind::DFA))
                    .build(&patterns)
                    .unwrap();
                m.find_overlapping_iter(&hay).count()
            }
            "teddy" => {
                let m = aho_corasick::packed::Config::new()
                    .builder()
                    .extend(&patterns)
                    .build()
                    .unwrap();
                m.find_iter(&hay).count()
            }
            other => panic!("unknown engine {}", other),
        };
        println!("{} matches: {}", engine, n);
        return;
    }

    footprints();
    let (rp, _) = route_workload(0);
    residency_sweep("shared-prefix routes", &rp, &route_workload(64 << 10).1, &|n| {
        route_workload(n).1
    });
    let (xp, _) = random_workload(0);
    residency_sweep("64 random len-8", &xp, &random_workload(64 << 10).1, &|n| {
        random_workload(n).1
    });
}
