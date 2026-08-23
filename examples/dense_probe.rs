//! Dense-lane microbenchmark: separates the Shift-Or scan cost (no-match
//! haystack) from per-match cost (match-dense English) and checks that the
//! final ordering pass is a linear run merge, not an n log n sort.
//!
//! Run: cargo run --release --example dense_probe

use sparrow::Builder;
use std::time::Instant;

const N: usize = 16 << 20;

fn bench(label: &str, m: &sparrow::Sparrow, hay: &[u8]) {
    let mut best = f64::INFINITY;
    let mut n = 0;
    for _ in 0..6 {
        let t = Instant::now();
        n = m.find_all(hay).len();
        best = best.min(t.elapsed().as_secs_f64());
    }
    println!(
        "{label:40} {:7.3} GB/s  {:.2} ns/byte  {n} matches",
        hay.len() as f64 / (1u64 << 30) as f64 / best,
        best * 1e9 / hay.len() as f64
    );
}

fn english() -> Vec<u8> {
    let sentences: [&str; 8] = [
        "The committee reviewed the quarterly figures before lunch. ",
        "A light rain moved across the valley and settled over the orchard. ",
        "Engineers deployed the new build to the staging cluster overnight. ",
        "She wrote three letters and mailed none of them. ",
        "Prices rose modestly while inventories continued to shrink. ",
        "The museum's east wing reopened after a decade of restoration. ",
        "Local farmers reported an unusually early harvest this year. ",
        "The orchestra tuned quietly as the hall filled with visitors. ",
    ];
    let mut seed = 0x5EEDu64;
    let mut hay = Vec::with_capacity(N + 128);
    while hay.len() < N {
        seed = seed.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = seed;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^= z >> 31;
        hay.extend_from_slice(sentences[(z % 8) as usize].as_bytes());
    }
    hay.truncate(N);
    hay
}

fn main() {
    let words = [
        "orchard", "cluster", "harvest", "orchestra", "restoration", "committee", "quarterly",
        "staging", "inventories", "museum", "letters", "valley", "overnight", "modestly",
        "decade", "farmers",
    ];
    let two = Builder::new().force_dense(true).build(&words).unwrap();
    let one = Builder::new().force_dense(true).build(&words[..7]).unwrap();
    println!(
        "lanes: 16 words -> {}, 7 words -> {}",
        two.dense_lane().unwrap().lane_count(),
        one.dense_lane().unwrap().lane_count()
    );

    let zeros = vec![b'x'; N];
    bench("16 words (2 lanes) / no-match", &two, &zeros);
    bench("7 words (1 lane) / no-match", &one, &zeros);

    let hay = english();
    bench("16 words (2 lanes) / english, dense hits", &two, &hay);
    let sparse = Builder::new().corpus_sample(&hay[..65536]).dense_lane(false).build(&words).unwrap();
    bench("16 words sparse-only / english", &sparse, &hay);

    // Streaming: packet-sized chunks vs whole-buffer.
    for chunk in [1500usize, 4096, 65536] {
        let mut best = f64::INFINITY;
        let mut n = 0usize;
        for _ in 0..4 {
            let t = Instant::now();
            let mut s = two.stream();
            n = hay.chunks(chunk).map(|c| s.push(c).len()).sum();
            best = best.min(t.elapsed().as_secs_f64());
        }
        println!(
            "stream {chunk:6}-byte chunks             {:7.3} GB/s  {n} matches",
            hay.len() as f64 / (1u64 << 30) as f64 / best
        );
    }

    // Streaming a sparse-routed (filter-fast) set: here the old
    // concat-copy per push was a real fraction of the work.
    let routes: Vec<String> = (0..16).map(|i| format!("GET /api/v1/route{:02}", i)).collect();
    let fast = Builder::new().corpus_sample(&hay[..65536]).build(&routes).unwrap();
    for chunk in [1500usize, 65536] {
        let mut best = f64::INFINITY;
        for _ in 0..4 {
            let t = Instant::now();
            let mut s = fast.stream();
            let n: usize = hay.chunks(chunk).map(|c| s.push(c).len()).sum();
            std::hint::black_box(n);
            best = best.min(t.elapsed().as_secs_f64());
        }
        println!(
            "stream sparse-fast {chunk:6}-byte chunks {:7.3} GB/s",
            hay.len() as f64 / (1u64 << 30) as f64 / best
        );
    }

    // Ordering pass: natural engine output should be a few sorted runs.
    let mut v = two.find_all_unsorted(&hay);
    let t = Instant::now();
    v.sort();
    let dt = t.elapsed().as_secs_f64();
    println!(
        "final ordering pass over {} matches: {:.2} ms = {:.3} ns/byte",
        v.len(),
        dt * 1e3,
        dt * 1e9 / N as f64
    );
}
