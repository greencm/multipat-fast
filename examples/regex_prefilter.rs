//! Regex prefilter benchmark (requires `--features prefilter`):
//! 50 IDS-style rules with pathologically shared literal prefixes over the
//! near-miss log haystack from the main bench, vs regex-automata alone.
//!
//! Run: cargo run --release --features prefilter --example regex_prefilter

use sparrow::prefilter::Prefilter;
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
    const MB: usize = 1 << 20;
    let routes = [
        "users", "carts", "items", "batch", "token", "audit", "quota", "flags", "goals",
        "notes", "tasks", "roles", "keys0", "sync1", "jobs2", "docs3", "acls4", "meta5",
        "logs6", "tags7", "sess8", "auth9", "queue", "stats", "hooks",
    ];
    // 50 rules: 2 per route, all sharing the "GET /api/v1/" prefix.
    let rules: Vec<String> = routes
        .iter()
        .flat_map(|r| {
            [
                format!(r"GET /api/v1/{}\?id=\d+ HTTP/1\.[01]", r),
                format!(r"GET /api/v1/{}/[0-9a-f]{{4,}} HTTP", r),
            ]
        })
        .collect();
    let refs: Vec<&str> = rules.iter().map(|s| s.as_str()).collect();

    let mut hay = Vec::with_capacity(16 * MB + 128);
    let mut i = 0usize;
    while hay.len() < 16 * MB {
        hay.extend_from_slice(
            format!("GET /api/v1/zz{:03}?p={} HTTP/1.1\nHost: svc-{}.internal\n", i % 1000, i % 7, i % 13)
                .as_bytes(),
        );
        if i % 97 == 0 {
            let r = routes[i / 97 % routes.len()];
            hay.extend_from_slice(format!("GET /api/v1/{}?id={} HTTP/1.0\n", r, i).as_bytes());
        }
        if i % 131 == 0 {
            let r = routes[i / 131 % routes.len()];
            hay.extend_from_slice(format!("GET /api/v1/{}/{:06x} HTTP/1.1\n", r, i * 2654435761).as_bytes());
        }
        i += 1;
    }
    hay.truncate(16 * MB);
    let gb = hay.len() as f64 / (1u64 << 30) as f64;

    let pf = Prefilter::new(&refs, Some(&hay[..65536])).unwrap();
    println!(
        "{} rules -> {} distinct literals, {} unfiltered",
        pf.regex_count(),
        pf.literals().len(),
        pf.unfiltered().len()
    );
    let (t, n) = best(|| pf.find_all(&hay).len());
    println!("sparrow prefilter + confirm   {:6.3} GB/s   {} matches", gb / t, n);
    let (t, n) = best(|| pf.candidates(&hay).len());
    println!("  (prefilter alone)           {:6.3} GB/s   {} candidates", gb / t, n);

    // regex-automata meta (what the prefilter confirms with), full scans.
    let res: Vec<regex_automata::meta::Regex> = refs
        .iter()
        .map(|p| {
            regex_automata::meta::Regex::builder()
                .syntax(regex_automata::util::syntax::Config::new().utf8(false))
                .build(p)
                .unwrap()
        })
        .collect();
    let (t, n) = best(|| res.iter().map(|re| re.find_iter(&hay).count()).sum());
    println!("regex-automata, 50x find_iter {:6.3} GB/s   {} matches", gb / t, n);
    let multi = regex_automata::meta::Regex::builder()
        .syntax(regex_automata::util::syntax::Config::new().utf8(false))
        .build_many(&refs)
        .unwrap();
    let (t, n) = best(|| multi.find_iter(&hay).count());
    println!("regex-automata multi-regex    {:6.3} GB/s   {} matches (leftmost, first-wins)", gb / t, n);
}
