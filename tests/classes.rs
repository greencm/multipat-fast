//! Byte-class patterns: every engine, the dense lane, and the router must
//! report exactly the reference match set for per-position byte sets.

use sparrow::{naive, BuildError, Builder, ByteSet, Engine, Pattern, PatternError};

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

const ENGINES: &[Engine] = &[Engine::Scalar, Engine::Avx2, Engine::Avx512, Engine::Neon];

fn check(pats: &[Pattern], hay: &[u8]) -> usize {
    let expect = naive::find_all_patterns(pats, hay);
    let corpus = &hay[..hay.len().min(1 << 16)];
    for &e in ENGINES {
        let base = Builder::new().corpus_sample(corpus).timed_referee(false);
        for (label, b) in [
            ("sparse", base.clone().dense_lane(false)),
            ("routed", base.clone()),
            ("dense", base.clone().force_dense(true)),
        ] {
            match b.force_engine(e).build_patterns(pats.to_vec()) {
                Ok(m) => assert_eq!(m.find_all(hay), expect, "{label} {e:?} {pats:?}"),
                Err(BuildError::EngineUnavailable) => {}
                Err(err) => panic!("{label} {e:?}: {err}"),
            }
        }
    }
    expect.len()
}

fn log_haystack(rng: &mut Rng, n: usize) -> Vec<u8> {
    let mut h = Vec::with_capacity(n + 64);
    let mut i = 0;
    while h.len() < n {
        let line = match rng.below(5) {
            0 => format!("GET /api/v{}/users/{} HTTP/1.1\n", rng.below(3), rng.below(100000)),
            1 => format!("POST /api/v1/items?id={:x} HTTP/1.1\n", rng.next() as u32),
            2 => format!("Host: svc-{}.internal\n", i % 13),
            3 => format!("X-Trace: {:08x}-{:04x}\n", rng.next() as u32, rng.next() as u16),
            _ => String::new(),
        };
        if line.is_empty() {
            h.extend_from_slice(b"\r\n\x00\x01\x7f binary \xff\xfe chunk\n");
        } else {
            h.extend_from_slice(line.as_bytes());
        }
        i += 1;
    }
    h.truncate(n);
    h
}

#[test]
fn parsed_classes_smoke() {
    let pats: Vec<Pattern> = [r"GET /api/v\d/", r"id=[0-9a-f]{", r"Host: svc-\d\d?", r"X-Trace: [0-9a-f]"]
        .iter()
        .map(|s| Pattern::parse(s).unwrap())
        .collect();
    let mut rng = Rng(1);
    let hay = log_haystack(&mut rng, 200_000);
    assert!(check(&pats, &hay) > 1000);
}

#[test]
fn builder_parse_entry_point() {
    let m = Builder::new().build_parsed([r"v\d/users", r"[\x00-\x1f]binary", "chunk."]).unwrap();
    let mut rng = Rng(2);
    let hay = log_haystack(&mut rng, 50_000);
    let pats: Vec<Pattern> =
        [r"v\d/users", r"[\x00-\x1f]binary", "chunk."].iter().map(|s| Pattern::parse(s).unwrap()).collect();
    assert_eq!(m.find_all(&hay), naive::find_all_patterns(&pats, &hay));
    assert_eq!(
        Builder::new().build_parsed(["ok", "[z-a]"]).err(),
        Some(BuildError::BadPattern(PatternError::BadRange(2)))
    );
}

#[test]
fn wide_classes_make_the_filter_useless_but_stay_correct() {
    // Every position is a wide class: nibble closure is ~everything, so the
    // sparse filter admits every anchor; the router should go dense (all
    // patterns fit) and either way the result must be exact.
    let pats: Vec<Pattern> = [r"\d\d\d\d", r"\w\w\w\w\w", r"[^a-z][^a-z][^a-z]", r"..\n"]
        .iter()
        .map(|s| Pattern::parse(s).unwrap())
        .collect();
    let mut rng = Rng(3);
    let hay = log_haystack(&mut rng, 300_000);
    check(&pats, &hay);
    let routed = Builder::new().corpus_sample(&hay[..65536]).timed_referee(false).build_patterns(pats.clone()).unwrap();
    assert!(routed.dense_lane().is_some(), "all-class short patterns should route dense");
}

#[test]
fn mixed_exact_and_class_patterns() {
    let pats: Vec<Pattern> = vec![
        Pattern::bytes(b"GET /api/v1/users"),
        Pattern::parse(r"GET /api/v[02]/users").unwrap(),
        Pattern::parse(r"svc-1\d").unwrap(),
        Pattern::bytes(b"HTTP/1.1\n"),
        Pattern::parse(r"\xff\xfe").unwrap(),
    ];
    let mut rng = Rng(4);
    let hay = log_haystack(&mut rng, 400_000);
    assert!(check(&pats, &hay) > 100);
}

#[test]
fn case_insensitive_applies_to_class_patterns() {
    let pats = vec![Pattern::parse(r"host: svc").unwrap(), Pattern::parse(r"[g]et /").unwrap()];
    let mut rng = Rng(5);
    let hay = log_haystack(&mut rng, 100_000);
    let m = Builder::new().ascii_case_insensitive(true).build_patterns(pats.clone()).unwrap();
    let folded: Vec<Pattern> = pats.iter().cloned().map(Pattern::ascii_case_fold).collect();
    let expect = naive::find_all_patterns(&folded, &hay);
    assert_eq!(m.find_all(&hay), expect);
    assert!(!expect.is_empty());
    // Without folding, "host:" (lowercase) never matches "Host:".
    let exact = Builder::new().build_patterns(pats.clone()).unwrap();
    assert!(exact.find_all(&hay).iter().all(|m| m.pattern == 1));
}

#[test]
fn byte_string_build_equals_pattern_build() {
    let words = ["Host", "GET", "users", "chunk"];
    let mut rng = Rng(6);
    let hay = log_haystack(&mut rng, 100_000);
    let a = Builder::new().build(words).unwrap().find_all(&hay);
    let b = Builder::new().build_patterns(words.iter().map(|w| Pattern::from(*w))).unwrap().find_all(&hay);
    assert_eq!(a, b);
    let c = Builder::new().wildcard_byte(Some(b'?')).build(["H?st", "v?/users"]).unwrap().find_all(&hay);
    let d = Builder::new().build_parsed(["H.st", "v./users"]).unwrap().find_all(&hay);
    assert_eq!(c, d);
}

#[test]
fn fuzz_class_patterns() {
    let mut rng = Rng(0xC1A55);
    for round in 0..150 {
        let alpha = 3 + rng.below(4);
        let np = 1 + rng.below(10);
        let mut pats = Vec::new();
        for _ in 0..np {
            let len = 1 + rng.below(7);
            let mut p = Pattern::new();
            for _ in 0..len {
                let set = match rng.below(4) {
                    0 => ByteSet::ANY,
                    1 => {
                        let mut s = ByteSet::EMPTY;
                        for _ in 0..1 + rng.below(3) {
                            s.insert(b'a' + rng.below(alpha) as u8);
                        }
                        s
                    }
                    _ => ByteSet::byte(b'a' + rng.below(alpha) as u8),
                };
                p.push(set);
            }
            pats.push(p);
        }
        let n = rng.below(if round % 10 == 0 { 70_000 } else { 500 });
        let hay: Vec<u8> = (0..n).map(|_| b'a' + rng.below(alpha) as u8).collect();
        check(&pats, &hay);
    }
}
