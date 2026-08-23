# SPARROW — sparse-position SIMD multi-pattern matching

SPARROW (*Sparse Position Adaptive Rejection Over Rolling Windows*) is a
multi-pattern (multi-literal) string matcher that keeps the hardware-proven
Teddy/Hyperscan SIMD primitive — PSHUFB nibble shuffles producing bucketed
candidate bitmaps, then exact verification — but turns every structural
choice those engines hardcode into an optimizer decision:

1. **Which pattern bytes the filter inspects.** Teddy inspects the first
   1–4 bytes; FDR the last bytes; Harry the whole literal — always a
   *contiguous* run. SPARROW samples a **sparse, optimizer-chosen set of
   ≤ 4 positions** anywhere in a 32-byte anchor window.
2. **How patterns share SIMD buckets.** Assigned by minimizing an exact
   expected-cost objective that models the PSHUFB nibble cross-product
   inflation — a filter loss all Teddy descendants pay and none account
   for.
3. **How much filter to buy.** The number of sampled positions, 8 vs 16
   buckets (one or two SIMD planes), and whether to split the pattern set
   into per-length cohort filters are all arbitrated by one calibrated
   cost model.
4. **Which configuration ships.** Finalists are re-scored by an
   **empirical referee** — the exact compiled filter run over a corpus
   sample, counting real candidates — which captures byte correlations
   (shared prefixes, protocol framing) that defeat every closed-form model.

Proved in [`docs/DESIGN.md`](docs/DESIGN.md): zero false negatives (under
all supported semantics, streaming included), exactness of the objective's
verification term under the i.i.d. byte model, termination +
move-optimality of bucket refinement, and dominance over the classic
contiguous-prefix configuration under the selection model. The same
document surveys the prior families (Aho–Corasick, FDR, Teddy, Harry,
DFC), states precisely which mechanisms are new, and reports an honest
negative result (a Markov-1 re-scorer, implemented and replaced by the
strictly better empirical referee).

## Results (16 MiB haystacks, one core, best of 5)

| Workload | SPARROW (best engine) | aho-corasick Teddy (packed) | aho-corasick DFA |
|---|---|---|---|
| English words / English text | 0.72 GB/s | 0.71 | 0.36 |
| 64 random len-8 / random bytes | **4.35 GB/s** | 3.38 | 0.38 |
| shared-prefix routes / near-miss log | **6.20 GB/s** | 0.69 | 1.39 |
| short words + long signatures (match-heavy) | 0.20 GB/s | 0.40 | 0.33 |

Apple M-series (NEON engine, same harness, plus a textbook Wu-Manber
comparator — the classic block-shift "skip" algorithm), with the two-prong
build (§ dense lane below):

| Workload | SPARROW two-prong | SPARROW sparse-only | Teddy (packed) | AC DFA | Wu-Manber (B=3) |
|---|---|---|---|---|---|
| English words / English text | 1.03 GB/s | 1.02 | 0.96 | 0.48 | 0.49 |
| 64 random len-8 / random bytes | **4.5 GB/s** | 4.5 | 4.5 | 0.51 | 0.83 |
| shared-prefix routes / near-miss log | **9.0 GB/s** | 9.0 | 1.2 | 2.4 | 1.8 |
| short words + long signatures (match-heavy) | **0.75 GB/s** | 0.36 | 0.43 | 0.50 | 0.37 |

Shared prefixes (`GET /api/v1/…`, magic numbers, protocol headers) are the
common case in IDS/log scanning and the structural blind spot of
prefix-anchored filters: there SPARROW is **9× the reference Teddy**, and
9× its own prefix-positions ablation — same kernel, so the win is entirely
the position optimizer + empirical referee. On random patterns it beats
Teddy by 29%; on match-heavy workloads (last row) filtering is irrelevant
by construction — the cost model predicts this, and the two-prong build
acts on it: the short, common patterns are routed to a bit-parallel
Shift-Or lane (flat cost per byte, no dependent loads), the long
signatures keep a sparse filter with a window no longer pinned by "the",
and the combination is 2.1× the sparse-only matcher and 1.5× the DFA.
Wu-Manber skips bytes but verifies on every shared-prefix near-miss; it
never beats a SIMD filter that touches every byte 16–64 at a time.

### Real corpus: Simple English Wikipedia (aarch64, Apple M-series)

30 MB of the `simplewiki` XML dump as haystack; pattern sets derived
deterministically from a held-out half of the file (the builder's corpus
sample is held out too — never the scanned region). Reproduce with
`cargo run --release --example wiki_bench` (the example header has the
two-line download recipe).

| Pattern set | matches | SPARROW | sparse-only | Teddy (packed) | AC DFA |
|---|---|---|---|---|---|
| (a) 16 common words (match-dense) | 702 K | 1.07 GB/s | 1.08 | 0.92 | 0.51 |
| (b) 64 mid-frequency words, len 6–12 | 16 K | **2.14** | 2.15 | 0.44 | 0.62 |
| (c) 256 rare words, len 8–20 | 251 | 0.36 | 0.36 | n/a | **0.60** |
| (d) 24 markup literals, shared prefixes | 122 K | **3.73** | 3.79 | 2.86 | 0.60 |
| (e) 16 common short + 32 rare long | 709 K | **0.92** | 0.80 | 0.23 | 0.51 |
| (a) leftmost-first | 702 K | **0.84** | — | — | 0.78 |
| (d) leftmost-first | 122 K | **3.58** | — | — | 2.62 |

Honest read: on real text SPARROW wins clearly where its design says it
should — many patterns (b: 5× Teddy), shared prefixes (d: 1.3× Teddy, 6×
the DFA), and the mixed set, where the timed referee routes the 16 short
words dense and beats both its own sparse-only build (+15%) and every
baseline. Match-dense common words (a) are a three-way tie with Teddy and
sparse-only — the referee correctly declined the dense lane there. The
loss is (c): 256 patterns split into two cohorts means two passes and a
high candidate rate, and the DFA wins by 1.7× — exactly the scale
weakness ROADMAP §3 (hashed verification, more planes) targets. Routing
matched the fastest measured configuration on every set.

### Pattern-set scaling (rare words from the wiki corpus, 32 MB haystack, aarch64/M-series)

| n patterns | SPARROW | AC DFA | AC contig-NFA | SPARROW KB | DFA KB | NFA KB | build (spw / DFA) |
|---|---|---|---|---|---|---|---|
| 64 | **3.31 GB/s** | 0.60 | 0.21 | 6 | 156 | 25 | 175 ms / 0.4 ms |
| 256 | **1.09** | 0.59 | 0.17 | 20 | 566 | 70 | 649 ms / 1 ms |
| 1 024 | 0.31 | 0.53 | 0.17 | 97 | 2 062 | 184 | 2.1 s / 4 ms |
| 4 096 | 0.13 | 0.40 | 0.16 | 270 | 7 592 | 517 | 2.6 s / 16 ms |
| 16 384 | 0.045 | 0.30 | 0.13 | 1 006 | 26 198 | 1 584 | 6.7 s / 67 ms |

(`cargo run --release --example scale_bench`; packed Teddy refuses every
set above 64.) The footprint story holds at scale — 26× smaller than the
DFA at 16 K patterns, comparable to the contiguous NFA — but throughput
does not: past ~500 patterns the *filter itself* saturates (candidate
rate 0.006/byte at 256 → 0.46/byte at 16 K; nibble closure of a
500-entry bucket passes almost everything), which no verification
speedup can recover. Hashed verification and 32 buckets moved the
crossover vs the DFA from ~300 to ~800 patterns and made 256 rare words
2.7× faster (0.36 → 0.98 GB/s on the wiki set), but SPARROW remains a
hundreds-of-patterns engine; honest scaling past that needs more filter
selectivity (more planes, more positions, or hierarchical filters —
ROADMAP §3), not faster confirmation.

## Features

- **Engines**: AVX-512BW (64-byte blocks, `VPTESTMB` extraction), AVX2,
  ARM NEON (aarch64; `TBL` shuffles, SHRN-idiom extraction), and a
  bit-identical portable scalar fallback — runtime-detected, monomorphized
  over configuration. The kernels use an offset-load
  structure (no carry registers or cross-lane shifts; that's what lifts
  the window from Teddy's 16 bytes to 32).
- **Byte-class patterns**: a pattern is a sequence of byte *sets*
  (`Pattern`/`ByteSet`, or the class syntax `build_parsed(["GET /api/v\d/",
  "[^\x00-\x1f]"])`). The sampled-position filter prices a class by its
  nibble closure and avoids sampling spread classes; the dense lane and the
  verifier test membership exactly. Case-insensitivity and the wildcard
  byte are now just special cases of this.
- **Semantics**: all overlapping matches (`find_all`); leftmost-first
  (`find_leftmost`, aho-corasick / regex compatible) and leftmost-longest
  (`find_leftmost_longest`) natively — the kernels are re-entered past
  each accepted match instead of filtering an overlapping list, and when
  every pattern sits in the dense lane the Shift-Or state simply resets at
  the match end, so the bytes inside accepted matches are never scanned
  (1.5–1.8× the materialize-and-filter reference on self-overlapping sets,
  parity with packed Teddy); streaming with cross-chunk matches
  (`stream()`), ASCII case-insensitive mode, single-byte pattern wildcards
  (`?`-globs).
- **Regex prefilter** (`--features prefilter`): extract each regex's
  required prefix literals (`regex-syntax`), SPARROW the union, confirm
  candidates with anchored `regex-automata` searches. No false negatives;
  regexes with no finite prefix set run unfiltered. On 50 IDS-style rules
  sharing the `GET /api/v1/` prefix over the near-miss log: **5.5 GB/s**
  vs 2.1 for regex-automata's own multi-regex engine (internal Teddy) and
  0.25 for per-regex scans.
- **Verification**: per-entry guard probes (the model-rarest unsampled
  pattern byte) reject most false candidates with one load.
- **Dense lane (two-prong build)**: patterns no sampled-position filter
  can handle cheaply — short, common words on match-dense text — are
  routed by the cost model to a multi-pattern Shift-Or lane (`src/dense.rs`:
  64-bit lanes, one bit per pattern byte, up to 4 lanes; four haystack
  segments scanned in lock-step so the shift/or chain never stalls; hits
  recorded branch-free and decoded out of the loop). The rest stay sparse.
  The model shortlists partitions; a timed referee then scans the corpus
  sample with the sparse-only baseline and the best splits as built and
  keeps the fastest (`routing_decision()` reports what it saw).
  `dense_lane(false)` disables the lane, `timed_referee(false)` makes the
  model decide alone (deterministic routing); `find_all_unsorted` skips
  the final run-merge when order doesn't matter.

## Usage

```rust
use sparrow::Sparrow;

let m = Sparrow::builder()
    .corpus_sample(traffic_sample)          // fit the model to your data
    .ascii_case_insensitive(false)
    .build(["GET /api/v1/users", "GET /api/v1/carts"]).unwrap();

for hit in m.find_all(log_bytes) {
    println!("pattern {} at {}..{}", hit.pattern, hit.start, hit.end);
}

let mut s = m.stream();                     // streaming, global offsets
for chunk in chunks { for hit in s.push(chunk) { /* … */ } }
```

Class patterns: `Sparrow::builder().build_parsed([r"Host: [a-z]+\.internal"])`
is *not* a regex — `+` is a literal byte; use `build_parsed([r"v\d/users"])`
for fixed-length classes, or build `Pattern`s from `ByteSet`s directly.

Other knobs: `max_positions(k)`, `wildcard_byte(Some(b'?'))`,
`positions(&[0,1,2,3])` (Teddy-style ablation), `corpus_scoring(false)`
(pure closed-form selection), `exhaustive_search(true)`,
`force_engine(Engine::Avx2)`, `dense_lane(false)`, `timed_referee(false)`.

## Repo layout

- `src/builder.rs` — the offline optimizer (byte classes, closure-exact
  objective, position search, greedy + local-search bucketing, plane and
  cohort arbitration, empirical referee, guard selection)
- `src/avx2.rs`, `src/avx512.rs`, `src/neon.rs`, `src/scalar.rs` — sparse
  runtime engines; `src/dense.rs` — the Shift-Or dense lane
- `docs/DESIGN.md` — survey, algorithm spec, theorems and proofs, novelty
  analysis, benchmark details
- `docs/CACHE.md` — working-set model, measured cache-residency sweeps and
  cachegrind miss counts, and modern-CPU implications (the compiled filter
  is 96–256 B and register-resident; the AC-DFA baseline takes one L1 miss
  per ~6 bytes once its table exceeds L1)
- `examples/cache_probe.rs` — footprint + cache-residency measurement tool
- `tests/correctness.rs` — differential fuzzing vs a brute-force oracle
  and vs `aho-corasick`, across all engines and semantics;
  `tests/dense.rs` — the same for the dense lane and the router (chunk and
  segment boundaries, multi-lane passes, streaming, semantics)
- `examples/bench.rs` — the benchmark harness (`cargo run --release
  --example bench`): SPARROW (two-prong, sparse-only, prefix ablation) vs
  aho-corasick DFA, packed Teddy, and a textbook Wu-Manber
- `examples/dense_probe.rs` — dense-lane microbenchmark (scan vs hit cost)
- `examples/regex_prefilter.rs` — the regex-prefilter benchmark
  (`cargo run --release --features prefilter --example regex_prefilter`)
