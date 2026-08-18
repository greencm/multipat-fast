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

Shared prefixes (`GET /api/v1/…`, magic numbers, protocol headers) are the
common case in IDS/log scanning and the structural blind spot of
prefix-anchored filters: there SPARROW is **9× the reference Teddy**, and
9× its own prefix-positions ablation — same kernel, so the win is entirely
the position optimizer + empirical referee. On random patterns it beats
Teddy by 29%; on match-heavy workloads (last row) filtering is irrelevant
by construction and the automaton wins — the cost model predicts this.

## Features

- **Engines**: AVX-512BW (64-byte blocks, `VPTESTMB` extraction), AVX2,
  and a bit-identical portable scalar fallback — runtime-detected,
  monomorphized over configuration. The kernels use an offset-load
  structure (no carry registers or cross-lane shifts; that's what lifts
  the window from Teddy's 16 bytes to 32).
- **Semantics**: all overlapping matches (`find_all`), leftmost
  non-overlapping (`find_leftmost_nonoverlapping`, aho-corasick
  leftmost-first compatible), streaming with cross-chunk matches
  (`stream()`), ASCII case-insensitive mode, single-byte pattern wildcards
  (`?`-globs).
- **Verification**: per-entry guard probes (the model-rarest unsampled
  pattern byte) reject most false candidates with one load.

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

Other knobs: `max_positions(k)`, `wildcard_byte(Some(b'?'))`,
`positions(&[0,1,2,3])` (Teddy-style ablation), `corpus_scoring(false)`
(pure closed-form selection), `exhaustive_search(true)`,
`force_engine(Engine::Avx2)`.

## Repo layout

- `src/builder.rs` — the offline optimizer (byte classes, closure-exact
  objective, position search, greedy + local-search bucketing, plane and
  cohort arbitration, empirical referee, guard selection)
- `src/avx2.rs`, `src/avx512.rs`, `src/scalar.rs` — runtime engines
- `docs/DESIGN.md` — survey, algorithm spec, theorems and proofs, novelty
  analysis, benchmark details
- `tests/correctness.rs` — differential fuzzing vs a brute-force oracle
  and vs `aho-corasick`, across all engines and semantics
- `examples/bench.rs` — the benchmark harness (`cargo run --release
  --example bench`)
