# SPARROW — sparse-position SIMD multi-pattern matching

SPARROW (*Sparse Position Adaptive Rejection Over Rolling Windows*) is a
multi-pattern (multi-literal) string matcher that keeps the hardware-proven
Teddy/Hyperscan SIMD kernel shape — PSHUFB nibble shuffles producing
8-bucket candidate bitmaps, then exact verification — but replaces the two
fixed structural choices every published engine hardcodes:

1. **Which pattern bytes the filter inspects.** Teddy inspects the first
   1–4 bytes; FDR the last bytes; Harry the whole literal — always a
   *contiguous* run. SPARROW samples a **sparse, optimizer-chosen set of
   ≤ 4 positions** anywhere in the patterns' common anchor window, at
   identical runtime cost.
2. **How patterns share SIMD buckets.** Existing engines bucket
   heuristically. SPARROW assigns patterns to buckets by minimizing an
   **exact expected-verification-cost objective** under a byte-distribution
   model — one that models the PSHUFB nibble cross-product inflation
   exactly, a filter loss all Teddy descendants pay but none account for.

Proved in [`docs/DESIGN.md`](docs/DESIGN.md): zero false negatives
(Thm 1), exactness of the cost objective under the i.i.d. byte model
(Thm 2), termination + move-optimality of the bucket refinement (Thm 3),
and model-dominance over the classic contiguous-prefix configuration
(Thm 4). The same document surveys the prior families (Aho–Corasick, FDR,
Teddy, Harry, DFC) and states precisely which mechanisms are new — and
what "provable" can and cannot mean for a novelty claim.

## Results (16 MiB haystacks, AVX2, one core, best of 5)

| Workload | SPARROW | SPARROW prefix-ablation | aho-corasick DFA | aho-corasick Teddy |
|---|---|---|---|---|
| English words / English text | 0.63 GB/s | 0.63 | 0.37 | 0.68 |
| 64 random len-8 / random bytes | 2.46 GB/s | 2.49 | 0.38 | 3.08 |
| shared-prefix routes / near-miss log | **2.80 GB/s** | 0.51 | 1.34 | 0.69 |

Shared prefixes (`GET /api/v1/…`, magic numbers, protocol headers) are the
common case in IDS/log scanning and are the structural blind spot of
prefix-anchored filters: there SPARROW is **4.1× faster than the reference
Teddy** (aho-corasick's packed searcher), and 5.5× faster than its own
prefix ablation — same kernel, different sampled positions, so the win is
attributable entirely to the optimizer.

## Usage

```rust
use sparrow::Sparrow;

let m = Sparrow::new(["GET /api/v1/users", "GET /api/v1/carts"]).unwrap();
for hit in m.find_all(log_bytes) {
    println!("pattern {} at {}..{}", hit.pattern, hit.start, hit.end);
}
```

Tune with the builder: `corpus_sample(&sample)` fits the byte model to your
traffic, `max_positions(k)` trades filter ops for selectivity,
`positions(&[0,1,2,3])` forces Teddy-style sampling (ablation),
`exhaustive_search(true)` widens the position search.

`find_all` reports **all** overlapping occurrences of all patterns, sorted;
an AVX2 kernel is selected at runtime with a bit-identical portable scalar
fallback.

## Repo layout

- `src/builder.rs` — the offline optimizer (byte model, closure-exact
  incremental cost, position search, greedy + local-search bucketing)
- `src/avx2.rs`, `src/scalar.rs` — runtime engines
- `docs/DESIGN.md` — survey, algorithm spec, theorems and proofs, honest
  novelty analysis, benchmark details
- `tests/correctness.rs` — differential fuzzing vs a brute-force oracle and
  vs `aho-corasick`, boundary/adversarial cases
- `examples/bench.rs` — the benchmark harness (`cargo run --release
  --example bench`)
