# Cache footprint and behavior analysis

Working-set estimates for SPARROW's compiled form, measured cache behavior
(residency sweeps + cachegrind simulation), and what both imply on modern
CPUs. Numbers below were produced by `examples/cache_probe.rs` on the
benchmark machine (virtualized Intel Xeon, 32 KB L1d / 1 MB L2 per core,
33 MB shared L3) and by `valgrind --tool=cachegrind` (32 KB L1, no
prefetcher simulated).

## 1. Working-set model

For a build with `C` cohorts, each with `k_c ≤ 4` sampled positions and
`m_c ∈ {1,2}` bucket planes, over `n` patterns of total length `L`:

| Component | Size | Touched |
|---|---|---|
| SIMD nibble tables | `Σ_c 32 · k_c · m_c` bytes (≤ 256 B/cohort) | every block — but register-resident (§2) |
| scalar byte tables | `Σ_c 256 · k_c · m_c` bytes (≤ 2 KB/cohort) | prelude (first 32 B) and tail only |
| bucket entries | `12 n` bytes (id + guard offset + guard byte) | only on filter candidates |
| pattern bytes | `L + 16 n` | only on guard-passing candidates |

Measured for the benchmark workloads (`Sparrow::memory_usage()` vs the
baselines' own `memory_usage()`):

| Pattern set | SPARROW total | SIMD-hot part | AC DFA | AC contiguous-NFA | packed Teddy |
|---|---|---|---|---|---|
| 16 shared-prefix routes | 1,584 B | **96 B** | 24,832 B | 2,052 B | 2,640 B |
| 64 random len-8 | 4,608 B | **256 B** | 526,336 B | 132,372 B | 5,120 B |

The structural difference: automaton size grows with *total pattern text*
(states × alphabet classes), while SPARROW's filter state is **constant in
`n`** — `n` only grows the verification-side entry/pattern arrays, which
are touched at the candidate rate the optimizer explicitly minimizes.

## 2. Register residency of the filter

The AVX-512 kernel needs `2 · k · m` table vectors plus ~7 working
registers: at the maximum configuration (k=4, two planes) that is 23 of 32
zmm registers — the **entire filter state lives in registers**, and the
steady-state loop performs zero table loads. The AVX2 kernel has 16 ymm
registers, so the maximum configuration spills a few tables to L1 (they
were just written, so these are ~5-cycle L1 hits, pipelined); typical
chosen configurations (k ≤ 3, one plane → 6 table vectors) fit fully.

Consequently the scan's only mandatory memory traffic is the haystack
itself: `k` unaligned loads per block that all land in the 1–2 lines the
block occupies (offsets differ by < 32), i.e. **1× compulsory streaming
traffic per cohort pass**, perfectly sequential and prefetch-friendly.

## 3. Measured residency behavior

Throughput while the (repeatedly scanned) haystack resides in successive
cache tiers:

| haystack | SPARROW routes | AC-DFA routes | SPARROW random | AC-DFA random |
|---|---|---|---|---|
| 16 KB (L1d) | 9.73 GB/s | 1.42 | 5.13¹ | 0.37 |
| 192 KB (L2) | 8.77 GB/s | 1.47 | 5.96 | 0.38 |
| 8 MB (L3) | 7.99 GB/s | 1.32 | 5.32 | 0.37 |
| 64 MB (DRAM) | 4.51 GB/s | 1.30 | 3.68 | 0.37 |

¹ small-buffer figures include per-call overhead (result alloc + sort);
the 192 KB row is the cleaner compute-bound number.

Two textbook signatures:

* **SPARROW is compute-bound until DRAM.** Throughput is flat within the
  cache hierarchy and drops ~45% only when the haystack streams from DRAM
  — it runs at the speed of its arithmetic, then at the speed of memory
  bandwidth. Faster memory (or hot data) directly translates to speed.
* **The DFA is latency-bound and residency-*insensitive*.** 0.37 GB/s
  regardless of where the haystack lives: its cost is the serialized
  dependent chain `state = table[state][byte]`. With the 514 KB table
  (16× L1d) each step is an L2-latency access; even the L1-resident
  25 KB table (routes row) caps at ~1.4 GB/s ≈ one L1 load-to-use chain
  per byte. No prefetcher can help a data-dependent walk.

## 4. Simulated miss counts (cachegrind, 4 MiB haystack, one pass)

4 MiB / 64 B lines = 65,536 compulsory read misses for streaming the
haystack. Measured D1 read misses (whole program, including build):

| Engine | D1 read misses | Above compulsory | Per haystack byte |
|---|---|---|---|
| SPARROW (random workload) | 102,320 | ~37 K (one-time build/haygen) | ~0 in scan loop |
| AC DFA (random workload) | **734,107** | ~669 K | **1 miss / 6.3 bytes** |
| SPARROW (routes) | 86,764 | ~21 K (build) | ~0 |
| AC DFA (routes) | 73,600 | ~8 K | ~0 (table fits L1) |

The AC number is the whole story of its flat 0.37 GB/s: two-thirds of a
million serialized L1 misses per 4 MiB scanned. SPARROW's scan adds
essentially nothing beyond the compulsory haystack stream. (Cachegrind
simulates no hardware prefetcher, so on real silicon the sequential
compulsory misses are largely hidden for SPARROW — but not the DFA's
dependent misses.)

## 5. Impacts on modern CPUs

* **L1d sizes (32–48 KB on Zen 4/5 and Golden/Redwood Cove, 128 KB on
  Apple M-series)** decide the DFA's fate: beyond ~10–50 KB of automaton
  (a few dozen patterns) it degrades to an L2-latency chain, and beyond
  ~1–2 MB to L3 latency. SPARROW's filter is size-independent of `n` and
  never leaves the register file — rule-set growth costs only candidate-
  rate, which the optimizer controls.
* **Latency vs throughput engines.** The DFA executes one dependent load
  per byte: ~5 cycles minimum (L1), ~14+ (L2) — 0.2–1.5 GB/s at 4 GHz, and
  out-of-order width is irrelevant. SPARROW's per-block work is `k·m`
  *independent* shuffle+AND chains: it exploits the 2 load ports + 1–2
  shuffle ports of current cores, so wide cores (Zen 5, Lion Cove,
  M-series) speed it up almost linearly while the DFA gains nothing.
* **Unaligned/split loads.** The offset-load kernel issues unaligned
  vector loads; a 32 B load splits a cache line every other block, a 64 B
  load nearly always. Modern cores handle split loads at ~1.1–1.5× cost
  (Zen 4 executes 512-bit loads as two 256-bit halves regardless), which
  the calibrated scan-cost constant already absorbs. No stores occur in
  the hot loop, so 4K aliasing and store-forwarding stalls cannot arise.
* **Bandwidth scaling and multicore.** From DRAM, one core sustains
  ~4.5 GB/s here; typical desktop/server sockets provide 30–100+ GB/s, so
  SPARROW scales near-linearly to ~8–20 cores on split haystacks before
  saturating memory — with a per-core cache footprint of a few KB. Two
  DFA threads, by contrast, each pin a private multi-hundred-KB table in
  L2 and still run at latency speed.
* **TLB.** Sequential scanning touches each 4 K page once (prefetchable,
  and next-page prefetchers on current cores hide most of it); huge pages
  are a minor win for SPARROW but a major one for multi-MB automata,
  whose random walks thrash the dTLB.
* **Length cohorts multiply streaming traffic.** `C` cohorts = `C` passes
  = `C×` DRAM traffic for LLC-exceeding haystacks. The cost model's
  per-cohort scan term charges for this; block-interleaving the cohorts
  into one pass (shared input loads, per-cohort table sets) is the natural
  future optimization if multi-cohort builds become common.
* **Verification locality.** Candidates re-read the just-scanned window
  (guaranteed L1-hot) and probe `buckets[b]` — currently a
  `Vec<Vec<Entry>>`, i.e. one pointer indirection per flagged bucket. For
  candidate-dense workloads a flattened CSR-style layout (offsets + one
  entry array) would remove a dependent load; entry+pattern data for
  realistic rule sets (≤ a few thousand patterns) stays L1/L2-resident.

## 5b. Footprint at scale

`examples/scale_bench.rs` (rare words from the Wikipedia corpus): SPARROW
compiles 16 K patterns into ~1 MB — filter tables are unchanged (the
SIMD-hot state stays 128–512 B; four planes at most), hashed-bucket
offset tables add 257 × 4 B per big bucket, and the rest is the patterns
themselves plus 8 B per entry. The AC DFA needs 26 MB for the same set
(every scanned byte then misses to L2/L3); the contiguous NFA is
comparable to SPARROW in size but pointer-chases. SPARROW's problem at
that scale is not memory, it is filter selectivity (see README scaling
table): the candidate rate, not the working set, is what degrades.

## 6. Reproducing

```
cargo run --release --example cache_probe            # footprints + sweeps
valgrind --tool=cachegrind ./target/release/examples/cache_probe cg2-ac
valgrind --tool=cachegrind ./target/release/examples/cache_probe cg2-sparrow
```
