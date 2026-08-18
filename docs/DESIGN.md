# SPARROW: Sparse Position Adaptive Rejection Over Rolling Windows

A SIMD multi-pattern matching algorithm with a new prefilter architecture and
provable guarantees. This document states the problem, surveys the existing
algorithm families, specifies SPARROW, proves its guarantees, and delimits
precisely what is provable (its properties) versus what is argued (its
novelty relative to the published families).

## 1. Problem

Given a set of patterns `P = {p_1, …, p_n}` over bytes and a haystack `H`,
report **every** occurrence `(i, q)` with `H[q .. q+|p_i|] = p_i`
(all patterns, all positions, overlaps included). This is the classic
multi-pattern (multi-literal) matching problem at the core of intrusion
detection (Snort/Suricata via Hyperscan), grep tooling (ripgrep via the
`aho-corasick` crate), and antivirus scanning. SPARROW additionally supports
ASCII-case-insensitive matching, single-byte wildcards in patterns
(`?`-globs), streaming input, and leftmost-nonoverlapping semantics.

## 2. State of the art

All high-performance solutions since ~2015 share one architecture: a cheap
**SIMD prefilter** that shortlists candidate positions, plus an exact
**verifier**. The families differ in *what the filter looks at* and *how
patterns share filter resources*:

| Algorithm | Filter inspects | Position choice | Bucket/grouping choice |
|---|---|---|---|
| Aho–Corasick (1975) | every byte via automaton | — (no filter) | — |
| Hyperscan **FDR** | last bytes of each literal, bit-packed "domains", shift-or | fixed: contiguous suffix | heuristic packing into bit-lanes |
| Hyperscan / `aho-corasick` **Teddy** (ICPP'21; NSDI'19) | first 1–4 bytes via PSHUFB low/high-nibble tables, 8 bucket bits per lane | **fixed: contiguous prefix positions 0..m** | heuristic (insertion order / size balancing) |
| **Harry** (ICPP'23) | whole literal, contiguous positional masks aggregated in SIMD | fixed: all contiguous positions | heuristic |
| **DFC** (NSDI'16) | 2-byte direct filter windows | fixed small windows | hash-based |
| Vectorized AC (SCPE 2019 etc.) | every byte; SIMD accelerates transitions | — | — |

Three structural constants stand out across every family:

1. **Contiguity.** The filter always inspects a *fixed, contiguous* run of
   pattern bytes (prefix, suffix, or whole literal). No published member of
   this family chooses *which* positions to inspect.
2. **Unmodeled filter loss.** The PSHUFB nibble trick approximates the byte
   set at each position by the *cross product* of its low- and high-nibble
   sets (e.g. bucket bytes `{0x41, 0x62}` also pass `0x42` and `0x61`).
   Every Teddy descendant pays this inflation, and none of them accounts
   for it when grouping patterns into buckets.
3. **Fixed resource spend.** How many filter bytes, how many buckets, and
   one filter for the whole pattern set — decided by the implementation,
   not by the workload.

These constants are exploitable weaknesses. Contiguous prefixes are
catastrophic when patterns share a common prefix ("GET /api/v1/…",
`\x7fELF`, TLS record headers…) — the *common case* in the workloads these
engines serve — and correlated bytes make heuristic resource choices
silently wrong.

## 3. The SPARROW design

SPARROW keeps the proven Teddy runtime shape — nibble shuffles producing
bucket bitmaps, ANDed and then verified — and turns every structural
constant into an optimizer decision:

* **Sparse sampled positions.** The filter inspects `k ≤ 4` positions
  `S = {s_1 < … < s_k}` chosen *anywhere* in the anchor window
  `[0, w)`, `w = min(min_i |p_i|, 32)` — not necessarily contiguous, not
  necessarily including position 0.
* **Optimized, closure-aware bucketing.** Patterns are assigned to 8 or 16
  buckets (1 or 2 SIMD "planes") minimizing an exact expected-cost
  objective that includes the nibble cross-product closure.
* **Cost-arbitrated resources.** The same objective — verification work
  plus a calibrated per-term scan cost — decides how many positions to
  sample, whether a second bucket plane pays for itself, and whether to
  split the pattern set into **length cohorts** (separate filters, each
  scanning the haystack once; the extra pass is priced in).
* **Empirical final selection.** Finalist configurations are re-scored by
  running the exact candidate filter over the corpus sample and counting
  real candidates — capturing byte correlations no closed-form model can
  (see §5.1).

### 3.1 Compiled form

For each plane, sampled position `s_j`, and bucket bit `b`:

* Each pattern presents a **class** at `s_j`: nibble sets
  `(Lo, Hi)` — singletons for an exact byte, both cases' nibbles under
  case-insensitivity, all nibbles for a wildcard byte.
* Shuffle tables `TL_j[ν]`, `TH_j[ν]` hold the OR of member bucket bits per
  nibble; the byte set actually recognized for bucket `b` is the closure
  `C_b(j) = { x : lo(x) ∈ Lo_b(j) ∧ hi(x) ∈ Hi_b(j) }`.
* Each bucket entry additionally stores a **guard probe**: the offset and
  byte of the model-rarest pattern byte *outside* the sampled positions.

Let `s_last = s_k` and `d_j = s_last − s_j`.

### 3.2 Runtime kernels (AVX2 and AVX-512BW)

The kernels use an **offset-load** structure: per block at offset `o`, for
each position `j` the input is loaded once *at `o − d_j`* (an unaligned
load), so every class mask is born already aligned to the anchor —
eliminating the carry registers, cross-lane `PALIGNR` shifting, and
per-block shift dispatch of Teddy-style kernels, and lifting the anchor
window from 16 to 32 bytes (the alignr idiom caps carry distances at 15).
Blocks start at offset 32; anchors `[0, 32)` and the tail run through the
scalar engine, which shares the same precomposed byte→bitmap tables.

Per block: `R_j^p = PSHUFB(TL_j^p, in_j & 0xF) & PSHUFB(TH_j^p, in_j >> 4)`
and `cand^p = ∧_j R_j^p` per plane `p`; a nonzero-byte test
(`VPCMPEQB`+`VPMOVMSKB` on AVX2, a single `VPTESTMB` mask on AVX-512)
gates extraction; each set bit `(b, t)` sends start `o + t − s_last` to the
verifier, which checks the guard probe before the full (case-/wildcard-
aware) comparison. Kernels are monomorphized over `(k, planes)` via const
generics, so position loops fully unroll. A portable scalar engine is
bit-identical in filter semantics on any architecture.

### 3.3 The optimizer

Let `π` be the corpus-estimated byte distribution and
`P_b(j) = Σ_{x ∈ C_b(j)} π(x)`. The **total model cost** per haystack byte
of a configuration (positions `S`, plane count `m`, assignment) is

```
T = Σ_{b nonempty} (c₀ + |bucket b|) · Π_{j=1..k} P_b(j)  +  γ · k · m
```

with `c₀ = 6` (measured per-candidate overhead: mispredicted branch, mask
store, bit loop) and `γ = 0.003 → 0.002` calibrated so one scan term
(~0.1 ns/byte) is priced in guard-probe units (~10 ns each). Section 4
proves the first term is the exact expected verification cost under i.i.d.
`π`.

Search: candidate position sets are all subsets (size ≤ 4) of the 9
most-discriminating window positions (all subsets with `exhaustive-search`),
scored by a greedy assignment; **finalists** — the best few overall, the
best set *of every size*, and always the contiguous prefix — are refined by
local search over both plane counts and re-scored by the **empirical
referee** (§5.1). The minimum total wins. Cohort partitioning (§3.4) then
compares the single filter against a length-banded split by summing each
side's total cost.

Greedy assignment inserts patterns in decreasing solo probability into the
bucket with the smallest exact marginal `ΔT`; refinement keeps moving
single patterns to strictly-better buckets until fixpoint.

### 3.4 Length cohorts

The anchor window is capped by the shortest pattern, so one short pattern
can blind the filter for everyone. When lengths are diverse, patterns are
banded (`[1,4), [4,8), [8,16), [16,32), [32,∞)`, small bands merged), each
band compiled into its own filter with its own window, and the split is
adopted iff the summed total cost — which charges each cohort its own scan
term, i.e. the extra haystack pass — beats the single filter.

## 4. Guarantees

**Lemma 1 (closure exactness).** For every bucket `b` and position `j`, the
byte set recognized by the compiled tables is exactly
`C_b(j) = {x : lo(x) ∈ Lo_b(j) ∧ hi(x) ∈ Hi_b(j)}`, where `(Lo_b, Hi_b)`
are the unions of member classes (case pairs and wildcards included).

*Proof.* Byte `x` passes iff bit `b` is set in both `TL_j[lo(x)]` and
`TH_j[hi(x)]`; by construction that holds iff `lo(x) ∈ Lo_b(j)` and
`hi(x) ∈ Hi_b(j)`. The scalar tables are defined as `TL_j[lo] & TH_j[hi]`,
hence identical. ∎

**Theorem 1 (zero false negatives).** Every occurrence `(i, q)` of every
pattern (under the configured semantics) is reported, by every engine.

*Proof.* Pattern `p_i` belongs to exactly one cohort; consider that
cohort's filter, bucket `b`, plane `p`. An occurrence at `q` means every
non-wildcard byte satisfies `byte_matches`, so for each sampled `s_j`,
`H[q+s_j] ∈ C_b(j)` (the pattern's class at `s_j` is contained in the
bucket union; Lemma 1). Consider anchor `t* = q + s_last < |H|` (since
`s_last < w ≤ |p_i|` and `q + |p_i| ≤ |H|`). Anchor `t*` is processed
exactly once: by the scalar prelude if `t* < 32`, by some SIMD block
`[o, o+B)` if `32 ≤ t* < blocks_end`, or by the scalar tail. In a block,
position `j`'s load at `o − d_j` puts `H[t* − d_j] = H[q + s_j]` in lane
`t* − o` (in bounds: `o ≥ 32 > d_j`), so bit `b` survives every ANDed term;
the scalar engine indexes the same bytes directly (`t* ≥ s_last ≥ d_j`).
The verifier computes start `t* − s_last = q ≥ 0`; the guard probe is at a
non-wildcard offset `g` with pattern byte `p_i[g]`, and `byte_matches
(H[q+g], p_i[g])` holds because the occurrence matches everywhere, so the
guard never rejects a true occurrence; the full comparison then succeeds
and `(i, q)` is emitted. Candidates are never dropped without
verification, and verification is exact for the configured semantics, so
no false positives are reported either. Streaming: a match ending in the
current chunk starts at most `max_len − 1` bytes earlier, which the carried
tail preserves; the `end > tail_len` filter reports each match exactly once
(ends inside the tail were reported by the previous push, by induction). ∎

**Theorem 2 (exact expected filter cost).** If haystack bytes are i.i.d.
with distribution `π`, the expected verification work per anchor is exactly
the first term of `T` in §3.3.

*Proof.* Bit `(b, t)` of `cand` is set iff `H[t − d_j] ∈ C_b(j)` for all
`j` (Lemma 1 plus kernel definition). The inspected offsets are pairwise
distinct because the `s_j` are, so under i.i.d. `π` the events are
independent: `Pr[bit (b,t)] = Π_j P_b(j)`. A set bit costs `c₀` overhead
plus `|bucket b|` probes. Linearity of expectation over buckets gives the
sum exactly. The closure `C_b(j)` — not the raw byte sets — appears, so
nibble inflation is fully accounted. ∎

**Theorem 3 (refinement terminates at move-optimality).** The local search
terminates and returns a partition where no single pattern can move to
another bucket with cost improvement `> ε` (`ε = 10⁻¹⁵`).

*Proof.* Each accepted move strictly decreases the objective by more than
`ε`; it is bounded below by 0 and starts finite, so at most `T₀/ε` moves
occur (a pass cap enforces this in practice). At exit, no improving move
exists — the definition of move-optimality. ∎

**Theorem 4 (never worse than the contiguous-prefix choice, under the
selection model).** The compiled configuration satisfies `T* ≤ T(prefix)`,
where both sides are scored by the same selection model (empirical referee
when enabled, closed-form otherwise) after the same greedy + refinement
pipeline.

*Proof.* The prefix set is unconditionally a finalist, undergoes the same
refinement and scoring, and the final configuration is the argmin over
finalists. (Cohort splitting can only replace this choice by a
strictly-cheaper one under the same model, preserving the inequality.) ∎

**Complexity.** Compile: polynomial — `O(#sets · n · buckets · k · 256)`
flops for the search plus `O(|corpus| · k)` per finalist for the referee.
Scan: `⌈|H|/B⌉ · O(k · m)` SIMD ops per cohort (B = 32 or 64) plus
expected verification work `|H| · T_verif`; worst case (adversarial
haystack) `O(|H| · Σ_b |b|)` probes — the same worst case as every
filter-based engine, with the expected case minimized within the searched
family.

## 5. What is new, and what "provable" means here

Mathematical novelty of an algorithm cannot be "proven" the way a theorem
can — it is a claim about the literature. What can be done honestly is
(a) prove the algorithm's properties (Theorems 1–4), and (b) exhibit the
mechanical deltas from every published family:

1. **Sparse, non-contiguous filter positions**, selected per pattern set by
   optimization. Teddy fixes positions `0..m`; FDR a contiguous suffix;
   Harry all positions; DFC fixed 2-byte windows. We are not aware of any
   published multi-pattern SIMD filter that optimizes *which* positions the
   filter inspects.
2. **Offset-load kernel structure.** Teddy-family kernels align class masks
   with cross-lane byte shifts carried between blocks; SPARROW instead
   re-loads the input at per-position offsets, deleting the carry state and
   shift dispatch and doubling the reachable window (16 → 32 bytes). (The
   trick is only useful *because* positions are sparse and few — a whole-
   literal matcher like Harry could not afford one load per position.)
3. **Closure-aware exact objective** (Lemma 1 + Theorem 2) driving bucket
   assignment — existing engines bucket heuristically and ignore nibble
   inflation.
4. **Cost-model-arbitrated resource allocation**: the number of sampled
   positions, 8-vs-16 buckets, and single-filter-vs-length-cohorts are all
   decided by one calibrated objective with a proven verification term —
   not hardcoded.
5. **Empirical finalist selection** (§5.1): final configurations are chosen
   by measured candidate counts on a corpus sample, not by a closed-form
   model.

The runtime primitive (nibble shuffles → bucket bitmaps → verify) is
deliberately *not* new — it is the hardware-proven Teddy shape, so gains
come from the provably-modeled offline choices plus the kernel
restructuring in (2).

### 5.1 Why an empirical referee (and not Markov-1)

The i.i.d. objective is exact under independence (Thm 2) but real data is
correlated, and the failure mode is systematic: positions shared by all
patterns *and* by the background (a common prefix like "GET /api/v1/")
look like `k` independent rare events (`p^k`) when they are one event
(`p`). A first-order Markov re-scorer was implemented and evaluated; it
narrows the gap (×5 on our log workload) but still underprices long-range
determinism by ~200×, because gap-bridging via `M^gap` diffuses through
transition branching. It was replaced by something strictly stronger and
cheaper: **run the exact compiled filter over the corpus sample and count
per-bucket candidates** — `O(|corpus| · k)` per finalist, exact for the
sample, and correct under *arbitrary* correlation. The closed-form
objective still powers the combinatorial search (it is exact per Thm 2 and
incrementally computable); the empirical referee makes the final call.
On the shared-prefix benchmark this single change moved the chosen
configuration from a correlation-blinded one (0.75 GB/s) to `[0, 1, 14]`
(6.2 GB/s).

## 6. Measured results

16 MiB haystacks, best of 5, one core (AVX2 + AVX-512BW machine; Rust
1.94, `-C lto`; this repo's `examples/bench.rs`; `aho-corasick` v1.1 —
its packed searcher is the reference Teddy implementation):

| Workload | SPARROW AVX-512 | SPARROW AVX2 | prefix ablation | AC DFA (ovlp) | Teddy (packed) |
|---|---|---|---|---|---|
| 16 English words / English text | 0.64 GB/s | 0.72 | 0.63 | 0.36 | 0.71 |
| 64 random len-8 / random bytes | **4.35 GB/s** | 3.30 | 4.26 | 0.38 | 3.38 |
| 16 shared-prefix routes / near-miss log | **6.20 GB/s** | 5.44 | 0.68 | 1.39 | 0.69 |
| 24 short words + 24 long signatures | 0.19 GB/s | 0.20 | 0.19 | 0.33 | 0.40 |

* **Shared prefixes** (the structural blind spot of prefix-anchored
  filters): **9× the reference Teddy** and 9× SPARROW's own prefix
  ablation — same kernel, different sampled positions, so the win is
  attributable entirely to the position optimizer + empirical referee
  (which chose `[0, 1, 14]`, k = 3).
* **Random patterns**: 4.35 GB/s vs Teddy's 3.38 — the 16-bucket plane
  mode and position choice give a 29% edge even on Teddy's home turf.
* **Match-heavy English**: parity with Teddy (0.72 vs 0.71) while
  reporting *all* overlapping matches (packed Teddy is leftmost-only).
* **The mixed workload is an honest loss**: 0.93M true matches
  (0.055/byte, dominated by "the"/"and") make every filter irrelevant and
  reward AC's amortized automaton. The model cost (0.57/byte) predicts
  exactly this; cohort splitting is correctly rejected because short-word
  verification dominates both ways.
* Match counts agree exactly with Aho–Corasick's overlapping iterator in
  all workloads (asserted in the harness, and in the differential fuzz
  tests against a brute-force oracle, across all three engines).

## 7. Limitations and extensions

* Selectivity is still bounded by the shortest cohort's window; a 1-byte
  pattern degrades its cohort to one position (as it would Teddy).
* The empirical referee is exact for the *sample*; a sample unlike the
  real traffic mis-ranks finalists (correctness is never affected —
  Theorem 1 is unconditional). The `corpus_sample` API exists precisely to
  close this gap.
* The scan-cost constants (`c₀`, `γ`) are calibrated for the benchmarked
  microarchitecture; a per-build microbenchmark could set them adaptively.
* AVX-512 VBMI (`VPERMB`, 64-entry tables → 6-bit classes) would shrink
  the closure inflation Lemma 1 quantifies; ARM NEON (`TBL` is a 16-byte
  shuffle) is a near-mechanical port of the scalar/AVX2 pair.
* Verification scans whole buckets; for very large rule sets a per-bucket
  hash or sorted-by-guard layout would sublinearize it.
