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
`aho-corasick` crate), and antivirus scanning.

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

Two structural constants stand out across every family:

1. **Contiguity.** The filter always inspects a *fixed, contiguous* run of
   pattern bytes (prefix, suffix, or whole literal). No published member of
   this family chooses *which* positions to inspect.
2. **Unmodeled filter loss.** The PSHUFB nibble trick approximates the byte
   set at each position by the *cross product* of its low- and high-nibble
   sets (e.g. bucket bytes `{0x41, 0x62}` also pass `0x42` and `0x61`).
   Every Teddy descendant pays this inflation, and none of them accounts
   for it when grouping patterns into buckets.

Both constants are exploitable weaknesses. Contiguous prefixes are
catastrophic when patterns share a common prefix ("GET /api/v1/…",
`\x7fELF`, TLS record headers…), which is the *common case* in the very
workloads these engines serve. And nibble inflation silently multiplies the
candidate rate when heuristic bucketing mixes patterns whose nibble sets
combine badly.

## 3. The SPARROW design

SPARROW keeps the proven Teddy runtime shape — nibble shuffles producing
8-bucket bitmaps, shifted and ANDed, then verified — and replaces both
structural constants with an offline optimizer:

* **Sparse sampled positions.** The filter inspects `k ≤ 4` positions
  `S = {s_1 < … < s_k}` chosen *anywhere* in the anchor window
  `[0, w)`, `w = min(min_i |p_i|, 16)` — not necessarily contiguous, not
  necessarily including position 0.
* **Model-driven, closure-aware optimization.** Both the position set and
  the assignment of patterns to the 8 buckets are chosen to minimize the
  *exact* expected verification cost per haystack byte under a byte
  distribution `π` (estimated from a user-supplied corpus sample, with
  Laplace smoothing), where the cost model includes the nibble
  cross-product closure *exactly* rather than ignoring it.

### 3.1 Compiled form

For each sampled position `s_j` and bucket `b ∈ {0..7}`:

* `Lo_b(j) = { lo(p[s_j]) : p ∈ bucket b }`, `Hi_b(j)` analogously
  (`lo(x) = x & 15`, `hi(x) = x >> 4`).
* Shuffle tables `TL_j[ν] = Σ_b [ν ∈ Lo_b(j)] · 2^b` and
  `TH_j[ν] = Σ_b [ν ∈ Hi_b(j)] · 2^b` (16 bytes each).
* The **closure class** actually recognized is
  `C_b(j) = { x : lo(x) ∈ Lo_b(j) ∧ hi(x) ∈ Hi_b(j) } ⊇ {bytes at s_j}`.

Let `s_last = s_k` and carry distances `d_j = s_last − s_j ∈ [0, 15]`.

### 3.2 Runtime kernel (AVX2)

Per 32-byte block at offset `o`, with per-position carry registers `prev_j`
(zero-initialized — "nothing matches before the buffer"):

1. `R_j = PSHUFB(TL_j, in & 0x0F) & PSHUFB(TH_j, (in >> 4) & 0x0F)` —
   byte `t` of `R_j` is the bitmap of buckets whose closure class at
   position `j` contains `H[o+t]`. (2 shuffles + 3 logic ops.)
2. `A_j = shift_carry(prev_j, R_j, d_j)` — byte `t` becomes `R_j` evaluated
   at `t − d_j`, borrowing the previous block's tail
   (`VPERM2I128` + `VPALIGNR`, the standard cross-lane byte-shift idiom).
3. `cand = A_1 & … & A_k`; `prev_j ← R_j`.
4. If `cand ≠ 0` (one `VPCMPEQB` + `VPMOVMSKB` test): for each nonzero byte
   `t` and each set bit `b`, the verifier memcmp-checks every pattern of
   bucket `b` at start `o + t − s_last`.

Total: `5k + k + 2 ≈ 26` cheap SIMD ops per 32 bytes at `k = 4`, identical
to a 4-byte Teddy — sparsity is free at runtime; only the *choice* of
positions moved offline. A portable scalar engine uses the precomposed
256-entry byte→bitmap tables (`TL_j[lo] & TH_j[hi]`), giving bit-identical
filter semantics on any architecture.

### 3.3 The optimizer

Let `π` be the model distribution and
`P_b(j) = Σ_{x ∈ C_b(j)} π(x)` the closure-class probability. Define the
objective

```
E(S, assignment) = Σ_{b nonempty} (c₀ + |bucket b|) · Π_{j=1..k} P_b(j)
```

(`c₀ = 2` models fixed per-candidate overhead in units of one pattern
comparison). Section 4 proves `E` is the exact expected verification cost
per haystack byte under i.i.d. `π`.

* **Position search:** positions are ranked by the rarity of the pattern
  bytes they expose (`Σ_i π(p_i[s])`); all subsets of size `≤ k` of the top
  9 (or of the whole window with `exhaustive-search`) are scored by a
  greedy assignment, *always including the contiguous prefix
  `{0..k−1}` as a baseline*.
* **Greedy assignment:** patterns are inserted in decreasing solo
  probability; each goes to the bucket with the smallest exact marginal
  `ΔE`. Closure probabilities are maintained incrementally in `O(16)` per
  update via row/column partial sums of the 16×16 nibble matrix, so the
  marginal including inflation is exact, not approximated.
* **Refinement:** the best few position sets (and always the prefix
  baseline) get a local search that keeps moving single patterns to
  strictly-better buckets until fixpoint; the overall minimum wins.

Compilation is polynomial and takes well under a second for typical rule
sets (hundreds of patterns).

## 4. Guarantees

**Lemma 1 (closure exactness).** For every bucket `b` and position `j`, the
byte set recognized by the compiled tables is exactly
`C_b(j) = {x : lo(x) ∈ Lo_b(j) ∧ hi(x) ∈ Hi_b(j)}`.

*Proof.* Byte `x` passes iff bit `b` is set in both `TL_j[lo(x)]` and
`TH_j[hi(x)]`; by construction that holds iff `lo(x) ∈ Lo_b(j)` and
`hi(x) ∈ Hi_b(j)`. The scalar tables are defined as
`TL_j[lo] & TH_j[hi]`, hence identical. ∎

**Theorem 1 (zero false negatives).** Every occurrence `(i, q)` of every
pattern is reported, by both engines.

*Proof.* Let pattern `p_i` sit in bucket `b` and occur at `q`, i.e.
`H[q + s] = p_i[s]` for all `s < |p_i|`. Consider anchor `t* = q + s_last`.
Since `s_last < w ≤ |p_i|` and `q + |p_i| ≤ |H|`, we get `t* < |H|`, so
`t*` lies in some processed block or the scalar tail (the two ranges
partition `[0, |H|)`). For each `j`: `t* − d_j = q + s_j ≥ 0`, and
`H[q + s_j] = p_i[s_j] ∈ C_b(j)` by Lemma 1 and table construction, so bit
`b` survives every ANDed term (the zero-initialized carry only affects
`t < d_j`, and `t* ≥ s_last ≥ d_j`). Hence `cand[t*]` has bit `b`; the
verifier computes start `t* − s_last = q ≥ 0`, compares
`H[q .. q+|p_i|]` with `p_i`, which succeeds, and emits `(i, q)`. The
scalar engine evaluates the same tables at the same anchors. No candidate
is ever dropped without verification; verification is an exact memcmp, so
no false positives are reported either. ∎

**Theorem 2 (exact expected filter cost).** If haystack bytes are i.i.d.
with distribution `π`, then for any anchor `t ≥ s_last`, the expected
number of pattern comparisons plus weighted candidate overhead per byte is
exactly `E(S, assignment)` from §3.3.

*Proof.* Bit `(b, t)` of `cand` is set iff `H[t − d_j] ∈ C_b(j)` for all
`j` (Lemma 1 plus kernel definition). The inspected offsets
`t − d_j = t − s_last + s_j` are pairwise distinct because the `s_j` are,
so under i.i.d. `π` the events are independent:
`Pr[bit (b,t)] = Π_j P_b(j)`. A set bit costs `c₀` overhead plus `|bucket
b|` comparisons (the verifier scans the whole bucket). Linearity of
expectation over buckets gives `E` exactly. Note the closure `C_b(j)` — not
the raw byte sets — appears, so nibble inflation is fully accounted. ∎

**Theorem 3 (refinement terminates at move-optimality).** The local search
terminates and returns a partition where no single pattern can move to
another bucket with cost improvement `> ε` (`ε = 10⁻¹⁵`).

*Proof.* Each accepted move strictly decreases `E` by more than `ε`; `E` is
bounded below by 0, and its initial value is finite, so at most `E₀/ε`
moves occur (a pass cap enforces this in practice). At exit, no improving
move exists — the definition of move-optimality. ∎

**Theorem 4 (never worse than the contiguous-prefix choice, under the
model).** The compiled configuration satisfies
`E* ≤ E(prefix, refined-greedy(prefix))`, i.e. under the model SPARROW's
choice is never worse than a Teddy-style prefix filter compiled through the
same pipeline.

*Proof.* The prefix set is unconditionally included among the finalists
that undergo greedy assignment + refinement, and the final configuration is
the argmin over finalists. ∎

**Complexity.** Compile: `O(#sets · n · 8 · 16k)` flops plus refinement;
search space is `O(Σ_{j≤4} C(9, j)) = 255` sets by default. Scan:
`⌈|H|/32⌉ · O(k)` SIMD ops plus expected `|H| · E` verification work;
worst-case (adversarial haystack) `O(|H| · Σ_b |b|)` comparisons — the same
worst case as every filter-based engine, with the expected case provably
minimized within the searched family.

## 5. What is new, and what "provable" means here

Mathematical novelty of an algorithm cannot be "proven" the way a theorem
can — it is a claim about the literature. What can be done honestly is
(a) prove the algorithm's properties (Theorems 1–4 above), and (b) exhibit
the precise mechanical deltas from every published family:

1. **Sparse, non-contiguous filter positions.** Teddy fixes positions
   `0..m`; FDR fixes a contiguous suffix domain; Harry uses all positions
   contiguously; DFC uses fixed 2-byte windows. SPARROW selects an
   arbitrary `k`-subset of the anchor window per pattern *set* (not per
   pattern), which is what lets it keep the one-shuffle-per-position
   runtime while escaping shared-prefix blindness. We are not aware of any
   published multi-pattern SIMD filter that optimizes *which* positions the
   SIMD filter inspects.
2. **Closure-aware exact objective.** No Teddy descendant models the
   PSHUFB nibble cross-product inflation; SPARROW's objective computes it
   exactly (Lemma 1 + Theorem 2) and both the position choice and the
   bucket partition minimize it.
3. **Bucket assignment as optimization with guarantees.** Existing engines
   bucket heuristically (insertion order, size balancing). SPARROW's
   greedy + local-search assignment carries Theorem 3's move-optimality
   and Theorem 4's dominance over the classic configuration.

The runtime kernel itself (nibble shuffles, shifted ANDs, verify) is
deliberately *not* new — it reuses the hardware-proven Teddy shape, so all
gains come from the provably-modeled offline choices.

## 6. Measured results

16 MiB haystacks, best of 5, one core (AVX2; Rust 1.94, `-C lto`,
this repo's `examples/bench.rs`; `aho-corasick` v1.1 as the baseline —
its packed searcher is the reference Teddy implementation):

| Workload | SPARROW | SPARROW (prefix ablation) | AC DFA (overlapping) | Teddy (packed, leftmost) |
|---|---|---|---|---|
| 16 English words / English text | 0.63 GB/s | 0.63 | 0.37 | 0.68 |
| 64 random len-8 / random bytes | 2.46 GB/s | 2.49 | 0.38 | 3.08 |
| 16 shared-prefix routes / near-miss log | **2.80 GB/s** | 0.51 | 1.34 | 0.69 |

* The shared-prefix workload is the structural weakness the design targets:
  SPARROW is **4.1× faster than the reference Teddy** and 5.5× faster than
  its own prefix ablation — the position optimizer, not kernel tuning,
  accounts for the whole win (the ablation shares every line of runtime
  code). The model predicted the gap: expected cost 4.2·10⁻⁶ vs
  2.2·10⁻⁵ per byte.
* On neutral workloads SPARROW matches the far more engineering-tuned
  Teddy within ~20% while additionally reporting *all* overlapping matches
  (Teddy's packed API is leftmost-first only).
* Match counts agree exactly with Aho–Corasick's overlapping iterator in
  all workloads (asserted in the harness, and in the differential fuzz
  tests against a brute-force oracle).

## 7. Limitations and extensions

* The anchor window is bounded by the shortest pattern; a 1-byte minimum
  pattern degrades the filter to one position (as it does Teddy). Length
  cohorts (multiple SPARROW instances by length class) are the natural
  extension.
* Theorem 2's expectation is exact under the i.i.d. model `π`; real data is
  correlated. The corpus-sample API narrows the gap empirically; a Markov-1
  objective is a straightforward extension (the closure algebra is
  unchanged; only the probability terms generalize).
* AVX-512 doubles block width; VBMI's `VPERMB` would allow 64-entry
  (6-bit) classes, shrinking the closure inflation Lemma 1 quantifies.
* Streaming (matches across buffer boundaries) needs the standard carry of
  `min_len − 1` history bytes plus the existing `prev_j` registers.
