# SPARROW roadmap: enhancements and extensions, critically assessed

A skeptical survey of what else this design space offers, written after
the two-prong build landed (`src/dense.rs`, `builder::build_routed`).
Every item: the idea, why it matters *here*, expected gain (with a number
where one can be justified from measurements in this repo), risk, effort
(S/M/L), and a verdict (**do / maybe / skip**). Numbers are Apple M-series
unless stated; the baseline is the table in README.md.

Two facts from the measurements shape everything below:

* The sparse kernel is already memory/throughput-flat (4.5 GB/s on random
  data; flat from L1 to DRAM in `docs/CACHE.md`). Its losses are all on
  **match-dense** inputs, where per-candidate and per-match costs rule.
* The dense lane is **instruction-throughput bound** (≈0.35 ns/byte/lane),
  and everything we tried that put a call or an unpredictable branch in a
  hot loop cost 1.5–3×. Output-path costs (sort, page faults, mispredicts)
  were larger than any kernel micro-optimisation.

---

## 1. Kernel level

### 1.1 Vectorised Shift-Or (NEON / AVX2 lanes)
Hold two (NEON) or four (AVX2) 64-bit lanes per vector: one table load per
segment instead of one per lane, one AND/OR per vector. The 128-bit shift
by one is ~3 ops on NEON (`shl`, `ushr`/`ext`, `orr`) vs 1 per scalar
lane, so per-segment-step cost goes from 2 lanes × (1 ld + 3 ALU) = 8 to
1 ld + ~5 ALU = 6. Event recording also halves (4 `st1q` instead of 8
stores). Expected: 20–30% on the 2-lane dense kernel (0.68 → ~0.5
ns/byte), more on AVX2 with 4 lanes. Lands in `src/dense.rs` as
`scan_chunk2_neon` behind the same `Event` interface.
Risk: register pressure with 4 segments × 1 vector state is fine; the
shift-with-carry across the 64-bit halves must be exact (bit 63 → bit 64
carries *within* a lane only if a pattern straddles — it never does, lanes
are independent, so no carry is needed: just `shl #1` per 64-bit element).
That removes the 3-op shift concern entirely: `vshlq_n_u64` is one op.
Effort: S–M. **Verdict: do** — cheap, and it is the only lever left on
the dense lane's scan cost.

### 1.2 AVX-512 VBMI 6-bit classes (`VPERMB`, 64-entry tables)
DESIGN.md §7 already names it. Closure inflation (Lemma 1) is the
nibble cross-product; with 6-bit/2-bit splits the cross product shrinks
for ASCII-heavy sets (letters differ mostly in low 6 bits). Gain depends
on how much candidate rate is closure-induced; the empirical referee can
measure that today by comparing `expected_candidates` against exact
byte-set probability. On the four bench workloads the referee already
shows positions beating closure loss; I'd guess < 10% on candidate rate
for ASCII sets, nothing for binary. Requires VBMI hardware (Ice Lake+).
Effort: M. **Verdict: maybe** — only after measuring closure loss on a
real rule set (Snort) shows it above ~20%.

### 1.3 Wider windows (> 32 bytes)
Offset loads cap the anchor window at 32. Patterns are routinely longer
(signatures, URLs) and the optimizer has shown it wants far positions
(`[0,1,14]`, `[9]`). Two loads per position (block at `o − d_j` with
`d_j ≤ 64`) cost one extra load per term on AVX2/NEON only if `d_j > 32`;
otherwise the kernel is unchanged. Candidate-rate gain is zero unless the
far bytes are more discriminating — true exactly for shared-prefix sets
with long common prefixes (> 32 bytes: some Snort content strings, HTTP
headers). Lands in `builder.rs` (`MAX_WINDOW`), `neon.rs`/`avx2.rs`
(`FIRST` = 64). Effort: S. **Verdict: do**, gated by the cost model
(a position with `d_j > 32` pays one extra term).

### 1.4 2-byte (q-gram) nibble tables
Teddy "fat" / Hyperscan FDR use 2-byte domains. A 16-bit index needs a
64 K-entry table per position → L2-resident, which breaks the
register-residency property CACHE.md is built on. The sparse optimizer
already achieves most of what a 2-gram buys by choosing *non-adjacent*
positions (two independent 1-byte samples ≈ one 2-gram in entropy, often
better). **Verdict: skip** — it trades the crate's best property for a
gain the optimizer mostly already captures.

### 1.5 Harry / DFC-style multi-stage filters
Second-stage filter between SIMD bitmap and full verify: e.g. a 1-byte
hash probe of 2–3 more positions before touching bucket entries. The
guard probe *is* a one-entry second stage; it rejects most false
candidates with one load. A bucket-level second stage would matter when
buckets are deep (§3), not at 16–64 patterns. Effort: M. **Verdict:
maybe**, folded into the scale work in §3.

### 1.6 Branchless verification
`verify_at` loops over bucket entries with a branch per entry. On
match-dense inputs these branches are unpredictable (which entry hits).
Converting to "compute all guard checks, popcount, then verify only
flagged entries" makes the common 1-entry case branch-free. Expected:
the 16-word English set has ~1 candidate per 30 bytes; if each costs one
mispredict (~4 ns) that is 0.13 ns/byte of a 1.0 ns/byte total — ~10%.
Lands in `lib.rs::verify_at`. Effort: S. **Verdict: do**, but measure
first with the same ablation method used for the dense lane.

### 1.7 Output-path costs still lurking
Found so far: `sort_unstable` (0.95 ns/byte), page faults on fresh
`Vec`s (0.4), mispredicts (0.3+). Still there: (a) `find_all` on 900 K
matches allocates a 22 MB `Vec<Match>` (24-byte `Match`) — a callback API
(`for_each_match(hay, FnMut(Match))`) or a compact 16-byte match (`u32`
pattern, `u32` len, `u64` start) halves that traffic; (b) the stable
`sort()` allocates n/2 scratch every call; a cheap `is_sorted` check
first skips it in the common single-cohort no-dense case; (c) streaming's
`tail` copy (see 1.8). Expected: 5–15% on match-dense rows, zero
elsewhere. Effort: S. **Verdict: do** (a) and (b) — the bench should also
report a count-only number so engine comparisons stop including
allocation.

*Status:* (a) done — `scan_with`/`count_all` landed (W4 0.75 → 1.02 GB/s
counting); the compact 8-byte internal run representation was implemented
and measured a wash (±1% on W4/W1 across three runs — run-buffer store
traffic is not the bottleneck; the event store and hit mispredicts are),
so it was dropped. (c) done, see §1.8.

### 1.8 Streaming kernel without the tail copy
`StreamScanner::push` concatenates `tail + chunk` into a fresh `Vec`
every call — an allocation and a full copy per chunk, which for
small chunks (packets, ~1.5 KB) dominates. Both kernels only need the
last `max_len − 1` bytes: scan the tail region via the scalar path over a
small stack buffer of `2·max_len`, then scan `chunk` directly with the
SIMD kernel and offset the results. Expected: for 1.5 KB chunks, removes
a malloc + memcpy per chunk (~200–400 ns vs ~300 ns of scanning) — up to
2× on packet-sized streams; nothing on large chunks. Lands in
`lib.rs::StreamScanner`. Effort: S–M. **Verdict: do** — streaming is
the IDS use case the README sells.

*Status:* done. Chunk scanned in place; only `tail + max_len − 1` bytes
are stitched and rescanned; the two sorted groups are merged, not
re-sorted. Measured (16 MiB english, 16-word set): sparse-routed
1500-byte chunks 3.0 → 6.3 GB/s, 64 KB 5.2 → 20.5; dense-routed 64 KB
0.73 → 0.87; dense-routed 1500-byte chunks −10% (two scans vs one —
accepted, documented on `StreamScanner::push`).

---

## 2. Model and optimizer

### 2.1 Calibrate sparse vs dense cost scales (the W1 mis-route)
The router picked dense on the 6–11-byte English set because the sparse
model predicted 2.7 ns/byte and measured 1.0. The sparse constants
(`CANDIDATE_OVERHEAD = 6`, `SCAN_COST_PER_TERM = 0.002`) were fit on x86
kernels; the NEON verify path is evidently cheaper per candidate.
Minimal fix: per-engine constants (`Engine::Neon` → `CANDIDATE_OVERHEAD
≈ 2.5`) measured by `examples/dense_probe.rs`-style ablation. Effort: S.
Risk: still a hand-fit; see 2.2 for the principled fix. **Verdict: do**
as a stopgap, because a 5% regression on a flagship row is embarrassing.

### 2.2 Empirical build-time referee that *times* candidates
Position selection already has an empirical referee (candidate counts on
the corpus sample). Routing does not. Build the top-2 or top-3 total
configurations (sparse-only, best two-prong) and time each on the corpus
sample (64 KB → ~50 µs per candidate); pick the fastest. Variance: at
64 KB, run-to-run jitter is a few %, smaller than the 2.7× model error.
Cost: +0.2–1 ms build time — acceptable for a build-once matcher, and
can be gated (`Builder::timed_referee(bool)`). Lands in
`builder::build_routed`. Effort: M. **Verdict: do** — it is the same
philosophy (§5.1 of DESIGN.md) applied one level up, and it makes every
future kernel change self-calibrating.

### 2.3 Per-pattern routing instead of length thresholds
Route each pattern by its own marginal cost: a long but extremely common
pattern (`"GET /"` repeated, `"\x00\x00\x00\x00"`) belongs in the dense
lane; a short but rare one (`"\xDE\xAD"`) belongs sparse. Exact: 2^n.
Practical: greedy — start sparse-only, repeatedly move the pattern with
the highest per-pattern candidate contribution (`empirical_verif_cost`
already counts per bucket; extend to per pattern) into the lane while
total cost falls. Gain: none on the current four workloads (length is
the right proxy there); real on binary/rule sets with common long
literals. Effort: M. **Verdict: maybe** — after a real rule-set corpus
(§6) shows a case.

### 2.4 Markov / richer statistics for position selection
DESIGN.md §5.1 already reports the honest negative result: Markov-1
re-scoring lost to the empirical referee. The remaining gap is the
*candidate set enumeration* (`candidate_position_sets` keeps the top 9
positions by i.i.d. rarity before the referee sees them). If the true
best set uses a position ranked 10th i.i.d. but highly discriminating
empirically, it is never tried. Fix: rank the pool by *empirical*
per-position marginal candidate rate (one corpus pass per position,
O(w·|corpus|)), not by model probability. Effort: S. **Verdict: do** —
small, and it directly widens the referee's reach.

### 2.5 Guard selection improvements
Guard = model-rarest unsampled byte. Two cheap upgrades: choose the guard
*empirically* (byte whose observed false-candidate rejection rate on the
corpus is highest — it can be read off the same pass as 2.4), and allow
a 2-byte guard (`u16` compare) when one byte isn't selective. Expected:
verification cost is `CANDIDATE_OVERHEAD + entries`; guards cut the
`entries` part, which is ≤ 1/7 of the sum at 16–64 patterns. Gain ≈ 5%
there, more at scale (§3). Effort: S. **Verdict: maybe**.

### 2.6 Adaptive re-optimisation at scan time
Observe candidate rate during the scan; if it exceeds the model's
prediction by some factor, swap to a second precompiled configuration
(e.g. sparse → dense, or different positions). Theoretically attractive,
practically: it requires compiling 2+ configurations, a sampling counter
in the hot loop (cheap if per block), and a switch that preserves
streaming state (trivial for Shift-Or, needs the tail for sparse). The
real question is whether the corpus sample misrepresents traffic often
enough to pay for this. No evidence yet. Effort: L. **Verdict: skip** for
now; `corpus_sample` is the documented answer. Revisit if users report
mis-routing on shifting traffic.

---

## 3. Scale: 1 K – 100 K patterns

### 3.1 Where the bucket structure breaks
16 buckets → at 1 K patterns 64 entries per bucket, at 100 K 6 K entries.
Every candidate bit then costs a linear walk of the bucket
(`verify_at`), and the *filter's* candidate rate also degrades: a bucket
with 64 members has near-full nibble closure at every position, so the
filter passes almost everything. Teddy has the same wall (aho-corasick
caps packed Teddy at a few hundred literals and falls back to AC). So the
honest statement: **SPARROW as-is is a < 500-pattern engine**, like
Teddy. Claiming otherwise needs §3.2–3.4.

### 3.2 More planes / wider bucket bitmaps
Planes are `MAX_PLANES = 2` (16 buckets). 8 planes = 64 buckets costs 4×
the shuffle work per position (the kernel's ALU budget: at 4 positions ×
8 planes the sparse kernel stops being memory-flat). Buys a 4× deeper
pattern set at the same candidate rate. Lands in `builder.rs`
(`MAX_PLANES`), kernels (`cand[]` array). Effort: S–M. **Verdict: do** up
to 4 planes, let the cost model choose — it already arbitrates 1 vs 2.

### 3.3 Hashed verification (per-bucket hash on sampled bytes)
Instead of walking the bucket, hash the `k` sampled bytes at the
candidate anchor (they are already loaded) into a per-bucket open-
addressing table of pattern ids. Turns verification into O(1 + true
collisions). At 100 K patterns and 64 buckets that is the difference
between 1.5 K-entry walks and a single probe. This is the DFC/FDR
"confirm" structure. Effort: M. **Verdict: do** if scale is a goal — it is
the single change that moves the ceiling from hundreds to tens of
thousands of patterns.

### 3.4 Hierarchical filters
First-stage sparse filter with few positions selects a *sub-filter*
(another bucket set with its own positions) rather than a bucket of
patterns. Equivalent to a 2-level trie over sampled bytes. Gains over 3.3
only when the sampled-byte hash collides heavily (highly repetitive rule
sets). Effort: L. **Verdict: skip** until 3.3 is measured on a real rule
set.

### 3.5 Memory footprint vs AC at scale
Today: SPARROW 1.6–4.6 KB vs AC-DFA 25–526 KB. At 100 K patterns an AC
DFA is hundreds of MB (or a contiguous NFA with pointer-chasing); SPARROW
with 3.3 would be: filter tables (unchanged, < 1 KB), hash tables (~8
bytes × patterns ≈ 800 KB), patterns themselves. That is a 100×
footprint advantage that *grows* with scale — the most compelling
scale story, but only credible with a measured 10 K-pattern run
(`examples/cache_probe.rs` already has the harness). Effort: S once 3.3
exists. **Verdict: do** as the headline of the scale work.

---

## 4. Semantics and extensions

### 4.1 Regex literal prefilter (Hyperscan-style)
Extract required literals from regexes, SPARROW them, run the regex only
around candidate positions. This is how Hyperscan and `regex`'s
`Teddy` integration get their speed; SPARROW's shared-prefix win
(`GET /api/v1/…` 9 GB/s vs 1.2) maps directly onto IDS rule literals,
which share prefixes pathologically. Integration point: a
`find_all`-like API returning candidate *windows* and an adapter for
`regex-automata`'s `Prefilter` trait. Effort: M (the trait adapter is
small; literal extraction exists in `regex-syntax`). **Verdict: do** —
it is the route to actual users, and the design's thesis was written for
exactly this workload.

### 4.2 Case folding beyond ASCII / UTF-8 awareness
Unicode case folding is multi-byte and length-changing (`ß` ↔ `SS`) — it
cannot be expressed as per-position byte classes. The honest scope:
*byte-exact* matching on UTF-8 bytes already works (UTF-8 is
self-synchronising, so byte matches are codepoint matches); simple
folding for 2-byte scripts (Latin-1 supplement, Cyrillic) is expressible
as byte classes at fixed positions and would fit `class_of`. Full
Unicode folding: **skip**, delegate to the regex layer (4.1).
Effort for the 2-byte-script subset: S. **Verdict: maybe**.

### 4.3 Byte-class patterns (`[0-9]`, `\x00-\x1f`)
`class_of` already handles a single wildcard byte and case pairs; a
general per-position byte set is the same `(Lo, Hi)` machinery with a
caveat: classes with spread nibbles (e.g. `[0-9A-Fa-f]`) inflate closure
badly and the optimizer will (correctly) avoid sampling those positions.
The dense lane handles arbitrary classes *exactly* (`B[c]` is built from
`byte_matches`). So: classes are cheap in the dense lane, expensive in
the filter, and the router already decides. API: `Pattern` type with
per-position sets. Effort: S–M. **Verdict: do** — it is the feature that
makes 4.1 useful (regex literal extraction yields classes constantly).

### 4.4 Approximate matching (Hamming / edit distance k) via the bit-parallel lane
Shift-Or extends to Wu–Manber approximate matching with `k+1` state
words per pattern (one per error count) and 3 extra ops per level. It is
the classic use of this machinery and fits `dense.rs` naturally (lane
capacity divides by `k+1`). The sparse filter cannot do it (a sampled
byte may be the one with the error) — unless positions are chosen so
that any `k` errors leave `≥ 1` sampled-position set intact, i.e. sample
`k+1` disjoint position sets and OR their candidates: a real and
publishable extension of the optimizer (it is the q-gram lemma in
sampled form). Effort: M for dense-only, L for the sparse extension.
**Verdict: maybe** — dense-only is a contained feature; the sparse
version is a paper, not a patch.

### 4.5 Leftmost / longest semantics natively in the kernel
Today `find_leftmost_nonoverlapping` post-filters `find_all`. On
match-dense inputs that materialises ~2–5× more matches than it
returns. A native version stops verification at the first match per
anchor and skips the filter forward past the match end (`t = end`),
which on match-heavy rows also skips *scanning*. Expected on W4: the
928 K overlapping matches collapse to ~300 K, removing maybe 30% of
output cost and some scan. Lands in kernels as a `Mode` const generic.
Effort: M. **Verdict: do** if the regex integration (4.1) needs it — it
does (regex semantics are leftmost).

### 4.6 Counting / first-match early exit
`count(hay)`, `is_match(hay)`, `find_first(hay)`: count avoids the
output path entirely (the sort and allocation were ~50% of match-dense
time); `is_match` can return at the first verified candidate. Trivial
with the `Event`/callback plumbing. Effort: S. **Verdict: do** — and
report `count` in the bench as the allocation-free number.

### 4.7 Multi-threaded chunk scanning
Both kernels are chunk-parallel with `max_len − 1` overlap; a
`find_all_par(hay, threads)` is a `chunks` + join + run merge. Scaling is
near-linear until DRAM bandwidth (~100 GB/s on M-series; the sparse
kernel at 9 GB/s/core saturates around 10 cores). Risk: none technical;
the question is whether the crate should pull in a thread pool
dependency (no — take a `&[&[u8]]` of chunks and let the caller
parallelise, or feature-gate `rayon`). Effort: S. **Verdict: do** as a
documented pattern + example, not a dependency.

---

## 5. Theory

### 5.1 Lower bound on candidate rate from position entropy
Claim to prove: for any filter that inspects `k` positions with nibble
closure and `B` buckets, the expected candidate rate is at least
`Σ_b Π_j P(C_b(j))` under the i.i.d. model — and SPARROW's optimizer
attains the minimum over position sets *within the enumerated pool*.
The first part is already what the objective computes (it is an
identity, not a bound). A real theorem would bound the rate *below* for
**any** position choice in terms of the patterns' per-position entropy
`H_j`: candidate rate `≥ 2^{-Σ_{j∈S} H_j}`-ish by a Fano-type argument
— showing the optimizer's choice is within a constant of optimal when
the corpus is i.i.d. Effort: M (a day of writing). **Verdict: maybe** —
nice for a paper; changes no code.

### 5.2 Optimality of greedy + local-search bucketing
Bucketing is a min-sum-of-products set partition — NP-hard in general
(it contains number partitioning). What *can* be shown: the local search
converges (monotone cost, finite states: already true) and a
`(1 + 1/e)`-style bound for the greedy step if the per-bucket cost is
submodular in members — it is not (closure is a union, cost is a product
of union probabilities; not submodular in general). Honest conclusion:
no clean optimality theorem; an *exhaustive* small-`n` comparison
(`n ≤ 12`, 8 buckets, brute force) showing greedy+refine within x% of
optimal is the credible evidence. Effort: S (it's a test). **Verdict:
do** the empirical gap test; **skip** the theorem.

### 5.3 When sparse provably dominates prefix sampling
Provable and short: if the patterns share a common prefix of length
`ℓ ≥ k`, the contiguous-prefix filter's candidate rate equals the
*corpus frequency of the prefix* (every bucket has identical classes at
all `k` positions), independent of bucket count; while a sparse set
containing any position `s > ℓ` where the patterns' bytes are pairwise
distinct achieves rate `≤ P(prefix) · max_b P(C_b(s))`. The ratio is the
measured 5.6× on W3. This is Theorem-2 material for DESIGN.md §4 and
costs nothing. Effort: S. **Verdict: do**.

---

## 6. Benchmarking honesty

### 6.1 x86 AVX-512 / AVX2 run of the current tree
README's first table is x86 from before the dense lane; the second is
NEON. A reviewer will ask for the same table on one x86 box with both
engines, including the two-prong and Wu-Manber columns. Effort: S
(access to a machine). **Verdict: do** — blocking for any external claim.

### 6.2 Real corpora and pattern sets
All four workloads are synthetic. Needed: Snort/Suricata community rules
(content strings: shared prefixes, binary, 1–10 K patterns), ClamAV
daily signatures (binary, long, 100 K+), a Wikipedia/Gutenberg text
dump with a stop-word + named-entity set, and a pcap (HTTP/TLS) for the
streaming path. These also exercise §3 and §2.3, which currently have no
evidence either way. Effort: M (licensing and loaders). **Verdict: do**
— nothing in §2–3 should be decided without them.

**Status:** Wikipedia done (`examples/wiki_bench.rs`, results in README —
including an honest loss on 256 rare patterns that motivates §3). Snort
rules and pcap corpora still open.

### 6.3 Hyperscan as a comparator
Hyperscan (or Vectorscan on ARM) is the engine reviewers will name
first; FDR + Teddy with a far better literal planner than aho-corasick's
packed Teddy. Expect it to be competitive or ahead on W1/W2 and behind
on W3. Binding via `hyperscan-sys` (x86) / Vectorscan (aarch64) in a
feature-gated bench. Effort: M. **Verdict: do** — a result without it is
not credible in this space.

### 6.4 Variance, scaling curves, build time
Report median ± MAD over ≥ 10 runs, not best-of-5; pattern-count scaling
curves (16, 64, 256, 1 K, 4 K) for every engine — this is where AC
crosses SPARROW and the reader wants to see it; build time per
configuration (the exhaustive search and referee are not free — state
it). Effort: S. **Verdict: do**.

### 6.5 Adversarial inputs
A reviewer will try: all-same-byte haystacks, haystack = pattern
repeated, patterns that are all prefixes of one another, and a corpus
sample that misrepresents the haystack (route wrong on purpose). The
first three are correctness-covered by tests but not benchmarked; the
last is the honest weakness of a corpus-fit design and deserves a
measured worst case. Effort: S. **Verdict: do**.

---

## Top 5, prioritised

1. **Timed build-time referee for routing (2.2) + per-engine constants
   (2.1)** — fixes the only measured regression (W1), and makes every
   later kernel change self-calibrating instead of hand-fit.
2. **Real corpora + Hyperscan comparator + x86 rerun (6.1–6.3)** — every
   decision in §2–3 is currently made on four synthetic workloads; the
   crate's claims are not reviewer-proof without these.
3. **Regex prefilter integration with byte-class patterns and native
   leftmost semantics (4.1, 4.3, 4.5)** — the path to users; the
   shared-prefix result was designed for exactly IDS/regex literal sets.
4. **Hashed verification + 4 planes (3.3, 3.2) with a 10 K-pattern
   footprint measurement (3.5)** — moves the ceiling from "a Teddy
   replacement" to "an AC replacement", where the footprint story is 100×.
5. **Output-path and streaming fixes (1.7, 1.8, 4.6) + vectorised
   Shift-Or (1.1)** — small, measurable, and the bench numbers on
   match-dense rows are still ~50% output cost.

Explicitly deprioritised: q-gram tables (1.4), scan-time adaptation
(2.6), hierarchical filters (3.4), full Unicode folding (4.2), and
optimality theorems for bucketing (5.2) — each either trades away the
design's core property or has no evidence it would pay.
