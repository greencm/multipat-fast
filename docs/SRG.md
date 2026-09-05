# srg — a subset ripgrep, built on SPARROW

`srg` is a small command-line search tool, shipped in this repo alongside
the library. It exists for one reason: to prove `sparrow` works as a real
end-user program, not only inside a benchmark harness. It is **not** a
ripgrep replacement — it implements a deliberately small, honest subset of
`rg`'s flags, documents exactly what it leaves out, and gets out of the
way otherwise.

If you want the full ripgrep, install ripgrep. If you want to see SPARROW
do actual work on your actual files — searching a big source tree for a
dozen literal words in one pass, or a log file against fifty IDS-style
regexes sharing a prefix — that's what `srg` is for.

## What it actually does differently

Ordinary `grep`/`rg` usage with several patterns runs one regex loop per
line, once per pattern (or builds one automaton per invocation). `srg`
instead reads each file whole into memory and runs **one scan of the whole
buffer against every pattern at once**, then maps the resulting match
spans back to line numbers with a single binary search. That's the actual
value SPARROW brings — see [`docs/DESIGN.md`](DESIGN.md) for why a
sparse-position SIMD filter beats a per-pattern loop, and the
[README results table](../README.md#results-16-mib-haystacks-one-core-best-of-5)
for the numbers.

Concretely: `srg -e a -e b -e c -e d file` is not four passes over `file`;
it's one.

## Install / build

```
cargo build --release --bin srg          # literal-pattern mode only
cargo build --release --bin srg --features prefilter   # + real regex mode
```

The binary lands at `target/release/srg`. Everything below assumes you've
either built it or are running it via `cargo run --release --bin srg --`.

## Quick start

```
# Find every TODO, FIXME, and use of `unsafe`, with line numbers,
# recursively, in one pass:
srg -n -e TODO -e FIXME -e unsafe src/

# Case-insensitive search of a single file:
srg -i "hello world" README.md

# Just the count of matching lines per file:
srg -c "unsafe" src/

# Just the filenames that contain a match:
srg -l "TODO" src/

# Invert: show lines that do NOT mention "test":
srg -v "test" src/lib.rs

# Whole-word only (so "class" doesn't match a search for "as"):
srg -w "as" src/lib.rs

# Only print the matched text, not the whole line:
srg -o -e "TODO" -e "FIXME" src/

# Read from a pipe instead of files:
git log --oneline | srg -i "fix"
```

## Use cases

### 1. Multi-keyword source audit, one pass
The natural use of `-e` (repeatable): grep a whole tree for a *list* of
literal terms in a single scan instead of one grep invocation per term.

```
srg -n -e "unwrap()" -e "expect(" -e "unsafe" -e "TODO" -e "FIXME" src/
```

Every one of those five patterns is checked against every byte of every
file in one pass per file — not five.

### 2. Security / dangerous-call sweep
Feed it a list of function names you want reviewed before a release —
this is the same shape of problem as the security-review workflows in
this project, applied directly from the shell:

```
srg -n -e "system(" -e "exec(" -e "eval(" -e "unsafe {" -e "transmute" src/
```

### 3. Log triage via stdin
`srg` reads stdin when given zero paths, so it composes with any log
source:

```
journalctl -u myservice --since today | srg -i -e error -e panic -e "OOM"
```

`srg` reads stdin to EOF in one pass, the same as any other input — it is
not a `tail -f`-style follower. Pipe a bounded amount of data (a
`--since`-scoped journal query, a saved log file) rather than a live,
unbounded stream.

### 4. Frequency check over a large text corpus
Point `-c` at a big file to get matching-*line* counts per pattern set
without writing a one-off script:

```
srg -c -e "the" -e "and" -e "of" bench_data/simplewiki-64MB.xml
```

(This is literally one of the workloads in
[`examples/wiki_bench.rs`](../examples/wiki_bench.rs) — `srg` is the same
matcher, wired up as a CLI instead of a `#[test]`.)

### 5. Shared-prefix IDS-style rules, as real regexes
Build with `--features prefilter` to turn patterns into byte-oriented
regexes matched through SPARROW's literal prefilter
(see [`docs/DESIGN.md` §3.6](DESIGN.md#36-regex-literal-prefilter)). This
is the shared-prefix case SPARROW is specifically good at — rules that all
start `GET /api/v1/...` and only diverge deep into the string:

```
cargo build --release --bin srg --features prefilter
srg -n \
  -e 'GET /api/v1/users\?id=\d+ HTTP/1\.[01]' \
  -e 'GET /api/v1/carts/[0-9a-f]{4,} HTTP' \
  access.log
```

### 6. Pull just the matched text out
`-o` prints the matched span instead of the whole line, so it composes
with the rest of your shell pipeline:

```
srg -o -e '\d+' access.log | sort | uniq -c
```

(This needs `--features prefilter` — `\d+` is a regex, and the base build
only matches literal text.)

(`-o` with multiple overlapping patterns de-duplicates identical spans, so
two patterns matching the same text at the same position print once.)

## Full flag reference

```
srg [OPTIONS] PATTERN [PATH...]
srg [OPTIONS] -e PATTERN [-e PATTERN...] [PATH...]
```

With zero `PATH` arguments, `srg` reads standard input as a single
(unnamed) file.

| Flag | Meaning |
|---|---|
| `-e PATTERN` | Add a pattern (repeatable). Using `-e` at all disables the leading positional pattern — every positional argument becomes a `PATH` instead. |
| `-i`, `--ignore-case` | Case-insensitive match. |
| `-v`, `--invert-match` | Print lines that do **not** match, instead of ones that do. |
| `-w`, `--word-regexp` | Only match whole words (checked against ASCII word-character neighbors in literal mode; wrapped in `\b(?:...)\b` in regex mode). |
| `-F`, `--fixed-strings` | Treat patterns as literal text. This is the default and *only* mode in the base build; with `--features prefilter` it forces literal matching instead of regex for that run. |
| `-o`, `--only-matching` | Print only the matched text, not the whole line. Ignored (with a one-time stderr warning) if combined with `-v`. |
| `-c`, `--count` | Print only a per-file count of matching lines (or, with `-v`, non-matching lines). Unlike ripgrep, files with a count of `0` are still listed (classic-grep behavior) rather than suppressed — expect `path:0` lines when recursing over a tree where most files don't match. |
| `-l`, `--files-with-matches` | Print only the names of files that contain a match. |
| `-n` | Show line numbers. This is the default. |
| `-N`, `--no-line-number` | Hide line numbers. |
| `-r`, `--recursive` | Accepted for muscle-memory familiarity; recursion into directories always happens, so this is a no-op. |
| `-h`, `--help` | Print the built-in help and exit. |
| `--version` | Print the version and exit. |

**Filename prefix**: shown automatically whenever more than one file ends
up being scanned (after directory expansion) — a single explicit file, or
stdin, never gets a `path:` prefix; two files, or one directory containing
two or more files, always do.

**Recursion**: automatic for any `PATH` that is a directory. Skips
dotfiles and dot-directories (so `.git` is covered) plus a small hardcoded
list — `target`, `node_modules`, `.hg`, `.svn` — and never follows
symlinks.

**Binary files**: skipped silently if the first 8000 bytes contain a NUL
byte (the same heuristic grep/rg use, simplified).

**Exit status**:

| Code | Meaning |
|---|---|
| `0` | A match was printed (or, with `-v`, a non-matching line was). |
| `1` | Nothing was printed; no error occurred. |
| `2` | A pattern or file error occurred. Other files are still processed. |

## What's deliberately not supported

`srg` will never grow these — reach for real ripgrep:

- `.gitignore` / `.ignore` file respecting (it only knows the hardcoded
  skip-list above, not your repo's ignore rules)
- Glob or file-type filtering (`-g`, `--type`)
- Context lines (`-A`/`-B`/`-C`)
- Color output
- Multiline patterns (each match is resolved to the line containing its
  *start* offset; a match cannot be displayed as spanning several lines)
- PCRE-only regex features (lookaround, backreferences) — regex mode is
  whatever `regex-automata` supports
- `--replace`, JSON output, or any output format besides plain text

## See also

- [`README.md`](../README.md) — the library itself, its benchmark results,
  and the two-prong dense/sparse design `srg` is built on.
- [`docs/DESIGN.md`](DESIGN.md) — the algorithm, the regex prefilter
  (§3.6), and native leftmost matching, all of which `srg` uses directly.
- [`docs/ROADMAP.md`](ROADMAP.md) §4.1 — why a CLI consumer mattered
  enough to build.
- [`tests/srg_cli.rs`](../tests/srg_cli.rs) — the executable spec: every
  behavior described above is asserted there against the built binary.

