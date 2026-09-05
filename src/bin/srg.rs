//! `srg` ("sparrow grep") — a deliberately partial ripgrep-compatible
//! search tool built on the `sparrow` crate, to show it working as a real
//! end-user command rather than only inside a benchmark harness.
//!
//! Design: each file is read whole into memory and scanned in ONE pass
//! (`Sparrow::find_all`, or `Prefilter::find_all` under `--features
//! prefilter`) to get every match span; spans are then mapped to line
//! numbers by binary-searching a precomputed table of line-start offsets.
//! That is the point of using sparrow here: a single whole-buffer scan
//! against many patterns at once, not a naive per-line regex loop.
//!
//! Run `srg --help` for the supported flag subset and what is deliberately
//! left out (`.gitignore`, globs, context lines, color, multiline, JSON).

use std::fs;
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};
use std::process::ExitCode;

const HELP: &str = r#"srg - a subset ripgrep built on the sparrow multi-pattern matcher

USAGE:
    srg [OPTIONS] PATTERN [PATH...]
    srg [OPTIONS] -e PATTERN [-e PATTERN...] [PATH...]

    With zero PATHs, srg reads standard input.

OPTIONS:
    -e PATTERN          add a pattern (repeatable); disables the leading
                         positional pattern, so all positionals become PATHs
    -i, --ignore-case    case-insensitive match
    -v, --invert-match   print non-matching lines instead
    -w, --word-regexp    match only whole words
    -F, --fixed-strings  treat patterns as literal text, not regex
                         (this is the default and only mode unless built
                         with --features prefilter)
    -o, --only-matching  print only the matched text, not the whole line
    -c, --count          print only a per-file match-line count
    -l, --files-with-matches
                         print only names of files containing a match
    -n                   show line numbers (default)
    -N, --no-line-number
                         hide line numbers
    -r, --recursive      accepted for familiarity; recursion into
                         directories always happens, this flag is a no-op
    -h, --help           print this help and exit
        --version        print the version and exit

SUPPORTED SUBSET (deliberately partial):
    literal multi-pattern search (many -e patterns in one pass)
    ignore-case, invert-match, word-regexp, only-matching, count,
    files-with-matches, line numbers, recursive directory walk,
    stdin input, binary-file skip, hidden-file skip
    with --features prefilter: patterns are regexes (byte-oriented,
    `(?-u)` semantics) via a SPARROW literal prefilter in front of
    regex-automata; -F forces literal mode even then

NOT SUPPORTED (use ripgrep itself if you need these):
    .gitignore / .ignore respecting (srg only skips dotfiles and a
    hardcoded list: target, node_modules, .git, .hg, .svn)
    glob/type filtering (-g, --type), context lines (-A/-B/-C),
    color output, multiline patterns, PCRE-only regex features,
    --replace, JSON output

EXIT STATUS:
    0  a match was found (or, with -v, a non-matching line was found)
    1  no match and no error
    2  a pattern or file error occurred (other files still get processed)
"#;

const SKIP_DIRS: &[&str] = &["target", "node_modules", ".git", ".hg", ".svn"];
/// Bytes sniffed from the start of a file to decide "binary" (grep/rg both
/// use a NUL-byte heuristic over a bounded prefix; simplified here).
const BINARY_SNIFF: usize = 8000;

struct Opts {
    ignore_case: bool,
    invert: bool,
    word: bool,
    fixed_strings: bool,
    only_matching: bool,
    count: bool,
    files_with_matches: bool,
    line_numbers: bool,
}

enum Mode {
    /// One literal `Sparrow` matcher over all patterns.
    Literal(sparrow::Sparrow),
    /// One `Prefilter` over all patterns, treated as regexes.
    #[cfg(feature = "prefilter")]
    Regex(sparrow::prefilter::Prefilter),
}

/// One match span plus which pattern/regex produced it (index is unused
/// today but kept for a future --only-matching-with-pattern-id style flag).
struct Span {
    start: usize,
    end: usize,
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match run(&args) {
        Ok(code) => code,
        Err(msg) => {
            eprintln!("srg: {msg}");
            ExitCode::from(2)
        }
    }
}

fn run(args: &[String]) -> Result<ExitCode, String> {
    let mut opts = Opts {
        ignore_case: false,
        invert: false,
        word: false,
        fixed_strings: false,
        only_matching: false,
        count: false,
        files_with_matches: false,
        line_numbers: true,
    };
    let mut patterns: Vec<String> = Vec::new();
    let mut paths: Vec<String> = Vec::new();
    let mut explicit_e = false;
    let mut positional_pattern: Option<String> = None;

    let mut it = args.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "-h" | "--help" => {
                print!("{HELP}");
                return Ok(ExitCode::SUCCESS);
            }
            "--version" => {
                println!("srg {}", env!("CARGO_PKG_VERSION"));
                return Ok(ExitCode::SUCCESS);
            }
            "-i" | "--ignore-case" => opts.ignore_case = true,
            "-v" | "--invert-match" => opts.invert = true,
            "-w" | "--word-regexp" => opts.word = true,
            "-F" | "--fixed-strings" => opts.fixed_strings = true,
            "-o" | "--only-matching" => opts.only_matching = true,
            "-c" | "--count" => opts.count = true,
            "-l" | "--files-with-matches" => opts.files_with_matches = true,
            "-n" => opts.line_numbers = true,
            "-N" | "--no-line-number" => opts.line_numbers = false,
            "-r" | "--recursive" => {}
            "-e" => {
                let p = it.next().ok_or("-e requires an argument")?;
                patterns.push(p.clone());
                explicit_e = true;
            }
            "--" => {
                for rest in it.by_ref() {
                    if positional_pattern.is_none() && !explicit_e {
                        positional_pattern = Some(rest.clone());
                    } else {
                        paths.push(rest.clone());
                    }
                }
            }
            s if s.starts_with('-') && s.len() > 1 => {
                return Err(format!("unrecognized option '{s}' (see --help)"));
            }
            s => {
                if positional_pattern.is_none() && !explicit_e {
                    positional_pattern = Some(s.to_string());
                } else {
                    paths.push(s.to_string());
                }
            }
        }
    }

    if opts.only_matching && opts.invert {
        eprintln!("srg: -o is ignored with -v (there is no matched text on a non-matching line)");
        opts.only_matching = false;
    }

    if let Some(p) = positional_pattern {
        patterns.push(p);
    }
    if patterns.is_empty() {
        return Err("no pattern given (see --help)".to_string());
    }

    let mode = build_mode(&patterns, &opts, &paths)?;

    let mut any_error = false;
    let files = if paths.is_empty() {
        vec![None]
    } else {
        expand_paths(&paths, &mut any_error)?
    };
    let show_filename = files.len() > 1;

    let mut any_printed = false;
    let stdout = io::stdout();
    let mut out = stdout.lock();

    for f in &files {
        let label = f.as_deref().map(|p: &Path| p.display().to_string());
        let data = match f {
            None => {
                let mut buf = Vec::new();
                if io::stdin().read_to_end(&mut buf).is_err() {
                    any_error = true;
                    continue;
                }
                buf
            }
            Some(p) => match fs::read(p) {
                Ok(b) => b,
                Err(e) => {
                    eprintln!("srg: {}: {e}", p.display());
                    any_error = true;
                    continue;
                }
            },
        };
        if is_binary(&data) {
            continue;
        }
        match scan_file(&mode, &opts, &data, label.as_deref(), show_filename, &mut out) {
            Ok(printed) => any_printed |= printed,
            Err(e) => {
                eprintln!("srg: {e}");
                any_error = true;
            }
        }
    }

    let _ = out.flush();
    if any_error {
        Ok(ExitCode::from(2))
    } else if any_printed {
        Ok(ExitCode::SUCCESS)
    } else {
        Ok(ExitCode::from(1))
    }
}

fn build_mode(patterns: &[String], opts: &Opts, _paths: &[String]) -> Result<Mode, String> {
    #[cfg(feature = "prefilter")]
    {
        if !opts.fixed_strings {
            let regexes: Vec<String> = patterns
                .iter()
                .map(|p| {
                    let p = if opts.word { format!(r"\b(?:{p})\b") } else { p.clone() };
                    if opts.ignore_case { format!("(?i){p}") } else { p }
                })
                .collect();
            let pf = sparrow::prefilter::Prefilter::new(&regexes, None)
                .map_err(|e| format!("regex error: {e}"))?;
            return Ok(Mode::Regex(pf));
        }
    }
    #[cfg(not(feature = "prefilter"))]
    {
        let _ = opts.word; // word-boundary handled post-hoc for literal mode
    }
    let m = sparrow::Builder::new()
        .ascii_case_insensitive(opts.ignore_case)
        .build(patterns)
        .map_err(|e| format!("pattern error: {e}"))?;
    Ok(Mode::Literal(m))
}

/// Expand PATH args into a flat list of files, walking directories
/// (skipping hidden entries, symlinks, and `SKIP_DIRS`). Sets `*any_error`
/// on any path that could not be read, without aborting the rest.
fn expand_paths(paths: &[String], any_error: &mut bool) -> Result<Vec<Option<PathBuf>>, String> {
    let mut out = Vec::new();
    for p in paths {
        let path = PathBuf::from(p);
        let meta = match fs::symlink_metadata(&path) {
            Ok(m) => m,
            Err(e) => {
                eprintln!("srg: {p}: {e}");
                *any_error = true;
                continue;
            }
        };
        if meta.is_symlink() {
            eprintln!("srg: {p}: skipping symlink");
            continue;
        }
        if meta.is_dir() {
            walk_dir(&path, &mut out, any_error);
        } else {
            out.push(Some(path));
        }
    }
    Ok(out)
}

fn walk_dir(dir: &Path, out: &mut Vec<Option<PathBuf>>, any_error: &mut bool) {
    let entries = match fs::read_dir(dir) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("srg: {}: {e}", dir.display());
            *any_error = true;
            return;
        }
    };
    let mut children: Vec<PathBuf> = entries.filter_map(|e| e.ok()).map(|e| e.path()).collect();
    children.sort();
    for path in children {
        let name = match path.file_name().and_then(|n| n.to_str()) {
            Some(n) => n,
            None => continue,
        };
        if name.starts_with('.') || SKIP_DIRS.contains(&name) {
            continue;
        }
        let meta = match fs::symlink_metadata(&path) {
            Ok(m) => m,
            Err(_) => continue,
        };
        if meta.is_symlink() {
            continue;
        }
        if meta.is_dir() {
            walk_dir(&path, out, any_error);
        } else if meta.is_file() {
            out.push(Some(path));
        }
    }
}

fn is_binary(data: &[u8]) -> bool {
    data[..data.len().min(BINARY_SNIFF)].contains(&0)
}

fn is_word_byte(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'_'
}

/// Match spans over the whole buffer, in one pass. Word-boundary filtering
/// for literal mode happens here (regex mode already wrapped the pattern
/// in `\b...\b` at build time).
fn spans(mode: &Mode, opts: &Opts, hay: &[u8]) -> Result<Vec<Span>, String> {
    match mode {
        Mode::Literal(m) => {
            let mut v: Vec<Span> = m
                .find_all(hay)
                .into_iter()
                .map(|mtc| Span { start: mtc.start, end: mtc.end })
                .collect();
            if opts.word {
                v.retain(|s| {
                    let before_ok = s.start == 0 || !is_word_byte(hay[s.start - 1]);
                    let after_ok = s.end >= hay.len() || !is_word_byte(hay[s.end]);
                    before_ok && after_ok
                });
            }
            Ok(v)
        }
        #[cfg(feature = "prefilter")]
        Mode::Regex(pf) => {
            Ok(pf.find_all(hay).into_iter().map(|m| Span { start: m.start, end: m.end }).collect())
        }
    }
}

/// Byte offset of the start of each line (line 0 starts at offset 0).
fn line_starts(data: &[u8]) -> Vec<usize> {
    let mut v = vec![0usize];
    for (i, &b) in data.iter().enumerate() {
        if b == b'\n' && i + 1 < data.len() {
            v.push(i + 1);
        }
    }
    v
}

/// Line index (0-based) containing byte offset `pos`. `starts` is sorted.
fn line_of(starts: &[usize], pos: usize) -> usize {
    starts.partition_point(|&s| s <= pos).saturating_sub(1)
}

fn line_text<'a>(data: &'a [u8], starts: &[usize], line: usize) -> &'a [u8] {
    let start = starts[line];
    let end = starts.get(line + 1).map_or(data.len(), |&e| e);
    let mut end = end;
    if end > start && data[end - 1] == b'\n' {
        end -= 1;
    }
    if end > start && data[end - 1] == b'\r' {
        end -= 1;
    }
    &data[start..end]
}

/// Write one line to `out`. If the reader on the other end of stdout has
/// gone away (e.g. piped into `head`), exit immediately and quietly — the
/// conventional Unix behavior — instead of returning an error that would
/// otherwise print an ugly "Broken pipe (os error 32)" for every
/// subsequent line.
fn write_line(out: &mut impl Write, line: &str) -> Result<(), String> {
    if let Err(e) = writeln!(out, "{line}") {
        if e.kind() == io::ErrorKind::BrokenPipe {
            std::process::exit(0);
        }
        return Err(e.to_string());
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn scan_file(
    mode: &Mode,
    opts: &Opts,
    data: &[u8],
    label: Option<&str>,
    show_filename: bool,
    out: &mut impl Write,
) -> Result<bool, String> {
    let all_spans = spans(mode, opts, data)?;
    let starts = line_starts(data);
    let n_lines = starts.len();

    let mut matched_line = vec![false; n_lines];
    for s in &all_spans {
        matched_line[line_of(&starts, s.start)] = true;
    }

    let mut printed = false;
    let prefix = |line_no: usize| -> String {
        match (show_filename, opts.line_numbers) {
            (true, true) => format!("{}:{}:", label.unwrap_or(""), line_no + 1),
            (true, false) => format!("{}:", label.unwrap_or("")),
            (false, true) => format!("{}:", line_no + 1),
            (false, false) => String::new(),
        }
    };

    if opts.only_matching {
        let mut dedup = all_spans;
        dedup.sort_by_key(|s| (s.start, s.end));
        dedup.dedup_by_key(|s| (s.start, s.end));
        for s in &dedup {
            let line_no = line_of(&starts, s.start);
            let p = prefix(line_no);
            let text = String::from_utf8_lossy(&data[s.start..s.end]);
            if p.is_empty() {
                write_line(out, &text)?;
            } else {
                write_line(out, &format!("{p}{text}"))?;
            }
            printed = true;
        }
        return Ok(printed);
    }

    if opts.files_with_matches {
        let hit = matched_line.iter().any(|&m| m != opts.invert);
        if hit {
            write_line(out, label.unwrap_or("(stdin)"))?;
            printed = true;
        }
        return Ok(printed);
    }

    if opts.count {
        let n = matched_line.iter().filter(|&&m| m != opts.invert).count();
        if show_filename {
            write_line(out, &format!("{}:{n}", label.unwrap_or("")))?;
        } else {
            write_line(out, &n.to_string())?;
        }
        printed = n > 0;
        return Ok(printed);
    }

    for (line_no, &hit) in matched_line.iter().enumerate().take(n_lines) {
        if hit == opts.invert {
            continue;
        }
        let text = line_text(data, &starts, line_no);
        let p = prefix(line_no);
        write_line(out, &format!("{p}{}", String::from_utf8_lossy(text)))?;
        printed = true;
    }
    Ok(printed)
}
