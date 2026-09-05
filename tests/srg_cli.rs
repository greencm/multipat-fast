//! Integration tests for the `srg` binary: spawn it as a subprocess against
//! temp files, the way a shell would. No `tempfile` crate — a unique
//! per-test scratch directory under `std::env::temp_dir()`, cleaned up on
//! drop.

use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicUsize, Ordering};

struct Scratch {
    dir: PathBuf,
}

impl Scratch {
    fn new(tag: &str) -> Scratch {
        static COUNTER: AtomicUsize = AtomicUsize::new(0);
        let n = COUNTER.fetch_add(1, Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!("srg-test-{}-{}-{}", std::process::id(), tag, n));
        fs::create_dir_all(&dir).unwrap();
        Scratch { dir }
    }

    fn write(&self, rel: &str, content: &str) -> PathBuf {
        let p = self.dir.join(rel);
        if let Some(parent) = p.parent() {
            fs::create_dir_all(parent).unwrap();
        }
        fs::write(&p, content).unwrap();
        p
    }

    fn path(&self, rel: &str) -> PathBuf {
        self.dir.join(rel)
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.dir);
    }
}

struct Output {
    status: i32,
    stdout: String,
    stderr: String,
}

fn srg(args: &[&str], cwd: Option<&Path>, stdin: Option<&str>) -> Output {
    let bin = env!("CARGO_BIN_EXE_srg");
    let mut cmd = Command::new(bin);
    cmd.args(args);
    if let Some(cwd) = cwd {
        cmd.current_dir(cwd);
    }
    cmd.stdin(Stdio::piped());
    cmd.stdout(Stdio::piped());
    cmd.stderr(Stdio::piped());
    let mut child = cmd.spawn().expect("spawn srg");
    if let Some(input) = stdin {
        child.stdin.take().unwrap().write_all(input.as_bytes()).unwrap();
    } else {
        drop(child.stdin.take());
    }
    let out = child.wait_with_output().expect("wait srg");
    Output {
        status: out.status.code().unwrap_or(-1),
        stdout: String::from_utf8_lossy(&out.stdout).into_owned(),
        stderr: String::from_utf8_lossy(&out.stderr).into_owned(),
    }
}

#[test]
fn basic_literal_match_with_line_numbers() {
    let s = Scratch::new("basic");
    let f = s.write("a.txt", "hello world\nfoo bar\nhello again\n");
    let o = srg(&["hello", f.to_str().unwrap()], None, None);
    assert_eq!(o.status, 0);
    assert_eq!(o.stdout, "1:hello world\n3:hello again\n");
}

#[test]
fn ignore_case() {
    let s = Scratch::new("icase");
    let f = s.write("a.txt", "Hello\nHELLO\nhello\nworld\n");
    let o = srg(&["-i", "-N", "hello", f.to_str().unwrap()], None, None);
    assert_eq!(o.status, 0);
    assert_eq!(o.stdout, "Hello\nHELLO\nhello\n");
}

#[test]
fn invert_match() {
    let s = Scratch::new("invert");
    let f = s.write("a.txt", "keep\nskip\nkeep\n");
    let o = srg(&["-v", "-N", "skip", f.to_str().unwrap()], None, None);
    assert_eq!(o.status, 0);
    assert_eq!(o.stdout, "keep\nkeep\n");
}

#[test]
fn count_mode() {
    let s = Scratch::new("count");
    let f = s.write("a.txt", "x\nx\ny\nx\n");
    let o = srg(&["-c", "x", f.to_str().unwrap()], None, None);
    assert_eq!(o.status, 0);
    assert_eq!(o.stdout.trim(), "3");
}

#[test]
fn count_mode_no_match_is_zero_and_exit_one() {
    let s = Scratch::new("count0");
    let f = s.write("a.txt", "y\nz\n");
    let o = srg(&["-c", "x", f.to_str().unwrap()], None, None);
    assert_eq!(o.status, 1);
    assert_eq!(o.stdout.trim(), "0");
}

#[test]
fn files_with_matches_mode() {
    let s = Scratch::new("l");
    let f1 = s.write("hit.txt", "needle here\n");
    let f2 = s.write("miss.txt", "nothing here\n");
    let o = srg(&["-l", "needle", f1.to_str().unwrap(), f2.to_str().unwrap()], None, None);
    assert_eq!(o.status, 0);
    assert!(o.stdout.contains("hit.txt"));
    assert!(!o.stdout.contains("miss.txt"));
}

#[test]
fn no_line_number_flag() {
    let s = Scratch::new("noln");
    let f = s.write("a.txt", "match\n");
    let o = srg(&["-N", "match", f.to_str().unwrap()], None, None);
    assert_eq!(o.stdout, "match\n");
}

#[test]
fn only_matching_with_repeated_e_multiple_spans_per_line() {
    let s = Scratch::new("only");
    let f = s.write("a.txt", "cat and dog\n");
    let o = srg(&["-N", "-o", "-e", "cat", "-e", "dog", f.to_str().unwrap()], None, None);
    let mut lines: Vec<&str> = o.stdout.lines().collect();
    lines.sort();
    assert_eq!(lines, vec!["cat", "dog"]);
}

#[test]
fn word_boundary() {
    let s = Scratch::new("word");
    let f = s.write("a.txt", "cat category cats cat\n");
    let o = srg(&["-N", "-o", "-w", "cat", f.to_str().unwrap()], None, None);
    // "cat" appears standalone twice (positions 0 and end); "category" and
    // "cats" must not match under -w.
    let count = o.stdout.lines().filter(|l| *l == "cat").count();
    assert_eq!(count, 2, "stdout was: {:?}", o.stdout);
}

#[test]
fn recursive_walk_skips_hidden_and_target_dirs() {
    let s = Scratch::new("walk");
    s.write("visible/real.txt", "needle outside\n");
    s.write(".git/hidden.txt", "needle in dotdir\n");
    s.write("target/build.txt", "needle in target\n");
    let o = srg(&["-N", "needle", s.path("").to_str().unwrap()], None, None);
    assert_eq!(o.status, 0);
    assert!(o.stdout.contains("needle outside"));
    assert!(!o.stdout.contains("needle in dotdir"));
    assert!(!o.stdout.contains("needle in target"));
}

#[test]
fn stdin_mode_no_filename_prefix() {
    let o = srg(&["beta"], None, Some("alpha\nbeta\ngamma\n"));
    assert_eq!(o.status, 0);
    assert_eq!(o.stdout, "2:beta\n");
}

#[test]
fn single_explicit_file_has_no_prefix_two_files_do() {
    let s = Scratch::new("prefix");
    let f1 = s.write("one.txt", "hit\n");
    let f2 = s.write("two.txt", "hit\n");
    let o1 = srg(&["-N", "hit", f1.to_str().unwrap()], None, None);
    assert_eq!(o1.stdout, "hit\n");
    let o2 = srg(&["-N", "hit", f1.to_str().unwrap(), f2.to_str().unwrap()], None, None);
    assert!(o2.stdout.contains(&format!("{}:hit", f1.display())));
    assert!(o2.stdout.contains(&format!("{}:hit", f2.display())));
}

#[test]
fn binary_file_is_skipped() {
    let s = Scratch::new("binary");
    let p = s.path("bin.dat");
    fs::write(&p, [b'n', b'e', b'e', b'd', b'l', b'e', 0u8, b'x']).unwrap();
    let text = s.write("text.txt", "needle here\n");
    // Two files are passed, so a filename prefix is expected on the one hit.
    let o = srg(&["-N", "needle", p.to_str().unwrap(), text.to_str().unwrap()], None, None);
    assert_eq!(o.status, 0);
    assert!(!o.stdout.contains("\0"), "binary file's content leaked into output");
    assert_eq!(o.stdout, format!("{}:needle here\n", text.display()));
}

#[test]
fn missing_file_argument_is_error_exit_but_continues() {
    let s = Scratch::new("missing");
    let good = s.write("good.txt", "hit\n");
    let o = srg(&["-N", "hit", "/no/such/path/at/all", good.to_str().unwrap()], None, None);
    assert_eq!(o.status, 2);
    assert!(o.stdout.contains("hit"));
    assert!(!o.stderr.is_empty());
}

#[test]
fn help_and_version_exit_zero() {
    let o = srg(&["--help"], None, None);
    assert_eq!(o.status, 0);
    assert!(o.stdout.contains("USAGE"));
    let o = srg(&["--version"], None, None);
    assert_eq!(o.status, 0);
    assert!(o.stdout.contains("srg"));
}

#[cfg(feature = "prefilter")]
mod regex_mode {
    use super::*;

    #[test]
    fn digit_class_regex() {
        let s = Scratch::new("re-digit");
        let f = s.write("a.txt", "order 42 shipped\nno numbers here\nid 7\n");
        let o = srg(&["-N", r"\d+", f.to_str().unwrap()], None, None);
        assert_eq!(o.status, 0);
        assert_eq!(o.stdout, "order 42 shipped\nid 7\n");
    }

    #[test]
    fn inline_case_fold_flag() {
        let s = Scratch::new("re-icase");
        let f = s.write("a.txt", "Hello\nworld\n");
        let o = srg(&["-N", "-i", "hello", f.to_str().unwrap()], None, None);
        assert_eq!(o.stdout, "Hello\n");
    }

    #[test]
    fn fixed_strings_forces_literal_even_with_feature_on() {
        let s = Scratch::new("re-F");
        // A pattern that is a regex metacharacter sequence must be matched
        // literally under -F, not interpreted as "one or more digits".
        let f = s.write("a.txt", "a.b\nacb\naxb\n");
        let o = srg(&["-N", "-F", "a.b", f.to_str().unwrap()], None, None);
        assert_eq!(o.stdout, "a.b\n");
    }

    /// Independent oracle using the `regex` crate directly (already a
    /// dev-dependency) to confirm the CLI's prefilter plumbing does not
    /// drop matches on a moderately adversarial pattern set.
    #[test]
    fn no_false_negatives_vs_regex_crate_oracle() {
        let hay = "abc123 def456 xz 789yz foo bar 000 baz9";
        let s = Scratch::new("re-oracle");
        let f = s.write("a.txt", &format!("{hay}\n"));
        let pattern = r"[a-z]+\d+";
        let o = srg(&["-N", "-o", pattern, f.to_str().unwrap()], None, None);
        let got: Vec<String> = o.stdout.lines().map(|l| l.to_string()).collect();

        let re = regex::Regex::new(pattern).unwrap();
        let want: Vec<String> = re.find_iter(hay).map(|m| m.as_str().to_string()).collect();
        assert_eq!(got, want);
    }
}
