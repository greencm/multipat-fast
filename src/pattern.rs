//! Byte-class patterns: a pattern is a sequence of byte *sets*, one per
//! position. Exact bytes, ASCII case pairs and the single-byte wildcard are
//! all special cases; arbitrary classes (`[0-9]`, `[^\x00-\x1f]`) are the
//! general one. The sampled-position filter sees a class through its
//! nibble closure (which inflates for spread classes — the cost model
//! prices that exactly and the optimizer avoids such positions); the dense
//! lane and the verifier test membership exactly.

use std::fmt;

/// A set of byte values (256-bit bitmap).
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct ByteSet([u64; 4]);

impl ByteSet {
    pub const EMPTY: ByteSet = ByteSet([0; 4]);
    pub const ANY: ByteSet = ByteSet([u64::MAX; 4]);

    /// The singleton set `{b}`.
    pub const fn byte(b: u8) -> ByteSet {
        let mut s = [0u64; 4];
        s[(b >> 6) as usize] = 1u64 << (b & 63);
        ByteSet(s)
    }
    /// The inclusive range `lo..=hi`.
    pub fn range(lo: u8, hi: u8) -> ByteSet {
        let mut s = ByteSet::EMPTY;
        for b in lo..=hi {
            s.insert(b);
        }
        s
    }
    #[inline(always)]
    pub fn contains(&self, b: u8) -> bool {
        (self.0[(b >> 6) as usize] >> (b & 63)) & 1 != 0
    }
    pub fn insert(&mut self, b: u8) {
        self.0[(b >> 6) as usize] |= 1u64 << (b & 63);
    }
    pub fn union(self, o: ByteSet) -> ByteSet {
        ByteSet([self.0[0] | o.0[0], self.0[1] | o.0[1], self.0[2] | o.0[2], self.0[3] | o.0[3]])
    }
    pub fn negate(self) -> ByteSet {
        ByteSet([!self.0[0], !self.0[1], !self.0[2], !self.0[3]])
    }
    /// Number of bytes in the set.
    pub fn len(&self) -> u32 {
        self.0.iter().map(|w| w.count_ones()).sum()
    }
    pub fn is_empty(&self) -> bool {
        self.0 == [0; 4]
    }
    pub fn is_singleton(&self) -> bool {
        self.len() == 1
    }
    /// Smallest member, if any.
    pub fn first(&self) -> Option<u8> {
        self.iter().next()
    }
    pub fn iter(&self) -> impl Iterator<Item = u8> + '_ {
        (0..=255u8).filter(move |&b| self.contains(b))
    }
    /// Close the set under ASCII case: every letter brings its other case.
    pub fn ascii_case_fold(self) -> ByteSet {
        let mut s = self;
        for b in self.iter() {
            if b.is_ascii_alphabetic() {
                s.insert(b ^ 0x20);
            }
        }
        s
    }
}

impl fmt::Debug for ByteSet {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if *self == ByteSet::ANY {
            return write!(f, ".");
        }
        if let (1, Some(b)) = (self.len(), self.first()) {
            return write!(f, "{:?}", b as char);
        }
        write!(f, "[")?;
        let mut b = 0usize;
        while b < 256 {
            if self.contains(b as u8) {
                let lo = b;
                while b + 1 < 256 && self.contains((b + 1) as u8) {
                    b += 1;
                }
                if lo == b {
                    write!(f, "{}", esc(lo as u8))?;
                } else {
                    write!(f, "{}-{}", esc(lo as u8), esc(b as u8))?;
                }
            }
            b += 1;
        }
        write!(f, "]")
    }
}

fn esc(b: u8) -> String {
    if b.is_ascii_graphic() && b != b'\\' && b != b']' && b != b'-' && b != b'^' {
        (b as char).to_string()
    } else {
        format!("\\x{:02x}", b)
    }
}

/// A pattern: one [`ByteSet`] per position. Build with [`Pattern::bytes`],
/// [`Pattern::parse`], or by pushing sets.
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct Pattern {
    sets: Vec<ByteSet>,
}

/// Errors from [`Pattern::parse`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PatternError {
    /// Unterminated `[...]` class or trailing `\`.
    Unterminated,
    /// Bad escape (`\xZZ`, unknown `\q`).
    BadEscape(usize),
    /// Range with `lo > hi` inside a class.
    BadRange(usize),
    /// A class with no members (`[]`, `[^\x00-\xff]`), or an empty pattern.
    Empty,
}

impl fmt::Display for PatternError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PatternError::Unterminated => write!(f, "unterminated class or escape"),
            PatternError::BadEscape(i) => write!(f, "bad escape at byte {}", i),
            PatternError::BadRange(i) => write!(f, "bad range at byte {}", i),
            PatternError::Empty => write!(f, "empty pattern or empty class"),
        }
    }
}
impl std::error::Error for PatternError {}

impl Pattern {
    /// An exact byte string.
    pub fn bytes(b: &[u8]) -> Pattern {
        Pattern { sets: b.iter().map(|&c| ByteSet::byte(c)).collect() }
    }
    pub fn new() -> Pattern {
        Pattern { sets: Vec::new() }
    }
    pub fn push(&mut self, set: ByteSet) -> &mut Pattern {
        self.sets.push(set);
        self
    }
    pub fn len(&self) -> usize {
        self.sets.len()
    }
    pub fn is_empty(&self) -> bool {
        self.sets.is_empty()
    }
    pub fn sets(&self) -> &[ByteSet] {
        &self.sets
    }
    /// True iff every position is a single byte.
    pub fn is_exact(&self) -> bool {
        self.sets.iter().all(|s| s.is_singleton())
    }
    /// Close every position under ASCII case.
    pub fn ascii_case_fold(mut self) -> Pattern {
        for s in &mut self.sets {
            *s = s.ascii_case_fold();
        }
        self
    }

    /// Parse a small glob/regex-like class syntax, byte oriented:
    /// literal bytes; `.` = any byte; `[abc]`, `[a-f0-9]`, `[^...]` classes
    /// (with `\xHH`, `\d`, `\w`, `\s`, `\n`, `\t`, `\r`, `\\`, `\]`, `\-`,
    /// `\^` escapes inside); `\d` `\w` `\s` `\D` `\W` `\S` `\xHH` and
    /// escaped metacharacters outside. Non-ASCII input is taken as its
    /// UTF-8 bytes.
    pub fn parse(s: &str) -> Result<Pattern, PatternError> {
        let b = s.as_bytes();
        let mut out = Pattern::new();
        let mut i = 0;
        while i < b.len() {
            match b[i] {
                b'.' => {
                    out.push(ByteSet::ANY);
                    i += 1;
                }
                b'\\' => {
                    let (set, n) = parse_escape(b, i)?;
                    out.push(set);
                    i += n;
                }
                b'[' => {
                    let (set, n) = parse_class(b, i)?;
                    out.push(set);
                    i += n;
                }
                c => {
                    out.push(ByteSet::byte(c));
                    i += 1;
                }
            }
        }
        if out.is_empty() {
            return Err(PatternError::Empty);
        }
        Ok(out)
    }
}

impl Default for Pattern {
    fn default() -> Self {
        Pattern::new()
    }
}

impl fmt::Debug for Pattern {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for s in &self.sets {
            if let (1, Some(b)) = (s.len(), s.first()) {
                write!(f, "{}", esc(b))?;
            } else {
                write!(f, "{:?}", s)?;
            }
        }
        Ok(())
    }
}

impl From<&[u8]> for Pattern {
    fn from(b: &[u8]) -> Pattern {
        Pattern::bytes(b)
    }
}
impl From<&str> for Pattern {
    fn from(s: &str) -> Pattern {
        Pattern::bytes(s.as_bytes())
    }
}
impl From<Vec<u8>> for Pattern {
    fn from(b: Vec<u8>) -> Pattern {
        Pattern::bytes(&b)
    }
}

/// Parse `\...` at `b[i]` (which is `\`). Returns (set, bytes consumed).
fn parse_escape(b: &[u8], i: usize) -> Result<(ByteSet, usize), PatternError> {
    let Some(&c) = b.get(i + 1) else { return Err(PatternError::Unterminated) };
    let digits = ByteSet::range(b'0', b'9');
    let word = digits.union(ByteSet::range(b'a', b'z')).union(ByteSet::range(b'A', b'Z')).union(ByteSet::byte(b'_'));
    let space = [b' ', b'\t', b'\n', b'\r', 0x0b, 0x0c].iter().fold(ByteSet::EMPTY, |s, &x| s.union(ByteSet::byte(x)));
    let set = match c {
        b'd' => digits,
        b'D' => digits.negate(),
        b'w' => word,
        b'W' => word.negate(),
        b's' => space,
        b'S' => space.negate(),
        b'n' => ByteSet::byte(b'\n'),
        b't' => ByteSet::byte(b'\t'),
        b'r' => ByteSet::byte(b'\r'),
        b'0' => ByteSet::byte(0),
        b'x' => {
            let h = b.get(i + 2..i + 4).ok_or(PatternError::BadEscape(i))?;
            let v = u8::from_str_radix(std::str::from_utf8(h).map_err(|_| PatternError::BadEscape(i))?, 16)
                .map_err(|_| PatternError::BadEscape(i))?;
            return Ok((ByteSet::byte(v), 4));
        }
        c if !c.is_ascii_alphanumeric() => ByteSet::byte(c),
        _ => return Err(PatternError::BadEscape(i)),
    };
    Ok((set, 2))
}

/// Parse `[...]` at `b[i]`. Returns (set, bytes consumed).
fn parse_class(b: &[u8], i: usize) -> Result<(ByteSet, usize), PatternError> {
    let mut j = i + 1;
    let negate = b.get(j) == Some(&b'^');
    if negate {
        j += 1;
    }
    let mut set = ByteSet::EMPTY;
    let mut first = true;
    loop {
        let Some(&c) = b.get(j) else { return Err(PatternError::Unterminated) };
        if c == b']' && !first {
            j += 1;
            break;
        }
        first = false;
        // One atom: a byte or an escape (escapes may be multi-byte sets).
        let (atom, n): (ByteSet, usize) = if c == b'\\' { parse_escape(b, j)? } else { (ByteSet::byte(c), 1) };
        j += n;
        // Range `lo-hi` only between single bytes.
        if b.get(j) == Some(&b'-') && b.get(j + 1).is_some_and(|&x| x != b']') && atom.is_singleton() {
            let lo = atom.first().unwrap();
            let (hi_set, m) = if b[j + 1] == b'\\' { parse_escape(b, j + 1)? } else { (ByteSet::byte(b[j + 1]), 1) };
            let Some(hi) = hi_set.first().filter(|_| hi_set.is_singleton()) else {
                return Err(PatternError::BadRange(j));
            };
            if lo > hi {
                return Err(PatternError::BadRange(j));
            }
            set = set.union(ByteSet::range(lo, hi));
            j += 1 + m;
        } else {
            set = set.union(atom);
        }
    }
    if negate {
        set = set.negate();
    }
    if set.is_empty() {
        return Err(PatternError::Empty);
    }
    Ok((set, j - i))
}

/// Compiled form of a pattern: per-position sets plus a representative
/// byte string and an `exact` flag for the memcmp fast path.
#[derive(Clone)]
pub(crate) struct Pat {
    pub bytes: Box<[u8]>,
    pub sets: Box<[ByteSet]>,
    pub exact: bool,
}

impl Pat {
    pub(crate) fn from_pattern(p: &Pattern) -> Pat {
        let sets: Box<[ByteSet]> = p.sets.clone().into_boxed_slice();
        let bytes: Box<[u8]> = sets.iter().map(|s| s.first().unwrap_or(0)).collect();
        let exact = sets.iter().all(|s| s.is_singleton());
        Pat { bytes, sets, exact }
    }
    #[inline(always)]
    pub(crate) fn len(&self) -> usize {
        self.sets.len()
    }
    /// Exact membership test of a haystack window against this pattern.
    #[inline]
    pub(crate) fn matches(&self, window: &[u8]) -> bool {
        if self.exact {
            return window == &*self.bytes;
        }
        window.iter().zip(self.sets.iter()).all(|(&h, s)| s.contains(h))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn parse_cases() {
        let p = Pattern::parse(r"GET /\d\d[a-f\x41]x.\[\]").unwrap();
        assert_eq!(p.len(), 12);
        assert!(p.sets()[5].contains(b'7') && !p.sets()[5].contains(b'a'));
        assert!(p.sets()[7].contains(b'A') && p.sets()[7].contains(b'c') && !p.sets()[7].contains(b'g'));
        assert_eq!(p.sets()[9], ByteSet::ANY);
        assert_eq!(p.sets()[10], ByteSet::byte(b'['));
        assert!(Pattern::parse("[^a]").unwrap().sets()[0].contains(b'b'));
        assert!(Pattern::parse("[]a]").unwrap().sets()[0].contains(b']'));
        assert!(Pattern::parse("[a-]").unwrap().sets()[0].contains(b'-'));
        assert_eq!(Pattern::parse("[z-a]"), Err(PatternError::BadRange(2)));
        assert_eq!(Pattern::parse("[^\\x00-\\xff]"), Err(PatternError::Empty));
        assert_eq!(Pattern::parse("ab["), Err(PatternError::Unterminated));
        assert_eq!(Pattern::parse(""), Err(PatternError::Empty));
    }
}
