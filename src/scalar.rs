//! Portable scalar engine. Semantically identical to the SIMD paths: it
//! uses the precomposed byte -> bucket-bitmap tables (which already include
//! the nibble cross-product closure), so all engines produce the same
//! candidates up to boundary bits that verification filters out.

use crate::{verify_at, Match, ScanCtx};

/// Scan candidate anchors t in [t_start, t_end). A candidate anchored at t
/// corresponds to a window starting at t - s_last; anchors below s_last are
/// skipped because their window would begin before the haystack.
pub(crate) fn find_in_range(
    ctx: &ScanCtx<'_>,
    hay: &[u8],
    t_start: usize,
    t_end: usize,
    out: &mut Vec<Match>,
) {
    let c = ctx.c;
    let t0 = t_start.max(c.s_last);
    let t_end = t_end.min(hay.len());
    for t in t0..t_end {
        for plane in 0..c.planes {
            let tbl = &c.byte_tbl[plane];
            let mut m = 0xFFu8;
            for j in 0..c.k {
                m &= tbl[j][hay[t - c.d[j]] as usize];
                if m == 0 {
                    break;
                }
            }
            if m != 0 {
                verify_at(ctx, hay, t, m, plane, out);
            }
        }
    }
}
