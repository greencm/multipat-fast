//! AVX2 engine: per 32-byte block, one shuffle-pair + shift + AND per
//! sampled position, then movemask-driven candidate extraction.
//!
//! For each sampled position j (with carry distance d_j = s_last - s_j) the
//! kernel computes a per-byte bucket bitmap R_j where byte t answers "does
//! hay[t] fall in some bucket's closure class at position j?", then aligns
//! all bitmaps to the anchor position via byte shifts that borrow the tail
//! of the previous block, and ANDs them. A surviving bit (bucket, t) means
//! every sampled byte of a window starting at t - s_last was compatible with
//! that bucket, and only then is the exact verifier consulted.

#![allow(unsafe_op_in_unsafe_fn)]

use crate::builder::{Compiled, MAX_K};
use crate::{scalar, verify_at, Match};
use core::arch::x86_64::*;

/// result[t] = concat(prev, cur)[32 + t - d], i.e. the current block's view
/// of R shifted d bytes toward higher addresses, borrowing from prev.
#[inline(always)]
unsafe fn shift_carry(prev: __m256i, cur: __m256i, d: usize) -> __m256i {
    macro_rules! sc {
        ($n:literal) => {
            _mm256_alignr_epi8::<$n>(cur, _mm256_permute2x128_si256(prev, cur, 0x21))
        };
    }
    match d {
        0 => cur,
        1 => sc!(15),
        2 => sc!(14),
        3 => sc!(13),
        4 => sc!(12),
        5 => sc!(11),
        6 => sc!(10),
        7 => sc!(9),
        8 => sc!(8),
        9 => sc!(7),
        10 => sc!(6),
        11 => sc!(5),
        12 => sc!(4),
        13 => sc!(3),
        14 => sc!(2),
        15 => sc!(1),
        _ => unreachable!("carry distance bounded by the 16-byte anchor window"),
    }
}

#[target_feature(enable = "avx2")]
pub(crate) unsafe fn find_all(c: &Compiled, hay: &[u8], out: &mut Vec<Match>) {
    let len = hay.len();
    let k = c.k;
    debug_assert!(k >= 1 && k <= MAX_K);

    let low_mask = _mm256_set1_epi8(0x0F);
    let zero = _mm256_setzero_si256();

    // Broadcast the 16-byte nibble tables to both 128-bit lanes.
    let mut tl = [zero; MAX_K];
    let mut th = [zero; MAX_K];
    for j in 0..k {
        tl[j] = _mm256_broadcastsi128_si256(_mm_loadu_si128(c.tl[j].as_ptr() as *const __m128i));
        th[j] = _mm256_broadcastsi128_si256(_mm_loadu_si128(c.th[j].as_ptr() as *const __m128i));
    }

    // Class-mask carry state; zero means "no byte before the buffer matches",
    // which is exactly the semantics we want at the start of the haystack.
    let mut prev = [zero; MAX_K];

    let mut o = 0usize;
    while o + 32 <= len {
        let input = _mm256_loadu_si256(hay.as_ptr().add(o) as *const __m256i);
        let lo_nib = _mm256_and_si256(input, low_mask);
        let hi_nib = _mm256_and_si256(_mm256_srli_epi16::<4>(input), low_mask);

        let mut cand = _mm256_set1_epi8(-1i8);
        for j in 0..k {
            let r = _mm256_and_si256(
                _mm256_shuffle_epi8(tl[j], lo_nib),
                _mm256_shuffle_epi8(th[j], hi_nib),
            );
            cand = _mm256_and_si256(cand, shift_carry(prev[j], r, c.shifts[j] as usize));
            prev[j] = r;
        }

        let nz = !(_mm256_movemask_epi8(_mm256_cmpeq_epi8(cand, zero)) as u32);
        if nz != 0 {
            let mut bytes = [0u8; 32];
            _mm256_storeu_si256(bytes.as_mut_ptr() as *mut __m256i, cand);
            let mut m = nz;
            while m != 0 {
                let t = m.trailing_zeros() as usize;
                verify_at(c, hay, o + t, bytes[t], out);
                m &= m - 1;
            }
        }
        o += 32;
    }

    // Remaining anchors (fewer than 32) via the scalar engine.
    scalar::find_in_range(c, hay, o, len, out);
}
