//! AVX2 engine. Per 32-byte block and sampled position j, the kernel loads
//! the haystack *at offset -d_j* (one unaligned load) so every class mask is
//! born already aligned to the anchor position — no carry registers, no
//! cross-lane byte shifts. One shuffle-pair + AND per position per plane,
//! then movemask-driven candidate extraction. Monomorphized over the number
//! of sampled positions K and bucket planes P so the position loop unrolls.

#![allow(unsafe_op_in_unsafe_fn)]

use crate::builder::{MAX_K, MAX_PLANES};
use crate::{scalar, verify_at, ScanCtx, Sink};
use core::arch::x86_64::*;

/// SIMD blocks start at this offset; anchors below it go to the scalar
/// prelude. Any d_j <= 31 is then loadable directly (o - d_j >= 1).
const FIRST: usize = 32;

#[target_feature(enable = "avx2")]
unsafe fn kernel<const K: usize, const P: usize, S: Sink>(ctx: &ScanCtx<'_>, hay: &[u8], out: &mut S) {
    let c = ctx.c;
    let len = hay.len();
    let low_mask = _mm256_set1_epi8(0x0F);
    let zero = _mm256_setzero_si256();
    let ones = _mm256_set1_epi8(-1i8);

    // Broadcast the 16-byte nibble tables to both 128-bit lanes.
    let mut tl = [[zero; MAX_K]; MAX_PLANES];
    let mut th = [[zero; MAX_K]; MAX_PLANES];
    for p in 0..P {
        for j in 0..K {
            tl[p][j] =
                _mm256_broadcastsi128_si256(_mm_loadu_si128(c.tl[p][j].as_ptr() as *const __m128i));
            th[p][j] =
                _mm256_broadcastsi128_si256(_mm_loadu_si128(c.th[p][j].as_ptr() as *const __m128i));
        }
    }

    scalar::find_in_range(ctx, hay, 0, FIRST, out);

    let mut o = FIRST;
    while o + 32 <= len {
        let mut cand = [ones; MAX_PLANES];
        for j in 0..K {
            let v = _mm256_loadu_si256(hay.as_ptr().add(o - c.d[j]) as *const __m256i);
            let lo = _mm256_and_si256(v, low_mask);
            let hi = _mm256_and_si256(_mm256_srli_epi16::<4>(v), low_mask);
            for p in 0..P {
                let r = _mm256_and_si256(
                    _mm256_shuffle_epi8(tl[p][j], lo),
                    _mm256_shuffle_epi8(th[p][j], hi),
                );
                cand[p] = _mm256_and_si256(cand[p], r);
            }
        }
        for p in 0..P {
            let nz = !(_mm256_movemask_epi8(_mm256_cmpeq_epi8(cand[p], zero)) as u32);
            if nz != 0 {
                let mut bytes = [0u8; 32];
                _mm256_storeu_si256(bytes.as_mut_ptr() as *mut __m256i, cand[p]);
                let mut m = nz;
                while m != 0 {
                    let t = m.trailing_zeros() as usize;
                    verify_at(ctx, hay, o + t, bytes[t], p, out);
                    m &= m - 1;
                }
            }
        }
        o += 32;
    }

    // Remaining anchors (fewer than 32) via the scalar engine.
    scalar::find_in_range(ctx, hay, o, len, out);
}

/// # Safety
/// Caller must ensure AVX2 is available.
pub(crate) unsafe fn find_all<S: Sink>(ctx: &ScanCtx<'_>, hay: &[u8], out: &mut S) {
    match (ctx.c.k, ctx.c.planes) {
        (1, 1) => kernel::<1, 1, S>(ctx, hay, out),
        (2, 1) => kernel::<2, 1, S>(ctx, hay, out),
        (3, 1) => kernel::<3, 1, S>(ctx, hay, out),
        (4, 1) => kernel::<4, 1, S>(ctx, hay, out),
        (1, 2) => kernel::<1, 2, S>(ctx, hay, out),
        (2, 2) => kernel::<2, 2, S>(ctx, hay, out),
        (3, 2) => kernel::<3, 2, S>(ctx, hay, out),
        (4, 2) => kernel::<4, 2, S>(ctx, hay, out),
        (1, 4) => kernel::<1, 4, S>(ctx, hay, out),
        (2, 4) => kernel::<2, 4, S>(ctx, hay, out),
        (3, 4) => kernel::<3, 4, S>(ctx, hay, out),
        (4, 4) => kernel::<4, 4, S>(ctx, hay, out),
        _ => unreachable!("k in 1..=4, planes in 1|2|4"),
    }
}
