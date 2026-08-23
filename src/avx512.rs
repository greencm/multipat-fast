//! AVX-512BW engine: same offset-load structure as the AVX2 kernel but on
//! 64-byte blocks, with candidate extraction via `vptestmb` (a nonzero-byte
//! mask register in one instruction, no compare + movemask round trip).

#![allow(unsafe_op_in_unsafe_fn)]

use crate::builder::{MAX_K, MAX_PLANES};
use crate::{scalar, verify_at, ScanCtx, Sink};
use core::arch::x86_64::*;

/// SIMD blocks start at this offset; anchors below it go to the scalar
/// prelude. Any d_j <= 31 is then loadable directly (o - d_j >= 1).
const FIRST: usize = 32;

#[target_feature(enable = "avx512f,avx512bw")]
unsafe fn kernel<const K: usize, const P: usize, S: Sink>(ctx: &ScanCtx<'_>, hay: &[u8], out: &mut S) {
    let c = ctx.c;
    let len = hay.len();
    let low_mask = _mm512_set1_epi8(0x0F);
    let ones = _mm512_set1_epi8(-1i8);

    // Broadcast the 16-byte nibble tables to all four 128-bit lanes.
    let mut tl = [[_mm512_setzero_si512(); MAX_K]; MAX_PLANES];
    let mut th = [[_mm512_setzero_si512(); MAX_K]; MAX_PLANES];
    for p in 0..P {
        for j in 0..K {
            tl[p][j] =
                _mm512_broadcast_i32x4(_mm_loadu_si128(c.tl[p][j].as_ptr() as *const __m128i));
            th[p][j] =
                _mm512_broadcast_i32x4(_mm_loadu_si128(c.th[p][j].as_ptr() as *const __m128i));
        }
    }

    scalar::find_in_range(ctx, hay, 0, FIRST, out);

    let mut o = FIRST;
    while o + 64 <= len {
        let mut cand = [ones; MAX_PLANES];
        for j in 0..K {
            let v = _mm512_loadu_si512(hay.as_ptr().add(o - c.d[j]) as *const __m512i);
            let lo = _mm512_and_si512(v, low_mask);
            let hi = _mm512_and_si512(_mm512_srli_epi16::<4>(v), low_mask);
            for p in 0..P {
                let r = _mm512_and_si512(
                    _mm512_shuffle_epi8(tl[p][j], lo),
                    _mm512_shuffle_epi8(th[p][j], hi),
                );
                cand[p] = _mm512_and_si512(cand[p], r);
            }
        }
        for p in 0..P {
            let mut m: u64 = _mm512_test_epi8_mask(cand[p], cand[p]);
            if m != 0 {
                let mut bytes = [0u8; 64];
                _mm512_storeu_si512(bytes.as_mut_ptr() as *mut __m512i, cand[p]);
                while m != 0 {
                    let t = m.trailing_zeros() as usize;
                    verify_at(ctx, hay, o + t, bytes[t], p, out);
                    m &= m - 1;
                }
            }
        }
        o += 64;
    }

    // Remaining anchors (fewer than 64) via the scalar engine.
    scalar::find_in_range(ctx, hay, o, len, out);
}

/// # Safety
/// Caller must ensure AVX-512F and AVX-512BW are available.
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
