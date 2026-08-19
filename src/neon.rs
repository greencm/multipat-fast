//! ARM NEON engine (aarch64): the same offset-load structure as the AVX2
//! kernel on 16-byte blocks. `TBL` (vqtbl1q_u8) is a natural 16-entry byte
//! shuffle, so the nibble tables port unchanged. Candidate extraction uses
//! the standard SHRN narrowing idiom (4 mask bits per byte in a u64) since
//! NEON has no movemask.

#![allow(unsafe_op_in_unsafe_fn)]

use crate::builder::{MAX_K, MAX_PLANES};
use crate::{scalar, verify_at, Match, ScanCtx};
use core::arch::aarch64::*;

/// SIMD blocks start at this offset; anchors below it go to the scalar
/// prelude. Any d_j <= 31 is then loadable directly (o - d_j >= 1).
const FIRST: usize = 32;

#[target_feature(enable = "neon")]
unsafe fn kernel<const K: usize, const P: usize>(ctx: &ScanCtx<'_>, hay: &[u8], out: &mut Vec<Match>) {
    let c = ctx.c;
    let len = hay.len();
    let low_mask = vdupq_n_u8(0x0F);
    let zero = vdupq_n_u8(0);

    let mut tl = [[zero; MAX_K]; MAX_PLANES];
    let mut th = [[zero; MAX_K]; MAX_PLANES];
    for p in 0..P {
        for j in 0..K {
            tl[p][j] = vld1q_u8(c.tl[p][j].as_ptr());
            th[p][j] = vld1q_u8(c.th[p][j].as_ptr());
        }
    }

    scalar::find_in_range(ctx, hay, 0, FIRST, out);

    let mut o = FIRST;
    while o + 16 <= len {
        let mut cand = [vdupq_n_u8(0xFF); MAX_PLANES];
        for j in 0..K {
            let v = vld1q_u8(hay.as_ptr().add(o - c.d[j]));
            let lo = vandq_u8(v, low_mask);
            let hi = vshrq_n_u8::<4>(v);
            for p in 0..P {
                let r = vandq_u8(vqtbl1q_u8(tl[p][j], lo), vqtbl1q_u8(th[p][j], hi));
                cand[p] = vandq_u8(cand[p], r);
            }
        }
        for p in 0..P {
            if vmaxvq_u8(cand[p]) != 0 {
                let mut bytes = [0u8; 16];
                vst1q_u8(bytes.as_mut_ptr(), cand[p]);
                // Nonzero-byte mask, 4 bits per byte: nibble t of `m` is
                // 0xF iff cand byte t is nonzero.
                let nz = vmvnq_u8(vceqq_u8(cand[p], zero));
                let m64 = vshrn_n_u16::<4>(vreinterpretq_u16_u8(nz));
                let mut m = vget_lane_u64::<0>(vreinterpret_u64_u8(m64));
                while m != 0 {
                    let t = (m.trailing_zeros() >> 2) as usize;
                    verify_at(ctx, hay, o + t, bytes[t], p, out);
                    m &= !(0xFu64 << (t * 4));
                }
            }
        }
        o += 16;
    }

    // Remaining anchors (fewer than 16) via the scalar engine.
    scalar::find_in_range(ctx, hay, o, len, out);
}

/// # Safety
/// Caller must ensure NEON is available (baseline on aarch64).
pub(crate) unsafe fn find_all(ctx: &ScanCtx<'_>, hay: &[u8], out: &mut Vec<Match>) {
    match (ctx.c.k, ctx.c.planes) {
        (1, 1) => kernel::<1, 1>(ctx, hay, out),
        (2, 1) => kernel::<2, 1>(ctx, hay, out),
        (3, 1) => kernel::<3, 1>(ctx, hay, out),
        (4, 1) => kernel::<4, 1>(ctx, hay, out),
        (1, 2) => kernel::<1, 2>(ctx, hay, out),
        (2, 2) => kernel::<2, 2>(ctx, hay, out),
        (3, 2) => kernel::<3, 2>(ctx, hay, out),
        (4, 2) => kernel::<4, 2>(ctx, hay, out),
        _ => unreachable!("k in 1..=4, planes in 1..=2"),
    }
}
