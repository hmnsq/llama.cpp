//******************************************************************************
// 2D F32 convolution on the ET-SoC-1 matrix engine (GGML CONV_2D layout).
//
// LAYOUT (matches GGML's standard CONV_2D, cwhn=false; wireable directly):
//   src1 input :  ne = [W, H, Cin, N=1]      memory: input [n][cin][h][w]
//   src0 filter:  ne = [Kw, Kh, Cin, Cout]   memory: filter[oc][ic][kh][kw]
//   dst  output:  ne = [W, H, Cout, N=1]     memory: output[n][oc][h][w]
//
// CONSTRAINTS (enforced at supports_op):
//   F32 throughout, N == 1, Cin % 16 == 0, Cout % 16 == 0, positive
//   stride/pad, dilation == 1. Tile/L2SCP limits are checked here.
//
// MEMORY MODEL:
//   Each active shire uses its own 2 MB local L2 SCP:
//     filter slice | pin buffer 0 | pin buffer 1? | output staging? | scratch
//
//   The filter slice contains only the output-channel tiles (`mt`) consumed
//   by this shire's tile assignment. That keeps hart-0's inner-loop
//   tensor_loads local to the shire and avoids packing unused filter slabs.
//
// THREADING (multi-minion, multi-shire):
//   PHASE 1 (per-shire filter pack): hart-1's pack this shire's filter
//     slice into local L2 SCP. Work is slab-striped across the 32 minions.
//
//   PHASE 2 (per-shire compute): hart-1's pack the input pin chunks while
//     hart-0's run the matrix engine. Pin double-buffering hides the next
//     chunk pack behind the current chunk's FMA pipeline when Cin does not
//     fit in one local buffer.
//
// PERFORMANCE STRATEGIES:
//   1. Local filter slice: pack only the `mt` values this shire consumes;
//      inner-loop tensor_loads stay shire-local.
//   2. Pin Cin streaming + chunk double-buffer: pack one
//      chunk while computing the prior one.
//   3. TenC save/restore: f0..f31 IS the TenC accumulator;
//      spill/refill via L2 SCP scratch lets each hart hold multiple
//      partial accumulators across chunks.
//   4. OW%16 staging: for partial-tile output, write to a
//      padded L2 SCP region then have one hart scalar-emit to DRAM.
//
// WHY THE FILTER PACK EXISTS:
//   GGML's OIHW filter has stride Kh*Kw*4 between consecutive Cin elements
//   (e.g. 36 bytes for 3x3) — usually NOT a multiple of 64, so plain
//   tensor_load cannot gather it directly. The per-slab pack into a
//   Cin-innermost form gives every per-tap slab a flat 64-byte row stride
//   and enables tensor_load.
//
//   Picking M=Cout, N=W means TenC's natural row stride matches NCHW
//   output's per-channel stride (H*W*4) — the output store is a clean
//   tensor_store with no transpose. The price is that conv_size/conv_ctrl
//   no longer help with W boundaries (mask gates M, not N), so we handle
//   boundaries up-front by zero-padding the input in L2SCP.
//******************************************************************************

#ಅನ್ತರ್ಭಾವ "ggml_tensor.h"
#ಅನ್ತರ್ಭಾವ "platform.h"
#ಅನ್ತರ್ಭಾವ "tensor.h"

#ಅನ್ತರ್ಭಾವ <etsoc/common/utils.h>
#ಅನ್ತರ್ಭಾವ <stdbool.h>
#ಅನ್ತರ್ಭಾವ <stdint.h>

#ವ್ಯಾಖ್ಯಾನಿಸು TILE               16 /* matrix engine native tile in M, K, N */
/* L1 SCP layout: A double-buffered, B single-buffered. Per the SDK doc
   `dst_start` is a 6-bit field (max 63) but empirical testing shows the
   physical L1 SCP per minion is 48 lines — writes to lines >= 48 corrupt.
   So we get 3 × 16-line buffers max: A_0, A_1, B. Pick A as the
   double-buffered operand (filter-slab loads, the longer of the two). */
#ವ್ಯಾಖ್ಯಾನಿಸು LSCP_A_0           0  /* A buffer 0 at L1 SCP lines 0..15  */
#ವ್ಯಾಖ್ಯಾನಿಸು LSCP_A_1           16 /* A buffer 1 at L1 SCP lines 16..31 */
#ವ್ಯಾಖ್ಯಾನಿಸು LSCP_B             32 /* B (single buffer) at lines 32..47 */
#ವ್ಯಾಖ್ಯಾನಿಸು N_MIN_ಪ್ರತಿ_SHIRE    32 /* ET-SoC-1 geometry: 32 minions/shire  */
#ವ್ಯಾಖ್ಯಾನಿಸು N_SHIRES           32 /* default active shire count           */
#ವ್ಯಾಖ್ಯಾನಿಸು MAX_TILES_ಪ್ರತಿ_HART 2  /* per-hart TenC slots (save/restore)   */
#ವ್ಯಾಖ್ಯಾನಿಸು MAX_DBL_BUFS       2  /* chunk pack buffers (double-buffered) */

/* Per-shire L2 SCP local budget. Per-shire SCP is 2 MB; we cap at
   1984 KB to leave 64 KB headroom for per-hart TenC scratch (32 minions ×
   2 slots × 1 KB), which lives at the tail of the SCP outside the pin
   sizing budget. Bigger budget here means bigger feasible chunk_KT,
   which means fewer chunks (each chunk costs 2 SHIRE barriers + ~30
   TenC save/restore events per hart). */
#ವ್ಯಾಖ್ಯಾನಿಸು LOCAL_BUDGET (1984 * 1024)

/* Cap on the per-shire filter region in local L2 SCP. The shire packs the
   mt values it can consume under the current tile assignment, rather than
   the whole Cout dimension. Reads in the inner loop are then fully
   shire-local — no NoC fanout. */
#ವ್ಯಾಖ್ಯಾನಿಸು LOCAL_FILTER_CAP (1024 * 1024)                      /* 1 MB / shire ceiling */

#ವ್ಯಾಖ್ಯಾನಿಸು SLAB_ಬೈಟುಗಳು ((uint64_t) TILE * TILE * ಗಾತ್ರ(ದಶಮಾಂಶ)) /* 1024 */
#ವ್ಯಾಖ್ಯಾನಿಸು SLAB_ಸಾಲುಗಳು ((SLAB_ಬೈಟುಗಳು + 63) / 64)                 /* 16   */

/* Upper bound on the number of distinct mt values a single shire may pack.
   This keeps the mt list stack-resident. Shapes that need more should fall
   back until the filter-slice bookkeeping is made dynamic. */
#ವ್ಯಾಖ್ಯಾನಿಸು MAX_MY_MT (N_MIN_ಪ್ರತಿ_SHIRE * MAX_TILES_ಪ್ರತಿ_HART)

ಪ್ರಕಾರವ್ಯಾಖ್ಯೆ ರಚನೆ {
    ಪೂರ್ಣಾಂಕ mt;
    ಪೂರ್ಣಾಂಕ mt_idx;
    ಪೂರ್ಣಾಂಕ oh;
    ಪೂರ್ಣಾಂಕ ow_ತಳ;
} conv_tile_t;

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಪೂರ್ಣಾಂಕ ceil_div_i32(ಪೂರ್ಣಾಂಕ x, ಪೂರ್ಣಾಂಕ y) {
    ಹಿಂತಿರುಗಿ (x + y - 1) / y;
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಪೂರ್ಣಾಂಕ ಸುತ್ತು_ಮೇಲೆ_tile_i32(ಪೂರ್ಣಾಂಕ x) {
    ಹಿಂತಿರುಗಿ (x + TILE - 1) & ~(TILE - 1);
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಪೂರ್ಣಾಂಕ min_i32(ಪೂರ್ಣಾಂಕ a, ಪೂರ್ಣಾಂಕ b) {
    ಹಿಂತಿರುಗಿ a < b ? a : b;
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ uint64_t min_u64(uint64_t a, uint64_t b) {
    ಹಿಂತಿರುಗಿ a < b ? a : b;
}

/* ===== Vector helpers for hart-1 pack ============================
   Both assume dst (and src for copy) are 32-byte aligned; n is in floats.
   The 8-element tail is handled scalar. f30/f31 are scratch — clobbered
   per-call via the asm clobber list. */
ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ vec_zero_aligned(ದಶಮಾಂಶ * dst, ಪೂರ್ಣಾಂಕ n) {
    ಪೂರ್ಣಾಂಕ       i  = 0;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ n8 = n & ~7;
    ಗಾಗಿ (; i < n8; i += 8) {
        __asm__ ಅಸ್ಥಿರ(
            "fsub.ps f31, f31, f31\n"
            "fsw.ps  f31, %[d]\n"
            : [d] "=m"(*(ದಶಮಾಂಶ (*)[8]) & dst[i])
            :
            : "f31");
    }
    ಗಾಗಿ (; i < n; ++i) {
        dst[i] = 0.0f;
    }
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ vec_copy_aligned(ದಶಮಾಂಶ * dst, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * src, ಪೂರ್ಣಾಂಕ n) {
    ಪೂರ್ಣಾಂಕ       i  = 0;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ n8 = n & ~7;
    ಗಾಗಿ (; i < n8; i += 8) {
        __asm__ ಅಸ್ಥಿರ(
            "flw.ps f30, %[s]\n"
            "fsw.ps f30, %[d]\n"
            : [d] "=m"(*(ದಶಮಾಂಶ (*)[8]) & dst[i])
            : [s] "m"(*(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ (*)[8]) & src[i])
            : "f30");
    }
    ಗಾಗಿ (; i < n; ++i) {
        dst[i] = src[i];
    }
}

/* ===== TenC save/restore =========================================
   The TenC accumulator IS the f0..f31 vector register file: row N occupies
   f(2N) and f(2N+1) (two 8-fp32 vector regs per row). We save by
   tensor_store-ing TILE rows × 64 bytes, and restore via 32 flw.ps after
   forcing L1D to refetch from the L2SCP backing (tensor_store bypasses L1D
   so the backing is always current). See feedback_tenc_save_restore.md. */
ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ tenc_restore_from_scratch(uint64_t scr) {
    FENCE;
    evict_ಗಮ್ಯ_l2((ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *) scr, TILE, 64);
    WAIT_CACHEOPS;
    __asm__ ಅಸ್ಥಿರ(
        "flw.ps f0,    0(%0)\n"
        "flw.ps f1,   32(%0)\n"
        "flw.ps f2,   64(%0)\n"
        "flw.ps f3,   96(%0)\n"
        "flw.ps f4,  128(%0)\n"
        "flw.ps f5,  160(%0)\n"
        "flw.ps f6,  192(%0)\n"
        "flw.ps f7,  224(%0)\n"
        "flw.ps f8,  256(%0)\n"
        "flw.ps f9,  288(%0)\n"
        "flw.ps f10, 320(%0)\n"
        "flw.ps f11, 352(%0)\n"
        "flw.ps f12, 384(%0)\n"
        "flw.ps f13, 416(%0)\n"
        "flw.ps f14, 448(%0)\n"
        "flw.ps f15, 480(%0)\n"
        "flw.ps f16, 512(%0)\n"
        "flw.ps f17, 544(%0)\n"
        "flw.ps f18, 576(%0)\n"
        "flw.ps f19, 608(%0)\n"
        "flw.ps f20, 640(%0)\n"
        "flw.ps f21, 672(%0)\n"
        "flw.ps f22, 704(%0)\n"
        "flw.ps f23, 736(%0)\n"
        "flw.ps f24, 768(%0)\n"
        "flw.ps f25, 800(%0)\n"
        "flw.ps f26, 832(%0)\n"
        "flw.ps f27, 864(%0)\n"
        "flw.ps f28, 896(%0)\n"
        "flw.ps f29, 928(%0)\n"
        "flw.ps f30, 960(%0)\n"
        "flw.ps f31, 992(%0)\n"
        :
        : "r"(scr)
        : "f0", "f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8", "f9", "f10", "f11", "f12", "f13", "f14", "f15", "f16",
          "f17", "f18", "f19", "f20", "f21", "f22", "f23", "f24", "f25", "f26", "f27", "f28", "f29", "f30", "f31",
          "memory");
}

/* ===== Pin pack context ==========================================
   Loop-invariant state hart-1 needs to pack one Cin chunk's worth of
   pin (Kw shifted, padded copies of input rows) into local L2 SCP. The
   filter is not touched in this struct; it is packed into the per-shire
   local slice before the per-chunk loop begins. */
ಪ್ರಕಾರವ್ಯಾಖ್ಯೆ ರಚನೆ {
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * ಒಳ_ತಳ;  /* DRAM input base [Cin][H][W]         */
    ಪೂರ್ಣಾಂಕ           Kw;
    ಪೂರ್ಣಾಂಕ           chunk_KT; /* number of K_TILES (=16-wide) per chunk */
    ಪೂರ್ಣಾಂಕ           H, W, Hp, Wp_a;
    ಪೂರ್ಣಾಂಕ           pad_h, pad_w, s0;
    ಪೂರ್ಣಾಂಕ           minion;          /* this hart's minion id (0..31) */
    uint64_t      pin_copy_floats; /* per-_s pin plane size in floats */
    uint64_t      l2_pad_ಒಳ_buf[MAX_DBL_BUFS];
    uint64_t      pin_chunk_ಬೈಟುಗಳು; /* one chunk pin buffer's total size */
} pin_ctx_t;

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಪೂರ್ಣಾಂಕ ಕಂಡುಹಿಡಿ_mt_idx(ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ * my_mt, ಪೂರ್ಣಾಂಕ n_my_mt, ಪೂರ್ಣಾಂಕ mt) {
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < n_my_mt; ++j) {
        ಆದರೆ (my_mt[j] == mt) {
            ಹಿಂತಿರುಗಿ j;
        }
    }
    ಹಿಂತಿರುಗಿ 0;
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ conv_tile_t decode_tile(ಪೂರ್ಣಾಂಕ t, ಪೂರ್ಣಾಂಕ M_TILES, ಪೂರ್ಣಾಂಕ w_tiles, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ * my_mt, ಪೂರ್ಣಾಂಕ n_my_mt) {
    conv_tile_t tile;
    tile.mt = t % M_TILES;
    t /= M_TILES;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ wt = t % w_tiles;
    t /= w_tiles;
    tile.oh      = t;
    tile.ow_ತಳ = wt * TILE;
    tile.mt_idx  = ಕಂಡುಹಿಡಿ_mt_idx(my_mt, n_my_mt, tile.mt);
    ಹಿಂತಿರುಗಿ tile;
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ uint64_t
filter_slab_addr(uint64_t l2_filter, ಪೂರ್ಣಾಂಕ Kw, ಪೂರ್ಣಾಂಕ K_TILES, ಪೂರ್ಣಾಂಕ n_my_mt, ಪೂರ್ಣಾಂಕ mt_idx, ಪೂರ್ಣಾಂಕ kh, ಪೂರ್ಣಾಂಕ kw, ಪೂರ್ಣಾಂಕ kt_global) {
    ಹಿಂತಿರುಗಿ l2_filter + (uint64_t) ((((kh * Kw + kw) * n_my_mt + mt_idx) * K_TILES + kt_global)) * SLAB_ಬೈಟುಗಳು;
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ uint64_t pin_tile_addr(uint64_t l2_pad_ಒಳ,
                                     uint64_t pin_copy_ಬೈಟುಗಳು,
                                     ಪೂರ್ಣಾಂಕ      ktc,
                                     ಪೂರ್ಣಾಂಕ      kw,
                                     ಪೂರ್ಣಾಂಕ      Hp,
                                     ಪೂರ್ಣಾಂಕ      Wp_a,
                                     ಪೂರ್ಣಾಂಕ      oh,
                                     ಪೂರ್ಣಾಂಕ      ow_ತಳ,
                                     ಪೂರ್ಣಾಂಕ      s1,
                                     ಪೂರ್ಣಾಂಕ      kh) {
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ir_pad = oh * s1 + kh;
    ಹಿಂತಿರುಗಿ l2_pad_ಒಳ + (uint64_t) kw * pin_copy_ಬೈಟುಗಳು +
           (((uint64_t) (ktc * TILE) * Hp + ir_pad) * Wp_a + ow_ತಳ) * ಗಾತ್ರ(ದಶಮಾಂಶ);
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಅಕ್ಷರ * output_tile_addr(ಅಕ್ಷರ *              ಹೊರಹೋಗು_ತಳ,
                                      ಸ್ಥಿರಾಂಕ conv_tile_t * tile,
                                      uint64_t            ಹೊರಹೋಗು_chan_stride,
                                      uint64_t            ಹೊರಹೋಗು_ಅಡ್ಡಸಾಲು_stride) {
    ಹಿಂತಿರುಗಿ ಹೊರಹೋಗು_ತಳ + (size_t) (tile->mt * TILE) * ಹೊರಹೋಗು_chan_stride + (size_t) tile->oh * ಹೊರಹೋಗು_ಅಡ್ಡಸಾಲು_stride +
           (size_t) tile->ow_ತಳ * ಗಾತ್ರ(ದಶಮಾಂಶ);
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ flush_range_ಗಮ್ಯ_l2(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * addr, uint64_t n_ಬೈಟುಗಳು) {
    ಸ್ಥಿರಾಂಕ uint64_t ಒಟ್ಟು_ಸಾಲುಗಳು = (n_ಬೈಟುಗಳು + 63) / 64;
    ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *   fl_addr     = (ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *) addr;
    ಗಾಗಿ (uint64_t ಮುಗಿದ = 0; ಮುಗಿದ < ಒಟ್ಟು_ಸಾಲುಗಳು;) {
        ಸ್ಥಿರಾಂಕ uint64_t batch = min_u64(ಒಟ್ಟು_ಸಾಲುಗಳು - ಮುಗಿದ, 16);
        flush_ಗಮ್ಯ_l2((ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *) (fl_addr + ಮುಗಿದ * 64), batch, 64);
        ಮುಗಿದ += batch;
    }
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ evict_range_past_l2(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * addr, uint64_t n_ಬೈಟುಗಳು) {
    ಸ್ಥಿರಾಂಕ uint64_t ಒಟ್ಟು_ಸಾಲುಗಳು = (n_ಬೈಟುಗಳು + 63) / 64;
    ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *   fl_addr     = (ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *) addr;
    ಗಾಗಿ (uint64_t ಮುಗಿದ = 0; ಮುಗಿದ < ಒಟ್ಟು_ಸಾಲುಗಳು;) {
        ಸ್ಥಿರಾಂಕ uint64_t batch = min_u64(ಒಟ್ಟು_ಸಾಲುಗಳು - ಮುಗಿದ, 16);
        evict_past_l2((ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *) (fl_addr + ಮುಗಿದ * 64), batch, 64);
        ಮುಗಿದ += batch;
    }
}

/* One matrix-engine tile for one Cin chunk. This is the main optimization
   surface: A is double-buffered, B is single-buffered due to L1 SCP space. */
ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ compute_tile_chunk(uint64_t            l2_filter,
                                      uint64_t            l2_pad_ಒಳ,
                                      uint64_t            pin_copy_ಬೈಟುಗಳು,
                                      ಪೂರ್ಣಾಂಕ                 Kh,
                                      ಪೂರ್ಣಾಂಕ                 Kw,
                                      ಪೂರ್ಣಾಂಕ                 K_TILES,
                                      ಪೂರ್ಣಾಂಕ                 chunk_KT,
                                      ಪೂರ್ಣಾಂಕ                 kt_ತಳ,
                                      ಪೂರ್ಣಾಂಕ                 n_my_mt,
                                      ಪೂರ್ಣಾಂಕ                 Hp,
                                      ಪೂರ್ಣಾಂಕ                 Wp_a,
                                      ಪೂರ್ಣಾಂಕ                 s1,
                                      uint64_t            a_ಅಡ್ಡಸಾಲು_stride,
                                      uint64_t            b_ಅಡ್ಡಸಾಲು_stride,
                                      ಸ್ಥಿರಾಂಕ conv_tile_t * tile,
                                      bool                ಮೊದಲ_fma_clears_tenc) {
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ      n_iters   = Kh * Kw * chunk_KT;
    ಸ್ಥಿರಾಂಕ uint64_t A_BUFS[2] = { LSCP_A_0, LSCP_A_1 };

    ಸ್ಥಿರಾಂಕ uint64_t a_addr0 = filter_slab_addr(l2_filter, Kw, K_TILES, n_my_mt, tile->mt_idx, 0, 0, kt_ತಳ);
    tensor_ತುಂಬು(false, false, A_BUFS[0], 0, 0, a_addr0, 0, (uint64_t) (TILE - 1), a_ಅಡ್ಡಸಾಲು_stride, 0);

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ iter = 0; iter < n_iters; ++iter) {
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ktc = iter % chunk_KT;
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ rem = iter / chunk_KT;
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ kw  = rem % Kw;
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ kh  = rem / Kw;

        ಸ್ಥಿರಾಂಕ uint64_t b_addr =
            pin_tile_addr(l2_pad_ಒಳ, pin_copy_ಬೈಟುಗಳು, ktc, kw, Hp, Wp_a, tile->oh, tile->ow_ತಳ, s1, kh);
        tensor_ತುಂಬು(false, false, LSCP_B, 0, 0, b_addr, 0, (uint64_t) (TILE - 1), b_ಅಡ್ಡಸಾಲು_stride, 1);

        tensor_wait(TENSOR_ತುಂಬು_WAIT_0);
        tensor_wait(TENSOR_ತುಂಬು_WAIT_1);

        ಆದರೆ (iter + 1 < n_iters) {
            ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ      ktc_n = (iter + 1) % chunk_KT;
            ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ      rem_n = (iter + 1) / chunk_KT;
            ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ      kw_n  = rem_n % Kw;
            ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ      kh_n  = rem_n / Kw;
            ಸ್ಥಿರಾಂಕ uint64_t a_addr_n =
                filter_slab_addr(l2_filter, Kw, K_TILES, n_my_mt, tile->mt_idx, kh_n, kw_n, kt_ತಳ + ktc_n);
            tensor_ತುಂಬು(false, false, A_BUFS[(iter + 1) & 1], 0, 0, a_addr_n, 0, (uint64_t) (TILE - 1), a_ಅಡ್ಡಸಾಲು_stride,
                        0);
        }

        tensor_fma(false, 3, (uint64_t) (TILE - 1), (uint64_t) (TILE - 1), 0, false, false, false, false, LSCP_B,
                   A_BUFS[iter & 1], 0, ಮೊದಲ_fma_clears_tenc && (iter == 0));
        tensor_wait(TENSOR_FMA_WAIT);
    }
}

/* Pack only the slabs this shire's tiles actually consume, into local
   L2 SCP. Slab layout in the filter buffer is [Kh][Kw][n_my_mt][K_TILES]
   of TILE×TILE slabs (Cin-innermost form). Distributed across the 32
   hart-1's of this shire by `slab % 32 == minion`.

   This deliberately favors local inner-loop reads over global filter fanout.
   Depending on tile shape, two shires may pack the same mt value; keep that
   tradeoff visible when experimenting with shared-filter layouts. */
ಸ್ಥಿರ ಶೂನ್ಯ pack_filter_local_mt(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * flt_ತಳ,
                                 ಪೂರ್ಣಾಂಕ           Kh,
                                 ಪೂರ್ಣಾಂಕ           Kw,
                                 ಪೂರ್ಣಾಂಕ           Cin,
                                 ಪೂರ್ಣಾಂಕ           K_TILES,
                                 ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ *   my_mt,
                                 ಪೂರ್ಣಾಂಕ           n_my_mt,
                                 ಪೂರ್ಣಾಂಕ           minion,
                                 uint64_t      l2_filter_ತಳ) {
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ    n_slabs = Kh * Kw * n_my_mt * K_TILES;
    ಸ್ಥಿರಾಂಕ size_t kstep   = (size_t) Kh * Kw; /* Cin stride in floats */

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ slab = minion; slab < n_slabs; slab += N_MIN_ಪ್ರತಿ_SHIRE) {
        ಪೂರ್ಣಾಂಕ       t  = slab;
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ kt = t % K_TILES;
        t /= K_TILES;
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ mt_idx = t % n_my_mt;
        t /= n_my_mt;
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ kw = t % Kw;
        t /= Kw;
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ kh = t;
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ mt = my_mt[mt_idx];

        ಸ್ಥಿರಾಂಕ uint64_t slab_ಸ್ಥಾನಾಂತರ = (uint64_t) slab * SLAB_ಬೈಟುಗಳು;
        ದಶಮಾಂಶ *        cell        = (ದಶಮಾಂಶ *) (l2_filter_ತಳ + slab_ಸ್ಥಾನಾಂತರ);

        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ oc_ಒಳ = 0; oc_ಒಳ < TILE; ++oc_ಒಳ) {
            ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ     oc  = mt * TILE + oc_ಒಳ;
            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * src = flt_ತಳ + (((size_t) oc * Cin + (size_t) kt * TILE) * Kh + kh) * Kw + kw;
            ದಶಮಾಂಶ *       ಅಡ್ಡಸಾಲು = cell + (size_t) oc_ಒಳ * TILE;
            ದಶಮಾಂಶ         scratch[TILE] __attribute__((aligned(32)));
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ ic_ಒಳ = 0; ic_ಒಳ < TILE; ++ic_ಒಳ) {
                scratch[ic_ಒಳ] = src[(size_t) ic_ಒಳ * kstep];
            }
            vec_copy_aligned(ಅಡ್ಡಸಾಲು, scratch, TILE);
        }
    }

    /* Flush this hart's dirty L1D lines for the slabs it wrote. */
    FENCE;
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ slab = minion; slab < n_slabs; slab += N_MIN_ಪ್ರತಿ_SHIRE) {
        ಸ್ಥಿರಾಂಕ uint64_t slab_ಸ್ಥಾನಾಂತರ = (uint64_t) slab * SLAB_ಬೈಟುಗಳು;
        flush_ಗಮ್ಯ_l2((ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *) (l2_filter_ತಳ + slab_ಸ್ಥಾನಾಂತರ), SLAB_ಸಾಲುಗಳು, 64);
    }
    WAIT_CACHEOPS;
}

/* Pack one Cin chunk of the input pin (Kw shifted padded copies) into the
   buf_idx side of local L2SCP. Work distributed across the 32 hart-1's in
   the shire by `plane % 32 == minion`. The final flush_to_l2 forces L1D
   write-back so hart-0's tensor_load sees the freshly written bytes. */
ಸ್ಥಿರ ಶೂನ್ಯ pack_pin_chunk(ಸ್ಥಿರಾಂಕ pin_ctx_t * ctx, ಪೂರ್ಣಾಂಕ chunk_id, ಪೂರ್ಣಾಂಕ buf_idx) {
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ kt_ತಳ  = chunk_id * ctx->chunk_KT;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ Kw       = ctx->Kw;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ chunk_KT = ctx->chunk_KT;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ H = ctx->H, W = ctx->W, Hp = ctx->Hp, Wp_a = ctx->Wp_a;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ pad_h = ctx->pad_h, pad_w = ctx->pad_w, s0 = ctx->s0;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ minion = ctx->minion;

    /* Pin pack: Kw shifted, padded copies of input rows. Bounds [vlo, vhi)
       hoisted outside the row loop so the inner loop is three regions
       (zero-prefix | bulk-copy | zero-suffix) with no per-element predicate. */
    ದಶಮಾಂಶ *   pin0         = (ದಶಮಾಂಶ *) ctx->l2_pad_ಒಳ_buf[buf_idx];
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ chunk_Cin    = chunk_KT * TILE;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ n_pin_planes = Kw * chunk_Cin;
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ p = minion; p < n_pin_planes; p += N_MIN_ಪ್ರತಿ_SHIRE) {
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ s     = p / chunk_Cin;
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ icc   = p % chunk_Cin;
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ic    = kt_ತಳ * TILE + icc;
        ದಶಮಾಂಶ *   pin_s = pin0 + (size_t) s * ctx->pin_copy_floats;

        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಸ್ಥಾನಾಂತರ = s - pad_w;
        ಪೂರ್ಣಾಂಕ       vlo    = 0;
        ಆಗ (vlo < Wp_a && (s0 * vlo + ಸ್ಥಾನಾಂತರ) < 0) {
            vlo++;
        }
        ಪೂರ್ಣಾಂಕ vhi = Wp_a;
        ಆಗ (vhi > vlo && (s0 * (vhi - 1) + ಸ್ಥಾನಾಂತರ) >= W) {
            vhi--;
        }
        ಸ್ಥಿರಾಂಕ bool aligned = (s0 == 1) && ((vlo & 7) == 0) && (((vlo + ಸ್ಥಾನಾಂತರ) & 7) == 0);

        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ r = 0; r < Hp; ++r) {
            ದಶಮಾಂಶ *   ಅಡ್ಡಸಾಲು    = pin_s + ((size_t) icc * Hp + r) * Wp_a;
            ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ real_h = r - pad_h;
            ಆದರೆ (real_h < 0 || real_h >= H) {
                vec_zero_aligned(ಅಡ್ಡಸಾಲು, Wp_a);
                ಮುಂದುವರಿ;
            }
            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * src_ಅಡ್ಡಸಾಲು = ctx->ಒಳ_ತಳ + ((size_t) ic * H + real_h) * W;

            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ cc = 0; cc < vlo; ++cc) {
                ಅಡ್ಡಸಾಲು[cc] = 0.0f;
            }

            ಆದರೆ (aligned) {
                vec_copy_aligned(ಅಡ್ಡಸಾಲು + vlo, src_ಅಡ್ಡಸಾಲು + vlo + ಸ್ಥಾನಾಂತರ, vhi - vlo);
            } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (s0 == 1) {
                ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * csrc = src_ಅಡ್ಡಸಾಲು + vlo + ಸ್ಥಾನಾಂತರ;
                ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ     n    = vhi - vlo;
                ಗಾಗಿ (ಪೂರ್ಣಾಂಕ cc = 0; cc < n; ++cc) {
                    ಅಡ್ಡಸಾಲು[vlo + cc] = csrc[cc];
                }
            } ಇಲ್ಲದಿದ್ದರೆ {
                ಗಾಗಿ (ಪೂರ್ಣಾಂಕ cc = vlo; cc < vhi; ++cc) {
                    ಅಡ್ಡಸಾಲು[cc] = src_ಅಡ್ಡಸಾಲು[s0 * cc + ಸ್ಥಾನಾಂತರ];
                }
            }

            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ cc = vhi; cc < Wp_a; ++cc) {
                ಅಡ್ಡಸಾಲು[cc] = 0.0f;
            }
        }
    }

    /* Flush this buffer's L1D-dirty lines down to L2SCP backing. */
    FENCE;
    flush_range_ಗಮ್ಯ_l2((ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *) ctx->l2_pad_ಒಳ_buf[buf_idx], ctx->pin_chunk_ಬೈಟುಗಳು);
    WAIT_CACHEOPS;
}

ಪೂರ್ಣಾಂಕ ನಮೂದು_ಬಿಂದು(ರಚನೆ ggml_et_ದ್ವಿಮಾನ_params * params, ಶೂನ್ಯ * env) {
    (ಶೂನ್ಯ) env;

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ shire   = get_shire_id();
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ hart_id = get_hart_id();
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ minion  = (hart_id >> 1) & 0x1F;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ hart1   = hart_id & 1;

    ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * flt = &params->src0; /* [Kw,Kh,Cin,Cout] */
    ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * ಒಳ  = &params->src1; /* [W, H, Cin,N=1 ] */
    ರಚನೆ ggml_tensor *       ಹೊರಹೋಗು = &params->dst;  /* [W, H, Cout,N=1] */

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ Kw   = (ಪೂರ್ಣಾಂಕ) flt->ne[0];
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ Kh   = (ಪೂರ್ಣಾಂಕ) flt->ne[1];
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ Cin  = (ಪೂರ್ಣಾಂಕ) flt->ne[2];
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ Cout = (ಪೂರ್ಣಾಂಕ) flt->ne[3];

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ W  = (ಪೂರ್ಣಾಂಕ) ಒಳ->ne[0];
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ H  = (ಪೂರ್ಣಾಂಕ) ಒಳ->ne[1];
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ OW = (ಪೂರ್ಣಾಂಕ) ಹೊರಹೋಗು->ne[0];
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ OH = (ಪೂರ್ಣಾಂಕ) ಹೊರಹೋಗು->ne[1];

    /* op_params layout (set by ggml_conv_2d):
         [0]=s0  [1]=s1  [2]=p0  [3]=p1  [4]=d0  [5]=d1   */
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ s0    = ಹೊರಹೋಗು->op_params[0];
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ s1    = ಹೊರಹೋಗು->op_params[1];
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ pad_w = ಹೊರಹೋಗು->op_params[2];
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ pad_h = ಹೊರಹೋಗು->op_params[3];

    ಆದರೆ (Cin <= 0 || Cout <= 0) {
        ಹಿಂತಿರುಗಿ -1;
    }
    ಆದರೆ (Cin % TILE != 0 || Cout % TILE != 0) {
        ಹಿಂತಿರುಗಿ -1;
    }
    ಆದರೆ (W <= 0 || H <= 0) {
        ಹಿಂತಿರುಗಿ -1;
    }
    ಆದರೆ (s0 <= 0 || s1 <= 0) {
        ಹಿಂತಿರುಗಿ -1;
    }
    ಆದರೆ (ಒಳ->ne[2] != Cin || ಒಳ->ne[3] != 1) {
        ಹಿಂತಿರುಗಿ -1;
    }
    ಆದರೆ (ಹೊರಹೋಗು->ne[2] != Cout || ಹೊರಹೋಗು->ne[3] != 1) {
        ಹಿಂತಿರುಗಿ -1;
    }
    ಆದರೆ (!flt->ದತ್ತಾಂಶ || !ಒಳ->ದತ್ತಾಂಶ || !ಹೊರಹೋಗು->ದತ್ತಾಂಶ) {
        ಹಿಂತಿರುಗಿ -1;
    }

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ K_TILES = Cin / TILE;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ M_TILES = Cout / TILE;

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ  Hp         = H + 2 * pad_h;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ  Wp_a       = ಸುತ್ತು_ಮೇಲೆ_tile_i32(OW);
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ  OW_pad     = Wp_a;
    ಸ್ಥಿರಾಂಕ bool need_stage = (OW % TILE != 0);

    /* ===================== Tile assignment & active-shire selection =====
       Computed up front because the per-shire mt set (and thus filter
       region size) depends on n_active_shires. */
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ w_tiles         = ceil_div_i32(OW, TILE);
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಒಟ್ಟು_tiles     = OH * w_tiles * M_TILES;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ n_ಸಕ್ರಿಯ_shires = need_stage ? 1 : min_i32(ಒಟ್ಟು_tiles, N_SHIRES);

    /* Inactive shires exit immediately. No global barrier — pack and
       barriers are now per-shire, so unused shires don't need to vote. */
    ಆದರೆ (shire >= n_ಸಕ್ರಿಯ_shires) {
        ಹಿಂತಿರುಗಿ 0;
    }

    /* ===================== Determine this shire's mt set ================
       Standard tile assignment: tile t is owned by
         shire  = t % n_active_shires
         minion = (t / n_active_shires) % N_MIN_PER_SHIRE
         slot   = t / (n_active_shires * N_MIN_PER_SHIRE)
       So the set of mt's this shire actually consumes is the set of
       (t % M_TILES) for all t this shire owns. Enumerate all shire-owned
       tiles, not just the first MAX_TILES_PER_HART slots; the one-chunk
       path can process more tiles serially. */
    ಪೂರ್ಣಾಂಕ my_mt[MAX_MY_MT];
    ಪೂರ್ಣಾಂಕ n_my_mt = 0;
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ t = shire; t < ಒಟ್ಟು_tiles; t += n_ಸಕ್ರಿಯ_shires) {
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ mt    = t % M_TILES;
        bool      ಸಿಕ್ಕಿತು = false;
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < n_my_mt; ++j) {
            ಆದರೆ (my_mt[j] == mt) {
                ಸಿಕ್ಕಿತು = true;
                ನಿಲ್ಲಿಸು;
            }
        }
        ಆದರೆ (!ಸಿಕ್ಕಿತು) {
            ಆದರೆ (n_my_mt >= MAX_MY_MT) {
                ಹಿಂತಿರುಗಿ -1;
            }
            my_mt[n_my_mt++] = mt;
        }
    }
    ಆದರೆ (n_my_mt == 0) {
        ಹಿಂತಿರುಗಿ 0; /* no tiles for this shire */
    }

    ಸ್ಥಿರಾಂಕ uint64_t filter_local_ಬೈಟುಗಳು = (uint64_t) Kh * Kw * n_my_mt * K_TILES * SLAB_ಬೈಟುಗಳು;
    ಆದರೆ (filter_local_ಬೈಟುಗಳು > LOCAL_FILTER_CAP) {
        ಹಿಂತಿರುಗಿ -1;
    }

    /* ===================== L2 SCP local layout =========================
       filter (this shire's mt slice) | pin_buf[0] | pin_buf[1]?
                                      | output_stage? | scratch (streaming) */
    ಸ್ಥಿರಾಂಕ uint64_t l2_ತಳ   = (uint64_t) et_shire_l2scp_local(0);
    ಸ್ಥಿರಾಂಕ uint64_t l2_filter = l2_ತಳ;

    /* Sizing for pin: budget = LOCAL_BUDGET - filter - output_stage.    */
    ಸ್ಥಿರಾಂಕ int64_t output_stage_ಬೈಟುಗಳು_ತುಂಬಿದ = need_stage ? (int64_t) Cout * OH * OW_pad * (int64_t) ಗಾತ್ರ(ದಶಮಾಂಶ) : 0;
    ಸ್ಥಿರಾಂಕ int64_t budget_for_chunks = (int64_t) LOCAL_BUDGET - (int64_t) filter_local_ಬೈಟುಗಳು - output_stage_ಬೈಟುಗಳು_ತುಂಬಿದ;
    ಆದರೆ (budget_for_chunks <= 0) {
        ಹಿಂತಿರುಗಿ -1;
    }
    ಸ್ಥಿರಾಂಕ int64_t ಪ್ರತಿ_KT_pin_ಬೈಟುಗಳು = (int64_t) Kw * TILE * Hp * Wp_a * (int64_t) ಗಾತ್ರ(ದಶಮಾಂಶ);

    ಪೂರ್ಣಾಂಕ chunk_KT;
    ಪೂರ್ಣಾಂಕ n_buffers;
    ಆದರೆ ((int64_t) K_TILES * ಪ್ರತಿ_KT_pin_ಬೈಟುಗಳು <= budget_for_chunks) {
        chunk_KT  = K_TILES;
        n_buffers = 1;
    } ಇಲ್ಲದಿದ್ದರೆ {
        chunk_KT = K_TILES;
        ಆಗ (chunk_KT > 1 && 2 * (int64_t) chunk_KT * ಪ್ರತಿ_KT_pin_ಬೈಟುಗಳು > budget_for_chunks) {
            chunk_KT--;
        }
        ಆಗ (chunk_KT > 1 && K_TILES % chunk_KT != 0) {
            chunk_KT--;
        }
        n_buffers = (chunk_KT < K_TILES) ? 2 : 1;
        ಆದರೆ (chunk_KT < 1) {
            ಹಿಂತಿರುಗಿ -1;
        }
    }
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ n_chunks = K_TILES / chunk_KT;

    /* Streaming keeps partial sums in MAX_TILES_PER_HART scratch slots per
       hart. The one-chunk path does not need scratch and can stream a longer
       tile list serially, but multi-chunk shapes must fit this fixed slot
       count until scratch scheduling is made more general. */
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ shire_tile_ಸಾಮರ್ಥ್ಯ = shire + MAX_TILES_ಪ್ರತಿ_HART * n_ಸಕ್ರಿಯ_shires * N_MIN_ಪ್ರತಿ_SHIRE;
    ಆದರೆ (n_chunks > 1 && shire_tile_ಸಾಮರ್ಥ್ಯ < ಒಟ್ಟು_tiles) {
        ಹಿಂತಿರುಗಿ -1;
    }

    ಸ್ಥಿರಾಂಕ uint64_t pin_copy_floats = (uint64_t) chunk_KT * TILE * Hp * Wp_a;
    ಸ್ಥಿರಾಂಕ uint64_t pin_copy_ಬೈಟುಗಳು  = pin_copy_floats * ಗಾತ್ರ(ದಶಮಾಂಶ);
    ಸ್ಥಿರಾಂಕ uint64_t pin_chunk_ಬೈಟುಗಳು = (uint64_t) Kw * pin_copy_ಬೈಟುಗಳು;

    ಸ್ಥಿರಾಂಕ uint64_t l2_pin_ತಳ              = l2_filter + filter_local_ಬೈಟುಗಳು;
    ಸ್ಥಿರಾಂಕ uint64_t l2_pin_buf[MAX_DBL_BUFS] = {
        l2_pin_ತಳ,
        l2_pin_ತಳ + pin_chunk_ಬೈಟುಗಳು,
    };

    ಸ್ಥಿರಾಂಕ uint64_t l2_output_stage = need_stage ? l2_pin_ತಳ + (uint64_t) n_buffers * pin_chunk_ಬೈಟುಗಳು : 0;

    ಸ್ಥಿರಾಂಕ uint64_t scratch_ಪ್ರತಿ_hart = (uint64_t) MAX_TILES_ಪ್ರತಿ_HART * (uint64_t) TILE * TILE * ಗಾತ್ರ(ದಶಮಾಂಶ);
    ಸ್ಥಿರಾಂಕ uint64_t l2_scratch_ತಳ  = need_stage ? l2_output_stage + (uint64_t) output_stage_ಬೈಟುಗಳು_ತುಂಬಿದ :
                                                   l2_pin_ತಳ + (uint64_t) n_buffers * pin_chunk_ಬೈಟುಗಳು;

    /* ===================== PHASE 1: Filter pack (per-shire mt slice) ====
       Hart-1's pack only this shire's mt slabs into local L2 SCP. The
       SHIRE barrier below ensures the filter is in L2 SCP backing before
       hart-0's first tensor_load. */
    ಆದರೆ (hart1) {
        pack_filter_local_mt((ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) flt->ದತ್ತಾಂಶ, Kh, Kw, Cin, K_TILES, my_mt, n_my_mt, minion, l2_filter);
    }

    /* ===================== Hart 1: pin packer (per chunk) ==============
       Double-buffered prefetch: pack chunk 0 synchronously, then per chunk c
       signal "buf c ready", pack chunk c+1 into the alternate buffer
       (overlaps hart-0's compute on c), signal "buf c done". */
    ಆದರೆ (hart1) {
        ಸ್ಥಿರಾಂಕ pin_ctx_t ctx = {
            .ಒಳ_ತಳ         = (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) ಒಳ->ದತ್ತಾಂಶ,
            .Kw              = Kw,
            .chunk_KT        = chunk_KT,
            .H               = H,
            .W               = W,
            .Hp              = Hp,
            .Wp_a            = Wp_a,
            .pad_h           = pad_h,
            .pad_w           = pad_w,
            .s0              = s0,
            .minion          = minion,
            .pin_copy_floats = pin_copy_floats,
            .l2_pad_ಒಳ_buf   = { l2_pin_buf[0], l2_pin_buf[1] },
            .pin_chunk_ಬೈಟುಗಳು = pin_chunk_ಬೈಟುಗಳು,
        };

        pack_pin_chunk(&ctx, 0, 0); /* prologue */

        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ c = 0; c < n_chunks; ++c) {
            et_barrier(ET_BARRIER_SHIRE); /* signal "buf c ready" */
            ಆದರೆ (n_buffers > 1 && c + 1 < n_chunks) {
                pack_pin_chunk(&ctx, c + 1, (c + 1) & 1);
            }
            et_barrier(ET_BARRIER_SHIRE); /* wait "buf c done" */
        }

        ಆದರೆ (need_stage) {
            et_barrier(ET_BARRIER_SHIRE);
        }
        ಹಿಂತಿರುಗಿ 0;
    }

    /* ===================== Hart 0: matrix engine ======================
       Two execution modes:
         - n_chunks == 1: full Cin in one shot. Each hart processes a list
           of tiles serially; TenC resets between tiles via first_pass=true.
         - n_chunks  > 1: streaming. Each hart owns up to MAX_TILES_PER_HART
           tiles. For each chunk c, restore TenC from scratch[k] (skip on
           c==0), accumulate this chunk's FMAs, then either save TenC back
           to scratch[k] (c < last) or tensor_store directly (c == last). */
    ಸಿದ್ಧತೆ_cache_scp();
    CLEAR_TENSOR_ERROR;

    ಅಕ್ಷರ * ಸ್ಥಿರಾಂಕ   ಹೊರಹೋಗು_ತಳ        = need_stage ? (ಅಕ್ಷರ *) l2_output_stage : (ಅಕ್ಷರ *) ಹೊರಹೋಗು->ದತ್ತಾಂಶ;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ      compute_OW      = need_stage ? OW_pad : OW;
    ಸ್ಥಿರಾಂಕ uint64_t ಹೊರಹೋಗು_chan_stride = (uint64_t) OH * (uint64_t) compute_OW * ಗಾತ್ರ(ದಶಮಾಂಶ);
    ಸ್ಥಿರಾಂಕ uint64_t ಹೊರಹೋಗು_ಅಡ್ಡಸಾಲು_stride  = (uint64_t) compute_OW * ಗಾತ್ರ(ದಶಮಾಂಶ);

    ಸ್ಥಿರಾಂಕ uint64_t a_ಅಡ್ಡಸಾಲು_stride = (uint64_t) TILE * ಗಾತ್ರ(ದಶಮಾಂಶ); /* 64 */
    ಸ್ಥಿರಾಂಕ uint64_t b_ಅಡ್ಡಸಾಲು_stride = (uint64_t) Hp * (uint64_t) Wp_a * ಗಾತ್ರ(ದಶಮಾಂಶ);

    /* Tile assignment: shire-strided so small workloads spread across
       shires before stacking minions in one shire. */
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ t_ಆರಂಭ  = shire + minion * n_ಸಕ್ರಿಯ_shires;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ t_stride = n_ಸಕ್ರಿಯ_shires * N_MIN_ಪ್ರತಿ_SHIRE;

    ಆದರೆ (n_chunks == 1) {
        et_barrier(ET_BARRIER_SHIRE); /* wait for the (only) pin chunk */

        ಸ್ಥಿರಾಂಕ uint64_t l2_pad_ಒಳ = l2_pin_buf[0];
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ t = t_ಆರಂಭ; t < ಒಟ್ಟು_tiles; t += t_stride) {
            ಸ್ಥಿರಾಂಕ conv_tile_t tile = decode_tile(t, M_TILES, w_tiles, my_mt, n_my_mt);
            compute_tile_chunk(l2_filter, l2_pad_ಒಳ, pin_copy_ಬೈಟುಗಳು, Kh, Kw, K_TILES, chunk_KT, 0, n_my_mt, Hp, Wp_a,
                               s1, a_ಅಡ್ಡಸಾಲು_stride, b_ಅಡ್ಡಸಾಲು_stride, &tile, /*first_fma_clears_tenc=*/true);

            ಅಕ್ಷರ * dst_addr = output_tile_addr(ಹೊರಹೋಗು_ತಳ, &tile, ಹೊರಹೋಗು_chan_stride, ಹೊರಹೋಗು_ಅಡ್ಡಸಾಲು_stride);
            tensor_store(0, 0, 3, (uint64_t) (TILE - 1), (uint64_t) dst_addr, 0, ಹೊರಹೋಗು_chan_stride);
            tensor_wait(TENSOR_STORE_WAIT);
        }

        et_barrier(ET_BARRIER_SHIRE); /* matches hart-1's second barrier */

    } ಇಲ್ಲದಿದ್ದರೆ {
        /* Streaming path: each hart owns up to MAX_TILES_PER_HART tiles. */
        ಪೂರ್ಣಾಂಕ my_tiles[MAX_TILES_ಪ್ರತಿ_HART];
        ಪೂರ್ಣಾಂಕ n_my_tiles = 0;
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ ಸ್ಥಾನ = 0; ಸ್ಥಾನ < MAX_TILES_ಪ್ರತಿ_HART; ++ಸ್ಥಾನ) {
            ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ t = t_ಆರಂಭ + ಸ್ಥಾನ * t_stride;
            ಆದರೆ (t < ಒಟ್ಟು_tiles) {
                my_tiles[n_my_tiles++] = t;
            }
        }

        conv_tile_t tiles[MAX_TILES_ಪ್ರತಿ_HART];
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ k = 0; k < n_my_tiles; ++k) {
            tiles[k] = decode_tile(my_tiles[k], M_TILES, w_tiles, my_mt, n_my_mt);
        }

        ಸ್ಥಿರಾಂಕ uint64_t my_scratch_ತಳ = l2_scratch_ತಳ + (uint64_t) minion * scratch_ಪ್ರತಿ_hart;

        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ c = 0; c < n_chunks; ++c) {
            et_barrier(ET_BARRIER_SHIRE); /* pin chunk c packed */

            ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ      buf       = c & 1;
            ಸ್ಥಿರಾಂಕ uint64_t l2_pad_ಒಳ = l2_pin_buf[buf];
            ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ      kt_ತಳ   = c * chunk_KT;

            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ k = 0; k < n_my_tiles; ++k) {
                ಸ್ಥಿರಾಂಕ conv_tile_t * tile = &tiles[k];
                ಸ್ಥಿರಾಂಕ uint64_t      scr  = my_scratch_ತಳ + (uint64_t) k * (TILE * TILE * ಗಾತ್ರ(ದಶಮಾಂಶ));

                ಸ್ಥಿರಾಂಕ bool ಮೊದಲ_pass_chunk = (c == 0);
                ಆದರೆ (!ಮೊದಲ_pass_chunk) {
                    tenc_restore_from_scratch(scr);
                }

                compute_tile_chunk(l2_filter, l2_pad_ಒಳ, pin_copy_ಬೈಟುಗಳು, Kh, Kw, K_TILES, chunk_KT, kt_ತಳ, n_my_mt,
                                   Hp, Wp_a, s1, a_ಅಡ್ಡಸಾಲು_stride, b_ಅಡ್ಡಸಾಲು_stride, tile, ಮೊದಲ_pass_chunk);

                ಆದರೆ (c == n_chunks - 1) {
                    ಅಕ್ಷರ * dst_addr = output_tile_addr(ಹೊರಹೋಗು_ತಳ, tile, ಹೊರಹೋಗು_chan_stride, ಹೊರಹೋಗು_ಅಡ್ಡಸಾಲು_stride);
                    tensor_store(0, 0, 3, (uint64_t) (TILE - 1), (uint64_t) dst_addr, 0, ಹೊರಹೋಗು_chan_stride);
                } ಇಲ್ಲದಿದ್ದರೆ {
                    tensor_store(0, 0, 3, (uint64_t) (TILE - 1), (uint64_t) scr, 0, 64);
                }
                tensor_wait(TENSOR_STORE_WAIT);
            }

            et_barrier(ET_BARRIER_SHIRE); /* hart-0 done with chunk c */
        }
    }

    FENCE;

    /* ----------------------- DRAM emit phase ---------------------------
       Only relevant when we staged into L2SCP because OW % 16 != 0. */
    ಆದರೆ (need_stage) {
        et_barrier(ET_BARRIER_SHIRE);

        ಆದರೆ (minion == 0) {
            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * stage = (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) l2_output_stage;
            ದಶಮಾಂಶ *       dram  = (ದಶಮಾಂಶ *) ಹೊರಹೋಗು->ದತ್ತಾಂಶ;
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ oc = 0; oc < Cout; ++oc) {
                ಗಾಗಿ (ಪೂರ್ಣಾಂಕ oh2 = 0; oh2 < OH; ++oh2) {
                    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * src = stage + ((size_t) oc * OH + oh2) * OW_pad;
                    ದಶಮಾಂಶ *       dst = dram + ((size_t) oc * OH + oh2) * OW;
                    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ ow2 = 0; ow2 < OW; ++ow2) {
                        dst[ow2] = src[ow2];
                    }
                }
            }
            FENCE;
            ಸ್ಥಿರಾಂಕ uint64_t ಒಟ್ಟು_ಬೈಟುಗಳು = (uint64_t) Cout * OH * OW * ಗಾತ್ರ(ದಶಮಾಂಶ);
            evict_range_past_l2((ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *) dram, ಒಟ್ಟು_ಬೈಟುಗಳು);
            WAIT_CACHEOPS;
        }
    }

    ಹಿಂತಿರುಗಿ 0;
}
