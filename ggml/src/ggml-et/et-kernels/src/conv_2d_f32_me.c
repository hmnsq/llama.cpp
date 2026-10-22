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

#ਅਨ੍ਤਰ੍ਭਾਵ "ggml_tensor.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "platform.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "tensor.h"

#ਅਨ੍ਤਰ੍ਭਾਵ <etsoc/common/utils.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <stdbool.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <stdint.h>

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ TILE               16 /* matrix engine native tile in M, K, N */
/* L1 SCP layout: A double-buffered, B single-buffered. Per the SDK doc
   `dst_start` is a 6-bit field (max 63) but empirical testing shows the
   physical L1 SCP per minion is 48 lines — writes to lines >= 48 corrupt.
   So we get 3 × 16-line buffers max: A_0, A_1, B. Pick A as the
   double-buffered operand (filter-slab loads, the longer of the two). */
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ LSCP_A_0           0  /* A buffer 0 at L1 SCP lines 0..15  */
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ LSCP_A_1           16 /* A buffer 1 at L1 SCP lines 16..31 */
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ LSCP_B             32 /* B (single buffer) at lines 32..47 */
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ N_MIN_ਪ੍ਰਤੀ_SHIRE    32 /* ET-SoC-1 geometry: 32 minions/shire  */
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ N_SHIRES           32 /* default active shire count           */
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ MAX_TILES_ਪ੍ਰਤੀ_HART 2  /* per-hart TenC slots (save/restore)   */
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ MAX_DBL_BUFS       2  /* chunk pack buffers (double-buffered) */

/* Per-shire L2 SCP local budget. Per-shire SCP is 2 MB; we cap at
   1984 KB to leave 64 KB headroom for per-hart TenC scratch (32 minions ×
   2 slots × 1 KB), which lives at the tail of the SCP outside the pin
   sizing budget. Bigger budget here means bigger feasible chunk_KT,
   which means fewer chunks (each chunk costs 2 SHIRE barriers + ~30
   TenC save/restore events per hart). */
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ LOCAL_BUDGET (1984 * 1024)

/* Cap on the per-shire filter region in local L2 SCP. The shire packs the
   mt values it can consume under the current tile assignment, rather than
   the whole Cout dimension. Reads in the inner loop are then fully
   shire-local — no NoC fanout. */
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ LOCAL_FILTER_CAP (1024 * 1024)                      /* 1 MB / shire ceiling */

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ SLAB_ਬਾਈਟਾਂ ((uint64_t) TILE * TILE * ਆਕਾਰ(ਦਸ਼ਮਲਵ)) /* 1024 */
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ SLAB_ਸਤਰਾਂ ((SLAB_ਬਾਈਟਾਂ + 63) / 64)                 /* 16   */

/* Upper bound on the number of distinct mt values a single shire may pack.
   This keeps the mt list stack-resident. Shapes that need more should fall
   back until the filter-slice bookkeeping is made dynamic. */
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ MAX_MY_MT (N_MIN_ਪ੍ਰਤੀ_SHIRE * MAX_TILES_ਪ੍ਰਤੀ_HART)

ਕਿਸਮਪਰਿਭਾਸ਼ਾ ਬਣਤਰ {
    ਪੂਰਨ mt;
    ਪੂਰਨ mt_idx;
    ਪੂਰਨ oh;
    ਪੂਰਨ ow_ਅਧਾਰ;
} conv_tile_t;

ਸਥਿਰ ਸਤਰਅੰਦਰ ਪੂਰਨ ceil_div_i32(ਪੂਰਨ x, ਪੂਰਨ y) {
    ਵਾਪਸ (x + y - 1) / y;
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਪੂਰਨ ਗੇੜਾ_ਉੱਪਰ_tile_i32(ਪੂਰਨ x) {
    ਵਾਪਸ (x + TILE - 1) & ~(TILE - 1);
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਪੂਰਨ min_i32(ਪੂਰਨ a, ਪੂਰਨ b) {
    ਵਾਪਸ a < b ? a : b;
}

ਸਥਿਰ ਸਤਰਅੰਦਰ uint64_t min_u64(uint64_t a, uint64_t b) {
    ਵਾਪਸ a < b ? a : b;
}

/* ===== Vector helpers for hart-1 pack ============================
   Both assume dst (and src for copy) are 32-byte aligned; n is in floats.
   The 8-element tail is handled scalar. f30/f31 are scratch — clobbered
   per-call via the asm clobber list. */
ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ vec_zero_aligned(ਦਸ਼ਮਲਵ * dst, ਪੂਰਨ n) {
    ਪੂਰਨ       i  = 0;
    ਅਚਲ ਪੂਰਨ n8 = n & ~7;
    ਲਈ (; i < n8; i += 8) {
        __asm__ ਅਸਥਿਰ(
            "fsub.ps f31, f31, f31\n"
            "fsw.ps  f31, %[d]\n"
            : [d] "=m"(*(ਦਸ਼ਮਲਵ (*)[8]) & dst[i])
            :
            : "f31");
    }
    ਲਈ (; i < n; ++i) {
        dst[i] = 0.0f;
    }
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ vec_copy_aligned(ਦਸ਼ਮਲਵ * dst, ਅਚਲ ਦਸ਼ਮਲਵ * src, ਪੂਰਨ n) {
    ਪੂਰਨ       i  = 0;
    ਅਚਲ ਪੂਰਨ n8 = n & ~7;
    ਲਈ (; i < n8; i += 8) {
        __asm__ ਅਸਥਿਰ(
            "flw.ps f30, %[s]\n"
            "fsw.ps f30, %[d]\n"
            : [d] "=m"(*(ਦਸ਼ਮਲਵ (*)[8]) & dst[i])
            : [s] "m"(*(ਅਚਲ ਦਸ਼ਮਲਵ (*)[8]) & src[i])
            : "f30");
    }
    ਲਈ (; i < n; ++i) {
        dst[i] = src[i];
    }
}

/* ===== TenC save/restore =========================================
   The TenC accumulator IS the f0..f31 vector register file: row N occupies
   f(2N) and f(2N+1) (two 8-fp32 vector regs per row). We save by
   tensor_store-ing TILE rows × 64 bytes, and restore via 32 flw.ps after
   forcing L1D to refetch from the L2SCP backing (tensor_store bypasses L1D
   so the backing is always current). See feedback_tenc_save_restore.md. */
ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ tenc_restore_from_scratch(uint64_t scr) {
    FENCE;
    evict_ਮੰਜ਼ਿਲ_l2((ਅਚਲ ਖਾਲੀ *) scr, TILE, 64);
    WAIT_CACHEOPS;
    __asm__ ਅਸਥਿਰ(
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
ਕਿਸਮਪਰਿਭਾਸ਼ਾ ਬਣਤਰ {
    ਅਚਲ ਦਸ਼ਮਲਵ * ਵਿੱਚ_ਅਧਾਰ;  /* DRAM input base [Cin][H][W]         */
    ਪੂਰਨ           Kw;
    ਪੂਰਨ           chunk_KT; /* number of K_TILES (=16-wide) per chunk */
    ਪੂਰਨ           H, W, Hp, Wp_a;
    ਪੂਰਨ           pad_h, pad_w, s0;
    ਪੂਰਨ           minion;          /* this hart's minion id (0..31) */
    uint64_t      pin_copy_floats; /* per-_s pin plane size in floats */
    uint64_t      l2_pad_ਵਿੱਚ_buf[MAX_DBL_BUFS];
    uint64_t      pin_chunk_ਬਾਈਟਾਂ; /* one chunk pin buffer's total size */
} pin_ctx_t;

ਸਥਿਰ ਸਤਰਅੰਦਰ ਪੂਰਨ ਲੱਭੋ_mt_idx(ਅਚਲ ਪੂਰਨ * my_mt, ਪੂਰਨ n_my_mt, ਪੂਰਨ mt) {
    ਲਈ (ਪੂਰਨ j = 0; j < n_my_mt; ++j) {
        ਜੇ (my_mt[j] == mt) {
            ਵਾਪਸ j;
        }
    }
    ਵਾਪਸ 0;
}

ਸਥਿਰ ਸਤਰਅੰਦਰ conv_tile_t decode_tile(ਪੂਰਨ t, ਪੂਰਨ M_TILES, ਪੂਰਨ w_tiles, ਅਚਲ ਪੂਰਨ * my_mt, ਪੂਰਨ n_my_mt) {
    conv_tile_t tile;
    tile.mt = t % M_TILES;
    t /= M_TILES;
    ਅਚਲ ਪੂਰਨ wt = t % w_tiles;
    t /= w_tiles;
    tile.oh      = t;
    tile.ow_ਅਧਾਰ = wt * TILE;
    tile.mt_idx  = ਲੱਭੋ_mt_idx(my_mt, n_my_mt, tile.mt);
    ਵਾਪਸ tile;
}

ਸਥਿਰ ਸਤਰਅੰਦਰ uint64_t
filter_slab_addr(uint64_t l2_filter, ਪੂਰਨ Kw, ਪੂਰਨ K_TILES, ਪੂਰਨ n_my_mt, ਪੂਰਨ mt_idx, ਪੂਰਨ kh, ਪੂਰਨ kw, ਪੂਰਨ kt_global) {
    ਵਾਪਸ l2_filter + (uint64_t) ((((kh * Kw + kw) * n_my_mt + mt_idx) * K_TILES + kt_global)) * SLAB_ਬਾਈਟਾਂ;
}

ਸਥਿਰ ਸਤਰਅੰਦਰ uint64_t pin_tile_addr(uint64_t l2_pad_ਵਿੱਚ,
                                     uint64_t pin_copy_ਬਾਈਟਾਂ,
                                     ਪੂਰਨ      ktc,
                                     ਪੂਰਨ      kw,
                                     ਪੂਰਨ      Hp,
                                     ਪੂਰਨ      Wp_a,
                                     ਪੂਰਨ      oh,
                                     ਪੂਰਨ      ow_ਅਧਾਰ,
                                     ਪੂਰਨ      s1,
                                     ਪੂਰਨ      kh) {
    ਅਚਲ ਪੂਰਨ ir_pad = oh * s1 + kh;
    ਵਾਪਸ l2_pad_ਵਿੱਚ + (uint64_t) kw * pin_copy_ਬਾਈਟਾਂ +
           (((uint64_t) (ktc * TILE) * Hp + ir_pad) * Wp_a + ow_ਅਧਾਰ) * ਆਕਾਰ(ਦਸ਼ਮਲਵ);
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਅੱਖਰ * output_tile_addr(ਅੱਖਰ *              ਬਾਹਰਲਾ_ਅਧਾਰ,
                                      ਅਚਲ conv_tile_t * tile,
                                      uint64_t            ਬਾਹਰਲਾ_chan_stride,
                                      uint64_t            ਬਾਹਰਲਾ_ਪੰਕਤੀ_stride) {
    ਵਾਪਸ ਬਾਹਰਲਾ_ਅਧਾਰ + (size_t) (tile->mt * TILE) * ਬਾਹਰਲਾ_chan_stride + (size_t) tile->oh * ਬਾਹਰਲਾ_ਪੰਕਤੀ_stride +
           (size_t) tile->ow_ਅਧਾਰ * ਆਕਾਰ(ਦਸ਼ਮਲਵ);
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ flush_range_ਮੰਜ਼ਿਲ_l2(ਅਚਲ ਖਾਲੀ * addr, uint64_t n_ਬਾਈਟਾਂ) {
    ਅਚਲ uint64_t ਕੁੱਲ_ਸਤਰਾਂ = (n_ਬਾਈਟਾਂ + 63) / 64;
    ਅਚਲ ਅੱਖਰ *   fl_addr     = (ਅਚਲ ਅੱਖਰ *) addr;
    ਲਈ (uint64_t ਪੂਰਾ = 0; ਪੂਰਾ < ਕੁੱਲ_ਸਤਰਾਂ;) {
        ਅਚਲ uint64_t batch = min_u64(ਕੁੱਲ_ਸਤਰਾਂ - ਪੂਰਾ, 16);
        flush_ਮੰਜ਼ਿਲ_l2((ਅਚਲ ਖਾਲੀ *) (fl_addr + ਪੂਰਾ * 64), batch, 64);
        ਪੂਰਾ += batch;
    }
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ evict_range_past_l2(ਅਚਲ ਖਾਲੀ * addr, uint64_t n_ਬਾਈਟਾਂ) {
    ਅਚਲ uint64_t ਕੁੱਲ_ਸਤਰਾਂ = (n_ਬਾਈਟਾਂ + 63) / 64;
    ਅਚਲ ਅੱਖਰ *   fl_addr     = (ਅਚਲ ਅੱਖਰ *) addr;
    ਲਈ (uint64_t ਪੂਰਾ = 0; ਪੂਰਾ < ਕੁੱਲ_ਸਤਰਾਂ;) {
        ਅਚਲ uint64_t batch = min_u64(ਕੁੱਲ_ਸਤਰਾਂ - ਪੂਰਾ, 16);
        evict_past_l2((ਅਚਲ ਖਾਲੀ *) (fl_addr + ਪੂਰਾ * 64), batch, 64);
        ਪੂਰਾ += batch;
    }
}

/* One matrix-engine tile for one Cin chunk. This is the main optimization
   surface: A is double-buffered, B is single-buffered due to L1 SCP space. */
ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ compute_tile_chunk(uint64_t            l2_filter,
                                      uint64_t            l2_pad_ਵਿੱਚ,
                                      uint64_t            pin_copy_ਬਾਈਟਾਂ,
                                      ਪੂਰਨ                 Kh,
                                      ਪੂਰਨ                 Kw,
                                      ਪੂਰਨ                 K_TILES,
                                      ਪੂਰਨ                 chunk_KT,
                                      ਪੂਰਨ                 kt_ਅਧਾਰ,
                                      ਪੂਰਨ                 n_my_mt,
                                      ਪੂਰਨ                 Hp,
                                      ਪੂਰਨ                 Wp_a,
                                      ਪੂਰਨ                 s1,
                                      uint64_t            a_ਪੰਕਤੀ_stride,
                                      uint64_t            b_ਪੰਕਤੀ_stride,
                                      ਅਚਲ conv_tile_t * tile,
                                      bool                ਪਹਿਲਾ_fma_clears_tenc) {
    ਅਚਲ ਪੂਰਨ      n_iters   = Kh * Kw * chunk_KT;
    ਅਚਲ uint64_t A_BUFS[2] = { LSCP_A_0, LSCP_A_1 };

    ਅਚਲ uint64_t a_addr0 = filter_slab_addr(l2_filter, Kw, K_TILES, n_my_mt, tile->mt_idx, 0, 0, kt_ਅਧਾਰ);
    tensor_ਲੱਦੋ(false, false, A_BUFS[0], 0, 0, a_addr0, 0, (uint64_t) (TILE - 1), a_ਪੰਕਤੀ_stride, 0);

    ਲਈ (ਪੂਰਨ iter = 0; iter < n_iters; ++iter) {
        ਅਚਲ ਪੂਰਨ ktc = iter % chunk_KT;
        ਅਚਲ ਪੂਰਨ rem = iter / chunk_KT;
        ਅਚਲ ਪੂਰਨ kw  = rem % Kw;
        ਅਚਲ ਪੂਰਨ kh  = rem / Kw;

        ਅਚਲ uint64_t b_addr =
            pin_tile_addr(l2_pad_ਵਿੱਚ, pin_copy_ਬਾਈਟਾਂ, ktc, kw, Hp, Wp_a, tile->oh, tile->ow_ਅਧਾਰ, s1, kh);
        tensor_ਲੱਦੋ(false, false, LSCP_B, 0, 0, b_addr, 0, (uint64_t) (TILE - 1), b_ਪੰਕਤੀ_stride, 1);

        tensor_wait(TENSOR_ਲੱਦੋ_WAIT_0);
        tensor_wait(TENSOR_ਲੱਦੋ_WAIT_1);

        ਜੇ (iter + 1 < n_iters) {
            ਅਚਲ ਪੂਰਨ      ktc_n = (iter + 1) % chunk_KT;
            ਅਚਲ ਪੂਰਨ      rem_n = (iter + 1) / chunk_KT;
            ਅਚਲ ਪੂਰਨ      kw_n  = rem_n % Kw;
            ਅਚਲ ਪੂਰਨ      kh_n  = rem_n / Kw;
            ਅਚਲ uint64_t a_addr_n =
                filter_slab_addr(l2_filter, Kw, K_TILES, n_my_mt, tile->mt_idx, kh_n, kw_n, kt_ਅਧਾਰ + ktc_n);
            tensor_ਲੱਦੋ(false, false, A_BUFS[(iter + 1) & 1], 0, 0, a_addr_n, 0, (uint64_t) (TILE - 1), a_ਪੰਕਤੀ_stride,
                        0);
        }

        tensor_fma(false, 3, (uint64_t) (TILE - 1), (uint64_t) (TILE - 1), 0, false, false, false, false, LSCP_B,
                   A_BUFS[iter & 1], 0, ਪਹਿਲਾ_fma_clears_tenc && (iter == 0));
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
ਸਥਿਰ ਖਾਲੀ pack_filter_local_mt(ਅਚਲ ਦਸ਼ਮਲਵ * flt_ਅਧਾਰ,
                                 ਪੂਰਨ           Kh,
                                 ਪੂਰਨ           Kw,
                                 ਪੂਰਨ           Cin,
                                 ਪੂਰਨ           K_TILES,
                                 ਅਚਲ ਪੂਰਨ *   my_mt,
                                 ਪੂਰਨ           n_my_mt,
                                 ਪੂਰਨ           minion,
                                 uint64_t      l2_filter_ਅਧਾਰ) {
    ਅਚਲ ਪੂਰਨ    n_slabs = Kh * Kw * n_my_mt * K_TILES;
    ਅਚਲ size_t kstep   = (size_t) Kh * Kw; /* Cin stride in floats */

    ਲਈ (ਪੂਰਨ slab = minion; slab < n_slabs; slab += N_MIN_ਪ੍ਰਤੀ_SHIRE) {
        ਪੂਰਨ       t  = slab;
        ਅਚਲ ਪੂਰਨ kt = t % K_TILES;
        t /= K_TILES;
        ਅਚਲ ਪੂਰਨ mt_idx = t % n_my_mt;
        t /= n_my_mt;
        ਅਚਲ ਪੂਰਨ kw = t % Kw;
        t /= Kw;
        ਅਚਲ ਪੂਰਨ kh = t;
        ਅਚਲ ਪੂਰਨ mt = my_mt[mt_idx];

        ਅਚਲ uint64_t slab_ਵਿੱਥ = (uint64_t) slab * SLAB_ਬਾਈਟਾਂ;
        ਦਸ਼ਮਲਵ *        cell        = (ਦਸ਼ਮਲਵ *) (l2_filter_ਅਧਾਰ + slab_ਵਿੱਥ);

        ਲਈ (ਪੂਰਨ oc_ਵਿੱਚ = 0; oc_ਵਿੱਚ < TILE; ++oc_ਵਿੱਚ) {
            ਅਚਲ ਪੂਰਨ     oc  = mt * TILE + oc_ਵਿੱਚ;
            ਅਚਲ ਦਸ਼ਮਲਵ * src = flt_ਅਧਾਰ + (((size_t) oc * Cin + (size_t) kt * TILE) * Kh + kh) * Kw + kw;
            ਦਸ਼ਮਲਵ *       ਪੰਕਤੀ = cell + (size_t) oc_ਵਿੱਚ * TILE;
            ਦਸ਼ਮਲਵ         scratch[TILE] __attribute__((aligned(32)));
            ਲਈ (ਪੂਰਨ ic_ਵਿੱਚ = 0; ic_ਵਿੱਚ < TILE; ++ic_ਵਿੱਚ) {
                scratch[ic_ਵਿੱਚ] = src[(size_t) ic_ਵਿੱਚ * kstep];
            }
            vec_copy_aligned(ਪੰਕਤੀ, scratch, TILE);
        }
    }

    /* Flush this hart's dirty L1D lines for the slabs it wrote. */
    FENCE;
    ਲਈ (ਪੂਰਨ slab = minion; slab < n_slabs; slab += N_MIN_ਪ੍ਰਤੀ_SHIRE) {
        ਅਚਲ uint64_t slab_ਵਿੱਥ = (uint64_t) slab * SLAB_ਬਾਈਟਾਂ;
        flush_ਮੰਜ਼ਿਲ_l2((ਅਚਲ ਖਾਲੀ *) (l2_filter_ਅਧਾਰ + slab_ਵਿੱਥ), SLAB_ਸਤਰਾਂ, 64);
    }
    WAIT_CACHEOPS;
}

/* Pack one Cin chunk of the input pin (Kw shifted padded copies) into the
   buf_idx side of local L2SCP. Work distributed across the 32 hart-1's in
   the shire by `plane % 32 == minion`. The final flush_to_l2 forces L1D
   write-back so hart-0's tensor_load sees the freshly written bytes. */
ਸਥਿਰ ਖਾਲੀ pack_pin_chunk(ਅਚਲ pin_ctx_t * ctx, ਪੂਰਨ chunk_id, ਪੂਰਨ buf_idx) {
    ਅਚਲ ਪੂਰਨ kt_ਅਧਾਰ  = chunk_id * ctx->chunk_KT;
    ਅਚਲ ਪੂਰਨ Kw       = ctx->Kw;
    ਅਚਲ ਪੂਰਨ chunk_KT = ctx->chunk_KT;
    ਅਚਲ ਪੂਰਨ H = ctx->H, W = ctx->W, Hp = ctx->Hp, Wp_a = ctx->Wp_a;
    ਅਚਲ ਪੂਰਨ pad_h = ctx->pad_h, pad_w = ctx->pad_w, s0 = ctx->s0;
    ਅਚਲ ਪੂਰਨ minion = ctx->minion;

    /* Pin pack: Kw shifted, padded copies of input rows. Bounds [vlo, vhi)
       hoisted outside the row loop so the inner loop is three regions
       (zero-prefix | bulk-copy | zero-suffix) with no per-element predicate. */
    ਦਸ਼ਮਲਵ *   pin0         = (ਦਸ਼ਮਲਵ *) ctx->l2_pad_ਵਿੱਚ_buf[buf_idx];
    ਅਚਲ ਪੂਰਨ chunk_Cin    = chunk_KT * TILE;
    ਅਚਲ ਪੂਰਨ n_pin_planes = Kw * chunk_Cin;
    ਲਈ (ਪੂਰਨ p = minion; p < n_pin_planes; p += N_MIN_ਪ੍ਰਤੀ_SHIRE) {
        ਅਚਲ ਪੂਰਨ s     = p / chunk_Cin;
        ਅਚਲ ਪੂਰਨ icc   = p % chunk_Cin;
        ਅਚਲ ਪੂਰਨ ic    = kt_ਅਧਾਰ * TILE + icc;
        ਦਸ਼ਮਲਵ *   pin_s = pin0 + (size_t) s * ctx->pin_copy_floats;

        ਅਚਲ ਪੂਰਨ ਵਿੱਥ = s - pad_w;
        ਪੂਰਨ       vlo    = 0;
        ਜਦੋਂ (vlo < Wp_a && (s0 * vlo + ਵਿੱਥ) < 0) {
            vlo++;
        }
        ਪੂਰਨ vhi = Wp_a;
        ਜਦੋਂ (vhi > vlo && (s0 * (vhi - 1) + ਵਿੱਥ) >= W) {
            vhi--;
        }
        ਅਚਲ bool aligned = (s0 == 1) && ((vlo & 7) == 0) && (((vlo + ਵਿੱਥ) & 7) == 0);

        ਲਈ (ਪੂਰਨ r = 0; r < Hp; ++r) {
            ਦਸ਼ਮਲਵ *   ਪੰਕਤੀ    = pin_s + ((size_t) icc * Hp + r) * Wp_a;
            ਅਚਲ ਪੂਰਨ real_h = r - pad_h;
            ਜੇ (real_h < 0 || real_h >= H) {
                vec_zero_aligned(ਪੰਕਤੀ, Wp_a);
                ਜਾਰੀ;
            }
            ਅਚਲ ਦਸ਼ਮਲਵ * src_ਪੰਕਤੀ = ctx->ਵਿੱਚ_ਅਧਾਰ + ((size_t) ic * H + real_h) * W;

            ਲਈ (ਪੂਰਨ cc = 0; cc < vlo; ++cc) {
                ਪੰਕਤੀ[cc] = 0.0f;
            }

            ਜੇ (aligned) {
                vec_copy_aligned(ਪੰਕਤੀ + vlo, src_ਪੰਕਤੀ + vlo + ਵਿੱਥ, vhi - vlo);
            } ਨਹੀਂਤਾਂ ਜੇ (s0 == 1) {
                ਅਚਲ ਦਸ਼ਮਲਵ * csrc = src_ਪੰਕਤੀ + vlo + ਵਿੱਥ;
                ਅਚਲ ਪੂਰਨ     n    = vhi - vlo;
                ਲਈ (ਪੂਰਨ cc = 0; cc < n; ++cc) {
                    ਪੰਕਤੀ[vlo + cc] = csrc[cc];
                }
            } ਨਹੀਂਤਾਂ {
                ਲਈ (ਪੂਰਨ cc = vlo; cc < vhi; ++cc) {
                    ਪੰਕਤੀ[cc] = src_ਪੰਕਤੀ[s0 * cc + ਵਿੱਥ];
                }
            }

            ਲਈ (ਪੂਰਨ cc = vhi; cc < Wp_a; ++cc) {
                ਪੰਕਤੀ[cc] = 0.0f;
            }
        }
    }

    /* Flush this buffer's L1D-dirty lines down to L2SCP backing. */
    FENCE;
    flush_range_ਮੰਜ਼ਿਲ_l2((ਅਚਲ ਖਾਲੀ *) ctx->l2_pad_ਵਿੱਚ_buf[buf_idx], ctx->pin_chunk_ਬਾਈਟਾਂ);
    WAIT_CACHEOPS;
}

ਪੂਰਨ ਇੰਦਰਾਜ_ਬਿੰਦੂ(ਬਣਤਰ ggml_et_ਦੁਆਧਾਰੀ_params * params, ਖਾਲੀ * env) {
    (ਖਾਲੀ) env;

    ਅਚਲ ਪੂਰਨ shire   = get_shire_id();
    ਅਚਲ ਪੂਰਨ hart_id = get_hart_id();
    ਅਚਲ ਪੂਰਨ minion  = (hart_id >> 1) & 0x1F;
    ਅਚਲ ਪੂਰਨ hart1   = hart_id & 1;

    ਅਚਲ ਬਣਤਰ ggml_tensor * flt = &params->src0; /* [Kw,Kh,Cin,Cout] */
    ਅਚਲ ਬਣਤਰ ggml_tensor * ਵਿੱਚ  = &params->src1; /* [W, H, Cin,N=1 ] */
    ਬਣਤਰ ggml_tensor *       ਬਾਹਰਲਾ = &params->dst;  /* [W, H, Cout,N=1] */

    ਅਚਲ ਪੂਰਨ Kw   = (ਪੂਰਨ) flt->ne[0];
    ਅਚਲ ਪੂਰਨ Kh   = (ਪੂਰਨ) flt->ne[1];
    ਅਚਲ ਪੂਰਨ Cin  = (ਪੂਰਨ) flt->ne[2];
    ਅਚਲ ਪੂਰਨ Cout = (ਪੂਰਨ) flt->ne[3];

    ਅਚਲ ਪੂਰਨ W  = (ਪੂਰਨ) ਵਿੱਚ->ne[0];
    ਅਚਲ ਪੂਰਨ H  = (ਪੂਰਨ) ਵਿੱਚ->ne[1];
    ਅਚਲ ਪੂਰਨ OW = (ਪੂਰਨ) ਬਾਹਰਲਾ->ne[0];
    ਅਚਲ ਪੂਰਨ OH = (ਪੂਰਨ) ਬਾਹਰਲਾ->ne[1];

    /* op_params layout (set by ggml_conv_2d):
         [0]=s0  [1]=s1  [2]=p0  [3]=p1  [4]=d0  [5]=d1   */
    ਅਚਲ ਪੂਰਨ s0    = ਬਾਹਰਲਾ->op_params[0];
    ਅਚਲ ਪੂਰਨ s1    = ਬਾਹਰਲਾ->op_params[1];
    ਅਚਲ ਪੂਰਨ pad_w = ਬਾਹਰਲਾ->op_params[2];
    ਅਚਲ ਪੂਰਨ pad_h = ਬਾਹਰਲਾ->op_params[3];

    ਜੇ (Cin <= 0 || Cout <= 0) {
        ਵਾਪਸ -1;
    }
    ਜੇ (Cin % TILE != 0 || Cout % TILE != 0) {
        ਵਾਪਸ -1;
    }
    ਜੇ (W <= 0 || H <= 0) {
        ਵਾਪਸ -1;
    }
    ਜੇ (s0 <= 0 || s1 <= 0) {
        ਵਾਪਸ -1;
    }
    ਜੇ (ਵਿੱਚ->ne[2] != Cin || ਵਿੱਚ->ne[3] != 1) {
        ਵਾਪਸ -1;
    }
    ਜੇ (ਬਾਹਰਲਾ->ne[2] != Cout || ਬਾਹਰਲਾ->ne[3] != 1) {
        ਵਾਪਸ -1;
    }
    ਜੇ (!flt->ਅੰਕੜੇ || !ਵਿੱਚ->ਅੰਕੜੇ || !ਬਾਹਰਲਾ->ਅੰਕੜੇ) {
        ਵਾਪਸ -1;
    }

    ਅਚਲ ਪੂਰਨ K_TILES = Cin / TILE;
    ਅਚਲ ਪੂਰਨ M_TILES = Cout / TILE;

    ਅਚਲ ਪੂਰਨ  Hp         = H + 2 * pad_h;
    ਅਚਲ ਪੂਰਨ  Wp_a       = ਗੇੜਾ_ਉੱਪਰ_tile_i32(OW);
    ਅਚਲ ਪੂਰਨ  OW_pad     = Wp_a;
    ਅਚਲ bool need_stage = (OW % TILE != 0);

    /* ===================== Tile assignment & active-shire selection =====
       Computed up front because the per-shire mt set (and thus filter
       region size) depends on n_active_shires. */
    ਅਚਲ ਪੂਰਨ w_tiles         = ceil_div_i32(OW, TILE);
    ਅਚਲ ਪੂਰਨ ਕੁੱਲ_tiles     = OH * w_tiles * M_TILES;
    ਅਚਲ ਪੂਰਨ n_ਸਰਗਰਮ_shires = need_stage ? 1 : min_i32(ਕੁੱਲ_tiles, N_SHIRES);

    /* Inactive shires exit immediately. No global barrier — pack and
       barriers are now per-shire, so unused shires don't need to vote. */
    ਜੇ (shire >= n_ਸਰਗਰਮ_shires) {
        ਵਾਪਸ 0;
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
    ਪੂਰਨ my_mt[MAX_MY_MT];
    ਪੂਰਨ n_my_mt = 0;
    ਲਈ (ਪੂਰਨ t = shire; t < ਕੁੱਲ_tiles; t += n_ਸਰਗਰਮ_shires) {
        ਅਚਲ ਪੂਰਨ mt    = t % M_TILES;
        bool      ਮਿਲਿਆ = false;
        ਲਈ (ਪੂਰਨ j = 0; j < n_my_mt; ++j) {
            ਜੇ (my_mt[j] == mt) {
                ਮਿਲਿਆ = true;
                ਰੋਕੋ;
            }
        }
        ਜੇ (!ਮਿਲਿਆ) {
            ਜੇ (n_my_mt >= MAX_MY_MT) {
                ਵਾਪਸ -1;
            }
            my_mt[n_my_mt++] = mt;
        }
    }
    ਜੇ (n_my_mt == 0) {
        ਵਾਪਸ 0; /* no tiles for this shire */
    }

    ਅਚਲ uint64_t filter_local_ਬਾਈਟਾਂ = (uint64_t) Kh * Kw * n_my_mt * K_TILES * SLAB_ਬਾਈਟਾਂ;
    ਜੇ (filter_local_ਬਾਈਟਾਂ > LOCAL_FILTER_CAP) {
        ਵਾਪਸ -1;
    }

    /* ===================== L2 SCP local layout =========================
       filter (this shire's mt slice) | pin_buf[0] | pin_buf[1]?
                                      | output_stage? | scratch (streaming) */
    ਅਚਲ uint64_t l2_ਅਧਾਰ   = (uint64_t) et_shire_l2scp_local(0);
    ਅਚਲ uint64_t l2_filter = l2_ਅਧਾਰ;

    /* Sizing for pin: budget = LOCAL_BUDGET - filter - output_stage.    */
    ਅਚਲ int64_t output_stage_ਬਾਈਟਾਂ_ਭਰਿਆ = need_stage ? (int64_t) Cout * OH * OW_pad * (int64_t) ਆਕਾਰ(ਦਸ਼ਮਲਵ) : 0;
    ਅਚਲ int64_t budget_for_chunks = (int64_t) LOCAL_BUDGET - (int64_t) filter_local_ਬਾਈਟਾਂ - output_stage_ਬਾਈਟਾਂ_ਭਰਿਆ;
    ਜੇ (budget_for_chunks <= 0) {
        ਵਾਪਸ -1;
    }
    ਅਚਲ int64_t ਪ੍ਰਤੀ_KT_pin_ਬਾਈਟਾਂ = (int64_t) Kw * TILE * Hp * Wp_a * (int64_t) ਆਕਾਰ(ਦਸ਼ਮਲਵ);

    ਪੂਰਨ chunk_KT;
    ਪੂਰਨ n_buffers;
    ਜੇ ((int64_t) K_TILES * ਪ੍ਰਤੀ_KT_pin_ਬਾਈਟਾਂ <= budget_for_chunks) {
        chunk_KT  = K_TILES;
        n_buffers = 1;
    } ਨਹੀਂਤਾਂ {
        chunk_KT = K_TILES;
        ਜਦੋਂ (chunk_KT > 1 && 2 * (int64_t) chunk_KT * ਪ੍ਰਤੀ_KT_pin_ਬਾਈਟਾਂ > budget_for_chunks) {
            chunk_KT--;
        }
        ਜਦੋਂ (chunk_KT > 1 && K_TILES % chunk_KT != 0) {
            chunk_KT--;
        }
        n_buffers = (chunk_KT < K_TILES) ? 2 : 1;
        ਜੇ (chunk_KT < 1) {
            ਵਾਪਸ -1;
        }
    }
    ਅਚਲ ਪੂਰਨ n_chunks = K_TILES / chunk_KT;

    /* Streaming keeps partial sums in MAX_TILES_PER_HART scratch slots per
       hart. The one-chunk path does not need scratch and can stream a longer
       tile list serially, but multi-chunk shapes must fit this fixed slot
       count until scratch scheduling is made more general. */
    ਅਚਲ ਪੂਰਨ shire_tile_ਸਮਰੱਥਾ = shire + MAX_TILES_ਪ੍ਰਤੀ_HART * n_ਸਰਗਰਮ_shires * N_MIN_ਪ੍ਰਤੀ_SHIRE;
    ਜੇ (n_chunks > 1 && shire_tile_ਸਮਰੱਥਾ < ਕੁੱਲ_tiles) {
        ਵਾਪਸ -1;
    }

    ਅਚਲ uint64_t pin_copy_floats = (uint64_t) chunk_KT * TILE * Hp * Wp_a;
    ਅਚਲ uint64_t pin_copy_ਬਾਈਟਾਂ  = pin_copy_floats * ਆਕਾਰ(ਦਸ਼ਮਲਵ);
    ਅਚਲ uint64_t pin_chunk_ਬਾਈਟਾਂ = (uint64_t) Kw * pin_copy_ਬਾਈਟਾਂ;

    ਅਚਲ uint64_t l2_pin_ਅਧਾਰ              = l2_filter + filter_local_ਬਾਈਟਾਂ;
    ਅਚਲ uint64_t l2_pin_buf[MAX_DBL_BUFS] = {
        l2_pin_ਅਧਾਰ,
        l2_pin_ਅਧਾਰ + pin_chunk_ਬਾਈਟਾਂ,
    };

    ਅਚਲ uint64_t l2_output_stage = need_stage ? l2_pin_ਅਧਾਰ + (uint64_t) n_buffers * pin_chunk_ਬਾਈਟਾਂ : 0;

    ਅਚਲ uint64_t scratch_ਪ੍ਰਤੀ_hart = (uint64_t) MAX_TILES_ਪ੍ਰਤੀ_HART * (uint64_t) TILE * TILE * ਆਕਾਰ(ਦਸ਼ਮਲਵ);
    ਅਚਲ uint64_t l2_scratch_ਅਧਾਰ  = need_stage ? l2_output_stage + (uint64_t) output_stage_ਬਾਈਟਾਂ_ਭਰਿਆ :
                                                   l2_pin_ਅਧਾਰ + (uint64_t) n_buffers * pin_chunk_ਬਾਈਟਾਂ;

    /* ===================== PHASE 1: Filter pack (per-shire mt slice) ====
       Hart-1's pack only this shire's mt slabs into local L2 SCP. The
       SHIRE barrier below ensures the filter is in L2 SCP backing before
       hart-0's first tensor_load. */
    ਜੇ (hart1) {
        pack_filter_local_mt((ਅਚਲ ਦਸ਼ਮਲਵ *) flt->ਅੰਕੜੇ, Kh, Kw, Cin, K_TILES, my_mt, n_my_mt, minion, l2_filter);
    }

    /* ===================== Hart 1: pin packer (per chunk) ==============
       Double-buffered prefetch: pack chunk 0 synchronously, then per chunk c
       signal "buf c ready", pack chunk c+1 into the alternate buffer
       (overlaps hart-0's compute on c), signal "buf c done". */
    ਜੇ (hart1) {
        ਅਚਲ pin_ctx_t ctx = {
            .ਵਿੱਚ_ਅਧਾਰ         = (ਅਚਲ ਦਸ਼ਮਲਵ *) ਵਿੱਚ->ਅੰਕੜੇ,
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
            .l2_pad_ਵਿੱਚ_buf   = { l2_pin_buf[0], l2_pin_buf[1] },
            .pin_chunk_ਬਾਈਟਾਂ = pin_chunk_ਬਾਈਟਾਂ,
        };

        pack_pin_chunk(&ctx, 0, 0); /* prologue */

        ਲਈ (ਪੂਰਨ c = 0; c < n_chunks; ++c) {
            et_barrier(ET_BARRIER_SHIRE); /* signal "buf c ready" */
            ਜੇ (n_buffers > 1 && c + 1 < n_chunks) {
                pack_pin_chunk(&ctx, c + 1, (c + 1) & 1);
            }
            et_barrier(ET_BARRIER_SHIRE); /* wait "buf c done" */
        }

        ਜੇ (need_stage) {
            et_barrier(ET_BARRIER_SHIRE);
        }
        ਵਾਪਸ 0;
    }

    /* ===================== Hart 0: matrix engine ======================
       Two execution modes:
         - n_chunks == 1: full Cin in one shot. Each hart processes a list
           of tiles serially; TenC resets between tiles via first_pass=true.
         - n_chunks  > 1: streaming. Each hart owns up to MAX_TILES_PER_HART
           tiles. For each chunk c, restore TenC from scratch[k] (skip on
           c==0), accumulate this chunk's FMAs, then either save TenC back
           to scratch[k] (c < last) or tensor_store directly (c == last). */
    ਤਿਆਰੀ_cache_scp();
    CLEAR_TENSOR_ERROR;

    ਅੱਖਰ * ਅਚਲ   ਬਾਹਰਲਾ_ਅਧਾਰ        = need_stage ? (ਅੱਖਰ *) l2_output_stage : (ਅੱਖਰ *) ਬਾਹਰਲਾ->ਅੰਕੜੇ;
    ਅਚਲ ਪੂਰਨ      compute_OW      = need_stage ? OW_pad : OW;
    ਅਚਲ uint64_t ਬਾਹਰਲਾ_chan_stride = (uint64_t) OH * (uint64_t) compute_OW * ਆਕਾਰ(ਦਸ਼ਮਲਵ);
    ਅਚਲ uint64_t ਬਾਹਰਲਾ_ਪੰਕਤੀ_stride  = (uint64_t) compute_OW * ਆਕਾਰ(ਦਸ਼ਮਲਵ);

    ਅਚਲ uint64_t a_ਪੰਕਤੀ_stride = (uint64_t) TILE * ਆਕਾਰ(ਦਸ਼ਮਲਵ); /* 64 */
    ਅਚਲ uint64_t b_ਪੰਕਤੀ_stride = (uint64_t) Hp * (uint64_t) Wp_a * ਆਕਾਰ(ਦਸ਼ਮਲਵ);

    /* Tile assignment: shire-strided so small workloads spread across
       shires before stacking minions in one shire. */
    ਅਚਲ ਪੂਰਨ t_ਸ਼ੁਰੂ  = shire + minion * n_ਸਰਗਰਮ_shires;
    ਅਚਲ ਪੂਰਨ t_stride = n_ਸਰਗਰਮ_shires * N_MIN_ਪ੍ਰਤੀ_SHIRE;

    ਜੇ (n_chunks == 1) {
        et_barrier(ET_BARRIER_SHIRE); /* wait for the (only) pin chunk */

        ਅਚਲ uint64_t l2_pad_ਵਿੱਚ = l2_pin_buf[0];
        ਲਈ (ਪੂਰਨ t = t_ਸ਼ੁਰੂ; t < ਕੁੱਲ_tiles; t += t_stride) {
            ਅਚਲ conv_tile_t tile = decode_tile(t, M_TILES, w_tiles, my_mt, n_my_mt);
            compute_tile_chunk(l2_filter, l2_pad_ਵਿੱਚ, pin_copy_ਬਾਈਟਾਂ, Kh, Kw, K_TILES, chunk_KT, 0, n_my_mt, Hp, Wp_a,
                               s1, a_ਪੰਕਤੀ_stride, b_ਪੰਕਤੀ_stride, &tile, /*first_fma_clears_tenc=*/true);

            ਅੱਖਰ * dst_addr = output_tile_addr(ਬਾਹਰਲਾ_ਅਧਾਰ, &tile, ਬਾਹਰਲਾ_chan_stride, ਬਾਹਰਲਾ_ਪੰਕਤੀ_stride);
            tensor_store(0, 0, 3, (uint64_t) (TILE - 1), (uint64_t) dst_addr, 0, ਬਾਹਰਲਾ_chan_stride);
            tensor_wait(TENSOR_STORE_WAIT);
        }

        et_barrier(ET_BARRIER_SHIRE); /* matches hart-1's second barrier */

    } ਨਹੀਂਤਾਂ {
        /* Streaming path: each hart owns up to MAX_TILES_PER_HART tiles. */
        ਪੂਰਨ my_tiles[MAX_TILES_ਪ੍ਰਤੀ_HART];
        ਪੂਰਨ n_my_tiles = 0;
        ਲਈ (ਪੂਰਨ ਥਾਂ = 0; ਥਾਂ < MAX_TILES_ਪ੍ਰਤੀ_HART; ++ਥਾਂ) {
            ਅਚਲ ਪੂਰਨ t = t_ਸ਼ੁਰੂ + ਥਾਂ * t_stride;
            ਜੇ (t < ਕੁੱਲ_tiles) {
                my_tiles[n_my_tiles++] = t;
            }
        }

        conv_tile_t tiles[MAX_TILES_ਪ੍ਰਤੀ_HART];
        ਲਈ (ਪੂਰਨ k = 0; k < n_my_tiles; ++k) {
            tiles[k] = decode_tile(my_tiles[k], M_TILES, w_tiles, my_mt, n_my_mt);
        }

        ਅਚਲ uint64_t my_scratch_ਅਧਾਰ = l2_scratch_ਅਧਾਰ + (uint64_t) minion * scratch_ਪ੍ਰਤੀ_hart;

        ਲਈ (ਪੂਰਨ c = 0; c < n_chunks; ++c) {
            et_barrier(ET_BARRIER_SHIRE); /* pin chunk c packed */

            ਅਚਲ ਪੂਰਨ      buf       = c & 1;
            ਅਚਲ uint64_t l2_pad_ਵਿੱਚ = l2_pin_buf[buf];
            ਅਚਲ ਪੂਰਨ      kt_ਅਧਾਰ   = c * chunk_KT;

            ਲਈ (ਪੂਰਨ k = 0; k < n_my_tiles; ++k) {
                ਅਚਲ conv_tile_t * tile = &tiles[k];
                ਅਚਲ uint64_t      scr  = my_scratch_ਅਧਾਰ + (uint64_t) k * (TILE * TILE * ਆਕਾਰ(ਦਸ਼ਮਲਵ));

                ਅਚਲ bool ਪਹਿਲਾ_pass_chunk = (c == 0);
                ਜੇ (!ਪਹਿਲਾ_pass_chunk) {
                    tenc_restore_from_scratch(scr);
                }

                compute_tile_chunk(l2_filter, l2_pad_ਵਿੱਚ, pin_copy_ਬਾਈਟਾਂ, Kh, Kw, K_TILES, chunk_KT, kt_ਅਧਾਰ, n_my_mt,
                                   Hp, Wp_a, s1, a_ਪੰਕਤੀ_stride, b_ਪੰਕਤੀ_stride, tile, ਪਹਿਲਾ_pass_chunk);

                ਜੇ (c == n_chunks - 1) {
                    ਅੱਖਰ * dst_addr = output_tile_addr(ਬਾਹਰਲਾ_ਅਧਾਰ, tile, ਬਾਹਰਲਾ_chan_stride, ਬਾਹਰਲਾ_ਪੰਕਤੀ_stride);
                    tensor_store(0, 0, 3, (uint64_t) (TILE - 1), (uint64_t) dst_addr, 0, ਬਾਹਰਲਾ_chan_stride);
                } ਨਹੀਂਤਾਂ {
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
    ਜੇ (need_stage) {
        et_barrier(ET_BARRIER_SHIRE);

        ਜੇ (minion == 0) {
            ਅਚਲ ਦਸ਼ਮਲਵ * stage = (ਅਚਲ ਦਸ਼ਮਲਵ *) l2_output_stage;
            ਦਸ਼ਮਲਵ *       dram  = (ਦਸ਼ਮਲਵ *) ਬਾਹਰਲਾ->ਅੰਕੜੇ;
            ਲਈ (ਪੂਰਨ oc = 0; oc < Cout; ++oc) {
                ਲਈ (ਪੂਰਨ oh2 = 0; oh2 < OH; ++oh2) {
                    ਅਚਲ ਦਸ਼ਮਲਵ * src = stage + ((size_t) oc * OH + oh2) * OW_pad;
                    ਦਸ਼ਮਲਵ *       dst = dram + ((size_t) oc * OH + oh2) * OW;
                    ਲਈ (ਪੂਰਨ ow2 = 0; ow2 < OW; ++ow2) {
                        dst[ow2] = src[ow2];
                    }
                }
            }
            FENCE;
            ਅਚਲ uint64_t ਕੁੱਲ_ਬਾਈਟਾਂ = (uint64_t) Cout * OH * OW * ਆਕਾਰ(ਦਸ਼ਮਲਵ);
            evict_range_past_l2((ਅਚਲ ਖਾਲੀ *) dram, ਕੁੱਲ_ਬਾਈਟਾਂ);
            WAIT_CACHEOPS;
        }
    }

    ਵਾਪਸ 0;
}
