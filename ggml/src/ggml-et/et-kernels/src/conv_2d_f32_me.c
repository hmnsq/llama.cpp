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

#అన్తర్భావ "ggml_tensor.h"
#అన్తర్భావ "platform.h"
#అన్తర్భావ "tensor.h"

#అన్తర్భావ <etsoc/common/utils.h>
#అన్తర్భావ <stdbool.h>
#అన్తర్భావ <stdint.h>

#నిర్వచించు TILE               16 /* matrix engine native tile in M, K, N */
/* L1 SCP layout: A double-buffered, B single-buffered. Per the SDK doc
   `dst_start` is a 6-bit field (max 63) but empirical testing shows the
   physical L1 SCP per minion is 48 lines — writes to lines >= 48 corrupt.
   So we get 3 × 16-line buffers max: A_0, A_1, B. Pick A as the
   double-buffered operand (filter-slab loads, the longer of the two). */
#నిర్వచించు LSCP_A_0           0  /* A buffer 0 at L1 SCP lines 0..15  */
#నిర్వచించు LSCP_A_1           16 /* A buffer 1 at L1 SCP lines 16..31 */
#నిర్వచించు LSCP_B             32 /* B (single buffer) at lines 32..47 */
#నిర్వచించు N_MIN_చొప్పున_SHIRE    32 /* ET-SoC-1 geometry: 32 minions/shire  */
#నిర్వచించు N_SHIRES           32 /* default active shire count           */
#నిర్వచించు MAX_TILES_చొప్పున_HART 2  /* per-hart TenC slots (save/restore)   */
#నిర్వచించు MAX_DBL_BUFS       2  /* chunk pack buffers (double-buffered) */

/* Per-shire L2 SCP local budget. Per-shire SCP is 2 MB; we cap at
   1984 KB to leave 64 KB headroom for per-hart TenC scratch (32 minions ×
   2 slots × 1 KB), which lives at the tail of the SCP outside the pin
   sizing budget. Bigger budget here means bigger feasible chunk_KT,
   which means fewer chunks (each chunk costs 2 SHIRE barriers + ~30
   TenC save/restore events per hart). */
#నిర్వచించు LOCAL_BUDGET (1984 * 1024)

/* Cap on the per-shire filter region in local L2 SCP. The shire packs the
   mt values it can consume under the current tile assignment, rather than
   the whole Cout dimension. Reads in the inner loop are then fully
   shire-local — no NoC fanout. */
#నిర్వచించు LOCAL_FILTER_CAP (1024 * 1024)                      /* 1 MB / shire ceiling */

#నిర్వచించు SLAB_బైట్లు ((uint64_t) TILE * TILE * పరిమాణం(దశాంశం)) /* 1024 */
#నిర్వచించు SLAB_పంక్తులు ((SLAB_బైట్లు + 63) / 64)                 /* 16   */

/* Upper bound on the number of distinct mt values a single shire may pack.
   This keeps the mt list stack-resident. Shapes that need more should fall
   back until the filter-slice bookkeeping is made dynamic. */
#నిర్వచించు MAX_MY_MT (N_MIN_చొప్పున_SHIRE * MAX_TILES_చొప్పున_HART)

రకనిర్వచనం నిర్మాణం {
    పూర్ణాంకం mt;
    పూర్ణాంకం mt_idx;
    పూర్ణాంకం oh;
    పూర్ణాంకం ow_పునాది;
} conv_tile_t;

స్థిర అంతఃవరుస పూర్ణాంకం ceil_div_i32(పూర్ణాంకం x, పూర్ణాంకం y) {
    తిరిగి (x + y - 1) / y;
}

స్థిర అంతఃవరుస పూర్ణాంకం విడత_పైకి_tile_i32(పూర్ణాంకం x) {
    తిరిగి (x + TILE - 1) & ~(TILE - 1);
}

స్థిర అంతఃవరుస పూర్ణాంకం min_i32(పూర్ణాంకం a, పూర్ణాంకం b) {
    తిరిగి a < b ? a : b;
}

స్థిర అంతఃవరుస uint64_t min_u64(uint64_t a, uint64_t b) {
    తిరిగి a < b ? a : b;
}

/* ===== Vector helpers for hart-1 pack ============================
   Both assume dst (and src for copy) are 32-byte aligned; n is in floats.
   The 8-element tail is handled scalar. f30/f31 are scratch — clobbered
   per-call via the asm clobber list. */
స్థిర అంతఃవరుస శూన్య vec_zero_aligned(దశాంశం * dst, పూర్ణాంకం n) {
    పూర్ణాంకం       i  = 0;
    స్థిరరాశి పూర్ణాంకం n8 = n & ~7;
    కొరకు (; i < n8; i += 8) {
        __asm__ అస్థిర(
            "fsub.ps f31, f31, f31\n"
            "fsw.ps  f31, %[d]\n"
            : [d] "=m"(*(దశాంశం (*)[8]) & dst[i])
            :
            : "f31");
    }
    కొరకు (; i < n; ++i) {
        dst[i] = 0.0f;
    }
}

స్థిర అంతఃవరుస శూన్య vec_copy_aligned(దశాంశం * dst, స్థిరరాశి దశాంశం * src, పూర్ణాంకం n) {
    పూర్ణాంకం       i  = 0;
    స్థిరరాశి పూర్ణాంకం n8 = n & ~7;
    కొరకు (; i < n8; i += 8) {
        __asm__ అస్థిర(
            "flw.ps f30, %[s]\n"
            "fsw.ps f30, %[d]\n"
            : [d] "=m"(*(దశాంశం (*)[8]) & dst[i])
            : [s] "m"(*(స్థిరరాశి దశాంశం (*)[8]) & src[i])
            : "f30");
    }
    కొరకు (; i < n; ++i) {
        dst[i] = src[i];
    }
}

/* ===== TenC save/restore =========================================
   The TenC accumulator IS the f0..f31 vector register file: row N occupies
   f(2N) and f(2N+1) (two 8-fp32 vector regs per row). We save by
   tensor_store-ing TILE rows × 64 bytes, and restore via 32 flw.ps after
   forcing L1D to refetch from the L2SCP backing (tensor_store bypasses L1D
   so the backing is always current). See feedback_tenc_save_restore.md. */
స్థిర అంతఃవరుస శూన్య tenc_restore_from_scratch(uint64_t scr) {
    FENCE;
    evict_గమ్యం_l2((స్థిరరాశి శూన్య *) scr, TILE, 64);
    WAIT_CACHEOPS;
    __asm__ అస్థిర(
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
రకనిర్వచనం నిర్మాణం {
    స్థిరరాశి దశాంశం * లోపల_పునాది;  /* DRAM input base [Cin][H][W]         */
    పూర్ణాంకం           Kw;
    పూర్ణాంకం           chunk_KT; /* number of K_TILES (=16-wide) per chunk */
    పూర్ణాంకం           H, W, Hp, Wp_a;
    పూర్ణాంకం           pad_h, pad_w, s0;
    పూర్ణాంకం           minion;          /* this hart's minion id (0..31) */
    uint64_t      pin_copy_floats; /* per-_s pin plane size in floats */
    uint64_t      l2_pad_లోపల_buf[MAX_DBL_BUFS];
    uint64_t      pin_chunk_బైట్లు; /* one chunk pin buffer's total size */
} pin_ctx_t;

స్థిర అంతఃవరుస పూర్ణాంకం కనుగొను_mt_idx(స్థిరరాశి పూర్ణాంకం * my_mt, పూర్ణాంకం n_my_mt, పూర్ణాంకం mt) {
    కొరకు (పూర్ణాంకం j = 0; j < n_my_mt; ++j) {
        ఒకవేళ (my_mt[j] == mt) {
            తిరిగి j;
        }
    }
    తిరిగి 0;
}

స్థిర అంతఃవరుస conv_tile_t decode_tile(పూర్ణాంకం t, పూర్ణాంకం M_TILES, పూర్ణాంకం w_tiles, స్థిరరాశి పూర్ణాంకం * my_mt, పూర్ణాంకం n_my_mt) {
    conv_tile_t tile;
    tile.mt = t % M_TILES;
    t /= M_TILES;
    స్థిరరాశి పూర్ణాంకం wt = t % w_tiles;
    t /= w_tiles;
    tile.oh      = t;
    tile.ow_పునాది = wt * TILE;
    tile.mt_idx  = కనుగొను_mt_idx(my_mt, n_my_mt, tile.mt);
    తిరిగి tile;
}

స్థిర అంతఃవరుస uint64_t
filter_slab_addr(uint64_t l2_filter, పూర్ణాంకం Kw, పూర్ణాంకం K_TILES, పూర్ణాంకం n_my_mt, పూర్ణాంకం mt_idx, పూర్ణాంకం kh, పూర్ణాంకం kw, పూర్ణాంకం kt_global) {
    తిరిగి l2_filter + (uint64_t) ((((kh * Kw + kw) * n_my_mt + mt_idx) * K_TILES + kt_global)) * SLAB_బైట్లు;
}

స్థిర అంతఃవరుస uint64_t pin_tile_addr(uint64_t l2_pad_లోపల,
                                     uint64_t pin_copy_బైట్లు,
                                     పూర్ణాంకం      ktc,
                                     పూర్ణాంకం      kw,
                                     పూర్ణాంకం      Hp,
                                     పూర్ణాంకం      Wp_a,
                                     పూర్ణాంకం      oh,
                                     పూర్ణాంకం      ow_పునాది,
                                     పూర్ణాంకం      s1,
                                     పూర్ణాంకం      kh) {
    స్థిరరాశి పూర్ణాంకం ir_pad = oh * s1 + kh;
    తిరిగి l2_pad_లోపల + (uint64_t) kw * pin_copy_బైట్లు +
           (((uint64_t) (ktc * TILE) * Hp + ir_pad) * Wp_a + ow_పునాది) * పరిమాణం(దశాంశం);
}

స్థిర అంతఃవరుస అక్షరం * output_tile_addr(అక్షరం *              బహిర్గతం_పునాది,
                                      స్థిరరాశి conv_tile_t * tile,
                                      uint64_t            బహిర్గతం_chan_stride,
                                      uint64_t            బహిర్గతం_అడ్డువరుస_stride) {
    తిరిగి బహిర్గతం_పునాది + (size_t) (tile->mt * TILE) * బహిర్గతం_chan_stride + (size_t) tile->oh * బహిర్గతం_అడ్డువరుస_stride +
           (size_t) tile->ow_పునాది * పరిమాణం(దశాంశం);
}

స్థిర అంతఃవరుస శూన్య flush_range_గమ్యం_l2(స్థిరరాశి శూన్య * addr, uint64_t n_బైట్లు) {
    స్థిరరాశి uint64_t మొత్తం_పంక్తులు = (n_బైట్లు + 63) / 64;
    స్థిరరాశి అక్షరం *   fl_addr     = (స్థిరరాశి అక్షరం *) addr;
    కొరకు (uint64_t పూర్తి = 0; పూర్తి < మొత్తం_పంక్తులు;) {
        స్థిరరాశి uint64_t batch = min_u64(మొత్తం_పంక్తులు - పూర్తి, 16);
        flush_గమ్యం_l2((స్థిరరాశి శూన్య *) (fl_addr + పూర్తి * 64), batch, 64);
        పూర్తి += batch;
    }
}

స్థిర అంతఃవరుస శూన్య evict_range_past_l2(స్థిరరాశి శూన్య * addr, uint64_t n_బైట్లు) {
    స్థిరరాశి uint64_t మొత్తం_పంక్తులు = (n_బైట్లు + 63) / 64;
    స్థిరరాశి అక్షరం *   fl_addr     = (స్థిరరాశి అక్షరం *) addr;
    కొరకు (uint64_t పూర్తి = 0; పూర్తి < మొత్తం_పంక్తులు;) {
        స్థిరరాశి uint64_t batch = min_u64(మొత్తం_పంక్తులు - పూర్తి, 16);
        evict_past_l2((స్థిరరాశి శూన్య *) (fl_addr + పూర్తి * 64), batch, 64);
        పూర్తి += batch;
    }
}

/* One matrix-engine tile for one Cin chunk. This is the main optimization
   surface: A is double-buffered, B is single-buffered due to L1 SCP space. */
స్థిర అంతఃవరుస శూన్య compute_tile_chunk(uint64_t            l2_filter,
                                      uint64_t            l2_pad_లోపల,
                                      uint64_t            pin_copy_బైట్లు,
                                      పూర్ణాంకం                 Kh,
                                      పూర్ణాంకం                 Kw,
                                      పూర్ణాంకం                 K_TILES,
                                      పూర్ణాంకం                 chunk_KT,
                                      పూర్ణాంకం                 kt_పునాది,
                                      పూర్ణాంకం                 n_my_mt,
                                      పూర్ణాంకం                 Hp,
                                      పూర్ణాంకం                 Wp_a,
                                      పూర్ణాంకం                 s1,
                                      uint64_t            a_అడ్డువరుస_stride,
                                      uint64_t            b_అడ్డువరుస_stride,
                                      స్థిరరాశి conv_tile_t * tile,
                                      bool                మొదటి_fma_clears_tenc) {
    స్థిరరాశి పూర్ణాంకం      n_iters   = Kh * Kw * chunk_KT;
    స్థిరరాశి uint64_t A_BUFS[2] = { LSCP_A_0, LSCP_A_1 };

    స్థిరరాశి uint64_t a_addr0 = filter_slab_addr(l2_filter, Kw, K_TILES, n_my_mt, tile->mt_idx, 0, 0, kt_పునాది);
    tensor_ఎక్కించు(false, false, A_BUFS[0], 0, 0, a_addr0, 0, (uint64_t) (TILE - 1), a_అడ్డువరుస_stride, 0);

    కొరకు (పూర్ణాంకం iter = 0; iter < n_iters; ++iter) {
        స్థిరరాశి పూర్ణాంకం ktc = iter % chunk_KT;
        స్థిరరాశి పూర్ణాంకం rem = iter / chunk_KT;
        స్థిరరాశి పూర్ణాంకం kw  = rem % Kw;
        స్థిరరాశి పూర్ణాంకం kh  = rem / Kw;

        స్థిరరాశి uint64_t b_addr =
            pin_tile_addr(l2_pad_లోపల, pin_copy_బైట్లు, ktc, kw, Hp, Wp_a, tile->oh, tile->ow_పునాది, s1, kh);
        tensor_ఎక్కించు(false, false, LSCP_B, 0, 0, b_addr, 0, (uint64_t) (TILE - 1), b_అడ్డువరుస_stride, 1);

        tensor_wait(TENSOR_ఎక్కించు_WAIT_0);
        tensor_wait(TENSOR_ఎక్కించు_WAIT_1);

        ఒకవేళ (iter + 1 < n_iters) {
            స్థిరరాశి పూర్ణాంకం      ktc_n = (iter + 1) % chunk_KT;
            స్థిరరాశి పూర్ణాంకం      rem_n = (iter + 1) / chunk_KT;
            స్థిరరాశి పూర్ణాంకం      kw_n  = rem_n % Kw;
            స్థిరరాశి పూర్ణాంకం      kh_n  = rem_n / Kw;
            స్థిరరాశి uint64_t a_addr_n =
                filter_slab_addr(l2_filter, Kw, K_TILES, n_my_mt, tile->mt_idx, kh_n, kw_n, kt_పునాది + ktc_n);
            tensor_ఎక్కించు(false, false, A_BUFS[(iter + 1) & 1], 0, 0, a_addr_n, 0, (uint64_t) (TILE - 1), a_అడ్డువరుస_stride,
                        0);
        }

        tensor_fma(false, 3, (uint64_t) (TILE - 1), (uint64_t) (TILE - 1), 0, false, false, false, false, LSCP_B,
                   A_BUFS[iter & 1], 0, మొదటి_fma_clears_tenc && (iter == 0));
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
స్థిర శూన్య pack_filter_local_mt(స్థిరరాశి దశాంశం * flt_పునాది,
                                 పూర్ణాంకం           Kh,
                                 పూర్ణాంకం           Kw,
                                 పూర్ణాంకం           Cin,
                                 పూర్ణాంకం           K_TILES,
                                 స్థిరరాశి పూర్ణాంకం *   my_mt,
                                 పూర్ణాంకం           n_my_mt,
                                 పూర్ణాంకం           minion,
                                 uint64_t      l2_filter_పునాది) {
    స్థిరరాశి పూర్ణాంకం    n_slabs = Kh * Kw * n_my_mt * K_TILES;
    స్థిరరాశి size_t kstep   = (size_t) Kh * Kw; /* Cin stride in floats */

    కొరకు (పూర్ణాంకం slab = minion; slab < n_slabs; slab += N_MIN_చొప్పున_SHIRE) {
        పూర్ణాంకం       t  = slab;
        స్థిరరాశి పూర్ణాంకం kt = t % K_TILES;
        t /= K_TILES;
        స్థిరరాశి పూర్ణాంకం mt_idx = t % n_my_mt;
        t /= n_my_mt;
        స్థిరరాశి పూర్ణాంకం kw = t % Kw;
        t /= Kw;
        స్థిరరాశి పూర్ణాంకం kh = t;
        స్థిరరాశి పూర్ణాంకం mt = my_mt[mt_idx];

        స్థిరరాశి uint64_t slab_స్థానభ్రంశం = (uint64_t) slab * SLAB_బైట్లు;
        దశాంశం *        cell        = (దశాంశం *) (l2_filter_పునాది + slab_స్థానభ్రంశం);

        కొరకు (పూర్ణాంకం oc_లోపల = 0; oc_లోపల < TILE; ++oc_లోపల) {
            స్థిరరాశి పూర్ణాంకం     oc  = mt * TILE + oc_లోపల;
            స్థిరరాశి దశాంశం * src = flt_పునాది + (((size_t) oc * Cin + (size_t) kt * TILE) * Kh + kh) * Kw + kw;
            దశాంశం *       అడ్డువరుస = cell + (size_t) oc_లోపల * TILE;
            దశాంశం         scratch[TILE] __attribute__((aligned(32)));
            కొరకు (పూర్ణాంకం ic_లోపల = 0; ic_లోపల < TILE; ++ic_లోపల) {
                scratch[ic_లోపల] = src[(size_t) ic_లోపల * kstep];
            }
            vec_copy_aligned(అడ్డువరుస, scratch, TILE);
        }
    }

    /* Flush this hart's dirty L1D lines for the slabs it wrote. */
    FENCE;
    కొరకు (పూర్ణాంకం slab = minion; slab < n_slabs; slab += N_MIN_చొప్పున_SHIRE) {
        స్థిరరాశి uint64_t slab_స్థానభ్రంశం = (uint64_t) slab * SLAB_బైట్లు;
        flush_గమ్యం_l2((స్థిరరాశి శూన్య *) (l2_filter_పునాది + slab_స్థానభ్రంశం), SLAB_పంక్తులు, 64);
    }
    WAIT_CACHEOPS;
}

/* Pack one Cin chunk of the input pin (Kw shifted padded copies) into the
   buf_idx side of local L2SCP. Work distributed across the 32 hart-1's in
   the shire by `plane % 32 == minion`. The final flush_to_l2 forces L1D
   write-back so hart-0's tensor_load sees the freshly written bytes. */
స్థిర శూన్య pack_pin_chunk(స్థిరరాశి pin_ctx_t * ctx, పూర్ణాంకం chunk_id, పూర్ణాంకం buf_idx) {
    స్థిరరాశి పూర్ణాంకం kt_పునాది  = chunk_id * ctx->chunk_KT;
    స్థిరరాశి పూర్ణాంకం Kw       = ctx->Kw;
    స్థిరరాశి పూర్ణాంకం chunk_KT = ctx->chunk_KT;
    స్థిరరాశి పూర్ణాంకం H = ctx->H, W = ctx->W, Hp = ctx->Hp, Wp_a = ctx->Wp_a;
    స్థిరరాశి పూర్ణాంకం pad_h = ctx->pad_h, pad_w = ctx->pad_w, s0 = ctx->s0;
    స్థిరరాశి పూర్ణాంకం minion = ctx->minion;

    /* Pin pack: Kw shifted, padded copies of input rows. Bounds [vlo, vhi)
       hoisted outside the row loop so the inner loop is three regions
       (zero-prefix | bulk-copy | zero-suffix) with no per-element predicate. */
    దశాంశం *   pin0         = (దశాంశం *) ctx->l2_pad_లోపల_buf[buf_idx];
    స్థిరరాశి పూర్ణాంకం chunk_Cin    = chunk_KT * TILE;
    స్థిరరాశి పూర్ణాంకం n_pin_planes = Kw * chunk_Cin;
    కొరకు (పూర్ణాంకం p = minion; p < n_pin_planes; p += N_MIN_చొప్పున_SHIRE) {
        స్థిరరాశి పూర్ణాంకం s     = p / chunk_Cin;
        స్థిరరాశి పూర్ణాంకం icc   = p % chunk_Cin;
        స్థిరరాశి పూర్ణాంకం ic    = kt_పునాది * TILE + icc;
        దశాంశం *   pin_s = pin0 + (size_t) s * ctx->pin_copy_floats;

        స్థిరరాశి పూర్ణాంకం స్థానభ్రంశం = s - pad_w;
        పూర్ణాంకం       vlo    = 0;
        అయితే (vlo < Wp_a && (s0 * vlo + స్థానభ్రంశం) < 0) {
            vlo++;
        }
        పూర్ణాంకం vhi = Wp_a;
        అయితే (vhi > vlo && (s0 * (vhi - 1) + స్థానభ్రంశం) >= W) {
            vhi--;
        }
        స్థిరరాశి bool aligned = (s0 == 1) && ((vlo & 7) == 0) && (((vlo + స్థానభ్రంశం) & 7) == 0);

        కొరకు (పూర్ణాంకం r = 0; r < Hp; ++r) {
            దశాంశం *   అడ్డువరుస    = pin_s + ((size_t) icc * Hp + r) * Wp_a;
            స్థిరరాశి పూర్ణాంకం real_h = r - pad_h;
            ఒకవేళ (real_h < 0 || real_h >= H) {
                vec_zero_aligned(అడ్డువరుస, Wp_a);
                కొనసాగు;
            }
            స్థిరరాశి దశాంశం * src_అడ్డువరుస = ctx->లోపల_పునాది + ((size_t) ic * H + real_h) * W;

            కొరకు (పూర్ణాంకం cc = 0; cc < vlo; ++cc) {
                అడ్డువరుస[cc] = 0.0f;
            }

            ఒకవేళ (aligned) {
                vec_copy_aligned(అడ్డువరుస + vlo, src_అడ్డువరుస + vlo + స్థానభ్రంశం, vhi - vlo);
            } లేకపోతే ఒకవేళ (s0 == 1) {
                స్థిరరాశి దశాంశం * csrc = src_అడ్డువరుస + vlo + స్థానభ్రంశం;
                స్థిరరాశి పూర్ణాంకం     n    = vhi - vlo;
                కొరకు (పూర్ణాంకం cc = 0; cc < n; ++cc) {
                    అడ్డువరుస[vlo + cc] = csrc[cc];
                }
            } లేకపోతే {
                కొరకు (పూర్ణాంకం cc = vlo; cc < vhi; ++cc) {
                    అడ్డువరుస[cc] = src_అడ్డువరుస[s0 * cc + స్థానభ్రంశం];
                }
            }

            కొరకు (పూర్ణాంకం cc = vhi; cc < Wp_a; ++cc) {
                అడ్డువరుస[cc] = 0.0f;
            }
        }
    }

    /* Flush this buffer's L1D-dirty lines down to L2SCP backing. */
    FENCE;
    flush_range_గమ్యం_l2((స్థిరరాశి శూన్య *) ctx->l2_pad_లోపల_buf[buf_idx], ctx->pin_chunk_బైట్లు);
    WAIT_CACHEOPS;
}

పూర్ణాంకం నమోదు_బిందువు(నిర్మాణం ggml_et_ద్విమాన_params * params, శూన్య * env) {
    (శూన్య) env;

    స్థిరరాశి పూర్ణాంకం shire   = get_shire_id();
    స్థిరరాశి పూర్ణాంకం hart_id = get_hart_id();
    స్థిరరాశి పూర్ణాంకం minion  = (hart_id >> 1) & 0x1F;
    స్థిరరాశి పూర్ణాంకం hart1   = hart_id & 1;

    స్థిరరాశి నిర్మాణం ggml_tensor * flt = &params->src0; /* [Kw,Kh,Cin,Cout] */
    స్థిరరాశి నిర్మాణం ggml_tensor * లోపల  = &params->src1; /* [W, H, Cin,N=1 ] */
    నిర్మాణం ggml_tensor *       బహిర్గతం = &params->dst;  /* [W, H, Cout,N=1] */

    స్థిరరాశి పూర్ణాంకం Kw   = (పూర్ణాంకం) flt->ne[0];
    స్థిరరాశి పూర్ణాంకం Kh   = (పూర్ణాంకం) flt->ne[1];
    స్థిరరాశి పూర్ణాంకం Cin  = (పూర్ణాంకం) flt->ne[2];
    స్థిరరాశి పూర్ణాంకం Cout = (పూర్ణాంకం) flt->ne[3];

    స్థిరరాశి పూర్ణాంకం W  = (పూర్ణాంకం) లోపల->ne[0];
    స్థిరరాశి పూర్ణాంకం H  = (పూర్ణాంకం) లోపల->ne[1];
    స్థిరరాశి పూర్ణాంకం OW = (పూర్ణాంకం) బహిర్గతం->ne[0];
    స్థిరరాశి పూర్ణాంకం OH = (పూర్ణాంకం) బహిర్గతం->ne[1];

    /* op_params layout (set by ggml_conv_2d):
         [0]=s0  [1]=s1  [2]=p0  [3]=p1  [4]=d0  [5]=d1   */
    స్థిరరాశి పూర్ణాంకం s0    = బహిర్గతం->op_params[0];
    స్థిరరాశి పూర్ణాంకం s1    = బహిర్గతం->op_params[1];
    స్థిరరాశి పూర్ణాంకం pad_w = బహిర్గతం->op_params[2];
    స్థిరరాశి పూర్ణాంకం pad_h = బహిర్గతం->op_params[3];

    ఒకవేళ (Cin <= 0 || Cout <= 0) {
        తిరిగి -1;
    }
    ఒకవేళ (Cin % TILE != 0 || Cout % TILE != 0) {
        తిరిగి -1;
    }
    ఒకవేళ (W <= 0 || H <= 0) {
        తిరిగి -1;
    }
    ఒకవేళ (s0 <= 0 || s1 <= 0) {
        తిరిగి -1;
    }
    ఒకవేళ (లోపల->ne[2] != Cin || లోపల->ne[3] != 1) {
        తిరిగి -1;
    }
    ఒకవేళ (బహిర్గతం->ne[2] != Cout || బహిర్గతం->ne[3] != 1) {
        తిరిగి -1;
    }
    ఒకవేళ (!flt->దత్తాంశం || !లోపల->దత్తాంశం || !బహిర్గతం->దత్తాంశం) {
        తిరిగి -1;
    }

    స్థిరరాశి పూర్ణాంకం K_TILES = Cin / TILE;
    స్థిరరాశి పూర్ణాంకం M_TILES = Cout / TILE;

    స్థిరరాశి పూర్ణాంకం  Hp         = H + 2 * pad_h;
    స్థిరరాశి పూర్ణాంకం  Wp_a       = విడత_పైకి_tile_i32(OW);
    స్థిరరాశి పూర్ణాంకం  OW_pad     = Wp_a;
    స్థిరరాశి bool need_stage = (OW % TILE != 0);

    /* ===================== Tile assignment & active-shire selection =====
       Computed up front because the per-shire mt set (and thus filter
       region size) depends on n_active_shires. */
    స్థిరరాశి పూర్ణాంకం w_tiles         = ceil_div_i32(OW, TILE);
    స్థిరరాశి పూర్ణాంకం మొత్తం_tiles     = OH * w_tiles * M_TILES;
    స్థిరరాశి పూర్ణాంకం n_చురుకైన_shires = need_stage ? 1 : min_i32(మొత్తం_tiles, N_SHIRES);

    /* Inactive shires exit immediately. No global barrier — pack and
       barriers are now per-shire, so unused shires don't need to vote. */
    ఒకవేళ (shire >= n_చురుకైన_shires) {
        తిరిగి 0;
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
    పూర్ణాంకం my_mt[MAX_MY_MT];
    పూర్ణాంకం n_my_mt = 0;
    కొరకు (పూర్ణాంకం t = shire; t < మొత్తం_tiles; t += n_చురుకైన_shires) {
        స్థిరరాశి పూర్ణాంకం mt    = t % M_TILES;
        bool      దొరికింది = false;
        కొరకు (పూర్ణాంకం j = 0; j < n_my_mt; ++j) {
            ఒకవేళ (my_mt[j] == mt) {
                దొరికింది = true;
                ఆపు;
            }
        }
        ఒకవేళ (!దొరికింది) {
            ఒకవేళ (n_my_mt >= MAX_MY_MT) {
                తిరిగి -1;
            }
            my_mt[n_my_mt++] = mt;
        }
    }
    ఒకవేళ (n_my_mt == 0) {
        తిరిగి 0; /* no tiles for this shire */
    }

    స్థిరరాశి uint64_t filter_local_బైట్లు = (uint64_t) Kh * Kw * n_my_mt * K_TILES * SLAB_బైట్లు;
    ఒకవేళ (filter_local_బైట్లు > LOCAL_FILTER_CAP) {
        తిరిగి -1;
    }

    /* ===================== L2 SCP local layout =========================
       filter (this shire's mt slice) | pin_buf[0] | pin_buf[1]?
                                      | output_stage? | scratch (streaming) */
    స్థిరరాశి uint64_t l2_పునాది   = (uint64_t) et_shire_l2scp_local(0);
    స్థిరరాశి uint64_t l2_filter = l2_పునాది;

    /* Sizing for pin: budget = LOCAL_BUDGET - filter - output_stage.    */
    స్థిరరాశి int64_t output_stage_బైట్లు_నిండిన = need_stage ? (int64_t) Cout * OH * OW_pad * (int64_t) పరిమాణం(దశాంశం) : 0;
    స్థిరరాశి int64_t budget_for_chunks = (int64_t) LOCAL_BUDGET - (int64_t) filter_local_బైట్లు - output_stage_బైట్లు_నిండిన;
    ఒకవేళ (budget_for_chunks <= 0) {
        తిరిగి -1;
    }
    స్థిరరాశి int64_t చొప్పున_KT_pin_బైట్లు = (int64_t) Kw * TILE * Hp * Wp_a * (int64_t) పరిమాణం(దశాంశం);

    పూర్ణాంకం chunk_KT;
    పూర్ణాంకం n_buffers;
    ఒకవేళ ((int64_t) K_TILES * చొప్పున_KT_pin_బైట్లు <= budget_for_chunks) {
        chunk_KT  = K_TILES;
        n_buffers = 1;
    } లేకపోతే {
        chunk_KT = K_TILES;
        అయితే (chunk_KT > 1 && 2 * (int64_t) chunk_KT * చొప్పున_KT_pin_బైట్లు > budget_for_chunks) {
            chunk_KT--;
        }
        అయితే (chunk_KT > 1 && K_TILES % chunk_KT != 0) {
            chunk_KT--;
        }
        n_buffers = (chunk_KT < K_TILES) ? 2 : 1;
        ఒకవేళ (chunk_KT < 1) {
            తిరిగి -1;
        }
    }
    స్థిరరాశి పూర్ణాంకం n_chunks = K_TILES / chunk_KT;

    /* Streaming keeps partial sums in MAX_TILES_PER_HART scratch slots per
       hart. The one-chunk path does not need scratch and can stream a longer
       tile list serially, but multi-chunk shapes must fit this fixed slot
       count until scratch scheduling is made more general. */
    స్థిరరాశి పూర్ణాంకం shire_tile_సామర్థ్యం = shire + MAX_TILES_చొప్పున_HART * n_చురుకైన_shires * N_MIN_చొప్పున_SHIRE;
    ఒకవేళ (n_chunks > 1 && shire_tile_సామర్థ్యం < మొత్తం_tiles) {
        తిరిగి -1;
    }

    స్థిరరాశి uint64_t pin_copy_floats = (uint64_t) chunk_KT * TILE * Hp * Wp_a;
    స్థిరరాశి uint64_t pin_copy_బైట్లు  = pin_copy_floats * పరిమాణం(దశాంశం);
    స్థిరరాశి uint64_t pin_chunk_బైట్లు = (uint64_t) Kw * pin_copy_బైట్లు;

    స్థిరరాశి uint64_t l2_pin_పునాది              = l2_filter + filter_local_బైట్లు;
    స్థిరరాశి uint64_t l2_pin_buf[MAX_DBL_BUFS] = {
        l2_pin_పునాది,
        l2_pin_పునాది + pin_chunk_బైట్లు,
    };

    స్థిరరాశి uint64_t l2_output_stage = need_stage ? l2_pin_పునాది + (uint64_t) n_buffers * pin_chunk_బైట్లు : 0;

    స్థిరరాశి uint64_t scratch_చొప్పున_hart = (uint64_t) MAX_TILES_చొప్పున_HART * (uint64_t) TILE * TILE * పరిమాణం(దశాంశం);
    స్థిరరాశి uint64_t l2_scratch_పునాది  = need_stage ? l2_output_stage + (uint64_t) output_stage_బైట్లు_నిండిన :
                                                   l2_pin_పునాది + (uint64_t) n_buffers * pin_chunk_బైట్లు;

    /* ===================== PHASE 1: Filter pack (per-shire mt slice) ====
       Hart-1's pack only this shire's mt slabs into local L2 SCP. The
       SHIRE barrier below ensures the filter is in L2 SCP backing before
       hart-0's first tensor_load. */
    ఒకవేళ (hart1) {
        pack_filter_local_mt((స్థిరరాశి దశాంశం *) flt->దత్తాంశం, Kh, Kw, Cin, K_TILES, my_mt, n_my_mt, minion, l2_filter);
    }

    /* ===================== Hart 1: pin packer (per chunk) ==============
       Double-buffered prefetch: pack chunk 0 synchronously, then per chunk c
       signal "buf c ready", pack chunk c+1 into the alternate buffer
       (overlaps hart-0's compute on c), signal "buf c done". */
    ఒకవేళ (hart1) {
        స్థిరరాశి pin_ctx_t ctx = {
            .లోపల_పునాది         = (స్థిరరాశి దశాంశం *) లోపల->దత్తాంశం,
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
            .l2_pad_లోపల_buf   = { l2_pin_buf[0], l2_pin_buf[1] },
            .pin_chunk_బైట్లు = pin_chunk_బైట్లు,
        };

        pack_pin_chunk(&ctx, 0, 0); /* prologue */

        కొరకు (పూర్ణాంకం c = 0; c < n_chunks; ++c) {
            et_barrier(ET_BARRIER_SHIRE); /* signal "buf c ready" */
            ఒకవేళ (n_buffers > 1 && c + 1 < n_chunks) {
                pack_pin_chunk(&ctx, c + 1, (c + 1) & 1);
            }
            et_barrier(ET_BARRIER_SHIRE); /* wait "buf c done" */
        }

        ఒకవేళ (need_stage) {
            et_barrier(ET_BARRIER_SHIRE);
        }
        తిరిగి 0;
    }

    /* ===================== Hart 0: matrix engine ======================
       Two execution modes:
         - n_chunks == 1: full Cin in one shot. Each hart processes a list
           of tiles serially; TenC resets between tiles via first_pass=true.
         - n_chunks  > 1: streaming. Each hart owns up to MAX_TILES_PER_HART
           tiles. For each chunk c, restore TenC from scratch[k] (skip on
           c==0), accumulate this chunk's FMAs, then either save TenC back
           to scratch[k] (c < last) or tensor_store directly (c == last). */
    అమరిక_cache_scp();
    CLEAR_TENSOR_ERROR;

    అక్షరం * స్థిరరాశి   బహిర్గతం_పునాది        = need_stage ? (అక్షరం *) l2_output_stage : (అక్షరం *) బహిర్గతం->దత్తాంశం;
    స్థిరరాశి పూర్ణాంకం      compute_OW      = need_stage ? OW_pad : OW;
    స్థిరరాశి uint64_t బహిర్గతం_chan_stride = (uint64_t) OH * (uint64_t) compute_OW * పరిమాణం(దశాంశం);
    స్థిరరాశి uint64_t బహిర్గతం_అడ్డువరుస_stride  = (uint64_t) compute_OW * పరిమాణం(దశాంశం);

    స్థిరరాశి uint64_t a_అడ్డువరుస_stride = (uint64_t) TILE * పరిమాణం(దశాంశం); /* 64 */
    స్థిరరాశి uint64_t b_అడ్డువరుస_stride = (uint64_t) Hp * (uint64_t) Wp_a * పరిమాణం(దశాంశం);

    /* Tile assignment: shire-strided so small workloads spread across
       shires before stacking minions in one shire. */
    స్థిరరాశి పూర్ణాంకం t_ప్రారంభం  = shire + minion * n_చురుకైన_shires;
    స్థిరరాశి పూర్ణాంకం t_stride = n_చురుకైన_shires * N_MIN_చొప్పున_SHIRE;

    ఒకవేళ (n_chunks == 1) {
        et_barrier(ET_BARRIER_SHIRE); /* wait for the (only) pin chunk */

        స్థిరరాశి uint64_t l2_pad_లోపల = l2_pin_buf[0];
        కొరకు (పూర్ణాంకం t = t_ప్రారంభం; t < మొత్తం_tiles; t += t_stride) {
            స్థిరరాశి conv_tile_t tile = decode_tile(t, M_TILES, w_tiles, my_mt, n_my_mt);
            compute_tile_chunk(l2_filter, l2_pad_లోపల, pin_copy_బైట్లు, Kh, Kw, K_TILES, chunk_KT, 0, n_my_mt, Hp, Wp_a,
                               s1, a_అడ్డువరుస_stride, b_అడ్డువరుస_stride, &tile, /*first_fma_clears_tenc=*/true);

            అక్షరం * dst_addr = output_tile_addr(బహిర్గతం_పునాది, &tile, బహిర్గతం_chan_stride, బహిర్గతం_అడ్డువరుస_stride);
            tensor_store(0, 0, 3, (uint64_t) (TILE - 1), (uint64_t) dst_addr, 0, బహిర్గతం_chan_stride);
            tensor_wait(TENSOR_STORE_WAIT);
        }

        et_barrier(ET_BARRIER_SHIRE); /* matches hart-1's second barrier */

    } లేకపోతే {
        /* Streaming path: each hart owns up to MAX_TILES_PER_HART tiles. */
        పూర్ణాంకం my_tiles[MAX_TILES_చొప్పున_HART];
        పూర్ణాంకం n_my_tiles = 0;
        కొరకు (పూర్ణాంకం స్థానం = 0; స్థానం < MAX_TILES_చొప్పున_HART; ++స్థానం) {
            స్థిరరాశి పూర్ణాంకం t = t_ప్రారంభం + స్థానం * t_stride;
            ఒకవేళ (t < మొత్తం_tiles) {
                my_tiles[n_my_tiles++] = t;
            }
        }

        conv_tile_t tiles[MAX_TILES_చొప్పున_HART];
        కొరకు (పూర్ణాంకం k = 0; k < n_my_tiles; ++k) {
            tiles[k] = decode_tile(my_tiles[k], M_TILES, w_tiles, my_mt, n_my_mt);
        }

        స్థిరరాశి uint64_t my_scratch_పునాది = l2_scratch_పునాది + (uint64_t) minion * scratch_చొప్పున_hart;

        కొరకు (పూర్ణాంకం c = 0; c < n_chunks; ++c) {
            et_barrier(ET_BARRIER_SHIRE); /* pin chunk c packed */

            స్థిరరాశి పూర్ణాంకం      buf       = c & 1;
            స్థిరరాశి uint64_t l2_pad_లోపల = l2_pin_buf[buf];
            స్థిరరాశి పూర్ణాంకం      kt_పునాది   = c * chunk_KT;

            కొరకు (పూర్ణాంకం k = 0; k < n_my_tiles; ++k) {
                స్థిరరాశి conv_tile_t * tile = &tiles[k];
                స్థిరరాశి uint64_t      scr  = my_scratch_పునాది + (uint64_t) k * (TILE * TILE * పరిమాణం(దశాంశం));

                స్థిరరాశి bool మొదటి_pass_chunk = (c == 0);
                ఒకవేళ (!మొదటి_pass_chunk) {
                    tenc_restore_from_scratch(scr);
                }

                compute_tile_chunk(l2_filter, l2_pad_లోపల, pin_copy_బైట్లు, Kh, Kw, K_TILES, chunk_KT, kt_పునాది, n_my_mt,
                                   Hp, Wp_a, s1, a_అడ్డువరుస_stride, b_అడ్డువరుస_stride, tile, మొదటి_pass_chunk);

                ఒకవేళ (c == n_chunks - 1) {
                    అక్షరం * dst_addr = output_tile_addr(బహిర్గతం_పునాది, tile, బహిర్గతం_chan_stride, బహిర్గతం_అడ్డువరుస_stride);
                    tensor_store(0, 0, 3, (uint64_t) (TILE - 1), (uint64_t) dst_addr, 0, బహిర్గతం_chan_stride);
                } లేకపోతే {
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
    ఒకవేళ (need_stage) {
        et_barrier(ET_BARRIER_SHIRE);

        ఒకవేళ (minion == 0) {
            స్థిరరాశి దశాంశం * stage = (స్థిరరాశి దశాంశం *) l2_output_stage;
            దశాంశం *       dram  = (దశాంశం *) బహిర్గతం->దత్తాంశం;
            కొరకు (పూర్ణాంకం oc = 0; oc < Cout; ++oc) {
                కొరకు (పూర్ణాంకం oh2 = 0; oh2 < OH; ++oh2) {
                    స్థిరరాశి దశాంశం * src = stage + ((size_t) oc * OH + oh2) * OW_pad;
                    దశాంశం *       dst = dram + ((size_t) oc * OH + oh2) * OW;
                    కొరకు (పూర్ణాంకం ow2 = 0; ow2 < OW; ++ow2) {
                        dst[ow2] = src[ow2];
                    }
                }
            }
            FENCE;
            స్థిరరాశి uint64_t మొత్తం_బైట్లు = (uint64_t) Cout * OH * OW * పరిమాణం(దశాంశం);
            evict_range_past_l2((స్థిరరాశి శూన్య *) dram, మొత్తం_బైట్లు);
            WAIT_CACHEOPS;
        }
    }

    తిరిగి 0;
}
