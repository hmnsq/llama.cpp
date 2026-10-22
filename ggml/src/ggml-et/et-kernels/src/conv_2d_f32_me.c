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

#شامل "ggml_tensor.h"
#شامل "platform.h"
#شامل "tensor.h"

#شامل <etsoc/common/utils.h>
#شامل <stdbool.h>
#شامل <stdint.h>

#تعریف_کرو TILE               16 /* matrix engine native tile in M, K, N */
/* L1 SCP layout: A double-buffered, B single-buffered. Per the SDK doc
   `dst_start` is a 6-bit field (max 63) but empirical testing shows the
   physical L1 SCP per minion is 48 lines — writes to lines >= 48 corrupt.
   So we get 3 × 16-line buffers max: A_0, A_1, B. Pick A as the
   double-buffered operand (filter-slab loads, the longer of the two). */
#تعریف_کرو LSCP_A_0           0  /* A buffer 0 at L1 SCP lines 0..15  */
#تعریف_کرو LSCP_A_1           16 /* A buffer 1 at L1 SCP lines 16..31 */
#تعریف_کرو LSCP_B             32 /* B (single buffer) at lines 32..47 */
#تعریف_کرو N_MIN_فی_SHIRE    32 /* ET-SoC-1 geometry: 32 minions/shire  */
#تعریف_کرو N_SHIRES           32 /* default active shire count           */
#تعریف_کرو MAX_TILES_فی_HART 2  /* per-hart TenC slots (save/restore)   */
#تعریف_کرو MAX_DBL_BUFS       2  /* chunk pack buffers (double-buffered) */

/* Per-shire L2 SCP local budget. Per-shire SCP is 2 MB; we cap at
   1984 KB to leave 64 KB headroom for per-hart TenC scratch (32 minions ×
   2 slots × 1 KB), which lives at the tail of the SCP outside the pin
   sizing budget. Bigger budget here means bigger feasible chunk_KT,
   which means fewer chunks (each chunk costs 2 SHIRE barriers + ~30
   TenC save/restore events per hart). */
#تعریف_کرو LOCAL_BUDGET (1984 * 1024)

/* Cap on the per-shire filter region in local L2 SCP. The shire packs the
   mt values it can consume under the current tile assignment, rather than
   the whole Cout dimension. Reads in the inner loop are then fully
   shire-local — no NoC fanout. */
#تعریف_کرو LOCAL_FILTER_CAP (1024 * 1024)                      /* 1 MB / shire ceiling */

#تعریف_کرو SLAB_بائٹیں ((uint64_t) TILE * TILE * حجم(اعشاری)) /* 1024 */
#تعریف_کرو SLAB_سطریں ((SLAB_بائٹیں + 63) / 64)                 /* 16   */

/* Upper bound on the number of distinct mt values a single shire may pack.
   This keeps the mt list stack-resident. Shapes that need more should fall
   back until the filter-slice bookkeeping is made dynamic. */
#تعریف_کرو MAX_MY_MT (N_MIN_فی_SHIRE * MAX_TILES_فی_HART)

قسمتعریف ساخت {
    صحیح mt;
    صحیح mt_idx;
    صحیح oh;
    صحیح ow_بنیاد;
} conv_tile_t;

ساکن اندرونی صحیح ceil_div_i32(صحیح x, صحیح y) {
    واپس (x + y - 1) / y;
}

ساکن اندرونی صحیح باری_اوپر_tile_i32(صحیح x) {
    واپس (x + TILE - 1) & ~(TILE - 1);
}

ساکن اندرونی صحیح min_i32(صحیح a, صحیح b) {
    واپس a < b ? a : b;
}

ساکن اندرونی uint64_t min_u64(uint64_t a, uint64_t b) {
    واپس a < b ? a : b;
}

/* ===== Vector helpers for hart-1 pack ============================
   Both assume dst (and src for copy) are 32-byte aligned; n is in floats.
   The 8-element tail is handled scalar. f30/f31 are scratch — clobbered
   per-call via the asm clobber list. */
ساکن اندرونی خالی vec_zero_aligned(اعشاری * dst, صحیح n) {
    صحیح       i  = 0;
    ثابت صحیح n8 = n & ~7;
    برائے (; i < n8; i += 8) {
        __asm__ متلون(
            "fsub.ps f31, f31, f31\n"
            "fsw.ps  f31, %[d]\n"
            : [d] "=m"(*(اعشاری (*)[8]) & dst[i])
            :
            : "f31");
    }
    برائے (; i < n; ++i) {
        dst[i] = 0.0f;
    }
}

ساکن اندرونی خالی vec_copy_aligned(اعشاری * dst, ثابت اعشاری * src, صحیح n) {
    صحیح       i  = 0;
    ثابت صحیح n8 = n & ~7;
    برائے (; i < n8; i += 8) {
        __asm__ متلون(
            "flw.ps f30, %[s]\n"
            "fsw.ps f30, %[d]\n"
            : [d] "=m"(*(اعشاری (*)[8]) & dst[i])
            : [s] "m"(*(ثابت اعشاری (*)[8]) & src[i])
            : "f30");
    }
    برائے (; i < n; ++i) {
        dst[i] = src[i];
    }
}

/* ===== TenC save/restore =========================================
   The TenC accumulator IS the f0..f31 vector register file: row N occupies
   f(2N) and f(2N+1) (two 8-fp32 vector regs per row). We save by
   tensor_store-ing TILE rows × 64 bytes, and restore via 32 flw.ps after
   forcing L1D to refetch from the L2SCP backing (tensor_store bypasses L1D
   so the backing is always current). See feedback_tenc_save_restore.md. */
ساکن اندرونی خالی tenc_restore_from_scratch(uint64_t scr) {
    FENCE;
    evict_مقصد_l2((ثابت خالی *) scr, TILE, 64);
    WAIT_CACHEOPS;
    __asm__ متلون(
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
قسمتعریف ساخت {
    ثابت اعشاری * اندر_بنیاد;  /* DRAM input base [Cin][H][W]         */
    صحیح           Kw;
    صحیح           chunk_KT; /* number of K_TILES (=16-wide) per chunk */
    صحیح           H, W, Hp, Wp_a;
    صحیح           pad_h, pad_w, s0;
    صحیح           minion;          /* this hart's minion id (0..31) */
    uint64_t      pin_copy_floats; /* per-_s pin plane size in floats */
    uint64_t      l2_pad_اندر_buf[MAX_DBL_BUFS];
    uint64_t      pin_chunk_بائٹیں; /* one chunk pin buffer's total size */
} pin_ctx_t;

ساکن اندرونی صحیح ڈھونڈو_mt_idx(ثابت صحیح * my_mt, صحیح n_my_mt, صحیح mt) {
    برائے (صحیح j = 0; j < n_my_mt; ++j) {
        اگر (my_mt[j] == mt) {
            واپس j;
        }
    }
    واپس 0;
}

ساکن اندرونی conv_tile_t decode_tile(صحیح t, صحیح M_TILES, صحیح w_tiles, ثابت صحیح * my_mt, صحیح n_my_mt) {
    conv_tile_t tile;
    tile.mt = t % M_TILES;
    t /= M_TILES;
    ثابت صحیح wt = t % w_tiles;
    t /= w_tiles;
    tile.oh      = t;
    tile.ow_بنیاد = wt * TILE;
    tile.mt_idx  = ڈھونڈو_mt_idx(my_mt, n_my_mt, tile.mt);
    واپس tile;
}

ساکن اندرونی uint64_t
filter_slab_addr(uint64_t l2_filter, صحیح Kw, صحیح K_TILES, صحیح n_my_mt, صحیح mt_idx, صحیح kh, صحیح kw, صحیح kt_global) {
    واپس l2_filter + (uint64_t) ((((kh * Kw + kw) * n_my_mt + mt_idx) * K_TILES + kt_global)) * SLAB_بائٹیں;
}

ساکن اندرونی uint64_t pin_tile_addr(uint64_t l2_pad_اندر,
                                     uint64_t pin_copy_بائٹیں,
                                     صحیح      ktc,
                                     صحیح      kw,
                                     صحیح      Hp,
                                     صحیح      Wp_a,
                                     صحیح      oh,
                                     صحیح      ow_بنیاد,
                                     صحیح      s1,
                                     صحیح      kh) {
    ثابت صحیح ir_pad = oh * s1 + kh;
    واپس l2_pad_اندر + (uint64_t) kw * pin_copy_بائٹیں +
           (((uint64_t) (ktc * TILE) * Hp + ir_pad) * Wp_a + ow_بنیاد) * حجم(اعشاری);
}

ساکن اندرونی حرف * output_tile_addr(حرف *              برآمد_بنیاد,
                                      ثابت conv_tile_t * tile,
                                      uint64_t            برآمد_chan_stride,
                                      uint64_t            برآمد_صف_stride) {
    واپس برآمد_بنیاد + (size_t) (tile->mt * TILE) * برآمد_chan_stride + (size_t) tile->oh * برآمد_صف_stride +
           (size_t) tile->ow_بنیاد * حجم(اعشاری);
}

ساکن اندرونی خالی flush_range_مقصد_l2(ثابت خالی * addr, uint64_t n_بائٹیں) {
    ثابت uint64_t کل_سطریں = (n_بائٹیں + 63) / 64;
    ثابت حرف *   fl_addr     = (ثابت حرف *) addr;
    برائے (uint64_t مکمل = 0; مکمل < کل_سطریں;) {
        ثابت uint64_t batch = min_u64(کل_سطریں - مکمل, 16);
        flush_مقصد_l2((ثابت خالی *) (fl_addr + مکمل * 64), batch, 64);
        مکمل += batch;
    }
}

ساکن اندرونی خالی evict_range_past_l2(ثابت خالی * addr, uint64_t n_بائٹیں) {
    ثابت uint64_t کل_سطریں = (n_بائٹیں + 63) / 64;
    ثابت حرف *   fl_addr     = (ثابت حرف *) addr;
    برائے (uint64_t مکمل = 0; مکمل < کل_سطریں;) {
        ثابت uint64_t batch = min_u64(کل_سطریں - مکمل, 16);
        evict_past_l2((ثابت خالی *) (fl_addr + مکمل * 64), batch, 64);
        مکمل += batch;
    }
}

/* One matrix-engine tile for one Cin chunk. This is the main optimization
   surface: A is double-buffered, B is single-buffered due to L1 SCP space. */
ساکن اندرونی خالی compute_tile_chunk(uint64_t            l2_filter,
                                      uint64_t            l2_pad_اندر,
                                      uint64_t            pin_copy_بائٹیں,
                                      صحیح                 Kh,
                                      صحیح                 Kw,
                                      صحیح                 K_TILES,
                                      صحیح                 chunk_KT,
                                      صحیح                 kt_بنیاد,
                                      صحیح                 n_my_mt,
                                      صحیح                 Hp,
                                      صحیح                 Wp_a,
                                      صحیح                 s1,
                                      uint64_t            a_صف_stride,
                                      uint64_t            b_صف_stride,
                                      ثابت conv_tile_t * tile,
                                      bool                پہلا_fma_clears_tenc) {
    ثابت صحیح      n_iters   = Kh * Kw * chunk_KT;
    ثابت uint64_t A_BUFS[2] = { LSCP_A_0, LSCP_A_1 };

    ثابت uint64_t a_addr0 = filter_slab_addr(l2_filter, Kw, K_TILES, n_my_mt, tile->mt_idx, 0, 0, kt_بنیاد);
    tensor_بھرو(false, false, A_BUFS[0], 0, 0, a_addr0, 0, (uint64_t) (TILE - 1), a_صف_stride, 0);

    برائے (صحیح iter = 0; iter < n_iters; ++iter) {
        ثابت صحیح ktc = iter % chunk_KT;
        ثابت صحیح rem = iter / chunk_KT;
        ثابت صحیح kw  = rem % Kw;
        ثابت صحیح kh  = rem / Kw;

        ثابت uint64_t b_addr =
            pin_tile_addr(l2_pad_اندر, pin_copy_بائٹیں, ktc, kw, Hp, Wp_a, tile->oh, tile->ow_بنیاد, s1, kh);
        tensor_بھرو(false, false, LSCP_B, 0, 0, b_addr, 0, (uint64_t) (TILE - 1), b_صف_stride, 1);

        tensor_wait(TENSOR_بھرو_WAIT_0);
        tensor_wait(TENSOR_بھرو_WAIT_1);

        اگر (iter + 1 < n_iters) {
            ثابت صحیح      ktc_n = (iter + 1) % chunk_KT;
            ثابت صحیح      rem_n = (iter + 1) / chunk_KT;
            ثابت صحیح      kw_n  = rem_n % Kw;
            ثابت صحیح      kh_n  = rem_n / Kw;
            ثابت uint64_t a_addr_n =
                filter_slab_addr(l2_filter, Kw, K_TILES, n_my_mt, tile->mt_idx, kh_n, kw_n, kt_بنیاد + ktc_n);
            tensor_بھرو(false, false, A_BUFS[(iter + 1) & 1], 0, 0, a_addr_n, 0, (uint64_t) (TILE - 1), a_صف_stride,
                        0);
        }

        tensor_fma(false, 3, (uint64_t) (TILE - 1), (uint64_t) (TILE - 1), 0, false, false, false, false, LSCP_B,
                   A_BUFS[iter & 1], 0, پہلا_fma_clears_tenc && (iter == 0));
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
ساکن خالی pack_filter_local_mt(ثابت اعشاری * flt_بنیاد,
                                 صحیح           Kh,
                                 صحیح           Kw,
                                 صحیح           Cin,
                                 صحیح           K_TILES,
                                 ثابت صحیح *   my_mt,
                                 صحیح           n_my_mt,
                                 صحیح           minion,
                                 uint64_t      l2_filter_بنیاد) {
    ثابت صحیح    n_slabs = Kh * Kw * n_my_mt * K_TILES;
    ثابت size_t kstep   = (size_t) Kh * Kw; /* Cin stride in floats */

    برائے (صحیح slab = minion; slab < n_slabs; slab += N_MIN_فی_SHIRE) {
        صحیح       t  = slab;
        ثابت صحیح kt = t % K_TILES;
        t /= K_TILES;
        ثابت صحیح mt_idx = t % n_my_mt;
        t /= n_my_mt;
        ثابت صحیح kw = t % Kw;
        t /= Kw;
        ثابت صحیح kh = t;
        ثابت صحیح mt = my_mt[mt_idx];

        ثابت uint64_t slab_انحراف = (uint64_t) slab * SLAB_بائٹیں;
        اعشاری *        cell        = (اعشاری *) (l2_filter_بنیاد + slab_انحراف);

        برائے (صحیح oc_اندر = 0; oc_اندر < TILE; ++oc_اندر) {
            ثابت صحیح     oc  = mt * TILE + oc_اندر;
            ثابت اعشاری * src = flt_بنیاد + (((size_t) oc * Cin + (size_t) kt * TILE) * Kh + kh) * Kw + kw;
            اعشاری *       صف = cell + (size_t) oc_اندر * TILE;
            اعشاری         scratch[TILE] __attribute__((aligned(32)));
            برائے (صحیح ic_اندر = 0; ic_اندر < TILE; ++ic_اندر) {
                scratch[ic_اندر] = src[(size_t) ic_اندر * kstep];
            }
            vec_copy_aligned(صف, scratch, TILE);
        }
    }

    /* Flush this hart's dirty L1D lines for the slabs it wrote. */
    FENCE;
    برائے (صحیح slab = minion; slab < n_slabs; slab += N_MIN_فی_SHIRE) {
        ثابت uint64_t slab_انحراف = (uint64_t) slab * SLAB_بائٹیں;
        flush_مقصد_l2((ثابت خالی *) (l2_filter_بنیاد + slab_انحراف), SLAB_سطریں, 64);
    }
    WAIT_CACHEOPS;
}

/* Pack one Cin chunk of the input pin (Kw shifted padded copies) into the
   buf_idx side of local L2SCP. Work distributed across the 32 hart-1's in
   the shire by `plane % 32 == minion`. The final flush_to_l2 forces L1D
   write-back so hart-0's tensor_load sees the freshly written bytes. */
ساکن خالی pack_pin_chunk(ثابت pin_ctx_t * ctx, صحیح chunk_id, صحیح buf_idx) {
    ثابت صحیح kt_بنیاد  = chunk_id * ctx->chunk_KT;
    ثابت صحیح Kw       = ctx->Kw;
    ثابت صحیح chunk_KT = ctx->chunk_KT;
    ثابت صحیح H = ctx->H, W = ctx->W, Hp = ctx->Hp, Wp_a = ctx->Wp_a;
    ثابت صحیح pad_h = ctx->pad_h, pad_w = ctx->pad_w, s0 = ctx->s0;
    ثابت صحیح minion = ctx->minion;

    /* Pin pack: Kw shifted, padded copies of input rows. Bounds [vlo, vhi)
       hoisted outside the row loop so the inner loop is three regions
       (zero-prefix | bulk-copy | zero-suffix) with no per-element predicate. */
    اعشاری *   pin0         = (اعشاری *) ctx->l2_pad_اندر_buf[buf_idx];
    ثابت صحیح chunk_Cin    = chunk_KT * TILE;
    ثابت صحیح n_pin_planes = Kw * chunk_Cin;
    برائے (صحیح p = minion; p < n_pin_planes; p += N_MIN_فی_SHIRE) {
        ثابت صحیح s     = p / chunk_Cin;
        ثابت صحیح icc   = p % chunk_Cin;
        ثابت صحیح ic    = kt_بنیاد * TILE + icc;
        اعشاری *   pin_s = pin0 + (size_t) s * ctx->pin_copy_floats;

        ثابت صحیح انحراف = s - pad_w;
        صحیح       vlo    = 0;
        جب (vlo < Wp_a && (s0 * vlo + انحراف) < 0) {
            vlo++;
        }
        صحیح vhi = Wp_a;
        جب (vhi > vlo && (s0 * (vhi - 1) + انحراف) >= W) {
            vhi--;
        }
        ثابت bool aligned = (s0 == 1) && ((vlo & 7) == 0) && (((vlo + انحراف) & 7) == 0);

        برائے (صحیح r = 0; r < Hp; ++r) {
            اعشاری *   صف    = pin_s + ((size_t) icc * Hp + r) * Wp_a;
            ثابت صحیح real_h = r - pad_h;
            اگر (real_h < 0 || real_h >= H) {
                vec_zero_aligned(صف, Wp_a);
                جاری;
            }
            ثابت اعشاری * src_صف = ctx->اندر_بنیاد + ((size_t) ic * H + real_h) * W;

            برائے (صحیح cc = 0; cc < vlo; ++cc) {
                صف[cc] = 0.0f;
            }

            اگر (aligned) {
                vec_copy_aligned(صف + vlo, src_صف + vlo + انحراف, vhi - vlo);
            } ورنہ اگر (s0 == 1) {
                ثابت اعشاری * csrc = src_صف + vlo + انحراف;
                ثابت صحیح     n    = vhi - vlo;
                برائے (صحیح cc = 0; cc < n; ++cc) {
                    صف[vlo + cc] = csrc[cc];
                }
            } ورنہ {
                برائے (صحیح cc = vlo; cc < vhi; ++cc) {
                    صف[cc] = src_صف[s0 * cc + انحراف];
                }
            }

            برائے (صحیح cc = vhi; cc < Wp_a; ++cc) {
                صف[cc] = 0.0f;
            }
        }
    }

    /* Flush this buffer's L1D-dirty lines down to L2SCP backing. */
    FENCE;
    flush_range_مقصد_l2((ثابت خالی *) ctx->l2_pad_اندر_buf[buf_idx], ctx->pin_chunk_بائٹیں);
    WAIT_CACHEOPS;
}

صحیح اندراج_نقطہ(ساخت ggml_et_ثنائی_params * params, خالی * env) {
    (خالی) env;

    ثابت صحیح shire   = get_shire_id();
    ثابت صحیح hart_id = get_hart_id();
    ثابت صحیح minion  = (hart_id >> 1) & 0x1F;
    ثابت صحیح hart1   = hart_id & 1;

    ثابت ساخت ggml_tensor * flt = &params->src0; /* [Kw,Kh,Cin,Cout] */
    ثابت ساخت ggml_tensor * اندر  = &params->src1; /* [W, H, Cin,N=1 ] */
    ساخت ggml_tensor *       برآمد = &params->dst;  /* [W, H, Cout,N=1] */

    ثابت صحیح Kw   = (صحیح) flt->ne[0];
    ثابت صحیح Kh   = (صحیح) flt->ne[1];
    ثابت صحیح Cin  = (صحیح) flt->ne[2];
    ثابت صحیح Cout = (صحیح) flt->ne[3];

    ثابت صحیح W  = (صحیح) اندر->ne[0];
    ثابت صحیح H  = (صحیح) اندر->ne[1];
    ثابت صحیح OW = (صحیح) برآمد->ne[0];
    ثابت صحیح OH = (صحیح) برآمد->ne[1];

    /* op_params layout (set by ggml_conv_2d):
         [0]=s0  [1]=s1  [2]=p0  [3]=p1  [4]=d0  [5]=d1   */
    ثابت صحیح s0    = برآمد->op_params[0];
    ثابت صحیح s1    = برآمد->op_params[1];
    ثابت صحیح pad_w = برآمد->op_params[2];
    ثابت صحیح pad_h = برآمد->op_params[3];

    اگر (Cin <= 0 || Cout <= 0) {
        واپس -1;
    }
    اگر (Cin % TILE != 0 || Cout % TILE != 0) {
        واپس -1;
    }
    اگر (W <= 0 || H <= 0) {
        واپس -1;
    }
    اگر (s0 <= 0 || s1 <= 0) {
        واپس -1;
    }
    اگر (اندر->ne[2] != Cin || اندر->ne[3] != 1) {
        واپس -1;
    }
    اگر (برآمد->ne[2] != Cout || برآمد->ne[3] != 1) {
        واپس -1;
    }
    اگر (!flt->ڈیٹا || !اندر->ڈیٹا || !برآمد->ڈیٹا) {
        واپس -1;
    }

    ثابت صحیح K_TILES = Cin / TILE;
    ثابت صحیح M_TILES = Cout / TILE;

    ثابت صحیح  Hp         = H + 2 * pad_h;
    ثابت صحیح  Wp_a       = باری_اوپر_tile_i32(OW);
    ثابت صحیح  OW_pad     = Wp_a;
    ثابت bool need_stage = (OW % TILE != 0);

    /* ===================== Tile assignment & active-shire selection =====
       Computed up front because the per-shire mt set (and thus filter
       region size) depends on n_active_shires. */
    ثابت صحیح w_tiles         = ceil_div_i32(OW, TILE);
    ثابت صحیح کل_tiles     = OH * w_tiles * M_TILES;
    ثابت صحیح n_فعال_shires = need_stage ? 1 : min_i32(کل_tiles, N_SHIRES);

    /* Inactive shires exit immediately. No global barrier — pack and
       barriers are now per-shire, so unused shires don't need to vote. */
    اگر (shire >= n_فعال_shires) {
        واپس 0;
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
    صحیح my_mt[MAX_MY_MT];
    صحیح n_my_mt = 0;
    برائے (صحیح t = shire; t < کل_tiles; t += n_فعال_shires) {
        ثابت صحیح mt    = t % M_TILES;
        bool      ملا = false;
        برائے (صحیح j = 0; j < n_my_mt; ++j) {
            اگر (my_mt[j] == mt) {
                ملا = true;
                توڑو;
            }
        }
        اگر (!ملا) {
            اگر (n_my_mt >= MAX_MY_MT) {
                واپس -1;
            }
            my_mt[n_my_mt++] = mt;
        }
    }
    اگر (n_my_mt == 0) {
        واپس 0; /* no tiles for this shire */
    }

    ثابت uint64_t filter_local_بائٹیں = (uint64_t) Kh * Kw * n_my_mt * K_TILES * SLAB_بائٹیں;
    اگر (filter_local_بائٹیں > LOCAL_FILTER_CAP) {
        واپس -1;
    }

    /* ===================== L2 SCP local layout =========================
       filter (this shire's mt slice) | pin_buf[0] | pin_buf[1]?
                                      | output_stage? | scratch (streaming) */
    ثابت uint64_t l2_بنیاد   = (uint64_t) et_shire_l2scp_local(0);
    ثابت uint64_t l2_filter = l2_بنیاد;

    /* Sizing for pin: budget = LOCAL_BUDGET - filter - output_stage.    */
    ثابت int64_t output_stage_بائٹیں_بھرا = need_stage ? (int64_t) Cout * OH * OW_pad * (int64_t) حجم(اعشاری) : 0;
    ثابت int64_t budget_for_chunks = (int64_t) LOCAL_BUDGET - (int64_t) filter_local_بائٹیں - output_stage_بائٹیں_بھرا;
    اگر (budget_for_chunks <= 0) {
        واپس -1;
    }
    ثابت int64_t فی_KT_pin_بائٹیں = (int64_t) Kw * TILE * Hp * Wp_a * (int64_t) حجم(اعشاری);

    صحیح chunk_KT;
    صحیح n_buffers;
    اگر ((int64_t) K_TILES * فی_KT_pin_بائٹیں <= budget_for_chunks) {
        chunk_KT  = K_TILES;
        n_buffers = 1;
    } ورنہ {
        chunk_KT = K_TILES;
        جب (chunk_KT > 1 && 2 * (int64_t) chunk_KT * فی_KT_pin_بائٹیں > budget_for_chunks) {
            chunk_KT--;
        }
        جب (chunk_KT > 1 && K_TILES % chunk_KT != 0) {
            chunk_KT--;
        }
        n_buffers = (chunk_KT < K_TILES) ? 2 : 1;
        اگر (chunk_KT < 1) {
            واپس -1;
        }
    }
    ثابت صحیح n_chunks = K_TILES / chunk_KT;

    /* Streaming keeps partial sums in MAX_TILES_PER_HART scratch slots per
       hart. The one-chunk path does not need scratch and can stream a longer
       tile list serially, but multi-chunk shapes must fit this fixed slot
       count until scratch scheduling is made more general. */
    ثابت صحیح shire_tile_استعداد = shire + MAX_TILES_فی_HART * n_فعال_shires * N_MIN_فی_SHIRE;
    اگر (n_chunks > 1 && shire_tile_استعداد < کل_tiles) {
        واپس -1;
    }

    ثابت uint64_t pin_copy_floats = (uint64_t) chunk_KT * TILE * Hp * Wp_a;
    ثابت uint64_t pin_copy_بائٹیں  = pin_copy_floats * حجم(اعشاری);
    ثابت uint64_t pin_chunk_بائٹیں = (uint64_t) Kw * pin_copy_بائٹیں;

    ثابت uint64_t l2_pin_بنیاد              = l2_filter + filter_local_بائٹیں;
    ثابت uint64_t l2_pin_buf[MAX_DBL_BUFS] = {
        l2_pin_بنیاد,
        l2_pin_بنیاد + pin_chunk_بائٹیں,
    };

    ثابت uint64_t l2_output_stage = need_stage ? l2_pin_بنیاد + (uint64_t) n_buffers * pin_chunk_بائٹیں : 0;

    ثابت uint64_t scratch_فی_hart = (uint64_t) MAX_TILES_فی_HART * (uint64_t) TILE * TILE * حجم(اعشاری);
    ثابت uint64_t l2_scratch_بنیاد  = need_stage ? l2_output_stage + (uint64_t) output_stage_بائٹیں_بھرا :
                                                   l2_pin_بنیاد + (uint64_t) n_buffers * pin_chunk_بائٹیں;

    /* ===================== PHASE 1: Filter pack (per-shire mt slice) ====
       Hart-1's pack only this shire's mt slabs into local L2 SCP. The
       SHIRE barrier below ensures the filter is in L2 SCP backing before
       hart-0's first tensor_load. */
    اگر (hart1) {
        pack_filter_local_mt((ثابت اعشاری *) flt->ڈیٹا, Kh, Kw, Cin, K_TILES, my_mt, n_my_mt, minion, l2_filter);
    }

    /* ===================== Hart 1: pin packer (per chunk) ==============
       Double-buffered prefetch: pack chunk 0 synchronously, then per chunk c
       signal "buf c ready", pack chunk c+1 into the alternate buffer
       (overlaps hart-0's compute on c), signal "buf c done". */
    اگر (hart1) {
        ثابت pin_ctx_t ctx = {
            .اندر_بنیاد         = (ثابت اعشاری *) اندر->ڈیٹا,
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
            .l2_pad_اندر_buf   = { l2_pin_buf[0], l2_pin_buf[1] },
            .pin_chunk_بائٹیں = pin_chunk_بائٹیں,
        };

        pack_pin_chunk(&ctx, 0, 0); /* prologue */

        برائے (صحیح c = 0; c < n_chunks; ++c) {
            et_barrier(ET_BARRIER_SHIRE); /* signal "buf c ready" */
            اگر (n_buffers > 1 && c + 1 < n_chunks) {
                pack_pin_chunk(&ctx, c + 1, (c + 1) & 1);
            }
            et_barrier(ET_BARRIER_SHIRE); /* wait "buf c done" */
        }

        اگر (need_stage) {
            et_barrier(ET_BARRIER_SHIRE);
        }
        واپس 0;
    }

    /* ===================== Hart 0: matrix engine ======================
       Two execution modes:
         - n_chunks == 1: full Cin in one shot. Each hart processes a list
           of tiles serially; TenC resets between tiles via first_pass=true.
         - n_chunks  > 1: streaming. Each hart owns up to MAX_TILES_PER_HART
           tiles. For each chunk c, restore TenC from scratch[k] (skip on
           c==0), accumulate this chunk's FMAs, then either save TenC back
           to scratch[k] (c < last) or tensor_store directly (c == last). */
    تیاری_cache_scp();
    CLEAR_TENSOR_ERROR;

    حرف * ثابت   برآمد_بنیاد        = need_stage ? (حرف *) l2_output_stage : (حرف *) برآمد->ڈیٹا;
    ثابت صحیح      compute_OW      = need_stage ? OW_pad : OW;
    ثابت uint64_t برآمد_chan_stride = (uint64_t) OH * (uint64_t) compute_OW * حجم(اعشاری);
    ثابت uint64_t برآمد_صف_stride  = (uint64_t) compute_OW * حجم(اعشاری);

    ثابت uint64_t a_صف_stride = (uint64_t) TILE * حجم(اعشاری); /* 64 */
    ثابت uint64_t b_صف_stride = (uint64_t) Hp * (uint64_t) Wp_a * حجم(اعشاری);

    /* Tile assignment: shire-strided so small workloads spread across
       shires before stacking minions in one shire. */
    ثابت صحیح t_آغاز  = shire + minion * n_فعال_shires;
    ثابت صحیح t_stride = n_فعال_shires * N_MIN_فی_SHIRE;

    اگر (n_chunks == 1) {
        et_barrier(ET_BARRIER_SHIRE); /* wait for the (only) pin chunk */

        ثابت uint64_t l2_pad_اندر = l2_pin_buf[0];
        برائے (صحیح t = t_آغاز; t < کل_tiles; t += t_stride) {
            ثابت conv_tile_t tile = decode_tile(t, M_TILES, w_tiles, my_mt, n_my_mt);
            compute_tile_chunk(l2_filter, l2_pad_اندر, pin_copy_بائٹیں, Kh, Kw, K_TILES, chunk_KT, 0, n_my_mt, Hp, Wp_a,
                               s1, a_صف_stride, b_صف_stride, &tile, /*first_fma_clears_tenc=*/true);

            حرف * dst_addr = output_tile_addr(برآمد_بنیاد, &tile, برآمد_chan_stride, برآمد_صف_stride);
            tensor_store(0, 0, 3, (uint64_t) (TILE - 1), (uint64_t) dst_addr, 0, برآمد_chan_stride);
            tensor_wait(TENSOR_STORE_WAIT);
        }

        et_barrier(ET_BARRIER_SHIRE); /* matches hart-1's second barrier */

    } ورنہ {
        /* Streaming path: each hart owns up to MAX_TILES_PER_HART tiles. */
        صحیح my_tiles[MAX_TILES_فی_HART];
        صحیح n_my_tiles = 0;
        برائے (صحیح خانہ = 0; خانہ < MAX_TILES_فی_HART; ++خانہ) {
            ثابت صحیح t = t_آغاز + خانہ * t_stride;
            اگر (t < کل_tiles) {
                my_tiles[n_my_tiles++] = t;
            }
        }

        conv_tile_t tiles[MAX_TILES_فی_HART];
        برائے (صحیح k = 0; k < n_my_tiles; ++k) {
            tiles[k] = decode_tile(my_tiles[k], M_TILES, w_tiles, my_mt, n_my_mt);
        }

        ثابت uint64_t my_scratch_بنیاد = l2_scratch_بنیاد + (uint64_t) minion * scratch_فی_hart;

        برائے (صحیح c = 0; c < n_chunks; ++c) {
            et_barrier(ET_BARRIER_SHIRE); /* pin chunk c packed */

            ثابت صحیح      buf       = c & 1;
            ثابت uint64_t l2_pad_اندر = l2_pin_buf[buf];
            ثابت صحیح      kt_بنیاد   = c * chunk_KT;

            برائے (صحیح k = 0; k < n_my_tiles; ++k) {
                ثابت conv_tile_t * tile = &tiles[k];
                ثابت uint64_t      scr  = my_scratch_بنیاد + (uint64_t) k * (TILE * TILE * حجم(اعشاری));

                ثابت bool پہلا_pass_chunk = (c == 0);
                اگر (!پہلا_pass_chunk) {
                    tenc_restore_from_scratch(scr);
                }

                compute_tile_chunk(l2_filter, l2_pad_اندر, pin_copy_بائٹیں, Kh, Kw, K_TILES, chunk_KT, kt_بنیاد, n_my_mt,
                                   Hp, Wp_a, s1, a_صف_stride, b_صف_stride, tile, پہلا_pass_chunk);

                اگر (c == n_chunks - 1) {
                    حرف * dst_addr = output_tile_addr(برآمد_بنیاد, tile, برآمد_chan_stride, برآمد_صف_stride);
                    tensor_store(0, 0, 3, (uint64_t) (TILE - 1), (uint64_t) dst_addr, 0, برآمد_chan_stride);
                } ورنہ {
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
    اگر (need_stage) {
        et_barrier(ET_BARRIER_SHIRE);

        اگر (minion == 0) {
            ثابت اعشاری * stage = (ثابت اعشاری *) l2_output_stage;
            اعشاری *       dram  = (اعشاری *) برآمد->ڈیٹا;
            برائے (صحیح oc = 0; oc < Cout; ++oc) {
                برائے (صحیح oh2 = 0; oh2 < OH; ++oh2) {
                    ثابت اعشاری * src = stage + ((size_t) oc * OH + oh2) * OW_pad;
                    اعشاری *       dst = dram + ((size_t) oc * OH + oh2) * OW;
                    برائے (صحیح ow2 = 0; ow2 < OW; ++ow2) {
                        dst[ow2] = src[ow2];
                    }
                }
            }
            FENCE;
            ثابت uint64_t کل_بائٹیں = (uint64_t) Cout * OH * OW * حجم(اعشاری);
            evict_range_past_l2((ثابت خالی *) dram, کل_بائٹیں);
            WAIT_CACHEOPS;
        }
    }

    واپس 0;
}
