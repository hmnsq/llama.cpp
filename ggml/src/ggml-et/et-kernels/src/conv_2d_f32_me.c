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

#ضمن "ggml_tensor.h"
#ضمن "platform.h"
#ضمن "tensor.h"

#ضمن <etsoc/common/utils.h>
#ضمن <stdbool.h>
#ضمن <stdint.h>

#define TILE               16 /* matrix engine native tile in M, K, N */
/* L1 SCP layout: A double-buffered, B single-buffered. Per the SDK doc
   `dst_start` is a 6-bit field (max 63) but empirical testing shows the
   physical L1 SCP per minion is 48 lines — writes to lines >= 48 corrupt.
   So we get 3 × 16-line buffers max: A_0, A_1, B. Pick A as the
   double-buffered operand (filter-slab loads, the longer of the two). */
#define LSCP_A_0           0  /* A buffer 0 at L1 SCP lines 0..15  */
#define LSCP_A_1           16 /* A buffer 1 at L1 SCP lines 16..31 */
#define LSCP_B             32 /* B (single buffer) at lines 32..47 */
#define N_MIN_PER_SHIRE    32 /* ET-SoC-1 geometry: 32 minions/shire  */
#define N_SHIRES           32 /* default active shire count           */
#define MAX_TILES_PER_HART 2  /* per-hart TenC slots (save/restore)   */
#define MAX_DBL_BUFS       2  /* chunk pack buffers (double-buffered) */

/* Per-shire L2 SCP local budget. Per-shire SCP is 2 MB; we cap at
   1984 KB to leave 64 KB headroom for per-hart TenC scratch (32 minions ×
   2 slots × 1 KB), which lives at the tail of the SCP outside the pin
   sizing budget. Bigger budget here means bigger feasible chunk_KT,
   which means fewer chunks (each chunk costs 2 SHIRE barriers + ~30
   TenC save/restore events per hart). */
#define LOCAL_BUDGET (1984 * 1024)

/* Cap on the per-shire filter region in local L2 SCP. The shire packs the
   mt values it can consume under the current tile assignment, rather than
   the whole Cout dimension. Reads in the inner loop are then fully
   shire-local — no NoC fanout. */
#define LOCAL_FILTER_CAP (1024 * 1024)                      /* 1 MB / shire ceiling */

#define SLAB_BYTES ((uint64_t) TILE * TILE * حجم(عائم)) /* 1024 */
#define SLAB_سطریں ((SLAB_BYTES + 63) / 64)                 /* 16   */

/* Upper bound on the number of distinct mt values a single shire may pack.
   This keeps the mt list stack-resident. Shapes that need more should fall
   back until the filter-slice bookkeeping is made dynamic. */
#define MAX_MY_MT (N_MIN_PER_SHIRE * MAX_TILES_PER_HART)

تعريفنوع بنية {
    صحيح mt;
    صحيح mt_idx;
    صحيح oh;
    صحيح ow_base;
} conv_tile_t;

ساكن مضمن صحيح ceil_div_i32(صحيح x, صحيح y) {
    أرجع (x + y - 1) / y;
}

ساكن مضمن صحيح round_up_tile_i32(صحيح x) {
    أرجع (x + TILE - 1) & ~(TILE - 1);
}

ساكن مضمن صحيح min_i32(صحيح a, صحيح b) {
    أرجع a < b ? a : b;
}

ساكن مضمن uint64_t min_u64(uint64_t a, uint64_t b) {
    أرجع a < b ? a : b;
}

/* ===== Vector helpers for hart-1 pack ============================
   Both assume dst (and src for copy) are 32-byte aligned; n is in floats.
   The 8-element tail is handled scalar. f30/f31 are scratch — clobbered
   per-call via the asm clobber list. */
ساكن مضمن فراغ vec_zero_aligned(عائم * dst, صحيح n) {
    صحيح       i  = 0;
    ثابت صحيح n8 = n & ~7;
    لكل (; i < n8; i += 8) {
        __asm__ متطاير(
            "fsub.ps f31, f31, f31\n"
            "fsw.ps  f31, %[d]\n"
            : [d] "=m"(*(عائم (*)[8]) & dst[i])
            :
            : "f31");
    }
    لكل (; i < n; ++i) {
        dst[i] = 0.0f;
    }
}

ساكن مضمن فراغ vec_copy_aligned(عائم * dst, ثابت عائم * src, صحيح n) {
    صحيح       i  = 0;
    ثابت صحيح n8 = n & ~7;
    لكل (; i < n8; i += 8) {
        __asm__ متطاير(
            "flw.ps f30, %[s]\n"
            "fsw.ps f30, %[d]\n"
            : [d] "=m"(*(عائم (*)[8]) & dst[i])
            : [s] "m"(*(ثابت عائم (*)[8]) & src[i])
            : "f30");
    }
    لكل (; i < n; ++i) {
        dst[i] = src[i];
    }
}

/* ===== TenC save/restore =========================================
   The TenC accumulator IS the f0..f31 vector register file: row N occupies
   f(2N) and f(2N+1) (two 8-fp32 vector regs per row). We save by
   tensor_store-ing TILE rows × 64 bytes, and restore via 32 flw.ps after
   forcing L1D to refetch from the L2SCP backing (tensor_store bypasses L1D
   so the backing is always current). See feedback_tenc_save_restore.md. */
ساكن مضمن فراغ tenc_restore_from_scratch(uint64_t scr) {
    FENCE;
    evict_to_l2((ثابت فراغ *) scr, TILE, 64);
    WAIT_CACHEOPS;
    __asm__ متطاير(
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
تعريفنوع بنية {
    ثابت عائم * in_base;  /* DRAM input base [Cin][H][W]         */
    صحيح           Kw;
    صحيح           chunk_KT; /* number of K_TILES (=16-wide) per chunk */
    صحيح           H, W, Hp, Wp_a;
    صحيح           pad_h, pad_w, s0;
    صحيح           minion;          /* this hart's minion id (0..31) */
    uint64_t      pin_copy_floats; /* per-_s pin plane size in floats */
    uint64_t      l2_pad_in_buf[MAX_DBL_BUFS];
    uint64_t      pin_chunk_bytes; /* one chunk pin buffer's total size */
} pin_ctx_t;

ساكن مضمن صحيح find_mt_idx(ثابت صحيح * my_mt, صحيح n_my_mt, صحيح mt) {
    لكل (صحيح j = 0; j < n_my_mt; ++j) {
        إذا (my_mt[j] == mt) {
            أرجع j;
        }
    }
    أرجع 0;
}

ساكن مضمن conv_tile_t decode_tile(صحيح t, صحيح M_TILES, صحيح w_tiles, ثابت صحيح * my_mt, صحيح n_my_mt) {
    conv_tile_t tile;
    tile.mt = t % M_TILES;
    t /= M_TILES;
    ثابت صحيح wt = t % w_tiles;
    t /= w_tiles;
    tile.oh      = t;
    tile.ow_base = wt * TILE;
    tile.mt_idx  = find_mt_idx(my_mt, n_my_mt, tile.mt);
    أرجع tile;
}

ساكن مضمن uint64_t
filter_slab_addr(uint64_t l2_filter, صحيح Kw, صحيح K_TILES, صحيح n_my_mt, صحيح mt_idx, صحيح kh, صحيح kw, صحيح kt_global) {
    أرجع l2_filter + (uint64_t) ((((kh * Kw + kw) * n_my_mt + mt_idx) * K_TILES + kt_global)) * SLAB_BYTES;
}

ساكن مضمن uint64_t pin_tile_addr(uint64_t l2_pad_in,
                                     uint64_t pin_copy_bytes,
                                     صحيح      ktc,
                                     صحيح      kw,
                                     صحيح      Hp,
                                     صحيح      Wp_a,
                                     صحيح      oh,
                                     صحيح      ow_base,
                                     صحيح      s1,
                                     صحيح      kh) {
    ثابت صحيح ir_pad = oh * s1 + kh;
    أرجع l2_pad_in + (uint64_t) kw * pin_copy_bytes +
           (((uint64_t) (ktc * TILE) * Hp + ir_pad) * Wp_a + ow_base) * حجم(عائم);
}

ساكن مضمن حرف * output_tile_addr(حرف *              out_base,
                                      ثابت conv_tile_t * tile,
                                      uint64_t            out_chan_stride,
                                      uint64_t            out_row_stride) {
    أرجع out_base + (size_t) (tile->mt * TILE) * out_chan_stride + (size_t) tile->oh * out_row_stride +
           (size_t) tile->ow_base * حجم(عائم);
}

ساكن مضمن فراغ flush_range_to_l2(ثابت فراغ * addr, uint64_t n_bytes) {
    ثابت uint64_t total_سطریں = (n_bytes + 63) / 64;
    ثابت حرف *   fl_addr     = (ثابت حرف *) addr;
    لكل (uint64_t done = 0; done < total_سطریں;) {
        ثابت uint64_t batch = min_u64(total_سطریں - done, 16);
        flush_to_l2((ثابت فراغ *) (fl_addr + done * 64), batch, 64);
        done += batch;
    }
}

ساكن مضمن فراغ evict_range_past_l2(ثابت فراغ * addr, uint64_t n_bytes) {
    ثابت uint64_t total_سطریں = (n_bytes + 63) / 64;
    ثابت حرف *   fl_addr     = (ثابت حرف *) addr;
    لكل (uint64_t done = 0; done < total_سطریں;) {
        ثابت uint64_t batch = min_u64(total_سطریں - done, 16);
        evict_past_l2((ثابت فراغ *) (fl_addr + done * 64), batch, 64);
        done += batch;
    }
}

/* One matrix-engine tile for one Cin chunk. This is the main optimization
   surface: A is double-buffered, B is single-buffered due to L1 SCP space. */
ساكن مضمن فراغ compute_tile_chunk(uint64_t            l2_filter,
                                      uint64_t            l2_pad_in,
                                      uint64_t            pin_copy_bytes,
                                      صحيح                 Kh,
                                      صحيح                 Kw,
                                      صحيح                 K_TILES,
                                      صحيح                 chunk_KT,
                                      صحيح                 kt_base,
                                      صحيح                 n_my_mt,
                                      صحيح                 Hp,
                                      صحيح                 Wp_a,
                                      صحيح                 s1,
                                      uint64_t            a_row_stride,
                                      uint64_t            b_row_stride,
                                      ثابت conv_tile_t * tile,
                                      bool                first_fma_clears_tenc) {
    ثابت صحيح      n_iters   = Kh * Kw * chunk_KT;
    ثابت uint64_t A_BUFS[2] = { LSCP_A_0, LSCP_A_1 };

    ثابت uint64_t a_addr0 = filter_slab_addr(l2_filter, Kw, K_TILES, n_my_mt, tile->mt_idx, 0, 0, kt_base);
    tensor_load(false, false, A_BUFS[0], 0, 0, a_addr0, 0, (uint64_t) (TILE - 1), a_row_stride, 0);

    لكل (صحيح iter = 0; iter < n_iters; ++iter) {
        ثابت صحيح ktc = iter % chunk_KT;
        ثابت صحيح rem = iter / chunk_KT;
        ثابت صحيح kw  = rem % Kw;
        ثابت صحيح kh  = rem / Kw;

        ثابت uint64_t b_addr =
            pin_tile_addr(l2_pad_in, pin_copy_bytes, ktc, kw, Hp, Wp_a, tile->oh, tile->ow_base, s1, kh);
        tensor_load(false, false, LSCP_B, 0, 0, b_addr, 0, (uint64_t) (TILE - 1), b_row_stride, 1);

        tensor_wait(TENSOR_LOAD_WAIT_0);
        tensor_wait(TENSOR_LOAD_WAIT_1);

        إذا (iter + 1 < n_iters) {
            ثابت صحيح      ktc_n = (iter + 1) % chunk_KT;
            ثابت صحيح      rem_n = (iter + 1) / chunk_KT;
            ثابت صحيح      kw_n  = rem_n % Kw;
            ثابت صحيح      kh_n  = rem_n / Kw;
            ثابت uint64_t a_addr_n =
                filter_slab_addr(l2_filter, Kw, K_TILES, n_my_mt, tile->mt_idx, kh_n, kw_n, kt_base + ktc_n);
            tensor_load(false, false, A_BUFS[(iter + 1) & 1], 0, 0, a_addr_n, 0, (uint64_t) (TILE - 1), a_row_stride,
                        0);
        }

        tensor_fma(false, 3, (uint64_t) (TILE - 1), (uint64_t) (TILE - 1), 0, false, false, false, false, LSCP_B,
                   A_BUFS[iter & 1], 0, first_fma_clears_tenc && (iter == 0));
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
ساكن فراغ pack_filter_local_mt(ثابت عائم * flt_base,
                                 صحيح           Kh,
                                 صحيح           Kw,
                                 صحيح           Cin,
                                 صحيح           K_TILES,
                                 ثابت صحيح *   my_mt,
                                 صحيح           n_my_mt,
                                 صحيح           minion,
                                 uint64_t      l2_filter_base) {
    ثابت صحيح    n_slabs = Kh * Kw * n_my_mt * K_TILES;
    ثابت size_t kstep   = (size_t) Kh * Kw; /* Cin stride in floats */

    لكل (صحيح slab = minion; slab < n_slabs; slab += N_MIN_PER_SHIRE) {
        صحيح       t  = slab;
        ثابت صحيح kt = t % K_TILES;
        t /= K_TILES;
        ثابت صحيح mt_idx = t % n_my_mt;
        t /= n_my_mt;
        ثابت صحيح kw = t % Kw;
        t /= Kw;
        ثابت صحيح kh = t;
        ثابت صحيح mt = my_mt[mt_idx];

        ثابت uint64_t slab_offset = (uint64_t) slab * SLAB_BYTES;
        عائم *        cell        = (عائم *) (l2_filter_base + slab_offset);

        لكل (صحيح oc_in = 0; oc_in < TILE; ++oc_in) {
            ثابت صحيح     oc  = mt * TILE + oc_in;
            ثابت عائم * src = flt_base + (((size_t) oc * Cin + (size_t) kt * TILE) * Kh + kh) * Kw + kw;
            عائم *       row = cell + (size_t) oc_in * TILE;
            عائم         scratch[TILE] __attribute__((aligned(32)));
            لكل (صحيح ic_in = 0; ic_in < TILE; ++ic_in) {
                scratch[ic_in] = src[(size_t) ic_in * kstep];
            }
            vec_copy_aligned(row, scratch, TILE);
        }
    }

    /* Flush this hart's dirty L1D lines for the slabs it wrote. */
    FENCE;
    لكل (صحيح slab = minion; slab < n_slabs; slab += N_MIN_PER_SHIRE) {
        ثابت uint64_t slab_offset = (uint64_t) slab * SLAB_BYTES;
        flush_to_l2((ثابت فراغ *) (l2_filter_base + slab_offset), SLAB_سطریں, 64);
    }
    WAIT_CACHEOPS;
}

/* Pack one Cin chunk of the input pin (Kw shifted padded copies) into the
   buf_idx side of local L2SCP. Work distributed across the 32 hart-1's in
   the shire by `plane % 32 == minion`. The final flush_to_l2 forces L1D
   write-back so hart-0's tensor_load sees the freshly written bytes. */
ساكن فراغ pack_pin_chunk(ثابت pin_ctx_t * ctx, صحيح chunk_id, صحيح buf_idx) {
    ثابت صحيح kt_base  = chunk_id * ctx->chunk_KT;
    ثابت صحيح Kw       = ctx->Kw;
    ثابت صحيح chunk_KT = ctx->chunk_KT;
    ثابت صحيح H = ctx->H, W = ctx->W, Hp = ctx->Hp, Wp_a = ctx->Wp_a;
    ثابت صحيح pad_h = ctx->pad_h, pad_w = ctx->pad_w, s0 = ctx->s0;
    ثابت صحيح minion = ctx->minion;

    /* Pin pack: Kw shifted, padded copies of input rows. Bounds [vlo, vhi)
       hoisted outside the row loop so the inner loop is three regions
       (zero-prefix | bulk-copy | zero-suffix) with no per-element predicate. */
    عائم *   pin0         = (عائم *) ctx->l2_pad_in_buf[buf_idx];
    ثابت صحيح chunk_Cin    = chunk_KT * TILE;
    ثابت صحيح n_pin_planes = Kw * chunk_Cin;
    لكل (صحيح p = minion; p < n_pin_planes; p += N_MIN_PER_SHIRE) {
        ثابت صحيح s     = p / chunk_Cin;
        ثابت صحيح icc   = p % chunk_Cin;
        ثابت صحيح ic    = kt_base * TILE + icc;
        عائم *   pin_s = pin0 + (size_t) s * ctx->pin_copy_floats;

        ثابت صحيح offset = s - pad_w;
        صحيح       vlo    = 0;
        طالما (vlo < Wp_a && (s0 * vlo + offset) < 0) {
            vlo++;
        }
        صحيح vhi = Wp_a;
        طالما (vhi > vlo && (s0 * (vhi - 1) + offset) >= W) {
            vhi--;
        }
        ثابت bool aligned = (s0 == 1) && ((vlo & 7) == 0) && (((vlo + offset) & 7) == 0);

        لكل (صحيح r = 0; r < Hp; ++r) {
            عائم *   row    = pin_s + ((size_t) icc * Hp + r) * Wp_a;
            ثابت صحيح real_h = r - pad_h;
            إذا (real_h < 0 || real_h >= H) {
                vec_zero_aligned(row, Wp_a);
                استمر;
            }
            ثابت عائم * src_row = ctx->in_base + ((size_t) ic * H + real_h) * W;

            لكل (صحيح cc = 0; cc < vlo; ++cc) {
                row[cc] = 0.0f;
            }

            إذا (aligned) {
                vec_copy_aligned(row + vlo, src_row + vlo + offset, vhi - vlo);
            } وإلا إذا (s0 == 1) {
                ثابت عائم * csrc = src_row + vlo + offset;
                ثابت صحيح     n    = vhi - vlo;
                لكل (صحيح cc = 0; cc < n; ++cc) {
                    row[vlo + cc] = csrc[cc];
                }
            } وإلا {
                لكل (صحيح cc = vlo; cc < vhi; ++cc) {
                    row[cc] = src_row[s0 * cc + offset];
                }
            }

            لكل (صحيح cc = vhi; cc < Wp_a; ++cc) {
                row[cc] = 0.0f;
            }
        }
    }

    /* Flush this buffer's L1D-dirty lines down to L2SCP backing. */
    FENCE;
    flush_range_to_l2((ثابت فراغ *) ctx->l2_pad_in_buf[buf_idx], ctx->pin_chunk_bytes);
    WAIT_CACHEOPS;
}

صحيح entry_point(بنية ggml_et_ثنائی_params * params, فراغ * env) {
    (فراغ) env;

    ثابت صحيح shire   = get_shire_id();
    ثابت صحيح hart_id = get_hart_id();
    ثابت صحيح minion  = (hart_id >> 1) & 0x1F;
    ثابت صحيح hart1   = hart_id & 1;

    ثابت بنية ggml_tensor * flt = &params->src0; /* [Kw,Kh,Cin,Cout] */
    ثابت بنية ggml_tensor * in  = &params->src1; /* [W, H, Cin,N=1 ] */
    بنية ggml_tensor *       out = &params->dst;  /* [W, H, Cout,N=1] */

    ثابت صحيح Kw   = (صحيح) flt->ne[0];
    ثابت صحيح Kh   = (صحيح) flt->ne[1];
    ثابت صحيح Cin  = (صحيح) flt->ne[2];
    ثابت صحيح Cout = (صحيح) flt->ne[3];

    ثابت صحيح W  = (صحيح) in->ne[0];
    ثابت صحيح H  = (صحيح) in->ne[1];
    ثابت صحيح OW = (صحيح) out->ne[0];
    ثابت صحيح OH = (صحيح) out->ne[1];

    /* op_params layout (set by ggml_conv_2d):
         [0]=s0  [1]=s1  [2]=p0  [3]=p1  [4]=d0  [5]=d1   */
    ثابت صحيح s0    = out->op_params[0];
    ثابت صحيح s1    = out->op_params[1];
    ثابت صحيح pad_w = out->op_params[2];
    ثابت صحيح pad_h = out->op_params[3];

    إذا (Cin <= 0 || Cout <= 0) {
        أرجع -1;
    }
    إذا (Cin % TILE != 0 || Cout % TILE != 0) {
        أرجع -1;
    }
    إذا (W <= 0 || H <= 0) {
        أرجع -1;
    }
    إذا (s0 <= 0 || s1 <= 0) {
        أرجع -1;
    }
    إذا (in->ne[2] != Cin || in->ne[3] != 1) {
        أرجع -1;
    }
    إذا (out->ne[2] != Cout || out->ne[3] != 1) {
        أرجع -1;
    }
    إذا (!flt->data || !in->data || !out->data) {
        أرجع -1;
    }

    ثابت صحيح K_TILES = Cin / TILE;
    ثابت صحيح M_TILES = Cout / TILE;

    ثابت صحيح  Hp         = H + 2 * pad_h;
    ثابت صحيح  Wp_a       = round_up_tile_i32(OW);
    ثابت صحيح  OW_pad     = Wp_a;
    ثابت bool need_stage = (OW % TILE != 0);

    /* ===================== Tile assignment & active-shire selection =====
       Computed up front because the per-shire mt set (and thus filter
       region size) depends on n_active_shires. */
    ثابت صحيح w_tiles         = ceil_div_i32(OW, TILE);
    ثابت صحيح total_tiles     = OH * w_tiles * M_TILES;
    ثابت صحيح n_active_shires = need_stage ? 1 : min_i32(total_tiles, N_SHIRES);

    /* Inactive shires exit immediately. No global barrier — pack and
       barriers are now per-shire, so unused shires don't need to vote. */
    إذا (shire >= n_active_shires) {
        أرجع 0;
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
    صحيح my_mt[MAX_MY_MT];
    صحيح n_my_mt = 0;
    لكل (صحيح t = shire; t < total_tiles; t += n_active_shires) {
        ثابت صحيح mt    = t % M_TILES;
        bool      found = false;
        لكل (صحيح j = 0; j < n_my_mt; ++j) {
            إذا (my_mt[j] == mt) {
                found = true;
                اكسر;
            }
        }
        إذا (!found) {
            إذا (n_my_mt >= MAX_MY_MT) {
                أرجع -1;
            }
            my_mt[n_my_mt++] = mt;
        }
    }
    إذا (n_my_mt == 0) {
        أرجع 0; /* no tiles for this shire */
    }

    ثابت uint64_t filter_local_bytes = (uint64_t) Kh * Kw * n_my_mt * K_TILES * SLAB_BYTES;
    إذا (filter_local_bytes > LOCAL_FILTER_CAP) {
        أرجع -1;
    }

    /* ===================== L2 SCP local layout =========================
       filter (this shire's mt slice) | pin_buf[0] | pin_buf[1]?
                                      | output_stage? | scratch (streaming) */
    ثابت uint64_t l2_base   = (uint64_t) et_shire_l2scp_local(0);
    ثابت uint64_t l2_filter = l2_base;

    /* Sizing for pin: budget = LOCAL_BUDGET - filter - output_stage.    */
    ثابت int64_t output_stage_bytes_full = need_stage ? (int64_t) Cout * OH * OW_pad * (int64_t) حجم(عائم) : 0;
    ثابت int64_t budget_for_chunks = (int64_t) LOCAL_BUDGET - (int64_t) filter_local_bytes - output_stage_bytes_full;
    إذا (budget_for_chunks <= 0) {
        أرجع -1;
    }
    ثابت int64_t per_KT_pin_bytes = (int64_t) Kw * TILE * Hp * Wp_a * (int64_t) حجم(عائم);

    صحيح chunk_KT;
    صحيح n_buffers;
    إذا ((int64_t) K_TILES * per_KT_pin_bytes <= budget_for_chunks) {
        chunk_KT  = K_TILES;
        n_buffers = 1;
    } وإلا {
        chunk_KT = K_TILES;
        طالما (chunk_KT > 1 && 2 * (int64_t) chunk_KT * per_KT_pin_bytes > budget_for_chunks) {
            chunk_KT--;
        }
        طالما (chunk_KT > 1 && K_TILES % chunk_KT != 0) {
            chunk_KT--;
        }
        n_buffers = (chunk_KT < K_TILES) ? 2 : 1;
        إذا (chunk_KT < 1) {
            أرجع -1;
        }
    }
    ثابت صحيح n_chunks = K_TILES / chunk_KT;

    /* Streaming keeps partial sums in MAX_TILES_PER_HART scratch slots per
       hart. The one-chunk path does not need scratch and can stream a longer
       tile list serially, but multi-chunk shapes must fit this fixed slot
       count until scratch scheduling is made more general. */
    ثابت صحيح shire_tile_capacity = shire + MAX_TILES_PER_HART * n_active_shires * N_MIN_PER_SHIRE;
    إذا (n_chunks > 1 && shire_tile_capacity < total_tiles) {
        أرجع -1;
    }

    ثابت uint64_t pin_copy_floats = (uint64_t) chunk_KT * TILE * Hp * Wp_a;
    ثابت uint64_t pin_copy_bytes  = pin_copy_floats * حجم(عائم);
    ثابت uint64_t pin_chunk_bytes = (uint64_t) Kw * pin_copy_bytes;

    ثابت uint64_t l2_pin_base              = l2_filter + filter_local_bytes;
    ثابت uint64_t l2_pin_buf[MAX_DBL_BUFS] = {
        l2_pin_base,
        l2_pin_base + pin_chunk_bytes,
    };

    ثابت uint64_t l2_output_stage = need_stage ? l2_pin_base + (uint64_t) n_buffers * pin_chunk_bytes : 0;

    ثابت uint64_t scratch_per_hart = (uint64_t) MAX_TILES_PER_HART * (uint64_t) TILE * TILE * حجم(عائم);
    ثابت uint64_t l2_scratch_base  = need_stage ? l2_output_stage + (uint64_t) output_stage_bytes_full :
                                                   l2_pin_base + (uint64_t) n_buffers * pin_chunk_bytes;

    /* ===================== PHASE 1: Filter pack (per-shire mt slice) ====
       Hart-1's pack only this shire's mt slabs into local L2 SCP. The
       SHIRE barrier below ensures the filter is in L2 SCP backing before
       hart-0's first tensor_load. */
    إذا (hart1) {
        pack_filter_local_mt((ثابت عائم *) flt->data, Kh, Kw, Cin, K_TILES, my_mt, n_my_mt, minion, l2_filter);
    }

    /* ===================== Hart 1: pin packer (per chunk) ==============
       Double-buffered prefetch: pack chunk 0 synchronously, then per chunk c
       signal "buf c ready", pack chunk c+1 into the alternate buffer
       (overlaps hart-0's compute on c), signal "buf c done". */
    إذا (hart1) {
        ثابت pin_ctx_t ctx = {
            .in_base         = (ثابت عائم *) in->data,
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
            .l2_pad_in_buf   = { l2_pin_buf[0], l2_pin_buf[1] },
            .pin_chunk_bytes = pin_chunk_bytes,
        };

        pack_pin_chunk(&ctx, 0, 0); /* prologue */

        لكل (صحيح c = 0; c < n_chunks; ++c) {
            et_barrier(ET_BARRIER_SHIRE); /* signal "buf c ready" */
            إذا (n_buffers > 1 && c + 1 < n_chunks) {
                pack_pin_chunk(&ctx, c + 1, (c + 1) & 1);
            }
            et_barrier(ET_BARRIER_SHIRE); /* wait "buf c done" */
        }

        إذا (need_stage) {
            et_barrier(ET_BARRIER_SHIRE);
        }
        أرجع 0;
    }

    /* ===================== Hart 0: matrix engine ======================
       Two execution modes:
         - n_chunks == 1: full Cin in one shot. Each hart processes a list
           of tiles serially; TenC resets between tiles via first_pass=true.
         - n_chunks  > 1: streaming. Each hart owns up to MAX_TILES_PER_HART
           tiles. For each chunk c, restore TenC from scratch[k] (skip on
           c==0), accumulate this chunk's FMAs, then either save TenC back
           to scratch[k] (c < last) or tensor_store directly (c == last). */
    setup_cache_scp();
    CLEAR_TENSOR_ERROR;

    حرف * ثابت   out_base        = need_stage ? (حرف *) l2_output_stage : (حرف *) out->data;
    ثابت صحيح      compute_OW      = need_stage ? OW_pad : OW;
    ثابت uint64_t out_chan_stride = (uint64_t) OH * (uint64_t) compute_OW * حجم(عائم);
    ثابت uint64_t out_row_stride  = (uint64_t) compute_OW * حجم(عائم);

    ثابت uint64_t a_row_stride = (uint64_t) TILE * حجم(عائم); /* 64 */
    ثابت uint64_t b_row_stride = (uint64_t) Hp * (uint64_t) Wp_a * حجم(عائم);

    /* Tile assignment: shire-strided so small workloads spread across
       shires before stacking minions in one shire. */
    ثابت صحيح t_start  = shire + minion * n_active_shires;
    ثابت صحيح t_stride = n_active_shires * N_MIN_PER_SHIRE;

    إذا (n_chunks == 1) {
        et_barrier(ET_BARRIER_SHIRE); /* wait for the (only) pin chunk */

        ثابت uint64_t l2_pad_in = l2_pin_buf[0];
        لكل (صحيح t = t_start; t < total_tiles; t += t_stride) {
            ثابت conv_tile_t tile = decode_tile(t, M_TILES, w_tiles, my_mt, n_my_mt);
            compute_tile_chunk(l2_filter, l2_pad_in, pin_copy_bytes, Kh, Kw, K_TILES, chunk_KT, 0, n_my_mt, Hp, Wp_a,
                               s1, a_row_stride, b_row_stride, &tile, /*first_fma_clears_tenc=*/true);

            حرف * dst_addr = output_tile_addr(out_base, &tile, out_chan_stride, out_row_stride);
            tensor_store(0, 0, 3, (uint64_t) (TILE - 1), (uint64_t) dst_addr, 0, out_chan_stride);
            tensor_wait(TENSOR_STORE_WAIT);
        }

        et_barrier(ET_BARRIER_SHIRE); /* matches hart-1's second barrier */

    } وإلا {
        /* Streaming path: each hart owns up to MAX_TILES_PER_HART tiles. */
        صحيح my_tiles[MAX_TILES_PER_HART];
        صحيح n_my_tiles = 0;
        لكل (صحيح slot = 0; slot < MAX_TILES_PER_HART; ++slot) {
            ثابت صحيح t = t_start + slot * t_stride;
            إذا (t < total_tiles) {
                my_tiles[n_my_tiles++] = t;
            }
        }

        conv_tile_t tiles[MAX_TILES_PER_HART];
        لكل (صحيح k = 0; k < n_my_tiles; ++k) {
            tiles[k] = decode_tile(my_tiles[k], M_TILES, w_tiles, my_mt, n_my_mt);
        }

        ثابت uint64_t my_scratch_base = l2_scratch_base + (uint64_t) minion * scratch_per_hart;

        لكل (صحيح c = 0; c < n_chunks; ++c) {
            et_barrier(ET_BARRIER_SHIRE); /* pin chunk c packed */

            ثابت صحيح      buf       = c & 1;
            ثابت uint64_t l2_pad_in = l2_pin_buf[buf];
            ثابت صحيح      kt_base   = c * chunk_KT;

            لكل (صحيح k = 0; k < n_my_tiles; ++k) {
                ثابت conv_tile_t * tile = &tiles[k];
                ثابت uint64_t      scr  = my_scratch_base + (uint64_t) k * (TILE * TILE * حجم(عائم));

                ثابت bool first_pass_chunk = (c == 0);
                إذا (!first_pass_chunk) {
                    tenc_restore_from_scratch(scr);
                }

                compute_tile_chunk(l2_filter, l2_pad_in, pin_copy_bytes, Kh, Kw, K_TILES, chunk_KT, kt_base, n_my_mt,
                                   Hp, Wp_a, s1, a_row_stride, b_row_stride, tile, first_pass_chunk);

                إذا (c == n_chunks - 1) {
                    حرف * dst_addr = output_tile_addr(out_base, tile, out_chan_stride, out_row_stride);
                    tensor_store(0, 0, 3, (uint64_t) (TILE - 1), (uint64_t) dst_addr, 0, out_chan_stride);
                } وإلا {
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
    إذا (need_stage) {
        et_barrier(ET_BARRIER_SHIRE);

        إذا (minion == 0) {
            ثابت عائم * stage = (ثابت عائم *) l2_output_stage;
            عائم *       dram  = (عائم *) out->data;
            لكل (صحيح oc = 0; oc < Cout; ++oc) {
                لكل (صحيح oh2 = 0; oh2 < OH; ++oh2) {
                    ثابت عائم * src = stage + ((size_t) oc * OH + oh2) * OW_pad;
                    عائم *       dst = dram + ((size_t) oc * OH + oh2) * OW;
                    لكل (صحيح ow2 = 0; ow2 < OW; ++ow2) {
                        dst[ow2] = src[ow2];
                    }
                }
            }
            FENCE;
            ثابت uint64_t total_bytes = (uint64_t) Cout * OH * OW * حجم(عائم);
            evict_range_past_l2((ثابت فراغ *) dram, total_bytes);
            WAIT_CACHEOPS;
        }
    }

    أرجع 0;
}
