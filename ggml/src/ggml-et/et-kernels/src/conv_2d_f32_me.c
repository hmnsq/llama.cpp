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

#அந்தர்பாவ "ggml_tensor.h"
#அந்தர்பாவ "platform.h"
#அந்தர்பாவ "tensor.h"

#அந்தர்பாவ <etsoc/common/utils.h>
#அந்தர்பாவ <stdbool.h>
#அந்தர்பாவ <stdint.h>

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

#define SLAB_BYTES ((uint64_t) TILE * TILE * அளவு(தசம)) /* 1024 */
#define SLAB_வரிகள் ((SLAB_BYTES + 63) / 64)                 /* 16   */

/* Upper bound on the number of distinct mt values a single shire may pack.
   This keeps the mt list stack-resident. Shapes that need more should fall
   back until the filter-slice bookkeeping is made dynamic. */
#define MAX_MY_MT (N_MIN_PER_SHIRE * MAX_TILES_PER_HART)

வகைவரை கட்டமைப்பு {
    முழு mt;
    முழு mt_idx;
    முழு oh;
    முழு ow_base;
} conv_tile_t;

நிலையான உள்வரி முழு ceil_div_i32(முழு x, முழு y) {
    திரும்பு (x + y - 1) / y;
}

நிலையான உள்வரி முழு round_up_tile_i32(முழு x) {
    திரும்பு (x + TILE - 1) & ~(TILE - 1);
}

நிலையான உள்வரி முழு min_i32(முழு a, முழு b) {
    திரும்பு a < b ? a : b;
}

நிலையான உள்வரி uint64_t min_u64(uint64_t a, uint64_t b) {
    திரும்பு a < b ? a : b;
}

/* ===== Vector helpers for hart-1 pack ============================
   Both assume dst (and src for copy) are 32-byte aligned; n is in floats.
   The 8-element tail is handled scalar. f30/f31 are scratch — clobbered
   per-call via the asm clobber list. */
நிலையான உள்வரி வெறுமை vec_zero_aligned(தசம * dst, முழு n) {
    முழு       i  = 0;
    மாறிலி முழு n8 = n & ~7;
    க்கு (; i < n8; i += 8) {
        __asm__ நிலையற்ற(
            "fsub.ps f31, f31, f31\n"
            "fsw.ps  f31, %[d]\n"
            : [d] "=m"(*(தசம (*)[8]) & dst[i])
            :
            : "f31");
    }
    க்கு (; i < n; ++i) {
        dst[i] = 0.0f;
    }
}

நிலையான உள்வரி வெறுமை vec_copy_aligned(தசம * dst, மாறிலி தசம * src, முழு n) {
    முழு       i  = 0;
    மாறிலி முழு n8 = n & ~7;
    க்கு (; i < n8; i += 8) {
        __asm__ நிலையற்ற(
            "flw.ps f30, %[s]\n"
            "fsw.ps f30, %[d]\n"
            : [d] "=m"(*(தசம (*)[8]) & dst[i])
            : [s] "m"(*(மாறிலி தசம (*)[8]) & src[i])
            : "f30");
    }
    க்கு (; i < n; ++i) {
        dst[i] = src[i];
    }
}

/* ===== TenC save/restore =========================================
   The TenC accumulator IS the f0..f31 vector register file: row N occupies
   f(2N) and f(2N+1) (two 8-fp32 vector regs per row). We save by
   tensor_store-ing TILE rows × 64 bytes, and restore via 32 flw.ps after
   forcing L1D to refetch from the L2SCP backing (tensor_store bypasses L1D
   so the backing is always current). See feedback_tenc_save_restore.md. */
நிலையான உள்வரி வெறுமை tenc_restore_from_scratch(uint64_t scr) {
    FENCE;
    evict_to_l2((மாறிலி வெறுமை *) scr, TILE, 64);
    WAIT_CACHEOPS;
    __asm__ நிலையற்ற(
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
வகைவரை கட்டமைப்பு {
    மாறிலி தசம * in_base;  /* DRAM input base [Cin][H][W]         */
    முழு           Kw;
    முழு           chunk_KT; /* number of K_TILES (=16-wide) per chunk */
    முழு           H, W, Hp, Wp_a;
    முழு           pad_h, pad_w, s0;
    முழு           minion;          /* this hart's minion id (0..31) */
    uint64_t      pin_copy_floats; /* per-_s pin plane size in floats */
    uint64_t      l2_pad_in_buf[MAX_DBL_BUFS];
    uint64_t      pin_chunk_bytes; /* one chunk pin buffer's total size */
} pin_ctx_t;

நிலையான உள்வரி முழு find_mt_idx(மாறிலி முழு * my_mt, முழு n_my_mt, முழு mt) {
    க்கு (முழு j = 0; j < n_my_mt; ++j) {
        எனில் (my_mt[j] == mt) {
            திரும்பு j;
        }
    }
    திரும்பு 0;
}

நிலையான உள்வரி conv_tile_t decode_tile(முழு t, முழு M_TILES, முழு w_tiles, மாறிலி முழு * my_mt, முழு n_my_mt) {
    conv_tile_t tile;
    tile.mt = t % M_TILES;
    t /= M_TILES;
    மாறிலி முழு wt = t % w_tiles;
    t /= w_tiles;
    tile.oh      = t;
    tile.ow_base = wt * TILE;
    tile.mt_idx  = find_mt_idx(my_mt, n_my_mt, tile.mt);
    திரும்பு tile;
}

நிலையான உள்வரி uint64_t
filter_slab_addr(uint64_t l2_filter, முழு Kw, முழு K_TILES, முழு n_my_mt, முழு mt_idx, முழு kh, முழு kw, முழு kt_global) {
    திரும்பு l2_filter + (uint64_t) ((((kh * Kw + kw) * n_my_mt + mt_idx) * K_TILES + kt_global)) * SLAB_BYTES;
}

நிலையான உள்வரி uint64_t pin_tile_addr(uint64_t l2_pad_in,
                                     uint64_t pin_copy_bytes,
                                     முழு      ktc,
                                     முழு      kw,
                                     முழு      Hp,
                                     முழு      Wp_a,
                                     முழு      oh,
                                     முழு      ow_base,
                                     முழு      s1,
                                     முழு      kh) {
    மாறிலி முழு ir_pad = oh * s1 + kh;
    திரும்பு l2_pad_in + (uint64_t) kw * pin_copy_bytes +
           (((uint64_t) (ktc * TILE) * Hp + ir_pad) * Wp_a + ow_base) * அளவு(தசம);
}

நிலையான உள்வரி எழுத்து * output_tile_addr(எழுத்து *              out_base,
                                      மாறிலி conv_tile_t * tile,
                                      uint64_t            out_chan_stride,
                                      uint64_t            out_row_stride) {
    திரும்பு out_base + (size_t) (tile->mt * TILE) * out_chan_stride + (size_t) tile->oh * out_row_stride +
           (size_t) tile->ow_base * அளவு(தசம);
}

நிலையான உள்வரி வெறுமை flush_range_to_l2(மாறிலி வெறுமை * addr, uint64_t n_bytes) {
    மாறிலி uint64_t total_வரிகள் = (n_bytes + 63) / 64;
    மாறிலி எழுத்து *   fl_addr     = (மாறிலி எழுத்து *) addr;
    க்கு (uint64_t done = 0; done < total_வரிகள்;) {
        மாறிலி uint64_t batch = min_u64(total_வரிகள் - done, 16);
        flush_to_l2((மாறிலி வெறுமை *) (fl_addr + done * 64), batch, 64);
        done += batch;
    }
}

நிலையான உள்வரி வெறுமை evict_range_past_l2(மாறிலி வெறுமை * addr, uint64_t n_bytes) {
    மாறிலி uint64_t total_வரிகள் = (n_bytes + 63) / 64;
    மாறிலி எழுத்து *   fl_addr     = (மாறிலி எழுத்து *) addr;
    க்கு (uint64_t done = 0; done < total_வரிகள்;) {
        மாறிலி uint64_t batch = min_u64(total_வரிகள் - done, 16);
        evict_past_l2((மாறிலி வெறுமை *) (fl_addr + done * 64), batch, 64);
        done += batch;
    }
}

/* One matrix-engine tile for one Cin chunk. This is the main optimization
   surface: A is double-buffered, B is single-buffered due to L1 SCP space. */
நிலையான உள்வரி வெறுமை compute_tile_chunk(uint64_t            l2_filter,
                                      uint64_t            l2_pad_in,
                                      uint64_t            pin_copy_bytes,
                                      முழு                 Kh,
                                      முழு                 Kw,
                                      முழு                 K_TILES,
                                      முழு                 chunk_KT,
                                      முழு                 kt_base,
                                      முழு                 n_my_mt,
                                      முழு                 Hp,
                                      முழு                 Wp_a,
                                      முழு                 s1,
                                      uint64_t            a_row_stride,
                                      uint64_t            b_row_stride,
                                      மாறிலி conv_tile_t * tile,
                                      bool                first_fma_clears_tenc) {
    மாறிலி முழு      n_iters   = Kh * Kw * chunk_KT;
    மாறிலி uint64_t A_BUFS[2] = { LSCP_A_0, LSCP_A_1 };

    மாறிலி uint64_t a_addr0 = filter_slab_addr(l2_filter, Kw, K_TILES, n_my_mt, tile->mt_idx, 0, 0, kt_base);
    tensor_load(false, false, A_BUFS[0], 0, 0, a_addr0, 0, (uint64_t) (TILE - 1), a_row_stride, 0);

    க்கு (முழு iter = 0; iter < n_iters; ++iter) {
        மாறிலி முழு ktc = iter % chunk_KT;
        மாறிலி முழு rem = iter / chunk_KT;
        மாறிலி முழு kw  = rem % Kw;
        மாறிலி முழு kh  = rem / Kw;

        மாறிலி uint64_t b_addr =
            pin_tile_addr(l2_pad_in, pin_copy_bytes, ktc, kw, Hp, Wp_a, tile->oh, tile->ow_base, s1, kh);
        tensor_load(false, false, LSCP_B, 0, 0, b_addr, 0, (uint64_t) (TILE - 1), b_row_stride, 1);

        tensor_wait(TENSOR_LOAD_WAIT_0);
        tensor_wait(TENSOR_LOAD_WAIT_1);

        எனில் (iter + 1 < n_iters) {
            மாறிலி முழு      ktc_n = (iter + 1) % chunk_KT;
            மாறிலி முழு      rem_n = (iter + 1) / chunk_KT;
            மாறிலி முழு      kw_n  = rem_n % Kw;
            மாறிலி முழு      kh_n  = rem_n / Kw;
            மாறிலி uint64_t a_addr_n =
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
நிலையான வெறுமை pack_filter_local_mt(மாறிலி தசம * flt_base,
                                 முழு           Kh,
                                 முழு           Kw,
                                 முழு           Cin,
                                 முழு           K_TILES,
                                 மாறிலி முழு *   my_mt,
                                 முழு           n_my_mt,
                                 முழு           minion,
                                 uint64_t      l2_filter_base) {
    மாறிலி முழு    n_slabs = Kh * Kw * n_my_mt * K_TILES;
    மாறிலி size_t kstep   = (size_t) Kh * Kw; /* Cin stride in floats */

    க்கு (முழு slab = minion; slab < n_slabs; slab += N_MIN_PER_SHIRE) {
        முழு       t  = slab;
        மாறிலி முழு kt = t % K_TILES;
        t /= K_TILES;
        மாறிலி முழு mt_idx = t % n_my_mt;
        t /= n_my_mt;
        மாறிலி முழு kw = t % Kw;
        t /= Kw;
        மாறிலி முழு kh = t;
        மாறிலி முழு mt = my_mt[mt_idx];

        மாறிலி uint64_t slab_offset = (uint64_t) slab * SLAB_BYTES;
        தசம *        cell        = (தசம *) (l2_filter_base + slab_offset);

        க்கு (முழு oc_in = 0; oc_in < TILE; ++oc_in) {
            மாறிலி முழு     oc  = mt * TILE + oc_in;
            மாறிலி தசம * src = flt_base + (((size_t) oc * Cin + (size_t) kt * TILE) * Kh + kh) * Kw + kw;
            தசம *       row = cell + (size_t) oc_in * TILE;
            தசம         scratch[TILE] __attribute__((aligned(32)));
            க்கு (முழு ic_in = 0; ic_in < TILE; ++ic_in) {
                scratch[ic_in] = src[(size_t) ic_in * kstep];
            }
            vec_copy_aligned(row, scratch, TILE);
        }
    }

    /* Flush this hart's dirty L1D lines for the slabs it wrote. */
    FENCE;
    க்கு (முழு slab = minion; slab < n_slabs; slab += N_MIN_PER_SHIRE) {
        மாறிலி uint64_t slab_offset = (uint64_t) slab * SLAB_BYTES;
        flush_to_l2((மாறிலி வெறுமை *) (l2_filter_base + slab_offset), SLAB_வரிகள், 64);
    }
    WAIT_CACHEOPS;
}

/* Pack one Cin chunk of the input pin (Kw shifted padded copies) into the
   buf_idx side of local L2SCP. Work distributed across the 32 hart-1's in
   the shire by `plane % 32 == minion`. The final flush_to_l2 forces L1D
   write-back so hart-0's tensor_load sees the freshly written bytes. */
நிலையான வெறுமை pack_pin_chunk(மாறிலி pin_ctx_t * ctx, முழு chunk_id, முழு buf_idx) {
    மாறிலி முழு kt_base  = chunk_id * ctx->chunk_KT;
    மாறிலி முழு Kw       = ctx->Kw;
    மாறிலி முழு chunk_KT = ctx->chunk_KT;
    மாறிலி முழு H = ctx->H, W = ctx->W, Hp = ctx->Hp, Wp_a = ctx->Wp_a;
    மாறிலி முழு pad_h = ctx->pad_h, pad_w = ctx->pad_w, s0 = ctx->s0;
    மாறிலி முழு minion = ctx->minion;

    /* Pin pack: Kw shifted, padded copies of input rows. Bounds [vlo, vhi)
       hoisted outside the row loop so the inner loop is three regions
       (zero-prefix | bulk-copy | zero-suffix) with no per-element predicate. */
    தசம *   pin0         = (தசம *) ctx->l2_pad_in_buf[buf_idx];
    மாறிலி முழு chunk_Cin    = chunk_KT * TILE;
    மாறிலி முழு n_pin_planes = Kw * chunk_Cin;
    க்கு (முழு p = minion; p < n_pin_planes; p += N_MIN_PER_SHIRE) {
        மாறிலி முழு s     = p / chunk_Cin;
        மாறிலி முழு icc   = p % chunk_Cin;
        மாறிலி முழு ic    = kt_base * TILE + icc;
        தசம *   pin_s = pin0 + (size_t) s * ctx->pin_copy_floats;

        மாறிலி முழு offset = s - pad_w;
        முழு       vlo    = 0;
        போது (vlo < Wp_a && (s0 * vlo + offset) < 0) {
            vlo++;
        }
        முழு vhi = Wp_a;
        போது (vhi > vlo && (s0 * (vhi - 1) + offset) >= W) {
            vhi--;
        }
        மாறிலி bool aligned = (s0 == 1) && ((vlo & 7) == 0) && (((vlo + offset) & 7) == 0);

        க்கு (முழு r = 0; r < Hp; ++r) {
            தசம *   row    = pin_s + ((size_t) icc * Hp + r) * Wp_a;
            மாறிலி முழு real_h = r - pad_h;
            எனில் (real_h < 0 || real_h >= H) {
                vec_zero_aligned(row, Wp_a);
                தொடர்;
            }
            மாறிலி தசம * src_row = ctx->in_base + ((size_t) ic * H + real_h) * W;

            க்கு (முழு cc = 0; cc < vlo; ++cc) {
                row[cc] = 0.0f;
            }

            எனில் (aligned) {
                vec_copy_aligned(row + vlo, src_row + vlo + offset, vhi - vlo);
            } இல்லையெனில் எனில் (s0 == 1) {
                மாறிலி தசம * csrc = src_row + vlo + offset;
                மாறிலி முழு     n    = vhi - vlo;
                க்கு (முழு cc = 0; cc < n; ++cc) {
                    row[vlo + cc] = csrc[cc];
                }
            } இல்லையெனில் {
                க்கு (முழு cc = vlo; cc < vhi; ++cc) {
                    row[cc] = src_row[s0 * cc + offset];
                }
            }

            க்கு (முழு cc = vhi; cc < Wp_a; ++cc) {
                row[cc] = 0.0f;
            }
        }
    }

    /* Flush this buffer's L1D-dirty lines down to L2SCP backing. */
    FENCE;
    flush_range_to_l2((மாறிலி வெறுமை *) ctx->l2_pad_in_buf[buf_idx], ctx->pin_chunk_bytes);
    WAIT_CACHEOPS;
}

முழு entry_point(கட்டமைப்பு ggml_et_இருமம்_params * params, வெறுமை * env) {
    (வெறுமை) env;

    மாறிலி முழு shire   = get_shire_id();
    மாறிலி முழு hart_id = get_hart_id();
    மாறிலி முழு minion  = (hart_id >> 1) & 0x1F;
    மாறிலி முழு hart1   = hart_id & 1;

    மாறிலி கட்டமைப்பு ggml_tensor * flt = &params->src0; /* [Kw,Kh,Cin,Cout] */
    மாறிலி கட்டமைப்பு ggml_tensor * in  = &params->src1; /* [W, H, Cin,N=1 ] */
    கட்டமைப்பு ggml_tensor *       out = &params->dst;  /* [W, H, Cout,N=1] */

    மாறிலி முழு Kw   = (முழு) flt->ne[0];
    மாறிலி முழு Kh   = (முழு) flt->ne[1];
    மாறிலி முழு Cin  = (முழு) flt->ne[2];
    மாறிலி முழு Cout = (முழு) flt->ne[3];

    மாறிலி முழு W  = (முழு) in->ne[0];
    மாறிலி முழு H  = (முழு) in->ne[1];
    மாறிலி முழு OW = (முழு) out->ne[0];
    மாறிலி முழு OH = (முழு) out->ne[1];

    /* op_params layout (set by ggml_conv_2d):
         [0]=s0  [1]=s1  [2]=p0  [3]=p1  [4]=d0  [5]=d1   */
    மாறிலி முழு s0    = out->op_params[0];
    மாறிலி முழு s1    = out->op_params[1];
    மாறிலி முழு pad_w = out->op_params[2];
    மாறிலி முழு pad_h = out->op_params[3];

    எனில் (Cin <= 0 || Cout <= 0) {
        திரும்பு -1;
    }
    எனில் (Cin % TILE != 0 || Cout % TILE != 0) {
        திரும்பு -1;
    }
    எனில் (W <= 0 || H <= 0) {
        திரும்பு -1;
    }
    எனில் (s0 <= 0 || s1 <= 0) {
        திரும்பு -1;
    }
    எனில் (in->ne[2] != Cin || in->ne[3] != 1) {
        திரும்பு -1;
    }
    எனில் (out->ne[2] != Cout || out->ne[3] != 1) {
        திரும்பு -1;
    }
    எனில் (!flt->data || !in->data || !out->data) {
        திரும்பு -1;
    }

    மாறிலி முழு K_TILES = Cin / TILE;
    மாறிலி முழு M_TILES = Cout / TILE;

    மாறிலி முழு  Hp         = H + 2 * pad_h;
    மாறிலி முழு  Wp_a       = round_up_tile_i32(OW);
    மாறிலி முழு  OW_pad     = Wp_a;
    மாறிலி bool need_stage = (OW % TILE != 0);

    /* ===================== Tile assignment & active-shire selection =====
       Computed up front because the per-shire mt set (and thus filter
       region size) depends on n_active_shires. */
    மாறிலி முழு w_tiles         = ceil_div_i32(OW, TILE);
    மாறிலி முழு total_tiles     = OH * w_tiles * M_TILES;
    மாறிலி முழு n_active_shires = need_stage ? 1 : min_i32(total_tiles, N_SHIRES);

    /* Inactive shires exit immediately. No global barrier — pack and
       barriers are now per-shire, so unused shires don't need to vote. */
    எனில் (shire >= n_active_shires) {
        திரும்பு 0;
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
    முழு my_mt[MAX_MY_MT];
    முழு n_my_mt = 0;
    க்கு (முழு t = shire; t < total_tiles; t += n_active_shires) {
        மாறிலி முழு mt    = t % M_TILES;
        bool      found = false;
        க்கு (முழு j = 0; j < n_my_mt; ++j) {
            எனில் (my_mt[j] == mt) {
                found = true;
                நிறுத்து;
            }
        }
        எனில் (!found) {
            எனில் (n_my_mt >= MAX_MY_MT) {
                திரும்பு -1;
            }
            my_mt[n_my_mt++] = mt;
        }
    }
    எனில் (n_my_mt == 0) {
        திரும்பு 0; /* no tiles for this shire */
    }

    மாறிலி uint64_t filter_local_bytes = (uint64_t) Kh * Kw * n_my_mt * K_TILES * SLAB_BYTES;
    எனில் (filter_local_bytes > LOCAL_FILTER_CAP) {
        திரும்பு -1;
    }

    /* ===================== L2 SCP local layout =========================
       filter (this shire's mt slice) | pin_buf[0] | pin_buf[1]?
                                      | output_stage? | scratch (streaming) */
    மாறிலி uint64_t l2_base   = (uint64_t) et_shire_l2scp_local(0);
    மாறிலி uint64_t l2_filter = l2_base;

    /* Sizing for pin: budget = LOCAL_BUDGET - filter - output_stage.    */
    மாறிலி int64_t output_stage_bytes_full = need_stage ? (int64_t) Cout * OH * OW_pad * (int64_t) அளவு(தசம) : 0;
    மாறிலி int64_t budget_for_chunks = (int64_t) LOCAL_BUDGET - (int64_t) filter_local_bytes - output_stage_bytes_full;
    எனில் (budget_for_chunks <= 0) {
        திரும்பு -1;
    }
    மாறிலி int64_t per_KT_pin_bytes = (int64_t) Kw * TILE * Hp * Wp_a * (int64_t) அளவு(தசம);

    முழு chunk_KT;
    முழு n_buffers;
    எனில் ((int64_t) K_TILES * per_KT_pin_bytes <= budget_for_chunks) {
        chunk_KT  = K_TILES;
        n_buffers = 1;
    } இல்லையெனில் {
        chunk_KT = K_TILES;
        போது (chunk_KT > 1 && 2 * (int64_t) chunk_KT * per_KT_pin_bytes > budget_for_chunks) {
            chunk_KT--;
        }
        போது (chunk_KT > 1 && K_TILES % chunk_KT != 0) {
            chunk_KT--;
        }
        n_buffers = (chunk_KT < K_TILES) ? 2 : 1;
        எனில் (chunk_KT < 1) {
            திரும்பு -1;
        }
    }
    மாறிலி முழு n_chunks = K_TILES / chunk_KT;

    /* Streaming keeps partial sums in MAX_TILES_PER_HART scratch slots per
       hart. The one-chunk path does not need scratch and can stream a longer
       tile list serially, but multi-chunk shapes must fit this fixed slot
       count until scratch scheduling is made more general. */
    மாறிலி முழு shire_tile_capacity = shire + MAX_TILES_PER_HART * n_active_shires * N_MIN_PER_SHIRE;
    எனில் (n_chunks > 1 && shire_tile_capacity < total_tiles) {
        திரும்பு -1;
    }

    மாறிலி uint64_t pin_copy_floats = (uint64_t) chunk_KT * TILE * Hp * Wp_a;
    மாறிலி uint64_t pin_copy_bytes  = pin_copy_floats * அளவு(தசம);
    மாறிலி uint64_t pin_chunk_bytes = (uint64_t) Kw * pin_copy_bytes;

    மாறிலி uint64_t l2_pin_base              = l2_filter + filter_local_bytes;
    மாறிலி uint64_t l2_pin_buf[MAX_DBL_BUFS] = {
        l2_pin_base,
        l2_pin_base + pin_chunk_bytes,
    };

    மாறிலி uint64_t l2_output_stage = need_stage ? l2_pin_base + (uint64_t) n_buffers * pin_chunk_bytes : 0;

    மாறிலி uint64_t scratch_per_hart = (uint64_t) MAX_TILES_PER_HART * (uint64_t) TILE * TILE * அளவு(தசம);
    மாறிலி uint64_t l2_scratch_base  = need_stage ? l2_output_stage + (uint64_t) output_stage_bytes_full :
                                                   l2_pin_base + (uint64_t) n_buffers * pin_chunk_bytes;

    /* ===================== PHASE 1: Filter pack (per-shire mt slice) ====
       Hart-1's pack only this shire's mt slabs into local L2 SCP. The
       SHIRE barrier below ensures the filter is in L2 SCP backing before
       hart-0's first tensor_load. */
    எனில் (hart1) {
        pack_filter_local_mt((மாறிலி தசம *) flt->data, Kh, Kw, Cin, K_TILES, my_mt, n_my_mt, minion, l2_filter);
    }

    /* ===================== Hart 1: pin packer (per chunk) ==============
       Double-buffered prefetch: pack chunk 0 synchronously, then per chunk c
       signal "buf c ready", pack chunk c+1 into the alternate buffer
       (overlaps hart-0's compute on c), signal "buf c done". */
    எனில் (hart1) {
        மாறிலி pin_ctx_t ctx = {
            .in_base         = (மாறிலி தசம *) in->data,
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

        க்கு (முழு c = 0; c < n_chunks; ++c) {
            et_barrier(ET_BARRIER_SHIRE); /* signal "buf c ready" */
            எனில் (n_buffers > 1 && c + 1 < n_chunks) {
                pack_pin_chunk(&ctx, c + 1, (c + 1) & 1);
            }
            et_barrier(ET_BARRIER_SHIRE); /* wait "buf c done" */
        }

        எனில் (need_stage) {
            et_barrier(ET_BARRIER_SHIRE);
        }
        திரும்பு 0;
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

    எழுத்து * மாறிலி   out_base        = need_stage ? (எழுத்து *) l2_output_stage : (எழுத்து *) out->data;
    மாறிலி முழு      compute_OW      = need_stage ? OW_pad : OW;
    மாறிலி uint64_t out_chan_stride = (uint64_t) OH * (uint64_t) compute_OW * அளவு(தசம);
    மாறிலி uint64_t out_row_stride  = (uint64_t) compute_OW * அளவு(தசம);

    மாறிலி uint64_t a_row_stride = (uint64_t) TILE * அளவு(தசம); /* 64 */
    மாறிலி uint64_t b_row_stride = (uint64_t) Hp * (uint64_t) Wp_a * அளவு(தசம);

    /* Tile assignment: shire-strided so small workloads spread across
       shires before stacking minions in one shire. */
    மாறிலி முழு t_start  = shire + minion * n_active_shires;
    மாறிலி முழு t_stride = n_active_shires * N_MIN_PER_SHIRE;

    எனில் (n_chunks == 1) {
        et_barrier(ET_BARRIER_SHIRE); /* wait for the (only) pin chunk */

        மாறிலி uint64_t l2_pad_in = l2_pin_buf[0];
        க்கு (முழு t = t_start; t < total_tiles; t += t_stride) {
            மாறிலி conv_tile_t tile = decode_tile(t, M_TILES, w_tiles, my_mt, n_my_mt);
            compute_tile_chunk(l2_filter, l2_pad_in, pin_copy_bytes, Kh, Kw, K_TILES, chunk_KT, 0, n_my_mt, Hp, Wp_a,
                               s1, a_row_stride, b_row_stride, &tile, /*first_fma_clears_tenc=*/true);

            எழுத்து * dst_addr = output_tile_addr(out_base, &tile, out_chan_stride, out_row_stride);
            tensor_store(0, 0, 3, (uint64_t) (TILE - 1), (uint64_t) dst_addr, 0, out_chan_stride);
            tensor_wait(TENSOR_STORE_WAIT);
        }

        et_barrier(ET_BARRIER_SHIRE); /* matches hart-1's second barrier */

    } இல்லையெனில் {
        /* Streaming path: each hart owns up to MAX_TILES_PER_HART tiles. */
        முழு my_tiles[MAX_TILES_PER_HART];
        முழு n_my_tiles = 0;
        க்கு (முழு slot = 0; slot < MAX_TILES_PER_HART; ++slot) {
            மாறிலி முழு t = t_start + slot * t_stride;
            எனில் (t < total_tiles) {
                my_tiles[n_my_tiles++] = t;
            }
        }

        conv_tile_t tiles[MAX_TILES_PER_HART];
        க்கு (முழு k = 0; k < n_my_tiles; ++k) {
            tiles[k] = decode_tile(my_tiles[k], M_TILES, w_tiles, my_mt, n_my_mt);
        }

        மாறிலி uint64_t my_scratch_base = l2_scratch_base + (uint64_t) minion * scratch_per_hart;

        க்கு (முழு c = 0; c < n_chunks; ++c) {
            et_barrier(ET_BARRIER_SHIRE); /* pin chunk c packed */

            மாறிலி முழு      buf       = c & 1;
            மாறிலி uint64_t l2_pad_in = l2_pin_buf[buf];
            மாறிலி முழு      kt_base   = c * chunk_KT;

            க்கு (முழு k = 0; k < n_my_tiles; ++k) {
                மாறிலி conv_tile_t * tile = &tiles[k];
                மாறிலி uint64_t      scr  = my_scratch_base + (uint64_t) k * (TILE * TILE * அளவு(தசம));

                மாறிலி bool first_pass_chunk = (c == 0);
                எனில் (!first_pass_chunk) {
                    tenc_restore_from_scratch(scr);
                }

                compute_tile_chunk(l2_filter, l2_pad_in, pin_copy_bytes, Kh, Kw, K_TILES, chunk_KT, kt_base, n_my_mt,
                                   Hp, Wp_a, s1, a_row_stride, b_row_stride, tile, first_pass_chunk);

                எனில் (c == n_chunks - 1) {
                    எழுத்து * dst_addr = output_tile_addr(out_base, tile, out_chan_stride, out_row_stride);
                    tensor_store(0, 0, 3, (uint64_t) (TILE - 1), (uint64_t) dst_addr, 0, out_chan_stride);
                } இல்லையெனில் {
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
    எனில் (need_stage) {
        et_barrier(ET_BARRIER_SHIRE);

        எனில் (minion == 0) {
            மாறிலி தசம * stage = (மாறிலி தசம *) l2_output_stage;
            தசம *       dram  = (தசம *) out->data;
            க்கு (முழு oc = 0; oc < Cout; ++oc) {
                க்கு (முழு oh2 = 0; oh2 < OH; ++oh2) {
                    மாறிலி தசம * src = stage + ((size_t) oc * OH + oh2) * OW_pad;
                    தசம *       dst = dram + ((size_t) oc * OH + oh2) * OW;
                    க்கு (முழு ow2 = 0; ow2 < OW; ++ow2) {
                        dst[ow2] = src[ow2];
                    }
                }
            }
            FENCE;
            மாறிலி uint64_t total_bytes = (uint64_t) Cout * OH * OW * அளவு(தசம);
            evict_range_past_l2((மாறிலி வெறுமை *) dram, total_bytes);
            WAIT_CACHEOPS;
        }
    }

    திரும்பு 0;
}
