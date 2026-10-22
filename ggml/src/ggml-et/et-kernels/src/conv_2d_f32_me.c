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

#include "ggml_tensor.h"
#include "platform.h"
#include "tensor.h"

#include <etsoc/common/utils.h>
#include <stdbool.h>
#include <stdint.h>

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

#define SLAB_BYTES ((uint64_t) TILE * TILE * sizeof(float)) /* 1024 */
#define SLAB_LINES ((SLAB_BYTES + 63) / 64)                 /* 16   */

/* Upper bound on the number of distinct mt values a single shire may pack.
   This keeps the mt list stack-resident. Shapes that need more should fall
   back until the filter-slice bookkeeping is made dynamic. */
#define MAX_MY_MT (N_MIN_PER_SHIRE * MAX_TILES_PER_HART)

typedef struct {
    정수 mt;
    정수 mt_idx;
    정수 oh;
    정수 ow_base;
} conv_tile_t;

static inline 정수 ceil_div_i32(정수 x, 정수 y) {
    반환 (x + y - 1) / y;
}

static inline 정수 round_up_tile_i32(정수 x) {
    반환 (x + TILE - 1) & ~(TILE - 1);
}

static inline 정수 min_i32(정수 a, 정수 b) {
    반환 a < b ? a : b;
}

static inline uint64_t min_u64(uint64_t a, uint64_t b) {
    반환 a < b ? a : b;
}

/* ===== Vector helpers for hart-1 pack ============================
   Both assume dst (and src for copy) are 32-byte aligned; n is in floats.
   The 8-element tail is handled scalar. f30/f31 are scratch — clobbered
   per-call via the asm clobber list. */
static inline void vec_zero_aligned(float * dst, 정수 n) {
    정수       i  = 0;
    const 정수 n8 = n & ~7;
    위해 (; i < n8; i += 8) {
        __asm__ volatile(
            "fsub.ps f31, f31, f31\n"
            "fsw.ps  f31, %[d]\n"
            : [d] "=m"(*(float (*)[8]) & dst[i])
            :
            : "f31");
    }
    위해 (; i < n; ++i) {
        dst[i] = 0.0f;
    }
}

static inline void vec_copy_aligned(float * dst, const float * src, 정수 n) {
    정수       i  = 0;
    const 정수 n8 = n & ~7;
    위해 (; i < n8; i += 8) {
        __asm__ volatile(
            "flw.ps f30, %[s]\n"
            "fsw.ps f30, %[d]\n"
            : [d] "=m"(*(float (*)[8]) & dst[i])
            : [s] "m"(*(const float (*)[8]) & src[i])
            : "f30");
    }
    위해 (; i < n; ++i) {
        dst[i] = src[i];
    }
}

/* ===== TenC save/restore =========================================
   The TenC accumulator IS the f0..f31 vector register file: row N occupies
   f(2N) and f(2N+1) (two 8-fp32 vector regs per row). We save by
   tensor_store-ing TILE rows × 64 bytes, and restore via 32 flw.ps after
   forcing L1D to refetch from the L2SCP backing (tensor_store bypasses L1D
   so the backing is always current). See feedback_tenc_save_restore.md. */
static inline void tenc_restore_from_scratch(uint64_t scr) {
    FENCE;
    evict_to_l2((const void *) scr, TILE, 64);
    WAIT_CACHEOPS;
    __asm__ volatile(
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
typedef struct {
    const float * in_base;  /* DRAM input base [Cin][H][W]         */
    정수           Kw;
    정수           chunk_KT; /* number of K_TILES (=16-wide) per chunk */
    정수           H, W, Hp, Wp_a;
    정수           pad_h, pad_w, s0;
    정수           minion;          /* this hart's minion id (0..31) */
    uint64_t      pin_copy_floats; /* per-_s pin plane size in floats */
    uint64_t      l2_pad_in_buf[MAX_DBL_BUFS];
    uint64_t      pin_chunk_bytes; /* one chunk pin buffer's total size */
} pin_ctx_t;

static inline 정수 find_mt_idx(const 정수 * my_mt, 정수 n_my_mt, 정수 mt) {
    위해 (정수 j = 0; j < n_my_mt; ++j) {
        만약 (my_mt[j] == mt) {
            반환 j;
        }
    }
    반환 0;
}

static inline conv_tile_t decode_tile(정수 t, 정수 M_TILES, 정수 w_tiles, const 정수 * my_mt, 정수 n_my_mt) {
    conv_tile_t tile;
    tile.mt = t % M_TILES;
    t /= M_TILES;
    const 정수 wt = t % w_tiles;
    t /= w_tiles;
    tile.oh      = t;
    tile.ow_base = wt * TILE;
    tile.mt_idx  = find_mt_idx(my_mt, n_my_mt, tile.mt);
    반환 tile;
}

static inline uint64_t
filter_slab_addr(uint64_t l2_filter, 정수 Kw, 정수 K_TILES, 정수 n_my_mt, 정수 mt_idx, 정수 kh, 정수 kw, 정수 kt_global) {
    반환 l2_filter + (uint64_t) ((((kh * Kw + kw) * n_my_mt + mt_idx) * K_TILES + kt_global)) * SLAB_BYTES;
}

static inline uint64_t pin_tile_addr(uint64_t l2_pad_in,
                                     uint64_t pin_copy_bytes,
                                     정수      ktc,
                                     정수      kw,
                                     정수      Hp,
                                     정수      Wp_a,
                                     정수      oh,
                                     정수      ow_base,
                                     정수      s1,
                                     정수      kh) {
    const 정수 ir_pad = oh * s1 + kh;
    반환 l2_pad_in + (uint64_t) kw * pin_copy_bytes +
           (((uint64_t) (ktc * TILE) * Hp + ir_pad) * Wp_a + ow_base) * sizeof(float);
}

static inline char * output_tile_addr(char *              out_base,
                                      const conv_tile_t * tile,
                                      uint64_t            out_chan_stride,
                                      uint64_t            out_row_stride) {
    반환 out_base + (size_t) (tile->mt * TILE) * out_chan_stride + (size_t) tile->oh * out_row_stride +
           (size_t) tile->ow_base * sizeof(float);
}

static inline void flush_range_to_l2(const void * addr, uint64_t n_bytes) {
    const uint64_t total_lines = (n_bytes + 63) / 64;
    const char *   fl_addr     = (const char *) addr;
    위해 (uint64_t done = 0; done < total_lines;) {
        const uint64_t batch = min_u64(total_lines - done, 16);
        flush_to_l2((const void *) (fl_addr + done * 64), batch, 64);
        done += batch;
    }
}

static inline void evict_range_past_l2(const void * addr, uint64_t n_bytes) {
    const uint64_t total_lines = (n_bytes + 63) / 64;
    const char *   fl_addr     = (const char *) addr;
    위해 (uint64_t done = 0; done < total_lines;) {
        const uint64_t batch = min_u64(total_lines - done, 16);
        evict_past_l2((const void *) (fl_addr + done * 64), batch, 64);
        done += batch;
    }
}

/* One matrix-engine tile for one Cin chunk. This is the main optimization
   surface: A is double-buffered, B is single-buffered due to L1 SCP space. */
static inline void compute_tile_chunk(uint64_t            l2_filter,
                                      uint64_t            l2_pad_in,
                                      uint64_t            pin_copy_bytes,
                                      정수                 Kh,
                                      정수                 Kw,
                                      정수                 K_TILES,
                                      정수                 chunk_KT,
                                      정수                 kt_base,
                                      정수                 n_my_mt,
                                      정수                 Hp,
                                      정수                 Wp_a,
                                      정수                 s1,
                                      uint64_t            a_row_stride,
                                      uint64_t            b_row_stride,
                                      const conv_tile_t * tile,
                                      bool                first_fma_clears_tenc) {
    const 정수      n_iters   = Kh * Kw * chunk_KT;
    const uint64_t A_BUFS[2] = { LSCP_A_0, LSCP_A_1 };

    const uint64_t a_addr0 = filter_slab_addr(l2_filter, Kw, K_TILES, n_my_mt, tile->mt_idx, 0, 0, kt_base);
    tensor_load(false, false, A_BUFS[0], 0, 0, a_addr0, 0, (uint64_t) (TILE - 1), a_row_stride, 0);

    위해 (정수 iter = 0; iter < n_iters; ++iter) {
        const 정수 ktc = iter % chunk_KT;
        const 정수 rem = iter / chunk_KT;
        const 정수 kw  = rem % Kw;
        const 정수 kh  = rem / Kw;

        const uint64_t b_addr =
            pin_tile_addr(l2_pad_in, pin_copy_bytes, ktc, kw, Hp, Wp_a, tile->oh, tile->ow_base, s1, kh);
        tensor_load(false, false, LSCP_B, 0, 0, b_addr, 0, (uint64_t) (TILE - 1), b_row_stride, 1);

        tensor_wait(TENSOR_LOAD_WAIT_0);
        tensor_wait(TENSOR_LOAD_WAIT_1);

        만약 (iter + 1 < n_iters) {
            const 정수      ktc_n = (iter + 1) % chunk_KT;
            const 정수      rem_n = (iter + 1) / chunk_KT;
            const 정수      kw_n  = rem_n % Kw;
            const 정수      kh_n  = rem_n / Kw;
            const uint64_t a_addr_n =
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
static void pack_filter_local_mt(const float * flt_base,
                                 정수           Kh,
                                 정수           Kw,
                                 정수           Cin,
                                 정수           K_TILES,
                                 const 정수 *   my_mt,
                                 정수           n_my_mt,
                                 정수           minion,
                                 uint64_t      l2_filter_base) {
    const 정수    n_slabs = Kh * Kw * n_my_mt * K_TILES;
    const size_t kstep   = (size_t) Kh * Kw; /* Cin stride in floats */

    위해 (정수 slab = minion; slab < n_slabs; slab += N_MIN_PER_SHIRE) {
        정수       t  = slab;
        const 정수 kt = t % K_TILES;
        t /= K_TILES;
        const 정수 mt_idx = t % n_my_mt;
        t /= n_my_mt;
        const 정수 kw = t % Kw;
        t /= Kw;
        const 정수 kh = t;
        const 정수 mt = my_mt[mt_idx];

        const uint64_t slab_offset = (uint64_t) slab * SLAB_BYTES;
        float *        cell        = (float *) (l2_filter_base + slab_offset);

        위해 (정수 oc_in = 0; oc_in < TILE; ++oc_in) {
            const 정수     oc  = mt * TILE + oc_in;
            const float * src = flt_base + (((size_t) oc * Cin + (size_t) kt * TILE) * Kh + kh) * Kw + kw;
            float *       row = cell + (size_t) oc_in * TILE;
            float         scratch[TILE] __attribute__((aligned(32)));
            위해 (정수 ic_in = 0; ic_in < TILE; ++ic_in) {
                scratch[ic_in] = src[(size_t) ic_in * kstep];
            }
            vec_copy_aligned(row, scratch, TILE);
        }
    }

    /* Flush this hart's dirty L1D lines for the slabs it wrote. */
    FENCE;
    위해 (정수 slab = minion; slab < n_slabs; slab += N_MIN_PER_SHIRE) {
        const uint64_t slab_offset = (uint64_t) slab * SLAB_BYTES;
        flush_to_l2((const void *) (l2_filter_base + slab_offset), SLAB_LINES, 64);
    }
    WAIT_CACHEOPS;
}

/* Pack one Cin chunk of the input pin (Kw shifted padded copies) into the
   buf_idx side of local L2SCP. Work distributed across the 32 hart-1's in
   the shire by `plane % 32 == minion`. The final flush_to_l2 forces L1D
   write-back so hart-0's tensor_load sees the freshly written bytes. */
static void pack_pin_chunk(const pin_ctx_t * ctx, 정수 chunk_id, 정수 buf_idx) {
    const 정수 kt_base  = chunk_id * ctx->chunk_KT;
    const 정수 Kw       = ctx->Kw;
    const 정수 chunk_KT = ctx->chunk_KT;
    const 정수 H = ctx->H, W = ctx->W, Hp = ctx->Hp, Wp_a = ctx->Wp_a;
    const 정수 pad_h = ctx->pad_h, pad_w = ctx->pad_w, s0 = ctx->s0;
    const 정수 minion = ctx->minion;

    /* Pin pack: Kw shifted, padded copies of input rows. Bounds [vlo, vhi)
       hoisted outside the row loop so the inner loop is three regions
       (zero-prefix | bulk-copy | zero-suffix) with no per-element predicate. */
    float *   pin0         = (float *) ctx->l2_pad_in_buf[buf_idx];
    const 정수 chunk_Cin    = chunk_KT * TILE;
    const 정수 n_pin_planes = Kw * chunk_Cin;
    위해 (정수 p = minion; p < n_pin_planes; p += N_MIN_PER_SHIRE) {
        const 정수 s     = p / chunk_Cin;
        const 정수 icc   = p % chunk_Cin;
        const 정수 ic    = kt_base * TILE + icc;
        float *   pin_s = pin0 + (size_t) s * ctx->pin_copy_floats;

        const 정수 offset = s - pad_w;
        정수       vlo    = 0;
        동안 (vlo < Wp_a && (s0 * vlo + offset) < 0) {
            vlo++;
        }
        정수 vhi = Wp_a;
        동안 (vhi > vlo && (s0 * (vhi - 1) + offset) >= W) {
            vhi--;
        }
        const bool aligned = (s0 == 1) && ((vlo & 7) == 0) && (((vlo + offset) & 7) == 0);

        위해 (정수 r = 0; r < Hp; ++r) {
            float *   row    = pin_s + ((size_t) icc * Hp + r) * Wp_a;
            const 정수 real_h = r - pad_h;
            만약 (real_h < 0 || real_h >= H) {
                vec_zero_aligned(row, Wp_a);
                continue;
            }
            const float * src_row = ctx->in_base + ((size_t) ic * H + real_h) * W;

            위해 (정수 cc = 0; cc < vlo; ++cc) {
                row[cc] = 0.0f;
            }

            만약 (aligned) {
                vec_copy_aligned(row + vlo, src_row + vlo + offset, vhi - vlo);
            } 아니면 만약 (s0 == 1) {
                const float * csrc = src_row + vlo + offset;
                const 정수     n    = vhi - vlo;
                위해 (정수 cc = 0; cc < n; ++cc) {
                    row[vlo + cc] = csrc[cc];
                }
            } 아니면 {
                위해 (정수 cc = vlo; cc < vhi; ++cc) {
                    row[cc] = src_row[s0 * cc + offset];
                }
            }

            위해 (정수 cc = vhi; cc < Wp_a; ++cc) {
                row[cc] = 0.0f;
            }
        }
    }

    /* Flush this buffer's L1D-dirty lines down to L2SCP backing. */
    FENCE;
    flush_range_to_l2((const void *) ctx->l2_pad_in_buf[buf_idx], ctx->pin_chunk_bytes);
    WAIT_CACHEOPS;
}

정수 entry_point(struct ggml_et_binary_params * params, void * env) {
    (void) env;

    const 정수 shire   = get_shire_id();
    const 정수 hart_id = get_hart_id();
    const 정수 minion  = (hart_id >> 1) & 0x1F;
    const 정수 hart1   = hart_id & 1;

    const struct ggml_tensor * flt = &params->src0; /* [Kw,Kh,Cin,Cout] */
    const struct ggml_tensor * in  = &params->src1; /* [W, H, Cin,N=1 ] */
    struct ggml_tensor *       out = &params->dst;  /* [W, H, Cout,N=1] */

    const 정수 Kw   = (정수) flt->ne[0];
    const 정수 Kh   = (정수) flt->ne[1];
    const 정수 Cin  = (정수) flt->ne[2];
    const 정수 Cout = (정수) flt->ne[3];

    const 정수 W  = (정수) in->ne[0];
    const 정수 H  = (정수) in->ne[1];
    const 정수 OW = (정수) out->ne[0];
    const 정수 OH = (정수) out->ne[1];

    /* op_params layout (set by ggml_conv_2d):
         [0]=s0  [1]=s1  [2]=p0  [3]=p1  [4]=d0  [5]=d1   */
    const 정수 s0    = out->op_params[0];
    const 정수 s1    = out->op_params[1];
    const 정수 pad_w = out->op_params[2];
    const 정수 pad_h = out->op_params[3];

    만약 (Cin <= 0 || Cout <= 0) {
        반환 -1;
    }
    만약 (Cin % TILE != 0 || Cout % TILE != 0) {
        반환 -1;
    }
    만약 (W <= 0 || H <= 0) {
        반환 -1;
    }
    만약 (s0 <= 0 || s1 <= 0) {
        반환 -1;
    }
    만약 (in->ne[2] != Cin || in->ne[3] != 1) {
        반환 -1;
    }
    만약 (out->ne[2] != Cout || out->ne[3] != 1) {
        반환 -1;
    }
    만약 (!flt->data || !in->data || !out->data) {
        반환 -1;
    }

    const 정수 K_TILES = Cin / TILE;
    const 정수 M_TILES = Cout / TILE;

    const 정수  Hp         = H + 2 * pad_h;
    const 정수  Wp_a       = round_up_tile_i32(OW);
    const 정수  OW_pad     = Wp_a;
    const bool need_stage = (OW % TILE != 0);

    /* ===================== Tile assignment & active-shire selection =====
       Computed up front because the per-shire mt set (and thus filter
       region size) depends on n_active_shires. */
    const 정수 w_tiles         = ceil_div_i32(OW, TILE);
    const 정수 total_tiles     = OH * w_tiles * M_TILES;
    const 정수 n_active_shires = need_stage ? 1 : min_i32(total_tiles, N_SHIRES);

    /* Inactive shires exit immediately. No global barrier — pack and
       barriers are now per-shire, so unused shires don't need to vote. */
    만약 (shire >= n_active_shires) {
        반환 0;
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
    정수 my_mt[MAX_MY_MT];
    정수 n_my_mt = 0;
    위해 (정수 t = shire; t < total_tiles; t += n_active_shires) {
        const 정수 mt    = t % M_TILES;
        bool      found = false;
        위해 (정수 j = 0; j < n_my_mt; ++j) {
            만약 (my_mt[j] == mt) {
                found = true;
                break;
            }
        }
        만약 (!found) {
            만약 (n_my_mt >= MAX_MY_MT) {
                반환 -1;
            }
            my_mt[n_my_mt++] = mt;
        }
    }
    만약 (n_my_mt == 0) {
        반환 0; /* no tiles for this shire */
    }

    const uint64_t filter_local_bytes = (uint64_t) Kh * Kw * n_my_mt * K_TILES * SLAB_BYTES;
    만약 (filter_local_bytes > LOCAL_FILTER_CAP) {
        반환 -1;
    }

    /* ===================== L2 SCP local layout =========================
       filter (this shire's mt slice) | pin_buf[0] | pin_buf[1]?
                                      | output_stage? | scratch (streaming) */
    const uint64_t l2_base   = (uint64_t) et_shire_l2scp_local(0);
    const uint64_t l2_filter = l2_base;

    /* Sizing for pin: budget = LOCAL_BUDGET - filter - output_stage.    */
    const int64_t output_stage_bytes_full = need_stage ? (int64_t) Cout * OH * OW_pad * (int64_t) sizeof(float) : 0;
    const int64_t budget_for_chunks = (int64_t) LOCAL_BUDGET - (int64_t) filter_local_bytes - output_stage_bytes_full;
    만약 (budget_for_chunks <= 0) {
        반환 -1;
    }
    const int64_t per_KT_pin_bytes = (int64_t) Kw * TILE * Hp * Wp_a * (int64_t) sizeof(float);

    정수 chunk_KT;
    정수 n_buffers;
    만약 ((int64_t) K_TILES * per_KT_pin_bytes <= budget_for_chunks) {
        chunk_KT  = K_TILES;
        n_buffers = 1;
    } 아니면 {
        chunk_KT = K_TILES;
        동안 (chunk_KT > 1 && 2 * (int64_t) chunk_KT * per_KT_pin_bytes > budget_for_chunks) {
            chunk_KT--;
        }
        동안 (chunk_KT > 1 && K_TILES % chunk_KT != 0) {
            chunk_KT--;
        }
        n_buffers = (chunk_KT < K_TILES) ? 2 : 1;
        만약 (chunk_KT < 1) {
            반환 -1;
        }
    }
    const 정수 n_chunks = K_TILES / chunk_KT;

    /* Streaming keeps partial sums in MAX_TILES_PER_HART scratch slots per
       hart. The one-chunk path does not need scratch and can stream a longer
       tile list serially, but multi-chunk shapes must fit this fixed slot
       count until scratch scheduling is made more general. */
    const 정수 shire_tile_capacity = shire + MAX_TILES_PER_HART * n_active_shires * N_MIN_PER_SHIRE;
    만약 (n_chunks > 1 && shire_tile_capacity < total_tiles) {
        반환 -1;
    }

    const uint64_t pin_copy_floats = (uint64_t) chunk_KT * TILE * Hp * Wp_a;
    const uint64_t pin_copy_bytes  = pin_copy_floats * sizeof(float);
    const uint64_t pin_chunk_bytes = (uint64_t) Kw * pin_copy_bytes;

    const uint64_t l2_pin_base              = l2_filter + filter_local_bytes;
    const uint64_t l2_pin_buf[MAX_DBL_BUFS] = {
        l2_pin_base,
        l2_pin_base + pin_chunk_bytes,
    };

    const uint64_t l2_output_stage = need_stage ? l2_pin_base + (uint64_t) n_buffers * pin_chunk_bytes : 0;

    const uint64_t scratch_per_hart = (uint64_t) MAX_TILES_PER_HART * (uint64_t) TILE * TILE * sizeof(float);
    const uint64_t l2_scratch_base  = need_stage ? l2_output_stage + (uint64_t) output_stage_bytes_full :
                                                   l2_pin_base + (uint64_t) n_buffers * pin_chunk_bytes;

    /* ===================== PHASE 1: Filter pack (per-shire mt slice) ====
       Hart-1's pack only this shire's mt slabs into local L2 SCP. The
       SHIRE barrier below ensures the filter is in L2 SCP backing before
       hart-0's first tensor_load. */
    만약 (hart1) {
        pack_filter_local_mt((const float *) flt->data, Kh, Kw, Cin, K_TILES, my_mt, n_my_mt, minion, l2_filter);
    }

    /* ===================== Hart 1: pin packer (per chunk) ==============
       Double-buffered prefetch: pack chunk 0 synchronously, then per chunk c
       signal "buf c ready", pack chunk c+1 into the alternate buffer
       (overlaps hart-0's compute on c), signal "buf c done". */
    만약 (hart1) {
        const pin_ctx_t ctx = {
            .in_base         = (const float *) in->data,
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

        위해 (정수 c = 0; c < n_chunks; ++c) {
            et_barrier(ET_BARRIER_SHIRE); /* signal "buf c ready" */
            만약 (n_buffers > 1 && c + 1 < n_chunks) {
                pack_pin_chunk(&ctx, c + 1, (c + 1) & 1);
            }
            et_barrier(ET_BARRIER_SHIRE); /* wait "buf c done" */
        }

        만약 (need_stage) {
            et_barrier(ET_BARRIER_SHIRE);
        }
        반환 0;
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

    char * const   out_base        = need_stage ? (char *) l2_output_stage : (char *) out->data;
    const 정수      compute_OW      = need_stage ? OW_pad : OW;
    const uint64_t out_chan_stride = (uint64_t) OH * (uint64_t) compute_OW * sizeof(float);
    const uint64_t out_row_stride  = (uint64_t) compute_OW * sizeof(float);

    const uint64_t a_row_stride = (uint64_t) TILE * sizeof(float); /* 64 */
    const uint64_t b_row_stride = (uint64_t) Hp * (uint64_t) Wp_a * sizeof(float);

    /* Tile assignment: shire-strided so small workloads spread across
       shires before stacking minions in one shire. */
    const 정수 t_start  = shire + minion * n_active_shires;
    const 정수 t_stride = n_active_shires * N_MIN_PER_SHIRE;

    만약 (n_chunks == 1) {
        et_barrier(ET_BARRIER_SHIRE); /* wait for the (only) pin chunk */

        const uint64_t l2_pad_in = l2_pin_buf[0];
        위해 (정수 t = t_start; t < total_tiles; t += t_stride) {
            const conv_tile_t tile = decode_tile(t, M_TILES, w_tiles, my_mt, n_my_mt);
            compute_tile_chunk(l2_filter, l2_pad_in, pin_copy_bytes, Kh, Kw, K_TILES, chunk_KT, 0, n_my_mt, Hp, Wp_a,
                               s1, a_row_stride, b_row_stride, &tile, /*first_fma_clears_tenc=*/true);

            char * dst_addr = output_tile_addr(out_base, &tile, out_chan_stride, out_row_stride);
            tensor_store(0, 0, 3, (uint64_t) (TILE - 1), (uint64_t) dst_addr, 0, out_chan_stride);
            tensor_wait(TENSOR_STORE_WAIT);
        }

        et_barrier(ET_BARRIER_SHIRE); /* matches hart-1's second barrier */

    } 아니면 {
        /* Streaming path: each hart owns up to MAX_TILES_PER_HART tiles. */
        정수 my_tiles[MAX_TILES_PER_HART];
        정수 n_my_tiles = 0;
        위해 (정수 slot = 0; slot < MAX_TILES_PER_HART; ++slot) {
            const 정수 t = t_start + slot * t_stride;
            만약 (t < total_tiles) {
                my_tiles[n_my_tiles++] = t;
            }
        }

        conv_tile_t tiles[MAX_TILES_PER_HART];
        위해 (정수 k = 0; k < n_my_tiles; ++k) {
            tiles[k] = decode_tile(my_tiles[k], M_TILES, w_tiles, my_mt, n_my_mt);
        }

        const uint64_t my_scratch_base = l2_scratch_base + (uint64_t) minion * scratch_per_hart;

        위해 (정수 c = 0; c < n_chunks; ++c) {
            et_barrier(ET_BARRIER_SHIRE); /* pin chunk c packed */

            const 정수      buf       = c & 1;
            const uint64_t l2_pad_in = l2_pin_buf[buf];
            const 정수      kt_base   = c * chunk_KT;

            위해 (정수 k = 0; k < n_my_tiles; ++k) {
                const conv_tile_t * tile = &tiles[k];
                const uint64_t      scr  = my_scratch_base + (uint64_t) k * (TILE * TILE * sizeof(float));

                const bool first_pass_chunk = (c == 0);
                만약 (!first_pass_chunk) {
                    tenc_restore_from_scratch(scr);
                }

                compute_tile_chunk(l2_filter, l2_pad_in, pin_copy_bytes, Kh, Kw, K_TILES, chunk_KT, kt_base, n_my_mt,
                                   Hp, Wp_a, s1, a_row_stride, b_row_stride, tile, first_pass_chunk);

                만약 (c == n_chunks - 1) {
                    char * dst_addr = output_tile_addr(out_base, tile, out_chan_stride, out_row_stride);
                    tensor_store(0, 0, 3, (uint64_t) (TILE - 1), (uint64_t) dst_addr, 0, out_chan_stride);
                } 아니면 {
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
    만약 (need_stage) {
        et_barrier(ET_BARRIER_SHIRE);

        만약 (minion == 0) {
            const float * stage = (const float *) l2_output_stage;
            float *       dram  = (float *) out->data;
            위해 (정수 oc = 0; oc < Cout; ++oc) {
                위해 (정수 oh2 = 0; oh2 < OH; ++oh2) {
                    const float * src = stage + ((size_t) oc * OH + oh2) * OW_pad;
                    float *       dst = dram + ((size_t) oc * OH + oh2) * OW;
                    위해 (정수 ow2 = 0; ow2 < OW; ++ow2) {
                        dst[ow2] = src[ow2];
                    }
                }
            }
            FENCE;
            const uint64_t total_bytes = (uint64_t) Cout * OH * OW * sizeof(float);
            evict_range_past_l2((const void *) dram, total_bytes);
            WAIT_CACHEOPS;
        }
    }

    반환 0;
}
