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

#समावेश "ggml_tensor.h"
#समावेश "platform.h"
#समावेश "tensor.h"

#समावेश <etsoc/common/utils.h>
#समावेश <stdbool.h>
#समावेश <stdint.h>

#घोषणा TILE               16 /* matrix engine native tile in M, K, N */
/* L1 SCP layout: A double-buffered, B single-buffered. Per the SDK doc
   `dst_start` is a 6-bit field (max 63) but empirical testing shows the
   physical L1 SCP per minion is 48 lines — writes to lines >= 48 corrupt.
   So we get 3 × 16-line buffers max: A_0, A_1, B. Pick A as the
   double-buffered operand (filter-slab loads, the longer of the two). */
#घोषणा LSCP_A_0           0  /* A buffer 0 at L1 SCP lines 0..15  */
#घोषणा LSCP_A_1           16 /* A buffer 1 at L1 SCP lines 16..31 */
#घोषणा LSCP_B             32 /* B (single buffer) at lines 32..47 */
#घोषणा N_MIN_PER_SHIRE    32 /* ET-SoC-1 geometry: 32 minions/shire  */
#घोषणा N_SHIRES           32 /* default active shire count           */
#घोषणा MAX_TILES_PER_HART 2  /* per-hart TenC slots (save/restore)   */
#घोषणा MAX_DBL_BUFS       2  /* chunk pack buffers (double-buffered) */

/* Per-shire L2 SCP local budget. Per-shire SCP is 2 MB; we cap at
   1984 KB to leave 64 KB headroom for per-hart TenC scratch (32 minions ×
   2 slots × 1 KB), which lives at the tail of the SCP outside the pin
   sizing budget. Bigger budget here means bigger feasible chunk_KT,
   which means fewer chunks (each chunk costs 2 SHIRE barriers + ~30
   TenC save/restore events per hart). */
#घोषणा LOCAL_BUDGET (1984 * 1024)

/* Cap on the per-shire filter region in local L2 SCP. The shire packs the
   mt values it can consume under the current tile assignment, rather than
   the whole Cout dimension. Reads in the inner loop are then fully
   shire-local — no NoC fanout. */
#घोषणा LOCAL_FILTER_CAP (1024 * 1024)                      /* 1 MB / shire ceiling */

#घोषणा SLAB_BYTES ((uint64_t) TILE * TILE * माप(भग्न)) /* 1024 */
#घोषणा SLAB_पंक्तियाँ ((SLAB_BYTES + 63) / 64)                 /* 16   */

/* Upper bound on the number of distinct mt values a single shire may pack.
   This keeps the mt list stack-resident. Shapes that need more should fall
   back until the filter-slice bookkeeping is made dynamic. */
#घोषणा MAX_MY_MT (N_MIN_PER_SHIRE * MAX_TILES_PER_HART)

प्रकार काष्ठा {
    पूर्णांक mt;
    पूर्णांक mt_idx;
    पूर्णांक oh;
    पूर्णांक ow_base;
} conv_tile_t;

अटल अंतरभूत पूर्णांक ceil_div_i32(पूर्णांक x, पूर्णांक y) {
    वापस (x + y - 1) / y;
}

अटल अंतरभूत पूर्णांक round_up_tile_i32(पूर्णांक x) {
    वापस (x + TILE - 1) & ~(TILE - 1);
}

अटल अंतरभूत पूर्णांक min_i32(पूर्णांक a, पूर्णांक b) {
    वापस a < b ? a : b;
}

अटल अंतरभूत uint64_t min_u64(uint64_t a, uint64_t b) {
    वापस a < b ? a : b;
}

/* ===== Vector helpers for hart-1 pack ============================
   Both assume dst (and src for copy) are 32-byte aligned; n is in floats.
   The 8-element tail is handled scalar. f30/f31 are scratch — clobbered
   per-call via the asm clobber list. */
अटल अंतरभूत व्योम vec_zero_aligned(भग्न * dst, पूर्णांक n) {
    पूर्णांक       i  = 0;
    स्थिर पूर्णांक n8 = n & ~7;
    क्रम (; i < n8; i += 8) {
        __asm__ अस्थिर(
            "fsub.ps f31, f31, f31\n"
            "fsw.ps  f31, %[d]\n"
            : [d] "=m"(*(भग्न (*)[8]) & dst[i])
            :
            : "f31");
    }
    क्रम (; i < n; ++i) {
        dst[i] = 0.0f;
    }
}

अटल अंतरभूत व्योम vec_copy_aligned(भग्न * dst, स्थिर भग्न * src, पूर्णांक n) {
    पूर्णांक       i  = 0;
    स्थिर पूर्णांक n8 = n & ~7;
    क्रम (; i < n8; i += 8) {
        __asm__ अस्थिर(
            "flw.ps f30, %[s]\n"
            "fsw.ps f30, %[d]\n"
            : [d] "=m"(*(भग्न (*)[8]) & dst[i])
            : [s] "m"(*(स्थिर भग्न (*)[8]) & src[i])
            : "f30");
    }
    क्रम (; i < n; ++i) {
        dst[i] = src[i];
    }
}

/* ===== TenC save/restore =========================================
   The TenC accumulator IS the f0..f31 vector register file: row N occupies
   f(2N) and f(2N+1) (two 8-fp32 vector regs per row). We save by
   tensor_store-ing TILE rows × 64 bytes, and restore via 32 flw.ps after
   forcing L1D to refetch from the L2SCP backing (tensor_store bypasses L1D
   so the backing is always current). See feedback_tenc_save_restore.md. */
अटल अंतरभूत व्योम tenc_restore_from_scratch(uint64_t scr) {
    FENCE;
    evict_to_l2((स्थिर व्योम *) scr, TILE, 64);
    WAIT_CACHEOPS;
    __asm__ अस्थिर(
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
प्रकार काष्ठा {
    स्थिर भग्न * in_base;  /* DRAM input base [Cin][H][W]         */
    पूर्णांक           Kw;
    पूर्णांक           chunk_KT; /* number of K_TILES (=16-wide) per chunk */
    पूर्णांक           H, W, Hp, Wp_a;
    पूर्णांक           pad_h, pad_w, s0;
    पूर्णांक           minion;          /* this hart's minion id (0..31) */
    uint64_t      pin_copy_floats; /* per-_s pin plane size in floats */
    uint64_t      l2_pad_in_buf[MAX_DBL_BUFS];
    uint64_t      pin_chunk_bytes; /* one chunk pin buffer's total size */
} pin_ctx_t;

अटल अंतरभूत पूर्णांक find_mt_idx(स्थिर पूर्णांक * my_mt, पूर्णांक n_my_mt, पूर्णांक mt) {
    क्रम (पूर्णांक j = 0; j < n_my_mt; ++j) {
        अगर (my_mt[j] == mt) {
            वापस j;
        }
    }
    वापस 0;
}

अटल अंतरभूत conv_tile_t decode_tile(पूर्णांक t, पूर्णांक M_TILES, पूर्णांक w_tiles, स्थिर पूर्णांक * my_mt, पूर्णांक n_my_mt) {
    conv_tile_t tile;
    tile.mt = t % M_TILES;
    t /= M_TILES;
    स्थिर पूर्णांक wt = t % w_tiles;
    t /= w_tiles;
    tile.oh      = t;
    tile.ow_base = wt * TILE;
    tile.mt_idx  = find_mt_idx(my_mt, n_my_mt, tile.mt);
    वापस tile;
}

अटल अंतरभूत uint64_t
filter_slab_addr(uint64_t l2_filter, पूर्णांक Kw, पूर्णांक K_TILES, पूर्णांक n_my_mt, पूर्णांक mt_idx, पूर्णांक kh, पूर्णांक kw, पूर्णांक kt_global) {
    वापस l2_filter + (uint64_t) ((((kh * Kw + kw) * n_my_mt + mt_idx) * K_TILES + kt_global)) * SLAB_BYTES;
}

अटल अंतरभूत uint64_t pin_tile_addr(uint64_t l2_pad_in,
                                     uint64_t pin_copy_bytes,
                                     पूर्णांक      ktc,
                                     पूर्णांक      kw,
                                     पूर्णांक      Hp,
                                     पूर्णांक      Wp_a,
                                     पूर्णांक      oh,
                                     पूर्णांक      ow_base,
                                     पूर्णांक      s1,
                                     पूर्णांक      kh) {
    स्थिर पूर्णांक ir_pad = oh * s1 + kh;
    वापस l2_pad_in + (uint64_t) kw * pin_copy_bytes +
           (((uint64_t) (ktc * TILE) * Hp + ir_pad) * Wp_a + ow_base) * माप(भग्न);
}

अटल अंतरभूत अक्षर * output_tile_addr(अक्षर *              out_base,
                                      स्थिर conv_tile_t * tile,
                                      uint64_t            out_chan_stride,
                                      uint64_t            out_row_stride) {
    वापस out_base + (माप_प्रकार) (tile->mt * TILE) * out_chan_stride + (माप_प्रकार) tile->oh * out_row_stride +
           (माप_प्रकार) tile->ow_base * माप(भग्न);
}

अटल अंतरभूत व्योम flush_range_to_l2(स्थिर व्योम * addr, uint64_t n_bytes) {
    स्थिर uint64_t योग_पंक्तियाँ = (n_bytes + 63) / 64;
    स्थिर अक्षर *   fl_addr     = (स्थिर अक्षर *) addr;
    क्रम (uint64_t done = 0; done < योग_पंक्तियाँ;) {
        स्थिर uint64_t batch = min_u64(योग_पंक्तियाँ - done, 16);
        flush_to_l2((स्थिर व्योम *) (fl_addr + done * 64), batch, 64);
        done += batch;
    }
}

अटल अंतरभूत व्योम evict_range_past_l2(स्थिर व्योम * addr, uint64_t n_bytes) {
    स्थिर uint64_t योग_पंक्तियाँ = (n_bytes + 63) / 64;
    स्थिर अक्षर *   fl_addr     = (स्थिर अक्षर *) addr;
    क्रम (uint64_t done = 0; done < योग_पंक्तियाँ;) {
        स्थिर uint64_t batch = min_u64(योग_पंक्तियाँ - done, 16);
        evict_past_l2((स्थिर व्योम *) (fl_addr + done * 64), batch, 64);
        done += batch;
    }
}

/* One matrix-engine tile for one Cin chunk. This is the main optimization
   surface: A is double-buffered, B is single-buffered due to L1 SCP space. */
अटल अंतरभूत व्योम compute_tile_chunk(uint64_t            l2_filter,
                                      uint64_t            l2_pad_in,
                                      uint64_t            pin_copy_bytes,
                                      पूर्णांक                 Kh,
                                      पूर्णांक                 Kw,
                                      पूर्णांक                 K_TILES,
                                      पूर्णांक                 chunk_KT,
                                      पूर्णांक                 kt_base,
                                      पूर्णांक                 n_my_mt,
                                      पूर्णांक                 Hp,
                                      पूर्णांक                 Wp_a,
                                      पूर्णांक                 s1,
                                      uint64_t            a_row_stride,
                                      uint64_t            b_row_stride,
                                      स्थिर conv_tile_t * tile,
                                      bool                first_fma_clears_tenc) {
    स्थिर पूर्णांक      n_iters   = Kh * Kw * chunk_KT;
    स्थिर uint64_t A_BUFS[2] = { LSCP_A_0, LSCP_A_1 };

    स्थिर uint64_t a_addr0 = filter_slab_addr(l2_filter, Kw, K_TILES, n_my_mt, tile->mt_idx, 0, 0, kt_base);
    tensor_load(false, false, A_BUFS[0], 0, 0, a_addr0, 0, (uint64_t) (TILE - 1), a_row_stride, 0);

    क्रम (पूर्णांक iter = 0; iter < n_iters; ++iter) {
        स्थिर पूर्णांक ktc = iter % chunk_KT;
        स्थिर पूर्णांक rem = iter / chunk_KT;
        स्थिर पूर्णांक kw  = rem % Kw;
        स्थिर पूर्णांक kh  = rem / Kw;

        स्थिर uint64_t b_addr =
            pin_tile_addr(l2_pad_in, pin_copy_bytes, ktc, kw, Hp, Wp_a, tile->oh, tile->ow_base, s1, kh);
        tensor_load(false, false, LSCP_B, 0, 0, b_addr, 0, (uint64_t) (TILE - 1), b_row_stride, 1);

        tensor_wait(TENSOR_LOAD_WAIT_0);
        tensor_wait(TENSOR_LOAD_WAIT_1);

        अगर (iter + 1 < n_iters) {
            स्थिर पूर्णांक      ktc_n = (iter + 1) % chunk_KT;
            स्थिर पूर्णांक      rem_n = (iter + 1) / chunk_KT;
            स्थिर पूर्णांक      kw_n  = rem_n % Kw;
            स्थिर पूर्णांक      kh_n  = rem_n / Kw;
            स्थिर uint64_t a_addr_n =
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
अटल व्योम pack_filter_local_mt(स्थिर भग्न * flt_base,
                                 पूर्णांक           Kh,
                                 पूर्णांक           Kw,
                                 पूर्णांक           Cin,
                                 पूर्णांक           K_TILES,
                                 स्थिर पूर्णांक *   my_mt,
                                 पूर्णांक           n_my_mt,
                                 पूर्णांक           minion,
                                 uint64_t      l2_filter_base) {
    स्थिर पूर्णांक    n_slabs = Kh * Kw * n_my_mt * K_TILES;
    स्थिर माप_प्रकार kstep   = (माप_प्रकार) Kh * Kw; /* Cin stride in floats */

    क्रम (पूर्णांक slab = minion; slab < n_slabs; slab += N_MIN_PER_SHIRE) {
        पूर्णांक       t  = slab;
        स्थिर पूर्णांक kt = t % K_TILES;
        t /= K_TILES;
        स्थिर पूर्णांक mt_idx = t % n_my_mt;
        t /= n_my_mt;
        स्थिर पूर्णांक kw = t % Kw;
        t /= Kw;
        स्थिर पूर्णांक kh = t;
        स्थिर पूर्णांक mt = my_mt[mt_idx];

        स्थिर uint64_t slab_offset = (uint64_t) slab * SLAB_BYTES;
        भग्न *        cell        = (भग्न *) (l2_filter_base + slab_offset);

        क्रम (पूर्णांक oc_in = 0; oc_in < TILE; ++oc_in) {
            स्थिर पूर्णांक     oc  = mt * TILE + oc_in;
            स्थिर भग्न * src = flt_base + (((माप_प्रकार) oc * Cin + (माप_प्रकार) kt * TILE) * Kh + kh) * Kw + kw;
            भग्न *       row = cell + (माप_प्रकार) oc_in * TILE;
            भग्न         scratch[TILE] __attribute__((aligned(32)));
            क्रम (पूर्णांक ic_in = 0; ic_in < TILE; ++ic_in) {
                scratch[ic_in] = src[(माप_प्रकार) ic_in * kstep];
            }
            vec_copy_aligned(row, scratch, TILE);
        }
    }

    /* Flush this hart's dirty L1D lines for the slabs it wrote. */
    FENCE;
    क्रम (पूर्णांक slab = minion; slab < n_slabs; slab += N_MIN_PER_SHIRE) {
        स्थिर uint64_t slab_offset = (uint64_t) slab * SLAB_BYTES;
        flush_to_l2((स्थिर व्योम *) (l2_filter_base + slab_offset), SLAB_पंक्तियाँ, 64);
    }
    WAIT_CACHEOPS;
}

/* Pack one Cin chunk of the input pin (Kw shifted padded copies) into the
   buf_idx side of local L2SCP. Work distributed across the 32 hart-1's in
   the shire by `plane % 32 == minion`. The final flush_to_l2 forces L1D
   write-back so hart-0's tensor_load sees the freshly written bytes. */
अटल व्योम pack_pin_chunk(स्थिर pin_ctx_t * ctx, पूर्णांक chunk_id, पूर्णांक buf_idx) {
    स्थिर पूर्णांक kt_base  = chunk_id * ctx->chunk_KT;
    स्थिर पूर्णांक Kw       = ctx->Kw;
    स्थिर पूर्णांक chunk_KT = ctx->chunk_KT;
    स्थिर पूर्णांक H = ctx->H, W = ctx->W, Hp = ctx->Hp, Wp_a = ctx->Wp_a;
    स्थिर पूर्णांक pad_h = ctx->pad_h, pad_w = ctx->pad_w, s0 = ctx->s0;
    स्थिर पूर्णांक minion = ctx->minion;

    /* Pin pack: Kw shifted, padded copies of input rows. Bounds [vlo, vhi)
       hoisted outside the row loop so the inner loop is three regions
       (zero-prefix | bulk-copy | zero-suffix) with no per-element predicate. */
    भग्न *   pin0         = (भग्न *) ctx->l2_pad_in_buf[buf_idx];
    स्थिर पूर्णांक chunk_Cin    = chunk_KT * TILE;
    स्थिर पूर्णांक n_pin_planes = Kw * chunk_Cin;
    क्रम (पूर्णांक p = minion; p < n_pin_planes; p += N_MIN_PER_SHIRE) {
        स्थिर पूर्णांक s     = p / chunk_Cin;
        स्थिर पूर्णांक icc   = p % chunk_Cin;
        स्थिर पूर्णांक ic    = kt_base * TILE + icc;
        भग्न *   pin_s = pin0 + (माप_प्रकार) s * ctx->pin_copy_floats;

        स्थिर पूर्णांक offset = s - pad_w;
        पूर्णांक       vlo    = 0;
        जबतक (vlo < Wp_a && (s0 * vlo + offset) < 0) {
            vlo++;
        }
        पूर्णांक vhi = Wp_a;
        जबतक (vhi > vlo && (s0 * (vhi - 1) + offset) >= W) {
            vhi--;
        }
        स्थिर bool aligned = (s0 == 1) && ((vlo & 7) == 0) && (((vlo + offset) & 7) == 0);

        क्रम (पूर्णांक r = 0; r < Hp; ++r) {
            भग्न *   row    = pin_s + ((माप_प्रकार) icc * Hp + r) * Wp_a;
            स्थिर पूर्णांक real_h = r - pad_h;
            अगर (real_h < 0 || real_h >= H) {
                vec_zero_aligned(row, Wp_a);
                जारी;
            }
            स्थिर भग्न * src_row = ctx->in_base + ((माप_प्रकार) ic * H + real_h) * W;

            क्रम (पूर्णांक cc = 0; cc < vlo; ++cc) {
                row[cc] = 0.0f;
            }

            अगर (aligned) {
                vec_copy_aligned(row + vlo, src_row + vlo + offset, vhi - vlo);
            } अन्यथा अगर (s0 == 1) {
                स्थिर भग्न * csrc = src_row + vlo + offset;
                स्थिर पूर्णांक     n    = vhi - vlo;
                क्रम (पूर्णांक cc = 0; cc < n; ++cc) {
                    row[vlo + cc] = csrc[cc];
                }
            } अन्यथा {
                क्रम (पूर्णांक cc = vlo; cc < vhi; ++cc) {
                    row[cc] = src_row[s0 * cc + offset];
                }
            }

            क्रम (पूर्णांक cc = vhi; cc < Wp_a; ++cc) {
                row[cc] = 0.0f;
            }
        }
    }

    /* Flush this buffer's L1D-dirty lines down to L2SCP backing. */
    FENCE;
    flush_range_to_l2((स्थिर व्योम *) ctx->l2_pad_in_buf[buf_idx], ctx->pin_chunk_bytes);
    WAIT_CACHEOPS;
}

पूर्णांक entry_point(काष्ठा ggml_et_द्विआधारी_params * params, व्योम * env) {
    (व्योम) env;

    स्थिर पूर्णांक shire   = get_shire_id();
    स्थिर पूर्णांक hart_id = get_hart_id();
    स्थिर पूर्णांक minion  = (hart_id >> 1) & 0x1F;
    स्थिर पूर्णांक hart1   = hart_id & 1;

    स्थिर काष्ठा ggml_tensor * flt = &params->src0; /* [Kw,Kh,Cin,Cout] */
    स्थिर काष्ठा ggml_tensor * in  = &params->src1; /* [W, H, Cin,N=1 ] */
    काष्ठा ggml_tensor *       out = &params->dst;  /* [W, H, Cout,N=1] */

    स्थिर पूर्णांक Kw   = (पूर्णांक) flt->ne[0];
    स्थिर पूर्णांक Kh   = (पूर्णांक) flt->ne[1];
    स्थिर पूर्णांक Cin  = (पूर्णांक) flt->ne[2];
    स्थिर पूर्णांक Cout = (पूर्णांक) flt->ne[3];

    स्थिर पूर्णांक W  = (पूर्णांक) in->ne[0];
    स्थिर पूर्णांक H  = (पूर्णांक) in->ne[1];
    स्थिर पूर्णांक OW = (पूर्णांक) out->ne[0];
    स्थिर पूर्णांक OH = (पूर्णांक) out->ne[1];

    /* op_params layout (set by ggml_conv_2d):
         [0]=s0  [1]=s1  [2]=p0  [3]=p1  [4]=d0  [5]=d1   */
    स्थिर पूर्णांक s0    = out->op_params[0];
    स्थिर पूर्णांक s1    = out->op_params[1];
    स्थिर पूर्णांक pad_w = out->op_params[2];
    स्थिर पूर्णांक pad_h = out->op_params[3];

    अगर (Cin <= 0 || Cout <= 0) {
        वापस -1;
    }
    अगर (Cin % TILE != 0 || Cout % TILE != 0) {
        वापस -1;
    }
    अगर (W <= 0 || H <= 0) {
        वापस -1;
    }
    अगर (s0 <= 0 || s1 <= 0) {
        वापस -1;
    }
    अगर (in->ne[2] != Cin || in->ne[3] != 1) {
        वापस -1;
    }
    अगर (out->ne[2] != Cout || out->ne[3] != 1) {
        वापस -1;
    }
    अगर (!flt->तथ्य || !in->तथ्य || !out->तथ्य) {
        वापस -1;
    }

    स्थिर पूर्णांक K_TILES = Cin / TILE;
    स्थिर पूर्णांक M_TILES = Cout / TILE;

    स्थिर पूर्णांक  Hp         = H + 2 * pad_h;
    स्थिर पूर्णांक  Wp_a       = round_up_tile_i32(OW);
    स्थिर पूर्णांक  OW_pad     = Wp_a;
    स्थिर bool need_stage = (OW % TILE != 0);

    /* ===================== Tile assignment & active-shire selection =====
       Computed up front because the per-shire mt set (and thus filter
       region size) depends on n_active_shires. */
    स्थिर पूर्णांक w_tiles         = ceil_div_i32(OW, TILE);
    स्थिर पूर्णांक योग_tiles     = OH * w_tiles * M_TILES;
    स्थिर पूर्णांक n_active_shires = need_stage ? 1 : min_i32(योग_tiles, N_SHIRES);

    /* Inactive shires exit immediately. No global barrier — pack and
       barriers are now per-shire, so unused shires don't need to vote. */
    अगर (shire >= n_active_shires) {
        वापस 0;
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
    पूर्णांक my_mt[MAX_MY_MT];
    पूर्णांक n_my_mt = 0;
    क्रम (पूर्णांक t = shire; t < योग_tiles; t += n_active_shires) {
        स्थिर पूर्णांक mt    = t % M_TILES;
        bool      found = false;
        क्रम (पूर्णांक j = 0; j < n_my_mt; ++j) {
            अगर (my_mt[j] == mt) {
                found = true;
                अवरोध;
            }
        }
        अगर (!found) {
            अगर (n_my_mt >= MAX_MY_MT) {
                वापस -1;
            }
            my_mt[n_my_mt++] = mt;
        }
    }
    अगर (n_my_mt == 0) {
        वापस 0; /* no tiles for this shire */
    }

    स्थिर uint64_t filter_local_bytes = (uint64_t) Kh * Kw * n_my_mt * K_TILES * SLAB_BYTES;
    अगर (filter_local_bytes > LOCAL_FILTER_CAP) {
        वापस -1;
    }

    /* ===================== L2 SCP local layout =========================
       filter (this shire's mt slice) | pin_buf[0] | pin_buf[1]?
                                      | output_stage? | scratch (streaming) */
    स्थिर uint64_t l2_base   = (uint64_t) et_shire_l2scp_local(0);
    स्थिर uint64_t l2_filter = l2_base;

    /* Sizing for pin: budget = LOCAL_BUDGET - filter - output_stage.    */
    स्थिर int64_t output_stage_bytes_full = need_stage ? (int64_t) Cout * OH * OW_pad * (int64_t) माप(भग्न) : 0;
    स्थिर int64_t budget_for_chunks = (int64_t) LOCAL_BUDGET - (int64_t) filter_local_bytes - output_stage_bytes_full;
    अगर (budget_for_chunks <= 0) {
        वापस -1;
    }
    स्थिर int64_t per_KT_pin_bytes = (int64_t) Kw * TILE * Hp * Wp_a * (int64_t) माप(भग्न);

    पूर्णांक chunk_KT;
    पूर्णांक n_buffers;
    अगर ((int64_t) K_TILES * per_KT_pin_bytes <= budget_for_chunks) {
        chunk_KT  = K_TILES;
        n_buffers = 1;
    } अन्यथा {
        chunk_KT = K_TILES;
        जबतक (chunk_KT > 1 && 2 * (int64_t) chunk_KT * per_KT_pin_bytes > budget_for_chunks) {
            chunk_KT--;
        }
        जबतक (chunk_KT > 1 && K_TILES % chunk_KT != 0) {
            chunk_KT--;
        }
        n_buffers = (chunk_KT < K_TILES) ? 2 : 1;
        अगर (chunk_KT < 1) {
            वापस -1;
        }
    }
    स्थिर पूर्णांक n_chunks = K_TILES / chunk_KT;

    /* Streaming keeps partial sums in MAX_TILES_PER_HART scratch slots per
       hart. The one-chunk path does not need scratch and can stream a longer
       tile list serially, but multi-chunk shapes must fit this fixed slot
       count until scratch scheduling is made more general. */
    स्थिर पूर्णांक shire_tile_capacity = shire + MAX_TILES_PER_HART * n_active_shires * N_MIN_PER_SHIRE;
    अगर (n_chunks > 1 && shire_tile_capacity < योग_tiles) {
        वापस -1;
    }

    स्थिर uint64_t pin_copy_floats = (uint64_t) chunk_KT * TILE * Hp * Wp_a;
    स्थिर uint64_t pin_copy_bytes  = pin_copy_floats * माप(भग्न);
    स्थिर uint64_t pin_chunk_bytes = (uint64_t) Kw * pin_copy_bytes;

    स्थिर uint64_t l2_pin_base              = l2_filter + filter_local_bytes;
    स्थिर uint64_t l2_pin_buf[MAX_DBL_BUFS] = {
        l2_pin_base,
        l2_pin_base + pin_chunk_bytes,
    };

    स्थिर uint64_t l2_output_stage = need_stage ? l2_pin_base + (uint64_t) n_buffers * pin_chunk_bytes : 0;

    स्थिर uint64_t scratch_per_hart = (uint64_t) MAX_TILES_PER_HART * (uint64_t) TILE * TILE * माप(भग्न);
    स्थिर uint64_t l2_scratch_base  = need_stage ? l2_output_stage + (uint64_t) output_stage_bytes_full :
                                                   l2_pin_base + (uint64_t) n_buffers * pin_chunk_bytes;

    /* ===================== PHASE 1: Filter pack (per-shire mt slice) ====
       Hart-1's pack only this shire's mt slabs into local L2 SCP. The
       SHIRE barrier below ensures the filter is in L2 SCP backing before
       hart-0's first tensor_load. */
    अगर (hart1) {
        pack_filter_local_mt((स्थिर भग्न *) flt->तथ्य, Kh, Kw, Cin, K_TILES, my_mt, n_my_mt, minion, l2_filter);
    }

    /* ===================== Hart 1: pin packer (per chunk) ==============
       Double-buffered prefetch: pack chunk 0 synchronously, then per chunk c
       signal "buf c ready", pack chunk c+1 into the alternate buffer
       (overlaps hart-0's compute on c), signal "buf c done". */
    अगर (hart1) {
        स्थिर pin_ctx_t ctx = {
            .in_base         = (स्थिर भग्न *) in->तथ्य,
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

        क्रम (पूर्णांक c = 0; c < n_chunks; ++c) {
            et_barrier(ET_BARRIER_SHIRE); /* signal "buf c ready" */
            अगर (n_buffers > 1 && c + 1 < n_chunks) {
                pack_pin_chunk(&ctx, c + 1, (c + 1) & 1);
            }
            et_barrier(ET_BARRIER_SHIRE); /* wait "buf c done" */
        }

        अगर (need_stage) {
            et_barrier(ET_BARRIER_SHIRE);
        }
        वापस 0;
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

    अक्षर * स्थिर   out_base        = need_stage ? (अक्षर *) l2_output_stage : (अक्षर *) out->तथ्य;
    स्थिर पूर्णांक      compute_OW      = need_stage ? OW_pad : OW;
    स्थिर uint64_t out_chan_stride = (uint64_t) OH * (uint64_t) compute_OW * माप(भग्न);
    स्थिर uint64_t out_row_stride  = (uint64_t) compute_OW * माप(भग्न);

    स्थिर uint64_t a_row_stride = (uint64_t) TILE * माप(भग्न); /* 64 */
    स्थिर uint64_t b_row_stride = (uint64_t) Hp * (uint64_t) Wp_a * माप(भग्न);

    /* Tile assignment: shire-strided so small workloads spread across
       shires before stacking minions in one shire. */
    स्थिर पूर्णांक t_start  = shire + minion * n_active_shires;
    स्थिर पूर्णांक t_stride = n_active_shires * N_MIN_PER_SHIRE;

    अगर (n_chunks == 1) {
        et_barrier(ET_BARRIER_SHIRE); /* wait for the (only) pin chunk */

        स्थिर uint64_t l2_pad_in = l2_pin_buf[0];
        क्रम (पूर्णांक t = t_start; t < योग_tiles; t += t_stride) {
            स्थिर conv_tile_t tile = decode_tile(t, M_TILES, w_tiles, my_mt, n_my_mt);
            compute_tile_chunk(l2_filter, l2_pad_in, pin_copy_bytes, Kh, Kw, K_TILES, chunk_KT, 0, n_my_mt, Hp, Wp_a,
                               s1, a_row_stride, b_row_stride, &tile, /*first_fma_clears_tenc=*/true);

            अक्षर * dst_addr = output_tile_addr(out_base, &tile, out_chan_stride, out_row_stride);
            tensor_store(0, 0, 3, (uint64_t) (TILE - 1), (uint64_t) dst_addr, 0, out_chan_stride);
            tensor_wait(TENSOR_STORE_WAIT);
        }

        et_barrier(ET_BARRIER_SHIRE); /* matches hart-1's second barrier */

    } अन्यथा {
        /* Streaming path: each hart owns up to MAX_TILES_PER_HART tiles. */
        पूर्णांक my_tiles[MAX_TILES_PER_HART];
        पूर्णांक n_my_tiles = 0;
        क्रम (पूर्णांक slot = 0; slot < MAX_TILES_PER_HART; ++slot) {
            स्थिर पूर्णांक t = t_start + slot * t_stride;
            अगर (t < योग_tiles) {
                my_tiles[n_my_tiles++] = t;
            }
        }

        conv_tile_t tiles[MAX_TILES_PER_HART];
        क्रम (पूर्णांक k = 0; k < n_my_tiles; ++k) {
            tiles[k] = decode_tile(my_tiles[k], M_TILES, w_tiles, my_mt, n_my_mt);
        }

        स्थिर uint64_t my_scratch_base = l2_scratch_base + (uint64_t) minion * scratch_per_hart;

        क्रम (पूर्णांक c = 0; c < n_chunks; ++c) {
            et_barrier(ET_BARRIER_SHIRE); /* pin chunk c packed */

            स्थिर पूर्णांक      buf       = c & 1;
            स्थिर uint64_t l2_pad_in = l2_pin_buf[buf];
            स्थिर पूर्णांक      kt_base   = c * chunk_KT;

            क्रम (पूर्णांक k = 0; k < n_my_tiles; ++k) {
                स्थिर conv_tile_t * tile = &tiles[k];
                स्थिर uint64_t      scr  = my_scratch_base + (uint64_t) k * (TILE * TILE * माप(भग्न));

                स्थिर bool first_pass_chunk = (c == 0);
                अगर (!first_pass_chunk) {
                    tenc_restore_from_scratch(scr);
                }

                compute_tile_chunk(l2_filter, l2_pad_in, pin_copy_bytes, Kh, Kw, K_TILES, chunk_KT, kt_base, n_my_mt,
                                   Hp, Wp_a, s1, a_row_stride, b_row_stride, tile, first_pass_chunk);

                अगर (c == n_chunks - 1) {
                    अक्षर * dst_addr = output_tile_addr(out_base, tile, out_chan_stride, out_row_stride);
                    tensor_store(0, 0, 3, (uint64_t) (TILE - 1), (uint64_t) dst_addr, 0, out_chan_stride);
                } अन्यथा {
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
    अगर (need_stage) {
        et_barrier(ET_BARRIER_SHIRE);

        अगर (minion == 0) {
            स्थिर भग्न * stage = (स्थिर भग्न *) l2_output_stage;
            भग्न *       dram  = (भग्न *) out->तथ्य;
            क्रम (पूर्णांक oc = 0; oc < Cout; ++oc) {
                क्रम (पूर्णांक oh2 = 0; oh2 < OH; ++oh2) {
                    स्थिर भग्न * src = stage + ((माप_प्रकार) oc * OH + oh2) * OW_pad;
                    भग्न *       dst = dram + ((माप_प्रकार) oc * OH + oh2) * OW;
                    क्रम (पूर्णांक ow2 = 0; ow2 < OW; ++ow2) {
                        dst[ow2] = src[ow2];
                    }
                }
            }
            FENCE;
            स्थिर uint64_t योग_bytes = (uint64_t) Cout * OH * OW * माप(भग्न);
            evict_range_past_l2((स्थिर व्योम *) dram, योग_bytes);
            WAIT_CACHEOPS;
        }
    }

    वापस 0;
}
