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

#અન્તર્ભાવ "ggml_tensor.h"
#અન્તર્ભાવ "platform.h"
#અન્તર્ભાવ "tensor.h"

#અન્તર્ભાવ <etsoc/common/utils.h>
#અન્તર્ભાવ <stdbool.h>
#અન્તર્ભાવ <stdint.h>

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

#define SLAB_BYTES ((uint64_t) TILE * TILE * કદ(દશાંશ)) /* 1024 */
#define SLAB_લીટીઓ ((SLAB_BYTES + 63) / 64)                 /* 16   */

/* Upper bound on the number of distinct mt values a single shire may pack.
   This keeps the mt list stack-resident. Shapes that need more should fall
   back until the filter-slice bookkeeping is made dynamic. */
#define MAX_MY_MT (N_MIN_PER_SHIRE * MAX_TILES_PER_HART)

પ્રકારવ્યાખ્યા સંરચના {
    પૂર્ણાંક mt;
    પૂર્ણાંક mt_idx;
    પૂર્ણાંક oh;
    પૂર્ણાંક ow_base;
} conv_tile_t;

સ્થિર અંતઃરેખ પૂર્ણાંક ceil_div_i32(પૂર્ણાંક x, પૂર્ણાંક y) {
    પાછા (x + y - 1) / y;
}

સ્થિર અંતઃરેખ પૂર્ણાંક round_up_tile_i32(પૂર્ણાંક x) {
    પાછા (x + TILE - 1) & ~(TILE - 1);
}

સ્થિર અંતઃરેખ પૂર્ણાંક min_i32(પૂર્ણાંક a, પૂર્ણાંક b) {
    પાછા a < b ? a : b;
}

સ્થિર અંતઃરેખ uint64_t min_u64(uint64_t a, uint64_t b) {
    પાછા a < b ? a : b;
}

/* ===== Vector helpers for hart-1 pack ============================
   Both assume dst (and src for copy) are 32-byte aligned; n is in floats.
   The 8-element tail is handled scalar. f30/f31 are scratch — clobbered
   per-call via the asm clobber list. */
સ્થિર અંતઃરેખ શૂન્ય vec_zero_aligned(દશાંશ * dst, પૂર્ણાંક n) {
    પૂર્ણાંક       i  = 0;
    અચલ પૂર્ણાંક n8 = n & ~7;
    માટે (; i < n8; i += 8) {
        __asm__ અસ્થિર(
            "fsub.ps f31, f31, f31\n"
            "fsw.ps  f31, %[d]\n"
            : [d] "=m"(*(દશાંશ (*)[8]) & dst[i])
            :
            : "f31");
    }
    માટે (; i < n; ++i) {
        dst[i] = 0.0f;
    }
}

સ્થિર અંતઃરેખ શૂન્ય vec_copy_aligned(દશાંશ * dst, અચલ દશાંશ * src, પૂર્ણાંક n) {
    પૂર્ણાંક       i  = 0;
    અચલ પૂર્ણાંક n8 = n & ~7;
    માટે (; i < n8; i += 8) {
        __asm__ અસ્થિર(
            "flw.ps f30, %[s]\n"
            "fsw.ps f30, %[d]\n"
            : [d] "=m"(*(દશાંશ (*)[8]) & dst[i])
            : [s] "m"(*(અચલ દશાંશ (*)[8]) & src[i])
            : "f30");
    }
    માટે (; i < n; ++i) {
        dst[i] = src[i];
    }
}

/* ===== TenC save/restore =========================================
   The TenC accumulator IS the f0..f31 vector register file: row N occupies
   f(2N) and f(2N+1) (two 8-fp32 vector regs per row). We save by
   tensor_store-ing TILE rows × 64 bytes, and restore via 32 flw.ps after
   forcing L1D to refetch from the L2SCP backing (tensor_store bypasses L1D
   so the backing is always current). See feedback_tenc_save_restore.md. */
સ્થિર અંતઃરેખ શૂન્ય tenc_restore_from_scratch(uint64_t scr) {
    FENCE;
    evict_to_l2((અચલ શૂન્ય *) scr, TILE, 64);
    WAIT_CACHEOPS;
    __asm__ અસ્થિર(
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
પ્રકારવ્યાખ્યા સંરચના {
    અચલ દશાંશ * in_base;  /* DRAM input base [Cin][H][W]         */
    પૂર્ણાંક           Kw;
    પૂર્ણાંક           chunk_KT; /* number of K_TILES (=16-wide) per chunk */
    પૂર્ણાંક           H, W, Hp, Wp_a;
    પૂર્ણાંક           pad_h, pad_w, s0;
    પૂર્ણાંક           minion;          /* this hart's minion id (0..31) */
    uint64_t      pin_copy_floats; /* per-_s pin plane size in floats */
    uint64_t      l2_pad_in_buf[MAX_DBL_BUFS];
    uint64_t      pin_chunk_bytes; /* one chunk pin buffer's total size */
} pin_ctx_t;

સ્થિર અંતઃરેખ પૂર્ણાંક find_mt_idx(અચલ પૂર્ણાંક * my_mt, પૂર્ણાંક n_my_mt, પૂર્ણાંક mt) {
    માટે (પૂર્ણાંક j = 0; j < n_my_mt; ++j) {
        જો (my_mt[j] == mt) {
            પાછા j;
        }
    }
    પાછા 0;
}

સ્થિર અંતઃરેખ conv_tile_t decode_tile(પૂર્ણાંક t, પૂર્ણાંક M_TILES, પૂર્ણાંક w_tiles, અચલ પૂર્ણાંક * my_mt, પૂર્ણાંક n_my_mt) {
    conv_tile_t tile;
    tile.mt = t % M_TILES;
    t /= M_TILES;
    અચલ પૂર્ણાંક wt = t % w_tiles;
    t /= w_tiles;
    tile.oh      = t;
    tile.ow_base = wt * TILE;
    tile.mt_idx  = find_mt_idx(my_mt, n_my_mt, tile.mt);
    પાછા tile;
}

સ્થિર અંતઃરેખ uint64_t
filter_slab_addr(uint64_t l2_filter, પૂર્ણાંક Kw, પૂર્ણાંક K_TILES, પૂર્ણાંક n_my_mt, પૂર્ણાંક mt_idx, પૂર્ણાંક kh, પૂર્ણાંક kw, પૂર્ણાંક kt_global) {
    પાછા l2_filter + (uint64_t) ((((kh * Kw + kw) * n_my_mt + mt_idx) * K_TILES + kt_global)) * SLAB_BYTES;
}

સ્થિર અંતઃરેખ uint64_t pin_tile_addr(uint64_t l2_pad_in,
                                     uint64_t pin_copy_bytes,
                                     પૂર્ણાંક      ktc,
                                     પૂર્ણાંક      kw,
                                     પૂર્ણાંક      Hp,
                                     પૂર્ણાંક      Wp_a,
                                     પૂર્ણાંક      oh,
                                     પૂર્ણાંક      ow_base,
                                     પૂર્ણાંક      s1,
                                     પૂર્ણાંક      kh) {
    અચલ પૂર્ણાંક ir_pad = oh * s1 + kh;
    પાછા l2_pad_in + (uint64_t) kw * pin_copy_bytes +
           (((uint64_t) (ktc * TILE) * Hp + ir_pad) * Wp_a + ow_base) * કદ(દશાંશ);
}

સ્થિર અંતઃરેખ અક્ષર * output_tile_addr(અક્ષર *              out_base,
                                      અચલ conv_tile_t * tile,
                                      uint64_t            out_chan_stride,
                                      uint64_t            out_row_stride) {
    પાછા out_base + (size_t) (tile->mt * TILE) * out_chan_stride + (size_t) tile->oh * out_row_stride +
           (size_t) tile->ow_base * કદ(દશાંશ);
}

સ્થિર અંતઃરેખ શૂન્ય flush_range_to_l2(અચલ શૂન્ય * addr, uint64_t n_bytes) {
    અચલ uint64_t total_લીટીઓ = (n_bytes + 63) / 64;
    અચલ અક્ષર *   fl_addr     = (અચલ અક્ષર *) addr;
    માટે (uint64_t done = 0; done < total_લીટીઓ;) {
        અચલ uint64_t batch = min_u64(total_લીટીઓ - done, 16);
        flush_to_l2((અચલ શૂન્ય *) (fl_addr + done * 64), batch, 64);
        done += batch;
    }
}

સ્થિર અંતઃરેખ શૂન્ય evict_range_past_l2(અચલ શૂન્ય * addr, uint64_t n_bytes) {
    અચલ uint64_t total_લીટીઓ = (n_bytes + 63) / 64;
    અચલ અક્ષર *   fl_addr     = (અચલ અક્ષર *) addr;
    માટે (uint64_t done = 0; done < total_લીટીઓ;) {
        અચલ uint64_t batch = min_u64(total_લીટીઓ - done, 16);
        evict_past_l2((અચલ શૂન્ય *) (fl_addr + done * 64), batch, 64);
        done += batch;
    }
}

/* One matrix-engine tile for one Cin chunk. This is the main optimization
   surface: A is double-buffered, B is single-buffered due to L1 SCP space. */
સ્થિર અંતઃરેખ શૂન્ય compute_tile_chunk(uint64_t            l2_filter,
                                      uint64_t            l2_pad_in,
                                      uint64_t            pin_copy_bytes,
                                      પૂર્ણાંક                 Kh,
                                      પૂર્ણાંક                 Kw,
                                      પૂર્ણાંક                 K_TILES,
                                      પૂર્ણાંક                 chunk_KT,
                                      પૂર્ણાંક                 kt_base,
                                      પૂર્ણાંક                 n_my_mt,
                                      પૂર્ણાંક                 Hp,
                                      પૂર્ણાંક                 Wp_a,
                                      પૂર્ણાંક                 s1,
                                      uint64_t            a_row_stride,
                                      uint64_t            b_row_stride,
                                      અચલ conv_tile_t * tile,
                                      bool                first_fma_clears_tenc) {
    અચલ પૂર્ણાંક      n_iters   = Kh * Kw * chunk_KT;
    અચલ uint64_t A_BUFS[2] = { LSCP_A_0, LSCP_A_1 };

    અચલ uint64_t a_addr0 = filter_slab_addr(l2_filter, Kw, K_TILES, n_my_mt, tile->mt_idx, 0, 0, kt_base);
    tensor_load(false, false, A_BUFS[0], 0, 0, a_addr0, 0, (uint64_t) (TILE - 1), a_row_stride, 0);

    માટે (પૂર્ણાંક iter = 0; iter < n_iters; ++iter) {
        અચલ પૂર્ણાંક ktc = iter % chunk_KT;
        અચલ પૂર્ણાંક rem = iter / chunk_KT;
        અચલ પૂર્ણાંક kw  = rem % Kw;
        અચલ પૂર્ણાંક kh  = rem / Kw;

        અચલ uint64_t b_addr =
            pin_tile_addr(l2_pad_in, pin_copy_bytes, ktc, kw, Hp, Wp_a, tile->oh, tile->ow_base, s1, kh);
        tensor_load(false, false, LSCP_B, 0, 0, b_addr, 0, (uint64_t) (TILE - 1), b_row_stride, 1);

        tensor_wait(TENSOR_LOAD_WAIT_0);
        tensor_wait(TENSOR_LOAD_WAIT_1);

        જો (iter + 1 < n_iters) {
            અચલ પૂર્ણાંક      ktc_n = (iter + 1) % chunk_KT;
            અચલ પૂર્ણાંક      rem_n = (iter + 1) / chunk_KT;
            અચલ પૂર્ણાંક      kw_n  = rem_n % Kw;
            અચલ પૂર્ણાંક      kh_n  = rem_n / Kw;
            અચલ uint64_t a_addr_n =
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
સ્થિર શૂન્ય pack_filter_local_mt(અચલ દશાંશ * flt_base,
                                 પૂર્ણાંક           Kh,
                                 પૂર્ણાંક           Kw,
                                 પૂર્ણાંક           Cin,
                                 પૂર્ણાંક           K_TILES,
                                 અચલ પૂર્ણાંક *   my_mt,
                                 પૂર્ણાંક           n_my_mt,
                                 પૂર્ણાંક           minion,
                                 uint64_t      l2_filter_base) {
    અચલ પૂર્ણાંક    n_slabs = Kh * Kw * n_my_mt * K_TILES;
    અચલ size_t kstep   = (size_t) Kh * Kw; /* Cin stride in floats */

    માટે (પૂર્ણાંક slab = minion; slab < n_slabs; slab += N_MIN_PER_SHIRE) {
        પૂર્ણાંક       t  = slab;
        અચલ પૂર્ણાંક kt = t % K_TILES;
        t /= K_TILES;
        અચલ પૂર્ણાંક mt_idx = t % n_my_mt;
        t /= n_my_mt;
        અચલ પૂર્ણાંક kw = t % Kw;
        t /= Kw;
        અચલ પૂર્ણાંક kh = t;
        અચલ પૂર્ણાંક mt = my_mt[mt_idx];

        અચલ uint64_t slab_offset = (uint64_t) slab * SLAB_BYTES;
        દશાંશ *        cell        = (દશાંશ *) (l2_filter_base + slab_offset);

        માટે (પૂર્ણાંક oc_in = 0; oc_in < TILE; ++oc_in) {
            અચલ પૂર્ણાંક     oc  = mt * TILE + oc_in;
            અચલ દશાંશ * src = flt_base + (((size_t) oc * Cin + (size_t) kt * TILE) * Kh + kh) * Kw + kw;
            દશાંશ *       row = cell + (size_t) oc_in * TILE;
            દશાંશ         scratch[TILE] __attribute__((aligned(32)));
            માટે (પૂર્ણાંક ic_in = 0; ic_in < TILE; ++ic_in) {
                scratch[ic_in] = src[(size_t) ic_in * kstep];
            }
            vec_copy_aligned(row, scratch, TILE);
        }
    }

    /* Flush this hart's dirty L1D lines for the slabs it wrote. */
    FENCE;
    માટે (પૂર્ણાંક slab = minion; slab < n_slabs; slab += N_MIN_PER_SHIRE) {
        અચલ uint64_t slab_offset = (uint64_t) slab * SLAB_BYTES;
        flush_to_l2((અચલ શૂન્ય *) (l2_filter_base + slab_offset), SLAB_લીટીઓ, 64);
    }
    WAIT_CACHEOPS;
}

/* Pack one Cin chunk of the input pin (Kw shifted padded copies) into the
   buf_idx side of local L2SCP. Work distributed across the 32 hart-1's in
   the shire by `plane % 32 == minion`. The final flush_to_l2 forces L1D
   write-back so hart-0's tensor_load sees the freshly written bytes. */
સ્થિર શૂન્ય pack_pin_chunk(અચલ pin_ctx_t * ctx, પૂર્ણાંક chunk_id, પૂર્ણાંક buf_idx) {
    અચલ પૂર્ણાંક kt_base  = chunk_id * ctx->chunk_KT;
    અચલ પૂર્ણાંક Kw       = ctx->Kw;
    અચલ પૂર્ણાંક chunk_KT = ctx->chunk_KT;
    અચલ પૂર્ણાંક H = ctx->H, W = ctx->W, Hp = ctx->Hp, Wp_a = ctx->Wp_a;
    અચલ પૂર્ણાંક pad_h = ctx->pad_h, pad_w = ctx->pad_w, s0 = ctx->s0;
    અચલ પૂર્ણાંક minion = ctx->minion;

    /* Pin pack: Kw shifted, padded copies of input rows. Bounds [vlo, vhi)
       hoisted outside the row loop so the inner loop is three regions
       (zero-prefix | bulk-copy | zero-suffix) with no per-element predicate. */
    દશાંશ *   pin0         = (દશાંશ *) ctx->l2_pad_in_buf[buf_idx];
    અચલ પૂર્ણાંક chunk_Cin    = chunk_KT * TILE;
    અચલ પૂર્ણાંક n_pin_planes = Kw * chunk_Cin;
    માટે (પૂર્ણાંક p = minion; p < n_pin_planes; p += N_MIN_PER_SHIRE) {
        અચલ પૂર્ણાંક s     = p / chunk_Cin;
        અચલ પૂર્ણાંક icc   = p % chunk_Cin;
        અચલ પૂર્ણાંક ic    = kt_base * TILE + icc;
        દશાંશ *   pin_s = pin0 + (size_t) s * ctx->pin_copy_floats;

        અચલ પૂર્ણાંક offset = s - pad_w;
        પૂર્ણાંક       vlo    = 0;
        જ્યારે (vlo < Wp_a && (s0 * vlo + offset) < 0) {
            vlo++;
        }
        પૂર્ણાંક vhi = Wp_a;
        જ્યારે (vhi > vlo && (s0 * (vhi - 1) + offset) >= W) {
            vhi--;
        }
        અચલ bool aligned = (s0 == 1) && ((vlo & 7) == 0) && (((vlo + offset) & 7) == 0);

        માટે (પૂર્ણાંક r = 0; r < Hp; ++r) {
            દશાંશ *   row    = pin_s + ((size_t) icc * Hp + r) * Wp_a;
            અચલ પૂર્ણાંક real_h = r - pad_h;
            જો (real_h < 0 || real_h >= H) {
                vec_zero_aligned(row, Wp_a);
                ચાલુ;
            }
            અચલ દશાંશ * src_row = ctx->in_base + ((size_t) ic * H + real_h) * W;

            માટે (પૂર્ણાંક cc = 0; cc < vlo; ++cc) {
                row[cc] = 0.0f;
            }

            જો (aligned) {
                vec_copy_aligned(row + vlo, src_row + vlo + offset, vhi - vlo);
            } નહીંતર જો (s0 == 1) {
                અચલ દશાંશ * csrc = src_row + vlo + offset;
                અચલ પૂર્ણાંક     n    = vhi - vlo;
                માટે (પૂર્ણાંક cc = 0; cc < n; ++cc) {
                    row[vlo + cc] = csrc[cc];
                }
            } નહીંતર {
                માટે (પૂર્ણાંક cc = vlo; cc < vhi; ++cc) {
                    row[cc] = src_row[s0 * cc + offset];
                }
            }

            માટે (પૂર્ણાંક cc = vhi; cc < Wp_a; ++cc) {
                row[cc] = 0.0f;
            }
        }
    }

    /* Flush this buffer's L1D-dirty lines down to L2SCP backing. */
    FENCE;
    flush_range_to_l2((અચલ શૂન્ય *) ctx->l2_pad_in_buf[buf_idx], ctx->pin_chunk_bytes);
    WAIT_CACHEOPS;
}

પૂર્ણાંક entry_point(સંરચના ggml_et_દ્વિઅંકી_params * params, શૂન્ય * env) {
    (શૂન્ય) env;

    અચલ પૂર્ણાંક shire   = get_shire_id();
    અચલ પૂર્ણાંક hart_id = get_hart_id();
    અચલ પૂર્ણાંક minion  = (hart_id >> 1) & 0x1F;
    અચલ પૂર્ણાંક hart1   = hart_id & 1;

    અચલ સંરચના ggml_tensor * flt = &params->src0; /* [Kw,Kh,Cin,Cout] */
    અચલ સંરચના ggml_tensor * in  = &params->src1; /* [W, H, Cin,N=1 ] */
    સંરચના ggml_tensor *       out = &params->dst;  /* [W, H, Cout,N=1] */

    અચલ પૂર્ણાંક Kw   = (પૂર્ણાંક) flt->ne[0];
    અચલ પૂર્ણાંક Kh   = (પૂર્ણાંક) flt->ne[1];
    અચલ પૂર્ણાંક Cin  = (પૂર્ણાંક) flt->ne[2];
    અચલ પૂર્ણાંક Cout = (પૂર્ણાંક) flt->ne[3];

    અચલ પૂર્ણાંક W  = (પૂર્ણાંક) in->ne[0];
    અચલ પૂર્ણાંક H  = (પૂર્ણાંક) in->ne[1];
    અચલ પૂર્ણાંક OW = (પૂર્ણાંક) out->ne[0];
    અચલ પૂર્ણાંક OH = (પૂર્ણાંક) out->ne[1];

    /* op_params layout (set by ggml_conv_2d):
         [0]=s0  [1]=s1  [2]=p0  [3]=p1  [4]=d0  [5]=d1   */
    અચલ પૂર્ણાંક s0    = out->op_params[0];
    અચલ પૂર્ણાંક s1    = out->op_params[1];
    અચલ પૂર્ણાંક pad_w = out->op_params[2];
    અચલ પૂર્ણાંક pad_h = out->op_params[3];

    જો (Cin <= 0 || Cout <= 0) {
        પાછા -1;
    }
    જો (Cin % TILE != 0 || Cout % TILE != 0) {
        પાછા -1;
    }
    જો (W <= 0 || H <= 0) {
        પાછા -1;
    }
    જો (s0 <= 0 || s1 <= 0) {
        પાછા -1;
    }
    જો (in->ne[2] != Cin || in->ne[3] != 1) {
        પાછા -1;
    }
    જો (out->ne[2] != Cout || out->ne[3] != 1) {
        પાછા -1;
    }
    જો (!flt->data || !in->data || !out->data) {
        પાછા -1;
    }

    અચલ પૂર્ણાંક K_TILES = Cin / TILE;
    અચલ પૂર્ણાંક M_TILES = Cout / TILE;

    અચલ પૂર્ણાંક  Hp         = H + 2 * pad_h;
    અચલ પૂર્ણાંક  Wp_a       = round_up_tile_i32(OW);
    અચલ પૂર્ણાંક  OW_pad     = Wp_a;
    અચલ bool need_stage = (OW % TILE != 0);

    /* ===================== Tile assignment & active-shire selection =====
       Computed up front because the per-shire mt set (and thus filter
       region size) depends on n_active_shires. */
    અચલ પૂર્ણાંક w_tiles         = ceil_div_i32(OW, TILE);
    અચલ પૂર્ણાંક total_tiles     = OH * w_tiles * M_TILES;
    અચલ પૂર્ણાંક n_active_shires = need_stage ? 1 : min_i32(total_tiles, N_SHIRES);

    /* Inactive shires exit immediately. No global barrier — pack and
       barriers are now per-shire, so unused shires don't need to vote. */
    જો (shire >= n_active_shires) {
        પાછા 0;
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
    પૂર્ણાંક my_mt[MAX_MY_MT];
    પૂર્ણાંક n_my_mt = 0;
    માટે (પૂર્ણાંક t = shire; t < total_tiles; t += n_active_shires) {
        અચલ પૂર્ણાંક mt    = t % M_TILES;
        bool      found = false;
        માટે (પૂર્ણાંક j = 0; j < n_my_mt; ++j) {
            જો (my_mt[j] == mt) {
                found = true;
                તોડો;
            }
        }
        જો (!found) {
            જો (n_my_mt >= MAX_MY_MT) {
                પાછા -1;
            }
            my_mt[n_my_mt++] = mt;
        }
    }
    જો (n_my_mt == 0) {
        પાછા 0; /* no tiles for this shire */
    }

    અચલ uint64_t filter_local_bytes = (uint64_t) Kh * Kw * n_my_mt * K_TILES * SLAB_BYTES;
    જો (filter_local_bytes > LOCAL_FILTER_CAP) {
        પાછા -1;
    }

    /* ===================== L2 SCP local layout =========================
       filter (this shire's mt slice) | pin_buf[0] | pin_buf[1]?
                                      | output_stage? | scratch (streaming) */
    અચલ uint64_t l2_base   = (uint64_t) et_shire_l2scp_local(0);
    અચલ uint64_t l2_filter = l2_base;

    /* Sizing for pin: budget = LOCAL_BUDGET - filter - output_stage.    */
    અચલ int64_t output_stage_bytes_full = need_stage ? (int64_t) Cout * OH * OW_pad * (int64_t) કદ(દશાંશ) : 0;
    અચલ int64_t budget_for_chunks = (int64_t) LOCAL_BUDGET - (int64_t) filter_local_bytes - output_stage_bytes_full;
    જો (budget_for_chunks <= 0) {
        પાછા -1;
    }
    અચલ int64_t per_KT_pin_bytes = (int64_t) Kw * TILE * Hp * Wp_a * (int64_t) કદ(દશાંશ);

    પૂર્ણાંક chunk_KT;
    પૂર્ણાંક n_buffers;
    જો ((int64_t) K_TILES * per_KT_pin_bytes <= budget_for_chunks) {
        chunk_KT  = K_TILES;
        n_buffers = 1;
    } નહીંતર {
        chunk_KT = K_TILES;
        જ્યારે (chunk_KT > 1 && 2 * (int64_t) chunk_KT * per_KT_pin_bytes > budget_for_chunks) {
            chunk_KT--;
        }
        જ્યારે (chunk_KT > 1 && K_TILES % chunk_KT != 0) {
            chunk_KT--;
        }
        n_buffers = (chunk_KT < K_TILES) ? 2 : 1;
        જો (chunk_KT < 1) {
            પાછા -1;
        }
    }
    અચલ પૂર્ણાંક n_chunks = K_TILES / chunk_KT;

    /* Streaming keeps partial sums in MAX_TILES_PER_HART scratch slots per
       hart. The one-chunk path does not need scratch and can stream a longer
       tile list serially, but multi-chunk shapes must fit this fixed slot
       count until scratch scheduling is made more general. */
    અચલ પૂર્ણાંક shire_tile_capacity = shire + MAX_TILES_PER_HART * n_active_shires * N_MIN_PER_SHIRE;
    જો (n_chunks > 1 && shire_tile_capacity < total_tiles) {
        પાછા -1;
    }

    અચલ uint64_t pin_copy_floats = (uint64_t) chunk_KT * TILE * Hp * Wp_a;
    અચલ uint64_t pin_copy_bytes  = pin_copy_floats * કદ(દશાંશ);
    અચલ uint64_t pin_chunk_bytes = (uint64_t) Kw * pin_copy_bytes;

    અચલ uint64_t l2_pin_base              = l2_filter + filter_local_bytes;
    અચલ uint64_t l2_pin_buf[MAX_DBL_BUFS] = {
        l2_pin_base,
        l2_pin_base + pin_chunk_bytes,
    };

    અચલ uint64_t l2_output_stage = need_stage ? l2_pin_base + (uint64_t) n_buffers * pin_chunk_bytes : 0;

    અચલ uint64_t scratch_per_hart = (uint64_t) MAX_TILES_PER_HART * (uint64_t) TILE * TILE * કદ(દશાંશ);
    અચલ uint64_t l2_scratch_base  = need_stage ? l2_output_stage + (uint64_t) output_stage_bytes_full :
                                                   l2_pin_base + (uint64_t) n_buffers * pin_chunk_bytes;

    /* ===================== PHASE 1: Filter pack (per-shire mt slice) ====
       Hart-1's pack only this shire's mt slabs into local L2 SCP. The
       SHIRE barrier below ensures the filter is in L2 SCP backing before
       hart-0's first tensor_load. */
    જો (hart1) {
        pack_filter_local_mt((અચલ દશાંશ *) flt->data, Kh, Kw, Cin, K_TILES, my_mt, n_my_mt, minion, l2_filter);
    }

    /* ===================== Hart 1: pin packer (per chunk) ==============
       Double-buffered prefetch: pack chunk 0 synchronously, then per chunk c
       signal "buf c ready", pack chunk c+1 into the alternate buffer
       (overlaps hart-0's compute on c), signal "buf c done". */
    જો (hart1) {
        અચલ pin_ctx_t ctx = {
            .in_base         = (અચલ દશાંશ *) in->data,
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

        માટે (પૂર્ણાંક c = 0; c < n_chunks; ++c) {
            et_barrier(ET_BARRIER_SHIRE); /* signal "buf c ready" */
            જો (n_buffers > 1 && c + 1 < n_chunks) {
                pack_pin_chunk(&ctx, c + 1, (c + 1) & 1);
            }
            et_barrier(ET_BARRIER_SHIRE); /* wait "buf c done" */
        }

        જો (need_stage) {
            et_barrier(ET_BARRIER_SHIRE);
        }
        પાછા 0;
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

    અક્ષર * અચલ   out_base        = need_stage ? (અક્ષર *) l2_output_stage : (અક્ષર *) out->data;
    અચલ પૂર્ણાંક      compute_OW      = need_stage ? OW_pad : OW;
    અચલ uint64_t out_chan_stride = (uint64_t) OH * (uint64_t) compute_OW * કદ(દશાંશ);
    અચલ uint64_t out_row_stride  = (uint64_t) compute_OW * કદ(દશાંશ);

    અચલ uint64_t a_row_stride = (uint64_t) TILE * કદ(દશાંશ); /* 64 */
    અચલ uint64_t b_row_stride = (uint64_t) Hp * (uint64_t) Wp_a * કદ(દશાંશ);

    /* Tile assignment: shire-strided so small workloads spread across
       shires before stacking minions in one shire. */
    અચલ પૂર્ણાંક t_start  = shire + minion * n_active_shires;
    અચલ પૂર્ણાંક t_stride = n_active_shires * N_MIN_PER_SHIRE;

    જો (n_chunks == 1) {
        et_barrier(ET_BARRIER_SHIRE); /* wait for the (only) pin chunk */

        અચલ uint64_t l2_pad_in = l2_pin_buf[0];
        માટે (પૂર્ણાંક t = t_start; t < total_tiles; t += t_stride) {
            અચલ conv_tile_t tile = decode_tile(t, M_TILES, w_tiles, my_mt, n_my_mt);
            compute_tile_chunk(l2_filter, l2_pad_in, pin_copy_bytes, Kh, Kw, K_TILES, chunk_KT, 0, n_my_mt, Hp, Wp_a,
                               s1, a_row_stride, b_row_stride, &tile, /*first_fma_clears_tenc=*/true);

            અક્ષર * dst_addr = output_tile_addr(out_base, &tile, out_chan_stride, out_row_stride);
            tensor_store(0, 0, 3, (uint64_t) (TILE - 1), (uint64_t) dst_addr, 0, out_chan_stride);
            tensor_wait(TENSOR_STORE_WAIT);
        }

        et_barrier(ET_BARRIER_SHIRE); /* matches hart-1's second barrier */

    } નહીંતર {
        /* Streaming path: each hart owns up to MAX_TILES_PER_HART tiles. */
        પૂર્ણાંક my_tiles[MAX_TILES_PER_HART];
        પૂર્ણાંક n_my_tiles = 0;
        માટે (પૂર્ણાંક slot = 0; slot < MAX_TILES_PER_HART; ++slot) {
            અચલ પૂર્ણાંક t = t_start + slot * t_stride;
            જો (t < total_tiles) {
                my_tiles[n_my_tiles++] = t;
            }
        }

        conv_tile_t tiles[MAX_TILES_PER_HART];
        માટે (પૂર્ણાંક k = 0; k < n_my_tiles; ++k) {
            tiles[k] = decode_tile(my_tiles[k], M_TILES, w_tiles, my_mt, n_my_mt);
        }

        અચલ uint64_t my_scratch_base = l2_scratch_base + (uint64_t) minion * scratch_per_hart;

        માટે (પૂર્ણાંક c = 0; c < n_chunks; ++c) {
            et_barrier(ET_BARRIER_SHIRE); /* pin chunk c packed */

            અચલ પૂર્ણાંક      buf       = c & 1;
            અચલ uint64_t l2_pad_in = l2_pin_buf[buf];
            અચલ પૂર્ણાંક      kt_base   = c * chunk_KT;

            માટે (પૂર્ણાંક k = 0; k < n_my_tiles; ++k) {
                અચલ conv_tile_t * tile = &tiles[k];
                અચલ uint64_t      scr  = my_scratch_base + (uint64_t) k * (TILE * TILE * કદ(દશાંશ));

                અચલ bool first_pass_chunk = (c == 0);
                જો (!first_pass_chunk) {
                    tenc_restore_from_scratch(scr);
                }

                compute_tile_chunk(l2_filter, l2_pad_in, pin_copy_bytes, Kh, Kw, K_TILES, chunk_KT, kt_base, n_my_mt,
                                   Hp, Wp_a, s1, a_row_stride, b_row_stride, tile, first_pass_chunk);

                જો (c == n_chunks - 1) {
                    અક્ષર * dst_addr = output_tile_addr(out_base, tile, out_chan_stride, out_row_stride);
                    tensor_store(0, 0, 3, (uint64_t) (TILE - 1), (uint64_t) dst_addr, 0, out_chan_stride);
                } નહીંતર {
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
    જો (need_stage) {
        et_barrier(ET_BARRIER_SHIRE);

        જો (minion == 0) {
            અચલ દશાંશ * stage = (અચલ દશાંશ *) l2_output_stage;
            દશાંશ *       dram  = (દશાંશ *) out->data;
            માટે (પૂર્ણાંક oc = 0; oc < Cout; ++oc) {
                માટે (પૂર્ણાંક oh2 = 0; oh2 < OH; ++oh2) {
                    અચલ દશાંશ * src = stage + ((size_t) oc * OH + oh2) * OW_pad;
                    દશાંશ *       dst = dram + ((size_t) oc * OH + oh2) * OW;
                    માટે (પૂર્ણાંક ow2 = 0; ow2 < OW; ++ow2) {
                        dst[ow2] = src[ow2];
                    }
                }
            }
            FENCE;
            અચલ uint64_t total_bytes = (uint64_t) Cout * OH * OW * કદ(દશાંશ);
            evict_range_past_l2((અચલ શૂન્ય *) dram, total_bytes);
            WAIT_CACHEOPS;
        }
    }

    પાછા 0;
}
