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

#अन्तर्भाव "ggml_tensor.h"
#अन्तर्भाव "platform.h"
#अन्तर्भाव "tensor.h"

#अन्तर्भाव <etsoc/common/utils.h>
#अन्तर्भाव <stdbool.h>
#अन्तर्भाव <stdint.h>

#परिभाषय TILE               16 /* matrix engine native tile in M, K, N */
/* L1 SCP layout: A double-buffered, B single-buffered. Per the SDK doc
   `dst_start` is a 6-bit field (max 63) but empirical testing shows the
   physical L1 SCP per minion is 48 lines — writes to lines >= 48 corrupt.
   So we get 3 × 16-line buffers max: A_0, A_1, B. Pick A as the
   double-buffered operand (filter-slab loads, the longer of the two). */
#परिभाषय LSCP_A_0           0  /* A buffer 0 at L1 SCP lines 0..15  */
#परिभाषय LSCP_A_1           16 /* A buffer 1 at L1 SCP lines 16..31 */
#परिभाषय LSCP_B             32 /* B (single buffer) at lines 32..47 */
#परिभाषय N_MIN_प्रत्येकम्_SHIRE    32 /* ET-SoC-1 geometry: 32 minions/shire  */
#परिभाषय N_SHIRES           32 /* default active shire count           */
#परिभाषय MAX_TILES_प्रत्येकम्_HART 2  /* per-hart TenC slots (save/restore)   */
#परिभाषय MAX_DBL_BUFS       2  /* chunk pack buffers (double-buffered) */

/* Per-shire L2 SCP local budget. Per-shire SCP is 2 MB; we cap at
   1984 KB to leave 64 KB headroom for per-hart TenC scratch (32 minions ×
   2 slots × 1 KB), which lives at the tail of the SCP outside the pin
   sizing budget. Bigger budget here means bigger feasible chunk_KT,
   which means fewer chunks (each chunk costs 2 SHIRE barriers + ~30
   TenC save/restore events per hart). */
#परिभाषय LOCAL_BUDGET (1984 * 1024)

/* Cap on the per-shire filter region in local L2 SCP. The shire packs the
   mt values it can consume under the current tile assignment, rather than
   the whole Cout dimension. Reads in the inner loop are then fully
   shire-local — no NoC fanout. */
#परिभाषय LOCAL_FILTER_CAP (1024 * 1024)                      /* 1 MB / shire ceiling */

#परिभाषय SLAB_अष्टकानि ((uint64_t) TILE * TILE * परिमाण(भिन्न)) /* 1024 */
#परिभाषय SLAB_पङ्क्तयः ((SLAB_अष्टकानि + 63) / 64)                 /* 16   */

/* Upper bound on the number of distinct mt values a single shire may pack.
   This keeps the mt list stack-resident. Shapes that need more should fall
   back until the filter-slice bookkeeping is made dynamic. */
#परिभाषय MAX_MY_MT (N_MIN_प्रत्येकम्_SHIRE * MAX_TILES_प्रत्येकम्_HART)

प्रकारनिर्देश संरचना {
    पूर्णाङ्क mt;
    पूर्णाङ्क mt_idx;
    पूर्णाङ्क oh;
    पूर्णाङ्क ow_आधारः;
} conv_tile_t;

स्थिर अन्तःस्थ पूर्णाङ्क ceil_div_i32(पूर्णाङ्क x, पूर्णाङ्क y) {
    प्रत्यागम (x + y - 1) / y;
}

स्थिर अन्तःस्थ पूर्णाङ्क आवर्तनम्_ऊर्ध्वम्_tile_i32(पूर्णाङ्क x) {
    प्रत्यागम (x + TILE - 1) & ~(TILE - 1);
}

स्थिर अन्तःस्थ पूर्णाङ्क min_i32(पूर्णाङ्क a, पूर्णाङ्क b) {
    प्रत्यागम a < b ? a : b;
}

स्थिर अन्तःस्थ uint64_t min_u64(uint64_t a, uint64_t b) {
    प्रत्यागम a < b ? a : b;
}

/* ===== Vector helpers for hart-1 pack ============================
   Both assume dst (and src for copy) are 32-byte aligned; n is in floats.
   The 8-element tail is handled scalar. f30/f31 are scratch — clobbered
   per-call via the asm clobber list. */
स्थिर अन्तःस्थ शून्य vec_zero_aligned(भिन्न * dst, पूर्णाङ्क n) {
    पूर्णाङ्क       i  = 0;
    ध्रुव पूर्णाङ्क n8 = n & ~7;
    प्रति (; i < n8; i += 8) {
        __asm__ चञ्चल(
            "fsub.ps f31, f31, f31\n"
            "fsw.ps  f31, %[d]\n"
            : [d] "=m"(*(भिन्न (*)[8]) & dst[i])
            :
            : "f31");
    }
    प्रति (; i < n; ++i) {
        dst[i] = 0.0f;
    }
}

स्थिर अन्तःस्थ शून्य vec_copy_aligned(भिन्न * dst, ध्रुव भिन्न * src, पूर्णाङ्क n) {
    पूर्णाङ्क       i  = 0;
    ध्रुव पूर्णाङ्क n8 = n & ~7;
    प्रति (; i < n8; i += 8) {
        __asm__ चञ्चल(
            "flw.ps f30, %[s]\n"
            "fsw.ps f30, %[d]\n"
            : [d] "=m"(*(भिन्न (*)[8]) & dst[i])
            : [s] "m"(*(ध्रुव भिन्न (*)[8]) & src[i])
            : "f30");
    }
    प्रति (; i < n; ++i) {
        dst[i] = src[i];
    }
}

/* ===== TenC save/restore =========================================
   The TenC accumulator IS the f0..f31 vector register file: row N occupies
   f(2N) and f(2N+1) (two 8-fp32 vector regs per row). We save by
   tensor_store-ing TILE rows × 64 bytes, and restore via 32 flw.ps after
   forcing L1D to refetch from the L2SCP backing (tensor_store bypasses L1D
   so the backing is always current). See feedback_tenc_save_restore.md. */
स्थिर अन्तःस्थ शून्य tenc_restore_from_scratch(uint64_t scr) {
    FENCE;
    evict_गन्तव्यम्_l2((ध्रुव शून्य *) scr, TILE, 64);
    WAIT_CACHEOPS;
    __asm__ चञ्चल(
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
प्रकारनिर्देश संरचना {
    ध्रुव भिन्न * अन्तर्_आधारः;  /* DRAM input base [Cin][H][W]         */
    पूर्णाङ्क           Kw;
    पूर्णाङ्क           chunk_KT; /* number of K_TILES (=16-wide) per chunk */
    पूर्णाङ्क           H, W, Hp, Wp_a;
    पूर्णाङ्क           pad_h, pad_w, s0;
    पूर्णाङ्क           minion;          /* this hart's minion id (0..31) */
    uint64_t      pin_copy_floats; /* per-_s pin plane size in floats */
    uint64_t      l2_pad_अन्तर्_buf[MAX_DBL_BUFS];
    uint64_t      pin_chunk_अष्टकानि; /* one chunk pin buffer's total size */
} pin_ctx_t;

स्थिर अन्तःस्थ पूर्णाङ्क find_mt_idx(ध्रुव पूर्णाङ्क * my_mt, पूर्णाङ्क n_my_mt, पूर्णाङ्क mt) {
    प्रति (पूर्णाङ्क j = 0; j < n_my_mt; ++j) {
        यदि (my_mt[j] == mt) {
            प्रत्यागम j;
        }
    }
    प्रत्यागम 0;
}

स्थिर अन्तःस्थ conv_tile_t decode_tile(पूर्णाङ्क t, पूर्णाङ्क M_TILES, पूर्णाङ्क w_tiles, ध्रुव पूर्णाङ्क * my_mt, पूर्णाङ्क n_my_mt) {
    conv_tile_t tile;
    tile.mt = t % M_TILES;
    t /= M_TILES;
    ध्रुव पूर्णाङ्क wt = t % w_tiles;
    t /= w_tiles;
    tile.oh      = t;
    tile.ow_आधारः = wt * TILE;
    tile.mt_idx  = find_mt_idx(my_mt, n_my_mt, tile.mt);
    प्रत्यागम tile;
}

स्थिर अन्तःस्थ uint64_t
filter_slab_addr(uint64_t l2_filter, पूर्णाङ्क Kw, पूर्णाङ्क K_TILES, पूर्णाङ्क n_my_mt, पूर्णाङ्क mt_idx, पूर्णाङ्क kh, पूर्णाङ्क kw, पूर्णाङ्क kt_global) {
    प्रत्यागम l2_filter + (uint64_t) ((((kh * Kw + kw) * n_my_mt + mt_idx) * K_TILES + kt_global)) * SLAB_अष्टकानि;
}

स्थिर अन्तःस्थ uint64_t pin_tile_addr(uint64_t l2_pad_अन्तर्,
                                     uint64_t pin_copy_अष्टकानि,
                                     पूर्णाङ्क      ktc,
                                     पूर्णाङ्क      kw,
                                     पूर्णाङ्क      Hp,
                                     पूर्णाङ्क      Wp_a,
                                     पूर्णाङ्क      oh,
                                     पूर्णाङ्क      ow_आधारः,
                                     पूर्णाङ्क      s1,
                                     पूर्णाङ्क      kh) {
    ध्रुव पूर्णाङ्क ir_pad = oh * s1 + kh;
    प्रत्यागम l2_pad_अन्तर् + (uint64_t) kw * pin_copy_अष्टकानि +
           (((uint64_t) (ktc * TILE) * Hp + ir_pad) * Wp_a + ow_आधारः) * परिमाण(भिन्न);
}

स्थिर अन्तःस्थ वर्ण * output_tile_addr(वर्ण *              बहिर्गमनम्_आधारः,
                                      ध्रुव conv_tile_t * tile,
                                      uint64_t            बहिर्गमनम्_chan_stride,
                                      uint64_t            बहिर्गमनम्_row_stride) {
    प्रत्यागम बहिर्गमनम्_आधारः + (size_t) (tile->mt * TILE) * बहिर्गमनम्_chan_stride + (size_t) tile->oh * बहिर्गमनम्_row_stride +
           (size_t) tile->ow_आधारः * परिमाण(भिन्न);
}

स्थिर अन्तःस्थ शून्य flush_range_गन्तव्यम्_l2(ध्रुव शून्य * addr, uint64_t n_अष्टकानि) {
    ध्रुव uint64_t योग_पङ्क्तयः = (n_अष्टकानि + 63) / 64;
    ध्रुव वर्ण *   fl_addr     = (ध्रुव वर्ण *) addr;
    प्रति (uint64_t समाप्तम् = 0; समाप्तम् < योग_पङ्क्तयः;) {
        ध्रुव uint64_t batch = min_u64(योग_पङ्क्तयः - समाप्तम्, 16);
        flush_गन्तव्यम्_l2((ध्रुव शून्य *) (fl_addr + समाप्तम् * 64), batch, 64);
        समाप्तम् += batch;
    }
}

स्थिर अन्तःस्थ शून्य evict_range_past_l2(ध्रुव शून्य * addr, uint64_t n_अष्टकानि) {
    ध्रुव uint64_t योग_पङ्क्तयः = (n_अष्टकानि + 63) / 64;
    ध्रुव वर्ण *   fl_addr     = (ध्रुव वर्ण *) addr;
    प्रति (uint64_t समाप्तम् = 0; समाप्तम् < योग_पङ्क्तयः;) {
        ध्रुव uint64_t batch = min_u64(योग_पङ्क्तयः - समाप्तम्, 16);
        evict_past_l2((ध्रुव शून्य *) (fl_addr + समाप्तम् * 64), batch, 64);
        समाप्तम् += batch;
    }
}

/* One matrix-engine tile for one Cin chunk. This is the main optimization
   surface: A is double-buffered, B is single-buffered due to L1 SCP space. */
स्थिर अन्तःस्थ शून्य compute_tile_chunk(uint64_t            l2_filter,
                                      uint64_t            l2_pad_अन्तर्,
                                      uint64_t            pin_copy_अष्टकानि,
                                      पूर्णाङ्क                 Kh,
                                      पूर्णाङ्क                 Kw,
                                      पूर्णाङ्क                 K_TILES,
                                      पूर्णाङ्क                 chunk_KT,
                                      पूर्णाङ्क                 kt_आधारः,
                                      पूर्णाङ्क                 n_my_mt,
                                      पूर्णाङ्क                 Hp,
                                      पूर्णाङ्क                 Wp_a,
                                      पूर्णाङ्क                 s1,
                                      uint64_t            a_row_stride,
                                      uint64_t            b_row_stride,
                                      ध्रुव conv_tile_t * tile,
                                      bool                first_fma_clears_tenc) {
    ध्रुव पूर्णाङ्क      n_iters   = Kh * Kw * chunk_KT;
    ध्रुव uint64_t A_BUFS[2] = { LSCP_A_0, LSCP_A_1 };

    ध्रुव uint64_t a_addr0 = filter_slab_addr(l2_filter, Kw, K_TILES, n_my_mt, tile->mt_idx, 0, 0, kt_आधारः);
    tensor_आरोपय(false, false, A_BUFS[0], 0, 0, a_addr0, 0, (uint64_t) (TILE - 1), a_row_stride, 0);

    प्रति (पूर्णाङ्क iter = 0; iter < n_iters; ++iter) {
        ध्रुव पूर्णाङ्क ktc = iter % chunk_KT;
        ध्रुव पूर्णाङ्क rem = iter / chunk_KT;
        ध्रुव पूर्णाङ्क kw  = rem % Kw;
        ध्रुव पूर्णाङ्क kh  = rem / Kw;

        ध्रुव uint64_t b_addr =
            pin_tile_addr(l2_pad_अन्तर्, pin_copy_अष्टकानि, ktc, kw, Hp, Wp_a, tile->oh, tile->ow_आधारः, s1, kh);
        tensor_आरोपय(false, false, LSCP_B, 0, 0, b_addr, 0, (uint64_t) (TILE - 1), b_row_stride, 1);

        tensor_wait(TENSOR_आरोपय_WAIT_0);
        tensor_wait(TENSOR_आरोपय_WAIT_1);

        यदि (iter + 1 < n_iters) {
            ध्रुव पूर्णाङ्क      ktc_n = (iter + 1) % chunk_KT;
            ध्रुव पूर्णाङ्क      rem_n = (iter + 1) / chunk_KT;
            ध्रुव पूर्णाङ्क      kw_n  = rem_n % Kw;
            ध्रुव पूर्णाङ्क      kh_n  = rem_n / Kw;
            ध्रुव uint64_t a_addr_n =
                filter_slab_addr(l2_filter, Kw, K_TILES, n_my_mt, tile->mt_idx, kh_n, kw_n, kt_आधारः + ktc_n);
            tensor_आरोपय(false, false, A_BUFS[(iter + 1) & 1], 0, 0, a_addr_n, 0, (uint64_t) (TILE - 1), a_row_stride,
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
स्थिर शून्य pack_filter_local_mt(ध्रुव भिन्न * flt_आधारः,
                                 पूर्णाङ्क           Kh,
                                 पूर्णाङ्क           Kw,
                                 पूर्णाङ्क           Cin,
                                 पूर्णाङ्क           K_TILES,
                                 ध्रुव पूर्णाङ्क *   my_mt,
                                 पूर्णाङ्क           n_my_mt,
                                 पूर्णाङ्क           minion,
                                 uint64_t      l2_filter_आधारः) {
    ध्रुव पूर्णाङ्क    n_slabs = Kh * Kw * n_my_mt * K_TILES;
    ध्रुव size_t kstep   = (size_t) Kh * Kw; /* Cin stride in floats */

    प्रति (पूर्णाङ्क slab = minion; slab < n_slabs; slab += N_MIN_प्रत्येकम्_SHIRE) {
        पूर्णाङ्क       t  = slab;
        ध्रुव पूर्णाङ्क kt = t % K_TILES;
        t /= K_TILES;
        ध्रुव पूर्णाङ्क mt_idx = t % n_my_mt;
        t /= n_my_mt;
        ध्रुव पूर्णाङ्क kw = t % Kw;
        t /= Kw;
        ध्रुव पूर्णाङ्क kh = t;
        ध्रुव पूर्णाङ्क mt = my_mt[mt_idx];

        ध्रुव uint64_t slab_विस्थापनम् = (uint64_t) slab * SLAB_अष्टकानि;
        भिन्न *        cell        = (भिन्न *) (l2_filter_आधारः + slab_विस्थापनम्);

        प्रति (पूर्णाङ्क oc_अन्तर् = 0; oc_अन्तर् < TILE; ++oc_अन्तर्) {
            ध्रुव पूर्णाङ्क     oc  = mt * TILE + oc_अन्तर्;
            ध्रुव भिन्न * src = flt_आधारः + (((size_t) oc * Cin + (size_t) kt * TILE) * Kh + kh) * Kw + kw;
            भिन्न *       row = cell + (size_t) oc_अन्तर् * TILE;
            भिन्न         scratch[TILE] __attribute__((aligned(32)));
            प्रति (पूर्णाङ्क ic_अन्तर् = 0; ic_अन्तर् < TILE; ++ic_अन्तर्) {
                scratch[ic_अन्तर्] = src[(size_t) ic_अन्तर् * kstep];
            }
            vec_copy_aligned(row, scratch, TILE);
        }
    }

    /* Flush this hart's dirty L1D lines for the slabs it wrote. */
    FENCE;
    प्रति (पूर्णाङ्क slab = minion; slab < n_slabs; slab += N_MIN_प्रत्येकम्_SHIRE) {
        ध्रुव uint64_t slab_विस्थापनम् = (uint64_t) slab * SLAB_अष्टकानि;
        flush_गन्तव्यम्_l2((ध्रुव शून्य *) (l2_filter_आधारः + slab_विस्थापनम्), SLAB_पङ्क्तयः, 64);
    }
    WAIT_CACHEOPS;
}

/* Pack one Cin chunk of the input pin (Kw shifted padded copies) into the
   buf_idx side of local L2SCP. Work distributed across the 32 hart-1's in
   the shire by `plane % 32 == minion`. The final flush_to_l2 forces L1D
   write-back so hart-0's tensor_load sees the freshly written bytes. */
स्थिर शून्य pack_pin_chunk(ध्रुव pin_ctx_t * ctx, पूर्णाङ्क chunk_id, पूर्णाङ्क buf_idx) {
    ध्रुव पूर्णाङ्क kt_आधारः  = chunk_id * ctx->chunk_KT;
    ध्रुव पूर्णाङ्क Kw       = ctx->Kw;
    ध्रुव पूर्णाङ्क chunk_KT = ctx->chunk_KT;
    ध्रुव पूर्णाङ्क H = ctx->H, W = ctx->W, Hp = ctx->Hp, Wp_a = ctx->Wp_a;
    ध्रुव पूर्णाङ्क pad_h = ctx->pad_h, pad_w = ctx->pad_w, s0 = ctx->s0;
    ध्रुव पूर्णाङ्क minion = ctx->minion;

    /* Pin pack: Kw shifted, padded copies of input rows. Bounds [vlo, vhi)
       hoisted outside the row loop so the inner loop is three regions
       (zero-prefix | bulk-copy | zero-suffix) with no per-element predicate. */
    भिन्न *   pin0         = (भिन्न *) ctx->l2_pad_अन्तर्_buf[buf_idx];
    ध्रुव पूर्णाङ्क chunk_Cin    = chunk_KT * TILE;
    ध्रुव पूर्णाङ्क n_pin_planes = Kw * chunk_Cin;
    प्रति (पूर्णाङ्क p = minion; p < n_pin_planes; p += N_MIN_प्रत्येकम्_SHIRE) {
        ध्रुव पूर्णाङ्क s     = p / chunk_Cin;
        ध्रुव पूर्णाङ्क icc   = p % chunk_Cin;
        ध्रुव पूर्णाङ्क ic    = kt_आधारः * TILE + icc;
        भिन्न *   pin_s = pin0 + (size_t) s * ctx->pin_copy_floats;

        ध्रुव पूर्णाङ्क विस्थापनम् = s - pad_w;
        पूर्णाङ्क       vlo    = 0;
        यावत् (vlo < Wp_a && (s0 * vlo + विस्थापनम्) < 0) {
            vlo++;
        }
        पूर्णाङ्क vhi = Wp_a;
        यावत् (vhi > vlo && (s0 * (vhi - 1) + विस्थापनम्) >= W) {
            vhi--;
        }
        ध्रुव bool aligned = (s0 == 1) && ((vlo & 7) == 0) && (((vlo + विस्थापनम्) & 7) == 0);

        प्रति (पूर्णाङ्क r = 0; r < Hp; ++r) {
            भिन्न *   row    = pin_s + ((size_t) icc * Hp + r) * Wp_a;
            ध्रुव पूर्णाङ्क real_h = r - pad_h;
            यदि (real_h < 0 || real_h >= H) {
                vec_zero_aligned(row, Wp_a);
                अनुवर्त;
            }
            ध्रुव भिन्न * src_row = ctx->अन्तर्_आधारः + ((size_t) ic * H + real_h) * W;

            प्रति (पूर्णाङ्क cc = 0; cc < vlo; ++cc) {
                row[cc] = 0.0f;
            }

            यदि (aligned) {
                vec_copy_aligned(row + vlo, src_row + vlo + विस्थापनम्, vhi - vlo);
            } अन्यथा यदि (s0 == 1) {
                ध्रुव भिन्न * csrc = src_row + vlo + विस्थापनम्;
                ध्रुव पूर्णाङ्क     n    = vhi - vlo;
                प्रति (पूर्णाङ्क cc = 0; cc < n; ++cc) {
                    row[vlo + cc] = csrc[cc];
                }
            } अन्यथा {
                प्रति (पूर्णाङ्क cc = vlo; cc < vhi; ++cc) {
                    row[cc] = src_row[s0 * cc + विस्थापनम्];
                }
            }

            प्रति (पूर्णाङ्क cc = vhi; cc < Wp_a; ++cc) {
                row[cc] = 0.0f;
            }
        }
    }

    /* Flush this buffer's L1D-dirty lines down to L2SCP backing. */
    FENCE;
    flush_range_गन्तव्यम्_l2((ध्रुव शून्य *) ctx->l2_pad_अन्तर्_buf[buf_idx], ctx->pin_chunk_अष्टकानि);
    WAIT_CACHEOPS;
}

पूर्णाङ्क प्रविष्टिः_point(संरचना ggml_et_द्विमान_params * params, शून्य * env) {
    (शून्य) env;

    ध्रुव पूर्णाङ्क shire   = get_shire_id();
    ध्रुव पूर्णाङ्क hart_id = get_hart_id();
    ध्रुव पूर्णाङ्क minion  = (hart_id >> 1) & 0x1F;
    ध्रुव पूर्णाङ्क hart1   = hart_id & 1;

    ध्रुव संरचना ggml_tensor * flt = &params->src0; /* [Kw,Kh,Cin,Cout] */
    ध्रुव संरचना ggml_tensor * अन्तर्  = &params->src1; /* [W, H, Cin,N=1 ] */
    संरचना ggml_tensor *       बहिर्गमनम् = &params->dst;  /* [W, H, Cout,N=1] */

    ध्रुव पूर्णाङ्क Kw   = (पूर्णाङ्क) flt->ne[0];
    ध्रुव पूर्णाङ्क Kh   = (पूर्णाङ्क) flt->ne[1];
    ध्रुव पूर्णाङ्क Cin  = (पूर्णाङ्क) flt->ne[2];
    ध्रुव पूर्णाङ्क Cout = (पूर्णाङ्क) flt->ne[3];

    ध्रुव पूर्णाङ्क W  = (पूर्णाङ्क) अन्तर्->ne[0];
    ध्रुव पूर्णाङ्क H  = (पूर्णाङ्क) अन्तर्->ne[1];
    ध्रुव पूर्णाङ्क OW = (पूर्णाङ्क) बहिर्गमनम्->ne[0];
    ध्रुव पूर्णाङ्क OH = (पूर्णाङ्क) बहिर्गमनम्->ne[1];

    /* op_params layout (set by ggml_conv_2d):
         [0]=s0  [1]=s1  [2]=p0  [3]=p1  [4]=d0  [5]=d1   */
    ध्रुव पूर्णाङ्क s0    = बहिर्गमनम्->op_params[0];
    ध्रुव पूर्णाङ्क s1    = बहिर्गमनम्->op_params[1];
    ध्रुव पूर्णाङ्क pad_w = बहिर्गमनम्->op_params[2];
    ध्रुव पूर्णाङ्क pad_h = बहिर्गमनम्->op_params[3];

    यदि (Cin <= 0 || Cout <= 0) {
        प्रत्यागम -1;
    }
    यदि (Cin % TILE != 0 || Cout % TILE != 0) {
        प्रत्यागम -1;
    }
    यदि (W <= 0 || H <= 0) {
        प्रत्यागम -1;
    }
    यदि (s0 <= 0 || s1 <= 0) {
        प्रत्यागम -1;
    }
    यदि (अन्तर्->ne[2] != Cin || अन्तर्->ne[3] != 1) {
        प्रत्यागम -1;
    }
    यदि (बहिर्गमनम्->ne[2] != Cout || बहिर्गमनम्->ne[3] != 1) {
        प्रत्यागम -1;
    }
    यदि (!flt->दत्त || !अन्तर्->दत्त || !बहिर्गमनम्->दत्त) {
        प्रत्यागम -1;
    }

    ध्रुव पूर्णाङ्क K_TILES = Cin / TILE;
    ध्रुव पूर्णाङ्क M_TILES = Cout / TILE;

    ध्रुव पूर्णाङ्क  Hp         = H + 2 * pad_h;
    ध्रुव पूर्णाङ्क  Wp_a       = आवर्तनम्_ऊर्ध्वम्_tile_i32(OW);
    ध्रुव पूर्णाङ्क  OW_pad     = Wp_a;
    ध्रुव bool need_stage = (OW % TILE != 0);

    /* ===================== Tile assignment & active-shire selection =====
       Computed up front because the per-shire mt set (and thus filter
       region size) depends on n_active_shires. */
    ध्रुव पूर्णाङ्क w_tiles         = ceil_div_i32(OW, TILE);
    ध्रुव पूर्णाङ्क योग_tiles     = OH * w_tiles * M_TILES;
    ध्रुव पूर्णाङ्क n_सक्रिय_shires = need_stage ? 1 : min_i32(योग_tiles, N_SHIRES);

    /* Inactive shires exit immediately. No global barrier — pack and
       barriers are now per-shire, so unused shires don't need to vote. */
    यदि (shire >= n_सक्रिय_shires) {
        प्रत्यागम 0;
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
    पूर्णाङ्क my_mt[MAX_MY_MT];
    पूर्णाङ्क n_my_mt = 0;
    प्रति (पूर्णाङ्क t = shire; t < योग_tiles; t += n_सक्रिय_shires) {
        ध्रुव पूर्णाङ्क mt    = t % M_TILES;
        bool      प्राप्तम् = false;
        प्रति (पूर्णाङ्क j = 0; j < n_my_mt; ++j) {
            यदि (my_mt[j] == mt) {
                प्राप्तम् = true;
                भङ्ग;
            }
        }
        यदि (!प्राप्तम्) {
            यदि (n_my_mt >= MAX_MY_MT) {
                प्रत्यागम -1;
            }
            my_mt[n_my_mt++] = mt;
        }
    }
    यदि (n_my_mt == 0) {
        प्रत्यागम 0; /* no tiles for this shire */
    }

    ध्रुव uint64_t filter_local_अष्टकानि = (uint64_t) Kh * Kw * n_my_mt * K_TILES * SLAB_अष्टकानि;
    यदि (filter_local_अष्टकानि > LOCAL_FILTER_CAP) {
        प्रत्यागम -1;
    }

    /* ===================== L2 SCP local layout =========================
       filter (this shire's mt slice) | pin_buf[0] | pin_buf[1]?
                                      | output_stage? | scratch (streaming) */
    ध्रुव uint64_t l2_आधारः   = (uint64_t) et_shire_l2scp_local(0);
    ध्रुव uint64_t l2_filter = l2_आधारः;

    /* Sizing for pin: budget = LOCAL_BUDGET - filter - output_stage.    */
    ध्रुव int64_t output_stage_अष्टकानि_पूर्णम् = need_stage ? (int64_t) Cout * OH * OW_pad * (int64_t) परिमाण(भिन्न) : 0;
    ध्रुव int64_t budget_for_chunks = (int64_t) LOCAL_BUDGET - (int64_t) filter_local_अष्टकानि - output_stage_अष्टकानि_पूर्णम्;
    यदि (budget_for_chunks <= 0) {
        प्रत्यागम -1;
    }
    ध्रुव int64_t प्रत्येकम्_KT_pin_अष्टकानि = (int64_t) Kw * TILE * Hp * Wp_a * (int64_t) परिमाण(भिन्न);

    पूर्णाङ्क chunk_KT;
    पूर्णाङ्क n_buffers;
    यदि ((int64_t) K_TILES * प्रत्येकम्_KT_pin_अष्टकानि <= budget_for_chunks) {
        chunk_KT  = K_TILES;
        n_buffers = 1;
    } अन्यथा {
        chunk_KT = K_TILES;
        यावत् (chunk_KT > 1 && 2 * (int64_t) chunk_KT * प्रत्येकम्_KT_pin_अष्टकानि > budget_for_chunks) {
            chunk_KT--;
        }
        यावत् (chunk_KT > 1 && K_TILES % chunk_KT != 0) {
            chunk_KT--;
        }
        n_buffers = (chunk_KT < K_TILES) ? 2 : 1;
        यदि (chunk_KT < 1) {
            प्रत्यागम -1;
        }
    }
    ध्रुव पूर्णाङ्क n_chunks = K_TILES / chunk_KT;

    /* Streaming keeps partial sums in MAX_TILES_PER_HART scratch slots per
       hart. The one-chunk path does not need scratch and can stream a longer
       tile list serially, but multi-chunk shapes must fit this fixed slot
       count until scratch scheduling is made more general. */
    ध्रुव पूर्णाङ्क shire_tile_क्षमता = shire + MAX_TILES_प्रत्येकम्_HART * n_सक्रिय_shires * N_MIN_प्रत्येकम्_SHIRE;
    यदि (n_chunks > 1 && shire_tile_क्षमता < योग_tiles) {
        प्रत्यागम -1;
    }

    ध्रुव uint64_t pin_copy_floats = (uint64_t) chunk_KT * TILE * Hp * Wp_a;
    ध्रुव uint64_t pin_copy_अष्टकानि  = pin_copy_floats * परिमाण(भिन्न);
    ध्रुव uint64_t pin_chunk_अष्टकानि = (uint64_t) Kw * pin_copy_अष्टकानि;

    ध्रुव uint64_t l2_pin_आधारः              = l2_filter + filter_local_अष्टकानि;
    ध्रुव uint64_t l2_pin_buf[MAX_DBL_BUFS] = {
        l2_pin_आधारः,
        l2_pin_आधारः + pin_chunk_अष्टकानि,
    };

    ध्रुव uint64_t l2_output_stage = need_stage ? l2_pin_आधारः + (uint64_t) n_buffers * pin_chunk_अष्टकानि : 0;

    ध्रुव uint64_t scratch_प्रत्येकम्_hart = (uint64_t) MAX_TILES_प्रत्येकम्_HART * (uint64_t) TILE * TILE * परिमाण(भिन्न);
    ध्रुव uint64_t l2_scratch_आधारः  = need_stage ? l2_output_stage + (uint64_t) output_stage_अष्टकानि_पूर्णम् :
                                                   l2_pin_आधारः + (uint64_t) n_buffers * pin_chunk_अष्टकानि;

    /* ===================== PHASE 1: Filter pack (per-shire mt slice) ====
       Hart-1's pack only this shire's mt slabs into local L2 SCP. The
       SHIRE barrier below ensures the filter is in L2 SCP backing before
       hart-0's first tensor_load. */
    यदि (hart1) {
        pack_filter_local_mt((ध्रुव भिन्न *) flt->दत्त, Kh, Kw, Cin, K_TILES, my_mt, n_my_mt, minion, l2_filter);
    }

    /* ===================== Hart 1: pin packer (per chunk) ==============
       Double-buffered prefetch: pack chunk 0 synchronously, then per chunk c
       signal "buf c ready", pack chunk c+1 into the alternate buffer
       (overlaps hart-0's compute on c), signal "buf c done". */
    यदि (hart1) {
        ध्रुव pin_ctx_t ctx = {
            .अन्तर्_आधारः         = (ध्रुव भिन्न *) अन्तर्->दत्त,
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
            .l2_pad_अन्तर्_buf   = { l2_pin_buf[0], l2_pin_buf[1] },
            .pin_chunk_अष्टकानि = pin_chunk_अष्टकानि,
        };

        pack_pin_chunk(&ctx, 0, 0); /* prologue */

        प्रति (पूर्णाङ्क c = 0; c < n_chunks; ++c) {
            et_barrier(ET_BARRIER_SHIRE); /* signal "buf c ready" */
            यदि (n_buffers > 1 && c + 1 < n_chunks) {
                pack_pin_chunk(&ctx, c + 1, (c + 1) & 1);
            }
            et_barrier(ET_BARRIER_SHIRE); /* wait "buf c done" */
        }

        यदि (need_stage) {
            et_barrier(ET_BARRIER_SHIRE);
        }
        प्रत्यागम 0;
    }

    /* ===================== Hart 0: matrix engine ======================
       Two execution modes:
         - n_chunks == 1: full Cin in one shot. Each hart processes a list
           of tiles serially; TenC resets between tiles via first_pass=true.
         - n_chunks  > 1: streaming. Each hart owns up to MAX_TILES_PER_HART
           tiles. For each chunk c, restore TenC from scratch[k] (skip on
           c==0), accumulate this chunk's FMAs, then either save TenC back
           to scratch[k] (c < last) or tensor_store directly (c == last). */
    सज्जता_cache_scp();
    CLEAR_TENSOR_ERROR;

    वर्ण * ध्रुव   बहिर्गमनम्_आधारः        = need_stage ? (वर्ण *) l2_output_stage : (वर्ण *) बहिर्गमनम्->दत्त;
    ध्रुव पूर्णाङ्क      compute_OW      = need_stage ? OW_pad : OW;
    ध्रुव uint64_t बहिर्गमनम्_chan_stride = (uint64_t) OH * (uint64_t) compute_OW * परिमाण(भिन्न);
    ध्रुव uint64_t बहिर्गमनम्_row_stride  = (uint64_t) compute_OW * परिमाण(भिन्न);

    ध्रुव uint64_t a_row_stride = (uint64_t) TILE * परिमाण(भिन्न); /* 64 */
    ध्रुव uint64_t b_row_stride = (uint64_t) Hp * (uint64_t) Wp_a * परिमाण(भिन्न);

    /* Tile assignment: shire-strided so small workloads spread across
       shires before stacking minions in one shire. */
    ध्रुव पूर्णाङ्क t_start  = shire + minion * n_सक्रिय_shires;
    ध्रुव पूर्णाङ्क t_stride = n_सक्रिय_shires * N_MIN_प्रत्येकम्_SHIRE;

    यदि (n_chunks == 1) {
        et_barrier(ET_BARRIER_SHIRE); /* wait for the (only) pin chunk */

        ध्रुव uint64_t l2_pad_अन्तर् = l2_pin_buf[0];
        प्रति (पूर्णाङ्क t = t_start; t < योग_tiles; t += t_stride) {
            ध्रुव conv_tile_t tile = decode_tile(t, M_TILES, w_tiles, my_mt, n_my_mt);
            compute_tile_chunk(l2_filter, l2_pad_अन्तर्, pin_copy_अष्टकानि, Kh, Kw, K_TILES, chunk_KT, 0, n_my_mt, Hp, Wp_a,
                               s1, a_row_stride, b_row_stride, &tile, /*first_fma_clears_tenc=*/true);

            वर्ण * dst_addr = output_tile_addr(बहिर्गमनम्_आधारः, &tile, बहिर्गमनम्_chan_stride, बहिर्गमनम्_row_stride);
            tensor_store(0, 0, 3, (uint64_t) (TILE - 1), (uint64_t) dst_addr, 0, बहिर्गमनम्_chan_stride);
            tensor_wait(TENSOR_STORE_WAIT);
        }

        et_barrier(ET_BARRIER_SHIRE); /* matches hart-1's second barrier */

    } अन्यथा {
        /* Streaming path: each hart owns up to MAX_TILES_PER_HART tiles. */
        पूर्णाङ्क my_tiles[MAX_TILES_प्रत्येकम्_HART];
        पूर्णाङ्क n_my_tiles = 0;
        प्रति (पूर्णाङ्क slot = 0; slot < MAX_TILES_प्रत्येकम्_HART; ++slot) {
            ध्रुव पूर्णाङ्क t = t_start + slot * t_stride;
            यदि (t < योग_tiles) {
                my_tiles[n_my_tiles++] = t;
            }
        }

        conv_tile_t tiles[MAX_TILES_प्रत्येकम्_HART];
        प्रति (पूर्णाङ्क k = 0; k < n_my_tiles; ++k) {
            tiles[k] = decode_tile(my_tiles[k], M_TILES, w_tiles, my_mt, n_my_mt);
        }

        ध्रुव uint64_t my_scratch_आधारः = l2_scratch_आधारः + (uint64_t) minion * scratch_प्रत्येकम्_hart;

        प्रति (पूर्णाङ्क c = 0; c < n_chunks; ++c) {
            et_barrier(ET_BARRIER_SHIRE); /* pin chunk c packed */

            ध्रुव पूर्णाङ्क      buf       = c & 1;
            ध्रुव uint64_t l2_pad_अन्तर् = l2_pin_buf[buf];
            ध्रुव पूर्णाङ्क      kt_आधारः   = c * chunk_KT;

            प्रति (पूर्णाङ्क k = 0; k < n_my_tiles; ++k) {
                ध्रुव conv_tile_t * tile = &tiles[k];
                ध्रुव uint64_t      scr  = my_scratch_आधारः + (uint64_t) k * (TILE * TILE * परिमाण(भिन्न));

                ध्रुव bool first_pass_chunk = (c == 0);
                यदि (!first_pass_chunk) {
                    tenc_restore_from_scratch(scr);
                }

                compute_tile_chunk(l2_filter, l2_pad_अन्तर्, pin_copy_अष्टकानि, Kh, Kw, K_TILES, chunk_KT, kt_आधारः, n_my_mt,
                                   Hp, Wp_a, s1, a_row_stride, b_row_stride, tile, first_pass_chunk);

                यदि (c == n_chunks - 1) {
                    वर्ण * dst_addr = output_tile_addr(बहिर्गमनम्_आधारः, tile, बहिर्गमनम्_chan_stride, बहिर्गमनम्_row_stride);
                    tensor_store(0, 0, 3, (uint64_t) (TILE - 1), (uint64_t) dst_addr, 0, बहिर्गमनम्_chan_stride);
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
    यदि (need_stage) {
        et_barrier(ET_BARRIER_SHIRE);

        यदि (minion == 0) {
            ध्रुव भिन्न * stage = (ध्रुव भिन्न *) l2_output_stage;
            भिन्न *       dram  = (भिन्न *) बहिर्गमनम्->दत्त;
            प्रति (पूर्णाङ्क oc = 0; oc < Cout; ++oc) {
                प्रति (पूर्णाङ्क oh2 = 0; oh2 < OH; ++oh2) {
                    ध्रुव भिन्न * src = stage + ((size_t) oc * OH + oh2) * OW_pad;
                    भिन्न *       dst = dram + ((size_t) oc * OH + oh2) * OW;
                    प्रति (पूर्णाङ्क ow2 = 0; ow2 < OW; ++ow2) {
                        dst[ow2] = src[ow2];
                    }
                }
            }
            FENCE;
            ध्रुव uint64_t योग_अष्टकानि = (uint64_t) Cout * OH * OW * परिमाण(भिन्न);
            evict_range_past_l2((ध्रुव शून्य *) dram, योग_अष्टकानि);
            WAIT_CACHEOPS;
        }
    }

    प्रत्यागम 0;
}
