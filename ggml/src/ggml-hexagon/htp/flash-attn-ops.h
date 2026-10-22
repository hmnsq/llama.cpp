#جيڪڏهن_وصف_نه_ٿيل HTP_FLASH_ATTN_OPS_H
#وصف_ڪريو HTP_FLASH_ATTN_OPS_H

#ضمن <stdint.h>
#ضمن <stddef.h>
#ضمن <stdbool.h>

#ضمن "hex-fastdiv.h"
#ضمن "hex-common.h"
#ضمن "htp-vtcm.h"

#جيڪڏهن_وصف_ٿيل __cplusplus
خارجي "C" {
#جيڪڏهن_پورو

// Tile constants (mirrored from hmx-utils.h for use on host side if needed)
#وصف_ڪريو HTP_FA_HMX_TILE_سائيز   2048
#وصف_ڪريو HMX_FP16_TILE_سائيز     2048
#وصف_ڪريو HMX_FP16_TILE_N_صفون   32
#وصف_ڪريو HMX_FP16_TILE_N_کانن_ڳڻپ   32
#وصف_ڪريو HMX_FP16_TILE_N_ELMS   1024

#وصف_ڪريو HVX_FA_DMA_CACHE_سائيز  128
#وصف_ڪريو HMX_FA_DMA_CACHE_سائيز  4


#وصف_ڪريو HTP_FA_M_INITIAL_VAL  -10000.0f

تعداد htp_fa_kernel_type {
    HTP_FA_KERNEL_UNSUPPORTED = 0,
    HTP_FA_KERNEL_HVX,
    HTP_FA_KERNEL_HMX
};

بنية htp_fa_kernel_params {
    uint8_t  kernel_type;        // enum htp_fa_kernel_type
    uint8_t  سر_ورهايو;         // 1 = partition by KV heads in multicore, 0 = token partition
    uint8_t  flags;              // reserved
    uint8_t  n_ڌاڳا;          // Number of threads to run

    // Common parameters
    uint16_t Br;
    uint16_t Bc;
    uint16_t n_kv_بلاڪون;        // also HVX's n_blocks
    uint16_t G;                  // GQA factor (n_heads / n_kv_heads)

    عائم    scale;
    عائم    max_bias;
    عائم    logit_softcap;
    uint32_t vtcm_سائيز;

    uint32_t qrows;
    uint32_t qrows_في_thread;
    uint32_t qrow_شروعات;
    عائم    m0;
    عائم    m1;
    uint32_t n_سر_log2;

    بنية fastdiv_قدرون src3_div2;
    بنية fastdiv_قدرون src3_div3;

    بنية fastdiv_قدرون نشر_rk2;
    بنية fastdiv_قدرون نشر_rk3;
    بنية fastdiv_قدرون نشر_rv2;
    بنية fastdiv_قدرون نشر_rv3;

    اتحاد {
        بنية {
            uint32_t g_br;
            uint32_t صف_buf_stride;
            uint32_t نقاب_buf_صف_stride;
            int32_t  نقاب_نشر;
            int32_t  pipeline;
            بنية fastdiv_قدرون div_G;
        } hmx;
        بنية {
            uint32_t سائيز_q_صف_padded;
            uint32_t سائيز_k_صف_padded;
            uint32_t سائيز_v_صف_padded;
            بنية fastdiv_قدرون src0_div21;
            بنية fastdiv_قدرون src0_div1;
        } hvx;
    } u;
};

#إذا defined(__cplusplus)
static_assert(حجم(بنية htp_fa_kernel_params) <= 128, "htp_fa_kernel_params is too large for kernel_params blob");
#جيڪڏهن_پورو

// VTCM region layout for the HMX flash-attention kernel.
//
// Single source of truth for both the host (which needs the total size to pick a
// (Br, Bc) tiling that fits the VTCM budget) and the device (which needs the actual
// byte offsets to place each scratch buffer). Building the layout once and reading
// offsets/total from it makes host estimate and device allocation impossible to
// desync -- previously they were duplicated formulas in two files and drifted.
//
// All fields are byte offsets / byte sizes -- no HVX_Vector type is named here so the
// header stays host-includable. The device casts (base + off_*) to the proper type.
// An offset of 0 marks a region that is not allocated for this configuration (only
// off_v_tiles[1], which exists only when pipelining); the device sets such pointers NULL.
بنية hmx_fa_vtcm_layout {
    // Byte offsets from vtcm_base for each region.
    size_t off_q_tiles;
    size_t off_q_dma;
    size_t off_o_tiles[2];
    size_t off_k_fp16[2];
    size_t off_v_fp16[2];
    size_t off_k_tiles[2];
    size_t off_v_tiles[2];
    size_t off_s_tiles[2];
    size_t off_p_tiles[2];
    size_t off_d_tiles[2];
    size_t off_d_inv_l;
    size_t off_m_vec;
    size_t off_l_vec;
    size_t off_s_rowmax;
    size_t off_p_rowsum;
    size_t off_صف_bufs;
    size_t off_hmx_scales_id;
    size_t off_hmx_scales_qk;
    size_t off_نقاب_buf;
    size_t off_slopes;
    size_t off_sinks;

    // Region byte sizes reused by the device at runtime (not just for allocation).
    size_t q_tile_بائيٽون;
    size_t o_tile_بائيٽون;
    size_t k_tile_بائيٽون;
    size_t s_tile_بائيٽون;       // S and P tiles (same size)
    size_t d_tile_بائيٽون;       // d_tiles[0..1] + d_inv_l, allocated back to back
    size_t m_سٽ_بائيٽون;       // one mask row
    size_t m_buf_جڳهه_بائيٽون;   // one dma_cache slot = align_up(Br * m_line_bytes, 4096)
    size_t کانو_vec_بائيٽون;
    size_t sinks_بائيٽون;

    // Derived strides.
    size_t صف_buf_stride;       // HVX vectors (128B) per row buffer
    size_t نقاب_buf_صف_stride;  // __fp16 elements per row in the mask buffer

    bool   pipeline;
    size_t ڪل_بائيٽون;
};

#وصف_ڪريو FA_MAX_KV_بلاڪون 1024

// Build the VTCM layout.

ساكن مضمن فراغ hmx_fa_vtcm_layout_build(بنية hmx_fa_vtcm_layout * L,
                                            size_t gqa_ضرب_ڪندڙ, size_t DK, size_t DV,
                                            size_t Br, size_t Bc, size_t n_ڌاڳا,
                                            bool pipeline, bool is_q_fp32,
                                            bool has_sinks, size_t n_heads) {
    ثابت size_t g_br         = hex_align_مٿي(gqa_ضرب_ڪندڙ * Br, HMX_FP16_TILE_N_صفون);
    ثابت size_t q_tile_سائيز  = hex_align_مٿي(g_br * DK   * حجم(__fp16), HTP_FA_HMX_TILE_سائيز);
    ثابت size_t o_tile_سائيز  = hex_align_مٿي(g_br * DV   * حجم(__fp16), HTP_FA_HMX_TILE_سائيز);
    ثابت size_t k_tile_سائيز  = hex_align_مٿي(Bc   * DK   * حجم(__fp16), HTP_FA_HMX_TILE_سائيز);
    ثابت size_t v_tile_سائيز  = hex_align_مٿي(Bc   * DV   * حجم(__fp16), HTP_FA_HMX_TILE_سائيز);
    ثابت size_t s_tile_سائيز  = hex_align_مٿي(g_br * Bc   * حجم(__fp16), HTP_FA_HMX_TILE_سائيز);

    // The rescale matrices are diagonal: the HMX kernels only ever load the g_br/32
    // tiles that sit on the diagonal, so store just those, packed back to back with
    // a stride of one tile.  The old [g_br, g_br] square layout allocated g_br/32
    // times more than it used, which is also why a second D buffer was unaffordable.
    ثابت size_t d_tile_سائيز = (g_br / HMX_FP16_TILE_N_صفون) * HTP_FA_HMX_TILE_سائيز;

    ثابت size_t q_dma_سائيز   = hex_align_مٿي(g_br * DK * (is_q_fp32 ? حجم(عائم) : حجم(__fp16)), 128);
    ثابت size_t k_dma_سائيز   = hex_align_مٿي(Bc * hex_ڦيرو_مٿي(DK * حجم(__fp16), 128), 128);
    ثابت size_t v_dma_سائيز   = hex_align_مٿي(Bc * hex_ڦيرو_مٿي(DV * حجم(__fp16), 128), 128);
    ثابت size_t کانو_vec_سائيز = hex_align_مٿي(g_br * حجم(عائم),  256);
    ثابت size_t صف_vec_سائيز = hex_align_مٿي(Bc   * حجم(__fp16), 256);
    ثابت size_t m_سٽ_سائيز  = hex_align_مٿي(Bc   * حجم(__fp16), 128);
    ثابت size_t m_buf_جڳهه   = hex_align_مٿي(Br * m_سٽ_سائيز, 256);
    ثابت size_t m_buf_سائيز   = m_buf_جڳهه * HMX_FA_DMA_CACHE_سائيز;
    ثابت size_t slopes_سائيز  = hex_align_مٿي(g_br * حجم(__fp16), 128);
    ثابت size_t sinks_سائيز   = hex_ڦيرو_مٿي(n_heads * حجم(عائم), 128);

    size_t off = 0;

    // Group A (Part 1 - HMX Tiled buffers)
    VTCM_LAYOUT_ALLOC(off, off_q_tiles,       q_tile_سائيز);
    VTCM_LAYOUT_ALLOC(off, off_o_tiles[0],    o_tile_سائيز);
    VTCM_LAYOUT_ALLOC(off, off_o_tiles[1],    o_tile_سائيز);
    VTCM_LAYOUT_ALLOC(off, off_d_tiles[0],    d_tile_سائيز);
    VTCM_LAYOUT_ALLOC_OPTIONAL(off, off_d_tiles[1], d_tile_سائيز, pipeline);
    VTCM_LAYOUT_ALLOC(off, off_d_inv_l,       d_tile_سائيز);

    // Group B & C share start offset (Group B tiles must be 2KB aligned)
    size_t off_گروهه_b_c = hex_align_مٿي(off, HTP_FA_HMX_TILE_سائيز);

    // Group B: Compute-only buffers
    size_t off_گروهه_b = off_گروهه_b_c;
    VTCM_LAYOUT_ALLOC(off_گروهه_b, off_k_tiles[0],    k_tile_سائيز);
    VTCM_LAYOUT_ALLOC_OPTIONAL(off_گروهه_b, off_k_tiles[1], k_tile_سائيز, pipeline);
    VTCM_LAYOUT_ALLOC(off_گروهه_b, off_v_tiles[0],    v_tile_سائيز);
    VTCM_LAYOUT_ALLOC_OPTIONAL(off_گروهه_b, off_v_tiles[1], v_tile_سائيز, pipeline);
    VTCM_LAYOUT_ALLOC(off_گروهه_b, off_s_tiles[0],    s_tile_سائيز);
    VTCM_LAYOUT_ALLOC_OPTIONAL(off_گروهه_b, off_s_tiles[1], s_tile_سائيز, pipeline);
    VTCM_LAYOUT_ALLOC(off_گروهه_b, off_p_tiles[0],    s_tile_سائيز);
    VTCM_LAYOUT_ALLOC_OPTIONAL(off_گروهه_b, off_p_tiles[1], s_tile_سائيز, pipeline);
    VTCM_LAYOUT_ALLOC(off_گروهه_b, off_s_rowmax,      کانو_vec_سائيز);
    VTCM_LAYOUT_ALLOC(off_گروهه_b, off_p_rowsum,      کانو_vec_سائيز);
    VTCM_LAYOUT_ALLOC(off_گروهه_b, off_صف_bufs,      صف_vec_سائيز * 2 * n_ڌاڳا);

    ثابت size_t گروهه_b_سائيز = off_گروهه_b - off_گروهه_b_c;

    // Group C: Q fetch DMA buffer
    size_t off_گروهه_c = off_گروهه_b_c;
    VTCM_LAYOUT_ALLOC(off_گروهه_c, off_q_dma,         q_dma_سائيز);

    ثابت size_t گروهه_c_سائيز = off_گروهه_c - off_گروهه_b_c;

    off = off_گروهه_b_c + hex_smax(گروهه_b_سائيز, گروهه_c_سائيز);

    // Group A (Part 2 - remaining non-HMX buffers)
    VTCM_LAYOUT_ALLOC(off, off_k_fp16[0],     k_dma_سائيز);
    VTCM_LAYOUT_ALLOC(off, off_k_fp16[1],     k_dma_سائيز);
    VTCM_LAYOUT_ALLOC(off, off_v_fp16[0],     v_dma_سائيز);
    VTCM_LAYOUT_ALLOC(off, off_v_fp16[1],     v_dma_سائيز);
    VTCM_LAYOUT_ALLOC(off, off_m_vec,         کانو_vec_سائيز);
    VTCM_LAYOUT_ALLOC(off, off_l_vec,         کانو_vec_سائيز);
    VTCM_LAYOUT_ALLOC(off, off_hmx_scales_id, 256);
    VTCM_LAYOUT_ALLOC(off, off_hmx_scales_qk, 256);
    VTCM_LAYOUT_ALLOC(off, off_نقاب_buf,      m_buf_سائيز);
    VTCM_LAYOUT_ALLOC(off, off_slopes,        slopes_سائيز);
    VTCM_LAYOUT_ALLOC_OPTIONAL(off, off_sinks, sinks_سائيز, has_sinks);

    L->q_tile_بائيٽون        = q_tile_سائيز;
    L->o_tile_بائيٽون        = o_tile_سائيز;
    L->k_tile_بائيٽون        = k_tile_سائيز;
    L->کانو_vec_بائيٽون       = کانو_vec_سائيز;
    L->s_tile_بائيٽون        = s_tile_سائيز;
    // Measured from the actual offsets rather than assumed to be N * d_tile_size, so
    // that inserting a region between them (or adding padding to VTCM_LAYOUT_ALLOC)
    // cannot silently leave the tail of the run unzeroed.
    L->d_tile_بائيٽون        = (L->off_d_inv_l + d_tile_سائيز) - L->off_d_tiles[0];
    L->m_سٽ_بائيٽون        = m_سٽ_سائيز;
    L->m_buf_جڳهه_بائيٽون    = m_buf_جڳهه;
    L->صف_buf_stride      = صف_vec_سائيز / 128;
    L->نقاب_buf_صف_stride = m_سٽ_سائيز / حجم(__fp16);
    L->sinks_بائيٽون         = has_sinks ? sinks_سائيز : 0;
    L->pipeline            = pipeline;
    L->ڪل_بائيٽون         = off;
}

// Exact VTCM usage for a given (gqa_factor, DK, DV, Br, Bc) configuration.
ساكن مضمن size_t hmx_fa_compute_vtcm_usage(size_t gqa_ضرب_ڪندڙ, size_t DK, size_t DV, size_t Br, size_t Bc, size_t n_ڌاڳا, bool pipeline, bool is_q_fp32, bool has_sinks, size_t n_heads) {
    بنية hmx_fa_vtcm_layout L;
    hmx_fa_vtcm_layout_build(&L, gqa_ضرب_ڪندڙ, DK, DV, Br, Bc, n_ڌاڳا, pipeline, is_q_fp32, has_sinks, n_heads);
    أرجع L.ڪل_بائيٽون;
}

#وصف_ڪريو FA_HVX_بلاڪ_سائيز 64
#وصف_ڪريو FA_HVX_G_MAX      8

بنية hvx_fa_vtcm_layout {
    size_t off_q;
    size_t off_k;
    size_t off_v;
    size_t off_m;
    size_t off_a;
    size_t off_sinks;

    size_t سائيز_q_بلاڪ;
    size_t سائيز_k_بلاڪ;
    size_t سائيز_v_بلاڪ;
    size_t سائيز_m_بلاڪ;
    size_t سائيز_vkq_acc;
    size_t سائيز_sinks;

    size_t ڪل_بائيٽون;
};

ساكن مضمن فراغ hvx_fa_vtcm_layout_build(بنية hvx_fa_vtcm_layout * L,
                                            size_t DK, size_t DV,
                                            bool is_q_fp32, bool has_نقاب,
                                            bool has_sinks, size_t n_heads,
                                            size_t n_ڌاڳا) {
    ثابت size_t سائيز_q_صف_padded = hex_ڦيرو_مٿي(DK * (is_q_fp32 ? 4 : 2), 128);
    ثابت size_t سائيز_k_صف_padded = hex_ڦيرو_مٿي(DK * حجم(__fp16), 128);
    ثابت size_t سائيز_v_صف_padded = hex_ڦيرو_مٿي(DV * حجم(__fp16), 128);

    ثابت size_t سائيز_q_بلاڪ = سائيز_q_صف_padded * FA_HVX_G_MAX;
    ثابت size_t سائيز_k_بلاڪ = سائيز_k_صف_padded * FA_HVX_بلاڪ_سائيز;
    ثابت size_t سائيز_v_بلاڪ = سائيز_v_صف_padded * FA_HVX_بلاڪ_سائيز;
    ثابت size_t سائيز_m_بلاڪ = hex_ڦيرو_مٿي(FA_HVX_بلاڪ_سائيز * حجم(__fp16), 128);
    ثابت size_t سائيز_vkq_acc = hex_ڦيرو_مٿي(DV * حجم(عائم), 128) * FA_HVX_G_MAX;
    ثابت size_t سائيز_sinks   = hex_ڦيرو_مٿي(n_heads * حجم(عائم), 128);

    size_t off = 0;

    VTCM_LAYOUT_ALLOC(off, off_q, سائيز_q_بلاڪ * n_ڌاڳا);
    VTCM_LAYOUT_ALLOC(off, off_k, سائيز_k_بلاڪ * 2 * n_ڌاڳا);
    VTCM_LAYOUT_ALLOC(off, off_v, سائيز_v_بلاڪ * 2 * n_ڌاڳا);
    VTCM_LAYOUT_ALLOC_OPTIONAL(off, off_m, سائيز_m_بلاڪ * HVX_FA_DMA_CACHE_سائيز * n_ڌاڳا, has_نقاب);
    VTCM_LAYOUT_ALLOC(off, off_a, سائيز_vkq_acc * n_ڌاڳا);
    VTCM_LAYOUT_ALLOC_OPTIONAL(off, off_sinks, سائيز_sinks, has_sinks);

    L->سائيز_q_بلاڪ = سائيز_q_بلاڪ;
    L->سائيز_k_بلاڪ = سائيز_k_بلاڪ;
    L->سائيز_v_بلاڪ = سائيز_v_بلاڪ;
    L->سائيز_m_بلاڪ = سائيز_m_بلاڪ;
    L->سائيز_vkq_acc = سائيز_vkq_acc;
    L->سائيز_sinks   = has_sinks ? سائيز_sinks : 0;
    L->ڪل_بائيٽون  = off;
}

ساكن مضمن size_t hvx_fa_compute_vtcm_usage(size_t DK, size_t DV, bool is_q_fp32, bool has_نقاب, bool has_sinks, size_t n_heads, size_t n_ڌاڳا) {
    بنية hvx_fa_vtcm_layout L;
    hvx_fa_vtcm_layout_build(&L, DK, DV, is_q_fp32, has_نقاب, has_sinks, n_heads, n_ڌاڳا);
    أرجع L.ڪل_بائيٽون;
}

#وصف_ڪريو FA_MIN_KV_بلاڪون 3

// Cost-based (Br, Bc) search for flash attention with pipeline constraint.
ساكن مضمن صحيح hmx_fa_ڳوليو_chunk_سائيز(size_t * Br_ٻاهريون,
                                         size_t * Bc_ٻاهريون,
                                         size_t   gqa_ضرب_ڪندڙ,
                                         size_t   DK,
                                         size_t   DV,
                                         size_t   qo_len,
                                         size_t   kv_len,
                                         size_t   vtcm_budget,
                                         size_t   n_ڌاڳا,
                                         bool     is_q_fp32,
                                         bool     has_sinks,
                                         size_t   n_heads) {
    ثابت size_t T       = HMX_FP16_TILE_N_صفون;  // 32
    ثابت size_t br_unit = hmx_ceil_div(T, gqa_ضرب_ڪندڙ);
    ثابت size_t bc_unit = HMX_FP16_TILE_N_کانن_ڳڻپ * 2;  // 64
    ثابت bool   can_pipeline = (kv_len >= FA_MIN_KV_بلاڪون * bc_unit && n_ڌاڳا >= 2);

    // Br_max: largest Br aligned to br_unit that does not exceed qo_len.
    ثابت size_t Br_max = qo_len >= br_unit ? hex_align_هيٺ_ڏانهن(qo_len, br_unit) : br_unit;

    // Pipeline constraint: cap Bc so n_kv_blocks >= FA_MIN_KV_BLOCKS.
    // Only relax when kv_len is too short to form enough blocks.
    size_t Bc_حد_بندي     = can_pipeline ? hex_align_هيٺ_ڏانهن(kv_len / FA_MIN_KV_بلاڪون, bc_unit) :
                                         (kv_len >= bc_unit ? hex_align_هيٺ_ڏانهن(kv_len, bc_unit) : bc_unit);
    // For small-Q SWA-sized KV caches (qo_len <= 16, kv_len <= 2048), cap Bc at 256 for block-skipping granularity
    إذا (qo_len <= 16 && kv_len <= 2048 && Bc_حد_بندي > 256) {
        Bc_حد_بندي = 256;
    }
    // Cost coefficients calibrated from profiling
    ثابت size_t c_q_fixed    = 800;   // per-Q-block: q_load + epilogue o_update + o_norm + o_store
    ثابت size_t c_iter_بنياد  = 200;   // per-KV-iter base (HMX dot/update + DMA)
    ثابت size_t c_softmax    = 600;   // per 64-row vector chunk on HVX

    size_t بهترين_لاڳت = SIZE_MAX, بهترين_mn = 0;
    size_t بهترين_Br = 0, بهترين_Bc = 0;

    لكل (size_t Br = Br_max; Br >= br_unit; Br -= br_unit) {
        // Try all Bc candidates from Bc_limit down to bc_unit
        لكل (size_t Bc = Bc_حد_بندي; Bc >= bc_unit; Bc -= bc_unit) {
            size_t vtcm_needed = hmx_fa_compute_vtcm_usage(gqa_ضرب_ڪندڙ, DK, DV, Br, Bc, n_ڌاڳا, can_pipeline, is_q_fp32, has_sinks, n_heads);
            إذا (vtcm_needed <= vtcm_budget) {
                // This Bc fits for this Br!
                ثابت size_t q_بلاڪون       = (qo_len + Br - 1) / Br;
                ثابت size_t kv_بلاڪون      = (kv_len + Bc - 1) / Bc;
                ثابت size_t actual_ڌاڳا = (kv_بلاڪون >= 3 && n_ڌاڳا >= 2) ? n_ڌاڳا : 1;
                ثابت size_t n_صفون_g       = Br * gqa_ضرب_ڪندڙ;
                ثابت size_t n_صف_vec_cnt  = (n_صفون_g + 63) / 64;
                ثابت size_t n_use          = n_صف_vec_cnt < actual_ڌاڳا ? n_صف_vec_cnt : actual_ڌاڳا;
                ثابت size_t vecs_في_t     = n_use > 0 ? (n_صف_vec_cnt + n_use - 1) / n_use : 1;

                ثابت size_t c_iter_actual  = c_iter_بنياد + c_softmax * vecs_في_t;
                ثابت size_t لاڳت           = q_بلاڪون * (c_q_fixed + kv_بلاڪون * c_iter_actual);
                ثابت size_t mn             = Br * Bc;

                إذا (لاڳت < بهترين_لاڳت || (لاڳت == بهترين_لاڳت && mn > بهترين_mn)) {
                    بهترين_لاڳت = لاڳت;
                    بهترين_mn   = mn;
                    بهترين_Br   = Br;
                    بهترين_Bc   = Bc;
                }
                // Since we iterate Bc from largest to smallest, this is the largest Bc that fits
                // for this Br. We can break to the next Br.
                اكسر;
            }
        }

        إذا (Br == br_unit) {
            اكسر;
        }
    }

    إذا (بهترين_Br == 0 || بهترين_Bc == 0) {
        أرجع -1;
    }

    *Br_ٻاهريون = بهترين_Br;
    *Bc_ٻاهريون = بهترين_Bc;
    أرجع 0;
}

#جيڪڏهن_وصف_ٿيل __cplusplus
}
#جيڪڏهن_پورو

#جيڪڏهن_پورو /* HTP_FLASH_ATTN_OPS_H */
