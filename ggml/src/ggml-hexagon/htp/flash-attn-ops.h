#यदि_अपरिभाषितम् HTP_FLASH_ATTN_OPS_H
#परिभाषय HTP_FLASH_ATTN_OPS_H

#अन्तर्भाव <stdint.h>
#अन्तर्भाव <stddef.h>
#अन्तर्भाव <stdbool.h>

#अन्तर्भाव "hex-fastdiv.h"
#अन्तर्भाव "hex-common.h"
#अन्तर्भाव "htp-vtcm.h"

#यदि_परिभाषितम् __cplusplus
बाह्य "C" {
#यदि_अन्तः

// Tile constants (mirrored from hmx-utils.h for use on host side if needed)
#परिभाषय HTP_FA_HMX_TILE_आकार   2048
#परिभाषय HMX_FP16_TILE_आकार     2048
#परिभाषय HMX_FP16_TILE_N_पङ्क्तिकाः   32
#परिभाषय HMX_FP16_TILE_N_स्तम्भसङ्ख्या   32
#परिभाषय HMX_FP16_TILE_N_ELMS   1024

#परिभाषय HVX_FA_DMA_CACHE_आकार  128
#परिभाषय HMX_FA_DMA_CACHE_आकार  4


#परिभाषय HTP_FA_M_INITIAL_VAL  -10000.0f

गणना htp_fa_kernel_type {
    HTP_FA_KERNEL_UNSUPPORTED = 0,
    HTP_FA_KERNEL_HVX,
    HTP_FA_KERNEL_HMX
};

संरचना htp_fa_kernel_params {
    uint8_t  kernel_type;        // enum htp_fa_kernel_type
    uint8_t  शीर्ष_विभज;         // 1 = partition by KV heads in multicore, 0 = token partition
    uint8_t  flags;              // reserved
    uint8_t  n_तन्तवः;          // Number of threads to run

    // Common parameters
    uint16_t Br;
    uint16_t Bc;
    uint16_t n_kv_खण्डानि;        // also HVX's n_blocks
    uint16_t G;                  // GQA factor (n_heads / n_kv_heads)

    भिन्न    scale;
    भिन्न    max_bias;
    भिन्न    logit_softcap;
    uint32_t vtcm_आकार;

    uint32_t qrows;
    uint32_t qrows_प्रत्येकम्_thread;
    uint32_t qrow_start;
    भिन्न    m0;
    भिन्न    m1;
    uint32_t n_शीर्ष_log2;

    संरचना fastdiv_मानानि src3_div2;
    संरचना fastdiv_मानानि src3_div3;

    संरचना fastdiv_मानानि प्रसारणम्_rk2;
    संरचना fastdiv_मानानि प्रसारणम्_rk3;
    संरचना fastdiv_मानानि प्रसारणम्_rv2;
    संरचना fastdiv_मानानि प्रसारणम्_rv3;

    सङ्घ {
        संरचना {
            uint32_t g_br;
            uint32_t row_buf_stride;
            uint32_t आवरणम्_buf_row_stride;
            int32_t  आवरणम्_प्रसारणम्;
            int32_t  pipeline;
            संरचना fastdiv_मानानि div_G;
        } hmx;
        संरचना {
            uint32_t आकार_q_row_padded;
            uint32_t आकार_k_row_padded;
            uint32_t आकार_v_row_padded;
            संरचना fastdiv_मानानि src0_div21;
            संरचना fastdiv_मानानि src0_div1;
        } hvx;
    } u;
};

#यदि defined(__cplusplus)
static_assert(परिमाण(संरचना htp_fa_kernel_params) <= 128, "htp_fa_kernel_params is too large for kernel_params blob");
#यदि_अन्तः

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
संरचना hmx_fa_vtcm_layout {
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
    size_t off_row_bufs;
    size_t off_hmx_scales_id;
    size_t off_hmx_scales_qk;
    size_t off_आवरणम्_buf;
    size_t off_slopes;
    size_t off_sinks;

    // Region byte sizes reused by the device at runtime (not just for allocation).
    size_t q_tile_अष्टकानि;
    size_t o_tile_अष्टकानि;
    size_t k_tile_अष्टकानि;
    size_t s_tile_अष्टकानि;       // S and P tiles (same size)
    size_t d_tile_अष्टकानि;       // d_tiles[0..1] + d_inv_l, allocated back to back
    size_t m_पङ्क्तिः_अष्टकानि;       // one mask row
    size_t m_buf_slot_अष्टकानि;   // one dma_cache slot = align_up(Br * m_line_bytes, 4096)
    size_t col_vec_अष्टकानि;
    size_t sinks_अष्टकानि;

    // Derived strides.
    size_t row_buf_stride;       // HVX vectors (128B) per row buffer
    size_t आवरणम्_buf_row_stride;  // __fp16 elements per row in the mask buffer

    bool   pipeline;
    size_t योग_अष्टकानि;
};

#परिभाषय FA_MAX_KV_खण्डानि 1024

// Build the VTCM layout.

स्थिर अन्तःस्थ शून्य hmx_fa_vtcm_layout_build(संरचना hmx_fa_vtcm_layout * L,
                                            size_t gqa_गुणकः, size_t DK, size_t DV,
                                            size_t Br, size_t Bc, size_t n_तन्तवः,
                                            bool pipeline, bool is_q_fp32,
                                            bool has_sinks, size_t n_heads) {
    ध्रुव size_t g_br         = hex_align_ऊर्ध्वम्(gqa_गुणकः * Br, HMX_FP16_TILE_N_पङ्क्तिकाः);
    ध्रुव size_t q_tile_आकार  = hex_align_ऊर्ध्वम्(g_br * DK   * परिमाण(__fp16), HTP_FA_HMX_TILE_आकार);
    ध्रुव size_t o_tile_आकार  = hex_align_ऊर्ध्वम्(g_br * DV   * परिमाण(__fp16), HTP_FA_HMX_TILE_आकार);
    ध्रुव size_t k_tile_आकार  = hex_align_ऊर्ध्वम्(Bc   * DK   * परिमाण(__fp16), HTP_FA_HMX_TILE_आकार);
    ध्रुव size_t v_tile_आकार  = hex_align_ऊर्ध्वम्(Bc   * DV   * परिमाण(__fp16), HTP_FA_HMX_TILE_आकार);
    ध्रुव size_t s_tile_आकार  = hex_align_ऊर्ध्वम्(g_br * Bc   * परिमाण(__fp16), HTP_FA_HMX_TILE_आकार);

    // The rescale matrices are diagonal: the HMX kernels only ever load the g_br/32
    // tiles that sit on the diagonal, so store just those, packed back to back with
    // a stride of one tile.  The old [g_br, g_br] square layout allocated g_br/32
    // times more than it used, which is also why a second D buffer was unaffordable.
    ध्रुव size_t d_tile_आकार = (g_br / HMX_FP16_TILE_N_पङ्क्तिकाः) * HTP_FA_HMX_TILE_आकार;

    ध्रुव size_t q_dma_आकार   = hex_align_ऊर्ध्वम्(g_br * DK * (is_q_fp32 ? परिमाण(भिन्न) : परिमाण(__fp16)), 128);
    ध्रुव size_t k_dma_आकार   = hex_align_ऊर्ध्वम्(Bc * hex_आवर्तनम्_ऊर्ध्वम्(DK * परिमाण(__fp16), 128), 128);
    ध्रुव size_t v_dma_आकार   = hex_align_ऊर्ध्वम्(Bc * hex_आवर्तनम्_ऊर्ध्वम्(DV * परिमाण(__fp16), 128), 128);
    ध्रुव size_t col_vec_आकार = hex_align_ऊर्ध्वम्(g_br * परिमाण(भिन्न),  256);
    ध्रुव size_t row_vec_आकार = hex_align_ऊर्ध्वम्(Bc   * परिमाण(__fp16), 256);
    ध्रुव size_t m_पङ्क्तिः_आकार  = hex_align_ऊर्ध्वम्(Bc   * परिमाण(__fp16), 128);
    ध्रुव size_t m_buf_slot   = hex_align_ऊर्ध्वम्(Br * m_पङ्क्तिः_आकार, 256);
    ध्रुव size_t m_buf_आकार   = m_buf_slot * HMX_FA_DMA_CACHE_आकार;
    ध्रुव size_t slopes_आकार  = hex_align_ऊर्ध्वम्(g_br * परिमाण(__fp16), 128);
    ध्रुव size_t sinks_आकार   = hex_आवर्तनम्_ऊर्ध्वम्(n_heads * परिमाण(भिन्न), 128);

    size_t off = 0;

    // Group A (Part 1 - HMX Tiled buffers)
    VTCM_LAYOUT_ALLOC(off, off_q_tiles,       q_tile_आकार);
    VTCM_LAYOUT_ALLOC(off, off_o_tiles[0],    o_tile_आकार);
    VTCM_LAYOUT_ALLOC(off, off_o_tiles[1],    o_tile_आकार);
    VTCM_LAYOUT_ALLOC(off, off_d_tiles[0],    d_tile_आकार);
    VTCM_LAYOUT_ALLOC_OPTIONAL(off, off_d_tiles[1], d_tile_आकार, pipeline);
    VTCM_LAYOUT_ALLOC(off, off_d_inv_l,       d_tile_आकार);

    // Group B & C share start offset (Group B tiles must be 2KB aligned)
    size_t off_गणः_b_c = hex_align_ऊर्ध्वम्(off, HTP_FA_HMX_TILE_आकार);

    // Group B: Compute-only buffers
    size_t off_गणः_b = off_गणः_b_c;
    VTCM_LAYOUT_ALLOC(off_गणः_b, off_k_tiles[0],    k_tile_आकार);
    VTCM_LAYOUT_ALLOC_OPTIONAL(off_गणः_b, off_k_tiles[1], k_tile_आकार, pipeline);
    VTCM_LAYOUT_ALLOC(off_गणः_b, off_v_tiles[0],    v_tile_आकार);
    VTCM_LAYOUT_ALLOC_OPTIONAL(off_गणः_b, off_v_tiles[1], v_tile_आकार, pipeline);
    VTCM_LAYOUT_ALLOC(off_गणः_b, off_s_tiles[0],    s_tile_आकार);
    VTCM_LAYOUT_ALLOC_OPTIONAL(off_गणः_b, off_s_tiles[1], s_tile_आकार, pipeline);
    VTCM_LAYOUT_ALLOC(off_गणः_b, off_p_tiles[0],    s_tile_आकार);
    VTCM_LAYOUT_ALLOC_OPTIONAL(off_गणः_b, off_p_tiles[1], s_tile_आकार, pipeline);
    VTCM_LAYOUT_ALLOC(off_गणः_b, off_s_rowmax,      col_vec_आकार);
    VTCM_LAYOUT_ALLOC(off_गणः_b, off_p_rowsum,      col_vec_आकार);
    VTCM_LAYOUT_ALLOC(off_गणः_b, off_row_bufs,      row_vec_आकार * 2 * n_तन्तवः);

    ध्रुव size_t गणः_b_आकार = off_गणः_b - off_गणः_b_c;

    // Group C: Q fetch DMA buffer
    size_t off_गणः_c = off_गणः_b_c;
    VTCM_LAYOUT_ALLOC(off_गणः_c, off_q_dma,         q_dma_आकार);

    ध्रुव size_t गणः_c_आकार = off_गणः_c - off_गणः_b_c;

    off = off_गणः_b_c + hex_smax(गणः_b_आकार, गणः_c_आकार);

    // Group A (Part 2 - remaining non-HMX buffers)
    VTCM_LAYOUT_ALLOC(off, off_k_fp16[0],     k_dma_आकार);
    VTCM_LAYOUT_ALLOC(off, off_k_fp16[1],     k_dma_आकार);
    VTCM_LAYOUT_ALLOC(off, off_v_fp16[0],     v_dma_आकार);
    VTCM_LAYOUT_ALLOC(off, off_v_fp16[1],     v_dma_आकार);
    VTCM_LAYOUT_ALLOC(off, off_m_vec,         col_vec_आकार);
    VTCM_LAYOUT_ALLOC(off, off_l_vec,         col_vec_आकार);
    VTCM_LAYOUT_ALLOC(off, off_hmx_scales_id, 256);
    VTCM_LAYOUT_ALLOC(off, off_hmx_scales_qk, 256);
    VTCM_LAYOUT_ALLOC(off, off_आवरणम्_buf,      m_buf_आकार);
    VTCM_LAYOUT_ALLOC(off, off_slopes,        slopes_आकार);
    VTCM_LAYOUT_ALLOC_OPTIONAL(off, off_sinks, sinks_आकार, has_sinks);

    L->q_tile_अष्टकानि        = q_tile_आकार;
    L->o_tile_अष्टकानि        = o_tile_आकार;
    L->k_tile_अष्टकानि        = k_tile_आकार;
    L->col_vec_अष्टकानि       = col_vec_आकार;
    L->s_tile_अष्टकानि        = s_tile_आकार;
    // Measured from the actual offsets rather than assumed to be N * d_tile_size, so
    // that inserting a region between them (or adding padding to VTCM_LAYOUT_ALLOC)
    // cannot silently leave the tail of the run unzeroed.
    L->d_tile_अष्टकानि        = (L->off_d_inv_l + d_tile_आकार) - L->off_d_tiles[0];
    L->m_पङ्क्तिः_अष्टकानि        = m_पङ्क्तिः_आकार;
    L->m_buf_slot_अष्टकानि    = m_buf_slot;
    L->row_buf_stride      = row_vec_आकार / 128;
    L->आवरणम्_buf_row_stride = m_पङ्क्तिः_आकार / परिमाण(__fp16);
    L->sinks_अष्टकानि         = has_sinks ? sinks_आकार : 0;
    L->pipeline            = pipeline;
    L->योग_अष्टकानि         = off;
}

// Exact VTCM usage for a given (gqa_factor, DK, DV, Br, Bc) configuration.
स्थिर अन्तःस्थ size_t hmx_fa_compute_vtcm_usage(size_t gqa_गुणकः, size_t DK, size_t DV, size_t Br, size_t Bc, size_t n_तन्तवः, bool pipeline, bool is_q_fp32, bool has_sinks, size_t n_heads) {
    संरचना hmx_fa_vtcm_layout L;
    hmx_fa_vtcm_layout_build(&L, gqa_गुणकः, DK, DV, Br, Bc, n_तन्तवः, pipeline, is_q_fp32, has_sinks, n_heads);
    प्रत्यागम L.योग_अष्टकानि;
}

#परिभाषय FA_HVX_खण्डम्_आकार 64
#परिभाषय FA_HVX_G_MAX      8

संरचना hvx_fa_vtcm_layout {
    size_t off_q;
    size_t off_k;
    size_t off_v;
    size_t off_m;
    size_t off_a;
    size_t off_sinks;

    size_t आकार_q_खण्डम्;
    size_t आकार_k_खण्डम्;
    size_t आकार_v_खण्डम्;
    size_t आकार_m_खण्डम्;
    size_t आकार_vkq_acc;
    size_t आकार_sinks;

    size_t योग_अष्टकानि;
};

स्थिर अन्तःस्थ शून्य hvx_fa_vtcm_layout_build(संरचना hvx_fa_vtcm_layout * L,
                                            size_t DK, size_t DV,
                                            bool is_q_fp32, bool has_आवरणम्,
                                            bool has_sinks, size_t n_heads,
                                            size_t n_तन्तवः) {
    ध्रुव size_t आकार_q_row_padded = hex_आवर्तनम्_ऊर्ध्वम्(DK * (is_q_fp32 ? 4 : 2), 128);
    ध्रुव size_t आकार_k_row_padded = hex_आवर्तनम्_ऊर्ध्वम्(DK * परिमाण(__fp16), 128);
    ध्रुव size_t आकार_v_row_padded = hex_आवर्तनम्_ऊर्ध्वम्(DV * परिमाण(__fp16), 128);

    ध्रुव size_t आकार_q_खण्डम् = आकार_q_row_padded * FA_HVX_G_MAX;
    ध्रुव size_t आकार_k_खण्डम् = आकार_k_row_padded * FA_HVX_खण्डम्_आकार;
    ध्रुव size_t आकार_v_खण्डम् = आकार_v_row_padded * FA_HVX_खण्डम्_आकार;
    ध्रुव size_t आकार_m_खण्डम् = hex_आवर्तनम्_ऊर्ध्वम्(FA_HVX_खण्डम्_आकार * परिमाण(__fp16), 128);
    ध्रुव size_t आकार_vkq_acc = hex_आवर्तनम्_ऊर्ध्वम्(DV * परिमाण(भिन्न), 128) * FA_HVX_G_MAX;
    ध्रुव size_t आकार_sinks   = hex_आवर्तनम्_ऊर्ध्वम्(n_heads * परिमाण(भिन्न), 128);

    size_t off = 0;

    VTCM_LAYOUT_ALLOC(off, off_q, आकार_q_खण्डम् * n_तन्तवः);
    VTCM_LAYOUT_ALLOC(off, off_k, आकार_k_खण्डम् * 2 * n_तन्तवः);
    VTCM_LAYOUT_ALLOC(off, off_v, आकार_v_खण्डम् * 2 * n_तन्तवः);
    VTCM_LAYOUT_ALLOC_OPTIONAL(off, off_m, आकार_m_खण्डम् * HVX_FA_DMA_CACHE_आकार * n_तन्तवः, has_आवरणम्);
    VTCM_LAYOUT_ALLOC(off, off_a, आकार_vkq_acc * n_तन्तवः);
    VTCM_LAYOUT_ALLOC_OPTIONAL(off, off_sinks, आकार_sinks, has_sinks);

    L->आकार_q_खण्डम् = आकार_q_खण्डम्;
    L->आकार_k_खण्डम् = आकार_k_खण्डम्;
    L->आकार_v_खण्डम् = आकार_v_खण्डम्;
    L->आकार_m_खण्डम् = आकार_m_खण्डम्;
    L->आकार_vkq_acc = आकार_vkq_acc;
    L->आकार_sinks   = has_sinks ? आकार_sinks : 0;
    L->योग_अष्टकानि  = off;
}

स्थिर अन्तःस्थ size_t hvx_fa_compute_vtcm_usage(size_t DK, size_t DV, bool is_q_fp32, bool has_आवरणम्, bool has_sinks, size_t n_heads, size_t n_तन्तवः) {
    संरचना hvx_fa_vtcm_layout L;
    hvx_fa_vtcm_layout_build(&L, DK, DV, is_q_fp32, has_आवरणम्, has_sinks, n_heads, n_तन्तवः);
    प्रत्यागम L.योग_अष्टकानि;
}

#परिभाषय FA_MIN_KV_खण्डानि 3

// Cost-based (Br, Bc) search for flash attention with pipeline constraint.
स्थिर अन्तःस्थ पूर्णाङ्क hmx_fa_find_chunk_आकार(size_t * Br_बहिर्गमनम्,
                                         size_t * Bc_बहिर्गमनम्,
                                         size_t   gqa_गुणकः,
                                         size_t   DK,
                                         size_t   DV,
                                         size_t   qo_len,
                                         size_t   kv_len,
                                         size_t   vtcm_budget,
                                         size_t   n_तन्तवः,
                                         bool     is_q_fp32,
                                         bool     has_sinks,
                                         size_t   n_heads) {
    ध्रुव size_t T       = HMX_FP16_TILE_N_पङ्क्तिकाः;  // 32
    ध्रुव size_t br_unit = hmx_ceil_div(T, gqa_गुणकः);
    ध्रुव size_t bc_unit = HMX_FP16_TILE_N_स्तम्भसङ्ख्या * 2;  // 64
    ध्रुव bool   can_pipeline = (kv_len >= FA_MIN_KV_खण्डानि * bc_unit && n_तन्तवः >= 2);

    // Br_max: largest Br aligned to br_unit that does not exceed qo_len.
    ध्रुव size_t Br_max = qo_len >= br_unit ? hex_align_अधस्तात्(qo_len, br_unit) : br_unit;

    // Pipeline constraint: cap Bc so n_kv_blocks >= FA_MIN_KV_BLOCKS.
    // Only relax when kv_len is too short to form enough blocks.
    size_t Bc_सीमा     = can_pipeline ? hex_align_अधस्तात्(kv_len / FA_MIN_KV_खण्डानि, bc_unit) :
                                         (kv_len >= bc_unit ? hex_align_अधस्तात्(kv_len, bc_unit) : bc_unit);
    // For small-Q SWA-sized KV caches (qo_len <= 16, kv_len <= 2048), cap Bc at 256 for block-skipping granularity
    यदि (qo_len <= 16 && kv_len <= 2048 && Bc_सीमा > 256) {
        Bc_सीमा = 256;
    }
    // Cost coefficients calibrated from profiling
    ध्रुव size_t c_q_fixed    = 800;   // per-Q-block: q_load + epilogue o_update + o_norm + o_store
    ध्रुव size_t c_iter_आधारः  = 200;   // per-KV-iter base (HMX dot/update + DMA)
    ध्रुव size_t c_softmax    = 600;   // per 64-row vector chunk on HVX

    size_t best_cost = SIZE_MAX, best_mn = 0;
    size_t best_Br = 0, best_Bc = 0;

    प्रति (size_t Br = Br_max; Br >= br_unit; Br -= br_unit) {
        // Try all Bc candidates from Bc_limit down to bc_unit
        प्रति (size_t Bc = Bc_सीमा; Bc >= bc_unit; Bc -= bc_unit) {
            size_t vtcm_needed = hmx_fa_compute_vtcm_usage(gqa_गुणकः, DK, DV, Br, Bc, n_तन्तवः, can_pipeline, is_q_fp32, has_sinks, n_heads);
            यदि (vtcm_needed <= vtcm_budget) {
                // This Bc fits for this Br!
                ध्रुव size_t q_खण्डानि       = (qo_len + Br - 1) / Br;
                ध्रुव size_t kv_खण्डानि      = (kv_len + Bc - 1) / Bc;
                ध्रुव size_t actual_तन्तवः = (kv_खण्डानि >= 3 && n_तन्तवः >= 2) ? n_तन्तवः : 1;
                ध्रुव size_t n_पङ्क्तिकाः_g       = Br * gqa_गुणकः;
                ध्रुव size_t n_row_vec_cnt  = (n_पङ्क्तिकाः_g + 63) / 64;
                ध्रुव size_t n_use          = n_row_vec_cnt < actual_तन्तवः ? n_row_vec_cnt : actual_तन्तवः;
                ध्रुव size_t vecs_प्रत्येकम्_t     = n_use > 0 ? (n_row_vec_cnt + n_use - 1) / n_use : 1;

                ध्रुव size_t c_iter_actual  = c_iter_आधारः + c_softmax * vecs_प्रत्येकम्_t;
                ध्रुव size_t cost           = q_खण्डानि * (c_q_fixed + kv_खण्डानि * c_iter_actual);
                ध्रुव size_t mn             = Br * Bc;

                यदि (cost < best_cost || (cost == best_cost && mn > best_mn)) {
                    best_cost = cost;
                    best_mn   = mn;
                    best_Br   = Br;
                    best_Bc   = Bc;
                }
                // Since we iterate Bc from largest to smallest, this is the largest Bc that fits
                // for this Br. We can break to the next Br.
                भङ्ग;
            }
        }

        यदि (Br == br_unit) {
            भङ्ग;
        }
    }

    यदि (best_Br == 0 || best_Bc == 0) {
        प्रत्यागम -1;
    }

    *Br_बहिर्गमनम् = best_Br;
    *Bc_बहिर्गमनम् = best_Bc;
    प्रत्यागम 0;
}

#यदि_परिभाषितम् __cplusplus
}
#यदि_अन्तः

#यदि_अन्तः /* HTP_FLASH_ATTN_OPS_H */
