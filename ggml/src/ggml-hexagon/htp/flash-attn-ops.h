#నిర్వచితం_కాకపోతే HTP_FLASH_ATTN_OPS_H
#నిర్వచించు HTP_FLASH_ATTN_OPS_H

#అన్తర్భావ <stdint.h>
#అన్తర్భావ <stddef.h>
#అన్తర్భావ <stdbool.h>

#అన్తర్భావ "hex-fastdiv.h"
#అన్తర్భావ "hex-common.h"
#అన్తర్భావ "htp-vtcm.h"

#నిర్వచితమైతే __cplusplus
బాహ్య "C" {
#అయితే_ముగింపు

// Tile constants (mirrored from hmx-utils.h for use on host side if needed)
#నిర్వచించు HTP_FA_HMX_TILE_కొలత   2048
#నిర్వచించు HMX_FP16_TILE_కొలత     2048
#నిర్వచించు HMX_FP16_TILE_N_అడ్డువరుసలు   32
#నిర్వచించు HMX_FP16_TILE_N_నిలువుల_సంఖ్య   32
#నిర్వచించు HMX_FP16_TILE_N_ELMS   1024

#నిర్వచించు HVX_FA_DMA_CACHE_కొలత  128
#నిర్వచించు HMX_FA_DMA_CACHE_కొలత  4


#నిర్వచించు HTP_FA_M_INITIAL_VAL  -10000.0f

గణన htp_fa_kernel_type {
    HTP_FA_KERNEL_UNSUPPORTED = 0,
    HTP_FA_KERNEL_HVX,
    HTP_FA_KERNEL_HMX
};

నిర్మాణం htp_fa_kernel_params {
    uint8_t  kernel_type;        // enum htp_fa_kernel_type
    uint8_t  తల_విభజించు;         // 1 = partition by KV heads in multicore, 0 = token partition
    uint8_t  flags;              // reserved
    uint8_t  n_దారాలు;          // Number of threads to run

    // Common parameters
    uint16_t Br;
    uint16_t Bc;
    uint16_t n_kv_ఖండాలు;        // also HVX's n_blocks
    uint16_t G;                  // GQA factor (n_heads / n_kv_heads)

    దశాంశం    scale;
    దశాంశం    max_bias;
    దశాంశం    logit_softcap;
    uint32_t vtcm_కొలత;

    uint32_t qrows;
    uint32_t qrows_చొప్పున_thread;
    uint32_t qrow_ప్రారంభం;
    దశాంశం    m0;
    దశాంశం    m1;
    uint32_t n_తల_log2;

    నిర్మాణం fastdiv_విలువలు src3_div2;
    నిర్మాణం fastdiv_విలువలు src3_div3;

    నిర్మాణం fastdiv_విలువలు ప్రసారం_rk2;
    నిర్మాణం fastdiv_విలువలు ప్రసారం_rk3;
    నిర్మాణం fastdiv_విలువలు ప్రసారం_rv2;
    నిర్మాణం fastdiv_విలువలు ప్రసారం_rv3;

    సంఘం {
        నిర్మాణం {
            uint32_t g_br;
            uint32_t అడ్డువరుస_buf_stride;
            uint32_t ముసుగు_buf_అడ్డువరుస_stride;
            int32_t  ముసుగు_ప్రసారం;
            int32_t  pipeline;
            నిర్మాణం fastdiv_విలువలు div_G;
        } hmx;
        నిర్మాణం {
            uint32_t కొలత_q_అడ్డువరుస_padded;
            uint32_t కొలత_k_అడ్డువరుస_padded;
            uint32_t కొలత_v_అడ్డువరుస_padded;
            నిర్మాణం fastdiv_విలువలు src0_div21;
            నిర్మాణం fastdiv_విలువలు src0_div1;
        } hvx;
    } u;
};

#ఒకవేళ defined(__cplusplus)
static_assert(పరిమాణం(నిర్మాణం htp_fa_kernel_params) <= 128, "htp_fa_kernel_params is too large for kernel_params blob");
#అయితే_ముగింపు

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
నిర్మాణం hmx_fa_vtcm_layout {
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
    size_t off_అడ్డువరుస_bufs;
    size_t off_hmx_scales_id;
    size_t off_hmx_scales_qk;
    size_t off_ముసుగు_buf;
    size_t off_slopes;
    size_t off_sinks;

    // Region byte sizes reused by the device at runtime (not just for allocation).
    size_t q_tile_బైట్లు;
    size_t o_tile_బైట్లు;
    size_t k_tile_బైట్లు;
    size_t s_tile_బైట్లు;       // S and P tiles (same size)
    size_t d_tile_బైట్లు;       // d_tiles[0..1] + d_inv_l, allocated back to back
    size_t m_పంక్తి_బైట్లు;       // one mask row
    size_t m_buf_స్థానం_బైట్లు;   // one dma_cache slot = align_up(Br * m_line_bytes, 4096)
    size_t నిలువు_vec_బైట్లు;
    size_t sinks_బైట్లు;

    // Derived strides.
    size_t అడ్డువరుస_buf_stride;       // HVX vectors (128B) per row buffer
    size_t ముసుగు_buf_అడ్డువరుస_stride;  // __fp16 elements per row in the mask buffer

    bool   pipeline;
    size_t మొత్తం_బైట్లు;
};

#నిర్వచించు FA_MAX_KV_ఖండాలు 1024

// Build the VTCM layout.

స్థిర అంతఃవరుస శూన్య hmx_fa_vtcm_layout_build(నిర్మాణం hmx_fa_vtcm_layout * L,
                                            size_t gqa_గుణకం, size_t DK, size_t DV,
                                            size_t Br, size_t Bc, size_t n_దారాలు,
                                            bool pipeline, bool is_q_fp32,
                                            bool has_sinks, size_t n_heads) {
    స్థిరరాశి size_t g_br         = hex_align_పైకి(gqa_గుణకం * Br, HMX_FP16_TILE_N_అడ్డువరుసలు);
    స్థిరరాశి size_t q_tile_కొలత  = hex_align_పైకి(g_br * DK   * పరిమాణం(__fp16), HTP_FA_HMX_TILE_కొలత);
    స్థిరరాశి size_t o_tile_కొలత  = hex_align_పైకి(g_br * DV   * పరిమాణం(__fp16), HTP_FA_HMX_TILE_కొలత);
    స్థిరరాశి size_t k_tile_కొలత  = hex_align_పైకి(Bc   * DK   * పరిమాణం(__fp16), HTP_FA_HMX_TILE_కొలత);
    స్థిరరాశి size_t v_tile_కొలత  = hex_align_పైకి(Bc   * DV   * పరిమాణం(__fp16), HTP_FA_HMX_TILE_కొలత);
    స్థిరరాశి size_t s_tile_కొలత  = hex_align_పైకి(g_br * Bc   * పరిమాణం(__fp16), HTP_FA_HMX_TILE_కొలత);

    // The rescale matrices are diagonal: the HMX kernels only ever load the g_br/32
    // tiles that sit on the diagonal, so store just those, packed back to back with
    // a stride of one tile.  The old [g_br, g_br] square layout allocated g_br/32
    // times more than it used, which is also why a second D buffer was unaffordable.
    స్థిరరాశి size_t d_tile_కొలత = (g_br / HMX_FP16_TILE_N_అడ్డువరుసలు) * HTP_FA_HMX_TILE_కొలత;

    స్థిరరాశి size_t q_dma_కొలత   = hex_align_పైకి(g_br * DK * (is_q_fp32 ? పరిమాణం(దశాంశం) : పరిమాణం(__fp16)), 128);
    స్థిరరాశి size_t k_dma_కొలత   = hex_align_పైకి(Bc * hex_విడత_పైకి(DK * పరిమాణం(__fp16), 128), 128);
    స్థిరరాశి size_t v_dma_కొలత   = hex_align_పైకి(Bc * hex_విడత_పైకి(DV * పరిమాణం(__fp16), 128), 128);
    స్థిరరాశి size_t నిలువు_vec_కొలత = hex_align_పైకి(g_br * పరిమాణం(దశాంశం),  256);
    స్థిరరాశి size_t అడ్డువరుస_vec_కొలత = hex_align_పైకి(Bc   * పరిమాణం(__fp16), 256);
    స్థిరరాశి size_t m_పంక్తి_కొలత  = hex_align_పైకి(Bc   * పరిమాణం(__fp16), 128);
    స్థిరరాశి size_t m_buf_స్థానం   = hex_align_పైకి(Br * m_పంక్తి_కొలత, 256);
    స్థిరరాశి size_t m_buf_కొలత   = m_buf_స్థానం * HMX_FA_DMA_CACHE_కొలత;
    స్థిరరాశి size_t slopes_కొలత  = hex_align_పైకి(g_br * పరిమాణం(__fp16), 128);
    స్థిరరాశి size_t sinks_కొలత   = hex_విడత_పైకి(n_heads * పరిమాణం(దశాంశం), 128);

    size_t off = 0;

    // Group A (Part 1 - HMX Tiled buffers)
    VTCM_LAYOUT_ALLOC(off, off_q_tiles,       q_tile_కొలత);
    VTCM_LAYOUT_ALLOC(off, off_o_tiles[0],    o_tile_కొలత);
    VTCM_LAYOUT_ALLOC(off, off_o_tiles[1],    o_tile_కొలత);
    VTCM_LAYOUT_ALLOC(off, off_d_tiles[0],    d_tile_కొలత);
    VTCM_LAYOUT_ALLOC_OPTIONAL(off, off_d_tiles[1], d_tile_కొలత, pipeline);
    VTCM_LAYOUT_ALLOC(off, off_d_inv_l,       d_tile_కొలత);

    // Group B & C share start offset (Group B tiles must be 2KB aligned)
    size_t off_సమూహం_b_c = hex_align_పైకి(off, HTP_FA_HMX_TILE_కొలత);

    // Group B: Compute-only buffers
    size_t off_సమూహం_b = off_సమూహం_b_c;
    VTCM_LAYOUT_ALLOC(off_సమూహం_b, off_k_tiles[0],    k_tile_కొలత);
    VTCM_LAYOUT_ALLOC_OPTIONAL(off_సమూహం_b, off_k_tiles[1], k_tile_కొలత, pipeline);
    VTCM_LAYOUT_ALLOC(off_సమూహం_b, off_v_tiles[0],    v_tile_కొలత);
    VTCM_LAYOUT_ALLOC_OPTIONAL(off_సమూహం_b, off_v_tiles[1], v_tile_కొలత, pipeline);
    VTCM_LAYOUT_ALLOC(off_సమూహం_b, off_s_tiles[0],    s_tile_కొలత);
    VTCM_LAYOUT_ALLOC_OPTIONAL(off_సమూహం_b, off_s_tiles[1], s_tile_కొలత, pipeline);
    VTCM_LAYOUT_ALLOC(off_సమూహం_b, off_p_tiles[0],    s_tile_కొలత);
    VTCM_LAYOUT_ALLOC_OPTIONAL(off_సమూహం_b, off_p_tiles[1], s_tile_కొలత, pipeline);
    VTCM_LAYOUT_ALLOC(off_సమూహం_b, off_s_rowmax,      నిలువు_vec_కొలత);
    VTCM_LAYOUT_ALLOC(off_సమూహం_b, off_p_rowsum,      నిలువు_vec_కొలత);
    VTCM_LAYOUT_ALLOC(off_సమూహం_b, off_అడ్డువరుస_bufs,      అడ్డువరుస_vec_కొలత * 2 * n_దారాలు);

    స్థిరరాశి size_t సమూహం_b_కొలత = off_సమూహం_b - off_సమూహం_b_c;

    // Group C: Q fetch DMA buffer
    size_t off_సమూహం_c = off_సమూహం_b_c;
    VTCM_LAYOUT_ALLOC(off_సమూహం_c, off_q_dma,         q_dma_కొలత);

    స్థిరరాశి size_t సమూహం_c_కొలత = off_సమూహం_c - off_సమూహం_b_c;

    off = off_సమూహం_b_c + hex_smax(సమూహం_b_కొలత, సమూహం_c_కొలత);

    // Group A (Part 2 - remaining non-HMX buffers)
    VTCM_LAYOUT_ALLOC(off, off_k_fp16[0],     k_dma_కొలత);
    VTCM_LAYOUT_ALLOC(off, off_k_fp16[1],     k_dma_కొలత);
    VTCM_LAYOUT_ALLOC(off, off_v_fp16[0],     v_dma_కొలత);
    VTCM_LAYOUT_ALLOC(off, off_v_fp16[1],     v_dma_కొలత);
    VTCM_LAYOUT_ALLOC(off, off_m_vec,         నిలువు_vec_కొలత);
    VTCM_LAYOUT_ALLOC(off, off_l_vec,         నిలువు_vec_కొలత);
    VTCM_LAYOUT_ALLOC(off, off_hmx_scales_id, 256);
    VTCM_LAYOUT_ALLOC(off, off_hmx_scales_qk, 256);
    VTCM_LAYOUT_ALLOC(off, off_ముసుగు_buf,      m_buf_కొలత);
    VTCM_LAYOUT_ALLOC(off, off_slopes,        slopes_కొలత);
    VTCM_LAYOUT_ALLOC_OPTIONAL(off, off_sinks, sinks_కొలత, has_sinks);

    L->q_tile_బైట్లు        = q_tile_కొలత;
    L->o_tile_బైట్లు        = o_tile_కొలత;
    L->k_tile_బైట్లు        = k_tile_కొలత;
    L->నిలువు_vec_బైట్లు       = నిలువు_vec_కొలత;
    L->s_tile_బైట్లు        = s_tile_కొలత;
    // Measured from the actual offsets rather than assumed to be N * d_tile_size, so
    // that inserting a region between them (or adding padding to VTCM_LAYOUT_ALLOC)
    // cannot silently leave the tail of the run unzeroed.
    L->d_tile_బైట్లు        = (L->off_d_inv_l + d_tile_కొలత) - L->off_d_tiles[0];
    L->m_పంక్తి_బైట్లు        = m_పంక్తి_కొలత;
    L->m_buf_స్థానం_బైట్లు    = m_buf_స్థానం;
    L->అడ్డువరుస_buf_stride      = అడ్డువరుస_vec_కొలత / 128;
    L->ముసుగు_buf_అడ్డువరుస_stride = m_పంక్తి_కొలత / పరిమాణం(__fp16);
    L->sinks_బైట్లు         = has_sinks ? sinks_కొలత : 0;
    L->pipeline            = pipeline;
    L->మొత్తం_బైట్లు         = off;
}

// Exact VTCM usage for a given (gqa_factor, DK, DV, Br, Bc) configuration.
స్థిర అంతఃవరుస size_t hmx_fa_compute_vtcm_usage(size_t gqa_గుణకం, size_t DK, size_t DV, size_t Br, size_t Bc, size_t n_దారాలు, bool pipeline, bool is_q_fp32, bool has_sinks, size_t n_heads) {
    నిర్మాణం hmx_fa_vtcm_layout L;
    hmx_fa_vtcm_layout_build(&L, gqa_గుణకం, DK, DV, Br, Bc, n_దారాలు, pipeline, is_q_fp32, has_sinks, n_heads);
    తిరిగి L.మొత్తం_బైట్లు;
}

#నిర్వచించు FA_HVX_ఖండం_కొలత 64
#నిర్వచించు FA_HVX_G_MAX      8

నిర్మాణం hvx_fa_vtcm_layout {
    size_t off_q;
    size_t off_k;
    size_t off_v;
    size_t off_m;
    size_t off_a;
    size_t off_sinks;

    size_t కొలత_q_ఖండం;
    size_t కొలత_k_ఖండం;
    size_t కొలత_v_ఖండం;
    size_t కొలత_m_ఖండం;
    size_t కొలత_vkq_acc;
    size_t కొలత_sinks;

    size_t మొత్తం_బైట్లు;
};

స్థిర అంతఃవరుస శూన్య hvx_fa_vtcm_layout_build(నిర్మాణం hvx_fa_vtcm_layout * L,
                                            size_t DK, size_t DV,
                                            bool is_q_fp32, bool has_ముసుగు,
                                            bool has_sinks, size_t n_heads,
                                            size_t n_దారాలు) {
    స్థిరరాశి size_t కొలత_q_అడ్డువరుస_padded = hex_విడత_పైకి(DK * (is_q_fp32 ? 4 : 2), 128);
    స్థిరరాశి size_t కొలత_k_అడ్డువరుస_padded = hex_విడత_పైకి(DK * పరిమాణం(__fp16), 128);
    స్థిరరాశి size_t కొలత_v_అడ్డువరుస_padded = hex_విడత_పైకి(DV * పరిమాణం(__fp16), 128);

    స్థిరరాశి size_t కొలత_q_ఖండం = కొలత_q_అడ్డువరుస_padded * FA_HVX_G_MAX;
    స్థిరరాశి size_t కొలత_k_ఖండం = కొలత_k_అడ్డువరుస_padded * FA_HVX_ఖండం_కొలత;
    స్థిరరాశి size_t కొలత_v_ఖండం = కొలత_v_అడ్డువరుస_padded * FA_HVX_ఖండం_కొలత;
    స్థిరరాశి size_t కొలత_m_ఖండం = hex_విడత_పైకి(FA_HVX_ఖండం_కొలత * పరిమాణం(__fp16), 128);
    స్థిరరాశి size_t కొలత_vkq_acc = hex_విడత_పైకి(DV * పరిమాణం(దశాంశం), 128) * FA_HVX_G_MAX;
    స్థిరరాశి size_t కొలత_sinks   = hex_విడత_పైకి(n_heads * పరిమాణం(దశాంశం), 128);

    size_t off = 0;

    VTCM_LAYOUT_ALLOC(off, off_q, కొలత_q_ఖండం * n_దారాలు);
    VTCM_LAYOUT_ALLOC(off, off_k, కొలత_k_ఖండం * 2 * n_దారాలు);
    VTCM_LAYOUT_ALLOC(off, off_v, కొలత_v_ఖండం * 2 * n_దారాలు);
    VTCM_LAYOUT_ALLOC_OPTIONAL(off, off_m, కొలత_m_ఖండం * HVX_FA_DMA_CACHE_కొలత * n_దారాలు, has_ముసుగు);
    VTCM_LAYOUT_ALLOC(off, off_a, కొలత_vkq_acc * n_దారాలు);
    VTCM_LAYOUT_ALLOC_OPTIONAL(off, off_sinks, కొలత_sinks, has_sinks);

    L->కొలత_q_ఖండం = కొలత_q_ఖండం;
    L->కొలత_k_ఖండం = కొలత_k_ఖండం;
    L->కొలత_v_ఖండం = కొలత_v_ఖండం;
    L->కొలత_m_ఖండం = కొలత_m_ఖండం;
    L->కొలత_vkq_acc = కొలత_vkq_acc;
    L->కొలత_sinks   = has_sinks ? కొలత_sinks : 0;
    L->మొత్తం_బైట్లు  = off;
}

స్థిర అంతఃవరుస size_t hvx_fa_compute_vtcm_usage(size_t DK, size_t DV, bool is_q_fp32, bool has_ముసుగు, bool has_sinks, size_t n_heads, size_t n_దారాలు) {
    నిర్మాణం hvx_fa_vtcm_layout L;
    hvx_fa_vtcm_layout_build(&L, DK, DV, is_q_fp32, has_ముసుగు, has_sinks, n_heads, n_దారాలు);
    తిరిగి L.మొత్తం_బైట్లు;
}

#నిర్వచించు FA_MIN_KV_ఖండాలు 3

// Cost-based (Br, Bc) search for flash attention with pipeline constraint.
స్థిర అంతఃవరుస పూర్ణాంకం hmx_fa_కనుగొను_chunk_కొలత(size_t * Br_బహిర్గతం,
                                         size_t * Bc_బహిర్గతం,
                                         size_t   gqa_గుణకం,
                                         size_t   DK,
                                         size_t   DV,
                                         size_t   qo_len,
                                         size_t   kv_len,
                                         size_t   vtcm_budget,
                                         size_t   n_దారాలు,
                                         bool     is_q_fp32,
                                         bool     has_sinks,
                                         size_t   n_heads) {
    స్థిరరాశి size_t T       = HMX_FP16_TILE_N_అడ్డువరుసలు;  // 32
    స్థిరరాశి size_t br_unit = hmx_ceil_div(T, gqa_గుణకం);
    స్థిరరాశి size_t bc_unit = HMX_FP16_TILE_N_నిలువుల_సంఖ్య * 2;  // 64
    స్థిరరాశి bool   can_pipeline = (kv_len >= FA_MIN_KV_ఖండాలు * bc_unit && n_దారాలు >= 2);

    // Br_max: largest Br aligned to br_unit that does not exceed qo_len.
    స్థిరరాశి size_t Br_max = qo_len >= br_unit ? hex_align_కిందికి(qo_len, br_unit) : br_unit;

    // Pipeline constraint: cap Bc so n_kv_blocks >= FA_MIN_KV_BLOCKS.
    // Only relax when kv_len is too short to form enough blocks.
    size_t Bc_హద్దు     = can_pipeline ? hex_align_కిందికి(kv_len / FA_MIN_KV_ఖండాలు, bc_unit) :
                                         (kv_len >= bc_unit ? hex_align_కిందికి(kv_len, bc_unit) : bc_unit);
    // For small-Q SWA-sized KV caches (qo_len <= 16, kv_len <= 2048), cap Bc at 256 for block-skipping granularity
    ఒకవేళ (qo_len <= 16 && kv_len <= 2048 && Bc_హద్దు > 256) {
        Bc_హద్దు = 256;
    }
    // Cost coefficients calibrated from profiling
    స్థిరరాశి size_t c_q_fixed    = 800;   // per-Q-block: q_load + epilogue o_update + o_norm + o_store
    స్థిరరాశి size_t c_iter_పునాది  = 200;   // per-KV-iter base (HMX dot/update + DMA)
    స్థిరరాశి size_t c_softmax    = 600;   // per 64-row vector chunk on HVX

    size_t ఉత్తమ_ఖర్చు = SIZE_MAX, ఉత్తమ_mn = 0;
    size_t ఉత్తమ_Br = 0, ఉత్తమ_Bc = 0;

    కొరకు (size_t Br = Br_max; Br >= br_unit; Br -= br_unit) {
        // Try all Bc candidates from Bc_limit down to bc_unit
        కొరకు (size_t Bc = Bc_హద్దు; Bc >= bc_unit; Bc -= bc_unit) {
            size_t vtcm_needed = hmx_fa_compute_vtcm_usage(gqa_గుణకం, DK, DV, Br, Bc, n_దారాలు, can_pipeline, is_q_fp32, has_sinks, n_heads);
            ఒకవేళ (vtcm_needed <= vtcm_budget) {
                // This Bc fits for this Br!
                స్థిరరాశి size_t q_ఖండాలు       = (qo_len + Br - 1) / Br;
                స్థిరరాశి size_t kv_ఖండాలు      = (kv_len + Bc - 1) / Bc;
                స్థిరరాశి size_t actual_దారాలు = (kv_ఖండాలు >= 3 && n_దారాలు >= 2) ? n_దారాలు : 1;
                స్థిరరాశి size_t n_అడ్డువరుసలు_g       = Br * gqa_గుణకం;
                స్థిరరాశి size_t n_అడ్డువరుస_vec_cnt  = (n_అడ్డువరుసలు_g + 63) / 64;
                స్థిరరాశి size_t n_use          = n_అడ్డువరుస_vec_cnt < actual_దారాలు ? n_అడ్డువరుస_vec_cnt : actual_దారాలు;
                స్థిరరాశి size_t vecs_చొప్పున_t     = n_use > 0 ? (n_అడ్డువరుస_vec_cnt + n_use - 1) / n_use : 1;

                స్థిరరాశి size_t c_iter_actual  = c_iter_పునాది + c_softmax * vecs_చొప్పున_t;
                స్థిరరాశి size_t ఖర్చు           = q_ఖండాలు * (c_q_fixed + kv_ఖండాలు * c_iter_actual);
                స్థిరరాశి size_t mn             = Br * Bc;

                ఒకవేళ (ఖర్చు < ఉత్తమ_ఖర్చు || (ఖర్చు == ఉత్తమ_ఖర్చు && mn > ఉత్తమ_mn)) {
                    ఉత్తమ_ఖర్చు = ఖర్చు;
                    ఉత్తమ_mn   = mn;
                    ఉత్తమ_Br   = Br;
                    ఉత్తమ_Bc   = Bc;
                }
                // Since we iterate Bc from largest to smallest, this is the largest Bc that fits
                // for this Br. We can break to the next Br.
                ఆపు;
            }
        }

        ఒకవేళ (Br == br_unit) {
            ఆపు;
        }
    }

    ఒకవేళ (ఉత్తమ_Br == 0 || ఉత్తమ_Bc == 0) {
        తిరిగి -1;
    }

    *Br_బహిర్గతం = ఉత్తమ_Br;
    *Bc_బహిర్గతం = ఉత్తమ_Bc;
    తిరిగి 0;
}

#నిర్వచితమైతే __cplusplus
}
#అయితే_ముగింపు

#అయితే_ముగింపు /* HTP_FLASH_ATTN_OPS_H */
