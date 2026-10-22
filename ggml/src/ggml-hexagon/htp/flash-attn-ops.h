#अगर_अघोषित HTP_FLASH_ATTN_OPS_H
#घोषणा HTP_FLASH_ATTN_OPS_H

#समावेश <stdint.h>
#समावेश <stddef.h>
#समावेश <stdbool.h>

#समावेश "hex-fastdiv.h"
#समावेश "hex-common.h"
#समावेश "htp-vtcm.h"

#अगर_घोषित __cplusplus
बाह्य "C" {
#पूर्ण_अगर

// Tile constants (mirrored from hmx-utils.h for use on host side if needed)
#घोषणा HTP_FA_HMX_TILE_आकार   2048
#घोषणा HMX_FP16_TILE_आकार     2048
#घोषणा HMX_FP16_TILE_N_ROWS   32
#घोषणा HMX_FP16_TILE_N_COLS   32
#घोषणा HMX_FP16_TILE_N_ELMS   1024

#घोषणा HVX_FA_DMA_CACHE_आकार  128
#घोषणा HMX_FA_DMA_CACHE_आकार  4


#घोषणा HTP_FA_M_INITIAL_VAL  -10000.0f

क्रमागत htp_fa_kernel_type {
    HTP_FA_KERNEL_UNSUPPORTED = 0,
    HTP_FA_KERNEL_HVX,
    HTP_FA_KERNEL_HMX
};

काष्ठा htp_fa_kernel_params {
    uint8_t  kernel_type;        // enum htp_fa_kernel_type
    uint8_t  शीर्ष_split;         // 1 = partition by KV heads in multicore, 0 = token partition
    uint8_t  flags;              // reserved
    uint8_t  n_threads;          // Number of threads to run

    // Common parameters
    uint16_t Br;
    uint16_t Bc;
    uint16_t n_kv_blocks;        // also HVX's n_blocks
    uint16_t G;                  // GQA factor (n_heads / n_kv_heads)

    भग्न    scale;
    भग्न    max_bias;
    भग्न    logit_softcap;
    uint32_t vtcm_आकार;

    uint32_t qrows;
    uint32_t qrows_per_thread;
    uint32_t qrow_start;
    भग्न    m0;
    भग्न    m1;
    uint32_t n_शीर्ष_log2;

    काष्ठा fastdiv_मूल्यावली src3_div2;
    काष्ठा fastdiv_मूल्यावली src3_div3;

    काष्ठा fastdiv_मूल्यावली broadcast_rk2;
    काष्ठा fastdiv_मूल्यावली broadcast_rk3;
    काष्ठा fastdiv_मूल्यावली broadcast_rv2;
    काष्ठा fastdiv_मूल्यावली broadcast_rv3;

    जोड़ {
        काष्ठा {
            uint32_t g_br;
            uint32_t row_buf_stride;
            uint32_t mask_buf_row_stride;
            int32_t  mask_broadcast;
            int32_t  pipeline;
            काष्ठा fastdiv_मूल्यावली div_G;
        } hmx;
        काष्ठा {
            uint32_t आकार_q_row_padded;
            uint32_t आकार_k_row_padded;
            uint32_t आकार_v_row_padded;
            काष्ठा fastdiv_मूल्यावली src0_div21;
            काष्ठा fastdiv_मूल्यावली src0_div1;
        } hvx;
    } u;
};

#अगर defined(__cplusplus)
static_assert(माप(काष्ठा htp_fa_kernel_params) <= 128, "htp_fa_kernel_params is too large for kernel_params blob");
#पूर्ण_अगर

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
काष्ठा hmx_fa_vtcm_layout {
    // Byte offsets from vtcm_base for each region.
    माप_प्रकार off_q_tiles;
    माप_प्रकार off_q_dma;
    माप_प्रकार off_o_tiles[2];
    माप_प्रकार off_k_fp16[2];
    माप_प्रकार off_v_fp16[2];
    माप_प्रकार off_k_tiles[2];
    माप_प्रकार off_v_tiles[2];
    माप_प्रकार off_s_tiles[2];
    माप_प्रकार off_p_tiles[2];
    माप_प्रकार off_d_tiles[2];
    माप_प्रकार off_d_inv_l;
    माप_प्रकार off_m_vec;
    माप_प्रकार off_l_vec;
    माप_प्रकार off_s_rowmax;
    माप_प्रकार off_p_rowsum;
    माप_प्रकार off_row_bufs;
    माप_प्रकार off_hmx_scales_id;
    माप_प्रकार off_hmx_scales_qk;
    माप_प्रकार off_mask_buf;
    माप_प्रकार off_slopes;
    माप_प्रकार off_sinks;

    // Region byte sizes reused by the device at runtime (not just for allocation).
    माप_प्रकार q_tile_bytes;
    माप_प्रकार o_tile_bytes;
    माप_प्रकार k_tile_bytes;
    माप_प्रकार s_tile_bytes;       // S and P tiles (same size)
    माप_प्रकार d_tile_bytes;       // d_tiles[0..1] + d_inv_l, allocated back to back
    माप_प्रकार m_पंक्ति_bytes;       // one mask row
    माप_प्रकार m_buf_slot_bytes;   // one dma_cache slot = align_up(Br * m_line_bytes, 4096)
    माप_प्रकार col_vec_bytes;
    माप_प्रकार sinks_bytes;

    // Derived strides.
    माप_प्रकार row_buf_stride;       // HVX vectors (128B) per row buffer
    माप_प्रकार mask_buf_row_stride;  // __fp16 elements per row in the mask buffer

    bool   pipeline;
    माप_प्रकार योग_bytes;
};

#घोषणा FA_MAX_KV_BLOCKS 1024

// Build the VTCM layout.

अटल अंतरभूत व्योम hmx_fa_vtcm_layout_build(काष्ठा hmx_fa_vtcm_layout * L,
                                            माप_प्रकार gqa_factor, माप_प्रकार DK, माप_प्रकार DV,
                                            माप_प्रकार Br, माप_प्रकार Bc, माप_प्रकार n_threads,
                                            bool pipeline, bool is_q_fp32,
                                            bool has_sinks, माप_प्रकार n_heads) {
    स्थिर माप_प्रकार g_br         = hex_align_up(gqa_factor * Br, HMX_FP16_TILE_N_ROWS);
    स्थिर माप_प्रकार q_tile_आकार  = hex_align_up(g_br * DK   * माप(__fp16), HTP_FA_HMX_TILE_आकार);
    स्थिर माप_प्रकार o_tile_आकार  = hex_align_up(g_br * DV   * माप(__fp16), HTP_FA_HMX_TILE_आकार);
    स्थिर माप_प्रकार k_tile_आकार  = hex_align_up(Bc   * DK   * माप(__fp16), HTP_FA_HMX_TILE_आकार);
    स्थिर माप_प्रकार v_tile_आकार  = hex_align_up(Bc   * DV   * माप(__fp16), HTP_FA_HMX_TILE_आकार);
    स्थिर माप_प्रकार s_tile_आकार  = hex_align_up(g_br * Bc   * माप(__fp16), HTP_FA_HMX_TILE_आकार);

    // The rescale matrices are diagonal: the HMX kernels only ever load the g_br/32
    // tiles that sit on the diagonal, so store just those, packed back to back with
    // a stride of one tile.  The old [g_br, g_br] square layout allocated g_br/32
    // times more than it used, which is also why a second D buffer was unaffordable.
    स्थिर माप_प्रकार d_tile_आकार = (g_br / HMX_FP16_TILE_N_ROWS) * HTP_FA_HMX_TILE_आकार;

    स्थिर माप_प्रकार q_dma_आकार   = hex_align_up(g_br * DK * (is_q_fp32 ? माप(भग्न) : माप(__fp16)), 128);
    स्थिर माप_प्रकार k_dma_आकार   = hex_align_up(Bc * hex_round_up(DK * माप(__fp16), 128), 128);
    स्थिर माप_प्रकार v_dma_आकार   = hex_align_up(Bc * hex_round_up(DV * माप(__fp16), 128), 128);
    स्थिर माप_प्रकार col_vec_आकार = hex_align_up(g_br * माप(भग्न),  256);
    स्थिर माप_प्रकार row_vec_आकार = hex_align_up(Bc   * माप(__fp16), 256);
    स्थिर माप_प्रकार m_पंक्ति_आकार  = hex_align_up(Bc   * माप(__fp16), 128);
    स्थिर माप_प्रकार m_buf_slot   = hex_align_up(Br * m_पंक्ति_आकार, 256);
    स्थिर माप_प्रकार m_buf_आकार   = m_buf_slot * HMX_FA_DMA_CACHE_आकार;
    स्थिर माप_प्रकार slopes_आकार  = hex_align_up(g_br * माप(__fp16), 128);
    स्थिर माप_प्रकार sinks_आकार   = hex_round_up(n_heads * माप(भग्न), 128);

    माप_प्रकार off = 0;

    // Group A (Part 1 - HMX Tiled buffers)
    VTCM_LAYOUT_ALLOC(off, off_q_tiles,       q_tile_आकार);
    VTCM_LAYOUT_ALLOC(off, off_o_tiles[0],    o_tile_आकार);
    VTCM_LAYOUT_ALLOC(off, off_o_tiles[1],    o_tile_आकार);
    VTCM_LAYOUT_ALLOC(off, off_d_tiles[0],    d_tile_आकार);
    VTCM_LAYOUT_ALLOC_OPTIONAL(off, off_d_tiles[1], d_tile_आकार, pipeline);
    VTCM_LAYOUT_ALLOC(off, off_d_inv_l,       d_tile_आकार);

    // Group B & C share start offset (Group B tiles must be 2KB aligned)
    माप_प्रकार off_group_b_c = hex_align_up(off, HTP_FA_HMX_TILE_आकार);

    // Group B: Compute-only buffers
    माप_प्रकार off_group_b = off_group_b_c;
    VTCM_LAYOUT_ALLOC(off_group_b, off_k_tiles[0],    k_tile_आकार);
    VTCM_LAYOUT_ALLOC_OPTIONAL(off_group_b, off_k_tiles[1], k_tile_आकार, pipeline);
    VTCM_LAYOUT_ALLOC(off_group_b, off_v_tiles[0],    v_tile_आकार);
    VTCM_LAYOUT_ALLOC_OPTIONAL(off_group_b, off_v_tiles[1], v_tile_आकार, pipeline);
    VTCM_LAYOUT_ALLOC(off_group_b, off_s_tiles[0],    s_tile_आकार);
    VTCM_LAYOUT_ALLOC_OPTIONAL(off_group_b, off_s_tiles[1], s_tile_आकार, pipeline);
    VTCM_LAYOUT_ALLOC(off_group_b, off_p_tiles[0],    s_tile_आकार);
    VTCM_LAYOUT_ALLOC_OPTIONAL(off_group_b, off_p_tiles[1], s_tile_आकार, pipeline);
    VTCM_LAYOUT_ALLOC(off_group_b, off_s_rowmax,      col_vec_आकार);
    VTCM_LAYOUT_ALLOC(off_group_b, off_p_rowsum,      col_vec_आकार);
    VTCM_LAYOUT_ALLOC(off_group_b, off_row_bufs,      row_vec_आकार * 2 * n_threads);

    स्थिर माप_प्रकार group_b_आकार = off_group_b - off_group_b_c;

    // Group C: Q fetch DMA buffer
    माप_प्रकार off_group_c = off_group_b_c;
    VTCM_LAYOUT_ALLOC(off_group_c, off_q_dma,         q_dma_आकार);

    स्थिर माप_प्रकार group_c_आकार = off_group_c - off_group_b_c;

    off = off_group_b_c + hex_smax(group_b_आकार, group_c_आकार);

    // Group A (Part 2 - remaining non-HMX buffers)
    VTCM_LAYOUT_ALLOC(off, off_k_fp16[0],     k_dma_आकार);
    VTCM_LAYOUT_ALLOC(off, off_k_fp16[1],     k_dma_आकार);
    VTCM_LAYOUT_ALLOC(off, off_v_fp16[0],     v_dma_आकार);
    VTCM_LAYOUT_ALLOC(off, off_v_fp16[1],     v_dma_आकार);
    VTCM_LAYOUT_ALLOC(off, off_m_vec,         col_vec_आकार);
    VTCM_LAYOUT_ALLOC(off, off_l_vec,         col_vec_आकार);
    VTCM_LAYOUT_ALLOC(off, off_hmx_scales_id, 256);
    VTCM_LAYOUT_ALLOC(off, off_hmx_scales_qk, 256);
    VTCM_LAYOUT_ALLOC(off, off_mask_buf,      m_buf_आकार);
    VTCM_LAYOUT_ALLOC(off, off_slopes,        slopes_आकार);
    VTCM_LAYOUT_ALLOC_OPTIONAL(off, off_sinks, sinks_आकार, has_sinks);

    L->q_tile_bytes        = q_tile_आकार;
    L->o_tile_bytes        = o_tile_आकार;
    L->k_tile_bytes        = k_tile_आकार;
    L->col_vec_bytes       = col_vec_आकार;
    L->s_tile_bytes        = s_tile_आकार;
    // Measured from the actual offsets rather than assumed to be N * d_tile_size, so
    // that inserting a region between them (or adding padding to VTCM_LAYOUT_ALLOC)
    // cannot silently leave the tail of the run unzeroed.
    L->d_tile_bytes        = (L->off_d_inv_l + d_tile_आकार) - L->off_d_tiles[0];
    L->m_पंक्ति_bytes        = m_पंक्ति_आकार;
    L->m_buf_slot_bytes    = m_buf_slot;
    L->row_buf_stride      = row_vec_आकार / 128;
    L->mask_buf_row_stride = m_पंक्ति_आकार / माप(__fp16);
    L->sinks_bytes         = has_sinks ? sinks_आकार : 0;
    L->pipeline            = pipeline;
    L->योग_bytes         = off;
}

// Exact VTCM usage for a given (gqa_factor, DK, DV, Br, Bc) configuration.
अटल अंतरभूत माप_प्रकार hmx_fa_compute_vtcm_usage(माप_प्रकार gqa_factor, माप_प्रकार DK, माप_प्रकार DV, माप_प्रकार Br, माप_प्रकार Bc, माप_प्रकार n_threads, bool pipeline, bool is_q_fp32, bool has_sinks, माप_प्रकार n_heads) {
    काष्ठा hmx_fa_vtcm_layout L;
    hmx_fa_vtcm_layout_build(&L, gqa_factor, DK, DV, Br, Bc, n_threads, pipeline, is_q_fp32, has_sinks, n_heads);
    वापस L.योग_bytes;
}

#घोषणा FA_HVX_BLOCK_आकार 64
#घोषणा FA_HVX_G_MAX      8

काष्ठा hvx_fa_vtcm_layout {
    माप_प्रकार off_q;
    माप_प्रकार off_k;
    माप_प्रकार off_v;
    माप_प्रकार off_m;
    माप_प्रकार off_a;
    माप_प्रकार off_sinks;

    माप_प्रकार आकार_q_block;
    माप_प्रकार आकार_k_block;
    माप_प्रकार आकार_v_block;
    माप_प्रकार आकार_m_block;
    माप_प्रकार आकार_vkq_acc;
    माप_प्रकार आकार_sinks;

    माप_प्रकार योग_bytes;
};

अटल अंतरभूत व्योम hvx_fa_vtcm_layout_build(काष्ठा hvx_fa_vtcm_layout * L,
                                            माप_प्रकार DK, माप_प्रकार DV,
                                            bool is_q_fp32, bool has_mask,
                                            bool has_sinks, माप_प्रकार n_heads,
                                            माप_प्रकार n_threads) {
    स्थिर माप_प्रकार आकार_q_row_padded = hex_round_up(DK * (is_q_fp32 ? 4 : 2), 128);
    स्थिर माप_प्रकार आकार_k_row_padded = hex_round_up(DK * माप(__fp16), 128);
    स्थिर माप_प्रकार आकार_v_row_padded = hex_round_up(DV * माप(__fp16), 128);

    स्थिर माप_प्रकार आकार_q_block = आकार_q_row_padded * FA_HVX_G_MAX;
    स्थिर माप_प्रकार आकार_k_block = आकार_k_row_padded * FA_HVX_BLOCK_आकार;
    स्थिर माप_प्रकार आकार_v_block = आकार_v_row_padded * FA_HVX_BLOCK_आकार;
    स्थिर माप_प्रकार आकार_m_block = hex_round_up(FA_HVX_BLOCK_आकार * माप(__fp16), 128);
    स्थिर माप_प्रकार आकार_vkq_acc = hex_round_up(DV * माप(भग्न), 128) * FA_HVX_G_MAX;
    स्थिर माप_प्रकार आकार_sinks   = hex_round_up(n_heads * माप(भग्न), 128);

    माप_प्रकार off = 0;

    VTCM_LAYOUT_ALLOC(off, off_q, आकार_q_block * n_threads);
    VTCM_LAYOUT_ALLOC(off, off_k, आकार_k_block * 2 * n_threads);
    VTCM_LAYOUT_ALLOC(off, off_v, आकार_v_block * 2 * n_threads);
    VTCM_LAYOUT_ALLOC_OPTIONAL(off, off_m, आकार_m_block * HVX_FA_DMA_CACHE_आकार * n_threads, has_mask);
    VTCM_LAYOUT_ALLOC(off, off_a, आकार_vkq_acc * n_threads);
    VTCM_LAYOUT_ALLOC_OPTIONAL(off, off_sinks, आकार_sinks, has_sinks);

    L->आकार_q_block = आकार_q_block;
    L->आकार_k_block = आकार_k_block;
    L->आकार_v_block = आकार_v_block;
    L->आकार_m_block = आकार_m_block;
    L->आकार_vkq_acc = आकार_vkq_acc;
    L->आकार_sinks   = has_sinks ? आकार_sinks : 0;
    L->योग_bytes  = off;
}

अटल अंतरभूत माप_प्रकार hvx_fa_compute_vtcm_usage(माप_प्रकार DK, माप_प्रकार DV, bool is_q_fp32, bool has_mask, bool has_sinks, माप_प्रकार n_heads, माप_प्रकार n_threads) {
    काष्ठा hvx_fa_vtcm_layout L;
    hvx_fa_vtcm_layout_build(&L, DK, DV, is_q_fp32, has_mask, has_sinks, n_heads, n_threads);
    वापस L.योग_bytes;
}

#घोषणा FA_MIN_KV_BLOCKS 3

// Cost-based (Br, Bc) search for flash attention with pipeline constraint.
अटल अंतरभूत पूर्णांक hmx_fa_find_chunk_आकार(माप_प्रकार * Br_out,
                                         माप_प्रकार * Bc_out,
                                         माप_प्रकार   gqa_factor,
                                         माप_प्रकार   DK,
                                         माप_प्रकार   DV,
                                         माप_प्रकार   qo_len,
                                         माप_प्रकार   kv_len,
                                         माप_प्रकार   vtcm_budget,
                                         माप_प्रकार   n_threads,
                                         bool     is_q_fp32,
                                         bool     has_sinks,
                                         माप_प्रकार   n_heads) {
    स्थिर माप_प्रकार T       = HMX_FP16_TILE_N_ROWS;  // 32
    स्थिर माप_प्रकार br_unit = hmx_ceil_div(T, gqa_factor);
    स्थिर माप_प्रकार bc_unit = HMX_FP16_TILE_N_COLS * 2;  // 64
    स्थिर bool   can_pipeline = (kv_len >= FA_MIN_KV_BLOCKS * bc_unit && n_threads >= 2);

    // Br_max: largest Br aligned to br_unit that does not exceed qo_len.
    स्थिर माप_प्रकार Br_max = qo_len >= br_unit ? hex_align_down(qo_len, br_unit) : br_unit;

    // Pipeline constraint: cap Bc so n_kv_blocks >= FA_MIN_KV_BLOCKS.
    // Only relax when kv_len is too short to form enough blocks.
    माप_प्रकार Bc_limit     = can_pipeline ? hex_align_down(kv_len / FA_MIN_KV_BLOCKS, bc_unit) :
                                         (kv_len >= bc_unit ? hex_align_down(kv_len, bc_unit) : bc_unit);
    // For small-Q SWA-sized KV caches (qo_len <= 16, kv_len <= 2048), cap Bc at 256 for block-skipping granularity
    अगर (qo_len <= 16 && kv_len <= 2048 && Bc_limit > 256) {
        Bc_limit = 256;
    }
    // Cost coefficients calibrated from profiling
    स्थिर माप_प्रकार c_q_fixed    = 800;   // per-Q-block: q_load + epilogue o_update + o_norm + o_store
    स्थिर माप_प्रकार c_iter_base  = 200;   // per-KV-iter base (HMX dot/update + DMA)
    स्थिर माप_प्रकार c_softmax    = 600;   // per 64-row vector chunk on HVX

    माप_प्रकार best_cost = SIZE_MAX, best_mn = 0;
    माप_प्रकार best_Br = 0, best_Bc = 0;

    क्रम (माप_प्रकार Br = Br_max; Br >= br_unit; Br -= br_unit) {
        // Try all Bc candidates from Bc_limit down to bc_unit
        क्रम (माप_प्रकार Bc = Bc_limit; Bc >= bc_unit; Bc -= bc_unit) {
            माप_प्रकार vtcm_needed = hmx_fa_compute_vtcm_usage(gqa_factor, DK, DV, Br, Bc, n_threads, can_pipeline, is_q_fp32, has_sinks, n_heads);
            अगर (vtcm_needed <= vtcm_budget) {
                // This Bc fits for this Br!
                स्थिर माप_प्रकार q_blocks       = (qo_len + Br - 1) / Br;
                स्थिर माप_प्रकार kv_blocks      = (kv_len + Bc - 1) / Bc;
                स्थिर माप_प्रकार actual_threads = (kv_blocks >= 3 && n_threads >= 2) ? n_threads : 1;
                स्थिर माप_प्रकार n_rows_g       = Br * gqa_factor;
                स्थिर माप_प्रकार n_row_vec_cnt  = (n_rows_g + 63) / 64;
                स्थिर माप_प्रकार n_use          = n_row_vec_cnt < actual_threads ? n_row_vec_cnt : actual_threads;
                स्थिर माप_प्रकार vecs_per_t     = n_use > 0 ? (n_row_vec_cnt + n_use - 1) / n_use : 1;

                स्थिर माप_प्रकार c_iter_actual  = c_iter_base + c_softmax * vecs_per_t;
                स्थिर माप_प्रकार cost           = q_blocks * (c_q_fixed + kv_blocks * c_iter_actual);
                स्थिर माप_प्रकार mn             = Br * Bc;

                अगर (cost < best_cost || (cost == best_cost && mn > best_mn)) {
                    best_cost = cost;
                    best_mn   = mn;
                    best_Br   = Br;
                    best_Bc   = Bc;
                }
                // Since we iterate Bc from largest to smallest, this is the largest Bc that fits
                // for this Br. We can break to the next Br.
                अवरोध;
            }
        }

        अगर (Br == br_unit) {
            अवरोध;
        }
    }

    अगर (best_Br == 0 || best_Bc == 0) {
        वापस -1;
    }

    *Br_out = best_Br;
    *Bc_out = best_Bc;
    वापस 0;
}

#अगर_घोषित __cplusplus
}
#पूर्ण_अगर

#पूर्ण_अगर /* HTP_FLASH_ATTN_OPS_H */
