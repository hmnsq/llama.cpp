#ವ್ಯಾಖ್ಯಾನಿತವಾಗದಿದ್ದರೆ HTP_FLASH_ATTN_OPS_H
#ವ್ಯಾಖ್ಯಾನಿಸು HTP_FLASH_ATTN_OPS_H

#ಅನ್ತರ್ಭಾವ <stdint.h>
#ಅನ್ತರ್ಭಾವ <stddef.h>
#ಅನ್ತರ್ಭಾವ <stdbool.h>

#ಅನ್ತರ್ಭಾವ "hex-fastdiv.h"
#ಅನ್ತರ್ಭಾವ "hex-common.h"
#ಅನ್ತರ್ಭಾವ "htp-vtcm.h"

#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ __cplusplus
ಬಾಹ್ಯ "C" {
#ಆದರೆ_ಅಂತ್ಯ

// Tile constants (mirrored from hmx-utils.h for use on host side if needed)
#ವ್ಯಾಖ್ಯಾನಿಸು HTP_FA_HMX_TILE_ಗಾತ್ರ   2048
#ವ್ಯಾಖ್ಯಾನಿಸು HMX_FP16_TILE_ಗಾತ್ರ     2048
#ವ್ಯಾಖ್ಯಾನಿಸು HMX_FP16_TILE_N_ಅಡ್ಡಸಾಲುಗಳು   32
#ವ್ಯಾಖ್ಯಾನಿಸು HMX_FP16_TILE_N_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ   32
#ವ್ಯಾಖ್ಯಾನಿಸು HMX_FP16_TILE_N_ELMS   1024

#ವ್ಯಾಖ್ಯಾನಿಸು HVX_FA_DMA_CACHE_ಗಾತ್ರ  128
#ವ್ಯಾಖ್ಯಾನಿಸು HMX_FA_DMA_CACHE_ಗಾತ್ರ  4


#ವ್ಯಾಖ್ಯಾನಿಸು HTP_FA_M_INITIAL_VAL  -10000.0f

ಎಣಿಕೆ htp_fa_kernel_type {
    HTP_FA_KERNEL_UNSUPPORTED = 0,
    HTP_FA_KERNEL_HVX,
    HTP_FA_KERNEL_HMX
};

ರಚನೆ htp_fa_kernel_params {
    uint8_t  kernel_type;        // enum htp_fa_kernel_type
    uint8_t  ತಲೆ_ವಿಭಜಿಸು;         // 1 = partition by KV heads in multicore, 0 = token partition
    uint8_t  flags;              // reserved
    uint8_t  n_ಎಳೆಗಳು;          // Number of threads to run

    // Common parameters
    uint16_t Br;
    uint16_t Bc;
    uint16_t n_kv_ಖಂಡಗಳು;        // also HVX's n_blocks
    uint16_t G;                  // GQA factor (n_heads / n_kv_heads)

    ದಶಮಾಂಶ    scale;
    ದಶಮಾಂಶ    max_bias;
    ದಶಮಾಂಶ    logit_softcap;
    uint32_t vtcm_ಗಾತ್ರ;

    uint32_t qrows;
    uint32_t qrows_ಪ್ರತಿ_thread;
    uint32_t qrow_ಆರಂಭ;
    ದಶಮಾಂಶ    m0;
    ದಶಮಾಂಶ    m1;
    uint32_t n_ತಲೆ_log2;

    ರಚನೆ fastdiv_ಮೌಲ್ಯಗಳು src3_div2;
    ರಚನೆ fastdiv_ಮೌಲ್ಯಗಳು src3_div3;

    ರಚನೆ fastdiv_ಮೌಲ್ಯಗಳು ಪ್ರಸಾರ_rk2;
    ರಚನೆ fastdiv_ಮೌಲ್ಯಗಳು ಪ್ರಸಾರ_rk3;
    ರಚನೆ fastdiv_ಮೌಲ್ಯಗಳು ಪ್ರಸಾರ_rv2;
    ರಚನೆ fastdiv_ಮೌಲ್ಯಗಳು ಪ್ರಸಾರ_rv3;

    ಒಕ್ಕೂಟ {
        ರಚನೆ {
            uint32_t g_br;
            uint32_t ಅಡ್ಡಸಾಲು_buf_stride;
            uint32_t ಮುಖವಾಡ_buf_ಅಡ್ಡಸಾಲು_stride;
            int32_t  ಮುಖವಾಡ_ಪ್ರಸಾರ;
            int32_t  pipeline;
            ರಚನೆ fastdiv_ಮೌಲ್ಯಗಳು div_G;
        } hmx;
        ರಚನೆ {
            uint32_t ಗಾತ್ರ_q_ಅಡ್ಡಸಾಲು_padded;
            uint32_t ಗಾತ್ರ_k_ಅಡ್ಡಸಾಲು_padded;
            uint32_t ಗಾತ್ರ_v_ಅಡ್ಡಸಾಲು_padded;
            ರಚನೆ fastdiv_ಮೌಲ್ಯಗಳು src0_div21;
            ರಚನೆ fastdiv_ಮೌಲ್ಯಗಳು src0_div1;
        } hvx;
    } u;
};

#ಆದರೆ defined(__cplusplus)
static_assert(ಗಾತ್ರ(ರಚನೆ htp_fa_kernel_params) <= 128, "htp_fa_kernel_params is too large for kernel_params blob");
#ಆದರೆ_ಅಂತ್ಯ

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
ರಚನೆ hmx_fa_vtcm_layout {
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
    size_t off_ಅಡ್ಡಸಾಲು_bufs;
    size_t off_hmx_scales_id;
    size_t off_hmx_scales_qk;
    size_t off_ಮುಖವಾಡ_buf;
    size_t off_slopes;
    size_t off_sinks;

    // Region byte sizes reused by the device at runtime (not just for allocation).
    size_t q_tile_ಬೈಟುಗಳು;
    size_t o_tile_ಬೈಟುಗಳು;
    size_t k_tile_ಬೈಟುಗಳು;
    size_t s_tile_ಬೈಟುಗಳು;       // S and P tiles (same size)
    size_t d_tile_ಬೈಟುಗಳು;       // d_tiles[0..1] + d_inv_l, allocated back to back
    size_t m_ಸಾಲು_ಬೈಟುಗಳು;       // one mask row
    size_t m_buf_ಸ್ಥಾನ_ಬೈಟುಗಳು;   // one dma_cache slot = align_up(Br * m_line_bytes, 4096)
    size_t ಲಂಬಸಾಲು_vec_ಬೈಟುಗಳು;
    size_t sinks_ಬೈಟುಗಳು;

    // Derived strides.
    size_t ಅಡ್ಡಸಾಲು_buf_stride;       // HVX vectors (128B) per row buffer
    size_t ಮುಖವಾಡ_buf_ಅಡ್ಡಸಾಲು_stride;  // __fp16 elements per row in the mask buffer

    bool   pipeline;
    size_t ಒಟ್ಟು_ಬೈಟುಗಳು;
};

#ವ್ಯಾಖ್ಯಾನಿಸು FA_MAX_KV_ಖಂಡಗಳು 1024

// Build the VTCM layout.

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ hmx_fa_vtcm_layout_build(ರಚನೆ hmx_fa_vtcm_layout * L,
                                            size_t gqa_ಗುಣಕ, size_t DK, size_t DV,
                                            size_t Br, size_t Bc, size_t n_ಎಳೆಗಳು,
                                            bool pipeline, bool is_q_fp32,
                                            bool has_sinks, size_t n_heads) {
    ಸ್ಥಿರಾಂಕ size_t g_br         = hex_align_ಮೇಲೆ(gqa_ಗುಣಕ * Br, HMX_FP16_TILE_N_ಅಡ್ಡಸಾಲುಗಳು);
    ಸ್ಥಿರಾಂಕ size_t q_tile_ಗಾತ್ರ  = hex_align_ಮೇಲೆ(g_br * DK   * ಗಾತ್ರ(__fp16), HTP_FA_HMX_TILE_ಗಾತ್ರ);
    ಸ್ಥಿರಾಂಕ size_t o_tile_ಗಾತ್ರ  = hex_align_ಮೇಲೆ(g_br * DV   * ಗಾತ್ರ(__fp16), HTP_FA_HMX_TILE_ಗಾತ್ರ);
    ಸ್ಥಿರಾಂಕ size_t k_tile_ಗಾತ್ರ  = hex_align_ಮೇಲೆ(Bc   * DK   * ಗಾತ್ರ(__fp16), HTP_FA_HMX_TILE_ಗಾತ್ರ);
    ಸ್ಥಿರಾಂಕ size_t v_tile_ಗಾತ್ರ  = hex_align_ಮೇಲೆ(Bc   * DV   * ಗಾತ್ರ(__fp16), HTP_FA_HMX_TILE_ಗಾತ್ರ);
    ಸ್ಥಿರಾಂಕ size_t s_tile_ಗಾತ್ರ  = hex_align_ಮೇಲೆ(g_br * Bc   * ಗಾತ್ರ(__fp16), HTP_FA_HMX_TILE_ಗಾತ್ರ);

    // The rescale matrices are diagonal: the HMX kernels only ever load the g_br/32
    // tiles that sit on the diagonal, so store just those, packed back to back with
    // a stride of one tile.  The old [g_br, g_br] square layout allocated g_br/32
    // times more than it used, which is also why a second D buffer was unaffordable.
    ಸ್ಥಿರಾಂಕ size_t d_tile_ಗಾತ್ರ = (g_br / HMX_FP16_TILE_N_ಅಡ್ಡಸಾಲುಗಳು) * HTP_FA_HMX_TILE_ಗಾತ್ರ;

    ಸ್ಥಿರಾಂಕ size_t q_dma_ಗಾತ್ರ   = hex_align_ಮೇಲೆ(g_br * DK * (is_q_fp32 ? ಗಾತ್ರ(ದಶಮಾಂಶ) : ಗಾತ್ರ(__fp16)), 128);
    ಸ್ಥಿರಾಂಕ size_t k_dma_ಗಾತ್ರ   = hex_align_ಮೇಲೆ(Bc * hex_ಸುತ್ತು_ಮೇಲೆ(DK * ಗಾತ್ರ(__fp16), 128), 128);
    ಸ್ಥಿರಾಂಕ size_t v_dma_ಗಾತ್ರ   = hex_align_ಮೇಲೆ(Bc * hex_ಸುತ್ತು_ಮೇಲೆ(DV * ಗಾತ್ರ(__fp16), 128), 128);
    ಸ್ಥಿರಾಂಕ size_t ಲಂಬಸಾಲು_vec_ಗಾತ್ರ = hex_align_ಮೇಲೆ(g_br * ಗಾತ್ರ(ದಶಮಾಂಶ),  256);
    ಸ್ಥಿರಾಂಕ size_t ಅಡ್ಡಸಾಲು_vec_ಗಾತ್ರ = hex_align_ಮೇಲೆ(Bc   * ಗಾತ್ರ(__fp16), 256);
    ಸ್ಥಿರಾಂಕ size_t m_ಸಾಲು_ಗಾತ್ರ  = hex_align_ಮೇಲೆ(Bc   * ಗಾತ್ರ(__fp16), 128);
    ಸ್ಥಿರಾಂಕ size_t m_buf_ಸ್ಥಾನ   = hex_align_ಮೇಲೆ(Br * m_ಸಾಲು_ಗಾತ್ರ, 256);
    ಸ್ಥಿರಾಂಕ size_t m_buf_ಗಾತ್ರ   = m_buf_ಸ್ಥಾನ * HMX_FA_DMA_CACHE_ಗಾತ್ರ;
    ಸ್ಥಿರಾಂಕ size_t slopes_ಗಾತ್ರ  = hex_align_ಮೇಲೆ(g_br * ಗಾತ್ರ(__fp16), 128);
    ಸ್ಥಿರಾಂಕ size_t sinks_ಗಾತ್ರ   = hex_ಸುತ್ತು_ಮೇಲೆ(n_heads * ಗಾತ್ರ(ದಶಮಾಂಶ), 128);

    size_t off = 0;

    // Group A (Part 1 - HMX Tiled buffers)
    VTCM_LAYOUT_ALLOC(off, off_q_tiles,       q_tile_ಗಾತ್ರ);
    VTCM_LAYOUT_ALLOC(off, off_o_tiles[0],    o_tile_ಗಾತ್ರ);
    VTCM_LAYOUT_ALLOC(off, off_o_tiles[1],    o_tile_ಗಾತ್ರ);
    VTCM_LAYOUT_ALLOC(off, off_d_tiles[0],    d_tile_ಗಾತ್ರ);
    VTCM_LAYOUT_ALLOC_OPTIONAL(off, off_d_tiles[1], d_tile_ಗಾತ್ರ, pipeline);
    VTCM_LAYOUT_ALLOC(off, off_d_inv_l,       d_tile_ಗಾತ್ರ);

    // Group B & C share start offset (Group B tiles must be 2KB aligned)
    size_t off_ಗುಂಪು_b_c = hex_align_ಮೇಲೆ(off, HTP_FA_HMX_TILE_ಗಾತ್ರ);

    // Group B: Compute-only buffers
    size_t off_ಗುಂಪು_b = off_ಗುಂಪು_b_c;
    VTCM_LAYOUT_ALLOC(off_ಗುಂಪು_b, off_k_tiles[0],    k_tile_ಗಾತ್ರ);
    VTCM_LAYOUT_ALLOC_OPTIONAL(off_ಗುಂಪು_b, off_k_tiles[1], k_tile_ಗಾತ್ರ, pipeline);
    VTCM_LAYOUT_ALLOC(off_ಗುಂಪು_b, off_v_tiles[0],    v_tile_ಗಾತ್ರ);
    VTCM_LAYOUT_ALLOC_OPTIONAL(off_ಗುಂಪು_b, off_v_tiles[1], v_tile_ಗಾತ್ರ, pipeline);
    VTCM_LAYOUT_ALLOC(off_ಗುಂಪು_b, off_s_tiles[0],    s_tile_ಗಾತ್ರ);
    VTCM_LAYOUT_ALLOC_OPTIONAL(off_ಗುಂಪು_b, off_s_tiles[1], s_tile_ಗಾತ್ರ, pipeline);
    VTCM_LAYOUT_ALLOC(off_ಗುಂಪು_b, off_p_tiles[0],    s_tile_ಗಾತ್ರ);
    VTCM_LAYOUT_ALLOC_OPTIONAL(off_ಗುಂಪು_b, off_p_tiles[1], s_tile_ಗಾತ್ರ, pipeline);
    VTCM_LAYOUT_ALLOC(off_ಗುಂಪು_b, off_s_rowmax,      ಲಂಬಸಾಲು_vec_ಗಾತ್ರ);
    VTCM_LAYOUT_ALLOC(off_ಗುಂಪು_b, off_p_rowsum,      ಲಂಬಸಾಲು_vec_ಗಾತ್ರ);
    VTCM_LAYOUT_ALLOC(off_ಗುಂಪು_b, off_ಅಡ್ಡಸಾಲು_bufs,      ಅಡ್ಡಸಾಲು_vec_ಗಾತ್ರ * 2 * n_ಎಳೆಗಳು);

    ಸ್ಥಿರಾಂಕ size_t ಗುಂಪು_b_ಗಾತ್ರ = off_ಗುಂಪು_b - off_ಗುಂಪು_b_c;

    // Group C: Q fetch DMA buffer
    size_t off_ಗುಂಪು_c = off_ಗುಂಪು_b_c;
    VTCM_LAYOUT_ALLOC(off_ಗುಂಪು_c, off_q_dma,         q_dma_ಗಾತ್ರ);

    ಸ್ಥಿರಾಂಕ size_t ಗುಂಪು_c_ಗಾತ್ರ = off_ಗುಂಪು_c - off_ಗುಂಪು_b_c;

    off = off_ಗುಂಪು_b_c + hex_smax(ಗುಂಪು_b_ಗಾತ್ರ, ಗುಂಪು_c_ಗಾತ್ರ);

    // Group A (Part 2 - remaining non-HMX buffers)
    VTCM_LAYOUT_ALLOC(off, off_k_fp16[0],     k_dma_ಗಾತ್ರ);
    VTCM_LAYOUT_ALLOC(off, off_k_fp16[1],     k_dma_ಗಾತ್ರ);
    VTCM_LAYOUT_ALLOC(off, off_v_fp16[0],     v_dma_ಗಾತ್ರ);
    VTCM_LAYOUT_ALLOC(off, off_v_fp16[1],     v_dma_ಗಾತ್ರ);
    VTCM_LAYOUT_ALLOC(off, off_m_vec,         ಲಂಬಸಾಲು_vec_ಗಾತ್ರ);
    VTCM_LAYOUT_ALLOC(off, off_l_vec,         ಲಂಬಸಾಲು_vec_ಗಾತ್ರ);
    VTCM_LAYOUT_ALLOC(off, off_hmx_scales_id, 256);
    VTCM_LAYOUT_ALLOC(off, off_hmx_scales_qk, 256);
    VTCM_LAYOUT_ALLOC(off, off_ಮುಖವಾಡ_buf,      m_buf_ಗಾತ್ರ);
    VTCM_LAYOUT_ALLOC(off, off_slopes,        slopes_ಗಾತ್ರ);
    VTCM_LAYOUT_ALLOC_OPTIONAL(off, off_sinks, sinks_ಗಾತ್ರ, has_sinks);

    L->q_tile_ಬೈಟುಗಳು        = q_tile_ಗಾತ್ರ;
    L->o_tile_ಬೈಟುಗಳು        = o_tile_ಗಾತ್ರ;
    L->k_tile_ಬೈಟುಗಳು        = k_tile_ಗಾತ್ರ;
    L->ಲಂಬಸಾಲು_vec_ಬೈಟುಗಳು       = ಲಂಬಸಾಲು_vec_ಗಾತ್ರ;
    L->s_tile_ಬೈಟುಗಳು        = s_tile_ಗಾತ್ರ;
    // Measured from the actual offsets rather than assumed to be N * d_tile_size, so
    // that inserting a region between them (or adding padding to VTCM_LAYOUT_ALLOC)
    // cannot silently leave the tail of the run unzeroed.
    L->d_tile_ಬೈಟುಗಳು        = (L->off_d_inv_l + d_tile_ಗಾತ್ರ) - L->off_d_tiles[0];
    L->m_ಸಾಲು_ಬೈಟುಗಳು        = m_ಸಾಲು_ಗಾತ್ರ;
    L->m_buf_ಸ್ಥಾನ_ಬೈಟುಗಳು    = m_buf_ಸ್ಥಾನ;
    L->ಅಡ್ಡಸಾಲು_buf_stride      = ಅಡ್ಡಸಾಲು_vec_ಗಾತ್ರ / 128;
    L->ಮುಖವಾಡ_buf_ಅಡ್ಡಸಾಲು_stride = m_ಸಾಲು_ಗಾತ್ರ / ಗಾತ್ರ(__fp16);
    L->sinks_ಬೈಟುಗಳು         = has_sinks ? sinks_ಗಾತ್ರ : 0;
    L->pipeline            = pipeline;
    L->ಒಟ್ಟು_ಬೈಟುಗಳು         = off;
}

// Exact VTCM usage for a given (gqa_factor, DK, DV, Br, Bc) configuration.
ಸ್ಥಿರ ಸಾಲೊಳಗೆ size_t hmx_fa_compute_vtcm_usage(size_t gqa_ಗುಣಕ, size_t DK, size_t DV, size_t Br, size_t Bc, size_t n_ಎಳೆಗಳು, bool pipeline, bool is_q_fp32, bool has_sinks, size_t n_heads) {
    ರಚನೆ hmx_fa_vtcm_layout L;
    hmx_fa_vtcm_layout_build(&L, gqa_ಗುಣಕ, DK, DV, Br, Bc, n_ಎಳೆಗಳು, pipeline, is_q_fp32, has_sinks, n_heads);
    ಹಿಂತಿರುಗಿ L.ಒಟ್ಟು_ಬೈಟುಗಳು;
}

#ವ್ಯಾಖ್ಯಾನಿಸು FA_HVX_ಖಂಡ_ಗಾತ್ರ 64
#ವ್ಯಾಖ್ಯಾನಿಸು FA_HVX_G_MAX      8

ರಚನೆ hvx_fa_vtcm_layout {
    size_t off_q;
    size_t off_k;
    size_t off_v;
    size_t off_m;
    size_t off_a;
    size_t off_sinks;

    size_t ಗಾತ್ರ_q_ಖಂಡ;
    size_t ಗಾತ್ರ_k_ಖಂಡ;
    size_t ಗಾತ್ರ_v_ಖಂಡ;
    size_t ಗಾತ್ರ_m_ಖಂಡ;
    size_t ಗಾತ್ರ_vkq_acc;
    size_t ಗಾತ್ರ_sinks;

    size_t ಒಟ್ಟು_ಬೈಟುಗಳು;
};

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ hvx_fa_vtcm_layout_build(ರಚನೆ hvx_fa_vtcm_layout * L,
                                            size_t DK, size_t DV,
                                            bool is_q_fp32, bool has_ಮುಖವಾಡ,
                                            bool has_sinks, size_t n_heads,
                                            size_t n_ಎಳೆಗಳು) {
    ಸ್ಥಿರಾಂಕ size_t ಗಾತ್ರ_q_ಅಡ್ಡಸಾಲು_padded = hex_ಸುತ್ತು_ಮೇಲೆ(DK * (is_q_fp32 ? 4 : 2), 128);
    ಸ್ಥಿರಾಂಕ size_t ಗಾತ್ರ_k_ಅಡ್ಡಸಾಲು_padded = hex_ಸುತ್ತು_ಮೇಲೆ(DK * ಗಾತ್ರ(__fp16), 128);
    ಸ್ಥಿರಾಂಕ size_t ಗಾತ್ರ_v_ಅಡ್ಡಸಾಲು_padded = hex_ಸುತ್ತು_ಮೇಲೆ(DV * ಗಾತ್ರ(__fp16), 128);

    ಸ್ಥಿರಾಂಕ size_t ಗಾತ್ರ_q_ಖಂಡ = ಗಾತ್ರ_q_ಅಡ್ಡಸಾಲು_padded * FA_HVX_G_MAX;
    ಸ್ಥಿರಾಂಕ size_t ಗಾತ್ರ_k_ಖಂಡ = ಗಾತ್ರ_k_ಅಡ್ಡಸಾಲು_padded * FA_HVX_ಖಂಡ_ಗಾತ್ರ;
    ಸ್ಥಿರಾಂಕ size_t ಗಾತ್ರ_v_ಖಂಡ = ಗಾತ್ರ_v_ಅಡ್ಡಸಾಲು_padded * FA_HVX_ಖಂಡ_ಗಾತ್ರ;
    ಸ್ಥಿರಾಂಕ size_t ಗಾತ್ರ_m_ಖಂಡ = hex_ಸುತ್ತು_ಮೇಲೆ(FA_HVX_ಖಂಡ_ಗಾತ್ರ * ಗಾತ್ರ(__fp16), 128);
    ಸ್ಥಿರಾಂಕ size_t ಗಾತ್ರ_vkq_acc = hex_ಸುತ್ತು_ಮೇಲೆ(DV * ಗಾತ್ರ(ದಶಮಾಂಶ), 128) * FA_HVX_G_MAX;
    ಸ್ಥಿರಾಂಕ size_t ಗಾತ್ರ_sinks   = hex_ಸುತ್ತು_ಮೇಲೆ(n_heads * ಗಾತ್ರ(ದಶಮಾಂಶ), 128);

    size_t off = 0;

    VTCM_LAYOUT_ALLOC(off, off_q, ಗಾತ್ರ_q_ಖಂಡ * n_ಎಳೆಗಳು);
    VTCM_LAYOUT_ALLOC(off, off_k, ಗಾತ್ರ_k_ಖಂಡ * 2 * n_ಎಳೆಗಳು);
    VTCM_LAYOUT_ALLOC(off, off_v, ಗಾತ್ರ_v_ಖಂಡ * 2 * n_ಎಳೆಗಳು);
    VTCM_LAYOUT_ALLOC_OPTIONAL(off, off_m, ಗಾತ್ರ_m_ಖಂಡ * HVX_FA_DMA_CACHE_ಗಾತ್ರ * n_ಎಳೆಗಳು, has_ಮುಖವಾಡ);
    VTCM_LAYOUT_ALLOC(off, off_a, ಗಾತ್ರ_vkq_acc * n_ಎಳೆಗಳು);
    VTCM_LAYOUT_ALLOC_OPTIONAL(off, off_sinks, ಗಾತ್ರ_sinks, has_sinks);

    L->ಗಾತ್ರ_q_ಖಂಡ = ಗಾತ್ರ_q_ಖಂಡ;
    L->ಗಾತ್ರ_k_ಖಂಡ = ಗಾತ್ರ_k_ಖಂಡ;
    L->ಗಾತ್ರ_v_ಖಂಡ = ಗಾತ್ರ_v_ಖಂಡ;
    L->ಗಾತ್ರ_m_ಖಂಡ = ಗಾತ್ರ_m_ಖಂಡ;
    L->ಗಾತ್ರ_vkq_acc = ಗಾತ್ರ_vkq_acc;
    L->ಗಾತ್ರ_sinks   = has_sinks ? ಗಾತ್ರ_sinks : 0;
    L->ಒಟ್ಟು_ಬೈಟುಗಳು  = off;
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ size_t hvx_fa_compute_vtcm_usage(size_t DK, size_t DV, bool is_q_fp32, bool has_ಮುಖವಾಡ, bool has_sinks, size_t n_heads, size_t n_ಎಳೆಗಳು) {
    ರಚನೆ hvx_fa_vtcm_layout L;
    hvx_fa_vtcm_layout_build(&L, DK, DV, is_q_fp32, has_ಮುಖವಾಡ, has_sinks, n_heads, n_ಎಳೆಗಳು);
    ಹಿಂತಿರುಗಿ L.ಒಟ್ಟು_ಬೈಟುಗಳು;
}

#ವ್ಯಾಖ್ಯಾನಿಸು FA_MIN_KV_ಖಂಡಗಳು 3

// Cost-based (Br, Bc) search for flash attention with pipeline constraint.
ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಪೂರ್ಣಾಂಕ hmx_fa_ಕಂಡುಹಿಡಿ_chunk_ಗಾತ್ರ(size_t * Br_ಹೊರಹೋಗು,
                                         size_t * Bc_ಹೊರಹೋಗು,
                                         size_t   gqa_ಗುಣಕ,
                                         size_t   DK,
                                         size_t   DV,
                                         size_t   qo_len,
                                         size_t   kv_len,
                                         size_t   vtcm_budget,
                                         size_t   n_ಎಳೆಗಳು,
                                         bool     is_q_fp32,
                                         bool     has_sinks,
                                         size_t   n_heads) {
    ಸ್ಥಿರಾಂಕ size_t T       = HMX_FP16_TILE_N_ಅಡ್ಡಸಾಲುಗಳು;  // 32
    ಸ್ಥಿರಾಂಕ size_t br_unit = hmx_ceil_div(T, gqa_ಗುಣಕ);
    ಸ್ಥಿರಾಂಕ size_t bc_unit = HMX_FP16_TILE_N_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ * 2;  // 64
    ಸ್ಥಿರಾಂಕ bool   can_pipeline = (kv_len >= FA_MIN_KV_ಖಂಡಗಳು * bc_unit && n_ಎಳೆಗಳು >= 2);

    // Br_max: largest Br aligned to br_unit that does not exceed qo_len.
    ಸ್ಥಿರಾಂಕ size_t Br_max = qo_len >= br_unit ? hex_align_ಕೆಳಕ್ಕೆ(qo_len, br_unit) : br_unit;

    // Pipeline constraint: cap Bc so n_kv_blocks >= FA_MIN_KV_BLOCKS.
    // Only relax when kv_len is too short to form enough blocks.
    size_t Bc_ಎಲ್ಲೆ     = can_pipeline ? hex_align_ಕೆಳಕ್ಕೆ(kv_len / FA_MIN_KV_ಖಂಡಗಳು, bc_unit) :
                                         (kv_len >= bc_unit ? hex_align_ಕೆಳಕ್ಕೆ(kv_len, bc_unit) : bc_unit);
    // For small-Q SWA-sized KV caches (qo_len <= 16, kv_len <= 2048), cap Bc at 256 for block-skipping granularity
    ಆದರೆ (qo_len <= 16 && kv_len <= 2048 && Bc_ಎಲ್ಲೆ > 256) {
        Bc_ಎಲ್ಲೆ = 256;
    }
    // Cost coefficients calibrated from profiling
    ಸ್ಥಿರಾಂಕ size_t c_q_fixed    = 800;   // per-Q-block: q_load + epilogue o_update + o_norm + o_store
    ಸ್ಥಿರಾಂಕ size_t c_iter_ತಳ  = 200;   // per-KV-iter base (HMX dot/update + DMA)
    ಸ್ಥಿರಾಂಕ size_t c_softmax    = 600;   // per 64-row vector chunk on HVX

    size_t ಉತ್ತಮ_ವೆಚ್ಚ = SIZE_MAX, ಉತ್ತಮ_mn = 0;
    size_t ಉತ್ತಮ_Br = 0, ಉತ್ತಮ_Bc = 0;

    ಗಾಗಿ (size_t Br = Br_max; Br >= br_unit; Br -= br_unit) {
        // Try all Bc candidates from Bc_limit down to bc_unit
        ಗಾಗಿ (size_t Bc = Bc_ಎಲ್ಲೆ; Bc >= bc_unit; Bc -= bc_unit) {
            size_t vtcm_needed = hmx_fa_compute_vtcm_usage(gqa_ಗುಣಕ, DK, DV, Br, Bc, n_ಎಳೆಗಳು, can_pipeline, is_q_fp32, has_sinks, n_heads);
            ಆದರೆ (vtcm_needed <= vtcm_budget) {
                // This Bc fits for this Br!
                ಸ್ಥಿರಾಂಕ size_t q_ಖಂಡಗಳು       = (qo_len + Br - 1) / Br;
                ಸ್ಥಿರಾಂಕ size_t kv_ಖಂಡಗಳು      = (kv_len + Bc - 1) / Bc;
                ಸ್ಥಿರಾಂಕ size_t actual_ಎಳೆಗಳು = (kv_ಖಂಡಗಳು >= 3 && n_ಎಳೆಗಳು >= 2) ? n_ಎಳೆಗಳು : 1;
                ಸ್ಥಿರಾಂಕ size_t n_ಅಡ್ಡಸಾಲುಗಳು_g       = Br * gqa_ಗುಣಕ;
                ಸ್ಥಿರಾಂಕ size_t n_ಅಡ್ಡಸಾಲು_vec_cnt  = (n_ಅಡ್ಡಸಾಲುಗಳು_g + 63) / 64;
                ಸ್ಥಿರಾಂಕ size_t n_use          = n_ಅಡ್ಡಸಾಲು_vec_cnt < actual_ಎಳೆಗಳು ? n_ಅಡ್ಡಸಾಲು_vec_cnt : actual_ಎಳೆಗಳು;
                ಸ್ಥಿರಾಂಕ size_t vecs_ಪ್ರತಿ_t     = n_use > 0 ? (n_ಅಡ್ಡಸಾಲು_vec_cnt + n_use - 1) / n_use : 1;

                ಸ್ಥಿರಾಂಕ size_t c_iter_actual  = c_iter_ತಳ + c_softmax * vecs_ಪ್ರತಿ_t;
                ಸ್ಥಿರಾಂಕ size_t ವೆಚ್ಚ           = q_ಖಂಡಗಳು * (c_q_fixed + kv_ಖಂಡಗಳು * c_iter_actual);
                ಸ್ಥಿರಾಂಕ size_t mn             = Br * Bc;

                ಆದರೆ (ವೆಚ್ಚ < ಉತ್ತಮ_ವೆಚ್ಚ || (ವೆಚ್ಚ == ಉತ್ತಮ_ವೆಚ್ಚ && mn > ಉತ್ತಮ_mn)) {
                    ಉತ್ತಮ_ವೆಚ್ಚ = ವೆಚ್ಚ;
                    ಉತ್ತಮ_mn   = mn;
                    ಉತ್ತಮ_Br   = Br;
                    ಉತ್ತಮ_Bc   = Bc;
                }
                // Since we iterate Bc from largest to smallest, this is the largest Bc that fits
                // for this Br. We can break to the next Br.
                ನಿಲ್ಲಿಸು;
            }
        }

        ಆದರೆ (Br == br_unit) {
            ನಿಲ್ಲಿಸು;
        }
    }

    ಆದರೆ (ಉತ್ತಮ_Br == 0 || ಉತ್ತಮ_Bc == 0) {
        ಹಿಂತಿರುಗಿ -1;
    }

    *Br_ಹೊರಹೋಗು = ಉತ್ತಮ_Br;
    *Bc_ಹೊರಹೋಗು = ಉತ್ತಮ_Bc;
    ಹಿಂತಿರುಗಿ 0;
}

#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ __cplusplus
}
#ಆದರೆ_ಅಂತ್ಯ

#ಆದರೆ_ಅಂತ್ಯ /* HTP_FLASH_ATTN_OPS_H */
