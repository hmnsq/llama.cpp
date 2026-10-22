#ਜੇ_ਅਪਰਿਭਾਸ਼ਿਤ HTP_FLASH_ATTN_OPS_H
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ HTP_FLASH_ATTN_OPS_H

#ਅਨ੍ਤਰ੍ਭਾਵ <stdint.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <stddef.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <stdbool.h>

#ਅਨ੍ਤਰ੍ਭਾਵ "hex-fastdiv.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "hex-common.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "htp-vtcm.h"

#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ __cplusplus
ਬਾਹਰੀ "C" {
#ਜੇ_ਅੰਤ

// Tile constants (mirrored from hmx-utils.h for use on host side if needed)
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ HTP_FA_HMX_TILE_ਮਾਪ   2048
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ HMX_FP16_TILE_ਮਾਪ     2048
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ HMX_FP16_TILE_N_ਪੰਕਤੀਆਂ   32
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ HMX_FP16_TILE_N_ਖਾਨੇ_ਗਿਣਤੀ   32
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ HMX_FP16_TILE_N_ELMS   1024

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ HVX_FA_DMA_CACHE_ਮਾਪ  128
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ HMX_FA_DMA_CACHE_ਮਾਪ  4


#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ HTP_FA_M_INITIAL_VAL  -10000.0f

ਗਿਣਤੀ htp_fa_kernel_type {
    HTP_FA_KERNEL_UNSUPPORTED = 0,
    HTP_FA_KERNEL_HVX,
    HTP_FA_KERNEL_HMX
};

ਬਣਤਰ htp_fa_kernel_params {
    uint8_t  kernel_type;        // enum htp_fa_kernel_type
    uint8_t  ਸਿਰ_ਵੰਡੋ;         // 1 = partition by KV heads in multicore, 0 = token partition
    uint8_t  flags;              // reserved
    uint8_t  n_ਧਾਗੇ;          // Number of threads to run

    // Common parameters
    uint16_t Br;
    uint16_t Bc;
    uint16_t n_kv_ਖੰਡਾਂ;        // also HVX's n_blocks
    uint16_t G;                  // GQA factor (n_heads / n_kv_heads)

    ਦਸ਼ਮਲਵ    scale;
    ਦਸ਼ਮਲਵ    max_bias;
    ਦਸ਼ਮਲਵ    logit_softcap;
    uint32_t vtcm_ਮਾਪ;

    uint32_t qrows;
    uint32_t qrows_ਪ੍ਰਤੀ_thread;
    uint32_t qrow_ਸ਼ੁਰੂ;
    ਦਸ਼ਮਲਵ    m0;
    ਦਸ਼ਮਲਵ    m1;
    uint32_t n_ਸਿਰ_log2;

    ਬਣਤਰ fastdiv_ਮੁੱਲਾਂ src3_div2;
    ਬਣਤਰ fastdiv_ਮੁੱਲਾਂ src3_div3;

    ਬਣਤਰ fastdiv_ਮੁੱਲਾਂ ਪ੍ਰਸਾਰਣ_rk2;
    ਬਣਤਰ fastdiv_ਮੁੱਲਾਂ ਪ੍ਰਸਾਰਣ_rk3;
    ਬਣਤਰ fastdiv_ਮੁੱਲਾਂ ਪ੍ਰਸਾਰਣ_rv2;
    ਬਣਤਰ fastdiv_ਮੁੱਲਾਂ ਪ੍ਰਸਾਰਣ_rv3;

    ਸੰਘ {
        ਬਣਤਰ {
            uint32_t g_br;
            uint32_t ਪੰਕਤੀ_buf_stride;
            uint32_t ਨਕਾਬ_buf_ਪੰਕਤੀ_stride;
            int32_t  ਨਕਾਬ_ਪ੍ਰਸਾਰਣ;
            int32_t  pipeline;
            ਬਣਤਰ fastdiv_ਮੁੱਲਾਂ div_G;
        } hmx;
        ਬਣਤਰ {
            uint32_t ਮਾਪ_q_ਪੰਕਤੀ_padded;
            uint32_t ਮਾਪ_k_ਪੰਕਤੀ_padded;
            uint32_t ਮਾਪ_v_ਪੰਕਤੀ_padded;
            ਬਣਤਰ fastdiv_ਮੁੱਲਾਂ src0_div21;
            ਬਣਤਰ fastdiv_ਮੁੱਲਾਂ src0_div1;
        } hvx;
    } u;
};

#ਜੇ defined(__cplusplus)
static_assert(ਆਕਾਰ(ਬਣਤਰ htp_fa_kernel_params) <= 128, "htp_fa_kernel_params is too large for kernel_params blob");
#ਜੇ_ਅੰਤ

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
ਬਣਤਰ hmx_fa_vtcm_layout {
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
    size_t off_ਪੰਕਤੀ_bufs;
    size_t off_hmx_scales_id;
    size_t off_hmx_scales_qk;
    size_t off_ਨਕਾਬ_buf;
    size_t off_slopes;
    size_t off_sinks;

    // Region byte sizes reused by the device at runtime (not just for allocation).
    size_t q_tile_ਬਾਈਟਾਂ;
    size_t o_tile_ਬਾਈਟਾਂ;
    size_t k_tile_ਬਾਈਟਾਂ;
    size_t s_tile_ਬਾਈਟਾਂ;       // S and P tiles (same size)
    size_t d_tile_ਬਾਈਟਾਂ;       // d_tiles[0..1] + d_inv_l, allocated back to back
    size_t m_ਸਤਰ_ਬਾਈਟਾਂ;       // one mask row
    size_t m_buf_ਥਾਂ_ਬਾਈਟਾਂ;   // one dma_cache slot = align_up(Br * m_line_bytes, 4096)
    size_t ਖਾਨਾ_vec_ਬਾਈਟਾਂ;
    size_t sinks_ਬਾਈਟਾਂ;

    // Derived strides.
    size_t ਪੰਕਤੀ_buf_stride;       // HVX vectors (128B) per row buffer
    size_t ਨਕਾਬ_buf_ਪੰਕਤੀ_stride;  // __fp16 elements per row in the mask buffer

    bool   pipeline;
    size_t ਕੁੱਲ_ਬਾਈਟਾਂ;
};

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ FA_MAX_KV_ਖੰਡਾਂ 1024

// Build the VTCM layout.

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ hmx_fa_vtcm_layout_build(ਬਣਤਰ hmx_fa_vtcm_layout * L,
                                            size_t gqa_ਗੁਣਕ, size_t DK, size_t DV,
                                            size_t Br, size_t Bc, size_t n_ਧਾਗੇ,
                                            bool pipeline, bool is_q_fp32,
                                            bool has_sinks, size_t n_heads) {
    ਅਚਲ size_t g_br         = hex_align_ਉੱਪਰ(gqa_ਗੁਣਕ * Br, HMX_FP16_TILE_N_ਪੰਕਤੀਆਂ);
    ਅਚਲ size_t q_tile_ਮਾਪ  = hex_align_ਉੱਪਰ(g_br * DK   * ਆਕਾਰ(__fp16), HTP_FA_HMX_TILE_ਮਾਪ);
    ਅਚਲ size_t o_tile_ਮਾਪ  = hex_align_ਉੱਪਰ(g_br * DV   * ਆਕਾਰ(__fp16), HTP_FA_HMX_TILE_ਮਾਪ);
    ਅਚਲ size_t k_tile_ਮਾਪ  = hex_align_ਉੱਪਰ(Bc   * DK   * ਆਕਾਰ(__fp16), HTP_FA_HMX_TILE_ਮਾਪ);
    ਅਚਲ size_t v_tile_ਮਾਪ  = hex_align_ਉੱਪਰ(Bc   * DV   * ਆਕਾਰ(__fp16), HTP_FA_HMX_TILE_ਮਾਪ);
    ਅਚਲ size_t s_tile_ਮਾਪ  = hex_align_ਉੱਪਰ(g_br * Bc   * ਆਕਾਰ(__fp16), HTP_FA_HMX_TILE_ਮਾਪ);

    // The rescale matrices are diagonal: the HMX kernels only ever load the g_br/32
    // tiles that sit on the diagonal, so store just those, packed back to back with
    // a stride of one tile.  The old [g_br, g_br] square layout allocated g_br/32
    // times more than it used, which is also why a second D buffer was unaffordable.
    ਅਚਲ size_t d_tile_ਮਾਪ = (g_br / HMX_FP16_TILE_N_ਪੰਕਤੀਆਂ) * HTP_FA_HMX_TILE_ਮਾਪ;

    ਅਚਲ size_t q_dma_ਮਾਪ   = hex_align_ਉੱਪਰ(g_br * DK * (is_q_fp32 ? ਆਕਾਰ(ਦਸ਼ਮਲਵ) : ਆਕਾਰ(__fp16)), 128);
    ਅਚਲ size_t k_dma_ਮਾਪ   = hex_align_ਉੱਪਰ(Bc * hex_ਗੇੜਾ_ਉੱਪਰ(DK * ਆਕਾਰ(__fp16), 128), 128);
    ਅਚਲ size_t v_dma_ਮਾਪ   = hex_align_ਉੱਪਰ(Bc * hex_ਗੇੜਾ_ਉੱਪਰ(DV * ਆਕਾਰ(__fp16), 128), 128);
    ਅਚਲ size_t ਖਾਨਾ_vec_ਮਾਪ = hex_align_ਉੱਪਰ(g_br * ਆਕਾਰ(ਦਸ਼ਮਲਵ),  256);
    ਅਚਲ size_t ਪੰਕਤੀ_vec_ਮਾਪ = hex_align_ਉੱਪਰ(Bc   * ਆਕਾਰ(__fp16), 256);
    ਅਚਲ size_t m_ਸਤਰ_ਮਾਪ  = hex_align_ਉੱਪਰ(Bc   * ਆਕਾਰ(__fp16), 128);
    ਅਚਲ size_t m_buf_ਥਾਂ   = hex_align_ਉੱਪਰ(Br * m_ਸਤਰ_ਮਾਪ, 256);
    ਅਚਲ size_t m_buf_ਮਾਪ   = m_buf_ਥਾਂ * HMX_FA_DMA_CACHE_ਮਾਪ;
    ਅਚਲ size_t slopes_ਮਾਪ  = hex_align_ਉੱਪਰ(g_br * ਆਕਾਰ(__fp16), 128);
    ਅਚਲ size_t sinks_ਮਾਪ   = hex_ਗੇੜਾ_ਉੱਪਰ(n_heads * ਆਕਾਰ(ਦਸ਼ਮਲਵ), 128);

    size_t off = 0;

    // Group A (Part 1 - HMX Tiled buffers)
    VTCM_LAYOUT_ALLOC(off, off_q_tiles,       q_tile_ਮਾਪ);
    VTCM_LAYOUT_ALLOC(off, off_o_tiles[0],    o_tile_ਮਾਪ);
    VTCM_LAYOUT_ALLOC(off, off_o_tiles[1],    o_tile_ਮਾਪ);
    VTCM_LAYOUT_ALLOC(off, off_d_tiles[0],    d_tile_ਮਾਪ);
    VTCM_LAYOUT_ALLOC_OPTIONAL(off, off_d_tiles[1], d_tile_ਮਾਪ, pipeline);
    VTCM_LAYOUT_ALLOC(off, off_d_inv_l,       d_tile_ਮਾਪ);

    // Group B & C share start offset (Group B tiles must be 2KB aligned)
    size_t off_ਸਮੂਹ_b_c = hex_align_ਉੱਪਰ(off, HTP_FA_HMX_TILE_ਮਾਪ);

    // Group B: Compute-only buffers
    size_t off_ਸਮੂਹ_b = off_ਸਮੂਹ_b_c;
    VTCM_LAYOUT_ALLOC(off_ਸਮੂਹ_b, off_k_tiles[0],    k_tile_ਮਾਪ);
    VTCM_LAYOUT_ALLOC_OPTIONAL(off_ਸਮੂਹ_b, off_k_tiles[1], k_tile_ਮਾਪ, pipeline);
    VTCM_LAYOUT_ALLOC(off_ਸਮੂਹ_b, off_v_tiles[0],    v_tile_ਮਾਪ);
    VTCM_LAYOUT_ALLOC_OPTIONAL(off_ਸਮੂਹ_b, off_v_tiles[1], v_tile_ਮਾਪ, pipeline);
    VTCM_LAYOUT_ALLOC(off_ਸਮੂਹ_b, off_s_tiles[0],    s_tile_ਮਾਪ);
    VTCM_LAYOUT_ALLOC_OPTIONAL(off_ਸਮੂਹ_b, off_s_tiles[1], s_tile_ਮਾਪ, pipeline);
    VTCM_LAYOUT_ALLOC(off_ਸਮੂਹ_b, off_p_tiles[0],    s_tile_ਮਾਪ);
    VTCM_LAYOUT_ALLOC_OPTIONAL(off_ਸਮੂਹ_b, off_p_tiles[1], s_tile_ਮਾਪ, pipeline);
    VTCM_LAYOUT_ALLOC(off_ਸਮੂਹ_b, off_s_rowmax,      ਖਾਨਾ_vec_ਮਾਪ);
    VTCM_LAYOUT_ALLOC(off_ਸਮੂਹ_b, off_p_rowsum,      ਖਾਨਾ_vec_ਮਾਪ);
    VTCM_LAYOUT_ALLOC(off_ਸਮੂਹ_b, off_ਪੰਕਤੀ_bufs,      ਪੰਕਤੀ_vec_ਮਾਪ * 2 * n_ਧਾਗੇ);

    ਅਚਲ size_t ਸਮੂਹ_b_ਮਾਪ = off_ਸਮੂਹ_b - off_ਸਮੂਹ_b_c;

    // Group C: Q fetch DMA buffer
    size_t off_ਸਮੂਹ_c = off_ਸਮੂਹ_b_c;
    VTCM_LAYOUT_ALLOC(off_ਸਮੂਹ_c, off_q_dma,         q_dma_ਮਾਪ);

    ਅਚਲ size_t ਸਮੂਹ_c_ਮਾਪ = off_ਸਮੂਹ_c - off_ਸਮੂਹ_b_c;

    off = off_ਸਮੂਹ_b_c + hex_smax(ਸਮੂਹ_b_ਮਾਪ, ਸਮੂਹ_c_ਮਾਪ);

    // Group A (Part 2 - remaining non-HMX buffers)
    VTCM_LAYOUT_ALLOC(off, off_k_fp16[0],     k_dma_ਮਾਪ);
    VTCM_LAYOUT_ALLOC(off, off_k_fp16[1],     k_dma_ਮਾਪ);
    VTCM_LAYOUT_ALLOC(off, off_v_fp16[0],     v_dma_ਮਾਪ);
    VTCM_LAYOUT_ALLOC(off, off_v_fp16[1],     v_dma_ਮਾਪ);
    VTCM_LAYOUT_ALLOC(off, off_m_vec,         ਖਾਨਾ_vec_ਮਾਪ);
    VTCM_LAYOUT_ALLOC(off, off_l_vec,         ਖਾਨਾ_vec_ਮਾਪ);
    VTCM_LAYOUT_ALLOC(off, off_hmx_scales_id, 256);
    VTCM_LAYOUT_ALLOC(off, off_hmx_scales_qk, 256);
    VTCM_LAYOUT_ALLOC(off, off_ਨਕਾਬ_buf,      m_buf_ਮਾਪ);
    VTCM_LAYOUT_ALLOC(off, off_slopes,        slopes_ਮਾਪ);
    VTCM_LAYOUT_ALLOC_OPTIONAL(off, off_sinks, sinks_ਮਾਪ, has_sinks);

    L->q_tile_ਬਾਈਟਾਂ        = q_tile_ਮਾਪ;
    L->o_tile_ਬਾਈਟਾਂ        = o_tile_ਮਾਪ;
    L->k_tile_ਬਾਈਟਾਂ        = k_tile_ਮਾਪ;
    L->ਖਾਨਾ_vec_ਬਾਈਟਾਂ       = ਖਾਨਾ_vec_ਮਾਪ;
    L->s_tile_ਬਾਈਟਾਂ        = s_tile_ਮਾਪ;
    // Measured from the actual offsets rather than assumed to be N * d_tile_size, so
    // that inserting a region between them (or adding padding to VTCM_LAYOUT_ALLOC)
    // cannot silently leave the tail of the run unzeroed.
    L->d_tile_ਬਾਈਟਾਂ        = (L->off_d_inv_l + d_tile_ਮਾਪ) - L->off_d_tiles[0];
    L->m_ਸਤਰ_ਬਾਈਟਾਂ        = m_ਸਤਰ_ਮਾਪ;
    L->m_buf_ਥਾਂ_ਬਾਈਟਾਂ    = m_buf_ਥਾਂ;
    L->ਪੰਕਤੀ_buf_stride      = ਪੰਕਤੀ_vec_ਮਾਪ / 128;
    L->ਨਕਾਬ_buf_ਪੰਕਤੀ_stride = m_ਸਤਰ_ਮਾਪ / ਆਕਾਰ(__fp16);
    L->sinks_ਬਾਈਟਾਂ         = has_sinks ? sinks_ਮਾਪ : 0;
    L->pipeline            = pipeline;
    L->ਕੁੱਲ_ਬਾਈਟਾਂ         = off;
}

// Exact VTCM usage for a given (gqa_factor, DK, DV, Br, Bc) configuration.
ਸਥਿਰ ਸਤਰਅੰਦਰ size_t hmx_fa_compute_vtcm_usage(size_t gqa_ਗੁਣਕ, size_t DK, size_t DV, size_t Br, size_t Bc, size_t n_ਧਾਗੇ, bool pipeline, bool is_q_fp32, bool has_sinks, size_t n_heads) {
    ਬਣਤਰ hmx_fa_vtcm_layout L;
    hmx_fa_vtcm_layout_build(&L, gqa_ਗੁਣਕ, DK, DV, Br, Bc, n_ਧਾਗੇ, pipeline, is_q_fp32, has_sinks, n_heads);
    ਵਾਪਸ L.ਕੁੱਲ_ਬਾਈਟਾਂ;
}

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ FA_HVX_ਖੰਡ_ਮਾਪ 64
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ FA_HVX_G_MAX      8

ਬਣਤਰ hvx_fa_vtcm_layout {
    size_t off_q;
    size_t off_k;
    size_t off_v;
    size_t off_m;
    size_t off_a;
    size_t off_sinks;

    size_t ਮਾਪ_q_ਖੰਡ;
    size_t ਮਾਪ_k_ਖੰਡ;
    size_t ਮਾਪ_v_ਖੰਡ;
    size_t ਮਾਪ_m_ਖੰਡ;
    size_t ਮਾਪ_vkq_acc;
    size_t ਮਾਪ_sinks;

    size_t ਕੁੱਲ_ਬਾਈਟਾਂ;
};

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ hvx_fa_vtcm_layout_build(ਬਣਤਰ hvx_fa_vtcm_layout * L,
                                            size_t DK, size_t DV,
                                            bool is_q_fp32, bool has_ਨਕਾਬ,
                                            bool has_sinks, size_t n_heads,
                                            size_t n_ਧਾਗੇ) {
    ਅਚਲ size_t ਮਾਪ_q_ਪੰਕਤੀ_padded = hex_ਗੇੜਾ_ਉੱਪਰ(DK * (is_q_fp32 ? 4 : 2), 128);
    ਅਚਲ size_t ਮਾਪ_k_ਪੰਕਤੀ_padded = hex_ਗੇੜਾ_ਉੱਪਰ(DK * ਆਕਾਰ(__fp16), 128);
    ਅਚਲ size_t ਮਾਪ_v_ਪੰਕਤੀ_padded = hex_ਗੇੜਾ_ਉੱਪਰ(DV * ਆਕਾਰ(__fp16), 128);

    ਅਚਲ size_t ਮਾਪ_q_ਖੰਡ = ਮਾਪ_q_ਪੰਕਤੀ_padded * FA_HVX_G_MAX;
    ਅਚਲ size_t ਮਾਪ_k_ਖੰਡ = ਮਾਪ_k_ਪੰਕਤੀ_padded * FA_HVX_ਖੰਡ_ਮਾਪ;
    ਅਚਲ size_t ਮਾਪ_v_ਖੰਡ = ਮਾਪ_v_ਪੰਕਤੀ_padded * FA_HVX_ਖੰਡ_ਮਾਪ;
    ਅਚਲ size_t ਮਾਪ_m_ਖੰਡ = hex_ਗੇੜਾ_ਉੱਪਰ(FA_HVX_ਖੰਡ_ਮਾਪ * ਆਕਾਰ(__fp16), 128);
    ਅਚਲ size_t ਮਾਪ_vkq_acc = hex_ਗੇੜਾ_ਉੱਪਰ(DV * ਆਕਾਰ(ਦਸ਼ਮਲਵ), 128) * FA_HVX_G_MAX;
    ਅਚਲ size_t ਮਾਪ_sinks   = hex_ਗੇੜਾ_ਉੱਪਰ(n_heads * ਆਕਾਰ(ਦਸ਼ਮਲਵ), 128);

    size_t off = 0;

    VTCM_LAYOUT_ALLOC(off, off_q, ਮਾਪ_q_ਖੰਡ * n_ਧਾਗੇ);
    VTCM_LAYOUT_ALLOC(off, off_k, ਮਾਪ_k_ਖੰਡ * 2 * n_ਧਾਗੇ);
    VTCM_LAYOUT_ALLOC(off, off_v, ਮਾਪ_v_ਖੰਡ * 2 * n_ਧਾਗੇ);
    VTCM_LAYOUT_ALLOC_OPTIONAL(off, off_m, ਮਾਪ_m_ਖੰਡ * HVX_FA_DMA_CACHE_ਮਾਪ * n_ਧਾਗੇ, has_ਨਕਾਬ);
    VTCM_LAYOUT_ALLOC(off, off_a, ਮਾਪ_vkq_acc * n_ਧਾਗੇ);
    VTCM_LAYOUT_ALLOC_OPTIONAL(off, off_sinks, ਮਾਪ_sinks, has_sinks);

    L->ਮਾਪ_q_ਖੰਡ = ਮਾਪ_q_ਖੰਡ;
    L->ਮਾਪ_k_ਖੰਡ = ਮਾਪ_k_ਖੰਡ;
    L->ਮਾਪ_v_ਖੰਡ = ਮਾਪ_v_ਖੰਡ;
    L->ਮਾਪ_m_ਖੰਡ = ਮਾਪ_m_ਖੰਡ;
    L->ਮਾਪ_vkq_acc = ਮਾਪ_vkq_acc;
    L->ਮਾਪ_sinks   = has_sinks ? ਮਾਪ_sinks : 0;
    L->ਕੁੱਲ_ਬਾਈਟਾਂ  = off;
}

ਸਥਿਰ ਸਤਰਅੰਦਰ size_t hvx_fa_compute_vtcm_usage(size_t DK, size_t DV, bool is_q_fp32, bool has_ਨਕਾਬ, bool has_sinks, size_t n_heads, size_t n_ਧਾਗੇ) {
    ਬਣਤਰ hvx_fa_vtcm_layout L;
    hvx_fa_vtcm_layout_build(&L, DK, DV, is_q_fp32, has_ਨਕਾਬ, has_sinks, n_heads, n_ਧਾਗੇ);
    ਵਾਪਸ L.ਕੁੱਲ_ਬਾਈਟਾਂ;
}

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ FA_MIN_KV_ਖੰਡਾਂ 3

// Cost-based (Br, Bc) search for flash attention with pipeline constraint.
ਸਥਿਰ ਸਤਰਅੰਦਰ ਪੂਰਨ hmx_fa_ਲੱਭੋ_chunk_ਮਾਪ(size_t * Br_ਬਾਹਰਲਾ,
                                         size_t * Bc_ਬਾਹਰਲਾ,
                                         size_t   gqa_ਗੁਣਕ,
                                         size_t   DK,
                                         size_t   DV,
                                         size_t   qo_len,
                                         size_t   kv_len,
                                         size_t   vtcm_budget,
                                         size_t   n_ਧਾਗੇ,
                                         bool     is_q_fp32,
                                         bool     has_sinks,
                                         size_t   n_heads) {
    ਅਚਲ size_t T       = HMX_FP16_TILE_N_ਪੰਕਤੀਆਂ;  // 32
    ਅਚਲ size_t br_unit = hmx_ceil_div(T, gqa_ਗੁਣਕ);
    ਅਚਲ size_t bc_unit = HMX_FP16_TILE_N_ਖਾਨੇ_ਗਿਣਤੀ * 2;  // 64
    ਅਚਲ bool   can_pipeline = (kv_len >= FA_MIN_KV_ਖੰਡਾਂ * bc_unit && n_ਧਾਗੇ >= 2);

    // Br_max: largest Br aligned to br_unit that does not exceed qo_len.
    ਅਚਲ size_t Br_max = qo_len >= br_unit ? hex_align_ਥੱਲੇ(qo_len, br_unit) : br_unit;

    // Pipeline constraint: cap Bc so n_kv_blocks >= FA_MIN_KV_BLOCKS.
    // Only relax when kv_len is too short to form enough blocks.
    size_t Bc_ਹੱਦਬੰਦੀ     = can_pipeline ? hex_align_ਥੱਲੇ(kv_len / FA_MIN_KV_ਖੰਡਾਂ, bc_unit) :
                                         (kv_len >= bc_unit ? hex_align_ਥੱਲੇ(kv_len, bc_unit) : bc_unit);
    // For small-Q SWA-sized KV caches (qo_len <= 16, kv_len <= 2048), cap Bc at 256 for block-skipping granularity
    ਜੇ (qo_len <= 16 && kv_len <= 2048 && Bc_ਹੱਦਬੰਦੀ > 256) {
        Bc_ਹੱਦਬੰਦੀ = 256;
    }
    // Cost coefficients calibrated from profiling
    ਅਚਲ size_t c_q_fixed    = 800;   // per-Q-block: q_load + epilogue o_update + o_norm + o_store
    ਅਚਲ size_t c_iter_ਅਧਾਰ  = 200;   // per-KV-iter base (HMX dot/update + DMA)
    ਅਚਲ size_t c_softmax    = 600;   // per 64-row vector chunk on HVX

    size_t ਉੱਤਮ_ਲਾਗਤ = SIZE_MAX, ਉੱਤਮ_mn = 0;
    size_t ਉੱਤਮ_Br = 0, ਉੱਤਮ_Bc = 0;

    ਲਈ (size_t Br = Br_max; Br >= br_unit; Br -= br_unit) {
        // Try all Bc candidates from Bc_limit down to bc_unit
        ਲਈ (size_t Bc = Bc_ਹੱਦਬੰਦੀ; Bc >= bc_unit; Bc -= bc_unit) {
            size_t vtcm_needed = hmx_fa_compute_vtcm_usage(gqa_ਗੁਣਕ, DK, DV, Br, Bc, n_ਧਾਗੇ, can_pipeline, is_q_fp32, has_sinks, n_heads);
            ਜੇ (vtcm_needed <= vtcm_budget) {
                // This Bc fits for this Br!
                ਅਚਲ size_t q_ਖੰਡਾਂ       = (qo_len + Br - 1) / Br;
                ਅਚਲ size_t kv_ਖੰਡਾਂ      = (kv_len + Bc - 1) / Bc;
                ਅਚਲ size_t actual_ਧਾਗੇ = (kv_ਖੰਡਾਂ >= 3 && n_ਧਾਗੇ >= 2) ? n_ਧਾਗੇ : 1;
                ਅਚਲ size_t n_ਪੰਕਤੀਆਂ_g       = Br * gqa_ਗੁਣਕ;
                ਅਚਲ size_t n_ਪੰਕਤੀ_vec_cnt  = (n_ਪੰਕਤੀਆਂ_g + 63) / 64;
                ਅਚਲ size_t n_use          = n_ਪੰਕਤੀ_vec_cnt < actual_ਧਾਗੇ ? n_ਪੰਕਤੀ_vec_cnt : actual_ਧਾਗੇ;
                ਅਚਲ size_t vecs_ਪ੍ਰਤੀ_t     = n_use > 0 ? (n_ਪੰਕਤੀ_vec_cnt + n_use - 1) / n_use : 1;

                ਅਚਲ size_t c_iter_actual  = c_iter_ਅਧਾਰ + c_softmax * vecs_ਪ੍ਰਤੀ_t;
                ਅਚਲ size_t ਲਾਗਤ           = q_ਖੰਡਾਂ * (c_q_fixed + kv_ਖੰਡਾਂ * c_iter_actual);
                ਅਚਲ size_t mn             = Br * Bc;

                ਜੇ (ਲਾਗਤ < ਉੱਤਮ_ਲਾਗਤ || (ਲਾਗਤ == ਉੱਤਮ_ਲਾਗਤ && mn > ਉੱਤਮ_mn)) {
                    ਉੱਤਮ_ਲਾਗਤ = ਲਾਗਤ;
                    ਉੱਤਮ_mn   = mn;
                    ਉੱਤਮ_Br   = Br;
                    ਉੱਤਮ_Bc   = Bc;
                }
                // Since we iterate Bc from largest to smallest, this is the largest Bc that fits
                // for this Br. We can break to the next Br.
                ਰੋਕੋ;
            }
        }

        ਜੇ (Br == br_unit) {
            ਰੋਕੋ;
        }
    }

    ਜੇ (ਉੱਤਮ_Br == 0 || ਉੱਤਮ_Bc == 0) {
        ਵਾਪਸ -1;
    }

    *Br_ਬਾਹਰਲਾ = ਉੱਤਮ_Br;
    *Bc_ਬਾਹਰਲਾ = ਉੱਤਮ_Bc;
    ਵਾਪਸ 0;
}

#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ __cplusplus
}
#ਜੇ_ਅੰਤ

#ਜੇ_ਅੰਤ /* HTP_FLASH_ATTN_OPS_H */
