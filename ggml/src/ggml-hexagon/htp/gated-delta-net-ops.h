#ਜੇ_ਅਪਰਿਭਾਸ਼ਿਤ HTP_GATED_DELTA_NET_OPS_H
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ HTP_GATED_DELTA_NET_OPS_H

#ਅਨ੍ਤਰ੍ਭਾਵ <stdint.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <stddef.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <stdbool.h>

#ਅਨ੍ਤਰ੍ਭਾਵ "hex-fastdiv.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "hex-common.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "htp-vtcm.h"

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ HTP_GDN_MAX_SV     128
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ HTP_GDN_CHUNK_ਮਾਪ 64
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ HTP_GDN_MIN_ਟੋਕਨ 8

#ਜੇ_ਅਪਰਿਭਾਸ਼ਿਤ HMX_FP16_TILE_ਮਾਪ
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ HMX_FP16_TILE_ਮਾਪ 2048
#ਜੇ_ਅੰਤ

ਗਿਣਤੀ htp_gdn_kernel_type {
    HTP_GDN_KERNEL_HVX_RECURRENT = 0,
    HTP_GDN_KERNEL_HMX_CHUNKED   = 1,
};

ਬਣਤਰ htp_gdn_kernel_params {
    uint8_t  kernel_type;
    uint8_t  pipeline;
    uint16_t chunk_ਮਾਪ;
    uint16_t n_chunks;
    uint16_t n_heads_batch;

    uint32_t n_ਧਾਗੇ;
    uint32_t S_v;
    uint32_t H;
    uint32_t n_ਟੋਕਨ;
    uint32_t n_seqs;
    uint32_t K;

    uint32_t ਕੁੱਲ_ਪੰਕਤੀਆਂ;
    uint32_t ਪੰਕਤੀ_ਸ਼ੁਰੂ;
    uint32_t nrows;
    uint32_t ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_thread;

    uint32_t kda;
    uint32_t state_aligned;
    uint32_t vtcm_ਪ੍ਰਤੀ_thread;
    uint32_t vtcm_ਮਾਪ;
    uint32_t state_seq_stride;
    uint32_t state_ਮਾਪ_ਪ੍ਰਤੀ_snap;

    ਦਸ਼ਮਲਵ    scale;

    ਬਣਤਰ fastdiv_ਮੁੱਲਾਂ div_H;
    ਬਣਤਰ fastdiv_ਮੁੱਲਾਂ div_q1;
    ਬਣਤਰ fastdiv_ਮੁੱਲਾਂ div_k1;
    ਬਣਤਰ fastdiv_ਮੁੱਲਾਂ div_rq3;
    ਬਣਤਰ fastdiv_ਮੁੱਲਾਂ div_rk3;
    ਬਣਤਰ fastdiv_ਮੁੱਲਾਂ div_n_ਧਾਗੇ;
};

#ਜੇ defined(__cplusplus)
static_assert(ਆਕਾਰ(ਬਣਤਰ htp_gdn_kernel_params) <= 128, "htp_gdn_kernel_params is too large for kernel_params blob");
#ਨਹੀਂਤਾਂ
_Static_assert(ਆਕਾਰ(ਬਣਤਰ htp_gdn_kernel_params) <= 128, "htp_gdn_kernel_params is too large for kernel_params blob");
#ਜੇ_ਅੰਤ

ਬਣਤਰ htp_gdn_vtcm_layout {
    size_t state_aligned;
    size_t ਬਾਈਟਾਂ_ਪ੍ਰਤੀ_thread;
    size_t ਕੁੱਲ_ਬਾਈਟਾਂ;
};

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ htp_gdn_vtcm_layout_build(
    ਬਣਤਰ htp_gdn_vtcm_layout * layout,
    uint32_t S_v,
    uint32_t n_ਧਾਗੇ
) {
    size_t state_ਬਾਈਟਾਂ = (size_t) S_v * S_v * ਆਕਾਰ(ਦਸ਼ਮਲਵ);
    layout->state_aligned = hex_ਗੇੜਾ_ਉੱਪਰ(state_ਬਾਈਟਾਂ, 128);
    layout->ਬਾਈਟਾਂ_ਪ੍ਰਤੀ_thread = 2 * layout->state_aligned;
    layout->ਕੁੱਲ_ਬਾਈਟਾਂ = layout->ਬਾਈਟਾਂ_ਪ੍ਰਤੀ_thread * n_ਧਾਗੇ;
}

ਬਣਤਰ htp_gdn_hmx_vtcm_layout {
    size_t off_s_state;
    size_t off_s_f16;
    size_t off_s_ਖਾਨਾ_tiles;
    size_t off_s_update_f32;
    size_t off_s_update_tiles;

    size_t off_q_f32[2];
    size_t off_k_f32[2];
    size_t off_v_f32[2];
    size_t off_g_f32[2];
    size_t off_b_f32[2];
    size_t off_g_raw[2];
    size_t off_b_raw[2];
    size_t off_o_f32[2];

    size_t off_v_inter_f32;
    size_t off_o_inter_f32;
    size_t off_o_intra_f32;
    size_t off_k_f16;
    size_t off_v_ਅਭਾਜ_f16;
    size_t off_delta_f16;
    size_t off_d_f16;

    size_t off_q_ਪੰਕਤੀ_tiles;
    size_t off_q_ਅਭਾਜ_ਪੰਕਤੀ_tiles;
    size_t off_k_ਪੰਕਤੀ_tiles;
    size_t off_k_ਖਾਨਾ_tiles;
    size_t off_k_ਅਭਾਜ_ਪੰਕਤੀ_tiles;
    size_t off_k_ਖਾਨਾ_tiles_64x128;
    size_t off_kk_tiles;
    size_t off_qk_tiles;
    size_t off_v_inter_tiles;
    size_t off_o_inter_tiles;
    size_t off_inv_ਪੰਕਤੀ_tiles;
    size_t off_a_ਪੰਕਤੀ_tiles;
    size_t off_v_ਅਭਾਜ_ਖਾਨਾ_tiles;
    size_t off_delta_tiles;
    size_t off_delta_ਖਾਨਾ_tiles;
    size_t off_o_intra_tiles;
    size_t off_d_ਪੰਕਤੀ_tiles;

    size_t off_gamma;
    size_t off_lambda_init;
    size_t off_decay_m;
    size_t off_decay_a;
    size_t off_ਪੰਕਤੀਆਂ_kk;
    size_t off_ਪੰਕਤੀਆਂ_qk;
    size_t off_ਪੰਕਤੀਆਂ_inv;
    size_t off_ਪੰਕਤੀਆਂ_a;

    size_t off_thread_scratch;
    size_t off_scales_1;

    size_t state_f32_ਬਾਈਟਾਂ;
    size_t state_f16_ਬਾਈਟਾਂ;
    size_t state_tiles_ਬਾਈਟਾਂ;
    size_t dma_chunk_ਬਾਈਟਾਂ;
    size_t act_f16_ਬਾਈਟਾਂ;
    size_t tile_64x_Sv_ਬਾਈਟਾਂ;
    size_t tile_64x64_ਬਾਈਟਾਂ;

    uint32_t n_heads_batch;
    uint32_t n_ਧਾਗੇ;
    bool     pipeline;
    size_t   ਕੁੱਲ_ਬਾਈਟਾਂ;
};

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ htp_gdn_hmx_vtcm_layout_build(
    ਬਣਤਰ htp_gdn_hmx_vtcm_layout * L,
    uint32_t S_v,
    uint32_t chunk_ਮਾਪ,
    uint32_t n_heads_batch,
    uint32_t n_ਧਾਗੇ,
    bool     pipeline
) {
    memset(L, 0, ਆਕਾਰ(*L));
    L->n_heads_batch = n_heads_batch;
    L->n_ਧਾਗੇ     = n_ਧਾਗੇ;
    L->pipeline      = pipeline;

    ਅਚਲ size_t bh = (size_t) n_heads_batch;
    ਅਚਲ size_t nth = (size_t) (n_ਧਾਗੇ > 0 ? n_ਧਾਗੇ : 1);

    ਅਚਲ size_t state_f32_sz   = hex_ਗੇੜਾ_ਉੱਪਰ(S_v * S_v * ਆਕਾਰ(ਦਸ਼ਮਲਵ), 2048);
    ਅਚਲ size_t state_f16_sz   = hex_ਗੇੜਾ_ਉੱਪਰ(S_v * S_v * ਆਕਾਰ(__fp16), 2048);
    ਅਚਲ size_t n_sv_tiles     = S_v / 32;
    ਅਚਲ size_t state_tiles_sz = n_sv_tiles * n_sv_tiles * 2048;

    ਅਚਲ size_t dma_chunk_sz   = hex_ਗੇੜਾ_ਉੱਪਰ(chunk_ਮਾਪ * S_v * ਆਕਾਰ(ਦਸ਼ਮਲਵ), 2048);
    ਅਚਲ size_t dma_scalar_sz  = hex_ਗੇੜਾ_ਉੱਪਰ(chunk_ਮਾਪ * ਆਕਾਰ(ਦਸ਼ਮਲਵ), 128);

    ਅਚਲ size_t act_f16_sz     = hex_ਗੇੜਾ_ਉੱਪਰ(chunk_ਮਾਪ * S_v * ਆਕਾਰ(__fp16), 2048);
    ਅਚਲ size_t tile_64xSv_sz  = 2 * n_sv_tiles * 2048;
    ਅਚਲ size_t tile_64x64_sz  = 4 * 2048;

    ਅਚਲ size_t decay_sz       = 64 * 64 * ਆਕਾਰ(__fp16);
    ਅਚਲ size_t ਪੰਕਤੀ_vecs_sz    = 64 * 128;

    L->state_f32_ਬਾਈਟਾਂ   = state_f32_sz;
    L->state_f16_ਬਾਈਟਾਂ   = state_f16_sz;
    L->state_tiles_ਬਾਈਟਾਂ = state_tiles_sz;
    L->dma_chunk_ਬਾਈਟਾਂ   = dma_chunk_sz;
    L->act_f16_ਬਾਈਟਾਂ     = act_f16_sz;
    L->tile_64x_Sv_ਬਾਈਟਾਂ  = tile_64xSv_sz;
    L->tile_64x64_ਬਾਈਟਾਂ  = tile_64x64_sz;

    size_t off = 0;

    VTCM_LAYOUT_ALLOC(off, off_s_state,        bh * state_f32_sz);
    VTCM_LAYOUT_ALLOC(off, off_s_f16,          bh * state_f16_sz);
    off = hex_align_ਉੱਪਰ(off, HMX_FP16_TILE_ਮਾਪ);
    VTCM_LAYOUT_ALLOC(off, off_s_ਖਾਨਾ_tiles,    bh * state_tiles_sz);
    VTCM_LAYOUT_ALLOC(off, off_s_update_f32,   bh * state_f32_sz);
    off = hex_align_ਉੱਪਰ(off, HMX_FP16_TILE_ਮਾਪ);
    VTCM_LAYOUT_ALLOC(off, off_s_update_tiles, bh * state_tiles_sz);

    VTCM_LAYOUT_ALLOC(off, off_q_f32[0], bh * dma_chunk_sz);
    VTCM_LAYOUT_ALLOC_OPTIONAL(off, off_q_f32[1], bh * dma_chunk_sz, pipeline);
    VTCM_LAYOUT_ALLOC(off, off_k_f32[0], bh * dma_chunk_sz);
    VTCM_LAYOUT_ALLOC_OPTIONAL(off, off_k_f32[1], bh * dma_chunk_sz, pipeline);
    VTCM_LAYOUT_ALLOC(off, off_v_f32[0], bh * dma_chunk_sz);
    VTCM_LAYOUT_ALLOC_OPTIONAL(off, off_v_f32[1], bh * dma_chunk_sz, pipeline);
    VTCM_LAYOUT_ALLOC(off, off_g_f32[0], bh * dma_scalar_sz);
    VTCM_LAYOUT_ALLOC_OPTIONAL(off, off_g_f32[1], bh * dma_scalar_sz, pipeline);
    VTCM_LAYOUT_ALLOC(off, off_b_f32[0], bh * dma_scalar_sz);
    VTCM_LAYOUT_ALLOC_OPTIONAL(off, off_b_f32[1], bh * dma_scalar_sz, pipeline);
    ਅਚਲ size_t raw_gb_sz = hex_ਗੇੜਾ_ਉੱਪਰ(bh * chunk_ਮਾਪ * ਆਕਾਰ(ਦਸ਼ਮਲਵ), 128);
    VTCM_LAYOUT_ALLOC(off, off_g_raw[0], raw_gb_sz);
    VTCM_LAYOUT_ALLOC_OPTIONAL(off, off_g_raw[1], raw_gb_sz, pipeline);
    VTCM_LAYOUT_ALLOC(off, off_b_raw[0], raw_gb_sz);
    VTCM_LAYOUT_ALLOC_OPTIONAL(off, off_b_raw[1], raw_gb_sz, pipeline);
    VTCM_LAYOUT_ALLOC(off, off_o_f32[0], bh * dma_chunk_sz);
    VTCM_LAYOUT_ALLOC_OPTIONAL(off, off_o_f32[1], bh * dma_chunk_sz, pipeline);

    VTCM_LAYOUT_ALLOC(off, off_v_inter_f32, bh * dma_chunk_sz);
    VTCM_LAYOUT_ALLOC(off, off_o_inter_f32, bh * dma_chunk_sz);
    VTCM_LAYOUT_ALLOC(off, off_o_intra_f32, bh * dma_chunk_sz);
    VTCM_LAYOUT_ALLOC(off, off_k_f16,       bh * act_f16_sz);
    VTCM_LAYOUT_ALLOC(off, off_v_ਅਭਾਜ_f16, bh * act_f16_sz);
    VTCM_LAYOUT_ALLOC(off, off_delta_f16,   bh * act_f16_sz);
    VTCM_LAYOUT_ALLOC(off, off_d_f16,       bh * act_f16_sz);

    off = hex_align_ਉੱਪਰ(off, HMX_FP16_TILE_ਮਾਪ);
    VTCM_LAYOUT_ALLOC(off, off_q_ਪੰਕਤੀ_tiles,        bh * tile_64xSv_sz);
    VTCM_LAYOUT_ALLOC(off, off_q_ਅਭਾਜ_ਪੰਕਤੀ_tiles,  bh * tile_64xSv_sz);
    VTCM_LAYOUT_ALLOC(off, off_k_ਪੰਕਤੀ_tiles,        bh * tile_64xSv_sz);
    VTCM_LAYOUT_ALLOC(off, off_k_ਖਾਨਾ_tiles,        bh * tile_64xSv_sz);
    VTCM_LAYOUT_ALLOC(off, off_k_ਅਭਾਜ_ਪੰਕਤੀ_tiles,  bh * tile_64xSv_sz);
    VTCM_LAYOUT_ALLOC(off, off_k_ਖਾਨਾ_tiles_64x128, bh * tile_64xSv_sz);
    VTCM_LAYOUT_ALLOC(off, off_kk_tiles,           bh * tile_64x64_sz);
    VTCM_LAYOUT_ALLOC(off, off_qk_tiles,           bh * tile_64x64_sz);
    VTCM_LAYOUT_ALLOC(off, off_v_inter_tiles,      bh * tile_64xSv_sz);
    VTCM_LAYOUT_ALLOC(off, off_o_inter_tiles,      bh * tile_64xSv_sz);
    VTCM_LAYOUT_ALLOC(off, off_inv_ਪੰਕਤੀ_tiles,      bh * tile_64x64_sz);
    VTCM_LAYOUT_ALLOC(off, off_a_ਪੰਕਤੀ_tiles,        bh * tile_64x64_sz);
    VTCM_LAYOUT_ALLOC(off, off_v_ਅਭਾਜ_ਖਾਨਾ_tiles,  bh * tile_64xSv_sz);
    VTCM_LAYOUT_ALLOC(off, off_delta_tiles,        bh * tile_64xSv_sz);
    VTCM_LAYOUT_ALLOC(off, off_delta_ਖਾਨਾ_tiles,    bh * tile_64xSv_sz);
    VTCM_LAYOUT_ALLOC(off, off_o_intra_tiles,      bh * tile_64xSv_sz);
    VTCM_LAYOUT_ALLOC(off, off_d_ਪੰਕਤੀ_tiles,        bh * tile_64xSv_sz);

    VTCM_LAYOUT_ALLOC(off, off_gamma,       bh * hex_ਗੇੜਾ_ਉੱਪਰ(chunk_ਮਾਪ * ਆਕਾਰ(ਦਸ਼ਮਲਵ), 128));
    VTCM_LAYOUT_ALLOC(off, off_lambda_init, bh * hex_ਗੇੜਾ_ਉੱਪਰ(chunk_ਮਾਪ * ਆਕਾਰ(ਦਸ਼ਮਲਵ), 128));
    VTCM_LAYOUT_ALLOC(off, off_decay_m,     bh * decay_sz);
    VTCM_LAYOUT_ALLOC(off, off_decay_a,     bh * decay_sz);
    VTCM_LAYOUT_ALLOC(off, off_ਪੰਕਤੀਆਂ_kk,     bh * ਪੰਕਤੀ_vecs_sz);
    VTCM_LAYOUT_ALLOC(off, off_ਪੰਕਤੀਆਂ_qk,     bh * ਪੰਕਤੀ_vecs_sz);
    VTCM_LAYOUT_ALLOC(off, off_ਪੰਕਤੀਆਂ_inv,    bh * ਪੰਕਤੀ_vecs_sz);
    VTCM_LAYOUT_ALLOC(off, off_ਪੰਕਤੀਆਂ_a,      bh * ਪੰਕਤੀ_vecs_sz);

    ਅਚਲ size_t thread_scratch_sz = 64 * 128;
    off = hex_align_ਉੱਪਰ(off, HMX_FP16_TILE_ਮਾਪ);
    VTCM_LAYOUT_ALLOC(off, off_thread_scratch, nth * thread_scratch_sz);
    off = hex_align_ਉੱਪਰ(off, HMX_FP16_TILE_ਮਾਪ);
    VTCM_LAYOUT_ALLOC(off, off_scales_1,       HMX_FP16_TILE_ਮਾਪ);

    L->ਕੁੱਲ_ਬਾਈਟਾਂ = off;
}

ਸਥਿਰ ਸਤਰਅੰਦਰ bool htp_gdn_hmx_solve_layout(
    ਬਣਤਰ htp_gdn_hmx_vtcm_layout * layout_ਬਾਹਰਲਾ,
    uint32_t S_v,
    uint32_t chunk_ਮਾਪ,
    uint32_t ਕੁੱਲ_ਪੰਕਤੀਆਂ,
    size_t   vtcm_budget,
    uint32_t n_ਧਾਗੇ,
    bool     pipeline,
    uint32_t * n_heads_batch_ਬਾਹਰਲਾ
) {
    uint32_t max_batch = 8;
    ਜੇ (max_batch > ਕੁੱਲ_ਪੰਕਤੀਆਂ) {
        max_batch = ਕੁੱਲ_ਪੰਕਤੀਆਂ;
    }
    ਜੇ (max_batch > n_ਧਾਗੇ) {
        max_batch = n_ਧਾਗੇ;
    }
    ਸਥਿਰ ਅਚਲ uint32_t candidates[] = { 8, 6, 4, 2, 1 };
    ਲਈ (size_t i = 0; i < ਆਕਾਰ(candidates) / ਆਕਾਰ(candidates[0]); ++i) {
        uint32_t bh = candidates[i];
        ਜੇ (bh > max_batch) {
            ਜਾਰੀ;
        }
        ਬਣਤਰ htp_gdn_hmx_vtcm_layout L;
        htp_gdn_hmx_vtcm_layout_build(&L, S_v, chunk_ਮਾਪ, bh, n_ਧਾਗੇ, pipeline);
        ਜੇ (L.ਕੁੱਲ_ਬਾਈਟਾਂ <= vtcm_budget) {
            *layout_ਬਾਹਰਲਾ = L;
            *n_heads_batch_ਬਾਹਰਲਾ = bh;
            ਵਾਪਸ true;
        }
    }
    ਜੇ (pipeline) {
        ਵਾਪਸ htp_gdn_hmx_solve_layout(layout_ਬਾਹਰਲਾ, S_v, chunk_ਮਾਪ, ਕੁੱਲ_ਪੰਕਤੀਆਂ, vtcm_budget, n_ਧਾਗੇ, false, n_heads_batch_ਬਾਹਰਲਾ);
    }
    ਵਾਪਸ false;
}

#ਜੇ_ਅੰਤ // HTP_GATED_DELTA_NET_OPS_H
