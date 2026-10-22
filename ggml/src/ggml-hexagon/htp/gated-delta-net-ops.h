#ವ್ಯಾಖ್ಯಾನಿತವಾಗದಿದ್ದರೆ HTP_GATED_DELTA_NET_OPS_H
#ವ್ಯಾಖ್ಯಾನಿಸು HTP_GATED_DELTA_NET_OPS_H

#ಅನ್ತರ್ಭಾವ <stdint.h>
#ಅನ್ತರ್ಭಾವ <stddef.h>
#ಅನ್ತರ್ಭಾವ <stdbool.h>

#ಅನ್ತರ್ಭಾವ "hex-fastdiv.h"
#ಅನ್ತರ್ಭಾವ "hex-common.h"
#ಅನ್ತರ್ಭಾವ "htp-vtcm.h"

#ವ್ಯಾಖ್ಯಾನಿಸು HTP_GDN_MAX_SV     128
#ವ್ಯಾಖ್ಯಾನಿಸು HTP_GDN_CHUNK_ಗಾತ್ರ 64
#ವ್ಯಾಖ್ಯಾನಿಸು HTP_GDN_MIN_ಟೋಕನುಗಳು 8

#ವ್ಯಾಖ್ಯಾನಿತವಾಗದಿದ್ದರೆ HMX_FP16_TILE_ಗಾತ್ರ
#ವ್ಯಾಖ್ಯಾನಿಸು HMX_FP16_TILE_ಗಾತ್ರ 2048
#ಆದರೆ_ಅಂತ್ಯ

ಎಣಿಕೆ htp_gdn_kernel_type {
    HTP_GDN_KERNEL_HVX_RECURRENT = 0,
    HTP_GDN_KERNEL_HMX_CHUNKED   = 1,
};

ರಚನೆ htp_gdn_kernel_params {
    uint8_t  kernel_type;
    uint8_t  pipeline;
    uint16_t chunk_ಗಾತ್ರ;
    uint16_t n_chunks;
    uint16_t n_heads_batch;

    uint32_t n_ಎಳೆಗಳು;
    uint32_t S_v;
    uint32_t H;
    uint32_t n_ಟೋಕನುಗಳು;
    uint32_t n_seqs;
    uint32_t K;

    uint32_t ಒಟ್ಟು_ಅಡ್ಡಸಾಲುಗಳು;
    uint32_t ಅಡ್ಡಸಾಲು_ಆರಂಭ;
    uint32_t nrows;
    uint32_t ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_thread;

    uint32_t kda;
    uint32_t state_aligned;
    uint32_t vtcm_ಪ್ರತಿ_thread;
    uint32_t vtcm_ಗಾತ್ರ;
    uint32_t state_seq_stride;
    uint32_t state_ಗಾತ್ರ_ಪ್ರತಿ_snap;

    ದಶಮಾಂಶ    scale;

    ರಚನೆ fastdiv_ಮೌಲ್ಯಗಳು div_H;
    ರಚನೆ fastdiv_ಮೌಲ್ಯಗಳು div_q1;
    ರಚನೆ fastdiv_ಮೌಲ್ಯಗಳು div_k1;
    ರಚನೆ fastdiv_ಮೌಲ್ಯಗಳು div_rq3;
    ರಚನೆ fastdiv_ಮೌಲ್ಯಗಳು div_rk3;
    ರಚನೆ fastdiv_ಮೌಲ್ಯಗಳು div_n_ಎಳೆಗಳು;
};

#ಆದರೆ defined(__cplusplus)
static_assert(ಗಾತ್ರ(ರಚನೆ htp_gdn_kernel_params) <= 128, "htp_gdn_kernel_params is too large for kernel_params blob");
#ಇಲ್ಲದಿದ್ದರೆ
_Static_assert(ಗಾತ್ರ(ರಚನೆ htp_gdn_kernel_params) <= 128, "htp_gdn_kernel_params is too large for kernel_params blob");
#ಆದರೆ_ಅಂತ್ಯ

ರಚನೆ htp_gdn_vtcm_layout {
    size_t state_aligned;
    size_t ಬೈಟುಗಳು_ಪ್ರತಿ_thread;
    size_t ಒಟ್ಟು_ಬೈಟುಗಳು;
};

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ htp_gdn_vtcm_layout_build(
    ರಚನೆ htp_gdn_vtcm_layout * layout,
    uint32_t S_v,
    uint32_t n_ಎಳೆಗಳು
) {
    size_t state_ಬೈಟುಗಳು = (size_t) S_v * S_v * ಗಾತ್ರ(ದಶಮಾಂಶ);
    layout->state_aligned = hex_ಸುತ್ತು_ಮೇಲೆ(state_ಬೈಟುಗಳು, 128);
    layout->ಬೈಟುಗಳು_ಪ್ರತಿ_thread = 2 * layout->state_aligned;
    layout->ಒಟ್ಟು_ಬೈಟುಗಳು = layout->ಬೈಟುಗಳು_ಪ್ರತಿ_thread * n_ಎಳೆಗಳು;
}

ರಚನೆ htp_gdn_hmx_vtcm_layout {
    size_t off_s_state;
    size_t off_s_f16;
    size_t off_s_ಲಂಬಸಾಲು_tiles;
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
    size_t off_v_ಅವಿಭಾಜ್ಯ_f16;
    size_t off_delta_f16;
    size_t off_d_f16;

    size_t off_q_ಅಡ್ಡಸಾಲು_tiles;
    size_t off_q_ಅವಿಭಾಜ್ಯ_ಅಡ್ಡಸಾಲು_tiles;
    size_t off_k_ಅಡ್ಡಸಾಲು_tiles;
    size_t off_k_ಲಂಬಸಾಲು_tiles;
    size_t off_k_ಅವಿಭಾಜ್ಯ_ಅಡ್ಡಸಾಲು_tiles;
    size_t off_k_ಲಂಬಸಾಲು_tiles_64x128;
    size_t off_kk_tiles;
    size_t off_qk_tiles;
    size_t off_v_inter_tiles;
    size_t off_o_inter_tiles;
    size_t off_inv_ಅಡ್ಡಸಾಲು_tiles;
    size_t off_a_ಅಡ್ಡಸಾಲು_tiles;
    size_t off_v_ಅವಿಭಾಜ್ಯ_ಲಂಬಸಾಲು_tiles;
    size_t off_delta_tiles;
    size_t off_delta_ಲಂಬಸಾಲು_tiles;
    size_t off_o_intra_tiles;
    size_t off_d_ಅಡ್ಡಸಾಲು_tiles;

    size_t off_gamma;
    size_t off_lambda_init;
    size_t off_decay_m;
    size_t off_decay_a;
    size_t off_ಅಡ್ಡಸಾಲುಗಳು_kk;
    size_t off_ಅಡ್ಡಸಾಲುಗಳು_qk;
    size_t off_ಅಡ್ಡಸಾಲುಗಳು_inv;
    size_t off_ಅಡ್ಡಸಾಲುಗಳು_a;

    size_t off_thread_scratch;
    size_t off_scales_1;

    size_t state_f32_ಬೈಟುಗಳು;
    size_t state_f16_ಬೈಟುಗಳು;
    size_t state_tiles_ಬೈಟುಗಳು;
    size_t dma_chunk_ಬೈಟುಗಳು;
    size_t act_f16_ಬೈಟುಗಳು;
    size_t tile_64x_Sv_ಬೈಟುಗಳು;
    size_t tile_64x64_ಬೈಟುಗಳು;

    uint32_t n_heads_batch;
    uint32_t n_ಎಳೆಗಳು;
    bool     pipeline;
    size_t   ಒಟ್ಟು_ಬೈಟುಗಳು;
};

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ htp_gdn_hmx_vtcm_layout_build(
    ರಚನೆ htp_gdn_hmx_vtcm_layout * L,
    uint32_t S_v,
    uint32_t chunk_ಗಾತ್ರ,
    uint32_t n_heads_batch,
    uint32_t n_ಎಳೆಗಳು,
    bool     pipeline
) {
    memset(L, 0, ಗಾತ್ರ(*L));
    L->n_heads_batch = n_heads_batch;
    L->n_ಎಳೆಗಳು     = n_ಎಳೆಗಳು;
    L->pipeline      = pipeline;

    ಸ್ಥಿರಾಂಕ size_t bh = (size_t) n_heads_batch;
    ಸ್ಥಿರಾಂಕ size_t nth = (size_t) (n_ಎಳೆಗಳು > 0 ? n_ಎಳೆಗಳು : 1);

    ಸ್ಥಿರಾಂಕ size_t state_f32_sz   = hex_ಸುತ್ತು_ಮೇಲೆ(S_v * S_v * ಗಾತ್ರ(ದಶಮಾಂಶ), 2048);
    ಸ್ಥಿರಾಂಕ size_t state_f16_sz   = hex_ಸುತ್ತು_ಮೇಲೆ(S_v * S_v * ಗಾತ್ರ(__fp16), 2048);
    ಸ್ಥಿರಾಂಕ size_t n_sv_tiles     = S_v / 32;
    ಸ್ಥಿರಾಂಕ size_t state_tiles_sz = n_sv_tiles * n_sv_tiles * 2048;

    ಸ್ಥಿರಾಂಕ size_t dma_chunk_sz   = hex_ಸುತ್ತು_ಮೇಲೆ(chunk_ಗಾತ್ರ * S_v * ಗಾತ್ರ(ದಶಮಾಂಶ), 2048);
    ಸ್ಥಿರಾಂಕ size_t dma_scalar_sz  = hex_ಸುತ್ತು_ಮೇಲೆ(chunk_ಗಾತ್ರ * ಗಾತ್ರ(ದಶಮಾಂಶ), 128);

    ಸ್ಥಿರಾಂಕ size_t act_f16_sz     = hex_ಸುತ್ತು_ಮೇಲೆ(chunk_ಗಾತ್ರ * S_v * ಗಾತ್ರ(__fp16), 2048);
    ಸ್ಥಿರಾಂಕ size_t tile_64xSv_sz  = 2 * n_sv_tiles * 2048;
    ಸ್ಥಿರಾಂಕ size_t tile_64x64_sz  = 4 * 2048;

    ಸ್ಥಿರಾಂಕ size_t decay_sz       = 64 * 64 * ಗಾತ್ರ(__fp16);
    ಸ್ಥಿರಾಂಕ size_t ಅಡ್ಡಸಾಲು_vecs_sz    = 64 * 128;

    L->state_f32_ಬೈಟುಗಳು   = state_f32_sz;
    L->state_f16_ಬೈಟುಗಳು   = state_f16_sz;
    L->state_tiles_ಬೈಟುಗಳು = state_tiles_sz;
    L->dma_chunk_ಬೈಟುಗಳು   = dma_chunk_sz;
    L->act_f16_ಬೈಟುಗಳು     = act_f16_sz;
    L->tile_64x_Sv_ಬೈಟುಗಳು  = tile_64xSv_sz;
    L->tile_64x64_ಬೈಟುಗಳು  = tile_64x64_sz;

    size_t off = 0;

    VTCM_LAYOUT_ALLOC(off, off_s_state,        bh * state_f32_sz);
    VTCM_LAYOUT_ALLOC(off, off_s_f16,          bh * state_f16_sz);
    off = hex_align_ಮೇಲೆ(off, HMX_FP16_TILE_ಗಾತ್ರ);
    VTCM_LAYOUT_ALLOC(off, off_s_ಲಂಬಸಾಲು_tiles,    bh * state_tiles_sz);
    VTCM_LAYOUT_ALLOC(off, off_s_update_f32,   bh * state_f32_sz);
    off = hex_align_ಮೇಲೆ(off, HMX_FP16_TILE_ಗಾತ್ರ);
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
    ಸ್ಥಿರಾಂಕ size_t raw_gb_sz = hex_ಸುತ್ತು_ಮೇಲೆ(bh * chunk_ಗಾತ್ರ * ಗಾತ್ರ(ದಶಮಾಂಶ), 128);
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
    VTCM_LAYOUT_ALLOC(off, off_v_ಅವಿಭಾಜ್ಯ_f16, bh * act_f16_sz);
    VTCM_LAYOUT_ALLOC(off, off_delta_f16,   bh * act_f16_sz);
    VTCM_LAYOUT_ALLOC(off, off_d_f16,       bh * act_f16_sz);

    off = hex_align_ಮೇಲೆ(off, HMX_FP16_TILE_ಗಾತ್ರ);
    VTCM_LAYOUT_ALLOC(off, off_q_ಅಡ್ಡಸಾಲು_tiles,        bh * tile_64xSv_sz);
    VTCM_LAYOUT_ALLOC(off, off_q_ಅವಿಭಾಜ್ಯ_ಅಡ್ಡಸಾಲು_tiles,  bh * tile_64xSv_sz);
    VTCM_LAYOUT_ALLOC(off, off_k_ಅಡ್ಡಸಾಲು_tiles,        bh * tile_64xSv_sz);
    VTCM_LAYOUT_ALLOC(off, off_k_ಲಂಬಸಾಲು_tiles,        bh * tile_64xSv_sz);
    VTCM_LAYOUT_ALLOC(off, off_k_ಅವಿಭಾಜ್ಯ_ಅಡ್ಡಸಾಲು_tiles,  bh * tile_64xSv_sz);
    VTCM_LAYOUT_ALLOC(off, off_k_ಲಂಬಸಾಲು_tiles_64x128, bh * tile_64xSv_sz);
    VTCM_LAYOUT_ALLOC(off, off_kk_tiles,           bh * tile_64x64_sz);
    VTCM_LAYOUT_ALLOC(off, off_qk_tiles,           bh * tile_64x64_sz);
    VTCM_LAYOUT_ALLOC(off, off_v_inter_tiles,      bh * tile_64xSv_sz);
    VTCM_LAYOUT_ALLOC(off, off_o_inter_tiles,      bh * tile_64xSv_sz);
    VTCM_LAYOUT_ALLOC(off, off_inv_ಅಡ್ಡಸಾಲು_tiles,      bh * tile_64x64_sz);
    VTCM_LAYOUT_ALLOC(off, off_a_ಅಡ್ಡಸಾಲು_tiles,        bh * tile_64x64_sz);
    VTCM_LAYOUT_ALLOC(off, off_v_ಅವಿಭಾಜ್ಯ_ಲಂಬಸಾಲು_tiles,  bh * tile_64xSv_sz);
    VTCM_LAYOUT_ALLOC(off, off_delta_tiles,        bh * tile_64xSv_sz);
    VTCM_LAYOUT_ALLOC(off, off_delta_ಲಂಬಸಾಲು_tiles,    bh * tile_64xSv_sz);
    VTCM_LAYOUT_ALLOC(off, off_o_intra_tiles,      bh * tile_64xSv_sz);
    VTCM_LAYOUT_ALLOC(off, off_d_ಅಡ್ಡಸಾಲು_tiles,        bh * tile_64xSv_sz);

    VTCM_LAYOUT_ALLOC(off, off_gamma,       bh * hex_ಸುತ್ತು_ಮೇಲೆ(chunk_ಗಾತ್ರ * ಗಾತ್ರ(ದಶಮಾಂಶ), 128));
    VTCM_LAYOUT_ALLOC(off, off_lambda_init, bh * hex_ಸುತ್ತು_ಮೇಲೆ(chunk_ಗಾತ್ರ * ಗಾತ್ರ(ದಶಮಾಂಶ), 128));
    VTCM_LAYOUT_ALLOC(off, off_decay_m,     bh * decay_sz);
    VTCM_LAYOUT_ALLOC(off, off_decay_a,     bh * decay_sz);
    VTCM_LAYOUT_ALLOC(off, off_ಅಡ್ಡಸಾಲುಗಳು_kk,     bh * ಅಡ್ಡಸಾಲು_vecs_sz);
    VTCM_LAYOUT_ALLOC(off, off_ಅಡ್ಡಸಾಲುಗಳು_qk,     bh * ಅಡ್ಡಸಾಲು_vecs_sz);
    VTCM_LAYOUT_ALLOC(off, off_ಅಡ್ಡಸಾಲುಗಳು_inv,    bh * ಅಡ್ಡಸಾಲು_vecs_sz);
    VTCM_LAYOUT_ALLOC(off, off_ಅಡ್ಡಸಾಲುಗಳು_a,      bh * ಅಡ್ಡಸಾಲು_vecs_sz);

    ಸ್ಥಿರಾಂಕ size_t thread_scratch_sz = 64 * 128;
    off = hex_align_ಮೇಲೆ(off, HMX_FP16_TILE_ಗಾತ್ರ);
    VTCM_LAYOUT_ALLOC(off, off_thread_scratch, nth * thread_scratch_sz);
    off = hex_align_ಮೇಲೆ(off, HMX_FP16_TILE_ಗಾತ್ರ);
    VTCM_LAYOUT_ALLOC(off, off_scales_1,       HMX_FP16_TILE_ಗಾತ್ರ);

    L->ಒಟ್ಟು_ಬೈಟುಗಳು = off;
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ bool htp_gdn_hmx_solve_layout(
    ರಚನೆ htp_gdn_hmx_vtcm_layout * layout_ಹೊರಹೋಗು,
    uint32_t S_v,
    uint32_t chunk_ಗಾತ್ರ,
    uint32_t ಒಟ್ಟು_ಅಡ್ಡಸಾಲುಗಳು,
    size_t   vtcm_budget,
    uint32_t n_ಎಳೆಗಳು,
    bool     pipeline,
    uint32_t * n_heads_batch_ಹೊರಹೋಗು
) {
    uint32_t max_batch = 8;
    ಆದರೆ (max_batch > ಒಟ್ಟು_ಅಡ್ಡಸಾಲುಗಳು) {
        max_batch = ಒಟ್ಟು_ಅಡ್ಡಸಾಲುಗಳು;
    }
    ಆದರೆ (max_batch > n_ಎಳೆಗಳು) {
        max_batch = n_ಎಳೆಗಳು;
    }
    ಸ್ಥಿರ ಸ್ಥಿರಾಂಕ uint32_t candidates[] = { 8, 6, 4, 2, 1 };
    ಗಾಗಿ (size_t i = 0; i < ಗಾತ್ರ(candidates) / ಗಾತ್ರ(candidates[0]); ++i) {
        uint32_t bh = candidates[i];
        ಆದರೆ (bh > max_batch) {
            ಮುಂದುವರಿ;
        }
        ರಚನೆ htp_gdn_hmx_vtcm_layout L;
        htp_gdn_hmx_vtcm_layout_build(&L, S_v, chunk_ಗಾತ್ರ, bh, n_ಎಳೆಗಳು, pipeline);
        ಆದರೆ (L.ಒಟ್ಟು_ಬೈಟುಗಳು <= vtcm_budget) {
            *layout_ಹೊರಹೋಗು = L;
            *n_heads_batch_ಹೊರಹೋಗು = bh;
            ಹಿಂತಿರುಗಿ true;
        }
    }
    ಆದರೆ (pipeline) {
        ಹಿಂತಿರುಗಿ htp_gdn_hmx_solve_layout(layout_ಹೊರಹೋಗು, S_v, chunk_ಗಾತ್ರ, ಒಟ್ಟು_ಅಡ್ಡಸಾಲುಗಳು, vtcm_budget, n_ಎಳೆಗಳು, false, n_heads_batch_ಹೊರಹೋಗು);
    }
    ಹಿಂತಿರುಗಿ false;
}

#ಆದರೆ_ಅಂತ್ಯ // HTP_GATED_DELTA_NET_OPS_H
