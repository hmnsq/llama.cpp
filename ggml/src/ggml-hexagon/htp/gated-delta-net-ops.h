#अगर_अघोषित HTP_GATED_DELTA_NET_OPS_H
#घोषणा HTP_GATED_DELTA_NET_OPS_H

#समावेश <stdint.h>
#समावेश <stddef.h>
#समावेश <stdbool.h>

#समावेश "hex-fastdiv.h"
#समावेश "hex-common.h"
#समावेश "htp-vtcm.h"

#घोषणा HTP_GDN_MAX_SV     128
#घोषणा HTP_GDN_CHUNK_आकार 64
#घोषणा HTP_GDN_MIN_TOKENS 8

#अगर_अघोषित HMX_FP16_TILE_आकार
#घोषणा HMX_FP16_TILE_आकार 2048
#पूर्ण_अगर

क्रमागत htp_gdn_kernel_type {
    HTP_GDN_KERNEL_HVX_RECURRENT = 0,
    HTP_GDN_KERNEL_HMX_CHUNKED   = 1,
};

काष्ठा htp_gdn_kernel_params {
    uint8_t  kernel_type;
    uint8_t  pipeline;
    uint16_t chunk_आकार;
    uint16_t n_chunks;
    uint16_t n_heads_batch;

    uint32_t n_threads;
    uint32_t S_v;
    uint32_t H;
    uint32_t n_tokens;
    uint32_t n_seqs;
    uint32_t K;

    uint32_t योग_rows;
    uint32_t row_start;
    uint32_t nrows;
    uint32_t rows_per_thread;

    uint32_t kda;
    uint32_t state_aligned;
    uint32_t vtcm_per_thread;
    uint32_t vtcm_आकार;
    uint32_t state_seq_stride;
    uint32_t state_आकार_per_snap;

    भग्न    scale;

    काष्ठा fastdiv_मूल्यावली div_H;
    काष्ठा fastdiv_मूल्यावली div_q1;
    काष्ठा fastdiv_मूल्यावली div_k1;
    काष्ठा fastdiv_मूल्यावली div_rq3;
    काष्ठा fastdiv_मूल्यावली div_rk3;
    काष्ठा fastdiv_मूल्यावली div_n_threads;
};

#अगर defined(__cplusplus)
static_assert(माप(काष्ठा htp_gdn_kernel_params) <= 128, "htp_gdn_kernel_params is too large for kernel_params blob");
#अन्यथा
_Static_assert(माप(काष्ठा htp_gdn_kernel_params) <= 128, "htp_gdn_kernel_params is too large for kernel_params blob");
#पूर्ण_अगर

काष्ठा htp_gdn_vtcm_layout {
    माप_प्रकार state_aligned;
    माप_प्रकार bytes_per_thread;
    माप_प्रकार योग_bytes;
};

अटल अंतरभूत व्योम htp_gdn_vtcm_layout_build(
    काष्ठा htp_gdn_vtcm_layout * layout,
    uint32_t S_v,
    uint32_t n_threads
) {
    माप_प्रकार state_bytes = (माप_प्रकार) S_v * S_v * माप(भग्न);
    layout->state_aligned = hex_round_up(state_bytes, 128);
    layout->bytes_per_thread = 2 * layout->state_aligned;
    layout->योग_bytes = layout->bytes_per_thread * n_threads;
}

काष्ठा htp_gdn_hmx_vtcm_layout {
    माप_प्रकार off_s_state;
    माप_प्रकार off_s_f16;
    माप_प्रकार off_s_col_tiles;
    माप_प्रकार off_s_update_f32;
    माप_प्रकार off_s_update_tiles;

    माप_प्रकार off_q_f32[2];
    माप_प्रकार off_k_f32[2];
    माप_प्रकार off_v_f32[2];
    माप_प्रकार off_g_f32[2];
    माप_प्रकार off_b_f32[2];
    माप_प्रकार off_g_raw[2];
    माप_प्रकार off_b_raw[2];
    माप_प्रकार off_o_f32[2];

    माप_प्रकार off_v_inter_f32;
    माप_प्रकार off_o_inter_f32;
    माप_प्रकार off_o_intra_f32;
    माप_प्रकार off_k_f16;
    माप_प्रकार off_v_अभाज्य_f16;
    माप_प्रकार off_delta_f16;
    माप_प्रकार off_d_f16;

    माप_प्रकार off_q_row_tiles;
    माप_प्रकार off_q_अभाज्य_row_tiles;
    माप_प्रकार off_k_row_tiles;
    माप_प्रकार off_k_col_tiles;
    माप_प्रकार off_k_अभाज्य_row_tiles;
    माप_प्रकार off_k_col_tiles_64x128;
    माप_प्रकार off_kk_tiles;
    माप_प्रकार off_qk_tiles;
    माप_प्रकार off_v_inter_tiles;
    माप_प्रकार off_o_inter_tiles;
    माप_प्रकार off_inv_row_tiles;
    माप_प्रकार off_a_row_tiles;
    माप_प्रकार off_v_अभाज्य_col_tiles;
    माप_प्रकार off_delta_tiles;
    माप_प्रकार off_delta_col_tiles;
    माप_प्रकार off_o_intra_tiles;
    माप_प्रकार off_d_row_tiles;

    माप_प्रकार off_gamma;
    माप_प्रकार off_lambda_init;
    माप_प्रकार off_decay_m;
    माप_प्रकार off_decay_a;
    माप_प्रकार off_rows_kk;
    माप_प्रकार off_rows_qk;
    माप_प्रकार off_rows_inv;
    माप_प्रकार off_rows_a;

    माप_प्रकार off_thread_scratch;
    माप_प्रकार off_scales_1;

    माप_प्रकार state_f32_bytes;
    माप_प्रकार state_f16_bytes;
    माप_प्रकार state_tiles_bytes;
    माप_प्रकार dma_chunk_bytes;
    माप_प्रकार act_f16_bytes;
    माप_प्रकार tile_64xSv_bytes;
    माप_प्रकार tile_64x64_bytes;

    uint32_t n_heads_batch;
    uint32_t n_threads;
    bool     pipeline;
    माप_प्रकार   योग_bytes;
};

अटल अंतरभूत व्योम htp_gdn_hmx_vtcm_layout_build(
    काष्ठा htp_gdn_hmx_vtcm_layout * L,
    uint32_t S_v,
    uint32_t chunk_आकार,
    uint32_t n_heads_batch,
    uint32_t n_threads,
    bool     pipeline
) {
    स_रखो(L, 0, माप(*L));
    L->n_heads_batch = n_heads_batch;
    L->n_threads     = n_threads;
    L->pipeline      = pipeline;

    स्थिर माप_प्रकार bh = (माप_प्रकार) n_heads_batch;
    स्थिर माप_प्रकार nth = (माप_प्रकार) (n_threads > 0 ? n_threads : 1);

    स्थिर माप_प्रकार state_f32_sz   = hex_round_up(S_v * S_v * माप(भग्न), 2048);
    स्थिर माप_प्रकार state_f16_sz   = hex_round_up(S_v * S_v * माप(__fp16), 2048);
    स्थिर माप_प्रकार n_sv_tiles     = S_v / 32;
    स्थिर माप_प्रकार state_tiles_sz = n_sv_tiles * n_sv_tiles * 2048;

    स्थिर माप_प्रकार dma_chunk_sz   = hex_round_up(chunk_आकार * S_v * माप(भग्न), 2048);
    स्थिर माप_प्रकार dma_scalar_sz  = hex_round_up(chunk_आकार * माप(भग्न), 128);

    स्थिर माप_प्रकार act_f16_sz     = hex_round_up(chunk_आकार * S_v * माप(__fp16), 2048);
    स्थिर माप_प्रकार tile_64xSv_sz  = 2 * n_sv_tiles * 2048;
    स्थिर माप_प्रकार tile_64x64_sz  = 4 * 2048;

    स्थिर माप_प्रकार decay_sz       = 64 * 64 * माप(__fp16);
    स्थिर माप_प्रकार row_vecs_sz    = 64 * 128;

    L->state_f32_bytes   = state_f32_sz;
    L->state_f16_bytes   = state_f16_sz;
    L->state_tiles_bytes = state_tiles_sz;
    L->dma_chunk_bytes   = dma_chunk_sz;
    L->act_f16_bytes     = act_f16_sz;
    L->tile_64xSv_bytes  = tile_64xSv_sz;
    L->tile_64x64_bytes  = tile_64x64_sz;

    माप_प्रकार off = 0;

    VTCM_LAYOUT_ALLOC(off, off_s_state,        bh * state_f32_sz);
    VTCM_LAYOUT_ALLOC(off, off_s_f16,          bh * state_f16_sz);
    off = hex_align_up(off, HMX_FP16_TILE_आकार);
    VTCM_LAYOUT_ALLOC(off, off_s_col_tiles,    bh * state_tiles_sz);
    VTCM_LAYOUT_ALLOC(off, off_s_update_f32,   bh * state_f32_sz);
    off = hex_align_up(off, HMX_FP16_TILE_आकार);
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
    स्थिर माप_प्रकार raw_gb_sz = hex_round_up(bh * chunk_आकार * माप(भग्न), 128);
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
    VTCM_LAYOUT_ALLOC(off, off_v_अभाज्य_f16, bh * act_f16_sz);
    VTCM_LAYOUT_ALLOC(off, off_delta_f16,   bh * act_f16_sz);
    VTCM_LAYOUT_ALLOC(off, off_d_f16,       bh * act_f16_sz);

    off = hex_align_up(off, HMX_FP16_TILE_आकार);
    VTCM_LAYOUT_ALLOC(off, off_q_row_tiles,        bh * tile_64xSv_sz);
    VTCM_LAYOUT_ALLOC(off, off_q_अभाज्य_row_tiles,  bh * tile_64xSv_sz);
    VTCM_LAYOUT_ALLOC(off, off_k_row_tiles,        bh * tile_64xSv_sz);
    VTCM_LAYOUT_ALLOC(off, off_k_col_tiles,        bh * tile_64xSv_sz);
    VTCM_LAYOUT_ALLOC(off, off_k_अभाज्य_row_tiles,  bh * tile_64xSv_sz);
    VTCM_LAYOUT_ALLOC(off, off_k_col_tiles_64x128, bh * tile_64xSv_sz);
    VTCM_LAYOUT_ALLOC(off, off_kk_tiles,           bh * tile_64x64_sz);
    VTCM_LAYOUT_ALLOC(off, off_qk_tiles,           bh * tile_64x64_sz);
    VTCM_LAYOUT_ALLOC(off, off_v_inter_tiles,      bh * tile_64xSv_sz);
    VTCM_LAYOUT_ALLOC(off, off_o_inter_tiles,      bh * tile_64xSv_sz);
    VTCM_LAYOUT_ALLOC(off, off_inv_row_tiles,      bh * tile_64x64_sz);
    VTCM_LAYOUT_ALLOC(off, off_a_row_tiles,        bh * tile_64x64_sz);
    VTCM_LAYOUT_ALLOC(off, off_v_अभाज्य_col_tiles,  bh * tile_64xSv_sz);
    VTCM_LAYOUT_ALLOC(off, off_delta_tiles,        bh * tile_64xSv_sz);
    VTCM_LAYOUT_ALLOC(off, off_delta_col_tiles,    bh * tile_64xSv_sz);
    VTCM_LAYOUT_ALLOC(off, off_o_intra_tiles,      bh * tile_64xSv_sz);
    VTCM_LAYOUT_ALLOC(off, off_d_row_tiles,        bh * tile_64xSv_sz);

    VTCM_LAYOUT_ALLOC(off, off_gamma,       bh * hex_round_up(chunk_आकार * माप(भग्न), 128));
    VTCM_LAYOUT_ALLOC(off, off_lambda_init, bh * hex_round_up(chunk_आकार * माप(भग्न), 128));
    VTCM_LAYOUT_ALLOC(off, off_decay_m,     bh * decay_sz);
    VTCM_LAYOUT_ALLOC(off, off_decay_a,     bh * decay_sz);
    VTCM_LAYOUT_ALLOC(off, off_rows_kk,     bh * row_vecs_sz);
    VTCM_LAYOUT_ALLOC(off, off_rows_qk,     bh * row_vecs_sz);
    VTCM_LAYOUT_ALLOC(off, off_rows_inv,    bh * row_vecs_sz);
    VTCM_LAYOUT_ALLOC(off, off_rows_a,      bh * row_vecs_sz);

    स्थिर माप_प्रकार thread_scratch_sz = 64 * 128;
    off = hex_align_up(off, HMX_FP16_TILE_आकार);
    VTCM_LAYOUT_ALLOC(off, off_thread_scratch, nth * thread_scratch_sz);
    off = hex_align_up(off, HMX_FP16_TILE_आकार);
    VTCM_LAYOUT_ALLOC(off, off_scales_1,       HMX_FP16_TILE_आकार);

    L->योग_bytes = off;
}

अटल अंतरभूत bool htp_gdn_hmx_solve_layout(
    काष्ठा htp_gdn_hmx_vtcm_layout * layout_out,
    uint32_t S_v,
    uint32_t chunk_आकार,
    uint32_t योग_rows,
    माप_प्रकार   vtcm_budget,
    uint32_t n_threads,
    bool     pipeline,
    uint32_t * n_heads_batch_out
) {
    uint32_t max_batch = 8;
    अगर (max_batch > योग_rows) {
        max_batch = योग_rows;
    }
    अगर (max_batch > n_threads) {
        max_batch = n_threads;
    }
    अटल स्थिर uint32_t candidates[] = { 8, 6, 4, 2, 1 };
    क्रम (माप_प्रकार i = 0; i < माप(candidates) / माप(candidates[0]); ++i) {
        uint32_t bh = candidates[i];
        अगर (bh > max_batch) {
            जारी;
        }
        काष्ठा htp_gdn_hmx_vtcm_layout L;
        htp_gdn_hmx_vtcm_layout_build(&L, S_v, chunk_आकार, bh, n_threads, pipeline);
        अगर (L.योग_bytes <= vtcm_budget) {
            *layout_out = L;
            *n_heads_batch_out = bh;
            वापस true;
        }
    }
    अगर (pipeline) {
        वापस htp_gdn_hmx_solve_layout(layout_out, S_v, chunk_आकार, योग_rows, vtcm_budget, n_threads, false, n_heads_batch_out);
    }
    वापस false;
}

#पूर्ण_अगर // HTP_GATED_DELTA_NET_OPS_H
