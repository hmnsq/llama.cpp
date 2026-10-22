#ifndef HTP_GATED_DELTA_NET_OPS_H
#define HTP_GATED_DELTA_NET_OPS_H

#अन्तर्भाव <stdint.h>
#अन्तर्भाव <stddef.h>
#अन्तर्भाव <stdbool.h>

#अन्तर्भाव "hex-fastdiv.h"
#अन्तर्भाव "hex-common.h"
#अन्तर्भाव "htp-vtcm.h"

#define HTP_GDN_MAX_SV     128
#define HTP_GDN_CHUNK_SIZE 64
#define HTP_GDN_MIN_TOKENS 8

#ifndef HMX_FP16_TILE_SIZE
#define HMX_FP16_TILE_SIZE 2048
#endif

गणना htp_gdn_kernel_type {
    HTP_GDN_KERNEL_HVX_RECURRENT = 0,
    HTP_GDN_KERNEL_HMX_CHUNKED   = 1,
};

संरचना htp_gdn_kernel_params {
    uint8_t  kernel_type;
    uint8_t  pipeline;
    uint16_t chunk_size;
    uint16_t n_chunks;
    uint16_t n_heads_batch;

    uint32_t n_threads;
    uint32_t S_v;
    uint32_t H;
    uint32_t n_tokens;
    uint32_t n_seqs;
    uint32_t K;

    uint32_t total_rows;
    uint32_t row_start;
    uint32_t nrows;
    uint32_t rows_per_thread;

    uint32_t kda;
    uint32_t state_aligned;
    uint32_t vtcm_per_thread;
    uint32_t vtcm_size;
    uint32_t state_seq_stride;
    uint32_t state_size_per_snap;

    दशांश    scale;

    संरचना fastdiv_मूल्ये div_H;
    संरचना fastdiv_मूल्ये div_q1;
    संरचना fastdiv_मूल्ये div_k1;
    संरचना fastdiv_मूल्ये div_rq3;
    संरचना fastdiv_मूल्ये div_rk3;
    संरचना fastdiv_मूल्ये div_n_threads;
};

#जर defined(__cplusplus)
static_assert(आकार(संरचना htp_gdn_kernel_params) <= 128, "htp_gdn_kernel_params is too large for kernel_params blob");
#अन्यथा
_Static_assert(आकार(संरचना htp_gdn_kernel_params) <= 128, "htp_gdn_kernel_params is too large for kernel_params blob");
#endif

संरचना htp_gdn_vtcm_layout {
    माप_प्रकार state_aligned;
    माप_प्रकार bytes_per_thread;
    माप_प्रकार total_bytes;
};

स्थिर ओळीत शून्य htp_gdn_vtcm_layout_build(
    संरचना htp_gdn_vtcm_layout * layout,
    uint32_t S_v,
    uint32_t n_threads
) {
    माप_प्रकार state_bytes = (माप_प्रकार) S_v * S_v * आकार(दशांश);
    layout->state_aligned = hex_round_up(state_bytes, 128);
    layout->bytes_per_thread = 2 * layout->state_aligned;
    layout->total_bytes = layout->bytes_per_thread * n_threads;
}

संरचना htp_gdn_hmx_vtcm_layout {
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
    माप_प्रकार off_v_मूळसंख्या_f16;
    माप_प्रकार off_delta_f16;
    माप_प्रकार off_d_f16;

    माप_प्रकार off_q_row_tiles;
    माप_प्रकार off_q_मूळसंख्या_row_tiles;
    माप_प्रकार off_k_row_tiles;
    माप_प्रकार off_k_col_tiles;
    माप_प्रकार off_k_मूळसंख्या_row_tiles;
    माप_प्रकार off_k_col_tiles_64x128;
    माप_प्रकार off_kk_tiles;
    माप_प्रकार off_qk_tiles;
    माप_प्रकार off_v_inter_tiles;
    माप_प्रकार off_o_inter_tiles;
    माप_प्रकार off_inv_row_tiles;
    माप_प्रकार off_a_row_tiles;
    माप_प्रकार off_v_मूळसंख्या_col_tiles;
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
    माप_प्रकार   total_bytes;
};

स्थिर ओळीत शून्य htp_gdn_hmx_vtcm_layout_build(
    संरचना htp_gdn_hmx_vtcm_layout * L,
    uint32_t S_v,
    uint32_t chunk_size,
    uint32_t n_heads_batch,
    uint32_t n_threads,
    bool     pipeline
) {
    स_रखो(L, 0, आकार(*L));
    L->n_heads_batch = n_heads_batch;
    L->n_threads     = n_threads;
    L->pipeline      = pipeline;

    स्थिरांक माप_प्रकार bh = (माप_प्रकार) n_heads_batch;
    स्थिरांक माप_प्रकार nth = (माप_प्रकार) (n_threads > 0 ? n_threads : 1);

    स्थिरांक माप_प्रकार state_f32_sz   = hex_round_up(S_v * S_v * आकार(दशांश), 2048);
    स्थिरांक माप_प्रकार state_f16_sz   = hex_round_up(S_v * S_v * आकार(__fp16), 2048);
    स्थिरांक माप_प्रकार n_sv_tiles     = S_v / 32;
    स्थिरांक माप_प्रकार state_tiles_sz = n_sv_tiles * n_sv_tiles * 2048;

    स्थिरांक माप_प्रकार dma_chunk_sz   = hex_round_up(chunk_size * S_v * आकार(दशांश), 2048);
    स्थिरांक माप_प्रकार dma_scalar_sz  = hex_round_up(chunk_size * आकार(दशांश), 128);

    स्थिरांक माप_प्रकार act_f16_sz     = hex_round_up(chunk_size * S_v * आकार(__fp16), 2048);
    स्थिरांक माप_प्रकार tile_64xSv_sz  = 2 * n_sv_tiles * 2048;
    स्थिरांक माप_प्रकार tile_64x64_sz  = 4 * 2048;

    स्थिरांक माप_प्रकार decay_sz       = 64 * 64 * आकार(__fp16);
    स्थिरांक माप_प्रकार row_vecs_sz    = 64 * 128;

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
    off = hex_align_up(off, HMX_FP16_TILE_SIZE);
    VTCM_LAYOUT_ALLOC(off, off_s_col_tiles,    bh * state_tiles_sz);
    VTCM_LAYOUT_ALLOC(off, off_s_update_f32,   bh * state_f32_sz);
    off = hex_align_up(off, HMX_FP16_TILE_SIZE);
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
    स्थिरांक माप_प्रकार raw_gb_sz = hex_round_up(bh * chunk_size * आकार(दशांश), 128);
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
    VTCM_LAYOUT_ALLOC(off, off_v_मूळसंख्या_f16, bh * act_f16_sz);
    VTCM_LAYOUT_ALLOC(off, off_delta_f16,   bh * act_f16_sz);
    VTCM_LAYOUT_ALLOC(off, off_d_f16,       bh * act_f16_sz);

    off = hex_align_up(off, HMX_FP16_TILE_SIZE);
    VTCM_LAYOUT_ALLOC(off, off_q_row_tiles,        bh * tile_64xSv_sz);
    VTCM_LAYOUT_ALLOC(off, off_q_मूळसंख्या_row_tiles,  bh * tile_64xSv_sz);
    VTCM_LAYOUT_ALLOC(off, off_k_row_tiles,        bh * tile_64xSv_sz);
    VTCM_LAYOUT_ALLOC(off, off_k_col_tiles,        bh * tile_64xSv_sz);
    VTCM_LAYOUT_ALLOC(off, off_k_मूळसंख्या_row_tiles,  bh * tile_64xSv_sz);
    VTCM_LAYOUT_ALLOC(off, off_k_col_tiles_64x128, bh * tile_64xSv_sz);
    VTCM_LAYOUT_ALLOC(off, off_kk_tiles,           bh * tile_64x64_sz);
    VTCM_LAYOUT_ALLOC(off, off_qk_tiles,           bh * tile_64x64_sz);
    VTCM_LAYOUT_ALLOC(off, off_v_inter_tiles,      bh * tile_64xSv_sz);
    VTCM_LAYOUT_ALLOC(off, off_o_inter_tiles,      bh * tile_64xSv_sz);
    VTCM_LAYOUT_ALLOC(off, off_inv_row_tiles,      bh * tile_64x64_sz);
    VTCM_LAYOUT_ALLOC(off, off_a_row_tiles,        bh * tile_64x64_sz);
    VTCM_LAYOUT_ALLOC(off, off_v_मूळसंख्या_col_tiles,  bh * tile_64xSv_sz);
    VTCM_LAYOUT_ALLOC(off, off_delta_tiles,        bh * tile_64xSv_sz);
    VTCM_LAYOUT_ALLOC(off, off_delta_col_tiles,    bh * tile_64xSv_sz);
    VTCM_LAYOUT_ALLOC(off, off_o_intra_tiles,      bh * tile_64xSv_sz);
    VTCM_LAYOUT_ALLOC(off, off_d_row_tiles,        bh * tile_64xSv_sz);

    VTCM_LAYOUT_ALLOC(off, off_gamma,       bh * hex_round_up(chunk_size * आकार(दशांश), 128));
    VTCM_LAYOUT_ALLOC(off, off_lambda_init, bh * hex_round_up(chunk_size * आकार(दशांश), 128));
    VTCM_LAYOUT_ALLOC(off, off_decay_m,     bh * decay_sz);
    VTCM_LAYOUT_ALLOC(off, off_decay_a,     bh * decay_sz);
    VTCM_LAYOUT_ALLOC(off, off_rows_kk,     bh * row_vecs_sz);
    VTCM_LAYOUT_ALLOC(off, off_rows_qk,     bh * row_vecs_sz);
    VTCM_LAYOUT_ALLOC(off, off_rows_inv,    bh * row_vecs_sz);
    VTCM_LAYOUT_ALLOC(off, off_rows_a,      bh * row_vecs_sz);

    स्थिरांक माप_प्रकार thread_scratch_sz = 64 * 128;
    off = hex_align_up(off, HMX_FP16_TILE_SIZE);
    VTCM_LAYOUT_ALLOC(off, off_thread_scratch, nth * thread_scratch_sz);
    off = hex_align_up(off, HMX_FP16_TILE_SIZE);
    VTCM_LAYOUT_ALLOC(off, off_scales_1,       HMX_FP16_TILE_SIZE);

    L->total_bytes = off;
}

स्थिर ओळीत bool htp_gdn_hmx_solve_layout(
    संरचना htp_gdn_hmx_vtcm_layout * layout_out,
    uint32_t S_v,
    uint32_t chunk_size,
    uint32_t total_rows,
    माप_प्रकार   vtcm_budget,
    uint32_t n_threads,
    bool     pipeline,
    uint32_t * n_heads_batch_out
) {
    uint32_t max_batch = 8;
    जर (max_batch > total_rows) {
        max_batch = total_rows;
    }
    जर (max_batch > n_threads) {
        max_batch = n_threads;
    }
    स्थिर स्थिरांक uint32_t candidates[] = { 8, 6, 4, 2, 1 };
    साठी (माप_प्रकार i = 0; i < आकार(candidates) / आकार(candidates[0]); ++i) {
        uint32_t bh = candidates[i];
        जर (bh > max_batch) {
            सुरूठेवा;
        }
        संरचना htp_gdn_hmx_vtcm_layout L;
        htp_gdn_hmx_vtcm_layout_build(&L, S_v, chunk_size, bh, n_threads, pipeline);
        जर (L.total_bytes <= vtcm_budget) {
            *layout_out = L;
            *n_heads_batch_out = bh;
            परत true;
        }
    }
    जर (pipeline) {
        परत htp_gdn_hmx_solve_layout(layout_out, S_v, chunk_size, total_rows, vtcm_budget, n_threads, false, n_heads_batch_out);
    }
    परत false;
}

#endif // HTP_GATED_DELTA_NET_OPS_H
