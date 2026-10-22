#నిర్వచితం_కాకపోతే HTP_GATED_DELTA_NET_OPS_H
#నిర్వచించు HTP_GATED_DELTA_NET_OPS_H

#అన్తర్భావ <stdint.h>
#అన్తర్భావ <stddef.h>
#అన్తర్భావ <stdbool.h>

#అన్తర్భావ "hex-fastdiv.h"
#అన్తర్భావ "hex-common.h"
#అన్తర్భావ "htp-vtcm.h"

#నిర్వచించు HTP_GDN_MAX_SV     128
#నిర్వచించు HTP_GDN_CHUNK_కొలత 64
#నిర్వచించు HTP_GDN_MIN_టోకెన్లు 8

#నిర్వచితం_కాకపోతే HMX_FP16_TILE_కొలత
#నిర్వచించు HMX_FP16_TILE_కొలత 2048
#అయితే_ముగింపు

గణన htp_gdn_kernel_type {
    HTP_GDN_KERNEL_HVX_RECURRENT = 0,
    HTP_GDN_KERNEL_HMX_CHUNKED   = 1,
};

నిర్మాణం htp_gdn_kernel_params {
    uint8_t  kernel_type;
    uint8_t  pipeline;
    uint16_t chunk_కొలత;
    uint16_t n_chunks;
    uint16_t n_heads_batch;

    uint32_t n_దారాలు;
    uint32_t S_v;
    uint32_t H;
    uint32_t n_టోకెన్లు;
    uint32_t n_seqs;
    uint32_t K;

    uint32_t మొత్తం_అడ్డువరుసలు;
    uint32_t అడ్డువరుస_ప్రారంభం;
    uint32_t nrows;
    uint32_t అడ్డువరుసలు_చొప్పున_thread;

    uint32_t kda;
    uint32_t state_aligned;
    uint32_t vtcm_చొప్పున_thread;
    uint32_t vtcm_కొలత;
    uint32_t state_seq_stride;
    uint32_t state_కొలత_చొప్పున_snap;

    దశాంశం    scale;

    నిర్మాణం fastdiv_విలువలు div_H;
    నిర్మాణం fastdiv_విలువలు div_q1;
    నిర్మాణం fastdiv_విలువలు div_k1;
    నిర్మాణం fastdiv_విలువలు div_rq3;
    నిర్మాణం fastdiv_విలువలు div_rk3;
    నిర్మాణం fastdiv_విలువలు div_n_దారాలు;
};

#ఒకవేళ defined(__cplusplus)
static_assert(పరిమాణం(నిర్మాణం htp_gdn_kernel_params) <= 128, "htp_gdn_kernel_params is too large for kernel_params blob");
#లేకపోతే
_Static_assert(పరిమాణం(నిర్మాణం htp_gdn_kernel_params) <= 128, "htp_gdn_kernel_params is too large for kernel_params blob");
#అయితే_ముగింపు

నిర్మాణం htp_gdn_vtcm_layout {
    size_t state_aligned;
    size_t బైట్లు_చొప్పున_thread;
    size_t మొత్తం_బైట్లు;
};

స్థిర అంతఃవరుస శూన్య htp_gdn_vtcm_layout_build(
    నిర్మాణం htp_gdn_vtcm_layout * layout,
    uint32_t S_v,
    uint32_t n_దారాలు
) {
    size_t state_బైట్లు = (size_t) S_v * S_v * పరిమాణం(దశాంశం);
    layout->state_aligned = hex_విడత_పైకి(state_బైట్లు, 128);
    layout->బైట్లు_చొప్పున_thread = 2 * layout->state_aligned;
    layout->మొత్తం_బైట్లు = layout->బైట్లు_చొప్పున_thread * n_దారాలు;
}

నిర్మాణం htp_gdn_hmx_vtcm_layout {
    size_t off_s_state;
    size_t off_s_f16;
    size_t off_s_నిలువు_tiles;
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
    size_t off_v_ప్రధానసంఖ్య_f16;
    size_t off_delta_f16;
    size_t off_d_f16;

    size_t off_q_అడ్డువరుస_tiles;
    size_t off_q_ప్రధానసంఖ్య_అడ్డువరుస_tiles;
    size_t off_k_అడ్డువరుస_tiles;
    size_t off_k_నిలువు_tiles;
    size_t off_k_ప్రధానసంఖ్య_అడ్డువరుస_tiles;
    size_t off_k_నిలువు_tiles_64x128;
    size_t off_kk_tiles;
    size_t off_qk_tiles;
    size_t off_v_inter_tiles;
    size_t off_o_inter_tiles;
    size_t off_inv_అడ్డువరుస_tiles;
    size_t off_a_అడ్డువరుస_tiles;
    size_t off_v_ప్రధానసంఖ్య_నిలువు_tiles;
    size_t off_delta_tiles;
    size_t off_delta_నిలువు_tiles;
    size_t off_o_intra_tiles;
    size_t off_d_అడ్డువరుస_tiles;

    size_t off_gamma;
    size_t off_lambda_init;
    size_t off_decay_m;
    size_t off_decay_a;
    size_t off_అడ్డువరుసలు_kk;
    size_t off_అడ్డువరుసలు_qk;
    size_t off_అడ్డువరుసలు_inv;
    size_t off_అడ్డువరుసలు_a;

    size_t off_thread_scratch;
    size_t off_scales_1;

    size_t state_f32_బైట్లు;
    size_t state_f16_బైట్లు;
    size_t state_tiles_బైట్లు;
    size_t dma_chunk_బైట్లు;
    size_t act_f16_బైట్లు;
    size_t tile_64x_Sv_బైట్లు;
    size_t tile_64x64_బైట్లు;

    uint32_t n_heads_batch;
    uint32_t n_దారాలు;
    bool     pipeline;
    size_t   మొత్తం_బైట్లు;
};

స్థిర అంతఃవరుస శూన్య htp_gdn_hmx_vtcm_layout_build(
    నిర్మాణం htp_gdn_hmx_vtcm_layout * L,
    uint32_t S_v,
    uint32_t chunk_కొలత,
    uint32_t n_heads_batch,
    uint32_t n_దారాలు,
    bool     pipeline
) {
    memset(L, 0, పరిమాణం(*L));
    L->n_heads_batch = n_heads_batch;
    L->n_దారాలు     = n_దారాలు;
    L->pipeline      = pipeline;

    స్థిరరాశి size_t bh = (size_t) n_heads_batch;
    స్థిరరాశి size_t nth = (size_t) (n_దారాలు > 0 ? n_దారాలు : 1);

    స్థిరరాశి size_t state_f32_sz   = hex_విడత_పైకి(S_v * S_v * పరిమాణం(దశాంశం), 2048);
    స్థిరరాశి size_t state_f16_sz   = hex_విడత_పైకి(S_v * S_v * పరిమాణం(__fp16), 2048);
    స్థిరరాశి size_t n_sv_tiles     = S_v / 32;
    స్థిరరాశి size_t state_tiles_sz = n_sv_tiles * n_sv_tiles * 2048;

    స్థిరరాశి size_t dma_chunk_sz   = hex_విడత_పైకి(chunk_కొలత * S_v * పరిమాణం(దశాంశం), 2048);
    స్థిరరాశి size_t dma_scalar_sz  = hex_విడత_పైకి(chunk_కొలత * పరిమాణం(దశాంశం), 128);

    స్థిరరాశి size_t act_f16_sz     = hex_విడత_పైకి(chunk_కొలత * S_v * పరిమాణం(__fp16), 2048);
    స్థిరరాశి size_t tile_64xSv_sz  = 2 * n_sv_tiles * 2048;
    స్థిరరాశి size_t tile_64x64_sz  = 4 * 2048;

    స్థిరరాశి size_t decay_sz       = 64 * 64 * పరిమాణం(__fp16);
    స్థిరరాశి size_t అడ్డువరుస_vecs_sz    = 64 * 128;

    L->state_f32_బైట్లు   = state_f32_sz;
    L->state_f16_బైట్లు   = state_f16_sz;
    L->state_tiles_బైట్లు = state_tiles_sz;
    L->dma_chunk_బైట్లు   = dma_chunk_sz;
    L->act_f16_బైట్లు     = act_f16_sz;
    L->tile_64x_Sv_బైట్లు  = tile_64xSv_sz;
    L->tile_64x64_బైట్లు  = tile_64x64_sz;

    size_t off = 0;

    VTCM_LAYOUT_ALLOC(off, off_s_state,        bh * state_f32_sz);
    VTCM_LAYOUT_ALLOC(off, off_s_f16,          bh * state_f16_sz);
    off = hex_align_పైకి(off, HMX_FP16_TILE_కొలత);
    VTCM_LAYOUT_ALLOC(off, off_s_నిలువు_tiles,    bh * state_tiles_sz);
    VTCM_LAYOUT_ALLOC(off, off_s_update_f32,   bh * state_f32_sz);
    off = hex_align_పైకి(off, HMX_FP16_TILE_కొలత);
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
    స్థిరరాశి size_t raw_gb_sz = hex_విడత_పైకి(bh * chunk_కొలత * పరిమాణం(దశాంశం), 128);
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
    VTCM_LAYOUT_ALLOC(off, off_v_ప్రధానసంఖ్య_f16, bh * act_f16_sz);
    VTCM_LAYOUT_ALLOC(off, off_delta_f16,   bh * act_f16_sz);
    VTCM_LAYOUT_ALLOC(off, off_d_f16,       bh * act_f16_sz);

    off = hex_align_పైకి(off, HMX_FP16_TILE_కొలత);
    VTCM_LAYOUT_ALLOC(off, off_q_అడ్డువరుస_tiles,        bh * tile_64xSv_sz);
    VTCM_LAYOUT_ALLOC(off, off_q_ప్రధానసంఖ్య_అడ్డువరుస_tiles,  bh * tile_64xSv_sz);
    VTCM_LAYOUT_ALLOC(off, off_k_అడ్డువరుస_tiles,        bh * tile_64xSv_sz);
    VTCM_LAYOUT_ALLOC(off, off_k_నిలువు_tiles,        bh * tile_64xSv_sz);
    VTCM_LAYOUT_ALLOC(off, off_k_ప్రధానసంఖ్య_అడ్డువరుస_tiles,  bh * tile_64xSv_sz);
    VTCM_LAYOUT_ALLOC(off, off_k_నిలువు_tiles_64x128, bh * tile_64xSv_sz);
    VTCM_LAYOUT_ALLOC(off, off_kk_tiles,           bh * tile_64x64_sz);
    VTCM_LAYOUT_ALLOC(off, off_qk_tiles,           bh * tile_64x64_sz);
    VTCM_LAYOUT_ALLOC(off, off_v_inter_tiles,      bh * tile_64xSv_sz);
    VTCM_LAYOUT_ALLOC(off, off_o_inter_tiles,      bh * tile_64xSv_sz);
    VTCM_LAYOUT_ALLOC(off, off_inv_అడ్డువరుస_tiles,      bh * tile_64x64_sz);
    VTCM_LAYOUT_ALLOC(off, off_a_అడ్డువరుస_tiles,        bh * tile_64x64_sz);
    VTCM_LAYOUT_ALLOC(off, off_v_ప్రధానసంఖ్య_నిలువు_tiles,  bh * tile_64xSv_sz);
    VTCM_LAYOUT_ALLOC(off, off_delta_tiles,        bh * tile_64xSv_sz);
    VTCM_LAYOUT_ALLOC(off, off_delta_నిలువు_tiles,    bh * tile_64xSv_sz);
    VTCM_LAYOUT_ALLOC(off, off_o_intra_tiles,      bh * tile_64xSv_sz);
    VTCM_LAYOUT_ALLOC(off, off_d_అడ్డువరుస_tiles,        bh * tile_64xSv_sz);

    VTCM_LAYOUT_ALLOC(off, off_gamma,       bh * hex_విడత_పైకి(chunk_కొలత * పరిమాణం(దశాంశం), 128));
    VTCM_LAYOUT_ALLOC(off, off_lambda_init, bh * hex_విడత_పైకి(chunk_కొలత * పరిమాణం(దశాంశం), 128));
    VTCM_LAYOUT_ALLOC(off, off_decay_m,     bh * decay_sz);
    VTCM_LAYOUT_ALLOC(off, off_decay_a,     bh * decay_sz);
    VTCM_LAYOUT_ALLOC(off, off_అడ్డువరుసలు_kk,     bh * అడ్డువరుస_vecs_sz);
    VTCM_LAYOUT_ALLOC(off, off_అడ్డువరుసలు_qk,     bh * అడ్డువరుస_vecs_sz);
    VTCM_LAYOUT_ALLOC(off, off_అడ్డువరుసలు_inv,    bh * అడ్డువరుస_vecs_sz);
    VTCM_LAYOUT_ALLOC(off, off_అడ్డువరుసలు_a,      bh * అడ్డువరుస_vecs_sz);

    స్థిరరాశి size_t thread_scratch_sz = 64 * 128;
    off = hex_align_పైకి(off, HMX_FP16_TILE_కొలత);
    VTCM_LAYOUT_ALLOC(off, off_thread_scratch, nth * thread_scratch_sz);
    off = hex_align_పైకి(off, HMX_FP16_TILE_కొలత);
    VTCM_LAYOUT_ALLOC(off, off_scales_1,       HMX_FP16_TILE_కొలత);

    L->మొత్తం_బైట్లు = off;
}

స్థిర అంతఃవరుస bool htp_gdn_hmx_solve_layout(
    నిర్మాణం htp_gdn_hmx_vtcm_layout * layout_బహిర్గతం,
    uint32_t S_v,
    uint32_t chunk_కొలత,
    uint32_t మొత్తం_అడ్డువరుసలు,
    size_t   vtcm_budget,
    uint32_t n_దారాలు,
    bool     pipeline,
    uint32_t * n_heads_batch_బహిర్గతం
) {
    uint32_t max_batch = 8;
    ఒకవేళ (max_batch > మొత్తం_అడ్డువరుసలు) {
        max_batch = మొత్తం_అడ్డువరుసలు;
    }
    ఒకవేళ (max_batch > n_దారాలు) {
        max_batch = n_దారాలు;
    }
    స్థిర స్థిరరాశి uint32_t candidates[] = { 8, 6, 4, 2, 1 };
    కొరకు (size_t i = 0; i < పరిమాణం(candidates) / పరిమాణం(candidates[0]); ++i) {
        uint32_t bh = candidates[i];
        ఒకవేళ (bh > max_batch) {
            కొనసాగు;
        }
        నిర్మాణం htp_gdn_hmx_vtcm_layout L;
        htp_gdn_hmx_vtcm_layout_build(&L, S_v, chunk_కొలత, bh, n_దారాలు, pipeline);
        ఒకవేళ (L.మొత్తం_బైట్లు <= vtcm_budget) {
            *layout_బహిర్గతం = L;
            *n_heads_batch_బహిర్గతం = bh;
            తిరిగి true;
        }
    }
    ఒకవేళ (pipeline) {
        తిరిగి htp_gdn_hmx_solve_layout(layout_బహిర్గతం, S_v, chunk_కొలత, మొత్తం_అడ్డువరుసలు, vtcm_budget, n_దారాలు, false, n_heads_batch_బహిర్గతం);
    }
    తిరిగి false;
}

#అయితే_ముగింపు // HTP_GATED_DELTA_NET_OPS_H
