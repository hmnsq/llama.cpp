#اگر_غیر_تعریف_شدہ HTP_GATED_DELTA_NET_OPS_H
#تعریف_کرو HTP_GATED_DELTA_NET_OPS_H

#شامل <stdint.h>
#شامل <stddef.h>
#شامل <stdbool.h>

#شامل "hex-fastdiv.h"
#شامل "hex-common.h"
#شامل "htp-vtcm.h"

#تعریف_کرو HTP_GDN_MAX_SV     128
#تعریف_کرو HTP_GDN_CHUNK_سائز 64
#تعریف_کرو HTP_GDN_MIN_علامات 8

#اگر_غیر_تعریف_شدہ HMX_FP16_TILE_سائز
#تعریف_کرو HMX_FP16_TILE_سائز 2048
#اگر_ختم

شمار htp_gdn_kernel_type {
    HTP_GDN_KERNEL_HVX_RECURRENT = 0,
    HTP_GDN_KERNEL_HMX_CHUNKED   = 1,
};

ساخت htp_gdn_kernel_params {
    uint8_t  kernel_type;
    uint8_t  pipeline;
    uint16_t chunk_سائز;
    uint16_t n_chunks;
    uint16_t n_heads_batch;

    uint32_t n_دھاگے;
    uint32_t S_v;
    uint32_t H;
    uint32_t n_علامات;
    uint32_t n_seqs;
    uint32_t K;

    uint32_t کل_صفیں;
    uint32_t صف_آغاز;
    uint32_t nrows;
    uint32_t صفیں_فی_thread;

    uint32_t kda;
    uint32_t state_aligned;
    uint32_t vtcm_فی_thread;
    uint32_t vtcm_سائز;
    uint32_t state_seq_stride;
    uint32_t state_سائز_فی_snap;

    اعشاری    scale;

    ساخت fastdiv_اقدار div_H;
    ساخت fastdiv_اقدار div_q1;
    ساخت fastdiv_اقدار div_k1;
    ساخت fastdiv_اقدار div_rq3;
    ساخت fastdiv_اقدار div_rk3;
    ساخت fastdiv_اقدار div_n_دھاگے;
};

#اگر defined(__cplusplus)
static_assert(حجم(ساخت htp_gdn_kernel_params) <= 128, "htp_gdn_kernel_params is too large for kernel_params blob");
#ورنہ
_Static_assert(حجم(ساخت htp_gdn_kernel_params) <= 128, "htp_gdn_kernel_params is too large for kernel_params blob");
#اگر_ختم

ساخت htp_gdn_vtcm_layout {
    size_t state_aligned;
    size_t بائٹیں_فی_thread;
    size_t کل_بائٹیں;
};

ساکن اندرونی خالی htp_gdn_vtcm_layout_build(
    ساخت htp_gdn_vtcm_layout * layout,
    uint32_t S_v,
    uint32_t n_دھاگے
) {
    size_t state_بائٹیں = (size_t) S_v * S_v * حجم(اعشاری);
    layout->state_aligned = hex_باری_اوپر(state_بائٹیں, 128);
    layout->بائٹیں_فی_thread = 2 * layout->state_aligned;
    layout->کل_بائٹیں = layout->بائٹیں_فی_thread * n_دھاگے;
}

ساخت htp_gdn_hmx_vtcm_layout {
    size_t off_s_state;
    size_t off_s_f16;
    size_t off_s_ستون_tiles;
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
    size_t off_v_مفرد_f16;
    size_t off_delta_f16;
    size_t off_d_f16;

    size_t off_q_صف_tiles;
    size_t off_q_مفرد_صف_tiles;
    size_t off_k_صف_tiles;
    size_t off_k_ستون_tiles;
    size_t off_k_مفرد_صف_tiles;
    size_t off_k_ستون_tiles_64x128;
    size_t off_kk_tiles;
    size_t off_qk_tiles;
    size_t off_v_inter_tiles;
    size_t off_o_inter_tiles;
    size_t off_inv_صف_tiles;
    size_t off_a_صف_tiles;
    size_t off_v_مفرد_ستون_tiles;
    size_t off_delta_tiles;
    size_t off_delta_ستون_tiles;
    size_t off_o_intra_tiles;
    size_t off_d_صف_tiles;

    size_t off_gamma;
    size_t off_lambda_init;
    size_t off_decay_m;
    size_t off_decay_a;
    size_t off_صفیں_kk;
    size_t off_صفیں_qk;
    size_t off_صفیں_inv;
    size_t off_صفیں_a;

    size_t off_thread_scratch;
    size_t off_scales_1;

    size_t state_f32_بائٹیں;
    size_t state_f16_بائٹیں;
    size_t state_tiles_بائٹیں;
    size_t dma_chunk_بائٹیں;
    size_t act_f16_بائٹیں;
    size_t tile_64x_Sv_بائٹیں;
    size_t tile_64x64_بائٹیں;

    uint32_t n_heads_batch;
    uint32_t n_دھاگے;
    bool     pipeline;
    size_t   کل_بائٹیں;
};

ساکن اندرونی خالی htp_gdn_hmx_vtcm_layout_build(
    ساخت htp_gdn_hmx_vtcm_layout * L,
    uint32_t S_v,
    uint32_t chunk_سائز,
    uint32_t n_heads_batch,
    uint32_t n_دھاگے,
    bool     pipeline
) {
    memset(L, 0, حجم(*L));
    L->n_heads_batch = n_heads_batch;
    L->n_دھاگے     = n_دھاگے;
    L->pipeline      = pipeline;

    ثابت size_t bh = (size_t) n_heads_batch;
    ثابت size_t nth = (size_t) (n_دھاگے > 0 ? n_دھاگے : 1);

    ثابت size_t state_f32_sz   = hex_باری_اوپر(S_v * S_v * حجم(اعشاری), 2048);
    ثابت size_t state_f16_sz   = hex_باری_اوپر(S_v * S_v * حجم(__fp16), 2048);
    ثابت size_t n_sv_tiles     = S_v / 32;
    ثابت size_t state_tiles_sz = n_sv_tiles * n_sv_tiles * 2048;

    ثابت size_t dma_chunk_sz   = hex_باری_اوپر(chunk_سائز * S_v * حجم(اعشاری), 2048);
    ثابت size_t dma_scalar_sz  = hex_باری_اوپر(chunk_سائز * حجم(اعشاری), 128);

    ثابت size_t act_f16_sz     = hex_باری_اوپر(chunk_سائز * S_v * حجم(__fp16), 2048);
    ثابت size_t tile_64xSv_sz  = 2 * n_sv_tiles * 2048;
    ثابت size_t tile_64x64_sz  = 4 * 2048;

    ثابت size_t decay_sz       = 64 * 64 * حجم(__fp16);
    ثابت size_t صف_vecs_sz    = 64 * 128;

    L->state_f32_بائٹیں   = state_f32_sz;
    L->state_f16_بائٹیں   = state_f16_sz;
    L->state_tiles_بائٹیں = state_tiles_sz;
    L->dma_chunk_بائٹیں   = dma_chunk_sz;
    L->act_f16_بائٹیں     = act_f16_sz;
    L->tile_64x_Sv_بائٹیں  = tile_64xSv_sz;
    L->tile_64x64_بائٹیں  = tile_64x64_sz;

    size_t off = 0;

    VTCM_LAYOUT_ALLOC(off, off_s_state,        bh * state_f32_sz);
    VTCM_LAYOUT_ALLOC(off, off_s_f16,          bh * state_f16_sz);
    off = hex_align_اوپر(off, HMX_FP16_TILE_سائز);
    VTCM_LAYOUT_ALLOC(off, off_s_ستون_tiles,    bh * state_tiles_sz);
    VTCM_LAYOUT_ALLOC(off, off_s_update_f32,   bh * state_f32_sz);
    off = hex_align_اوپر(off, HMX_FP16_TILE_سائز);
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
    ثابت size_t raw_gb_sz = hex_باری_اوپر(bh * chunk_سائز * حجم(اعشاری), 128);
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
    VTCM_LAYOUT_ALLOC(off, off_v_مفرد_f16, bh * act_f16_sz);
    VTCM_LAYOUT_ALLOC(off, off_delta_f16,   bh * act_f16_sz);
    VTCM_LAYOUT_ALLOC(off, off_d_f16,       bh * act_f16_sz);

    off = hex_align_اوپر(off, HMX_FP16_TILE_سائز);
    VTCM_LAYOUT_ALLOC(off, off_q_صف_tiles,        bh * tile_64xSv_sz);
    VTCM_LAYOUT_ALLOC(off, off_q_مفرد_صف_tiles,  bh * tile_64xSv_sz);
    VTCM_LAYOUT_ALLOC(off, off_k_صف_tiles,        bh * tile_64xSv_sz);
    VTCM_LAYOUT_ALLOC(off, off_k_ستون_tiles,        bh * tile_64xSv_sz);
    VTCM_LAYOUT_ALLOC(off, off_k_مفرد_صف_tiles,  bh * tile_64xSv_sz);
    VTCM_LAYOUT_ALLOC(off, off_k_ستون_tiles_64x128, bh * tile_64xSv_sz);
    VTCM_LAYOUT_ALLOC(off, off_kk_tiles,           bh * tile_64x64_sz);
    VTCM_LAYOUT_ALLOC(off, off_qk_tiles,           bh * tile_64x64_sz);
    VTCM_LAYOUT_ALLOC(off, off_v_inter_tiles,      bh * tile_64xSv_sz);
    VTCM_LAYOUT_ALLOC(off, off_o_inter_tiles,      bh * tile_64xSv_sz);
    VTCM_LAYOUT_ALLOC(off, off_inv_صف_tiles,      bh * tile_64x64_sz);
    VTCM_LAYOUT_ALLOC(off, off_a_صف_tiles,        bh * tile_64x64_sz);
    VTCM_LAYOUT_ALLOC(off, off_v_مفرد_ستون_tiles,  bh * tile_64xSv_sz);
    VTCM_LAYOUT_ALLOC(off, off_delta_tiles,        bh * tile_64xSv_sz);
    VTCM_LAYOUT_ALLOC(off, off_delta_ستون_tiles,    bh * tile_64xSv_sz);
    VTCM_LAYOUT_ALLOC(off, off_o_intra_tiles,      bh * tile_64xSv_sz);
    VTCM_LAYOUT_ALLOC(off, off_d_صف_tiles,        bh * tile_64xSv_sz);

    VTCM_LAYOUT_ALLOC(off, off_gamma,       bh * hex_باری_اوپر(chunk_سائز * حجم(اعشاری), 128));
    VTCM_LAYOUT_ALLOC(off, off_lambda_init, bh * hex_باری_اوپر(chunk_سائز * حجم(اعشاری), 128));
    VTCM_LAYOUT_ALLOC(off, off_decay_m,     bh * decay_sz);
    VTCM_LAYOUT_ALLOC(off, off_decay_a,     bh * decay_sz);
    VTCM_LAYOUT_ALLOC(off, off_صفیں_kk,     bh * صف_vecs_sz);
    VTCM_LAYOUT_ALLOC(off, off_صفیں_qk,     bh * صف_vecs_sz);
    VTCM_LAYOUT_ALLOC(off, off_صفیں_inv,    bh * صف_vecs_sz);
    VTCM_LAYOUT_ALLOC(off, off_صفیں_a,      bh * صف_vecs_sz);

    ثابت size_t thread_scratch_sz = 64 * 128;
    off = hex_align_اوپر(off, HMX_FP16_TILE_سائز);
    VTCM_LAYOUT_ALLOC(off, off_thread_scratch, nth * thread_scratch_sz);
    off = hex_align_اوپر(off, HMX_FP16_TILE_سائز);
    VTCM_LAYOUT_ALLOC(off, off_scales_1,       HMX_FP16_TILE_سائز);

    L->کل_بائٹیں = off;
}

ساکن اندرونی bool htp_gdn_hmx_solve_layout(
    ساخت htp_gdn_hmx_vtcm_layout * layout_برآمد,
    uint32_t S_v,
    uint32_t chunk_سائز,
    uint32_t کل_صفیں,
    size_t   vtcm_budget,
    uint32_t n_دھاگے,
    bool     pipeline,
    uint32_t * n_heads_batch_برآمد
) {
    uint32_t max_batch = 8;
    اگر (max_batch > کل_صفیں) {
        max_batch = کل_صفیں;
    }
    اگر (max_batch > n_دھاگے) {
        max_batch = n_دھاگے;
    }
    ساکن ثابت uint32_t candidates[] = { 8, 6, 4, 2, 1 };
    برائے (size_t i = 0; i < حجم(candidates) / حجم(candidates[0]); ++i) {
        uint32_t bh = candidates[i];
        اگر (bh > max_batch) {
            جاری;
        }
        ساخت htp_gdn_hmx_vtcm_layout L;
        htp_gdn_hmx_vtcm_layout_build(&L, S_v, chunk_سائز, bh, n_دھاگے, pipeline);
        اگر (L.کل_بائٹیں <= vtcm_budget) {
            *layout_برآمد = L;
            *n_heads_batch_برآمد = bh;
            واپس true;
        }
    }
    اگر (pipeline) {
        واپس htp_gdn_hmx_solve_layout(layout_برآمد, S_v, chunk_سائز, کل_صفیں, vtcm_budget, n_دھاگے, false, n_heads_batch_برآمد);
    }
    واپس false;
}

#اگر_ختم // HTP_GATED_DELTA_NET_OPS_H
