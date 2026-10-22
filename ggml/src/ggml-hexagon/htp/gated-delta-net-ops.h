#جيڪڏهن_وصف_نه_ٿيل HTP_GATED_DELTA_NET_OPS_H
#وصف_ڪريو HTP_GATED_DELTA_NET_OPS_H

#ضمن <stdint.h>
#ضمن <stddef.h>
#ضمن <stdbool.h>

#ضمن "hex-fastdiv.h"
#ضمن "hex-common.h"
#ضمن "htp-vtcm.h"

#وصف_ڪريو HTP_GDN_MAX_SV     128
#وصف_ڪريو HTP_GDN_CHUNK_سائيز 64
#وصف_ڪريو HTP_GDN_MIN_ٽوڪن 8

#جيڪڏهن_وصف_نه_ٿيل HMX_FP16_TILE_سائيز
#وصف_ڪريو HMX_FP16_TILE_سائيز 2048
#جيڪڏهن_پورو

تعداد htp_gdn_kernel_type {
    HTP_GDN_KERNEL_HVX_RECURRENT = 0,
    HTP_GDN_KERNEL_HMX_CHUNKED   = 1,
};

بنية htp_gdn_kernel_params {
    uint8_t  kernel_type;
    uint8_t  pipeline;
    uint16_t chunk_سائيز;
    uint16_t n_chunks;
    uint16_t n_heads_batch;

    uint32_t n_ڌاڳا;
    uint32_t S_v;
    uint32_t H;
    uint32_t n_ٽوڪن;
    uint32_t n_seqs;
    uint32_t K;

    uint32_t ڪل_صفون;
    uint32_t صف_شروعات;
    uint32_t nrows;
    uint32_t صفون_في_thread;

    uint32_t kda;
    uint32_t state_aligned;
    uint32_t vtcm_في_thread;
    uint32_t vtcm_سائيز;
    uint32_t state_seq_stride;
    uint32_t state_سائيز_في_snap;

    عائم    scale;

    بنية fastdiv_قدرون div_H;
    بنية fastdiv_قدرون div_q1;
    بنية fastdiv_قدرون div_k1;
    بنية fastdiv_قدرون div_rq3;
    بنية fastdiv_قدرون div_rk3;
    بنية fastdiv_قدرون div_n_ڌاڳا;
};

#إذا defined(__cplusplus)
static_assert(حجم(بنية htp_gdn_kernel_params) <= 128, "htp_gdn_kernel_params is too large for kernel_params blob");
#وإلا
_Static_assert(حجم(بنية htp_gdn_kernel_params) <= 128, "htp_gdn_kernel_params is too large for kernel_params blob");
#جيڪڏهن_پورو

بنية htp_gdn_vtcm_layout {
    size_t state_aligned;
    size_t بائيٽون_في_thread;
    size_t ڪل_بائيٽون;
};

ساكن مضمن فراغ htp_gdn_vtcm_layout_build(
    بنية htp_gdn_vtcm_layout * layout,
    uint32_t S_v,
    uint32_t n_ڌاڳا
) {
    size_t state_بائيٽون = (size_t) S_v * S_v * حجم(عائم);
    layout->state_aligned = hex_ڦيرو_مٿي(state_بائيٽون, 128);
    layout->بائيٽون_في_thread = 2 * layout->state_aligned;
    layout->ڪل_بائيٽون = layout->بائيٽون_في_thread * n_ڌاڳا;
}

بنية htp_gdn_hmx_vtcm_layout {
    size_t off_s_state;
    size_t off_s_f16;
    size_t off_s_کانو_tiles;
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
    size_t off_k_کانو_tiles;
    size_t off_k_مفرد_صف_tiles;
    size_t off_k_کانو_tiles_64x128;
    size_t off_kk_tiles;
    size_t off_qk_tiles;
    size_t off_v_inter_tiles;
    size_t off_o_inter_tiles;
    size_t off_inv_صف_tiles;
    size_t off_a_صف_tiles;
    size_t off_v_مفرد_کانو_tiles;
    size_t off_delta_tiles;
    size_t off_delta_کانو_tiles;
    size_t off_o_intra_tiles;
    size_t off_d_صف_tiles;

    size_t off_gamma;
    size_t off_lambda_init;
    size_t off_decay_m;
    size_t off_decay_a;
    size_t off_صفون_kk;
    size_t off_صفون_qk;
    size_t off_صفون_inv;
    size_t off_صفون_a;

    size_t off_thread_scratch;
    size_t off_scales_1;

    size_t state_f32_بائيٽون;
    size_t state_f16_بائيٽون;
    size_t state_tiles_بائيٽون;
    size_t dma_chunk_بائيٽون;
    size_t act_f16_بائيٽون;
    size_t tile_64x_Sv_بائيٽون;
    size_t tile_64x64_بائيٽون;

    uint32_t n_heads_batch;
    uint32_t n_ڌاڳا;
    bool     pipeline;
    size_t   ڪل_بائيٽون;
};

ساكن مضمن فراغ htp_gdn_hmx_vtcm_layout_build(
    بنية htp_gdn_hmx_vtcm_layout * L,
    uint32_t S_v,
    uint32_t chunk_سائيز,
    uint32_t n_heads_batch,
    uint32_t n_ڌاڳا,
    bool     pipeline
) {
    memset(L, 0, حجم(*L));
    L->n_heads_batch = n_heads_batch;
    L->n_ڌاڳا     = n_ڌاڳا;
    L->pipeline      = pipeline;

    ثابت size_t bh = (size_t) n_heads_batch;
    ثابت size_t nth = (size_t) (n_ڌاڳا > 0 ? n_ڌاڳا : 1);

    ثابت size_t state_f32_sz   = hex_ڦيرو_مٿي(S_v * S_v * حجم(عائم), 2048);
    ثابت size_t state_f16_sz   = hex_ڦيرو_مٿي(S_v * S_v * حجم(__fp16), 2048);
    ثابت size_t n_sv_tiles     = S_v / 32;
    ثابت size_t state_tiles_sz = n_sv_tiles * n_sv_tiles * 2048;

    ثابت size_t dma_chunk_sz   = hex_ڦيرو_مٿي(chunk_سائيز * S_v * حجم(عائم), 2048);
    ثابت size_t dma_scalar_sz  = hex_ڦيرو_مٿي(chunk_سائيز * حجم(عائم), 128);

    ثابت size_t act_f16_sz     = hex_ڦيرو_مٿي(chunk_سائيز * S_v * حجم(__fp16), 2048);
    ثابت size_t tile_64xSv_sz  = 2 * n_sv_tiles * 2048;
    ثابت size_t tile_64x64_sz  = 4 * 2048;

    ثابت size_t decay_sz       = 64 * 64 * حجم(__fp16);
    ثابت size_t صف_vecs_sz    = 64 * 128;

    L->state_f32_بائيٽون   = state_f32_sz;
    L->state_f16_بائيٽون   = state_f16_sz;
    L->state_tiles_بائيٽون = state_tiles_sz;
    L->dma_chunk_بائيٽون   = dma_chunk_sz;
    L->act_f16_بائيٽون     = act_f16_sz;
    L->tile_64x_Sv_بائيٽون  = tile_64xSv_sz;
    L->tile_64x64_بائيٽون  = tile_64x64_sz;

    size_t off = 0;

    VTCM_LAYOUT_ALLOC(off, off_s_state,        bh * state_f32_sz);
    VTCM_LAYOUT_ALLOC(off, off_s_f16,          bh * state_f16_sz);
    off = hex_align_مٿي(off, HMX_FP16_TILE_سائيز);
    VTCM_LAYOUT_ALLOC(off, off_s_کانو_tiles,    bh * state_tiles_sz);
    VTCM_LAYOUT_ALLOC(off, off_s_update_f32,   bh * state_f32_sz);
    off = hex_align_مٿي(off, HMX_FP16_TILE_سائيز);
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
    ثابت size_t raw_gb_sz = hex_ڦيرو_مٿي(bh * chunk_سائيز * حجم(عائم), 128);
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

    off = hex_align_مٿي(off, HMX_FP16_TILE_سائيز);
    VTCM_LAYOUT_ALLOC(off, off_q_صف_tiles,        bh * tile_64xSv_sz);
    VTCM_LAYOUT_ALLOC(off, off_q_مفرد_صف_tiles,  bh * tile_64xSv_sz);
    VTCM_LAYOUT_ALLOC(off, off_k_صف_tiles,        bh * tile_64xSv_sz);
    VTCM_LAYOUT_ALLOC(off, off_k_کانو_tiles,        bh * tile_64xSv_sz);
    VTCM_LAYOUT_ALLOC(off, off_k_مفرد_صف_tiles,  bh * tile_64xSv_sz);
    VTCM_LAYOUT_ALLOC(off, off_k_کانو_tiles_64x128, bh * tile_64xSv_sz);
    VTCM_LAYOUT_ALLOC(off, off_kk_tiles,           bh * tile_64x64_sz);
    VTCM_LAYOUT_ALLOC(off, off_qk_tiles,           bh * tile_64x64_sz);
    VTCM_LAYOUT_ALLOC(off, off_v_inter_tiles,      bh * tile_64xSv_sz);
    VTCM_LAYOUT_ALLOC(off, off_o_inter_tiles,      bh * tile_64xSv_sz);
    VTCM_LAYOUT_ALLOC(off, off_inv_صف_tiles,      bh * tile_64x64_sz);
    VTCM_LAYOUT_ALLOC(off, off_a_صف_tiles,        bh * tile_64x64_sz);
    VTCM_LAYOUT_ALLOC(off, off_v_مفرد_کانو_tiles,  bh * tile_64xSv_sz);
    VTCM_LAYOUT_ALLOC(off, off_delta_tiles,        bh * tile_64xSv_sz);
    VTCM_LAYOUT_ALLOC(off, off_delta_کانو_tiles,    bh * tile_64xSv_sz);
    VTCM_LAYOUT_ALLOC(off, off_o_intra_tiles,      bh * tile_64xSv_sz);
    VTCM_LAYOUT_ALLOC(off, off_d_صف_tiles,        bh * tile_64xSv_sz);

    VTCM_LAYOUT_ALLOC(off, off_gamma,       bh * hex_ڦيرو_مٿي(chunk_سائيز * حجم(عائم), 128));
    VTCM_LAYOUT_ALLOC(off, off_lambda_init, bh * hex_ڦيرو_مٿي(chunk_سائيز * حجم(عائم), 128));
    VTCM_LAYOUT_ALLOC(off, off_decay_m,     bh * decay_sz);
    VTCM_LAYOUT_ALLOC(off, off_decay_a,     bh * decay_sz);
    VTCM_LAYOUT_ALLOC(off, off_صفون_kk,     bh * صف_vecs_sz);
    VTCM_LAYOUT_ALLOC(off, off_صفون_qk,     bh * صف_vecs_sz);
    VTCM_LAYOUT_ALLOC(off, off_صفون_inv,    bh * صف_vecs_sz);
    VTCM_LAYOUT_ALLOC(off, off_صفون_a,      bh * صف_vecs_sz);

    ثابت size_t thread_scratch_sz = 64 * 128;
    off = hex_align_مٿي(off, HMX_FP16_TILE_سائيز);
    VTCM_LAYOUT_ALLOC(off, off_thread_scratch, nth * thread_scratch_sz);
    off = hex_align_مٿي(off, HMX_FP16_TILE_سائيز);
    VTCM_LAYOUT_ALLOC(off, off_scales_1,       HMX_FP16_TILE_سائيز);

    L->ڪل_بائيٽون = off;
}

ساكن مضمن bool htp_gdn_hmx_solve_layout(
    بنية htp_gdn_hmx_vtcm_layout * layout_ٻاهريون,
    uint32_t S_v,
    uint32_t chunk_سائيز,
    uint32_t ڪل_صفون,
    size_t   vtcm_budget,
    uint32_t n_ڌاڳا,
    bool     pipeline,
    uint32_t * n_heads_batch_ٻاهريون
) {
    uint32_t max_batch = 8;
    إذا (max_batch > ڪل_صفون) {
        max_batch = ڪل_صفون;
    }
    إذا (max_batch > n_ڌاڳا) {
        max_batch = n_ڌاڳا;
    }
    ساكن ثابت uint32_t candidates[] = { 8, 6, 4, 2, 1 };
    لكل (size_t i = 0; i < حجم(candidates) / حجم(candidates[0]); ++i) {
        uint32_t bh = candidates[i];
        إذا (bh > max_batch) {
            استمر;
        }
        بنية htp_gdn_hmx_vtcm_layout L;
        htp_gdn_hmx_vtcm_layout_build(&L, S_v, chunk_سائيز, bh, n_ڌاڳا, pipeline);
        إذا (L.ڪل_بائيٽون <= vtcm_budget) {
            *layout_ٻاهريون = L;
            *n_heads_batch_ٻاهريون = bh;
            أرجع true;
        }
    }
    إذا (pipeline) {
        أرجع htp_gdn_hmx_solve_layout(layout_ٻاهريون, S_v, chunk_سائيز, ڪل_صفون, vtcm_budget, n_ڌاڳا, false, n_heads_batch_ٻاهريون);
    }
    أرجع false;
}

#جيڪڏهن_پورو // HTP_GATED_DELTA_NET_OPS_H
