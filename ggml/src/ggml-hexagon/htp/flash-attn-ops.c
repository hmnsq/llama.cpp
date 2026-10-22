#సూచన clang diagnostic ignored "-Wunused-variable"
#సూచన clang diagnostic ignored "-Wunused-function"
#సూచన clang diagnostic ignored "-Wunused-but-set-variable"

#అన్తర్భావ <assert.h>
#అన్తర్భావ <HAP_compute_res.h>
#అన్తర్భావ <HAP_farf.h>
#అన్తర్భావ <math.h>
#అన్తర్భావ <stdbool.h>
#అన్తర్భావ <stdatomic.h>
#అన్తర్భావ <stddef.h>
#అన్తర్భావ <stdint.h>
#అన్తర్భావ <string.h>

#అన్తర్భావ "dma-queue.h"
#అన్తర్భావ "hex-fastdiv.h"
#అన్తర్భావ "hex-profile.h"
#అన్తర్భావ "hmx-queue.h"
#అన్తర్భావ "hmx-utils.h"
#అన్తర్భావ "hvx-utils.h"
#అన్తర్భావ "hvx-dump.h"
#అన్తర్భావ "hvx-copy.h"
#అన్తర్భావ "hvx-reduce.h"
#అన్తర్భావ "hvx-flash-attn.h"
#అన్తర్భావ "htp-vtcm.h"
#అన్తర్భావ "work-queue.h"

#నిర్వచించు GGML_COMMON_DECL_C
#అన్తర్భావ "ggml-common.h"
#అన్తర్భావ "htp-ctx.h"
#అన్తర్భావ "htp-ops.h"
#అన్తర్భావ "htp-tensor.h"
#అన్తర్భావ "hvx-quant.h"

#అన్తర్భావ "flash-attn-ops.h"
#అన్తర్భావ "hvx-fa-kernels.h"
#అన్తర్భావ "hmx-fa-kernels.h"

// Must be multiple of 32
#నిర్వచించు FLASH_ATTN_ఖండం_కొలత (32 * 2)

నిర్మాణం htp_fa_context {
    స్థిరరాశి నిర్మాణం htp_ops_context * octx;

    నిర్మాణం fastdiv_విలువలు src0_div21;
    నిర్మాణం fastdiv_విలువలు src0_div1;

    నిర్మాణం fastdiv_విలువలు ప్రసారం_rk2;
    నిర్మాణం fastdiv_విలువలు ప్రసారం_rk3;
    నిర్మాణం fastdiv_విలువలు ప్రసారం_rv2;
    నిర్మాణం fastdiv_విలువలు ప్రసారం_rv3;

    నిర్మాణం fastdiv_విలువలు src3_div2;
    నిర్మాణం fastdiv_విలువలు src3_div3;

    దశాంశం scale;
    దశాంశం max_bias;
    bool  has_softcap;
    __fp16 logit_softcap;

    uint32_t n_తల_log2;
    దశాంశం m0;
    దశాంశం m1;
    __fp16 slopes[512];

    uint32_t n_ఖండాలు;
    uint32_t n_చురుకైన_ఖండాలు;
    bool     shared_చురుకైన_ఖండాలు;

    size_t కొలత_q_అడ్డువరుస_padded;
    size_t కొలత_k_అడ్డువరుస_padded;
    size_t కొలత_v_అడ్డువరుస_padded;

    size_t కొలత_k_ఖండం;
    size_t కొలత_v_ఖండం;
    size_t కొలత_m_ఖండం;

    uint32_t qrows;
    uint32_t qrows_చొప్పున_thread;
    uint32_t qrow_ప్రారంభం;

    bool is_q_fp32;

    size_t కొలత_q_ఖండం;
    size_t కొలత_vkq_acc;

    uint8_t *  spad_q;
    uint8_t *  spad_k;
    uint8_t *  spad_v;
    uint8_t *  spad_m;
    uint8_t *  spad_a;
    దశాంశం *    spad_sinks;
    uint16_t   చురుకైన_ఖండాలు[FA_MAX_KV_ఖండాలు];

    స్థిరరాశి నిర్మాణం htp_tensor * k;
    స్థిరరాశి నిర్మాణం htp_tensor * v;
};

నిర్మాణం hmx_fa_context {
    స్థిరరాశి నిర్మాణం htp_ops_context * octx;
    స్థిరరాశి నిర్మాణం htp_tensor *      sinks;  // attention sinks (src[4]), NULL if absent
    స్థిరరాశి నిర్మాణం htp_tensor *      k;
    స్థిరరాశి నిర్మాణం htp_tensor *      v;
    bool         pipeline;  // true when n_kv_blocks >= FA_MIN_KV_BLOCKS && n_threads >= 2
    uint32_t     n_దారాలు;

    // Op parameters
    __fp16       scale;
    దశాంశం        max_bias;
    bool         has_softcap;
    __fp16       logit_softcap;
    uint32_t     n_తల_log2;
    దశాంశం        m0, m1;

    // Dimensions
    uint32_t     DK, DV;
    uint32_t     DK_pad, DV_pad;  // head_dim rounded up to 64 for HMX tiling
    uint32_t     n_kv;        // kv_len
    uint32_t     n_kv_heads;  // number of KV heads
    uint32_t     n_heads;     // number of Q heads
    uint32_t     G;           // GQA factor = n_heads / n_kv_heads
    నిర్మాణం fastdiv_విలువలు div_G;
    నిర్మాణం fastdiv_విలువలు src3_div2;
    నిర్మాణం fastdiv_విలువలు src3_div3;
    uint32_t     n_kv_ఖండాలు;
    uint32_t     neq1;        // Q token count

    // Types
    bool         is_q_fp32;
    bool         is_dst_fp32;

    // Dynamic block sizes
    uint32_t     Br;    // Q tokens per block (before GQA expansion)
    uint32_t     Bc;
    uint32_t     g_br;  // hex_align_up(G * Br, 32) - actual tile row dim

    // VTCM buffers (allocated by vtcm_seq_alloc)
    __fp16 *     vtcm_q_dma;           // Q DMA fetch buffer
    __fp16 *     vtcm_q_tiles;         // Q tile format [g_br, D]
    __fp16 *     vtcm_o_tiles[2];      // O ping-pong [g_br, D]
    __fp16 *     vtcm_k_fp16[2];       // K DMA double-buffer [Bc, D]
    __fp16 *     vtcm_v_fp16[2];       // V DMA double-buffer [Bc, D]
    __fp16 *     vtcm_k_tiles[2];      // K tiles (transposed, double-buffered)
    __fp16 *     vtcm_v_tiles[2];      // V tiles (column-major, double-buffered)
    __fp16 *     vtcm_s_tiles[2];      // S = QK^T [g_br, Bc] (double-buffered)
    __fp16 *     vtcm_p_tiles[2];      // P = softmax(S) [g_br, Bc]
    __fp16 *     vtcm_d_tiles[2];      // Diagonal rescale, g_br/32 packed diagonal tiles (double-buffered)
    __fp16 *     vtcm_d_inv_l;         // Diagonal rescale (1/l), same packed layout
    HVX_సదిశ * vtcm_m_vec;           // Row max [g_br]
    HVX_సదిశ * vtcm_l_vec;           // Row sum [g_br]
    HVX_సదిశ * vtcm_s_rowmax;        // Softmax intermediate [g_br]
    HVX_సదిశ * vtcm_p_rowsum;        // Softmax intermediate [g_br]
    HVX_సదిశ * vtcm_అడ్డువరుస_bufs;        // Per-thread softmax row scratch [n_threads][2][Bc/64]
    uint8_t *    vtcm_hmx_scales_id;   // HMX output scales (identity)
    uint8_t *    vtcm_hmx_scales_qk;   // HMX output scales (qk_scale)
    __fp16 *     vtcm_ముసుగు_buf;        // VTCM mask buffer [Br * m_line], DMA'd per KV block
    __fp16 *     vtcm_slopes;          // ALiBi slopes [g_br]
    దశాంశం *      vtcm_sinks;           // Attention sinks
    size_t       అడ్డువరుస_buf_stride;       // HVX vectors per row buffer (Bc/64)
    size_t       ముసుగు_buf_అడ్డువరుస_stride;  // elements (__fp16) per row in mask buffer
    size_t       q_tile_బైట్లు;
    size_t       k_tile_బైట్లు;
    size_t       o_tile_బైట్లు;
    size_t       నిలువు_vec_బైట్లు;
    size_t       d_tile_బైట్లు;
    bool         ముసుగు_ప్రసారం;       // true when mask->ne[2] == 1 (head-independent, single 2D DMA)
    dma_cache_fa m_cache;
};

// Scan mask rows [q_start, q_start + n_rows_q) and build compact list of KV blocks with any entry != -inf.
స్థిర uint32_t fa_పరిశీలన_చురుకైన_ఖండాలు(
        dma_వరుస *               dma_q,
        స్థిరరాశి నిర్మాణం htp_tensor * ముసుగు,
        uint32_t                  q_ప్రారంభం,
        uint32_t                  n_అడ్డువరుసలు_q,
        uint32_t                  im2,
        uint32_t                  im3,
        uint32_t                  nek1,
        uint32_t                  Bc,
        uint32_t                  n_kv_ఖండాలు,
        శూన్య *                    vtcm_scratch,
        size_t                    scratch_బైట్లు,
        uint16_t *                చురుకైన_ఖండాలు) {
    స్థిరరాశి size_t అడ్డువరుస_బైట్లు  = (size_t) nek1 * పరిమాణం(__fp16);
    స్థిరరాశి size_t అడ్డువరుస_stride = hex_విడత_పైకి(అడ్డువరుస_బైట్లు, 128);

    ఒకవేళ (!ముసుగు || n_kv_ఖండాలు <= 1 || n_kv_ఖండాలు > FA_MAX_KV_ఖండాలు ||
        (size_t) n_అడ్డువరుసలు_q * అడ్డువరుస_stride > scratch_బైట్లు) {
        #సూచన clang loop vectorize(disable)
        కొరకు (uint32_t b = 0; b < n_kv_ఖండాలు; ++b) {
            చురుకైన_ఖండాలు[b] = (uint16_t) b;
        }
        తిరిగి n_kv_ఖండాలు;
    }

    స్థిరరాశి dma_addr_t m_src = ముసుగు->దత్తాంశం + q_ప్రారంభం * ముసుగు->nb[1] + im2 * ముసుగు->nb[2] + im3 * ముసుగు->nb[3];
    dma_వరుస_నెట్టు(dma_q, dma_make_దత్తాంశం(vtcm_scratch, m_src), అడ్డువరుస_stride, ముసుగు->nb[1], అడ్డువరుస_బైట్లు, n_అడ్డువరుసలు_q);
    స్థిరరాశి uint8_t * m_buf = (స్థిరరాశి uint8_t *) dma_వరుస_తీయి(dma_q).dst;

    స్థిరరాశి HVX_సదిశ v_ninf   = Q6_Vh_vsplat_R(0xFC00);
    uint32_t         n_చురుకైన = 0;

    కొరకు (uint32_t b = 0; b < n_kv_ఖండాలు; ++b) {
        స్థిరరాశి uint32_t col0 = b * Bc;
        స్థిరరాశి uint32_t నిలువుల_సంఖ్య_2 = MIN(Bc, nek1 - col0);

        కొరకు (uint32_t r = 0; r < n_అడ్డువరుసలు_q; ++r) {
            స్థిరరాశి __fp16 * అడ్డువరుస_ptr = (స్థిరరాశి __fp16 *) (m_buf + r * అడ్డువరుస_stride) + col0;
            HVX_సదిశ v_any = Q6_V_vzero();
            uint32_t c = 0;
            కొరకు (; c + 64 <= నిలువుల_సంఖ్య_2; c += 64) {
                స్థిరరాశి HVX_సదిశ v_m = *(స్థిరరాశి HVX_సదిశ *) (అడ్డువరుస_ptr + c);
                v_any = Q6_V_vor_VV(v_any, Q6_V_vxor_VV(v_m, v_ninf));
            }
            ఒకవేళ (c < నిలువుల_సంఖ్య_2) {
                స్థిరరాశి uint32_t rem = నిలువుల_సంఖ్య_2 - c;
                స్థిరరాశి HVX_సదిశ_Pred q_rem = Q6_Q_vsetq2_R(rem * పరిమాణం(__fp16));
                స్థిరరాశి HVX_సదిశ v_m = *(స్థిరరాశి HVX_సదిశ *) (అడ్డువరుస_ptr + c);
                స్థిరరాశి HVX_సదిశ v_diff = Q6_V_vand_QV(q_rem, Q6_V_vxor_VV(v_m, v_ninf));
                v_any = Q6_V_vor_VV(v_any, v_diff);
            }
            v_any = Q6_V_vor_VV(v_any, Q6_V_vror_VR(v_any, 64));
            v_any = Q6_V_vor_VV(v_any, Q6_V_vror_VR(v_any, 32));
            v_any = Q6_V_vor_VV(v_any, Q6_V_vror_VR(v_any, 16));
            v_any = Q6_V_vor_VV(v_any, Q6_V_vror_VR(v_any, 8));
            v_any = Q6_V_vor_VV(v_any, Q6_V_vror_VR(v_any, 4));
            ఒకవేళ (Q6_R_vextract_VR(v_any, 0) != 0) {
                చురుకైన_ఖండాలు[n_చురుకైన++] = (uint16_t) b;
                ఆపు;
            }
        }
    }

    ఒకవేళ (n_చురుకైన == 0 && n_kv_ఖండాలు > 0) {
        చురుకైన_ఖండాలు[0] = 0;
        n_చురుకైన = 1;
    }
    తిరిగి n_చురుకైన;
}

స్థిర శూన్య flash_attn_ext_f16_thread(గుర్తులేని పూర్ణాంకం nth, గుర్తులేని పూర్ణాంకం ith, శూన్య * దత్తాంశం) {
    నిర్మాణం htp_fa_context * factx = (నిర్మాణం htp_fa_context *) దత్తాంశం;
    స్థిరరాశి నిర్మాణం htp_ops_context * octx = factx->octx;
    స్థిరరాశి నిర్మాణం htp_tensor * q     = octx->src[0];
    స్థిరరాశి నిర్మాణం htp_tensor * k     = octx->src[1];
    స్థిరరాశి నిర్మాణం htp_tensor * v     = octx->src[2];
    స్థిరరాశి నిర్మాణం htp_tensor * ముసుగు  = octx->src[3];
    స్థిరరాశి నిర్మాణం htp_tensor * sinks = octx->src[4];
    స్థిరరాశి నిర్మాణం htp_tensor * dst   = octx->dst;

    స్థిరరాశి uint32_t neq0 = q->ne[0];
    స్థిరరాశి uint32_t neq1 = q->ne[1];
    స్థిరరాశి uint32_t neq2 = q->ne[2];
    స్థిరరాశి uint32_t neq3 = q->ne[3];

    స్థిరరాశి uint32_t nek0 = k->ne[0];
    స్థిరరాశి uint32_t nek1 = k->ne[1];
    స్థిరరాశి uint32_t nek2 = k->ne[2];
    స్థిరరాశి uint32_t nek3 = k->ne[3];

    స్థిరరాశి uint32_t nev0 = v->ne[0];
    స్థిరరాశి uint32_t nev1 = v->ne[1];
    స్థిరరాశి uint32_t nev2 = v->ne[2];
    స్థిరరాశి uint32_t nev3 = v->ne[3];

    స్థిరరాశి uint32_t nbq1 = q->nb[1];
    స్థిరరాశి uint32_t nbq2 = q->nb[2];
    స్థిరరాశి uint32_t nbq3 = q->nb[3];

    స్థిరరాశి uint32_t nbk1 = k->nb[1];
    స్థిరరాశి uint32_t nbk2 = k->nb[2];
    స్థిరరాశి uint32_t nbk3 = k->nb[3];

    స్థిరరాశి uint32_t nbv1 = v->nb[1];
    స్థిరరాశి uint32_t nbv2 = v->nb[2];
    స్థిరరాశి uint32_t nbv3 = v->nb[3];

    స్థిరరాశి uint32_t ne1 = dst->ne[1];
    స్థిరరాశి uint32_t ne2 = dst->ne[2];
    స్థిరరాశి uint32_t ne3 = dst->ne[3];

    స్థిరరాశి uint32_t nb1 = dst->nb[1];
    స్థిరరాశి uint32_t nb2 = dst->nb[2];
    స్థిరరాశి uint32_t nb3 = dst->nb[3];

    // total rows in q
    స్థిరరాశి uint32_t dr  = factx->qrows_చొప్పున_thread;
    స్థిరరాశి uint32_t ir0 = factx->qrow_ప్రారంభం + dr * ith;
    స్థిరరాశి uint32_t ir1 = MIN(ir0 + dr, factx->qrow_ప్రారంభం + factx->qrows);

    ఒకవేళ (ir0 >= ir1) తిరిగి;

    నిర్మాణం htp_thread_trace * tr = &octx->ctx->trace[ith];

    dma_వరుస * dma_q = octx->ctx->dma[ith];

    స్థిరరాశి uint32_t DK = nek0;
    స్థిరరాశి uint32_t DV = nev0;

    స్థిరరాశి size_t కొలత_q_అడ్డువరుస = DK * ((q->type == HTP_TYPE_F32) ? 4 : 2);
    స్థిరరాశి size_t కొలత_k_అడ్డువరుస = htp_tensor_get_అడ్డువరుస_కొలత(k->type, DK);
    స్థిరరాశి size_t కొలత_v_అడ్డువరుస = htp_tensor_get_అడ్డువరుస_కొలత(v->type, DV);

    // Scratchpad buffers for Q, K, V, Mask, and VKQ32 accumulator
    uint8_t * spad_q = factx->spad_q + factx->కొలత_q_ఖండం * ith;
    uint8_t * spad_k = factx->spad_k + factx->కొలత_k_ఖండం * 2 * ith;
    uint8_t * spad_v = factx->spad_v + factx->కొలత_v_ఖండం * 2 * ith;
    uint8_t * spad_m = factx->spad_m + (ముసుగు ? factx->కొలత_m_ఖండం * HVX_FA_DMA_CACHE_కొలత : 0) * ith;
    uint8_t * spad_a = factx->spad_a + factx->కొలత_vkq_acc * ith;

    dma_cache_dm m_cache;
    dma_cache_dm_init(&m_cache, spad_m, factx->కొలత_m_ఖండం, HVX_FA_DMA_CACHE_కొలత);

    uint16_t thread_చురుకైన_ఖండాలు[FA_MAX_KV_ఖండాలు];
    స్థిరరాశి uint16_t * చురుకైన_ఖండాలు = factx->shared_చురుకైన_ఖండాలు ? factx->చురుకైన_ఖండాలు : thread_చురుకైన_ఖండాలు;
    uint32_t n_చురుకైన_ఖండాలు       = factx->n_చురుకైన_ఖండాలు;
    dma_addr_t prev_mp_పునాది        = 0;

    ఒకవేళ (!factx->shared_చురుకైన_ఖండాలు && !ముసుగు) {
        #సూచన clang loop vectorize(disable)
        కొరకు (uint32_t b = 0; b < factx->n_ఖండాలు; ++b) {
            thread_చురుకైన_ఖండాలు[b] = (uint16_t) b;
        }
    }

    స్థిరరాశి size_t కొలత_vkq_acc_single = hex_విడత_పైకి(DV * పరిమాణం(దశాంశం), 128);

    uint32_t ir = ir0;
    అయితే (ir < ir1) {
        స్థిరరాశి uint32_t iq3 = fastdiv(ir, &factx->src0_div21);
        స్థిరరాశి uint32_t iq2 = fastdiv(ir - iq3*neq2*neq1, &factx->src0_div1);
        స్థిరరాశి uint32_t iq1 = (ir - iq3*neq2*neq1 - iq2 * neq1);

        స్థిరరాశి uint32_t ik3 = fastdiv(iq3, &factx->ప్రసారం_rk3);
        స్థిరరాశి uint32_t ik2 = fastdiv(iq2, &factx->ప్రసారం_rk2);

        స్థిరరాశి uint32_t iv3 = fastdiv(iq3, &factx->ప్రసారం_rv3);
        స్థిరరాశి uint32_t iv2 = fastdiv(iq2, &factx->ప్రసారం_rv2);

        uint32_t G_local = 1;
        ఒకవేళ (neq1 == 1 && (ముసుగు == ఖాళీ || ముసుగు->ne[2] == 1)) {
            అయితే (ir + G_local < ir1 && G_local < FA_HVX_G_MAX) {
                స్థిరరాశి uint32_t తదుపరి_ir = ir + G_local;
                స్థిరరాశి uint32_t తదుపరి_iq3 = fastdiv(తదుపరి_ir, &factx->src0_div21);
                స్థిరరాశి uint32_t తదుపరి_iq2 = fastdiv(తదుపరి_ir - తదుపరి_iq3*neq2*neq1, &factx->src0_div1);
                స్థిరరాశి uint32_t తదుపరి_iq1 = (తదుపరి_ir - తదుపరి_iq3*neq2*neq1 - తదుపరి_iq2 * neq1);

                స్థిరరాశి uint32_t తదుపరి_ik3 = fastdiv(తదుపరి_iq3, &factx->ప్రసారం_rk3);
                స్థిరరాశి uint32_t తదుపరి_ik2 = fastdiv(తదుపరి_iq2, &factx->ప్రసారం_rk2);

                స్థిరరాశి uint32_t తదుపరి_iv3 = fastdiv(తదుపరి_iq3, &factx->ప్రసారం_rv3);
                స్థిరరాశి uint32_t తదుపరి_iv2 = fastdiv(తదుపరి_iq2, &factx->ప్రసారం_rv2);

                ఒకవేళ (తదుపరి_ik2 != ik2 || తదుపరి_ik3 != ik3 || తదుపరి_iv2 != iv2 || తదుపరి_iv3 != iv3 || తదుపరి_iq1 != iq1 || తదుపరి_iq3 != iq3) {
                    ఆపు;
                }
                G_local++;
            }
        }

        dma_addr_t mp_పునాది = 0;
        ఒకవేళ (ముసుగు) {
            స్థిరరాశి uint32_t im2 = fastmodulo(iq2, ముసుగు->ne[2], &factx->src3_div2);
            స్థిరరాశి uint32_t im3 = fastmodulo(iq3, ముసుగు->ne[3], &factx->src3_div3);
            mp_పునాది = ముసుగు->దత్తాంశం + iq1*ముసుగు->nb[1] + im2*ముసుగు->nb[2] + im3*ముసుగు->nb[3];
            ఒకవేళ (!factx->shared_చురుకైన_ఖండాలు && mp_పునాది != prev_mp_పునాది) {
                n_చురుకైన_ఖండాలు = fa_పరిశీలన_చురుకైన_ఖండాలు(
                    dma_q, ముసుగు, iq1, 1, im2, im3, nek1, FLASH_ATTN_ఖండం_కొలత,
                    factx->n_ఖండాలు, spad_k, factx->కొలత_k_ఖండం * 2, thread_చురుకైన_ఖండాలు);
                prev_mp_పునాది = mp_పునాది;
            }
        }

        uint32_t heads[FA_HVX_G_MAX];
        HVX_సదిశ వాలు_vecs[FA_HVX_G_MAX] __attribute__((aligned(128)));
        HVX_సదిశ S_vec[FA_HVX_G_MAX]      __attribute__((aligned(128)));
        HVX_సదిశ M_vec[FA_HVX_G_MAX]      __attribute__((aligned(128)));
        uint8_t * q_ptrs[FA_HVX_G_MAX];
        దశాంశం * vkq_ptrs[FA_HVX_G_MAX];

        కొరకు (uint32_t g = 0; g < G_local; ++g) {
            స్థిరరాశి uint32_t r = ir + g;
            స్థిరరాశి uint32_t r_iq3 = fastdiv(r, &factx->src0_div21);
            స్థిరరాశి uint32_t r_iq2 = fastdiv(r - r_iq3*neq2*neq1, &factx->src0_div1);
            స్థిరరాశి uint32_t r_iq1 = (r - r_iq3*neq2*neq1 - r_iq2 * neq1);

            heads[g] = r_iq2;
            స్థిరరాశి __fp16 వాలు = factx->slopes[r_iq2];
            వాలు_vecs[g] = hvx_vec_splat_f16(వాలు);

            S_vec[g] = hvx_vec_splat_f32(0.0f);
            M_vec[g] = hvx_vec_splat_f32(HTP_FA_M_INITIAL_VAL);

            uint8_t * q_dst = spad_q + g * factx->కొలత_q_అడ్డువరుస_padded;
            q_ptrs[g] = q_dst;

            దశాంశం * vkq_dst = (దశాంశం *)(spad_a + g * కొలత_vkq_acc_single);
            vkq_ptrs[g] = vkq_dst;
            hvx_splat_f32_a((uint8_t *) vkq_dst, 0, DV);

            // Fetch Q row g
            స్థిరరాశి dma_addr_t q_అడ్డువరుస_ptr = q->దత్తాంశం + r_iq1*nbq1 + r_iq2*nbq2 + r_iq3*nbq3;
            dma_వరుస_నెట్టు(dma_q, dma_make_దత్తాంశం(q_dst, q_అడ్డువరుస_ptr), factx->కొలత_q_అడ్డువరుస_padded, nbq1, కొలత_q_అడ్డువరుస, 1);
        }

        // Prefetch first two active blocks
        కొరకు (uint32_t అడుగు = 0; అడుగు < MIN(n_చురుకైన_ఖండాలు, 2); ++అడుగు) {
            స్థిరరాశి uint32_t ib = చురుకైన_ఖండాలు[అడుగు];
            స్థిరరాశి uint32_t ic_ప్రారంభం = ib * FLASH_ATTN_ఖండం_కొలత;
            స్థిరరాశి uint32_t current_ఖండం_కొలత = MIN(FLASH_ATTN_ఖండం_కొలత, nek1 - ic_ప్రారంభం);

            // K
            స్థిరరాశి dma_addr_t k_src = k->దత్తాంశం + ic_ప్రారంభం*nbk1 + ik2*nbk2 + ik3*nbk3;
            uint8_t * k_dst = spad_k + (అడుగు % 2) * factx->కొలత_k_ఖండం;
            dma_వరుస_నెట్టు(dma_q, dma_make_దత్తాంశం(k_dst, k_src), factx->కొలత_k_అడ్డువరుస_padded, nbk1, కొలత_k_అడ్డువరుస, current_ఖండం_కొలత);

            // V
            స్థిరరాశి dma_addr_t v_src = v->దత్తాంశం + ic_ప్రారంభం*nbv1 + iv2*nbv2 + iv3*nbv3;
            uint8_t * v_dst = spad_v + (అడుగు % 2) * factx->కొలత_v_ఖండం;
            dma_వరుస_నెట్టు(dma_q, dma_make_దత్తాంశం(v_dst, v_src), factx->కొలత_v_అడ్డువరుస_padded, nbv1, కొలత_v_అడ్డువరుస, current_ఖండం_కొలత);

            // Mask
            ఒకవేళ (ముసుగు) {
                స్థిరరాశి dma_addr_t m_src = mp_పునాది + ic_ప్రారంభం * పరిమాణం(__fp16);
                dma_cache_dm_నెట్టు(dma_q, &m_cache, ib, m_src, current_ఖండం_కొలత * 2, current_ఖండం_కొలత * 2, current_ఖండం_కొలత * 2, 1);
            }
        }

        // Pop all Q rows
        కొరకు (uint32_t g = 0; g < G_local; ++g) {
            uint8_t * q_ptr_vtcm = (శూన్య *) dma_వరుస_తీయి(dma_q).dst;
            ఒకవేళ (factx->is_q_fp32) {
                hvx_copy_f16_f32_aa(q_ptr_vtcm, q_ptr_vtcm, DK);
            }
        }

        స్థిరరాశి HVX_సదిశ v_neg_inf = Q6_Vh_vsplat_R(0xfbff);
        స్థిరరాశి bool has_softcap     = factx->has_softcap;
        స్థిరరాశి HVX_సదిశ v_cap     = has_softcap ? hvx_vec_splat_f16(factx->logit_softcap) : Q6_V_vzero();
        స్థిరరాశి HVX_సదిశ vinf      = Q6_Vh_vsplat_R(0xFC00);
        స్థిరరాశి HVX_సదిశ vmin      = Q6_Vh_vsplat_R(0xFBFF);
        స్థిరరాశి HVX_సదిశ v_log2e   = hvx_vec_splat_f16(EXP_LOG2E_F);
        స్థిరరాశి uint32_t stride_v2   = factx->కొలత_v_అడ్డువరుస_padded * 2;

        కొరకు (uint32_t అడుగు = 0; అడుగు < n_చురుకైన_ఖండాలు; ++అడుగు) {
            స్థిరరాశి uint32_t ib = చురుకైన_ఖండాలు[అడుగు];
            స్థిరరాశి uint32_t ic_ప్రారంభం = ib * FLASH_ATTN_ఖండం_కొలత;
            స్థిరరాశి uint32_t current_ఖండం_కొలత = MIN(FLASH_ATTN_ఖండం_కొలత, nek1 - ic_ప్రారంభం);

            // Wait for DMA
            uint8_t * k_పునాది = (శూన్య *) dma_వరుస_తీయి(dma_q).dst; // K
            uint8_t * v_పునాది = (శూన్య *) dma_వరుస_తీయి(dma_q).dst; // V
            __fp16  * m_పునాది = ముసుగు ? (__fp16 *) dma_వరుస_తీయి(dma_q).dst : ఖాళీ; // M

            ఒకవేళ (factx->k->type == HTP_TYPE_Q8_0) {
                htp_trace_event_ప్రారంభం(tr, HTP_TRACE_EVT_HVX_FA_K_PREP, ir);
                కొరకు (uint32_t r = 0; r < current_ఖండం_కొలత; ++r) {
                    __fp16 * అడ్డువరుస_k = (__fp16 *)(k_పునాది + r * factx->కొలత_k_అడ్డువరుస_padded);
                    hvx_dequantize_అడ్డువరుస_flat_q8_0_f16(అడ్డువరుస_k, అడ్డువరుస_k, DK);
                }
                htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_FA_K_PREP, ir);
            }
            ఒకవేళ (factx->v->type == HTP_TYPE_Q8_0) {
                htp_trace_event_ప్రారంభం(tr, HTP_TRACE_EVT_HVX_FA_V_PREP, ir);
                కొరకు (uint32_t r = 0; r < current_ఖండం_కొలత; ++r) {
                    __fp16 * అడ్డువరుస_v = (__fp16 *)(v_పునాది + r * factx->కొలత_v_అడ్డువరుస_padded);
                    hvx_dequantize_అడ్డువరుస_flat_q8_0_f16(అడ్డువరుస_v, అడ్డువరుస_v, DV);
                }
                htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_FA_V_PREP, ir);
            }

            కొరకు (uint32_t g = 0; g < G_local; ++g) {
                స్థిరరాశి uint32_t తల_ir = ir + g;
                uint8_t * q_ptr_vtcm = q_ptrs[g];
                దశాంశం * VKQ32 = vkq_ptrs[g];
                స్థిరరాశి HVX_సదిశ వాలు_vec = వాలు_vecs[g];

                htp_trace_event_ప్రారంభం(tr, HTP_TRACE_EVT_HVX_FA_QK, తల_ir);

                HVX_సదిశ scores_f16 = Q6_V_vzero();
                ఒకవేళ (current_ఖండం_కొలత > 0) {
                    HVX_సదిశ scores0 = hvx_dot_f16_f16_aa_rx32(q_ptr_vtcm, k_పునాది, factx->కొలత_k_అడ్డువరుస_padded, DK, factx->scale);
                    HVX_సదిశ scores1 = (current_ఖండం_కొలత > 32) ? hvx_dot_f16_f16_aa_rx32(q_ptr_vtcm, k_పునాది + 32 * factx->కొలత_k_అడ్డువరుస_padded, factx->కొలత_k_అడ్డువరుస_padded, DK, factx->scale) : Q6_V_vzero();
                    scores_f16 = hvx_vec_f32_గమ్యం_f16(scores0, scores1);
                }

                ఒకవేళ (has_softcap) {
                    scores_f16 = hvx_vec_tanh_f16(scores_f16);
                    scores_f16 = hvx_vec_mul_f16_f16(scores_f16, v_cap);
                }

                HVX_సదిశ_Pred q_తోక_keep = Q6_Q_vsetq2_R(current_ఖండం_కొలత * పరిమాణం(__fp16));

                ఒకవేళ (ముసుగు) {
                    HVX_సదిశ m_vals_f16 = *(స్థిరరాశి HVX_UVector *) m_పునాది;
                    HVX_సదిశ_Pred is_inf = Q6_Q_vcmp_eq_VhVh(m_vals_f16, vinf);
                    m_vals_f16 = Q6_V_vmux_QVV(is_inf, vmin, m_vals_f16);

                    HVX_సదిశ m_scaled = hvx_vec_mul_f16_f16(m_vals_f16, వాలు_vec);
                    scores_f16 = Q6_V_vmux_QVV(q_తోక_keep, hvx_vec_కూడు_f16_f16(scores_f16, m_scaled), v_neg_inf);
                } లేకపోతే {
                    scores_f16 = Q6_V_vmux_QVV(q_తోక_keep, scores_f16, v_neg_inf);
                }

                HVX_సదిశ v_max_f16 = hvx_vec_reduce_max_f16(scores_f16);
                HVX_సదిశ v_max     = Q6_V_lo_W(hvx_vec_f16_గమ్యం_f32(v_max_f16));
                htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_FA_QK, తల_ir);

                // prefetch K for step + 2 after last head finished QK
                ఒకవేళ (g + 1 == G_local && అడుగు + 2 < n_చురుకైన_ఖండాలు) {
                    స్థిరరాశి uint32_t తదుపరి_ib = చురుకైన_ఖండాలు[అడుగు + 2];
                    స్థిరరాశి uint32_t తదుపరి_ic_ప్రారంభం = తదుపరి_ib * FLASH_ATTN_ఖండం_కొలత;
                    స్థిరరాశి uint32_t తదుపరి_ఖండం_కొలత = MIN(FLASH_ATTN_ఖండం_కొలత, nek1 - తదుపరి_ic_ప్రారంభం);

                    స్థిరరాశి dma_addr_t k_src = k->దత్తాంశం + తదుపరి_ic_ప్రారంభం*nbk1 + ik2*nbk2 + ik3*nbk3;
                    dma_వరుస_నెట్టు(dma_q, dma_make_దత్తాంశం(k_పునాది, k_src), factx->కొలత_k_అడ్డువరుస_padded, nbk1, కొలత_k_అడ్డువరుస, తదుపరి_ఖండం_కొలత);
                }

                htp_trace_event_ప్రారంభం(tr, HTP_TRACE_EVT_HVX_FA_SFM, తల_ir);
                {
                    HVX_సదిశ M_new_vec = Q6_Vsf_vmax_VsfVsf(v_max, M_vec[g]);
                    HVX_సదిశ diff_vec  = HVX_OP_SUB_F32(M_vec[g], M_new_vec);

                    HVX_సదిశ diff_f16   = hvx_vec_f32_గమ్యం_f16(diff_vec, diff_vec);
                    HVX_సదిశ diff_base2 = hvx_vec_mul_f16_f16(diff_f16, v_log2e);
                    HVX_సదిశ ms_f16     = hvx_vec_exp2_f16(diff_base2);
                    HVX_సదిశ ms_vec     = Q6_V_lo_W(hvx_vec_f16_గమ్యం_f32(ms_f16));

                    M_vec[g] = M_new_vec;

                    HVX_సదిశ v_m_vec_f16 = hvx_vec_f32_గమ్యం_f16(M_vec[g], M_vec[g]);
                    HVX_సదిశ v_s_minus_m = Q6_Vqf16_vsub_VhfVhf(scores_f16, v_m_vec_f16);
                    HVX_సదిశ v_s_minus_m_base2 = hvx_vec_mul_f16_f16(Q6_Vhf_equals_Vqf16(v_s_minus_m), v_log2e);

                    HVX_సదిశ P = hvx_vec_exp2_f16(v_s_minus_m_base2);
                    P = Q6_V_vmux_QVV(q_తోక_keep, P, Q6_V_vzero());

                    HVX_సదిశ_Pair P_pair = hvx_vec_f16_గమ్యం_f32(P);
                    HVX_సదిశ P0 = Q6_V_lo_W(P_pair);
                    HVX_సదిశ P1 = Q6_V_hi_W(P_pair);
                    HVX_సదిశ p_కూడిక_vec = hvx_vec_reduce_కూడిక_f32(HVX_OP_కూడు_F32(P0, P1));

                    S_vec[g] = HVX_OP_కూడు_F32(HVX_OP_MUL_F32(S_vec[g], ms_vec), p_కూడిక_vec);

                    స్థిరరాశి uint8_t * v_ptr = v_పునాది;

                    ఒకవేళ (DV == 64) {
                        HVX_సదిశ_Pair vkq0 = *((స్థిరరాశి HVX_సదిశ_Pair *) VKQ32);
                        vkq0 = Q6_W_vcombine_VV(
                            HVX_OP_MUL_F32(Q6_V_hi_W(vkq0), ms_vec),
                            HVX_OP_MUL_F32(Q6_V_lo_W(vkq0), ms_vec)
                        );

                        కొరకు (uint32_t j = 0; j < current_ఖండం_కొలత; j += 2) {
                            HVX_సదిశ S0 = hvx_vec_repl_f16(Q6_V_vror_VR(P, j * 2));
                            స్థిరరాశి HVX_సదిశ * vx0 = (స్థిరరాశి HVX_సదిశ *) v_ptr;
                            ఒకవేళ (j + 1 == current_ఖండం_కొలత) {
                                vkq0 = hvx_vec_mpyacc_f32_f16(vkq0, Q6_Vh_vshuff_Vh(vx0[0]), S0);
                                ఆపు;
                            }

                            HVX_సదిశ S1 = hvx_vec_repl_f16(Q6_V_vror_VR(P, (j + 1) * 2));
                            స్థిరరాశి HVX_సదిశ * vx1 = (స్థిరరాశి HVX_సదిశ *) (v_ptr + factx->కొలత_v_అడ్డువరుస_padded);
                            vkq0 = hvx_vec_mpyacc_f32_f16(vkq0, Q6_Vh_vshuff_Vh(vx0[0]), S0);
                            vkq0 = hvx_vec_mpyacc_f32_f16(vkq0, Q6_Vh_vshuff_Vh(vx1[0]), S1);
                            v_ptr += stride_v2;
                        }

                        *((HVX_సదిశ_Pair *) VKQ32) = vkq0;
                    } లేకపోతే ఒకవేళ (DV == 128) {
                        HVX_సదిశ_Pair vkq0 = ((స్థిరరాశి HVX_సదిశ_Pair *) VKQ32)[0];
                        HVX_సదిశ_Pair vkq1 = ((స్థిరరాశి HVX_సదిశ_Pair *) VKQ32)[1];
                        vkq0 = Q6_W_vcombine_VV(
                            HVX_OP_MUL_F32(Q6_V_hi_W(vkq0), ms_vec),
                            HVX_OP_MUL_F32(Q6_V_lo_W(vkq0), ms_vec)
                        );
                        vkq1 = Q6_W_vcombine_VV(
                            HVX_OP_MUL_F32(Q6_V_hi_W(vkq1), ms_vec),
                            HVX_OP_MUL_F32(Q6_V_lo_W(vkq1), ms_vec)
                        );

                        కొరకు (uint32_t j = 0; j < current_ఖండం_కొలత; j += 2) {
                            HVX_సదిశ S0 = hvx_vec_repl_f16(Q6_V_vror_VR(P, j * 2));
                            స్థిరరాశి HVX_సదిశ * vx0 = (స్థిరరాశి HVX_సదిశ *) v_ptr;
                            ఒకవేళ (j + 1 == current_ఖండం_కొలత) {
                                vkq0 = hvx_vec_mpyacc_f32_f16(vkq0, Q6_Vh_vshuff_Vh(vx0[0]), S0);
                                vkq1 = hvx_vec_mpyacc_f32_f16(vkq1, Q6_Vh_vshuff_Vh(vx0[1]), S0);
                                ఆపు;
                            }

                            HVX_సదిశ S1 = hvx_vec_repl_f16(Q6_V_vror_VR(P, (j + 1) * 2));
                            స్థిరరాశి HVX_సదిశ * vx1 = (స్థిరరాశి HVX_సదిశ *) (v_ptr + factx->కొలత_v_అడ్డువరుస_padded);
                            vkq0 = hvx_vec_mpyacc_f32_f16(vkq0, Q6_Vh_vshuff_Vh(vx0[0]), S0);
                            vkq0 = hvx_vec_mpyacc_f32_f16(vkq0, Q6_Vh_vshuff_Vh(vx1[0]), S1);
                            vkq1 = hvx_vec_mpyacc_f32_f16(vkq1, Q6_Vh_vshuff_Vh(vx0[1]), S0);
                            vkq1 = hvx_vec_mpyacc_f32_f16(vkq1, Q6_Vh_vshuff_Vh(vx1[1]), S1);
                            v_ptr += stride_v2;
                        }

                        ((HVX_సదిశ_Pair *) VKQ32)[0] = vkq0;
                        ((HVX_సదిశ_Pair *) VKQ32)[1] = vkq1;
                    } లేకపోతే ఒకవేళ (DV == 256) {
                        HVX_సదిశ_Pair vkq0 = ((స్థిరరాశి HVX_సదిశ_Pair *) VKQ32)[0];
                        HVX_సదిశ_Pair vkq1 = ((స్థిరరాశి HVX_సదిశ_Pair *) VKQ32)[1];
                        HVX_సదిశ_Pair vkq2 = ((స్థిరరాశి HVX_సదిశ_Pair *) VKQ32)[2];
                        HVX_సదిశ_Pair vkq3 = ((స్థిరరాశి HVX_సదిశ_Pair *) VKQ32)[3];
                        vkq0 = Q6_W_vcombine_VV(
                            HVX_OP_MUL_F32(Q6_V_hi_W(vkq0), ms_vec),
                            HVX_OP_MUL_F32(Q6_V_lo_W(vkq0), ms_vec)
                        );
                        vkq1 = Q6_W_vcombine_VV(
                            HVX_OP_MUL_F32(Q6_V_hi_W(vkq1), ms_vec),
                            HVX_OP_MUL_F32(Q6_V_lo_W(vkq1), ms_vec)
                        );
                        vkq2 = Q6_W_vcombine_VV(
                            HVX_OP_MUL_F32(Q6_V_hi_W(vkq2), ms_vec),
                            HVX_OP_MUL_F32(Q6_V_lo_W(vkq2), ms_vec)
                        );
                        vkq3 = Q6_W_vcombine_VV(
                            HVX_OP_MUL_F32(Q6_V_hi_W(vkq3), ms_vec),
                            HVX_OP_MUL_F32(Q6_V_lo_W(vkq3), ms_vec)
                        );

                        కొరకు (uint32_t j = 0; j < current_ఖండం_కొలత; j += 2) {
                            HVX_సదిశ S0 = hvx_vec_repl_f16(Q6_V_vror_VR(P, j * 2));
                            స్థిరరాశి HVX_సదిశ * vx0 = (స్థిరరాశి HVX_సదిశ *) v_ptr;
                            ఒకవేళ (j + 1 == current_ఖండం_కొలత) {
                                vkq0 = hvx_vec_mpyacc_f32_f16(vkq0, Q6_Vh_vshuff_Vh(vx0[0]), S0);
                                vkq1 = hvx_vec_mpyacc_f32_f16(vkq1, Q6_Vh_vshuff_Vh(vx0[1]), S0);
                                vkq2 = hvx_vec_mpyacc_f32_f16(vkq2, Q6_Vh_vshuff_Vh(vx0[2]), S0);
                                vkq3 = hvx_vec_mpyacc_f32_f16(vkq3, Q6_Vh_vshuff_Vh(vx0[3]), S0);
                                ఆపు;
                            }

                            HVX_సదిశ S1 = hvx_vec_repl_f16(Q6_V_vror_VR(P, (j + 1) * 2));
                            స్థిరరాశి HVX_సదిశ * vx1 = (స్థిరరాశి HVX_సదిశ *) (v_ptr + factx->కొలత_v_అడ్డువరుస_padded);
                            vkq0 = hvx_vec_mpyacc_f32_f16(vkq0, Q6_Vh_vshuff_Vh(vx0[0]), S0);
                            vkq0 = hvx_vec_mpyacc_f32_f16(vkq0, Q6_Vh_vshuff_Vh(vx1[0]), S1);
                            vkq1 = hvx_vec_mpyacc_f32_f16(vkq1, Q6_Vh_vshuff_Vh(vx0[1]), S0);
                            vkq1 = hvx_vec_mpyacc_f32_f16(vkq1, Q6_Vh_vshuff_Vh(vx1[1]), S1);
                            vkq2 = hvx_vec_mpyacc_f32_f16(vkq2, Q6_Vh_vshuff_Vh(vx0[2]), S0);
                            vkq2 = hvx_vec_mpyacc_f32_f16(vkq2, Q6_Vh_vshuff_Vh(vx1[2]), S1);
                            vkq3 = hvx_vec_mpyacc_f32_f16(vkq3, Q6_Vh_vshuff_Vh(vx0[3]), S0);
                            vkq3 = hvx_vec_mpyacc_f32_f16(vkq3, Q6_Vh_vshuff_Vh(vx1[3]), S1);
                            v_ptr += stride_v2;
                        }

                        ((HVX_సదిశ_Pair *) VKQ32)[0] = vkq0;
                        ((HVX_సదిశ_Pair *) VKQ32)[1] = vkq1;
                        ((HVX_సదిశ_Pair *) VKQ32)[2] = vkq2;
                        ((HVX_సదిశ_Pair *) VKQ32)[3] = vkq3;
                    } లేకపోతే {
                        hvx_scale_vec_f32_aa((uint8_t *) VKQ32, (స్థిరరాశి uint8_t *) VKQ32, DV, ms_vec);

                        కొరకు (uint32_t j = 0; j < current_ఖండం_కొలత; j += 2) {
                            ఒకవేళ (j + 1 == current_ఖండం_కొలత) {
                                HVX_సదిశ S0 = hvx_vec_repl_f16(Q6_V_vror_VR(P, j * 2));
                                hvx_mad_f32_f16_aa_vec(VKQ32, v_ptr, S0, DV);
                                ఆపు;
                            }

                            HVX_సదిశ S0 = hvx_vec_repl_f16(Q6_V_vror_VR(P, j * 2));
                            HVX_సదిశ S1 = hvx_vec_repl_f16(Q6_V_vror_VR(P, (j + 1) * 2));

                            hvx_mad_f32_f16_aa_rx2_vec(VKQ32, v_ptr, v_ptr + factx->కొలత_v_అడ్డువరుస_padded, S0, S1, DV);
                            v_ptr += stride_v2;
                        }
                    }
                }
                htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_FA_SFM, తల_ir);

                // prefetch V and mask for step + 2 after last head finished V accumulation
                ఒకవేళ (g + 1 == G_local && అడుగు + 2 < n_చురుకైన_ఖండాలు) {
                    స్థిరరాశి uint32_t తదుపరి_ib = చురుకైన_ఖండాలు[అడుగు + 2];
                    స్థిరరాశి uint32_t తదుపరి_ic_ప్రారంభం = తదుపరి_ib * FLASH_ATTN_ఖండం_కొలత;
                    స్థిరరాశి uint32_t తదుపరి_ఖండం_కొలత = MIN(FLASH_ATTN_ఖండం_కొలత, nek1 - తదుపరి_ic_ప్రారంభం);

                    // V
                    స్థిరరాశి dma_addr_t v_src = v->దత్తాంశం + తదుపరి_ic_ప్రారంభం*nbv1 + iv2*nbv2 + iv3*nbv3;
                    dma_వరుస_నెట్టు(dma_q, dma_make_దత్తాంశం(v_పునాది, v_src), factx->కొలత_v_అడ్డువరుస_padded, nbv1, కొలత_v_అడ్డువరుస, తదుపరి_ఖండం_కొలత);

                    // Mask
                    ఒకవేళ (ముసుగు) {
                        స్థిరరాశి dma_addr_t m_src = mp_పునాది + తదుపరి_ic_ప్రారంభం * పరిమాణం(__fp16);
                        dma_cache_dm_నెట్టు(dma_q, &m_cache, తదుపరి_ib, m_src, తదుపరి_ఖండం_కొలత * 2, తదుపరి_ఖండం_కొలత * 2, తదుపరి_ఖండం_కొలత * 2, 1);
                    }
                }
            } // end for g
        } // end for step

        కొరకు (uint32_t g = 0; g < G_local; ++g) {
            స్థిరరాశి uint32_t తల_ir = ir + g;
            స్థిరరాశి uint32_t h = heads[g];
            దశాంశం * VKQ32 = vkq_ptrs[g];

            htp_trace_event_ప్రారంభం(tr, HTP_TRACE_EVT_HVX_O_PROC, తల_ir);

            దశాంశం M = hvx_vec_get_f32(M_vec[g]);
            దశాంశం S = hvx_vec_get_f32(S_vec[g]);

            ఒకవేళ (sinks) {
                స్థిరరాశి దశాంశం s = factx->spad_sinks[h];

                దశాంశం vs = 1.0f;

                ఒకవేళ (s > M) {
                    HVX_సదిశ diff_vec = hvx_vec_splat_f32(M - s);
                    HVX_సదిశ ms_vec   = hvx_vec_exp_f32(diff_vec);
                    hvx_scale_vec_f32_aa((uint8_t *) VKQ32, (స్థిరరాశి uint8_t *) VKQ32, DV, ms_vec);

                    దశాంశం ms = hvx_vec_get_f32(ms_vec);
                    S = S * ms + vs;
                } లేకపోతే {
                    HVX_సదిశ diff_vec = hvx_vec_splat_f32(s - M);
                    vs = hvx_vec_get_f32(hvx_vec_exp_f32(diff_vec));
                    S += vs;
                }
            }

            స్థిరరాశి దశాంశం S_inv = S == 0.0f ? 0.0f : 1.0f/S;
            hvx_scale_f32_aa((uint8_t *) VKQ32, (స్థిరరాశి uint8_t *) VKQ32, DV, S_inv);

            స్థిరరాశి uint32_t r_iq3 = fastdiv(తల_ir, &factx->src0_div21);
            స్థిరరాశి uint32_t r_iq2 = fastdiv(తల_ir - r_iq3*neq2*neq1, &factx->src0_div1);
            స్థిరరాశి uint32_t r_iq1 = (తల_ir - r_iq3*neq2*neq1 - r_iq2 * neq1);

            uint8_t * dst_ptr = (uint8_t *) dst->దత్తాంశం + r_iq2 * dst->nb[1] + r_iq1 * dst->nb[2] + r_iq3 * dst->nb[3];

            ఒకవేళ (dst->type == HTP_TYPE_F32) {
                hvx_copy_f32_ua(dst_ptr, (uint8_t *) VKQ32, DV);
            } లేకపోతే ఒకవేళ (dst->type == HTP_TYPE_F16) {
                hvx_copy_f16_f32_ua(dst_ptr, (uint8_t *) VKQ32, DV);
            }
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_O_PROC, తల_ir);
        }

        ir += G_local;
    }
}

// ============================================================================
// HMX Phase args and thread logic
// ============================================================================

రకనిర్వచనం నిర్మాణం {
    నిర్మాణం hmx_fa_context * factx;
    uint32_t                kv_అడ్డువరుసలు;
    size_t                  src_stride;
    శూన్య *                  curr_k;
    uint32_t                kv_ప్రారంభం;
    uint32_t                అడ్డువరుసలు_చొప్పున_t;
    size_t                  buf_idx;
} fa_k_int_వాదనలు_t;

స్థిర శూన్య fa_k_interleave_thread(గుర్తులేని పూర్ణాంకం n, గుర్తులేని పూర్ణాంకం i, శూన్య * దత్తాంశం) {
    fa_k_int_వాదనలు_t *       వాదనలు_2  = (fa_k_int_వాదనలు_t *) దత్తాంశం;
    నిర్మాణం hmx_fa_context * factx = వాదనలు_2->factx;

    స్థిరరాశి uint32_t మొత్తం_అడ్డువరుసలు = వాదనలు_2->kv_అడ్డువరుసలు;
    స్థిరరాశి uint32_t అడ్డువరుసలు_చొప్పున_t = వాదనలు_2->అడ్డువరుసలు_చొప్పున_t;
    స్థిరరాశి uint32_t ప్రారంభం      = i * అడ్డువరుసలు_చొప్పున_t;
    స్థిరరాశి uint32_t చివర        = (uint32_t) hex_smin(ప్రారంభం + అడ్డువరుసలు_చొప్పున_t, మొత్తం_అడ్డువరుసలు);

    ఒకవేళ (ప్రారంభం >= మొత్తం_అడ్డువరుసలు) {
        తిరిగి;
    }

    నిర్మాణం htp_thread_trace * tr = &factx->octx->ctx->trace[i];
    htp_trace_event_ప్రారంభం(tr, HTP_TRACE_EVT_HVX_FA_K_PREP, (uint16_t) (వాదనలు_2->kv_ప్రారంభం + ప్రారంభం));
    ఒకవేళ (factx->k->type == HTP_TYPE_Q8_0) {
        కొరకు (uint32_t r = ప్రారంభం; r < చివర; ++r) {
            __fp16 * అడ్డువరుస_k = (__fp16 *)((అక్షరం *)వాదనలు_2->curr_k + r * వాదనలు_2->src_stride * పరిమాణం(__fp16));
            hvx_dequantize_అడ్డువరుస_flat_q8_0_f16(అడ్డువరుస_k, అడ్డువరుస_k, factx->DK);
        }
    }
    hmx_interleave_అడ్డువరుసలు_గమ్యం_tiles(factx->vtcm_k_tiles[వాదనలు_2->buf_idx], (స్థిరరాశి __fp16 *) వాదనలు_2->curr_k, మొత్తం_అడ్డువరుసలు, factx->DK_pad,
                             వాదనలు_2->src_stride, ప్రారంభం, చివర);
    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_FA_K_PREP, (uint16_t) (వాదనలు_2->kv_ప్రారంభం + ప్రారంభం));
}

స్థిర శూన్య fa_phase_k_interleave(నిర్మాణం hmx_fa_context * factx, uint32_t kv_అడ్డువరుసలు, size_t src_stride, శూన్య * curr_k, uint32_t kv_ప్రారంభం, size_t buf_idx) {
    work_వరుస_t wp = factx->octx->ctx->work_వరుస;
    uint32_t n = 1;
    ఒకవేళ (factx->n_దారాలు > 1 && kv_అడ్డువరుసలు >= factx->n_దారాలు * 2) {
        n = factx->n_దారాలు;
    }
    uint32_t అడ్డువరుసలు_చొప్పున_t = hex_align_పైకి(hmx_ceil_div(kv_అడ్డువరుసలు, n), 2);
    fa_k_int_వాదనలు_t వాదనలు_2 = { factx, kv_అడ్డువరుసలు, src_stride, curr_k, kv_ప్రారంభం, అడ్డువరుసలు_చొప్పున_t, buf_idx };
    ఒకవేళ (n > 1) {
        work_వరుస_నిరంతరం(wp, fa_k_interleave_thread, &వాదనలు_2, n);
    } లేకపోతే {
        fa_k_interleave_thread(1, 0, &వాదనలు_2);
    }
}

రకనిర్వచనం నిర్మాణం {
    నిర్మాణం hmx_fa_context * factx;
    uint32_t                kv_అడ్డువరుసలు;
    size_t                  src_stride;
    శూన్య *                  v_src;
    శూన్య *                  v_tiles_dst;
    size_t                  n_నిలువు_tiles;
    uint32_t                kv_ప్రారంభం;
    uint32_t                అడ్డువరుసలు_చొప్పున_t;
} fa_v_int_వాదనలు_t;

స్థిర శూన్య fa_v_interleave_thread(గుర్తులేని పూర్ణాంకం n, గుర్తులేని పూర్ణాంకం i, శూన్య * దత్తాంశం) {
    fa_v_int_వాదనలు_t *       వాదనలు_2  = (fa_v_int_వాదనలు_t *) దత్తాంశం;
    నిర్మాణం hmx_fa_context * factx = వాదనలు_2->factx;

    స్థిరరాశి uint32_t మొత్తం_అడ్డువరుసలు = వాదనలు_2->kv_అడ్డువరుసలు;
    స్థిరరాశి uint32_t అడ్డువరుసలు_చొప్పున_t = వాదనలు_2->అడ్డువరుసలు_చొప్పున_t;
    స్థిరరాశి uint32_t ప్రారంభం      = i * అడ్డువరుసలు_చొప్పున_t;
    స్థిరరాశి uint32_t చివర        = (uint32_t) hex_smin(ప్రారంభం + అడ్డువరుసలు_చొప్పున_t, మొత్తం_అడ్డువరుసలు);

    ఒకవేళ (ప్రారంభం >= మొత్తం_అడ్డువరుసలు) {
        తిరిగి;
    }

    __fp16 * v_tiles_dst = (__fp16 *) వాదనలు_2->v_tiles_dst;

    నిర్మాణం htp_thread_trace * tr = &factx->octx->ctx->trace[i];
    htp_trace_event_ప్రారంభం(tr, HTP_TRACE_EVT_HVX_FA_V_PREP, (uint16_t) (వాదనలు_2->kv_ప్రారంభం + ప్రారంభం));
    ఒకవేళ (factx->v->type == HTP_TYPE_Q8_0) {
        కొరకు (uint32_t r = ప్రారంభం; r < చివర; ++r) {
            __fp16 * అడ్డువరుస_v = (__fp16 *)((అక్షరం *)వాదనలు_2->v_src + r * వాదనలు_2->src_stride * పరిమాణం(__fp16));
            hvx_dequantize_అడ్డువరుస_flat_q8_0_f16(అడ్డువరుస_v, అడ్డువరుస_v, factx->DV);
        }
    }
    hmx_interleave_నిలువుల_సంఖ్య_గమ్యం_tiles(v_tiles_dst, (స్థిరరాశి __fp16 *) వాదనలు_2->v_src, మొత్తం_అడ్డువరుసలు, factx->DV_pad,
                             వాదనలు_2->src_stride, (uint32_t) వాదనలు_2->n_నిలువు_tiles, ప్రారంభం, చివర);
    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_FA_V_PREP, (uint16_t) (వాదనలు_2->kv_ప్రారంభం + ప్రారంభం));
}

స్థిర శూన్య fa_phase_v_interleave(నిర్మాణం hmx_fa_context * factx,
                                  uint32_t                kv_అడ్డువరుసలు,
                                  size_t                  src_stride,
                                  శూన్య *                  v_src,
                                  శూన్య *                  v_tiles_dst,
                                  size_t                  n_నిలువు_tiles,
                                  uint32_t                kv_ప్రారంభం) {
    work_వరుస_t wp = factx->octx->ctx->work_వరుస;
    uint32_t n = 1;
    ఒకవేళ (factx->n_దారాలు > 1 && kv_అడ్డువరుసలు >= factx->n_దారాలు * 2) {
        n = factx->n_దారాలు;
    }
    uint32_t అడ్డువరుసలు_చొప్పున_t = hex_align_పైకి(hmx_ceil_div(kv_అడ్డువరుసలు, n), 2);
    fa_v_int_వాదనలు_t వాదనలు_2 = { factx, kv_అడ్డువరుసలు, src_stride, v_src, v_tiles_dst, n_నిలువు_tiles, kv_ప్రారంభం, అడ్డువరుసలు_చొప్పున_t };
    ఒకవేళ (n > 1) {
        work_వరుస_నిరంతరం(wp, fa_v_interleave_thread, &వాదనలు_2, n);
    } లేకపోతే {
        fa_v_interleave_thread(1, 0, &వాదనలు_2);
    }
}

రకనిర్వచనం నిర్మాణం {
    నిర్మాణం hmx_fa_context *   factx;
    స్థిరరాశి నిర్మాణం htp_tensor * q;
    uint32_t                  q_ప్రారంభం;
    uint32_t                  kv_తల;
    uint32_t                  ib3;
    size_t                    n_అడ్డువరుసలు_g;
    size_t                    అడ్డువరుసలు_చొప్పున_t;
    size_t                    n_అడ్డువరుసలు_q;
    bool                      q_transposed;
    atomic_uint               barrier;
} fa_q_ఎక్కించు_వాదనలు_t;

స్థిర శూన్య fa_q_ఎక్కించు_thread(గుర్తులేని పూర్ణాంకం n, గుర్తులేని పూర్ణాంకం i, శూన్య * దత్తాంశం) {
    fa_q_ఎక్కించు_వాదనలు_t *      వాదనలు_2  = (fa_q_ఎక్కించు_వాదనలు_t *) దత్తాంశం;
    నిర్మాణం hmx_fa_context * factx = వాదనలు_2->factx;

    స్థిరరాశి size_t n_అడ్డువరుసలు_g = వాదనలు_2->n_అడ్డువరుసలు_g;
    స్థిరరాశి size_t G        = factx->G;
    స్థిరరాశి size_t DK       = factx->DK;

    // Partition the padded Q rows (g_br) across threads.
    // Keep start/end even so r and r+1 are always in the same thread's range.
    స్థిరరాశి size_t అడ్డువరుసలు_చొప్పున_t = వాదనలు_2->అడ్డువరుసలు_చొప్పున_t;
    స్థిరరాశి size_t ప్రారంభం      = (size_t) i * అడ్డువరుసలు_చొప్పున_t;
    స్థిరరాశి size_t చివర        = hex_smin(ప్రారంభం + అడ్డువరుసలు_చొప్పున_t, factx->g_br);

    నిర్మాణం htp_thread_trace * tr = &factx->octx->ctx->trace[i];
    htp_trace_event_ప్రారంభం(tr, HTP_TRACE_EVT_HVX_FA_Q_PREP, (uint16_t) (వాదనలు_2->q_ప్రారంభం * G + ప్రారంభం));

    // Parallel initialization of per-block state
    {
        స్థిరరాశి uint32_t g_br = factx->g_br;
        స్థిరరాశి uint32_t DV   = factx->DV;

        స్థిరరాశి size_t నిలువు_vec_బైట్లు = factx->నిలువు_vec_బైట్లు;
        స్థిరరాశి size_t d_tile_బైట్లు  = factx->d_tile_బైట్లు;

        // Initialize vtcm_l_vec & vtcm_m_vec
        స్థిరరాశి size_t l_బైట్లు_చొప్పున_t = hex_align_పైకి(నిలువు_vec_బైట్లు / n, 128);
        స్థిరరాశి size_t l_ప్రారంభం       = i * l_బైట్లు_చొప్పున_t;
        స్థిరరాశి size_t l_చివర         = hex_smin(l_ప్రారంభం + l_బైట్లు_చొప్పున_t, నిలువు_vec_బైట్లు);

        స్థిరరాశి size_t m_బైట్లు_చొప్పున_t = hex_align_పైకి(నిలువు_vec_బైట్లు / n, 128);
        స్థిరరాశి size_t m_ప్రారంభం       = i * m_బైట్లు_చొప్పున_t;
        స్థిరరాశి size_t m_చివర         = hex_smin(m_ప్రారంభం + m_బైట్లు_చొప్పున_t, నిలువు_vec_బైట్లు);

        ఒకవేళ (l_ప్రారంభం < నిలువు_vec_బైట్లు) {
            hvx_splat_u8_a((అక్షరం *) factx->vtcm_l_vec + l_ప్రారంభం, 0, l_చివర - l_ప్రారంభం);
        }
        ఒకవేళ (m_ప్రారంభం < నిలువు_vec_బైట్లు) {
            hvx_splat_f32_a((అక్షరం *) factx->vtcm_m_vec + m_ప్రారంభం, HTP_FA_M_INITIAL_VAL, (m_చివర - m_ప్రారంభం) / పరిమాణం(దశాంశం));
        }

        // Zero the whole rescale region: vtcm_d_tiles[0], the optional vtcm_d_tiles[1]
        // and vtcm_d_inv_l are equal-sized and allocated back to back, so one run covers
        // them all.  The scatter only ever writes the diagonal, ignore the rest.
        స్థిరరాశి size_t d_బైట్లు_చొప్పున_t = hex_align_పైకి(d_tile_బైట్లు / n, 128);
        స్థిరరాశి size_t d_ప్రారంభం       = i * d_బైట్లు_చొప్పున_t;
        స్థిరరాశి size_t d_చివర         = hex_smin(d_ప్రారంభం + d_బైట్లు_చొప్పున_t, d_tile_బైట్లు);
        ఒకవేళ (d_ప్రారంభం < d_tile_బైట్లు) {
            hvx_splat_u8_a((అక్షరం *) factx->vtcm_d_tiles[0] + d_ప్రారంభం, 0, d_చివర - d_ప్రారంభం);
        }
    }

    ఒకవేళ (ప్రారంభం < factx->g_br) {
        స్థిరరాశి నిర్మాణం htp_tensor * q       = వాదనలు_2->q;
        స్థిరరాశి uint32_t            q_ప్రారంభం = వాదనలు_2->q_ప్రారంభం;
        స్థిరరాశి uint32_t            kv_తల = వాదనలు_2->kv_తల;
        స్థిరరాశి uint32_t            ib3     = వాదనలు_2->ib3;

        స్థిరరాశి bool use_q_dma = (factx->vtcm_q_dma != ఖాళీ);

        __fp16 * q_tiles = factx->vtcm_q_tiles;
        స్థిరరాశి size_t DK_pad = factx->DK_pad;
        ఒకవేళ (use_q_dma) {
            స్థిరరాశి size_t g_అడ్డువరుసలు_చివర = hex_smin(చివర, n_అడ్డువరుసలు_g);
            స్థిరరాశి uint32_t d_హద్దు = factx->is_q_fp32 ? DK / 32 : DK / 64;

            uint8_t * q_flat  = (uint8_t *) factx->vtcm_q_dma;
            ఒకవేళ (DK_pad != DK) {
                ఒకవేళ (factx->is_q_fp32) {
                    hmx_fa_q_prep_fp32_pad(q_tiles, q_flat, ప్రారంభం, చివర, g_అడ్డువరుసలు_చివర, DK, DK_pad, G, వాదనలు_2->n_అడ్డువరుసలు_q, &factx->div_G, వాదనలు_2->q_transposed);
                } లేకపోతే {
                    hmx_fa_q_prep_fp16_pad(q_tiles, q_flat, ప్రారంభం, చివర, g_అడ్డువరుసలు_చివర, DK, DK_pad, G, వాదనలు_2->n_అడ్డువరుసలు_q, &factx->div_G, వాదనలు_2->q_transposed);
                }
            } లేకపోతే ఒకవేళ (factx->is_q_fp32) {
                ఎంపిక (d_హద్దు) {
                స్థితి 2:  hmx_fa_q_prep_fp32_d2(q_tiles, q_flat, ప్రారంభం, చివర, g_అడ్డువరుసలు_చివర, DK, G, వాదనలు_2->n_అడ్డువరుసలు_q, &factx->div_G, వాదనలు_2->q_transposed); ఆపు;
                స్థితి 4:  hmx_fa_q_prep_fp32_d4(q_tiles, q_flat, ప్రారంభం, చివర, g_అడ్డువరుసలు_చివర, DK, G, వాదనలు_2->n_అడ్డువరుసలు_q, &factx->div_G, వాదనలు_2->q_transposed); ఆపు;
                అప్రమేయ: hmx_fa_q_prep_fp32(   q_tiles, q_flat, ప్రారంభం, చివర, g_అడ్డువరుసలు_చివర, DK, G, వాదనలు_2->n_అడ్డువరుసలు_q, &factx->div_G, d_హద్దు, వాదనలు_2->q_transposed); ఆపు;
                }
            } లేకపోతే {
                ఎంపిక (d_హద్దు) {
                స్థితి 1:  hmx_fa_q_prep_fp16_d1(q_tiles, q_flat, ప్రారంభం, చివర, g_అడ్డువరుసలు_చివర, DK, G, వాదనలు_2->n_అడ్డువరుసలు_q, &factx->div_G, వాదనలు_2->q_transposed); ఆపు;
                స్థితి 2:  hmx_fa_q_prep_fp16_d2(q_tiles, q_flat, ప్రారంభం, చివర, g_అడ్డువరుసలు_చివర, DK, G, వాదనలు_2->n_అడ్డువరుసలు_q, &factx->div_G, వాదనలు_2->q_transposed); ఆపు;
                అప్రమేయ: hmx_fa_q_prep_fp16(   q_tiles, q_flat, ప్రారంభం, చివర, g_అడ్డువరుసలు_చివర, DK, G, వాదనలు_2->n_అడ్డువరుసలు_q, &factx->div_G, d_హద్దు, వాదనలు_2->q_transposed); ఆపు;
                }
            }
        } లేకపోతే {
            // Fallback: direct-from-DDR/L2 path
            hmx_fa_q_prep_fallback(q_tiles, q->దత్తాంశం, q->nb[1], q->nb[2], q->nb[3],
                                   q_ప్రారంభం, kv_తల, ib3, ప్రారంభం, చివర, n_అడ్డువరుసలు_g, G, DK, DK_pad, factx->is_q_fp32, &factx->div_G);
        }
    }

    // Synchronize threads before zeroing out vtcm_o_tiles[0] to prevent race condition
    ఒకవేళ (n > 1) {
        atomic_fetch_sub(&వాదనలు_2->barrier, 1);
        అయితే (atomic_ఎక్కించు(&వాదనలు_2->barrier) > 0) {
            // spin wait
        }
    }

    // Zero out vtcm_o_tiles[0] as it was used as temp_q_vtcm
    {
        స్థిరరాశి uint32_t g_br = factx->g_br;
        స్థిరరాశి uint32_t DV   = factx->DV;
        స్థిరరాశి size_t o_tile_బైట్లు  = factx->o_tile_బైట్లు;
        స్థిరరాశి size_t o_బైట్లు_చొప్పున_t = hex_align_పైకి(o_tile_బైట్లు / n, 128);
        స్థిరరాశి size_t o_ప్రారంభం       = i * o_బైట్లు_చొప్పున_t;
        స్థిరరాశి size_t o_చివర         = hex_smin(o_ప్రారంభం + o_బైట్లు_చొప్పున_t, o_tile_బైట్లు);
        ఒకవేళ (o_ప్రారంభం < o_tile_బైట్లు) {
            hvx_splat_u8_a((అక్షరం *) factx->vtcm_o_tiles[0] + o_ప్రారంభం, 0, o_చివర - o_ప్రారంభం);
        }
    }
    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_FA_Q_PREP, (uint16_t) (వాదనలు_2->q_ప్రారంభం * G + ప్రారంభం));
}

స్థిర శూన్య fa_phase_q_ఎక్కించు(నిర్మాణం hmx_fa_context *   factx,
                            స్థిరరాశి నిర్మాణం htp_tensor * q,
                            uint32_t                  q_ప్రారంభం,
                            uint32_t                  kv_తల,
                            uint32_t                  ib3,
                            size_t                    n_అడ్డువరుసలు_g) {
    work_వరుస_t wp = factx->octx->ctx->work_వరుస;
    uint32_t n = 1;
    ఒకవేళ (factx->n_దారాలు > 1 && n_అడ్డువరుసలు_g >= (size_t) (factx->n_దారాలు * 2)) {
        n = factx->n_దారాలు;
    }
    size_t అడ్డువరుసలు_చొప్పున_t = hex_align_పైకి(hmx_ceil_div(factx->g_br, n), 2);
    స్థిరరాశి uint32_t n_అడ్డువరుసలు_q = hex_smin(factx->Br, factx->neq1 - q_ప్రారంభం);
    fa_q_ఎక్కించు_వాదనలు_t వాదనలు_2;
    వాదనలు_2.factx = factx;
    వాదనలు_2.q = q;
    వాదనలు_2.q_ప్రారంభం = q_ప్రారంభం;
    వాదనలు_2.kv_తల = kv_తల;
    వాదనలు_2.ib3 = ib3;
    వాదనలు_2.n_అడ్డువరుసలు_g = n_అడ్డువరుసలు_g;
    వాదనలు_2.అడ్డువరుసలు_చొప్పున_t = అడ్డువరుసలు_చొప్పున_t;
    వాదనలు_2.n_అడ్డువరుసలు_q = n_అడ్డువరుసలు_q;
    వాదనలు_2.q_transposed = q->nb[1] < q->nb[2];
    atomic_init(&వాదనలు_2.barrier, n);
    ఒకవేళ (n > 1) {
        work_వరుస_నిరంతరం(wp, fa_q_ఎక్కించు_thread, &వాదనలు_2, n);
    } లేకపోతే {
        fa_q_ఎక్కించు_thread(1, 0, &వాదనలు_2);
    }
}

రకనిర్వచనం నిర్మాణం {
    నిర్మాణం hmx_fa_context *   factx;
    స్థిరరాశి నిర్మాణం htp_tensor * dst;
    స్థిరరాశి __fp16 *            o_tile_src;
    uint32_t                  q_ప్రారంభం;
    uint32_t                  kv_తల;
    uint32_t                  ib3;
    size_t                    n_అడ్డువరుసలు_g;
    size_t                    అడ్డువరుసలు_చొప్పున_t;
} fa_o_store_వాదనలు_t;

స్థిర శూన్య fa_o_store_thread_f32(గుర్తులేని పూర్ణాంకం n, గుర్తులేని పూర్ణాంకం i, శూన్య * దత్తాంశం) {
    fa_o_store_వాదనలు_t *     వాదనలు_2  = (fa_o_store_వాదనలు_t *) దత్తాంశం;
    నిర్మాణం hmx_fa_context * factx = వాదనలు_2->factx;

    స్థిరరాశి size_t n_అడ్డువరుసలు_g = వాదనలు_2->n_అడ్డువరుసలు_g;
    స్థిరరాశి size_t G        = factx->G;
    స్థిరరాశి size_t DV       = factx->DV;

    స్థిరరాశి size_t అడ్డువరుసలు_చొప్పున_t = వాదనలు_2->అడ్డువరుసలు_చొప్పున_t;
    స్థిరరాశి size_t ప్రారంభం      = (size_t) i * అడ్డువరుసలు_చొప్పున_t;
    స్థిరరాశి size_t చివర        = hex_smin(ప్రారంభం + అడ్డువరుసలు_చొప్పున_t, n_అడ్డువరుసలు_g);

    ఒకవేళ (ప్రారంభం >= n_అడ్డువరుసలు_g) {
        తిరిగి;
    }

    నిర్మాణం htp_thread_trace * tr = &factx->octx->ctx->trace[i];
    htp_trace_event_ప్రారంభం(tr, HTP_TRACE_EVT_HVX_O_PROC, (uint16_t) (వాదనలు_2->q_ప్రారంభం * G + ప్రారంభం));

    స్థిరరాశి నిర్మాణం htp_tensor * dst        = వాదనలు_2->dst;
    స్థిరరాశి __fp16 *            o_tile_src = వాదనలు_2->o_tile_src;
    స్థిరరాశి uint32_t            q_ప్రారంభం    = వాదనలు_2->q_ప్రారంభం;
    స్థిరరాశి uint32_t            kv_తల    = వాదనలు_2->kv_తల;
    స్థిరరాశి uint32_t            ib3        = వాదనలు_2->ib3;

    స్థిరరాశి size_t DV_pad = factx->DV_pad;

    size_t q_idx = fastdiv(ప్రారంభం, &factx->div_G);
    size_t h_idx = fastmodulo(ప్రారంభం, G, &factx->div_G);

    కొరకు (size_t r = ప్రారంభం; r < చివర; ++r) {
        దశాంశం * బహిర్గతం = (దశాంశం *) ((uint8_t *) dst->దత్తాంశం + (kv_తల * G + h_idx) * dst->nb[1] +
                                 (q_ప్రారంభం + q_idx) * dst->nb[2] + ib3 * dst->nb[3]);

        size_t         r0            = r / HMX_FP16_TILE_N_అడ్డువరుసలు;
        size_t         r1            = r % HMX_FP16_TILE_N_అడ్డువరుసలు;
        స్థిరరాశి __fp16 * tile_అడ్డువరుస_పునాది = o_tile_src + r0 * HMX_FP16_TILE_N_అడ్డువరుసలు * DV_pad;

        కొరకు (uint32_t d = 0; d < DV / 32; ++d) {
            స్థిరరాశి HVX_సదిశ * లోపల_tile = (స్థిరరాశి HVX_సదిశ *) (tile_అడ్డువరుస_పునాది + d * HMX_FP16_TILE_N_ELMS);
            HVX_సదిశ_Pair     vp      = hvx_vec_f16_గమ్యం_f32_shuff(లోపల_tile[r1 / 2]);
            ఒకవేళ (r1 % 2 == 0) {
                *(HVX_UVector *) (బహిర్గతం + d * 32) = Q6_V_lo_W(vp);
            } లేకపోతే {
                *(HVX_UVector *) (బహిర్గతం + d * 32) = Q6_V_hi_W(vp);
            }
        }
        // Ragged tail: DV not a multiple of 32 (e.g. 72 -> last 8 lanes). Partial vector-write
        // for the remaining (DV % 32) floats.
        స్థిరరాశి uint32_t d_తోక = DV / 32;
        స్థిరరాశి uint32_t rem    = DV - d_తోక * 32;
        ఒకవేళ (rem) {
            స్థిరరాశి HVX_సదిశ * లోపల_tile = (స్థిరరాశి HVX_సదిశ *) (tile_అడ్డువరుస_పునాది + d_తోక * HMX_FP16_TILE_N_ELMS);
            HVX_సదిశ_Pair     vp      = hvx_vec_f16_గమ్యం_f32_shuff(లోపల_tile[r1 / 2]);
            HVX_సదిశ         vd      = (r1 % 2 == 0) ? Q6_V_lo_W(vp) : Q6_V_hi_W(vp);
            hvx_vec_store_u((శూన్య *) (బహిర్గతం + d_తోక * 32), rem * పరిమాణం(దశాంశం), vd);
        }

        h_idx++;
        ఒకవేళ (h_idx == G) {
            h_idx = 0;
            q_idx++;
        }
    }
    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_O_PROC, (uint16_t) (వాదనలు_2->q_ప్రారంభం * G + ప్రారంభం));
}

స్థిర శూన్య fa_o_store_thread_f16(గుర్తులేని పూర్ణాంకం n, గుర్తులేని పూర్ణాంకం i, శూన్య * దత్తాంశం) {
    fa_o_store_వాదనలు_t *     వాదనలు_2  = (fa_o_store_వాదనలు_t *) దత్తాంశం;
    నిర్మాణం hmx_fa_context * factx = వాదనలు_2->factx;

    స్థిరరాశి size_t n_అడ్డువరుసలు_g   = వాదనలు_2->n_అడ్డువరుసలు_g;
    స్థిరరాశి size_t అడ్డువరుసలు_చొప్పున_t = వాదనలు_2->అడ్డువరుసలు_చొప్పున_t;
    స్థిరరాశి size_t G          = factx->G;
    స్థిరరాశి size_t DV         = factx->DV;
    స్థిరరాశి size_t ప్రారంభం      = (size_t) i * అడ్డువరుసలు_చొప్పున_t;
    స్థిరరాశి size_t చివర        = hex_smin(ప్రారంభం + అడ్డువరుసలు_చొప్పున_t, n_అడ్డువరుసలు_g);

    ఒకవేళ (ప్రారంభం >= n_అడ్డువరుసలు_g) {
        తిరిగి;
    }

    నిర్మాణం htp_thread_trace * tr = &factx->octx->ctx->trace[i];
    htp_trace_event_ప్రారంభం(tr, HTP_TRACE_EVT_HVX_O_PROC, (uint16_t) (వాదనలు_2->q_ప్రారంభం * G + ప్రారంభం));

    స్థిరరాశి నిర్మాణం htp_tensor * dst        = వాదనలు_2->dst;
    స్థిరరాశి __fp16 *            o_tile_src = వాదనలు_2->o_tile_src;
    స్థిరరాశి uint32_t            q_ప్రారంభం    = వాదనలు_2->q_ప్రారంభం;
    స్థిరరాశి uint32_t            kv_తల    = వాదనలు_2->kv_తల;
    స్థిరరాశి uint32_t            ib3        = వాదనలు_2->ib3;

    // O-tiles use the padded head dim (DV_pad); dst holds the real DV lanes.
    స్థిరరాశి size_t DV_pad = factx->DV_pad;

    size_t q_idx = fastdiv(ప్రారంభం, &factx->div_G);
    size_t h_idx = fastmodulo(ప్రారంభం, G, &factx->div_G);

    కొరకు (size_t r = ప్రారంభం; r < చివర; ++r) {
        __fp16 * బహిర్గతం = (__fp16 *) ((uint8_t *) dst->దత్తాంశం + (kv_తల * G + h_idx) * dst->nb[1] +
                                   (q_ప్రారంభం + q_idx) * dst->nb[2] + ib3 * dst->nb[3]);

        size_t         r0            = r / HMX_FP16_TILE_N_అడ్డువరుసలు;
        size_t         r1            = r % HMX_FP16_TILE_N_అడ్డువరుసలు;
        స్థిరరాశి __fp16 * tile_అడ్డువరుస_పునాది = o_tile_src + r0 * HMX_FP16_TILE_N_అడ్డువరుసలు * DV_pad;

        కొరకు (uint32_t d = 0; d < DV / 64; ++d) {
            స్థిరరాశి __fp16 *     లోపల_dtile = tile_అడ్డువరుస_పునాది + d * HMX_FP16_TILE_N_ELMS * 2;
            స్థిరరాశి HVX_సదిశ * pv_in0   = ((స్థిరరాశి HVX_సదిశ *) లోపల_dtile) + r1 / 2;
            స్థిరరాశి HVX_సదిశ * pv_in1   = pv_in0 + 16;
            HVX_సదిశ_Pair     vp       = Q6_W_vdeal_VVR(*pv_in1, *pv_in0, -2);
            ఒకవేళ (r1 % 2 == 0) {
                *(HVX_UVector *) (బహిర్గతం + d * 64) = Q6_V_lo_W(vp);
            } లేకపోతే {
                *(HVX_UVector *) (బహిర్గతం + d * 64) = Q6_V_hi_W(vp);
            }
        }
        // Ragged tail when DV is not a multiple of 64.
        స్థిరరాశి uint32_t d_తోక = DV / 64;
        స్థిరరాశి uint32_t rem    = DV - d_తోక * 64;
        ఒకవేళ (rem) {
            స్థిరరాశి __fp16 *     లోపల_dtile = tile_అడ్డువరుస_పునాది + d_తోక * HMX_FP16_TILE_N_ELMS * 2;
            స్థిరరాశి HVX_సదిశ * pv_in0   = ((స్థిరరాశి HVX_సదిశ *) లోపల_dtile) + r1 / 2;
            స్థిరరాశి HVX_సదిశ * pv_in1   = pv_in0 + 16;
            HVX_సదిశ_Pair     vp       = Q6_W_vdeal_VVR(*pv_in1, *pv_in0, -2);
            HVX_సదిశ         vd       = (r1 % 2 == 0) ? Q6_V_lo_W(vp) : Q6_V_hi_W(vp);
            hvx_vec_store_u((శూన్య *) (బహిర్గతం + d_తోక * 64), rem * పరిమాణం(__fp16), vd);
        }

        h_idx++;
        ఒకవేళ (h_idx == G) {
            h_idx = 0;
            q_idx++;
        }
    }
    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_O_PROC, (uint16_t) (వాదనలు_2->q_ప్రారంభం * G + ప్రారంభం));
}

స్థిర శూన్య fa_phase_o_store(నిర్మాణం hmx_fa_context *   factx,
                             స్థిరరాశి నిర్మాణం htp_tensor * dst,
                             స్థిరరాశి __fp16 *            o_tile_src,
                             uint32_t                  q_ప్రారంభం,
                             uint32_t                  kv_తల,
                             uint32_t                  ib3,
                             size_t                    n_అడ్డువరుసలు_g) {
    work_వరుస_t wp = factx->octx->ctx->work_వరుస;
    uint32_t n = 1;
    ఒకవేళ (factx->n_దారాలు > 1 && n_అడ్డువరుసలు_g >= (size_t) (factx->n_దారాలు * 2)) {
        n = factx->n_దారాలు;
    }
    size_t అడ్డువరుసలు_చొప్పున_t = hmx_ceil_div(n_అడ్డువరుసలు_g, n);
    fa_o_store_వాదనలు_t వాదనలు_2 = { factx, dst, o_tile_src, q_ప్రారంభం, kv_తల, ib3, n_అడ్డువరుసలు_g, అడ్డువరుసలు_చొప్పున_t };
    కార్మికుడు_callback_t store_fn = factx->is_dst_fp32 ? fa_o_store_thread_f32 : fa_o_store_thread_f16;
    ఒకవేళ (n > 1) {
        work_వరుస_నిరంతరం(wp, store_fn, &వాదనలు_2, n);
    } లేకపోతే {
        store_fn(1, 0, &వాదనలు_2);
    }
}

రకనిర్వచనం నిర్మాణం {
    నిర్మాణం hmx_fa_context *   factx;
    size_t                    buf_idx;
    size_t                    kv_అడ్డువరుసలు;
    size_t                    n_అడ్డువరుసలు_g;
    size_t                    n_నిలువు_tiles;
    size_t                    n_tiles_చొప్పున_bc;
    size_t                    n_అడ్డువరుస_tiles;
    size_t                    n_అడ్డువరుస_tiles_g_br;
    uint32_t                  Bc;
    uint32_t                  G;
    uint32_t                  kv_తల;
    uint32_t                  kv_ప్రారంభం;
    uint32_t                  q_ప్రారంభం;
    uint32_t                  ib3;
    bool                      has_alibi;  // true when max_bias != 0 (need slope * mask + add)
    __fp16 *                  slopes;
    స్థిరరాశి నిర్మాణం htp_tensor * ముసుగు;
    స్థిరరాశి __fp16 *            ముసుగు_vtcm;             // VTCM mask buffer base (NULL = DDR fallback)
    size_t                    ముసుగు_vtcm_అడ్డువరుస_stride;  // elements (__fp16) per row in VTCM mask buffer
    నిర్మాణం fastdiv_విలువలు     thread_div;
} fa_softmax_వాదనలు_t;

స్థిర అంతఃవరుస శూన్య fa_softmax_impl(
    గుర్తులేని పూర్ణాంకం n, గుర్తులేని పూర్ణాంకం i, శూన్య * దత్తాంశం,
    స్థిరరాశి bool has_ముసుగు,
    స్థిరరాశి bool ముసుగు_ప్రసారం,
    స్థిరరాశి bool is_g1,
    స్థిరరాశి bool has_alibi,
    స్థిరరాశి bool has_softcap
) {
    fa_softmax_వాదనలు_t *     వాదనలు_2  = (fa_softmax_వాదనలు_t *) దత్తాంశం;
    నిర్మాణం hmx_fa_context * factx = వాదనలు_2->factx;

    స్థిరరాశి size_t n_అడ్డువరుసలు_g       = వాదనలు_2->n_అడ్డువరుసలు_g;
    స్థిరరాశి size_t kv_అడ్డువరుసలు        = వాదనలు_2->kv_అడ్డువరుసలు;
    స్థిరరాశి size_t Bc             = వాదనలు_2->Bc;
    స్థిరరాశి size_t G              = వాదనలు_2->G;
    స్థిరరాశి size_t n_tiles_చొప్పున_bc = వాదనలు_2->n_tiles_చొప్పున_bc;
    స్థిరరాశి size_t n_అడ్డువరుస_vec_cnt  = hmx_ceil_div(n_అడ్డువరుసలు_g, 64);
    స్థిరరాశి uint32_t im3          = has_ముసుగు ? fastmodulo(వాదనలు_2->ib3, వాదనలు_2->ముసుగు->ne[3], &factx->src3_div3) : 0;

    size_t vec_ప్రారంభం = 0;
    size_t vec_చివర   = n_అడ్డువరుస_vec_cnt;
    ఒకవేళ (n > 1) {
        స్థిరరాశి size_t vecs_చొప్పున_t = fastdiv(n_అడ్డువరుస_vec_cnt + n - 1, &వాదనలు_2->thread_div);
        vec_ప్రారంభం = i * vecs_చొప్పున_t;
        vec_చివర   = hex_smin(vec_ప్రారంభం + vecs_చొప్పున_t, n_అడ్డువరుస_vec_cnt);
    }

    ఒకవేళ (vec_ప్రారంభం >= n_అడ్డువరుస_vec_cnt) {
        తిరిగి;
    }

    నిర్మాణం htp_thread_trace * tr = &factx->octx->ctx->trace[i];
    htp_trace_event_ప్రారంభం(tr, HTP_TRACE_EVT_HVX_FA_SFM, (uint16_t) (వాదనలు_2->q_ప్రారంభం * G + vec_ప్రారంభం * 64));

    // Per-thread row scratch: thread i uses bufs at offset i * 2 * stride
    స్థిరరాశి size_t అడ్డువరుస_buf_stride = factx->అడ్డువరుస_buf_stride;
    HVX_సదిశ * my_అడ్డువరుస_buf0    = factx->vtcm_అడ్డువరుస_bufs + i * 2 * అడ్డువరుస_buf_stride;
    HVX_సదిశ * my_అడ్డువరుస_buf1    = my_అడ్డువరుస_buf0 + అడ్డువరుస_buf_stride;

    స్థిరరాశి HVX_సదిశ v_neg_inf = Q6_Vh_vsplat_R(0xfbff);

    కొరకు (size_t r_vec_idx = vec_ప్రారంభం; r_vec_idx < vec_చివర; ++r_vec_idx) {
        HVX_సదిశ rowmax_acc_v = v_neg_inf;
        HVX_సదిశ rowsum_acc_v = Q6_V_vzero();
        HVX_సదిశ m_prev_v0    = factx->vtcm_m_vec[r_vec_idx * 2 + 0];
        HVX_సదిశ m_prev_v1    = factx->vtcm_m_vec[r_vec_idx * 2 + 1];

        HVX_సదిశ v_slopes = Q6_V_vzero();
        ఒకవేళ (has_alibi) {
            v_slopes = hvx_vmem(వాదనలు_2->slopes + r_vec_idx * 64);
        }

        కొరకు (uint32_t r_vec_off = 0; r_vec_off < 64; r_vec_off += 2) {
            uint32_t r = r_vec_idx * 64 + r_vec_off;
            ఒకవేళ (r >= hex_align_పైకి(n_అడ్డువరుసలు_g, 2)) {
                ఆపు;
            }

            uint32_t r0 = r / HMX_FP16_TILE_N_అడ్డువరుసలు;
            uint32_t r1 = r % HMX_FP16_TILE_N_అడ్డువరుసలు;

            స్థిరరాశి __fp16 * s_ld_పునాది = factx->vtcm_s_tiles[వాదనలు_2->buf_idx] + r0 * HMX_FP16_TILE_N_అడ్డువరుసలు * Bc;
            __fp16 *       p_st_పునాది = factx->vtcm_p_tiles[వాదనలు_2->buf_idx] + r0 * HMX_FP16_TILE_N_అడ్డువరుసలు * Bc;

            // Decode 2 rows from S tiles into per-thread row buffers
            ఒకవేళ (has_softcap) {
                స్థిరరాశి HVX_సదిశ v_cap = hvx_vec_splat_f16(factx->logit_softcap);
                కొరకు (size_t c = 0; c < kv_అడ్డువరుసలు; c += 64) {
                    size_t             ci       = c / 64;
                    స్థిరరాశి __fp16 *     లోపల_dtile = s_ld_పునాది + ci * HMX_FP16_TILE_N_ELMS * 2;
                    స్థిరరాశి HVX_సదిశ * pv_s_in0 = ((స్థిరరాశి HVX_సదిశ *) లోపల_dtile) + r1 / 2;
                    స్థిరరాశి HVX_సదిశ * pv_s_in1 = pv_s_in0 + 16;

                    HVX_సదిశ_Pair vp_s_drow = Q6_W_vdeal_VVR(*pv_s_in1, *pv_s_in0, -2);
                    HVX_సదిశ     v_s_row0  = Q6_V_lo_W(vp_s_drow);
                    HVX_సదిశ     v_s_row1  = Q6_V_hi_W(vp_s_drow);

                    HVX_సదిశ t0   = hvx_vec_tanh_f16(v_s_row0);
                    my_అడ్డువరుస_buf0[ci] = hvx_vec_mul_f16_f16(t0, v_cap);

                    HVX_సదిశ t1   = hvx_vec_tanh_f16(v_s_row1);
                    my_అడ్డువరుస_buf1[ci] = hvx_vec_mul_f16_f16(t1, v_cap);
                }
            } లేకపోతే {
                size_t c = 0;
                కొరకు (; c + 64 < kv_అడ్డువరుసలు; c += 128) {
                    size_t             ci0       = c / 64;
                    size_t             ci1       = ci0 + 1;
                    స్థిరరాశి __fp16 *     లోపల_dtile0 = s_ld_పునాది + ci0 * HMX_FP16_TILE_N_ELMS * 2;
                    స్థిరరాశి __fp16 *     లోపల_dtile1 = s_ld_పునాది + ci1 * HMX_FP16_TILE_N_ELMS * 2;
                    స్థిరరాశి HVX_సదిశ * pv_s_in0_0 = ((స్థిరరాశి HVX_సదిశ *) లోపల_dtile0) + r1 / 2;
                    స్థిరరాశి HVX_సదిశ * pv_s_in1_0 = pv_s_in0_0 + 16;
                    స్థిరరాశి HVX_సదిశ * pv_s_in0_1 = ((స్థిరరాశి HVX_సదిశ *) లోపల_dtile1) + r1 / 2;
                    స్థిరరాశి HVX_సదిశ * pv_s_in1_1 = pv_s_in0_1 + 16;

                    HVX_సదిశ_Pair vp_s_drow0 = Q6_W_vdeal_VVR(*pv_s_in1_0, *pv_s_in0_0, -2);
                    my_అడ్డువరుస_buf0[ci0]          = Q6_V_lo_W(vp_s_drow0);
                    my_అడ్డువరుస_buf1[ci0]          = Q6_V_hi_W(vp_s_drow0);

                    HVX_సదిశ_Pair vp_s_drow1 = Q6_W_vdeal_VVR(*pv_s_in1_1, *pv_s_in0_1, -2);
                    my_అడ్డువరుస_buf0[ci1]          = Q6_V_lo_W(vp_s_drow1);
                    my_అడ్డువరుస_buf1[ci1]          = Q6_V_hi_W(vp_s_drow1);
                }
                కొరకు (; c < kv_అడ్డువరుసలు; c += 64) {
                    size_t             ci       = c / 64;
                    స్థిరరాశి __fp16 *     లోపల_dtile = s_ld_పునాది + ci * HMX_FP16_TILE_N_ELMS * 2;
                    స్థిరరాశి HVX_సదిశ * pv_s_in0 = ((స్థిరరాశి HVX_సదిశ *) లోపల_dtile) + r1 / 2;
                    స్థిరరాశి HVX_సదిశ * pv_s_in1 = pv_s_in0 + 16;

                    HVX_సదిశ_Pair vp_s_drow = Q6_W_vdeal_VVR(*pv_s_in1, *pv_s_in0, -2);
                    my_అడ్డువరుస_buf0[ci]          = Q6_V_lo_W(vp_s_drow);
                    my_అడ్డువరుస_buf1[ci]          = Q6_V_hi_W(vp_s_drow);
                }
            }

            // Apply mask & compute rowmax(S)
            HVX_సదిశ v_slope0 = Q6_V_vzero();
            HVX_సదిశ v_slope1 = Q6_V_vzero();
            ఒకవేళ (has_alibi) {
                v_slope0 = hvx_vec_repl_f16(Q6_V_vror_VR(v_slopes, r_vec_off * 2));
                v_slope1 = (r + 1 < n_అడ్డువరుసలు_g) ? hvx_vec_repl_f16(Q6_V_vror_VR(v_slopes, (r_vec_off + 1) * 2)) : Q6_V_vzero();
            }

            స్థిరరాశి HVX_సదిశ v_పరిమితి = Q6_Vh_vsplat_R(0xcc00);  // fp16 -16.0

            HVX_సదిశ v_s_rowmax0 = v_neg_inf;
            HVX_సదిశ v_s_rowmax1 = v_neg_inf;
            ఒకవేళ (has_ముసుగు) {
                కొరకు (size_t c = 0; c < kv_అడ్డువరుసలు; c += 64) {
                    size_t         ci          = c / 64;
                    స్థిరరాశి size_t   ne          = hex_smin(kv_అడ్డువరుసలు - c, 64);
                    HVX_సదిశ_Pred q_తోక_keep = Q6_Q_vsetq2_R(ne * పరిమాణం(__fp16));

                    HVX_సదిశ v_mask0, v_mask1;

                    ఒకవేళ (ముసుగు_ప్రసారం) {
                        ఒకవేళ (is_g1) {
                            స్థిరరాశి size_t qi0 = r + 0;
                            v_mask0 = *(స్థిరరాశి HVX_సదిశ *) (వాదనలు_2->ముసుగు_vtcm + qi0 * వాదనలు_2->ముసుగు_vtcm_అడ్డువరుస_stride + c);
                            v_mask1 = v_neg_inf;
                            ఒకవేళ (r + 1 < n_అడ్డువరుసలు_g) {
                                స్థిరరాశి size_t qi1 = r + 1;
                                v_mask1 = *(స్థిరరాశి HVX_సదిశ *) (వాదనలు_2->ముసుగు_vtcm + qi1 * వాదనలు_2->ముసుగు_vtcm_అడ్డువరుస_stride + c);
                            }
                        } లేకపోతే {
                            స్థిరరాశి size_t qi0 = fastdiv(r + 0, &factx->div_G);
                            v_mask0 = *(స్థిరరాశి HVX_సదిశ *) (వాదనలు_2->ముసుగు_vtcm + qi0 * వాదనలు_2->ముసుగు_vtcm_అడ్డువరుస_stride + c);
                            v_mask1 = v_neg_inf;
                            ఒకవేళ (r + 1 < n_అడ్డువరుసలు_g) {
                                స్థిరరాశి size_t qi1 = fastdiv(r + 1, &factx->div_G);
                                ఒకవేళ (qi1 == qi0) {
                                    v_mask1 = v_mask0;
                                } లేకపోతే {
                                    v_mask1 = *(స్థిరరాశి HVX_సదిశ *) (వాదనలు_2->ముసుగు_vtcm + qi1 * వాదనలు_2->ముసుగు_vtcm_అడ్డువరుస_stride + c);
                                }
                            }
                        }
                    } లేకపోతే {
                        // Head-dependent mask: pre-interleaved per row r.
                        స్థిరరాశి size_t r0 = r + 0;
                        v_mask0 = *(స్థిరరాశి HVX_సదిశ *) (వాదనలు_2->ముసుగు_vtcm + r0 * వాదనలు_2->ముసుగు_vtcm_అడ్డువరుస_stride + c);
                        v_mask1 = v_neg_inf;
                        ఒకవేళ (r + 1 < n_అడ్డువరుసలు_g) {
                            స్థిరరాశి size_t r1 = r + 1;
                            v_mask1 = *(స్థిరరాశి HVX_సదిశ *) (వాదనలు_2->ముసుగు_vtcm + r1 * వాదనలు_2->ముసుగు_vtcm_అడ్డువరుస_stride + c);
                        }
                    }

                    // Threshold: mask values below -16.0 are treated as -inf (causal mask).
                    HVX_సదిశ_Pred q_keep0 = Q6_Q_and_QQ(Q6_Q_vcmp_gt_VhfVhf(v_mask0, v_పరిమితి), q_తోక_keep);
                    HVX_సదిశ_Pred q_keep1 = Q6_Q_and_QQ(Q6_Q_vcmp_gt_VhfVhf(v_mask1, v_పరిమితి), q_తోక_keep);

                    // Scale mask values by log2(e) for base-2 calculations
                    స్థిరరాశి HVX_సదిశ v_log2e = hvx_vec_splat_f16(EXP_LOG2E_F);
                    HVX_సదిశ v_mask0_scaled = hvx_vec_mul_f16_f16(v_mask0, v_log2e);
                    HVX_సదిశ v_mask1_scaled = hvx_vec_mul_f16_f16(v_mask1, v_log2e);

                    ఒకవేళ (has_alibi) {
                        HVX_సదిశ v_sm0 = hvx_vec_mul_f16_f16(v_mask0_scaled, v_slope0);
                        HVX_సదిశ v_sm1 = hvx_vec_mul_f16_f16(v_mask1_scaled, v_slope1);
                        my_అడ్డువరుస_buf0[ci]  = Q6_V_vmux_QVV(q_keep0, hvx_vec_కూడు_f16_f16(my_అడ్డువరుస_buf0[ci], v_sm0), v_neg_inf);
                        my_అడ్డువరుస_buf1[ci]  = Q6_V_vmux_QVV(q_keep1, hvx_vec_కూడు_f16_f16(my_అడ్డువరుస_buf1[ci], v_sm1), v_neg_inf);
                    } లేకపోతే {
                        my_అడ్డువరుస_buf0[ci] = Q6_V_vmux_QVV(q_keep0, hvx_vec_కూడు_f16_f16(my_అడ్డువరుస_buf0[ci], v_mask0_scaled), v_neg_inf);
                        my_అడ్డువరుస_buf1[ci] = Q6_V_vmux_QVV(q_keep1, hvx_vec_కూడు_f16_f16(my_అడ్డువరుస_buf1[ci], v_mask1_scaled), v_neg_inf);
                    }

                    v_s_rowmax0 = Q6_Vhf_vmax_VhfVhf(v_s_rowmax0, my_అడ్డువరుస_buf0[ci]);
                    v_s_rowmax1 = Q6_Vhf_vmax_VhfVhf(v_s_rowmax1, my_అడ్డువరుస_buf1[ci]);
                }
            } లేకపోతే {
                size_t c = 0;
                కొరకు (; c + 64 < kv_అడ్డువరుసలు; c += 128) {
                    size_t ci0 = c / 64;
                    size_t ci1 = ci0 + 1;
                    v_s_rowmax0 = Q6_Vhf_vmax_VhfVhf(v_s_rowmax0, my_అడ్డువరుస_buf0[ci0]);
                    v_s_rowmax1 = Q6_Vhf_vmax_VhfVhf(v_s_rowmax1, my_అడ్డువరుస_buf1[ci0]);
                    v_s_rowmax0 = Q6_Vhf_vmax_VhfVhf(v_s_rowmax0, my_అడ్డువరుస_buf0[ci1]);
                    v_s_rowmax1 = Q6_Vhf_vmax_VhfVhf(v_s_rowmax1, my_అడ్డువరుస_buf1[ci1]);
                }
                కొరకు (; c < kv_అడ్డువరుసలు; c += 64) {
                    size_t         ci          = c / 64;
                    స్థిరరాశి size_t   ne          = hex_smin(kv_అడ్డువరుసలు - c, 64);
                    HVX_సదిశ_Pred q_తోక_keep = Q6_Q_vsetq2_R(ne * పరిమాణం(__fp16));
                    ఒకవేళ (ne < 64) {
                        my_అడ్డువరుస_buf0[ci] = Q6_V_vmux_QVV(q_తోక_keep, my_అడ్డువరుస_buf0[ci], v_neg_inf);
                        my_అడ్డువరుస_buf1[ci] = Q6_V_vmux_QVV(q_తోక_keep, my_అడ్డువరుస_buf1[ci], v_neg_inf);
                    }
                    v_s_rowmax0 = Q6_Vhf_vmax_VhfVhf(v_s_rowmax0, my_అడ్డువరుస_buf0[ci]);
                    v_s_rowmax1 = Q6_Vhf_vmax_VhfVhf(v_s_rowmax1, my_అడ్డువరుస_buf1[ci]);
                }
            }

            v_s_rowmax0 = hvx_vec_reduce_max_f16(v_s_rowmax0);
            v_s_rowmax1 = hvx_vec_reduce_max_f16(v_s_rowmax1);

            // Splat m_prev[r], m_prev[r+1] from the float per-row accumulators and convert to fp16 vectors
            HVX_సదిశ v_m_prev0, v_m_prev1;
            ఒకవేళ (r_vec_off < 32) {
                HVX_సదిశ v0 = hvx_vec_repl_f32(Q6_V_vror_VR(m_prev_v0, r_vec_off * 4));
                v_m_prev0 = hvx_vec_f32_గమ్యం_f16(v0, v0);
                ఒకవేళ (r + 1 < n_అడ్డువరుసలు_g) {
                    HVX_సదిశ v1 = hvx_vec_repl_f32(Q6_V_vror_VR(m_prev_v0, (r_vec_off + 1) * 4));
                    v_m_prev1 = hvx_vec_f32_గమ్యం_f16(v1, v1);
                } లేకపోతే {
                    v_m_prev1 = Q6_V_vzero();
                }
            } లేకపోతే {
                HVX_సదిశ v0 = hvx_vec_repl_f32(Q6_V_vror_VR(m_prev_v1, (r_vec_off - 32) * 4));
                v_m_prev0 = hvx_vec_f32_గమ్యం_f16(v0, v0);
                ఒకవేళ (r + 1 < n_అడ్డువరుసలు_g) {
                    HVX_సదిశ v1 = hvx_vec_repl_f32(Q6_V_vror_VR(m_prev_v1, (r_vec_off + 1 - 32) * 4));
                    v_m_prev1 = hvx_vec_f32_గమ్యం_f16(v1, v1);
                } లేకపోతే {
                    v_m_prev1 = Q6_V_vzero();
                }
            }

            HVX_సదిశ v_dup_m0 = Q6_Vhf_vmax_VhfVhf(v_m_prev0, v_s_rowmax0);
            HVX_సదిశ v_dup_m1 = Q6_Vhf_vmax_VhfVhf(v_m_prev1, v_s_rowmax1);

            // Insert row r, r+1 rowmax into rowmax_acc_v
            {
                HVX_సదిశ_Pred p_ప్రారంభం = Q6_Q_vsetq_R(r_vec_off * 2);
                HVX_సదిశ_Pred p_mid   = Q6_Q_vsetq_R((r_vec_off + 1) * 2);
                HVX_సదిశ_Pred p_చివర   = Q6_Q_vsetq2_R((r_vec_off + 2) * 2);
                HVX_సదిశ_Pred p_lane0 = Q6_Q_and_QQn(p_mid, p_ప్రారంభం);
                HVX_సదిశ_Pred p_lane1 = Q6_Q_and_QQn(p_చివర, p_mid);
                rowmax_acc_v           = Q6_V_vmux_QVV(p_lane0, v_dup_m0, rowmax_acc_v);
                rowmax_acc_v           = Q6_V_vmux_QVV(p_lane1, v_dup_m1, rowmax_acc_v);
            }

            // Compute P = exp(S - m_new)
            స్థిరరాశి HVX_సదిశ v_zero      = Q6_V_vzero();
            HVX_సదిశ       v_p_rowsum0 = v_zero;
            HVX_సదిశ       v_p_rowsum1 = v_zero;

            size_t c = 0;
            కొరకు (; c + 64 < kv_అడ్డువరుసలు; c += 128) {
                size_t     ci0          = c / 64;
                size_t     ci1          = ci0 + 1;

                HVX_సదిశ v_s_minus_m0_0 = Q6_Vqf16_vsub_VhfVhf(my_అడ్డువరుస_buf0[ci0], v_dup_m0);
                HVX_సదిశ v_s_minus_m1_0 = Q6_Vqf16_vsub_VhfVhf(my_అడ్డువరుస_buf1[ci0], v_dup_m1);
                HVX_సదిశ v_s_minus_m0_1 = Q6_Vqf16_vsub_VhfVhf(my_అడ్డువరుస_buf0[ci1], v_dup_m0);
                HVX_సదిశ v_s_minus_m1_1 = Q6_Vqf16_vsub_VhfVhf(my_అడ్డువరుస_buf1[ci1], v_dup_m1);

                HVX_సదిశ v_p_row0_hf_0  = hvx_vec_exp2_f16(Q6_Vhf_equals_Vqf16(v_s_minus_m0_0));
                HVX_సదిశ v_p_row1_hf_0  = hvx_vec_exp2_f16(Q6_Vhf_equals_Vqf16(v_s_minus_m1_0));
                HVX_సదిశ v_p_row0_hf_1  = hvx_vec_exp2_f16(Q6_Vhf_equals_Vqf16(v_s_minus_m0_1));
                HVX_సదిశ v_p_row1_hf_1  = hvx_vec_exp2_f16(Q6_Vhf_equals_Vqf16(v_s_minus_m1_1));

                __fp16 *     బహిర్గతం_dtile0  = p_st_పునాది + ci0 * HMX_FP16_TILE_N_ELMS * 2;
                __fp16 *     బహిర్గతం_dtile1  = p_st_పునాది + ci1 * HMX_FP16_TILE_N_ELMS * 2;
                HVX_సదిశ * pv_p_out0_0  = ((HVX_సదిశ *) బహిర్గతం_dtile0) + r1 / 2;
                HVX_సదిశ * pv_p_out1_0  = pv_p_out0_0 + 16;
                HVX_సదిశ * pv_p_out0_1  = ((HVX_సదిశ *) బహిర్గతం_dtile1) + r1 / 2;
                HVX_సదిశ * pv_p_out1_1  = pv_p_out0_1 + 16;

                HVX_సదిశ_Pair vp_p_dual0 = Q6_W_vshuff_VVR(v_p_row1_hf_0, v_p_row0_hf_0, -2);
                *pv_p_out0_0               = Q6_V_lo_W(vp_p_dual0);
                *pv_p_out1_0               = Q6_V_hi_W(vp_p_dual0);

                HVX_సదిశ_Pair vp_p_dual1 = Q6_W_vshuff_VVR(v_p_row1_hf_1, v_p_row0_hf_1, -2);
                *pv_p_out0_1               = Q6_V_lo_W(vp_p_dual1);
                *pv_p_out1_1               = Q6_V_hi_W(vp_p_dual1);

                HVX_సదిశ_Pair vp_p0_0 = hvx_vec_f16_గమ్యం_f32_shuff(v_p_row0_hf_0);
                HVX_సదిశ_Pair vp_p1_0 = hvx_vec_f16_గమ్యం_f32_shuff(v_p_row1_hf_0);
                HVX_సదిశ_Pair vp_p0_1 = hvx_vec_f16_గమ్యం_f32_shuff(v_p_row0_hf_1);
                HVX_సదిశ_Pair vp_p1_1 = hvx_vec_f16_గమ్యం_f32_shuff(v_p_row1_hf_1);

                v_p_rowsum0 = Q6_Vqf32_vadd_Vqf32Vqf32(v_p_rowsum0, Q6_Vqf32_vadd_VsfVsf(Q6_V_lo_W(vp_p0_0), Q6_V_hi_W(vp_p0_0)));
                v_p_rowsum0 = Q6_Vqf32_vadd_Vqf32Vqf32(v_p_rowsum0, Q6_Vqf32_vadd_VsfVsf(Q6_V_lo_W(vp_p0_1), Q6_V_hi_W(vp_p0_1)));
                v_p_rowsum1 = Q6_Vqf32_vadd_Vqf32Vqf32(v_p_rowsum1, Q6_Vqf32_vadd_VsfVsf(Q6_V_lo_W(vp_p1_0), Q6_V_hi_W(vp_p1_0)));
                v_p_rowsum1 = Q6_Vqf32_vadd_Vqf32Vqf32(v_p_rowsum1, Q6_Vqf32_vadd_VsfVsf(Q6_V_lo_W(vp_p1_1), Q6_V_hi_W(vp_p1_1)));
            }
            కొరకు (size_t c_rem = c; c_rem < kv_అడ్డువరుసలు; c_rem += 64) {
                size_t     ci           = c_rem / 64;
                HVX_సదిశ v_s_minus_m0 = Q6_Vqf16_vsub_VhfVhf(my_అడ్డువరుస_buf0[ci], v_dup_m0);
                HVX_సదిశ v_s_minus_m1 = Q6_Vqf16_vsub_VhfVhf(my_అడ్డువరుస_buf1[ci], v_dup_m1);

                HVX_సదిశ v_p_row0_hf  = hvx_vec_exp2_f16(Q6_Vhf_equals_Vqf16(v_s_minus_m0));
                HVX_సదిశ v_p_row1_hf  = hvx_vec_exp2_f16(Q6_Vhf_equals_Vqf16(v_s_minus_m1));
                __fp16 *     బహిర్గతం_dtile  = p_st_పునాది + ci * HMX_FP16_TILE_N_ELMS * 2;
                HVX_సదిశ * pv_p_out0  = ((HVX_సదిశ *) బహిర్గతం_dtile) + r1 / 2;
                HVX_సదిశ * pv_p_out1  = pv_p_out0 + 16;

                HVX_సదిశ_Pair vp_p_dual = Q6_W_vshuff_VVR(v_p_row1_hf, v_p_row0_hf, -2);
                *pv_p_out0               = Q6_V_lo_W(vp_p_dual);
                *pv_p_out1               = Q6_V_hi_W(vp_p_dual);

                HVX_సదిశ_Pair vp_p0 = hvx_vec_f16_గమ్యం_f32_shuff(v_p_row0_hf);
                HVX_సదిశ_Pair vp_p1 = hvx_vec_f16_గమ్యం_f32_shuff(v_p_row1_hf);

                v_p_rowsum0 = Q6_Vqf32_vadd_Vqf32Vqf32(v_p_rowsum0, Q6_Vqf32_vadd_VsfVsf(Q6_V_lo_W(vp_p0), Q6_V_hi_W(vp_p0)));
                v_p_rowsum1 = Q6_Vqf32_vadd_Vqf32Vqf32(v_p_rowsum1, Q6_Vqf32_vadd_VsfVsf(Q6_V_lo_W(vp_p1), Q6_V_hi_W(vp_p1)));
            }

            HVX_సదిశ rowsum0_sf = hvx_vec_reduce_కూడిక_f32(Q6_Vsf_equals_Vqf32(v_p_rowsum0));
            HVX_సదిశ rowsum1_sf = hvx_vec_reduce_కూడిక_f32(Q6_Vsf_equals_Vqf32(v_p_rowsum1));
            {
                HVX_సదిశ rv0_v = hvx_vec_f32_గమ్యం_f16(rowsum0_sf, rowsum0_sf);
                HVX_సదిశ rv1_v = hvx_vec_f32_గమ్యం_f16(rowsum1_sf, rowsum1_sf);

                HVX_సదిశ_Pred p_ప్రారంభం = Q6_Q_vsetq_R(r_vec_off * 2);
                HVX_సదిశ_Pred p_mid   = Q6_Q_vsetq_R((r_vec_off + 1) * 2);
                HVX_సదిశ_Pred p_చివర   = Q6_Q_vsetq2_R((r_vec_off + 2) * 2);
                HVX_సదిశ_Pred p_lane0 = Q6_Q_and_QQn(p_mid, p_ప్రారంభం);
                HVX_సదిశ_Pred p_lane1 = Q6_Q_and_QQn(p_చివర, p_mid);
                rowsum_acc_v           = Q6_V_vmux_QVV(p_lane0, rv0_v, rowsum_acc_v);
                rowsum_acc_v           = Q6_V_vmux_QVV(p_lane1, rv1_v, rowsum_acc_v);
            }
        }

        // Inline fa_ml_update_and_build_d for this vector (lock-free and in parallel)
        HVX_సదిశ_Pair rowmax_acc_pair    = hvx_vec_f16_గమ్యం_f32(rowmax_acc_v);
        HVX_సదిశ     v_rowmax_acc_f32_0 = Q6_V_lo_W(rowmax_acc_pair);
        HVX_సదిశ     v_rowmax_acc_f32_1 = Q6_V_hi_W(rowmax_acc_pair);

        HVX_సదిశ v_m_curr0 = Q6_Vsf_vmax_VsfVsf(m_prev_v0, v_rowmax_acc_f32_0);
        HVX_సదిశ v_m_curr1 = Q6_Vsf_vmax_VsfVsf(m_prev_v1, v_rowmax_acc_f32_1);

        HVX_సదిశ v_m_diff0 = HVX_OP_SUB_F32(m_prev_v0, v_m_curr0);
        HVX_సదిశ v_m_diff1 = HVX_OP_SUB_F32(m_prev_v1, v_m_curr1);

        HVX_సదిశ v_m_diff_f16   = hvx_vec_f32_గమ్యం_f16(v_m_diff0, v_m_diff1);
        HVX_సదిశ exp_m_diff_f16 = hvx_vec_exp2_f16(v_m_diff_f16);

        HVX_సదిశ_Pair exp_m_diff_pair = hvx_vec_f16_గమ్యం_f32(exp_m_diff_f16);
        HVX_సదిశ exp_m_diff0 = Q6_V_lo_W(exp_m_diff_pair);
        HVX_సదిశ exp_m_diff1 = Q6_V_hi_W(exp_m_diff_pair);

        HVX_సదిశ_Pair rowsum_acc_pair = hvx_vec_f16_గమ్యం_f32(rowsum_acc_v);
        HVX_సదిశ     v_rowsum_acc_f32_0 = Q6_V_lo_W(rowsum_acc_pair);
        HVX_సదిశ     v_rowsum_acc_f32_1 = Q6_V_hi_W(rowsum_acc_pair);

        HVX_సదిశ l_prev_v0 = factx->vtcm_l_vec[r_vec_idx * 2 + 0];
        HVX_సదిశ l_prev_v1 = factx->vtcm_l_vec[r_vec_idx * 2 + 1];
        HVX_సదిశ v_l_curr0 = HVX_OP_కూడు_F32(HVX_OP_MUL_F32(l_prev_v0, exp_m_diff0), v_rowsum_acc_f32_0);
        HVX_సదిశ v_l_curr1 = HVX_OP_కూడు_F32(HVX_OP_MUL_F32(l_prev_v1, exp_m_diff1), v_rowsum_acc_f32_1);

        factx->vtcm_m_vec[r_vec_idx * 2 + 0] = v_m_curr0;
        factx->vtcm_m_vec[r_vec_idx * 2 + 1] = v_m_curr1;
        factx->vtcm_l_vec[r_vec_idx * 2 + 0] = v_l_curr0;
        factx->vtcm_l_vec[r_vec_idx * 2 + 1] = v_l_curr1;

        // Build diagonal tile D = diag(exp(m_diff))
        స్థిరరాశి HVX_సదిశ     v_offsets = *(స్థిరరాశి HVX_సదిశ *) d_tile_scatter_offsets;
        స్థిరరాశి HVX_సదిశ_Pred q_32_ముసుగు = Q6_Q_vsetq_R(32 * పరిమాణం(__fp16));
        HVX_సదిశ           v_exp_m_diff = exp_m_diff_f16;

        __fp16 * స్థిరరాశి d_tiles_బహిర్గతం = factx->vtcm_d_tiles[వాదనలు_2->buf_idx];

        size_t t0 = r_vec_idx * 2;
        ఒకవేళ (t0 < వాదనలు_2->n_అడ్డువరుస_tiles) {
            స్థిరరాశి HVX_సదిశ v_content = v_exp_m_diff;
            __fp16 *         బహిర్గతం_పునాది  = d_tiles_బహిర్గతం + t0 * HMX_FP16_TILE_N_ELMS;
            Q6_vscatter_QRMVhV(q_32_ముసుగు, (size_t) బహిర్గతం_పునాది, HMX_FP16_TILE_కొలత - 1, v_offsets, v_content);
        }

        size_t t1 = r_vec_idx * 2 + 1;
        ఒకవేళ (t1 < వాదనలు_2->n_అడ్డువరుస_tiles) {
            స్థిరరాశి HVX_సదిశ v_content = Q6_V_vror_VR(v_exp_m_diff, 64);
            __fp16 *         బహిర్గతం_పునాది  = d_tiles_బహిర్గతం + t1 * HMX_FP16_TILE_N_ELMS;
            Q6_vscatter_QRMVhV(q_32_ముసుగు, (size_t) బహిర్గతం_పునాది, HMX_FP16_TILE_కొలత - 1, v_offsets, v_content);
        }
    }
    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_FA_SFM, (uint16_t) (వాదనలు_2->q_ప్రారంభం * G + vec_ప్రారంభం * 64));
}

స్థిర శూన్య fa_softmax_thread_nomask(గుర్తులేని పూర్ణాంకం n, గుర్తులేని పూర్ణాంకం i, శూన్య * దత్తాంశం) {
    fa_softmax_impl(n, i, దత్తాంశం,
                    /*has_mask=*/false,
                    /*mask_broadcast=*/false,
                    /*is_g1=*/false,
                    /*has_alibi=*/false,
                    /*has_softcap=*/false);
}

స్థిర శూన్య fa_softmax_thread_ముసుగు_ప్రసారం_g1(గుర్తులేని పూర్ణాంకం n, గుర్తులేని పూర్ణాంకం i, శూన్య * దత్తాంశం) {
    fa_softmax_impl(n, i, దత్తాంశం,
                    /*has_mask=*/true,
                    /*mask_broadcast=*/true,
                    /*is_g1=*/true,
                    /*has_alibi=*/false,
                    /*has_softcap=*/false);
}

స్థిర శూన్య fa_softmax_thread_ముసుగు_ప్రసారం_gn(గుర్తులేని పూర్ణాంకం n, గుర్తులేని పూర్ణాంకం i, శూన్య * దత్తాంశం) {
    fa_softmax_impl(n, i, దత్తాంశం,
                    /*has_mask=*/true,
                    /*mask_broadcast=*/true,
                    /*is_g1=*/false,
                    /*has_alibi=*/false,
                    /*has_softcap=*/false);
}

స్థిర శూన్య fa_softmax_thread(గుర్తులేని పూర్ణాంకం n, గుర్తులేని పూర్ణాంకం i, శూన్య * దత్తాంశం) {
    fa_softmax_వాదనలు_t *     వాదనలు_2  = (fa_softmax_వాదనలు_t *) దత్తాంశం;
    నిర్మాణం hmx_fa_context * factx = వాదనలు_2->factx;

    స్థిరరాశి bool has_ముసుగు       = (వాదనలు_2->ముసుగు != ఖాళీ);
    స్థిరరాశి bool ముసుగు_ప్రసారం = factx->ముసుగు_ప్రసారం;
    స్థిరరాశి bool is_g1          = (వాదనలు_2->G == 1);
    స్థిరరాశి bool has_alibi      = వాదనలు_2->has_alibi;
    స్థిరరాశి bool has_softcap    = factx->has_softcap;

    fa_softmax_impl(n, i, దత్తాంశం, has_ముసుగు, ముసుగు_ప్రసారం, is_g1, has_alibi, has_softcap);
}

స్థిర అంతఃవరుస HVX_సదిశ hvx_fa_ఎక్కించు_sinks(స్థిరరాశి దశాంశం * sinks_దత్తాంశం, size_t idx, size_t లెక్క) {
    స్థిరరాశి HVX_సదిశ * v_src    = (స్థిరరాశి HVX_సదిశ *) sinks_దత్తాంశం + (idx >> 5);
    స్థిరరాశి size_t       lane_off = idx & 31;
    స్థిరరాశి size_t       byte_off = lane_off * పరిమాణం(దశాంశం);
    తిరిగి (lane_off + లెక్క > 32)
        ? Q6_V_valign_VVR(v_src[1], v_src[0], byte_off)
        : Q6_V_vror_VR(v_src[0], byte_off);
}

స్థిర __attribute__((noinline)) శూన్య fa_build_d_diag_inv_l(నిర్మాణం hmx_fa_context * factx,
                                                            size_t                  n_అడ్డువరుస_tiles,
                                                            size_t                  n_అడ్డువరుస_tiles_g_br,
                                                            uint32_t                kv_తల,
                                                            size_t                  n_అడ్డువరుసలు_g) {
    (శూన్య) n_అడ్డువరుస_tiles_g_br;

    // sinks: l += exp(sink - m * ln2) in f32, m and l are in the log2 domain
    ఒకవేళ (factx->sinks) {
        స్థిరరాశి దశాంశం *    sinks_దత్తాంశం   = factx->vtcm_sinks;
        స్థిరరాశి size_t     G            = factx->G;
        స్థిరరాశి size_t     తల_పునాది    = (size_t) kv_తల * G;
        స్థిరరాశి HVX_సదిశ v_ln2        = Q6_V_vsplat_R(EXP_LOGN2);
        స్థిరరాశి HVX_సదిశ v_తల_sinks = (G <= 32) ? hvx_fa_ఎక్కించు_sinks(sinks_దత్తాంశం, తల_పునాది, G) : Q6_V_vzero();
        HVX_సదిశ       v_sink       = Q6_V_vzero();
        size_t           g_prev       = SIZE_MAX;

        కొరకు (size_t i = 0; i < n_అడ్డువరుస_tiles; ++i) {
            // the sink pattern of a tile only depends on (i * 32) % G, rebuild it when that changes
            స్థిరరాశి size_t r0 = i * 32;
            స్థిరరాశి size_t g0 = fastmodulo(r0, G, &factx->div_G);
            ఒకవేళ (g0 != g_prev) {
                స్థిరరాశి size_t rem_g = G - g0;
                ఒకవేళ (G <= 32) {
                    స్థిరరాశి HVX_సదిశ v_lo = Q6_V_vror_VR(v_తల_sinks, g0 * పరిమాణం(దశాంశం));
                    స్థిరరాశి HVX_సదిశ v_hi = Q6_V_vror_VR(v_తల_sinks, 128 - rem_g * పరిమాణం(దశాంశం));
                    v_sink = Q6_V_vmux_QVV(Q6_Q_vsetq2_R(rem_g * పరిమాణం(దశాంశం)), v_lo, v_hi);
                    కొరకు (size_t len = G; len < 32; len <<= 1) {
                        స్థిరరాశి HVX_సదిశ_Pred q_keep    = Q6_Q_vsetq_R(len * పరిమాణం(దశాంశం));
                        స్థిరరాశి HVX_సదిశ     v_shifted = Q6_V_vror_VR(v_sink, 128 - len * పరిమాణం(దశాంశం));
                        v_sink = Q6_V_vmux_QVV(q_keep, v_sink, v_shifted);
                    }
                } లేకపోతే ఒకవేళ (rem_g >= 32) {
                    v_sink = hvx_fa_ఎక్కించు_sinks(sinks_దత్తాంశం, తల_పునాది + g0, 32);
                } లేకపోతే {
                    స్థిరరాశి HVX_సదిశ v_lo = hvx_fa_ఎక్కించు_sinks(sinks_దత్తాంశం, తల_పునాది + g0, rem_g);
                    స్థిరరాశి HVX_సదిశ v_hi = hvx_fa_ఎక్కించు_sinks(sinks_దత్తాంశం, తల_పునాది, 32 - rem_g);
                    v_sink = Q6_V_vmux_QVV(Q6_Q_vsetq_R(rem_g * పరిమాణం(దశాంశం)), v_lo, Q6_V_vror_VR(v_hi, 128 - rem_g * పరిమాణం(దశాంశం)));
                }
                g_prev = g0;
            }

            HVX_సదిశ v_d = HVX_OP_SUB_F32(v_sink, HVX_OP_MUL_F32(factx->vtcm_m_vec[i], v_ln2));
            ఒకవేళ (r0 + 32 > n_అడ్డువరుసలు_g) {
                స్థిరరాశి HVX_సదిశ_Pred q_valid = Q6_Q_vsetq_R((n_అడ్డువరుసలు_g - r0) * పరిమాణం(దశాంశం));
                v_d = Q6_V_vmux_QVV(q_valid, v_d, Q6_V_vzero());
            }
            factx->vtcm_l_vec[i] = HVX_OP_కూడు_F32(factx->vtcm_l_vec[i], hvx_vec_exp_f32(v_d));
        }
    }

    స్థిరరాశి HVX_సదిశ     v_offsets = *(స్థిరరాశి HVX_సదిశ *) d_tile_scatter_offsets;
    స్థిరరాశి HVX_సదిశ_Pred q_32_ముసుగు = Q6_Q_vsetq_R(32 * పరిమాణం(__fp16));
    స్థిరరాశి HVX_సదిశ     one       = hvx_vec_splat_f32(1.0f);

    HVX_సదిశ v_content = Q6_V_vzero();
    కొరకు (size_t i = 0; i < n_అడ్డువరుస_tiles; ++i) {
        ఒకవేళ ((i % 2) == 0) {
            HVX_సదిశ inv_lo = HVX_OP_MUL_F32(one, hvx_vec_inverse_f32(factx->vtcm_l_vec[i]));
            HVX_సదిశ inv_hi = (i + 1 < n_అడ్డువరుస_tiles) ? HVX_OP_MUL_F32(one, hvx_vec_inverse_f32(factx->vtcm_l_vec[i + 1])) : Q6_V_vzero();
            v_content = hvx_vec_f32_గమ్యం_f16(inv_lo, inv_hi);
        } లేకపోతే {
            v_content = Q6_V_vror_VR(v_content, 64);
        }

        __fp16 * బహిర్గతం_పునాది = factx->vtcm_d_inv_l + i * HMX_FP16_TILE_N_ELMS;
        Q6_vscatter_QRMVhV(q_32_ముసుగు, (size_t) బహిర్గతం_పునాది, HMX_FP16_TILE_కొలత - 1, v_offsets, v_content);
    }
}

స్థిర శూన్య fa_phase_softmax_and_build_d(నిర్మాణం hmx_fa_context * factx,
                                         fa_softmax_వాదనలు_t *     sargs,
                                         size_t                  n_అడ్డువరుస_tiles,
                                         size_t                  n_అడ్డువరుస_tiles_g_br) {
    work_వరుస_t wp = factx->octx->ctx->work_వరుస;
    స్థిరరాశి size_t n_అడ్డువరుస_vec_cnt = hmx_ceil_div(sargs->n_అడ్డువరుసలు_g, 64);

    కార్మికుడు_callback_t softmax_fn = fa_softmax_thread;
    ఒకవేళ (sargs->ముసుగు == ఖాళీ && !factx->has_softcap && !sargs->has_alibi) {
        softmax_fn = fa_softmax_thread_nomask;
    } లేకపోతే ఒకవేళ (sargs->ముసుగు != ఖాళీ && factx->ముసుగు_ప్రసారం && !factx->has_softcap && !sargs->has_alibi) {
        ఒకవేళ (sargs->G == 1) {
            softmax_fn = fa_softmax_thread_ముసుగు_ప్రసారం_g1;
        } లేకపోతే {
            softmax_fn = fa_softmax_thread_ముసుగు_ప్రసారం_gn;
        }
    }

    ఒకవేళ (factx->n_దారాలు > 1 && n_అడ్డువరుస_vec_cnt >= 2) {
        uint32_t n_use = (uint32_t) hex_smin((size_t) factx->n_దారాలు, n_అడ్డువరుస_vec_cnt);
        sargs->thread_div = init_fastdiv_విలువలు(n_use);
        work_వరుస_నిరంతరం(wp, softmax_fn, sargs, n_use);
    } లేకపోతే {
        softmax_fn(1, 0, sargs);
    }
}

// ============================================================================
// HMX job structs and worker functions
// ============================================================================

రకనిర్వచనం నిర్మాణం {
    స్థిరరాశి __fp16 * q_tiles;
    స్థిరరాశి __fp16 * k_tiles;
    __fp16 *       s_tiles;
    size_t         n_అడ్డువరుస_tiles;
    size_t         n_నిలువు_tiles;
    size_t         n_dot_tiles;  // DK / 32
    size_t         n_tiles_చొప్పున_bc;
    uint8_t *      hmx_scales;
} hmx_fa_qk_పని_t;

స్థిర శూన్య hmx_fa_qk_dot_కార్మికుడు(శూన్య * దత్తాంశం) {
    hmx_fa_qk_పని_t * పని            = (hmx_fa_qk_పని_t *) దత్తాంశం;
    స్థిరరాశి size_t      n_అడ్డువరుస_tiles    = పని->n_అడ్డువరుస_tiles;
    స్థిరరాశి size_t      n_నిలువు_tiles    = పని->n_నిలువు_tiles;
    స్థిరరాశి size_t      n_dot_tiles    = పని->n_dot_tiles;
    స్థిరరాశి size_t      n_tiles_చొప్పున_bc = పని->n_tiles_చొప్పున_bc;
    స్థిరరాశి __fp16 * పరిమిత q_tiles  = పని->q_tiles;
    స్థిరరాశి __fp16 * పరిమిత k_tiles  = పని->k_tiles;
    __fp16 * పరిమిత s_tiles        = పని->s_tiles;
    __builtin_assume(n_అడ్డువరుస_tiles > 0);
    __builtin_assume(n_నిలువు_tiles > 0);
    __builtin_assume(n_dot_tiles > 0);

    asm అస్థిర(HMX_సమితి_BIAS("%0") :: "r"((గుర్తులేని పూర్ణాంకం)పని->hmx_scales));
    స్థిరరాశి size_t dot_stride = n_dot_tiles * HMX_FP16_TILE_N_ELMS;
    కొరకు (size_t r = 0; r < n_అడ్డువరుస_tiles; ++r) {
        స్థిరరాశి __fp16 * అడ్డువరుస_tiles = q_tiles + r * dot_stride;
        స్థిరరాశి __fp16 * నిలువు_tiles = k_tiles;
        __fp16 *       బహిర్గతం_tile  = s_tiles + r * n_tiles_చొప్పున_bc * HMX_FP16_TILE_N_ELMS;

        కొరకు (size_t c = 0; c < n_నిలువు_tiles; ++c) {
            hmx_fa_qk_dot_tile(అడ్డువరుస_tiles, నిలువు_tiles, బహిర్గతం_tile, n_dot_tiles);
            నిలువు_tiles += dot_stride;
            బహిర్గతం_tile  += HMX_FP16_TILE_N_ELMS;
        }
    }
}

రకనిర్వచనం నిర్మాణం {
    __fp16 *       o_curr;
    స్థిరరాశి __fp16 * o_prev;
    స్థిరరాశి __fp16 * p_tiles;
    స్థిరరాశి __fp16 * v_tiles;
    స్థిరరాశి __fp16 * d_tiles;
    uint8_t *      hmx_scales;
    size_t         n_అడ్డువరుస_tiles;
    size_t         n_నిలువు_tiles;
    size_t         n_అడ్డువరుస_tiles_g_br;
    size_t         n_tiles_చొప్పున_bc;
    size_t         DV;
} hmx_fa_o_update_పని_t;

స్థిర శూన్య hmx_fa_o_update_కార్మికుడు(శూన్య * దత్తాంశం) {
    hmx_fa_o_update_పని_t * పని              = (hmx_fa_o_update_పని_t *) దత్తాంశం;
    స్థిరరాశి size_t            n_అడ్డువరుస_tiles      = పని->n_అడ్డువరుస_tiles;
    స్థిరరాశి size_t            n_నిలువు_tiles      = పని->n_నిలువు_tiles;
    స్థిరరాశి size_t            n_అడ్డువరుస_tiles_g_br = పని->n_అడ్డువరుస_tiles_g_br;
    స్థిరరాశి size_t            n_tiles_చొప్పున_bc   = పని->n_tiles_చొప్పున_bc;
    స్థిరరాశి size_t            DV_tiles         = పని->DV / 32;
    స్థిరరాశి __fp16 * పరిమిత d_tiles          = పని->d_tiles;
    స్థిరరాశి __fp16 * పరిమిత p_tiles          = పని->p_tiles;
    స్థిరరాశి __fp16 * పరిమిత v_tiles          = పని->v_tiles;
    స్థిరరాశి __fp16 * పరిమిత o_prev           = పని->o_prev;
    __fp16 * పరిమిత o_curr                 = పని->o_curr;
    __builtin_assume(n_అడ్డువరుస_tiles > 0);
    __builtin_assume(n_నిలువు_tiles > 0);
    __builtin_assume(DV_tiles > 0);

    asm అస్థిర(HMX_సమితి_BIAS("%0") :: "r"((గుర్తులేని పూర్ణాంకం)పని->hmx_scales));
    స్థిరరాశి size_t o_stride = n_అడ్డువరుస_tiles_g_br * HMX_FP16_TILE_N_ELMS;
    స్థిరరాశి size_t v_stride = n_tiles_చొప్పున_bc * HMX_FP16_TILE_N_ELMS;
    కొరకు (size_t r = 0; r < n_అడ్డువరుస_tiles; ++r) {
        స్థిరరాశి __fp16 * d_diag     = d_tiles + r * HMX_FP16_TILE_N_ELMS;
        స్థిరరాశి __fp16 * p_tile_లోపల  = p_tiles + (r * n_tiles_చొప్పున_bc) * HMX_FP16_TILE_N_ELMS;
        స్థిరరాశి __fp16 * o_rc       = o_prev + r * HMX_FP16_TILE_N_ELMS;
        స్థిరరాశి __fp16 * v_tile_లోపల  = v_tiles;
        __fp16       * o_tile_బహిర్గతం = o_curr + r * HMX_FP16_TILE_N_ELMS;

        కొరకు (size_t c = 0; c < DV_tiles; ++c) {
            hmx_fa_o_update_tile(d_diag, o_rc, p_tile_లోపల, v_tile_లోపల, o_tile_బహిర్గతం, n_నిలువు_tiles);
            o_rc       += o_stride;
            v_tile_లోపల  += v_stride;
            o_tile_బహిర్గతం += o_stride;
        }
    }
}

రకనిర్వచనం నిర్మాణం {
    __fp16 *       o_curr;   // output (row-major tile layout)
    స్థిరరాశి __fp16 * o_prev;   // input (column-major tile layout)
    స్థిరరాశి __fp16 * d_tiles;  // diag(1/l) tiles
    uint8_t *      hmx_scales;
    size_t         n_అడ్డువరుస_tiles;
    size_t         n_అడ్డువరుస_tiles_g_br;
    size_t         DV;
} hmx_fa_o_norm_పని_t;

స్థిర శూన్య hmx_fa_o_norm_కార్మికుడు(శూన్య * దత్తాంశం) {
    hmx_fa_o_norm_పని_t * పని              = (hmx_fa_o_norm_పని_t *) దత్తాంశం;
    స్థిరరాశి size_t          n_అడ్డువరుస_tiles      = పని->n_అడ్డువరుస_tiles;
    స్థిరరాశి size_t          n_అడ్డువరుస_tiles_g_br = పని->n_అడ్డువరుస_tiles_g_br;
    స్థిరరాశి size_t          DV_tiles         = పని->DV / 32;
    స్థిరరాశి __fp16 * పరిమిత d_tiles        = పని->d_tiles;
    స్థిరరాశి __fp16 * పరిమిత o_prev         = పని->o_prev;
    __fp16 * పరిమిత o_curr               = పని->o_curr;
    __builtin_assume(n_అడ్డువరుస_tiles > 0);
    __builtin_assume(DV_tiles > 0);

    asm అస్థిర(HMX_సమితి_BIAS("%0") :: "r"((గుర్తులేని పూర్ణాంకం)పని->hmx_scales));
    స్థిరరాశి size_t o_stride = n_అడ్డువరుస_tiles_g_br * HMX_FP16_TILE_N_ELMS;
    కొరకు (size_t r = 0; r < n_అడ్డువరుస_tiles; ++r) {
        స్థిరరాశి __fp16 * d_diag = d_tiles + r * HMX_FP16_TILE_N_ELMS;
        స్థిరరాశి __fp16 * o_rc = o_prev + r * HMX_FP16_TILE_N_ELMS;
        __fp16 *       o_బహిర్గతం = o_curr + r * DV_tiles * HMX_FP16_TILE_N_ELMS;

        కొరకు (size_t c = 0; c < DV_tiles; ++c) {
            hmx_fa_o_norm_tile(d_diag, o_rc, o_బహిర్గతం);
            o_rc  += o_stride;
            o_బహిర్గతం += HMX_FP16_TILE_N_ELMS;
        }
    }
}

// Populate per-GQA-row ALiBi slopes for a given KV head.
స్థిర __attribute__((noinline)) శూన్య fa_compute_slopes(
                              స్థిరరాశి నిర్మాణం hmx_fa_context * factx,
                              uint32_t                      kv_తల,
                              size_t                        n_అడ్డువరుసలు_g) {
    __fp16 * slopes = factx->vtcm_slopes;
    ఒకవేళ (factx->max_bias == 0.0f) {
        hvx_splat_f16_a(slopes, 1.0f, n_అడ్డువరుసలు_g);
        తిరిగి;
    }

    స్థిరరాశి uint32_t G           = factx->G;
    స్థిరరాశి uint32_t n_తల_log2 = factx->n_తల_log2;
    స్థిరరాశి దశాంశం    m0          = factx->m0;
    స్థిరరాశి దశాంశం    m1          = factx->m1;

    __fp16 తాత్కాలిక_slopes[512] __attribute__((aligned(128)));
    ఒకవేళ (G <= 32) {
        // Fast path: Compute G unique slope values in vector registers
        HVX_సదిశ v_val = hvx_alibi_slopes(kv_తల, G, n_తల_log2, m0, m1);

        __fp16 తాత్కాలిక_slopes_aligned[64] __attribute__((aligned(128)));
        hvx_vmem(తాత్కాలిక_slopes_aligned) = hvx_vec_f32_గమ్యం_f16(v_val, Q6_V_vzero());

        కొరకు (uint32_t i = 0; i < G; ++i) {
            తాత్కాలిక_slopes[i] = తాత్కాలిక_slopes_aligned[i];
        }
    } లేకపోతే {
        // Fallback path: G > 32 (rare configurations)
        కొరకు (uint32_t i = 0; i < G; ++i) {
            తాత్కాలిక_slopes[i] = (__fp16)alibi_వాలు(kv_తల * G + i, n_తల_log2, m0, m1);
        }
    }

    // Allocate stack buffer to avoid scalar writes to VTCM (which generates L2 misses)
    __fp16 local_slopes[n_అడ్డువరుసలు_g] __attribute__((aligned(128)));
    కొరకు (size_t r = 0; r < n_అడ్డువరుసలు_g; ++r) {
        local_slopes[r] = తాత్కాలిక_slopes[fastmodulo(r, G, &factx->div_G)];
    }

    // Copy to VTCM slopes using HVX block copy (both are aligned to 128 bytes)
    hvx_copy_f16_aa((uint8_t *)slopes, (స్థిరరాశి uint8_t *)local_slopes, n_అడ్డువరుసలు_g);
}

స్థిర శూన్య fa_నెట్టు_ముసుగు_dma_gqa(
    dma_వరుస *               dma_q,
    స్థిరరాశి నిర్మాణం htp_tensor * ముసుగు,
    uint32_t                  q_ప్రారంభం,
    uint32_t                  im3,
    uint32_t                  kv_ప్రారంభం,
    uint32_t                  kv_తల,
    uint32_t                  G,
    uint32_t                  m_పంక్తి_బైట్లు,
    uint32_t                  kv_అడ్డువరుసలు,
    uint32_t                  n_అడ్డువరుసలు_q,
    నిర్మాణం hmx_fa_context *   factx
) {
    కొరకు (uint32_t g = 0; g < G; ++g) {
        స్థిరరాశి uint32_t h_idx = kv_తల * G + g;
        స్థిరరాశి uint32_t im2 = fastmodulo(h_idx, ముసుగు->ne[2], &factx->src3_div2);
        స్థిరరాశి dma_addr_t ms_src = ముసుగు->దత్తాంశం + q_ప్రారంభం * ముసుగు->nb[1] +
                                  im2 * ముసుగు->nb[2] + im3 * ముసుగు->nb[3] + kv_ప్రారంభం * పరిమాణం(__fp16);
        uint8_t * ms_dst = (uint8_t *) factx->vtcm_ముసుగు_buf + g * m_పంక్తి_బైట్లు;
        dma_వరుస_నెట్టు(dma_q, dma_make_దత్తాంశం(ms_dst, ms_src), G * m_పంక్తి_బైట్లు, ముసుగు->nb[1], kv_అడ్డువరుసలు * పరిమాణం(__fp16), n_అడ్డువరుసలు_q);
    }
}

స్థిర శూన్య fa_తీయి_ముసుగు_dma_gqa(dma_వరుస * dma_q, uint32_t G) {
    కొరకు (uint32_t g = 0; g < G; ++g) {
        dma_వరుస_తీయి(dma_q);
    }
}

స్థిర అంతఃవరుస శూన్య fa_prefetch_ఖండం(dma_వరుస * dma_q, స్థిరరాశి నిర్మాణం htp_tensor * k, స్థిరరాశి నిర్మాణం htp_tensor * v, స్థిరరాశి నిర్మాణం htp_tensor * ముసుగు,
                                     uint32_t b, size_t Bc, size_t కొలత_k_అడ్డువరుస_padded, size_t కొలత_k_అడ్డువరుస, size_t కొలత_v_అడ్డువరుస_padded, size_t కొలత_v_అడ్డువరుస,
                                     uint32_t ik2, uint32_t ik3, uint32_t iv2, uint32_t iv3, uint32_t q_ప్రారంభం, uint32_t im3, uint32_t kv_తల, uint32_t G,
                                     size_t m_పంక్తి_బైట్లు, size_t n_అడ్డువరుసలు_q, size_t nek1, size_t prefetch_buf, నిర్మాణం hmx_fa_context * factx) {
    స్థిరరాశి uint32_t  prefetch_ప్రారంభం = b * Bc;
    స్థిరరాశి uint32_t  prefetch_అడ్డువరుసలు  = hex_smin(Bc, nek1 - prefetch_ప్రారంభం);
    స్థిరరాశి dma_addr_t k_prefetch_src = k->దత్తాంశం + prefetch_ప్రారంభం * k->nb[1] + ik2 * k->nb[2] + ik3 * k->nb[3];
    dma_వరుస_నెట్టు(dma_q, dma_make_దత్తాంశం(factx->vtcm_k_fp16[prefetch_buf], k_prefetch_src), కొలత_k_అడ్డువరుస_padded, k->nb[1], కొలత_k_అడ్డువరుస, prefetch_అడ్డువరుసలు);
    స్థిరరాశి dma_addr_t v_prefetch_src = v->దత్తాంశం + prefetch_ప్రారంభం * v->nb[1] + iv2 * v->nb[2] + iv3 * v->nb[3];
    dma_వరుస_నెట్టు(dma_q, dma_make_దత్తాంశం(factx->vtcm_v_fp16[prefetch_buf], v_prefetch_src), కొలత_v_అడ్డువరుస_padded, v->nb[1], కొలత_v_అడ్డువరుస, prefetch_అడ్డువరుసలు);

    ఒకవేళ (ముసుగు) {
        ఒకవేళ (__builtin_expect(factx->ముసుగు_ప్రసారం, true)) {
            స్థిరరాశి dma_addr_t ms_src = ముసుగు->దత్తాంశం + q_ప్రారంభం * ముసుగు->nb[1] + im3 * ముసుగు->nb[3] + prefetch_ప్రారంభం * పరిమాణం(__fp16);
            dma_cache_fa_నెట్టు(dma_q, &factx->m_cache, ms_src, m_పంక్తి_బైట్లు, ముసుగు->nb[1], prefetch_అడ్డువరుసలు * పరిమాణం(__fp16), n_అడ్డువరుసలు_q);
        } లేకపోతే {
            fa_నెట్టు_ముసుగు_dma_gqa(dma_q, ముసుగు, q_ప్రారంభం, im3, prefetch_ప్రారంభం, kv_తల, G, m_పంక్తి_బైట్లు, prefetch_అడ్డువరుసలు, n_అడ్డువరుసలు_q, factx);
        }
    }
}

// ============================================================================
// Core HMX flash attention algorithm (GQA-merged)
// ============================================================================

పూర్ణాంకం hmx_flash_attn_ext(నిర్మాణం htp_ops_context * octx) {
    నిర్మాణం htp_thread_trace * tr_hvx = &octx->ctx->trace[0];
    నిర్మాణం htp_thread_trace * tr_hmx = &octx->ctx->trace[HTP_MAX_NTHREADS];
    స్థిరరాశి నిర్మాణం htp_tensor * q    = octx->src[0];
    స్థిరరాశి నిర్మాణం htp_tensor * k    = octx->src[1];
    స్థిరరాశి నిర్మాణం htp_tensor * v    = octx->src[2];
    స్థిరరాశి నిర్మాణం htp_tensor * ముసుగు = (octx->src[3] && octx->src[3]->దత్తాంశం) ? octx->src[3] : ఖాళీ;
    స్థిరరాశి నిర్మాణం htp_tensor * dst  = octx->dst;

    నిర్మాణం htp_context * స్థిరరాశి ctx = octx->ctx;

    ఒకవేళ (!ctx->hmx_enabled) {
        తిరిగి HTP_స్థితి_NO_SUPPORT;
    }

    // Dimensions
    స్థిరరాశి uint32_t neq0 = q->ne[0];  // head_dim (DK)
    స్థిరరాశి uint32_t neq1 = q->ne[1];  // n_tokens
    స్థిరరాశి uint32_t neq2 = q->ne[2];  // n_heads
    స్థిరరాశి uint32_t neq3 = q->ne[3];  // n_seqs

    స్థిరరాశి uint32_t nek0 = k->ne[0];  // head_dim
    స్థిరరాశి uint32_t nek1 = k->ne[1];  // kv_len

    స్థిరరాశి uint32_t nev0 = v->ne[0];  // head_dim (DV)

    స్థిరరాశి uint32_t DK = neq0;
    స్థిరరాశి uint32_t DV = nev0;

    // HMX tiles head_dim in units of 64. head_dim need not be 64- (or 32-) aligned:
    // we can operate on DK/DV rounded up to 64 with tail lanes [D, D_pad) zero-filled.
    స్థిరరాశి uint32_t DK_pad = hex_విడత_పైకి(DK, 64);
    స్థిరరాశి uint32_t DV_pad = hex_విడత_పైకి(DV, 64);
    ఒకవేళ (DK == 0 || DV == 0) {
        తిరిగి HTP_స్థితి_NO_SUPPORT;
    }

    స్థిరరాశి నిర్మాణం htp_fa_kernel_params * kparams = (స్థిరరాశి నిర్మాణం htp_fa_kernel_params *) octx->kernel_params;
    స్థిరరాశి uint32_t n_kv_heads = k->ne[2];

    // ======== Build context ========
    నిర్మాణం hmx_fa_context factx;
    memset(&factx, 0, పరిమాణం(factx));
    factx.octx           = octx;
    factx.sinks          = octx->src[4];  // NULL if this op has no attention sinks
    factx.k              = k;
    factx.v              = v;
    factx.n_దారాలు      = kparams->n_దారాలు;
    factx.DK             = DK;
    factx.DV             = DV;
    factx.DK_pad         = DK_pad;
    factx.DV_pad         = DV_pad;
    factx.n_kv           = nek1;
    factx.n_kv_heads     = n_kv_heads;
    factx.n_heads        = neq2;
    factx.G              = kparams->G;
    factx.div_G          = kparams->u.hmx.div_G;
    factx.neq1           = neq1;
    factx.Br             = kparams->Br;
    factx.Bc             = kparams->Bc;
    factx.g_br           = kparams->u.hmx.g_br;
    factx.n_kv_ఖండాలు    = kparams->n_kv_ఖండాలు;
    factx.is_q_fp32      = (q->type == HTP_TYPE_F32);
    factx.is_dst_fp32    = (dst->type == HTP_TYPE_F32);
    factx.pipeline       = (kparams->u.hmx.pipeline != 0);
    factx.ముసుగు_ప్రసారం = (kparams->u.hmx.ముసుగు_ప్రసారం != 0);
    ఒకవేళ (ముసుగు) {
        factx.src3_div2  = kparams->src3_div2;
        factx.src3_div3  = kparams->src3_div3;
    }

    factx.has_softcap   = (kparams->logit_softcap != 0.0f);
    factx.scale         = (__fp16) kparams->scale;
    factx.max_bias      = kparams->max_bias;
    factx.logit_softcap = 0;
    ఒకవేళ (factx.has_softcap) {
        factx.logit_softcap = (__fp16) kparams->logit_softcap;
    }

    factx.n_తల_log2 = kparams->n_తల_log2;
    factx.m0          = kparams->m0;
    factx.m1          = kparams->m1;

    స్థిరరాశి uint32_t Br = factx.Br;
    స్థిరరాశి uint32_t Bc = factx.Bc;
    స్థిరరాశి uint32_t g_br = factx.g_br;
    స్థిరరాశి bool pipeline = factx.pipeline;
    స్థిరరాశి uint32_t n_దారాలు = factx.n_దారాలు;
    స్థిరరాశి uint32_t G = factx.G;

    // Multi-device: prefer head-parallel partitioning (each core owns a disjoint head
    // shard), falling back to Q-block (token) split when heads don't divide evenly.
    స్థిరరాశి uint32_t n_q_ఖండాలు = (neq1 + Br - 1) / Br;
    uint32_t q_ప్రారంభం_min  = 0;
    uint32_t q_ప్రారంభం_max  = neq1;
    uint32_t kv_తల_min  = 0;
    uint32_t kv_తల_max  = n_kv_heads;

    ఒకవేళ (octx->ctx->mdev.లెక్క > 1) {
        స్థిరరాశి uint32_t mdev_లెక్క = octx->ctx->mdev.లెక్క;
        స్థిరరాశి uint32_t mdev_idx   = octx->ctx->mdev.idx;
        స్థిరరాశి uint32_t dst_e_కొలత = (dst->type == HTP_TYPE_F32) ? పరిమాణం(దశాంశం) : పరిమాణం(__fp16);
        స్థిరరాశి bool can_విభజించు      = htp_tensor_can_అడ్డువరుస_విభజన(dst, dst_e_కొలత);

        ఒకవేళ (kparams->తల_విభజించు && can_విభజించు && n_kv_heads >= mdev_లెక్క && n_kv_heads % mdev_లెక్క == 0) {
            స్థిరరాశి uint32_t kv_చొప్పున_core = n_kv_heads / mdev_లెక్క;
            kv_తల_min = mdev_idx * kv_చొప్పున_core;
            kv_తల_max = kv_తల_min + kv_చొప్పున_core;
        } లేకపోతే {
            స్థిరరాశి నిర్మాణం htp_tensor_mdev_range range = htp_tensor_mdev_విభజన(n_q_ఖండాలు, can_విభజించు ? 1 : 0, mdev_idx, mdev_లెక్క, &octx->ctx->mdev.లెక్క_div);
            స్థిరరాశి uint32_t ఖండం_ప్రారంభం = range.ప్రారంభం;
            స్థిరరాశి uint32_t ఖండం_చివర   = range.ప్రారంభం + range.లెక్క;

            ఒకవేళ (ఖండం_ప్రారంభం >= ఖండం_చివర) {
                తిరిగి HTP_స్థితి_OK;
            }

            q_ప్రారంభం_min = ఖండం_ప్రారంభం * Br;
            q_ప్రారంభం_max = MIN(ఖండం_చివర * Br, neq1);
        }
    }

    // ======== VTCM allocation (GQA-aware) ========
    // K/V row sizes drive the DMA descriptors (not the VTCM layout) and are used
    // throughout the KV loop below. The DMA copies only the real DK/DV columns; the
    // staging rows are padded to hold DK_pad/DV_pad columns (tail zero-filled below)
    // so the HMX interleave/tile logic can operate on 64-aligned head dims.
    స్థిరరాశి size_t కొలత_k_అడ్డువరుస        = htp_tensor_get_అడ్డువరుస_కొలత(k->type, DK);
    స్థిరరాశి size_t కొలత_v_అడ్డువరుస        = htp_tensor_get_అడ్డువరుస_కొలత(v->type, DV);
    స్థిరరాశి size_t కొలత_k_అడ్డువరుస_padded = hex_విడత_పైకి(DK_pad * పరిమాణం(__fp16), 128);
    స్థిరరాశి size_t కొలత_v_అడ్డువరుస_padded = hex_విడత_పైకి(DV_pad * పరిమాణం(__fp16), 128);

    // Build the VTCM layout once (shared with the host estimator) and place every
    // scratch buffer at its computed offset. Padded head dims size the HMX tiles.
    నిర్మాణం hmx_fa_vtcm_layout L;
    hmx_fa_vtcm_layout_build(&L, G, DK_pad, DV_pad, Br, Bc, n_దారాలు, pipeline, factx.is_q_fp32, factx.sinks != ఖాళీ, factx.n_heads);

    ఒకవేళ (L.మొత్తం_బైట్లు > ctx->vtcm_కొలత) {
        తిరిగి HTP_స్థితి_VTCM_TOO_SMALL;
    }

    uint8_t * స్థిరరాశి పునాది = ctx->vtcm_పునాది;

    factx.vtcm_q_dma          = VTCM_LAYOUT_PTR(__fp16, పునాది, L.off_q_dma);
    factx.vtcm_q_tiles        = VTCM_LAYOUT_PTR(__fp16, పునాది, L.off_q_tiles);
    factx.vtcm_o_tiles[0]     = VTCM_LAYOUT_PTR(__fp16, పునాది, L.off_o_tiles[0]);
    factx.vtcm_o_tiles[1]     = VTCM_LAYOUT_PTR(__fp16, పునాది, L.off_o_tiles[1]);
    factx.vtcm_k_fp16[0]      = VTCM_LAYOUT_PTR(__fp16, పునాది, L.off_k_fp16[0]);
    factx.vtcm_k_fp16[1]      = VTCM_LAYOUT_PTR(__fp16, పునాది, L.off_k_fp16[1]);
    factx.vtcm_v_fp16[0]      = VTCM_LAYOUT_PTR(__fp16, పునాది, L.off_v_fp16[0]);
    factx.vtcm_v_fp16[1]      = VTCM_LAYOUT_PTR(__fp16, పునాది, L.off_v_fp16[1]);
    factx.vtcm_k_tiles[0]     = VTCM_LAYOUT_PTR(__fp16, పునాది, L.off_k_tiles[0]);
    factx.vtcm_k_tiles[1]     = VTCM_LAYOUT_PTR_OPTIONAL(__fp16, పునాది, L.off_k_tiles[1], pipeline);
    factx.vtcm_v_tiles[0]     = VTCM_LAYOUT_PTR(__fp16, పునాది, L.off_v_tiles[0]);
    factx.vtcm_v_tiles[1]     = VTCM_LAYOUT_PTR_OPTIONAL(__fp16, పునాది, L.off_v_tiles[1], pipeline);
    factx.vtcm_s_tiles[0]     = VTCM_LAYOUT_PTR(__fp16, పునాది, L.off_s_tiles[0]);
    factx.vtcm_s_tiles[1]     = VTCM_LAYOUT_PTR_OPTIONAL(__fp16, పునాది, L.off_s_tiles[1], pipeline);
    factx.vtcm_p_tiles[0]     = VTCM_LAYOUT_PTR(__fp16, పునాది, L.off_p_tiles[0]);
    factx.vtcm_p_tiles[1]     = VTCM_LAYOUT_PTR_OPTIONAL(__fp16, పునాది, L.off_p_tiles[1], pipeline);
    factx.vtcm_d_tiles[0]     = VTCM_LAYOUT_PTR(__fp16, పునాది, L.off_d_tiles[0]);
    factx.vtcm_d_tiles[1]     = VTCM_LAYOUT_PTR_OPTIONAL(__fp16, పునాది, L.off_d_tiles[1], pipeline);
    factx.vtcm_d_inv_l        = VTCM_LAYOUT_PTR(__fp16, పునాది, L.off_d_inv_l);
    factx.vtcm_m_vec          = VTCM_LAYOUT_PTR(HVX_సదిశ, పునాది, L.off_m_vec);
    factx.vtcm_l_vec          = VTCM_LAYOUT_PTR(HVX_సదిశ, పునాది, L.off_l_vec);
    factx.vtcm_s_rowmax       = VTCM_LAYOUT_PTR(HVX_సదిశ, పునాది, L.off_s_rowmax);
    factx.vtcm_p_rowsum       = VTCM_LAYOUT_PTR(HVX_సదిశ, పునాది, L.off_p_rowsum);
    factx.vtcm_అడ్డువరుస_bufs       = VTCM_LAYOUT_PTR(HVX_సదిశ, పునాది, L.off_అడ్డువరుస_bufs);
    factx.అడ్డువరుస_buf_stride      = L.అడ్డువరుస_buf_stride;
    factx.vtcm_hmx_scales_id  = VTCM_LAYOUT_PTR(uint8_t, పునాది, L.off_hmx_scales_id);
    factx.vtcm_hmx_scales_qk  = VTCM_LAYOUT_PTR(uint8_t, పునాది, L.off_hmx_scales_qk);
    factx.vtcm_ముసుగు_buf       = VTCM_LAYOUT_PTR(__fp16, పునాది, L.off_ముసుగు_buf);
    factx.ముసుగు_buf_అడ్డువరుస_stride = L.ముసుగు_buf_అడ్డువరుస_stride;
    factx.q_tile_బైట్లు        = L.q_tile_బైట్లు;
    factx.k_tile_బైట్లు        = L.k_tile_బైట్లు;
    factx.o_tile_బైట్లు        = L.o_tile_బైట్లు;
    factx.నిలువు_vec_బైట్లు       = L.నిలువు_vec_బైట్లు;
    factx.d_tile_బైట్లు        = L.d_tile_బైట్లు;
    factx.vtcm_slopes         = VTCM_LAYOUT_PTR(__fp16, పునాది, L.off_slopes);
    factx.vtcm_sinks          = VTCM_LAYOUT_PTR_OPTIONAL(దశాంశం, పునాది, L.off_sinks, factx.sinks != ఖాళీ);

    స్థిరరాశి size_t m_పంక్తి_బైట్లు = L.m_పంక్తి_బైట్లు;  // used by the mask DMAs in the KV loop

    dma_cache_fa_init(&factx.m_cache, (uint8_t *) factx.vtcm_ముసుగు_buf, L.m_buf_స్థానం_బైట్లు, HMX_FA_DMA_CACHE_కొలత);

    // Head-dim padding: the K/V DMA staging buffers and the flat-Q buffer are laid out
    // with padded row strides (size_{k,v,q}_row_padded, covering D_pad columns) but the
    // DMA only writes the real D columns per row. Zero the whole staging buffers once up
    // front so tail lanes [D, D_pad) stay zero for all KV blocks. No-op when already aligned.
    ఒకవేళ (DK_pad != DK || DV_pad != DV) {
        స్థిరరాశి size_t k_buf_బైట్లు = (size_t) factx.Bc * కొలత_k_అడ్డువరుస_padded;
        స్థిరరాశి size_t v_buf_బైట్లు = (size_t) factx.Bc * కొలత_v_అడ్డువరుస_padded;
        hvx_splat_u8_a((అక్షరం *) factx.vtcm_k_fp16[0], 0, k_buf_బైట్లు);
        hvx_splat_u8_a((అక్షరం *) factx.vtcm_k_fp16[1], 0, k_buf_బైట్లు);
        hvx_splat_u8_a((అక్షరం *) factx.vtcm_v_fp16[0], 0, v_buf_బైట్లు);
        hvx_splat_u8_a((అక్షరం *) factx.vtcm_v_fp16[1], 0, v_buf_బైట్లు);
        // Flat-Q DMA scratch
        ఒకవేళ (factx.vtcm_q_dma) {
            స్థిరరాశి size_t q_dma_బైట్లు = hex_align_పైకి(factx.g_br * DK * (factx.is_q_fp32 ? పరిమాణం(దశాంశం) : పరిమాణం(__fp16)), 128);
            hvx_splat_u8_a((అక్షరం *) factx.vtcm_q_dma, 0, q_dma_బైట్లు);
        }
    }

    // ======== Initialize HMX output scales ========
    hmx_init_స్తంభం_scales(factx.vtcm_hmx_scales_id, Q6_V_vsplat_R(0x3c00)); // 1.0
    {
        సంఘం { __fp16 f; uint16_t u; } sc = { .f = factx.scale };
        hmx_init_స్తంభం_scales(factx.vtcm_hmx_scales_qk, Q6_V_vsplat_R(sc.u));
    }

    // ======== DMA setup ========
    dma_వరుస * స్థిరరాశి dma_q = ctx->dma[0];

    ఒకవేళ (factx.sinks) {
        dma_వరుస_నెట్టు(dma_q, dma_make_దత్తాంశం(factx.vtcm_sinks, factx.sinks->దత్తాంశం), L.sinks_బైట్లు, 0, factx.sinks->కొలత, 1);
        dma_వరుస_తీయి(dma_q);
    }

    స్థిరరాశి size_t n_అడ్డువరుస_tiles_g_br = g_br / HMX_FP16_TILE_N_అడ్డువరుసలు;
    స్థిరరాశి size_t n_tiles_చొప్పున_bc   = Bc / HMX_FP16_TILE_N_నిలువుల_సంఖ్య;

    స్థిరరాశి size_t qo_element_కొలత = factx.is_q_fp32 ? పరిమాణం(దశాంశం) : పరిమాణం(__fp16);

    స్థిరరాశి bool q_transposed                 = q->nb[1] < q->nb[2];
    స్థిరరాశి size_t q_src_stride               = q_transposed ? q->nb[2] : q->nb[1];
    స్థిరరాశి size_t q_అడ్డువరుస_బైట్లు_untransposed   = factx.G * factx.DK * qo_element_కొలత;
    స్థిరరాశి size_t q_అడ్డువరుస_బైట్లు_trans_గుణకం   = factx.DK * qo_element_కొలత;

    // ======== Reusable job descriptors for pipeline ========
    hmx_fa_qk_పని_t       qk_పని;
    hmx_fa_o_update_పని_t ou_పని;
    hmx_fa_o_norm_పని_t   on_పని;

    // ======== Main loop ========
    కొరకు (uint32_t ib3 = 0; ib3 < neq3; ++ib3) {
        స్థిరరాశి uint32_t im3 = ముసుగు ? fastmodulo(ib3, ముసుగు->ne[3], &factx.src3_div3) : 0;
        కొరకు (uint32_t q_ప్రారంభం = q_ప్రారంభం_min; q_ప్రారంభం < q_ప్రారంభం_max; q_ప్రారంభం += Br) {
            స్థిరరాశి uint32_t n_అడ్డువరుసలు_q    = hex_smin(Br, neq1 - q_ప్రారంభం);
            స్థిరరాశి size_t   n_అడ్డువరుసలు_g    = n_అడ్డువరుసలు_q * G;
            స్థిరరాశి size_t   g_br_actual = hex_align_పైకి(n_అడ్డువరుసలు_g, HMX_FP16_TILE_N_అడ్డువరుసలు);
            స్థిరరాశి size_t   n_అడ్డువరుస_tiles = g_br_actual / HMX_FP16_TILE_N_అడ్డువరుసలు;

            uint16_t చురుకైన_ఖండాలు[FA_MAX_KV_ఖండాలు];
            uint32_t n_చురుకైన_ఖండాలు = factx.n_kv_ఖండాలు;
            ఒకవేళ (ముసుగు && factx.ముసుగు_ప్రసారం && factx.n_kv_ఖండాలు > 1) {
                స్థిరరాశి size_t scratch_బైట్లు = factx.k_tile_బైట్లు * (factx.pipeline ? 2 : 1);
                n_చురుకైన_ఖండాలు = fa_పరిశీలన_చురుకైన_ఖండాలు(
                    dma_q, ముసుగు, q_ప్రారంభం, n_అడ్డువరుసలు_q, 0, im3, nek1, Bc,
                    factx.n_kv_ఖండాలు, factx.vtcm_k_tiles[0], scratch_బైట్లు, చురుకైన_ఖండాలు);
            } లేకపోతే {
                #సూచన clang loop vectorize(disable)
                కొరకు (uint32_t b = 0; b < factx.n_kv_ఖండాలు; ++b) {
                    చురుకైన_ఖండాలు[b] = (uint16_t) b;
                }
            }
            స్థిరరాశి uint32_t b0       = (n_చురుకైన_ఖండాలు > 0) ? చురుకైన_ఖండాలు[0] : 0;
            స్థిరరాశి uint32_t b0_ప్రారంభం = b0 * Bc;
            స్థిరరాశి uint32_t b0_అడ్డువరుసలు  = hex_smin(Bc, nek1 - b0_ప్రారంభం);

            కొరకు (uint32_t kv_తల = kv_తల_min; kv_తల < kv_తల_max; ++kv_తల) {
                స్థిరరాశి uint32_t ik2 = kv_తల;
                స్థిరరాశి uint32_t ik3 = fastdiv(ib3, &kparams->ప్రసారం_rk3);
                స్థిరరాశి uint32_t iv2 = kv_తల;
                స్థిరరాశి uint32_t iv3 = fastdiv(ib3, &kparams->ప్రసారం_rv3);

                // 1. Push Q and KV DMAs for the first kv_head of this q_start block.
                // Subsequent kv_head iterations are enqueued early at the end of the previous iteration.
                ఒకవేళ (kv_తల == kv_తల_min) {
                    స్థిరరాశి dma_addr_t q_ptr = q->దత్తాంశం + q_ప్రారంభం * q->nb[1] +
                                            (kv_తల * factx.G) * q->nb[2] + ib3 * q->nb[3];
                    స్థిరరాశి size_t q_అడ్డువరుస_బైట్లు = q_transposed ? n_అడ్డువరుసలు_q * q_అడ్డువరుస_బైట్లు_trans_గుణకం : q_అడ్డువరుస_బైట్లు_untransposed;
                    స్థిరరాశి size_t n_అడ్డువరుసలు      = q_transposed ? factx.G : n_అడ్డువరుసలు_q;
                    dma_వరుస_నెట్టు(dma_q, dma_make_దత్తాంశం(factx.vtcm_q_dma, q_ptr), q_అడ్డువరుస_బైట్లు, hex_smax(q_src_stride, q_అడ్డువరుస_బైట్లు), q_అడ్డువరుస_బైట్లు, n_అడ్డువరుసలు);

                    ఒకవేళ (n_చురుకైన_ఖండాలు > 0) {
                        స్థిరరాశి dma_addr_t k_src = k->దత్తాంశం + b0_ప్రారంభం * k->nb[1] + ik2 * k->nb[2] + ik3 * k->nb[3];
                        dma_వరుస_నెట్టు(dma_q, dma_make_దత్తాంశం(factx.vtcm_k_fp16[0], k_src), కొలత_k_అడ్డువరుస_padded, k->nb[1], కొలత_k_అడ్డువరుస, b0_అడ్డువరుసలు);

                        స్థిరరాశి dma_addr_t v_src = v->దత్తాంశం + b0_ప్రారంభం * v->nb[1] + iv2 * v->nb[2] + iv3 * v->nb[3];
                        dma_వరుస_నెట్టు(dma_q, dma_make_దత్తాంశం(factx.vtcm_v_fp16[0], v_src), కొలత_v_అడ్డువరుస_padded, v->nb[1], కొలత_v_అడ్డువరుస, b0_అడ్డువరుసలు);

                        ఒకవేళ (factx.pipeline && ముసుగు) {
                            ఒకవేళ (__builtin_expect(factx.ముసుగు_ప్రసారం, true)) {
                                స్థిరరాశి dma_addr_t ms_src = ముసుగు->దత్తాంశం + q_ప్రారంభం * ముసుగు->nb[1] + im3 * ముసుగు->nb[3] + b0_ప్రారంభం * పరిమాణం(__fp16);
                                dma_cache_fa_నెట్టు(dma_q, &factx.m_cache, ms_src, m_పంక్తి_బైట్లు, ముసుగు->nb[1], b0_అడ్డువరుసలు * పరిమాణం(__fp16), n_అడ్డువరుసలు_q);
                            } లేకపోతే {
                                fa_నెట్టు_ముసుగు_dma_gqa(dma_q, ముసుగు, q_ప్రారంభం, im3, b0_ప్రారంభం, kv_తల, G, m_పంక్తి_బైట్లు, b0_అడ్డువరుసలు, n_అడ్డువరుసలు_q, &factx);
                            }
                        }
                    }
                }

                // 2. Pop Q DMA (blocks until Q is loaded)
                dma_వరుస_తీయి(dma_q);

                // ---- Load Q block & Initialize per-block state ----
                fa_phase_q_ఎక్కించు(&factx, q, q_ప్రారంభం, kv_తల, ib3, n_అడ్డువరుసలు_g);

                __fp16 * o_tile_prev = factx.vtcm_o_tiles[0];
                __fp16 * o_tile_curr = factx.vtcm_o_tiles[1];

                // ---- KV block loop with DMA double-buffering ----
                size_t buf_idx = 0;

                htp_trace_event_ప్రారంభం(tr_hvx, HTP_TRACE_EVT_HVX_A_PREP, (uint16_t) q_ప్రారంభం);
                fa_compute_slopes(&factx, kv_తల, n_అడ్డువరుసలు_g);
                htp_trace_event_stop(tr_hvx, HTP_TRACE_EVT_HVX_A_PREP, (uint16_t) q_ప్రారంభం);

                స్థిరరాశి size_t k_src_stride = కొలత_k_అడ్డువరుస_padded / పరిమాణం(__fp16);
                స్థిరరాశి size_t v_src_stride = కొలత_v_అడ్డువరుస_padded / పరిమాణం(__fp16);

                hmx_వరుస_t hmx_q = ctx->hmx_వరుస;

                ఒకవేళ (factx.pipeline) {
                    // Double-buffered job structs because HMX queue runs asynchronously
                    hmx_fa_qk_పని_t qk_పని[2];
                    hmx_fa_o_update_పని_t ou_పని[2];

                    // Prefetch second active block early if there are multiple active blocks
                    ఒకవేళ (n_చురుకైన_ఖండాలు > 1) {
                        fa_prefetch_ఖండం(dma_q, k, v, ముసుగు, చురుకైన_ఖండాలు[1], Bc, కొలత_k_అడ్డువరుస_padded, కొలత_k_అడ్డువరుస, కొలత_v_అడ్డువరుస_padded, కొలత_v_అడ్డువరుస,
                                          ik2, ik3, iv2, iv3, q_ప్రారంభం, im3, kv_తల, G, m_పంక్తి_బైట్లు, n_అడ్డువరుసలు_q, nek1, 1, &factx);
                    }

                    // Prep and start QK-dot(0)
                    శూన్య * curr_k0 = (శూన్య *) dma_వరుస_తీయి(dma_q).dst;
                    fa_phase_k_interleave(&factx, b0_అడ్డువరుసలు, k_src_stride, curr_k0, b0_ప్రారంభం, 0);

                    qk_పని[0].q_tiles        = factx.vtcm_q_tiles;
                    qk_పని[0].k_tiles        = factx.vtcm_k_tiles[0];
                    qk_పని[0].s_tiles        = factx.vtcm_s_tiles[0];
                    qk_పని[0].n_అడ్డువరుస_tiles    = n_అడ్డువరుస_tiles;
                    qk_పని[0].n_నిలువు_tiles    = hmx_ceil_div(b0_అడ్డువరుసలు, HMX_FP16_TILE_N_నిలువుల_సంఖ్య);
                    qk_పని[0].n_dot_tiles    = DK_pad / 32;
                    qk_పని[0].n_tiles_చొప్పున_bc = n_tiles_చొప్పున_bc;
                    qk_పని[0].hmx_scales     = factx.vtcm_hmx_scales_qk;
                    hmx_వరుస_నెట్టు(hmx_q, hmx_వరుస_make_desc(hmx_fa_qk_dot_కార్మికుడు, &qk_పని[0]));

                    కొరకు (uint32_t అడుగు = 0; అడుగు < n_చురుకైన_ఖండాలు; ++అడుగు) {
                        స్థిరరాశి uint32_t kv_blk      = చురుకైన_ఖండాలు[అడుగు];
                        స్థిరరాశి uint32_t kv_ప్రారంభం    = kv_blk * Bc;
                        స్థిరరాశి uint32_t kv_అడ్డువరుసలు     = hex_smin(Bc, nek1 - kv_ప్రారంభం);
                        స్థిరరాశి size_t   n_నిలువు_tiles = hmx_ceil_div(kv_అడ్డువరుసలు, HMX_FP16_TILE_N_నిలువుల_సంఖ్య);

                        // ---- 1. Pop and run V-prep for current block ----
                        శూన్య * curr_v = (శూన్య *) dma_వరుస_తీయి(dma_q).dst;
                        fa_phase_v_interleave(&factx, kv_అడ్డువరుసలు, v_src_stride, curr_v, factx.vtcm_v_tiles[buf_idx], n_tiles_చొప్పున_bc, kv_ప్రారంభం);

                        // ---- 2. Pop and run mask-prep for current block ----
                        __fp16 * current_ముసుగు_vtcm = ఖాళీ;
                        ఒకవేళ (ముసుగు) {
                            ఒకవేళ (__builtin_expect(factx.ముసుగు_ప్రసారం, true)) {
                                current_ముసుగు_vtcm = (__fp16 *) dma_వరుస_తీయి(dma_q).dst;
                            } లేకపోతే {
                                fa_తీయి_ముసుగు_dma_gqa(dma_q, G);
                                current_ముసుగు_vtcm = factx.vtcm_ముసుగు_buf;
                            }
                        }

                        // ---- 3. Start HMX O update for step - 1 (reads P[1 - buf_idx], V[1 - buf_idx], D) ----
                        // O update relies on the previous block's P and V tiles.
                        // O update MUST be pushed before the next block's QK-dot: hmx_queue_pop() retires the
                        // oldest descriptor, so push order alone decides which pop waits for which job.
                        // If OU went in after QK(i+1), the pop below would retire QK(i+1) and leave
                        // OU(i-1) in flight into the next iteration, where V-prep overwrites V[prev_buf].
                        ఒకవేళ (అడుగు > 0) {
                            స్థిరరాశి uint32_t prev_blk      = చురుకైన_ఖండాలు[అడుగు - 1];
                            స్థిరరాశి size_t prev_buf        = 1 - buf_idx;
                            ou_పని[prev_buf].o_curr      = o_tile_curr;
                            ou_పని[prev_buf].o_prev      = o_tile_prev;
                            ou_పని[prev_buf].p_tiles     = factx.vtcm_p_tiles[prev_buf];
                            ou_పని[prev_buf].v_tiles     = factx.vtcm_v_tiles[prev_buf];
                            ou_పని[prev_buf].d_tiles     = factx.vtcm_d_tiles[prev_buf];
                            ou_పని[prev_buf].hmx_scales  = factx.vtcm_hmx_scales_id;
                            ou_పని[prev_buf].n_అడ్డువరుస_tiles = n_అడ్డువరుస_tiles;
                            ou_పని[prev_buf].n_నిలువు_tiles =
                                hmx_ceil_div(hex_smin(Bc, nek1 - prev_blk * Bc), HMX_FP16_TILE_N_నిలువుల_సంఖ్య);
                            ou_పని[prev_buf].n_అడ్డువరుస_tiles_g_br = n_అడ్డువరుస_tiles_g_br;
                            ou_పని[prev_buf].n_tiles_చొప్పున_bc   = n_tiles_చొప్పున_bc;
                            ou_పని[prev_buf].DV               = DV_pad;
                            hmx_వరుస_నెట్టు(hmx_q, hmx_వరుస_make_desc(hmx_fa_o_update_కార్మికుడు, &ou_పని[prev_buf]));
                        }

                        // ---- 4. Pop and run K-prep for next active block & push next QK-dot ----
                        ఒకవేళ (అడుగు + 1 < n_చురుకైన_ఖండాలు) {
                            స్థిరరాశి uint32_t తదుపరి_blk   = చురుకైన_ఖండాలు[అడుగు + 1];
                            స్థిరరాశి uint32_t తదుపరి_ప్రారంభం = తదుపరి_blk * Bc;
                            స్థిరరాశి uint32_t తదుపరి_అడ్డువరుసలు  = hex_smin(Bc, nek1 - తదుపరి_ప్రారంభం);
                            స్థిరరాశి size_t   తదుపరి_buf   = 1 - buf_idx;

                            శూన్య * తదుపరి_k = (శూన్య *) dma_వరుస_తీయి(dma_q).dst;
                            fa_phase_k_interleave(&factx, తదుపరి_అడ్డువరుసలు, k_src_stride, తదుపరి_k, తదుపరి_ప్రారంభం, తదుపరి_buf);

                            qk_పని[తదుపరి_buf].q_tiles        = factx.vtcm_q_tiles;
                            qk_పని[తదుపరి_buf].k_tiles        = factx.vtcm_k_tiles[తదుపరి_buf];
                            qk_పని[తదుపరి_buf].s_tiles        = factx.vtcm_s_tiles[తదుపరి_buf];
                            qk_పని[తదుపరి_buf].n_అడ్డువరుస_tiles    = n_అడ్డువరుస_tiles;
                            qk_పని[తదుపరి_buf].n_నిలువు_tiles    = hmx_ceil_div(తదుపరి_అడ్డువరుసలు, HMX_FP16_TILE_N_నిలువుల_సంఖ్య);
                            qk_పని[తదుపరి_buf].n_dot_tiles    = DK_pad / 32;
                            qk_పని[తదుపరి_buf].n_tiles_చొప్పున_bc = n_tiles_చొప్పున_bc;
                            qk_పని[తదుపరి_buf].hmx_scales     = factx.vtcm_hmx_scales_qk;
                            hmx_వరుస_నెట్టు(hmx_q, hmx_వరుస_make_desc(hmx_fa_qk_dot_కార్మికుడు, &qk_పని[తదుపరి_buf]));
                        }

                        // ---- 5. Wait for current block's QK-dot to finish ----
                        hmx_వరుస_తీయి(hmx_q);

                        // ---- 6. Phase 2: softmax + build_D ----
                        fa_softmax_వాదనలు_t sargs;
                        memset(&sargs, 0, పరిమాణం(sargs));
                        sargs.factx                = &factx;
                        sargs.buf_idx              = buf_idx;
                        sargs.kv_అడ్డువరుసలు              = kv_అడ్డువరుసలు;
                        sargs.n_అడ్డువరుసలు_g             = n_అడ్డువరుసలు_g;
                        sargs.n_నిలువు_tiles          = n_నిలువు_tiles;
                        sargs.n_tiles_చొప్పున_bc       = n_tiles_చొప్పున_bc;
                        sargs.n_అడ్డువరుస_tiles          = n_అడ్డువరుస_tiles;
                        sargs.n_అడ్డువరుస_tiles_g_br     = n_అడ్డువరుస_tiles_g_br;
                        sargs.Bc                   = Bc;
                        sargs.G                    = G;
                        sargs.kv_తల              = kv_తల;
                        sargs.kv_ప్రారంభం             = kv_ప్రారంభం;
                        sargs.q_ప్రారంభం              = q_ప్రారంభం;
                        sargs.ib3                  = ib3;
                        sargs.has_alibi            = (factx.max_bias != 0.0f);
                        sargs.ముసుగు                 = ముసుగు;
                        sargs.ముసుగు_vtcm            = current_ముసుగు_vtcm;
                        sargs.ముసుగు_vtcm_అడ్డువరుస_stride = factx.ముసుగు_buf_అడ్డువరుస_stride;
                        sargs.slopes               = factx.vtcm_slopes;

                        // Run Softmax on HVX (blocking call)
                        fa_phase_softmax_and_build_d(&factx, &sargs, n_అడ్డువరుస_tiles, n_అడ్డువరుస_tiles_g_br);

                        // Wait for HMX O update for step - 1 to finish
                        ఒకవేళ (అడుగు > 0) {
                            hmx_వరుస_తీయి(hmx_q);
                            hex_మార్పిడి_ptr((శూన్య **) &o_tile_curr, (శూన్య **) &o_tile_prev);
                        }

                        // Prefetch block step + 2
                        ఒకవేళ (అడుగు + 2 < n_చురుకైన_ఖండాలు) {
                            fa_prefetch_ఖండం(dma_q, k, v, ముసుగు, చురుకైన_ఖండాలు[అడుగు + 2], Bc, కొలత_k_అడ్డువరుస_padded, కొలత_k_అడ్డువరుస, కొలత_v_అడ్డువరుస_padded, కొలత_v_అడ్డువరుస,
                                              ik2, ik3, iv2, iv3, q_ప్రారంభం, im3, kv_తల, G, m_పంక్తి_బైట్లు, n_అడ్డువరుసలు_q, nek1, buf_idx, &factx);
                        }

                        buf_idx = 1 - buf_idx;
                    }

                    // Epilogue
                    ఒకవేళ (n_చురుకైన_ఖండాలు > 0) {
                        స్థిరరాశి uint32_t చివరి_blk = చురుకైన_ఖండాలు[n_చురుకైన_ఖండాలు - 1];
                        స్థిరరాశి size_t చివరి_నిలువుల_సంఖ్య  = hmx_ceil_div(hex_smin(Bc, nek1 - చివరి_blk * Bc), HMX_FP16_TILE_N_నిలువుల_సంఖ్య);
                        ou_పని[0].o_curr           = o_tile_curr;
                        ou_పని[0].o_prev           = o_tile_prev;
                        ou_పని[0].p_tiles          = factx.vtcm_p_tiles[1 - buf_idx];
                        ou_పని[0].v_tiles          = factx.vtcm_v_tiles[1 - buf_idx];
                        ou_పని[0].d_tiles          = factx.vtcm_d_tiles[1 - buf_idx];
                        ou_పని[0].hmx_scales       = factx.vtcm_hmx_scales_id;
                        ou_పని[0].n_అడ్డువరుస_tiles      = n_అడ్డువరుస_tiles;
                        ou_పని[0].n_నిలువు_tiles      = చివరి_నిలువుల_సంఖ్య;
                        ou_పని[0].n_అడ్డువరుస_tiles_g_br = n_అడ్డువరుస_tiles_g_br;
                        ou_పని[0].n_tiles_చొప్పున_bc   = n_tiles_చొప్పున_bc;
                        ou_పని[0].DV               = DV_pad;
                        hmx_వరుస_నెట్టు(hmx_q, hmx_వరుస_make_desc(hmx_fa_o_update_కార్మికుడు, &ou_పని[0]));

                        // Overlapped: run HVX build diag inv L while HMX is busy executing the update
                        htp_trace_event_ప్రారంభం(tr_hvx, HTP_TRACE_EVT_HVX_O_PROC, (uint16_t) q_ప్రారంభం);
                        fa_build_d_diag_inv_l(&factx, n_అడ్డువరుస_tiles, n_అడ్డువరుస_tiles_g_br, kv_తల, n_అడ్డువరుసలు_g);
                        htp_trace_event_stop(tr_hvx, HTP_TRACE_EVT_HVX_O_PROC, (uint16_t) q_ప్రారంభం);
                        hmx_వరుస_తీయి(hmx_q);

                        hex_మార్పిడి_ptr((శూన్య **) &o_tile_curr, (శూన్య **) &o_tile_prev);
                    }

                } లేకపోతే {
                    // Fallback path
                    కొరకు (uint32_t అడుగు = 0; అడుగు < n_చురుకైన_ఖండాలు; ++అడుగు) {
                        స్థిరరాశి uint32_t kv_blk      = చురుకైన_ఖండాలు[అడుగు];
                        స్థిరరాశి uint32_t kv_ప్రారంభం    = kv_blk * Bc;
                        స్థిరరాశి uint32_t kv_అడ్డువరుసలు     = hex_smin(Bc, nek1 - kv_ప్రారంభం);
                        స్థిరరాశి size_t   n_నిలువు_tiles = hmx_ceil_div(kv_అడ్డువరుసలు, HMX_FP16_TILE_N_నిలువుల_సంఖ్య);

                        ఒకవేళ (ముసుగు) {
                            ఒకవేళ (__builtin_expect(factx.ముసుగు_ప్రసారం, true)) {
                                స్థిరరాశి dma_addr_t ms_src = ముసుగు->దత్తాంశం + q_ప్రారంభం * ముసుగు->nb[1] + im3 * ముసుగు->nb[3] + kv_ప్రారంభం * పరిమాణం(__fp16);
                                dma_cache_fa_నెట్టు(dma_q, &factx.m_cache, ms_src, m_పంక్తి_బైట్లు, ముసుగు->nb[1], kv_అడ్డువరుసలు * పరిమాణం(__fp16), n_అడ్డువరుసలు_q);
                            } లేకపోతే {
                                fa_నెట్టు_ముసుగు_dma_gqa(dma_q, ముసుగు, q_ప్రారంభం, im3, kv_ప్రారంభం, kv_తల, G, m_పంక్తి_బైట్లు, kv_అడ్డువరుసలు, n_అడ్డువరుసలు_q, &factx);
                            }
                        }

                        ఒకవేళ (అడుగు + 1 < n_చురుకైన_ఖండాలు) {
                            స్థిరరాశి uint32_t  prefetch_blk   = చురుకైన_ఖండాలు[అడుగు + 1];
                            స్థిరరాశి uint32_t  prefetch_ప్రారంభం = prefetch_blk * Bc;
                            స్థిరరాశి uint32_t  prefetch_అడ్డువరుసలు  = hex_smin(Bc, nek1 - prefetch_ప్రారంభం);
                            స్థిరరాశి size_t    prefetch_buf   = 1 - buf_idx;
                            స్థిరరాశి dma_addr_t k_prefetch_src = k->దత్తాంశం + prefetch_ప్రారంభం * k->nb[1] + ik2 * k->nb[2] + ik3 * k->nb[3];
                            dma_వరుస_నెట్టు(dma_q, dma_make_దత్తాంశం(factx.vtcm_k_fp16[prefetch_buf], k_prefetch_src), కొలత_k_అడ్డువరుస_padded, k->nb[1], కొలత_k_అడ్డువరుస, prefetch_అడ్డువరుసలు);
                            స్థిరరాశి dma_addr_t v_prefetch_src = v->దత్తాంశం + prefetch_ప్రారంభం * v->nb[1] + iv2 * v->nb[2] + iv3 * v->nb[3];
                            dma_వరుస_నెట్టు(dma_q, dma_make_దత్తాంశం(factx.vtcm_v_fp16[prefetch_buf], v_prefetch_src), కొలత_v_అడ్డువరుస_padded, v->nb[1], కొలత_v_అడ్డువరుస, prefetch_అడ్డువరుసలు);
                        }

                        // Wait for current K DMA and interleave
                        శూన్య * curr_k = (శూన్య *) dma_వరుస_తీయి(dma_q).dst;
                        fa_phase_k_interleave(&factx, kv_అడ్డువరుసలు, k_src_stride, curr_k, kv_ప్రారంభం, 0);

                        {
                            qk_పని.q_tiles        = factx.vtcm_q_tiles;
                            qk_పని.k_tiles        = factx.vtcm_k_tiles[0];
                            qk_పని.s_tiles        = factx.vtcm_s_tiles[0];
                            qk_పని.n_అడ్డువరుస_tiles    = n_అడ్డువరుస_tiles;
                            qk_పని.n_నిలువు_tiles    = n_నిలువు_tiles;
                            qk_పని.n_dot_tiles    = (size_t) (DK_pad / 32);
                            qk_పని.n_tiles_చొప్పున_bc = n_tiles_చొప్పున_bc;
                            qk_పని.hmx_scales     = factx.vtcm_hmx_scales_qk;

                            hmx_వరుస_నెట్టు(ctx->hmx_వరుస, hmx_వరుస_make_desc(hmx_fa_qk_dot_కార్మికుడు, &qk_పని));
                            hmx_వరుస_తీయి(ctx->hmx_వరుస);
                        }

                        // Wait for current V DMA and interleave
                        శూన్య * curr_v = (శూన్య *) dma_వరుస_తీయి(dma_q).dst;
                        fa_phase_v_interleave(&factx, kv_అడ్డువరుసలు, v_src_stride, curr_v, factx.vtcm_v_tiles[0], n_tiles_చొప్పున_bc, kv_ప్రారంభం);

                        // ---- Phase 3: softmax + build_D ----
                        __fp16 * current_ముసుగు_vtcm = ఖాళీ;
                        ఒకవేళ (ముసుగు) {
                            ఒకవేళ (__builtin_expect(factx.ముసుగు_ప్రసారం, true)) {
                                current_ముసుగు_vtcm = (__fp16 *) dma_వరుస_తీయి(dma_q).dst;
                            } లేకపోతే {
                                fa_తీయి_ముసుగు_dma_gqa(dma_q, G);
                                current_ముసుగు_vtcm = factx.vtcm_ముసుగు_buf;
                            }
                        }

                        fa_softmax_వాదనలు_t sargs;
                        memset(&sargs, 0, పరిమాణం(sargs));
                        sargs.factx                = &factx;
                        sargs.kv_అడ్డువరుసలు              = kv_అడ్డువరుసలు;
                        sargs.n_అడ్డువరుసలు_g             = n_అడ్డువరుసలు_g;
                        sargs.n_నిలువు_tiles          = n_నిలువు_tiles;
                        sargs.n_tiles_చొప్పున_bc       = n_tiles_చొప్పున_bc;
                        sargs.n_అడ్డువరుస_tiles          = n_అడ్డువరుస_tiles;
                        sargs.n_అడ్డువరుస_tiles_g_br     = n_అడ్డువరుస_tiles_g_br;
                        sargs.Bc                   = Bc;
                        sargs.G                    = G;
                        sargs.kv_తల              = kv_తల;
                        sargs.kv_ప్రారంభం             = kv_ప్రారంభం;
                        sargs.q_ప్రారంభం              = q_ప్రారంభం;
                        sargs.ib3                  = ib3;
                        sargs.has_alibi            = (factx.max_bias != 0.0f);
                        sargs.ముసుగు                 = ముసుగు;
                        sargs.ముసుగు_vtcm            = current_ముసుగు_vtcm;
                        sargs.ముసుగు_vtcm_అడ్డువరుస_stride = factx.ముసుగు_buf_అడ్డువరుస_stride;
                        sargs.slopes               = factx.vtcm_slopes;
                        fa_phase_softmax_and_build_d(&factx, &sargs, n_అడ్డువరుస_tiles, n_అడ్డువరుస_tiles_g_br);

                        {
                            ou_పని.o_curr           = o_tile_curr;
                            ou_పని.o_prev           = o_tile_prev;
                            ou_పని.p_tiles          = factx.vtcm_p_tiles[0];
                            ou_పని.v_tiles          = factx.vtcm_v_tiles[0];
                            ou_పని.d_tiles          = factx.vtcm_d_tiles[0];
                            ou_పని.hmx_scales       = factx.vtcm_hmx_scales_id;
                            ou_పని.n_అడ్డువరుస_tiles      = n_అడ్డువరుస_tiles;
                            ou_పని.n_నిలువు_tiles      = n_నిలువు_tiles;
                            ou_పని.n_అడ్డువరుస_tiles_g_br = n_అడ్డువరుస_tiles_g_br;
                            ou_పని.n_tiles_చొప్పున_bc   = n_tiles_చొప్పున_bc;
                            ou_పని.DV               = DV_pad;

                            hmx_వరుస_నెట్టు(ctx->hmx_వరుస, hmx_వరుస_make_desc(hmx_fa_o_update_కార్మికుడు, &ou_పని));
                            ఒకవేళ (అడుగు + 1 == n_చురుకైన_ఖండాలు) {
                                // Overlapped: run HVX build diag inv L while HMX is busy executing the update
                                htp_trace_event_ప్రారంభం(tr_hvx, HTP_TRACE_EVT_HVX_O_PROC, (uint16_t) q_ప్రారంభం);
                                fa_build_d_diag_inv_l(&factx, n_అడ్డువరుస_tiles, n_అడ్డువరుస_tiles_g_br, kv_తల, n_అడ్డువరుసలు_g);
                                htp_trace_event_stop(tr_hvx, HTP_TRACE_EVT_HVX_O_PROC, (uint16_t) q_ప్రారంభం);
                            }
                            hmx_వరుస_తీయి(ctx->hmx_వరుస);

                            hex_మార్పిడి_ptr((శూన్య **) &o_tile_curr, (శూన్య **) &o_tile_prev);
                        }

                        buf_idx = 1 - buf_idx;
                    }
                }

                // Enqueue DMAs for the next kv_head early so they overlap with O-PROC
                స్థిరరాశి uint32_t తదుపరి_kv_తల = kv_తల + 1;
                స్థిరరాశి bool has_తదుపరి = (తదుపరి_kv_తల < kv_తల_max);

                ఒకవేళ (has_తదుపరి) {
                    స్థిరరాశి dma_addr_t తదుపరి_q_ptr = q->దత్తాంశం + q_ప్రారంభం * q->nb[1] + (తదుపరి_kv_తల * factx.G) * q->nb[2] + ib3 * q->nb[3];
                    స్థిరరాశి size_t తదుపరి_q_అడ్డువరుస_బైట్లు = q_transposed ? n_అడ్డువరుసలు_q * q_అడ్డువరుస_బైట్లు_trans_గుణకం : q_అడ్డువరుస_బైట్లు_untransposed;
                    స్థిరరాశి size_t తదుపరి_n_అడ్డువరుసలు      = q_transposed ? factx.G : n_అడ్డువరుసలు_q;
                    dma_వరుస_నెట్టు(dma_q, dma_make_దత్తాంశం(factx.vtcm_q_dma, తదుపరి_q_ptr), తదుపరి_q_అడ్డువరుస_బైట్లు, hex_smax(q_src_stride, తదుపరి_q_అడ్డువరుస_బైట్లు), తదుపరి_q_అడ్డువరుస_బైట్లు, తదుపరి_n_అడ్డువరుసలు);

                    ఒకవేళ (n_చురుకైన_ఖండాలు > 0) {
                        స్థిరరాశి uint32_t తదుపరి_ik2 = తదుపరి_kv_తల;
                        స్థిరరాశి uint32_t తదుపరి_iv2 = తదుపరి_kv_తల;

                        స్థిరరాశి dma_addr_t తదుపరి_k_src = k->దత్తాంశం + b0_ప్రారంభం * k->nb[1] + తదుపరి_ik2 * k->nb[2] + ik3 * k->nb[3];
                        dma_వరుస_నెట్టు(dma_q, dma_make_దత్తాంశం(factx.vtcm_k_fp16[0], తదుపరి_k_src), కొలత_k_అడ్డువరుస_padded, k->nb[1], కొలత_k_అడ్డువరుస, b0_అడ్డువరుసలు);

                        స్థిరరాశి dma_addr_t తదుపరి_v_src = v->దత్తాంశం + b0_ప్రారంభం * v->nb[1] + తదుపరి_iv2 * v->nb[2] + iv3 * v->nb[3];
                        dma_వరుస_నెట్టు(dma_q, dma_make_దత్తాంశం(factx.vtcm_v_fp16[0], తదుపరి_v_src), కొలత_v_అడ్డువరుస_padded, v->nb[1], కొలత_v_అడ్డువరుస, b0_అడ్డువరుసలు);

                        ఒకవేళ (factx.pipeline && ముసుగు) {
                            ఒకవేళ (__builtin_expect(factx.ముసుగు_ప్రసారం, true)) {
                                స్థిరరాశి dma_addr_t ms_src = ముసుగు->దత్తాంశం + q_ప్రారంభం * ముసుగు->nb[1] + im3 * ముసుగు->nb[3] + b0_ప్రారంభం * పరిమాణం(__fp16);
                                dma_cache_fa_నెట్టు(dma_q, &factx.m_cache, ms_src, m_పంక్తి_బైట్లు, ముసుగు->nb[1], b0_అడ్డువరుసలు * పరిమాణం(__fp16), n_అడ్డువరుసలు_q);
                            } లేకపోతే {
                                fa_నెట్టు_ముసుగు_dma_gqa(dma_q, ముసుగు, q_ప్రారంభం, im3, b0_ప్రారంభం, తదుపరి_kv_తల, G, m_పంక్తి_బైట్లు, b0_అడ్డువరుసలు, n_అడ్డువరుసలు_q, &factx);
                            }
                        }
                    }
                }

                // ---- Final normalization ----
                {
                    on_పని.o_curr           = o_tile_curr;
                    on_పని.o_prev           = o_tile_prev;
                    on_పని.d_tiles          = factx.vtcm_d_inv_l;
                    on_పని.hmx_scales       = factx.vtcm_hmx_scales_id;
                    on_పని.n_అడ్డువరుస_tiles      = n_అడ్డువరుస_tiles;
                    on_పని.n_అడ్డువరుస_tiles_g_br = n_అడ్డువరుస_tiles_g_br;
                    on_పని.DV               = DV_pad;
                    hmx_వరుస_నెట్టు(ctx->hmx_వరుస, hmx_వరుస_make_desc(hmx_fa_o_norm_కార్మికుడు, &on_పని));
                    hmx_వరుస_తీయి(ctx->hmx_వరుస);
                }

                // ---- Store O block ----
                fa_phase_o_store(&factx, dst, o_tile_curr, q_ప్రారంభం, kv_తల, ib3, n_అడ్డువరుసలు_g);
            }
        }
    }

    తిరిగి HTP_స్థితి_OK;
}

పూర్ణాంకం op_flash_attn_ext(నిర్మాణం htp_ops_context * octx) {
    స్థిరరాశి నిర్మాణం htp_tensor * q    = octx->src[0];
    స్థిరరాశి నిర్మాణం htp_tensor * k    = octx->src[1];
    స్థిరరాశి నిర్మాణం htp_tensor * v    = octx->src[2];
    స్థిరరాశి నిర్మాణం htp_tensor * ముసుగు = octx->src[3];
    స్థిరరాశి నిర్మాణం htp_tensor * dst  = octx->dst;

    // Check support
    ఒకవేళ ((q->type != HTP_TYPE_F16 && q->type != HTP_TYPE_F32) ||
        (k->type != HTP_TYPE_F16 && k->type != HTP_TYPE_Q8_0) ||
        (v->type != HTP_TYPE_F16 && v->type != HTP_TYPE_Q8_0)) {
        తిరిగి HTP_స్థితి_NO_SUPPORT;
    }

    ఒకవేళ (htp_tensor_is_extended(dst)) {
        తిరిగి HTP_స్థితి_NO_SUPPORT;
    }

    స్థిరరాశి నిర్మాణం htp_fa_kernel_params * kparams = (స్థిరరాశి నిర్మాణం htp_fa_kernel_params *) octx->kernel_params;

    ఒకవేళ (kparams->kernel_type == HTP_FA_KERNEL_UNSUPPORTED || kparams->n_kv_ఖండాలు > FA_MAX_KV_ఖండాలు) {
        తిరిగి HTP_స్థితి_NO_SUPPORT;
    }

    ఒకవేళ (!htp_ops_context_సమితి_n_దారాలు(octx, kparams->n_దారాలు)) {
        తిరిగి HTP_స్థితి_INVAL_PARAMS;
    }

    ఒకవేళ (kparams->kernel_type == HTP_FA_KERNEL_HMX) {
        తిరిగి hmx_flash_attn_ext(octx);
    }

    నిర్మాణం htp_fa_context factx;
    factx.octx = octx;
    factx.k = k;
    factx.v = v;

    factx.src0_div21 = kparams->u.hvx.src0_div21;
    factx.src0_div1  = kparams->u.hvx.src0_div1;

    factx.ప్రసారం_rk2 = kparams->ప్రసారం_rk2;
    factx.ప్రసారం_rk3 = kparams->ప్రసారం_rk3;
    factx.ప్రసారం_rv2 = kparams->ప్రసారం_rv2;
    factx.ప్రసారం_rv3 = kparams->ప్రసారం_rv3;

    ఒకవేళ (ముసుగు) {
        factx.src3_div2 = kparams->src3_div2;
        factx.src3_div3 = kparams->src3_div3;
    }

    factx.is_q_fp32 = (q->type == HTP_TYPE_F32);
    factx.కొలత_q_అడ్డువరుస_padded = kparams->u.hvx.కొలత_q_అడ్డువరుస_padded;
    factx.కొలత_k_అడ్డువరుస_padded = kparams->u.hvx.కొలత_k_అడ్డువరుస_padded;
    factx.కొలత_v_అడ్డువరుస_padded = kparams->u.hvx.కొలత_v_అడ్డువరుస_padded;

    factx.n_ఖండాలు = kparams->n_kv_ఖండాలు;

    factx.scale = kparams->scale;
    factx.max_bias = kparams->max_bias;
    factx.has_softcap = (kparams->logit_softcap != 0.0f);
    factx.logit_softcap = 0;
    ఒకవేళ (factx.has_softcap) {
        factx.logit_softcap = (__fp16) kparams->logit_softcap;
    }

    factx.n_తల_log2 = kparams->n_తల_log2;
    factx.m0          = kparams->m0;
    factx.m1          = kparams->m1;

    స్థిరరాశి uint32_t n_తల = q->ne[2];
    ఒకవేళ (n_తల > 512) {
        తిరిగి HTP_స్థితి_NO_SUPPORT;
    }
    కొరకు (uint32_t h = 0; h < n_తల; ++h) {
        factx.slopes[h] = (__fp16) ((kparams->max_bias > 0.0f) ? alibi_వాలు(h, factx.n_తల_log2, factx.m0, factx.m1) : 1.0f);
    }

    // total rows in q
    స్థిరరాశి uint32_t neq1 = q->ne[1];
    స్థిరరాశి uint32_t neq2 = q->ne[2];
    స్థిరరాశి uint32_t neq3 = q->ne[3];
    స్థిరరాశి uint32_t మొత్తం_qrows = neq1 * neq2 * neq3;

    uint32_t qrow_ప్రారంభం = 0;
    uint32_t qrows      = మొత్తం_qrows;

    ఒకవేళ (octx->ctx->mdev.లెక్క > 1) {
        స్థిరరాశి uint32_t mdev_లెక్క = octx->ctx->mdev.లెక్క;
        స్థిరరాశి uint32_t mdev_idx   = octx->ctx->mdev.idx;
        స్థిరరాశి uint32_t n_kv_heads = k->ne[2];
        స్థిరరాశి uint32_t dst_e_కొలత = (dst->type == HTP_TYPE_F32) ? పరిమాణం(దశాంశం) : పరిమాణం(__fp16);
        స్థిరరాశి bool can_విభజించు      = htp_tensor_can_అడ్డువరుస_విభజన(dst, dst_e_కొలత);

        // head range is contiguous in flat row space only when neq3 == 1
        ఒకవేళ (kparams->తల_విభజించు && can_విభజించు && neq3 == 1 && n_kv_heads >= mdev_లెక్క && n_kv_heads % mdev_లెక్క == 0) {
            స్థిరరాశి uint32_t G              = kparams->G;
            స్థిరరాశి uint32_t kv_చొప్పున_core    = n_kv_heads / mdev_లెక్క;
            స్థిరరాశి uint32_t heads_చొప్పున_core = kv_చొప్పున_core * G;
            స్థిరరాశి uint32_t తల_ప్రారంభం     = mdev_idx * heads_చొప్పున_core;
            qrow_ప్రారంభం = తల_ప్రారంభం * neq1;
            qrows      = heads_చొప్పున_core * neq1;
        } లేకపోతే {
            స్థిరరాశి నిర్మాణం htp_tensor_mdev_range range = htp_tensor_mdev_విభజన(మొత్తం_qrows, can_విభజించు ? 1 : 0, mdev_idx, mdev_లెక్క, &octx->ctx->mdev.లెక్క_div);
            qrow_ప్రారంభం = range.ప్రారంభం;
            qrows      = range.లెక్క;
        }
    }

    ఒకవేళ (qrows == 0) {
        తిరిగి HTP_స్థితి_OK;
    }

    స్థిరరాశి uint32_t n_దారాలు = octx->n_దారాలు;

    factx.qrows            = qrows;
    factx.qrow_ప్రారంభం       = qrow_ప్రారంభం;
    factx.qrows_చొప్పున_thread = fastdiv(qrows + n_దారాలు - 1, &octx->n_దారాలు_div);

    స్థిరరాశి bool has_ముసుగు  = (ముసుగు != ఖాళీ);
    స్థిరరాశి bool has_sinks = (octx->src[4] != ఖాళీ);
    నిర్మాణం hvx_fa_vtcm_layout L;
    hvx_fa_vtcm_layout_build(&L, k->ne[0], v->ne[0], factx.is_q_fp32, has_ముసుగు, has_sinks, n_తల, n_దారాలు);

    ఒకవేళ (L.మొత్తం_బైట్లు > octx->ctx->vtcm_కొలత) {
        తిరిగి HTP_స్థితి_VTCM_TOO_SMALL;
    }

    factx.కొలత_q_ఖండం = L.కొలత_q_ఖండం;
    factx.కొలత_k_ఖండం = L.కొలత_k_ఖండం;
    factx.కొలత_v_ఖండం = L.కొలత_v_ఖండం;
    factx.కొలత_m_ఖండం = L.కొలత_m_ఖండం;
    factx.కొలత_vkq_acc = L.కొలత_vkq_acc;

    uint8_t * స్థిరరాశి పునాది = octx->ctx->vtcm_పునాది;

    factx.spad_q               = VTCM_LAYOUT_PTR(uint8_t, పునాది, L.off_q);
    factx.spad_k               = VTCM_LAYOUT_PTR(uint8_t, పునాది, L.off_k);
    factx.spad_v               = VTCM_LAYOUT_PTR(uint8_t, పునాది, L.off_v);
    factx.spad_m               = VTCM_LAYOUT_PTR_OPTIONAL(uint8_t, పునాది, L.off_m, has_ముసుగు);
    factx.spad_a               = VTCM_LAYOUT_PTR(uint8_t, పునాది, L.off_a);
    factx.spad_sinks           = VTCM_LAYOUT_PTR_OPTIONAL(దశాంశం, పునాది, L.off_sinks, has_sinks);
    factx.n_చురుకైన_ఖండాలు      = factx.n_ఖండాలు;
    factx.shared_చురుకైన_ఖండాలు = false;

    ఒకవేళ (has_ముసుగు && neq1 == 1 && neq3 == 1 && ముసుగు->ne[2] == 1 && factx.n_ఖండాలు > 1 && factx.n_ఖండాలు <= FA_MAX_KV_ఖండాలు) {
        factx.n_చురుకైన_ఖండాలు = fa_పరిశీలన_చురుకైన_ఖండాలు(
            octx->ctx->dma[0], ముసుగు, 0, 1, 0, 0, k->ne[1], FLASH_ATTN_ఖండం_కొలత,
            factx.n_ఖండాలు, factx.spad_k, factx.కొలత_k_ఖండం * 2, factx.చురుకైన_ఖండాలు);
        factx.shared_చురుకైన_ఖండాలు = true;
    }

    ఒకవేళ (has_sinks) {
        స్థిరరాశి నిర్మాణం htp_tensor * sinks = octx->src[4];
        dma_వరుస * dma_q = octx->ctx->dma[0];
        dma_వరుస_నెట్టు(dma_q, dma_make_దత్తాంశం(factx.spad_sinks, sinks->దత్తాంశం), L.కొలత_sinks, 0, sinks->కొలత, 1);
        dma_వరుస_తీయి(dma_q);
    }

    work_వరుస_నిరంతరం(octx->ctx->work_వరుస, flash_attn_ext_f16_thread, &factx, n_దారాలు);

    తిరిగి HTP_స్థితి_OK;
}
