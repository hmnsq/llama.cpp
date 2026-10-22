#निर्देशः clang diagnostic ignored "-Wunused-variable"
#निर्देशः clang diagnostic ignored "-Wunused-function"
#निर्देशः clang diagnostic ignored "-Wunused-but-set-variable"

#अन्तर्भाव <assert.h>
#अन्तर्भाव <HAP_compute_res.h>
#अन्तर्भाव <HAP_farf.h>
#अन्तर्भाव <math.h>
#अन्तर्भाव <stdbool.h>
#अन्तर्भाव <stdatomic.h>
#अन्तर्भाव <stddef.h>
#अन्तर्भाव <stdint.h>
#अन्तर्भाव <string.h>

#अन्तर्भाव "dma-queue.h"
#अन्तर्भाव "hex-fastdiv.h"
#अन्तर्भाव "hex-profile.h"
#अन्तर्भाव "hmx-queue.h"
#अन्तर्भाव "hmx-utils.h"
#अन्तर्भाव "hvx-utils.h"
#अन्तर्भाव "hvx-dump.h"
#अन्तर्भाव "hvx-copy.h"
#अन्तर्भाव "hvx-reduce.h"
#अन्तर्भाव "hvx-flash-attn.h"
#अन्तर्भाव "htp-vtcm.h"
#अन्तर्भाव "work-queue.h"

#परिभाषय GGML_COMMON_DECL_C
#अन्तर्भाव "ggml-common.h"
#अन्तर्भाव "htp-ctx.h"
#अन्तर्भाव "htp-ops.h"
#अन्तर्भाव "htp-tensor.h"
#अन्तर्भाव "hvx-quant.h"

#अन्तर्भाव "flash-attn-ops.h"
#अन्तर्भाव "hvx-fa-kernels.h"
#अन्तर्भाव "hmx-fa-kernels.h"

// Must be multiple of 32
#परिभाषय FLASH_ATTN_खण्डम्_आकार (32 * 2)

संरचना htp_fa_context {
    ध्रुव संरचना htp_ops_context * octx;

    संरचना fastdiv_मानानि src0_div21;
    संरचना fastdiv_मानानि src0_div1;

    संरचना fastdiv_मानानि प्रसारणम्_rk2;
    संरचना fastdiv_मानानि प्रसारणम्_rk3;
    संरचना fastdiv_मानानि प्रसारणम्_rv2;
    संरचना fastdiv_मानानि प्रसारणम्_rv3;

    संरचना fastdiv_मानानि src3_div2;
    संरचना fastdiv_मानानि src3_div3;

    भिन्न scale;
    भिन्न max_bias;
    bool  has_softcap;
    __fp16 logit_softcap;

    uint32_t n_शीर्ष_log2;
    भिन्न m0;
    भिन्न m1;
    __fp16 slopes[512];

    uint32_t n_खण्डानि;
    uint32_t n_सक्रिय_खण्डानि;
    bool     shared_सक्रिय_खण्डानि;

    size_t आकार_q_row_padded;
    size_t आकार_k_row_padded;
    size_t आकार_v_row_padded;

    size_t आकार_k_खण्डम्;
    size_t आकार_v_खण्डम्;
    size_t आकार_m_खण्डम्;

    uint32_t qrows;
    uint32_t qrows_प्रत्येकम्_thread;
    uint32_t qrow_start;

    bool is_q_fp32;

    size_t आकार_q_खण्डम्;
    size_t आकार_vkq_acc;

    uint8_t *  spad_q;
    uint8_t *  spad_k;
    uint8_t *  spad_v;
    uint8_t *  spad_m;
    uint8_t *  spad_a;
    भिन्न *    spad_sinks;
    uint16_t   सक्रिय_खण्डानि[FA_MAX_KV_खण्डानि];

    ध्रुव संरचना htp_tensor * k;
    ध्रुव संरचना htp_tensor * v;
};

संरचना hmx_fa_context {
    ध्रुव संरचना htp_ops_context * octx;
    ध्रुव संरचना htp_tensor *      sinks;  // attention sinks (src[4]), NULL if absent
    ध्रुव संरचना htp_tensor *      k;
    ध्रुव संरचना htp_tensor *      v;
    bool         pipeline;  // true when n_kv_blocks >= FA_MIN_KV_BLOCKS && n_threads >= 2
    uint32_t     n_तन्तवः;

    // Op parameters
    __fp16       scale;
    भिन्न        max_bias;
    bool         has_softcap;
    __fp16       logit_softcap;
    uint32_t     n_शीर्ष_log2;
    भिन्न        m0, m1;

    // Dimensions
    uint32_t     DK, DV;
    uint32_t     DK_pad, DV_pad;  // head_dim rounded up to 64 for HMX tiling
    uint32_t     n_kv;        // kv_len
    uint32_t     n_kv_heads;  // number of KV heads
    uint32_t     n_heads;     // number of Q heads
    uint32_t     G;           // GQA factor = n_heads / n_kv_heads
    संरचना fastdiv_मानानि div_G;
    संरचना fastdiv_मानानि src3_div2;
    संरचना fastdiv_मानानि src3_div3;
    uint32_t     n_kv_खण्डानि;
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
    HVX_सदिशम् * vtcm_m_vec;           // Row max [g_br]
    HVX_सदिशम् * vtcm_l_vec;           // Row sum [g_br]
    HVX_सदिशम् * vtcm_s_rowmax;        // Softmax intermediate [g_br]
    HVX_सदिशम् * vtcm_p_rowsum;        // Softmax intermediate [g_br]
    HVX_सदिशम् * vtcm_row_bufs;        // Per-thread softmax row scratch [n_threads][2][Bc/64]
    uint8_t *    vtcm_hmx_scales_id;   // HMX output scales (identity)
    uint8_t *    vtcm_hmx_scales_qk;   // HMX output scales (qk_scale)
    __fp16 *     vtcm_आवरणम्_buf;        // VTCM mask buffer [Br * m_line], DMA'd per KV block
    __fp16 *     vtcm_slopes;          // ALiBi slopes [g_br]
    भिन्न *      vtcm_sinks;           // Attention sinks
    size_t       row_buf_stride;       // HVX vectors per row buffer (Bc/64)
    size_t       आवरणम्_buf_row_stride;  // elements (__fp16) per row in mask buffer
    size_t       q_tile_अष्टकानि;
    size_t       k_tile_अष्टकानि;
    size_t       o_tile_अष्टकानि;
    size_t       col_vec_अष्टकानि;
    size_t       d_tile_अष्टकानि;
    bool         आवरणम्_प्रसारणम्;       // true when mask->ne[2] == 1 (head-independent, single 2D DMA)
    dma_cache_fa m_cache;
};

// Scan mask rows [q_start, q_start + n_rows_q) and build compact list of KV blocks with any entry != -inf.
स्थिर uint32_t fa_अवलोकनम्_सक्रिय_खण्डानि(
        dma_queue *               dma_q,
        ध्रुव संरचना htp_tensor * आवरणम्,
        uint32_t                  q_start,
        uint32_t                  n_पङ्क्तिकाः_q,
        uint32_t                  im2,
        uint32_t                  im3,
        uint32_t                  nek1,
        uint32_t                  Bc,
        uint32_t                  n_kv_खण्डानि,
        शून्य *                    vtcm_scratch,
        size_t                    scratch_अष्टकानि,
        uint16_t *                सक्रिय_खण्डानि) {
    ध्रुव size_t row_अष्टकानि  = (size_t) nek1 * परिमाण(__fp16);
    ध्रुव size_t row_stride = hex_आवर्तनम्_ऊर्ध्वम्(row_अष्टकानि, 128);

    यदि (!आवरणम् || n_kv_खण्डानि <= 1 || n_kv_खण्डानि > FA_MAX_KV_खण्डानि ||
        (size_t) n_पङ्क्तिकाः_q * row_stride > scratch_अष्टकानि) {
        #निर्देशः clang loop vectorize(disable)
        प्रति (uint32_t b = 0; b < n_kv_खण्डानि; ++b) {
            सक्रिय_खण्डानि[b] = (uint16_t) b;
        }
        प्रत्यागम n_kv_खण्डानि;
    }

    ध्रुव dma_addr_t m_src = आवरणम्->दत्त + q_start * आवरणम्->nb[1] + im2 * आवरणम्->nb[2] + im3 * आवरणम्->nb[3];
    dma_queue_प्रक्षिप(dma_q, dma_make_दत्त(vtcm_scratch, m_src), row_stride, आवरणम्->nb[1], row_अष्टकानि, n_पङ्क्तिकाः_q);
    ध्रुव uint8_t * m_buf = (ध्रुव uint8_t *) dma_queue_निष्कासय(dma_q).dst;

    ध्रुव HVX_सदिशम् v_ninf   = Q6_Vh_vsplat_R(0xFC00);
    uint32_t         n_सक्रिय = 0;

    प्रति (uint32_t b = 0; b < n_kv_खण्डानि; ++b) {
        ध्रुव uint32_t col0 = b * Bc;
        ध्रुव uint32_t स्तम्भसङ्ख्या_2 = MIN(Bc, nek1 - col0);

        प्रति (uint32_t r = 0; r < n_पङ्क्तिकाः_q; ++r) {
            ध्रुव __fp16 * row_ptr = (ध्रुव __fp16 *) (m_buf + r * row_stride) + col0;
            HVX_सदिशम् v_any = Q6_V_vzero();
            uint32_t c = 0;
            प्रति (; c + 64 <= स्तम्भसङ्ख्या_2; c += 64) {
                ध्रुव HVX_सदिशम् v_m = *(ध्रुव HVX_सदिशम् *) (row_ptr + c);
                v_any = Q6_V_vor_VV(v_any, Q6_V_vxor_VV(v_m, v_ninf));
            }
            यदि (c < स्तम्भसङ्ख्या_2) {
                ध्रुव uint32_t rem = स्तम्भसङ्ख्या_2 - c;
                ध्रुव HVX_सदिशम्_Pred q_rem = Q6_Q_vsetq2_R(rem * परिमाण(__fp16));
                ध्रुव HVX_सदिशम् v_m = *(ध्रुव HVX_सदिशम् *) (row_ptr + c);
                ध्रुव HVX_सदिशम् v_diff = Q6_V_vand_QV(q_rem, Q6_V_vxor_VV(v_m, v_ninf));
                v_any = Q6_V_vor_VV(v_any, v_diff);
            }
            v_any = Q6_V_vor_VV(v_any, Q6_V_vror_VR(v_any, 64));
            v_any = Q6_V_vor_VV(v_any, Q6_V_vror_VR(v_any, 32));
            v_any = Q6_V_vor_VV(v_any, Q6_V_vror_VR(v_any, 16));
            v_any = Q6_V_vor_VV(v_any, Q6_V_vror_VR(v_any, 8));
            v_any = Q6_V_vor_VV(v_any, Q6_V_vror_VR(v_any, 4));
            यदि (Q6_R_vextract_VR(v_any, 0) != 0) {
                सक्रिय_खण्डानि[n_सक्रिय++] = (uint16_t) b;
                भङ्ग;
            }
        }
    }

    यदि (n_सक्रिय == 0 && n_kv_खण्डानि > 0) {
        सक्रिय_खण्डानि[0] = 0;
        n_सक्रिय = 1;
    }
    प्रत्यागम n_सक्रिय;
}

स्थिर शून्य flash_attn_ext_f16_thread(अचिह्नित पूर्णाङ्क nth, अचिह्नित पूर्णाङ्क ith, शून्य * दत्त) {
    संरचना htp_fa_context * factx = (संरचना htp_fa_context *) दत्त;
    ध्रुव संरचना htp_ops_context * octx = factx->octx;
    ध्रुव संरचना htp_tensor * q     = octx->src[0];
    ध्रुव संरचना htp_tensor * k     = octx->src[1];
    ध्रुव संरचना htp_tensor * v     = octx->src[2];
    ध्रुव संरचना htp_tensor * आवरणम्  = octx->src[3];
    ध्रुव संरचना htp_tensor * sinks = octx->src[4];
    ध्रुव संरचना htp_tensor * dst   = octx->dst;

    ध्रुव uint32_t neq0 = q->ne[0];
    ध्रुव uint32_t neq1 = q->ne[1];
    ध्रुव uint32_t neq2 = q->ne[2];
    ध्रुव uint32_t neq3 = q->ne[3];

    ध्रुव uint32_t nek0 = k->ne[0];
    ध्रुव uint32_t nek1 = k->ne[1];
    ध्रुव uint32_t nek2 = k->ne[2];
    ध्रुव uint32_t nek3 = k->ne[3];

    ध्रुव uint32_t nev0 = v->ne[0];
    ध्रुव uint32_t nev1 = v->ne[1];
    ध्रुव uint32_t nev2 = v->ne[2];
    ध्रुव uint32_t nev3 = v->ne[3];

    ध्रुव uint32_t nbq1 = q->nb[1];
    ध्रुव uint32_t nbq2 = q->nb[2];
    ध्रुव uint32_t nbq3 = q->nb[3];

    ध्रुव uint32_t nbk1 = k->nb[1];
    ध्रुव uint32_t nbk2 = k->nb[2];
    ध्रुव uint32_t nbk3 = k->nb[3];

    ध्रुव uint32_t nbv1 = v->nb[1];
    ध्रुव uint32_t nbv2 = v->nb[2];
    ध्रुव uint32_t nbv3 = v->nb[3];

    ध्रुव uint32_t ne1 = dst->ne[1];
    ध्रुव uint32_t ne2 = dst->ne[2];
    ध्रुव uint32_t ne3 = dst->ne[3];

    ध्रुव uint32_t nb1 = dst->nb[1];
    ध्रुव uint32_t nb2 = dst->nb[2];
    ध्रुव uint32_t nb3 = dst->nb[3];

    // total rows in q
    ध्रुव uint32_t dr  = factx->qrows_प्रत्येकम्_thread;
    ध्रुव uint32_t ir0 = factx->qrow_start + dr * ith;
    ध्रुव uint32_t ir1 = MIN(ir0 + dr, factx->qrow_start + factx->qrows);

    यदि (ir0 >= ir1) प्रत्यागम;

    संरचना htp_thread_trace * tr = &octx->ctx->trace[ith];

    dma_queue * dma_q = octx->ctx->dma[ith];

    ध्रुव uint32_t DK = nek0;
    ध्रुव uint32_t DV = nev0;

    ध्रुव size_t आकार_q_row = DK * ((q->type == HTP_TYPE_F32) ? 4 : 2);
    ध्रुव size_t आकार_k_row = htp_tensor_get_row_आकार(k->type, DK);
    ध्रुव size_t आकार_v_row = htp_tensor_get_row_आकार(v->type, DV);

    // Scratchpad buffers for Q, K, V, Mask, and VKQ32 accumulator
    uint8_t * spad_q = factx->spad_q + factx->आकार_q_खण्डम् * ith;
    uint8_t * spad_k = factx->spad_k + factx->आकार_k_खण्डम् * 2 * ith;
    uint8_t * spad_v = factx->spad_v + factx->आकार_v_खण्डम् * 2 * ith;
    uint8_t * spad_m = factx->spad_m + (आवरणम् ? factx->आकार_m_खण्डम् * HVX_FA_DMA_CACHE_आकार : 0) * ith;
    uint8_t * spad_a = factx->spad_a + factx->आकार_vkq_acc * ith;

    dma_cache_dm m_cache;
    dma_cache_dm_init(&m_cache, spad_m, factx->आकार_m_खण्डम्, HVX_FA_DMA_CACHE_आकार);

    uint16_t thread_सक्रिय_खण्डानि[FA_MAX_KV_खण्डानि];
    ध्रुव uint16_t * सक्रिय_खण्डानि = factx->shared_सक्रिय_खण्डानि ? factx->सक्रिय_खण्डानि : thread_सक्रिय_खण्डानि;
    uint32_t n_सक्रिय_खण्डानि       = factx->n_सक्रिय_खण्डानि;
    dma_addr_t पूर्व_mp_आधारः        = 0;

    यदि (!factx->shared_सक्रिय_खण्डानि && !आवरणम्) {
        #निर्देशः clang loop vectorize(disable)
        प्रति (uint32_t b = 0; b < factx->n_खण्डानि; ++b) {
            thread_सक्रिय_खण्डानि[b] = (uint16_t) b;
        }
    }

    ध्रुव size_t आकार_vkq_acc_single = hex_आवर्तनम्_ऊर्ध्वम्(DV * परिमाण(भिन्न), 128);

    uint32_t ir = ir0;
    यावत् (ir < ir1) {
        ध्रुव uint32_t iq3 = fastdiv(ir, &factx->src0_div21);
        ध्रुव uint32_t iq2 = fastdiv(ir - iq3*neq2*neq1, &factx->src0_div1);
        ध्रुव uint32_t iq1 = (ir - iq3*neq2*neq1 - iq2 * neq1);

        ध्रुव uint32_t ik3 = fastdiv(iq3, &factx->प्रसारणम्_rk3);
        ध्रुव uint32_t ik2 = fastdiv(iq2, &factx->प्रसारणम्_rk2);

        ध्रुव uint32_t iv3 = fastdiv(iq3, &factx->प्रसारणम्_rv3);
        ध्रुव uint32_t iv2 = fastdiv(iq2, &factx->प्रसारणम्_rv2);

        uint32_t G_local = 1;
        यदि (neq1 == 1 && (आवरणम् == रिक्त || आवरणम्->ne[2] == 1)) {
            यावत् (ir + G_local < ir1 && G_local < FA_HVX_G_MAX) {
                ध्रुव uint32_t अग्र_ir = ir + G_local;
                ध्रुव uint32_t अग्र_iq3 = fastdiv(अग्र_ir, &factx->src0_div21);
                ध्रुव uint32_t अग्र_iq2 = fastdiv(अग्र_ir - अग्र_iq3*neq2*neq1, &factx->src0_div1);
                ध्रुव uint32_t अग्र_iq1 = (अग्र_ir - अग्र_iq3*neq2*neq1 - अग्र_iq2 * neq1);

                ध्रुव uint32_t अग्र_ik3 = fastdiv(अग्र_iq3, &factx->प्रसारणम्_rk3);
                ध्रुव uint32_t अग्र_ik2 = fastdiv(अग्र_iq2, &factx->प्रसारणम्_rk2);

                ध्रुव uint32_t अग्र_iv3 = fastdiv(अग्र_iq3, &factx->प्रसारणम्_rv3);
                ध्रुव uint32_t अग्र_iv2 = fastdiv(अग्र_iq2, &factx->प्रसारणम्_rv2);

                यदि (अग्र_ik2 != ik2 || अग्र_ik3 != ik3 || अग्र_iv2 != iv2 || अग्र_iv3 != iv3 || अग्र_iq1 != iq1 || अग्र_iq3 != iq3) {
                    भङ्ग;
                }
                G_local++;
            }
        }

        dma_addr_t mp_आधारः = 0;
        यदि (आवरणम्) {
            ध्रुव uint32_t im2 = fastmodulo(iq2, आवरणम्->ne[2], &factx->src3_div2);
            ध्रुव uint32_t im3 = fastmodulo(iq3, आवरणम्->ne[3], &factx->src3_div3);
            mp_आधारः = आवरणम्->दत्त + iq1*आवरणम्->nb[1] + im2*आवरणम्->nb[2] + im3*आवरणम्->nb[3];
            यदि (!factx->shared_सक्रिय_खण्डानि && mp_आधारः != पूर्व_mp_आधारः) {
                n_सक्रिय_खण्डानि = fa_अवलोकनम्_सक्रिय_खण्डानि(
                    dma_q, आवरणम्, iq1, 1, im2, im3, nek1, FLASH_ATTN_खण्डम्_आकार,
                    factx->n_खण्डानि, spad_k, factx->आकार_k_खण्डम् * 2, thread_सक्रिय_खण्डानि);
                पूर्व_mp_आधारः = mp_आधारः;
            }
        }

        uint32_t heads[FA_HVX_G_MAX];
        HVX_सदिशम् नतिः_vecs[FA_HVX_G_MAX] __attribute__((aligned(128)));
        HVX_सदिशम् S_vec[FA_HVX_G_MAX]      __attribute__((aligned(128)));
        HVX_सदिशम् M_vec[FA_HVX_G_MAX]      __attribute__((aligned(128)));
        uint8_t * q_ptrs[FA_HVX_G_MAX];
        भिन्न * vkq_ptrs[FA_HVX_G_MAX];

        प्रति (uint32_t g = 0; g < G_local; ++g) {
            ध्रुव uint32_t r = ir + g;
            ध्रुव uint32_t r_iq3 = fastdiv(r, &factx->src0_div21);
            ध्रुव uint32_t r_iq2 = fastdiv(r - r_iq3*neq2*neq1, &factx->src0_div1);
            ध्रुव uint32_t r_iq1 = (r - r_iq3*neq2*neq1 - r_iq2 * neq1);

            heads[g] = r_iq2;
            ध्रुव __fp16 नतिः = factx->slopes[r_iq2];
            नतिः_vecs[g] = hvx_vec_splat_f16(नतिः);

            S_vec[g] = hvx_vec_splat_f32(0.0f);
            M_vec[g] = hvx_vec_splat_f32(HTP_FA_M_INITIAL_VAL);

            uint8_t * q_dst = spad_q + g * factx->आकार_q_row_padded;
            q_ptrs[g] = q_dst;

            भिन्न * vkq_dst = (भिन्न *)(spad_a + g * आकार_vkq_acc_single);
            vkq_ptrs[g] = vkq_dst;
            hvx_splat_f32_a((uint8_t *) vkq_dst, 0, DV);

            // Fetch Q row g
            ध्रुव dma_addr_t q_row_ptr = q->दत्त + r_iq1*nbq1 + r_iq2*nbq2 + r_iq3*nbq3;
            dma_queue_प्रक्षिप(dma_q, dma_make_दत्त(q_dst, q_row_ptr), factx->आकार_q_row_padded, nbq1, आकार_q_row, 1);
        }

        // Prefetch first two active blocks
        प्रति (uint32_t चरणः = 0; चरणः < MIN(n_सक्रिय_खण्डानि, 2); ++चरणः) {
            ध्रुव uint32_t ib = सक्रिय_खण्डानि[चरणः];
            ध्रुव uint32_t ic_start = ib * FLASH_ATTN_खण्डम्_आकार;
            ध्रुव uint32_t current_खण्डम्_आकार = MIN(FLASH_ATTN_खण्डम्_आकार, nek1 - ic_start);

            // K
            ध्रुव dma_addr_t k_src = k->दत्त + ic_start*nbk1 + ik2*nbk2 + ik3*nbk3;
            uint8_t * k_dst = spad_k + (चरणः % 2) * factx->आकार_k_खण्डम्;
            dma_queue_प्रक्षिप(dma_q, dma_make_दत्त(k_dst, k_src), factx->आकार_k_row_padded, nbk1, आकार_k_row, current_खण्डम्_आकार);

            // V
            ध्रुव dma_addr_t v_src = v->दत्त + ic_start*nbv1 + iv2*nbv2 + iv3*nbv3;
            uint8_t * v_dst = spad_v + (चरणः % 2) * factx->आकार_v_खण्डम्;
            dma_queue_प्रक्षिप(dma_q, dma_make_दत्त(v_dst, v_src), factx->आकार_v_row_padded, nbv1, आकार_v_row, current_खण्डम्_आकार);

            // Mask
            यदि (आवरणम्) {
                ध्रुव dma_addr_t m_src = mp_आधारः + ic_start * परिमाण(__fp16);
                dma_cache_dm_प्रक्षिप(dma_q, &m_cache, ib, m_src, current_खण्डम्_आकार * 2, current_खण्डम्_आकार * 2, current_खण्डम्_आकार * 2, 1);
            }
        }

        // Pop all Q rows
        प्रति (uint32_t g = 0; g < G_local; ++g) {
            uint8_t * q_ptr_vtcm = (शून्य *) dma_queue_निष्कासय(dma_q).dst;
            यदि (factx->is_q_fp32) {
                hvx_copy_f16_f32_aa(q_ptr_vtcm, q_ptr_vtcm, DK);
            }
        }

        ध्रुव HVX_सदिशम् v_neg_inf = Q6_Vh_vsplat_R(0xfbff);
        ध्रुव bool has_softcap     = factx->has_softcap;
        ध्रुव HVX_सदिशम् v_cap     = has_softcap ? hvx_vec_splat_f16(factx->logit_softcap) : Q6_V_vzero();
        ध्रुव HVX_सदिशम् vinf      = Q6_Vh_vsplat_R(0xFC00);
        ध्रुव HVX_सदिशम् vmin      = Q6_Vh_vsplat_R(0xFBFF);
        ध्रुव HVX_सदिशम् v_log2e   = hvx_vec_splat_f16(EXP_LOG2E_F);
        ध्रुव uint32_t stride_v2   = factx->आकार_v_row_padded * 2;

        प्रति (uint32_t चरणः = 0; चरणः < n_सक्रिय_खण्डानि; ++चरणः) {
            ध्रुव uint32_t ib = सक्रिय_खण्डानि[चरणः];
            ध्रुव uint32_t ic_start = ib * FLASH_ATTN_खण्डम्_आकार;
            ध्रुव uint32_t current_खण्डम्_आकार = MIN(FLASH_ATTN_खण्डम्_आकार, nek1 - ic_start);

            // Wait for DMA
            uint8_t * k_आधारः = (शून्य *) dma_queue_निष्कासय(dma_q).dst; // K
            uint8_t * v_आधारः = (शून्य *) dma_queue_निष्कासय(dma_q).dst; // V
            __fp16  * m_आधारः = आवरणम् ? (__fp16 *) dma_queue_निष्कासय(dma_q).dst : रिक्त; // M

            यदि (factx->k->type == HTP_TYPE_Q8_0) {
                htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_FA_K_PREP, ir);
                प्रति (uint32_t r = 0; r < current_खण्डम्_आकार; ++r) {
                    __fp16 * row_k = (__fp16 *)(k_आधारः + r * factx->आकार_k_row_padded);
                    hvx_dequantize_row_flat_q8_0_f16(row_k, row_k, DK);
                }
                htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_FA_K_PREP, ir);
            }
            यदि (factx->v->type == HTP_TYPE_Q8_0) {
                htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_FA_V_PREP, ir);
                प्रति (uint32_t r = 0; r < current_खण्डम्_आकार; ++r) {
                    __fp16 * row_v = (__fp16 *)(v_आधारः + r * factx->आकार_v_row_padded);
                    hvx_dequantize_row_flat_q8_0_f16(row_v, row_v, DV);
                }
                htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_FA_V_PREP, ir);
            }

            प्रति (uint32_t g = 0; g < G_local; ++g) {
                ध्रुव uint32_t शीर्ष_ir = ir + g;
                uint8_t * q_ptr_vtcm = q_ptrs[g];
                भिन्न * VKQ32 = vkq_ptrs[g];
                ध्रुव HVX_सदिशम् नतिः_vec = नतिः_vecs[g];

                htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_FA_QK, शीर्ष_ir);

                HVX_सदिशम् scores_f16 = Q6_V_vzero();
                यदि (current_खण्डम्_आकार > 0) {
                    HVX_सदिशम् scores0 = hvx_dot_f16_f16_aa_rx32(q_ptr_vtcm, k_आधारः, factx->आकार_k_row_padded, DK, factx->scale);
                    HVX_सदिशम् scores1 = (current_खण्डम्_आकार > 32) ? hvx_dot_f16_f16_aa_rx32(q_ptr_vtcm, k_आधारः + 32 * factx->आकार_k_row_padded, factx->आकार_k_row_padded, DK, factx->scale) : Q6_V_vzero();
                    scores_f16 = hvx_vec_f32_गन्तव्यम्_f16(scores0, scores1);
                }

                यदि (has_softcap) {
                    scores_f16 = hvx_vec_tanh_f16(scores_f16);
                    scores_f16 = hvx_vec_mul_f16_f16(scores_f16, v_cap);
                }

                HVX_सदिशम्_Pred q_पुच्छ_keep = Q6_Q_vsetq2_R(current_खण्डम्_आकार * परिमाण(__fp16));

                यदि (आवरणम्) {
                    HVX_सदिशम् m_vals_f16 = *(ध्रुव HVX_UVector *) m_आधारः;
                    HVX_सदिशम्_Pred is_inf = Q6_Q_vcmp_eq_VhVh(m_vals_f16, vinf);
                    m_vals_f16 = Q6_V_vmux_QVV(is_inf, vmin, m_vals_f16);

                    HVX_सदिशम् m_scaled = hvx_vec_mul_f16_f16(m_vals_f16, नतिः_vec);
                    scores_f16 = Q6_V_vmux_QVV(q_पुच्छ_keep, hvx_vec_सङ्कलय_f16_f16(scores_f16, m_scaled), v_neg_inf);
                } अन्यथा {
                    scores_f16 = Q6_V_vmux_QVV(q_पुच्छ_keep, scores_f16, v_neg_inf);
                }

                HVX_सदिशम् v_max_f16 = hvx_vec_reduce_max_f16(scores_f16);
                HVX_सदिशम् v_max     = Q6_V_lo_W(hvx_vec_f16_गन्तव्यम्_f32(v_max_f16));
                htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_FA_QK, शीर्ष_ir);

                // prefetch K for step + 2 after last head finished QK
                यदि (g + 1 == G_local && चरणः + 2 < n_सक्रिय_खण्डानि) {
                    ध्रुव uint32_t अग्र_ib = सक्रिय_खण्डानि[चरणः + 2];
                    ध्रुव uint32_t अग्र_ic_start = अग्र_ib * FLASH_ATTN_खण्डम्_आकार;
                    ध्रुव uint32_t अग्र_खण्डम्_आकार = MIN(FLASH_ATTN_खण्डम्_आकार, nek1 - अग्र_ic_start);

                    ध्रुव dma_addr_t k_src = k->दत्त + अग्र_ic_start*nbk1 + ik2*nbk2 + ik3*nbk3;
                    dma_queue_प्रक्षिप(dma_q, dma_make_दत्त(k_आधारः, k_src), factx->आकार_k_row_padded, nbk1, आकार_k_row, अग्र_खण्डम्_आकार);
                }

                htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_FA_SFM, शीर्ष_ir);
                {
                    HVX_सदिशम् M_new_vec = Q6_Vsf_vmax_VsfVsf(v_max, M_vec[g]);
                    HVX_सदिशम् diff_vec  = HVX_OP_SUB_F32(M_vec[g], M_new_vec);

                    HVX_सदिशम् diff_f16   = hvx_vec_f32_गन्तव्यम्_f16(diff_vec, diff_vec);
                    HVX_सदिशम् diff_base2 = hvx_vec_mul_f16_f16(diff_f16, v_log2e);
                    HVX_सदिशम् ms_f16     = hvx_vec_exp2_f16(diff_base2);
                    HVX_सदिशम् ms_vec     = Q6_V_lo_W(hvx_vec_f16_गन्तव्यम्_f32(ms_f16));

                    M_vec[g] = M_new_vec;

                    HVX_सदिशम् v_m_vec_f16 = hvx_vec_f32_गन्तव्यम्_f16(M_vec[g], M_vec[g]);
                    HVX_सदिशम् v_s_minus_m = Q6_Vqf16_vsub_VhfVhf(scores_f16, v_m_vec_f16);
                    HVX_सदिशम् v_s_minus_m_base2 = hvx_vec_mul_f16_f16(Q6_Vhf_equals_Vqf16(v_s_minus_m), v_log2e);

                    HVX_सदिशम् P = hvx_vec_exp2_f16(v_s_minus_m_base2);
                    P = Q6_V_vmux_QVV(q_पुच्छ_keep, P, Q6_V_vzero());

                    HVX_सदिशम्_Pair P_pair = hvx_vec_f16_गन्तव्यम्_f32(P);
                    HVX_सदिशम् P0 = Q6_V_lo_W(P_pair);
                    HVX_सदिशम् P1 = Q6_V_hi_W(P_pair);
                    HVX_सदिशम् p_योग_vec = hvx_vec_reduce_योग_f32(HVX_OP_सङ्कलय_F32(P0, P1));

                    S_vec[g] = HVX_OP_सङ्कलय_F32(HVX_OP_MUL_F32(S_vec[g], ms_vec), p_योग_vec);

                    ध्रुव uint8_t * v_ptr = v_आधारः;

                    यदि (DV == 64) {
                        HVX_सदिशम्_Pair vkq0 = *((ध्रुव HVX_सदिशम्_Pair *) VKQ32);
                        vkq0 = Q6_W_vcombine_VV(
                            HVX_OP_MUL_F32(Q6_V_hi_W(vkq0), ms_vec),
                            HVX_OP_MUL_F32(Q6_V_lo_W(vkq0), ms_vec)
                        );

                        प्रति (uint32_t j = 0; j < current_खण्डम्_आकार; j += 2) {
                            HVX_सदिशम् S0 = hvx_vec_repl_f16(Q6_V_vror_VR(P, j * 2));
                            ध्रुव HVX_सदिशम् * vx0 = (ध्रुव HVX_सदिशम् *) v_ptr;
                            यदि (j + 1 == current_खण्डम्_आकार) {
                                vkq0 = hvx_vec_mpyacc_f32_f16(vkq0, Q6_Vh_vshuff_Vh(vx0[0]), S0);
                                भङ्ग;
                            }

                            HVX_सदिशम् S1 = hvx_vec_repl_f16(Q6_V_vror_VR(P, (j + 1) * 2));
                            ध्रुव HVX_सदिशम् * vx1 = (ध्रुव HVX_सदिशम् *) (v_ptr + factx->आकार_v_row_padded);
                            vkq0 = hvx_vec_mpyacc_f32_f16(vkq0, Q6_Vh_vshuff_Vh(vx0[0]), S0);
                            vkq0 = hvx_vec_mpyacc_f32_f16(vkq0, Q6_Vh_vshuff_Vh(vx1[0]), S1);
                            v_ptr += stride_v2;
                        }

                        *((HVX_सदिशम्_Pair *) VKQ32) = vkq0;
                    } अन्यथा यदि (DV == 128) {
                        HVX_सदिशम्_Pair vkq0 = ((ध्रुव HVX_सदिशम्_Pair *) VKQ32)[0];
                        HVX_सदिशम्_Pair vkq1 = ((ध्रुव HVX_सदिशम्_Pair *) VKQ32)[1];
                        vkq0 = Q6_W_vcombine_VV(
                            HVX_OP_MUL_F32(Q6_V_hi_W(vkq0), ms_vec),
                            HVX_OP_MUL_F32(Q6_V_lo_W(vkq0), ms_vec)
                        );
                        vkq1 = Q6_W_vcombine_VV(
                            HVX_OP_MUL_F32(Q6_V_hi_W(vkq1), ms_vec),
                            HVX_OP_MUL_F32(Q6_V_lo_W(vkq1), ms_vec)
                        );

                        प्रति (uint32_t j = 0; j < current_खण्डम्_आकार; j += 2) {
                            HVX_सदिशम् S0 = hvx_vec_repl_f16(Q6_V_vror_VR(P, j * 2));
                            ध्रुव HVX_सदिशम् * vx0 = (ध्रुव HVX_सदिशम् *) v_ptr;
                            यदि (j + 1 == current_खण्डम्_आकार) {
                                vkq0 = hvx_vec_mpyacc_f32_f16(vkq0, Q6_Vh_vshuff_Vh(vx0[0]), S0);
                                vkq1 = hvx_vec_mpyacc_f32_f16(vkq1, Q6_Vh_vshuff_Vh(vx0[1]), S0);
                                भङ्ग;
                            }

                            HVX_सदिशम् S1 = hvx_vec_repl_f16(Q6_V_vror_VR(P, (j + 1) * 2));
                            ध्रुव HVX_सदिशम् * vx1 = (ध्रुव HVX_सदिशम् *) (v_ptr + factx->आकार_v_row_padded);
                            vkq0 = hvx_vec_mpyacc_f32_f16(vkq0, Q6_Vh_vshuff_Vh(vx0[0]), S0);
                            vkq0 = hvx_vec_mpyacc_f32_f16(vkq0, Q6_Vh_vshuff_Vh(vx1[0]), S1);
                            vkq1 = hvx_vec_mpyacc_f32_f16(vkq1, Q6_Vh_vshuff_Vh(vx0[1]), S0);
                            vkq1 = hvx_vec_mpyacc_f32_f16(vkq1, Q6_Vh_vshuff_Vh(vx1[1]), S1);
                            v_ptr += stride_v2;
                        }

                        ((HVX_सदिशम्_Pair *) VKQ32)[0] = vkq0;
                        ((HVX_सदिशम्_Pair *) VKQ32)[1] = vkq1;
                    } अन्यथा यदि (DV == 256) {
                        HVX_सदिशम्_Pair vkq0 = ((ध्रुव HVX_सदिशम्_Pair *) VKQ32)[0];
                        HVX_सदिशम्_Pair vkq1 = ((ध्रुव HVX_सदिशम्_Pair *) VKQ32)[1];
                        HVX_सदिशम्_Pair vkq2 = ((ध्रुव HVX_सदिशम्_Pair *) VKQ32)[2];
                        HVX_सदिशम्_Pair vkq3 = ((ध्रुव HVX_सदिशम्_Pair *) VKQ32)[3];
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

                        प्रति (uint32_t j = 0; j < current_खण्डम्_आकार; j += 2) {
                            HVX_सदिशम् S0 = hvx_vec_repl_f16(Q6_V_vror_VR(P, j * 2));
                            ध्रुव HVX_सदिशम् * vx0 = (ध्रुव HVX_सदिशम् *) v_ptr;
                            यदि (j + 1 == current_खण्डम्_आकार) {
                                vkq0 = hvx_vec_mpyacc_f32_f16(vkq0, Q6_Vh_vshuff_Vh(vx0[0]), S0);
                                vkq1 = hvx_vec_mpyacc_f32_f16(vkq1, Q6_Vh_vshuff_Vh(vx0[1]), S0);
                                vkq2 = hvx_vec_mpyacc_f32_f16(vkq2, Q6_Vh_vshuff_Vh(vx0[2]), S0);
                                vkq3 = hvx_vec_mpyacc_f32_f16(vkq3, Q6_Vh_vshuff_Vh(vx0[3]), S0);
                                भङ्ग;
                            }

                            HVX_सदिशम् S1 = hvx_vec_repl_f16(Q6_V_vror_VR(P, (j + 1) * 2));
                            ध्रुव HVX_सदिशम् * vx1 = (ध्रुव HVX_सदिशम् *) (v_ptr + factx->आकार_v_row_padded);
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

                        ((HVX_सदिशम्_Pair *) VKQ32)[0] = vkq0;
                        ((HVX_सदिशम्_Pair *) VKQ32)[1] = vkq1;
                        ((HVX_सदिशम्_Pair *) VKQ32)[2] = vkq2;
                        ((HVX_सदिशम्_Pair *) VKQ32)[3] = vkq3;
                    } अन्यथा {
                        hvx_scale_vec_f32_aa((uint8_t *) VKQ32, (ध्रुव uint8_t *) VKQ32, DV, ms_vec);

                        प्रति (uint32_t j = 0; j < current_खण्डम्_आकार; j += 2) {
                            यदि (j + 1 == current_खण्डम्_आकार) {
                                HVX_सदिशम् S0 = hvx_vec_repl_f16(Q6_V_vror_VR(P, j * 2));
                                hvx_mad_f32_f16_aa_vec(VKQ32, v_ptr, S0, DV);
                                भङ्ग;
                            }

                            HVX_सदिशम् S0 = hvx_vec_repl_f16(Q6_V_vror_VR(P, j * 2));
                            HVX_सदिशम् S1 = hvx_vec_repl_f16(Q6_V_vror_VR(P, (j + 1) * 2));

                            hvx_mad_f32_f16_aa_rx2_vec(VKQ32, v_ptr, v_ptr + factx->आकार_v_row_padded, S0, S1, DV);
                            v_ptr += stride_v2;
                        }
                    }
                }
                htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_FA_SFM, शीर्ष_ir);

                // prefetch V and mask for step + 2 after last head finished V accumulation
                यदि (g + 1 == G_local && चरणः + 2 < n_सक्रिय_खण्डानि) {
                    ध्रुव uint32_t अग्र_ib = सक्रिय_खण्डानि[चरणः + 2];
                    ध्रुव uint32_t अग्र_ic_start = अग्र_ib * FLASH_ATTN_खण्डम्_आकार;
                    ध्रुव uint32_t अग्र_खण्डम्_आकार = MIN(FLASH_ATTN_खण्डम्_आकार, nek1 - अग्र_ic_start);

                    // V
                    ध्रुव dma_addr_t v_src = v->दत्त + अग्र_ic_start*nbv1 + iv2*nbv2 + iv3*nbv3;
                    dma_queue_प्रक्षिप(dma_q, dma_make_दत्त(v_आधारः, v_src), factx->आकार_v_row_padded, nbv1, आकार_v_row, अग्र_खण्डम्_आकार);

                    // Mask
                    यदि (आवरणम्) {
                        ध्रुव dma_addr_t m_src = mp_आधारः + अग्र_ic_start * परिमाण(__fp16);
                        dma_cache_dm_प्रक्षिप(dma_q, &m_cache, अग्र_ib, m_src, अग्र_खण्डम्_आकार * 2, अग्र_खण्डम्_आकार * 2, अग्र_खण्डम्_आकार * 2, 1);
                    }
                }
            } // end for g
        } // end for step

        प्रति (uint32_t g = 0; g < G_local; ++g) {
            ध्रुव uint32_t शीर्ष_ir = ir + g;
            ध्रुव uint32_t h = heads[g];
            भिन्न * VKQ32 = vkq_ptrs[g];

            htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_O_PROC, शीर्ष_ir);

            भिन्न M = hvx_vec_get_f32(M_vec[g]);
            भिन्न S = hvx_vec_get_f32(S_vec[g]);

            यदि (sinks) {
                ध्रुव भिन्न s = factx->spad_sinks[h];

                भिन्न vs = 1.0f;

                यदि (s > M) {
                    HVX_सदिशम् diff_vec = hvx_vec_splat_f32(M - s);
                    HVX_सदिशम् ms_vec   = hvx_vec_exp_f32(diff_vec);
                    hvx_scale_vec_f32_aa((uint8_t *) VKQ32, (ध्रुव uint8_t *) VKQ32, DV, ms_vec);

                    भिन्न ms = hvx_vec_get_f32(ms_vec);
                    S = S * ms + vs;
                } अन्यथा {
                    HVX_सदिशम् diff_vec = hvx_vec_splat_f32(s - M);
                    vs = hvx_vec_get_f32(hvx_vec_exp_f32(diff_vec));
                    S += vs;
                }
            }

            ध्रुव भिन्न S_inv = S == 0.0f ? 0.0f : 1.0f/S;
            hvx_scale_f32_aa((uint8_t *) VKQ32, (ध्रुव uint8_t *) VKQ32, DV, S_inv);

            ध्रुव uint32_t r_iq3 = fastdiv(शीर्ष_ir, &factx->src0_div21);
            ध्रुव uint32_t r_iq2 = fastdiv(शीर्ष_ir - r_iq3*neq2*neq1, &factx->src0_div1);
            ध्रुव uint32_t r_iq1 = (शीर्ष_ir - r_iq3*neq2*neq1 - r_iq2 * neq1);

            uint8_t * dst_ptr = (uint8_t *) dst->दत्त + r_iq2 * dst->nb[1] + r_iq1 * dst->nb[2] + r_iq3 * dst->nb[3];

            यदि (dst->type == HTP_TYPE_F32) {
                hvx_copy_f32_ua(dst_ptr, (uint8_t *) VKQ32, DV);
            } अन्यथा यदि (dst->type == HTP_TYPE_F16) {
                hvx_copy_f16_f32_ua(dst_ptr, (uint8_t *) VKQ32, DV);
            }
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_O_PROC, शीर्ष_ir);
        }

        ir += G_local;
    }
}

// ============================================================================
// HMX Phase args and thread logic
// ============================================================================

प्रकारनिर्देश संरचना {
    संरचना hmx_fa_context * factx;
    uint32_t                kv_पङ्क्तिकाः;
    size_t                  src_stride;
    शून्य *                  curr_k;
    uint32_t                kv_start;
    uint32_t                पङ्क्तिकाः_प्रत्येकम्_t;
    size_t                  buf_idx;
} fa_k_int_तर्काः_t;

स्थिर शून्य fa_k_interleave_thread(अचिह्नित पूर्णाङ्क n, अचिह्नित पूर्णाङ्क i, शून्य * दत्त) {
    fa_k_int_तर्काः_t *       तर्काः_2  = (fa_k_int_तर्काः_t *) दत्त;
    संरचना hmx_fa_context * factx = तर्काः_2->factx;

    ध्रुव uint32_t योग_पङ्क्तिकाः = तर्काः_2->kv_पङ्क्तिकाः;
    ध्रुव uint32_t पङ्क्तिकाः_प्रत्येकम्_t = तर्काः_2->पङ्क्तिकाः_प्रत्येकम्_t;
    ध्रुव uint32_t start      = i * पङ्क्तिकाः_प्रत्येकम्_t;
    ध्रुव uint32_t अन्तः_2        = (uint32_t) hex_smin(start + पङ्क्तिकाः_प्रत्येकम्_t, योग_पङ्क्तिकाः);

    यदि (start >= योग_पङ्क्तिकाः) {
        प्रत्यागम;
    }

    संरचना htp_thread_trace * tr = &factx->octx->ctx->trace[i];
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_FA_K_PREP, (uint16_t) (तर्काः_2->kv_start + start));
    यदि (factx->k->type == HTP_TYPE_Q8_0) {
        प्रति (uint32_t r = start; r < अन्तः_2; ++r) {
            __fp16 * row_k = (__fp16 *)((वर्ण *)तर्काः_2->curr_k + r * तर्काः_2->src_stride * परिमाण(__fp16));
            hvx_dequantize_row_flat_q8_0_f16(row_k, row_k, factx->DK);
        }
    }
    hmx_interleave_पङ्क्तिकाः_गन्तव्यम्_tiles(factx->vtcm_k_tiles[तर्काः_2->buf_idx], (ध्रुव __fp16 *) तर्काः_2->curr_k, योग_पङ्क्तिकाः, factx->DK_pad,
                             तर्काः_2->src_stride, start, अन्तः_2);
    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_FA_K_PREP, (uint16_t) (तर्काः_2->kv_start + start));
}

स्थिर शून्य fa_phase_k_interleave(संरचना hmx_fa_context * factx, uint32_t kv_पङ्क्तिकाः, size_t src_stride, शून्य * curr_k, uint32_t kv_start, size_t buf_idx) {
    work_queue_t wp = factx->octx->ctx->work_queue;
    uint32_t n = 1;
    यदि (factx->n_तन्तवः > 1 && kv_पङ्क्तिकाः >= factx->n_तन्तवः * 2) {
        n = factx->n_तन्तवः;
    }
    uint32_t पङ्क्तिकाः_प्रत्येकम्_t = hex_align_ऊर्ध्वम्(hmx_ceil_div(kv_पङ्क्तिकाः, n), 2);
    fa_k_int_तर्काः_t तर्काः_2 = { factx, kv_पङ्क्तिकाः, src_stride, curr_k, kv_start, पङ्क्तिकाः_प्रत्येकम्_t, buf_idx };
    यदि (n > 1) {
        work_queue_सततम्(wp, fa_k_interleave_thread, &तर्काः_2, n);
    } अन्यथा {
        fa_k_interleave_thread(1, 0, &तर्काः_2);
    }
}

प्रकारनिर्देश संरचना {
    संरचना hmx_fa_context * factx;
    uint32_t                kv_पङ्क्तिकाः;
    size_t                  src_stride;
    शून्य *                  v_src;
    शून्य *                  v_tiles_dst;
    size_t                  n_col_tiles;
    uint32_t                kv_start;
    uint32_t                पङ्क्तिकाः_प्रत्येकम्_t;
} fa_v_int_तर्काः_t;

स्थिर शून्य fa_v_interleave_thread(अचिह्नित पूर्णाङ्क n, अचिह्नित पूर्णाङ्क i, शून्य * दत्त) {
    fa_v_int_तर्काः_t *       तर्काः_2  = (fa_v_int_तर्काः_t *) दत्त;
    संरचना hmx_fa_context * factx = तर्काः_2->factx;

    ध्रुव uint32_t योग_पङ्क्तिकाः = तर्काः_2->kv_पङ्क्तिकाः;
    ध्रुव uint32_t पङ्क्तिकाः_प्रत्येकम्_t = तर्काः_2->पङ्क्तिकाः_प्रत्येकम्_t;
    ध्रुव uint32_t start      = i * पङ्क्तिकाः_प्रत्येकम्_t;
    ध्रुव uint32_t अन्तः_2        = (uint32_t) hex_smin(start + पङ्क्तिकाः_प्रत्येकम्_t, योग_पङ्क्तिकाः);

    यदि (start >= योग_पङ्क्तिकाः) {
        प्रत्यागम;
    }

    __fp16 * v_tiles_dst = (__fp16 *) तर्काः_2->v_tiles_dst;

    संरचना htp_thread_trace * tr = &factx->octx->ctx->trace[i];
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_FA_V_PREP, (uint16_t) (तर्काः_2->kv_start + start));
    यदि (factx->v->type == HTP_TYPE_Q8_0) {
        प्रति (uint32_t r = start; r < अन्तः_2; ++r) {
            __fp16 * row_v = (__fp16 *)((वर्ण *)तर्काः_2->v_src + r * तर्काः_2->src_stride * परिमाण(__fp16));
            hvx_dequantize_row_flat_q8_0_f16(row_v, row_v, factx->DV);
        }
    }
    hmx_interleave_स्तम्भसङ्ख्या_गन्तव्यम्_tiles(v_tiles_dst, (ध्रुव __fp16 *) तर्काः_2->v_src, योग_पङ्क्तिकाः, factx->DV_pad,
                             तर्काः_2->src_stride, (uint32_t) तर्काः_2->n_col_tiles, start, अन्तः_2);
    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_FA_V_PREP, (uint16_t) (तर्काः_2->kv_start + start));
}

स्थिर शून्य fa_phase_v_interleave(संरचना hmx_fa_context * factx,
                                  uint32_t                kv_पङ्क्तिकाः,
                                  size_t                  src_stride,
                                  शून्य *                  v_src,
                                  शून्य *                  v_tiles_dst,
                                  size_t                  n_col_tiles,
                                  uint32_t                kv_start) {
    work_queue_t wp = factx->octx->ctx->work_queue;
    uint32_t n = 1;
    यदि (factx->n_तन्तवः > 1 && kv_पङ्क्तिकाः >= factx->n_तन्तवः * 2) {
        n = factx->n_तन्तवः;
    }
    uint32_t पङ्क्तिकाः_प्रत्येकम्_t = hex_align_ऊर्ध्वम्(hmx_ceil_div(kv_पङ्क्तिकाः, n), 2);
    fa_v_int_तर्काः_t तर्काः_2 = { factx, kv_पङ्क्तिकाः, src_stride, v_src, v_tiles_dst, n_col_tiles, kv_start, पङ्क्तिकाः_प्रत्येकम्_t };
    यदि (n > 1) {
        work_queue_सततम्(wp, fa_v_interleave_thread, &तर्काः_2, n);
    } अन्यथा {
        fa_v_interleave_thread(1, 0, &तर्काः_2);
    }
}

प्रकारनिर्देश संरचना {
    संरचना hmx_fa_context *   factx;
    ध्रुव संरचना htp_tensor * q;
    uint32_t                  q_start;
    uint32_t                  kv_शीर्ष;
    uint32_t                  ib3;
    size_t                    n_पङ्क्तिकाः_g;
    size_t                    पङ्क्तिकाः_प्रत्येकम्_t;
    size_t                    n_पङ्क्तिकाः_q;
    bool                      q_transposed;
    atomic_uint               barrier;
} fa_q_आरोपय_तर्काः_t;

स्थिर शून्य fa_q_आरोपय_thread(अचिह्नित पूर्णाङ्क n, अचिह्नित पूर्णाङ्क i, शून्य * दत्त) {
    fa_q_आरोपय_तर्काः_t *      तर्काः_2  = (fa_q_आरोपय_तर्काः_t *) दत्त;
    संरचना hmx_fa_context * factx = तर्काः_2->factx;

    ध्रुव size_t n_पङ्क्तिकाः_g = तर्काः_2->n_पङ्क्तिकाः_g;
    ध्रुव size_t G        = factx->G;
    ध्रुव size_t DK       = factx->DK;

    // Partition the padded Q rows (g_br) across threads.
    // Keep start/end even so r and r+1 are always in the same thread's range.
    ध्रुव size_t पङ्क्तिकाः_प्रत्येकम्_t = तर्काः_2->पङ्क्तिकाः_प्रत्येकम्_t;
    ध्रुव size_t start      = (size_t) i * पङ्क्तिकाः_प्रत्येकम्_t;
    ध्रुव size_t अन्तः_2        = hex_smin(start + पङ्क्तिकाः_प्रत्येकम्_t, factx->g_br);

    संरचना htp_thread_trace * tr = &factx->octx->ctx->trace[i];
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_FA_Q_PREP, (uint16_t) (तर्काः_2->q_start * G + start));

    // Parallel initialization of per-block state
    {
        ध्रुव uint32_t g_br = factx->g_br;
        ध्रुव uint32_t DV   = factx->DV;

        ध्रुव size_t col_vec_अष्टकानि = factx->col_vec_अष्टकानि;
        ध्रुव size_t d_tile_अष्टकानि  = factx->d_tile_अष्टकानि;

        // Initialize vtcm_l_vec & vtcm_m_vec
        ध्रुव size_t l_अष्टकानि_प्रत्येकम्_t = hex_align_ऊर्ध्वम्(col_vec_अष्टकानि / n, 128);
        ध्रुव size_t l_start       = i * l_अष्टकानि_प्रत्येकम्_t;
        ध्रुव size_t l_अन्तः         = hex_smin(l_start + l_अष्टकानि_प्रत्येकम्_t, col_vec_अष्टकानि);

        ध्रुव size_t m_अष्टकानि_प्रत्येकम्_t = hex_align_ऊर्ध्वम्(col_vec_अष्टकानि / n, 128);
        ध्रुव size_t m_start       = i * m_अष्टकानि_प्रत्येकम्_t;
        ध्रुव size_t m_अन्तः         = hex_smin(m_start + m_अष्टकानि_प्रत्येकम्_t, col_vec_अष्टकानि);

        यदि (l_start < col_vec_अष्टकानि) {
            hvx_splat_u8_a((वर्ण *) factx->vtcm_l_vec + l_start, 0, l_अन्तः - l_start);
        }
        यदि (m_start < col_vec_अष्टकानि) {
            hvx_splat_f32_a((वर्ण *) factx->vtcm_m_vec + m_start, HTP_FA_M_INITIAL_VAL, (m_अन्तः - m_start) / परिमाण(भिन्न));
        }

        // Zero the whole rescale region: vtcm_d_tiles[0], the optional vtcm_d_tiles[1]
        // and vtcm_d_inv_l are equal-sized and allocated back to back, so one run covers
        // them all.  The scatter only ever writes the diagonal, ignore the rest.
        ध्रुव size_t d_अष्टकानि_प्रत्येकम्_t = hex_align_ऊर्ध्वम्(d_tile_अष्टकानि / n, 128);
        ध्रुव size_t d_start       = i * d_अष्टकानि_प्रत्येकम्_t;
        ध्रुव size_t d_अन्तः         = hex_smin(d_start + d_अष्टकानि_प्रत्येकम्_t, d_tile_अष्टकानि);
        यदि (d_start < d_tile_अष्टकानि) {
            hvx_splat_u8_a((वर्ण *) factx->vtcm_d_tiles[0] + d_start, 0, d_अन्तः - d_start);
        }
    }

    यदि (start < factx->g_br) {
        ध्रुव संरचना htp_tensor * q       = तर्काः_2->q;
        ध्रुव uint32_t            q_start = तर्काः_2->q_start;
        ध्रुव uint32_t            kv_शीर्ष = तर्काः_2->kv_शीर्ष;
        ध्रुव uint32_t            ib3     = तर्काः_2->ib3;

        ध्रुव bool use_q_dma = (factx->vtcm_q_dma != रिक्त);

        __fp16 * q_tiles = factx->vtcm_q_tiles;
        ध्रुव size_t DK_pad = factx->DK_pad;
        यदि (use_q_dma) {
            ध्रुव size_t g_पङ्क्तिकाः_अन्तः = hex_smin(अन्तः_2, n_पङ्क्तिकाः_g);
            ध्रुव uint32_t d_सीमा = factx->is_q_fp32 ? DK / 32 : DK / 64;

            uint8_t * q_flat  = (uint8_t *) factx->vtcm_q_dma;
            यदि (DK_pad != DK) {
                यदि (factx->is_q_fp32) {
                    hmx_fa_q_prep_fp32_pad(q_tiles, q_flat, start, अन्तः_2, g_पङ्क्तिकाः_अन्तः, DK, DK_pad, G, तर्काः_2->n_पङ्क्तिकाः_q, &factx->div_G, तर्काः_2->q_transposed);
                } अन्यथा {
                    hmx_fa_q_prep_fp16_pad(q_tiles, q_flat, start, अन्तः_2, g_पङ्क्तिकाः_अन्तः, DK, DK_pad, G, तर्काः_2->n_पङ्क्तिकाः_q, &factx->div_G, तर्काः_2->q_transposed);
                }
            } अन्यथा यदि (factx->is_q_fp32) {
                विकल्प (d_सीमा) {
                पक्ष 2:  hmx_fa_q_prep_fp32_d2(q_tiles, q_flat, start, अन्तः_2, g_पङ्क्तिकाः_अन्तः, DK, G, तर्काः_2->n_पङ्क्तिकाः_q, &factx->div_G, तर्काः_2->q_transposed); भङ्ग;
                पक्ष 4:  hmx_fa_q_prep_fp32_d4(q_tiles, q_flat, start, अन्तः_2, g_पङ्क्तिकाः_अन्तः, DK, G, तर्काः_2->n_पङ्क्तिकाः_q, &factx->div_G, तर्काः_2->q_transposed); भङ्ग;
                स्वभाव: hmx_fa_q_prep_fp32(   q_tiles, q_flat, start, अन्तः_2, g_पङ्क्तिकाः_अन्तः, DK, G, तर्काः_2->n_पङ्क्तिकाः_q, &factx->div_G, d_सीमा, तर्काः_2->q_transposed); भङ्ग;
                }
            } अन्यथा {
                विकल्प (d_सीमा) {
                पक्ष 1:  hmx_fa_q_prep_fp16_d1(q_tiles, q_flat, start, अन्तः_2, g_पङ्क्तिकाः_अन्तः, DK, G, तर्काः_2->n_पङ्क्तिकाः_q, &factx->div_G, तर्काः_2->q_transposed); भङ्ग;
                पक्ष 2:  hmx_fa_q_prep_fp16_d2(q_tiles, q_flat, start, अन्तः_2, g_पङ्क्तिकाः_अन्तः, DK, G, तर्काः_2->n_पङ्क्तिकाः_q, &factx->div_G, तर्काः_2->q_transposed); भङ्ग;
                स्वभाव: hmx_fa_q_prep_fp16(   q_tiles, q_flat, start, अन्तः_2, g_पङ्क्तिकाः_अन्तः, DK, G, तर्काः_2->n_पङ्क्तिकाः_q, &factx->div_G, d_सीमा, तर्काः_2->q_transposed); भङ्ग;
                }
            }
        } अन्यथा {
            // Fallback: direct-from-DDR/L2 path
            hmx_fa_q_prep_fallback(q_tiles, q->दत्त, q->nb[1], q->nb[2], q->nb[3],
                                   q_start, kv_शीर्ष, ib3, start, अन्तः_2, n_पङ्क्तिकाः_g, G, DK, DK_pad, factx->is_q_fp32, &factx->div_G);
        }
    }

    // Synchronize threads before zeroing out vtcm_o_tiles[0] to prevent race condition
    यदि (n > 1) {
        atomic_fetch_sub(&तर्काः_2->barrier, 1);
        यावत् (atomic_आरोपय(&तर्काः_2->barrier) > 0) {
            // spin wait
        }
    }

    // Zero out vtcm_o_tiles[0] as it was used as temp_q_vtcm
    {
        ध्रुव uint32_t g_br = factx->g_br;
        ध्रुव uint32_t DV   = factx->DV;
        ध्रुव size_t o_tile_अष्टकानि  = factx->o_tile_अष्टकानि;
        ध्रुव size_t o_अष्टकानि_प्रत्येकम्_t = hex_align_ऊर्ध्वम्(o_tile_अष्टकानि / n, 128);
        ध्रुव size_t o_start       = i * o_अष्टकानि_प्रत्येकम्_t;
        ध्रुव size_t o_अन्तः         = hex_smin(o_start + o_अष्टकानि_प्रत्येकम्_t, o_tile_अष्टकानि);
        यदि (o_start < o_tile_अष्टकानि) {
            hvx_splat_u8_a((वर्ण *) factx->vtcm_o_tiles[0] + o_start, 0, o_अन्तः - o_start);
        }
    }
    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_FA_Q_PREP, (uint16_t) (तर्काः_2->q_start * G + start));
}

स्थिर शून्य fa_phase_q_आरोपय(संरचना hmx_fa_context *   factx,
                            ध्रुव संरचना htp_tensor * q,
                            uint32_t                  q_start,
                            uint32_t                  kv_शीर्ष,
                            uint32_t                  ib3,
                            size_t                    n_पङ्क्तिकाः_g) {
    work_queue_t wp = factx->octx->ctx->work_queue;
    uint32_t n = 1;
    यदि (factx->n_तन्तवः > 1 && n_पङ्क्तिकाः_g >= (size_t) (factx->n_तन्तवः * 2)) {
        n = factx->n_तन्तवः;
    }
    size_t पङ्क्तिकाः_प्रत्येकम्_t = hex_align_ऊर्ध्वम्(hmx_ceil_div(factx->g_br, n), 2);
    ध्रुव uint32_t n_पङ्क्तिकाः_q = hex_smin(factx->Br, factx->neq1 - q_start);
    fa_q_आरोपय_तर्काः_t तर्काः_2;
    तर्काः_2.factx = factx;
    तर्काः_2.q = q;
    तर्काः_2.q_start = q_start;
    तर्काः_2.kv_शीर्ष = kv_शीर्ष;
    तर्काः_2.ib3 = ib3;
    तर्काः_2.n_पङ्क्तिकाः_g = n_पङ्क्तिकाः_g;
    तर्काः_2.पङ्क्तिकाः_प्रत्येकम्_t = पङ्क्तिकाः_प्रत्येकम्_t;
    तर्काः_2.n_पङ्क्तिकाः_q = n_पङ्क्तिकाः_q;
    तर्काः_2.q_transposed = q->nb[1] < q->nb[2];
    atomic_init(&तर्काः_2.barrier, n);
    यदि (n > 1) {
        work_queue_सततम्(wp, fa_q_आरोपय_thread, &तर्काः_2, n);
    } अन्यथा {
        fa_q_आरोपय_thread(1, 0, &तर्काः_2);
    }
}

प्रकारनिर्देश संरचना {
    संरचना hmx_fa_context *   factx;
    ध्रुव संरचना htp_tensor * dst;
    ध्रुव __fp16 *            o_tile_src;
    uint32_t                  q_start;
    uint32_t                  kv_शीर्ष;
    uint32_t                  ib3;
    size_t                    n_पङ्क्तिकाः_g;
    size_t                    पङ्क्तिकाः_प्रत्येकम्_t;
} fa_o_store_तर्काः_t;

स्थिर शून्य fa_o_store_thread_f32(अचिह्नित पूर्णाङ्क n, अचिह्नित पूर्णाङ्क i, शून्य * दत्त) {
    fa_o_store_तर्काः_t *     तर्काः_2  = (fa_o_store_तर्काः_t *) दत्त;
    संरचना hmx_fa_context * factx = तर्काः_2->factx;

    ध्रुव size_t n_पङ्क्तिकाः_g = तर्काः_2->n_पङ्क्तिकाः_g;
    ध्रुव size_t G        = factx->G;
    ध्रुव size_t DV       = factx->DV;

    ध्रुव size_t पङ्क्तिकाः_प्रत्येकम्_t = तर्काः_2->पङ्क्तिकाः_प्रत्येकम्_t;
    ध्रुव size_t start      = (size_t) i * पङ्क्तिकाः_प्रत्येकम्_t;
    ध्रुव size_t अन्तः_2        = hex_smin(start + पङ्क्तिकाः_प्रत्येकम्_t, n_पङ्क्तिकाः_g);

    यदि (start >= n_पङ्क्तिकाः_g) {
        प्रत्यागम;
    }

    संरचना htp_thread_trace * tr = &factx->octx->ctx->trace[i];
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_O_PROC, (uint16_t) (तर्काः_2->q_start * G + start));

    ध्रुव संरचना htp_tensor * dst        = तर्काः_2->dst;
    ध्रुव __fp16 *            o_tile_src = तर्काः_2->o_tile_src;
    ध्रुव uint32_t            q_start    = तर्काः_2->q_start;
    ध्रुव uint32_t            kv_शीर्ष    = तर्काः_2->kv_शीर्ष;
    ध्रुव uint32_t            ib3        = तर्काः_2->ib3;

    ध्रुव size_t DV_pad = factx->DV_pad;

    size_t q_idx = fastdiv(start, &factx->div_G);
    size_t h_idx = fastmodulo(start, G, &factx->div_G);

    प्रति (size_t r = start; r < अन्तः_2; ++r) {
        भिन्न * बहिर्गमनम् = (भिन्न *) ((uint8_t *) dst->दत्त + (kv_शीर्ष * G + h_idx) * dst->nb[1] +
                                 (q_start + q_idx) * dst->nb[2] + ib3 * dst->nb[3]);

        size_t         r0            = r / HMX_FP16_TILE_N_पङ्क्तिकाः;
        size_t         r1            = r % HMX_FP16_TILE_N_पङ्क्तिकाः;
        ध्रुव __fp16 * tile_row_आधारः = o_tile_src + r0 * HMX_FP16_TILE_N_पङ्क्तिकाः * DV_pad;

        प्रति (uint32_t d = 0; d < DV / 32; ++d) {
            ध्रुव HVX_सदिशम् * अन्तर्_tile = (ध्रुव HVX_सदिशम् *) (tile_row_आधारः + d * HMX_FP16_TILE_N_ELMS);
            HVX_सदिशम्_Pair     vp      = hvx_vec_f16_गन्तव्यम्_f32_shuff(अन्तर्_tile[r1 / 2]);
            यदि (r1 % 2 == 0) {
                *(HVX_UVector *) (बहिर्गमनम् + d * 32) = Q6_V_lo_W(vp);
            } अन्यथा {
                *(HVX_UVector *) (बहिर्गमनम् + d * 32) = Q6_V_hi_W(vp);
            }
        }
        // Ragged tail: DV not a multiple of 32 (e.g. 72 -> last 8 lanes). Partial vector-write
        // for the remaining (DV % 32) floats.
        ध्रुव uint32_t d_पुच्छ = DV / 32;
        ध्रुव uint32_t rem    = DV - d_पुच्छ * 32;
        यदि (rem) {
            ध्रुव HVX_सदिशम् * अन्तर्_tile = (ध्रुव HVX_सदिशम् *) (tile_row_आधारः + d_पुच्छ * HMX_FP16_TILE_N_ELMS);
            HVX_सदिशम्_Pair     vp      = hvx_vec_f16_गन्तव्यम्_f32_shuff(अन्तर्_tile[r1 / 2]);
            HVX_सदिशम्         vd      = (r1 % 2 == 0) ? Q6_V_lo_W(vp) : Q6_V_hi_W(vp);
            hvx_vec_store_u((शून्य *) (बहिर्गमनम् + d_पुच्छ * 32), rem * परिमाण(भिन्न), vd);
        }

        h_idx++;
        यदि (h_idx == G) {
            h_idx = 0;
            q_idx++;
        }
    }
    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_O_PROC, (uint16_t) (तर्काः_2->q_start * G + start));
}

स्थिर शून्य fa_o_store_thread_f16(अचिह्नित पूर्णाङ्क n, अचिह्नित पूर्णाङ्क i, शून्य * दत्त) {
    fa_o_store_तर्काः_t *     तर्काः_2  = (fa_o_store_तर्काः_t *) दत्त;
    संरचना hmx_fa_context * factx = तर्काः_2->factx;

    ध्रुव size_t n_पङ्क्तिकाः_g   = तर्काः_2->n_पङ्क्तिकाः_g;
    ध्रुव size_t पङ्क्तिकाः_प्रत्येकम्_t = तर्काः_2->पङ्क्तिकाः_प्रत्येकम्_t;
    ध्रुव size_t G          = factx->G;
    ध्रुव size_t DV         = factx->DV;
    ध्रुव size_t start      = (size_t) i * पङ्क्तिकाः_प्रत्येकम्_t;
    ध्रुव size_t अन्तः_2        = hex_smin(start + पङ्क्तिकाः_प्रत्येकम्_t, n_पङ्क्तिकाः_g);

    यदि (start >= n_पङ्क्तिकाः_g) {
        प्रत्यागम;
    }

    संरचना htp_thread_trace * tr = &factx->octx->ctx->trace[i];
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_O_PROC, (uint16_t) (तर्काः_2->q_start * G + start));

    ध्रुव संरचना htp_tensor * dst        = तर्काः_2->dst;
    ध्रुव __fp16 *            o_tile_src = तर्काः_2->o_tile_src;
    ध्रुव uint32_t            q_start    = तर्काः_2->q_start;
    ध्रुव uint32_t            kv_शीर्ष    = तर्काः_2->kv_शीर्ष;
    ध्रुव uint32_t            ib3        = तर्काः_2->ib3;

    // O-tiles use the padded head dim (DV_pad); dst holds the real DV lanes.
    ध्रुव size_t DV_pad = factx->DV_pad;

    size_t q_idx = fastdiv(start, &factx->div_G);
    size_t h_idx = fastmodulo(start, G, &factx->div_G);

    प्रति (size_t r = start; r < अन्तः_2; ++r) {
        __fp16 * बहिर्गमनम् = (__fp16 *) ((uint8_t *) dst->दत्त + (kv_शीर्ष * G + h_idx) * dst->nb[1] +
                                   (q_start + q_idx) * dst->nb[2] + ib3 * dst->nb[3]);

        size_t         r0            = r / HMX_FP16_TILE_N_पङ्क्तिकाः;
        size_t         r1            = r % HMX_FP16_TILE_N_पङ्क्तिकाः;
        ध्रुव __fp16 * tile_row_आधारः = o_tile_src + r0 * HMX_FP16_TILE_N_पङ्क्तिकाः * DV_pad;

        प्रति (uint32_t d = 0; d < DV / 64; ++d) {
            ध्रुव __fp16 *     अन्तर्_dtile = tile_row_आधारः + d * HMX_FP16_TILE_N_ELMS * 2;
            ध्रुव HVX_सदिशम् * pv_in0   = ((ध्रुव HVX_सदिशम् *) अन्तर्_dtile) + r1 / 2;
            ध्रुव HVX_सदिशम् * pv_in1   = pv_in0 + 16;
            HVX_सदिशम्_Pair     vp       = Q6_W_vdeal_VVR(*pv_in1, *pv_in0, -2);
            यदि (r1 % 2 == 0) {
                *(HVX_UVector *) (बहिर्गमनम् + d * 64) = Q6_V_lo_W(vp);
            } अन्यथा {
                *(HVX_UVector *) (बहिर्गमनम् + d * 64) = Q6_V_hi_W(vp);
            }
        }
        // Ragged tail when DV is not a multiple of 64.
        ध्रुव uint32_t d_पुच्छ = DV / 64;
        ध्रुव uint32_t rem    = DV - d_पुच्छ * 64;
        यदि (rem) {
            ध्रुव __fp16 *     अन्तर्_dtile = tile_row_आधारः + d_पुच्छ * HMX_FP16_TILE_N_ELMS * 2;
            ध्रुव HVX_सदिशम् * pv_in0   = ((ध्रुव HVX_सदिशम् *) अन्तर्_dtile) + r1 / 2;
            ध्रुव HVX_सदिशम् * pv_in1   = pv_in0 + 16;
            HVX_सदिशम्_Pair     vp       = Q6_W_vdeal_VVR(*pv_in1, *pv_in0, -2);
            HVX_सदिशम्         vd       = (r1 % 2 == 0) ? Q6_V_lo_W(vp) : Q6_V_hi_W(vp);
            hvx_vec_store_u((शून्य *) (बहिर्गमनम् + d_पुच्छ * 64), rem * परिमाण(__fp16), vd);
        }

        h_idx++;
        यदि (h_idx == G) {
            h_idx = 0;
            q_idx++;
        }
    }
    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_O_PROC, (uint16_t) (तर्काः_2->q_start * G + start));
}

स्थिर शून्य fa_phase_o_store(संरचना hmx_fa_context *   factx,
                             ध्रुव संरचना htp_tensor * dst,
                             ध्रुव __fp16 *            o_tile_src,
                             uint32_t                  q_start,
                             uint32_t                  kv_शीर्ष,
                             uint32_t                  ib3,
                             size_t                    n_पङ्क्तिकाः_g) {
    work_queue_t wp = factx->octx->ctx->work_queue;
    uint32_t n = 1;
    यदि (factx->n_तन्तवः > 1 && n_पङ्क्तिकाः_g >= (size_t) (factx->n_तन्तवः * 2)) {
        n = factx->n_तन्तवः;
    }
    size_t पङ्क्तिकाः_प्रत्येकम्_t = hmx_ceil_div(n_पङ्क्तिकाः_g, n);
    fa_o_store_तर्काः_t तर्काः_2 = { factx, dst, o_tile_src, q_start, kv_शीर्ष, ib3, n_पङ्क्तिकाः_g, पङ्क्तिकाः_प्रत्येकम्_t };
    कार्यकर्ता_callback_t store_fn = factx->is_dst_fp32 ? fa_o_store_thread_f32 : fa_o_store_thread_f16;
    यदि (n > 1) {
        work_queue_सततम्(wp, store_fn, &तर्काः_2, n);
    } अन्यथा {
        store_fn(1, 0, &तर्काः_2);
    }
}

प्रकारनिर्देश संरचना {
    संरचना hmx_fa_context *   factx;
    size_t                    buf_idx;
    size_t                    kv_पङ्क्तिकाः;
    size_t                    n_पङ्क्तिकाः_g;
    size_t                    n_col_tiles;
    size_t                    n_tiles_प्रत्येकम्_bc;
    size_t                    n_row_tiles;
    size_t                    n_row_tiles_g_br;
    uint32_t                  Bc;
    uint32_t                  G;
    uint32_t                  kv_शीर्ष;
    uint32_t                  kv_start;
    uint32_t                  q_start;
    uint32_t                  ib3;
    bool                      has_alibi;  // true when max_bias != 0 (need slope * mask + add)
    __fp16 *                  slopes;
    ध्रुव संरचना htp_tensor * आवरणम्;
    ध्रुव __fp16 *            आवरणम्_vtcm;             // VTCM mask buffer base (NULL = DDR fallback)
    size_t                    आवरणम्_vtcm_row_stride;  // elements (__fp16) per row in VTCM mask buffer
    संरचना fastdiv_मानानि     thread_div;
} fa_softmax_तर्काः_t;

स्थिर अन्तःस्थ शून्य fa_softmax_impl(
    अचिह्नित पूर्णाङ्क n, अचिह्नित पूर्णाङ्क i, शून्य * दत्त,
    ध्रुव bool has_आवरणम्,
    ध्रुव bool आवरणम्_प्रसारणम्,
    ध्रुव bool is_g1,
    ध्रुव bool has_alibi,
    ध्रुव bool has_softcap
) {
    fa_softmax_तर्काः_t *     तर्काः_2  = (fa_softmax_तर्काः_t *) दत्त;
    संरचना hmx_fa_context * factx = तर्काः_2->factx;

    ध्रुव size_t n_पङ्क्तिकाः_g       = तर्काः_2->n_पङ्क्तिकाः_g;
    ध्रुव size_t kv_पङ्क्तिकाः        = तर्काः_2->kv_पङ्क्तिकाः;
    ध्रुव size_t Bc             = तर्काः_2->Bc;
    ध्रुव size_t G              = तर्काः_2->G;
    ध्रुव size_t n_tiles_प्रत्येकम्_bc = तर्काः_2->n_tiles_प्रत्येकम्_bc;
    ध्रुव size_t n_row_vec_cnt  = hmx_ceil_div(n_पङ्क्तिकाः_g, 64);
    ध्रुव uint32_t im3          = has_आवरणम् ? fastmodulo(तर्काः_2->ib3, तर्काः_2->आवरणम्->ne[3], &factx->src3_div3) : 0;

    size_t vec_start = 0;
    size_t vec_अन्तः   = n_row_vec_cnt;
    यदि (n > 1) {
        ध्रुव size_t vecs_प्रत्येकम्_t = fastdiv(n_row_vec_cnt + n - 1, &तर्काः_2->thread_div);
        vec_start = i * vecs_प्रत्येकम्_t;
        vec_अन्तः   = hex_smin(vec_start + vecs_प्रत्येकम्_t, n_row_vec_cnt);
    }

    यदि (vec_start >= n_row_vec_cnt) {
        प्रत्यागम;
    }

    संरचना htp_thread_trace * tr = &factx->octx->ctx->trace[i];
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_FA_SFM, (uint16_t) (तर्काः_2->q_start * G + vec_start * 64));

    // Per-thread row scratch: thread i uses bufs at offset i * 2 * stride
    ध्रुव size_t row_buf_stride = factx->row_buf_stride;
    HVX_सदिशम् * my_row_buf0    = factx->vtcm_row_bufs + i * 2 * row_buf_stride;
    HVX_सदिशम् * my_row_buf1    = my_row_buf0 + row_buf_stride;

    ध्रुव HVX_सदिशम् v_neg_inf = Q6_Vh_vsplat_R(0xfbff);

    प्रति (size_t r_vec_idx = vec_start; r_vec_idx < vec_अन्तः; ++r_vec_idx) {
        HVX_सदिशम् rowmax_acc_v = v_neg_inf;
        HVX_सदिशम् rowsum_acc_v = Q6_V_vzero();
        HVX_सदिशम् m_पूर्व_v0    = factx->vtcm_m_vec[r_vec_idx * 2 + 0];
        HVX_सदिशम् m_पूर्व_v1    = factx->vtcm_m_vec[r_vec_idx * 2 + 1];

        HVX_सदिशम् v_slopes = Q6_V_vzero();
        यदि (has_alibi) {
            v_slopes = hvx_vmem(तर्काः_2->slopes + r_vec_idx * 64);
        }

        प्रति (uint32_t r_vec_off = 0; r_vec_off < 64; r_vec_off += 2) {
            uint32_t r = r_vec_idx * 64 + r_vec_off;
            यदि (r >= hex_align_ऊर्ध्वम्(n_पङ्क्तिकाः_g, 2)) {
                भङ्ग;
            }

            uint32_t r0 = r / HMX_FP16_TILE_N_पङ्क्तिकाः;
            uint32_t r1 = r % HMX_FP16_TILE_N_पङ्क्तिकाः;

            ध्रुव __fp16 * s_ld_आधारः = factx->vtcm_s_tiles[तर्काः_2->buf_idx] + r0 * HMX_FP16_TILE_N_पङ्क्तिकाः * Bc;
            __fp16 *       p_st_आधारः = factx->vtcm_p_tiles[तर्काः_2->buf_idx] + r0 * HMX_FP16_TILE_N_पङ्क्तिकाः * Bc;

            // Decode 2 rows from S tiles into per-thread row buffers
            यदि (has_softcap) {
                ध्रुव HVX_सदिशम् v_cap = hvx_vec_splat_f16(factx->logit_softcap);
                प्रति (size_t c = 0; c < kv_पङ्क्तिकाः; c += 64) {
                    size_t             ci       = c / 64;
                    ध्रुव __fp16 *     अन्तर्_dtile = s_ld_आधारः + ci * HMX_FP16_TILE_N_ELMS * 2;
                    ध्रुव HVX_सदिशम् * pv_s_in0 = ((ध्रुव HVX_सदिशम् *) अन्तर्_dtile) + r1 / 2;
                    ध्रुव HVX_सदिशम् * pv_s_in1 = pv_s_in0 + 16;

                    HVX_सदिशम्_Pair vp_s_drow = Q6_W_vdeal_VVR(*pv_s_in1, *pv_s_in0, -2);
                    HVX_सदिशम्     v_s_row0  = Q6_V_lo_W(vp_s_drow);
                    HVX_सदिशम्     v_s_row1  = Q6_V_hi_W(vp_s_drow);

                    HVX_सदिशम् t0   = hvx_vec_tanh_f16(v_s_row0);
                    my_row_buf0[ci] = hvx_vec_mul_f16_f16(t0, v_cap);

                    HVX_सदिशम् t1   = hvx_vec_tanh_f16(v_s_row1);
                    my_row_buf1[ci] = hvx_vec_mul_f16_f16(t1, v_cap);
                }
            } अन्यथा {
                size_t c = 0;
                प्रति (; c + 64 < kv_पङ्क्तिकाः; c += 128) {
                    size_t             ci0       = c / 64;
                    size_t             ci1       = ci0 + 1;
                    ध्रुव __fp16 *     अन्तर्_dtile0 = s_ld_आधारः + ci0 * HMX_FP16_TILE_N_ELMS * 2;
                    ध्रुव __fp16 *     अन्तर्_dtile1 = s_ld_आधारः + ci1 * HMX_FP16_TILE_N_ELMS * 2;
                    ध्रुव HVX_सदिशम् * pv_s_in0_0 = ((ध्रुव HVX_सदिशम् *) अन्तर्_dtile0) + r1 / 2;
                    ध्रुव HVX_सदिशम् * pv_s_in1_0 = pv_s_in0_0 + 16;
                    ध्रुव HVX_सदिशम् * pv_s_in0_1 = ((ध्रुव HVX_सदिशम् *) अन्तर्_dtile1) + r1 / 2;
                    ध्रुव HVX_सदिशम् * pv_s_in1_1 = pv_s_in0_1 + 16;

                    HVX_सदिशम्_Pair vp_s_drow0 = Q6_W_vdeal_VVR(*pv_s_in1_0, *pv_s_in0_0, -2);
                    my_row_buf0[ci0]          = Q6_V_lo_W(vp_s_drow0);
                    my_row_buf1[ci0]          = Q6_V_hi_W(vp_s_drow0);

                    HVX_सदिशम्_Pair vp_s_drow1 = Q6_W_vdeal_VVR(*pv_s_in1_1, *pv_s_in0_1, -2);
                    my_row_buf0[ci1]          = Q6_V_lo_W(vp_s_drow1);
                    my_row_buf1[ci1]          = Q6_V_hi_W(vp_s_drow1);
                }
                प्रति (; c < kv_पङ्क्तिकाः; c += 64) {
                    size_t             ci       = c / 64;
                    ध्रुव __fp16 *     अन्तर्_dtile = s_ld_आधारः + ci * HMX_FP16_TILE_N_ELMS * 2;
                    ध्रुव HVX_सदिशम् * pv_s_in0 = ((ध्रुव HVX_सदिशम् *) अन्तर्_dtile) + r1 / 2;
                    ध्रुव HVX_सदिशम् * pv_s_in1 = pv_s_in0 + 16;

                    HVX_सदिशम्_Pair vp_s_drow = Q6_W_vdeal_VVR(*pv_s_in1, *pv_s_in0, -2);
                    my_row_buf0[ci]          = Q6_V_lo_W(vp_s_drow);
                    my_row_buf1[ci]          = Q6_V_hi_W(vp_s_drow);
                }
            }

            // Apply mask & compute rowmax(S)
            HVX_सदिशम् v_slope0 = Q6_V_vzero();
            HVX_सदिशम् v_slope1 = Q6_V_vzero();
            यदि (has_alibi) {
                v_slope0 = hvx_vec_repl_f16(Q6_V_vror_VR(v_slopes, r_vec_off * 2));
                v_slope1 = (r + 1 < n_पङ्क्तिकाः_g) ? hvx_vec_repl_f16(Q6_V_vror_VR(v_slopes, (r_vec_off + 1) * 2)) : Q6_V_vzero();
            }

            ध्रुव HVX_सदिशम् v_देहली = Q6_Vh_vsplat_R(0xcc00);  // fp16 -16.0

            HVX_सदिशम् v_s_rowmax0 = v_neg_inf;
            HVX_सदिशम् v_s_rowmax1 = v_neg_inf;
            यदि (has_आवरणम्) {
                प्रति (size_t c = 0; c < kv_पङ्क्तिकाः; c += 64) {
                    size_t         ci          = c / 64;
                    ध्रुव size_t   ne          = hex_smin(kv_पङ्क्तिकाः - c, 64);
                    HVX_सदिशम्_Pred q_पुच्छ_keep = Q6_Q_vsetq2_R(ne * परिमाण(__fp16));

                    HVX_सदिशम् v_mask0, v_mask1;

                    यदि (आवरणम्_प्रसारणम्) {
                        यदि (is_g1) {
                            ध्रुव size_t qi0 = r + 0;
                            v_mask0 = *(ध्रुव HVX_सदिशम् *) (तर्काः_2->आवरणम्_vtcm + qi0 * तर्काः_2->आवरणम्_vtcm_row_stride + c);
                            v_mask1 = v_neg_inf;
                            यदि (r + 1 < n_पङ्क्तिकाः_g) {
                                ध्रुव size_t qi1 = r + 1;
                                v_mask1 = *(ध्रुव HVX_सदिशम् *) (तर्काः_2->आवरणम्_vtcm + qi1 * तर्काः_2->आवरणम्_vtcm_row_stride + c);
                            }
                        } अन्यथा {
                            ध्रुव size_t qi0 = fastdiv(r + 0, &factx->div_G);
                            v_mask0 = *(ध्रुव HVX_सदिशम् *) (तर्काः_2->आवरणम्_vtcm + qi0 * तर्काः_2->आवरणम्_vtcm_row_stride + c);
                            v_mask1 = v_neg_inf;
                            यदि (r + 1 < n_पङ्क्तिकाः_g) {
                                ध्रुव size_t qi1 = fastdiv(r + 1, &factx->div_G);
                                यदि (qi1 == qi0) {
                                    v_mask1 = v_mask0;
                                } अन्यथा {
                                    v_mask1 = *(ध्रुव HVX_सदिशम् *) (तर्काः_2->आवरणम्_vtcm + qi1 * तर्काः_2->आवरणम्_vtcm_row_stride + c);
                                }
                            }
                        }
                    } अन्यथा {
                        // Head-dependent mask: pre-interleaved per row r.
                        ध्रुव size_t r0 = r + 0;
                        v_mask0 = *(ध्रुव HVX_सदिशम् *) (तर्काः_2->आवरणम्_vtcm + r0 * तर्काः_2->आवरणम्_vtcm_row_stride + c);
                        v_mask1 = v_neg_inf;
                        यदि (r + 1 < n_पङ्क्तिकाः_g) {
                            ध्रुव size_t r1 = r + 1;
                            v_mask1 = *(ध्रुव HVX_सदिशम् *) (तर्काः_2->आवरणम्_vtcm + r1 * तर्काः_2->आवरणम्_vtcm_row_stride + c);
                        }
                    }

                    // Threshold: mask values below -16.0 are treated as -inf (causal mask).
                    HVX_सदिशम्_Pred q_keep0 = Q6_Q_and_QQ(Q6_Q_vcmp_gt_VhfVhf(v_mask0, v_देहली), q_पुच्छ_keep);
                    HVX_सदिशम्_Pred q_keep1 = Q6_Q_and_QQ(Q6_Q_vcmp_gt_VhfVhf(v_mask1, v_देहली), q_पुच्छ_keep);

                    // Scale mask values by log2(e) for base-2 calculations
                    ध्रुव HVX_सदिशम् v_log2e = hvx_vec_splat_f16(EXP_LOG2E_F);
                    HVX_सदिशम् v_mask0_scaled = hvx_vec_mul_f16_f16(v_mask0, v_log2e);
                    HVX_सदिशम् v_mask1_scaled = hvx_vec_mul_f16_f16(v_mask1, v_log2e);

                    यदि (has_alibi) {
                        HVX_सदिशम् v_sm0 = hvx_vec_mul_f16_f16(v_mask0_scaled, v_slope0);
                        HVX_सदिशम् v_sm1 = hvx_vec_mul_f16_f16(v_mask1_scaled, v_slope1);
                        my_row_buf0[ci]  = Q6_V_vmux_QVV(q_keep0, hvx_vec_सङ्कलय_f16_f16(my_row_buf0[ci], v_sm0), v_neg_inf);
                        my_row_buf1[ci]  = Q6_V_vmux_QVV(q_keep1, hvx_vec_सङ्कलय_f16_f16(my_row_buf1[ci], v_sm1), v_neg_inf);
                    } अन्यथा {
                        my_row_buf0[ci] = Q6_V_vmux_QVV(q_keep0, hvx_vec_सङ्कलय_f16_f16(my_row_buf0[ci], v_mask0_scaled), v_neg_inf);
                        my_row_buf1[ci] = Q6_V_vmux_QVV(q_keep1, hvx_vec_सङ्कलय_f16_f16(my_row_buf1[ci], v_mask1_scaled), v_neg_inf);
                    }

                    v_s_rowmax0 = Q6_Vhf_vmax_VhfVhf(v_s_rowmax0, my_row_buf0[ci]);
                    v_s_rowmax1 = Q6_Vhf_vmax_VhfVhf(v_s_rowmax1, my_row_buf1[ci]);
                }
            } अन्यथा {
                size_t c = 0;
                प्रति (; c + 64 < kv_पङ्क्तिकाः; c += 128) {
                    size_t ci0 = c / 64;
                    size_t ci1 = ci0 + 1;
                    v_s_rowmax0 = Q6_Vhf_vmax_VhfVhf(v_s_rowmax0, my_row_buf0[ci0]);
                    v_s_rowmax1 = Q6_Vhf_vmax_VhfVhf(v_s_rowmax1, my_row_buf1[ci0]);
                    v_s_rowmax0 = Q6_Vhf_vmax_VhfVhf(v_s_rowmax0, my_row_buf0[ci1]);
                    v_s_rowmax1 = Q6_Vhf_vmax_VhfVhf(v_s_rowmax1, my_row_buf1[ci1]);
                }
                प्रति (; c < kv_पङ्क्तिकाः; c += 64) {
                    size_t         ci          = c / 64;
                    ध्रुव size_t   ne          = hex_smin(kv_पङ्क्तिकाः - c, 64);
                    HVX_सदिशम्_Pred q_पुच्छ_keep = Q6_Q_vsetq2_R(ne * परिमाण(__fp16));
                    यदि (ne < 64) {
                        my_row_buf0[ci] = Q6_V_vmux_QVV(q_पुच्छ_keep, my_row_buf0[ci], v_neg_inf);
                        my_row_buf1[ci] = Q6_V_vmux_QVV(q_पुच्छ_keep, my_row_buf1[ci], v_neg_inf);
                    }
                    v_s_rowmax0 = Q6_Vhf_vmax_VhfVhf(v_s_rowmax0, my_row_buf0[ci]);
                    v_s_rowmax1 = Q6_Vhf_vmax_VhfVhf(v_s_rowmax1, my_row_buf1[ci]);
                }
            }

            v_s_rowmax0 = hvx_vec_reduce_max_f16(v_s_rowmax0);
            v_s_rowmax1 = hvx_vec_reduce_max_f16(v_s_rowmax1);

            // Splat m_prev[r], m_prev[r+1] from the float per-row accumulators and convert to fp16 vectors
            HVX_सदिशम् v_m_prev0, v_m_prev1;
            यदि (r_vec_off < 32) {
                HVX_सदिशम् v0 = hvx_vec_repl_f32(Q6_V_vror_VR(m_पूर्व_v0, r_vec_off * 4));
                v_m_prev0 = hvx_vec_f32_गन्तव्यम्_f16(v0, v0);
                यदि (r + 1 < n_पङ्क्तिकाः_g) {
                    HVX_सदिशम् v1 = hvx_vec_repl_f32(Q6_V_vror_VR(m_पूर्व_v0, (r_vec_off + 1) * 4));
                    v_m_prev1 = hvx_vec_f32_गन्तव्यम्_f16(v1, v1);
                } अन्यथा {
                    v_m_prev1 = Q6_V_vzero();
                }
            } अन्यथा {
                HVX_सदिशम् v0 = hvx_vec_repl_f32(Q6_V_vror_VR(m_पूर्व_v1, (r_vec_off - 32) * 4));
                v_m_prev0 = hvx_vec_f32_गन्तव्यम्_f16(v0, v0);
                यदि (r + 1 < n_पङ्क्तिकाः_g) {
                    HVX_सदिशम् v1 = hvx_vec_repl_f32(Q6_V_vror_VR(m_पूर्व_v1, (r_vec_off + 1 - 32) * 4));
                    v_m_prev1 = hvx_vec_f32_गन्तव्यम्_f16(v1, v1);
                } अन्यथा {
                    v_m_prev1 = Q6_V_vzero();
                }
            }

            HVX_सदिशम् v_dup_m0 = Q6_Vhf_vmax_VhfVhf(v_m_prev0, v_s_rowmax0);
            HVX_सदिशम् v_dup_m1 = Q6_Vhf_vmax_VhfVhf(v_m_prev1, v_s_rowmax1);

            // Insert row r, r+1 rowmax into rowmax_acc_v
            {
                HVX_सदिशम्_Pred p_start = Q6_Q_vsetq_R(r_vec_off * 2);
                HVX_सदिशम्_Pred p_mid   = Q6_Q_vsetq_R((r_vec_off + 1) * 2);
                HVX_सदिशम्_Pred p_अन्तः   = Q6_Q_vsetq2_R((r_vec_off + 2) * 2);
                HVX_सदिशम्_Pred p_lane0 = Q6_Q_and_QQn(p_mid, p_start);
                HVX_सदिशम्_Pred p_lane1 = Q6_Q_and_QQn(p_अन्तः, p_mid);
                rowmax_acc_v           = Q6_V_vmux_QVV(p_lane0, v_dup_m0, rowmax_acc_v);
                rowmax_acc_v           = Q6_V_vmux_QVV(p_lane1, v_dup_m1, rowmax_acc_v);
            }

            // Compute P = exp(S - m_new)
            ध्रुव HVX_सदिशम् v_zero      = Q6_V_vzero();
            HVX_सदिशम्       v_p_rowsum0 = v_zero;
            HVX_सदिशम्       v_p_rowsum1 = v_zero;

            size_t c = 0;
            प्रति (; c + 64 < kv_पङ्क्तिकाः; c += 128) {
                size_t     ci0          = c / 64;
                size_t     ci1          = ci0 + 1;

                HVX_सदिशम् v_s_minus_m0_0 = Q6_Vqf16_vsub_VhfVhf(my_row_buf0[ci0], v_dup_m0);
                HVX_सदिशम् v_s_minus_m1_0 = Q6_Vqf16_vsub_VhfVhf(my_row_buf1[ci0], v_dup_m1);
                HVX_सदिशम् v_s_minus_m0_1 = Q6_Vqf16_vsub_VhfVhf(my_row_buf0[ci1], v_dup_m0);
                HVX_सदिशम् v_s_minus_m1_1 = Q6_Vqf16_vsub_VhfVhf(my_row_buf1[ci1], v_dup_m1);

                HVX_सदिशम् v_p_row0_hf_0  = hvx_vec_exp2_f16(Q6_Vhf_equals_Vqf16(v_s_minus_m0_0));
                HVX_सदिशम् v_p_row1_hf_0  = hvx_vec_exp2_f16(Q6_Vhf_equals_Vqf16(v_s_minus_m1_0));
                HVX_सदिशम् v_p_row0_hf_1  = hvx_vec_exp2_f16(Q6_Vhf_equals_Vqf16(v_s_minus_m0_1));
                HVX_सदिशम् v_p_row1_hf_1  = hvx_vec_exp2_f16(Q6_Vhf_equals_Vqf16(v_s_minus_m1_1));

                __fp16 *     बहिर्गमनम्_dtile0  = p_st_आधारः + ci0 * HMX_FP16_TILE_N_ELMS * 2;
                __fp16 *     बहिर्गमनम्_dtile1  = p_st_आधारः + ci1 * HMX_FP16_TILE_N_ELMS * 2;
                HVX_सदिशम् * pv_p_out0_0  = ((HVX_सदिशम् *) बहिर्गमनम्_dtile0) + r1 / 2;
                HVX_सदिशम् * pv_p_out1_0  = pv_p_out0_0 + 16;
                HVX_सदिशम् * pv_p_out0_1  = ((HVX_सदिशम् *) बहिर्गमनम्_dtile1) + r1 / 2;
                HVX_सदिशम् * pv_p_out1_1  = pv_p_out0_1 + 16;

                HVX_सदिशम्_Pair vp_p_dual0 = Q6_W_vshuff_VVR(v_p_row1_hf_0, v_p_row0_hf_0, -2);
                *pv_p_out0_0               = Q6_V_lo_W(vp_p_dual0);
                *pv_p_out1_0               = Q6_V_hi_W(vp_p_dual0);

                HVX_सदिशम्_Pair vp_p_dual1 = Q6_W_vshuff_VVR(v_p_row1_hf_1, v_p_row0_hf_1, -2);
                *pv_p_out0_1               = Q6_V_lo_W(vp_p_dual1);
                *pv_p_out1_1               = Q6_V_hi_W(vp_p_dual1);

                HVX_सदिशम्_Pair vp_p0_0 = hvx_vec_f16_गन्तव्यम्_f32_shuff(v_p_row0_hf_0);
                HVX_सदिशम्_Pair vp_p1_0 = hvx_vec_f16_गन्तव्यम्_f32_shuff(v_p_row1_hf_0);
                HVX_सदिशम्_Pair vp_p0_1 = hvx_vec_f16_गन्तव्यम्_f32_shuff(v_p_row0_hf_1);
                HVX_सदिशम्_Pair vp_p1_1 = hvx_vec_f16_गन्तव्यम्_f32_shuff(v_p_row1_hf_1);

                v_p_rowsum0 = Q6_Vqf32_vadd_Vqf32Vqf32(v_p_rowsum0, Q6_Vqf32_vadd_VsfVsf(Q6_V_lo_W(vp_p0_0), Q6_V_hi_W(vp_p0_0)));
                v_p_rowsum0 = Q6_Vqf32_vadd_Vqf32Vqf32(v_p_rowsum0, Q6_Vqf32_vadd_VsfVsf(Q6_V_lo_W(vp_p0_1), Q6_V_hi_W(vp_p0_1)));
                v_p_rowsum1 = Q6_Vqf32_vadd_Vqf32Vqf32(v_p_rowsum1, Q6_Vqf32_vadd_VsfVsf(Q6_V_lo_W(vp_p1_0), Q6_V_hi_W(vp_p1_0)));
                v_p_rowsum1 = Q6_Vqf32_vadd_Vqf32Vqf32(v_p_rowsum1, Q6_Vqf32_vadd_VsfVsf(Q6_V_lo_W(vp_p1_1), Q6_V_hi_W(vp_p1_1)));
            }
            प्रति (size_t c_rem = c; c_rem < kv_पङ्क्तिकाः; c_rem += 64) {
                size_t     ci           = c_rem / 64;
                HVX_सदिशम् v_s_minus_m0 = Q6_Vqf16_vsub_VhfVhf(my_row_buf0[ci], v_dup_m0);
                HVX_सदिशम् v_s_minus_m1 = Q6_Vqf16_vsub_VhfVhf(my_row_buf1[ci], v_dup_m1);

                HVX_सदिशम् v_p_row0_hf  = hvx_vec_exp2_f16(Q6_Vhf_equals_Vqf16(v_s_minus_m0));
                HVX_सदिशम् v_p_row1_hf  = hvx_vec_exp2_f16(Q6_Vhf_equals_Vqf16(v_s_minus_m1));
                __fp16 *     बहिर्गमनम्_dtile  = p_st_आधारः + ci * HMX_FP16_TILE_N_ELMS * 2;
                HVX_सदिशम् * pv_p_out0  = ((HVX_सदिशम् *) बहिर्गमनम्_dtile) + r1 / 2;
                HVX_सदिशम् * pv_p_out1  = pv_p_out0 + 16;

                HVX_सदिशम्_Pair vp_p_dual = Q6_W_vshuff_VVR(v_p_row1_hf, v_p_row0_hf, -2);
                *pv_p_out0               = Q6_V_lo_W(vp_p_dual);
                *pv_p_out1               = Q6_V_hi_W(vp_p_dual);

                HVX_सदिशम्_Pair vp_p0 = hvx_vec_f16_गन्तव्यम्_f32_shuff(v_p_row0_hf);
                HVX_सदिशम्_Pair vp_p1 = hvx_vec_f16_गन्तव्यम्_f32_shuff(v_p_row1_hf);

                v_p_rowsum0 = Q6_Vqf32_vadd_Vqf32Vqf32(v_p_rowsum0, Q6_Vqf32_vadd_VsfVsf(Q6_V_lo_W(vp_p0), Q6_V_hi_W(vp_p0)));
                v_p_rowsum1 = Q6_Vqf32_vadd_Vqf32Vqf32(v_p_rowsum1, Q6_Vqf32_vadd_VsfVsf(Q6_V_lo_W(vp_p1), Q6_V_hi_W(vp_p1)));
            }

            HVX_सदिशम् rowsum0_sf = hvx_vec_reduce_योग_f32(Q6_Vsf_equals_Vqf32(v_p_rowsum0));
            HVX_सदिशम् rowsum1_sf = hvx_vec_reduce_योग_f32(Q6_Vsf_equals_Vqf32(v_p_rowsum1));
            {
                HVX_सदिशम् rv0_v = hvx_vec_f32_गन्तव्यम्_f16(rowsum0_sf, rowsum0_sf);
                HVX_सदिशम् rv1_v = hvx_vec_f32_गन्तव्यम्_f16(rowsum1_sf, rowsum1_sf);

                HVX_सदिशम्_Pred p_start = Q6_Q_vsetq_R(r_vec_off * 2);
                HVX_सदिशम्_Pred p_mid   = Q6_Q_vsetq_R((r_vec_off + 1) * 2);
                HVX_सदिशम्_Pred p_अन्तः   = Q6_Q_vsetq2_R((r_vec_off + 2) * 2);
                HVX_सदिशम्_Pred p_lane0 = Q6_Q_and_QQn(p_mid, p_start);
                HVX_सदिशम्_Pred p_lane1 = Q6_Q_and_QQn(p_अन्तः, p_mid);
                rowsum_acc_v           = Q6_V_vmux_QVV(p_lane0, rv0_v, rowsum_acc_v);
                rowsum_acc_v           = Q6_V_vmux_QVV(p_lane1, rv1_v, rowsum_acc_v);
            }
        }

        // Inline fa_ml_update_and_build_d for this vector (lock-free and in parallel)
        HVX_सदिशम्_Pair rowmax_acc_pair    = hvx_vec_f16_गन्तव्यम्_f32(rowmax_acc_v);
        HVX_सदिशम्     v_rowmax_acc_f32_0 = Q6_V_lo_W(rowmax_acc_pair);
        HVX_सदिशम्     v_rowmax_acc_f32_1 = Q6_V_hi_W(rowmax_acc_pair);

        HVX_सदिशम् v_m_curr0 = Q6_Vsf_vmax_VsfVsf(m_पूर्व_v0, v_rowmax_acc_f32_0);
        HVX_सदिशम् v_m_curr1 = Q6_Vsf_vmax_VsfVsf(m_पूर्व_v1, v_rowmax_acc_f32_1);

        HVX_सदिशम् v_m_diff0 = HVX_OP_SUB_F32(m_पूर्व_v0, v_m_curr0);
        HVX_सदिशम् v_m_diff1 = HVX_OP_SUB_F32(m_पूर्व_v1, v_m_curr1);

        HVX_सदिशम् v_m_diff_f16   = hvx_vec_f32_गन्तव्यम्_f16(v_m_diff0, v_m_diff1);
        HVX_सदिशम् exp_m_diff_f16 = hvx_vec_exp2_f16(v_m_diff_f16);

        HVX_सदिशम्_Pair exp_m_diff_pair = hvx_vec_f16_गन्तव्यम्_f32(exp_m_diff_f16);
        HVX_सदिशम् exp_m_diff0 = Q6_V_lo_W(exp_m_diff_pair);
        HVX_सदिशम् exp_m_diff1 = Q6_V_hi_W(exp_m_diff_pair);

        HVX_सदिशम्_Pair rowsum_acc_pair = hvx_vec_f16_गन्तव्यम्_f32(rowsum_acc_v);
        HVX_सदिशम्     v_rowsum_acc_f32_0 = Q6_V_lo_W(rowsum_acc_pair);
        HVX_सदिशम्     v_rowsum_acc_f32_1 = Q6_V_hi_W(rowsum_acc_pair);

        HVX_सदिशम् l_पूर्व_v0 = factx->vtcm_l_vec[r_vec_idx * 2 + 0];
        HVX_सदिशम् l_पूर्व_v1 = factx->vtcm_l_vec[r_vec_idx * 2 + 1];
        HVX_सदिशम् v_l_curr0 = HVX_OP_सङ्कलय_F32(HVX_OP_MUL_F32(l_पूर्व_v0, exp_m_diff0), v_rowsum_acc_f32_0);
        HVX_सदिशम् v_l_curr1 = HVX_OP_सङ्कलय_F32(HVX_OP_MUL_F32(l_पूर्व_v1, exp_m_diff1), v_rowsum_acc_f32_1);

        factx->vtcm_m_vec[r_vec_idx * 2 + 0] = v_m_curr0;
        factx->vtcm_m_vec[r_vec_idx * 2 + 1] = v_m_curr1;
        factx->vtcm_l_vec[r_vec_idx * 2 + 0] = v_l_curr0;
        factx->vtcm_l_vec[r_vec_idx * 2 + 1] = v_l_curr1;

        // Build diagonal tile D = diag(exp(m_diff))
        ध्रुव HVX_सदिशम्     v_offsets = *(ध्रुव HVX_सदिशम् *) d_tile_scatter_offsets;
        ध्रुव HVX_सदिशम्_Pred q_32_आवरणम् = Q6_Q_vsetq_R(32 * परिमाण(__fp16));
        HVX_सदिशम्           v_exp_m_diff = exp_m_diff_f16;

        __fp16 * ध्रुव d_tiles_बहिर्गमनम् = factx->vtcm_d_tiles[तर्काः_2->buf_idx];

        size_t t0 = r_vec_idx * 2;
        यदि (t0 < तर्काः_2->n_row_tiles) {
            ध्रुव HVX_सदिशम् v_content = v_exp_m_diff;
            __fp16 *         बहिर्गमनम्_आधारः  = d_tiles_बहिर्गमनम् + t0 * HMX_FP16_TILE_N_ELMS;
            Q6_vscatter_QRMVhV(q_32_आवरणम्, (size_t) बहिर्गमनम्_आधारः, HMX_FP16_TILE_आकार - 1, v_offsets, v_content);
        }

        size_t t1 = r_vec_idx * 2 + 1;
        यदि (t1 < तर्काः_2->n_row_tiles) {
            ध्रुव HVX_सदिशम् v_content = Q6_V_vror_VR(v_exp_m_diff, 64);
            __fp16 *         बहिर्गमनम्_आधारः  = d_tiles_बहिर्गमनम् + t1 * HMX_FP16_TILE_N_ELMS;
            Q6_vscatter_QRMVhV(q_32_आवरणम्, (size_t) बहिर्गमनम्_आधारः, HMX_FP16_TILE_आकार - 1, v_offsets, v_content);
        }
    }
    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_FA_SFM, (uint16_t) (तर्काः_2->q_start * G + vec_start * 64));
}

स्थिर शून्य fa_softmax_thread_nomask(अचिह्नित पूर्णाङ्क n, अचिह्नित पूर्णाङ्क i, शून्य * दत्त) {
    fa_softmax_impl(n, i, दत्त,
                    /*has_mask=*/false,
                    /*mask_broadcast=*/false,
                    /*is_g1=*/false,
                    /*has_alibi=*/false,
                    /*has_softcap=*/false);
}

स्थिर शून्य fa_softmax_thread_आवरणम्_प्रसारणम्_g1(अचिह्नित पूर्णाङ्क n, अचिह्नित पूर्णाङ्क i, शून्य * दत्त) {
    fa_softmax_impl(n, i, दत्त,
                    /*has_mask=*/true,
                    /*mask_broadcast=*/true,
                    /*is_g1=*/true,
                    /*has_alibi=*/false,
                    /*has_softcap=*/false);
}

स्थिर शून्य fa_softmax_thread_आवरणम्_प्रसारणम्_gn(अचिह्नित पूर्णाङ्क n, अचिह्नित पूर्णाङ्क i, शून्य * दत्त) {
    fa_softmax_impl(n, i, दत्त,
                    /*has_mask=*/true,
                    /*mask_broadcast=*/true,
                    /*is_g1=*/false,
                    /*has_alibi=*/false,
                    /*has_softcap=*/false);
}

स्थिर शून्य fa_softmax_thread(अचिह्नित पूर्णाङ्क n, अचिह्नित पूर्णाङ्क i, शून्य * दत्त) {
    fa_softmax_तर्काः_t *     तर्काः_2  = (fa_softmax_तर्काः_t *) दत्त;
    संरचना hmx_fa_context * factx = तर्काः_2->factx;

    ध्रुव bool has_आवरणम्       = (तर्काः_2->आवरणम् != रिक्त);
    ध्रुव bool आवरणम्_प्रसारणम् = factx->आवरणम्_प्रसारणम्;
    ध्रुव bool is_g1          = (तर्काः_2->G == 1);
    ध्रुव bool has_alibi      = तर्काः_2->has_alibi;
    ध्रुव bool has_softcap    = factx->has_softcap;

    fa_softmax_impl(n, i, दत्त, has_आवरणम्, आवरणम्_प्रसारणम्, is_g1, has_alibi, has_softcap);
}

स्थिर अन्तःस्थ HVX_सदिशम् hvx_fa_आरोपय_sinks(ध्रुव भिन्न * sinks_दत्त, size_t idx, size_t गणनम्) {
    ध्रुव HVX_सदिशम् * v_src    = (ध्रुव HVX_सदिशम् *) sinks_दत्त + (idx >> 5);
    ध्रुव size_t       lane_off = idx & 31;
    ध्रुव size_t       byte_off = lane_off * परिमाण(भिन्न);
    प्रत्यागम (lane_off + गणनम् > 32)
        ? Q6_V_valign_VVR(v_src[1], v_src[0], byte_off)
        : Q6_V_vror_VR(v_src[0], byte_off);
}

स्थिर __attribute__((noinline)) शून्य fa_build_d_diag_inv_l(संरचना hmx_fa_context * factx,
                                                            size_t                  n_row_tiles,
                                                            size_t                  n_row_tiles_g_br,
                                                            uint32_t                kv_शीर्ष,
                                                            size_t                  n_पङ्क्तिकाः_g) {
    (शून्य) n_row_tiles_g_br;

    // sinks: l += exp(sink - m * ln2) in f32, m and l are in the log2 domain
    यदि (factx->sinks) {
        ध्रुव भिन्न *    sinks_दत्त   = factx->vtcm_sinks;
        ध्रुव size_t     G            = factx->G;
        ध्रुव size_t     शीर्ष_आधारः    = (size_t) kv_शीर्ष * G;
        ध्रुव HVX_सदिशम् v_ln2        = Q6_V_vsplat_R(EXP_LOGN2);
        ध्रुव HVX_सदिशम् v_शीर्ष_sinks = (G <= 32) ? hvx_fa_आरोपय_sinks(sinks_दत्त, शीर्ष_आधारः, G) : Q6_V_vzero();
        HVX_सदिशम्       v_sink       = Q6_V_vzero();
        size_t           g_पूर्व       = SIZE_MAX;

        प्रति (size_t i = 0; i < n_row_tiles; ++i) {
            // the sink pattern of a tile only depends on (i * 32) % G, rebuild it when that changes
            ध्रुव size_t r0 = i * 32;
            ध्रुव size_t g0 = fastmodulo(r0, G, &factx->div_G);
            यदि (g0 != g_पूर्व) {
                ध्रुव size_t rem_g = G - g0;
                यदि (G <= 32) {
                    ध्रुव HVX_सदिशम् v_lo = Q6_V_vror_VR(v_शीर्ष_sinks, g0 * परिमाण(भिन्न));
                    ध्रुव HVX_सदिशम् v_hi = Q6_V_vror_VR(v_शीर्ष_sinks, 128 - rem_g * परिमाण(भिन्न));
                    v_sink = Q6_V_vmux_QVV(Q6_Q_vsetq2_R(rem_g * परिमाण(भिन्न)), v_lo, v_hi);
                    प्रति (size_t len = G; len < 32; len <<= 1) {
                        ध्रुव HVX_सदिशम्_Pred q_keep    = Q6_Q_vsetq_R(len * परिमाण(भिन्न));
                        ध्रुव HVX_सदिशम्     v_shifted = Q6_V_vror_VR(v_sink, 128 - len * परिमाण(भिन्न));
                        v_sink = Q6_V_vmux_QVV(q_keep, v_sink, v_shifted);
                    }
                } अन्यथा यदि (rem_g >= 32) {
                    v_sink = hvx_fa_आरोपय_sinks(sinks_दत्त, शीर्ष_आधारः + g0, 32);
                } अन्यथा {
                    ध्रुव HVX_सदिशम् v_lo = hvx_fa_आरोपय_sinks(sinks_दत्त, शीर्ष_आधारः + g0, rem_g);
                    ध्रुव HVX_सदिशम् v_hi = hvx_fa_आरोपय_sinks(sinks_दत्त, शीर्ष_आधारः, 32 - rem_g);
                    v_sink = Q6_V_vmux_QVV(Q6_Q_vsetq_R(rem_g * परिमाण(भिन्न)), v_lo, Q6_V_vror_VR(v_hi, 128 - rem_g * परिमाण(भिन्न)));
                }
                g_पूर्व = g0;
            }

            HVX_सदिशम् v_d = HVX_OP_SUB_F32(v_sink, HVX_OP_MUL_F32(factx->vtcm_m_vec[i], v_ln2));
            यदि (r0 + 32 > n_पङ्क्तिकाः_g) {
                ध्रुव HVX_सदिशम्_Pred q_valid = Q6_Q_vsetq_R((n_पङ्क्तिकाः_g - r0) * परिमाण(भिन्न));
                v_d = Q6_V_vmux_QVV(q_valid, v_d, Q6_V_vzero());
            }
            factx->vtcm_l_vec[i] = HVX_OP_सङ्कलय_F32(factx->vtcm_l_vec[i], hvx_vec_exp_f32(v_d));
        }
    }

    ध्रुव HVX_सदिशम्     v_offsets = *(ध्रुव HVX_सदिशम् *) d_tile_scatter_offsets;
    ध्रुव HVX_सदिशम्_Pred q_32_आवरणम् = Q6_Q_vsetq_R(32 * परिमाण(__fp16));
    ध्रुव HVX_सदिशम्     one       = hvx_vec_splat_f32(1.0f);

    HVX_सदिशम् v_content = Q6_V_vzero();
    प्रति (size_t i = 0; i < n_row_tiles; ++i) {
        यदि ((i % 2) == 0) {
            HVX_सदिशम् inv_lo = HVX_OP_MUL_F32(one, hvx_vec_inverse_f32(factx->vtcm_l_vec[i]));
            HVX_सदिशम् inv_hi = (i + 1 < n_row_tiles) ? HVX_OP_MUL_F32(one, hvx_vec_inverse_f32(factx->vtcm_l_vec[i + 1])) : Q6_V_vzero();
            v_content = hvx_vec_f32_गन्तव्यम्_f16(inv_lo, inv_hi);
        } अन्यथा {
            v_content = Q6_V_vror_VR(v_content, 64);
        }

        __fp16 * बहिर्गमनम्_आधारः = factx->vtcm_d_inv_l + i * HMX_FP16_TILE_N_ELMS;
        Q6_vscatter_QRMVhV(q_32_आवरणम्, (size_t) बहिर्गमनम्_आधारः, HMX_FP16_TILE_आकार - 1, v_offsets, v_content);
    }
}

स्थिर शून्य fa_phase_softmax_and_build_d(संरचना hmx_fa_context * factx,
                                         fa_softmax_तर्काः_t *     sargs,
                                         size_t                  n_row_tiles,
                                         size_t                  n_row_tiles_g_br) {
    work_queue_t wp = factx->octx->ctx->work_queue;
    ध्रुव size_t n_row_vec_cnt = hmx_ceil_div(sargs->n_पङ्क्तिकाः_g, 64);

    कार्यकर्ता_callback_t softmax_fn = fa_softmax_thread;
    यदि (sargs->आवरणम् == रिक्त && !factx->has_softcap && !sargs->has_alibi) {
        softmax_fn = fa_softmax_thread_nomask;
    } अन्यथा यदि (sargs->आवरणम् != रिक्त && factx->आवरणम्_प्रसारणम् && !factx->has_softcap && !sargs->has_alibi) {
        यदि (sargs->G == 1) {
            softmax_fn = fa_softmax_thread_आवरणम्_प्रसारणम्_g1;
        } अन्यथा {
            softmax_fn = fa_softmax_thread_आवरणम्_प्रसारणम्_gn;
        }
    }

    यदि (factx->n_तन्तवः > 1 && n_row_vec_cnt >= 2) {
        uint32_t n_use = (uint32_t) hex_smin((size_t) factx->n_तन्तवः, n_row_vec_cnt);
        sargs->thread_div = init_fastdiv_मानानि(n_use);
        work_queue_सततम्(wp, softmax_fn, sargs, n_use);
    } अन्यथा {
        softmax_fn(1, 0, sargs);
    }
}

// ============================================================================
// HMX job structs and worker functions
// ============================================================================

प्रकारनिर्देश संरचना {
    ध्रुव __fp16 * q_tiles;
    ध्रुव __fp16 * k_tiles;
    __fp16 *       s_tiles;
    size_t         n_row_tiles;
    size_t         n_col_tiles;
    size_t         n_dot_tiles;  // DK / 32
    size_t         n_tiles_प्रत्येकम्_bc;
    uint8_t *      hmx_scales;
} hmx_fa_qk_job_t;

स्थिर शून्य hmx_fa_qk_dot_कार्यकर्ता(शून्य * दत्त) {
    hmx_fa_qk_job_t * job            = (hmx_fa_qk_job_t *) दत्त;
    ध्रुव size_t      n_row_tiles    = job->n_row_tiles;
    ध्रुव size_t      n_col_tiles    = job->n_col_tiles;
    ध्रुव size_t      n_dot_tiles    = job->n_dot_tiles;
    ध्रुव size_t      n_tiles_प्रत्येकम्_bc = job->n_tiles_प्रत्येकम्_bc;
    ध्रुव __fp16 * सीमित q_tiles  = job->q_tiles;
    ध्रुव __fp16 * सीमित k_tiles  = job->k_tiles;
    __fp16 * सीमित s_tiles        = job->s_tiles;
    __builtin_assume(n_row_tiles > 0);
    __builtin_assume(n_col_tiles > 0);
    __builtin_assume(n_dot_tiles > 0);

    asm चञ्चल(HMX_समुच्चयः_BIAS("%0") :: "r"((अचिह्नित पूर्णाङ्क)job->hmx_scales));
    ध्रुव size_t dot_stride = n_dot_tiles * HMX_FP16_TILE_N_ELMS;
    प्रति (size_t r = 0; r < n_row_tiles; ++r) {
        ध्रुव __fp16 * row_tiles = q_tiles + r * dot_stride;
        ध्रुव __fp16 * col_tiles = k_tiles;
        __fp16 *       बहिर्गमनम्_tile  = s_tiles + r * n_tiles_प्रत्येकम्_bc * HMX_FP16_TILE_N_ELMS;

        प्रति (size_t c = 0; c < n_col_tiles; ++c) {
            hmx_fa_qk_dot_tile(row_tiles, col_tiles, बहिर्गमनम्_tile, n_dot_tiles);
            col_tiles += dot_stride;
            बहिर्गमनम्_tile  += HMX_FP16_TILE_N_ELMS;
        }
    }
}

प्रकारनिर्देश संरचना {
    __fp16 *       o_curr;
    ध्रुव __fp16 * o_पूर्व;
    ध्रुव __fp16 * p_tiles;
    ध्रुव __fp16 * v_tiles;
    ध्रुव __fp16 * d_tiles;
    uint8_t *      hmx_scales;
    size_t         n_row_tiles;
    size_t         n_col_tiles;
    size_t         n_row_tiles_g_br;
    size_t         n_tiles_प्रत्येकम्_bc;
    size_t         DV;
} hmx_fa_o_update_job_t;

स्थिर शून्य hmx_fa_o_update_कार्यकर्ता(शून्य * दत्त) {
    hmx_fa_o_update_job_t * job              = (hmx_fa_o_update_job_t *) दत्त;
    ध्रुव size_t            n_row_tiles      = job->n_row_tiles;
    ध्रुव size_t            n_col_tiles      = job->n_col_tiles;
    ध्रुव size_t            n_row_tiles_g_br = job->n_row_tiles_g_br;
    ध्रुव size_t            n_tiles_प्रत्येकम्_bc   = job->n_tiles_प्रत्येकम्_bc;
    ध्रुव size_t            DV_tiles         = job->DV / 32;
    ध्रुव __fp16 * सीमित d_tiles          = job->d_tiles;
    ध्रुव __fp16 * सीमित p_tiles          = job->p_tiles;
    ध्रुव __fp16 * सीमित v_tiles          = job->v_tiles;
    ध्रुव __fp16 * सीमित o_पूर्व           = job->o_पूर्व;
    __fp16 * सीमित o_curr                 = job->o_curr;
    __builtin_assume(n_row_tiles > 0);
    __builtin_assume(n_col_tiles > 0);
    __builtin_assume(DV_tiles > 0);

    asm चञ्चल(HMX_समुच्चयः_BIAS("%0") :: "r"((अचिह्नित पूर्णाङ्क)job->hmx_scales));
    ध्रुव size_t o_stride = n_row_tiles_g_br * HMX_FP16_TILE_N_ELMS;
    ध्रुव size_t v_stride = n_tiles_प्रत्येकम्_bc * HMX_FP16_TILE_N_ELMS;
    प्रति (size_t r = 0; r < n_row_tiles; ++r) {
        ध्रुव __fp16 * d_diag     = d_tiles + r * HMX_FP16_TILE_N_ELMS;
        ध्रुव __fp16 * p_tile_अन्तर्  = p_tiles + (r * n_tiles_प्रत्येकम्_bc) * HMX_FP16_TILE_N_ELMS;
        ध्रुव __fp16 * o_rc       = o_पूर्व + r * HMX_FP16_TILE_N_ELMS;
        ध्रुव __fp16 * v_tile_अन्तर्  = v_tiles;
        __fp16       * o_tile_बहिर्गमनम् = o_curr + r * HMX_FP16_TILE_N_ELMS;

        प्रति (size_t c = 0; c < DV_tiles; ++c) {
            hmx_fa_o_update_tile(d_diag, o_rc, p_tile_अन्तर्, v_tile_अन्तर्, o_tile_बहिर्गमनम्, n_col_tiles);
            o_rc       += o_stride;
            v_tile_अन्तर्  += v_stride;
            o_tile_बहिर्गमनम् += o_stride;
        }
    }
}

प्रकारनिर्देश संरचना {
    __fp16 *       o_curr;   // output (row-major tile layout)
    ध्रुव __fp16 * o_पूर्व;   // input (column-major tile layout)
    ध्रुव __fp16 * d_tiles;  // diag(1/l) tiles
    uint8_t *      hmx_scales;
    size_t         n_row_tiles;
    size_t         n_row_tiles_g_br;
    size_t         DV;
} hmx_fa_o_norm_job_t;

स्थिर शून्य hmx_fa_o_norm_कार्यकर्ता(शून्य * दत्त) {
    hmx_fa_o_norm_job_t * job              = (hmx_fa_o_norm_job_t *) दत्त;
    ध्रुव size_t          n_row_tiles      = job->n_row_tiles;
    ध्रुव size_t          n_row_tiles_g_br = job->n_row_tiles_g_br;
    ध्रुव size_t          DV_tiles         = job->DV / 32;
    ध्रुव __fp16 * सीमित d_tiles        = job->d_tiles;
    ध्रुव __fp16 * सीमित o_पूर्व         = job->o_पूर्व;
    __fp16 * सीमित o_curr               = job->o_curr;
    __builtin_assume(n_row_tiles > 0);
    __builtin_assume(DV_tiles > 0);

    asm चञ्चल(HMX_समुच्चयः_BIAS("%0") :: "r"((अचिह्नित पूर्णाङ्क)job->hmx_scales));
    ध्रुव size_t o_stride = n_row_tiles_g_br * HMX_FP16_TILE_N_ELMS;
    प्रति (size_t r = 0; r < n_row_tiles; ++r) {
        ध्रुव __fp16 * d_diag = d_tiles + r * HMX_FP16_TILE_N_ELMS;
        ध्रुव __fp16 * o_rc = o_पूर्व + r * HMX_FP16_TILE_N_ELMS;
        __fp16 *       o_बहिर्गमनम् = o_curr + r * DV_tiles * HMX_FP16_TILE_N_ELMS;

        प्रति (size_t c = 0; c < DV_tiles; ++c) {
            hmx_fa_o_norm_tile(d_diag, o_rc, o_बहिर्गमनम्);
            o_rc  += o_stride;
            o_बहिर्गमनम् += HMX_FP16_TILE_N_ELMS;
        }
    }
}

// Populate per-GQA-row ALiBi slopes for a given KV head.
स्थिर __attribute__((noinline)) शून्य fa_compute_slopes(
                              ध्रुव संरचना hmx_fa_context * factx,
                              uint32_t                      kv_शीर्ष,
                              size_t                        n_पङ्क्तिकाः_g) {
    __fp16 * slopes = factx->vtcm_slopes;
    यदि (factx->max_bias == 0.0f) {
        hvx_splat_f16_a(slopes, 1.0f, n_पङ्क्तिकाः_g);
        प्रत्यागम;
    }

    ध्रुव uint32_t G           = factx->G;
    ध्रुव uint32_t n_शीर्ष_log2 = factx->n_शीर्ष_log2;
    ध्रुव भिन्न    m0          = factx->m0;
    ध्रुव भिन्न    m1          = factx->m1;

    __fp16 अस्थायी_slopes[512] __attribute__((aligned(128)));
    यदि (G <= 32) {
        // Fast path: Compute G unique slope values in vector registers
        HVX_सदिशम् v_val = hvx_alibi_slopes(kv_शीर्ष, G, n_शीर्ष_log2, m0, m1);

        __fp16 अस्थायी_slopes_aligned[64] __attribute__((aligned(128)));
        hvx_vmem(अस्थायी_slopes_aligned) = hvx_vec_f32_गन्तव्यम्_f16(v_val, Q6_V_vzero());

        प्रति (uint32_t i = 0; i < G; ++i) {
            अस्थायी_slopes[i] = अस्थायी_slopes_aligned[i];
        }
    } अन्यथा {
        // Fallback path: G > 32 (rare configurations)
        प्रति (uint32_t i = 0; i < G; ++i) {
            अस्थायी_slopes[i] = (__fp16)alibi_नतिः(kv_शीर्ष * G + i, n_शीर्ष_log2, m0, m1);
        }
    }

    // Allocate stack buffer to avoid scalar writes to VTCM (which generates L2 misses)
    __fp16 local_slopes[n_पङ्क्तिकाः_g] __attribute__((aligned(128)));
    प्रति (size_t r = 0; r < n_पङ्क्तिकाः_g; ++r) {
        local_slopes[r] = अस्थायी_slopes[fastmodulo(r, G, &factx->div_G)];
    }

    // Copy to VTCM slopes using HVX block copy (both are aligned to 128 bytes)
    hvx_copy_f16_aa((uint8_t *)slopes, (ध्रुव uint8_t *)local_slopes, n_पङ्क्तिकाः_g);
}

स्थिर शून्य fa_प्रक्षिप_आवरणम्_dma_gqa(
    dma_queue *               dma_q,
    ध्रुव संरचना htp_tensor * आवरणम्,
    uint32_t                  q_start,
    uint32_t                  im3,
    uint32_t                  kv_start,
    uint32_t                  kv_शीर्ष,
    uint32_t                  G,
    uint32_t                  m_पङ्क्तिः_अष्टकानि,
    uint32_t                  kv_पङ्क्तिकाः,
    uint32_t                  n_पङ्क्तिकाः_q,
    संरचना hmx_fa_context *   factx
) {
    प्रति (uint32_t g = 0; g < G; ++g) {
        ध्रुव uint32_t h_idx = kv_शीर्ष * G + g;
        ध्रुव uint32_t im2 = fastmodulo(h_idx, आवरणम्->ne[2], &factx->src3_div2);
        ध्रुव dma_addr_t ms_src = आवरणम्->दत्त + q_start * आवरणम्->nb[1] +
                                  im2 * आवरणम्->nb[2] + im3 * आवरणम्->nb[3] + kv_start * परिमाण(__fp16);
        uint8_t * ms_dst = (uint8_t *) factx->vtcm_आवरणम्_buf + g * m_पङ्क्तिः_अष्टकानि;
        dma_queue_प्रक्षिप(dma_q, dma_make_दत्त(ms_dst, ms_src), G * m_पङ्क्तिः_अष्टकानि, आवरणम्->nb[1], kv_पङ्क्तिकाः * परिमाण(__fp16), n_पङ्क्तिकाः_q);
    }
}

स्थिर शून्य fa_निष्कासय_आवरणम्_dma_gqa(dma_queue * dma_q, uint32_t G) {
    प्रति (uint32_t g = 0; g < G; ++g) {
        dma_queue_निष्कासय(dma_q);
    }
}

स्थिर अन्तःस्थ शून्य fa_prefetch_खण्डम्(dma_queue * dma_q, ध्रुव संरचना htp_tensor * k, ध्रुव संरचना htp_tensor * v, ध्रुव संरचना htp_tensor * आवरणम्,
                                     uint32_t b, size_t Bc, size_t आकार_k_row_padded, size_t आकार_k_row, size_t आकार_v_row_padded, size_t आकार_v_row,
                                     uint32_t ik2, uint32_t ik3, uint32_t iv2, uint32_t iv3, uint32_t q_start, uint32_t im3, uint32_t kv_शीर्ष, uint32_t G,
                                     size_t m_पङ्क्तिः_अष्टकानि, size_t n_पङ्क्तिकाः_q, size_t nek1, size_t prefetch_buf, संरचना hmx_fa_context * factx) {
    ध्रुव uint32_t  prefetch_start = b * Bc;
    ध्रुव uint32_t  prefetch_पङ्क्तिकाः  = hex_smin(Bc, nek1 - prefetch_start);
    ध्रुव dma_addr_t k_prefetch_src = k->दत्त + prefetch_start * k->nb[1] + ik2 * k->nb[2] + ik3 * k->nb[3];
    dma_queue_प्रक्षिप(dma_q, dma_make_दत्त(factx->vtcm_k_fp16[prefetch_buf], k_prefetch_src), आकार_k_row_padded, k->nb[1], आकार_k_row, prefetch_पङ्क्तिकाः);
    ध्रुव dma_addr_t v_prefetch_src = v->दत्त + prefetch_start * v->nb[1] + iv2 * v->nb[2] + iv3 * v->nb[3];
    dma_queue_प्रक्षिप(dma_q, dma_make_दत्त(factx->vtcm_v_fp16[prefetch_buf], v_prefetch_src), आकार_v_row_padded, v->nb[1], आकार_v_row, prefetch_पङ्क्तिकाः);

    यदि (आवरणम्) {
        यदि (__builtin_expect(factx->आवरणम्_प्रसारणम्, true)) {
            ध्रुव dma_addr_t ms_src = आवरणम्->दत्त + q_start * आवरणम्->nb[1] + im3 * आवरणम्->nb[3] + prefetch_start * परिमाण(__fp16);
            dma_cache_fa_प्रक्षिप(dma_q, &factx->m_cache, ms_src, m_पङ्क्तिः_अष्टकानि, आवरणम्->nb[1], prefetch_पङ्क्तिकाः * परिमाण(__fp16), n_पङ्क्तिकाः_q);
        } अन्यथा {
            fa_प्रक्षिप_आवरणम्_dma_gqa(dma_q, आवरणम्, q_start, im3, prefetch_start, kv_शीर्ष, G, m_पङ्क्तिः_अष्टकानि, prefetch_पङ्क्तिकाः, n_पङ्क्तिकाः_q, factx);
        }
    }
}

// ============================================================================
// Core HMX flash attention algorithm (GQA-merged)
// ============================================================================

पूर्णाङ्क hmx_flash_attn_ext(संरचना htp_ops_context * octx) {
    संरचना htp_thread_trace * tr_hvx = &octx->ctx->trace[0];
    संरचना htp_thread_trace * tr_hmx = &octx->ctx->trace[HTP_MAX_NTHREADS];
    ध्रुव संरचना htp_tensor * q    = octx->src[0];
    ध्रुव संरचना htp_tensor * k    = octx->src[1];
    ध्रुव संरचना htp_tensor * v    = octx->src[2];
    ध्रुव संरचना htp_tensor * आवरणम् = (octx->src[3] && octx->src[3]->दत्त) ? octx->src[3] : रिक्त;
    ध्रुव संरचना htp_tensor * dst  = octx->dst;

    संरचना htp_context * ध्रुव ctx = octx->ctx;

    यदि (!ctx->hmx_enabled) {
        प्रत्यागम HTP_स्थितिः_NO_SUPPORT;
    }

    // Dimensions
    ध्रुव uint32_t neq0 = q->ne[0];  // head_dim (DK)
    ध्रुव uint32_t neq1 = q->ne[1];  // n_tokens
    ध्रुव uint32_t neq2 = q->ne[2];  // n_heads
    ध्रुव uint32_t neq3 = q->ne[3];  // n_seqs

    ध्रुव uint32_t nek0 = k->ne[0];  // head_dim
    ध्रुव uint32_t nek1 = k->ne[1];  // kv_len

    ध्रुव uint32_t nev0 = v->ne[0];  // head_dim (DV)

    ध्रुव uint32_t DK = neq0;
    ध्रुव uint32_t DV = nev0;

    // HMX tiles head_dim in units of 64. head_dim need not be 64- (or 32-) aligned:
    // we can operate on DK/DV rounded up to 64 with tail lanes [D, D_pad) zero-filled.
    ध्रुव uint32_t DK_pad = hex_आवर्तनम्_ऊर्ध्वम्(DK, 64);
    ध्रुव uint32_t DV_pad = hex_आवर्तनम्_ऊर्ध्वम्(DV, 64);
    यदि (DK == 0 || DV == 0) {
        प्रत्यागम HTP_स्थितिः_NO_SUPPORT;
    }

    ध्रुव संरचना htp_fa_kernel_params * kparams = (ध्रुव संरचना htp_fa_kernel_params *) octx->kernel_params;
    ध्रुव uint32_t n_kv_heads = k->ne[2];

    // ======== Build context ========
    संरचना hmx_fa_context factx;
    स्मृतिं_पूरय(&factx, 0, परिमाण(factx));
    factx.octx           = octx;
    factx.sinks          = octx->src[4];  // NULL if this op has no attention sinks
    factx.k              = k;
    factx.v              = v;
    factx.n_तन्तवः      = kparams->n_तन्तवः;
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
    factx.n_kv_खण्डानि    = kparams->n_kv_खण्डानि;
    factx.is_q_fp32      = (q->type == HTP_TYPE_F32);
    factx.is_dst_fp32    = (dst->type == HTP_TYPE_F32);
    factx.pipeline       = (kparams->u.hmx.pipeline != 0);
    factx.आवरणम्_प्रसारणम् = (kparams->u.hmx.आवरणम्_प्रसारणम् != 0);
    यदि (आवरणम्) {
        factx.src3_div2  = kparams->src3_div2;
        factx.src3_div3  = kparams->src3_div3;
    }

    factx.has_softcap   = (kparams->logit_softcap != 0.0f);
    factx.scale         = (__fp16) kparams->scale;
    factx.max_bias      = kparams->max_bias;
    factx.logit_softcap = 0;
    यदि (factx.has_softcap) {
        factx.logit_softcap = (__fp16) kparams->logit_softcap;
    }

    factx.n_शीर्ष_log2 = kparams->n_शीर्ष_log2;
    factx.m0          = kparams->m0;
    factx.m1          = kparams->m1;

    ध्रुव uint32_t Br = factx.Br;
    ध्रुव uint32_t Bc = factx.Bc;
    ध्रुव uint32_t g_br = factx.g_br;
    ध्रुव bool pipeline = factx.pipeline;
    ध्रुव uint32_t n_तन्तवः = factx.n_तन्तवः;
    ध्रुव uint32_t G = factx.G;

    // Multi-device: prefer head-parallel partitioning (each core owns a disjoint head
    // shard), falling back to Q-block (token) split when heads don't divide evenly.
    ध्रुव uint32_t n_q_खण्डानि = (neq1 + Br - 1) / Br;
    uint32_t q_start_min  = 0;
    uint32_t q_start_max  = neq1;
    uint32_t kv_शीर्ष_min  = 0;
    uint32_t kv_शीर्ष_max  = n_kv_heads;

    यदि (octx->ctx->mdev.गणनम् > 1) {
        ध्रुव uint32_t mdev_गणनम् = octx->ctx->mdev.गणनम्;
        ध्रुव uint32_t mdev_idx   = octx->ctx->mdev.idx;
        ध्रुव uint32_t dst_e_आकार = (dst->type == HTP_TYPE_F32) ? परिमाण(भिन्न) : परिमाण(__fp16);
        ध्रुव bool can_विभज      = htp_tensor_can_row_partition(dst, dst_e_आकार);

        यदि (kparams->शीर्ष_विभज && can_विभज && n_kv_heads >= mdev_गणनम् && n_kv_heads % mdev_गणनम् == 0) {
            ध्रुव uint32_t kv_प्रत्येकम्_core = n_kv_heads / mdev_गणनम्;
            kv_शीर्ष_min = mdev_idx * kv_प्रत्येकम्_core;
            kv_शीर्ष_max = kv_शीर्ष_min + kv_प्रत्येकम्_core;
        } अन्यथा {
            ध्रुव संरचना htp_tensor_mdev_range range = htp_tensor_mdev_partition(n_q_खण्डानि, can_विभज ? 1 : 0, mdev_idx, mdev_गणनम्, &octx->ctx->mdev.गणनम्_div);
            ध्रुव uint32_t खण्डम्_start = range.start;
            ध्रुव uint32_t खण्डम्_अन्तः   = range.start + range.गणनम्;

            यदि (खण्डम्_start >= खण्डम्_अन्तः) {
                प्रत्यागम HTP_स्थितिः_OK;
            }

            q_start_min = खण्डम्_start * Br;
            q_start_max = MIN(खण्डम्_अन्तः * Br, neq1);
        }
    }

    // ======== VTCM allocation (GQA-aware) ========
    // K/V row sizes drive the DMA descriptors (not the VTCM layout) and are used
    // throughout the KV loop below. The DMA copies only the real DK/DV columns; the
    // staging rows are padded to hold DK_pad/DV_pad columns (tail zero-filled below)
    // so the HMX interleave/tile logic can operate on 64-aligned head dims.
    ध्रुव size_t आकार_k_row        = htp_tensor_get_row_आकार(k->type, DK);
    ध्रुव size_t आकार_v_row        = htp_tensor_get_row_आकार(v->type, DV);
    ध्रुव size_t आकार_k_row_padded = hex_आवर्तनम्_ऊर्ध्वम्(DK_pad * परिमाण(__fp16), 128);
    ध्रुव size_t आकार_v_row_padded = hex_आवर्तनम्_ऊर्ध्वम्(DV_pad * परिमाण(__fp16), 128);

    // Build the VTCM layout once (shared with the host estimator) and place every
    // scratch buffer at its computed offset. Padded head dims size the HMX tiles.
    संरचना hmx_fa_vtcm_layout L;
    hmx_fa_vtcm_layout_build(&L, G, DK_pad, DV_pad, Br, Bc, n_तन्तवः, pipeline, factx.is_q_fp32, factx.sinks != रिक्त, factx.n_heads);

    यदि (L.योग_अष्टकानि > ctx->vtcm_आकार) {
        प्रत्यागम HTP_स्थितिः_VTCM_TOO_SMALL;
    }

    uint8_t * ध्रुव आधारः = ctx->vtcm_आधारः;

    factx.vtcm_q_dma          = VTCM_LAYOUT_PTR(__fp16, आधारः, L.off_q_dma);
    factx.vtcm_q_tiles        = VTCM_LAYOUT_PTR(__fp16, आधारः, L.off_q_tiles);
    factx.vtcm_o_tiles[0]     = VTCM_LAYOUT_PTR(__fp16, आधारः, L.off_o_tiles[0]);
    factx.vtcm_o_tiles[1]     = VTCM_LAYOUT_PTR(__fp16, आधारः, L.off_o_tiles[1]);
    factx.vtcm_k_fp16[0]      = VTCM_LAYOUT_PTR(__fp16, आधारः, L.off_k_fp16[0]);
    factx.vtcm_k_fp16[1]      = VTCM_LAYOUT_PTR(__fp16, आधारः, L.off_k_fp16[1]);
    factx.vtcm_v_fp16[0]      = VTCM_LAYOUT_PTR(__fp16, आधारः, L.off_v_fp16[0]);
    factx.vtcm_v_fp16[1]      = VTCM_LAYOUT_PTR(__fp16, आधारः, L.off_v_fp16[1]);
    factx.vtcm_k_tiles[0]     = VTCM_LAYOUT_PTR(__fp16, आधारः, L.off_k_tiles[0]);
    factx.vtcm_k_tiles[1]     = VTCM_LAYOUT_PTR_OPTIONAL(__fp16, आधारः, L.off_k_tiles[1], pipeline);
    factx.vtcm_v_tiles[0]     = VTCM_LAYOUT_PTR(__fp16, आधारः, L.off_v_tiles[0]);
    factx.vtcm_v_tiles[1]     = VTCM_LAYOUT_PTR_OPTIONAL(__fp16, आधारः, L.off_v_tiles[1], pipeline);
    factx.vtcm_s_tiles[0]     = VTCM_LAYOUT_PTR(__fp16, आधारः, L.off_s_tiles[0]);
    factx.vtcm_s_tiles[1]     = VTCM_LAYOUT_PTR_OPTIONAL(__fp16, आधारः, L.off_s_tiles[1], pipeline);
    factx.vtcm_p_tiles[0]     = VTCM_LAYOUT_PTR(__fp16, आधारः, L.off_p_tiles[0]);
    factx.vtcm_p_tiles[1]     = VTCM_LAYOUT_PTR_OPTIONAL(__fp16, आधारः, L.off_p_tiles[1], pipeline);
    factx.vtcm_d_tiles[0]     = VTCM_LAYOUT_PTR(__fp16, आधारः, L.off_d_tiles[0]);
    factx.vtcm_d_tiles[1]     = VTCM_LAYOUT_PTR_OPTIONAL(__fp16, आधारः, L.off_d_tiles[1], pipeline);
    factx.vtcm_d_inv_l        = VTCM_LAYOUT_PTR(__fp16, आधारः, L.off_d_inv_l);
    factx.vtcm_m_vec          = VTCM_LAYOUT_PTR(HVX_सदिशम्, आधारः, L.off_m_vec);
    factx.vtcm_l_vec          = VTCM_LAYOUT_PTR(HVX_सदिशम्, आधारः, L.off_l_vec);
    factx.vtcm_s_rowmax       = VTCM_LAYOUT_PTR(HVX_सदिशम्, आधारः, L.off_s_rowmax);
    factx.vtcm_p_rowsum       = VTCM_LAYOUT_PTR(HVX_सदिशम्, आधारः, L.off_p_rowsum);
    factx.vtcm_row_bufs       = VTCM_LAYOUT_PTR(HVX_सदिशम्, आधारः, L.off_row_bufs);
    factx.row_buf_stride      = L.row_buf_stride;
    factx.vtcm_hmx_scales_id  = VTCM_LAYOUT_PTR(uint8_t, आधारः, L.off_hmx_scales_id);
    factx.vtcm_hmx_scales_qk  = VTCM_LAYOUT_PTR(uint8_t, आधारः, L.off_hmx_scales_qk);
    factx.vtcm_आवरणम्_buf       = VTCM_LAYOUT_PTR(__fp16, आधारः, L.off_आवरणम्_buf);
    factx.आवरणम्_buf_row_stride = L.आवरणम्_buf_row_stride;
    factx.q_tile_अष्टकानि        = L.q_tile_अष्टकानि;
    factx.k_tile_अष्टकानि        = L.k_tile_अष्टकानि;
    factx.o_tile_अष्टकानि        = L.o_tile_अष्टकानि;
    factx.col_vec_अष्टकानि       = L.col_vec_अष्टकानि;
    factx.d_tile_अष्टकानि        = L.d_tile_अष्टकानि;
    factx.vtcm_slopes         = VTCM_LAYOUT_PTR(__fp16, आधारः, L.off_slopes);
    factx.vtcm_sinks          = VTCM_LAYOUT_PTR_OPTIONAL(भिन्न, आधारः, L.off_sinks, factx.sinks != रिक्त);

    ध्रुव size_t m_पङ्क्तिः_अष्टकानि = L.m_पङ्क्तिः_अष्टकानि;  // used by the mask DMAs in the KV loop

    dma_cache_fa_init(&factx.m_cache, (uint8_t *) factx.vtcm_आवरणम्_buf, L.m_buf_slot_अष्टकानि, HMX_FA_DMA_CACHE_आकार);

    // Head-dim padding: the K/V DMA staging buffers and the flat-Q buffer are laid out
    // with padded row strides (size_{k,v,q}_row_padded, covering D_pad columns) but the
    // DMA only writes the real D columns per row. Zero the whole staging buffers once up
    // front so tail lanes [D, D_pad) stay zero for all KV blocks. No-op when already aligned.
    यदि (DK_pad != DK || DV_pad != DV) {
        ध्रुव size_t k_buf_अष्टकानि = (size_t) factx.Bc * आकार_k_row_padded;
        ध्रुव size_t v_buf_अष्टकानि = (size_t) factx.Bc * आकार_v_row_padded;
        hvx_splat_u8_a((वर्ण *) factx.vtcm_k_fp16[0], 0, k_buf_अष्टकानि);
        hvx_splat_u8_a((वर्ण *) factx.vtcm_k_fp16[1], 0, k_buf_अष्टकानि);
        hvx_splat_u8_a((वर्ण *) factx.vtcm_v_fp16[0], 0, v_buf_अष्टकानि);
        hvx_splat_u8_a((वर्ण *) factx.vtcm_v_fp16[1], 0, v_buf_अष्टकानि);
        // Flat-Q DMA scratch
        यदि (factx.vtcm_q_dma) {
            ध्रुव size_t q_dma_अष्टकानि = hex_align_ऊर्ध्वम्(factx.g_br * DK * (factx.is_q_fp32 ? परिमाण(भिन्न) : परिमाण(__fp16)), 128);
            hvx_splat_u8_a((वर्ण *) factx.vtcm_q_dma, 0, q_dma_अष्टकानि);
        }
    }

    // ======== Initialize HMX output scales ========
    hmx_init_column_scales(factx.vtcm_hmx_scales_id, Q6_V_vsplat_R(0x3c00)); // 1.0
    {
        सङ्घ { __fp16 f; uint16_t u; } sc = { .f = factx.scale };
        hmx_init_column_scales(factx.vtcm_hmx_scales_qk, Q6_V_vsplat_R(sc.u));
    }

    // ======== DMA setup ========
    dma_queue * ध्रुव dma_q = ctx->dma[0];

    यदि (factx.sinks) {
        dma_queue_प्रक्षिप(dma_q, dma_make_दत्त(factx.vtcm_sinks, factx.sinks->दत्त), L.sinks_अष्टकानि, 0, factx.sinks->आकार, 1);
        dma_queue_निष्कासय(dma_q);
    }

    ध्रुव size_t n_row_tiles_g_br = g_br / HMX_FP16_TILE_N_पङ्क्तिकाः;
    ध्रुव size_t n_tiles_प्रत्येकम्_bc   = Bc / HMX_FP16_TILE_N_स्तम्भसङ्ख्या;

    ध्रुव size_t qo_element_आकार = factx.is_q_fp32 ? परिमाण(भिन्न) : परिमाण(__fp16);

    ध्रुव bool q_transposed                 = q->nb[1] < q->nb[2];
    ध्रुव size_t q_src_stride               = q_transposed ? q->nb[2] : q->nb[1];
    ध्रुव size_t q_row_अष्टकानि_untransposed   = factx.G * factx.DK * qo_element_आकार;
    ध्रुव size_t q_row_अष्टकानि_trans_गुणकः   = factx.DK * qo_element_आकार;

    // ======== Reusable job descriptors for pipeline ========
    hmx_fa_qk_job_t       qk_job;
    hmx_fa_o_update_job_t ou_job;
    hmx_fa_o_norm_job_t   on_job;

    // ======== Main loop ========
    प्रति (uint32_t ib3 = 0; ib3 < neq3; ++ib3) {
        ध्रुव uint32_t im3 = आवरणम् ? fastmodulo(ib3, आवरणम्->ne[3], &factx.src3_div3) : 0;
        प्रति (uint32_t q_start = q_start_min; q_start < q_start_max; q_start += Br) {
            ध्रुव uint32_t n_पङ्क्तिकाः_q    = hex_smin(Br, neq1 - q_start);
            ध्रुव size_t   n_पङ्क्तिकाः_g    = n_पङ्क्तिकाः_q * G;
            ध्रुव size_t   g_br_actual = hex_align_ऊर्ध्वम्(n_पङ्क्तिकाः_g, HMX_FP16_TILE_N_पङ्क्तिकाः);
            ध्रुव size_t   n_row_tiles = g_br_actual / HMX_FP16_TILE_N_पङ्क्तिकाः;

            uint16_t सक्रिय_खण्डानि[FA_MAX_KV_खण्डानि];
            uint32_t n_सक्रिय_खण्डानि = factx.n_kv_खण्डानि;
            यदि (आवरणम् && factx.आवरणम्_प्रसारणम् && factx.n_kv_खण्डानि > 1) {
                ध्रुव size_t scratch_अष्टकानि = factx.k_tile_अष्टकानि * (factx.pipeline ? 2 : 1);
                n_सक्रिय_खण्डानि = fa_अवलोकनम्_सक्रिय_खण्डानि(
                    dma_q, आवरणम्, q_start, n_पङ्क्तिकाः_q, 0, im3, nek1, Bc,
                    factx.n_kv_खण्डानि, factx.vtcm_k_tiles[0], scratch_अष्टकानि, सक्रिय_खण्डानि);
            } अन्यथा {
                #निर्देशः clang loop vectorize(disable)
                प्रति (uint32_t b = 0; b < factx.n_kv_खण्डानि; ++b) {
                    सक्रिय_खण्डानि[b] = (uint16_t) b;
                }
            }
            ध्रुव uint32_t b0       = (n_सक्रिय_खण्डानि > 0) ? सक्रिय_खण्डानि[0] : 0;
            ध्रुव uint32_t b0_start = b0 * Bc;
            ध्रुव uint32_t b0_पङ्क्तिकाः  = hex_smin(Bc, nek1 - b0_start);

            प्रति (uint32_t kv_शीर्ष = kv_शीर्ष_min; kv_शीर्ष < kv_शीर्ष_max; ++kv_शीर्ष) {
                ध्रुव uint32_t ik2 = kv_शीर्ष;
                ध्रुव uint32_t ik3 = fastdiv(ib3, &kparams->प्रसारणम्_rk3);
                ध्रुव uint32_t iv2 = kv_शीर्ष;
                ध्रुव uint32_t iv3 = fastdiv(ib3, &kparams->प्रसारणम्_rv3);

                // 1. Push Q and KV DMAs for the first kv_head of this q_start block.
                // Subsequent kv_head iterations are enqueued early at the end of the previous iteration.
                यदि (kv_शीर्ष == kv_शीर्ष_min) {
                    ध्रुव dma_addr_t q_ptr = q->दत्त + q_start * q->nb[1] +
                                            (kv_शीर्ष * factx.G) * q->nb[2] + ib3 * q->nb[3];
                    ध्रुव size_t q_row_अष्टकानि = q_transposed ? n_पङ्क्तिकाः_q * q_row_अष्टकानि_trans_गुणकः : q_row_अष्टकानि_untransposed;
                    ध्रुव size_t n_पङ्क्तिकाः      = q_transposed ? factx.G : n_पङ्क्तिकाः_q;
                    dma_queue_प्रक्षिप(dma_q, dma_make_दत्त(factx.vtcm_q_dma, q_ptr), q_row_अष्टकानि, hex_smax(q_src_stride, q_row_अष्टकानि), q_row_अष्टकानि, n_पङ्क्तिकाः);

                    यदि (n_सक्रिय_खण्डानि > 0) {
                        ध्रुव dma_addr_t k_src = k->दत्त + b0_start * k->nb[1] + ik2 * k->nb[2] + ik3 * k->nb[3];
                        dma_queue_प्रक्षिप(dma_q, dma_make_दत्त(factx.vtcm_k_fp16[0], k_src), आकार_k_row_padded, k->nb[1], आकार_k_row, b0_पङ्क्तिकाः);

                        ध्रुव dma_addr_t v_src = v->दत्त + b0_start * v->nb[1] + iv2 * v->nb[2] + iv3 * v->nb[3];
                        dma_queue_प्रक्षिप(dma_q, dma_make_दत्त(factx.vtcm_v_fp16[0], v_src), आकार_v_row_padded, v->nb[1], आकार_v_row, b0_पङ्क्तिकाः);

                        यदि (factx.pipeline && आवरणम्) {
                            यदि (__builtin_expect(factx.आवरणम्_प्रसारणम्, true)) {
                                ध्रुव dma_addr_t ms_src = आवरणम्->दत्त + q_start * आवरणम्->nb[1] + im3 * आवरणम्->nb[3] + b0_start * परिमाण(__fp16);
                                dma_cache_fa_प्रक्षिप(dma_q, &factx.m_cache, ms_src, m_पङ्क्तिः_अष्टकानि, आवरणम्->nb[1], b0_पङ्क्तिकाः * परिमाण(__fp16), n_पङ्क्तिकाः_q);
                            } अन्यथा {
                                fa_प्रक्षिप_आवरणम्_dma_gqa(dma_q, आवरणम्, q_start, im3, b0_start, kv_शीर्ष, G, m_पङ्क्तिः_अष्टकानि, b0_पङ्क्तिकाः, n_पङ्क्तिकाः_q, &factx);
                            }
                        }
                    }
                }

                // 2. Pop Q DMA (blocks until Q is loaded)
                dma_queue_निष्कासय(dma_q);

                // ---- Load Q block & Initialize per-block state ----
                fa_phase_q_आरोपय(&factx, q, q_start, kv_शीर्ष, ib3, n_पङ्क्तिकाः_g);

                __fp16 * o_tile_पूर्व = factx.vtcm_o_tiles[0];
                __fp16 * o_tile_curr = factx.vtcm_o_tiles[1];

                // ---- KV block loop with DMA double-buffering ----
                size_t buf_idx = 0;

                htp_trace_event_start(tr_hvx, HTP_TRACE_EVT_HVX_A_PREP, (uint16_t) q_start);
                fa_compute_slopes(&factx, kv_शीर्ष, n_पङ्क्तिकाः_g);
                htp_trace_event_stop(tr_hvx, HTP_TRACE_EVT_HVX_A_PREP, (uint16_t) q_start);

                ध्रुव size_t k_src_stride = आकार_k_row_padded / परिमाण(__fp16);
                ध्रुव size_t v_src_stride = आकार_v_row_padded / परिमाण(__fp16);

                hmx_queue_t hmx_q = ctx->hmx_queue;

                यदि (factx.pipeline) {
                    // Double-buffered job structs because HMX queue runs asynchronously
                    hmx_fa_qk_job_t qk_job[2];
                    hmx_fa_o_update_job_t ou_job[2];

                    // Prefetch second active block early if there are multiple active blocks
                    यदि (n_सक्रिय_खण्डानि > 1) {
                        fa_prefetch_खण्डम्(dma_q, k, v, आवरणम्, सक्रिय_खण्डानि[1], Bc, आकार_k_row_padded, आकार_k_row, आकार_v_row_padded, आकार_v_row,
                                          ik2, ik3, iv2, iv3, q_start, im3, kv_शीर्ष, G, m_पङ्क्तिः_अष्टकानि, n_पङ्क्तिकाः_q, nek1, 1, &factx);
                    }

                    // Prep and start QK-dot(0)
                    शून्य * curr_k0 = (शून्य *) dma_queue_निष्कासय(dma_q).dst;
                    fa_phase_k_interleave(&factx, b0_पङ्क्तिकाः, k_src_stride, curr_k0, b0_start, 0);

                    qk_job[0].q_tiles        = factx.vtcm_q_tiles;
                    qk_job[0].k_tiles        = factx.vtcm_k_tiles[0];
                    qk_job[0].s_tiles        = factx.vtcm_s_tiles[0];
                    qk_job[0].n_row_tiles    = n_row_tiles;
                    qk_job[0].n_col_tiles    = hmx_ceil_div(b0_पङ्क्तिकाः, HMX_FP16_TILE_N_स्तम्भसङ्ख्या);
                    qk_job[0].n_dot_tiles    = DK_pad / 32;
                    qk_job[0].n_tiles_प्रत्येकम्_bc = n_tiles_प्रत्येकम्_bc;
                    qk_job[0].hmx_scales     = factx.vtcm_hmx_scales_qk;
                    hmx_queue_प्रक्षिप(hmx_q, hmx_queue_make_desc(hmx_fa_qk_dot_कार्यकर्ता, &qk_job[0]));

                    प्रति (uint32_t चरणः = 0; चरणः < n_सक्रिय_खण्डानि; ++चरणः) {
                        ध्रुव uint32_t kv_blk      = सक्रिय_खण्डानि[चरणः];
                        ध्रुव uint32_t kv_start    = kv_blk * Bc;
                        ध्रुव uint32_t kv_पङ्क्तिकाः     = hex_smin(Bc, nek1 - kv_start);
                        ध्रुव size_t   n_col_tiles = hmx_ceil_div(kv_पङ्क्तिकाः, HMX_FP16_TILE_N_स्तम्भसङ्ख्या);

                        // ---- 1. Pop and run V-prep for current block ----
                        शून्य * curr_v = (शून्य *) dma_queue_निष्कासय(dma_q).dst;
                        fa_phase_v_interleave(&factx, kv_पङ्क्तिकाः, v_src_stride, curr_v, factx.vtcm_v_tiles[buf_idx], n_tiles_प्रत्येकम्_bc, kv_start);

                        // ---- 2. Pop and run mask-prep for current block ----
                        __fp16 * current_आवरणम्_vtcm = रिक्त;
                        यदि (आवरणम्) {
                            यदि (__builtin_expect(factx.आवरणम्_प्रसारणम्, true)) {
                                current_आवरणम्_vtcm = (__fp16 *) dma_queue_निष्कासय(dma_q).dst;
                            } अन्यथा {
                                fa_निष्कासय_आवरणम्_dma_gqa(dma_q, G);
                                current_आवरणम्_vtcm = factx.vtcm_आवरणम्_buf;
                            }
                        }

                        // ---- 3. Start HMX O update for step - 1 (reads P[1 - buf_idx], V[1 - buf_idx], D) ----
                        // O update relies on the previous block's P and V tiles.
                        // O update MUST be pushed before the next block's QK-dot: hmx_queue_pop() retires the
                        // oldest descriptor, so push order alone decides which pop waits for which job.
                        // If OU went in after QK(i+1), the pop below would retire QK(i+1) and leave
                        // OU(i-1) in flight into the next iteration, where V-prep overwrites V[prev_buf].
                        यदि (चरणः > 0) {
                            ध्रुव uint32_t पूर्व_blk      = सक्रिय_खण्डानि[चरणः - 1];
                            ध्रुव size_t पूर्व_buf        = 1 - buf_idx;
                            ou_job[पूर्व_buf].o_curr      = o_tile_curr;
                            ou_job[पूर्व_buf].o_पूर्व      = o_tile_पूर्व;
                            ou_job[पूर्व_buf].p_tiles     = factx.vtcm_p_tiles[पूर्व_buf];
                            ou_job[पूर्व_buf].v_tiles     = factx.vtcm_v_tiles[पूर्व_buf];
                            ou_job[पूर्व_buf].d_tiles     = factx.vtcm_d_tiles[पूर्व_buf];
                            ou_job[पूर्व_buf].hmx_scales  = factx.vtcm_hmx_scales_id;
                            ou_job[पूर्व_buf].n_row_tiles = n_row_tiles;
                            ou_job[पूर्व_buf].n_col_tiles =
                                hmx_ceil_div(hex_smin(Bc, nek1 - पूर्व_blk * Bc), HMX_FP16_TILE_N_स्तम्भसङ्ख्या);
                            ou_job[पूर्व_buf].n_row_tiles_g_br = n_row_tiles_g_br;
                            ou_job[पूर्व_buf].n_tiles_प्रत्येकम्_bc   = n_tiles_प्रत्येकम्_bc;
                            ou_job[पूर्व_buf].DV               = DV_pad;
                            hmx_queue_प्रक्षिप(hmx_q, hmx_queue_make_desc(hmx_fa_o_update_कार्यकर्ता, &ou_job[पूर्व_buf]));
                        }

                        // ---- 4. Pop and run K-prep for next active block & push next QK-dot ----
                        यदि (चरणः + 1 < n_सक्रिय_खण्डानि) {
                            ध्रुव uint32_t अग्र_blk   = सक्रिय_खण्डानि[चरणः + 1];
                            ध्रुव uint32_t अग्र_start = अग्र_blk * Bc;
                            ध्रुव uint32_t अग्र_पङ्क्तिकाः  = hex_smin(Bc, nek1 - अग्र_start);
                            ध्रुव size_t   अग्र_buf   = 1 - buf_idx;

                            शून्य * अग्र_k = (शून्य *) dma_queue_निष्कासय(dma_q).dst;
                            fa_phase_k_interleave(&factx, अग्र_पङ्क्तिकाः, k_src_stride, अग्र_k, अग्र_start, अग्र_buf);

                            qk_job[अग्र_buf].q_tiles        = factx.vtcm_q_tiles;
                            qk_job[अग्र_buf].k_tiles        = factx.vtcm_k_tiles[अग्र_buf];
                            qk_job[अग्र_buf].s_tiles        = factx.vtcm_s_tiles[अग्र_buf];
                            qk_job[अग्र_buf].n_row_tiles    = n_row_tiles;
                            qk_job[अग्र_buf].n_col_tiles    = hmx_ceil_div(अग्र_पङ्क्तिकाः, HMX_FP16_TILE_N_स्तम्भसङ्ख्या);
                            qk_job[अग्र_buf].n_dot_tiles    = DK_pad / 32;
                            qk_job[अग्र_buf].n_tiles_प्रत्येकम्_bc = n_tiles_प्रत्येकम्_bc;
                            qk_job[अग्र_buf].hmx_scales     = factx.vtcm_hmx_scales_qk;
                            hmx_queue_प्रक्षिप(hmx_q, hmx_queue_make_desc(hmx_fa_qk_dot_कार्यकर्ता, &qk_job[अग्र_buf]));
                        }

                        // ---- 5. Wait for current block's QK-dot to finish ----
                        hmx_queue_निष्कासय(hmx_q);

                        // ---- 6. Phase 2: softmax + build_D ----
                        fa_softmax_तर्काः_t sargs;
                        स्मृतिं_पूरय(&sargs, 0, परिमाण(sargs));
                        sargs.factx                = &factx;
                        sargs.buf_idx              = buf_idx;
                        sargs.kv_पङ्क्तिकाः              = kv_पङ्क्तिकाः;
                        sargs.n_पङ्क्तिकाः_g             = n_पङ्क्तिकाः_g;
                        sargs.n_col_tiles          = n_col_tiles;
                        sargs.n_tiles_प्रत्येकम्_bc       = n_tiles_प्रत्येकम्_bc;
                        sargs.n_row_tiles          = n_row_tiles;
                        sargs.n_row_tiles_g_br     = n_row_tiles_g_br;
                        sargs.Bc                   = Bc;
                        sargs.G                    = G;
                        sargs.kv_शीर्ष              = kv_शीर्ष;
                        sargs.kv_start             = kv_start;
                        sargs.q_start              = q_start;
                        sargs.ib3                  = ib3;
                        sargs.has_alibi            = (factx.max_bias != 0.0f);
                        sargs.आवरणम्                 = आवरणम्;
                        sargs.आवरणम्_vtcm            = current_आवरणम्_vtcm;
                        sargs.आवरणम्_vtcm_row_stride = factx.आवरणम्_buf_row_stride;
                        sargs.slopes               = factx.vtcm_slopes;

                        // Run Softmax on HVX (blocking call)
                        fa_phase_softmax_and_build_d(&factx, &sargs, n_row_tiles, n_row_tiles_g_br);

                        // Wait for HMX O update for step - 1 to finish
                        यदि (चरणः > 0) {
                            hmx_queue_निष्कासय(hmx_q);
                            hex_swap_ptr((शून्य **) &o_tile_curr, (शून्य **) &o_tile_पूर्व);
                        }

                        // Prefetch block step + 2
                        यदि (चरणः + 2 < n_सक्रिय_खण्डानि) {
                            fa_prefetch_खण्डम्(dma_q, k, v, आवरणम्, सक्रिय_खण्डानि[चरणः + 2], Bc, आकार_k_row_padded, आकार_k_row, आकार_v_row_padded, आकार_v_row,
                                              ik2, ik3, iv2, iv3, q_start, im3, kv_शीर्ष, G, m_पङ्क्तिः_अष्टकानि, n_पङ्क्तिकाः_q, nek1, buf_idx, &factx);
                        }

                        buf_idx = 1 - buf_idx;
                    }

                    // Epilogue
                    यदि (n_सक्रिय_खण्डानि > 0) {
                        ध्रुव uint32_t चरमः_blk = सक्रिय_खण्डानि[n_सक्रिय_खण्डानि - 1];
                        ध्रुव size_t चरमः_स्तम्भसङ्ख्या  = hmx_ceil_div(hex_smin(Bc, nek1 - चरमः_blk * Bc), HMX_FP16_TILE_N_स्तम्भसङ्ख्या);
                        ou_job[0].o_curr           = o_tile_curr;
                        ou_job[0].o_पूर्व           = o_tile_पूर्व;
                        ou_job[0].p_tiles          = factx.vtcm_p_tiles[1 - buf_idx];
                        ou_job[0].v_tiles          = factx.vtcm_v_tiles[1 - buf_idx];
                        ou_job[0].d_tiles          = factx.vtcm_d_tiles[1 - buf_idx];
                        ou_job[0].hmx_scales       = factx.vtcm_hmx_scales_id;
                        ou_job[0].n_row_tiles      = n_row_tiles;
                        ou_job[0].n_col_tiles      = चरमः_स्तम्भसङ्ख्या;
                        ou_job[0].n_row_tiles_g_br = n_row_tiles_g_br;
                        ou_job[0].n_tiles_प्रत्येकम्_bc   = n_tiles_प्रत्येकम्_bc;
                        ou_job[0].DV               = DV_pad;
                        hmx_queue_प्रक्षिप(hmx_q, hmx_queue_make_desc(hmx_fa_o_update_कार्यकर्ता, &ou_job[0]));

                        // Overlapped: run HVX build diag inv L while HMX is busy executing the update
                        htp_trace_event_start(tr_hvx, HTP_TRACE_EVT_HVX_O_PROC, (uint16_t) q_start);
                        fa_build_d_diag_inv_l(&factx, n_row_tiles, n_row_tiles_g_br, kv_शीर्ष, n_पङ्क्तिकाः_g);
                        htp_trace_event_stop(tr_hvx, HTP_TRACE_EVT_HVX_O_PROC, (uint16_t) q_start);
                        hmx_queue_निष्कासय(hmx_q);

                        hex_swap_ptr((शून्य **) &o_tile_curr, (शून्य **) &o_tile_पूर्व);
                    }

                } अन्यथा {
                    // Fallback path
                    प्रति (uint32_t चरणः = 0; चरणः < n_सक्रिय_खण्डानि; ++चरणः) {
                        ध्रुव uint32_t kv_blk      = सक्रिय_खण्डानि[चरणः];
                        ध्रुव uint32_t kv_start    = kv_blk * Bc;
                        ध्रुव uint32_t kv_पङ्क्तिकाः     = hex_smin(Bc, nek1 - kv_start);
                        ध्रुव size_t   n_col_tiles = hmx_ceil_div(kv_पङ्क्तिकाः, HMX_FP16_TILE_N_स्तम्भसङ्ख्या);

                        यदि (आवरणम्) {
                            यदि (__builtin_expect(factx.आवरणम्_प्रसारणम्, true)) {
                                ध्रुव dma_addr_t ms_src = आवरणम्->दत्त + q_start * आवरणम्->nb[1] + im3 * आवरणम्->nb[3] + kv_start * परिमाण(__fp16);
                                dma_cache_fa_प्रक्षिप(dma_q, &factx.m_cache, ms_src, m_पङ्क्तिः_अष्टकानि, आवरणम्->nb[1], kv_पङ्क्तिकाः * परिमाण(__fp16), n_पङ्क्तिकाः_q);
                            } अन्यथा {
                                fa_प्रक्षिप_आवरणम्_dma_gqa(dma_q, आवरणम्, q_start, im3, kv_start, kv_शीर्ष, G, m_पङ्क्तिः_अष्टकानि, kv_पङ्क्तिकाः, n_पङ्क्तिकाः_q, &factx);
                            }
                        }

                        यदि (चरणः + 1 < n_सक्रिय_खण्डानि) {
                            ध्रुव uint32_t  prefetch_blk   = सक्रिय_खण्डानि[चरणः + 1];
                            ध्रुव uint32_t  prefetch_start = prefetch_blk * Bc;
                            ध्रुव uint32_t  prefetch_पङ्क्तिकाः  = hex_smin(Bc, nek1 - prefetch_start);
                            ध्रुव size_t    prefetch_buf   = 1 - buf_idx;
                            ध्रुव dma_addr_t k_prefetch_src = k->दत्त + prefetch_start * k->nb[1] + ik2 * k->nb[2] + ik3 * k->nb[3];
                            dma_queue_प्रक्षिप(dma_q, dma_make_दत्त(factx.vtcm_k_fp16[prefetch_buf], k_prefetch_src), आकार_k_row_padded, k->nb[1], आकार_k_row, prefetch_पङ्क्तिकाः);
                            ध्रुव dma_addr_t v_prefetch_src = v->दत्त + prefetch_start * v->nb[1] + iv2 * v->nb[2] + iv3 * v->nb[3];
                            dma_queue_प्रक्षिप(dma_q, dma_make_दत्त(factx.vtcm_v_fp16[prefetch_buf], v_prefetch_src), आकार_v_row_padded, v->nb[1], आकार_v_row, prefetch_पङ्क्तिकाः);
                        }

                        // Wait for current K DMA and interleave
                        शून्य * curr_k = (शून्य *) dma_queue_निष्कासय(dma_q).dst;
                        fa_phase_k_interleave(&factx, kv_पङ्क्तिकाः, k_src_stride, curr_k, kv_start, 0);

                        {
                            qk_job.q_tiles        = factx.vtcm_q_tiles;
                            qk_job.k_tiles        = factx.vtcm_k_tiles[0];
                            qk_job.s_tiles        = factx.vtcm_s_tiles[0];
                            qk_job.n_row_tiles    = n_row_tiles;
                            qk_job.n_col_tiles    = n_col_tiles;
                            qk_job.n_dot_tiles    = (size_t) (DK_pad / 32);
                            qk_job.n_tiles_प्रत्येकम्_bc = n_tiles_प्रत्येकम्_bc;
                            qk_job.hmx_scales     = factx.vtcm_hmx_scales_qk;

                            hmx_queue_प्रक्षिप(ctx->hmx_queue, hmx_queue_make_desc(hmx_fa_qk_dot_कार्यकर्ता, &qk_job));
                            hmx_queue_निष्कासय(ctx->hmx_queue);
                        }

                        // Wait for current V DMA and interleave
                        शून्य * curr_v = (शून्य *) dma_queue_निष्कासय(dma_q).dst;
                        fa_phase_v_interleave(&factx, kv_पङ्क्तिकाः, v_src_stride, curr_v, factx.vtcm_v_tiles[0], n_tiles_प्रत्येकम्_bc, kv_start);

                        // ---- Phase 3: softmax + build_D ----
                        __fp16 * current_आवरणम्_vtcm = रिक्त;
                        यदि (आवरणम्) {
                            यदि (__builtin_expect(factx.आवरणम्_प्रसारणम्, true)) {
                                current_आवरणम्_vtcm = (__fp16 *) dma_queue_निष्कासय(dma_q).dst;
                            } अन्यथा {
                                fa_निष्कासय_आवरणम्_dma_gqa(dma_q, G);
                                current_आवरणम्_vtcm = factx.vtcm_आवरणम्_buf;
                            }
                        }

                        fa_softmax_तर्काः_t sargs;
                        स्मृतिं_पूरय(&sargs, 0, परिमाण(sargs));
                        sargs.factx                = &factx;
                        sargs.kv_पङ्क्तिकाः              = kv_पङ्क्तिकाः;
                        sargs.n_पङ्क्तिकाः_g             = n_पङ्क्तिकाः_g;
                        sargs.n_col_tiles          = n_col_tiles;
                        sargs.n_tiles_प्रत्येकम्_bc       = n_tiles_प्रत्येकम्_bc;
                        sargs.n_row_tiles          = n_row_tiles;
                        sargs.n_row_tiles_g_br     = n_row_tiles_g_br;
                        sargs.Bc                   = Bc;
                        sargs.G                    = G;
                        sargs.kv_शीर्ष              = kv_शीर्ष;
                        sargs.kv_start             = kv_start;
                        sargs.q_start              = q_start;
                        sargs.ib3                  = ib3;
                        sargs.has_alibi            = (factx.max_bias != 0.0f);
                        sargs.आवरणम्                 = आवरणम्;
                        sargs.आवरणम्_vtcm            = current_आवरणम्_vtcm;
                        sargs.आवरणम्_vtcm_row_stride = factx.आवरणम्_buf_row_stride;
                        sargs.slopes               = factx.vtcm_slopes;
                        fa_phase_softmax_and_build_d(&factx, &sargs, n_row_tiles, n_row_tiles_g_br);

                        {
                            ou_job.o_curr           = o_tile_curr;
                            ou_job.o_पूर्व           = o_tile_पूर्व;
                            ou_job.p_tiles          = factx.vtcm_p_tiles[0];
                            ou_job.v_tiles          = factx.vtcm_v_tiles[0];
                            ou_job.d_tiles          = factx.vtcm_d_tiles[0];
                            ou_job.hmx_scales       = factx.vtcm_hmx_scales_id;
                            ou_job.n_row_tiles      = n_row_tiles;
                            ou_job.n_col_tiles      = n_col_tiles;
                            ou_job.n_row_tiles_g_br = n_row_tiles_g_br;
                            ou_job.n_tiles_प्रत्येकम्_bc   = n_tiles_प्रत्येकम्_bc;
                            ou_job.DV               = DV_pad;

                            hmx_queue_प्रक्षिप(ctx->hmx_queue, hmx_queue_make_desc(hmx_fa_o_update_कार्यकर्ता, &ou_job));
                            यदि (चरणः + 1 == n_सक्रिय_खण्डानि) {
                                // Overlapped: run HVX build diag inv L while HMX is busy executing the update
                                htp_trace_event_start(tr_hvx, HTP_TRACE_EVT_HVX_O_PROC, (uint16_t) q_start);
                                fa_build_d_diag_inv_l(&factx, n_row_tiles, n_row_tiles_g_br, kv_शीर्ष, n_पङ्क्तिकाः_g);
                                htp_trace_event_stop(tr_hvx, HTP_TRACE_EVT_HVX_O_PROC, (uint16_t) q_start);
                            }
                            hmx_queue_निष्कासय(ctx->hmx_queue);

                            hex_swap_ptr((शून्य **) &o_tile_curr, (शून्य **) &o_tile_पूर्व);
                        }

                        buf_idx = 1 - buf_idx;
                    }
                }

                // Enqueue DMAs for the next kv_head early so they overlap with O-PROC
                ध्रुव uint32_t अग्र_kv_शीर्ष = kv_शीर्ष + 1;
                ध्रुव bool has_अग्र = (अग्र_kv_शीर्ष < kv_शीर्ष_max);

                यदि (has_अग्र) {
                    ध्रुव dma_addr_t अग्र_q_ptr = q->दत्त + q_start * q->nb[1] + (अग्र_kv_शीर्ष * factx.G) * q->nb[2] + ib3 * q->nb[3];
                    ध्रुव size_t अग्र_q_row_अष्टकानि = q_transposed ? n_पङ्क्तिकाः_q * q_row_अष्टकानि_trans_गुणकः : q_row_अष्टकानि_untransposed;
                    ध्रुव size_t अग्र_n_पङ्क्तिकाः      = q_transposed ? factx.G : n_पङ्क्तिकाः_q;
                    dma_queue_प्रक्षिप(dma_q, dma_make_दत्त(factx.vtcm_q_dma, अग्र_q_ptr), अग्र_q_row_अष्टकानि, hex_smax(q_src_stride, अग्र_q_row_अष्टकानि), अग्र_q_row_अष्टकानि, अग्र_n_पङ्क्तिकाः);

                    यदि (n_सक्रिय_खण्डानि > 0) {
                        ध्रुव uint32_t अग्र_ik2 = अग्र_kv_शीर्ष;
                        ध्रुव uint32_t अग्र_iv2 = अग्र_kv_शीर्ष;

                        ध्रुव dma_addr_t अग्र_k_src = k->दत्त + b0_start * k->nb[1] + अग्र_ik2 * k->nb[2] + ik3 * k->nb[3];
                        dma_queue_प्रक्षिप(dma_q, dma_make_दत्त(factx.vtcm_k_fp16[0], अग्र_k_src), आकार_k_row_padded, k->nb[1], आकार_k_row, b0_पङ्क्तिकाः);

                        ध्रुव dma_addr_t अग्र_v_src = v->दत्त + b0_start * v->nb[1] + अग्र_iv2 * v->nb[2] + iv3 * v->nb[3];
                        dma_queue_प्रक्षिप(dma_q, dma_make_दत्त(factx.vtcm_v_fp16[0], अग्र_v_src), आकार_v_row_padded, v->nb[1], आकार_v_row, b0_पङ्क्तिकाः);

                        यदि (factx.pipeline && आवरणम्) {
                            यदि (__builtin_expect(factx.आवरणम्_प्रसारणम्, true)) {
                                ध्रुव dma_addr_t ms_src = आवरणम्->दत्त + q_start * आवरणम्->nb[1] + im3 * आवरणम्->nb[3] + b0_start * परिमाण(__fp16);
                                dma_cache_fa_प्रक्षिप(dma_q, &factx.m_cache, ms_src, m_पङ्क्तिः_अष्टकानि, आवरणम्->nb[1], b0_पङ्क्तिकाः * परिमाण(__fp16), n_पङ्क्तिकाः_q);
                            } अन्यथा {
                                fa_प्रक्षिप_आवरणम्_dma_gqa(dma_q, आवरणम्, q_start, im3, b0_start, अग्र_kv_शीर्ष, G, m_पङ्क्तिः_अष्टकानि, b0_पङ्क्तिकाः, n_पङ्क्तिकाः_q, &factx);
                            }
                        }
                    }
                }

                // ---- Final normalization ----
                {
                    on_job.o_curr           = o_tile_curr;
                    on_job.o_पूर्व           = o_tile_पूर्व;
                    on_job.d_tiles          = factx.vtcm_d_inv_l;
                    on_job.hmx_scales       = factx.vtcm_hmx_scales_id;
                    on_job.n_row_tiles      = n_row_tiles;
                    on_job.n_row_tiles_g_br = n_row_tiles_g_br;
                    on_job.DV               = DV_pad;
                    hmx_queue_प्रक्षिप(ctx->hmx_queue, hmx_queue_make_desc(hmx_fa_o_norm_कार्यकर्ता, &on_job));
                    hmx_queue_निष्कासय(ctx->hmx_queue);
                }

                // ---- Store O block ----
                fa_phase_o_store(&factx, dst, o_tile_curr, q_start, kv_शीर्ष, ib3, n_पङ्क्तिकाः_g);
            }
        }
    }

    प्रत्यागम HTP_स्थितिः_OK;
}

पूर्णाङ्क op_flash_attn_ext(संरचना htp_ops_context * octx) {
    ध्रुव संरचना htp_tensor * q    = octx->src[0];
    ध्रुव संरचना htp_tensor * k    = octx->src[1];
    ध्रुव संरचना htp_tensor * v    = octx->src[2];
    ध्रुव संरचना htp_tensor * आवरणम् = octx->src[3];
    ध्रुव संरचना htp_tensor * dst  = octx->dst;

    // Check support
    यदि ((q->type != HTP_TYPE_F16 && q->type != HTP_TYPE_F32) ||
        (k->type != HTP_TYPE_F16 && k->type != HTP_TYPE_Q8_0) ||
        (v->type != HTP_TYPE_F16 && v->type != HTP_TYPE_Q8_0)) {
        प्रत्यागम HTP_स्थितिः_NO_SUPPORT;
    }

    यदि (htp_tensor_is_extended(dst)) {
        प्रत्यागम HTP_स्थितिः_NO_SUPPORT;
    }

    ध्रुव संरचना htp_fa_kernel_params * kparams = (ध्रुव संरचना htp_fa_kernel_params *) octx->kernel_params;

    यदि (kparams->kernel_type == HTP_FA_KERNEL_UNSUPPORTED || kparams->n_kv_खण्डानि > FA_MAX_KV_खण्डानि) {
        प्रत्यागम HTP_स्थितिः_NO_SUPPORT;
    }

    यदि (!htp_ops_context_समुच्चयः_n_तन्तवः(octx, kparams->n_तन्तवः)) {
        प्रत्यागम HTP_स्थितिः_INVAL_PARAMS;
    }

    यदि (kparams->kernel_type == HTP_FA_KERNEL_HMX) {
        प्रत्यागम hmx_flash_attn_ext(octx);
    }

    संरचना htp_fa_context factx;
    factx.octx = octx;
    factx.k = k;
    factx.v = v;

    factx.src0_div21 = kparams->u.hvx.src0_div21;
    factx.src0_div1  = kparams->u.hvx.src0_div1;

    factx.प्रसारणम्_rk2 = kparams->प्रसारणम्_rk2;
    factx.प्रसारणम्_rk3 = kparams->प्रसारणम्_rk3;
    factx.प्रसारणम्_rv2 = kparams->प्रसारणम्_rv2;
    factx.प्रसारणम्_rv3 = kparams->प्रसारणम्_rv3;

    यदि (आवरणम्) {
        factx.src3_div2 = kparams->src3_div2;
        factx.src3_div3 = kparams->src3_div3;
    }

    factx.is_q_fp32 = (q->type == HTP_TYPE_F32);
    factx.आकार_q_row_padded = kparams->u.hvx.आकार_q_row_padded;
    factx.आकार_k_row_padded = kparams->u.hvx.आकार_k_row_padded;
    factx.आकार_v_row_padded = kparams->u.hvx.आकार_v_row_padded;

    factx.n_खण्डानि = kparams->n_kv_खण्डानि;

    factx.scale = kparams->scale;
    factx.max_bias = kparams->max_bias;
    factx.has_softcap = (kparams->logit_softcap != 0.0f);
    factx.logit_softcap = 0;
    यदि (factx.has_softcap) {
        factx.logit_softcap = (__fp16) kparams->logit_softcap;
    }

    factx.n_शीर्ष_log2 = kparams->n_शीर्ष_log2;
    factx.m0          = kparams->m0;
    factx.m1          = kparams->m1;

    ध्रुव uint32_t n_शीर्ष = q->ne[2];
    यदि (n_शीर्ष > 512) {
        प्रत्यागम HTP_स्थितिः_NO_SUPPORT;
    }
    प्रति (uint32_t h = 0; h < n_शीर्ष; ++h) {
        factx.slopes[h] = (__fp16) ((kparams->max_bias > 0.0f) ? alibi_नतिः(h, factx.n_शीर्ष_log2, factx.m0, factx.m1) : 1.0f);
    }

    // total rows in q
    ध्रुव uint32_t neq1 = q->ne[1];
    ध्रुव uint32_t neq2 = q->ne[2];
    ध्रुव uint32_t neq3 = q->ne[3];
    ध्रुव uint32_t योग_qrows = neq1 * neq2 * neq3;

    uint32_t qrow_start = 0;
    uint32_t qrows      = योग_qrows;

    यदि (octx->ctx->mdev.गणनम् > 1) {
        ध्रुव uint32_t mdev_गणनम् = octx->ctx->mdev.गणनम्;
        ध्रुव uint32_t mdev_idx   = octx->ctx->mdev.idx;
        ध्रुव uint32_t n_kv_heads = k->ne[2];
        ध्रुव uint32_t dst_e_आकार = (dst->type == HTP_TYPE_F32) ? परिमाण(भिन्न) : परिमाण(__fp16);
        ध्रुव bool can_विभज      = htp_tensor_can_row_partition(dst, dst_e_आकार);

        // head range is contiguous in flat row space only when neq3 == 1
        यदि (kparams->शीर्ष_विभज && can_विभज && neq3 == 1 && n_kv_heads >= mdev_गणनम् && n_kv_heads % mdev_गणनम् == 0) {
            ध्रुव uint32_t G              = kparams->G;
            ध्रुव uint32_t kv_प्रत्येकम्_core    = n_kv_heads / mdev_गणनम्;
            ध्रुव uint32_t heads_प्रत्येकम्_core = kv_प्रत्येकम्_core * G;
            ध्रुव uint32_t शीर्ष_start     = mdev_idx * heads_प्रत्येकम्_core;
            qrow_start = शीर्ष_start * neq1;
            qrows      = heads_प्रत्येकम्_core * neq1;
        } अन्यथा {
            ध्रुव संरचना htp_tensor_mdev_range range = htp_tensor_mdev_partition(योग_qrows, can_विभज ? 1 : 0, mdev_idx, mdev_गणनम्, &octx->ctx->mdev.गणनम्_div);
            qrow_start = range.start;
            qrows      = range.गणनम्;
        }
    }

    यदि (qrows == 0) {
        प्रत्यागम HTP_स्थितिः_OK;
    }

    ध्रुव uint32_t n_तन्तवः = octx->n_तन्तवः;

    factx.qrows            = qrows;
    factx.qrow_start       = qrow_start;
    factx.qrows_प्रत्येकम्_thread = fastdiv(qrows + n_तन्तवः - 1, &octx->n_तन्तवः_div);

    ध्रुव bool has_आवरणम्  = (आवरणम् != रिक्त);
    ध्रुव bool has_sinks = (octx->src[4] != रिक्त);
    संरचना hvx_fa_vtcm_layout L;
    hvx_fa_vtcm_layout_build(&L, k->ne[0], v->ne[0], factx.is_q_fp32, has_आवरणम्, has_sinks, n_शीर्ष, n_तन्तवः);

    यदि (L.योग_अष्टकानि > octx->ctx->vtcm_आकार) {
        प्रत्यागम HTP_स्थितिः_VTCM_TOO_SMALL;
    }

    factx.आकार_q_खण्डम् = L.आकार_q_खण्डम्;
    factx.आकार_k_खण्डम् = L.आकार_k_खण्डम्;
    factx.आकार_v_खण्डम् = L.आकार_v_खण्डम्;
    factx.आकार_m_खण्डम् = L.आकार_m_खण्डम्;
    factx.आकार_vkq_acc = L.आकार_vkq_acc;

    uint8_t * ध्रुव आधारः = octx->ctx->vtcm_आधारः;

    factx.spad_q               = VTCM_LAYOUT_PTR(uint8_t, आधारः, L.off_q);
    factx.spad_k               = VTCM_LAYOUT_PTR(uint8_t, आधारः, L.off_k);
    factx.spad_v               = VTCM_LAYOUT_PTR(uint8_t, आधारः, L.off_v);
    factx.spad_m               = VTCM_LAYOUT_PTR_OPTIONAL(uint8_t, आधारः, L.off_m, has_आवरणम्);
    factx.spad_a               = VTCM_LAYOUT_PTR(uint8_t, आधारः, L.off_a);
    factx.spad_sinks           = VTCM_LAYOUT_PTR_OPTIONAL(भिन्न, आधारः, L.off_sinks, has_sinks);
    factx.n_सक्रिय_खण्डानि      = factx.n_खण्डानि;
    factx.shared_सक्रिय_खण्डानि = false;

    यदि (has_आवरणम् && neq1 == 1 && neq3 == 1 && आवरणम्->ne[2] == 1 && factx.n_खण्डानि > 1 && factx.n_खण्डानि <= FA_MAX_KV_खण्डानि) {
        factx.n_सक्रिय_खण्डानि = fa_अवलोकनम्_सक्रिय_खण्डानि(
            octx->ctx->dma[0], आवरणम्, 0, 1, 0, 0, k->ne[1], FLASH_ATTN_खण्डम्_आकार,
            factx.n_खण्डानि, factx.spad_k, factx.आकार_k_खण्डम् * 2, factx.सक्रिय_खण्डानि);
        factx.shared_सक्रिय_खण्डानि = true;
    }

    यदि (has_sinks) {
        ध्रुव संरचना htp_tensor * sinks = octx->src[4];
        dma_queue * dma_q = octx->ctx->dma[0];
        dma_queue_प्रक्षिप(dma_q, dma_make_दत्त(factx.spad_sinks, sinks->दत्त), L.आकार_sinks, 0, sinks->आकार, 1);
        dma_queue_निष्कासय(dma_q);
    }

    work_queue_सततम्(octx->ctx->work_queue, flash_attn_ext_f16_thread, &factx, n_तन्तवः);

    प्रत्यागम HTP_स्थितिः_OK;
}
