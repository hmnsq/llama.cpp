#هدايت clang diagnostic ignored "-Wunused-variable"
#هدايت clang diagnostic ignored "-Wunused-function"
#هدايت clang diagnostic ignored "-Wunused-but-set-variable"

#ضمن <assert.h>
#ضمن <HAP_compute_res.h>
#ضمن <HAP_farf.h>
#ضمن <math.h>
#ضمن <stdbool.h>
#ضمن <stdatomic.h>
#ضمن <stddef.h>
#ضمن <stdint.h>
#ضمن <string.h>

#ضمن "dma-queue.h"
#ضمن "hex-fastdiv.h"
#ضمن "hex-profile.h"
#ضمن "hmx-queue.h"
#ضمن "hmx-utils.h"
#ضمن "hvx-utils.h"
#ضمن "hvx-dump.h"
#ضمن "hvx-copy.h"
#ضمن "hvx-reduce.h"
#ضمن "hvx-flash-attn.h"
#ضمن "htp-vtcm.h"
#ضمن "work-queue.h"

#وصف_ڪريو GGML_COMMON_DECL_C
#ضمن "ggml-common.h"
#ضمن "htp-ctx.h"
#ضمن "htp-ops.h"
#ضمن "htp-tensor.h"
#ضمن "hvx-quant.h"

#ضمن "flash-attn-ops.h"
#ضمن "hvx-fa-kernels.h"
#ضمن "hmx-fa-kernels.h"

// Must be multiple of 32
#وصف_ڪريو FLASH_ATTN_بلاڪ_سائيز (32 * 2)

بنية htp_fa_context {
    ثابت بنية htp_ops_context * octx;

    بنية fastdiv_قدرون src0_div21;
    بنية fastdiv_قدرون src0_div1;

    بنية fastdiv_قدرون نشر_rk2;
    بنية fastdiv_قدرون نشر_rk3;
    بنية fastdiv_قدرون نشر_rv2;
    بنية fastdiv_قدرون نشر_rv3;

    بنية fastdiv_قدرون src3_div2;
    بنية fastdiv_قدرون src3_div3;

    عائم scale;
    عائم max_bias;
    bool  has_softcap;
    __fp16 logit_softcap;

    uint32_t n_سر_log2;
    عائم m0;
    عائم m1;
    __fp16 slopes[512];

    uint32_t n_بلاڪون;
    uint32_t n_سرگرم_بلاڪون;
    bool     shared_سرگرم_بلاڪون;

    size_t سائيز_q_صف_padded;
    size_t سائيز_k_صف_padded;
    size_t سائيز_v_صف_padded;

    size_t سائيز_k_بلاڪ;
    size_t سائيز_v_بلاڪ;
    size_t سائيز_m_بلاڪ;

    uint32_t qrows;
    uint32_t qrows_في_thread;
    uint32_t qrow_شروعات;

    bool is_q_fp32;

    size_t سائيز_q_بلاڪ;
    size_t سائيز_vkq_acc;

    uint8_t *  spad_q;
    uint8_t *  spad_k;
    uint8_t *  spad_v;
    uint8_t *  spad_m;
    uint8_t *  spad_a;
    عائم *    spad_sinks;
    uint16_t   سرگرم_بلاڪون[FA_MAX_KV_بلاڪون];

    ثابت بنية htp_tensor * k;
    ثابت بنية htp_tensor * v;
};

بنية hmx_fa_context {
    ثابت بنية htp_ops_context * octx;
    ثابت بنية htp_tensor *      sinks;  // attention sinks (src[4]), NULL if absent
    ثابت بنية htp_tensor *      k;
    ثابت بنية htp_tensor *      v;
    bool         pipeline;  // true when n_kv_blocks >= FA_MIN_KV_BLOCKS && n_threads >= 2
    uint32_t     n_ڌاڳا;

    // Op parameters
    __fp16       scale;
    عائم        max_bias;
    bool         has_softcap;
    __fp16       logit_softcap;
    uint32_t     n_سر_log2;
    عائم        m0, m1;

    // Dimensions
    uint32_t     DK, DV;
    uint32_t     DK_pad, DV_pad;  // head_dim rounded up to 64 for HMX tiling
    uint32_t     n_kv;        // kv_len
    uint32_t     n_kv_heads;  // number of KV heads
    uint32_t     n_heads;     // number of Q heads
    uint32_t     G;           // GQA factor = n_heads / n_kv_heads
    بنية fastdiv_قدرون div_G;
    بنية fastdiv_قدرون src3_div2;
    بنية fastdiv_قدرون src3_div3;
    uint32_t     n_kv_بلاڪون;
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
    HVX_ويڪٽر * vtcm_m_vec;           // Row max [g_br]
    HVX_ويڪٽر * vtcm_l_vec;           // Row sum [g_br]
    HVX_ويڪٽر * vtcm_s_rowmax;        // Softmax intermediate [g_br]
    HVX_ويڪٽر * vtcm_p_rowsum;        // Softmax intermediate [g_br]
    HVX_ويڪٽر * vtcm_صف_bufs;        // Per-thread softmax row scratch [n_threads][2][Bc/64]
    uint8_t *    vtcm_hmx_scales_id;   // HMX output scales (identity)
    uint8_t *    vtcm_hmx_scales_qk;   // HMX output scales (qk_scale)
    __fp16 *     vtcm_نقاب_buf;        // VTCM mask buffer [Br * m_line], DMA'd per KV block
    __fp16 *     vtcm_slopes;          // ALiBi slopes [g_br]
    عائم *      vtcm_sinks;           // Attention sinks
    size_t       صف_buf_stride;       // HVX vectors per row buffer (Bc/64)
    size_t       نقاب_buf_صف_stride;  // elements (__fp16) per row in mask buffer
    size_t       q_tile_بائيٽون;
    size_t       k_tile_بائيٽون;
    size_t       o_tile_بائيٽون;
    size_t       کانو_vec_بائيٽون;
    size_t       d_tile_بائيٽون;
    bool         نقاب_نشر;       // true when mask->ne[2] == 1 (head-independent, single 2D DMA)
    dma_cache_fa m_cache;
};

// Scan mask rows [q_start, q_start + n_rows_q) and build compact list of KV blocks with any entry != -inf.
ساكن uint32_t fa_جاچ_سرگرم_بلاڪون(
        dma_قطار *               dma_q,
        ثابت بنية htp_tensor * نقاب,
        uint32_t                  q_شروعات,
        uint32_t                  n_صفون_q,
        uint32_t                  im2,
        uint32_t                  im3,
        uint32_t                  nek1,
        uint32_t                  Bc,
        uint32_t                  n_kv_بلاڪون,
        فراغ *                    vtcm_scratch,
        size_t                    scratch_بائيٽون,
        uint16_t *                سرگرم_بلاڪون) {
    ثابت size_t صف_بائيٽون  = (size_t) nek1 * حجم(__fp16);
    ثابت size_t صف_stride = hex_ڦيرو_مٿي(صف_بائيٽون, 128);

    إذا (!نقاب || n_kv_بلاڪون <= 1 || n_kv_بلاڪون > FA_MAX_KV_بلاڪون ||
        (size_t) n_صفون_q * صف_stride > scratch_بائيٽون) {
        #هدايت clang loop vectorize(disable)
        لكل (uint32_t b = 0; b < n_kv_بلاڪون; ++b) {
            سرگرم_بلاڪون[b] = (uint16_t) b;
        }
        أرجع n_kv_بلاڪون;
    }

    ثابت dma_addr_t m_src = نقاب->معلومات + q_شروعات * نقاب->nb[1] + im2 * نقاب->nb[2] + im3 * نقاب->nb[3];
    dma_قطار_ڌڪو(dma_q, dma_make_معلومات(vtcm_scratch, m_src), صف_stride, نقاب->nb[1], صف_بائيٽون, n_صفون_q);
    ثابت uint8_t * m_buf = (ثابت uint8_t *) dma_قطار_ڪڍو(dma_q).dst;

    ثابت HVX_ويڪٽر v_ninf   = Q6_Vh_vsplat_R(0xFC00);
    uint32_t         n_سرگرم = 0;

    لكل (uint32_t b = 0; b < n_kv_بلاڪون; ++b) {
        ثابت uint32_t col0 = b * Bc;
        ثابت uint32_t کانن_ڳڻپ_2 = MIN(Bc, nek1 - col0);

        لكل (uint32_t r = 0; r < n_صفون_q; ++r) {
            ثابت __fp16 * صف_ptr = (ثابت __fp16 *) (m_buf + r * صف_stride) + col0;
            HVX_ويڪٽر v_any = Q6_V_vzero();
            uint32_t c = 0;
            لكل (; c + 64 <= کانن_ڳڻپ_2; c += 64) {
                ثابت HVX_ويڪٽر v_m = *(ثابت HVX_ويڪٽر *) (صف_ptr + c);
                v_any = Q6_V_vor_VV(v_any, Q6_V_vxor_VV(v_m, v_ninf));
            }
            إذا (c < کانن_ڳڻپ_2) {
                ثابت uint32_t rem = کانن_ڳڻپ_2 - c;
                ثابت HVX_ويڪٽر_Pred q_rem = Q6_Q_vsetq2_R(rem * حجم(__fp16));
                ثابت HVX_ويڪٽر v_m = *(ثابت HVX_ويڪٽر *) (صف_ptr + c);
                ثابت HVX_ويڪٽر v_diff = Q6_V_vand_QV(q_rem, Q6_V_vxor_VV(v_m, v_ninf));
                v_any = Q6_V_vor_VV(v_any, v_diff);
            }
            v_any = Q6_V_vor_VV(v_any, Q6_V_vror_VR(v_any, 64));
            v_any = Q6_V_vor_VV(v_any, Q6_V_vror_VR(v_any, 32));
            v_any = Q6_V_vor_VV(v_any, Q6_V_vror_VR(v_any, 16));
            v_any = Q6_V_vor_VV(v_any, Q6_V_vror_VR(v_any, 8));
            v_any = Q6_V_vor_VV(v_any, Q6_V_vror_VR(v_any, 4));
            إذا (Q6_R_vextract_VR(v_any, 0) != 0) {
                سرگرم_بلاڪون[n_سرگرم++] = (uint16_t) b;
                اكسر;
            }
        }
    }

    إذا (n_سرگرم == 0 && n_kv_بلاڪون > 0) {
        سرگرم_بلاڪون[0] = 0;
        n_سرگرم = 1;
    }
    أرجع n_سرگرم;
}

ساكن فراغ flash_attn_ext_f16_thread(بنانشان صحيح nth, بنانشان صحيح ith, فراغ * معلومات) {
    بنية htp_fa_context * factx = (بنية htp_fa_context *) معلومات;
    ثابت بنية htp_ops_context * octx = factx->octx;
    ثابت بنية htp_tensor * q     = octx->src[0];
    ثابت بنية htp_tensor * k     = octx->src[1];
    ثابت بنية htp_tensor * v     = octx->src[2];
    ثابت بنية htp_tensor * نقاب  = octx->src[3];
    ثابت بنية htp_tensor * sinks = octx->src[4];
    ثابت بنية htp_tensor * dst   = octx->dst;

    ثابت uint32_t neq0 = q->ne[0];
    ثابت uint32_t neq1 = q->ne[1];
    ثابت uint32_t neq2 = q->ne[2];
    ثابت uint32_t neq3 = q->ne[3];

    ثابت uint32_t nek0 = k->ne[0];
    ثابت uint32_t nek1 = k->ne[1];
    ثابت uint32_t nek2 = k->ne[2];
    ثابت uint32_t nek3 = k->ne[3];

    ثابت uint32_t nev0 = v->ne[0];
    ثابت uint32_t nev1 = v->ne[1];
    ثابت uint32_t nev2 = v->ne[2];
    ثابت uint32_t nev3 = v->ne[3];

    ثابت uint32_t nbq1 = q->nb[1];
    ثابت uint32_t nbq2 = q->nb[2];
    ثابت uint32_t nbq3 = q->nb[3];

    ثابت uint32_t nbk1 = k->nb[1];
    ثابت uint32_t nbk2 = k->nb[2];
    ثابت uint32_t nbk3 = k->nb[3];

    ثابت uint32_t nbv1 = v->nb[1];
    ثابت uint32_t nbv2 = v->nb[2];
    ثابت uint32_t nbv3 = v->nb[3];

    ثابت uint32_t ne1 = dst->ne[1];
    ثابت uint32_t ne2 = dst->ne[2];
    ثابت uint32_t ne3 = dst->ne[3];

    ثابت uint32_t nb1 = dst->nb[1];
    ثابت uint32_t nb2 = dst->nb[2];
    ثابت uint32_t nb3 = dst->nb[3];

    // total rows in q
    ثابت uint32_t dr  = factx->qrows_في_thread;
    ثابت uint32_t ir0 = factx->qrow_شروعات + dr * ith;
    ثابت uint32_t ir1 = MIN(ir0 + dr, factx->qrow_شروعات + factx->qrows);

    إذا (ir0 >= ir1) أرجع;

    بنية htp_thread_trace * tr = &octx->ctx->trace[ith];

    dma_قطار * dma_q = octx->ctx->dma[ith];

    ثابت uint32_t DK = nek0;
    ثابت uint32_t DV = nev0;

    ثابت size_t سائيز_q_صف = DK * ((q->type == HTP_TYPE_F32) ? 4 : 2);
    ثابت size_t سائيز_k_صف = htp_tensor_get_صف_سائيز(k->type, DK);
    ثابت size_t سائيز_v_صف = htp_tensor_get_صف_سائيز(v->type, DV);

    // Scratchpad buffers for Q, K, V, Mask, and VKQ32 accumulator
    uint8_t * spad_q = factx->spad_q + factx->سائيز_q_بلاڪ * ith;
    uint8_t * spad_k = factx->spad_k + factx->سائيز_k_بلاڪ * 2 * ith;
    uint8_t * spad_v = factx->spad_v + factx->سائيز_v_بلاڪ * 2 * ith;
    uint8_t * spad_m = factx->spad_m + (نقاب ? factx->سائيز_m_بلاڪ * HVX_FA_DMA_CACHE_سائيز : 0) * ith;
    uint8_t * spad_a = factx->spad_a + factx->سائيز_vkq_acc * ith;

    dma_cache_dm m_cache;
    dma_cache_dm_init(&m_cache, spad_m, factx->سائيز_m_بلاڪ, HVX_FA_DMA_CACHE_سائيز);

    uint16_t thread_سرگرم_بلاڪون[FA_MAX_KV_بلاڪون];
    ثابت uint16_t * سرگرم_بلاڪون = factx->shared_سرگرم_بلاڪون ? factx->سرگرم_بلاڪون : thread_سرگرم_بلاڪون;
    uint32_t n_سرگرم_بلاڪون       = factx->n_سرگرم_بلاڪون;
    dma_addr_t prev_mp_بنياد        = 0;

    إذا (!factx->shared_سرگرم_بلاڪون && !نقاب) {
        #هدايت clang loop vectorize(disable)
        لكل (uint32_t b = 0; b < factx->n_بلاڪون; ++b) {
            thread_سرگرم_بلاڪون[b] = (uint16_t) b;
        }
    }

    ثابت size_t سائيز_vkq_acc_single = hex_ڦيرو_مٿي(DV * حجم(عائم), 128);

    uint32_t ir = ir0;
    طالما (ir < ir1) {
        ثابت uint32_t iq3 = fastdiv(ir, &factx->src0_div21);
        ثابت uint32_t iq2 = fastdiv(ir - iq3*neq2*neq1, &factx->src0_div1);
        ثابت uint32_t iq1 = (ir - iq3*neq2*neq1 - iq2 * neq1);

        ثابت uint32_t ik3 = fastdiv(iq3, &factx->نشر_rk3);
        ثابت uint32_t ik2 = fastdiv(iq2, &factx->نشر_rk2);

        ثابت uint32_t iv3 = fastdiv(iq3, &factx->نشر_rv3);
        ثابت uint32_t iv2 = fastdiv(iq2, &factx->نشر_rv2);

        uint32_t G_local = 1;
        إذا (neq1 == 1 && (نقاب == عدم || نقاب->ne[2] == 1)) {
            طالما (ir + G_local < ir1 && G_local < FA_HVX_G_MAX) {
                ثابت uint32_t ايندڙ_ir = ir + G_local;
                ثابت uint32_t ايندڙ_iq3 = fastdiv(ايندڙ_ir, &factx->src0_div21);
                ثابت uint32_t ايندڙ_iq2 = fastdiv(ايندڙ_ir - ايندڙ_iq3*neq2*neq1, &factx->src0_div1);
                ثابت uint32_t ايندڙ_iq1 = (ايندڙ_ir - ايندڙ_iq3*neq2*neq1 - ايندڙ_iq2 * neq1);

                ثابت uint32_t ايندڙ_ik3 = fastdiv(ايندڙ_iq3, &factx->نشر_rk3);
                ثابت uint32_t ايندڙ_ik2 = fastdiv(ايندڙ_iq2, &factx->نشر_rk2);

                ثابت uint32_t ايندڙ_iv3 = fastdiv(ايندڙ_iq3, &factx->نشر_rv3);
                ثابت uint32_t ايندڙ_iv2 = fastdiv(ايندڙ_iq2, &factx->نشر_rv2);

                إذا (ايندڙ_ik2 != ik2 || ايندڙ_ik3 != ik3 || ايندڙ_iv2 != iv2 || ايندڙ_iv3 != iv3 || ايندڙ_iq1 != iq1 || ايندڙ_iq3 != iq3) {
                    اكسر;
                }
                G_local++;
            }
        }

        dma_addr_t mp_بنياد = 0;
        إذا (نقاب) {
            ثابت uint32_t im2 = fastmodulo(iq2, نقاب->ne[2], &factx->src3_div2);
            ثابت uint32_t im3 = fastmodulo(iq3, نقاب->ne[3], &factx->src3_div3);
            mp_بنياد = نقاب->معلومات + iq1*نقاب->nb[1] + im2*نقاب->nb[2] + im3*نقاب->nb[3];
            إذا (!factx->shared_سرگرم_بلاڪون && mp_بنياد != prev_mp_بنياد) {
                n_سرگرم_بلاڪون = fa_جاچ_سرگرم_بلاڪون(
                    dma_q, نقاب, iq1, 1, im2, im3, nek1, FLASH_ATTN_بلاڪ_سائيز,
                    factx->n_بلاڪون, spad_k, factx->سائيز_k_بلاڪ * 2, thread_سرگرم_بلاڪون);
                prev_mp_بنياد = mp_بنياد;
            }
        }

        uint32_t heads[FA_HVX_G_MAX];
        HVX_ويڪٽر لاهي_vecs[FA_HVX_G_MAX] __attribute__((aligned(128)));
        HVX_ويڪٽر S_vec[FA_HVX_G_MAX]      __attribute__((aligned(128)));
        HVX_ويڪٽر M_vec[FA_HVX_G_MAX]      __attribute__((aligned(128)));
        uint8_t * q_ptrs[FA_HVX_G_MAX];
        عائم * vkq_ptrs[FA_HVX_G_MAX];

        لكل (uint32_t g = 0; g < G_local; ++g) {
            ثابت uint32_t r = ir + g;
            ثابت uint32_t r_iq3 = fastdiv(r, &factx->src0_div21);
            ثابت uint32_t r_iq2 = fastdiv(r - r_iq3*neq2*neq1, &factx->src0_div1);
            ثابت uint32_t r_iq1 = (r - r_iq3*neq2*neq1 - r_iq2 * neq1);

            heads[g] = r_iq2;
            ثابت __fp16 لاهي = factx->slopes[r_iq2];
            لاهي_vecs[g] = hvx_vec_splat_f16(لاهي);

            S_vec[g] = hvx_vec_splat_f32(0.0f);
            M_vec[g] = hvx_vec_splat_f32(HTP_FA_M_INITIAL_VAL);

            uint8_t * q_dst = spad_q + g * factx->سائيز_q_صف_padded;
            q_ptrs[g] = q_dst;

            عائم * vkq_dst = (عائم *)(spad_a + g * سائيز_vkq_acc_single);
            vkq_ptrs[g] = vkq_dst;
            hvx_splat_f32_a((uint8_t *) vkq_dst, 0, DV);

            // Fetch Q row g
            ثابت dma_addr_t q_صف_ptr = q->معلومات + r_iq1*nbq1 + r_iq2*nbq2 + r_iq3*nbq3;
            dma_قطار_ڌڪو(dma_q, dma_make_معلومات(q_dst, q_صف_ptr), factx->سائيز_q_صف_padded, nbq1, سائيز_q_صف, 1);
        }

        // Prefetch first two active blocks
        لكل (uint32_t قدم = 0; قدم < MIN(n_سرگرم_بلاڪون, 2); ++قدم) {
            ثابت uint32_t ib = سرگرم_بلاڪون[قدم];
            ثابت uint32_t ic_شروعات = ib * FLASH_ATTN_بلاڪ_سائيز;
            ثابت uint32_t current_بلاڪ_سائيز = MIN(FLASH_ATTN_بلاڪ_سائيز, nek1 - ic_شروعات);

            // K
            ثابت dma_addr_t k_src = k->معلومات + ic_شروعات*nbk1 + ik2*nbk2 + ik3*nbk3;
            uint8_t * k_dst = spad_k + (قدم % 2) * factx->سائيز_k_بلاڪ;
            dma_قطار_ڌڪو(dma_q, dma_make_معلومات(k_dst, k_src), factx->سائيز_k_صف_padded, nbk1, سائيز_k_صف, current_بلاڪ_سائيز);

            // V
            ثابت dma_addr_t v_src = v->معلومات + ic_شروعات*nbv1 + iv2*nbv2 + iv3*nbv3;
            uint8_t * v_dst = spad_v + (قدم % 2) * factx->سائيز_v_بلاڪ;
            dma_قطار_ڌڪو(dma_q, dma_make_معلومات(v_dst, v_src), factx->سائيز_v_صف_padded, nbv1, سائيز_v_صف, current_بلاڪ_سائيز);

            // Mask
            إذا (نقاب) {
                ثابت dma_addr_t m_src = mp_بنياد + ic_شروعات * حجم(__fp16);
                dma_cache_dm_ڌڪو(dma_q, &m_cache, ib, m_src, current_بلاڪ_سائيز * 2, current_بلاڪ_سائيز * 2, current_بلاڪ_سائيز * 2, 1);
            }
        }

        // Pop all Q rows
        لكل (uint32_t g = 0; g < G_local; ++g) {
            uint8_t * q_ptr_vtcm = (فراغ *) dma_قطار_ڪڍو(dma_q).dst;
            إذا (factx->is_q_fp32) {
                hvx_copy_f16_f32_aa(q_ptr_vtcm, q_ptr_vtcm, DK);
            }
        }

        ثابت HVX_ويڪٽر v_neg_inf = Q6_Vh_vsplat_R(0xfbff);
        ثابت bool has_softcap     = factx->has_softcap;
        ثابت HVX_ويڪٽر v_cap     = has_softcap ? hvx_vec_splat_f16(factx->logit_softcap) : Q6_V_vzero();
        ثابت HVX_ويڪٽر vinf      = Q6_Vh_vsplat_R(0xFC00);
        ثابت HVX_ويڪٽر vmin      = Q6_Vh_vsplat_R(0xFBFF);
        ثابت HVX_ويڪٽر v_log2e   = hvx_vec_splat_f16(EXP_LOG2E_F);
        ثابت uint32_t stride_v2   = factx->سائيز_v_صف_padded * 2;

        لكل (uint32_t قدم = 0; قدم < n_سرگرم_بلاڪون; ++قدم) {
            ثابت uint32_t ib = سرگرم_بلاڪون[قدم];
            ثابت uint32_t ic_شروعات = ib * FLASH_ATTN_بلاڪ_سائيز;
            ثابت uint32_t current_بلاڪ_سائيز = MIN(FLASH_ATTN_بلاڪ_سائيز, nek1 - ic_شروعات);

            // Wait for DMA
            uint8_t * k_بنياد = (فراغ *) dma_قطار_ڪڍو(dma_q).dst; // K
            uint8_t * v_بنياد = (فراغ *) dma_قطار_ڪڍو(dma_q).dst; // V
            __fp16  * m_بنياد = نقاب ? (__fp16 *) dma_قطار_ڪڍو(dma_q).dst : عدم; // M

            إذا (factx->k->type == HTP_TYPE_Q8_0) {
                htp_trace_event_شروعات(tr, HTP_TRACE_EVT_HVX_FA_K_PREP, ir);
                لكل (uint32_t r = 0; r < current_بلاڪ_سائيز; ++r) {
                    __fp16 * صف_k = (__fp16 *)(k_بنياد + r * factx->سائيز_k_صف_padded);
                    hvx_dequantize_صف_flat_q8_0_f16(صف_k, صف_k, DK);
                }
                htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_FA_K_PREP, ir);
            }
            إذا (factx->v->type == HTP_TYPE_Q8_0) {
                htp_trace_event_شروعات(tr, HTP_TRACE_EVT_HVX_FA_V_PREP, ir);
                لكل (uint32_t r = 0; r < current_بلاڪ_سائيز; ++r) {
                    __fp16 * صف_v = (__fp16 *)(v_بنياد + r * factx->سائيز_v_صف_padded);
                    hvx_dequantize_صف_flat_q8_0_f16(صف_v, صف_v, DV);
                }
                htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_FA_V_PREP, ir);
            }

            لكل (uint32_t g = 0; g < G_local; ++g) {
                ثابت uint32_t سر_ir = ir + g;
                uint8_t * q_ptr_vtcm = q_ptrs[g];
                عائم * VKQ32 = vkq_ptrs[g];
                ثابت HVX_ويڪٽر لاهي_vec = لاهي_vecs[g];

                htp_trace_event_شروعات(tr, HTP_TRACE_EVT_HVX_FA_QK, سر_ir);

                HVX_ويڪٽر scores_f16 = Q6_V_vzero();
                إذا (current_بلاڪ_سائيز > 0) {
                    HVX_ويڪٽر scores0 = hvx_dot_f16_f16_aa_rx32(q_ptr_vtcm, k_بنياد, factx->سائيز_k_صف_padded, DK, factx->scale);
                    HVX_ويڪٽر scores1 = (current_بلاڪ_سائيز > 32) ? hvx_dot_f16_f16_aa_rx32(q_ptr_vtcm, k_بنياد + 32 * factx->سائيز_k_صف_padded, factx->سائيز_k_صف_padded, DK, factx->scale) : Q6_V_vzero();
                    scores_f16 = hvx_vec_f32_منزل_f16(scores0, scores1);
                }

                إذا (has_softcap) {
                    scores_f16 = hvx_vec_tanh_f16(scores_f16);
                    scores_f16 = hvx_vec_mul_f16_f16(scores_f16, v_cap);
                }

                HVX_ويڪٽر_Pred q_پڇ_keep = Q6_Q_vsetq2_R(current_بلاڪ_سائيز * حجم(__fp16));

                إذا (نقاب) {
                    HVX_ويڪٽر m_vals_f16 = *(ثابت HVX_UVector *) m_بنياد;
                    HVX_ويڪٽر_Pred is_inf = Q6_Q_vcmp_eq_VhVh(m_vals_f16, vinf);
                    m_vals_f16 = Q6_V_vmux_QVV(is_inf, vmin, m_vals_f16);

                    HVX_ويڪٽر m_scaled = hvx_vec_mul_f16_f16(m_vals_f16, لاهي_vec);
                    scores_f16 = Q6_V_vmux_QVV(q_پڇ_keep, hvx_vec_جوڙيو_f16_f16(scores_f16, m_scaled), v_neg_inf);
                } وإلا {
                    scores_f16 = Q6_V_vmux_QVV(q_پڇ_keep, scores_f16, v_neg_inf);
                }

                HVX_ويڪٽر v_max_f16 = hvx_vec_reduce_max_f16(scores_f16);
                HVX_ويڪٽر v_max     = Q6_V_lo_W(hvx_vec_f16_منزل_f32(v_max_f16));
                htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_FA_QK, سر_ir);

                // prefetch K for step + 2 after last head finished QK
                إذا (g + 1 == G_local && قدم + 2 < n_سرگرم_بلاڪون) {
                    ثابت uint32_t ايندڙ_ib = سرگرم_بلاڪون[قدم + 2];
                    ثابت uint32_t ايندڙ_ic_شروعات = ايندڙ_ib * FLASH_ATTN_بلاڪ_سائيز;
                    ثابت uint32_t ايندڙ_بلاڪ_سائيز = MIN(FLASH_ATTN_بلاڪ_سائيز, nek1 - ايندڙ_ic_شروعات);

                    ثابت dma_addr_t k_src = k->معلومات + ايندڙ_ic_شروعات*nbk1 + ik2*nbk2 + ik3*nbk3;
                    dma_قطار_ڌڪو(dma_q, dma_make_معلومات(k_بنياد, k_src), factx->سائيز_k_صف_padded, nbk1, سائيز_k_صف, ايندڙ_بلاڪ_سائيز);
                }

                htp_trace_event_شروعات(tr, HTP_TRACE_EVT_HVX_FA_SFM, سر_ir);
                {
                    HVX_ويڪٽر M_new_vec = Q6_Vsf_vmax_VsfVsf(v_max, M_vec[g]);
                    HVX_ويڪٽر diff_vec  = HVX_OP_SUB_F32(M_vec[g], M_new_vec);

                    HVX_ويڪٽر diff_f16   = hvx_vec_f32_منزل_f16(diff_vec, diff_vec);
                    HVX_ويڪٽر diff_base2 = hvx_vec_mul_f16_f16(diff_f16, v_log2e);
                    HVX_ويڪٽر ms_f16     = hvx_vec_exp2_f16(diff_base2);
                    HVX_ويڪٽر ms_vec     = Q6_V_lo_W(hvx_vec_f16_منزل_f32(ms_f16));

                    M_vec[g] = M_new_vec;

                    HVX_ويڪٽر v_m_vec_f16 = hvx_vec_f32_منزل_f16(M_vec[g], M_vec[g]);
                    HVX_ويڪٽر v_s_minus_m = Q6_Vqf16_vsub_VhfVhf(scores_f16, v_m_vec_f16);
                    HVX_ويڪٽر v_s_minus_m_base2 = hvx_vec_mul_f16_f16(Q6_Vhf_equals_Vqf16(v_s_minus_m), v_log2e);

                    HVX_ويڪٽر P = hvx_vec_exp2_f16(v_s_minus_m_base2);
                    P = Q6_V_vmux_QVV(q_پڇ_keep, P, Q6_V_vzero());

                    HVX_ويڪٽر_Pair P_pair = hvx_vec_f16_منزل_f32(P);
                    HVX_ويڪٽر P0 = Q6_V_lo_W(P_pair);
                    HVX_ويڪٽر P1 = Q6_V_hi_W(P_pair);
                    HVX_ويڪٽر p_جوڙ_vec = hvx_vec_reduce_جوڙ_f32(HVX_OP_جوڙيو_F32(P0, P1));

                    S_vec[g] = HVX_OP_جوڙيو_F32(HVX_OP_MUL_F32(S_vec[g], ms_vec), p_جوڙ_vec);

                    ثابت uint8_t * v_ptr = v_بنياد;

                    إذا (DV == 64) {
                        HVX_ويڪٽر_Pair vkq0 = *((ثابت HVX_ويڪٽر_Pair *) VKQ32);
                        vkq0 = Q6_W_vcombine_VV(
                            HVX_OP_MUL_F32(Q6_V_hi_W(vkq0), ms_vec),
                            HVX_OP_MUL_F32(Q6_V_lo_W(vkq0), ms_vec)
                        );

                        لكل (uint32_t j = 0; j < current_بلاڪ_سائيز; j += 2) {
                            HVX_ويڪٽر S0 = hvx_vec_repl_f16(Q6_V_vror_VR(P, j * 2));
                            ثابت HVX_ويڪٽر * vx0 = (ثابت HVX_ويڪٽر *) v_ptr;
                            إذا (j + 1 == current_بلاڪ_سائيز) {
                                vkq0 = hvx_vec_mpyacc_f32_f16(vkq0, Q6_Vh_vshuff_Vh(vx0[0]), S0);
                                اكسر;
                            }

                            HVX_ويڪٽر S1 = hvx_vec_repl_f16(Q6_V_vror_VR(P, (j + 1) * 2));
                            ثابت HVX_ويڪٽر * vx1 = (ثابت HVX_ويڪٽر *) (v_ptr + factx->سائيز_v_صف_padded);
                            vkq0 = hvx_vec_mpyacc_f32_f16(vkq0, Q6_Vh_vshuff_Vh(vx0[0]), S0);
                            vkq0 = hvx_vec_mpyacc_f32_f16(vkq0, Q6_Vh_vshuff_Vh(vx1[0]), S1);
                            v_ptr += stride_v2;
                        }

                        *((HVX_ويڪٽر_Pair *) VKQ32) = vkq0;
                    } وإلا إذا (DV == 128) {
                        HVX_ويڪٽر_Pair vkq0 = ((ثابت HVX_ويڪٽر_Pair *) VKQ32)[0];
                        HVX_ويڪٽر_Pair vkq1 = ((ثابت HVX_ويڪٽر_Pair *) VKQ32)[1];
                        vkq0 = Q6_W_vcombine_VV(
                            HVX_OP_MUL_F32(Q6_V_hi_W(vkq0), ms_vec),
                            HVX_OP_MUL_F32(Q6_V_lo_W(vkq0), ms_vec)
                        );
                        vkq1 = Q6_W_vcombine_VV(
                            HVX_OP_MUL_F32(Q6_V_hi_W(vkq1), ms_vec),
                            HVX_OP_MUL_F32(Q6_V_lo_W(vkq1), ms_vec)
                        );

                        لكل (uint32_t j = 0; j < current_بلاڪ_سائيز; j += 2) {
                            HVX_ويڪٽر S0 = hvx_vec_repl_f16(Q6_V_vror_VR(P, j * 2));
                            ثابت HVX_ويڪٽر * vx0 = (ثابت HVX_ويڪٽر *) v_ptr;
                            إذا (j + 1 == current_بلاڪ_سائيز) {
                                vkq0 = hvx_vec_mpyacc_f32_f16(vkq0, Q6_Vh_vshuff_Vh(vx0[0]), S0);
                                vkq1 = hvx_vec_mpyacc_f32_f16(vkq1, Q6_Vh_vshuff_Vh(vx0[1]), S0);
                                اكسر;
                            }

                            HVX_ويڪٽر S1 = hvx_vec_repl_f16(Q6_V_vror_VR(P, (j + 1) * 2));
                            ثابت HVX_ويڪٽر * vx1 = (ثابت HVX_ويڪٽر *) (v_ptr + factx->سائيز_v_صف_padded);
                            vkq0 = hvx_vec_mpyacc_f32_f16(vkq0, Q6_Vh_vshuff_Vh(vx0[0]), S0);
                            vkq0 = hvx_vec_mpyacc_f32_f16(vkq0, Q6_Vh_vshuff_Vh(vx1[0]), S1);
                            vkq1 = hvx_vec_mpyacc_f32_f16(vkq1, Q6_Vh_vshuff_Vh(vx0[1]), S0);
                            vkq1 = hvx_vec_mpyacc_f32_f16(vkq1, Q6_Vh_vshuff_Vh(vx1[1]), S1);
                            v_ptr += stride_v2;
                        }

                        ((HVX_ويڪٽر_Pair *) VKQ32)[0] = vkq0;
                        ((HVX_ويڪٽر_Pair *) VKQ32)[1] = vkq1;
                    } وإلا إذا (DV == 256) {
                        HVX_ويڪٽر_Pair vkq0 = ((ثابت HVX_ويڪٽر_Pair *) VKQ32)[0];
                        HVX_ويڪٽر_Pair vkq1 = ((ثابت HVX_ويڪٽر_Pair *) VKQ32)[1];
                        HVX_ويڪٽر_Pair vkq2 = ((ثابت HVX_ويڪٽر_Pair *) VKQ32)[2];
                        HVX_ويڪٽر_Pair vkq3 = ((ثابت HVX_ويڪٽر_Pair *) VKQ32)[3];
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

                        لكل (uint32_t j = 0; j < current_بلاڪ_سائيز; j += 2) {
                            HVX_ويڪٽر S0 = hvx_vec_repl_f16(Q6_V_vror_VR(P, j * 2));
                            ثابت HVX_ويڪٽر * vx0 = (ثابت HVX_ويڪٽر *) v_ptr;
                            إذا (j + 1 == current_بلاڪ_سائيز) {
                                vkq0 = hvx_vec_mpyacc_f32_f16(vkq0, Q6_Vh_vshuff_Vh(vx0[0]), S0);
                                vkq1 = hvx_vec_mpyacc_f32_f16(vkq1, Q6_Vh_vshuff_Vh(vx0[1]), S0);
                                vkq2 = hvx_vec_mpyacc_f32_f16(vkq2, Q6_Vh_vshuff_Vh(vx0[2]), S0);
                                vkq3 = hvx_vec_mpyacc_f32_f16(vkq3, Q6_Vh_vshuff_Vh(vx0[3]), S0);
                                اكسر;
                            }

                            HVX_ويڪٽر S1 = hvx_vec_repl_f16(Q6_V_vror_VR(P, (j + 1) * 2));
                            ثابت HVX_ويڪٽر * vx1 = (ثابت HVX_ويڪٽر *) (v_ptr + factx->سائيز_v_صف_padded);
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

                        ((HVX_ويڪٽر_Pair *) VKQ32)[0] = vkq0;
                        ((HVX_ويڪٽر_Pair *) VKQ32)[1] = vkq1;
                        ((HVX_ويڪٽر_Pair *) VKQ32)[2] = vkq2;
                        ((HVX_ويڪٽر_Pair *) VKQ32)[3] = vkq3;
                    } وإلا {
                        hvx_scale_vec_f32_aa((uint8_t *) VKQ32, (ثابت uint8_t *) VKQ32, DV, ms_vec);

                        لكل (uint32_t j = 0; j < current_بلاڪ_سائيز; j += 2) {
                            إذا (j + 1 == current_بلاڪ_سائيز) {
                                HVX_ويڪٽر S0 = hvx_vec_repl_f16(Q6_V_vror_VR(P, j * 2));
                                hvx_mad_f32_f16_aa_vec(VKQ32, v_ptr, S0, DV);
                                اكسر;
                            }

                            HVX_ويڪٽر S0 = hvx_vec_repl_f16(Q6_V_vror_VR(P, j * 2));
                            HVX_ويڪٽر S1 = hvx_vec_repl_f16(Q6_V_vror_VR(P, (j + 1) * 2));

                            hvx_mad_f32_f16_aa_rx2_vec(VKQ32, v_ptr, v_ptr + factx->سائيز_v_صف_padded, S0, S1, DV);
                            v_ptr += stride_v2;
                        }
                    }
                }
                htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_FA_SFM, سر_ir);

                // prefetch V and mask for step + 2 after last head finished V accumulation
                إذا (g + 1 == G_local && قدم + 2 < n_سرگرم_بلاڪون) {
                    ثابت uint32_t ايندڙ_ib = سرگرم_بلاڪون[قدم + 2];
                    ثابت uint32_t ايندڙ_ic_شروعات = ايندڙ_ib * FLASH_ATTN_بلاڪ_سائيز;
                    ثابت uint32_t ايندڙ_بلاڪ_سائيز = MIN(FLASH_ATTN_بلاڪ_سائيز, nek1 - ايندڙ_ic_شروعات);

                    // V
                    ثابت dma_addr_t v_src = v->معلومات + ايندڙ_ic_شروعات*nbv1 + iv2*nbv2 + iv3*nbv3;
                    dma_قطار_ڌڪو(dma_q, dma_make_معلومات(v_بنياد, v_src), factx->سائيز_v_صف_padded, nbv1, سائيز_v_صف, ايندڙ_بلاڪ_سائيز);

                    // Mask
                    إذا (نقاب) {
                        ثابت dma_addr_t m_src = mp_بنياد + ايندڙ_ic_شروعات * حجم(__fp16);
                        dma_cache_dm_ڌڪو(dma_q, &m_cache, ايندڙ_ib, m_src, ايندڙ_بلاڪ_سائيز * 2, ايندڙ_بلاڪ_سائيز * 2, ايندڙ_بلاڪ_سائيز * 2, 1);
                    }
                }
            } // end for g
        } // end for step

        لكل (uint32_t g = 0; g < G_local; ++g) {
            ثابت uint32_t سر_ir = ir + g;
            ثابت uint32_t h = heads[g];
            عائم * VKQ32 = vkq_ptrs[g];

            htp_trace_event_شروعات(tr, HTP_TRACE_EVT_HVX_O_PROC, سر_ir);

            عائم M = hvx_vec_get_f32(M_vec[g]);
            عائم S = hvx_vec_get_f32(S_vec[g]);

            إذا (sinks) {
                ثابت عائم s = factx->spad_sinks[h];

                عائم vs = 1.0f;

                إذا (s > M) {
                    HVX_ويڪٽر diff_vec = hvx_vec_splat_f32(M - s);
                    HVX_ويڪٽر ms_vec   = hvx_vec_exp_f32(diff_vec);
                    hvx_scale_vec_f32_aa((uint8_t *) VKQ32, (ثابت uint8_t *) VKQ32, DV, ms_vec);

                    عائم ms = hvx_vec_get_f32(ms_vec);
                    S = S * ms + vs;
                } وإلا {
                    HVX_ويڪٽر diff_vec = hvx_vec_splat_f32(s - M);
                    vs = hvx_vec_get_f32(hvx_vec_exp_f32(diff_vec));
                    S += vs;
                }
            }

            ثابت عائم S_inv = S == 0.0f ? 0.0f : 1.0f/S;
            hvx_scale_f32_aa((uint8_t *) VKQ32, (ثابت uint8_t *) VKQ32, DV, S_inv);

            ثابت uint32_t r_iq3 = fastdiv(سر_ir, &factx->src0_div21);
            ثابت uint32_t r_iq2 = fastdiv(سر_ir - r_iq3*neq2*neq1, &factx->src0_div1);
            ثابت uint32_t r_iq1 = (سر_ir - r_iq3*neq2*neq1 - r_iq2 * neq1);

            uint8_t * dst_ptr = (uint8_t *) dst->معلومات + r_iq2 * dst->nb[1] + r_iq1 * dst->nb[2] + r_iq3 * dst->nb[3];

            إذا (dst->type == HTP_TYPE_F32) {
                hvx_copy_f32_ua(dst_ptr, (uint8_t *) VKQ32, DV);
            } وإلا إذا (dst->type == HTP_TYPE_F16) {
                hvx_copy_f16_f32_ua(dst_ptr, (uint8_t *) VKQ32, DV);
            }
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_O_PROC, سر_ir);
        }

        ir += G_local;
    }
}

// ============================================================================
// HMX Phase args and thread logic
// ============================================================================

تعريفنوع بنية {
    بنية hmx_fa_context * factx;
    uint32_t                kv_صفون;
    size_t                  src_stride;
    فراغ *                  curr_k;
    uint32_t                kv_شروعات;
    uint32_t                صفون_في_t;
    size_t                  buf_idx;
} fa_k_int_دليلون_t;

ساكن فراغ fa_k_interleave_thread(بنانشان صحيح n, بنانشان صحيح i, فراغ * معلومات) {
    fa_k_int_دليلون_t *       دليلون_2  = (fa_k_int_دليلون_t *) معلومات;
    بنية hmx_fa_context * factx = دليلون_2->factx;

    ثابت uint32_t ڪل_صفون = دليلون_2->kv_صفون;
    ثابت uint32_t صفون_في_t = دليلون_2->صفون_في_t;
    ثابت uint32_t شروعات      = i * صفون_في_t;
    ثابت uint32_t پڇاڙي        = (uint32_t) hex_smin(شروعات + صفون_في_t, ڪل_صفون);

    إذا (شروعات >= ڪل_صفون) {
        أرجع;
    }

    بنية htp_thread_trace * tr = &factx->octx->ctx->trace[i];
    htp_trace_event_شروعات(tr, HTP_TRACE_EVT_HVX_FA_K_PREP, (uint16_t) (دليلون_2->kv_شروعات + شروعات));
    إذا (factx->k->type == HTP_TYPE_Q8_0) {
        لكل (uint32_t r = شروعات; r < پڇاڙي; ++r) {
            __fp16 * صف_k = (__fp16 *)((حرف *)دليلون_2->curr_k + r * دليلون_2->src_stride * حجم(__fp16));
            hvx_dequantize_صف_flat_q8_0_f16(صف_k, صف_k, factx->DK);
        }
    }
    hmx_interleave_صفون_منزل_tiles(factx->vtcm_k_tiles[دليلون_2->buf_idx], (ثابت __fp16 *) دليلون_2->curr_k, ڪل_صفون, factx->DK_pad,
                             دليلون_2->src_stride, شروعات, پڇاڙي);
    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_FA_K_PREP, (uint16_t) (دليلون_2->kv_شروعات + شروعات));
}

ساكن فراغ fa_phase_k_interleave(بنية hmx_fa_context * factx, uint32_t kv_صفون, size_t src_stride, فراغ * curr_k, uint32_t kv_شروعات, size_t buf_idx) {
    work_قطار_t wp = factx->octx->ctx->work_قطار;
    uint32_t n = 1;
    إذا (factx->n_ڌاڳا > 1 && kv_صفون >= factx->n_ڌاڳا * 2) {
        n = factx->n_ڌاڳا;
    }
    uint32_t صفون_في_t = hex_align_مٿي(hmx_ceil_div(kv_صفون, n), 2);
    fa_k_int_دليلون_t دليلون_2 = { factx, kv_صفون, src_stride, curr_k, kv_شروعات, صفون_في_t, buf_idx };
    إذا (n > 1) {
        work_قطار_لاڳيتو(wp, fa_k_interleave_thread, &دليلون_2, n);
    } وإلا {
        fa_k_interleave_thread(1, 0, &دليلون_2);
    }
}

تعريفنوع بنية {
    بنية hmx_fa_context * factx;
    uint32_t                kv_صفون;
    size_t                  src_stride;
    فراغ *                  v_src;
    فراغ *                  v_tiles_dst;
    size_t                  n_کانو_tiles;
    uint32_t                kv_شروعات;
    uint32_t                صفون_في_t;
} fa_v_int_دليلون_t;

ساكن فراغ fa_v_interleave_thread(بنانشان صحيح n, بنانشان صحيح i, فراغ * معلومات) {
    fa_v_int_دليلون_t *       دليلون_2  = (fa_v_int_دليلون_t *) معلومات;
    بنية hmx_fa_context * factx = دليلون_2->factx;

    ثابت uint32_t ڪل_صفون = دليلون_2->kv_صفون;
    ثابت uint32_t صفون_في_t = دليلون_2->صفون_في_t;
    ثابت uint32_t شروعات      = i * صفون_في_t;
    ثابت uint32_t پڇاڙي        = (uint32_t) hex_smin(شروعات + صفون_في_t, ڪل_صفون);

    إذا (شروعات >= ڪل_صفون) {
        أرجع;
    }

    __fp16 * v_tiles_dst = (__fp16 *) دليلون_2->v_tiles_dst;

    بنية htp_thread_trace * tr = &factx->octx->ctx->trace[i];
    htp_trace_event_شروعات(tr, HTP_TRACE_EVT_HVX_FA_V_PREP, (uint16_t) (دليلون_2->kv_شروعات + شروعات));
    إذا (factx->v->type == HTP_TYPE_Q8_0) {
        لكل (uint32_t r = شروعات; r < پڇاڙي; ++r) {
            __fp16 * صف_v = (__fp16 *)((حرف *)دليلون_2->v_src + r * دليلون_2->src_stride * حجم(__fp16));
            hvx_dequantize_صف_flat_q8_0_f16(صف_v, صف_v, factx->DV);
        }
    }
    hmx_interleave_کانن_ڳڻپ_منزل_tiles(v_tiles_dst, (ثابت __fp16 *) دليلون_2->v_src, ڪل_صفون, factx->DV_pad,
                             دليلون_2->src_stride, (uint32_t) دليلون_2->n_کانو_tiles, شروعات, پڇاڙي);
    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_FA_V_PREP, (uint16_t) (دليلون_2->kv_شروعات + شروعات));
}

ساكن فراغ fa_phase_v_interleave(بنية hmx_fa_context * factx,
                                  uint32_t                kv_صفون,
                                  size_t                  src_stride,
                                  فراغ *                  v_src,
                                  فراغ *                  v_tiles_dst,
                                  size_t                  n_کانو_tiles,
                                  uint32_t                kv_شروعات) {
    work_قطار_t wp = factx->octx->ctx->work_قطار;
    uint32_t n = 1;
    إذا (factx->n_ڌاڳا > 1 && kv_صفون >= factx->n_ڌاڳا * 2) {
        n = factx->n_ڌاڳا;
    }
    uint32_t صفون_في_t = hex_align_مٿي(hmx_ceil_div(kv_صفون, n), 2);
    fa_v_int_دليلون_t دليلون_2 = { factx, kv_صفون, src_stride, v_src, v_tiles_dst, n_کانو_tiles, kv_شروعات, صفون_في_t };
    إذا (n > 1) {
        work_قطار_لاڳيتو(wp, fa_v_interleave_thread, &دليلون_2, n);
    } وإلا {
        fa_v_interleave_thread(1, 0, &دليلون_2);
    }
}

تعريفنوع بنية {
    بنية hmx_fa_context *   factx;
    ثابت بنية htp_tensor * q;
    uint32_t                  q_شروعات;
    uint32_t                  kv_سر;
    uint32_t                  ib3;
    size_t                    n_صفون_g;
    size_t                    صفون_في_t;
    size_t                    n_صفون_q;
    bool                      q_transposed;
    atomic_uint               barrier;
} fa_q_ڀريو_دليلون_t;

ساكن فراغ fa_q_ڀريو_thread(بنانشان صحيح n, بنانشان صحيح i, فراغ * معلومات) {
    fa_q_ڀريو_دليلون_t *      دليلون_2  = (fa_q_ڀريو_دليلون_t *) معلومات;
    بنية hmx_fa_context * factx = دليلون_2->factx;

    ثابت size_t n_صفون_g = دليلون_2->n_صفون_g;
    ثابت size_t G        = factx->G;
    ثابت size_t DK       = factx->DK;

    // Partition the padded Q rows (g_br) across threads.
    // Keep start/end even so r and r+1 are always in the same thread's range.
    ثابت size_t صفون_في_t = دليلون_2->صفون_في_t;
    ثابت size_t شروعات      = (size_t) i * صفون_في_t;
    ثابت size_t پڇاڙي        = hex_smin(شروعات + صفون_في_t, factx->g_br);

    بنية htp_thread_trace * tr = &factx->octx->ctx->trace[i];
    htp_trace_event_شروعات(tr, HTP_TRACE_EVT_HVX_FA_Q_PREP, (uint16_t) (دليلون_2->q_شروعات * G + شروعات));

    // Parallel initialization of per-block state
    {
        ثابت uint32_t g_br = factx->g_br;
        ثابت uint32_t DV   = factx->DV;

        ثابت size_t کانو_vec_بائيٽون = factx->کانو_vec_بائيٽون;
        ثابت size_t d_tile_بائيٽون  = factx->d_tile_بائيٽون;

        // Initialize vtcm_l_vec & vtcm_m_vec
        ثابت size_t l_بائيٽون_في_t = hex_align_مٿي(کانو_vec_بائيٽون / n, 128);
        ثابت size_t l_شروعات       = i * l_بائيٽون_في_t;
        ثابت size_t l_پڇاڙي         = hex_smin(l_شروعات + l_بائيٽون_في_t, کانو_vec_بائيٽون);

        ثابت size_t m_بائيٽون_في_t = hex_align_مٿي(کانو_vec_بائيٽون / n, 128);
        ثابت size_t m_شروعات       = i * m_بائيٽون_في_t;
        ثابت size_t m_پڇاڙي         = hex_smin(m_شروعات + m_بائيٽون_في_t, کانو_vec_بائيٽون);

        إذا (l_شروعات < کانو_vec_بائيٽون) {
            hvx_splat_u8_a((حرف *) factx->vtcm_l_vec + l_شروعات, 0, l_پڇاڙي - l_شروعات);
        }
        إذا (m_شروعات < کانو_vec_بائيٽون) {
            hvx_splat_f32_a((حرف *) factx->vtcm_m_vec + m_شروعات, HTP_FA_M_INITIAL_VAL, (m_پڇاڙي - m_شروعات) / حجم(عائم));
        }

        // Zero the whole rescale region: vtcm_d_tiles[0], the optional vtcm_d_tiles[1]
        // and vtcm_d_inv_l are equal-sized and allocated back to back, so one run covers
        // them all.  The scatter only ever writes the diagonal, ignore the rest.
        ثابت size_t d_بائيٽون_في_t = hex_align_مٿي(d_tile_بائيٽون / n, 128);
        ثابت size_t d_شروعات       = i * d_بائيٽون_في_t;
        ثابت size_t d_پڇاڙي         = hex_smin(d_شروعات + d_بائيٽون_في_t, d_tile_بائيٽون);
        إذا (d_شروعات < d_tile_بائيٽون) {
            hvx_splat_u8_a((حرف *) factx->vtcm_d_tiles[0] + d_شروعات, 0, d_پڇاڙي - d_شروعات);
        }
    }

    إذا (شروعات < factx->g_br) {
        ثابت بنية htp_tensor * q       = دليلون_2->q;
        ثابت uint32_t            q_شروعات = دليلون_2->q_شروعات;
        ثابت uint32_t            kv_سر = دليلون_2->kv_سر;
        ثابت uint32_t            ib3     = دليلون_2->ib3;

        ثابت bool use_q_dma = (factx->vtcm_q_dma != عدم);

        __fp16 * q_tiles = factx->vtcm_q_tiles;
        ثابت size_t DK_pad = factx->DK_pad;
        إذا (use_q_dma) {
            ثابت size_t g_صفون_پڇاڙي = hex_smin(پڇاڙي, n_صفون_g);
            ثابت uint32_t d_حد_بندي = factx->is_q_fp32 ? DK / 32 : DK / 64;

            uint8_t * q_flat  = (uint8_t *) factx->vtcm_q_dma;
            إذا (DK_pad != DK) {
                إذا (factx->is_q_fp32) {
                    hmx_fa_q_prep_fp32_pad(q_tiles, q_flat, شروعات, پڇاڙي, g_صفون_پڇاڙي, DK, DK_pad, G, دليلون_2->n_صفون_q, &factx->div_G, دليلون_2->q_transposed);
                } وإلا {
                    hmx_fa_q_prep_fp16_pad(q_tiles, q_flat, شروعات, پڇاڙي, g_صفون_پڇاڙي, DK, DK_pad, G, دليلون_2->n_صفون_q, &factx->div_G, دليلون_2->q_transposed);
                }
            } وإلا إذا (factx->is_q_fp32) {
                بدل (d_حد_بندي) {
                حالة 2:  hmx_fa_q_prep_fp32_d2(q_tiles, q_flat, شروعات, پڇاڙي, g_صفون_پڇاڙي, DK, G, دليلون_2->n_صفون_q, &factx->div_G, دليلون_2->q_transposed); اكسر;
                حالة 4:  hmx_fa_q_prep_fp32_d4(q_tiles, q_flat, شروعات, پڇاڙي, g_صفون_پڇاڙي, DK, G, دليلون_2->n_صفون_q, &factx->div_G, دليلون_2->q_transposed); اكسر;
                افتراضي: hmx_fa_q_prep_fp32(   q_tiles, q_flat, شروعات, پڇاڙي, g_صفون_پڇاڙي, DK, G, دليلون_2->n_صفون_q, &factx->div_G, d_حد_بندي, دليلون_2->q_transposed); اكسر;
                }
            } وإلا {
                بدل (d_حد_بندي) {
                حالة 1:  hmx_fa_q_prep_fp16_d1(q_tiles, q_flat, شروعات, پڇاڙي, g_صفون_پڇاڙي, DK, G, دليلون_2->n_صفون_q, &factx->div_G, دليلون_2->q_transposed); اكسر;
                حالة 2:  hmx_fa_q_prep_fp16_d2(q_tiles, q_flat, شروعات, پڇاڙي, g_صفون_پڇاڙي, DK, G, دليلون_2->n_صفون_q, &factx->div_G, دليلون_2->q_transposed); اكسر;
                افتراضي: hmx_fa_q_prep_fp16(   q_tiles, q_flat, شروعات, پڇاڙي, g_صفون_پڇاڙي, DK, G, دليلون_2->n_صفون_q, &factx->div_G, d_حد_بندي, دليلون_2->q_transposed); اكسر;
                }
            }
        } وإلا {
            // Fallback: direct-from-DDR/L2 path
            hmx_fa_q_prep_fallback(q_tiles, q->معلومات, q->nb[1], q->nb[2], q->nb[3],
                                   q_شروعات, kv_سر, ib3, شروعات, پڇاڙي, n_صفون_g, G, DK, DK_pad, factx->is_q_fp32, &factx->div_G);
        }
    }

    // Synchronize threads before zeroing out vtcm_o_tiles[0] to prevent race condition
    إذا (n > 1) {
        atomic_fetch_sub(&دليلون_2->barrier, 1);
        طالما (atomic_ڀريو(&دليلون_2->barrier) > 0) {
            // spin wait
        }
    }

    // Zero out vtcm_o_tiles[0] as it was used as temp_q_vtcm
    {
        ثابت uint32_t g_br = factx->g_br;
        ثابت uint32_t DV   = factx->DV;
        ثابت size_t o_tile_بائيٽون  = factx->o_tile_بائيٽون;
        ثابت size_t o_بائيٽون_في_t = hex_align_مٿي(o_tile_بائيٽون / n, 128);
        ثابت size_t o_شروعات       = i * o_بائيٽون_في_t;
        ثابت size_t o_پڇاڙي         = hex_smin(o_شروعات + o_بائيٽون_في_t, o_tile_بائيٽون);
        إذا (o_شروعات < o_tile_بائيٽون) {
            hvx_splat_u8_a((حرف *) factx->vtcm_o_tiles[0] + o_شروعات, 0, o_پڇاڙي - o_شروعات);
        }
    }
    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_FA_Q_PREP, (uint16_t) (دليلون_2->q_شروعات * G + شروعات));
}

ساكن فراغ fa_phase_q_ڀريو(بنية hmx_fa_context *   factx,
                            ثابت بنية htp_tensor * q,
                            uint32_t                  q_شروعات,
                            uint32_t                  kv_سر,
                            uint32_t                  ib3,
                            size_t                    n_صفون_g) {
    work_قطار_t wp = factx->octx->ctx->work_قطار;
    uint32_t n = 1;
    إذا (factx->n_ڌاڳا > 1 && n_صفون_g >= (size_t) (factx->n_ڌاڳا * 2)) {
        n = factx->n_ڌاڳا;
    }
    size_t صفون_في_t = hex_align_مٿي(hmx_ceil_div(factx->g_br, n), 2);
    ثابت uint32_t n_صفون_q = hex_smin(factx->Br, factx->neq1 - q_شروعات);
    fa_q_ڀريو_دليلون_t دليلون_2;
    دليلون_2.factx = factx;
    دليلون_2.q = q;
    دليلون_2.q_شروعات = q_شروعات;
    دليلون_2.kv_سر = kv_سر;
    دليلون_2.ib3 = ib3;
    دليلون_2.n_صفون_g = n_صفون_g;
    دليلون_2.صفون_في_t = صفون_في_t;
    دليلون_2.n_صفون_q = n_صفون_q;
    دليلون_2.q_transposed = q->nb[1] < q->nb[2];
    atomic_init(&دليلون_2.barrier, n);
    إذا (n > 1) {
        work_قطار_لاڳيتو(wp, fa_q_ڀريو_thread, &دليلون_2, n);
    } وإلا {
        fa_q_ڀريو_thread(1, 0, &دليلون_2);
    }
}

تعريفنوع بنية {
    بنية hmx_fa_context *   factx;
    ثابت بنية htp_tensor * dst;
    ثابت __fp16 *            o_tile_src;
    uint32_t                  q_شروعات;
    uint32_t                  kv_سر;
    uint32_t                  ib3;
    size_t                    n_صفون_g;
    size_t                    صفون_في_t;
} fa_o_store_دليلون_t;

ساكن فراغ fa_o_store_thread_f32(بنانشان صحيح n, بنانشان صحيح i, فراغ * معلومات) {
    fa_o_store_دليلون_t *     دليلون_2  = (fa_o_store_دليلون_t *) معلومات;
    بنية hmx_fa_context * factx = دليلون_2->factx;

    ثابت size_t n_صفون_g = دليلون_2->n_صفون_g;
    ثابت size_t G        = factx->G;
    ثابت size_t DV       = factx->DV;

    ثابت size_t صفون_في_t = دليلون_2->صفون_في_t;
    ثابت size_t شروعات      = (size_t) i * صفون_في_t;
    ثابت size_t پڇاڙي        = hex_smin(شروعات + صفون_في_t, n_صفون_g);

    إذا (شروعات >= n_صفون_g) {
        أرجع;
    }

    بنية htp_thread_trace * tr = &factx->octx->ctx->trace[i];
    htp_trace_event_شروعات(tr, HTP_TRACE_EVT_HVX_O_PROC, (uint16_t) (دليلون_2->q_شروعات * G + شروعات));

    ثابت بنية htp_tensor * dst        = دليلون_2->dst;
    ثابت __fp16 *            o_tile_src = دليلون_2->o_tile_src;
    ثابت uint32_t            q_شروعات    = دليلون_2->q_شروعات;
    ثابت uint32_t            kv_سر    = دليلون_2->kv_سر;
    ثابت uint32_t            ib3        = دليلون_2->ib3;

    ثابت size_t DV_pad = factx->DV_pad;

    size_t q_idx = fastdiv(شروعات, &factx->div_G);
    size_t h_idx = fastmodulo(شروعات, G, &factx->div_G);

    لكل (size_t r = شروعات; r < پڇاڙي; ++r) {
        عائم * ٻاهريون = (عائم *) ((uint8_t *) dst->معلومات + (kv_سر * G + h_idx) * dst->nb[1] +
                                 (q_شروعات + q_idx) * dst->nb[2] + ib3 * dst->nb[3]);

        size_t         r0            = r / HMX_FP16_TILE_N_صفون;
        size_t         r1            = r % HMX_FP16_TILE_N_صفون;
        ثابت __fp16 * tile_صف_بنياد = o_tile_src + r0 * HMX_FP16_TILE_N_صفون * DV_pad;

        لكل (uint32_t d = 0; d < DV / 32; ++d) {
            ثابت HVX_ويڪٽر * اندريون_tile = (ثابت HVX_ويڪٽر *) (tile_صف_بنياد + d * HMX_FP16_TILE_N_ELMS);
            HVX_ويڪٽر_Pair     vp      = hvx_vec_f16_منزل_f32_shuff(اندريون_tile[r1 / 2]);
            إذا (r1 % 2 == 0) {
                *(HVX_UVector *) (ٻاهريون + d * 32) = Q6_V_lo_W(vp);
            } وإلا {
                *(HVX_UVector *) (ٻاهريون + d * 32) = Q6_V_hi_W(vp);
            }
        }
        // Ragged tail: DV not a multiple of 32 (e.g. 72 -> last 8 lanes). Partial vector-write
        // for the remaining (DV % 32) floats.
        ثابت uint32_t d_پڇ = DV / 32;
        ثابت uint32_t rem    = DV - d_پڇ * 32;
        إذا (rem) {
            ثابت HVX_ويڪٽر * اندريون_tile = (ثابت HVX_ويڪٽر *) (tile_صف_بنياد + d_پڇ * HMX_FP16_TILE_N_ELMS);
            HVX_ويڪٽر_Pair     vp      = hvx_vec_f16_منزل_f32_shuff(اندريون_tile[r1 / 2]);
            HVX_ويڪٽر         vd      = (r1 % 2 == 0) ? Q6_V_lo_W(vp) : Q6_V_hi_W(vp);
            hvx_vec_store_u((فراغ *) (ٻاهريون + d_پڇ * 32), rem * حجم(عائم), vd);
        }

        h_idx++;
        إذا (h_idx == G) {
            h_idx = 0;
            q_idx++;
        }
    }
    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_O_PROC, (uint16_t) (دليلون_2->q_شروعات * G + شروعات));
}

ساكن فراغ fa_o_store_thread_f16(بنانشان صحيح n, بنانشان صحيح i, فراغ * معلومات) {
    fa_o_store_دليلون_t *     دليلون_2  = (fa_o_store_دليلون_t *) معلومات;
    بنية hmx_fa_context * factx = دليلون_2->factx;

    ثابت size_t n_صفون_g   = دليلون_2->n_صفون_g;
    ثابت size_t صفون_في_t = دليلون_2->صفون_في_t;
    ثابت size_t G          = factx->G;
    ثابت size_t DV         = factx->DV;
    ثابت size_t شروعات      = (size_t) i * صفون_في_t;
    ثابت size_t پڇاڙي        = hex_smin(شروعات + صفون_في_t, n_صفون_g);

    إذا (شروعات >= n_صفون_g) {
        أرجع;
    }

    بنية htp_thread_trace * tr = &factx->octx->ctx->trace[i];
    htp_trace_event_شروعات(tr, HTP_TRACE_EVT_HVX_O_PROC, (uint16_t) (دليلون_2->q_شروعات * G + شروعات));

    ثابت بنية htp_tensor * dst        = دليلون_2->dst;
    ثابت __fp16 *            o_tile_src = دليلون_2->o_tile_src;
    ثابت uint32_t            q_شروعات    = دليلون_2->q_شروعات;
    ثابت uint32_t            kv_سر    = دليلون_2->kv_سر;
    ثابت uint32_t            ib3        = دليلون_2->ib3;

    // O-tiles use the padded head dim (DV_pad); dst holds the real DV lanes.
    ثابت size_t DV_pad = factx->DV_pad;

    size_t q_idx = fastdiv(شروعات, &factx->div_G);
    size_t h_idx = fastmodulo(شروعات, G, &factx->div_G);

    لكل (size_t r = شروعات; r < پڇاڙي; ++r) {
        __fp16 * ٻاهريون = (__fp16 *) ((uint8_t *) dst->معلومات + (kv_سر * G + h_idx) * dst->nb[1] +
                                   (q_شروعات + q_idx) * dst->nb[2] + ib3 * dst->nb[3]);

        size_t         r0            = r / HMX_FP16_TILE_N_صفون;
        size_t         r1            = r % HMX_FP16_TILE_N_صفون;
        ثابت __fp16 * tile_صف_بنياد = o_tile_src + r0 * HMX_FP16_TILE_N_صفون * DV_pad;

        لكل (uint32_t d = 0; d < DV / 64; ++d) {
            ثابت __fp16 *     اندريون_dtile = tile_صف_بنياد + d * HMX_FP16_TILE_N_ELMS * 2;
            ثابت HVX_ويڪٽر * pv_in0   = ((ثابت HVX_ويڪٽر *) اندريون_dtile) + r1 / 2;
            ثابت HVX_ويڪٽر * pv_in1   = pv_in0 + 16;
            HVX_ويڪٽر_Pair     vp       = Q6_W_vdeal_VVR(*pv_in1, *pv_in0, -2);
            إذا (r1 % 2 == 0) {
                *(HVX_UVector *) (ٻاهريون + d * 64) = Q6_V_lo_W(vp);
            } وإلا {
                *(HVX_UVector *) (ٻاهريون + d * 64) = Q6_V_hi_W(vp);
            }
        }
        // Ragged tail when DV is not a multiple of 64.
        ثابت uint32_t d_پڇ = DV / 64;
        ثابت uint32_t rem    = DV - d_پڇ * 64;
        إذا (rem) {
            ثابت __fp16 *     اندريون_dtile = tile_صف_بنياد + d_پڇ * HMX_FP16_TILE_N_ELMS * 2;
            ثابت HVX_ويڪٽر * pv_in0   = ((ثابت HVX_ويڪٽر *) اندريون_dtile) + r1 / 2;
            ثابت HVX_ويڪٽر * pv_in1   = pv_in0 + 16;
            HVX_ويڪٽر_Pair     vp       = Q6_W_vdeal_VVR(*pv_in1, *pv_in0, -2);
            HVX_ويڪٽر         vd       = (r1 % 2 == 0) ? Q6_V_lo_W(vp) : Q6_V_hi_W(vp);
            hvx_vec_store_u((فراغ *) (ٻاهريون + d_پڇ * 64), rem * حجم(__fp16), vd);
        }

        h_idx++;
        إذا (h_idx == G) {
            h_idx = 0;
            q_idx++;
        }
    }
    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_O_PROC, (uint16_t) (دليلون_2->q_شروعات * G + شروعات));
}

ساكن فراغ fa_phase_o_store(بنية hmx_fa_context *   factx,
                             ثابت بنية htp_tensor * dst,
                             ثابت __fp16 *            o_tile_src,
                             uint32_t                  q_شروعات,
                             uint32_t                  kv_سر,
                             uint32_t                  ib3,
                             size_t                    n_صفون_g) {
    work_قطار_t wp = factx->octx->ctx->work_قطار;
    uint32_t n = 1;
    إذا (factx->n_ڌاڳا > 1 && n_صفون_g >= (size_t) (factx->n_ڌاڳا * 2)) {
        n = factx->n_ڌاڳا;
    }
    size_t صفون_في_t = hmx_ceil_div(n_صفون_g, n);
    fa_o_store_دليلون_t دليلون_2 = { factx, dst, o_tile_src, q_شروعات, kv_سر, ib3, n_صفون_g, صفون_في_t };
    ڪم_ڪندڙ_callback_t store_fn = factx->is_dst_fp32 ? fa_o_store_thread_f32 : fa_o_store_thread_f16;
    إذا (n > 1) {
        work_قطار_لاڳيتو(wp, store_fn, &دليلون_2, n);
    } وإلا {
        store_fn(1, 0, &دليلون_2);
    }
}

تعريفنوع بنية {
    بنية hmx_fa_context *   factx;
    size_t                    buf_idx;
    size_t                    kv_صفون;
    size_t                    n_صفون_g;
    size_t                    n_کانو_tiles;
    size_t                    n_tiles_في_bc;
    size_t                    n_صف_tiles;
    size_t                    n_صف_tiles_g_br;
    uint32_t                  Bc;
    uint32_t                  G;
    uint32_t                  kv_سر;
    uint32_t                  kv_شروعات;
    uint32_t                  q_شروعات;
    uint32_t                  ib3;
    bool                      has_alibi;  // true when max_bias != 0 (need slope * mask + add)
    __fp16 *                  slopes;
    ثابت بنية htp_tensor * نقاب;
    ثابت __fp16 *            نقاب_vtcm;             // VTCM mask buffer base (NULL = DDR fallback)
    size_t                    نقاب_vtcm_صف_stride;  // elements (__fp16) per row in VTCM mask buffer
    بنية fastdiv_قدرون     thread_div;
} fa_softmax_دليلون_t;

ساكن مضمن فراغ fa_softmax_impl(
    بنانشان صحيح n, بنانشان صحيح i, فراغ * معلومات,
    ثابت bool has_نقاب,
    ثابت bool نقاب_نشر,
    ثابت bool is_g1,
    ثابت bool has_alibi,
    ثابت bool has_softcap
) {
    fa_softmax_دليلون_t *     دليلون_2  = (fa_softmax_دليلون_t *) معلومات;
    بنية hmx_fa_context * factx = دليلون_2->factx;

    ثابت size_t n_صفون_g       = دليلون_2->n_صفون_g;
    ثابت size_t kv_صفون        = دليلون_2->kv_صفون;
    ثابت size_t Bc             = دليلون_2->Bc;
    ثابت size_t G              = دليلون_2->G;
    ثابت size_t n_tiles_في_bc = دليلون_2->n_tiles_في_bc;
    ثابت size_t n_صف_vec_cnt  = hmx_ceil_div(n_صفون_g, 64);
    ثابت uint32_t im3          = has_نقاب ? fastmodulo(دليلون_2->ib3, دليلون_2->نقاب->ne[3], &factx->src3_div3) : 0;

    size_t vec_شروعات = 0;
    size_t vec_پڇاڙي   = n_صف_vec_cnt;
    إذا (n > 1) {
        ثابت size_t vecs_في_t = fastdiv(n_صف_vec_cnt + n - 1, &دليلون_2->thread_div);
        vec_شروعات = i * vecs_في_t;
        vec_پڇاڙي   = hex_smin(vec_شروعات + vecs_في_t, n_صف_vec_cnt);
    }

    إذا (vec_شروعات >= n_صف_vec_cnt) {
        أرجع;
    }

    بنية htp_thread_trace * tr = &factx->octx->ctx->trace[i];
    htp_trace_event_شروعات(tr, HTP_TRACE_EVT_HVX_FA_SFM, (uint16_t) (دليلون_2->q_شروعات * G + vec_شروعات * 64));

    // Per-thread row scratch: thread i uses bufs at offset i * 2 * stride
    ثابت size_t صف_buf_stride = factx->صف_buf_stride;
    HVX_ويڪٽر * my_صف_buf0    = factx->vtcm_صف_bufs + i * 2 * صف_buf_stride;
    HVX_ويڪٽر * my_صف_buf1    = my_صف_buf0 + صف_buf_stride;

    ثابت HVX_ويڪٽر v_neg_inf = Q6_Vh_vsplat_R(0xfbff);

    لكل (size_t r_vec_idx = vec_شروعات; r_vec_idx < vec_پڇاڙي; ++r_vec_idx) {
        HVX_ويڪٽر rowmax_acc_v = v_neg_inf;
        HVX_ويڪٽر rowsum_acc_v = Q6_V_vzero();
        HVX_ويڪٽر m_prev_v0    = factx->vtcm_m_vec[r_vec_idx * 2 + 0];
        HVX_ويڪٽر m_prev_v1    = factx->vtcm_m_vec[r_vec_idx * 2 + 1];

        HVX_ويڪٽر v_slopes = Q6_V_vzero();
        إذا (has_alibi) {
            v_slopes = hvx_vmem(دليلون_2->slopes + r_vec_idx * 64);
        }

        لكل (uint32_t r_vec_off = 0; r_vec_off < 64; r_vec_off += 2) {
            uint32_t r = r_vec_idx * 64 + r_vec_off;
            إذا (r >= hex_align_مٿي(n_صفون_g, 2)) {
                اكسر;
            }

            uint32_t r0 = r / HMX_FP16_TILE_N_صفون;
            uint32_t r1 = r % HMX_FP16_TILE_N_صفون;

            ثابت __fp16 * s_ld_بنياد = factx->vtcm_s_tiles[دليلون_2->buf_idx] + r0 * HMX_FP16_TILE_N_صفون * Bc;
            __fp16 *       p_st_بنياد = factx->vtcm_p_tiles[دليلون_2->buf_idx] + r0 * HMX_FP16_TILE_N_صفون * Bc;

            // Decode 2 rows from S tiles into per-thread row buffers
            إذا (has_softcap) {
                ثابت HVX_ويڪٽر v_cap = hvx_vec_splat_f16(factx->logit_softcap);
                لكل (size_t c = 0; c < kv_صفون; c += 64) {
                    size_t             ci       = c / 64;
                    ثابت __fp16 *     اندريون_dtile = s_ld_بنياد + ci * HMX_FP16_TILE_N_ELMS * 2;
                    ثابت HVX_ويڪٽر * pv_s_in0 = ((ثابت HVX_ويڪٽر *) اندريون_dtile) + r1 / 2;
                    ثابت HVX_ويڪٽر * pv_s_in1 = pv_s_in0 + 16;

                    HVX_ويڪٽر_Pair vp_s_drow = Q6_W_vdeal_VVR(*pv_s_in1, *pv_s_in0, -2);
                    HVX_ويڪٽر     v_s_row0  = Q6_V_lo_W(vp_s_drow);
                    HVX_ويڪٽر     v_s_row1  = Q6_V_hi_W(vp_s_drow);

                    HVX_ويڪٽر t0   = hvx_vec_tanh_f16(v_s_row0);
                    my_صف_buf0[ci] = hvx_vec_mul_f16_f16(t0, v_cap);

                    HVX_ويڪٽر t1   = hvx_vec_tanh_f16(v_s_row1);
                    my_صف_buf1[ci] = hvx_vec_mul_f16_f16(t1, v_cap);
                }
            } وإلا {
                size_t c = 0;
                لكل (; c + 64 < kv_صفون; c += 128) {
                    size_t             ci0       = c / 64;
                    size_t             ci1       = ci0 + 1;
                    ثابت __fp16 *     اندريون_dtile0 = s_ld_بنياد + ci0 * HMX_FP16_TILE_N_ELMS * 2;
                    ثابت __fp16 *     اندريون_dtile1 = s_ld_بنياد + ci1 * HMX_FP16_TILE_N_ELMS * 2;
                    ثابت HVX_ويڪٽر * pv_s_in0_0 = ((ثابت HVX_ويڪٽر *) اندريون_dtile0) + r1 / 2;
                    ثابت HVX_ويڪٽر * pv_s_in1_0 = pv_s_in0_0 + 16;
                    ثابت HVX_ويڪٽر * pv_s_in0_1 = ((ثابت HVX_ويڪٽر *) اندريون_dtile1) + r1 / 2;
                    ثابت HVX_ويڪٽر * pv_s_in1_1 = pv_s_in0_1 + 16;

                    HVX_ويڪٽر_Pair vp_s_drow0 = Q6_W_vdeal_VVR(*pv_s_in1_0, *pv_s_in0_0, -2);
                    my_صف_buf0[ci0]          = Q6_V_lo_W(vp_s_drow0);
                    my_صف_buf1[ci0]          = Q6_V_hi_W(vp_s_drow0);

                    HVX_ويڪٽر_Pair vp_s_drow1 = Q6_W_vdeal_VVR(*pv_s_in1_1, *pv_s_in0_1, -2);
                    my_صف_buf0[ci1]          = Q6_V_lo_W(vp_s_drow1);
                    my_صف_buf1[ci1]          = Q6_V_hi_W(vp_s_drow1);
                }
                لكل (; c < kv_صفون; c += 64) {
                    size_t             ci       = c / 64;
                    ثابت __fp16 *     اندريون_dtile = s_ld_بنياد + ci * HMX_FP16_TILE_N_ELMS * 2;
                    ثابت HVX_ويڪٽر * pv_s_in0 = ((ثابت HVX_ويڪٽر *) اندريون_dtile) + r1 / 2;
                    ثابت HVX_ويڪٽر * pv_s_in1 = pv_s_in0 + 16;

                    HVX_ويڪٽر_Pair vp_s_drow = Q6_W_vdeal_VVR(*pv_s_in1, *pv_s_in0, -2);
                    my_صف_buf0[ci]          = Q6_V_lo_W(vp_s_drow);
                    my_صف_buf1[ci]          = Q6_V_hi_W(vp_s_drow);
                }
            }

            // Apply mask & compute rowmax(S)
            HVX_ويڪٽر v_slope0 = Q6_V_vzero();
            HVX_ويڪٽر v_slope1 = Q6_V_vzero();
            إذا (has_alibi) {
                v_slope0 = hvx_vec_repl_f16(Q6_V_vror_VR(v_slopes, r_vec_off * 2));
                v_slope1 = (r + 1 < n_صفون_g) ? hvx_vec_repl_f16(Q6_V_vror_VR(v_slopes, (r_vec_off + 1) * 2)) : Q6_V_vzero();
            }

            ثابت HVX_ويڪٽر v_حد = Q6_Vh_vsplat_R(0xcc00);  // fp16 -16.0

            HVX_ويڪٽر v_s_rowmax0 = v_neg_inf;
            HVX_ويڪٽر v_s_rowmax1 = v_neg_inf;
            إذا (has_نقاب) {
                لكل (size_t c = 0; c < kv_صفون; c += 64) {
                    size_t         ci          = c / 64;
                    ثابت size_t   ne          = hex_smin(kv_صفون - c, 64);
                    HVX_ويڪٽر_Pred q_پڇ_keep = Q6_Q_vsetq2_R(ne * حجم(__fp16));

                    HVX_ويڪٽر v_mask0, v_mask1;

                    إذا (نقاب_نشر) {
                        إذا (is_g1) {
                            ثابت size_t qi0 = r + 0;
                            v_mask0 = *(ثابت HVX_ويڪٽر *) (دليلون_2->نقاب_vtcm + qi0 * دليلون_2->نقاب_vtcm_صف_stride + c);
                            v_mask1 = v_neg_inf;
                            إذا (r + 1 < n_صفون_g) {
                                ثابت size_t qi1 = r + 1;
                                v_mask1 = *(ثابت HVX_ويڪٽر *) (دليلون_2->نقاب_vtcm + qi1 * دليلون_2->نقاب_vtcm_صف_stride + c);
                            }
                        } وإلا {
                            ثابت size_t qi0 = fastdiv(r + 0, &factx->div_G);
                            v_mask0 = *(ثابت HVX_ويڪٽر *) (دليلون_2->نقاب_vtcm + qi0 * دليلون_2->نقاب_vtcm_صف_stride + c);
                            v_mask1 = v_neg_inf;
                            إذا (r + 1 < n_صفون_g) {
                                ثابت size_t qi1 = fastdiv(r + 1, &factx->div_G);
                                إذا (qi1 == qi0) {
                                    v_mask1 = v_mask0;
                                } وإلا {
                                    v_mask1 = *(ثابت HVX_ويڪٽر *) (دليلون_2->نقاب_vtcm + qi1 * دليلون_2->نقاب_vtcm_صف_stride + c);
                                }
                            }
                        }
                    } وإلا {
                        // Head-dependent mask: pre-interleaved per row r.
                        ثابت size_t r0 = r + 0;
                        v_mask0 = *(ثابت HVX_ويڪٽر *) (دليلون_2->نقاب_vtcm + r0 * دليلون_2->نقاب_vtcm_صف_stride + c);
                        v_mask1 = v_neg_inf;
                        إذا (r + 1 < n_صفون_g) {
                            ثابت size_t r1 = r + 1;
                            v_mask1 = *(ثابت HVX_ويڪٽر *) (دليلون_2->نقاب_vtcm + r1 * دليلون_2->نقاب_vtcm_صف_stride + c);
                        }
                    }

                    // Threshold: mask values below -16.0 are treated as -inf (causal mask).
                    HVX_ويڪٽر_Pred q_keep0 = Q6_Q_and_QQ(Q6_Q_vcmp_gt_VhfVhf(v_mask0, v_حد), q_پڇ_keep);
                    HVX_ويڪٽر_Pred q_keep1 = Q6_Q_and_QQ(Q6_Q_vcmp_gt_VhfVhf(v_mask1, v_حد), q_پڇ_keep);

                    // Scale mask values by log2(e) for base-2 calculations
                    ثابت HVX_ويڪٽر v_log2e = hvx_vec_splat_f16(EXP_LOG2E_F);
                    HVX_ويڪٽر v_mask0_scaled = hvx_vec_mul_f16_f16(v_mask0, v_log2e);
                    HVX_ويڪٽر v_mask1_scaled = hvx_vec_mul_f16_f16(v_mask1, v_log2e);

                    إذا (has_alibi) {
                        HVX_ويڪٽر v_sm0 = hvx_vec_mul_f16_f16(v_mask0_scaled, v_slope0);
                        HVX_ويڪٽر v_sm1 = hvx_vec_mul_f16_f16(v_mask1_scaled, v_slope1);
                        my_صف_buf0[ci]  = Q6_V_vmux_QVV(q_keep0, hvx_vec_جوڙيو_f16_f16(my_صف_buf0[ci], v_sm0), v_neg_inf);
                        my_صف_buf1[ci]  = Q6_V_vmux_QVV(q_keep1, hvx_vec_جوڙيو_f16_f16(my_صف_buf1[ci], v_sm1), v_neg_inf);
                    } وإلا {
                        my_صف_buf0[ci] = Q6_V_vmux_QVV(q_keep0, hvx_vec_جوڙيو_f16_f16(my_صف_buf0[ci], v_mask0_scaled), v_neg_inf);
                        my_صف_buf1[ci] = Q6_V_vmux_QVV(q_keep1, hvx_vec_جوڙيو_f16_f16(my_صف_buf1[ci], v_mask1_scaled), v_neg_inf);
                    }

                    v_s_rowmax0 = Q6_Vhf_vmax_VhfVhf(v_s_rowmax0, my_صف_buf0[ci]);
                    v_s_rowmax1 = Q6_Vhf_vmax_VhfVhf(v_s_rowmax1, my_صف_buf1[ci]);
                }
            } وإلا {
                size_t c = 0;
                لكل (; c + 64 < kv_صفون; c += 128) {
                    size_t ci0 = c / 64;
                    size_t ci1 = ci0 + 1;
                    v_s_rowmax0 = Q6_Vhf_vmax_VhfVhf(v_s_rowmax0, my_صف_buf0[ci0]);
                    v_s_rowmax1 = Q6_Vhf_vmax_VhfVhf(v_s_rowmax1, my_صف_buf1[ci0]);
                    v_s_rowmax0 = Q6_Vhf_vmax_VhfVhf(v_s_rowmax0, my_صف_buf0[ci1]);
                    v_s_rowmax1 = Q6_Vhf_vmax_VhfVhf(v_s_rowmax1, my_صف_buf1[ci1]);
                }
                لكل (; c < kv_صفون; c += 64) {
                    size_t         ci          = c / 64;
                    ثابت size_t   ne          = hex_smin(kv_صفون - c, 64);
                    HVX_ويڪٽر_Pred q_پڇ_keep = Q6_Q_vsetq2_R(ne * حجم(__fp16));
                    إذا (ne < 64) {
                        my_صف_buf0[ci] = Q6_V_vmux_QVV(q_پڇ_keep, my_صف_buf0[ci], v_neg_inf);
                        my_صف_buf1[ci] = Q6_V_vmux_QVV(q_پڇ_keep, my_صف_buf1[ci], v_neg_inf);
                    }
                    v_s_rowmax0 = Q6_Vhf_vmax_VhfVhf(v_s_rowmax0, my_صف_buf0[ci]);
                    v_s_rowmax1 = Q6_Vhf_vmax_VhfVhf(v_s_rowmax1, my_صف_buf1[ci]);
                }
            }

            v_s_rowmax0 = hvx_vec_reduce_max_f16(v_s_rowmax0);
            v_s_rowmax1 = hvx_vec_reduce_max_f16(v_s_rowmax1);

            // Splat m_prev[r], m_prev[r+1] from the float per-row accumulators and convert to fp16 vectors
            HVX_ويڪٽر v_m_prev0, v_m_prev1;
            إذا (r_vec_off < 32) {
                HVX_ويڪٽر v0 = hvx_vec_repl_f32(Q6_V_vror_VR(m_prev_v0, r_vec_off * 4));
                v_m_prev0 = hvx_vec_f32_منزل_f16(v0, v0);
                إذا (r + 1 < n_صفون_g) {
                    HVX_ويڪٽر v1 = hvx_vec_repl_f32(Q6_V_vror_VR(m_prev_v0, (r_vec_off + 1) * 4));
                    v_m_prev1 = hvx_vec_f32_منزل_f16(v1, v1);
                } وإلا {
                    v_m_prev1 = Q6_V_vzero();
                }
            } وإلا {
                HVX_ويڪٽر v0 = hvx_vec_repl_f32(Q6_V_vror_VR(m_prev_v1, (r_vec_off - 32) * 4));
                v_m_prev0 = hvx_vec_f32_منزل_f16(v0, v0);
                إذا (r + 1 < n_صفون_g) {
                    HVX_ويڪٽر v1 = hvx_vec_repl_f32(Q6_V_vror_VR(m_prev_v1, (r_vec_off + 1 - 32) * 4));
                    v_m_prev1 = hvx_vec_f32_منزل_f16(v1, v1);
                } وإلا {
                    v_m_prev1 = Q6_V_vzero();
                }
            }

            HVX_ويڪٽر v_dup_m0 = Q6_Vhf_vmax_VhfVhf(v_m_prev0, v_s_rowmax0);
            HVX_ويڪٽر v_dup_m1 = Q6_Vhf_vmax_VhfVhf(v_m_prev1, v_s_rowmax1);

            // Insert row r, r+1 rowmax into rowmax_acc_v
            {
                HVX_ويڪٽر_Pred p_شروعات = Q6_Q_vsetq_R(r_vec_off * 2);
                HVX_ويڪٽر_Pred p_mid   = Q6_Q_vsetq_R((r_vec_off + 1) * 2);
                HVX_ويڪٽر_Pred p_پڇاڙي   = Q6_Q_vsetq2_R((r_vec_off + 2) * 2);
                HVX_ويڪٽر_Pred p_lane0 = Q6_Q_and_QQn(p_mid, p_شروعات);
                HVX_ويڪٽر_Pred p_lane1 = Q6_Q_and_QQn(p_پڇاڙي, p_mid);
                rowmax_acc_v           = Q6_V_vmux_QVV(p_lane0, v_dup_m0, rowmax_acc_v);
                rowmax_acc_v           = Q6_V_vmux_QVV(p_lane1, v_dup_m1, rowmax_acc_v);
            }

            // Compute P = exp(S - m_new)
            ثابت HVX_ويڪٽر v_zero      = Q6_V_vzero();
            HVX_ويڪٽر       v_p_rowsum0 = v_zero;
            HVX_ويڪٽر       v_p_rowsum1 = v_zero;

            size_t c = 0;
            لكل (; c + 64 < kv_صفون; c += 128) {
                size_t     ci0          = c / 64;
                size_t     ci1          = ci0 + 1;

                HVX_ويڪٽر v_s_minus_m0_0 = Q6_Vqf16_vsub_VhfVhf(my_صف_buf0[ci0], v_dup_m0);
                HVX_ويڪٽر v_s_minus_m1_0 = Q6_Vqf16_vsub_VhfVhf(my_صف_buf1[ci0], v_dup_m1);
                HVX_ويڪٽر v_s_minus_m0_1 = Q6_Vqf16_vsub_VhfVhf(my_صف_buf0[ci1], v_dup_m0);
                HVX_ويڪٽر v_s_minus_m1_1 = Q6_Vqf16_vsub_VhfVhf(my_صف_buf1[ci1], v_dup_m1);

                HVX_ويڪٽر v_p_row0_hf_0  = hvx_vec_exp2_f16(Q6_Vhf_equals_Vqf16(v_s_minus_m0_0));
                HVX_ويڪٽر v_p_row1_hf_0  = hvx_vec_exp2_f16(Q6_Vhf_equals_Vqf16(v_s_minus_m1_0));
                HVX_ويڪٽر v_p_row0_hf_1  = hvx_vec_exp2_f16(Q6_Vhf_equals_Vqf16(v_s_minus_m0_1));
                HVX_ويڪٽر v_p_row1_hf_1  = hvx_vec_exp2_f16(Q6_Vhf_equals_Vqf16(v_s_minus_m1_1));

                __fp16 *     ٻاهريون_dtile0  = p_st_بنياد + ci0 * HMX_FP16_TILE_N_ELMS * 2;
                __fp16 *     ٻاهريون_dtile1  = p_st_بنياد + ci1 * HMX_FP16_TILE_N_ELMS * 2;
                HVX_ويڪٽر * pv_p_out0_0  = ((HVX_ويڪٽر *) ٻاهريون_dtile0) + r1 / 2;
                HVX_ويڪٽر * pv_p_out1_0  = pv_p_out0_0 + 16;
                HVX_ويڪٽر * pv_p_out0_1  = ((HVX_ويڪٽر *) ٻاهريون_dtile1) + r1 / 2;
                HVX_ويڪٽر * pv_p_out1_1  = pv_p_out0_1 + 16;

                HVX_ويڪٽر_Pair vp_p_dual0 = Q6_W_vshuff_VVR(v_p_row1_hf_0, v_p_row0_hf_0, -2);
                *pv_p_out0_0               = Q6_V_lo_W(vp_p_dual0);
                *pv_p_out1_0               = Q6_V_hi_W(vp_p_dual0);

                HVX_ويڪٽر_Pair vp_p_dual1 = Q6_W_vshuff_VVR(v_p_row1_hf_1, v_p_row0_hf_1, -2);
                *pv_p_out0_1               = Q6_V_lo_W(vp_p_dual1);
                *pv_p_out1_1               = Q6_V_hi_W(vp_p_dual1);

                HVX_ويڪٽر_Pair vp_p0_0 = hvx_vec_f16_منزل_f32_shuff(v_p_row0_hf_0);
                HVX_ويڪٽر_Pair vp_p1_0 = hvx_vec_f16_منزل_f32_shuff(v_p_row1_hf_0);
                HVX_ويڪٽر_Pair vp_p0_1 = hvx_vec_f16_منزل_f32_shuff(v_p_row0_hf_1);
                HVX_ويڪٽر_Pair vp_p1_1 = hvx_vec_f16_منزل_f32_shuff(v_p_row1_hf_1);

                v_p_rowsum0 = Q6_Vqf32_vadd_Vqf32Vqf32(v_p_rowsum0, Q6_Vqf32_vadd_VsfVsf(Q6_V_lo_W(vp_p0_0), Q6_V_hi_W(vp_p0_0)));
                v_p_rowsum0 = Q6_Vqf32_vadd_Vqf32Vqf32(v_p_rowsum0, Q6_Vqf32_vadd_VsfVsf(Q6_V_lo_W(vp_p0_1), Q6_V_hi_W(vp_p0_1)));
                v_p_rowsum1 = Q6_Vqf32_vadd_Vqf32Vqf32(v_p_rowsum1, Q6_Vqf32_vadd_VsfVsf(Q6_V_lo_W(vp_p1_0), Q6_V_hi_W(vp_p1_0)));
                v_p_rowsum1 = Q6_Vqf32_vadd_Vqf32Vqf32(v_p_rowsum1, Q6_Vqf32_vadd_VsfVsf(Q6_V_lo_W(vp_p1_1), Q6_V_hi_W(vp_p1_1)));
            }
            لكل (size_t c_rem = c; c_rem < kv_صفون; c_rem += 64) {
                size_t     ci           = c_rem / 64;
                HVX_ويڪٽر v_s_minus_m0 = Q6_Vqf16_vsub_VhfVhf(my_صف_buf0[ci], v_dup_m0);
                HVX_ويڪٽر v_s_minus_m1 = Q6_Vqf16_vsub_VhfVhf(my_صف_buf1[ci], v_dup_m1);

                HVX_ويڪٽر v_p_row0_hf  = hvx_vec_exp2_f16(Q6_Vhf_equals_Vqf16(v_s_minus_m0));
                HVX_ويڪٽر v_p_row1_hf  = hvx_vec_exp2_f16(Q6_Vhf_equals_Vqf16(v_s_minus_m1));
                __fp16 *     ٻاهريون_dtile  = p_st_بنياد + ci * HMX_FP16_TILE_N_ELMS * 2;
                HVX_ويڪٽر * pv_p_out0  = ((HVX_ويڪٽر *) ٻاهريون_dtile) + r1 / 2;
                HVX_ويڪٽر * pv_p_out1  = pv_p_out0 + 16;

                HVX_ويڪٽر_Pair vp_p_dual = Q6_W_vshuff_VVR(v_p_row1_hf, v_p_row0_hf, -2);
                *pv_p_out0               = Q6_V_lo_W(vp_p_dual);
                *pv_p_out1               = Q6_V_hi_W(vp_p_dual);

                HVX_ويڪٽر_Pair vp_p0 = hvx_vec_f16_منزل_f32_shuff(v_p_row0_hf);
                HVX_ويڪٽر_Pair vp_p1 = hvx_vec_f16_منزل_f32_shuff(v_p_row1_hf);

                v_p_rowsum0 = Q6_Vqf32_vadd_Vqf32Vqf32(v_p_rowsum0, Q6_Vqf32_vadd_VsfVsf(Q6_V_lo_W(vp_p0), Q6_V_hi_W(vp_p0)));
                v_p_rowsum1 = Q6_Vqf32_vadd_Vqf32Vqf32(v_p_rowsum1, Q6_Vqf32_vadd_VsfVsf(Q6_V_lo_W(vp_p1), Q6_V_hi_W(vp_p1)));
            }

            HVX_ويڪٽر rowsum0_sf = hvx_vec_reduce_جوڙ_f32(Q6_Vsf_equals_Vqf32(v_p_rowsum0));
            HVX_ويڪٽر rowsum1_sf = hvx_vec_reduce_جوڙ_f32(Q6_Vsf_equals_Vqf32(v_p_rowsum1));
            {
                HVX_ويڪٽر rv0_v = hvx_vec_f32_منزل_f16(rowsum0_sf, rowsum0_sf);
                HVX_ويڪٽر rv1_v = hvx_vec_f32_منزل_f16(rowsum1_sf, rowsum1_sf);

                HVX_ويڪٽر_Pred p_شروعات = Q6_Q_vsetq_R(r_vec_off * 2);
                HVX_ويڪٽر_Pred p_mid   = Q6_Q_vsetq_R((r_vec_off + 1) * 2);
                HVX_ويڪٽر_Pred p_پڇاڙي   = Q6_Q_vsetq2_R((r_vec_off + 2) * 2);
                HVX_ويڪٽر_Pred p_lane0 = Q6_Q_and_QQn(p_mid, p_شروعات);
                HVX_ويڪٽر_Pred p_lane1 = Q6_Q_and_QQn(p_پڇاڙي, p_mid);
                rowsum_acc_v           = Q6_V_vmux_QVV(p_lane0, rv0_v, rowsum_acc_v);
                rowsum_acc_v           = Q6_V_vmux_QVV(p_lane1, rv1_v, rowsum_acc_v);
            }
        }

        // Inline fa_ml_update_and_build_d for this vector (lock-free and in parallel)
        HVX_ويڪٽر_Pair rowmax_acc_pair    = hvx_vec_f16_منزل_f32(rowmax_acc_v);
        HVX_ويڪٽر     v_rowmax_acc_f32_0 = Q6_V_lo_W(rowmax_acc_pair);
        HVX_ويڪٽر     v_rowmax_acc_f32_1 = Q6_V_hi_W(rowmax_acc_pair);

        HVX_ويڪٽر v_m_curr0 = Q6_Vsf_vmax_VsfVsf(m_prev_v0, v_rowmax_acc_f32_0);
        HVX_ويڪٽر v_m_curr1 = Q6_Vsf_vmax_VsfVsf(m_prev_v1, v_rowmax_acc_f32_1);

        HVX_ويڪٽر v_m_diff0 = HVX_OP_SUB_F32(m_prev_v0, v_m_curr0);
        HVX_ويڪٽر v_m_diff1 = HVX_OP_SUB_F32(m_prev_v1, v_m_curr1);

        HVX_ويڪٽر v_m_diff_f16   = hvx_vec_f32_منزل_f16(v_m_diff0, v_m_diff1);
        HVX_ويڪٽر exp_m_diff_f16 = hvx_vec_exp2_f16(v_m_diff_f16);

        HVX_ويڪٽر_Pair exp_m_diff_pair = hvx_vec_f16_منزل_f32(exp_m_diff_f16);
        HVX_ويڪٽر exp_m_diff0 = Q6_V_lo_W(exp_m_diff_pair);
        HVX_ويڪٽر exp_m_diff1 = Q6_V_hi_W(exp_m_diff_pair);

        HVX_ويڪٽر_Pair rowsum_acc_pair = hvx_vec_f16_منزل_f32(rowsum_acc_v);
        HVX_ويڪٽر     v_rowsum_acc_f32_0 = Q6_V_lo_W(rowsum_acc_pair);
        HVX_ويڪٽر     v_rowsum_acc_f32_1 = Q6_V_hi_W(rowsum_acc_pair);

        HVX_ويڪٽر l_prev_v0 = factx->vtcm_l_vec[r_vec_idx * 2 + 0];
        HVX_ويڪٽر l_prev_v1 = factx->vtcm_l_vec[r_vec_idx * 2 + 1];
        HVX_ويڪٽر v_l_curr0 = HVX_OP_جوڙيو_F32(HVX_OP_MUL_F32(l_prev_v0, exp_m_diff0), v_rowsum_acc_f32_0);
        HVX_ويڪٽر v_l_curr1 = HVX_OP_جوڙيو_F32(HVX_OP_MUL_F32(l_prev_v1, exp_m_diff1), v_rowsum_acc_f32_1);

        factx->vtcm_m_vec[r_vec_idx * 2 + 0] = v_m_curr0;
        factx->vtcm_m_vec[r_vec_idx * 2 + 1] = v_m_curr1;
        factx->vtcm_l_vec[r_vec_idx * 2 + 0] = v_l_curr0;
        factx->vtcm_l_vec[r_vec_idx * 2 + 1] = v_l_curr1;

        // Build diagonal tile D = diag(exp(m_diff))
        ثابت HVX_ويڪٽر     v_offsets = *(ثابت HVX_ويڪٽر *) d_tile_scatter_offsets;
        ثابت HVX_ويڪٽر_Pred q_32_نقاب = Q6_Q_vsetq_R(32 * حجم(__fp16));
        HVX_ويڪٽر           v_exp_m_diff = exp_m_diff_f16;

        __fp16 * ثابت d_tiles_ٻاهريون = factx->vtcm_d_tiles[دليلون_2->buf_idx];

        size_t t0 = r_vec_idx * 2;
        إذا (t0 < دليلون_2->n_صف_tiles) {
            ثابت HVX_ويڪٽر v_content = v_exp_m_diff;
            __fp16 *         ٻاهريون_بنياد  = d_tiles_ٻاهريون + t0 * HMX_FP16_TILE_N_ELMS;
            Q6_vscatter_QRMVhV(q_32_نقاب, (size_t) ٻاهريون_بنياد, HMX_FP16_TILE_سائيز - 1, v_offsets, v_content);
        }

        size_t t1 = r_vec_idx * 2 + 1;
        إذا (t1 < دليلون_2->n_صف_tiles) {
            ثابت HVX_ويڪٽر v_content = Q6_V_vror_VR(v_exp_m_diff, 64);
            __fp16 *         ٻاهريون_بنياد  = d_tiles_ٻاهريون + t1 * HMX_FP16_TILE_N_ELMS;
            Q6_vscatter_QRMVhV(q_32_نقاب, (size_t) ٻاهريون_بنياد, HMX_FP16_TILE_سائيز - 1, v_offsets, v_content);
        }
    }
    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_FA_SFM, (uint16_t) (دليلون_2->q_شروعات * G + vec_شروعات * 64));
}

ساكن فراغ fa_softmax_thread_nomask(بنانشان صحيح n, بنانشان صحيح i, فراغ * معلومات) {
    fa_softmax_impl(n, i, معلومات,
                    /*has_mask=*/false,
                    /*mask_broadcast=*/false,
                    /*is_g1=*/false,
                    /*has_alibi=*/false,
                    /*has_softcap=*/false);
}

ساكن فراغ fa_softmax_thread_نقاب_نشر_g1(بنانشان صحيح n, بنانشان صحيح i, فراغ * معلومات) {
    fa_softmax_impl(n, i, معلومات,
                    /*has_mask=*/true,
                    /*mask_broadcast=*/true,
                    /*is_g1=*/true,
                    /*has_alibi=*/false,
                    /*has_softcap=*/false);
}

ساكن فراغ fa_softmax_thread_نقاب_نشر_gn(بنانشان صحيح n, بنانشان صحيح i, فراغ * معلومات) {
    fa_softmax_impl(n, i, معلومات,
                    /*has_mask=*/true,
                    /*mask_broadcast=*/true,
                    /*is_g1=*/false,
                    /*has_alibi=*/false,
                    /*has_softcap=*/false);
}

ساكن فراغ fa_softmax_thread(بنانشان صحيح n, بنانشان صحيح i, فراغ * معلومات) {
    fa_softmax_دليلون_t *     دليلون_2  = (fa_softmax_دليلون_t *) معلومات;
    بنية hmx_fa_context * factx = دليلون_2->factx;

    ثابت bool has_نقاب       = (دليلون_2->نقاب != عدم);
    ثابت bool نقاب_نشر = factx->نقاب_نشر;
    ثابت bool is_g1          = (دليلون_2->G == 1);
    ثابت bool has_alibi      = دليلون_2->has_alibi;
    ثابت bool has_softcap    = factx->has_softcap;

    fa_softmax_impl(n, i, معلومات, has_نقاب, نقاب_نشر, is_g1, has_alibi, has_softcap);
}

ساكن مضمن HVX_ويڪٽر hvx_fa_ڀريو_sinks(ثابت عائم * sinks_معلومات, size_t idx, size_t ڳڻپ) {
    ثابت HVX_ويڪٽر * v_src    = (ثابت HVX_ويڪٽر *) sinks_معلومات + (idx >> 5);
    ثابت size_t       lane_off = idx & 31;
    ثابت size_t       byte_off = lane_off * حجم(عائم);
    أرجع (lane_off + ڳڻپ > 32)
        ? Q6_V_valign_VVR(v_src[1], v_src[0], byte_off)
        : Q6_V_vror_VR(v_src[0], byte_off);
}

ساكن __attribute__((noinline)) فراغ fa_build_d_diag_inv_l(بنية hmx_fa_context * factx,
                                                            size_t                  n_صف_tiles,
                                                            size_t                  n_صف_tiles_g_br,
                                                            uint32_t                kv_سر,
                                                            size_t                  n_صفون_g) {
    (فراغ) n_صف_tiles_g_br;

    // sinks: l += exp(sink - m * ln2) in f32, m and l are in the log2 domain
    إذا (factx->sinks) {
        ثابت عائم *    sinks_معلومات   = factx->vtcm_sinks;
        ثابت size_t     G            = factx->G;
        ثابت size_t     سر_بنياد    = (size_t) kv_سر * G;
        ثابت HVX_ويڪٽر v_ln2        = Q6_V_vsplat_R(EXP_LOGN2);
        ثابت HVX_ويڪٽر v_سر_sinks = (G <= 32) ? hvx_fa_ڀريو_sinks(sinks_معلومات, سر_بنياد, G) : Q6_V_vzero();
        HVX_ويڪٽر       v_sink       = Q6_V_vzero();
        size_t           g_prev       = SIZE_MAX;

        لكل (size_t i = 0; i < n_صف_tiles; ++i) {
            // the sink pattern of a tile only depends on (i * 32) % G, rebuild it when that changes
            ثابت size_t r0 = i * 32;
            ثابت size_t g0 = fastmodulo(r0, G, &factx->div_G);
            إذا (g0 != g_prev) {
                ثابت size_t rem_g = G - g0;
                إذا (G <= 32) {
                    ثابت HVX_ويڪٽر v_lo = Q6_V_vror_VR(v_سر_sinks, g0 * حجم(عائم));
                    ثابت HVX_ويڪٽر v_hi = Q6_V_vror_VR(v_سر_sinks, 128 - rem_g * حجم(عائم));
                    v_sink = Q6_V_vmux_QVV(Q6_Q_vsetq2_R(rem_g * حجم(عائم)), v_lo, v_hi);
                    لكل (size_t len = G; len < 32; len <<= 1) {
                        ثابت HVX_ويڪٽر_Pred q_keep    = Q6_Q_vsetq_R(len * حجم(عائم));
                        ثابت HVX_ويڪٽر     v_shifted = Q6_V_vror_VR(v_sink, 128 - len * حجم(عائم));
                        v_sink = Q6_V_vmux_QVV(q_keep, v_sink, v_shifted);
                    }
                } وإلا إذا (rem_g >= 32) {
                    v_sink = hvx_fa_ڀريو_sinks(sinks_معلومات, سر_بنياد + g0, 32);
                } وإلا {
                    ثابت HVX_ويڪٽر v_lo = hvx_fa_ڀريو_sinks(sinks_معلومات, سر_بنياد + g0, rem_g);
                    ثابت HVX_ويڪٽر v_hi = hvx_fa_ڀريو_sinks(sinks_معلومات, سر_بنياد, 32 - rem_g);
                    v_sink = Q6_V_vmux_QVV(Q6_Q_vsetq_R(rem_g * حجم(عائم)), v_lo, Q6_V_vror_VR(v_hi, 128 - rem_g * حجم(عائم)));
                }
                g_prev = g0;
            }

            HVX_ويڪٽر v_d = HVX_OP_SUB_F32(v_sink, HVX_OP_MUL_F32(factx->vtcm_m_vec[i], v_ln2));
            إذا (r0 + 32 > n_صفون_g) {
                ثابت HVX_ويڪٽر_Pred q_valid = Q6_Q_vsetq_R((n_صفون_g - r0) * حجم(عائم));
                v_d = Q6_V_vmux_QVV(q_valid, v_d, Q6_V_vzero());
            }
            factx->vtcm_l_vec[i] = HVX_OP_جوڙيو_F32(factx->vtcm_l_vec[i], hvx_vec_exp_f32(v_d));
        }
    }

    ثابت HVX_ويڪٽر     v_offsets = *(ثابت HVX_ويڪٽر *) d_tile_scatter_offsets;
    ثابت HVX_ويڪٽر_Pred q_32_نقاب = Q6_Q_vsetq_R(32 * حجم(__fp16));
    ثابت HVX_ويڪٽر     one       = hvx_vec_splat_f32(1.0f);

    HVX_ويڪٽر v_content = Q6_V_vzero();
    لكل (size_t i = 0; i < n_صف_tiles; ++i) {
        إذا ((i % 2) == 0) {
            HVX_ويڪٽر inv_lo = HVX_OP_MUL_F32(one, hvx_vec_inverse_f32(factx->vtcm_l_vec[i]));
            HVX_ويڪٽر inv_hi = (i + 1 < n_صف_tiles) ? HVX_OP_MUL_F32(one, hvx_vec_inverse_f32(factx->vtcm_l_vec[i + 1])) : Q6_V_vzero();
            v_content = hvx_vec_f32_منزل_f16(inv_lo, inv_hi);
        } وإلا {
            v_content = Q6_V_vror_VR(v_content, 64);
        }

        __fp16 * ٻاهريون_بنياد = factx->vtcm_d_inv_l + i * HMX_FP16_TILE_N_ELMS;
        Q6_vscatter_QRMVhV(q_32_نقاب, (size_t) ٻاهريون_بنياد, HMX_FP16_TILE_سائيز - 1, v_offsets, v_content);
    }
}

ساكن فراغ fa_phase_softmax_and_build_d(بنية hmx_fa_context * factx,
                                         fa_softmax_دليلون_t *     sargs,
                                         size_t                  n_صف_tiles,
                                         size_t                  n_صف_tiles_g_br) {
    work_قطار_t wp = factx->octx->ctx->work_قطار;
    ثابت size_t n_صف_vec_cnt = hmx_ceil_div(sargs->n_صفون_g, 64);

    ڪم_ڪندڙ_callback_t softmax_fn = fa_softmax_thread;
    إذا (sargs->نقاب == عدم && !factx->has_softcap && !sargs->has_alibi) {
        softmax_fn = fa_softmax_thread_nomask;
    } وإلا إذا (sargs->نقاب != عدم && factx->نقاب_نشر && !factx->has_softcap && !sargs->has_alibi) {
        إذا (sargs->G == 1) {
            softmax_fn = fa_softmax_thread_نقاب_نشر_g1;
        } وإلا {
            softmax_fn = fa_softmax_thread_نقاب_نشر_gn;
        }
    }

    إذا (factx->n_ڌاڳا > 1 && n_صف_vec_cnt >= 2) {
        uint32_t n_use = (uint32_t) hex_smin((size_t) factx->n_ڌاڳا, n_صف_vec_cnt);
        sargs->thread_div = init_fastdiv_قدرون(n_use);
        work_قطار_لاڳيتو(wp, softmax_fn, sargs, n_use);
    } وإلا {
        softmax_fn(1, 0, sargs);
    }
}

// ============================================================================
// HMX job structs and worker functions
// ============================================================================

تعريفنوع بنية {
    ثابت __fp16 * q_tiles;
    ثابت __fp16 * k_tiles;
    __fp16 *       s_tiles;
    size_t         n_صف_tiles;
    size_t         n_کانو_tiles;
    size_t         n_dot_tiles;  // DK / 32
    size_t         n_tiles_في_bc;
    uint8_t *      hmx_scales;
} hmx_fa_qk_ڪم_t;

ساكن فراغ hmx_fa_qk_dot_ڪم_ڪندڙ(فراغ * معلومات) {
    hmx_fa_qk_ڪم_t * ڪم            = (hmx_fa_qk_ڪم_t *) معلومات;
    ثابت size_t      n_صف_tiles    = ڪم->n_صف_tiles;
    ثابت size_t      n_کانو_tiles    = ڪم->n_کانو_tiles;
    ثابت size_t      n_dot_tiles    = ڪم->n_dot_tiles;
    ثابت size_t      n_tiles_في_bc = ڪم->n_tiles_في_bc;
    ثابت __fp16 * محدود q_tiles  = ڪم->q_tiles;
    ثابت __fp16 * محدود k_tiles  = ڪم->k_tiles;
    __fp16 * محدود s_tiles        = ڪم->s_tiles;
    __builtin_assume(n_صف_tiles > 0);
    __builtin_assume(n_کانو_tiles > 0);
    __builtin_assume(n_dot_tiles > 0);

    asm متطاير(HMX_سيٽ_BIAS("%0") :: "r"((بنانشان صحيح)ڪم->hmx_scales));
    ثابت size_t dot_stride = n_dot_tiles * HMX_FP16_TILE_N_ELMS;
    لكل (size_t r = 0; r < n_صف_tiles; ++r) {
        ثابت __fp16 * صف_tiles = q_tiles + r * dot_stride;
        ثابت __fp16 * کانو_tiles = k_tiles;
        __fp16 *       ٻاهريون_tile  = s_tiles + r * n_tiles_في_bc * HMX_FP16_TILE_N_ELMS;

        لكل (size_t c = 0; c < n_کانو_tiles; ++c) {
            hmx_fa_qk_dot_tile(صف_tiles, کانو_tiles, ٻاهريون_tile, n_dot_tiles);
            کانو_tiles += dot_stride;
            ٻاهريون_tile  += HMX_FP16_TILE_N_ELMS;
        }
    }
}

تعريفنوع بنية {
    __fp16 *       o_curr;
    ثابت __fp16 * o_prev;
    ثابت __fp16 * p_tiles;
    ثابت __fp16 * v_tiles;
    ثابت __fp16 * d_tiles;
    uint8_t *      hmx_scales;
    size_t         n_صف_tiles;
    size_t         n_کانو_tiles;
    size_t         n_صف_tiles_g_br;
    size_t         n_tiles_في_bc;
    size_t         DV;
} hmx_fa_o_update_ڪم_t;

ساكن فراغ hmx_fa_o_update_ڪم_ڪندڙ(فراغ * معلومات) {
    hmx_fa_o_update_ڪم_t * ڪم              = (hmx_fa_o_update_ڪم_t *) معلومات;
    ثابت size_t            n_صف_tiles      = ڪم->n_صف_tiles;
    ثابت size_t            n_کانو_tiles      = ڪم->n_کانو_tiles;
    ثابت size_t            n_صف_tiles_g_br = ڪم->n_صف_tiles_g_br;
    ثابت size_t            n_tiles_في_bc   = ڪم->n_tiles_في_bc;
    ثابت size_t            DV_tiles         = ڪم->DV / 32;
    ثابت __fp16 * محدود d_tiles          = ڪم->d_tiles;
    ثابت __fp16 * محدود p_tiles          = ڪم->p_tiles;
    ثابت __fp16 * محدود v_tiles          = ڪم->v_tiles;
    ثابت __fp16 * محدود o_prev           = ڪم->o_prev;
    __fp16 * محدود o_curr                 = ڪم->o_curr;
    __builtin_assume(n_صف_tiles > 0);
    __builtin_assume(n_کانو_tiles > 0);
    __builtin_assume(DV_tiles > 0);

    asm متطاير(HMX_سيٽ_BIAS("%0") :: "r"((بنانشان صحيح)ڪم->hmx_scales));
    ثابت size_t o_stride = n_صف_tiles_g_br * HMX_FP16_TILE_N_ELMS;
    ثابت size_t v_stride = n_tiles_في_bc * HMX_FP16_TILE_N_ELMS;
    لكل (size_t r = 0; r < n_صف_tiles; ++r) {
        ثابت __fp16 * d_diag     = d_tiles + r * HMX_FP16_TILE_N_ELMS;
        ثابت __fp16 * p_tile_اندريون  = p_tiles + (r * n_tiles_في_bc) * HMX_FP16_TILE_N_ELMS;
        ثابت __fp16 * o_rc       = o_prev + r * HMX_FP16_TILE_N_ELMS;
        ثابت __fp16 * v_tile_اندريون  = v_tiles;
        __fp16       * o_tile_ٻاهريون = o_curr + r * HMX_FP16_TILE_N_ELMS;

        لكل (size_t c = 0; c < DV_tiles; ++c) {
            hmx_fa_o_update_tile(d_diag, o_rc, p_tile_اندريون, v_tile_اندريون, o_tile_ٻاهريون, n_کانو_tiles);
            o_rc       += o_stride;
            v_tile_اندريون  += v_stride;
            o_tile_ٻاهريون += o_stride;
        }
    }
}

تعريفنوع بنية {
    __fp16 *       o_curr;   // output (row-major tile layout)
    ثابت __fp16 * o_prev;   // input (column-major tile layout)
    ثابت __fp16 * d_tiles;  // diag(1/l) tiles
    uint8_t *      hmx_scales;
    size_t         n_صف_tiles;
    size_t         n_صف_tiles_g_br;
    size_t         DV;
} hmx_fa_o_norm_ڪم_t;

ساكن فراغ hmx_fa_o_norm_ڪم_ڪندڙ(فراغ * معلومات) {
    hmx_fa_o_norm_ڪم_t * ڪم              = (hmx_fa_o_norm_ڪم_t *) معلومات;
    ثابت size_t          n_صف_tiles      = ڪم->n_صف_tiles;
    ثابت size_t          n_صف_tiles_g_br = ڪم->n_صف_tiles_g_br;
    ثابت size_t          DV_tiles         = ڪم->DV / 32;
    ثابت __fp16 * محدود d_tiles        = ڪم->d_tiles;
    ثابت __fp16 * محدود o_prev         = ڪم->o_prev;
    __fp16 * محدود o_curr               = ڪم->o_curr;
    __builtin_assume(n_صف_tiles > 0);
    __builtin_assume(DV_tiles > 0);

    asm متطاير(HMX_سيٽ_BIAS("%0") :: "r"((بنانشان صحيح)ڪم->hmx_scales));
    ثابت size_t o_stride = n_صف_tiles_g_br * HMX_FP16_TILE_N_ELMS;
    لكل (size_t r = 0; r < n_صف_tiles; ++r) {
        ثابت __fp16 * d_diag = d_tiles + r * HMX_FP16_TILE_N_ELMS;
        ثابت __fp16 * o_rc = o_prev + r * HMX_FP16_TILE_N_ELMS;
        __fp16 *       o_ٻاهريون = o_curr + r * DV_tiles * HMX_FP16_TILE_N_ELMS;

        لكل (size_t c = 0; c < DV_tiles; ++c) {
            hmx_fa_o_norm_tile(d_diag, o_rc, o_ٻاهريون);
            o_rc  += o_stride;
            o_ٻاهريون += HMX_FP16_TILE_N_ELMS;
        }
    }
}

// Populate per-GQA-row ALiBi slopes for a given KV head.
ساكن __attribute__((noinline)) فراغ fa_compute_slopes(
                              ثابت بنية hmx_fa_context * factx,
                              uint32_t                      kv_سر,
                              size_t                        n_صفون_g) {
    __fp16 * slopes = factx->vtcm_slopes;
    إذا (factx->max_bias == 0.0f) {
        hvx_splat_f16_a(slopes, 1.0f, n_صفون_g);
        أرجع;
    }

    ثابت uint32_t G           = factx->G;
    ثابت uint32_t n_سر_log2 = factx->n_سر_log2;
    ثابت عائم    m0          = factx->m0;
    ثابت عائم    m1          = factx->m1;

    __fp16 عارضي_slopes[512] __attribute__((aligned(128)));
    إذا (G <= 32) {
        // Fast path: Compute G unique slope values in vector registers
        HVX_ويڪٽر v_val = hvx_alibi_slopes(kv_سر, G, n_سر_log2, m0, m1);

        __fp16 عارضي_slopes_aligned[64] __attribute__((aligned(128)));
        hvx_vmem(عارضي_slopes_aligned) = hvx_vec_f32_منزل_f16(v_val, Q6_V_vzero());

        لكل (uint32_t i = 0; i < G; ++i) {
            عارضي_slopes[i] = عارضي_slopes_aligned[i];
        }
    } وإلا {
        // Fallback path: G > 32 (rare configurations)
        لكل (uint32_t i = 0; i < G; ++i) {
            عارضي_slopes[i] = (__fp16)alibi_لاهي(kv_سر * G + i, n_سر_log2, m0, m1);
        }
    }

    // Allocate stack buffer to avoid scalar writes to VTCM (which generates L2 misses)
    __fp16 local_slopes[n_صفون_g] __attribute__((aligned(128)));
    لكل (size_t r = 0; r < n_صفون_g; ++r) {
        local_slopes[r] = عارضي_slopes[fastmodulo(r, G, &factx->div_G)];
    }

    // Copy to VTCM slopes using HVX block copy (both are aligned to 128 bytes)
    hvx_copy_f16_aa((uint8_t *)slopes, (ثابت uint8_t *)local_slopes, n_صفون_g);
}

ساكن فراغ fa_ڌڪو_نقاب_dma_gqa(
    dma_قطار *               dma_q,
    ثابت بنية htp_tensor * نقاب,
    uint32_t                  q_شروعات,
    uint32_t                  im3,
    uint32_t                  kv_شروعات,
    uint32_t                  kv_سر,
    uint32_t                  G,
    uint32_t                  m_سٽ_بائيٽون,
    uint32_t                  kv_صفون,
    uint32_t                  n_صفون_q,
    بنية hmx_fa_context *   factx
) {
    لكل (uint32_t g = 0; g < G; ++g) {
        ثابت uint32_t h_idx = kv_سر * G + g;
        ثابت uint32_t im2 = fastmodulo(h_idx, نقاب->ne[2], &factx->src3_div2);
        ثابت dma_addr_t ms_src = نقاب->معلومات + q_شروعات * نقاب->nb[1] +
                                  im2 * نقاب->nb[2] + im3 * نقاب->nb[3] + kv_شروعات * حجم(__fp16);
        uint8_t * ms_dst = (uint8_t *) factx->vtcm_نقاب_buf + g * m_سٽ_بائيٽون;
        dma_قطار_ڌڪو(dma_q, dma_make_معلومات(ms_dst, ms_src), G * m_سٽ_بائيٽون, نقاب->nb[1], kv_صفون * حجم(__fp16), n_صفون_q);
    }
}

ساكن فراغ fa_ڪڍو_نقاب_dma_gqa(dma_قطار * dma_q, uint32_t G) {
    لكل (uint32_t g = 0; g < G; ++g) {
        dma_قطار_ڪڍو(dma_q);
    }
}

ساكن مضمن فراغ fa_prefetch_بلاڪ(dma_قطار * dma_q, ثابت بنية htp_tensor * k, ثابت بنية htp_tensor * v, ثابت بنية htp_tensor * نقاب,
                                     uint32_t b, size_t Bc, size_t سائيز_k_صف_padded, size_t سائيز_k_صف, size_t سائيز_v_صف_padded, size_t سائيز_v_صف,
                                     uint32_t ik2, uint32_t ik3, uint32_t iv2, uint32_t iv3, uint32_t q_شروعات, uint32_t im3, uint32_t kv_سر, uint32_t G,
                                     size_t m_سٽ_بائيٽون, size_t n_صفون_q, size_t nek1, size_t prefetch_buf, بنية hmx_fa_context * factx) {
    ثابت uint32_t  prefetch_شروعات = b * Bc;
    ثابت uint32_t  prefetch_صفون  = hex_smin(Bc, nek1 - prefetch_شروعات);
    ثابت dma_addr_t k_prefetch_src = k->معلومات + prefetch_شروعات * k->nb[1] + ik2 * k->nb[2] + ik3 * k->nb[3];
    dma_قطار_ڌڪو(dma_q, dma_make_معلومات(factx->vtcm_k_fp16[prefetch_buf], k_prefetch_src), سائيز_k_صف_padded, k->nb[1], سائيز_k_صف, prefetch_صفون);
    ثابت dma_addr_t v_prefetch_src = v->معلومات + prefetch_شروعات * v->nb[1] + iv2 * v->nb[2] + iv3 * v->nb[3];
    dma_قطار_ڌڪو(dma_q, dma_make_معلومات(factx->vtcm_v_fp16[prefetch_buf], v_prefetch_src), سائيز_v_صف_padded, v->nb[1], سائيز_v_صف, prefetch_صفون);

    إذا (نقاب) {
        إذا (__builtin_expect(factx->نقاب_نشر, true)) {
            ثابت dma_addr_t ms_src = نقاب->معلومات + q_شروعات * نقاب->nb[1] + im3 * نقاب->nb[3] + prefetch_شروعات * حجم(__fp16);
            dma_cache_fa_ڌڪو(dma_q, &factx->m_cache, ms_src, m_سٽ_بائيٽون, نقاب->nb[1], prefetch_صفون * حجم(__fp16), n_صفون_q);
        } وإلا {
            fa_ڌڪو_نقاب_dma_gqa(dma_q, نقاب, q_شروعات, im3, prefetch_شروعات, kv_سر, G, m_سٽ_بائيٽون, prefetch_صفون, n_صفون_q, factx);
        }
    }
}

// ============================================================================
// Core HMX flash attention algorithm (GQA-merged)
// ============================================================================

صحيح hmx_flash_attn_ext(بنية htp_ops_context * octx) {
    بنية htp_thread_trace * tr_hvx = &octx->ctx->trace[0];
    بنية htp_thread_trace * tr_hmx = &octx->ctx->trace[HTP_MAX_NTHREADS];
    ثابت بنية htp_tensor * q    = octx->src[0];
    ثابت بنية htp_tensor * k    = octx->src[1];
    ثابت بنية htp_tensor * v    = octx->src[2];
    ثابت بنية htp_tensor * نقاب = (octx->src[3] && octx->src[3]->معلومات) ? octx->src[3] : عدم;
    ثابت بنية htp_tensor * dst  = octx->dst;

    بنية htp_context * ثابت ctx = octx->ctx;

    إذا (!ctx->hmx_enabled) {
        أرجع HTP_حالت_NO_SUPPORT;
    }

    // Dimensions
    ثابت uint32_t neq0 = q->ne[0];  // head_dim (DK)
    ثابت uint32_t neq1 = q->ne[1];  // n_tokens
    ثابت uint32_t neq2 = q->ne[2];  // n_heads
    ثابت uint32_t neq3 = q->ne[3];  // n_seqs

    ثابت uint32_t nek0 = k->ne[0];  // head_dim
    ثابت uint32_t nek1 = k->ne[1];  // kv_len

    ثابت uint32_t nev0 = v->ne[0];  // head_dim (DV)

    ثابت uint32_t DK = neq0;
    ثابت uint32_t DV = nev0;

    // HMX tiles head_dim in units of 64. head_dim need not be 64- (or 32-) aligned:
    // we can operate on DK/DV rounded up to 64 with tail lanes [D, D_pad) zero-filled.
    ثابت uint32_t DK_pad = hex_ڦيرو_مٿي(DK, 64);
    ثابت uint32_t DV_pad = hex_ڦيرو_مٿي(DV, 64);
    إذا (DK == 0 || DV == 0) {
        أرجع HTP_حالت_NO_SUPPORT;
    }

    ثابت بنية htp_fa_kernel_params * kparams = (ثابت بنية htp_fa_kernel_params *) octx->kernel_params;
    ثابت uint32_t n_kv_heads = k->ne[2];

    // ======== Build context ========
    بنية hmx_fa_context factx;
    memset(&factx, 0, حجم(factx));
    factx.octx           = octx;
    factx.sinks          = octx->src[4];  // NULL if this op has no attention sinks
    factx.k              = k;
    factx.v              = v;
    factx.n_ڌاڳا      = kparams->n_ڌاڳا;
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
    factx.n_kv_بلاڪون    = kparams->n_kv_بلاڪون;
    factx.is_q_fp32      = (q->type == HTP_TYPE_F32);
    factx.is_dst_fp32    = (dst->type == HTP_TYPE_F32);
    factx.pipeline       = (kparams->u.hmx.pipeline != 0);
    factx.نقاب_نشر = (kparams->u.hmx.نقاب_نشر != 0);
    إذا (نقاب) {
        factx.src3_div2  = kparams->src3_div2;
        factx.src3_div3  = kparams->src3_div3;
    }

    factx.has_softcap   = (kparams->logit_softcap != 0.0f);
    factx.scale         = (__fp16) kparams->scale;
    factx.max_bias      = kparams->max_bias;
    factx.logit_softcap = 0;
    إذا (factx.has_softcap) {
        factx.logit_softcap = (__fp16) kparams->logit_softcap;
    }

    factx.n_سر_log2 = kparams->n_سر_log2;
    factx.m0          = kparams->m0;
    factx.m1          = kparams->m1;

    ثابت uint32_t Br = factx.Br;
    ثابت uint32_t Bc = factx.Bc;
    ثابت uint32_t g_br = factx.g_br;
    ثابت bool pipeline = factx.pipeline;
    ثابت uint32_t n_ڌاڳا = factx.n_ڌاڳا;
    ثابت uint32_t G = factx.G;

    // Multi-device: prefer head-parallel partitioning (each core owns a disjoint head
    // shard), falling back to Q-block (token) split when heads don't divide evenly.
    ثابت uint32_t n_q_بلاڪون = (neq1 + Br - 1) / Br;
    uint32_t q_شروعات_min  = 0;
    uint32_t q_شروعات_max  = neq1;
    uint32_t kv_سر_min  = 0;
    uint32_t kv_سر_max  = n_kv_heads;

    إذا (octx->ctx->mdev.ڳڻپ > 1) {
        ثابت uint32_t mdev_ڳڻپ = octx->ctx->mdev.ڳڻپ;
        ثابت uint32_t mdev_idx   = octx->ctx->mdev.idx;
        ثابت uint32_t dst_e_سائيز = (dst->type == HTP_TYPE_F32) ? حجم(عائم) : حجم(__fp16);
        ثابت bool can_ورهايو      = htp_tensor_can_صف_حصا_ڪريو(dst, dst_e_سائيز);

        إذا (kparams->سر_ورهايو && can_ورهايو && n_kv_heads >= mdev_ڳڻپ && n_kv_heads % mdev_ڳڻپ == 0) {
            ثابت uint32_t kv_في_core = n_kv_heads / mdev_ڳڻپ;
            kv_سر_min = mdev_idx * kv_في_core;
            kv_سر_max = kv_سر_min + kv_في_core;
        } وإلا {
            ثابت بنية htp_tensor_mdev_range range = htp_tensor_mdev_حصا_ڪريو(n_q_بلاڪون, can_ورهايو ? 1 : 0, mdev_idx, mdev_ڳڻپ, &octx->ctx->mdev.ڳڻپ_div);
            ثابت uint32_t بلاڪ_شروعات = range.شروعات;
            ثابت uint32_t بلاڪ_پڇاڙي   = range.شروعات + range.ڳڻپ;

            إذا (بلاڪ_شروعات >= بلاڪ_پڇاڙي) {
                أرجع HTP_حالت_OK;
            }

            q_شروعات_min = بلاڪ_شروعات * Br;
            q_شروعات_max = MIN(بلاڪ_پڇاڙي * Br, neq1);
        }
    }

    // ======== VTCM allocation (GQA-aware) ========
    // K/V row sizes drive the DMA descriptors (not the VTCM layout) and are used
    // throughout the KV loop below. The DMA copies only the real DK/DV columns; the
    // staging rows are padded to hold DK_pad/DV_pad columns (tail zero-filled below)
    // so the HMX interleave/tile logic can operate on 64-aligned head dims.
    ثابت size_t سائيز_k_صف        = htp_tensor_get_صف_سائيز(k->type, DK);
    ثابت size_t سائيز_v_صف        = htp_tensor_get_صف_سائيز(v->type, DV);
    ثابت size_t سائيز_k_صف_padded = hex_ڦيرو_مٿي(DK_pad * حجم(__fp16), 128);
    ثابت size_t سائيز_v_صف_padded = hex_ڦيرو_مٿي(DV_pad * حجم(__fp16), 128);

    // Build the VTCM layout once (shared with the host estimator) and place every
    // scratch buffer at its computed offset. Padded head dims size the HMX tiles.
    بنية hmx_fa_vtcm_layout L;
    hmx_fa_vtcm_layout_build(&L, G, DK_pad, DV_pad, Br, Bc, n_ڌاڳا, pipeline, factx.is_q_fp32, factx.sinks != عدم, factx.n_heads);

    إذا (L.ڪل_بائيٽون > ctx->vtcm_سائيز) {
        أرجع HTP_حالت_VTCM_TOO_SMALL;
    }

    uint8_t * ثابت بنياد = ctx->vtcm_بنياد;

    factx.vtcm_q_dma          = VTCM_LAYOUT_PTR(__fp16, بنياد, L.off_q_dma);
    factx.vtcm_q_tiles        = VTCM_LAYOUT_PTR(__fp16, بنياد, L.off_q_tiles);
    factx.vtcm_o_tiles[0]     = VTCM_LAYOUT_PTR(__fp16, بنياد, L.off_o_tiles[0]);
    factx.vtcm_o_tiles[1]     = VTCM_LAYOUT_PTR(__fp16, بنياد, L.off_o_tiles[1]);
    factx.vtcm_k_fp16[0]      = VTCM_LAYOUT_PTR(__fp16, بنياد, L.off_k_fp16[0]);
    factx.vtcm_k_fp16[1]      = VTCM_LAYOUT_PTR(__fp16, بنياد, L.off_k_fp16[1]);
    factx.vtcm_v_fp16[0]      = VTCM_LAYOUT_PTR(__fp16, بنياد, L.off_v_fp16[0]);
    factx.vtcm_v_fp16[1]      = VTCM_LAYOUT_PTR(__fp16, بنياد, L.off_v_fp16[1]);
    factx.vtcm_k_tiles[0]     = VTCM_LAYOUT_PTR(__fp16, بنياد, L.off_k_tiles[0]);
    factx.vtcm_k_tiles[1]     = VTCM_LAYOUT_PTR_OPTIONAL(__fp16, بنياد, L.off_k_tiles[1], pipeline);
    factx.vtcm_v_tiles[0]     = VTCM_LAYOUT_PTR(__fp16, بنياد, L.off_v_tiles[0]);
    factx.vtcm_v_tiles[1]     = VTCM_LAYOUT_PTR_OPTIONAL(__fp16, بنياد, L.off_v_tiles[1], pipeline);
    factx.vtcm_s_tiles[0]     = VTCM_LAYOUT_PTR(__fp16, بنياد, L.off_s_tiles[0]);
    factx.vtcm_s_tiles[1]     = VTCM_LAYOUT_PTR_OPTIONAL(__fp16, بنياد, L.off_s_tiles[1], pipeline);
    factx.vtcm_p_tiles[0]     = VTCM_LAYOUT_PTR(__fp16, بنياد, L.off_p_tiles[0]);
    factx.vtcm_p_tiles[1]     = VTCM_LAYOUT_PTR_OPTIONAL(__fp16, بنياد, L.off_p_tiles[1], pipeline);
    factx.vtcm_d_tiles[0]     = VTCM_LAYOUT_PTR(__fp16, بنياد, L.off_d_tiles[0]);
    factx.vtcm_d_tiles[1]     = VTCM_LAYOUT_PTR_OPTIONAL(__fp16, بنياد, L.off_d_tiles[1], pipeline);
    factx.vtcm_d_inv_l        = VTCM_LAYOUT_PTR(__fp16, بنياد, L.off_d_inv_l);
    factx.vtcm_m_vec          = VTCM_LAYOUT_PTR(HVX_ويڪٽر, بنياد, L.off_m_vec);
    factx.vtcm_l_vec          = VTCM_LAYOUT_PTR(HVX_ويڪٽر, بنياد, L.off_l_vec);
    factx.vtcm_s_rowmax       = VTCM_LAYOUT_PTR(HVX_ويڪٽر, بنياد, L.off_s_rowmax);
    factx.vtcm_p_rowsum       = VTCM_LAYOUT_PTR(HVX_ويڪٽر, بنياد, L.off_p_rowsum);
    factx.vtcm_صف_bufs       = VTCM_LAYOUT_PTR(HVX_ويڪٽر, بنياد, L.off_صف_bufs);
    factx.صف_buf_stride      = L.صف_buf_stride;
    factx.vtcm_hmx_scales_id  = VTCM_LAYOUT_PTR(uint8_t, بنياد, L.off_hmx_scales_id);
    factx.vtcm_hmx_scales_qk  = VTCM_LAYOUT_PTR(uint8_t, بنياد, L.off_hmx_scales_qk);
    factx.vtcm_نقاب_buf       = VTCM_LAYOUT_PTR(__fp16, بنياد, L.off_نقاب_buf);
    factx.نقاب_buf_صف_stride = L.نقاب_buf_صف_stride;
    factx.q_tile_بائيٽون        = L.q_tile_بائيٽون;
    factx.k_tile_بائيٽون        = L.k_tile_بائيٽون;
    factx.o_tile_بائيٽون        = L.o_tile_بائيٽون;
    factx.کانو_vec_بائيٽون       = L.کانو_vec_بائيٽون;
    factx.d_tile_بائيٽون        = L.d_tile_بائيٽون;
    factx.vtcm_slopes         = VTCM_LAYOUT_PTR(__fp16, بنياد, L.off_slopes);
    factx.vtcm_sinks          = VTCM_LAYOUT_PTR_OPTIONAL(عائم, بنياد, L.off_sinks, factx.sinks != عدم);

    ثابت size_t m_سٽ_بائيٽون = L.m_سٽ_بائيٽون;  // used by the mask DMAs in the KV loop

    dma_cache_fa_init(&factx.m_cache, (uint8_t *) factx.vtcm_نقاب_buf, L.m_buf_جڳهه_بائيٽون, HMX_FA_DMA_CACHE_سائيز);

    // Head-dim padding: the K/V DMA staging buffers and the flat-Q buffer are laid out
    // with padded row strides (size_{k,v,q}_row_padded, covering D_pad columns) but the
    // DMA only writes the real D columns per row. Zero the whole staging buffers once up
    // front so tail lanes [D, D_pad) stay zero for all KV blocks. No-op when already aligned.
    إذا (DK_pad != DK || DV_pad != DV) {
        ثابت size_t k_buf_بائيٽون = (size_t) factx.Bc * سائيز_k_صف_padded;
        ثابت size_t v_buf_بائيٽون = (size_t) factx.Bc * سائيز_v_صف_padded;
        hvx_splat_u8_a((حرف *) factx.vtcm_k_fp16[0], 0, k_buf_بائيٽون);
        hvx_splat_u8_a((حرف *) factx.vtcm_k_fp16[1], 0, k_buf_بائيٽون);
        hvx_splat_u8_a((حرف *) factx.vtcm_v_fp16[0], 0, v_buf_بائيٽون);
        hvx_splat_u8_a((حرف *) factx.vtcm_v_fp16[1], 0, v_buf_بائيٽون);
        // Flat-Q DMA scratch
        إذا (factx.vtcm_q_dma) {
            ثابت size_t q_dma_بائيٽون = hex_align_مٿي(factx.g_br * DK * (factx.is_q_fp32 ? حجم(عائم) : حجم(__fp16)), 128);
            hvx_splat_u8_a((حرف *) factx.vtcm_q_dma, 0, q_dma_بائيٽون);
        }
    }

    // ======== Initialize HMX output scales ========
    hmx_init_ڪالم_scales(factx.vtcm_hmx_scales_id, Q6_V_vsplat_R(0x3c00)); // 1.0
    {
        اتحاد { __fp16 f; uint16_t u; } sc = { .f = factx.scale };
        hmx_init_ڪالم_scales(factx.vtcm_hmx_scales_qk, Q6_V_vsplat_R(sc.u));
    }

    // ======== DMA setup ========
    dma_قطار * ثابت dma_q = ctx->dma[0];

    إذا (factx.sinks) {
        dma_قطار_ڌڪو(dma_q, dma_make_معلومات(factx.vtcm_sinks, factx.sinks->معلومات), L.sinks_بائيٽون, 0, factx.sinks->سائيز, 1);
        dma_قطار_ڪڍو(dma_q);
    }

    ثابت size_t n_صف_tiles_g_br = g_br / HMX_FP16_TILE_N_صفون;
    ثابت size_t n_tiles_في_bc   = Bc / HMX_FP16_TILE_N_کانن_ڳڻپ;

    ثابت size_t qo_element_سائيز = factx.is_q_fp32 ? حجم(عائم) : حجم(__fp16);

    ثابت bool q_transposed                 = q->nb[1] < q->nb[2];
    ثابت size_t q_src_stride               = q_transposed ? q->nb[2] : q->nb[1];
    ثابت size_t q_صف_بائيٽون_untransposed   = factx.G * factx.DK * qo_element_سائيز;
    ثابت size_t q_صف_بائيٽون_trans_ضرب_ڪندڙ   = factx.DK * qo_element_سائيز;

    // ======== Reusable job descriptors for pipeline ========
    hmx_fa_qk_ڪم_t       qk_ڪم;
    hmx_fa_o_update_ڪم_t ou_ڪم;
    hmx_fa_o_norm_ڪم_t   on_ڪم;

    // ======== Main loop ========
    لكل (uint32_t ib3 = 0; ib3 < neq3; ++ib3) {
        ثابت uint32_t im3 = نقاب ? fastmodulo(ib3, نقاب->ne[3], &factx.src3_div3) : 0;
        لكل (uint32_t q_شروعات = q_شروعات_min; q_شروعات < q_شروعات_max; q_شروعات += Br) {
            ثابت uint32_t n_صفون_q    = hex_smin(Br, neq1 - q_شروعات);
            ثابت size_t   n_صفون_g    = n_صفون_q * G;
            ثابت size_t   g_br_actual = hex_align_مٿي(n_صفون_g, HMX_FP16_TILE_N_صفون);
            ثابت size_t   n_صف_tiles = g_br_actual / HMX_FP16_TILE_N_صفون;

            uint16_t سرگرم_بلاڪون[FA_MAX_KV_بلاڪون];
            uint32_t n_سرگرم_بلاڪون = factx.n_kv_بلاڪون;
            إذا (نقاب && factx.نقاب_نشر && factx.n_kv_بلاڪون > 1) {
                ثابت size_t scratch_بائيٽون = factx.k_tile_بائيٽون * (factx.pipeline ? 2 : 1);
                n_سرگرم_بلاڪون = fa_جاچ_سرگرم_بلاڪون(
                    dma_q, نقاب, q_شروعات, n_صفون_q, 0, im3, nek1, Bc,
                    factx.n_kv_بلاڪون, factx.vtcm_k_tiles[0], scratch_بائيٽون, سرگرم_بلاڪون);
            } وإلا {
                #هدايت clang loop vectorize(disable)
                لكل (uint32_t b = 0; b < factx.n_kv_بلاڪون; ++b) {
                    سرگرم_بلاڪون[b] = (uint16_t) b;
                }
            }
            ثابت uint32_t b0       = (n_سرگرم_بلاڪون > 0) ? سرگرم_بلاڪون[0] : 0;
            ثابت uint32_t b0_شروعات = b0 * Bc;
            ثابت uint32_t b0_صفون  = hex_smin(Bc, nek1 - b0_شروعات);

            لكل (uint32_t kv_سر = kv_سر_min; kv_سر < kv_سر_max; ++kv_سر) {
                ثابت uint32_t ik2 = kv_سر;
                ثابت uint32_t ik3 = fastdiv(ib3, &kparams->نشر_rk3);
                ثابت uint32_t iv2 = kv_سر;
                ثابت uint32_t iv3 = fastdiv(ib3, &kparams->نشر_rv3);

                // 1. Push Q and KV DMAs for the first kv_head of this q_start block.
                // Subsequent kv_head iterations are enqueued early at the end of the previous iteration.
                إذا (kv_سر == kv_سر_min) {
                    ثابت dma_addr_t q_ptr = q->معلومات + q_شروعات * q->nb[1] +
                                            (kv_سر * factx.G) * q->nb[2] + ib3 * q->nb[3];
                    ثابت size_t q_صف_بائيٽون = q_transposed ? n_صفون_q * q_صف_بائيٽون_trans_ضرب_ڪندڙ : q_صف_بائيٽون_untransposed;
                    ثابت size_t n_صفون      = q_transposed ? factx.G : n_صفون_q;
                    dma_قطار_ڌڪو(dma_q, dma_make_معلومات(factx.vtcm_q_dma, q_ptr), q_صف_بائيٽون, hex_smax(q_src_stride, q_صف_بائيٽون), q_صف_بائيٽون, n_صفون);

                    إذا (n_سرگرم_بلاڪون > 0) {
                        ثابت dma_addr_t k_src = k->معلومات + b0_شروعات * k->nb[1] + ik2 * k->nb[2] + ik3 * k->nb[3];
                        dma_قطار_ڌڪو(dma_q, dma_make_معلومات(factx.vtcm_k_fp16[0], k_src), سائيز_k_صف_padded, k->nb[1], سائيز_k_صف, b0_صفون);

                        ثابت dma_addr_t v_src = v->معلومات + b0_شروعات * v->nb[1] + iv2 * v->nb[2] + iv3 * v->nb[3];
                        dma_قطار_ڌڪو(dma_q, dma_make_معلومات(factx.vtcm_v_fp16[0], v_src), سائيز_v_صف_padded, v->nb[1], سائيز_v_صف, b0_صفون);

                        إذا (factx.pipeline && نقاب) {
                            إذا (__builtin_expect(factx.نقاب_نشر, true)) {
                                ثابت dma_addr_t ms_src = نقاب->معلومات + q_شروعات * نقاب->nb[1] + im3 * نقاب->nb[3] + b0_شروعات * حجم(__fp16);
                                dma_cache_fa_ڌڪو(dma_q, &factx.m_cache, ms_src, m_سٽ_بائيٽون, نقاب->nb[1], b0_صفون * حجم(__fp16), n_صفون_q);
                            } وإلا {
                                fa_ڌڪو_نقاب_dma_gqa(dma_q, نقاب, q_شروعات, im3, b0_شروعات, kv_سر, G, m_سٽ_بائيٽون, b0_صفون, n_صفون_q, &factx);
                            }
                        }
                    }
                }

                // 2. Pop Q DMA (blocks until Q is loaded)
                dma_قطار_ڪڍو(dma_q);

                // ---- Load Q block & Initialize per-block state ----
                fa_phase_q_ڀريو(&factx, q, q_شروعات, kv_سر, ib3, n_صفون_g);

                __fp16 * o_tile_prev = factx.vtcm_o_tiles[0];
                __fp16 * o_tile_curr = factx.vtcm_o_tiles[1];

                // ---- KV block loop with DMA double-buffering ----
                size_t buf_idx = 0;

                htp_trace_event_شروعات(tr_hvx, HTP_TRACE_EVT_HVX_A_PREP, (uint16_t) q_شروعات);
                fa_compute_slopes(&factx, kv_سر, n_صفون_g);
                htp_trace_event_stop(tr_hvx, HTP_TRACE_EVT_HVX_A_PREP, (uint16_t) q_شروعات);

                ثابت size_t k_src_stride = سائيز_k_صف_padded / حجم(__fp16);
                ثابت size_t v_src_stride = سائيز_v_صف_padded / حجم(__fp16);

                hmx_قطار_t hmx_q = ctx->hmx_قطار;

                إذا (factx.pipeline) {
                    // Double-buffered job structs because HMX queue runs asynchronously
                    hmx_fa_qk_ڪم_t qk_ڪم[2];
                    hmx_fa_o_update_ڪم_t ou_ڪم[2];

                    // Prefetch second active block early if there are multiple active blocks
                    إذا (n_سرگرم_بلاڪون > 1) {
                        fa_prefetch_بلاڪ(dma_q, k, v, نقاب, سرگرم_بلاڪون[1], Bc, سائيز_k_صف_padded, سائيز_k_صف, سائيز_v_صف_padded, سائيز_v_صف,
                                          ik2, ik3, iv2, iv3, q_شروعات, im3, kv_سر, G, m_سٽ_بائيٽون, n_صفون_q, nek1, 1, &factx);
                    }

                    // Prep and start QK-dot(0)
                    فراغ * curr_k0 = (فراغ *) dma_قطار_ڪڍو(dma_q).dst;
                    fa_phase_k_interleave(&factx, b0_صفون, k_src_stride, curr_k0, b0_شروعات, 0);

                    qk_ڪم[0].q_tiles        = factx.vtcm_q_tiles;
                    qk_ڪم[0].k_tiles        = factx.vtcm_k_tiles[0];
                    qk_ڪم[0].s_tiles        = factx.vtcm_s_tiles[0];
                    qk_ڪم[0].n_صف_tiles    = n_صف_tiles;
                    qk_ڪم[0].n_کانو_tiles    = hmx_ceil_div(b0_صفون, HMX_FP16_TILE_N_کانن_ڳڻپ);
                    qk_ڪم[0].n_dot_tiles    = DK_pad / 32;
                    qk_ڪم[0].n_tiles_في_bc = n_tiles_في_bc;
                    qk_ڪم[0].hmx_scales     = factx.vtcm_hmx_scales_qk;
                    hmx_قطار_ڌڪو(hmx_q, hmx_قطار_make_desc(hmx_fa_qk_dot_ڪم_ڪندڙ, &qk_ڪم[0]));

                    لكل (uint32_t قدم = 0; قدم < n_سرگرم_بلاڪون; ++قدم) {
                        ثابت uint32_t kv_blk      = سرگرم_بلاڪون[قدم];
                        ثابت uint32_t kv_شروعات    = kv_blk * Bc;
                        ثابت uint32_t kv_صفون     = hex_smin(Bc, nek1 - kv_شروعات);
                        ثابت size_t   n_کانو_tiles = hmx_ceil_div(kv_صفون, HMX_FP16_TILE_N_کانن_ڳڻپ);

                        // ---- 1. Pop and run V-prep for current block ----
                        فراغ * curr_v = (فراغ *) dma_قطار_ڪڍو(dma_q).dst;
                        fa_phase_v_interleave(&factx, kv_صفون, v_src_stride, curr_v, factx.vtcm_v_tiles[buf_idx], n_tiles_في_bc, kv_شروعات);

                        // ---- 2. Pop and run mask-prep for current block ----
                        __fp16 * current_نقاب_vtcm = عدم;
                        إذا (نقاب) {
                            إذا (__builtin_expect(factx.نقاب_نشر, true)) {
                                current_نقاب_vtcm = (__fp16 *) dma_قطار_ڪڍو(dma_q).dst;
                            } وإلا {
                                fa_ڪڍو_نقاب_dma_gqa(dma_q, G);
                                current_نقاب_vtcm = factx.vtcm_نقاب_buf;
                            }
                        }

                        // ---- 3. Start HMX O update for step - 1 (reads P[1 - buf_idx], V[1 - buf_idx], D) ----
                        // O update relies on the previous block's P and V tiles.
                        // O update MUST be pushed before the next block's QK-dot: hmx_queue_pop() retires the
                        // oldest descriptor, so push order alone decides which pop waits for which job.
                        // If OU went in after QK(i+1), the pop below would retire QK(i+1) and leave
                        // OU(i-1) in flight into the next iteration, where V-prep overwrites V[prev_buf].
                        إذا (قدم > 0) {
                            ثابت uint32_t prev_blk      = سرگرم_بلاڪون[قدم - 1];
                            ثابت size_t prev_buf        = 1 - buf_idx;
                            ou_ڪم[prev_buf].o_curr      = o_tile_curr;
                            ou_ڪم[prev_buf].o_prev      = o_tile_prev;
                            ou_ڪم[prev_buf].p_tiles     = factx.vtcm_p_tiles[prev_buf];
                            ou_ڪم[prev_buf].v_tiles     = factx.vtcm_v_tiles[prev_buf];
                            ou_ڪم[prev_buf].d_tiles     = factx.vtcm_d_tiles[prev_buf];
                            ou_ڪم[prev_buf].hmx_scales  = factx.vtcm_hmx_scales_id;
                            ou_ڪم[prev_buf].n_صف_tiles = n_صف_tiles;
                            ou_ڪم[prev_buf].n_کانو_tiles =
                                hmx_ceil_div(hex_smin(Bc, nek1 - prev_blk * Bc), HMX_FP16_TILE_N_کانن_ڳڻپ);
                            ou_ڪم[prev_buf].n_صف_tiles_g_br = n_صف_tiles_g_br;
                            ou_ڪم[prev_buf].n_tiles_في_bc   = n_tiles_في_bc;
                            ou_ڪم[prev_buf].DV               = DV_pad;
                            hmx_قطار_ڌڪو(hmx_q, hmx_قطار_make_desc(hmx_fa_o_update_ڪم_ڪندڙ, &ou_ڪم[prev_buf]));
                        }

                        // ---- 4. Pop and run K-prep for next active block & push next QK-dot ----
                        إذا (قدم + 1 < n_سرگرم_بلاڪون) {
                            ثابت uint32_t ايندڙ_blk   = سرگرم_بلاڪون[قدم + 1];
                            ثابت uint32_t ايندڙ_شروعات = ايندڙ_blk * Bc;
                            ثابت uint32_t ايندڙ_صفون  = hex_smin(Bc, nek1 - ايندڙ_شروعات);
                            ثابت size_t   ايندڙ_buf   = 1 - buf_idx;

                            فراغ * ايندڙ_k = (فراغ *) dma_قطار_ڪڍو(dma_q).dst;
                            fa_phase_k_interleave(&factx, ايندڙ_صفون, k_src_stride, ايندڙ_k, ايندڙ_شروعات, ايندڙ_buf);

                            qk_ڪم[ايندڙ_buf].q_tiles        = factx.vtcm_q_tiles;
                            qk_ڪم[ايندڙ_buf].k_tiles        = factx.vtcm_k_tiles[ايندڙ_buf];
                            qk_ڪم[ايندڙ_buf].s_tiles        = factx.vtcm_s_tiles[ايندڙ_buf];
                            qk_ڪم[ايندڙ_buf].n_صف_tiles    = n_صف_tiles;
                            qk_ڪم[ايندڙ_buf].n_کانو_tiles    = hmx_ceil_div(ايندڙ_صفون, HMX_FP16_TILE_N_کانن_ڳڻپ);
                            qk_ڪم[ايندڙ_buf].n_dot_tiles    = DK_pad / 32;
                            qk_ڪم[ايندڙ_buf].n_tiles_في_bc = n_tiles_في_bc;
                            qk_ڪم[ايندڙ_buf].hmx_scales     = factx.vtcm_hmx_scales_qk;
                            hmx_قطار_ڌڪو(hmx_q, hmx_قطار_make_desc(hmx_fa_qk_dot_ڪم_ڪندڙ, &qk_ڪم[ايندڙ_buf]));
                        }

                        // ---- 5. Wait for current block's QK-dot to finish ----
                        hmx_قطار_ڪڍو(hmx_q);

                        // ---- 6. Phase 2: softmax + build_D ----
                        fa_softmax_دليلون_t sargs;
                        memset(&sargs, 0, حجم(sargs));
                        sargs.factx                = &factx;
                        sargs.buf_idx              = buf_idx;
                        sargs.kv_صفون              = kv_صفون;
                        sargs.n_صفون_g             = n_صفون_g;
                        sargs.n_کانو_tiles          = n_کانو_tiles;
                        sargs.n_tiles_في_bc       = n_tiles_في_bc;
                        sargs.n_صف_tiles          = n_صف_tiles;
                        sargs.n_صف_tiles_g_br     = n_صف_tiles_g_br;
                        sargs.Bc                   = Bc;
                        sargs.G                    = G;
                        sargs.kv_سر              = kv_سر;
                        sargs.kv_شروعات             = kv_شروعات;
                        sargs.q_شروعات              = q_شروعات;
                        sargs.ib3                  = ib3;
                        sargs.has_alibi            = (factx.max_bias != 0.0f);
                        sargs.نقاب                 = نقاب;
                        sargs.نقاب_vtcm            = current_نقاب_vtcm;
                        sargs.نقاب_vtcm_صف_stride = factx.نقاب_buf_صف_stride;
                        sargs.slopes               = factx.vtcm_slopes;

                        // Run Softmax on HVX (blocking call)
                        fa_phase_softmax_and_build_d(&factx, &sargs, n_صف_tiles, n_صف_tiles_g_br);

                        // Wait for HMX O update for step - 1 to finish
                        إذا (قدم > 0) {
                            hmx_قطار_ڪڍو(hmx_q);
                            hex_مٽا_سٽا_ptr((فراغ **) &o_tile_curr, (فراغ **) &o_tile_prev);
                        }

                        // Prefetch block step + 2
                        إذا (قدم + 2 < n_سرگرم_بلاڪون) {
                            fa_prefetch_بلاڪ(dma_q, k, v, نقاب, سرگرم_بلاڪون[قدم + 2], Bc, سائيز_k_صف_padded, سائيز_k_صف, سائيز_v_صف_padded, سائيز_v_صف,
                                              ik2, ik3, iv2, iv3, q_شروعات, im3, kv_سر, G, m_سٽ_بائيٽون, n_صفون_q, nek1, buf_idx, &factx);
                        }

                        buf_idx = 1 - buf_idx;
                    }

                    // Epilogue
                    إذا (n_سرگرم_بلاڪون > 0) {
                        ثابت uint32_t آخري_blk = سرگرم_بلاڪون[n_سرگرم_بلاڪون - 1];
                        ثابت size_t آخري_کانن_ڳڻپ  = hmx_ceil_div(hex_smin(Bc, nek1 - آخري_blk * Bc), HMX_FP16_TILE_N_کانن_ڳڻپ);
                        ou_ڪم[0].o_curr           = o_tile_curr;
                        ou_ڪم[0].o_prev           = o_tile_prev;
                        ou_ڪم[0].p_tiles          = factx.vtcm_p_tiles[1 - buf_idx];
                        ou_ڪم[0].v_tiles          = factx.vtcm_v_tiles[1 - buf_idx];
                        ou_ڪم[0].d_tiles          = factx.vtcm_d_tiles[1 - buf_idx];
                        ou_ڪم[0].hmx_scales       = factx.vtcm_hmx_scales_id;
                        ou_ڪم[0].n_صف_tiles      = n_صف_tiles;
                        ou_ڪم[0].n_کانو_tiles      = آخري_کانن_ڳڻپ;
                        ou_ڪم[0].n_صف_tiles_g_br = n_صف_tiles_g_br;
                        ou_ڪم[0].n_tiles_في_bc   = n_tiles_في_bc;
                        ou_ڪم[0].DV               = DV_pad;
                        hmx_قطار_ڌڪو(hmx_q, hmx_قطار_make_desc(hmx_fa_o_update_ڪم_ڪندڙ, &ou_ڪم[0]));

                        // Overlapped: run HVX build diag inv L while HMX is busy executing the update
                        htp_trace_event_شروعات(tr_hvx, HTP_TRACE_EVT_HVX_O_PROC, (uint16_t) q_شروعات);
                        fa_build_d_diag_inv_l(&factx, n_صف_tiles, n_صف_tiles_g_br, kv_سر, n_صفون_g);
                        htp_trace_event_stop(tr_hvx, HTP_TRACE_EVT_HVX_O_PROC, (uint16_t) q_شروعات);
                        hmx_قطار_ڪڍو(hmx_q);

                        hex_مٽا_سٽا_ptr((فراغ **) &o_tile_curr, (فراغ **) &o_tile_prev);
                    }

                } وإلا {
                    // Fallback path
                    لكل (uint32_t قدم = 0; قدم < n_سرگرم_بلاڪون; ++قدم) {
                        ثابت uint32_t kv_blk      = سرگرم_بلاڪون[قدم];
                        ثابت uint32_t kv_شروعات    = kv_blk * Bc;
                        ثابت uint32_t kv_صفون     = hex_smin(Bc, nek1 - kv_شروعات);
                        ثابت size_t   n_کانو_tiles = hmx_ceil_div(kv_صفون, HMX_FP16_TILE_N_کانن_ڳڻپ);

                        إذا (نقاب) {
                            إذا (__builtin_expect(factx.نقاب_نشر, true)) {
                                ثابت dma_addr_t ms_src = نقاب->معلومات + q_شروعات * نقاب->nb[1] + im3 * نقاب->nb[3] + kv_شروعات * حجم(__fp16);
                                dma_cache_fa_ڌڪو(dma_q, &factx.m_cache, ms_src, m_سٽ_بائيٽون, نقاب->nb[1], kv_صفون * حجم(__fp16), n_صفون_q);
                            } وإلا {
                                fa_ڌڪو_نقاب_dma_gqa(dma_q, نقاب, q_شروعات, im3, kv_شروعات, kv_سر, G, m_سٽ_بائيٽون, kv_صفون, n_صفون_q, &factx);
                            }
                        }

                        إذا (قدم + 1 < n_سرگرم_بلاڪون) {
                            ثابت uint32_t  prefetch_blk   = سرگرم_بلاڪون[قدم + 1];
                            ثابت uint32_t  prefetch_شروعات = prefetch_blk * Bc;
                            ثابت uint32_t  prefetch_صفون  = hex_smin(Bc, nek1 - prefetch_شروعات);
                            ثابت size_t    prefetch_buf   = 1 - buf_idx;
                            ثابت dma_addr_t k_prefetch_src = k->معلومات + prefetch_شروعات * k->nb[1] + ik2 * k->nb[2] + ik3 * k->nb[3];
                            dma_قطار_ڌڪو(dma_q, dma_make_معلومات(factx.vtcm_k_fp16[prefetch_buf], k_prefetch_src), سائيز_k_صف_padded, k->nb[1], سائيز_k_صف, prefetch_صفون);
                            ثابت dma_addr_t v_prefetch_src = v->معلومات + prefetch_شروعات * v->nb[1] + iv2 * v->nb[2] + iv3 * v->nb[3];
                            dma_قطار_ڌڪو(dma_q, dma_make_معلومات(factx.vtcm_v_fp16[prefetch_buf], v_prefetch_src), سائيز_v_صف_padded, v->nb[1], سائيز_v_صف, prefetch_صفون);
                        }

                        // Wait for current K DMA and interleave
                        فراغ * curr_k = (فراغ *) dma_قطار_ڪڍو(dma_q).dst;
                        fa_phase_k_interleave(&factx, kv_صفون, k_src_stride, curr_k, kv_شروعات, 0);

                        {
                            qk_ڪم.q_tiles        = factx.vtcm_q_tiles;
                            qk_ڪم.k_tiles        = factx.vtcm_k_tiles[0];
                            qk_ڪم.s_tiles        = factx.vtcm_s_tiles[0];
                            qk_ڪم.n_صف_tiles    = n_صف_tiles;
                            qk_ڪم.n_کانو_tiles    = n_کانو_tiles;
                            qk_ڪم.n_dot_tiles    = (size_t) (DK_pad / 32);
                            qk_ڪم.n_tiles_في_bc = n_tiles_في_bc;
                            qk_ڪم.hmx_scales     = factx.vtcm_hmx_scales_qk;

                            hmx_قطار_ڌڪو(ctx->hmx_قطار, hmx_قطار_make_desc(hmx_fa_qk_dot_ڪم_ڪندڙ, &qk_ڪم));
                            hmx_قطار_ڪڍو(ctx->hmx_قطار);
                        }

                        // Wait for current V DMA and interleave
                        فراغ * curr_v = (فراغ *) dma_قطار_ڪڍو(dma_q).dst;
                        fa_phase_v_interleave(&factx, kv_صفون, v_src_stride, curr_v, factx.vtcm_v_tiles[0], n_tiles_في_bc, kv_شروعات);

                        // ---- Phase 3: softmax + build_D ----
                        __fp16 * current_نقاب_vtcm = عدم;
                        إذا (نقاب) {
                            إذا (__builtin_expect(factx.نقاب_نشر, true)) {
                                current_نقاب_vtcm = (__fp16 *) dma_قطار_ڪڍو(dma_q).dst;
                            } وإلا {
                                fa_ڪڍو_نقاب_dma_gqa(dma_q, G);
                                current_نقاب_vtcm = factx.vtcm_نقاب_buf;
                            }
                        }

                        fa_softmax_دليلون_t sargs;
                        memset(&sargs, 0, حجم(sargs));
                        sargs.factx                = &factx;
                        sargs.kv_صفون              = kv_صفون;
                        sargs.n_صفون_g             = n_صفون_g;
                        sargs.n_کانو_tiles          = n_کانو_tiles;
                        sargs.n_tiles_في_bc       = n_tiles_في_bc;
                        sargs.n_صف_tiles          = n_صف_tiles;
                        sargs.n_صف_tiles_g_br     = n_صف_tiles_g_br;
                        sargs.Bc                   = Bc;
                        sargs.G                    = G;
                        sargs.kv_سر              = kv_سر;
                        sargs.kv_شروعات             = kv_شروعات;
                        sargs.q_شروعات              = q_شروعات;
                        sargs.ib3                  = ib3;
                        sargs.has_alibi            = (factx.max_bias != 0.0f);
                        sargs.نقاب                 = نقاب;
                        sargs.نقاب_vtcm            = current_نقاب_vtcm;
                        sargs.نقاب_vtcm_صف_stride = factx.نقاب_buf_صف_stride;
                        sargs.slopes               = factx.vtcm_slopes;
                        fa_phase_softmax_and_build_d(&factx, &sargs, n_صف_tiles, n_صف_tiles_g_br);

                        {
                            ou_ڪم.o_curr           = o_tile_curr;
                            ou_ڪم.o_prev           = o_tile_prev;
                            ou_ڪم.p_tiles          = factx.vtcm_p_tiles[0];
                            ou_ڪم.v_tiles          = factx.vtcm_v_tiles[0];
                            ou_ڪم.d_tiles          = factx.vtcm_d_tiles[0];
                            ou_ڪم.hmx_scales       = factx.vtcm_hmx_scales_id;
                            ou_ڪم.n_صف_tiles      = n_صف_tiles;
                            ou_ڪم.n_کانو_tiles      = n_کانو_tiles;
                            ou_ڪم.n_صف_tiles_g_br = n_صف_tiles_g_br;
                            ou_ڪم.n_tiles_في_bc   = n_tiles_في_bc;
                            ou_ڪم.DV               = DV_pad;

                            hmx_قطار_ڌڪو(ctx->hmx_قطار, hmx_قطار_make_desc(hmx_fa_o_update_ڪم_ڪندڙ, &ou_ڪم));
                            إذا (قدم + 1 == n_سرگرم_بلاڪون) {
                                // Overlapped: run HVX build diag inv L while HMX is busy executing the update
                                htp_trace_event_شروعات(tr_hvx, HTP_TRACE_EVT_HVX_O_PROC, (uint16_t) q_شروعات);
                                fa_build_d_diag_inv_l(&factx, n_صف_tiles, n_صف_tiles_g_br, kv_سر, n_صفون_g);
                                htp_trace_event_stop(tr_hvx, HTP_TRACE_EVT_HVX_O_PROC, (uint16_t) q_شروعات);
                            }
                            hmx_قطار_ڪڍو(ctx->hmx_قطار);

                            hex_مٽا_سٽا_ptr((فراغ **) &o_tile_curr, (فراغ **) &o_tile_prev);
                        }

                        buf_idx = 1 - buf_idx;
                    }
                }

                // Enqueue DMAs for the next kv_head early so they overlap with O-PROC
                ثابت uint32_t ايندڙ_kv_سر = kv_سر + 1;
                ثابت bool has_ايندڙ = (ايندڙ_kv_سر < kv_سر_max);

                إذا (has_ايندڙ) {
                    ثابت dma_addr_t ايندڙ_q_ptr = q->معلومات + q_شروعات * q->nb[1] + (ايندڙ_kv_سر * factx.G) * q->nb[2] + ib3 * q->nb[3];
                    ثابت size_t ايندڙ_q_صف_بائيٽون = q_transposed ? n_صفون_q * q_صف_بائيٽون_trans_ضرب_ڪندڙ : q_صف_بائيٽون_untransposed;
                    ثابت size_t ايندڙ_n_صفون      = q_transposed ? factx.G : n_صفون_q;
                    dma_قطار_ڌڪو(dma_q, dma_make_معلومات(factx.vtcm_q_dma, ايندڙ_q_ptr), ايندڙ_q_صف_بائيٽون, hex_smax(q_src_stride, ايندڙ_q_صف_بائيٽون), ايندڙ_q_صف_بائيٽون, ايندڙ_n_صفون);

                    إذا (n_سرگرم_بلاڪون > 0) {
                        ثابت uint32_t ايندڙ_ik2 = ايندڙ_kv_سر;
                        ثابت uint32_t ايندڙ_iv2 = ايندڙ_kv_سر;

                        ثابت dma_addr_t ايندڙ_k_src = k->معلومات + b0_شروعات * k->nb[1] + ايندڙ_ik2 * k->nb[2] + ik3 * k->nb[3];
                        dma_قطار_ڌڪو(dma_q, dma_make_معلومات(factx.vtcm_k_fp16[0], ايندڙ_k_src), سائيز_k_صف_padded, k->nb[1], سائيز_k_صف, b0_صفون);

                        ثابت dma_addr_t ايندڙ_v_src = v->معلومات + b0_شروعات * v->nb[1] + ايندڙ_iv2 * v->nb[2] + iv3 * v->nb[3];
                        dma_قطار_ڌڪو(dma_q, dma_make_معلومات(factx.vtcm_v_fp16[0], ايندڙ_v_src), سائيز_v_صف_padded, v->nb[1], سائيز_v_صف, b0_صفون);

                        إذا (factx.pipeline && نقاب) {
                            إذا (__builtin_expect(factx.نقاب_نشر, true)) {
                                ثابت dma_addr_t ms_src = نقاب->معلومات + q_شروعات * نقاب->nb[1] + im3 * نقاب->nb[3] + b0_شروعات * حجم(__fp16);
                                dma_cache_fa_ڌڪو(dma_q, &factx.m_cache, ms_src, m_سٽ_بائيٽون, نقاب->nb[1], b0_صفون * حجم(__fp16), n_صفون_q);
                            } وإلا {
                                fa_ڌڪو_نقاب_dma_gqa(dma_q, نقاب, q_شروعات, im3, b0_شروعات, ايندڙ_kv_سر, G, m_سٽ_بائيٽون, b0_صفون, n_صفون_q, &factx);
                            }
                        }
                    }
                }

                // ---- Final normalization ----
                {
                    on_ڪم.o_curr           = o_tile_curr;
                    on_ڪم.o_prev           = o_tile_prev;
                    on_ڪم.d_tiles          = factx.vtcm_d_inv_l;
                    on_ڪم.hmx_scales       = factx.vtcm_hmx_scales_id;
                    on_ڪم.n_صف_tiles      = n_صف_tiles;
                    on_ڪم.n_صف_tiles_g_br = n_صف_tiles_g_br;
                    on_ڪم.DV               = DV_pad;
                    hmx_قطار_ڌڪو(ctx->hmx_قطار, hmx_قطار_make_desc(hmx_fa_o_norm_ڪم_ڪندڙ, &on_ڪم));
                    hmx_قطار_ڪڍو(ctx->hmx_قطار);
                }

                // ---- Store O block ----
                fa_phase_o_store(&factx, dst, o_tile_curr, q_شروعات, kv_سر, ib3, n_صفون_g);
            }
        }
    }

    أرجع HTP_حالت_OK;
}

صحيح op_flash_attn_ext(بنية htp_ops_context * octx) {
    ثابت بنية htp_tensor * q    = octx->src[0];
    ثابت بنية htp_tensor * k    = octx->src[1];
    ثابت بنية htp_tensor * v    = octx->src[2];
    ثابت بنية htp_tensor * نقاب = octx->src[3];
    ثابت بنية htp_tensor * dst  = octx->dst;

    // Check support
    إذا ((q->type != HTP_TYPE_F16 && q->type != HTP_TYPE_F32) ||
        (k->type != HTP_TYPE_F16 && k->type != HTP_TYPE_Q8_0) ||
        (v->type != HTP_TYPE_F16 && v->type != HTP_TYPE_Q8_0)) {
        أرجع HTP_حالت_NO_SUPPORT;
    }

    إذا (htp_tensor_is_extended(dst)) {
        أرجع HTP_حالت_NO_SUPPORT;
    }

    ثابت بنية htp_fa_kernel_params * kparams = (ثابت بنية htp_fa_kernel_params *) octx->kernel_params;

    إذا (kparams->kernel_type == HTP_FA_KERNEL_UNSUPPORTED || kparams->n_kv_بلاڪون > FA_MAX_KV_بلاڪون) {
        أرجع HTP_حالت_NO_SUPPORT;
    }

    إذا (!htp_ops_context_سيٽ_n_ڌاڳا(octx, kparams->n_ڌاڳا)) {
        أرجع HTP_حالت_INVAL_PARAMS;
    }

    إذا (kparams->kernel_type == HTP_FA_KERNEL_HMX) {
        أرجع hmx_flash_attn_ext(octx);
    }

    بنية htp_fa_context factx;
    factx.octx = octx;
    factx.k = k;
    factx.v = v;

    factx.src0_div21 = kparams->u.hvx.src0_div21;
    factx.src0_div1  = kparams->u.hvx.src0_div1;

    factx.نشر_rk2 = kparams->نشر_rk2;
    factx.نشر_rk3 = kparams->نشر_rk3;
    factx.نشر_rv2 = kparams->نشر_rv2;
    factx.نشر_rv3 = kparams->نشر_rv3;

    إذا (نقاب) {
        factx.src3_div2 = kparams->src3_div2;
        factx.src3_div3 = kparams->src3_div3;
    }

    factx.is_q_fp32 = (q->type == HTP_TYPE_F32);
    factx.سائيز_q_صف_padded = kparams->u.hvx.سائيز_q_صف_padded;
    factx.سائيز_k_صف_padded = kparams->u.hvx.سائيز_k_صف_padded;
    factx.سائيز_v_صف_padded = kparams->u.hvx.سائيز_v_صف_padded;

    factx.n_بلاڪون = kparams->n_kv_بلاڪون;

    factx.scale = kparams->scale;
    factx.max_bias = kparams->max_bias;
    factx.has_softcap = (kparams->logit_softcap != 0.0f);
    factx.logit_softcap = 0;
    إذا (factx.has_softcap) {
        factx.logit_softcap = (__fp16) kparams->logit_softcap;
    }

    factx.n_سر_log2 = kparams->n_سر_log2;
    factx.m0          = kparams->m0;
    factx.m1          = kparams->m1;

    ثابت uint32_t n_سر = q->ne[2];
    إذا (n_سر > 512) {
        أرجع HTP_حالت_NO_SUPPORT;
    }
    لكل (uint32_t h = 0; h < n_سر; ++h) {
        factx.slopes[h] = (__fp16) ((kparams->max_bias > 0.0f) ? alibi_لاهي(h, factx.n_سر_log2, factx.m0, factx.m1) : 1.0f);
    }

    // total rows in q
    ثابت uint32_t neq1 = q->ne[1];
    ثابت uint32_t neq2 = q->ne[2];
    ثابت uint32_t neq3 = q->ne[3];
    ثابت uint32_t ڪل_qrows = neq1 * neq2 * neq3;

    uint32_t qrow_شروعات = 0;
    uint32_t qrows      = ڪل_qrows;

    إذا (octx->ctx->mdev.ڳڻپ > 1) {
        ثابت uint32_t mdev_ڳڻپ = octx->ctx->mdev.ڳڻپ;
        ثابت uint32_t mdev_idx   = octx->ctx->mdev.idx;
        ثابت uint32_t n_kv_heads = k->ne[2];
        ثابت uint32_t dst_e_سائيز = (dst->type == HTP_TYPE_F32) ? حجم(عائم) : حجم(__fp16);
        ثابت bool can_ورهايو      = htp_tensor_can_صف_حصا_ڪريو(dst, dst_e_سائيز);

        // head range is contiguous in flat row space only when neq3 == 1
        إذا (kparams->سر_ورهايو && can_ورهايو && neq3 == 1 && n_kv_heads >= mdev_ڳڻپ && n_kv_heads % mdev_ڳڻپ == 0) {
            ثابت uint32_t G              = kparams->G;
            ثابت uint32_t kv_في_core    = n_kv_heads / mdev_ڳڻپ;
            ثابت uint32_t heads_في_core = kv_في_core * G;
            ثابت uint32_t سر_شروعات     = mdev_idx * heads_في_core;
            qrow_شروعات = سر_شروعات * neq1;
            qrows      = heads_في_core * neq1;
        } وإلا {
            ثابت بنية htp_tensor_mdev_range range = htp_tensor_mdev_حصا_ڪريو(ڪل_qrows, can_ورهايو ? 1 : 0, mdev_idx, mdev_ڳڻپ, &octx->ctx->mdev.ڳڻپ_div);
            qrow_شروعات = range.شروعات;
            qrows      = range.ڳڻپ;
        }
    }

    إذا (qrows == 0) {
        أرجع HTP_حالت_OK;
    }

    ثابت uint32_t n_ڌاڳا = octx->n_ڌاڳا;

    factx.qrows            = qrows;
    factx.qrow_شروعات       = qrow_شروعات;
    factx.qrows_في_thread = fastdiv(qrows + n_ڌاڳا - 1, &octx->n_ڌاڳا_div);

    ثابت bool has_نقاب  = (نقاب != عدم);
    ثابت bool has_sinks = (octx->src[4] != عدم);
    بنية hvx_fa_vtcm_layout L;
    hvx_fa_vtcm_layout_build(&L, k->ne[0], v->ne[0], factx.is_q_fp32, has_نقاب, has_sinks, n_سر, n_ڌاڳا);

    إذا (L.ڪل_بائيٽون > octx->ctx->vtcm_سائيز) {
        أرجع HTP_حالت_VTCM_TOO_SMALL;
    }

    factx.سائيز_q_بلاڪ = L.سائيز_q_بلاڪ;
    factx.سائيز_k_بلاڪ = L.سائيز_k_بلاڪ;
    factx.سائيز_v_بلاڪ = L.سائيز_v_بلاڪ;
    factx.سائيز_m_بلاڪ = L.سائيز_m_بلاڪ;
    factx.سائيز_vkq_acc = L.سائيز_vkq_acc;

    uint8_t * ثابت بنياد = octx->ctx->vtcm_بنياد;

    factx.spad_q               = VTCM_LAYOUT_PTR(uint8_t, بنياد, L.off_q);
    factx.spad_k               = VTCM_LAYOUT_PTR(uint8_t, بنياد, L.off_k);
    factx.spad_v               = VTCM_LAYOUT_PTR(uint8_t, بنياد, L.off_v);
    factx.spad_m               = VTCM_LAYOUT_PTR_OPTIONAL(uint8_t, بنياد, L.off_m, has_نقاب);
    factx.spad_a               = VTCM_LAYOUT_PTR(uint8_t, بنياد, L.off_a);
    factx.spad_sinks           = VTCM_LAYOUT_PTR_OPTIONAL(عائم, بنياد, L.off_sinks, has_sinks);
    factx.n_سرگرم_بلاڪون      = factx.n_بلاڪون;
    factx.shared_سرگرم_بلاڪون = false;

    إذا (has_نقاب && neq1 == 1 && neq3 == 1 && نقاب->ne[2] == 1 && factx.n_بلاڪون > 1 && factx.n_بلاڪون <= FA_MAX_KV_بلاڪون) {
        factx.n_سرگرم_بلاڪون = fa_جاچ_سرگرم_بلاڪون(
            octx->ctx->dma[0], نقاب, 0, 1, 0, 0, k->ne[1], FLASH_ATTN_بلاڪ_سائيز,
            factx.n_بلاڪون, factx.spad_k, factx.سائيز_k_بلاڪ * 2, factx.سرگرم_بلاڪون);
        factx.shared_سرگرم_بلاڪون = true;
    }

    إذا (has_sinks) {
        ثابت بنية htp_tensor * sinks = octx->src[4];
        dma_قطار * dma_q = octx->ctx->dma[0];
        dma_قطار_ڌڪو(dma_q, dma_make_معلومات(factx.spad_sinks, sinks->معلومات), L.سائيز_sinks, 0, sinks->سائيز, 1);
        dma_قطار_ڪڍو(dma_q);
    }

    work_قطار_لاڳيتو(octx->ctx->work_قطار, flash_attn_ext_f16_thread, &factx, n_ڌاڳا);

    أرجع HTP_حالت_OK;
}
