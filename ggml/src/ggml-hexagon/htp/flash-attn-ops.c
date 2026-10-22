#رهنمود clang diagnostic ignored "-Wunused-variable"
#رهنمود clang diagnostic ignored "-Wunused-function"
#رهنمود clang diagnostic ignored "-Wunused-but-set-variable"

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

#تعریف_کن GGML_COMMON_DECL_C
#ضمن "ggml-common.h"
#ضمن "htp-ctx.h"
#ضمن "htp-ops.h"
#ضمن "htp-tensor.h"
#ضمن "hvx-quant.h"

#ضمن "flash-attn-ops.h"
#ضمن "hvx-fa-kernels.h"
#ضمن "hmx-fa-kernels.h"

// Must be multiple of 32
#تعریف_کن FLASH_ATTN_بلوک_اندازه (32 * 2)

بنية htp_fa_context {
    ثابت بنية htp_ops_context * octx;

    بنية fastdiv_مقادیر src0_div21;
    بنية fastdiv_مقادیر src0_div1;

    بنية fastdiv_مقادیر پخش_rk2;
    بنية fastdiv_مقادیر پخش_rk3;
    بنية fastdiv_مقادیر پخش_rv2;
    بنية fastdiv_مقادیر پخش_rv3;

    بنية fastdiv_مقادیر src3_div2;
    بنية fastdiv_مقادیر src3_div3;

    عائم scale;
    عائم max_bias;
    bool  has_softcap;
    __fp16 logit_softcap;

    uint32_t n_سر_log2;
    عائم m0;
    عائم m1;
    __fp16 slopes[512];

    uint32_t n_بلوکها;
    uint32_t n_فعال_بلوکها;
    bool     shared_فعال_بلوکها;

    size_t اندازه_q_ردیف_padded;
    size_t اندازه_k_ردیف_padded;
    size_t اندازه_v_ردیف_padded;

    size_t اندازه_k_بلوک;
    size_t اندازه_v_بلوک;
    size_t اندازه_m_بلوک;

    uint32_t qrows;
    uint32_t qrows_در_هر_thread;
    uint32_t qrow_شروع;

    bool is_q_fp32;

    size_t اندازه_q_بلوک;
    size_t اندازه_vkq_acc;

    uint8_t *  spad_q;
    uint8_t *  spad_k;
    uint8_t *  spad_v;
    uint8_t *  spad_m;
    uint8_t *  spad_a;
    عائم *    spad_sinks;
    uint16_t   فعال_بلوکها[FA_MAX_KV_بلوکها];

    ثابت بنية htp_tensor * k;
    ثابت بنية htp_tensor * v;
};

بنية hmx_fa_context {
    ثابت بنية htp_ops_context * octx;
    ثابت بنية htp_tensor *      sinks;  // attention sinks (src[4]), NULL if absent
    ثابت بنية htp_tensor *      k;
    ثابت بنية htp_tensor *      v;
    bool         pipeline;  // true when n_kv_blocks >= FA_MIN_KV_BLOCKS && n_threads >= 2
    uint32_t     n_نخها;

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
    بنية fastdiv_مقادیر div_G;
    بنية fastdiv_مقادیر src3_div2;
    بنية fastdiv_مقادیر src3_div3;
    uint32_t     n_kv_بلوکها;
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
    HVX_وکتور * vtcm_m_vec;           // Row max [g_br]
    HVX_وکتور * vtcm_l_vec;           // Row sum [g_br]
    HVX_وکتور * vtcm_s_rowmax;        // Softmax intermediate [g_br]
    HVX_وکتور * vtcm_p_rowsum;        // Softmax intermediate [g_br]
    HVX_وکتور * vtcm_ردیف_bufs;        // Per-thread softmax row scratch [n_threads][2][Bc/64]
    uint8_t *    vtcm_hmx_scales_id;   // HMX output scales (identity)
    uint8_t *    vtcm_hmx_scales_qk;   // HMX output scales (qk_scale)
    __fp16 *     vtcm_ماسک_buf;        // VTCM mask buffer [Br * m_line], DMA'd per KV block
    __fp16 *     vtcm_slopes;          // ALiBi slopes [g_br]
    عائم *      vtcm_sinks;           // Attention sinks
    size_t       ردیف_buf_stride;       // HVX vectors per row buffer (Bc/64)
    size_t       ماسک_buf_ردیف_stride;  // elements (__fp16) per row in mask buffer
    size_t       q_tile_بایتها;
    size_t       k_tile_بایتها;
    size_t       o_tile_بایتها;
    size_t       ستون_vec_بایتها;
    size_t       d_tile_بایتها;
    bool         ماسک_پخش;       // true when mask->ne[2] == 1 (head-independent, single 2D DMA)
    dma_cache_fa m_cache;
};

// Scan mask rows [q_start, q_start + n_rows_q) and build compact list of KV blocks with any entry != -inf.
ساكن uint32_t fa_پیمایش_فعال_بلوکها(
        dma_صف *               dma_q,
        ثابت بنية htp_tensor * ماسک,
        uint32_t                  q_شروع,
        uint32_t                  n_ردیفها_q,
        uint32_t                  im2,
        uint32_t                  im3,
        uint32_t                  nek1,
        uint32_t                  Bc,
        uint32_t                  n_kv_بلوکها,
        فراغ *                    vtcm_scratch,
        size_t                    scratch_بایتها,
        uint16_t *                فعال_بلوکها) {
    ثابت size_t ردیف_بایتها  = (size_t) nek1 * حجم(__fp16);
    ثابت size_t ردیف_stride = hex_دور_به_بالا(ردیف_بایتها, 128);

    إذا (!ماسک || n_kv_بلوکها <= 1 || n_kv_بلوکها > FA_MAX_KV_بلوکها ||
        (size_t) n_ردیفها_q * ردیف_stride > scratch_بایتها) {
        #رهنمود clang loop vectorize(disable)
        لكل (uint32_t b = 0; b < n_kv_بلوکها; ++b) {
            فعال_بلوکها[b] = (uint16_t) b;
        }
        أرجع n_kv_بلوکها;
    }

    ثابت dma_addr_t m_src = ماسک->داده + q_شروع * ماسک->nb[1] + im2 * ماسک->nb[2] + im3 * ماسک->nb[3];
    dma_صف_افزودن(dma_q, dma_make_داده(vtcm_scratch, m_src), ردیف_stride, ماسک->nb[1], ردیف_بایتها, n_ردیفها_q);
    ثابت uint8_t * m_buf = (ثابت uint8_t *) dma_صف_برداشت(dma_q).dst;

    ثابت HVX_وکتور v_ninf   = Q6_Vh_vsplat_R(0xFC00);
    uint32_t         n_فعال = 0;

    لكل (uint32_t b = 0; b < n_kv_بلوکها; ++b) {
        ثابت uint32_t col0 = b * Bc;
        ثابت uint32_t تعداد_ستون_2 = MIN(Bc, nek1 - col0);

        لكل (uint32_t r = 0; r < n_ردیفها_q; ++r) {
            ثابت __fp16 * ردیف_ptr = (ثابت __fp16 *) (m_buf + r * ردیف_stride) + col0;
            HVX_وکتور v_any = Q6_V_vzero();
            uint32_t c = 0;
            لكل (; c + 64 <= تعداد_ستون_2; c += 64) {
                ثابت HVX_وکتور v_m = *(ثابت HVX_وکتور *) (ردیف_ptr + c);
                v_any = Q6_V_vor_VV(v_any, Q6_V_vxor_VV(v_m, v_ninf));
            }
            إذا (c < تعداد_ستون_2) {
                ثابت uint32_t rem = تعداد_ستون_2 - c;
                ثابت HVX_وکتور_Pred q_rem = Q6_Q_vsetq2_R(rem * حجم(__fp16));
                ثابت HVX_وکتور v_m = *(ثابت HVX_وکتور *) (ردیف_ptr + c);
                ثابت HVX_وکتور v_diff = Q6_V_vand_QV(q_rem, Q6_V_vxor_VV(v_m, v_ninf));
                v_any = Q6_V_vor_VV(v_any, v_diff);
            }
            v_any = Q6_V_vor_VV(v_any, Q6_V_vror_VR(v_any, 64));
            v_any = Q6_V_vor_VV(v_any, Q6_V_vror_VR(v_any, 32));
            v_any = Q6_V_vor_VV(v_any, Q6_V_vror_VR(v_any, 16));
            v_any = Q6_V_vor_VV(v_any, Q6_V_vror_VR(v_any, 8));
            v_any = Q6_V_vor_VV(v_any, Q6_V_vror_VR(v_any, 4));
            إذا (Q6_R_vextract_VR(v_any, 0) != 0) {
                فعال_بلوکها[n_فعال++] = (uint16_t) b;
                اكسر;
            }
        }
    }

    إذا (n_فعال == 0 && n_kv_بلوکها > 0) {
        فعال_بلوکها[0] = 0;
        n_فعال = 1;
    }
    أرجع n_فعال;
}

ساكن فراغ flash_attn_ext_f16_thread(بیعلامت صحيح nth, بیعلامت صحيح ith, فراغ * داده) {
    بنية htp_fa_context * factx = (بنية htp_fa_context *) داده;
    ثابت بنية htp_ops_context * octx = factx->octx;
    ثابت بنية htp_tensor * q     = octx->src[0];
    ثابت بنية htp_tensor * k     = octx->src[1];
    ثابت بنية htp_tensor * v     = octx->src[2];
    ثابت بنية htp_tensor * ماسک  = octx->src[3];
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
    ثابت uint32_t dr  = factx->qrows_در_هر_thread;
    ثابت uint32_t ir0 = factx->qrow_شروع + dr * ith;
    ثابت uint32_t ir1 = MIN(ir0 + dr, factx->qrow_شروع + factx->qrows);

    إذا (ir0 >= ir1) أرجع;

    بنية htp_thread_trace * tr = &octx->ctx->trace[ith];

    dma_صف * dma_q = octx->ctx->dma[ith];

    ثابت uint32_t DK = nek0;
    ثابت uint32_t DV = nev0;

    ثابت size_t اندازه_q_ردیف = DK * ((q->type == HTP_TYPE_F32) ? 4 : 2);
    ثابت size_t اندازه_k_ردیف = htp_tensor_get_ردیف_اندازه(k->type, DK);
    ثابت size_t اندازه_v_ردیف = htp_tensor_get_ردیف_اندازه(v->type, DV);

    // Scratchpad buffers for Q, K, V, Mask, and VKQ32 accumulator
    uint8_t * spad_q = factx->spad_q + factx->اندازه_q_بلوک * ith;
    uint8_t * spad_k = factx->spad_k + factx->اندازه_k_بلوک * 2 * ith;
    uint8_t * spad_v = factx->spad_v + factx->اندازه_v_بلوک * 2 * ith;
    uint8_t * spad_m = factx->spad_m + (ماسک ? factx->اندازه_m_بلوک * HVX_FA_DMA_CACHE_اندازه : 0) * ith;
    uint8_t * spad_a = factx->spad_a + factx->اندازه_vkq_acc * ith;

    dma_cache_dm m_cache;
    dma_cache_dm_init(&m_cache, spad_m, factx->اندازه_m_بلوک, HVX_FA_DMA_CACHE_اندازه);

    uint16_t thread_فعال_بلوکها[FA_MAX_KV_بلوکها];
    ثابت uint16_t * فعال_بلوکها = factx->shared_فعال_بلوکها ? factx->فعال_بلوکها : thread_فعال_بلوکها;
    uint32_t n_فعال_بلوکها       = factx->n_فعال_بلوکها;
    dma_addr_t prev_mp_پایه        = 0;

    إذا (!factx->shared_فعال_بلوکها && !ماسک) {
        #رهنمود clang loop vectorize(disable)
        لكل (uint32_t b = 0; b < factx->n_بلوکها; ++b) {
            thread_فعال_بلوکها[b] = (uint16_t) b;
        }
    }

    ثابت size_t اندازه_vkq_acc_single = hex_دور_به_بالا(DV * حجم(عائم), 128);

    uint32_t ir = ir0;
    طالما (ir < ir1) {
        ثابت uint32_t iq3 = fastdiv(ir, &factx->src0_div21);
        ثابت uint32_t iq2 = fastdiv(ir - iq3*neq2*neq1, &factx->src0_div1);
        ثابت uint32_t iq1 = (ir - iq3*neq2*neq1 - iq2 * neq1);

        ثابت uint32_t ik3 = fastdiv(iq3, &factx->پخش_rk3);
        ثابت uint32_t ik2 = fastdiv(iq2, &factx->پخش_rk2);

        ثابت uint32_t iv3 = fastdiv(iq3, &factx->پخش_rv3);
        ثابت uint32_t iv2 = fastdiv(iq2, &factx->پخش_rv2);

        uint32_t G_local = 1;
        إذا (neq1 == 1 && (ماسک == عدم || ماسک->ne[2] == 1)) {
            طالما (ir + G_local < ir1 && G_local < FA_HVX_G_MAX) {
                ثابت uint32_t بعدی_ir = ir + G_local;
                ثابت uint32_t بعدی_iq3 = fastdiv(بعدی_ir, &factx->src0_div21);
                ثابت uint32_t بعدی_iq2 = fastdiv(بعدی_ir - بعدی_iq3*neq2*neq1, &factx->src0_div1);
                ثابت uint32_t بعدی_iq1 = (بعدی_ir - بعدی_iq3*neq2*neq1 - بعدی_iq2 * neq1);

                ثابت uint32_t بعدی_ik3 = fastdiv(بعدی_iq3, &factx->پخش_rk3);
                ثابت uint32_t بعدی_ik2 = fastdiv(بعدی_iq2, &factx->پخش_rk2);

                ثابت uint32_t بعدی_iv3 = fastdiv(بعدی_iq3, &factx->پخش_rv3);
                ثابت uint32_t بعدی_iv2 = fastdiv(بعدی_iq2, &factx->پخش_rv2);

                إذا (بعدی_ik2 != ik2 || بعدی_ik3 != ik3 || بعدی_iv2 != iv2 || بعدی_iv3 != iv3 || بعدی_iq1 != iq1 || بعدی_iq3 != iq3) {
                    اكسر;
                }
                G_local++;
            }
        }

        dma_addr_t mp_پایه = 0;
        إذا (ماسک) {
            ثابت uint32_t im2 = fastmodulo(iq2, ماسک->ne[2], &factx->src3_div2);
            ثابت uint32_t im3 = fastmodulo(iq3, ماسک->ne[3], &factx->src3_div3);
            mp_پایه = ماسک->داده + iq1*ماسک->nb[1] + im2*ماسک->nb[2] + im3*ماسک->nb[3];
            إذا (!factx->shared_فعال_بلوکها && mp_پایه != prev_mp_پایه) {
                n_فعال_بلوکها = fa_پیمایش_فعال_بلوکها(
                    dma_q, ماسک, iq1, 1, im2, im3, nek1, FLASH_ATTN_بلوک_اندازه,
                    factx->n_بلوکها, spad_k, factx->اندازه_k_بلوک * 2, thread_فعال_بلوکها);
                prev_mp_پایه = mp_پایه;
            }
        }

        uint32_t heads[FA_HVX_G_MAX];
        HVX_وکتور شیب_vecs[FA_HVX_G_MAX] __attribute__((aligned(128)));
        HVX_وکتور S_vec[FA_HVX_G_MAX]      __attribute__((aligned(128)));
        HVX_وکتور M_vec[FA_HVX_G_MAX]      __attribute__((aligned(128)));
        uint8_t * q_ptrs[FA_HVX_G_MAX];
        عائم * vkq_ptrs[FA_HVX_G_MAX];

        لكل (uint32_t g = 0; g < G_local; ++g) {
            ثابت uint32_t r = ir + g;
            ثابت uint32_t r_iq3 = fastdiv(r, &factx->src0_div21);
            ثابت uint32_t r_iq2 = fastdiv(r - r_iq3*neq2*neq1, &factx->src0_div1);
            ثابت uint32_t r_iq1 = (r - r_iq3*neq2*neq1 - r_iq2 * neq1);

            heads[g] = r_iq2;
            ثابت __fp16 شیب = factx->slopes[r_iq2];
            شیب_vecs[g] = hvx_vec_splat_f16(شیب);

            S_vec[g] = hvx_vec_splat_f32(0.0f);
            M_vec[g] = hvx_vec_splat_f32(HTP_FA_M_INITIAL_VAL);

            uint8_t * q_dst = spad_q + g * factx->اندازه_q_ردیف_padded;
            q_ptrs[g] = q_dst;

            عائم * vkq_dst = (عائم *)(spad_a + g * اندازه_vkq_acc_single);
            vkq_ptrs[g] = vkq_dst;
            hvx_splat_f32_a((uint8_t *) vkq_dst, 0, DV);

            // Fetch Q row g
            ثابت dma_addr_t q_ردیف_ptr = q->داده + r_iq1*nbq1 + r_iq2*nbq2 + r_iq3*nbq3;
            dma_صف_افزودن(dma_q, dma_make_داده(q_dst, q_ردیف_ptr), factx->اندازه_q_ردیف_padded, nbq1, اندازه_q_ردیف, 1);
        }

        // Prefetch first two active blocks
        لكل (uint32_t گام = 0; گام < MIN(n_فعال_بلوکها, 2); ++گام) {
            ثابت uint32_t ib = فعال_بلوکها[گام];
            ثابت uint32_t ic_شروع = ib * FLASH_ATTN_بلوک_اندازه;
            ثابت uint32_t current_بلوک_اندازه = MIN(FLASH_ATTN_بلوک_اندازه, nek1 - ic_شروع);

            // K
            ثابت dma_addr_t k_src = k->داده + ic_شروع*nbk1 + ik2*nbk2 + ik3*nbk3;
            uint8_t * k_dst = spad_k + (گام % 2) * factx->اندازه_k_بلوک;
            dma_صف_افزودن(dma_q, dma_make_داده(k_dst, k_src), factx->اندازه_k_ردیف_padded, nbk1, اندازه_k_ردیف, current_بلوک_اندازه);

            // V
            ثابت dma_addr_t v_src = v->داده + ic_شروع*nbv1 + iv2*nbv2 + iv3*nbv3;
            uint8_t * v_dst = spad_v + (گام % 2) * factx->اندازه_v_بلوک;
            dma_صف_افزودن(dma_q, dma_make_داده(v_dst, v_src), factx->اندازه_v_ردیف_padded, nbv1, اندازه_v_ردیف, current_بلوک_اندازه);

            // Mask
            إذا (ماسک) {
                ثابت dma_addr_t m_src = mp_پایه + ic_شروع * حجم(__fp16);
                dma_cache_dm_افزودن(dma_q, &m_cache, ib, m_src, current_بلوک_اندازه * 2, current_بلوک_اندازه * 2, current_بلوک_اندازه * 2, 1);
            }
        }

        // Pop all Q rows
        لكل (uint32_t g = 0; g < G_local; ++g) {
            uint8_t * q_ptr_vtcm = (فراغ *) dma_صف_برداشت(dma_q).dst;
            إذا (factx->is_q_fp32) {
                hvx_copy_f16_f32_aa(q_ptr_vtcm, q_ptr_vtcm, DK);
            }
        }

        ثابت HVX_وکتور v_neg_inf = Q6_Vh_vsplat_R(0xfbff);
        ثابت bool has_softcap     = factx->has_softcap;
        ثابت HVX_وکتور v_cap     = has_softcap ? hvx_vec_splat_f16(factx->logit_softcap) : Q6_V_vzero();
        ثابت HVX_وکتور vinf      = Q6_Vh_vsplat_R(0xFC00);
        ثابت HVX_وکتور vmin      = Q6_Vh_vsplat_R(0xFBFF);
        ثابت HVX_وکتور v_log2e   = hvx_vec_splat_f16(EXP_LOG2E_F);
        ثابت uint32_t stride_v2   = factx->اندازه_v_ردیف_padded * 2;

        لكل (uint32_t گام = 0; گام < n_فعال_بلوکها; ++گام) {
            ثابت uint32_t ib = فعال_بلوکها[گام];
            ثابت uint32_t ic_شروع = ib * FLASH_ATTN_بلوک_اندازه;
            ثابت uint32_t current_بلوک_اندازه = MIN(FLASH_ATTN_بلوک_اندازه, nek1 - ic_شروع);

            // Wait for DMA
            uint8_t * k_پایه = (فراغ *) dma_صف_برداشت(dma_q).dst; // K
            uint8_t * v_پایه = (فراغ *) dma_صف_برداشت(dma_q).dst; // V
            __fp16  * m_پایه = ماسک ? (__fp16 *) dma_صف_برداشت(dma_q).dst : عدم; // M

            إذا (factx->k->type == HTP_TYPE_Q8_0) {
                htp_trace_event_شروع(tr, HTP_TRACE_EVT_HVX_FA_K_PREP, ir);
                لكل (uint32_t r = 0; r < current_بلوک_اندازه; ++r) {
                    __fp16 * ردیف_k = (__fp16 *)(k_پایه + r * factx->اندازه_k_ردیف_padded);
                    hvx_dequantize_ردیف_flat_q8_0_f16(ردیف_k, ردیف_k, DK);
                }
                htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_FA_K_PREP, ir);
            }
            إذا (factx->v->type == HTP_TYPE_Q8_0) {
                htp_trace_event_شروع(tr, HTP_TRACE_EVT_HVX_FA_V_PREP, ir);
                لكل (uint32_t r = 0; r < current_بلوک_اندازه; ++r) {
                    __fp16 * ردیف_v = (__fp16 *)(v_پایه + r * factx->اندازه_v_ردیف_padded);
                    hvx_dequantize_ردیف_flat_q8_0_f16(ردیف_v, ردیف_v, DV);
                }
                htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_FA_V_PREP, ir);
            }

            لكل (uint32_t g = 0; g < G_local; ++g) {
                ثابت uint32_t سر_ir = ir + g;
                uint8_t * q_ptr_vtcm = q_ptrs[g];
                عائم * VKQ32 = vkq_ptrs[g];
                ثابت HVX_وکتور شیب_vec = شیب_vecs[g];

                htp_trace_event_شروع(tr, HTP_TRACE_EVT_HVX_FA_QK, سر_ir);

                HVX_وکتور scores_f16 = Q6_V_vzero();
                إذا (current_بلوک_اندازه > 0) {
                    HVX_وکتور scores0 = hvx_dot_f16_f16_aa_rx32(q_ptr_vtcm, k_پایه, factx->اندازه_k_ردیف_padded, DK, factx->scale);
                    HVX_وکتور scores1 = (current_بلوک_اندازه > 32) ? hvx_dot_f16_f16_aa_rx32(q_ptr_vtcm, k_پایه + 32 * factx->اندازه_k_ردیف_padded, factx->اندازه_k_ردیف_padded, DK, factx->scale) : Q6_V_vzero();
                    scores_f16 = hvx_vec_f32_مقصد_f16(scores0, scores1);
                }

                إذا (has_softcap) {
                    scores_f16 = hvx_vec_tanh_f16(scores_f16);
                    scores_f16 = hvx_vec_mul_f16_f16(scores_f16, v_cap);
                }

                HVX_وکتور_Pred q_دم_keep = Q6_Q_vsetq2_R(current_بلوک_اندازه * حجم(__fp16));

                إذا (ماسک) {
                    HVX_وکتور m_vals_f16 = *(ثابت HVX_UVector *) m_پایه;
                    HVX_وکتور_Pred is_inf = Q6_Q_vcmp_eq_VhVh(m_vals_f16, vinf);
                    m_vals_f16 = Q6_V_vmux_QVV(is_inf, vmin, m_vals_f16);

                    HVX_وکتور m_scaled = hvx_vec_mul_f16_f16(m_vals_f16, شیب_vec);
                    scores_f16 = Q6_V_vmux_QVV(q_دم_keep, hvx_vec_جمع_کن_f16_f16(scores_f16, m_scaled), v_neg_inf);
                } وإلا {
                    scores_f16 = Q6_V_vmux_QVV(q_دم_keep, scores_f16, v_neg_inf);
                }

                HVX_وکتور v_max_f16 = hvx_vec_reduce_max_f16(scores_f16);
                HVX_وکتور v_max     = Q6_V_lo_W(hvx_vec_f16_مقصد_f32(v_max_f16));
                htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_FA_QK, سر_ir);

                // prefetch K for step + 2 after last head finished QK
                إذا (g + 1 == G_local && گام + 2 < n_فعال_بلوکها) {
                    ثابت uint32_t بعدی_ib = فعال_بلوکها[گام + 2];
                    ثابت uint32_t بعدی_ic_شروع = بعدی_ib * FLASH_ATTN_بلوک_اندازه;
                    ثابت uint32_t بعدی_بلوک_اندازه = MIN(FLASH_ATTN_بلوک_اندازه, nek1 - بعدی_ic_شروع);

                    ثابت dma_addr_t k_src = k->داده + بعدی_ic_شروع*nbk1 + ik2*nbk2 + ik3*nbk3;
                    dma_صف_افزودن(dma_q, dma_make_داده(k_پایه, k_src), factx->اندازه_k_ردیف_padded, nbk1, اندازه_k_ردیف, بعدی_بلوک_اندازه);
                }

                htp_trace_event_شروع(tr, HTP_TRACE_EVT_HVX_FA_SFM, سر_ir);
                {
                    HVX_وکتور M_new_vec = Q6_Vsf_vmax_VsfVsf(v_max, M_vec[g]);
                    HVX_وکتور diff_vec  = HVX_OP_SUB_F32(M_vec[g], M_new_vec);

                    HVX_وکتور diff_f16   = hvx_vec_f32_مقصد_f16(diff_vec, diff_vec);
                    HVX_وکتور diff_base2 = hvx_vec_mul_f16_f16(diff_f16, v_log2e);
                    HVX_وکتور ms_f16     = hvx_vec_exp2_f16(diff_base2);
                    HVX_وکتور ms_vec     = Q6_V_lo_W(hvx_vec_f16_مقصد_f32(ms_f16));

                    M_vec[g] = M_new_vec;

                    HVX_وکتور v_m_vec_f16 = hvx_vec_f32_مقصد_f16(M_vec[g], M_vec[g]);
                    HVX_وکتور v_s_minus_m = Q6_Vqf16_vsub_VhfVhf(scores_f16, v_m_vec_f16);
                    HVX_وکتور v_s_minus_m_base2 = hvx_vec_mul_f16_f16(Q6_Vhf_equals_Vqf16(v_s_minus_m), v_log2e);

                    HVX_وکتور P = hvx_vec_exp2_f16(v_s_minus_m_base2);
                    P = Q6_V_vmux_QVV(q_دم_keep, P, Q6_V_vzero());

                    HVX_وکتور_Pair P_pair = hvx_vec_f16_مقصد_f32(P);
                    HVX_وکتور P0 = Q6_V_lo_W(P_pair);
                    HVX_وکتور P1 = Q6_V_hi_W(P_pair);
                    HVX_وکتور p_جمع_vec = hvx_vec_reduce_جمع_f32(HVX_OP_جمع_کن_F32(P0, P1));

                    S_vec[g] = HVX_OP_جمع_کن_F32(HVX_OP_MUL_F32(S_vec[g], ms_vec), p_جمع_vec);

                    ثابت uint8_t * v_ptr = v_پایه;

                    إذا (DV == 64) {
                        HVX_وکتور_Pair vkq0 = *((ثابت HVX_وکتور_Pair *) VKQ32);
                        vkq0 = Q6_W_vcombine_VV(
                            HVX_OP_MUL_F32(Q6_V_hi_W(vkq0), ms_vec),
                            HVX_OP_MUL_F32(Q6_V_lo_W(vkq0), ms_vec)
                        );

                        لكل (uint32_t j = 0; j < current_بلوک_اندازه; j += 2) {
                            HVX_وکتور S0 = hvx_vec_repl_f16(Q6_V_vror_VR(P, j * 2));
                            ثابت HVX_وکتور * vx0 = (ثابت HVX_وکتور *) v_ptr;
                            إذا (j + 1 == current_بلوک_اندازه) {
                                vkq0 = hvx_vec_mpyacc_f32_f16(vkq0, Q6_Vh_vshuff_Vh(vx0[0]), S0);
                                اكسر;
                            }

                            HVX_وکتور S1 = hvx_vec_repl_f16(Q6_V_vror_VR(P, (j + 1) * 2));
                            ثابت HVX_وکتور * vx1 = (ثابت HVX_وکتور *) (v_ptr + factx->اندازه_v_ردیف_padded);
                            vkq0 = hvx_vec_mpyacc_f32_f16(vkq0, Q6_Vh_vshuff_Vh(vx0[0]), S0);
                            vkq0 = hvx_vec_mpyacc_f32_f16(vkq0, Q6_Vh_vshuff_Vh(vx1[0]), S1);
                            v_ptr += stride_v2;
                        }

                        *((HVX_وکتور_Pair *) VKQ32) = vkq0;
                    } وإلا إذا (DV == 128) {
                        HVX_وکتور_Pair vkq0 = ((ثابت HVX_وکتور_Pair *) VKQ32)[0];
                        HVX_وکتور_Pair vkq1 = ((ثابت HVX_وکتور_Pair *) VKQ32)[1];
                        vkq0 = Q6_W_vcombine_VV(
                            HVX_OP_MUL_F32(Q6_V_hi_W(vkq0), ms_vec),
                            HVX_OP_MUL_F32(Q6_V_lo_W(vkq0), ms_vec)
                        );
                        vkq1 = Q6_W_vcombine_VV(
                            HVX_OP_MUL_F32(Q6_V_hi_W(vkq1), ms_vec),
                            HVX_OP_MUL_F32(Q6_V_lo_W(vkq1), ms_vec)
                        );

                        لكل (uint32_t j = 0; j < current_بلوک_اندازه; j += 2) {
                            HVX_وکتور S0 = hvx_vec_repl_f16(Q6_V_vror_VR(P, j * 2));
                            ثابت HVX_وکتور * vx0 = (ثابت HVX_وکتور *) v_ptr;
                            إذا (j + 1 == current_بلوک_اندازه) {
                                vkq0 = hvx_vec_mpyacc_f32_f16(vkq0, Q6_Vh_vshuff_Vh(vx0[0]), S0);
                                vkq1 = hvx_vec_mpyacc_f32_f16(vkq1, Q6_Vh_vshuff_Vh(vx0[1]), S0);
                                اكسر;
                            }

                            HVX_وکتور S1 = hvx_vec_repl_f16(Q6_V_vror_VR(P, (j + 1) * 2));
                            ثابت HVX_وکتور * vx1 = (ثابت HVX_وکتور *) (v_ptr + factx->اندازه_v_ردیف_padded);
                            vkq0 = hvx_vec_mpyacc_f32_f16(vkq0, Q6_Vh_vshuff_Vh(vx0[0]), S0);
                            vkq0 = hvx_vec_mpyacc_f32_f16(vkq0, Q6_Vh_vshuff_Vh(vx1[0]), S1);
                            vkq1 = hvx_vec_mpyacc_f32_f16(vkq1, Q6_Vh_vshuff_Vh(vx0[1]), S0);
                            vkq1 = hvx_vec_mpyacc_f32_f16(vkq1, Q6_Vh_vshuff_Vh(vx1[1]), S1);
                            v_ptr += stride_v2;
                        }

                        ((HVX_وکتور_Pair *) VKQ32)[0] = vkq0;
                        ((HVX_وکتور_Pair *) VKQ32)[1] = vkq1;
                    } وإلا إذا (DV == 256) {
                        HVX_وکتور_Pair vkq0 = ((ثابت HVX_وکتور_Pair *) VKQ32)[0];
                        HVX_وکتور_Pair vkq1 = ((ثابت HVX_وکتور_Pair *) VKQ32)[1];
                        HVX_وکتور_Pair vkq2 = ((ثابت HVX_وکتور_Pair *) VKQ32)[2];
                        HVX_وکتور_Pair vkq3 = ((ثابت HVX_وکتور_Pair *) VKQ32)[3];
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

                        لكل (uint32_t j = 0; j < current_بلوک_اندازه; j += 2) {
                            HVX_وکتور S0 = hvx_vec_repl_f16(Q6_V_vror_VR(P, j * 2));
                            ثابت HVX_وکتور * vx0 = (ثابت HVX_وکتور *) v_ptr;
                            إذا (j + 1 == current_بلوک_اندازه) {
                                vkq0 = hvx_vec_mpyacc_f32_f16(vkq0, Q6_Vh_vshuff_Vh(vx0[0]), S0);
                                vkq1 = hvx_vec_mpyacc_f32_f16(vkq1, Q6_Vh_vshuff_Vh(vx0[1]), S0);
                                vkq2 = hvx_vec_mpyacc_f32_f16(vkq2, Q6_Vh_vshuff_Vh(vx0[2]), S0);
                                vkq3 = hvx_vec_mpyacc_f32_f16(vkq3, Q6_Vh_vshuff_Vh(vx0[3]), S0);
                                اكسر;
                            }

                            HVX_وکتور S1 = hvx_vec_repl_f16(Q6_V_vror_VR(P, (j + 1) * 2));
                            ثابت HVX_وکتور * vx1 = (ثابت HVX_وکتور *) (v_ptr + factx->اندازه_v_ردیف_padded);
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

                        ((HVX_وکتور_Pair *) VKQ32)[0] = vkq0;
                        ((HVX_وکتور_Pair *) VKQ32)[1] = vkq1;
                        ((HVX_وکتور_Pair *) VKQ32)[2] = vkq2;
                        ((HVX_وکتور_Pair *) VKQ32)[3] = vkq3;
                    } وإلا {
                        hvx_scale_vec_f32_aa((uint8_t *) VKQ32, (ثابت uint8_t *) VKQ32, DV, ms_vec);

                        لكل (uint32_t j = 0; j < current_بلوک_اندازه; j += 2) {
                            إذا (j + 1 == current_بلوک_اندازه) {
                                HVX_وکتور S0 = hvx_vec_repl_f16(Q6_V_vror_VR(P, j * 2));
                                hvx_mad_f32_f16_aa_vec(VKQ32, v_ptr, S0, DV);
                                اكسر;
                            }

                            HVX_وکتور S0 = hvx_vec_repl_f16(Q6_V_vror_VR(P, j * 2));
                            HVX_وکتور S1 = hvx_vec_repl_f16(Q6_V_vror_VR(P, (j + 1) * 2));

                            hvx_mad_f32_f16_aa_rx2_vec(VKQ32, v_ptr, v_ptr + factx->اندازه_v_ردیف_padded, S0, S1, DV);
                            v_ptr += stride_v2;
                        }
                    }
                }
                htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_FA_SFM, سر_ir);

                // prefetch V and mask for step + 2 after last head finished V accumulation
                إذا (g + 1 == G_local && گام + 2 < n_فعال_بلوکها) {
                    ثابت uint32_t بعدی_ib = فعال_بلوکها[گام + 2];
                    ثابت uint32_t بعدی_ic_شروع = بعدی_ib * FLASH_ATTN_بلوک_اندازه;
                    ثابت uint32_t بعدی_بلوک_اندازه = MIN(FLASH_ATTN_بلوک_اندازه, nek1 - بعدی_ic_شروع);

                    // V
                    ثابت dma_addr_t v_src = v->داده + بعدی_ic_شروع*nbv1 + iv2*nbv2 + iv3*nbv3;
                    dma_صف_افزودن(dma_q, dma_make_داده(v_پایه, v_src), factx->اندازه_v_ردیف_padded, nbv1, اندازه_v_ردیف, بعدی_بلوک_اندازه);

                    // Mask
                    إذا (ماسک) {
                        ثابت dma_addr_t m_src = mp_پایه + بعدی_ic_شروع * حجم(__fp16);
                        dma_cache_dm_افزودن(dma_q, &m_cache, بعدی_ib, m_src, بعدی_بلوک_اندازه * 2, بعدی_بلوک_اندازه * 2, بعدی_بلوک_اندازه * 2, 1);
                    }
                }
            } // end for g
        } // end for step

        لكل (uint32_t g = 0; g < G_local; ++g) {
            ثابت uint32_t سر_ir = ir + g;
            ثابت uint32_t h = heads[g];
            عائم * VKQ32 = vkq_ptrs[g];

            htp_trace_event_شروع(tr, HTP_TRACE_EVT_HVX_O_PROC, سر_ir);

            عائم M = hvx_vec_get_f32(M_vec[g]);
            عائم S = hvx_vec_get_f32(S_vec[g]);

            إذا (sinks) {
                ثابت عائم s = factx->spad_sinks[h];

                عائم vs = 1.0f;

                إذا (s > M) {
                    HVX_وکتور diff_vec = hvx_vec_splat_f32(M - s);
                    HVX_وکتور ms_vec   = hvx_vec_exp_f32(diff_vec);
                    hvx_scale_vec_f32_aa((uint8_t *) VKQ32, (ثابت uint8_t *) VKQ32, DV, ms_vec);

                    عائم ms = hvx_vec_get_f32(ms_vec);
                    S = S * ms + vs;
                } وإلا {
                    HVX_وکتور diff_vec = hvx_vec_splat_f32(s - M);
                    vs = hvx_vec_get_f32(hvx_vec_exp_f32(diff_vec));
                    S += vs;
                }
            }

            ثابت عائم S_inv = S == 0.0f ? 0.0f : 1.0f/S;
            hvx_scale_f32_aa((uint8_t *) VKQ32, (ثابت uint8_t *) VKQ32, DV, S_inv);

            ثابت uint32_t r_iq3 = fastdiv(سر_ir, &factx->src0_div21);
            ثابت uint32_t r_iq2 = fastdiv(سر_ir - r_iq3*neq2*neq1, &factx->src0_div1);
            ثابت uint32_t r_iq1 = (سر_ir - r_iq3*neq2*neq1 - r_iq2 * neq1);

            uint8_t * dst_ptr = (uint8_t *) dst->داده + r_iq2 * dst->nb[1] + r_iq1 * dst->nb[2] + r_iq3 * dst->nb[3];

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
    uint32_t                kv_ردیفها;
    size_t                  src_stride;
    فراغ *                  curr_k;
    uint32_t                kv_شروع;
    uint32_t                ردیفها_در_هر_t;
    size_t                  buf_idx;
} fa_k_int_آرگومانها_t;

ساكن فراغ fa_k_interleave_thread(بیعلامت صحيح n, بیعلامت صحيح i, فراغ * داده) {
    fa_k_int_آرگومانها_t *       آرگومانها_2  = (fa_k_int_آرگومانها_t *) داده;
    بنية hmx_fa_context * factx = آرگومانها_2->factx;

    ثابت uint32_t مجموع_ردیفها = آرگومانها_2->kv_ردیفها;
    ثابت uint32_t ردیفها_در_هر_t = آرگومانها_2->ردیفها_در_هر_t;
    ثابت uint32_t شروع      = i * ردیفها_در_هر_t;
    ثابت uint32_t انتها        = (uint32_t) hex_smin(شروع + ردیفها_در_هر_t, مجموع_ردیفها);

    إذا (شروع >= مجموع_ردیفها) {
        أرجع;
    }

    بنية htp_thread_trace * tr = &factx->octx->ctx->trace[i];
    htp_trace_event_شروع(tr, HTP_TRACE_EVT_HVX_FA_K_PREP, (uint16_t) (آرگومانها_2->kv_شروع + شروع));
    إذا (factx->k->type == HTP_TYPE_Q8_0) {
        لكل (uint32_t r = شروع; r < انتها; ++r) {
            __fp16 * ردیف_k = (__fp16 *)((حرف *)آرگومانها_2->curr_k + r * آرگومانها_2->src_stride * حجم(__fp16));
            hvx_dequantize_ردیف_flat_q8_0_f16(ردیف_k, ردیف_k, factx->DK);
        }
    }
    hmx_interleave_ردیفها_مقصد_tiles(factx->vtcm_k_tiles[آرگومانها_2->buf_idx], (ثابت __fp16 *) آرگومانها_2->curr_k, مجموع_ردیفها, factx->DK_pad,
                             آرگومانها_2->src_stride, شروع, انتها);
    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_FA_K_PREP, (uint16_t) (آرگومانها_2->kv_شروع + شروع));
}

ساكن فراغ fa_phase_k_interleave(بنية hmx_fa_context * factx, uint32_t kv_ردیفها, size_t src_stride, فراغ * curr_k, uint32_t kv_شروع, size_t buf_idx) {
    work_صف_t wp = factx->octx->ctx->work_صف;
    uint32_t n = 1;
    إذا (factx->n_نخها > 1 && kv_ردیفها >= factx->n_نخها * 2) {
        n = factx->n_نخها;
    }
    uint32_t ردیفها_در_هر_t = hex_align_به_بالا(hmx_ceil_div(kv_ردیفها, n), 2);
    fa_k_int_آرگومانها_t آرگومانها_2 = { factx, kv_ردیفها, src_stride, curr_k, kv_شروع, ردیفها_در_هر_t, buf_idx };
    إذا (n > 1) {
        work_صف_پیاپی(wp, fa_k_interleave_thread, &آرگومانها_2, n);
    } وإلا {
        fa_k_interleave_thread(1, 0, &آرگومانها_2);
    }
}

تعريفنوع بنية {
    بنية hmx_fa_context * factx;
    uint32_t                kv_ردیفها;
    size_t                  src_stride;
    فراغ *                  v_src;
    فراغ *                  v_tiles_dst;
    size_t                  n_ستون_tiles;
    uint32_t                kv_شروع;
    uint32_t                ردیفها_در_هر_t;
} fa_v_int_آرگومانها_t;

ساكن فراغ fa_v_interleave_thread(بیعلامت صحيح n, بیعلامت صحيح i, فراغ * داده) {
    fa_v_int_آرگومانها_t *       آرگومانها_2  = (fa_v_int_آرگومانها_t *) داده;
    بنية hmx_fa_context * factx = آرگومانها_2->factx;

    ثابت uint32_t مجموع_ردیفها = آرگومانها_2->kv_ردیفها;
    ثابت uint32_t ردیفها_در_هر_t = آرگومانها_2->ردیفها_در_هر_t;
    ثابت uint32_t شروع      = i * ردیفها_در_هر_t;
    ثابت uint32_t انتها        = (uint32_t) hex_smin(شروع + ردیفها_در_هر_t, مجموع_ردیفها);

    إذا (شروع >= مجموع_ردیفها) {
        أرجع;
    }

    __fp16 * v_tiles_dst = (__fp16 *) آرگومانها_2->v_tiles_dst;

    بنية htp_thread_trace * tr = &factx->octx->ctx->trace[i];
    htp_trace_event_شروع(tr, HTP_TRACE_EVT_HVX_FA_V_PREP, (uint16_t) (آرگومانها_2->kv_شروع + شروع));
    إذا (factx->v->type == HTP_TYPE_Q8_0) {
        لكل (uint32_t r = شروع; r < انتها; ++r) {
            __fp16 * ردیف_v = (__fp16 *)((حرف *)آرگومانها_2->v_src + r * آرگومانها_2->src_stride * حجم(__fp16));
            hvx_dequantize_ردیف_flat_q8_0_f16(ردیف_v, ردیف_v, factx->DV);
        }
    }
    hmx_interleave_تعداد_ستون_مقصد_tiles(v_tiles_dst, (ثابت __fp16 *) آرگومانها_2->v_src, مجموع_ردیفها, factx->DV_pad,
                             آرگومانها_2->src_stride, (uint32_t) آرگومانها_2->n_ستون_tiles, شروع, انتها);
    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_FA_V_PREP, (uint16_t) (آرگومانها_2->kv_شروع + شروع));
}

ساكن فراغ fa_phase_v_interleave(بنية hmx_fa_context * factx,
                                  uint32_t                kv_ردیفها,
                                  size_t                  src_stride,
                                  فراغ *                  v_src,
                                  فراغ *                  v_tiles_dst,
                                  size_t                  n_ستون_tiles,
                                  uint32_t                kv_شروع) {
    work_صف_t wp = factx->octx->ctx->work_صف;
    uint32_t n = 1;
    إذا (factx->n_نخها > 1 && kv_ردیفها >= factx->n_نخها * 2) {
        n = factx->n_نخها;
    }
    uint32_t ردیفها_در_هر_t = hex_align_به_بالا(hmx_ceil_div(kv_ردیفها, n), 2);
    fa_v_int_آرگومانها_t آرگومانها_2 = { factx, kv_ردیفها, src_stride, v_src, v_tiles_dst, n_ستون_tiles, kv_شروع, ردیفها_در_هر_t };
    إذا (n > 1) {
        work_صف_پیاپی(wp, fa_v_interleave_thread, &آرگومانها_2, n);
    } وإلا {
        fa_v_interleave_thread(1, 0, &آرگومانها_2);
    }
}

تعريفنوع بنية {
    بنية hmx_fa_context *   factx;
    ثابت بنية htp_tensor * q;
    uint32_t                  q_شروع;
    uint32_t                  kv_سر;
    uint32_t                  ib3;
    size_t                    n_ردیفها_g;
    size_t                    ردیفها_در_هر_t;
    size_t                    n_ردیفها_q;
    bool                      q_transposed;
    atomic_uint               barrier;
} fa_q_بارگیری_آرگومانها_t;

ساكن فراغ fa_q_بارگیری_thread(بیعلامت صحيح n, بیعلامت صحيح i, فراغ * داده) {
    fa_q_بارگیری_آرگومانها_t *      آرگومانها_2  = (fa_q_بارگیری_آرگومانها_t *) داده;
    بنية hmx_fa_context * factx = آرگومانها_2->factx;

    ثابت size_t n_ردیفها_g = آرگومانها_2->n_ردیفها_g;
    ثابت size_t G        = factx->G;
    ثابت size_t DK       = factx->DK;

    // Partition the padded Q rows (g_br) across threads.
    // Keep start/end even so r and r+1 are always in the same thread's range.
    ثابت size_t ردیفها_در_هر_t = آرگومانها_2->ردیفها_در_هر_t;
    ثابت size_t شروع      = (size_t) i * ردیفها_در_هر_t;
    ثابت size_t انتها        = hex_smin(شروع + ردیفها_در_هر_t, factx->g_br);

    بنية htp_thread_trace * tr = &factx->octx->ctx->trace[i];
    htp_trace_event_شروع(tr, HTP_TRACE_EVT_HVX_FA_Q_PREP, (uint16_t) (آرگومانها_2->q_شروع * G + شروع));

    // Parallel initialization of per-block state
    {
        ثابت uint32_t g_br = factx->g_br;
        ثابت uint32_t DV   = factx->DV;

        ثابت size_t ستون_vec_بایتها = factx->ستون_vec_بایتها;
        ثابت size_t d_tile_بایتها  = factx->d_tile_بایتها;

        // Initialize vtcm_l_vec & vtcm_m_vec
        ثابت size_t l_بایتها_در_هر_t = hex_align_به_بالا(ستون_vec_بایتها / n, 128);
        ثابت size_t l_شروع       = i * l_بایتها_در_هر_t;
        ثابت size_t l_انتها         = hex_smin(l_شروع + l_بایتها_در_هر_t, ستون_vec_بایتها);

        ثابت size_t m_بایتها_در_هر_t = hex_align_به_بالا(ستون_vec_بایتها / n, 128);
        ثابت size_t m_شروع       = i * m_بایتها_در_هر_t;
        ثابت size_t m_انتها         = hex_smin(m_شروع + m_بایتها_در_هر_t, ستون_vec_بایتها);

        إذا (l_شروع < ستون_vec_بایتها) {
            hvx_splat_u8_a((حرف *) factx->vtcm_l_vec + l_شروع, 0, l_انتها - l_شروع);
        }
        إذا (m_شروع < ستون_vec_بایتها) {
            hvx_splat_f32_a((حرف *) factx->vtcm_m_vec + m_شروع, HTP_FA_M_INITIAL_VAL, (m_انتها - m_شروع) / حجم(عائم));
        }

        // Zero the whole rescale region: vtcm_d_tiles[0], the optional vtcm_d_tiles[1]
        // and vtcm_d_inv_l are equal-sized and allocated back to back, so one run covers
        // them all.  The scatter only ever writes the diagonal, ignore the rest.
        ثابت size_t d_بایتها_در_هر_t = hex_align_به_بالا(d_tile_بایتها / n, 128);
        ثابت size_t d_شروع       = i * d_بایتها_در_هر_t;
        ثابت size_t d_انتها         = hex_smin(d_شروع + d_بایتها_در_هر_t, d_tile_بایتها);
        إذا (d_شروع < d_tile_بایتها) {
            hvx_splat_u8_a((حرف *) factx->vtcm_d_tiles[0] + d_شروع, 0, d_انتها - d_شروع);
        }
    }

    إذا (شروع < factx->g_br) {
        ثابت بنية htp_tensor * q       = آرگومانها_2->q;
        ثابت uint32_t            q_شروع = آرگومانها_2->q_شروع;
        ثابت uint32_t            kv_سر = آرگومانها_2->kv_سر;
        ثابت uint32_t            ib3     = آرگومانها_2->ib3;

        ثابت bool use_q_dma = (factx->vtcm_q_dma != عدم);

        __fp16 * q_tiles = factx->vtcm_q_tiles;
        ثابت size_t DK_pad = factx->DK_pad;
        إذا (use_q_dma) {
            ثابت size_t g_ردیفها_انتها = hex_smin(انتها, n_ردیفها_g);
            ثابت uint32_t d_حد = factx->is_q_fp32 ? DK / 32 : DK / 64;

            uint8_t * q_flat  = (uint8_t *) factx->vtcm_q_dma;
            إذا (DK_pad != DK) {
                إذا (factx->is_q_fp32) {
                    hmx_fa_q_prep_fp32_pad(q_tiles, q_flat, شروع, انتها, g_ردیفها_انتها, DK, DK_pad, G, آرگومانها_2->n_ردیفها_q, &factx->div_G, آرگومانها_2->q_transposed);
                } وإلا {
                    hmx_fa_q_prep_fp16_pad(q_tiles, q_flat, شروع, انتها, g_ردیفها_انتها, DK, DK_pad, G, آرگومانها_2->n_ردیفها_q, &factx->div_G, آرگومانها_2->q_transposed);
                }
            } وإلا إذا (factx->is_q_fp32) {
                بدل (d_حد) {
                حالة 2:  hmx_fa_q_prep_fp32_d2(q_tiles, q_flat, شروع, انتها, g_ردیفها_انتها, DK, G, آرگومانها_2->n_ردیفها_q, &factx->div_G, آرگومانها_2->q_transposed); اكسر;
                حالة 4:  hmx_fa_q_prep_fp32_d4(q_tiles, q_flat, شروع, انتها, g_ردیفها_انتها, DK, G, آرگومانها_2->n_ردیفها_q, &factx->div_G, آرگومانها_2->q_transposed); اكسر;
                افتراضي: hmx_fa_q_prep_fp32(   q_tiles, q_flat, شروع, انتها, g_ردیفها_انتها, DK, G, آرگومانها_2->n_ردیفها_q, &factx->div_G, d_حد, آرگومانها_2->q_transposed); اكسر;
                }
            } وإلا {
                بدل (d_حد) {
                حالة 1:  hmx_fa_q_prep_fp16_d1(q_tiles, q_flat, شروع, انتها, g_ردیفها_انتها, DK, G, آرگومانها_2->n_ردیفها_q, &factx->div_G, آرگومانها_2->q_transposed); اكسر;
                حالة 2:  hmx_fa_q_prep_fp16_d2(q_tiles, q_flat, شروع, انتها, g_ردیفها_انتها, DK, G, آرگومانها_2->n_ردیفها_q, &factx->div_G, آرگومانها_2->q_transposed); اكسر;
                افتراضي: hmx_fa_q_prep_fp16(   q_tiles, q_flat, شروع, انتها, g_ردیفها_انتها, DK, G, آرگومانها_2->n_ردیفها_q, &factx->div_G, d_حد, آرگومانها_2->q_transposed); اكسر;
                }
            }
        } وإلا {
            // Fallback: direct-from-DDR/L2 path
            hmx_fa_q_prep_fallback(q_tiles, q->داده, q->nb[1], q->nb[2], q->nb[3],
                                   q_شروع, kv_سر, ib3, شروع, انتها, n_ردیفها_g, G, DK, DK_pad, factx->is_q_fp32, &factx->div_G);
        }
    }

    // Synchronize threads before zeroing out vtcm_o_tiles[0] to prevent race condition
    إذا (n > 1) {
        atomic_fetch_sub(&آرگومانها_2->barrier, 1);
        طالما (atomic_بارگیری(&آرگومانها_2->barrier) > 0) {
            // spin wait
        }
    }

    // Zero out vtcm_o_tiles[0] as it was used as temp_q_vtcm
    {
        ثابت uint32_t g_br = factx->g_br;
        ثابت uint32_t DV   = factx->DV;
        ثابت size_t o_tile_بایتها  = factx->o_tile_بایتها;
        ثابت size_t o_بایتها_در_هر_t = hex_align_به_بالا(o_tile_بایتها / n, 128);
        ثابت size_t o_شروع       = i * o_بایتها_در_هر_t;
        ثابت size_t o_انتها         = hex_smin(o_شروع + o_بایتها_در_هر_t, o_tile_بایتها);
        إذا (o_شروع < o_tile_بایتها) {
            hvx_splat_u8_a((حرف *) factx->vtcm_o_tiles[0] + o_شروع, 0, o_انتها - o_شروع);
        }
    }
    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_FA_Q_PREP, (uint16_t) (آرگومانها_2->q_شروع * G + شروع));
}

ساكن فراغ fa_phase_q_بارگیری(بنية hmx_fa_context *   factx,
                            ثابت بنية htp_tensor * q,
                            uint32_t                  q_شروع,
                            uint32_t                  kv_سر,
                            uint32_t                  ib3,
                            size_t                    n_ردیفها_g) {
    work_صف_t wp = factx->octx->ctx->work_صف;
    uint32_t n = 1;
    إذا (factx->n_نخها > 1 && n_ردیفها_g >= (size_t) (factx->n_نخها * 2)) {
        n = factx->n_نخها;
    }
    size_t ردیفها_در_هر_t = hex_align_به_بالا(hmx_ceil_div(factx->g_br, n), 2);
    ثابت uint32_t n_ردیفها_q = hex_smin(factx->Br, factx->neq1 - q_شروع);
    fa_q_بارگیری_آرگومانها_t آرگومانها_2;
    آرگومانها_2.factx = factx;
    آرگومانها_2.q = q;
    آرگومانها_2.q_شروع = q_شروع;
    آرگومانها_2.kv_سر = kv_سر;
    آرگومانها_2.ib3 = ib3;
    آرگومانها_2.n_ردیفها_g = n_ردیفها_g;
    آرگومانها_2.ردیفها_در_هر_t = ردیفها_در_هر_t;
    آرگومانها_2.n_ردیفها_q = n_ردیفها_q;
    آرگومانها_2.q_transposed = q->nb[1] < q->nb[2];
    atomic_init(&آرگومانها_2.barrier, n);
    إذا (n > 1) {
        work_صف_پیاپی(wp, fa_q_بارگیری_thread, &آرگومانها_2, n);
    } وإلا {
        fa_q_بارگیری_thread(1, 0, &آرگومانها_2);
    }
}

تعريفنوع بنية {
    بنية hmx_fa_context *   factx;
    ثابت بنية htp_tensor * dst;
    ثابت __fp16 *            o_tile_src;
    uint32_t                  q_شروع;
    uint32_t                  kv_سر;
    uint32_t                  ib3;
    size_t                    n_ردیفها_g;
    size_t                    ردیفها_در_هر_t;
} fa_o_store_آرگومانها_t;

ساكن فراغ fa_o_store_thread_f32(بیعلامت صحيح n, بیعلامت صحيح i, فراغ * داده) {
    fa_o_store_آرگومانها_t *     آرگومانها_2  = (fa_o_store_آرگومانها_t *) داده;
    بنية hmx_fa_context * factx = آرگومانها_2->factx;

    ثابت size_t n_ردیفها_g = آرگومانها_2->n_ردیفها_g;
    ثابت size_t G        = factx->G;
    ثابت size_t DV       = factx->DV;

    ثابت size_t ردیفها_در_هر_t = آرگومانها_2->ردیفها_در_هر_t;
    ثابت size_t شروع      = (size_t) i * ردیفها_در_هر_t;
    ثابت size_t انتها        = hex_smin(شروع + ردیفها_در_هر_t, n_ردیفها_g);

    إذا (شروع >= n_ردیفها_g) {
        أرجع;
    }

    بنية htp_thread_trace * tr = &factx->octx->ctx->trace[i];
    htp_trace_event_شروع(tr, HTP_TRACE_EVT_HVX_O_PROC, (uint16_t) (آرگومانها_2->q_شروع * G + شروع));

    ثابت بنية htp_tensor * dst        = آرگومانها_2->dst;
    ثابت __fp16 *            o_tile_src = آرگومانها_2->o_tile_src;
    ثابت uint32_t            q_شروع    = آرگومانها_2->q_شروع;
    ثابت uint32_t            kv_سر    = آرگومانها_2->kv_سر;
    ثابت uint32_t            ib3        = آرگومانها_2->ib3;

    ثابت size_t DV_pad = factx->DV_pad;

    size_t q_idx = fastdiv(شروع, &factx->div_G);
    size_t h_idx = fastmodulo(شروع, G, &factx->div_G);

    لكل (size_t r = شروع; r < انتها; ++r) {
        عائم * بیرون = (عائم *) ((uint8_t *) dst->داده + (kv_سر * G + h_idx) * dst->nb[1] +
                                 (q_شروع + q_idx) * dst->nb[2] + ib3 * dst->nb[3]);

        size_t         r0            = r / HMX_FP16_TILE_N_ردیفها;
        size_t         r1            = r % HMX_FP16_TILE_N_ردیفها;
        ثابت __fp16 * tile_ردیف_پایه = o_tile_src + r0 * HMX_FP16_TILE_N_ردیفها * DV_pad;

        لكل (uint32_t d = 0; d < DV / 32; ++d) {
            ثابت HVX_وکتور * درون_tile = (ثابت HVX_وکتور *) (tile_ردیف_پایه + d * HMX_FP16_TILE_N_ELMS);
            HVX_وکتور_Pair     vp      = hvx_vec_f16_مقصد_f32_shuff(درون_tile[r1 / 2]);
            إذا (r1 % 2 == 0) {
                *(HVX_UVector *) (بیرون + d * 32) = Q6_V_lo_W(vp);
            } وإلا {
                *(HVX_UVector *) (بیرون + d * 32) = Q6_V_hi_W(vp);
            }
        }
        // Ragged tail: DV not a multiple of 32 (e.g. 72 -> last 8 lanes). Partial vector-write
        // for the remaining (DV % 32) floats.
        ثابت uint32_t d_دم = DV / 32;
        ثابت uint32_t rem    = DV - d_دم * 32;
        إذا (rem) {
            ثابت HVX_وکتور * درون_tile = (ثابت HVX_وکتور *) (tile_ردیف_پایه + d_دم * HMX_FP16_TILE_N_ELMS);
            HVX_وکتور_Pair     vp      = hvx_vec_f16_مقصد_f32_shuff(درون_tile[r1 / 2]);
            HVX_وکتور         vd      = (r1 % 2 == 0) ? Q6_V_lo_W(vp) : Q6_V_hi_W(vp);
            hvx_vec_store_u((فراغ *) (بیرون + d_دم * 32), rem * حجم(عائم), vd);
        }

        h_idx++;
        إذا (h_idx == G) {
            h_idx = 0;
            q_idx++;
        }
    }
    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_O_PROC, (uint16_t) (آرگومانها_2->q_شروع * G + شروع));
}

ساكن فراغ fa_o_store_thread_f16(بیعلامت صحيح n, بیعلامت صحيح i, فراغ * داده) {
    fa_o_store_آرگومانها_t *     آرگومانها_2  = (fa_o_store_آرگومانها_t *) داده;
    بنية hmx_fa_context * factx = آرگومانها_2->factx;

    ثابت size_t n_ردیفها_g   = آرگومانها_2->n_ردیفها_g;
    ثابت size_t ردیفها_در_هر_t = آرگومانها_2->ردیفها_در_هر_t;
    ثابت size_t G          = factx->G;
    ثابت size_t DV         = factx->DV;
    ثابت size_t شروع      = (size_t) i * ردیفها_در_هر_t;
    ثابت size_t انتها        = hex_smin(شروع + ردیفها_در_هر_t, n_ردیفها_g);

    إذا (شروع >= n_ردیفها_g) {
        أرجع;
    }

    بنية htp_thread_trace * tr = &factx->octx->ctx->trace[i];
    htp_trace_event_شروع(tr, HTP_TRACE_EVT_HVX_O_PROC, (uint16_t) (آرگومانها_2->q_شروع * G + شروع));

    ثابت بنية htp_tensor * dst        = آرگومانها_2->dst;
    ثابت __fp16 *            o_tile_src = آرگومانها_2->o_tile_src;
    ثابت uint32_t            q_شروع    = آرگومانها_2->q_شروع;
    ثابت uint32_t            kv_سر    = آرگومانها_2->kv_سر;
    ثابت uint32_t            ib3        = آرگومانها_2->ib3;

    // O-tiles use the padded head dim (DV_pad); dst holds the real DV lanes.
    ثابت size_t DV_pad = factx->DV_pad;

    size_t q_idx = fastdiv(شروع, &factx->div_G);
    size_t h_idx = fastmodulo(شروع, G, &factx->div_G);

    لكل (size_t r = شروع; r < انتها; ++r) {
        __fp16 * بیرون = (__fp16 *) ((uint8_t *) dst->داده + (kv_سر * G + h_idx) * dst->nb[1] +
                                   (q_شروع + q_idx) * dst->nb[2] + ib3 * dst->nb[3]);

        size_t         r0            = r / HMX_FP16_TILE_N_ردیفها;
        size_t         r1            = r % HMX_FP16_TILE_N_ردیفها;
        ثابت __fp16 * tile_ردیف_پایه = o_tile_src + r0 * HMX_FP16_TILE_N_ردیفها * DV_pad;

        لكل (uint32_t d = 0; d < DV / 64; ++d) {
            ثابت __fp16 *     درون_dtile = tile_ردیف_پایه + d * HMX_FP16_TILE_N_ELMS * 2;
            ثابت HVX_وکتور * pv_in0   = ((ثابت HVX_وکتور *) درون_dtile) + r1 / 2;
            ثابت HVX_وکتور * pv_in1   = pv_in0 + 16;
            HVX_وکتور_Pair     vp       = Q6_W_vdeal_VVR(*pv_in1, *pv_in0, -2);
            إذا (r1 % 2 == 0) {
                *(HVX_UVector *) (بیرون + d * 64) = Q6_V_lo_W(vp);
            } وإلا {
                *(HVX_UVector *) (بیرون + d * 64) = Q6_V_hi_W(vp);
            }
        }
        // Ragged tail when DV is not a multiple of 64.
        ثابت uint32_t d_دم = DV / 64;
        ثابت uint32_t rem    = DV - d_دم * 64;
        إذا (rem) {
            ثابت __fp16 *     درون_dtile = tile_ردیف_پایه + d_دم * HMX_FP16_TILE_N_ELMS * 2;
            ثابت HVX_وکتور * pv_in0   = ((ثابت HVX_وکتور *) درون_dtile) + r1 / 2;
            ثابت HVX_وکتور * pv_in1   = pv_in0 + 16;
            HVX_وکتور_Pair     vp       = Q6_W_vdeal_VVR(*pv_in1, *pv_in0, -2);
            HVX_وکتور         vd       = (r1 % 2 == 0) ? Q6_V_lo_W(vp) : Q6_V_hi_W(vp);
            hvx_vec_store_u((فراغ *) (بیرون + d_دم * 64), rem * حجم(__fp16), vd);
        }

        h_idx++;
        إذا (h_idx == G) {
            h_idx = 0;
            q_idx++;
        }
    }
    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_O_PROC, (uint16_t) (آرگومانها_2->q_شروع * G + شروع));
}

ساكن فراغ fa_phase_o_store(بنية hmx_fa_context *   factx,
                             ثابت بنية htp_tensor * dst,
                             ثابت __fp16 *            o_tile_src,
                             uint32_t                  q_شروع,
                             uint32_t                  kv_سر,
                             uint32_t                  ib3,
                             size_t                    n_ردیفها_g) {
    work_صف_t wp = factx->octx->ctx->work_صف;
    uint32_t n = 1;
    إذا (factx->n_نخها > 1 && n_ردیفها_g >= (size_t) (factx->n_نخها * 2)) {
        n = factx->n_نخها;
    }
    size_t ردیفها_در_هر_t = hmx_ceil_div(n_ردیفها_g, n);
    fa_o_store_آرگومانها_t آرگومانها_2 = { factx, dst, o_tile_src, q_شروع, kv_سر, ib3, n_ردیفها_g, ردیفها_در_هر_t };
    کارگر_callback_t store_fn = factx->is_dst_fp32 ? fa_o_store_thread_f32 : fa_o_store_thread_f16;
    إذا (n > 1) {
        work_صف_پیاپی(wp, store_fn, &آرگومانها_2, n);
    } وإلا {
        store_fn(1, 0, &آرگومانها_2);
    }
}

تعريفنوع بنية {
    بنية hmx_fa_context *   factx;
    size_t                    buf_idx;
    size_t                    kv_ردیفها;
    size_t                    n_ردیفها_g;
    size_t                    n_ستون_tiles;
    size_t                    n_tiles_در_هر_bc;
    size_t                    n_ردیف_tiles;
    size_t                    n_ردیف_tiles_g_br;
    uint32_t                  Bc;
    uint32_t                  G;
    uint32_t                  kv_سر;
    uint32_t                  kv_شروع;
    uint32_t                  q_شروع;
    uint32_t                  ib3;
    bool                      has_alibi;  // true when max_bias != 0 (need slope * mask + add)
    __fp16 *                  slopes;
    ثابت بنية htp_tensor * ماسک;
    ثابت __fp16 *            ماسک_vtcm;             // VTCM mask buffer base (NULL = DDR fallback)
    size_t                    ماسک_vtcm_ردیف_stride;  // elements (__fp16) per row in VTCM mask buffer
    بنية fastdiv_مقادیر     thread_div;
} fa_softmax_آرگومانها_t;

ساكن مضمن فراغ fa_softmax_impl(
    بیعلامت صحيح n, بیعلامت صحيح i, فراغ * داده,
    ثابت bool has_ماسک,
    ثابت bool ماسک_پخش,
    ثابت bool is_g1,
    ثابت bool has_alibi,
    ثابت bool has_softcap
) {
    fa_softmax_آرگومانها_t *     آرگومانها_2  = (fa_softmax_آرگومانها_t *) داده;
    بنية hmx_fa_context * factx = آرگومانها_2->factx;

    ثابت size_t n_ردیفها_g       = آرگومانها_2->n_ردیفها_g;
    ثابت size_t kv_ردیفها        = آرگومانها_2->kv_ردیفها;
    ثابت size_t Bc             = آرگومانها_2->Bc;
    ثابت size_t G              = آرگومانها_2->G;
    ثابت size_t n_tiles_در_هر_bc = آرگومانها_2->n_tiles_در_هر_bc;
    ثابت size_t n_ردیف_vec_cnt  = hmx_ceil_div(n_ردیفها_g, 64);
    ثابت uint32_t im3          = has_ماسک ? fastmodulo(آرگومانها_2->ib3, آرگومانها_2->ماسک->ne[3], &factx->src3_div3) : 0;

    size_t vec_شروع = 0;
    size_t vec_انتها   = n_ردیف_vec_cnt;
    إذا (n > 1) {
        ثابت size_t vecs_در_هر_t = fastdiv(n_ردیف_vec_cnt + n - 1, &آرگومانها_2->thread_div);
        vec_شروع = i * vecs_در_هر_t;
        vec_انتها   = hex_smin(vec_شروع + vecs_در_هر_t, n_ردیف_vec_cnt);
    }

    إذا (vec_شروع >= n_ردیف_vec_cnt) {
        أرجع;
    }

    بنية htp_thread_trace * tr = &factx->octx->ctx->trace[i];
    htp_trace_event_شروع(tr, HTP_TRACE_EVT_HVX_FA_SFM, (uint16_t) (آرگومانها_2->q_شروع * G + vec_شروع * 64));

    // Per-thread row scratch: thread i uses bufs at offset i * 2 * stride
    ثابت size_t ردیف_buf_stride = factx->ردیف_buf_stride;
    HVX_وکتور * my_ردیف_buf0    = factx->vtcm_ردیف_bufs + i * 2 * ردیف_buf_stride;
    HVX_وکتور * my_ردیف_buf1    = my_ردیف_buf0 + ردیف_buf_stride;

    ثابت HVX_وکتور v_neg_inf = Q6_Vh_vsplat_R(0xfbff);

    لكل (size_t r_vec_idx = vec_شروع; r_vec_idx < vec_انتها; ++r_vec_idx) {
        HVX_وکتور rowmax_acc_v = v_neg_inf;
        HVX_وکتور rowsum_acc_v = Q6_V_vzero();
        HVX_وکتور m_prev_v0    = factx->vtcm_m_vec[r_vec_idx * 2 + 0];
        HVX_وکتور m_prev_v1    = factx->vtcm_m_vec[r_vec_idx * 2 + 1];

        HVX_وکتور v_slopes = Q6_V_vzero();
        إذا (has_alibi) {
            v_slopes = hvx_vmem(آرگومانها_2->slopes + r_vec_idx * 64);
        }

        لكل (uint32_t r_vec_off = 0; r_vec_off < 64; r_vec_off += 2) {
            uint32_t r = r_vec_idx * 64 + r_vec_off;
            إذا (r >= hex_align_به_بالا(n_ردیفها_g, 2)) {
                اكسر;
            }

            uint32_t r0 = r / HMX_FP16_TILE_N_ردیفها;
            uint32_t r1 = r % HMX_FP16_TILE_N_ردیفها;

            ثابت __fp16 * s_ld_پایه = factx->vtcm_s_tiles[آرگومانها_2->buf_idx] + r0 * HMX_FP16_TILE_N_ردیفها * Bc;
            __fp16 *       p_st_پایه = factx->vtcm_p_tiles[آرگومانها_2->buf_idx] + r0 * HMX_FP16_TILE_N_ردیفها * Bc;

            // Decode 2 rows from S tiles into per-thread row buffers
            إذا (has_softcap) {
                ثابت HVX_وکتور v_cap = hvx_vec_splat_f16(factx->logit_softcap);
                لكل (size_t c = 0; c < kv_ردیفها; c += 64) {
                    size_t             ci       = c / 64;
                    ثابت __fp16 *     درون_dtile = s_ld_پایه + ci * HMX_FP16_TILE_N_ELMS * 2;
                    ثابت HVX_وکتور * pv_s_in0 = ((ثابت HVX_وکتور *) درون_dtile) + r1 / 2;
                    ثابت HVX_وکتور * pv_s_in1 = pv_s_in0 + 16;

                    HVX_وکتور_Pair vp_s_drow = Q6_W_vdeal_VVR(*pv_s_in1, *pv_s_in0, -2);
                    HVX_وکتور     v_s_row0  = Q6_V_lo_W(vp_s_drow);
                    HVX_وکتور     v_s_row1  = Q6_V_hi_W(vp_s_drow);

                    HVX_وکتور t0   = hvx_vec_tanh_f16(v_s_row0);
                    my_ردیف_buf0[ci] = hvx_vec_mul_f16_f16(t0, v_cap);

                    HVX_وکتور t1   = hvx_vec_tanh_f16(v_s_row1);
                    my_ردیف_buf1[ci] = hvx_vec_mul_f16_f16(t1, v_cap);
                }
            } وإلا {
                size_t c = 0;
                لكل (; c + 64 < kv_ردیفها; c += 128) {
                    size_t             ci0       = c / 64;
                    size_t             ci1       = ci0 + 1;
                    ثابت __fp16 *     درون_dtile0 = s_ld_پایه + ci0 * HMX_FP16_TILE_N_ELMS * 2;
                    ثابت __fp16 *     درون_dtile1 = s_ld_پایه + ci1 * HMX_FP16_TILE_N_ELMS * 2;
                    ثابت HVX_وکتور * pv_s_in0_0 = ((ثابت HVX_وکتور *) درون_dtile0) + r1 / 2;
                    ثابت HVX_وکتور * pv_s_in1_0 = pv_s_in0_0 + 16;
                    ثابت HVX_وکتور * pv_s_in0_1 = ((ثابت HVX_وکتور *) درون_dtile1) + r1 / 2;
                    ثابت HVX_وکتور * pv_s_in1_1 = pv_s_in0_1 + 16;

                    HVX_وکتور_Pair vp_s_drow0 = Q6_W_vdeal_VVR(*pv_s_in1_0, *pv_s_in0_0, -2);
                    my_ردیف_buf0[ci0]          = Q6_V_lo_W(vp_s_drow0);
                    my_ردیف_buf1[ci0]          = Q6_V_hi_W(vp_s_drow0);

                    HVX_وکتور_Pair vp_s_drow1 = Q6_W_vdeal_VVR(*pv_s_in1_1, *pv_s_in0_1, -2);
                    my_ردیف_buf0[ci1]          = Q6_V_lo_W(vp_s_drow1);
                    my_ردیف_buf1[ci1]          = Q6_V_hi_W(vp_s_drow1);
                }
                لكل (; c < kv_ردیفها; c += 64) {
                    size_t             ci       = c / 64;
                    ثابت __fp16 *     درون_dtile = s_ld_پایه + ci * HMX_FP16_TILE_N_ELMS * 2;
                    ثابت HVX_وکتور * pv_s_in0 = ((ثابت HVX_وکتور *) درون_dtile) + r1 / 2;
                    ثابت HVX_وکتور * pv_s_in1 = pv_s_in0 + 16;

                    HVX_وکتور_Pair vp_s_drow = Q6_W_vdeal_VVR(*pv_s_in1, *pv_s_in0, -2);
                    my_ردیف_buf0[ci]          = Q6_V_lo_W(vp_s_drow);
                    my_ردیف_buf1[ci]          = Q6_V_hi_W(vp_s_drow);
                }
            }

            // Apply mask & compute rowmax(S)
            HVX_وکتور v_slope0 = Q6_V_vzero();
            HVX_وکتور v_slope1 = Q6_V_vzero();
            إذا (has_alibi) {
                v_slope0 = hvx_vec_repl_f16(Q6_V_vror_VR(v_slopes, r_vec_off * 2));
                v_slope1 = (r + 1 < n_ردیفها_g) ? hvx_vec_repl_f16(Q6_V_vror_VR(v_slopes, (r_vec_off + 1) * 2)) : Q6_V_vzero();
            }

            ثابت HVX_وکتور v_آستانه = Q6_Vh_vsplat_R(0xcc00);  // fp16 -16.0

            HVX_وکتور v_s_rowmax0 = v_neg_inf;
            HVX_وکتور v_s_rowmax1 = v_neg_inf;
            إذا (has_ماسک) {
                لكل (size_t c = 0; c < kv_ردیفها; c += 64) {
                    size_t         ci          = c / 64;
                    ثابت size_t   ne          = hex_smin(kv_ردیفها - c, 64);
                    HVX_وکتور_Pred q_دم_keep = Q6_Q_vsetq2_R(ne * حجم(__fp16));

                    HVX_وکتور v_mask0, v_mask1;

                    إذا (ماسک_پخش) {
                        إذا (is_g1) {
                            ثابت size_t qi0 = r + 0;
                            v_mask0 = *(ثابت HVX_وکتور *) (آرگومانها_2->ماسک_vtcm + qi0 * آرگومانها_2->ماسک_vtcm_ردیف_stride + c);
                            v_mask1 = v_neg_inf;
                            إذا (r + 1 < n_ردیفها_g) {
                                ثابت size_t qi1 = r + 1;
                                v_mask1 = *(ثابت HVX_وکتور *) (آرگومانها_2->ماسک_vtcm + qi1 * آرگومانها_2->ماسک_vtcm_ردیف_stride + c);
                            }
                        } وإلا {
                            ثابت size_t qi0 = fastdiv(r + 0, &factx->div_G);
                            v_mask0 = *(ثابت HVX_وکتور *) (آرگومانها_2->ماسک_vtcm + qi0 * آرگومانها_2->ماسک_vtcm_ردیف_stride + c);
                            v_mask1 = v_neg_inf;
                            إذا (r + 1 < n_ردیفها_g) {
                                ثابت size_t qi1 = fastdiv(r + 1, &factx->div_G);
                                إذا (qi1 == qi0) {
                                    v_mask1 = v_mask0;
                                } وإلا {
                                    v_mask1 = *(ثابت HVX_وکتور *) (آرگومانها_2->ماسک_vtcm + qi1 * آرگومانها_2->ماسک_vtcm_ردیف_stride + c);
                                }
                            }
                        }
                    } وإلا {
                        // Head-dependent mask: pre-interleaved per row r.
                        ثابت size_t r0 = r + 0;
                        v_mask0 = *(ثابت HVX_وکتور *) (آرگومانها_2->ماسک_vtcm + r0 * آرگومانها_2->ماسک_vtcm_ردیف_stride + c);
                        v_mask1 = v_neg_inf;
                        إذا (r + 1 < n_ردیفها_g) {
                            ثابت size_t r1 = r + 1;
                            v_mask1 = *(ثابت HVX_وکتور *) (آرگومانها_2->ماسک_vtcm + r1 * آرگومانها_2->ماسک_vtcm_ردیف_stride + c);
                        }
                    }

                    // Threshold: mask values below -16.0 are treated as -inf (causal mask).
                    HVX_وکتور_Pred q_keep0 = Q6_Q_and_QQ(Q6_Q_vcmp_gt_VhfVhf(v_mask0, v_آستانه), q_دم_keep);
                    HVX_وکتور_Pred q_keep1 = Q6_Q_and_QQ(Q6_Q_vcmp_gt_VhfVhf(v_mask1, v_آستانه), q_دم_keep);

                    // Scale mask values by log2(e) for base-2 calculations
                    ثابت HVX_وکتور v_log2e = hvx_vec_splat_f16(EXP_LOG2E_F);
                    HVX_وکتور v_mask0_scaled = hvx_vec_mul_f16_f16(v_mask0, v_log2e);
                    HVX_وکتور v_mask1_scaled = hvx_vec_mul_f16_f16(v_mask1, v_log2e);

                    إذا (has_alibi) {
                        HVX_وکتور v_sm0 = hvx_vec_mul_f16_f16(v_mask0_scaled, v_slope0);
                        HVX_وکتور v_sm1 = hvx_vec_mul_f16_f16(v_mask1_scaled, v_slope1);
                        my_ردیف_buf0[ci]  = Q6_V_vmux_QVV(q_keep0, hvx_vec_جمع_کن_f16_f16(my_ردیف_buf0[ci], v_sm0), v_neg_inf);
                        my_ردیف_buf1[ci]  = Q6_V_vmux_QVV(q_keep1, hvx_vec_جمع_کن_f16_f16(my_ردیف_buf1[ci], v_sm1), v_neg_inf);
                    } وإلا {
                        my_ردیف_buf0[ci] = Q6_V_vmux_QVV(q_keep0, hvx_vec_جمع_کن_f16_f16(my_ردیف_buf0[ci], v_mask0_scaled), v_neg_inf);
                        my_ردیف_buf1[ci] = Q6_V_vmux_QVV(q_keep1, hvx_vec_جمع_کن_f16_f16(my_ردیف_buf1[ci], v_mask1_scaled), v_neg_inf);
                    }

                    v_s_rowmax0 = Q6_Vhf_vmax_VhfVhf(v_s_rowmax0, my_ردیف_buf0[ci]);
                    v_s_rowmax1 = Q6_Vhf_vmax_VhfVhf(v_s_rowmax1, my_ردیف_buf1[ci]);
                }
            } وإلا {
                size_t c = 0;
                لكل (; c + 64 < kv_ردیفها; c += 128) {
                    size_t ci0 = c / 64;
                    size_t ci1 = ci0 + 1;
                    v_s_rowmax0 = Q6_Vhf_vmax_VhfVhf(v_s_rowmax0, my_ردیف_buf0[ci0]);
                    v_s_rowmax1 = Q6_Vhf_vmax_VhfVhf(v_s_rowmax1, my_ردیف_buf1[ci0]);
                    v_s_rowmax0 = Q6_Vhf_vmax_VhfVhf(v_s_rowmax0, my_ردیف_buf0[ci1]);
                    v_s_rowmax1 = Q6_Vhf_vmax_VhfVhf(v_s_rowmax1, my_ردیف_buf1[ci1]);
                }
                لكل (; c < kv_ردیفها; c += 64) {
                    size_t         ci          = c / 64;
                    ثابت size_t   ne          = hex_smin(kv_ردیفها - c, 64);
                    HVX_وکتور_Pred q_دم_keep = Q6_Q_vsetq2_R(ne * حجم(__fp16));
                    إذا (ne < 64) {
                        my_ردیف_buf0[ci] = Q6_V_vmux_QVV(q_دم_keep, my_ردیف_buf0[ci], v_neg_inf);
                        my_ردیف_buf1[ci] = Q6_V_vmux_QVV(q_دم_keep, my_ردیف_buf1[ci], v_neg_inf);
                    }
                    v_s_rowmax0 = Q6_Vhf_vmax_VhfVhf(v_s_rowmax0, my_ردیف_buf0[ci]);
                    v_s_rowmax1 = Q6_Vhf_vmax_VhfVhf(v_s_rowmax1, my_ردیف_buf1[ci]);
                }
            }

            v_s_rowmax0 = hvx_vec_reduce_max_f16(v_s_rowmax0);
            v_s_rowmax1 = hvx_vec_reduce_max_f16(v_s_rowmax1);

            // Splat m_prev[r], m_prev[r+1] from the float per-row accumulators and convert to fp16 vectors
            HVX_وکتور v_m_prev0, v_m_prev1;
            إذا (r_vec_off < 32) {
                HVX_وکتور v0 = hvx_vec_repl_f32(Q6_V_vror_VR(m_prev_v0, r_vec_off * 4));
                v_m_prev0 = hvx_vec_f32_مقصد_f16(v0, v0);
                إذا (r + 1 < n_ردیفها_g) {
                    HVX_وکتور v1 = hvx_vec_repl_f32(Q6_V_vror_VR(m_prev_v0, (r_vec_off + 1) * 4));
                    v_m_prev1 = hvx_vec_f32_مقصد_f16(v1, v1);
                } وإلا {
                    v_m_prev1 = Q6_V_vzero();
                }
            } وإلا {
                HVX_وکتور v0 = hvx_vec_repl_f32(Q6_V_vror_VR(m_prev_v1, (r_vec_off - 32) * 4));
                v_m_prev0 = hvx_vec_f32_مقصد_f16(v0, v0);
                إذا (r + 1 < n_ردیفها_g) {
                    HVX_وکتور v1 = hvx_vec_repl_f32(Q6_V_vror_VR(m_prev_v1, (r_vec_off + 1 - 32) * 4));
                    v_m_prev1 = hvx_vec_f32_مقصد_f16(v1, v1);
                } وإلا {
                    v_m_prev1 = Q6_V_vzero();
                }
            }

            HVX_وکتور v_dup_m0 = Q6_Vhf_vmax_VhfVhf(v_m_prev0, v_s_rowmax0);
            HVX_وکتور v_dup_m1 = Q6_Vhf_vmax_VhfVhf(v_m_prev1, v_s_rowmax1);

            // Insert row r, r+1 rowmax into rowmax_acc_v
            {
                HVX_وکتور_Pred p_شروع = Q6_Q_vsetq_R(r_vec_off * 2);
                HVX_وکتور_Pred p_mid   = Q6_Q_vsetq_R((r_vec_off + 1) * 2);
                HVX_وکتور_Pred p_انتها   = Q6_Q_vsetq2_R((r_vec_off + 2) * 2);
                HVX_وکتور_Pred p_lane0 = Q6_Q_and_QQn(p_mid, p_شروع);
                HVX_وکتور_Pred p_lane1 = Q6_Q_and_QQn(p_انتها, p_mid);
                rowmax_acc_v           = Q6_V_vmux_QVV(p_lane0, v_dup_m0, rowmax_acc_v);
                rowmax_acc_v           = Q6_V_vmux_QVV(p_lane1, v_dup_m1, rowmax_acc_v);
            }

            // Compute P = exp(S - m_new)
            ثابت HVX_وکتور v_zero      = Q6_V_vzero();
            HVX_وکتور       v_p_rowsum0 = v_zero;
            HVX_وکتور       v_p_rowsum1 = v_zero;

            size_t c = 0;
            لكل (; c + 64 < kv_ردیفها; c += 128) {
                size_t     ci0          = c / 64;
                size_t     ci1          = ci0 + 1;

                HVX_وکتور v_s_minus_m0_0 = Q6_Vqf16_vsub_VhfVhf(my_ردیف_buf0[ci0], v_dup_m0);
                HVX_وکتور v_s_minus_m1_0 = Q6_Vqf16_vsub_VhfVhf(my_ردیف_buf1[ci0], v_dup_m1);
                HVX_وکتور v_s_minus_m0_1 = Q6_Vqf16_vsub_VhfVhf(my_ردیف_buf0[ci1], v_dup_m0);
                HVX_وکتور v_s_minus_m1_1 = Q6_Vqf16_vsub_VhfVhf(my_ردیف_buf1[ci1], v_dup_m1);

                HVX_وکتور v_p_row0_hf_0  = hvx_vec_exp2_f16(Q6_Vhf_equals_Vqf16(v_s_minus_m0_0));
                HVX_وکتور v_p_row1_hf_0  = hvx_vec_exp2_f16(Q6_Vhf_equals_Vqf16(v_s_minus_m1_0));
                HVX_وکتور v_p_row0_hf_1  = hvx_vec_exp2_f16(Q6_Vhf_equals_Vqf16(v_s_minus_m0_1));
                HVX_وکتور v_p_row1_hf_1  = hvx_vec_exp2_f16(Q6_Vhf_equals_Vqf16(v_s_minus_m1_1));

                __fp16 *     بیرون_dtile0  = p_st_پایه + ci0 * HMX_FP16_TILE_N_ELMS * 2;
                __fp16 *     بیرون_dtile1  = p_st_پایه + ci1 * HMX_FP16_TILE_N_ELMS * 2;
                HVX_وکتور * pv_p_out0_0  = ((HVX_وکتور *) بیرون_dtile0) + r1 / 2;
                HVX_وکتور * pv_p_out1_0  = pv_p_out0_0 + 16;
                HVX_وکتور * pv_p_out0_1  = ((HVX_وکتور *) بیرون_dtile1) + r1 / 2;
                HVX_وکتور * pv_p_out1_1  = pv_p_out0_1 + 16;

                HVX_وکتور_Pair vp_p_dual0 = Q6_W_vshuff_VVR(v_p_row1_hf_0, v_p_row0_hf_0, -2);
                *pv_p_out0_0               = Q6_V_lo_W(vp_p_dual0);
                *pv_p_out1_0               = Q6_V_hi_W(vp_p_dual0);

                HVX_وکتور_Pair vp_p_dual1 = Q6_W_vshuff_VVR(v_p_row1_hf_1, v_p_row0_hf_1, -2);
                *pv_p_out0_1               = Q6_V_lo_W(vp_p_dual1);
                *pv_p_out1_1               = Q6_V_hi_W(vp_p_dual1);

                HVX_وکتور_Pair vp_p0_0 = hvx_vec_f16_مقصد_f32_shuff(v_p_row0_hf_0);
                HVX_وکتور_Pair vp_p1_0 = hvx_vec_f16_مقصد_f32_shuff(v_p_row1_hf_0);
                HVX_وکتور_Pair vp_p0_1 = hvx_vec_f16_مقصد_f32_shuff(v_p_row0_hf_1);
                HVX_وکتور_Pair vp_p1_1 = hvx_vec_f16_مقصد_f32_shuff(v_p_row1_hf_1);

                v_p_rowsum0 = Q6_Vqf32_vadd_Vqf32Vqf32(v_p_rowsum0, Q6_Vqf32_vadd_VsfVsf(Q6_V_lo_W(vp_p0_0), Q6_V_hi_W(vp_p0_0)));
                v_p_rowsum0 = Q6_Vqf32_vadd_Vqf32Vqf32(v_p_rowsum0, Q6_Vqf32_vadd_VsfVsf(Q6_V_lo_W(vp_p0_1), Q6_V_hi_W(vp_p0_1)));
                v_p_rowsum1 = Q6_Vqf32_vadd_Vqf32Vqf32(v_p_rowsum1, Q6_Vqf32_vadd_VsfVsf(Q6_V_lo_W(vp_p1_0), Q6_V_hi_W(vp_p1_0)));
                v_p_rowsum1 = Q6_Vqf32_vadd_Vqf32Vqf32(v_p_rowsum1, Q6_Vqf32_vadd_VsfVsf(Q6_V_lo_W(vp_p1_1), Q6_V_hi_W(vp_p1_1)));
            }
            لكل (size_t c_rem = c; c_rem < kv_ردیفها; c_rem += 64) {
                size_t     ci           = c_rem / 64;
                HVX_وکتور v_s_minus_m0 = Q6_Vqf16_vsub_VhfVhf(my_ردیف_buf0[ci], v_dup_m0);
                HVX_وکتور v_s_minus_m1 = Q6_Vqf16_vsub_VhfVhf(my_ردیف_buf1[ci], v_dup_m1);

                HVX_وکتور v_p_row0_hf  = hvx_vec_exp2_f16(Q6_Vhf_equals_Vqf16(v_s_minus_m0));
                HVX_وکتور v_p_row1_hf  = hvx_vec_exp2_f16(Q6_Vhf_equals_Vqf16(v_s_minus_m1));
                __fp16 *     بیرون_dtile  = p_st_پایه + ci * HMX_FP16_TILE_N_ELMS * 2;
                HVX_وکتور * pv_p_out0  = ((HVX_وکتور *) بیرون_dtile) + r1 / 2;
                HVX_وکتور * pv_p_out1  = pv_p_out0 + 16;

                HVX_وکتور_Pair vp_p_dual = Q6_W_vshuff_VVR(v_p_row1_hf, v_p_row0_hf, -2);
                *pv_p_out0               = Q6_V_lo_W(vp_p_dual);
                *pv_p_out1               = Q6_V_hi_W(vp_p_dual);

                HVX_وکتور_Pair vp_p0 = hvx_vec_f16_مقصد_f32_shuff(v_p_row0_hf);
                HVX_وکتور_Pair vp_p1 = hvx_vec_f16_مقصد_f32_shuff(v_p_row1_hf);

                v_p_rowsum0 = Q6_Vqf32_vadd_Vqf32Vqf32(v_p_rowsum0, Q6_Vqf32_vadd_VsfVsf(Q6_V_lo_W(vp_p0), Q6_V_hi_W(vp_p0)));
                v_p_rowsum1 = Q6_Vqf32_vadd_Vqf32Vqf32(v_p_rowsum1, Q6_Vqf32_vadd_VsfVsf(Q6_V_lo_W(vp_p1), Q6_V_hi_W(vp_p1)));
            }

            HVX_وکتور rowsum0_sf = hvx_vec_reduce_جمع_f32(Q6_Vsf_equals_Vqf32(v_p_rowsum0));
            HVX_وکتور rowsum1_sf = hvx_vec_reduce_جمع_f32(Q6_Vsf_equals_Vqf32(v_p_rowsum1));
            {
                HVX_وکتور rv0_v = hvx_vec_f32_مقصد_f16(rowsum0_sf, rowsum0_sf);
                HVX_وکتور rv1_v = hvx_vec_f32_مقصد_f16(rowsum1_sf, rowsum1_sf);

                HVX_وکتور_Pred p_شروع = Q6_Q_vsetq_R(r_vec_off * 2);
                HVX_وکتور_Pred p_mid   = Q6_Q_vsetq_R((r_vec_off + 1) * 2);
                HVX_وکتور_Pred p_انتها   = Q6_Q_vsetq2_R((r_vec_off + 2) * 2);
                HVX_وکتور_Pred p_lane0 = Q6_Q_and_QQn(p_mid, p_شروع);
                HVX_وکتور_Pred p_lane1 = Q6_Q_and_QQn(p_انتها, p_mid);
                rowsum_acc_v           = Q6_V_vmux_QVV(p_lane0, rv0_v, rowsum_acc_v);
                rowsum_acc_v           = Q6_V_vmux_QVV(p_lane1, rv1_v, rowsum_acc_v);
            }
        }

        // Inline fa_ml_update_and_build_d for this vector (lock-free and in parallel)
        HVX_وکتور_Pair rowmax_acc_pair    = hvx_vec_f16_مقصد_f32(rowmax_acc_v);
        HVX_وکتور     v_rowmax_acc_f32_0 = Q6_V_lo_W(rowmax_acc_pair);
        HVX_وکتور     v_rowmax_acc_f32_1 = Q6_V_hi_W(rowmax_acc_pair);

        HVX_وکتور v_m_curr0 = Q6_Vsf_vmax_VsfVsf(m_prev_v0, v_rowmax_acc_f32_0);
        HVX_وکتور v_m_curr1 = Q6_Vsf_vmax_VsfVsf(m_prev_v1, v_rowmax_acc_f32_1);

        HVX_وکتور v_m_diff0 = HVX_OP_SUB_F32(m_prev_v0, v_m_curr0);
        HVX_وکتور v_m_diff1 = HVX_OP_SUB_F32(m_prev_v1, v_m_curr1);

        HVX_وکتور v_m_diff_f16   = hvx_vec_f32_مقصد_f16(v_m_diff0, v_m_diff1);
        HVX_وکتور exp_m_diff_f16 = hvx_vec_exp2_f16(v_m_diff_f16);

        HVX_وکتور_Pair exp_m_diff_pair = hvx_vec_f16_مقصد_f32(exp_m_diff_f16);
        HVX_وکتور exp_m_diff0 = Q6_V_lo_W(exp_m_diff_pair);
        HVX_وکتور exp_m_diff1 = Q6_V_hi_W(exp_m_diff_pair);

        HVX_وکتور_Pair rowsum_acc_pair = hvx_vec_f16_مقصد_f32(rowsum_acc_v);
        HVX_وکتور     v_rowsum_acc_f32_0 = Q6_V_lo_W(rowsum_acc_pair);
        HVX_وکتور     v_rowsum_acc_f32_1 = Q6_V_hi_W(rowsum_acc_pair);

        HVX_وکتور l_prev_v0 = factx->vtcm_l_vec[r_vec_idx * 2 + 0];
        HVX_وکتور l_prev_v1 = factx->vtcm_l_vec[r_vec_idx * 2 + 1];
        HVX_وکتور v_l_curr0 = HVX_OP_جمع_کن_F32(HVX_OP_MUL_F32(l_prev_v0, exp_m_diff0), v_rowsum_acc_f32_0);
        HVX_وکتور v_l_curr1 = HVX_OP_جمع_کن_F32(HVX_OP_MUL_F32(l_prev_v1, exp_m_diff1), v_rowsum_acc_f32_1);

        factx->vtcm_m_vec[r_vec_idx * 2 + 0] = v_m_curr0;
        factx->vtcm_m_vec[r_vec_idx * 2 + 1] = v_m_curr1;
        factx->vtcm_l_vec[r_vec_idx * 2 + 0] = v_l_curr0;
        factx->vtcm_l_vec[r_vec_idx * 2 + 1] = v_l_curr1;

        // Build diagonal tile D = diag(exp(m_diff))
        ثابت HVX_وکتور     v_offsets = *(ثابت HVX_وکتور *) d_tile_scatter_offsets;
        ثابت HVX_وکتور_Pred q_32_ماسک = Q6_Q_vsetq_R(32 * حجم(__fp16));
        HVX_وکتور           v_exp_m_diff = exp_m_diff_f16;

        __fp16 * ثابت d_tiles_بیرون = factx->vtcm_d_tiles[آرگومانها_2->buf_idx];

        size_t t0 = r_vec_idx * 2;
        إذا (t0 < آرگومانها_2->n_ردیف_tiles) {
            ثابت HVX_وکتور v_content = v_exp_m_diff;
            __fp16 *         بیرون_پایه  = d_tiles_بیرون + t0 * HMX_FP16_TILE_N_ELMS;
            Q6_vscatter_QRMVhV(q_32_ماسک, (size_t) بیرون_پایه, HMX_FP16_TILE_اندازه - 1, v_offsets, v_content);
        }

        size_t t1 = r_vec_idx * 2 + 1;
        إذا (t1 < آرگومانها_2->n_ردیف_tiles) {
            ثابت HVX_وکتور v_content = Q6_V_vror_VR(v_exp_m_diff, 64);
            __fp16 *         بیرون_پایه  = d_tiles_بیرون + t1 * HMX_FP16_TILE_N_ELMS;
            Q6_vscatter_QRMVhV(q_32_ماسک, (size_t) بیرون_پایه, HMX_FP16_TILE_اندازه - 1, v_offsets, v_content);
        }
    }
    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_FA_SFM, (uint16_t) (آرگومانها_2->q_شروع * G + vec_شروع * 64));
}

ساكن فراغ fa_softmax_thread_nomask(بیعلامت صحيح n, بیعلامت صحيح i, فراغ * داده) {
    fa_softmax_impl(n, i, داده,
                    /*has_mask=*/false,
                    /*mask_broadcast=*/false,
                    /*is_g1=*/false,
                    /*has_alibi=*/false,
                    /*has_softcap=*/false);
}

ساكن فراغ fa_softmax_thread_ماسک_پخش_g1(بیعلامت صحيح n, بیعلامت صحيح i, فراغ * داده) {
    fa_softmax_impl(n, i, داده,
                    /*has_mask=*/true,
                    /*mask_broadcast=*/true,
                    /*is_g1=*/true,
                    /*has_alibi=*/false,
                    /*has_softcap=*/false);
}

ساكن فراغ fa_softmax_thread_ماسک_پخش_gn(بیعلامت صحيح n, بیعلامت صحيح i, فراغ * داده) {
    fa_softmax_impl(n, i, داده,
                    /*has_mask=*/true,
                    /*mask_broadcast=*/true,
                    /*is_g1=*/false,
                    /*has_alibi=*/false,
                    /*has_softcap=*/false);
}

ساكن فراغ fa_softmax_thread(بیعلامت صحيح n, بیعلامت صحيح i, فراغ * داده) {
    fa_softmax_آرگومانها_t *     آرگومانها_2  = (fa_softmax_آرگومانها_t *) داده;
    بنية hmx_fa_context * factx = آرگومانها_2->factx;

    ثابت bool has_ماسک       = (آرگومانها_2->ماسک != عدم);
    ثابت bool ماسک_پخش = factx->ماسک_پخش;
    ثابت bool is_g1          = (آرگومانها_2->G == 1);
    ثابت bool has_alibi      = آرگومانها_2->has_alibi;
    ثابت bool has_softcap    = factx->has_softcap;

    fa_softmax_impl(n, i, داده, has_ماسک, ماسک_پخش, is_g1, has_alibi, has_softcap);
}

ساكن مضمن HVX_وکتور hvx_fa_بارگیری_sinks(ثابت عائم * sinks_داده, size_t idx, size_t شمار) {
    ثابت HVX_وکتور * v_src    = (ثابت HVX_وکتور *) sinks_داده + (idx >> 5);
    ثابت size_t       lane_off = idx & 31;
    ثابت size_t       byte_off = lane_off * حجم(عائم);
    أرجع (lane_off + شمار > 32)
        ? Q6_V_valign_VVR(v_src[1], v_src[0], byte_off)
        : Q6_V_vror_VR(v_src[0], byte_off);
}

ساكن __attribute__((noinline)) فراغ fa_build_d_diag_inv_l(بنية hmx_fa_context * factx,
                                                            size_t                  n_ردیف_tiles,
                                                            size_t                  n_ردیف_tiles_g_br,
                                                            uint32_t                kv_سر,
                                                            size_t                  n_ردیفها_g) {
    (فراغ) n_ردیف_tiles_g_br;

    // sinks: l += exp(sink - m * ln2) in f32, m and l are in the log2 domain
    إذا (factx->sinks) {
        ثابت عائم *    sinks_داده   = factx->vtcm_sinks;
        ثابت size_t     G            = factx->G;
        ثابت size_t     سر_پایه    = (size_t) kv_سر * G;
        ثابت HVX_وکتور v_ln2        = Q6_V_vsplat_R(EXP_LOGN2);
        ثابت HVX_وکتور v_سر_sinks = (G <= 32) ? hvx_fa_بارگیری_sinks(sinks_داده, سر_پایه, G) : Q6_V_vzero();
        HVX_وکتور       v_sink       = Q6_V_vzero();
        size_t           g_prev       = SIZE_MAX;

        لكل (size_t i = 0; i < n_ردیف_tiles; ++i) {
            // the sink pattern of a tile only depends on (i * 32) % G, rebuild it when that changes
            ثابت size_t r0 = i * 32;
            ثابت size_t g0 = fastmodulo(r0, G, &factx->div_G);
            إذا (g0 != g_prev) {
                ثابت size_t rem_g = G - g0;
                إذا (G <= 32) {
                    ثابت HVX_وکتور v_lo = Q6_V_vror_VR(v_سر_sinks, g0 * حجم(عائم));
                    ثابت HVX_وکتور v_hi = Q6_V_vror_VR(v_سر_sinks, 128 - rem_g * حجم(عائم));
                    v_sink = Q6_V_vmux_QVV(Q6_Q_vsetq2_R(rem_g * حجم(عائم)), v_lo, v_hi);
                    لكل (size_t len = G; len < 32; len <<= 1) {
                        ثابت HVX_وکتور_Pred q_keep    = Q6_Q_vsetq_R(len * حجم(عائم));
                        ثابت HVX_وکتور     v_shifted = Q6_V_vror_VR(v_sink, 128 - len * حجم(عائم));
                        v_sink = Q6_V_vmux_QVV(q_keep, v_sink, v_shifted);
                    }
                } وإلا إذا (rem_g >= 32) {
                    v_sink = hvx_fa_بارگیری_sinks(sinks_داده, سر_پایه + g0, 32);
                } وإلا {
                    ثابت HVX_وکتور v_lo = hvx_fa_بارگیری_sinks(sinks_داده, سر_پایه + g0, rem_g);
                    ثابت HVX_وکتور v_hi = hvx_fa_بارگیری_sinks(sinks_داده, سر_پایه, 32 - rem_g);
                    v_sink = Q6_V_vmux_QVV(Q6_Q_vsetq_R(rem_g * حجم(عائم)), v_lo, Q6_V_vror_VR(v_hi, 128 - rem_g * حجم(عائم)));
                }
                g_prev = g0;
            }

            HVX_وکتور v_d = HVX_OP_SUB_F32(v_sink, HVX_OP_MUL_F32(factx->vtcm_m_vec[i], v_ln2));
            إذا (r0 + 32 > n_ردیفها_g) {
                ثابت HVX_وکتور_Pred q_valid = Q6_Q_vsetq_R((n_ردیفها_g - r0) * حجم(عائم));
                v_d = Q6_V_vmux_QVV(q_valid, v_d, Q6_V_vzero());
            }
            factx->vtcm_l_vec[i] = HVX_OP_جمع_کن_F32(factx->vtcm_l_vec[i], hvx_vec_exp_f32(v_d));
        }
    }

    ثابت HVX_وکتور     v_offsets = *(ثابت HVX_وکتور *) d_tile_scatter_offsets;
    ثابت HVX_وکتور_Pred q_32_ماسک = Q6_Q_vsetq_R(32 * حجم(__fp16));
    ثابت HVX_وکتور     one       = hvx_vec_splat_f32(1.0f);

    HVX_وکتور v_content = Q6_V_vzero();
    لكل (size_t i = 0; i < n_ردیف_tiles; ++i) {
        إذا ((i % 2) == 0) {
            HVX_وکتور inv_lo = HVX_OP_MUL_F32(one, hvx_vec_inverse_f32(factx->vtcm_l_vec[i]));
            HVX_وکتور inv_hi = (i + 1 < n_ردیف_tiles) ? HVX_OP_MUL_F32(one, hvx_vec_inverse_f32(factx->vtcm_l_vec[i + 1])) : Q6_V_vzero();
            v_content = hvx_vec_f32_مقصد_f16(inv_lo, inv_hi);
        } وإلا {
            v_content = Q6_V_vror_VR(v_content, 64);
        }

        __fp16 * بیرون_پایه = factx->vtcm_d_inv_l + i * HMX_FP16_TILE_N_ELMS;
        Q6_vscatter_QRMVhV(q_32_ماسک, (size_t) بیرون_پایه, HMX_FP16_TILE_اندازه - 1, v_offsets, v_content);
    }
}

ساكن فراغ fa_phase_softmax_and_build_d(بنية hmx_fa_context * factx,
                                         fa_softmax_آرگومانها_t *     sargs,
                                         size_t                  n_ردیف_tiles,
                                         size_t                  n_ردیف_tiles_g_br) {
    work_صف_t wp = factx->octx->ctx->work_صف;
    ثابت size_t n_ردیف_vec_cnt = hmx_ceil_div(sargs->n_ردیفها_g, 64);

    کارگر_callback_t softmax_fn = fa_softmax_thread;
    إذا (sargs->ماسک == عدم && !factx->has_softcap && !sargs->has_alibi) {
        softmax_fn = fa_softmax_thread_nomask;
    } وإلا إذا (sargs->ماسک != عدم && factx->ماسک_پخش && !factx->has_softcap && !sargs->has_alibi) {
        إذا (sargs->G == 1) {
            softmax_fn = fa_softmax_thread_ماسک_پخش_g1;
        } وإلا {
            softmax_fn = fa_softmax_thread_ماسک_پخش_gn;
        }
    }

    إذا (factx->n_نخها > 1 && n_ردیف_vec_cnt >= 2) {
        uint32_t n_use = (uint32_t) hex_smin((size_t) factx->n_نخها, n_ردیف_vec_cnt);
        sargs->thread_div = init_fastdiv_مقادیر(n_use);
        work_صف_پیاپی(wp, softmax_fn, sargs, n_use);
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
    size_t         n_ردیف_tiles;
    size_t         n_ستون_tiles;
    size_t         n_dot_tiles;  // DK / 32
    size_t         n_tiles_در_هر_bc;
    uint8_t *      hmx_scales;
} hmx_fa_qk_کار_t;

ساكن فراغ hmx_fa_qk_dot_کارگر(فراغ * داده) {
    hmx_fa_qk_کار_t * کار            = (hmx_fa_qk_کار_t *) داده;
    ثابت size_t      n_ردیف_tiles    = کار->n_ردیف_tiles;
    ثابت size_t      n_ستون_tiles    = کار->n_ستون_tiles;
    ثابت size_t      n_dot_tiles    = کار->n_dot_tiles;
    ثابت size_t      n_tiles_در_هر_bc = کار->n_tiles_در_هر_bc;
    ثابت __fp16 * محدود q_tiles  = کار->q_tiles;
    ثابت __fp16 * محدود k_tiles  = کار->k_tiles;
    __fp16 * محدود s_tiles        = کار->s_tiles;
    __builtin_assume(n_ردیف_tiles > 0);
    __builtin_assume(n_ستون_tiles > 0);
    __builtin_assume(n_dot_tiles > 0);

    asm متطاير(HMX_مجموعه_BIAS("%0") :: "r"((بیعلامت صحيح)کار->hmx_scales));
    ثابت size_t dot_stride = n_dot_tiles * HMX_FP16_TILE_N_ELMS;
    لكل (size_t r = 0; r < n_ردیف_tiles; ++r) {
        ثابت __fp16 * ردیف_tiles = q_tiles + r * dot_stride;
        ثابت __fp16 * ستون_tiles = k_tiles;
        __fp16 *       بیرون_tile  = s_tiles + r * n_tiles_در_هر_bc * HMX_FP16_TILE_N_ELMS;

        لكل (size_t c = 0; c < n_ستون_tiles; ++c) {
            hmx_fa_qk_dot_tile(ردیف_tiles, ستون_tiles, بیرون_tile, n_dot_tiles);
            ستون_tiles += dot_stride;
            بیرون_tile  += HMX_FP16_TILE_N_ELMS;
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
    size_t         n_ردیف_tiles;
    size_t         n_ستون_tiles;
    size_t         n_ردیف_tiles_g_br;
    size_t         n_tiles_در_هر_bc;
    size_t         DV;
} hmx_fa_o_update_کار_t;

ساكن فراغ hmx_fa_o_update_کارگر(فراغ * داده) {
    hmx_fa_o_update_کار_t * کار              = (hmx_fa_o_update_کار_t *) داده;
    ثابت size_t            n_ردیف_tiles      = کار->n_ردیف_tiles;
    ثابت size_t            n_ستون_tiles      = کار->n_ستون_tiles;
    ثابت size_t            n_ردیف_tiles_g_br = کار->n_ردیف_tiles_g_br;
    ثابت size_t            n_tiles_در_هر_bc   = کار->n_tiles_در_هر_bc;
    ثابت size_t            DV_tiles         = کار->DV / 32;
    ثابت __fp16 * محدود d_tiles          = کار->d_tiles;
    ثابت __fp16 * محدود p_tiles          = کار->p_tiles;
    ثابت __fp16 * محدود v_tiles          = کار->v_tiles;
    ثابت __fp16 * محدود o_prev           = کار->o_prev;
    __fp16 * محدود o_curr                 = کار->o_curr;
    __builtin_assume(n_ردیف_tiles > 0);
    __builtin_assume(n_ستون_tiles > 0);
    __builtin_assume(DV_tiles > 0);

    asm متطاير(HMX_مجموعه_BIAS("%0") :: "r"((بیعلامت صحيح)کار->hmx_scales));
    ثابت size_t o_stride = n_ردیف_tiles_g_br * HMX_FP16_TILE_N_ELMS;
    ثابت size_t v_stride = n_tiles_در_هر_bc * HMX_FP16_TILE_N_ELMS;
    لكل (size_t r = 0; r < n_ردیف_tiles; ++r) {
        ثابت __fp16 * d_diag     = d_tiles + r * HMX_FP16_TILE_N_ELMS;
        ثابت __fp16 * p_tile_درون  = p_tiles + (r * n_tiles_در_هر_bc) * HMX_FP16_TILE_N_ELMS;
        ثابت __fp16 * o_rc       = o_prev + r * HMX_FP16_TILE_N_ELMS;
        ثابت __fp16 * v_tile_درون  = v_tiles;
        __fp16       * o_tile_بیرون = o_curr + r * HMX_FP16_TILE_N_ELMS;

        لكل (size_t c = 0; c < DV_tiles; ++c) {
            hmx_fa_o_update_tile(d_diag, o_rc, p_tile_درون, v_tile_درون, o_tile_بیرون, n_ستون_tiles);
            o_rc       += o_stride;
            v_tile_درون  += v_stride;
            o_tile_بیرون += o_stride;
        }
    }
}

تعريفنوع بنية {
    __fp16 *       o_curr;   // output (row-major tile layout)
    ثابت __fp16 * o_prev;   // input (column-major tile layout)
    ثابت __fp16 * d_tiles;  // diag(1/l) tiles
    uint8_t *      hmx_scales;
    size_t         n_ردیف_tiles;
    size_t         n_ردیف_tiles_g_br;
    size_t         DV;
} hmx_fa_o_norm_کار_t;

ساكن فراغ hmx_fa_o_norm_کارگر(فراغ * داده) {
    hmx_fa_o_norm_کار_t * کار              = (hmx_fa_o_norm_کار_t *) داده;
    ثابت size_t          n_ردیف_tiles      = کار->n_ردیف_tiles;
    ثابت size_t          n_ردیف_tiles_g_br = کار->n_ردیف_tiles_g_br;
    ثابت size_t          DV_tiles         = کار->DV / 32;
    ثابت __fp16 * محدود d_tiles        = کار->d_tiles;
    ثابت __fp16 * محدود o_prev         = کار->o_prev;
    __fp16 * محدود o_curr               = کار->o_curr;
    __builtin_assume(n_ردیف_tiles > 0);
    __builtin_assume(DV_tiles > 0);

    asm متطاير(HMX_مجموعه_BIAS("%0") :: "r"((بیعلامت صحيح)کار->hmx_scales));
    ثابت size_t o_stride = n_ردیف_tiles_g_br * HMX_FP16_TILE_N_ELMS;
    لكل (size_t r = 0; r < n_ردیف_tiles; ++r) {
        ثابت __fp16 * d_diag = d_tiles + r * HMX_FP16_TILE_N_ELMS;
        ثابت __fp16 * o_rc = o_prev + r * HMX_FP16_TILE_N_ELMS;
        __fp16 *       o_بیرون = o_curr + r * DV_tiles * HMX_FP16_TILE_N_ELMS;

        لكل (size_t c = 0; c < DV_tiles; ++c) {
            hmx_fa_o_norm_tile(d_diag, o_rc, o_بیرون);
            o_rc  += o_stride;
            o_بیرون += HMX_FP16_TILE_N_ELMS;
        }
    }
}

// Populate per-GQA-row ALiBi slopes for a given KV head.
ساكن __attribute__((noinline)) فراغ fa_compute_slopes(
                              ثابت بنية hmx_fa_context * factx,
                              uint32_t                      kv_سر,
                              size_t                        n_ردیفها_g) {
    __fp16 * slopes = factx->vtcm_slopes;
    إذا (factx->max_bias == 0.0f) {
        hvx_splat_f16_a(slopes, 1.0f, n_ردیفها_g);
        أرجع;
    }

    ثابت uint32_t G           = factx->G;
    ثابت uint32_t n_سر_log2 = factx->n_سر_log2;
    ثابت عائم    m0          = factx->m0;
    ثابت عائم    m1          = factx->m1;

    __fp16 موقت_slopes[512] __attribute__((aligned(128)));
    إذا (G <= 32) {
        // Fast path: Compute G unique slope values in vector registers
        HVX_وکتور v_val = hvx_alibi_slopes(kv_سر, G, n_سر_log2, m0, m1);

        __fp16 موقت_slopes_aligned[64] __attribute__((aligned(128)));
        hvx_vmem(موقت_slopes_aligned) = hvx_vec_f32_مقصد_f16(v_val, Q6_V_vzero());

        لكل (uint32_t i = 0; i < G; ++i) {
            موقت_slopes[i] = موقت_slopes_aligned[i];
        }
    } وإلا {
        // Fallback path: G > 32 (rare configurations)
        لكل (uint32_t i = 0; i < G; ++i) {
            موقت_slopes[i] = (__fp16)alibi_شیب(kv_سر * G + i, n_سر_log2, m0, m1);
        }
    }

    // Allocate stack buffer to avoid scalar writes to VTCM (which generates L2 misses)
    __fp16 local_slopes[n_ردیفها_g] __attribute__((aligned(128)));
    لكل (size_t r = 0; r < n_ردیفها_g; ++r) {
        local_slopes[r] = موقت_slopes[fastmodulo(r, G, &factx->div_G)];
    }

    // Copy to VTCM slopes using HVX block copy (both are aligned to 128 bytes)
    hvx_copy_f16_aa((uint8_t *)slopes, (ثابت uint8_t *)local_slopes, n_ردیفها_g);
}

ساكن فراغ fa_افزودن_ماسک_dma_gqa(
    dma_صف *               dma_q,
    ثابت بنية htp_tensor * ماسک,
    uint32_t                  q_شروع,
    uint32_t                  im3,
    uint32_t                  kv_شروع,
    uint32_t                  kv_سر,
    uint32_t                  G,
    uint32_t                  m_سطر_بایتها,
    uint32_t                  kv_ردیفها,
    uint32_t                  n_ردیفها_q,
    بنية hmx_fa_context *   factx
) {
    لكل (uint32_t g = 0; g < G; ++g) {
        ثابت uint32_t h_idx = kv_سر * G + g;
        ثابت uint32_t im2 = fastmodulo(h_idx, ماسک->ne[2], &factx->src3_div2);
        ثابت dma_addr_t ms_src = ماسک->داده + q_شروع * ماسک->nb[1] +
                                  im2 * ماسک->nb[2] + im3 * ماسک->nb[3] + kv_شروع * حجم(__fp16);
        uint8_t * ms_dst = (uint8_t *) factx->vtcm_ماسک_buf + g * m_سطر_بایتها;
        dma_صف_افزودن(dma_q, dma_make_داده(ms_dst, ms_src), G * m_سطر_بایتها, ماسک->nb[1], kv_ردیفها * حجم(__fp16), n_ردیفها_q);
    }
}

ساكن فراغ fa_برداشت_ماسک_dma_gqa(dma_صف * dma_q, uint32_t G) {
    لكل (uint32_t g = 0; g < G; ++g) {
        dma_صف_برداشت(dma_q);
    }
}

ساكن مضمن فراغ fa_prefetch_بلوک(dma_صف * dma_q, ثابت بنية htp_tensor * k, ثابت بنية htp_tensor * v, ثابت بنية htp_tensor * ماسک,
                                     uint32_t b, size_t Bc, size_t اندازه_k_ردیف_padded, size_t اندازه_k_ردیف, size_t اندازه_v_ردیف_padded, size_t اندازه_v_ردیف,
                                     uint32_t ik2, uint32_t ik3, uint32_t iv2, uint32_t iv3, uint32_t q_شروع, uint32_t im3, uint32_t kv_سر, uint32_t G,
                                     size_t m_سطر_بایتها, size_t n_ردیفها_q, size_t nek1, size_t prefetch_buf, بنية hmx_fa_context * factx) {
    ثابت uint32_t  prefetch_شروع = b * Bc;
    ثابت uint32_t  prefetch_ردیفها  = hex_smin(Bc, nek1 - prefetch_شروع);
    ثابت dma_addr_t k_prefetch_src = k->داده + prefetch_شروع * k->nb[1] + ik2 * k->nb[2] + ik3 * k->nb[3];
    dma_صف_افزودن(dma_q, dma_make_داده(factx->vtcm_k_fp16[prefetch_buf], k_prefetch_src), اندازه_k_ردیف_padded, k->nb[1], اندازه_k_ردیف, prefetch_ردیفها);
    ثابت dma_addr_t v_prefetch_src = v->داده + prefetch_شروع * v->nb[1] + iv2 * v->nb[2] + iv3 * v->nb[3];
    dma_صف_افزودن(dma_q, dma_make_داده(factx->vtcm_v_fp16[prefetch_buf], v_prefetch_src), اندازه_v_ردیف_padded, v->nb[1], اندازه_v_ردیف, prefetch_ردیفها);

    إذا (ماسک) {
        إذا (__builtin_expect(factx->ماسک_پخش, true)) {
            ثابت dma_addr_t ms_src = ماسک->داده + q_شروع * ماسک->nb[1] + im3 * ماسک->nb[3] + prefetch_شروع * حجم(__fp16);
            dma_cache_fa_افزودن(dma_q, &factx->m_cache, ms_src, m_سطر_بایتها, ماسک->nb[1], prefetch_ردیفها * حجم(__fp16), n_ردیفها_q);
        } وإلا {
            fa_افزودن_ماسک_dma_gqa(dma_q, ماسک, q_شروع, im3, prefetch_شروع, kv_سر, G, m_سطر_بایتها, prefetch_ردیفها, n_ردیفها_q, factx);
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
    ثابت بنية htp_tensor * ماسک = (octx->src[3] && octx->src[3]->داده) ? octx->src[3] : عدم;
    ثابت بنية htp_tensor * dst  = octx->dst;

    بنية htp_context * ثابت ctx = octx->ctx;

    إذا (!ctx->hmx_enabled) {
        أرجع HTP_وضعیت_NO_SUPPORT;
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
    ثابت uint32_t DK_pad = hex_دور_به_بالا(DK, 64);
    ثابت uint32_t DV_pad = hex_دور_به_بالا(DV, 64);
    إذا (DK == 0 || DV == 0) {
        أرجع HTP_وضعیت_NO_SUPPORT;
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
    factx.n_نخها      = kparams->n_نخها;
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
    factx.n_kv_بلوکها    = kparams->n_kv_بلوکها;
    factx.is_q_fp32      = (q->type == HTP_TYPE_F32);
    factx.is_dst_fp32    = (dst->type == HTP_TYPE_F32);
    factx.pipeline       = (kparams->u.hmx.pipeline != 0);
    factx.ماسک_پخش = (kparams->u.hmx.ماسک_پخش != 0);
    إذا (ماسک) {
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
    ثابت uint32_t n_نخها = factx.n_نخها;
    ثابت uint32_t G = factx.G;

    // Multi-device: prefer head-parallel partitioning (each core owns a disjoint head
    // shard), falling back to Q-block (token) split when heads don't divide evenly.
    ثابت uint32_t n_q_بلوکها = (neq1 + Br - 1) / Br;
    uint32_t q_شروع_min  = 0;
    uint32_t q_شروع_max  = neq1;
    uint32_t kv_سر_min  = 0;
    uint32_t kv_سر_max  = n_kv_heads;

    إذا (octx->ctx->mdev.شمار > 1) {
        ثابت uint32_t mdev_شمار = octx->ctx->mdev.شمار;
        ثابت uint32_t mdev_idx   = octx->ctx->mdev.idx;
        ثابت uint32_t dst_e_اندازه = (dst->type == HTP_TYPE_F32) ? حجم(عائم) : حجم(__fp16);
        ثابت bool can_تقسیم      = htp_tensor_can_ردیف_افراز(dst, dst_e_اندازه);

        إذا (kparams->سر_تقسیم && can_تقسیم && n_kv_heads >= mdev_شمار && n_kv_heads % mdev_شمار == 0) {
            ثابت uint32_t kv_در_هر_core = n_kv_heads / mdev_شمار;
            kv_سر_min = mdev_idx * kv_در_هر_core;
            kv_سر_max = kv_سر_min + kv_در_هر_core;
        } وإلا {
            ثابت بنية htp_tensor_mdev_range range = htp_tensor_mdev_افراز(n_q_بلوکها, can_تقسیم ? 1 : 0, mdev_idx, mdev_شمار, &octx->ctx->mdev.شمار_div);
            ثابت uint32_t بلوک_شروع = range.شروع;
            ثابت uint32_t بلوک_انتها   = range.شروع + range.شمار;

            إذا (بلوک_شروع >= بلوک_انتها) {
                أرجع HTP_وضعیت_OK;
            }

            q_شروع_min = بلوک_شروع * Br;
            q_شروع_max = MIN(بلوک_انتها * Br, neq1);
        }
    }

    // ======== VTCM allocation (GQA-aware) ========
    // K/V row sizes drive the DMA descriptors (not the VTCM layout) and are used
    // throughout the KV loop below. The DMA copies only the real DK/DV columns; the
    // staging rows are padded to hold DK_pad/DV_pad columns (tail zero-filled below)
    // so the HMX interleave/tile logic can operate on 64-aligned head dims.
    ثابت size_t اندازه_k_ردیف        = htp_tensor_get_ردیف_اندازه(k->type, DK);
    ثابت size_t اندازه_v_ردیف        = htp_tensor_get_ردیف_اندازه(v->type, DV);
    ثابت size_t اندازه_k_ردیف_padded = hex_دور_به_بالا(DK_pad * حجم(__fp16), 128);
    ثابت size_t اندازه_v_ردیف_padded = hex_دور_به_بالا(DV_pad * حجم(__fp16), 128);

    // Build the VTCM layout once (shared with the host estimator) and place every
    // scratch buffer at its computed offset. Padded head dims size the HMX tiles.
    بنية hmx_fa_vtcm_layout L;
    hmx_fa_vtcm_layout_build(&L, G, DK_pad, DV_pad, Br, Bc, n_نخها, pipeline, factx.is_q_fp32, factx.sinks != عدم, factx.n_heads);

    إذا (L.مجموع_بایتها > ctx->vtcm_اندازه) {
        أرجع HTP_وضعیت_VTCM_TOO_SMALL;
    }

    uint8_t * ثابت پایه = ctx->vtcm_پایه;

    factx.vtcm_q_dma          = VTCM_LAYOUT_PTR(__fp16, پایه, L.off_q_dma);
    factx.vtcm_q_tiles        = VTCM_LAYOUT_PTR(__fp16, پایه, L.off_q_tiles);
    factx.vtcm_o_tiles[0]     = VTCM_LAYOUT_PTR(__fp16, پایه, L.off_o_tiles[0]);
    factx.vtcm_o_tiles[1]     = VTCM_LAYOUT_PTR(__fp16, پایه, L.off_o_tiles[1]);
    factx.vtcm_k_fp16[0]      = VTCM_LAYOUT_PTR(__fp16, پایه, L.off_k_fp16[0]);
    factx.vtcm_k_fp16[1]      = VTCM_LAYOUT_PTR(__fp16, پایه, L.off_k_fp16[1]);
    factx.vtcm_v_fp16[0]      = VTCM_LAYOUT_PTR(__fp16, پایه, L.off_v_fp16[0]);
    factx.vtcm_v_fp16[1]      = VTCM_LAYOUT_PTR(__fp16, پایه, L.off_v_fp16[1]);
    factx.vtcm_k_tiles[0]     = VTCM_LAYOUT_PTR(__fp16, پایه, L.off_k_tiles[0]);
    factx.vtcm_k_tiles[1]     = VTCM_LAYOUT_PTR_OPTIONAL(__fp16, پایه, L.off_k_tiles[1], pipeline);
    factx.vtcm_v_tiles[0]     = VTCM_LAYOUT_PTR(__fp16, پایه, L.off_v_tiles[0]);
    factx.vtcm_v_tiles[1]     = VTCM_LAYOUT_PTR_OPTIONAL(__fp16, پایه, L.off_v_tiles[1], pipeline);
    factx.vtcm_s_tiles[0]     = VTCM_LAYOUT_PTR(__fp16, پایه, L.off_s_tiles[0]);
    factx.vtcm_s_tiles[1]     = VTCM_LAYOUT_PTR_OPTIONAL(__fp16, پایه, L.off_s_tiles[1], pipeline);
    factx.vtcm_p_tiles[0]     = VTCM_LAYOUT_PTR(__fp16, پایه, L.off_p_tiles[0]);
    factx.vtcm_p_tiles[1]     = VTCM_LAYOUT_PTR_OPTIONAL(__fp16, پایه, L.off_p_tiles[1], pipeline);
    factx.vtcm_d_tiles[0]     = VTCM_LAYOUT_PTR(__fp16, پایه, L.off_d_tiles[0]);
    factx.vtcm_d_tiles[1]     = VTCM_LAYOUT_PTR_OPTIONAL(__fp16, پایه, L.off_d_tiles[1], pipeline);
    factx.vtcm_d_inv_l        = VTCM_LAYOUT_PTR(__fp16, پایه, L.off_d_inv_l);
    factx.vtcm_m_vec          = VTCM_LAYOUT_PTR(HVX_وکتور, پایه, L.off_m_vec);
    factx.vtcm_l_vec          = VTCM_LAYOUT_PTR(HVX_وکتور, پایه, L.off_l_vec);
    factx.vtcm_s_rowmax       = VTCM_LAYOUT_PTR(HVX_وکتور, پایه, L.off_s_rowmax);
    factx.vtcm_p_rowsum       = VTCM_LAYOUT_PTR(HVX_وکتور, پایه, L.off_p_rowsum);
    factx.vtcm_ردیف_bufs       = VTCM_LAYOUT_PTR(HVX_وکتور, پایه, L.off_ردیف_bufs);
    factx.ردیف_buf_stride      = L.ردیف_buf_stride;
    factx.vtcm_hmx_scales_id  = VTCM_LAYOUT_PTR(uint8_t, پایه, L.off_hmx_scales_id);
    factx.vtcm_hmx_scales_qk  = VTCM_LAYOUT_PTR(uint8_t, پایه, L.off_hmx_scales_qk);
    factx.vtcm_ماسک_buf       = VTCM_LAYOUT_PTR(__fp16, پایه, L.off_ماسک_buf);
    factx.ماسک_buf_ردیف_stride = L.ماسک_buf_ردیف_stride;
    factx.q_tile_بایتها        = L.q_tile_بایتها;
    factx.k_tile_بایتها        = L.k_tile_بایتها;
    factx.o_tile_بایتها        = L.o_tile_بایتها;
    factx.ستون_vec_بایتها       = L.ستون_vec_بایتها;
    factx.d_tile_بایتها        = L.d_tile_بایتها;
    factx.vtcm_slopes         = VTCM_LAYOUT_PTR(__fp16, پایه, L.off_slopes);
    factx.vtcm_sinks          = VTCM_LAYOUT_PTR_OPTIONAL(عائم, پایه, L.off_sinks, factx.sinks != عدم);

    ثابت size_t m_سطر_بایتها = L.m_سطر_بایتها;  // used by the mask DMAs in the KV loop

    dma_cache_fa_init(&factx.m_cache, (uint8_t *) factx.vtcm_ماسک_buf, L.m_buf_جایگاه_بایتها, HMX_FA_DMA_CACHE_اندازه);

    // Head-dim padding: the K/V DMA staging buffers and the flat-Q buffer are laid out
    // with padded row strides (size_{k,v,q}_row_padded, covering D_pad columns) but the
    // DMA only writes the real D columns per row. Zero the whole staging buffers once up
    // front so tail lanes [D, D_pad) stay zero for all KV blocks. No-op when already aligned.
    إذا (DK_pad != DK || DV_pad != DV) {
        ثابت size_t k_buf_بایتها = (size_t) factx.Bc * اندازه_k_ردیف_padded;
        ثابت size_t v_buf_بایتها = (size_t) factx.Bc * اندازه_v_ردیف_padded;
        hvx_splat_u8_a((حرف *) factx.vtcm_k_fp16[0], 0, k_buf_بایتها);
        hvx_splat_u8_a((حرف *) factx.vtcm_k_fp16[1], 0, k_buf_بایتها);
        hvx_splat_u8_a((حرف *) factx.vtcm_v_fp16[0], 0, v_buf_بایتها);
        hvx_splat_u8_a((حرف *) factx.vtcm_v_fp16[1], 0, v_buf_بایتها);
        // Flat-Q DMA scratch
        إذا (factx.vtcm_q_dma) {
            ثابت size_t q_dma_بایتها = hex_align_به_بالا(factx.g_br * DK * (factx.is_q_fp32 ? حجم(عائم) : حجم(__fp16)), 128);
            hvx_splat_u8_a((حرف *) factx.vtcm_q_dma, 0, q_dma_بایتها);
        }
    }

    // ======== Initialize HMX output scales ========
    hmx_init_ستونی_scales(factx.vtcm_hmx_scales_id, Q6_V_vsplat_R(0x3c00)); // 1.0
    {
        اتحاد { __fp16 f; uint16_t u; } sc = { .f = factx.scale };
        hmx_init_ستونی_scales(factx.vtcm_hmx_scales_qk, Q6_V_vsplat_R(sc.u));
    }

    // ======== DMA setup ========
    dma_صف * ثابت dma_q = ctx->dma[0];

    إذا (factx.sinks) {
        dma_صف_افزودن(dma_q, dma_make_داده(factx.vtcm_sinks, factx.sinks->داده), L.sinks_بایتها, 0, factx.sinks->اندازه, 1);
        dma_صف_برداشت(dma_q);
    }

    ثابت size_t n_ردیف_tiles_g_br = g_br / HMX_FP16_TILE_N_ردیفها;
    ثابت size_t n_tiles_در_هر_bc   = Bc / HMX_FP16_TILE_N_تعداد_ستون;

    ثابت size_t qo_element_اندازه = factx.is_q_fp32 ? حجم(عائم) : حجم(__fp16);

    ثابت bool q_transposed                 = q->nb[1] < q->nb[2];
    ثابت size_t q_src_stride               = q_transposed ? q->nb[2] : q->nb[1];
    ثابت size_t q_ردیف_بایتها_untransposed   = factx.G * factx.DK * qo_element_اندازه;
    ثابت size_t q_ردیف_بایتها_trans_ضریب   = factx.DK * qo_element_اندازه;

    // ======== Reusable job descriptors for pipeline ========
    hmx_fa_qk_کار_t       qk_کار;
    hmx_fa_o_update_کار_t ou_کار;
    hmx_fa_o_norm_کار_t   on_کار;

    // ======== Main loop ========
    لكل (uint32_t ib3 = 0; ib3 < neq3; ++ib3) {
        ثابت uint32_t im3 = ماسک ? fastmodulo(ib3, ماسک->ne[3], &factx.src3_div3) : 0;
        لكل (uint32_t q_شروع = q_شروع_min; q_شروع < q_شروع_max; q_شروع += Br) {
            ثابت uint32_t n_ردیفها_q    = hex_smin(Br, neq1 - q_شروع);
            ثابت size_t   n_ردیفها_g    = n_ردیفها_q * G;
            ثابت size_t   g_br_actual = hex_align_به_بالا(n_ردیفها_g, HMX_FP16_TILE_N_ردیفها);
            ثابت size_t   n_ردیف_tiles = g_br_actual / HMX_FP16_TILE_N_ردیفها;

            uint16_t فعال_بلوکها[FA_MAX_KV_بلوکها];
            uint32_t n_فعال_بلوکها = factx.n_kv_بلوکها;
            إذا (ماسک && factx.ماسک_پخش && factx.n_kv_بلوکها > 1) {
                ثابت size_t scratch_بایتها = factx.k_tile_بایتها * (factx.pipeline ? 2 : 1);
                n_فعال_بلوکها = fa_پیمایش_فعال_بلوکها(
                    dma_q, ماسک, q_شروع, n_ردیفها_q, 0, im3, nek1, Bc,
                    factx.n_kv_بلوکها, factx.vtcm_k_tiles[0], scratch_بایتها, فعال_بلوکها);
            } وإلا {
                #رهنمود clang loop vectorize(disable)
                لكل (uint32_t b = 0; b < factx.n_kv_بلوکها; ++b) {
                    فعال_بلوکها[b] = (uint16_t) b;
                }
            }
            ثابت uint32_t b0       = (n_فعال_بلوکها > 0) ? فعال_بلوکها[0] : 0;
            ثابت uint32_t b0_شروع = b0 * Bc;
            ثابت uint32_t b0_ردیفها  = hex_smin(Bc, nek1 - b0_شروع);

            لكل (uint32_t kv_سر = kv_سر_min; kv_سر < kv_سر_max; ++kv_سر) {
                ثابت uint32_t ik2 = kv_سر;
                ثابت uint32_t ik3 = fastdiv(ib3, &kparams->پخش_rk3);
                ثابت uint32_t iv2 = kv_سر;
                ثابت uint32_t iv3 = fastdiv(ib3, &kparams->پخش_rv3);

                // 1. Push Q and KV DMAs for the first kv_head of this q_start block.
                // Subsequent kv_head iterations are enqueued early at the end of the previous iteration.
                إذا (kv_سر == kv_سر_min) {
                    ثابت dma_addr_t q_ptr = q->داده + q_شروع * q->nb[1] +
                                            (kv_سر * factx.G) * q->nb[2] + ib3 * q->nb[3];
                    ثابت size_t q_ردیف_بایتها = q_transposed ? n_ردیفها_q * q_ردیف_بایتها_trans_ضریب : q_ردیف_بایتها_untransposed;
                    ثابت size_t n_ردیفها      = q_transposed ? factx.G : n_ردیفها_q;
                    dma_صف_افزودن(dma_q, dma_make_داده(factx.vtcm_q_dma, q_ptr), q_ردیف_بایتها, hex_smax(q_src_stride, q_ردیف_بایتها), q_ردیف_بایتها, n_ردیفها);

                    إذا (n_فعال_بلوکها > 0) {
                        ثابت dma_addr_t k_src = k->داده + b0_شروع * k->nb[1] + ik2 * k->nb[2] + ik3 * k->nb[3];
                        dma_صف_افزودن(dma_q, dma_make_داده(factx.vtcm_k_fp16[0], k_src), اندازه_k_ردیف_padded, k->nb[1], اندازه_k_ردیف, b0_ردیفها);

                        ثابت dma_addr_t v_src = v->داده + b0_شروع * v->nb[1] + iv2 * v->nb[2] + iv3 * v->nb[3];
                        dma_صف_افزودن(dma_q, dma_make_داده(factx.vtcm_v_fp16[0], v_src), اندازه_v_ردیف_padded, v->nb[1], اندازه_v_ردیف, b0_ردیفها);

                        إذا (factx.pipeline && ماسک) {
                            إذا (__builtin_expect(factx.ماسک_پخش, true)) {
                                ثابت dma_addr_t ms_src = ماسک->داده + q_شروع * ماسک->nb[1] + im3 * ماسک->nb[3] + b0_شروع * حجم(__fp16);
                                dma_cache_fa_افزودن(dma_q, &factx.m_cache, ms_src, m_سطر_بایتها, ماسک->nb[1], b0_ردیفها * حجم(__fp16), n_ردیفها_q);
                            } وإلا {
                                fa_افزودن_ماسک_dma_gqa(dma_q, ماسک, q_شروع, im3, b0_شروع, kv_سر, G, m_سطر_بایتها, b0_ردیفها, n_ردیفها_q, &factx);
                            }
                        }
                    }
                }

                // 2. Pop Q DMA (blocks until Q is loaded)
                dma_صف_برداشت(dma_q);

                // ---- Load Q block & Initialize per-block state ----
                fa_phase_q_بارگیری(&factx, q, q_شروع, kv_سر, ib3, n_ردیفها_g);

                __fp16 * o_tile_prev = factx.vtcm_o_tiles[0];
                __fp16 * o_tile_curr = factx.vtcm_o_tiles[1];

                // ---- KV block loop with DMA double-buffering ----
                size_t buf_idx = 0;

                htp_trace_event_شروع(tr_hvx, HTP_TRACE_EVT_HVX_A_PREP, (uint16_t) q_شروع);
                fa_compute_slopes(&factx, kv_سر, n_ردیفها_g);
                htp_trace_event_stop(tr_hvx, HTP_TRACE_EVT_HVX_A_PREP, (uint16_t) q_شروع);

                ثابت size_t k_src_stride = اندازه_k_ردیف_padded / حجم(__fp16);
                ثابت size_t v_src_stride = اندازه_v_ردیف_padded / حجم(__fp16);

                hmx_صف_t hmx_q = ctx->hmx_صف;

                إذا (factx.pipeline) {
                    // Double-buffered job structs because HMX queue runs asynchronously
                    hmx_fa_qk_کار_t qk_کار[2];
                    hmx_fa_o_update_کار_t ou_کار[2];

                    // Prefetch second active block early if there are multiple active blocks
                    إذا (n_فعال_بلوکها > 1) {
                        fa_prefetch_بلوک(dma_q, k, v, ماسک, فعال_بلوکها[1], Bc, اندازه_k_ردیف_padded, اندازه_k_ردیف, اندازه_v_ردیف_padded, اندازه_v_ردیف,
                                          ik2, ik3, iv2, iv3, q_شروع, im3, kv_سر, G, m_سطر_بایتها, n_ردیفها_q, nek1, 1, &factx);
                    }

                    // Prep and start QK-dot(0)
                    فراغ * curr_k0 = (فراغ *) dma_صف_برداشت(dma_q).dst;
                    fa_phase_k_interleave(&factx, b0_ردیفها, k_src_stride, curr_k0, b0_شروع, 0);

                    qk_کار[0].q_tiles        = factx.vtcm_q_tiles;
                    qk_کار[0].k_tiles        = factx.vtcm_k_tiles[0];
                    qk_کار[0].s_tiles        = factx.vtcm_s_tiles[0];
                    qk_کار[0].n_ردیف_tiles    = n_ردیف_tiles;
                    qk_کار[0].n_ستون_tiles    = hmx_ceil_div(b0_ردیفها, HMX_FP16_TILE_N_تعداد_ستون);
                    qk_کار[0].n_dot_tiles    = DK_pad / 32;
                    qk_کار[0].n_tiles_در_هر_bc = n_tiles_در_هر_bc;
                    qk_کار[0].hmx_scales     = factx.vtcm_hmx_scales_qk;
                    hmx_صف_افزودن(hmx_q, hmx_صف_make_desc(hmx_fa_qk_dot_کارگر, &qk_کار[0]));

                    لكل (uint32_t گام = 0; گام < n_فعال_بلوکها; ++گام) {
                        ثابت uint32_t kv_blk      = فعال_بلوکها[گام];
                        ثابت uint32_t kv_شروع    = kv_blk * Bc;
                        ثابت uint32_t kv_ردیفها     = hex_smin(Bc, nek1 - kv_شروع);
                        ثابت size_t   n_ستون_tiles = hmx_ceil_div(kv_ردیفها, HMX_FP16_TILE_N_تعداد_ستون);

                        // ---- 1. Pop and run V-prep for current block ----
                        فراغ * curr_v = (فراغ *) dma_صف_برداشت(dma_q).dst;
                        fa_phase_v_interleave(&factx, kv_ردیفها, v_src_stride, curr_v, factx.vtcm_v_tiles[buf_idx], n_tiles_در_هر_bc, kv_شروع);

                        // ---- 2. Pop and run mask-prep for current block ----
                        __fp16 * current_ماسک_vtcm = عدم;
                        إذا (ماسک) {
                            إذا (__builtin_expect(factx.ماسک_پخش, true)) {
                                current_ماسک_vtcm = (__fp16 *) dma_صف_برداشت(dma_q).dst;
                            } وإلا {
                                fa_برداشت_ماسک_dma_gqa(dma_q, G);
                                current_ماسک_vtcm = factx.vtcm_ماسک_buf;
                            }
                        }

                        // ---- 3. Start HMX O update for step - 1 (reads P[1 - buf_idx], V[1 - buf_idx], D) ----
                        // O update relies on the previous block's P and V tiles.
                        // O update MUST be pushed before the next block's QK-dot: hmx_queue_pop() retires the
                        // oldest descriptor, so push order alone decides which pop waits for which job.
                        // If OU went in after QK(i+1), the pop below would retire QK(i+1) and leave
                        // OU(i-1) in flight into the next iteration, where V-prep overwrites V[prev_buf].
                        إذا (گام > 0) {
                            ثابت uint32_t prev_blk      = فعال_بلوکها[گام - 1];
                            ثابت size_t prev_buf        = 1 - buf_idx;
                            ou_کار[prev_buf].o_curr      = o_tile_curr;
                            ou_کار[prev_buf].o_prev      = o_tile_prev;
                            ou_کار[prev_buf].p_tiles     = factx.vtcm_p_tiles[prev_buf];
                            ou_کار[prev_buf].v_tiles     = factx.vtcm_v_tiles[prev_buf];
                            ou_کار[prev_buf].d_tiles     = factx.vtcm_d_tiles[prev_buf];
                            ou_کار[prev_buf].hmx_scales  = factx.vtcm_hmx_scales_id;
                            ou_کار[prev_buf].n_ردیف_tiles = n_ردیف_tiles;
                            ou_کار[prev_buf].n_ستون_tiles =
                                hmx_ceil_div(hex_smin(Bc, nek1 - prev_blk * Bc), HMX_FP16_TILE_N_تعداد_ستون);
                            ou_کار[prev_buf].n_ردیف_tiles_g_br = n_ردیف_tiles_g_br;
                            ou_کار[prev_buf].n_tiles_در_هر_bc   = n_tiles_در_هر_bc;
                            ou_کار[prev_buf].DV               = DV_pad;
                            hmx_صف_افزودن(hmx_q, hmx_صف_make_desc(hmx_fa_o_update_کارگر, &ou_کار[prev_buf]));
                        }

                        // ---- 4. Pop and run K-prep for next active block & push next QK-dot ----
                        إذا (گام + 1 < n_فعال_بلوکها) {
                            ثابت uint32_t بعدی_blk   = فعال_بلوکها[گام + 1];
                            ثابت uint32_t بعدی_شروع = بعدی_blk * Bc;
                            ثابت uint32_t بعدی_ردیفها  = hex_smin(Bc, nek1 - بعدی_شروع);
                            ثابت size_t   بعدی_buf   = 1 - buf_idx;

                            فراغ * بعدی_k = (فراغ *) dma_صف_برداشت(dma_q).dst;
                            fa_phase_k_interleave(&factx, بعدی_ردیفها, k_src_stride, بعدی_k, بعدی_شروع, بعدی_buf);

                            qk_کار[بعدی_buf].q_tiles        = factx.vtcm_q_tiles;
                            qk_کار[بعدی_buf].k_tiles        = factx.vtcm_k_tiles[بعدی_buf];
                            qk_کار[بعدی_buf].s_tiles        = factx.vtcm_s_tiles[بعدی_buf];
                            qk_کار[بعدی_buf].n_ردیف_tiles    = n_ردیف_tiles;
                            qk_کار[بعدی_buf].n_ستون_tiles    = hmx_ceil_div(بعدی_ردیفها, HMX_FP16_TILE_N_تعداد_ستون);
                            qk_کار[بعدی_buf].n_dot_tiles    = DK_pad / 32;
                            qk_کار[بعدی_buf].n_tiles_در_هر_bc = n_tiles_در_هر_bc;
                            qk_کار[بعدی_buf].hmx_scales     = factx.vtcm_hmx_scales_qk;
                            hmx_صف_افزودن(hmx_q, hmx_صف_make_desc(hmx_fa_qk_dot_کارگر, &qk_کار[بعدی_buf]));
                        }

                        // ---- 5. Wait for current block's QK-dot to finish ----
                        hmx_صف_برداشت(hmx_q);

                        // ---- 6. Phase 2: softmax + build_D ----
                        fa_softmax_آرگومانها_t sargs;
                        memset(&sargs, 0, حجم(sargs));
                        sargs.factx                = &factx;
                        sargs.buf_idx              = buf_idx;
                        sargs.kv_ردیفها              = kv_ردیفها;
                        sargs.n_ردیفها_g             = n_ردیفها_g;
                        sargs.n_ستون_tiles          = n_ستون_tiles;
                        sargs.n_tiles_در_هر_bc       = n_tiles_در_هر_bc;
                        sargs.n_ردیف_tiles          = n_ردیف_tiles;
                        sargs.n_ردیف_tiles_g_br     = n_ردیف_tiles_g_br;
                        sargs.Bc                   = Bc;
                        sargs.G                    = G;
                        sargs.kv_سر              = kv_سر;
                        sargs.kv_شروع             = kv_شروع;
                        sargs.q_شروع              = q_شروع;
                        sargs.ib3                  = ib3;
                        sargs.has_alibi            = (factx.max_bias != 0.0f);
                        sargs.ماسک                 = ماسک;
                        sargs.ماسک_vtcm            = current_ماسک_vtcm;
                        sargs.ماسک_vtcm_ردیف_stride = factx.ماسک_buf_ردیف_stride;
                        sargs.slopes               = factx.vtcm_slopes;

                        // Run Softmax on HVX (blocking call)
                        fa_phase_softmax_and_build_d(&factx, &sargs, n_ردیف_tiles, n_ردیف_tiles_g_br);

                        // Wait for HMX O update for step - 1 to finish
                        إذا (گام > 0) {
                            hmx_صف_برداشت(hmx_q);
                            hex_جابجا_ptr((فراغ **) &o_tile_curr, (فراغ **) &o_tile_prev);
                        }

                        // Prefetch block step + 2
                        إذا (گام + 2 < n_فعال_بلوکها) {
                            fa_prefetch_بلوک(dma_q, k, v, ماسک, فعال_بلوکها[گام + 2], Bc, اندازه_k_ردیف_padded, اندازه_k_ردیف, اندازه_v_ردیف_padded, اندازه_v_ردیف,
                                              ik2, ik3, iv2, iv3, q_شروع, im3, kv_سر, G, m_سطر_بایتها, n_ردیفها_q, nek1, buf_idx, &factx);
                        }

                        buf_idx = 1 - buf_idx;
                    }

                    // Epilogue
                    إذا (n_فعال_بلوکها > 0) {
                        ثابت uint32_t آخرین_blk = فعال_بلوکها[n_فعال_بلوکها - 1];
                        ثابت size_t آخرین_تعداد_ستون  = hmx_ceil_div(hex_smin(Bc, nek1 - آخرین_blk * Bc), HMX_FP16_TILE_N_تعداد_ستون);
                        ou_کار[0].o_curr           = o_tile_curr;
                        ou_کار[0].o_prev           = o_tile_prev;
                        ou_کار[0].p_tiles          = factx.vtcm_p_tiles[1 - buf_idx];
                        ou_کار[0].v_tiles          = factx.vtcm_v_tiles[1 - buf_idx];
                        ou_کار[0].d_tiles          = factx.vtcm_d_tiles[1 - buf_idx];
                        ou_کار[0].hmx_scales       = factx.vtcm_hmx_scales_id;
                        ou_کار[0].n_ردیف_tiles      = n_ردیف_tiles;
                        ou_کار[0].n_ستون_tiles      = آخرین_تعداد_ستون;
                        ou_کار[0].n_ردیف_tiles_g_br = n_ردیف_tiles_g_br;
                        ou_کار[0].n_tiles_در_هر_bc   = n_tiles_در_هر_bc;
                        ou_کار[0].DV               = DV_pad;
                        hmx_صف_افزودن(hmx_q, hmx_صف_make_desc(hmx_fa_o_update_کارگر, &ou_کار[0]));

                        // Overlapped: run HVX build diag inv L while HMX is busy executing the update
                        htp_trace_event_شروع(tr_hvx, HTP_TRACE_EVT_HVX_O_PROC, (uint16_t) q_شروع);
                        fa_build_d_diag_inv_l(&factx, n_ردیف_tiles, n_ردیف_tiles_g_br, kv_سر, n_ردیفها_g);
                        htp_trace_event_stop(tr_hvx, HTP_TRACE_EVT_HVX_O_PROC, (uint16_t) q_شروع);
                        hmx_صف_برداشت(hmx_q);

                        hex_جابجا_ptr((فراغ **) &o_tile_curr, (فراغ **) &o_tile_prev);
                    }

                } وإلا {
                    // Fallback path
                    لكل (uint32_t گام = 0; گام < n_فعال_بلوکها; ++گام) {
                        ثابت uint32_t kv_blk      = فعال_بلوکها[گام];
                        ثابت uint32_t kv_شروع    = kv_blk * Bc;
                        ثابت uint32_t kv_ردیفها     = hex_smin(Bc, nek1 - kv_شروع);
                        ثابت size_t   n_ستون_tiles = hmx_ceil_div(kv_ردیفها, HMX_FP16_TILE_N_تعداد_ستون);

                        إذا (ماسک) {
                            إذا (__builtin_expect(factx.ماسک_پخش, true)) {
                                ثابت dma_addr_t ms_src = ماسک->داده + q_شروع * ماسک->nb[1] + im3 * ماسک->nb[3] + kv_شروع * حجم(__fp16);
                                dma_cache_fa_افزودن(dma_q, &factx.m_cache, ms_src, m_سطر_بایتها, ماسک->nb[1], kv_ردیفها * حجم(__fp16), n_ردیفها_q);
                            } وإلا {
                                fa_افزودن_ماسک_dma_gqa(dma_q, ماسک, q_شروع, im3, kv_شروع, kv_سر, G, m_سطر_بایتها, kv_ردیفها, n_ردیفها_q, &factx);
                            }
                        }

                        إذا (گام + 1 < n_فعال_بلوکها) {
                            ثابت uint32_t  prefetch_blk   = فعال_بلوکها[گام + 1];
                            ثابت uint32_t  prefetch_شروع = prefetch_blk * Bc;
                            ثابت uint32_t  prefetch_ردیفها  = hex_smin(Bc, nek1 - prefetch_شروع);
                            ثابت size_t    prefetch_buf   = 1 - buf_idx;
                            ثابت dma_addr_t k_prefetch_src = k->داده + prefetch_شروع * k->nb[1] + ik2 * k->nb[2] + ik3 * k->nb[3];
                            dma_صف_افزودن(dma_q, dma_make_داده(factx.vtcm_k_fp16[prefetch_buf], k_prefetch_src), اندازه_k_ردیف_padded, k->nb[1], اندازه_k_ردیف, prefetch_ردیفها);
                            ثابت dma_addr_t v_prefetch_src = v->داده + prefetch_شروع * v->nb[1] + iv2 * v->nb[2] + iv3 * v->nb[3];
                            dma_صف_افزودن(dma_q, dma_make_داده(factx.vtcm_v_fp16[prefetch_buf], v_prefetch_src), اندازه_v_ردیف_padded, v->nb[1], اندازه_v_ردیف, prefetch_ردیفها);
                        }

                        // Wait for current K DMA and interleave
                        فراغ * curr_k = (فراغ *) dma_صف_برداشت(dma_q).dst;
                        fa_phase_k_interleave(&factx, kv_ردیفها, k_src_stride, curr_k, kv_شروع, 0);

                        {
                            qk_کار.q_tiles        = factx.vtcm_q_tiles;
                            qk_کار.k_tiles        = factx.vtcm_k_tiles[0];
                            qk_کار.s_tiles        = factx.vtcm_s_tiles[0];
                            qk_کار.n_ردیف_tiles    = n_ردیف_tiles;
                            qk_کار.n_ستون_tiles    = n_ستون_tiles;
                            qk_کار.n_dot_tiles    = (size_t) (DK_pad / 32);
                            qk_کار.n_tiles_در_هر_bc = n_tiles_در_هر_bc;
                            qk_کار.hmx_scales     = factx.vtcm_hmx_scales_qk;

                            hmx_صف_افزودن(ctx->hmx_صف, hmx_صف_make_desc(hmx_fa_qk_dot_کارگر, &qk_کار));
                            hmx_صف_برداشت(ctx->hmx_صف);
                        }

                        // Wait for current V DMA and interleave
                        فراغ * curr_v = (فراغ *) dma_صف_برداشت(dma_q).dst;
                        fa_phase_v_interleave(&factx, kv_ردیفها, v_src_stride, curr_v, factx.vtcm_v_tiles[0], n_tiles_در_هر_bc, kv_شروع);

                        // ---- Phase 3: softmax + build_D ----
                        __fp16 * current_ماسک_vtcm = عدم;
                        إذا (ماسک) {
                            إذا (__builtin_expect(factx.ماسک_پخش, true)) {
                                current_ماسک_vtcm = (__fp16 *) dma_صف_برداشت(dma_q).dst;
                            } وإلا {
                                fa_برداشت_ماسک_dma_gqa(dma_q, G);
                                current_ماسک_vtcm = factx.vtcm_ماسک_buf;
                            }
                        }

                        fa_softmax_آرگومانها_t sargs;
                        memset(&sargs, 0, حجم(sargs));
                        sargs.factx                = &factx;
                        sargs.kv_ردیفها              = kv_ردیفها;
                        sargs.n_ردیفها_g             = n_ردیفها_g;
                        sargs.n_ستون_tiles          = n_ستون_tiles;
                        sargs.n_tiles_در_هر_bc       = n_tiles_در_هر_bc;
                        sargs.n_ردیف_tiles          = n_ردیف_tiles;
                        sargs.n_ردیف_tiles_g_br     = n_ردیف_tiles_g_br;
                        sargs.Bc                   = Bc;
                        sargs.G                    = G;
                        sargs.kv_سر              = kv_سر;
                        sargs.kv_شروع             = kv_شروع;
                        sargs.q_شروع              = q_شروع;
                        sargs.ib3                  = ib3;
                        sargs.has_alibi            = (factx.max_bias != 0.0f);
                        sargs.ماسک                 = ماسک;
                        sargs.ماسک_vtcm            = current_ماسک_vtcm;
                        sargs.ماسک_vtcm_ردیف_stride = factx.ماسک_buf_ردیف_stride;
                        sargs.slopes               = factx.vtcm_slopes;
                        fa_phase_softmax_and_build_d(&factx, &sargs, n_ردیف_tiles, n_ردیف_tiles_g_br);

                        {
                            ou_کار.o_curr           = o_tile_curr;
                            ou_کار.o_prev           = o_tile_prev;
                            ou_کار.p_tiles          = factx.vtcm_p_tiles[0];
                            ou_کار.v_tiles          = factx.vtcm_v_tiles[0];
                            ou_کار.d_tiles          = factx.vtcm_d_tiles[0];
                            ou_کار.hmx_scales       = factx.vtcm_hmx_scales_id;
                            ou_کار.n_ردیف_tiles      = n_ردیف_tiles;
                            ou_کار.n_ستون_tiles      = n_ستون_tiles;
                            ou_کار.n_ردیف_tiles_g_br = n_ردیف_tiles_g_br;
                            ou_کار.n_tiles_در_هر_bc   = n_tiles_در_هر_bc;
                            ou_کار.DV               = DV_pad;

                            hmx_صف_افزودن(ctx->hmx_صف, hmx_صف_make_desc(hmx_fa_o_update_کارگر, &ou_کار));
                            إذا (گام + 1 == n_فعال_بلوکها) {
                                // Overlapped: run HVX build diag inv L while HMX is busy executing the update
                                htp_trace_event_شروع(tr_hvx, HTP_TRACE_EVT_HVX_O_PROC, (uint16_t) q_شروع);
                                fa_build_d_diag_inv_l(&factx, n_ردیف_tiles, n_ردیف_tiles_g_br, kv_سر, n_ردیفها_g);
                                htp_trace_event_stop(tr_hvx, HTP_TRACE_EVT_HVX_O_PROC, (uint16_t) q_شروع);
                            }
                            hmx_صف_برداشت(ctx->hmx_صف);

                            hex_جابجا_ptr((فراغ **) &o_tile_curr, (فراغ **) &o_tile_prev);
                        }

                        buf_idx = 1 - buf_idx;
                    }
                }

                // Enqueue DMAs for the next kv_head early so they overlap with O-PROC
                ثابت uint32_t بعدی_kv_سر = kv_سر + 1;
                ثابت bool has_بعدی = (بعدی_kv_سر < kv_سر_max);

                إذا (has_بعدی) {
                    ثابت dma_addr_t بعدی_q_ptr = q->داده + q_شروع * q->nb[1] + (بعدی_kv_سر * factx.G) * q->nb[2] + ib3 * q->nb[3];
                    ثابت size_t بعدی_q_ردیف_بایتها = q_transposed ? n_ردیفها_q * q_ردیف_بایتها_trans_ضریب : q_ردیف_بایتها_untransposed;
                    ثابت size_t بعدی_n_ردیفها      = q_transposed ? factx.G : n_ردیفها_q;
                    dma_صف_افزودن(dma_q, dma_make_داده(factx.vtcm_q_dma, بعدی_q_ptr), بعدی_q_ردیف_بایتها, hex_smax(q_src_stride, بعدی_q_ردیف_بایتها), بعدی_q_ردیف_بایتها, بعدی_n_ردیفها);

                    إذا (n_فعال_بلوکها > 0) {
                        ثابت uint32_t بعدی_ik2 = بعدی_kv_سر;
                        ثابت uint32_t بعدی_iv2 = بعدی_kv_سر;

                        ثابت dma_addr_t بعدی_k_src = k->داده + b0_شروع * k->nb[1] + بعدی_ik2 * k->nb[2] + ik3 * k->nb[3];
                        dma_صف_افزودن(dma_q, dma_make_داده(factx.vtcm_k_fp16[0], بعدی_k_src), اندازه_k_ردیف_padded, k->nb[1], اندازه_k_ردیف, b0_ردیفها);

                        ثابت dma_addr_t بعدی_v_src = v->داده + b0_شروع * v->nb[1] + بعدی_iv2 * v->nb[2] + iv3 * v->nb[3];
                        dma_صف_افزودن(dma_q, dma_make_داده(factx.vtcm_v_fp16[0], بعدی_v_src), اندازه_v_ردیف_padded, v->nb[1], اندازه_v_ردیف, b0_ردیفها);

                        إذا (factx.pipeline && ماسک) {
                            إذا (__builtin_expect(factx.ماسک_پخش, true)) {
                                ثابت dma_addr_t ms_src = ماسک->داده + q_شروع * ماسک->nb[1] + im3 * ماسک->nb[3] + b0_شروع * حجم(__fp16);
                                dma_cache_fa_افزودن(dma_q, &factx.m_cache, ms_src, m_سطر_بایتها, ماسک->nb[1], b0_ردیفها * حجم(__fp16), n_ردیفها_q);
                            } وإلا {
                                fa_افزودن_ماسک_dma_gqa(dma_q, ماسک, q_شروع, im3, b0_شروع, بعدی_kv_سر, G, m_سطر_بایتها, b0_ردیفها, n_ردیفها_q, &factx);
                            }
                        }
                    }
                }

                // ---- Final normalization ----
                {
                    on_کار.o_curr           = o_tile_curr;
                    on_کار.o_prev           = o_tile_prev;
                    on_کار.d_tiles          = factx.vtcm_d_inv_l;
                    on_کار.hmx_scales       = factx.vtcm_hmx_scales_id;
                    on_کار.n_ردیف_tiles      = n_ردیف_tiles;
                    on_کار.n_ردیف_tiles_g_br = n_ردیف_tiles_g_br;
                    on_کار.DV               = DV_pad;
                    hmx_صف_افزودن(ctx->hmx_صف, hmx_صف_make_desc(hmx_fa_o_norm_کارگر, &on_کار));
                    hmx_صف_برداشت(ctx->hmx_صف);
                }

                // ---- Store O block ----
                fa_phase_o_store(&factx, dst, o_tile_curr, q_شروع, kv_سر, ib3, n_ردیفها_g);
            }
        }
    }

    أرجع HTP_وضعیت_OK;
}

صحيح op_flash_attn_ext(بنية htp_ops_context * octx) {
    ثابت بنية htp_tensor * q    = octx->src[0];
    ثابت بنية htp_tensor * k    = octx->src[1];
    ثابت بنية htp_tensor * v    = octx->src[2];
    ثابت بنية htp_tensor * ماسک = octx->src[3];
    ثابت بنية htp_tensor * dst  = octx->dst;

    // Check support
    إذا ((q->type != HTP_TYPE_F16 && q->type != HTP_TYPE_F32) ||
        (k->type != HTP_TYPE_F16 && k->type != HTP_TYPE_Q8_0) ||
        (v->type != HTP_TYPE_F16 && v->type != HTP_TYPE_Q8_0)) {
        أرجع HTP_وضعیت_NO_SUPPORT;
    }

    إذا (htp_tensor_is_extended(dst)) {
        أرجع HTP_وضعیت_NO_SUPPORT;
    }

    ثابت بنية htp_fa_kernel_params * kparams = (ثابت بنية htp_fa_kernel_params *) octx->kernel_params;

    إذا (kparams->kernel_type == HTP_FA_KERNEL_UNSUPPORTED || kparams->n_kv_بلوکها > FA_MAX_KV_بلوکها) {
        أرجع HTP_وضعیت_NO_SUPPORT;
    }

    إذا (!htp_ops_context_مجموعه_n_نخها(octx, kparams->n_نخها)) {
        أرجع HTP_وضعیت_INVAL_PARAMS;
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

    factx.پخش_rk2 = kparams->پخش_rk2;
    factx.پخش_rk3 = kparams->پخش_rk3;
    factx.پخش_rv2 = kparams->پخش_rv2;
    factx.پخش_rv3 = kparams->پخش_rv3;

    إذا (ماسک) {
        factx.src3_div2 = kparams->src3_div2;
        factx.src3_div3 = kparams->src3_div3;
    }

    factx.is_q_fp32 = (q->type == HTP_TYPE_F32);
    factx.اندازه_q_ردیف_padded = kparams->u.hvx.اندازه_q_ردیف_padded;
    factx.اندازه_k_ردیف_padded = kparams->u.hvx.اندازه_k_ردیف_padded;
    factx.اندازه_v_ردیف_padded = kparams->u.hvx.اندازه_v_ردیف_padded;

    factx.n_بلوکها = kparams->n_kv_بلوکها;

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
        أرجع HTP_وضعیت_NO_SUPPORT;
    }
    لكل (uint32_t h = 0; h < n_سر; ++h) {
        factx.slopes[h] = (__fp16) ((kparams->max_bias > 0.0f) ? alibi_شیب(h, factx.n_سر_log2, factx.m0, factx.m1) : 1.0f);
    }

    // total rows in q
    ثابت uint32_t neq1 = q->ne[1];
    ثابت uint32_t neq2 = q->ne[2];
    ثابت uint32_t neq3 = q->ne[3];
    ثابت uint32_t مجموع_qrows = neq1 * neq2 * neq3;

    uint32_t qrow_شروع = 0;
    uint32_t qrows      = مجموع_qrows;

    إذا (octx->ctx->mdev.شمار > 1) {
        ثابت uint32_t mdev_شمار = octx->ctx->mdev.شمار;
        ثابت uint32_t mdev_idx   = octx->ctx->mdev.idx;
        ثابت uint32_t n_kv_heads = k->ne[2];
        ثابت uint32_t dst_e_اندازه = (dst->type == HTP_TYPE_F32) ? حجم(عائم) : حجم(__fp16);
        ثابت bool can_تقسیم      = htp_tensor_can_ردیف_افراز(dst, dst_e_اندازه);

        // head range is contiguous in flat row space only when neq3 == 1
        إذا (kparams->سر_تقسیم && can_تقسیم && neq3 == 1 && n_kv_heads >= mdev_شمار && n_kv_heads % mdev_شمار == 0) {
            ثابت uint32_t G              = kparams->G;
            ثابت uint32_t kv_در_هر_core    = n_kv_heads / mdev_شمار;
            ثابت uint32_t heads_در_هر_core = kv_در_هر_core * G;
            ثابت uint32_t سر_شروع     = mdev_idx * heads_در_هر_core;
            qrow_شروع = سر_شروع * neq1;
            qrows      = heads_در_هر_core * neq1;
        } وإلا {
            ثابت بنية htp_tensor_mdev_range range = htp_tensor_mdev_افراز(مجموع_qrows, can_تقسیم ? 1 : 0, mdev_idx, mdev_شمار, &octx->ctx->mdev.شمار_div);
            qrow_شروع = range.شروع;
            qrows      = range.شمار;
        }
    }

    إذا (qrows == 0) {
        أرجع HTP_وضعیت_OK;
    }

    ثابت uint32_t n_نخها = octx->n_نخها;

    factx.qrows            = qrows;
    factx.qrow_شروع       = qrow_شروع;
    factx.qrows_در_هر_thread = fastdiv(qrows + n_نخها - 1, &octx->n_نخها_div);

    ثابت bool has_ماسک  = (ماسک != عدم);
    ثابت bool has_sinks = (octx->src[4] != عدم);
    بنية hvx_fa_vtcm_layout L;
    hvx_fa_vtcm_layout_build(&L, k->ne[0], v->ne[0], factx.is_q_fp32, has_ماسک, has_sinks, n_سر, n_نخها);

    إذا (L.مجموع_بایتها > octx->ctx->vtcm_اندازه) {
        أرجع HTP_وضعیت_VTCM_TOO_SMALL;
    }

    factx.اندازه_q_بلوک = L.اندازه_q_بلوک;
    factx.اندازه_k_بلوک = L.اندازه_k_بلوک;
    factx.اندازه_v_بلوک = L.اندازه_v_بلوک;
    factx.اندازه_m_بلوک = L.اندازه_m_بلوک;
    factx.اندازه_vkq_acc = L.اندازه_vkq_acc;

    uint8_t * ثابت پایه = octx->ctx->vtcm_پایه;

    factx.spad_q               = VTCM_LAYOUT_PTR(uint8_t, پایه, L.off_q);
    factx.spad_k               = VTCM_LAYOUT_PTR(uint8_t, پایه, L.off_k);
    factx.spad_v               = VTCM_LAYOUT_PTR(uint8_t, پایه, L.off_v);
    factx.spad_m               = VTCM_LAYOUT_PTR_OPTIONAL(uint8_t, پایه, L.off_m, has_ماسک);
    factx.spad_a               = VTCM_LAYOUT_PTR(uint8_t, پایه, L.off_a);
    factx.spad_sinks           = VTCM_LAYOUT_PTR_OPTIONAL(عائم, پایه, L.off_sinks, has_sinks);
    factx.n_فعال_بلوکها      = factx.n_بلوکها;
    factx.shared_فعال_بلوکها = false;

    إذا (has_ماسک && neq1 == 1 && neq3 == 1 && ماسک->ne[2] == 1 && factx.n_بلوکها > 1 && factx.n_بلوکها <= FA_MAX_KV_بلوکها) {
        factx.n_فعال_بلوکها = fa_پیمایش_فعال_بلوکها(
            octx->ctx->dma[0], ماسک, 0, 1, 0, 0, k->ne[1], FLASH_ATTN_بلوک_اندازه,
            factx.n_بلوکها, factx.spad_k, factx.اندازه_k_بلوک * 2, factx.فعال_بلوکها);
        factx.shared_فعال_بلوکها = true;
    }

    إذا (has_sinks) {
        ثابت بنية htp_tensor * sinks = octx->src[4];
        dma_صف * dma_q = octx->ctx->dma[0];
        dma_صف_افزودن(dma_q, dma_make_داده(factx.spad_sinks, sinks->داده), L.اندازه_sinks, 0, sinks->اندازه, 1);
        dma_صف_برداشت(dma_q);
    }

    work_صف_پیاپی(octx->ctx->work_صف, flash_attn_ext_f16_thread, &factx, n_نخها);

    أرجع HTP_وضعیت_OK;
}
