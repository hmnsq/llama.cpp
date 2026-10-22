#لارښوونه clang diagnostic ignored "-Wunused-variable"
#لارښوونه clang diagnostic ignored "-Wunused-function"
#لارښوونه clang diagnostic ignored "-Wunused-but-set-variable"

#ضمن <HAP_farf.h>
#ضمن <HAP_perf.h>

#ضمن <math.h>
#ضمن <string.h>

#ضمن "dma-queue.h"
#ضمن "work-queue.h"
#ضمن "hvx-utils.h"
#ضمن "hex-fastdiv.h"
#ضمن "hex-common.h"
#ضمن "hex-profile.h"

#تعریف_کړه GGML_COMMON_DECL_C
#ضمن "ggml-common.h"
#ضمن "htp-ctx.h"
#ضمن "htp-ops.h"
#ضمن "htp-tensor.h"
#ضمن "htp-vtcm.h"
#ضمن "htp/softmax-ops.h"
#ضمن "hvx-flash-attn.h"

بنية htp_softmax_context {
    بنية htp_ops_context * octx;
    ثابت بنية htp_softmax_kernel_params * kparams;

    فراغ *   compute;

    dma_addr_t معلومات_src0;
    dma_addr_t معلومات_src1;
    dma_addr_t معلومات_dst;

    uint8_t * vtcm_src0;
    uint8_t * vtcm_src1;
    uint8_t * vtcm_dst;

    uint32_t  vtcm_src0_اندازه_فی_thread;
    uint32_t  vtcm_src1_اندازه_فی_thread;
    uint32_t  vtcm_dst_اندازه_فی_thread;

    uint32_t  src0_spad_half_اندازه;
    uint32_t  src1_spad_half_اندازه;
    uint32_t  dst_spad_half_اندازه;

    uint32_t  src0_لیکه_اندازه_aligned;
    uint32_t  src1_لیکه_اندازه_aligned;
    uint32_t  dst_لیکه_اندازه_aligned;

    bool     use_f16;

    uint32_t n_سر;
    uint32_t n_سر_log2;

    عائم    scale;
    عائم    max_bias;
    عائم    m0;
    عائم    m1;

    بنية fastdiv_ارزښتونه div_ne01;
    بنية fastdiv_ارزښتونه div_ne02;
    بنية fastdiv_ارزښتونه div_ne12;
    بنية fastdiv_ارزښتونه div_ne13;

    uint32_t src0_nrows_فی_thread;
    uint32_t لیکه_پیل;
    uint32_t nrows;

    عائم    slopes[512] __attribute__((aligned(128)));
};

تعريفنوع فراغ (*softmax_compute_fn_t)(
    فراغ * محدود dst,
    ثابت فراغ * محدود src0,
    ثابت فراغ * محدود ماسک,
    uint32_t ne00,
    عائم scale,
    عائم میلان
);

ساكن فراغ hvx_fast_softmax_prep_f16(ثابت uint8_t * محدود src,
                                      uint8_t * محدود dst,
                                      ثابت صحيح عدد_elems,
                                      عائم     scale,
                                      ثابت uint8_t * محدود ماسک,
                                      عائم میلان) {
    ثابت HVX_ویکټور * محدود v_src  = (ثابت HVX_ویکټور *) src;
    HVX_ویکټور * محدود v_dst        = (HVX_ویکټور *) dst;
    ثابت HVX_ویکټور * محدود v_ماسک = (ثابت HVX_ویکټور *) ماسک;

    HVX_ویکټور scale_vec = hvx_vec_splat_f32(scale);
    HVX_ویکټور میلان_vec = hvx_vec_splat_f32(میلان);

    ثابت صحيح nvec_64 = عدد_elems / VLEN_FP16;
    ثابت صحيح nloe_64 = عدد_elems % VLEN_FP16;

    #لارښوونه unroll(2)
    لكل (صحيح i = 0; i < nvec_64; i++) {
        HVX_ویکټور_Pair p = hvx_vec_f16_منزل_f32(v_ماسک[i]);
        HVX_ویکټور m0 = Q6_V_lo_W(p);
        HVX_ویکټور m1 = Q6_V_hi_W(p);

        HVX_ویکټور s0 = v_src[2 * i];
        HVX_ویکټور s1 = v_src[2 * i + 1];

        HVX_ویکټور v0 = Q6_Vqf32_vadd_Vqf32Vqf32(Q6_Vqf32_vmpy_VsfVsf(s0, scale_vec), Q6_Vqf32_vmpy_VsfVsf(m0, میلان_vec));
        HVX_ویکټور v1 = Q6_Vqf32_vadd_Vqf32Vqf32(Q6_Vqf32_vmpy_VsfVsf(s1, scale_vec), Q6_Vqf32_vmpy_VsfVsf(m1, میلان_vec));

        v_dst[2 * i]     = Q6_Vsf_equals_Vqf32(v0);
        v_dst[2 * i + 1] = Q6_Vsf_equals_Vqf32(v1);
    }

    إذا (nloe_64 > 0) {
        HVX_ویکټور_Pair p = hvx_vec_f16_منزل_f32(v_ماسک[nvec_64]);
        HVX_ویکټور m0 = Q6_V_lo_W(p);

        HVX_ویکټور s0 = v_src[2 * nvec_64];
        HVX_ویکټور v0 = Q6_Vqf32_vadd_Vqf32Vqf32(Q6_Vqf32_vmpy_VsfVsf(s0, scale_vec), Q6_Vqf32_vmpy_VsfVsf(m0, میلان_vec));

        إذا (nloe_64 <= VLEN_FP32) {
            hvx_vec_store_a(&v_dst[2 * nvec_64], nloe_64 * حجم(عائم), Q6_Vsf_equals_Vqf32(v0));
        } وإلا {
            v_dst[2 * nvec_64] = Q6_Vsf_equals_Vqf32(v0);

            HVX_ویکټور m1 = Q6_V_hi_W(p);
            HVX_ویکټور s1 = v_src[2 * nvec_64 + 1];
            HVX_ویکټور v1 = Q6_Vqf32_vadd_Vqf32Vqf32(Q6_Vqf32_vmpy_VsfVsf(s1, scale_vec), Q6_Vqf32_vmpy_VsfVsf(m1, میلان_vec));

            hvx_vec_store_a(&v_dst[2 * nvec_64 + 1], (nloe_64 - VLEN_FP32) * حجم(عائم), Q6_Vsf_equals_Vqf32(v1));
        }
    }
}

ساكن فراغ hvx_fast_softmax_prep_f32(ثابت uint8_t * محدود src,
                                      uint8_t * محدود dst,
                                      ثابت صحيح عدد_elems,
                                      عائم     scale,
                                      ثابت uint8_t * محدود ماسک,
                                      عائم میلان) {
    ثابت HVX_ویکټور * محدود v_src  = (ثابت HVX_ویکټور *) src;
    HVX_ویکټور * محدود v_dst        = (HVX_ویکټور *) dst;
    ثابت HVX_ویکټور * محدود v_ماسک = (ثابت HVX_ویکټور *) ماسک;

    HVX_ویکټور scale_vec = hvx_vec_splat_f32(scale);
    HVX_ویکټور میلان_vec = hvx_vec_splat_f32(میلان);

    ثابت صحيح nvec = عدد_elems / VLEN_FP32;
    ثابت صحيح nloe = عدد_elems % VLEN_FP32;

    #لارښوونه unroll(4)
    لكل (صحيح i = 0; i < nvec; i++) {
        HVX_ویکټور v1 = v_src[i];
        HVX_ویکټور v3 = v_ماسک[i];

        HVX_ویکټور v2 = Q6_Vqf32_vmpy_VsfVsf(v1, scale_vec);
        HVX_ویکټور v4 = Q6_Vqf32_vmpy_VsfVsf(v3, میلان_vec);
        HVX_ویکټور v5 = Q6_Vqf32_vadd_Vqf32Vqf32(v2, v4);

        v_dst[i] = Q6_Vsf_equals_Vqf32(v5);
    }

    إذا (nloe > 0) {
        HVX_ویکټور v1 = v_src[nvec];
        HVX_ویکټور v3 = v_ماسک[nvec];

        HVX_ویکټور v2 = Q6_Vqf32_vmpy_VsfVsf(v1, scale_vec);
        HVX_ویکټور v4 = Q6_Vqf32_vmpy_VsfVsf(v3, میلان_vec);
        HVX_ویکټور v5 = Q6_Vqf32_vadd_Vqf32Vqf32(v2, v4);

        hvx_vec_store_a(&v_dst[nvec], nloe * حجم(عائم), Q6_Vsf_equals_Vqf32(v5));
    }
}

ساكن فراغ hvx_fast_softmax_f32(ثابت uint8_t * محدود src, uint8_t * محدود dst, ثابت صحيح عدد_elems) {
    ثابت HVX_ویکټور * محدود v_src = (ثابت HVX_ویکټور *) src;
    HVX_ویکټور * محدود v_dst       = (HVX_ویکټور *) dst;

    ثابت صحيح nvec = عدد_elems / VLEN_FP32;
    ثابت صحيح nloe = عدد_elems % VLEN_FP32;

    HVX_ویکټور max_vec = hvx_vec_splat_f32(((ثابت عائم *) src)[0]);

    #لارښوونه unroll(2)
    لكل (صحيح i = 0; i < nvec; i++) {
        HVX_ویکټور v1 = v_src[i];
        max_vec       = Q6_Vsf_vmax_VsfVsf(max_vec, v1);
    }

    إذا (nloe > 0) {
        HVX_ویکټور_Pred q_ماسک = Q6_Q_vsetq_R(nloe * حجم(عائم));
        HVX_ویکټور neg_inf    = hvx_vec_splat_f32(-INFINITY);
        HVX_ویکټور v_لکۍ     = Q6_V_vmux_QVV(q_ماسک, v_src[nvec], neg_inf);
        max_vec               = Q6_Vsf_vmax_VsfVsf(max_vec, v_لکۍ);
    }

    max_vec = hvx_vec_reduce_max_f32(max_vec);

    HVX_ویکټور مجموعه_vec = Q6_V_vsplat_R(0x00000000);

    #لارښوونه unroll(2)
    لكل (صحيح i = 0; i < nvec; i++) {
        HVX_ویکټور v1 = v_src[i];
        HVX_ویکټور v2 = Q6_Vqf32_vsub_VsfVsf(v1, max_vec);

        HVX_ویکټور v3 = hvx_vec_exp_f32(Q6_Vsf_equals_Vqf32(v2));

        مجموعه_vec = Q6_Vqf32_vadd_VsfVsf(Q6_Vsf_equals_Vqf32(مجموعه_vec), v3);

        v_dst[i] = v3;
    }

    إذا (nloe > 0) {
        HVX_ویکټور_Pred q_ماسک = Q6_Q_vsetq_R(nloe * حجم(عائم));
        HVX_ویکټور v1     = v_src[nvec];
        HVX_ویکټور v2     = Q6_Vqf32_vsub_VsfVsf(v1, max_vec);
        HVX_ویکټور v3     = hvx_vec_exp_f32(Q6_Vsf_equals_Vqf32(v2));
        HVX_ویکټور v3_pad = Q6_V_vmux_QVV(q_ماسک, v3, Q6_V_vzero());

        مجموعه_vec     = Q6_Vqf32_vadd_VsfVsf(Q6_Vsf_equals_Vqf32(مجموعه_vec), v3_pad);
        v_dst[nvec] = v3_pad;
    }

    مجموعه_vec = hvx_vec_reduce_مجموعه_f32(Q6_Vsf_equals_Vqf32(مجموعه_vec));

    HVX_ویکټور_Pred pos_مجموعه   = Q6_Q_vcmp_gt_VwVw(مجموعه_vec, Q6_V_vzero());
    HVX_ویکټور     v4        = hvx_vec_inverse_f32(مجموعه_vec);
    HVX_ویکټور     scale_vec = Q6_V_vmux_QVV(pos_مجموعه, v4, hvx_vec_splat_f32(1.0f));

    #لارښوونه unroll(2)
    لكل (صحيح i = 0; i < nvec; i++) {
        HVX_ویکټور v1 = v_dst[i];
        HVX_ویکټور v2 = Q6_Vqf32_vmpy_VsfVsf(v1, scale_vec);
        v_dst[i]      = Q6_Vsf_equals_Vqf32(v2);
    }

    إذا (nloe > 0) {
        HVX_ویکټور v1 = v_dst[nvec];
        HVX_ویکټور v2 = Q6_Vqf32_vmpy_VsfVsf(v1, scale_vec);
        hvx_vec_store_a(&v_dst[nvec], nloe * حجم(عائم), Q6_Vsf_equals_Vqf32(v2));
    }
}

ساكن فراغ compute_fast_softmax_f32_nomask(
    فراغ * محدود dst,
    ثابت فراغ * محدود src0,
    ثابت فراغ * محدود ماسک,
    uint32_t ne00,
    عائم scale,
    عائم میلان
) {
    (فراغ) ماسک;
    (فراغ) میلان;
    hvx_scale_f32((uint8_t *) dst, (ثابت uint8_t *) src0, ne00, scale);
    hvx_fast_softmax_f32((ثابت uint8_t *) dst, (uint8_t *) dst, ne00);
}

ساكن فراغ compute_fast_softmax_f32_ماسک_f32(
    فراغ * محدود dst,
    ثابت فراغ * محدود src0,
    ثابت فراغ * محدود ماسک,
    uint32_t ne00,
    عائم scale,
    عائم میلان
) {
    hvx_fast_softmax_prep_f32((ثابت uint8_t *) src0, (uint8_t *) dst, ne00, scale, (ثابت uint8_t *) ماسک, میلان);
    hvx_fast_softmax_f32((ثابت uint8_t *) dst, (uint8_t *) dst, ne00);
}

ساكن فراغ compute_fast_softmax_f32_ماسک_f16(
    فراغ * محدود dst,
    ثابت فراغ * محدود src0,
    ثابت فراغ * محدود ماسک,
    uint32_t ne00,
    عائم scale,
    عائم میلان
) {
    hvx_fast_softmax_prep_f16((ثابت uint8_t *) src0, (uint8_t *) dst, ne00, scale, (ثابت uint8_t *) ماسک, میلان);
    hvx_fast_softmax_f32((ثابت uint8_t *) dst, (uint8_t *) dst, ne00);
}

ساكن ثابت softmax_compute_fn_t softmax_kernels[HTP_SOFTMAX_KERNEL_شمېر] = {
    [HTP_SOFTMAX_KERNEL_NOMASK]   = compute_fast_softmax_f32_nomask,
    [HTP_SOFTMAX_KERNEL_ماسک_F32] = compute_fast_softmax_f32_ماسک_f32,
    [HTP_SOFTMAX_KERNEL_ماسک_F16] = compute_fast_softmax_f32_ماسک_f16,
};

ساكن فراغ softmax_thread_dma(بینښه صحيح nth, بینښه صحيح ith, فراغ * معلومات) {
    (فراغ) nth;
    ثابت بنية htp_softmax_context * smctx = (ثابت بنية htp_softmax_context *) معلومات;
    بنية htp_ops_context * octx = smctx->octx;
    ثابت بنية htp_tensor * src0 = octx->src[0];
    ثابت بنية htp_tensor * dst  = octx->dst;
    بنية htp_thread_trace * tr = &octx->ctx->trace[ith];

    ثابت uint32_t src0_nrows            = smctx->nrows;
    ثابت uint32_t src0_nrows_فی_thread = smctx->src0_nrows_فی_thread;

    ثابت uint32_t src0_پیل_لیکه = smctx->لیکه_پیل + src0_nrows_فی_thread * ith;
    ثابت uint32_t src0_پای_ټکی_لیکه   = MIN(src0_پیل_لیکه + src0_nrows_فی_thread, smctx->لیکه_پیل + src0_nrows);

    إذا (src0_پیل_لیکه >= src0_پای_ټکی_لیکه) {
        أرجع;
    }

    ثابت dma_addr_t معلومات_src0 = smctx->معلومات_src0;
    ثابت dma_addr_t معلومات_dst  = smctx->معلومات_dst;

    ثابت size_t src0_لیکه_اندازه = src0->ne[0] * حجم(عائم);
    ثابت size_t dst_لیکه_اندازه  = src0->ne[0] * حجم(عائم);

    uint8_t * src0_vtcm_بنسټ = smctx->vtcm_src0 + (ith * smctx->vtcm_src0_اندازه_فی_thread);
    uint8_t * dst_vtcm_بنسټ  = smctx->vtcm_dst  + (ith * smctx->vtcm_dst_اندازه_فی_thread);

    ثابت size_t src0_vtcm_half = smctx->src0_spad_half_اندازه;
    ثابت size_t dst_vtcm_half  = smctx->dst_spad_half_اندازه;

    dma_کتار * dma_q = octx->ctx->dma[ith];

    لكل (uint32_t r = src0_پیل_لیکه, idx = 0; r < src0_پای_ټکی_لیکه && idx < 2; r++, idx++) {
        dma_addr_t cur_dst  = معلومات_dst  + r * dst_لیکه_اندازه;
        dma_addr_t cur_src0 = معلومات_src0 + r * src0_لیکه_اندازه;
        فراغ * d_spad = dst_vtcm_بنسټ  + idx * dst_vtcm_half;
        فراغ * s_spad = src0_vtcm_بنسټ + idx * src0_vtcm_half;

        dma_کتار_ټېل(dma_q, dma_make_معلومات(cur_dst, d_spad),
                       dst_لیکه_اندازه, smctx->dst_لیکه_اندازه_aligned, dst_لیکه_اندازه, 0);
        dma_کتار_ټېل(dma_q, dma_make_معلومات(s_spad, cur_src0),
                       smctx->src0_لیکه_اندازه_aligned, src0_لیکه_اندازه, src0_لیکه_اندازه, 1);
    }

    softmax_compute_fn_t compute = (softmax_compute_fn_t) smctx->compute;
    ثابت uint32_t ne00 = src0->ne[0];

    لكل (uint32_t r = src0_پیل_لیکه; r < src0_پای_ټکی_لیکه; ++r) {
        فراغ * d_spad = (فراغ *) dma_کتار_راوباسه(dma_q).src;
        فراغ * s_spad = (فراغ *) dma_کتار_راوباسه(dma_q).dst;

        htp_trace_event_پیل(tr, HTP_TRACE_EVT_HVX_COMP, r);
        compute(d_spad, s_spad, عدم, ne00, smctx->scale, 1.0f);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, r);

        dma_addr_t cur_dst = معلومات_dst + r * dst_لیکه_اندازه;
        dma_کتار_ټېل(dma_q, dma_make_معلومات(cur_dst, d_spad),
                       dst_لیکه_اندازه, smctx->dst_لیکه_اندازه_aligned, dst_لیکه_اندازه, 1);

        ثابت uint32_t بل_r = r + 2;
        إذا (بل_r < src0_پای_ټکی_لیکه) {
            dma_addr_t بل_src0 = معلومات_src0 + بل_r * src0_لیکه_اندازه;
            dma_کتار_ټېل(dma_q, dma_make_معلومات(s_spad, بل_src0),
                           smctx->src0_لیکه_اندازه_aligned, src0_لیکه_اندازه, src0_لیکه_اندازه, 1);
        }
    }

    dma_کتار_flush(dma_q);
}

ساكن فراغ softmax_thread_ماسک_dma(بینښه صحيح nth, بینښه صحيح ith, فراغ * معلومات) {
    (فراغ) nth;
    ثابت بنية htp_softmax_context * smctx = (ثابت بنية htp_softmax_context *) معلومات;
    بنية htp_ops_context * octx = smctx->octx;
    ثابت بنية htp_tensor * src0 = octx->src[0];
    ثابت بنية htp_tensor * src1 = octx->src[1];
    ثابت بنية htp_tensor * dst  = octx->dst;
    بنية htp_thread_trace * tr = &octx->ctx->trace[ith];

    ثابت uint32_t src0_nrows            = smctx->nrows;
    ثابت uint32_t src0_nrows_فی_thread = smctx->src0_nrows_فی_thread;

    ثابت uint32_t src0_پیل_لیکه = smctx->لیکه_پیل + src0_nrows_فی_thread * ith;
    ثابت uint32_t src0_پای_ټکی_لیکه   = MIN(src0_پیل_لیکه + src0_nrows_فی_thread, smctx->لیکه_پیل + src0_nrows);

    إذا (src0_پیل_لیکه >= src0_پای_ټکی_لیکه) {
        أرجع;
    }

    ثابت dma_addr_t معلومات_src0 = smctx->معلومات_src0;
    ثابت dma_addr_t معلومات_src1 = smctx->معلومات_src1;
    ثابت dma_addr_t معلومات_dst  = smctx->معلومات_dst;

    ثابت size_t src0_لیکه_اندازه = src0->ne[0] * حجم(عائم);
    ثابت size_t dst_لیکه_اندازه  = src0->ne[0] * حجم(عائم);
    ثابت size_t ماسک_لیکه_اندازه = smctx->use_f16 ? (src1->ne[0] * حجم(__fp16)) : (src1->ne[0] * حجم(عائم));

    uint8_t * src0_vtcm_بنسټ = smctx->vtcm_src0 + (ith * smctx->vtcm_src0_اندازه_فی_thread);
    uint8_t * src1_vtcm_بنسټ = smctx->vtcm_src1 + (ith * smctx->vtcm_src1_اندازه_فی_thread);
    uint8_t * dst_vtcm_بنسټ  = smctx->vtcm_dst  + (ith * smctx->vtcm_dst_اندازه_فی_thread);

    ثابت size_t src0_vtcm_half = smctx->src0_spad_half_اندازه;
    ثابت size_t src1_vtcm_half = smctx->src1_spad_half_اندازه;
    ثابت size_t dst_vtcm_half  = smctx->dst_spad_half_اندازه;

    ثابت uint32_t nb11 = src1->nb[1];
    ثابت uint32_t nb12 = src1->nb[2];
    ثابت uint32_t nb13 = src1->nb[3];

    ثابت uint32_t ne00 = src0->ne[0];
    ثابت uint32_t ne01 = src0->ne[1];
    ثابت uint32_t ne02 = src0->ne[2];
    ثابت uint32_t ne03 = src0->ne[3];
    ثابت uint32_t ne12 = src1->ne[2];
    ثابت uint32_t ne13 = src1->ne[3];

    ثابت بنية fastdiv_ارزښتونه * div_ne01 = &smctx->div_ne01;
    ثابت بنية fastdiv_ارزښتونه * div_ne02 = &smctx->div_ne02;
    ثابت بنية fastdiv_ارزښتونه * div_ne12 = &smctx->div_ne12;
    ثابت بنية fastdiv_ارزښتونه * div_ne13 = &smctx->div_ne13;

    dma_کتار * dma_q = octx->ctx->dma[ith];

    لكل (uint32_t r = src0_پیل_لیکه, idx = 0; r < src0_پای_ټکی_لیکه && idx < 2; r++, idx++) {
        dma_addr_t cur_dst  = معلومات_dst  + r * dst_لیکه_اندازه;
        dma_addr_t cur_src0 = معلومات_src0 + r * src0_لیکه_اندازه;

        uint32_t i1 = fastmodulo(r, ne01, div_ne01);
        uint32_t r_div_ne01 = fastdiv(r, div_ne01);
        uint32_t i2 = fastmodulo(r_div_ne01, ne02, div_ne02);
        uint32_t i3 = fastdiv(r_div_ne01, div_ne02);
        uint32_t i12 = (ne12 == ne02) ? i2 : fastmodulo(i2, ne12, div_ne12);
        uint32_t i13 = (ne13 == ne03) ? i3 : fastmodulo(i3, ne13, div_ne13);
        dma_addr_t cur_src1 = معلومات_src1 + i1 * nb11 + i12 * nb12 + i13 * nb13;

        فراغ * d_spad = dst_vtcm_بنسټ  + idx * dst_vtcm_half;
        فراغ * s_spad = src0_vtcm_بنسټ + idx * src0_vtcm_half;
        فراغ * m_spad = src1_vtcm_بنسټ + idx * src1_vtcm_half;

        dma_کتار_ټېل(dma_q, dma_make_معلومات(cur_dst, d_spad),
                       dst_لیکه_اندازه, smctx->dst_لیکه_اندازه_aligned, dst_لیکه_اندازه, 0);
        dma_کتار_ټېل(dma_q, dma_make_معلومات(s_spad, cur_src0),
                       smctx->src0_لیکه_اندازه_aligned, src0_لیکه_اندازه, src0_لیکه_اندازه, 1);
        dma_کتار_ټېل(dma_q, dma_make_معلومات(m_spad, cur_src1),
                       smctx->src1_لیکه_اندازه_aligned, ماسک_لیکه_اندازه, ماسک_لیکه_اندازه, 1);
    }

    softmax_compute_fn_t compute = (softmax_compute_fn_t) smctx->compute;
    ثابت bool has_bias = smctx->max_bias > 0.0f;
    uint32_t prev_i2 = (uint32_t)-1;
    عائم میلان = 1.0f;

    لكل (uint32_t r = src0_پیل_لیکه; r < src0_پای_ټکی_لیکه; ++r) {
        فراغ * d_spad = (فراغ *) (uintptr_t) dma_کتار_راوباسه(dma_q).src;
        فراغ * s_spad = (فراغ *) (uintptr_t) dma_کتار_راوباسه(dma_q).dst;
        فراغ * m_spad = (فراغ *) (uintptr_t) dma_کتار_راوباسه(dma_q).dst;

        إذا (has_bias) {
            uint32_t r_div_ne01 = fastdiv(r, div_ne01);
            uint32_t i2 = fastmodulo(r_div_ne01, ne02, div_ne02);
            إذا (i2 != prev_i2) {
                میلان = smctx->slopes[i2];
                prev_i2 = i2;
            }
        }

        htp_trace_event_پیل(tr, HTP_TRACE_EVT_HVX_COMP, r);
        compute(d_spad, s_spad, m_spad, ne00, smctx->scale, میلان);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, r);

        dma_addr_t cur_dst = معلومات_dst + r * dst_لیکه_اندازه;
        dma_کتار_ټېل(dma_q, dma_make_معلومات(cur_dst, d_spad),
                       dst_لیکه_اندازه, smctx->dst_لیکه_اندازه_aligned, dst_لیکه_اندازه, 1);

        ثابت uint32_t بل_r = r + 2;
        إذا (بل_r < src0_پای_ټکی_لیکه) {
            dma_addr_t بل_src0 = معلومات_src0 + بل_r * src0_لیکه_اندازه;

            uint32_t ni1 = fastmodulo(بل_r, ne01, div_ne01);
            uint32_t nr_div_ne01 = fastdiv(بل_r, div_ne01);
            uint32_t ni2 = fastmodulo(nr_div_ne01, ne02, div_ne02);
            uint32_t ni3 = fastdiv(nr_div_ne01, div_ne02);
            uint32_t ni12 = (ne12 == ne02) ? ni2 : fastmodulo(ni2, ne12, div_ne12);
            uint32_t ni13 = (ne13 == ne03) ? ni3 : fastmodulo(ni3, ne13, div_ne13);
            dma_addr_t بل_src1 = معلومات_src1 + ni1 * nb11 + ni12 * nb12 + ni13 * nb13;

            dma_کتار_ټېل(dma_q, dma_make_معلومات(s_spad, بل_src0),
                           smctx->src0_لیکه_اندازه_aligned, src0_لیکه_اندازه, src0_لیکه_اندازه, 1);
            dma_کتار_ټېل(dma_q, dma_make_معلومات(m_spad, بل_src1),
                           smctx->src1_لیکه_اندازه_aligned, ماسک_لیکه_اندازه, ماسک_لیکه_اندازه, 1);
        }
    }

    dma_کتار_flush(dma_q);
}

ساكن صحيح execute_op_softmax_f32(بنية htp_ops_context * octx) {
    ثابت بنية htp_tensor * src0 = octx->src[0];
    ثابت بنية htp_tensor * dst  = octx->dst;

    ثابت حرف * op_type = "softmax-f32";

    ثابت بنية htp_softmax_kernel_params * kparams =
        (ثابت بنية htp_softmax_kernel_params *) octx->kernel_params;

    إذا (!htp_ops_context_سېټ_n_تارونه(octx, kparams->n_تارونه)) {
        أرجع HTP_حالت_INVAL_PARAMS;
    }

    إذا (kparams->kernel_id >= HTP_SOFTMAX_KERNEL_شمېر) {
        أرجع HTP_حالت_INVAL_PARAMS;
    }

    إذا (octx->ctx->vtcm_اندازه < (size_t) kparams->vtcm_اندازه) {
        FARF(ERROR, "%s : current VTCM reservation %zu is too small, needed %u\n",
             op_type, octx->ctx->vtcm_اندازه, kparams->vtcm_اندازه);
        أرجع HTP_حالت_VTCM_TOO_SMALL;
    }

    ثابت uint32_t src0_nrows = src0->ne[1] * src0->ne[2] * src0->ne[3];
    ثابت size_t elem_اندازه    = حجم(عائم);
    ثابت size_t dst_لیکه_اندازه = dst->nb[1];

    uint32_t لیکه_پیل = 0;
    uint32_t nrows     = src0_nrows;

    إذا (octx->ctx->mdev.شمېر > 1) {
        uint32_t لیکې_فی_chunk = 0;
        htp_tensor_mdev_لیکې_فی_chunk(dst, (uint32_t) elem_اندازه, (uint32_t) dst_لیکه_اندازه, &لیکې_فی_chunk);
        ثابت بنية htp_tensor_mdev_range range = htp_tensor_mdev_برخه_کول(
            src0_nrows, لیکې_فی_chunk, octx->ctx->mdev.idx, octx->ctx->mdev.شمېر, &octx->ctx->mdev.شمېر_div);
        لیکه_پیل = range.پیل;
        nrows     = range.شمېر;
        إذا (nrows < octx->n_تارونه) {
            htp_ops_context_سېټ_n_تارونه(octx, nrows ? nrows : 1);
        }
    }

    إذا (nrows == 0) {
        أرجع HTP_حالت_OK;
    }

    ثابت uint32_t n_تارونه = octx->n_تارونه;
    uint8_t * ثابت vtcm_بنسټ = (uint8_t *) octx->ctx->vtcm_بنسټ;

    ثابت uint32_t off_src0 = 0;
    ثابت uint32_t off_dst  = off_src0 + kparams->vtcm_src0_اندازه_فی_thread * kparams->n_تارونه;
    ثابت uint32_t off_src1 = off_dst  + kparams->vtcm_dst_اندازه_فی_thread  * kparams->n_تارونه;

    بنية htp_softmax_context smctx = {
        .octx                  = octx,
        .kparams               = kparams,
        .compute               = (فراغ *) softmax_kernels[kparams->kernel_id],

        .معلومات_src0             = src0->معلومات,
        .معلومات_src1             = kparams->use_src1 ? octx->src[1]->معلومات : 0,
        .معلومات_dst              = dst->معلومات,

        .vtcm_src0             = VTCM_LAYOUT_PTR(uint8_t, vtcm_بنسټ, off_src0),
        .vtcm_dst              = VTCM_LAYOUT_PTR(uint8_t, vtcm_بنسټ, off_dst),
        .vtcm_src1             = VTCM_LAYOUT_PTR_OPTIONAL(uint8_t, vtcm_بنسټ, off_src1, kparams->use_src1),

        .vtcm_src0_اندازه_فی_thread = kparams->vtcm_src0_اندازه_فی_thread,
        .vtcm_src1_اندازه_فی_thread = kparams->vtcm_src1_اندازه_فی_thread,
        .vtcm_dst_اندازه_فی_thread  = kparams->vtcm_dst_اندازه_فی_thread,

        .src0_spad_half_اندازه   = kparams->src0_spad_half_اندازه,
        .src1_spad_half_اندازه   = kparams->src1_spad_half_اندازه,
        .dst_spad_half_اندازه    = kparams->dst_spad_half_اندازه,

        .src0_لیکه_اندازه_aligned = kparams->src0_لیکه_اندازه_aligned,
        .src1_لیکه_اندازه_aligned = kparams->src1_لیکه_اندازه_aligned,
        .dst_لیکه_اندازه_aligned  = kparams->dst_لیکه_اندازه_aligned,

        .use_f16               = kparams->use_f16 != 0,

        .n_سر                = kparams->n_سر,
        .n_سر_log2           = kparams->n_سر_log2,

        .scale                 = kparams->scale,
        .max_bias              = kparams->max_bias,
        .m0                    = kparams->m0,
        .m1                    = kparams->m1,

        .div_ne01              = kparams->div_ne01,
        .div_ne02              = kparams->div_ne02,
        .div_ne12              = kparams->div_ne12,
        .div_ne13              = kparams->div_ne13,

        .src0_nrows_فی_thread = fastdiv(nrows + n_تارونه - 1, &octx->n_تارونه_div),
        .لیکه_پیل             = لیکه_پیل,
        .nrows                 = nrows,
    };

    إذا (kparams->max_bias > 0.0f && kparams->use_src1) {
        إذا (kparams->n_سر > 512) {
            أرجع HTP_حالت_INVAL_PARAMS;
        }
        لكل (uint32_t h = 0; h < kparams->n_سر; h += 32) {
            HVX_ویکټور v_slopes = hvx_alibi_slopes(h, 1, kparams->n_سر_log2, kparams->m0, kparams->m1);
            hvx_vmem(&smctx.slopes[h]) = v_slopes;
        }
    }

    work_کتار_func_t task_func = kparams->use_src1 ? softmax_thread_ماسک_dma : softmax_thread_dma;
    work_کتار_پرلپسې(octx->ctx->work_کتار, task_func, &smctx, n_تارونه);

    أرجع HTP_حالت_OK;
}

صحيح op_softmax(بنية htp_ops_context * octx) {
    صحيح err = HTP_حالت_OK;

    بدل (octx->src[0]->type) {
        حالة HTP_TYPE_F32:
            err = execute_op_softmax_f32(octx);
            اكسر;

        افتراضي:
            err = HTP_حالت_NO_SUPPORT;
            اكسر;
    }

    أرجع err;
}
