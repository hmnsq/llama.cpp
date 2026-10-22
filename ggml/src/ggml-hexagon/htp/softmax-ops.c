#ہدایت clang diagnostic ignored "-Wunused-variable"
#ہدایت clang diagnostic ignored "-Wunused-function"
#ہدایت clang diagnostic ignored "-Wunused-but-set-variable"

#شامل <HAP_farf.h>
#شامل <HAP_perf.h>

#شامل <math.h>
#شامل <string.h>

#شامل "dma-queue.h"
#شامل "work-queue.h"
#شامل "hvx-utils.h"
#شامل "hex-fastdiv.h"
#شامل "hex-common.h"
#شامل "hex-profile.h"

#تعریف_کرو GGML_COMMON_DECL_C
#شامل "ggml-common.h"
#شامل "htp-ctx.h"
#شامل "htp-ops.h"
#شامل "htp-tensor.h"
#شامل "htp-vtcm.h"
#شامل "htp/softmax-ops.h"
#شامل "hvx-flash-attn.h"

ساخت htp_softmax_context {
    ساخت htp_ops_context * octx;
    ثابت ساخت htp_softmax_kernel_params * kparams;

    خالی *   compute;

    dma_addr_t ڈیٹا_src0;
    dma_addr_t ڈیٹا_src1;
    dma_addr_t ڈیٹا_dst;

    uint8_t * vtcm_src0;
    uint8_t * vtcm_src1;
    uint8_t * vtcm_dst;

    uint32_t  vtcm_src0_سائز_فی_thread;
    uint32_t  vtcm_src1_سائز_فی_thread;
    uint32_t  vtcm_dst_سائز_فی_thread;

    uint32_t  src0_spad_half_سائز;
    uint32_t  src1_spad_half_سائز;
    uint32_t  dst_spad_half_سائز;

    uint32_t  src0_صف_سائز_aligned;
    uint32_t  src1_صف_سائز_aligned;
    uint32_t  dst_صف_سائز_aligned;

    bool     use_f16;

    uint32_t n_سر;
    uint32_t n_سر_log2;

    اعشاری    scale;
    اعشاری    max_bias;
    اعشاری    m0;
    اعشاری    m1;

    ساخت fastdiv_اقدار div_ne01;
    ساخت fastdiv_اقدار div_ne02;
    ساخت fastdiv_اقدار div_ne12;
    ساخت fastdiv_اقدار div_ne13;

    uint32_t src0_nrows_فی_thread;
    uint32_t صف_آغاز;
    uint32_t nrows;

    اعشاری    slopes[512] __attribute__((aligned(128)));
};

قسمتعریف خالی (*softmax_compute_fn_t)(
    خالی * محدود dst,
    ثابت خالی * محدود src0,
    ثابت خالی * محدود نقاب,
    uint32_t ne00,
    اعشاری scale,
    اعشاری ڈھلان
);

ساکن خالی hvx_fast_softmax_prep_f16(ثابت uint8_t * محدود src,
                                      uint8_t * محدود dst,
                                      ثابت صحیح عدد_elems,
                                      اعشاری     scale,
                                      ثابت uint8_t * محدود نقاب,
                                      اعشاری ڈھلان) {
    ثابت HVX_سمتیہ * محدود v_src  = (ثابت HVX_سمتیہ *) src;
    HVX_سمتیہ * محدود v_dst        = (HVX_سمتیہ *) dst;
    ثابت HVX_سمتیہ * محدود v_نقاب = (ثابت HVX_سمتیہ *) نقاب;

    HVX_سمتیہ scale_vec = hvx_vec_splat_f32(scale);
    HVX_سمتیہ ڈھلان_vec = hvx_vec_splat_f32(ڈھلان);

    ثابت صحیح nvec_64 = عدد_elems / VLEN_FP16;
    ثابت صحیح nloe_64 = عدد_elems % VLEN_FP16;

    #ہدایت unroll(2)
    برائے (صحیح i = 0; i < nvec_64; i++) {
        HVX_سمتیہ_Pair p = hvx_vec_f16_مقصد_f32(v_نقاب[i]);
        HVX_سمتیہ m0 = Q6_V_lo_W(p);
        HVX_سمتیہ m1 = Q6_V_hi_W(p);

        HVX_سمتیہ s0 = v_src[2 * i];
        HVX_سمتیہ s1 = v_src[2 * i + 1];

        HVX_سمتیہ v0 = Q6_Vqf32_vadd_Vqf32Vqf32(Q6_Vqf32_vmpy_VsfVsf(s0, scale_vec), Q6_Vqf32_vmpy_VsfVsf(m0, ڈھلان_vec));
        HVX_سمتیہ v1 = Q6_Vqf32_vadd_Vqf32Vqf32(Q6_Vqf32_vmpy_VsfVsf(s1, scale_vec), Q6_Vqf32_vmpy_VsfVsf(m1, ڈھلان_vec));

        v_dst[2 * i]     = Q6_Vsf_equals_Vqf32(v0);
        v_dst[2 * i + 1] = Q6_Vsf_equals_Vqf32(v1);
    }

    اگر (nloe_64 > 0) {
        HVX_سمتیہ_Pair p = hvx_vec_f16_مقصد_f32(v_نقاب[nvec_64]);
        HVX_سمتیہ m0 = Q6_V_lo_W(p);

        HVX_سمتیہ s0 = v_src[2 * nvec_64];
        HVX_سمتیہ v0 = Q6_Vqf32_vadd_Vqf32Vqf32(Q6_Vqf32_vmpy_VsfVsf(s0, scale_vec), Q6_Vqf32_vmpy_VsfVsf(m0, ڈھلان_vec));

        اگر (nloe_64 <= VLEN_FP32) {
            hvx_vec_store_a(&v_dst[2 * nvec_64], nloe_64 * حجم(اعشاری), Q6_Vsf_equals_Vqf32(v0));
        } ورنہ {
            v_dst[2 * nvec_64] = Q6_Vsf_equals_Vqf32(v0);

            HVX_سمتیہ m1 = Q6_V_hi_W(p);
            HVX_سمتیہ s1 = v_src[2 * nvec_64 + 1];
            HVX_سمتیہ v1 = Q6_Vqf32_vadd_Vqf32Vqf32(Q6_Vqf32_vmpy_VsfVsf(s1, scale_vec), Q6_Vqf32_vmpy_VsfVsf(m1, ڈھلان_vec));

            hvx_vec_store_a(&v_dst[2 * nvec_64 + 1], (nloe_64 - VLEN_FP32) * حجم(اعشاری), Q6_Vsf_equals_Vqf32(v1));
        }
    }
}

ساکن خالی hvx_fast_softmax_prep_f32(ثابت uint8_t * محدود src,
                                      uint8_t * محدود dst,
                                      ثابت صحیح عدد_elems,
                                      اعشاری     scale,
                                      ثابت uint8_t * محدود نقاب,
                                      اعشاری ڈھلان) {
    ثابت HVX_سمتیہ * محدود v_src  = (ثابت HVX_سمتیہ *) src;
    HVX_سمتیہ * محدود v_dst        = (HVX_سمتیہ *) dst;
    ثابت HVX_سمتیہ * محدود v_نقاب = (ثابت HVX_سمتیہ *) نقاب;

    HVX_سمتیہ scale_vec = hvx_vec_splat_f32(scale);
    HVX_سمتیہ ڈھلان_vec = hvx_vec_splat_f32(ڈھلان);

    ثابت صحیح nvec = عدد_elems / VLEN_FP32;
    ثابت صحیح nloe = عدد_elems % VLEN_FP32;

    #ہدایت unroll(4)
    برائے (صحیح i = 0; i < nvec; i++) {
        HVX_سمتیہ v1 = v_src[i];
        HVX_سمتیہ v3 = v_نقاب[i];

        HVX_سمتیہ v2 = Q6_Vqf32_vmpy_VsfVsf(v1, scale_vec);
        HVX_سمتیہ v4 = Q6_Vqf32_vmpy_VsfVsf(v3, ڈھلان_vec);
        HVX_سمتیہ v5 = Q6_Vqf32_vadd_Vqf32Vqf32(v2, v4);

        v_dst[i] = Q6_Vsf_equals_Vqf32(v5);
    }

    اگر (nloe > 0) {
        HVX_سمتیہ v1 = v_src[nvec];
        HVX_سمتیہ v3 = v_نقاب[nvec];

        HVX_سمتیہ v2 = Q6_Vqf32_vmpy_VsfVsf(v1, scale_vec);
        HVX_سمتیہ v4 = Q6_Vqf32_vmpy_VsfVsf(v3, ڈھلان_vec);
        HVX_سمتیہ v5 = Q6_Vqf32_vadd_Vqf32Vqf32(v2, v4);

        hvx_vec_store_a(&v_dst[nvec], nloe * حجم(اعشاری), Q6_Vsf_equals_Vqf32(v5));
    }
}

ساکن خالی hvx_fast_softmax_f32(ثابت uint8_t * محدود src, uint8_t * محدود dst, ثابت صحیح عدد_elems) {
    ثابت HVX_سمتیہ * محدود v_src = (ثابت HVX_سمتیہ *) src;
    HVX_سمتیہ * محدود v_dst       = (HVX_سمتیہ *) dst;

    ثابت صحیح nvec = عدد_elems / VLEN_FP32;
    ثابت صحیح nloe = عدد_elems % VLEN_FP32;

    HVX_سمتیہ max_vec = hvx_vec_splat_f32(((ثابت اعشاری *) src)[0]);

    #ہدایت unroll(2)
    برائے (صحیح i = 0; i < nvec; i++) {
        HVX_سمتیہ v1 = v_src[i];
        max_vec       = Q6_Vsf_vmax_VsfVsf(max_vec, v1);
    }

    اگر (nloe > 0) {
        HVX_سمتیہ_Pred q_نقاب = Q6_Q_vsetq_R(nloe * حجم(اعشاری));
        HVX_سمتیہ neg_inf    = hvx_vec_splat_f32(-INFINITY);
        HVX_سمتیہ v_دم     = Q6_V_vmux_QVV(q_نقاب, v_src[nvec], neg_inf);
        max_vec               = Q6_Vsf_vmax_VsfVsf(max_vec, v_دم);
    }

    max_vec = hvx_vec_reduce_max_f32(max_vec);

    HVX_سمتیہ مجموع_vec = Q6_V_vsplat_R(0x00000000);

    #ہدایت unroll(2)
    برائے (صحیح i = 0; i < nvec; i++) {
        HVX_سمتیہ v1 = v_src[i];
        HVX_سمتیہ v2 = Q6_Vqf32_vsub_VsfVsf(v1, max_vec);

        HVX_سمتیہ v3 = hvx_vec_exp_f32(Q6_Vsf_equals_Vqf32(v2));

        مجموع_vec = Q6_Vqf32_vadd_VsfVsf(Q6_Vsf_equals_Vqf32(مجموع_vec), v3);

        v_dst[i] = v3;
    }

    اگر (nloe > 0) {
        HVX_سمتیہ_Pred q_نقاب = Q6_Q_vsetq_R(nloe * حجم(اعشاری));
        HVX_سمتیہ v1     = v_src[nvec];
        HVX_سمتیہ v2     = Q6_Vqf32_vsub_VsfVsf(v1, max_vec);
        HVX_سمتیہ v3     = hvx_vec_exp_f32(Q6_Vsf_equals_Vqf32(v2));
        HVX_سمتیہ v3_pad = Q6_V_vmux_QVV(q_نقاب, v3, Q6_V_vzero());

        مجموع_vec     = Q6_Vqf32_vadd_VsfVsf(Q6_Vsf_equals_Vqf32(مجموع_vec), v3_pad);
        v_dst[nvec] = v3_pad;
    }

    مجموع_vec = hvx_vec_reduce_مجموع_f32(Q6_Vsf_equals_Vqf32(مجموع_vec));

    HVX_سمتیہ_Pred pos_مجموع   = Q6_Q_vcmp_gt_VwVw(مجموع_vec, Q6_V_vzero());
    HVX_سمتیہ     v4        = hvx_vec_inverse_f32(مجموع_vec);
    HVX_سمتیہ     scale_vec = Q6_V_vmux_QVV(pos_مجموع, v4, hvx_vec_splat_f32(1.0f));

    #ہدایت unroll(2)
    برائے (صحیح i = 0; i < nvec; i++) {
        HVX_سمتیہ v1 = v_dst[i];
        HVX_سمتیہ v2 = Q6_Vqf32_vmpy_VsfVsf(v1, scale_vec);
        v_dst[i]      = Q6_Vsf_equals_Vqf32(v2);
    }

    اگر (nloe > 0) {
        HVX_سمتیہ v1 = v_dst[nvec];
        HVX_سمتیہ v2 = Q6_Vqf32_vmpy_VsfVsf(v1, scale_vec);
        hvx_vec_store_a(&v_dst[nvec], nloe * حجم(اعشاری), Q6_Vsf_equals_Vqf32(v2));
    }
}

ساکن خالی compute_fast_softmax_f32_nomask(
    خالی * محدود dst,
    ثابت خالی * محدود src0,
    ثابت خالی * محدود نقاب,
    uint32_t ne00,
    اعشاری scale,
    اعشاری ڈھلان
) {
    (خالی) نقاب;
    (خالی) ڈھلان;
    hvx_scale_f32((uint8_t *) dst, (ثابت uint8_t *) src0, ne00, scale);
    hvx_fast_softmax_f32((ثابت uint8_t *) dst, (uint8_t *) dst, ne00);
}

ساکن خالی compute_fast_softmax_f32_نقاب_f32(
    خالی * محدود dst,
    ثابت خالی * محدود src0,
    ثابت خالی * محدود نقاب,
    uint32_t ne00,
    اعشاری scale,
    اعشاری ڈھلان
) {
    hvx_fast_softmax_prep_f32((ثابت uint8_t *) src0, (uint8_t *) dst, ne00, scale, (ثابت uint8_t *) نقاب, ڈھلان);
    hvx_fast_softmax_f32((ثابت uint8_t *) dst, (uint8_t *) dst, ne00);
}

ساکن خالی compute_fast_softmax_f32_نقاب_f16(
    خالی * محدود dst,
    ثابت خالی * محدود src0,
    ثابت خالی * محدود نقاب,
    uint32_t ne00,
    اعشاری scale,
    اعشاری ڈھلان
) {
    hvx_fast_softmax_prep_f16((ثابت uint8_t *) src0, (uint8_t *) dst, ne00, scale, (ثابت uint8_t *) نقاب, ڈھلان);
    hvx_fast_softmax_f32((ثابت uint8_t *) dst, (uint8_t *) dst, ne00);
}

ساکن ثابت softmax_compute_fn_t softmax_kernels[HTP_SOFTMAX_KERNEL_گنتی] = {
    [HTP_SOFTMAX_KERNEL_NOMASK]   = compute_fast_softmax_f32_nomask,
    [HTP_SOFTMAX_KERNEL_نقاب_F32] = compute_fast_softmax_f32_نقاب_f32,
    [HTP_SOFTMAX_KERNEL_نقاب_F16] = compute_fast_softmax_f32_نقاب_f16,
};

ساکن خالی softmax_thread_dma(غیرعلامتی صحیح nth, غیرعلامتی صحیح ith, خالی * ڈیٹا) {
    (خالی) nth;
    ثابت ساخت htp_softmax_context * smctx = (ثابت ساخت htp_softmax_context *) ڈیٹا;
    ساخت htp_ops_context * octx = smctx->octx;
    ثابت ساخت htp_tensor * src0 = octx->src[0];
    ثابت ساخت htp_tensor * dst  = octx->dst;
    ساخت htp_thread_trace * tr = &octx->ctx->trace[ith];

    ثابت uint32_t src0_nrows            = smctx->nrows;
    ثابت uint32_t src0_nrows_فی_thread = smctx->src0_nrows_فی_thread;

    ثابت uint32_t src0_آغاز_صف = smctx->صف_آغاز + src0_nrows_فی_thread * ith;
    ثابت uint32_t src0_انتہا_صف   = MIN(src0_آغاز_صف + src0_nrows_فی_thread, smctx->صف_آغاز + src0_nrows);

    اگر (src0_آغاز_صف >= src0_انتہا_صف) {
        واپس;
    }

    ثابت dma_addr_t ڈیٹا_src0 = smctx->ڈیٹا_src0;
    ثابت dma_addr_t ڈیٹا_dst  = smctx->ڈیٹا_dst;

    ثابت size_t src0_صف_سائز = src0->ne[0] * حجم(اعشاری);
    ثابت size_t dst_صف_سائز  = src0->ne[0] * حجم(اعشاری);

    uint8_t * src0_vtcm_بنیاد = smctx->vtcm_src0 + (ith * smctx->vtcm_src0_سائز_فی_thread);
    uint8_t * dst_vtcm_بنیاد  = smctx->vtcm_dst  + (ith * smctx->vtcm_dst_سائز_فی_thread);

    ثابت size_t src0_vtcm_half = smctx->src0_spad_half_سائز;
    ثابت size_t dst_vtcm_half  = smctx->dst_spad_half_سائز;

    dma_قطار * dma_q = octx->ctx->dma[ith];

    برائے (uint32_t r = src0_آغاز_صف, idx = 0; r < src0_انتہا_صف && idx < 2; r++, idx++) {
        dma_addr_t cur_dst  = ڈیٹا_dst  + r * dst_صف_سائز;
        dma_addr_t cur_src0 = ڈیٹا_src0 + r * src0_صف_سائز;
        خالی * d_spad = dst_vtcm_بنیاد  + idx * dst_vtcm_half;
        خالی * s_spad = src0_vtcm_بنیاد + idx * src0_vtcm_half;

        dma_قطار_ڈالو(dma_q, dma_make_ڈیٹا(cur_dst, d_spad),
                       dst_صف_سائز, smctx->dst_صف_سائز_aligned, dst_صف_سائز, 0);
        dma_قطار_ڈالو(dma_q, dma_make_ڈیٹا(s_spad, cur_src0),
                       smctx->src0_صف_سائز_aligned, src0_صف_سائز, src0_صف_سائز, 1);
    }

    softmax_compute_fn_t compute = (softmax_compute_fn_t) smctx->compute;
    ثابت uint32_t ne00 = src0->ne[0];

    برائے (uint32_t r = src0_آغاز_صف; r < src0_انتہا_صف; ++r) {
        خالی * d_spad = (خالی *) dma_قطار_نکالو(dma_q).src;
        خالی * s_spad = (خالی *) dma_قطار_نکالو(dma_q).dst;

        htp_trace_event_آغاز(tr, HTP_TRACE_EVT_HVX_COMP, r);
        compute(d_spad, s_spad, تہی, ne00, smctx->scale, 1.0f);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, r);

        dma_addr_t cur_dst = ڈیٹا_dst + r * dst_صف_سائز;
        dma_قطار_ڈالو(dma_q, dma_make_ڈیٹا(cur_dst, d_spad),
                       dst_صف_سائز, smctx->dst_صف_سائز_aligned, dst_صف_سائز, 1);

        ثابت uint32_t اگلا_r = r + 2;
        اگر (اگلا_r < src0_انتہا_صف) {
            dma_addr_t اگلا_src0 = ڈیٹا_src0 + اگلا_r * src0_صف_سائز;
            dma_قطار_ڈالو(dma_q, dma_make_ڈیٹا(s_spad, اگلا_src0),
                           smctx->src0_صف_سائز_aligned, src0_صف_سائز, src0_صف_سائز, 1);
        }
    }

    dma_قطار_flush(dma_q);
}

ساکن خالی softmax_thread_نقاب_dma(غیرعلامتی صحیح nth, غیرعلامتی صحیح ith, خالی * ڈیٹا) {
    (خالی) nth;
    ثابت ساخت htp_softmax_context * smctx = (ثابت ساخت htp_softmax_context *) ڈیٹا;
    ساخت htp_ops_context * octx = smctx->octx;
    ثابت ساخت htp_tensor * src0 = octx->src[0];
    ثابت ساخت htp_tensor * src1 = octx->src[1];
    ثابت ساخت htp_tensor * dst  = octx->dst;
    ساخت htp_thread_trace * tr = &octx->ctx->trace[ith];

    ثابت uint32_t src0_nrows            = smctx->nrows;
    ثابت uint32_t src0_nrows_فی_thread = smctx->src0_nrows_فی_thread;

    ثابت uint32_t src0_آغاز_صف = smctx->صف_آغاز + src0_nrows_فی_thread * ith;
    ثابت uint32_t src0_انتہا_صف   = MIN(src0_آغاز_صف + src0_nrows_فی_thread, smctx->صف_آغاز + src0_nrows);

    اگر (src0_آغاز_صف >= src0_انتہا_صف) {
        واپس;
    }

    ثابت dma_addr_t ڈیٹا_src0 = smctx->ڈیٹا_src0;
    ثابت dma_addr_t ڈیٹا_src1 = smctx->ڈیٹا_src1;
    ثابت dma_addr_t ڈیٹا_dst  = smctx->ڈیٹا_dst;

    ثابت size_t src0_صف_سائز = src0->ne[0] * حجم(اعشاری);
    ثابت size_t dst_صف_سائز  = src0->ne[0] * حجم(اعشاری);
    ثابت size_t نقاب_صف_سائز = smctx->use_f16 ? (src1->ne[0] * حجم(__fp16)) : (src1->ne[0] * حجم(اعشاری));

    uint8_t * src0_vtcm_بنیاد = smctx->vtcm_src0 + (ith * smctx->vtcm_src0_سائز_فی_thread);
    uint8_t * src1_vtcm_بنیاد = smctx->vtcm_src1 + (ith * smctx->vtcm_src1_سائز_فی_thread);
    uint8_t * dst_vtcm_بنیاد  = smctx->vtcm_dst  + (ith * smctx->vtcm_dst_سائز_فی_thread);

    ثابت size_t src0_vtcm_half = smctx->src0_spad_half_سائز;
    ثابت size_t src1_vtcm_half = smctx->src1_spad_half_سائز;
    ثابت size_t dst_vtcm_half  = smctx->dst_spad_half_سائز;

    ثابت uint32_t nb11 = src1->nb[1];
    ثابت uint32_t nb12 = src1->nb[2];
    ثابت uint32_t nb13 = src1->nb[3];

    ثابت uint32_t ne00 = src0->ne[0];
    ثابت uint32_t ne01 = src0->ne[1];
    ثابت uint32_t ne02 = src0->ne[2];
    ثابت uint32_t ne03 = src0->ne[3];
    ثابت uint32_t ne12 = src1->ne[2];
    ثابت uint32_t ne13 = src1->ne[3];

    ثابت ساخت fastdiv_اقدار * div_ne01 = &smctx->div_ne01;
    ثابت ساخت fastdiv_اقدار * div_ne02 = &smctx->div_ne02;
    ثابت ساخت fastdiv_اقدار * div_ne12 = &smctx->div_ne12;
    ثابت ساخت fastdiv_اقدار * div_ne13 = &smctx->div_ne13;

    dma_قطار * dma_q = octx->ctx->dma[ith];

    برائے (uint32_t r = src0_آغاز_صف, idx = 0; r < src0_انتہا_صف && idx < 2; r++, idx++) {
        dma_addr_t cur_dst  = ڈیٹا_dst  + r * dst_صف_سائز;
        dma_addr_t cur_src0 = ڈیٹا_src0 + r * src0_صف_سائز;

        uint32_t i1 = fastmodulo(r, ne01, div_ne01);
        uint32_t r_div_ne01 = fastdiv(r, div_ne01);
        uint32_t i2 = fastmodulo(r_div_ne01, ne02, div_ne02);
        uint32_t i3 = fastdiv(r_div_ne01, div_ne02);
        uint32_t i12 = (ne12 == ne02) ? i2 : fastmodulo(i2, ne12, div_ne12);
        uint32_t i13 = (ne13 == ne03) ? i3 : fastmodulo(i3, ne13, div_ne13);
        dma_addr_t cur_src1 = ڈیٹا_src1 + i1 * nb11 + i12 * nb12 + i13 * nb13;

        خالی * d_spad = dst_vtcm_بنیاد  + idx * dst_vtcm_half;
        خالی * s_spad = src0_vtcm_بنیاد + idx * src0_vtcm_half;
        خالی * m_spad = src1_vtcm_بنیاد + idx * src1_vtcm_half;

        dma_قطار_ڈالو(dma_q, dma_make_ڈیٹا(cur_dst, d_spad),
                       dst_صف_سائز, smctx->dst_صف_سائز_aligned, dst_صف_سائز, 0);
        dma_قطار_ڈالو(dma_q, dma_make_ڈیٹا(s_spad, cur_src0),
                       smctx->src0_صف_سائز_aligned, src0_صف_سائز, src0_صف_سائز, 1);
        dma_قطار_ڈالو(dma_q, dma_make_ڈیٹا(m_spad, cur_src1),
                       smctx->src1_صف_سائز_aligned, نقاب_صف_سائز, نقاب_صف_سائز, 1);
    }

    softmax_compute_fn_t compute = (softmax_compute_fn_t) smctx->compute;
    ثابت bool has_bias = smctx->max_bias > 0.0f;
    uint32_t پچھلا_i2 = (uint32_t)-1;
    اعشاری ڈھلان = 1.0f;

    برائے (uint32_t r = src0_آغاز_صف; r < src0_انتہا_صف; ++r) {
        خالی * d_spad = (خالی *) (uintptr_t) dma_قطار_نکالو(dma_q).src;
        خالی * s_spad = (خالی *) (uintptr_t) dma_قطار_نکالو(dma_q).dst;
        خالی * m_spad = (خالی *) (uintptr_t) dma_قطار_نکالو(dma_q).dst;

        اگر (has_bias) {
            uint32_t r_div_ne01 = fastdiv(r, div_ne01);
            uint32_t i2 = fastmodulo(r_div_ne01, ne02, div_ne02);
            اگر (i2 != پچھلا_i2) {
                ڈھلان = smctx->slopes[i2];
                پچھلا_i2 = i2;
            }
        }

        htp_trace_event_آغاز(tr, HTP_TRACE_EVT_HVX_COMP, r);
        compute(d_spad, s_spad, m_spad, ne00, smctx->scale, ڈھلان);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, r);

        dma_addr_t cur_dst = ڈیٹا_dst + r * dst_صف_سائز;
        dma_قطار_ڈالو(dma_q, dma_make_ڈیٹا(cur_dst, d_spad),
                       dst_صف_سائز, smctx->dst_صف_سائز_aligned, dst_صف_سائز, 1);

        ثابت uint32_t اگلا_r = r + 2;
        اگر (اگلا_r < src0_انتہا_صف) {
            dma_addr_t اگلا_src0 = ڈیٹا_src0 + اگلا_r * src0_صف_سائز;

            uint32_t ni1 = fastmodulo(اگلا_r, ne01, div_ne01);
            uint32_t nr_div_ne01 = fastdiv(اگلا_r, div_ne01);
            uint32_t ni2 = fastmodulo(nr_div_ne01, ne02, div_ne02);
            uint32_t ni3 = fastdiv(nr_div_ne01, div_ne02);
            uint32_t ni12 = (ne12 == ne02) ? ni2 : fastmodulo(ni2, ne12, div_ne12);
            uint32_t ni13 = (ne13 == ne03) ? ni3 : fastmodulo(ni3, ne13, div_ne13);
            dma_addr_t اگلا_src1 = ڈیٹا_src1 + ni1 * nb11 + ni12 * nb12 + ni13 * nb13;

            dma_قطار_ڈالو(dma_q, dma_make_ڈیٹا(s_spad, اگلا_src0),
                           smctx->src0_صف_سائز_aligned, src0_صف_سائز, src0_صف_سائز, 1);
            dma_قطار_ڈالو(dma_q, dma_make_ڈیٹا(m_spad, اگلا_src1),
                           smctx->src1_صف_سائز_aligned, نقاب_صف_سائز, نقاب_صف_سائز, 1);
        }
    }

    dma_قطار_flush(dma_q);
}

ساکن صحیح execute_op_softmax_f32(ساخت htp_ops_context * octx) {
    ثابت ساخت htp_tensor * src0 = octx->src[0];
    ثابت ساخت htp_tensor * dst  = octx->dst;

    ثابت حرف * op_type = "softmax-f32";

    ثابت ساخت htp_softmax_kernel_params * kparams =
        (ثابت ساخت htp_softmax_kernel_params *) octx->kernel_params;

    اگر (!htp_ops_context_سیٹ_n_دھاگے(octx, kparams->n_دھاگے)) {
        واپس HTP_حالت_INVAL_PARAMS;
    }

    اگر (kparams->kernel_id >= HTP_SOFTMAX_KERNEL_گنتی) {
        واپس HTP_حالت_INVAL_PARAMS;
    }

    اگر (octx->ctx->vtcm_سائز < (size_t) kparams->vtcm_سائز) {
        FARF(ERROR, "%s : current VTCM reservation %zu is too small, needed %u\n",
             op_type, octx->ctx->vtcm_سائز, kparams->vtcm_سائز);
        واپس HTP_حالت_VTCM_TOO_SMALL;
    }

    ثابت uint32_t src0_nrows = src0->ne[1] * src0->ne[2] * src0->ne[3];
    ثابت size_t elem_سائز    = حجم(اعشاری);
    ثابت size_t dst_صف_سائز = dst->nb[1];

    uint32_t صف_آغاز = 0;
    uint32_t nrows     = src0_nrows;

    اگر (octx->ctx->mdev.گنتی > 1) {
        uint32_t صفیں_فی_chunk = 0;
        htp_tensor_mdev_صفیں_فی_chunk(dst, (uint32_t) elem_سائز, (uint32_t) dst_صف_سائز, &صفیں_فی_chunk);
        ثابت ساخت htp_tensor_mdev_range range = htp_tensor_mdev_تقسیم(
            src0_nrows, صفیں_فی_chunk, octx->ctx->mdev.idx, octx->ctx->mdev.گنتی, &octx->ctx->mdev.گنتی_div);
        صف_آغاز = range.آغاز;
        nrows     = range.گنتی;
        اگر (nrows < octx->n_دھاگے) {
            htp_ops_context_سیٹ_n_دھاگے(octx, nrows ? nrows : 1);
        }
    }

    اگر (nrows == 0) {
        واپس HTP_حالت_OK;
    }

    ثابت uint32_t n_دھاگے = octx->n_دھاگے;
    uint8_t * ثابت vtcm_بنیاد = (uint8_t *) octx->ctx->vtcm_بنیاد;

    ثابت uint32_t off_src0 = 0;
    ثابت uint32_t off_dst  = off_src0 + kparams->vtcm_src0_سائز_فی_thread * kparams->n_دھاگے;
    ثابت uint32_t off_src1 = off_dst  + kparams->vtcm_dst_سائز_فی_thread  * kparams->n_دھاگے;

    ساخت htp_softmax_context smctx = {
        .octx                  = octx,
        .kparams               = kparams,
        .compute               = (خالی *) softmax_kernels[kparams->kernel_id],

        .ڈیٹا_src0             = src0->ڈیٹا,
        .ڈیٹا_src1             = kparams->use_src1 ? octx->src[1]->ڈیٹا : 0,
        .ڈیٹا_dst              = dst->ڈیٹا,

        .vtcm_src0             = VTCM_LAYOUT_PTR(uint8_t, vtcm_بنیاد, off_src0),
        .vtcm_dst              = VTCM_LAYOUT_PTR(uint8_t, vtcm_بنیاد, off_dst),
        .vtcm_src1             = VTCM_LAYOUT_PTR_OPTIONAL(uint8_t, vtcm_بنیاد, off_src1, kparams->use_src1),

        .vtcm_src0_سائز_فی_thread = kparams->vtcm_src0_سائز_فی_thread,
        .vtcm_src1_سائز_فی_thread = kparams->vtcm_src1_سائز_فی_thread,
        .vtcm_dst_سائز_فی_thread  = kparams->vtcm_dst_سائز_فی_thread,

        .src0_spad_half_سائز   = kparams->src0_spad_half_سائز,
        .src1_spad_half_سائز   = kparams->src1_spad_half_سائز,
        .dst_spad_half_سائز    = kparams->dst_spad_half_سائز,

        .src0_صف_سائز_aligned = kparams->src0_صف_سائز_aligned,
        .src1_صف_سائز_aligned = kparams->src1_صف_سائز_aligned,
        .dst_صف_سائز_aligned  = kparams->dst_صف_سائز_aligned,

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

        .src0_nrows_فی_thread = fastdiv(nrows + n_دھاگے - 1, &octx->n_دھاگے_div),
        .صف_آغاز             = صف_آغاز,
        .nrows                 = nrows,
    };

    اگر (kparams->max_bias > 0.0f && kparams->use_src1) {
        اگر (kparams->n_سر > 512) {
            واپس HTP_حالت_INVAL_PARAMS;
        }
        برائے (uint32_t h = 0; h < kparams->n_سر; h += 32) {
            HVX_سمتیہ v_slopes = hvx_alibi_slopes(h, 1, kparams->n_سر_log2, kparams->m0, kparams->m1);
            hvx_vmem(&smctx.slopes[h]) = v_slopes;
        }
    }

    work_قطار_func_t task_func = kparams->use_src1 ? softmax_thread_نقاب_dma : softmax_thread_dma;
    work_قطار_تسلسل(octx->ctx->work_قطار, task_func, &smctx, n_دھاگے);

    واپس HTP_حالت_OK;
}

صحیح op_softmax(ساخت htp_ops_context * octx) {
    صحیح err = HTP_حالت_OK;

    انتخاب (octx->src[0]->type) {
        حالت HTP_TYPE_F32:
            err = execute_op_softmax_f32(octx);
            توڑو;

        طےشدہ:
            err = HTP_حالت_NO_SUPPORT;
            توڑو;
    }

    واپس err;
}
