#సూచన clang diagnostic ignored "-Wunused-variable"
#సూచన clang diagnostic ignored "-Wunused-function"
#సూచన clang diagnostic ignored "-Wunused-but-set-variable"

#అన్తర్భావ <HAP_farf.h>
#అన్తర్భావ <HAP_perf.h>

#అన్తర్భావ <math.h>
#అన్తర్భావ <string.h>

#అన్తర్భావ "dma-queue.h"
#అన్తర్భావ "work-queue.h"
#అన్తర్భావ "hvx-utils.h"
#అన్తర్భావ "hex-fastdiv.h"
#అన్తర్భావ "hex-common.h"
#అన్తర్భావ "hex-profile.h"

#నిర్వచించు GGML_COMMON_DECL_C
#అన్తర్భావ "ggml-common.h"
#అన్తర్భావ "htp-ctx.h"
#అన్తర్భావ "htp-ops.h"
#అన్తర్భావ "htp-tensor.h"
#అన్తర్భావ "htp-vtcm.h"
#అన్తర్భావ "htp/softmax-ops.h"
#అన్తర్భావ "hvx-flash-attn.h"

నిర్మాణం htp_softmax_context {
    నిర్మాణం htp_ops_context * octx;
    స్థిరరాశి నిర్మాణం htp_softmax_kernel_params * kparams;

    శూన్య *   compute;

    dma_addr_t దత్తాంశం_src0;
    dma_addr_t దత్తాంశం_src1;
    dma_addr_t దత్తాంశం_dst;

    uint8_t * vtcm_src0;
    uint8_t * vtcm_src1;
    uint8_t * vtcm_dst;

    uint32_t  vtcm_src0_కొలత_చొప్పున_thread;
    uint32_t  vtcm_src1_కొలత_చొప్పున_thread;
    uint32_t  vtcm_dst_కొలత_చొప్పున_thread;

    uint32_t  src0_spad_half_కొలత;
    uint32_t  src1_spad_half_కొలత;
    uint32_t  dst_spad_half_కొలత;

    uint32_t  src0_అడ్డువరుస_కొలత_aligned;
    uint32_t  src1_అడ్డువరుస_కొలత_aligned;
    uint32_t  dst_అడ్డువరుస_కొలత_aligned;

    bool     use_f16;

    uint32_t n_తల;
    uint32_t n_తల_log2;

    దశాంశం    scale;
    దశాంశం    max_bias;
    దశాంశం    m0;
    దశాంశం    m1;

    నిర్మాణం fastdiv_విలువలు div_ne01;
    నిర్మాణం fastdiv_విలువలు div_ne02;
    నిర్మాణం fastdiv_విలువలు div_ne12;
    నిర్మాణం fastdiv_విలువలు div_ne13;

    uint32_t src0_nrows_చొప్పున_thread;
    uint32_t అడ్డువరుస_ప్రారంభం;
    uint32_t nrows;

    దశాంశం    slopes[512] __attribute__((aligned(128)));
};

రకనిర్వచనం శూన్య (*softmax_compute_fn_t)(
    శూన్య * పరిమిత dst,
    స్థిరరాశి శూన్య * పరిమిత src0,
    స్థిరరాశి శూన్య * పరిమిత ముసుగు,
    uint32_t ne00,
    దశాంశం scale,
    దశాంశం వాలు
);

స్థిర శూన్య hvx_fast_softmax_prep_f16(స్థిరరాశి uint8_t * పరిమిత src,
                                      uint8_t * పరిమిత dst,
                                      స్థిరరాశి పూర్ణాంకం అంకె_elems,
                                      దశాంశం     scale,
                                      స్థిరరాశి uint8_t * పరిమిత ముసుగు,
                                      దశాంశం వాలు) {
    స్థిరరాశి HVX_సదిశ * పరిమిత v_src  = (స్థిరరాశి HVX_సదిశ *) src;
    HVX_సదిశ * పరిమిత v_dst        = (HVX_సదిశ *) dst;
    స్థిరరాశి HVX_సదిశ * పరిమిత v_ముసుగు = (స్థిరరాశి HVX_సదిశ *) ముసుగు;

    HVX_సదిశ scale_vec = hvx_vec_splat_f32(scale);
    HVX_సదిశ వాలు_vec = hvx_vec_splat_f32(వాలు);

    స్థిరరాశి పూర్ణాంకం nvec_64 = అంకె_elems / VLEN_FP16;
    స్థిరరాశి పూర్ణాంకం nloe_64 = అంకె_elems % VLEN_FP16;

    #సూచన unroll(2)
    కొరకు (పూర్ణాంకం i = 0; i < nvec_64; i++) {
        HVX_సదిశ_Pair p = hvx_vec_f16_గమ్యం_f32(v_ముసుగు[i]);
        HVX_సదిశ m0 = Q6_V_lo_W(p);
        HVX_సదిశ m1 = Q6_V_hi_W(p);

        HVX_సదిశ s0 = v_src[2 * i];
        HVX_సదిశ s1 = v_src[2 * i + 1];

        HVX_సదిశ v0 = Q6_Vqf32_vadd_Vqf32Vqf32(Q6_Vqf32_vmpy_VsfVsf(s0, scale_vec), Q6_Vqf32_vmpy_VsfVsf(m0, వాలు_vec));
        HVX_సదిశ v1 = Q6_Vqf32_vadd_Vqf32Vqf32(Q6_Vqf32_vmpy_VsfVsf(s1, scale_vec), Q6_Vqf32_vmpy_VsfVsf(m1, వాలు_vec));

        v_dst[2 * i]     = Q6_Vsf_equals_Vqf32(v0);
        v_dst[2 * i + 1] = Q6_Vsf_equals_Vqf32(v1);
    }

    ఒకవేళ (nloe_64 > 0) {
        HVX_సదిశ_Pair p = hvx_vec_f16_గమ్యం_f32(v_ముసుగు[nvec_64]);
        HVX_సదిశ m0 = Q6_V_lo_W(p);

        HVX_సదిశ s0 = v_src[2 * nvec_64];
        HVX_సదిశ v0 = Q6_Vqf32_vadd_Vqf32Vqf32(Q6_Vqf32_vmpy_VsfVsf(s0, scale_vec), Q6_Vqf32_vmpy_VsfVsf(m0, వాలు_vec));

        ఒకవేళ (nloe_64 <= VLEN_FP32) {
            hvx_vec_store_a(&v_dst[2 * nvec_64], nloe_64 * పరిమాణం(దశాంశం), Q6_Vsf_equals_Vqf32(v0));
        } లేకపోతే {
            v_dst[2 * nvec_64] = Q6_Vsf_equals_Vqf32(v0);

            HVX_సదిశ m1 = Q6_V_hi_W(p);
            HVX_సదిశ s1 = v_src[2 * nvec_64 + 1];
            HVX_సదిశ v1 = Q6_Vqf32_vadd_Vqf32Vqf32(Q6_Vqf32_vmpy_VsfVsf(s1, scale_vec), Q6_Vqf32_vmpy_VsfVsf(m1, వాలు_vec));

            hvx_vec_store_a(&v_dst[2 * nvec_64 + 1], (nloe_64 - VLEN_FP32) * పరిమాణం(దశాంశం), Q6_Vsf_equals_Vqf32(v1));
        }
    }
}

స్థిర శూన్య hvx_fast_softmax_prep_f32(స్థిరరాశి uint8_t * పరిమిత src,
                                      uint8_t * పరిమిత dst,
                                      స్థిరరాశి పూర్ణాంకం అంకె_elems,
                                      దశాంశం     scale,
                                      స్థిరరాశి uint8_t * పరిమిత ముసుగు,
                                      దశాంశం వాలు) {
    స్థిరరాశి HVX_సదిశ * పరిమిత v_src  = (స్థిరరాశి HVX_సదిశ *) src;
    HVX_సదిశ * పరిమిత v_dst        = (HVX_సదిశ *) dst;
    స్థిరరాశి HVX_సదిశ * పరిమిత v_ముసుగు = (స్థిరరాశి HVX_సదిశ *) ముసుగు;

    HVX_సదిశ scale_vec = hvx_vec_splat_f32(scale);
    HVX_సదిశ వాలు_vec = hvx_vec_splat_f32(వాలు);

    స్థిరరాశి పూర్ణాంకం nvec = అంకె_elems / VLEN_FP32;
    స్థిరరాశి పూర్ణాంకం nloe = అంకె_elems % VLEN_FP32;

    #సూచన unroll(4)
    కొరకు (పూర్ణాంకం i = 0; i < nvec; i++) {
        HVX_సదిశ v1 = v_src[i];
        HVX_సదిశ v3 = v_ముసుగు[i];

        HVX_సదిశ v2 = Q6_Vqf32_vmpy_VsfVsf(v1, scale_vec);
        HVX_సదిశ v4 = Q6_Vqf32_vmpy_VsfVsf(v3, వాలు_vec);
        HVX_సదిశ v5 = Q6_Vqf32_vadd_Vqf32Vqf32(v2, v4);

        v_dst[i] = Q6_Vsf_equals_Vqf32(v5);
    }

    ఒకవేళ (nloe > 0) {
        HVX_సదిశ v1 = v_src[nvec];
        HVX_సదిశ v3 = v_ముసుగు[nvec];

        HVX_సదిశ v2 = Q6_Vqf32_vmpy_VsfVsf(v1, scale_vec);
        HVX_సదిశ v4 = Q6_Vqf32_vmpy_VsfVsf(v3, వాలు_vec);
        HVX_సదిశ v5 = Q6_Vqf32_vadd_Vqf32Vqf32(v2, v4);

        hvx_vec_store_a(&v_dst[nvec], nloe * పరిమాణం(దశాంశం), Q6_Vsf_equals_Vqf32(v5));
    }
}

స్థిర శూన్య hvx_fast_softmax_f32(స్థిరరాశి uint8_t * పరిమిత src, uint8_t * పరిమిత dst, స్థిరరాశి పూర్ణాంకం అంకె_elems) {
    స్థిరరాశి HVX_సదిశ * పరిమిత v_src = (స్థిరరాశి HVX_సదిశ *) src;
    HVX_సదిశ * పరిమిత v_dst       = (HVX_సదిశ *) dst;

    స్థిరరాశి పూర్ణాంకం nvec = అంకె_elems / VLEN_FP32;
    స్థిరరాశి పూర్ణాంకం nloe = అంకె_elems % VLEN_FP32;

    HVX_సదిశ max_vec = hvx_vec_splat_f32(((స్థిరరాశి దశాంశం *) src)[0]);

    #సూచన unroll(2)
    కొరకు (పూర్ణాంకం i = 0; i < nvec; i++) {
        HVX_సదిశ v1 = v_src[i];
        max_vec       = Q6_Vsf_vmax_VsfVsf(max_vec, v1);
    }

    ఒకవేళ (nloe > 0) {
        HVX_సదిశ_Pred q_ముసుగు = Q6_Q_vsetq_R(nloe * పరిమాణం(దశాంశం));
        HVX_సదిశ neg_inf    = hvx_vec_splat_f32(-INFINITY);
        HVX_సదిశ v_తోక     = Q6_V_vmux_QVV(q_ముసుగు, v_src[nvec], neg_inf);
        max_vec               = Q6_Vsf_vmax_VsfVsf(max_vec, v_తోక);
    }

    max_vec = hvx_vec_reduce_max_f32(max_vec);

    HVX_సదిశ కూడిక_vec = Q6_V_vsplat_R(0x00000000);

    #సూచన unroll(2)
    కొరకు (పూర్ణాంకం i = 0; i < nvec; i++) {
        HVX_సదిశ v1 = v_src[i];
        HVX_సదిశ v2 = Q6_Vqf32_vsub_VsfVsf(v1, max_vec);

        HVX_సదిశ v3 = hvx_vec_exp_f32(Q6_Vsf_equals_Vqf32(v2));

        కూడిక_vec = Q6_Vqf32_vadd_VsfVsf(Q6_Vsf_equals_Vqf32(కూడిక_vec), v3);

        v_dst[i] = v3;
    }

    ఒకవేళ (nloe > 0) {
        HVX_సదిశ_Pred q_ముసుగు = Q6_Q_vsetq_R(nloe * పరిమాణం(దశాంశం));
        HVX_సదిశ v1     = v_src[nvec];
        HVX_సదిశ v2     = Q6_Vqf32_vsub_VsfVsf(v1, max_vec);
        HVX_సదిశ v3     = hvx_vec_exp_f32(Q6_Vsf_equals_Vqf32(v2));
        HVX_సదిశ v3_pad = Q6_V_vmux_QVV(q_ముసుగు, v3, Q6_V_vzero());

        కూడిక_vec     = Q6_Vqf32_vadd_VsfVsf(Q6_Vsf_equals_Vqf32(కూడిక_vec), v3_pad);
        v_dst[nvec] = v3_pad;
    }

    కూడిక_vec = hvx_vec_reduce_కూడిక_f32(Q6_Vsf_equals_Vqf32(కూడిక_vec));

    HVX_సదిశ_Pred pos_కూడిక   = Q6_Q_vcmp_gt_VwVw(కూడిక_vec, Q6_V_vzero());
    HVX_సదిశ     v4        = hvx_vec_inverse_f32(కూడిక_vec);
    HVX_సదిశ     scale_vec = Q6_V_vmux_QVV(pos_కూడిక, v4, hvx_vec_splat_f32(1.0f));

    #సూచన unroll(2)
    కొరకు (పూర్ణాంకం i = 0; i < nvec; i++) {
        HVX_సదిశ v1 = v_dst[i];
        HVX_సదిశ v2 = Q6_Vqf32_vmpy_VsfVsf(v1, scale_vec);
        v_dst[i]      = Q6_Vsf_equals_Vqf32(v2);
    }

    ఒకవేళ (nloe > 0) {
        HVX_సదిశ v1 = v_dst[nvec];
        HVX_సదిశ v2 = Q6_Vqf32_vmpy_VsfVsf(v1, scale_vec);
        hvx_vec_store_a(&v_dst[nvec], nloe * పరిమాణం(దశాంశం), Q6_Vsf_equals_Vqf32(v2));
    }
}

స్థిర శూన్య compute_fast_softmax_f32_nomask(
    శూన్య * పరిమిత dst,
    స్థిరరాశి శూన్య * పరిమిత src0,
    స్థిరరాశి శూన్య * పరిమిత ముసుగు,
    uint32_t ne00,
    దశాంశం scale,
    దశాంశం వాలు
) {
    (శూన్య) ముసుగు;
    (శూన్య) వాలు;
    hvx_scale_f32((uint8_t *) dst, (స్థిరరాశి uint8_t *) src0, ne00, scale);
    hvx_fast_softmax_f32((స్థిరరాశి uint8_t *) dst, (uint8_t *) dst, ne00);
}

స్థిర శూన్య compute_fast_softmax_f32_ముసుగు_f32(
    శూన్య * పరిమిత dst,
    స్థిరరాశి శూన్య * పరిమిత src0,
    స్థిరరాశి శూన్య * పరిమిత ముసుగు,
    uint32_t ne00,
    దశాంశం scale,
    దశాంశం వాలు
) {
    hvx_fast_softmax_prep_f32((స్థిరరాశి uint8_t *) src0, (uint8_t *) dst, ne00, scale, (స్థిరరాశి uint8_t *) ముసుగు, వాలు);
    hvx_fast_softmax_f32((స్థిరరాశి uint8_t *) dst, (uint8_t *) dst, ne00);
}

స్థిర శూన్య compute_fast_softmax_f32_ముసుగు_f16(
    శూన్య * పరిమిత dst,
    స్థిరరాశి శూన్య * పరిమిత src0,
    స్థిరరాశి శూన్య * పరిమిత ముసుగు,
    uint32_t ne00,
    దశాంశం scale,
    దశాంశం వాలు
) {
    hvx_fast_softmax_prep_f16((స్థిరరాశి uint8_t *) src0, (uint8_t *) dst, ne00, scale, (స్థిరరాశి uint8_t *) ముసుగు, వాలు);
    hvx_fast_softmax_f32((స్థిరరాశి uint8_t *) dst, (uint8_t *) dst, ne00);
}

స్థిర స్థిరరాశి softmax_compute_fn_t softmax_kernels[HTP_SOFTMAX_KERNEL_లెక్క] = {
    [HTP_SOFTMAX_KERNEL_NOMASK]   = compute_fast_softmax_f32_nomask,
    [HTP_SOFTMAX_KERNEL_ముసుగు_F32] = compute_fast_softmax_f32_ముసుగు_f32,
    [HTP_SOFTMAX_KERNEL_ముసుగు_F16] = compute_fast_softmax_f32_ముసుగు_f16,
};

స్థిర శూన్య softmax_thread_dma(గుర్తులేని పూర్ణాంకం nth, గుర్తులేని పూర్ణాంకం ith, శూన్య * దత్తాంశం) {
    (శూన్య) nth;
    స్థిరరాశి నిర్మాణం htp_softmax_context * smctx = (స్థిరరాశి నిర్మాణం htp_softmax_context *) దత్తాంశం;
    నిర్మాణం htp_ops_context * octx = smctx->octx;
    స్థిరరాశి నిర్మాణం htp_tensor * src0 = octx->src[0];
    స్థిరరాశి నిర్మాణం htp_tensor * dst  = octx->dst;
    నిర్మాణం htp_thread_trace * tr = &octx->ctx->trace[ith];

    స్థిరరాశి uint32_t src0_nrows            = smctx->nrows;
    స్థిరరాశి uint32_t src0_nrows_చొప్పున_thread = smctx->src0_nrows_చొప్పున_thread;

    స్థిరరాశి uint32_t src0_ప్రారంభం_అడ్డువరుస = smctx->అడ్డువరుస_ప్రారంభం + src0_nrows_చొప్పున_thread * ith;
    స్థిరరాశి uint32_t src0_చివర_అడ్డువరుస   = MIN(src0_ప్రారంభం_అడ్డువరుస + src0_nrows_చొప్పున_thread, smctx->అడ్డువరుస_ప్రారంభం + src0_nrows);

    ఒకవేళ (src0_ప్రారంభం_అడ్డువరుస >= src0_చివర_అడ్డువరుస) {
        తిరిగి;
    }

    స్థిరరాశి dma_addr_t దత్తాంశం_src0 = smctx->దత్తాంశం_src0;
    స్థిరరాశి dma_addr_t దత్తాంశం_dst  = smctx->దత్తాంశం_dst;

    స్థిరరాశి size_t src0_అడ్డువరుస_కొలత = src0->ne[0] * పరిమాణం(దశాంశం);
    స్థిరరాశి size_t dst_అడ్డువరుస_కొలత  = src0->ne[0] * పరిమాణం(దశాంశం);

    uint8_t * src0_vtcm_పునాది = smctx->vtcm_src0 + (ith * smctx->vtcm_src0_కొలత_చొప్పున_thread);
    uint8_t * dst_vtcm_పునాది  = smctx->vtcm_dst  + (ith * smctx->vtcm_dst_కొలత_చొప్పున_thread);

    స్థిరరాశి size_t src0_vtcm_half = smctx->src0_spad_half_కొలత;
    స్థిరరాశి size_t dst_vtcm_half  = smctx->dst_spad_half_కొలత;

    dma_వరుస * dma_q = octx->ctx->dma[ith];

    కొరకు (uint32_t r = src0_ప్రారంభం_అడ్డువరుస, idx = 0; r < src0_చివర_అడ్డువరుస && idx < 2; r++, idx++) {
        dma_addr_t cur_dst  = దత్తాంశం_dst  + r * dst_అడ్డువరుస_కొలత;
        dma_addr_t cur_src0 = దత్తాంశం_src0 + r * src0_అడ్డువరుస_కొలత;
        శూన్య * d_spad = dst_vtcm_పునాది  + idx * dst_vtcm_half;
        శూన్య * s_spad = src0_vtcm_పునాది + idx * src0_vtcm_half;

        dma_వరుస_నెట్టు(dma_q, dma_make_దత్తాంశం(cur_dst, d_spad),
                       dst_అడ్డువరుస_కొలత, smctx->dst_అడ్డువరుస_కొలత_aligned, dst_అడ్డువరుస_కొలత, 0);
        dma_వరుస_నెట్టు(dma_q, dma_make_దత్తాంశం(s_spad, cur_src0),
                       smctx->src0_అడ్డువరుస_కొలత_aligned, src0_అడ్డువరుస_కొలత, src0_అడ్డువరుస_కొలత, 1);
    }

    softmax_compute_fn_t compute = (softmax_compute_fn_t) smctx->compute;
    స్థిరరాశి uint32_t ne00 = src0->ne[0];

    కొరకు (uint32_t r = src0_ప్రారంభం_అడ్డువరుస; r < src0_చివర_అడ్డువరుస; ++r) {
        శూన్య * d_spad = (శూన్య *) dma_వరుస_తీయి(dma_q).src;
        శూన్య * s_spad = (శూన్య *) dma_వరుస_తీయి(dma_q).dst;

        htp_trace_event_ప్రారంభం(tr, HTP_TRACE_EVT_HVX_COMP, r);
        compute(d_spad, s_spad, ఖాళీ, ne00, smctx->scale, 1.0f);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, r);

        dma_addr_t cur_dst = దత్తాంశం_dst + r * dst_అడ్డువరుస_కొలత;
        dma_వరుస_నెట్టు(dma_q, dma_make_దత్తాంశం(cur_dst, d_spad),
                       dst_అడ్డువరుస_కొలత, smctx->dst_అడ్డువరుస_కొలత_aligned, dst_అడ్డువరుస_కొలత, 1);

        స్థిరరాశి uint32_t తదుపరి_r = r + 2;
        ఒకవేళ (తదుపరి_r < src0_చివర_అడ్డువరుస) {
            dma_addr_t తదుపరి_src0 = దత్తాంశం_src0 + తదుపరి_r * src0_అడ్డువరుస_కొలత;
            dma_వరుస_నెట్టు(dma_q, dma_make_దత్తాంశం(s_spad, తదుపరి_src0),
                           smctx->src0_అడ్డువరుస_కొలత_aligned, src0_అడ్డువరుస_కొలత, src0_అడ్డువరుస_కొలత, 1);
        }
    }

    dma_వరుస_flush(dma_q);
}

స్థిర శూన్య softmax_thread_ముసుగు_dma(గుర్తులేని పూర్ణాంకం nth, గుర్తులేని పూర్ణాంకం ith, శూన్య * దత్తాంశం) {
    (శూన్య) nth;
    స్థిరరాశి నిర్మాణం htp_softmax_context * smctx = (స్థిరరాశి నిర్మాణం htp_softmax_context *) దత్తాంశం;
    నిర్మాణం htp_ops_context * octx = smctx->octx;
    స్థిరరాశి నిర్మాణం htp_tensor * src0 = octx->src[0];
    స్థిరరాశి నిర్మాణం htp_tensor * src1 = octx->src[1];
    స్థిరరాశి నిర్మాణం htp_tensor * dst  = octx->dst;
    నిర్మాణం htp_thread_trace * tr = &octx->ctx->trace[ith];

    స్థిరరాశి uint32_t src0_nrows            = smctx->nrows;
    స్థిరరాశి uint32_t src0_nrows_చొప్పున_thread = smctx->src0_nrows_చొప్పున_thread;

    స్థిరరాశి uint32_t src0_ప్రారంభం_అడ్డువరుస = smctx->అడ్డువరుస_ప్రారంభం + src0_nrows_చొప్పున_thread * ith;
    స్థిరరాశి uint32_t src0_చివర_అడ్డువరుస   = MIN(src0_ప్రారంభం_అడ్డువరుస + src0_nrows_చొప్పున_thread, smctx->అడ్డువరుస_ప్రారంభం + src0_nrows);

    ఒకవేళ (src0_ప్రారంభం_అడ్డువరుస >= src0_చివర_అడ్డువరుస) {
        తిరిగి;
    }

    స్థిరరాశి dma_addr_t దత్తాంశం_src0 = smctx->దత్తాంశం_src0;
    స్థిరరాశి dma_addr_t దత్తాంశం_src1 = smctx->దత్తాంశం_src1;
    స్థిరరాశి dma_addr_t దత్తాంశం_dst  = smctx->దత్తాంశం_dst;

    స్థిరరాశి size_t src0_అడ్డువరుస_కొలత = src0->ne[0] * పరిమాణం(దశాంశం);
    స్థిరరాశి size_t dst_అడ్డువరుస_కొలత  = src0->ne[0] * పరిమాణం(దశాంశం);
    స్థిరరాశి size_t ముసుగు_అడ్డువరుస_కొలత = smctx->use_f16 ? (src1->ne[0] * పరిమాణం(__fp16)) : (src1->ne[0] * పరిమాణం(దశాంశం));

    uint8_t * src0_vtcm_పునాది = smctx->vtcm_src0 + (ith * smctx->vtcm_src0_కొలత_చొప్పున_thread);
    uint8_t * src1_vtcm_పునాది = smctx->vtcm_src1 + (ith * smctx->vtcm_src1_కొలత_చొప్పున_thread);
    uint8_t * dst_vtcm_పునాది  = smctx->vtcm_dst  + (ith * smctx->vtcm_dst_కొలత_చొప్పున_thread);

    స్థిరరాశి size_t src0_vtcm_half = smctx->src0_spad_half_కొలత;
    స్థిరరాశి size_t src1_vtcm_half = smctx->src1_spad_half_కొలత;
    స్థిరరాశి size_t dst_vtcm_half  = smctx->dst_spad_half_కొలత;

    స్థిరరాశి uint32_t nb11 = src1->nb[1];
    స్థిరరాశి uint32_t nb12 = src1->nb[2];
    స్థిరరాశి uint32_t nb13 = src1->nb[3];

    స్థిరరాశి uint32_t ne00 = src0->ne[0];
    స్థిరరాశి uint32_t ne01 = src0->ne[1];
    స్థిరరాశి uint32_t ne02 = src0->ne[2];
    స్థిరరాశి uint32_t ne03 = src0->ne[3];
    స్థిరరాశి uint32_t ne12 = src1->ne[2];
    స్థిరరాశి uint32_t ne13 = src1->ne[3];

    స్థిరరాశి నిర్మాణం fastdiv_విలువలు * div_ne01 = &smctx->div_ne01;
    స్థిరరాశి నిర్మాణం fastdiv_విలువలు * div_ne02 = &smctx->div_ne02;
    స్థిరరాశి నిర్మాణం fastdiv_విలువలు * div_ne12 = &smctx->div_ne12;
    స్థిరరాశి నిర్మాణం fastdiv_విలువలు * div_ne13 = &smctx->div_ne13;

    dma_వరుస * dma_q = octx->ctx->dma[ith];

    కొరకు (uint32_t r = src0_ప్రారంభం_అడ్డువరుస, idx = 0; r < src0_చివర_అడ్డువరుస && idx < 2; r++, idx++) {
        dma_addr_t cur_dst  = దత్తాంశం_dst  + r * dst_అడ్డువరుస_కొలత;
        dma_addr_t cur_src0 = దత్తాంశం_src0 + r * src0_అడ్డువరుస_కొలత;

        uint32_t i1 = fastmodulo(r, ne01, div_ne01);
        uint32_t r_div_ne01 = fastdiv(r, div_ne01);
        uint32_t i2 = fastmodulo(r_div_ne01, ne02, div_ne02);
        uint32_t i3 = fastdiv(r_div_ne01, div_ne02);
        uint32_t i12 = (ne12 == ne02) ? i2 : fastmodulo(i2, ne12, div_ne12);
        uint32_t i13 = (ne13 == ne03) ? i3 : fastmodulo(i3, ne13, div_ne13);
        dma_addr_t cur_src1 = దత్తాంశం_src1 + i1 * nb11 + i12 * nb12 + i13 * nb13;

        శూన్య * d_spad = dst_vtcm_పునాది  + idx * dst_vtcm_half;
        శూన్య * s_spad = src0_vtcm_పునాది + idx * src0_vtcm_half;
        శూన్య * m_spad = src1_vtcm_పునాది + idx * src1_vtcm_half;

        dma_వరుస_నెట్టు(dma_q, dma_make_దత్తాంశం(cur_dst, d_spad),
                       dst_అడ్డువరుస_కొలత, smctx->dst_అడ్డువరుస_కొలత_aligned, dst_అడ్డువరుస_కొలత, 0);
        dma_వరుస_నెట్టు(dma_q, dma_make_దత్తాంశం(s_spad, cur_src0),
                       smctx->src0_అడ్డువరుస_కొలత_aligned, src0_అడ్డువరుస_కొలత, src0_అడ్డువరుస_కొలత, 1);
        dma_వరుస_నెట్టు(dma_q, dma_make_దత్తాంశం(m_spad, cur_src1),
                       smctx->src1_అడ్డువరుస_కొలత_aligned, ముసుగు_అడ్డువరుస_కొలత, ముసుగు_అడ్డువరుస_కొలత, 1);
    }

    softmax_compute_fn_t compute = (softmax_compute_fn_t) smctx->compute;
    స్థిరరాశి bool has_bias = smctx->max_bias > 0.0f;
    uint32_t prev_i2 = (uint32_t)-1;
    దశాంశం వాలు = 1.0f;

    కొరకు (uint32_t r = src0_ప్రారంభం_అడ్డువరుస; r < src0_చివర_అడ్డువరుస; ++r) {
        శూన్య * d_spad = (శూన్య *) (uintptr_t) dma_వరుస_తీయి(dma_q).src;
        శూన్య * s_spad = (శూన్య *) (uintptr_t) dma_వరుస_తీయి(dma_q).dst;
        శూన్య * m_spad = (శూన్య *) (uintptr_t) dma_వరుస_తీయి(dma_q).dst;

        ఒకవేళ (has_bias) {
            uint32_t r_div_ne01 = fastdiv(r, div_ne01);
            uint32_t i2 = fastmodulo(r_div_ne01, ne02, div_ne02);
            ఒకవేళ (i2 != prev_i2) {
                వాలు = smctx->slopes[i2];
                prev_i2 = i2;
            }
        }

        htp_trace_event_ప్రారంభం(tr, HTP_TRACE_EVT_HVX_COMP, r);
        compute(d_spad, s_spad, m_spad, ne00, smctx->scale, వాలు);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, r);

        dma_addr_t cur_dst = దత్తాంశం_dst + r * dst_అడ్డువరుస_కొలత;
        dma_వరుస_నెట్టు(dma_q, dma_make_దత్తాంశం(cur_dst, d_spad),
                       dst_అడ్డువరుస_కొలత, smctx->dst_అడ్డువరుస_కొలత_aligned, dst_అడ్డువరుస_కొలత, 1);

        స్థిరరాశి uint32_t తదుపరి_r = r + 2;
        ఒకవేళ (తదుపరి_r < src0_చివర_అడ్డువరుస) {
            dma_addr_t తదుపరి_src0 = దత్తాంశం_src0 + తదుపరి_r * src0_అడ్డువరుస_కొలత;

            uint32_t ni1 = fastmodulo(తదుపరి_r, ne01, div_ne01);
            uint32_t nr_div_ne01 = fastdiv(తదుపరి_r, div_ne01);
            uint32_t ni2 = fastmodulo(nr_div_ne01, ne02, div_ne02);
            uint32_t ni3 = fastdiv(nr_div_ne01, div_ne02);
            uint32_t ni12 = (ne12 == ne02) ? ni2 : fastmodulo(ni2, ne12, div_ne12);
            uint32_t ni13 = (ne13 == ne03) ? ni3 : fastmodulo(ni3, ne13, div_ne13);
            dma_addr_t తదుపరి_src1 = దత్తాంశం_src1 + ni1 * nb11 + ni12 * nb12 + ni13 * nb13;

            dma_వరుస_నెట్టు(dma_q, dma_make_దత్తాంశం(s_spad, తదుపరి_src0),
                           smctx->src0_అడ్డువరుస_కొలత_aligned, src0_అడ్డువరుస_కొలత, src0_అడ్డువరుస_కొలత, 1);
            dma_వరుస_నెట్టు(dma_q, dma_make_దత్తాంశం(m_spad, తదుపరి_src1),
                           smctx->src1_అడ్డువరుస_కొలత_aligned, ముసుగు_అడ్డువరుస_కొలత, ముసుగు_అడ్డువరుస_కొలత, 1);
        }
    }

    dma_వరుస_flush(dma_q);
}

స్థిర పూర్ణాంకం execute_op_softmax_f32(నిర్మాణం htp_ops_context * octx) {
    స్థిరరాశి నిర్మాణం htp_tensor * src0 = octx->src[0];
    స్థిరరాశి నిర్మాణం htp_tensor * dst  = octx->dst;

    స్థిరరాశి అక్షరం * op_type = "softmax-f32";

    స్థిరరాశి నిర్మాణం htp_softmax_kernel_params * kparams =
        (స్థిరరాశి నిర్మాణం htp_softmax_kernel_params *) octx->kernel_params;

    ఒకవేళ (!htp_ops_context_సమితి_n_దారాలు(octx, kparams->n_దారాలు)) {
        తిరిగి HTP_స్థితి_INVAL_PARAMS;
    }

    ఒకవేళ (kparams->kernel_id >= HTP_SOFTMAX_KERNEL_లెక్క) {
        తిరిగి HTP_స్థితి_INVAL_PARAMS;
    }

    ఒకవేళ (octx->ctx->vtcm_కొలత < (size_t) kparams->vtcm_కొలత) {
        FARF(ERROR, "%s : current VTCM reservation %zu is too small, needed %u\n",
             op_type, octx->ctx->vtcm_కొలత, kparams->vtcm_కొలత);
        తిరిగి HTP_స్థితి_VTCM_TOO_SMALL;
    }

    స్థిరరాశి uint32_t src0_nrows = src0->ne[1] * src0->ne[2] * src0->ne[3];
    స్థిరరాశి size_t elem_కొలత    = పరిమాణం(దశాంశం);
    స్థిరరాశి size_t dst_అడ్డువరుస_కొలత = dst->nb[1];

    uint32_t అడ్డువరుస_ప్రారంభం = 0;
    uint32_t nrows     = src0_nrows;

    ఒకవేళ (octx->ctx->mdev.లెక్క > 1) {
        uint32_t అడ్డువరుసలు_చొప్పున_chunk = 0;
        htp_tensor_mdev_అడ్డువరుసలు_చొప్పున_chunk(dst, (uint32_t) elem_కొలత, (uint32_t) dst_అడ్డువరుస_కొలత, &అడ్డువరుసలు_చొప్పున_chunk);
        స్థిరరాశి నిర్మాణం htp_tensor_mdev_range range = htp_tensor_mdev_విభజన(
            src0_nrows, అడ్డువరుసలు_చొప్పున_chunk, octx->ctx->mdev.idx, octx->ctx->mdev.లెక్క, &octx->ctx->mdev.లెక్క_div);
        అడ్డువరుస_ప్రారంభం = range.ప్రారంభం;
        nrows     = range.లెక్క;
        ఒకవేళ (nrows < octx->n_దారాలు) {
            htp_ops_context_సమితి_n_దారాలు(octx, nrows ? nrows : 1);
        }
    }

    ఒకవేళ (nrows == 0) {
        తిరిగి HTP_స్థితి_OK;
    }

    స్థిరరాశి uint32_t n_దారాలు = octx->n_దారాలు;
    uint8_t * స్థిరరాశి vtcm_పునాది = (uint8_t *) octx->ctx->vtcm_పునాది;

    స్థిరరాశి uint32_t off_src0 = 0;
    స్థిరరాశి uint32_t off_dst  = off_src0 + kparams->vtcm_src0_కొలత_చొప్పున_thread * kparams->n_దారాలు;
    స్థిరరాశి uint32_t off_src1 = off_dst  + kparams->vtcm_dst_కొలత_చొప్పున_thread  * kparams->n_దారాలు;

    నిర్మాణం htp_softmax_context smctx = {
        .octx                  = octx,
        .kparams               = kparams,
        .compute               = (శూన్య *) softmax_kernels[kparams->kernel_id],

        .దత్తాంశం_src0             = src0->దత్తాంశం,
        .దత్తాంశం_src1             = kparams->use_src1 ? octx->src[1]->దత్తాంశం : 0,
        .దత్తాంశం_dst              = dst->దత్తాంశం,

        .vtcm_src0             = VTCM_LAYOUT_PTR(uint8_t, vtcm_పునాది, off_src0),
        .vtcm_dst              = VTCM_LAYOUT_PTR(uint8_t, vtcm_పునాది, off_dst),
        .vtcm_src1             = VTCM_LAYOUT_PTR_OPTIONAL(uint8_t, vtcm_పునాది, off_src1, kparams->use_src1),

        .vtcm_src0_కొలత_చొప్పున_thread = kparams->vtcm_src0_కొలత_చొప్పున_thread,
        .vtcm_src1_కొలత_చొప్పున_thread = kparams->vtcm_src1_కొలత_చొప్పున_thread,
        .vtcm_dst_కొలత_చొప్పున_thread  = kparams->vtcm_dst_కొలత_చొప్పున_thread,

        .src0_spad_half_కొలత   = kparams->src0_spad_half_కొలత,
        .src1_spad_half_కొలత   = kparams->src1_spad_half_కొలత,
        .dst_spad_half_కొలత    = kparams->dst_spad_half_కొలత,

        .src0_అడ్డువరుస_కొలత_aligned = kparams->src0_అడ్డువరుస_కొలత_aligned,
        .src1_అడ్డువరుస_కొలత_aligned = kparams->src1_అడ్డువరుస_కొలత_aligned,
        .dst_అడ్డువరుస_కొలత_aligned  = kparams->dst_అడ్డువరుస_కొలత_aligned,

        .use_f16               = kparams->use_f16 != 0,

        .n_తల                = kparams->n_తల,
        .n_తల_log2           = kparams->n_తల_log2,

        .scale                 = kparams->scale,
        .max_bias              = kparams->max_bias,
        .m0                    = kparams->m0,
        .m1                    = kparams->m1,

        .div_ne01              = kparams->div_ne01,
        .div_ne02              = kparams->div_ne02,
        .div_ne12              = kparams->div_ne12,
        .div_ne13              = kparams->div_ne13,

        .src0_nrows_చొప్పున_thread = fastdiv(nrows + n_దారాలు - 1, &octx->n_దారాలు_div),
        .అడ్డువరుస_ప్రారంభం             = అడ్డువరుస_ప్రారంభం,
        .nrows                 = nrows,
    };

    ఒకవేళ (kparams->max_bias > 0.0f && kparams->use_src1) {
        ఒకవేళ (kparams->n_తల > 512) {
            తిరిగి HTP_స్థితి_INVAL_PARAMS;
        }
        కొరకు (uint32_t h = 0; h < kparams->n_తల; h += 32) {
            HVX_సదిశ v_slopes = hvx_alibi_slopes(h, 1, kparams->n_తల_log2, kparams->m0, kparams->m1);
            hvx_vmem(&smctx.slopes[h]) = v_slopes;
        }
    }

    work_వరుస_func_t task_func = kparams->use_src1 ? softmax_thread_ముసుగు_dma : softmax_thread_dma;
    work_వరుస_నిరంతరం(octx->ctx->work_వరుస, task_func, &smctx, n_దారాలు);

    తిరిగి HTP_స్థితి_OK;
}

పూర్ణాంకం op_softmax(నిర్మాణం htp_ops_context * octx) {
    పూర్ణాంకం err = HTP_స్థితి_OK;

    ఎంపిక (octx->src[0]->type) {
        స్థితి HTP_TYPE_F32:
            err = execute_op_softmax_f32(octx);
            ఆపు;

        అప్రమేయ:
            err = HTP_స్థితి_NO_SUPPORT;
            ఆపు;
    }

    తిరిగి err;
}
