#निर्देशः clang diagnostic ignored "-Wunused-variable"
#निर्देशः clang diagnostic ignored "-Wunused-function"
#निर्देशः clang diagnostic ignored "-Wunused-but-set-variable"

#अन्तर्भाव <HAP_farf.h>
#अन्तर्भाव <HAP_perf.h>

#अन्तर्भाव <math.h>
#अन्तर्भाव <string.h>

#अन्तर्भाव "dma-queue.h"
#अन्तर्भाव "work-queue.h"
#अन्तर्भाव "hvx-utils.h"
#अन्तर्भाव "hex-fastdiv.h"
#अन्तर्भाव "hex-common.h"
#अन्तर्भाव "hex-profile.h"

#परिभाषय GGML_COMMON_DECL_C
#अन्तर्भाव "ggml-common.h"
#अन्तर्भाव "htp-ctx.h"
#अन्तर्भाव "htp-ops.h"
#अन्तर्भाव "htp-tensor.h"
#अन्तर्भाव "htp-vtcm.h"
#अन्तर्भाव "htp/softmax-ops.h"
#अन्तर्भाव "hvx-flash-attn.h"

संरचना htp_softmax_context {
    संरचना htp_ops_context * octx;
    ध्रुव संरचना htp_softmax_kernel_params * kparams;

    शून्य *   compute;

    dma_addr_t दत्त_src0;
    dma_addr_t दत्त_src1;
    dma_addr_t दत्त_dst;

    uint8_t * vtcm_src0;
    uint8_t * vtcm_src1;
    uint8_t * vtcm_dst;

    uint32_t  vtcm_src0_आकार_प्रत्येकम्_thread;
    uint32_t  vtcm_src1_आकार_प्रत्येकम्_thread;
    uint32_t  vtcm_dst_आकार_प्रत्येकम्_thread;

    uint32_t  src0_spad_half_आकार;
    uint32_t  src1_spad_half_आकार;
    uint32_t  dst_spad_half_आकार;

    uint32_t  src0_row_आकार_aligned;
    uint32_t  src1_row_आकार_aligned;
    uint32_t  dst_row_आकार_aligned;

    bool     use_f16;

    uint32_t n_शीर्ष;
    uint32_t n_शीर्ष_log2;

    भिन्न    scale;
    भिन्न    max_bias;
    भिन्न    m0;
    भिन्न    m1;

    संरचना fastdiv_मानानि div_ne01;
    संरचना fastdiv_मानानि div_ne02;
    संरचना fastdiv_मानानि div_ne12;
    संरचना fastdiv_मानानि div_ne13;

    uint32_t src0_nrows_प्रत्येकम्_thread;
    uint32_t row_start;
    uint32_t nrows;

    भिन्न    slopes[512] __attribute__((aligned(128)));
};

प्रकारनिर्देश शून्य (*softmax_compute_fn_t)(
    शून्य * सीमित dst,
    ध्रुव शून्य * सीमित src0,
    ध्रुव शून्य * सीमित आवरणम्,
    uint32_t ne00,
    भिन्न scale,
    भिन्न नतिः
);

स्थिर शून्य hvx_fast_softmax_prep_f16(ध्रुव uint8_t * सीमित src,
                                      uint8_t * सीमित dst,
                                      ध्रुव पूर्णाङ्क अङ्कः_elems,
                                      भिन्न     scale,
                                      ध्रुव uint8_t * सीमित आवरणम्,
                                      भिन्न नतिः) {
    ध्रुव HVX_सदिशम् * सीमित v_src  = (ध्रुव HVX_सदिशम् *) src;
    HVX_सदिशम् * सीमित v_dst        = (HVX_सदिशम् *) dst;
    ध्रुव HVX_सदिशम् * सीमित v_आवरणम् = (ध्रुव HVX_सदिशम् *) आवरणम्;

    HVX_सदिशम् scale_vec = hvx_vec_splat_f32(scale);
    HVX_सदिशम् नतिः_vec = hvx_vec_splat_f32(नतिः);

    ध्रुव पूर्णाङ्क nvec_64 = अङ्कः_elems / VLEN_FP16;
    ध्रुव पूर्णाङ्क nloe_64 = अङ्कः_elems % VLEN_FP16;

    #निर्देशः unroll(2)
    प्रति (पूर्णाङ्क i = 0; i < nvec_64; i++) {
        HVX_सदिशम्_Pair p = hvx_vec_f16_गन्तव्यम्_f32(v_आवरणम्[i]);
        HVX_सदिशम् m0 = Q6_V_lo_W(p);
        HVX_सदिशम् m1 = Q6_V_hi_W(p);

        HVX_सदिशम् s0 = v_src[2 * i];
        HVX_सदिशम् s1 = v_src[2 * i + 1];

        HVX_सदिशम् v0 = Q6_Vqf32_vadd_Vqf32Vqf32(Q6_Vqf32_vmpy_VsfVsf(s0, scale_vec), Q6_Vqf32_vmpy_VsfVsf(m0, नतिः_vec));
        HVX_सदिशम् v1 = Q6_Vqf32_vadd_Vqf32Vqf32(Q6_Vqf32_vmpy_VsfVsf(s1, scale_vec), Q6_Vqf32_vmpy_VsfVsf(m1, नतिः_vec));

        v_dst[2 * i]     = Q6_Vsf_equals_Vqf32(v0);
        v_dst[2 * i + 1] = Q6_Vsf_equals_Vqf32(v1);
    }

    यदि (nloe_64 > 0) {
        HVX_सदिशम्_Pair p = hvx_vec_f16_गन्तव्यम्_f32(v_आवरणम्[nvec_64]);
        HVX_सदिशम् m0 = Q6_V_lo_W(p);

        HVX_सदिशम् s0 = v_src[2 * nvec_64];
        HVX_सदिशम् v0 = Q6_Vqf32_vadd_Vqf32Vqf32(Q6_Vqf32_vmpy_VsfVsf(s0, scale_vec), Q6_Vqf32_vmpy_VsfVsf(m0, नतिः_vec));

        यदि (nloe_64 <= VLEN_FP32) {
            hvx_vec_store_a(&v_dst[2 * nvec_64], nloe_64 * परिमाण(भिन्न), Q6_Vsf_equals_Vqf32(v0));
        } अन्यथा {
            v_dst[2 * nvec_64] = Q6_Vsf_equals_Vqf32(v0);

            HVX_सदिशम् m1 = Q6_V_hi_W(p);
            HVX_सदिशम् s1 = v_src[2 * nvec_64 + 1];
            HVX_सदिशम् v1 = Q6_Vqf32_vadd_Vqf32Vqf32(Q6_Vqf32_vmpy_VsfVsf(s1, scale_vec), Q6_Vqf32_vmpy_VsfVsf(m1, नतिः_vec));

            hvx_vec_store_a(&v_dst[2 * nvec_64 + 1], (nloe_64 - VLEN_FP32) * परिमाण(भिन्न), Q6_Vsf_equals_Vqf32(v1));
        }
    }
}

स्थिर शून्य hvx_fast_softmax_prep_f32(ध्रुव uint8_t * सीमित src,
                                      uint8_t * सीमित dst,
                                      ध्रुव पूर्णाङ्क अङ्कः_elems,
                                      भिन्न     scale,
                                      ध्रुव uint8_t * सीमित आवरणम्,
                                      भिन्न नतिः) {
    ध्रुव HVX_सदिशम् * सीमित v_src  = (ध्रुव HVX_सदिशम् *) src;
    HVX_सदिशम् * सीमित v_dst        = (HVX_सदिशम् *) dst;
    ध्रुव HVX_सदिशम् * सीमित v_आवरणम् = (ध्रुव HVX_सदिशम् *) आवरणम्;

    HVX_सदिशम् scale_vec = hvx_vec_splat_f32(scale);
    HVX_सदिशम् नतिः_vec = hvx_vec_splat_f32(नतिः);

    ध्रुव पूर्णाङ्क nvec = अङ्कः_elems / VLEN_FP32;
    ध्रुव पूर्णाङ्क nloe = अङ्कः_elems % VLEN_FP32;

    #निर्देशः unroll(4)
    प्रति (पूर्णाङ्क i = 0; i < nvec; i++) {
        HVX_सदिशम् v1 = v_src[i];
        HVX_सदिशम् v3 = v_आवरणम्[i];

        HVX_सदिशम् v2 = Q6_Vqf32_vmpy_VsfVsf(v1, scale_vec);
        HVX_सदिशम् v4 = Q6_Vqf32_vmpy_VsfVsf(v3, नतिः_vec);
        HVX_सदिशम् v5 = Q6_Vqf32_vadd_Vqf32Vqf32(v2, v4);

        v_dst[i] = Q6_Vsf_equals_Vqf32(v5);
    }

    यदि (nloe > 0) {
        HVX_सदिशम् v1 = v_src[nvec];
        HVX_सदिशम् v3 = v_आवरणम्[nvec];

        HVX_सदिशम् v2 = Q6_Vqf32_vmpy_VsfVsf(v1, scale_vec);
        HVX_सदिशम् v4 = Q6_Vqf32_vmpy_VsfVsf(v3, नतिः_vec);
        HVX_सदिशम् v5 = Q6_Vqf32_vadd_Vqf32Vqf32(v2, v4);

        hvx_vec_store_a(&v_dst[nvec], nloe * परिमाण(भिन्न), Q6_Vsf_equals_Vqf32(v5));
    }
}

स्थिर शून्य hvx_fast_softmax_f32(ध्रुव uint8_t * सीमित src, uint8_t * सीमित dst, ध्रुव पूर्णाङ्क अङ्कः_elems) {
    ध्रुव HVX_सदिशम् * सीमित v_src = (ध्रुव HVX_सदिशम् *) src;
    HVX_सदिशम् * सीमित v_dst       = (HVX_सदिशम् *) dst;

    ध्रुव पूर्णाङ्क nvec = अङ्कः_elems / VLEN_FP32;
    ध्रुव पूर्णाङ्क nloe = अङ्कः_elems % VLEN_FP32;

    HVX_सदिशम् max_vec = hvx_vec_splat_f32(((ध्रुव भिन्न *) src)[0]);

    #निर्देशः unroll(2)
    प्रति (पूर्णाङ्क i = 0; i < nvec; i++) {
        HVX_सदिशम् v1 = v_src[i];
        max_vec       = Q6_Vsf_vmax_VsfVsf(max_vec, v1);
    }

    यदि (nloe > 0) {
        HVX_सदिशम्_Pred q_आवरणम् = Q6_Q_vsetq_R(nloe * परिमाण(भिन्न));
        HVX_सदिशम् neg_inf    = hvx_vec_splat_f32(-INFINITY);
        HVX_सदिशम् v_पुच्छ     = Q6_V_vmux_QVV(q_आवरणम्, v_src[nvec], neg_inf);
        max_vec               = Q6_Vsf_vmax_VsfVsf(max_vec, v_पुच्छ);
    }

    max_vec = hvx_vec_reduce_max_f32(max_vec);

    HVX_सदिशम् योग_vec = Q6_V_vsplat_R(0x00000000);

    #निर्देशः unroll(2)
    प्रति (पूर्णाङ्क i = 0; i < nvec; i++) {
        HVX_सदिशम् v1 = v_src[i];
        HVX_सदिशम् v2 = Q6_Vqf32_vsub_VsfVsf(v1, max_vec);

        HVX_सदिशम् v3 = hvx_vec_exp_f32(Q6_Vsf_equals_Vqf32(v2));

        योग_vec = Q6_Vqf32_vadd_VsfVsf(Q6_Vsf_equals_Vqf32(योग_vec), v3);

        v_dst[i] = v3;
    }

    यदि (nloe > 0) {
        HVX_सदिशम्_Pred q_आवरणम् = Q6_Q_vsetq_R(nloe * परिमाण(भिन्न));
        HVX_सदिशम् v1     = v_src[nvec];
        HVX_सदिशम् v2     = Q6_Vqf32_vsub_VsfVsf(v1, max_vec);
        HVX_सदिशम् v3     = hvx_vec_exp_f32(Q6_Vsf_equals_Vqf32(v2));
        HVX_सदिशम् v3_pad = Q6_V_vmux_QVV(q_आवरणम्, v3, Q6_V_vzero());

        योग_vec     = Q6_Vqf32_vadd_VsfVsf(Q6_Vsf_equals_Vqf32(योग_vec), v3_pad);
        v_dst[nvec] = v3_pad;
    }

    योग_vec = hvx_vec_reduce_योग_f32(Q6_Vsf_equals_Vqf32(योग_vec));

    HVX_सदिशम्_Pred pos_योग   = Q6_Q_vcmp_gt_VwVw(योग_vec, Q6_V_vzero());
    HVX_सदिशम्     v4        = hvx_vec_inverse_f32(योग_vec);
    HVX_सदिशम्     scale_vec = Q6_V_vmux_QVV(pos_योग, v4, hvx_vec_splat_f32(1.0f));

    #निर्देशः unroll(2)
    प्रति (पूर्णाङ्क i = 0; i < nvec; i++) {
        HVX_सदिशम् v1 = v_dst[i];
        HVX_सदिशम् v2 = Q6_Vqf32_vmpy_VsfVsf(v1, scale_vec);
        v_dst[i]      = Q6_Vsf_equals_Vqf32(v2);
    }

    यदि (nloe > 0) {
        HVX_सदिशम् v1 = v_dst[nvec];
        HVX_सदिशम् v2 = Q6_Vqf32_vmpy_VsfVsf(v1, scale_vec);
        hvx_vec_store_a(&v_dst[nvec], nloe * परिमाण(भिन्न), Q6_Vsf_equals_Vqf32(v2));
    }
}

स्थिर शून्य compute_fast_softmax_f32_nomask(
    शून्य * सीमित dst,
    ध्रुव शून्य * सीमित src0,
    ध्रुव शून्य * सीमित आवरणम्,
    uint32_t ne00,
    भिन्न scale,
    भिन्न नतिः
) {
    (शून्य) आवरणम्;
    (शून्य) नतिः;
    hvx_scale_f32((uint8_t *) dst, (ध्रुव uint8_t *) src0, ne00, scale);
    hvx_fast_softmax_f32((ध्रुव uint8_t *) dst, (uint8_t *) dst, ne00);
}

स्थिर शून्य compute_fast_softmax_f32_आवरणम्_f32(
    शून्य * सीमित dst,
    ध्रुव शून्य * सीमित src0,
    ध्रुव शून्य * सीमित आवरणम्,
    uint32_t ne00,
    भिन्न scale,
    भिन्न नतिः
) {
    hvx_fast_softmax_prep_f32((ध्रुव uint8_t *) src0, (uint8_t *) dst, ne00, scale, (ध्रुव uint8_t *) आवरणम्, नतिः);
    hvx_fast_softmax_f32((ध्रुव uint8_t *) dst, (uint8_t *) dst, ne00);
}

स्थिर शून्य compute_fast_softmax_f32_आवरणम्_f16(
    शून्य * सीमित dst,
    ध्रुव शून्य * सीमित src0,
    ध्रुव शून्य * सीमित आवरणम्,
    uint32_t ne00,
    भिन्न scale,
    भिन्न नतिः
) {
    hvx_fast_softmax_prep_f16((ध्रुव uint8_t *) src0, (uint8_t *) dst, ne00, scale, (ध्रुव uint8_t *) आवरणम्, नतिः);
    hvx_fast_softmax_f32((ध्रुव uint8_t *) dst, (uint8_t *) dst, ne00);
}

स्थिर ध्रुव softmax_compute_fn_t softmax_kernels[HTP_SOFTMAX_KERNEL_गणनम्] = {
    [HTP_SOFTMAX_KERNEL_NOMASK]   = compute_fast_softmax_f32_nomask,
    [HTP_SOFTMAX_KERNEL_आवरणम्_F32] = compute_fast_softmax_f32_आवरणम्_f32,
    [HTP_SOFTMAX_KERNEL_आवरणम्_F16] = compute_fast_softmax_f32_आवरणम्_f16,
};

स्थिर शून्य softmax_thread_dma(अचिह्नित पूर्णाङ्क nth, अचिह्नित पूर्णाङ्क ith, शून्य * दत्त) {
    (शून्य) nth;
    ध्रुव संरचना htp_softmax_context * smctx = (ध्रुव संरचना htp_softmax_context *) दत्त;
    संरचना htp_ops_context * octx = smctx->octx;
    ध्रुव संरचना htp_tensor * src0 = octx->src[0];
    ध्रुव संरचना htp_tensor * dst  = octx->dst;
    संरचना htp_thread_trace * tr = &octx->ctx->trace[ith];

    ध्रुव uint32_t src0_nrows            = smctx->nrows;
    ध्रुव uint32_t src0_nrows_प्रत्येकम्_thread = smctx->src0_nrows_प्रत्येकम्_thread;

    ध्रुव uint32_t src0_start_row = smctx->row_start + src0_nrows_प्रत्येकम्_thread * ith;
    ध्रुव uint32_t src0_अन्तः_row   = MIN(src0_start_row + src0_nrows_प्रत्येकम्_thread, smctx->row_start + src0_nrows);

    यदि (src0_start_row >= src0_अन्तः_row) {
        प्रत्यागम;
    }

    ध्रुव dma_addr_t दत्त_src0 = smctx->दत्त_src0;
    ध्रुव dma_addr_t दत्त_dst  = smctx->दत्त_dst;

    ध्रुव size_t src0_row_आकार = src0->ne[0] * परिमाण(भिन्न);
    ध्रुव size_t dst_row_आकार  = src0->ne[0] * परिमाण(भिन्न);

    uint8_t * src0_vtcm_आधारः = smctx->vtcm_src0 + (ith * smctx->vtcm_src0_आकार_प्रत्येकम्_thread);
    uint8_t * dst_vtcm_आधारः  = smctx->vtcm_dst  + (ith * smctx->vtcm_dst_आकार_प्रत्येकम्_thread);

    ध्रुव size_t src0_vtcm_half = smctx->src0_spad_half_आकार;
    ध्रुव size_t dst_vtcm_half  = smctx->dst_spad_half_आकार;

    dma_queue * dma_q = octx->ctx->dma[ith];

    प्रति (uint32_t r = src0_start_row, idx = 0; r < src0_अन्तः_row && idx < 2; r++, idx++) {
        dma_addr_t cur_dst  = दत्त_dst  + r * dst_row_आकार;
        dma_addr_t cur_src0 = दत्त_src0 + r * src0_row_आकार;
        शून्य * d_spad = dst_vtcm_आधारः  + idx * dst_vtcm_half;
        शून्य * s_spad = src0_vtcm_आधारः + idx * src0_vtcm_half;

        dma_queue_प्रक्षिप(dma_q, dma_make_दत्त(cur_dst, d_spad),
                       dst_row_आकार, smctx->dst_row_आकार_aligned, dst_row_आकार, 0);
        dma_queue_प्रक्षिप(dma_q, dma_make_दत्त(s_spad, cur_src0),
                       smctx->src0_row_आकार_aligned, src0_row_आकार, src0_row_आकार, 1);
    }

    softmax_compute_fn_t compute = (softmax_compute_fn_t) smctx->compute;
    ध्रुव uint32_t ne00 = src0->ne[0];

    प्रति (uint32_t r = src0_start_row; r < src0_अन्तः_row; ++r) {
        शून्य * d_spad = (शून्य *) dma_queue_निष्कासय(dma_q).src;
        शून्य * s_spad = (शून्य *) dma_queue_निष्कासय(dma_q).dst;

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, r);
        compute(d_spad, s_spad, रिक्त, ne00, smctx->scale, 1.0f);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, r);

        dma_addr_t cur_dst = दत्त_dst + r * dst_row_आकार;
        dma_queue_प्रक्षिप(dma_q, dma_make_दत्त(cur_dst, d_spad),
                       dst_row_आकार, smctx->dst_row_आकार_aligned, dst_row_आकार, 1);

        ध्रुव uint32_t अग्र_r = r + 2;
        यदि (अग्र_r < src0_अन्तः_row) {
            dma_addr_t अग्र_src0 = दत्त_src0 + अग्र_r * src0_row_आकार;
            dma_queue_प्रक्षिप(dma_q, dma_make_दत्त(s_spad, अग्र_src0),
                           smctx->src0_row_आकार_aligned, src0_row_आकार, src0_row_आकार, 1);
        }
    }

    dma_queue_flush(dma_q);
}

स्थिर शून्य softmax_thread_आवरणम्_dma(अचिह्नित पूर्णाङ्क nth, अचिह्नित पूर्णाङ्क ith, शून्य * दत्त) {
    (शून्य) nth;
    ध्रुव संरचना htp_softmax_context * smctx = (ध्रुव संरचना htp_softmax_context *) दत्त;
    संरचना htp_ops_context * octx = smctx->octx;
    ध्रुव संरचना htp_tensor * src0 = octx->src[0];
    ध्रुव संरचना htp_tensor * src1 = octx->src[1];
    ध्रुव संरचना htp_tensor * dst  = octx->dst;
    संरचना htp_thread_trace * tr = &octx->ctx->trace[ith];

    ध्रुव uint32_t src0_nrows            = smctx->nrows;
    ध्रुव uint32_t src0_nrows_प्रत्येकम्_thread = smctx->src0_nrows_प्रत्येकम्_thread;

    ध्रुव uint32_t src0_start_row = smctx->row_start + src0_nrows_प्रत्येकम्_thread * ith;
    ध्रुव uint32_t src0_अन्तः_row   = MIN(src0_start_row + src0_nrows_प्रत्येकम्_thread, smctx->row_start + src0_nrows);

    यदि (src0_start_row >= src0_अन्तः_row) {
        प्रत्यागम;
    }

    ध्रुव dma_addr_t दत्त_src0 = smctx->दत्त_src0;
    ध्रुव dma_addr_t दत्त_src1 = smctx->दत्त_src1;
    ध्रुव dma_addr_t दत्त_dst  = smctx->दत्त_dst;

    ध्रुव size_t src0_row_आकार = src0->ne[0] * परिमाण(भिन्न);
    ध्रुव size_t dst_row_आकार  = src0->ne[0] * परिमाण(भिन्न);
    ध्रुव size_t आवरणम्_row_आकार = smctx->use_f16 ? (src1->ne[0] * परिमाण(__fp16)) : (src1->ne[0] * परिमाण(भिन्न));

    uint8_t * src0_vtcm_आधारः = smctx->vtcm_src0 + (ith * smctx->vtcm_src0_आकार_प्रत्येकम्_thread);
    uint8_t * src1_vtcm_आधारः = smctx->vtcm_src1 + (ith * smctx->vtcm_src1_आकार_प्रत्येकम्_thread);
    uint8_t * dst_vtcm_आधारः  = smctx->vtcm_dst  + (ith * smctx->vtcm_dst_आकार_प्रत्येकम्_thread);

    ध्रुव size_t src0_vtcm_half = smctx->src0_spad_half_आकार;
    ध्रुव size_t src1_vtcm_half = smctx->src1_spad_half_आकार;
    ध्रुव size_t dst_vtcm_half  = smctx->dst_spad_half_आकार;

    ध्रुव uint32_t nb11 = src1->nb[1];
    ध्रुव uint32_t nb12 = src1->nb[2];
    ध्रुव uint32_t nb13 = src1->nb[3];

    ध्रुव uint32_t ne00 = src0->ne[0];
    ध्रुव uint32_t ne01 = src0->ne[1];
    ध्रुव uint32_t ne02 = src0->ne[2];
    ध्रुव uint32_t ne03 = src0->ne[3];
    ध्रुव uint32_t ne12 = src1->ne[2];
    ध्रुव uint32_t ne13 = src1->ne[3];

    ध्रुव संरचना fastdiv_मानानि * div_ne01 = &smctx->div_ne01;
    ध्रुव संरचना fastdiv_मानानि * div_ne02 = &smctx->div_ne02;
    ध्रुव संरचना fastdiv_मानानि * div_ne12 = &smctx->div_ne12;
    ध्रुव संरचना fastdiv_मानानि * div_ne13 = &smctx->div_ne13;

    dma_queue * dma_q = octx->ctx->dma[ith];

    प्रति (uint32_t r = src0_start_row, idx = 0; r < src0_अन्तः_row && idx < 2; r++, idx++) {
        dma_addr_t cur_dst  = दत्त_dst  + r * dst_row_आकार;
        dma_addr_t cur_src0 = दत्त_src0 + r * src0_row_आकार;

        uint32_t i1 = fastmodulo(r, ne01, div_ne01);
        uint32_t r_div_ne01 = fastdiv(r, div_ne01);
        uint32_t i2 = fastmodulo(r_div_ne01, ne02, div_ne02);
        uint32_t i3 = fastdiv(r_div_ne01, div_ne02);
        uint32_t i12 = (ne12 == ne02) ? i2 : fastmodulo(i2, ne12, div_ne12);
        uint32_t i13 = (ne13 == ne03) ? i3 : fastmodulo(i3, ne13, div_ne13);
        dma_addr_t cur_src1 = दत्त_src1 + i1 * nb11 + i12 * nb12 + i13 * nb13;

        शून्य * d_spad = dst_vtcm_आधारः  + idx * dst_vtcm_half;
        शून्य * s_spad = src0_vtcm_आधारः + idx * src0_vtcm_half;
        शून्य * m_spad = src1_vtcm_आधारः + idx * src1_vtcm_half;

        dma_queue_प्रक्षिप(dma_q, dma_make_दत्त(cur_dst, d_spad),
                       dst_row_आकार, smctx->dst_row_आकार_aligned, dst_row_आकार, 0);
        dma_queue_प्रक्षिप(dma_q, dma_make_दत्त(s_spad, cur_src0),
                       smctx->src0_row_आकार_aligned, src0_row_आकार, src0_row_आकार, 1);
        dma_queue_प्रक्षिप(dma_q, dma_make_दत्त(m_spad, cur_src1),
                       smctx->src1_row_आकार_aligned, आवरणम्_row_आकार, आवरणम्_row_आकार, 1);
    }

    softmax_compute_fn_t compute = (softmax_compute_fn_t) smctx->compute;
    ध्रुव bool has_bias = smctx->max_bias > 0.0f;
    uint32_t पूर्व_i2 = (uint32_t)-1;
    भिन्न नतिः = 1.0f;

    प्रति (uint32_t r = src0_start_row; r < src0_अन्तः_row; ++r) {
        शून्य * d_spad = (शून्य *) (uintptr_t) dma_queue_निष्कासय(dma_q).src;
        शून्य * s_spad = (शून्य *) (uintptr_t) dma_queue_निष्कासय(dma_q).dst;
        शून्य * m_spad = (शून्य *) (uintptr_t) dma_queue_निष्कासय(dma_q).dst;

        यदि (has_bias) {
            uint32_t r_div_ne01 = fastdiv(r, div_ne01);
            uint32_t i2 = fastmodulo(r_div_ne01, ne02, div_ne02);
            यदि (i2 != पूर्व_i2) {
                नतिः = smctx->slopes[i2];
                पूर्व_i2 = i2;
            }
        }

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, r);
        compute(d_spad, s_spad, m_spad, ne00, smctx->scale, नतिः);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, r);

        dma_addr_t cur_dst = दत्त_dst + r * dst_row_आकार;
        dma_queue_प्रक्षिप(dma_q, dma_make_दत्त(cur_dst, d_spad),
                       dst_row_आकार, smctx->dst_row_आकार_aligned, dst_row_आकार, 1);

        ध्रुव uint32_t अग्र_r = r + 2;
        यदि (अग्र_r < src0_अन्तः_row) {
            dma_addr_t अग्र_src0 = दत्त_src0 + अग्र_r * src0_row_आकार;

            uint32_t ni1 = fastmodulo(अग्र_r, ne01, div_ne01);
            uint32_t nr_div_ne01 = fastdiv(अग्र_r, div_ne01);
            uint32_t ni2 = fastmodulo(nr_div_ne01, ne02, div_ne02);
            uint32_t ni3 = fastdiv(nr_div_ne01, div_ne02);
            uint32_t ni12 = (ne12 == ne02) ? ni2 : fastmodulo(ni2, ne12, div_ne12);
            uint32_t ni13 = (ne13 == ne03) ? ni3 : fastmodulo(ni3, ne13, div_ne13);
            dma_addr_t अग्र_src1 = दत्त_src1 + ni1 * nb11 + ni12 * nb12 + ni13 * nb13;

            dma_queue_प्रक्षिप(dma_q, dma_make_दत्त(s_spad, अग्र_src0),
                           smctx->src0_row_आकार_aligned, src0_row_आकार, src0_row_आकार, 1);
            dma_queue_प्रक्षिप(dma_q, dma_make_दत्त(m_spad, अग्र_src1),
                           smctx->src1_row_आकार_aligned, आवरणम्_row_आकार, आवरणम्_row_आकार, 1);
        }
    }

    dma_queue_flush(dma_q);
}

स्थिर पूर्णाङ्क execute_op_softmax_f32(संरचना htp_ops_context * octx) {
    ध्रुव संरचना htp_tensor * src0 = octx->src[0];
    ध्रुव संरचना htp_tensor * dst  = octx->dst;

    ध्रुव वर्ण * op_type = "softmax-f32";

    ध्रुव संरचना htp_softmax_kernel_params * kparams =
        (ध्रुव संरचना htp_softmax_kernel_params *) octx->kernel_params;

    यदि (!htp_ops_context_समुच्चयः_n_तन्तवः(octx, kparams->n_तन्तवः)) {
        प्रत्यागम HTP_स्थितिः_INVAL_PARAMS;
    }

    यदि (kparams->kernel_id >= HTP_SOFTMAX_KERNEL_गणनम्) {
        प्रत्यागम HTP_स्थितिः_INVAL_PARAMS;
    }

    यदि (octx->ctx->vtcm_आकार < (size_t) kparams->vtcm_आकार) {
        FARF(ERROR, "%s : current VTCM reservation %zu is too small, needed %u\n",
             op_type, octx->ctx->vtcm_आकार, kparams->vtcm_आकार);
        प्रत्यागम HTP_स्थितिः_VTCM_TOO_SMALL;
    }

    ध्रुव uint32_t src0_nrows = src0->ne[1] * src0->ne[2] * src0->ne[3];
    ध्रुव size_t elem_आकार    = परिमाण(भिन्न);
    ध्रुव size_t dst_row_आकार = dst->nb[1];

    uint32_t row_start = 0;
    uint32_t nrows     = src0_nrows;

    यदि (octx->ctx->mdev.गणनम् > 1) {
        uint32_t पङ्क्तिकाः_प्रत्येकम्_chunk = 0;
        htp_tensor_mdev_पङ्क्तिकाः_प्रत्येकम्_chunk(dst, (uint32_t) elem_आकार, (uint32_t) dst_row_आकार, &पङ्क्तिकाः_प्रत्येकम्_chunk);
        ध्रुव संरचना htp_tensor_mdev_range range = htp_tensor_mdev_partition(
            src0_nrows, पङ्क्तिकाः_प्रत्येकम्_chunk, octx->ctx->mdev.idx, octx->ctx->mdev.गणनम्, &octx->ctx->mdev.गणनम्_div);
        row_start = range.start;
        nrows     = range.गणनम्;
        यदि (nrows < octx->n_तन्तवः) {
            htp_ops_context_समुच्चयः_n_तन्तवः(octx, nrows ? nrows : 1);
        }
    }

    यदि (nrows == 0) {
        प्रत्यागम HTP_स्थितिः_OK;
    }

    ध्रुव uint32_t n_तन्तवः = octx->n_तन्तवः;
    uint8_t * ध्रुव vtcm_आधारः = (uint8_t *) octx->ctx->vtcm_आधारः;

    ध्रुव uint32_t off_src0 = 0;
    ध्रुव uint32_t off_dst  = off_src0 + kparams->vtcm_src0_आकार_प्रत्येकम्_thread * kparams->n_तन्तवः;
    ध्रुव uint32_t off_src1 = off_dst  + kparams->vtcm_dst_आकार_प्रत्येकम्_thread  * kparams->n_तन्तवः;

    संरचना htp_softmax_context smctx = {
        .octx                  = octx,
        .kparams               = kparams,
        .compute               = (शून्य *) softmax_kernels[kparams->kernel_id],

        .दत्त_src0             = src0->दत्त,
        .दत्त_src1             = kparams->use_src1 ? octx->src[1]->दत्त : 0,
        .दत्त_dst              = dst->दत्त,

        .vtcm_src0             = VTCM_LAYOUT_PTR(uint8_t, vtcm_आधारः, off_src0),
        .vtcm_dst              = VTCM_LAYOUT_PTR(uint8_t, vtcm_आधारः, off_dst),
        .vtcm_src1             = VTCM_LAYOUT_PTR_OPTIONAL(uint8_t, vtcm_आधारः, off_src1, kparams->use_src1),

        .vtcm_src0_आकार_प्रत्येकम्_thread = kparams->vtcm_src0_आकार_प्रत्येकम्_thread,
        .vtcm_src1_आकार_प्रत्येकम्_thread = kparams->vtcm_src1_आकार_प्रत्येकम्_thread,
        .vtcm_dst_आकार_प्रत्येकम्_thread  = kparams->vtcm_dst_आकार_प्रत्येकम्_thread,

        .src0_spad_half_आकार   = kparams->src0_spad_half_आकार,
        .src1_spad_half_आकार   = kparams->src1_spad_half_आकार,
        .dst_spad_half_आकार    = kparams->dst_spad_half_आकार,

        .src0_row_आकार_aligned = kparams->src0_row_आकार_aligned,
        .src1_row_आकार_aligned = kparams->src1_row_आकार_aligned,
        .dst_row_आकार_aligned  = kparams->dst_row_आकार_aligned,

        .use_f16               = kparams->use_f16 != 0,

        .n_शीर्ष                = kparams->n_शीर्ष,
        .n_शीर्ष_log2           = kparams->n_शीर्ष_log2,

        .scale                 = kparams->scale,
        .max_bias              = kparams->max_bias,
        .m0                    = kparams->m0,
        .m1                    = kparams->m1,

        .div_ne01              = kparams->div_ne01,
        .div_ne02              = kparams->div_ne02,
        .div_ne12              = kparams->div_ne12,
        .div_ne13              = kparams->div_ne13,

        .src0_nrows_प्रत्येकम्_thread = fastdiv(nrows + n_तन्तवः - 1, &octx->n_तन्तवः_div),
        .row_start             = row_start,
        .nrows                 = nrows,
    };

    यदि (kparams->max_bias > 0.0f && kparams->use_src1) {
        यदि (kparams->n_शीर्ष > 512) {
            प्रत्यागम HTP_स्थितिः_INVAL_PARAMS;
        }
        प्रति (uint32_t h = 0; h < kparams->n_शीर्ष; h += 32) {
            HVX_सदिशम् v_slopes = hvx_alibi_slopes(h, 1, kparams->n_शीर्ष_log2, kparams->m0, kparams->m1);
            hvx_vmem(&smctx.slopes[h]) = v_slopes;
        }
    }

    work_queue_func_t task_func = kparams->use_src1 ? softmax_thread_आवरणम्_dma : softmax_thread_dma;
    work_queue_सततम्(octx->ctx->work_queue, task_func, &smctx, n_तन्तवः);

    प्रत्यागम HTP_स्थितिः_OK;
}

पूर्णाङ्क op_softmax(संरचना htp_ops_context * octx) {
    पूर्णाङ्क err = HTP_स्थितिः_OK;

    विकल्प (octx->src[0]->type) {
        पक्ष HTP_TYPE_F32:
            err = execute_op_softmax_f32(octx);
            भङ्ग;

        स्वभाव:
            err = HTP_स्थितिः_NO_SUPPORT;
            भङ्ग;
    }

    प्रत्यागम err;
}
