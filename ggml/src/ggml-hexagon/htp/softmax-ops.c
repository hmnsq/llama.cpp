#ਹਦਾਇਤ clang diagnostic ignored "-Wunused-variable"
#ਹਦਾਇਤ clang diagnostic ignored "-Wunused-function"
#ਹਦਾਇਤ clang diagnostic ignored "-Wunused-but-set-variable"

#ਅਨ੍ਤਰ੍ਭਾਵ <HAP_farf.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <HAP_perf.h>

#ਅਨ੍ਤਰ੍ਭਾਵ <math.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <string.h>

#ਅਨ੍ਤਰ੍ਭਾਵ "dma-queue.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "work-queue.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "hvx-utils.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "hex-fastdiv.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "hex-common.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "hex-profile.h"

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_COMMON_DECL_C
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-common.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "htp-ctx.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "htp-ops.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "htp-tensor.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "htp-vtcm.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "htp/softmax-ops.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "hvx-flash-attn.h"

ਬਣਤਰ htp_softmax_context {
    ਬਣਤਰ htp_ops_context * octx;
    ਅਚਲ ਬਣਤਰ htp_softmax_kernel_params * kparams;

    ਖਾਲੀ *   compute;

    dma_addr_t ਅੰਕੜੇ_src0;
    dma_addr_t ਅੰਕੜੇ_src1;
    dma_addr_t ਅੰਕੜੇ_dst;

    uint8_t * vtcm_src0;
    uint8_t * vtcm_src1;
    uint8_t * vtcm_dst;

    uint32_t  vtcm_src0_ਮਾਪ_ਪ੍ਰਤੀ_thread;
    uint32_t  vtcm_src1_ਮਾਪ_ਪ੍ਰਤੀ_thread;
    uint32_t  vtcm_dst_ਮਾਪ_ਪ੍ਰਤੀ_thread;

    uint32_t  src0_spad_half_ਮਾਪ;
    uint32_t  src1_spad_half_ਮਾਪ;
    uint32_t  dst_spad_half_ਮਾਪ;

    uint32_t  src0_ਪੰਕਤੀ_ਮਾਪ_aligned;
    uint32_t  src1_ਪੰਕਤੀ_ਮਾਪ_aligned;
    uint32_t  dst_ਪੰਕਤੀ_ਮਾਪ_aligned;

    bool     use_f16;

    uint32_t n_ਸਿਰ;
    uint32_t n_ਸਿਰ_log2;

    ਦਸ਼ਮਲਵ    scale;
    ਦਸ਼ਮਲਵ    max_bias;
    ਦਸ਼ਮਲਵ    m0;
    ਦਸ਼ਮਲਵ    m1;

    ਬਣਤਰ fastdiv_ਮੁੱਲਾਂ div_ne01;
    ਬਣਤਰ fastdiv_ਮੁੱਲਾਂ div_ne02;
    ਬਣਤਰ fastdiv_ਮੁੱਲਾਂ div_ne12;
    ਬਣਤਰ fastdiv_ਮੁੱਲਾਂ div_ne13;

    uint32_t src0_nrows_ਪ੍ਰਤੀ_thread;
    uint32_t ਪੰਕਤੀ_ਸ਼ੁਰੂ;
    uint32_t nrows;

    ਦਸ਼ਮਲਵ    slopes[512] __attribute__((aligned(128)));
};

ਕਿਸਮਪਰਿਭਾਸ਼ਾ ਖਾਲੀ (*softmax_compute_fn_t)(
    ਖਾਲੀ * ਸੀਮਿਤ dst,
    ਅਚਲ ਖਾਲੀ * ਸੀਮਿਤ src0,
    ਅਚਲ ਖਾਲੀ * ਸੀਮਿਤ ਨਕਾਬ,
    uint32_t ne00,
    ਦਸ਼ਮਲਵ scale,
    ਦਸ਼ਮਲਵ ਢਲਾਨ
);

ਸਥਿਰ ਖਾਲੀ hvx_fast_softmax_prep_f16(ਅਚਲ uint8_t * ਸੀਮਿਤ src,
                                      uint8_t * ਸੀਮਿਤ dst,
                                      ਅਚਲ ਪੂਰਨ ਅੰਕ_elems,
                                      ਦਸ਼ਮਲਵ     scale,
                                      ਅਚਲ uint8_t * ਸੀਮਿਤ ਨਕਾਬ,
                                      ਦਸ਼ਮਲਵ ਢਲਾਨ) {
    ਅਚਲ HVX_ਸਦਿਸ਼ * ਸੀਮਿਤ v_src  = (ਅਚਲ HVX_ਸਦਿਸ਼ *) src;
    HVX_ਸਦਿਸ਼ * ਸੀਮਿਤ v_dst        = (HVX_ਸਦਿਸ਼ *) dst;
    ਅਚਲ HVX_ਸਦਿਸ਼ * ਸੀਮਿਤ v_ਨਕਾਬ = (ਅਚਲ HVX_ਸਦਿਸ਼ *) ਨਕਾਬ;

    HVX_ਸਦਿਸ਼ scale_vec = hvx_vec_splat_f32(scale);
    HVX_ਸਦਿਸ਼ ਢਲਾਨ_vec = hvx_vec_splat_f32(ਢਲਾਨ);

    ਅਚਲ ਪੂਰਨ nvec_64 = ਅੰਕ_elems / VLEN_FP16;
    ਅਚਲ ਪੂਰਨ nloe_64 = ਅੰਕ_elems % VLEN_FP16;

    #ਹਦਾਇਤ unroll(2)
    ਲਈ (ਪੂਰਨ i = 0; i < nvec_64; i++) {
        HVX_ਸਦਿਸ਼_Pair p = hvx_vec_f16_ਮੰਜ਼ਿਲ_f32(v_ਨਕਾਬ[i]);
        HVX_ਸਦਿਸ਼ m0 = Q6_V_lo_W(p);
        HVX_ਸਦਿਸ਼ m1 = Q6_V_hi_W(p);

        HVX_ਸਦਿਸ਼ s0 = v_src[2 * i];
        HVX_ਸਦਿਸ਼ s1 = v_src[2 * i + 1];

        HVX_ਸਦਿਸ਼ v0 = Q6_Vqf32_vadd_Vqf32Vqf32(Q6_Vqf32_vmpy_VsfVsf(s0, scale_vec), Q6_Vqf32_vmpy_VsfVsf(m0, ਢਲਾਨ_vec));
        HVX_ਸਦਿਸ਼ v1 = Q6_Vqf32_vadd_Vqf32Vqf32(Q6_Vqf32_vmpy_VsfVsf(s1, scale_vec), Q6_Vqf32_vmpy_VsfVsf(m1, ਢਲਾਨ_vec));

        v_dst[2 * i]     = Q6_Vsf_equals_Vqf32(v0);
        v_dst[2 * i + 1] = Q6_Vsf_equals_Vqf32(v1);
    }

    ਜੇ (nloe_64 > 0) {
        HVX_ਸਦਿਸ਼_Pair p = hvx_vec_f16_ਮੰਜ਼ਿਲ_f32(v_ਨਕਾਬ[nvec_64]);
        HVX_ਸਦਿਸ਼ m0 = Q6_V_lo_W(p);

        HVX_ਸਦਿਸ਼ s0 = v_src[2 * nvec_64];
        HVX_ਸਦਿਸ਼ v0 = Q6_Vqf32_vadd_Vqf32Vqf32(Q6_Vqf32_vmpy_VsfVsf(s0, scale_vec), Q6_Vqf32_vmpy_VsfVsf(m0, ਢਲਾਨ_vec));

        ਜੇ (nloe_64 <= VLEN_FP32) {
            hvx_vec_store_a(&v_dst[2 * nvec_64], nloe_64 * ਆਕਾਰ(ਦਸ਼ਮਲਵ), Q6_Vsf_equals_Vqf32(v0));
        } ਨਹੀਂਤਾਂ {
            v_dst[2 * nvec_64] = Q6_Vsf_equals_Vqf32(v0);

            HVX_ਸਦਿਸ਼ m1 = Q6_V_hi_W(p);
            HVX_ਸਦਿਸ਼ s1 = v_src[2 * nvec_64 + 1];
            HVX_ਸਦਿਸ਼ v1 = Q6_Vqf32_vadd_Vqf32Vqf32(Q6_Vqf32_vmpy_VsfVsf(s1, scale_vec), Q6_Vqf32_vmpy_VsfVsf(m1, ਢਲਾਨ_vec));

            hvx_vec_store_a(&v_dst[2 * nvec_64 + 1], (nloe_64 - VLEN_FP32) * ਆਕਾਰ(ਦਸ਼ਮਲਵ), Q6_Vsf_equals_Vqf32(v1));
        }
    }
}

ਸਥਿਰ ਖਾਲੀ hvx_fast_softmax_prep_f32(ਅਚਲ uint8_t * ਸੀਮਿਤ src,
                                      uint8_t * ਸੀਮਿਤ dst,
                                      ਅਚਲ ਪੂਰਨ ਅੰਕ_elems,
                                      ਦਸ਼ਮਲਵ     scale,
                                      ਅਚਲ uint8_t * ਸੀਮਿਤ ਨਕਾਬ,
                                      ਦਸ਼ਮਲਵ ਢਲਾਨ) {
    ਅਚਲ HVX_ਸਦਿਸ਼ * ਸੀਮਿਤ v_src  = (ਅਚਲ HVX_ਸਦਿਸ਼ *) src;
    HVX_ਸਦਿਸ਼ * ਸੀਮਿਤ v_dst        = (HVX_ਸਦਿਸ਼ *) dst;
    ਅਚਲ HVX_ਸਦਿਸ਼ * ਸੀਮਿਤ v_ਨਕਾਬ = (ਅਚਲ HVX_ਸਦਿਸ਼ *) ਨਕਾਬ;

    HVX_ਸਦਿਸ਼ scale_vec = hvx_vec_splat_f32(scale);
    HVX_ਸਦਿਸ਼ ਢਲਾਨ_vec = hvx_vec_splat_f32(ਢਲਾਨ);

    ਅਚਲ ਪੂਰਨ nvec = ਅੰਕ_elems / VLEN_FP32;
    ਅਚਲ ਪੂਰਨ nloe = ਅੰਕ_elems % VLEN_FP32;

    #ਹਦਾਇਤ unroll(4)
    ਲਈ (ਪੂਰਨ i = 0; i < nvec; i++) {
        HVX_ਸਦਿਸ਼ v1 = v_src[i];
        HVX_ਸਦਿਸ਼ v3 = v_ਨਕਾਬ[i];

        HVX_ਸਦਿਸ਼ v2 = Q6_Vqf32_vmpy_VsfVsf(v1, scale_vec);
        HVX_ਸਦਿਸ਼ v4 = Q6_Vqf32_vmpy_VsfVsf(v3, ਢਲਾਨ_vec);
        HVX_ਸਦਿਸ਼ v5 = Q6_Vqf32_vadd_Vqf32Vqf32(v2, v4);

        v_dst[i] = Q6_Vsf_equals_Vqf32(v5);
    }

    ਜੇ (nloe > 0) {
        HVX_ਸਦਿਸ਼ v1 = v_src[nvec];
        HVX_ਸਦਿਸ਼ v3 = v_ਨਕਾਬ[nvec];

        HVX_ਸਦਿਸ਼ v2 = Q6_Vqf32_vmpy_VsfVsf(v1, scale_vec);
        HVX_ਸਦਿਸ਼ v4 = Q6_Vqf32_vmpy_VsfVsf(v3, ਢਲਾਨ_vec);
        HVX_ਸਦਿਸ਼ v5 = Q6_Vqf32_vadd_Vqf32Vqf32(v2, v4);

        hvx_vec_store_a(&v_dst[nvec], nloe * ਆਕਾਰ(ਦਸ਼ਮਲਵ), Q6_Vsf_equals_Vqf32(v5));
    }
}

ਸਥਿਰ ਖਾਲੀ hvx_fast_softmax_f32(ਅਚਲ uint8_t * ਸੀਮਿਤ src, uint8_t * ਸੀਮਿਤ dst, ਅਚਲ ਪੂਰਨ ਅੰਕ_elems) {
    ਅਚਲ HVX_ਸਦਿਸ਼ * ਸੀਮਿਤ v_src = (ਅਚਲ HVX_ਸਦਿਸ਼ *) src;
    HVX_ਸਦਿਸ਼ * ਸੀਮਿਤ v_dst       = (HVX_ਸਦਿਸ਼ *) dst;

    ਅਚਲ ਪੂਰਨ nvec = ਅੰਕ_elems / VLEN_FP32;
    ਅਚਲ ਪੂਰਨ nloe = ਅੰਕ_elems % VLEN_FP32;

    HVX_ਸਦਿਸ਼ max_vec = hvx_vec_splat_f32(((ਅਚਲ ਦਸ਼ਮਲਵ *) src)[0]);

    #ਹਦਾਇਤ unroll(2)
    ਲਈ (ਪੂਰਨ i = 0; i < nvec; i++) {
        HVX_ਸਦਿਸ਼ v1 = v_src[i];
        max_vec       = Q6_Vsf_vmax_VsfVsf(max_vec, v1);
    }

    ਜੇ (nloe > 0) {
        HVX_ਸਦਿਸ਼_Pred q_ਨਕਾਬ = Q6_Q_vsetq_R(nloe * ਆਕਾਰ(ਦਸ਼ਮਲਵ));
        HVX_ਸਦਿਸ਼ neg_inf    = hvx_vec_splat_f32(-INFINITY);
        HVX_ਸਦਿਸ਼ v_ਪੂਛ     = Q6_V_vmux_QVV(q_ਨਕਾਬ, v_src[nvec], neg_inf);
        max_vec               = Q6_Vsf_vmax_VsfVsf(max_vec, v_ਪੂਛ);
    }

    max_vec = hvx_vec_reduce_max_f32(max_vec);

    HVX_ਸਦਿਸ਼ ਜੋੜ_vec = Q6_V_vsplat_R(0x00000000);

    #ਹਦਾਇਤ unroll(2)
    ਲਈ (ਪੂਰਨ i = 0; i < nvec; i++) {
        HVX_ਸਦਿਸ਼ v1 = v_src[i];
        HVX_ਸਦਿਸ਼ v2 = Q6_Vqf32_vsub_VsfVsf(v1, max_vec);

        HVX_ਸਦਿਸ਼ v3 = hvx_vec_exp_f32(Q6_Vsf_equals_Vqf32(v2));

        ਜੋੜ_vec = Q6_Vqf32_vadd_VsfVsf(Q6_Vsf_equals_Vqf32(ਜੋੜ_vec), v3);

        v_dst[i] = v3;
    }

    ਜੇ (nloe > 0) {
        HVX_ਸਦਿਸ਼_Pred q_ਨਕਾਬ = Q6_Q_vsetq_R(nloe * ਆਕਾਰ(ਦਸ਼ਮਲਵ));
        HVX_ਸਦਿਸ਼ v1     = v_src[nvec];
        HVX_ਸਦਿਸ਼ v2     = Q6_Vqf32_vsub_VsfVsf(v1, max_vec);
        HVX_ਸਦਿਸ਼ v3     = hvx_vec_exp_f32(Q6_Vsf_equals_Vqf32(v2));
        HVX_ਸਦਿਸ਼ v3_pad = Q6_V_vmux_QVV(q_ਨਕਾਬ, v3, Q6_V_vzero());

        ਜੋੜ_vec     = Q6_Vqf32_vadd_VsfVsf(Q6_Vsf_equals_Vqf32(ਜੋੜ_vec), v3_pad);
        v_dst[nvec] = v3_pad;
    }

    ਜੋੜ_vec = hvx_vec_reduce_ਜੋੜ_f32(Q6_Vsf_equals_Vqf32(ਜੋੜ_vec));

    HVX_ਸਦਿਸ਼_Pred pos_ਜੋੜ   = Q6_Q_vcmp_gt_VwVw(ਜੋੜ_vec, Q6_V_vzero());
    HVX_ਸਦਿਸ਼     v4        = hvx_vec_inverse_f32(ਜੋੜ_vec);
    HVX_ਸਦਿਸ਼     scale_vec = Q6_V_vmux_QVV(pos_ਜੋੜ, v4, hvx_vec_splat_f32(1.0f));

    #ਹਦਾਇਤ unroll(2)
    ਲਈ (ਪੂਰਨ i = 0; i < nvec; i++) {
        HVX_ਸਦਿਸ਼ v1 = v_dst[i];
        HVX_ਸਦਿਸ਼ v2 = Q6_Vqf32_vmpy_VsfVsf(v1, scale_vec);
        v_dst[i]      = Q6_Vsf_equals_Vqf32(v2);
    }

    ਜੇ (nloe > 0) {
        HVX_ਸਦਿਸ਼ v1 = v_dst[nvec];
        HVX_ਸਦਿਸ਼ v2 = Q6_Vqf32_vmpy_VsfVsf(v1, scale_vec);
        hvx_vec_store_a(&v_dst[nvec], nloe * ਆਕਾਰ(ਦਸ਼ਮਲਵ), Q6_Vsf_equals_Vqf32(v2));
    }
}

ਸਥਿਰ ਖਾਲੀ compute_fast_softmax_f32_nomask(
    ਖਾਲੀ * ਸੀਮਿਤ dst,
    ਅਚਲ ਖਾਲੀ * ਸੀਮਿਤ src0,
    ਅਚਲ ਖਾਲੀ * ਸੀਮਿਤ ਨਕਾਬ,
    uint32_t ne00,
    ਦਸ਼ਮਲਵ scale,
    ਦਸ਼ਮਲਵ ਢਲਾਨ
) {
    (ਖਾਲੀ) ਨਕਾਬ;
    (ਖਾਲੀ) ਢਲਾਨ;
    hvx_scale_f32((uint8_t *) dst, (ਅਚਲ uint8_t *) src0, ne00, scale);
    hvx_fast_softmax_f32((ਅਚਲ uint8_t *) dst, (uint8_t *) dst, ne00);
}

ਸਥਿਰ ਖਾਲੀ compute_fast_softmax_f32_ਨਕਾਬ_f32(
    ਖਾਲੀ * ਸੀਮਿਤ dst,
    ਅਚਲ ਖਾਲੀ * ਸੀਮਿਤ src0,
    ਅਚਲ ਖਾਲੀ * ਸੀਮਿਤ ਨਕਾਬ,
    uint32_t ne00,
    ਦਸ਼ਮਲਵ scale,
    ਦਸ਼ਮਲਵ ਢਲਾਨ
) {
    hvx_fast_softmax_prep_f32((ਅਚਲ uint8_t *) src0, (uint8_t *) dst, ne00, scale, (ਅਚਲ uint8_t *) ਨਕਾਬ, ਢਲਾਨ);
    hvx_fast_softmax_f32((ਅਚਲ uint8_t *) dst, (uint8_t *) dst, ne00);
}

ਸਥਿਰ ਖਾਲੀ compute_fast_softmax_f32_ਨਕਾਬ_f16(
    ਖਾਲੀ * ਸੀਮਿਤ dst,
    ਅਚਲ ਖਾਲੀ * ਸੀਮਿਤ src0,
    ਅਚਲ ਖਾਲੀ * ਸੀਮਿਤ ਨਕਾਬ,
    uint32_t ne00,
    ਦਸ਼ਮਲਵ scale,
    ਦਸ਼ਮਲਵ ਢਲਾਨ
) {
    hvx_fast_softmax_prep_f16((ਅਚਲ uint8_t *) src0, (uint8_t *) dst, ne00, scale, (ਅਚਲ uint8_t *) ਨਕਾਬ, ਢਲਾਨ);
    hvx_fast_softmax_f32((ਅਚਲ uint8_t *) dst, (uint8_t *) dst, ne00);
}

ਸਥਿਰ ਅਚਲ softmax_compute_fn_t softmax_kernels[HTP_SOFTMAX_KERNEL_ਗਿਣਤੀ] = {
    [HTP_SOFTMAX_KERNEL_NOMASK]   = compute_fast_softmax_f32_nomask,
    [HTP_SOFTMAX_KERNEL_ਨਕਾਬ_F32] = compute_fast_softmax_f32_ਨਕਾਬ_f32,
    [HTP_SOFTMAX_KERNEL_ਨਕਾਬ_F16] = compute_fast_softmax_f32_ਨਕਾਬ_f16,
};

ਸਥਿਰ ਖਾਲੀ softmax_thread_dma(ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ nth, ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ ith, ਖਾਲੀ * ਅੰਕੜੇ) {
    (ਖਾਲੀ) nth;
    ਅਚਲ ਬਣਤਰ htp_softmax_context * smctx = (ਅਚਲ ਬਣਤਰ htp_softmax_context *) ਅੰਕੜੇ;
    ਬਣਤਰ htp_ops_context * octx = smctx->octx;
    ਅਚਲ ਬਣਤਰ htp_tensor * src0 = octx->src[0];
    ਅਚਲ ਬਣਤਰ htp_tensor * dst  = octx->dst;
    ਬਣਤਰ htp_thread_trace * tr = &octx->ctx->trace[ith];

    ਅਚਲ uint32_t src0_nrows            = smctx->nrows;
    ਅਚਲ uint32_t src0_nrows_ਪ੍ਰਤੀ_thread = smctx->src0_nrows_ਪ੍ਰਤੀ_thread;

    ਅਚਲ uint32_t src0_ਸ਼ੁਰੂ_ਪੰਕਤੀ = smctx->ਪੰਕਤੀ_ਸ਼ੁਰੂ + src0_nrows_ਪ੍ਰਤੀ_thread * ith;
    ਅਚਲ uint32_t src0_ਸਿਰਾ_ਪੰਕਤੀ   = MIN(src0_ਸ਼ੁਰੂ_ਪੰਕਤੀ + src0_nrows_ਪ੍ਰਤੀ_thread, smctx->ਪੰਕਤੀ_ਸ਼ੁਰੂ + src0_nrows);

    ਜੇ (src0_ਸ਼ੁਰੂ_ਪੰਕਤੀ >= src0_ਸਿਰਾ_ਪੰਕਤੀ) {
        ਵਾਪਸ;
    }

    ਅਚਲ dma_addr_t ਅੰਕੜੇ_src0 = smctx->ਅੰਕੜੇ_src0;
    ਅਚਲ dma_addr_t ਅੰਕੜੇ_dst  = smctx->ਅੰਕੜੇ_dst;

    ਅਚਲ size_t src0_ਪੰਕਤੀ_ਮਾਪ = src0->ne[0] * ਆਕਾਰ(ਦਸ਼ਮਲਵ);
    ਅਚਲ size_t dst_ਪੰਕਤੀ_ਮਾਪ  = src0->ne[0] * ਆਕਾਰ(ਦਸ਼ਮਲਵ);

    uint8_t * src0_vtcm_ਅਧਾਰ = smctx->vtcm_src0 + (ith * smctx->vtcm_src0_ਮਾਪ_ਪ੍ਰਤੀ_thread);
    uint8_t * dst_vtcm_ਅਧਾਰ  = smctx->vtcm_dst  + (ith * smctx->vtcm_dst_ਮਾਪ_ਪ੍ਰਤੀ_thread);

    ਅਚਲ size_t src0_vtcm_half = smctx->src0_spad_half_ਮਾਪ;
    ਅਚਲ size_t dst_vtcm_half  = smctx->dst_spad_half_ਮਾਪ;

    dma_ਕਤਾਰ * dma_q = octx->ctx->dma[ith];

    ਲਈ (uint32_t r = src0_ਸ਼ੁਰੂ_ਪੰਕਤੀ, idx = 0; r < src0_ਸਿਰਾ_ਪੰਕਤੀ && idx < 2; r++, idx++) {
        dma_addr_t cur_dst  = ਅੰਕੜੇ_dst  + r * dst_ਪੰਕਤੀ_ਮਾਪ;
        dma_addr_t cur_src0 = ਅੰਕੜੇ_src0 + r * src0_ਪੰਕਤੀ_ਮਾਪ;
        ਖਾਲੀ * d_spad = dst_vtcm_ਅਧਾਰ  + idx * dst_vtcm_half;
        ਖਾਲੀ * s_spad = src0_vtcm_ਅਧਾਰ + idx * src0_vtcm_half;

        dma_ਕਤਾਰ_ਧੱਕੋ(dma_q, dma_make_ਅੰਕੜੇ(cur_dst, d_spad),
                       dst_ਪੰਕਤੀ_ਮਾਪ, smctx->dst_ਪੰਕਤੀ_ਮਾਪ_aligned, dst_ਪੰਕਤੀ_ਮਾਪ, 0);
        dma_ਕਤਾਰ_ਧੱਕੋ(dma_q, dma_make_ਅੰਕੜੇ(s_spad, cur_src0),
                       smctx->src0_ਪੰਕਤੀ_ਮਾਪ_aligned, src0_ਪੰਕਤੀ_ਮਾਪ, src0_ਪੰਕਤੀ_ਮਾਪ, 1);
    }

    softmax_compute_fn_t compute = (softmax_compute_fn_t) smctx->compute;
    ਅਚਲ uint32_t ne00 = src0->ne[0];

    ਲਈ (uint32_t r = src0_ਸ਼ੁਰੂ_ਪੰਕਤੀ; r < src0_ਸਿਰਾ_ਪੰਕਤੀ; ++r) {
        ਖਾਲੀ * d_spad = (ਖਾਲੀ *) dma_ਕਤਾਰ_ਕੱਢੋ(dma_q).src;
        ਖਾਲੀ * s_spad = (ਖਾਲੀ *) dma_ਕਤਾਰ_ਕੱਢੋ(dma_q).dst;

        htp_trace_event_ਸ਼ੁਰੂ(tr, HTP_TRACE_EVT_HVX_COMP, r);
        compute(d_spad, s_spad, ਨਿਰਾ, ne00, smctx->scale, 1.0f);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, r);

        dma_addr_t cur_dst = ਅੰਕੜੇ_dst + r * dst_ਪੰਕਤੀ_ਮਾਪ;
        dma_ਕਤਾਰ_ਧੱਕੋ(dma_q, dma_make_ਅੰਕੜੇ(cur_dst, d_spad),
                       dst_ਪੰਕਤੀ_ਮਾਪ, smctx->dst_ਪੰਕਤੀ_ਮਾਪ_aligned, dst_ਪੰਕਤੀ_ਮਾਪ, 1);

        ਅਚਲ uint32_t ਅਗਲਾ_r = r + 2;
        ਜੇ (ਅਗਲਾ_r < src0_ਸਿਰਾ_ਪੰਕਤੀ) {
            dma_addr_t ਅਗਲਾ_src0 = ਅੰਕੜੇ_src0 + ਅਗਲਾ_r * src0_ਪੰਕਤੀ_ਮਾਪ;
            dma_ਕਤਾਰ_ਧੱਕੋ(dma_q, dma_make_ਅੰਕੜੇ(s_spad, ਅਗਲਾ_src0),
                           smctx->src0_ਪੰਕਤੀ_ਮਾਪ_aligned, src0_ਪੰਕਤੀ_ਮਾਪ, src0_ਪੰਕਤੀ_ਮਾਪ, 1);
        }
    }

    dma_ਕਤਾਰ_flush(dma_q);
}

ਸਥਿਰ ਖਾਲੀ softmax_thread_ਨਕਾਬ_dma(ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ nth, ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ ith, ਖਾਲੀ * ਅੰਕੜੇ) {
    (ਖਾਲੀ) nth;
    ਅਚਲ ਬਣਤਰ htp_softmax_context * smctx = (ਅਚਲ ਬਣਤਰ htp_softmax_context *) ਅੰਕੜੇ;
    ਬਣਤਰ htp_ops_context * octx = smctx->octx;
    ਅਚਲ ਬਣਤਰ htp_tensor * src0 = octx->src[0];
    ਅਚਲ ਬਣਤਰ htp_tensor * src1 = octx->src[1];
    ਅਚਲ ਬਣਤਰ htp_tensor * dst  = octx->dst;
    ਬਣਤਰ htp_thread_trace * tr = &octx->ctx->trace[ith];

    ਅਚਲ uint32_t src0_nrows            = smctx->nrows;
    ਅਚਲ uint32_t src0_nrows_ਪ੍ਰਤੀ_thread = smctx->src0_nrows_ਪ੍ਰਤੀ_thread;

    ਅਚਲ uint32_t src0_ਸ਼ੁਰੂ_ਪੰਕਤੀ = smctx->ਪੰਕਤੀ_ਸ਼ੁਰੂ + src0_nrows_ਪ੍ਰਤੀ_thread * ith;
    ਅਚਲ uint32_t src0_ਸਿਰਾ_ਪੰਕਤੀ   = MIN(src0_ਸ਼ੁਰੂ_ਪੰਕਤੀ + src0_nrows_ਪ੍ਰਤੀ_thread, smctx->ਪੰਕਤੀ_ਸ਼ੁਰੂ + src0_nrows);

    ਜੇ (src0_ਸ਼ੁਰੂ_ਪੰਕਤੀ >= src0_ਸਿਰਾ_ਪੰਕਤੀ) {
        ਵਾਪਸ;
    }

    ਅਚਲ dma_addr_t ਅੰਕੜੇ_src0 = smctx->ਅੰਕੜੇ_src0;
    ਅਚਲ dma_addr_t ਅੰਕੜੇ_src1 = smctx->ਅੰਕੜੇ_src1;
    ਅਚਲ dma_addr_t ਅੰਕੜੇ_dst  = smctx->ਅੰਕੜੇ_dst;

    ਅਚਲ size_t src0_ਪੰਕਤੀ_ਮਾਪ = src0->ne[0] * ਆਕਾਰ(ਦਸ਼ਮਲਵ);
    ਅਚਲ size_t dst_ਪੰਕਤੀ_ਮਾਪ  = src0->ne[0] * ਆਕਾਰ(ਦਸ਼ਮਲਵ);
    ਅਚਲ size_t ਨਕਾਬ_ਪੰਕਤੀ_ਮਾਪ = smctx->use_f16 ? (src1->ne[0] * ਆਕਾਰ(__fp16)) : (src1->ne[0] * ਆਕਾਰ(ਦਸ਼ਮਲਵ));

    uint8_t * src0_vtcm_ਅਧਾਰ = smctx->vtcm_src0 + (ith * smctx->vtcm_src0_ਮਾਪ_ਪ੍ਰਤੀ_thread);
    uint8_t * src1_vtcm_ਅਧਾਰ = smctx->vtcm_src1 + (ith * smctx->vtcm_src1_ਮਾਪ_ਪ੍ਰਤੀ_thread);
    uint8_t * dst_vtcm_ਅਧਾਰ  = smctx->vtcm_dst  + (ith * smctx->vtcm_dst_ਮਾਪ_ਪ੍ਰਤੀ_thread);

    ਅਚਲ size_t src0_vtcm_half = smctx->src0_spad_half_ਮਾਪ;
    ਅਚਲ size_t src1_vtcm_half = smctx->src1_spad_half_ਮਾਪ;
    ਅਚਲ size_t dst_vtcm_half  = smctx->dst_spad_half_ਮਾਪ;

    ਅਚਲ uint32_t nb11 = src1->nb[1];
    ਅਚਲ uint32_t nb12 = src1->nb[2];
    ਅਚਲ uint32_t nb13 = src1->nb[3];

    ਅਚਲ uint32_t ne00 = src0->ne[0];
    ਅਚਲ uint32_t ne01 = src0->ne[1];
    ਅਚਲ uint32_t ne02 = src0->ne[2];
    ਅਚਲ uint32_t ne03 = src0->ne[3];
    ਅਚਲ uint32_t ne12 = src1->ne[2];
    ਅਚਲ uint32_t ne13 = src1->ne[3];

    ਅਚਲ ਬਣਤਰ fastdiv_ਮੁੱਲਾਂ * div_ne01 = &smctx->div_ne01;
    ਅਚਲ ਬਣਤਰ fastdiv_ਮੁੱਲਾਂ * div_ne02 = &smctx->div_ne02;
    ਅਚਲ ਬਣਤਰ fastdiv_ਮੁੱਲਾਂ * div_ne12 = &smctx->div_ne12;
    ਅਚਲ ਬਣਤਰ fastdiv_ਮੁੱਲਾਂ * div_ne13 = &smctx->div_ne13;

    dma_ਕਤਾਰ * dma_q = octx->ctx->dma[ith];

    ਲਈ (uint32_t r = src0_ਸ਼ੁਰੂ_ਪੰਕਤੀ, idx = 0; r < src0_ਸਿਰਾ_ਪੰਕਤੀ && idx < 2; r++, idx++) {
        dma_addr_t cur_dst  = ਅੰਕੜੇ_dst  + r * dst_ਪੰਕਤੀ_ਮਾਪ;
        dma_addr_t cur_src0 = ਅੰਕੜੇ_src0 + r * src0_ਪੰਕਤੀ_ਮਾਪ;

        uint32_t i1 = fastmodulo(r, ne01, div_ne01);
        uint32_t r_div_ne01 = fastdiv(r, div_ne01);
        uint32_t i2 = fastmodulo(r_div_ne01, ne02, div_ne02);
        uint32_t i3 = fastdiv(r_div_ne01, div_ne02);
        uint32_t i12 = (ne12 == ne02) ? i2 : fastmodulo(i2, ne12, div_ne12);
        uint32_t i13 = (ne13 == ne03) ? i3 : fastmodulo(i3, ne13, div_ne13);
        dma_addr_t cur_src1 = ਅੰਕੜੇ_src1 + i1 * nb11 + i12 * nb12 + i13 * nb13;

        ਖਾਲੀ * d_spad = dst_vtcm_ਅਧਾਰ  + idx * dst_vtcm_half;
        ਖਾਲੀ * s_spad = src0_vtcm_ਅਧਾਰ + idx * src0_vtcm_half;
        ਖਾਲੀ * m_spad = src1_vtcm_ਅਧਾਰ + idx * src1_vtcm_half;

        dma_ਕਤਾਰ_ਧੱਕੋ(dma_q, dma_make_ਅੰਕੜੇ(cur_dst, d_spad),
                       dst_ਪੰਕਤੀ_ਮਾਪ, smctx->dst_ਪੰਕਤੀ_ਮਾਪ_aligned, dst_ਪੰਕਤੀ_ਮਾਪ, 0);
        dma_ਕਤਾਰ_ਧੱਕੋ(dma_q, dma_make_ਅੰਕੜੇ(s_spad, cur_src0),
                       smctx->src0_ਪੰਕਤੀ_ਮਾਪ_aligned, src0_ਪੰਕਤੀ_ਮਾਪ, src0_ਪੰਕਤੀ_ਮਾਪ, 1);
        dma_ਕਤਾਰ_ਧੱਕੋ(dma_q, dma_make_ਅੰਕੜੇ(m_spad, cur_src1),
                       smctx->src1_ਪੰਕਤੀ_ਮਾਪ_aligned, ਨਕਾਬ_ਪੰਕਤੀ_ਮਾਪ, ਨਕਾਬ_ਪੰਕਤੀ_ਮਾਪ, 1);
    }

    softmax_compute_fn_t compute = (softmax_compute_fn_t) smctx->compute;
    ਅਚਲ bool has_bias = smctx->max_bias > 0.0f;
    uint32_t prev_i2 = (uint32_t)-1;
    ਦਸ਼ਮਲਵ ਢਲਾਨ = 1.0f;

    ਲਈ (uint32_t r = src0_ਸ਼ੁਰੂ_ਪੰਕਤੀ; r < src0_ਸਿਰਾ_ਪੰਕਤੀ; ++r) {
        ਖਾਲੀ * d_spad = (ਖਾਲੀ *) (uintptr_t) dma_ਕਤਾਰ_ਕੱਢੋ(dma_q).src;
        ਖਾਲੀ * s_spad = (ਖਾਲੀ *) (uintptr_t) dma_ਕਤਾਰ_ਕੱਢੋ(dma_q).dst;
        ਖਾਲੀ * m_spad = (ਖਾਲੀ *) (uintptr_t) dma_ਕਤਾਰ_ਕੱਢੋ(dma_q).dst;

        ਜੇ (has_bias) {
            uint32_t r_div_ne01 = fastdiv(r, div_ne01);
            uint32_t i2 = fastmodulo(r_div_ne01, ne02, div_ne02);
            ਜੇ (i2 != prev_i2) {
                ਢਲਾਨ = smctx->slopes[i2];
                prev_i2 = i2;
            }
        }

        htp_trace_event_ਸ਼ੁਰੂ(tr, HTP_TRACE_EVT_HVX_COMP, r);
        compute(d_spad, s_spad, m_spad, ne00, smctx->scale, ਢਲਾਨ);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, r);

        dma_addr_t cur_dst = ਅੰਕੜੇ_dst + r * dst_ਪੰਕਤੀ_ਮਾਪ;
        dma_ਕਤਾਰ_ਧੱਕੋ(dma_q, dma_make_ਅੰਕੜੇ(cur_dst, d_spad),
                       dst_ਪੰਕਤੀ_ਮਾਪ, smctx->dst_ਪੰਕਤੀ_ਮਾਪ_aligned, dst_ਪੰਕਤੀ_ਮਾਪ, 1);

        ਅਚਲ uint32_t ਅਗਲਾ_r = r + 2;
        ਜੇ (ਅਗਲਾ_r < src0_ਸਿਰਾ_ਪੰਕਤੀ) {
            dma_addr_t ਅਗਲਾ_src0 = ਅੰਕੜੇ_src0 + ਅਗਲਾ_r * src0_ਪੰਕਤੀ_ਮਾਪ;

            uint32_t ni1 = fastmodulo(ਅਗਲਾ_r, ne01, div_ne01);
            uint32_t nr_div_ne01 = fastdiv(ਅਗਲਾ_r, div_ne01);
            uint32_t ni2 = fastmodulo(nr_div_ne01, ne02, div_ne02);
            uint32_t ni3 = fastdiv(nr_div_ne01, div_ne02);
            uint32_t ni12 = (ne12 == ne02) ? ni2 : fastmodulo(ni2, ne12, div_ne12);
            uint32_t ni13 = (ne13 == ne03) ? ni3 : fastmodulo(ni3, ne13, div_ne13);
            dma_addr_t ਅਗਲਾ_src1 = ਅੰਕੜੇ_src1 + ni1 * nb11 + ni12 * nb12 + ni13 * nb13;

            dma_ਕਤਾਰ_ਧੱਕੋ(dma_q, dma_make_ਅੰਕੜੇ(s_spad, ਅਗਲਾ_src0),
                           smctx->src0_ਪੰਕਤੀ_ਮਾਪ_aligned, src0_ਪੰਕਤੀ_ਮਾਪ, src0_ਪੰਕਤੀ_ਮਾਪ, 1);
            dma_ਕਤਾਰ_ਧੱਕੋ(dma_q, dma_make_ਅੰਕੜੇ(m_spad, ਅਗਲਾ_src1),
                           smctx->src1_ਪੰਕਤੀ_ਮਾਪ_aligned, ਨਕਾਬ_ਪੰਕਤੀ_ਮਾਪ, ਨਕਾਬ_ਪੰਕਤੀ_ਮਾਪ, 1);
        }
    }

    dma_ਕਤਾਰ_flush(dma_q);
}

ਸਥਿਰ ਪੂਰਨ execute_op_softmax_f32(ਬਣਤਰ htp_ops_context * octx) {
    ਅਚਲ ਬਣਤਰ htp_tensor * src0 = octx->src[0];
    ਅਚਲ ਬਣਤਰ htp_tensor * dst  = octx->dst;

    ਅਚਲ ਅੱਖਰ * op_type = "softmax-f32";

    ਅਚਲ ਬਣਤਰ htp_softmax_kernel_params * kparams =
        (ਅਚਲ ਬਣਤਰ htp_softmax_kernel_params *) octx->kernel_params;

    ਜੇ (!htp_ops_context_ਸੈੱਟ_n_ਧਾਗੇ(octx, kparams->n_ਧਾਗੇ)) {
        ਵਾਪਸ HTP_ਸਥਿਤੀ_INVAL_PARAMS;
    }

    ਜੇ (kparams->kernel_id >= HTP_SOFTMAX_KERNEL_ਗਿਣਤੀ) {
        ਵਾਪਸ HTP_ਸਥਿਤੀ_INVAL_PARAMS;
    }

    ਜੇ (octx->ctx->vtcm_ਮਾਪ < (size_t) kparams->vtcm_ਮਾਪ) {
        FARF(ERROR, "%s : current VTCM reservation %zu is too small, needed %u\n",
             op_type, octx->ctx->vtcm_ਮਾਪ, kparams->vtcm_ਮਾਪ);
        ਵਾਪਸ HTP_ਸਥਿਤੀ_VTCM_TOO_SMALL;
    }

    ਅਚਲ uint32_t src0_nrows = src0->ne[1] * src0->ne[2] * src0->ne[3];
    ਅਚਲ size_t elem_ਮਾਪ    = ਆਕਾਰ(ਦਸ਼ਮਲਵ);
    ਅਚਲ size_t dst_ਪੰਕਤੀ_ਮਾਪ = dst->nb[1];

    uint32_t ਪੰਕਤੀ_ਸ਼ੁਰੂ = 0;
    uint32_t nrows     = src0_nrows;

    ਜੇ (octx->ctx->mdev.ਗਿਣਤੀ_2 > 1) {
        uint32_t ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_chunk = 0;
        htp_tensor_mdev_ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_chunk(dst, (uint32_t) elem_ਮਾਪ, (uint32_t) dst_ਪੰਕਤੀ_ਮਾਪ, &ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_chunk);
        ਅਚਲ ਬਣਤਰ htp_tensor_mdev_range range = htp_tensor_mdev_ਹਿੱਸੇ_ਕਰੋ(
            src0_nrows, ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_chunk, octx->ctx->mdev.idx, octx->ctx->mdev.ਗਿਣਤੀ_2, &octx->ctx->mdev.ਗਿਣਤੀ_div);
        ਪੰਕਤੀ_ਸ਼ੁਰੂ = range.ਸ਼ੁਰੂ;
        nrows     = range.ਗਿਣਤੀ_2;
        ਜੇ (nrows < octx->n_ਧਾਗੇ) {
            htp_ops_context_ਸੈੱਟ_n_ਧਾਗੇ(octx, nrows ? nrows : 1);
        }
    }

    ਜੇ (nrows == 0) {
        ਵਾਪਸ HTP_ਸਥਿਤੀ_OK;
    }

    ਅਚਲ uint32_t n_ਧਾਗੇ = octx->n_ਧਾਗੇ;
    uint8_t * ਅਚਲ vtcm_ਅਧਾਰ = (uint8_t *) octx->ctx->vtcm_ਅਧਾਰ;

    ਅਚਲ uint32_t off_src0 = 0;
    ਅਚਲ uint32_t off_dst  = off_src0 + kparams->vtcm_src0_ਮਾਪ_ਪ੍ਰਤੀ_thread * kparams->n_ਧਾਗੇ;
    ਅਚਲ uint32_t off_src1 = off_dst  + kparams->vtcm_dst_ਮਾਪ_ਪ੍ਰਤੀ_thread  * kparams->n_ਧਾਗੇ;

    ਬਣਤਰ htp_softmax_context smctx = {
        .octx                  = octx,
        .kparams               = kparams,
        .compute               = (ਖਾਲੀ *) softmax_kernels[kparams->kernel_id],

        .ਅੰਕੜੇ_src0             = src0->ਅੰਕੜੇ,
        .ਅੰਕੜੇ_src1             = kparams->use_src1 ? octx->src[1]->ਅੰਕੜੇ : 0,
        .ਅੰਕੜੇ_dst              = dst->ਅੰਕੜੇ,

        .vtcm_src0             = VTCM_LAYOUT_PTR(uint8_t, vtcm_ਅਧਾਰ, off_src0),
        .vtcm_dst              = VTCM_LAYOUT_PTR(uint8_t, vtcm_ਅਧਾਰ, off_dst),
        .vtcm_src1             = VTCM_LAYOUT_PTR_OPTIONAL(uint8_t, vtcm_ਅਧਾਰ, off_src1, kparams->use_src1),

        .vtcm_src0_ਮਾਪ_ਪ੍ਰਤੀ_thread = kparams->vtcm_src0_ਮਾਪ_ਪ੍ਰਤੀ_thread,
        .vtcm_src1_ਮਾਪ_ਪ੍ਰਤੀ_thread = kparams->vtcm_src1_ਮਾਪ_ਪ੍ਰਤੀ_thread,
        .vtcm_dst_ਮਾਪ_ਪ੍ਰਤੀ_thread  = kparams->vtcm_dst_ਮਾਪ_ਪ੍ਰਤੀ_thread,

        .src0_spad_half_ਮਾਪ   = kparams->src0_spad_half_ਮਾਪ,
        .src1_spad_half_ਮਾਪ   = kparams->src1_spad_half_ਮਾਪ,
        .dst_spad_half_ਮਾਪ    = kparams->dst_spad_half_ਮਾਪ,

        .src0_ਪੰਕਤੀ_ਮਾਪ_aligned = kparams->src0_ਪੰਕਤੀ_ਮਾਪ_aligned,
        .src1_ਪੰਕਤੀ_ਮਾਪ_aligned = kparams->src1_ਪੰਕਤੀ_ਮਾਪ_aligned,
        .dst_ਪੰਕਤੀ_ਮਾਪ_aligned  = kparams->dst_ਪੰਕਤੀ_ਮਾਪ_aligned,

        .use_f16               = kparams->use_f16 != 0,

        .n_ਸਿਰ                = kparams->n_ਸਿਰ,
        .n_ਸਿਰ_log2           = kparams->n_ਸਿਰ_log2,

        .scale                 = kparams->scale,
        .max_bias              = kparams->max_bias,
        .m0                    = kparams->m0,
        .m1                    = kparams->m1,

        .div_ne01              = kparams->div_ne01,
        .div_ne02              = kparams->div_ne02,
        .div_ne12              = kparams->div_ne12,
        .div_ne13              = kparams->div_ne13,

        .src0_nrows_ਪ੍ਰਤੀ_thread = fastdiv(nrows + n_ਧਾਗੇ - 1, &octx->n_ਧਾਗੇ_div),
        .ਪੰਕਤੀ_ਸ਼ੁਰੂ             = ਪੰਕਤੀ_ਸ਼ੁਰੂ,
        .nrows                 = nrows,
    };

    ਜੇ (kparams->max_bias > 0.0f && kparams->use_src1) {
        ਜੇ (kparams->n_ਸਿਰ > 512) {
            ਵਾਪਸ HTP_ਸਥਿਤੀ_INVAL_PARAMS;
        }
        ਲਈ (uint32_t h = 0; h < kparams->n_ਸਿਰ; h += 32) {
            HVX_ਸਦਿਸ਼ v_slopes = hvx_alibi_slopes(h, 1, kparams->n_ਸਿਰ_log2, kparams->m0, kparams->m1);
            hvx_vmem(&smctx.slopes[h]) = v_slopes;
        }
    }

    work_ਕਤਾਰ_func_t task_func = kparams->use_src1 ? softmax_thread_ਨਕਾਬ_dma : softmax_thread_dma;
    work_ਕਤਾਰ_ਲੜੀਵਾਰ(octx->ctx->work_ਕਤਾਰ, task_func, &smctx, n_ਧਾਗੇ);

    ਵਾਪਸ HTP_ਸਥਿਤੀ_OK;
}

ਪੂਰਨ op_softmax(ਬਣਤਰ htp_ops_context * octx) {
    ਪੂਰਨ err = HTP_ਸਥਿਤੀ_OK;

    ਚੋਣ (octx->src[0]->type) {
        ਹਾਲਤ HTP_TYPE_F32:
            err = execute_op_softmax_f32(octx);
            ਰੋਕੋ;

        ਮੂਲ:
            err = HTP_ਸਥਿਤੀ_NO_SUPPORT;
            ਰੋਕੋ;
    }

    ਵਾਪਸ err;
}
