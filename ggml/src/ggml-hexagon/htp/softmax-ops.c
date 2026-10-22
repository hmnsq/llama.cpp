#ಸೂಚನೆ clang diagnostic ignored "-Wunused-variable"
#ಸೂಚನೆ clang diagnostic ignored "-Wunused-function"
#ಸೂಚನೆ clang diagnostic ignored "-Wunused-but-set-variable"

#ಅನ್ತರ್ಭಾವ <HAP_farf.h>
#ಅನ್ತರ್ಭಾವ <HAP_perf.h>

#ಅನ್ತರ್ಭಾವ <math.h>
#ಅನ್ತರ್ಭಾವ <string.h>

#ಅನ್ತರ್ಭಾವ "dma-queue.h"
#ಅನ್ತರ್ಭಾವ "work-queue.h"
#ಅನ್ತರ್ಭಾವ "hvx-utils.h"
#ಅನ್ತರ್ಭಾವ "hex-fastdiv.h"
#ಅನ್ತರ್ಭಾವ "hex-common.h"
#ಅನ್ತರ್ಭಾವ "hex-profile.h"

#ವ್ಯಾಖ್ಯಾನಿಸು GGML_COMMON_DECL_C
#ಅನ್ತರ್ಭಾವ "ggml-common.h"
#ಅನ್ತರ್ಭಾವ "htp-ctx.h"
#ಅನ್ತರ್ಭಾವ "htp-ops.h"
#ಅನ್ತರ್ಭಾವ "htp-tensor.h"
#ಅನ್ತರ್ಭಾವ "htp-vtcm.h"
#ಅನ್ತರ್ಭಾವ "htp/softmax-ops.h"
#ಅನ್ತರ್ಭಾವ "hvx-flash-attn.h"

ರಚನೆ htp_softmax_context {
    ರಚನೆ htp_ops_context * octx;
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_softmax_kernel_params * kparams;

    ಶೂನ್ಯ *   compute;

    dma_addr_t ದತ್ತಾಂಶ_src0;
    dma_addr_t ದತ್ತಾಂಶ_src1;
    dma_addr_t ದತ್ತಾಂಶ_dst;

    uint8_t * vtcm_src0;
    uint8_t * vtcm_src1;
    uint8_t * vtcm_dst;

    uint32_t  vtcm_src0_ಗಾತ್ರ_ಪ್ರತಿ_thread;
    uint32_t  vtcm_src1_ಗಾತ್ರ_ಪ್ರತಿ_thread;
    uint32_t  vtcm_dst_ಗಾತ್ರ_ಪ್ರತಿ_thread;

    uint32_t  src0_spad_half_ಗಾತ್ರ;
    uint32_t  src1_spad_half_ಗಾತ್ರ;
    uint32_t  dst_spad_half_ಗಾತ್ರ;

    uint32_t  src0_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned;
    uint32_t  src1_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned;
    uint32_t  dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned;

    bool     use_f16;

    uint32_t n_ತಲೆ;
    uint32_t n_ತಲೆ_log2;

    ದಶಮಾಂಶ    scale;
    ದಶಮಾಂಶ    max_bias;
    ದಶಮಾಂಶ    m0;
    ದಶಮಾಂಶ    m1;

    ರಚನೆ fastdiv_ಮೌಲ್ಯಗಳು div_ne01;
    ರಚನೆ fastdiv_ಮೌಲ್ಯಗಳು div_ne02;
    ರಚನೆ fastdiv_ಮೌಲ್ಯಗಳು div_ne12;
    ರಚನೆ fastdiv_ಮೌಲ್ಯಗಳು div_ne13;

    uint32_t src0_nrows_ಪ್ರತಿ_thread;
    uint32_t ಅಡ್ಡಸಾಲು_ಆರಂಭ;
    uint32_t nrows;

    ದಶಮಾಂಶ    slopes[512] __attribute__((aligned(128)));
};

ಪ್ರಕಾರವ್ಯಾಖ್ಯೆ ಶೂನ್ಯ (*softmax_compute_fn_t)(
    ಶೂನ್ಯ * ಸೀಮಿತ dst,
    ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * ಸೀಮಿತ src0,
    ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * ಸೀಮಿತ ಮುಖವಾಡ,
    uint32_t ne00,
    ದಶಮಾಂಶ scale,
    ದಶಮಾಂಶ ಇಳಿಜಾರು
);

ಸ್ಥಿರ ಶೂನ್ಯ hvx_fast_softmax_prep_f16(ಸ್ಥಿರಾಂಕ uint8_t * ಸೀಮಿತ src,
                                      uint8_t * ಸೀಮಿತ dst,
                                      ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಅಂಕೆ_elems,
                                      ದಶಮಾಂಶ     scale,
                                      ಸ್ಥಿರಾಂಕ uint8_t * ಸೀಮಿತ ಮುಖವಾಡ,
                                      ದಶಮಾಂಶ ಇಳಿಜಾರು) {
    ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ * ಸೀಮಿತ v_src  = (ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ *) src;
    HVX_ಸದಿಶ * ಸೀಮಿತ v_dst        = (HVX_ಸದಿಶ *) dst;
    ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ * ಸೀಮಿತ v_ಮುಖವಾಡ = (ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ *) ಮುಖವಾಡ;

    HVX_ಸದಿಶ scale_vec = hvx_vec_splat_f32(scale);
    HVX_ಸದಿಶ ಇಳಿಜಾರು_vec = hvx_vec_splat_f32(ಇಳಿಜಾರು);

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nvec_64 = ಅಂಕೆ_elems / VLEN_FP16;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nloe_64 = ಅಂಕೆ_elems % VLEN_FP16;

    #ಸೂಚನೆ unroll(2)
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < nvec_64; i++) {
        HVX_ಸದಿಶ_Pair p = hvx_vec_f16_ಗಮ್ಯ_f32(v_ಮುಖವಾಡ[i]);
        HVX_ಸದಿಶ m0 = Q6_V_lo_W(p);
        HVX_ಸದಿಶ m1 = Q6_V_hi_W(p);

        HVX_ಸದಿಶ s0 = v_src[2 * i];
        HVX_ಸದಿಶ s1 = v_src[2 * i + 1];

        HVX_ಸದಿಶ v0 = Q6_Vqf32_vadd_Vqf32Vqf32(Q6_Vqf32_vmpy_VsfVsf(s0, scale_vec), Q6_Vqf32_vmpy_VsfVsf(m0, ಇಳಿಜಾರು_vec));
        HVX_ಸದಿಶ v1 = Q6_Vqf32_vadd_Vqf32Vqf32(Q6_Vqf32_vmpy_VsfVsf(s1, scale_vec), Q6_Vqf32_vmpy_VsfVsf(m1, ಇಳಿಜಾರು_vec));

        v_dst[2 * i]     = Q6_Vsf_equals_Vqf32(v0);
        v_dst[2 * i + 1] = Q6_Vsf_equals_Vqf32(v1);
    }

    ಆದರೆ (nloe_64 > 0) {
        HVX_ಸದಿಶ_Pair p = hvx_vec_f16_ಗಮ್ಯ_f32(v_ಮುಖವಾಡ[nvec_64]);
        HVX_ಸದಿಶ m0 = Q6_V_lo_W(p);

        HVX_ಸದಿಶ s0 = v_src[2 * nvec_64];
        HVX_ಸದಿಶ v0 = Q6_Vqf32_vadd_Vqf32Vqf32(Q6_Vqf32_vmpy_VsfVsf(s0, scale_vec), Q6_Vqf32_vmpy_VsfVsf(m0, ಇಳಿಜಾರು_vec));

        ಆದರೆ (nloe_64 <= VLEN_FP32) {
            hvx_vec_store_a(&v_dst[2 * nvec_64], nloe_64 * ಗಾತ್ರ(ದಶಮಾಂಶ), Q6_Vsf_equals_Vqf32(v0));
        } ಇಲ್ಲದಿದ್ದರೆ {
            v_dst[2 * nvec_64] = Q6_Vsf_equals_Vqf32(v0);

            HVX_ಸದಿಶ m1 = Q6_V_hi_W(p);
            HVX_ಸದಿಶ s1 = v_src[2 * nvec_64 + 1];
            HVX_ಸದಿಶ v1 = Q6_Vqf32_vadd_Vqf32Vqf32(Q6_Vqf32_vmpy_VsfVsf(s1, scale_vec), Q6_Vqf32_vmpy_VsfVsf(m1, ಇಳಿಜಾರು_vec));

            hvx_vec_store_a(&v_dst[2 * nvec_64 + 1], (nloe_64 - VLEN_FP32) * ಗಾತ್ರ(ದಶಮಾಂಶ), Q6_Vsf_equals_Vqf32(v1));
        }
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ hvx_fast_softmax_prep_f32(ಸ್ಥಿರಾಂಕ uint8_t * ಸೀಮಿತ src,
                                      uint8_t * ಸೀಮಿತ dst,
                                      ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಅಂಕೆ_elems,
                                      ದಶಮಾಂಶ     scale,
                                      ಸ್ಥಿರಾಂಕ uint8_t * ಸೀಮಿತ ಮುಖವಾಡ,
                                      ದಶಮಾಂಶ ಇಳಿಜಾರು) {
    ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ * ಸೀಮಿತ v_src  = (ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ *) src;
    HVX_ಸದಿಶ * ಸೀಮಿತ v_dst        = (HVX_ಸದಿಶ *) dst;
    ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ * ಸೀಮಿತ v_ಮುಖವಾಡ = (ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ *) ಮುಖವಾಡ;

    HVX_ಸದಿಶ scale_vec = hvx_vec_splat_f32(scale);
    HVX_ಸದಿಶ ಇಳಿಜಾರು_vec = hvx_vec_splat_f32(ಇಳಿಜಾರು);

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nvec = ಅಂಕೆ_elems / VLEN_FP32;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nloe = ಅಂಕೆ_elems % VLEN_FP32;

    #ಸೂಚನೆ unroll(4)
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < nvec; i++) {
        HVX_ಸದಿಶ v1 = v_src[i];
        HVX_ಸದಿಶ v3 = v_ಮುಖವಾಡ[i];

        HVX_ಸದಿಶ v2 = Q6_Vqf32_vmpy_VsfVsf(v1, scale_vec);
        HVX_ಸದಿಶ v4 = Q6_Vqf32_vmpy_VsfVsf(v3, ಇಳಿಜಾರು_vec);
        HVX_ಸದಿಶ v5 = Q6_Vqf32_vadd_Vqf32Vqf32(v2, v4);

        v_dst[i] = Q6_Vsf_equals_Vqf32(v5);
    }

    ಆದರೆ (nloe > 0) {
        HVX_ಸದಿಶ v1 = v_src[nvec];
        HVX_ಸದಿಶ v3 = v_ಮುಖವಾಡ[nvec];

        HVX_ಸದಿಶ v2 = Q6_Vqf32_vmpy_VsfVsf(v1, scale_vec);
        HVX_ಸದಿಶ v4 = Q6_Vqf32_vmpy_VsfVsf(v3, ಇಳಿಜಾರು_vec);
        HVX_ಸದಿಶ v5 = Q6_Vqf32_vadd_Vqf32Vqf32(v2, v4);

        hvx_vec_store_a(&v_dst[nvec], nloe * ಗಾತ್ರ(ದಶಮಾಂಶ), Q6_Vsf_equals_Vqf32(v5));
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ hvx_fast_softmax_f32(ಸ್ಥಿರಾಂಕ uint8_t * ಸೀಮಿತ src, uint8_t * ಸೀಮಿತ dst, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಅಂಕೆ_elems) {
    ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ * ಸೀಮಿತ v_src = (ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ *) src;
    HVX_ಸದಿಶ * ಸೀಮಿತ v_dst       = (HVX_ಸದಿಶ *) dst;

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nvec = ಅಂಕೆ_elems / VLEN_FP32;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nloe = ಅಂಕೆ_elems % VLEN_FP32;

    HVX_ಸದಿಶ max_vec = hvx_vec_splat_f32(((ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) src)[0]);

    #ಸೂಚನೆ unroll(2)
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < nvec; i++) {
        HVX_ಸದಿಶ v1 = v_src[i];
        max_vec       = Q6_Vsf_vmax_VsfVsf(max_vec, v1);
    }

    ಆದರೆ (nloe > 0) {
        HVX_ಸದಿಶ_Pred q_ಮುಖವಾಡ = Q6_Q_vsetq_R(nloe * ಗಾತ್ರ(ದಶಮಾಂಶ));
        HVX_ಸದಿಶ neg_inf    = hvx_vec_splat_f32(-INFINITY);
        HVX_ಸದಿಶ v_ಬಾಲ     = Q6_V_vmux_QVV(q_ಮುಖವಾಡ, v_src[nvec], neg_inf);
        max_vec               = Q6_Vsf_vmax_VsfVsf(max_vec, v_ಬಾಲ);
    }

    max_vec = hvx_vec_reduce_max_f32(max_vec);

    HVX_ಸದಿಶ ಮೊತ್ತ_vec = Q6_V_vsplat_R(0x00000000);

    #ಸೂಚನೆ unroll(2)
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < nvec; i++) {
        HVX_ಸದಿಶ v1 = v_src[i];
        HVX_ಸದಿಶ v2 = Q6_Vqf32_vsub_VsfVsf(v1, max_vec);

        HVX_ಸದಿಶ v3 = hvx_vec_exp_f32(Q6_Vsf_equals_Vqf32(v2));

        ಮೊತ್ತ_vec = Q6_Vqf32_vadd_VsfVsf(Q6_Vsf_equals_Vqf32(ಮೊತ್ತ_vec), v3);

        v_dst[i] = v3;
    }

    ಆದರೆ (nloe > 0) {
        HVX_ಸದಿಶ_Pred q_ಮುಖವಾಡ = Q6_Q_vsetq_R(nloe * ಗಾತ್ರ(ದಶಮಾಂಶ));
        HVX_ಸದಿಶ v1     = v_src[nvec];
        HVX_ಸದಿಶ v2     = Q6_Vqf32_vsub_VsfVsf(v1, max_vec);
        HVX_ಸದಿಶ v3     = hvx_vec_exp_f32(Q6_Vsf_equals_Vqf32(v2));
        HVX_ಸದಿಶ v3_pad = Q6_V_vmux_QVV(q_ಮುಖವಾಡ, v3, Q6_V_vzero());

        ಮೊತ್ತ_vec     = Q6_Vqf32_vadd_VsfVsf(Q6_Vsf_equals_Vqf32(ಮೊತ್ತ_vec), v3_pad);
        v_dst[nvec] = v3_pad;
    }

    ಮೊತ್ತ_vec = hvx_vec_reduce_ಮೊತ್ತ_f32(Q6_Vsf_equals_Vqf32(ಮೊತ್ತ_vec));

    HVX_ಸದಿಶ_Pred pos_ಮೊತ್ತ   = Q6_Q_vcmp_gt_VwVw(ಮೊತ್ತ_vec, Q6_V_vzero());
    HVX_ಸದಿಶ     v4        = hvx_vec_inverse_f32(ಮೊತ್ತ_vec);
    HVX_ಸದಿಶ     scale_vec = Q6_V_vmux_QVV(pos_ಮೊತ್ತ, v4, hvx_vec_splat_f32(1.0f));

    #ಸೂಚನೆ unroll(2)
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < nvec; i++) {
        HVX_ಸದಿಶ v1 = v_dst[i];
        HVX_ಸದಿಶ v2 = Q6_Vqf32_vmpy_VsfVsf(v1, scale_vec);
        v_dst[i]      = Q6_Vsf_equals_Vqf32(v2);
    }

    ಆದರೆ (nloe > 0) {
        HVX_ಸದಿಶ v1 = v_dst[nvec];
        HVX_ಸದಿಶ v2 = Q6_Vqf32_vmpy_VsfVsf(v1, scale_vec);
        hvx_vec_store_a(&v_dst[nvec], nloe * ಗಾತ್ರ(ದಶಮಾಂಶ), Q6_Vsf_equals_Vqf32(v2));
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ compute_fast_softmax_f32_nomask(
    ಶೂನ್ಯ * ಸೀಮಿತ dst,
    ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * ಸೀಮಿತ src0,
    ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * ಸೀಮಿತ ಮುಖವಾಡ,
    uint32_t ne00,
    ದಶಮಾಂಶ scale,
    ದಶಮಾಂಶ ಇಳಿಜಾರು
) {
    (ಶೂನ್ಯ) ಮುಖವಾಡ;
    (ಶೂನ್ಯ) ಇಳಿಜಾರು;
    hvx_scale_f32((uint8_t *) dst, (ಸ್ಥಿರಾಂಕ uint8_t *) src0, ne00, scale);
    hvx_fast_softmax_f32((ಸ್ಥಿರಾಂಕ uint8_t *) dst, (uint8_t *) dst, ne00);
}

ಸ್ಥಿರ ಶೂನ್ಯ compute_fast_softmax_f32_ಮುಖವಾಡ_f32(
    ಶೂನ್ಯ * ಸೀಮಿತ dst,
    ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * ಸೀಮಿತ src0,
    ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * ಸೀಮಿತ ಮುಖವಾಡ,
    uint32_t ne00,
    ದಶಮಾಂಶ scale,
    ದಶಮಾಂಶ ಇಳಿಜಾರು
) {
    hvx_fast_softmax_prep_f32((ಸ್ಥಿರಾಂಕ uint8_t *) src0, (uint8_t *) dst, ne00, scale, (ಸ್ಥಿರಾಂಕ uint8_t *) ಮುಖವಾಡ, ಇಳಿಜಾರು);
    hvx_fast_softmax_f32((ಸ್ಥಿರಾಂಕ uint8_t *) dst, (uint8_t *) dst, ne00);
}

ಸ್ಥಿರ ಶೂನ್ಯ compute_fast_softmax_f32_ಮುಖವಾಡ_f16(
    ಶೂನ್ಯ * ಸೀಮಿತ dst,
    ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * ಸೀಮಿತ src0,
    ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * ಸೀಮಿತ ಮುಖವಾಡ,
    uint32_t ne00,
    ದಶಮಾಂಶ scale,
    ದಶಮಾಂಶ ಇಳಿಜಾರು
) {
    hvx_fast_softmax_prep_f16((ಸ್ಥಿರಾಂಕ uint8_t *) src0, (uint8_t *) dst, ne00, scale, (ಸ್ಥಿರಾಂಕ uint8_t *) ಮುಖವಾಡ, ಇಳಿಜಾರು);
    hvx_fast_softmax_f32((ಸ್ಥಿರಾಂಕ uint8_t *) dst, (uint8_t *) dst, ne00);
}

ಸ್ಥಿರ ಸ್ಥಿರಾಂಕ softmax_compute_fn_t softmax_kernels[HTP_SOFTMAX_KERNEL_ಎಣಿಕೆ] = {
    [HTP_SOFTMAX_KERNEL_NOMASK]   = compute_fast_softmax_f32_nomask,
    [HTP_SOFTMAX_KERNEL_ಮುಖವಾಡ_F32] = compute_fast_softmax_f32_ಮುಖವಾಡ_f32,
    [HTP_SOFTMAX_KERNEL_ಮುಖವಾಡ_F16] = compute_fast_softmax_f32_ಮುಖವಾಡ_f16,
};

ಸ್ಥಿರ ಶೂನ್ಯ softmax_thread_dma(ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ nth, ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ ith, ಶೂನ್ಯ * ದತ್ತಾಂಶ) {
    (ಶೂನ್ಯ) nth;
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_softmax_context * smctx = (ಸ್ಥಿರಾಂಕ ರಚನೆ htp_softmax_context *) ದತ್ತಾಂಶ;
    ರಚನೆ htp_ops_context * octx = smctx->octx;
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * src0 = octx->src[0];
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * dst  = octx->dst;
    ರಚನೆ htp_thread_trace * tr = &octx->ctx->trace[ith];

    ಸ್ಥಿರಾಂಕ uint32_t src0_nrows            = smctx->nrows;
    ಸ್ಥಿರಾಂಕ uint32_t src0_nrows_ಪ್ರತಿ_thread = smctx->src0_nrows_ಪ್ರತಿ_thread;

    ಸ್ಥಿರಾಂಕ uint32_t src0_ಆರಂಭ_ಅಡ್ಡಸಾಲು = smctx->ಅಡ್ಡಸಾಲು_ಆರಂಭ + src0_nrows_ಪ್ರತಿ_thread * ith;
    ಸ್ಥಿರಾಂಕ uint32_t src0_ಕೊನೆ_ಅಡ್ಡಸಾಲು   = MIN(src0_ಆರಂಭ_ಅಡ್ಡಸಾಲು + src0_nrows_ಪ್ರತಿ_thread, smctx->ಅಡ್ಡಸಾಲು_ಆರಂಭ + src0_nrows);

    ಆದರೆ (src0_ಆರಂಭ_ಅಡ್ಡಸಾಲು >= src0_ಕೊನೆ_ಅಡ್ಡಸಾಲು) {
        ಹಿಂತಿರುಗಿ;
    }

    ಸ್ಥಿರಾಂಕ dma_addr_t ದತ್ತಾಂಶ_src0 = smctx->ದತ್ತಾಂಶ_src0;
    ಸ್ಥಿರಾಂಕ dma_addr_t ದತ್ತಾಂಶ_dst  = smctx->ದತ್ತಾಂಶ_dst;

    ಸ್ಥಿರಾಂಕ size_t src0_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ = src0->ne[0] * ಗಾತ್ರ(ದಶಮಾಂಶ);
    ಸ್ಥಿರಾಂಕ size_t dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ  = src0->ne[0] * ಗಾತ್ರ(ದಶಮಾಂಶ);

    uint8_t * src0_vtcm_ತಳ = smctx->vtcm_src0 + (ith * smctx->vtcm_src0_ಗಾತ್ರ_ಪ್ರತಿ_thread);
    uint8_t * dst_vtcm_ತಳ  = smctx->vtcm_dst  + (ith * smctx->vtcm_dst_ಗಾತ್ರ_ಪ್ರತಿ_thread);

    ಸ್ಥಿರಾಂಕ size_t src0_vtcm_half = smctx->src0_spad_half_ಗಾತ್ರ;
    ಸ್ಥಿರಾಂಕ size_t dst_vtcm_half  = smctx->dst_spad_half_ಗಾತ್ರ;

    dma_ಸರತಿ * dma_q = octx->ctx->dma[ith];

    ಗಾಗಿ (uint32_t r = src0_ಆರಂಭ_ಅಡ್ಡಸಾಲು, idx = 0; r < src0_ಕೊನೆ_ಅಡ್ಡಸಾಲು && idx < 2; r++, idx++) {
        dma_addr_t cur_dst  = ದತ್ತಾಂಶ_dst  + r * dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ;
        dma_addr_t cur_src0 = ದತ್ತಾಂಶ_src0 + r * src0_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ;
        ಶೂನ್ಯ * d_spad = dst_vtcm_ತಳ  + idx * dst_vtcm_half;
        ಶೂನ್ಯ * s_spad = src0_vtcm_ತಳ + idx * src0_vtcm_half;

        dma_ಸರತಿ_ತಳ್ಳು(dma_q, dma_make_ದತ್ತಾಂಶ(cur_dst, d_spad),
                       dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ, smctx->dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned, dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ, 0);
        dma_ಸರತಿ_ತಳ್ಳು(dma_q, dma_make_ದತ್ತಾಂಶ(s_spad, cur_src0),
                       smctx->src0_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned, src0_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ, src0_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ, 1);
    }

    softmax_compute_fn_t compute = (softmax_compute_fn_t) smctx->compute;
    ಸ್ಥಿರಾಂಕ uint32_t ne00 = src0->ne[0];

    ಗಾಗಿ (uint32_t r = src0_ಆರಂಭ_ಅಡ್ಡಸಾಲು; r < src0_ಕೊನೆ_ಅಡ್ಡಸಾಲು; ++r) {
        ಶೂನ್ಯ * d_spad = (ಶೂನ್ಯ *) dma_ಸರತಿ_ತೆಗೆ(dma_q).src;
        ಶೂನ್ಯ * s_spad = (ಶೂನ್ಯ *) dma_ಸರತಿ_ತೆಗೆ(dma_q).dst;

        htp_trace_event_ಆರಂಭ(tr, HTP_TRACE_EVT_HVX_COMP, r);
        compute(d_spad, s_spad, ಖಾಲಿ, ne00, smctx->scale, 1.0f);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, r);

        dma_addr_t cur_dst = ದತ್ತಾಂಶ_dst + r * dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ;
        dma_ಸರತಿ_ತಳ್ಳು(dma_q, dma_make_ದತ್ತಾಂಶ(cur_dst, d_spad),
                       dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ, smctx->dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned, dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ, 1);

        ಸ್ಥಿರಾಂಕ uint32_t ಮುಂದಿನ_r = r + 2;
        ಆದರೆ (ಮುಂದಿನ_r < src0_ಕೊನೆ_ಅಡ್ಡಸಾಲು) {
            dma_addr_t ಮುಂದಿನ_src0 = ದತ್ತಾಂಶ_src0 + ಮುಂದಿನ_r * src0_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ;
            dma_ಸರತಿ_ತಳ್ಳು(dma_q, dma_make_ದತ್ತಾಂಶ(s_spad, ಮುಂದಿನ_src0),
                           smctx->src0_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned, src0_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ, src0_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ, 1);
        }
    }

    dma_ಸರತಿ_flush(dma_q);
}

ಸ್ಥಿರ ಶೂನ್ಯ softmax_thread_ಮುಖವಾಡ_dma(ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ nth, ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ ith, ಶೂನ್ಯ * ದತ್ತಾಂಶ) {
    (ಶೂನ್ಯ) nth;
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_softmax_context * smctx = (ಸ್ಥಿರಾಂಕ ರಚನೆ htp_softmax_context *) ದತ್ತಾಂಶ;
    ರಚನೆ htp_ops_context * octx = smctx->octx;
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * src0 = octx->src[0];
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * src1 = octx->src[1];
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * dst  = octx->dst;
    ರಚನೆ htp_thread_trace * tr = &octx->ctx->trace[ith];

    ಸ್ಥಿರಾಂಕ uint32_t src0_nrows            = smctx->nrows;
    ಸ್ಥಿರಾಂಕ uint32_t src0_nrows_ಪ್ರತಿ_thread = smctx->src0_nrows_ಪ್ರತಿ_thread;

    ಸ್ಥಿರಾಂಕ uint32_t src0_ಆರಂಭ_ಅಡ್ಡಸಾಲು = smctx->ಅಡ್ಡಸಾಲು_ಆರಂಭ + src0_nrows_ಪ್ರತಿ_thread * ith;
    ಸ್ಥಿರಾಂಕ uint32_t src0_ಕೊನೆ_ಅಡ್ಡಸಾಲು   = MIN(src0_ಆರಂಭ_ಅಡ್ಡಸಾಲು + src0_nrows_ಪ್ರತಿ_thread, smctx->ಅಡ್ಡಸಾಲು_ಆರಂಭ + src0_nrows);

    ಆದರೆ (src0_ಆರಂಭ_ಅಡ್ಡಸಾಲು >= src0_ಕೊನೆ_ಅಡ್ಡಸಾಲು) {
        ಹಿಂತಿರುಗಿ;
    }

    ಸ್ಥಿರಾಂಕ dma_addr_t ದತ್ತಾಂಶ_src0 = smctx->ದತ್ತಾಂಶ_src0;
    ಸ್ಥಿರಾಂಕ dma_addr_t ದತ್ತಾಂಶ_src1 = smctx->ದತ್ತಾಂಶ_src1;
    ಸ್ಥಿರಾಂಕ dma_addr_t ದತ್ತಾಂಶ_dst  = smctx->ದತ್ತಾಂಶ_dst;

    ಸ್ಥಿರಾಂಕ size_t src0_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ = src0->ne[0] * ಗಾತ್ರ(ದಶಮಾಂಶ);
    ಸ್ಥಿರಾಂಕ size_t dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ  = src0->ne[0] * ಗಾತ್ರ(ದಶಮಾಂಶ);
    ಸ್ಥಿರಾಂಕ size_t ಮುಖವಾಡ_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ = smctx->use_f16 ? (src1->ne[0] * ಗಾತ್ರ(__fp16)) : (src1->ne[0] * ಗಾತ್ರ(ದಶಮಾಂಶ));

    uint8_t * src0_vtcm_ತಳ = smctx->vtcm_src0 + (ith * smctx->vtcm_src0_ಗಾತ್ರ_ಪ್ರತಿ_thread);
    uint8_t * src1_vtcm_ತಳ = smctx->vtcm_src1 + (ith * smctx->vtcm_src1_ಗಾತ್ರ_ಪ್ರತಿ_thread);
    uint8_t * dst_vtcm_ತಳ  = smctx->vtcm_dst  + (ith * smctx->vtcm_dst_ಗಾತ್ರ_ಪ್ರತಿ_thread);

    ಸ್ಥಿರಾಂಕ size_t src0_vtcm_half = smctx->src0_spad_half_ಗಾತ್ರ;
    ಸ್ಥಿರಾಂಕ size_t src1_vtcm_half = smctx->src1_spad_half_ಗಾತ್ರ;
    ಸ್ಥಿರಾಂಕ size_t dst_vtcm_half  = smctx->dst_spad_half_ಗಾತ್ರ;

    ಸ್ಥಿರಾಂಕ uint32_t nb11 = src1->nb[1];
    ಸ್ಥಿರಾಂಕ uint32_t nb12 = src1->nb[2];
    ಸ್ಥಿರಾಂಕ uint32_t nb13 = src1->nb[3];

    ಸ್ಥಿರಾಂಕ uint32_t ne00 = src0->ne[0];
    ಸ್ಥಿರಾಂಕ uint32_t ne01 = src0->ne[1];
    ಸ್ಥಿರಾಂಕ uint32_t ne02 = src0->ne[2];
    ಸ್ಥಿರಾಂಕ uint32_t ne03 = src0->ne[3];
    ಸ್ಥಿರಾಂಕ uint32_t ne12 = src1->ne[2];
    ಸ್ಥಿರಾಂಕ uint32_t ne13 = src1->ne[3];

    ಸ್ಥಿರಾಂಕ ರಚನೆ fastdiv_ಮೌಲ್ಯಗಳು * div_ne01 = &smctx->div_ne01;
    ಸ್ಥಿರಾಂಕ ರಚನೆ fastdiv_ಮೌಲ್ಯಗಳು * div_ne02 = &smctx->div_ne02;
    ಸ್ಥಿರಾಂಕ ರಚನೆ fastdiv_ಮೌಲ್ಯಗಳು * div_ne12 = &smctx->div_ne12;
    ಸ್ಥಿರಾಂಕ ರಚನೆ fastdiv_ಮೌಲ್ಯಗಳು * div_ne13 = &smctx->div_ne13;

    dma_ಸರತಿ * dma_q = octx->ctx->dma[ith];

    ಗಾಗಿ (uint32_t r = src0_ಆರಂಭ_ಅಡ್ಡಸಾಲು, idx = 0; r < src0_ಕೊನೆ_ಅಡ್ಡಸಾಲು && idx < 2; r++, idx++) {
        dma_addr_t cur_dst  = ದತ್ತಾಂಶ_dst  + r * dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ;
        dma_addr_t cur_src0 = ದತ್ತಾಂಶ_src0 + r * src0_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ;

        uint32_t i1 = fastmodulo(r, ne01, div_ne01);
        uint32_t r_div_ne01 = fastdiv(r, div_ne01);
        uint32_t i2 = fastmodulo(r_div_ne01, ne02, div_ne02);
        uint32_t i3 = fastdiv(r_div_ne01, div_ne02);
        uint32_t i12 = (ne12 == ne02) ? i2 : fastmodulo(i2, ne12, div_ne12);
        uint32_t i13 = (ne13 == ne03) ? i3 : fastmodulo(i3, ne13, div_ne13);
        dma_addr_t cur_src1 = ದತ್ತಾಂಶ_src1 + i1 * nb11 + i12 * nb12 + i13 * nb13;

        ಶೂನ್ಯ * d_spad = dst_vtcm_ತಳ  + idx * dst_vtcm_half;
        ಶೂನ್ಯ * s_spad = src0_vtcm_ತಳ + idx * src0_vtcm_half;
        ಶೂನ್ಯ * m_spad = src1_vtcm_ತಳ + idx * src1_vtcm_half;

        dma_ಸರತಿ_ತಳ್ಳು(dma_q, dma_make_ದತ್ತಾಂಶ(cur_dst, d_spad),
                       dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ, smctx->dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned, dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ, 0);
        dma_ಸರತಿ_ತಳ್ಳು(dma_q, dma_make_ದತ್ತಾಂಶ(s_spad, cur_src0),
                       smctx->src0_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned, src0_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ, src0_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ, 1);
        dma_ಸರತಿ_ತಳ್ಳು(dma_q, dma_make_ದತ್ತಾಂಶ(m_spad, cur_src1),
                       smctx->src1_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned, ಮುಖವಾಡ_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ, ಮುಖವಾಡ_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ, 1);
    }

    softmax_compute_fn_t compute = (softmax_compute_fn_t) smctx->compute;
    ಸ್ಥಿರಾಂಕ bool has_bias = smctx->max_bias > 0.0f;
    uint32_t prev_i2 = (uint32_t)-1;
    ದಶಮಾಂಶ ಇಳಿಜಾರು = 1.0f;

    ಗಾಗಿ (uint32_t r = src0_ಆರಂಭ_ಅಡ್ಡಸಾಲು; r < src0_ಕೊನೆ_ಅಡ್ಡಸಾಲು; ++r) {
        ಶೂನ್ಯ * d_spad = (ಶೂನ್ಯ *) (uintptr_t) dma_ಸರತಿ_ತೆಗೆ(dma_q).src;
        ಶೂನ್ಯ * s_spad = (ಶೂನ್ಯ *) (uintptr_t) dma_ಸರತಿ_ತೆಗೆ(dma_q).dst;
        ಶೂನ್ಯ * m_spad = (ಶೂನ್ಯ *) (uintptr_t) dma_ಸರತಿ_ತೆಗೆ(dma_q).dst;

        ಆದರೆ (has_bias) {
            uint32_t r_div_ne01 = fastdiv(r, div_ne01);
            uint32_t i2 = fastmodulo(r_div_ne01, ne02, div_ne02);
            ಆದರೆ (i2 != prev_i2) {
                ಇಳಿಜಾರು = smctx->slopes[i2];
                prev_i2 = i2;
            }
        }

        htp_trace_event_ಆರಂಭ(tr, HTP_TRACE_EVT_HVX_COMP, r);
        compute(d_spad, s_spad, m_spad, ne00, smctx->scale, ಇಳಿಜಾರು);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, r);

        dma_addr_t cur_dst = ದತ್ತಾಂಶ_dst + r * dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ;
        dma_ಸರತಿ_ತಳ್ಳು(dma_q, dma_make_ದತ್ತಾಂಶ(cur_dst, d_spad),
                       dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ, smctx->dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned, dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ, 1);

        ಸ್ಥಿರಾಂಕ uint32_t ಮುಂದಿನ_r = r + 2;
        ಆದರೆ (ಮುಂದಿನ_r < src0_ಕೊನೆ_ಅಡ್ಡಸಾಲು) {
            dma_addr_t ಮುಂದಿನ_src0 = ದತ್ತಾಂಶ_src0 + ಮುಂದಿನ_r * src0_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ;

            uint32_t ni1 = fastmodulo(ಮುಂದಿನ_r, ne01, div_ne01);
            uint32_t nr_div_ne01 = fastdiv(ಮುಂದಿನ_r, div_ne01);
            uint32_t ni2 = fastmodulo(nr_div_ne01, ne02, div_ne02);
            uint32_t ni3 = fastdiv(nr_div_ne01, div_ne02);
            uint32_t ni12 = (ne12 == ne02) ? ni2 : fastmodulo(ni2, ne12, div_ne12);
            uint32_t ni13 = (ne13 == ne03) ? ni3 : fastmodulo(ni3, ne13, div_ne13);
            dma_addr_t ಮುಂದಿನ_src1 = ದತ್ತಾಂಶ_src1 + ni1 * nb11 + ni12 * nb12 + ni13 * nb13;

            dma_ಸರತಿ_ತಳ್ಳು(dma_q, dma_make_ದತ್ತಾಂಶ(s_spad, ಮುಂದಿನ_src0),
                           smctx->src0_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned, src0_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ, src0_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ, 1);
            dma_ಸರತಿ_ತಳ್ಳು(dma_q, dma_make_ದತ್ತಾಂಶ(m_spad, ಮುಂದಿನ_src1),
                           smctx->src1_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned, ಮುಖವಾಡ_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ, ಮುಖವಾಡ_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ, 1);
        }
    }

    dma_ಸರತಿ_flush(dma_q);
}

ಸ್ಥಿರ ಪೂರ್ಣಾಂಕ execute_op_softmax_f32(ರಚನೆ htp_ops_context * octx) {
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * src0 = octx->src[0];
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * dst  = octx->dst;

    ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * op_type = "softmax-f32";

    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_softmax_kernel_params * kparams =
        (ಸ್ಥಿರಾಂಕ ರಚನೆ htp_softmax_kernel_params *) octx->kernel_params;

    ಆದರೆ (!htp_ops_context_ಗಣ_n_ಎಳೆಗಳು(octx, kparams->n_ಎಳೆಗಳು)) {
        ಹಿಂತಿರುಗಿ HTP_ಸ್ಥಿತಿ_INVAL_PARAMS;
    }

    ಆದರೆ (kparams->kernel_id >= HTP_SOFTMAX_KERNEL_ಎಣಿಕೆ) {
        ಹಿಂತಿರುಗಿ HTP_ಸ್ಥಿತಿ_INVAL_PARAMS;
    }

    ಆದರೆ (octx->ctx->vtcm_ಗಾತ್ರ < (size_t) kparams->vtcm_ಗಾತ್ರ) {
        FARF(ERROR, "%s : current VTCM reservation %zu is too small, needed %u\n",
             op_type, octx->ctx->vtcm_ಗಾತ್ರ, kparams->vtcm_ಗಾತ್ರ);
        ಹಿಂತಿರುಗಿ HTP_ಸ್ಥಿತಿ_VTCM_TOO_SMALL;
    }

    ಸ್ಥಿರಾಂಕ uint32_t src0_nrows = src0->ne[1] * src0->ne[2] * src0->ne[3];
    ಸ್ಥಿರಾಂಕ size_t elem_ಗಾತ್ರ    = ಗಾತ್ರ(ದಶಮಾಂಶ);
    ಸ್ಥಿರಾಂಕ size_t dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ = dst->nb[1];

    uint32_t ಅಡ್ಡಸಾಲು_ಆರಂಭ = 0;
    uint32_t nrows     = src0_nrows;

    ಆದರೆ (octx->ctx->mdev.ಎಣಿಕೆ_2 > 1) {
        uint32_t ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_chunk = 0;
        htp_tensor_mdev_ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_chunk(dst, (uint32_t) elem_ಗಾತ್ರ, (uint32_t) dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ, &ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_chunk);
        ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor_mdev_range range = htp_tensor_mdev_ವಿಭಜನೆ(
            src0_nrows, ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_chunk, octx->ctx->mdev.idx, octx->ctx->mdev.ಎಣಿಕೆ_2, &octx->ctx->mdev.ಎಣಿಕೆ_div);
        ಅಡ್ಡಸಾಲು_ಆರಂಭ = range.ಆರಂಭ;
        nrows     = range.ಎಣಿಕೆ_2;
        ಆದರೆ (nrows < octx->n_ಎಳೆಗಳು) {
            htp_ops_context_ಗಣ_n_ಎಳೆಗಳು(octx, nrows ? nrows : 1);
        }
    }

    ಆದರೆ (nrows == 0) {
        ಹಿಂತಿರುಗಿ HTP_ಸ್ಥಿತಿ_OK;
    }

    ಸ್ಥಿರಾಂಕ uint32_t n_ಎಳೆಗಳು = octx->n_ಎಳೆಗಳು;
    uint8_t * ಸ್ಥಿರಾಂಕ vtcm_ತಳ = (uint8_t *) octx->ctx->vtcm_ತಳ;

    ಸ್ಥಿರಾಂಕ uint32_t off_src0 = 0;
    ಸ್ಥಿರಾಂಕ uint32_t off_dst  = off_src0 + kparams->vtcm_src0_ಗಾತ್ರ_ಪ್ರತಿ_thread * kparams->n_ಎಳೆಗಳು;
    ಸ್ಥಿರಾಂಕ uint32_t off_src1 = off_dst  + kparams->vtcm_dst_ಗಾತ್ರ_ಪ್ರತಿ_thread  * kparams->n_ಎಳೆಗಳು;

    ರಚನೆ htp_softmax_context smctx = {
        .octx                  = octx,
        .kparams               = kparams,
        .compute               = (ಶೂನ್ಯ *) softmax_kernels[kparams->kernel_id],

        .ದತ್ತಾಂಶ_src0             = src0->ದತ್ತಾಂಶ,
        .ದತ್ತಾಂಶ_src1             = kparams->use_src1 ? octx->src[1]->ದತ್ತಾಂಶ : 0,
        .ದತ್ತಾಂಶ_dst              = dst->ದತ್ತಾಂಶ,

        .vtcm_src0             = VTCM_LAYOUT_PTR(uint8_t, vtcm_ತಳ, off_src0),
        .vtcm_dst              = VTCM_LAYOUT_PTR(uint8_t, vtcm_ತಳ, off_dst),
        .vtcm_src1             = VTCM_LAYOUT_PTR_OPTIONAL(uint8_t, vtcm_ತಳ, off_src1, kparams->use_src1),

        .vtcm_src0_ಗಾತ್ರ_ಪ್ರತಿ_thread = kparams->vtcm_src0_ಗಾತ್ರ_ಪ್ರತಿ_thread,
        .vtcm_src1_ಗಾತ್ರ_ಪ್ರತಿ_thread = kparams->vtcm_src1_ಗಾತ್ರ_ಪ್ರತಿ_thread,
        .vtcm_dst_ಗಾತ್ರ_ಪ್ರತಿ_thread  = kparams->vtcm_dst_ಗಾತ್ರ_ಪ್ರತಿ_thread,

        .src0_spad_half_ಗಾತ್ರ   = kparams->src0_spad_half_ಗಾತ್ರ,
        .src1_spad_half_ಗಾತ್ರ   = kparams->src1_spad_half_ಗಾತ್ರ,
        .dst_spad_half_ಗಾತ್ರ    = kparams->dst_spad_half_ಗಾತ್ರ,

        .src0_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned = kparams->src0_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned,
        .src1_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned = kparams->src1_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned,
        .dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned  = kparams->dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned,

        .use_f16               = kparams->use_f16 != 0,

        .n_ತಲೆ                = kparams->n_ತಲೆ,
        .n_ತಲೆ_log2           = kparams->n_ತಲೆ_log2,

        .scale                 = kparams->scale,
        .max_bias              = kparams->max_bias,
        .m0                    = kparams->m0,
        .m1                    = kparams->m1,

        .div_ne01              = kparams->div_ne01,
        .div_ne02              = kparams->div_ne02,
        .div_ne12              = kparams->div_ne12,
        .div_ne13              = kparams->div_ne13,

        .src0_nrows_ಪ್ರತಿ_thread = fastdiv(nrows + n_ಎಳೆಗಳು - 1, &octx->n_ಎಳೆಗಳು_div),
        .ಅಡ್ಡಸಾಲು_ಆರಂಭ             = ಅಡ್ಡಸಾಲು_ಆರಂಭ,
        .nrows                 = nrows,
    };

    ಆದರೆ (kparams->max_bias > 0.0f && kparams->use_src1) {
        ಆದರೆ (kparams->n_ತಲೆ > 512) {
            ಹಿಂತಿರುಗಿ HTP_ಸ್ಥಿತಿ_INVAL_PARAMS;
        }
        ಗಾಗಿ (uint32_t h = 0; h < kparams->n_ತಲೆ; h += 32) {
            HVX_ಸದಿಶ v_slopes = hvx_alibi_slopes(h, 1, kparams->n_ತಲೆ_log2, kparams->m0, kparams->m1);
            hvx_vmem(&smctx.slopes[h]) = v_slopes;
        }
    }

    work_ಸರತಿ_func_t task_func = kparams->use_src1 ? softmax_thread_ಮುಖವಾಡ_dma : softmax_thread_dma;
    work_ಸರತಿ_ಸತತ(octx->ctx->work_ಸರತಿ, task_func, &smctx, n_ಎಳೆಗಳು);

    ಹಿಂತಿರುಗಿ HTP_ಸ್ಥಿತಿ_OK;
}

ಪೂರ್ಣಾಂಕ op_softmax(ರಚನೆ htp_ops_context * octx) {
    ಪೂರ್ಣಾಂಕ err = HTP_ಸ್ಥಿತಿ_OK;

    ಆಯ್ಕೆ (octx->src[0]->type) {
        ಸ್ಥಿತಿ HTP_TYPE_F32:
            err = execute_op_softmax_f32(octx);
            ನಿಲ್ಲಿಸು;

        ಪೂರ್ವನಿಯೋಜಿತ:
            err = HTP_ಸ್ಥಿತಿ_NO_SUPPORT;
            ನಿಲ್ಲಿಸು;
    }

    ಹಿಂತಿರುಗಿ err;
}
