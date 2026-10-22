#ಸೂಚನೆ clang diagnostic ignored "-Wunused-variable"
#ಸೂಚನೆ clang diagnostic ignored "-Wunused-function"
#ಸೂಚನೆ clang diagnostic ignored "-Wunused-but-set-variable"

#ಅನ್ತರ್ಭಾವ <HAP_farf.h>
#ಅನ್ತರ್ಭಾವ <HAP_perf.h>

#ಅನ್ತರ್ಭಾವ <math.h>
#ಅನ್ತರ್ಭಾವ <string.h>

#ಅನ್ತರ್ಭಾವ "dma-queue.h"
#ಅನ್ತರ್ಭಾವ "hvx-utils.h"

#ವ್ಯಾಖ್ಯಾನಿಸು GGML_COMMON_DECL_C
#ಅನ್ತರ್ಭಾವ "ggml-common.h"
#ಅನ್ತರ್ಭಾವ "hex-common.h"
#ಅನ್ತರ್ಭಾವ "hex-profile.h"
#ಅನ್ತರ್ಭಾವ "binary-ops.h"
#ಅನ್ತರ್ಭಾವ "htp-ctx.h"
#ಅನ್ತರ್ಭಾವ "htp-ops.h"
#ಅನ್ತರ್ಭಾವ "htp-tensor.h"

#ವ್ಯಾಖ್ಯಾನಿತವಾಗದಿದ್ದರೆ MIN
#ವ್ಯಾಖ್ಯಾನಿಸು MIN(a, b) ((a) < (b) ? (a) : (b))
#ಆದರೆ_ಅಂತ್ಯ

// Context for binary operations
ರಚನೆ htp_ದ್ವಿಮಾನ_context {
    ರಚನೆ htp_ops_context * octx;
    ರಚನೆ htp_ದ್ವಿಮಾನ_vtcm_layout vtcm_layout;
    uint8_t * vtcm_ತಳ;

    ರಚನೆ fastdiv_ಮೌಲ್ಯಗಳು src0_dim1_div; // ne01
    ರಚನೆ fastdiv_ಮೌಲ್ಯಗಳು src0_dim2_div; // ne02
    ರಚನೆ fastdiv_ಮೌಲ್ಯಗಳು src0_dim12_div;// ne03

    ರಚನೆ fastdiv_ಮೌಲ್ಯಗಳು src1_dim1_div; // ne11
    ರಚನೆ fastdiv_ಮೌಲ್ಯಗಳು src1_dim2_div; // ne12
    ರಚನೆ fastdiv_ಮೌಲ್ಯಗಳು src1_dim3_div; // ne13

    uint32_t ಖಂಡ_max;
    uint32_t nrows_ಪ್ರತಿ_thread;
    uint32_t ಒಟ್ಟು_ಅಡ್ಡಸಾಲುಗಳು;
    uint32_t ಅಡ್ಡಸಾಲು_ಆರಂಭ;
    size_t   src0_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned;
    size_t   src1_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned;
    size_t   dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned;
    size_t   ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_ಬೈಟುಗಳು;

    bool ವಿಭಜಿಸು_at_ne01;
    bool ವಿಭಜಿಸು_at_ne02;

    ಶೂನ್ಯ * compute;
};

#ವ್ಯಾಖ್ಯಾನಿಸು htp_ದ್ವಿಮಾನ_preamble                        \
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * src0 = octx->src[0]; \
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * src1 = octx->src[1]; \
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * dst  = octx->dst;    \
                                                   \
    ಸ್ಥಿರಾಂಕ uint32_t ne00 = src0->ne[0];             \
    ಸ್ಥಿರಾಂಕ uint32_t ne01 = src0->ne[1];             \
    ಸ್ಥಿರಾಂಕ uint32_t ne02 = src0->ne[2];             \
    ಸ್ಥಿರಾಂಕ uint32_t ne03 = src0->ne[3];             \
                                                   \
    ಸ್ಥಿರಾಂಕ uint32_t ne10 = src1->ne[0];             \
    ಸ್ಥಿರಾಂಕ uint32_t ne11 = src1->ne[1];             \
    ಸ್ಥಿರಾಂಕ uint32_t ne12 = src1->ne[2];             \
    ಸ್ಥಿರಾಂಕ uint32_t ne13 = src1->ne[3];             \
                                                   \
    ಸ್ಥಿರಾಂಕ uint32_t nb01 = src0->nb[1];             \
    ಸ್ಥಿರಾಂಕ uint32_t nb02 = src0->nb[2];             \
    ಸ್ಥಿರಾಂಕ uint32_t nb03 = src0->nb[3];             \
                                                   \
    ಸ್ಥಿರಾಂಕ uint32_t nb11 = src1->nb[1];             \
    ಸ್ಥಿರಾಂಕ uint32_t nb12 = src1->nb[2];             \
    ಸ್ಥಿರಾಂಕ uint32_t nb13 = src1->nb[3];             \
                                                   \
    ಸ್ಥಿರಾಂಕ uint32_t nb1 = dst->nb[1];               \
    ಸ್ಥಿರಾಂಕ uint32_t nb2 = dst->nb[2];               \
    ಸ್ಥಿರಾಂಕ uint32_t nb3 = dst->nb[3];

ಸ್ಥಿರ ಸಾಲೊಳಗೆ uint32_t calc_ಖಂಡ_ಗಾತ್ರ(ರಚನೆ htp_ದ್ವಿಮಾನ_context * bctx, uint32_t ir, uint32_t ಕೊನೆ_ಅಡ್ಡಸಾಲು, uint32_t ne01, uint32_t ne02) {
    uint32_t i03, i02, i01, rem;
    i03 = fastdiv(ir, &bctx->src0_dim12_div);
    rem = ir - i03 * (ne02 * ne01);
    i02 = fastdiv(rem, &bctx->src0_dim1_div);
    i01 = rem - i02 * ne01;

    uint32_t ಅಡ್ಡಸಾಲುಗಳು_ಎಡ = ಕೊನೆ_ಅಡ್ಡಸಾಲು - ir;
    uint32_t ಖಂಡ_ಎಲ್ಲೆ = ಅಡ್ಡಸಾಲುಗಳು_ಎಡ;

    ಆದರೆ (bctx->ವಿಭಜಿಸು_at_ne01) {
        ಖಂಡ_ಎಲ್ಲೆ = MIN(ಖಂಡ_ಎಲ್ಲೆ, ne01 - i01);
    }
    ಆದರೆ (bctx->ವಿಭಜಿಸು_at_ne02) {
         uint32_t ಅಡ್ಡಸಾಲುಗಳು_ಒಳ_plane = (ne02 * ne01) - rem;
         ಖಂಡ_ಎಲ್ಲೆ = MIN(ಖಂಡ_ಎಲ್ಲೆ, ಅಡ್ಡಸಾಲುಗಳು_ಒಳ_plane);
    }

    ಹಿಂತಿರುಗಿ MIN(bctx->ಖಂಡ_max, ಖಂಡ_ಎಲ್ಲೆ);
}

// Out-of-line compute micro-kernels

ಪ್ರಕಾರವ್ಯಾಖ್ಯೆ ಶೂನ್ಯ (*compute_scalar_t)(
    uint8_t * dst, ಸ್ಥಿರಾಂಕ uint8_t * src0, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * s1_ಕೋಷ್ಟಕ,
    uint32_t cur_i11, uint32_t ne11, uint32_t n_ಅಡ್ಡಸಾಲುಗಳು,
    size_t dst_stride, size_t src0_stride, uint32_t ne00);

#ವ್ಯಾಖ್ಯಾನಿಸು DEFINE_COMPUTE_SCALAR(NAME, TYPE, HVX_STMT)             \
ಸ್ಥಿರ ಶೂನ್ಯ compute_scalar_##NAME(                              \
    uint8_t * dst, ಸ್ಥಿರಾಂಕ uint8_t * src0, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * s1_ಕೋಷ್ಟಕ, \
    uint32_t cur_i11, uint32_t ne11, uint32_t n_ಅಡ್ಡಸಾಲುಗಳು,           \
    size_t dst_stride, size_t src0_stride, uint32_t ne00) {     \
    ಸ್ಥಿರಾಂಕ TYPE * ಕೋಷ್ಟಕ = (ಸ್ಥಿರಾಂಕ TYPE *) s1_ಕೋಷ್ಟಕ;               \
    ಗಾಗಿ (uint32_t r = 0; r < n_ಅಡ್ಡಸಾಲುಗಳು; r++) {                     \
        uint8_t * r_dst = dst + r * dst_stride;                 \
        ಸ್ಥಿರಾಂಕ uint8_t * r_src0 = src0 + r * src0_stride;        \
        TYPE val = ಕೋಷ್ಟಕ[cur_i11];                              \
        HVX_STMT;                                               \
        ಆದರೆ (ne11 > 1 && ++cur_i11 == ne11) {                    \
            cur_i11 = 0;                                        \
        }                                                       \
    }                                                           \
}

DEFINE_COMPUTE_SCALAR(ಕೂಡಿಸು_f32, ದಶಮಾಂಶ,    hvx_ಕೂಡಿಸು_scalar_f32_aa(r_dst, r_src0, val, ne00))
DEFINE_COMPUTE_SCALAR(ಕೂಡಿಸು_f16, _Float16, hvx_ಕೂಡಿಸು_scalar_f16_aa(r_dst, r_src0, val, ne00))
DEFINE_COMPUTE_SCALAR(sub_f32, ದಶಮಾಂಶ,    hvx_sub_scalar_f32_aa(r_dst, r_src0, val, ne00))
DEFINE_COMPUTE_SCALAR(sub_f16, _Float16, hvx_sub_scalar_f16_aa(r_dst, r_src0, val, ne00))
DEFINE_COMPUTE_SCALAR(mul_f32, ದಶಮಾಂಶ,    hvx_mul_scalar_f32_aa(r_dst, r_src0, val, ne00))
DEFINE_COMPUTE_SCALAR(mul_f16, _Float16, hvx_mul_scalar_f16_aa(r_dst, r_src0, val, ne00))
DEFINE_COMPUTE_SCALAR(div_f32, ದಶಮಾಂಶ,    hvx_mul_scalar_f32_aa(r_dst, r_src0, 1.0f / (val), ne00))
DEFINE_COMPUTE_SCALAR(div_f16, _Float16, hvx_div_scalar_f16_aa(r_dst, r_src0, val, ne00))

ಪ್ರಕಾರವ್ಯಾಖ್ಯೆ ಶೂನ್ಯ (*compute_same_shape_t)(
    uint8_t * dst, ಸ್ಥಿರಾಂಕ uint8_t * src0, ಸ್ಥಿರಾಂಕ uint8_t * src1,
    uint32_t n_ಅಡ್ಡಸಾಲುಗಳು, size_t dst_stride, size_t src0_stride, size_t src1_stride, uint32_t ne00);

#ವ್ಯಾಖ್ಯಾನಿಸು DEFINE_COMPUTE_SAME_SHAPE(NAME, HVX_FN)                                                  \
ಸ್ಥಿರ ಶೂನ್ಯ compute_same_shape_##NAME(                                                           \
    uint8_t * dst, ಸ್ಥಿರಾಂಕ uint8_t * src0, ಸ್ಥಿರಾಂಕ uint8_t * src1,                                   \
    uint32_t n_ಅಡ್ಡಸಾಲುಗಳು, size_t dst_stride, size_t src0_stride, size_t src1_stride, uint32_t ne00) { \
    ಗಾಗಿ (uint32_t r = 0; r < n_ಅಡ್ಡಸಾಲುಗಳು; r++) {                                                      \
        HVX_FN(dst + r * dst_stride, src0 + r * src0_stride, src1 + r * src1_stride, ne00);      \
    }                                                                                            \
}

DEFINE_COMPUTE_SAME_SHAPE(ಕೂಡಿಸು_f32, hvx_ಕೂಡಿಸು_f32_aaa)
DEFINE_COMPUTE_SAME_SHAPE(ಕೂಡಿಸು_f16, hvx_ಕೂಡಿಸು_f16_aaa)
DEFINE_COMPUTE_SAME_SHAPE(sub_f32, hvx_sub_f32_aaa)
DEFINE_COMPUTE_SAME_SHAPE(sub_f16, hvx_sub_f16_aaa)
DEFINE_COMPUTE_SAME_SHAPE(mul_f32, hvx_mul_f32_aaa)
DEFINE_COMPUTE_SAME_SHAPE(mul_f16, hvx_mul_f16_aaa)
DEFINE_COMPUTE_SAME_SHAPE(div_f32, hvx_div_f32_aaa)
DEFINE_COMPUTE_SAME_SHAPE(div_f16, hvx_div_f16_aaa)

ಪ್ರಕಾರವ್ಯಾಖ್ಯೆ ಶೂನ್ಯ (*compute_ಅಡ್ಡಸಾಲು_bcast_t)(
    uint8_t * dst, ಸ್ಥಿರಾಂಕ uint8_t * src0, ಸ್ಥಿರಾಂಕ uint8_t * src1,
    uint32_t n_ಅಡ್ಡಸಾಲುಗಳು, size_t dst_stride, size_t src0_stride, uint32_t ne00);

#ವ್ಯಾಖ್ಯಾನಿಸು DEFINE_COMPUTE_ಅಡ್ಡಸಾಲು_BCAST(NAME, HVX_FN)                                               \
ಸ್ಥಿರ ಶೂನ್ಯ compute_ಅಡ್ಡಸಾಲು_bcast##NAME(                                                        \
    uint8_t * dst, ಸ್ಥಿರಾಂಕ uint8_t * src0, ಸ್ಥಿರಾಂಕ uint8_t * src1,                               \
    uint32_t n_ಅಡ್ಡಸಾಲುಗಳು, size_t dst_stride, size_t src0_stride, uint32_t ne00) {                 \
    ಗಾಗಿ (uint32_t r = 0; r < n_ಅಡ್ಡಸಾಲುಗಳು; r++) {                                                  \
        HVX_FN(dst + r * dst_stride, src0 + r * src0_stride, src1, ne00);                    \
    }                                                                                        \
}

DEFINE_COMPUTE_ಅಡ್ಡಸಾಲು_BCAST(ಕೂಡಿಸು_f32, hvx_ಕೂಡಿಸು_f32_aaa)
DEFINE_COMPUTE_ಅಡ್ಡಸಾಲು_BCAST(ಕೂಡಿಸು_f16, hvx_ಕೂಡಿಸು_f16_aaa)
DEFINE_COMPUTE_ಅಡ್ಡಸಾಲು_BCAST(sub_f32, hvx_sub_f32_aaa)
DEFINE_COMPUTE_ಅಡ್ಡಸಾಲು_BCAST(sub_f16, hvx_sub_f16_aaa)
DEFINE_COMPUTE_ಅಡ್ಡಸಾಲು_BCAST(mul_f32, hvx_mul_f32_aaa)
DEFINE_COMPUTE_ಅಡ್ಡಸಾಲು_BCAST(mul_f16, hvx_mul_f16_aaa)
DEFINE_COMPUTE_ಅಡ್ಡಸಾಲು_BCAST(div_f32, hvx_div_f32_aaa)
DEFINE_COMPUTE_ಅಡ್ಡಸಾಲು_BCAST(div_f16, hvx_div_f16_aaa)

ಪ್ರಕಾರವ್ಯಾಖ್ಯೆ ಶೂನ್ಯ (*compute_complex_t)(
    uint8_t * dst, ಸ್ಥಿರಾಂಕ uint8_t * src0, ಸ್ಥಿರಾಂಕ uint8_t * src1_plane,
    uint32_t i01, uint32_t ne11, ಸ್ಥಿರಾಂಕ ರಚನೆ fastdiv_ಮೌಲ್ಯಗಳು * div11, uint32_t src1_stride,
    uint32_t n_ಅಡ್ಡಸಾಲುಗಳು, size_t dst_stride, size_t src0_stride, uint32_t ne00);

#ವ್ಯಾಖ್ಯಾನಿಸು DEFINE_COMPUTE_COMPLEX(NAME, HVX_FN)                                                 \
ಸ್ಥಿರ ಶೂನ್ಯ compute_complex_##NAME(                                                          \
    uint8_t * dst, ಸ್ಥಿರಾಂಕ uint8_t * src0, ಸ್ಥಿರಾಂಕ uint8_t * src1_plane,                         \
    uint32_t i01, uint32_t ne11, ಸ್ಥಿರಾಂಕ ರಚನೆ fastdiv_ಮೌಲ್ಯಗಳು * div11, uint32_t src1_stride,  \
    uint32_t n_ಅಡ್ಡಸಾಲುಗಳು, size_t dst_stride, size_t src0_stride, uint32_t ne00) {                 \
    ಗಾಗಿ (uint32_t r = 0; r < n_ಅಡ್ಡಸಾಲುಗಳು; r++) {                                                  \
        uint32_t i11 = fastmodulo(i01 + r, ne11, div11);                                     \
        ಸ್ಥಿರಾಂಕ uint8_t * r_src1 = src1_plane + i11 * src1_stride;                             \
        HVX_FN(dst + r * dst_stride, src0 + r * src0_stride, r_src1, ne00);                  \
    }                                                                                        \
}

DEFINE_COMPUTE_COMPLEX(ಕೂಡಿಸು_f32, hvx_ಕೂಡಿಸು_f32_aau)
DEFINE_COMPUTE_COMPLEX(ಕೂಡಿಸು_f16, hvx_ಕೂಡಿಸು_f16_aau)
DEFINE_COMPUTE_COMPLEX(sub_f32, hvx_sub_f32_aau)
DEFINE_COMPUTE_COMPLEX(sub_f16, hvx_sub_f16_aau)
DEFINE_COMPUTE_COMPLEX(mul_f32, hvx_mul_f32_aau)
DEFINE_COMPUTE_COMPLEX(mul_f16, hvx_mul_f16_aau)
DEFINE_COMPUTE_COMPLEX(div_f32, hvx_div_f32_aau)
DEFINE_COMPUTE_COMPLEX(div_f16, hvx_div_f16_aau)

ಪ್ರಕಾರವ್ಯಾಖ್ಯೆ ಶೂನ್ಯ (*compute_repeat_t)(
    uint8_t * dst, ಸ್ಥಿರಾಂಕ uint8_t * src0, ಸ್ಥಿರಾಂಕ uint8_t * src1_plane,
    uint32_t i01, uint32_t ne11, ಸ್ಥಿರಾಂಕ ರಚನೆ fastdiv_ಮೌಲ್ಯಗಳು * div11, uint32_t src1_stride,
    uint32_t n_ಅಡ್ಡಸಾಲುಗಳು, size_t dst_stride, size_t src0_stride, uint32_t ne00, uint32_t ne10);

#ವ್ಯಾಖ್ಯಾನಿಸು DEFINE_COMPUTE_REPEAT(NAME, TYPE, HVX_FN)                                            \
ಸ್ಥಿರ ಶೂನ್ಯ compute_repeat_##NAME(                                                           \
    uint8_t * dst, ಸ್ಥಿರಾಂಕ uint8_t * src0, ಸ್ಥಿರಾಂಕ uint8_t * src1_plane,                         \
    uint32_t i01, uint32_t ne11, ಸ್ಥಿರಾಂಕ ರಚನೆ fastdiv_ಮೌಲ್ಯಗಳು * div11, uint32_t src1_stride,  \
    uint32_t n_ಅಡ್ಡಸಾಲುಗಳು, size_t dst_stride, size_t src0_stride, uint32_t ne00, uint32_t ne10) {  \
    ಗಾಗಿ (uint32_t r = 0; r < n_ಅಡ್ಡಸಾಲುಗಳು; r++) {                                                  \
        uint32_t i11 = fastmodulo(i01 + r, ne11, div11);                                     \
        ಸ್ಥಿರಾಂಕ uint8_t * r_src1_ಅಡ್ಡಸಾಲು = src1_plane + i11 * src1_stride;                         \
        uint8_t * r_dst = dst + r * dst_stride;                                              \
        ಸ್ಥಿರಾಂಕ uint8_t * r_src0 = src0 + r * src0_stride;                                     \
        ಗಾಗಿ (uint32_t c = 0; c < ne00; c += ne10) {                                          \
            uint32_t len = MIN(ne10, ne00 - c);                                              \
            HVX_FN(r_dst + c * ಗಾತ್ರ(TYPE), r_src0 + c * ಗಾತ್ರ(TYPE), r_src1_ಅಡ್ಡಸಾಲು, len);    \
        }                                                                                    \
    }                                                                                        \
}

DEFINE_COMPUTE_REPEAT(ಕೂಡಿಸು_f32, ದಶಮಾಂಶ,    hvx_ಕೂಡಿಸು_f32_uuu)
DEFINE_COMPUTE_REPEAT(ಕೂಡಿಸು_f16, _Float16, hvx_ಕೂಡಿಸು_f16_uuu)
DEFINE_COMPUTE_REPEAT(sub_f32, ದಶಮಾಂಶ,    hvx_sub_f32_uuu)
DEFINE_COMPUTE_REPEAT(sub_f16, _Float16, hvx_sub_f16_uuu)
DEFINE_COMPUTE_REPEAT(mul_f32, ದಶಮಾಂಶ,    hvx_mul_f32_uuu)
DEFINE_COMPUTE_REPEAT(mul_f16, _Float16, hvx_mul_f16_uuu)
DEFINE_COMPUTE_REPEAT(div_f32, ದಶಮಾಂಶ,    hvx_div_f32_uuu)
DEFINE_COMPUTE_REPEAT(div_f16, _Float16, hvx_div_f16_uuu)

ಪ್ರಕಾರವ್ಯಾಖ್ಯೆ ಶೂನ್ಯ (*compute_ಕೂಡಿಸು_id_t)(
    uint8_t * dst, ಸ್ಥಿರಾಂಕ uint8_t * src0, ಸ್ಥಿರಾಂಕ uint8_t * src1_ದತ್ತಾಂಶ, ಸ್ಥಿರಾಂಕ int32_t * src2_ids,
    uint32_t i01, uint32_t i02, uint32_t ne01, uint32_t src1_stride,
    uint32_t n_ಅಡ್ಡಸಾಲುಗಳು, size_t dst_stride, size_t src0_stride, uint32_t ne00);

ಸ್ಥಿರ ಶೂನ್ಯ compute_ಕೂಡಿಸು_id_f32(
    uint8_t * dst, ಸ್ಥಿರಾಂಕ uint8_t * src0, ಸ್ಥಿರಾಂಕ uint8_t * src1_ದತ್ತಾಂಶ, ಸ್ಥಿರಾಂಕ int32_t * src2_ids,
    uint32_t i01, uint32_t i02, uint32_t ne01, uint32_t src1_stride,
    uint32_t n_ಅಡ್ಡಸಾಲುಗಳು, size_t dst_stride, size_t src0_stride, uint32_t ne00) {
    ಗಾಗಿ (uint32_t r = 0; r < n_ಅಡ್ಡಸಾಲುಗಳು; r++) {
        uint32_t r_i01 = i01 + r;
        ಸ್ಥಿರಾಂಕ int32_t idx = src2_ids[i02 * ne01 + r_i01];
        ಆದರೆ (idx < 0) {
            ಸ್ಮರಣೆ_ನಕಲು(dst + r * dst_stride, src0 + r * src0_stride, ne00 * ಗಾತ್ರ(ದಶಮಾಂಶ));
            ಮುಂದುವರಿ;
        }
        ಸ್ಥಿರಾಂಕ uint8_t * r_src1 = src1_ದತ್ತಾಂಶ + idx * src1_stride;
        ಸ್ಥಿರಾಂಕ uint8_t * r_src0 = src0 + r * src0_stride;
        uint8_t * r_dst = dst + r * dst_stride;
        hvx_ಕೂಡಿಸು_f32_aaa(r_dst, r_src0, r_src1, ne00);
    }
}

// 1. Scalar src1 in VTCM via DMA (ne10 == 1)
ಸ್ಥಿರ ಶೂನ್ಯ ದ್ವಿಮಾನ_thread_scalar(ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ nth, ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ ith, ಶೂನ್ಯ * ದತ್ತಾಂಶ) {
    ರಚನೆ htp_ದ್ವಿಮಾನ_context * bctx = (ರಚನೆ htp_ದ್ವಿಮಾನ_context *) ದತ್ತಾಂಶ;
    ರಚನೆ htp_ops_context * octx = bctx->octx;
    htp_ದ್ವಿಮಾನ_preamble;

    ಸ್ಥಿರಾಂಕ uint32_t elem_ಗಾತ್ರ = (src0->type == HTP_TYPE_F32) ? ಗಾತ್ರ(ದಶಮಾಂಶ) : ಗಾತ್ರ(_Float16);
    ಸ್ಥಿರಾಂಕ size_t plane_stride = (size_t) ne12 * ne11 * elem_ಗಾತ್ರ;
    ಸ್ಥಿರಾಂಕ size_t ಅಡ್ಡಸಾಲು_stride   = (size_t) ne11 * elem_ಗಾತ್ರ;

    ಸ್ಥಿರಾಂಕ uint32_t ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_ಬೈಟುಗಳು = bctx->ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_ಬೈಟುಗಳು;
    ಸ್ಥಿರಾಂಕ uint32_t ಆರಂಭ_ಅಡ್ಡಸಾಲು = bctx->ಅಡ್ಡಸಾಲು_ಆರಂಭ + bctx->nrows_ಪ್ರತಿ_thread * ith;
    ಸ್ಥಿರಾಂಕ uint32_t ಕೊನೆ_ಅಡ್ಡಸಾಲು   = MIN(ಆರಂಭ_ಅಡ್ಡಸಾಲು + bctx->nrows_ಪ್ರತಿ_thread, bctx->ಅಡ್ಡಸಾಲು_ಆರಂಭ + bctx->ಒಟ್ಟು_ಅಡ್ಡಸಾಲುಗಳು);
    ಆದರೆ (ಆರಂಭ_ಅಡ್ಡಸಾಲು >= ಕೊನೆ_ಅಡ್ಡಸಾಲು) ಹಿಂತಿರುಗಿ;

    FARF(ಮೇಲಿನ_2, "binary-scalar: %d/%d (%u:%u) row-size %u (%u)",
         ith, nth, ಆರಂಭ_ಅಡ್ಡಸಾಲು, ಕೊನೆ_ಅಡ್ಡಸಾಲು, nb01, bctx->dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned);

    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_ದ್ವಿಮಾನ_vtcm_layout * layout = &bctx->vtcm_layout;
    uint8_t * src0_spad_ತಳ = VTCM_LAYOUT_PTR(uint8_t, bctx->vtcm_ತಳ, layout->off_src0) + (ith * layout->src0_ಬೈಟುಗಳು_ಪ್ರತಿ_thread);
    uint8_t * dst_spad_ತಳ  = VTCM_LAYOUT_PTR(uint8_t, bctx->vtcm_ತಳ, layout->off_dst)  + (ith * layout->dst_ಬೈಟುಗಳು_ಪ್ರತಿ_thread);
    size_t src0_spad_half    = layout->src0_spad_half_ಗಾತ್ರ;
    size_t dst_spad_half     = layout->dst_spad_half_ಗಾತ್ರ;
    ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * s1_ಕೋಷ್ಟಕ    = VTCM_LAYOUT_PTR(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ, bctx->vtcm_ತಳ, layout->off_src1);

    dma_ಸರತಿ * dma_q = octx->ctx->dma[ith];
    uint32_t ir_prefetch = ಆರಂಭ_ಅಡ್ಡಸಾಲು;
    ಪೂರ್ಣಾಂಕ spad_idx = 0;

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ k = 0; k < 2 && ir_prefetch < ಕೊನೆ_ಅಡ್ಡಸಾಲು; k++) {
        uint32_t current_ಖಂಡ_ಗಾತ್ರ = calc_ಖಂಡ_ಗಾತ್ರ(bctx, ir_prefetch, ಕೊನೆ_ಅಡ್ಡಸಾಲು, ne01, ne02);
        uint32_t i03, i02, i01, rem;
        i03 = fastdiv(ir_prefetch, &bctx->src0_dim12_div);
        rem = ir_prefetch - i03 * (ne02 * ne01);
        i02 = fastdiv(rem, &bctx->src0_dim1_div);
        i01 = rem - i02 * ne01;

        dma_addr_t src0_curr = src0->ದತ್ತಾಂಶ + i03 * nb03 + i02 * nb02 + i01 * nb01;
        dma_addr_t dst_curr  = dst->ದತ್ತಾಂಶ  + i03 * nb3  + i02 * nb2  + i01 * nb1;

        uint8_t * s0_spad = src0_spad_ತಳ + spad_idx * src0_spad_half;
        uint8_t * d_spad  = dst_spad_ತಳ  + spad_idx * dst_spad_half;

        dma_ಸರತಿ_ತಳ್ಳು(dma_q, dma_make_ದತ್ತಾಂಶ(dst_curr, d_spad), nb1, bctx->dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned, ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_ಬೈಟುಗಳು, 0);
        dma_ಸರತಿ_ತಳ್ಳು(dma_q, dma_make_ದತ್ತಾಂಶ(s0_spad, src0_curr), bctx->src0_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned, nb01, ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_ಬೈಟುಗಳು,
                                                             current_ಖಂಡ_ಗಾತ್ರ);
        ir_prefetch += current_ಖಂಡ_ಗಾತ್ರ;
        spad_idx ^= 1;
    }

    ರಚನೆ htp_thread_trace * tr = &octx->ctx->trace[ith];
    compute_scalar_t compute = (compute_scalar_t) bctx->compute;

    ಗಾಗಿ (uint32_t ir = ಆರಂಭ_ಅಡ್ಡಸಾಲು; ir < ಕೊನೆ_ಅಡ್ಡಸಾಲು; ) {
        uint32_t current_ಖಂಡ_ಗಾತ್ರ = calc_ಖಂಡ_ಗಾತ್ರ(bctx, ir, ಕೊನೆ_ಅಡ್ಡಸಾಲು, ne01, ne02);

        uint8_t * d_spad  = (uint8_t *) dma_ಸರತಿ_ತೆಗೆ(dma_q).src;
        uint8_t * s0_spad = (uint8_t *) dma_ಸರತಿ_ತೆಗೆ(dma_q).dst;

        uint32_t i03, i02, i01, rem;
        i03 = fastdiv(ir, &bctx->src0_dim12_div);
        rem = ir - i03 * (ne02 * ne01);
        i02 = fastdiv(rem, &bctx->src0_dim1_div);
        i01 = rem - i02 * ne01;

        uint32_t i13 = fastmodulo(i03, ne13, &bctx->src1_dim3_div);
        uint32_t i12 = fastmodulo(i02, ne12, &bctx->src1_dim2_div);
        uint32_t i11 = fastmodulo(i01, ne11, &bctx->src1_dim1_div);
        ಸ್ಥಿರಾಂಕ uint8_t * scalar_ಕೋಷ್ಟಕ = (ಸ್ಥಿರಾಂಕ uint8_t *) s1_ಕೋಷ್ಟಕ +
            i13 * plane_stride + i12 * ಅಡ್ಡಸಾಲು_stride;

        htp_trace_event_ಆರಂಭ(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);
        compute(d_spad, s0_spad, scalar_ಕೋಷ್ಟಕ, i11, ne11, current_ಖಂಡ_ಗಾತ್ರ,
                bctx->dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned, bctx->src0_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned, ne00);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);

        dma_addr_t dst_curr = dst->ದತ್ತಾಂಶ + i03 * nb3 + i02 * nb2 + i01 * nb1;
        dma_ಸರತಿ_ತಳ್ಳು(dma_q, dma_make_ದತ್ತಾಂಶ(dst_curr, d_spad), nb1, bctx->dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned, ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_ಬೈಟುಗಳು,
                                                                current_ಖಂಡ_ಗಾತ್ರ);

        ಆದರೆ (ir_prefetch < ಕೊನೆ_ಅಡ್ಡಸಾಲು) {
            uint32_t ಮುಂದಿನ_ಖಂಡ_ಗಾತ್ರ = calc_ಖಂಡ_ಗಾತ್ರ(bctx, ir_prefetch, ಕೊನೆ_ಅಡ್ಡಸಾಲು, ne01, ne02);
            uint32_t p03, p02, p01, prem;
            p03 = fastdiv(ir_prefetch, &bctx->src0_dim12_div);
            prem = ir_prefetch - p03 * (ne02 * ne01);
            p02 = fastdiv(prem, &bctx->src0_dim1_div);
            p01 = prem - p02 * ne01;
            dma_addr_t s0_ಮುಂದಿನ = src0->ದತ್ತಾಂಶ + p03 * nb03 + p02 * nb02 + p01 * nb01;
            dma_ಸರತಿ_ತಳ್ಳು(dma_q, dma_make_ದತ್ತಾಂಶ(s0_spad, s0_ಮುಂದಿನ), bctx->src0_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned, nb01, ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_ಬೈಟುಗಳು,
                                                               ಮುಂದಿನ_ಖಂಡ_ಗಾತ್ರ);
            ir_prefetch += ಮುಂದಿನ_ಖಂಡ_ಗಾತ್ರ;
        }
        ir += current_ಖಂಡ_ಗಾತ್ರ;
    }

    dma_ಸರತಿ_flush(dma_q);
}

// 2. Vector Same Shape (ne1x == ne0x) or Simple Broadcast
ಸ್ಥಿರ ಶೂನ್ಯ ದ್ವಿಮಾನ_thread_ಸದಿಶ_same_shape(ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ nth, ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ ith, ಶೂನ್ಯ * ದತ್ತಾಂಶ) {
    ರಚನೆ htp_ದ್ವಿಮಾನ_context * bctx = (ರಚನೆ htp_ದ್ವಿಮಾನ_context *) ದತ್ತಾಂಶ;
    ರಚನೆ htp_ops_context * octx = bctx->octx;
    htp_ದ್ವಿಮಾನ_preamble;

    ಸ್ಥಿರಾಂಕ uint32_t ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_ಬೈಟುಗಳು = bctx->ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_ಬೈಟುಗಳು;
    ಸ್ಥಿರಾಂಕ uint32_t ಆರಂಭ_ಅಡ್ಡಸಾಲು = bctx->ಅಡ್ಡಸಾಲು_ಆರಂಭ + bctx->nrows_ಪ್ರತಿ_thread * ith;
    ಸ್ಥಿರಾಂಕ uint32_t ಕೊನೆ_ಅಡ್ಡಸಾಲು   = MIN(ಆರಂಭ_ಅಡ್ಡಸಾಲು + bctx->nrows_ಪ್ರತಿ_thread, bctx->ಅಡ್ಡಸಾಲು_ಆರಂಭ + bctx->ಒಟ್ಟು_ಅಡ್ಡಸಾಲುಗಳು);
    ಆದರೆ (ಆರಂಭ_ಅಡ್ಡಸಾಲು >= ಕೊನೆ_ಅಡ್ಡಸಾಲು) ಹಿಂತಿರುಗಿ;

    FARF(ಮೇಲಿನ_2, "binary-same-shape: %d/%d (%u:%u) row-size %u (%u)",
         ith, nth, ಆರಂಭ_ಅಡ್ಡಸಾಲು, ಕೊನೆ_ಅಡ್ಡಸಾಲು, nb01, bctx->dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned);

    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_ದ್ವಿಮಾನ_vtcm_layout * layout = &bctx->vtcm_layout;
    uint8_t * src0_spad_ತಳ = VTCM_LAYOUT_PTR(uint8_t, bctx->vtcm_ತಳ, layout->off_src0) + (ith * layout->src0_ಬೈಟುಗಳು_ಪ್ರತಿ_thread);
    uint8_t * src1_spad_ತಳ = VTCM_LAYOUT_PTR(uint8_t, bctx->vtcm_ತಳ, layout->off_src1) + (ith * layout->src1_ಬೈಟುಗಳು_ಪ್ರತಿ_thread);
    uint8_t * dst_spad_ತಳ  = VTCM_LAYOUT_PTR(uint8_t, bctx->vtcm_ತಳ, layout->off_dst)  + (ith * layout->dst_ಬೈಟುಗಳು_ಪ್ರತಿ_thread);

    size_t src0_spad_half = layout->src0_spad_half_ಗಾತ್ರ;
    size_t src1_spad_half = layout->src1_spad_half_ಗಾತ್ರ;
    size_t dst_spad_half  = layout->dst_spad_half_ಗಾತ್ರ;

    dma_ಸರತಿ * dma_q = octx->ctx->dma[ith];
    uint32_t ir_prefetch = ಆರಂಭ_ಅಡ್ಡಸಾಲು;
    ಪೂರ್ಣಾಂಕ spad_idx = 0;

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ k = 0; k < 2 && ir_prefetch < ಕೊನೆ_ಅಡ್ಡಸಾಲು; k++) {
        uint32_t current_ಖಂಡ_ಗಾತ್ರ = calc_ಖಂಡ_ಗಾತ್ರ(bctx, ir_prefetch, ಕೊನೆ_ಅಡ್ಡಸಾಲು, ne01, ne02);
        uint32_t i03, i02, i01, rem;
        i03 = fastdiv(ir_prefetch, &bctx->src0_dim12_div);
        rem = ir_prefetch - i03 * (ne02 * ne01);
        i02 = fastdiv(rem, &bctx->src0_dim1_div);
        i01 = rem - i02 * ne01;

        uint32_t i13 = (ne13 == 1) ? 0 : i03;
        uint32_t i12 = (ne12 == 1) ? 0 : i02;
        uint32_t i11 = (ne11 == 1) ? 0 : i01;

        dma_addr_t src0_curr = src0->ದತ್ತಾಂಶ + i03 * nb03 + i02 * nb02 + i01 * nb01;
        dma_addr_t src1_curr = src1->ದತ್ತಾಂಶ + i13 * nb13 + i12 * nb12 + i11 * nb11;
        dma_addr_t dst_curr  = dst->ದತ್ತಾಂಶ  + i03 * nb3  + i02 * nb2  + i01 * nb1;

        uint8_t * s0_spad = src0_spad_ತಳ + spad_idx * src0_spad_half;
        uint8_t * s1_spad = src1_spad_ತಳ + spad_idx * src1_spad_half;
        uint8_t * d_spad  = dst_spad_ತಳ  + spad_idx * dst_spad_half;

        dma_ಸರತಿ_ತಳ್ಳು(dma_q, dma_make_ದತ್ತಾಂಶ(dst_curr, d_spad), nb1, bctx->dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned, ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_ಬೈಟುಗಳು, 0);
        dma_ಸರತಿ_ತಳ್ಳು(dma_q, dma_make_ದತ್ತಾಂಶ(s0_spad, src0_curr), bctx->src0_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned, nb01, ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_ಬೈಟುಗಳು, current_ಖಂಡ_ಗಾತ್ರ);
        dma_ಸರತಿ_ತಳ್ಳು(dma_q, dma_make_ದತ್ತಾಂಶ(s1_spad, src1_curr), bctx->src1_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned, nb11, ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_ಬೈಟುಗಳು, current_ಖಂಡ_ಗಾತ್ರ);
        ir_prefetch += current_ಖಂಡ_ಗಾತ್ರ;
        spad_idx ^= 1;
    }

    ರಚನೆ htp_thread_trace * tr = &octx->ctx->trace[ith];
    compute_same_shape_t compute = (compute_same_shape_t) bctx->compute;

    ಗಾಗಿ (uint32_t ir = ಆರಂಭ_ಅಡ್ಡಸಾಲು; ir < ಕೊನೆ_ಅಡ್ಡಸಾಲು; ) {
        uint32_t current_ಖಂಡ_ಗಾತ್ರ = calc_ಖಂಡ_ಗಾತ್ರ(bctx, ir, ಕೊನೆ_ಅಡ್ಡಸಾಲು, ne01, ne02);
        uint8_t * d_spad  = (uint8_t *) dma_ಸರತಿ_ತೆಗೆ(dma_q).src;
        uint8_t * s0_spad = (uint8_t *) dma_ಸರತಿ_ತೆಗೆ(dma_q).dst;
        uint8_t * s1_spad = (uint8_t *) dma_ಸರತಿ_ತೆಗೆ(dma_q).dst;

        htp_trace_event_ಆರಂಭ(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);
        compute(d_spad, s0_spad, s1_spad, current_ಖಂಡ_ಗಾತ್ರ,
                bctx->dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned, bctx->src0_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned, bctx->src1_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned, ne00);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);

        uint32_t i03, i02, i01, rem;
        i03 = fastdiv(ir, &bctx->src0_dim12_div);
        rem = ir - i03 * (ne02 * ne01);
        i02 = fastdiv(rem, &bctx->src0_dim1_div);
        i01 = rem - i02 * ne01;
        dma_addr_t dst_curr = dst->ದತ್ತಾಂಶ + i03 * nb3 + i02 * nb2 + i01 * nb1;
        dma_ಸರತಿ_ತಳ್ಳು(dma_q, dma_make_ದತ್ತಾಂಶ(dst_curr, d_spad), nb1, bctx->dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned, ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_ಬೈಟುಗಳು, current_ಖಂಡ_ಗಾತ್ರ);

        ಆದರೆ (ir_prefetch < ಕೊನೆ_ಅಡ್ಡಸಾಲು) {
            uint32_t ಮುಂದಿನ_ಖಂಡ_ಗಾತ್ರ = calc_ಖಂಡ_ಗಾತ್ರ(bctx, ir_prefetch, ಕೊನೆ_ಅಡ್ಡಸಾಲು, ne01, ne02);
            uint32_t p03, p02, p01, prem;
            p03 = fastdiv(ir_prefetch, &bctx->src0_dim12_div);
            prem = ir_prefetch - p03 * (ne02 * ne01);
            p02 = fastdiv(prem, &bctx->src0_dim1_div);
            p01 = prem - p02 * ne01;

            uint32_t p13 = (ne13 == 1) ? 0 : p03;
            uint32_t p12 = (ne12 == 1) ? 0 : p02;
            uint32_t p11 = (ne11 == 1) ? 0 : p01;

            dma_addr_t s0_ಮುಂದಿನ = src0->ದತ್ತಾಂಶ + p03 * nb03 + p02 * nb02 + p01 * nb01;
            dma_addr_t s1_ಮುಂದಿನ = src1->ದತ್ತಾಂಶ + p13 * nb13 + p12 * nb12 + p11 * nb11;

            dma_ಸರತಿ_ತಳ್ಳು(dma_q, dma_make_ದತ್ತಾಂಶ(s0_spad, s0_ಮುಂದಿನ), bctx->src0_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned, nb01, ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_ಬೈಟುಗಳು, ಮುಂದಿನ_ಖಂಡ_ಗಾತ್ರ);
            dma_ಸರತಿ_ತಳ್ಳು(dma_q, dma_make_ದತ್ತಾಂಶ(s1_spad, s1_ಮುಂದಿನ), bctx->src1_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned, nb11, ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_ಬೈಟುಗಳು, ಮುಂದಿನ_ಖಂಡ_ಗಾತ್ರ);

            ir_prefetch += ಮುಂದಿನ_ಖಂಡ_ಗಾತ್ರ;
        }
        ir += current_ಖಂಡ_ಗಾತ್ರ;
    }

    dma_ಸರತಿ_flush(dma_q);
}

// 3. Row Broadcast (ne11 == 1, ne12 == 1, single row src1)
ಸ್ಥಿರ ಶೂನ್ಯ ದ್ವಿಮಾನ_thread_ಸದಿಶ_ಅಡ್ಡಸಾಲು_ಪ್ರಸಾರ(ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ nth, ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ ith, ಶೂನ್ಯ * ದತ್ತಾಂಶ) {
    ರಚನೆ htp_ದ್ವಿಮಾನ_context * bctx = (ರಚನೆ htp_ದ್ವಿಮಾನ_context *) ದತ್ತಾಂಶ;
    ರಚನೆ htp_ops_context * octx = bctx->octx;
    htp_ದ್ವಿಮಾನ_preamble;

    ಸ್ಥಿರಾಂಕ uint32_t ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_ಬೈಟುಗಳು = bctx->ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_ಬೈಟುಗಳು;
    ಸ್ಥಿರಾಂಕ uint32_t ಆರಂಭ_ಅಡ್ಡಸಾಲು  = bctx->ಅಡ್ಡಸಾಲು_ಆರಂಭ + bctx->nrows_ಪ್ರತಿ_thread * ith;
    ಸ್ಥಿರಾಂಕ uint32_t ಕೊನೆ_ಅಡ್ಡಸಾಲು    = MIN(ಆರಂಭ_ಅಡ್ಡಸಾಲು + bctx->nrows_ಪ್ರತಿ_thread, bctx->ಅಡ್ಡಸಾಲು_ಆರಂಭ + bctx->ಒಟ್ಟು_ಅಡ್ಡಸಾಲುಗಳು);
    ಆದರೆ (ಆರಂಭ_ಅಡ್ಡಸಾಲು >= ಕೊನೆ_ಅಡ್ಡಸಾಲು) ಹಿಂತಿರುಗಿ;

    FARF(ಮೇಲಿನ_2, "binary-row-bcast: %d/%d (%u:%u) row-size %u (%u)",
         ith, nth, ಆರಂಭ_ಅಡ್ಡಸಾಲು, ಕೊನೆ_ಅಡ್ಡಸಾಲು, nb01, bctx->dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned);

    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_ದ್ವಿಮಾನ_vtcm_layout * layout = &bctx->vtcm_layout;
    uint8_t * src0_spad_ತಳ = VTCM_LAYOUT_PTR(uint8_t, bctx->vtcm_ತಳ, layout->off_src0) + (ith * layout->src0_ಬೈಟುಗಳು_ಪ್ರತಿ_thread);
    uint8_t * dst_spad_ತಳ  = VTCM_LAYOUT_PTR(uint8_t, bctx->vtcm_ತಳ, layout->off_dst)  + (ith * layout->dst_ಬೈಟುಗಳು_ಪ್ರತಿ_thread);

    size_t src0_spad_half = layout->src0_spad_half_ಗಾತ್ರ;
    size_t dst_spad_half  = layout->dst_spad_half_ಗಾತ್ರ;

    dma_ಸರತಿ * dma_q = octx->ctx->dma[ith];
    uint32_t ir_prefetch = ಆರಂಭ_ಅಡ್ಡಸಾಲು;
    ಪೂರ್ಣಾಂಕ spad_idx = 0;

    ಶೂನ್ಯ * s1_ptr = VTCM_LAYOUT_PTR(ಶೂನ್ಯ, bctx->vtcm_ತಳ, layout->off_src1);

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ k = 0; k < 2 && ir_prefetch < ಕೊನೆ_ಅಡ್ಡಸಾಲು; k++) {
        uint32_t current_ಖಂಡ_ಗಾತ್ರ = calc_ಖಂಡ_ಗಾತ್ರ(bctx, ir_prefetch, ಕೊನೆ_ಅಡ್ಡಸಾಲು, ne01, ne02);
        uint32_t i03 = fastdiv(ir_prefetch, &bctx->src0_dim12_div);
        uint32_t rem = ir_prefetch - i03 * (ne02 * ne01);
        uint32_t i02 = fastdiv(rem, &bctx->src0_dim1_div);
        uint32_t i01 = rem - i02 * ne01;

        dma_addr_t src0_curr = src0->ದತ್ತಾಂಶ + i03 * nb03 + i02 * nb02 + i01 * nb01;
        dma_addr_t dst_curr  = dst->ದತ್ತಾಂಶ  + i03 * nb3  + i02 * nb2  + i01 * nb1;

        uint8_t * s0_spad = src0_spad_ತಳ + spad_idx * src0_spad_half;
        uint8_t * d_spad  = dst_spad_ತಳ  + spad_idx * dst_spad_half;

        dma_ಸರತಿ_ತಳ್ಳು(dma_q, dma_make_ದತ್ತಾಂಶ(dst_curr, d_spad), nb1, bctx->dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned, ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_ಬೈಟುಗಳು, 0);
        dma_ಸರತಿ_ತಳ್ಳು(dma_q, dma_make_ದತ್ತಾಂಶ(s0_spad, src0_curr), bctx->src0_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned, nb01, ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_ಬೈಟುಗಳು, current_ಖಂಡ_ಗಾತ್ರ);
        ir_prefetch += current_ಖಂಡ_ಗಾತ್ರ;
        spad_idx ^= 1;
    }

    ರಚನೆ htp_thread_trace * tr = &octx->ctx->trace[ith];
    compute_ಅಡ್ಡಸಾಲು_bcast_t compute = (compute_ಅಡ್ಡಸಾಲು_bcast_t) bctx->compute;

    ಗಾಗಿ (uint32_t ir = ಆರಂಭ_ಅಡ್ಡಸಾಲು; ir < ಕೊನೆ_ಅಡ್ಡಸಾಲು; ) {
        uint32_t current_ಖಂಡ_ಗಾತ್ರ = calc_ಖಂಡ_ಗಾತ್ರ(bctx, ir, ಕೊನೆ_ಅಡ್ಡಸಾಲು, ne01, ne02);
        uint8_t * d_spad  = (uint8_t *) dma_ಸರತಿ_ತೆಗೆ(dma_q).src;
        uint8_t * s0_spad = (uint8_t *) dma_ಸರತಿ_ತೆಗೆ(dma_q).dst;

        htp_trace_event_ಆರಂಭ(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);
        compute(d_spad, s0_spad, (ಸ್ಥಿರಾಂಕ uint8_t *)s1_ptr, current_ಖಂಡ_ಗಾತ್ರ,
                bctx->dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned, bctx->src0_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned, ne00);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);

        uint32_t i03 = fastdiv(ir, &bctx->src0_dim12_div);
        uint32_t rem = ir - i03 * (ne02 * ne01);
        uint32_t i02 = fastdiv(rem, &bctx->src0_dim1_div);
        uint32_t i01 = rem - i02 * ne01;
        dma_addr_t dst_curr = dst->ದತ್ತಾಂಶ + i03 * nb3 + i02 * nb2 + i01 * nb1;
        dma_ಸರತಿ_ತಳ್ಳು(dma_q, dma_make_ದತ್ತಾಂಶ(dst_curr, d_spad), nb1, bctx->dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned, ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_ಬೈಟುಗಳು, current_ಖಂಡ_ಗಾತ್ರ);

        ಆದರೆ (ir_prefetch < ಕೊನೆ_ಅಡ್ಡಸಾಲು) {
            uint32_t ಮುಂದಿನ_ಖಂಡ_ಗಾತ್ರ = calc_ಖಂಡ_ಗಾತ್ರ(bctx, ir_prefetch, ಕೊನೆ_ಅಡ್ಡಸಾಲು, ne01, ne02);
            uint32_t p03, p02, p01, prem;
            p03  = fastdiv(ir_prefetch, &bctx->src0_dim12_div);
            prem = ir_prefetch - p03 * (ne02 * ne01);
            p02  = fastdiv(prem, &bctx->src0_dim1_div);
            p01  = prem - p02 * ne01;
            dma_addr_t s0_ಮುಂದಿನ = src0->ದತ್ತಾಂಶ + p03 * nb03 + p02 * nb02 + p01 * nb01;
            dma_ಸರತಿ_ತಳ್ಳು(dma_q, dma_make_ದತ್ತಾಂಶ(s0_spad, s0_ಮುಂದಿನ), bctx->src0_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned, nb01, ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_ಬೈಟುಗಳು, ಮುಂದಿನ_ಖಂಡ_ಗಾತ್ರ);
            ir_prefetch += ಮುಂದಿನ_ಖಂಡ_ಗಾತ್ರ;
        }
        ir += current_ಖಂಡ_ಗಾತ್ರ;
    }

    dma_ಸರತಿ_flush(dma_q);
}

// 4. Vector Complex (ne10 == ne00, complex broadcast)
ಸ್ಥಿರ ಶೂನ್ಯ ದ್ವಿಮಾನ_thread_ಸದಿಶ_complex(ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ nth, ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ ith, ಶೂನ್ಯ * ದತ್ತಾಂಶ) {
    ರಚನೆ htp_ದ್ವಿಮಾನ_context * bctx = (ರಚನೆ htp_ದ್ವಿಮಾನ_context *) ದತ್ತಾಂಶ;
    ರಚನೆ htp_ops_context * octx = bctx->octx;
    htp_ದ್ವಿಮಾನ_preamble;

    ಸ್ಥಿರಾಂಕ uint32_t ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_ಬೈಟುಗಳು = bctx->ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_ಬೈಟುಗಳು;
    ಸ್ಥಿರಾಂಕ uint32_t ಆರಂಭ_ಅಡ್ಡಸಾಲು  = bctx->ಅಡ್ಡಸಾಲು_ಆರಂಭ + bctx->nrows_ಪ್ರತಿ_thread * ith;
    ಸ್ಥಿರಾಂಕ uint32_t ಕೊನೆ_ಅಡ್ಡಸಾಲು    = MIN(ಆರಂಭ_ಅಡ್ಡಸಾಲು + bctx->nrows_ಪ್ರತಿ_thread, bctx->ಅಡ್ಡಸಾಲು_ಆರಂಭ + bctx->ಒಟ್ಟು_ಅಡ್ಡಸಾಲುಗಳು);
    ಆದರೆ (ಆರಂಭ_ಅಡ್ಡಸಾಲು >= ಕೊನೆ_ಅಡ್ಡಸಾಲು) ಹಿಂತಿರುಗಿ;

    FARF(ಮೇಲಿನ_2, "binary-complex: %d/%d (%u:%u) row-size %u (%u)",
         ith, nth, ಆರಂಭ_ಅಡ್ಡಸಾಲು, ಕೊನೆ_ಅಡ್ಡಸಾಲು, nb01, bctx->dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned);

    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_ದ್ವಿಮಾನ_vtcm_layout * layout = &bctx->vtcm_layout;
    uint8_t * src0_spad_ತಳ = VTCM_LAYOUT_PTR(uint8_t, bctx->vtcm_ತಳ, layout->off_src0) + (ith * layout->src0_ಬೈಟುಗಳು_ಪ್ರತಿ_thread);
    uint8_t * dst_spad_ತಳ  = VTCM_LAYOUT_PTR(uint8_t, bctx->vtcm_ತಳ, layout->off_dst)  + (ith * layout->dst_ಬೈಟುಗಳು_ಪ್ರತಿ_thread);
    size_t src0_spad_half    = layout->src0_spad_half_ಗಾತ್ರ;
    size_t dst_spad_half     = layout->dst_spad_half_ಗಾತ್ರ;
    ಸ್ಥಿರಾಂಕ uint8_t * vtcm_src1 = VTCM_LAYOUT_PTR(ಸ್ಥಿರಾಂಕ uint8_t, bctx->vtcm_ತಳ, layout->off_src1);

    ಸ್ಥಿರಾಂಕ uint32_t elem_ಗಾತ್ರ = (src0->type == HTP_TYPE_F32) ? ಗಾತ್ರ(ದಶಮಾಂಶ) : ಗಾತ್ರ(_Float16);
    ಸ್ಥಿರಾಂಕ size_t vtcm_ಅಡ್ಡಸಾಲು_stride   = (size_t) ne10 * elem_ಗಾತ್ರ;
    ಸ್ಥಿರಾಂಕ size_t vtcm_plane_stride = (size_t) ne11 * vtcm_ಅಡ್ಡಸಾಲು_stride;
    ಸ್ಥಿರಾಂಕ size_t vtcm_vol_stride   = (size_t) ne12 * vtcm_plane_stride;

    dma_ಸರತಿ * dma_q = octx->ctx->dma[ith];
    uint32_t ir_prefetch = ಆರಂಭ_ಅಡ್ಡಸಾಲು;
    ಪೂರ್ಣಾಂಕ spad_idx = 0;

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ k = 0; k < 2 && ir_prefetch < ಕೊನೆ_ಅಡ್ಡಸಾಲು; k++) {
        uint32_t current_ಖಂಡ_ಗಾತ್ರ = calc_ಖಂಡ_ಗಾತ್ರ(bctx, ir_prefetch, ಕೊನೆ_ಅಡ್ಡಸಾಲು, ne01, ne02);
        uint32_t i03 = fastdiv(ir_prefetch, &bctx->src0_dim12_div);
        uint32_t rem = ir_prefetch - i03 * (ne02 * ne01);
        uint32_t i02 = fastdiv(rem, &bctx->src0_dim1_div);
        uint32_t i01 = rem - i02 * ne01;

        dma_addr_t src0_curr = src0->ದತ್ತಾಂಶ + i03 * nb03 + i02 * nb02 + i01 * nb01;
        dma_addr_t dst_curr  = dst->ದತ್ತಾಂಶ  + i03 * nb3  + i02 * nb2  + i01 * nb1;

        uint8_t * s0_spad = src0_spad_ತಳ + spad_idx * src0_spad_half;
        uint8_t * d_spad  = dst_spad_ತಳ  + spad_idx * dst_spad_half;

        dma_ಸರತಿ_ತಳ್ಳು(dma_q, dma_make_ದತ್ತಾಂಶ(dst_curr, d_spad), nb1, bctx->dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned, ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_ಬೈಟುಗಳು, 0);
        dma_ಸರತಿ_ತಳ್ಳು(dma_q, dma_make_ದತ್ತಾಂಶ(s0_spad, src0_curr), bctx->src0_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned, nb01, ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_ಬೈಟುಗಳು, current_ಖಂಡ_ಗಾತ್ರ);
        ir_prefetch += current_ಖಂಡ_ಗಾತ್ರ;
        spad_idx ^= 1;
    }

    ರಚನೆ htp_thread_trace * tr = &octx->ctx->trace[ith];
    compute_complex_t compute = (compute_complex_t) bctx->compute;

    ಗಾಗಿ (uint32_t ir = ಆರಂಭ_ಅಡ್ಡಸಾಲು; ir < ಕೊನೆ_ಅಡ್ಡಸಾಲು; ) {
        uint32_t current_ಖಂಡ_ಗಾತ್ರ = calc_ಖಂಡ_ಗಾತ್ರ(bctx, ir, ಕೊನೆ_ಅಡ್ಡಸಾಲು, ne01, ne02);
        uint8_t * d_spad = (uint8_t *) dma_ಸರತಿ_ತೆಗೆ(dma_q).src;
        uint8_t * s0_spad = (uint8_t *) dma_ಸರತಿ_ತೆಗೆ(dma_q).dst;

        uint32_t i03 = fastdiv(ir, &bctx->src0_dim12_div);
        uint32_t rem = ir - i03 * (ne02 * ne01);
        uint32_t i02 = fastdiv(rem, &bctx->src0_dim1_div);
        uint32_t i01 = rem - i02 * ne01;

        uint32_t i13 = fastmodulo(i03, ne13, &bctx->src1_dim3_div);
        uint32_t i12 = fastmodulo(i02, ne12, &bctx->src1_dim2_div);
        ಸ್ಥಿರಾಂಕ uint8_t * src1_plane = vtcm_src1 + i13 * vtcm_vol_stride + i12 * vtcm_plane_stride;

        htp_trace_event_ಆರಂಭ(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);
        compute(d_spad, s0_spad, src1_plane, i01, ne11, &bctx->src1_dim1_div, vtcm_ಅಡ್ಡಸಾಲು_stride,
                current_ಖಂಡ_ಗಾತ್ರ, bctx->dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned, bctx->src0_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned, ne00);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);

        dma_addr_t dst_curr = dst->ದತ್ತಾಂಶ + i03 * nb3 + i02 * nb2 + i01 * nb1;
        dma_ಸರತಿ_ತಳ್ಳು(dma_q, dma_make_ದತ್ತಾಂಶ(dst_curr, d_spad), nb1, bctx->dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned, ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_ಬೈಟುಗಳು, current_ಖಂಡ_ಗಾತ್ರ);

        ಆದರೆ (ir_prefetch < ಕೊನೆ_ಅಡ್ಡಸಾಲು) {
            uint32_t ಮುಂದಿನ_ಖಂಡ_ಗಾತ್ರ = calc_ಖಂಡ_ಗಾತ್ರ(bctx, ir_prefetch, ಕೊನೆ_ಅಡ್ಡಸಾಲು, ne01, ne02);
            uint32_t p03, p02, p01, prem;
            p03  = fastdiv(ir_prefetch, &bctx->src0_dim12_div);
            prem = ir_prefetch - p03 * (ne02 * ne01);
            p02  = fastdiv(prem, &bctx->src0_dim1_div);
            p01  = prem - p02 * ne01;
            dma_addr_t s0_ಮುಂದಿನ = src0->ದತ್ತಾಂಶ + p03 * nb03 + p02 * nb02 + p01 * nb01;
            dma_ಸರತಿ_ತಳ್ಳು(dma_q, dma_make_ದತ್ತಾಂಶ(s0_spad, s0_ಮುಂದಿನ), bctx->src0_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned, nb01, ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_ಬೈಟುಗಳು, ಮುಂದಿನ_ಖಂಡ_ಗಾತ್ರ);
            ir_prefetch += ಮುಂದಿನ_ಖಂಡ_ಗಾತ್ರ;
        }
        ir += current_ಖಂಡ_ಗಾತ್ರ;
    }

    dma_ಸರತಿ_flush(dma_q);
}

// 5. Element Repeat (ne10 != ne00)
ಸ್ಥಿರ ಶೂನ್ಯ ದ್ವಿಮಾನ_thread_element_repeat(ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ nth, ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ ith, ಶೂನ್ಯ * ದತ್ತಾಂಶ) {
    ರಚನೆ htp_ದ್ವಿಮಾನ_context * bctx = (ರಚನೆ htp_ದ್ವಿಮಾನ_context *) ದತ್ತಾಂಶ;
    ರಚನೆ htp_ops_context * octx = bctx->octx;
    htp_ದ್ವಿಮಾನ_preamble;

    ಸ್ಥಿರಾಂಕ uint32_t ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_ಬೈಟುಗಳು = bctx->ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_ಬೈಟುಗಳು;
    ಸ್ಥಿರಾಂಕ uint32_t ಆರಂಭ_ಅಡ್ಡಸಾಲು  = bctx->ಅಡ್ಡಸಾಲು_ಆರಂಭ + bctx->nrows_ಪ್ರತಿ_thread * ith;
    ಸ್ಥಿರಾಂಕ uint32_t ಕೊನೆ_ಅಡ್ಡಸಾಲು    = MIN(ಆರಂಭ_ಅಡ್ಡಸಾಲು + bctx->nrows_ಪ್ರತಿ_thread, bctx->ಅಡ್ಡಸಾಲು_ಆರಂಭ + bctx->ಒಟ್ಟು_ಅಡ್ಡಸಾಲುಗಳು);
    ಆದರೆ (ಆರಂಭ_ಅಡ್ಡಸಾಲು >= ಕೊನೆ_ಅಡ್ಡಸಾಲು) ಹಿಂತಿರುಗಿ;

    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_ದ್ವಿಮಾನ_vtcm_layout * layout = &bctx->vtcm_layout;
    uint8_t * src0_spad_ತಳ = VTCM_LAYOUT_PTR(uint8_t, bctx->vtcm_ತಳ, layout->off_src0) + (ith * layout->src0_ಬೈಟುಗಳು_ಪ್ರತಿ_thread);
    uint8_t * dst_spad_ತಳ  = VTCM_LAYOUT_PTR(uint8_t, bctx->vtcm_ತಳ, layout->off_dst)  + (ith * layout->dst_ಬೈಟುಗಳು_ಪ್ರತಿ_thread);
    size_t src0_spad_half    = layout->src0_spad_half_ಗಾತ್ರ;
    size_t dst_spad_half     = layout->dst_spad_half_ಗಾತ್ರ;
    ಸ್ಥಿರಾಂಕ uint8_t * vtcm_src1 = VTCM_LAYOUT_PTR(ಸ್ಥಿರಾಂಕ uint8_t, bctx->vtcm_ತಳ, layout->off_src1);

    ಸ್ಥಿರಾಂಕ uint32_t elem_ಗಾತ್ರ = (src0->type == HTP_TYPE_F32) ? ಗಾತ್ರ(ದಶಮಾಂಶ) : ಗಾತ್ರ(_Float16);
    ಸ್ಥಿರಾಂಕ size_t vtcm_ಅಡ್ಡಸಾಲು_stride   = (size_t) ne10 * elem_ಗಾತ್ರ;
    ಸ್ಥಿರಾಂಕ size_t vtcm_plane_stride = (size_t) ne11 * vtcm_ಅಡ್ಡಸಾಲು_stride;
    ಸ್ಥಿರಾಂಕ size_t vtcm_vol_stride   = (size_t) ne12 * vtcm_plane_stride;

    FARF(ಮೇಲಿನ_2, "binary-repeat: %d/%d (%u:%u) row-size %u (%u)",
         ith, nth, ಆರಂಭ_ಅಡ್ಡಸಾಲು, ಕೊನೆ_ಅಡ್ಡಸಾಲು, nb01, bctx->dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned);

    dma_ಸರತಿ * dma_q = octx->ctx->dma[ith];
    uint32_t ir_prefetch = ಆರಂಭ_ಅಡ್ಡಸಾಲು;
    ಪೂರ್ಣಾಂಕ spad_idx = 0;

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ k = 0; k < 2 && ir_prefetch < ಕೊನೆ_ಅಡ್ಡಸಾಲು; k++) {
        uint32_t current_ಖಂಡ_ಗಾತ್ರ = calc_ಖಂಡ_ಗಾತ್ರ(bctx, ir_prefetch, ಕೊನೆ_ಅಡ್ಡಸಾಲು, ne01, ne02);
        uint32_t i03 = fastdiv(ir_prefetch, &bctx->src0_dim12_div);
        uint32_t rem = ir_prefetch - i03 * (ne02 * ne01);
        uint32_t i02 = fastdiv(rem, &bctx->src0_dim1_div);
        uint32_t i01 = rem - i02 * ne01;

        dma_addr_t src0_curr = src0->ದತ್ತಾಂಶ + i03 * nb03 + i02 * nb02 + i01 * nb01;
        dma_addr_t dst_curr  = dst->ದತ್ತಾಂಶ  + i03 * nb3  + i02 * nb2  + i01 * nb1;

        uint8_t * s0_spad = src0_spad_ತಳ + spad_idx * src0_spad_half;
        uint8_t * d_spad  = dst_spad_ತಳ  + spad_idx * dst_spad_half;

        dma_ಸರತಿ_ತಳ್ಳು(dma_q, dma_make_ದತ್ತಾಂಶ(dst_curr, d_spad), nb1, bctx->dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned, ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_ಬೈಟುಗಳು, 0);
        dma_ಸರತಿ_ತಳ್ಳು(dma_q, dma_make_ದತ್ತಾಂಶ(s0_spad, src0_curr), bctx->src0_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned, nb01, ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_ಬೈಟುಗಳು, current_ಖಂಡ_ಗಾತ್ರ);
        ir_prefetch += current_ಖಂಡ_ಗಾತ್ರ;
        spad_idx ^= 1;
    }

    ರಚನೆ htp_thread_trace * tr = &octx->ctx->trace[ith];
    compute_repeat_t compute = (compute_repeat_t) bctx->compute;

    ಗಾಗಿ (uint32_t ir = ಆರಂಭ_ಅಡ್ಡಸಾಲು; ir < ಕೊನೆ_ಅಡ್ಡಸಾಲು; ) {
        uint32_t current_ಖಂಡ_ಗಾತ್ರ = calc_ಖಂಡ_ಗಾತ್ರ(bctx, ir, ಕೊನೆ_ಅಡ್ಡಸಾಲು, ne01, ne02);
        uint8_t * d_spad  = (uint8_t *) dma_ಸರತಿ_ತೆಗೆ(dma_q).src;
        uint8_t * s0_spad = (uint8_t *) dma_ಸರತಿ_ತೆಗೆ(dma_q).dst;

        uint32_t i03 = fastdiv(ir, &bctx->src0_dim12_div);
        uint32_t rem = ir - i03 * (ne02 * ne01);
        uint32_t i02 = fastdiv(rem, &bctx->src0_dim1_div);
        uint32_t i01 = rem - i02 * ne01;

        uint32_t i13 = fastmodulo(i03, ne13, &bctx->src1_dim3_div);
        uint32_t i12 = fastmodulo(i02, ne12, &bctx->src1_dim2_div);
        ಸ್ಥಿರಾಂಕ uint8_t * src1_plane = vtcm_src1 + i13 * vtcm_vol_stride + i12 * vtcm_plane_stride;

        htp_trace_event_ಆರಂಭ(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);
        compute(d_spad, s0_spad, src1_plane, i01, ne11, &bctx->src1_dim1_div, vtcm_ಅಡ್ಡಸಾಲು_stride,
                current_ಖಂಡ_ಗಾತ್ರ, bctx->dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned, bctx->src0_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned, ne00, ne10);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);

        dma_addr_t dst_curr = dst->ದತ್ತಾಂಶ + i03 * nb3 + i02 * nb2 + i01 * nb1;
        dma_ಸರತಿ_ತಳ್ಳು(dma_q, dma_make_ದತ್ತಾಂಶ(dst_curr, d_spad), nb1, bctx->dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned, ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_ಬೈಟುಗಳು, current_ಖಂಡ_ಗಾತ್ರ);

        ಆದರೆ (ir_prefetch < ಕೊನೆ_ಅಡ್ಡಸಾಲು) {
            uint32_t ಮುಂದಿನ_ಖಂಡ_ಗಾತ್ರ = calc_ಖಂಡ_ಗಾತ್ರ(bctx, ir_prefetch, ಕೊನೆ_ಅಡ್ಡಸಾಲು, ne01, ne02);
            uint32_t p03, p02, p01, prem;
            p03  = fastdiv(ir_prefetch, &bctx->src0_dim12_div);
            prem = ir_prefetch - p03 * (ne02 * ne01);
            p02  = fastdiv(prem, &bctx->src0_dim1_div);
            p01  = prem - p02 * ne01;
            dma_addr_t s0_ಮುಂದಿನ = src0->ದತ್ತಾಂಶ + p03 * nb03 + p02 * nb02 + p01 * nb01;
            dma_ಸರತಿ_ತಳ್ಳು(dma_q, dma_make_ದತ್ತಾಂಶ(s0_spad, s0_ಮುಂದಿನ), bctx->src0_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned, nb01, ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_ಬೈಟುಗಳು, ಮುಂದಿನ_ಖಂಡ_ಗಾತ್ರ);
            ir_prefetch += ಮುಂದಿನ_ಖಂಡ_ಗಾತ್ರ;
        }
        ir += current_ಖಂಡ_ಗಾತ್ರ;
    }

    dma_ಸರತಿ_flush(dma_q);
}

// 6. ADD_ID (src1 gathered via src2 indices)
ಸ್ಥಿರ ಶೂನ್ಯ ದ್ವಿಮಾನ_thread_ಕೂಡಿಸು_id_f32(ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ nth, ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ ith, ಶೂನ್ಯ * ದತ್ತಾಂಶ) {
    ರಚನೆ htp_ದ್ವಿಮಾನ_context * bctx = (ರಚನೆ htp_ದ್ವಿಮಾನ_context *) ದತ್ತಾಂಶ;
    ರಚನೆ htp_ops_context * octx = bctx->octx;

    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * src0 = octx->src[0];
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * src1 = octx->src[1];
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * dst  = octx->dst;

    ಸ್ಥಿರಾಂಕ uint32_t ne00 = src0->ne[0];
    ಸ್ಥಿರಾಂಕ uint32_t ne01 = src0->ne[1];
    ಸ್ಥಿರಾಂಕ uint32_t ne02 = src0->ne[2];
    ಸ್ಥಿರಾಂಕ uint32_t ne03 = src0->ne[3];

    ಸ್ಥಿರಾಂಕ uint32_t nb01 = src0->nb[1];
    ಸ್ಥಿರಾಂಕ uint32_t nb02 = src0->nb[2];
    ಸ್ಥಿರಾಂಕ uint32_t nb03 = src0->nb[3];
    ಸ್ಥಿರಾಂಕ uint32_t src1_stride = bctx->src1_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned;

    ಸ್ಥಿರಾಂಕ uint32_t nb1 = dst->nb[1];
    ಸ್ಥಿರಾಂಕ uint32_t nb2 = dst->nb[2];
    ಸ್ಥಿರಾಂಕ uint32_t nb3 = dst->nb[3];

    ಸ್ಥಿರಾಂಕ uint32_t ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_ಬೈಟುಗಳು = bctx->ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_ಬೈಟುಗಳು;
    ಸ್ಥಿರಾಂಕ uint32_t ಆರಂಭ_ಅಡ್ಡಸಾಲು = bctx->ಅಡ್ಡಸಾಲು_ಆರಂಭ + bctx->nrows_ಪ್ರತಿ_thread * ith;
    ಸ್ಥಿರಾಂಕ uint32_t ಕೊನೆ_ಅಡ್ಡಸಾಲು   = MIN(ಆರಂಭ_ಅಡ್ಡಸಾಲು + bctx->nrows_ಪ್ರತಿ_thread, bctx->ಅಡ್ಡಸಾಲು_ಆರಂಭ + bctx->ಒಟ್ಟು_ಅಡ್ಡಸಾಲುಗಳು);
    ಆದರೆ (ಆರಂಭ_ಅಡ್ಡಸಾಲು >= ಕೊನೆ_ಅಡ್ಡಸಾಲು) ಹಿಂತಿರುಗಿ;

    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_ದ್ವಿಮಾನ_vtcm_layout * layout = &bctx->vtcm_layout;
    uint8_t * src0_spad_ತಳ = VTCM_LAYOUT_PTR(uint8_t, bctx->vtcm_ತಳ, layout->off_src0) + (ith * layout->src0_ಬೈಟುಗಳು_ಪ್ರತಿ_thread);
    uint8_t * dst_spad_ತಳ  = VTCM_LAYOUT_PTR(uint8_t, bctx->vtcm_ತಳ, layout->off_dst)  + (ith * layout->dst_ಬೈಟುಗಳು_ಪ್ರತಿ_thread);
    ಸ್ಥಿರಾಂಕ uint8_t * vtcm_src1 = VTCM_LAYOUT_PTR(ಸ್ಥಿರಾಂಕ uint8_t, bctx->vtcm_ತಳ, layout->off_src1);
    ಸ್ಥಿರಾಂಕ int32_t * vtcm_src2 = VTCM_LAYOUT_PTR(ಸ್ಥಿರಾಂಕ int32_t, bctx->vtcm_ತಳ, layout->off_src2);
    size_t src0_spad_half    = layout->src0_spad_half_ಗಾತ್ರ;
    size_t dst_spad_half     = layout->dst_spad_half_ಗಾತ್ರ;

    dma_ಸರತಿ * dma_q = octx->ctx->dma[ith];
    uint32_t ir_prefetch = ಆರಂಭ_ಅಡ್ಡಸಾಲು;
    ಪೂರ್ಣಾಂಕ spad_idx = 0;

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ k = 0; k < 2 && ir_prefetch < ಕೊನೆ_ಅಡ್ಡಸಾಲು; k++) {
        uint32_t current_ಖಂಡ_ಗಾತ್ರ = calc_ಖಂಡ_ಗಾತ್ರ(bctx, ir_prefetch, ಕೊನೆ_ಅಡ್ಡಸಾಲು, ne01, ne02);
        uint32_t i03 = fastdiv(ir_prefetch, &bctx->src0_dim12_div);
        uint32_t rem = ir_prefetch - i03 * (ne02 * ne01);
        uint32_t i02 = fastdiv(rem, &bctx->src0_dim1_div);
        uint32_t i01 = rem - i02 * ne01;

        dma_addr_t src0_curr = src0->ದತ್ತಾಂಶ + i03 * nb03 + i02 * nb02 + i01 * nb01;
        dma_addr_t dst_curr  = dst->ದತ್ತಾಂಶ  + i03 * nb3  + i02 * nb2  + i01 * nb1;

        uint8_t * s0_spad = src0_spad_ತಳ + spad_idx * src0_spad_half;
        uint8_t * d_spad  = dst_spad_ತಳ  + spad_idx * dst_spad_half;

        dma_ಸರತಿ_ತಳ್ಳು(dma_q, dma_make_ದತ್ತಾಂಶ(dst_curr, d_spad), nb1, bctx->dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned, ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_ಬೈಟುಗಳು, 0);
        dma_ಸರತಿ_ತಳ್ಳು(dma_q, dma_make_ದತ್ತಾಂಶ(s0_spad, src0_curr), bctx->src0_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned, nb01, ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_ಬೈಟುಗಳು, current_ಖಂಡ_ಗಾತ್ರ);
        ir_prefetch += current_ಖಂಡ_ಗಾತ್ರ;
        spad_idx ^= 1;
    }

    ರಚನೆ htp_thread_trace * tr = &octx->ctx->trace[ith];

    ಗಾಗಿ (uint32_t ir = ಆರಂಭ_ಅಡ್ಡಸಾಲು; ir < ಕೊನೆ_ಅಡ್ಡಸಾಲು; ) {
        uint32_t current_ಖಂಡ_ಗಾತ್ರ = calc_ಖಂಡ_ಗಾತ್ರ(bctx, ir, ಕೊನೆ_ಅಡ್ಡಸಾಲು, ne01, ne02);
        uint8_t * d_spad = (uint8_t *) dma_ಸರತಿ_ತೆಗೆ(dma_q).src;
        uint8_t * s0_spad = (uint8_t *) dma_ಸರತಿ_ತೆಗೆ(dma_q).dst;

        uint32_t i03 = fastdiv(ir, &bctx->src0_dim12_div);
        uint32_t rem = ir - i03 * (ne02 * ne01);
        uint32_t i02 = fastdiv(rem, &bctx->src0_dim1_div);
        uint32_t i01 = rem - i02 * ne01;

        htp_trace_event_ಆರಂಭ(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);
        compute_ಕೂಡಿಸು_id_t compute = (compute_ಕೂಡಿಸು_id_t) bctx->compute;
        compute(d_spad, s0_spad, vtcm_src1, vtcm_src2,
                i01, i02, ne01, src1_stride,
                current_ಖಂಡ_ಗಾತ್ರ, bctx->dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned, bctx->src0_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned, ne00);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);

        dma_addr_t dst_curr = dst->ದತ್ತಾಂಶ + i03 * nb3 + i02 * nb2 + i01 * nb1;
        dma_ಸರತಿ_ತಳ್ಳು(dma_q, dma_make_ದತ್ತಾಂಶ(dst_curr, d_spad), nb1, bctx->dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned, ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_ಬೈಟುಗಳು, current_ಖಂಡ_ಗಾತ್ರ);

        ಆದರೆ (ir_prefetch < ಕೊನೆ_ಅಡ್ಡಸಾಲು) {
             uint32_t ಮುಂದಿನ_ಖಂಡ_ಗಾತ್ರ = calc_ಖಂಡ_ಗಾತ್ರ(bctx, ir_prefetch, ಕೊನೆ_ಅಡ್ಡಸಾಲು, ne01, ne02);
             uint32_t p03, p02, p01, prem;
             p03  = fastdiv(ir_prefetch, &bctx->src0_dim12_div);
             prem = ir_prefetch - p03 * (ne02 * ne01);
             p02  = fastdiv(prem, &bctx->src0_dim1_div);
             p01  = prem - p02 * ne01;
             dma_addr_t s0_ಮುಂದಿನ = src0->ದತ್ತಾಂಶ + p03 * nb03 + p02 * nb02 + p01 * nb01;
             dma_ಸರತಿ_ತಳ್ಳು(dma_q, dma_make_ದತ್ತಾಂಶ(s0_spad, s0_ಮುಂದಿನ), bctx->src0_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned, nb01, ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_ಬೈಟುಗಳು, ಮುಂದಿನ_ಖಂಡ_ಗಾತ್ರ);
             ir_prefetch += ಮುಂದಿನ_ಖಂಡ_ಗಾತ್ರ;
        }
        ir += current_ಖಂಡ_ಗಾತ್ರ;
    }

    dma_ಸರತಿ_flush(dma_q);
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ hvx_div_scalar_f32(uint8_t * ಸೀಮಿತ dst, ಸ್ಥಿರಾಂಕ uint8_t * ಸೀಮಿತ src, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ val, ಸ್ಥಿರಾಂಕ uint32_t ಅಂಕೆ_elems) {
    hvx_mul_scalar_f32(dst, src, 1.0f / val, ಅಂಕೆ_elems);
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ hvx_div_scalar_f16(uint8_t * ಸೀಮಿತ dst, ಸ್ಥಿರಾಂಕ uint8_t * ಸೀಮಿತ src, ಸ್ಥಿರಾಂಕ _Float16 val, ಸ್ಥಿರಾಂಕ uint32_t ಅಂಕೆ_elems) {
    hvx_div_scalar_f16_aa(dst, src, val, ಅಂಕೆ_elems);
}

ಪ್ರಕಾರವ್ಯಾಖ್ಯೆ ಶೂನ್ಯ (*compute_ದ್ವಿಮಾನ_chunked_t)(
    uint8_t * ಸೀಮಿತ dst,
    ಸ್ಥಿರಾಂಕ uint8_t * ಸೀಮಿತ src0,
    ಸ್ಥಿರಾಂಕ uint8_t * ಸೀಮಿತ src1,
    ಸ್ಥಿರಾಂಕ uint32_t ಅಂಕೆ_elems
);

ಪ್ರಕಾರವ್ಯಾಖ್ಯೆ ಶೂನ್ಯ (*compute_ದ್ವಿಮಾನ_scalar_chunked_f32_t)(
    uint8_t * ಸೀಮಿತ dst,
    ಸ್ಥಿರಾಂಕ uint8_t * ಸೀಮಿತ src0,
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ val,
    ಸ್ಥಿರಾಂಕ uint32_t ಅಂಕೆ_elems
);

ಪ್ರಕಾರವ್ಯಾಖ್ಯೆ ಶೂನ್ಯ (*compute_ದ್ವಿಮಾನ_scalar_chunked_f16_t)(
    uint8_t * ಸೀಮಿತ dst,
    ಸ್ಥಿರಾಂಕ uint8_t * ಸೀಮಿತ src0,
    ಸ್ಥಿರಾಂಕ _Float16 val,
    ಸ್ಥಿರಾಂಕ uint32_t ಅಂಕೆ_elems
);

ರಚನೆ ದ್ವಿಮಾನ_chunked_context {
    ರಚನೆ htp_ops_context * octx;
    ರಚನೆ htp_ದ್ವಿಮಾನ_vtcm_layout vtcm_layout;
    uint8_t *                vtcm_ತಳ;
    uint32_t                 elem_ಆರಂಭ;
    uint32_t                 nelem;
    uint32_t                 chunk_ಗಾತ್ರ;
    uint32_t                 chunks_ಪ್ರತಿ_thread;
    uint32_t                 ಒಟ್ಟು_chunks;
    bool                     is_scalar;
    ದಶಮಾಂಶ                    scalar_f32;
    _Float16                 scalar_f16;
    compute_ದ್ವಿಮಾನ_chunked_t compute;
    compute_ದ್ವಿಮಾನ_scalar_chunked_f32_t compute_scalar_f32;
    compute_ದ್ವಿಮಾನ_scalar_chunked_f16_t compute_scalar_f16;
};

ಸ್ಥಿರ ಶೂನ್ಯ ದ್ವಿಮಾನ_thread_chunked(ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ nth, ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ ith, ಶೂನ್ಯ * ದತ್ತಾಂಶ) {
    (ಶೂನ್ಯ) nth;
    ರಚನೆ ದ್ವಿಮಾನ_chunked_context * ctx = (ರಚನೆ ದ್ವಿಮಾನ_chunked_context *) ದತ್ತಾಂಶ;
    ರಚನೆ htp_ops_context * octx = ctx->octx;
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * src0 = octx->src[0];
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * src1 = octx->src[1];
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * dst  = octx->dst;

    ಸ್ಥಿರಾಂಕ uint32_t ಆರಂಭ_chunk = ctx->chunks_ಪ್ರತಿ_thread * ith;
    ಸ್ಥಿರಾಂಕ uint32_t ಕೊನೆ_chunk   = MIN(ಆರಂಭ_chunk + ctx->chunks_ಪ್ರತಿ_thread, ctx->ಒಟ್ಟು_chunks);
    ಆದರೆ (ಆರಂಭ_chunk >= ಕೊನೆ_chunk) {
        ಹಿಂತಿರುಗಿ;
    }

    ಸ್ಥಿರಾಂಕ uint32_t src0_type   = src0->type;
    ಸ್ಥಿರಾಂಕ size_t   elem_ಗಾತ್ರ   = (src0_type == HTP_TYPE_F32) ? ಗಾತ್ರ(ದಶಮಾಂಶ) : ಗಾತ್ರ(_Float16);
    ಸ್ಥಿರಾಂಕ uint32_t chunk_ಗಾತ್ರ  = ctx->chunk_ಗಾತ್ರ;
    ಸ್ಥಿರಾಂಕ size_t   chunk_ಬೈಟುಗಳು = ctx->vtcm_layout.src0_spad_half_ಗಾತ್ರ;

    FARF(ಮೇಲಿನ_2, "binary-chunked: %d/%d (%u:%u) chunks %u elems %u",
         ith, nth, ಆರಂಭ_chunk, ಕೊನೆ_chunk, ctx->ಒಟ್ಟು_chunks, ctx->nelem);

    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_ದ್ವಿಮಾನ_vtcm_layout * layout = &ctx->vtcm_layout;
    uint8_t * src0_spad_ತಳ = VTCM_LAYOUT_PTR(uint8_t, ctx->vtcm_ತಳ, layout->off_src0) + (ith * layout->src0_ಬೈಟುಗಳು_ಪ್ರತಿ_thread);
    uint8_t * src1_spad_ತಳ = ctx->is_scalar ? ಖಾಲಿ : (VTCM_LAYOUT_PTR(uint8_t, ctx->vtcm_ತಳ, layout->off_src1) + (ith * layout->src1_ಬೈಟುಗಳು_ಪ್ರತಿ_thread));
    uint8_t * dst_spad_ತಳ  = VTCM_LAYOUT_PTR(uint8_t, ctx->vtcm_ತಳ, layout->off_dst)  + (ith * layout->dst_ಬೈಟುಗಳು_ಪ್ರತಿ_thread);

    dma_ಸರತಿ * dma_q = octx->ctx->dma[ith];
    uint32_t prefetch_chunk = ಆರಂಭ_chunk;
    ಪೂರ್ಣಾಂಕ spad_idx = 0;

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ k = 0; k < 2 && prefetch_chunk < ಕೊನೆ_chunk; k++) {
        ಸ್ಥಿರಾಂಕ uint32_t c_ಆರಂಭ   = ctx->elem_ಆರಂಭ + prefetch_chunk * chunk_ಗಾತ್ರ;
        ಸ್ಥಿರಾಂಕ uint32_t c_ಕೊನೆ     = MIN(c_ಆರಂಭ + chunk_ಗಾತ್ರ, ctx->elem_ಆರಂಭ + ctx->nelem);
        ಸ್ಥಿರಾಂಕ uint32_t cur_elems = c_ಕೊನೆ - c_ಆರಂಭ;
        ಸ್ಥಿರಾಂಕ uint32_t cur_ಬೈಟುಗಳು = cur_elems * elem_ಗಾತ್ರ;

        dma_addr_t s0_curr = src0->ದತ್ತಾಂಶ + (size_t) c_ಆರಂಭ * elem_ಗಾತ್ರ;
        dma_addr_t d_curr  = dst->ದತ್ತಾಂಶ  + (size_t) c_ಆರಂಭ * elem_ಗಾತ್ರ;

        uint8_t * s0_spad = src0_spad_ತಳ + spad_idx * chunk_ಬೈಟುಗಳು;
        uint8_t * d_spad  = dst_spad_ತಳ  + spad_idx * chunk_ಬೈಟುಗಳು;

        dma_ಸರತಿ_ತಳ್ಳು(dma_q, dma_make_ದತ್ತಾಂಶ(d_curr, d_spad), chunk_ಬೈಟುಗಳು, chunk_ಬೈಟುಗಳು, cur_ಬೈಟುಗಳು, 0);
        dma_ಸರತಿ_ತಳ್ಳು(dma_q, dma_make_ದತ್ತಾಂಶ(s0_spad, s0_curr), chunk_ಬೈಟುಗಳು, chunk_ಬೈಟುಗಳು, cur_ಬೈಟುಗಳು, 1);
        ಆದರೆ (!ctx->is_scalar) {
            dma_addr_t s1_curr = src1->ದತ್ತಾಂಶ + (size_t) c_ಆರಂಭ * elem_ಗಾತ್ರ;
            uint8_t * s1_spad = src1_spad_ತಳ + spad_idx * chunk_ಬೈಟುಗಳು;
            dma_ಸರತಿ_ತಳ್ಳು(dma_q, dma_make_ದತ್ತಾಂಶ(s1_spad, s1_curr), chunk_ಬೈಟುಗಳು, chunk_ಬೈಟುಗಳು, cur_ಬೈಟುಗಳು, 1);
        }

        prefetch_chunk++;
        spad_idx ^= 1;
    }

    ರಚನೆ htp_thread_trace * tr = &octx->ctx->trace[ith];

    ಗಾಗಿ (uint32_t c = ಆರಂಭ_chunk; c < ಕೊನೆ_chunk; c++) {
        ಸ್ಥಿರಾಂಕ uint32_t c_ಆರಂಭ   = ctx->elem_ಆರಂಭ + c * chunk_ಗಾತ್ರ;
        ಸ್ಥಿರಾಂಕ uint32_t c_ಕೊನೆ     = MIN(c_ಆರಂಭ + chunk_ಗಾತ್ರ, ctx->elem_ಆರಂಭ + ctx->nelem);
        ಸ್ಥಿರಾಂಕ uint32_t cur_elems = c_ಕೊನೆ - c_ಆರಂಭ;
        ಸ್ಥಿರಾಂಕ uint32_t cur_ಬೈಟುಗಳು = cur_elems * elem_ಗಾತ್ರ;

        uint8_t * d_spad  = (uint8_t *) dma_ಸರತಿ_ತೆಗೆ(dma_q).src;
        uint8_t * s0_spad = (uint8_t *) dma_ಸರತಿ_ತೆಗೆ(dma_q).dst;
        uint8_t * s1_spad = ctx->is_scalar ? ಖಾಲಿ : (uint8_t *) dma_ಸರತಿ_ತೆಗೆ(dma_q).dst;

        htp_trace_event_ಆರಂಭ(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) c);
        ಆದರೆ (ctx->is_scalar) {
            ಆದರೆ (src0_type == HTP_TYPE_F32) {
                ctx->compute_scalar_f32(d_spad, s0_spad, ctx->scalar_f32, cur_elems);
            } ಇಲ್ಲದಿದ್ದರೆ {
                ctx->compute_scalar_f16(d_spad, s0_spad, ctx->scalar_f16, cur_elems);
            }
        } ಇಲ್ಲದಿದ್ದರೆ {
            ctx->compute(d_spad, s0_spad, s1_spad, cur_elems);
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) c);

        dma_addr_t dst_curr = dst->ದತ್ತಾಂಶ + (size_t) c_ಆರಂಭ * elem_ಗಾತ್ರ;
        dma_ಸರತಿ_ತಳ್ಳು(dma_q, dma_make_ದತ್ತಾಂಶ(dst_curr, d_spad), chunk_ಬೈಟುಗಳು, chunk_ಬೈಟುಗಳು, cur_ಬೈಟುಗಳು, 1);

        ಆದರೆ (prefetch_chunk < ಕೊನೆ_chunk) {
            ಸ್ಥಿರಾಂಕ uint32_t pc_ಆರಂಭ  = ctx->elem_ಆರಂಭ + prefetch_chunk * chunk_ಗಾತ್ರ;
            ಸ್ಥಿರಾಂಕ uint32_t pc_ಕೊನೆ    = MIN(pc_ಆರಂಭ + chunk_ಗಾತ್ರ, ctx->elem_ಆರಂಭ + ctx->nelem);
            ಸ್ಥಿರಾಂಕ uint32_t p_elems   = pc_ಕೊನೆ - pc_ಆರಂಭ;
            ಸ್ಥಿರಾಂಕ uint32_t p_ಬೈಟುಗಳು   = p_elems * elem_ಗಾತ್ರ;

            dma_addr_t s0_ಮುಂದಿನ = src0->ದತ್ತಾಂಶ + (size_t) pc_ಆರಂಭ * elem_ಗಾತ್ರ;
            dma_ಸರತಿ_ತಳ್ಳು(dma_q, dma_make_ದತ್ತಾಂಶ(s0_spad, s0_ಮುಂದಿನ), chunk_ಬೈಟುಗಳು, chunk_ಬೈಟುಗಳು, p_ಬೈಟುಗಳು, 1);
            ಆದರೆ (!ctx->is_scalar) {
                dma_addr_t s1_ಮುಂದಿನ = src1->ದತ್ತಾಂಶ + (size_t) pc_ಆರಂಭ * elem_ಗಾತ್ರ;
                dma_ಸರತಿ_ತಳ್ಳು(dma_q, dma_make_ದತ್ತಾಂಶ(s1_spad, s1_ಮುಂದಿನ), chunk_ಬೈಟುಗಳು, chunk_ಬೈಟುಗಳು, p_ಬೈಟುಗಳು, 1);
            }

            prefetch_chunk++;
        }
    }

    dma_ಸರತಿ_flush(dma_q);
}

ಸ್ಥಿರ ಪೂರ್ಣಾಂಕ execute_op_ದ್ವಿಮಾನ(ರಚನೆ htp_ops_context * octx) {
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * src0 = octx->src[0];
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * src1 = octx->src[1];
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * dst  = octx->dst;
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_ದ್ವಿಮಾನ_kernel_params * kparams = (ಸ್ಥಿರಾಂಕ ರಚನೆ htp_ದ್ವಿಮಾನ_kernel_params *) octx->kernel_params;

    ಸ್ಥಿರಾಂಕ uint32_t src0_nrows = src0->ne[1] * src0->ne[2] * src0->ne[3];

    // Use packed row sizes for VTCM allocation and alignment
    ಸ್ಥಿರಾಂಕ uint32_t src0_type = octx->src[0]->type;
    ಸ್ಥಿರಾಂಕ size_t elem_ಗಾತ್ರ = (src0_type == HTP_TYPE_F32) ? ಗಾತ್ರ(ದಶಮಾಂಶ) : ಗಾತ್ರ(_Float16);
    ಸ್ಥಿರಾಂಕ size_t src0_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ = src0->ne[0] * elem_ಗಾತ್ರ;
    ಸ್ಥಿರಾಂಕ size_t src1_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ = src1->ne[0] * elem_ಗಾತ್ರ;
    ಸ್ಥಿರಾಂಕ size_t dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ  = dst->ne[0]  * elem_ಗಾತ್ರ;

    ಆದರೆ (kparams->kernel_type == HTP_ದ್ವಿಮಾನ_KERNEL_CHUNKED) {
        ಸ್ಥಿರಾಂಕ bool is_scalar = kparams->is_scalar || (src1->ne[0] == 1 && src1->ne[1] == 1 && src1->ne[2] == 1 && src1->ne[3] == 1);

        compute_ದ್ವಿಮಾನ_chunked_t compute = ಖಾಲಿ;
        compute_ದ್ವಿಮಾನ_scalar_chunked_f32_t compute_scalar_f32 = ಖಾಲಿ;
        compute_ದ್ವಿಮಾನ_scalar_chunked_f16_t compute_scalar_f16 = ಖಾಲಿ;

        ಆದರೆ (is_scalar) {
            ಆದರೆ (src0_type == HTP_TYPE_F32) {
                ಆಯ್ಕೆ (octx->op) {
                    ಸ್ಥಿತಿ HTP_OP_ಕೂಡಿಸು: compute_scalar_f32 = hvx_ಕೂಡಿಸು_scalar_f32; ನಿಲ್ಲಿಸು;
                    ಸ್ಥಿತಿ HTP_OP_SUB: compute_scalar_f32 = hvx_sub_scalar_f32; ನಿಲ್ಲಿಸು;
                    ಸ್ಥಿತಿ HTP_OP_MUL: compute_scalar_f32 = hvx_mul_scalar_f32; ನಿಲ್ಲಿಸು;
                    ಸ್ಥಿತಿ HTP_OP_DIV: compute_scalar_f32 = hvx_div_scalar_f32; ನಿಲ್ಲಿಸು;
                    ಪೂರ್ವನಿಯೋಜಿತ: ನಿಲ್ಲಿಸು;
                }
            } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (src0_type == HTP_TYPE_F16) {
                ಆಯ್ಕೆ (octx->op) {
                    ಸ್ಥಿತಿ HTP_OP_ಕೂಡಿಸು: compute_scalar_f16 = hvx_ಕೂಡಿಸು_scalar_f16; ನಿಲ್ಲಿಸು;
                    ಸ್ಥಿತಿ HTP_OP_SUB: compute_scalar_f16 = hvx_sub_scalar_f16; ನಿಲ್ಲಿಸು;
                    ಸ್ಥಿತಿ HTP_OP_MUL: compute_scalar_f16 = hvx_mul_scalar_f16; ನಿಲ್ಲಿಸು;
                    ಸ್ಥಿತಿ HTP_OP_DIV: compute_scalar_f16 = hvx_div_scalar_f16; ನಿಲ್ಲಿಸು;
                    ಪೂರ್ವನಿಯೋಜಿತ: ನಿಲ್ಲಿಸು;
                }
            }
            ಆದರೆ (!compute_scalar_f32 && !compute_scalar_f16) {
                ಹಿಂತಿರುಗಿ HTP_ಸ್ಥಿತಿ_NO_SUPPORT;
            }
        } ಇಲ್ಲದಿದ್ದರೆ {
            ಆದರೆ (src0_type == HTP_TYPE_F32) {
                ಆಯ್ಕೆ (octx->op) {
                    ಸ್ಥಿತಿ HTP_OP_ಕೂಡಿಸು: compute = hvx_ಕೂಡಿಸು_f32; ನಿಲ್ಲಿಸು;
                    ಸ್ಥಿತಿ HTP_OP_SUB: compute = hvx_sub_f32; ನಿಲ್ಲಿಸು;
                    ಸ್ಥಿತಿ HTP_OP_MUL: compute = hvx_mul_f32; ನಿಲ್ಲಿಸು;
                    ಸ್ಥಿತಿ HTP_OP_DIV: compute = hvx_div_f32; ನಿಲ್ಲಿಸು;
                    ಪೂರ್ವನಿಯೋಜಿತ: ನಿಲ್ಲಿಸು;
                }
            } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (src0_type == HTP_TYPE_F16) {
                ಆಯ್ಕೆ (octx->op) {
                    ಸ್ಥಿತಿ HTP_OP_ಕೂಡಿಸು: compute = hvx_ಕೂಡಿಸು_f16; ನಿಲ್ಲಿಸು;
                    ಸ್ಥಿತಿ HTP_OP_SUB: compute = hvx_sub_f16; ನಿಲ್ಲಿಸು;
                    ಸ್ಥಿತಿ HTP_OP_MUL: compute = hvx_mul_f16; ನಿಲ್ಲಿಸು;
                    ಸ್ಥಿತಿ HTP_OP_DIV: compute = hvx_div_f16; ನಿಲ್ಲಿಸು;
                    ಪೂರ್ವನಿಯೋಜಿತ: ನಿಲ್ಲಿಸು;
                }
            }
            ಆದರೆ (!compute) {
                ಹಿಂತಿರುಗಿ HTP_ಸ್ಥಿತಿ_NO_SUPPORT;
            }
        }

        ಸ್ಥಿರಾಂಕ uint32_t ಒಟ್ಟು_elems = (uint32_t) (src0->ne[0] * src0->ne[1] * src0->ne[2] * src0->ne[3]);
        ಆದರೆ (ಒಟ್ಟು_elems == 0) {
            ಹಿಂತಿರುಗಿ HTP_ಸ್ಥಿತಿ_OK;
        }

        uint32_t elem_ಆರಂಭ = 0;
        uint32_t nelem      = ಒಟ್ಟು_elems;
        ಸ್ಥಿರಾಂಕ uint32_t elems_ಪ್ರತಿ_ಸಾಲು = (src0_type == HTP_TYPE_F32) ? 32 : 64;

        ಆದರೆ (octx->ctx->mdev.ಎಣಿಕೆ_2 > 1) {
            ಸ್ಥಿರಾಂಕ bool can_ವಿಭಜಿಸು = htp_tensor_mdev_ದತ್ತಾಂಶ_aligned(dst) &&
                                   htp_tensor_mdev_ದತ್ತಾಂಶ_aligned(src0) &&
                                   (is_scalar || htp_tensor_mdev_ದತ್ತಾಂಶ_aligned(src1)) &&
                                   htp_tensor_is_contiguous(dst, elem_ಗಾತ್ರ) &&
                                   htp_tensor_is_contiguous(src0, elem_ಗಾತ್ರ) &&
                                   (is_scalar || htp_tensor_is_contiguous(src1, elem_ಗಾತ್ರ));
            ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor_mdev_range range = htp_tensor_mdev_ವಿಭಜನೆ(
                ಒಟ್ಟು_elems, can_ವಿಭಜಿಸು ? elems_ಪ್ರತಿ_ಸಾಲು : 0,
                octx->ctx->mdev.idx, octx->ctx->mdev.ಎಣಿಕೆ_2, &octx->ctx->mdev.ಎಣಿಕೆ_div);
            elem_ಆರಂಭ = range.ಆರಂಭ;
            nelem      = range.ಎಣಿಕೆ_2;
        }

        ಆದರೆ (nelem == 0) {
            ಹಿಂತಿರುಗಿ HTP_ಸ್ಥಿತಿ_OK;
        }

        ಆದರೆ (!htp_ops_context_ಗಣ_n_ಎಳೆಗಳು(octx, kparams->n_ಎಳೆಗಳು)) {
            ಹಿಂತಿರುಗಿ HTP_ಸ್ಥಿತಿ_INVAL_PARAMS;
        }

        ರಚನೆ htp_ದ್ವಿಮಾನ_vtcm_layout vtcm_layout;
        htp_ದ್ವಿಮಾನ_vtcm_layout_build(&vtcm_layout, kparams, octx->ctx->vtcm_ಗಾತ್ರ);
        ಆದರೆ (vtcm_layout.ಒಟ್ಟು_ಬೈಟುಗಳು == 0 || vtcm_layout.ಒಟ್ಟು_ಬೈಟುಗಳು > octx->ctx->vtcm_ಗಾತ್ರ) {
            ಹಿಂತಿರುಗಿ HTP_ಸ್ಥಿತಿ_VTCM_TOO_SMALL;
        }

        ಸ್ಥಿರಾಂಕ uint32_t chunk_ಗಾತ್ರ = kparams->chunk_ಗಾತ್ರ > 0 ? kparams->chunk_ಗಾತ್ರ : (32768 / elem_ಗಾತ್ರ);
        ಸ್ಥಿರಾಂಕ uint32_t ಒಟ್ಟು_chunks = (nelem + chunk_ಗಾತ್ರ - 1) / chunk_ಗಾತ್ರ;
        ಸ್ಥಿರಾಂಕ uint32_t n_ಎಳೆಗಳು = (ಒಟ್ಟು_chunks >= 2) ? MIN(octx->n_ಎಳೆಗಳು, ಒಟ್ಟು_chunks) : 1;
        ಸ್ಥಿರಾಂಕ uint32_t chunks_ಪ್ರತಿ_thread = (ಒಟ್ಟು_chunks + n_ಎಳೆಗಳು - 1) / n_ಎಳೆಗಳು;

        ದಶಮಾಂಶ scalar_f32 = 0.0f;
        _Float16 scalar_f16 = 0;
        ಆದರೆ (is_scalar) {
            uint8_t * vtcm_src1 = VTCM_LAYOUT_PTR(uint8_t, octx->ctx->vtcm_ತಳ, vtcm_layout.off_src1);
            dma_ಸರತಿ * dma_q = octx->ctx->dma[0];
            dma_ಸರತಿ_ತಳ್ಳು(dma_q, dma_make_ದತ್ತಾಂಶ(vtcm_src1, src1->ದತ್ತಾಂಶ), 128, 0, elem_ಗಾತ್ರ, 1);
            dma_ಸರತಿ_ತೆಗೆ(dma_q);

            ಆದರೆ (src0_type == HTP_TYPE_F32) {
                scalar_f32 = ((ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) vtcm_src1)[0];
            } ಇಲ್ಲದಿದ್ದರೆ {
                scalar_f16 = ((ಸ್ಥಿರಾಂಕ _Float16 *) vtcm_src1)[0];
            }
        }

        ರಚನೆ ದ್ವಿಮಾನ_chunked_context cctx = {
            .octx                = octx,
            .vtcm_layout         = vtcm_layout,
            .vtcm_ತಳ           = (uint8_t *) octx->ctx->vtcm_ತಳ,
            .elem_ಆರಂಭ          = elem_ಆರಂಭ,
            .nelem               = nelem,
            .chunk_ಗಾತ್ರ          = chunk_ಗಾತ್ರ,
            .chunks_ಪ್ರತಿ_thread   = chunks_ಪ್ರತಿ_thread,
            .ಒಟ್ಟು_chunks        = ಒಟ್ಟು_chunks,
            .is_scalar           = is_scalar,
            .scalar_f32          = scalar_f32,
            .scalar_f16          = scalar_f16,
            .compute             = compute,
            .compute_scalar_f32  = compute_scalar_f32,
            .compute_scalar_f16  = compute_scalar_f16,
        };

        work_ಸರತಿ_ಸತತ(octx->ctx->work_ಸರತಿ, ದ್ವಿಮಾನ_thread_chunked, &cctx, n_ಎಳೆಗಳು);
        ಹಿಂತಿರುಗಿ HTP_ಸ್ಥಿತಿ_OK;
    }

    uint32_t ಅಡ್ಡಸಾಲು_ಆರಂಭ = 0;
    uint32_t nrows     = src0_nrows;

    ಆದರೆ (octx->ctx->mdev.ಎಣಿಕೆ_2 > 1) {
        uint32_t ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_chunk = 0;
        htp_tensor_mdev_ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_chunk(dst, (uint32_t) elem_ಗಾತ್ರ, (uint32_t) dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ, &ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_chunk);
        ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor_mdev_range range = htp_tensor_mdev_ವಿಭಜನೆ(src0_nrows, ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_chunk,
                                                       octx->ctx->mdev.idx, octx->ctx->mdev.ಎಣಿಕೆ_2, &octx->ctx->mdev.ಎಣಿಕೆ_div);
        ಅಡ್ಡಸಾಲು_ಆರಂಭ = range.ಆರಂಭ;
        nrows     = range.ಎಣಿಕೆ_2;
    }

    ಆದರೆ (nrows == 0) {
        ಹಿಂತಿರುಗಿ HTP_ಸ್ಥಿತಿ_OK;
    }

    ಆದರೆ (!htp_ops_context_ಗಣ_n_ಎಳೆಗಳು(octx, kparams->n_ಎಳೆಗಳು)) {
        ಹಿಂತಿರುಗಿ HTP_ಸ್ಥಿತಿ_INVAL_PARAMS;
    }

    ಸ್ಥಿರಾಂಕ uint32_t n_ಎಳೆಗಳು = octx->n_ಎಳೆಗಳು;
    ಸ್ಥಿರಾಂಕ size_t src0_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned = kparams->src0_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned;
    ಸ್ಥಿರಾಂಕ size_t src1_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned = kparams->src1_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned;
    ಸ್ಥಿರಾಂಕ size_t dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned  = kparams->dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned;

    ರಚನೆ htp_ದ್ವಿಮಾನ_context bctx;
    bctx.vtcm_ತಳ = (uint8_t *) octx->ctx->vtcm_ತಳ;
    htp_ದ್ವಿಮಾನ_vtcm_layout_build(&bctx.vtcm_layout, kparams, octx->ctx->vtcm_ಗಾತ್ರ);

    ಆದರೆ (bctx.vtcm_layout.ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_ಬಫರ್ == 0 || bctx.vtcm_layout.ಒಟ್ಟು_ಬೈಟುಗಳು > octx->ctx->vtcm_ಗಾತ್ರ) {
        ಹಿಂತಿರುಗಿ HTP_ಸ್ಥಿತಿ_VTCM_TOO_SMALL;
    }

    dma_ಸರತಿ * dma_q = octx->ctx->dma[0];
    uint8_t * vtcm_src1 = VTCM_LAYOUT_PTR(uint8_t, bctx.vtcm_ತಳ, bctx.vtcm_layout.off_src1);
    ಆದರೆ (kparams->kernel_type == HTP_ದ್ವಿಮಾನ_KERNEL_ಅಡ್ಡಸಾಲು_BCAST) {
        dma_ಸರತಿ_ತಳ್ಳು(dma_q, dma_make_ದತ್ತಾಂಶ(vtcm_src1, src1->ದತ್ತಾಂಶ), bctx.vtcm_layout.src1_ಗಾತ್ರ, 0, src1->ne[0] * elem_ಗಾತ್ರ, 1);
    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (kparams->kernel_type == HTP_ದ್ವಿಮಾನ_KERNEL_SCALAR ||
               kparams->kernel_type == HTP_ದ್ವಿಮಾನ_KERNEL_REPEAT ||
               kparams->kernel_type == HTP_ದ್ವಿಮಾನ_KERNEL_COMPLEX) {
        ಆದರೆ (!kparams->src1_strided) {
            ಸ್ಥಿರಾಂಕ size_t src1_ಬೈಟುಗಳು = (size_t) src1->ne[0] * src1->ne[1] * src1->ne[2] * src1->ne[3] * elem_ಗಾತ್ರ;
            dma_ಸರತಿ_ತಳ್ಳು(dma_q, dma_make_ದತ್ತಾಂಶ(vtcm_src1, src1->ದತ್ತಾಂಶ), bctx.vtcm_layout.src1_ಗಾತ್ರ, 0, src1_ಬೈಟುಗಳು, 1);
        } ಇಲ್ಲದಿದ್ದರೆ {
            ಸ್ಥಿರಾಂಕ uint32_t ne11 = src1->ne[1];
            ಸ್ಥಿರಾಂಕ uint32_t ne12 = src1->ne[2];
            ಸ್ಥಿರಾಂಕ uint32_t ne13 = src1->ne[3];
            ಸ್ಥಿರಾಂಕ size_t ಅಡ್ಡಸಾಲು_ಬೈಟುಗಳು = src1->ne[0] * elem_ಗಾತ್ರ;
            ಗಾಗಿ (uint32_t i13 = 0; i13 < ne13; i13++) {
                ಗಾಗಿ (uint32_t i12 = 0; i12 < ne12; i12++) {
                    dma_addr_t src_addr = src1->ದತ್ತಾಂಶ + i13 * src1->nb[3] + i12 * src1->nb[2];
                    uint8_t * dst_ptr = vtcm_src1 + (i13 * ne12 * ne11 + i12 * ne11) * ಅಡ್ಡಸಾಲು_ಬೈಟುಗಳು;
                    ಆಗ (!dma_ಸರತಿ_ತಳ್ಳು(dma_q, dma_make_ದತ್ತಾಂಶ(dst_ptr, src_addr),
                                           ಅಡ್ಡಸಾಲು_ಬೈಟುಗಳು, src1->nb[1], ಅಡ್ಡಸಾಲು_ಬೈಟುಗಳು, ne11)) {
                        dma_ಸರತಿ_ತೆಗೆ(dma_q);
                    }
                }
            }
        }
    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (kparams->kernel_type == HTP_ದ್ವಿಮಾನ_KERNEL_ಕೂಡಿಸು_ID) {
        ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * src2 = octx->src[2];
        uint8_t * vtcm_src2 = VTCM_LAYOUT_PTR(uint8_t, bctx.vtcm_ತಳ, bctx.vtcm_layout.off_src2);
        dma_ಸರತಿ_ತಳ್ಳು(dma_q, dma_make_ದತ್ತಾಂಶ(vtcm_src1, src1->ದತ್ತಾಂಶ),
                       kparams->src1_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned, src1->nb[1],
                       src1->ne[0] * elem_ಗಾತ್ರ, src1->ne[1]);
        ಸ್ಥಿರಾಂಕ size_t src2_ಅಡ್ಡಸಾಲು_ಬೈಟುಗಳು = src2->ne[0] * ಗಾತ್ರ(int32_t);
        dma_ಸರತಿ_ತಳ್ಳು(dma_q, dma_make_ದತ್ತಾಂಶ(vtcm_src2, src2->ದತ್ತಾಂಶ),
                       src2_ಅಡ್ಡಸಾಲು_ಬೈಟುಗಳು, src2->nb[1],
                       src2_ಅಡ್ಡಸಾಲು_ಬೈಟುಗಳು, src2->ne[1]);
    }

    bctx.octx                  = octx;
    bctx.nrows_ಪ್ರತಿ_thread      = fastdiv(nrows + n_ಎಳೆಗಳು - 1, &octx->n_ಎಳೆಗಳು_div);
    bctx.ಒಟ್ಟು_ಅಡ್ಡಸಾಲುಗಳು            = nrows;
    bctx.ಅಡ್ಡಸಾಲು_ಆರಂಭ             = ಅಡ್ಡಸಾಲು_ಆರಂಭ;
    bctx.ಖಂಡ_max             = bctx.vtcm_layout.ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_ಬಫರ್;
    bctx.src0_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned = src0_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned;
    bctx.src1_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned = src1_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned;
    bctx.dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned  = dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned;
    bctx.ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_ಬೈಟುಗಳು        = src0_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ;

    bctx.src0_dim1_div  = init_fastdiv_ಮೌಲ್ಯಗಳು(src0->ne[1]);
    bctx.src0_dim2_div  = init_fastdiv_ಮೌಲ್ಯಗಳು(src0->ne[2]);
    bctx.src0_dim12_div = init_fastdiv_ಮೌಲ್ಯಗಳು(src0->ne[1] * src0->ne[2]);

    bctx.src1_dim1_div  = init_fastdiv_ಮೌಲ್ಯಗಳು(src1->ne[1]);
    bctx.src1_dim2_div  = init_fastdiv_ಮೌಲ್ಯಗಳು(src1->ne[2]);
    bctx.src1_dim3_div  = init_fastdiv_ಮೌಲ್ಯಗಳು(src1->ne[3]);

    bctx.ವಿಭಜಿಸು_at_ne01  = kparams->ವಿಭಜಿಸು_at_ne01;
    bctx.ವಿಭಜಿಸು_at_ne02  = kparams->ವಿಭಜಿಸು_at_ne02;

    ಕೆಲಸಗಾರ_callback_t ಕೆಲಸಗಾರ_func = ಖಾಲಿ;
    ಶೂನ್ಯ * compute_func = ಖಾಲಿ;

    ಆಯ್ಕೆ (kparams->kernel_type) {
        ಸ್ಥಿತಿ HTP_ದ್ವಿಮಾನ_KERNEL_SAME_SHAPE:
            ಕೆಲಸಗಾರ_func = ದ್ವಿಮಾನ_thread_ಸದಿಶ_same_shape;
            ಆದರೆ (src0_type == HTP_TYPE_F32) {
                ಆಯ್ಕೆ (octx->op) {
                    ಸ್ಥಿತಿ HTP_OP_ಕೂಡಿಸು: compute_func = compute_same_shape_ಕೂಡಿಸು_f32; ನಿಲ್ಲಿಸು;
                    ಸ್ಥಿತಿ HTP_OP_SUB: compute_func = compute_same_shape_sub_f32; ನಿಲ್ಲಿಸು;
                    ಸ್ಥಿತಿ HTP_OP_MUL: compute_func = compute_same_shape_mul_f32; ನಿಲ್ಲಿಸು;
                    ಸ್ಥಿತಿ HTP_OP_DIV: compute_func = compute_same_shape_div_f32; ನಿಲ್ಲಿಸು;
                    ಪೂರ್ವನಿಯೋಜಿತ: ನಿಲ್ಲಿಸು;
                }
            } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (src0_type == HTP_TYPE_F16) {
                ಆಯ್ಕೆ (octx->op) {
                    ಸ್ಥಿತಿ HTP_OP_ಕೂಡಿಸು: compute_func = compute_same_shape_ಕೂಡಿಸು_f16; ನಿಲ್ಲಿಸು;
                    ಸ್ಥಿತಿ HTP_OP_SUB: compute_func = compute_same_shape_sub_f16; ನಿಲ್ಲಿಸು;
                    ಸ್ಥಿತಿ HTP_OP_MUL: compute_func = compute_same_shape_mul_f16; ನಿಲ್ಲಿಸು;
                    ಸ್ಥಿತಿ HTP_OP_DIV: compute_func = compute_same_shape_div_f16; ನಿಲ್ಲಿಸು;
                    ಪೂರ್ವನಿಯೋಜಿತ: ನಿಲ್ಲಿಸು;
                }
            }
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ HTP_ದ್ವಿಮಾನ_KERNEL_ಅಡ್ಡಸಾಲು_BCAST:
            ಕೆಲಸಗಾರ_func = ದ್ವಿಮಾನ_thread_ಸದಿಶ_ಅಡ್ಡಸಾಲು_ಪ್ರಸಾರ;
            ಆದರೆ (src0_type == HTP_TYPE_F32) {
                ಆಯ್ಕೆ (octx->op) {
                    ಸ್ಥಿತಿ HTP_OP_ಕೂಡಿಸು: compute_func = compute_ಅಡ್ಡಸಾಲು_bcast_ಕೂಡಿಸು_f32; ನಿಲ್ಲಿಸು;
                    ಸ್ಥಿತಿ HTP_OP_SUB: compute_func = compute_ಅಡ್ಡಸಾಲು_bcast_sub_f32; ನಿಲ್ಲಿಸು;
                    ಸ್ಥಿತಿ HTP_OP_MUL: compute_func = compute_ಅಡ್ಡಸಾಲು_bcast_mul_f32; ನಿಲ್ಲಿಸು;
                    ಸ್ಥಿತಿ HTP_OP_DIV: compute_func = compute_ಅಡ್ಡಸಾಲು_bcast_div_f32; ನಿಲ್ಲಿಸು;
                    ಪೂರ್ವನಿಯೋಜಿತ: ನಿಲ್ಲಿಸು;
                }
            } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (src0_type == HTP_TYPE_F16) {
                ಆಯ್ಕೆ (octx->op) {
                    ಸ್ಥಿತಿ HTP_OP_ಕೂಡಿಸು: compute_func = compute_ಅಡ್ಡಸಾಲು_bcast_ಕೂಡಿಸು_f16; ನಿಲ್ಲಿಸು;
                    ಸ್ಥಿತಿ HTP_OP_SUB: compute_func = compute_ಅಡ್ಡಸಾಲು_bcast_sub_f16; ನಿಲ್ಲಿಸು;
                    ಸ್ಥಿತಿ HTP_OP_MUL: compute_func = compute_ಅಡ್ಡಸಾಲು_bcast_mul_f16; ನಿಲ್ಲಿಸು;
                    ಸ್ಥಿತಿ HTP_OP_DIV: compute_func = compute_ಅಡ್ಡಸಾಲು_bcast_div_f16; ನಿಲ್ಲಿಸು;
                    ಪೂರ್ವನಿಯೋಜಿತ: ನಿಲ್ಲಿಸು;
                }
            }
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ HTP_ದ್ವಿಮಾನ_KERNEL_SCALAR:
            ಕೆಲಸಗಾರ_func = ದ್ವಿಮಾನ_thread_scalar;
            ಆದರೆ (src0_type == HTP_TYPE_F32) {
                ಆಯ್ಕೆ (octx->op) {
                    ಸ್ಥಿತಿ HTP_OP_ಕೂಡಿಸು: compute_func = compute_scalar_ಕೂಡಿಸು_f32; ನಿಲ್ಲಿಸು;
                    ಸ್ಥಿತಿ HTP_OP_SUB: compute_func = compute_scalar_sub_f32; ನಿಲ್ಲಿಸು;
                    ಸ್ಥಿತಿ HTP_OP_MUL: compute_func = compute_scalar_mul_f32; ನಿಲ್ಲಿಸು;
                    ಸ್ಥಿತಿ HTP_OP_DIV: compute_func = compute_scalar_div_f32; ನಿಲ್ಲಿಸು;
                    ಪೂರ್ವನಿಯೋಜಿತ: ನಿಲ್ಲಿಸು;
                }
            } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (src0_type == HTP_TYPE_F16) {
                ಆಯ್ಕೆ (octx->op) {
                    ಸ್ಥಿತಿ HTP_OP_ಕೂಡಿಸು: compute_func = compute_scalar_ಕೂಡಿಸು_f16; ನಿಲ್ಲಿಸು;
                    ಸ್ಥಿತಿ HTP_OP_SUB: compute_func = compute_scalar_sub_f16; ನಿಲ್ಲಿಸು;
                    ಸ್ಥಿತಿ HTP_OP_MUL: compute_func = compute_scalar_mul_f16; ನಿಲ್ಲಿಸು;
                    ಸ್ಥಿತಿ HTP_OP_DIV: compute_func = compute_scalar_div_f16; ನಿಲ್ಲಿಸು;
                    ಪೂರ್ವನಿಯೋಜಿತ: ನಿಲ್ಲಿಸು;
                }
            }
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ HTP_ದ್ವಿಮಾನ_KERNEL_COMPLEX:
            ಕೆಲಸಗಾರ_func = ದ್ವಿಮಾನ_thread_ಸದಿಶ_complex;
            ಆದರೆ (src0_type == HTP_TYPE_F32) {
                ಆಯ್ಕೆ (octx->op) {
                    ಸ್ಥಿತಿ HTP_OP_ಕೂಡಿಸು: compute_func = compute_complex_ಕೂಡಿಸು_f32; ನಿಲ್ಲಿಸು;
                    ಸ್ಥಿತಿ HTP_OP_SUB: compute_func = compute_complex_sub_f32; ನಿಲ್ಲಿಸು;
                    ಸ್ಥಿತಿ HTP_OP_MUL: compute_func = compute_complex_mul_f32; ನಿಲ್ಲಿಸು;
                    ಸ್ಥಿತಿ HTP_OP_DIV: compute_func = compute_complex_div_f32; ನಿಲ್ಲಿಸು;
                    ಪೂರ್ವನಿಯೋಜಿತ: ನಿಲ್ಲಿಸು;
                }
            } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (src0_type == HTP_TYPE_F16) {
                ಆಯ್ಕೆ (octx->op) {
                    ಸ್ಥಿತಿ HTP_OP_ಕೂಡಿಸು: compute_func = compute_complex_ಕೂಡಿಸು_f16; ನಿಲ್ಲಿಸು;
                    ಸ್ಥಿತಿ HTP_OP_SUB: compute_func = compute_complex_sub_f16; ನಿಲ್ಲಿಸು;
                    ಸ್ಥಿತಿ HTP_OP_MUL: compute_func = compute_complex_mul_f16; ನಿಲ್ಲಿಸು;
                    ಸ್ಥಿತಿ HTP_OP_DIV: compute_func = compute_complex_div_f16; ನಿಲ್ಲಿಸು;
                    ಪೂರ್ವನಿಯೋಜಿತ: ನಿಲ್ಲಿಸು;
                }
            }
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ HTP_ದ್ವಿಮಾನ_KERNEL_REPEAT:
            ಕೆಲಸಗಾರ_func = ದ್ವಿಮಾನ_thread_element_repeat;
            ಆದರೆ (src0_type == HTP_TYPE_F32) {
                ಆಯ್ಕೆ (octx->op) {
                    ಸ್ಥಿತಿ HTP_OP_ಕೂಡಿಸು: compute_func = compute_repeat_ಕೂಡಿಸು_f32; ನಿಲ್ಲಿಸು;
                    ಸ್ಥಿತಿ HTP_OP_SUB: compute_func = compute_repeat_sub_f32; ನಿಲ್ಲಿಸು;
                    ಸ್ಥಿತಿ HTP_OP_MUL: compute_func = compute_repeat_mul_f32; ನಿಲ್ಲಿಸು;
                    ಸ್ಥಿತಿ HTP_OP_DIV: compute_func = compute_repeat_div_f32; ನಿಲ್ಲಿಸು;
                    ಪೂರ್ವನಿಯೋಜಿತ: ನಿಲ್ಲಿಸು;
                }
            } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (src0_type == HTP_TYPE_F16) {
                ಆಯ್ಕೆ (octx->op) {
                    ಸ್ಥಿತಿ HTP_OP_ಕೂಡಿಸು: compute_func = compute_repeat_ಕೂಡಿಸು_f16; ನಿಲ್ಲಿಸು;
                    ಸ್ಥಿತಿ HTP_OP_SUB: compute_func = compute_repeat_sub_f16; ನಿಲ್ಲಿಸು;
                    ಸ್ಥಿತಿ HTP_OP_MUL: compute_func = compute_repeat_mul_f16; ನಿಲ್ಲಿಸು;
                    ಸ್ಥಿತಿ HTP_OP_DIV: compute_func = compute_repeat_div_f16; ನಿಲ್ಲಿಸು;
                    ಪೂರ್ವನಿಯೋಜಿತ: ನಿಲ್ಲಿಸು;
                }
            }
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ HTP_ದ್ವಿಮಾನ_KERNEL_ಕೂಡಿಸು_ID:
            ಆದರೆ (octx->op == HTP_OP_ಕೂಡಿಸು_ID && src0_type == HTP_TYPE_F32) {
                ಕೆಲಸಗಾರ_func = ದ್ವಿಮಾನ_thread_ಕೂಡಿಸು_id_f32;
                compute_func = (ಶೂನ್ಯ *) compute_ಕೂಡಿಸು_id_f32;
            }
            ನಿಲ್ಲಿಸು;
        ಪೂರ್ವನಿಯೋಜಿತ: ನಿಲ್ಲಿಸು;
    }

    ಆದರೆ (!ಕೆಲಸಗಾರ_func || !compute_func) {
        ಹಿಂತಿರುಗಿ HTP_ಸ್ಥಿತಿ_NO_SUPPORT;
    }

    bctx.compute = compute_func;

    ಆದರೆ (kparams->kernel_type == HTP_ದ್ವಿಮಾನ_KERNEL_ಅಡ್ಡಸಾಲು_BCAST ||
        kparams->kernel_type == HTP_ದ್ವಿಮಾನ_KERNEL_SCALAR ||
        kparams->kernel_type == HTP_ದ್ವಿಮಾನ_KERNEL_REPEAT ||
        kparams->kernel_type == HTP_ದ್ವಿಮಾನ_KERNEL_COMPLEX ||
        kparams->kernel_type == HTP_ದ್ವಿಮಾನ_KERNEL_ಕೂಡಿಸು_ID) {
        dma_ಸರತಿ_flush(dma_q);
    }

    work_ಸರತಿ_ಸತತ(octx->ctx->work_ಸರತಿ, ಕೆಲಸಗಾರ_func, &bctx, n_ಎಳೆಗಳು);

    ಹಿಂತಿರುಗಿ HTP_ಸ್ಥಿತಿ_OK;
}

ಪೂರ್ಣಾಂಕ op_ದ್ವಿಮಾನ(ರಚನೆ htp_ops_context * octx) {
    ಸ್ಥಿರಾಂಕ uint32_t src0_type = octx->src[0]->type;
    ಆದರೆ ((src0_type == HTP_TYPE_F32) || (src0_type == HTP_TYPE_F16)) {
        ಹಿಂತಿರುಗಿ execute_op_ದ್ವಿಮಾನ(octx);
    }

    ಹಿಂತಿರುಗಿ HTP_ಸ್ಥಿತಿ_NO_SUPPORT;
}
