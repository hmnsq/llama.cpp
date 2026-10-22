#ਹਦਾਇਤ clang diagnostic ignored "-Wunused-variable"
#ਹਦਾਇਤ clang diagnostic ignored "-Wunused-function"
#ਹਦਾਇਤ clang diagnostic ignored "-Wunused-but-set-variable"

#ਅਨ੍ਤਰ੍ਭਾਵ <HAP_farf.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <HAP_perf.h>

#ਅਨ੍ਤਰ੍ਭਾਵ <math.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <string.h>

#ਅਨ੍ਤਰ੍ਭਾਵ "dma-queue.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "hvx-utils.h"

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_COMMON_DECL_C
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-common.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "hex-common.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "hex-profile.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "binary-ops.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "htp-ctx.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "htp-ops.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "htp-tensor.h"

#ਜੇ_ਅਪਰਿਭਾਸ਼ਿਤ MIN
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ MIN(a, b) ((a) < (b) ? (a) : (b))
#ਜੇ_ਅੰਤ

// Context for binary operations
ਬਣਤਰ htp_ਦੁਆਧਾਰੀ_context {
    ਬਣਤਰ htp_ops_context * octx;
    ਬਣਤਰ htp_ਦੁਆਧਾਰੀ_vtcm_layout vtcm_layout;
    uint8_t * vtcm_ਅਧਾਰ;

    ਬਣਤਰ fastdiv_ਮੁੱਲਾਂ src0_dim1_div; // ne01
    ਬਣਤਰ fastdiv_ਮੁੱਲਾਂ src0_dim2_div; // ne02
    ਬਣਤਰ fastdiv_ਮੁੱਲਾਂ src0_dim12_div;// ne03

    ਬਣਤਰ fastdiv_ਮੁੱਲਾਂ src1_dim1_div; // ne11
    ਬਣਤਰ fastdiv_ਮੁੱਲਾਂ src1_dim2_div; // ne12
    ਬਣਤਰ fastdiv_ਮੁੱਲਾਂ src1_dim3_div; // ne13

    uint32_t ਖੰਡ_max;
    uint32_t nrows_ਪ੍ਰਤੀ_thread;
    uint32_t ਕੁੱਲ_ਪੰਕਤੀਆਂ;
    uint32_t ਪੰਕਤੀ_ਸ਼ੁਰੂ;
    size_t   src0_ਪੰਕਤੀ_ਮਾਪ_aligned;
    size_t   src1_ਪੰਕਤੀ_ਮਾਪ_aligned;
    size_t   dst_ਪੰਕਤੀ_ਮਾਪ_aligned;
    size_t   ਪੰਕਤੀ_ਮਾਪ_ਬਾਈਟਾਂ;

    bool ਵੰਡੋ_at_ne01;
    bool ਵੰਡੋ_at_ne02;

    ਖਾਲੀ * compute;
};

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ htp_ਦੁਆਧਾਰੀ_preamble                        \
    ਅਚਲ ਬਣਤਰ htp_tensor * src0 = octx->src[0]; \
    ਅਚਲ ਬਣਤਰ htp_tensor * src1 = octx->src[1]; \
    ਅਚਲ ਬਣਤਰ htp_tensor * dst  = octx->dst;    \
                                                   \
    ਅਚਲ uint32_t ne00 = src0->ne[0];             \
    ਅਚਲ uint32_t ne01 = src0->ne[1];             \
    ਅਚਲ uint32_t ne02 = src0->ne[2];             \
    ਅਚਲ uint32_t ne03 = src0->ne[3];             \
                                                   \
    ਅਚਲ uint32_t ne10 = src1->ne[0];             \
    ਅਚਲ uint32_t ne11 = src1->ne[1];             \
    ਅਚਲ uint32_t ne12 = src1->ne[2];             \
    ਅਚਲ uint32_t ne13 = src1->ne[3];             \
                                                   \
    ਅਚਲ uint32_t nb01 = src0->nb[1];             \
    ਅਚਲ uint32_t nb02 = src0->nb[2];             \
    ਅਚਲ uint32_t nb03 = src0->nb[3];             \
                                                   \
    ਅਚਲ uint32_t nb11 = src1->nb[1];             \
    ਅਚਲ uint32_t nb12 = src1->nb[2];             \
    ਅਚਲ uint32_t nb13 = src1->nb[3];             \
                                                   \
    ਅਚਲ uint32_t nb1 = dst->nb[1];               \
    ਅਚਲ uint32_t nb2 = dst->nb[2];               \
    ਅਚਲ uint32_t nb3 = dst->nb[3];

ਸਥਿਰ ਸਤਰਅੰਦਰ uint32_t calc_ਖੰਡ_ਮਾਪ(ਬਣਤਰ htp_ਦੁਆਧਾਰੀ_context * bctx, uint32_t ir, uint32_t ਸਿਰਾ_ਪੰਕਤੀ, uint32_t ne01, uint32_t ne02) {
    uint32_t i03, i02, i01, rem;
    i03 = fastdiv(ir, &bctx->src0_dim12_div);
    rem = ir - i03 * (ne02 * ne01);
    i02 = fastdiv(rem, &bctx->src0_dim1_div);
    i01 = rem - i02 * ne01;

    uint32_t ਪੰਕਤੀਆਂ_ਖੱਬਾ = ਸਿਰਾ_ਪੰਕਤੀ - ir;
    uint32_t ਖੰਡ_ਹੱਦਬੰਦੀ = ਪੰਕਤੀਆਂ_ਖੱਬਾ;

    ਜੇ (bctx->ਵੰਡੋ_at_ne01) {
        ਖੰਡ_ਹੱਦਬੰਦੀ = MIN(ਖੰਡ_ਹੱਦਬੰਦੀ, ne01 - i01);
    }
    ਜੇ (bctx->ਵੰਡੋ_at_ne02) {
         uint32_t ਪੰਕਤੀਆਂ_ਵਿੱਚ_plane = (ne02 * ne01) - rem;
         ਖੰਡ_ਹੱਦਬੰਦੀ = MIN(ਖੰਡ_ਹੱਦਬੰਦੀ, ਪੰਕਤੀਆਂ_ਵਿੱਚ_plane);
    }

    ਵਾਪਸ MIN(bctx->ਖੰਡ_max, ਖੰਡ_ਹੱਦਬੰਦੀ);
}

// Out-of-line compute micro-kernels

ਕਿਸਮਪਰਿਭਾਸ਼ਾ ਖਾਲੀ (*compute_scalar_t)(
    uint8_t * dst, ਅਚਲ uint8_t * src0, ਅਚਲ ਖਾਲੀ * s1_ਸਾਰਣੀ,
    uint32_t cur_i11, uint32_t ne11, uint32_t n_ਪੰਕਤੀਆਂ,
    size_t dst_stride, size_t src0_stride, uint32_t ne00);

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ DEFINE_COMPUTE_SCALAR(NAME, TYPE, HVX_STMT)             \
ਸਥਿਰ ਖਾਲੀ compute_scalar_##NAME(                              \
    uint8_t * dst, ਅਚਲ uint8_t * src0, ਅਚਲ ਖਾਲੀ * s1_ਸਾਰਣੀ, \
    uint32_t cur_i11, uint32_t ne11, uint32_t n_ਪੰਕਤੀਆਂ,           \
    size_t dst_stride, size_t src0_stride, uint32_t ne00) {     \
    ਅਚਲ TYPE * ਸਾਰਣੀ = (ਅਚਲ TYPE *) s1_ਸਾਰਣੀ;               \
    ਲਈ (uint32_t r = 0; r < n_ਪੰਕਤੀਆਂ; r++) {                     \
        uint8_t * r_dst = dst + r * dst_stride;                 \
        ਅਚਲ uint8_t * r_src0 = src0 + r * src0_stride;        \
        TYPE val = ਸਾਰਣੀ[cur_i11];                              \
        HVX_STMT;                                               \
        ਜੇ (ne11 > 1 && ++cur_i11 == ne11) {                    \
            cur_i11 = 0;                                        \
        }                                                       \
    }                                                           \
}

DEFINE_COMPUTE_SCALAR(ਜੋੜੋ_f32, ਦਸ਼ਮਲਵ,    hvx_ਜੋੜੋ_scalar_f32_aa(r_dst, r_src0, val, ne00))
DEFINE_COMPUTE_SCALAR(ਜੋੜੋ_f16, _Float16, hvx_ਜੋੜੋ_scalar_f16_aa(r_dst, r_src0, val, ne00))
DEFINE_COMPUTE_SCALAR(sub_f32, ਦਸ਼ਮਲਵ,    hvx_sub_scalar_f32_aa(r_dst, r_src0, val, ne00))
DEFINE_COMPUTE_SCALAR(sub_f16, _Float16, hvx_sub_scalar_f16_aa(r_dst, r_src0, val, ne00))
DEFINE_COMPUTE_SCALAR(mul_f32, ਦਸ਼ਮਲਵ,    hvx_mul_scalar_f32_aa(r_dst, r_src0, val, ne00))
DEFINE_COMPUTE_SCALAR(mul_f16, _Float16, hvx_mul_scalar_f16_aa(r_dst, r_src0, val, ne00))
DEFINE_COMPUTE_SCALAR(div_f32, ਦਸ਼ਮਲਵ,    hvx_mul_scalar_f32_aa(r_dst, r_src0, 1.0f / (val), ne00))
DEFINE_COMPUTE_SCALAR(div_f16, _Float16, hvx_div_scalar_f16_aa(r_dst, r_src0, val, ne00))

ਕਿਸਮਪਰਿਭਾਸ਼ਾ ਖਾਲੀ (*compute_same_shape_t)(
    uint8_t * dst, ਅਚਲ uint8_t * src0, ਅਚਲ uint8_t * src1,
    uint32_t n_ਪੰਕਤੀਆਂ, size_t dst_stride, size_t src0_stride, size_t src1_stride, uint32_t ne00);

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ DEFINE_COMPUTE_SAME_SHAPE(NAME, HVX_FN)                                                  \
ਸਥਿਰ ਖਾਲੀ compute_same_shape_##NAME(                                                           \
    uint8_t * dst, ਅਚਲ uint8_t * src0, ਅਚਲ uint8_t * src1,                                   \
    uint32_t n_ਪੰਕਤੀਆਂ, size_t dst_stride, size_t src0_stride, size_t src1_stride, uint32_t ne00) { \
    ਲਈ (uint32_t r = 0; r < n_ਪੰਕਤੀਆਂ; r++) {                                                      \
        HVX_FN(dst + r * dst_stride, src0 + r * src0_stride, src1 + r * src1_stride, ne00);      \
    }                                                                                            \
}

DEFINE_COMPUTE_SAME_SHAPE(ਜੋੜੋ_f32, hvx_ਜੋੜੋ_f32_aaa)
DEFINE_COMPUTE_SAME_SHAPE(ਜੋੜੋ_f16, hvx_ਜੋੜੋ_f16_aaa)
DEFINE_COMPUTE_SAME_SHAPE(sub_f32, hvx_sub_f32_aaa)
DEFINE_COMPUTE_SAME_SHAPE(sub_f16, hvx_sub_f16_aaa)
DEFINE_COMPUTE_SAME_SHAPE(mul_f32, hvx_mul_f32_aaa)
DEFINE_COMPUTE_SAME_SHAPE(mul_f16, hvx_mul_f16_aaa)
DEFINE_COMPUTE_SAME_SHAPE(div_f32, hvx_div_f32_aaa)
DEFINE_COMPUTE_SAME_SHAPE(div_f16, hvx_div_f16_aaa)

ਕਿਸਮਪਰਿਭਾਸ਼ਾ ਖਾਲੀ (*compute_ਪੰਕਤੀ_bcast_t)(
    uint8_t * dst, ਅਚਲ uint8_t * src0, ਅਚਲ uint8_t * src1,
    uint32_t n_ਪੰਕਤੀਆਂ, size_t dst_stride, size_t src0_stride, uint32_t ne00);

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ DEFINE_COMPUTE_ਪੰਕਤੀ_BCAST(NAME, HVX_FN)                                               \
ਸਥਿਰ ਖਾਲੀ compute_ਪੰਕਤੀ_bcast##NAME(                                                        \
    uint8_t * dst, ਅਚਲ uint8_t * src0, ਅਚਲ uint8_t * src1,                               \
    uint32_t n_ਪੰਕਤੀਆਂ, size_t dst_stride, size_t src0_stride, uint32_t ne00) {                 \
    ਲਈ (uint32_t r = 0; r < n_ਪੰਕਤੀਆਂ; r++) {                                                  \
        HVX_FN(dst + r * dst_stride, src0 + r * src0_stride, src1, ne00);                    \
    }                                                                                        \
}

DEFINE_COMPUTE_ਪੰਕਤੀ_BCAST(ਜੋੜੋ_f32, hvx_ਜੋੜੋ_f32_aaa)
DEFINE_COMPUTE_ਪੰਕਤੀ_BCAST(ਜੋੜੋ_f16, hvx_ਜੋੜੋ_f16_aaa)
DEFINE_COMPUTE_ਪੰਕਤੀ_BCAST(sub_f32, hvx_sub_f32_aaa)
DEFINE_COMPUTE_ਪੰਕਤੀ_BCAST(sub_f16, hvx_sub_f16_aaa)
DEFINE_COMPUTE_ਪੰਕਤੀ_BCAST(mul_f32, hvx_mul_f32_aaa)
DEFINE_COMPUTE_ਪੰਕਤੀ_BCAST(mul_f16, hvx_mul_f16_aaa)
DEFINE_COMPUTE_ਪੰਕਤੀ_BCAST(div_f32, hvx_div_f32_aaa)
DEFINE_COMPUTE_ਪੰਕਤੀ_BCAST(div_f16, hvx_div_f16_aaa)

ਕਿਸਮਪਰਿਭਾਸ਼ਾ ਖਾਲੀ (*compute_complex_t)(
    uint8_t * dst, ਅਚਲ uint8_t * src0, ਅਚਲ uint8_t * src1_plane,
    uint32_t i01, uint32_t ne11, ਅਚਲ ਬਣਤਰ fastdiv_ਮੁੱਲਾਂ * div11, uint32_t src1_stride,
    uint32_t n_ਪੰਕਤੀਆਂ, size_t dst_stride, size_t src0_stride, uint32_t ne00);

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ DEFINE_COMPUTE_COMPLEX(NAME, HVX_FN)                                                 \
ਸਥਿਰ ਖਾਲੀ compute_complex_##NAME(                                                          \
    uint8_t * dst, ਅਚਲ uint8_t * src0, ਅਚਲ uint8_t * src1_plane,                         \
    uint32_t i01, uint32_t ne11, ਅਚਲ ਬਣਤਰ fastdiv_ਮੁੱਲਾਂ * div11, uint32_t src1_stride,  \
    uint32_t n_ਪੰਕਤੀਆਂ, size_t dst_stride, size_t src0_stride, uint32_t ne00) {                 \
    ਲਈ (uint32_t r = 0; r < n_ਪੰਕਤੀਆਂ; r++) {                                                  \
        uint32_t i11 = fastmodulo(i01 + r, ne11, div11);                                     \
        ਅਚਲ uint8_t * r_src1 = src1_plane + i11 * src1_stride;                             \
        HVX_FN(dst + r * dst_stride, src0 + r * src0_stride, r_src1, ne00);                  \
    }                                                                                        \
}

DEFINE_COMPUTE_COMPLEX(ਜੋੜੋ_f32, hvx_ਜੋੜੋ_f32_aau)
DEFINE_COMPUTE_COMPLEX(ਜੋੜੋ_f16, hvx_ਜੋੜੋ_f16_aau)
DEFINE_COMPUTE_COMPLEX(sub_f32, hvx_sub_f32_aau)
DEFINE_COMPUTE_COMPLEX(sub_f16, hvx_sub_f16_aau)
DEFINE_COMPUTE_COMPLEX(mul_f32, hvx_mul_f32_aau)
DEFINE_COMPUTE_COMPLEX(mul_f16, hvx_mul_f16_aau)
DEFINE_COMPUTE_COMPLEX(div_f32, hvx_div_f32_aau)
DEFINE_COMPUTE_COMPLEX(div_f16, hvx_div_f16_aau)

ਕਿਸਮਪਰਿਭਾਸ਼ਾ ਖਾਲੀ (*compute_repeat_t)(
    uint8_t * dst, ਅਚਲ uint8_t * src0, ਅਚਲ uint8_t * src1_plane,
    uint32_t i01, uint32_t ne11, ਅਚਲ ਬਣਤਰ fastdiv_ਮੁੱਲਾਂ * div11, uint32_t src1_stride,
    uint32_t n_ਪੰਕਤੀਆਂ, size_t dst_stride, size_t src0_stride, uint32_t ne00, uint32_t ne10);

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ DEFINE_COMPUTE_REPEAT(NAME, TYPE, HVX_FN)                                            \
ਸਥਿਰ ਖਾਲੀ compute_repeat_##NAME(                                                           \
    uint8_t * dst, ਅਚਲ uint8_t * src0, ਅਚਲ uint8_t * src1_plane,                         \
    uint32_t i01, uint32_t ne11, ਅਚਲ ਬਣਤਰ fastdiv_ਮੁੱਲਾਂ * div11, uint32_t src1_stride,  \
    uint32_t n_ਪੰਕਤੀਆਂ, size_t dst_stride, size_t src0_stride, uint32_t ne00, uint32_t ne10) {  \
    ਲਈ (uint32_t r = 0; r < n_ਪੰਕਤੀਆਂ; r++) {                                                  \
        uint32_t i11 = fastmodulo(i01 + r, ne11, div11);                                     \
        ਅਚਲ uint8_t * r_src1_ਪੰਕਤੀ = src1_plane + i11 * src1_stride;                         \
        uint8_t * r_dst = dst + r * dst_stride;                                              \
        ਅਚਲ uint8_t * r_src0 = src0 + r * src0_stride;                                     \
        ਲਈ (uint32_t c = 0; c < ne00; c += ne10) {                                          \
            uint32_t len = MIN(ne10, ne00 - c);                                              \
            HVX_FN(r_dst + c * ਆਕਾਰ(TYPE), r_src0 + c * ਆਕਾਰ(TYPE), r_src1_ਪੰਕਤੀ, len);    \
        }                                                                                    \
    }                                                                                        \
}

DEFINE_COMPUTE_REPEAT(ਜੋੜੋ_f32, ਦਸ਼ਮਲਵ,    hvx_ਜੋੜੋ_f32_uuu)
DEFINE_COMPUTE_REPEAT(ਜੋੜੋ_f16, _Float16, hvx_ਜੋੜੋ_f16_uuu)
DEFINE_COMPUTE_REPEAT(sub_f32, ਦਸ਼ਮਲਵ,    hvx_sub_f32_uuu)
DEFINE_COMPUTE_REPEAT(sub_f16, _Float16, hvx_sub_f16_uuu)
DEFINE_COMPUTE_REPEAT(mul_f32, ਦਸ਼ਮਲਵ,    hvx_mul_f32_uuu)
DEFINE_COMPUTE_REPEAT(mul_f16, _Float16, hvx_mul_f16_uuu)
DEFINE_COMPUTE_REPEAT(div_f32, ਦਸ਼ਮਲਵ,    hvx_div_f32_uuu)
DEFINE_COMPUTE_REPEAT(div_f16, _Float16, hvx_div_f16_uuu)

ਕਿਸਮਪਰਿਭਾਸ਼ਾ ਖਾਲੀ (*compute_ਜੋੜੋ_id_t)(
    uint8_t * dst, ਅਚਲ uint8_t * src0, ਅਚਲ uint8_t * src1_ਅੰਕੜੇ, ਅਚਲ int32_t * src2_ids,
    uint32_t i01, uint32_t i02, uint32_t ne01, uint32_t src1_stride,
    uint32_t n_ਪੰਕਤੀਆਂ, size_t dst_stride, size_t src0_stride, uint32_t ne00);

ਸਥਿਰ ਖਾਲੀ compute_ਜੋੜੋ_id_f32(
    uint8_t * dst, ਅਚਲ uint8_t * src0, ਅਚਲ uint8_t * src1_ਅੰਕੜੇ, ਅਚਲ int32_t * src2_ids,
    uint32_t i01, uint32_t i02, uint32_t ne01, uint32_t src1_stride,
    uint32_t n_ਪੰਕਤੀਆਂ, size_t dst_stride, size_t src0_stride, uint32_t ne00) {
    ਲਈ (uint32_t r = 0; r < n_ਪੰਕਤੀਆਂ; r++) {
        uint32_t r_i01 = i01 + r;
        ਅਚਲ int32_t idx = src2_ids[i02 * ne01 + r_i01];
        ਜੇ (idx < 0) {
            ਸਮ੍ਰਿਤੀ_ਨਕਲ(dst + r * dst_stride, src0 + r * src0_stride, ne00 * ਆਕਾਰ(ਦਸ਼ਮਲਵ));
            ਜਾਰੀ;
        }
        ਅਚਲ uint8_t * r_src1 = src1_ਅੰਕੜੇ + idx * src1_stride;
        ਅਚਲ uint8_t * r_src0 = src0 + r * src0_stride;
        uint8_t * r_dst = dst + r * dst_stride;
        hvx_ਜੋੜੋ_f32_aaa(r_dst, r_src0, r_src1, ne00);
    }
}

// 1. Scalar src1 in VTCM via DMA (ne10 == 1)
ਸਥਿਰ ਖਾਲੀ ਦੁਆਧਾਰੀ_thread_scalar(ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ nth, ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ ith, ਖਾਲੀ * ਅੰਕੜੇ) {
    ਬਣਤਰ htp_ਦੁਆਧਾਰੀ_context * bctx = (ਬਣਤਰ htp_ਦੁਆਧਾਰੀ_context *) ਅੰਕੜੇ;
    ਬਣਤਰ htp_ops_context * octx = bctx->octx;
    htp_ਦੁਆਧਾਰੀ_preamble;

    ਅਚਲ uint32_t elem_ਮਾਪ = (src0->type == HTP_TYPE_F32) ? ਆਕਾਰ(ਦਸ਼ਮਲਵ) : ਆਕਾਰ(_Float16);
    ਅਚਲ size_t plane_stride = (size_t) ne12 * ne11 * elem_ਮਾਪ;
    ਅਚਲ size_t ਪੰਕਤੀ_stride   = (size_t) ne11 * elem_ਮਾਪ;

    ਅਚਲ uint32_t ਪੰਕਤੀ_ਮਾਪ_ਬਾਈਟਾਂ = bctx->ਪੰਕਤੀ_ਮਾਪ_ਬਾਈਟਾਂ;
    ਅਚਲ uint32_t ਸ਼ੁਰੂ_ਪੰਕਤੀ = bctx->ਪੰਕਤੀ_ਸ਼ੁਰੂ + bctx->nrows_ਪ੍ਰਤੀ_thread * ith;
    ਅਚਲ uint32_t ਸਿਰਾ_ਪੰਕਤੀ   = MIN(ਸ਼ੁਰੂ_ਪੰਕਤੀ + bctx->nrows_ਪ੍ਰਤੀ_thread, bctx->ਪੰਕਤੀ_ਸ਼ੁਰੂ + bctx->ਕੁੱਲ_ਪੰਕਤੀਆਂ);
    ਜੇ (ਸ਼ੁਰੂ_ਪੰਕਤੀ >= ਸਿਰਾ_ਪੰਕਤੀ) ਵਾਪਸ;

    FARF(ਉੱਚਾ_2, "binary-scalar: %d/%d (%u:%u) row-size %u (%u)",
         ith, nth, ਸ਼ੁਰੂ_ਪੰਕਤੀ, ਸਿਰਾ_ਪੰਕਤੀ, nb01, bctx->dst_ਪੰਕਤੀ_ਮਾਪ_aligned);

    ਅਚਲ ਬਣਤਰ htp_ਦੁਆਧਾਰੀ_vtcm_layout * layout = &bctx->vtcm_layout;
    uint8_t * src0_spad_ਅਧਾਰ = VTCM_LAYOUT_PTR(uint8_t, bctx->vtcm_ਅਧਾਰ, layout->off_src0) + (ith * layout->src0_ਬਾਈਟਾਂ_ਪ੍ਰਤੀ_thread);
    uint8_t * dst_spad_ਅਧਾਰ  = VTCM_LAYOUT_PTR(uint8_t, bctx->vtcm_ਅਧਾਰ, layout->off_dst)  + (ith * layout->dst_ਬਾਈਟਾਂ_ਪ੍ਰਤੀ_thread);
    size_t src0_spad_half    = layout->src0_spad_half_ਮਾਪ;
    size_t dst_spad_half     = layout->dst_spad_half_ਮਾਪ;
    ਅਚਲ ਖਾਲੀ * s1_ਸਾਰਣੀ    = VTCM_LAYOUT_PTR(ਅਚਲ ਖਾਲੀ, bctx->vtcm_ਅਧਾਰ, layout->off_src1);

    dma_ਕਤਾਰ * dma_q = octx->ctx->dma[ith];
    uint32_t ir_prefetch = ਸ਼ੁਰੂ_ਪੰਕਤੀ;
    ਪੂਰਨ spad_idx = 0;

    ਲਈ (ਪੂਰਨ k = 0; k < 2 && ir_prefetch < ਸਿਰਾ_ਪੰਕਤੀ; k++) {
        uint32_t current_ਖੰਡ_ਮਾਪ = calc_ਖੰਡ_ਮਾਪ(bctx, ir_prefetch, ਸਿਰਾ_ਪੰਕਤੀ, ne01, ne02);
        uint32_t i03, i02, i01, rem;
        i03 = fastdiv(ir_prefetch, &bctx->src0_dim12_div);
        rem = ir_prefetch - i03 * (ne02 * ne01);
        i02 = fastdiv(rem, &bctx->src0_dim1_div);
        i01 = rem - i02 * ne01;

        dma_addr_t src0_curr = src0->ਅੰਕੜੇ + i03 * nb03 + i02 * nb02 + i01 * nb01;
        dma_addr_t dst_curr  = dst->ਅੰਕੜੇ  + i03 * nb3  + i02 * nb2  + i01 * nb1;

        uint8_t * s0_spad = src0_spad_ਅਧਾਰ + spad_idx * src0_spad_half;
        uint8_t * d_spad  = dst_spad_ਅਧਾਰ  + spad_idx * dst_spad_half;

        dma_ਕਤਾਰ_ਧੱਕੋ(dma_q, dma_make_ਅੰਕੜੇ(dst_curr, d_spad), nb1, bctx->dst_ਪੰਕਤੀ_ਮਾਪ_aligned, ਪੰਕਤੀ_ਮਾਪ_ਬਾਈਟਾਂ, 0);
        dma_ਕਤਾਰ_ਧੱਕੋ(dma_q, dma_make_ਅੰਕੜੇ(s0_spad, src0_curr), bctx->src0_ਪੰਕਤੀ_ਮਾਪ_aligned, nb01, ਪੰਕਤੀ_ਮਾਪ_ਬਾਈਟਾਂ,
                                                             current_ਖੰਡ_ਮਾਪ);
        ir_prefetch += current_ਖੰਡ_ਮਾਪ;
        spad_idx ^= 1;
    }

    ਬਣਤਰ htp_thread_trace * tr = &octx->ctx->trace[ith];
    compute_scalar_t compute = (compute_scalar_t) bctx->compute;

    ਲਈ (uint32_t ir = ਸ਼ੁਰੂ_ਪੰਕਤੀ; ir < ਸਿਰਾ_ਪੰਕਤੀ; ) {
        uint32_t current_ਖੰਡ_ਮਾਪ = calc_ਖੰਡ_ਮਾਪ(bctx, ir, ਸਿਰਾ_ਪੰਕਤੀ, ne01, ne02);

        uint8_t * d_spad  = (uint8_t *) dma_ਕਤਾਰ_ਕੱਢੋ(dma_q).src;
        uint8_t * s0_spad = (uint8_t *) dma_ਕਤਾਰ_ਕੱਢੋ(dma_q).dst;

        uint32_t i03, i02, i01, rem;
        i03 = fastdiv(ir, &bctx->src0_dim12_div);
        rem = ir - i03 * (ne02 * ne01);
        i02 = fastdiv(rem, &bctx->src0_dim1_div);
        i01 = rem - i02 * ne01;

        uint32_t i13 = fastmodulo(i03, ne13, &bctx->src1_dim3_div);
        uint32_t i12 = fastmodulo(i02, ne12, &bctx->src1_dim2_div);
        uint32_t i11 = fastmodulo(i01, ne11, &bctx->src1_dim1_div);
        ਅਚਲ uint8_t * scalar_ਸਾਰਣੀ = (ਅਚਲ uint8_t *) s1_ਸਾਰਣੀ +
            i13 * plane_stride + i12 * ਪੰਕਤੀ_stride;

        htp_trace_event_ਸ਼ੁਰੂ(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);
        compute(d_spad, s0_spad, scalar_ਸਾਰਣੀ, i11, ne11, current_ਖੰਡ_ਮਾਪ,
                bctx->dst_ਪੰਕਤੀ_ਮਾਪ_aligned, bctx->src0_ਪੰਕਤੀ_ਮਾਪ_aligned, ne00);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);

        dma_addr_t dst_curr = dst->ਅੰਕੜੇ + i03 * nb3 + i02 * nb2 + i01 * nb1;
        dma_ਕਤਾਰ_ਧੱਕੋ(dma_q, dma_make_ਅੰਕੜੇ(dst_curr, d_spad), nb1, bctx->dst_ਪੰਕਤੀ_ਮਾਪ_aligned, ਪੰਕਤੀ_ਮਾਪ_ਬਾਈਟਾਂ,
                                                                current_ਖੰਡ_ਮਾਪ);

        ਜੇ (ir_prefetch < ਸਿਰਾ_ਪੰਕਤੀ) {
            uint32_t ਅਗਲਾ_ਖੰਡ_ਮਾਪ = calc_ਖੰਡ_ਮਾਪ(bctx, ir_prefetch, ਸਿਰਾ_ਪੰਕਤੀ, ne01, ne02);
            uint32_t p03, p02, p01, prem;
            p03 = fastdiv(ir_prefetch, &bctx->src0_dim12_div);
            prem = ir_prefetch - p03 * (ne02 * ne01);
            p02 = fastdiv(prem, &bctx->src0_dim1_div);
            p01 = prem - p02 * ne01;
            dma_addr_t s0_ਅਗਲਾ = src0->ਅੰਕੜੇ + p03 * nb03 + p02 * nb02 + p01 * nb01;
            dma_ਕਤਾਰ_ਧੱਕੋ(dma_q, dma_make_ਅੰਕੜੇ(s0_spad, s0_ਅਗਲਾ), bctx->src0_ਪੰਕਤੀ_ਮਾਪ_aligned, nb01, ਪੰਕਤੀ_ਮਾਪ_ਬਾਈਟਾਂ,
                                                               ਅਗਲਾ_ਖੰਡ_ਮਾਪ);
            ir_prefetch += ਅਗਲਾ_ਖੰਡ_ਮਾਪ;
        }
        ir += current_ਖੰਡ_ਮਾਪ;
    }

    dma_ਕਤਾਰ_flush(dma_q);
}

// 2. Vector Same Shape (ne1x == ne0x) or Simple Broadcast
ਸਥਿਰ ਖਾਲੀ ਦੁਆਧਾਰੀ_thread_ਸਦਿਸ਼_same_shape(ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ nth, ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ ith, ਖਾਲੀ * ਅੰਕੜੇ) {
    ਬਣਤਰ htp_ਦੁਆਧਾਰੀ_context * bctx = (ਬਣਤਰ htp_ਦੁਆਧਾਰੀ_context *) ਅੰਕੜੇ;
    ਬਣਤਰ htp_ops_context * octx = bctx->octx;
    htp_ਦੁਆਧਾਰੀ_preamble;

    ਅਚਲ uint32_t ਪੰਕਤੀ_ਮਾਪ_ਬਾਈਟਾਂ = bctx->ਪੰਕਤੀ_ਮਾਪ_ਬਾਈਟਾਂ;
    ਅਚਲ uint32_t ਸ਼ੁਰੂ_ਪੰਕਤੀ = bctx->ਪੰਕਤੀ_ਸ਼ੁਰੂ + bctx->nrows_ਪ੍ਰਤੀ_thread * ith;
    ਅਚਲ uint32_t ਸਿਰਾ_ਪੰਕਤੀ   = MIN(ਸ਼ੁਰੂ_ਪੰਕਤੀ + bctx->nrows_ਪ੍ਰਤੀ_thread, bctx->ਪੰਕਤੀ_ਸ਼ੁਰੂ + bctx->ਕੁੱਲ_ਪੰਕਤੀਆਂ);
    ਜੇ (ਸ਼ੁਰੂ_ਪੰਕਤੀ >= ਸਿਰਾ_ਪੰਕਤੀ) ਵਾਪਸ;

    FARF(ਉੱਚਾ_2, "binary-same-shape: %d/%d (%u:%u) row-size %u (%u)",
         ith, nth, ਸ਼ੁਰੂ_ਪੰਕਤੀ, ਸਿਰਾ_ਪੰਕਤੀ, nb01, bctx->dst_ਪੰਕਤੀ_ਮਾਪ_aligned);

    ਅਚਲ ਬਣਤਰ htp_ਦੁਆਧਾਰੀ_vtcm_layout * layout = &bctx->vtcm_layout;
    uint8_t * src0_spad_ਅਧਾਰ = VTCM_LAYOUT_PTR(uint8_t, bctx->vtcm_ਅਧਾਰ, layout->off_src0) + (ith * layout->src0_ਬਾਈਟਾਂ_ਪ੍ਰਤੀ_thread);
    uint8_t * src1_spad_ਅਧਾਰ = VTCM_LAYOUT_PTR(uint8_t, bctx->vtcm_ਅਧਾਰ, layout->off_src1) + (ith * layout->src1_ਬਾਈਟਾਂ_ਪ੍ਰਤੀ_thread);
    uint8_t * dst_spad_ਅਧਾਰ  = VTCM_LAYOUT_PTR(uint8_t, bctx->vtcm_ਅਧਾਰ, layout->off_dst)  + (ith * layout->dst_ਬਾਈਟਾਂ_ਪ੍ਰਤੀ_thread);

    size_t src0_spad_half = layout->src0_spad_half_ਮਾਪ;
    size_t src1_spad_half = layout->src1_spad_half_ਮਾਪ;
    size_t dst_spad_half  = layout->dst_spad_half_ਮਾਪ;

    dma_ਕਤਾਰ * dma_q = octx->ctx->dma[ith];
    uint32_t ir_prefetch = ਸ਼ੁਰੂ_ਪੰਕਤੀ;
    ਪੂਰਨ spad_idx = 0;

    ਲਈ (ਪੂਰਨ k = 0; k < 2 && ir_prefetch < ਸਿਰਾ_ਪੰਕਤੀ; k++) {
        uint32_t current_ਖੰਡ_ਮਾਪ = calc_ਖੰਡ_ਮਾਪ(bctx, ir_prefetch, ਸਿਰਾ_ਪੰਕਤੀ, ne01, ne02);
        uint32_t i03, i02, i01, rem;
        i03 = fastdiv(ir_prefetch, &bctx->src0_dim12_div);
        rem = ir_prefetch - i03 * (ne02 * ne01);
        i02 = fastdiv(rem, &bctx->src0_dim1_div);
        i01 = rem - i02 * ne01;

        uint32_t i13 = (ne13 == 1) ? 0 : i03;
        uint32_t i12 = (ne12 == 1) ? 0 : i02;
        uint32_t i11 = (ne11 == 1) ? 0 : i01;

        dma_addr_t src0_curr = src0->ਅੰਕੜੇ + i03 * nb03 + i02 * nb02 + i01 * nb01;
        dma_addr_t src1_curr = src1->ਅੰਕੜੇ + i13 * nb13 + i12 * nb12 + i11 * nb11;
        dma_addr_t dst_curr  = dst->ਅੰਕੜੇ  + i03 * nb3  + i02 * nb2  + i01 * nb1;

        uint8_t * s0_spad = src0_spad_ਅਧਾਰ + spad_idx * src0_spad_half;
        uint8_t * s1_spad = src1_spad_ਅਧਾਰ + spad_idx * src1_spad_half;
        uint8_t * d_spad  = dst_spad_ਅਧਾਰ  + spad_idx * dst_spad_half;

        dma_ਕਤਾਰ_ਧੱਕੋ(dma_q, dma_make_ਅੰਕੜੇ(dst_curr, d_spad), nb1, bctx->dst_ਪੰਕਤੀ_ਮਾਪ_aligned, ਪੰਕਤੀ_ਮਾਪ_ਬਾਈਟਾਂ, 0);
        dma_ਕਤਾਰ_ਧੱਕੋ(dma_q, dma_make_ਅੰਕੜੇ(s0_spad, src0_curr), bctx->src0_ਪੰਕਤੀ_ਮਾਪ_aligned, nb01, ਪੰਕਤੀ_ਮਾਪ_ਬਾਈਟਾਂ, current_ਖੰਡ_ਮਾਪ);
        dma_ਕਤਾਰ_ਧੱਕੋ(dma_q, dma_make_ਅੰਕੜੇ(s1_spad, src1_curr), bctx->src1_ਪੰਕਤੀ_ਮਾਪ_aligned, nb11, ਪੰਕਤੀ_ਮਾਪ_ਬਾਈਟਾਂ, current_ਖੰਡ_ਮਾਪ);
        ir_prefetch += current_ਖੰਡ_ਮਾਪ;
        spad_idx ^= 1;
    }

    ਬਣਤਰ htp_thread_trace * tr = &octx->ctx->trace[ith];
    compute_same_shape_t compute = (compute_same_shape_t) bctx->compute;

    ਲਈ (uint32_t ir = ਸ਼ੁਰੂ_ਪੰਕਤੀ; ir < ਸਿਰਾ_ਪੰਕਤੀ; ) {
        uint32_t current_ਖੰਡ_ਮਾਪ = calc_ਖੰਡ_ਮਾਪ(bctx, ir, ਸਿਰਾ_ਪੰਕਤੀ, ne01, ne02);
        uint8_t * d_spad  = (uint8_t *) dma_ਕਤਾਰ_ਕੱਢੋ(dma_q).src;
        uint8_t * s0_spad = (uint8_t *) dma_ਕਤਾਰ_ਕੱਢੋ(dma_q).dst;
        uint8_t * s1_spad = (uint8_t *) dma_ਕਤਾਰ_ਕੱਢੋ(dma_q).dst;

        htp_trace_event_ਸ਼ੁਰੂ(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);
        compute(d_spad, s0_spad, s1_spad, current_ਖੰਡ_ਮਾਪ,
                bctx->dst_ਪੰਕਤੀ_ਮਾਪ_aligned, bctx->src0_ਪੰਕਤੀ_ਮਾਪ_aligned, bctx->src1_ਪੰਕਤੀ_ਮਾਪ_aligned, ne00);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);

        uint32_t i03, i02, i01, rem;
        i03 = fastdiv(ir, &bctx->src0_dim12_div);
        rem = ir - i03 * (ne02 * ne01);
        i02 = fastdiv(rem, &bctx->src0_dim1_div);
        i01 = rem - i02 * ne01;
        dma_addr_t dst_curr = dst->ਅੰਕੜੇ + i03 * nb3 + i02 * nb2 + i01 * nb1;
        dma_ਕਤਾਰ_ਧੱਕੋ(dma_q, dma_make_ਅੰਕੜੇ(dst_curr, d_spad), nb1, bctx->dst_ਪੰਕਤੀ_ਮਾਪ_aligned, ਪੰਕਤੀ_ਮਾਪ_ਬਾਈਟਾਂ, current_ਖੰਡ_ਮਾਪ);

        ਜੇ (ir_prefetch < ਸਿਰਾ_ਪੰਕਤੀ) {
            uint32_t ਅਗਲਾ_ਖੰਡ_ਮਾਪ = calc_ਖੰਡ_ਮਾਪ(bctx, ir_prefetch, ਸਿਰਾ_ਪੰਕਤੀ, ne01, ne02);
            uint32_t p03, p02, p01, prem;
            p03 = fastdiv(ir_prefetch, &bctx->src0_dim12_div);
            prem = ir_prefetch - p03 * (ne02 * ne01);
            p02 = fastdiv(prem, &bctx->src0_dim1_div);
            p01 = prem - p02 * ne01;

            uint32_t p13 = (ne13 == 1) ? 0 : p03;
            uint32_t p12 = (ne12 == 1) ? 0 : p02;
            uint32_t p11 = (ne11 == 1) ? 0 : p01;

            dma_addr_t s0_ਅਗਲਾ = src0->ਅੰਕੜੇ + p03 * nb03 + p02 * nb02 + p01 * nb01;
            dma_addr_t s1_ਅਗਲਾ = src1->ਅੰਕੜੇ + p13 * nb13 + p12 * nb12 + p11 * nb11;

            dma_ਕਤਾਰ_ਧੱਕੋ(dma_q, dma_make_ਅੰਕੜੇ(s0_spad, s0_ਅਗਲਾ), bctx->src0_ਪੰਕਤੀ_ਮਾਪ_aligned, nb01, ਪੰਕਤੀ_ਮਾਪ_ਬਾਈਟਾਂ, ਅਗਲਾ_ਖੰਡ_ਮਾਪ);
            dma_ਕਤਾਰ_ਧੱਕੋ(dma_q, dma_make_ਅੰਕੜੇ(s1_spad, s1_ਅਗਲਾ), bctx->src1_ਪੰਕਤੀ_ਮਾਪ_aligned, nb11, ਪੰਕਤੀ_ਮਾਪ_ਬਾਈਟਾਂ, ਅਗਲਾ_ਖੰਡ_ਮਾਪ);

            ir_prefetch += ਅਗਲਾ_ਖੰਡ_ਮਾਪ;
        }
        ir += current_ਖੰਡ_ਮਾਪ;
    }

    dma_ਕਤਾਰ_flush(dma_q);
}

// 3. Row Broadcast (ne11 == 1, ne12 == 1, single row src1)
ਸਥਿਰ ਖਾਲੀ ਦੁਆਧਾਰੀ_thread_ਸਦਿਸ਼_ਪੰਕਤੀ_ਪ੍ਰਸਾਰਣ(ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ nth, ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ ith, ਖਾਲੀ * ਅੰਕੜੇ) {
    ਬਣਤਰ htp_ਦੁਆਧਾਰੀ_context * bctx = (ਬਣਤਰ htp_ਦੁਆਧਾਰੀ_context *) ਅੰਕੜੇ;
    ਬਣਤਰ htp_ops_context * octx = bctx->octx;
    htp_ਦੁਆਧਾਰੀ_preamble;

    ਅਚਲ uint32_t ਪੰਕਤੀ_ਮਾਪ_ਬਾਈਟਾਂ = bctx->ਪੰਕਤੀ_ਮਾਪ_ਬਾਈਟਾਂ;
    ਅਚਲ uint32_t ਸ਼ੁਰੂ_ਪੰਕਤੀ  = bctx->ਪੰਕਤੀ_ਸ਼ੁਰੂ + bctx->nrows_ਪ੍ਰਤੀ_thread * ith;
    ਅਚਲ uint32_t ਸਿਰਾ_ਪੰਕਤੀ    = MIN(ਸ਼ੁਰੂ_ਪੰਕਤੀ + bctx->nrows_ਪ੍ਰਤੀ_thread, bctx->ਪੰਕਤੀ_ਸ਼ੁਰੂ + bctx->ਕੁੱਲ_ਪੰਕਤੀਆਂ);
    ਜੇ (ਸ਼ੁਰੂ_ਪੰਕਤੀ >= ਸਿਰਾ_ਪੰਕਤੀ) ਵਾਪਸ;

    FARF(ਉੱਚਾ_2, "binary-row-bcast: %d/%d (%u:%u) row-size %u (%u)",
         ith, nth, ਸ਼ੁਰੂ_ਪੰਕਤੀ, ਸਿਰਾ_ਪੰਕਤੀ, nb01, bctx->dst_ਪੰਕਤੀ_ਮਾਪ_aligned);

    ਅਚਲ ਬਣਤਰ htp_ਦੁਆਧਾਰੀ_vtcm_layout * layout = &bctx->vtcm_layout;
    uint8_t * src0_spad_ਅਧਾਰ = VTCM_LAYOUT_PTR(uint8_t, bctx->vtcm_ਅਧਾਰ, layout->off_src0) + (ith * layout->src0_ਬਾਈਟਾਂ_ਪ੍ਰਤੀ_thread);
    uint8_t * dst_spad_ਅਧਾਰ  = VTCM_LAYOUT_PTR(uint8_t, bctx->vtcm_ਅਧਾਰ, layout->off_dst)  + (ith * layout->dst_ਬਾਈਟਾਂ_ਪ੍ਰਤੀ_thread);

    size_t src0_spad_half = layout->src0_spad_half_ਮਾਪ;
    size_t dst_spad_half  = layout->dst_spad_half_ਮਾਪ;

    dma_ਕਤਾਰ * dma_q = octx->ctx->dma[ith];
    uint32_t ir_prefetch = ਸ਼ੁਰੂ_ਪੰਕਤੀ;
    ਪੂਰਨ spad_idx = 0;

    ਖਾਲੀ * s1_ptr = VTCM_LAYOUT_PTR(ਖਾਲੀ, bctx->vtcm_ਅਧਾਰ, layout->off_src1);

    ਲਈ (ਪੂਰਨ k = 0; k < 2 && ir_prefetch < ਸਿਰਾ_ਪੰਕਤੀ; k++) {
        uint32_t current_ਖੰਡ_ਮਾਪ = calc_ਖੰਡ_ਮਾਪ(bctx, ir_prefetch, ਸਿਰਾ_ਪੰਕਤੀ, ne01, ne02);
        uint32_t i03 = fastdiv(ir_prefetch, &bctx->src0_dim12_div);
        uint32_t rem = ir_prefetch - i03 * (ne02 * ne01);
        uint32_t i02 = fastdiv(rem, &bctx->src0_dim1_div);
        uint32_t i01 = rem - i02 * ne01;

        dma_addr_t src0_curr = src0->ਅੰਕੜੇ + i03 * nb03 + i02 * nb02 + i01 * nb01;
        dma_addr_t dst_curr  = dst->ਅੰਕੜੇ  + i03 * nb3  + i02 * nb2  + i01 * nb1;

        uint8_t * s0_spad = src0_spad_ਅਧਾਰ + spad_idx * src0_spad_half;
        uint8_t * d_spad  = dst_spad_ਅਧਾਰ  + spad_idx * dst_spad_half;

        dma_ਕਤਾਰ_ਧੱਕੋ(dma_q, dma_make_ਅੰਕੜੇ(dst_curr, d_spad), nb1, bctx->dst_ਪੰਕਤੀ_ਮਾਪ_aligned, ਪੰਕਤੀ_ਮਾਪ_ਬਾਈਟਾਂ, 0);
        dma_ਕਤਾਰ_ਧੱਕੋ(dma_q, dma_make_ਅੰਕੜੇ(s0_spad, src0_curr), bctx->src0_ਪੰਕਤੀ_ਮਾਪ_aligned, nb01, ਪੰਕਤੀ_ਮਾਪ_ਬਾਈਟਾਂ, current_ਖੰਡ_ਮਾਪ);
        ir_prefetch += current_ਖੰਡ_ਮਾਪ;
        spad_idx ^= 1;
    }

    ਬਣਤਰ htp_thread_trace * tr = &octx->ctx->trace[ith];
    compute_ਪੰਕਤੀ_bcast_t compute = (compute_ਪੰਕਤੀ_bcast_t) bctx->compute;

    ਲਈ (uint32_t ir = ਸ਼ੁਰੂ_ਪੰਕਤੀ; ir < ਸਿਰਾ_ਪੰਕਤੀ; ) {
        uint32_t current_ਖੰਡ_ਮਾਪ = calc_ਖੰਡ_ਮਾਪ(bctx, ir, ਸਿਰਾ_ਪੰਕਤੀ, ne01, ne02);
        uint8_t * d_spad  = (uint8_t *) dma_ਕਤਾਰ_ਕੱਢੋ(dma_q).src;
        uint8_t * s0_spad = (uint8_t *) dma_ਕਤਾਰ_ਕੱਢੋ(dma_q).dst;

        htp_trace_event_ਸ਼ੁਰੂ(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);
        compute(d_spad, s0_spad, (ਅਚਲ uint8_t *)s1_ptr, current_ਖੰਡ_ਮਾਪ,
                bctx->dst_ਪੰਕਤੀ_ਮਾਪ_aligned, bctx->src0_ਪੰਕਤੀ_ਮਾਪ_aligned, ne00);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);

        uint32_t i03 = fastdiv(ir, &bctx->src0_dim12_div);
        uint32_t rem = ir - i03 * (ne02 * ne01);
        uint32_t i02 = fastdiv(rem, &bctx->src0_dim1_div);
        uint32_t i01 = rem - i02 * ne01;
        dma_addr_t dst_curr = dst->ਅੰਕੜੇ + i03 * nb3 + i02 * nb2 + i01 * nb1;
        dma_ਕਤਾਰ_ਧੱਕੋ(dma_q, dma_make_ਅੰਕੜੇ(dst_curr, d_spad), nb1, bctx->dst_ਪੰਕਤੀ_ਮਾਪ_aligned, ਪੰਕਤੀ_ਮਾਪ_ਬਾਈਟਾਂ, current_ਖੰਡ_ਮਾਪ);

        ਜੇ (ir_prefetch < ਸਿਰਾ_ਪੰਕਤੀ) {
            uint32_t ਅਗਲਾ_ਖੰਡ_ਮਾਪ = calc_ਖੰਡ_ਮਾਪ(bctx, ir_prefetch, ਸਿਰਾ_ਪੰਕਤੀ, ne01, ne02);
            uint32_t p03, p02, p01, prem;
            p03  = fastdiv(ir_prefetch, &bctx->src0_dim12_div);
            prem = ir_prefetch - p03 * (ne02 * ne01);
            p02  = fastdiv(prem, &bctx->src0_dim1_div);
            p01  = prem - p02 * ne01;
            dma_addr_t s0_ਅਗਲਾ = src0->ਅੰਕੜੇ + p03 * nb03 + p02 * nb02 + p01 * nb01;
            dma_ਕਤਾਰ_ਧੱਕੋ(dma_q, dma_make_ਅੰਕੜੇ(s0_spad, s0_ਅਗਲਾ), bctx->src0_ਪੰਕਤੀ_ਮਾਪ_aligned, nb01, ਪੰਕਤੀ_ਮਾਪ_ਬਾਈਟਾਂ, ਅਗਲਾ_ਖੰਡ_ਮਾਪ);
            ir_prefetch += ਅਗਲਾ_ਖੰਡ_ਮਾਪ;
        }
        ir += current_ਖੰਡ_ਮਾਪ;
    }

    dma_ਕਤਾਰ_flush(dma_q);
}

// 4. Vector Complex (ne10 == ne00, complex broadcast)
ਸਥਿਰ ਖਾਲੀ ਦੁਆਧਾਰੀ_thread_ਸਦਿਸ਼_complex(ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ nth, ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ ith, ਖਾਲੀ * ਅੰਕੜੇ) {
    ਬਣਤਰ htp_ਦੁਆਧਾਰੀ_context * bctx = (ਬਣਤਰ htp_ਦੁਆਧਾਰੀ_context *) ਅੰਕੜੇ;
    ਬਣਤਰ htp_ops_context * octx = bctx->octx;
    htp_ਦੁਆਧਾਰੀ_preamble;

    ਅਚਲ uint32_t ਪੰਕਤੀ_ਮਾਪ_ਬਾਈਟਾਂ = bctx->ਪੰਕਤੀ_ਮਾਪ_ਬਾਈਟਾਂ;
    ਅਚਲ uint32_t ਸ਼ੁਰੂ_ਪੰਕਤੀ  = bctx->ਪੰਕਤੀ_ਸ਼ੁਰੂ + bctx->nrows_ਪ੍ਰਤੀ_thread * ith;
    ਅਚਲ uint32_t ਸਿਰਾ_ਪੰਕਤੀ    = MIN(ਸ਼ੁਰੂ_ਪੰਕਤੀ + bctx->nrows_ਪ੍ਰਤੀ_thread, bctx->ਪੰਕਤੀ_ਸ਼ੁਰੂ + bctx->ਕੁੱਲ_ਪੰਕਤੀਆਂ);
    ਜੇ (ਸ਼ੁਰੂ_ਪੰਕਤੀ >= ਸਿਰਾ_ਪੰਕਤੀ) ਵਾਪਸ;

    FARF(ਉੱਚਾ_2, "binary-complex: %d/%d (%u:%u) row-size %u (%u)",
         ith, nth, ਸ਼ੁਰੂ_ਪੰਕਤੀ, ਸਿਰਾ_ਪੰਕਤੀ, nb01, bctx->dst_ਪੰਕਤੀ_ਮਾਪ_aligned);

    ਅਚਲ ਬਣਤਰ htp_ਦੁਆਧਾਰੀ_vtcm_layout * layout = &bctx->vtcm_layout;
    uint8_t * src0_spad_ਅਧਾਰ = VTCM_LAYOUT_PTR(uint8_t, bctx->vtcm_ਅਧਾਰ, layout->off_src0) + (ith * layout->src0_ਬਾਈਟਾਂ_ਪ੍ਰਤੀ_thread);
    uint8_t * dst_spad_ਅਧਾਰ  = VTCM_LAYOUT_PTR(uint8_t, bctx->vtcm_ਅਧਾਰ, layout->off_dst)  + (ith * layout->dst_ਬਾਈਟਾਂ_ਪ੍ਰਤੀ_thread);
    size_t src0_spad_half    = layout->src0_spad_half_ਮਾਪ;
    size_t dst_spad_half     = layout->dst_spad_half_ਮਾਪ;
    ਅਚਲ uint8_t * vtcm_src1 = VTCM_LAYOUT_PTR(ਅਚਲ uint8_t, bctx->vtcm_ਅਧਾਰ, layout->off_src1);

    ਅਚਲ uint32_t elem_ਮਾਪ = (src0->type == HTP_TYPE_F32) ? ਆਕਾਰ(ਦਸ਼ਮਲਵ) : ਆਕਾਰ(_Float16);
    ਅਚਲ size_t vtcm_ਪੰਕਤੀ_stride   = (size_t) ne10 * elem_ਮਾਪ;
    ਅਚਲ size_t vtcm_plane_stride = (size_t) ne11 * vtcm_ਪੰਕਤੀ_stride;
    ਅਚਲ size_t vtcm_vol_stride   = (size_t) ne12 * vtcm_plane_stride;

    dma_ਕਤਾਰ * dma_q = octx->ctx->dma[ith];
    uint32_t ir_prefetch = ਸ਼ੁਰੂ_ਪੰਕਤੀ;
    ਪੂਰਨ spad_idx = 0;

    ਲਈ (ਪੂਰਨ k = 0; k < 2 && ir_prefetch < ਸਿਰਾ_ਪੰਕਤੀ; k++) {
        uint32_t current_ਖੰਡ_ਮਾਪ = calc_ਖੰਡ_ਮਾਪ(bctx, ir_prefetch, ਸਿਰਾ_ਪੰਕਤੀ, ne01, ne02);
        uint32_t i03 = fastdiv(ir_prefetch, &bctx->src0_dim12_div);
        uint32_t rem = ir_prefetch - i03 * (ne02 * ne01);
        uint32_t i02 = fastdiv(rem, &bctx->src0_dim1_div);
        uint32_t i01 = rem - i02 * ne01;

        dma_addr_t src0_curr = src0->ਅੰਕੜੇ + i03 * nb03 + i02 * nb02 + i01 * nb01;
        dma_addr_t dst_curr  = dst->ਅੰਕੜੇ  + i03 * nb3  + i02 * nb2  + i01 * nb1;

        uint8_t * s0_spad = src0_spad_ਅਧਾਰ + spad_idx * src0_spad_half;
        uint8_t * d_spad  = dst_spad_ਅਧਾਰ  + spad_idx * dst_spad_half;

        dma_ਕਤਾਰ_ਧੱਕੋ(dma_q, dma_make_ਅੰਕੜੇ(dst_curr, d_spad), nb1, bctx->dst_ਪੰਕਤੀ_ਮਾਪ_aligned, ਪੰਕਤੀ_ਮਾਪ_ਬਾਈਟਾਂ, 0);
        dma_ਕਤਾਰ_ਧੱਕੋ(dma_q, dma_make_ਅੰਕੜੇ(s0_spad, src0_curr), bctx->src0_ਪੰਕਤੀ_ਮਾਪ_aligned, nb01, ਪੰਕਤੀ_ਮਾਪ_ਬਾਈਟਾਂ, current_ਖੰਡ_ਮਾਪ);
        ir_prefetch += current_ਖੰਡ_ਮਾਪ;
        spad_idx ^= 1;
    }

    ਬਣਤਰ htp_thread_trace * tr = &octx->ctx->trace[ith];
    compute_complex_t compute = (compute_complex_t) bctx->compute;

    ਲਈ (uint32_t ir = ਸ਼ੁਰੂ_ਪੰਕਤੀ; ir < ਸਿਰਾ_ਪੰਕਤੀ; ) {
        uint32_t current_ਖੰਡ_ਮਾਪ = calc_ਖੰਡ_ਮਾਪ(bctx, ir, ਸਿਰਾ_ਪੰਕਤੀ, ne01, ne02);
        uint8_t * d_spad = (uint8_t *) dma_ਕਤਾਰ_ਕੱਢੋ(dma_q).src;
        uint8_t * s0_spad = (uint8_t *) dma_ਕਤਾਰ_ਕੱਢੋ(dma_q).dst;

        uint32_t i03 = fastdiv(ir, &bctx->src0_dim12_div);
        uint32_t rem = ir - i03 * (ne02 * ne01);
        uint32_t i02 = fastdiv(rem, &bctx->src0_dim1_div);
        uint32_t i01 = rem - i02 * ne01;

        uint32_t i13 = fastmodulo(i03, ne13, &bctx->src1_dim3_div);
        uint32_t i12 = fastmodulo(i02, ne12, &bctx->src1_dim2_div);
        ਅਚਲ uint8_t * src1_plane = vtcm_src1 + i13 * vtcm_vol_stride + i12 * vtcm_plane_stride;

        htp_trace_event_ਸ਼ੁਰੂ(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);
        compute(d_spad, s0_spad, src1_plane, i01, ne11, &bctx->src1_dim1_div, vtcm_ਪੰਕਤੀ_stride,
                current_ਖੰਡ_ਮਾਪ, bctx->dst_ਪੰਕਤੀ_ਮਾਪ_aligned, bctx->src0_ਪੰਕਤੀ_ਮਾਪ_aligned, ne00);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);

        dma_addr_t dst_curr = dst->ਅੰਕੜੇ + i03 * nb3 + i02 * nb2 + i01 * nb1;
        dma_ਕਤਾਰ_ਧੱਕੋ(dma_q, dma_make_ਅੰਕੜੇ(dst_curr, d_spad), nb1, bctx->dst_ਪੰਕਤੀ_ਮਾਪ_aligned, ਪੰਕਤੀ_ਮਾਪ_ਬਾਈਟਾਂ, current_ਖੰਡ_ਮਾਪ);

        ਜੇ (ir_prefetch < ਸਿਰਾ_ਪੰਕਤੀ) {
            uint32_t ਅਗਲਾ_ਖੰਡ_ਮਾਪ = calc_ਖੰਡ_ਮਾਪ(bctx, ir_prefetch, ਸਿਰਾ_ਪੰਕਤੀ, ne01, ne02);
            uint32_t p03, p02, p01, prem;
            p03  = fastdiv(ir_prefetch, &bctx->src0_dim12_div);
            prem = ir_prefetch - p03 * (ne02 * ne01);
            p02  = fastdiv(prem, &bctx->src0_dim1_div);
            p01  = prem - p02 * ne01;
            dma_addr_t s0_ਅਗਲਾ = src0->ਅੰਕੜੇ + p03 * nb03 + p02 * nb02 + p01 * nb01;
            dma_ਕਤਾਰ_ਧੱਕੋ(dma_q, dma_make_ਅੰਕੜੇ(s0_spad, s0_ਅਗਲਾ), bctx->src0_ਪੰਕਤੀ_ਮਾਪ_aligned, nb01, ਪੰਕਤੀ_ਮਾਪ_ਬਾਈਟਾਂ, ਅਗਲਾ_ਖੰਡ_ਮਾਪ);
            ir_prefetch += ਅਗਲਾ_ਖੰਡ_ਮਾਪ;
        }
        ir += current_ਖੰਡ_ਮਾਪ;
    }

    dma_ਕਤਾਰ_flush(dma_q);
}

// 5. Element Repeat (ne10 != ne00)
ਸਥਿਰ ਖਾਲੀ ਦੁਆਧਾਰੀ_thread_element_repeat(ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ nth, ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ ith, ਖਾਲੀ * ਅੰਕੜੇ) {
    ਬਣਤਰ htp_ਦੁਆਧਾਰੀ_context * bctx = (ਬਣਤਰ htp_ਦੁਆਧਾਰੀ_context *) ਅੰਕੜੇ;
    ਬਣਤਰ htp_ops_context * octx = bctx->octx;
    htp_ਦੁਆਧਾਰੀ_preamble;

    ਅਚਲ uint32_t ਪੰਕਤੀ_ਮਾਪ_ਬਾਈਟਾਂ = bctx->ਪੰਕਤੀ_ਮਾਪ_ਬਾਈਟਾਂ;
    ਅਚਲ uint32_t ਸ਼ੁਰੂ_ਪੰਕਤੀ  = bctx->ਪੰਕਤੀ_ਸ਼ੁਰੂ + bctx->nrows_ਪ੍ਰਤੀ_thread * ith;
    ਅਚਲ uint32_t ਸਿਰਾ_ਪੰਕਤੀ    = MIN(ਸ਼ੁਰੂ_ਪੰਕਤੀ + bctx->nrows_ਪ੍ਰਤੀ_thread, bctx->ਪੰਕਤੀ_ਸ਼ੁਰੂ + bctx->ਕੁੱਲ_ਪੰਕਤੀਆਂ);
    ਜੇ (ਸ਼ੁਰੂ_ਪੰਕਤੀ >= ਸਿਰਾ_ਪੰਕਤੀ) ਵਾਪਸ;

    ਅਚਲ ਬਣਤਰ htp_ਦੁਆਧਾਰੀ_vtcm_layout * layout = &bctx->vtcm_layout;
    uint8_t * src0_spad_ਅਧਾਰ = VTCM_LAYOUT_PTR(uint8_t, bctx->vtcm_ਅਧਾਰ, layout->off_src0) + (ith * layout->src0_ਬਾਈਟਾਂ_ਪ੍ਰਤੀ_thread);
    uint8_t * dst_spad_ਅਧਾਰ  = VTCM_LAYOUT_PTR(uint8_t, bctx->vtcm_ਅਧਾਰ, layout->off_dst)  + (ith * layout->dst_ਬਾਈਟਾਂ_ਪ੍ਰਤੀ_thread);
    size_t src0_spad_half    = layout->src0_spad_half_ਮਾਪ;
    size_t dst_spad_half     = layout->dst_spad_half_ਮਾਪ;
    ਅਚਲ uint8_t * vtcm_src1 = VTCM_LAYOUT_PTR(ਅਚਲ uint8_t, bctx->vtcm_ਅਧਾਰ, layout->off_src1);

    ਅਚਲ uint32_t elem_ਮਾਪ = (src0->type == HTP_TYPE_F32) ? ਆਕਾਰ(ਦਸ਼ਮਲਵ) : ਆਕਾਰ(_Float16);
    ਅਚਲ size_t vtcm_ਪੰਕਤੀ_stride   = (size_t) ne10 * elem_ਮਾਪ;
    ਅਚਲ size_t vtcm_plane_stride = (size_t) ne11 * vtcm_ਪੰਕਤੀ_stride;
    ਅਚਲ size_t vtcm_vol_stride   = (size_t) ne12 * vtcm_plane_stride;

    FARF(ਉੱਚਾ_2, "binary-repeat: %d/%d (%u:%u) row-size %u (%u)",
         ith, nth, ਸ਼ੁਰੂ_ਪੰਕਤੀ, ਸਿਰਾ_ਪੰਕਤੀ, nb01, bctx->dst_ਪੰਕਤੀ_ਮਾਪ_aligned);

    dma_ਕਤਾਰ * dma_q = octx->ctx->dma[ith];
    uint32_t ir_prefetch = ਸ਼ੁਰੂ_ਪੰਕਤੀ;
    ਪੂਰਨ spad_idx = 0;

    ਲਈ (ਪੂਰਨ k = 0; k < 2 && ir_prefetch < ਸਿਰਾ_ਪੰਕਤੀ; k++) {
        uint32_t current_ਖੰਡ_ਮਾਪ = calc_ਖੰਡ_ਮਾਪ(bctx, ir_prefetch, ਸਿਰਾ_ਪੰਕਤੀ, ne01, ne02);
        uint32_t i03 = fastdiv(ir_prefetch, &bctx->src0_dim12_div);
        uint32_t rem = ir_prefetch - i03 * (ne02 * ne01);
        uint32_t i02 = fastdiv(rem, &bctx->src0_dim1_div);
        uint32_t i01 = rem - i02 * ne01;

        dma_addr_t src0_curr = src0->ਅੰਕੜੇ + i03 * nb03 + i02 * nb02 + i01 * nb01;
        dma_addr_t dst_curr  = dst->ਅੰਕੜੇ  + i03 * nb3  + i02 * nb2  + i01 * nb1;

        uint8_t * s0_spad = src0_spad_ਅਧਾਰ + spad_idx * src0_spad_half;
        uint8_t * d_spad  = dst_spad_ਅਧਾਰ  + spad_idx * dst_spad_half;

        dma_ਕਤਾਰ_ਧੱਕੋ(dma_q, dma_make_ਅੰਕੜੇ(dst_curr, d_spad), nb1, bctx->dst_ਪੰਕਤੀ_ਮਾਪ_aligned, ਪੰਕਤੀ_ਮਾਪ_ਬਾਈਟਾਂ, 0);
        dma_ਕਤਾਰ_ਧੱਕੋ(dma_q, dma_make_ਅੰਕੜੇ(s0_spad, src0_curr), bctx->src0_ਪੰਕਤੀ_ਮਾਪ_aligned, nb01, ਪੰਕਤੀ_ਮਾਪ_ਬਾਈਟਾਂ, current_ਖੰਡ_ਮਾਪ);
        ir_prefetch += current_ਖੰਡ_ਮਾਪ;
        spad_idx ^= 1;
    }

    ਬਣਤਰ htp_thread_trace * tr = &octx->ctx->trace[ith];
    compute_repeat_t compute = (compute_repeat_t) bctx->compute;

    ਲਈ (uint32_t ir = ਸ਼ੁਰੂ_ਪੰਕਤੀ; ir < ਸਿਰਾ_ਪੰਕਤੀ; ) {
        uint32_t current_ਖੰਡ_ਮਾਪ = calc_ਖੰਡ_ਮਾਪ(bctx, ir, ਸਿਰਾ_ਪੰਕਤੀ, ne01, ne02);
        uint8_t * d_spad  = (uint8_t *) dma_ਕਤਾਰ_ਕੱਢੋ(dma_q).src;
        uint8_t * s0_spad = (uint8_t *) dma_ਕਤਾਰ_ਕੱਢੋ(dma_q).dst;

        uint32_t i03 = fastdiv(ir, &bctx->src0_dim12_div);
        uint32_t rem = ir - i03 * (ne02 * ne01);
        uint32_t i02 = fastdiv(rem, &bctx->src0_dim1_div);
        uint32_t i01 = rem - i02 * ne01;

        uint32_t i13 = fastmodulo(i03, ne13, &bctx->src1_dim3_div);
        uint32_t i12 = fastmodulo(i02, ne12, &bctx->src1_dim2_div);
        ਅਚਲ uint8_t * src1_plane = vtcm_src1 + i13 * vtcm_vol_stride + i12 * vtcm_plane_stride;

        htp_trace_event_ਸ਼ੁਰੂ(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);
        compute(d_spad, s0_spad, src1_plane, i01, ne11, &bctx->src1_dim1_div, vtcm_ਪੰਕਤੀ_stride,
                current_ਖੰਡ_ਮਾਪ, bctx->dst_ਪੰਕਤੀ_ਮਾਪ_aligned, bctx->src0_ਪੰਕਤੀ_ਮਾਪ_aligned, ne00, ne10);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);

        dma_addr_t dst_curr = dst->ਅੰਕੜੇ + i03 * nb3 + i02 * nb2 + i01 * nb1;
        dma_ਕਤਾਰ_ਧੱਕੋ(dma_q, dma_make_ਅੰਕੜੇ(dst_curr, d_spad), nb1, bctx->dst_ਪੰਕਤੀ_ਮਾਪ_aligned, ਪੰਕਤੀ_ਮਾਪ_ਬਾਈਟਾਂ, current_ਖੰਡ_ਮਾਪ);

        ਜੇ (ir_prefetch < ਸਿਰਾ_ਪੰਕਤੀ) {
            uint32_t ਅਗਲਾ_ਖੰਡ_ਮਾਪ = calc_ਖੰਡ_ਮਾਪ(bctx, ir_prefetch, ਸਿਰਾ_ਪੰਕਤੀ, ne01, ne02);
            uint32_t p03, p02, p01, prem;
            p03  = fastdiv(ir_prefetch, &bctx->src0_dim12_div);
            prem = ir_prefetch - p03 * (ne02 * ne01);
            p02  = fastdiv(prem, &bctx->src0_dim1_div);
            p01  = prem - p02 * ne01;
            dma_addr_t s0_ਅਗਲਾ = src0->ਅੰਕੜੇ + p03 * nb03 + p02 * nb02 + p01 * nb01;
            dma_ਕਤਾਰ_ਧੱਕੋ(dma_q, dma_make_ਅੰਕੜੇ(s0_spad, s0_ਅਗਲਾ), bctx->src0_ਪੰਕਤੀ_ਮਾਪ_aligned, nb01, ਪੰਕਤੀ_ਮਾਪ_ਬਾਈਟਾਂ, ਅਗਲਾ_ਖੰਡ_ਮਾਪ);
            ir_prefetch += ਅਗਲਾ_ਖੰਡ_ਮਾਪ;
        }
        ir += current_ਖੰਡ_ਮਾਪ;
    }

    dma_ਕਤਾਰ_flush(dma_q);
}

// 6. ADD_ID (src1 gathered via src2 indices)
ਸਥਿਰ ਖਾਲੀ ਦੁਆਧਾਰੀ_thread_ਜੋੜੋ_id_f32(ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ nth, ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ ith, ਖਾਲੀ * ਅੰਕੜੇ) {
    ਬਣਤਰ htp_ਦੁਆਧਾਰੀ_context * bctx = (ਬਣਤਰ htp_ਦੁਆਧਾਰੀ_context *) ਅੰਕੜੇ;
    ਬਣਤਰ htp_ops_context * octx = bctx->octx;

    ਅਚਲ ਬਣਤਰ htp_tensor * src0 = octx->src[0];
    ਅਚਲ ਬਣਤਰ htp_tensor * src1 = octx->src[1];
    ਅਚਲ ਬਣਤਰ htp_tensor * dst  = octx->dst;

    ਅਚਲ uint32_t ne00 = src0->ne[0];
    ਅਚਲ uint32_t ne01 = src0->ne[1];
    ਅਚਲ uint32_t ne02 = src0->ne[2];
    ਅਚਲ uint32_t ne03 = src0->ne[3];

    ਅਚਲ uint32_t nb01 = src0->nb[1];
    ਅਚਲ uint32_t nb02 = src0->nb[2];
    ਅਚਲ uint32_t nb03 = src0->nb[3];
    ਅਚਲ uint32_t src1_stride = bctx->src1_ਪੰਕਤੀ_ਮਾਪ_aligned;

    ਅਚਲ uint32_t nb1 = dst->nb[1];
    ਅਚਲ uint32_t nb2 = dst->nb[2];
    ਅਚਲ uint32_t nb3 = dst->nb[3];

    ਅਚਲ uint32_t ਪੰਕਤੀ_ਮਾਪ_ਬਾਈਟਾਂ = bctx->ਪੰਕਤੀ_ਮਾਪ_ਬਾਈਟਾਂ;
    ਅਚਲ uint32_t ਸ਼ੁਰੂ_ਪੰਕਤੀ = bctx->ਪੰਕਤੀ_ਸ਼ੁਰੂ + bctx->nrows_ਪ੍ਰਤੀ_thread * ith;
    ਅਚਲ uint32_t ਸਿਰਾ_ਪੰਕਤੀ   = MIN(ਸ਼ੁਰੂ_ਪੰਕਤੀ + bctx->nrows_ਪ੍ਰਤੀ_thread, bctx->ਪੰਕਤੀ_ਸ਼ੁਰੂ + bctx->ਕੁੱਲ_ਪੰਕਤੀਆਂ);
    ਜੇ (ਸ਼ੁਰੂ_ਪੰਕਤੀ >= ਸਿਰਾ_ਪੰਕਤੀ) ਵਾਪਸ;

    ਅਚਲ ਬਣਤਰ htp_ਦੁਆਧਾਰੀ_vtcm_layout * layout = &bctx->vtcm_layout;
    uint8_t * src0_spad_ਅਧਾਰ = VTCM_LAYOUT_PTR(uint8_t, bctx->vtcm_ਅਧਾਰ, layout->off_src0) + (ith * layout->src0_ਬਾਈਟਾਂ_ਪ੍ਰਤੀ_thread);
    uint8_t * dst_spad_ਅਧਾਰ  = VTCM_LAYOUT_PTR(uint8_t, bctx->vtcm_ਅਧਾਰ, layout->off_dst)  + (ith * layout->dst_ਬਾਈਟਾਂ_ਪ੍ਰਤੀ_thread);
    ਅਚਲ uint8_t * vtcm_src1 = VTCM_LAYOUT_PTR(ਅਚਲ uint8_t, bctx->vtcm_ਅਧਾਰ, layout->off_src1);
    ਅਚਲ int32_t * vtcm_src2 = VTCM_LAYOUT_PTR(ਅਚਲ int32_t, bctx->vtcm_ਅਧਾਰ, layout->off_src2);
    size_t src0_spad_half    = layout->src0_spad_half_ਮਾਪ;
    size_t dst_spad_half     = layout->dst_spad_half_ਮਾਪ;

    dma_ਕਤਾਰ * dma_q = octx->ctx->dma[ith];
    uint32_t ir_prefetch = ਸ਼ੁਰੂ_ਪੰਕਤੀ;
    ਪੂਰਨ spad_idx = 0;

    ਲਈ (ਪੂਰਨ k = 0; k < 2 && ir_prefetch < ਸਿਰਾ_ਪੰਕਤੀ; k++) {
        uint32_t current_ਖੰਡ_ਮਾਪ = calc_ਖੰਡ_ਮਾਪ(bctx, ir_prefetch, ਸਿਰਾ_ਪੰਕਤੀ, ne01, ne02);
        uint32_t i03 = fastdiv(ir_prefetch, &bctx->src0_dim12_div);
        uint32_t rem = ir_prefetch - i03 * (ne02 * ne01);
        uint32_t i02 = fastdiv(rem, &bctx->src0_dim1_div);
        uint32_t i01 = rem - i02 * ne01;

        dma_addr_t src0_curr = src0->ਅੰਕੜੇ + i03 * nb03 + i02 * nb02 + i01 * nb01;
        dma_addr_t dst_curr  = dst->ਅੰਕੜੇ  + i03 * nb3  + i02 * nb2  + i01 * nb1;

        uint8_t * s0_spad = src0_spad_ਅਧਾਰ + spad_idx * src0_spad_half;
        uint8_t * d_spad  = dst_spad_ਅਧਾਰ  + spad_idx * dst_spad_half;

        dma_ਕਤਾਰ_ਧੱਕੋ(dma_q, dma_make_ਅੰਕੜੇ(dst_curr, d_spad), nb1, bctx->dst_ਪੰਕਤੀ_ਮਾਪ_aligned, ਪੰਕਤੀ_ਮਾਪ_ਬਾਈਟਾਂ, 0);
        dma_ਕਤਾਰ_ਧੱਕੋ(dma_q, dma_make_ਅੰਕੜੇ(s0_spad, src0_curr), bctx->src0_ਪੰਕਤੀ_ਮਾਪ_aligned, nb01, ਪੰਕਤੀ_ਮਾਪ_ਬਾਈਟਾਂ, current_ਖੰਡ_ਮਾਪ);
        ir_prefetch += current_ਖੰਡ_ਮਾਪ;
        spad_idx ^= 1;
    }

    ਬਣਤਰ htp_thread_trace * tr = &octx->ctx->trace[ith];

    ਲਈ (uint32_t ir = ਸ਼ੁਰੂ_ਪੰਕਤੀ; ir < ਸਿਰਾ_ਪੰਕਤੀ; ) {
        uint32_t current_ਖੰਡ_ਮਾਪ = calc_ਖੰਡ_ਮਾਪ(bctx, ir, ਸਿਰਾ_ਪੰਕਤੀ, ne01, ne02);
        uint8_t * d_spad = (uint8_t *) dma_ਕਤਾਰ_ਕੱਢੋ(dma_q).src;
        uint8_t * s0_spad = (uint8_t *) dma_ਕਤਾਰ_ਕੱਢੋ(dma_q).dst;

        uint32_t i03 = fastdiv(ir, &bctx->src0_dim12_div);
        uint32_t rem = ir - i03 * (ne02 * ne01);
        uint32_t i02 = fastdiv(rem, &bctx->src0_dim1_div);
        uint32_t i01 = rem - i02 * ne01;

        htp_trace_event_ਸ਼ੁਰੂ(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);
        compute_ਜੋੜੋ_id_t compute = (compute_ਜੋੜੋ_id_t) bctx->compute;
        compute(d_spad, s0_spad, vtcm_src1, vtcm_src2,
                i01, i02, ne01, src1_stride,
                current_ਖੰਡ_ਮਾਪ, bctx->dst_ਪੰਕਤੀ_ਮਾਪ_aligned, bctx->src0_ਪੰਕਤੀ_ਮਾਪ_aligned, ne00);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);

        dma_addr_t dst_curr = dst->ਅੰਕੜੇ + i03 * nb3 + i02 * nb2 + i01 * nb1;
        dma_ਕਤਾਰ_ਧੱਕੋ(dma_q, dma_make_ਅੰਕੜੇ(dst_curr, d_spad), nb1, bctx->dst_ਪੰਕਤੀ_ਮਾਪ_aligned, ਪੰਕਤੀ_ਮਾਪ_ਬਾਈਟਾਂ, current_ਖੰਡ_ਮਾਪ);

        ਜੇ (ir_prefetch < ਸਿਰਾ_ਪੰਕਤੀ) {
             uint32_t ਅਗਲਾ_ਖੰਡ_ਮਾਪ = calc_ਖੰਡ_ਮਾਪ(bctx, ir_prefetch, ਸਿਰਾ_ਪੰਕਤੀ, ne01, ne02);
             uint32_t p03, p02, p01, prem;
             p03  = fastdiv(ir_prefetch, &bctx->src0_dim12_div);
             prem = ir_prefetch - p03 * (ne02 * ne01);
             p02  = fastdiv(prem, &bctx->src0_dim1_div);
             p01  = prem - p02 * ne01;
             dma_addr_t s0_ਅਗਲਾ = src0->ਅੰਕੜੇ + p03 * nb03 + p02 * nb02 + p01 * nb01;
             dma_ਕਤਾਰ_ਧੱਕੋ(dma_q, dma_make_ਅੰਕੜੇ(s0_spad, s0_ਅਗਲਾ), bctx->src0_ਪੰਕਤੀ_ਮਾਪ_aligned, nb01, ਪੰਕਤੀ_ਮਾਪ_ਬਾਈਟਾਂ, ਅਗਲਾ_ਖੰਡ_ਮਾਪ);
             ir_prefetch += ਅਗਲਾ_ਖੰਡ_ਮਾਪ;
        }
        ir += current_ਖੰਡ_ਮਾਪ;
    }

    dma_ਕਤਾਰ_flush(dma_q);
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ hvx_div_scalar_f32(uint8_t * ਸੀਮਿਤ dst, ਅਚਲ uint8_t * ਸੀਮਿਤ src, ਅਚਲ ਦਸ਼ਮਲਵ val, ਅਚਲ uint32_t ਅੰਕ_elems) {
    hvx_mul_scalar_f32(dst, src, 1.0f / val, ਅੰਕ_elems);
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ hvx_div_scalar_f16(uint8_t * ਸੀਮਿਤ dst, ਅਚਲ uint8_t * ਸੀਮਿਤ src, ਅਚਲ _Float16 val, ਅਚਲ uint32_t ਅੰਕ_elems) {
    hvx_div_scalar_f16_aa(dst, src, val, ਅੰਕ_elems);
}

ਕਿਸਮਪਰਿਭਾਸ਼ਾ ਖਾਲੀ (*compute_ਦੁਆਧਾਰੀ_chunked_t)(
    uint8_t * ਸੀਮਿਤ dst,
    ਅਚਲ uint8_t * ਸੀਮਿਤ src0,
    ਅਚਲ uint8_t * ਸੀਮਿਤ src1,
    ਅਚਲ uint32_t ਅੰਕ_elems
);

ਕਿਸਮਪਰਿਭਾਸ਼ਾ ਖਾਲੀ (*compute_ਦੁਆਧਾਰੀ_scalar_chunked_f32_t)(
    uint8_t * ਸੀਮਿਤ dst,
    ਅਚਲ uint8_t * ਸੀਮਿਤ src0,
    ਅਚਲ ਦਸ਼ਮਲਵ val,
    ਅਚਲ uint32_t ਅੰਕ_elems
);

ਕਿਸਮਪਰਿਭਾਸ਼ਾ ਖਾਲੀ (*compute_ਦੁਆਧਾਰੀ_scalar_chunked_f16_t)(
    uint8_t * ਸੀਮਿਤ dst,
    ਅਚਲ uint8_t * ਸੀਮਿਤ src0,
    ਅਚਲ _Float16 val,
    ਅਚਲ uint32_t ਅੰਕ_elems
);

ਬਣਤਰ ਦੁਆਧਾਰੀ_chunked_context {
    ਬਣਤਰ htp_ops_context * octx;
    ਬਣਤਰ htp_ਦੁਆਧਾਰੀ_vtcm_layout vtcm_layout;
    uint8_t *                vtcm_ਅਧਾਰ;
    uint32_t                 elem_ਸ਼ੁਰੂ;
    uint32_t                 nelem;
    uint32_t                 chunk_ਮਾਪ;
    uint32_t                 chunks_ਪ੍ਰਤੀ_thread;
    uint32_t                 ਕੁੱਲ_chunks;
    bool                     is_scalar;
    ਦਸ਼ਮਲਵ                    scalar_f32;
    _Float16                 scalar_f16;
    compute_ਦੁਆਧਾਰੀ_chunked_t compute;
    compute_ਦੁਆਧਾਰੀ_scalar_chunked_f32_t compute_scalar_f32;
    compute_ਦੁਆਧਾਰੀ_scalar_chunked_f16_t compute_scalar_f16;
};

ਸਥਿਰ ਖਾਲੀ ਦੁਆਧਾਰੀ_thread_chunked(ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ nth, ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ ith, ਖਾਲੀ * ਅੰਕੜੇ) {
    (ਖਾਲੀ) nth;
    ਬਣਤਰ ਦੁਆਧਾਰੀ_chunked_context * ctx = (ਬਣਤਰ ਦੁਆਧਾਰੀ_chunked_context *) ਅੰਕੜੇ;
    ਬਣਤਰ htp_ops_context * octx = ctx->octx;
    ਅਚਲ ਬਣਤਰ htp_tensor * src0 = octx->src[0];
    ਅਚਲ ਬਣਤਰ htp_tensor * src1 = octx->src[1];
    ਅਚਲ ਬਣਤਰ htp_tensor * dst  = octx->dst;

    ਅਚਲ uint32_t ਸ਼ੁਰੂ_chunk = ctx->chunks_ਪ੍ਰਤੀ_thread * ith;
    ਅਚਲ uint32_t ਸਿਰਾ_chunk   = MIN(ਸ਼ੁਰੂ_chunk + ctx->chunks_ਪ੍ਰਤੀ_thread, ctx->ਕੁੱਲ_chunks);
    ਜੇ (ਸ਼ੁਰੂ_chunk >= ਸਿਰਾ_chunk) {
        ਵਾਪਸ;
    }

    ਅਚਲ uint32_t src0_type   = src0->type;
    ਅਚਲ size_t   elem_ਮਾਪ   = (src0_type == HTP_TYPE_F32) ? ਆਕਾਰ(ਦਸ਼ਮਲਵ) : ਆਕਾਰ(_Float16);
    ਅਚਲ uint32_t chunk_ਮਾਪ  = ctx->chunk_ਮਾਪ;
    ਅਚਲ size_t   chunk_ਬਾਈਟਾਂ = ctx->vtcm_layout.src0_spad_half_ਮਾਪ;

    FARF(ਉੱਚਾ_2, "binary-chunked: %d/%d (%u:%u) chunks %u elems %u",
         ith, nth, ਸ਼ੁਰੂ_chunk, ਸਿਰਾ_chunk, ctx->ਕੁੱਲ_chunks, ctx->nelem);

    ਅਚਲ ਬਣਤਰ htp_ਦੁਆਧਾਰੀ_vtcm_layout * layout = &ctx->vtcm_layout;
    uint8_t * src0_spad_ਅਧਾਰ = VTCM_LAYOUT_PTR(uint8_t, ctx->vtcm_ਅਧਾਰ, layout->off_src0) + (ith * layout->src0_ਬਾਈਟਾਂ_ਪ੍ਰਤੀ_thread);
    uint8_t * src1_spad_ਅਧਾਰ = ctx->is_scalar ? ਨਿਰਾ : (VTCM_LAYOUT_PTR(uint8_t, ctx->vtcm_ਅਧਾਰ, layout->off_src1) + (ith * layout->src1_ਬਾਈਟਾਂ_ਪ੍ਰਤੀ_thread));
    uint8_t * dst_spad_ਅਧਾਰ  = VTCM_LAYOUT_PTR(uint8_t, ctx->vtcm_ਅਧਾਰ, layout->off_dst)  + (ith * layout->dst_ਬਾਈਟਾਂ_ਪ੍ਰਤੀ_thread);

    dma_ਕਤਾਰ * dma_q = octx->ctx->dma[ith];
    uint32_t prefetch_chunk = ਸ਼ੁਰੂ_chunk;
    ਪੂਰਨ spad_idx = 0;

    ਲਈ (ਪੂਰਨ k = 0; k < 2 && prefetch_chunk < ਸਿਰਾ_chunk; k++) {
        ਅਚਲ uint32_t c_ਸ਼ੁਰੂ   = ctx->elem_ਸ਼ੁਰੂ + prefetch_chunk * chunk_ਮਾਪ;
        ਅਚਲ uint32_t c_ਸਿਰਾ     = MIN(c_ਸ਼ੁਰੂ + chunk_ਮਾਪ, ctx->elem_ਸ਼ੁਰੂ + ctx->nelem);
        ਅਚਲ uint32_t cur_elems = c_ਸਿਰਾ - c_ਸ਼ੁਰੂ;
        ਅਚਲ uint32_t cur_ਬਾਈਟਾਂ = cur_elems * elem_ਮਾਪ;

        dma_addr_t s0_curr = src0->ਅੰਕੜੇ + (size_t) c_ਸ਼ੁਰੂ * elem_ਮਾਪ;
        dma_addr_t d_curr  = dst->ਅੰਕੜੇ  + (size_t) c_ਸ਼ੁਰੂ * elem_ਮਾਪ;

        uint8_t * s0_spad = src0_spad_ਅਧਾਰ + spad_idx * chunk_ਬਾਈਟਾਂ;
        uint8_t * d_spad  = dst_spad_ਅਧਾਰ  + spad_idx * chunk_ਬਾਈਟਾਂ;

        dma_ਕਤਾਰ_ਧੱਕੋ(dma_q, dma_make_ਅੰਕੜੇ(d_curr, d_spad), chunk_ਬਾਈਟਾਂ, chunk_ਬਾਈਟਾਂ, cur_ਬਾਈਟਾਂ, 0);
        dma_ਕਤਾਰ_ਧੱਕੋ(dma_q, dma_make_ਅੰਕੜੇ(s0_spad, s0_curr), chunk_ਬਾਈਟਾਂ, chunk_ਬਾਈਟਾਂ, cur_ਬਾਈਟਾਂ, 1);
        ਜੇ (!ctx->is_scalar) {
            dma_addr_t s1_curr = src1->ਅੰਕੜੇ + (size_t) c_ਸ਼ੁਰੂ * elem_ਮਾਪ;
            uint8_t * s1_spad = src1_spad_ਅਧਾਰ + spad_idx * chunk_ਬਾਈਟਾਂ;
            dma_ਕਤਾਰ_ਧੱਕੋ(dma_q, dma_make_ਅੰਕੜੇ(s1_spad, s1_curr), chunk_ਬਾਈਟਾਂ, chunk_ਬਾਈਟਾਂ, cur_ਬਾਈਟਾਂ, 1);
        }

        prefetch_chunk++;
        spad_idx ^= 1;
    }

    ਬਣਤਰ htp_thread_trace * tr = &octx->ctx->trace[ith];

    ਲਈ (uint32_t c = ਸ਼ੁਰੂ_chunk; c < ਸਿਰਾ_chunk; c++) {
        ਅਚਲ uint32_t c_ਸ਼ੁਰੂ   = ctx->elem_ਸ਼ੁਰੂ + c * chunk_ਮਾਪ;
        ਅਚਲ uint32_t c_ਸਿਰਾ     = MIN(c_ਸ਼ੁਰੂ + chunk_ਮਾਪ, ctx->elem_ਸ਼ੁਰੂ + ctx->nelem);
        ਅਚਲ uint32_t cur_elems = c_ਸਿਰਾ - c_ਸ਼ੁਰੂ;
        ਅਚਲ uint32_t cur_ਬਾਈਟਾਂ = cur_elems * elem_ਮਾਪ;

        uint8_t * d_spad  = (uint8_t *) dma_ਕਤਾਰ_ਕੱਢੋ(dma_q).src;
        uint8_t * s0_spad = (uint8_t *) dma_ਕਤਾਰ_ਕੱਢੋ(dma_q).dst;
        uint8_t * s1_spad = ctx->is_scalar ? ਨਿਰਾ : (uint8_t *) dma_ਕਤਾਰ_ਕੱਢੋ(dma_q).dst;

        htp_trace_event_ਸ਼ੁਰੂ(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) c);
        ਜੇ (ctx->is_scalar) {
            ਜੇ (src0_type == HTP_TYPE_F32) {
                ctx->compute_scalar_f32(d_spad, s0_spad, ctx->scalar_f32, cur_elems);
            } ਨਹੀਂਤਾਂ {
                ctx->compute_scalar_f16(d_spad, s0_spad, ctx->scalar_f16, cur_elems);
            }
        } ਨਹੀਂਤਾਂ {
            ctx->compute(d_spad, s0_spad, s1_spad, cur_elems);
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) c);

        dma_addr_t dst_curr = dst->ਅੰਕੜੇ + (size_t) c_ਸ਼ੁਰੂ * elem_ਮਾਪ;
        dma_ਕਤਾਰ_ਧੱਕੋ(dma_q, dma_make_ਅੰਕੜੇ(dst_curr, d_spad), chunk_ਬਾਈਟਾਂ, chunk_ਬਾਈਟਾਂ, cur_ਬਾਈਟਾਂ, 1);

        ਜੇ (prefetch_chunk < ਸਿਰਾ_chunk) {
            ਅਚਲ uint32_t pc_ਸ਼ੁਰੂ  = ctx->elem_ਸ਼ੁਰੂ + prefetch_chunk * chunk_ਮਾਪ;
            ਅਚਲ uint32_t pc_ਸਿਰਾ    = MIN(pc_ਸ਼ੁਰੂ + chunk_ਮਾਪ, ctx->elem_ਸ਼ੁਰੂ + ctx->nelem);
            ਅਚਲ uint32_t p_elems   = pc_ਸਿਰਾ - pc_ਸ਼ੁਰੂ;
            ਅਚਲ uint32_t p_ਬਾਈਟਾਂ   = p_elems * elem_ਮਾਪ;

            dma_addr_t s0_ਅਗਲਾ = src0->ਅੰਕੜੇ + (size_t) pc_ਸ਼ੁਰੂ * elem_ਮਾਪ;
            dma_ਕਤਾਰ_ਧੱਕੋ(dma_q, dma_make_ਅੰਕੜੇ(s0_spad, s0_ਅਗਲਾ), chunk_ਬਾਈਟਾਂ, chunk_ਬਾਈਟਾਂ, p_ਬਾਈਟਾਂ, 1);
            ਜੇ (!ctx->is_scalar) {
                dma_addr_t s1_ਅਗਲਾ = src1->ਅੰਕੜੇ + (size_t) pc_ਸ਼ੁਰੂ * elem_ਮਾਪ;
                dma_ਕਤਾਰ_ਧੱਕੋ(dma_q, dma_make_ਅੰਕੜੇ(s1_spad, s1_ਅਗਲਾ), chunk_ਬਾਈਟਾਂ, chunk_ਬਾਈਟਾਂ, p_ਬਾਈਟਾਂ, 1);
            }

            prefetch_chunk++;
        }
    }

    dma_ਕਤਾਰ_flush(dma_q);
}

ਸਥਿਰ ਪੂਰਨ execute_op_ਦੁਆਧਾਰੀ(ਬਣਤਰ htp_ops_context * octx) {
    ਅਚਲ ਬਣਤਰ htp_tensor * src0 = octx->src[0];
    ਅਚਲ ਬਣਤਰ htp_tensor * src1 = octx->src[1];
    ਅਚਲ ਬਣਤਰ htp_tensor * dst  = octx->dst;
    ਅਚਲ ਬਣਤਰ htp_ਦੁਆਧਾਰੀ_kernel_params * kparams = (ਅਚਲ ਬਣਤਰ htp_ਦੁਆਧਾਰੀ_kernel_params *) octx->kernel_params;

    ਅਚਲ uint32_t src0_nrows = src0->ne[1] * src0->ne[2] * src0->ne[3];

    // Use packed row sizes for VTCM allocation and alignment
    ਅਚਲ uint32_t src0_type = octx->src[0]->type;
    ਅਚਲ size_t elem_ਮਾਪ = (src0_type == HTP_TYPE_F32) ? ਆਕਾਰ(ਦਸ਼ਮਲਵ) : ਆਕਾਰ(_Float16);
    ਅਚਲ size_t src0_ਪੰਕਤੀ_ਮਾਪ = src0->ne[0] * elem_ਮਾਪ;
    ਅਚਲ size_t src1_ਪੰਕਤੀ_ਮਾਪ = src1->ne[0] * elem_ਮਾਪ;
    ਅਚਲ size_t dst_ਪੰਕਤੀ_ਮਾਪ  = dst->ne[0]  * elem_ਮਾਪ;

    ਜੇ (kparams->kernel_type == HTP_ਦੁਆਧਾਰੀ_KERNEL_CHUNKED) {
        ਅਚਲ bool is_scalar = kparams->is_scalar || (src1->ne[0] == 1 && src1->ne[1] == 1 && src1->ne[2] == 1 && src1->ne[3] == 1);

        compute_ਦੁਆਧਾਰੀ_chunked_t compute = ਨਿਰਾ;
        compute_ਦੁਆਧਾਰੀ_scalar_chunked_f32_t compute_scalar_f32 = ਨਿਰਾ;
        compute_ਦੁਆਧਾਰੀ_scalar_chunked_f16_t compute_scalar_f16 = ਨਿਰਾ;

        ਜੇ (is_scalar) {
            ਜੇ (src0_type == HTP_TYPE_F32) {
                ਚੋਣ (octx->op) {
                    ਹਾਲਤ HTP_OP_ਜੋੜੋ: compute_scalar_f32 = hvx_ਜੋੜੋ_scalar_f32; ਰੋਕੋ;
                    ਹਾਲਤ HTP_OP_SUB: compute_scalar_f32 = hvx_sub_scalar_f32; ਰੋਕੋ;
                    ਹਾਲਤ HTP_OP_MUL: compute_scalar_f32 = hvx_mul_scalar_f32; ਰੋਕੋ;
                    ਹਾਲਤ HTP_OP_DIV: compute_scalar_f32 = hvx_div_scalar_f32; ਰੋਕੋ;
                    ਮੂਲ: ਰੋਕੋ;
                }
            } ਨਹੀਂਤਾਂ ਜੇ (src0_type == HTP_TYPE_F16) {
                ਚੋਣ (octx->op) {
                    ਹਾਲਤ HTP_OP_ਜੋੜੋ: compute_scalar_f16 = hvx_ਜੋੜੋ_scalar_f16; ਰੋਕੋ;
                    ਹਾਲਤ HTP_OP_SUB: compute_scalar_f16 = hvx_sub_scalar_f16; ਰੋਕੋ;
                    ਹਾਲਤ HTP_OP_MUL: compute_scalar_f16 = hvx_mul_scalar_f16; ਰੋਕੋ;
                    ਹਾਲਤ HTP_OP_DIV: compute_scalar_f16 = hvx_div_scalar_f16; ਰੋਕੋ;
                    ਮੂਲ: ਰੋਕੋ;
                }
            }
            ਜੇ (!compute_scalar_f32 && !compute_scalar_f16) {
                ਵਾਪਸ HTP_ਸਥਿਤੀ_NO_SUPPORT;
            }
        } ਨਹੀਂਤਾਂ {
            ਜੇ (src0_type == HTP_TYPE_F32) {
                ਚੋਣ (octx->op) {
                    ਹਾਲਤ HTP_OP_ਜੋੜੋ: compute = hvx_ਜੋੜੋ_f32; ਰੋਕੋ;
                    ਹਾਲਤ HTP_OP_SUB: compute = hvx_sub_f32; ਰੋਕੋ;
                    ਹਾਲਤ HTP_OP_MUL: compute = hvx_mul_f32; ਰੋਕੋ;
                    ਹਾਲਤ HTP_OP_DIV: compute = hvx_div_f32; ਰੋਕੋ;
                    ਮੂਲ: ਰੋਕੋ;
                }
            } ਨਹੀਂਤਾਂ ਜੇ (src0_type == HTP_TYPE_F16) {
                ਚੋਣ (octx->op) {
                    ਹਾਲਤ HTP_OP_ਜੋੜੋ: compute = hvx_ਜੋੜੋ_f16; ਰੋਕੋ;
                    ਹਾਲਤ HTP_OP_SUB: compute = hvx_sub_f16; ਰੋਕੋ;
                    ਹਾਲਤ HTP_OP_MUL: compute = hvx_mul_f16; ਰੋਕੋ;
                    ਹਾਲਤ HTP_OP_DIV: compute = hvx_div_f16; ਰੋਕੋ;
                    ਮੂਲ: ਰੋਕੋ;
                }
            }
            ਜੇ (!compute) {
                ਵਾਪਸ HTP_ਸਥਿਤੀ_NO_SUPPORT;
            }
        }

        ਅਚਲ uint32_t ਕੁੱਲ_elems = (uint32_t) (src0->ne[0] * src0->ne[1] * src0->ne[2] * src0->ne[3]);
        ਜੇ (ਕੁੱਲ_elems == 0) {
            ਵਾਪਸ HTP_ਸਥਿਤੀ_OK;
        }

        uint32_t elem_ਸ਼ੁਰੂ = 0;
        uint32_t nelem      = ਕੁੱਲ_elems;
        ਅਚਲ uint32_t elems_ਪ੍ਰਤੀ_ਸਤਰ = (src0_type == HTP_TYPE_F32) ? 32 : 64;

        ਜੇ (octx->ctx->mdev.ਗਿਣਤੀ_2 > 1) {
            ਅਚਲ bool can_ਵੰਡੋ = htp_tensor_mdev_ਅੰਕੜੇ_aligned(dst) &&
                                   htp_tensor_mdev_ਅੰਕੜੇ_aligned(src0) &&
                                   (is_scalar || htp_tensor_mdev_ਅੰਕੜੇ_aligned(src1)) &&
                                   htp_tensor_is_contiguous(dst, elem_ਮਾਪ) &&
                                   htp_tensor_is_contiguous(src0, elem_ਮਾਪ) &&
                                   (is_scalar || htp_tensor_is_contiguous(src1, elem_ਮਾਪ));
            ਅਚਲ ਬਣਤਰ htp_tensor_mdev_range range = htp_tensor_mdev_ਹਿੱਸੇ_ਕਰੋ(
                ਕੁੱਲ_elems, can_ਵੰਡੋ ? elems_ਪ੍ਰਤੀ_ਸਤਰ : 0,
                octx->ctx->mdev.idx, octx->ctx->mdev.ਗਿਣਤੀ_2, &octx->ctx->mdev.ਗਿਣਤੀ_div);
            elem_ਸ਼ੁਰੂ = range.ਸ਼ੁਰੂ;
            nelem      = range.ਗਿਣਤੀ_2;
        }

        ਜੇ (nelem == 0) {
            ਵਾਪਸ HTP_ਸਥਿਤੀ_OK;
        }

        ਜੇ (!htp_ops_context_ਸੈੱਟ_n_ਧਾਗੇ(octx, kparams->n_ਧਾਗੇ)) {
            ਵਾਪਸ HTP_ਸਥਿਤੀ_INVAL_PARAMS;
        }

        ਬਣਤਰ htp_ਦੁਆਧਾਰੀ_vtcm_layout vtcm_layout;
        htp_ਦੁਆਧਾਰੀ_vtcm_layout_build(&vtcm_layout, kparams, octx->ctx->vtcm_ਮਾਪ);
        ਜੇ (vtcm_layout.ਕੁੱਲ_ਬਾਈਟਾਂ == 0 || vtcm_layout.ਕੁੱਲ_ਬਾਈਟਾਂ > octx->ctx->vtcm_ਮਾਪ) {
            ਵਾਪਸ HTP_ਸਥਿਤੀ_VTCM_TOO_SMALL;
        }

        ਅਚਲ uint32_t chunk_ਮਾਪ = kparams->chunk_ਮਾਪ > 0 ? kparams->chunk_ਮਾਪ : (32768 / elem_ਮਾਪ);
        ਅਚਲ uint32_t ਕੁੱਲ_chunks = (nelem + chunk_ਮਾਪ - 1) / chunk_ਮਾਪ;
        ਅਚਲ uint32_t n_ਧਾਗੇ = (ਕੁੱਲ_chunks >= 2) ? MIN(octx->n_ਧਾਗੇ, ਕੁੱਲ_chunks) : 1;
        ਅਚਲ uint32_t chunks_ਪ੍ਰਤੀ_thread = (ਕੁੱਲ_chunks + n_ਧਾਗੇ - 1) / n_ਧਾਗੇ;

        ਦਸ਼ਮਲਵ scalar_f32 = 0.0f;
        _Float16 scalar_f16 = 0;
        ਜੇ (is_scalar) {
            uint8_t * vtcm_src1 = VTCM_LAYOUT_PTR(uint8_t, octx->ctx->vtcm_ਅਧਾਰ, vtcm_layout.off_src1);
            dma_ਕਤਾਰ * dma_q = octx->ctx->dma[0];
            dma_ਕਤਾਰ_ਧੱਕੋ(dma_q, dma_make_ਅੰਕੜੇ(vtcm_src1, src1->ਅੰਕੜੇ), 128, 0, elem_ਮਾਪ, 1);
            dma_ਕਤਾਰ_ਕੱਢੋ(dma_q);

            ਜੇ (src0_type == HTP_TYPE_F32) {
                scalar_f32 = ((ਅਚਲ ਦਸ਼ਮਲਵ *) vtcm_src1)[0];
            } ਨਹੀਂਤਾਂ {
                scalar_f16 = ((ਅਚਲ _Float16 *) vtcm_src1)[0];
            }
        }

        ਬਣਤਰ ਦੁਆਧਾਰੀ_chunked_context cctx = {
            .octx                = octx,
            .vtcm_layout         = vtcm_layout,
            .vtcm_ਅਧਾਰ           = (uint8_t *) octx->ctx->vtcm_ਅਧਾਰ,
            .elem_ਸ਼ੁਰੂ          = elem_ਸ਼ੁਰੂ,
            .nelem               = nelem,
            .chunk_ਮਾਪ          = chunk_ਮਾਪ,
            .chunks_ਪ੍ਰਤੀ_thread   = chunks_ਪ੍ਰਤੀ_thread,
            .ਕੁੱਲ_chunks        = ਕੁੱਲ_chunks,
            .is_scalar           = is_scalar,
            .scalar_f32          = scalar_f32,
            .scalar_f16          = scalar_f16,
            .compute             = compute,
            .compute_scalar_f32  = compute_scalar_f32,
            .compute_scalar_f16  = compute_scalar_f16,
        };

        work_ਕਤਾਰ_ਲੜੀਵਾਰ(octx->ctx->work_ਕਤਾਰ, ਦੁਆਧਾਰੀ_thread_chunked, &cctx, n_ਧਾਗੇ);
        ਵਾਪਸ HTP_ਸਥਿਤੀ_OK;
    }

    uint32_t ਪੰਕਤੀ_ਸ਼ੁਰੂ = 0;
    uint32_t nrows     = src0_nrows;

    ਜੇ (octx->ctx->mdev.ਗਿਣਤੀ_2 > 1) {
        uint32_t ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_chunk = 0;
        htp_tensor_mdev_ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_chunk(dst, (uint32_t) elem_ਮਾਪ, (uint32_t) dst_ਪੰਕਤੀ_ਮਾਪ, &ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_chunk);
        ਅਚਲ ਬਣਤਰ htp_tensor_mdev_range range = htp_tensor_mdev_ਹਿੱਸੇ_ਕਰੋ(src0_nrows, ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_chunk,
                                                       octx->ctx->mdev.idx, octx->ctx->mdev.ਗਿਣਤੀ_2, &octx->ctx->mdev.ਗਿਣਤੀ_div);
        ਪੰਕਤੀ_ਸ਼ੁਰੂ = range.ਸ਼ੁਰੂ;
        nrows     = range.ਗਿਣਤੀ_2;
    }

    ਜੇ (nrows == 0) {
        ਵਾਪਸ HTP_ਸਥਿਤੀ_OK;
    }

    ਜੇ (!htp_ops_context_ਸੈੱਟ_n_ਧਾਗੇ(octx, kparams->n_ਧਾਗੇ)) {
        ਵਾਪਸ HTP_ਸਥਿਤੀ_INVAL_PARAMS;
    }

    ਅਚਲ uint32_t n_ਧਾਗੇ = octx->n_ਧਾਗੇ;
    ਅਚਲ size_t src0_ਪੰਕਤੀ_ਮਾਪ_aligned = kparams->src0_ਪੰਕਤੀ_ਮਾਪ_aligned;
    ਅਚਲ size_t src1_ਪੰਕਤੀ_ਮਾਪ_aligned = kparams->src1_ਪੰਕਤੀ_ਮਾਪ_aligned;
    ਅਚਲ size_t dst_ਪੰਕਤੀ_ਮਾਪ_aligned  = kparams->dst_ਪੰਕਤੀ_ਮਾਪ_aligned;

    ਬਣਤਰ htp_ਦੁਆਧਾਰੀ_context bctx;
    bctx.vtcm_ਅਧਾਰ = (uint8_t *) octx->ctx->vtcm_ਅਧਾਰ;
    htp_ਦੁਆਧਾਰੀ_vtcm_layout_build(&bctx.vtcm_layout, kparams, octx->ctx->vtcm_ਮਾਪ);

    ਜੇ (bctx.vtcm_layout.ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_ਬਫਰ == 0 || bctx.vtcm_layout.ਕੁੱਲ_ਬਾਈਟਾਂ > octx->ctx->vtcm_ਮਾਪ) {
        ਵਾਪਸ HTP_ਸਥਿਤੀ_VTCM_TOO_SMALL;
    }

    dma_ਕਤਾਰ * dma_q = octx->ctx->dma[0];
    uint8_t * vtcm_src1 = VTCM_LAYOUT_PTR(uint8_t, bctx.vtcm_ਅਧਾਰ, bctx.vtcm_layout.off_src1);
    ਜੇ (kparams->kernel_type == HTP_ਦੁਆਧਾਰੀ_KERNEL_ਪੰਕਤੀ_BCAST) {
        dma_ਕਤਾਰ_ਧੱਕੋ(dma_q, dma_make_ਅੰਕੜੇ(vtcm_src1, src1->ਅੰਕੜੇ), bctx.vtcm_layout.src1_ਮਾਪ, 0, src1->ne[0] * elem_ਮਾਪ, 1);
    } ਨਹੀਂਤਾਂ ਜੇ (kparams->kernel_type == HTP_ਦੁਆਧਾਰੀ_KERNEL_SCALAR ||
               kparams->kernel_type == HTP_ਦੁਆਧਾਰੀ_KERNEL_REPEAT ||
               kparams->kernel_type == HTP_ਦੁਆਧਾਰੀ_KERNEL_COMPLEX) {
        ਜੇ (!kparams->src1_strided) {
            ਅਚਲ size_t src1_ਬਾਈਟਾਂ = (size_t) src1->ne[0] * src1->ne[1] * src1->ne[2] * src1->ne[3] * elem_ਮਾਪ;
            dma_ਕਤਾਰ_ਧੱਕੋ(dma_q, dma_make_ਅੰਕੜੇ(vtcm_src1, src1->ਅੰਕੜੇ), bctx.vtcm_layout.src1_ਮਾਪ, 0, src1_ਬਾਈਟਾਂ, 1);
        } ਨਹੀਂਤਾਂ {
            ਅਚਲ uint32_t ne11 = src1->ne[1];
            ਅਚਲ uint32_t ne12 = src1->ne[2];
            ਅਚਲ uint32_t ne13 = src1->ne[3];
            ਅਚਲ size_t ਪੰਕਤੀ_ਬਾਈਟਾਂ = src1->ne[0] * elem_ਮਾਪ;
            ਲਈ (uint32_t i13 = 0; i13 < ne13; i13++) {
                ਲਈ (uint32_t i12 = 0; i12 < ne12; i12++) {
                    dma_addr_t src_addr = src1->ਅੰਕੜੇ + i13 * src1->nb[3] + i12 * src1->nb[2];
                    uint8_t * dst_ptr = vtcm_src1 + (i13 * ne12 * ne11 + i12 * ne11) * ਪੰਕਤੀ_ਬਾਈਟਾਂ;
                    ਜਦੋਂ (!dma_ਕਤਾਰ_ਧੱਕੋ(dma_q, dma_make_ਅੰਕੜੇ(dst_ptr, src_addr),
                                           ਪੰਕਤੀ_ਬਾਈਟਾਂ, src1->nb[1], ਪੰਕਤੀ_ਬਾਈਟਾਂ, ne11)) {
                        dma_ਕਤਾਰ_ਕੱਢੋ(dma_q);
                    }
                }
            }
        }
    } ਨਹੀਂਤਾਂ ਜੇ (kparams->kernel_type == HTP_ਦੁਆਧਾਰੀ_KERNEL_ਜੋੜੋ_ID) {
        ਅਚਲ ਬਣਤਰ htp_tensor * src2 = octx->src[2];
        uint8_t * vtcm_src2 = VTCM_LAYOUT_PTR(uint8_t, bctx.vtcm_ਅਧਾਰ, bctx.vtcm_layout.off_src2);
        dma_ਕਤਾਰ_ਧੱਕੋ(dma_q, dma_make_ਅੰਕੜੇ(vtcm_src1, src1->ਅੰਕੜੇ),
                       kparams->src1_ਪੰਕਤੀ_ਮਾਪ_aligned, src1->nb[1],
                       src1->ne[0] * elem_ਮਾਪ, src1->ne[1]);
        ਅਚਲ size_t src2_ਪੰਕਤੀ_ਬਾਈਟਾਂ = src2->ne[0] * ਆਕਾਰ(int32_t);
        dma_ਕਤਾਰ_ਧੱਕੋ(dma_q, dma_make_ਅੰਕੜੇ(vtcm_src2, src2->ਅੰਕੜੇ),
                       src2_ਪੰਕਤੀ_ਬਾਈਟਾਂ, src2->nb[1],
                       src2_ਪੰਕਤੀ_ਬਾਈਟਾਂ, src2->ne[1]);
    }

    bctx.octx                  = octx;
    bctx.nrows_ਪ੍ਰਤੀ_thread      = fastdiv(nrows + n_ਧਾਗੇ - 1, &octx->n_ਧਾਗੇ_div);
    bctx.ਕੁੱਲ_ਪੰਕਤੀਆਂ            = nrows;
    bctx.ਪੰਕਤੀ_ਸ਼ੁਰੂ             = ਪੰਕਤੀ_ਸ਼ੁਰੂ;
    bctx.ਖੰਡ_max             = bctx.vtcm_layout.ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_ਬਫਰ;
    bctx.src0_ਪੰਕਤੀ_ਮਾਪ_aligned = src0_ਪੰਕਤੀ_ਮਾਪ_aligned;
    bctx.src1_ਪੰਕਤੀ_ਮਾਪ_aligned = src1_ਪੰਕਤੀ_ਮਾਪ_aligned;
    bctx.dst_ਪੰਕਤੀ_ਮਾਪ_aligned  = dst_ਪੰਕਤੀ_ਮਾਪ_aligned;
    bctx.ਪੰਕਤੀ_ਮਾਪ_ਬਾਈਟਾਂ        = src0_ਪੰਕਤੀ_ਮਾਪ;

    bctx.src0_dim1_div  = init_fastdiv_ਮੁੱਲਾਂ(src0->ne[1]);
    bctx.src0_dim2_div  = init_fastdiv_ਮੁੱਲਾਂ(src0->ne[2]);
    bctx.src0_dim12_div = init_fastdiv_ਮੁੱਲਾਂ(src0->ne[1] * src0->ne[2]);

    bctx.src1_dim1_div  = init_fastdiv_ਮੁੱਲਾਂ(src1->ne[1]);
    bctx.src1_dim2_div  = init_fastdiv_ਮੁੱਲਾਂ(src1->ne[2]);
    bctx.src1_dim3_div  = init_fastdiv_ਮੁੱਲਾਂ(src1->ne[3]);

    bctx.ਵੰਡੋ_at_ne01  = kparams->ਵੰਡੋ_at_ne01;
    bctx.ਵੰਡੋ_at_ne02  = kparams->ਵੰਡੋ_at_ne02;

    ਕਾਮਾ_callback_t ਕਾਮਾ_func = ਨਿਰਾ;
    ਖਾਲੀ * compute_func = ਨਿਰਾ;

    ਚੋਣ (kparams->kernel_type) {
        ਹਾਲਤ HTP_ਦੁਆਧਾਰੀ_KERNEL_SAME_SHAPE:
            ਕਾਮਾ_func = ਦੁਆਧਾਰੀ_thread_ਸਦਿਸ਼_same_shape;
            ਜੇ (src0_type == HTP_TYPE_F32) {
                ਚੋਣ (octx->op) {
                    ਹਾਲਤ HTP_OP_ਜੋੜੋ: compute_func = compute_same_shape_ਜੋੜੋ_f32; ਰੋਕੋ;
                    ਹਾਲਤ HTP_OP_SUB: compute_func = compute_same_shape_sub_f32; ਰੋਕੋ;
                    ਹਾਲਤ HTP_OP_MUL: compute_func = compute_same_shape_mul_f32; ਰੋਕੋ;
                    ਹਾਲਤ HTP_OP_DIV: compute_func = compute_same_shape_div_f32; ਰੋਕੋ;
                    ਮੂਲ: ਰੋਕੋ;
                }
            } ਨਹੀਂਤਾਂ ਜੇ (src0_type == HTP_TYPE_F16) {
                ਚੋਣ (octx->op) {
                    ਹਾਲਤ HTP_OP_ਜੋੜੋ: compute_func = compute_same_shape_ਜੋੜੋ_f16; ਰੋਕੋ;
                    ਹਾਲਤ HTP_OP_SUB: compute_func = compute_same_shape_sub_f16; ਰੋਕੋ;
                    ਹਾਲਤ HTP_OP_MUL: compute_func = compute_same_shape_mul_f16; ਰੋਕੋ;
                    ਹਾਲਤ HTP_OP_DIV: compute_func = compute_same_shape_div_f16; ਰੋਕੋ;
                    ਮੂਲ: ਰੋਕੋ;
                }
            }
            ਰੋਕੋ;
        ਹਾਲਤ HTP_ਦੁਆਧਾਰੀ_KERNEL_ਪੰਕਤੀ_BCAST:
            ਕਾਮਾ_func = ਦੁਆਧਾਰੀ_thread_ਸਦਿਸ਼_ਪੰਕਤੀ_ਪ੍ਰਸਾਰਣ;
            ਜੇ (src0_type == HTP_TYPE_F32) {
                ਚੋਣ (octx->op) {
                    ਹਾਲਤ HTP_OP_ਜੋੜੋ: compute_func = compute_ਪੰਕਤੀ_bcast_ਜੋੜੋ_f32; ਰੋਕੋ;
                    ਹਾਲਤ HTP_OP_SUB: compute_func = compute_ਪੰਕਤੀ_bcast_sub_f32; ਰੋਕੋ;
                    ਹਾਲਤ HTP_OP_MUL: compute_func = compute_ਪੰਕਤੀ_bcast_mul_f32; ਰੋਕੋ;
                    ਹਾਲਤ HTP_OP_DIV: compute_func = compute_ਪੰਕਤੀ_bcast_div_f32; ਰੋਕੋ;
                    ਮੂਲ: ਰੋਕੋ;
                }
            } ਨਹੀਂਤਾਂ ਜੇ (src0_type == HTP_TYPE_F16) {
                ਚੋਣ (octx->op) {
                    ਹਾਲਤ HTP_OP_ਜੋੜੋ: compute_func = compute_ਪੰਕਤੀ_bcast_ਜੋੜੋ_f16; ਰੋਕੋ;
                    ਹਾਲਤ HTP_OP_SUB: compute_func = compute_ਪੰਕਤੀ_bcast_sub_f16; ਰੋਕੋ;
                    ਹਾਲਤ HTP_OP_MUL: compute_func = compute_ਪੰਕਤੀ_bcast_mul_f16; ਰੋਕੋ;
                    ਹਾਲਤ HTP_OP_DIV: compute_func = compute_ਪੰਕਤੀ_bcast_div_f16; ਰੋਕੋ;
                    ਮੂਲ: ਰੋਕੋ;
                }
            }
            ਰੋਕੋ;
        ਹਾਲਤ HTP_ਦੁਆਧਾਰੀ_KERNEL_SCALAR:
            ਕਾਮਾ_func = ਦੁਆਧਾਰੀ_thread_scalar;
            ਜੇ (src0_type == HTP_TYPE_F32) {
                ਚੋਣ (octx->op) {
                    ਹਾਲਤ HTP_OP_ਜੋੜੋ: compute_func = compute_scalar_ਜੋੜੋ_f32; ਰੋਕੋ;
                    ਹਾਲਤ HTP_OP_SUB: compute_func = compute_scalar_sub_f32; ਰੋਕੋ;
                    ਹਾਲਤ HTP_OP_MUL: compute_func = compute_scalar_mul_f32; ਰੋਕੋ;
                    ਹਾਲਤ HTP_OP_DIV: compute_func = compute_scalar_div_f32; ਰੋਕੋ;
                    ਮੂਲ: ਰੋਕੋ;
                }
            } ਨਹੀਂਤਾਂ ਜੇ (src0_type == HTP_TYPE_F16) {
                ਚੋਣ (octx->op) {
                    ਹਾਲਤ HTP_OP_ਜੋੜੋ: compute_func = compute_scalar_ਜੋੜੋ_f16; ਰੋਕੋ;
                    ਹਾਲਤ HTP_OP_SUB: compute_func = compute_scalar_sub_f16; ਰੋਕੋ;
                    ਹਾਲਤ HTP_OP_MUL: compute_func = compute_scalar_mul_f16; ਰੋਕੋ;
                    ਹਾਲਤ HTP_OP_DIV: compute_func = compute_scalar_div_f16; ਰੋਕੋ;
                    ਮੂਲ: ਰੋਕੋ;
                }
            }
            ਰੋਕੋ;
        ਹਾਲਤ HTP_ਦੁਆਧਾਰੀ_KERNEL_COMPLEX:
            ਕਾਮਾ_func = ਦੁਆਧਾਰੀ_thread_ਸਦਿਸ਼_complex;
            ਜੇ (src0_type == HTP_TYPE_F32) {
                ਚੋਣ (octx->op) {
                    ਹਾਲਤ HTP_OP_ਜੋੜੋ: compute_func = compute_complex_ਜੋੜੋ_f32; ਰੋਕੋ;
                    ਹਾਲਤ HTP_OP_SUB: compute_func = compute_complex_sub_f32; ਰੋਕੋ;
                    ਹਾਲਤ HTP_OP_MUL: compute_func = compute_complex_mul_f32; ਰੋਕੋ;
                    ਹਾਲਤ HTP_OP_DIV: compute_func = compute_complex_div_f32; ਰੋਕੋ;
                    ਮੂਲ: ਰੋਕੋ;
                }
            } ਨਹੀਂਤਾਂ ਜੇ (src0_type == HTP_TYPE_F16) {
                ਚੋਣ (octx->op) {
                    ਹਾਲਤ HTP_OP_ਜੋੜੋ: compute_func = compute_complex_ਜੋੜੋ_f16; ਰੋਕੋ;
                    ਹਾਲਤ HTP_OP_SUB: compute_func = compute_complex_sub_f16; ਰੋਕੋ;
                    ਹਾਲਤ HTP_OP_MUL: compute_func = compute_complex_mul_f16; ਰੋਕੋ;
                    ਹਾਲਤ HTP_OP_DIV: compute_func = compute_complex_div_f16; ਰੋਕੋ;
                    ਮੂਲ: ਰੋਕੋ;
                }
            }
            ਰੋਕੋ;
        ਹਾਲਤ HTP_ਦੁਆਧਾਰੀ_KERNEL_REPEAT:
            ਕਾਮਾ_func = ਦੁਆਧਾਰੀ_thread_element_repeat;
            ਜੇ (src0_type == HTP_TYPE_F32) {
                ਚੋਣ (octx->op) {
                    ਹਾਲਤ HTP_OP_ਜੋੜੋ: compute_func = compute_repeat_ਜੋੜੋ_f32; ਰੋਕੋ;
                    ਹਾਲਤ HTP_OP_SUB: compute_func = compute_repeat_sub_f32; ਰੋਕੋ;
                    ਹਾਲਤ HTP_OP_MUL: compute_func = compute_repeat_mul_f32; ਰੋਕੋ;
                    ਹਾਲਤ HTP_OP_DIV: compute_func = compute_repeat_div_f32; ਰੋਕੋ;
                    ਮੂਲ: ਰੋਕੋ;
                }
            } ਨਹੀਂਤਾਂ ਜੇ (src0_type == HTP_TYPE_F16) {
                ਚੋਣ (octx->op) {
                    ਹਾਲਤ HTP_OP_ਜੋੜੋ: compute_func = compute_repeat_ਜੋੜੋ_f16; ਰੋਕੋ;
                    ਹਾਲਤ HTP_OP_SUB: compute_func = compute_repeat_sub_f16; ਰੋਕੋ;
                    ਹਾਲਤ HTP_OP_MUL: compute_func = compute_repeat_mul_f16; ਰੋਕੋ;
                    ਹਾਲਤ HTP_OP_DIV: compute_func = compute_repeat_div_f16; ਰੋਕੋ;
                    ਮੂਲ: ਰੋਕੋ;
                }
            }
            ਰੋਕੋ;
        ਹਾਲਤ HTP_ਦੁਆਧਾਰੀ_KERNEL_ਜੋੜੋ_ID:
            ਜੇ (octx->op == HTP_OP_ਜੋੜੋ_ID && src0_type == HTP_TYPE_F32) {
                ਕਾਮਾ_func = ਦੁਆਧਾਰੀ_thread_ਜੋੜੋ_id_f32;
                compute_func = (ਖਾਲੀ *) compute_ਜੋੜੋ_id_f32;
            }
            ਰੋਕੋ;
        ਮੂਲ: ਰੋਕੋ;
    }

    ਜੇ (!ਕਾਮਾ_func || !compute_func) {
        ਵਾਪਸ HTP_ਸਥਿਤੀ_NO_SUPPORT;
    }

    bctx.compute = compute_func;

    ਜੇ (kparams->kernel_type == HTP_ਦੁਆਧਾਰੀ_KERNEL_ਪੰਕਤੀ_BCAST ||
        kparams->kernel_type == HTP_ਦੁਆਧਾਰੀ_KERNEL_SCALAR ||
        kparams->kernel_type == HTP_ਦੁਆਧਾਰੀ_KERNEL_REPEAT ||
        kparams->kernel_type == HTP_ਦੁਆਧਾਰੀ_KERNEL_COMPLEX ||
        kparams->kernel_type == HTP_ਦੁਆਧਾਰੀ_KERNEL_ਜੋੜੋ_ID) {
        dma_ਕਤਾਰ_flush(dma_q);
    }

    work_ਕਤਾਰ_ਲੜੀਵਾਰ(octx->ctx->work_ਕਤਾਰ, ਕਾਮਾ_func, &bctx, n_ਧਾਗੇ);

    ਵਾਪਸ HTP_ਸਥਿਤੀ_OK;
}

ਪੂਰਨ op_ਦੁਆਧਾਰੀ(ਬਣਤਰ htp_ops_context * octx) {
    ਅਚਲ uint32_t src0_type = octx->src[0]->type;
    ਜੇ ((src0_type == HTP_TYPE_F32) || (src0_type == HTP_TYPE_F16)) {
        ਵਾਪਸ execute_op_ਦੁਆਧਾਰੀ(octx);
    }

    ਵਾਪਸ HTP_ਸਥਿਤੀ_NO_SUPPORT;
}
