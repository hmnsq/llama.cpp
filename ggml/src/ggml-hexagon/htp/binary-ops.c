#لارښوونه clang diagnostic ignored "-Wunused-variable"
#لارښوونه clang diagnostic ignored "-Wunused-function"
#لارښوونه clang diagnostic ignored "-Wunused-but-set-variable"

#ضمن <HAP_farf.h>
#ضمن <HAP_perf.h>

#ضمن <math.h>
#ضمن <string.h>

#ضمن "dma-queue.h"
#ضمن "hvx-utils.h"

#تعریف_کړه GGML_COMMON_DECL_C
#ضمن "ggml-common.h"
#ضمن "hex-common.h"
#ضمن "hex-profile.h"
#ضمن "binary-ops.h"
#ضمن "htp-ctx.h"
#ضمن "htp-ops.h"
#ضمن "htp-tensor.h"

#که_نه_تعریف_شوی MIN
#تعریف_کړه MIN(a, b) ((a) < (b) ? (a) : (b))
#د_که_پای

// Context for binary operations
بنية htp_دوګونی_context {
    بنية htp_ops_context * octx;
    بنية htp_دوګونی_vtcm_layout vtcm_layout;
    uint8_t * vtcm_بنسټ;

    بنية fastdiv_ارزښتونه src0_dim1_div; // ne01
    بنية fastdiv_ارزښتونه src0_dim2_div; // ne02
    بنية fastdiv_ارزښتونه src0_dim12_div;// ne03

    بنية fastdiv_ارزښتونه src1_dim1_div; // ne11
    بنية fastdiv_ارزښتونه src1_dim2_div; // ne12
    بنية fastdiv_ارزښتونه src1_dim3_div; // ne13

    uint32_t بلاک_max;
    uint32_t nrows_فی_thread;
    uint32_t ټول_لیکې;
    uint32_t لیکه_پیل;
    size_t   src0_لیکه_اندازه_aligned;
    size_t   src1_لیکه_اندازه_aligned;
    size_t   dst_لیکه_اندازه_aligned;
    size_t   لیکه_اندازه_بایټونه;

    bool وېشل_at_ne01;
    bool وېشل_at_ne02;

    فراغ * compute;
};

#تعریف_کړه htp_دوګونی_preamble                        \
    ثابت بنية htp_tensor * src0 = octx->src[0]; \
    ثابت بنية htp_tensor * src1 = octx->src[1]; \
    ثابت بنية htp_tensor * dst  = octx->dst;    \
                                                   \
    ثابت uint32_t ne00 = src0->ne[0];             \
    ثابت uint32_t ne01 = src0->ne[1];             \
    ثابت uint32_t ne02 = src0->ne[2];             \
    ثابت uint32_t ne03 = src0->ne[3];             \
                                                   \
    ثابت uint32_t ne10 = src1->ne[0];             \
    ثابت uint32_t ne11 = src1->ne[1];             \
    ثابت uint32_t ne12 = src1->ne[2];             \
    ثابت uint32_t ne13 = src1->ne[3];             \
                                                   \
    ثابت uint32_t nb01 = src0->nb[1];             \
    ثابت uint32_t nb02 = src0->nb[2];             \
    ثابت uint32_t nb03 = src0->nb[3];             \
                                                   \
    ثابت uint32_t nb11 = src1->nb[1];             \
    ثابت uint32_t nb12 = src1->nb[2];             \
    ثابت uint32_t nb13 = src1->nb[3];             \
                                                   \
    ثابت uint32_t nb1 = dst->nb[1];               \
    ثابت uint32_t nb2 = dst->nb[2];               \
    ثابت uint32_t nb3 = dst->nb[3];

ساكن مضمن uint32_t calc_بلاک_اندازه(بنية htp_دوګونی_context * bctx, uint32_t ir, uint32_t پای_ټکی_لیکه, uint32_t ne01, uint32_t ne02) {
    uint32_t i03, i02, i01, rem;
    i03 = fastdiv(ir, &bctx->src0_dim12_div);
    rem = ir - i03 * (ne02 * ne01);
    i02 = fastdiv(rem, &bctx->src0_dim1_div);
    i01 = rem - i02 * ne01;

    uint32_t لیکې_کیڼ = پای_ټکی_لیکه - ir;
    uint32_t بلاک_حد = لیکې_کیڼ;

    إذا (bctx->وېشل_at_ne01) {
        بلاک_حد = MIN(بلاک_حد, ne01 - i01);
    }
    إذا (bctx->وېشل_at_ne02) {
         uint32_t لیکې_دننه_plane = (ne02 * ne01) - rem;
         بلاک_حد = MIN(بلاک_حد, لیکې_دننه_plane);
    }

    أرجع MIN(bctx->بلاک_max, بلاک_حد);
}

// Out-of-line compute micro-kernels

تعريفنوع فراغ (*compute_scalar_t)(
    uint8_t * dst, ثابت uint8_t * src0, ثابت فراغ * s1_جدول,
    uint32_t cur_i11, uint32_t ne11, uint32_t n_لیکې,
    size_t dst_stride, size_t src0_stride, uint32_t ne00);

#تعریف_کړه DEFINE_COMPUTE_SCALAR(NAME, TYPE, HVX_STMT)             \
ساكن فراغ compute_scalar_##NAME(                              \
    uint8_t * dst, ثابت uint8_t * src0, ثابت فراغ * s1_جدول, \
    uint32_t cur_i11, uint32_t ne11, uint32_t n_لیکې,           \
    size_t dst_stride, size_t src0_stride, uint32_t ne00) {     \
    ثابت TYPE * جدول = (ثابت TYPE *) s1_جدول;               \
    لكل (uint32_t r = 0; r < n_لیکې; r++) {                     \
        uint8_t * r_dst = dst + r * dst_stride;                 \
        ثابت uint8_t * r_src0 = src0 + r * src0_stride;        \
        TYPE val = جدول[cur_i11];                              \
        HVX_STMT;                                               \
        إذا (ne11 > 1 && ++cur_i11 == ne11) {                    \
            cur_i11 = 0;                                        \
        }                                                       \
    }                                                           \
}

DEFINE_COMPUTE_SCALAR(جمع_کړه_f32, عائم,    hvx_جمع_کړه_scalar_f32_aa(r_dst, r_src0, val, ne00))
DEFINE_COMPUTE_SCALAR(جمع_کړه_f16, _Float16, hvx_جمع_کړه_scalar_f16_aa(r_dst, r_src0, val, ne00))
DEFINE_COMPUTE_SCALAR(sub_f32, عائم,    hvx_sub_scalar_f32_aa(r_dst, r_src0, val, ne00))
DEFINE_COMPUTE_SCALAR(sub_f16, _Float16, hvx_sub_scalar_f16_aa(r_dst, r_src0, val, ne00))
DEFINE_COMPUTE_SCALAR(mul_f32, عائم,    hvx_mul_scalar_f32_aa(r_dst, r_src0, val, ne00))
DEFINE_COMPUTE_SCALAR(mul_f16, _Float16, hvx_mul_scalar_f16_aa(r_dst, r_src0, val, ne00))
DEFINE_COMPUTE_SCALAR(div_f32, عائم,    hvx_mul_scalar_f32_aa(r_dst, r_src0, 1.0f / (val), ne00))
DEFINE_COMPUTE_SCALAR(div_f16, _Float16, hvx_div_scalar_f16_aa(r_dst, r_src0, val, ne00))

تعريفنوع فراغ (*compute_same_shape_t)(
    uint8_t * dst, ثابت uint8_t * src0, ثابت uint8_t * src1,
    uint32_t n_لیکې, size_t dst_stride, size_t src0_stride, size_t src1_stride, uint32_t ne00);

#تعریف_کړه DEFINE_COMPUTE_SAME_SHAPE(NAME, HVX_FN)                                                  \
ساكن فراغ compute_same_shape_##NAME(                                                           \
    uint8_t * dst, ثابت uint8_t * src0, ثابت uint8_t * src1,                                   \
    uint32_t n_لیکې, size_t dst_stride, size_t src0_stride, size_t src1_stride, uint32_t ne00) { \
    لكل (uint32_t r = 0; r < n_لیکې; r++) {                                                      \
        HVX_FN(dst + r * dst_stride, src0 + r * src0_stride, src1 + r * src1_stride, ne00);      \
    }                                                                                            \
}

DEFINE_COMPUTE_SAME_SHAPE(جمع_کړه_f32, hvx_جمع_کړه_f32_aaa)
DEFINE_COMPUTE_SAME_SHAPE(جمع_کړه_f16, hvx_جمع_کړه_f16_aaa)
DEFINE_COMPUTE_SAME_SHAPE(sub_f32, hvx_sub_f32_aaa)
DEFINE_COMPUTE_SAME_SHAPE(sub_f16, hvx_sub_f16_aaa)
DEFINE_COMPUTE_SAME_SHAPE(mul_f32, hvx_mul_f32_aaa)
DEFINE_COMPUTE_SAME_SHAPE(mul_f16, hvx_mul_f16_aaa)
DEFINE_COMPUTE_SAME_SHAPE(div_f32, hvx_div_f32_aaa)
DEFINE_COMPUTE_SAME_SHAPE(div_f16, hvx_div_f16_aaa)

تعريفنوع فراغ (*compute_لیکه_bcast_t)(
    uint8_t * dst, ثابت uint8_t * src0, ثابت uint8_t * src1,
    uint32_t n_لیکې, size_t dst_stride, size_t src0_stride, uint32_t ne00);

#تعریف_کړه DEFINE_COMPUTE_لیکه_BCAST(NAME, HVX_FN)                                               \
ساكن فراغ compute_لیکه_bcast##NAME(                                                        \
    uint8_t * dst, ثابت uint8_t * src0, ثابت uint8_t * src1,                               \
    uint32_t n_لیکې, size_t dst_stride, size_t src0_stride, uint32_t ne00) {                 \
    لكل (uint32_t r = 0; r < n_لیکې; r++) {                                                  \
        HVX_FN(dst + r * dst_stride, src0 + r * src0_stride, src1, ne00);                    \
    }                                                                                        \
}

DEFINE_COMPUTE_لیکه_BCAST(جمع_کړه_f32, hvx_جمع_کړه_f32_aaa)
DEFINE_COMPUTE_لیکه_BCAST(جمع_کړه_f16, hvx_جمع_کړه_f16_aaa)
DEFINE_COMPUTE_لیکه_BCAST(sub_f32, hvx_sub_f32_aaa)
DEFINE_COMPUTE_لیکه_BCAST(sub_f16, hvx_sub_f16_aaa)
DEFINE_COMPUTE_لیکه_BCAST(mul_f32, hvx_mul_f32_aaa)
DEFINE_COMPUTE_لیکه_BCAST(mul_f16, hvx_mul_f16_aaa)
DEFINE_COMPUTE_لیکه_BCAST(div_f32, hvx_div_f32_aaa)
DEFINE_COMPUTE_لیکه_BCAST(div_f16, hvx_div_f16_aaa)

تعريفنوع فراغ (*compute_complex_t)(
    uint8_t * dst, ثابت uint8_t * src0, ثابت uint8_t * src1_plane,
    uint32_t i01, uint32_t ne11, ثابت بنية fastdiv_ارزښتونه * div11, uint32_t src1_stride,
    uint32_t n_لیکې, size_t dst_stride, size_t src0_stride, uint32_t ne00);

#تعریف_کړه DEFINE_COMPUTE_COMPLEX(NAME, HVX_FN)                                                 \
ساكن فراغ compute_complex_##NAME(                                                          \
    uint8_t * dst, ثابت uint8_t * src0, ثابت uint8_t * src1_plane,                         \
    uint32_t i01, uint32_t ne11, ثابت بنية fastdiv_ارزښتونه * div11, uint32_t src1_stride,  \
    uint32_t n_لیکې, size_t dst_stride, size_t src0_stride, uint32_t ne00) {                 \
    لكل (uint32_t r = 0; r < n_لیکې; r++) {                                                  \
        uint32_t i11 = fastmodulo(i01 + r, ne11, div11);                                     \
        ثابت uint8_t * r_src1 = src1_plane + i11 * src1_stride;                             \
        HVX_FN(dst + r * dst_stride, src0 + r * src0_stride, r_src1, ne00);                  \
    }                                                                                        \
}

DEFINE_COMPUTE_COMPLEX(جمع_کړه_f32, hvx_جمع_کړه_f32_aau)
DEFINE_COMPUTE_COMPLEX(جمع_کړه_f16, hvx_جمع_کړه_f16_aau)
DEFINE_COMPUTE_COMPLEX(sub_f32, hvx_sub_f32_aau)
DEFINE_COMPUTE_COMPLEX(sub_f16, hvx_sub_f16_aau)
DEFINE_COMPUTE_COMPLEX(mul_f32, hvx_mul_f32_aau)
DEFINE_COMPUTE_COMPLEX(mul_f16, hvx_mul_f16_aau)
DEFINE_COMPUTE_COMPLEX(div_f32, hvx_div_f32_aau)
DEFINE_COMPUTE_COMPLEX(div_f16, hvx_div_f16_aau)

تعريفنوع فراغ (*compute_repeat_t)(
    uint8_t * dst, ثابت uint8_t * src0, ثابت uint8_t * src1_plane,
    uint32_t i01, uint32_t ne11, ثابت بنية fastdiv_ارزښتونه * div11, uint32_t src1_stride,
    uint32_t n_لیکې, size_t dst_stride, size_t src0_stride, uint32_t ne00, uint32_t ne10);

#تعریف_کړه DEFINE_COMPUTE_REPEAT(NAME, TYPE, HVX_FN)                                            \
ساكن فراغ compute_repeat_##NAME(                                                           \
    uint8_t * dst, ثابت uint8_t * src0, ثابت uint8_t * src1_plane,                         \
    uint32_t i01, uint32_t ne11, ثابت بنية fastdiv_ارزښتونه * div11, uint32_t src1_stride,  \
    uint32_t n_لیکې, size_t dst_stride, size_t src0_stride, uint32_t ne00, uint32_t ne10) {  \
    لكل (uint32_t r = 0; r < n_لیکې; r++) {                                                  \
        uint32_t i11 = fastmodulo(i01 + r, ne11, div11);                                     \
        ثابت uint8_t * r_src1_لیکه = src1_plane + i11 * src1_stride;                         \
        uint8_t * r_dst = dst + r * dst_stride;                                              \
        ثابت uint8_t * r_src0 = src0 + r * src0_stride;                                     \
        لكل (uint32_t c = 0; c < ne00; c += ne10) {                                          \
            uint32_t len = MIN(ne10, ne00 - c);                                              \
            HVX_FN(r_dst + c * حجم(TYPE), r_src0 + c * حجم(TYPE), r_src1_لیکه, len);    \
        }                                                                                    \
    }                                                                                        \
}

DEFINE_COMPUTE_REPEAT(جمع_کړه_f32, عائم,    hvx_جمع_کړه_f32_uuu)
DEFINE_COMPUTE_REPEAT(جمع_کړه_f16, _Float16, hvx_جمع_کړه_f16_uuu)
DEFINE_COMPUTE_REPEAT(sub_f32, عائم,    hvx_sub_f32_uuu)
DEFINE_COMPUTE_REPEAT(sub_f16, _Float16, hvx_sub_f16_uuu)
DEFINE_COMPUTE_REPEAT(mul_f32, عائم,    hvx_mul_f32_uuu)
DEFINE_COMPUTE_REPEAT(mul_f16, _Float16, hvx_mul_f16_uuu)
DEFINE_COMPUTE_REPEAT(div_f32, عائم,    hvx_div_f32_uuu)
DEFINE_COMPUTE_REPEAT(div_f16, _Float16, hvx_div_f16_uuu)

تعريفنوع فراغ (*compute_جمع_کړه_id_t)(
    uint8_t * dst, ثابت uint8_t * src0, ثابت uint8_t * src1_معلومات, ثابت int32_t * src2_ids,
    uint32_t i01, uint32_t i02, uint32_t ne01, uint32_t src1_stride,
    uint32_t n_لیکې, size_t dst_stride, size_t src0_stride, uint32_t ne00);

ساكن فراغ compute_جمع_کړه_id_f32(
    uint8_t * dst, ثابت uint8_t * src0, ثابت uint8_t * src1_معلومات, ثابت int32_t * src2_ids,
    uint32_t i01, uint32_t i02, uint32_t ne01, uint32_t src1_stride,
    uint32_t n_لیکې, size_t dst_stride, size_t src0_stride, uint32_t ne00) {
    لكل (uint32_t r = 0; r < n_لیکې; r++) {
        uint32_t r_i01 = i01 + r;
        ثابت int32_t idx = src2_ids[i02 * ne01 + r_i01];
        إذا (idx < 0) {
            حافظه_نقل(dst + r * dst_stride, src0 + r * src0_stride, ne00 * حجم(عائم));
            استمر;
        }
        ثابت uint8_t * r_src1 = src1_معلومات + idx * src1_stride;
        ثابت uint8_t * r_src0 = src0 + r * src0_stride;
        uint8_t * r_dst = dst + r * dst_stride;
        hvx_جمع_کړه_f32_aaa(r_dst, r_src0, r_src1, ne00);
    }
}

// 1. Scalar src1 in VTCM via DMA (ne10 == 1)
ساكن فراغ دوګونی_thread_scalar(بینښه صحيح nth, بینښه صحيح ith, فراغ * معلومات) {
    بنية htp_دوګونی_context * bctx = (بنية htp_دوګونی_context *) معلومات;
    بنية htp_ops_context * octx = bctx->octx;
    htp_دوګونی_preamble;

    ثابت uint32_t elem_اندازه = (src0->type == HTP_TYPE_F32) ? حجم(عائم) : حجم(_Float16);
    ثابت size_t plane_stride = (size_t) ne12 * ne11 * elem_اندازه;
    ثابت size_t لیکه_stride   = (size_t) ne11 * elem_اندازه;

    ثابت uint32_t لیکه_اندازه_بایټونه = bctx->لیکه_اندازه_بایټونه;
    ثابت uint32_t پیل_لیکه = bctx->لیکه_پیل + bctx->nrows_فی_thread * ith;
    ثابت uint32_t پای_ټکی_لیکه   = MIN(پیل_لیکه + bctx->nrows_فی_thread, bctx->لیکه_پیل + bctx->ټول_لیکې);
    إذا (پیل_لیکه >= پای_ټکی_لیکه) أرجع;

    FARF(لوړ_2, "binary-scalar: %d/%d (%u:%u) row-size %u (%u)",
         ith, nth, پیل_لیکه, پای_ټکی_لیکه, nb01, bctx->dst_لیکه_اندازه_aligned);

    ثابت بنية htp_دوګونی_vtcm_layout * layout = &bctx->vtcm_layout;
    uint8_t * src0_spad_بنسټ = VTCM_LAYOUT_PTR(uint8_t, bctx->vtcm_بنسټ, layout->off_src0) + (ith * layout->src0_بایټونه_فی_thread);
    uint8_t * dst_spad_بنسټ  = VTCM_LAYOUT_PTR(uint8_t, bctx->vtcm_بنسټ, layout->off_dst)  + (ith * layout->dst_بایټونه_فی_thread);
    size_t src0_spad_half    = layout->src0_spad_half_اندازه;
    size_t dst_spad_half     = layout->dst_spad_half_اندازه;
    ثابت فراغ * s1_جدول    = VTCM_LAYOUT_PTR(ثابت فراغ, bctx->vtcm_بنسټ, layout->off_src1);

    dma_کتار * dma_q = octx->ctx->dma[ith];
    uint32_t ir_prefetch = پیل_لیکه;
    صحيح spad_idx = 0;

    لكل (صحيح k = 0; k < 2 && ir_prefetch < پای_ټکی_لیکه; k++) {
        uint32_t current_بلاک_اندازه = calc_بلاک_اندازه(bctx, ir_prefetch, پای_ټکی_لیکه, ne01, ne02);
        uint32_t i03, i02, i01, rem;
        i03 = fastdiv(ir_prefetch, &bctx->src0_dim12_div);
        rem = ir_prefetch - i03 * (ne02 * ne01);
        i02 = fastdiv(rem, &bctx->src0_dim1_div);
        i01 = rem - i02 * ne01;

        dma_addr_t src0_curr = src0->معلومات + i03 * nb03 + i02 * nb02 + i01 * nb01;
        dma_addr_t dst_curr  = dst->معلومات  + i03 * nb3  + i02 * nb2  + i01 * nb1;

        uint8_t * s0_spad = src0_spad_بنسټ + spad_idx * src0_spad_half;
        uint8_t * d_spad  = dst_spad_بنسټ  + spad_idx * dst_spad_half;

        dma_کتار_ټېل(dma_q, dma_make_معلومات(dst_curr, d_spad), nb1, bctx->dst_لیکه_اندازه_aligned, لیکه_اندازه_بایټونه, 0);
        dma_کتار_ټېل(dma_q, dma_make_معلومات(s0_spad, src0_curr), bctx->src0_لیکه_اندازه_aligned, nb01, لیکه_اندازه_بایټونه,
                                                             current_بلاک_اندازه);
        ir_prefetch += current_بلاک_اندازه;
        spad_idx ^= 1;
    }

    بنية htp_thread_trace * tr = &octx->ctx->trace[ith];
    compute_scalar_t compute = (compute_scalar_t) bctx->compute;

    لكل (uint32_t ir = پیل_لیکه; ir < پای_ټکی_لیکه; ) {
        uint32_t current_بلاک_اندازه = calc_بلاک_اندازه(bctx, ir, پای_ټکی_لیکه, ne01, ne02);

        uint8_t * d_spad  = (uint8_t *) dma_کتار_راوباسه(dma_q).src;
        uint8_t * s0_spad = (uint8_t *) dma_کتار_راوباسه(dma_q).dst;

        uint32_t i03, i02, i01, rem;
        i03 = fastdiv(ir, &bctx->src0_dim12_div);
        rem = ir - i03 * (ne02 * ne01);
        i02 = fastdiv(rem, &bctx->src0_dim1_div);
        i01 = rem - i02 * ne01;

        uint32_t i13 = fastmodulo(i03, ne13, &bctx->src1_dim3_div);
        uint32_t i12 = fastmodulo(i02, ne12, &bctx->src1_dim2_div);
        uint32_t i11 = fastmodulo(i01, ne11, &bctx->src1_dim1_div);
        ثابت uint8_t * scalar_جدول = (ثابت uint8_t *) s1_جدول +
            i13 * plane_stride + i12 * لیکه_stride;

        htp_trace_event_پیل(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);
        compute(d_spad, s0_spad, scalar_جدول, i11, ne11, current_بلاک_اندازه,
                bctx->dst_لیکه_اندازه_aligned, bctx->src0_لیکه_اندازه_aligned, ne00);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);

        dma_addr_t dst_curr = dst->معلومات + i03 * nb3 + i02 * nb2 + i01 * nb1;
        dma_کتار_ټېل(dma_q, dma_make_معلومات(dst_curr, d_spad), nb1, bctx->dst_لیکه_اندازه_aligned, لیکه_اندازه_بایټونه,
                                                                current_بلاک_اندازه);

        إذا (ir_prefetch < پای_ټکی_لیکه) {
            uint32_t بل_بلاک_اندازه = calc_بلاک_اندازه(bctx, ir_prefetch, پای_ټکی_لیکه, ne01, ne02);
            uint32_t p03, p02, p01, prem;
            p03 = fastdiv(ir_prefetch, &bctx->src0_dim12_div);
            prem = ir_prefetch - p03 * (ne02 * ne01);
            p02 = fastdiv(prem, &bctx->src0_dim1_div);
            p01 = prem - p02 * ne01;
            dma_addr_t s0_بل = src0->معلومات + p03 * nb03 + p02 * nb02 + p01 * nb01;
            dma_کتار_ټېل(dma_q, dma_make_معلومات(s0_spad, s0_بل), bctx->src0_لیکه_اندازه_aligned, nb01, لیکه_اندازه_بایټونه,
                                                               بل_بلاک_اندازه);
            ir_prefetch += بل_بلاک_اندازه;
        }
        ir += current_بلاک_اندازه;
    }

    dma_کتار_flush(dma_q);
}

// 2. Vector Same Shape (ne1x == ne0x) or Simple Broadcast
ساكن فراغ دوګونی_thread_ویکټور_same_shape(بینښه صحيح nth, بینښه صحيح ith, فراغ * معلومات) {
    بنية htp_دوګونی_context * bctx = (بنية htp_دوګونی_context *) معلومات;
    بنية htp_ops_context * octx = bctx->octx;
    htp_دوګونی_preamble;

    ثابت uint32_t لیکه_اندازه_بایټونه = bctx->لیکه_اندازه_بایټونه;
    ثابت uint32_t پیل_لیکه = bctx->لیکه_پیل + bctx->nrows_فی_thread * ith;
    ثابت uint32_t پای_ټکی_لیکه   = MIN(پیل_لیکه + bctx->nrows_فی_thread, bctx->لیکه_پیل + bctx->ټول_لیکې);
    إذا (پیل_لیکه >= پای_ټکی_لیکه) أرجع;

    FARF(لوړ_2, "binary-same-shape: %d/%d (%u:%u) row-size %u (%u)",
         ith, nth, پیل_لیکه, پای_ټکی_لیکه, nb01, bctx->dst_لیکه_اندازه_aligned);

    ثابت بنية htp_دوګونی_vtcm_layout * layout = &bctx->vtcm_layout;
    uint8_t * src0_spad_بنسټ = VTCM_LAYOUT_PTR(uint8_t, bctx->vtcm_بنسټ, layout->off_src0) + (ith * layout->src0_بایټونه_فی_thread);
    uint8_t * src1_spad_بنسټ = VTCM_LAYOUT_PTR(uint8_t, bctx->vtcm_بنسټ, layout->off_src1) + (ith * layout->src1_بایټونه_فی_thread);
    uint8_t * dst_spad_بنسټ  = VTCM_LAYOUT_PTR(uint8_t, bctx->vtcm_بنسټ, layout->off_dst)  + (ith * layout->dst_بایټونه_فی_thread);

    size_t src0_spad_half = layout->src0_spad_half_اندازه;
    size_t src1_spad_half = layout->src1_spad_half_اندازه;
    size_t dst_spad_half  = layout->dst_spad_half_اندازه;

    dma_کتار * dma_q = octx->ctx->dma[ith];
    uint32_t ir_prefetch = پیل_لیکه;
    صحيح spad_idx = 0;

    لكل (صحيح k = 0; k < 2 && ir_prefetch < پای_ټکی_لیکه; k++) {
        uint32_t current_بلاک_اندازه = calc_بلاک_اندازه(bctx, ir_prefetch, پای_ټکی_لیکه, ne01, ne02);
        uint32_t i03, i02, i01, rem;
        i03 = fastdiv(ir_prefetch, &bctx->src0_dim12_div);
        rem = ir_prefetch - i03 * (ne02 * ne01);
        i02 = fastdiv(rem, &bctx->src0_dim1_div);
        i01 = rem - i02 * ne01;

        uint32_t i13 = (ne13 == 1) ? 0 : i03;
        uint32_t i12 = (ne12 == 1) ? 0 : i02;
        uint32_t i11 = (ne11 == 1) ? 0 : i01;

        dma_addr_t src0_curr = src0->معلومات + i03 * nb03 + i02 * nb02 + i01 * nb01;
        dma_addr_t src1_curr = src1->معلومات + i13 * nb13 + i12 * nb12 + i11 * nb11;
        dma_addr_t dst_curr  = dst->معلومات  + i03 * nb3  + i02 * nb2  + i01 * nb1;

        uint8_t * s0_spad = src0_spad_بنسټ + spad_idx * src0_spad_half;
        uint8_t * s1_spad = src1_spad_بنسټ + spad_idx * src1_spad_half;
        uint8_t * d_spad  = dst_spad_بنسټ  + spad_idx * dst_spad_half;

        dma_کتار_ټېل(dma_q, dma_make_معلومات(dst_curr, d_spad), nb1, bctx->dst_لیکه_اندازه_aligned, لیکه_اندازه_بایټونه, 0);
        dma_کتار_ټېل(dma_q, dma_make_معلومات(s0_spad, src0_curr), bctx->src0_لیکه_اندازه_aligned, nb01, لیکه_اندازه_بایټونه, current_بلاک_اندازه);
        dma_کتار_ټېل(dma_q, dma_make_معلومات(s1_spad, src1_curr), bctx->src1_لیکه_اندازه_aligned, nb11, لیکه_اندازه_بایټونه, current_بلاک_اندازه);
        ir_prefetch += current_بلاک_اندازه;
        spad_idx ^= 1;
    }

    بنية htp_thread_trace * tr = &octx->ctx->trace[ith];
    compute_same_shape_t compute = (compute_same_shape_t) bctx->compute;

    لكل (uint32_t ir = پیل_لیکه; ir < پای_ټکی_لیکه; ) {
        uint32_t current_بلاک_اندازه = calc_بلاک_اندازه(bctx, ir, پای_ټکی_لیکه, ne01, ne02);
        uint8_t * d_spad  = (uint8_t *) dma_کتار_راوباسه(dma_q).src;
        uint8_t * s0_spad = (uint8_t *) dma_کتار_راوباسه(dma_q).dst;
        uint8_t * s1_spad = (uint8_t *) dma_کتار_راوباسه(dma_q).dst;

        htp_trace_event_پیل(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);
        compute(d_spad, s0_spad, s1_spad, current_بلاک_اندازه,
                bctx->dst_لیکه_اندازه_aligned, bctx->src0_لیکه_اندازه_aligned, bctx->src1_لیکه_اندازه_aligned, ne00);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);

        uint32_t i03, i02, i01, rem;
        i03 = fastdiv(ir, &bctx->src0_dim12_div);
        rem = ir - i03 * (ne02 * ne01);
        i02 = fastdiv(rem, &bctx->src0_dim1_div);
        i01 = rem - i02 * ne01;
        dma_addr_t dst_curr = dst->معلومات + i03 * nb3 + i02 * nb2 + i01 * nb1;
        dma_کتار_ټېل(dma_q, dma_make_معلومات(dst_curr, d_spad), nb1, bctx->dst_لیکه_اندازه_aligned, لیکه_اندازه_بایټونه, current_بلاک_اندازه);

        إذا (ir_prefetch < پای_ټکی_لیکه) {
            uint32_t بل_بلاک_اندازه = calc_بلاک_اندازه(bctx, ir_prefetch, پای_ټکی_لیکه, ne01, ne02);
            uint32_t p03, p02, p01, prem;
            p03 = fastdiv(ir_prefetch, &bctx->src0_dim12_div);
            prem = ir_prefetch - p03 * (ne02 * ne01);
            p02 = fastdiv(prem, &bctx->src0_dim1_div);
            p01 = prem - p02 * ne01;

            uint32_t p13 = (ne13 == 1) ? 0 : p03;
            uint32_t p12 = (ne12 == 1) ? 0 : p02;
            uint32_t p11 = (ne11 == 1) ? 0 : p01;

            dma_addr_t s0_بل = src0->معلومات + p03 * nb03 + p02 * nb02 + p01 * nb01;
            dma_addr_t s1_بل = src1->معلومات + p13 * nb13 + p12 * nb12 + p11 * nb11;

            dma_کتار_ټېل(dma_q, dma_make_معلومات(s0_spad, s0_بل), bctx->src0_لیکه_اندازه_aligned, nb01, لیکه_اندازه_بایټونه, بل_بلاک_اندازه);
            dma_کتار_ټېل(dma_q, dma_make_معلومات(s1_spad, s1_بل), bctx->src1_لیکه_اندازه_aligned, nb11, لیکه_اندازه_بایټونه, بل_بلاک_اندازه);

            ir_prefetch += بل_بلاک_اندازه;
        }
        ir += current_بلاک_اندازه;
    }

    dma_کتار_flush(dma_q);
}

// 3. Row Broadcast (ne11 == 1, ne12 == 1, single row src1)
ساكن فراغ دوګونی_thread_ویکټور_لیکه_خپرونه(بینښه صحيح nth, بینښه صحيح ith, فراغ * معلومات) {
    بنية htp_دوګونی_context * bctx = (بنية htp_دوګونی_context *) معلومات;
    بنية htp_ops_context * octx = bctx->octx;
    htp_دوګونی_preamble;

    ثابت uint32_t لیکه_اندازه_بایټونه = bctx->لیکه_اندازه_بایټونه;
    ثابت uint32_t پیل_لیکه  = bctx->لیکه_پیل + bctx->nrows_فی_thread * ith;
    ثابت uint32_t پای_ټکی_لیکه    = MIN(پیل_لیکه + bctx->nrows_فی_thread, bctx->لیکه_پیل + bctx->ټول_لیکې);
    إذا (پیل_لیکه >= پای_ټکی_لیکه) أرجع;

    FARF(لوړ_2, "binary-row-bcast: %d/%d (%u:%u) row-size %u (%u)",
         ith, nth, پیل_لیکه, پای_ټکی_لیکه, nb01, bctx->dst_لیکه_اندازه_aligned);

    ثابت بنية htp_دوګونی_vtcm_layout * layout = &bctx->vtcm_layout;
    uint8_t * src0_spad_بنسټ = VTCM_LAYOUT_PTR(uint8_t, bctx->vtcm_بنسټ, layout->off_src0) + (ith * layout->src0_بایټونه_فی_thread);
    uint8_t * dst_spad_بنسټ  = VTCM_LAYOUT_PTR(uint8_t, bctx->vtcm_بنسټ, layout->off_dst)  + (ith * layout->dst_بایټونه_فی_thread);

    size_t src0_spad_half = layout->src0_spad_half_اندازه;
    size_t dst_spad_half  = layout->dst_spad_half_اندازه;

    dma_کتار * dma_q = octx->ctx->dma[ith];
    uint32_t ir_prefetch = پیل_لیکه;
    صحيح spad_idx = 0;

    فراغ * s1_ptr = VTCM_LAYOUT_PTR(فراغ, bctx->vtcm_بنسټ, layout->off_src1);

    لكل (صحيح k = 0; k < 2 && ir_prefetch < پای_ټکی_لیکه; k++) {
        uint32_t current_بلاک_اندازه = calc_بلاک_اندازه(bctx, ir_prefetch, پای_ټکی_لیکه, ne01, ne02);
        uint32_t i03 = fastdiv(ir_prefetch, &bctx->src0_dim12_div);
        uint32_t rem = ir_prefetch - i03 * (ne02 * ne01);
        uint32_t i02 = fastdiv(rem, &bctx->src0_dim1_div);
        uint32_t i01 = rem - i02 * ne01;

        dma_addr_t src0_curr = src0->معلومات + i03 * nb03 + i02 * nb02 + i01 * nb01;
        dma_addr_t dst_curr  = dst->معلومات  + i03 * nb3  + i02 * nb2  + i01 * nb1;

        uint8_t * s0_spad = src0_spad_بنسټ + spad_idx * src0_spad_half;
        uint8_t * d_spad  = dst_spad_بنسټ  + spad_idx * dst_spad_half;

        dma_کتار_ټېل(dma_q, dma_make_معلومات(dst_curr, d_spad), nb1, bctx->dst_لیکه_اندازه_aligned, لیکه_اندازه_بایټونه, 0);
        dma_کتار_ټېل(dma_q, dma_make_معلومات(s0_spad, src0_curr), bctx->src0_لیکه_اندازه_aligned, nb01, لیکه_اندازه_بایټونه, current_بلاک_اندازه);
        ir_prefetch += current_بلاک_اندازه;
        spad_idx ^= 1;
    }

    بنية htp_thread_trace * tr = &octx->ctx->trace[ith];
    compute_لیکه_bcast_t compute = (compute_لیکه_bcast_t) bctx->compute;

    لكل (uint32_t ir = پیل_لیکه; ir < پای_ټکی_لیکه; ) {
        uint32_t current_بلاک_اندازه = calc_بلاک_اندازه(bctx, ir, پای_ټکی_لیکه, ne01, ne02);
        uint8_t * d_spad  = (uint8_t *) dma_کتار_راوباسه(dma_q).src;
        uint8_t * s0_spad = (uint8_t *) dma_کتار_راوباسه(dma_q).dst;

        htp_trace_event_پیل(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);
        compute(d_spad, s0_spad, (ثابت uint8_t *)s1_ptr, current_بلاک_اندازه,
                bctx->dst_لیکه_اندازه_aligned, bctx->src0_لیکه_اندازه_aligned, ne00);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);

        uint32_t i03 = fastdiv(ir, &bctx->src0_dim12_div);
        uint32_t rem = ir - i03 * (ne02 * ne01);
        uint32_t i02 = fastdiv(rem, &bctx->src0_dim1_div);
        uint32_t i01 = rem - i02 * ne01;
        dma_addr_t dst_curr = dst->معلومات + i03 * nb3 + i02 * nb2 + i01 * nb1;
        dma_کتار_ټېل(dma_q, dma_make_معلومات(dst_curr, d_spad), nb1, bctx->dst_لیکه_اندازه_aligned, لیکه_اندازه_بایټونه, current_بلاک_اندازه);

        إذا (ir_prefetch < پای_ټکی_لیکه) {
            uint32_t بل_بلاک_اندازه = calc_بلاک_اندازه(bctx, ir_prefetch, پای_ټکی_لیکه, ne01, ne02);
            uint32_t p03, p02, p01, prem;
            p03  = fastdiv(ir_prefetch, &bctx->src0_dim12_div);
            prem = ir_prefetch - p03 * (ne02 * ne01);
            p02  = fastdiv(prem, &bctx->src0_dim1_div);
            p01  = prem - p02 * ne01;
            dma_addr_t s0_بل = src0->معلومات + p03 * nb03 + p02 * nb02 + p01 * nb01;
            dma_کتار_ټېل(dma_q, dma_make_معلومات(s0_spad, s0_بل), bctx->src0_لیکه_اندازه_aligned, nb01, لیکه_اندازه_بایټونه, بل_بلاک_اندازه);
            ir_prefetch += بل_بلاک_اندازه;
        }
        ir += current_بلاک_اندازه;
    }

    dma_کتار_flush(dma_q);
}

// 4. Vector Complex (ne10 == ne00, complex broadcast)
ساكن فراغ دوګونی_thread_ویکټور_complex(بینښه صحيح nth, بینښه صحيح ith, فراغ * معلومات) {
    بنية htp_دوګونی_context * bctx = (بنية htp_دوګونی_context *) معلومات;
    بنية htp_ops_context * octx = bctx->octx;
    htp_دوګونی_preamble;

    ثابت uint32_t لیکه_اندازه_بایټونه = bctx->لیکه_اندازه_بایټونه;
    ثابت uint32_t پیل_لیکه  = bctx->لیکه_پیل + bctx->nrows_فی_thread * ith;
    ثابت uint32_t پای_ټکی_لیکه    = MIN(پیل_لیکه + bctx->nrows_فی_thread, bctx->لیکه_پیل + bctx->ټول_لیکې);
    إذا (پیل_لیکه >= پای_ټکی_لیکه) أرجع;

    FARF(لوړ_2, "binary-complex: %d/%d (%u:%u) row-size %u (%u)",
         ith, nth, پیل_لیکه, پای_ټکی_لیکه, nb01, bctx->dst_لیکه_اندازه_aligned);

    ثابت بنية htp_دوګونی_vtcm_layout * layout = &bctx->vtcm_layout;
    uint8_t * src0_spad_بنسټ = VTCM_LAYOUT_PTR(uint8_t, bctx->vtcm_بنسټ, layout->off_src0) + (ith * layout->src0_بایټونه_فی_thread);
    uint8_t * dst_spad_بنسټ  = VTCM_LAYOUT_PTR(uint8_t, bctx->vtcm_بنسټ, layout->off_dst)  + (ith * layout->dst_بایټونه_فی_thread);
    size_t src0_spad_half    = layout->src0_spad_half_اندازه;
    size_t dst_spad_half     = layout->dst_spad_half_اندازه;
    ثابت uint8_t * vtcm_src1 = VTCM_LAYOUT_PTR(ثابت uint8_t, bctx->vtcm_بنسټ, layout->off_src1);

    ثابت uint32_t elem_اندازه = (src0->type == HTP_TYPE_F32) ? حجم(عائم) : حجم(_Float16);
    ثابت size_t vtcm_لیکه_stride   = (size_t) ne10 * elem_اندازه;
    ثابت size_t vtcm_plane_stride = (size_t) ne11 * vtcm_لیکه_stride;
    ثابت size_t vtcm_vol_stride   = (size_t) ne12 * vtcm_plane_stride;

    dma_کتار * dma_q = octx->ctx->dma[ith];
    uint32_t ir_prefetch = پیل_لیکه;
    صحيح spad_idx = 0;

    لكل (صحيح k = 0; k < 2 && ir_prefetch < پای_ټکی_لیکه; k++) {
        uint32_t current_بلاک_اندازه = calc_بلاک_اندازه(bctx, ir_prefetch, پای_ټکی_لیکه, ne01, ne02);
        uint32_t i03 = fastdiv(ir_prefetch, &bctx->src0_dim12_div);
        uint32_t rem = ir_prefetch - i03 * (ne02 * ne01);
        uint32_t i02 = fastdiv(rem, &bctx->src0_dim1_div);
        uint32_t i01 = rem - i02 * ne01;

        dma_addr_t src0_curr = src0->معلومات + i03 * nb03 + i02 * nb02 + i01 * nb01;
        dma_addr_t dst_curr  = dst->معلومات  + i03 * nb3  + i02 * nb2  + i01 * nb1;

        uint8_t * s0_spad = src0_spad_بنسټ + spad_idx * src0_spad_half;
        uint8_t * d_spad  = dst_spad_بنسټ  + spad_idx * dst_spad_half;

        dma_کتار_ټېل(dma_q, dma_make_معلومات(dst_curr, d_spad), nb1, bctx->dst_لیکه_اندازه_aligned, لیکه_اندازه_بایټونه, 0);
        dma_کتار_ټېل(dma_q, dma_make_معلومات(s0_spad, src0_curr), bctx->src0_لیکه_اندازه_aligned, nb01, لیکه_اندازه_بایټونه, current_بلاک_اندازه);
        ir_prefetch += current_بلاک_اندازه;
        spad_idx ^= 1;
    }

    بنية htp_thread_trace * tr = &octx->ctx->trace[ith];
    compute_complex_t compute = (compute_complex_t) bctx->compute;

    لكل (uint32_t ir = پیل_لیکه; ir < پای_ټکی_لیکه; ) {
        uint32_t current_بلاک_اندازه = calc_بلاک_اندازه(bctx, ir, پای_ټکی_لیکه, ne01, ne02);
        uint8_t * d_spad = (uint8_t *) dma_کتار_راوباسه(dma_q).src;
        uint8_t * s0_spad = (uint8_t *) dma_کتار_راوباسه(dma_q).dst;

        uint32_t i03 = fastdiv(ir, &bctx->src0_dim12_div);
        uint32_t rem = ir - i03 * (ne02 * ne01);
        uint32_t i02 = fastdiv(rem, &bctx->src0_dim1_div);
        uint32_t i01 = rem - i02 * ne01;

        uint32_t i13 = fastmodulo(i03, ne13, &bctx->src1_dim3_div);
        uint32_t i12 = fastmodulo(i02, ne12, &bctx->src1_dim2_div);
        ثابت uint8_t * src1_plane = vtcm_src1 + i13 * vtcm_vol_stride + i12 * vtcm_plane_stride;

        htp_trace_event_پیل(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);
        compute(d_spad, s0_spad, src1_plane, i01, ne11, &bctx->src1_dim1_div, vtcm_لیکه_stride,
                current_بلاک_اندازه, bctx->dst_لیکه_اندازه_aligned, bctx->src0_لیکه_اندازه_aligned, ne00);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);

        dma_addr_t dst_curr = dst->معلومات + i03 * nb3 + i02 * nb2 + i01 * nb1;
        dma_کتار_ټېل(dma_q, dma_make_معلومات(dst_curr, d_spad), nb1, bctx->dst_لیکه_اندازه_aligned, لیکه_اندازه_بایټونه, current_بلاک_اندازه);

        إذا (ir_prefetch < پای_ټکی_لیکه) {
            uint32_t بل_بلاک_اندازه = calc_بلاک_اندازه(bctx, ir_prefetch, پای_ټکی_لیکه, ne01, ne02);
            uint32_t p03, p02, p01, prem;
            p03  = fastdiv(ir_prefetch, &bctx->src0_dim12_div);
            prem = ir_prefetch - p03 * (ne02 * ne01);
            p02  = fastdiv(prem, &bctx->src0_dim1_div);
            p01  = prem - p02 * ne01;
            dma_addr_t s0_بل = src0->معلومات + p03 * nb03 + p02 * nb02 + p01 * nb01;
            dma_کتار_ټېل(dma_q, dma_make_معلومات(s0_spad, s0_بل), bctx->src0_لیکه_اندازه_aligned, nb01, لیکه_اندازه_بایټونه, بل_بلاک_اندازه);
            ir_prefetch += بل_بلاک_اندازه;
        }
        ir += current_بلاک_اندازه;
    }

    dma_کتار_flush(dma_q);
}

// 5. Element Repeat (ne10 != ne00)
ساكن فراغ دوګونی_thread_element_repeat(بینښه صحيح nth, بینښه صحيح ith, فراغ * معلومات) {
    بنية htp_دوګونی_context * bctx = (بنية htp_دوګونی_context *) معلومات;
    بنية htp_ops_context * octx = bctx->octx;
    htp_دوګونی_preamble;

    ثابت uint32_t لیکه_اندازه_بایټونه = bctx->لیکه_اندازه_بایټونه;
    ثابت uint32_t پیل_لیکه  = bctx->لیکه_پیل + bctx->nrows_فی_thread * ith;
    ثابت uint32_t پای_ټکی_لیکه    = MIN(پیل_لیکه + bctx->nrows_فی_thread, bctx->لیکه_پیل + bctx->ټول_لیکې);
    إذا (پیل_لیکه >= پای_ټکی_لیکه) أرجع;

    ثابت بنية htp_دوګونی_vtcm_layout * layout = &bctx->vtcm_layout;
    uint8_t * src0_spad_بنسټ = VTCM_LAYOUT_PTR(uint8_t, bctx->vtcm_بنسټ, layout->off_src0) + (ith * layout->src0_بایټونه_فی_thread);
    uint8_t * dst_spad_بنسټ  = VTCM_LAYOUT_PTR(uint8_t, bctx->vtcm_بنسټ, layout->off_dst)  + (ith * layout->dst_بایټونه_فی_thread);
    size_t src0_spad_half    = layout->src0_spad_half_اندازه;
    size_t dst_spad_half     = layout->dst_spad_half_اندازه;
    ثابت uint8_t * vtcm_src1 = VTCM_LAYOUT_PTR(ثابت uint8_t, bctx->vtcm_بنسټ, layout->off_src1);

    ثابت uint32_t elem_اندازه = (src0->type == HTP_TYPE_F32) ? حجم(عائم) : حجم(_Float16);
    ثابت size_t vtcm_لیکه_stride   = (size_t) ne10 * elem_اندازه;
    ثابت size_t vtcm_plane_stride = (size_t) ne11 * vtcm_لیکه_stride;
    ثابت size_t vtcm_vol_stride   = (size_t) ne12 * vtcm_plane_stride;

    FARF(لوړ_2, "binary-repeat: %d/%d (%u:%u) row-size %u (%u)",
         ith, nth, پیل_لیکه, پای_ټکی_لیکه, nb01, bctx->dst_لیکه_اندازه_aligned);

    dma_کتار * dma_q = octx->ctx->dma[ith];
    uint32_t ir_prefetch = پیل_لیکه;
    صحيح spad_idx = 0;

    لكل (صحيح k = 0; k < 2 && ir_prefetch < پای_ټکی_لیکه; k++) {
        uint32_t current_بلاک_اندازه = calc_بلاک_اندازه(bctx, ir_prefetch, پای_ټکی_لیکه, ne01, ne02);
        uint32_t i03 = fastdiv(ir_prefetch, &bctx->src0_dim12_div);
        uint32_t rem = ir_prefetch - i03 * (ne02 * ne01);
        uint32_t i02 = fastdiv(rem, &bctx->src0_dim1_div);
        uint32_t i01 = rem - i02 * ne01;

        dma_addr_t src0_curr = src0->معلومات + i03 * nb03 + i02 * nb02 + i01 * nb01;
        dma_addr_t dst_curr  = dst->معلومات  + i03 * nb3  + i02 * nb2  + i01 * nb1;

        uint8_t * s0_spad = src0_spad_بنسټ + spad_idx * src0_spad_half;
        uint8_t * d_spad  = dst_spad_بنسټ  + spad_idx * dst_spad_half;

        dma_کتار_ټېل(dma_q, dma_make_معلومات(dst_curr, d_spad), nb1, bctx->dst_لیکه_اندازه_aligned, لیکه_اندازه_بایټونه, 0);
        dma_کتار_ټېل(dma_q, dma_make_معلومات(s0_spad, src0_curr), bctx->src0_لیکه_اندازه_aligned, nb01, لیکه_اندازه_بایټونه, current_بلاک_اندازه);
        ir_prefetch += current_بلاک_اندازه;
        spad_idx ^= 1;
    }

    بنية htp_thread_trace * tr = &octx->ctx->trace[ith];
    compute_repeat_t compute = (compute_repeat_t) bctx->compute;

    لكل (uint32_t ir = پیل_لیکه; ir < پای_ټکی_لیکه; ) {
        uint32_t current_بلاک_اندازه = calc_بلاک_اندازه(bctx, ir, پای_ټکی_لیکه, ne01, ne02);
        uint8_t * d_spad  = (uint8_t *) dma_کتار_راوباسه(dma_q).src;
        uint8_t * s0_spad = (uint8_t *) dma_کتار_راوباسه(dma_q).dst;

        uint32_t i03 = fastdiv(ir, &bctx->src0_dim12_div);
        uint32_t rem = ir - i03 * (ne02 * ne01);
        uint32_t i02 = fastdiv(rem, &bctx->src0_dim1_div);
        uint32_t i01 = rem - i02 * ne01;

        uint32_t i13 = fastmodulo(i03, ne13, &bctx->src1_dim3_div);
        uint32_t i12 = fastmodulo(i02, ne12, &bctx->src1_dim2_div);
        ثابت uint8_t * src1_plane = vtcm_src1 + i13 * vtcm_vol_stride + i12 * vtcm_plane_stride;

        htp_trace_event_پیل(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);
        compute(d_spad, s0_spad, src1_plane, i01, ne11, &bctx->src1_dim1_div, vtcm_لیکه_stride,
                current_بلاک_اندازه, bctx->dst_لیکه_اندازه_aligned, bctx->src0_لیکه_اندازه_aligned, ne00, ne10);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);

        dma_addr_t dst_curr = dst->معلومات + i03 * nb3 + i02 * nb2 + i01 * nb1;
        dma_کتار_ټېل(dma_q, dma_make_معلومات(dst_curr, d_spad), nb1, bctx->dst_لیکه_اندازه_aligned, لیکه_اندازه_بایټونه, current_بلاک_اندازه);

        إذا (ir_prefetch < پای_ټکی_لیکه) {
            uint32_t بل_بلاک_اندازه = calc_بلاک_اندازه(bctx, ir_prefetch, پای_ټکی_لیکه, ne01, ne02);
            uint32_t p03, p02, p01, prem;
            p03  = fastdiv(ir_prefetch, &bctx->src0_dim12_div);
            prem = ir_prefetch - p03 * (ne02 * ne01);
            p02  = fastdiv(prem, &bctx->src0_dim1_div);
            p01  = prem - p02 * ne01;
            dma_addr_t s0_بل = src0->معلومات + p03 * nb03 + p02 * nb02 + p01 * nb01;
            dma_کتار_ټېل(dma_q, dma_make_معلومات(s0_spad, s0_بل), bctx->src0_لیکه_اندازه_aligned, nb01, لیکه_اندازه_بایټونه, بل_بلاک_اندازه);
            ir_prefetch += بل_بلاک_اندازه;
        }
        ir += current_بلاک_اندازه;
    }

    dma_کتار_flush(dma_q);
}

// 6. ADD_ID (src1 gathered via src2 indices)
ساكن فراغ دوګونی_thread_جمع_کړه_id_f32(بینښه صحيح nth, بینښه صحيح ith, فراغ * معلومات) {
    بنية htp_دوګونی_context * bctx = (بنية htp_دوګونی_context *) معلومات;
    بنية htp_ops_context * octx = bctx->octx;

    ثابت بنية htp_tensor * src0 = octx->src[0];
    ثابت بنية htp_tensor * src1 = octx->src[1];
    ثابت بنية htp_tensor * dst  = octx->dst;

    ثابت uint32_t ne00 = src0->ne[0];
    ثابت uint32_t ne01 = src0->ne[1];
    ثابت uint32_t ne02 = src0->ne[2];
    ثابت uint32_t ne03 = src0->ne[3];

    ثابت uint32_t nb01 = src0->nb[1];
    ثابت uint32_t nb02 = src0->nb[2];
    ثابت uint32_t nb03 = src0->nb[3];
    ثابت uint32_t src1_stride = bctx->src1_لیکه_اندازه_aligned;

    ثابت uint32_t nb1 = dst->nb[1];
    ثابت uint32_t nb2 = dst->nb[2];
    ثابت uint32_t nb3 = dst->nb[3];

    ثابت uint32_t لیکه_اندازه_بایټونه = bctx->لیکه_اندازه_بایټونه;
    ثابت uint32_t پیل_لیکه = bctx->لیکه_پیل + bctx->nrows_فی_thread * ith;
    ثابت uint32_t پای_ټکی_لیکه   = MIN(پیل_لیکه + bctx->nrows_فی_thread, bctx->لیکه_پیل + bctx->ټول_لیکې);
    إذا (پیل_لیکه >= پای_ټکی_لیکه) أرجع;

    ثابت بنية htp_دوګونی_vtcm_layout * layout = &bctx->vtcm_layout;
    uint8_t * src0_spad_بنسټ = VTCM_LAYOUT_PTR(uint8_t, bctx->vtcm_بنسټ, layout->off_src0) + (ith * layout->src0_بایټونه_فی_thread);
    uint8_t * dst_spad_بنسټ  = VTCM_LAYOUT_PTR(uint8_t, bctx->vtcm_بنسټ, layout->off_dst)  + (ith * layout->dst_بایټونه_فی_thread);
    ثابت uint8_t * vtcm_src1 = VTCM_LAYOUT_PTR(ثابت uint8_t, bctx->vtcm_بنسټ, layout->off_src1);
    ثابت int32_t * vtcm_src2 = VTCM_LAYOUT_PTR(ثابت int32_t, bctx->vtcm_بنسټ, layout->off_src2);
    size_t src0_spad_half    = layout->src0_spad_half_اندازه;
    size_t dst_spad_half     = layout->dst_spad_half_اندازه;

    dma_کتار * dma_q = octx->ctx->dma[ith];
    uint32_t ir_prefetch = پیل_لیکه;
    صحيح spad_idx = 0;

    لكل (صحيح k = 0; k < 2 && ir_prefetch < پای_ټکی_لیکه; k++) {
        uint32_t current_بلاک_اندازه = calc_بلاک_اندازه(bctx, ir_prefetch, پای_ټکی_لیکه, ne01, ne02);
        uint32_t i03 = fastdiv(ir_prefetch, &bctx->src0_dim12_div);
        uint32_t rem = ir_prefetch - i03 * (ne02 * ne01);
        uint32_t i02 = fastdiv(rem, &bctx->src0_dim1_div);
        uint32_t i01 = rem - i02 * ne01;

        dma_addr_t src0_curr = src0->معلومات + i03 * nb03 + i02 * nb02 + i01 * nb01;
        dma_addr_t dst_curr  = dst->معلومات  + i03 * nb3  + i02 * nb2  + i01 * nb1;

        uint8_t * s0_spad = src0_spad_بنسټ + spad_idx * src0_spad_half;
        uint8_t * d_spad  = dst_spad_بنسټ  + spad_idx * dst_spad_half;

        dma_کتار_ټېل(dma_q, dma_make_معلومات(dst_curr, d_spad), nb1, bctx->dst_لیکه_اندازه_aligned, لیکه_اندازه_بایټونه, 0);
        dma_کتار_ټېل(dma_q, dma_make_معلومات(s0_spad, src0_curr), bctx->src0_لیکه_اندازه_aligned, nb01, لیکه_اندازه_بایټونه, current_بلاک_اندازه);
        ir_prefetch += current_بلاک_اندازه;
        spad_idx ^= 1;
    }

    بنية htp_thread_trace * tr = &octx->ctx->trace[ith];

    لكل (uint32_t ir = پیل_لیکه; ir < پای_ټکی_لیکه; ) {
        uint32_t current_بلاک_اندازه = calc_بلاک_اندازه(bctx, ir, پای_ټکی_لیکه, ne01, ne02);
        uint8_t * d_spad = (uint8_t *) dma_کتار_راوباسه(dma_q).src;
        uint8_t * s0_spad = (uint8_t *) dma_کتار_راوباسه(dma_q).dst;

        uint32_t i03 = fastdiv(ir, &bctx->src0_dim12_div);
        uint32_t rem = ir - i03 * (ne02 * ne01);
        uint32_t i02 = fastdiv(rem, &bctx->src0_dim1_div);
        uint32_t i01 = rem - i02 * ne01;

        htp_trace_event_پیل(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);
        compute_جمع_کړه_id_t compute = (compute_جمع_کړه_id_t) bctx->compute;
        compute(d_spad, s0_spad, vtcm_src1, vtcm_src2,
                i01, i02, ne01, src1_stride,
                current_بلاک_اندازه, bctx->dst_لیکه_اندازه_aligned, bctx->src0_لیکه_اندازه_aligned, ne00);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);

        dma_addr_t dst_curr = dst->معلومات + i03 * nb3 + i02 * nb2 + i01 * nb1;
        dma_کتار_ټېل(dma_q, dma_make_معلومات(dst_curr, d_spad), nb1, bctx->dst_لیکه_اندازه_aligned, لیکه_اندازه_بایټونه, current_بلاک_اندازه);

        إذا (ir_prefetch < پای_ټکی_لیکه) {
             uint32_t بل_بلاک_اندازه = calc_بلاک_اندازه(bctx, ir_prefetch, پای_ټکی_لیکه, ne01, ne02);
             uint32_t p03, p02, p01, prem;
             p03  = fastdiv(ir_prefetch, &bctx->src0_dim12_div);
             prem = ir_prefetch - p03 * (ne02 * ne01);
             p02  = fastdiv(prem, &bctx->src0_dim1_div);
             p01  = prem - p02 * ne01;
             dma_addr_t s0_بل = src0->معلومات + p03 * nb03 + p02 * nb02 + p01 * nb01;
             dma_کتار_ټېل(dma_q, dma_make_معلومات(s0_spad, s0_بل), bctx->src0_لیکه_اندازه_aligned, nb01, لیکه_اندازه_بایټونه, بل_بلاک_اندازه);
             ir_prefetch += بل_بلاک_اندازه;
        }
        ir += current_بلاک_اندازه;
    }

    dma_کتار_flush(dma_q);
}

ساكن مضمن فراغ hvx_div_scalar_f32(uint8_t * محدود dst, ثابت uint8_t * محدود src, ثابت عائم val, ثابت uint32_t عدد_elems) {
    hvx_mul_scalar_f32(dst, src, 1.0f / val, عدد_elems);
}

ساكن مضمن فراغ hvx_div_scalar_f16(uint8_t * محدود dst, ثابت uint8_t * محدود src, ثابت _Float16 val, ثابت uint32_t عدد_elems) {
    hvx_div_scalar_f16_aa(dst, src, val, عدد_elems);
}

تعريفنوع فراغ (*compute_دوګونی_chunked_t)(
    uint8_t * محدود dst,
    ثابت uint8_t * محدود src0,
    ثابت uint8_t * محدود src1,
    ثابت uint32_t عدد_elems
);

تعريفنوع فراغ (*compute_دوګونی_scalar_chunked_f32_t)(
    uint8_t * محدود dst,
    ثابت uint8_t * محدود src0,
    ثابت عائم val,
    ثابت uint32_t عدد_elems
);

تعريفنوع فراغ (*compute_دوګونی_scalar_chunked_f16_t)(
    uint8_t * محدود dst,
    ثابت uint8_t * محدود src0,
    ثابت _Float16 val,
    ثابت uint32_t عدد_elems
);

بنية دوګونی_chunked_context {
    بنية htp_ops_context * octx;
    بنية htp_دوګونی_vtcm_layout vtcm_layout;
    uint8_t *                vtcm_بنسټ;
    uint32_t                 elem_پیل;
    uint32_t                 nelem;
    uint32_t                 chunk_اندازه;
    uint32_t                 chunks_فی_thread;
    uint32_t                 ټول_chunks;
    bool                     is_scalar;
    عائم                    scalar_f32;
    _Float16                 scalar_f16;
    compute_دوګونی_chunked_t compute;
    compute_دوګونی_scalar_chunked_f32_t compute_scalar_f32;
    compute_دوګونی_scalar_chunked_f16_t compute_scalar_f16;
};

ساكن فراغ دوګونی_thread_chunked(بینښه صحيح nth, بینښه صحيح ith, فراغ * معلومات) {
    (فراغ) nth;
    بنية دوګونی_chunked_context * ctx = (بنية دوګونی_chunked_context *) معلومات;
    بنية htp_ops_context * octx = ctx->octx;
    ثابت بنية htp_tensor * src0 = octx->src[0];
    ثابت بنية htp_tensor * src1 = octx->src[1];
    ثابت بنية htp_tensor * dst  = octx->dst;

    ثابت uint32_t پیل_chunk = ctx->chunks_فی_thread * ith;
    ثابت uint32_t پای_ټکی_chunk   = MIN(پیل_chunk + ctx->chunks_فی_thread, ctx->ټول_chunks);
    إذا (پیل_chunk >= پای_ټکی_chunk) {
        أرجع;
    }

    ثابت uint32_t src0_type   = src0->type;
    ثابت size_t   elem_اندازه   = (src0_type == HTP_TYPE_F32) ? حجم(عائم) : حجم(_Float16);
    ثابت uint32_t chunk_اندازه  = ctx->chunk_اندازه;
    ثابت size_t   chunk_بایټونه = ctx->vtcm_layout.src0_spad_half_اندازه;

    FARF(لوړ_2, "binary-chunked: %d/%d (%u:%u) chunks %u elems %u",
         ith, nth, پیل_chunk, پای_ټکی_chunk, ctx->ټول_chunks, ctx->nelem);

    ثابت بنية htp_دوګونی_vtcm_layout * layout = &ctx->vtcm_layout;
    uint8_t * src0_spad_بنسټ = VTCM_LAYOUT_PTR(uint8_t, ctx->vtcm_بنسټ, layout->off_src0) + (ith * layout->src0_بایټونه_فی_thread);
    uint8_t * src1_spad_بنسټ = ctx->is_scalar ? عدم : (VTCM_LAYOUT_PTR(uint8_t, ctx->vtcm_بنسټ, layout->off_src1) + (ith * layout->src1_بایټونه_فی_thread));
    uint8_t * dst_spad_بنسټ  = VTCM_LAYOUT_PTR(uint8_t, ctx->vtcm_بنسټ, layout->off_dst)  + (ith * layout->dst_بایټونه_فی_thread);

    dma_کتار * dma_q = octx->ctx->dma[ith];
    uint32_t prefetch_chunk = پیل_chunk;
    صحيح spad_idx = 0;

    لكل (صحيح k = 0; k < 2 && prefetch_chunk < پای_ټکی_chunk; k++) {
        ثابت uint32_t c_پیل   = ctx->elem_پیل + prefetch_chunk * chunk_اندازه;
        ثابت uint32_t c_پای_ټکی     = MIN(c_پیل + chunk_اندازه, ctx->elem_پیل + ctx->nelem);
        ثابت uint32_t cur_elems = c_پای_ټکی - c_پیل;
        ثابت uint32_t cur_بایټونه = cur_elems * elem_اندازه;

        dma_addr_t s0_curr = src0->معلومات + (size_t) c_پیل * elem_اندازه;
        dma_addr_t d_curr  = dst->معلومات  + (size_t) c_پیل * elem_اندازه;

        uint8_t * s0_spad = src0_spad_بنسټ + spad_idx * chunk_بایټونه;
        uint8_t * d_spad  = dst_spad_بنسټ  + spad_idx * chunk_بایټونه;

        dma_کتار_ټېل(dma_q, dma_make_معلومات(d_curr, d_spad), chunk_بایټونه, chunk_بایټونه, cur_بایټونه, 0);
        dma_کتار_ټېل(dma_q, dma_make_معلومات(s0_spad, s0_curr), chunk_بایټونه, chunk_بایټونه, cur_بایټونه, 1);
        إذا (!ctx->is_scalar) {
            dma_addr_t s1_curr = src1->معلومات + (size_t) c_پیل * elem_اندازه;
            uint8_t * s1_spad = src1_spad_بنسټ + spad_idx * chunk_بایټونه;
            dma_کتار_ټېل(dma_q, dma_make_معلومات(s1_spad, s1_curr), chunk_بایټونه, chunk_بایټونه, cur_بایټونه, 1);
        }

        prefetch_chunk++;
        spad_idx ^= 1;
    }

    بنية htp_thread_trace * tr = &octx->ctx->trace[ith];

    لكل (uint32_t c = پیل_chunk; c < پای_ټکی_chunk; c++) {
        ثابت uint32_t c_پیل   = ctx->elem_پیل + c * chunk_اندازه;
        ثابت uint32_t c_پای_ټکی     = MIN(c_پیل + chunk_اندازه, ctx->elem_پیل + ctx->nelem);
        ثابت uint32_t cur_elems = c_پای_ټکی - c_پیل;
        ثابت uint32_t cur_بایټونه = cur_elems * elem_اندازه;

        uint8_t * d_spad  = (uint8_t *) dma_کتار_راوباسه(dma_q).src;
        uint8_t * s0_spad = (uint8_t *) dma_کتار_راوباسه(dma_q).dst;
        uint8_t * s1_spad = ctx->is_scalar ? عدم : (uint8_t *) dma_کتار_راوباسه(dma_q).dst;

        htp_trace_event_پیل(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) c);
        إذا (ctx->is_scalar) {
            إذا (src0_type == HTP_TYPE_F32) {
                ctx->compute_scalar_f32(d_spad, s0_spad, ctx->scalar_f32, cur_elems);
            } وإلا {
                ctx->compute_scalar_f16(d_spad, s0_spad, ctx->scalar_f16, cur_elems);
            }
        } وإلا {
            ctx->compute(d_spad, s0_spad, s1_spad, cur_elems);
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) c);

        dma_addr_t dst_curr = dst->معلومات + (size_t) c_پیل * elem_اندازه;
        dma_کتار_ټېل(dma_q, dma_make_معلومات(dst_curr, d_spad), chunk_بایټونه, chunk_بایټونه, cur_بایټونه, 1);

        إذا (prefetch_chunk < پای_ټکی_chunk) {
            ثابت uint32_t pc_پیل  = ctx->elem_پیل + prefetch_chunk * chunk_اندازه;
            ثابت uint32_t pc_پای_ټکی    = MIN(pc_پیل + chunk_اندازه, ctx->elem_پیل + ctx->nelem);
            ثابت uint32_t p_elems   = pc_پای_ټکی - pc_پیل;
            ثابت uint32_t p_بایټونه   = p_elems * elem_اندازه;

            dma_addr_t s0_بل = src0->معلومات + (size_t) pc_پیل * elem_اندازه;
            dma_کتار_ټېل(dma_q, dma_make_معلومات(s0_spad, s0_بل), chunk_بایټونه, chunk_بایټونه, p_بایټونه, 1);
            إذا (!ctx->is_scalar) {
                dma_addr_t s1_بل = src1->معلومات + (size_t) pc_پیل * elem_اندازه;
                dma_کتار_ټېل(dma_q, dma_make_معلومات(s1_spad, s1_بل), chunk_بایټونه, chunk_بایټونه, p_بایټونه, 1);
            }

            prefetch_chunk++;
        }
    }

    dma_کتار_flush(dma_q);
}

ساكن صحيح execute_op_دوګونی(بنية htp_ops_context * octx) {
    ثابت بنية htp_tensor * src0 = octx->src[0];
    ثابت بنية htp_tensor * src1 = octx->src[1];
    ثابت بنية htp_tensor * dst  = octx->dst;
    ثابت بنية htp_دوګونی_kernel_params * kparams = (ثابت بنية htp_دوګونی_kernel_params *) octx->kernel_params;

    ثابت uint32_t src0_nrows = src0->ne[1] * src0->ne[2] * src0->ne[3];

    // Use packed row sizes for VTCM allocation and alignment
    ثابت uint32_t src0_type = octx->src[0]->type;
    ثابت size_t elem_اندازه = (src0_type == HTP_TYPE_F32) ? حجم(عائم) : حجم(_Float16);
    ثابت size_t src0_لیکه_اندازه = src0->ne[0] * elem_اندازه;
    ثابت size_t src1_لیکه_اندازه = src1->ne[0] * elem_اندازه;
    ثابت size_t dst_لیکه_اندازه  = dst->ne[0]  * elem_اندازه;

    إذا (kparams->kernel_type == HTP_دوګونی_KERNEL_CHUNKED) {
        ثابت bool is_scalar = kparams->is_scalar || (src1->ne[0] == 1 && src1->ne[1] == 1 && src1->ne[2] == 1 && src1->ne[3] == 1);

        compute_دوګونی_chunked_t compute = عدم;
        compute_دوګونی_scalar_chunked_f32_t compute_scalar_f32 = عدم;
        compute_دوګونی_scalar_chunked_f16_t compute_scalar_f16 = عدم;

        إذا (is_scalar) {
            إذا (src0_type == HTP_TYPE_F32) {
                بدل (octx->op) {
                    حالة HTP_OP_جمع_کړه: compute_scalar_f32 = hvx_جمع_کړه_scalar_f32; اكسر;
                    حالة HTP_OP_SUB: compute_scalar_f32 = hvx_sub_scalar_f32; اكسر;
                    حالة HTP_OP_MUL: compute_scalar_f32 = hvx_mul_scalar_f32; اكسر;
                    حالة HTP_OP_DIV: compute_scalar_f32 = hvx_div_scalar_f32; اكسر;
                    افتراضي: اكسر;
                }
            } وإلا إذا (src0_type == HTP_TYPE_F16) {
                بدل (octx->op) {
                    حالة HTP_OP_جمع_کړه: compute_scalar_f16 = hvx_جمع_کړه_scalar_f16; اكسر;
                    حالة HTP_OP_SUB: compute_scalar_f16 = hvx_sub_scalar_f16; اكسر;
                    حالة HTP_OP_MUL: compute_scalar_f16 = hvx_mul_scalar_f16; اكسر;
                    حالة HTP_OP_DIV: compute_scalar_f16 = hvx_div_scalar_f16; اكسر;
                    افتراضي: اكسر;
                }
            }
            إذا (!compute_scalar_f32 && !compute_scalar_f16) {
                أرجع HTP_حالت_NO_SUPPORT;
            }
        } وإلا {
            إذا (src0_type == HTP_TYPE_F32) {
                بدل (octx->op) {
                    حالة HTP_OP_جمع_کړه: compute = hvx_جمع_کړه_f32; اكسر;
                    حالة HTP_OP_SUB: compute = hvx_sub_f32; اكسر;
                    حالة HTP_OP_MUL: compute = hvx_mul_f32; اكسر;
                    حالة HTP_OP_DIV: compute = hvx_div_f32; اكسر;
                    افتراضي: اكسر;
                }
            } وإلا إذا (src0_type == HTP_TYPE_F16) {
                بدل (octx->op) {
                    حالة HTP_OP_جمع_کړه: compute = hvx_جمع_کړه_f16; اكسر;
                    حالة HTP_OP_SUB: compute = hvx_sub_f16; اكسر;
                    حالة HTP_OP_MUL: compute = hvx_mul_f16; اكسر;
                    حالة HTP_OP_DIV: compute = hvx_div_f16; اكسر;
                    افتراضي: اكسر;
                }
            }
            إذا (!compute) {
                أرجع HTP_حالت_NO_SUPPORT;
            }
        }

        ثابت uint32_t ټول_elems = (uint32_t) (src0->ne[0] * src0->ne[1] * src0->ne[2] * src0->ne[3]);
        إذا (ټول_elems == 0) {
            أرجع HTP_حالت_OK;
        }

        uint32_t elem_پیل = 0;
        uint32_t nelem      = ټول_elems;
        ثابت uint32_t elems_فی_کرښه = (src0_type == HTP_TYPE_F32) ? 32 : 64;

        إذا (octx->ctx->mdev.شمېر > 1) {
            ثابت bool can_وېشل = htp_tensor_mdev_معلومات_aligned(dst) &&
                                   htp_tensor_mdev_معلومات_aligned(src0) &&
                                   (is_scalar || htp_tensor_mdev_معلومات_aligned(src1)) &&
                                   htp_tensor_is_contiguous(dst, elem_اندازه) &&
                                   htp_tensor_is_contiguous(src0, elem_اندازه) &&
                                   (is_scalar || htp_tensor_is_contiguous(src1, elem_اندازه));
            ثابت بنية htp_tensor_mdev_range range = htp_tensor_mdev_برخه_کول(
                ټول_elems, can_وېشل ? elems_فی_کرښه : 0,
                octx->ctx->mdev.idx, octx->ctx->mdev.شمېر, &octx->ctx->mdev.شمېر_div);
            elem_پیل = range.پیل;
            nelem      = range.شمېر;
        }

        إذا (nelem == 0) {
            أرجع HTP_حالت_OK;
        }

        إذا (!htp_ops_context_سېټ_n_تارونه(octx, kparams->n_تارونه)) {
            أرجع HTP_حالت_INVAL_PARAMS;
        }

        بنية htp_دوګونی_vtcm_layout vtcm_layout;
        htp_دوګونی_vtcm_layout_build(&vtcm_layout, kparams, octx->ctx->vtcm_اندازه);
        إذا (vtcm_layout.ټول_بایټونه == 0 || vtcm_layout.ټول_بایټونه > octx->ctx->vtcm_اندازه) {
            أرجع HTP_حالت_VTCM_TOO_SMALL;
        }

        ثابت uint32_t chunk_اندازه = kparams->chunk_اندازه > 0 ? kparams->chunk_اندازه : (32768 / elem_اندازه);
        ثابت uint32_t ټول_chunks = (nelem + chunk_اندازه - 1) / chunk_اندازه;
        ثابت uint32_t n_تارونه = (ټول_chunks >= 2) ? MIN(octx->n_تارونه, ټول_chunks) : 1;
        ثابت uint32_t chunks_فی_thread = (ټول_chunks + n_تارونه - 1) / n_تارونه;

        عائم scalar_f32 = 0.0f;
        _Float16 scalar_f16 = 0;
        إذا (is_scalar) {
            uint8_t * vtcm_src1 = VTCM_LAYOUT_PTR(uint8_t, octx->ctx->vtcm_بنسټ, vtcm_layout.off_src1);
            dma_کتار * dma_q = octx->ctx->dma[0];
            dma_کتار_ټېل(dma_q, dma_make_معلومات(vtcm_src1, src1->معلومات), 128, 0, elem_اندازه, 1);
            dma_کتار_راوباسه(dma_q);

            إذا (src0_type == HTP_TYPE_F32) {
                scalar_f32 = ((ثابت عائم *) vtcm_src1)[0];
            } وإلا {
                scalar_f16 = ((ثابت _Float16 *) vtcm_src1)[0];
            }
        }

        بنية دوګونی_chunked_context cctx = {
            .octx                = octx,
            .vtcm_layout         = vtcm_layout,
            .vtcm_بنسټ           = (uint8_t *) octx->ctx->vtcm_بنسټ,
            .elem_پیل          = elem_پیل,
            .nelem               = nelem,
            .chunk_اندازه          = chunk_اندازه,
            .chunks_فی_thread   = chunks_فی_thread,
            .ټول_chunks        = ټول_chunks,
            .is_scalar           = is_scalar,
            .scalar_f32          = scalar_f32,
            .scalar_f16          = scalar_f16,
            .compute             = compute,
            .compute_scalar_f32  = compute_scalar_f32,
            .compute_scalar_f16  = compute_scalar_f16,
        };

        work_کتار_پرلپسې(octx->ctx->work_کتار, دوګونی_thread_chunked, &cctx, n_تارونه);
        أرجع HTP_حالت_OK;
    }

    uint32_t لیکه_پیل = 0;
    uint32_t nrows     = src0_nrows;

    إذا (octx->ctx->mdev.شمېر > 1) {
        uint32_t لیکې_فی_chunk = 0;
        htp_tensor_mdev_لیکې_فی_chunk(dst, (uint32_t) elem_اندازه, (uint32_t) dst_لیکه_اندازه, &لیکې_فی_chunk);
        ثابت بنية htp_tensor_mdev_range range = htp_tensor_mdev_برخه_کول(src0_nrows, لیکې_فی_chunk,
                                                       octx->ctx->mdev.idx, octx->ctx->mdev.شمېر, &octx->ctx->mdev.شمېر_div);
        لیکه_پیل = range.پیل;
        nrows     = range.شمېر;
    }

    إذا (nrows == 0) {
        أرجع HTP_حالت_OK;
    }

    إذا (!htp_ops_context_سېټ_n_تارونه(octx, kparams->n_تارونه)) {
        أرجع HTP_حالت_INVAL_PARAMS;
    }

    ثابت uint32_t n_تارونه = octx->n_تارونه;
    ثابت size_t src0_لیکه_اندازه_aligned = kparams->src0_لیکه_اندازه_aligned;
    ثابت size_t src1_لیکه_اندازه_aligned = kparams->src1_لیکه_اندازه_aligned;
    ثابت size_t dst_لیکه_اندازه_aligned  = kparams->dst_لیکه_اندازه_aligned;

    بنية htp_دوګونی_context bctx;
    bctx.vtcm_بنسټ = (uint8_t *) octx->ctx->vtcm_بنسټ;
    htp_دوګونی_vtcm_layout_build(&bctx.vtcm_layout, kparams, octx->ctx->vtcm_اندازه);

    إذا (bctx.vtcm_layout.لیکې_فی_بفر == 0 || bctx.vtcm_layout.ټول_بایټونه > octx->ctx->vtcm_اندازه) {
        أرجع HTP_حالت_VTCM_TOO_SMALL;
    }

    dma_کتار * dma_q = octx->ctx->dma[0];
    uint8_t * vtcm_src1 = VTCM_LAYOUT_PTR(uint8_t, bctx.vtcm_بنسټ, bctx.vtcm_layout.off_src1);
    إذا (kparams->kernel_type == HTP_دوګونی_KERNEL_لیکه_BCAST) {
        dma_کتار_ټېل(dma_q, dma_make_معلومات(vtcm_src1, src1->معلومات), bctx.vtcm_layout.src1_اندازه, 0, src1->ne[0] * elem_اندازه, 1);
    } وإلا إذا (kparams->kernel_type == HTP_دوګونی_KERNEL_SCALAR ||
               kparams->kernel_type == HTP_دوګونی_KERNEL_REPEAT ||
               kparams->kernel_type == HTP_دوګونی_KERNEL_COMPLEX) {
        إذا (!kparams->src1_strided) {
            ثابت size_t src1_بایټونه = (size_t) src1->ne[0] * src1->ne[1] * src1->ne[2] * src1->ne[3] * elem_اندازه;
            dma_کتار_ټېل(dma_q, dma_make_معلومات(vtcm_src1, src1->معلومات), bctx.vtcm_layout.src1_اندازه, 0, src1_بایټونه, 1);
        } وإلا {
            ثابت uint32_t ne11 = src1->ne[1];
            ثابت uint32_t ne12 = src1->ne[2];
            ثابت uint32_t ne13 = src1->ne[3];
            ثابت size_t لیکه_بایټونه = src1->ne[0] * elem_اندازه;
            لكل (uint32_t i13 = 0; i13 < ne13; i13++) {
                لكل (uint32_t i12 = 0; i12 < ne12; i12++) {
                    dma_addr_t src_addr = src1->معلومات + i13 * src1->nb[3] + i12 * src1->nb[2];
                    uint8_t * dst_ptr = vtcm_src1 + (i13 * ne12 * ne11 + i12 * ne11) * لیکه_بایټونه;
                    طالما (!dma_کتار_ټېل(dma_q, dma_make_معلومات(dst_ptr, src_addr),
                                           لیکه_بایټونه, src1->nb[1], لیکه_بایټونه, ne11)) {
                        dma_کتار_راوباسه(dma_q);
                    }
                }
            }
        }
    } وإلا إذا (kparams->kernel_type == HTP_دوګونی_KERNEL_جمع_کړه_ID) {
        ثابت بنية htp_tensor * src2 = octx->src[2];
        uint8_t * vtcm_src2 = VTCM_LAYOUT_PTR(uint8_t, bctx.vtcm_بنسټ, bctx.vtcm_layout.off_src2);
        dma_کتار_ټېل(dma_q, dma_make_معلومات(vtcm_src1, src1->معلومات),
                       kparams->src1_لیکه_اندازه_aligned, src1->nb[1],
                       src1->ne[0] * elem_اندازه, src1->ne[1]);
        ثابت size_t src2_لیکه_بایټونه = src2->ne[0] * حجم(int32_t);
        dma_کتار_ټېل(dma_q, dma_make_معلومات(vtcm_src2, src2->معلومات),
                       src2_لیکه_بایټونه, src2->nb[1],
                       src2_لیکه_بایټونه, src2->ne[1]);
    }

    bctx.octx                  = octx;
    bctx.nrows_فی_thread      = fastdiv(nrows + n_تارونه - 1, &octx->n_تارونه_div);
    bctx.ټول_لیکې            = nrows;
    bctx.لیکه_پیل             = لیکه_پیل;
    bctx.بلاک_max             = bctx.vtcm_layout.لیکې_فی_بفر;
    bctx.src0_لیکه_اندازه_aligned = src0_لیکه_اندازه_aligned;
    bctx.src1_لیکه_اندازه_aligned = src1_لیکه_اندازه_aligned;
    bctx.dst_لیکه_اندازه_aligned  = dst_لیکه_اندازه_aligned;
    bctx.لیکه_اندازه_بایټونه        = src0_لیکه_اندازه;

    bctx.src0_dim1_div  = init_fastdiv_ارزښتونه(src0->ne[1]);
    bctx.src0_dim2_div  = init_fastdiv_ارزښتونه(src0->ne[2]);
    bctx.src0_dim12_div = init_fastdiv_ارزښتونه(src0->ne[1] * src0->ne[2]);

    bctx.src1_dim1_div  = init_fastdiv_ارزښتونه(src1->ne[1]);
    bctx.src1_dim2_div  = init_fastdiv_ارزښتونه(src1->ne[2]);
    bctx.src1_dim3_div  = init_fastdiv_ارزښتونه(src1->ne[3]);

    bctx.وېشل_at_ne01  = kparams->وېشل_at_ne01;
    bctx.وېشل_at_ne02  = kparams->وېشل_at_ne02;

    کارګر_callback_t کارګر_func = عدم;
    فراغ * compute_func = عدم;

    بدل (kparams->kernel_type) {
        حالة HTP_دوګونی_KERNEL_SAME_SHAPE:
            کارګر_func = دوګونی_thread_ویکټور_same_shape;
            إذا (src0_type == HTP_TYPE_F32) {
                بدل (octx->op) {
                    حالة HTP_OP_جمع_کړه: compute_func = compute_same_shape_جمع_کړه_f32; اكسر;
                    حالة HTP_OP_SUB: compute_func = compute_same_shape_sub_f32; اكسر;
                    حالة HTP_OP_MUL: compute_func = compute_same_shape_mul_f32; اكسر;
                    حالة HTP_OP_DIV: compute_func = compute_same_shape_div_f32; اكسر;
                    افتراضي: اكسر;
                }
            } وإلا إذا (src0_type == HTP_TYPE_F16) {
                بدل (octx->op) {
                    حالة HTP_OP_جمع_کړه: compute_func = compute_same_shape_جمع_کړه_f16; اكسر;
                    حالة HTP_OP_SUB: compute_func = compute_same_shape_sub_f16; اكسر;
                    حالة HTP_OP_MUL: compute_func = compute_same_shape_mul_f16; اكسر;
                    حالة HTP_OP_DIV: compute_func = compute_same_shape_div_f16; اكسر;
                    افتراضي: اكسر;
                }
            }
            اكسر;
        حالة HTP_دوګونی_KERNEL_لیکه_BCAST:
            کارګر_func = دوګونی_thread_ویکټور_لیکه_خپرونه;
            إذا (src0_type == HTP_TYPE_F32) {
                بدل (octx->op) {
                    حالة HTP_OP_جمع_کړه: compute_func = compute_لیکه_bcast_جمع_کړه_f32; اكسر;
                    حالة HTP_OP_SUB: compute_func = compute_لیکه_bcast_sub_f32; اكسر;
                    حالة HTP_OP_MUL: compute_func = compute_لیکه_bcast_mul_f32; اكسر;
                    حالة HTP_OP_DIV: compute_func = compute_لیکه_bcast_div_f32; اكسر;
                    افتراضي: اكسر;
                }
            } وإلا إذا (src0_type == HTP_TYPE_F16) {
                بدل (octx->op) {
                    حالة HTP_OP_جمع_کړه: compute_func = compute_لیکه_bcast_جمع_کړه_f16; اكسر;
                    حالة HTP_OP_SUB: compute_func = compute_لیکه_bcast_sub_f16; اكسر;
                    حالة HTP_OP_MUL: compute_func = compute_لیکه_bcast_mul_f16; اكسر;
                    حالة HTP_OP_DIV: compute_func = compute_لیکه_bcast_div_f16; اكسر;
                    افتراضي: اكسر;
                }
            }
            اكسر;
        حالة HTP_دوګونی_KERNEL_SCALAR:
            کارګر_func = دوګونی_thread_scalar;
            إذا (src0_type == HTP_TYPE_F32) {
                بدل (octx->op) {
                    حالة HTP_OP_جمع_کړه: compute_func = compute_scalar_جمع_کړه_f32; اكسر;
                    حالة HTP_OP_SUB: compute_func = compute_scalar_sub_f32; اكسر;
                    حالة HTP_OP_MUL: compute_func = compute_scalar_mul_f32; اكسر;
                    حالة HTP_OP_DIV: compute_func = compute_scalar_div_f32; اكسر;
                    افتراضي: اكسر;
                }
            } وإلا إذا (src0_type == HTP_TYPE_F16) {
                بدل (octx->op) {
                    حالة HTP_OP_جمع_کړه: compute_func = compute_scalar_جمع_کړه_f16; اكسر;
                    حالة HTP_OP_SUB: compute_func = compute_scalar_sub_f16; اكسر;
                    حالة HTP_OP_MUL: compute_func = compute_scalar_mul_f16; اكسر;
                    حالة HTP_OP_DIV: compute_func = compute_scalar_div_f16; اكسر;
                    افتراضي: اكسر;
                }
            }
            اكسر;
        حالة HTP_دوګونی_KERNEL_COMPLEX:
            کارګر_func = دوګونی_thread_ویکټور_complex;
            إذا (src0_type == HTP_TYPE_F32) {
                بدل (octx->op) {
                    حالة HTP_OP_جمع_کړه: compute_func = compute_complex_جمع_کړه_f32; اكسر;
                    حالة HTP_OP_SUB: compute_func = compute_complex_sub_f32; اكسر;
                    حالة HTP_OP_MUL: compute_func = compute_complex_mul_f32; اكسر;
                    حالة HTP_OP_DIV: compute_func = compute_complex_div_f32; اكسر;
                    افتراضي: اكسر;
                }
            } وإلا إذا (src0_type == HTP_TYPE_F16) {
                بدل (octx->op) {
                    حالة HTP_OP_جمع_کړه: compute_func = compute_complex_جمع_کړه_f16; اكسر;
                    حالة HTP_OP_SUB: compute_func = compute_complex_sub_f16; اكسر;
                    حالة HTP_OP_MUL: compute_func = compute_complex_mul_f16; اكسر;
                    حالة HTP_OP_DIV: compute_func = compute_complex_div_f16; اكسر;
                    افتراضي: اكسر;
                }
            }
            اكسر;
        حالة HTP_دوګونی_KERNEL_REPEAT:
            کارګر_func = دوګونی_thread_element_repeat;
            إذا (src0_type == HTP_TYPE_F32) {
                بدل (octx->op) {
                    حالة HTP_OP_جمع_کړه: compute_func = compute_repeat_جمع_کړه_f32; اكسر;
                    حالة HTP_OP_SUB: compute_func = compute_repeat_sub_f32; اكسر;
                    حالة HTP_OP_MUL: compute_func = compute_repeat_mul_f32; اكسر;
                    حالة HTP_OP_DIV: compute_func = compute_repeat_div_f32; اكسر;
                    افتراضي: اكسر;
                }
            } وإلا إذا (src0_type == HTP_TYPE_F16) {
                بدل (octx->op) {
                    حالة HTP_OP_جمع_کړه: compute_func = compute_repeat_جمع_کړه_f16; اكسر;
                    حالة HTP_OP_SUB: compute_func = compute_repeat_sub_f16; اكسر;
                    حالة HTP_OP_MUL: compute_func = compute_repeat_mul_f16; اكسر;
                    حالة HTP_OP_DIV: compute_func = compute_repeat_div_f16; اكسر;
                    افتراضي: اكسر;
                }
            }
            اكسر;
        حالة HTP_دوګونی_KERNEL_جمع_کړه_ID:
            إذا (octx->op == HTP_OP_جمع_کړه_ID && src0_type == HTP_TYPE_F32) {
                کارګر_func = دوګونی_thread_جمع_کړه_id_f32;
                compute_func = (فراغ *) compute_جمع_کړه_id_f32;
            }
            اكسر;
        افتراضي: اكسر;
    }

    إذا (!کارګر_func || !compute_func) {
        أرجع HTP_حالت_NO_SUPPORT;
    }

    bctx.compute = compute_func;

    إذا (kparams->kernel_type == HTP_دوګونی_KERNEL_لیکه_BCAST ||
        kparams->kernel_type == HTP_دوګونی_KERNEL_SCALAR ||
        kparams->kernel_type == HTP_دوګونی_KERNEL_REPEAT ||
        kparams->kernel_type == HTP_دوګونی_KERNEL_COMPLEX ||
        kparams->kernel_type == HTP_دوګونی_KERNEL_جمع_کړه_ID) {
        dma_کتار_flush(dma_q);
    }

    work_کتار_پرلپسې(octx->ctx->work_کتار, کارګر_func, &bctx, n_تارونه);

    أرجع HTP_حالت_OK;
}

صحيح op_دوګونی(بنية htp_ops_context * octx) {
    ثابت uint32_t src0_type = octx->src[0]->type;
    إذا ((src0_type == HTP_TYPE_F32) || (src0_type == HTP_TYPE_F16)) {
        أرجع execute_op_دوګونی(octx);
    }

    أرجع HTP_حالت_NO_SUPPORT;
}
