#رهنمود clang diagnostic ignored "-Wunused-variable"
#رهنمود clang diagnostic ignored "-Wunused-function"
#رهنمود clang diagnostic ignored "-Wunused-but-set-variable"

#ضمن <HAP_farf.h>
#ضمن <HAP_perf.h>

#ضمن <math.h>
#ضمن <string.h>

#ضمن "dma-queue.h"
#ضمن "hvx-utils.h"

#تعریف_کن GGML_COMMON_DECL_C
#ضمن "ggml-common.h"
#ضمن "hex-common.h"
#ضمن "hex-profile.h"
#ضمن "binary-ops.h"
#ضمن "htp-ctx.h"
#ضمن "htp-ops.h"
#ضمن "htp-tensor.h"

#اگر_تعریف_نشده MIN
#تعریف_کن MIN(a, b) ((a) < (b) ? (a) : (b))
#پایان_اگر

// Context for binary operations
بنية htp_دودویی_context {
    بنية htp_ops_context * octx;
    بنية htp_دودویی_vtcm_layout vtcm_layout;
    uint8_t * vtcm_پایه;

    بنية fastdiv_مقادیر src0_dim1_div; // ne01
    بنية fastdiv_مقادیر src0_dim2_div; // ne02
    بنية fastdiv_مقادیر src0_dim12_div;// ne03

    بنية fastdiv_مقادیر src1_dim1_div; // ne11
    بنية fastdiv_مقادیر src1_dim2_div; // ne12
    بنية fastdiv_مقادیر src1_dim3_div; // ne13

    uint32_t بلوک_max;
    uint32_t nrows_در_هر_thread;
    uint32_t مجموع_ردیفها;
    uint32_t ردیف_شروع;
    size_t   src0_ردیف_اندازه_aligned;
    size_t   src1_ردیف_اندازه_aligned;
    size_t   dst_ردیف_اندازه_aligned;
    size_t   ردیف_اندازه_بایتها;

    bool تقسیم_at_ne01;
    bool تقسیم_at_ne02;

    فراغ * compute;
};

#تعریف_کن htp_دودویی_preamble                        \
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

ساكن مضمن uint32_t calc_بلوک_اندازه(بنية htp_دودویی_context * bctx, uint32_t ir, uint32_t انتها_ردیف, uint32_t ne01, uint32_t ne02) {
    uint32_t i03, i02, i01, rem;
    i03 = fastdiv(ir, &bctx->src0_dim12_div);
    rem = ir - i03 * (ne02 * ne01);
    i02 = fastdiv(rem, &bctx->src0_dim1_div);
    i01 = rem - i02 * ne01;

    uint32_t ردیفها_چپ = انتها_ردیف - ir;
    uint32_t بلوک_حد = ردیفها_چپ;

    إذا (bctx->تقسیم_at_ne01) {
        بلوک_حد = MIN(بلوک_حد, ne01 - i01);
    }
    إذا (bctx->تقسیم_at_ne02) {
         uint32_t ردیفها_درون_plane = (ne02 * ne01) - rem;
         بلوک_حد = MIN(بلوک_حد, ردیفها_درون_plane);
    }

    أرجع MIN(bctx->بلوک_max, بلوک_حد);
}

// Out-of-line compute micro-kernels

تعريفنوع فراغ (*compute_scalar_t)(
    uint8_t * dst, ثابت uint8_t * src0, ثابت فراغ * s1_جدول,
    uint32_t cur_i11, uint32_t ne11, uint32_t n_ردیفها,
    size_t dst_stride, size_t src0_stride, uint32_t ne00);

#تعریف_کن DEFINE_COMPUTE_SCALAR(NAME, TYPE, HVX_STMT)             \
ساكن فراغ compute_scalar_##NAME(                              \
    uint8_t * dst, ثابت uint8_t * src0, ثابت فراغ * s1_جدول, \
    uint32_t cur_i11, uint32_t ne11, uint32_t n_ردیفها,           \
    size_t dst_stride, size_t src0_stride, uint32_t ne00) {     \
    ثابت TYPE * جدول = (ثابت TYPE *) s1_جدول;               \
    لكل (uint32_t r = 0; r < n_ردیفها; r++) {                     \
        uint8_t * r_dst = dst + r * dst_stride;                 \
        ثابت uint8_t * r_src0 = src0 + r * src0_stride;        \
        TYPE val = جدول[cur_i11];                              \
        HVX_STMT;                                               \
        إذا (ne11 > 1 && ++cur_i11 == ne11) {                    \
            cur_i11 = 0;                                        \
        }                                                       \
    }                                                           \
}

DEFINE_COMPUTE_SCALAR(جمع_کن_f32, عائم,    hvx_جمع_کن_scalar_f32_aa(r_dst, r_src0, val, ne00))
DEFINE_COMPUTE_SCALAR(جمع_کن_f16, _Float16, hvx_جمع_کن_scalar_f16_aa(r_dst, r_src0, val, ne00))
DEFINE_COMPUTE_SCALAR(sub_f32, عائم,    hvx_sub_scalar_f32_aa(r_dst, r_src0, val, ne00))
DEFINE_COMPUTE_SCALAR(sub_f16, _Float16, hvx_sub_scalar_f16_aa(r_dst, r_src0, val, ne00))
DEFINE_COMPUTE_SCALAR(mul_f32, عائم,    hvx_mul_scalar_f32_aa(r_dst, r_src0, val, ne00))
DEFINE_COMPUTE_SCALAR(mul_f16, _Float16, hvx_mul_scalar_f16_aa(r_dst, r_src0, val, ne00))
DEFINE_COMPUTE_SCALAR(div_f32, عائم,    hvx_mul_scalar_f32_aa(r_dst, r_src0, 1.0f / (val), ne00))
DEFINE_COMPUTE_SCALAR(div_f16, _Float16, hvx_div_scalar_f16_aa(r_dst, r_src0, val, ne00))

تعريفنوع فراغ (*compute_same_shape_t)(
    uint8_t * dst, ثابت uint8_t * src0, ثابت uint8_t * src1,
    uint32_t n_ردیفها, size_t dst_stride, size_t src0_stride, size_t src1_stride, uint32_t ne00);

#تعریف_کن DEFINE_COMPUTE_SAME_SHAPE(NAME, HVX_FN)                                                  \
ساكن فراغ compute_same_shape_##NAME(                                                           \
    uint8_t * dst, ثابت uint8_t * src0, ثابت uint8_t * src1,                                   \
    uint32_t n_ردیفها, size_t dst_stride, size_t src0_stride, size_t src1_stride, uint32_t ne00) { \
    لكل (uint32_t r = 0; r < n_ردیفها; r++) {                                                      \
        HVX_FN(dst + r * dst_stride, src0 + r * src0_stride, src1 + r * src1_stride, ne00);      \
    }                                                                                            \
}

DEFINE_COMPUTE_SAME_SHAPE(جمع_کن_f32, hvx_جمع_کن_f32_aaa)
DEFINE_COMPUTE_SAME_SHAPE(جمع_کن_f16, hvx_جمع_کن_f16_aaa)
DEFINE_COMPUTE_SAME_SHAPE(sub_f32, hvx_sub_f32_aaa)
DEFINE_COMPUTE_SAME_SHAPE(sub_f16, hvx_sub_f16_aaa)
DEFINE_COMPUTE_SAME_SHAPE(mul_f32, hvx_mul_f32_aaa)
DEFINE_COMPUTE_SAME_SHAPE(mul_f16, hvx_mul_f16_aaa)
DEFINE_COMPUTE_SAME_SHAPE(div_f32, hvx_div_f32_aaa)
DEFINE_COMPUTE_SAME_SHAPE(div_f16, hvx_div_f16_aaa)

تعريفنوع فراغ (*compute_ردیف_bcast_t)(
    uint8_t * dst, ثابت uint8_t * src0, ثابت uint8_t * src1,
    uint32_t n_ردیفها, size_t dst_stride, size_t src0_stride, uint32_t ne00);

#تعریف_کن DEFINE_COMPUTE_ردیف_BCAST(NAME, HVX_FN)                                               \
ساكن فراغ compute_ردیف_bcast##NAME(                                                        \
    uint8_t * dst, ثابت uint8_t * src0, ثابت uint8_t * src1,                               \
    uint32_t n_ردیفها, size_t dst_stride, size_t src0_stride, uint32_t ne00) {                 \
    لكل (uint32_t r = 0; r < n_ردیفها; r++) {                                                  \
        HVX_FN(dst + r * dst_stride, src0 + r * src0_stride, src1, ne00);                    \
    }                                                                                        \
}

DEFINE_COMPUTE_ردیف_BCAST(جمع_کن_f32, hvx_جمع_کن_f32_aaa)
DEFINE_COMPUTE_ردیف_BCAST(جمع_کن_f16, hvx_جمع_کن_f16_aaa)
DEFINE_COMPUTE_ردیف_BCAST(sub_f32, hvx_sub_f32_aaa)
DEFINE_COMPUTE_ردیف_BCAST(sub_f16, hvx_sub_f16_aaa)
DEFINE_COMPUTE_ردیف_BCAST(mul_f32, hvx_mul_f32_aaa)
DEFINE_COMPUTE_ردیف_BCAST(mul_f16, hvx_mul_f16_aaa)
DEFINE_COMPUTE_ردیف_BCAST(div_f32, hvx_div_f32_aaa)
DEFINE_COMPUTE_ردیف_BCAST(div_f16, hvx_div_f16_aaa)

تعريفنوع فراغ (*compute_complex_t)(
    uint8_t * dst, ثابت uint8_t * src0, ثابت uint8_t * src1_plane,
    uint32_t i01, uint32_t ne11, ثابت بنية fastdiv_مقادیر * div11, uint32_t src1_stride,
    uint32_t n_ردیفها, size_t dst_stride, size_t src0_stride, uint32_t ne00);

#تعریف_کن DEFINE_COMPUTE_COMPLEX(NAME, HVX_FN)                                                 \
ساكن فراغ compute_complex_##NAME(                                                          \
    uint8_t * dst, ثابت uint8_t * src0, ثابت uint8_t * src1_plane,                         \
    uint32_t i01, uint32_t ne11, ثابت بنية fastdiv_مقادیر * div11, uint32_t src1_stride,  \
    uint32_t n_ردیفها, size_t dst_stride, size_t src0_stride, uint32_t ne00) {                 \
    لكل (uint32_t r = 0; r < n_ردیفها; r++) {                                                  \
        uint32_t i11 = fastmodulo(i01 + r, ne11, div11);                                     \
        ثابت uint8_t * r_src1 = src1_plane + i11 * src1_stride;                             \
        HVX_FN(dst + r * dst_stride, src0 + r * src0_stride, r_src1, ne00);                  \
    }                                                                                        \
}

DEFINE_COMPUTE_COMPLEX(جمع_کن_f32, hvx_جمع_کن_f32_aau)
DEFINE_COMPUTE_COMPLEX(جمع_کن_f16, hvx_جمع_کن_f16_aau)
DEFINE_COMPUTE_COMPLEX(sub_f32, hvx_sub_f32_aau)
DEFINE_COMPUTE_COMPLEX(sub_f16, hvx_sub_f16_aau)
DEFINE_COMPUTE_COMPLEX(mul_f32, hvx_mul_f32_aau)
DEFINE_COMPUTE_COMPLEX(mul_f16, hvx_mul_f16_aau)
DEFINE_COMPUTE_COMPLEX(div_f32, hvx_div_f32_aau)
DEFINE_COMPUTE_COMPLEX(div_f16, hvx_div_f16_aau)

تعريفنوع فراغ (*compute_repeat_t)(
    uint8_t * dst, ثابت uint8_t * src0, ثابت uint8_t * src1_plane,
    uint32_t i01, uint32_t ne11, ثابت بنية fastdiv_مقادیر * div11, uint32_t src1_stride,
    uint32_t n_ردیفها, size_t dst_stride, size_t src0_stride, uint32_t ne00, uint32_t ne10);

#تعریف_کن DEFINE_COMPUTE_REPEAT(NAME, TYPE, HVX_FN)                                            \
ساكن فراغ compute_repeat_##NAME(                                                           \
    uint8_t * dst, ثابت uint8_t * src0, ثابت uint8_t * src1_plane,                         \
    uint32_t i01, uint32_t ne11, ثابت بنية fastdiv_مقادیر * div11, uint32_t src1_stride,  \
    uint32_t n_ردیفها, size_t dst_stride, size_t src0_stride, uint32_t ne00, uint32_t ne10) {  \
    لكل (uint32_t r = 0; r < n_ردیفها; r++) {                                                  \
        uint32_t i11 = fastmodulo(i01 + r, ne11, div11);                                     \
        ثابت uint8_t * r_src1_ردیف = src1_plane + i11 * src1_stride;                         \
        uint8_t * r_dst = dst + r * dst_stride;                                              \
        ثابت uint8_t * r_src0 = src0 + r * src0_stride;                                     \
        لكل (uint32_t c = 0; c < ne00; c += ne10) {                                          \
            uint32_t len = MIN(ne10, ne00 - c);                                              \
            HVX_FN(r_dst + c * حجم(TYPE), r_src0 + c * حجم(TYPE), r_src1_ردیف, len);    \
        }                                                                                    \
    }                                                                                        \
}

DEFINE_COMPUTE_REPEAT(جمع_کن_f32, عائم,    hvx_جمع_کن_f32_uuu)
DEFINE_COMPUTE_REPEAT(جمع_کن_f16, _Float16, hvx_جمع_کن_f16_uuu)
DEFINE_COMPUTE_REPEAT(sub_f32, عائم,    hvx_sub_f32_uuu)
DEFINE_COMPUTE_REPEAT(sub_f16, _Float16, hvx_sub_f16_uuu)
DEFINE_COMPUTE_REPEAT(mul_f32, عائم,    hvx_mul_f32_uuu)
DEFINE_COMPUTE_REPEAT(mul_f16, _Float16, hvx_mul_f16_uuu)
DEFINE_COMPUTE_REPEAT(div_f32, عائم,    hvx_div_f32_uuu)
DEFINE_COMPUTE_REPEAT(div_f16, _Float16, hvx_div_f16_uuu)

تعريفنوع فراغ (*compute_جمع_کن_id_t)(
    uint8_t * dst, ثابت uint8_t * src0, ثابت uint8_t * src1_داده, ثابت int32_t * src2_ids,
    uint32_t i01, uint32_t i02, uint32_t ne01, uint32_t src1_stride,
    uint32_t n_ردیفها, size_t dst_stride, size_t src0_stride, uint32_t ne00);

ساكن فراغ compute_جمع_کن_id_f32(
    uint8_t * dst, ثابت uint8_t * src0, ثابت uint8_t * src1_داده, ثابت int32_t * src2_ids,
    uint32_t i01, uint32_t i02, uint32_t ne01, uint32_t src1_stride,
    uint32_t n_ردیفها, size_t dst_stride, size_t src0_stride, uint32_t ne00) {
    لكل (uint32_t r = 0; r < n_ردیفها; r++) {
        uint32_t r_i01 = i01 + r;
        ثابت int32_t idx = src2_ids[i02 * ne01 + r_i01];
        إذا (idx < 0) {
            کپی_حافظه(dst + r * dst_stride, src0 + r * src0_stride, ne00 * حجم(عائم));
            استمر;
        }
        ثابت uint8_t * r_src1 = src1_داده + idx * src1_stride;
        ثابت uint8_t * r_src0 = src0 + r * src0_stride;
        uint8_t * r_dst = dst + r * dst_stride;
        hvx_جمع_کن_f32_aaa(r_dst, r_src0, r_src1, ne00);
    }
}

// 1. Scalar src1 in VTCM via DMA (ne10 == 1)
ساكن فراغ دودویی_thread_scalar(بیعلامت صحيح nth, بیعلامت صحيح ith, فراغ * داده) {
    بنية htp_دودویی_context * bctx = (بنية htp_دودویی_context *) داده;
    بنية htp_ops_context * octx = bctx->octx;
    htp_دودویی_preamble;

    ثابت uint32_t elem_اندازه = (src0->type == HTP_TYPE_F32) ? حجم(عائم) : حجم(_Float16);
    ثابت size_t plane_stride = (size_t) ne12 * ne11 * elem_اندازه;
    ثابت size_t ردیف_stride   = (size_t) ne11 * elem_اندازه;

    ثابت uint32_t ردیف_اندازه_بایتها = bctx->ردیف_اندازه_بایتها;
    ثابت uint32_t شروع_ردیف = bctx->ردیف_شروع + bctx->nrows_در_هر_thread * ith;
    ثابت uint32_t انتها_ردیف   = MIN(شروع_ردیف + bctx->nrows_در_هر_thread, bctx->ردیف_شروع + bctx->مجموع_ردیفها);
    إذا (شروع_ردیف >= انتها_ردیف) أرجع;

    FARF(بالا_2, "binary-scalar: %d/%d (%u:%u) row-size %u (%u)",
         ith, nth, شروع_ردیف, انتها_ردیف, nb01, bctx->dst_ردیف_اندازه_aligned);

    ثابت بنية htp_دودویی_vtcm_layout * layout = &bctx->vtcm_layout;
    uint8_t * src0_spad_پایه = VTCM_LAYOUT_PTR(uint8_t, bctx->vtcm_پایه, layout->off_src0) + (ith * layout->src0_بایتها_در_هر_thread);
    uint8_t * dst_spad_پایه  = VTCM_LAYOUT_PTR(uint8_t, bctx->vtcm_پایه, layout->off_dst)  + (ith * layout->dst_بایتها_در_هر_thread);
    size_t src0_spad_half    = layout->src0_spad_half_اندازه;
    size_t dst_spad_half     = layout->dst_spad_half_اندازه;
    ثابت فراغ * s1_جدول    = VTCM_LAYOUT_PTR(ثابت فراغ, bctx->vtcm_پایه, layout->off_src1);

    dma_صف * dma_q = octx->ctx->dma[ith];
    uint32_t ir_prefetch = شروع_ردیف;
    صحيح spad_idx = 0;

    لكل (صحيح k = 0; k < 2 && ir_prefetch < انتها_ردیف; k++) {
        uint32_t current_بلوک_اندازه = calc_بلوک_اندازه(bctx, ir_prefetch, انتها_ردیف, ne01, ne02);
        uint32_t i03, i02, i01, rem;
        i03 = fastdiv(ir_prefetch, &bctx->src0_dim12_div);
        rem = ir_prefetch - i03 * (ne02 * ne01);
        i02 = fastdiv(rem, &bctx->src0_dim1_div);
        i01 = rem - i02 * ne01;

        dma_addr_t src0_curr = src0->داده + i03 * nb03 + i02 * nb02 + i01 * nb01;
        dma_addr_t dst_curr  = dst->داده  + i03 * nb3  + i02 * nb2  + i01 * nb1;

        uint8_t * s0_spad = src0_spad_پایه + spad_idx * src0_spad_half;
        uint8_t * d_spad  = dst_spad_پایه  + spad_idx * dst_spad_half;

        dma_صف_افزودن(dma_q, dma_make_داده(dst_curr, d_spad), nb1, bctx->dst_ردیف_اندازه_aligned, ردیف_اندازه_بایتها, 0);
        dma_صف_افزودن(dma_q, dma_make_داده(s0_spad, src0_curr), bctx->src0_ردیف_اندازه_aligned, nb01, ردیف_اندازه_بایتها,
                                                             current_بلوک_اندازه);
        ir_prefetch += current_بلوک_اندازه;
        spad_idx ^= 1;
    }

    بنية htp_thread_trace * tr = &octx->ctx->trace[ith];
    compute_scalar_t compute = (compute_scalar_t) bctx->compute;

    لكل (uint32_t ir = شروع_ردیف; ir < انتها_ردیف; ) {
        uint32_t current_بلوک_اندازه = calc_بلوک_اندازه(bctx, ir, انتها_ردیف, ne01, ne02);

        uint8_t * d_spad  = (uint8_t *) dma_صف_برداشت(dma_q).src;
        uint8_t * s0_spad = (uint8_t *) dma_صف_برداشت(dma_q).dst;

        uint32_t i03, i02, i01, rem;
        i03 = fastdiv(ir, &bctx->src0_dim12_div);
        rem = ir - i03 * (ne02 * ne01);
        i02 = fastdiv(rem, &bctx->src0_dim1_div);
        i01 = rem - i02 * ne01;

        uint32_t i13 = fastmodulo(i03, ne13, &bctx->src1_dim3_div);
        uint32_t i12 = fastmodulo(i02, ne12, &bctx->src1_dim2_div);
        uint32_t i11 = fastmodulo(i01, ne11, &bctx->src1_dim1_div);
        ثابت uint8_t * scalar_جدول = (ثابت uint8_t *) s1_جدول +
            i13 * plane_stride + i12 * ردیف_stride;

        htp_trace_event_شروع(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);
        compute(d_spad, s0_spad, scalar_جدول, i11, ne11, current_بلوک_اندازه,
                bctx->dst_ردیف_اندازه_aligned, bctx->src0_ردیف_اندازه_aligned, ne00);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);

        dma_addr_t dst_curr = dst->داده + i03 * nb3 + i02 * nb2 + i01 * nb1;
        dma_صف_افزودن(dma_q, dma_make_داده(dst_curr, d_spad), nb1, bctx->dst_ردیف_اندازه_aligned, ردیف_اندازه_بایتها,
                                                                current_بلوک_اندازه);

        إذا (ir_prefetch < انتها_ردیف) {
            uint32_t بعدی_بلوک_اندازه = calc_بلوک_اندازه(bctx, ir_prefetch, انتها_ردیف, ne01, ne02);
            uint32_t p03, p02, p01, prem;
            p03 = fastdiv(ir_prefetch, &bctx->src0_dim12_div);
            prem = ir_prefetch - p03 * (ne02 * ne01);
            p02 = fastdiv(prem, &bctx->src0_dim1_div);
            p01 = prem - p02 * ne01;
            dma_addr_t s0_بعدی = src0->داده + p03 * nb03 + p02 * nb02 + p01 * nb01;
            dma_صف_افزودن(dma_q, dma_make_داده(s0_spad, s0_بعدی), bctx->src0_ردیف_اندازه_aligned, nb01, ردیف_اندازه_بایتها,
                                                               بعدی_بلوک_اندازه);
            ir_prefetch += بعدی_بلوک_اندازه;
        }
        ir += current_بلوک_اندازه;
    }

    dma_صف_flush(dma_q);
}

// 2. Vector Same Shape (ne1x == ne0x) or Simple Broadcast
ساكن فراغ دودویی_thread_وکتور_same_shape(بیعلامت صحيح nth, بیعلامت صحيح ith, فراغ * داده) {
    بنية htp_دودویی_context * bctx = (بنية htp_دودویی_context *) داده;
    بنية htp_ops_context * octx = bctx->octx;
    htp_دودویی_preamble;

    ثابت uint32_t ردیف_اندازه_بایتها = bctx->ردیف_اندازه_بایتها;
    ثابت uint32_t شروع_ردیف = bctx->ردیف_شروع + bctx->nrows_در_هر_thread * ith;
    ثابت uint32_t انتها_ردیف   = MIN(شروع_ردیف + bctx->nrows_در_هر_thread, bctx->ردیف_شروع + bctx->مجموع_ردیفها);
    إذا (شروع_ردیف >= انتها_ردیف) أرجع;

    FARF(بالا_2, "binary-same-shape: %d/%d (%u:%u) row-size %u (%u)",
         ith, nth, شروع_ردیف, انتها_ردیف, nb01, bctx->dst_ردیف_اندازه_aligned);

    ثابت بنية htp_دودویی_vtcm_layout * layout = &bctx->vtcm_layout;
    uint8_t * src0_spad_پایه = VTCM_LAYOUT_PTR(uint8_t, bctx->vtcm_پایه, layout->off_src0) + (ith * layout->src0_بایتها_در_هر_thread);
    uint8_t * src1_spad_پایه = VTCM_LAYOUT_PTR(uint8_t, bctx->vtcm_پایه, layout->off_src1) + (ith * layout->src1_بایتها_در_هر_thread);
    uint8_t * dst_spad_پایه  = VTCM_LAYOUT_PTR(uint8_t, bctx->vtcm_پایه, layout->off_dst)  + (ith * layout->dst_بایتها_در_هر_thread);

    size_t src0_spad_half = layout->src0_spad_half_اندازه;
    size_t src1_spad_half = layout->src1_spad_half_اندازه;
    size_t dst_spad_half  = layout->dst_spad_half_اندازه;

    dma_صف * dma_q = octx->ctx->dma[ith];
    uint32_t ir_prefetch = شروع_ردیف;
    صحيح spad_idx = 0;

    لكل (صحيح k = 0; k < 2 && ir_prefetch < انتها_ردیف; k++) {
        uint32_t current_بلوک_اندازه = calc_بلوک_اندازه(bctx, ir_prefetch, انتها_ردیف, ne01, ne02);
        uint32_t i03, i02, i01, rem;
        i03 = fastdiv(ir_prefetch, &bctx->src0_dim12_div);
        rem = ir_prefetch - i03 * (ne02 * ne01);
        i02 = fastdiv(rem, &bctx->src0_dim1_div);
        i01 = rem - i02 * ne01;

        uint32_t i13 = (ne13 == 1) ? 0 : i03;
        uint32_t i12 = (ne12 == 1) ? 0 : i02;
        uint32_t i11 = (ne11 == 1) ? 0 : i01;

        dma_addr_t src0_curr = src0->داده + i03 * nb03 + i02 * nb02 + i01 * nb01;
        dma_addr_t src1_curr = src1->داده + i13 * nb13 + i12 * nb12 + i11 * nb11;
        dma_addr_t dst_curr  = dst->داده  + i03 * nb3  + i02 * nb2  + i01 * nb1;

        uint8_t * s0_spad = src0_spad_پایه + spad_idx * src0_spad_half;
        uint8_t * s1_spad = src1_spad_پایه + spad_idx * src1_spad_half;
        uint8_t * d_spad  = dst_spad_پایه  + spad_idx * dst_spad_half;

        dma_صف_افزودن(dma_q, dma_make_داده(dst_curr, d_spad), nb1, bctx->dst_ردیف_اندازه_aligned, ردیف_اندازه_بایتها, 0);
        dma_صف_افزودن(dma_q, dma_make_داده(s0_spad, src0_curr), bctx->src0_ردیف_اندازه_aligned, nb01, ردیف_اندازه_بایتها, current_بلوک_اندازه);
        dma_صف_افزودن(dma_q, dma_make_داده(s1_spad, src1_curr), bctx->src1_ردیف_اندازه_aligned, nb11, ردیف_اندازه_بایتها, current_بلوک_اندازه);
        ir_prefetch += current_بلوک_اندازه;
        spad_idx ^= 1;
    }

    بنية htp_thread_trace * tr = &octx->ctx->trace[ith];
    compute_same_shape_t compute = (compute_same_shape_t) bctx->compute;

    لكل (uint32_t ir = شروع_ردیف; ir < انتها_ردیف; ) {
        uint32_t current_بلوک_اندازه = calc_بلوک_اندازه(bctx, ir, انتها_ردیف, ne01, ne02);
        uint8_t * d_spad  = (uint8_t *) dma_صف_برداشت(dma_q).src;
        uint8_t * s0_spad = (uint8_t *) dma_صف_برداشت(dma_q).dst;
        uint8_t * s1_spad = (uint8_t *) dma_صف_برداشت(dma_q).dst;

        htp_trace_event_شروع(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);
        compute(d_spad, s0_spad, s1_spad, current_بلوک_اندازه,
                bctx->dst_ردیف_اندازه_aligned, bctx->src0_ردیف_اندازه_aligned, bctx->src1_ردیف_اندازه_aligned, ne00);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);

        uint32_t i03, i02, i01, rem;
        i03 = fastdiv(ir, &bctx->src0_dim12_div);
        rem = ir - i03 * (ne02 * ne01);
        i02 = fastdiv(rem, &bctx->src0_dim1_div);
        i01 = rem - i02 * ne01;
        dma_addr_t dst_curr = dst->داده + i03 * nb3 + i02 * nb2 + i01 * nb1;
        dma_صف_افزودن(dma_q, dma_make_داده(dst_curr, d_spad), nb1, bctx->dst_ردیف_اندازه_aligned, ردیف_اندازه_بایتها, current_بلوک_اندازه);

        إذا (ir_prefetch < انتها_ردیف) {
            uint32_t بعدی_بلوک_اندازه = calc_بلوک_اندازه(bctx, ir_prefetch, انتها_ردیف, ne01, ne02);
            uint32_t p03, p02, p01, prem;
            p03 = fastdiv(ir_prefetch, &bctx->src0_dim12_div);
            prem = ir_prefetch - p03 * (ne02 * ne01);
            p02 = fastdiv(prem, &bctx->src0_dim1_div);
            p01 = prem - p02 * ne01;

            uint32_t p13 = (ne13 == 1) ? 0 : p03;
            uint32_t p12 = (ne12 == 1) ? 0 : p02;
            uint32_t p11 = (ne11 == 1) ? 0 : p01;

            dma_addr_t s0_بعدی = src0->داده + p03 * nb03 + p02 * nb02 + p01 * nb01;
            dma_addr_t s1_بعدی = src1->داده + p13 * nb13 + p12 * nb12 + p11 * nb11;

            dma_صف_افزودن(dma_q, dma_make_داده(s0_spad, s0_بعدی), bctx->src0_ردیف_اندازه_aligned, nb01, ردیف_اندازه_بایتها, بعدی_بلوک_اندازه);
            dma_صف_افزودن(dma_q, dma_make_داده(s1_spad, s1_بعدی), bctx->src1_ردیف_اندازه_aligned, nb11, ردیف_اندازه_بایتها, بعدی_بلوک_اندازه);

            ir_prefetch += بعدی_بلوک_اندازه;
        }
        ir += current_بلوک_اندازه;
    }

    dma_صف_flush(dma_q);
}

// 3. Row Broadcast (ne11 == 1, ne12 == 1, single row src1)
ساكن فراغ دودویی_thread_وکتور_ردیف_پخش(بیعلامت صحيح nth, بیعلامت صحيح ith, فراغ * داده) {
    بنية htp_دودویی_context * bctx = (بنية htp_دودویی_context *) داده;
    بنية htp_ops_context * octx = bctx->octx;
    htp_دودویی_preamble;

    ثابت uint32_t ردیف_اندازه_بایتها = bctx->ردیف_اندازه_بایتها;
    ثابت uint32_t شروع_ردیف  = bctx->ردیف_شروع + bctx->nrows_در_هر_thread * ith;
    ثابت uint32_t انتها_ردیف    = MIN(شروع_ردیف + bctx->nrows_در_هر_thread, bctx->ردیف_شروع + bctx->مجموع_ردیفها);
    إذا (شروع_ردیف >= انتها_ردیف) أرجع;

    FARF(بالا_2, "binary-row-bcast: %d/%d (%u:%u) row-size %u (%u)",
         ith, nth, شروع_ردیف, انتها_ردیف, nb01, bctx->dst_ردیف_اندازه_aligned);

    ثابت بنية htp_دودویی_vtcm_layout * layout = &bctx->vtcm_layout;
    uint8_t * src0_spad_پایه = VTCM_LAYOUT_PTR(uint8_t, bctx->vtcm_پایه, layout->off_src0) + (ith * layout->src0_بایتها_در_هر_thread);
    uint8_t * dst_spad_پایه  = VTCM_LAYOUT_PTR(uint8_t, bctx->vtcm_پایه, layout->off_dst)  + (ith * layout->dst_بایتها_در_هر_thread);

    size_t src0_spad_half = layout->src0_spad_half_اندازه;
    size_t dst_spad_half  = layout->dst_spad_half_اندازه;

    dma_صف * dma_q = octx->ctx->dma[ith];
    uint32_t ir_prefetch = شروع_ردیف;
    صحيح spad_idx = 0;

    فراغ * s1_ptr = VTCM_LAYOUT_PTR(فراغ, bctx->vtcm_پایه, layout->off_src1);

    لكل (صحيح k = 0; k < 2 && ir_prefetch < انتها_ردیف; k++) {
        uint32_t current_بلوک_اندازه = calc_بلوک_اندازه(bctx, ir_prefetch, انتها_ردیف, ne01, ne02);
        uint32_t i03 = fastdiv(ir_prefetch, &bctx->src0_dim12_div);
        uint32_t rem = ir_prefetch - i03 * (ne02 * ne01);
        uint32_t i02 = fastdiv(rem, &bctx->src0_dim1_div);
        uint32_t i01 = rem - i02 * ne01;

        dma_addr_t src0_curr = src0->داده + i03 * nb03 + i02 * nb02 + i01 * nb01;
        dma_addr_t dst_curr  = dst->داده  + i03 * nb3  + i02 * nb2  + i01 * nb1;

        uint8_t * s0_spad = src0_spad_پایه + spad_idx * src0_spad_half;
        uint8_t * d_spad  = dst_spad_پایه  + spad_idx * dst_spad_half;

        dma_صف_افزودن(dma_q, dma_make_داده(dst_curr, d_spad), nb1, bctx->dst_ردیف_اندازه_aligned, ردیف_اندازه_بایتها, 0);
        dma_صف_افزودن(dma_q, dma_make_داده(s0_spad, src0_curr), bctx->src0_ردیف_اندازه_aligned, nb01, ردیف_اندازه_بایتها, current_بلوک_اندازه);
        ir_prefetch += current_بلوک_اندازه;
        spad_idx ^= 1;
    }

    بنية htp_thread_trace * tr = &octx->ctx->trace[ith];
    compute_ردیف_bcast_t compute = (compute_ردیف_bcast_t) bctx->compute;

    لكل (uint32_t ir = شروع_ردیف; ir < انتها_ردیف; ) {
        uint32_t current_بلوک_اندازه = calc_بلوک_اندازه(bctx, ir, انتها_ردیف, ne01, ne02);
        uint8_t * d_spad  = (uint8_t *) dma_صف_برداشت(dma_q).src;
        uint8_t * s0_spad = (uint8_t *) dma_صف_برداشت(dma_q).dst;

        htp_trace_event_شروع(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);
        compute(d_spad, s0_spad, (ثابت uint8_t *)s1_ptr, current_بلوک_اندازه,
                bctx->dst_ردیف_اندازه_aligned, bctx->src0_ردیف_اندازه_aligned, ne00);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);

        uint32_t i03 = fastdiv(ir, &bctx->src0_dim12_div);
        uint32_t rem = ir - i03 * (ne02 * ne01);
        uint32_t i02 = fastdiv(rem, &bctx->src0_dim1_div);
        uint32_t i01 = rem - i02 * ne01;
        dma_addr_t dst_curr = dst->داده + i03 * nb3 + i02 * nb2 + i01 * nb1;
        dma_صف_افزودن(dma_q, dma_make_داده(dst_curr, d_spad), nb1, bctx->dst_ردیف_اندازه_aligned, ردیف_اندازه_بایتها, current_بلوک_اندازه);

        إذا (ir_prefetch < انتها_ردیف) {
            uint32_t بعدی_بلوک_اندازه = calc_بلوک_اندازه(bctx, ir_prefetch, انتها_ردیف, ne01, ne02);
            uint32_t p03, p02, p01, prem;
            p03  = fastdiv(ir_prefetch, &bctx->src0_dim12_div);
            prem = ir_prefetch - p03 * (ne02 * ne01);
            p02  = fastdiv(prem, &bctx->src0_dim1_div);
            p01  = prem - p02 * ne01;
            dma_addr_t s0_بعدی = src0->داده + p03 * nb03 + p02 * nb02 + p01 * nb01;
            dma_صف_افزودن(dma_q, dma_make_داده(s0_spad, s0_بعدی), bctx->src0_ردیف_اندازه_aligned, nb01, ردیف_اندازه_بایتها, بعدی_بلوک_اندازه);
            ir_prefetch += بعدی_بلوک_اندازه;
        }
        ir += current_بلوک_اندازه;
    }

    dma_صف_flush(dma_q);
}

// 4. Vector Complex (ne10 == ne00, complex broadcast)
ساكن فراغ دودویی_thread_وکتور_complex(بیعلامت صحيح nth, بیعلامت صحيح ith, فراغ * داده) {
    بنية htp_دودویی_context * bctx = (بنية htp_دودویی_context *) داده;
    بنية htp_ops_context * octx = bctx->octx;
    htp_دودویی_preamble;

    ثابت uint32_t ردیف_اندازه_بایتها = bctx->ردیف_اندازه_بایتها;
    ثابت uint32_t شروع_ردیف  = bctx->ردیف_شروع + bctx->nrows_در_هر_thread * ith;
    ثابت uint32_t انتها_ردیف    = MIN(شروع_ردیف + bctx->nrows_در_هر_thread, bctx->ردیف_شروع + bctx->مجموع_ردیفها);
    إذا (شروع_ردیف >= انتها_ردیف) أرجع;

    FARF(بالا_2, "binary-complex: %d/%d (%u:%u) row-size %u (%u)",
         ith, nth, شروع_ردیف, انتها_ردیف, nb01, bctx->dst_ردیف_اندازه_aligned);

    ثابت بنية htp_دودویی_vtcm_layout * layout = &bctx->vtcm_layout;
    uint8_t * src0_spad_پایه = VTCM_LAYOUT_PTR(uint8_t, bctx->vtcm_پایه, layout->off_src0) + (ith * layout->src0_بایتها_در_هر_thread);
    uint8_t * dst_spad_پایه  = VTCM_LAYOUT_PTR(uint8_t, bctx->vtcm_پایه, layout->off_dst)  + (ith * layout->dst_بایتها_در_هر_thread);
    size_t src0_spad_half    = layout->src0_spad_half_اندازه;
    size_t dst_spad_half     = layout->dst_spad_half_اندازه;
    ثابت uint8_t * vtcm_src1 = VTCM_LAYOUT_PTR(ثابت uint8_t, bctx->vtcm_پایه, layout->off_src1);

    ثابت uint32_t elem_اندازه = (src0->type == HTP_TYPE_F32) ? حجم(عائم) : حجم(_Float16);
    ثابت size_t vtcm_ردیف_stride   = (size_t) ne10 * elem_اندازه;
    ثابت size_t vtcm_plane_stride = (size_t) ne11 * vtcm_ردیف_stride;
    ثابت size_t vtcm_vol_stride   = (size_t) ne12 * vtcm_plane_stride;

    dma_صف * dma_q = octx->ctx->dma[ith];
    uint32_t ir_prefetch = شروع_ردیف;
    صحيح spad_idx = 0;

    لكل (صحيح k = 0; k < 2 && ir_prefetch < انتها_ردیف; k++) {
        uint32_t current_بلوک_اندازه = calc_بلوک_اندازه(bctx, ir_prefetch, انتها_ردیف, ne01, ne02);
        uint32_t i03 = fastdiv(ir_prefetch, &bctx->src0_dim12_div);
        uint32_t rem = ir_prefetch - i03 * (ne02 * ne01);
        uint32_t i02 = fastdiv(rem, &bctx->src0_dim1_div);
        uint32_t i01 = rem - i02 * ne01;

        dma_addr_t src0_curr = src0->داده + i03 * nb03 + i02 * nb02 + i01 * nb01;
        dma_addr_t dst_curr  = dst->داده  + i03 * nb3  + i02 * nb2  + i01 * nb1;

        uint8_t * s0_spad = src0_spad_پایه + spad_idx * src0_spad_half;
        uint8_t * d_spad  = dst_spad_پایه  + spad_idx * dst_spad_half;

        dma_صف_افزودن(dma_q, dma_make_داده(dst_curr, d_spad), nb1, bctx->dst_ردیف_اندازه_aligned, ردیف_اندازه_بایتها, 0);
        dma_صف_افزودن(dma_q, dma_make_داده(s0_spad, src0_curr), bctx->src0_ردیف_اندازه_aligned, nb01, ردیف_اندازه_بایتها, current_بلوک_اندازه);
        ir_prefetch += current_بلوک_اندازه;
        spad_idx ^= 1;
    }

    بنية htp_thread_trace * tr = &octx->ctx->trace[ith];
    compute_complex_t compute = (compute_complex_t) bctx->compute;

    لكل (uint32_t ir = شروع_ردیف; ir < انتها_ردیف; ) {
        uint32_t current_بلوک_اندازه = calc_بلوک_اندازه(bctx, ir, انتها_ردیف, ne01, ne02);
        uint8_t * d_spad = (uint8_t *) dma_صف_برداشت(dma_q).src;
        uint8_t * s0_spad = (uint8_t *) dma_صف_برداشت(dma_q).dst;

        uint32_t i03 = fastdiv(ir, &bctx->src0_dim12_div);
        uint32_t rem = ir - i03 * (ne02 * ne01);
        uint32_t i02 = fastdiv(rem, &bctx->src0_dim1_div);
        uint32_t i01 = rem - i02 * ne01;

        uint32_t i13 = fastmodulo(i03, ne13, &bctx->src1_dim3_div);
        uint32_t i12 = fastmodulo(i02, ne12, &bctx->src1_dim2_div);
        ثابت uint8_t * src1_plane = vtcm_src1 + i13 * vtcm_vol_stride + i12 * vtcm_plane_stride;

        htp_trace_event_شروع(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);
        compute(d_spad, s0_spad, src1_plane, i01, ne11, &bctx->src1_dim1_div, vtcm_ردیف_stride,
                current_بلوک_اندازه, bctx->dst_ردیف_اندازه_aligned, bctx->src0_ردیف_اندازه_aligned, ne00);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);

        dma_addr_t dst_curr = dst->داده + i03 * nb3 + i02 * nb2 + i01 * nb1;
        dma_صف_افزودن(dma_q, dma_make_داده(dst_curr, d_spad), nb1, bctx->dst_ردیف_اندازه_aligned, ردیف_اندازه_بایتها, current_بلوک_اندازه);

        إذا (ir_prefetch < انتها_ردیف) {
            uint32_t بعدی_بلوک_اندازه = calc_بلوک_اندازه(bctx, ir_prefetch, انتها_ردیف, ne01, ne02);
            uint32_t p03, p02, p01, prem;
            p03  = fastdiv(ir_prefetch, &bctx->src0_dim12_div);
            prem = ir_prefetch - p03 * (ne02 * ne01);
            p02  = fastdiv(prem, &bctx->src0_dim1_div);
            p01  = prem - p02 * ne01;
            dma_addr_t s0_بعدی = src0->داده + p03 * nb03 + p02 * nb02 + p01 * nb01;
            dma_صف_افزودن(dma_q, dma_make_داده(s0_spad, s0_بعدی), bctx->src0_ردیف_اندازه_aligned, nb01, ردیف_اندازه_بایتها, بعدی_بلوک_اندازه);
            ir_prefetch += بعدی_بلوک_اندازه;
        }
        ir += current_بلوک_اندازه;
    }

    dma_صف_flush(dma_q);
}

// 5. Element Repeat (ne10 != ne00)
ساكن فراغ دودویی_thread_element_repeat(بیعلامت صحيح nth, بیعلامت صحيح ith, فراغ * داده) {
    بنية htp_دودویی_context * bctx = (بنية htp_دودویی_context *) داده;
    بنية htp_ops_context * octx = bctx->octx;
    htp_دودویی_preamble;

    ثابت uint32_t ردیف_اندازه_بایتها = bctx->ردیف_اندازه_بایتها;
    ثابت uint32_t شروع_ردیف  = bctx->ردیف_شروع + bctx->nrows_در_هر_thread * ith;
    ثابت uint32_t انتها_ردیف    = MIN(شروع_ردیف + bctx->nrows_در_هر_thread, bctx->ردیف_شروع + bctx->مجموع_ردیفها);
    إذا (شروع_ردیف >= انتها_ردیف) أرجع;

    ثابت بنية htp_دودویی_vtcm_layout * layout = &bctx->vtcm_layout;
    uint8_t * src0_spad_پایه = VTCM_LAYOUT_PTR(uint8_t, bctx->vtcm_پایه, layout->off_src0) + (ith * layout->src0_بایتها_در_هر_thread);
    uint8_t * dst_spad_پایه  = VTCM_LAYOUT_PTR(uint8_t, bctx->vtcm_پایه, layout->off_dst)  + (ith * layout->dst_بایتها_در_هر_thread);
    size_t src0_spad_half    = layout->src0_spad_half_اندازه;
    size_t dst_spad_half     = layout->dst_spad_half_اندازه;
    ثابت uint8_t * vtcm_src1 = VTCM_LAYOUT_PTR(ثابت uint8_t, bctx->vtcm_پایه, layout->off_src1);

    ثابت uint32_t elem_اندازه = (src0->type == HTP_TYPE_F32) ? حجم(عائم) : حجم(_Float16);
    ثابت size_t vtcm_ردیف_stride   = (size_t) ne10 * elem_اندازه;
    ثابت size_t vtcm_plane_stride = (size_t) ne11 * vtcm_ردیف_stride;
    ثابت size_t vtcm_vol_stride   = (size_t) ne12 * vtcm_plane_stride;

    FARF(بالا_2, "binary-repeat: %d/%d (%u:%u) row-size %u (%u)",
         ith, nth, شروع_ردیف, انتها_ردیف, nb01, bctx->dst_ردیف_اندازه_aligned);

    dma_صف * dma_q = octx->ctx->dma[ith];
    uint32_t ir_prefetch = شروع_ردیف;
    صحيح spad_idx = 0;

    لكل (صحيح k = 0; k < 2 && ir_prefetch < انتها_ردیف; k++) {
        uint32_t current_بلوک_اندازه = calc_بلوک_اندازه(bctx, ir_prefetch, انتها_ردیف, ne01, ne02);
        uint32_t i03 = fastdiv(ir_prefetch, &bctx->src0_dim12_div);
        uint32_t rem = ir_prefetch - i03 * (ne02 * ne01);
        uint32_t i02 = fastdiv(rem, &bctx->src0_dim1_div);
        uint32_t i01 = rem - i02 * ne01;

        dma_addr_t src0_curr = src0->داده + i03 * nb03 + i02 * nb02 + i01 * nb01;
        dma_addr_t dst_curr  = dst->داده  + i03 * nb3  + i02 * nb2  + i01 * nb1;

        uint8_t * s0_spad = src0_spad_پایه + spad_idx * src0_spad_half;
        uint8_t * d_spad  = dst_spad_پایه  + spad_idx * dst_spad_half;

        dma_صف_افزودن(dma_q, dma_make_داده(dst_curr, d_spad), nb1, bctx->dst_ردیف_اندازه_aligned, ردیف_اندازه_بایتها, 0);
        dma_صف_افزودن(dma_q, dma_make_داده(s0_spad, src0_curr), bctx->src0_ردیف_اندازه_aligned, nb01, ردیف_اندازه_بایتها, current_بلوک_اندازه);
        ir_prefetch += current_بلوک_اندازه;
        spad_idx ^= 1;
    }

    بنية htp_thread_trace * tr = &octx->ctx->trace[ith];
    compute_repeat_t compute = (compute_repeat_t) bctx->compute;

    لكل (uint32_t ir = شروع_ردیف; ir < انتها_ردیف; ) {
        uint32_t current_بلوک_اندازه = calc_بلوک_اندازه(bctx, ir, انتها_ردیف, ne01, ne02);
        uint8_t * d_spad  = (uint8_t *) dma_صف_برداشت(dma_q).src;
        uint8_t * s0_spad = (uint8_t *) dma_صف_برداشت(dma_q).dst;

        uint32_t i03 = fastdiv(ir, &bctx->src0_dim12_div);
        uint32_t rem = ir - i03 * (ne02 * ne01);
        uint32_t i02 = fastdiv(rem, &bctx->src0_dim1_div);
        uint32_t i01 = rem - i02 * ne01;

        uint32_t i13 = fastmodulo(i03, ne13, &bctx->src1_dim3_div);
        uint32_t i12 = fastmodulo(i02, ne12, &bctx->src1_dim2_div);
        ثابت uint8_t * src1_plane = vtcm_src1 + i13 * vtcm_vol_stride + i12 * vtcm_plane_stride;

        htp_trace_event_شروع(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);
        compute(d_spad, s0_spad, src1_plane, i01, ne11, &bctx->src1_dim1_div, vtcm_ردیف_stride,
                current_بلوک_اندازه, bctx->dst_ردیف_اندازه_aligned, bctx->src0_ردیف_اندازه_aligned, ne00, ne10);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);

        dma_addr_t dst_curr = dst->داده + i03 * nb3 + i02 * nb2 + i01 * nb1;
        dma_صف_افزودن(dma_q, dma_make_داده(dst_curr, d_spad), nb1, bctx->dst_ردیف_اندازه_aligned, ردیف_اندازه_بایتها, current_بلوک_اندازه);

        إذا (ir_prefetch < انتها_ردیف) {
            uint32_t بعدی_بلوک_اندازه = calc_بلوک_اندازه(bctx, ir_prefetch, انتها_ردیف, ne01, ne02);
            uint32_t p03, p02, p01, prem;
            p03  = fastdiv(ir_prefetch, &bctx->src0_dim12_div);
            prem = ir_prefetch - p03 * (ne02 * ne01);
            p02  = fastdiv(prem, &bctx->src0_dim1_div);
            p01  = prem - p02 * ne01;
            dma_addr_t s0_بعدی = src0->داده + p03 * nb03 + p02 * nb02 + p01 * nb01;
            dma_صف_افزودن(dma_q, dma_make_داده(s0_spad, s0_بعدی), bctx->src0_ردیف_اندازه_aligned, nb01, ردیف_اندازه_بایتها, بعدی_بلوک_اندازه);
            ir_prefetch += بعدی_بلوک_اندازه;
        }
        ir += current_بلوک_اندازه;
    }

    dma_صف_flush(dma_q);
}

// 6. ADD_ID (src1 gathered via src2 indices)
ساكن فراغ دودویی_thread_جمع_کن_id_f32(بیعلامت صحيح nth, بیعلامت صحيح ith, فراغ * داده) {
    بنية htp_دودویی_context * bctx = (بنية htp_دودویی_context *) داده;
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
    ثابت uint32_t src1_stride = bctx->src1_ردیف_اندازه_aligned;

    ثابت uint32_t nb1 = dst->nb[1];
    ثابت uint32_t nb2 = dst->nb[2];
    ثابت uint32_t nb3 = dst->nb[3];

    ثابت uint32_t ردیف_اندازه_بایتها = bctx->ردیف_اندازه_بایتها;
    ثابت uint32_t شروع_ردیف = bctx->ردیف_شروع + bctx->nrows_در_هر_thread * ith;
    ثابت uint32_t انتها_ردیف   = MIN(شروع_ردیف + bctx->nrows_در_هر_thread, bctx->ردیف_شروع + bctx->مجموع_ردیفها);
    إذا (شروع_ردیف >= انتها_ردیف) أرجع;

    ثابت بنية htp_دودویی_vtcm_layout * layout = &bctx->vtcm_layout;
    uint8_t * src0_spad_پایه = VTCM_LAYOUT_PTR(uint8_t, bctx->vtcm_پایه, layout->off_src0) + (ith * layout->src0_بایتها_در_هر_thread);
    uint8_t * dst_spad_پایه  = VTCM_LAYOUT_PTR(uint8_t, bctx->vtcm_پایه, layout->off_dst)  + (ith * layout->dst_بایتها_در_هر_thread);
    ثابت uint8_t * vtcm_src1 = VTCM_LAYOUT_PTR(ثابت uint8_t, bctx->vtcm_پایه, layout->off_src1);
    ثابت int32_t * vtcm_src2 = VTCM_LAYOUT_PTR(ثابت int32_t, bctx->vtcm_پایه, layout->off_src2);
    size_t src0_spad_half    = layout->src0_spad_half_اندازه;
    size_t dst_spad_half     = layout->dst_spad_half_اندازه;

    dma_صف * dma_q = octx->ctx->dma[ith];
    uint32_t ir_prefetch = شروع_ردیف;
    صحيح spad_idx = 0;

    لكل (صحيح k = 0; k < 2 && ir_prefetch < انتها_ردیف; k++) {
        uint32_t current_بلوک_اندازه = calc_بلوک_اندازه(bctx, ir_prefetch, انتها_ردیف, ne01, ne02);
        uint32_t i03 = fastdiv(ir_prefetch, &bctx->src0_dim12_div);
        uint32_t rem = ir_prefetch - i03 * (ne02 * ne01);
        uint32_t i02 = fastdiv(rem, &bctx->src0_dim1_div);
        uint32_t i01 = rem - i02 * ne01;

        dma_addr_t src0_curr = src0->داده + i03 * nb03 + i02 * nb02 + i01 * nb01;
        dma_addr_t dst_curr  = dst->داده  + i03 * nb3  + i02 * nb2  + i01 * nb1;

        uint8_t * s0_spad = src0_spad_پایه + spad_idx * src0_spad_half;
        uint8_t * d_spad  = dst_spad_پایه  + spad_idx * dst_spad_half;

        dma_صف_افزودن(dma_q, dma_make_داده(dst_curr, d_spad), nb1, bctx->dst_ردیف_اندازه_aligned, ردیف_اندازه_بایتها, 0);
        dma_صف_افزودن(dma_q, dma_make_داده(s0_spad, src0_curr), bctx->src0_ردیف_اندازه_aligned, nb01, ردیف_اندازه_بایتها, current_بلوک_اندازه);
        ir_prefetch += current_بلوک_اندازه;
        spad_idx ^= 1;
    }

    بنية htp_thread_trace * tr = &octx->ctx->trace[ith];

    لكل (uint32_t ir = شروع_ردیف; ir < انتها_ردیف; ) {
        uint32_t current_بلوک_اندازه = calc_بلوک_اندازه(bctx, ir, انتها_ردیف, ne01, ne02);
        uint8_t * d_spad = (uint8_t *) dma_صف_برداشت(dma_q).src;
        uint8_t * s0_spad = (uint8_t *) dma_صف_برداشت(dma_q).dst;

        uint32_t i03 = fastdiv(ir, &bctx->src0_dim12_div);
        uint32_t rem = ir - i03 * (ne02 * ne01);
        uint32_t i02 = fastdiv(rem, &bctx->src0_dim1_div);
        uint32_t i01 = rem - i02 * ne01;

        htp_trace_event_شروع(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);
        compute_جمع_کن_id_t compute = (compute_جمع_کن_id_t) bctx->compute;
        compute(d_spad, s0_spad, vtcm_src1, vtcm_src2,
                i01, i02, ne01, src1_stride,
                current_بلوک_اندازه, bctx->dst_ردیف_اندازه_aligned, bctx->src0_ردیف_اندازه_aligned, ne00);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);

        dma_addr_t dst_curr = dst->داده + i03 * nb3 + i02 * nb2 + i01 * nb1;
        dma_صف_افزودن(dma_q, dma_make_داده(dst_curr, d_spad), nb1, bctx->dst_ردیف_اندازه_aligned, ردیف_اندازه_بایتها, current_بلوک_اندازه);

        إذا (ir_prefetch < انتها_ردیف) {
             uint32_t بعدی_بلوک_اندازه = calc_بلوک_اندازه(bctx, ir_prefetch, انتها_ردیف, ne01, ne02);
             uint32_t p03, p02, p01, prem;
             p03  = fastdiv(ir_prefetch, &bctx->src0_dim12_div);
             prem = ir_prefetch - p03 * (ne02 * ne01);
             p02  = fastdiv(prem, &bctx->src0_dim1_div);
             p01  = prem - p02 * ne01;
             dma_addr_t s0_بعدی = src0->داده + p03 * nb03 + p02 * nb02 + p01 * nb01;
             dma_صف_افزودن(dma_q, dma_make_داده(s0_spad, s0_بعدی), bctx->src0_ردیف_اندازه_aligned, nb01, ردیف_اندازه_بایتها, بعدی_بلوک_اندازه);
             ir_prefetch += بعدی_بلوک_اندازه;
        }
        ir += current_بلوک_اندازه;
    }

    dma_صف_flush(dma_q);
}

ساكن مضمن فراغ hvx_div_scalar_f32(uint8_t * محدود dst, ثابت uint8_t * محدود src, ثابت عائم val, ثابت uint32_t رقم_elems) {
    hvx_mul_scalar_f32(dst, src, 1.0f / val, رقم_elems);
}

ساكن مضمن فراغ hvx_div_scalar_f16(uint8_t * محدود dst, ثابت uint8_t * محدود src, ثابت _Float16 val, ثابت uint32_t رقم_elems) {
    hvx_div_scalar_f16_aa(dst, src, val, رقم_elems);
}

تعريفنوع فراغ (*compute_دودویی_chunked_t)(
    uint8_t * محدود dst,
    ثابت uint8_t * محدود src0,
    ثابت uint8_t * محدود src1,
    ثابت uint32_t رقم_elems
);

تعريفنوع فراغ (*compute_دودویی_scalar_chunked_f32_t)(
    uint8_t * محدود dst,
    ثابت uint8_t * محدود src0,
    ثابت عائم val,
    ثابت uint32_t رقم_elems
);

تعريفنوع فراغ (*compute_دودویی_scalar_chunked_f16_t)(
    uint8_t * محدود dst,
    ثابت uint8_t * محدود src0,
    ثابت _Float16 val,
    ثابت uint32_t رقم_elems
);

بنية دودویی_chunked_context {
    بنية htp_ops_context * octx;
    بنية htp_دودویی_vtcm_layout vtcm_layout;
    uint8_t *                vtcm_پایه;
    uint32_t                 elem_شروع;
    uint32_t                 nelem;
    uint32_t                 chunk_اندازه;
    uint32_t                 chunks_در_هر_thread;
    uint32_t                 مجموع_chunks;
    bool                     is_scalar;
    عائم                    scalar_f32;
    _Float16                 scalar_f16;
    compute_دودویی_chunked_t compute;
    compute_دودویی_scalar_chunked_f32_t compute_scalar_f32;
    compute_دودویی_scalar_chunked_f16_t compute_scalar_f16;
};

ساكن فراغ دودویی_thread_chunked(بیعلامت صحيح nth, بیعلامت صحيح ith, فراغ * داده) {
    (فراغ) nth;
    بنية دودویی_chunked_context * ctx = (بنية دودویی_chunked_context *) داده;
    بنية htp_ops_context * octx = ctx->octx;
    ثابت بنية htp_tensor * src0 = octx->src[0];
    ثابت بنية htp_tensor * src1 = octx->src[1];
    ثابت بنية htp_tensor * dst  = octx->dst;

    ثابت uint32_t شروع_chunk = ctx->chunks_در_هر_thread * ith;
    ثابت uint32_t انتها_chunk   = MIN(شروع_chunk + ctx->chunks_در_هر_thread, ctx->مجموع_chunks);
    إذا (شروع_chunk >= انتها_chunk) {
        أرجع;
    }

    ثابت uint32_t src0_type   = src0->type;
    ثابت size_t   elem_اندازه   = (src0_type == HTP_TYPE_F32) ? حجم(عائم) : حجم(_Float16);
    ثابت uint32_t chunk_اندازه  = ctx->chunk_اندازه;
    ثابت size_t   chunk_بایتها = ctx->vtcm_layout.src0_spad_half_اندازه;

    FARF(بالا_2, "binary-chunked: %d/%d (%u:%u) chunks %u elems %u",
         ith, nth, شروع_chunk, انتها_chunk, ctx->مجموع_chunks, ctx->nelem);

    ثابت بنية htp_دودویی_vtcm_layout * layout = &ctx->vtcm_layout;
    uint8_t * src0_spad_پایه = VTCM_LAYOUT_PTR(uint8_t, ctx->vtcm_پایه, layout->off_src0) + (ith * layout->src0_بایتها_در_هر_thread);
    uint8_t * src1_spad_پایه = ctx->is_scalar ? عدم : (VTCM_LAYOUT_PTR(uint8_t, ctx->vtcm_پایه, layout->off_src1) + (ith * layout->src1_بایتها_در_هر_thread));
    uint8_t * dst_spad_پایه  = VTCM_LAYOUT_PTR(uint8_t, ctx->vtcm_پایه, layout->off_dst)  + (ith * layout->dst_بایتها_در_هر_thread);

    dma_صف * dma_q = octx->ctx->dma[ith];
    uint32_t prefetch_chunk = شروع_chunk;
    صحيح spad_idx = 0;

    لكل (صحيح k = 0; k < 2 && prefetch_chunk < انتها_chunk; k++) {
        ثابت uint32_t c_شروع   = ctx->elem_شروع + prefetch_chunk * chunk_اندازه;
        ثابت uint32_t c_انتها     = MIN(c_شروع + chunk_اندازه, ctx->elem_شروع + ctx->nelem);
        ثابت uint32_t cur_elems = c_انتها - c_شروع;
        ثابت uint32_t cur_بایتها = cur_elems * elem_اندازه;

        dma_addr_t s0_curr = src0->داده + (size_t) c_شروع * elem_اندازه;
        dma_addr_t d_curr  = dst->داده  + (size_t) c_شروع * elem_اندازه;

        uint8_t * s0_spad = src0_spad_پایه + spad_idx * chunk_بایتها;
        uint8_t * d_spad  = dst_spad_پایه  + spad_idx * chunk_بایتها;

        dma_صف_افزودن(dma_q, dma_make_داده(d_curr, d_spad), chunk_بایتها, chunk_بایتها, cur_بایتها, 0);
        dma_صف_افزودن(dma_q, dma_make_داده(s0_spad, s0_curr), chunk_بایتها, chunk_بایتها, cur_بایتها, 1);
        إذا (!ctx->is_scalar) {
            dma_addr_t s1_curr = src1->داده + (size_t) c_شروع * elem_اندازه;
            uint8_t * s1_spad = src1_spad_پایه + spad_idx * chunk_بایتها;
            dma_صف_افزودن(dma_q, dma_make_داده(s1_spad, s1_curr), chunk_بایتها, chunk_بایتها, cur_بایتها, 1);
        }

        prefetch_chunk++;
        spad_idx ^= 1;
    }

    بنية htp_thread_trace * tr = &octx->ctx->trace[ith];

    لكل (uint32_t c = شروع_chunk; c < انتها_chunk; c++) {
        ثابت uint32_t c_شروع   = ctx->elem_شروع + c * chunk_اندازه;
        ثابت uint32_t c_انتها     = MIN(c_شروع + chunk_اندازه, ctx->elem_شروع + ctx->nelem);
        ثابت uint32_t cur_elems = c_انتها - c_شروع;
        ثابت uint32_t cur_بایتها = cur_elems * elem_اندازه;

        uint8_t * d_spad  = (uint8_t *) dma_صف_برداشت(dma_q).src;
        uint8_t * s0_spad = (uint8_t *) dma_صف_برداشت(dma_q).dst;
        uint8_t * s1_spad = ctx->is_scalar ? عدم : (uint8_t *) dma_صف_برداشت(dma_q).dst;

        htp_trace_event_شروع(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) c);
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

        dma_addr_t dst_curr = dst->داده + (size_t) c_شروع * elem_اندازه;
        dma_صف_افزودن(dma_q, dma_make_داده(dst_curr, d_spad), chunk_بایتها, chunk_بایتها, cur_بایتها, 1);

        إذا (prefetch_chunk < انتها_chunk) {
            ثابت uint32_t pc_شروع  = ctx->elem_شروع + prefetch_chunk * chunk_اندازه;
            ثابت uint32_t pc_انتها    = MIN(pc_شروع + chunk_اندازه, ctx->elem_شروع + ctx->nelem);
            ثابت uint32_t p_elems   = pc_انتها - pc_شروع;
            ثابت uint32_t p_بایتها   = p_elems * elem_اندازه;

            dma_addr_t s0_بعدی = src0->داده + (size_t) pc_شروع * elem_اندازه;
            dma_صف_افزودن(dma_q, dma_make_داده(s0_spad, s0_بعدی), chunk_بایتها, chunk_بایتها, p_بایتها, 1);
            إذا (!ctx->is_scalar) {
                dma_addr_t s1_بعدی = src1->داده + (size_t) pc_شروع * elem_اندازه;
                dma_صف_افزودن(dma_q, dma_make_داده(s1_spad, s1_بعدی), chunk_بایتها, chunk_بایتها, p_بایتها, 1);
            }

            prefetch_chunk++;
        }
    }

    dma_صف_flush(dma_q);
}

ساكن صحيح execute_op_دودویی(بنية htp_ops_context * octx) {
    ثابت بنية htp_tensor * src0 = octx->src[0];
    ثابت بنية htp_tensor * src1 = octx->src[1];
    ثابت بنية htp_tensor * dst  = octx->dst;
    ثابت بنية htp_دودویی_kernel_params * kparams = (ثابت بنية htp_دودویی_kernel_params *) octx->kernel_params;

    ثابت uint32_t src0_nrows = src0->ne[1] * src0->ne[2] * src0->ne[3];

    // Use packed row sizes for VTCM allocation and alignment
    ثابت uint32_t src0_type = octx->src[0]->type;
    ثابت size_t elem_اندازه = (src0_type == HTP_TYPE_F32) ? حجم(عائم) : حجم(_Float16);
    ثابت size_t src0_ردیف_اندازه = src0->ne[0] * elem_اندازه;
    ثابت size_t src1_ردیف_اندازه = src1->ne[0] * elem_اندازه;
    ثابت size_t dst_ردیف_اندازه  = dst->ne[0]  * elem_اندازه;

    إذا (kparams->kernel_type == HTP_دودویی_KERNEL_CHUNKED) {
        ثابت bool is_scalar = kparams->is_scalar || (src1->ne[0] == 1 && src1->ne[1] == 1 && src1->ne[2] == 1 && src1->ne[3] == 1);

        compute_دودویی_chunked_t compute = عدم;
        compute_دودویی_scalar_chunked_f32_t compute_scalar_f32 = عدم;
        compute_دودویی_scalar_chunked_f16_t compute_scalar_f16 = عدم;

        إذا (is_scalar) {
            إذا (src0_type == HTP_TYPE_F32) {
                بدل (octx->op) {
                    حالة HTP_OP_جمع_کن: compute_scalar_f32 = hvx_جمع_کن_scalar_f32; اكسر;
                    حالة HTP_OP_SUB: compute_scalar_f32 = hvx_sub_scalar_f32; اكسر;
                    حالة HTP_OP_MUL: compute_scalar_f32 = hvx_mul_scalar_f32; اكسر;
                    حالة HTP_OP_DIV: compute_scalar_f32 = hvx_div_scalar_f32; اكسر;
                    افتراضي: اكسر;
                }
            } وإلا إذا (src0_type == HTP_TYPE_F16) {
                بدل (octx->op) {
                    حالة HTP_OP_جمع_کن: compute_scalar_f16 = hvx_جمع_کن_scalar_f16; اكسر;
                    حالة HTP_OP_SUB: compute_scalar_f16 = hvx_sub_scalar_f16; اكسر;
                    حالة HTP_OP_MUL: compute_scalar_f16 = hvx_mul_scalar_f16; اكسر;
                    حالة HTP_OP_DIV: compute_scalar_f16 = hvx_div_scalar_f16; اكسر;
                    افتراضي: اكسر;
                }
            }
            إذا (!compute_scalar_f32 && !compute_scalar_f16) {
                أرجع HTP_وضعیت_NO_SUPPORT;
            }
        } وإلا {
            إذا (src0_type == HTP_TYPE_F32) {
                بدل (octx->op) {
                    حالة HTP_OP_جمع_کن: compute = hvx_جمع_کن_f32; اكسر;
                    حالة HTP_OP_SUB: compute = hvx_sub_f32; اكسر;
                    حالة HTP_OP_MUL: compute = hvx_mul_f32; اكسر;
                    حالة HTP_OP_DIV: compute = hvx_div_f32; اكسر;
                    افتراضي: اكسر;
                }
            } وإلا إذا (src0_type == HTP_TYPE_F16) {
                بدل (octx->op) {
                    حالة HTP_OP_جمع_کن: compute = hvx_جمع_کن_f16; اكسر;
                    حالة HTP_OP_SUB: compute = hvx_sub_f16; اكسر;
                    حالة HTP_OP_MUL: compute = hvx_mul_f16; اكسر;
                    حالة HTP_OP_DIV: compute = hvx_div_f16; اكسر;
                    افتراضي: اكسر;
                }
            }
            إذا (!compute) {
                أرجع HTP_وضعیت_NO_SUPPORT;
            }
        }

        ثابت uint32_t مجموع_elems = (uint32_t) (src0->ne[0] * src0->ne[1] * src0->ne[2] * src0->ne[3]);
        إذا (مجموع_elems == 0) {
            أرجع HTP_وضعیت_OK;
        }

        uint32_t elem_شروع = 0;
        uint32_t nelem      = مجموع_elems;
        ثابت uint32_t elems_در_هر_سطر = (src0_type == HTP_TYPE_F32) ? 32 : 64;

        إذا (octx->ctx->mdev.شمار > 1) {
            ثابت bool can_تقسیم = htp_tensor_mdev_داده_aligned(dst) &&
                                   htp_tensor_mdev_داده_aligned(src0) &&
                                   (is_scalar || htp_tensor_mdev_داده_aligned(src1)) &&
                                   htp_tensor_is_contiguous(dst, elem_اندازه) &&
                                   htp_tensor_is_contiguous(src0, elem_اندازه) &&
                                   (is_scalar || htp_tensor_is_contiguous(src1, elem_اندازه));
            ثابت بنية htp_tensor_mdev_range range = htp_tensor_mdev_افراز(
                مجموع_elems, can_تقسیم ? elems_در_هر_سطر : 0,
                octx->ctx->mdev.idx, octx->ctx->mdev.شمار, &octx->ctx->mdev.شمار_div);
            elem_شروع = range.شروع;
            nelem      = range.شمار;
        }

        إذا (nelem == 0) {
            أرجع HTP_وضعیت_OK;
        }

        إذا (!htp_ops_context_مجموعه_n_نخها(octx, kparams->n_نخها)) {
            أرجع HTP_وضعیت_INVAL_PARAMS;
        }

        بنية htp_دودویی_vtcm_layout vtcm_layout;
        htp_دودویی_vtcm_layout_build(&vtcm_layout, kparams, octx->ctx->vtcm_اندازه);
        إذا (vtcm_layout.مجموع_بایتها == 0 || vtcm_layout.مجموع_بایتها > octx->ctx->vtcm_اندازه) {
            أرجع HTP_وضعیت_VTCM_TOO_SMALL;
        }

        ثابت uint32_t chunk_اندازه = kparams->chunk_اندازه > 0 ? kparams->chunk_اندازه : (32768 / elem_اندازه);
        ثابت uint32_t مجموع_chunks = (nelem + chunk_اندازه - 1) / chunk_اندازه;
        ثابت uint32_t n_نخها = (مجموع_chunks >= 2) ? MIN(octx->n_نخها, مجموع_chunks) : 1;
        ثابت uint32_t chunks_در_هر_thread = (مجموع_chunks + n_نخها - 1) / n_نخها;

        عائم scalar_f32 = 0.0f;
        _Float16 scalar_f16 = 0;
        إذا (is_scalar) {
            uint8_t * vtcm_src1 = VTCM_LAYOUT_PTR(uint8_t, octx->ctx->vtcm_پایه, vtcm_layout.off_src1);
            dma_صف * dma_q = octx->ctx->dma[0];
            dma_صف_افزودن(dma_q, dma_make_داده(vtcm_src1, src1->داده), 128, 0, elem_اندازه, 1);
            dma_صف_برداشت(dma_q);

            إذا (src0_type == HTP_TYPE_F32) {
                scalar_f32 = ((ثابت عائم *) vtcm_src1)[0];
            } وإلا {
                scalar_f16 = ((ثابت _Float16 *) vtcm_src1)[0];
            }
        }

        بنية دودویی_chunked_context cctx = {
            .octx                = octx,
            .vtcm_layout         = vtcm_layout,
            .vtcm_پایه           = (uint8_t *) octx->ctx->vtcm_پایه,
            .elem_شروع          = elem_شروع,
            .nelem               = nelem,
            .chunk_اندازه          = chunk_اندازه,
            .chunks_در_هر_thread   = chunks_در_هر_thread,
            .مجموع_chunks        = مجموع_chunks,
            .is_scalar           = is_scalar,
            .scalar_f32          = scalar_f32,
            .scalar_f16          = scalar_f16,
            .compute             = compute,
            .compute_scalar_f32  = compute_scalar_f32,
            .compute_scalar_f16  = compute_scalar_f16,
        };

        work_صف_پیاپی(octx->ctx->work_صف, دودویی_thread_chunked, &cctx, n_نخها);
        أرجع HTP_وضعیت_OK;
    }

    uint32_t ردیف_شروع = 0;
    uint32_t nrows     = src0_nrows;

    إذا (octx->ctx->mdev.شمار > 1) {
        uint32_t ردیفها_در_هر_chunk = 0;
        htp_tensor_mdev_ردیفها_در_هر_chunk(dst, (uint32_t) elem_اندازه, (uint32_t) dst_ردیف_اندازه, &ردیفها_در_هر_chunk);
        ثابت بنية htp_tensor_mdev_range range = htp_tensor_mdev_افراز(src0_nrows, ردیفها_در_هر_chunk,
                                                       octx->ctx->mdev.idx, octx->ctx->mdev.شمار, &octx->ctx->mdev.شمار_div);
        ردیف_شروع = range.شروع;
        nrows     = range.شمار;
    }

    إذا (nrows == 0) {
        أرجع HTP_وضعیت_OK;
    }

    إذا (!htp_ops_context_مجموعه_n_نخها(octx, kparams->n_نخها)) {
        أرجع HTP_وضعیت_INVAL_PARAMS;
    }

    ثابت uint32_t n_نخها = octx->n_نخها;
    ثابت size_t src0_ردیف_اندازه_aligned = kparams->src0_ردیف_اندازه_aligned;
    ثابت size_t src1_ردیف_اندازه_aligned = kparams->src1_ردیف_اندازه_aligned;
    ثابت size_t dst_ردیف_اندازه_aligned  = kparams->dst_ردیف_اندازه_aligned;

    بنية htp_دودویی_context bctx;
    bctx.vtcm_پایه = (uint8_t *) octx->ctx->vtcm_پایه;
    htp_دودویی_vtcm_layout_build(&bctx.vtcm_layout, kparams, octx->ctx->vtcm_اندازه);

    إذا (bctx.vtcm_layout.ردیفها_در_هر_میانگیر == 0 || bctx.vtcm_layout.مجموع_بایتها > octx->ctx->vtcm_اندازه) {
        أرجع HTP_وضعیت_VTCM_TOO_SMALL;
    }

    dma_صف * dma_q = octx->ctx->dma[0];
    uint8_t * vtcm_src1 = VTCM_LAYOUT_PTR(uint8_t, bctx.vtcm_پایه, bctx.vtcm_layout.off_src1);
    إذا (kparams->kernel_type == HTP_دودویی_KERNEL_ردیف_BCAST) {
        dma_صف_افزودن(dma_q, dma_make_داده(vtcm_src1, src1->داده), bctx.vtcm_layout.src1_اندازه, 0, src1->ne[0] * elem_اندازه, 1);
    } وإلا إذا (kparams->kernel_type == HTP_دودویی_KERNEL_SCALAR ||
               kparams->kernel_type == HTP_دودویی_KERNEL_REPEAT ||
               kparams->kernel_type == HTP_دودویی_KERNEL_COMPLEX) {
        إذا (!kparams->src1_strided) {
            ثابت size_t src1_بایتها = (size_t) src1->ne[0] * src1->ne[1] * src1->ne[2] * src1->ne[3] * elem_اندازه;
            dma_صف_افزودن(dma_q, dma_make_داده(vtcm_src1, src1->داده), bctx.vtcm_layout.src1_اندازه, 0, src1_بایتها, 1);
        } وإلا {
            ثابت uint32_t ne11 = src1->ne[1];
            ثابت uint32_t ne12 = src1->ne[2];
            ثابت uint32_t ne13 = src1->ne[3];
            ثابت size_t ردیف_بایتها = src1->ne[0] * elem_اندازه;
            لكل (uint32_t i13 = 0; i13 < ne13; i13++) {
                لكل (uint32_t i12 = 0; i12 < ne12; i12++) {
                    dma_addr_t src_addr = src1->داده + i13 * src1->nb[3] + i12 * src1->nb[2];
                    uint8_t * dst_ptr = vtcm_src1 + (i13 * ne12 * ne11 + i12 * ne11) * ردیف_بایتها;
                    طالما (!dma_صف_افزودن(dma_q, dma_make_داده(dst_ptr, src_addr),
                                           ردیف_بایتها, src1->nb[1], ردیف_بایتها, ne11)) {
                        dma_صف_برداشت(dma_q);
                    }
                }
            }
        }
    } وإلا إذا (kparams->kernel_type == HTP_دودویی_KERNEL_جمع_کن_ID) {
        ثابت بنية htp_tensor * src2 = octx->src[2];
        uint8_t * vtcm_src2 = VTCM_LAYOUT_PTR(uint8_t, bctx.vtcm_پایه, bctx.vtcm_layout.off_src2);
        dma_صف_افزودن(dma_q, dma_make_داده(vtcm_src1, src1->داده),
                       kparams->src1_ردیف_اندازه_aligned, src1->nb[1],
                       src1->ne[0] * elem_اندازه, src1->ne[1]);
        ثابت size_t src2_ردیف_بایتها = src2->ne[0] * حجم(int32_t);
        dma_صف_افزودن(dma_q, dma_make_داده(vtcm_src2, src2->داده),
                       src2_ردیف_بایتها, src2->nb[1],
                       src2_ردیف_بایتها, src2->ne[1]);
    }

    bctx.octx                  = octx;
    bctx.nrows_در_هر_thread      = fastdiv(nrows + n_نخها - 1, &octx->n_نخها_div);
    bctx.مجموع_ردیفها            = nrows;
    bctx.ردیف_شروع             = ردیف_شروع;
    bctx.بلوک_max             = bctx.vtcm_layout.ردیفها_در_هر_میانگیر;
    bctx.src0_ردیف_اندازه_aligned = src0_ردیف_اندازه_aligned;
    bctx.src1_ردیف_اندازه_aligned = src1_ردیف_اندازه_aligned;
    bctx.dst_ردیف_اندازه_aligned  = dst_ردیف_اندازه_aligned;
    bctx.ردیف_اندازه_بایتها        = src0_ردیف_اندازه;

    bctx.src0_dim1_div  = init_fastdiv_مقادیر(src0->ne[1]);
    bctx.src0_dim2_div  = init_fastdiv_مقادیر(src0->ne[2]);
    bctx.src0_dim12_div = init_fastdiv_مقادیر(src0->ne[1] * src0->ne[2]);

    bctx.src1_dim1_div  = init_fastdiv_مقادیر(src1->ne[1]);
    bctx.src1_dim2_div  = init_fastdiv_مقادیر(src1->ne[2]);
    bctx.src1_dim3_div  = init_fastdiv_مقادیر(src1->ne[3]);

    bctx.تقسیم_at_ne01  = kparams->تقسیم_at_ne01;
    bctx.تقسیم_at_ne02  = kparams->تقسیم_at_ne02;

    کارگر_callback_t کارگر_func = عدم;
    فراغ * compute_func = عدم;

    بدل (kparams->kernel_type) {
        حالة HTP_دودویی_KERNEL_SAME_SHAPE:
            کارگر_func = دودویی_thread_وکتور_same_shape;
            إذا (src0_type == HTP_TYPE_F32) {
                بدل (octx->op) {
                    حالة HTP_OP_جمع_کن: compute_func = compute_same_shape_جمع_کن_f32; اكسر;
                    حالة HTP_OP_SUB: compute_func = compute_same_shape_sub_f32; اكسر;
                    حالة HTP_OP_MUL: compute_func = compute_same_shape_mul_f32; اكسر;
                    حالة HTP_OP_DIV: compute_func = compute_same_shape_div_f32; اكسر;
                    افتراضي: اكسر;
                }
            } وإلا إذا (src0_type == HTP_TYPE_F16) {
                بدل (octx->op) {
                    حالة HTP_OP_جمع_کن: compute_func = compute_same_shape_جمع_کن_f16; اكسر;
                    حالة HTP_OP_SUB: compute_func = compute_same_shape_sub_f16; اكسر;
                    حالة HTP_OP_MUL: compute_func = compute_same_shape_mul_f16; اكسر;
                    حالة HTP_OP_DIV: compute_func = compute_same_shape_div_f16; اكسر;
                    افتراضي: اكسر;
                }
            }
            اكسر;
        حالة HTP_دودویی_KERNEL_ردیف_BCAST:
            کارگر_func = دودویی_thread_وکتور_ردیف_پخش;
            إذا (src0_type == HTP_TYPE_F32) {
                بدل (octx->op) {
                    حالة HTP_OP_جمع_کن: compute_func = compute_ردیف_bcast_جمع_کن_f32; اكسر;
                    حالة HTP_OP_SUB: compute_func = compute_ردیف_bcast_sub_f32; اكسر;
                    حالة HTP_OP_MUL: compute_func = compute_ردیف_bcast_mul_f32; اكسر;
                    حالة HTP_OP_DIV: compute_func = compute_ردیف_bcast_div_f32; اكسر;
                    افتراضي: اكسر;
                }
            } وإلا إذا (src0_type == HTP_TYPE_F16) {
                بدل (octx->op) {
                    حالة HTP_OP_جمع_کن: compute_func = compute_ردیف_bcast_جمع_کن_f16; اكسر;
                    حالة HTP_OP_SUB: compute_func = compute_ردیف_bcast_sub_f16; اكسر;
                    حالة HTP_OP_MUL: compute_func = compute_ردیف_bcast_mul_f16; اكسر;
                    حالة HTP_OP_DIV: compute_func = compute_ردیف_bcast_div_f16; اكسر;
                    افتراضي: اكسر;
                }
            }
            اكسر;
        حالة HTP_دودویی_KERNEL_SCALAR:
            کارگر_func = دودویی_thread_scalar;
            إذا (src0_type == HTP_TYPE_F32) {
                بدل (octx->op) {
                    حالة HTP_OP_جمع_کن: compute_func = compute_scalar_جمع_کن_f32; اكسر;
                    حالة HTP_OP_SUB: compute_func = compute_scalar_sub_f32; اكسر;
                    حالة HTP_OP_MUL: compute_func = compute_scalar_mul_f32; اكسر;
                    حالة HTP_OP_DIV: compute_func = compute_scalar_div_f32; اكسر;
                    افتراضي: اكسر;
                }
            } وإلا إذا (src0_type == HTP_TYPE_F16) {
                بدل (octx->op) {
                    حالة HTP_OP_جمع_کن: compute_func = compute_scalar_جمع_کن_f16; اكسر;
                    حالة HTP_OP_SUB: compute_func = compute_scalar_sub_f16; اكسر;
                    حالة HTP_OP_MUL: compute_func = compute_scalar_mul_f16; اكسر;
                    حالة HTP_OP_DIV: compute_func = compute_scalar_div_f16; اكسر;
                    افتراضي: اكسر;
                }
            }
            اكسر;
        حالة HTP_دودویی_KERNEL_COMPLEX:
            کارگر_func = دودویی_thread_وکتور_complex;
            إذا (src0_type == HTP_TYPE_F32) {
                بدل (octx->op) {
                    حالة HTP_OP_جمع_کن: compute_func = compute_complex_جمع_کن_f32; اكسر;
                    حالة HTP_OP_SUB: compute_func = compute_complex_sub_f32; اكسر;
                    حالة HTP_OP_MUL: compute_func = compute_complex_mul_f32; اكسر;
                    حالة HTP_OP_DIV: compute_func = compute_complex_div_f32; اكسر;
                    افتراضي: اكسر;
                }
            } وإلا إذا (src0_type == HTP_TYPE_F16) {
                بدل (octx->op) {
                    حالة HTP_OP_جمع_کن: compute_func = compute_complex_جمع_کن_f16; اكسر;
                    حالة HTP_OP_SUB: compute_func = compute_complex_sub_f16; اكسر;
                    حالة HTP_OP_MUL: compute_func = compute_complex_mul_f16; اكسر;
                    حالة HTP_OP_DIV: compute_func = compute_complex_div_f16; اكسر;
                    افتراضي: اكسر;
                }
            }
            اكسر;
        حالة HTP_دودویی_KERNEL_REPEAT:
            کارگر_func = دودویی_thread_element_repeat;
            إذا (src0_type == HTP_TYPE_F32) {
                بدل (octx->op) {
                    حالة HTP_OP_جمع_کن: compute_func = compute_repeat_جمع_کن_f32; اكسر;
                    حالة HTP_OP_SUB: compute_func = compute_repeat_sub_f32; اكسر;
                    حالة HTP_OP_MUL: compute_func = compute_repeat_mul_f32; اكسر;
                    حالة HTP_OP_DIV: compute_func = compute_repeat_div_f32; اكسر;
                    افتراضي: اكسر;
                }
            } وإلا إذا (src0_type == HTP_TYPE_F16) {
                بدل (octx->op) {
                    حالة HTP_OP_جمع_کن: compute_func = compute_repeat_جمع_کن_f16; اكسر;
                    حالة HTP_OP_SUB: compute_func = compute_repeat_sub_f16; اكسر;
                    حالة HTP_OP_MUL: compute_func = compute_repeat_mul_f16; اكسر;
                    حالة HTP_OP_DIV: compute_func = compute_repeat_div_f16; اكسر;
                    افتراضي: اكسر;
                }
            }
            اكسر;
        حالة HTP_دودویی_KERNEL_جمع_کن_ID:
            إذا (octx->op == HTP_OP_جمع_کن_ID && src0_type == HTP_TYPE_F32) {
                کارگر_func = دودویی_thread_جمع_کن_id_f32;
                compute_func = (فراغ *) compute_جمع_کن_id_f32;
            }
            اكسر;
        افتراضي: اكسر;
    }

    إذا (!کارگر_func || !compute_func) {
        أرجع HTP_وضعیت_NO_SUPPORT;
    }

    bctx.compute = compute_func;

    إذا (kparams->kernel_type == HTP_دودویی_KERNEL_ردیف_BCAST ||
        kparams->kernel_type == HTP_دودویی_KERNEL_SCALAR ||
        kparams->kernel_type == HTP_دودویی_KERNEL_REPEAT ||
        kparams->kernel_type == HTP_دودویی_KERNEL_COMPLEX ||
        kparams->kernel_type == HTP_دودویی_KERNEL_جمع_کن_ID) {
        dma_صف_flush(dma_q);
    }

    work_صف_پیاپی(octx->ctx->work_صف, کارگر_func, &bctx, n_نخها);

    أرجع HTP_وضعیت_OK;
}

صحيح op_دودویی(بنية htp_ops_context * octx) {
    ثابت uint32_t src0_type = octx->src[0]->type;
    إذا ((src0_type == HTP_TYPE_F32) || (src0_type == HTP_TYPE_F16)) {
        أرجع execute_op_دودویی(octx);
    }

    أرجع HTP_وضعیت_NO_SUPPORT;
}
