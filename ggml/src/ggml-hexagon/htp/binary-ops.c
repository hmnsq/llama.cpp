#هدايت clang diagnostic ignored "-Wunused-variable"
#هدايت clang diagnostic ignored "-Wunused-function"
#هدايت clang diagnostic ignored "-Wunused-but-set-variable"

#ضمن <HAP_farf.h>
#ضمن <HAP_perf.h>

#ضمن <math.h>
#ضمن <string.h>

#ضمن "dma-queue.h"
#ضمن "hvx-utils.h"

#وصف_ڪريو GGML_COMMON_DECL_C
#ضمن "ggml-common.h"
#ضمن "hex-common.h"
#ضمن "hex-profile.h"
#ضمن "binary-ops.h"
#ضمن "htp-ctx.h"
#ضمن "htp-ops.h"
#ضمن "htp-tensor.h"

#جيڪڏهن_وصف_نه_ٿيل MIN
#وصف_ڪريو MIN(a, b) ((a) < (b) ? (a) : (b))
#جيڪڏهن_پورو

// Context for binary operations
بنية htp_ٻٽو_context {
    بنية htp_ops_context * octx;
    بنية htp_ٻٽو_vtcm_layout vtcm_layout;
    uint8_t * vtcm_بنياد;

    بنية fastdiv_قدرون src0_dim1_div; // ne01
    بنية fastdiv_قدرون src0_dim2_div; // ne02
    بنية fastdiv_قدرون src0_dim12_div;// ne03

    بنية fastdiv_قدرون src1_dim1_div; // ne11
    بنية fastdiv_قدرون src1_dim2_div; // ne12
    بنية fastdiv_قدرون src1_dim3_div; // ne13

    uint32_t بلاڪ_max;
    uint32_t nrows_في_thread;
    uint32_t ڪل_صفون;
    uint32_t صف_شروعات;
    size_t   src0_صف_سائيز_aligned;
    size_t   src1_صف_سائيز_aligned;
    size_t   dst_صف_سائيز_aligned;
    size_t   صف_سائيز_بائيٽون;

    bool ورهايو_at_ne01;
    bool ورهايو_at_ne02;

    فراغ * compute;
};

#وصف_ڪريو htp_ٻٽو_preamble                        \
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

ساكن مضمن uint32_t calc_بلاڪ_سائيز(بنية htp_ٻٽو_context * bctx, uint32_t ir, uint32_t پڇاڙي_صف, uint32_t ne01, uint32_t ne02) {
    uint32_t i03, i02, i01, rem;
    i03 = fastdiv(ir, &bctx->src0_dim12_div);
    rem = ir - i03 * (ne02 * ne01);
    i02 = fastdiv(rem, &bctx->src0_dim1_div);
    i01 = rem - i02 * ne01;

    uint32_t صفون_کاٻو = پڇاڙي_صف - ir;
    uint32_t بلاڪ_حد_بندي = صفون_کاٻو;

    إذا (bctx->ورهايو_at_ne01) {
        بلاڪ_حد_بندي = MIN(بلاڪ_حد_بندي, ne01 - i01);
    }
    إذا (bctx->ورهايو_at_ne02) {
         uint32_t صفون_اندريون_plane = (ne02 * ne01) - rem;
         بلاڪ_حد_بندي = MIN(بلاڪ_حد_بندي, صفون_اندريون_plane);
    }

    أرجع MIN(bctx->بلاڪ_max, بلاڪ_حد_بندي);
}

// Out-of-line compute micro-kernels

تعريفنوع فراغ (*compute_scalar_t)(
    uint8_t * dst, ثابت uint8_t * src0, ثابت فراغ * s1_جدول,
    uint32_t cur_i11, uint32_t ne11, uint32_t n_صفون,
    size_t dst_stride, size_t src0_stride, uint32_t ne00);

#وصف_ڪريو DEFINE_COMPUTE_SCALAR(NAME, TYPE, HVX_STMT)             \
ساكن فراغ compute_scalar_##NAME(                              \
    uint8_t * dst, ثابت uint8_t * src0, ثابت فراغ * s1_جدول, \
    uint32_t cur_i11, uint32_t ne11, uint32_t n_صفون,           \
    size_t dst_stride, size_t src0_stride, uint32_t ne00) {     \
    ثابت TYPE * جدول = (ثابت TYPE *) s1_جدول;               \
    لكل (uint32_t r = 0; r < n_صفون; r++) {                     \
        uint8_t * r_dst = dst + r * dst_stride;                 \
        ثابت uint8_t * r_src0 = src0 + r * src0_stride;        \
        TYPE val = جدول[cur_i11];                              \
        HVX_STMT;                                               \
        إذا (ne11 > 1 && ++cur_i11 == ne11) {                    \
            cur_i11 = 0;                                        \
        }                                                       \
    }                                                           \
}

DEFINE_COMPUTE_SCALAR(جوڙيو_f32, عائم,    hvx_جوڙيو_scalar_f32_aa(r_dst, r_src0, val, ne00))
DEFINE_COMPUTE_SCALAR(جوڙيو_f16, _Float16, hvx_جوڙيو_scalar_f16_aa(r_dst, r_src0, val, ne00))
DEFINE_COMPUTE_SCALAR(sub_f32, عائم,    hvx_sub_scalar_f32_aa(r_dst, r_src0, val, ne00))
DEFINE_COMPUTE_SCALAR(sub_f16, _Float16, hvx_sub_scalar_f16_aa(r_dst, r_src0, val, ne00))
DEFINE_COMPUTE_SCALAR(mul_f32, عائم,    hvx_mul_scalar_f32_aa(r_dst, r_src0, val, ne00))
DEFINE_COMPUTE_SCALAR(mul_f16, _Float16, hvx_mul_scalar_f16_aa(r_dst, r_src0, val, ne00))
DEFINE_COMPUTE_SCALAR(div_f32, عائم,    hvx_mul_scalar_f32_aa(r_dst, r_src0, 1.0f / (val), ne00))
DEFINE_COMPUTE_SCALAR(div_f16, _Float16, hvx_div_scalar_f16_aa(r_dst, r_src0, val, ne00))

تعريفنوع فراغ (*compute_same_shape_t)(
    uint8_t * dst, ثابت uint8_t * src0, ثابت uint8_t * src1,
    uint32_t n_صفون, size_t dst_stride, size_t src0_stride, size_t src1_stride, uint32_t ne00);

#وصف_ڪريو DEFINE_COMPUTE_SAME_SHAPE(NAME, HVX_FN)                                                  \
ساكن فراغ compute_same_shape_##NAME(                                                           \
    uint8_t * dst, ثابت uint8_t * src0, ثابت uint8_t * src1,                                   \
    uint32_t n_صفون, size_t dst_stride, size_t src0_stride, size_t src1_stride, uint32_t ne00) { \
    لكل (uint32_t r = 0; r < n_صفون; r++) {                                                      \
        HVX_FN(dst + r * dst_stride, src0 + r * src0_stride, src1 + r * src1_stride, ne00);      \
    }                                                                                            \
}

DEFINE_COMPUTE_SAME_SHAPE(جوڙيو_f32, hvx_جوڙيو_f32_aaa)
DEFINE_COMPUTE_SAME_SHAPE(جوڙيو_f16, hvx_جوڙيو_f16_aaa)
DEFINE_COMPUTE_SAME_SHAPE(sub_f32, hvx_sub_f32_aaa)
DEFINE_COMPUTE_SAME_SHAPE(sub_f16, hvx_sub_f16_aaa)
DEFINE_COMPUTE_SAME_SHAPE(mul_f32, hvx_mul_f32_aaa)
DEFINE_COMPUTE_SAME_SHAPE(mul_f16, hvx_mul_f16_aaa)
DEFINE_COMPUTE_SAME_SHAPE(div_f32, hvx_div_f32_aaa)
DEFINE_COMPUTE_SAME_SHAPE(div_f16, hvx_div_f16_aaa)

تعريفنوع فراغ (*compute_صف_bcast_t)(
    uint8_t * dst, ثابت uint8_t * src0, ثابت uint8_t * src1,
    uint32_t n_صفون, size_t dst_stride, size_t src0_stride, uint32_t ne00);

#وصف_ڪريو DEFINE_COMPUTE_صف_BCAST(NAME, HVX_FN)                                               \
ساكن فراغ compute_صف_bcast##NAME(                                                        \
    uint8_t * dst, ثابت uint8_t * src0, ثابت uint8_t * src1,                               \
    uint32_t n_صفون, size_t dst_stride, size_t src0_stride, uint32_t ne00) {                 \
    لكل (uint32_t r = 0; r < n_صفون; r++) {                                                  \
        HVX_FN(dst + r * dst_stride, src0 + r * src0_stride, src1, ne00);                    \
    }                                                                                        \
}

DEFINE_COMPUTE_صف_BCAST(جوڙيو_f32, hvx_جوڙيو_f32_aaa)
DEFINE_COMPUTE_صف_BCAST(جوڙيو_f16, hvx_جوڙيو_f16_aaa)
DEFINE_COMPUTE_صف_BCAST(sub_f32, hvx_sub_f32_aaa)
DEFINE_COMPUTE_صف_BCAST(sub_f16, hvx_sub_f16_aaa)
DEFINE_COMPUTE_صف_BCAST(mul_f32, hvx_mul_f32_aaa)
DEFINE_COMPUTE_صف_BCAST(mul_f16, hvx_mul_f16_aaa)
DEFINE_COMPUTE_صف_BCAST(div_f32, hvx_div_f32_aaa)
DEFINE_COMPUTE_صف_BCAST(div_f16, hvx_div_f16_aaa)

تعريفنوع فراغ (*compute_complex_t)(
    uint8_t * dst, ثابت uint8_t * src0, ثابت uint8_t * src1_plane,
    uint32_t i01, uint32_t ne11, ثابت بنية fastdiv_قدرون * div11, uint32_t src1_stride,
    uint32_t n_صفون, size_t dst_stride, size_t src0_stride, uint32_t ne00);

#وصف_ڪريو DEFINE_COMPUTE_COMPLEX(NAME, HVX_FN)                                                 \
ساكن فراغ compute_complex_##NAME(                                                          \
    uint8_t * dst, ثابت uint8_t * src0, ثابت uint8_t * src1_plane,                         \
    uint32_t i01, uint32_t ne11, ثابت بنية fastdiv_قدرون * div11, uint32_t src1_stride,  \
    uint32_t n_صفون, size_t dst_stride, size_t src0_stride, uint32_t ne00) {                 \
    لكل (uint32_t r = 0; r < n_صفون; r++) {                                                  \
        uint32_t i11 = fastmodulo(i01 + r, ne11, div11);                                     \
        ثابت uint8_t * r_src1 = src1_plane + i11 * src1_stride;                             \
        HVX_FN(dst + r * dst_stride, src0 + r * src0_stride, r_src1, ne00);                  \
    }                                                                                        \
}

DEFINE_COMPUTE_COMPLEX(جوڙيو_f32, hvx_جوڙيو_f32_aau)
DEFINE_COMPUTE_COMPLEX(جوڙيو_f16, hvx_جوڙيو_f16_aau)
DEFINE_COMPUTE_COMPLEX(sub_f32, hvx_sub_f32_aau)
DEFINE_COMPUTE_COMPLEX(sub_f16, hvx_sub_f16_aau)
DEFINE_COMPUTE_COMPLEX(mul_f32, hvx_mul_f32_aau)
DEFINE_COMPUTE_COMPLEX(mul_f16, hvx_mul_f16_aau)
DEFINE_COMPUTE_COMPLEX(div_f32, hvx_div_f32_aau)
DEFINE_COMPUTE_COMPLEX(div_f16, hvx_div_f16_aau)

تعريفنوع فراغ (*compute_repeat_t)(
    uint8_t * dst, ثابت uint8_t * src0, ثابت uint8_t * src1_plane,
    uint32_t i01, uint32_t ne11, ثابت بنية fastdiv_قدرون * div11, uint32_t src1_stride,
    uint32_t n_صفون, size_t dst_stride, size_t src0_stride, uint32_t ne00, uint32_t ne10);

#وصف_ڪريو DEFINE_COMPUTE_REPEAT(NAME, TYPE, HVX_FN)                                            \
ساكن فراغ compute_repeat_##NAME(                                                           \
    uint8_t * dst, ثابت uint8_t * src0, ثابت uint8_t * src1_plane,                         \
    uint32_t i01, uint32_t ne11, ثابت بنية fastdiv_قدرون * div11, uint32_t src1_stride,  \
    uint32_t n_صفون, size_t dst_stride, size_t src0_stride, uint32_t ne00, uint32_t ne10) {  \
    لكل (uint32_t r = 0; r < n_صفون; r++) {                                                  \
        uint32_t i11 = fastmodulo(i01 + r, ne11, div11);                                     \
        ثابت uint8_t * r_src1_صف = src1_plane + i11 * src1_stride;                         \
        uint8_t * r_dst = dst + r * dst_stride;                                              \
        ثابت uint8_t * r_src0 = src0 + r * src0_stride;                                     \
        لكل (uint32_t c = 0; c < ne00; c += ne10) {                                          \
            uint32_t len = MIN(ne10, ne00 - c);                                              \
            HVX_FN(r_dst + c * حجم(TYPE), r_src0 + c * حجم(TYPE), r_src1_صف, len);    \
        }                                                                                    \
    }                                                                                        \
}

DEFINE_COMPUTE_REPEAT(جوڙيو_f32, عائم,    hvx_جوڙيو_f32_uuu)
DEFINE_COMPUTE_REPEAT(جوڙيو_f16, _Float16, hvx_جوڙيو_f16_uuu)
DEFINE_COMPUTE_REPEAT(sub_f32, عائم,    hvx_sub_f32_uuu)
DEFINE_COMPUTE_REPEAT(sub_f16, _Float16, hvx_sub_f16_uuu)
DEFINE_COMPUTE_REPEAT(mul_f32, عائم,    hvx_mul_f32_uuu)
DEFINE_COMPUTE_REPEAT(mul_f16, _Float16, hvx_mul_f16_uuu)
DEFINE_COMPUTE_REPEAT(div_f32, عائم,    hvx_div_f32_uuu)
DEFINE_COMPUTE_REPEAT(div_f16, _Float16, hvx_div_f16_uuu)

تعريفنوع فراغ (*compute_جوڙيو_id_t)(
    uint8_t * dst, ثابت uint8_t * src0, ثابت uint8_t * src1_معلومات, ثابت int32_t * src2_ids,
    uint32_t i01, uint32_t i02, uint32_t ne01, uint32_t src1_stride,
    uint32_t n_صفون, size_t dst_stride, size_t src0_stride, uint32_t ne00);

ساكن فراغ compute_جوڙيو_id_f32(
    uint8_t * dst, ثابت uint8_t * src0, ثابت uint8_t * src1_معلومات, ثابت int32_t * src2_ids,
    uint32_t i01, uint32_t i02, uint32_t ne01, uint32_t src1_stride,
    uint32_t n_صفون, size_t dst_stride, size_t src0_stride, uint32_t ne00) {
    لكل (uint32_t r = 0; r < n_صفون; r++) {
        uint32_t r_i01 = i01 + r;
        ثابت int32_t idx = src2_ids[i02 * ne01 + r_i01];
        إذا (idx < 0) {
            يادگيري_نقل(dst + r * dst_stride, src0 + r * src0_stride, ne00 * حجم(عائم));
            استمر;
        }
        ثابت uint8_t * r_src1 = src1_معلومات + idx * src1_stride;
        ثابت uint8_t * r_src0 = src0 + r * src0_stride;
        uint8_t * r_dst = dst + r * dst_stride;
        hvx_جوڙيو_f32_aaa(r_dst, r_src0, r_src1, ne00);
    }
}

// 1. Scalar src1 in VTCM via DMA (ne10 == 1)
ساكن فراغ ٻٽو_thread_scalar(بنانشان صحيح nth, بنانشان صحيح ith, فراغ * معلومات) {
    بنية htp_ٻٽو_context * bctx = (بنية htp_ٻٽو_context *) معلومات;
    بنية htp_ops_context * octx = bctx->octx;
    htp_ٻٽو_preamble;

    ثابت uint32_t elem_سائيز = (src0->type == HTP_TYPE_F32) ? حجم(عائم) : حجم(_Float16);
    ثابت size_t plane_stride = (size_t) ne12 * ne11 * elem_سائيز;
    ثابت size_t صف_stride   = (size_t) ne11 * elem_سائيز;

    ثابت uint32_t صف_سائيز_بائيٽون = bctx->صف_سائيز_بائيٽون;
    ثابت uint32_t شروعات_صف = bctx->صف_شروعات + bctx->nrows_في_thread * ith;
    ثابت uint32_t پڇاڙي_صف   = MIN(شروعات_صف + bctx->nrows_في_thread, bctx->صف_شروعات + bctx->ڪل_صفون);
    إذا (شروعات_صف >= پڇاڙي_صف) أرجع;

    FARF(مٿاهون_2, "binary-scalar: %d/%d (%u:%u) row-size %u (%u)",
         ith, nth, شروعات_صف, پڇاڙي_صف, nb01, bctx->dst_صف_سائيز_aligned);

    ثابت بنية htp_ٻٽو_vtcm_layout * layout = &bctx->vtcm_layout;
    uint8_t * src0_spad_بنياد = VTCM_LAYOUT_PTR(uint8_t, bctx->vtcm_بنياد, layout->off_src0) + (ith * layout->src0_بائيٽون_في_thread);
    uint8_t * dst_spad_بنياد  = VTCM_LAYOUT_PTR(uint8_t, bctx->vtcm_بنياد, layout->off_dst)  + (ith * layout->dst_بائيٽون_في_thread);
    size_t src0_spad_half    = layout->src0_spad_half_سائيز;
    size_t dst_spad_half     = layout->dst_spad_half_سائيز;
    ثابت فراغ * s1_جدول    = VTCM_LAYOUT_PTR(ثابت فراغ, bctx->vtcm_بنياد, layout->off_src1);

    dma_قطار * dma_q = octx->ctx->dma[ith];
    uint32_t ir_prefetch = شروعات_صف;
    صحيح spad_idx = 0;

    لكل (صحيح k = 0; k < 2 && ir_prefetch < پڇاڙي_صف; k++) {
        uint32_t current_بلاڪ_سائيز = calc_بلاڪ_سائيز(bctx, ir_prefetch, پڇاڙي_صف, ne01, ne02);
        uint32_t i03, i02, i01, rem;
        i03 = fastdiv(ir_prefetch, &bctx->src0_dim12_div);
        rem = ir_prefetch - i03 * (ne02 * ne01);
        i02 = fastdiv(rem, &bctx->src0_dim1_div);
        i01 = rem - i02 * ne01;

        dma_addr_t src0_curr = src0->معلومات + i03 * nb03 + i02 * nb02 + i01 * nb01;
        dma_addr_t dst_curr  = dst->معلومات  + i03 * nb3  + i02 * nb2  + i01 * nb1;

        uint8_t * s0_spad = src0_spad_بنياد + spad_idx * src0_spad_half;
        uint8_t * d_spad  = dst_spad_بنياد  + spad_idx * dst_spad_half;

        dma_قطار_ڌڪو(dma_q, dma_make_معلومات(dst_curr, d_spad), nb1, bctx->dst_صف_سائيز_aligned, صف_سائيز_بائيٽون, 0);
        dma_قطار_ڌڪو(dma_q, dma_make_معلومات(s0_spad, src0_curr), bctx->src0_صف_سائيز_aligned, nb01, صف_سائيز_بائيٽون,
                                                             current_بلاڪ_سائيز);
        ir_prefetch += current_بلاڪ_سائيز;
        spad_idx ^= 1;
    }

    بنية htp_thread_trace * tr = &octx->ctx->trace[ith];
    compute_scalar_t compute = (compute_scalar_t) bctx->compute;

    لكل (uint32_t ir = شروعات_صف; ir < پڇاڙي_صف; ) {
        uint32_t current_بلاڪ_سائيز = calc_بلاڪ_سائيز(bctx, ir, پڇاڙي_صف, ne01, ne02);

        uint8_t * d_spad  = (uint8_t *) dma_قطار_ڪڍو(dma_q).src;
        uint8_t * s0_spad = (uint8_t *) dma_قطار_ڪڍو(dma_q).dst;

        uint32_t i03, i02, i01, rem;
        i03 = fastdiv(ir, &bctx->src0_dim12_div);
        rem = ir - i03 * (ne02 * ne01);
        i02 = fastdiv(rem, &bctx->src0_dim1_div);
        i01 = rem - i02 * ne01;

        uint32_t i13 = fastmodulo(i03, ne13, &bctx->src1_dim3_div);
        uint32_t i12 = fastmodulo(i02, ne12, &bctx->src1_dim2_div);
        uint32_t i11 = fastmodulo(i01, ne11, &bctx->src1_dim1_div);
        ثابت uint8_t * scalar_جدول = (ثابت uint8_t *) s1_جدول +
            i13 * plane_stride + i12 * صف_stride;

        htp_trace_event_شروعات(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);
        compute(d_spad, s0_spad, scalar_جدول, i11, ne11, current_بلاڪ_سائيز,
                bctx->dst_صف_سائيز_aligned, bctx->src0_صف_سائيز_aligned, ne00);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);

        dma_addr_t dst_curr = dst->معلومات + i03 * nb3 + i02 * nb2 + i01 * nb1;
        dma_قطار_ڌڪو(dma_q, dma_make_معلومات(dst_curr, d_spad), nb1, bctx->dst_صف_سائيز_aligned, صف_سائيز_بائيٽون,
                                                                current_بلاڪ_سائيز);

        إذا (ir_prefetch < پڇاڙي_صف) {
            uint32_t ايندڙ_بلاڪ_سائيز = calc_بلاڪ_سائيز(bctx, ir_prefetch, پڇاڙي_صف, ne01, ne02);
            uint32_t p03, p02, p01, prem;
            p03 = fastdiv(ir_prefetch, &bctx->src0_dim12_div);
            prem = ir_prefetch - p03 * (ne02 * ne01);
            p02 = fastdiv(prem, &bctx->src0_dim1_div);
            p01 = prem - p02 * ne01;
            dma_addr_t s0_ايندڙ = src0->معلومات + p03 * nb03 + p02 * nb02 + p01 * nb01;
            dma_قطار_ڌڪو(dma_q, dma_make_معلومات(s0_spad, s0_ايندڙ), bctx->src0_صف_سائيز_aligned, nb01, صف_سائيز_بائيٽون,
                                                               ايندڙ_بلاڪ_سائيز);
            ir_prefetch += ايندڙ_بلاڪ_سائيز;
        }
        ir += current_بلاڪ_سائيز;
    }

    dma_قطار_flush(dma_q);
}

// 2. Vector Same Shape (ne1x == ne0x) or Simple Broadcast
ساكن فراغ ٻٽو_thread_ويڪٽر_same_shape(بنانشان صحيح nth, بنانشان صحيح ith, فراغ * معلومات) {
    بنية htp_ٻٽو_context * bctx = (بنية htp_ٻٽو_context *) معلومات;
    بنية htp_ops_context * octx = bctx->octx;
    htp_ٻٽو_preamble;

    ثابت uint32_t صف_سائيز_بائيٽون = bctx->صف_سائيز_بائيٽون;
    ثابت uint32_t شروعات_صف = bctx->صف_شروعات + bctx->nrows_في_thread * ith;
    ثابت uint32_t پڇاڙي_صف   = MIN(شروعات_صف + bctx->nrows_في_thread, bctx->صف_شروعات + bctx->ڪل_صفون);
    إذا (شروعات_صف >= پڇاڙي_صف) أرجع;

    FARF(مٿاهون_2, "binary-same-shape: %d/%d (%u:%u) row-size %u (%u)",
         ith, nth, شروعات_صف, پڇاڙي_صف, nb01, bctx->dst_صف_سائيز_aligned);

    ثابت بنية htp_ٻٽو_vtcm_layout * layout = &bctx->vtcm_layout;
    uint8_t * src0_spad_بنياد = VTCM_LAYOUT_PTR(uint8_t, bctx->vtcm_بنياد, layout->off_src0) + (ith * layout->src0_بائيٽون_في_thread);
    uint8_t * src1_spad_بنياد = VTCM_LAYOUT_PTR(uint8_t, bctx->vtcm_بنياد, layout->off_src1) + (ith * layout->src1_بائيٽون_في_thread);
    uint8_t * dst_spad_بنياد  = VTCM_LAYOUT_PTR(uint8_t, bctx->vtcm_بنياد, layout->off_dst)  + (ith * layout->dst_بائيٽون_في_thread);

    size_t src0_spad_half = layout->src0_spad_half_سائيز;
    size_t src1_spad_half = layout->src1_spad_half_سائيز;
    size_t dst_spad_half  = layout->dst_spad_half_سائيز;

    dma_قطار * dma_q = octx->ctx->dma[ith];
    uint32_t ir_prefetch = شروعات_صف;
    صحيح spad_idx = 0;

    لكل (صحيح k = 0; k < 2 && ir_prefetch < پڇاڙي_صف; k++) {
        uint32_t current_بلاڪ_سائيز = calc_بلاڪ_سائيز(bctx, ir_prefetch, پڇاڙي_صف, ne01, ne02);
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

        uint8_t * s0_spad = src0_spad_بنياد + spad_idx * src0_spad_half;
        uint8_t * s1_spad = src1_spad_بنياد + spad_idx * src1_spad_half;
        uint8_t * d_spad  = dst_spad_بنياد  + spad_idx * dst_spad_half;

        dma_قطار_ڌڪو(dma_q, dma_make_معلومات(dst_curr, d_spad), nb1, bctx->dst_صف_سائيز_aligned, صف_سائيز_بائيٽون, 0);
        dma_قطار_ڌڪو(dma_q, dma_make_معلومات(s0_spad, src0_curr), bctx->src0_صف_سائيز_aligned, nb01, صف_سائيز_بائيٽون, current_بلاڪ_سائيز);
        dma_قطار_ڌڪو(dma_q, dma_make_معلومات(s1_spad, src1_curr), bctx->src1_صف_سائيز_aligned, nb11, صف_سائيز_بائيٽون, current_بلاڪ_سائيز);
        ir_prefetch += current_بلاڪ_سائيز;
        spad_idx ^= 1;
    }

    بنية htp_thread_trace * tr = &octx->ctx->trace[ith];
    compute_same_shape_t compute = (compute_same_shape_t) bctx->compute;

    لكل (uint32_t ir = شروعات_صف; ir < پڇاڙي_صف; ) {
        uint32_t current_بلاڪ_سائيز = calc_بلاڪ_سائيز(bctx, ir, پڇاڙي_صف, ne01, ne02);
        uint8_t * d_spad  = (uint8_t *) dma_قطار_ڪڍو(dma_q).src;
        uint8_t * s0_spad = (uint8_t *) dma_قطار_ڪڍو(dma_q).dst;
        uint8_t * s1_spad = (uint8_t *) dma_قطار_ڪڍو(dma_q).dst;

        htp_trace_event_شروعات(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);
        compute(d_spad, s0_spad, s1_spad, current_بلاڪ_سائيز,
                bctx->dst_صف_سائيز_aligned, bctx->src0_صف_سائيز_aligned, bctx->src1_صف_سائيز_aligned, ne00);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);

        uint32_t i03, i02, i01, rem;
        i03 = fastdiv(ir, &bctx->src0_dim12_div);
        rem = ir - i03 * (ne02 * ne01);
        i02 = fastdiv(rem, &bctx->src0_dim1_div);
        i01 = rem - i02 * ne01;
        dma_addr_t dst_curr = dst->معلومات + i03 * nb3 + i02 * nb2 + i01 * nb1;
        dma_قطار_ڌڪو(dma_q, dma_make_معلومات(dst_curr, d_spad), nb1, bctx->dst_صف_سائيز_aligned, صف_سائيز_بائيٽون, current_بلاڪ_سائيز);

        إذا (ir_prefetch < پڇاڙي_صف) {
            uint32_t ايندڙ_بلاڪ_سائيز = calc_بلاڪ_سائيز(bctx, ir_prefetch, پڇاڙي_صف, ne01, ne02);
            uint32_t p03, p02, p01, prem;
            p03 = fastdiv(ir_prefetch, &bctx->src0_dim12_div);
            prem = ir_prefetch - p03 * (ne02 * ne01);
            p02 = fastdiv(prem, &bctx->src0_dim1_div);
            p01 = prem - p02 * ne01;

            uint32_t p13 = (ne13 == 1) ? 0 : p03;
            uint32_t p12 = (ne12 == 1) ? 0 : p02;
            uint32_t p11 = (ne11 == 1) ? 0 : p01;

            dma_addr_t s0_ايندڙ = src0->معلومات + p03 * nb03 + p02 * nb02 + p01 * nb01;
            dma_addr_t s1_ايندڙ = src1->معلومات + p13 * nb13 + p12 * nb12 + p11 * nb11;

            dma_قطار_ڌڪو(dma_q, dma_make_معلومات(s0_spad, s0_ايندڙ), bctx->src0_صف_سائيز_aligned, nb01, صف_سائيز_بائيٽون, ايندڙ_بلاڪ_سائيز);
            dma_قطار_ڌڪو(dma_q, dma_make_معلومات(s1_spad, s1_ايندڙ), bctx->src1_صف_سائيز_aligned, nb11, صف_سائيز_بائيٽون, ايندڙ_بلاڪ_سائيز);

            ir_prefetch += ايندڙ_بلاڪ_سائيز;
        }
        ir += current_بلاڪ_سائيز;
    }

    dma_قطار_flush(dma_q);
}

// 3. Row Broadcast (ne11 == 1, ne12 == 1, single row src1)
ساكن فراغ ٻٽو_thread_ويڪٽر_صف_نشر(بنانشان صحيح nth, بنانشان صحيح ith, فراغ * معلومات) {
    بنية htp_ٻٽو_context * bctx = (بنية htp_ٻٽو_context *) معلومات;
    بنية htp_ops_context * octx = bctx->octx;
    htp_ٻٽو_preamble;

    ثابت uint32_t صف_سائيز_بائيٽون = bctx->صف_سائيز_بائيٽون;
    ثابت uint32_t شروعات_صف  = bctx->صف_شروعات + bctx->nrows_في_thread * ith;
    ثابت uint32_t پڇاڙي_صف    = MIN(شروعات_صف + bctx->nrows_في_thread, bctx->صف_شروعات + bctx->ڪل_صفون);
    إذا (شروعات_صف >= پڇاڙي_صف) أرجع;

    FARF(مٿاهون_2, "binary-row-bcast: %d/%d (%u:%u) row-size %u (%u)",
         ith, nth, شروعات_صف, پڇاڙي_صف, nb01, bctx->dst_صف_سائيز_aligned);

    ثابت بنية htp_ٻٽو_vtcm_layout * layout = &bctx->vtcm_layout;
    uint8_t * src0_spad_بنياد = VTCM_LAYOUT_PTR(uint8_t, bctx->vtcm_بنياد, layout->off_src0) + (ith * layout->src0_بائيٽون_في_thread);
    uint8_t * dst_spad_بنياد  = VTCM_LAYOUT_PTR(uint8_t, bctx->vtcm_بنياد, layout->off_dst)  + (ith * layout->dst_بائيٽون_في_thread);

    size_t src0_spad_half = layout->src0_spad_half_سائيز;
    size_t dst_spad_half  = layout->dst_spad_half_سائيز;

    dma_قطار * dma_q = octx->ctx->dma[ith];
    uint32_t ir_prefetch = شروعات_صف;
    صحيح spad_idx = 0;

    فراغ * s1_ptr = VTCM_LAYOUT_PTR(فراغ, bctx->vtcm_بنياد, layout->off_src1);

    لكل (صحيح k = 0; k < 2 && ir_prefetch < پڇاڙي_صف; k++) {
        uint32_t current_بلاڪ_سائيز = calc_بلاڪ_سائيز(bctx, ir_prefetch, پڇاڙي_صف, ne01, ne02);
        uint32_t i03 = fastdiv(ir_prefetch, &bctx->src0_dim12_div);
        uint32_t rem = ir_prefetch - i03 * (ne02 * ne01);
        uint32_t i02 = fastdiv(rem, &bctx->src0_dim1_div);
        uint32_t i01 = rem - i02 * ne01;

        dma_addr_t src0_curr = src0->معلومات + i03 * nb03 + i02 * nb02 + i01 * nb01;
        dma_addr_t dst_curr  = dst->معلومات  + i03 * nb3  + i02 * nb2  + i01 * nb1;

        uint8_t * s0_spad = src0_spad_بنياد + spad_idx * src0_spad_half;
        uint8_t * d_spad  = dst_spad_بنياد  + spad_idx * dst_spad_half;

        dma_قطار_ڌڪو(dma_q, dma_make_معلومات(dst_curr, d_spad), nb1, bctx->dst_صف_سائيز_aligned, صف_سائيز_بائيٽون, 0);
        dma_قطار_ڌڪو(dma_q, dma_make_معلومات(s0_spad, src0_curr), bctx->src0_صف_سائيز_aligned, nb01, صف_سائيز_بائيٽون, current_بلاڪ_سائيز);
        ir_prefetch += current_بلاڪ_سائيز;
        spad_idx ^= 1;
    }

    بنية htp_thread_trace * tr = &octx->ctx->trace[ith];
    compute_صف_bcast_t compute = (compute_صف_bcast_t) bctx->compute;

    لكل (uint32_t ir = شروعات_صف; ir < پڇاڙي_صف; ) {
        uint32_t current_بلاڪ_سائيز = calc_بلاڪ_سائيز(bctx, ir, پڇاڙي_صف, ne01, ne02);
        uint8_t * d_spad  = (uint8_t *) dma_قطار_ڪڍو(dma_q).src;
        uint8_t * s0_spad = (uint8_t *) dma_قطار_ڪڍو(dma_q).dst;

        htp_trace_event_شروعات(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);
        compute(d_spad, s0_spad, (ثابت uint8_t *)s1_ptr, current_بلاڪ_سائيز,
                bctx->dst_صف_سائيز_aligned, bctx->src0_صف_سائيز_aligned, ne00);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);

        uint32_t i03 = fastdiv(ir, &bctx->src0_dim12_div);
        uint32_t rem = ir - i03 * (ne02 * ne01);
        uint32_t i02 = fastdiv(rem, &bctx->src0_dim1_div);
        uint32_t i01 = rem - i02 * ne01;
        dma_addr_t dst_curr = dst->معلومات + i03 * nb3 + i02 * nb2 + i01 * nb1;
        dma_قطار_ڌڪو(dma_q, dma_make_معلومات(dst_curr, d_spad), nb1, bctx->dst_صف_سائيز_aligned, صف_سائيز_بائيٽون, current_بلاڪ_سائيز);

        إذا (ir_prefetch < پڇاڙي_صف) {
            uint32_t ايندڙ_بلاڪ_سائيز = calc_بلاڪ_سائيز(bctx, ir_prefetch, پڇاڙي_صف, ne01, ne02);
            uint32_t p03, p02, p01, prem;
            p03  = fastdiv(ir_prefetch, &bctx->src0_dim12_div);
            prem = ir_prefetch - p03 * (ne02 * ne01);
            p02  = fastdiv(prem, &bctx->src0_dim1_div);
            p01  = prem - p02 * ne01;
            dma_addr_t s0_ايندڙ = src0->معلومات + p03 * nb03 + p02 * nb02 + p01 * nb01;
            dma_قطار_ڌڪو(dma_q, dma_make_معلومات(s0_spad, s0_ايندڙ), bctx->src0_صف_سائيز_aligned, nb01, صف_سائيز_بائيٽون, ايندڙ_بلاڪ_سائيز);
            ir_prefetch += ايندڙ_بلاڪ_سائيز;
        }
        ir += current_بلاڪ_سائيز;
    }

    dma_قطار_flush(dma_q);
}

// 4. Vector Complex (ne10 == ne00, complex broadcast)
ساكن فراغ ٻٽو_thread_ويڪٽر_complex(بنانشان صحيح nth, بنانشان صحيح ith, فراغ * معلومات) {
    بنية htp_ٻٽو_context * bctx = (بنية htp_ٻٽو_context *) معلومات;
    بنية htp_ops_context * octx = bctx->octx;
    htp_ٻٽو_preamble;

    ثابت uint32_t صف_سائيز_بائيٽون = bctx->صف_سائيز_بائيٽون;
    ثابت uint32_t شروعات_صف  = bctx->صف_شروعات + bctx->nrows_في_thread * ith;
    ثابت uint32_t پڇاڙي_صف    = MIN(شروعات_صف + bctx->nrows_في_thread, bctx->صف_شروعات + bctx->ڪل_صفون);
    إذا (شروعات_صف >= پڇاڙي_صف) أرجع;

    FARF(مٿاهون_2, "binary-complex: %d/%d (%u:%u) row-size %u (%u)",
         ith, nth, شروعات_صف, پڇاڙي_صف, nb01, bctx->dst_صف_سائيز_aligned);

    ثابت بنية htp_ٻٽو_vtcm_layout * layout = &bctx->vtcm_layout;
    uint8_t * src0_spad_بنياد = VTCM_LAYOUT_PTR(uint8_t, bctx->vtcm_بنياد, layout->off_src0) + (ith * layout->src0_بائيٽون_في_thread);
    uint8_t * dst_spad_بنياد  = VTCM_LAYOUT_PTR(uint8_t, bctx->vtcm_بنياد, layout->off_dst)  + (ith * layout->dst_بائيٽون_في_thread);
    size_t src0_spad_half    = layout->src0_spad_half_سائيز;
    size_t dst_spad_half     = layout->dst_spad_half_سائيز;
    ثابت uint8_t * vtcm_src1 = VTCM_LAYOUT_PTR(ثابت uint8_t, bctx->vtcm_بنياد, layout->off_src1);

    ثابت uint32_t elem_سائيز = (src0->type == HTP_TYPE_F32) ? حجم(عائم) : حجم(_Float16);
    ثابت size_t vtcm_صف_stride   = (size_t) ne10 * elem_سائيز;
    ثابت size_t vtcm_plane_stride = (size_t) ne11 * vtcm_صف_stride;
    ثابت size_t vtcm_vol_stride   = (size_t) ne12 * vtcm_plane_stride;

    dma_قطار * dma_q = octx->ctx->dma[ith];
    uint32_t ir_prefetch = شروعات_صف;
    صحيح spad_idx = 0;

    لكل (صحيح k = 0; k < 2 && ir_prefetch < پڇاڙي_صف; k++) {
        uint32_t current_بلاڪ_سائيز = calc_بلاڪ_سائيز(bctx, ir_prefetch, پڇاڙي_صف, ne01, ne02);
        uint32_t i03 = fastdiv(ir_prefetch, &bctx->src0_dim12_div);
        uint32_t rem = ir_prefetch - i03 * (ne02 * ne01);
        uint32_t i02 = fastdiv(rem, &bctx->src0_dim1_div);
        uint32_t i01 = rem - i02 * ne01;

        dma_addr_t src0_curr = src0->معلومات + i03 * nb03 + i02 * nb02 + i01 * nb01;
        dma_addr_t dst_curr  = dst->معلومات  + i03 * nb3  + i02 * nb2  + i01 * nb1;

        uint8_t * s0_spad = src0_spad_بنياد + spad_idx * src0_spad_half;
        uint8_t * d_spad  = dst_spad_بنياد  + spad_idx * dst_spad_half;

        dma_قطار_ڌڪو(dma_q, dma_make_معلومات(dst_curr, d_spad), nb1, bctx->dst_صف_سائيز_aligned, صف_سائيز_بائيٽون, 0);
        dma_قطار_ڌڪو(dma_q, dma_make_معلومات(s0_spad, src0_curr), bctx->src0_صف_سائيز_aligned, nb01, صف_سائيز_بائيٽون, current_بلاڪ_سائيز);
        ir_prefetch += current_بلاڪ_سائيز;
        spad_idx ^= 1;
    }

    بنية htp_thread_trace * tr = &octx->ctx->trace[ith];
    compute_complex_t compute = (compute_complex_t) bctx->compute;

    لكل (uint32_t ir = شروعات_صف; ir < پڇاڙي_صف; ) {
        uint32_t current_بلاڪ_سائيز = calc_بلاڪ_سائيز(bctx, ir, پڇاڙي_صف, ne01, ne02);
        uint8_t * d_spad = (uint8_t *) dma_قطار_ڪڍو(dma_q).src;
        uint8_t * s0_spad = (uint8_t *) dma_قطار_ڪڍو(dma_q).dst;

        uint32_t i03 = fastdiv(ir, &bctx->src0_dim12_div);
        uint32_t rem = ir - i03 * (ne02 * ne01);
        uint32_t i02 = fastdiv(rem, &bctx->src0_dim1_div);
        uint32_t i01 = rem - i02 * ne01;

        uint32_t i13 = fastmodulo(i03, ne13, &bctx->src1_dim3_div);
        uint32_t i12 = fastmodulo(i02, ne12, &bctx->src1_dim2_div);
        ثابت uint8_t * src1_plane = vtcm_src1 + i13 * vtcm_vol_stride + i12 * vtcm_plane_stride;

        htp_trace_event_شروعات(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);
        compute(d_spad, s0_spad, src1_plane, i01, ne11, &bctx->src1_dim1_div, vtcm_صف_stride,
                current_بلاڪ_سائيز, bctx->dst_صف_سائيز_aligned, bctx->src0_صف_سائيز_aligned, ne00);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);

        dma_addr_t dst_curr = dst->معلومات + i03 * nb3 + i02 * nb2 + i01 * nb1;
        dma_قطار_ڌڪو(dma_q, dma_make_معلومات(dst_curr, d_spad), nb1, bctx->dst_صف_سائيز_aligned, صف_سائيز_بائيٽون, current_بلاڪ_سائيز);

        إذا (ir_prefetch < پڇاڙي_صف) {
            uint32_t ايندڙ_بلاڪ_سائيز = calc_بلاڪ_سائيز(bctx, ir_prefetch, پڇاڙي_صف, ne01, ne02);
            uint32_t p03, p02, p01, prem;
            p03  = fastdiv(ir_prefetch, &bctx->src0_dim12_div);
            prem = ir_prefetch - p03 * (ne02 * ne01);
            p02  = fastdiv(prem, &bctx->src0_dim1_div);
            p01  = prem - p02 * ne01;
            dma_addr_t s0_ايندڙ = src0->معلومات + p03 * nb03 + p02 * nb02 + p01 * nb01;
            dma_قطار_ڌڪو(dma_q, dma_make_معلومات(s0_spad, s0_ايندڙ), bctx->src0_صف_سائيز_aligned, nb01, صف_سائيز_بائيٽون, ايندڙ_بلاڪ_سائيز);
            ir_prefetch += ايندڙ_بلاڪ_سائيز;
        }
        ir += current_بلاڪ_سائيز;
    }

    dma_قطار_flush(dma_q);
}

// 5. Element Repeat (ne10 != ne00)
ساكن فراغ ٻٽو_thread_element_repeat(بنانشان صحيح nth, بنانشان صحيح ith, فراغ * معلومات) {
    بنية htp_ٻٽو_context * bctx = (بنية htp_ٻٽو_context *) معلومات;
    بنية htp_ops_context * octx = bctx->octx;
    htp_ٻٽو_preamble;

    ثابت uint32_t صف_سائيز_بائيٽون = bctx->صف_سائيز_بائيٽون;
    ثابت uint32_t شروعات_صف  = bctx->صف_شروعات + bctx->nrows_في_thread * ith;
    ثابت uint32_t پڇاڙي_صف    = MIN(شروعات_صف + bctx->nrows_في_thread, bctx->صف_شروعات + bctx->ڪل_صفون);
    إذا (شروعات_صف >= پڇاڙي_صف) أرجع;

    ثابت بنية htp_ٻٽو_vtcm_layout * layout = &bctx->vtcm_layout;
    uint8_t * src0_spad_بنياد = VTCM_LAYOUT_PTR(uint8_t, bctx->vtcm_بنياد, layout->off_src0) + (ith * layout->src0_بائيٽون_في_thread);
    uint8_t * dst_spad_بنياد  = VTCM_LAYOUT_PTR(uint8_t, bctx->vtcm_بنياد, layout->off_dst)  + (ith * layout->dst_بائيٽون_في_thread);
    size_t src0_spad_half    = layout->src0_spad_half_سائيز;
    size_t dst_spad_half     = layout->dst_spad_half_سائيز;
    ثابت uint8_t * vtcm_src1 = VTCM_LAYOUT_PTR(ثابت uint8_t, bctx->vtcm_بنياد, layout->off_src1);

    ثابت uint32_t elem_سائيز = (src0->type == HTP_TYPE_F32) ? حجم(عائم) : حجم(_Float16);
    ثابت size_t vtcm_صف_stride   = (size_t) ne10 * elem_سائيز;
    ثابت size_t vtcm_plane_stride = (size_t) ne11 * vtcm_صف_stride;
    ثابت size_t vtcm_vol_stride   = (size_t) ne12 * vtcm_plane_stride;

    FARF(مٿاهون_2, "binary-repeat: %d/%d (%u:%u) row-size %u (%u)",
         ith, nth, شروعات_صف, پڇاڙي_صف, nb01, bctx->dst_صف_سائيز_aligned);

    dma_قطار * dma_q = octx->ctx->dma[ith];
    uint32_t ir_prefetch = شروعات_صف;
    صحيح spad_idx = 0;

    لكل (صحيح k = 0; k < 2 && ir_prefetch < پڇاڙي_صف; k++) {
        uint32_t current_بلاڪ_سائيز = calc_بلاڪ_سائيز(bctx, ir_prefetch, پڇاڙي_صف, ne01, ne02);
        uint32_t i03 = fastdiv(ir_prefetch, &bctx->src0_dim12_div);
        uint32_t rem = ir_prefetch - i03 * (ne02 * ne01);
        uint32_t i02 = fastdiv(rem, &bctx->src0_dim1_div);
        uint32_t i01 = rem - i02 * ne01;

        dma_addr_t src0_curr = src0->معلومات + i03 * nb03 + i02 * nb02 + i01 * nb01;
        dma_addr_t dst_curr  = dst->معلومات  + i03 * nb3  + i02 * nb2  + i01 * nb1;

        uint8_t * s0_spad = src0_spad_بنياد + spad_idx * src0_spad_half;
        uint8_t * d_spad  = dst_spad_بنياد  + spad_idx * dst_spad_half;

        dma_قطار_ڌڪو(dma_q, dma_make_معلومات(dst_curr, d_spad), nb1, bctx->dst_صف_سائيز_aligned, صف_سائيز_بائيٽون, 0);
        dma_قطار_ڌڪو(dma_q, dma_make_معلومات(s0_spad, src0_curr), bctx->src0_صف_سائيز_aligned, nb01, صف_سائيز_بائيٽون, current_بلاڪ_سائيز);
        ir_prefetch += current_بلاڪ_سائيز;
        spad_idx ^= 1;
    }

    بنية htp_thread_trace * tr = &octx->ctx->trace[ith];
    compute_repeat_t compute = (compute_repeat_t) bctx->compute;

    لكل (uint32_t ir = شروعات_صف; ir < پڇاڙي_صف; ) {
        uint32_t current_بلاڪ_سائيز = calc_بلاڪ_سائيز(bctx, ir, پڇاڙي_صف, ne01, ne02);
        uint8_t * d_spad  = (uint8_t *) dma_قطار_ڪڍو(dma_q).src;
        uint8_t * s0_spad = (uint8_t *) dma_قطار_ڪڍو(dma_q).dst;

        uint32_t i03 = fastdiv(ir, &bctx->src0_dim12_div);
        uint32_t rem = ir - i03 * (ne02 * ne01);
        uint32_t i02 = fastdiv(rem, &bctx->src0_dim1_div);
        uint32_t i01 = rem - i02 * ne01;

        uint32_t i13 = fastmodulo(i03, ne13, &bctx->src1_dim3_div);
        uint32_t i12 = fastmodulo(i02, ne12, &bctx->src1_dim2_div);
        ثابت uint8_t * src1_plane = vtcm_src1 + i13 * vtcm_vol_stride + i12 * vtcm_plane_stride;

        htp_trace_event_شروعات(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);
        compute(d_spad, s0_spad, src1_plane, i01, ne11, &bctx->src1_dim1_div, vtcm_صف_stride,
                current_بلاڪ_سائيز, bctx->dst_صف_سائيز_aligned, bctx->src0_صف_سائيز_aligned, ne00, ne10);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);

        dma_addr_t dst_curr = dst->معلومات + i03 * nb3 + i02 * nb2 + i01 * nb1;
        dma_قطار_ڌڪو(dma_q, dma_make_معلومات(dst_curr, d_spad), nb1, bctx->dst_صف_سائيز_aligned, صف_سائيز_بائيٽون, current_بلاڪ_سائيز);

        إذا (ir_prefetch < پڇاڙي_صف) {
            uint32_t ايندڙ_بلاڪ_سائيز = calc_بلاڪ_سائيز(bctx, ir_prefetch, پڇاڙي_صف, ne01, ne02);
            uint32_t p03, p02, p01, prem;
            p03  = fastdiv(ir_prefetch, &bctx->src0_dim12_div);
            prem = ir_prefetch - p03 * (ne02 * ne01);
            p02  = fastdiv(prem, &bctx->src0_dim1_div);
            p01  = prem - p02 * ne01;
            dma_addr_t s0_ايندڙ = src0->معلومات + p03 * nb03 + p02 * nb02 + p01 * nb01;
            dma_قطار_ڌڪو(dma_q, dma_make_معلومات(s0_spad, s0_ايندڙ), bctx->src0_صف_سائيز_aligned, nb01, صف_سائيز_بائيٽون, ايندڙ_بلاڪ_سائيز);
            ir_prefetch += ايندڙ_بلاڪ_سائيز;
        }
        ir += current_بلاڪ_سائيز;
    }

    dma_قطار_flush(dma_q);
}

// 6. ADD_ID (src1 gathered via src2 indices)
ساكن فراغ ٻٽو_thread_جوڙيو_id_f32(بنانشان صحيح nth, بنانشان صحيح ith, فراغ * معلومات) {
    بنية htp_ٻٽو_context * bctx = (بنية htp_ٻٽو_context *) معلومات;
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
    ثابت uint32_t src1_stride = bctx->src1_صف_سائيز_aligned;

    ثابت uint32_t nb1 = dst->nb[1];
    ثابت uint32_t nb2 = dst->nb[2];
    ثابت uint32_t nb3 = dst->nb[3];

    ثابت uint32_t صف_سائيز_بائيٽون = bctx->صف_سائيز_بائيٽون;
    ثابت uint32_t شروعات_صف = bctx->صف_شروعات + bctx->nrows_في_thread * ith;
    ثابت uint32_t پڇاڙي_صف   = MIN(شروعات_صف + bctx->nrows_في_thread, bctx->صف_شروعات + bctx->ڪل_صفون);
    إذا (شروعات_صف >= پڇاڙي_صف) أرجع;

    ثابت بنية htp_ٻٽو_vtcm_layout * layout = &bctx->vtcm_layout;
    uint8_t * src0_spad_بنياد = VTCM_LAYOUT_PTR(uint8_t, bctx->vtcm_بنياد, layout->off_src0) + (ith * layout->src0_بائيٽون_في_thread);
    uint8_t * dst_spad_بنياد  = VTCM_LAYOUT_PTR(uint8_t, bctx->vtcm_بنياد, layout->off_dst)  + (ith * layout->dst_بائيٽون_في_thread);
    ثابت uint8_t * vtcm_src1 = VTCM_LAYOUT_PTR(ثابت uint8_t, bctx->vtcm_بنياد, layout->off_src1);
    ثابت int32_t * vtcm_src2 = VTCM_LAYOUT_PTR(ثابت int32_t, bctx->vtcm_بنياد, layout->off_src2);
    size_t src0_spad_half    = layout->src0_spad_half_سائيز;
    size_t dst_spad_half     = layout->dst_spad_half_سائيز;

    dma_قطار * dma_q = octx->ctx->dma[ith];
    uint32_t ir_prefetch = شروعات_صف;
    صحيح spad_idx = 0;

    لكل (صحيح k = 0; k < 2 && ir_prefetch < پڇاڙي_صف; k++) {
        uint32_t current_بلاڪ_سائيز = calc_بلاڪ_سائيز(bctx, ir_prefetch, پڇاڙي_صف, ne01, ne02);
        uint32_t i03 = fastdiv(ir_prefetch, &bctx->src0_dim12_div);
        uint32_t rem = ir_prefetch - i03 * (ne02 * ne01);
        uint32_t i02 = fastdiv(rem, &bctx->src0_dim1_div);
        uint32_t i01 = rem - i02 * ne01;

        dma_addr_t src0_curr = src0->معلومات + i03 * nb03 + i02 * nb02 + i01 * nb01;
        dma_addr_t dst_curr  = dst->معلومات  + i03 * nb3  + i02 * nb2  + i01 * nb1;

        uint8_t * s0_spad = src0_spad_بنياد + spad_idx * src0_spad_half;
        uint8_t * d_spad  = dst_spad_بنياد  + spad_idx * dst_spad_half;

        dma_قطار_ڌڪو(dma_q, dma_make_معلومات(dst_curr, d_spad), nb1, bctx->dst_صف_سائيز_aligned, صف_سائيز_بائيٽون, 0);
        dma_قطار_ڌڪو(dma_q, dma_make_معلومات(s0_spad, src0_curr), bctx->src0_صف_سائيز_aligned, nb01, صف_سائيز_بائيٽون, current_بلاڪ_سائيز);
        ir_prefetch += current_بلاڪ_سائيز;
        spad_idx ^= 1;
    }

    بنية htp_thread_trace * tr = &octx->ctx->trace[ith];

    لكل (uint32_t ir = شروعات_صف; ir < پڇاڙي_صف; ) {
        uint32_t current_بلاڪ_سائيز = calc_بلاڪ_سائيز(bctx, ir, پڇاڙي_صف, ne01, ne02);
        uint8_t * d_spad = (uint8_t *) dma_قطار_ڪڍو(dma_q).src;
        uint8_t * s0_spad = (uint8_t *) dma_قطار_ڪڍو(dma_q).dst;

        uint32_t i03 = fastdiv(ir, &bctx->src0_dim12_div);
        uint32_t rem = ir - i03 * (ne02 * ne01);
        uint32_t i02 = fastdiv(rem, &bctx->src0_dim1_div);
        uint32_t i01 = rem - i02 * ne01;

        htp_trace_event_شروعات(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);
        compute_جوڙيو_id_t compute = (compute_جوڙيو_id_t) bctx->compute;
        compute(d_spad, s0_spad, vtcm_src1, vtcm_src2,
                i01, i02, ne01, src1_stride,
                current_بلاڪ_سائيز, bctx->dst_صف_سائيز_aligned, bctx->src0_صف_سائيز_aligned, ne00);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);

        dma_addr_t dst_curr = dst->معلومات + i03 * nb3 + i02 * nb2 + i01 * nb1;
        dma_قطار_ڌڪو(dma_q, dma_make_معلومات(dst_curr, d_spad), nb1, bctx->dst_صف_سائيز_aligned, صف_سائيز_بائيٽون, current_بلاڪ_سائيز);

        إذا (ir_prefetch < پڇاڙي_صف) {
             uint32_t ايندڙ_بلاڪ_سائيز = calc_بلاڪ_سائيز(bctx, ir_prefetch, پڇاڙي_صف, ne01, ne02);
             uint32_t p03, p02, p01, prem;
             p03  = fastdiv(ir_prefetch, &bctx->src0_dim12_div);
             prem = ir_prefetch - p03 * (ne02 * ne01);
             p02  = fastdiv(prem, &bctx->src0_dim1_div);
             p01  = prem - p02 * ne01;
             dma_addr_t s0_ايندڙ = src0->معلومات + p03 * nb03 + p02 * nb02 + p01 * nb01;
             dma_قطار_ڌڪو(dma_q, dma_make_معلومات(s0_spad, s0_ايندڙ), bctx->src0_صف_سائيز_aligned, nb01, صف_سائيز_بائيٽون, ايندڙ_بلاڪ_سائيز);
             ir_prefetch += ايندڙ_بلاڪ_سائيز;
        }
        ir += current_بلاڪ_سائيز;
    }

    dma_قطار_flush(dma_q);
}

ساكن مضمن فراغ hvx_div_scalar_f32(uint8_t * محدود dst, ثابت uint8_t * محدود src, ثابت عائم val, ثابت uint32_t انگ_elems) {
    hvx_mul_scalar_f32(dst, src, 1.0f / val, انگ_elems);
}

ساكن مضمن فراغ hvx_div_scalar_f16(uint8_t * محدود dst, ثابت uint8_t * محدود src, ثابت _Float16 val, ثابت uint32_t انگ_elems) {
    hvx_div_scalar_f16_aa(dst, src, val, انگ_elems);
}

تعريفنوع فراغ (*compute_ٻٽو_chunked_t)(
    uint8_t * محدود dst,
    ثابت uint8_t * محدود src0,
    ثابت uint8_t * محدود src1,
    ثابت uint32_t انگ_elems
);

تعريفنوع فراغ (*compute_ٻٽو_scalar_chunked_f32_t)(
    uint8_t * محدود dst,
    ثابت uint8_t * محدود src0,
    ثابت عائم val,
    ثابت uint32_t انگ_elems
);

تعريفنوع فراغ (*compute_ٻٽو_scalar_chunked_f16_t)(
    uint8_t * محدود dst,
    ثابت uint8_t * محدود src0,
    ثابت _Float16 val,
    ثابت uint32_t انگ_elems
);

بنية ٻٽو_chunked_context {
    بنية htp_ops_context * octx;
    بنية htp_ٻٽو_vtcm_layout vtcm_layout;
    uint8_t *                vtcm_بنياد;
    uint32_t                 elem_شروعات;
    uint32_t                 nelem;
    uint32_t                 chunk_سائيز;
    uint32_t                 chunks_في_thread;
    uint32_t                 ڪل_chunks;
    bool                     is_scalar;
    عائم                    scalar_f32;
    _Float16                 scalar_f16;
    compute_ٻٽو_chunked_t compute;
    compute_ٻٽو_scalar_chunked_f32_t compute_scalar_f32;
    compute_ٻٽو_scalar_chunked_f16_t compute_scalar_f16;
};

ساكن فراغ ٻٽو_thread_chunked(بنانشان صحيح nth, بنانشان صحيح ith, فراغ * معلومات) {
    (فراغ) nth;
    بنية ٻٽو_chunked_context * ctx = (بنية ٻٽو_chunked_context *) معلومات;
    بنية htp_ops_context * octx = ctx->octx;
    ثابت بنية htp_tensor * src0 = octx->src[0];
    ثابت بنية htp_tensor * src1 = octx->src[1];
    ثابت بنية htp_tensor * dst  = octx->dst;

    ثابت uint32_t شروعات_chunk = ctx->chunks_في_thread * ith;
    ثابت uint32_t پڇاڙي_chunk   = MIN(شروعات_chunk + ctx->chunks_في_thread, ctx->ڪل_chunks);
    إذا (شروعات_chunk >= پڇاڙي_chunk) {
        أرجع;
    }

    ثابت uint32_t src0_type   = src0->type;
    ثابت size_t   elem_سائيز   = (src0_type == HTP_TYPE_F32) ? حجم(عائم) : حجم(_Float16);
    ثابت uint32_t chunk_سائيز  = ctx->chunk_سائيز;
    ثابت size_t   chunk_بائيٽون = ctx->vtcm_layout.src0_spad_half_سائيز;

    FARF(مٿاهون_2, "binary-chunked: %d/%d (%u:%u) chunks %u elems %u",
         ith, nth, شروعات_chunk, پڇاڙي_chunk, ctx->ڪل_chunks, ctx->nelem);

    ثابت بنية htp_ٻٽو_vtcm_layout * layout = &ctx->vtcm_layout;
    uint8_t * src0_spad_بنياد = VTCM_LAYOUT_PTR(uint8_t, ctx->vtcm_بنياد, layout->off_src0) + (ith * layout->src0_بائيٽون_في_thread);
    uint8_t * src1_spad_بنياد = ctx->is_scalar ? عدم : (VTCM_LAYOUT_PTR(uint8_t, ctx->vtcm_بنياد, layout->off_src1) + (ith * layout->src1_بائيٽون_في_thread));
    uint8_t * dst_spad_بنياد  = VTCM_LAYOUT_PTR(uint8_t, ctx->vtcm_بنياد, layout->off_dst)  + (ith * layout->dst_بائيٽون_في_thread);

    dma_قطار * dma_q = octx->ctx->dma[ith];
    uint32_t prefetch_chunk = شروعات_chunk;
    صحيح spad_idx = 0;

    لكل (صحيح k = 0; k < 2 && prefetch_chunk < پڇاڙي_chunk; k++) {
        ثابت uint32_t c_شروعات   = ctx->elem_شروعات + prefetch_chunk * chunk_سائيز;
        ثابت uint32_t c_پڇاڙي     = MIN(c_شروعات + chunk_سائيز, ctx->elem_شروعات + ctx->nelem);
        ثابت uint32_t cur_elems = c_پڇاڙي - c_شروعات;
        ثابت uint32_t cur_بائيٽون = cur_elems * elem_سائيز;

        dma_addr_t s0_curr = src0->معلومات + (size_t) c_شروعات * elem_سائيز;
        dma_addr_t d_curr  = dst->معلومات  + (size_t) c_شروعات * elem_سائيز;

        uint8_t * s0_spad = src0_spad_بنياد + spad_idx * chunk_بائيٽون;
        uint8_t * d_spad  = dst_spad_بنياد  + spad_idx * chunk_بائيٽون;

        dma_قطار_ڌڪو(dma_q, dma_make_معلومات(d_curr, d_spad), chunk_بائيٽون, chunk_بائيٽون, cur_بائيٽون, 0);
        dma_قطار_ڌڪو(dma_q, dma_make_معلومات(s0_spad, s0_curr), chunk_بائيٽون, chunk_بائيٽون, cur_بائيٽون, 1);
        إذا (!ctx->is_scalar) {
            dma_addr_t s1_curr = src1->معلومات + (size_t) c_شروعات * elem_سائيز;
            uint8_t * s1_spad = src1_spad_بنياد + spad_idx * chunk_بائيٽون;
            dma_قطار_ڌڪو(dma_q, dma_make_معلومات(s1_spad, s1_curr), chunk_بائيٽون, chunk_بائيٽون, cur_بائيٽون, 1);
        }

        prefetch_chunk++;
        spad_idx ^= 1;
    }

    بنية htp_thread_trace * tr = &octx->ctx->trace[ith];

    لكل (uint32_t c = شروعات_chunk; c < پڇاڙي_chunk; c++) {
        ثابت uint32_t c_شروعات   = ctx->elem_شروعات + c * chunk_سائيز;
        ثابت uint32_t c_پڇاڙي     = MIN(c_شروعات + chunk_سائيز, ctx->elem_شروعات + ctx->nelem);
        ثابت uint32_t cur_elems = c_پڇاڙي - c_شروعات;
        ثابت uint32_t cur_بائيٽون = cur_elems * elem_سائيز;

        uint8_t * d_spad  = (uint8_t *) dma_قطار_ڪڍو(dma_q).src;
        uint8_t * s0_spad = (uint8_t *) dma_قطار_ڪڍو(dma_q).dst;
        uint8_t * s1_spad = ctx->is_scalar ? عدم : (uint8_t *) dma_قطار_ڪڍو(dma_q).dst;

        htp_trace_event_شروعات(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) c);
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

        dma_addr_t dst_curr = dst->معلومات + (size_t) c_شروعات * elem_سائيز;
        dma_قطار_ڌڪو(dma_q, dma_make_معلومات(dst_curr, d_spad), chunk_بائيٽون, chunk_بائيٽون, cur_بائيٽون, 1);

        إذا (prefetch_chunk < پڇاڙي_chunk) {
            ثابت uint32_t pc_شروعات  = ctx->elem_شروعات + prefetch_chunk * chunk_سائيز;
            ثابت uint32_t pc_پڇاڙي    = MIN(pc_شروعات + chunk_سائيز, ctx->elem_شروعات + ctx->nelem);
            ثابت uint32_t p_elems   = pc_پڇاڙي - pc_شروعات;
            ثابت uint32_t p_بائيٽون   = p_elems * elem_سائيز;

            dma_addr_t s0_ايندڙ = src0->معلومات + (size_t) pc_شروعات * elem_سائيز;
            dma_قطار_ڌڪو(dma_q, dma_make_معلومات(s0_spad, s0_ايندڙ), chunk_بائيٽون, chunk_بائيٽون, p_بائيٽون, 1);
            إذا (!ctx->is_scalar) {
                dma_addr_t s1_ايندڙ = src1->معلومات + (size_t) pc_شروعات * elem_سائيز;
                dma_قطار_ڌڪو(dma_q, dma_make_معلومات(s1_spad, s1_ايندڙ), chunk_بائيٽون, chunk_بائيٽون, p_بائيٽون, 1);
            }

            prefetch_chunk++;
        }
    }

    dma_قطار_flush(dma_q);
}

ساكن صحيح execute_op_ٻٽو(بنية htp_ops_context * octx) {
    ثابت بنية htp_tensor * src0 = octx->src[0];
    ثابت بنية htp_tensor * src1 = octx->src[1];
    ثابت بنية htp_tensor * dst  = octx->dst;
    ثابت بنية htp_ٻٽو_kernel_params * kparams = (ثابت بنية htp_ٻٽو_kernel_params *) octx->kernel_params;

    ثابت uint32_t src0_nrows = src0->ne[1] * src0->ne[2] * src0->ne[3];

    // Use packed row sizes for VTCM allocation and alignment
    ثابت uint32_t src0_type = octx->src[0]->type;
    ثابت size_t elem_سائيز = (src0_type == HTP_TYPE_F32) ? حجم(عائم) : حجم(_Float16);
    ثابت size_t src0_صف_سائيز = src0->ne[0] * elem_سائيز;
    ثابت size_t src1_صف_سائيز = src1->ne[0] * elem_سائيز;
    ثابت size_t dst_صف_سائيز  = dst->ne[0]  * elem_سائيز;

    إذا (kparams->kernel_type == HTP_ٻٽو_KERNEL_CHUNKED) {
        ثابت bool is_scalar = kparams->is_scalar || (src1->ne[0] == 1 && src1->ne[1] == 1 && src1->ne[2] == 1 && src1->ne[3] == 1);

        compute_ٻٽو_chunked_t compute = عدم;
        compute_ٻٽو_scalar_chunked_f32_t compute_scalar_f32 = عدم;
        compute_ٻٽو_scalar_chunked_f16_t compute_scalar_f16 = عدم;

        إذا (is_scalar) {
            إذا (src0_type == HTP_TYPE_F32) {
                بدل (octx->op) {
                    حالة HTP_OP_جوڙيو: compute_scalar_f32 = hvx_جوڙيو_scalar_f32; اكسر;
                    حالة HTP_OP_SUB: compute_scalar_f32 = hvx_sub_scalar_f32; اكسر;
                    حالة HTP_OP_MUL: compute_scalar_f32 = hvx_mul_scalar_f32; اكسر;
                    حالة HTP_OP_DIV: compute_scalar_f32 = hvx_div_scalar_f32; اكسر;
                    افتراضي: اكسر;
                }
            } وإلا إذا (src0_type == HTP_TYPE_F16) {
                بدل (octx->op) {
                    حالة HTP_OP_جوڙيو: compute_scalar_f16 = hvx_جوڙيو_scalar_f16; اكسر;
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
                    حالة HTP_OP_جوڙيو: compute = hvx_جوڙيو_f32; اكسر;
                    حالة HTP_OP_SUB: compute = hvx_sub_f32; اكسر;
                    حالة HTP_OP_MUL: compute = hvx_mul_f32; اكسر;
                    حالة HTP_OP_DIV: compute = hvx_div_f32; اكسر;
                    افتراضي: اكسر;
                }
            } وإلا إذا (src0_type == HTP_TYPE_F16) {
                بدل (octx->op) {
                    حالة HTP_OP_جوڙيو: compute = hvx_جوڙيو_f16; اكسر;
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

        ثابت uint32_t ڪل_elems = (uint32_t) (src0->ne[0] * src0->ne[1] * src0->ne[2] * src0->ne[3]);
        إذا (ڪل_elems == 0) {
            أرجع HTP_حالت_OK;
        }

        uint32_t elem_شروعات = 0;
        uint32_t nelem      = ڪل_elems;
        ثابت uint32_t elems_في_سٽ = (src0_type == HTP_TYPE_F32) ? 32 : 64;

        إذا (octx->ctx->mdev.ڳڻپ > 1) {
            ثابت bool can_ورهايو = htp_tensor_mdev_معلومات_aligned(dst) &&
                                   htp_tensor_mdev_معلومات_aligned(src0) &&
                                   (is_scalar || htp_tensor_mdev_معلومات_aligned(src1)) &&
                                   htp_tensor_is_contiguous(dst, elem_سائيز) &&
                                   htp_tensor_is_contiguous(src0, elem_سائيز) &&
                                   (is_scalar || htp_tensor_is_contiguous(src1, elem_سائيز));
            ثابت بنية htp_tensor_mdev_range range = htp_tensor_mdev_حصا_ڪريو(
                ڪل_elems, can_ورهايو ? elems_في_سٽ : 0,
                octx->ctx->mdev.idx, octx->ctx->mdev.ڳڻپ, &octx->ctx->mdev.ڳڻپ_div);
            elem_شروعات = range.شروعات;
            nelem      = range.ڳڻپ;
        }

        إذا (nelem == 0) {
            أرجع HTP_حالت_OK;
        }

        إذا (!htp_ops_context_سيٽ_n_ڌاڳا(octx, kparams->n_ڌاڳا)) {
            أرجع HTP_حالت_INVAL_PARAMS;
        }

        بنية htp_ٻٽو_vtcm_layout vtcm_layout;
        htp_ٻٽو_vtcm_layout_build(&vtcm_layout, kparams, octx->ctx->vtcm_سائيز);
        إذا (vtcm_layout.ڪل_بائيٽون == 0 || vtcm_layout.ڪل_بائيٽون > octx->ctx->vtcm_سائيز) {
            أرجع HTP_حالت_VTCM_TOO_SMALL;
        }

        ثابت uint32_t chunk_سائيز = kparams->chunk_سائيز > 0 ? kparams->chunk_سائيز : (32768 / elem_سائيز);
        ثابت uint32_t ڪل_chunks = (nelem + chunk_سائيز - 1) / chunk_سائيز;
        ثابت uint32_t n_ڌاڳا = (ڪل_chunks >= 2) ? MIN(octx->n_ڌاڳا, ڪل_chunks) : 1;
        ثابت uint32_t chunks_في_thread = (ڪل_chunks + n_ڌاڳا - 1) / n_ڌاڳا;

        عائم scalar_f32 = 0.0f;
        _Float16 scalar_f16 = 0;
        إذا (is_scalar) {
            uint8_t * vtcm_src1 = VTCM_LAYOUT_PTR(uint8_t, octx->ctx->vtcm_بنياد, vtcm_layout.off_src1);
            dma_قطار * dma_q = octx->ctx->dma[0];
            dma_قطار_ڌڪو(dma_q, dma_make_معلومات(vtcm_src1, src1->معلومات), 128, 0, elem_سائيز, 1);
            dma_قطار_ڪڍو(dma_q);

            إذا (src0_type == HTP_TYPE_F32) {
                scalar_f32 = ((ثابت عائم *) vtcm_src1)[0];
            } وإلا {
                scalar_f16 = ((ثابت _Float16 *) vtcm_src1)[0];
            }
        }

        بنية ٻٽو_chunked_context cctx = {
            .octx                = octx,
            .vtcm_layout         = vtcm_layout,
            .vtcm_بنياد           = (uint8_t *) octx->ctx->vtcm_بنياد,
            .elem_شروعات          = elem_شروعات,
            .nelem               = nelem,
            .chunk_سائيز          = chunk_سائيز,
            .chunks_في_thread   = chunks_في_thread,
            .ڪل_chunks        = ڪل_chunks,
            .is_scalar           = is_scalar,
            .scalar_f32          = scalar_f32,
            .scalar_f16          = scalar_f16,
            .compute             = compute,
            .compute_scalar_f32  = compute_scalar_f32,
            .compute_scalar_f16  = compute_scalar_f16,
        };

        work_قطار_لاڳيتو(octx->ctx->work_قطار, ٻٽو_thread_chunked, &cctx, n_ڌاڳا);
        أرجع HTP_حالت_OK;
    }

    uint32_t صف_شروعات = 0;
    uint32_t nrows     = src0_nrows;

    إذا (octx->ctx->mdev.ڳڻپ > 1) {
        uint32_t صفون_في_chunk = 0;
        htp_tensor_mdev_صفون_في_chunk(dst, (uint32_t) elem_سائيز, (uint32_t) dst_صف_سائيز, &صفون_في_chunk);
        ثابت بنية htp_tensor_mdev_range range = htp_tensor_mdev_حصا_ڪريو(src0_nrows, صفون_في_chunk,
                                                       octx->ctx->mdev.idx, octx->ctx->mdev.ڳڻپ, &octx->ctx->mdev.ڳڻپ_div);
        صف_شروعات = range.شروعات;
        nrows     = range.ڳڻپ;
    }

    إذا (nrows == 0) {
        أرجع HTP_حالت_OK;
    }

    إذا (!htp_ops_context_سيٽ_n_ڌاڳا(octx, kparams->n_ڌاڳا)) {
        أرجع HTP_حالت_INVAL_PARAMS;
    }

    ثابت uint32_t n_ڌاڳا = octx->n_ڌاڳا;
    ثابت size_t src0_صف_سائيز_aligned = kparams->src0_صف_سائيز_aligned;
    ثابت size_t src1_صف_سائيز_aligned = kparams->src1_صف_سائيز_aligned;
    ثابت size_t dst_صف_سائيز_aligned  = kparams->dst_صف_سائيز_aligned;

    بنية htp_ٻٽو_context bctx;
    bctx.vtcm_بنياد = (uint8_t *) octx->ctx->vtcm_بنياد;
    htp_ٻٽو_vtcm_layout_build(&bctx.vtcm_layout, kparams, octx->ctx->vtcm_سائيز);

    إذا (bctx.vtcm_layout.صفون_في_بفر == 0 || bctx.vtcm_layout.ڪل_بائيٽون > octx->ctx->vtcm_سائيز) {
        أرجع HTP_حالت_VTCM_TOO_SMALL;
    }

    dma_قطار * dma_q = octx->ctx->dma[0];
    uint8_t * vtcm_src1 = VTCM_LAYOUT_PTR(uint8_t, bctx.vtcm_بنياد, bctx.vtcm_layout.off_src1);
    إذا (kparams->kernel_type == HTP_ٻٽو_KERNEL_صف_BCAST) {
        dma_قطار_ڌڪو(dma_q, dma_make_معلومات(vtcm_src1, src1->معلومات), bctx.vtcm_layout.src1_سائيز, 0, src1->ne[0] * elem_سائيز, 1);
    } وإلا إذا (kparams->kernel_type == HTP_ٻٽو_KERNEL_SCALAR ||
               kparams->kernel_type == HTP_ٻٽو_KERNEL_REPEAT ||
               kparams->kernel_type == HTP_ٻٽو_KERNEL_COMPLEX) {
        إذا (!kparams->src1_strided) {
            ثابت size_t src1_بائيٽون = (size_t) src1->ne[0] * src1->ne[1] * src1->ne[2] * src1->ne[3] * elem_سائيز;
            dma_قطار_ڌڪو(dma_q, dma_make_معلومات(vtcm_src1, src1->معلومات), bctx.vtcm_layout.src1_سائيز, 0, src1_بائيٽون, 1);
        } وإلا {
            ثابت uint32_t ne11 = src1->ne[1];
            ثابت uint32_t ne12 = src1->ne[2];
            ثابت uint32_t ne13 = src1->ne[3];
            ثابت size_t صف_بائيٽون = src1->ne[0] * elem_سائيز;
            لكل (uint32_t i13 = 0; i13 < ne13; i13++) {
                لكل (uint32_t i12 = 0; i12 < ne12; i12++) {
                    dma_addr_t src_addr = src1->معلومات + i13 * src1->nb[3] + i12 * src1->nb[2];
                    uint8_t * dst_ptr = vtcm_src1 + (i13 * ne12 * ne11 + i12 * ne11) * صف_بائيٽون;
                    طالما (!dma_قطار_ڌڪو(dma_q, dma_make_معلومات(dst_ptr, src_addr),
                                           صف_بائيٽون, src1->nb[1], صف_بائيٽون, ne11)) {
                        dma_قطار_ڪڍو(dma_q);
                    }
                }
            }
        }
    } وإلا إذا (kparams->kernel_type == HTP_ٻٽو_KERNEL_جوڙيو_ID) {
        ثابت بنية htp_tensor * src2 = octx->src[2];
        uint8_t * vtcm_src2 = VTCM_LAYOUT_PTR(uint8_t, bctx.vtcm_بنياد, bctx.vtcm_layout.off_src2);
        dma_قطار_ڌڪو(dma_q, dma_make_معلومات(vtcm_src1, src1->معلومات),
                       kparams->src1_صف_سائيز_aligned, src1->nb[1],
                       src1->ne[0] * elem_سائيز, src1->ne[1]);
        ثابت size_t src2_صف_بائيٽون = src2->ne[0] * حجم(int32_t);
        dma_قطار_ڌڪو(dma_q, dma_make_معلومات(vtcm_src2, src2->معلومات),
                       src2_صف_بائيٽون, src2->nb[1],
                       src2_صف_بائيٽون, src2->ne[1]);
    }

    bctx.octx                  = octx;
    bctx.nrows_في_thread      = fastdiv(nrows + n_ڌاڳا - 1, &octx->n_ڌاڳا_div);
    bctx.ڪل_صفون            = nrows;
    bctx.صف_شروعات             = صف_شروعات;
    bctx.بلاڪ_max             = bctx.vtcm_layout.صفون_في_بفر;
    bctx.src0_صف_سائيز_aligned = src0_صف_سائيز_aligned;
    bctx.src1_صف_سائيز_aligned = src1_صف_سائيز_aligned;
    bctx.dst_صف_سائيز_aligned  = dst_صف_سائيز_aligned;
    bctx.صف_سائيز_بائيٽون        = src0_صف_سائيز;

    bctx.src0_dim1_div  = init_fastdiv_قدرون(src0->ne[1]);
    bctx.src0_dim2_div  = init_fastdiv_قدرون(src0->ne[2]);
    bctx.src0_dim12_div = init_fastdiv_قدرون(src0->ne[1] * src0->ne[2]);

    bctx.src1_dim1_div  = init_fastdiv_قدرون(src1->ne[1]);
    bctx.src1_dim2_div  = init_fastdiv_قدرون(src1->ne[2]);
    bctx.src1_dim3_div  = init_fastdiv_قدرون(src1->ne[3]);

    bctx.ورهايو_at_ne01  = kparams->ورهايو_at_ne01;
    bctx.ورهايو_at_ne02  = kparams->ورهايو_at_ne02;

    ڪم_ڪندڙ_callback_t ڪم_ڪندڙ_func = عدم;
    فراغ * compute_func = عدم;

    بدل (kparams->kernel_type) {
        حالة HTP_ٻٽو_KERNEL_SAME_SHAPE:
            ڪم_ڪندڙ_func = ٻٽو_thread_ويڪٽر_same_shape;
            إذا (src0_type == HTP_TYPE_F32) {
                بدل (octx->op) {
                    حالة HTP_OP_جوڙيو: compute_func = compute_same_shape_جوڙيو_f32; اكسر;
                    حالة HTP_OP_SUB: compute_func = compute_same_shape_sub_f32; اكسر;
                    حالة HTP_OP_MUL: compute_func = compute_same_shape_mul_f32; اكسر;
                    حالة HTP_OP_DIV: compute_func = compute_same_shape_div_f32; اكسر;
                    افتراضي: اكسر;
                }
            } وإلا إذا (src0_type == HTP_TYPE_F16) {
                بدل (octx->op) {
                    حالة HTP_OP_جوڙيو: compute_func = compute_same_shape_جوڙيو_f16; اكسر;
                    حالة HTP_OP_SUB: compute_func = compute_same_shape_sub_f16; اكسر;
                    حالة HTP_OP_MUL: compute_func = compute_same_shape_mul_f16; اكسر;
                    حالة HTP_OP_DIV: compute_func = compute_same_shape_div_f16; اكسر;
                    افتراضي: اكسر;
                }
            }
            اكسر;
        حالة HTP_ٻٽو_KERNEL_صف_BCAST:
            ڪم_ڪندڙ_func = ٻٽو_thread_ويڪٽر_صف_نشر;
            إذا (src0_type == HTP_TYPE_F32) {
                بدل (octx->op) {
                    حالة HTP_OP_جوڙيو: compute_func = compute_صف_bcast_جوڙيو_f32; اكسر;
                    حالة HTP_OP_SUB: compute_func = compute_صف_bcast_sub_f32; اكسر;
                    حالة HTP_OP_MUL: compute_func = compute_صف_bcast_mul_f32; اكسر;
                    حالة HTP_OP_DIV: compute_func = compute_صف_bcast_div_f32; اكسر;
                    افتراضي: اكسر;
                }
            } وإلا إذا (src0_type == HTP_TYPE_F16) {
                بدل (octx->op) {
                    حالة HTP_OP_جوڙيو: compute_func = compute_صف_bcast_جوڙيو_f16; اكسر;
                    حالة HTP_OP_SUB: compute_func = compute_صف_bcast_sub_f16; اكسر;
                    حالة HTP_OP_MUL: compute_func = compute_صف_bcast_mul_f16; اكسر;
                    حالة HTP_OP_DIV: compute_func = compute_صف_bcast_div_f16; اكسر;
                    افتراضي: اكسر;
                }
            }
            اكسر;
        حالة HTP_ٻٽو_KERNEL_SCALAR:
            ڪم_ڪندڙ_func = ٻٽو_thread_scalar;
            إذا (src0_type == HTP_TYPE_F32) {
                بدل (octx->op) {
                    حالة HTP_OP_جوڙيو: compute_func = compute_scalar_جوڙيو_f32; اكسر;
                    حالة HTP_OP_SUB: compute_func = compute_scalar_sub_f32; اكسر;
                    حالة HTP_OP_MUL: compute_func = compute_scalar_mul_f32; اكسر;
                    حالة HTP_OP_DIV: compute_func = compute_scalar_div_f32; اكسر;
                    افتراضي: اكسر;
                }
            } وإلا إذا (src0_type == HTP_TYPE_F16) {
                بدل (octx->op) {
                    حالة HTP_OP_جوڙيو: compute_func = compute_scalar_جوڙيو_f16; اكسر;
                    حالة HTP_OP_SUB: compute_func = compute_scalar_sub_f16; اكسر;
                    حالة HTP_OP_MUL: compute_func = compute_scalar_mul_f16; اكسر;
                    حالة HTP_OP_DIV: compute_func = compute_scalar_div_f16; اكسر;
                    افتراضي: اكسر;
                }
            }
            اكسر;
        حالة HTP_ٻٽو_KERNEL_COMPLEX:
            ڪم_ڪندڙ_func = ٻٽو_thread_ويڪٽر_complex;
            إذا (src0_type == HTP_TYPE_F32) {
                بدل (octx->op) {
                    حالة HTP_OP_جوڙيو: compute_func = compute_complex_جوڙيو_f32; اكسر;
                    حالة HTP_OP_SUB: compute_func = compute_complex_sub_f32; اكسر;
                    حالة HTP_OP_MUL: compute_func = compute_complex_mul_f32; اكسر;
                    حالة HTP_OP_DIV: compute_func = compute_complex_div_f32; اكسر;
                    افتراضي: اكسر;
                }
            } وإلا إذا (src0_type == HTP_TYPE_F16) {
                بدل (octx->op) {
                    حالة HTP_OP_جوڙيو: compute_func = compute_complex_جوڙيو_f16; اكسر;
                    حالة HTP_OP_SUB: compute_func = compute_complex_sub_f16; اكسر;
                    حالة HTP_OP_MUL: compute_func = compute_complex_mul_f16; اكسر;
                    حالة HTP_OP_DIV: compute_func = compute_complex_div_f16; اكسر;
                    افتراضي: اكسر;
                }
            }
            اكسر;
        حالة HTP_ٻٽو_KERNEL_REPEAT:
            ڪم_ڪندڙ_func = ٻٽو_thread_element_repeat;
            إذا (src0_type == HTP_TYPE_F32) {
                بدل (octx->op) {
                    حالة HTP_OP_جوڙيو: compute_func = compute_repeat_جوڙيو_f32; اكسر;
                    حالة HTP_OP_SUB: compute_func = compute_repeat_sub_f32; اكسر;
                    حالة HTP_OP_MUL: compute_func = compute_repeat_mul_f32; اكسر;
                    حالة HTP_OP_DIV: compute_func = compute_repeat_div_f32; اكسر;
                    افتراضي: اكسر;
                }
            } وإلا إذا (src0_type == HTP_TYPE_F16) {
                بدل (octx->op) {
                    حالة HTP_OP_جوڙيو: compute_func = compute_repeat_جوڙيو_f16; اكسر;
                    حالة HTP_OP_SUB: compute_func = compute_repeat_sub_f16; اكسر;
                    حالة HTP_OP_MUL: compute_func = compute_repeat_mul_f16; اكسر;
                    حالة HTP_OP_DIV: compute_func = compute_repeat_div_f16; اكسر;
                    افتراضي: اكسر;
                }
            }
            اكسر;
        حالة HTP_ٻٽو_KERNEL_جوڙيو_ID:
            إذا (octx->op == HTP_OP_جوڙيو_ID && src0_type == HTP_TYPE_F32) {
                ڪم_ڪندڙ_func = ٻٽو_thread_جوڙيو_id_f32;
                compute_func = (فراغ *) compute_جوڙيو_id_f32;
            }
            اكسر;
        افتراضي: اكسر;
    }

    إذا (!ڪم_ڪندڙ_func || !compute_func) {
        أرجع HTP_حالت_NO_SUPPORT;
    }

    bctx.compute = compute_func;

    إذا (kparams->kernel_type == HTP_ٻٽو_KERNEL_صف_BCAST ||
        kparams->kernel_type == HTP_ٻٽو_KERNEL_SCALAR ||
        kparams->kernel_type == HTP_ٻٽو_KERNEL_REPEAT ||
        kparams->kernel_type == HTP_ٻٽو_KERNEL_COMPLEX ||
        kparams->kernel_type == HTP_ٻٽو_KERNEL_جوڙيو_ID) {
        dma_قطار_flush(dma_q);
    }

    work_قطار_لاڳيتو(octx->ctx->work_قطار, ڪم_ڪندڙ_func, &bctx, n_ڌاڳا);

    أرجع HTP_حالت_OK;
}

صحيح op_ٻٽو(بنية htp_ops_context * octx) {
    ثابت uint32_t src0_type = octx->src[0]->type;
    إذا ((src0_type == HTP_TYPE_F32) || (src0_type == HTP_TYPE_F16)) {
        أرجع execute_op_ٻٽو(octx);
    }

    أرجع HTP_حالت_NO_SUPPORT;
}
