#ہدایت clang diagnostic ignored "-Wunused-variable"
#ہدایت clang diagnostic ignored "-Wunused-function"
#ہدایت clang diagnostic ignored "-Wunused-but-set-variable"

#شامل <HAP_farf.h>
#شامل <HAP_perf.h>

#شامل <math.h>
#شامل <string.h>

#شامل "dma-queue.h"
#شامل "hvx-utils.h"

#تعریف_کرو GGML_COMMON_DECL_C
#شامل "ggml-common.h"
#شامل "hex-common.h"
#شامل "hex-profile.h"
#شامل "binary-ops.h"
#شامل "htp-ctx.h"
#شامل "htp-ops.h"
#شامل "htp-tensor.h"

#اگر_غیر_تعریف_شدہ MIN
#تعریف_کرو MIN(a, b) ((a) < (b) ? (a) : (b))
#اگر_ختم

// Context for binary operations
ساخت htp_ثنائی_context {
    ساخت htp_ops_context * octx;
    ساخت htp_ثنائی_vtcm_layout vtcm_layout;
    uint8_t * vtcm_بنیاد;

    ساخت fastdiv_اقدار src0_dim1_div; // ne01
    ساخت fastdiv_اقدار src0_dim2_div; // ne02
    ساخت fastdiv_اقدار src0_dim12_div;// ne03

    ساخت fastdiv_اقدار src1_dim1_div; // ne11
    ساخت fastdiv_اقدار src1_dim2_div; // ne12
    ساخت fastdiv_اقدار src1_dim3_div; // ne13

    uint32_t بلاک_max;
    uint32_t nrows_فی_thread;
    uint32_t کل_صفیں;
    uint32_t صف_آغاز;
    size_t   src0_صف_سائز_aligned;
    size_t   src1_صف_سائز_aligned;
    size_t   dst_صف_سائز_aligned;
    size_t   صف_سائز_بائٹیں;

    bool بانٹو_at_ne01;
    bool بانٹو_at_ne02;

    خالی * compute;
};

#تعریف_کرو htp_ثنائی_preamble                        \
    ثابت ساخت htp_tensor * src0 = octx->src[0]; \
    ثابت ساخت htp_tensor * src1 = octx->src[1]; \
    ثابت ساخت htp_tensor * dst  = octx->dst;    \
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

ساکن اندرونی uint32_t calc_بلاک_سائز(ساخت htp_ثنائی_context * bctx, uint32_t ir, uint32_t انتہا_صف, uint32_t ne01, uint32_t ne02) {
    uint32_t i03, i02, i01, rem;
    i03 = fastdiv(ir, &bctx->src0_dim12_div);
    rem = ir - i03 * (ne02 * ne01);
    i02 = fastdiv(rem, &bctx->src0_dim1_div);
    i01 = rem - i02 * ne01;

    uint32_t صفیں_بایاں = انتہا_صف - ir;
    uint32_t بلاک_حد_بندی = صفیں_بایاں;

    اگر (bctx->بانٹو_at_ne01) {
        بلاک_حد_بندی = MIN(بلاک_حد_بندی, ne01 - i01);
    }
    اگر (bctx->بانٹو_at_ne02) {
         uint32_t صفیں_اندر_plane = (ne02 * ne01) - rem;
         بلاک_حد_بندی = MIN(بلاک_حد_بندی, صفیں_اندر_plane);
    }

    واپس MIN(bctx->بلاک_max, بلاک_حد_بندی);
}

// Out-of-line compute micro-kernels

قسمتعریف خالی (*compute_scalar_t)(
    uint8_t * dst, ثابت uint8_t * src0, ثابت خالی * s1_جدول,
    uint32_t cur_i11, uint32_t ne11, uint32_t n_صفیں,
    size_t dst_stride, size_t src0_stride, uint32_t ne00);

#تعریف_کرو DEFINE_COMPUTE_SCALAR(NAME, TYPE, HVX_STMT)             \
ساکن خالی compute_scalar_##NAME(                              \
    uint8_t * dst, ثابت uint8_t * src0, ثابت خالی * s1_جدول, \
    uint32_t cur_i11, uint32_t ne11, uint32_t n_صفیں,           \
    size_t dst_stride, size_t src0_stride, uint32_t ne00) {     \
    ثابت TYPE * جدول = (ثابت TYPE *) s1_جدول;               \
    برائے (uint32_t r = 0; r < n_صفیں; r++) {                     \
        uint8_t * r_dst = dst + r * dst_stride;                 \
        ثابت uint8_t * r_src0 = src0 + r * src0_stride;        \
        TYPE val = جدول[cur_i11];                              \
        HVX_STMT;                                               \
        اگر (ne11 > 1 && ++cur_i11 == ne11) {                    \
            cur_i11 = 0;                                        \
        }                                                       \
    }                                                           \
}

DEFINE_COMPUTE_SCALAR(جوڑو_f32, اعشاری,    hvx_جوڑو_scalar_f32_aa(r_dst, r_src0, val, ne00))
DEFINE_COMPUTE_SCALAR(جوڑو_f16, _Float16, hvx_جوڑو_scalar_f16_aa(r_dst, r_src0, val, ne00))
DEFINE_COMPUTE_SCALAR(sub_f32, اعشاری,    hvx_sub_scalar_f32_aa(r_dst, r_src0, val, ne00))
DEFINE_COMPUTE_SCALAR(sub_f16, _Float16, hvx_sub_scalar_f16_aa(r_dst, r_src0, val, ne00))
DEFINE_COMPUTE_SCALAR(mul_f32, اعشاری,    hvx_mul_scalar_f32_aa(r_dst, r_src0, val, ne00))
DEFINE_COMPUTE_SCALAR(mul_f16, _Float16, hvx_mul_scalar_f16_aa(r_dst, r_src0, val, ne00))
DEFINE_COMPUTE_SCALAR(div_f32, اعشاری,    hvx_mul_scalar_f32_aa(r_dst, r_src0, 1.0f / (val), ne00))
DEFINE_COMPUTE_SCALAR(div_f16, _Float16, hvx_div_scalar_f16_aa(r_dst, r_src0, val, ne00))

قسمتعریف خالی (*compute_same_shape_t)(
    uint8_t * dst, ثابت uint8_t * src0, ثابت uint8_t * src1,
    uint32_t n_صفیں, size_t dst_stride, size_t src0_stride, size_t src1_stride, uint32_t ne00);

#تعریف_کرو DEFINE_COMPUTE_SAME_SHAPE(NAME, HVX_FN)                                                  \
ساکن خالی compute_same_shape_##NAME(                                                           \
    uint8_t * dst, ثابت uint8_t * src0, ثابت uint8_t * src1,                                   \
    uint32_t n_صفیں, size_t dst_stride, size_t src0_stride, size_t src1_stride, uint32_t ne00) { \
    برائے (uint32_t r = 0; r < n_صفیں; r++) {                                                      \
        HVX_FN(dst + r * dst_stride, src0 + r * src0_stride, src1 + r * src1_stride, ne00);      \
    }                                                                                            \
}

DEFINE_COMPUTE_SAME_SHAPE(جوڑو_f32, hvx_جوڑو_f32_aaa)
DEFINE_COMPUTE_SAME_SHAPE(جوڑو_f16, hvx_جوڑو_f16_aaa)
DEFINE_COMPUTE_SAME_SHAPE(sub_f32, hvx_sub_f32_aaa)
DEFINE_COMPUTE_SAME_SHAPE(sub_f16, hvx_sub_f16_aaa)
DEFINE_COMPUTE_SAME_SHAPE(mul_f32, hvx_mul_f32_aaa)
DEFINE_COMPUTE_SAME_SHAPE(mul_f16, hvx_mul_f16_aaa)
DEFINE_COMPUTE_SAME_SHAPE(div_f32, hvx_div_f32_aaa)
DEFINE_COMPUTE_SAME_SHAPE(div_f16, hvx_div_f16_aaa)

قسمتعریف خالی (*compute_صف_bcast_t)(
    uint8_t * dst, ثابت uint8_t * src0, ثابت uint8_t * src1,
    uint32_t n_صفیں, size_t dst_stride, size_t src0_stride, uint32_t ne00);

#تعریف_کرو DEFINE_COMPUTE_صف_BCAST(NAME, HVX_FN)                                               \
ساکن خالی compute_صف_bcast##NAME(                                                        \
    uint8_t * dst, ثابت uint8_t * src0, ثابت uint8_t * src1,                               \
    uint32_t n_صفیں, size_t dst_stride, size_t src0_stride, uint32_t ne00) {                 \
    برائے (uint32_t r = 0; r < n_صفیں; r++) {                                                  \
        HVX_FN(dst + r * dst_stride, src0 + r * src0_stride, src1, ne00);                    \
    }                                                                                        \
}

DEFINE_COMPUTE_صف_BCAST(جوڑو_f32, hvx_جوڑو_f32_aaa)
DEFINE_COMPUTE_صف_BCAST(جوڑو_f16, hvx_جوڑو_f16_aaa)
DEFINE_COMPUTE_صف_BCAST(sub_f32, hvx_sub_f32_aaa)
DEFINE_COMPUTE_صف_BCAST(sub_f16, hvx_sub_f16_aaa)
DEFINE_COMPUTE_صف_BCAST(mul_f32, hvx_mul_f32_aaa)
DEFINE_COMPUTE_صف_BCAST(mul_f16, hvx_mul_f16_aaa)
DEFINE_COMPUTE_صف_BCAST(div_f32, hvx_div_f32_aaa)
DEFINE_COMPUTE_صف_BCAST(div_f16, hvx_div_f16_aaa)

قسمتعریف خالی (*compute_complex_t)(
    uint8_t * dst, ثابت uint8_t * src0, ثابت uint8_t * src1_plane,
    uint32_t i01, uint32_t ne11, ثابت ساخت fastdiv_اقدار * div11, uint32_t src1_stride,
    uint32_t n_صفیں, size_t dst_stride, size_t src0_stride, uint32_t ne00);

#تعریف_کرو DEFINE_COMPUTE_COMPLEX(NAME, HVX_FN)                                                 \
ساکن خالی compute_complex_##NAME(                                                          \
    uint8_t * dst, ثابت uint8_t * src0, ثابت uint8_t * src1_plane,                         \
    uint32_t i01, uint32_t ne11, ثابت ساخت fastdiv_اقدار * div11, uint32_t src1_stride,  \
    uint32_t n_صفیں, size_t dst_stride, size_t src0_stride, uint32_t ne00) {                 \
    برائے (uint32_t r = 0; r < n_صفیں; r++) {                                                  \
        uint32_t i11 = fastmodulo(i01 + r, ne11, div11);                                     \
        ثابت uint8_t * r_src1 = src1_plane + i11 * src1_stride;                             \
        HVX_FN(dst + r * dst_stride, src0 + r * src0_stride, r_src1, ne00);                  \
    }                                                                                        \
}

DEFINE_COMPUTE_COMPLEX(جوڑو_f32, hvx_جوڑو_f32_aau)
DEFINE_COMPUTE_COMPLEX(جوڑو_f16, hvx_جوڑو_f16_aau)
DEFINE_COMPUTE_COMPLEX(sub_f32, hvx_sub_f32_aau)
DEFINE_COMPUTE_COMPLEX(sub_f16, hvx_sub_f16_aau)
DEFINE_COMPUTE_COMPLEX(mul_f32, hvx_mul_f32_aau)
DEFINE_COMPUTE_COMPLEX(mul_f16, hvx_mul_f16_aau)
DEFINE_COMPUTE_COMPLEX(div_f32, hvx_div_f32_aau)
DEFINE_COMPUTE_COMPLEX(div_f16, hvx_div_f16_aau)

قسمتعریف خالی (*compute_repeat_t)(
    uint8_t * dst, ثابت uint8_t * src0, ثابت uint8_t * src1_plane,
    uint32_t i01, uint32_t ne11, ثابت ساخت fastdiv_اقدار * div11, uint32_t src1_stride,
    uint32_t n_صفیں, size_t dst_stride, size_t src0_stride, uint32_t ne00, uint32_t ne10);

#تعریف_کرو DEFINE_COMPUTE_REPEAT(NAME, TYPE, HVX_FN)                                            \
ساکن خالی compute_repeat_##NAME(                                                           \
    uint8_t * dst, ثابت uint8_t * src0, ثابت uint8_t * src1_plane,                         \
    uint32_t i01, uint32_t ne11, ثابت ساخت fastdiv_اقدار * div11, uint32_t src1_stride,  \
    uint32_t n_صفیں, size_t dst_stride, size_t src0_stride, uint32_t ne00, uint32_t ne10) {  \
    برائے (uint32_t r = 0; r < n_صفیں; r++) {                                                  \
        uint32_t i11 = fastmodulo(i01 + r, ne11, div11);                                     \
        ثابت uint8_t * r_src1_صف = src1_plane + i11 * src1_stride;                         \
        uint8_t * r_dst = dst + r * dst_stride;                                              \
        ثابت uint8_t * r_src0 = src0 + r * src0_stride;                                     \
        برائے (uint32_t c = 0; c < ne00; c += ne10) {                                          \
            uint32_t len = MIN(ne10, ne00 - c);                                              \
            HVX_FN(r_dst + c * حجم(TYPE), r_src0 + c * حجم(TYPE), r_src1_صف, len);    \
        }                                                                                    \
    }                                                                                        \
}

DEFINE_COMPUTE_REPEAT(جوڑو_f32, اعشاری,    hvx_جوڑو_f32_uuu)
DEFINE_COMPUTE_REPEAT(جوڑو_f16, _Float16, hvx_جوڑو_f16_uuu)
DEFINE_COMPUTE_REPEAT(sub_f32, اعشاری,    hvx_sub_f32_uuu)
DEFINE_COMPUTE_REPEAT(sub_f16, _Float16, hvx_sub_f16_uuu)
DEFINE_COMPUTE_REPEAT(mul_f32, اعشاری,    hvx_mul_f32_uuu)
DEFINE_COMPUTE_REPEAT(mul_f16, _Float16, hvx_mul_f16_uuu)
DEFINE_COMPUTE_REPEAT(div_f32, اعشاری,    hvx_div_f32_uuu)
DEFINE_COMPUTE_REPEAT(div_f16, _Float16, hvx_div_f16_uuu)

قسمتعریف خالی (*compute_جوڑو_id_t)(
    uint8_t * dst, ثابت uint8_t * src0, ثابت uint8_t * src1_ڈیٹا, ثابت int32_t * src2_ids,
    uint32_t i01, uint32_t i02, uint32_t ne01, uint32_t src1_stride,
    uint32_t n_صفیں, size_t dst_stride, size_t src0_stride, uint32_t ne00);

ساکن خالی compute_جوڑو_id_f32(
    uint8_t * dst, ثابت uint8_t * src0, ثابت uint8_t * src1_ڈیٹا, ثابت int32_t * src2_ids,
    uint32_t i01, uint32_t i02, uint32_t ne01, uint32_t src1_stride,
    uint32_t n_صفیں, size_t dst_stride, size_t src0_stride, uint32_t ne00) {
    برائے (uint32_t r = 0; r < n_صفیں; r++) {
        uint32_t r_i01 = i01 + r;
        ثابت int32_t idx = src2_ids[i02 * ne01 + r_i01];
        اگر (idx < 0) {
            حافظہ_نقل(dst + r * dst_stride, src0 + r * src0_stride, ne00 * حجم(اعشاری));
            جاری;
        }
        ثابت uint8_t * r_src1 = src1_ڈیٹا + idx * src1_stride;
        ثابت uint8_t * r_src0 = src0 + r * src0_stride;
        uint8_t * r_dst = dst + r * dst_stride;
        hvx_جوڑو_f32_aaa(r_dst, r_src0, r_src1, ne00);
    }
}

// 1. Scalar src1 in VTCM via DMA (ne10 == 1)
ساکن خالی ثنائی_thread_scalar(غیرعلامتی صحیح nth, غیرعلامتی صحیح ith, خالی * ڈیٹا) {
    ساخت htp_ثنائی_context * bctx = (ساخت htp_ثنائی_context *) ڈیٹا;
    ساخت htp_ops_context * octx = bctx->octx;
    htp_ثنائی_preamble;

    ثابت uint32_t elem_سائز = (src0->type == HTP_TYPE_F32) ? حجم(اعشاری) : حجم(_Float16);
    ثابت size_t plane_stride = (size_t) ne12 * ne11 * elem_سائز;
    ثابت size_t صف_stride   = (size_t) ne11 * elem_سائز;

    ثابت uint32_t صف_سائز_بائٹیں = bctx->صف_سائز_بائٹیں;
    ثابت uint32_t آغاز_صف = bctx->صف_آغاز + bctx->nrows_فی_thread * ith;
    ثابت uint32_t انتہا_صف   = MIN(آغاز_صف + bctx->nrows_فی_thread, bctx->صف_آغاز + bctx->کل_صفیں);
    اگر (آغاز_صف >= انتہا_صف) واپس;

    FARF(بلند_2, "binary-scalar: %d/%d (%u:%u) row-size %u (%u)",
         ith, nth, آغاز_صف, انتہا_صف, nb01, bctx->dst_صف_سائز_aligned);

    ثابت ساخت htp_ثنائی_vtcm_layout * layout = &bctx->vtcm_layout;
    uint8_t * src0_spad_بنیاد = VTCM_LAYOUT_PTR(uint8_t, bctx->vtcm_بنیاد, layout->off_src0) + (ith * layout->src0_بائٹیں_فی_thread);
    uint8_t * dst_spad_بنیاد  = VTCM_LAYOUT_PTR(uint8_t, bctx->vtcm_بنیاد, layout->off_dst)  + (ith * layout->dst_بائٹیں_فی_thread);
    size_t src0_spad_half    = layout->src0_spad_half_سائز;
    size_t dst_spad_half     = layout->dst_spad_half_سائز;
    ثابت خالی * s1_جدول    = VTCM_LAYOUT_PTR(ثابت خالی, bctx->vtcm_بنیاد, layout->off_src1);

    dma_قطار * dma_q = octx->ctx->dma[ith];
    uint32_t ir_prefetch = آغاز_صف;
    صحیح spad_idx = 0;

    برائے (صحیح k = 0; k < 2 && ir_prefetch < انتہا_صف; k++) {
        uint32_t current_بلاک_سائز = calc_بلاک_سائز(bctx, ir_prefetch, انتہا_صف, ne01, ne02);
        uint32_t i03, i02, i01, rem;
        i03 = fastdiv(ir_prefetch, &bctx->src0_dim12_div);
        rem = ir_prefetch - i03 * (ne02 * ne01);
        i02 = fastdiv(rem, &bctx->src0_dim1_div);
        i01 = rem - i02 * ne01;

        dma_addr_t src0_curr = src0->ڈیٹا + i03 * nb03 + i02 * nb02 + i01 * nb01;
        dma_addr_t dst_curr  = dst->ڈیٹا  + i03 * nb3  + i02 * nb2  + i01 * nb1;

        uint8_t * s0_spad = src0_spad_بنیاد + spad_idx * src0_spad_half;
        uint8_t * d_spad  = dst_spad_بنیاد  + spad_idx * dst_spad_half;

        dma_قطار_ڈالو(dma_q, dma_make_ڈیٹا(dst_curr, d_spad), nb1, bctx->dst_صف_سائز_aligned, صف_سائز_بائٹیں, 0);
        dma_قطار_ڈالو(dma_q, dma_make_ڈیٹا(s0_spad, src0_curr), bctx->src0_صف_سائز_aligned, nb01, صف_سائز_بائٹیں,
                                                             current_بلاک_سائز);
        ir_prefetch += current_بلاک_سائز;
        spad_idx ^= 1;
    }

    ساخت htp_thread_trace * tr = &octx->ctx->trace[ith];
    compute_scalar_t compute = (compute_scalar_t) bctx->compute;

    برائے (uint32_t ir = آغاز_صف; ir < انتہا_صف; ) {
        uint32_t current_بلاک_سائز = calc_بلاک_سائز(bctx, ir, انتہا_صف, ne01, ne02);

        uint8_t * d_spad  = (uint8_t *) dma_قطار_نکالو(dma_q).src;
        uint8_t * s0_spad = (uint8_t *) dma_قطار_نکالو(dma_q).dst;

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

        htp_trace_event_آغاز(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);
        compute(d_spad, s0_spad, scalar_جدول, i11, ne11, current_بلاک_سائز,
                bctx->dst_صف_سائز_aligned, bctx->src0_صف_سائز_aligned, ne00);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);

        dma_addr_t dst_curr = dst->ڈیٹا + i03 * nb3 + i02 * nb2 + i01 * nb1;
        dma_قطار_ڈالو(dma_q, dma_make_ڈیٹا(dst_curr, d_spad), nb1, bctx->dst_صف_سائز_aligned, صف_سائز_بائٹیں,
                                                                current_بلاک_سائز);

        اگر (ir_prefetch < انتہا_صف) {
            uint32_t اگلا_بلاک_سائز = calc_بلاک_سائز(bctx, ir_prefetch, انتہا_صف, ne01, ne02);
            uint32_t p03, p02, p01, prem;
            p03 = fastdiv(ir_prefetch, &bctx->src0_dim12_div);
            prem = ir_prefetch - p03 * (ne02 * ne01);
            p02 = fastdiv(prem, &bctx->src0_dim1_div);
            p01 = prem - p02 * ne01;
            dma_addr_t s0_اگلا = src0->ڈیٹا + p03 * nb03 + p02 * nb02 + p01 * nb01;
            dma_قطار_ڈالو(dma_q, dma_make_ڈیٹا(s0_spad, s0_اگلا), bctx->src0_صف_سائز_aligned, nb01, صف_سائز_بائٹیں,
                                                               اگلا_بلاک_سائز);
            ir_prefetch += اگلا_بلاک_سائز;
        }
        ir += current_بلاک_سائز;
    }

    dma_قطار_flush(dma_q);
}

// 2. Vector Same Shape (ne1x == ne0x) or Simple Broadcast
ساکن خالی ثنائی_thread_سمتیہ_same_shape(غیرعلامتی صحیح nth, غیرعلامتی صحیح ith, خالی * ڈیٹا) {
    ساخت htp_ثنائی_context * bctx = (ساخت htp_ثنائی_context *) ڈیٹا;
    ساخت htp_ops_context * octx = bctx->octx;
    htp_ثنائی_preamble;

    ثابت uint32_t صف_سائز_بائٹیں = bctx->صف_سائز_بائٹیں;
    ثابت uint32_t آغاز_صف = bctx->صف_آغاز + bctx->nrows_فی_thread * ith;
    ثابت uint32_t انتہا_صف   = MIN(آغاز_صف + bctx->nrows_فی_thread, bctx->صف_آغاز + bctx->کل_صفیں);
    اگر (آغاز_صف >= انتہا_صف) واپس;

    FARF(بلند_2, "binary-same-shape: %d/%d (%u:%u) row-size %u (%u)",
         ith, nth, آغاز_صف, انتہا_صف, nb01, bctx->dst_صف_سائز_aligned);

    ثابت ساخت htp_ثنائی_vtcm_layout * layout = &bctx->vtcm_layout;
    uint8_t * src0_spad_بنیاد = VTCM_LAYOUT_PTR(uint8_t, bctx->vtcm_بنیاد, layout->off_src0) + (ith * layout->src0_بائٹیں_فی_thread);
    uint8_t * src1_spad_بنیاد = VTCM_LAYOUT_PTR(uint8_t, bctx->vtcm_بنیاد, layout->off_src1) + (ith * layout->src1_بائٹیں_فی_thread);
    uint8_t * dst_spad_بنیاد  = VTCM_LAYOUT_PTR(uint8_t, bctx->vtcm_بنیاد, layout->off_dst)  + (ith * layout->dst_بائٹیں_فی_thread);

    size_t src0_spad_half = layout->src0_spad_half_سائز;
    size_t src1_spad_half = layout->src1_spad_half_سائز;
    size_t dst_spad_half  = layout->dst_spad_half_سائز;

    dma_قطار * dma_q = octx->ctx->dma[ith];
    uint32_t ir_prefetch = آغاز_صف;
    صحیح spad_idx = 0;

    برائے (صحیح k = 0; k < 2 && ir_prefetch < انتہا_صف; k++) {
        uint32_t current_بلاک_سائز = calc_بلاک_سائز(bctx, ir_prefetch, انتہا_صف, ne01, ne02);
        uint32_t i03, i02, i01, rem;
        i03 = fastdiv(ir_prefetch, &bctx->src0_dim12_div);
        rem = ir_prefetch - i03 * (ne02 * ne01);
        i02 = fastdiv(rem, &bctx->src0_dim1_div);
        i01 = rem - i02 * ne01;

        uint32_t i13 = (ne13 == 1) ? 0 : i03;
        uint32_t i12 = (ne12 == 1) ? 0 : i02;
        uint32_t i11 = (ne11 == 1) ? 0 : i01;

        dma_addr_t src0_curr = src0->ڈیٹا + i03 * nb03 + i02 * nb02 + i01 * nb01;
        dma_addr_t src1_curr = src1->ڈیٹا + i13 * nb13 + i12 * nb12 + i11 * nb11;
        dma_addr_t dst_curr  = dst->ڈیٹا  + i03 * nb3  + i02 * nb2  + i01 * nb1;

        uint8_t * s0_spad = src0_spad_بنیاد + spad_idx * src0_spad_half;
        uint8_t * s1_spad = src1_spad_بنیاد + spad_idx * src1_spad_half;
        uint8_t * d_spad  = dst_spad_بنیاد  + spad_idx * dst_spad_half;

        dma_قطار_ڈالو(dma_q, dma_make_ڈیٹا(dst_curr, d_spad), nb1, bctx->dst_صف_سائز_aligned, صف_سائز_بائٹیں, 0);
        dma_قطار_ڈالو(dma_q, dma_make_ڈیٹا(s0_spad, src0_curr), bctx->src0_صف_سائز_aligned, nb01, صف_سائز_بائٹیں, current_بلاک_سائز);
        dma_قطار_ڈالو(dma_q, dma_make_ڈیٹا(s1_spad, src1_curr), bctx->src1_صف_سائز_aligned, nb11, صف_سائز_بائٹیں, current_بلاک_سائز);
        ir_prefetch += current_بلاک_سائز;
        spad_idx ^= 1;
    }

    ساخت htp_thread_trace * tr = &octx->ctx->trace[ith];
    compute_same_shape_t compute = (compute_same_shape_t) bctx->compute;

    برائے (uint32_t ir = آغاز_صف; ir < انتہا_صف; ) {
        uint32_t current_بلاک_سائز = calc_بلاک_سائز(bctx, ir, انتہا_صف, ne01, ne02);
        uint8_t * d_spad  = (uint8_t *) dma_قطار_نکالو(dma_q).src;
        uint8_t * s0_spad = (uint8_t *) dma_قطار_نکالو(dma_q).dst;
        uint8_t * s1_spad = (uint8_t *) dma_قطار_نکالو(dma_q).dst;

        htp_trace_event_آغاز(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);
        compute(d_spad, s0_spad, s1_spad, current_بلاک_سائز,
                bctx->dst_صف_سائز_aligned, bctx->src0_صف_سائز_aligned, bctx->src1_صف_سائز_aligned, ne00);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);

        uint32_t i03, i02, i01, rem;
        i03 = fastdiv(ir, &bctx->src0_dim12_div);
        rem = ir - i03 * (ne02 * ne01);
        i02 = fastdiv(rem, &bctx->src0_dim1_div);
        i01 = rem - i02 * ne01;
        dma_addr_t dst_curr = dst->ڈیٹا + i03 * nb3 + i02 * nb2 + i01 * nb1;
        dma_قطار_ڈالو(dma_q, dma_make_ڈیٹا(dst_curr, d_spad), nb1, bctx->dst_صف_سائز_aligned, صف_سائز_بائٹیں, current_بلاک_سائز);

        اگر (ir_prefetch < انتہا_صف) {
            uint32_t اگلا_بلاک_سائز = calc_بلاک_سائز(bctx, ir_prefetch, انتہا_صف, ne01, ne02);
            uint32_t p03, p02, p01, prem;
            p03 = fastdiv(ir_prefetch, &bctx->src0_dim12_div);
            prem = ir_prefetch - p03 * (ne02 * ne01);
            p02 = fastdiv(prem, &bctx->src0_dim1_div);
            p01 = prem - p02 * ne01;

            uint32_t p13 = (ne13 == 1) ? 0 : p03;
            uint32_t p12 = (ne12 == 1) ? 0 : p02;
            uint32_t p11 = (ne11 == 1) ? 0 : p01;

            dma_addr_t s0_اگلا = src0->ڈیٹا + p03 * nb03 + p02 * nb02 + p01 * nb01;
            dma_addr_t s1_اگلا = src1->ڈیٹا + p13 * nb13 + p12 * nb12 + p11 * nb11;

            dma_قطار_ڈالو(dma_q, dma_make_ڈیٹا(s0_spad, s0_اگلا), bctx->src0_صف_سائز_aligned, nb01, صف_سائز_بائٹیں, اگلا_بلاک_سائز);
            dma_قطار_ڈالو(dma_q, dma_make_ڈیٹا(s1_spad, s1_اگلا), bctx->src1_صف_سائز_aligned, nb11, صف_سائز_بائٹیں, اگلا_بلاک_سائز);

            ir_prefetch += اگلا_بلاک_سائز;
        }
        ir += current_بلاک_سائز;
    }

    dma_قطار_flush(dma_q);
}

// 3. Row Broadcast (ne11 == 1, ne12 == 1, single row src1)
ساکن خالی ثنائی_thread_سمتیہ_صف_نشر(غیرعلامتی صحیح nth, غیرعلامتی صحیح ith, خالی * ڈیٹا) {
    ساخت htp_ثنائی_context * bctx = (ساخت htp_ثنائی_context *) ڈیٹا;
    ساخت htp_ops_context * octx = bctx->octx;
    htp_ثنائی_preamble;

    ثابت uint32_t صف_سائز_بائٹیں = bctx->صف_سائز_بائٹیں;
    ثابت uint32_t آغاز_صف  = bctx->صف_آغاز + bctx->nrows_فی_thread * ith;
    ثابت uint32_t انتہا_صف    = MIN(آغاز_صف + bctx->nrows_فی_thread, bctx->صف_آغاز + bctx->کل_صفیں);
    اگر (آغاز_صف >= انتہا_صف) واپس;

    FARF(بلند_2, "binary-row-bcast: %d/%d (%u:%u) row-size %u (%u)",
         ith, nth, آغاز_صف, انتہا_صف, nb01, bctx->dst_صف_سائز_aligned);

    ثابت ساخت htp_ثنائی_vtcm_layout * layout = &bctx->vtcm_layout;
    uint8_t * src0_spad_بنیاد = VTCM_LAYOUT_PTR(uint8_t, bctx->vtcm_بنیاد, layout->off_src0) + (ith * layout->src0_بائٹیں_فی_thread);
    uint8_t * dst_spad_بنیاد  = VTCM_LAYOUT_PTR(uint8_t, bctx->vtcm_بنیاد, layout->off_dst)  + (ith * layout->dst_بائٹیں_فی_thread);

    size_t src0_spad_half = layout->src0_spad_half_سائز;
    size_t dst_spad_half  = layout->dst_spad_half_سائز;

    dma_قطار * dma_q = octx->ctx->dma[ith];
    uint32_t ir_prefetch = آغاز_صف;
    صحیح spad_idx = 0;

    خالی * s1_ptr = VTCM_LAYOUT_PTR(خالی, bctx->vtcm_بنیاد, layout->off_src1);

    برائے (صحیح k = 0; k < 2 && ir_prefetch < انتہا_صف; k++) {
        uint32_t current_بلاک_سائز = calc_بلاک_سائز(bctx, ir_prefetch, انتہا_صف, ne01, ne02);
        uint32_t i03 = fastdiv(ir_prefetch, &bctx->src0_dim12_div);
        uint32_t rem = ir_prefetch - i03 * (ne02 * ne01);
        uint32_t i02 = fastdiv(rem, &bctx->src0_dim1_div);
        uint32_t i01 = rem - i02 * ne01;

        dma_addr_t src0_curr = src0->ڈیٹا + i03 * nb03 + i02 * nb02 + i01 * nb01;
        dma_addr_t dst_curr  = dst->ڈیٹا  + i03 * nb3  + i02 * nb2  + i01 * nb1;

        uint8_t * s0_spad = src0_spad_بنیاد + spad_idx * src0_spad_half;
        uint8_t * d_spad  = dst_spad_بنیاد  + spad_idx * dst_spad_half;

        dma_قطار_ڈالو(dma_q, dma_make_ڈیٹا(dst_curr, d_spad), nb1, bctx->dst_صف_سائز_aligned, صف_سائز_بائٹیں, 0);
        dma_قطار_ڈالو(dma_q, dma_make_ڈیٹا(s0_spad, src0_curr), bctx->src0_صف_سائز_aligned, nb01, صف_سائز_بائٹیں, current_بلاک_سائز);
        ir_prefetch += current_بلاک_سائز;
        spad_idx ^= 1;
    }

    ساخت htp_thread_trace * tr = &octx->ctx->trace[ith];
    compute_صف_bcast_t compute = (compute_صف_bcast_t) bctx->compute;

    برائے (uint32_t ir = آغاز_صف; ir < انتہا_صف; ) {
        uint32_t current_بلاک_سائز = calc_بلاک_سائز(bctx, ir, انتہا_صف, ne01, ne02);
        uint8_t * d_spad  = (uint8_t *) dma_قطار_نکالو(dma_q).src;
        uint8_t * s0_spad = (uint8_t *) dma_قطار_نکالو(dma_q).dst;

        htp_trace_event_آغاز(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);
        compute(d_spad, s0_spad, (ثابت uint8_t *)s1_ptr, current_بلاک_سائز,
                bctx->dst_صف_سائز_aligned, bctx->src0_صف_سائز_aligned, ne00);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);

        uint32_t i03 = fastdiv(ir, &bctx->src0_dim12_div);
        uint32_t rem = ir - i03 * (ne02 * ne01);
        uint32_t i02 = fastdiv(rem, &bctx->src0_dim1_div);
        uint32_t i01 = rem - i02 * ne01;
        dma_addr_t dst_curr = dst->ڈیٹا + i03 * nb3 + i02 * nb2 + i01 * nb1;
        dma_قطار_ڈالو(dma_q, dma_make_ڈیٹا(dst_curr, d_spad), nb1, bctx->dst_صف_سائز_aligned, صف_سائز_بائٹیں, current_بلاک_سائز);

        اگر (ir_prefetch < انتہا_صف) {
            uint32_t اگلا_بلاک_سائز = calc_بلاک_سائز(bctx, ir_prefetch, انتہا_صف, ne01, ne02);
            uint32_t p03, p02, p01, prem;
            p03  = fastdiv(ir_prefetch, &bctx->src0_dim12_div);
            prem = ir_prefetch - p03 * (ne02 * ne01);
            p02  = fastdiv(prem, &bctx->src0_dim1_div);
            p01  = prem - p02 * ne01;
            dma_addr_t s0_اگلا = src0->ڈیٹا + p03 * nb03 + p02 * nb02 + p01 * nb01;
            dma_قطار_ڈالو(dma_q, dma_make_ڈیٹا(s0_spad, s0_اگلا), bctx->src0_صف_سائز_aligned, nb01, صف_سائز_بائٹیں, اگلا_بلاک_سائز);
            ir_prefetch += اگلا_بلاک_سائز;
        }
        ir += current_بلاک_سائز;
    }

    dma_قطار_flush(dma_q);
}

// 4. Vector Complex (ne10 == ne00, complex broadcast)
ساکن خالی ثنائی_thread_سمتیہ_complex(غیرعلامتی صحیح nth, غیرعلامتی صحیح ith, خالی * ڈیٹا) {
    ساخت htp_ثنائی_context * bctx = (ساخت htp_ثنائی_context *) ڈیٹا;
    ساخت htp_ops_context * octx = bctx->octx;
    htp_ثنائی_preamble;

    ثابت uint32_t صف_سائز_بائٹیں = bctx->صف_سائز_بائٹیں;
    ثابت uint32_t آغاز_صف  = bctx->صف_آغاز + bctx->nrows_فی_thread * ith;
    ثابت uint32_t انتہا_صف    = MIN(آغاز_صف + bctx->nrows_فی_thread, bctx->صف_آغاز + bctx->کل_صفیں);
    اگر (آغاز_صف >= انتہا_صف) واپس;

    FARF(بلند_2, "binary-complex: %d/%d (%u:%u) row-size %u (%u)",
         ith, nth, آغاز_صف, انتہا_صف, nb01, bctx->dst_صف_سائز_aligned);

    ثابت ساخت htp_ثنائی_vtcm_layout * layout = &bctx->vtcm_layout;
    uint8_t * src0_spad_بنیاد = VTCM_LAYOUT_PTR(uint8_t, bctx->vtcm_بنیاد, layout->off_src0) + (ith * layout->src0_بائٹیں_فی_thread);
    uint8_t * dst_spad_بنیاد  = VTCM_LAYOUT_PTR(uint8_t, bctx->vtcm_بنیاد, layout->off_dst)  + (ith * layout->dst_بائٹیں_فی_thread);
    size_t src0_spad_half    = layout->src0_spad_half_سائز;
    size_t dst_spad_half     = layout->dst_spad_half_سائز;
    ثابت uint8_t * vtcm_src1 = VTCM_LAYOUT_PTR(ثابت uint8_t, bctx->vtcm_بنیاد, layout->off_src1);

    ثابت uint32_t elem_سائز = (src0->type == HTP_TYPE_F32) ? حجم(اعشاری) : حجم(_Float16);
    ثابت size_t vtcm_صف_stride   = (size_t) ne10 * elem_سائز;
    ثابت size_t vtcm_plane_stride = (size_t) ne11 * vtcm_صف_stride;
    ثابت size_t vtcm_vol_stride   = (size_t) ne12 * vtcm_plane_stride;

    dma_قطار * dma_q = octx->ctx->dma[ith];
    uint32_t ir_prefetch = آغاز_صف;
    صحیح spad_idx = 0;

    برائے (صحیح k = 0; k < 2 && ir_prefetch < انتہا_صف; k++) {
        uint32_t current_بلاک_سائز = calc_بلاک_سائز(bctx, ir_prefetch, انتہا_صف, ne01, ne02);
        uint32_t i03 = fastdiv(ir_prefetch, &bctx->src0_dim12_div);
        uint32_t rem = ir_prefetch - i03 * (ne02 * ne01);
        uint32_t i02 = fastdiv(rem, &bctx->src0_dim1_div);
        uint32_t i01 = rem - i02 * ne01;

        dma_addr_t src0_curr = src0->ڈیٹا + i03 * nb03 + i02 * nb02 + i01 * nb01;
        dma_addr_t dst_curr  = dst->ڈیٹا  + i03 * nb3  + i02 * nb2  + i01 * nb1;

        uint8_t * s0_spad = src0_spad_بنیاد + spad_idx * src0_spad_half;
        uint8_t * d_spad  = dst_spad_بنیاد  + spad_idx * dst_spad_half;

        dma_قطار_ڈالو(dma_q, dma_make_ڈیٹا(dst_curr, d_spad), nb1, bctx->dst_صف_سائز_aligned, صف_سائز_بائٹیں, 0);
        dma_قطار_ڈالو(dma_q, dma_make_ڈیٹا(s0_spad, src0_curr), bctx->src0_صف_سائز_aligned, nb01, صف_سائز_بائٹیں, current_بلاک_سائز);
        ir_prefetch += current_بلاک_سائز;
        spad_idx ^= 1;
    }

    ساخت htp_thread_trace * tr = &octx->ctx->trace[ith];
    compute_complex_t compute = (compute_complex_t) bctx->compute;

    برائے (uint32_t ir = آغاز_صف; ir < انتہا_صف; ) {
        uint32_t current_بلاک_سائز = calc_بلاک_سائز(bctx, ir, انتہا_صف, ne01, ne02);
        uint8_t * d_spad = (uint8_t *) dma_قطار_نکالو(dma_q).src;
        uint8_t * s0_spad = (uint8_t *) dma_قطار_نکالو(dma_q).dst;

        uint32_t i03 = fastdiv(ir, &bctx->src0_dim12_div);
        uint32_t rem = ir - i03 * (ne02 * ne01);
        uint32_t i02 = fastdiv(rem, &bctx->src0_dim1_div);
        uint32_t i01 = rem - i02 * ne01;

        uint32_t i13 = fastmodulo(i03, ne13, &bctx->src1_dim3_div);
        uint32_t i12 = fastmodulo(i02, ne12, &bctx->src1_dim2_div);
        ثابت uint8_t * src1_plane = vtcm_src1 + i13 * vtcm_vol_stride + i12 * vtcm_plane_stride;

        htp_trace_event_آغاز(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);
        compute(d_spad, s0_spad, src1_plane, i01, ne11, &bctx->src1_dim1_div, vtcm_صف_stride,
                current_بلاک_سائز, bctx->dst_صف_سائز_aligned, bctx->src0_صف_سائز_aligned, ne00);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);

        dma_addr_t dst_curr = dst->ڈیٹا + i03 * nb3 + i02 * nb2 + i01 * nb1;
        dma_قطار_ڈالو(dma_q, dma_make_ڈیٹا(dst_curr, d_spad), nb1, bctx->dst_صف_سائز_aligned, صف_سائز_بائٹیں, current_بلاک_سائز);

        اگر (ir_prefetch < انتہا_صف) {
            uint32_t اگلا_بلاک_سائز = calc_بلاک_سائز(bctx, ir_prefetch, انتہا_صف, ne01, ne02);
            uint32_t p03, p02, p01, prem;
            p03  = fastdiv(ir_prefetch, &bctx->src0_dim12_div);
            prem = ir_prefetch - p03 * (ne02 * ne01);
            p02  = fastdiv(prem, &bctx->src0_dim1_div);
            p01  = prem - p02 * ne01;
            dma_addr_t s0_اگلا = src0->ڈیٹا + p03 * nb03 + p02 * nb02 + p01 * nb01;
            dma_قطار_ڈالو(dma_q, dma_make_ڈیٹا(s0_spad, s0_اگلا), bctx->src0_صف_سائز_aligned, nb01, صف_سائز_بائٹیں, اگلا_بلاک_سائز);
            ir_prefetch += اگلا_بلاک_سائز;
        }
        ir += current_بلاک_سائز;
    }

    dma_قطار_flush(dma_q);
}

// 5. Element Repeat (ne10 != ne00)
ساکن خالی ثنائی_thread_element_repeat(غیرعلامتی صحیح nth, غیرعلامتی صحیح ith, خالی * ڈیٹا) {
    ساخت htp_ثنائی_context * bctx = (ساخت htp_ثنائی_context *) ڈیٹا;
    ساخت htp_ops_context * octx = bctx->octx;
    htp_ثنائی_preamble;

    ثابت uint32_t صف_سائز_بائٹیں = bctx->صف_سائز_بائٹیں;
    ثابت uint32_t آغاز_صف  = bctx->صف_آغاز + bctx->nrows_فی_thread * ith;
    ثابت uint32_t انتہا_صف    = MIN(آغاز_صف + bctx->nrows_فی_thread, bctx->صف_آغاز + bctx->کل_صفیں);
    اگر (آغاز_صف >= انتہا_صف) واپس;

    ثابت ساخت htp_ثنائی_vtcm_layout * layout = &bctx->vtcm_layout;
    uint8_t * src0_spad_بنیاد = VTCM_LAYOUT_PTR(uint8_t, bctx->vtcm_بنیاد, layout->off_src0) + (ith * layout->src0_بائٹیں_فی_thread);
    uint8_t * dst_spad_بنیاد  = VTCM_LAYOUT_PTR(uint8_t, bctx->vtcm_بنیاد, layout->off_dst)  + (ith * layout->dst_بائٹیں_فی_thread);
    size_t src0_spad_half    = layout->src0_spad_half_سائز;
    size_t dst_spad_half     = layout->dst_spad_half_سائز;
    ثابت uint8_t * vtcm_src1 = VTCM_LAYOUT_PTR(ثابت uint8_t, bctx->vtcm_بنیاد, layout->off_src1);

    ثابت uint32_t elem_سائز = (src0->type == HTP_TYPE_F32) ? حجم(اعشاری) : حجم(_Float16);
    ثابت size_t vtcm_صف_stride   = (size_t) ne10 * elem_سائز;
    ثابت size_t vtcm_plane_stride = (size_t) ne11 * vtcm_صف_stride;
    ثابت size_t vtcm_vol_stride   = (size_t) ne12 * vtcm_plane_stride;

    FARF(بلند_2, "binary-repeat: %d/%d (%u:%u) row-size %u (%u)",
         ith, nth, آغاز_صف, انتہا_صف, nb01, bctx->dst_صف_سائز_aligned);

    dma_قطار * dma_q = octx->ctx->dma[ith];
    uint32_t ir_prefetch = آغاز_صف;
    صحیح spad_idx = 0;

    برائے (صحیح k = 0; k < 2 && ir_prefetch < انتہا_صف; k++) {
        uint32_t current_بلاک_سائز = calc_بلاک_سائز(bctx, ir_prefetch, انتہا_صف, ne01, ne02);
        uint32_t i03 = fastdiv(ir_prefetch, &bctx->src0_dim12_div);
        uint32_t rem = ir_prefetch - i03 * (ne02 * ne01);
        uint32_t i02 = fastdiv(rem, &bctx->src0_dim1_div);
        uint32_t i01 = rem - i02 * ne01;

        dma_addr_t src0_curr = src0->ڈیٹا + i03 * nb03 + i02 * nb02 + i01 * nb01;
        dma_addr_t dst_curr  = dst->ڈیٹا  + i03 * nb3  + i02 * nb2  + i01 * nb1;

        uint8_t * s0_spad = src0_spad_بنیاد + spad_idx * src0_spad_half;
        uint8_t * d_spad  = dst_spad_بنیاد  + spad_idx * dst_spad_half;

        dma_قطار_ڈالو(dma_q, dma_make_ڈیٹا(dst_curr, d_spad), nb1, bctx->dst_صف_سائز_aligned, صف_سائز_بائٹیں, 0);
        dma_قطار_ڈالو(dma_q, dma_make_ڈیٹا(s0_spad, src0_curr), bctx->src0_صف_سائز_aligned, nb01, صف_سائز_بائٹیں, current_بلاک_سائز);
        ir_prefetch += current_بلاک_سائز;
        spad_idx ^= 1;
    }

    ساخت htp_thread_trace * tr = &octx->ctx->trace[ith];
    compute_repeat_t compute = (compute_repeat_t) bctx->compute;

    برائے (uint32_t ir = آغاز_صف; ir < انتہا_صف; ) {
        uint32_t current_بلاک_سائز = calc_بلاک_سائز(bctx, ir, انتہا_صف, ne01, ne02);
        uint8_t * d_spad  = (uint8_t *) dma_قطار_نکالو(dma_q).src;
        uint8_t * s0_spad = (uint8_t *) dma_قطار_نکالو(dma_q).dst;

        uint32_t i03 = fastdiv(ir, &bctx->src0_dim12_div);
        uint32_t rem = ir - i03 * (ne02 * ne01);
        uint32_t i02 = fastdiv(rem, &bctx->src0_dim1_div);
        uint32_t i01 = rem - i02 * ne01;

        uint32_t i13 = fastmodulo(i03, ne13, &bctx->src1_dim3_div);
        uint32_t i12 = fastmodulo(i02, ne12, &bctx->src1_dim2_div);
        ثابت uint8_t * src1_plane = vtcm_src1 + i13 * vtcm_vol_stride + i12 * vtcm_plane_stride;

        htp_trace_event_آغاز(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);
        compute(d_spad, s0_spad, src1_plane, i01, ne11, &bctx->src1_dim1_div, vtcm_صف_stride,
                current_بلاک_سائز, bctx->dst_صف_سائز_aligned, bctx->src0_صف_سائز_aligned, ne00, ne10);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);

        dma_addr_t dst_curr = dst->ڈیٹا + i03 * nb3 + i02 * nb2 + i01 * nb1;
        dma_قطار_ڈالو(dma_q, dma_make_ڈیٹا(dst_curr, d_spad), nb1, bctx->dst_صف_سائز_aligned, صف_سائز_بائٹیں, current_بلاک_سائز);

        اگر (ir_prefetch < انتہا_صف) {
            uint32_t اگلا_بلاک_سائز = calc_بلاک_سائز(bctx, ir_prefetch, انتہا_صف, ne01, ne02);
            uint32_t p03, p02, p01, prem;
            p03  = fastdiv(ir_prefetch, &bctx->src0_dim12_div);
            prem = ir_prefetch - p03 * (ne02 * ne01);
            p02  = fastdiv(prem, &bctx->src0_dim1_div);
            p01  = prem - p02 * ne01;
            dma_addr_t s0_اگلا = src0->ڈیٹا + p03 * nb03 + p02 * nb02 + p01 * nb01;
            dma_قطار_ڈالو(dma_q, dma_make_ڈیٹا(s0_spad, s0_اگلا), bctx->src0_صف_سائز_aligned, nb01, صف_سائز_بائٹیں, اگلا_بلاک_سائز);
            ir_prefetch += اگلا_بلاک_سائز;
        }
        ir += current_بلاک_سائز;
    }

    dma_قطار_flush(dma_q);
}

// 6. ADD_ID (src1 gathered via src2 indices)
ساکن خالی ثنائی_thread_جوڑو_id_f32(غیرعلامتی صحیح nth, غیرعلامتی صحیح ith, خالی * ڈیٹا) {
    ساخت htp_ثنائی_context * bctx = (ساخت htp_ثنائی_context *) ڈیٹا;
    ساخت htp_ops_context * octx = bctx->octx;

    ثابت ساخت htp_tensor * src0 = octx->src[0];
    ثابت ساخت htp_tensor * src1 = octx->src[1];
    ثابت ساخت htp_tensor * dst  = octx->dst;

    ثابت uint32_t ne00 = src0->ne[0];
    ثابت uint32_t ne01 = src0->ne[1];
    ثابت uint32_t ne02 = src0->ne[2];
    ثابت uint32_t ne03 = src0->ne[3];

    ثابت uint32_t nb01 = src0->nb[1];
    ثابت uint32_t nb02 = src0->nb[2];
    ثابت uint32_t nb03 = src0->nb[3];
    ثابت uint32_t src1_stride = bctx->src1_صف_سائز_aligned;

    ثابت uint32_t nb1 = dst->nb[1];
    ثابت uint32_t nb2 = dst->nb[2];
    ثابت uint32_t nb3 = dst->nb[3];

    ثابت uint32_t صف_سائز_بائٹیں = bctx->صف_سائز_بائٹیں;
    ثابت uint32_t آغاز_صف = bctx->صف_آغاز + bctx->nrows_فی_thread * ith;
    ثابت uint32_t انتہا_صف   = MIN(آغاز_صف + bctx->nrows_فی_thread, bctx->صف_آغاز + bctx->کل_صفیں);
    اگر (آغاز_صف >= انتہا_صف) واپس;

    ثابت ساخت htp_ثنائی_vtcm_layout * layout = &bctx->vtcm_layout;
    uint8_t * src0_spad_بنیاد = VTCM_LAYOUT_PTR(uint8_t, bctx->vtcm_بنیاد, layout->off_src0) + (ith * layout->src0_بائٹیں_فی_thread);
    uint8_t * dst_spad_بنیاد  = VTCM_LAYOUT_PTR(uint8_t, bctx->vtcm_بنیاد, layout->off_dst)  + (ith * layout->dst_بائٹیں_فی_thread);
    ثابت uint8_t * vtcm_src1 = VTCM_LAYOUT_PTR(ثابت uint8_t, bctx->vtcm_بنیاد, layout->off_src1);
    ثابت int32_t * vtcm_src2 = VTCM_LAYOUT_PTR(ثابت int32_t, bctx->vtcm_بنیاد, layout->off_src2);
    size_t src0_spad_half    = layout->src0_spad_half_سائز;
    size_t dst_spad_half     = layout->dst_spad_half_سائز;

    dma_قطار * dma_q = octx->ctx->dma[ith];
    uint32_t ir_prefetch = آغاز_صف;
    صحیح spad_idx = 0;

    برائے (صحیح k = 0; k < 2 && ir_prefetch < انتہا_صف; k++) {
        uint32_t current_بلاک_سائز = calc_بلاک_سائز(bctx, ir_prefetch, انتہا_صف, ne01, ne02);
        uint32_t i03 = fastdiv(ir_prefetch, &bctx->src0_dim12_div);
        uint32_t rem = ir_prefetch - i03 * (ne02 * ne01);
        uint32_t i02 = fastdiv(rem, &bctx->src0_dim1_div);
        uint32_t i01 = rem - i02 * ne01;

        dma_addr_t src0_curr = src0->ڈیٹا + i03 * nb03 + i02 * nb02 + i01 * nb01;
        dma_addr_t dst_curr  = dst->ڈیٹا  + i03 * nb3  + i02 * nb2  + i01 * nb1;

        uint8_t * s0_spad = src0_spad_بنیاد + spad_idx * src0_spad_half;
        uint8_t * d_spad  = dst_spad_بنیاد  + spad_idx * dst_spad_half;

        dma_قطار_ڈالو(dma_q, dma_make_ڈیٹا(dst_curr, d_spad), nb1, bctx->dst_صف_سائز_aligned, صف_سائز_بائٹیں, 0);
        dma_قطار_ڈالو(dma_q, dma_make_ڈیٹا(s0_spad, src0_curr), bctx->src0_صف_سائز_aligned, nb01, صف_سائز_بائٹیں, current_بلاک_سائز);
        ir_prefetch += current_بلاک_سائز;
        spad_idx ^= 1;
    }

    ساخت htp_thread_trace * tr = &octx->ctx->trace[ith];

    برائے (uint32_t ir = آغاز_صف; ir < انتہا_صف; ) {
        uint32_t current_بلاک_سائز = calc_بلاک_سائز(bctx, ir, انتہا_صف, ne01, ne02);
        uint8_t * d_spad = (uint8_t *) dma_قطار_نکالو(dma_q).src;
        uint8_t * s0_spad = (uint8_t *) dma_قطار_نکالو(dma_q).dst;

        uint32_t i03 = fastdiv(ir, &bctx->src0_dim12_div);
        uint32_t rem = ir - i03 * (ne02 * ne01);
        uint32_t i02 = fastdiv(rem, &bctx->src0_dim1_div);
        uint32_t i01 = rem - i02 * ne01;

        htp_trace_event_آغاز(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);
        compute_جوڑو_id_t compute = (compute_جوڑو_id_t) bctx->compute;
        compute(d_spad, s0_spad, vtcm_src1, vtcm_src2,
                i01, i02, ne01, src1_stride,
                current_بلاک_سائز, bctx->dst_صف_سائز_aligned, bctx->src0_صف_سائز_aligned, ne00);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);

        dma_addr_t dst_curr = dst->ڈیٹا + i03 * nb3 + i02 * nb2 + i01 * nb1;
        dma_قطار_ڈالو(dma_q, dma_make_ڈیٹا(dst_curr, d_spad), nb1, bctx->dst_صف_سائز_aligned, صف_سائز_بائٹیں, current_بلاک_سائز);

        اگر (ir_prefetch < انتہا_صف) {
             uint32_t اگلا_بلاک_سائز = calc_بلاک_سائز(bctx, ir_prefetch, انتہا_صف, ne01, ne02);
             uint32_t p03, p02, p01, prem;
             p03  = fastdiv(ir_prefetch, &bctx->src0_dim12_div);
             prem = ir_prefetch - p03 * (ne02 * ne01);
             p02  = fastdiv(prem, &bctx->src0_dim1_div);
             p01  = prem - p02 * ne01;
             dma_addr_t s0_اگلا = src0->ڈیٹا + p03 * nb03 + p02 * nb02 + p01 * nb01;
             dma_قطار_ڈالو(dma_q, dma_make_ڈیٹا(s0_spad, s0_اگلا), bctx->src0_صف_سائز_aligned, nb01, صف_سائز_بائٹیں, اگلا_بلاک_سائز);
             ir_prefetch += اگلا_بلاک_سائز;
        }
        ir += current_بلاک_سائز;
    }

    dma_قطار_flush(dma_q);
}

ساکن اندرونی خالی hvx_div_scalar_f32(uint8_t * محدود dst, ثابت uint8_t * محدود src, ثابت اعشاری val, ثابت uint32_t عدد_elems) {
    hvx_mul_scalar_f32(dst, src, 1.0f / val, عدد_elems);
}

ساکن اندرونی خالی hvx_div_scalar_f16(uint8_t * محدود dst, ثابت uint8_t * محدود src, ثابت _Float16 val, ثابت uint32_t عدد_elems) {
    hvx_div_scalar_f16_aa(dst, src, val, عدد_elems);
}

قسمتعریف خالی (*compute_ثنائی_chunked_t)(
    uint8_t * محدود dst,
    ثابت uint8_t * محدود src0,
    ثابت uint8_t * محدود src1,
    ثابت uint32_t عدد_elems
);

قسمتعریف خالی (*compute_ثنائی_scalar_chunked_f32_t)(
    uint8_t * محدود dst,
    ثابت uint8_t * محدود src0,
    ثابت اعشاری val,
    ثابت uint32_t عدد_elems
);

قسمتعریف خالی (*compute_ثنائی_scalar_chunked_f16_t)(
    uint8_t * محدود dst,
    ثابت uint8_t * محدود src0,
    ثابت _Float16 val,
    ثابت uint32_t عدد_elems
);

ساخت ثنائی_chunked_context {
    ساخت htp_ops_context * octx;
    ساخت htp_ثنائی_vtcm_layout vtcm_layout;
    uint8_t *                vtcm_بنیاد;
    uint32_t                 elem_آغاز;
    uint32_t                 nelem;
    uint32_t                 chunk_سائز;
    uint32_t                 chunks_فی_thread;
    uint32_t                 کل_chunks;
    bool                     is_scalar;
    اعشاری                    scalar_f32;
    _Float16                 scalar_f16;
    compute_ثنائی_chunked_t compute;
    compute_ثنائی_scalar_chunked_f32_t compute_scalar_f32;
    compute_ثنائی_scalar_chunked_f16_t compute_scalar_f16;
};

ساکن خالی ثنائی_thread_chunked(غیرعلامتی صحیح nth, غیرعلامتی صحیح ith, خالی * ڈیٹا) {
    (خالی) nth;
    ساخت ثنائی_chunked_context * ctx = (ساخت ثنائی_chunked_context *) ڈیٹا;
    ساخت htp_ops_context * octx = ctx->octx;
    ثابت ساخت htp_tensor * src0 = octx->src[0];
    ثابت ساخت htp_tensor * src1 = octx->src[1];
    ثابت ساخت htp_tensor * dst  = octx->dst;

    ثابت uint32_t آغاز_chunk = ctx->chunks_فی_thread * ith;
    ثابت uint32_t انتہا_chunk   = MIN(آغاز_chunk + ctx->chunks_فی_thread, ctx->کل_chunks);
    اگر (آغاز_chunk >= انتہا_chunk) {
        واپس;
    }

    ثابت uint32_t src0_type   = src0->type;
    ثابت size_t   elem_سائز   = (src0_type == HTP_TYPE_F32) ? حجم(اعشاری) : حجم(_Float16);
    ثابت uint32_t chunk_سائز  = ctx->chunk_سائز;
    ثابت size_t   chunk_بائٹیں = ctx->vtcm_layout.src0_spad_half_سائز;

    FARF(بلند_2, "binary-chunked: %d/%d (%u:%u) chunks %u elems %u",
         ith, nth, آغاز_chunk, انتہا_chunk, ctx->کل_chunks, ctx->nelem);

    ثابت ساخت htp_ثنائی_vtcm_layout * layout = &ctx->vtcm_layout;
    uint8_t * src0_spad_بنیاد = VTCM_LAYOUT_PTR(uint8_t, ctx->vtcm_بنیاد, layout->off_src0) + (ith * layout->src0_بائٹیں_فی_thread);
    uint8_t * src1_spad_بنیاد = ctx->is_scalar ? تہی : (VTCM_LAYOUT_PTR(uint8_t, ctx->vtcm_بنیاد, layout->off_src1) + (ith * layout->src1_بائٹیں_فی_thread));
    uint8_t * dst_spad_بنیاد  = VTCM_LAYOUT_PTR(uint8_t, ctx->vtcm_بنیاد, layout->off_dst)  + (ith * layout->dst_بائٹیں_فی_thread);

    dma_قطار * dma_q = octx->ctx->dma[ith];
    uint32_t prefetch_chunk = آغاز_chunk;
    صحیح spad_idx = 0;

    برائے (صحیح k = 0; k < 2 && prefetch_chunk < انتہا_chunk; k++) {
        ثابت uint32_t c_آغاز   = ctx->elem_آغاز + prefetch_chunk * chunk_سائز;
        ثابت uint32_t c_انتہا     = MIN(c_آغاز + chunk_سائز, ctx->elem_آغاز + ctx->nelem);
        ثابت uint32_t cur_elems = c_انتہا - c_آغاز;
        ثابت uint32_t cur_بائٹیں = cur_elems * elem_سائز;

        dma_addr_t s0_curr = src0->ڈیٹا + (size_t) c_آغاز * elem_سائز;
        dma_addr_t d_curr  = dst->ڈیٹا  + (size_t) c_آغاز * elem_سائز;

        uint8_t * s0_spad = src0_spad_بنیاد + spad_idx * chunk_بائٹیں;
        uint8_t * d_spad  = dst_spad_بنیاد  + spad_idx * chunk_بائٹیں;

        dma_قطار_ڈالو(dma_q, dma_make_ڈیٹا(d_curr, d_spad), chunk_بائٹیں, chunk_بائٹیں, cur_بائٹیں, 0);
        dma_قطار_ڈالو(dma_q, dma_make_ڈیٹا(s0_spad, s0_curr), chunk_بائٹیں, chunk_بائٹیں, cur_بائٹیں, 1);
        اگر (!ctx->is_scalar) {
            dma_addr_t s1_curr = src1->ڈیٹا + (size_t) c_آغاز * elem_سائز;
            uint8_t * s1_spad = src1_spad_بنیاد + spad_idx * chunk_بائٹیں;
            dma_قطار_ڈالو(dma_q, dma_make_ڈیٹا(s1_spad, s1_curr), chunk_بائٹیں, chunk_بائٹیں, cur_بائٹیں, 1);
        }

        prefetch_chunk++;
        spad_idx ^= 1;
    }

    ساخت htp_thread_trace * tr = &octx->ctx->trace[ith];

    برائے (uint32_t c = آغاز_chunk; c < انتہا_chunk; c++) {
        ثابت uint32_t c_آغاز   = ctx->elem_آغاز + c * chunk_سائز;
        ثابت uint32_t c_انتہا     = MIN(c_آغاز + chunk_سائز, ctx->elem_آغاز + ctx->nelem);
        ثابت uint32_t cur_elems = c_انتہا - c_آغاز;
        ثابت uint32_t cur_بائٹیں = cur_elems * elem_سائز;

        uint8_t * d_spad  = (uint8_t *) dma_قطار_نکالو(dma_q).src;
        uint8_t * s0_spad = (uint8_t *) dma_قطار_نکالو(dma_q).dst;
        uint8_t * s1_spad = ctx->is_scalar ? تہی : (uint8_t *) dma_قطار_نکالو(dma_q).dst;

        htp_trace_event_آغاز(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) c);
        اگر (ctx->is_scalar) {
            اگر (src0_type == HTP_TYPE_F32) {
                ctx->compute_scalar_f32(d_spad, s0_spad, ctx->scalar_f32, cur_elems);
            } ورنہ {
                ctx->compute_scalar_f16(d_spad, s0_spad, ctx->scalar_f16, cur_elems);
            }
        } ورنہ {
            ctx->compute(d_spad, s0_spad, s1_spad, cur_elems);
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) c);

        dma_addr_t dst_curr = dst->ڈیٹا + (size_t) c_آغاز * elem_سائز;
        dma_قطار_ڈالو(dma_q, dma_make_ڈیٹا(dst_curr, d_spad), chunk_بائٹیں, chunk_بائٹیں, cur_بائٹیں, 1);

        اگر (prefetch_chunk < انتہا_chunk) {
            ثابت uint32_t pc_آغاز  = ctx->elem_آغاز + prefetch_chunk * chunk_سائز;
            ثابت uint32_t pc_انتہا    = MIN(pc_آغاز + chunk_سائز, ctx->elem_آغاز + ctx->nelem);
            ثابت uint32_t p_elems   = pc_انتہا - pc_آغاز;
            ثابت uint32_t p_بائٹیں   = p_elems * elem_سائز;

            dma_addr_t s0_اگلا = src0->ڈیٹا + (size_t) pc_آغاز * elem_سائز;
            dma_قطار_ڈالو(dma_q, dma_make_ڈیٹا(s0_spad, s0_اگلا), chunk_بائٹیں, chunk_بائٹیں, p_بائٹیں, 1);
            اگر (!ctx->is_scalar) {
                dma_addr_t s1_اگلا = src1->ڈیٹا + (size_t) pc_آغاز * elem_سائز;
                dma_قطار_ڈالو(dma_q, dma_make_ڈیٹا(s1_spad, s1_اگلا), chunk_بائٹیں, chunk_بائٹیں, p_بائٹیں, 1);
            }

            prefetch_chunk++;
        }
    }

    dma_قطار_flush(dma_q);
}

ساکن صحیح execute_op_ثنائی(ساخت htp_ops_context * octx) {
    ثابت ساخت htp_tensor * src0 = octx->src[0];
    ثابت ساخت htp_tensor * src1 = octx->src[1];
    ثابت ساخت htp_tensor * dst  = octx->dst;
    ثابت ساخت htp_ثنائی_kernel_params * kparams = (ثابت ساخت htp_ثنائی_kernel_params *) octx->kernel_params;

    ثابت uint32_t src0_nrows = src0->ne[1] * src0->ne[2] * src0->ne[3];

    // Use packed row sizes for VTCM allocation and alignment
    ثابت uint32_t src0_type = octx->src[0]->type;
    ثابت size_t elem_سائز = (src0_type == HTP_TYPE_F32) ? حجم(اعشاری) : حجم(_Float16);
    ثابت size_t src0_صف_سائز = src0->ne[0] * elem_سائز;
    ثابت size_t src1_صف_سائز = src1->ne[0] * elem_سائز;
    ثابت size_t dst_صف_سائز  = dst->ne[0]  * elem_سائز;

    اگر (kparams->kernel_type == HTP_ثنائی_KERNEL_CHUNKED) {
        ثابت bool is_scalar = kparams->is_scalar || (src1->ne[0] == 1 && src1->ne[1] == 1 && src1->ne[2] == 1 && src1->ne[3] == 1);

        compute_ثنائی_chunked_t compute = تہی;
        compute_ثنائی_scalar_chunked_f32_t compute_scalar_f32 = تہی;
        compute_ثنائی_scalar_chunked_f16_t compute_scalar_f16 = تہی;

        اگر (is_scalar) {
            اگر (src0_type == HTP_TYPE_F32) {
                انتخاب (octx->op) {
                    حالت HTP_OP_جوڑو: compute_scalar_f32 = hvx_جوڑو_scalar_f32; توڑو;
                    حالت HTP_OP_SUB: compute_scalar_f32 = hvx_sub_scalar_f32; توڑو;
                    حالت HTP_OP_MUL: compute_scalar_f32 = hvx_mul_scalar_f32; توڑو;
                    حالت HTP_OP_DIV: compute_scalar_f32 = hvx_div_scalar_f32; توڑو;
                    طےشدہ: توڑو;
                }
            } ورنہ اگر (src0_type == HTP_TYPE_F16) {
                انتخاب (octx->op) {
                    حالت HTP_OP_جوڑو: compute_scalar_f16 = hvx_جوڑو_scalar_f16; توڑو;
                    حالت HTP_OP_SUB: compute_scalar_f16 = hvx_sub_scalar_f16; توڑو;
                    حالت HTP_OP_MUL: compute_scalar_f16 = hvx_mul_scalar_f16; توڑو;
                    حالت HTP_OP_DIV: compute_scalar_f16 = hvx_div_scalar_f16; توڑو;
                    طےشدہ: توڑو;
                }
            }
            اگر (!compute_scalar_f32 && !compute_scalar_f16) {
                واپس HTP_حالت_NO_SUPPORT;
            }
        } ورنہ {
            اگر (src0_type == HTP_TYPE_F32) {
                انتخاب (octx->op) {
                    حالت HTP_OP_جوڑو: compute = hvx_جوڑو_f32; توڑو;
                    حالت HTP_OP_SUB: compute = hvx_sub_f32; توڑو;
                    حالت HTP_OP_MUL: compute = hvx_mul_f32; توڑو;
                    حالت HTP_OP_DIV: compute = hvx_div_f32; توڑو;
                    طےشدہ: توڑو;
                }
            } ورنہ اگر (src0_type == HTP_TYPE_F16) {
                انتخاب (octx->op) {
                    حالت HTP_OP_جوڑو: compute = hvx_جوڑو_f16; توڑو;
                    حالت HTP_OP_SUB: compute = hvx_sub_f16; توڑو;
                    حالت HTP_OP_MUL: compute = hvx_mul_f16; توڑو;
                    حالت HTP_OP_DIV: compute = hvx_div_f16; توڑو;
                    طےشدہ: توڑو;
                }
            }
            اگر (!compute) {
                واپس HTP_حالت_NO_SUPPORT;
            }
        }

        ثابت uint32_t کل_elems = (uint32_t) (src0->ne[0] * src0->ne[1] * src0->ne[2] * src0->ne[3]);
        اگر (کل_elems == 0) {
            واپس HTP_حالت_OK;
        }

        uint32_t elem_آغاز = 0;
        uint32_t nelem      = کل_elems;
        ثابت uint32_t elems_فی_سطر = (src0_type == HTP_TYPE_F32) ? 32 : 64;

        اگر (octx->ctx->mdev.گنتی > 1) {
            ثابت bool can_بانٹو = htp_tensor_mdev_ڈیٹا_aligned(dst) &&
                                   htp_tensor_mdev_ڈیٹا_aligned(src0) &&
                                   (is_scalar || htp_tensor_mdev_ڈیٹا_aligned(src1)) &&
                                   htp_tensor_is_contiguous(dst, elem_سائز) &&
                                   htp_tensor_is_contiguous(src0, elem_سائز) &&
                                   (is_scalar || htp_tensor_is_contiguous(src1, elem_سائز));
            ثابت ساخت htp_tensor_mdev_range range = htp_tensor_mdev_تقسیم(
                کل_elems, can_بانٹو ? elems_فی_سطر : 0,
                octx->ctx->mdev.idx, octx->ctx->mdev.گنتی, &octx->ctx->mdev.گنتی_div);
            elem_آغاز = range.آغاز;
            nelem      = range.گنتی;
        }

        اگر (nelem == 0) {
            واپس HTP_حالت_OK;
        }

        اگر (!htp_ops_context_سیٹ_n_دھاگے(octx, kparams->n_دھاگے)) {
            واپس HTP_حالت_INVAL_PARAMS;
        }

        ساخت htp_ثنائی_vtcm_layout vtcm_layout;
        htp_ثنائی_vtcm_layout_build(&vtcm_layout, kparams, octx->ctx->vtcm_سائز);
        اگر (vtcm_layout.کل_بائٹیں == 0 || vtcm_layout.کل_بائٹیں > octx->ctx->vtcm_سائز) {
            واپس HTP_حالت_VTCM_TOO_SMALL;
        }

        ثابت uint32_t chunk_سائز = kparams->chunk_سائز > 0 ? kparams->chunk_سائز : (32768 / elem_سائز);
        ثابت uint32_t کل_chunks = (nelem + chunk_سائز - 1) / chunk_سائز;
        ثابت uint32_t n_دھاگے = (کل_chunks >= 2) ? MIN(octx->n_دھاگے, کل_chunks) : 1;
        ثابت uint32_t chunks_فی_thread = (کل_chunks + n_دھاگے - 1) / n_دھاگے;

        اعشاری scalar_f32 = 0.0f;
        _Float16 scalar_f16 = 0;
        اگر (is_scalar) {
            uint8_t * vtcm_src1 = VTCM_LAYOUT_PTR(uint8_t, octx->ctx->vtcm_بنیاد, vtcm_layout.off_src1);
            dma_قطار * dma_q = octx->ctx->dma[0];
            dma_قطار_ڈالو(dma_q, dma_make_ڈیٹا(vtcm_src1, src1->ڈیٹا), 128, 0, elem_سائز, 1);
            dma_قطار_نکالو(dma_q);

            اگر (src0_type == HTP_TYPE_F32) {
                scalar_f32 = ((ثابت اعشاری *) vtcm_src1)[0];
            } ورنہ {
                scalar_f16 = ((ثابت _Float16 *) vtcm_src1)[0];
            }
        }

        ساخت ثنائی_chunked_context cctx = {
            .octx                = octx,
            .vtcm_layout         = vtcm_layout,
            .vtcm_بنیاد           = (uint8_t *) octx->ctx->vtcm_بنیاد,
            .elem_آغاز          = elem_آغاز,
            .nelem               = nelem,
            .chunk_سائز          = chunk_سائز,
            .chunks_فی_thread   = chunks_فی_thread,
            .کل_chunks        = کل_chunks,
            .is_scalar           = is_scalar,
            .scalar_f32          = scalar_f32,
            .scalar_f16          = scalar_f16,
            .compute             = compute,
            .compute_scalar_f32  = compute_scalar_f32,
            .compute_scalar_f16  = compute_scalar_f16,
        };

        work_قطار_تسلسل(octx->ctx->work_قطار, ثنائی_thread_chunked, &cctx, n_دھاگے);
        واپس HTP_حالت_OK;
    }

    uint32_t صف_آغاز = 0;
    uint32_t nrows     = src0_nrows;

    اگر (octx->ctx->mdev.گنتی > 1) {
        uint32_t صفیں_فی_chunk = 0;
        htp_tensor_mdev_صفیں_فی_chunk(dst, (uint32_t) elem_سائز, (uint32_t) dst_صف_سائز, &صفیں_فی_chunk);
        ثابت ساخت htp_tensor_mdev_range range = htp_tensor_mdev_تقسیم(src0_nrows, صفیں_فی_chunk,
                                                       octx->ctx->mdev.idx, octx->ctx->mdev.گنتی, &octx->ctx->mdev.گنتی_div);
        صف_آغاز = range.آغاز;
        nrows     = range.گنتی;
    }

    اگر (nrows == 0) {
        واپس HTP_حالت_OK;
    }

    اگر (!htp_ops_context_سیٹ_n_دھاگے(octx, kparams->n_دھاگے)) {
        واپس HTP_حالت_INVAL_PARAMS;
    }

    ثابت uint32_t n_دھاگے = octx->n_دھاگے;
    ثابت size_t src0_صف_سائز_aligned = kparams->src0_صف_سائز_aligned;
    ثابت size_t src1_صف_سائز_aligned = kparams->src1_صف_سائز_aligned;
    ثابت size_t dst_صف_سائز_aligned  = kparams->dst_صف_سائز_aligned;

    ساخت htp_ثنائی_context bctx;
    bctx.vtcm_بنیاد = (uint8_t *) octx->ctx->vtcm_بنیاد;
    htp_ثنائی_vtcm_layout_build(&bctx.vtcm_layout, kparams, octx->ctx->vtcm_سائز);

    اگر (bctx.vtcm_layout.صفیں_فی_بفر == 0 || bctx.vtcm_layout.کل_بائٹیں > octx->ctx->vtcm_سائز) {
        واپس HTP_حالت_VTCM_TOO_SMALL;
    }

    dma_قطار * dma_q = octx->ctx->dma[0];
    uint8_t * vtcm_src1 = VTCM_LAYOUT_PTR(uint8_t, bctx.vtcm_بنیاد, bctx.vtcm_layout.off_src1);
    اگر (kparams->kernel_type == HTP_ثنائی_KERNEL_صف_BCAST) {
        dma_قطار_ڈالو(dma_q, dma_make_ڈیٹا(vtcm_src1, src1->ڈیٹا), bctx.vtcm_layout.src1_سائز, 0, src1->ne[0] * elem_سائز, 1);
    } ورنہ اگر (kparams->kernel_type == HTP_ثنائی_KERNEL_SCALAR ||
               kparams->kernel_type == HTP_ثنائی_KERNEL_REPEAT ||
               kparams->kernel_type == HTP_ثنائی_KERNEL_COMPLEX) {
        اگر (!kparams->src1_strided) {
            ثابت size_t src1_بائٹیں = (size_t) src1->ne[0] * src1->ne[1] * src1->ne[2] * src1->ne[3] * elem_سائز;
            dma_قطار_ڈالو(dma_q, dma_make_ڈیٹا(vtcm_src1, src1->ڈیٹا), bctx.vtcm_layout.src1_سائز, 0, src1_بائٹیں, 1);
        } ورنہ {
            ثابت uint32_t ne11 = src1->ne[1];
            ثابت uint32_t ne12 = src1->ne[2];
            ثابت uint32_t ne13 = src1->ne[3];
            ثابت size_t صف_بائٹیں = src1->ne[0] * elem_سائز;
            برائے (uint32_t i13 = 0; i13 < ne13; i13++) {
                برائے (uint32_t i12 = 0; i12 < ne12; i12++) {
                    dma_addr_t src_addr = src1->ڈیٹا + i13 * src1->nb[3] + i12 * src1->nb[2];
                    uint8_t * dst_ptr = vtcm_src1 + (i13 * ne12 * ne11 + i12 * ne11) * صف_بائٹیں;
                    جب (!dma_قطار_ڈالو(dma_q, dma_make_ڈیٹا(dst_ptr, src_addr),
                                           صف_بائٹیں, src1->nb[1], صف_بائٹیں, ne11)) {
                        dma_قطار_نکالو(dma_q);
                    }
                }
            }
        }
    } ورنہ اگر (kparams->kernel_type == HTP_ثنائی_KERNEL_جوڑو_ID) {
        ثابت ساخت htp_tensor * src2 = octx->src[2];
        uint8_t * vtcm_src2 = VTCM_LAYOUT_PTR(uint8_t, bctx.vtcm_بنیاد, bctx.vtcm_layout.off_src2);
        dma_قطار_ڈالو(dma_q, dma_make_ڈیٹا(vtcm_src1, src1->ڈیٹا),
                       kparams->src1_صف_سائز_aligned, src1->nb[1],
                       src1->ne[0] * elem_سائز, src1->ne[1]);
        ثابت size_t src2_صف_بائٹیں = src2->ne[0] * حجم(int32_t);
        dma_قطار_ڈالو(dma_q, dma_make_ڈیٹا(vtcm_src2, src2->ڈیٹا),
                       src2_صف_بائٹیں, src2->nb[1],
                       src2_صف_بائٹیں, src2->ne[1]);
    }

    bctx.octx                  = octx;
    bctx.nrows_فی_thread      = fastdiv(nrows + n_دھاگے - 1, &octx->n_دھاگے_div);
    bctx.کل_صفیں            = nrows;
    bctx.صف_آغاز             = صف_آغاز;
    bctx.بلاک_max             = bctx.vtcm_layout.صفیں_فی_بفر;
    bctx.src0_صف_سائز_aligned = src0_صف_سائز_aligned;
    bctx.src1_صف_سائز_aligned = src1_صف_سائز_aligned;
    bctx.dst_صف_سائز_aligned  = dst_صف_سائز_aligned;
    bctx.صف_سائز_بائٹیں        = src0_صف_سائز;

    bctx.src0_dim1_div  = init_fastdiv_اقدار(src0->ne[1]);
    bctx.src0_dim2_div  = init_fastdiv_اقدار(src0->ne[2]);
    bctx.src0_dim12_div = init_fastdiv_اقدار(src0->ne[1] * src0->ne[2]);

    bctx.src1_dim1_div  = init_fastdiv_اقدار(src1->ne[1]);
    bctx.src1_dim2_div  = init_fastdiv_اقدار(src1->ne[2]);
    bctx.src1_dim3_div  = init_fastdiv_اقدار(src1->ne[3]);

    bctx.بانٹو_at_ne01  = kparams->بانٹو_at_ne01;
    bctx.بانٹو_at_ne02  = kparams->بانٹو_at_ne02;

    کارکن_callback_t کارکن_func = تہی;
    خالی * compute_func = تہی;

    انتخاب (kparams->kernel_type) {
        حالت HTP_ثنائی_KERNEL_SAME_SHAPE:
            کارکن_func = ثنائی_thread_سمتیہ_same_shape;
            اگر (src0_type == HTP_TYPE_F32) {
                انتخاب (octx->op) {
                    حالت HTP_OP_جوڑو: compute_func = compute_same_shape_جوڑو_f32; توڑو;
                    حالت HTP_OP_SUB: compute_func = compute_same_shape_sub_f32; توڑو;
                    حالت HTP_OP_MUL: compute_func = compute_same_shape_mul_f32; توڑو;
                    حالت HTP_OP_DIV: compute_func = compute_same_shape_div_f32; توڑو;
                    طےشدہ: توڑو;
                }
            } ورنہ اگر (src0_type == HTP_TYPE_F16) {
                انتخاب (octx->op) {
                    حالت HTP_OP_جوڑو: compute_func = compute_same_shape_جوڑو_f16; توڑو;
                    حالت HTP_OP_SUB: compute_func = compute_same_shape_sub_f16; توڑو;
                    حالت HTP_OP_MUL: compute_func = compute_same_shape_mul_f16; توڑو;
                    حالت HTP_OP_DIV: compute_func = compute_same_shape_div_f16; توڑو;
                    طےشدہ: توڑو;
                }
            }
            توڑو;
        حالت HTP_ثنائی_KERNEL_صف_BCAST:
            کارکن_func = ثنائی_thread_سمتیہ_صف_نشر;
            اگر (src0_type == HTP_TYPE_F32) {
                انتخاب (octx->op) {
                    حالت HTP_OP_جوڑو: compute_func = compute_صف_bcast_جوڑو_f32; توڑو;
                    حالت HTP_OP_SUB: compute_func = compute_صف_bcast_sub_f32; توڑو;
                    حالت HTP_OP_MUL: compute_func = compute_صف_bcast_mul_f32; توڑو;
                    حالت HTP_OP_DIV: compute_func = compute_صف_bcast_div_f32; توڑو;
                    طےشدہ: توڑو;
                }
            } ورنہ اگر (src0_type == HTP_TYPE_F16) {
                انتخاب (octx->op) {
                    حالت HTP_OP_جوڑو: compute_func = compute_صف_bcast_جوڑو_f16; توڑو;
                    حالت HTP_OP_SUB: compute_func = compute_صف_bcast_sub_f16; توڑو;
                    حالت HTP_OP_MUL: compute_func = compute_صف_bcast_mul_f16; توڑو;
                    حالت HTP_OP_DIV: compute_func = compute_صف_bcast_div_f16; توڑو;
                    طےشدہ: توڑو;
                }
            }
            توڑو;
        حالت HTP_ثنائی_KERNEL_SCALAR:
            کارکن_func = ثنائی_thread_scalar;
            اگر (src0_type == HTP_TYPE_F32) {
                انتخاب (octx->op) {
                    حالت HTP_OP_جوڑو: compute_func = compute_scalar_جوڑو_f32; توڑو;
                    حالت HTP_OP_SUB: compute_func = compute_scalar_sub_f32; توڑو;
                    حالت HTP_OP_MUL: compute_func = compute_scalar_mul_f32; توڑو;
                    حالت HTP_OP_DIV: compute_func = compute_scalar_div_f32; توڑو;
                    طےشدہ: توڑو;
                }
            } ورنہ اگر (src0_type == HTP_TYPE_F16) {
                انتخاب (octx->op) {
                    حالت HTP_OP_جوڑو: compute_func = compute_scalar_جوڑو_f16; توڑو;
                    حالت HTP_OP_SUB: compute_func = compute_scalar_sub_f16; توڑو;
                    حالت HTP_OP_MUL: compute_func = compute_scalar_mul_f16; توڑو;
                    حالت HTP_OP_DIV: compute_func = compute_scalar_div_f16; توڑو;
                    طےشدہ: توڑو;
                }
            }
            توڑو;
        حالت HTP_ثنائی_KERNEL_COMPLEX:
            کارکن_func = ثنائی_thread_سمتیہ_complex;
            اگر (src0_type == HTP_TYPE_F32) {
                انتخاب (octx->op) {
                    حالت HTP_OP_جوڑو: compute_func = compute_complex_جوڑو_f32; توڑو;
                    حالت HTP_OP_SUB: compute_func = compute_complex_sub_f32; توڑو;
                    حالت HTP_OP_MUL: compute_func = compute_complex_mul_f32; توڑو;
                    حالت HTP_OP_DIV: compute_func = compute_complex_div_f32; توڑو;
                    طےشدہ: توڑو;
                }
            } ورنہ اگر (src0_type == HTP_TYPE_F16) {
                انتخاب (octx->op) {
                    حالت HTP_OP_جوڑو: compute_func = compute_complex_جوڑو_f16; توڑو;
                    حالت HTP_OP_SUB: compute_func = compute_complex_sub_f16; توڑو;
                    حالت HTP_OP_MUL: compute_func = compute_complex_mul_f16; توڑو;
                    حالت HTP_OP_DIV: compute_func = compute_complex_div_f16; توڑو;
                    طےشدہ: توڑو;
                }
            }
            توڑو;
        حالت HTP_ثنائی_KERNEL_REPEAT:
            کارکن_func = ثنائی_thread_element_repeat;
            اگر (src0_type == HTP_TYPE_F32) {
                انتخاب (octx->op) {
                    حالت HTP_OP_جوڑو: compute_func = compute_repeat_جوڑو_f32; توڑو;
                    حالت HTP_OP_SUB: compute_func = compute_repeat_sub_f32; توڑو;
                    حالت HTP_OP_MUL: compute_func = compute_repeat_mul_f32; توڑو;
                    حالت HTP_OP_DIV: compute_func = compute_repeat_div_f32; توڑو;
                    طےشدہ: توڑو;
                }
            } ورنہ اگر (src0_type == HTP_TYPE_F16) {
                انتخاب (octx->op) {
                    حالت HTP_OP_جوڑو: compute_func = compute_repeat_جوڑو_f16; توڑو;
                    حالت HTP_OP_SUB: compute_func = compute_repeat_sub_f16; توڑو;
                    حالت HTP_OP_MUL: compute_func = compute_repeat_mul_f16; توڑو;
                    حالت HTP_OP_DIV: compute_func = compute_repeat_div_f16; توڑو;
                    طےشدہ: توڑو;
                }
            }
            توڑو;
        حالت HTP_ثنائی_KERNEL_جوڑو_ID:
            اگر (octx->op == HTP_OP_جوڑو_ID && src0_type == HTP_TYPE_F32) {
                کارکن_func = ثنائی_thread_جوڑو_id_f32;
                compute_func = (خالی *) compute_جوڑو_id_f32;
            }
            توڑو;
        طےشدہ: توڑو;
    }

    اگر (!کارکن_func || !compute_func) {
        واپس HTP_حالت_NO_SUPPORT;
    }

    bctx.compute = compute_func;

    اگر (kparams->kernel_type == HTP_ثنائی_KERNEL_صف_BCAST ||
        kparams->kernel_type == HTP_ثنائی_KERNEL_SCALAR ||
        kparams->kernel_type == HTP_ثنائی_KERNEL_REPEAT ||
        kparams->kernel_type == HTP_ثنائی_KERNEL_COMPLEX ||
        kparams->kernel_type == HTP_ثنائی_KERNEL_جوڑو_ID) {
        dma_قطار_flush(dma_q);
    }

    work_قطار_تسلسل(octx->ctx->work_قطار, کارکن_func, &bctx, n_دھاگے);

    واپس HTP_حالت_OK;
}

صحیح op_ثنائی(ساخت htp_ops_context * octx) {
    ثابت uint32_t src0_type = octx->src[0]->type;
    اگر ((src0_type == HTP_TYPE_F32) || (src0_type == HTP_TYPE_F16)) {
        واپس execute_op_ثنائی(octx);
    }

    واپس HTP_حالت_NO_SUPPORT;
}
