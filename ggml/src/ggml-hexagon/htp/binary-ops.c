#निर्देशः clang diagnostic ignored "-Wunused-variable"
#निर्देशः clang diagnostic ignored "-Wunused-function"
#निर्देशः clang diagnostic ignored "-Wunused-but-set-variable"

#अन्तर्भाव <HAP_farf.h>
#अन्तर्भाव <HAP_perf.h>

#अन्तर्भाव <math.h>
#अन्तर्भाव <string.h>

#अन्तर्भाव "dma-queue.h"
#अन्तर्भाव "hvx-utils.h"

#परिभाषय GGML_COMMON_DECL_C
#अन्तर्भाव "ggml-common.h"
#अन्तर्भाव "hex-common.h"
#अन्तर्भाव "hex-profile.h"
#अन्तर्भाव "binary-ops.h"
#अन्तर्भाव "htp-ctx.h"
#अन्तर्भाव "htp-ops.h"
#अन्तर्भाव "htp-tensor.h"

#यदि_अपरिभाषितम् MIN
#परिभाषय MIN(a, b) ((a) < (b) ? (a) : (b))
#यदि_अन्तः

// Context for binary operations
संरचना htp_द्विमान_context {
    संरचना htp_ops_context * octx;
    संरचना htp_द्विमान_vtcm_layout vtcm_layout;
    uint8_t * vtcm_आधारः;

    संरचना fastdiv_मानानि src0_dim1_div; // ne01
    संरचना fastdiv_मानानि src0_dim2_div; // ne02
    संरचना fastdiv_मानानि src0_dim12_div;// ne03

    संरचना fastdiv_मानानि src1_dim1_div; // ne11
    संरचना fastdiv_मानानि src1_dim2_div; // ne12
    संरचना fastdiv_मानानि src1_dim3_div; // ne13

    uint32_t खण्डम्_max;
    uint32_t nrows_प्रत्येकम्_thread;
    uint32_t योग_पङ्क्तिकाः;
    uint32_t row_start;
    size_t   src0_row_आकार_aligned;
    size_t   src1_row_आकार_aligned;
    size_t   dst_row_आकार_aligned;
    size_t   row_आकार_अष्टकानि;

    bool विभज_at_ne01;
    bool विभज_at_ne02;

    शून्य * compute;
};

#परिभाषय htp_द्विमान_preamble                        \
    ध्रुव संरचना htp_tensor * src0 = octx->src[0]; \
    ध्रुव संरचना htp_tensor * src1 = octx->src[1]; \
    ध्रुव संरचना htp_tensor * dst  = octx->dst;    \
                                                   \
    ध्रुव uint32_t ne00 = src0->ne[0];             \
    ध्रुव uint32_t ne01 = src0->ne[1];             \
    ध्रुव uint32_t ne02 = src0->ne[2];             \
    ध्रुव uint32_t ne03 = src0->ne[3];             \
                                                   \
    ध्रुव uint32_t ne10 = src1->ne[0];             \
    ध्रुव uint32_t ne11 = src1->ne[1];             \
    ध्रुव uint32_t ne12 = src1->ne[2];             \
    ध्रुव uint32_t ne13 = src1->ne[3];             \
                                                   \
    ध्रुव uint32_t nb01 = src0->nb[1];             \
    ध्रुव uint32_t nb02 = src0->nb[2];             \
    ध्रुव uint32_t nb03 = src0->nb[3];             \
                                                   \
    ध्रुव uint32_t nb11 = src1->nb[1];             \
    ध्रुव uint32_t nb12 = src1->nb[2];             \
    ध्रुव uint32_t nb13 = src1->nb[3];             \
                                                   \
    ध्रुव uint32_t nb1 = dst->nb[1];               \
    ध्रुव uint32_t nb2 = dst->nb[2];               \
    ध्रुव uint32_t nb3 = dst->nb[3];

स्थिर अन्तःस्थ uint32_t calc_खण्डम्_आकार(संरचना htp_द्विमान_context * bctx, uint32_t ir, uint32_t अन्तः_row, uint32_t ne01, uint32_t ne02) {
    uint32_t i03, i02, i01, rem;
    i03 = fastdiv(ir, &bctx->src0_dim12_div);
    rem = ir - i03 * (ne02 * ne01);
    i02 = fastdiv(rem, &bctx->src0_dim1_div);
    i01 = rem - i02 * ne01;

    uint32_t पङ्क्तिकाः_वामः = अन्तः_row - ir;
    uint32_t खण्डम्_सीमा = पङ्क्तिकाः_वामः;

    यदि (bctx->विभज_at_ne01) {
        खण्डम्_सीमा = MIN(खण्डम्_सीमा, ne01 - i01);
    }
    यदि (bctx->विभज_at_ne02) {
         uint32_t पङ्क्तिकाः_अन्तर्_plane = (ne02 * ne01) - rem;
         खण्डम्_सीमा = MIN(खण्डम्_सीमा, पङ्क्तिकाः_अन्तर्_plane);
    }

    प्रत्यागम MIN(bctx->खण्डम्_max, खण्डम्_सीमा);
}

// Out-of-line compute micro-kernels

प्रकारनिर्देश शून्य (*compute_scalar_t)(
    uint8_t * dst, ध्रुव uint8_t * src0, ध्रुव शून्य * s1_सारणी,
    uint32_t cur_i11, uint32_t ne11, uint32_t n_पङ्क्तिकाः,
    size_t dst_stride, size_t src0_stride, uint32_t ne00);

#परिभाषय DEFINE_COMPUTE_SCALAR(NAME, TYPE, HVX_STMT)             \
स्थिर शून्य compute_scalar_##NAME(                              \
    uint8_t * dst, ध्रुव uint8_t * src0, ध्रुव शून्य * s1_सारणी, \
    uint32_t cur_i11, uint32_t ne11, uint32_t n_पङ्क्तिकाः,           \
    size_t dst_stride, size_t src0_stride, uint32_t ne00) {     \
    ध्रुव TYPE * सारणी = (ध्रुव TYPE *) s1_सारणी;               \
    प्रति (uint32_t r = 0; r < n_पङ्क्तिकाः; r++) {                     \
        uint8_t * r_dst = dst + r * dst_stride;                 \
        ध्रुव uint8_t * r_src0 = src0 + r * src0_stride;        \
        TYPE val = सारणी[cur_i11];                              \
        HVX_STMT;                                               \
        यदि (ne11 > 1 && ++cur_i11 == ne11) {                    \
            cur_i11 = 0;                                        \
        }                                                       \
    }                                                           \
}

DEFINE_COMPUTE_SCALAR(सङ्कलय_f32, भिन्न,    hvx_सङ्कलय_scalar_f32_aa(r_dst, r_src0, val, ne00))
DEFINE_COMPUTE_SCALAR(सङ्कलय_f16, _Float16, hvx_सङ्कलय_scalar_f16_aa(r_dst, r_src0, val, ne00))
DEFINE_COMPUTE_SCALAR(sub_f32, भिन्न,    hvx_sub_scalar_f32_aa(r_dst, r_src0, val, ne00))
DEFINE_COMPUTE_SCALAR(sub_f16, _Float16, hvx_sub_scalar_f16_aa(r_dst, r_src0, val, ne00))
DEFINE_COMPUTE_SCALAR(mul_f32, भिन्न,    hvx_mul_scalar_f32_aa(r_dst, r_src0, val, ne00))
DEFINE_COMPUTE_SCALAR(mul_f16, _Float16, hvx_mul_scalar_f16_aa(r_dst, r_src0, val, ne00))
DEFINE_COMPUTE_SCALAR(div_f32, भिन्न,    hvx_mul_scalar_f32_aa(r_dst, r_src0, 1.0f / (val), ne00))
DEFINE_COMPUTE_SCALAR(div_f16, _Float16, hvx_div_scalar_f16_aa(r_dst, r_src0, val, ne00))

प्रकारनिर्देश शून्य (*compute_same_shape_t)(
    uint8_t * dst, ध्रुव uint8_t * src0, ध्रुव uint8_t * src1,
    uint32_t n_पङ्क्तिकाः, size_t dst_stride, size_t src0_stride, size_t src1_stride, uint32_t ne00);

#परिभाषय DEFINE_COMPUTE_SAME_SHAPE(NAME, HVX_FN)                                                  \
स्थिर शून्य compute_same_shape_##NAME(                                                           \
    uint8_t * dst, ध्रुव uint8_t * src0, ध्रुव uint8_t * src1,                                   \
    uint32_t n_पङ्क्तिकाः, size_t dst_stride, size_t src0_stride, size_t src1_stride, uint32_t ne00) { \
    प्रति (uint32_t r = 0; r < n_पङ्क्तिकाः; r++) {                                                      \
        HVX_FN(dst + r * dst_stride, src0 + r * src0_stride, src1 + r * src1_stride, ne00);      \
    }                                                                                            \
}

DEFINE_COMPUTE_SAME_SHAPE(सङ्कलय_f32, hvx_सङ्कलय_f32_aaa)
DEFINE_COMPUTE_SAME_SHAPE(सङ्कलय_f16, hvx_सङ्कलय_f16_aaa)
DEFINE_COMPUTE_SAME_SHAPE(sub_f32, hvx_sub_f32_aaa)
DEFINE_COMPUTE_SAME_SHAPE(sub_f16, hvx_sub_f16_aaa)
DEFINE_COMPUTE_SAME_SHAPE(mul_f32, hvx_mul_f32_aaa)
DEFINE_COMPUTE_SAME_SHAPE(mul_f16, hvx_mul_f16_aaa)
DEFINE_COMPUTE_SAME_SHAPE(div_f32, hvx_div_f32_aaa)
DEFINE_COMPUTE_SAME_SHAPE(div_f16, hvx_div_f16_aaa)

प्रकारनिर्देश शून्य (*compute_row_bcast_t)(
    uint8_t * dst, ध्रुव uint8_t * src0, ध्रुव uint8_t * src1,
    uint32_t n_पङ्क्तिकाः, size_t dst_stride, size_t src0_stride, uint32_t ne00);

#परिभाषय DEFINE_COMPUTE_ROW_BCAST(NAME, HVX_FN)                                               \
स्थिर शून्य compute_row_bcast_##NAME(                                                        \
    uint8_t * dst, ध्रुव uint8_t * src0, ध्रुव uint8_t * src1,                               \
    uint32_t n_पङ्क्तिकाः, size_t dst_stride, size_t src0_stride, uint32_t ne00) {                 \
    प्रति (uint32_t r = 0; r < n_पङ्क्तिकाः; r++) {                                                  \
        HVX_FN(dst + r * dst_stride, src0 + r * src0_stride, src1, ne00);                    \
    }                                                                                        \
}

DEFINE_COMPUTE_ROW_BCAST(सङ्कलय_f32, hvx_सङ्कलय_f32_aaa)
DEFINE_COMPUTE_ROW_BCAST(सङ्कलय_f16, hvx_सङ्कलय_f16_aaa)
DEFINE_COMPUTE_ROW_BCAST(sub_f32, hvx_sub_f32_aaa)
DEFINE_COMPUTE_ROW_BCAST(sub_f16, hvx_sub_f16_aaa)
DEFINE_COMPUTE_ROW_BCAST(mul_f32, hvx_mul_f32_aaa)
DEFINE_COMPUTE_ROW_BCAST(mul_f16, hvx_mul_f16_aaa)
DEFINE_COMPUTE_ROW_BCAST(div_f32, hvx_div_f32_aaa)
DEFINE_COMPUTE_ROW_BCAST(div_f16, hvx_div_f16_aaa)

प्रकारनिर्देश शून्य (*compute_complex_t)(
    uint8_t * dst, ध्रुव uint8_t * src0, ध्रुव uint8_t * src1_plane,
    uint32_t i01, uint32_t ne11, ध्रुव संरचना fastdiv_मानानि * div11, uint32_t src1_stride,
    uint32_t n_पङ्क्तिकाः, size_t dst_stride, size_t src0_stride, uint32_t ne00);

#परिभाषय DEFINE_COMPUTE_COMPLEX(NAME, HVX_FN)                                                 \
स्थिर शून्य compute_complex_##NAME(                                                          \
    uint8_t * dst, ध्रुव uint8_t * src0, ध्रुव uint8_t * src1_plane,                         \
    uint32_t i01, uint32_t ne11, ध्रुव संरचना fastdiv_मानानि * div11, uint32_t src1_stride,  \
    uint32_t n_पङ्क्तिकाः, size_t dst_stride, size_t src0_stride, uint32_t ne00) {                 \
    प्रति (uint32_t r = 0; r < n_पङ्क्तिकाः; r++) {                                                  \
        uint32_t i11 = fastmodulo(i01 + r, ne11, div11);                                     \
        ध्रुव uint8_t * r_src1 = src1_plane + i11 * src1_stride;                             \
        HVX_FN(dst + r * dst_stride, src0 + r * src0_stride, r_src1, ne00);                  \
    }                                                                                        \
}

DEFINE_COMPUTE_COMPLEX(सङ्कलय_f32, hvx_सङ्कलय_f32_aau)
DEFINE_COMPUTE_COMPLEX(सङ्कलय_f16, hvx_सङ्कलय_f16_aau)
DEFINE_COMPUTE_COMPLEX(sub_f32, hvx_sub_f32_aau)
DEFINE_COMPUTE_COMPLEX(sub_f16, hvx_sub_f16_aau)
DEFINE_COMPUTE_COMPLEX(mul_f32, hvx_mul_f32_aau)
DEFINE_COMPUTE_COMPLEX(mul_f16, hvx_mul_f16_aau)
DEFINE_COMPUTE_COMPLEX(div_f32, hvx_div_f32_aau)
DEFINE_COMPUTE_COMPLEX(div_f16, hvx_div_f16_aau)

प्रकारनिर्देश शून्य (*compute_repeat_t)(
    uint8_t * dst, ध्रुव uint8_t * src0, ध्रुव uint8_t * src1_plane,
    uint32_t i01, uint32_t ne11, ध्रुव संरचना fastdiv_मानानि * div11, uint32_t src1_stride,
    uint32_t n_पङ्क्तिकाः, size_t dst_stride, size_t src0_stride, uint32_t ne00, uint32_t ne10);

#परिभाषय DEFINE_COMPUTE_REPEAT(NAME, TYPE, HVX_FN)                                            \
स्थिर शून्य compute_repeat_##NAME(                                                           \
    uint8_t * dst, ध्रुव uint8_t * src0, ध्रुव uint8_t * src1_plane,                         \
    uint32_t i01, uint32_t ne11, ध्रुव संरचना fastdiv_मानानि * div11, uint32_t src1_stride,  \
    uint32_t n_पङ्क्तिकाः, size_t dst_stride, size_t src0_stride, uint32_t ne00, uint32_t ne10) {  \
    प्रति (uint32_t r = 0; r < n_पङ्क्तिकाः; r++) {                                                  \
        uint32_t i11 = fastmodulo(i01 + r, ne11, div11);                                     \
        ध्रुव uint8_t * r_src1_row = src1_plane + i11 * src1_stride;                         \
        uint8_t * r_dst = dst + r * dst_stride;                                              \
        ध्रुव uint8_t * r_src0 = src0 + r * src0_stride;                                     \
        प्रति (uint32_t c = 0; c < ne00; c += ne10) {                                          \
            uint32_t len = MIN(ne10, ne00 - c);                                              \
            HVX_FN(r_dst + c * परिमाण(TYPE), r_src0 + c * परिमाण(TYPE), r_src1_row, len);    \
        }                                                                                    \
    }                                                                                        \
}

DEFINE_COMPUTE_REPEAT(सङ्कलय_f32, भिन्न,    hvx_सङ्कलय_f32_uuu)
DEFINE_COMPUTE_REPEAT(सङ्कलय_f16, _Float16, hvx_सङ्कलय_f16_uuu)
DEFINE_COMPUTE_REPEAT(sub_f32, भिन्न,    hvx_sub_f32_uuu)
DEFINE_COMPUTE_REPEAT(sub_f16, _Float16, hvx_sub_f16_uuu)
DEFINE_COMPUTE_REPEAT(mul_f32, भिन्न,    hvx_mul_f32_uuu)
DEFINE_COMPUTE_REPEAT(mul_f16, _Float16, hvx_mul_f16_uuu)
DEFINE_COMPUTE_REPEAT(div_f32, भिन्न,    hvx_div_f32_uuu)
DEFINE_COMPUTE_REPEAT(div_f16, _Float16, hvx_div_f16_uuu)

प्रकारनिर्देश शून्य (*compute_सङ्कलय_id_t)(
    uint8_t * dst, ध्रुव uint8_t * src0, ध्रुव uint8_t * src1_दत्त, ध्रुव int32_t * src2_ids,
    uint32_t i01, uint32_t i02, uint32_t ne01, uint32_t src1_stride,
    uint32_t n_पङ्क्तिकाः, size_t dst_stride, size_t src0_stride, uint32_t ne00);

स्थिर शून्य compute_सङ्कलय_id_f32(
    uint8_t * dst, ध्रुव uint8_t * src0, ध्रुव uint8_t * src1_दत्त, ध्रुव int32_t * src2_ids,
    uint32_t i01, uint32_t i02, uint32_t ne01, uint32_t src1_stride,
    uint32_t n_पङ्क्तिकाः, size_t dst_stride, size_t src0_stride, uint32_t ne00) {
    प्रति (uint32_t r = 0; r < n_पङ्क्तिकाः; r++) {
        uint32_t r_i01 = i01 + r;
        ध्रुव int32_t idx = src2_ids[i02 * ne01 + r_i01];
        यदि (idx < 0) {
            स्मृतिं_प्रतिलिख(dst + r * dst_stride, src0 + r * src0_stride, ne00 * परिमाण(भिन्न));
            अनुवर्त;
        }
        ध्रुव uint8_t * r_src1 = src1_दत्त + idx * src1_stride;
        ध्रुव uint8_t * r_src0 = src0 + r * src0_stride;
        uint8_t * r_dst = dst + r * dst_stride;
        hvx_सङ्कलय_f32_aaa(r_dst, r_src0, r_src1, ne00);
    }
}

// 1. Scalar src1 in VTCM via DMA (ne10 == 1)
स्थिर शून्य द्विमान_thread_scalar(अचिह्नित पूर्णाङ्क nth, अचिह्नित पूर्णाङ्क ith, शून्य * दत्त) {
    संरचना htp_द्विमान_context * bctx = (संरचना htp_द्विमान_context *) दत्त;
    संरचना htp_ops_context * octx = bctx->octx;
    htp_द्विमान_preamble;

    ध्रुव uint32_t elem_आकार = (src0->type == HTP_TYPE_F32) ? परिमाण(भिन्न) : परिमाण(_Float16);
    ध्रुव size_t plane_stride = (size_t) ne12 * ne11 * elem_आकार;
    ध्रुव size_t row_stride   = (size_t) ne11 * elem_आकार;

    ध्रुव uint32_t row_आकार_अष्टकानि = bctx->row_आकार_अष्टकानि;
    ध्रुव uint32_t start_row = bctx->row_start + bctx->nrows_प्रत्येकम्_thread * ith;
    ध्रुव uint32_t अन्तः_row   = MIN(start_row + bctx->nrows_प्रत्येकम्_thread, bctx->row_start + bctx->योग_पङ्क्तिकाः);
    यदि (start_row >= अन्तः_row) प्रत्यागम;

    FARF(उच्चः_2, "binary-scalar: %d/%d (%u:%u) row-size %u (%u)",
         ith, nth, start_row, अन्तः_row, nb01, bctx->dst_row_आकार_aligned);

    ध्रुव संरचना htp_द्विमान_vtcm_layout * layout = &bctx->vtcm_layout;
    uint8_t * src0_spad_आधारः = VTCM_LAYOUT_PTR(uint8_t, bctx->vtcm_आधारः, layout->off_src0) + (ith * layout->src0_अष्टकानि_प्रत्येकम्_thread);
    uint8_t * dst_spad_आधारः  = VTCM_LAYOUT_PTR(uint8_t, bctx->vtcm_आधारः, layout->off_dst)  + (ith * layout->dst_अष्टकानि_प्रत्येकम्_thread);
    size_t src0_spad_half    = layout->src0_spad_half_आकार;
    size_t dst_spad_half     = layout->dst_spad_half_आकार;
    ध्रुव शून्य * s1_सारणी    = VTCM_LAYOUT_PTR(ध्रुव शून्य, bctx->vtcm_आधारः, layout->off_src1);

    dma_queue * dma_q = octx->ctx->dma[ith];
    uint32_t ir_prefetch = start_row;
    पूर्णाङ्क spad_idx = 0;

    प्रति (पूर्णाङ्क k = 0; k < 2 && ir_prefetch < अन्तः_row; k++) {
        uint32_t current_खण्डम्_आकार = calc_खण्डम्_आकार(bctx, ir_prefetch, अन्तः_row, ne01, ne02);
        uint32_t i03, i02, i01, rem;
        i03 = fastdiv(ir_prefetch, &bctx->src0_dim12_div);
        rem = ir_prefetch - i03 * (ne02 * ne01);
        i02 = fastdiv(rem, &bctx->src0_dim1_div);
        i01 = rem - i02 * ne01;

        dma_addr_t src0_curr = src0->दत्त + i03 * nb03 + i02 * nb02 + i01 * nb01;
        dma_addr_t dst_curr  = dst->दत्त  + i03 * nb3  + i02 * nb2  + i01 * nb1;

        uint8_t * s0_spad = src0_spad_आधारः + spad_idx * src0_spad_half;
        uint8_t * d_spad  = dst_spad_आधारः  + spad_idx * dst_spad_half;

        dma_queue_प्रक्षिप(dma_q, dma_make_दत्त(dst_curr, d_spad), nb1, bctx->dst_row_आकार_aligned, row_आकार_अष्टकानि, 0);
        dma_queue_प्रक्षिप(dma_q, dma_make_दत्त(s0_spad, src0_curr), bctx->src0_row_आकार_aligned, nb01, row_आकार_अष्टकानि,
                                                             current_खण्डम्_आकार);
        ir_prefetch += current_खण्डम्_आकार;
        spad_idx ^= 1;
    }

    संरचना htp_thread_trace * tr = &octx->ctx->trace[ith];
    compute_scalar_t compute = (compute_scalar_t) bctx->compute;

    प्रति (uint32_t ir = start_row; ir < अन्तः_row; ) {
        uint32_t current_खण्डम्_आकार = calc_खण्डम्_आकार(bctx, ir, अन्तः_row, ne01, ne02);

        uint8_t * d_spad  = (uint8_t *) dma_queue_निष्कासय(dma_q).src;
        uint8_t * s0_spad = (uint8_t *) dma_queue_निष्कासय(dma_q).dst;

        uint32_t i03, i02, i01, rem;
        i03 = fastdiv(ir, &bctx->src0_dim12_div);
        rem = ir - i03 * (ne02 * ne01);
        i02 = fastdiv(rem, &bctx->src0_dim1_div);
        i01 = rem - i02 * ne01;

        uint32_t i13 = fastmodulo(i03, ne13, &bctx->src1_dim3_div);
        uint32_t i12 = fastmodulo(i02, ne12, &bctx->src1_dim2_div);
        uint32_t i11 = fastmodulo(i01, ne11, &bctx->src1_dim1_div);
        ध्रुव uint8_t * scalar_सारणी = (ध्रुव uint8_t *) s1_सारणी +
            i13 * plane_stride + i12 * row_stride;

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);
        compute(d_spad, s0_spad, scalar_सारणी, i11, ne11, current_खण्डम्_आकार,
                bctx->dst_row_आकार_aligned, bctx->src0_row_आकार_aligned, ne00);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);

        dma_addr_t dst_curr = dst->दत्त + i03 * nb3 + i02 * nb2 + i01 * nb1;
        dma_queue_प्रक्षिप(dma_q, dma_make_दत्त(dst_curr, d_spad), nb1, bctx->dst_row_आकार_aligned, row_आकार_अष्टकानि,
                                                                current_खण्डम्_आकार);

        यदि (ir_prefetch < अन्तः_row) {
            uint32_t अग्र_खण्डम्_आकार = calc_खण्डम्_आकार(bctx, ir_prefetch, अन्तः_row, ne01, ne02);
            uint32_t p03, p02, p01, prem;
            p03 = fastdiv(ir_prefetch, &bctx->src0_dim12_div);
            prem = ir_prefetch - p03 * (ne02 * ne01);
            p02 = fastdiv(prem, &bctx->src0_dim1_div);
            p01 = prem - p02 * ne01;
            dma_addr_t s0_अग्र = src0->दत्त + p03 * nb03 + p02 * nb02 + p01 * nb01;
            dma_queue_प्रक्षिप(dma_q, dma_make_दत्त(s0_spad, s0_अग्र), bctx->src0_row_आकार_aligned, nb01, row_आकार_अष्टकानि,
                                                               अग्र_खण्डम्_आकार);
            ir_prefetch += अग्र_खण्डम्_आकार;
        }
        ir += current_खण्डम्_आकार;
    }

    dma_queue_flush(dma_q);
}

// 2. Vector Same Shape (ne1x == ne0x) or Simple Broadcast
स्थिर शून्य द्विमान_thread_सदिशम्_same_shape(अचिह्नित पूर्णाङ्क nth, अचिह्नित पूर्णाङ्क ith, शून्य * दत्त) {
    संरचना htp_द्विमान_context * bctx = (संरचना htp_द्विमान_context *) दत्त;
    संरचना htp_ops_context * octx = bctx->octx;
    htp_द्विमान_preamble;

    ध्रुव uint32_t row_आकार_अष्टकानि = bctx->row_आकार_अष्टकानि;
    ध्रुव uint32_t start_row = bctx->row_start + bctx->nrows_प्रत्येकम्_thread * ith;
    ध्रुव uint32_t अन्तः_row   = MIN(start_row + bctx->nrows_प्रत्येकम्_thread, bctx->row_start + bctx->योग_पङ्क्तिकाः);
    यदि (start_row >= अन्तः_row) प्रत्यागम;

    FARF(उच्चः_2, "binary-same-shape: %d/%d (%u:%u) row-size %u (%u)",
         ith, nth, start_row, अन्तः_row, nb01, bctx->dst_row_आकार_aligned);

    ध्रुव संरचना htp_द्विमान_vtcm_layout * layout = &bctx->vtcm_layout;
    uint8_t * src0_spad_आधारः = VTCM_LAYOUT_PTR(uint8_t, bctx->vtcm_आधारः, layout->off_src0) + (ith * layout->src0_अष्टकानि_प्रत्येकम्_thread);
    uint8_t * src1_spad_आधारः = VTCM_LAYOUT_PTR(uint8_t, bctx->vtcm_आधारः, layout->off_src1) + (ith * layout->src1_अष्टकानि_प्रत्येकम्_thread);
    uint8_t * dst_spad_आधारः  = VTCM_LAYOUT_PTR(uint8_t, bctx->vtcm_आधारः, layout->off_dst)  + (ith * layout->dst_अष्टकानि_प्रत्येकम्_thread);

    size_t src0_spad_half = layout->src0_spad_half_आकार;
    size_t src1_spad_half = layout->src1_spad_half_आकार;
    size_t dst_spad_half  = layout->dst_spad_half_आकार;

    dma_queue * dma_q = octx->ctx->dma[ith];
    uint32_t ir_prefetch = start_row;
    पूर्णाङ्क spad_idx = 0;

    प्रति (पूर्णाङ्क k = 0; k < 2 && ir_prefetch < अन्तः_row; k++) {
        uint32_t current_खण्डम्_आकार = calc_खण्डम्_आकार(bctx, ir_prefetch, अन्तः_row, ne01, ne02);
        uint32_t i03, i02, i01, rem;
        i03 = fastdiv(ir_prefetch, &bctx->src0_dim12_div);
        rem = ir_prefetch - i03 * (ne02 * ne01);
        i02 = fastdiv(rem, &bctx->src0_dim1_div);
        i01 = rem - i02 * ne01;

        uint32_t i13 = (ne13 == 1) ? 0 : i03;
        uint32_t i12 = (ne12 == 1) ? 0 : i02;
        uint32_t i11 = (ne11 == 1) ? 0 : i01;

        dma_addr_t src0_curr = src0->दत्त + i03 * nb03 + i02 * nb02 + i01 * nb01;
        dma_addr_t src1_curr = src1->दत्त + i13 * nb13 + i12 * nb12 + i11 * nb11;
        dma_addr_t dst_curr  = dst->दत्त  + i03 * nb3  + i02 * nb2  + i01 * nb1;

        uint8_t * s0_spad = src0_spad_आधारः + spad_idx * src0_spad_half;
        uint8_t * s1_spad = src1_spad_आधारः + spad_idx * src1_spad_half;
        uint8_t * d_spad  = dst_spad_आधारः  + spad_idx * dst_spad_half;

        dma_queue_प्रक्षिप(dma_q, dma_make_दत्त(dst_curr, d_spad), nb1, bctx->dst_row_आकार_aligned, row_आकार_अष्टकानि, 0);
        dma_queue_प्रक्षिप(dma_q, dma_make_दत्त(s0_spad, src0_curr), bctx->src0_row_आकार_aligned, nb01, row_आकार_अष्टकानि, current_खण्डम्_आकार);
        dma_queue_प्रक्षिप(dma_q, dma_make_दत्त(s1_spad, src1_curr), bctx->src1_row_आकार_aligned, nb11, row_आकार_अष्टकानि, current_खण्डम्_आकार);
        ir_prefetch += current_खण्डम्_आकार;
        spad_idx ^= 1;
    }

    संरचना htp_thread_trace * tr = &octx->ctx->trace[ith];
    compute_same_shape_t compute = (compute_same_shape_t) bctx->compute;

    प्रति (uint32_t ir = start_row; ir < अन्तः_row; ) {
        uint32_t current_खण्डम्_आकार = calc_खण्डम्_आकार(bctx, ir, अन्तः_row, ne01, ne02);
        uint8_t * d_spad  = (uint8_t *) dma_queue_निष्कासय(dma_q).src;
        uint8_t * s0_spad = (uint8_t *) dma_queue_निष्कासय(dma_q).dst;
        uint8_t * s1_spad = (uint8_t *) dma_queue_निष्कासय(dma_q).dst;

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);
        compute(d_spad, s0_spad, s1_spad, current_खण्डम्_आकार,
                bctx->dst_row_आकार_aligned, bctx->src0_row_आकार_aligned, bctx->src1_row_आकार_aligned, ne00);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);

        uint32_t i03, i02, i01, rem;
        i03 = fastdiv(ir, &bctx->src0_dim12_div);
        rem = ir - i03 * (ne02 * ne01);
        i02 = fastdiv(rem, &bctx->src0_dim1_div);
        i01 = rem - i02 * ne01;
        dma_addr_t dst_curr = dst->दत्त + i03 * nb3 + i02 * nb2 + i01 * nb1;
        dma_queue_प्रक्षिप(dma_q, dma_make_दत्त(dst_curr, d_spad), nb1, bctx->dst_row_आकार_aligned, row_आकार_अष्टकानि, current_खण्डम्_आकार);

        यदि (ir_prefetch < अन्तः_row) {
            uint32_t अग्र_खण्डम्_आकार = calc_खण्डम्_आकार(bctx, ir_prefetch, अन्तः_row, ne01, ne02);
            uint32_t p03, p02, p01, prem;
            p03 = fastdiv(ir_prefetch, &bctx->src0_dim12_div);
            prem = ir_prefetch - p03 * (ne02 * ne01);
            p02 = fastdiv(prem, &bctx->src0_dim1_div);
            p01 = prem - p02 * ne01;

            uint32_t p13 = (ne13 == 1) ? 0 : p03;
            uint32_t p12 = (ne12 == 1) ? 0 : p02;
            uint32_t p11 = (ne11 == 1) ? 0 : p01;

            dma_addr_t s0_अग्र = src0->दत्त + p03 * nb03 + p02 * nb02 + p01 * nb01;
            dma_addr_t s1_अग्र = src1->दत्त + p13 * nb13 + p12 * nb12 + p11 * nb11;

            dma_queue_प्रक्षिप(dma_q, dma_make_दत्त(s0_spad, s0_अग्र), bctx->src0_row_आकार_aligned, nb01, row_आकार_अष्टकानि, अग्र_खण्डम्_आकार);
            dma_queue_प्रक्षिप(dma_q, dma_make_दत्त(s1_spad, s1_अग्र), bctx->src1_row_आकार_aligned, nb11, row_आकार_अष्टकानि, अग्र_खण्डम्_आकार);

            ir_prefetch += अग्र_खण्डम्_आकार;
        }
        ir += current_खण्डम्_आकार;
    }

    dma_queue_flush(dma_q);
}

// 3. Row Broadcast (ne11 == 1, ne12 == 1, single row src1)
स्थिर शून्य द्विमान_thread_सदिशम्_row_प्रसारणम्(अचिह्नित पूर्णाङ्क nth, अचिह्नित पूर्णाङ्क ith, शून्य * दत्त) {
    संरचना htp_द्विमान_context * bctx = (संरचना htp_द्विमान_context *) दत्त;
    संरचना htp_ops_context * octx = bctx->octx;
    htp_द्विमान_preamble;

    ध्रुव uint32_t row_आकार_अष्टकानि = bctx->row_आकार_अष्टकानि;
    ध्रुव uint32_t start_row  = bctx->row_start + bctx->nrows_प्रत्येकम्_thread * ith;
    ध्रुव uint32_t अन्तः_row    = MIN(start_row + bctx->nrows_प्रत्येकम्_thread, bctx->row_start + bctx->योग_पङ्क्तिकाः);
    यदि (start_row >= अन्तः_row) प्रत्यागम;

    FARF(उच्चः_2, "binary-row-bcast: %d/%d (%u:%u) row-size %u (%u)",
         ith, nth, start_row, अन्तः_row, nb01, bctx->dst_row_आकार_aligned);

    ध्रुव संरचना htp_द्विमान_vtcm_layout * layout = &bctx->vtcm_layout;
    uint8_t * src0_spad_आधारः = VTCM_LAYOUT_PTR(uint8_t, bctx->vtcm_आधारः, layout->off_src0) + (ith * layout->src0_अष्टकानि_प्रत्येकम्_thread);
    uint8_t * dst_spad_आधारः  = VTCM_LAYOUT_PTR(uint8_t, bctx->vtcm_आधारः, layout->off_dst)  + (ith * layout->dst_अष्टकानि_प्रत्येकम्_thread);

    size_t src0_spad_half = layout->src0_spad_half_आकार;
    size_t dst_spad_half  = layout->dst_spad_half_आकार;

    dma_queue * dma_q = octx->ctx->dma[ith];
    uint32_t ir_prefetch = start_row;
    पूर्णाङ्क spad_idx = 0;

    शून्य * s1_ptr = VTCM_LAYOUT_PTR(शून्य, bctx->vtcm_आधारः, layout->off_src1);

    प्रति (पूर्णाङ्क k = 0; k < 2 && ir_prefetch < अन्तः_row; k++) {
        uint32_t current_खण्डम्_आकार = calc_खण्डम्_आकार(bctx, ir_prefetch, अन्तः_row, ne01, ne02);
        uint32_t i03 = fastdiv(ir_prefetch, &bctx->src0_dim12_div);
        uint32_t rem = ir_prefetch - i03 * (ne02 * ne01);
        uint32_t i02 = fastdiv(rem, &bctx->src0_dim1_div);
        uint32_t i01 = rem - i02 * ne01;

        dma_addr_t src0_curr = src0->दत्त + i03 * nb03 + i02 * nb02 + i01 * nb01;
        dma_addr_t dst_curr  = dst->दत्त  + i03 * nb3  + i02 * nb2  + i01 * nb1;

        uint8_t * s0_spad = src0_spad_आधारः + spad_idx * src0_spad_half;
        uint8_t * d_spad  = dst_spad_आधारः  + spad_idx * dst_spad_half;

        dma_queue_प्रक्षिप(dma_q, dma_make_दत्त(dst_curr, d_spad), nb1, bctx->dst_row_आकार_aligned, row_आकार_अष्टकानि, 0);
        dma_queue_प्रक्षिप(dma_q, dma_make_दत्त(s0_spad, src0_curr), bctx->src0_row_आकार_aligned, nb01, row_आकार_अष्टकानि, current_खण्डम्_आकार);
        ir_prefetch += current_खण्डम्_आकार;
        spad_idx ^= 1;
    }

    संरचना htp_thread_trace * tr = &octx->ctx->trace[ith];
    compute_row_bcast_t compute = (compute_row_bcast_t) bctx->compute;

    प्रति (uint32_t ir = start_row; ir < अन्तः_row; ) {
        uint32_t current_खण्डम्_आकार = calc_खण्डम्_आकार(bctx, ir, अन्तः_row, ne01, ne02);
        uint8_t * d_spad  = (uint8_t *) dma_queue_निष्कासय(dma_q).src;
        uint8_t * s0_spad = (uint8_t *) dma_queue_निष्कासय(dma_q).dst;

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);
        compute(d_spad, s0_spad, (ध्रुव uint8_t *)s1_ptr, current_खण्डम्_आकार,
                bctx->dst_row_आकार_aligned, bctx->src0_row_आकार_aligned, ne00);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);

        uint32_t i03 = fastdiv(ir, &bctx->src0_dim12_div);
        uint32_t rem = ir - i03 * (ne02 * ne01);
        uint32_t i02 = fastdiv(rem, &bctx->src0_dim1_div);
        uint32_t i01 = rem - i02 * ne01;
        dma_addr_t dst_curr = dst->दत्त + i03 * nb3 + i02 * nb2 + i01 * nb1;
        dma_queue_प्रक्षिप(dma_q, dma_make_दत्त(dst_curr, d_spad), nb1, bctx->dst_row_आकार_aligned, row_आकार_अष्टकानि, current_खण्डम्_आकार);

        यदि (ir_prefetch < अन्तः_row) {
            uint32_t अग्र_खण्डम्_आकार = calc_खण्डम्_आकार(bctx, ir_prefetch, अन्तः_row, ne01, ne02);
            uint32_t p03, p02, p01, prem;
            p03  = fastdiv(ir_prefetch, &bctx->src0_dim12_div);
            prem = ir_prefetch - p03 * (ne02 * ne01);
            p02  = fastdiv(prem, &bctx->src0_dim1_div);
            p01  = prem - p02 * ne01;
            dma_addr_t s0_अग्र = src0->दत्त + p03 * nb03 + p02 * nb02 + p01 * nb01;
            dma_queue_प्रक्षिप(dma_q, dma_make_दत्त(s0_spad, s0_अग्र), bctx->src0_row_आकार_aligned, nb01, row_आकार_अष्टकानि, अग्र_खण्डम्_आकार);
            ir_prefetch += अग्र_खण्डम्_आकार;
        }
        ir += current_खण्डम्_आकार;
    }

    dma_queue_flush(dma_q);
}

// 4. Vector Complex (ne10 == ne00, complex broadcast)
स्थिर शून्य द्विमान_thread_सदिशम्_complex(अचिह्नित पूर्णाङ्क nth, अचिह्नित पूर्णाङ्क ith, शून्य * दत्त) {
    संरचना htp_द्विमान_context * bctx = (संरचना htp_द्विमान_context *) दत्त;
    संरचना htp_ops_context * octx = bctx->octx;
    htp_द्विमान_preamble;

    ध्रुव uint32_t row_आकार_अष्टकानि = bctx->row_आकार_अष्टकानि;
    ध्रुव uint32_t start_row  = bctx->row_start + bctx->nrows_प्रत्येकम्_thread * ith;
    ध्रुव uint32_t अन्तः_row    = MIN(start_row + bctx->nrows_प्रत्येकम्_thread, bctx->row_start + bctx->योग_पङ्क्तिकाः);
    यदि (start_row >= अन्तः_row) प्रत्यागम;

    FARF(उच्चः_2, "binary-complex: %d/%d (%u:%u) row-size %u (%u)",
         ith, nth, start_row, अन्तः_row, nb01, bctx->dst_row_आकार_aligned);

    ध्रुव संरचना htp_द्विमान_vtcm_layout * layout = &bctx->vtcm_layout;
    uint8_t * src0_spad_आधारः = VTCM_LAYOUT_PTR(uint8_t, bctx->vtcm_आधारः, layout->off_src0) + (ith * layout->src0_अष्टकानि_प्रत्येकम्_thread);
    uint8_t * dst_spad_आधारः  = VTCM_LAYOUT_PTR(uint8_t, bctx->vtcm_आधारः, layout->off_dst)  + (ith * layout->dst_अष्टकानि_प्रत्येकम्_thread);
    size_t src0_spad_half    = layout->src0_spad_half_आकार;
    size_t dst_spad_half     = layout->dst_spad_half_आकार;
    ध्रुव uint8_t * vtcm_src1 = VTCM_LAYOUT_PTR(ध्रुव uint8_t, bctx->vtcm_आधारः, layout->off_src1);

    ध्रुव uint32_t elem_आकार = (src0->type == HTP_TYPE_F32) ? परिमाण(भिन्न) : परिमाण(_Float16);
    ध्रुव size_t vtcm_row_stride   = (size_t) ne10 * elem_आकार;
    ध्रुव size_t vtcm_plane_stride = (size_t) ne11 * vtcm_row_stride;
    ध्रुव size_t vtcm_vol_stride   = (size_t) ne12 * vtcm_plane_stride;

    dma_queue * dma_q = octx->ctx->dma[ith];
    uint32_t ir_prefetch = start_row;
    पूर्णाङ्क spad_idx = 0;

    प्रति (पूर्णाङ्क k = 0; k < 2 && ir_prefetch < अन्तः_row; k++) {
        uint32_t current_खण्डम्_आकार = calc_खण्डम्_आकार(bctx, ir_prefetch, अन्तः_row, ne01, ne02);
        uint32_t i03 = fastdiv(ir_prefetch, &bctx->src0_dim12_div);
        uint32_t rem = ir_prefetch - i03 * (ne02 * ne01);
        uint32_t i02 = fastdiv(rem, &bctx->src0_dim1_div);
        uint32_t i01 = rem - i02 * ne01;

        dma_addr_t src0_curr = src0->दत्त + i03 * nb03 + i02 * nb02 + i01 * nb01;
        dma_addr_t dst_curr  = dst->दत्त  + i03 * nb3  + i02 * nb2  + i01 * nb1;

        uint8_t * s0_spad = src0_spad_आधारः + spad_idx * src0_spad_half;
        uint8_t * d_spad  = dst_spad_आधारः  + spad_idx * dst_spad_half;

        dma_queue_प्रक्षिप(dma_q, dma_make_दत्त(dst_curr, d_spad), nb1, bctx->dst_row_आकार_aligned, row_आकार_अष्टकानि, 0);
        dma_queue_प्रक्षिप(dma_q, dma_make_दत्त(s0_spad, src0_curr), bctx->src0_row_आकार_aligned, nb01, row_आकार_अष्टकानि, current_खण्डम्_आकार);
        ir_prefetch += current_खण्डम्_आकार;
        spad_idx ^= 1;
    }

    संरचना htp_thread_trace * tr = &octx->ctx->trace[ith];
    compute_complex_t compute = (compute_complex_t) bctx->compute;

    प्रति (uint32_t ir = start_row; ir < अन्तः_row; ) {
        uint32_t current_खण्डम्_आकार = calc_खण्डम्_आकार(bctx, ir, अन्तः_row, ne01, ne02);
        uint8_t * d_spad = (uint8_t *) dma_queue_निष्कासय(dma_q).src;
        uint8_t * s0_spad = (uint8_t *) dma_queue_निष्कासय(dma_q).dst;

        uint32_t i03 = fastdiv(ir, &bctx->src0_dim12_div);
        uint32_t rem = ir - i03 * (ne02 * ne01);
        uint32_t i02 = fastdiv(rem, &bctx->src0_dim1_div);
        uint32_t i01 = rem - i02 * ne01;

        uint32_t i13 = fastmodulo(i03, ne13, &bctx->src1_dim3_div);
        uint32_t i12 = fastmodulo(i02, ne12, &bctx->src1_dim2_div);
        ध्रुव uint8_t * src1_plane = vtcm_src1 + i13 * vtcm_vol_stride + i12 * vtcm_plane_stride;

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);
        compute(d_spad, s0_spad, src1_plane, i01, ne11, &bctx->src1_dim1_div, vtcm_row_stride,
                current_खण्डम्_आकार, bctx->dst_row_आकार_aligned, bctx->src0_row_आकार_aligned, ne00);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);

        dma_addr_t dst_curr = dst->दत्त + i03 * nb3 + i02 * nb2 + i01 * nb1;
        dma_queue_प्रक्षिप(dma_q, dma_make_दत्त(dst_curr, d_spad), nb1, bctx->dst_row_आकार_aligned, row_आकार_अष्टकानि, current_खण्डम्_आकार);

        यदि (ir_prefetch < अन्तः_row) {
            uint32_t अग्र_खण्डम्_आकार = calc_खण्डम्_आकार(bctx, ir_prefetch, अन्तः_row, ne01, ne02);
            uint32_t p03, p02, p01, prem;
            p03  = fastdiv(ir_prefetch, &bctx->src0_dim12_div);
            prem = ir_prefetch - p03 * (ne02 * ne01);
            p02  = fastdiv(prem, &bctx->src0_dim1_div);
            p01  = prem - p02 * ne01;
            dma_addr_t s0_अग्र = src0->दत्त + p03 * nb03 + p02 * nb02 + p01 * nb01;
            dma_queue_प्रक्षिप(dma_q, dma_make_दत्त(s0_spad, s0_अग्र), bctx->src0_row_आकार_aligned, nb01, row_आकार_अष्टकानि, अग्र_खण्डम्_आकार);
            ir_prefetch += अग्र_खण्डम्_आकार;
        }
        ir += current_खण्डम्_आकार;
    }

    dma_queue_flush(dma_q);
}

// 5. Element Repeat (ne10 != ne00)
स्थिर शून्य द्विमान_thread_element_repeat(अचिह्नित पूर्णाङ्क nth, अचिह्नित पूर्णाङ्क ith, शून्य * दत्त) {
    संरचना htp_द्विमान_context * bctx = (संरचना htp_द्विमान_context *) दत्त;
    संरचना htp_ops_context * octx = bctx->octx;
    htp_द्विमान_preamble;

    ध्रुव uint32_t row_आकार_अष्टकानि = bctx->row_आकार_अष्टकानि;
    ध्रुव uint32_t start_row  = bctx->row_start + bctx->nrows_प्रत्येकम्_thread * ith;
    ध्रुव uint32_t अन्तः_row    = MIN(start_row + bctx->nrows_प्रत्येकम्_thread, bctx->row_start + bctx->योग_पङ्क्तिकाः);
    यदि (start_row >= अन्तः_row) प्रत्यागम;

    ध्रुव संरचना htp_द्विमान_vtcm_layout * layout = &bctx->vtcm_layout;
    uint8_t * src0_spad_आधारः = VTCM_LAYOUT_PTR(uint8_t, bctx->vtcm_आधारः, layout->off_src0) + (ith * layout->src0_अष्टकानि_प्रत्येकम्_thread);
    uint8_t * dst_spad_आधारः  = VTCM_LAYOUT_PTR(uint8_t, bctx->vtcm_आधारः, layout->off_dst)  + (ith * layout->dst_अष्टकानि_प्रत्येकम्_thread);
    size_t src0_spad_half    = layout->src0_spad_half_आकार;
    size_t dst_spad_half     = layout->dst_spad_half_आकार;
    ध्रुव uint8_t * vtcm_src1 = VTCM_LAYOUT_PTR(ध्रुव uint8_t, bctx->vtcm_आधारः, layout->off_src1);

    ध्रुव uint32_t elem_आकार = (src0->type == HTP_TYPE_F32) ? परिमाण(भिन्न) : परिमाण(_Float16);
    ध्रुव size_t vtcm_row_stride   = (size_t) ne10 * elem_आकार;
    ध्रुव size_t vtcm_plane_stride = (size_t) ne11 * vtcm_row_stride;
    ध्रुव size_t vtcm_vol_stride   = (size_t) ne12 * vtcm_plane_stride;

    FARF(उच्चः_2, "binary-repeat: %d/%d (%u:%u) row-size %u (%u)",
         ith, nth, start_row, अन्तः_row, nb01, bctx->dst_row_आकार_aligned);

    dma_queue * dma_q = octx->ctx->dma[ith];
    uint32_t ir_prefetch = start_row;
    पूर्णाङ्क spad_idx = 0;

    प्रति (पूर्णाङ्क k = 0; k < 2 && ir_prefetch < अन्तः_row; k++) {
        uint32_t current_खण्डम्_आकार = calc_खण्डम्_आकार(bctx, ir_prefetch, अन्तः_row, ne01, ne02);
        uint32_t i03 = fastdiv(ir_prefetch, &bctx->src0_dim12_div);
        uint32_t rem = ir_prefetch - i03 * (ne02 * ne01);
        uint32_t i02 = fastdiv(rem, &bctx->src0_dim1_div);
        uint32_t i01 = rem - i02 * ne01;

        dma_addr_t src0_curr = src0->दत्त + i03 * nb03 + i02 * nb02 + i01 * nb01;
        dma_addr_t dst_curr  = dst->दत्त  + i03 * nb3  + i02 * nb2  + i01 * nb1;

        uint8_t * s0_spad = src0_spad_आधारः + spad_idx * src0_spad_half;
        uint8_t * d_spad  = dst_spad_आधारः  + spad_idx * dst_spad_half;

        dma_queue_प्रक्षिप(dma_q, dma_make_दत्त(dst_curr, d_spad), nb1, bctx->dst_row_आकार_aligned, row_आकार_अष्टकानि, 0);
        dma_queue_प्रक्षिप(dma_q, dma_make_दत्त(s0_spad, src0_curr), bctx->src0_row_आकार_aligned, nb01, row_आकार_अष्टकानि, current_खण्डम्_आकार);
        ir_prefetch += current_खण्डम्_आकार;
        spad_idx ^= 1;
    }

    संरचना htp_thread_trace * tr = &octx->ctx->trace[ith];
    compute_repeat_t compute = (compute_repeat_t) bctx->compute;

    प्रति (uint32_t ir = start_row; ir < अन्तः_row; ) {
        uint32_t current_खण्डम्_आकार = calc_खण्डम्_आकार(bctx, ir, अन्तः_row, ne01, ne02);
        uint8_t * d_spad  = (uint8_t *) dma_queue_निष्कासय(dma_q).src;
        uint8_t * s0_spad = (uint8_t *) dma_queue_निष्कासय(dma_q).dst;

        uint32_t i03 = fastdiv(ir, &bctx->src0_dim12_div);
        uint32_t rem = ir - i03 * (ne02 * ne01);
        uint32_t i02 = fastdiv(rem, &bctx->src0_dim1_div);
        uint32_t i01 = rem - i02 * ne01;

        uint32_t i13 = fastmodulo(i03, ne13, &bctx->src1_dim3_div);
        uint32_t i12 = fastmodulo(i02, ne12, &bctx->src1_dim2_div);
        ध्रुव uint8_t * src1_plane = vtcm_src1 + i13 * vtcm_vol_stride + i12 * vtcm_plane_stride;

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);
        compute(d_spad, s0_spad, src1_plane, i01, ne11, &bctx->src1_dim1_div, vtcm_row_stride,
                current_खण्डम्_आकार, bctx->dst_row_आकार_aligned, bctx->src0_row_आकार_aligned, ne00, ne10);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);

        dma_addr_t dst_curr = dst->दत्त + i03 * nb3 + i02 * nb2 + i01 * nb1;
        dma_queue_प्रक्षिप(dma_q, dma_make_दत्त(dst_curr, d_spad), nb1, bctx->dst_row_आकार_aligned, row_आकार_अष्टकानि, current_खण्डम्_आकार);

        यदि (ir_prefetch < अन्तः_row) {
            uint32_t अग्र_खण्डम्_आकार = calc_खण्डम्_आकार(bctx, ir_prefetch, अन्तः_row, ne01, ne02);
            uint32_t p03, p02, p01, prem;
            p03  = fastdiv(ir_prefetch, &bctx->src0_dim12_div);
            prem = ir_prefetch - p03 * (ne02 * ne01);
            p02  = fastdiv(prem, &bctx->src0_dim1_div);
            p01  = prem - p02 * ne01;
            dma_addr_t s0_अग्र = src0->दत्त + p03 * nb03 + p02 * nb02 + p01 * nb01;
            dma_queue_प्रक्षिप(dma_q, dma_make_दत्त(s0_spad, s0_अग्र), bctx->src0_row_आकार_aligned, nb01, row_आकार_अष्टकानि, अग्र_खण्डम्_आकार);
            ir_prefetch += अग्र_खण्डम्_आकार;
        }
        ir += current_खण्डम्_आकार;
    }

    dma_queue_flush(dma_q);
}

// 6. ADD_ID (src1 gathered via src2 indices)
स्थिर शून्य द्विमान_thread_सङ्कलय_id_f32(अचिह्नित पूर्णाङ्क nth, अचिह्नित पूर्णाङ्क ith, शून्य * दत्त) {
    संरचना htp_द्विमान_context * bctx = (संरचना htp_द्विमान_context *) दत्त;
    संरचना htp_ops_context * octx = bctx->octx;

    ध्रुव संरचना htp_tensor * src0 = octx->src[0];
    ध्रुव संरचना htp_tensor * src1 = octx->src[1];
    ध्रुव संरचना htp_tensor * dst  = octx->dst;

    ध्रुव uint32_t ne00 = src0->ne[0];
    ध्रुव uint32_t ne01 = src0->ne[1];
    ध्रुव uint32_t ne02 = src0->ne[2];
    ध्रुव uint32_t ne03 = src0->ne[3];

    ध्रुव uint32_t nb01 = src0->nb[1];
    ध्रुव uint32_t nb02 = src0->nb[2];
    ध्रुव uint32_t nb03 = src0->nb[3];
    ध्रुव uint32_t src1_stride = bctx->src1_row_आकार_aligned;

    ध्रुव uint32_t nb1 = dst->nb[1];
    ध्रुव uint32_t nb2 = dst->nb[2];
    ध्रुव uint32_t nb3 = dst->nb[3];

    ध्रुव uint32_t row_आकार_अष्टकानि = bctx->row_आकार_अष्टकानि;
    ध्रुव uint32_t start_row = bctx->row_start + bctx->nrows_प्रत्येकम्_thread * ith;
    ध्रुव uint32_t अन्तः_row   = MIN(start_row + bctx->nrows_प्रत्येकम्_thread, bctx->row_start + bctx->योग_पङ्क्तिकाः);
    यदि (start_row >= अन्तः_row) प्रत्यागम;

    ध्रुव संरचना htp_द्विमान_vtcm_layout * layout = &bctx->vtcm_layout;
    uint8_t * src0_spad_आधारः = VTCM_LAYOUT_PTR(uint8_t, bctx->vtcm_आधारः, layout->off_src0) + (ith * layout->src0_अष्टकानि_प्रत्येकम्_thread);
    uint8_t * dst_spad_आधारः  = VTCM_LAYOUT_PTR(uint8_t, bctx->vtcm_आधारः, layout->off_dst)  + (ith * layout->dst_अष्टकानि_प्रत्येकम्_thread);
    ध्रुव uint8_t * vtcm_src1 = VTCM_LAYOUT_PTR(ध्रुव uint8_t, bctx->vtcm_आधारः, layout->off_src1);
    ध्रुव int32_t * vtcm_src2 = VTCM_LAYOUT_PTR(ध्रुव int32_t, bctx->vtcm_आधारः, layout->off_src2);
    size_t src0_spad_half    = layout->src0_spad_half_आकार;
    size_t dst_spad_half     = layout->dst_spad_half_आकार;

    dma_queue * dma_q = octx->ctx->dma[ith];
    uint32_t ir_prefetch = start_row;
    पूर्णाङ्क spad_idx = 0;

    प्रति (पूर्णाङ्क k = 0; k < 2 && ir_prefetch < अन्तः_row; k++) {
        uint32_t current_खण्डम्_आकार = calc_खण्डम्_आकार(bctx, ir_prefetch, अन्तः_row, ne01, ne02);
        uint32_t i03 = fastdiv(ir_prefetch, &bctx->src0_dim12_div);
        uint32_t rem = ir_prefetch - i03 * (ne02 * ne01);
        uint32_t i02 = fastdiv(rem, &bctx->src0_dim1_div);
        uint32_t i01 = rem - i02 * ne01;

        dma_addr_t src0_curr = src0->दत्त + i03 * nb03 + i02 * nb02 + i01 * nb01;
        dma_addr_t dst_curr  = dst->दत्त  + i03 * nb3  + i02 * nb2  + i01 * nb1;

        uint8_t * s0_spad = src0_spad_आधारः + spad_idx * src0_spad_half;
        uint8_t * d_spad  = dst_spad_आधारः  + spad_idx * dst_spad_half;

        dma_queue_प्रक्षिप(dma_q, dma_make_दत्त(dst_curr, d_spad), nb1, bctx->dst_row_आकार_aligned, row_आकार_अष्टकानि, 0);
        dma_queue_प्रक्षिप(dma_q, dma_make_दत्त(s0_spad, src0_curr), bctx->src0_row_आकार_aligned, nb01, row_आकार_अष्टकानि, current_खण्डम्_आकार);
        ir_prefetch += current_खण्डम्_आकार;
        spad_idx ^= 1;
    }

    संरचना htp_thread_trace * tr = &octx->ctx->trace[ith];

    प्रति (uint32_t ir = start_row; ir < अन्तः_row; ) {
        uint32_t current_खण्डम्_आकार = calc_खण्डम्_आकार(bctx, ir, अन्तः_row, ne01, ne02);
        uint8_t * d_spad = (uint8_t *) dma_queue_निष्कासय(dma_q).src;
        uint8_t * s0_spad = (uint8_t *) dma_queue_निष्कासय(dma_q).dst;

        uint32_t i03 = fastdiv(ir, &bctx->src0_dim12_div);
        uint32_t rem = ir - i03 * (ne02 * ne01);
        uint32_t i02 = fastdiv(rem, &bctx->src0_dim1_div);
        uint32_t i01 = rem - i02 * ne01;

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);
        compute_सङ्कलय_id_t compute = (compute_सङ्कलय_id_t) bctx->compute;
        compute(d_spad, s0_spad, vtcm_src1, vtcm_src2,
                i01, i02, ne01, src1_stride,
                current_खण्डम्_आकार, bctx->dst_row_आकार_aligned, bctx->src0_row_आकार_aligned, ne00);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);

        dma_addr_t dst_curr = dst->दत्त + i03 * nb3 + i02 * nb2 + i01 * nb1;
        dma_queue_प्रक्षिप(dma_q, dma_make_दत्त(dst_curr, d_spad), nb1, bctx->dst_row_आकार_aligned, row_आकार_अष्टकानि, current_खण्डम्_आकार);

        यदि (ir_prefetch < अन्तः_row) {
             uint32_t अग्र_खण्डम्_आकार = calc_खण्डम्_आकार(bctx, ir_prefetch, अन्तः_row, ne01, ne02);
             uint32_t p03, p02, p01, prem;
             p03  = fastdiv(ir_prefetch, &bctx->src0_dim12_div);
             prem = ir_prefetch - p03 * (ne02 * ne01);
             p02  = fastdiv(prem, &bctx->src0_dim1_div);
             p01  = prem - p02 * ne01;
             dma_addr_t s0_अग्र = src0->दत्त + p03 * nb03 + p02 * nb02 + p01 * nb01;
             dma_queue_प्रक्षिप(dma_q, dma_make_दत्त(s0_spad, s0_अग्र), bctx->src0_row_आकार_aligned, nb01, row_आकार_अष्टकानि, अग्र_खण्डम्_आकार);
             ir_prefetch += अग्र_खण्डम्_आकार;
        }
        ir += current_खण्डम्_आकार;
    }

    dma_queue_flush(dma_q);
}

स्थिर अन्तःस्थ शून्य hvx_div_scalar_f32(uint8_t * सीमित dst, ध्रुव uint8_t * सीमित src, ध्रुव भिन्न val, ध्रुव uint32_t अङ्कः_elems) {
    hvx_mul_scalar_f32(dst, src, 1.0f / val, अङ्कः_elems);
}

स्थिर अन्तःस्थ शून्य hvx_div_scalar_f16(uint8_t * सीमित dst, ध्रुव uint8_t * सीमित src, ध्रुव _Float16 val, ध्रुव uint32_t अङ्कः_elems) {
    hvx_div_scalar_f16_aa(dst, src, val, अङ्कः_elems);
}

प्रकारनिर्देश शून्य (*compute_द्विमान_chunked_t)(
    uint8_t * सीमित dst,
    ध्रुव uint8_t * सीमित src0,
    ध्रुव uint8_t * सीमित src1,
    ध्रुव uint32_t अङ्कः_elems
);

प्रकारनिर्देश शून्य (*compute_द्विमान_scalar_chunked_f32_t)(
    uint8_t * सीमित dst,
    ध्रुव uint8_t * सीमित src0,
    ध्रुव भिन्न val,
    ध्रुव uint32_t अङ्कः_elems
);

प्रकारनिर्देश शून्य (*compute_द्विमान_scalar_chunked_f16_t)(
    uint8_t * सीमित dst,
    ध्रुव uint8_t * सीमित src0,
    ध्रुव _Float16 val,
    ध्रुव uint32_t अङ्कः_elems
);

संरचना द्विमान_chunked_context {
    संरचना htp_ops_context * octx;
    संरचना htp_द्विमान_vtcm_layout vtcm_layout;
    uint8_t *                vtcm_आधारः;
    uint32_t                 elem_start;
    uint32_t                 nelem;
    uint32_t                 chunk_आकार;
    uint32_t                 chunks_प्रत्येकम्_thread;
    uint32_t                 योग_chunks;
    bool                     is_scalar;
    भिन्न                    scalar_f32;
    _Float16                 scalar_f16;
    compute_द्विमान_chunked_t compute;
    compute_द्विमान_scalar_chunked_f32_t compute_scalar_f32;
    compute_द्विमान_scalar_chunked_f16_t compute_scalar_f16;
};

स्थिर शून्य द्विमान_thread_chunked(अचिह्नित पूर्णाङ्क nth, अचिह्नित पूर्णाङ्क ith, शून्य * दत्त) {
    (शून्य) nth;
    संरचना द्विमान_chunked_context * ctx = (संरचना द्विमान_chunked_context *) दत्त;
    संरचना htp_ops_context * octx = ctx->octx;
    ध्रुव संरचना htp_tensor * src0 = octx->src[0];
    ध्रुव संरचना htp_tensor * src1 = octx->src[1];
    ध्रुव संरचना htp_tensor * dst  = octx->dst;

    ध्रुव uint32_t start_chunk = ctx->chunks_प्रत्येकम्_thread * ith;
    ध्रुव uint32_t अन्तः_chunk   = MIN(start_chunk + ctx->chunks_प्रत्येकम्_thread, ctx->योग_chunks);
    यदि (start_chunk >= अन्तः_chunk) {
        प्रत्यागम;
    }

    ध्रुव uint32_t src0_type   = src0->type;
    ध्रुव size_t   elem_आकार   = (src0_type == HTP_TYPE_F32) ? परिमाण(भिन्न) : परिमाण(_Float16);
    ध्रुव uint32_t chunk_आकार  = ctx->chunk_आकार;
    ध्रुव size_t   chunk_अष्टकानि = ctx->vtcm_layout.src0_spad_half_आकार;

    FARF(उच्चः_2, "binary-chunked: %d/%d (%u:%u) chunks %u elems %u",
         ith, nth, start_chunk, अन्तः_chunk, ctx->योग_chunks, ctx->nelem);

    ध्रुव संरचना htp_द्विमान_vtcm_layout * layout = &ctx->vtcm_layout;
    uint8_t * src0_spad_आधारः = VTCM_LAYOUT_PTR(uint8_t, ctx->vtcm_आधारः, layout->off_src0) + (ith * layout->src0_अष्टकानि_प्रत्येकम्_thread);
    uint8_t * src1_spad_आधारः = ctx->is_scalar ? रिक्त : (VTCM_LAYOUT_PTR(uint8_t, ctx->vtcm_आधारः, layout->off_src1) + (ith * layout->src1_अष्टकानि_प्रत्येकम्_thread));
    uint8_t * dst_spad_आधारः  = VTCM_LAYOUT_PTR(uint8_t, ctx->vtcm_आधारः, layout->off_dst)  + (ith * layout->dst_अष्टकानि_प्रत्येकम्_thread);

    dma_queue * dma_q = octx->ctx->dma[ith];
    uint32_t prefetch_chunk = start_chunk;
    पूर्णाङ्क spad_idx = 0;

    प्रति (पूर्णाङ्क k = 0; k < 2 && prefetch_chunk < अन्तः_chunk; k++) {
        ध्रुव uint32_t c_start   = ctx->elem_start + prefetch_chunk * chunk_आकार;
        ध्रुव uint32_t c_अन्तः     = MIN(c_start + chunk_आकार, ctx->elem_start + ctx->nelem);
        ध्रुव uint32_t cur_elems = c_अन्तः - c_start;
        ध्रुव uint32_t cur_अष्टकानि = cur_elems * elem_आकार;

        dma_addr_t s0_curr = src0->दत्त + (size_t) c_start * elem_आकार;
        dma_addr_t d_curr  = dst->दत्त  + (size_t) c_start * elem_आकार;

        uint8_t * s0_spad = src0_spad_आधारः + spad_idx * chunk_अष्टकानि;
        uint8_t * d_spad  = dst_spad_आधारः  + spad_idx * chunk_अष्टकानि;

        dma_queue_प्रक्षिप(dma_q, dma_make_दत्त(d_curr, d_spad), chunk_अष्टकानि, chunk_अष्टकानि, cur_अष्टकानि, 0);
        dma_queue_प्रक्षिप(dma_q, dma_make_दत्त(s0_spad, s0_curr), chunk_अष्टकानि, chunk_अष्टकानि, cur_अष्टकानि, 1);
        यदि (!ctx->is_scalar) {
            dma_addr_t s1_curr = src1->दत्त + (size_t) c_start * elem_आकार;
            uint8_t * s1_spad = src1_spad_आधारः + spad_idx * chunk_अष्टकानि;
            dma_queue_प्रक्षिप(dma_q, dma_make_दत्त(s1_spad, s1_curr), chunk_अष्टकानि, chunk_अष्टकानि, cur_अष्टकानि, 1);
        }

        prefetch_chunk++;
        spad_idx ^= 1;
    }

    संरचना htp_thread_trace * tr = &octx->ctx->trace[ith];

    प्रति (uint32_t c = start_chunk; c < अन्तः_chunk; c++) {
        ध्रुव uint32_t c_start   = ctx->elem_start + c * chunk_आकार;
        ध्रुव uint32_t c_अन्तः     = MIN(c_start + chunk_आकार, ctx->elem_start + ctx->nelem);
        ध्रुव uint32_t cur_elems = c_अन्तः - c_start;
        ध्रुव uint32_t cur_अष्टकानि = cur_elems * elem_आकार;

        uint8_t * d_spad  = (uint8_t *) dma_queue_निष्कासय(dma_q).src;
        uint8_t * s0_spad = (uint8_t *) dma_queue_निष्कासय(dma_q).dst;
        uint8_t * s1_spad = ctx->is_scalar ? रिक्त : (uint8_t *) dma_queue_निष्कासय(dma_q).dst;

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) c);
        यदि (ctx->is_scalar) {
            यदि (src0_type == HTP_TYPE_F32) {
                ctx->compute_scalar_f32(d_spad, s0_spad, ctx->scalar_f32, cur_elems);
            } अन्यथा {
                ctx->compute_scalar_f16(d_spad, s0_spad, ctx->scalar_f16, cur_elems);
            }
        } अन्यथा {
            ctx->compute(d_spad, s0_spad, s1_spad, cur_elems);
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) c);

        dma_addr_t dst_curr = dst->दत्त + (size_t) c_start * elem_आकार;
        dma_queue_प्रक्षिप(dma_q, dma_make_दत्त(dst_curr, d_spad), chunk_अष्टकानि, chunk_अष्टकानि, cur_अष्टकानि, 1);

        यदि (prefetch_chunk < अन्तः_chunk) {
            ध्रुव uint32_t pc_start  = ctx->elem_start + prefetch_chunk * chunk_आकार;
            ध्रुव uint32_t pc_अन्तः    = MIN(pc_start + chunk_आकार, ctx->elem_start + ctx->nelem);
            ध्रुव uint32_t p_elems   = pc_अन्तः - pc_start;
            ध्रुव uint32_t p_अष्टकानि   = p_elems * elem_आकार;

            dma_addr_t s0_अग्र = src0->दत्त + (size_t) pc_start * elem_आकार;
            dma_queue_प्रक्षिप(dma_q, dma_make_दत्त(s0_spad, s0_अग्र), chunk_अष्टकानि, chunk_अष्टकानि, p_अष्टकानि, 1);
            यदि (!ctx->is_scalar) {
                dma_addr_t s1_अग्र = src1->दत्त + (size_t) pc_start * elem_आकार;
                dma_queue_प्रक्षिप(dma_q, dma_make_दत्त(s1_spad, s1_अग्र), chunk_अष्टकानि, chunk_अष्टकानि, p_अष्टकानि, 1);
            }

            prefetch_chunk++;
        }
    }

    dma_queue_flush(dma_q);
}

स्थिर पूर्णाङ्क execute_op_द्विमान(संरचना htp_ops_context * octx) {
    ध्रुव संरचना htp_tensor * src0 = octx->src[0];
    ध्रुव संरचना htp_tensor * src1 = octx->src[1];
    ध्रुव संरचना htp_tensor * dst  = octx->dst;
    ध्रुव संरचना htp_द्विमान_kernel_params * kparams = (ध्रुव संरचना htp_द्विमान_kernel_params *) octx->kernel_params;

    ध्रुव uint32_t src0_nrows = src0->ne[1] * src0->ne[2] * src0->ne[3];

    // Use packed row sizes for VTCM allocation and alignment
    ध्रुव uint32_t src0_type = octx->src[0]->type;
    ध्रुव size_t elem_आकार = (src0_type == HTP_TYPE_F32) ? परिमाण(भिन्न) : परिमाण(_Float16);
    ध्रुव size_t src0_row_आकार = src0->ne[0] * elem_आकार;
    ध्रुव size_t src1_row_आकार = src1->ne[0] * elem_आकार;
    ध्रुव size_t dst_row_आकार  = dst->ne[0]  * elem_आकार;

    यदि (kparams->kernel_type == HTP_द्विमान_KERNEL_CHUNKED) {
        ध्रुव bool is_scalar = kparams->is_scalar || (src1->ne[0] == 1 && src1->ne[1] == 1 && src1->ne[2] == 1 && src1->ne[3] == 1);

        compute_द्विमान_chunked_t compute = रिक्त;
        compute_द्विमान_scalar_chunked_f32_t compute_scalar_f32 = रिक्त;
        compute_द्विमान_scalar_chunked_f16_t compute_scalar_f16 = रिक्त;

        यदि (is_scalar) {
            यदि (src0_type == HTP_TYPE_F32) {
                विकल्प (octx->op) {
                    पक्ष HTP_OP_सङ्कलय: compute_scalar_f32 = hvx_सङ्कलय_scalar_f32; भङ्ग;
                    पक्ष HTP_OP_SUB: compute_scalar_f32 = hvx_sub_scalar_f32; भङ्ग;
                    पक्ष HTP_OP_MUL: compute_scalar_f32 = hvx_mul_scalar_f32; भङ्ग;
                    पक्ष HTP_OP_DIV: compute_scalar_f32 = hvx_div_scalar_f32; भङ्ग;
                    स्वभाव: भङ्ग;
                }
            } अन्यथा यदि (src0_type == HTP_TYPE_F16) {
                विकल्प (octx->op) {
                    पक्ष HTP_OP_सङ्कलय: compute_scalar_f16 = hvx_सङ्कलय_scalar_f16; भङ्ग;
                    पक्ष HTP_OP_SUB: compute_scalar_f16 = hvx_sub_scalar_f16; भङ्ग;
                    पक्ष HTP_OP_MUL: compute_scalar_f16 = hvx_mul_scalar_f16; भङ्ग;
                    पक्ष HTP_OP_DIV: compute_scalar_f16 = hvx_div_scalar_f16; भङ्ग;
                    स्वभाव: भङ्ग;
                }
            }
            यदि (!compute_scalar_f32 && !compute_scalar_f16) {
                प्रत्यागम HTP_स्थितिः_NO_SUPPORT;
            }
        } अन्यथा {
            यदि (src0_type == HTP_TYPE_F32) {
                विकल्प (octx->op) {
                    पक्ष HTP_OP_सङ्कलय: compute = hvx_सङ्कलय_f32; भङ्ग;
                    पक्ष HTP_OP_SUB: compute = hvx_sub_f32; भङ्ग;
                    पक्ष HTP_OP_MUL: compute = hvx_mul_f32; भङ्ग;
                    पक्ष HTP_OP_DIV: compute = hvx_div_f32; भङ्ग;
                    स्वभाव: भङ्ग;
                }
            } अन्यथा यदि (src0_type == HTP_TYPE_F16) {
                विकल्प (octx->op) {
                    पक्ष HTP_OP_सङ्कलय: compute = hvx_सङ्कलय_f16; भङ्ग;
                    पक्ष HTP_OP_SUB: compute = hvx_sub_f16; भङ्ग;
                    पक्ष HTP_OP_MUL: compute = hvx_mul_f16; भङ्ग;
                    पक्ष HTP_OP_DIV: compute = hvx_div_f16; भङ्ग;
                    स्वभाव: भङ्ग;
                }
            }
            यदि (!compute) {
                प्रत्यागम HTP_स्थितिः_NO_SUPPORT;
            }
        }

        ध्रुव uint32_t योग_elems = (uint32_t) (src0->ne[0] * src0->ne[1] * src0->ne[2] * src0->ne[3]);
        यदि (योग_elems == 0) {
            प्रत्यागम HTP_स्थितिः_OK;
        }

        uint32_t elem_start = 0;
        uint32_t nelem      = योग_elems;
        ध्रुव uint32_t elems_प्रत्येकम्_पङ्क्तिः = (src0_type == HTP_TYPE_F32) ? 32 : 64;

        यदि (octx->ctx->mdev.गणनम् > 1) {
            ध्रुव bool can_विभज = htp_tensor_mdev_दत्त_aligned(dst) &&
                                   htp_tensor_mdev_दत्त_aligned(src0) &&
                                   (is_scalar || htp_tensor_mdev_दत्त_aligned(src1)) &&
                                   htp_tensor_is_contiguous(dst, elem_आकार) &&
                                   htp_tensor_is_contiguous(src0, elem_आकार) &&
                                   (is_scalar || htp_tensor_is_contiguous(src1, elem_आकार));
            ध्रुव संरचना htp_tensor_mdev_range range = htp_tensor_mdev_partition(
                योग_elems, can_विभज ? elems_प्रत्येकम्_पङ्क्तिः : 0,
                octx->ctx->mdev.idx, octx->ctx->mdev.गणनम्, &octx->ctx->mdev.गणनम्_div);
            elem_start = range.start;
            nelem      = range.गणनम्;
        }

        यदि (nelem == 0) {
            प्रत्यागम HTP_स्थितिः_OK;
        }

        यदि (!htp_ops_context_समुच्चयः_n_तन्तवः(octx, kparams->n_तन्तवः)) {
            प्रत्यागम HTP_स्थितिः_INVAL_PARAMS;
        }

        संरचना htp_द्विमान_vtcm_layout vtcm_layout;
        htp_द्विमान_vtcm_layout_build(&vtcm_layout, kparams, octx->ctx->vtcm_आकार);
        यदि (vtcm_layout.योग_अष्टकानि == 0 || vtcm_layout.योग_अष्टकानि > octx->ctx->vtcm_आकार) {
            प्रत्यागम HTP_स्थितिः_VTCM_TOO_SMALL;
        }

        ध्रुव uint32_t chunk_आकार = kparams->chunk_आकार > 0 ? kparams->chunk_आकार : (32768 / elem_आकार);
        ध्रुव uint32_t योग_chunks = (nelem + chunk_आकार - 1) / chunk_आकार;
        ध्रुव uint32_t n_तन्तवः = (योग_chunks >= 2) ? MIN(octx->n_तन्तवः, योग_chunks) : 1;
        ध्रुव uint32_t chunks_प्रत्येकम्_thread = (योग_chunks + n_तन्तवः - 1) / n_तन्तवः;

        भिन्न scalar_f32 = 0.0f;
        _Float16 scalar_f16 = 0;
        यदि (is_scalar) {
            uint8_t * vtcm_src1 = VTCM_LAYOUT_PTR(uint8_t, octx->ctx->vtcm_आधारः, vtcm_layout.off_src1);
            dma_queue * dma_q = octx->ctx->dma[0];
            dma_queue_प्रक्षिप(dma_q, dma_make_दत्त(vtcm_src1, src1->दत्त), 128, 0, elem_आकार, 1);
            dma_queue_निष्कासय(dma_q);

            यदि (src0_type == HTP_TYPE_F32) {
                scalar_f32 = ((ध्रुव भिन्न *) vtcm_src1)[0];
            } अन्यथा {
                scalar_f16 = ((ध्रुव _Float16 *) vtcm_src1)[0];
            }
        }

        संरचना द्विमान_chunked_context cctx = {
            .octx                = octx,
            .vtcm_layout         = vtcm_layout,
            .vtcm_आधारः           = (uint8_t *) octx->ctx->vtcm_आधारः,
            .elem_start          = elem_start,
            .nelem               = nelem,
            .chunk_आकार          = chunk_आकार,
            .chunks_प्रत्येकम्_thread   = chunks_प्रत्येकम्_thread,
            .योग_chunks        = योग_chunks,
            .is_scalar           = is_scalar,
            .scalar_f32          = scalar_f32,
            .scalar_f16          = scalar_f16,
            .compute             = compute,
            .compute_scalar_f32  = compute_scalar_f32,
            .compute_scalar_f16  = compute_scalar_f16,
        };

        work_queue_सततम्(octx->ctx->work_queue, द्विमान_thread_chunked, &cctx, n_तन्तवः);
        प्रत्यागम HTP_स्थितिः_OK;
    }

    uint32_t row_start = 0;
    uint32_t nrows     = src0_nrows;

    यदि (octx->ctx->mdev.गणनम् > 1) {
        uint32_t पङ्क्तिकाः_प्रत्येकम्_chunk = 0;
        htp_tensor_mdev_पङ्क्तिकाः_प्रत्येकम्_chunk(dst, (uint32_t) elem_आकार, (uint32_t) dst_row_आकार, &पङ्क्तिकाः_प्रत्येकम्_chunk);
        ध्रुव संरचना htp_tensor_mdev_range range = htp_tensor_mdev_partition(src0_nrows, पङ्क्तिकाः_प्रत्येकम्_chunk,
                                                       octx->ctx->mdev.idx, octx->ctx->mdev.गणनम्, &octx->ctx->mdev.गणनम्_div);
        row_start = range.start;
        nrows     = range.गणनम्;
    }

    यदि (nrows == 0) {
        प्रत्यागम HTP_स्थितिः_OK;
    }

    यदि (!htp_ops_context_समुच्चयः_n_तन्तवः(octx, kparams->n_तन्तवः)) {
        प्रत्यागम HTP_स्थितिः_INVAL_PARAMS;
    }

    ध्रुव uint32_t n_तन्तवः = octx->n_तन्तवः;
    ध्रुव size_t src0_row_आकार_aligned = kparams->src0_row_आकार_aligned;
    ध्रुव size_t src1_row_आकार_aligned = kparams->src1_row_आकार_aligned;
    ध्रुव size_t dst_row_आकार_aligned  = kparams->dst_row_आकार_aligned;

    संरचना htp_द्विमान_context bctx;
    bctx.vtcm_आधारः = (uint8_t *) octx->ctx->vtcm_आधारः;
    htp_द्विमान_vtcm_layout_build(&bctx.vtcm_layout, kparams, octx->ctx->vtcm_आकार);

    यदि (bctx.vtcm_layout.पङ्क्तिकाः_प्रत्येकम्_बफर == 0 || bctx.vtcm_layout.योग_अष्टकानि > octx->ctx->vtcm_आकार) {
        प्रत्यागम HTP_स्थितिः_VTCM_TOO_SMALL;
    }

    dma_queue * dma_q = octx->ctx->dma[0];
    uint8_t * vtcm_src1 = VTCM_LAYOUT_PTR(uint8_t, bctx.vtcm_आधारः, bctx.vtcm_layout.off_src1);
    यदि (kparams->kernel_type == HTP_द्विमान_KERNEL_ROW_BCAST) {
        dma_queue_प्रक्षिप(dma_q, dma_make_दत्त(vtcm_src1, src1->दत्त), bctx.vtcm_layout.src1_आकार, 0, src1->ne[0] * elem_आकार, 1);
    } अन्यथा यदि (kparams->kernel_type == HTP_द्विमान_KERNEL_SCALAR ||
               kparams->kernel_type == HTP_द्विमान_KERNEL_REPEAT ||
               kparams->kernel_type == HTP_द्विमान_KERNEL_COMPLEX) {
        यदि (!kparams->src1_strided) {
            ध्रुव size_t src1_अष्टकानि = (size_t) src1->ne[0] * src1->ne[1] * src1->ne[2] * src1->ne[3] * elem_आकार;
            dma_queue_प्रक्षिप(dma_q, dma_make_दत्त(vtcm_src1, src1->दत्त), bctx.vtcm_layout.src1_आकार, 0, src1_अष्टकानि, 1);
        } अन्यथा {
            ध्रुव uint32_t ne11 = src1->ne[1];
            ध्रुव uint32_t ne12 = src1->ne[2];
            ध्रुव uint32_t ne13 = src1->ne[3];
            ध्रुव size_t row_अष्टकानि = src1->ne[0] * elem_आकार;
            प्रति (uint32_t i13 = 0; i13 < ne13; i13++) {
                प्रति (uint32_t i12 = 0; i12 < ne12; i12++) {
                    dma_addr_t src_addr = src1->दत्त + i13 * src1->nb[3] + i12 * src1->nb[2];
                    uint8_t * dst_ptr = vtcm_src1 + (i13 * ne12 * ne11 + i12 * ne11) * row_अष्टकानि;
                    यावत् (!dma_queue_प्रक्षिप(dma_q, dma_make_दत्त(dst_ptr, src_addr),
                                           row_अष्टकानि, src1->nb[1], row_अष्टकानि, ne11)) {
                        dma_queue_निष्कासय(dma_q);
                    }
                }
            }
        }
    } अन्यथा यदि (kparams->kernel_type == HTP_द्विमान_KERNEL_सङ्कलय_ID) {
        ध्रुव संरचना htp_tensor * src2 = octx->src[2];
        uint8_t * vtcm_src2 = VTCM_LAYOUT_PTR(uint8_t, bctx.vtcm_आधारः, bctx.vtcm_layout.off_src2);
        dma_queue_प्रक्षिप(dma_q, dma_make_दत्त(vtcm_src1, src1->दत्त),
                       kparams->src1_row_आकार_aligned, src1->nb[1],
                       src1->ne[0] * elem_आकार, src1->ne[1]);
        ध्रुव size_t src2_row_अष्टकानि = src2->ne[0] * परिमाण(int32_t);
        dma_queue_प्रक्षिप(dma_q, dma_make_दत्त(vtcm_src2, src2->दत्त),
                       src2_row_अष्टकानि, src2->nb[1],
                       src2_row_अष्टकानि, src2->ne[1]);
    }

    bctx.octx                  = octx;
    bctx.nrows_प्रत्येकम्_thread      = fastdiv(nrows + n_तन्तवः - 1, &octx->n_तन्तवः_div);
    bctx.योग_पङ्क्तिकाः            = nrows;
    bctx.row_start             = row_start;
    bctx.खण्डम्_max             = bctx.vtcm_layout.पङ्क्तिकाः_प्रत्येकम्_बफर;
    bctx.src0_row_आकार_aligned = src0_row_आकार_aligned;
    bctx.src1_row_आकार_aligned = src1_row_आकार_aligned;
    bctx.dst_row_आकार_aligned  = dst_row_आकार_aligned;
    bctx.row_आकार_अष्टकानि        = src0_row_आकार;

    bctx.src0_dim1_div  = init_fastdiv_मानानि(src0->ne[1]);
    bctx.src0_dim2_div  = init_fastdiv_मानानि(src0->ne[2]);
    bctx.src0_dim12_div = init_fastdiv_मानानि(src0->ne[1] * src0->ne[2]);

    bctx.src1_dim1_div  = init_fastdiv_मानानि(src1->ne[1]);
    bctx.src1_dim2_div  = init_fastdiv_मानानि(src1->ne[2]);
    bctx.src1_dim3_div  = init_fastdiv_मानानि(src1->ne[3]);

    bctx.विभज_at_ne01  = kparams->विभज_at_ne01;
    bctx.विभज_at_ne02  = kparams->विभज_at_ne02;

    कार्यकर्ता_callback_t कार्यकर्ता_func = रिक्त;
    शून्य * compute_func = रिक्त;

    विकल्प (kparams->kernel_type) {
        पक्ष HTP_द्विमान_KERNEL_SAME_SHAPE:
            कार्यकर्ता_func = द्विमान_thread_सदिशम्_same_shape;
            यदि (src0_type == HTP_TYPE_F32) {
                विकल्प (octx->op) {
                    पक्ष HTP_OP_सङ्कलय: compute_func = compute_same_shape_सङ्कलय_f32; भङ्ग;
                    पक्ष HTP_OP_SUB: compute_func = compute_same_shape_sub_f32; भङ्ग;
                    पक्ष HTP_OP_MUL: compute_func = compute_same_shape_mul_f32; भङ्ग;
                    पक्ष HTP_OP_DIV: compute_func = compute_same_shape_div_f32; भङ्ग;
                    स्वभाव: भङ्ग;
                }
            } अन्यथा यदि (src0_type == HTP_TYPE_F16) {
                विकल्प (octx->op) {
                    पक्ष HTP_OP_सङ्कलय: compute_func = compute_same_shape_सङ्कलय_f16; भङ्ग;
                    पक्ष HTP_OP_SUB: compute_func = compute_same_shape_sub_f16; भङ्ग;
                    पक्ष HTP_OP_MUL: compute_func = compute_same_shape_mul_f16; भङ्ग;
                    पक्ष HTP_OP_DIV: compute_func = compute_same_shape_div_f16; भङ्ग;
                    स्वभाव: भङ्ग;
                }
            }
            भङ्ग;
        पक्ष HTP_द्विमान_KERNEL_ROW_BCAST:
            कार्यकर्ता_func = द्विमान_thread_सदिशम्_row_प्रसारणम्;
            यदि (src0_type == HTP_TYPE_F32) {
                विकल्प (octx->op) {
                    पक्ष HTP_OP_सङ्कलय: compute_func = compute_row_bcast_सङ्कलय_f32; भङ्ग;
                    पक्ष HTP_OP_SUB: compute_func = compute_row_bcast_sub_f32; भङ्ग;
                    पक्ष HTP_OP_MUL: compute_func = compute_row_bcast_mul_f32; भङ्ग;
                    पक्ष HTP_OP_DIV: compute_func = compute_row_bcast_div_f32; भङ्ग;
                    स्वभाव: भङ्ग;
                }
            } अन्यथा यदि (src0_type == HTP_TYPE_F16) {
                विकल्प (octx->op) {
                    पक्ष HTP_OP_सङ्कलय: compute_func = compute_row_bcast_सङ्कलय_f16; भङ्ग;
                    पक्ष HTP_OP_SUB: compute_func = compute_row_bcast_sub_f16; भङ्ग;
                    पक्ष HTP_OP_MUL: compute_func = compute_row_bcast_mul_f16; भङ्ग;
                    पक्ष HTP_OP_DIV: compute_func = compute_row_bcast_div_f16; भङ्ग;
                    स्वभाव: भङ्ग;
                }
            }
            भङ्ग;
        पक्ष HTP_द्विमान_KERNEL_SCALAR:
            कार्यकर्ता_func = द्विमान_thread_scalar;
            यदि (src0_type == HTP_TYPE_F32) {
                विकल्प (octx->op) {
                    पक्ष HTP_OP_सङ्कलय: compute_func = compute_scalar_सङ्कलय_f32; भङ्ग;
                    पक्ष HTP_OP_SUB: compute_func = compute_scalar_sub_f32; भङ्ग;
                    पक्ष HTP_OP_MUL: compute_func = compute_scalar_mul_f32; भङ्ग;
                    पक्ष HTP_OP_DIV: compute_func = compute_scalar_div_f32; भङ्ग;
                    स्वभाव: भङ्ग;
                }
            } अन्यथा यदि (src0_type == HTP_TYPE_F16) {
                विकल्प (octx->op) {
                    पक्ष HTP_OP_सङ्कलय: compute_func = compute_scalar_सङ्कलय_f16; भङ्ग;
                    पक्ष HTP_OP_SUB: compute_func = compute_scalar_sub_f16; भङ्ग;
                    पक्ष HTP_OP_MUL: compute_func = compute_scalar_mul_f16; भङ्ग;
                    पक्ष HTP_OP_DIV: compute_func = compute_scalar_div_f16; भङ्ग;
                    स्वभाव: भङ्ग;
                }
            }
            भङ्ग;
        पक्ष HTP_द्विमान_KERNEL_COMPLEX:
            कार्यकर्ता_func = द्विमान_thread_सदिशम्_complex;
            यदि (src0_type == HTP_TYPE_F32) {
                विकल्प (octx->op) {
                    पक्ष HTP_OP_सङ्कलय: compute_func = compute_complex_सङ्कलय_f32; भङ्ग;
                    पक्ष HTP_OP_SUB: compute_func = compute_complex_sub_f32; भङ्ग;
                    पक्ष HTP_OP_MUL: compute_func = compute_complex_mul_f32; भङ्ग;
                    पक्ष HTP_OP_DIV: compute_func = compute_complex_div_f32; भङ्ग;
                    स्वभाव: भङ्ग;
                }
            } अन्यथा यदि (src0_type == HTP_TYPE_F16) {
                विकल्प (octx->op) {
                    पक्ष HTP_OP_सङ्कलय: compute_func = compute_complex_सङ्कलय_f16; भङ्ग;
                    पक्ष HTP_OP_SUB: compute_func = compute_complex_sub_f16; भङ्ग;
                    पक्ष HTP_OP_MUL: compute_func = compute_complex_mul_f16; भङ्ग;
                    पक्ष HTP_OP_DIV: compute_func = compute_complex_div_f16; भङ्ग;
                    स्वभाव: भङ्ग;
                }
            }
            भङ्ग;
        पक्ष HTP_द्विमान_KERNEL_REPEAT:
            कार्यकर्ता_func = द्विमान_thread_element_repeat;
            यदि (src0_type == HTP_TYPE_F32) {
                विकल्प (octx->op) {
                    पक्ष HTP_OP_सङ्कलय: compute_func = compute_repeat_सङ्कलय_f32; भङ्ग;
                    पक्ष HTP_OP_SUB: compute_func = compute_repeat_sub_f32; भङ्ग;
                    पक्ष HTP_OP_MUL: compute_func = compute_repeat_mul_f32; भङ्ग;
                    पक्ष HTP_OP_DIV: compute_func = compute_repeat_div_f32; भङ्ग;
                    स्वभाव: भङ्ग;
                }
            } अन्यथा यदि (src0_type == HTP_TYPE_F16) {
                विकल्प (octx->op) {
                    पक्ष HTP_OP_सङ्कलय: compute_func = compute_repeat_सङ्कलय_f16; भङ्ग;
                    पक्ष HTP_OP_SUB: compute_func = compute_repeat_sub_f16; भङ्ग;
                    पक्ष HTP_OP_MUL: compute_func = compute_repeat_mul_f16; भङ्ग;
                    पक्ष HTP_OP_DIV: compute_func = compute_repeat_div_f16; भङ्ग;
                    स्वभाव: भङ्ग;
                }
            }
            भङ्ग;
        पक्ष HTP_द्विमान_KERNEL_सङ्कलय_ID:
            यदि (octx->op == HTP_OP_सङ्कलय_ID && src0_type == HTP_TYPE_F32) {
                कार्यकर्ता_func = द्विमान_thread_सङ्कलय_id_f32;
                compute_func = (शून्य *) compute_सङ्कलय_id_f32;
            }
            भङ्ग;
        स्वभाव: भङ्ग;
    }

    यदि (!कार्यकर्ता_func || !compute_func) {
        प्रत्यागम HTP_स्थितिः_NO_SUPPORT;
    }

    bctx.compute = compute_func;

    यदि (kparams->kernel_type == HTP_द्विमान_KERNEL_ROW_BCAST ||
        kparams->kernel_type == HTP_द्विमान_KERNEL_SCALAR ||
        kparams->kernel_type == HTP_द्विमान_KERNEL_REPEAT ||
        kparams->kernel_type == HTP_द्विमान_KERNEL_COMPLEX ||
        kparams->kernel_type == HTP_द्विमान_KERNEL_सङ्कलय_ID) {
        dma_queue_flush(dma_q);
    }

    work_queue_सततम्(octx->ctx->work_queue, कार्यकर्ता_func, &bctx, n_तन्तवः);

    प्रत्यागम HTP_स्थितिः_OK;
}

पूर्णाङ्क op_द्विमान(संरचना htp_ops_context * octx) {
    ध्रुव uint32_t src0_type = octx->src[0]->type;
    यदि ((src0_type == HTP_TYPE_F32) || (src0_type == HTP_TYPE_F16)) {
        प्रत्यागम execute_op_द्विमान(octx);
    }

    प्रत्यागम HTP_स्थितिः_NO_SUPPORT;
}
