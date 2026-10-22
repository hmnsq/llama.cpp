#pragma clang diagnostic ignored "-Wunused-variable"
#pragma clang diagnostic ignored "-Wunused-function"
#pragma clang diagnostic ignored "-Wunused-but-set-variable"

#अन्तर्भाव <HAP_farf.h>
#अन्तर्भाव <HAP_perf.h>

#अन्तर्भाव <math.h>
#अन्तर्भाव <string.h>

#अन्तर्भाव "dma-queue.h"
#अन्तर्भाव "hvx-utils.h"

#define GGML_COMMON_DECL_C
#अन्तर्भाव "ggml-common.h"
#अन्तर्भाव "hex-common.h"
#अन्तर्भाव "hex-profile.h"
#अन्तर्भाव "binary-ops.h"
#अन्तर्भाव "htp-ctx.h"
#अन्तर्भाव "htp-ops.h"
#अन्तर्भाव "htp-tensor.h"

#ifndef MIN
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#endif

// Context for binary operations
संरचना htp_द्विआधारी_context {
    संरचना htp_ops_context * octx;
    संरचना htp_द्विआधारी_vtcm_layout vtcm_layout;
    uint8_t * vtcm_base;

    संरचना fastdiv_मानहरू src0_dim1_div; // ne01
    संरचना fastdiv_मानहरू src0_dim2_div; // ne02
    संरचना fastdiv_मानहरू src0_dim12_div;// ne03

    संरचना fastdiv_मानहरू src1_dim1_div; // ne11
    संरचना fastdiv_मानहरू src1_dim2_div; // ne12
    संरचना fastdiv_मानहरू src1_dim3_div; // ne13

    uint32_t block_max;
    uint32_t nrows_per_thread;
    uint32_t total_rows;
    uint32_t row_start;
    माप_प्रकार   src0_row_size_aligned;
    माप_प्रकार   src1_row_size_aligned;
    माप_प्रकार   dst_row_size_aligned;
    माप_प्रकार   row_size_bytes;

    bool split_at_ne01;
    bool split_at_ne02;

    शून्य * compute;
};

#define htp_द्विआधारी_preamble                        \
    अचर संरचना htp_tensor * src0 = octx->src[0]; \
    अचर संरचना htp_tensor * src1 = octx->src[1]; \
    अचर संरचना htp_tensor * dst  = octx->dst;    \
                                                   \
    अचर uint32_t ne00 = src0->ne[0];             \
    अचर uint32_t ne01 = src0->ne[1];             \
    अचर uint32_t ne02 = src0->ne[2];             \
    अचर uint32_t ne03 = src0->ne[3];             \
                                                   \
    अचर uint32_t ne10 = src1->ne[0];             \
    अचर uint32_t ne11 = src1->ne[1];             \
    अचर uint32_t ne12 = src1->ne[2];             \
    अचर uint32_t ne13 = src1->ne[3];             \
                                                   \
    अचर uint32_t nb01 = src0->nb[1];             \
    अचर uint32_t nb02 = src0->nb[2];             \
    अचर uint32_t nb03 = src0->nb[3];             \
                                                   \
    अचर uint32_t nb11 = src1->nb[1];             \
    अचर uint32_t nb12 = src1->nb[2];             \
    अचर uint32_t nb13 = src1->nb[3];             \
                                                   \
    अचर uint32_t nb1 = dst->nb[1];               \
    अचर uint32_t nb2 = dst->nb[2];               \
    अचर uint32_t nb3 = dst->nb[3];

स्थिर अन्तःस्थ uint32_t calc_block_size(संरचना htp_द्विआधारी_context * bctx, uint32_t ir, uint32_t end_row, uint32_t ne01, uint32_t ne02) {
    uint32_t i03, i02, i01, rem;
    i03 = fastdiv(ir, &bctx->src0_dim12_div);
    rem = ir - i03 * (ne02 * ne01);
    i02 = fastdiv(rem, &bctx->src0_dim1_div);
    i01 = rem - i02 * ne01;

    uint32_t rows_बायाँ = end_row - ir;
    uint32_t block_limit = rows_बायाँ;

    यदि (bctx->split_at_ne01) {
        block_limit = MIN(block_limit, ne01 - i01);
    }
    यदि (bctx->split_at_ne02) {
         uint32_t rows_in_plane = (ne02 * ne01) - rem;
         block_limit = MIN(block_limit, rows_in_plane);
    }

    फर्काऊ MIN(bctx->block_max, block_limit);
}

// Out-of-line compute micro-kernels

प्रकारनिर्देश शून्य (*compute_scalar_t)(
    uint8_t * dst, अचर uint8_t * src0, अचर शून्य * s1_table,
    uint32_t cur_i11, uint32_t ne11, uint32_t n_rows,
    माप_प्रकार dst_stride, माप_प्रकार src0_stride, uint32_t ne00);

#define DEFINE_COMPUTE_SCALAR(NAME, TYPE, HVX_STMT)             \
स्थिर शून्य compute_scalar_##NAME(                              \
    uint8_t * dst, अचर uint8_t * src0, अचर शून्य * s1_table, \
    uint32_t cur_i11, uint32_t ne11, uint32_t n_rows,           \
    माप_प्रकार dst_stride, माप_प्रकार src0_stride, uint32_t ne00) {     \
    अचर TYPE * table = (अचर TYPE *) s1_table;               \
    लागि (uint32_t r = 0; r < n_rows; r++) {                     \
        uint8_t * r_dst = dst + r * dst_stride;                 \
        अचर uint8_t * r_src0 = src0 + r * src0_stride;        \
        TYPE val = table[cur_i11];                              \
        HVX_STMT;                                               \
        यदि (ne11 > 1 && ++cur_i11 == ne11) {                    \
            cur_i11 = 0;                                        \
        }                                                       \
    }                                                           \
}

DEFINE_COMPUTE_SCALAR(add_f32, भिन्न,    hvx_add_scalar_f32_aa(r_dst, r_src0, val, ne00))
DEFINE_COMPUTE_SCALAR(add_f16, _Float16, hvx_add_scalar_f16_aa(r_dst, r_src0, val, ne00))
DEFINE_COMPUTE_SCALAR(sub_f32, भिन्न,    hvx_sub_scalar_f32_aa(r_dst, r_src0, val, ne00))
DEFINE_COMPUTE_SCALAR(sub_f16, _Float16, hvx_sub_scalar_f16_aa(r_dst, r_src0, val, ne00))
DEFINE_COMPUTE_SCALAR(mul_f32, भिन्न,    hvx_mul_scalar_f32_aa(r_dst, r_src0, val, ne00))
DEFINE_COMPUTE_SCALAR(mul_f16, _Float16, hvx_mul_scalar_f16_aa(r_dst, r_src0, val, ne00))
DEFINE_COMPUTE_SCALAR(div_f32, भिन्न,    hvx_mul_scalar_f32_aa(r_dst, r_src0, 1.0f / (val), ne00))
DEFINE_COMPUTE_SCALAR(div_f16, _Float16, hvx_div_scalar_f16_aa(r_dst, r_src0, val, ne00))

प्रकारनिर्देश शून्य (*compute_same_shape_t)(
    uint8_t * dst, अचर uint8_t * src0, अचर uint8_t * src1,
    uint32_t n_rows, माप_प्रकार dst_stride, माप_प्रकार src0_stride, माप_प्रकार src1_stride, uint32_t ne00);

#define DEFINE_COMPUTE_SAME_SHAPE(NAME, HVX_FN)                                                  \
स्थिर शून्य compute_same_shape_##NAME(                                                           \
    uint8_t * dst, अचर uint8_t * src0, अचर uint8_t * src1,                                   \
    uint32_t n_rows, माप_प्रकार dst_stride, माप_प्रकार src0_stride, माप_प्रकार src1_stride, uint32_t ne00) { \
    लागि (uint32_t r = 0; r < n_rows; r++) {                                                      \
        HVX_FN(dst + r * dst_stride, src0 + r * src0_stride, src1 + r * src1_stride, ne00);      \
    }                                                                                            \
}

DEFINE_COMPUTE_SAME_SHAPE(add_f32, hvx_add_f32_aaa)
DEFINE_COMPUTE_SAME_SHAPE(add_f16, hvx_add_f16_aaa)
DEFINE_COMPUTE_SAME_SHAPE(sub_f32, hvx_sub_f32_aaa)
DEFINE_COMPUTE_SAME_SHAPE(sub_f16, hvx_sub_f16_aaa)
DEFINE_COMPUTE_SAME_SHAPE(mul_f32, hvx_mul_f32_aaa)
DEFINE_COMPUTE_SAME_SHAPE(mul_f16, hvx_mul_f16_aaa)
DEFINE_COMPUTE_SAME_SHAPE(div_f32, hvx_div_f32_aaa)
DEFINE_COMPUTE_SAME_SHAPE(div_f16, hvx_div_f16_aaa)

प्रकारनिर्देश शून्य (*compute_row_bcast_t)(
    uint8_t * dst, अचर uint8_t * src0, अचर uint8_t * src1,
    uint32_t n_rows, माप_प्रकार dst_stride, माप_प्रकार src0_stride, uint32_t ne00);

#define DEFINE_COMPUTE_ROW_BCAST(NAME, HVX_FN)                                               \
स्थिर शून्य compute_row_bcast_##NAME(                                                        \
    uint8_t * dst, अचर uint8_t * src0, अचर uint8_t * src1,                               \
    uint32_t n_rows, माप_प्रकार dst_stride, माप_प्रकार src0_stride, uint32_t ne00) {                 \
    लागि (uint32_t r = 0; r < n_rows; r++) {                                                  \
        HVX_FN(dst + r * dst_stride, src0 + r * src0_stride, src1, ne00);                    \
    }                                                                                        \
}

DEFINE_COMPUTE_ROW_BCAST(add_f32, hvx_add_f32_aaa)
DEFINE_COMPUTE_ROW_BCAST(add_f16, hvx_add_f16_aaa)
DEFINE_COMPUTE_ROW_BCAST(sub_f32, hvx_sub_f32_aaa)
DEFINE_COMPUTE_ROW_BCAST(sub_f16, hvx_sub_f16_aaa)
DEFINE_COMPUTE_ROW_BCAST(mul_f32, hvx_mul_f32_aaa)
DEFINE_COMPUTE_ROW_BCAST(mul_f16, hvx_mul_f16_aaa)
DEFINE_COMPUTE_ROW_BCAST(div_f32, hvx_div_f32_aaa)
DEFINE_COMPUTE_ROW_BCAST(div_f16, hvx_div_f16_aaa)

प्रकारनिर्देश शून्य (*compute_complex_t)(
    uint8_t * dst, अचर uint8_t * src0, अचर uint8_t * src1_plane,
    uint32_t i01, uint32_t ne11, अचर संरचना fastdiv_मानहरू * div11, uint32_t src1_stride,
    uint32_t n_rows, माप_प्रकार dst_stride, माप_प्रकार src0_stride, uint32_t ne00);

#define DEFINE_COMPUTE_COMPLEX(NAME, HVX_FN)                                                 \
स्थिर शून्य compute_complex_##NAME(                                                          \
    uint8_t * dst, अचर uint8_t * src0, अचर uint8_t * src1_plane,                         \
    uint32_t i01, uint32_t ne11, अचर संरचना fastdiv_मानहरू * div11, uint32_t src1_stride,  \
    uint32_t n_rows, माप_प्रकार dst_stride, माप_प्रकार src0_stride, uint32_t ne00) {                 \
    लागि (uint32_t r = 0; r < n_rows; r++) {                                                  \
        uint32_t i11 = fastmodulo(i01 + r, ne11, div11);                                     \
        अचर uint8_t * r_src1 = src1_plane + i11 * src1_stride;                             \
        HVX_FN(dst + r * dst_stride, src0 + r * src0_stride, r_src1, ne00);                  \
    }                                                                                        \
}

DEFINE_COMPUTE_COMPLEX(add_f32, hvx_add_f32_aau)
DEFINE_COMPUTE_COMPLEX(add_f16, hvx_add_f16_aau)
DEFINE_COMPUTE_COMPLEX(sub_f32, hvx_sub_f32_aau)
DEFINE_COMPUTE_COMPLEX(sub_f16, hvx_sub_f16_aau)
DEFINE_COMPUTE_COMPLEX(mul_f32, hvx_mul_f32_aau)
DEFINE_COMPUTE_COMPLEX(mul_f16, hvx_mul_f16_aau)
DEFINE_COMPUTE_COMPLEX(div_f32, hvx_div_f32_aau)
DEFINE_COMPUTE_COMPLEX(div_f16, hvx_div_f16_aau)

प्रकारनिर्देश शून्य (*compute_repeat_t)(
    uint8_t * dst, अचर uint8_t * src0, अचर uint8_t * src1_plane,
    uint32_t i01, uint32_t ne11, अचर संरचना fastdiv_मानहरू * div11, uint32_t src1_stride,
    uint32_t n_rows, माप_प्रकार dst_stride, माप_प्रकार src0_stride, uint32_t ne00, uint32_t ne10);

#define DEFINE_COMPUTE_REPEAT(NAME, TYPE, HVX_FN)                                            \
स्थिर शून्य compute_repeat_##NAME(                                                           \
    uint8_t * dst, अचर uint8_t * src0, अचर uint8_t * src1_plane,                         \
    uint32_t i01, uint32_t ne11, अचर संरचना fastdiv_मानहरू * div11, uint32_t src1_stride,  \
    uint32_t n_rows, माप_प्रकार dst_stride, माप_प्रकार src0_stride, uint32_t ne00, uint32_t ne10) {  \
    लागि (uint32_t r = 0; r < n_rows; r++) {                                                  \
        uint32_t i11 = fastmodulo(i01 + r, ne11, div11);                                     \
        अचर uint8_t * r_src1_row = src1_plane + i11 * src1_stride;                         \
        uint8_t * r_dst = dst + r * dst_stride;                                              \
        अचर uint8_t * r_src0 = src0 + r * src0_stride;                                     \
        लागि (uint32_t c = 0; c < ne00; c += ne10) {                                          \
            uint32_t len = MIN(ne10, ne00 - c);                                              \
            HVX_FN(r_dst + c * परिमाण(TYPE), r_src0 + c * परिमाण(TYPE), r_src1_row, len);    \
        }                                                                                    \
    }                                                                                        \
}

DEFINE_COMPUTE_REPEAT(add_f32, भिन्न,    hvx_add_f32_uuu)
DEFINE_COMPUTE_REPEAT(add_f16, _Float16, hvx_add_f16_uuu)
DEFINE_COMPUTE_REPEAT(sub_f32, भिन्न,    hvx_sub_f32_uuu)
DEFINE_COMPUTE_REPEAT(sub_f16, _Float16, hvx_sub_f16_uuu)
DEFINE_COMPUTE_REPEAT(mul_f32, भिन्न,    hvx_mul_f32_uuu)
DEFINE_COMPUTE_REPEAT(mul_f16, _Float16, hvx_mul_f16_uuu)
DEFINE_COMPUTE_REPEAT(div_f32, भिन्न,    hvx_div_f32_uuu)
DEFINE_COMPUTE_REPEAT(div_f16, _Float16, hvx_div_f16_uuu)

प्रकारनिर्देश शून्य (*compute_add_id_t)(
    uint8_t * dst, अचर uint8_t * src0, अचर uint8_t * src1_data, अचर int32_t * src2_ids,
    uint32_t i01, uint32_t i02, uint32_t ne01, uint32_t src1_stride,
    uint32_t n_rows, माप_प्रकार dst_stride, माप_प्रकार src0_stride, uint32_t ne00);

स्थिर शून्य compute_add_id_f32(
    uint8_t * dst, अचर uint8_t * src0, अचर uint8_t * src1_data, अचर int32_t * src2_ids,
    uint32_t i01, uint32_t i02, uint32_t ne01, uint32_t src1_stride,
    uint32_t n_rows, माप_प्रकार dst_stride, माप_प्रकार src0_stride, uint32_t ne00) {
    लागि (uint32_t r = 0; r < n_rows; r++) {
        uint32_t r_i01 = i01 + r;
        अचर int32_t idx = src2_ids[i02 * ne01 + r_i01];
        यदि (idx < 0) {
            स_नकल(dst + r * dst_stride, src0 + r * src0_stride, ne00 * परिमाण(भिन्न));
            जारी;
        }
        अचर uint8_t * r_src1 = src1_data + idx * src1_stride;
        अचर uint8_t * r_src0 = src0 + r * src0_stride;
        uint8_t * r_dst = dst + r * dst_stride;
        hvx_add_f32_aaa(r_dst, r_src0, r_src1, ne00);
    }
}

// 1. Scalar src1 in VTCM via DMA (ne10 == 1)
स्थिर शून्य द्विआधारी_thread_scalar(अचिन्हित पूर्णाङ्क nth, अचिन्हित पूर्णाङ्क ith, शून्य * data) {
    संरचना htp_द्विआधारी_context * bctx = (संरचना htp_द्विआधारी_context *) data;
    संरचना htp_ops_context * octx = bctx->octx;
    htp_द्विआधारी_preamble;

    अचर uint32_t elem_size = (src0->type == HTP_TYPE_F32) ? परिमाण(भिन्न) : परिमाण(_Float16);
    अचर माप_प्रकार plane_stride = (माप_प्रकार) ne12 * ne11 * elem_size;
    अचर माप_प्रकार row_stride   = (माप_प्रकार) ne11 * elem_size;

    अचर uint32_t row_size_bytes = bctx->row_size_bytes;
    अचर uint32_t start_row = bctx->row_start + bctx->nrows_per_thread * ith;
    अचर uint32_t end_row   = MIN(start_row + bctx->nrows_per_thread, bctx->row_start + bctx->total_rows);
    यदि (start_row >= end_row) फर्काऊ;

    FARF(माथिल्लो_2, "binary-scalar: %d/%d (%u:%u) row-size %u (%u)",
         ith, nth, start_row, end_row, nb01, bctx->dst_row_size_aligned);

    अचर संरचना htp_द्विआधारी_vtcm_layout * layout = &bctx->vtcm_layout;
    uint8_t * src0_spad_base = VTCM_LAYOUT_PTR(uint8_t, bctx->vtcm_base, layout->off_src0) + (ith * layout->src0_bytes_per_thread);
    uint8_t * dst_spad_base  = VTCM_LAYOUT_PTR(uint8_t, bctx->vtcm_base, layout->off_dst)  + (ith * layout->dst_bytes_per_thread);
    माप_प्रकार src0_spad_half    = layout->src0_spad_half_size;
    माप_प्रकार dst_spad_half     = layout->dst_spad_half_size;
    अचर शून्य * s1_table    = VTCM_LAYOUT_PTR(अचर शून्य, bctx->vtcm_base, layout->off_src1);

    dma_queue * dma_q = octx->ctx->dma[ith];
    uint32_t ir_prefetch = start_row;
    पूर्णाङ्क spad_idx = 0;

    लागि (पूर्णाङ्क k = 0; k < 2 && ir_prefetch < end_row; k++) {
        uint32_t current_block_size = calc_block_size(bctx, ir_prefetch, end_row, ne01, ne02);
        uint32_t i03, i02, i01, rem;
        i03 = fastdiv(ir_prefetch, &bctx->src0_dim12_div);
        rem = ir_prefetch - i03 * (ne02 * ne01);
        i02 = fastdiv(rem, &bctx->src0_dim1_div);
        i01 = rem - i02 * ne01;

        dma_addr_t src0_curr = src0->data + i03 * nb03 + i02 * nb02 + i01 * nb01;
        dma_addr_t dst_curr  = dst->data  + i03 * nb3  + i02 * nb2  + i01 * nb1;

        uint8_t * s0_spad = src0_spad_base + spad_idx * src0_spad_half;
        uint8_t * d_spad  = dst_spad_base  + spad_idx * dst_spad_half;

        dma_queue_थप(dma_q, dma_make_data(dst_curr, d_spad), nb1, bctx->dst_row_size_aligned, row_size_bytes, 0);
        dma_queue_थप(dma_q, dma_make_data(s0_spad, src0_curr), bctx->src0_row_size_aligned, nb01, row_size_bytes,
                                                             current_block_size);
        ir_prefetch += current_block_size;
        spad_idx ^= 1;
    }

    संरचना htp_thread_trace * tr = &octx->ctx->trace[ith];
    compute_scalar_t compute = (compute_scalar_t) bctx->compute;

    लागि (uint32_t ir = start_row; ir < end_row; ) {
        uint32_t current_block_size = calc_block_size(bctx, ir, end_row, ne01, ne02);

        uint8_t * d_spad  = (uint8_t *) dma_queue_झिक(dma_q).src;
        uint8_t * s0_spad = (uint8_t *) dma_queue_झिक(dma_q).dst;

        uint32_t i03, i02, i01, rem;
        i03 = fastdiv(ir, &bctx->src0_dim12_div);
        rem = ir - i03 * (ne02 * ne01);
        i02 = fastdiv(rem, &bctx->src0_dim1_div);
        i01 = rem - i02 * ne01;

        uint32_t i13 = fastmodulo(i03, ne13, &bctx->src1_dim3_div);
        uint32_t i12 = fastmodulo(i02, ne12, &bctx->src1_dim2_div);
        uint32_t i11 = fastmodulo(i01, ne11, &bctx->src1_dim1_div);
        अचर uint8_t * scalar_table = (अचर uint8_t *) s1_table +
            i13 * plane_stride + i12 * row_stride;

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);
        compute(d_spad, s0_spad, scalar_table, i11, ne11, current_block_size,
                bctx->dst_row_size_aligned, bctx->src0_row_size_aligned, ne00);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);

        dma_addr_t dst_curr = dst->data + i03 * nb3 + i02 * nb2 + i01 * nb1;
        dma_queue_थप(dma_q, dma_make_data(dst_curr, d_spad), nb1, bctx->dst_row_size_aligned, row_size_bytes,
                                                                current_block_size);

        यदि (ir_prefetch < end_row) {
            uint32_t next_block_size = calc_block_size(bctx, ir_prefetch, end_row, ne01, ne02);
            uint32_t p03, p02, p01, prem;
            p03 = fastdiv(ir_prefetch, &bctx->src0_dim12_div);
            prem = ir_prefetch - p03 * (ne02 * ne01);
            p02 = fastdiv(prem, &bctx->src0_dim1_div);
            p01 = prem - p02 * ne01;
            dma_addr_t s0_next = src0->data + p03 * nb03 + p02 * nb02 + p01 * nb01;
            dma_queue_थप(dma_q, dma_make_data(s0_spad, s0_next), bctx->src0_row_size_aligned, nb01, row_size_bytes,
                                                               next_block_size);
            ir_prefetch += next_block_size;
        }
        ir += current_block_size;
    }

    dma_queue_flush(dma_q);
}

// 2. Vector Same Shape (ne1x == ne0x) or Simple Broadcast
स्थिर शून्य द्विआधारी_thread_vector_same_shape(अचिन्हित पूर्णाङ्क nth, अचिन्हित पूर्णाङ्क ith, शून्य * data) {
    संरचना htp_द्विआधारी_context * bctx = (संरचना htp_द्विआधारी_context *) data;
    संरचना htp_ops_context * octx = bctx->octx;
    htp_द्विआधारी_preamble;

    अचर uint32_t row_size_bytes = bctx->row_size_bytes;
    अचर uint32_t start_row = bctx->row_start + bctx->nrows_per_thread * ith;
    अचर uint32_t end_row   = MIN(start_row + bctx->nrows_per_thread, bctx->row_start + bctx->total_rows);
    यदि (start_row >= end_row) फर्काऊ;

    FARF(माथिल्लो_2, "binary-same-shape: %d/%d (%u:%u) row-size %u (%u)",
         ith, nth, start_row, end_row, nb01, bctx->dst_row_size_aligned);

    अचर संरचना htp_द्विआधारी_vtcm_layout * layout = &bctx->vtcm_layout;
    uint8_t * src0_spad_base = VTCM_LAYOUT_PTR(uint8_t, bctx->vtcm_base, layout->off_src0) + (ith * layout->src0_bytes_per_thread);
    uint8_t * src1_spad_base = VTCM_LAYOUT_PTR(uint8_t, bctx->vtcm_base, layout->off_src1) + (ith * layout->src1_bytes_per_thread);
    uint8_t * dst_spad_base  = VTCM_LAYOUT_PTR(uint8_t, bctx->vtcm_base, layout->off_dst)  + (ith * layout->dst_bytes_per_thread);

    माप_प्रकार src0_spad_half = layout->src0_spad_half_size;
    माप_प्रकार src1_spad_half = layout->src1_spad_half_size;
    माप_प्रकार dst_spad_half  = layout->dst_spad_half_size;

    dma_queue * dma_q = octx->ctx->dma[ith];
    uint32_t ir_prefetch = start_row;
    पूर्णाङ्क spad_idx = 0;

    लागि (पूर्णाङ्क k = 0; k < 2 && ir_prefetch < end_row; k++) {
        uint32_t current_block_size = calc_block_size(bctx, ir_prefetch, end_row, ne01, ne02);
        uint32_t i03, i02, i01, rem;
        i03 = fastdiv(ir_prefetch, &bctx->src0_dim12_div);
        rem = ir_prefetch - i03 * (ne02 * ne01);
        i02 = fastdiv(rem, &bctx->src0_dim1_div);
        i01 = rem - i02 * ne01;

        uint32_t i13 = (ne13 == 1) ? 0 : i03;
        uint32_t i12 = (ne12 == 1) ? 0 : i02;
        uint32_t i11 = (ne11 == 1) ? 0 : i01;

        dma_addr_t src0_curr = src0->data + i03 * nb03 + i02 * nb02 + i01 * nb01;
        dma_addr_t src1_curr = src1->data + i13 * nb13 + i12 * nb12 + i11 * nb11;
        dma_addr_t dst_curr  = dst->data  + i03 * nb3  + i02 * nb2  + i01 * nb1;

        uint8_t * s0_spad = src0_spad_base + spad_idx * src0_spad_half;
        uint8_t * s1_spad = src1_spad_base + spad_idx * src1_spad_half;
        uint8_t * d_spad  = dst_spad_base  + spad_idx * dst_spad_half;

        dma_queue_थप(dma_q, dma_make_data(dst_curr, d_spad), nb1, bctx->dst_row_size_aligned, row_size_bytes, 0);
        dma_queue_थप(dma_q, dma_make_data(s0_spad, src0_curr), bctx->src0_row_size_aligned, nb01, row_size_bytes, current_block_size);
        dma_queue_थप(dma_q, dma_make_data(s1_spad, src1_curr), bctx->src1_row_size_aligned, nb11, row_size_bytes, current_block_size);
        ir_prefetch += current_block_size;
        spad_idx ^= 1;
    }

    संरचना htp_thread_trace * tr = &octx->ctx->trace[ith];
    compute_same_shape_t compute = (compute_same_shape_t) bctx->compute;

    लागि (uint32_t ir = start_row; ir < end_row; ) {
        uint32_t current_block_size = calc_block_size(bctx, ir, end_row, ne01, ne02);
        uint8_t * d_spad  = (uint8_t *) dma_queue_झिक(dma_q).src;
        uint8_t * s0_spad = (uint8_t *) dma_queue_झिक(dma_q).dst;
        uint8_t * s1_spad = (uint8_t *) dma_queue_झिक(dma_q).dst;

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);
        compute(d_spad, s0_spad, s1_spad, current_block_size,
                bctx->dst_row_size_aligned, bctx->src0_row_size_aligned, bctx->src1_row_size_aligned, ne00);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);

        uint32_t i03, i02, i01, rem;
        i03 = fastdiv(ir, &bctx->src0_dim12_div);
        rem = ir - i03 * (ne02 * ne01);
        i02 = fastdiv(rem, &bctx->src0_dim1_div);
        i01 = rem - i02 * ne01;
        dma_addr_t dst_curr = dst->data + i03 * nb3 + i02 * nb2 + i01 * nb1;
        dma_queue_थप(dma_q, dma_make_data(dst_curr, d_spad), nb1, bctx->dst_row_size_aligned, row_size_bytes, current_block_size);

        यदि (ir_prefetch < end_row) {
            uint32_t next_block_size = calc_block_size(bctx, ir_prefetch, end_row, ne01, ne02);
            uint32_t p03, p02, p01, prem;
            p03 = fastdiv(ir_prefetch, &bctx->src0_dim12_div);
            prem = ir_prefetch - p03 * (ne02 * ne01);
            p02 = fastdiv(prem, &bctx->src0_dim1_div);
            p01 = prem - p02 * ne01;

            uint32_t p13 = (ne13 == 1) ? 0 : p03;
            uint32_t p12 = (ne12 == 1) ? 0 : p02;
            uint32_t p11 = (ne11 == 1) ? 0 : p01;

            dma_addr_t s0_next = src0->data + p03 * nb03 + p02 * nb02 + p01 * nb01;
            dma_addr_t s1_next = src1->data + p13 * nb13 + p12 * nb12 + p11 * nb11;

            dma_queue_थप(dma_q, dma_make_data(s0_spad, s0_next), bctx->src0_row_size_aligned, nb01, row_size_bytes, next_block_size);
            dma_queue_थप(dma_q, dma_make_data(s1_spad, s1_next), bctx->src1_row_size_aligned, nb11, row_size_bytes, next_block_size);

            ir_prefetch += next_block_size;
        }
        ir += current_block_size;
    }

    dma_queue_flush(dma_q);
}

// 3. Row Broadcast (ne11 == 1, ne12 == 1, single row src1)
स्थिर शून्य द्विआधारी_thread_vector_row_broadcast(अचिन्हित पूर्णाङ्क nth, अचिन्हित पूर्णाङ्क ith, शून्य * data) {
    संरचना htp_द्विआधारी_context * bctx = (संरचना htp_द्विआधारी_context *) data;
    संरचना htp_ops_context * octx = bctx->octx;
    htp_द्विआधारी_preamble;

    अचर uint32_t row_size_bytes = bctx->row_size_bytes;
    अचर uint32_t start_row  = bctx->row_start + bctx->nrows_per_thread * ith;
    अचर uint32_t end_row    = MIN(start_row + bctx->nrows_per_thread, bctx->row_start + bctx->total_rows);
    यदि (start_row >= end_row) फर्काऊ;

    FARF(माथिल्लो_2, "binary-row-bcast: %d/%d (%u:%u) row-size %u (%u)",
         ith, nth, start_row, end_row, nb01, bctx->dst_row_size_aligned);

    अचर संरचना htp_द्विआधारी_vtcm_layout * layout = &bctx->vtcm_layout;
    uint8_t * src0_spad_base = VTCM_LAYOUT_PTR(uint8_t, bctx->vtcm_base, layout->off_src0) + (ith * layout->src0_bytes_per_thread);
    uint8_t * dst_spad_base  = VTCM_LAYOUT_PTR(uint8_t, bctx->vtcm_base, layout->off_dst)  + (ith * layout->dst_bytes_per_thread);

    माप_प्रकार src0_spad_half = layout->src0_spad_half_size;
    माप_प्रकार dst_spad_half  = layout->dst_spad_half_size;

    dma_queue * dma_q = octx->ctx->dma[ith];
    uint32_t ir_prefetch = start_row;
    पूर्णाङ्क spad_idx = 0;

    शून्य * s1_ptr = VTCM_LAYOUT_PTR(शून्य, bctx->vtcm_base, layout->off_src1);

    लागि (पूर्णाङ्क k = 0; k < 2 && ir_prefetch < end_row; k++) {
        uint32_t current_block_size = calc_block_size(bctx, ir_prefetch, end_row, ne01, ne02);
        uint32_t i03 = fastdiv(ir_prefetch, &bctx->src0_dim12_div);
        uint32_t rem = ir_prefetch - i03 * (ne02 * ne01);
        uint32_t i02 = fastdiv(rem, &bctx->src0_dim1_div);
        uint32_t i01 = rem - i02 * ne01;

        dma_addr_t src0_curr = src0->data + i03 * nb03 + i02 * nb02 + i01 * nb01;
        dma_addr_t dst_curr  = dst->data  + i03 * nb3  + i02 * nb2  + i01 * nb1;

        uint8_t * s0_spad = src0_spad_base + spad_idx * src0_spad_half;
        uint8_t * d_spad  = dst_spad_base  + spad_idx * dst_spad_half;

        dma_queue_थप(dma_q, dma_make_data(dst_curr, d_spad), nb1, bctx->dst_row_size_aligned, row_size_bytes, 0);
        dma_queue_थप(dma_q, dma_make_data(s0_spad, src0_curr), bctx->src0_row_size_aligned, nb01, row_size_bytes, current_block_size);
        ir_prefetch += current_block_size;
        spad_idx ^= 1;
    }

    संरचना htp_thread_trace * tr = &octx->ctx->trace[ith];
    compute_row_bcast_t compute = (compute_row_bcast_t) bctx->compute;

    लागि (uint32_t ir = start_row; ir < end_row; ) {
        uint32_t current_block_size = calc_block_size(bctx, ir, end_row, ne01, ne02);
        uint8_t * d_spad  = (uint8_t *) dma_queue_झिक(dma_q).src;
        uint8_t * s0_spad = (uint8_t *) dma_queue_झिक(dma_q).dst;

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);
        compute(d_spad, s0_spad, (अचर uint8_t *)s1_ptr, current_block_size,
                bctx->dst_row_size_aligned, bctx->src0_row_size_aligned, ne00);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);

        uint32_t i03 = fastdiv(ir, &bctx->src0_dim12_div);
        uint32_t rem = ir - i03 * (ne02 * ne01);
        uint32_t i02 = fastdiv(rem, &bctx->src0_dim1_div);
        uint32_t i01 = rem - i02 * ne01;
        dma_addr_t dst_curr = dst->data + i03 * nb3 + i02 * nb2 + i01 * nb1;
        dma_queue_थप(dma_q, dma_make_data(dst_curr, d_spad), nb1, bctx->dst_row_size_aligned, row_size_bytes, current_block_size);

        यदि (ir_prefetch < end_row) {
            uint32_t next_block_size = calc_block_size(bctx, ir_prefetch, end_row, ne01, ne02);
            uint32_t p03, p02, p01, prem;
            p03  = fastdiv(ir_prefetch, &bctx->src0_dim12_div);
            prem = ir_prefetch - p03 * (ne02 * ne01);
            p02  = fastdiv(prem, &bctx->src0_dim1_div);
            p01  = prem - p02 * ne01;
            dma_addr_t s0_next = src0->data + p03 * nb03 + p02 * nb02 + p01 * nb01;
            dma_queue_थप(dma_q, dma_make_data(s0_spad, s0_next), bctx->src0_row_size_aligned, nb01, row_size_bytes, next_block_size);
            ir_prefetch += next_block_size;
        }
        ir += current_block_size;
    }

    dma_queue_flush(dma_q);
}

// 4. Vector Complex (ne10 == ne00, complex broadcast)
स्थिर शून्य द्विआधारी_thread_vector_complex(अचिन्हित पूर्णाङ्क nth, अचिन्हित पूर्णाङ्क ith, शून्य * data) {
    संरचना htp_द्विआधारी_context * bctx = (संरचना htp_द्विआधारी_context *) data;
    संरचना htp_ops_context * octx = bctx->octx;
    htp_द्विआधारी_preamble;

    अचर uint32_t row_size_bytes = bctx->row_size_bytes;
    अचर uint32_t start_row  = bctx->row_start + bctx->nrows_per_thread * ith;
    अचर uint32_t end_row    = MIN(start_row + bctx->nrows_per_thread, bctx->row_start + bctx->total_rows);
    यदि (start_row >= end_row) फर्काऊ;

    FARF(माथिल्लो_2, "binary-complex: %d/%d (%u:%u) row-size %u (%u)",
         ith, nth, start_row, end_row, nb01, bctx->dst_row_size_aligned);

    अचर संरचना htp_द्विआधारी_vtcm_layout * layout = &bctx->vtcm_layout;
    uint8_t * src0_spad_base = VTCM_LAYOUT_PTR(uint8_t, bctx->vtcm_base, layout->off_src0) + (ith * layout->src0_bytes_per_thread);
    uint8_t * dst_spad_base  = VTCM_LAYOUT_PTR(uint8_t, bctx->vtcm_base, layout->off_dst)  + (ith * layout->dst_bytes_per_thread);
    माप_प्रकार src0_spad_half    = layout->src0_spad_half_size;
    माप_प्रकार dst_spad_half     = layout->dst_spad_half_size;
    अचर uint8_t * vtcm_src1 = VTCM_LAYOUT_PTR(अचर uint8_t, bctx->vtcm_base, layout->off_src1);

    अचर uint32_t elem_size = (src0->type == HTP_TYPE_F32) ? परिमाण(भिन्न) : परिमाण(_Float16);
    अचर माप_प्रकार vtcm_row_stride   = (माप_प्रकार) ne10 * elem_size;
    अचर माप_प्रकार vtcm_plane_stride = (माप_प्रकार) ne11 * vtcm_row_stride;
    अचर माप_प्रकार vtcm_vol_stride   = (माप_प्रकार) ne12 * vtcm_plane_stride;

    dma_queue * dma_q = octx->ctx->dma[ith];
    uint32_t ir_prefetch = start_row;
    पूर्णाङ्क spad_idx = 0;

    लागि (पूर्णाङ्क k = 0; k < 2 && ir_prefetch < end_row; k++) {
        uint32_t current_block_size = calc_block_size(bctx, ir_prefetch, end_row, ne01, ne02);
        uint32_t i03 = fastdiv(ir_prefetch, &bctx->src0_dim12_div);
        uint32_t rem = ir_prefetch - i03 * (ne02 * ne01);
        uint32_t i02 = fastdiv(rem, &bctx->src0_dim1_div);
        uint32_t i01 = rem - i02 * ne01;

        dma_addr_t src0_curr = src0->data + i03 * nb03 + i02 * nb02 + i01 * nb01;
        dma_addr_t dst_curr  = dst->data  + i03 * nb3  + i02 * nb2  + i01 * nb1;

        uint8_t * s0_spad = src0_spad_base + spad_idx * src0_spad_half;
        uint8_t * d_spad  = dst_spad_base  + spad_idx * dst_spad_half;

        dma_queue_थप(dma_q, dma_make_data(dst_curr, d_spad), nb1, bctx->dst_row_size_aligned, row_size_bytes, 0);
        dma_queue_थप(dma_q, dma_make_data(s0_spad, src0_curr), bctx->src0_row_size_aligned, nb01, row_size_bytes, current_block_size);
        ir_prefetch += current_block_size;
        spad_idx ^= 1;
    }

    संरचना htp_thread_trace * tr = &octx->ctx->trace[ith];
    compute_complex_t compute = (compute_complex_t) bctx->compute;

    लागि (uint32_t ir = start_row; ir < end_row; ) {
        uint32_t current_block_size = calc_block_size(bctx, ir, end_row, ne01, ne02);
        uint8_t * d_spad = (uint8_t *) dma_queue_झिक(dma_q).src;
        uint8_t * s0_spad = (uint8_t *) dma_queue_झिक(dma_q).dst;

        uint32_t i03 = fastdiv(ir, &bctx->src0_dim12_div);
        uint32_t rem = ir - i03 * (ne02 * ne01);
        uint32_t i02 = fastdiv(rem, &bctx->src0_dim1_div);
        uint32_t i01 = rem - i02 * ne01;

        uint32_t i13 = fastmodulo(i03, ne13, &bctx->src1_dim3_div);
        uint32_t i12 = fastmodulo(i02, ne12, &bctx->src1_dim2_div);
        अचर uint8_t * src1_plane = vtcm_src1 + i13 * vtcm_vol_stride + i12 * vtcm_plane_stride;

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);
        compute(d_spad, s0_spad, src1_plane, i01, ne11, &bctx->src1_dim1_div, vtcm_row_stride,
                current_block_size, bctx->dst_row_size_aligned, bctx->src0_row_size_aligned, ne00);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);

        dma_addr_t dst_curr = dst->data + i03 * nb3 + i02 * nb2 + i01 * nb1;
        dma_queue_थप(dma_q, dma_make_data(dst_curr, d_spad), nb1, bctx->dst_row_size_aligned, row_size_bytes, current_block_size);

        यदि (ir_prefetch < end_row) {
            uint32_t next_block_size = calc_block_size(bctx, ir_prefetch, end_row, ne01, ne02);
            uint32_t p03, p02, p01, prem;
            p03  = fastdiv(ir_prefetch, &bctx->src0_dim12_div);
            prem = ir_prefetch - p03 * (ne02 * ne01);
            p02  = fastdiv(prem, &bctx->src0_dim1_div);
            p01  = prem - p02 * ne01;
            dma_addr_t s0_next = src0->data + p03 * nb03 + p02 * nb02 + p01 * nb01;
            dma_queue_थप(dma_q, dma_make_data(s0_spad, s0_next), bctx->src0_row_size_aligned, nb01, row_size_bytes, next_block_size);
            ir_prefetch += next_block_size;
        }
        ir += current_block_size;
    }

    dma_queue_flush(dma_q);
}

// 5. Element Repeat (ne10 != ne00)
स्थिर शून्य द्विआधारी_thread_element_repeat(अचिन्हित पूर्णाङ्क nth, अचिन्हित पूर्णाङ्क ith, शून्य * data) {
    संरचना htp_द्विआधारी_context * bctx = (संरचना htp_द्विआधारी_context *) data;
    संरचना htp_ops_context * octx = bctx->octx;
    htp_द्विआधारी_preamble;

    अचर uint32_t row_size_bytes = bctx->row_size_bytes;
    अचर uint32_t start_row  = bctx->row_start + bctx->nrows_per_thread * ith;
    अचर uint32_t end_row    = MIN(start_row + bctx->nrows_per_thread, bctx->row_start + bctx->total_rows);
    यदि (start_row >= end_row) फर्काऊ;

    अचर संरचना htp_द्विआधारी_vtcm_layout * layout = &bctx->vtcm_layout;
    uint8_t * src0_spad_base = VTCM_LAYOUT_PTR(uint8_t, bctx->vtcm_base, layout->off_src0) + (ith * layout->src0_bytes_per_thread);
    uint8_t * dst_spad_base  = VTCM_LAYOUT_PTR(uint8_t, bctx->vtcm_base, layout->off_dst)  + (ith * layout->dst_bytes_per_thread);
    माप_प्रकार src0_spad_half    = layout->src0_spad_half_size;
    माप_प्रकार dst_spad_half     = layout->dst_spad_half_size;
    अचर uint8_t * vtcm_src1 = VTCM_LAYOUT_PTR(अचर uint8_t, bctx->vtcm_base, layout->off_src1);

    अचर uint32_t elem_size = (src0->type == HTP_TYPE_F32) ? परिमाण(भिन्न) : परिमाण(_Float16);
    अचर माप_प्रकार vtcm_row_stride   = (माप_प्रकार) ne10 * elem_size;
    अचर माप_प्रकार vtcm_plane_stride = (माप_प्रकार) ne11 * vtcm_row_stride;
    अचर माप_प्रकार vtcm_vol_stride   = (माप_प्रकार) ne12 * vtcm_plane_stride;

    FARF(माथिल्लो_2, "binary-repeat: %d/%d (%u:%u) row-size %u (%u)",
         ith, nth, start_row, end_row, nb01, bctx->dst_row_size_aligned);

    dma_queue * dma_q = octx->ctx->dma[ith];
    uint32_t ir_prefetch = start_row;
    पूर्णाङ्क spad_idx = 0;

    लागि (पूर्णाङ्क k = 0; k < 2 && ir_prefetch < end_row; k++) {
        uint32_t current_block_size = calc_block_size(bctx, ir_prefetch, end_row, ne01, ne02);
        uint32_t i03 = fastdiv(ir_prefetch, &bctx->src0_dim12_div);
        uint32_t rem = ir_prefetch - i03 * (ne02 * ne01);
        uint32_t i02 = fastdiv(rem, &bctx->src0_dim1_div);
        uint32_t i01 = rem - i02 * ne01;

        dma_addr_t src0_curr = src0->data + i03 * nb03 + i02 * nb02 + i01 * nb01;
        dma_addr_t dst_curr  = dst->data  + i03 * nb3  + i02 * nb2  + i01 * nb1;

        uint8_t * s0_spad = src0_spad_base + spad_idx * src0_spad_half;
        uint8_t * d_spad  = dst_spad_base  + spad_idx * dst_spad_half;

        dma_queue_थप(dma_q, dma_make_data(dst_curr, d_spad), nb1, bctx->dst_row_size_aligned, row_size_bytes, 0);
        dma_queue_थप(dma_q, dma_make_data(s0_spad, src0_curr), bctx->src0_row_size_aligned, nb01, row_size_bytes, current_block_size);
        ir_prefetch += current_block_size;
        spad_idx ^= 1;
    }

    संरचना htp_thread_trace * tr = &octx->ctx->trace[ith];
    compute_repeat_t compute = (compute_repeat_t) bctx->compute;

    लागि (uint32_t ir = start_row; ir < end_row; ) {
        uint32_t current_block_size = calc_block_size(bctx, ir, end_row, ne01, ne02);
        uint8_t * d_spad  = (uint8_t *) dma_queue_झिक(dma_q).src;
        uint8_t * s0_spad = (uint8_t *) dma_queue_झिक(dma_q).dst;

        uint32_t i03 = fastdiv(ir, &bctx->src0_dim12_div);
        uint32_t rem = ir - i03 * (ne02 * ne01);
        uint32_t i02 = fastdiv(rem, &bctx->src0_dim1_div);
        uint32_t i01 = rem - i02 * ne01;

        uint32_t i13 = fastmodulo(i03, ne13, &bctx->src1_dim3_div);
        uint32_t i12 = fastmodulo(i02, ne12, &bctx->src1_dim2_div);
        अचर uint8_t * src1_plane = vtcm_src1 + i13 * vtcm_vol_stride + i12 * vtcm_plane_stride;

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);
        compute(d_spad, s0_spad, src1_plane, i01, ne11, &bctx->src1_dim1_div, vtcm_row_stride,
                current_block_size, bctx->dst_row_size_aligned, bctx->src0_row_size_aligned, ne00, ne10);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);

        dma_addr_t dst_curr = dst->data + i03 * nb3 + i02 * nb2 + i01 * nb1;
        dma_queue_थप(dma_q, dma_make_data(dst_curr, d_spad), nb1, bctx->dst_row_size_aligned, row_size_bytes, current_block_size);

        यदि (ir_prefetch < end_row) {
            uint32_t next_block_size = calc_block_size(bctx, ir_prefetch, end_row, ne01, ne02);
            uint32_t p03, p02, p01, prem;
            p03  = fastdiv(ir_prefetch, &bctx->src0_dim12_div);
            prem = ir_prefetch - p03 * (ne02 * ne01);
            p02  = fastdiv(prem, &bctx->src0_dim1_div);
            p01  = prem - p02 * ne01;
            dma_addr_t s0_next = src0->data + p03 * nb03 + p02 * nb02 + p01 * nb01;
            dma_queue_थप(dma_q, dma_make_data(s0_spad, s0_next), bctx->src0_row_size_aligned, nb01, row_size_bytes, next_block_size);
            ir_prefetch += next_block_size;
        }
        ir += current_block_size;
    }

    dma_queue_flush(dma_q);
}

// 6. ADD_ID (src1 gathered via src2 indices)
स्थिर शून्य द्विआधारी_thread_add_id_f32(अचिन्हित पूर्णाङ्क nth, अचिन्हित पूर्णाङ्क ith, शून्य * data) {
    संरचना htp_द्विआधारी_context * bctx = (संरचना htp_द्विआधारी_context *) data;
    संरचना htp_ops_context * octx = bctx->octx;

    अचर संरचना htp_tensor * src0 = octx->src[0];
    अचर संरचना htp_tensor * src1 = octx->src[1];
    अचर संरचना htp_tensor * dst  = octx->dst;

    अचर uint32_t ne00 = src0->ne[0];
    अचर uint32_t ne01 = src0->ne[1];
    अचर uint32_t ne02 = src0->ne[2];
    अचर uint32_t ne03 = src0->ne[3];

    अचर uint32_t nb01 = src0->nb[1];
    अचर uint32_t nb02 = src0->nb[2];
    अचर uint32_t nb03 = src0->nb[3];
    अचर uint32_t src1_stride = bctx->src1_row_size_aligned;

    अचर uint32_t nb1 = dst->nb[1];
    अचर uint32_t nb2 = dst->nb[2];
    अचर uint32_t nb3 = dst->nb[3];

    अचर uint32_t row_size_bytes = bctx->row_size_bytes;
    अचर uint32_t start_row = bctx->row_start + bctx->nrows_per_thread * ith;
    अचर uint32_t end_row   = MIN(start_row + bctx->nrows_per_thread, bctx->row_start + bctx->total_rows);
    यदि (start_row >= end_row) फर्काऊ;

    अचर संरचना htp_द्विआधारी_vtcm_layout * layout = &bctx->vtcm_layout;
    uint8_t * src0_spad_base = VTCM_LAYOUT_PTR(uint8_t, bctx->vtcm_base, layout->off_src0) + (ith * layout->src0_bytes_per_thread);
    uint8_t * dst_spad_base  = VTCM_LAYOUT_PTR(uint8_t, bctx->vtcm_base, layout->off_dst)  + (ith * layout->dst_bytes_per_thread);
    अचर uint8_t * vtcm_src1 = VTCM_LAYOUT_PTR(अचर uint8_t, bctx->vtcm_base, layout->off_src1);
    अचर int32_t * vtcm_src2 = VTCM_LAYOUT_PTR(अचर int32_t, bctx->vtcm_base, layout->off_src2);
    माप_प्रकार src0_spad_half    = layout->src0_spad_half_size;
    माप_प्रकार dst_spad_half     = layout->dst_spad_half_size;

    dma_queue * dma_q = octx->ctx->dma[ith];
    uint32_t ir_prefetch = start_row;
    पूर्णाङ्क spad_idx = 0;

    लागि (पूर्णाङ्क k = 0; k < 2 && ir_prefetch < end_row; k++) {
        uint32_t current_block_size = calc_block_size(bctx, ir_prefetch, end_row, ne01, ne02);
        uint32_t i03 = fastdiv(ir_prefetch, &bctx->src0_dim12_div);
        uint32_t rem = ir_prefetch - i03 * (ne02 * ne01);
        uint32_t i02 = fastdiv(rem, &bctx->src0_dim1_div);
        uint32_t i01 = rem - i02 * ne01;

        dma_addr_t src0_curr = src0->data + i03 * nb03 + i02 * nb02 + i01 * nb01;
        dma_addr_t dst_curr  = dst->data  + i03 * nb3  + i02 * nb2  + i01 * nb1;

        uint8_t * s0_spad = src0_spad_base + spad_idx * src0_spad_half;
        uint8_t * d_spad  = dst_spad_base  + spad_idx * dst_spad_half;

        dma_queue_थप(dma_q, dma_make_data(dst_curr, d_spad), nb1, bctx->dst_row_size_aligned, row_size_bytes, 0);
        dma_queue_थप(dma_q, dma_make_data(s0_spad, src0_curr), bctx->src0_row_size_aligned, nb01, row_size_bytes, current_block_size);
        ir_prefetch += current_block_size;
        spad_idx ^= 1;
    }

    संरचना htp_thread_trace * tr = &octx->ctx->trace[ith];

    लागि (uint32_t ir = start_row; ir < end_row; ) {
        uint32_t current_block_size = calc_block_size(bctx, ir, end_row, ne01, ne02);
        uint8_t * d_spad = (uint8_t *) dma_queue_झिक(dma_q).src;
        uint8_t * s0_spad = (uint8_t *) dma_queue_झिक(dma_q).dst;

        uint32_t i03 = fastdiv(ir, &bctx->src0_dim12_div);
        uint32_t rem = ir - i03 * (ne02 * ne01);
        uint32_t i02 = fastdiv(rem, &bctx->src0_dim1_div);
        uint32_t i01 = rem - i02 * ne01;

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);
        compute_add_id_t compute = (compute_add_id_t) bctx->compute;
        compute(d_spad, s0_spad, vtcm_src1, vtcm_src2,
                i01, i02, ne01, src1_stride,
                current_block_size, bctx->dst_row_size_aligned, bctx->src0_row_size_aligned, ne00);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);

        dma_addr_t dst_curr = dst->data + i03 * nb3 + i02 * nb2 + i01 * nb1;
        dma_queue_थप(dma_q, dma_make_data(dst_curr, d_spad), nb1, bctx->dst_row_size_aligned, row_size_bytes, current_block_size);

        यदि (ir_prefetch < end_row) {
             uint32_t next_block_size = calc_block_size(bctx, ir_prefetch, end_row, ne01, ne02);
             uint32_t p03, p02, p01, prem;
             p03  = fastdiv(ir_prefetch, &bctx->src0_dim12_div);
             prem = ir_prefetch - p03 * (ne02 * ne01);
             p02  = fastdiv(prem, &bctx->src0_dim1_div);
             p01  = prem - p02 * ne01;
             dma_addr_t s0_next = src0->data + p03 * nb03 + p02 * nb02 + p01 * nb01;
             dma_queue_थप(dma_q, dma_make_data(s0_spad, s0_next), bctx->src0_row_size_aligned, nb01, row_size_bytes, next_block_size);
             ir_prefetch += next_block_size;
        }
        ir += current_block_size;
    }

    dma_queue_flush(dma_q);
}

स्थिर अन्तःस्थ शून्य hvx_div_scalar_f32(uint8_t * restrict dst, अचर uint8_t * restrict src, अचर भिन्न val, अचर uint32_t अङ्क_elems) {
    hvx_mul_scalar_f32(dst, src, 1.0f / val, अङ्क_elems);
}

स्थिर अन्तःस्थ शून्य hvx_div_scalar_f16(uint8_t * restrict dst, अचर uint8_t * restrict src, अचर _Float16 val, अचर uint32_t अङ्क_elems) {
    hvx_div_scalar_f16_aa(dst, src, val, अङ्क_elems);
}

प्रकारनिर्देश शून्य (*compute_द्विआधारी_chunked_t)(
    uint8_t * restrict dst,
    अचर uint8_t * restrict src0,
    अचर uint8_t * restrict src1,
    अचर uint32_t अङ्क_elems
);

प्रकारनिर्देश शून्य (*compute_द्विआधारी_scalar_chunked_f32_t)(
    uint8_t * restrict dst,
    अचर uint8_t * restrict src0,
    अचर भिन्न val,
    अचर uint32_t अङ्क_elems
);

प्रकारनिर्देश शून्य (*compute_द्विआधारी_scalar_chunked_f16_t)(
    uint8_t * restrict dst,
    अचर uint8_t * restrict src0,
    अचर _Float16 val,
    अचर uint32_t अङ्क_elems
);

संरचना द्विआधारी_chunked_context {
    संरचना htp_ops_context * octx;
    संरचना htp_द्विआधारी_vtcm_layout vtcm_layout;
    uint8_t *                vtcm_base;
    uint32_t                 elem_start;
    uint32_t                 nelem;
    uint32_t                 chunk_size;
    uint32_t                 chunks_per_thread;
    uint32_t                 total_chunks;
    bool                     is_scalar;
    भिन्न                    scalar_f32;
    _Float16                 scalar_f16;
    compute_द्विआधारी_chunked_t compute;
    compute_द्विआधारी_scalar_chunked_f32_t compute_scalar_f32;
    compute_द्विआधारी_scalar_chunked_f16_t compute_scalar_f16;
};

स्थिर शून्य द्विआधारी_thread_chunked(अचिन्हित पूर्णाङ्क nth, अचिन्हित पूर्णाङ्क ith, शून्य * data) {
    (शून्य) nth;
    संरचना द्विआधारी_chunked_context * ctx = (संरचना द्विआधारी_chunked_context *) data;
    संरचना htp_ops_context * octx = ctx->octx;
    अचर संरचना htp_tensor * src0 = octx->src[0];
    अचर संरचना htp_tensor * src1 = octx->src[1];
    अचर संरचना htp_tensor * dst  = octx->dst;

    अचर uint32_t start_chunk = ctx->chunks_per_thread * ith;
    अचर uint32_t end_chunk   = MIN(start_chunk + ctx->chunks_per_thread, ctx->total_chunks);
    यदि (start_chunk >= end_chunk) {
        फर्काऊ;
    }

    अचर uint32_t src0_type   = src0->type;
    अचर माप_प्रकार   elem_size   = (src0_type == HTP_TYPE_F32) ? परिमाण(भिन्न) : परिमाण(_Float16);
    अचर uint32_t chunk_size  = ctx->chunk_size;
    अचर माप_प्रकार   chunk_bytes = ctx->vtcm_layout.src0_spad_half_size;

    FARF(माथिल्लो_2, "binary-chunked: %d/%d (%u:%u) chunks %u elems %u",
         ith, nth, start_chunk, end_chunk, ctx->total_chunks, ctx->nelem);

    अचर संरचना htp_द्विआधारी_vtcm_layout * layout = &ctx->vtcm_layout;
    uint8_t * src0_spad_base = VTCM_LAYOUT_PTR(uint8_t, ctx->vtcm_base, layout->off_src0) + (ith * layout->src0_bytes_per_thread);
    uint8_t * src1_spad_base = ctx->is_scalar ? रिक्त : (VTCM_LAYOUT_PTR(uint8_t, ctx->vtcm_base, layout->off_src1) + (ith * layout->src1_bytes_per_thread));
    uint8_t * dst_spad_base  = VTCM_LAYOUT_PTR(uint8_t, ctx->vtcm_base, layout->off_dst)  + (ith * layout->dst_bytes_per_thread);

    dma_queue * dma_q = octx->ctx->dma[ith];
    uint32_t prefetch_chunk = start_chunk;
    पूर्णाङ्क spad_idx = 0;

    लागि (पूर्णाङ्क k = 0; k < 2 && prefetch_chunk < end_chunk; k++) {
        अचर uint32_t c_start   = ctx->elem_start + prefetch_chunk * chunk_size;
        अचर uint32_t c_end     = MIN(c_start + chunk_size, ctx->elem_start + ctx->nelem);
        अचर uint32_t cur_elems = c_end - c_start;
        अचर uint32_t cur_bytes = cur_elems * elem_size;

        dma_addr_t s0_curr = src0->data + (माप_प्रकार) c_start * elem_size;
        dma_addr_t d_curr  = dst->data  + (माप_प्रकार) c_start * elem_size;

        uint8_t * s0_spad = src0_spad_base + spad_idx * chunk_bytes;
        uint8_t * d_spad  = dst_spad_base  + spad_idx * chunk_bytes;

        dma_queue_थप(dma_q, dma_make_data(d_curr, d_spad), chunk_bytes, chunk_bytes, cur_bytes, 0);
        dma_queue_थप(dma_q, dma_make_data(s0_spad, s0_curr), chunk_bytes, chunk_bytes, cur_bytes, 1);
        यदि (!ctx->is_scalar) {
            dma_addr_t s1_curr = src1->data + (माप_प्रकार) c_start * elem_size;
            uint8_t * s1_spad = src1_spad_base + spad_idx * chunk_bytes;
            dma_queue_थप(dma_q, dma_make_data(s1_spad, s1_curr), chunk_bytes, chunk_bytes, cur_bytes, 1);
        }

        prefetch_chunk++;
        spad_idx ^= 1;
    }

    संरचना htp_thread_trace * tr = &octx->ctx->trace[ith];

    लागि (uint32_t c = start_chunk; c < end_chunk; c++) {
        अचर uint32_t c_start   = ctx->elem_start + c * chunk_size;
        अचर uint32_t c_end     = MIN(c_start + chunk_size, ctx->elem_start + ctx->nelem);
        अचर uint32_t cur_elems = c_end - c_start;
        अचर uint32_t cur_bytes = cur_elems * elem_size;

        uint8_t * d_spad  = (uint8_t *) dma_queue_झिक(dma_q).src;
        uint8_t * s0_spad = (uint8_t *) dma_queue_झिक(dma_q).dst;
        uint8_t * s1_spad = ctx->is_scalar ? रिक्त : (uint8_t *) dma_queue_झिक(dma_q).dst;

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

        dma_addr_t dst_curr = dst->data + (माप_प्रकार) c_start * elem_size;
        dma_queue_थप(dma_q, dma_make_data(dst_curr, d_spad), chunk_bytes, chunk_bytes, cur_bytes, 1);

        यदि (prefetch_chunk < end_chunk) {
            अचर uint32_t pc_start  = ctx->elem_start + prefetch_chunk * chunk_size;
            अचर uint32_t pc_end    = MIN(pc_start + chunk_size, ctx->elem_start + ctx->nelem);
            अचर uint32_t p_elems   = pc_end - pc_start;
            अचर uint32_t p_bytes   = p_elems * elem_size;

            dma_addr_t s0_next = src0->data + (माप_प्रकार) pc_start * elem_size;
            dma_queue_थप(dma_q, dma_make_data(s0_spad, s0_next), chunk_bytes, chunk_bytes, p_bytes, 1);
            यदि (!ctx->is_scalar) {
                dma_addr_t s1_next = src1->data + (माप_प्रकार) pc_start * elem_size;
                dma_queue_थप(dma_q, dma_make_data(s1_spad, s1_next), chunk_bytes, chunk_bytes, p_bytes, 1);
            }

            prefetch_chunk++;
        }
    }

    dma_queue_flush(dma_q);
}

स्थिर पूर्णाङ्क execute_op_द्विआधारी(संरचना htp_ops_context * octx) {
    अचर संरचना htp_tensor * src0 = octx->src[0];
    अचर संरचना htp_tensor * src1 = octx->src[1];
    अचर संरचना htp_tensor * dst  = octx->dst;
    अचर संरचना htp_द्विआधारी_kernel_params * kparams = (अचर संरचना htp_द्विआधारी_kernel_params *) octx->kernel_params;

    अचर uint32_t src0_nrows = src0->ne[1] * src0->ne[2] * src0->ne[3];

    // Use packed row sizes for VTCM allocation and alignment
    अचर uint32_t src0_type = octx->src[0]->type;
    अचर माप_प्रकार elem_size = (src0_type == HTP_TYPE_F32) ? परिमाण(भिन्न) : परिमाण(_Float16);
    अचर माप_प्रकार src0_row_size = src0->ne[0] * elem_size;
    अचर माप_प्रकार src1_row_size = src1->ne[0] * elem_size;
    अचर माप_प्रकार dst_row_size  = dst->ne[0]  * elem_size;

    यदि (kparams->kernel_type == HTP_द्विआधारी_KERNEL_CHUNKED) {
        अचर bool is_scalar = kparams->is_scalar || (src1->ne[0] == 1 && src1->ne[1] == 1 && src1->ne[2] == 1 && src1->ne[3] == 1);

        compute_द्विआधारी_chunked_t compute = रिक्त;
        compute_द्विआधारी_scalar_chunked_f32_t compute_scalar_f32 = रिक्त;
        compute_द्विआधारी_scalar_chunked_f16_t compute_scalar_f16 = रिक्त;

        यदि (is_scalar) {
            यदि (src0_type == HTP_TYPE_F32) {
                विकल्प (octx->op) {
                    पक्ष HTP_OP_ADD: compute_scalar_f32 = hvx_add_scalar_f32; रोक;
                    पक्ष HTP_OP_SUB: compute_scalar_f32 = hvx_sub_scalar_f32; रोक;
                    पक्ष HTP_OP_MUL: compute_scalar_f32 = hvx_mul_scalar_f32; रोक;
                    पक्ष HTP_OP_DIV: compute_scalar_f32 = hvx_div_scalar_f32; रोक;
                    स्वभाव: रोक;
                }
            } अन्यथा यदि (src0_type == HTP_TYPE_F16) {
                विकल्प (octx->op) {
                    पक्ष HTP_OP_ADD: compute_scalar_f16 = hvx_add_scalar_f16; रोक;
                    पक्ष HTP_OP_SUB: compute_scalar_f16 = hvx_sub_scalar_f16; रोक;
                    पक्ष HTP_OP_MUL: compute_scalar_f16 = hvx_mul_scalar_f16; रोक;
                    पक्ष HTP_OP_DIV: compute_scalar_f16 = hvx_div_scalar_f16; रोक;
                    स्वभाव: रोक;
                }
            }
            यदि (!compute_scalar_f32 && !compute_scalar_f16) {
                फर्काऊ HTP_STATUS_NO_SUPPORT;
            }
        } अन्यथा {
            यदि (src0_type == HTP_TYPE_F32) {
                विकल्प (octx->op) {
                    पक्ष HTP_OP_ADD: compute = hvx_add_f32; रोक;
                    पक्ष HTP_OP_SUB: compute = hvx_sub_f32; रोक;
                    पक्ष HTP_OP_MUL: compute = hvx_mul_f32; रोक;
                    पक्ष HTP_OP_DIV: compute = hvx_div_f32; रोक;
                    स्वभाव: रोक;
                }
            } अन्यथा यदि (src0_type == HTP_TYPE_F16) {
                विकल्प (octx->op) {
                    पक्ष HTP_OP_ADD: compute = hvx_add_f16; रोक;
                    पक्ष HTP_OP_SUB: compute = hvx_sub_f16; रोक;
                    पक्ष HTP_OP_MUL: compute = hvx_mul_f16; रोक;
                    पक्ष HTP_OP_DIV: compute = hvx_div_f16; रोक;
                    स्वभाव: रोक;
                }
            }
            यदि (!compute) {
                फर्काऊ HTP_STATUS_NO_SUPPORT;
            }
        }

        अचर uint32_t total_elems = (uint32_t) (src0->ne[0] * src0->ne[1] * src0->ne[2] * src0->ne[3]);
        यदि (total_elems == 0) {
            फर्काऊ HTP_STATUS_OK;
        }

        uint32_t elem_start = 0;
        uint32_t nelem      = total_elems;
        अचर uint32_t elems_per_पङ्क्ति = (src0_type == HTP_TYPE_F32) ? 32 : 64;

        यदि (octx->ctx->mdev.गन्ती > 1) {
            अचर bool can_split = htp_tensor_mdev_data_aligned(dst) &&
                                   htp_tensor_mdev_data_aligned(src0) &&
                                   (is_scalar || htp_tensor_mdev_data_aligned(src1)) &&
                                   htp_tensor_is_contiguous(dst, elem_size) &&
                                   htp_tensor_is_contiguous(src0, elem_size) &&
                                   (is_scalar || htp_tensor_is_contiguous(src1, elem_size));
            अचर संरचना htp_tensor_mdev_range range = htp_tensor_mdev_partition(
                total_elems, can_split ? elems_per_पङ्क्ति : 0,
                octx->ctx->mdev.idx, octx->ctx->mdev.गन्ती, &octx->ctx->mdev.गन्ती_div);
            elem_start = range.start;
            nelem      = range.गन्ती;
        }

        यदि (nelem == 0) {
            फर्काऊ HTP_STATUS_OK;
        }

        यदि (!htp_ops_context_set_n_threads(octx, kparams->n_threads)) {
            फर्काऊ HTP_STATUS_INVAL_PARAMS;
        }

        संरचना htp_द्विआधारी_vtcm_layout vtcm_layout;
        htp_द्विआधारी_vtcm_layout_build(&vtcm_layout, kparams, octx->ctx->vtcm_size);
        यदि (vtcm_layout.total_bytes == 0 || vtcm_layout.total_bytes > octx->ctx->vtcm_size) {
            फर्काऊ HTP_STATUS_VTCM_TOO_SMALL;
        }

        अचर uint32_t chunk_size = kparams->chunk_size > 0 ? kparams->chunk_size : (32768 / elem_size);
        अचर uint32_t total_chunks = (nelem + chunk_size - 1) / chunk_size;
        अचर uint32_t n_threads = (total_chunks >= 2) ? MIN(octx->n_threads, total_chunks) : 1;
        अचर uint32_t chunks_per_thread = (total_chunks + n_threads - 1) / n_threads;

        भिन्न scalar_f32 = 0.0f;
        _Float16 scalar_f16 = 0;
        यदि (is_scalar) {
            uint8_t * vtcm_src1 = VTCM_LAYOUT_PTR(uint8_t, octx->ctx->vtcm_base, vtcm_layout.off_src1);
            dma_queue * dma_q = octx->ctx->dma[0];
            dma_queue_थप(dma_q, dma_make_data(vtcm_src1, src1->data), 128, 0, elem_size, 1);
            dma_queue_झिक(dma_q);

            यदि (src0_type == HTP_TYPE_F32) {
                scalar_f32 = ((अचर भिन्न *) vtcm_src1)[0];
            } अन्यथा {
                scalar_f16 = ((अचर _Float16 *) vtcm_src1)[0];
            }
        }

        संरचना द्विआधारी_chunked_context cctx = {
            .octx                = octx,
            .vtcm_layout         = vtcm_layout,
            .vtcm_base           = (uint8_t *) octx->ctx->vtcm_base,
            .elem_start          = elem_start,
            .nelem               = nelem,
            .chunk_size          = chunk_size,
            .chunks_per_thread   = chunks_per_thread,
            .total_chunks        = total_chunks,
            .is_scalar           = is_scalar,
            .scalar_f32          = scalar_f32,
            .scalar_f16          = scalar_f16,
            .compute             = compute,
            .compute_scalar_f32  = compute_scalar_f32,
            .compute_scalar_f16  = compute_scalar_f16,
        };

        work_queue_run(octx->ctx->work_queue, द्विआधारी_thread_chunked, &cctx, n_threads);
        फर्काऊ HTP_STATUS_OK;
    }

    uint32_t row_start = 0;
    uint32_t nrows     = src0_nrows;

    यदि (octx->ctx->mdev.गन्ती > 1) {
        uint32_t rows_per_chunk = 0;
        htp_tensor_mdev_rows_per_chunk(dst, (uint32_t) elem_size, (uint32_t) dst_row_size, &rows_per_chunk);
        अचर संरचना htp_tensor_mdev_range range = htp_tensor_mdev_partition(src0_nrows, rows_per_chunk,
                                                       octx->ctx->mdev.idx, octx->ctx->mdev.गन्ती, &octx->ctx->mdev.गन्ती_div);
        row_start = range.start;
        nrows     = range.गन्ती;
    }

    यदि (nrows == 0) {
        फर्काऊ HTP_STATUS_OK;
    }

    यदि (!htp_ops_context_set_n_threads(octx, kparams->n_threads)) {
        फर्काऊ HTP_STATUS_INVAL_PARAMS;
    }

    अचर uint32_t n_threads = octx->n_threads;
    अचर माप_प्रकार src0_row_size_aligned = kparams->src0_row_size_aligned;
    अचर माप_प्रकार src1_row_size_aligned = kparams->src1_row_size_aligned;
    अचर माप_प्रकार dst_row_size_aligned  = kparams->dst_row_size_aligned;

    संरचना htp_द्विआधारी_context bctx;
    bctx.vtcm_base = (uint8_t *) octx->ctx->vtcm_base;
    htp_द्विआधारी_vtcm_layout_build(&bctx.vtcm_layout, kparams, octx->ctx->vtcm_size);

    यदि (bctx.vtcm_layout.rows_per_buffer == 0 || bctx.vtcm_layout.total_bytes > octx->ctx->vtcm_size) {
        फर्काऊ HTP_STATUS_VTCM_TOO_SMALL;
    }

    dma_queue * dma_q = octx->ctx->dma[0];
    uint8_t * vtcm_src1 = VTCM_LAYOUT_PTR(uint8_t, bctx.vtcm_base, bctx.vtcm_layout.off_src1);
    यदि (kparams->kernel_type == HTP_द्विआधारी_KERNEL_ROW_BCAST) {
        dma_queue_थप(dma_q, dma_make_data(vtcm_src1, src1->data), bctx.vtcm_layout.src1_size, 0, src1->ne[0] * elem_size, 1);
    } अन्यथा यदि (kparams->kernel_type == HTP_द्विआधारी_KERNEL_SCALAR ||
               kparams->kernel_type == HTP_द्विआधारी_KERNEL_REPEAT ||
               kparams->kernel_type == HTP_द्विआधारी_KERNEL_COMPLEX) {
        यदि (!kparams->src1_strided) {
            अचर माप_प्रकार src1_bytes = (माप_प्रकार) src1->ne[0] * src1->ne[1] * src1->ne[2] * src1->ne[3] * elem_size;
            dma_queue_थप(dma_q, dma_make_data(vtcm_src1, src1->data), bctx.vtcm_layout.src1_size, 0, src1_bytes, 1);
        } अन्यथा {
            अचर uint32_t ne11 = src1->ne[1];
            अचर uint32_t ne12 = src1->ne[2];
            अचर uint32_t ne13 = src1->ne[3];
            अचर माप_प्रकार row_bytes = src1->ne[0] * elem_size;
            लागि (uint32_t i13 = 0; i13 < ne13; i13++) {
                लागि (uint32_t i12 = 0; i12 < ne12; i12++) {
                    dma_addr_t src_addr = src1->data + i13 * src1->nb[3] + i12 * src1->nb[2];
                    uint8_t * dst_ptr = vtcm_src1 + (i13 * ne12 * ne11 + i12 * ne11) * row_bytes;
                    जबसम्म (!dma_queue_थप(dma_q, dma_make_data(dst_ptr, src_addr),
                                           row_bytes, src1->nb[1], row_bytes, ne11)) {
                        dma_queue_झिक(dma_q);
                    }
                }
            }
        }
    } अन्यथा यदि (kparams->kernel_type == HTP_द्विआधारी_KERNEL_ADD_ID) {
        अचर संरचना htp_tensor * src2 = octx->src[2];
        uint8_t * vtcm_src2 = VTCM_LAYOUT_PTR(uint8_t, bctx.vtcm_base, bctx.vtcm_layout.off_src2);
        dma_queue_थप(dma_q, dma_make_data(vtcm_src1, src1->data),
                       kparams->src1_row_size_aligned, src1->nb[1],
                       src1->ne[0] * elem_size, src1->ne[1]);
        अचर माप_प्रकार src2_row_bytes = src2->ne[0] * परिमाण(int32_t);
        dma_queue_थप(dma_q, dma_make_data(vtcm_src2, src2->data),
                       src2_row_bytes, src2->nb[1],
                       src2_row_bytes, src2->ne[1]);
    }

    bctx.octx                  = octx;
    bctx.nrows_per_thread      = fastdiv(nrows + n_threads - 1, &octx->n_threads_div);
    bctx.total_rows            = nrows;
    bctx.row_start             = row_start;
    bctx.block_max             = bctx.vtcm_layout.rows_per_buffer;
    bctx.src0_row_size_aligned = src0_row_size_aligned;
    bctx.src1_row_size_aligned = src1_row_size_aligned;
    bctx.dst_row_size_aligned  = dst_row_size_aligned;
    bctx.row_size_bytes        = src0_row_size;

    bctx.src0_dim1_div  = init_fastdiv_मानहरू(src0->ne[1]);
    bctx.src0_dim2_div  = init_fastdiv_मानहरू(src0->ne[2]);
    bctx.src0_dim12_div = init_fastdiv_मानहरू(src0->ne[1] * src0->ne[2]);

    bctx.src1_dim1_div  = init_fastdiv_मानहरू(src1->ne[1]);
    bctx.src1_dim2_div  = init_fastdiv_मानहरू(src1->ne[2]);
    bctx.src1_dim3_div  = init_fastdiv_मानहरू(src1->ne[3]);

    bctx.split_at_ne01  = kparams->split_at_ne01;
    bctx.split_at_ne02  = kparams->split_at_ne02;

    worker_callback_t worker_func = रिक्त;
    शून्य * compute_func = रिक्त;

    विकल्प (kparams->kernel_type) {
        पक्ष HTP_द्विआधारी_KERNEL_SAME_SHAPE:
            worker_func = द्विआधारी_thread_vector_same_shape;
            यदि (src0_type == HTP_TYPE_F32) {
                विकल्प (octx->op) {
                    पक्ष HTP_OP_ADD: compute_func = compute_same_shape_add_f32; रोक;
                    पक्ष HTP_OP_SUB: compute_func = compute_same_shape_sub_f32; रोक;
                    पक्ष HTP_OP_MUL: compute_func = compute_same_shape_mul_f32; रोक;
                    पक्ष HTP_OP_DIV: compute_func = compute_same_shape_div_f32; रोक;
                    स्वभाव: रोक;
                }
            } अन्यथा यदि (src0_type == HTP_TYPE_F16) {
                विकल्प (octx->op) {
                    पक्ष HTP_OP_ADD: compute_func = compute_same_shape_add_f16; रोक;
                    पक्ष HTP_OP_SUB: compute_func = compute_same_shape_sub_f16; रोक;
                    पक्ष HTP_OP_MUL: compute_func = compute_same_shape_mul_f16; रोक;
                    पक्ष HTP_OP_DIV: compute_func = compute_same_shape_div_f16; रोक;
                    स्वभाव: रोक;
                }
            }
            रोक;
        पक्ष HTP_द्विआधारी_KERNEL_ROW_BCAST:
            worker_func = द्विआधारी_thread_vector_row_broadcast;
            यदि (src0_type == HTP_TYPE_F32) {
                विकल्प (octx->op) {
                    पक्ष HTP_OP_ADD: compute_func = compute_row_bcast_add_f32; रोक;
                    पक्ष HTP_OP_SUB: compute_func = compute_row_bcast_sub_f32; रोक;
                    पक्ष HTP_OP_MUL: compute_func = compute_row_bcast_mul_f32; रोक;
                    पक्ष HTP_OP_DIV: compute_func = compute_row_bcast_div_f32; रोक;
                    स्वभाव: रोक;
                }
            } अन्यथा यदि (src0_type == HTP_TYPE_F16) {
                विकल्प (octx->op) {
                    पक्ष HTP_OP_ADD: compute_func = compute_row_bcast_add_f16; रोक;
                    पक्ष HTP_OP_SUB: compute_func = compute_row_bcast_sub_f16; रोक;
                    पक्ष HTP_OP_MUL: compute_func = compute_row_bcast_mul_f16; रोक;
                    पक्ष HTP_OP_DIV: compute_func = compute_row_bcast_div_f16; रोक;
                    स्वभाव: रोक;
                }
            }
            रोक;
        पक्ष HTP_द्विआधारी_KERNEL_SCALAR:
            worker_func = द्विआधारी_thread_scalar;
            यदि (src0_type == HTP_TYPE_F32) {
                विकल्प (octx->op) {
                    पक्ष HTP_OP_ADD: compute_func = compute_scalar_add_f32; रोक;
                    पक्ष HTP_OP_SUB: compute_func = compute_scalar_sub_f32; रोक;
                    पक्ष HTP_OP_MUL: compute_func = compute_scalar_mul_f32; रोक;
                    पक्ष HTP_OP_DIV: compute_func = compute_scalar_div_f32; रोक;
                    स्वभाव: रोक;
                }
            } अन्यथा यदि (src0_type == HTP_TYPE_F16) {
                विकल्प (octx->op) {
                    पक्ष HTP_OP_ADD: compute_func = compute_scalar_add_f16; रोक;
                    पक्ष HTP_OP_SUB: compute_func = compute_scalar_sub_f16; रोक;
                    पक्ष HTP_OP_MUL: compute_func = compute_scalar_mul_f16; रोक;
                    पक्ष HTP_OP_DIV: compute_func = compute_scalar_div_f16; रोक;
                    स्वभाव: रोक;
                }
            }
            रोक;
        पक्ष HTP_द्विआधारी_KERNEL_COMPLEX:
            worker_func = द्विआधारी_thread_vector_complex;
            यदि (src0_type == HTP_TYPE_F32) {
                विकल्प (octx->op) {
                    पक्ष HTP_OP_ADD: compute_func = compute_complex_add_f32; रोक;
                    पक्ष HTP_OP_SUB: compute_func = compute_complex_sub_f32; रोक;
                    पक्ष HTP_OP_MUL: compute_func = compute_complex_mul_f32; रोक;
                    पक्ष HTP_OP_DIV: compute_func = compute_complex_div_f32; रोक;
                    स्वभाव: रोक;
                }
            } अन्यथा यदि (src0_type == HTP_TYPE_F16) {
                विकल्प (octx->op) {
                    पक्ष HTP_OP_ADD: compute_func = compute_complex_add_f16; रोक;
                    पक्ष HTP_OP_SUB: compute_func = compute_complex_sub_f16; रोक;
                    पक्ष HTP_OP_MUL: compute_func = compute_complex_mul_f16; रोक;
                    पक्ष HTP_OP_DIV: compute_func = compute_complex_div_f16; रोक;
                    स्वभाव: रोक;
                }
            }
            रोक;
        पक्ष HTP_द्विआधारी_KERNEL_REPEAT:
            worker_func = द्विआधारी_thread_element_repeat;
            यदि (src0_type == HTP_TYPE_F32) {
                विकल्प (octx->op) {
                    पक्ष HTP_OP_ADD: compute_func = compute_repeat_add_f32; रोक;
                    पक्ष HTP_OP_SUB: compute_func = compute_repeat_sub_f32; रोक;
                    पक्ष HTP_OP_MUL: compute_func = compute_repeat_mul_f32; रोक;
                    पक्ष HTP_OP_DIV: compute_func = compute_repeat_div_f32; रोक;
                    स्वभाव: रोक;
                }
            } अन्यथा यदि (src0_type == HTP_TYPE_F16) {
                विकल्प (octx->op) {
                    पक्ष HTP_OP_ADD: compute_func = compute_repeat_add_f16; रोक;
                    पक्ष HTP_OP_SUB: compute_func = compute_repeat_sub_f16; रोक;
                    पक्ष HTP_OP_MUL: compute_func = compute_repeat_mul_f16; रोक;
                    पक्ष HTP_OP_DIV: compute_func = compute_repeat_div_f16; रोक;
                    स्वभाव: रोक;
                }
            }
            रोक;
        पक्ष HTP_द्विआधारी_KERNEL_ADD_ID:
            यदि (octx->op == HTP_OP_ADD_ID && src0_type == HTP_TYPE_F32) {
                worker_func = द्विआधारी_thread_add_id_f32;
                compute_func = (शून्य *) compute_add_id_f32;
            }
            रोक;
        स्वभाव: रोक;
    }

    यदि (!worker_func || !compute_func) {
        फर्काऊ HTP_STATUS_NO_SUPPORT;
    }

    bctx.compute = compute_func;

    यदि (kparams->kernel_type == HTP_द्विआधारी_KERNEL_ROW_BCAST ||
        kparams->kernel_type == HTP_द्विआधारी_KERNEL_SCALAR ||
        kparams->kernel_type == HTP_द्विआधारी_KERNEL_REPEAT ||
        kparams->kernel_type == HTP_द्विआधारी_KERNEL_COMPLEX ||
        kparams->kernel_type == HTP_द्विआधारी_KERNEL_ADD_ID) {
        dma_queue_flush(dma_q);
    }

    work_queue_run(octx->ctx->work_queue, worker_func, &bctx, n_threads);

    फर्काऊ HTP_STATUS_OK;
}

पूर्णाङ्क op_द्विआधारी(संरचना htp_ops_context * octx) {
    अचर uint32_t src0_type = octx->src[0]->type;
    यदि ((src0_type == HTP_TYPE_F32) || (src0_type == HTP_TYPE_F16)) {
        फर्काऊ execute_op_द्विआधारी(octx);
    }

    फर्काऊ HTP_STATUS_NO_SUPPORT;
}
