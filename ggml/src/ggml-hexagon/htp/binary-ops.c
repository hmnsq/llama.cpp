#आशय clang diagnostic ignored "-Wunused-variable"
#आशय clang diagnostic ignored "-Wunused-function"
#आशय clang diagnostic ignored "-Wunused-but-set-variable"

#समावेश <HAP_farf.h>
#समावेश <HAP_perf.h>

#समावेश <math.h>
#समावेश <string.h>

#समावेश "dma-queue.h"
#समावेश "hvx-utils.h"

#घोषणा GGML_COMMON_DECL_C
#समावेश "ggml-common.h"
#समावेश "hex-common.h"
#समावेश "hex-profile.h"
#समावेश "binary-ops.h"
#समावेश "htp-ctx.h"
#समावेश "htp-ops.h"
#समावेश "htp-tensor.h"

#अगर_अघोषित MIN
#घोषणा MIN(a, b) ((a) < (b) ? (a) : (b))
#पूर्ण_अगर

// Context for binary operations
काष्ठा htp_द्विआधारी_context {
    काष्ठा htp_ops_context * octx;
    काष्ठा htp_द्विआधारी_vtcm_layout vtcm_layout;
    uint8_t * vtcm_base;

    काष्ठा fastdiv_मूल्यावली src0_dim1_div; // ne01
    काष्ठा fastdiv_मूल्यावली src0_dim2_div; // ne02
    काष्ठा fastdiv_मूल्यावली src0_dim12_div;// ne03

    काष्ठा fastdiv_मूल्यावली src1_dim1_div; // ne11
    काष्ठा fastdiv_मूल्यावली src1_dim2_div; // ne12
    काष्ठा fastdiv_मूल्यावली src1_dim3_div; // ne13

    uint32_t block_max;
    uint32_t nrows_per_thread;
    uint32_t योग_rows;
    uint32_t row_start;
    माप_प्रकार   src0_row_आकार_aligned;
    माप_प्रकार   src1_row_आकार_aligned;
    माप_प्रकार   dst_row_आकार_aligned;
    माप_प्रकार   row_आकार_bytes;

    bool split_at_ne01;
    bool split_at_ne02;

    व्योम * compute;
};

#घोषणा htp_द्विआधारी_preamble                        \
    स्थिर काष्ठा htp_tensor * src0 = octx->src[0]; \
    स्थिर काष्ठा htp_tensor * src1 = octx->src[1]; \
    स्थिर काष्ठा htp_tensor * dst  = octx->dst;    \
                                                   \
    स्थिर uint32_t ne00 = src0->ne[0];             \
    स्थिर uint32_t ne01 = src0->ne[1];             \
    स्थिर uint32_t ne02 = src0->ne[2];             \
    स्थिर uint32_t ne03 = src0->ne[3];             \
                                                   \
    स्थिर uint32_t ne10 = src1->ne[0];             \
    स्थिर uint32_t ne11 = src1->ne[1];             \
    स्थिर uint32_t ne12 = src1->ne[2];             \
    स्थिर uint32_t ne13 = src1->ne[3];             \
                                                   \
    स्थिर uint32_t nb01 = src0->nb[1];             \
    स्थिर uint32_t nb02 = src0->nb[2];             \
    स्थिर uint32_t nb03 = src0->nb[3];             \
                                                   \
    स्थिर uint32_t nb11 = src1->nb[1];             \
    स्थिर uint32_t nb12 = src1->nb[2];             \
    स्थिर uint32_t nb13 = src1->nb[3];             \
                                                   \
    स्थिर uint32_t nb1 = dst->nb[1];               \
    स्थिर uint32_t nb2 = dst->nb[2];               \
    स्थिर uint32_t nb3 = dst->nb[3];

अटल अंतरभूत uint32_t calc_block_आकार(काष्ठा htp_द्विआधारी_context * bctx, uint32_t ir, uint32_t end_row, uint32_t ne01, uint32_t ne02) {
    uint32_t i03, i02, i01, rem;
    i03 = fastdiv(ir, &bctx->src0_dim12_div);
    rem = ir - i03 * (ne02 * ne01);
    i02 = fastdiv(rem, &bctx->src0_dim1_div);
    i01 = rem - i02 * ne01;

    uint32_t rows_बायाँ = end_row - ir;
    uint32_t block_limit = rows_बायाँ;

    अगर (bctx->split_at_ne01) {
        block_limit = MIN(block_limit, ne01 - i01);
    }
    अगर (bctx->split_at_ne02) {
         uint32_t rows_in_plane = (ne02 * ne01) - rem;
         block_limit = MIN(block_limit, rows_in_plane);
    }

    वापस MIN(bctx->block_max, block_limit);
}

// Out-of-line compute micro-kernels

प्रकार व्योम (*compute_scalar_t)(
    uint8_t * dst, स्थिर uint8_t * src0, स्थिर व्योम * s1_table,
    uint32_t cur_i11, uint32_t ne11, uint32_t n_rows,
    माप_प्रकार dst_stride, माप_प्रकार src0_stride, uint32_t ne00);

#घोषणा DEFINE_COMPUTE_SCALAR(NAME, TYPE, HVX_STMT)             \
अटल व्योम compute_scalar_##NAME(                              \
    uint8_t * dst, स्थिर uint8_t * src0, स्थिर व्योम * s1_table, \
    uint32_t cur_i11, uint32_t ne11, uint32_t n_rows,           \
    माप_प्रकार dst_stride, माप_प्रकार src0_stride, uint32_t ne00) {     \
    स्थिर TYPE * table = (स्थिर TYPE *) s1_table;               \
    क्रम (uint32_t r = 0; r < n_rows; r++) {                     \
        uint8_t * r_dst = dst + r * dst_stride;                 \
        स्थिर uint8_t * r_src0 = src0 + r * src0_stride;        \
        TYPE val = table[cur_i11];                              \
        HVX_STMT;                                               \
        अगर (ne11 > 1 && ++cur_i11 == ne11) {                    \
            cur_i11 = 0;                                        \
        }                                                       \
    }                                                           \
}

DEFINE_COMPUTE_SCALAR(add_f32, भग्न,    hvx_add_scalar_f32_aa(r_dst, r_src0, val, ne00))
DEFINE_COMPUTE_SCALAR(add_f16, _Float16, hvx_add_scalar_f16_aa(r_dst, r_src0, val, ne00))
DEFINE_COMPUTE_SCALAR(sub_f32, भग्न,    hvx_sub_scalar_f32_aa(r_dst, r_src0, val, ne00))
DEFINE_COMPUTE_SCALAR(sub_f16, _Float16, hvx_sub_scalar_f16_aa(r_dst, r_src0, val, ne00))
DEFINE_COMPUTE_SCALAR(mul_f32, भग्न,    hvx_mul_scalar_f32_aa(r_dst, r_src0, val, ne00))
DEFINE_COMPUTE_SCALAR(mul_f16, _Float16, hvx_mul_scalar_f16_aa(r_dst, r_src0, val, ne00))
DEFINE_COMPUTE_SCALAR(div_f32, भग्न,    hvx_mul_scalar_f32_aa(r_dst, r_src0, 1.0f / (val), ne00))
DEFINE_COMPUTE_SCALAR(div_f16, _Float16, hvx_div_scalar_f16_aa(r_dst, r_src0, val, ne00))

प्रकार व्योम (*compute_same_shape_t)(
    uint8_t * dst, स्थिर uint8_t * src0, स्थिर uint8_t * src1,
    uint32_t n_rows, माप_प्रकार dst_stride, माप_प्रकार src0_stride, माप_प्रकार src1_stride, uint32_t ne00);

#घोषणा DEFINE_COMPUTE_SAME_SHAPE(NAME, HVX_FN)                                                  \
अटल व्योम compute_same_shape_##NAME(                                                           \
    uint8_t * dst, स्थिर uint8_t * src0, स्थिर uint8_t * src1,                                   \
    uint32_t n_rows, माप_प्रकार dst_stride, माप_प्रकार src0_stride, माप_प्रकार src1_stride, uint32_t ne00) { \
    क्रम (uint32_t r = 0; r < n_rows; r++) {                                                      \
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

प्रकार व्योम (*compute_row_bcast_t)(
    uint8_t * dst, स्थिर uint8_t * src0, स्थिर uint8_t * src1,
    uint32_t n_rows, माप_प्रकार dst_stride, माप_प्रकार src0_stride, uint32_t ne00);

#घोषणा DEFINE_COMPUTE_ROW_BCAST(NAME, HVX_FN)                                               \
अटल व्योम compute_row_bcast_##NAME(                                                        \
    uint8_t * dst, स्थिर uint8_t * src0, स्थिर uint8_t * src1,                               \
    uint32_t n_rows, माप_प्रकार dst_stride, माप_प्रकार src0_stride, uint32_t ne00) {                 \
    क्रम (uint32_t r = 0; r < n_rows; r++) {                                                  \
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

प्रकार व्योम (*compute_complex_t)(
    uint8_t * dst, स्थिर uint8_t * src0, स्थिर uint8_t * src1_plane,
    uint32_t i01, uint32_t ne11, स्थिर काष्ठा fastdiv_मूल्यावली * div11, uint32_t src1_stride,
    uint32_t n_rows, माप_प्रकार dst_stride, माप_प्रकार src0_stride, uint32_t ne00);

#घोषणा DEFINE_COMPUTE_COMPLEX(NAME, HVX_FN)                                                 \
अटल व्योम compute_complex_##NAME(                                                          \
    uint8_t * dst, स्थिर uint8_t * src0, स्थिर uint8_t * src1_plane,                         \
    uint32_t i01, uint32_t ne11, स्थिर काष्ठा fastdiv_मूल्यावली * div11, uint32_t src1_stride,  \
    uint32_t n_rows, माप_प्रकार dst_stride, माप_प्रकार src0_stride, uint32_t ne00) {                 \
    क्रम (uint32_t r = 0; r < n_rows; r++) {                                                  \
        uint32_t i11 = fastmodulo(i01 + r, ne11, div11);                                     \
        स्थिर uint8_t * r_src1 = src1_plane + i11 * src1_stride;                             \
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

प्रकार व्योम (*compute_repeat_t)(
    uint8_t * dst, स्थिर uint8_t * src0, स्थिर uint8_t * src1_plane,
    uint32_t i01, uint32_t ne11, स्थिर काष्ठा fastdiv_मूल्यावली * div11, uint32_t src1_stride,
    uint32_t n_rows, माप_प्रकार dst_stride, माप_प्रकार src0_stride, uint32_t ne00, uint32_t ne10);

#घोषणा DEFINE_COMPUTE_REPEAT(NAME, TYPE, HVX_FN)                                            \
अटल व्योम compute_repeat_##NAME(                                                           \
    uint8_t * dst, स्थिर uint8_t * src0, स्थिर uint8_t * src1_plane,                         \
    uint32_t i01, uint32_t ne11, स्थिर काष्ठा fastdiv_मूल्यावली * div11, uint32_t src1_stride,  \
    uint32_t n_rows, माप_प्रकार dst_stride, माप_प्रकार src0_stride, uint32_t ne00, uint32_t ne10) {  \
    क्रम (uint32_t r = 0; r < n_rows; r++) {                                                  \
        uint32_t i11 = fastmodulo(i01 + r, ne11, div11);                                     \
        स्थिर uint8_t * r_src1_row = src1_plane + i11 * src1_stride;                         \
        uint8_t * r_dst = dst + r * dst_stride;                                              \
        स्थिर uint8_t * r_src0 = src0 + r * src0_stride;                                     \
        क्रम (uint32_t c = 0; c < ne00; c += ne10) {                                          \
            uint32_t len = MIN(ne10, ne00 - c);                                              \
            HVX_FN(r_dst + c * माप(TYPE), r_src0 + c * माप(TYPE), r_src1_row, len);    \
        }                                                                                    \
    }                                                                                        \
}

DEFINE_COMPUTE_REPEAT(add_f32, भग्न,    hvx_add_f32_uuu)
DEFINE_COMPUTE_REPEAT(add_f16, _Float16, hvx_add_f16_uuu)
DEFINE_COMPUTE_REPEAT(sub_f32, भग्न,    hvx_sub_f32_uuu)
DEFINE_COMPUTE_REPEAT(sub_f16, _Float16, hvx_sub_f16_uuu)
DEFINE_COMPUTE_REPEAT(mul_f32, भग्न,    hvx_mul_f32_uuu)
DEFINE_COMPUTE_REPEAT(mul_f16, _Float16, hvx_mul_f16_uuu)
DEFINE_COMPUTE_REPEAT(div_f32, भग्न,    hvx_div_f32_uuu)
DEFINE_COMPUTE_REPEAT(div_f16, _Float16, hvx_div_f16_uuu)

प्रकार व्योम (*compute_add_id_t)(
    uint8_t * dst, स्थिर uint8_t * src0, स्थिर uint8_t * src1_तथ्य, स्थिर int32_t * src2_ids,
    uint32_t i01, uint32_t i02, uint32_t ne01, uint32_t src1_stride,
    uint32_t n_rows, माप_प्रकार dst_stride, माप_प्रकार src0_stride, uint32_t ne00);

अटल व्योम compute_add_id_f32(
    uint8_t * dst, स्थिर uint8_t * src0, स्थिर uint8_t * src1_तथ्य, स्थिर int32_t * src2_ids,
    uint32_t i01, uint32_t i02, uint32_t ne01, uint32_t src1_stride,
    uint32_t n_rows, माप_प्रकार dst_stride, माप_प्रकार src0_stride, uint32_t ne00) {
    क्रम (uint32_t r = 0; r < n_rows; r++) {
        uint32_t r_i01 = i01 + r;
        स्थिर int32_t idx = src2_ids[i02 * ne01 + r_i01];
        अगर (idx < 0) {
            स_नकल(dst + r * dst_stride, src0 + r * src0_stride, ne00 * माप(भग्न));
            जारी;
        }
        स्थिर uint8_t * r_src1 = src1_तथ्य + idx * src1_stride;
        स्थिर uint8_t * r_src0 = src0 + r * src0_stride;
        uint8_t * r_dst = dst + r * dst_stride;
        hvx_add_f32_aaa(r_dst, r_src0, r_src1, ne00);
    }
}

// 1. Scalar src1 in VTCM via DMA (ne10 == 1)
अटल व्योम द्विआधारी_thread_scalar(अचिन्हित पूर्णांक nth, अचिन्हित पूर्णांक ith, व्योम * तथ्य) {
    काष्ठा htp_द्विआधारी_context * bctx = (काष्ठा htp_द्विआधारी_context *) तथ्य;
    काष्ठा htp_ops_context * octx = bctx->octx;
    htp_द्विआधारी_preamble;

    स्थिर uint32_t elem_आकार = (src0->type == HTP_TYPE_F32) ? माप(भग्न) : माप(_Float16);
    स्थिर माप_प्रकार plane_stride = (माप_प्रकार) ne12 * ne11 * elem_आकार;
    स्थिर माप_प्रकार row_stride   = (माप_प्रकार) ne11 * elem_आकार;

    स्थिर uint32_t row_आकार_bytes = bctx->row_आकार_bytes;
    स्थिर uint32_t start_row = bctx->row_start + bctx->nrows_per_thread * ith;
    स्थिर uint32_t end_row   = MIN(start_row + bctx->nrows_per_thread, bctx->row_start + bctx->योग_rows);
    अगर (start_row >= end_row) वापस;

    FARF(ऊपरी_2, "binary-scalar: %d/%d (%u:%u) row-size %u (%u)",
         ith, nth, start_row, end_row, nb01, bctx->dst_row_आकार_aligned);

    स्थिर काष्ठा htp_द्विआधारी_vtcm_layout * layout = &bctx->vtcm_layout;
    uint8_t * src0_spad_base = VTCM_LAYOUT_PTR(uint8_t, bctx->vtcm_base, layout->off_src0) + (ith * layout->src0_bytes_per_thread);
    uint8_t * dst_spad_base  = VTCM_LAYOUT_PTR(uint8_t, bctx->vtcm_base, layout->off_dst)  + (ith * layout->dst_bytes_per_thread);
    माप_प्रकार src0_spad_half    = layout->src0_spad_half_आकार;
    माप_प्रकार dst_spad_half     = layout->dst_spad_half_आकार;
    स्थिर व्योम * s1_table    = VTCM_LAYOUT_PTR(स्थिर व्योम, bctx->vtcm_base, layout->off_src1);

    dma_queue * dma_q = octx->ctx->dma[ith];
    uint32_t ir_prefetch = start_row;
    पूर्णांक spad_idx = 0;

    क्रम (पूर्णांक k = 0; k < 2 && ir_prefetch < end_row; k++) {
        uint32_t current_block_आकार = calc_block_आकार(bctx, ir_prefetch, end_row, ne01, ne02);
        uint32_t i03, i02, i01, rem;
        i03 = fastdiv(ir_prefetch, &bctx->src0_dim12_div);
        rem = ir_prefetch - i03 * (ne02 * ne01);
        i02 = fastdiv(rem, &bctx->src0_dim1_div);
        i01 = rem - i02 * ne01;

        dma_addr_t src0_curr = src0->तथ्य + i03 * nb03 + i02 * nb02 + i01 * nb01;
        dma_addr_t dst_curr  = dst->तथ्य  + i03 * nb3  + i02 * nb2  + i01 * nb1;

        uint8_t * s0_spad = src0_spad_base + spad_idx * src0_spad_half;
        uint8_t * d_spad  = dst_spad_base  + spad_idx * dst_spad_half;

        dma_queue_धकेलो(dma_q, dma_make_तथ्य(dst_curr, d_spad), nb1, bctx->dst_row_आकार_aligned, row_आकार_bytes, 0);
        dma_queue_धकेलो(dma_q, dma_make_तथ्य(s0_spad, src0_curr), bctx->src0_row_आकार_aligned, nb01, row_आकार_bytes,
                                                             current_block_आकार);
        ir_prefetch += current_block_आकार;
        spad_idx ^= 1;
    }

    काष्ठा htp_thread_trace * tr = &octx->ctx->trace[ith];
    compute_scalar_t compute = (compute_scalar_t) bctx->compute;

    क्रम (uint32_t ir = start_row; ir < end_row; ) {
        uint32_t current_block_आकार = calc_block_आकार(bctx, ir, end_row, ne01, ne02);

        uint8_t * d_spad  = (uint8_t *) dma_queue_निकालो(dma_q).src;
        uint8_t * s0_spad = (uint8_t *) dma_queue_निकालो(dma_q).dst;

        uint32_t i03, i02, i01, rem;
        i03 = fastdiv(ir, &bctx->src0_dim12_div);
        rem = ir - i03 * (ne02 * ne01);
        i02 = fastdiv(rem, &bctx->src0_dim1_div);
        i01 = rem - i02 * ne01;

        uint32_t i13 = fastmodulo(i03, ne13, &bctx->src1_dim3_div);
        uint32_t i12 = fastmodulo(i02, ne12, &bctx->src1_dim2_div);
        uint32_t i11 = fastmodulo(i01, ne11, &bctx->src1_dim1_div);
        स्थिर uint8_t * scalar_table = (स्थिर uint8_t *) s1_table +
            i13 * plane_stride + i12 * row_stride;

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);
        compute(d_spad, s0_spad, scalar_table, i11, ne11, current_block_आकार,
                bctx->dst_row_आकार_aligned, bctx->src0_row_आकार_aligned, ne00);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);

        dma_addr_t dst_curr = dst->तथ्य + i03 * nb3 + i02 * nb2 + i01 * nb1;
        dma_queue_धकेलो(dma_q, dma_make_तथ्य(dst_curr, d_spad), nb1, bctx->dst_row_आकार_aligned, row_आकार_bytes,
                                                                current_block_आकार);

        अगर (ir_prefetch < end_row) {
            uint32_t अग्र_block_आकार = calc_block_आकार(bctx, ir_prefetch, end_row, ne01, ne02);
            uint32_t p03, p02, p01, prem;
            p03 = fastdiv(ir_prefetch, &bctx->src0_dim12_div);
            prem = ir_prefetch - p03 * (ne02 * ne01);
            p02 = fastdiv(prem, &bctx->src0_dim1_div);
            p01 = prem - p02 * ne01;
            dma_addr_t s0_अग्र = src0->तथ्य + p03 * nb03 + p02 * nb02 + p01 * nb01;
            dma_queue_धकेलो(dma_q, dma_make_तथ्य(s0_spad, s0_अग्र), bctx->src0_row_आकार_aligned, nb01, row_आकार_bytes,
                                                               अग्र_block_आकार);
            ir_prefetch += अग्र_block_आकार;
        }
        ir += current_block_आकार;
    }

    dma_queue_flush(dma_q);
}

// 2. Vector Same Shape (ne1x == ne0x) or Simple Broadcast
अटल व्योम द्विआधारी_thread_vector_same_shape(अचिन्हित पूर्णांक nth, अचिन्हित पूर्णांक ith, व्योम * तथ्य) {
    काष्ठा htp_द्विआधारी_context * bctx = (काष्ठा htp_द्विआधारी_context *) तथ्य;
    काष्ठा htp_ops_context * octx = bctx->octx;
    htp_द्विआधारी_preamble;

    स्थिर uint32_t row_आकार_bytes = bctx->row_आकार_bytes;
    स्थिर uint32_t start_row = bctx->row_start + bctx->nrows_per_thread * ith;
    स्थिर uint32_t end_row   = MIN(start_row + bctx->nrows_per_thread, bctx->row_start + bctx->योग_rows);
    अगर (start_row >= end_row) वापस;

    FARF(ऊपरी_2, "binary-same-shape: %d/%d (%u:%u) row-size %u (%u)",
         ith, nth, start_row, end_row, nb01, bctx->dst_row_आकार_aligned);

    स्थिर काष्ठा htp_द्विआधारी_vtcm_layout * layout = &bctx->vtcm_layout;
    uint8_t * src0_spad_base = VTCM_LAYOUT_PTR(uint8_t, bctx->vtcm_base, layout->off_src0) + (ith * layout->src0_bytes_per_thread);
    uint8_t * src1_spad_base = VTCM_LAYOUT_PTR(uint8_t, bctx->vtcm_base, layout->off_src1) + (ith * layout->src1_bytes_per_thread);
    uint8_t * dst_spad_base  = VTCM_LAYOUT_PTR(uint8_t, bctx->vtcm_base, layout->off_dst)  + (ith * layout->dst_bytes_per_thread);

    माप_प्रकार src0_spad_half = layout->src0_spad_half_आकार;
    माप_प्रकार src1_spad_half = layout->src1_spad_half_आकार;
    माप_प्रकार dst_spad_half  = layout->dst_spad_half_आकार;

    dma_queue * dma_q = octx->ctx->dma[ith];
    uint32_t ir_prefetch = start_row;
    पूर्णांक spad_idx = 0;

    क्रम (पूर्णांक k = 0; k < 2 && ir_prefetch < end_row; k++) {
        uint32_t current_block_आकार = calc_block_आकार(bctx, ir_prefetch, end_row, ne01, ne02);
        uint32_t i03, i02, i01, rem;
        i03 = fastdiv(ir_prefetch, &bctx->src0_dim12_div);
        rem = ir_prefetch - i03 * (ne02 * ne01);
        i02 = fastdiv(rem, &bctx->src0_dim1_div);
        i01 = rem - i02 * ne01;

        uint32_t i13 = (ne13 == 1) ? 0 : i03;
        uint32_t i12 = (ne12 == 1) ? 0 : i02;
        uint32_t i11 = (ne11 == 1) ? 0 : i01;

        dma_addr_t src0_curr = src0->तथ्य + i03 * nb03 + i02 * nb02 + i01 * nb01;
        dma_addr_t src1_curr = src1->तथ्य + i13 * nb13 + i12 * nb12 + i11 * nb11;
        dma_addr_t dst_curr  = dst->तथ्य  + i03 * nb3  + i02 * nb2  + i01 * nb1;

        uint8_t * s0_spad = src0_spad_base + spad_idx * src0_spad_half;
        uint8_t * s1_spad = src1_spad_base + spad_idx * src1_spad_half;
        uint8_t * d_spad  = dst_spad_base  + spad_idx * dst_spad_half;

        dma_queue_धकेलो(dma_q, dma_make_तथ्य(dst_curr, d_spad), nb1, bctx->dst_row_आकार_aligned, row_आकार_bytes, 0);
        dma_queue_धकेलो(dma_q, dma_make_तथ्य(s0_spad, src0_curr), bctx->src0_row_आकार_aligned, nb01, row_आकार_bytes, current_block_आकार);
        dma_queue_धकेलो(dma_q, dma_make_तथ्य(s1_spad, src1_curr), bctx->src1_row_आकार_aligned, nb11, row_आकार_bytes, current_block_आकार);
        ir_prefetch += current_block_आकार;
        spad_idx ^= 1;
    }

    काष्ठा htp_thread_trace * tr = &octx->ctx->trace[ith];
    compute_same_shape_t compute = (compute_same_shape_t) bctx->compute;

    क्रम (uint32_t ir = start_row; ir < end_row; ) {
        uint32_t current_block_आकार = calc_block_आकार(bctx, ir, end_row, ne01, ne02);
        uint8_t * d_spad  = (uint8_t *) dma_queue_निकालो(dma_q).src;
        uint8_t * s0_spad = (uint8_t *) dma_queue_निकालो(dma_q).dst;
        uint8_t * s1_spad = (uint8_t *) dma_queue_निकालो(dma_q).dst;

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);
        compute(d_spad, s0_spad, s1_spad, current_block_आकार,
                bctx->dst_row_आकार_aligned, bctx->src0_row_आकार_aligned, bctx->src1_row_आकार_aligned, ne00);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);

        uint32_t i03, i02, i01, rem;
        i03 = fastdiv(ir, &bctx->src0_dim12_div);
        rem = ir - i03 * (ne02 * ne01);
        i02 = fastdiv(rem, &bctx->src0_dim1_div);
        i01 = rem - i02 * ne01;
        dma_addr_t dst_curr = dst->तथ्य + i03 * nb3 + i02 * nb2 + i01 * nb1;
        dma_queue_धकेलो(dma_q, dma_make_तथ्य(dst_curr, d_spad), nb1, bctx->dst_row_आकार_aligned, row_आकार_bytes, current_block_आकार);

        अगर (ir_prefetch < end_row) {
            uint32_t अग्र_block_आकार = calc_block_आकार(bctx, ir_prefetch, end_row, ne01, ne02);
            uint32_t p03, p02, p01, prem;
            p03 = fastdiv(ir_prefetch, &bctx->src0_dim12_div);
            prem = ir_prefetch - p03 * (ne02 * ne01);
            p02 = fastdiv(prem, &bctx->src0_dim1_div);
            p01 = prem - p02 * ne01;

            uint32_t p13 = (ne13 == 1) ? 0 : p03;
            uint32_t p12 = (ne12 == 1) ? 0 : p02;
            uint32_t p11 = (ne11 == 1) ? 0 : p01;

            dma_addr_t s0_अग्र = src0->तथ्य + p03 * nb03 + p02 * nb02 + p01 * nb01;
            dma_addr_t s1_अग्र = src1->तथ्य + p13 * nb13 + p12 * nb12 + p11 * nb11;

            dma_queue_धकेलो(dma_q, dma_make_तथ्य(s0_spad, s0_अग्र), bctx->src0_row_आकार_aligned, nb01, row_आकार_bytes, अग्र_block_आकार);
            dma_queue_धकेलो(dma_q, dma_make_तथ्य(s1_spad, s1_अग्र), bctx->src1_row_आकार_aligned, nb11, row_आकार_bytes, अग्र_block_आकार);

            ir_prefetch += अग्र_block_आकार;
        }
        ir += current_block_आकार;
    }

    dma_queue_flush(dma_q);
}

// 3. Row Broadcast (ne11 == 1, ne12 == 1, single row src1)
अटल व्योम द्विआधारी_thread_vector_row_broadcast(अचिन्हित पूर्णांक nth, अचिन्हित पूर्णांक ith, व्योम * तथ्य) {
    काष्ठा htp_द्विआधारी_context * bctx = (काष्ठा htp_द्विआधारी_context *) तथ्य;
    काष्ठा htp_ops_context * octx = bctx->octx;
    htp_द्विआधारी_preamble;

    स्थिर uint32_t row_आकार_bytes = bctx->row_आकार_bytes;
    स्थिर uint32_t start_row  = bctx->row_start + bctx->nrows_per_thread * ith;
    स्थिर uint32_t end_row    = MIN(start_row + bctx->nrows_per_thread, bctx->row_start + bctx->योग_rows);
    अगर (start_row >= end_row) वापस;

    FARF(ऊपरी_2, "binary-row-bcast: %d/%d (%u:%u) row-size %u (%u)",
         ith, nth, start_row, end_row, nb01, bctx->dst_row_आकार_aligned);

    स्थिर काष्ठा htp_द्विआधारी_vtcm_layout * layout = &bctx->vtcm_layout;
    uint8_t * src0_spad_base = VTCM_LAYOUT_PTR(uint8_t, bctx->vtcm_base, layout->off_src0) + (ith * layout->src0_bytes_per_thread);
    uint8_t * dst_spad_base  = VTCM_LAYOUT_PTR(uint8_t, bctx->vtcm_base, layout->off_dst)  + (ith * layout->dst_bytes_per_thread);

    माप_प्रकार src0_spad_half = layout->src0_spad_half_आकार;
    माप_प्रकार dst_spad_half  = layout->dst_spad_half_आकार;

    dma_queue * dma_q = octx->ctx->dma[ith];
    uint32_t ir_prefetch = start_row;
    पूर्णांक spad_idx = 0;

    व्योम * s1_ptr = VTCM_LAYOUT_PTR(व्योम, bctx->vtcm_base, layout->off_src1);

    क्रम (पूर्णांक k = 0; k < 2 && ir_prefetch < end_row; k++) {
        uint32_t current_block_आकार = calc_block_आकार(bctx, ir_prefetch, end_row, ne01, ne02);
        uint32_t i03 = fastdiv(ir_prefetch, &bctx->src0_dim12_div);
        uint32_t rem = ir_prefetch - i03 * (ne02 * ne01);
        uint32_t i02 = fastdiv(rem, &bctx->src0_dim1_div);
        uint32_t i01 = rem - i02 * ne01;

        dma_addr_t src0_curr = src0->तथ्य + i03 * nb03 + i02 * nb02 + i01 * nb01;
        dma_addr_t dst_curr  = dst->तथ्य  + i03 * nb3  + i02 * nb2  + i01 * nb1;

        uint8_t * s0_spad = src0_spad_base + spad_idx * src0_spad_half;
        uint8_t * d_spad  = dst_spad_base  + spad_idx * dst_spad_half;

        dma_queue_धकेलो(dma_q, dma_make_तथ्य(dst_curr, d_spad), nb1, bctx->dst_row_आकार_aligned, row_आकार_bytes, 0);
        dma_queue_धकेलो(dma_q, dma_make_तथ्य(s0_spad, src0_curr), bctx->src0_row_आकार_aligned, nb01, row_आकार_bytes, current_block_आकार);
        ir_prefetch += current_block_आकार;
        spad_idx ^= 1;
    }

    काष्ठा htp_thread_trace * tr = &octx->ctx->trace[ith];
    compute_row_bcast_t compute = (compute_row_bcast_t) bctx->compute;

    क्रम (uint32_t ir = start_row; ir < end_row; ) {
        uint32_t current_block_आकार = calc_block_आकार(bctx, ir, end_row, ne01, ne02);
        uint8_t * d_spad  = (uint8_t *) dma_queue_निकालो(dma_q).src;
        uint8_t * s0_spad = (uint8_t *) dma_queue_निकालो(dma_q).dst;

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);
        compute(d_spad, s0_spad, (स्थिर uint8_t *)s1_ptr, current_block_आकार,
                bctx->dst_row_आकार_aligned, bctx->src0_row_आकार_aligned, ne00);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);

        uint32_t i03 = fastdiv(ir, &bctx->src0_dim12_div);
        uint32_t rem = ir - i03 * (ne02 * ne01);
        uint32_t i02 = fastdiv(rem, &bctx->src0_dim1_div);
        uint32_t i01 = rem - i02 * ne01;
        dma_addr_t dst_curr = dst->तथ्य + i03 * nb3 + i02 * nb2 + i01 * nb1;
        dma_queue_धकेलो(dma_q, dma_make_तथ्य(dst_curr, d_spad), nb1, bctx->dst_row_आकार_aligned, row_आकार_bytes, current_block_आकार);

        अगर (ir_prefetch < end_row) {
            uint32_t अग्र_block_आकार = calc_block_आकार(bctx, ir_prefetch, end_row, ne01, ne02);
            uint32_t p03, p02, p01, prem;
            p03  = fastdiv(ir_prefetch, &bctx->src0_dim12_div);
            prem = ir_prefetch - p03 * (ne02 * ne01);
            p02  = fastdiv(prem, &bctx->src0_dim1_div);
            p01  = prem - p02 * ne01;
            dma_addr_t s0_अग्र = src0->तथ्य + p03 * nb03 + p02 * nb02 + p01 * nb01;
            dma_queue_धकेलो(dma_q, dma_make_तथ्य(s0_spad, s0_अग्र), bctx->src0_row_आकार_aligned, nb01, row_आकार_bytes, अग्र_block_आकार);
            ir_prefetch += अग्र_block_आकार;
        }
        ir += current_block_आकार;
    }

    dma_queue_flush(dma_q);
}

// 4. Vector Complex (ne10 == ne00, complex broadcast)
अटल व्योम द्विआधारी_thread_vector_complex(अचिन्हित पूर्णांक nth, अचिन्हित पूर्णांक ith, व्योम * तथ्य) {
    काष्ठा htp_द्विआधारी_context * bctx = (काष्ठा htp_द्विआधारी_context *) तथ्य;
    काष्ठा htp_ops_context * octx = bctx->octx;
    htp_द्विआधारी_preamble;

    स्थिर uint32_t row_आकार_bytes = bctx->row_आकार_bytes;
    स्थिर uint32_t start_row  = bctx->row_start + bctx->nrows_per_thread * ith;
    स्थिर uint32_t end_row    = MIN(start_row + bctx->nrows_per_thread, bctx->row_start + bctx->योग_rows);
    अगर (start_row >= end_row) वापस;

    FARF(ऊपरी_2, "binary-complex: %d/%d (%u:%u) row-size %u (%u)",
         ith, nth, start_row, end_row, nb01, bctx->dst_row_आकार_aligned);

    स्थिर काष्ठा htp_द्विआधारी_vtcm_layout * layout = &bctx->vtcm_layout;
    uint8_t * src0_spad_base = VTCM_LAYOUT_PTR(uint8_t, bctx->vtcm_base, layout->off_src0) + (ith * layout->src0_bytes_per_thread);
    uint8_t * dst_spad_base  = VTCM_LAYOUT_PTR(uint8_t, bctx->vtcm_base, layout->off_dst)  + (ith * layout->dst_bytes_per_thread);
    माप_प्रकार src0_spad_half    = layout->src0_spad_half_आकार;
    माप_प्रकार dst_spad_half     = layout->dst_spad_half_आकार;
    स्थिर uint8_t * vtcm_src1 = VTCM_LAYOUT_PTR(स्थिर uint8_t, bctx->vtcm_base, layout->off_src1);

    स्थिर uint32_t elem_आकार = (src0->type == HTP_TYPE_F32) ? माप(भग्न) : माप(_Float16);
    स्थिर माप_प्रकार vtcm_row_stride   = (माप_प्रकार) ne10 * elem_आकार;
    स्थिर माप_प्रकार vtcm_plane_stride = (माप_प्रकार) ne11 * vtcm_row_stride;
    स्थिर माप_प्रकार vtcm_vol_stride   = (माप_प्रकार) ne12 * vtcm_plane_stride;

    dma_queue * dma_q = octx->ctx->dma[ith];
    uint32_t ir_prefetch = start_row;
    पूर्णांक spad_idx = 0;

    क्रम (पूर्णांक k = 0; k < 2 && ir_prefetch < end_row; k++) {
        uint32_t current_block_आकार = calc_block_आकार(bctx, ir_prefetch, end_row, ne01, ne02);
        uint32_t i03 = fastdiv(ir_prefetch, &bctx->src0_dim12_div);
        uint32_t rem = ir_prefetch - i03 * (ne02 * ne01);
        uint32_t i02 = fastdiv(rem, &bctx->src0_dim1_div);
        uint32_t i01 = rem - i02 * ne01;

        dma_addr_t src0_curr = src0->तथ्य + i03 * nb03 + i02 * nb02 + i01 * nb01;
        dma_addr_t dst_curr  = dst->तथ्य  + i03 * nb3  + i02 * nb2  + i01 * nb1;

        uint8_t * s0_spad = src0_spad_base + spad_idx * src0_spad_half;
        uint8_t * d_spad  = dst_spad_base  + spad_idx * dst_spad_half;

        dma_queue_धकेलो(dma_q, dma_make_तथ्य(dst_curr, d_spad), nb1, bctx->dst_row_आकार_aligned, row_आकार_bytes, 0);
        dma_queue_धकेलो(dma_q, dma_make_तथ्य(s0_spad, src0_curr), bctx->src0_row_आकार_aligned, nb01, row_आकार_bytes, current_block_आकार);
        ir_prefetch += current_block_आकार;
        spad_idx ^= 1;
    }

    काष्ठा htp_thread_trace * tr = &octx->ctx->trace[ith];
    compute_complex_t compute = (compute_complex_t) bctx->compute;

    क्रम (uint32_t ir = start_row; ir < end_row; ) {
        uint32_t current_block_आकार = calc_block_आकार(bctx, ir, end_row, ne01, ne02);
        uint8_t * d_spad = (uint8_t *) dma_queue_निकालो(dma_q).src;
        uint8_t * s0_spad = (uint8_t *) dma_queue_निकालो(dma_q).dst;

        uint32_t i03 = fastdiv(ir, &bctx->src0_dim12_div);
        uint32_t rem = ir - i03 * (ne02 * ne01);
        uint32_t i02 = fastdiv(rem, &bctx->src0_dim1_div);
        uint32_t i01 = rem - i02 * ne01;

        uint32_t i13 = fastmodulo(i03, ne13, &bctx->src1_dim3_div);
        uint32_t i12 = fastmodulo(i02, ne12, &bctx->src1_dim2_div);
        स्थिर uint8_t * src1_plane = vtcm_src1 + i13 * vtcm_vol_stride + i12 * vtcm_plane_stride;

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);
        compute(d_spad, s0_spad, src1_plane, i01, ne11, &bctx->src1_dim1_div, vtcm_row_stride,
                current_block_आकार, bctx->dst_row_आकार_aligned, bctx->src0_row_आकार_aligned, ne00);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);

        dma_addr_t dst_curr = dst->तथ्य + i03 * nb3 + i02 * nb2 + i01 * nb1;
        dma_queue_धकेलो(dma_q, dma_make_तथ्य(dst_curr, d_spad), nb1, bctx->dst_row_आकार_aligned, row_आकार_bytes, current_block_आकार);

        अगर (ir_prefetch < end_row) {
            uint32_t अग्र_block_आकार = calc_block_आकार(bctx, ir_prefetch, end_row, ne01, ne02);
            uint32_t p03, p02, p01, prem;
            p03  = fastdiv(ir_prefetch, &bctx->src0_dim12_div);
            prem = ir_prefetch - p03 * (ne02 * ne01);
            p02  = fastdiv(prem, &bctx->src0_dim1_div);
            p01  = prem - p02 * ne01;
            dma_addr_t s0_अग्र = src0->तथ्य + p03 * nb03 + p02 * nb02 + p01 * nb01;
            dma_queue_धकेलो(dma_q, dma_make_तथ्य(s0_spad, s0_अग्र), bctx->src0_row_आकार_aligned, nb01, row_आकार_bytes, अग्र_block_आकार);
            ir_prefetch += अग्र_block_आकार;
        }
        ir += current_block_आकार;
    }

    dma_queue_flush(dma_q);
}

// 5. Element Repeat (ne10 != ne00)
अटल व्योम द्विआधारी_thread_element_repeat(अचिन्हित पूर्णांक nth, अचिन्हित पूर्णांक ith, व्योम * तथ्य) {
    काष्ठा htp_द्विआधारी_context * bctx = (काष्ठा htp_द्विआधारी_context *) तथ्य;
    काष्ठा htp_ops_context * octx = bctx->octx;
    htp_द्विआधारी_preamble;

    स्थिर uint32_t row_आकार_bytes = bctx->row_आकार_bytes;
    स्थिर uint32_t start_row  = bctx->row_start + bctx->nrows_per_thread * ith;
    स्थिर uint32_t end_row    = MIN(start_row + bctx->nrows_per_thread, bctx->row_start + bctx->योग_rows);
    अगर (start_row >= end_row) वापस;

    स्थिर काष्ठा htp_द्विआधारी_vtcm_layout * layout = &bctx->vtcm_layout;
    uint8_t * src0_spad_base = VTCM_LAYOUT_PTR(uint8_t, bctx->vtcm_base, layout->off_src0) + (ith * layout->src0_bytes_per_thread);
    uint8_t * dst_spad_base  = VTCM_LAYOUT_PTR(uint8_t, bctx->vtcm_base, layout->off_dst)  + (ith * layout->dst_bytes_per_thread);
    माप_प्रकार src0_spad_half    = layout->src0_spad_half_आकार;
    माप_प्रकार dst_spad_half     = layout->dst_spad_half_आकार;
    स्थिर uint8_t * vtcm_src1 = VTCM_LAYOUT_PTR(स्थिर uint8_t, bctx->vtcm_base, layout->off_src1);

    स्थिर uint32_t elem_आकार = (src0->type == HTP_TYPE_F32) ? माप(भग्न) : माप(_Float16);
    स्थिर माप_प्रकार vtcm_row_stride   = (माप_प्रकार) ne10 * elem_आकार;
    स्थिर माप_प्रकार vtcm_plane_stride = (माप_प्रकार) ne11 * vtcm_row_stride;
    स्थिर माप_प्रकार vtcm_vol_stride   = (माप_प्रकार) ne12 * vtcm_plane_stride;

    FARF(ऊपरी_2, "binary-repeat: %d/%d (%u:%u) row-size %u (%u)",
         ith, nth, start_row, end_row, nb01, bctx->dst_row_आकार_aligned);

    dma_queue * dma_q = octx->ctx->dma[ith];
    uint32_t ir_prefetch = start_row;
    पूर्णांक spad_idx = 0;

    क्रम (पूर्णांक k = 0; k < 2 && ir_prefetch < end_row; k++) {
        uint32_t current_block_आकार = calc_block_आकार(bctx, ir_prefetch, end_row, ne01, ne02);
        uint32_t i03 = fastdiv(ir_prefetch, &bctx->src0_dim12_div);
        uint32_t rem = ir_prefetch - i03 * (ne02 * ne01);
        uint32_t i02 = fastdiv(rem, &bctx->src0_dim1_div);
        uint32_t i01 = rem - i02 * ne01;

        dma_addr_t src0_curr = src0->तथ्य + i03 * nb03 + i02 * nb02 + i01 * nb01;
        dma_addr_t dst_curr  = dst->तथ्य  + i03 * nb3  + i02 * nb2  + i01 * nb1;

        uint8_t * s0_spad = src0_spad_base + spad_idx * src0_spad_half;
        uint8_t * d_spad  = dst_spad_base  + spad_idx * dst_spad_half;

        dma_queue_धकेलो(dma_q, dma_make_तथ्य(dst_curr, d_spad), nb1, bctx->dst_row_आकार_aligned, row_आकार_bytes, 0);
        dma_queue_धकेलो(dma_q, dma_make_तथ्य(s0_spad, src0_curr), bctx->src0_row_आकार_aligned, nb01, row_आकार_bytes, current_block_आकार);
        ir_prefetch += current_block_आकार;
        spad_idx ^= 1;
    }

    काष्ठा htp_thread_trace * tr = &octx->ctx->trace[ith];
    compute_repeat_t compute = (compute_repeat_t) bctx->compute;

    क्रम (uint32_t ir = start_row; ir < end_row; ) {
        uint32_t current_block_आकार = calc_block_आकार(bctx, ir, end_row, ne01, ne02);
        uint8_t * d_spad  = (uint8_t *) dma_queue_निकालो(dma_q).src;
        uint8_t * s0_spad = (uint8_t *) dma_queue_निकालो(dma_q).dst;

        uint32_t i03 = fastdiv(ir, &bctx->src0_dim12_div);
        uint32_t rem = ir - i03 * (ne02 * ne01);
        uint32_t i02 = fastdiv(rem, &bctx->src0_dim1_div);
        uint32_t i01 = rem - i02 * ne01;

        uint32_t i13 = fastmodulo(i03, ne13, &bctx->src1_dim3_div);
        uint32_t i12 = fastmodulo(i02, ne12, &bctx->src1_dim2_div);
        स्थिर uint8_t * src1_plane = vtcm_src1 + i13 * vtcm_vol_stride + i12 * vtcm_plane_stride;

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);
        compute(d_spad, s0_spad, src1_plane, i01, ne11, &bctx->src1_dim1_div, vtcm_row_stride,
                current_block_आकार, bctx->dst_row_आकार_aligned, bctx->src0_row_आकार_aligned, ne00, ne10);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);

        dma_addr_t dst_curr = dst->तथ्य + i03 * nb3 + i02 * nb2 + i01 * nb1;
        dma_queue_धकेलो(dma_q, dma_make_तथ्य(dst_curr, d_spad), nb1, bctx->dst_row_आकार_aligned, row_आकार_bytes, current_block_आकार);

        अगर (ir_prefetch < end_row) {
            uint32_t अग्र_block_आकार = calc_block_आकार(bctx, ir_prefetch, end_row, ne01, ne02);
            uint32_t p03, p02, p01, prem;
            p03  = fastdiv(ir_prefetch, &bctx->src0_dim12_div);
            prem = ir_prefetch - p03 * (ne02 * ne01);
            p02  = fastdiv(prem, &bctx->src0_dim1_div);
            p01  = prem - p02 * ne01;
            dma_addr_t s0_अग्र = src0->तथ्य + p03 * nb03 + p02 * nb02 + p01 * nb01;
            dma_queue_धकेलो(dma_q, dma_make_तथ्य(s0_spad, s0_अग्र), bctx->src0_row_आकार_aligned, nb01, row_आकार_bytes, अग्र_block_आकार);
            ir_prefetch += अग्र_block_आकार;
        }
        ir += current_block_आकार;
    }

    dma_queue_flush(dma_q);
}

// 6. ADD_ID (src1 gathered via src2 indices)
अटल व्योम द्विआधारी_thread_add_id_f32(अचिन्हित पूर्णांक nth, अचिन्हित पूर्णांक ith, व्योम * तथ्य) {
    काष्ठा htp_द्विआधारी_context * bctx = (काष्ठा htp_द्विआधारी_context *) तथ्य;
    काष्ठा htp_ops_context * octx = bctx->octx;

    स्थिर काष्ठा htp_tensor * src0 = octx->src[0];
    स्थिर काष्ठा htp_tensor * src1 = octx->src[1];
    स्थिर काष्ठा htp_tensor * dst  = octx->dst;

    स्थिर uint32_t ne00 = src0->ne[0];
    स्थिर uint32_t ne01 = src0->ne[1];
    स्थिर uint32_t ne02 = src0->ne[2];
    स्थिर uint32_t ne03 = src0->ne[3];

    स्थिर uint32_t nb01 = src0->nb[1];
    स्थिर uint32_t nb02 = src0->nb[2];
    स्थिर uint32_t nb03 = src0->nb[3];
    स्थिर uint32_t src1_stride = bctx->src1_row_आकार_aligned;

    स्थिर uint32_t nb1 = dst->nb[1];
    स्थिर uint32_t nb2 = dst->nb[2];
    स्थिर uint32_t nb3 = dst->nb[3];

    स्थिर uint32_t row_आकार_bytes = bctx->row_आकार_bytes;
    स्थिर uint32_t start_row = bctx->row_start + bctx->nrows_per_thread * ith;
    स्थिर uint32_t end_row   = MIN(start_row + bctx->nrows_per_thread, bctx->row_start + bctx->योग_rows);
    अगर (start_row >= end_row) वापस;

    स्थिर काष्ठा htp_द्विआधारी_vtcm_layout * layout = &bctx->vtcm_layout;
    uint8_t * src0_spad_base = VTCM_LAYOUT_PTR(uint8_t, bctx->vtcm_base, layout->off_src0) + (ith * layout->src0_bytes_per_thread);
    uint8_t * dst_spad_base  = VTCM_LAYOUT_PTR(uint8_t, bctx->vtcm_base, layout->off_dst)  + (ith * layout->dst_bytes_per_thread);
    स्थिर uint8_t * vtcm_src1 = VTCM_LAYOUT_PTR(स्थिर uint8_t, bctx->vtcm_base, layout->off_src1);
    स्थिर int32_t * vtcm_src2 = VTCM_LAYOUT_PTR(स्थिर int32_t, bctx->vtcm_base, layout->off_src2);
    माप_प्रकार src0_spad_half    = layout->src0_spad_half_आकार;
    माप_प्रकार dst_spad_half     = layout->dst_spad_half_आकार;

    dma_queue * dma_q = octx->ctx->dma[ith];
    uint32_t ir_prefetch = start_row;
    पूर्णांक spad_idx = 0;

    क्रम (पूर्णांक k = 0; k < 2 && ir_prefetch < end_row; k++) {
        uint32_t current_block_आकार = calc_block_आकार(bctx, ir_prefetch, end_row, ne01, ne02);
        uint32_t i03 = fastdiv(ir_prefetch, &bctx->src0_dim12_div);
        uint32_t rem = ir_prefetch - i03 * (ne02 * ne01);
        uint32_t i02 = fastdiv(rem, &bctx->src0_dim1_div);
        uint32_t i01 = rem - i02 * ne01;

        dma_addr_t src0_curr = src0->तथ्य + i03 * nb03 + i02 * nb02 + i01 * nb01;
        dma_addr_t dst_curr  = dst->तथ्य  + i03 * nb3  + i02 * nb2  + i01 * nb1;

        uint8_t * s0_spad = src0_spad_base + spad_idx * src0_spad_half;
        uint8_t * d_spad  = dst_spad_base  + spad_idx * dst_spad_half;

        dma_queue_धकेलो(dma_q, dma_make_तथ्य(dst_curr, d_spad), nb1, bctx->dst_row_आकार_aligned, row_आकार_bytes, 0);
        dma_queue_धकेलो(dma_q, dma_make_तथ्य(s0_spad, src0_curr), bctx->src0_row_आकार_aligned, nb01, row_आकार_bytes, current_block_आकार);
        ir_prefetch += current_block_आकार;
        spad_idx ^= 1;
    }

    काष्ठा htp_thread_trace * tr = &octx->ctx->trace[ith];

    क्रम (uint32_t ir = start_row; ir < end_row; ) {
        uint32_t current_block_आकार = calc_block_आकार(bctx, ir, end_row, ne01, ne02);
        uint8_t * d_spad = (uint8_t *) dma_queue_निकालो(dma_q).src;
        uint8_t * s0_spad = (uint8_t *) dma_queue_निकालो(dma_q).dst;

        uint32_t i03 = fastdiv(ir, &bctx->src0_dim12_div);
        uint32_t rem = ir - i03 * (ne02 * ne01);
        uint32_t i02 = fastdiv(rem, &bctx->src0_dim1_div);
        uint32_t i01 = rem - i02 * ne01;

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);
        compute_add_id_t compute = (compute_add_id_t) bctx->compute;
        compute(d_spad, s0_spad, vtcm_src1, vtcm_src2,
                i01, i02, ne01, src1_stride,
                current_block_आकार, bctx->dst_row_आकार_aligned, bctx->src0_row_आकार_aligned, ne00);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);

        dma_addr_t dst_curr = dst->तथ्य + i03 * nb3 + i02 * nb2 + i01 * nb1;
        dma_queue_धकेलो(dma_q, dma_make_तथ्य(dst_curr, d_spad), nb1, bctx->dst_row_आकार_aligned, row_आकार_bytes, current_block_आकार);

        अगर (ir_prefetch < end_row) {
             uint32_t अग्र_block_आकार = calc_block_आकार(bctx, ir_prefetch, end_row, ne01, ne02);
             uint32_t p03, p02, p01, prem;
             p03  = fastdiv(ir_prefetch, &bctx->src0_dim12_div);
             prem = ir_prefetch - p03 * (ne02 * ne01);
             p02  = fastdiv(prem, &bctx->src0_dim1_div);
             p01  = prem - p02 * ne01;
             dma_addr_t s0_अग्र = src0->तथ्य + p03 * nb03 + p02 * nb02 + p01 * nb01;
             dma_queue_धकेलो(dma_q, dma_make_तथ्य(s0_spad, s0_अग्र), bctx->src0_row_आकार_aligned, nb01, row_आकार_bytes, अग्र_block_आकार);
             ir_prefetch += अग्र_block_आकार;
        }
        ir += current_block_आकार;
    }

    dma_queue_flush(dma_q);
}

अटल अंतरभूत व्योम hvx_div_scalar_f32(uint8_t * restrict dst, स्थिर uint8_t * restrict src, स्थिर भग्न val, स्थिर uint32_t अंक_elems) {
    hvx_mul_scalar_f32(dst, src, 1.0f / val, अंक_elems);
}

अटल अंतरभूत व्योम hvx_div_scalar_f16(uint8_t * restrict dst, स्थिर uint8_t * restrict src, स्थिर _Float16 val, स्थिर uint32_t अंक_elems) {
    hvx_div_scalar_f16_aa(dst, src, val, अंक_elems);
}

प्रकार व्योम (*compute_द्विआधारी_chunked_t)(
    uint8_t * restrict dst,
    स्थिर uint8_t * restrict src0,
    स्थिर uint8_t * restrict src1,
    स्थिर uint32_t अंक_elems
);

प्रकार व्योम (*compute_द्विआधारी_scalar_chunked_f32_t)(
    uint8_t * restrict dst,
    स्थिर uint8_t * restrict src0,
    स्थिर भग्न val,
    स्थिर uint32_t अंक_elems
);

प्रकार व्योम (*compute_द्विआधारी_scalar_chunked_f16_t)(
    uint8_t * restrict dst,
    स्थिर uint8_t * restrict src0,
    स्थिर _Float16 val,
    स्थिर uint32_t अंक_elems
);

काष्ठा द्विआधारी_chunked_context {
    काष्ठा htp_ops_context * octx;
    काष्ठा htp_द्विआधारी_vtcm_layout vtcm_layout;
    uint8_t *                vtcm_base;
    uint32_t                 elem_start;
    uint32_t                 nelem;
    uint32_t                 chunk_आकार;
    uint32_t                 chunks_per_thread;
    uint32_t                 योग_chunks;
    bool                     is_scalar;
    भग्न                    scalar_f32;
    _Float16                 scalar_f16;
    compute_द्विआधारी_chunked_t compute;
    compute_द्विआधारी_scalar_chunked_f32_t compute_scalar_f32;
    compute_द्विआधारी_scalar_chunked_f16_t compute_scalar_f16;
};

अटल व्योम द्विआधारी_thread_chunked(अचिन्हित पूर्णांक nth, अचिन्हित पूर्णांक ith, व्योम * तथ्य) {
    (व्योम) nth;
    काष्ठा द्विआधारी_chunked_context * ctx = (काष्ठा द्विआधारी_chunked_context *) तथ्य;
    काष्ठा htp_ops_context * octx = ctx->octx;
    स्थिर काष्ठा htp_tensor * src0 = octx->src[0];
    स्थिर काष्ठा htp_tensor * src1 = octx->src[1];
    स्थिर काष्ठा htp_tensor * dst  = octx->dst;

    स्थिर uint32_t start_chunk = ctx->chunks_per_thread * ith;
    स्थिर uint32_t end_chunk   = MIN(start_chunk + ctx->chunks_per_thread, ctx->योग_chunks);
    अगर (start_chunk >= end_chunk) {
        वापस;
    }

    स्थिर uint32_t src0_type   = src0->type;
    स्थिर माप_प्रकार   elem_आकार   = (src0_type == HTP_TYPE_F32) ? माप(भग्न) : माप(_Float16);
    स्थिर uint32_t chunk_आकार  = ctx->chunk_आकार;
    स्थिर माप_प्रकार   chunk_bytes = ctx->vtcm_layout.src0_spad_half_आकार;

    FARF(ऊपरी_2, "binary-chunked: %d/%d (%u:%u) chunks %u elems %u",
         ith, nth, start_chunk, end_chunk, ctx->योग_chunks, ctx->nelem);

    स्थिर काष्ठा htp_द्विआधारी_vtcm_layout * layout = &ctx->vtcm_layout;
    uint8_t * src0_spad_base = VTCM_LAYOUT_PTR(uint8_t, ctx->vtcm_base, layout->off_src0) + (ith * layout->src0_bytes_per_thread);
    uint8_t * src1_spad_base = ctx->is_scalar ? शून्य : (VTCM_LAYOUT_PTR(uint8_t, ctx->vtcm_base, layout->off_src1) + (ith * layout->src1_bytes_per_thread));
    uint8_t * dst_spad_base  = VTCM_LAYOUT_PTR(uint8_t, ctx->vtcm_base, layout->off_dst)  + (ith * layout->dst_bytes_per_thread);

    dma_queue * dma_q = octx->ctx->dma[ith];
    uint32_t prefetch_chunk = start_chunk;
    पूर्णांक spad_idx = 0;

    क्रम (पूर्णांक k = 0; k < 2 && prefetch_chunk < end_chunk; k++) {
        स्थिर uint32_t c_start   = ctx->elem_start + prefetch_chunk * chunk_आकार;
        स्थिर uint32_t c_end     = MIN(c_start + chunk_आकार, ctx->elem_start + ctx->nelem);
        स्थिर uint32_t cur_elems = c_end - c_start;
        स्थिर uint32_t cur_bytes = cur_elems * elem_आकार;

        dma_addr_t s0_curr = src0->तथ्य + (माप_प्रकार) c_start * elem_आकार;
        dma_addr_t d_curr  = dst->तथ्य  + (माप_प्रकार) c_start * elem_आकार;

        uint8_t * s0_spad = src0_spad_base + spad_idx * chunk_bytes;
        uint8_t * d_spad  = dst_spad_base  + spad_idx * chunk_bytes;

        dma_queue_धकेलो(dma_q, dma_make_तथ्य(d_curr, d_spad), chunk_bytes, chunk_bytes, cur_bytes, 0);
        dma_queue_धकेलो(dma_q, dma_make_तथ्य(s0_spad, s0_curr), chunk_bytes, chunk_bytes, cur_bytes, 1);
        अगर (!ctx->is_scalar) {
            dma_addr_t s1_curr = src1->तथ्य + (माप_प्रकार) c_start * elem_आकार;
            uint8_t * s1_spad = src1_spad_base + spad_idx * chunk_bytes;
            dma_queue_धकेलो(dma_q, dma_make_तथ्य(s1_spad, s1_curr), chunk_bytes, chunk_bytes, cur_bytes, 1);
        }

        prefetch_chunk++;
        spad_idx ^= 1;
    }

    काष्ठा htp_thread_trace * tr = &octx->ctx->trace[ith];

    क्रम (uint32_t c = start_chunk; c < end_chunk; c++) {
        स्थिर uint32_t c_start   = ctx->elem_start + c * chunk_आकार;
        स्थिर uint32_t c_end     = MIN(c_start + chunk_आकार, ctx->elem_start + ctx->nelem);
        स्थिर uint32_t cur_elems = c_end - c_start;
        स्थिर uint32_t cur_bytes = cur_elems * elem_आकार;

        uint8_t * d_spad  = (uint8_t *) dma_queue_निकालो(dma_q).src;
        uint8_t * s0_spad = (uint8_t *) dma_queue_निकालो(dma_q).dst;
        uint8_t * s1_spad = ctx->is_scalar ? शून्य : (uint8_t *) dma_queue_निकालो(dma_q).dst;

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) c);
        अगर (ctx->is_scalar) {
            अगर (src0_type == HTP_TYPE_F32) {
                ctx->compute_scalar_f32(d_spad, s0_spad, ctx->scalar_f32, cur_elems);
            } अन्यथा {
                ctx->compute_scalar_f16(d_spad, s0_spad, ctx->scalar_f16, cur_elems);
            }
        } अन्यथा {
            ctx->compute(d_spad, s0_spad, s1_spad, cur_elems);
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) c);

        dma_addr_t dst_curr = dst->तथ्य + (माप_प्रकार) c_start * elem_आकार;
        dma_queue_धकेलो(dma_q, dma_make_तथ्य(dst_curr, d_spad), chunk_bytes, chunk_bytes, cur_bytes, 1);

        अगर (prefetch_chunk < end_chunk) {
            स्थिर uint32_t pc_start  = ctx->elem_start + prefetch_chunk * chunk_आकार;
            स्थिर uint32_t pc_end    = MIN(pc_start + chunk_आकार, ctx->elem_start + ctx->nelem);
            स्थिर uint32_t p_elems   = pc_end - pc_start;
            स्थिर uint32_t p_bytes   = p_elems * elem_आकार;

            dma_addr_t s0_अग्र = src0->तथ्य + (माप_प्रकार) pc_start * elem_आकार;
            dma_queue_धकेलो(dma_q, dma_make_तथ्य(s0_spad, s0_अग्र), chunk_bytes, chunk_bytes, p_bytes, 1);
            अगर (!ctx->is_scalar) {
                dma_addr_t s1_अग्र = src1->तथ्य + (माप_प्रकार) pc_start * elem_आकार;
                dma_queue_धकेलो(dma_q, dma_make_तथ्य(s1_spad, s1_अग्र), chunk_bytes, chunk_bytes, p_bytes, 1);
            }

            prefetch_chunk++;
        }
    }

    dma_queue_flush(dma_q);
}

अटल पूर्णांक execute_op_द्विआधारी(काष्ठा htp_ops_context * octx) {
    स्थिर काष्ठा htp_tensor * src0 = octx->src[0];
    स्थिर काष्ठा htp_tensor * src1 = octx->src[1];
    स्थिर काष्ठा htp_tensor * dst  = octx->dst;
    स्थिर काष्ठा htp_द्विआधारी_kernel_params * kparams = (स्थिर काष्ठा htp_द्विआधारी_kernel_params *) octx->kernel_params;

    स्थिर uint32_t src0_nrows = src0->ne[1] * src0->ne[2] * src0->ne[3];

    // Use packed row sizes for VTCM allocation and alignment
    स्थिर uint32_t src0_type = octx->src[0]->type;
    स्थिर माप_प्रकार elem_आकार = (src0_type == HTP_TYPE_F32) ? माप(भग्न) : माप(_Float16);
    स्थिर माप_प्रकार src0_row_आकार = src0->ne[0] * elem_आकार;
    स्थिर माप_प्रकार src1_row_आकार = src1->ne[0] * elem_आकार;
    स्थिर माप_प्रकार dst_row_आकार  = dst->ne[0]  * elem_आकार;

    अगर (kparams->kernel_type == HTP_द्विआधारी_KERNEL_CHUNKED) {
        स्थिर bool is_scalar = kparams->is_scalar || (src1->ne[0] == 1 && src1->ne[1] == 1 && src1->ne[2] == 1 && src1->ne[3] == 1);

        compute_द्विआधारी_chunked_t compute = शून्य;
        compute_द्विआधारी_scalar_chunked_f32_t compute_scalar_f32 = शून्य;
        compute_द्विआधारी_scalar_chunked_f16_t compute_scalar_f16 = शून्य;

        अगर (is_scalar) {
            अगर (src0_type == HTP_TYPE_F32) {
                चयन (octx->op) {
                    हाल HTP_OP_ADD: compute_scalar_f32 = hvx_add_scalar_f32; अवरोध;
                    हाल HTP_OP_SUB: compute_scalar_f32 = hvx_sub_scalar_f32; अवरोध;
                    हाल HTP_OP_MUL: compute_scalar_f32 = hvx_mul_scalar_f32; अवरोध;
                    हाल HTP_OP_DIV: compute_scalar_f32 = hvx_div_scalar_f32; अवरोध;
                    शेष: अवरोध;
                }
            } अन्यथा अगर (src0_type == HTP_TYPE_F16) {
                चयन (octx->op) {
                    हाल HTP_OP_ADD: compute_scalar_f16 = hvx_add_scalar_f16; अवरोध;
                    हाल HTP_OP_SUB: compute_scalar_f16 = hvx_sub_scalar_f16; अवरोध;
                    हाल HTP_OP_MUL: compute_scalar_f16 = hvx_mul_scalar_f16; अवरोध;
                    हाल HTP_OP_DIV: compute_scalar_f16 = hvx_div_scalar_f16; अवरोध;
                    शेष: अवरोध;
                }
            }
            अगर (!compute_scalar_f32 && !compute_scalar_f16) {
                वापस HTP_STATUS_NO_SUPPORT;
            }
        } अन्यथा {
            अगर (src0_type == HTP_TYPE_F32) {
                चयन (octx->op) {
                    हाल HTP_OP_ADD: compute = hvx_add_f32; अवरोध;
                    हाल HTP_OP_SUB: compute = hvx_sub_f32; अवरोध;
                    हाल HTP_OP_MUL: compute = hvx_mul_f32; अवरोध;
                    हाल HTP_OP_DIV: compute = hvx_div_f32; अवरोध;
                    शेष: अवरोध;
                }
            } अन्यथा अगर (src0_type == HTP_TYPE_F16) {
                चयन (octx->op) {
                    हाल HTP_OP_ADD: compute = hvx_add_f16; अवरोध;
                    हाल HTP_OP_SUB: compute = hvx_sub_f16; अवरोध;
                    हाल HTP_OP_MUL: compute = hvx_mul_f16; अवरोध;
                    हाल HTP_OP_DIV: compute = hvx_div_f16; अवरोध;
                    शेष: अवरोध;
                }
            }
            अगर (!compute) {
                वापस HTP_STATUS_NO_SUPPORT;
            }
        }

        स्थिर uint32_t योग_elems = (uint32_t) (src0->ne[0] * src0->ne[1] * src0->ne[2] * src0->ne[3]);
        अगर (योग_elems == 0) {
            वापस HTP_STATUS_OK;
        }

        uint32_t elem_start = 0;
        uint32_t nelem      = योग_elems;
        स्थिर uint32_t elems_per_पंक्ति = (src0_type == HTP_TYPE_F32) ? 32 : 64;

        अगर (octx->ctx->mdev.गिनती > 1) {
            स्थिर bool can_split = htp_tensor_mdev_तथ्य_aligned(dst) &&
                                   htp_tensor_mdev_तथ्य_aligned(src0) &&
                                   (is_scalar || htp_tensor_mdev_तथ्य_aligned(src1)) &&
                                   htp_tensor_is_contiguous(dst, elem_आकार) &&
                                   htp_tensor_is_contiguous(src0, elem_आकार) &&
                                   (is_scalar || htp_tensor_is_contiguous(src1, elem_आकार));
            स्थिर काष्ठा htp_tensor_mdev_range range = htp_tensor_mdev_partition(
                योग_elems, can_split ? elems_per_पंक्ति : 0,
                octx->ctx->mdev.idx, octx->ctx->mdev.गिनती, &octx->ctx->mdev.गिनती_div);
            elem_start = range.start;
            nelem      = range.गिनती;
        }

        अगर (nelem == 0) {
            वापस HTP_STATUS_OK;
        }

        अगर (!htp_ops_context_set_n_threads(octx, kparams->n_threads)) {
            वापस HTP_STATUS_INVAL_PARAMS;
        }

        काष्ठा htp_द्विआधारी_vtcm_layout vtcm_layout;
        htp_द्विआधारी_vtcm_layout_build(&vtcm_layout, kparams, octx->ctx->vtcm_आकार);
        अगर (vtcm_layout.योग_bytes == 0 || vtcm_layout.योग_bytes > octx->ctx->vtcm_आकार) {
            वापस HTP_STATUS_VTCM_TOO_SMALL;
        }

        स्थिर uint32_t chunk_आकार = kparams->chunk_आकार > 0 ? kparams->chunk_आकार : (32768 / elem_आकार);
        स्थिर uint32_t योग_chunks = (nelem + chunk_आकार - 1) / chunk_आकार;
        स्थिर uint32_t n_threads = (योग_chunks >= 2) ? MIN(octx->n_threads, योग_chunks) : 1;
        स्थिर uint32_t chunks_per_thread = (योग_chunks + n_threads - 1) / n_threads;

        भग्न scalar_f32 = 0.0f;
        _Float16 scalar_f16 = 0;
        अगर (is_scalar) {
            uint8_t * vtcm_src1 = VTCM_LAYOUT_PTR(uint8_t, octx->ctx->vtcm_base, vtcm_layout.off_src1);
            dma_queue * dma_q = octx->ctx->dma[0];
            dma_queue_धकेलो(dma_q, dma_make_तथ्य(vtcm_src1, src1->तथ्य), 128, 0, elem_आकार, 1);
            dma_queue_निकालो(dma_q);

            अगर (src0_type == HTP_TYPE_F32) {
                scalar_f32 = ((स्थिर भग्न *) vtcm_src1)[0];
            } अन्यथा {
                scalar_f16 = ((स्थिर _Float16 *) vtcm_src1)[0];
            }
        }

        काष्ठा द्विआधारी_chunked_context cctx = {
            .octx                = octx,
            .vtcm_layout         = vtcm_layout,
            .vtcm_base           = (uint8_t *) octx->ctx->vtcm_base,
            .elem_start          = elem_start,
            .nelem               = nelem,
            .chunk_आकार          = chunk_आकार,
            .chunks_per_thread   = chunks_per_thread,
            .योग_chunks        = योग_chunks,
            .is_scalar           = is_scalar,
            .scalar_f32          = scalar_f32,
            .scalar_f16          = scalar_f16,
            .compute             = compute,
            .compute_scalar_f32  = compute_scalar_f32,
            .compute_scalar_f16  = compute_scalar_f16,
        };

        work_queue_run(octx->ctx->work_queue, द्विआधारी_thread_chunked, &cctx, n_threads);
        वापस HTP_STATUS_OK;
    }

    uint32_t row_start = 0;
    uint32_t nrows     = src0_nrows;

    अगर (octx->ctx->mdev.गिनती > 1) {
        uint32_t rows_per_chunk = 0;
        htp_tensor_mdev_rows_per_chunk(dst, (uint32_t) elem_आकार, (uint32_t) dst_row_आकार, &rows_per_chunk);
        स्थिर काष्ठा htp_tensor_mdev_range range = htp_tensor_mdev_partition(src0_nrows, rows_per_chunk,
                                                       octx->ctx->mdev.idx, octx->ctx->mdev.गिनती, &octx->ctx->mdev.गिनती_div);
        row_start = range.start;
        nrows     = range.गिनती;
    }

    अगर (nrows == 0) {
        वापस HTP_STATUS_OK;
    }

    अगर (!htp_ops_context_set_n_threads(octx, kparams->n_threads)) {
        वापस HTP_STATUS_INVAL_PARAMS;
    }

    स्थिर uint32_t n_threads = octx->n_threads;
    स्थिर माप_प्रकार src0_row_आकार_aligned = kparams->src0_row_आकार_aligned;
    स्थिर माप_प्रकार src1_row_आकार_aligned = kparams->src1_row_आकार_aligned;
    स्थिर माप_प्रकार dst_row_आकार_aligned  = kparams->dst_row_आकार_aligned;

    काष्ठा htp_द्विआधारी_context bctx;
    bctx.vtcm_base = (uint8_t *) octx->ctx->vtcm_base;
    htp_द्विआधारी_vtcm_layout_build(&bctx.vtcm_layout, kparams, octx->ctx->vtcm_आकार);

    अगर (bctx.vtcm_layout.rows_per_बफर == 0 || bctx.vtcm_layout.योग_bytes > octx->ctx->vtcm_आकार) {
        वापस HTP_STATUS_VTCM_TOO_SMALL;
    }

    dma_queue * dma_q = octx->ctx->dma[0];
    uint8_t * vtcm_src1 = VTCM_LAYOUT_PTR(uint8_t, bctx.vtcm_base, bctx.vtcm_layout.off_src1);
    अगर (kparams->kernel_type == HTP_द्विआधारी_KERNEL_ROW_BCAST) {
        dma_queue_धकेलो(dma_q, dma_make_तथ्य(vtcm_src1, src1->तथ्य), bctx.vtcm_layout.src1_आकार, 0, src1->ne[0] * elem_आकार, 1);
    } अन्यथा अगर (kparams->kernel_type == HTP_द्विआधारी_KERNEL_SCALAR ||
               kparams->kernel_type == HTP_द्विआधारी_KERNEL_REPEAT ||
               kparams->kernel_type == HTP_द्विआधारी_KERNEL_COMPLEX) {
        अगर (!kparams->src1_strided) {
            स्थिर माप_प्रकार src1_bytes = (माप_प्रकार) src1->ne[0] * src1->ne[1] * src1->ne[2] * src1->ne[3] * elem_आकार;
            dma_queue_धकेलो(dma_q, dma_make_तथ्य(vtcm_src1, src1->तथ्य), bctx.vtcm_layout.src1_आकार, 0, src1_bytes, 1);
        } अन्यथा {
            स्थिर uint32_t ne11 = src1->ne[1];
            स्थिर uint32_t ne12 = src1->ne[2];
            स्थिर uint32_t ne13 = src1->ne[3];
            स्थिर माप_प्रकार row_bytes = src1->ne[0] * elem_आकार;
            क्रम (uint32_t i13 = 0; i13 < ne13; i13++) {
                क्रम (uint32_t i12 = 0; i12 < ne12; i12++) {
                    dma_addr_t src_addr = src1->तथ्य + i13 * src1->nb[3] + i12 * src1->nb[2];
                    uint8_t * dst_ptr = vtcm_src1 + (i13 * ne12 * ne11 + i12 * ne11) * row_bytes;
                    जबतक (!dma_queue_धकेलो(dma_q, dma_make_तथ्य(dst_ptr, src_addr),
                                           row_bytes, src1->nb[1], row_bytes, ne11)) {
                        dma_queue_निकालो(dma_q);
                    }
                }
            }
        }
    } अन्यथा अगर (kparams->kernel_type == HTP_द्विआधारी_KERNEL_ADD_ID) {
        स्थिर काष्ठा htp_tensor * src2 = octx->src[2];
        uint8_t * vtcm_src2 = VTCM_LAYOUT_PTR(uint8_t, bctx.vtcm_base, bctx.vtcm_layout.off_src2);
        dma_queue_धकेलो(dma_q, dma_make_तथ्य(vtcm_src1, src1->तथ्य),
                       kparams->src1_row_आकार_aligned, src1->nb[1],
                       src1->ne[0] * elem_आकार, src1->ne[1]);
        स्थिर माप_प्रकार src2_row_bytes = src2->ne[0] * माप(int32_t);
        dma_queue_धकेलो(dma_q, dma_make_तथ्य(vtcm_src2, src2->तथ्य),
                       src2_row_bytes, src2->nb[1],
                       src2_row_bytes, src2->ne[1]);
    }

    bctx.octx                  = octx;
    bctx.nrows_per_thread      = fastdiv(nrows + n_threads - 1, &octx->n_threads_div);
    bctx.योग_rows            = nrows;
    bctx.row_start             = row_start;
    bctx.block_max             = bctx.vtcm_layout.rows_per_बफर;
    bctx.src0_row_आकार_aligned = src0_row_आकार_aligned;
    bctx.src1_row_आकार_aligned = src1_row_आकार_aligned;
    bctx.dst_row_आकार_aligned  = dst_row_आकार_aligned;
    bctx.row_आकार_bytes        = src0_row_आकार;

    bctx.src0_dim1_div  = init_fastdiv_मूल्यावली(src0->ne[1]);
    bctx.src0_dim2_div  = init_fastdiv_मूल्यावली(src0->ne[2]);
    bctx.src0_dim12_div = init_fastdiv_मूल्यावली(src0->ne[1] * src0->ne[2]);

    bctx.src1_dim1_div  = init_fastdiv_मूल्यावली(src1->ne[1]);
    bctx.src1_dim2_div  = init_fastdiv_मूल्यावली(src1->ne[2]);
    bctx.src1_dim3_div  = init_fastdiv_मूल्यावली(src1->ne[3]);

    bctx.split_at_ne01  = kparams->split_at_ne01;
    bctx.split_at_ne02  = kparams->split_at_ne02;

    worker_callback_t worker_func = शून्य;
    व्योम * compute_func = शून्य;

    चयन (kparams->kernel_type) {
        हाल HTP_द्विआधारी_KERNEL_SAME_SHAPE:
            worker_func = द्विआधारी_thread_vector_same_shape;
            अगर (src0_type == HTP_TYPE_F32) {
                चयन (octx->op) {
                    हाल HTP_OP_ADD: compute_func = compute_same_shape_add_f32; अवरोध;
                    हाल HTP_OP_SUB: compute_func = compute_same_shape_sub_f32; अवरोध;
                    हाल HTP_OP_MUL: compute_func = compute_same_shape_mul_f32; अवरोध;
                    हाल HTP_OP_DIV: compute_func = compute_same_shape_div_f32; अवरोध;
                    शेष: अवरोध;
                }
            } अन्यथा अगर (src0_type == HTP_TYPE_F16) {
                चयन (octx->op) {
                    हाल HTP_OP_ADD: compute_func = compute_same_shape_add_f16; अवरोध;
                    हाल HTP_OP_SUB: compute_func = compute_same_shape_sub_f16; अवरोध;
                    हाल HTP_OP_MUL: compute_func = compute_same_shape_mul_f16; अवरोध;
                    हाल HTP_OP_DIV: compute_func = compute_same_shape_div_f16; अवरोध;
                    शेष: अवरोध;
                }
            }
            अवरोध;
        हाल HTP_द्विआधारी_KERNEL_ROW_BCAST:
            worker_func = द्विआधारी_thread_vector_row_broadcast;
            अगर (src0_type == HTP_TYPE_F32) {
                चयन (octx->op) {
                    हाल HTP_OP_ADD: compute_func = compute_row_bcast_add_f32; अवरोध;
                    हाल HTP_OP_SUB: compute_func = compute_row_bcast_sub_f32; अवरोध;
                    हाल HTP_OP_MUL: compute_func = compute_row_bcast_mul_f32; अवरोध;
                    हाल HTP_OP_DIV: compute_func = compute_row_bcast_div_f32; अवरोध;
                    शेष: अवरोध;
                }
            } अन्यथा अगर (src0_type == HTP_TYPE_F16) {
                चयन (octx->op) {
                    हाल HTP_OP_ADD: compute_func = compute_row_bcast_add_f16; अवरोध;
                    हाल HTP_OP_SUB: compute_func = compute_row_bcast_sub_f16; अवरोध;
                    हाल HTP_OP_MUL: compute_func = compute_row_bcast_mul_f16; अवरोध;
                    हाल HTP_OP_DIV: compute_func = compute_row_bcast_div_f16; अवरोध;
                    शेष: अवरोध;
                }
            }
            अवरोध;
        हाल HTP_द्विआधारी_KERNEL_SCALAR:
            worker_func = द्विआधारी_thread_scalar;
            अगर (src0_type == HTP_TYPE_F32) {
                चयन (octx->op) {
                    हाल HTP_OP_ADD: compute_func = compute_scalar_add_f32; अवरोध;
                    हाल HTP_OP_SUB: compute_func = compute_scalar_sub_f32; अवरोध;
                    हाल HTP_OP_MUL: compute_func = compute_scalar_mul_f32; अवरोध;
                    हाल HTP_OP_DIV: compute_func = compute_scalar_div_f32; अवरोध;
                    शेष: अवरोध;
                }
            } अन्यथा अगर (src0_type == HTP_TYPE_F16) {
                चयन (octx->op) {
                    हाल HTP_OP_ADD: compute_func = compute_scalar_add_f16; अवरोध;
                    हाल HTP_OP_SUB: compute_func = compute_scalar_sub_f16; अवरोध;
                    हाल HTP_OP_MUL: compute_func = compute_scalar_mul_f16; अवरोध;
                    हाल HTP_OP_DIV: compute_func = compute_scalar_div_f16; अवरोध;
                    शेष: अवरोध;
                }
            }
            अवरोध;
        हाल HTP_द्विआधारी_KERNEL_COMPLEX:
            worker_func = द्विआधारी_thread_vector_complex;
            अगर (src0_type == HTP_TYPE_F32) {
                चयन (octx->op) {
                    हाल HTP_OP_ADD: compute_func = compute_complex_add_f32; अवरोध;
                    हाल HTP_OP_SUB: compute_func = compute_complex_sub_f32; अवरोध;
                    हाल HTP_OP_MUL: compute_func = compute_complex_mul_f32; अवरोध;
                    हाल HTP_OP_DIV: compute_func = compute_complex_div_f32; अवरोध;
                    शेष: अवरोध;
                }
            } अन्यथा अगर (src0_type == HTP_TYPE_F16) {
                चयन (octx->op) {
                    हाल HTP_OP_ADD: compute_func = compute_complex_add_f16; अवरोध;
                    हाल HTP_OP_SUB: compute_func = compute_complex_sub_f16; अवरोध;
                    हाल HTP_OP_MUL: compute_func = compute_complex_mul_f16; अवरोध;
                    हाल HTP_OP_DIV: compute_func = compute_complex_div_f16; अवरोध;
                    शेष: अवरोध;
                }
            }
            अवरोध;
        हाल HTP_द्विआधारी_KERNEL_REPEAT:
            worker_func = द्विआधारी_thread_element_repeat;
            अगर (src0_type == HTP_TYPE_F32) {
                चयन (octx->op) {
                    हाल HTP_OP_ADD: compute_func = compute_repeat_add_f32; अवरोध;
                    हाल HTP_OP_SUB: compute_func = compute_repeat_sub_f32; अवरोध;
                    हाल HTP_OP_MUL: compute_func = compute_repeat_mul_f32; अवरोध;
                    हाल HTP_OP_DIV: compute_func = compute_repeat_div_f32; अवरोध;
                    शेष: अवरोध;
                }
            } अन्यथा अगर (src0_type == HTP_TYPE_F16) {
                चयन (octx->op) {
                    हाल HTP_OP_ADD: compute_func = compute_repeat_add_f16; अवरोध;
                    हाल HTP_OP_SUB: compute_func = compute_repeat_sub_f16; अवरोध;
                    हाल HTP_OP_MUL: compute_func = compute_repeat_mul_f16; अवरोध;
                    हाल HTP_OP_DIV: compute_func = compute_repeat_div_f16; अवरोध;
                    शेष: अवरोध;
                }
            }
            अवरोध;
        हाल HTP_द्विआधारी_KERNEL_ADD_ID:
            अगर (octx->op == HTP_OP_ADD_ID && src0_type == HTP_TYPE_F32) {
                worker_func = द्विआधारी_thread_add_id_f32;
                compute_func = (व्योम *) compute_add_id_f32;
            }
            अवरोध;
        शेष: अवरोध;
    }

    अगर (!worker_func || !compute_func) {
        वापस HTP_STATUS_NO_SUPPORT;
    }

    bctx.compute = compute_func;

    अगर (kparams->kernel_type == HTP_द्विआधारी_KERNEL_ROW_BCAST ||
        kparams->kernel_type == HTP_द्विआधारी_KERNEL_SCALAR ||
        kparams->kernel_type == HTP_द्विआधारी_KERNEL_REPEAT ||
        kparams->kernel_type == HTP_द्विआधारी_KERNEL_COMPLEX ||
        kparams->kernel_type == HTP_द्विआधारी_KERNEL_ADD_ID) {
        dma_queue_flush(dma_q);
    }

    work_queue_run(octx->ctx->work_queue, worker_func, &bctx, n_threads);

    वापस HTP_STATUS_OK;
}

पूर्णांक op_द्विआधारी(काष्ठा htp_ops_context * octx) {
    स्थिर uint32_t src0_type = octx->src[0]->type;
    अगर ((src0_type == HTP_TYPE_F32) || (src0_type == HTP_TYPE_F16)) {
        वापस execute_op_द्विआधारी(octx);
    }

    वापस HTP_STATUS_NO_SUPPORT;
}
