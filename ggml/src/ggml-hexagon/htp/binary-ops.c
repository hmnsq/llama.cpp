#pragma clang diagnostic ignored "-Wunused-variable"
#pragma clang diagnostic ignored "-Wunused-function"
#pragma clang diagnostic ignored "-Wunused-but-set-variable"

#অন্তর্ভাব <HAP_farf.h>
#অন্তর্ভাব <HAP_perf.h>

#অন্তর্ভাব <math.h>
#অন্তর্ভাব <string.h>

#অন্তর্ভাব "dma-queue.h"
#অন্তর্ভাব "hvx-utils.h"

#define GGML_COMMON_DECL_C
#অন্তর্ভাব "ggml-common.h"
#অন্তর্ভাব "hex-common.h"
#অন্তর্ভাব "hex-profile.h"
#অন্তর্ভাব "binary-ops.h"
#অন্তর্ভাব "htp-ctx.h"
#অন্তর্ভাব "htp-ops.h"
#অন্তর্ভাব "htp-tensor.h"

#ifndef MIN
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#endif

// Context for binary operations
কাঠামো htp_দ্বিমিক_context {
    কাঠামো htp_ops_context * octx;
    কাঠামো htp_দ্বিমিক_vtcm_layout vtcm_layout;
    uint8_t * vtcm_base;

    কাঠামো fastdiv_মানসমূহ src0_dim1_div; // ne01
    কাঠামো fastdiv_মানসমূহ src0_dim2_div; // ne02
    কাঠামো fastdiv_মানসমূহ src0_dim12_div;// ne03

    কাঠামো fastdiv_মানসমূহ src1_dim1_div; // ne11
    কাঠামো fastdiv_মানসমূহ src1_dim2_div; // ne12
    কাঠামো fastdiv_মানসমূহ src1_dim3_div; // ne13

    uint32_t block_max;
    uint32_t nrows_per_thread;
    uint32_t যোগ_rows;
    uint32_t row_start;
    size_t   src0_row_আকার_aligned;
    size_t   src1_row_আকার_aligned;
    size_t   dst_row_আকার_aligned;
    size_t   row_আকার_bytes;

    bool split_at_ne01;
    bool split_at_ne02;

    শূন্য * compute;
};

#define htp_দ্বিমিক_preamble                        \
    ধ্রুবক কাঠামো htp_tensor * src0 = octx->src[0]; \
    ধ্রুবক কাঠামো htp_tensor * src1 = octx->src[1]; \
    ধ্রুবক কাঠামো htp_tensor * dst  = octx->dst;    \
                                                   \
    ধ্রুবক uint32_t ne00 = src0->ne[0];             \
    ধ্রুবক uint32_t ne01 = src0->ne[1];             \
    ধ্রুবক uint32_t ne02 = src0->ne[2];             \
    ধ্রুবক uint32_t ne03 = src0->ne[3];             \
                                                   \
    ধ্রুবক uint32_t ne10 = src1->ne[0];             \
    ধ্রুবক uint32_t ne11 = src1->ne[1];             \
    ধ্রুবক uint32_t ne12 = src1->ne[2];             \
    ধ্রুবক uint32_t ne13 = src1->ne[3];             \
                                                   \
    ধ্রুবক uint32_t nb01 = src0->nb[1];             \
    ধ্রুবক uint32_t nb02 = src0->nb[2];             \
    ধ্রুবক uint32_t nb03 = src0->nb[3];             \
                                                   \
    ধ্রুবক uint32_t nb11 = src1->nb[1];             \
    ধ্রুবক uint32_t nb12 = src1->nb[2];             \
    ধ্রুবক uint32_t nb13 = src1->nb[3];             \
                                                   \
    ধ্রুবক uint32_t nb1 = dst->nb[1];               \
    ধ্রুবক uint32_t nb2 = dst->nb[2];               \
    ধ্রুবক uint32_t nb3 = dst->nb[3];

স্থির অন্তঃরেখ uint32_t calc_block_আকার(কাঠামো htp_দ্বিমিক_context * bctx, uint32_t ir, uint32_t end_row, uint32_t ne01, uint32_t ne02) {
    uint32_t i03, i02, i01, rem;
    i03 = fastdiv(ir, &bctx->src0_dim12_div);
    rem = ir - i03 * (ne02 * ne01);
    i02 = fastdiv(rem, &bctx->src0_dim1_div);
    i01 = rem - i02 * ne01;

    uint32_t rows_বাম = end_row - ir;
    uint32_t block_limit = rows_বাম;

    যদি (bctx->split_at_ne01) {
        block_limit = MIN(block_limit, ne01 - i01);
    }
    যদি (bctx->split_at_ne02) {
         uint32_t rows_in_plane = (ne02 * ne01) - rem;
         block_limit = MIN(block_limit, rows_in_plane);
    }

    ফেরত MIN(bctx->block_max, block_limit);
}

// Out-of-line compute micro-kernels

প্রকারলেখ শূন্য (*compute_scalar_t)(
    uint8_t * dst, ধ্রুবক uint8_t * src0, ধ্রুবক শূন্য * s1_table,
    uint32_t cur_i11, uint32_t ne11, uint32_t n_rows,
    size_t dst_stride, size_t src0_stride, uint32_t ne00);

#define DEFINE_COMPUTE_SCALAR(NAME, TYPE, HVX_STMT)             \
স্থির শূন্য compute_scalar_##NAME(                              \
    uint8_t * dst, ধ্রুবক uint8_t * src0, ধ্রুবক শূন্য * s1_table, \
    uint32_t cur_i11, uint32_t ne11, uint32_t n_rows,           \
    size_t dst_stride, size_t src0_stride, uint32_t ne00) {     \
    ধ্রুবক TYPE * table = (ধ্রুবক TYPE *) s1_table;               \
    জন্য (uint32_t r = 0; r < n_rows; r++) {                     \
        uint8_t * r_dst = dst + r * dst_stride;                 \
        ধ্রুবক uint8_t * r_src0 = src0 + r * src0_stride;        \
        TYPE val = table[cur_i11];                              \
        HVX_STMT;                                               \
        যদি (ne11 > 1 && ++cur_i11 == ne11) {                    \
            cur_i11 = 0;                                        \
        }                                                       \
    }                                                           \
}

DEFINE_COMPUTE_SCALAR(add_f32, দশমিক,    hvx_add_scalar_f32_aa(r_dst, r_src0, val, ne00))
DEFINE_COMPUTE_SCALAR(add_f16, _Float16, hvx_add_scalar_f16_aa(r_dst, r_src0, val, ne00))
DEFINE_COMPUTE_SCALAR(sub_f32, দশমিক,    hvx_sub_scalar_f32_aa(r_dst, r_src0, val, ne00))
DEFINE_COMPUTE_SCALAR(sub_f16, _Float16, hvx_sub_scalar_f16_aa(r_dst, r_src0, val, ne00))
DEFINE_COMPUTE_SCALAR(mul_f32, দশমিক,    hvx_mul_scalar_f32_aa(r_dst, r_src0, val, ne00))
DEFINE_COMPUTE_SCALAR(mul_f16, _Float16, hvx_mul_scalar_f16_aa(r_dst, r_src0, val, ne00))
DEFINE_COMPUTE_SCALAR(div_f32, দশমিক,    hvx_mul_scalar_f32_aa(r_dst, r_src0, 1.0f / (val), ne00))
DEFINE_COMPUTE_SCALAR(div_f16, _Float16, hvx_div_scalar_f16_aa(r_dst, r_src0, val, ne00))

প্রকারলেখ শূন্য (*compute_same_shape_t)(
    uint8_t * dst, ধ্রুবক uint8_t * src0, ধ্রুবক uint8_t * src1,
    uint32_t n_rows, size_t dst_stride, size_t src0_stride, size_t src1_stride, uint32_t ne00);

#define DEFINE_COMPUTE_SAME_SHAPE(NAME, HVX_FN)                                                  \
স্থির শূন্য compute_same_shape_##NAME(                                                           \
    uint8_t * dst, ধ্রুবক uint8_t * src0, ধ্রুবক uint8_t * src1,                                   \
    uint32_t n_rows, size_t dst_stride, size_t src0_stride, size_t src1_stride, uint32_t ne00) { \
    জন্য (uint32_t r = 0; r < n_rows; r++) {                                                      \
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

প্রকারলেখ শূন্য (*compute_row_bcast_t)(
    uint8_t * dst, ধ্রুবক uint8_t * src0, ধ্রুবক uint8_t * src1,
    uint32_t n_rows, size_t dst_stride, size_t src0_stride, uint32_t ne00);

#define DEFINE_COMPUTE_ROW_BCAST(NAME, HVX_FN)                                               \
স্থির শূন্য compute_row_bcast_##NAME(                                                        \
    uint8_t * dst, ধ্রুবক uint8_t * src0, ধ্রুবক uint8_t * src1,                               \
    uint32_t n_rows, size_t dst_stride, size_t src0_stride, uint32_t ne00) {                 \
    জন্য (uint32_t r = 0; r < n_rows; r++) {                                                  \
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

প্রকারলেখ শূন্য (*compute_complex_t)(
    uint8_t * dst, ধ্রুবক uint8_t * src0, ধ্রুবক uint8_t * src1_plane,
    uint32_t i01, uint32_t ne11, ধ্রুবক কাঠামো fastdiv_মানসমূহ * div11, uint32_t src1_stride,
    uint32_t n_rows, size_t dst_stride, size_t src0_stride, uint32_t ne00);

#define DEFINE_COMPUTE_COMPLEX(NAME, HVX_FN)                                                 \
স্থির শূন্য compute_complex_##NAME(                                                          \
    uint8_t * dst, ধ্রুবক uint8_t * src0, ধ্রুবক uint8_t * src1_plane,                         \
    uint32_t i01, uint32_t ne11, ধ্রুবক কাঠামো fastdiv_মানসমূহ * div11, uint32_t src1_stride,  \
    uint32_t n_rows, size_t dst_stride, size_t src0_stride, uint32_t ne00) {                 \
    জন্য (uint32_t r = 0; r < n_rows; r++) {                                                  \
        uint32_t i11 = fastmodulo(i01 + r, ne11, div11);                                     \
        ধ্রুবক uint8_t * r_src1 = src1_plane + i11 * src1_stride;                             \
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

প্রকারলেখ শূন্য (*compute_repeat_t)(
    uint8_t * dst, ধ্রুবক uint8_t * src0, ধ্রুবক uint8_t * src1_plane,
    uint32_t i01, uint32_t ne11, ধ্রুবক কাঠামো fastdiv_মানসমূহ * div11, uint32_t src1_stride,
    uint32_t n_rows, size_t dst_stride, size_t src0_stride, uint32_t ne00, uint32_t ne10);

#define DEFINE_COMPUTE_REPEAT(NAME, TYPE, HVX_FN)                                            \
স্থির শূন্য compute_repeat_##NAME(                                                           \
    uint8_t * dst, ধ্রুবক uint8_t * src0, ধ্রুবক uint8_t * src1_plane,                         \
    uint32_t i01, uint32_t ne11, ধ্রুবক কাঠামো fastdiv_মানসমূহ * div11, uint32_t src1_stride,  \
    uint32_t n_rows, size_t dst_stride, size_t src0_stride, uint32_t ne00, uint32_t ne10) {  \
    জন্য (uint32_t r = 0; r < n_rows; r++) {                                                  \
        uint32_t i11 = fastmodulo(i01 + r, ne11, div11);                                     \
        ধ্রুবক uint8_t * r_src1_row = src1_plane + i11 * src1_stride;                         \
        uint8_t * r_dst = dst + r * dst_stride;                                              \
        ধ্রুবক uint8_t * r_src0 = src0 + r * src0_stride;                                     \
        জন্য (uint32_t c = 0; c < ne00; c += ne10) {                                          \
            uint32_t len = MIN(ne10, ne00 - c);                                              \
            HVX_FN(r_dst + c * আকার(TYPE), r_src0 + c * আকার(TYPE), r_src1_row, len);    \
        }                                                                                    \
    }                                                                                        \
}

DEFINE_COMPUTE_REPEAT(add_f32, দশমিক,    hvx_add_f32_uuu)
DEFINE_COMPUTE_REPEAT(add_f16, _Float16, hvx_add_f16_uuu)
DEFINE_COMPUTE_REPEAT(sub_f32, দশমিক,    hvx_sub_f32_uuu)
DEFINE_COMPUTE_REPEAT(sub_f16, _Float16, hvx_sub_f16_uuu)
DEFINE_COMPUTE_REPEAT(mul_f32, দশমিক,    hvx_mul_f32_uuu)
DEFINE_COMPUTE_REPEAT(mul_f16, _Float16, hvx_mul_f16_uuu)
DEFINE_COMPUTE_REPEAT(div_f32, দশমিক,    hvx_div_f32_uuu)
DEFINE_COMPUTE_REPEAT(div_f16, _Float16, hvx_div_f16_uuu)

প্রকারলেখ শূন্য (*compute_add_id_t)(
    uint8_t * dst, ধ্রুবক uint8_t * src0, ধ্রুবক uint8_t * src1_তথ্য, ধ্রুবক int32_t * src2_ids,
    uint32_t i01, uint32_t i02, uint32_t ne01, uint32_t src1_stride,
    uint32_t n_rows, size_t dst_stride, size_t src0_stride, uint32_t ne00);

স্থির শূন্য compute_add_id_f32(
    uint8_t * dst, ধ্রুবক uint8_t * src0, ধ্রুবক uint8_t * src1_তথ্য, ধ্রুবক int32_t * src2_ids,
    uint32_t i01, uint32_t i02, uint32_t ne01, uint32_t src1_stride,
    uint32_t n_rows, size_t dst_stride, size_t src0_stride, uint32_t ne00) {
    জন্য (uint32_t r = 0; r < n_rows; r++) {
        uint32_t r_i01 = i01 + r;
        ধ্রুবক int32_t idx = src2_ids[i02 * ne01 + r_i01];
        যদি (idx < 0) {
            memcpy(dst + r * dst_stride, src0 + r * src0_stride, ne00 * আকার(দশমিক));
            চালু;
        }
        ধ্রুবক uint8_t * r_src1 = src1_তথ্য + idx * src1_stride;
        ধ্রুবক uint8_t * r_src0 = src0 + r * src0_stride;
        uint8_t * r_dst = dst + r * dst_stride;
        hvx_add_f32_aaa(r_dst, r_src0, r_src1, ne00);
    }
}

// 1. Scalar src1 in VTCM via DMA (ne10 == 1)
স্থির শূন্য দ্বিমিক_thread_scalar(unsigned পূর্ণসংখ্যা nth, unsigned পূর্ণসংখ্যা ith, শূন্য * তথ্য) {
    কাঠামো htp_দ্বিমিক_context * bctx = (কাঠামো htp_দ্বিমিক_context *) তথ্য;
    কাঠামো htp_ops_context * octx = bctx->octx;
    htp_দ্বিমিক_preamble;

    ধ্রুবক uint32_t elem_আকার = (src0->type == HTP_TYPE_F32) ? আকার(দশমিক) : আকার(_Float16);
    ধ্রুবক size_t plane_stride = (size_t) ne12 * ne11 * elem_আকার;
    ধ্রুবক size_t row_stride   = (size_t) ne11 * elem_আকার;

    ধ্রুবক uint32_t row_আকার_bytes = bctx->row_আকার_bytes;
    ধ্রুবক uint32_t start_row = bctx->row_start + bctx->nrows_per_thread * ith;
    ধ্রুবক uint32_t end_row   = MIN(start_row + bctx->nrows_per_thread, bctx->row_start + bctx->যোগ_rows);
    যদি (start_row >= end_row) ফেরত;

    FARF(উপরের_2, "binary-scalar: %d/%d (%u:%u) row-size %u (%u)",
         ith, nth, start_row, end_row, nb01, bctx->dst_row_আকার_aligned);

    ধ্রুবক কাঠামো htp_দ্বিমিক_vtcm_layout * layout = &bctx->vtcm_layout;
    uint8_t * src0_spad_base = VTCM_LAYOUT_PTR(uint8_t, bctx->vtcm_base, layout->off_src0) + (ith * layout->src0_bytes_per_thread);
    uint8_t * dst_spad_base  = VTCM_LAYOUT_PTR(uint8_t, bctx->vtcm_base, layout->off_dst)  + (ith * layout->dst_bytes_per_thread);
    size_t src0_spad_half    = layout->src0_spad_half_আকার;
    size_t dst_spad_half     = layout->dst_spad_half_আকার;
    ধ্রুবক শূন্য * s1_table    = VTCM_LAYOUT_PTR(ধ্রুবক শূন্য, bctx->vtcm_base, layout->off_src1);

    dma_queue * dma_q = octx->ctx->dma[ith];
    uint32_t ir_prefetch = start_row;
    পূর্ণসংখ্যা spad_idx = 0;

    জন্য (পূর্ণসংখ্যা k = 0; k < 2 && ir_prefetch < end_row; k++) {
        uint32_t current_block_আকার = calc_block_আকার(bctx, ir_prefetch, end_row, ne01, ne02);
        uint32_t i03, i02, i01, rem;
        i03 = fastdiv(ir_prefetch, &bctx->src0_dim12_div);
        rem = ir_prefetch - i03 * (ne02 * ne01);
        i02 = fastdiv(rem, &bctx->src0_dim1_div);
        i01 = rem - i02 * ne01;

        dma_addr_t src0_curr = src0->তথ্য + i03 * nb03 + i02 * nb02 + i01 * nb01;
        dma_addr_t dst_curr  = dst->তথ্য  + i03 * nb3  + i02 * nb2  + i01 * nb1;

        uint8_t * s0_spad = src0_spad_base + spad_idx * src0_spad_half;
        uint8_t * d_spad  = dst_spad_base  + spad_idx * dst_spad_half;

        dma_queue_ঠেলো(dma_q, dma_make_তথ্য(dst_curr, d_spad), nb1, bctx->dst_row_আকার_aligned, row_আকার_bytes, 0);
        dma_queue_ঠেলো(dma_q, dma_make_তথ্য(s0_spad, src0_curr), bctx->src0_row_আকার_aligned, nb01, row_আকার_bytes,
                                                             current_block_আকার);
        ir_prefetch += current_block_আকার;
        spad_idx ^= 1;
    }

    কাঠামো htp_thread_trace * tr = &octx->ctx->trace[ith];
    compute_scalar_t compute = (compute_scalar_t) bctx->compute;

    জন্য (uint32_t ir = start_row; ir < end_row; ) {
        uint32_t current_block_আকার = calc_block_আকার(bctx, ir, end_row, ne01, ne02);

        uint8_t * d_spad  = (uint8_t *) dma_queue_বের_করো(dma_q).src;
        uint8_t * s0_spad = (uint8_t *) dma_queue_বের_করো(dma_q).dst;

        uint32_t i03, i02, i01, rem;
        i03 = fastdiv(ir, &bctx->src0_dim12_div);
        rem = ir - i03 * (ne02 * ne01);
        i02 = fastdiv(rem, &bctx->src0_dim1_div);
        i01 = rem - i02 * ne01;

        uint32_t i13 = fastmodulo(i03, ne13, &bctx->src1_dim3_div);
        uint32_t i12 = fastmodulo(i02, ne12, &bctx->src1_dim2_div);
        uint32_t i11 = fastmodulo(i01, ne11, &bctx->src1_dim1_div);
        ধ্রুবক uint8_t * scalar_table = (ধ্রুবক uint8_t *) s1_table +
            i13 * plane_stride + i12 * row_stride;

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);
        compute(d_spad, s0_spad, scalar_table, i11, ne11, current_block_আকার,
                bctx->dst_row_আকার_aligned, bctx->src0_row_আকার_aligned, ne00);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);

        dma_addr_t dst_curr = dst->তথ্য + i03 * nb3 + i02 * nb2 + i01 * nb1;
        dma_queue_ঠেলো(dma_q, dma_make_তথ্য(dst_curr, d_spad), nb1, bctx->dst_row_আকার_aligned, row_আকার_bytes,
                                                                current_block_আকার);

        যদি (ir_prefetch < end_row) {
            uint32_t পরবর্তী_block_আকার = calc_block_আকার(bctx, ir_prefetch, end_row, ne01, ne02);
            uint32_t p03, p02, p01, prem;
            p03 = fastdiv(ir_prefetch, &bctx->src0_dim12_div);
            prem = ir_prefetch - p03 * (ne02 * ne01);
            p02 = fastdiv(prem, &bctx->src0_dim1_div);
            p01 = prem - p02 * ne01;
            dma_addr_t s0_পরবর্তী = src0->তথ্য + p03 * nb03 + p02 * nb02 + p01 * nb01;
            dma_queue_ঠেলো(dma_q, dma_make_তথ্য(s0_spad, s0_পরবর্তী), bctx->src0_row_আকার_aligned, nb01, row_আকার_bytes,
                                                               পরবর্তী_block_আকার);
            ir_prefetch += পরবর্তী_block_আকার;
        }
        ir += current_block_আকার;
    }

    dma_queue_flush(dma_q);
}

// 2. Vector Same Shape (ne1x == ne0x) or Simple Broadcast
স্থির শূন্য দ্বিমিক_thread_vector_same_shape(unsigned পূর্ণসংখ্যা nth, unsigned পূর্ণসংখ্যা ith, শূন্য * তথ্য) {
    কাঠামো htp_দ্বিমিক_context * bctx = (কাঠামো htp_দ্বিমিক_context *) তথ্য;
    কাঠামো htp_ops_context * octx = bctx->octx;
    htp_দ্বিমিক_preamble;

    ধ্রুবক uint32_t row_আকার_bytes = bctx->row_আকার_bytes;
    ধ্রুবক uint32_t start_row = bctx->row_start + bctx->nrows_per_thread * ith;
    ধ্রুবক uint32_t end_row   = MIN(start_row + bctx->nrows_per_thread, bctx->row_start + bctx->যোগ_rows);
    যদি (start_row >= end_row) ফেরত;

    FARF(উপরের_2, "binary-same-shape: %d/%d (%u:%u) row-size %u (%u)",
         ith, nth, start_row, end_row, nb01, bctx->dst_row_আকার_aligned);

    ধ্রুবক কাঠামো htp_দ্বিমিক_vtcm_layout * layout = &bctx->vtcm_layout;
    uint8_t * src0_spad_base = VTCM_LAYOUT_PTR(uint8_t, bctx->vtcm_base, layout->off_src0) + (ith * layout->src0_bytes_per_thread);
    uint8_t * src1_spad_base = VTCM_LAYOUT_PTR(uint8_t, bctx->vtcm_base, layout->off_src1) + (ith * layout->src1_bytes_per_thread);
    uint8_t * dst_spad_base  = VTCM_LAYOUT_PTR(uint8_t, bctx->vtcm_base, layout->off_dst)  + (ith * layout->dst_bytes_per_thread);

    size_t src0_spad_half = layout->src0_spad_half_আকার;
    size_t src1_spad_half = layout->src1_spad_half_আকার;
    size_t dst_spad_half  = layout->dst_spad_half_আকার;

    dma_queue * dma_q = octx->ctx->dma[ith];
    uint32_t ir_prefetch = start_row;
    পূর্ণসংখ্যা spad_idx = 0;

    জন্য (পূর্ণসংখ্যা k = 0; k < 2 && ir_prefetch < end_row; k++) {
        uint32_t current_block_আকার = calc_block_আকার(bctx, ir_prefetch, end_row, ne01, ne02);
        uint32_t i03, i02, i01, rem;
        i03 = fastdiv(ir_prefetch, &bctx->src0_dim12_div);
        rem = ir_prefetch - i03 * (ne02 * ne01);
        i02 = fastdiv(rem, &bctx->src0_dim1_div);
        i01 = rem - i02 * ne01;

        uint32_t i13 = (ne13 == 1) ? 0 : i03;
        uint32_t i12 = (ne12 == 1) ? 0 : i02;
        uint32_t i11 = (ne11 == 1) ? 0 : i01;

        dma_addr_t src0_curr = src0->তথ্য + i03 * nb03 + i02 * nb02 + i01 * nb01;
        dma_addr_t src1_curr = src1->তথ্য + i13 * nb13 + i12 * nb12 + i11 * nb11;
        dma_addr_t dst_curr  = dst->তথ্য  + i03 * nb3  + i02 * nb2  + i01 * nb1;

        uint8_t * s0_spad = src0_spad_base + spad_idx * src0_spad_half;
        uint8_t * s1_spad = src1_spad_base + spad_idx * src1_spad_half;
        uint8_t * d_spad  = dst_spad_base  + spad_idx * dst_spad_half;

        dma_queue_ঠেলো(dma_q, dma_make_তথ্য(dst_curr, d_spad), nb1, bctx->dst_row_আকার_aligned, row_আকার_bytes, 0);
        dma_queue_ঠেলো(dma_q, dma_make_তথ্য(s0_spad, src0_curr), bctx->src0_row_আকার_aligned, nb01, row_আকার_bytes, current_block_আকার);
        dma_queue_ঠেলো(dma_q, dma_make_তথ্য(s1_spad, src1_curr), bctx->src1_row_আকার_aligned, nb11, row_আকার_bytes, current_block_আকার);
        ir_prefetch += current_block_আকার;
        spad_idx ^= 1;
    }

    কাঠামো htp_thread_trace * tr = &octx->ctx->trace[ith];
    compute_same_shape_t compute = (compute_same_shape_t) bctx->compute;

    জন্য (uint32_t ir = start_row; ir < end_row; ) {
        uint32_t current_block_আকার = calc_block_আকার(bctx, ir, end_row, ne01, ne02);
        uint8_t * d_spad  = (uint8_t *) dma_queue_বের_করো(dma_q).src;
        uint8_t * s0_spad = (uint8_t *) dma_queue_বের_করো(dma_q).dst;
        uint8_t * s1_spad = (uint8_t *) dma_queue_বের_করো(dma_q).dst;

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);
        compute(d_spad, s0_spad, s1_spad, current_block_আকার,
                bctx->dst_row_আকার_aligned, bctx->src0_row_আকার_aligned, bctx->src1_row_আকার_aligned, ne00);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);

        uint32_t i03, i02, i01, rem;
        i03 = fastdiv(ir, &bctx->src0_dim12_div);
        rem = ir - i03 * (ne02 * ne01);
        i02 = fastdiv(rem, &bctx->src0_dim1_div);
        i01 = rem - i02 * ne01;
        dma_addr_t dst_curr = dst->তথ্য + i03 * nb3 + i02 * nb2 + i01 * nb1;
        dma_queue_ঠেলো(dma_q, dma_make_তথ্য(dst_curr, d_spad), nb1, bctx->dst_row_আকার_aligned, row_আকার_bytes, current_block_আকার);

        যদি (ir_prefetch < end_row) {
            uint32_t পরবর্তী_block_আকার = calc_block_আকার(bctx, ir_prefetch, end_row, ne01, ne02);
            uint32_t p03, p02, p01, prem;
            p03 = fastdiv(ir_prefetch, &bctx->src0_dim12_div);
            prem = ir_prefetch - p03 * (ne02 * ne01);
            p02 = fastdiv(prem, &bctx->src0_dim1_div);
            p01 = prem - p02 * ne01;

            uint32_t p13 = (ne13 == 1) ? 0 : p03;
            uint32_t p12 = (ne12 == 1) ? 0 : p02;
            uint32_t p11 = (ne11 == 1) ? 0 : p01;

            dma_addr_t s0_পরবর্তী = src0->তথ্য + p03 * nb03 + p02 * nb02 + p01 * nb01;
            dma_addr_t s1_পরবর্তী = src1->তথ্য + p13 * nb13 + p12 * nb12 + p11 * nb11;

            dma_queue_ঠেলো(dma_q, dma_make_তথ্য(s0_spad, s0_পরবর্তী), bctx->src0_row_আকার_aligned, nb01, row_আকার_bytes, পরবর্তী_block_আকার);
            dma_queue_ঠেলো(dma_q, dma_make_তথ্য(s1_spad, s1_পরবর্তী), bctx->src1_row_আকার_aligned, nb11, row_আকার_bytes, পরবর্তী_block_আকার);

            ir_prefetch += পরবর্তী_block_আকার;
        }
        ir += current_block_আকার;
    }

    dma_queue_flush(dma_q);
}

// 3. Row Broadcast (ne11 == 1, ne12 == 1, single row src1)
স্থির শূন্য দ্বিমিক_thread_vector_row_broadcast(unsigned পূর্ণসংখ্যা nth, unsigned পূর্ণসংখ্যা ith, শূন্য * তথ্য) {
    কাঠামো htp_দ্বিমিক_context * bctx = (কাঠামো htp_দ্বিমিক_context *) তথ্য;
    কাঠামো htp_ops_context * octx = bctx->octx;
    htp_দ্বিমিক_preamble;

    ধ্রুবক uint32_t row_আকার_bytes = bctx->row_আকার_bytes;
    ধ্রুবক uint32_t start_row  = bctx->row_start + bctx->nrows_per_thread * ith;
    ধ্রুবক uint32_t end_row    = MIN(start_row + bctx->nrows_per_thread, bctx->row_start + bctx->যোগ_rows);
    যদি (start_row >= end_row) ফেরত;

    FARF(উপরের_2, "binary-row-bcast: %d/%d (%u:%u) row-size %u (%u)",
         ith, nth, start_row, end_row, nb01, bctx->dst_row_আকার_aligned);

    ধ্রুবক কাঠামো htp_দ্বিমিক_vtcm_layout * layout = &bctx->vtcm_layout;
    uint8_t * src0_spad_base = VTCM_LAYOUT_PTR(uint8_t, bctx->vtcm_base, layout->off_src0) + (ith * layout->src0_bytes_per_thread);
    uint8_t * dst_spad_base  = VTCM_LAYOUT_PTR(uint8_t, bctx->vtcm_base, layout->off_dst)  + (ith * layout->dst_bytes_per_thread);

    size_t src0_spad_half = layout->src0_spad_half_আকার;
    size_t dst_spad_half  = layout->dst_spad_half_আকার;

    dma_queue * dma_q = octx->ctx->dma[ith];
    uint32_t ir_prefetch = start_row;
    পূর্ণসংখ্যা spad_idx = 0;

    শূন্য * s1_ptr = VTCM_LAYOUT_PTR(শূন্য, bctx->vtcm_base, layout->off_src1);

    জন্য (পূর্ণসংখ্যা k = 0; k < 2 && ir_prefetch < end_row; k++) {
        uint32_t current_block_আকার = calc_block_আকার(bctx, ir_prefetch, end_row, ne01, ne02);
        uint32_t i03 = fastdiv(ir_prefetch, &bctx->src0_dim12_div);
        uint32_t rem = ir_prefetch - i03 * (ne02 * ne01);
        uint32_t i02 = fastdiv(rem, &bctx->src0_dim1_div);
        uint32_t i01 = rem - i02 * ne01;

        dma_addr_t src0_curr = src0->তথ্য + i03 * nb03 + i02 * nb02 + i01 * nb01;
        dma_addr_t dst_curr  = dst->তথ্য  + i03 * nb3  + i02 * nb2  + i01 * nb1;

        uint8_t * s0_spad = src0_spad_base + spad_idx * src0_spad_half;
        uint8_t * d_spad  = dst_spad_base  + spad_idx * dst_spad_half;

        dma_queue_ঠেলো(dma_q, dma_make_তথ্য(dst_curr, d_spad), nb1, bctx->dst_row_আকার_aligned, row_আকার_bytes, 0);
        dma_queue_ঠেলো(dma_q, dma_make_তথ্য(s0_spad, src0_curr), bctx->src0_row_আকার_aligned, nb01, row_আকার_bytes, current_block_আকার);
        ir_prefetch += current_block_আকার;
        spad_idx ^= 1;
    }

    কাঠামো htp_thread_trace * tr = &octx->ctx->trace[ith];
    compute_row_bcast_t compute = (compute_row_bcast_t) bctx->compute;

    জন্য (uint32_t ir = start_row; ir < end_row; ) {
        uint32_t current_block_আকার = calc_block_আকার(bctx, ir, end_row, ne01, ne02);
        uint8_t * d_spad  = (uint8_t *) dma_queue_বের_করো(dma_q).src;
        uint8_t * s0_spad = (uint8_t *) dma_queue_বের_করো(dma_q).dst;

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);
        compute(d_spad, s0_spad, (ধ্রুবক uint8_t *)s1_ptr, current_block_আকার,
                bctx->dst_row_আকার_aligned, bctx->src0_row_আকার_aligned, ne00);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);

        uint32_t i03 = fastdiv(ir, &bctx->src0_dim12_div);
        uint32_t rem = ir - i03 * (ne02 * ne01);
        uint32_t i02 = fastdiv(rem, &bctx->src0_dim1_div);
        uint32_t i01 = rem - i02 * ne01;
        dma_addr_t dst_curr = dst->তথ্য + i03 * nb3 + i02 * nb2 + i01 * nb1;
        dma_queue_ঠেলো(dma_q, dma_make_তথ্য(dst_curr, d_spad), nb1, bctx->dst_row_আকার_aligned, row_আকার_bytes, current_block_আকার);

        যদি (ir_prefetch < end_row) {
            uint32_t পরবর্তী_block_আকার = calc_block_আকার(bctx, ir_prefetch, end_row, ne01, ne02);
            uint32_t p03, p02, p01, prem;
            p03  = fastdiv(ir_prefetch, &bctx->src0_dim12_div);
            prem = ir_prefetch - p03 * (ne02 * ne01);
            p02  = fastdiv(prem, &bctx->src0_dim1_div);
            p01  = prem - p02 * ne01;
            dma_addr_t s0_পরবর্তী = src0->তথ্য + p03 * nb03 + p02 * nb02 + p01 * nb01;
            dma_queue_ঠেলো(dma_q, dma_make_তথ্য(s0_spad, s0_পরবর্তী), bctx->src0_row_আকার_aligned, nb01, row_আকার_bytes, পরবর্তী_block_আকার);
            ir_prefetch += পরবর্তী_block_আকার;
        }
        ir += current_block_আকার;
    }

    dma_queue_flush(dma_q);
}

// 4. Vector Complex (ne10 == ne00, complex broadcast)
স্থির শূন্য দ্বিমিক_thread_vector_complex(unsigned পূর্ণসংখ্যা nth, unsigned পূর্ণসংখ্যা ith, শূন্য * তথ্য) {
    কাঠামো htp_দ্বিমিক_context * bctx = (কাঠামো htp_দ্বিমিক_context *) তথ্য;
    কাঠামো htp_ops_context * octx = bctx->octx;
    htp_দ্বিমিক_preamble;

    ধ্রুবক uint32_t row_আকার_bytes = bctx->row_আকার_bytes;
    ধ্রুবক uint32_t start_row  = bctx->row_start + bctx->nrows_per_thread * ith;
    ধ্রুবক uint32_t end_row    = MIN(start_row + bctx->nrows_per_thread, bctx->row_start + bctx->যোগ_rows);
    যদি (start_row >= end_row) ফেরত;

    FARF(উপরের_2, "binary-complex: %d/%d (%u:%u) row-size %u (%u)",
         ith, nth, start_row, end_row, nb01, bctx->dst_row_আকার_aligned);

    ধ্রুবক কাঠামো htp_দ্বিমিক_vtcm_layout * layout = &bctx->vtcm_layout;
    uint8_t * src0_spad_base = VTCM_LAYOUT_PTR(uint8_t, bctx->vtcm_base, layout->off_src0) + (ith * layout->src0_bytes_per_thread);
    uint8_t * dst_spad_base  = VTCM_LAYOUT_PTR(uint8_t, bctx->vtcm_base, layout->off_dst)  + (ith * layout->dst_bytes_per_thread);
    size_t src0_spad_half    = layout->src0_spad_half_আকার;
    size_t dst_spad_half     = layout->dst_spad_half_আকার;
    ধ্রুবক uint8_t * vtcm_src1 = VTCM_LAYOUT_PTR(ধ্রুবক uint8_t, bctx->vtcm_base, layout->off_src1);

    ধ্রুবক uint32_t elem_আকার = (src0->type == HTP_TYPE_F32) ? আকার(দশমিক) : আকার(_Float16);
    ধ্রুবক size_t vtcm_row_stride   = (size_t) ne10 * elem_আকার;
    ধ্রুবক size_t vtcm_plane_stride = (size_t) ne11 * vtcm_row_stride;
    ধ্রুবক size_t vtcm_vol_stride   = (size_t) ne12 * vtcm_plane_stride;

    dma_queue * dma_q = octx->ctx->dma[ith];
    uint32_t ir_prefetch = start_row;
    পূর্ণসংখ্যা spad_idx = 0;

    জন্য (পূর্ণসংখ্যা k = 0; k < 2 && ir_prefetch < end_row; k++) {
        uint32_t current_block_আকার = calc_block_আকার(bctx, ir_prefetch, end_row, ne01, ne02);
        uint32_t i03 = fastdiv(ir_prefetch, &bctx->src0_dim12_div);
        uint32_t rem = ir_prefetch - i03 * (ne02 * ne01);
        uint32_t i02 = fastdiv(rem, &bctx->src0_dim1_div);
        uint32_t i01 = rem - i02 * ne01;

        dma_addr_t src0_curr = src0->তথ্য + i03 * nb03 + i02 * nb02 + i01 * nb01;
        dma_addr_t dst_curr  = dst->তথ্য  + i03 * nb3  + i02 * nb2  + i01 * nb1;

        uint8_t * s0_spad = src0_spad_base + spad_idx * src0_spad_half;
        uint8_t * d_spad  = dst_spad_base  + spad_idx * dst_spad_half;

        dma_queue_ঠেলো(dma_q, dma_make_তথ্য(dst_curr, d_spad), nb1, bctx->dst_row_আকার_aligned, row_আকার_bytes, 0);
        dma_queue_ঠেলো(dma_q, dma_make_তথ্য(s0_spad, src0_curr), bctx->src0_row_আকার_aligned, nb01, row_আকার_bytes, current_block_আকার);
        ir_prefetch += current_block_আকার;
        spad_idx ^= 1;
    }

    কাঠামো htp_thread_trace * tr = &octx->ctx->trace[ith];
    compute_complex_t compute = (compute_complex_t) bctx->compute;

    জন্য (uint32_t ir = start_row; ir < end_row; ) {
        uint32_t current_block_আকার = calc_block_আকার(bctx, ir, end_row, ne01, ne02);
        uint8_t * d_spad = (uint8_t *) dma_queue_বের_করো(dma_q).src;
        uint8_t * s0_spad = (uint8_t *) dma_queue_বের_করো(dma_q).dst;

        uint32_t i03 = fastdiv(ir, &bctx->src0_dim12_div);
        uint32_t rem = ir - i03 * (ne02 * ne01);
        uint32_t i02 = fastdiv(rem, &bctx->src0_dim1_div);
        uint32_t i01 = rem - i02 * ne01;

        uint32_t i13 = fastmodulo(i03, ne13, &bctx->src1_dim3_div);
        uint32_t i12 = fastmodulo(i02, ne12, &bctx->src1_dim2_div);
        ধ্রুবক uint8_t * src1_plane = vtcm_src1 + i13 * vtcm_vol_stride + i12 * vtcm_plane_stride;

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);
        compute(d_spad, s0_spad, src1_plane, i01, ne11, &bctx->src1_dim1_div, vtcm_row_stride,
                current_block_আকার, bctx->dst_row_আকার_aligned, bctx->src0_row_আকার_aligned, ne00);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);

        dma_addr_t dst_curr = dst->তথ্য + i03 * nb3 + i02 * nb2 + i01 * nb1;
        dma_queue_ঠেলো(dma_q, dma_make_তথ্য(dst_curr, d_spad), nb1, bctx->dst_row_আকার_aligned, row_আকার_bytes, current_block_আকার);

        যদি (ir_prefetch < end_row) {
            uint32_t পরবর্তী_block_আকার = calc_block_আকার(bctx, ir_prefetch, end_row, ne01, ne02);
            uint32_t p03, p02, p01, prem;
            p03  = fastdiv(ir_prefetch, &bctx->src0_dim12_div);
            prem = ir_prefetch - p03 * (ne02 * ne01);
            p02  = fastdiv(prem, &bctx->src0_dim1_div);
            p01  = prem - p02 * ne01;
            dma_addr_t s0_পরবর্তী = src0->তথ্য + p03 * nb03 + p02 * nb02 + p01 * nb01;
            dma_queue_ঠেলো(dma_q, dma_make_তথ্য(s0_spad, s0_পরবর্তী), bctx->src0_row_আকার_aligned, nb01, row_আকার_bytes, পরবর্তী_block_আকার);
            ir_prefetch += পরবর্তী_block_আকার;
        }
        ir += current_block_আকার;
    }

    dma_queue_flush(dma_q);
}

// 5. Element Repeat (ne10 != ne00)
স্থির শূন্য দ্বিমিক_thread_element_repeat(unsigned পূর্ণসংখ্যা nth, unsigned পূর্ণসংখ্যা ith, শূন্য * তথ্য) {
    কাঠামো htp_দ্বিমিক_context * bctx = (কাঠামো htp_দ্বিমিক_context *) তথ্য;
    কাঠামো htp_ops_context * octx = bctx->octx;
    htp_দ্বিমিক_preamble;

    ধ্রুবক uint32_t row_আকার_bytes = bctx->row_আকার_bytes;
    ধ্রুবক uint32_t start_row  = bctx->row_start + bctx->nrows_per_thread * ith;
    ধ্রুবক uint32_t end_row    = MIN(start_row + bctx->nrows_per_thread, bctx->row_start + bctx->যোগ_rows);
    যদি (start_row >= end_row) ফেরত;

    ধ্রুবক কাঠামো htp_দ্বিমিক_vtcm_layout * layout = &bctx->vtcm_layout;
    uint8_t * src0_spad_base = VTCM_LAYOUT_PTR(uint8_t, bctx->vtcm_base, layout->off_src0) + (ith * layout->src0_bytes_per_thread);
    uint8_t * dst_spad_base  = VTCM_LAYOUT_PTR(uint8_t, bctx->vtcm_base, layout->off_dst)  + (ith * layout->dst_bytes_per_thread);
    size_t src0_spad_half    = layout->src0_spad_half_আকার;
    size_t dst_spad_half     = layout->dst_spad_half_আকার;
    ধ্রুবক uint8_t * vtcm_src1 = VTCM_LAYOUT_PTR(ধ্রুবক uint8_t, bctx->vtcm_base, layout->off_src1);

    ধ্রুবক uint32_t elem_আকার = (src0->type == HTP_TYPE_F32) ? আকার(দশমিক) : আকার(_Float16);
    ধ্রুবক size_t vtcm_row_stride   = (size_t) ne10 * elem_আকার;
    ধ্রুবক size_t vtcm_plane_stride = (size_t) ne11 * vtcm_row_stride;
    ধ্রুবক size_t vtcm_vol_stride   = (size_t) ne12 * vtcm_plane_stride;

    FARF(উপরের_2, "binary-repeat: %d/%d (%u:%u) row-size %u (%u)",
         ith, nth, start_row, end_row, nb01, bctx->dst_row_আকার_aligned);

    dma_queue * dma_q = octx->ctx->dma[ith];
    uint32_t ir_prefetch = start_row;
    পূর্ণসংখ্যা spad_idx = 0;

    জন্য (পূর্ণসংখ্যা k = 0; k < 2 && ir_prefetch < end_row; k++) {
        uint32_t current_block_আকার = calc_block_আকার(bctx, ir_prefetch, end_row, ne01, ne02);
        uint32_t i03 = fastdiv(ir_prefetch, &bctx->src0_dim12_div);
        uint32_t rem = ir_prefetch - i03 * (ne02 * ne01);
        uint32_t i02 = fastdiv(rem, &bctx->src0_dim1_div);
        uint32_t i01 = rem - i02 * ne01;

        dma_addr_t src0_curr = src0->তথ্য + i03 * nb03 + i02 * nb02 + i01 * nb01;
        dma_addr_t dst_curr  = dst->তথ্য  + i03 * nb3  + i02 * nb2  + i01 * nb1;

        uint8_t * s0_spad = src0_spad_base + spad_idx * src0_spad_half;
        uint8_t * d_spad  = dst_spad_base  + spad_idx * dst_spad_half;

        dma_queue_ঠেলো(dma_q, dma_make_তথ্য(dst_curr, d_spad), nb1, bctx->dst_row_আকার_aligned, row_আকার_bytes, 0);
        dma_queue_ঠেলো(dma_q, dma_make_তথ্য(s0_spad, src0_curr), bctx->src0_row_আকার_aligned, nb01, row_আকার_bytes, current_block_আকার);
        ir_prefetch += current_block_আকার;
        spad_idx ^= 1;
    }

    কাঠামো htp_thread_trace * tr = &octx->ctx->trace[ith];
    compute_repeat_t compute = (compute_repeat_t) bctx->compute;

    জন্য (uint32_t ir = start_row; ir < end_row; ) {
        uint32_t current_block_আকার = calc_block_আকার(bctx, ir, end_row, ne01, ne02);
        uint8_t * d_spad  = (uint8_t *) dma_queue_বের_করো(dma_q).src;
        uint8_t * s0_spad = (uint8_t *) dma_queue_বের_করো(dma_q).dst;

        uint32_t i03 = fastdiv(ir, &bctx->src0_dim12_div);
        uint32_t rem = ir - i03 * (ne02 * ne01);
        uint32_t i02 = fastdiv(rem, &bctx->src0_dim1_div);
        uint32_t i01 = rem - i02 * ne01;

        uint32_t i13 = fastmodulo(i03, ne13, &bctx->src1_dim3_div);
        uint32_t i12 = fastmodulo(i02, ne12, &bctx->src1_dim2_div);
        ধ্রুবক uint8_t * src1_plane = vtcm_src1 + i13 * vtcm_vol_stride + i12 * vtcm_plane_stride;

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);
        compute(d_spad, s0_spad, src1_plane, i01, ne11, &bctx->src1_dim1_div, vtcm_row_stride,
                current_block_আকার, bctx->dst_row_আকার_aligned, bctx->src0_row_আকার_aligned, ne00, ne10);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);

        dma_addr_t dst_curr = dst->তথ্য + i03 * nb3 + i02 * nb2 + i01 * nb1;
        dma_queue_ঠেলো(dma_q, dma_make_তথ্য(dst_curr, d_spad), nb1, bctx->dst_row_আকার_aligned, row_আকার_bytes, current_block_আকার);

        যদি (ir_prefetch < end_row) {
            uint32_t পরবর্তী_block_আকার = calc_block_আকার(bctx, ir_prefetch, end_row, ne01, ne02);
            uint32_t p03, p02, p01, prem;
            p03  = fastdiv(ir_prefetch, &bctx->src0_dim12_div);
            prem = ir_prefetch - p03 * (ne02 * ne01);
            p02  = fastdiv(prem, &bctx->src0_dim1_div);
            p01  = prem - p02 * ne01;
            dma_addr_t s0_পরবর্তী = src0->তথ্য + p03 * nb03 + p02 * nb02 + p01 * nb01;
            dma_queue_ঠেলো(dma_q, dma_make_তথ্য(s0_spad, s0_পরবর্তী), bctx->src0_row_আকার_aligned, nb01, row_আকার_bytes, পরবর্তী_block_আকার);
            ir_prefetch += পরবর্তী_block_আকার;
        }
        ir += current_block_আকার;
    }

    dma_queue_flush(dma_q);
}

// 6. ADD_ID (src1 gathered via src2 indices)
স্থির শূন্য দ্বিমিক_thread_add_id_f32(unsigned পূর্ণসংখ্যা nth, unsigned পূর্ণসংখ্যা ith, শূন্য * তথ্য) {
    কাঠামো htp_দ্বিমিক_context * bctx = (কাঠামো htp_দ্বিমিক_context *) তথ্য;
    কাঠামো htp_ops_context * octx = bctx->octx;

    ধ্রুবক কাঠামো htp_tensor * src0 = octx->src[0];
    ধ্রুবক কাঠামো htp_tensor * src1 = octx->src[1];
    ধ্রুবক কাঠামো htp_tensor * dst  = octx->dst;

    ধ্রুবক uint32_t ne00 = src0->ne[0];
    ধ্রুবক uint32_t ne01 = src0->ne[1];
    ধ্রুবক uint32_t ne02 = src0->ne[2];
    ধ্রুবক uint32_t ne03 = src0->ne[3];

    ধ্রুবক uint32_t nb01 = src0->nb[1];
    ধ্রুবক uint32_t nb02 = src0->nb[2];
    ধ্রুবক uint32_t nb03 = src0->nb[3];
    ধ্রুবক uint32_t src1_stride = bctx->src1_row_আকার_aligned;

    ধ্রুবক uint32_t nb1 = dst->nb[1];
    ধ্রুবক uint32_t nb2 = dst->nb[2];
    ধ্রুবক uint32_t nb3 = dst->nb[3];

    ধ্রুবক uint32_t row_আকার_bytes = bctx->row_আকার_bytes;
    ধ্রুবক uint32_t start_row = bctx->row_start + bctx->nrows_per_thread * ith;
    ধ্রুবক uint32_t end_row   = MIN(start_row + bctx->nrows_per_thread, bctx->row_start + bctx->যোগ_rows);
    যদি (start_row >= end_row) ফেরত;

    ধ্রুবক কাঠামো htp_দ্বিমিক_vtcm_layout * layout = &bctx->vtcm_layout;
    uint8_t * src0_spad_base = VTCM_LAYOUT_PTR(uint8_t, bctx->vtcm_base, layout->off_src0) + (ith * layout->src0_bytes_per_thread);
    uint8_t * dst_spad_base  = VTCM_LAYOUT_PTR(uint8_t, bctx->vtcm_base, layout->off_dst)  + (ith * layout->dst_bytes_per_thread);
    ধ্রুবক uint8_t * vtcm_src1 = VTCM_LAYOUT_PTR(ধ্রুবক uint8_t, bctx->vtcm_base, layout->off_src1);
    ধ্রুবক int32_t * vtcm_src2 = VTCM_LAYOUT_PTR(ধ্রুবক int32_t, bctx->vtcm_base, layout->off_src2);
    size_t src0_spad_half    = layout->src0_spad_half_আকার;
    size_t dst_spad_half     = layout->dst_spad_half_আকার;

    dma_queue * dma_q = octx->ctx->dma[ith];
    uint32_t ir_prefetch = start_row;
    পূর্ণসংখ্যা spad_idx = 0;

    জন্য (পূর্ণসংখ্যা k = 0; k < 2 && ir_prefetch < end_row; k++) {
        uint32_t current_block_আকার = calc_block_আকার(bctx, ir_prefetch, end_row, ne01, ne02);
        uint32_t i03 = fastdiv(ir_prefetch, &bctx->src0_dim12_div);
        uint32_t rem = ir_prefetch - i03 * (ne02 * ne01);
        uint32_t i02 = fastdiv(rem, &bctx->src0_dim1_div);
        uint32_t i01 = rem - i02 * ne01;

        dma_addr_t src0_curr = src0->তথ্য + i03 * nb03 + i02 * nb02 + i01 * nb01;
        dma_addr_t dst_curr  = dst->তথ্য  + i03 * nb3  + i02 * nb2  + i01 * nb1;

        uint8_t * s0_spad = src0_spad_base + spad_idx * src0_spad_half;
        uint8_t * d_spad  = dst_spad_base  + spad_idx * dst_spad_half;

        dma_queue_ঠেলো(dma_q, dma_make_তথ্য(dst_curr, d_spad), nb1, bctx->dst_row_আকার_aligned, row_আকার_bytes, 0);
        dma_queue_ঠেলো(dma_q, dma_make_তথ্য(s0_spad, src0_curr), bctx->src0_row_আকার_aligned, nb01, row_আকার_bytes, current_block_আকার);
        ir_prefetch += current_block_আকার;
        spad_idx ^= 1;
    }

    কাঠামো htp_thread_trace * tr = &octx->ctx->trace[ith];

    জন্য (uint32_t ir = start_row; ir < end_row; ) {
        uint32_t current_block_আকার = calc_block_আকার(bctx, ir, end_row, ne01, ne02);
        uint8_t * d_spad = (uint8_t *) dma_queue_বের_করো(dma_q).src;
        uint8_t * s0_spad = (uint8_t *) dma_queue_বের_করো(dma_q).dst;

        uint32_t i03 = fastdiv(ir, &bctx->src0_dim12_div);
        uint32_t rem = ir - i03 * (ne02 * ne01);
        uint32_t i02 = fastdiv(rem, &bctx->src0_dim1_div);
        uint32_t i01 = rem - i02 * ne01;

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);
        compute_add_id_t compute = (compute_add_id_t) bctx->compute;
        compute(d_spad, s0_spad, vtcm_src1, vtcm_src2,
                i01, i02, ne01, src1_stride,
                current_block_আকার, bctx->dst_row_আকার_aligned, bctx->src0_row_আকার_aligned, ne00);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);

        dma_addr_t dst_curr = dst->তথ্য + i03 * nb3 + i02 * nb2 + i01 * nb1;
        dma_queue_ঠেলো(dma_q, dma_make_তথ্য(dst_curr, d_spad), nb1, bctx->dst_row_আকার_aligned, row_আকার_bytes, current_block_আকার);

        যদি (ir_prefetch < end_row) {
             uint32_t পরবর্তী_block_আকার = calc_block_আকার(bctx, ir_prefetch, end_row, ne01, ne02);
             uint32_t p03, p02, p01, prem;
             p03  = fastdiv(ir_prefetch, &bctx->src0_dim12_div);
             prem = ir_prefetch - p03 * (ne02 * ne01);
             p02  = fastdiv(prem, &bctx->src0_dim1_div);
             p01  = prem - p02 * ne01;
             dma_addr_t s0_পরবর্তী = src0->তথ্য + p03 * nb03 + p02 * nb02 + p01 * nb01;
             dma_queue_ঠেলো(dma_q, dma_make_তথ্য(s0_spad, s0_পরবর্তী), bctx->src0_row_আকার_aligned, nb01, row_আকার_bytes, পরবর্তী_block_আকার);
             ir_prefetch += পরবর্তী_block_আকার;
        }
        ir += current_block_আকার;
    }

    dma_queue_flush(dma_q);
}

স্থির অন্তঃরেখ শূন্য hvx_div_scalar_f32(uint8_t * restrict dst, ধ্রুবক uint8_t * restrict src, ধ্রুবক দশমিক val, ধ্রুবক uint32_t অঙ্ক_elems) {
    hvx_mul_scalar_f32(dst, src, 1.0f / val, অঙ্ক_elems);
}

স্থির অন্তঃরেখ শূন্য hvx_div_scalar_f16(uint8_t * restrict dst, ধ্রুবক uint8_t * restrict src, ধ্রুবক _Float16 val, ধ্রুবক uint32_t অঙ্ক_elems) {
    hvx_div_scalar_f16_aa(dst, src, val, অঙ্ক_elems);
}

প্রকারলেখ শূন্য (*compute_দ্বিমিক_chunked_t)(
    uint8_t * restrict dst,
    ধ্রুবক uint8_t * restrict src0,
    ধ্রুবক uint8_t * restrict src1,
    ধ্রুবক uint32_t অঙ্ক_elems
);

প্রকারলেখ শূন্য (*compute_দ্বিমিক_scalar_chunked_f32_t)(
    uint8_t * restrict dst,
    ধ্রুবক uint8_t * restrict src0,
    ধ্রুবক দশমিক val,
    ধ্রুবক uint32_t অঙ্ক_elems
);

প্রকারলেখ শূন্য (*compute_দ্বিমিক_scalar_chunked_f16_t)(
    uint8_t * restrict dst,
    ধ্রুবক uint8_t * restrict src0,
    ধ্রুবক _Float16 val,
    ধ্রুবক uint32_t অঙ্ক_elems
);

কাঠামো দ্বিমিক_chunked_context {
    কাঠামো htp_ops_context * octx;
    কাঠামো htp_দ্বিমিক_vtcm_layout vtcm_layout;
    uint8_t *                vtcm_base;
    uint32_t                 elem_start;
    uint32_t                 nelem;
    uint32_t                 chunk_আকার;
    uint32_t                 chunks_per_thread;
    uint32_t                 যোগ_chunks;
    bool                     is_scalar;
    দশমিক                    scalar_f32;
    _Float16                 scalar_f16;
    compute_দ্বিমিক_chunked_t compute;
    compute_দ্বিমিক_scalar_chunked_f32_t compute_scalar_f32;
    compute_দ্বিমিক_scalar_chunked_f16_t compute_scalar_f16;
};

স্থির শূন্য দ্বিমিক_thread_chunked(unsigned পূর্ণসংখ্যা nth, unsigned পূর্ণসংখ্যা ith, শূন্য * তথ্য) {
    (শূন্য) nth;
    কাঠামো দ্বিমিক_chunked_context * ctx = (কাঠামো দ্বিমিক_chunked_context *) তথ্য;
    কাঠামো htp_ops_context * octx = ctx->octx;
    ধ্রুবক কাঠামো htp_tensor * src0 = octx->src[0];
    ধ্রুবক কাঠামো htp_tensor * src1 = octx->src[1];
    ধ্রুবক কাঠামো htp_tensor * dst  = octx->dst;

    ধ্রুবক uint32_t start_chunk = ctx->chunks_per_thread * ith;
    ধ্রুবক uint32_t end_chunk   = MIN(start_chunk + ctx->chunks_per_thread, ctx->যোগ_chunks);
    যদি (start_chunk >= end_chunk) {
        ফেরত;
    }

    ধ্রুবক uint32_t src0_type   = src0->type;
    ধ্রুবক size_t   elem_আকার   = (src0_type == HTP_TYPE_F32) ? আকার(দশমিক) : আকার(_Float16);
    ধ্রুবক uint32_t chunk_আকার  = ctx->chunk_আকার;
    ধ্রুবক size_t   chunk_bytes = ctx->vtcm_layout.src0_spad_half_আকার;

    FARF(উপরের_2, "binary-chunked: %d/%d (%u:%u) chunks %u elems %u",
         ith, nth, start_chunk, end_chunk, ctx->যোগ_chunks, ctx->nelem);

    ধ্রুবক কাঠামো htp_দ্বিমিক_vtcm_layout * layout = &ctx->vtcm_layout;
    uint8_t * src0_spad_base = VTCM_LAYOUT_PTR(uint8_t, ctx->vtcm_base, layout->off_src0) + (ith * layout->src0_bytes_per_thread);
    uint8_t * src1_spad_base = ctx->is_scalar ? ফাঁকা : (VTCM_LAYOUT_PTR(uint8_t, ctx->vtcm_base, layout->off_src1) + (ith * layout->src1_bytes_per_thread));
    uint8_t * dst_spad_base  = VTCM_LAYOUT_PTR(uint8_t, ctx->vtcm_base, layout->off_dst)  + (ith * layout->dst_bytes_per_thread);

    dma_queue * dma_q = octx->ctx->dma[ith];
    uint32_t prefetch_chunk = start_chunk;
    পূর্ণসংখ্যা spad_idx = 0;

    জন্য (পূর্ণসংখ্যা k = 0; k < 2 && prefetch_chunk < end_chunk; k++) {
        ধ্রুবক uint32_t c_start   = ctx->elem_start + prefetch_chunk * chunk_আকার;
        ধ্রুবক uint32_t c_end     = MIN(c_start + chunk_আকার, ctx->elem_start + ctx->nelem);
        ধ্রুবক uint32_t cur_elems = c_end - c_start;
        ধ্রুবক uint32_t cur_bytes = cur_elems * elem_আকার;

        dma_addr_t s0_curr = src0->তথ্য + (size_t) c_start * elem_আকার;
        dma_addr_t d_curr  = dst->তথ্য  + (size_t) c_start * elem_আকার;

        uint8_t * s0_spad = src0_spad_base + spad_idx * chunk_bytes;
        uint8_t * d_spad  = dst_spad_base  + spad_idx * chunk_bytes;

        dma_queue_ঠেলো(dma_q, dma_make_তথ্য(d_curr, d_spad), chunk_bytes, chunk_bytes, cur_bytes, 0);
        dma_queue_ঠেলো(dma_q, dma_make_তথ্য(s0_spad, s0_curr), chunk_bytes, chunk_bytes, cur_bytes, 1);
        যদি (!ctx->is_scalar) {
            dma_addr_t s1_curr = src1->তথ্য + (size_t) c_start * elem_আকার;
            uint8_t * s1_spad = src1_spad_base + spad_idx * chunk_bytes;
            dma_queue_ঠেলো(dma_q, dma_make_তথ্য(s1_spad, s1_curr), chunk_bytes, chunk_bytes, cur_bytes, 1);
        }

        prefetch_chunk++;
        spad_idx ^= 1;
    }

    কাঠামো htp_thread_trace * tr = &octx->ctx->trace[ith];

    জন্য (uint32_t c = start_chunk; c < end_chunk; c++) {
        ধ্রুবক uint32_t c_start   = ctx->elem_start + c * chunk_আকার;
        ধ্রুবক uint32_t c_end     = MIN(c_start + chunk_আকার, ctx->elem_start + ctx->nelem);
        ধ্রুবক uint32_t cur_elems = c_end - c_start;
        ধ্রুবক uint32_t cur_bytes = cur_elems * elem_আকার;

        uint8_t * d_spad  = (uint8_t *) dma_queue_বের_করো(dma_q).src;
        uint8_t * s0_spad = (uint8_t *) dma_queue_বের_করো(dma_q).dst;
        uint8_t * s1_spad = ctx->is_scalar ? ফাঁকা : (uint8_t *) dma_queue_বের_করো(dma_q).dst;

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) c);
        যদি (ctx->is_scalar) {
            যদি (src0_type == HTP_TYPE_F32) {
                ctx->compute_scalar_f32(d_spad, s0_spad, ctx->scalar_f32, cur_elems);
            } অন্যথা {
                ctx->compute_scalar_f16(d_spad, s0_spad, ctx->scalar_f16, cur_elems);
            }
        } অন্যথা {
            ctx->compute(d_spad, s0_spad, s1_spad, cur_elems);
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) c);

        dma_addr_t dst_curr = dst->তথ্য + (size_t) c_start * elem_আকার;
        dma_queue_ঠেলো(dma_q, dma_make_তথ্য(dst_curr, d_spad), chunk_bytes, chunk_bytes, cur_bytes, 1);

        যদি (prefetch_chunk < end_chunk) {
            ধ্রুবক uint32_t pc_start  = ctx->elem_start + prefetch_chunk * chunk_আকার;
            ধ্রুবক uint32_t pc_end    = MIN(pc_start + chunk_আকার, ctx->elem_start + ctx->nelem);
            ধ্রুবক uint32_t p_elems   = pc_end - pc_start;
            ধ্রুবক uint32_t p_bytes   = p_elems * elem_আকার;

            dma_addr_t s0_পরবর্তী = src0->তথ্য + (size_t) pc_start * elem_আকার;
            dma_queue_ঠেলো(dma_q, dma_make_তথ্য(s0_spad, s0_পরবর্তী), chunk_bytes, chunk_bytes, p_bytes, 1);
            যদি (!ctx->is_scalar) {
                dma_addr_t s1_পরবর্তী = src1->তথ্য + (size_t) pc_start * elem_আকার;
                dma_queue_ঠেলো(dma_q, dma_make_তথ্য(s1_spad, s1_পরবর্তী), chunk_bytes, chunk_bytes, p_bytes, 1);
            }

            prefetch_chunk++;
        }
    }

    dma_queue_flush(dma_q);
}

স্থির পূর্ণসংখ্যা execute_op_দ্বিমিক(কাঠামো htp_ops_context * octx) {
    ধ্রুবক কাঠামো htp_tensor * src0 = octx->src[0];
    ধ্রুবক কাঠামো htp_tensor * src1 = octx->src[1];
    ধ্রুবক কাঠামো htp_tensor * dst  = octx->dst;
    ধ্রুবক কাঠামো htp_দ্বিমিক_kernel_params * kparams = (ধ্রুবক কাঠামো htp_দ্বিমিক_kernel_params *) octx->kernel_params;

    ধ্রুবক uint32_t src0_nrows = src0->ne[1] * src0->ne[2] * src0->ne[3];

    // Use packed row sizes for VTCM allocation and alignment
    ধ্রুবক uint32_t src0_type = octx->src[0]->type;
    ধ্রুবক size_t elem_আকার = (src0_type == HTP_TYPE_F32) ? আকার(দশমিক) : আকার(_Float16);
    ধ্রুবক size_t src0_row_আকার = src0->ne[0] * elem_আকার;
    ধ্রুবক size_t src1_row_আকার = src1->ne[0] * elem_আকার;
    ধ্রুবক size_t dst_row_আকার  = dst->ne[0]  * elem_আকার;

    যদি (kparams->kernel_type == HTP_দ্বিমিক_KERNEL_CHUNKED) {
        ধ্রুবক bool is_scalar = kparams->is_scalar || (src1->ne[0] == 1 && src1->ne[1] == 1 && src1->ne[2] == 1 && src1->ne[3] == 1);

        compute_দ্বিমিক_chunked_t compute = ফাঁকা;
        compute_দ্বিমিক_scalar_chunked_f32_t compute_scalar_f32 = ফাঁকা;
        compute_দ্বিমিক_scalar_chunked_f16_t compute_scalar_f16 = ফাঁকা;

        যদি (is_scalar) {
            যদি (src0_type == HTP_TYPE_F32) {
                নির্বাচন (octx->op) {
                    ক্ষেত্র HTP_OP_ADD: compute_scalar_f32 = hvx_add_scalar_f32; বিরতি;
                    ক্ষেত্র HTP_OP_SUB: compute_scalar_f32 = hvx_sub_scalar_f32; বিরতি;
                    ক্ষেত্র HTP_OP_MUL: compute_scalar_f32 = hvx_mul_scalar_f32; বিরতি;
                    ক্ষেত্র HTP_OP_DIV: compute_scalar_f32 = hvx_div_scalar_f32; বিরতি;
                    পূর্বনির্ধারিত: বিরতি;
                }
            } অন্যথা যদি (src0_type == HTP_TYPE_F16) {
                নির্বাচন (octx->op) {
                    ক্ষেত্র HTP_OP_ADD: compute_scalar_f16 = hvx_add_scalar_f16; বিরতি;
                    ক্ষেত্র HTP_OP_SUB: compute_scalar_f16 = hvx_sub_scalar_f16; বিরতি;
                    ক্ষেত্র HTP_OP_MUL: compute_scalar_f16 = hvx_mul_scalar_f16; বিরতি;
                    ক্ষেত্র HTP_OP_DIV: compute_scalar_f16 = hvx_div_scalar_f16; বিরতি;
                    পূর্বনির্ধারিত: বিরতি;
                }
            }
            যদি (!compute_scalar_f32 && !compute_scalar_f16) {
                ফেরত HTP_STATUS_NO_SUPPORT;
            }
        } অন্যথা {
            যদি (src0_type == HTP_TYPE_F32) {
                নির্বাচন (octx->op) {
                    ক্ষেত্র HTP_OP_ADD: compute = hvx_add_f32; বিরতি;
                    ক্ষেত্র HTP_OP_SUB: compute = hvx_sub_f32; বিরতি;
                    ক্ষেত্র HTP_OP_MUL: compute = hvx_mul_f32; বিরতি;
                    ক্ষেত্র HTP_OP_DIV: compute = hvx_div_f32; বিরতি;
                    পূর্বনির্ধারিত: বিরতি;
                }
            } অন্যথা যদি (src0_type == HTP_TYPE_F16) {
                নির্বাচন (octx->op) {
                    ক্ষেত্র HTP_OP_ADD: compute = hvx_add_f16; বিরতি;
                    ক্ষেত্র HTP_OP_SUB: compute = hvx_sub_f16; বিরতি;
                    ক্ষেত্র HTP_OP_MUL: compute = hvx_mul_f16; বিরতি;
                    ক্ষেত্র HTP_OP_DIV: compute = hvx_div_f16; বিরতি;
                    পূর্বনির্ধারিত: বিরতি;
                }
            }
            যদি (!compute) {
                ফেরত HTP_STATUS_NO_SUPPORT;
            }
        }

        ধ্রুবক uint32_t যোগ_elems = (uint32_t) (src0->ne[0] * src0->ne[1] * src0->ne[2] * src0->ne[3]);
        যদি (যোগ_elems == 0) {
            ফেরত HTP_STATUS_OK;
        }

        uint32_t elem_start = 0;
        uint32_t nelem      = যোগ_elems;
        ধ্রুবক uint32_t elems_per_সারি = (src0_type == HTP_TYPE_F32) ? 32 : 64;

        যদি (octx->ctx->mdev.গুনতি > 1) {
            ধ্রুবক bool can_split = htp_tensor_mdev_তথ্য_aligned(dst) &&
                                   htp_tensor_mdev_তথ্য_aligned(src0) &&
                                   (is_scalar || htp_tensor_mdev_তথ্য_aligned(src1)) &&
                                   htp_tensor_is_contiguous(dst, elem_আকার) &&
                                   htp_tensor_is_contiguous(src0, elem_আকার) &&
                                   (is_scalar || htp_tensor_is_contiguous(src1, elem_আকার));
            ধ্রুবক কাঠামো htp_tensor_mdev_range range = htp_tensor_mdev_partition(
                যোগ_elems, can_split ? elems_per_সারি : 0,
                octx->ctx->mdev.idx, octx->ctx->mdev.গুনতি, &octx->ctx->mdev.গুনতি_div);
            elem_start = range.start;
            nelem      = range.গুনতি;
        }

        যদি (nelem == 0) {
            ফেরত HTP_STATUS_OK;
        }

        যদি (!htp_ops_context_set_n_threads(octx, kparams->n_threads)) {
            ফেরত HTP_STATUS_INVAL_PARAMS;
        }

        কাঠামো htp_দ্বিমিক_vtcm_layout vtcm_layout;
        htp_দ্বিমিক_vtcm_layout_build(&vtcm_layout, kparams, octx->ctx->vtcm_আকার);
        যদি (vtcm_layout.যোগ_bytes == 0 || vtcm_layout.যোগ_bytes > octx->ctx->vtcm_আকার) {
            ফেরত HTP_STATUS_VTCM_TOO_SMALL;
        }

        ধ্রুবক uint32_t chunk_আকার = kparams->chunk_আকার > 0 ? kparams->chunk_আকার : (32768 / elem_আকার);
        ধ্রুবক uint32_t যোগ_chunks = (nelem + chunk_আকার - 1) / chunk_আকার;
        ধ্রুবক uint32_t n_threads = (যোগ_chunks >= 2) ? MIN(octx->n_threads, যোগ_chunks) : 1;
        ধ্রুবক uint32_t chunks_per_thread = (যোগ_chunks + n_threads - 1) / n_threads;

        দশমিক scalar_f32 = 0.0f;
        _Float16 scalar_f16 = 0;
        যদি (is_scalar) {
            uint8_t * vtcm_src1 = VTCM_LAYOUT_PTR(uint8_t, octx->ctx->vtcm_base, vtcm_layout.off_src1);
            dma_queue * dma_q = octx->ctx->dma[0];
            dma_queue_ঠেলো(dma_q, dma_make_তথ্য(vtcm_src1, src1->তথ্য), 128, 0, elem_আকার, 1);
            dma_queue_বের_করো(dma_q);

            যদি (src0_type == HTP_TYPE_F32) {
                scalar_f32 = ((ধ্রুবক দশমিক *) vtcm_src1)[0];
            } অন্যথা {
                scalar_f16 = ((ধ্রুবক _Float16 *) vtcm_src1)[0];
            }
        }

        কাঠামো দ্বিমিক_chunked_context cctx = {
            .octx                = octx,
            .vtcm_layout         = vtcm_layout,
            .vtcm_base           = (uint8_t *) octx->ctx->vtcm_base,
            .elem_start          = elem_start,
            .nelem               = nelem,
            .chunk_আকার          = chunk_আকার,
            .chunks_per_thread   = chunks_per_thread,
            .যোগ_chunks        = যোগ_chunks,
            .is_scalar           = is_scalar,
            .scalar_f32          = scalar_f32,
            .scalar_f16          = scalar_f16,
            .compute             = compute,
            .compute_scalar_f32  = compute_scalar_f32,
            .compute_scalar_f16  = compute_scalar_f16,
        };

        work_queue_run(octx->ctx->work_queue, দ্বিমিক_thread_chunked, &cctx, n_threads);
        ফেরত HTP_STATUS_OK;
    }

    uint32_t row_start = 0;
    uint32_t nrows     = src0_nrows;

    যদি (octx->ctx->mdev.গুনতি > 1) {
        uint32_t rows_per_chunk = 0;
        htp_tensor_mdev_rows_per_chunk(dst, (uint32_t) elem_আকার, (uint32_t) dst_row_আকার, &rows_per_chunk);
        ধ্রুবক কাঠামো htp_tensor_mdev_range range = htp_tensor_mdev_partition(src0_nrows, rows_per_chunk,
                                                       octx->ctx->mdev.idx, octx->ctx->mdev.গুনতি, &octx->ctx->mdev.গুনতি_div);
        row_start = range.start;
        nrows     = range.গুনতি;
    }

    যদি (nrows == 0) {
        ফেরত HTP_STATUS_OK;
    }

    যদি (!htp_ops_context_set_n_threads(octx, kparams->n_threads)) {
        ফেরত HTP_STATUS_INVAL_PARAMS;
    }

    ধ্রুবক uint32_t n_threads = octx->n_threads;
    ধ্রুবক size_t src0_row_আকার_aligned = kparams->src0_row_আকার_aligned;
    ধ্রুবক size_t src1_row_আকার_aligned = kparams->src1_row_আকার_aligned;
    ধ্রুবক size_t dst_row_আকার_aligned  = kparams->dst_row_আকার_aligned;

    কাঠামো htp_দ্বিমিক_context bctx;
    bctx.vtcm_base = (uint8_t *) octx->ctx->vtcm_base;
    htp_দ্বিমিক_vtcm_layout_build(&bctx.vtcm_layout, kparams, octx->ctx->vtcm_আকার);

    যদি (bctx.vtcm_layout.rows_per_বাফার == 0 || bctx.vtcm_layout.যোগ_bytes > octx->ctx->vtcm_আকার) {
        ফেরত HTP_STATUS_VTCM_TOO_SMALL;
    }

    dma_queue * dma_q = octx->ctx->dma[0];
    uint8_t * vtcm_src1 = VTCM_LAYOUT_PTR(uint8_t, bctx.vtcm_base, bctx.vtcm_layout.off_src1);
    যদি (kparams->kernel_type == HTP_দ্বিমিক_KERNEL_ROW_BCAST) {
        dma_queue_ঠেলো(dma_q, dma_make_তথ্য(vtcm_src1, src1->তথ্য), bctx.vtcm_layout.src1_আকার, 0, src1->ne[0] * elem_আকার, 1);
    } অন্যথা যদি (kparams->kernel_type == HTP_দ্বিমিক_KERNEL_SCALAR ||
               kparams->kernel_type == HTP_দ্বিমিক_KERNEL_REPEAT ||
               kparams->kernel_type == HTP_দ্বিমিক_KERNEL_COMPLEX) {
        যদি (!kparams->src1_strided) {
            ধ্রুবক size_t src1_bytes = (size_t) src1->ne[0] * src1->ne[1] * src1->ne[2] * src1->ne[3] * elem_আকার;
            dma_queue_ঠেলো(dma_q, dma_make_তথ্য(vtcm_src1, src1->তথ্য), bctx.vtcm_layout.src1_আকার, 0, src1_bytes, 1);
        } অন্যথা {
            ধ্রুবক uint32_t ne11 = src1->ne[1];
            ধ্রুবক uint32_t ne12 = src1->ne[2];
            ধ্রুবক uint32_t ne13 = src1->ne[3];
            ধ্রুবক size_t row_bytes = src1->ne[0] * elem_আকার;
            জন্য (uint32_t i13 = 0; i13 < ne13; i13++) {
                জন্য (uint32_t i12 = 0; i12 < ne12; i12++) {
                    dma_addr_t src_addr = src1->তথ্য + i13 * src1->nb[3] + i12 * src1->nb[2];
                    uint8_t * dst_ptr = vtcm_src1 + (i13 * ne12 * ne11 + i12 * ne11) * row_bytes;
                    যখন (!dma_queue_ঠেলো(dma_q, dma_make_তথ্য(dst_ptr, src_addr),
                                           row_bytes, src1->nb[1], row_bytes, ne11)) {
                        dma_queue_বের_করো(dma_q);
                    }
                }
            }
        }
    } অন্যথা যদি (kparams->kernel_type == HTP_দ্বিমিক_KERNEL_ADD_ID) {
        ধ্রুবক কাঠামো htp_tensor * src2 = octx->src[2];
        uint8_t * vtcm_src2 = VTCM_LAYOUT_PTR(uint8_t, bctx.vtcm_base, bctx.vtcm_layout.off_src2);
        dma_queue_ঠেলো(dma_q, dma_make_তথ্য(vtcm_src1, src1->তথ্য),
                       kparams->src1_row_আকার_aligned, src1->nb[1],
                       src1->ne[0] * elem_আকার, src1->ne[1]);
        ধ্রুবক size_t src2_row_bytes = src2->ne[0] * আকার(int32_t);
        dma_queue_ঠেলো(dma_q, dma_make_তথ্য(vtcm_src2, src2->তথ্য),
                       src2_row_bytes, src2->nb[1],
                       src2_row_bytes, src2->ne[1]);
    }

    bctx.octx                  = octx;
    bctx.nrows_per_thread      = fastdiv(nrows + n_threads - 1, &octx->n_threads_div);
    bctx.যোগ_rows            = nrows;
    bctx.row_start             = row_start;
    bctx.block_max             = bctx.vtcm_layout.rows_per_বাফার;
    bctx.src0_row_আকার_aligned = src0_row_আকার_aligned;
    bctx.src1_row_আকার_aligned = src1_row_আকার_aligned;
    bctx.dst_row_আকার_aligned  = dst_row_আকার_aligned;
    bctx.row_আকার_bytes        = src0_row_আকার;

    bctx.src0_dim1_div  = init_fastdiv_মানসমূহ(src0->ne[1]);
    bctx.src0_dim2_div  = init_fastdiv_মানসমূহ(src0->ne[2]);
    bctx.src0_dim12_div = init_fastdiv_মানসমূহ(src0->ne[1] * src0->ne[2]);

    bctx.src1_dim1_div  = init_fastdiv_মানসমূহ(src1->ne[1]);
    bctx.src1_dim2_div  = init_fastdiv_মানসমূহ(src1->ne[2]);
    bctx.src1_dim3_div  = init_fastdiv_মানসমূহ(src1->ne[3]);

    bctx.split_at_ne01  = kparams->split_at_ne01;
    bctx.split_at_ne02  = kparams->split_at_ne02;

    worker_callback_t worker_func = ফাঁকা;
    শূন্য * compute_func = ফাঁকা;

    নির্বাচন (kparams->kernel_type) {
        ক্ষেত্র HTP_দ্বিমিক_KERNEL_SAME_SHAPE:
            worker_func = দ্বিমিক_thread_vector_same_shape;
            যদি (src0_type == HTP_TYPE_F32) {
                নির্বাচন (octx->op) {
                    ক্ষেত্র HTP_OP_ADD: compute_func = compute_same_shape_add_f32; বিরতি;
                    ক্ষেত্র HTP_OP_SUB: compute_func = compute_same_shape_sub_f32; বিরতি;
                    ক্ষেত্র HTP_OP_MUL: compute_func = compute_same_shape_mul_f32; বিরতি;
                    ক্ষেত্র HTP_OP_DIV: compute_func = compute_same_shape_div_f32; বিরতি;
                    পূর্বনির্ধারিত: বিরতি;
                }
            } অন্যথা যদি (src0_type == HTP_TYPE_F16) {
                নির্বাচন (octx->op) {
                    ক্ষেত্র HTP_OP_ADD: compute_func = compute_same_shape_add_f16; বিরতি;
                    ক্ষেত্র HTP_OP_SUB: compute_func = compute_same_shape_sub_f16; বিরতি;
                    ক্ষেত্র HTP_OP_MUL: compute_func = compute_same_shape_mul_f16; বিরতি;
                    ক্ষেত্র HTP_OP_DIV: compute_func = compute_same_shape_div_f16; বিরতি;
                    পূর্বনির্ধারিত: বিরতি;
                }
            }
            বিরতি;
        ক্ষেত্র HTP_দ্বিমিক_KERNEL_ROW_BCAST:
            worker_func = দ্বিমিক_thread_vector_row_broadcast;
            যদি (src0_type == HTP_TYPE_F32) {
                নির্বাচন (octx->op) {
                    ক্ষেত্র HTP_OP_ADD: compute_func = compute_row_bcast_add_f32; বিরতি;
                    ক্ষেত্র HTP_OP_SUB: compute_func = compute_row_bcast_sub_f32; বিরতি;
                    ক্ষেত্র HTP_OP_MUL: compute_func = compute_row_bcast_mul_f32; বিরতি;
                    ক্ষেত্র HTP_OP_DIV: compute_func = compute_row_bcast_div_f32; বিরতি;
                    পূর্বনির্ধারিত: বিরতি;
                }
            } অন্যথা যদি (src0_type == HTP_TYPE_F16) {
                নির্বাচন (octx->op) {
                    ক্ষেত্র HTP_OP_ADD: compute_func = compute_row_bcast_add_f16; বিরতি;
                    ক্ষেত্র HTP_OP_SUB: compute_func = compute_row_bcast_sub_f16; বিরতি;
                    ক্ষেত্র HTP_OP_MUL: compute_func = compute_row_bcast_mul_f16; বিরতি;
                    ক্ষেত্র HTP_OP_DIV: compute_func = compute_row_bcast_div_f16; বিরতি;
                    পূর্বনির্ধারিত: বিরতি;
                }
            }
            বিরতি;
        ক্ষেত্র HTP_দ্বিমিক_KERNEL_SCALAR:
            worker_func = দ্বিমিক_thread_scalar;
            যদি (src0_type == HTP_TYPE_F32) {
                নির্বাচন (octx->op) {
                    ক্ষেত্র HTP_OP_ADD: compute_func = compute_scalar_add_f32; বিরতি;
                    ক্ষেত্র HTP_OP_SUB: compute_func = compute_scalar_sub_f32; বিরতি;
                    ক্ষেত্র HTP_OP_MUL: compute_func = compute_scalar_mul_f32; বিরতি;
                    ক্ষেত্র HTP_OP_DIV: compute_func = compute_scalar_div_f32; বিরতি;
                    পূর্বনির্ধারিত: বিরতি;
                }
            } অন্যথা যদি (src0_type == HTP_TYPE_F16) {
                নির্বাচন (octx->op) {
                    ক্ষেত্র HTP_OP_ADD: compute_func = compute_scalar_add_f16; বিরতি;
                    ক্ষেত্র HTP_OP_SUB: compute_func = compute_scalar_sub_f16; বিরতি;
                    ক্ষেত্র HTP_OP_MUL: compute_func = compute_scalar_mul_f16; বিরতি;
                    ক্ষেত্র HTP_OP_DIV: compute_func = compute_scalar_div_f16; বিরতি;
                    পূর্বনির্ধারিত: বিরতি;
                }
            }
            বিরতি;
        ক্ষেত্র HTP_দ্বিমিক_KERNEL_COMPLEX:
            worker_func = দ্বিমিক_thread_vector_complex;
            যদি (src0_type == HTP_TYPE_F32) {
                নির্বাচন (octx->op) {
                    ক্ষেত্র HTP_OP_ADD: compute_func = compute_complex_add_f32; বিরতি;
                    ক্ষেত্র HTP_OP_SUB: compute_func = compute_complex_sub_f32; বিরতি;
                    ক্ষেত্র HTP_OP_MUL: compute_func = compute_complex_mul_f32; বিরতি;
                    ক্ষেত্র HTP_OP_DIV: compute_func = compute_complex_div_f32; বিরতি;
                    পূর্বনির্ধারিত: বিরতি;
                }
            } অন্যথা যদি (src0_type == HTP_TYPE_F16) {
                নির্বাচন (octx->op) {
                    ক্ষেত্র HTP_OP_ADD: compute_func = compute_complex_add_f16; বিরতি;
                    ক্ষেত্র HTP_OP_SUB: compute_func = compute_complex_sub_f16; বিরতি;
                    ক্ষেত্র HTP_OP_MUL: compute_func = compute_complex_mul_f16; বিরতি;
                    ক্ষেত্র HTP_OP_DIV: compute_func = compute_complex_div_f16; বিরতি;
                    পূর্বনির্ধারিত: বিরতি;
                }
            }
            বিরতি;
        ক্ষেত্র HTP_দ্বিমিক_KERNEL_REPEAT:
            worker_func = দ্বিমিক_thread_element_repeat;
            যদি (src0_type == HTP_TYPE_F32) {
                নির্বাচন (octx->op) {
                    ক্ষেত্র HTP_OP_ADD: compute_func = compute_repeat_add_f32; বিরতি;
                    ক্ষেত্র HTP_OP_SUB: compute_func = compute_repeat_sub_f32; বিরতি;
                    ক্ষেত্র HTP_OP_MUL: compute_func = compute_repeat_mul_f32; বিরতি;
                    ক্ষেত্র HTP_OP_DIV: compute_func = compute_repeat_div_f32; বিরতি;
                    পূর্বনির্ধারিত: বিরতি;
                }
            } অন্যথা যদি (src0_type == HTP_TYPE_F16) {
                নির্বাচন (octx->op) {
                    ক্ষেত্র HTP_OP_ADD: compute_func = compute_repeat_add_f16; বিরতি;
                    ক্ষেত্র HTP_OP_SUB: compute_func = compute_repeat_sub_f16; বিরতি;
                    ক্ষেত্র HTP_OP_MUL: compute_func = compute_repeat_mul_f16; বিরতি;
                    ক্ষেত্র HTP_OP_DIV: compute_func = compute_repeat_div_f16; বিরতি;
                    পূর্বনির্ধারিত: বিরতি;
                }
            }
            বিরতি;
        ক্ষেত্র HTP_দ্বিমিক_KERNEL_ADD_ID:
            যদি (octx->op == HTP_OP_ADD_ID && src0_type == HTP_TYPE_F32) {
                worker_func = দ্বিমিক_thread_add_id_f32;
                compute_func = (শূন্য *) compute_add_id_f32;
            }
            বিরতি;
        পূর্বনির্ধারিত: বিরতি;
    }

    যদি (!worker_func || !compute_func) {
        ফেরত HTP_STATUS_NO_SUPPORT;
    }

    bctx.compute = compute_func;

    যদি (kparams->kernel_type == HTP_দ্বিমিক_KERNEL_ROW_BCAST ||
        kparams->kernel_type == HTP_দ্বিমিক_KERNEL_SCALAR ||
        kparams->kernel_type == HTP_দ্বিমিক_KERNEL_REPEAT ||
        kparams->kernel_type == HTP_দ্বিমিক_KERNEL_COMPLEX ||
        kparams->kernel_type == HTP_দ্বিমিক_KERNEL_ADD_ID) {
        dma_queue_flush(dma_q);
    }

    work_queue_run(octx->ctx->work_queue, worker_func, &bctx, n_threads);

    ফেরত HTP_STATUS_OK;
}

পূর্ণসংখ্যা op_দ্বিমিক(কাঠামো htp_ops_context * octx) {
    ধ্রুবক uint32_t src0_type = octx->src[0]->type;
    যদি ((src0_type == HTP_TYPE_F32) || (src0_type == HTP_TYPE_F16)) {
        ফেরত execute_op_দ্বিমিক(octx);
    }

    ফেরত HTP_STATUS_NO_SUPPORT;
}
