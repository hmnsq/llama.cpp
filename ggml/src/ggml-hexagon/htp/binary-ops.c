#pragma clang diagnostic ignored "-Wunused-variable"
#pragma clang diagnostic ignored "-Wunused-function"
#pragma clang diagnostic ignored "-Wunused-but-set-variable"

#அந்தர்பாவ <HAP_farf.h>
#அந்தர்பாவ <HAP_perf.h>

#அந்தர்பாவ <math.h>
#அந்தர்பாவ <string.h>

#அந்தர்பாவ "dma-queue.h"
#அந்தர்பாவ "hvx-utils.h"

#define GGML_COMMON_DECL_C
#அந்தர்பாவ "ggml-common.h"
#அந்தர்பாவ "hex-common.h"
#அந்தர்பாவ "hex-profile.h"
#அந்தர்பாவ "binary-ops.h"
#அந்தர்பாவ "htp-ctx.h"
#அந்தர்பாவ "htp-ops.h"
#அந்தர்பாவ "htp-tensor.h"

#ifndef MIN
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#endif

// Context for binary operations
கட்டமைப்பு htp_இருமம்_context {
    கட்டமைப்பு htp_ops_context * octx;
    கட்டமைப்பு htp_இருமம்_vtcm_layout vtcm_layout;
    uint8_t * vtcm_base;

    கட்டமைப்பு fastdiv_மதிப்புகள் src0_dim1_div; // ne01
    கட்டமைப்பு fastdiv_மதிப்புகள் src0_dim2_div; // ne02
    கட்டமைப்பு fastdiv_மதிப்புகள் src0_dim12_div;// ne03

    கட்டமைப்பு fastdiv_மதிப்புகள் src1_dim1_div; // ne11
    கட்டமைப்பு fastdiv_மதிப்புகள் src1_dim2_div; // ne12
    கட்டமைப்பு fastdiv_மதிப்புகள் src1_dim3_div; // ne13

    uint32_t block_max;
    uint32_t nrows_per_thread;
    uint32_t total_rows;
    uint32_t row_start;
    size_t   src0_row_size_aligned;
    size_t   src1_row_size_aligned;
    size_t   dst_row_size_aligned;
    size_t   row_size_bytes;

    bool split_at_ne01;
    bool split_at_ne02;

    வெறுமை * compute;
};

#define htp_இருமம்_preamble                        \
    மாறிலி கட்டமைப்பு htp_tensor * src0 = octx->src[0]; \
    மாறிலி கட்டமைப்பு htp_tensor * src1 = octx->src[1]; \
    மாறிலி கட்டமைப்பு htp_tensor * dst  = octx->dst;    \
                                                   \
    மாறிலி uint32_t ne00 = src0->ne[0];             \
    மாறிலி uint32_t ne01 = src0->ne[1];             \
    மாறிலி uint32_t ne02 = src0->ne[2];             \
    மாறிலி uint32_t ne03 = src0->ne[3];             \
                                                   \
    மாறிலி uint32_t ne10 = src1->ne[0];             \
    மாறிலி uint32_t ne11 = src1->ne[1];             \
    மாறிலி uint32_t ne12 = src1->ne[2];             \
    மாறிலி uint32_t ne13 = src1->ne[3];             \
                                                   \
    மாறிலி uint32_t nb01 = src0->nb[1];             \
    மாறிலி uint32_t nb02 = src0->nb[2];             \
    மாறிலி uint32_t nb03 = src0->nb[3];             \
                                                   \
    மாறிலி uint32_t nb11 = src1->nb[1];             \
    மாறிலி uint32_t nb12 = src1->nb[2];             \
    மாறிலி uint32_t nb13 = src1->nb[3];             \
                                                   \
    மாறிலி uint32_t nb1 = dst->nb[1];               \
    மாறிலி uint32_t nb2 = dst->nb[2];               \
    மாறிலி uint32_t nb3 = dst->nb[3];

நிலையான உள்வரி uint32_t calc_block_size(கட்டமைப்பு htp_இருமம்_context * bctx, uint32_t ir, uint32_t end_row, uint32_t ne01, uint32_t ne02) {
    uint32_t i03, i02, i01, rem;
    i03 = fastdiv(ir, &bctx->src0_dim12_div);
    rem = ir - i03 * (ne02 * ne01);
    i02 = fastdiv(rem, &bctx->src0_dim1_div);
    i01 = rem - i02 * ne01;

    uint32_t rows_இடது = end_row - ir;
    uint32_t block_limit = rows_இடது;

    எனில் (bctx->split_at_ne01) {
        block_limit = MIN(block_limit, ne01 - i01);
    }
    எனில் (bctx->split_at_ne02) {
         uint32_t rows_in_plane = (ne02 * ne01) - rem;
         block_limit = MIN(block_limit, rows_in_plane);
    }

    திரும்பு MIN(bctx->block_max, block_limit);
}

// Out-of-line compute micro-kernels

வகைவரை வெறுமை (*compute_scalar_t)(
    uint8_t * dst, மாறிலி uint8_t * src0, மாறிலி வெறுமை * s1_table,
    uint32_t cur_i11, uint32_t ne11, uint32_t n_rows,
    size_t dst_stride, size_t src0_stride, uint32_t ne00);

#define DEFINE_COMPUTE_SCALAR(NAME, TYPE, HVX_STMT)             \
நிலையான வெறுமை compute_scalar_##NAME(                              \
    uint8_t * dst, மாறிலி uint8_t * src0, மாறிலி வெறுமை * s1_table, \
    uint32_t cur_i11, uint32_t ne11, uint32_t n_rows,           \
    size_t dst_stride, size_t src0_stride, uint32_t ne00) {     \
    மாறிலி TYPE * table = (மாறிலி TYPE *) s1_table;               \
    க்கு (uint32_t r = 0; r < n_rows; r++) {                     \
        uint8_t * r_dst = dst + r * dst_stride;                 \
        மாறிலி uint8_t * r_src0 = src0 + r * src0_stride;        \
        TYPE val = table[cur_i11];                              \
        HVX_STMT;                                               \
        எனில் (ne11 > 1 && ++cur_i11 == ne11) {                    \
            cur_i11 = 0;                                        \
        }                                                       \
    }                                                           \
}

DEFINE_COMPUTE_SCALAR(add_f32, தசம,    hvx_add_scalar_f32_aa(r_dst, r_src0, val, ne00))
DEFINE_COMPUTE_SCALAR(add_f16, _Float16, hvx_add_scalar_f16_aa(r_dst, r_src0, val, ne00))
DEFINE_COMPUTE_SCALAR(sub_f32, தசம,    hvx_sub_scalar_f32_aa(r_dst, r_src0, val, ne00))
DEFINE_COMPUTE_SCALAR(sub_f16, _Float16, hvx_sub_scalar_f16_aa(r_dst, r_src0, val, ne00))
DEFINE_COMPUTE_SCALAR(mul_f32, தசம,    hvx_mul_scalar_f32_aa(r_dst, r_src0, val, ne00))
DEFINE_COMPUTE_SCALAR(mul_f16, _Float16, hvx_mul_scalar_f16_aa(r_dst, r_src0, val, ne00))
DEFINE_COMPUTE_SCALAR(div_f32, தசம,    hvx_mul_scalar_f32_aa(r_dst, r_src0, 1.0f / (val), ne00))
DEFINE_COMPUTE_SCALAR(div_f16, _Float16, hvx_div_scalar_f16_aa(r_dst, r_src0, val, ne00))

வகைவரை வெறுமை (*compute_same_shape_t)(
    uint8_t * dst, மாறிலி uint8_t * src0, மாறிலி uint8_t * src1,
    uint32_t n_rows, size_t dst_stride, size_t src0_stride, size_t src1_stride, uint32_t ne00);

#define DEFINE_COMPUTE_SAME_SHAPE(NAME, HVX_FN)                                                  \
நிலையான வெறுமை compute_same_shape_##NAME(                                                           \
    uint8_t * dst, மாறிலி uint8_t * src0, மாறிலி uint8_t * src1,                                   \
    uint32_t n_rows, size_t dst_stride, size_t src0_stride, size_t src1_stride, uint32_t ne00) { \
    க்கு (uint32_t r = 0; r < n_rows; r++) {                                                      \
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

வகைவரை வெறுமை (*compute_row_bcast_t)(
    uint8_t * dst, மாறிலி uint8_t * src0, மாறிலி uint8_t * src1,
    uint32_t n_rows, size_t dst_stride, size_t src0_stride, uint32_t ne00);

#define DEFINE_COMPUTE_ROW_BCAST(NAME, HVX_FN)                                               \
நிலையான வெறுமை compute_row_bcast_##NAME(                                                        \
    uint8_t * dst, மாறிலி uint8_t * src0, மாறிலி uint8_t * src1,                               \
    uint32_t n_rows, size_t dst_stride, size_t src0_stride, uint32_t ne00) {                 \
    க்கு (uint32_t r = 0; r < n_rows; r++) {                                                  \
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

வகைவரை வெறுமை (*compute_complex_t)(
    uint8_t * dst, மாறிலி uint8_t * src0, மாறிலி uint8_t * src1_plane,
    uint32_t i01, uint32_t ne11, மாறிலி கட்டமைப்பு fastdiv_மதிப்புகள் * div11, uint32_t src1_stride,
    uint32_t n_rows, size_t dst_stride, size_t src0_stride, uint32_t ne00);

#define DEFINE_COMPUTE_COMPLEX(NAME, HVX_FN)                                                 \
நிலையான வெறுமை compute_complex_##NAME(                                                          \
    uint8_t * dst, மாறிலி uint8_t * src0, மாறிலி uint8_t * src1_plane,                         \
    uint32_t i01, uint32_t ne11, மாறிலி கட்டமைப்பு fastdiv_மதிப்புகள் * div11, uint32_t src1_stride,  \
    uint32_t n_rows, size_t dst_stride, size_t src0_stride, uint32_t ne00) {                 \
    க்கு (uint32_t r = 0; r < n_rows; r++) {                                                  \
        uint32_t i11 = fastmodulo(i01 + r, ne11, div11);                                     \
        மாறிலி uint8_t * r_src1 = src1_plane + i11 * src1_stride;                             \
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

வகைவரை வெறுமை (*compute_repeat_t)(
    uint8_t * dst, மாறிலி uint8_t * src0, மாறிலி uint8_t * src1_plane,
    uint32_t i01, uint32_t ne11, மாறிலி கட்டமைப்பு fastdiv_மதிப்புகள் * div11, uint32_t src1_stride,
    uint32_t n_rows, size_t dst_stride, size_t src0_stride, uint32_t ne00, uint32_t ne10);

#define DEFINE_COMPUTE_REPEAT(NAME, TYPE, HVX_FN)                                            \
நிலையான வெறுமை compute_repeat_##NAME(                                                           \
    uint8_t * dst, மாறிலி uint8_t * src0, மாறிலி uint8_t * src1_plane,                         \
    uint32_t i01, uint32_t ne11, மாறிலி கட்டமைப்பு fastdiv_மதிப்புகள் * div11, uint32_t src1_stride,  \
    uint32_t n_rows, size_t dst_stride, size_t src0_stride, uint32_t ne00, uint32_t ne10) {  \
    க்கு (uint32_t r = 0; r < n_rows; r++) {                                                  \
        uint32_t i11 = fastmodulo(i01 + r, ne11, div11);                                     \
        மாறிலி uint8_t * r_src1_row = src1_plane + i11 * src1_stride;                         \
        uint8_t * r_dst = dst + r * dst_stride;                                              \
        மாறிலி uint8_t * r_src0 = src0 + r * src0_stride;                                     \
        க்கு (uint32_t c = 0; c < ne00; c += ne10) {                                          \
            uint32_t len = MIN(ne10, ne00 - c);                                              \
            HVX_FN(r_dst + c * அளவு(TYPE), r_src0 + c * அளவு(TYPE), r_src1_row, len);    \
        }                                                                                    \
    }                                                                                        \
}

DEFINE_COMPUTE_REPEAT(add_f32, தசம,    hvx_add_f32_uuu)
DEFINE_COMPUTE_REPEAT(add_f16, _Float16, hvx_add_f16_uuu)
DEFINE_COMPUTE_REPEAT(sub_f32, தசம,    hvx_sub_f32_uuu)
DEFINE_COMPUTE_REPEAT(sub_f16, _Float16, hvx_sub_f16_uuu)
DEFINE_COMPUTE_REPEAT(mul_f32, தசம,    hvx_mul_f32_uuu)
DEFINE_COMPUTE_REPEAT(mul_f16, _Float16, hvx_mul_f16_uuu)
DEFINE_COMPUTE_REPEAT(div_f32, தசம,    hvx_div_f32_uuu)
DEFINE_COMPUTE_REPEAT(div_f16, _Float16, hvx_div_f16_uuu)

வகைவரை வெறுமை (*compute_add_id_t)(
    uint8_t * dst, மாறிலி uint8_t * src0, மாறிலி uint8_t * src1_data, மாறிலி int32_t * src2_ids,
    uint32_t i01, uint32_t i02, uint32_t ne01, uint32_t src1_stride,
    uint32_t n_rows, size_t dst_stride, size_t src0_stride, uint32_t ne00);

நிலையான வெறுமை compute_add_id_f32(
    uint8_t * dst, மாறிலி uint8_t * src0, மாறிலி uint8_t * src1_data, மாறிலி int32_t * src2_ids,
    uint32_t i01, uint32_t i02, uint32_t ne01, uint32_t src1_stride,
    uint32_t n_rows, size_t dst_stride, size_t src0_stride, uint32_t ne00) {
    க்கு (uint32_t r = 0; r < n_rows; r++) {
        uint32_t r_i01 = i01 + r;
        மாறிலி int32_t idx = src2_ids[i02 * ne01 + r_i01];
        எனில் (idx < 0) {
            memcpy(dst + r * dst_stride, src0 + r * src0_stride, ne00 * அளவு(தசம));
            தொடர்;
        }
        மாறிலி uint8_t * r_src1 = src1_data + idx * src1_stride;
        மாறிலி uint8_t * r_src0 = src0 + r * src0_stride;
        uint8_t * r_dst = dst + r * dst_stride;
        hvx_add_f32_aaa(r_dst, r_src0, r_src1, ne00);
    }
}

// 1. Scalar src1 in VTCM via DMA (ne10 == 1)
நிலையான வெறுமை இருமம்_thread_scalar(unsigned முழு nth, unsigned முழு ith, வெறுமை * data) {
    கட்டமைப்பு htp_இருமம்_context * bctx = (கட்டமைப்பு htp_இருமம்_context *) data;
    கட்டமைப்பு htp_ops_context * octx = bctx->octx;
    htp_இருமம்_preamble;

    மாறிலி uint32_t elem_size = (src0->type == HTP_TYPE_F32) ? அளவு(தசம) : அளவு(_Float16);
    மாறிலி size_t plane_stride = (size_t) ne12 * ne11 * elem_size;
    மாறிலி size_t row_stride   = (size_t) ne11 * elem_size;

    மாறிலி uint32_t row_size_bytes = bctx->row_size_bytes;
    மாறிலி uint32_t start_row = bctx->row_start + bctx->nrows_per_thread * ith;
    மாறிலி uint32_t end_row   = MIN(start_row + bctx->nrows_per_thread, bctx->row_start + bctx->total_rows);
    எனில் (start_row >= end_row) திரும்பு;

    FARF(உயர்வு_2, "binary-scalar: %d/%d (%u:%u) row-size %u (%u)",
         ith, nth, start_row, end_row, nb01, bctx->dst_row_size_aligned);

    மாறிலி கட்டமைப்பு htp_இருமம்_vtcm_layout * layout = &bctx->vtcm_layout;
    uint8_t * src0_spad_base = VTCM_LAYOUT_PTR(uint8_t, bctx->vtcm_base, layout->off_src0) + (ith * layout->src0_bytes_per_thread);
    uint8_t * dst_spad_base  = VTCM_LAYOUT_PTR(uint8_t, bctx->vtcm_base, layout->off_dst)  + (ith * layout->dst_bytes_per_thread);
    size_t src0_spad_half    = layout->src0_spad_half_size;
    size_t dst_spad_half     = layout->dst_spad_half_size;
    மாறிலி வெறுமை * s1_table    = VTCM_LAYOUT_PTR(மாறிலி வெறுமை, bctx->vtcm_base, layout->off_src1);

    dma_queue * dma_q = octx->ctx->dma[ith];
    uint32_t ir_prefetch = start_row;
    முழு spad_idx = 0;

    க்கு (முழு k = 0; k < 2 && ir_prefetch < end_row; k++) {
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

        dma_queue_தள்ளு(dma_q, dma_make_data(dst_curr, d_spad), nb1, bctx->dst_row_size_aligned, row_size_bytes, 0);
        dma_queue_தள்ளு(dma_q, dma_make_data(s0_spad, src0_curr), bctx->src0_row_size_aligned, nb01, row_size_bytes,
                                                             current_block_size);
        ir_prefetch += current_block_size;
        spad_idx ^= 1;
    }

    கட்டமைப்பு htp_thread_trace * tr = &octx->ctx->trace[ith];
    compute_scalar_t compute = (compute_scalar_t) bctx->compute;

    க்கு (uint32_t ir = start_row; ir < end_row; ) {
        uint32_t current_block_size = calc_block_size(bctx, ir, end_row, ne01, ne02);

        uint8_t * d_spad  = (uint8_t *) dma_queue_எடு(dma_q).src;
        uint8_t * s0_spad = (uint8_t *) dma_queue_எடு(dma_q).dst;

        uint32_t i03, i02, i01, rem;
        i03 = fastdiv(ir, &bctx->src0_dim12_div);
        rem = ir - i03 * (ne02 * ne01);
        i02 = fastdiv(rem, &bctx->src0_dim1_div);
        i01 = rem - i02 * ne01;

        uint32_t i13 = fastmodulo(i03, ne13, &bctx->src1_dim3_div);
        uint32_t i12 = fastmodulo(i02, ne12, &bctx->src1_dim2_div);
        uint32_t i11 = fastmodulo(i01, ne11, &bctx->src1_dim1_div);
        மாறிலி uint8_t * scalar_table = (மாறிலி uint8_t *) s1_table +
            i13 * plane_stride + i12 * row_stride;

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);
        compute(d_spad, s0_spad, scalar_table, i11, ne11, current_block_size,
                bctx->dst_row_size_aligned, bctx->src0_row_size_aligned, ne00);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);

        dma_addr_t dst_curr = dst->data + i03 * nb3 + i02 * nb2 + i01 * nb1;
        dma_queue_தள்ளு(dma_q, dma_make_data(dst_curr, d_spad), nb1, bctx->dst_row_size_aligned, row_size_bytes,
                                                                current_block_size);

        எனில் (ir_prefetch < end_row) {
            uint32_t next_block_size = calc_block_size(bctx, ir_prefetch, end_row, ne01, ne02);
            uint32_t p03, p02, p01, prem;
            p03 = fastdiv(ir_prefetch, &bctx->src0_dim12_div);
            prem = ir_prefetch - p03 * (ne02 * ne01);
            p02 = fastdiv(prem, &bctx->src0_dim1_div);
            p01 = prem - p02 * ne01;
            dma_addr_t s0_next = src0->data + p03 * nb03 + p02 * nb02 + p01 * nb01;
            dma_queue_தள்ளு(dma_q, dma_make_data(s0_spad, s0_next), bctx->src0_row_size_aligned, nb01, row_size_bytes,
                                                               next_block_size);
            ir_prefetch += next_block_size;
        }
        ir += current_block_size;
    }

    dma_queue_flush(dma_q);
}

// 2. Vector Same Shape (ne1x == ne0x) or Simple Broadcast
நிலையான வெறுமை இருமம்_thread_vector_same_shape(unsigned முழு nth, unsigned முழு ith, வெறுமை * data) {
    கட்டமைப்பு htp_இருமம்_context * bctx = (கட்டமைப்பு htp_இருமம்_context *) data;
    கட்டமைப்பு htp_ops_context * octx = bctx->octx;
    htp_இருமம்_preamble;

    மாறிலி uint32_t row_size_bytes = bctx->row_size_bytes;
    மாறிலி uint32_t start_row = bctx->row_start + bctx->nrows_per_thread * ith;
    மாறிலி uint32_t end_row   = MIN(start_row + bctx->nrows_per_thread, bctx->row_start + bctx->total_rows);
    எனில் (start_row >= end_row) திரும்பு;

    FARF(உயர்வு_2, "binary-same-shape: %d/%d (%u:%u) row-size %u (%u)",
         ith, nth, start_row, end_row, nb01, bctx->dst_row_size_aligned);

    மாறிலி கட்டமைப்பு htp_இருமம்_vtcm_layout * layout = &bctx->vtcm_layout;
    uint8_t * src0_spad_base = VTCM_LAYOUT_PTR(uint8_t, bctx->vtcm_base, layout->off_src0) + (ith * layout->src0_bytes_per_thread);
    uint8_t * src1_spad_base = VTCM_LAYOUT_PTR(uint8_t, bctx->vtcm_base, layout->off_src1) + (ith * layout->src1_bytes_per_thread);
    uint8_t * dst_spad_base  = VTCM_LAYOUT_PTR(uint8_t, bctx->vtcm_base, layout->off_dst)  + (ith * layout->dst_bytes_per_thread);

    size_t src0_spad_half = layout->src0_spad_half_size;
    size_t src1_spad_half = layout->src1_spad_half_size;
    size_t dst_spad_half  = layout->dst_spad_half_size;

    dma_queue * dma_q = octx->ctx->dma[ith];
    uint32_t ir_prefetch = start_row;
    முழு spad_idx = 0;

    க்கு (முழு k = 0; k < 2 && ir_prefetch < end_row; k++) {
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

        dma_queue_தள்ளு(dma_q, dma_make_data(dst_curr, d_spad), nb1, bctx->dst_row_size_aligned, row_size_bytes, 0);
        dma_queue_தள்ளு(dma_q, dma_make_data(s0_spad, src0_curr), bctx->src0_row_size_aligned, nb01, row_size_bytes, current_block_size);
        dma_queue_தள்ளு(dma_q, dma_make_data(s1_spad, src1_curr), bctx->src1_row_size_aligned, nb11, row_size_bytes, current_block_size);
        ir_prefetch += current_block_size;
        spad_idx ^= 1;
    }

    கட்டமைப்பு htp_thread_trace * tr = &octx->ctx->trace[ith];
    compute_same_shape_t compute = (compute_same_shape_t) bctx->compute;

    க்கு (uint32_t ir = start_row; ir < end_row; ) {
        uint32_t current_block_size = calc_block_size(bctx, ir, end_row, ne01, ne02);
        uint8_t * d_spad  = (uint8_t *) dma_queue_எடு(dma_q).src;
        uint8_t * s0_spad = (uint8_t *) dma_queue_எடு(dma_q).dst;
        uint8_t * s1_spad = (uint8_t *) dma_queue_எடு(dma_q).dst;

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
        dma_queue_தள்ளு(dma_q, dma_make_data(dst_curr, d_spad), nb1, bctx->dst_row_size_aligned, row_size_bytes, current_block_size);

        எனில் (ir_prefetch < end_row) {
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

            dma_queue_தள்ளு(dma_q, dma_make_data(s0_spad, s0_next), bctx->src0_row_size_aligned, nb01, row_size_bytes, next_block_size);
            dma_queue_தள்ளு(dma_q, dma_make_data(s1_spad, s1_next), bctx->src1_row_size_aligned, nb11, row_size_bytes, next_block_size);

            ir_prefetch += next_block_size;
        }
        ir += current_block_size;
    }

    dma_queue_flush(dma_q);
}

// 3. Row Broadcast (ne11 == 1, ne12 == 1, single row src1)
நிலையான வெறுமை இருமம்_thread_vector_row_broadcast(unsigned முழு nth, unsigned முழு ith, வெறுமை * data) {
    கட்டமைப்பு htp_இருமம்_context * bctx = (கட்டமைப்பு htp_இருமம்_context *) data;
    கட்டமைப்பு htp_ops_context * octx = bctx->octx;
    htp_இருமம்_preamble;

    மாறிலி uint32_t row_size_bytes = bctx->row_size_bytes;
    மாறிலி uint32_t start_row  = bctx->row_start + bctx->nrows_per_thread * ith;
    மாறிலி uint32_t end_row    = MIN(start_row + bctx->nrows_per_thread, bctx->row_start + bctx->total_rows);
    எனில் (start_row >= end_row) திரும்பு;

    FARF(உயர்வு_2, "binary-row-bcast: %d/%d (%u:%u) row-size %u (%u)",
         ith, nth, start_row, end_row, nb01, bctx->dst_row_size_aligned);

    மாறிலி கட்டமைப்பு htp_இருமம்_vtcm_layout * layout = &bctx->vtcm_layout;
    uint8_t * src0_spad_base = VTCM_LAYOUT_PTR(uint8_t, bctx->vtcm_base, layout->off_src0) + (ith * layout->src0_bytes_per_thread);
    uint8_t * dst_spad_base  = VTCM_LAYOUT_PTR(uint8_t, bctx->vtcm_base, layout->off_dst)  + (ith * layout->dst_bytes_per_thread);

    size_t src0_spad_half = layout->src0_spad_half_size;
    size_t dst_spad_half  = layout->dst_spad_half_size;

    dma_queue * dma_q = octx->ctx->dma[ith];
    uint32_t ir_prefetch = start_row;
    முழு spad_idx = 0;

    வெறுமை * s1_ptr = VTCM_LAYOUT_PTR(வெறுமை, bctx->vtcm_base, layout->off_src1);

    க்கு (முழு k = 0; k < 2 && ir_prefetch < end_row; k++) {
        uint32_t current_block_size = calc_block_size(bctx, ir_prefetch, end_row, ne01, ne02);
        uint32_t i03 = fastdiv(ir_prefetch, &bctx->src0_dim12_div);
        uint32_t rem = ir_prefetch - i03 * (ne02 * ne01);
        uint32_t i02 = fastdiv(rem, &bctx->src0_dim1_div);
        uint32_t i01 = rem - i02 * ne01;

        dma_addr_t src0_curr = src0->data + i03 * nb03 + i02 * nb02 + i01 * nb01;
        dma_addr_t dst_curr  = dst->data  + i03 * nb3  + i02 * nb2  + i01 * nb1;

        uint8_t * s0_spad = src0_spad_base + spad_idx * src0_spad_half;
        uint8_t * d_spad  = dst_spad_base  + spad_idx * dst_spad_half;

        dma_queue_தள்ளு(dma_q, dma_make_data(dst_curr, d_spad), nb1, bctx->dst_row_size_aligned, row_size_bytes, 0);
        dma_queue_தள்ளு(dma_q, dma_make_data(s0_spad, src0_curr), bctx->src0_row_size_aligned, nb01, row_size_bytes, current_block_size);
        ir_prefetch += current_block_size;
        spad_idx ^= 1;
    }

    கட்டமைப்பு htp_thread_trace * tr = &octx->ctx->trace[ith];
    compute_row_bcast_t compute = (compute_row_bcast_t) bctx->compute;

    க்கு (uint32_t ir = start_row; ir < end_row; ) {
        uint32_t current_block_size = calc_block_size(bctx, ir, end_row, ne01, ne02);
        uint8_t * d_spad  = (uint8_t *) dma_queue_எடு(dma_q).src;
        uint8_t * s0_spad = (uint8_t *) dma_queue_எடு(dma_q).dst;

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);
        compute(d_spad, s0_spad, (மாறிலி uint8_t *)s1_ptr, current_block_size,
                bctx->dst_row_size_aligned, bctx->src0_row_size_aligned, ne00);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);

        uint32_t i03 = fastdiv(ir, &bctx->src0_dim12_div);
        uint32_t rem = ir - i03 * (ne02 * ne01);
        uint32_t i02 = fastdiv(rem, &bctx->src0_dim1_div);
        uint32_t i01 = rem - i02 * ne01;
        dma_addr_t dst_curr = dst->data + i03 * nb3 + i02 * nb2 + i01 * nb1;
        dma_queue_தள்ளு(dma_q, dma_make_data(dst_curr, d_spad), nb1, bctx->dst_row_size_aligned, row_size_bytes, current_block_size);

        எனில் (ir_prefetch < end_row) {
            uint32_t next_block_size = calc_block_size(bctx, ir_prefetch, end_row, ne01, ne02);
            uint32_t p03, p02, p01, prem;
            p03  = fastdiv(ir_prefetch, &bctx->src0_dim12_div);
            prem = ir_prefetch - p03 * (ne02 * ne01);
            p02  = fastdiv(prem, &bctx->src0_dim1_div);
            p01  = prem - p02 * ne01;
            dma_addr_t s0_next = src0->data + p03 * nb03 + p02 * nb02 + p01 * nb01;
            dma_queue_தள்ளு(dma_q, dma_make_data(s0_spad, s0_next), bctx->src0_row_size_aligned, nb01, row_size_bytes, next_block_size);
            ir_prefetch += next_block_size;
        }
        ir += current_block_size;
    }

    dma_queue_flush(dma_q);
}

// 4. Vector Complex (ne10 == ne00, complex broadcast)
நிலையான வெறுமை இருமம்_thread_vector_complex(unsigned முழு nth, unsigned முழு ith, வெறுமை * data) {
    கட்டமைப்பு htp_இருமம்_context * bctx = (கட்டமைப்பு htp_இருமம்_context *) data;
    கட்டமைப்பு htp_ops_context * octx = bctx->octx;
    htp_இருமம்_preamble;

    மாறிலி uint32_t row_size_bytes = bctx->row_size_bytes;
    மாறிலி uint32_t start_row  = bctx->row_start + bctx->nrows_per_thread * ith;
    மாறிலி uint32_t end_row    = MIN(start_row + bctx->nrows_per_thread, bctx->row_start + bctx->total_rows);
    எனில் (start_row >= end_row) திரும்பு;

    FARF(உயர்வு_2, "binary-complex: %d/%d (%u:%u) row-size %u (%u)",
         ith, nth, start_row, end_row, nb01, bctx->dst_row_size_aligned);

    மாறிலி கட்டமைப்பு htp_இருமம்_vtcm_layout * layout = &bctx->vtcm_layout;
    uint8_t * src0_spad_base = VTCM_LAYOUT_PTR(uint8_t, bctx->vtcm_base, layout->off_src0) + (ith * layout->src0_bytes_per_thread);
    uint8_t * dst_spad_base  = VTCM_LAYOUT_PTR(uint8_t, bctx->vtcm_base, layout->off_dst)  + (ith * layout->dst_bytes_per_thread);
    size_t src0_spad_half    = layout->src0_spad_half_size;
    size_t dst_spad_half     = layout->dst_spad_half_size;
    மாறிலி uint8_t * vtcm_src1 = VTCM_LAYOUT_PTR(மாறிலி uint8_t, bctx->vtcm_base, layout->off_src1);

    மாறிலி uint32_t elem_size = (src0->type == HTP_TYPE_F32) ? அளவு(தசம) : அளவு(_Float16);
    மாறிலி size_t vtcm_row_stride   = (size_t) ne10 * elem_size;
    மாறிலி size_t vtcm_plane_stride = (size_t) ne11 * vtcm_row_stride;
    மாறிலி size_t vtcm_vol_stride   = (size_t) ne12 * vtcm_plane_stride;

    dma_queue * dma_q = octx->ctx->dma[ith];
    uint32_t ir_prefetch = start_row;
    முழு spad_idx = 0;

    க்கு (முழு k = 0; k < 2 && ir_prefetch < end_row; k++) {
        uint32_t current_block_size = calc_block_size(bctx, ir_prefetch, end_row, ne01, ne02);
        uint32_t i03 = fastdiv(ir_prefetch, &bctx->src0_dim12_div);
        uint32_t rem = ir_prefetch - i03 * (ne02 * ne01);
        uint32_t i02 = fastdiv(rem, &bctx->src0_dim1_div);
        uint32_t i01 = rem - i02 * ne01;

        dma_addr_t src0_curr = src0->data + i03 * nb03 + i02 * nb02 + i01 * nb01;
        dma_addr_t dst_curr  = dst->data  + i03 * nb3  + i02 * nb2  + i01 * nb1;

        uint8_t * s0_spad = src0_spad_base + spad_idx * src0_spad_half;
        uint8_t * d_spad  = dst_spad_base  + spad_idx * dst_spad_half;

        dma_queue_தள்ளு(dma_q, dma_make_data(dst_curr, d_spad), nb1, bctx->dst_row_size_aligned, row_size_bytes, 0);
        dma_queue_தள்ளு(dma_q, dma_make_data(s0_spad, src0_curr), bctx->src0_row_size_aligned, nb01, row_size_bytes, current_block_size);
        ir_prefetch += current_block_size;
        spad_idx ^= 1;
    }

    கட்டமைப்பு htp_thread_trace * tr = &octx->ctx->trace[ith];
    compute_complex_t compute = (compute_complex_t) bctx->compute;

    க்கு (uint32_t ir = start_row; ir < end_row; ) {
        uint32_t current_block_size = calc_block_size(bctx, ir, end_row, ne01, ne02);
        uint8_t * d_spad = (uint8_t *) dma_queue_எடு(dma_q).src;
        uint8_t * s0_spad = (uint8_t *) dma_queue_எடு(dma_q).dst;

        uint32_t i03 = fastdiv(ir, &bctx->src0_dim12_div);
        uint32_t rem = ir - i03 * (ne02 * ne01);
        uint32_t i02 = fastdiv(rem, &bctx->src0_dim1_div);
        uint32_t i01 = rem - i02 * ne01;

        uint32_t i13 = fastmodulo(i03, ne13, &bctx->src1_dim3_div);
        uint32_t i12 = fastmodulo(i02, ne12, &bctx->src1_dim2_div);
        மாறிலி uint8_t * src1_plane = vtcm_src1 + i13 * vtcm_vol_stride + i12 * vtcm_plane_stride;

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);
        compute(d_spad, s0_spad, src1_plane, i01, ne11, &bctx->src1_dim1_div, vtcm_row_stride,
                current_block_size, bctx->dst_row_size_aligned, bctx->src0_row_size_aligned, ne00);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);

        dma_addr_t dst_curr = dst->data + i03 * nb3 + i02 * nb2 + i01 * nb1;
        dma_queue_தள்ளு(dma_q, dma_make_data(dst_curr, d_spad), nb1, bctx->dst_row_size_aligned, row_size_bytes, current_block_size);

        எனில் (ir_prefetch < end_row) {
            uint32_t next_block_size = calc_block_size(bctx, ir_prefetch, end_row, ne01, ne02);
            uint32_t p03, p02, p01, prem;
            p03  = fastdiv(ir_prefetch, &bctx->src0_dim12_div);
            prem = ir_prefetch - p03 * (ne02 * ne01);
            p02  = fastdiv(prem, &bctx->src0_dim1_div);
            p01  = prem - p02 * ne01;
            dma_addr_t s0_next = src0->data + p03 * nb03 + p02 * nb02 + p01 * nb01;
            dma_queue_தள்ளு(dma_q, dma_make_data(s0_spad, s0_next), bctx->src0_row_size_aligned, nb01, row_size_bytes, next_block_size);
            ir_prefetch += next_block_size;
        }
        ir += current_block_size;
    }

    dma_queue_flush(dma_q);
}

// 5. Element Repeat (ne10 != ne00)
நிலையான வெறுமை இருமம்_thread_element_repeat(unsigned முழு nth, unsigned முழு ith, வெறுமை * data) {
    கட்டமைப்பு htp_இருமம்_context * bctx = (கட்டமைப்பு htp_இருமம்_context *) data;
    கட்டமைப்பு htp_ops_context * octx = bctx->octx;
    htp_இருமம்_preamble;

    மாறிலி uint32_t row_size_bytes = bctx->row_size_bytes;
    மாறிலி uint32_t start_row  = bctx->row_start + bctx->nrows_per_thread * ith;
    மாறிலி uint32_t end_row    = MIN(start_row + bctx->nrows_per_thread, bctx->row_start + bctx->total_rows);
    எனில் (start_row >= end_row) திரும்பு;

    மாறிலி கட்டமைப்பு htp_இருமம்_vtcm_layout * layout = &bctx->vtcm_layout;
    uint8_t * src0_spad_base = VTCM_LAYOUT_PTR(uint8_t, bctx->vtcm_base, layout->off_src0) + (ith * layout->src0_bytes_per_thread);
    uint8_t * dst_spad_base  = VTCM_LAYOUT_PTR(uint8_t, bctx->vtcm_base, layout->off_dst)  + (ith * layout->dst_bytes_per_thread);
    size_t src0_spad_half    = layout->src0_spad_half_size;
    size_t dst_spad_half     = layout->dst_spad_half_size;
    மாறிலி uint8_t * vtcm_src1 = VTCM_LAYOUT_PTR(மாறிலி uint8_t, bctx->vtcm_base, layout->off_src1);

    மாறிலி uint32_t elem_size = (src0->type == HTP_TYPE_F32) ? அளவு(தசம) : அளவு(_Float16);
    மாறிலி size_t vtcm_row_stride   = (size_t) ne10 * elem_size;
    மாறிலி size_t vtcm_plane_stride = (size_t) ne11 * vtcm_row_stride;
    மாறிலி size_t vtcm_vol_stride   = (size_t) ne12 * vtcm_plane_stride;

    FARF(உயர்வு_2, "binary-repeat: %d/%d (%u:%u) row-size %u (%u)",
         ith, nth, start_row, end_row, nb01, bctx->dst_row_size_aligned);

    dma_queue * dma_q = octx->ctx->dma[ith];
    uint32_t ir_prefetch = start_row;
    முழு spad_idx = 0;

    க்கு (முழு k = 0; k < 2 && ir_prefetch < end_row; k++) {
        uint32_t current_block_size = calc_block_size(bctx, ir_prefetch, end_row, ne01, ne02);
        uint32_t i03 = fastdiv(ir_prefetch, &bctx->src0_dim12_div);
        uint32_t rem = ir_prefetch - i03 * (ne02 * ne01);
        uint32_t i02 = fastdiv(rem, &bctx->src0_dim1_div);
        uint32_t i01 = rem - i02 * ne01;

        dma_addr_t src0_curr = src0->data + i03 * nb03 + i02 * nb02 + i01 * nb01;
        dma_addr_t dst_curr  = dst->data  + i03 * nb3  + i02 * nb2  + i01 * nb1;

        uint8_t * s0_spad = src0_spad_base + spad_idx * src0_spad_half;
        uint8_t * d_spad  = dst_spad_base  + spad_idx * dst_spad_half;

        dma_queue_தள்ளு(dma_q, dma_make_data(dst_curr, d_spad), nb1, bctx->dst_row_size_aligned, row_size_bytes, 0);
        dma_queue_தள்ளு(dma_q, dma_make_data(s0_spad, src0_curr), bctx->src0_row_size_aligned, nb01, row_size_bytes, current_block_size);
        ir_prefetch += current_block_size;
        spad_idx ^= 1;
    }

    கட்டமைப்பு htp_thread_trace * tr = &octx->ctx->trace[ith];
    compute_repeat_t compute = (compute_repeat_t) bctx->compute;

    க்கு (uint32_t ir = start_row; ir < end_row; ) {
        uint32_t current_block_size = calc_block_size(bctx, ir, end_row, ne01, ne02);
        uint8_t * d_spad  = (uint8_t *) dma_queue_எடு(dma_q).src;
        uint8_t * s0_spad = (uint8_t *) dma_queue_எடு(dma_q).dst;

        uint32_t i03 = fastdiv(ir, &bctx->src0_dim12_div);
        uint32_t rem = ir - i03 * (ne02 * ne01);
        uint32_t i02 = fastdiv(rem, &bctx->src0_dim1_div);
        uint32_t i01 = rem - i02 * ne01;

        uint32_t i13 = fastmodulo(i03, ne13, &bctx->src1_dim3_div);
        uint32_t i12 = fastmodulo(i02, ne12, &bctx->src1_dim2_div);
        மாறிலி uint8_t * src1_plane = vtcm_src1 + i13 * vtcm_vol_stride + i12 * vtcm_plane_stride;

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);
        compute(d_spad, s0_spad, src1_plane, i01, ne11, &bctx->src1_dim1_div, vtcm_row_stride,
                current_block_size, bctx->dst_row_size_aligned, bctx->src0_row_size_aligned, ne00, ne10);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);

        dma_addr_t dst_curr = dst->data + i03 * nb3 + i02 * nb2 + i01 * nb1;
        dma_queue_தள்ளு(dma_q, dma_make_data(dst_curr, d_spad), nb1, bctx->dst_row_size_aligned, row_size_bytes, current_block_size);

        எனில் (ir_prefetch < end_row) {
            uint32_t next_block_size = calc_block_size(bctx, ir_prefetch, end_row, ne01, ne02);
            uint32_t p03, p02, p01, prem;
            p03  = fastdiv(ir_prefetch, &bctx->src0_dim12_div);
            prem = ir_prefetch - p03 * (ne02 * ne01);
            p02  = fastdiv(prem, &bctx->src0_dim1_div);
            p01  = prem - p02 * ne01;
            dma_addr_t s0_next = src0->data + p03 * nb03 + p02 * nb02 + p01 * nb01;
            dma_queue_தள்ளு(dma_q, dma_make_data(s0_spad, s0_next), bctx->src0_row_size_aligned, nb01, row_size_bytes, next_block_size);
            ir_prefetch += next_block_size;
        }
        ir += current_block_size;
    }

    dma_queue_flush(dma_q);
}

// 6. ADD_ID (src1 gathered via src2 indices)
நிலையான வெறுமை இருமம்_thread_add_id_f32(unsigned முழு nth, unsigned முழு ith, வெறுமை * data) {
    கட்டமைப்பு htp_இருமம்_context * bctx = (கட்டமைப்பு htp_இருமம்_context *) data;
    கட்டமைப்பு htp_ops_context * octx = bctx->octx;

    மாறிலி கட்டமைப்பு htp_tensor * src0 = octx->src[0];
    மாறிலி கட்டமைப்பு htp_tensor * src1 = octx->src[1];
    மாறிலி கட்டமைப்பு htp_tensor * dst  = octx->dst;

    மாறிலி uint32_t ne00 = src0->ne[0];
    மாறிலி uint32_t ne01 = src0->ne[1];
    மாறிலி uint32_t ne02 = src0->ne[2];
    மாறிலி uint32_t ne03 = src0->ne[3];

    மாறிலி uint32_t nb01 = src0->nb[1];
    மாறிலி uint32_t nb02 = src0->nb[2];
    மாறிலி uint32_t nb03 = src0->nb[3];
    மாறிலி uint32_t src1_stride = bctx->src1_row_size_aligned;

    மாறிலி uint32_t nb1 = dst->nb[1];
    மாறிலி uint32_t nb2 = dst->nb[2];
    மாறிலி uint32_t nb3 = dst->nb[3];

    மாறிலி uint32_t row_size_bytes = bctx->row_size_bytes;
    மாறிலி uint32_t start_row = bctx->row_start + bctx->nrows_per_thread * ith;
    மாறிலி uint32_t end_row   = MIN(start_row + bctx->nrows_per_thread, bctx->row_start + bctx->total_rows);
    எனில் (start_row >= end_row) திரும்பு;

    மாறிலி கட்டமைப்பு htp_இருமம்_vtcm_layout * layout = &bctx->vtcm_layout;
    uint8_t * src0_spad_base = VTCM_LAYOUT_PTR(uint8_t, bctx->vtcm_base, layout->off_src0) + (ith * layout->src0_bytes_per_thread);
    uint8_t * dst_spad_base  = VTCM_LAYOUT_PTR(uint8_t, bctx->vtcm_base, layout->off_dst)  + (ith * layout->dst_bytes_per_thread);
    மாறிலி uint8_t * vtcm_src1 = VTCM_LAYOUT_PTR(மாறிலி uint8_t, bctx->vtcm_base, layout->off_src1);
    மாறிலி int32_t * vtcm_src2 = VTCM_LAYOUT_PTR(மாறிலி int32_t, bctx->vtcm_base, layout->off_src2);
    size_t src0_spad_half    = layout->src0_spad_half_size;
    size_t dst_spad_half     = layout->dst_spad_half_size;

    dma_queue * dma_q = octx->ctx->dma[ith];
    uint32_t ir_prefetch = start_row;
    முழு spad_idx = 0;

    க்கு (முழு k = 0; k < 2 && ir_prefetch < end_row; k++) {
        uint32_t current_block_size = calc_block_size(bctx, ir_prefetch, end_row, ne01, ne02);
        uint32_t i03 = fastdiv(ir_prefetch, &bctx->src0_dim12_div);
        uint32_t rem = ir_prefetch - i03 * (ne02 * ne01);
        uint32_t i02 = fastdiv(rem, &bctx->src0_dim1_div);
        uint32_t i01 = rem - i02 * ne01;

        dma_addr_t src0_curr = src0->data + i03 * nb03 + i02 * nb02 + i01 * nb01;
        dma_addr_t dst_curr  = dst->data  + i03 * nb3  + i02 * nb2  + i01 * nb1;

        uint8_t * s0_spad = src0_spad_base + spad_idx * src0_spad_half;
        uint8_t * d_spad  = dst_spad_base  + spad_idx * dst_spad_half;

        dma_queue_தள்ளு(dma_q, dma_make_data(dst_curr, d_spad), nb1, bctx->dst_row_size_aligned, row_size_bytes, 0);
        dma_queue_தள்ளு(dma_q, dma_make_data(s0_spad, src0_curr), bctx->src0_row_size_aligned, nb01, row_size_bytes, current_block_size);
        ir_prefetch += current_block_size;
        spad_idx ^= 1;
    }

    கட்டமைப்பு htp_thread_trace * tr = &octx->ctx->trace[ith];

    க்கு (uint32_t ir = start_row; ir < end_row; ) {
        uint32_t current_block_size = calc_block_size(bctx, ir, end_row, ne01, ne02);
        uint8_t * d_spad = (uint8_t *) dma_queue_எடு(dma_q).src;
        uint8_t * s0_spad = (uint8_t *) dma_queue_எடு(dma_q).dst;

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
        dma_queue_தள்ளு(dma_q, dma_make_data(dst_curr, d_spad), nb1, bctx->dst_row_size_aligned, row_size_bytes, current_block_size);

        எனில் (ir_prefetch < end_row) {
             uint32_t next_block_size = calc_block_size(bctx, ir_prefetch, end_row, ne01, ne02);
             uint32_t p03, p02, p01, prem;
             p03  = fastdiv(ir_prefetch, &bctx->src0_dim12_div);
             prem = ir_prefetch - p03 * (ne02 * ne01);
             p02  = fastdiv(prem, &bctx->src0_dim1_div);
             p01  = prem - p02 * ne01;
             dma_addr_t s0_next = src0->data + p03 * nb03 + p02 * nb02 + p01 * nb01;
             dma_queue_தள்ளு(dma_q, dma_make_data(s0_spad, s0_next), bctx->src0_row_size_aligned, nb01, row_size_bytes, next_block_size);
             ir_prefetch += next_block_size;
        }
        ir += current_block_size;
    }

    dma_queue_flush(dma_q);
}

நிலையான உள்வரி வெறுமை hvx_div_scalar_f32(uint8_t * restrict dst, மாறிலி uint8_t * restrict src, மாறிலி தசம val, மாறிலி uint32_t இலக்கம்_elems) {
    hvx_mul_scalar_f32(dst, src, 1.0f / val, இலக்கம்_elems);
}

நிலையான உள்வரி வெறுமை hvx_div_scalar_f16(uint8_t * restrict dst, மாறிலி uint8_t * restrict src, மாறிலி _Float16 val, மாறிலி uint32_t இலக்கம்_elems) {
    hvx_div_scalar_f16_aa(dst, src, val, இலக்கம்_elems);
}

வகைவரை வெறுமை (*compute_இருமம்_chunked_t)(
    uint8_t * restrict dst,
    மாறிலி uint8_t * restrict src0,
    மாறிலி uint8_t * restrict src1,
    மாறிலி uint32_t இலக்கம்_elems
);

வகைவரை வெறுமை (*compute_இருமம்_scalar_chunked_f32_t)(
    uint8_t * restrict dst,
    மாறிலி uint8_t * restrict src0,
    மாறிலி தசம val,
    மாறிலி uint32_t இலக்கம்_elems
);

வகைவரை வெறுமை (*compute_இருமம்_scalar_chunked_f16_t)(
    uint8_t * restrict dst,
    மாறிலி uint8_t * restrict src0,
    மாறிலி _Float16 val,
    மாறிலி uint32_t இலக்கம்_elems
);

கட்டமைப்பு இருமம்_chunked_context {
    கட்டமைப்பு htp_ops_context * octx;
    கட்டமைப்பு htp_இருமம்_vtcm_layout vtcm_layout;
    uint8_t *                vtcm_base;
    uint32_t                 elem_start;
    uint32_t                 nelem;
    uint32_t                 chunk_size;
    uint32_t                 chunks_per_thread;
    uint32_t                 total_chunks;
    bool                     is_scalar;
    தசம                    scalar_f32;
    _Float16                 scalar_f16;
    compute_இருமம்_chunked_t compute;
    compute_இருமம்_scalar_chunked_f32_t compute_scalar_f32;
    compute_இருமம்_scalar_chunked_f16_t compute_scalar_f16;
};

நிலையான வெறுமை இருமம்_thread_chunked(unsigned முழு nth, unsigned முழு ith, வெறுமை * data) {
    (வெறுமை) nth;
    கட்டமைப்பு இருமம்_chunked_context * ctx = (கட்டமைப்பு இருமம்_chunked_context *) data;
    கட்டமைப்பு htp_ops_context * octx = ctx->octx;
    மாறிலி கட்டமைப்பு htp_tensor * src0 = octx->src[0];
    மாறிலி கட்டமைப்பு htp_tensor * src1 = octx->src[1];
    மாறிலி கட்டமைப்பு htp_tensor * dst  = octx->dst;

    மாறிலி uint32_t start_chunk = ctx->chunks_per_thread * ith;
    மாறிலி uint32_t end_chunk   = MIN(start_chunk + ctx->chunks_per_thread, ctx->total_chunks);
    எனில் (start_chunk >= end_chunk) {
        திரும்பு;
    }

    மாறிலி uint32_t src0_type   = src0->type;
    மாறிலி size_t   elem_size   = (src0_type == HTP_TYPE_F32) ? அளவு(தசம) : அளவு(_Float16);
    மாறிலி uint32_t chunk_size  = ctx->chunk_size;
    மாறிலி size_t   chunk_bytes = ctx->vtcm_layout.src0_spad_half_size;

    FARF(உயர்வு_2, "binary-chunked: %d/%d (%u:%u) chunks %u elems %u",
         ith, nth, start_chunk, end_chunk, ctx->total_chunks, ctx->nelem);

    மாறிலி கட்டமைப்பு htp_இருமம்_vtcm_layout * layout = &ctx->vtcm_layout;
    uint8_t * src0_spad_base = VTCM_LAYOUT_PTR(uint8_t, ctx->vtcm_base, layout->off_src0) + (ith * layout->src0_bytes_per_thread);
    uint8_t * src1_spad_base = ctx->is_scalar ? வெற்று : (VTCM_LAYOUT_PTR(uint8_t, ctx->vtcm_base, layout->off_src1) + (ith * layout->src1_bytes_per_thread));
    uint8_t * dst_spad_base  = VTCM_LAYOUT_PTR(uint8_t, ctx->vtcm_base, layout->off_dst)  + (ith * layout->dst_bytes_per_thread);

    dma_queue * dma_q = octx->ctx->dma[ith];
    uint32_t prefetch_chunk = start_chunk;
    முழு spad_idx = 0;

    க்கு (முழு k = 0; k < 2 && prefetch_chunk < end_chunk; k++) {
        மாறிலி uint32_t c_start   = ctx->elem_start + prefetch_chunk * chunk_size;
        மாறிலி uint32_t c_end     = MIN(c_start + chunk_size, ctx->elem_start + ctx->nelem);
        மாறிலி uint32_t cur_elems = c_end - c_start;
        மாறிலி uint32_t cur_bytes = cur_elems * elem_size;

        dma_addr_t s0_curr = src0->data + (size_t) c_start * elem_size;
        dma_addr_t d_curr  = dst->data  + (size_t) c_start * elem_size;

        uint8_t * s0_spad = src0_spad_base + spad_idx * chunk_bytes;
        uint8_t * d_spad  = dst_spad_base  + spad_idx * chunk_bytes;

        dma_queue_தள்ளு(dma_q, dma_make_data(d_curr, d_spad), chunk_bytes, chunk_bytes, cur_bytes, 0);
        dma_queue_தள்ளு(dma_q, dma_make_data(s0_spad, s0_curr), chunk_bytes, chunk_bytes, cur_bytes, 1);
        எனில் (!ctx->is_scalar) {
            dma_addr_t s1_curr = src1->data + (size_t) c_start * elem_size;
            uint8_t * s1_spad = src1_spad_base + spad_idx * chunk_bytes;
            dma_queue_தள்ளு(dma_q, dma_make_data(s1_spad, s1_curr), chunk_bytes, chunk_bytes, cur_bytes, 1);
        }

        prefetch_chunk++;
        spad_idx ^= 1;
    }

    கட்டமைப்பு htp_thread_trace * tr = &octx->ctx->trace[ith];

    க்கு (uint32_t c = start_chunk; c < end_chunk; c++) {
        மாறிலி uint32_t c_start   = ctx->elem_start + c * chunk_size;
        மாறிலி uint32_t c_end     = MIN(c_start + chunk_size, ctx->elem_start + ctx->nelem);
        மாறிலி uint32_t cur_elems = c_end - c_start;
        மாறிலி uint32_t cur_bytes = cur_elems * elem_size;

        uint8_t * d_spad  = (uint8_t *) dma_queue_எடு(dma_q).src;
        uint8_t * s0_spad = (uint8_t *) dma_queue_எடு(dma_q).dst;
        uint8_t * s1_spad = ctx->is_scalar ? வெற்று : (uint8_t *) dma_queue_எடு(dma_q).dst;

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) c);
        எனில் (ctx->is_scalar) {
            எனில் (src0_type == HTP_TYPE_F32) {
                ctx->compute_scalar_f32(d_spad, s0_spad, ctx->scalar_f32, cur_elems);
            } இல்லையெனில் {
                ctx->compute_scalar_f16(d_spad, s0_spad, ctx->scalar_f16, cur_elems);
            }
        } இல்லையெனில் {
            ctx->compute(d_spad, s0_spad, s1_spad, cur_elems);
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) c);

        dma_addr_t dst_curr = dst->data + (size_t) c_start * elem_size;
        dma_queue_தள்ளு(dma_q, dma_make_data(dst_curr, d_spad), chunk_bytes, chunk_bytes, cur_bytes, 1);

        எனில் (prefetch_chunk < end_chunk) {
            மாறிலி uint32_t pc_start  = ctx->elem_start + prefetch_chunk * chunk_size;
            மாறிலி uint32_t pc_end    = MIN(pc_start + chunk_size, ctx->elem_start + ctx->nelem);
            மாறிலி uint32_t p_elems   = pc_end - pc_start;
            மாறிலி uint32_t p_bytes   = p_elems * elem_size;

            dma_addr_t s0_next = src0->data + (size_t) pc_start * elem_size;
            dma_queue_தள்ளு(dma_q, dma_make_data(s0_spad, s0_next), chunk_bytes, chunk_bytes, p_bytes, 1);
            எனில் (!ctx->is_scalar) {
                dma_addr_t s1_next = src1->data + (size_t) pc_start * elem_size;
                dma_queue_தள்ளு(dma_q, dma_make_data(s1_spad, s1_next), chunk_bytes, chunk_bytes, p_bytes, 1);
            }

            prefetch_chunk++;
        }
    }

    dma_queue_flush(dma_q);
}

நிலையான முழு execute_op_இருமம்(கட்டமைப்பு htp_ops_context * octx) {
    மாறிலி கட்டமைப்பு htp_tensor * src0 = octx->src[0];
    மாறிலி கட்டமைப்பு htp_tensor * src1 = octx->src[1];
    மாறிலி கட்டமைப்பு htp_tensor * dst  = octx->dst;
    மாறிலி கட்டமைப்பு htp_இருமம்_kernel_params * kparams = (மாறிலி கட்டமைப்பு htp_இருமம்_kernel_params *) octx->kernel_params;

    மாறிலி uint32_t src0_nrows = src0->ne[1] * src0->ne[2] * src0->ne[3];

    // Use packed row sizes for VTCM allocation and alignment
    மாறிலி uint32_t src0_type = octx->src[0]->type;
    மாறிலி size_t elem_size = (src0_type == HTP_TYPE_F32) ? அளவு(தசம) : அளவு(_Float16);
    மாறிலி size_t src0_row_size = src0->ne[0] * elem_size;
    மாறிலி size_t src1_row_size = src1->ne[0] * elem_size;
    மாறிலி size_t dst_row_size  = dst->ne[0]  * elem_size;

    எனில் (kparams->kernel_type == HTP_இருமம்_KERNEL_CHUNKED) {
        மாறிலி bool is_scalar = kparams->is_scalar || (src1->ne[0] == 1 && src1->ne[1] == 1 && src1->ne[2] == 1 && src1->ne[3] == 1);

        compute_இருமம்_chunked_t compute = வெற்று;
        compute_இருமம்_scalar_chunked_f32_t compute_scalar_f32 = வெற்று;
        compute_இருமம்_scalar_chunked_f16_t compute_scalar_f16 = வெற்று;

        எனில் (is_scalar) {
            எனில் (src0_type == HTP_TYPE_F32) {
                தேர்வு (octx->op) {
                    நிலை HTP_OP_ADD: compute_scalar_f32 = hvx_add_scalar_f32; நிறுத்து;
                    நிலை HTP_OP_SUB: compute_scalar_f32 = hvx_sub_scalar_f32; நிறுத்து;
                    நிலை HTP_OP_MUL: compute_scalar_f32 = hvx_mul_scalar_f32; நிறுத்து;
                    நிலை HTP_OP_DIV: compute_scalar_f32 = hvx_div_scalar_f32; நிறுத்து;
                    இயல்பு: நிறுத்து;
                }
            } இல்லையெனில் எனில் (src0_type == HTP_TYPE_F16) {
                தேர்வு (octx->op) {
                    நிலை HTP_OP_ADD: compute_scalar_f16 = hvx_add_scalar_f16; நிறுத்து;
                    நிலை HTP_OP_SUB: compute_scalar_f16 = hvx_sub_scalar_f16; நிறுத்து;
                    நிலை HTP_OP_MUL: compute_scalar_f16 = hvx_mul_scalar_f16; நிறுத்து;
                    நிலை HTP_OP_DIV: compute_scalar_f16 = hvx_div_scalar_f16; நிறுத்து;
                    இயல்பு: நிறுத்து;
                }
            }
            எனில் (!compute_scalar_f32 && !compute_scalar_f16) {
                திரும்பு HTP_STATUS_NO_SUPPORT;
            }
        } இல்லையெனில் {
            எனில் (src0_type == HTP_TYPE_F32) {
                தேர்வு (octx->op) {
                    நிலை HTP_OP_ADD: compute = hvx_add_f32; நிறுத்து;
                    நிலை HTP_OP_SUB: compute = hvx_sub_f32; நிறுத்து;
                    நிலை HTP_OP_MUL: compute = hvx_mul_f32; நிறுத்து;
                    நிலை HTP_OP_DIV: compute = hvx_div_f32; நிறுத்து;
                    இயல்பு: நிறுத்து;
                }
            } இல்லையெனில் எனில் (src0_type == HTP_TYPE_F16) {
                தேர்வு (octx->op) {
                    நிலை HTP_OP_ADD: compute = hvx_add_f16; நிறுத்து;
                    நிலை HTP_OP_SUB: compute = hvx_sub_f16; நிறுத்து;
                    நிலை HTP_OP_MUL: compute = hvx_mul_f16; நிறுத்து;
                    நிலை HTP_OP_DIV: compute = hvx_div_f16; நிறுத்து;
                    இயல்பு: நிறுத்து;
                }
            }
            எனில் (!compute) {
                திரும்பு HTP_STATUS_NO_SUPPORT;
            }
        }

        மாறிலி uint32_t total_elems = (uint32_t) (src0->ne[0] * src0->ne[1] * src0->ne[2] * src0->ne[3]);
        எனில் (total_elems == 0) {
            திரும்பு HTP_STATUS_OK;
        }

        uint32_t elem_start = 0;
        uint32_t nelem      = total_elems;
        மாறிலி uint32_t elems_per_வரி = (src0_type == HTP_TYPE_F32) ? 32 : 64;

        எனில் (octx->ctx->mdev.மொத்தம் > 1) {
            மாறிலி bool can_split = htp_tensor_mdev_data_aligned(dst) &&
                                   htp_tensor_mdev_data_aligned(src0) &&
                                   (is_scalar || htp_tensor_mdev_data_aligned(src1)) &&
                                   htp_tensor_is_contiguous(dst, elem_size) &&
                                   htp_tensor_is_contiguous(src0, elem_size) &&
                                   (is_scalar || htp_tensor_is_contiguous(src1, elem_size));
            மாறிலி கட்டமைப்பு htp_tensor_mdev_range range = htp_tensor_mdev_partition(
                total_elems, can_split ? elems_per_வரி : 0,
                octx->ctx->mdev.idx, octx->ctx->mdev.மொத்தம், &octx->ctx->mdev.மொத்தம்_div);
            elem_start = range.start;
            nelem      = range.மொத்தம்;
        }

        எனில் (nelem == 0) {
            திரும்பு HTP_STATUS_OK;
        }

        எனில் (!htp_ops_context_set_n_threads(octx, kparams->n_threads)) {
            திரும்பு HTP_STATUS_INVAL_PARAMS;
        }

        கட்டமைப்பு htp_இருமம்_vtcm_layout vtcm_layout;
        htp_இருமம்_vtcm_layout_build(&vtcm_layout, kparams, octx->ctx->vtcm_size);
        எனில் (vtcm_layout.total_bytes == 0 || vtcm_layout.total_bytes > octx->ctx->vtcm_size) {
            திரும்பு HTP_STATUS_VTCM_TOO_SMALL;
        }

        மாறிலி uint32_t chunk_size = kparams->chunk_size > 0 ? kparams->chunk_size : (32768 / elem_size);
        மாறிலி uint32_t total_chunks = (nelem + chunk_size - 1) / chunk_size;
        மாறிலி uint32_t n_threads = (total_chunks >= 2) ? MIN(octx->n_threads, total_chunks) : 1;
        மாறிலி uint32_t chunks_per_thread = (total_chunks + n_threads - 1) / n_threads;

        தசம scalar_f32 = 0.0f;
        _Float16 scalar_f16 = 0;
        எனில் (is_scalar) {
            uint8_t * vtcm_src1 = VTCM_LAYOUT_PTR(uint8_t, octx->ctx->vtcm_base, vtcm_layout.off_src1);
            dma_queue * dma_q = octx->ctx->dma[0];
            dma_queue_தள்ளு(dma_q, dma_make_data(vtcm_src1, src1->data), 128, 0, elem_size, 1);
            dma_queue_எடு(dma_q);

            எனில் (src0_type == HTP_TYPE_F32) {
                scalar_f32 = ((மாறிலி தசம *) vtcm_src1)[0];
            } இல்லையெனில் {
                scalar_f16 = ((மாறிலி _Float16 *) vtcm_src1)[0];
            }
        }

        கட்டமைப்பு இருமம்_chunked_context cctx = {
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

        work_queue_run(octx->ctx->work_queue, இருமம்_thread_chunked, &cctx, n_threads);
        திரும்பு HTP_STATUS_OK;
    }

    uint32_t row_start = 0;
    uint32_t nrows     = src0_nrows;

    எனில் (octx->ctx->mdev.மொத்தம் > 1) {
        uint32_t rows_per_chunk = 0;
        htp_tensor_mdev_rows_per_chunk(dst, (uint32_t) elem_size, (uint32_t) dst_row_size, &rows_per_chunk);
        மாறிலி கட்டமைப்பு htp_tensor_mdev_range range = htp_tensor_mdev_partition(src0_nrows, rows_per_chunk,
                                                       octx->ctx->mdev.idx, octx->ctx->mdev.மொத்தம், &octx->ctx->mdev.மொத்தம்_div);
        row_start = range.start;
        nrows     = range.மொத்தம்;
    }

    எனில் (nrows == 0) {
        திரும்பு HTP_STATUS_OK;
    }

    எனில் (!htp_ops_context_set_n_threads(octx, kparams->n_threads)) {
        திரும்பு HTP_STATUS_INVAL_PARAMS;
    }

    மாறிலி uint32_t n_threads = octx->n_threads;
    மாறிலி size_t src0_row_size_aligned = kparams->src0_row_size_aligned;
    மாறிலி size_t src1_row_size_aligned = kparams->src1_row_size_aligned;
    மாறிலி size_t dst_row_size_aligned  = kparams->dst_row_size_aligned;

    கட்டமைப்பு htp_இருமம்_context bctx;
    bctx.vtcm_base = (uint8_t *) octx->ctx->vtcm_base;
    htp_இருமம்_vtcm_layout_build(&bctx.vtcm_layout, kparams, octx->ctx->vtcm_size);

    எனில் (bctx.vtcm_layout.rows_per_buffer == 0 || bctx.vtcm_layout.total_bytes > octx->ctx->vtcm_size) {
        திரும்பு HTP_STATUS_VTCM_TOO_SMALL;
    }

    dma_queue * dma_q = octx->ctx->dma[0];
    uint8_t * vtcm_src1 = VTCM_LAYOUT_PTR(uint8_t, bctx.vtcm_base, bctx.vtcm_layout.off_src1);
    எனில் (kparams->kernel_type == HTP_இருமம்_KERNEL_ROW_BCAST) {
        dma_queue_தள்ளு(dma_q, dma_make_data(vtcm_src1, src1->data), bctx.vtcm_layout.src1_size, 0, src1->ne[0] * elem_size, 1);
    } இல்லையெனில் எனில் (kparams->kernel_type == HTP_இருமம்_KERNEL_SCALAR ||
               kparams->kernel_type == HTP_இருமம்_KERNEL_REPEAT ||
               kparams->kernel_type == HTP_இருமம்_KERNEL_COMPLEX) {
        எனில் (!kparams->src1_strided) {
            மாறிலி size_t src1_bytes = (size_t) src1->ne[0] * src1->ne[1] * src1->ne[2] * src1->ne[3] * elem_size;
            dma_queue_தள்ளு(dma_q, dma_make_data(vtcm_src1, src1->data), bctx.vtcm_layout.src1_size, 0, src1_bytes, 1);
        } இல்லையெனில் {
            மாறிலி uint32_t ne11 = src1->ne[1];
            மாறிலி uint32_t ne12 = src1->ne[2];
            மாறிலி uint32_t ne13 = src1->ne[3];
            மாறிலி size_t row_bytes = src1->ne[0] * elem_size;
            க்கு (uint32_t i13 = 0; i13 < ne13; i13++) {
                க்கு (uint32_t i12 = 0; i12 < ne12; i12++) {
                    dma_addr_t src_addr = src1->data + i13 * src1->nb[3] + i12 * src1->nb[2];
                    uint8_t * dst_ptr = vtcm_src1 + (i13 * ne12 * ne11 + i12 * ne11) * row_bytes;
                    போது (!dma_queue_தள்ளு(dma_q, dma_make_data(dst_ptr, src_addr),
                                           row_bytes, src1->nb[1], row_bytes, ne11)) {
                        dma_queue_எடு(dma_q);
                    }
                }
            }
        }
    } இல்லையெனில் எனில் (kparams->kernel_type == HTP_இருமம்_KERNEL_ADD_ID) {
        மாறிலி கட்டமைப்பு htp_tensor * src2 = octx->src[2];
        uint8_t * vtcm_src2 = VTCM_LAYOUT_PTR(uint8_t, bctx.vtcm_base, bctx.vtcm_layout.off_src2);
        dma_queue_தள்ளு(dma_q, dma_make_data(vtcm_src1, src1->data),
                       kparams->src1_row_size_aligned, src1->nb[1],
                       src1->ne[0] * elem_size, src1->ne[1]);
        மாறிலி size_t src2_row_bytes = src2->ne[0] * அளவு(int32_t);
        dma_queue_தள்ளு(dma_q, dma_make_data(vtcm_src2, src2->data),
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

    bctx.src0_dim1_div  = init_fastdiv_மதிப்புகள்(src0->ne[1]);
    bctx.src0_dim2_div  = init_fastdiv_மதிப்புகள்(src0->ne[2]);
    bctx.src0_dim12_div = init_fastdiv_மதிப்புகள்(src0->ne[1] * src0->ne[2]);

    bctx.src1_dim1_div  = init_fastdiv_மதிப்புகள்(src1->ne[1]);
    bctx.src1_dim2_div  = init_fastdiv_மதிப்புகள்(src1->ne[2]);
    bctx.src1_dim3_div  = init_fastdiv_மதிப்புகள்(src1->ne[3]);

    bctx.split_at_ne01  = kparams->split_at_ne01;
    bctx.split_at_ne02  = kparams->split_at_ne02;

    worker_callback_t worker_func = வெற்று;
    வெறுமை * compute_func = வெற்று;

    தேர்வு (kparams->kernel_type) {
        நிலை HTP_இருமம்_KERNEL_SAME_SHAPE:
            worker_func = இருமம்_thread_vector_same_shape;
            எனில் (src0_type == HTP_TYPE_F32) {
                தேர்வு (octx->op) {
                    நிலை HTP_OP_ADD: compute_func = compute_same_shape_add_f32; நிறுத்து;
                    நிலை HTP_OP_SUB: compute_func = compute_same_shape_sub_f32; நிறுத்து;
                    நிலை HTP_OP_MUL: compute_func = compute_same_shape_mul_f32; நிறுத்து;
                    நிலை HTP_OP_DIV: compute_func = compute_same_shape_div_f32; நிறுத்து;
                    இயல்பு: நிறுத்து;
                }
            } இல்லையெனில் எனில் (src0_type == HTP_TYPE_F16) {
                தேர்வு (octx->op) {
                    நிலை HTP_OP_ADD: compute_func = compute_same_shape_add_f16; நிறுத்து;
                    நிலை HTP_OP_SUB: compute_func = compute_same_shape_sub_f16; நிறுத்து;
                    நிலை HTP_OP_MUL: compute_func = compute_same_shape_mul_f16; நிறுத்து;
                    நிலை HTP_OP_DIV: compute_func = compute_same_shape_div_f16; நிறுத்து;
                    இயல்பு: நிறுத்து;
                }
            }
            நிறுத்து;
        நிலை HTP_இருமம்_KERNEL_ROW_BCAST:
            worker_func = இருமம்_thread_vector_row_broadcast;
            எனில் (src0_type == HTP_TYPE_F32) {
                தேர்வு (octx->op) {
                    நிலை HTP_OP_ADD: compute_func = compute_row_bcast_add_f32; நிறுத்து;
                    நிலை HTP_OP_SUB: compute_func = compute_row_bcast_sub_f32; நிறுத்து;
                    நிலை HTP_OP_MUL: compute_func = compute_row_bcast_mul_f32; நிறுத்து;
                    நிலை HTP_OP_DIV: compute_func = compute_row_bcast_div_f32; நிறுத்து;
                    இயல்பு: நிறுத்து;
                }
            } இல்லையெனில் எனில் (src0_type == HTP_TYPE_F16) {
                தேர்வு (octx->op) {
                    நிலை HTP_OP_ADD: compute_func = compute_row_bcast_add_f16; நிறுத்து;
                    நிலை HTP_OP_SUB: compute_func = compute_row_bcast_sub_f16; நிறுத்து;
                    நிலை HTP_OP_MUL: compute_func = compute_row_bcast_mul_f16; நிறுத்து;
                    நிலை HTP_OP_DIV: compute_func = compute_row_bcast_div_f16; நிறுத்து;
                    இயல்பு: நிறுத்து;
                }
            }
            நிறுத்து;
        நிலை HTP_இருமம்_KERNEL_SCALAR:
            worker_func = இருமம்_thread_scalar;
            எனில் (src0_type == HTP_TYPE_F32) {
                தேர்வு (octx->op) {
                    நிலை HTP_OP_ADD: compute_func = compute_scalar_add_f32; நிறுத்து;
                    நிலை HTP_OP_SUB: compute_func = compute_scalar_sub_f32; நிறுத்து;
                    நிலை HTP_OP_MUL: compute_func = compute_scalar_mul_f32; நிறுத்து;
                    நிலை HTP_OP_DIV: compute_func = compute_scalar_div_f32; நிறுத்து;
                    இயல்பு: நிறுத்து;
                }
            } இல்லையெனில் எனில் (src0_type == HTP_TYPE_F16) {
                தேர்வு (octx->op) {
                    நிலை HTP_OP_ADD: compute_func = compute_scalar_add_f16; நிறுத்து;
                    நிலை HTP_OP_SUB: compute_func = compute_scalar_sub_f16; நிறுத்து;
                    நிலை HTP_OP_MUL: compute_func = compute_scalar_mul_f16; நிறுத்து;
                    நிலை HTP_OP_DIV: compute_func = compute_scalar_div_f16; நிறுத்து;
                    இயல்பு: நிறுத்து;
                }
            }
            நிறுத்து;
        நிலை HTP_இருமம்_KERNEL_COMPLEX:
            worker_func = இருமம்_thread_vector_complex;
            எனில் (src0_type == HTP_TYPE_F32) {
                தேர்வு (octx->op) {
                    நிலை HTP_OP_ADD: compute_func = compute_complex_add_f32; நிறுத்து;
                    நிலை HTP_OP_SUB: compute_func = compute_complex_sub_f32; நிறுத்து;
                    நிலை HTP_OP_MUL: compute_func = compute_complex_mul_f32; நிறுத்து;
                    நிலை HTP_OP_DIV: compute_func = compute_complex_div_f32; நிறுத்து;
                    இயல்பு: நிறுத்து;
                }
            } இல்லையெனில் எனில் (src0_type == HTP_TYPE_F16) {
                தேர்வு (octx->op) {
                    நிலை HTP_OP_ADD: compute_func = compute_complex_add_f16; நிறுத்து;
                    நிலை HTP_OP_SUB: compute_func = compute_complex_sub_f16; நிறுத்து;
                    நிலை HTP_OP_MUL: compute_func = compute_complex_mul_f16; நிறுத்து;
                    நிலை HTP_OP_DIV: compute_func = compute_complex_div_f16; நிறுத்து;
                    இயல்பு: நிறுத்து;
                }
            }
            நிறுத்து;
        நிலை HTP_இருமம்_KERNEL_REPEAT:
            worker_func = இருமம்_thread_element_repeat;
            எனில் (src0_type == HTP_TYPE_F32) {
                தேர்வு (octx->op) {
                    நிலை HTP_OP_ADD: compute_func = compute_repeat_add_f32; நிறுத்து;
                    நிலை HTP_OP_SUB: compute_func = compute_repeat_sub_f32; நிறுத்து;
                    நிலை HTP_OP_MUL: compute_func = compute_repeat_mul_f32; நிறுத்து;
                    நிலை HTP_OP_DIV: compute_func = compute_repeat_div_f32; நிறுத்து;
                    இயல்பு: நிறுத்து;
                }
            } இல்லையெனில் எனில் (src0_type == HTP_TYPE_F16) {
                தேர்வு (octx->op) {
                    நிலை HTP_OP_ADD: compute_func = compute_repeat_add_f16; நிறுத்து;
                    நிலை HTP_OP_SUB: compute_func = compute_repeat_sub_f16; நிறுத்து;
                    நிலை HTP_OP_MUL: compute_func = compute_repeat_mul_f16; நிறுத்து;
                    நிலை HTP_OP_DIV: compute_func = compute_repeat_div_f16; நிறுத்து;
                    இயல்பு: நிறுத்து;
                }
            }
            நிறுத்து;
        நிலை HTP_இருமம்_KERNEL_ADD_ID:
            எனில் (octx->op == HTP_OP_ADD_ID && src0_type == HTP_TYPE_F32) {
                worker_func = இருமம்_thread_add_id_f32;
                compute_func = (வெறுமை *) compute_add_id_f32;
            }
            நிறுத்து;
        இயல்பு: நிறுத்து;
    }

    எனில் (!worker_func || !compute_func) {
        திரும்பு HTP_STATUS_NO_SUPPORT;
    }

    bctx.compute = compute_func;

    எனில் (kparams->kernel_type == HTP_இருமம்_KERNEL_ROW_BCAST ||
        kparams->kernel_type == HTP_இருமம்_KERNEL_SCALAR ||
        kparams->kernel_type == HTP_இருமம்_KERNEL_REPEAT ||
        kparams->kernel_type == HTP_இருமம்_KERNEL_COMPLEX ||
        kparams->kernel_type == HTP_இருமம்_KERNEL_ADD_ID) {
        dma_queue_flush(dma_q);
    }

    work_queue_run(octx->ctx->work_queue, worker_func, &bctx, n_threads);

    திரும்பு HTP_STATUS_OK;
}

முழு op_இருமம்(கட்டமைப்பு htp_ops_context * octx) {
    மாறிலி uint32_t src0_type = octx->src[0]->type;
    எனில் ((src0_type == HTP_TYPE_F32) || (src0_type == HTP_TYPE_F16)) {
        திரும்பு execute_op_இருமம்(octx);
    }

    திரும்பு HTP_STATUS_NO_SUPPORT;
}
