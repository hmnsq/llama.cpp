#pragma clang diagnostic ignored "-Wunused-variable"
#pragma clang diagnostic ignored "-Wunused-function"
#pragma clang diagnostic ignored "-Wunused-but-set-variable"

#അന്തര്ഭാവ <HAP_farf.h>
#അന്തര്ഭാവ <HAP_perf.h>

#അന്തര്ഭാവ <math.h>
#അന്തര്ഭാവ <string.h>

#അന്തര്ഭാവ "dma-queue.h"
#അന്തര്ഭാവ "hvx-utils.h"

#define GGML_COMMON_DECL_C
#അന്തര്ഭാവ "ggml-common.h"
#അന്തര്ഭാവ "hex-common.h"
#അന്തര്ഭാവ "hex-profile.h"
#അന്തര്ഭാവ "binary-ops.h"
#അന്തര്ഭാവ "htp-ctx.h"
#അന്തര്ഭാവ "htp-ops.h"
#അന്തര്ഭാവ "htp-tensor.h"

#ifndef MIN
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#endif

// Context for binary operations
ഘടന htp_ദ്വിമാന_context {
    ഘടന htp_ops_context * octx;
    ഘടന htp_ദ്വിമാന_vtcm_layout vtcm_layout;
    uint8_t * vtcm_base;

    ഘടന fastdiv_മൂല്യങ്ങൾ src0_dim1_div; // ne01
    ഘടന fastdiv_മൂല്യങ്ങൾ src0_dim2_div; // ne02
    ഘടന fastdiv_മൂല്യങ്ങൾ src0_dim12_div;// ne03

    ഘടന fastdiv_മൂല്യങ്ങൾ src1_dim1_div; // ne11
    ഘടന fastdiv_മൂല്യങ്ങൾ src1_dim2_div; // ne12
    ഘടന fastdiv_മൂല്യങ്ങൾ src1_dim3_div; // ne13

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

    ശൂന്യം * compute;
};

#define htp_ദ്വിമാന_preamble                        \
    സ്ഥിരം ഘടന htp_tensor * src0 = octx->src[0]; \
    സ്ഥിരം ഘടന htp_tensor * src1 = octx->src[1]; \
    സ്ഥിരം ഘടന htp_tensor * dst  = octx->dst;    \
                                                   \
    സ്ഥിരം uint32_t ne00 = src0->ne[0];             \
    സ്ഥിരം uint32_t ne01 = src0->ne[1];             \
    സ്ഥിരം uint32_t ne02 = src0->ne[2];             \
    സ്ഥിരം uint32_t ne03 = src0->ne[3];             \
                                                   \
    സ്ഥിരം uint32_t ne10 = src1->ne[0];             \
    സ്ഥിരം uint32_t ne11 = src1->ne[1];             \
    സ്ഥിരം uint32_t ne12 = src1->ne[2];             \
    സ്ഥിരം uint32_t ne13 = src1->ne[3];             \
                                                   \
    സ്ഥിരം uint32_t nb01 = src0->nb[1];             \
    സ്ഥിരം uint32_t nb02 = src0->nb[2];             \
    സ്ഥിരം uint32_t nb03 = src0->nb[3];             \
                                                   \
    സ്ഥിരം uint32_t nb11 = src1->nb[1];             \
    സ്ഥിരം uint32_t nb12 = src1->nb[2];             \
    സ്ഥിരം uint32_t nb13 = src1->nb[3];             \
                                                   \
    സ്ഥിരം uint32_t nb1 = dst->nb[1];               \
    സ്ഥിരം uint32_t nb2 = dst->nb[2];               \
    സ്ഥിരം uint32_t nb3 = dst->nb[3];

സ്ഥിതം വരിയിൽ uint32_t calc_block_size(ഘടന htp_ദ്വിമാന_context * bctx, uint32_t ir, uint32_t end_row, uint32_t ne01, uint32_t ne02) {
    uint32_t i03, i02, i01, rem;
    i03 = fastdiv(ir, &bctx->src0_dim12_div);
    rem = ir - i03 * (ne02 * ne01);
    i02 = fastdiv(rem, &bctx->src0_dim1_div);
    i01 = rem - i02 * ne01;

    uint32_t rows_ഇടത് = end_row - ir;
    uint32_t block_limit = rows_ഇടത്;

    എങ്കിൽ (bctx->split_at_ne01) {
        block_limit = MIN(block_limit, ne01 - i01);
    }
    എങ്കിൽ (bctx->split_at_ne02) {
         uint32_t rows_in_plane = (ne02 * ne01) - rem;
         block_limit = MIN(block_limit, rows_in_plane);
    }

    തിരികെ MIN(bctx->block_max, block_limit);
}

// Out-of-line compute micro-kernels

തരനിർവചനം ശൂന്യം (*compute_scalar_t)(
    uint8_t * dst, സ്ഥിരം uint8_t * src0, സ്ഥിരം ശൂന്യം * s1_table,
    uint32_t cur_i11, uint32_t ne11, uint32_t n_rows,
    size_t dst_stride, size_t src0_stride, uint32_t ne00);

#define DEFINE_COMPUTE_SCALAR(NAME, TYPE, HVX_STMT)             \
സ്ഥിതം ശൂന്യം compute_scalar_##NAME(                              \
    uint8_t * dst, സ്ഥിരം uint8_t * src0, സ്ഥിരം ശൂന്യം * s1_table, \
    uint32_t cur_i11, uint32_t ne11, uint32_t n_rows,           \
    size_t dst_stride, size_t src0_stride, uint32_t ne00) {     \
    സ്ഥിരം TYPE * table = (സ്ഥിരം TYPE *) s1_table;               \
    വേണ്ടി (uint32_t r = 0; r < n_rows; r++) {                     \
        uint8_t * r_dst = dst + r * dst_stride;                 \
        സ്ഥിരം uint8_t * r_src0 = src0 + r * src0_stride;        \
        TYPE val = table[cur_i11];                              \
        HVX_STMT;                                               \
        എങ്കിൽ (ne11 > 1 && ++cur_i11 == ne11) {                    \
            cur_i11 = 0;                                        \
        }                                                       \
    }                                                           \
}

DEFINE_COMPUTE_SCALAR(add_f32, ദശാംശം,    hvx_add_scalar_f32_aa(r_dst, r_src0, val, ne00))
DEFINE_COMPUTE_SCALAR(add_f16, _Float16, hvx_add_scalar_f16_aa(r_dst, r_src0, val, ne00))
DEFINE_COMPUTE_SCALAR(sub_f32, ദശാംശം,    hvx_sub_scalar_f32_aa(r_dst, r_src0, val, ne00))
DEFINE_COMPUTE_SCALAR(sub_f16, _Float16, hvx_sub_scalar_f16_aa(r_dst, r_src0, val, ne00))
DEFINE_COMPUTE_SCALAR(mul_f32, ദശാംശം,    hvx_mul_scalar_f32_aa(r_dst, r_src0, val, ne00))
DEFINE_COMPUTE_SCALAR(mul_f16, _Float16, hvx_mul_scalar_f16_aa(r_dst, r_src0, val, ne00))
DEFINE_COMPUTE_SCALAR(div_f32, ദശാംശം,    hvx_mul_scalar_f32_aa(r_dst, r_src0, 1.0f / (val), ne00))
DEFINE_COMPUTE_SCALAR(div_f16, _Float16, hvx_div_scalar_f16_aa(r_dst, r_src0, val, ne00))

തരനിർവചനം ശൂന്യം (*compute_same_shape_t)(
    uint8_t * dst, സ്ഥിരം uint8_t * src0, സ്ഥിരം uint8_t * src1,
    uint32_t n_rows, size_t dst_stride, size_t src0_stride, size_t src1_stride, uint32_t ne00);

#define DEFINE_COMPUTE_SAME_SHAPE(NAME, HVX_FN)                                                  \
സ്ഥിതം ശൂന്യം compute_same_shape_##NAME(                                                           \
    uint8_t * dst, സ്ഥിരം uint8_t * src0, സ്ഥിരം uint8_t * src1,                                   \
    uint32_t n_rows, size_t dst_stride, size_t src0_stride, size_t src1_stride, uint32_t ne00) { \
    വേണ്ടി (uint32_t r = 0; r < n_rows; r++) {                                                      \
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

തരനിർവചനം ശൂന്യം (*compute_row_bcast_t)(
    uint8_t * dst, സ്ഥിരം uint8_t * src0, സ്ഥിരം uint8_t * src1,
    uint32_t n_rows, size_t dst_stride, size_t src0_stride, uint32_t ne00);

#define DEFINE_COMPUTE_ROW_BCAST(NAME, HVX_FN)                                               \
സ്ഥിതം ശൂന്യം compute_row_bcast_##NAME(                                                        \
    uint8_t * dst, സ്ഥിരം uint8_t * src0, സ്ഥിരം uint8_t * src1,                               \
    uint32_t n_rows, size_t dst_stride, size_t src0_stride, uint32_t ne00) {                 \
    വേണ്ടി (uint32_t r = 0; r < n_rows; r++) {                                                  \
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

തരനിർവചനം ശൂന്യം (*compute_complex_t)(
    uint8_t * dst, സ്ഥിരം uint8_t * src0, സ്ഥിരം uint8_t * src1_plane,
    uint32_t i01, uint32_t ne11, സ്ഥിരം ഘടന fastdiv_മൂല്യങ്ങൾ * div11, uint32_t src1_stride,
    uint32_t n_rows, size_t dst_stride, size_t src0_stride, uint32_t ne00);

#define DEFINE_COMPUTE_COMPLEX(NAME, HVX_FN)                                                 \
സ്ഥിതം ശൂന്യം compute_complex_##NAME(                                                          \
    uint8_t * dst, സ്ഥിരം uint8_t * src0, സ്ഥിരം uint8_t * src1_plane,                         \
    uint32_t i01, uint32_t ne11, സ്ഥിരം ഘടന fastdiv_മൂല്യങ്ങൾ * div11, uint32_t src1_stride,  \
    uint32_t n_rows, size_t dst_stride, size_t src0_stride, uint32_t ne00) {                 \
    വേണ്ടി (uint32_t r = 0; r < n_rows; r++) {                                                  \
        uint32_t i11 = fastmodulo(i01 + r, ne11, div11);                                     \
        സ്ഥിരം uint8_t * r_src1 = src1_plane + i11 * src1_stride;                             \
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

തരനിർവചനം ശൂന്യം (*compute_repeat_t)(
    uint8_t * dst, സ്ഥിരം uint8_t * src0, സ്ഥിരം uint8_t * src1_plane,
    uint32_t i01, uint32_t ne11, സ്ഥിരം ഘടന fastdiv_മൂല്യങ്ങൾ * div11, uint32_t src1_stride,
    uint32_t n_rows, size_t dst_stride, size_t src0_stride, uint32_t ne00, uint32_t ne10);

#define DEFINE_COMPUTE_REPEAT(NAME, TYPE, HVX_FN)                                            \
സ്ഥിതം ശൂന്യം compute_repeat_##NAME(                                                           \
    uint8_t * dst, സ്ഥിരം uint8_t * src0, സ്ഥിരം uint8_t * src1_plane,                         \
    uint32_t i01, uint32_t ne11, സ്ഥിരം ഘടന fastdiv_മൂല്യങ്ങൾ * div11, uint32_t src1_stride,  \
    uint32_t n_rows, size_t dst_stride, size_t src0_stride, uint32_t ne00, uint32_t ne10) {  \
    വേണ്ടി (uint32_t r = 0; r < n_rows; r++) {                                                  \
        uint32_t i11 = fastmodulo(i01 + r, ne11, div11);                                     \
        സ്ഥിരം uint8_t * r_src1_row = src1_plane + i11 * src1_stride;                         \
        uint8_t * r_dst = dst + r * dst_stride;                                              \
        സ്ഥിരം uint8_t * r_src0 = src0 + r * src0_stride;                                     \
        വേണ്ടി (uint32_t c = 0; c < ne00; c += ne10) {                                          \
            uint32_t len = MIN(ne10, ne00 - c);                                              \
            HVX_FN(r_dst + c * വലുപ്പം(TYPE), r_src0 + c * വലുപ്പം(TYPE), r_src1_row, len);    \
        }                                                                                    \
    }                                                                                        \
}

DEFINE_COMPUTE_REPEAT(add_f32, ദശാംശം,    hvx_add_f32_uuu)
DEFINE_COMPUTE_REPEAT(add_f16, _Float16, hvx_add_f16_uuu)
DEFINE_COMPUTE_REPEAT(sub_f32, ദശാംശം,    hvx_sub_f32_uuu)
DEFINE_COMPUTE_REPEAT(sub_f16, _Float16, hvx_sub_f16_uuu)
DEFINE_COMPUTE_REPEAT(mul_f32, ദശാംശം,    hvx_mul_f32_uuu)
DEFINE_COMPUTE_REPEAT(mul_f16, _Float16, hvx_mul_f16_uuu)
DEFINE_COMPUTE_REPEAT(div_f32, ദശാംശം,    hvx_div_f32_uuu)
DEFINE_COMPUTE_REPEAT(div_f16, _Float16, hvx_div_f16_uuu)

തരനിർവചനം ശൂന്യം (*compute_add_id_t)(
    uint8_t * dst, സ്ഥിരം uint8_t * src0, സ്ഥിരം uint8_t * src1_data, സ്ഥിരം int32_t * src2_ids,
    uint32_t i01, uint32_t i02, uint32_t ne01, uint32_t src1_stride,
    uint32_t n_rows, size_t dst_stride, size_t src0_stride, uint32_t ne00);

സ്ഥിതം ശൂന്യം compute_add_id_f32(
    uint8_t * dst, സ്ഥിരം uint8_t * src0, സ്ഥിരം uint8_t * src1_data, സ്ഥിരം int32_t * src2_ids,
    uint32_t i01, uint32_t i02, uint32_t ne01, uint32_t src1_stride,
    uint32_t n_rows, size_t dst_stride, size_t src0_stride, uint32_t ne00) {
    വേണ്ടി (uint32_t r = 0; r < n_rows; r++) {
        uint32_t r_i01 = i01 + r;
        സ്ഥിരം int32_t idx = src2_ids[i02 * ne01 + r_i01];
        എങ്കിൽ (idx < 0) {
            memcpy(dst + r * dst_stride, src0 + r * src0_stride, ne00 * വലുപ്പം(ദശാംശം));
            തുടരുക;
        }
        സ്ഥിരം uint8_t * r_src1 = src1_data + idx * src1_stride;
        സ്ഥിരം uint8_t * r_src0 = src0 + r * src0_stride;
        uint8_t * r_dst = dst + r * dst_stride;
        hvx_add_f32_aaa(r_dst, r_src0, r_src1, ne00);
    }
}

// 1. Scalar src1 in VTCM via DMA (ne10 == 1)
സ്ഥിതം ശൂന്യം ദ്വിമാന_thread_scalar(unsigned പൂർണ്ണസംഖ്യ nth, unsigned പൂർണ്ണസംഖ്യ ith, ശൂന്യം * data) {
    ഘടന htp_ദ്വിമാന_context * bctx = (ഘടന htp_ദ്വിമാന_context *) data;
    ഘടന htp_ops_context * octx = bctx->octx;
    htp_ദ്വിമാന_preamble;

    സ്ഥിരം uint32_t elem_size = (src0->type == HTP_TYPE_F32) ? വലുപ്പം(ദശാംശം) : വലുപ്പം(_Float16);
    സ്ഥിരം size_t plane_stride = (size_t) ne12 * ne11 * elem_size;
    സ്ഥിരം size_t row_stride   = (size_t) ne11 * elem_size;

    സ്ഥിരം uint32_t row_size_bytes = bctx->row_size_bytes;
    സ്ഥിരം uint32_t start_row = bctx->row_start + bctx->nrows_per_thread * ith;
    സ്ഥിരം uint32_t end_row   = MIN(start_row + bctx->nrows_per_thread, bctx->row_start + bctx->total_rows);
    എങ്കിൽ (start_row >= end_row) തിരികെ;

    FARF(മുകളിലെ_2, "binary-scalar: %d/%d (%u:%u) row-size %u (%u)",
         ith, nth, start_row, end_row, nb01, bctx->dst_row_size_aligned);

    സ്ഥിരം ഘടന htp_ദ്വിമാന_vtcm_layout * layout = &bctx->vtcm_layout;
    uint8_t * src0_spad_base = VTCM_LAYOUT_PTR(uint8_t, bctx->vtcm_base, layout->off_src0) + (ith * layout->src0_bytes_per_thread);
    uint8_t * dst_spad_base  = VTCM_LAYOUT_PTR(uint8_t, bctx->vtcm_base, layout->off_dst)  + (ith * layout->dst_bytes_per_thread);
    size_t src0_spad_half    = layout->src0_spad_half_size;
    size_t dst_spad_half     = layout->dst_spad_half_size;
    സ്ഥിരം ശൂന്യം * s1_table    = VTCM_LAYOUT_PTR(സ്ഥിരം ശൂന്യം, bctx->vtcm_base, layout->off_src1);

    dma_queue * dma_q = octx->ctx->dma[ith];
    uint32_t ir_prefetch = start_row;
    പൂർണ്ണസംഖ്യ spad_idx = 0;

    വേണ്ടി (പൂർണ്ണസംഖ്യ k = 0; k < 2 && ir_prefetch < end_row; k++) {
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

        dma_queue_തള്ളുക(dma_q, dma_make_data(dst_curr, d_spad), nb1, bctx->dst_row_size_aligned, row_size_bytes, 0);
        dma_queue_തള്ളുക(dma_q, dma_make_data(s0_spad, src0_curr), bctx->src0_row_size_aligned, nb01, row_size_bytes,
                                                             current_block_size);
        ir_prefetch += current_block_size;
        spad_idx ^= 1;
    }

    ഘടന htp_thread_trace * tr = &octx->ctx->trace[ith];
    compute_scalar_t compute = (compute_scalar_t) bctx->compute;

    വേണ്ടി (uint32_t ir = start_row; ir < end_row; ) {
        uint32_t current_block_size = calc_block_size(bctx, ir, end_row, ne01, ne02);

        uint8_t * d_spad  = (uint8_t *) dma_queue_എടുക്കുക(dma_q).src;
        uint8_t * s0_spad = (uint8_t *) dma_queue_എടുക്കുക(dma_q).dst;

        uint32_t i03, i02, i01, rem;
        i03 = fastdiv(ir, &bctx->src0_dim12_div);
        rem = ir - i03 * (ne02 * ne01);
        i02 = fastdiv(rem, &bctx->src0_dim1_div);
        i01 = rem - i02 * ne01;

        uint32_t i13 = fastmodulo(i03, ne13, &bctx->src1_dim3_div);
        uint32_t i12 = fastmodulo(i02, ne12, &bctx->src1_dim2_div);
        uint32_t i11 = fastmodulo(i01, ne11, &bctx->src1_dim1_div);
        സ്ഥിരം uint8_t * scalar_table = (സ്ഥിരം uint8_t *) s1_table +
            i13 * plane_stride + i12 * row_stride;

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);
        compute(d_spad, s0_spad, scalar_table, i11, ne11, current_block_size,
                bctx->dst_row_size_aligned, bctx->src0_row_size_aligned, ne00);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);

        dma_addr_t dst_curr = dst->data + i03 * nb3 + i02 * nb2 + i01 * nb1;
        dma_queue_തള്ളുക(dma_q, dma_make_data(dst_curr, d_spad), nb1, bctx->dst_row_size_aligned, row_size_bytes,
                                                                current_block_size);

        എങ്കിൽ (ir_prefetch < end_row) {
            uint32_t next_block_size = calc_block_size(bctx, ir_prefetch, end_row, ne01, ne02);
            uint32_t p03, p02, p01, prem;
            p03 = fastdiv(ir_prefetch, &bctx->src0_dim12_div);
            prem = ir_prefetch - p03 * (ne02 * ne01);
            p02 = fastdiv(prem, &bctx->src0_dim1_div);
            p01 = prem - p02 * ne01;
            dma_addr_t s0_next = src0->data + p03 * nb03 + p02 * nb02 + p01 * nb01;
            dma_queue_തള്ളുക(dma_q, dma_make_data(s0_spad, s0_next), bctx->src0_row_size_aligned, nb01, row_size_bytes,
                                                               next_block_size);
            ir_prefetch += next_block_size;
        }
        ir += current_block_size;
    }

    dma_queue_flush(dma_q);
}

// 2. Vector Same Shape (ne1x == ne0x) or Simple Broadcast
സ്ഥിതം ശൂന്യം ദ്വിമാന_thread_vector_same_shape(unsigned പൂർണ്ണസംഖ്യ nth, unsigned പൂർണ്ണസംഖ്യ ith, ശൂന്യം * data) {
    ഘടന htp_ദ്വിമാന_context * bctx = (ഘടന htp_ദ്വിമാന_context *) data;
    ഘടന htp_ops_context * octx = bctx->octx;
    htp_ദ്വിമാന_preamble;

    സ്ഥിരം uint32_t row_size_bytes = bctx->row_size_bytes;
    സ്ഥിരം uint32_t start_row = bctx->row_start + bctx->nrows_per_thread * ith;
    സ്ഥിരം uint32_t end_row   = MIN(start_row + bctx->nrows_per_thread, bctx->row_start + bctx->total_rows);
    എങ്കിൽ (start_row >= end_row) തിരികെ;

    FARF(മുകളിലെ_2, "binary-same-shape: %d/%d (%u:%u) row-size %u (%u)",
         ith, nth, start_row, end_row, nb01, bctx->dst_row_size_aligned);

    സ്ഥിരം ഘടന htp_ദ്വിമാന_vtcm_layout * layout = &bctx->vtcm_layout;
    uint8_t * src0_spad_base = VTCM_LAYOUT_PTR(uint8_t, bctx->vtcm_base, layout->off_src0) + (ith * layout->src0_bytes_per_thread);
    uint8_t * src1_spad_base = VTCM_LAYOUT_PTR(uint8_t, bctx->vtcm_base, layout->off_src1) + (ith * layout->src1_bytes_per_thread);
    uint8_t * dst_spad_base  = VTCM_LAYOUT_PTR(uint8_t, bctx->vtcm_base, layout->off_dst)  + (ith * layout->dst_bytes_per_thread);

    size_t src0_spad_half = layout->src0_spad_half_size;
    size_t src1_spad_half = layout->src1_spad_half_size;
    size_t dst_spad_half  = layout->dst_spad_half_size;

    dma_queue * dma_q = octx->ctx->dma[ith];
    uint32_t ir_prefetch = start_row;
    പൂർണ്ണസംഖ്യ spad_idx = 0;

    വേണ്ടി (പൂർണ്ണസംഖ്യ k = 0; k < 2 && ir_prefetch < end_row; k++) {
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

        dma_queue_തള്ളുക(dma_q, dma_make_data(dst_curr, d_spad), nb1, bctx->dst_row_size_aligned, row_size_bytes, 0);
        dma_queue_തള്ളുക(dma_q, dma_make_data(s0_spad, src0_curr), bctx->src0_row_size_aligned, nb01, row_size_bytes, current_block_size);
        dma_queue_തള്ളുക(dma_q, dma_make_data(s1_spad, src1_curr), bctx->src1_row_size_aligned, nb11, row_size_bytes, current_block_size);
        ir_prefetch += current_block_size;
        spad_idx ^= 1;
    }

    ഘടന htp_thread_trace * tr = &octx->ctx->trace[ith];
    compute_same_shape_t compute = (compute_same_shape_t) bctx->compute;

    വേണ്ടി (uint32_t ir = start_row; ir < end_row; ) {
        uint32_t current_block_size = calc_block_size(bctx, ir, end_row, ne01, ne02);
        uint8_t * d_spad  = (uint8_t *) dma_queue_എടുക്കുക(dma_q).src;
        uint8_t * s0_spad = (uint8_t *) dma_queue_എടുക്കുക(dma_q).dst;
        uint8_t * s1_spad = (uint8_t *) dma_queue_എടുക്കുക(dma_q).dst;

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
        dma_queue_തള്ളുക(dma_q, dma_make_data(dst_curr, d_spad), nb1, bctx->dst_row_size_aligned, row_size_bytes, current_block_size);

        എങ്കിൽ (ir_prefetch < end_row) {
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

            dma_queue_തള്ളുക(dma_q, dma_make_data(s0_spad, s0_next), bctx->src0_row_size_aligned, nb01, row_size_bytes, next_block_size);
            dma_queue_തള്ളുക(dma_q, dma_make_data(s1_spad, s1_next), bctx->src1_row_size_aligned, nb11, row_size_bytes, next_block_size);

            ir_prefetch += next_block_size;
        }
        ir += current_block_size;
    }

    dma_queue_flush(dma_q);
}

// 3. Row Broadcast (ne11 == 1, ne12 == 1, single row src1)
സ്ഥിതം ശൂന്യം ദ്വിമാന_thread_vector_row_broadcast(unsigned പൂർണ്ണസംഖ്യ nth, unsigned പൂർണ്ണസംഖ്യ ith, ശൂന്യം * data) {
    ഘടന htp_ദ്വിമാന_context * bctx = (ഘടന htp_ദ്വിമാന_context *) data;
    ഘടന htp_ops_context * octx = bctx->octx;
    htp_ദ്വിമാന_preamble;

    സ്ഥിരം uint32_t row_size_bytes = bctx->row_size_bytes;
    സ്ഥിരം uint32_t start_row  = bctx->row_start + bctx->nrows_per_thread * ith;
    സ്ഥിരം uint32_t end_row    = MIN(start_row + bctx->nrows_per_thread, bctx->row_start + bctx->total_rows);
    എങ്കിൽ (start_row >= end_row) തിരികെ;

    FARF(മുകളിലെ_2, "binary-row-bcast: %d/%d (%u:%u) row-size %u (%u)",
         ith, nth, start_row, end_row, nb01, bctx->dst_row_size_aligned);

    സ്ഥിരം ഘടന htp_ദ്വിമാന_vtcm_layout * layout = &bctx->vtcm_layout;
    uint8_t * src0_spad_base = VTCM_LAYOUT_PTR(uint8_t, bctx->vtcm_base, layout->off_src0) + (ith * layout->src0_bytes_per_thread);
    uint8_t * dst_spad_base  = VTCM_LAYOUT_PTR(uint8_t, bctx->vtcm_base, layout->off_dst)  + (ith * layout->dst_bytes_per_thread);

    size_t src0_spad_half = layout->src0_spad_half_size;
    size_t dst_spad_half  = layout->dst_spad_half_size;

    dma_queue * dma_q = octx->ctx->dma[ith];
    uint32_t ir_prefetch = start_row;
    പൂർണ്ണസംഖ്യ spad_idx = 0;

    ശൂന്യം * s1_ptr = VTCM_LAYOUT_PTR(ശൂന്യം, bctx->vtcm_base, layout->off_src1);

    വേണ്ടി (പൂർണ്ണസംഖ്യ k = 0; k < 2 && ir_prefetch < end_row; k++) {
        uint32_t current_block_size = calc_block_size(bctx, ir_prefetch, end_row, ne01, ne02);
        uint32_t i03 = fastdiv(ir_prefetch, &bctx->src0_dim12_div);
        uint32_t rem = ir_prefetch - i03 * (ne02 * ne01);
        uint32_t i02 = fastdiv(rem, &bctx->src0_dim1_div);
        uint32_t i01 = rem - i02 * ne01;

        dma_addr_t src0_curr = src0->data + i03 * nb03 + i02 * nb02 + i01 * nb01;
        dma_addr_t dst_curr  = dst->data  + i03 * nb3  + i02 * nb2  + i01 * nb1;

        uint8_t * s0_spad = src0_spad_base + spad_idx * src0_spad_half;
        uint8_t * d_spad  = dst_spad_base  + spad_idx * dst_spad_half;

        dma_queue_തള്ളുക(dma_q, dma_make_data(dst_curr, d_spad), nb1, bctx->dst_row_size_aligned, row_size_bytes, 0);
        dma_queue_തള്ളുക(dma_q, dma_make_data(s0_spad, src0_curr), bctx->src0_row_size_aligned, nb01, row_size_bytes, current_block_size);
        ir_prefetch += current_block_size;
        spad_idx ^= 1;
    }

    ഘടന htp_thread_trace * tr = &octx->ctx->trace[ith];
    compute_row_bcast_t compute = (compute_row_bcast_t) bctx->compute;

    വേണ്ടി (uint32_t ir = start_row; ir < end_row; ) {
        uint32_t current_block_size = calc_block_size(bctx, ir, end_row, ne01, ne02);
        uint8_t * d_spad  = (uint8_t *) dma_queue_എടുക്കുക(dma_q).src;
        uint8_t * s0_spad = (uint8_t *) dma_queue_എടുക്കുക(dma_q).dst;

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);
        compute(d_spad, s0_spad, (സ്ഥിരം uint8_t *)s1_ptr, current_block_size,
                bctx->dst_row_size_aligned, bctx->src0_row_size_aligned, ne00);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);

        uint32_t i03 = fastdiv(ir, &bctx->src0_dim12_div);
        uint32_t rem = ir - i03 * (ne02 * ne01);
        uint32_t i02 = fastdiv(rem, &bctx->src0_dim1_div);
        uint32_t i01 = rem - i02 * ne01;
        dma_addr_t dst_curr = dst->data + i03 * nb3 + i02 * nb2 + i01 * nb1;
        dma_queue_തള്ളുക(dma_q, dma_make_data(dst_curr, d_spad), nb1, bctx->dst_row_size_aligned, row_size_bytes, current_block_size);

        എങ്കിൽ (ir_prefetch < end_row) {
            uint32_t next_block_size = calc_block_size(bctx, ir_prefetch, end_row, ne01, ne02);
            uint32_t p03, p02, p01, prem;
            p03  = fastdiv(ir_prefetch, &bctx->src0_dim12_div);
            prem = ir_prefetch - p03 * (ne02 * ne01);
            p02  = fastdiv(prem, &bctx->src0_dim1_div);
            p01  = prem - p02 * ne01;
            dma_addr_t s0_next = src0->data + p03 * nb03 + p02 * nb02 + p01 * nb01;
            dma_queue_തള്ളുക(dma_q, dma_make_data(s0_spad, s0_next), bctx->src0_row_size_aligned, nb01, row_size_bytes, next_block_size);
            ir_prefetch += next_block_size;
        }
        ir += current_block_size;
    }

    dma_queue_flush(dma_q);
}

// 4. Vector Complex (ne10 == ne00, complex broadcast)
സ്ഥിതം ശൂന്യം ദ്വിമാന_thread_vector_complex(unsigned പൂർണ്ണസംഖ്യ nth, unsigned പൂർണ്ണസംഖ്യ ith, ശൂന്യം * data) {
    ഘടന htp_ദ്വിമാന_context * bctx = (ഘടന htp_ദ്വിമാന_context *) data;
    ഘടന htp_ops_context * octx = bctx->octx;
    htp_ദ്വിമാന_preamble;

    സ്ഥിരം uint32_t row_size_bytes = bctx->row_size_bytes;
    സ്ഥിരം uint32_t start_row  = bctx->row_start + bctx->nrows_per_thread * ith;
    സ്ഥിരം uint32_t end_row    = MIN(start_row + bctx->nrows_per_thread, bctx->row_start + bctx->total_rows);
    എങ്കിൽ (start_row >= end_row) തിരികെ;

    FARF(മുകളിലെ_2, "binary-complex: %d/%d (%u:%u) row-size %u (%u)",
         ith, nth, start_row, end_row, nb01, bctx->dst_row_size_aligned);

    സ്ഥിരം ഘടന htp_ദ്വിമാന_vtcm_layout * layout = &bctx->vtcm_layout;
    uint8_t * src0_spad_base = VTCM_LAYOUT_PTR(uint8_t, bctx->vtcm_base, layout->off_src0) + (ith * layout->src0_bytes_per_thread);
    uint8_t * dst_spad_base  = VTCM_LAYOUT_PTR(uint8_t, bctx->vtcm_base, layout->off_dst)  + (ith * layout->dst_bytes_per_thread);
    size_t src0_spad_half    = layout->src0_spad_half_size;
    size_t dst_spad_half     = layout->dst_spad_half_size;
    സ്ഥിരം uint8_t * vtcm_src1 = VTCM_LAYOUT_PTR(സ്ഥിരം uint8_t, bctx->vtcm_base, layout->off_src1);

    സ്ഥിരം uint32_t elem_size = (src0->type == HTP_TYPE_F32) ? വലുപ്പം(ദശാംശം) : വലുപ്പം(_Float16);
    സ്ഥിരം size_t vtcm_row_stride   = (size_t) ne10 * elem_size;
    സ്ഥിരം size_t vtcm_plane_stride = (size_t) ne11 * vtcm_row_stride;
    സ്ഥിരം size_t vtcm_vol_stride   = (size_t) ne12 * vtcm_plane_stride;

    dma_queue * dma_q = octx->ctx->dma[ith];
    uint32_t ir_prefetch = start_row;
    പൂർണ്ണസംഖ്യ spad_idx = 0;

    വേണ്ടി (പൂർണ്ണസംഖ്യ k = 0; k < 2 && ir_prefetch < end_row; k++) {
        uint32_t current_block_size = calc_block_size(bctx, ir_prefetch, end_row, ne01, ne02);
        uint32_t i03 = fastdiv(ir_prefetch, &bctx->src0_dim12_div);
        uint32_t rem = ir_prefetch - i03 * (ne02 * ne01);
        uint32_t i02 = fastdiv(rem, &bctx->src0_dim1_div);
        uint32_t i01 = rem - i02 * ne01;

        dma_addr_t src0_curr = src0->data + i03 * nb03 + i02 * nb02 + i01 * nb01;
        dma_addr_t dst_curr  = dst->data  + i03 * nb3  + i02 * nb2  + i01 * nb1;

        uint8_t * s0_spad = src0_spad_base + spad_idx * src0_spad_half;
        uint8_t * d_spad  = dst_spad_base  + spad_idx * dst_spad_half;

        dma_queue_തള്ളുക(dma_q, dma_make_data(dst_curr, d_spad), nb1, bctx->dst_row_size_aligned, row_size_bytes, 0);
        dma_queue_തള്ളുക(dma_q, dma_make_data(s0_spad, src0_curr), bctx->src0_row_size_aligned, nb01, row_size_bytes, current_block_size);
        ir_prefetch += current_block_size;
        spad_idx ^= 1;
    }

    ഘടന htp_thread_trace * tr = &octx->ctx->trace[ith];
    compute_complex_t compute = (compute_complex_t) bctx->compute;

    വേണ്ടി (uint32_t ir = start_row; ir < end_row; ) {
        uint32_t current_block_size = calc_block_size(bctx, ir, end_row, ne01, ne02);
        uint8_t * d_spad = (uint8_t *) dma_queue_എടുക്കുക(dma_q).src;
        uint8_t * s0_spad = (uint8_t *) dma_queue_എടുക്കുക(dma_q).dst;

        uint32_t i03 = fastdiv(ir, &bctx->src0_dim12_div);
        uint32_t rem = ir - i03 * (ne02 * ne01);
        uint32_t i02 = fastdiv(rem, &bctx->src0_dim1_div);
        uint32_t i01 = rem - i02 * ne01;

        uint32_t i13 = fastmodulo(i03, ne13, &bctx->src1_dim3_div);
        uint32_t i12 = fastmodulo(i02, ne12, &bctx->src1_dim2_div);
        സ്ഥിരം uint8_t * src1_plane = vtcm_src1 + i13 * vtcm_vol_stride + i12 * vtcm_plane_stride;

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);
        compute(d_spad, s0_spad, src1_plane, i01, ne11, &bctx->src1_dim1_div, vtcm_row_stride,
                current_block_size, bctx->dst_row_size_aligned, bctx->src0_row_size_aligned, ne00);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);

        dma_addr_t dst_curr = dst->data + i03 * nb3 + i02 * nb2 + i01 * nb1;
        dma_queue_തള്ളുക(dma_q, dma_make_data(dst_curr, d_spad), nb1, bctx->dst_row_size_aligned, row_size_bytes, current_block_size);

        എങ്കിൽ (ir_prefetch < end_row) {
            uint32_t next_block_size = calc_block_size(bctx, ir_prefetch, end_row, ne01, ne02);
            uint32_t p03, p02, p01, prem;
            p03  = fastdiv(ir_prefetch, &bctx->src0_dim12_div);
            prem = ir_prefetch - p03 * (ne02 * ne01);
            p02  = fastdiv(prem, &bctx->src0_dim1_div);
            p01  = prem - p02 * ne01;
            dma_addr_t s0_next = src0->data + p03 * nb03 + p02 * nb02 + p01 * nb01;
            dma_queue_തള്ളുക(dma_q, dma_make_data(s0_spad, s0_next), bctx->src0_row_size_aligned, nb01, row_size_bytes, next_block_size);
            ir_prefetch += next_block_size;
        }
        ir += current_block_size;
    }

    dma_queue_flush(dma_q);
}

// 5. Element Repeat (ne10 != ne00)
സ്ഥിതം ശൂന്യം ദ്വിമാന_thread_element_repeat(unsigned പൂർണ്ണസംഖ്യ nth, unsigned പൂർണ്ണസംഖ്യ ith, ശൂന്യം * data) {
    ഘടന htp_ദ്വിമാന_context * bctx = (ഘടന htp_ദ്വിമാന_context *) data;
    ഘടന htp_ops_context * octx = bctx->octx;
    htp_ദ്വിമാന_preamble;

    സ്ഥിരം uint32_t row_size_bytes = bctx->row_size_bytes;
    സ്ഥിരം uint32_t start_row  = bctx->row_start + bctx->nrows_per_thread * ith;
    സ്ഥിരം uint32_t end_row    = MIN(start_row + bctx->nrows_per_thread, bctx->row_start + bctx->total_rows);
    എങ്കിൽ (start_row >= end_row) തിരികെ;

    സ്ഥിരം ഘടന htp_ദ്വിമാന_vtcm_layout * layout = &bctx->vtcm_layout;
    uint8_t * src0_spad_base = VTCM_LAYOUT_PTR(uint8_t, bctx->vtcm_base, layout->off_src0) + (ith * layout->src0_bytes_per_thread);
    uint8_t * dst_spad_base  = VTCM_LAYOUT_PTR(uint8_t, bctx->vtcm_base, layout->off_dst)  + (ith * layout->dst_bytes_per_thread);
    size_t src0_spad_half    = layout->src0_spad_half_size;
    size_t dst_spad_half     = layout->dst_spad_half_size;
    സ്ഥിരം uint8_t * vtcm_src1 = VTCM_LAYOUT_PTR(സ്ഥിരം uint8_t, bctx->vtcm_base, layout->off_src1);

    സ്ഥിരം uint32_t elem_size = (src0->type == HTP_TYPE_F32) ? വലുപ്പം(ദശാംശം) : വലുപ്പം(_Float16);
    സ്ഥിരം size_t vtcm_row_stride   = (size_t) ne10 * elem_size;
    സ്ഥിരം size_t vtcm_plane_stride = (size_t) ne11 * vtcm_row_stride;
    സ്ഥിരം size_t vtcm_vol_stride   = (size_t) ne12 * vtcm_plane_stride;

    FARF(മുകളിലെ_2, "binary-repeat: %d/%d (%u:%u) row-size %u (%u)",
         ith, nth, start_row, end_row, nb01, bctx->dst_row_size_aligned);

    dma_queue * dma_q = octx->ctx->dma[ith];
    uint32_t ir_prefetch = start_row;
    പൂർണ്ണസംഖ്യ spad_idx = 0;

    വേണ്ടി (പൂർണ്ണസംഖ്യ k = 0; k < 2 && ir_prefetch < end_row; k++) {
        uint32_t current_block_size = calc_block_size(bctx, ir_prefetch, end_row, ne01, ne02);
        uint32_t i03 = fastdiv(ir_prefetch, &bctx->src0_dim12_div);
        uint32_t rem = ir_prefetch - i03 * (ne02 * ne01);
        uint32_t i02 = fastdiv(rem, &bctx->src0_dim1_div);
        uint32_t i01 = rem - i02 * ne01;

        dma_addr_t src0_curr = src0->data + i03 * nb03 + i02 * nb02 + i01 * nb01;
        dma_addr_t dst_curr  = dst->data  + i03 * nb3  + i02 * nb2  + i01 * nb1;

        uint8_t * s0_spad = src0_spad_base + spad_idx * src0_spad_half;
        uint8_t * d_spad  = dst_spad_base  + spad_idx * dst_spad_half;

        dma_queue_തള്ളുക(dma_q, dma_make_data(dst_curr, d_spad), nb1, bctx->dst_row_size_aligned, row_size_bytes, 0);
        dma_queue_തള്ളുക(dma_q, dma_make_data(s0_spad, src0_curr), bctx->src0_row_size_aligned, nb01, row_size_bytes, current_block_size);
        ir_prefetch += current_block_size;
        spad_idx ^= 1;
    }

    ഘടന htp_thread_trace * tr = &octx->ctx->trace[ith];
    compute_repeat_t compute = (compute_repeat_t) bctx->compute;

    വേണ്ടി (uint32_t ir = start_row; ir < end_row; ) {
        uint32_t current_block_size = calc_block_size(bctx, ir, end_row, ne01, ne02);
        uint8_t * d_spad  = (uint8_t *) dma_queue_എടുക്കുക(dma_q).src;
        uint8_t * s0_spad = (uint8_t *) dma_queue_എടുക്കുക(dma_q).dst;

        uint32_t i03 = fastdiv(ir, &bctx->src0_dim12_div);
        uint32_t rem = ir - i03 * (ne02 * ne01);
        uint32_t i02 = fastdiv(rem, &bctx->src0_dim1_div);
        uint32_t i01 = rem - i02 * ne01;

        uint32_t i13 = fastmodulo(i03, ne13, &bctx->src1_dim3_div);
        uint32_t i12 = fastmodulo(i02, ne12, &bctx->src1_dim2_div);
        സ്ഥിരം uint8_t * src1_plane = vtcm_src1 + i13 * vtcm_vol_stride + i12 * vtcm_plane_stride;

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);
        compute(d_spad, s0_spad, src1_plane, i01, ne11, &bctx->src1_dim1_div, vtcm_row_stride,
                current_block_size, bctx->dst_row_size_aligned, bctx->src0_row_size_aligned, ne00, ne10);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);

        dma_addr_t dst_curr = dst->data + i03 * nb3 + i02 * nb2 + i01 * nb1;
        dma_queue_തള്ളുക(dma_q, dma_make_data(dst_curr, d_spad), nb1, bctx->dst_row_size_aligned, row_size_bytes, current_block_size);

        എങ്കിൽ (ir_prefetch < end_row) {
            uint32_t next_block_size = calc_block_size(bctx, ir_prefetch, end_row, ne01, ne02);
            uint32_t p03, p02, p01, prem;
            p03  = fastdiv(ir_prefetch, &bctx->src0_dim12_div);
            prem = ir_prefetch - p03 * (ne02 * ne01);
            p02  = fastdiv(prem, &bctx->src0_dim1_div);
            p01  = prem - p02 * ne01;
            dma_addr_t s0_next = src0->data + p03 * nb03 + p02 * nb02 + p01 * nb01;
            dma_queue_തള്ളുക(dma_q, dma_make_data(s0_spad, s0_next), bctx->src0_row_size_aligned, nb01, row_size_bytes, next_block_size);
            ir_prefetch += next_block_size;
        }
        ir += current_block_size;
    }

    dma_queue_flush(dma_q);
}

// 6. ADD_ID (src1 gathered via src2 indices)
സ്ഥിതം ശൂന്യം ദ്വിമാന_thread_add_id_f32(unsigned പൂർണ്ണസംഖ്യ nth, unsigned പൂർണ്ണസംഖ്യ ith, ശൂന്യം * data) {
    ഘടന htp_ദ്വിമാന_context * bctx = (ഘടന htp_ദ്വിമാന_context *) data;
    ഘടന htp_ops_context * octx = bctx->octx;

    സ്ഥിരം ഘടന htp_tensor * src0 = octx->src[0];
    സ്ഥിരം ഘടന htp_tensor * src1 = octx->src[1];
    സ്ഥിരം ഘടന htp_tensor * dst  = octx->dst;

    സ്ഥിരം uint32_t ne00 = src0->ne[0];
    സ്ഥിരം uint32_t ne01 = src0->ne[1];
    സ്ഥിരം uint32_t ne02 = src0->ne[2];
    സ്ഥിരം uint32_t ne03 = src0->ne[3];

    സ്ഥിരം uint32_t nb01 = src0->nb[1];
    സ്ഥിരം uint32_t nb02 = src0->nb[2];
    സ്ഥിരം uint32_t nb03 = src0->nb[3];
    സ്ഥിരം uint32_t src1_stride = bctx->src1_row_size_aligned;

    സ്ഥിരം uint32_t nb1 = dst->nb[1];
    സ്ഥിരം uint32_t nb2 = dst->nb[2];
    സ്ഥിരം uint32_t nb3 = dst->nb[3];

    സ്ഥിരം uint32_t row_size_bytes = bctx->row_size_bytes;
    സ്ഥിരം uint32_t start_row = bctx->row_start + bctx->nrows_per_thread * ith;
    സ്ഥിരം uint32_t end_row   = MIN(start_row + bctx->nrows_per_thread, bctx->row_start + bctx->total_rows);
    എങ്കിൽ (start_row >= end_row) തിരികെ;

    സ്ഥിരം ഘടന htp_ദ്വിമാന_vtcm_layout * layout = &bctx->vtcm_layout;
    uint8_t * src0_spad_base = VTCM_LAYOUT_PTR(uint8_t, bctx->vtcm_base, layout->off_src0) + (ith * layout->src0_bytes_per_thread);
    uint8_t * dst_spad_base  = VTCM_LAYOUT_PTR(uint8_t, bctx->vtcm_base, layout->off_dst)  + (ith * layout->dst_bytes_per_thread);
    സ്ഥിരം uint8_t * vtcm_src1 = VTCM_LAYOUT_PTR(സ്ഥിരം uint8_t, bctx->vtcm_base, layout->off_src1);
    സ്ഥിരം int32_t * vtcm_src2 = VTCM_LAYOUT_PTR(സ്ഥിരം int32_t, bctx->vtcm_base, layout->off_src2);
    size_t src0_spad_half    = layout->src0_spad_half_size;
    size_t dst_spad_half     = layout->dst_spad_half_size;

    dma_queue * dma_q = octx->ctx->dma[ith];
    uint32_t ir_prefetch = start_row;
    പൂർണ്ണസംഖ്യ spad_idx = 0;

    വേണ്ടി (പൂർണ്ണസംഖ്യ k = 0; k < 2 && ir_prefetch < end_row; k++) {
        uint32_t current_block_size = calc_block_size(bctx, ir_prefetch, end_row, ne01, ne02);
        uint32_t i03 = fastdiv(ir_prefetch, &bctx->src0_dim12_div);
        uint32_t rem = ir_prefetch - i03 * (ne02 * ne01);
        uint32_t i02 = fastdiv(rem, &bctx->src0_dim1_div);
        uint32_t i01 = rem - i02 * ne01;

        dma_addr_t src0_curr = src0->data + i03 * nb03 + i02 * nb02 + i01 * nb01;
        dma_addr_t dst_curr  = dst->data  + i03 * nb3  + i02 * nb2  + i01 * nb1;

        uint8_t * s0_spad = src0_spad_base + spad_idx * src0_spad_half;
        uint8_t * d_spad  = dst_spad_base  + spad_idx * dst_spad_half;

        dma_queue_തള്ളുക(dma_q, dma_make_data(dst_curr, d_spad), nb1, bctx->dst_row_size_aligned, row_size_bytes, 0);
        dma_queue_തള്ളുക(dma_q, dma_make_data(s0_spad, src0_curr), bctx->src0_row_size_aligned, nb01, row_size_bytes, current_block_size);
        ir_prefetch += current_block_size;
        spad_idx ^= 1;
    }

    ഘടന htp_thread_trace * tr = &octx->ctx->trace[ith];

    വേണ്ടി (uint32_t ir = start_row; ir < end_row; ) {
        uint32_t current_block_size = calc_block_size(bctx, ir, end_row, ne01, ne02);
        uint8_t * d_spad = (uint8_t *) dma_queue_എടുക്കുക(dma_q).src;
        uint8_t * s0_spad = (uint8_t *) dma_queue_എടുക്കുക(dma_q).dst;

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
        dma_queue_തള്ളുക(dma_q, dma_make_data(dst_curr, d_spad), nb1, bctx->dst_row_size_aligned, row_size_bytes, current_block_size);

        എങ്കിൽ (ir_prefetch < end_row) {
             uint32_t next_block_size = calc_block_size(bctx, ir_prefetch, end_row, ne01, ne02);
             uint32_t p03, p02, p01, prem;
             p03  = fastdiv(ir_prefetch, &bctx->src0_dim12_div);
             prem = ir_prefetch - p03 * (ne02 * ne01);
             p02  = fastdiv(prem, &bctx->src0_dim1_div);
             p01  = prem - p02 * ne01;
             dma_addr_t s0_next = src0->data + p03 * nb03 + p02 * nb02 + p01 * nb01;
             dma_queue_തള്ളുക(dma_q, dma_make_data(s0_spad, s0_next), bctx->src0_row_size_aligned, nb01, row_size_bytes, next_block_size);
             ir_prefetch += next_block_size;
        }
        ir += current_block_size;
    }

    dma_queue_flush(dma_q);
}

സ്ഥിതം വരിയിൽ ശൂന്യം hvx_div_scalar_f32(uint8_t * restrict dst, സ്ഥിരം uint8_t * restrict src, സ്ഥിരം ദശാംശം val, സ്ഥിരം uint32_t അക്കം_elems) {
    hvx_mul_scalar_f32(dst, src, 1.0f / val, അക്കം_elems);
}

സ്ഥിതം വരിയിൽ ശൂന്യം hvx_div_scalar_f16(uint8_t * restrict dst, സ്ഥിരം uint8_t * restrict src, സ്ഥിരം _Float16 val, സ്ഥിരം uint32_t അക്കം_elems) {
    hvx_div_scalar_f16_aa(dst, src, val, അക്കം_elems);
}

തരനിർവചനം ശൂന്യം (*compute_ദ്വിമാന_chunked_t)(
    uint8_t * restrict dst,
    സ്ഥിരം uint8_t * restrict src0,
    സ്ഥിരം uint8_t * restrict src1,
    സ്ഥിരം uint32_t അക്കം_elems
);

തരനിർവചനം ശൂന്യം (*compute_ദ്വിമാന_scalar_chunked_f32_t)(
    uint8_t * restrict dst,
    സ്ഥിരം uint8_t * restrict src0,
    സ്ഥിരം ദശാംശം val,
    സ്ഥിരം uint32_t അക്കം_elems
);

തരനിർവചനം ശൂന്യം (*compute_ദ്വിമാന_scalar_chunked_f16_t)(
    uint8_t * restrict dst,
    സ്ഥിരം uint8_t * restrict src0,
    സ്ഥിരം _Float16 val,
    സ്ഥിരം uint32_t അക്കം_elems
);

ഘടന ദ്വിമാന_chunked_context {
    ഘടന htp_ops_context * octx;
    ഘടന htp_ദ്വിമാന_vtcm_layout vtcm_layout;
    uint8_t *                vtcm_base;
    uint32_t                 elem_start;
    uint32_t                 nelem;
    uint32_t                 chunk_size;
    uint32_t                 chunks_per_thread;
    uint32_t                 total_chunks;
    bool                     is_scalar;
    ദശാംശം                    scalar_f32;
    _Float16                 scalar_f16;
    compute_ദ്വിമാന_chunked_t compute;
    compute_ദ്വിമാന_scalar_chunked_f32_t compute_scalar_f32;
    compute_ദ്വിമാന_scalar_chunked_f16_t compute_scalar_f16;
};

സ്ഥിതം ശൂന്യം ദ്വിമാന_thread_chunked(unsigned പൂർണ്ണസംഖ്യ nth, unsigned പൂർണ്ണസംഖ്യ ith, ശൂന്യം * data) {
    (ശൂന്യം) nth;
    ഘടന ദ്വിമാന_chunked_context * ctx = (ഘടന ദ്വിമാന_chunked_context *) data;
    ഘടന htp_ops_context * octx = ctx->octx;
    സ്ഥിരം ഘടന htp_tensor * src0 = octx->src[0];
    സ്ഥിരം ഘടന htp_tensor * src1 = octx->src[1];
    സ്ഥിരം ഘടന htp_tensor * dst  = octx->dst;

    സ്ഥിരം uint32_t start_chunk = ctx->chunks_per_thread * ith;
    സ്ഥിരം uint32_t end_chunk   = MIN(start_chunk + ctx->chunks_per_thread, ctx->total_chunks);
    എങ്കിൽ (start_chunk >= end_chunk) {
        തിരികെ;
    }

    സ്ഥിരം uint32_t src0_type   = src0->type;
    സ്ഥിരം size_t   elem_size   = (src0_type == HTP_TYPE_F32) ? വലുപ്പം(ദശാംശം) : വലുപ്പം(_Float16);
    സ്ഥിരം uint32_t chunk_size  = ctx->chunk_size;
    സ്ഥിരം size_t   chunk_bytes = ctx->vtcm_layout.src0_spad_half_size;

    FARF(മുകളിലെ_2, "binary-chunked: %d/%d (%u:%u) chunks %u elems %u",
         ith, nth, start_chunk, end_chunk, ctx->total_chunks, ctx->nelem);

    സ്ഥിരം ഘടന htp_ദ്വിമാന_vtcm_layout * layout = &ctx->vtcm_layout;
    uint8_t * src0_spad_base = VTCM_LAYOUT_PTR(uint8_t, ctx->vtcm_base, layout->off_src0) + (ith * layout->src0_bytes_per_thread);
    uint8_t * src1_spad_base = ctx->is_scalar ? ശൂന്യ : (VTCM_LAYOUT_PTR(uint8_t, ctx->vtcm_base, layout->off_src1) + (ith * layout->src1_bytes_per_thread));
    uint8_t * dst_spad_base  = VTCM_LAYOUT_PTR(uint8_t, ctx->vtcm_base, layout->off_dst)  + (ith * layout->dst_bytes_per_thread);

    dma_queue * dma_q = octx->ctx->dma[ith];
    uint32_t prefetch_chunk = start_chunk;
    പൂർണ്ണസംഖ്യ spad_idx = 0;

    വേണ്ടി (പൂർണ്ണസംഖ്യ k = 0; k < 2 && prefetch_chunk < end_chunk; k++) {
        സ്ഥിരം uint32_t c_start   = ctx->elem_start + prefetch_chunk * chunk_size;
        സ്ഥിരം uint32_t c_end     = MIN(c_start + chunk_size, ctx->elem_start + ctx->nelem);
        സ്ഥിരം uint32_t cur_elems = c_end - c_start;
        സ്ഥിരം uint32_t cur_bytes = cur_elems * elem_size;

        dma_addr_t s0_curr = src0->data + (size_t) c_start * elem_size;
        dma_addr_t d_curr  = dst->data  + (size_t) c_start * elem_size;

        uint8_t * s0_spad = src0_spad_base + spad_idx * chunk_bytes;
        uint8_t * d_spad  = dst_spad_base  + spad_idx * chunk_bytes;

        dma_queue_തള്ളുക(dma_q, dma_make_data(d_curr, d_spad), chunk_bytes, chunk_bytes, cur_bytes, 0);
        dma_queue_തള്ളുക(dma_q, dma_make_data(s0_spad, s0_curr), chunk_bytes, chunk_bytes, cur_bytes, 1);
        എങ്കിൽ (!ctx->is_scalar) {
            dma_addr_t s1_curr = src1->data + (size_t) c_start * elem_size;
            uint8_t * s1_spad = src1_spad_base + spad_idx * chunk_bytes;
            dma_queue_തള്ളുക(dma_q, dma_make_data(s1_spad, s1_curr), chunk_bytes, chunk_bytes, cur_bytes, 1);
        }

        prefetch_chunk++;
        spad_idx ^= 1;
    }

    ഘടന htp_thread_trace * tr = &octx->ctx->trace[ith];

    വേണ്ടി (uint32_t c = start_chunk; c < end_chunk; c++) {
        സ്ഥിരം uint32_t c_start   = ctx->elem_start + c * chunk_size;
        സ്ഥിരം uint32_t c_end     = MIN(c_start + chunk_size, ctx->elem_start + ctx->nelem);
        സ്ഥിരം uint32_t cur_elems = c_end - c_start;
        സ്ഥിരം uint32_t cur_bytes = cur_elems * elem_size;

        uint8_t * d_spad  = (uint8_t *) dma_queue_എടുക്കുക(dma_q).src;
        uint8_t * s0_spad = (uint8_t *) dma_queue_എടുക്കുക(dma_q).dst;
        uint8_t * s1_spad = ctx->is_scalar ? ശൂന്യ : (uint8_t *) dma_queue_എടുക്കുക(dma_q).dst;

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) c);
        എങ്കിൽ (ctx->is_scalar) {
            എങ്കിൽ (src0_type == HTP_TYPE_F32) {
                ctx->compute_scalar_f32(d_spad, s0_spad, ctx->scalar_f32, cur_elems);
            } അല്ലെങ്കിൽ {
                ctx->compute_scalar_f16(d_spad, s0_spad, ctx->scalar_f16, cur_elems);
            }
        } അല്ലെങ്കിൽ {
            ctx->compute(d_spad, s0_spad, s1_spad, cur_elems);
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) c);

        dma_addr_t dst_curr = dst->data + (size_t) c_start * elem_size;
        dma_queue_തള്ളുക(dma_q, dma_make_data(dst_curr, d_spad), chunk_bytes, chunk_bytes, cur_bytes, 1);

        എങ്കിൽ (prefetch_chunk < end_chunk) {
            സ്ഥിരം uint32_t pc_start  = ctx->elem_start + prefetch_chunk * chunk_size;
            സ്ഥിരം uint32_t pc_end    = MIN(pc_start + chunk_size, ctx->elem_start + ctx->nelem);
            സ്ഥിരം uint32_t p_elems   = pc_end - pc_start;
            സ്ഥിരം uint32_t p_bytes   = p_elems * elem_size;

            dma_addr_t s0_next = src0->data + (size_t) pc_start * elem_size;
            dma_queue_തള്ളുക(dma_q, dma_make_data(s0_spad, s0_next), chunk_bytes, chunk_bytes, p_bytes, 1);
            എങ്കിൽ (!ctx->is_scalar) {
                dma_addr_t s1_next = src1->data + (size_t) pc_start * elem_size;
                dma_queue_തള്ളുക(dma_q, dma_make_data(s1_spad, s1_next), chunk_bytes, chunk_bytes, p_bytes, 1);
            }

            prefetch_chunk++;
        }
    }

    dma_queue_flush(dma_q);
}

സ്ഥിതം പൂർണ്ണസംഖ്യ execute_op_ദ്വിമാന(ഘടന htp_ops_context * octx) {
    സ്ഥിരം ഘടന htp_tensor * src0 = octx->src[0];
    സ്ഥിരം ഘടന htp_tensor * src1 = octx->src[1];
    സ്ഥിരം ഘടന htp_tensor * dst  = octx->dst;
    സ്ഥിരം ഘടന htp_ദ്വിമാന_kernel_params * kparams = (സ്ഥിരം ഘടന htp_ദ്വിമാന_kernel_params *) octx->kernel_params;

    സ്ഥിരം uint32_t src0_nrows = src0->ne[1] * src0->ne[2] * src0->ne[3];

    // Use packed row sizes for VTCM allocation and alignment
    സ്ഥിരം uint32_t src0_type = octx->src[0]->type;
    സ്ഥിരം size_t elem_size = (src0_type == HTP_TYPE_F32) ? വലുപ്പം(ദശാംശം) : വലുപ്പം(_Float16);
    സ്ഥിരം size_t src0_row_size = src0->ne[0] * elem_size;
    സ്ഥിരം size_t src1_row_size = src1->ne[0] * elem_size;
    സ്ഥിരം size_t dst_row_size  = dst->ne[0]  * elem_size;

    എങ്കിൽ (kparams->kernel_type == HTP_ദ്വിമാന_KERNEL_CHUNKED) {
        സ്ഥിരം bool is_scalar = kparams->is_scalar || (src1->ne[0] == 1 && src1->ne[1] == 1 && src1->ne[2] == 1 && src1->ne[3] == 1);

        compute_ദ്വിമാന_chunked_t compute = ശൂന്യ;
        compute_ദ്വിമാന_scalar_chunked_f32_t compute_scalar_f32 = ശൂന്യ;
        compute_ദ്വിമാന_scalar_chunked_f16_t compute_scalar_f16 = ശൂന്യ;

        എങ്കിൽ (is_scalar) {
            എങ്കിൽ (src0_type == HTP_TYPE_F32) {
                തിരഞ്ഞെടുപ്പ് (octx->op) {
                    അവസ്ഥ HTP_OP_ADD: compute_scalar_f32 = hvx_add_scalar_f32; നിർത്തുക;
                    അവസ്ഥ HTP_OP_SUB: compute_scalar_f32 = hvx_sub_scalar_f32; നിർത്തുക;
                    അവസ്ഥ HTP_OP_MUL: compute_scalar_f32 = hvx_mul_scalar_f32; നിർത്തുക;
                    അവസ്ഥ HTP_OP_DIV: compute_scalar_f32 = hvx_div_scalar_f32; നിർത്തുക;
                    സ്ഥിരസ്ഥിതി: നിർത്തുക;
                }
            } അല്ലെങ്കിൽ എങ്കിൽ (src0_type == HTP_TYPE_F16) {
                തിരഞ്ഞെടുപ്പ് (octx->op) {
                    അവസ്ഥ HTP_OP_ADD: compute_scalar_f16 = hvx_add_scalar_f16; നിർത്തുക;
                    അവസ്ഥ HTP_OP_SUB: compute_scalar_f16 = hvx_sub_scalar_f16; നിർത്തുക;
                    അവസ്ഥ HTP_OP_MUL: compute_scalar_f16 = hvx_mul_scalar_f16; നിർത്തുക;
                    അവസ്ഥ HTP_OP_DIV: compute_scalar_f16 = hvx_div_scalar_f16; നിർത്തുക;
                    സ്ഥിരസ്ഥിതി: നിർത്തുക;
                }
            }
            എങ്കിൽ (!compute_scalar_f32 && !compute_scalar_f16) {
                തിരികെ HTP_STATUS_NO_SUPPORT;
            }
        } അല്ലെങ്കിൽ {
            എങ്കിൽ (src0_type == HTP_TYPE_F32) {
                തിരഞ്ഞെടുപ്പ് (octx->op) {
                    അവസ്ഥ HTP_OP_ADD: compute = hvx_add_f32; നിർത്തുക;
                    അവസ്ഥ HTP_OP_SUB: compute = hvx_sub_f32; നിർത്തുക;
                    അവസ്ഥ HTP_OP_MUL: compute = hvx_mul_f32; നിർത്തുക;
                    അവസ്ഥ HTP_OP_DIV: compute = hvx_div_f32; നിർത്തുക;
                    സ്ഥിരസ്ഥിതി: നിർത്തുക;
                }
            } അല്ലെങ്കിൽ എങ്കിൽ (src0_type == HTP_TYPE_F16) {
                തിരഞ്ഞെടുപ്പ് (octx->op) {
                    അവസ്ഥ HTP_OP_ADD: compute = hvx_add_f16; നിർത്തുക;
                    അവസ്ഥ HTP_OP_SUB: compute = hvx_sub_f16; നിർത്തുക;
                    അവസ്ഥ HTP_OP_MUL: compute = hvx_mul_f16; നിർത്തുക;
                    അവസ്ഥ HTP_OP_DIV: compute = hvx_div_f16; നിർത്തുക;
                    സ്ഥിരസ്ഥിതി: നിർത്തുക;
                }
            }
            എങ്കിൽ (!compute) {
                തിരികെ HTP_STATUS_NO_SUPPORT;
            }
        }

        സ്ഥിരം uint32_t total_elems = (uint32_t) (src0->ne[0] * src0->ne[1] * src0->ne[2] * src0->ne[3]);
        എങ്കിൽ (total_elems == 0) {
            തിരികെ HTP_STATUS_OK;
        }

        uint32_t elem_start = 0;
        uint32_t nelem      = total_elems;
        സ്ഥിരം uint32_t elems_per_വരി = (src0_type == HTP_TYPE_F32) ? 32 : 64;

        എങ്കിൽ (octx->ctx->mdev.എണ്ണം > 1) {
            സ്ഥിരം bool can_split = htp_tensor_mdev_data_aligned(dst) &&
                                   htp_tensor_mdev_data_aligned(src0) &&
                                   (is_scalar || htp_tensor_mdev_data_aligned(src1)) &&
                                   htp_tensor_is_contiguous(dst, elem_size) &&
                                   htp_tensor_is_contiguous(src0, elem_size) &&
                                   (is_scalar || htp_tensor_is_contiguous(src1, elem_size));
            സ്ഥിരം ഘടന htp_tensor_mdev_range range = htp_tensor_mdev_partition(
                total_elems, can_split ? elems_per_വരി : 0,
                octx->ctx->mdev.idx, octx->ctx->mdev.എണ്ണം, &octx->ctx->mdev.എണ്ണം_div);
            elem_start = range.start;
            nelem      = range.എണ്ണം;
        }

        എങ്കിൽ (nelem == 0) {
            തിരികെ HTP_STATUS_OK;
        }

        എങ്കിൽ (!htp_ops_context_set_n_threads(octx, kparams->n_threads)) {
            തിരികെ HTP_STATUS_INVAL_PARAMS;
        }

        ഘടന htp_ദ്വിമാന_vtcm_layout vtcm_layout;
        htp_ദ്വിമാന_vtcm_layout_build(&vtcm_layout, kparams, octx->ctx->vtcm_size);
        എങ്കിൽ (vtcm_layout.total_bytes == 0 || vtcm_layout.total_bytes > octx->ctx->vtcm_size) {
            തിരികെ HTP_STATUS_VTCM_TOO_SMALL;
        }

        സ്ഥിരം uint32_t chunk_size = kparams->chunk_size > 0 ? kparams->chunk_size : (32768 / elem_size);
        സ്ഥിരം uint32_t total_chunks = (nelem + chunk_size - 1) / chunk_size;
        സ്ഥിരം uint32_t n_threads = (total_chunks >= 2) ? MIN(octx->n_threads, total_chunks) : 1;
        സ്ഥിരം uint32_t chunks_per_thread = (total_chunks + n_threads - 1) / n_threads;

        ദശാംശം scalar_f32 = 0.0f;
        _Float16 scalar_f16 = 0;
        എങ്കിൽ (is_scalar) {
            uint8_t * vtcm_src1 = VTCM_LAYOUT_PTR(uint8_t, octx->ctx->vtcm_base, vtcm_layout.off_src1);
            dma_queue * dma_q = octx->ctx->dma[0];
            dma_queue_തള്ളുക(dma_q, dma_make_data(vtcm_src1, src1->data), 128, 0, elem_size, 1);
            dma_queue_എടുക്കുക(dma_q);

            എങ്കിൽ (src0_type == HTP_TYPE_F32) {
                scalar_f32 = ((സ്ഥിരം ദശാംശം *) vtcm_src1)[0];
            } അല്ലെങ്കിൽ {
                scalar_f16 = ((സ്ഥിരം _Float16 *) vtcm_src1)[0];
            }
        }

        ഘടന ദ്വിമാന_chunked_context cctx = {
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

        work_queue_run(octx->ctx->work_queue, ദ്വിമാന_thread_chunked, &cctx, n_threads);
        തിരികെ HTP_STATUS_OK;
    }

    uint32_t row_start = 0;
    uint32_t nrows     = src0_nrows;

    എങ്കിൽ (octx->ctx->mdev.എണ്ണം > 1) {
        uint32_t rows_per_chunk = 0;
        htp_tensor_mdev_rows_per_chunk(dst, (uint32_t) elem_size, (uint32_t) dst_row_size, &rows_per_chunk);
        സ്ഥിരം ഘടന htp_tensor_mdev_range range = htp_tensor_mdev_partition(src0_nrows, rows_per_chunk,
                                                       octx->ctx->mdev.idx, octx->ctx->mdev.എണ്ണം, &octx->ctx->mdev.എണ്ണം_div);
        row_start = range.start;
        nrows     = range.എണ്ണം;
    }

    എങ്കിൽ (nrows == 0) {
        തിരികെ HTP_STATUS_OK;
    }

    എങ്കിൽ (!htp_ops_context_set_n_threads(octx, kparams->n_threads)) {
        തിരികെ HTP_STATUS_INVAL_PARAMS;
    }

    സ്ഥിരം uint32_t n_threads = octx->n_threads;
    സ്ഥിരം size_t src0_row_size_aligned = kparams->src0_row_size_aligned;
    സ്ഥിരം size_t src1_row_size_aligned = kparams->src1_row_size_aligned;
    സ്ഥിരം size_t dst_row_size_aligned  = kparams->dst_row_size_aligned;

    ഘടന htp_ദ്വിമാന_context bctx;
    bctx.vtcm_base = (uint8_t *) octx->ctx->vtcm_base;
    htp_ദ്വിമാന_vtcm_layout_build(&bctx.vtcm_layout, kparams, octx->ctx->vtcm_size);

    എങ്കിൽ (bctx.vtcm_layout.rows_per_buffer == 0 || bctx.vtcm_layout.total_bytes > octx->ctx->vtcm_size) {
        തിരികെ HTP_STATUS_VTCM_TOO_SMALL;
    }

    dma_queue * dma_q = octx->ctx->dma[0];
    uint8_t * vtcm_src1 = VTCM_LAYOUT_PTR(uint8_t, bctx.vtcm_base, bctx.vtcm_layout.off_src1);
    എങ്കിൽ (kparams->kernel_type == HTP_ദ്വിമാന_KERNEL_ROW_BCAST) {
        dma_queue_തള്ളുക(dma_q, dma_make_data(vtcm_src1, src1->data), bctx.vtcm_layout.src1_size, 0, src1->ne[0] * elem_size, 1);
    } അല്ലെങ്കിൽ എങ്കിൽ (kparams->kernel_type == HTP_ദ്വിമാന_KERNEL_SCALAR ||
               kparams->kernel_type == HTP_ദ്വിമാന_KERNEL_REPEAT ||
               kparams->kernel_type == HTP_ദ്വിമാന_KERNEL_COMPLEX) {
        എങ്കിൽ (!kparams->src1_strided) {
            സ്ഥിരം size_t src1_bytes = (size_t) src1->ne[0] * src1->ne[1] * src1->ne[2] * src1->ne[3] * elem_size;
            dma_queue_തള്ളുക(dma_q, dma_make_data(vtcm_src1, src1->data), bctx.vtcm_layout.src1_size, 0, src1_bytes, 1);
        } അല്ലെങ്കിൽ {
            സ്ഥിരം uint32_t ne11 = src1->ne[1];
            സ്ഥിരം uint32_t ne12 = src1->ne[2];
            സ്ഥിരം uint32_t ne13 = src1->ne[3];
            സ്ഥിരം size_t row_bytes = src1->ne[0] * elem_size;
            വേണ്ടി (uint32_t i13 = 0; i13 < ne13; i13++) {
                വേണ്ടി (uint32_t i12 = 0; i12 < ne12; i12++) {
                    dma_addr_t src_addr = src1->data + i13 * src1->nb[3] + i12 * src1->nb[2];
                    uint8_t * dst_ptr = vtcm_src1 + (i13 * ne12 * ne11 + i12 * ne11) * row_bytes;
                    ആകുമ്പോൾ (!dma_queue_തള്ളുക(dma_q, dma_make_data(dst_ptr, src_addr),
                                           row_bytes, src1->nb[1], row_bytes, ne11)) {
                        dma_queue_എടുക്കുക(dma_q);
                    }
                }
            }
        }
    } അല്ലെങ്കിൽ എങ്കിൽ (kparams->kernel_type == HTP_ദ്വിമാന_KERNEL_ADD_ID) {
        സ്ഥിരം ഘടന htp_tensor * src2 = octx->src[2];
        uint8_t * vtcm_src2 = VTCM_LAYOUT_PTR(uint8_t, bctx.vtcm_base, bctx.vtcm_layout.off_src2);
        dma_queue_തള്ളുക(dma_q, dma_make_data(vtcm_src1, src1->data),
                       kparams->src1_row_size_aligned, src1->nb[1],
                       src1->ne[0] * elem_size, src1->ne[1]);
        സ്ഥിരം size_t src2_row_bytes = src2->ne[0] * വലുപ്പം(int32_t);
        dma_queue_തള്ളുക(dma_q, dma_make_data(vtcm_src2, src2->data),
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

    bctx.src0_dim1_div  = init_fastdiv_മൂല്യങ്ങൾ(src0->ne[1]);
    bctx.src0_dim2_div  = init_fastdiv_മൂല്യങ്ങൾ(src0->ne[2]);
    bctx.src0_dim12_div = init_fastdiv_മൂല്യങ്ങൾ(src0->ne[1] * src0->ne[2]);

    bctx.src1_dim1_div  = init_fastdiv_മൂല്യങ്ങൾ(src1->ne[1]);
    bctx.src1_dim2_div  = init_fastdiv_മൂല്യങ്ങൾ(src1->ne[2]);
    bctx.src1_dim3_div  = init_fastdiv_മൂല്യങ്ങൾ(src1->ne[3]);

    bctx.split_at_ne01  = kparams->split_at_ne01;
    bctx.split_at_ne02  = kparams->split_at_ne02;

    worker_callback_t worker_func = ശൂന്യ;
    ശൂന്യം * compute_func = ശൂന്യ;

    തിരഞ്ഞെടുപ്പ് (kparams->kernel_type) {
        അവസ്ഥ HTP_ദ്വിമാന_KERNEL_SAME_SHAPE:
            worker_func = ദ്വിമാന_thread_vector_same_shape;
            എങ്കിൽ (src0_type == HTP_TYPE_F32) {
                തിരഞ്ഞെടുപ്പ് (octx->op) {
                    അവസ്ഥ HTP_OP_ADD: compute_func = compute_same_shape_add_f32; നിർത്തുക;
                    അവസ്ഥ HTP_OP_SUB: compute_func = compute_same_shape_sub_f32; നിർത്തുക;
                    അവസ്ഥ HTP_OP_MUL: compute_func = compute_same_shape_mul_f32; നിർത്തുക;
                    അവസ്ഥ HTP_OP_DIV: compute_func = compute_same_shape_div_f32; നിർത്തുക;
                    സ്ഥിരസ്ഥിതി: നിർത്തുക;
                }
            } അല്ലെങ്കിൽ എങ്കിൽ (src0_type == HTP_TYPE_F16) {
                തിരഞ്ഞെടുപ്പ് (octx->op) {
                    അവസ്ഥ HTP_OP_ADD: compute_func = compute_same_shape_add_f16; നിർത്തുക;
                    അവസ്ഥ HTP_OP_SUB: compute_func = compute_same_shape_sub_f16; നിർത്തുക;
                    അവസ്ഥ HTP_OP_MUL: compute_func = compute_same_shape_mul_f16; നിർത്തുക;
                    അവസ്ഥ HTP_OP_DIV: compute_func = compute_same_shape_div_f16; നിർത്തുക;
                    സ്ഥിരസ്ഥിതി: നിർത്തുക;
                }
            }
            നിർത്തുക;
        അവസ്ഥ HTP_ദ്വിമാന_KERNEL_ROW_BCAST:
            worker_func = ദ്വിമാന_thread_vector_row_broadcast;
            എങ്കിൽ (src0_type == HTP_TYPE_F32) {
                തിരഞ്ഞെടുപ്പ് (octx->op) {
                    അവസ്ഥ HTP_OP_ADD: compute_func = compute_row_bcast_add_f32; നിർത്തുക;
                    അവസ്ഥ HTP_OP_SUB: compute_func = compute_row_bcast_sub_f32; നിർത്തുക;
                    അവസ്ഥ HTP_OP_MUL: compute_func = compute_row_bcast_mul_f32; നിർത്തുക;
                    അവസ്ഥ HTP_OP_DIV: compute_func = compute_row_bcast_div_f32; നിർത്തുക;
                    സ്ഥിരസ്ഥിതി: നിർത്തുക;
                }
            } അല്ലെങ്കിൽ എങ്കിൽ (src0_type == HTP_TYPE_F16) {
                തിരഞ്ഞെടുപ്പ് (octx->op) {
                    അവസ്ഥ HTP_OP_ADD: compute_func = compute_row_bcast_add_f16; നിർത്തുക;
                    അവസ്ഥ HTP_OP_SUB: compute_func = compute_row_bcast_sub_f16; നിർത്തുക;
                    അവസ്ഥ HTP_OP_MUL: compute_func = compute_row_bcast_mul_f16; നിർത്തുക;
                    അവസ്ഥ HTP_OP_DIV: compute_func = compute_row_bcast_div_f16; നിർത്തുക;
                    സ്ഥിരസ്ഥിതി: നിർത്തുക;
                }
            }
            നിർത്തുക;
        അവസ്ഥ HTP_ദ്വിമാന_KERNEL_SCALAR:
            worker_func = ദ്വിമാന_thread_scalar;
            എങ്കിൽ (src0_type == HTP_TYPE_F32) {
                തിരഞ്ഞെടുപ്പ് (octx->op) {
                    അവസ്ഥ HTP_OP_ADD: compute_func = compute_scalar_add_f32; നിർത്തുക;
                    അവസ്ഥ HTP_OP_SUB: compute_func = compute_scalar_sub_f32; നിർത്തുക;
                    അവസ്ഥ HTP_OP_MUL: compute_func = compute_scalar_mul_f32; നിർത്തുക;
                    അവസ്ഥ HTP_OP_DIV: compute_func = compute_scalar_div_f32; നിർത്തുക;
                    സ്ഥിരസ്ഥിതി: നിർത്തുക;
                }
            } അല്ലെങ്കിൽ എങ്കിൽ (src0_type == HTP_TYPE_F16) {
                തിരഞ്ഞെടുപ്പ് (octx->op) {
                    അവസ്ഥ HTP_OP_ADD: compute_func = compute_scalar_add_f16; നിർത്തുക;
                    അവസ്ഥ HTP_OP_SUB: compute_func = compute_scalar_sub_f16; നിർത്തുക;
                    അവസ്ഥ HTP_OP_MUL: compute_func = compute_scalar_mul_f16; നിർത്തുക;
                    അവസ്ഥ HTP_OP_DIV: compute_func = compute_scalar_div_f16; നിർത്തുക;
                    സ്ഥിരസ്ഥിതി: നിർത്തുക;
                }
            }
            നിർത്തുക;
        അവസ്ഥ HTP_ദ്വിമാന_KERNEL_COMPLEX:
            worker_func = ദ്വിമാന_thread_vector_complex;
            എങ്കിൽ (src0_type == HTP_TYPE_F32) {
                തിരഞ്ഞെടുപ്പ് (octx->op) {
                    അവസ്ഥ HTP_OP_ADD: compute_func = compute_complex_add_f32; നിർത്തുക;
                    അവസ്ഥ HTP_OP_SUB: compute_func = compute_complex_sub_f32; നിർത്തുക;
                    അവസ്ഥ HTP_OP_MUL: compute_func = compute_complex_mul_f32; നിർത്തുക;
                    അവസ്ഥ HTP_OP_DIV: compute_func = compute_complex_div_f32; നിർത്തുക;
                    സ്ഥിരസ്ഥിതി: നിർത്തുക;
                }
            } അല്ലെങ്കിൽ എങ്കിൽ (src0_type == HTP_TYPE_F16) {
                തിരഞ്ഞെടുപ്പ് (octx->op) {
                    അവസ്ഥ HTP_OP_ADD: compute_func = compute_complex_add_f16; നിർത്തുക;
                    അവസ്ഥ HTP_OP_SUB: compute_func = compute_complex_sub_f16; നിർത്തുക;
                    അവസ്ഥ HTP_OP_MUL: compute_func = compute_complex_mul_f16; നിർത്തുക;
                    അവസ്ഥ HTP_OP_DIV: compute_func = compute_complex_div_f16; നിർത്തുക;
                    സ്ഥിരസ്ഥിതി: നിർത്തുക;
                }
            }
            നിർത്തുക;
        അവസ്ഥ HTP_ദ്വിമാന_KERNEL_REPEAT:
            worker_func = ദ്വിമാന_thread_element_repeat;
            എങ്കിൽ (src0_type == HTP_TYPE_F32) {
                തിരഞ്ഞെടുപ്പ് (octx->op) {
                    അവസ്ഥ HTP_OP_ADD: compute_func = compute_repeat_add_f32; നിർത്തുക;
                    അവസ്ഥ HTP_OP_SUB: compute_func = compute_repeat_sub_f32; നിർത്തുക;
                    അവസ്ഥ HTP_OP_MUL: compute_func = compute_repeat_mul_f32; നിർത്തുക;
                    അവസ്ഥ HTP_OP_DIV: compute_func = compute_repeat_div_f32; നിർത്തുക;
                    സ്ഥിരസ്ഥിതി: നിർത്തുക;
                }
            } അല്ലെങ്കിൽ എങ്കിൽ (src0_type == HTP_TYPE_F16) {
                തിരഞ്ഞെടുപ്പ് (octx->op) {
                    അവസ്ഥ HTP_OP_ADD: compute_func = compute_repeat_add_f16; നിർത്തുക;
                    അവസ്ഥ HTP_OP_SUB: compute_func = compute_repeat_sub_f16; നിർത്തുക;
                    അവസ്ഥ HTP_OP_MUL: compute_func = compute_repeat_mul_f16; നിർത്തുക;
                    അവസ്ഥ HTP_OP_DIV: compute_func = compute_repeat_div_f16; നിർത്തുക;
                    സ്ഥിരസ്ഥിതി: നിർത്തുക;
                }
            }
            നിർത്തുക;
        അവസ്ഥ HTP_ദ്വിമാന_KERNEL_ADD_ID:
            എങ്കിൽ (octx->op == HTP_OP_ADD_ID && src0_type == HTP_TYPE_F32) {
                worker_func = ദ്വിമാന_thread_add_id_f32;
                compute_func = (ശൂന്യം *) compute_add_id_f32;
            }
            നിർത്തുക;
        സ്ഥിരസ്ഥിതി: നിർത്തുക;
    }

    എങ്കിൽ (!worker_func || !compute_func) {
        തിരികെ HTP_STATUS_NO_SUPPORT;
    }

    bctx.compute = compute_func;

    എങ്കിൽ (kparams->kernel_type == HTP_ദ്വിമാന_KERNEL_ROW_BCAST ||
        kparams->kernel_type == HTP_ദ്വിമാന_KERNEL_SCALAR ||
        kparams->kernel_type == HTP_ദ്വിമാന_KERNEL_REPEAT ||
        kparams->kernel_type == HTP_ദ്വിമാന_KERNEL_COMPLEX ||
        kparams->kernel_type == HTP_ദ്വിമാന_KERNEL_ADD_ID) {
        dma_queue_flush(dma_q);
    }

    work_queue_run(octx->ctx->work_queue, worker_func, &bctx, n_threads);

    തിരികെ HTP_STATUS_OK;
}

പൂർണ്ണസംഖ്യ op_ദ്വിമാന(ഘടന htp_ops_context * octx) {
    സ്ഥിരം uint32_t src0_type = octx->src[0]->type;
    എങ്കിൽ ((src0_type == HTP_TYPE_F32) || (src0_type == HTP_TYPE_F16)) {
        തിരികെ execute_op_ദ്വിമാന(octx);
    }

    തിരികെ HTP_STATUS_NO_SUPPORT;
}
