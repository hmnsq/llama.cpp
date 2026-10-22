#pragma clang diagnostic ignored "-Wgnu-zero-variadic-macro-arguments"
#pragma clang diagnostic ignored "-Wunused-function"
#pragma clang diagnostic ignored "-Wunused-variable"
#pragma clang diagnostic ignored "-Wunused-but-set-variable"

#ضمن <HAP_farf.h>
#ضمن <HAP_perf.h>
#ضمن <HAP_compute_res.h>

#ضمن <math.h>
#ضمن <string.h>
#ضمن <stdatomic.h>

#ضمن "dma-queue.h"
#ضمن "hvx-utils.h"
#ضمن "hvx-dump.h"
#ضمن "hvx-arith.h"
#ضمن "hvx-reduce.h"

#define GGML_COMMON_DECL_C
#ضمن "ggml-common.h"
#ضمن "htp-ctx.h"
#ضمن "htp-ops.h"
#ضمن "htp-tensor.h"
#ضمن "matmul-ops.h"
#ضمن "htp-vtcm.h"

تعريفنوع بنية {
    عائم        *dst;
    dma_addr_t    bias_addr;
    size_t        bias_bytes;
    dma_addr_t    act_dma_addr;
    dma_addr_t    weight;
    dma_queue *   weight_dma;
    صحيح           m;
    صحيح           k;
    صحيح           n;
    صحيح           act_stride;
    uint32_t      act_elem_size;  // 4=F32 src1, 2=F16 src1
    صحيح           weight_stride;
    صحيح           weight_type;
    صحيح           dst_stride;
    صحيح           dst_cols;
    uint32_t      bias_stride;
    صحيح           tile_size;
    صحيح           aligned_tile_size;
    صحيح           ne02;
    صحيح           ne03;
    صحيح           ne12;
    صحيح           ne13;
    size_t        src0_nb2;
    size_t        src0_nb3;
    size_t        act_nb2;
    size_t        act_nb3;
    size_t        bias_nb2;
    size_t        bias_nb3;
    size_t        dst_nb2;
    size_t        dst_nb3;
    صحيح           r2;
    صحيح           r3;
    بنية fastdiv_values div_r2;
    بنية fastdiv_values div_r3;
} hmx_mm_batched_params_t;

ساكن bool htp_matmul_has_extended_weight(ثابت بنية htp_ops_context * octx, uint32_t n_weights) {
    لكل (uint32_t i = 0; i < n_weights; ++i) {
        ܐܢ (htp_tensor_is_extended(octx->src[i])) {
            ܦܢܐ true;
        }
    }
    ܦܢܐ false;
}

بنية htp_mm_context {
    ثابت حرف * type;
    بنية htp_ops_context * octx;
    ثابت بنية htp_tensor * act;

    فراغ (*vec_dot_1x1)(ثابت uint32_t n, عائم * restrict s0,
         ثابت فراغ * restrict vx0,
         ثابت فراغ * restrict vy0);

    فراغ (*vec_dot_2x1)(ثابت uint32_t n, عائم * restrict s0,
         ثابت فراغ * restrict vx0, ثابت فراغ * restrict vx1,
         ثابت فراغ * restrict vy0);

    فراغ (*vec_dot_2x2)(ثابت uint32_t n, عائم * restrict s0, عائم * restrict s1,
         ثابت فراغ * restrict vx0, ثابت فراغ * restrict vx1,
         ثابت فراغ * restrict vy0, ثابت فراغ * restrict vy1);

    فراغ (*vec_dot_32x1)(ثابت uint32_t n, عائم * restrict s,
         ثابت فراغ * restrict vx,
         ثابت فراغ * restrict vy, uint32_t valid_rows,
         ثابت عائم * restrict sz);

    فراغ (*vec_dot_32x2)(ثابت uint32_t n, عائم * restrict s0, عائم * restrict s1,
         ثابت فراغ * restrict vx,
         ثابت فراغ * restrict vy0, ثابت فراغ * restrict vy1, uint32_t valid_rows,
         ثابت عائم * restrict sz0, ثابت عائم * restrict sz1);

    // Precomputed values
    uint32_t src0_nrows_per_thread;
    uint32_t src0_row_start;
    uint32_t src0_row_end;
    uint32_t src0_row_size_padded;
    uint32_t act_nrows;
    uint32_t cur_m_start;
    uint32_t cur_m_rows;

    بنية fastdiv_values mm_div_ne12_ne1;
    بنية fastdiv_values mm_div_ne1;
    بنية fastdiv_values mm_div_r2;
    بنية fastdiv_values mm_div_r3;
    بنية fastdiv_values mm_div_ne11;

    // Precomputed block-parallel quantization values
    uint32_t          quant_ib_first[WORK_QUEUE_MAX_N_THREADS];
    uint32_t          quant_ib_last[WORK_QUEUE_MAX_N_THREADS];
    uint32_t          quant_r[WORK_QUEUE_MAX_N_THREADS];
    uint32_t          quant_c[WORK_QUEUE_MAX_N_THREADS];
    uint32_t          n_quant_tasks;
    uint32_t          n_quant_rows_per_thread;

    // Fields for scattered mapping & HMX support in MUL_MAT_ID
    ثابت uint32_t * matrix_row_counts;
    ثابت بنية mmid_row_mapping * matrix_rows;
    uint32_t mapping_stride;

    // Dynamic VTCM pointers allocated sequentially
    uint8_t * vtcm_src0;
    uint8_t * vtcm_act;
    uint8_t * vtcm_bias;
    uint8_t * vtcm_dst;
    uint8_t * vtcm_act_raw;

    // Cached strides
    uint32_t vtcm_src0_stride;
    uint32_t vtcm_act_stride;
    uint32_t vtcm_act_raw_stride;

    // Cached thread offsets/sizes
    uint32_t vtcm_src0_size_per_thread;
    uint32_t vtcm_act_size_per_thread;
    uint32_t vtcm_dst_size_per_thread;
};

ساكن صحيح htp_mm_init_context(
    بنية htp_ops_context * octx,
    ثابت بنية htp_mm_kernel_params * kparams
) {
    ܐܢ (!htp_ops_context_set_n_threads(octx, (uint32_t) kparams->n_threads)) {
        ܦܢܐ HTP_STATUS_INVAL_PARAMS;
    }

    ܐܢ (kparams->n_hmx) {
        ܐܢ (kparams->n_act_threads <= 0 || kparams->n_act_threads > (int32_t) octx->n_threads) {
            ܦܢܐ HTP_STATUS_INVAL_PARAMS;
        }
    }

    ܦܢܐ HTP_STATUS_OK;
}

// vdelta control to expand first 32 e8m0 values into 32 uint32 elements
ساكن ثابت uint8_t __attribute__((aligned(128))) expand_x32_e8m0[128] = {
    0x00, 0x00, 0x00, 0x00, 0x01, 0x04, 0x00, 0x00, 0x02, 0x00, 0x08, 0x08, 0x01, 0x02, 0x00, 0x04, 0x04, 0x00, 0x00,
    0x00, 0x11, 0x10, 0x10, 0x10, 0x02, 0x00, 0x04, 0x00, 0x01, 0x02, 0x08, 0x08, 0x08, 0x08, 0x00, 0x00, 0x01, 0x04,
    0x00, 0x00, 0x22, 0x20, 0x20, 0x20, 0x21, 0x22, 0x20, 0x24, 0x04, 0x00, 0x00, 0x00, 0x09, 0x08, 0x00, 0x00, 0x02,
    0x00, 0x04, 0x00, 0x11, 0x12, 0x10, 0x10, 0x10, 0x10, 0x10, 0x10, 0x01, 0x04, 0x00, 0x00, 0x02, 0x00, 0x08, 0x08,
    0x01, 0x02, 0x00, 0x04, 0x44, 0x40, 0x40, 0x40, 0x41, 0x40, 0x40, 0x40, 0x42, 0x40, 0x44, 0x40, 0x41, 0x42, 0x48,
    0x48, 0x08, 0x08, 0x00, 0x00, 0x01, 0x04, 0x00, 0x00, 0x12, 0x10, 0x10, 0x10, 0x01, 0x02, 0x00, 0x04, 0x04, 0x00,
    0x00, 0x00, 0x09, 0x08, 0x00, 0x00, 0x22, 0x20, 0x24, 0x20, 0x21, 0x22, 0x20, 0x20,
};

// IQ4_NL dequantization LUT: maps 4-bit index (0-15) to int8 kvalue
// kvalues: -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113
ساكن ثابت uint8_t __attribute__((aligned(VLEN))) kvalues_iq4nl_lut[] = {
    0x81, 0, 0x98, 0, 0xAD, 0, 0xBF, 0, 0xCF, 0, 0xDD, 0, 0xEA, 0, 0xF6, 0, 0x01, 0, 0x0D, 0, 0x19, 0, 0x26, 0,
    0x35, 0, 0x45, 0, 0x59, 0, 0x71, 0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0,
    0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0,
    0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0,
    0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0,
};

ساكن ثابت uint8_t __attribute__((aligned(VLEN))) kvalues_mxfp4_lut[] = {
    0,    0, 1,    0, 2,    0, 3, 0, 4, 0, 6, 0, 8, 0, 12, 0, 0, 0, 0xff, 0, 0xfe, 0, 0xfd, 0, 0xfc, 0,
    0xfa, 0, 0xf8, 0, 0xf4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 0,    0, 0,    0, 0,    0, 0,    0,
    0,    0, 0,    0, 0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 0,    0, 0,    0, 0,    0, 0,    0,
    0,    0, 0,    0, 0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 0,    0, 0,    0, 0,    0, 0,    0,
    0,    0, 0,    0, 0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 0,    0, 0,    0, 0,    0,
};

#define htp_matmul_tensors_preamble                                 \
    ثابت بنية htp_tensor * restrict src0 = octx->src[0];         \
    ثابت بنية htp_tensor * restrict src1 = octx->src[1];         \
    ثابت بنية htp_tensor * restrict src2 = octx->src[2];         \
    ثابت بنية htp_tensor * restrict  dst = octx->dst;            \
    ثابت بنية htp_tensor * restrict  act = src1;                 \
                                                                    \
    ثابت uint32_t ne00 = src0->ne[0];                              \
    ثابت uint32_t ne01 = src0->ne[1];                              \
    ثابت uint32_t ne02 = src0->ne[2];                              \
    ثابت uint32_t ne03 = src0->ne[3];                              \
                                                                    \
    ثابت uint32_t ne10 = src1->ne[0];                              \
    ثابت uint32_t ne11 = src1->ne[1];                              \
    ثابت uint32_t ne12 = src1->ne[2];                              \
    ثابت uint32_t ne13 = src1->ne[3];                              \
                                                                    \
    ثابت uint32_t ne20 = src2 ? src2->ne[0] : 0;                   \
    ثابت uint32_t ne21 = src2 ? src2->ne[1] : 0;                   \
    ثابت uint32_t ne22 = src2 ? src2->ne[2] : 0;                   \
    ثابت uint32_t ne23 = src2 ? src2->ne[3] : 0;                   \
                                                                    \
    ثابت uint32_t ne0 = dst->ne[0];                                \
    ثابت uint32_t ne1 = dst->ne[1];                                \
    ثابت uint32_t ne2 = dst->ne[2];                                \
    ثابت uint32_t ne3 = dst->ne[3];                                \
                                                                    \
    ثابت uint32_t nb00 = src0->nb[0];                              \
    ثابت uint32_t nb01 = src0->nb[1];                              \
    ثابت uint32_t nb02 = src0->nb[2];                              \
    ثابت uint32_t nb03 = src0->nb[3];                              \
                                                                    \
    ثابت uint32_t nb10 = src1->nb[0];                              \
    ثابت uint32_t nb11 = src1->nb[1];                              \
    ثابت uint32_t nb12 = src1->nb[2];                              \
    ثابت uint32_t nb13 = src1->nb[3];                              \
                                                                    \
    ثابت uint32_t nb0 = dst->nb[0];                                \
    ثابت uint32_t nb1 = dst->nb[1];                                \
    ثابت uint32_t nb2 = dst->nb[2];                                \
    ثابت uint32_t nb3 = dst->nb[3];

#define htp_matmul_preamble                                         \
    بنية htp_mm_context * mmctx  = data;                          \
    بنية htp_ops_context * octx  = mmctx->octx;                   \
    dma_queue * dma_q              = octx->ctx->dma[ith];           \
    uint32_t src0_nrows_per_thread = mmctx->src0_nrows_per_thread;  \
    htp_matmul_tensors_preamble;




// hvx kernels first: the HMX Q6_K dequantizer reuses unpack_q6_k_group from there
#ضمن "hvx-mm-kernels-tiled.h"
#ضمن "hvx-mm-kernels-float.h"
#ضمن "hmx-mm-kernels-tiled.h"

ساكن فراغ hvx_mm_2d_tiled(unsigned صحيح nth, unsigned صحيح ith, فراغ * data) {
    htp_matmul_preamble;

    ثابت uint32_t src0_nrows = mmctx->src0_row_end - mmctx->src0_row_start;
    ثابت uint32_t act_nrows = mmctx->cur_m_rows ? mmctx->cur_m_rows : (ne11 * ne12 * ne13);
    ثابت uint32_t cur_m_start = mmctx->cur_m_start;

    ثابت uint32_t src0_start_row  = mmctx->src0_row_start + src0_nrows_per_thread * ith;
    ثابت uint32_t src0_end_row    = MIN(src0_start_row + src0_nrows_per_thread, mmctx->src0_row_end);

    بنية htp_thread_trace * tr = &octx->ctx->trace[ith];

    ثابت بنية htp_mm_kernel_params * kparams = (ثابت بنية htp_mm_kernel_params *) octx->kernel_params;
    ثابت uint32_t n_prefetch = kparams->n_prefetch;
    أكد(n_prefetch >= 2 && n_prefetch <= HTP_MM_MAX_PREFETCH && (n_prefetch & (n_prefetch - 1)) == 0);

    ثابت size_t dst_row_size  = nb1;
    ثابت size_t act_row_size = nb11;
    ثابت size_t act_stride = mmctx->vtcm_act_stride;
    ثابت size_t src2_stride = src2 ? ((src2->ne[1] == 1) ? 0 : src2->nb[1]) : 0;

    uint8_t * restrict vtcm_dst_ptr  = mmctx->vtcm_dst  + mmctx->vtcm_dst_size_per_thread  * ith;
    uint8_t * restrict vtcm_src0_ptr = mmctx->vtcm_src0 + mmctx->vtcm_src0_size_per_thread * ith;
    uint8_t * restrict act_data = mmctx->vtcm_act;

    ثابت dma_addr_t src0_row = src0->data;

    ثابت uint32_t tile_size = htp_tensor_get_tile_size(src0->type);
    ثابت uint32_t aligned_tile_size = htp_tensor_get_tile_padded_size(src0->type);

    uint32_t n_k_tiles_w = ne00 / htp_tensor_get_tile_k(src0->type);
    uint32_t tile_row_stride = n_k_tiles_w * tile_size;
    uint32_t tile_row_transfer_size_aligned = n_k_tiles_w * aligned_tile_size;

    uint32_t ct_start = src0_start_row / 32;
    uint32_t ct_end   = (src0_end_row + 31) / 32;

    uint32_t push_ct = ct_start;
    ܐܢ (src0_start_row < src0_end_row) {
        لكل (uint32_t d = 0; d < n_prefetch && push_ct < ct_end; d++, push_ct++) {
            dma_queue_push(dma_q, dma_make_data(vtcm_src0_ptr + d * tile_row_transfer_size_aligned,
                           src0_row + push_ct * tile_row_stride), aligned_tile_size, tile_size, tile_size, n_k_tiles_w);
        }
    }

    ܐܢ (src0_start_row >= src0_end_row) {
        ܦܢܐ;
    }

    لكل (uint32_t ct = ct_start; ct < ct_end; ct++) {
        ثابت uint8_t * w_tile = (فراغ *) dma_queue_pop(dma_q).dst;

        صحيح valid_rows = (صحيح)ne0 - (صحيح)(ct * 32);
        valid_rows = MIN(32, MAX(0, valid_rows));

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, ct);
        uint32_t ir1 = 0;
        لكل (; ir1 + 1 < act_nrows; ir1 += 2) {
            ثابت uint8_t * restrict act_col0 = (ثابت uint8_t *) (act_data + (ir1+0) * act_stride);
            ثابت uint8_t * restrict act_col1 = (ثابت uint8_t *) (act_data + (ir1+1) * act_stride);
            عائم * restrict dst_row0 = (عائم *) (dst->data + ((cur_m_start + ir1+0) * dst_row_size));
            عائم * restrict dst_row1 = (عائم *) (dst->data + ((cur_m_start + ir1+1) * dst_row_size));

            عائم * dst_ptr0 = &dst_row0[ct * 32];
            عائم * dst_ptr1 = &dst_row1[ct * 32];

            ثابت عائم * src2_ptr0 = عدم;
            ثابت عائم * src2_ptr1 = عدم;
            ܐܢ (src2) {
                ثابت عائم * restrict src2_row0 = (ثابت عائم *) ((ثابت uint8_t *) src2->data + ((cur_m_start + ir1+0) * src2_stride));
                ثابت عائم * restrict src2_row1 = (ثابت عائم *) ((ثابت uint8_t *) src2->data + ((cur_m_start + ir1+1) * src2_stride));
                src2_ptr0 = &src2_row0[ct * 32];
                src2_ptr1 = &src2_row1[ct * 32];
            }
            mmctx->vec_dot_32x2(ne10, dst_ptr0, dst_ptr1, w_tile, act_col0, act_col1, valid_rows, src2_ptr0, src2_ptr1);
        }

        لكل (; ir1 < act_nrows; ++ir1) {
            ثابت uint8_t * restrict act_col = (ثابت uint8_t *) (act_data + ir1 * act_stride);
            عائم * restrict dst_row          = (عائم *) (dst->data + ((cur_m_start + ir1) * dst_row_size));
            عائم * dst_ptr = &dst_row[ct * 32];

            ثابت عائم * src2_ptr = عدم;
            ܐܢ (src2) {
                ثابت عائم * restrict src2_row = (ثابت عائم *) ((ثابت uint8_t *) src2->data + ((cur_m_start + ir1) * src2_stride));
                src2_ptr = &src2_row[ct * 32];
            }
            mmctx->vec_dot_32x1(ne10, dst_ptr, w_tile, act_col, valid_rows, src2_ptr);
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ct);

        ܐܢ (push_ct < ct_end) {
            dma_queue_push(dma_q, dma_make_data(w_tile, src0_row + push_ct * tile_row_stride),
                           aligned_tile_size, tile_size, tile_size, n_k_tiles_w);
            push_ct++;
        }
    }
}

ساكن فراغ hvx_mv_2d_tiled(unsigned صحيح nth, unsigned صحيح ith, فراغ * data) {
    htp_matmul_preamble;

    ثابت uint32_t src0_nrows = mmctx->src0_row_end - mmctx->src0_row_start;

    ثابت uint32_t src0_start_row  = mmctx->src0_row_start + src0_nrows_per_thread * ith;
    ثابت uint32_t src0_end_row    = MIN(src0_start_row + src0_nrows_per_thread, mmctx->src0_row_end);

    بنية htp_thread_trace * tr = &octx->ctx->trace[ith];

    ثابت بنية htp_mm_kernel_params * kparams = (ثابت بنية htp_mm_kernel_params *) octx->kernel_params;
    ثابت uint32_t n_prefetch = kparams->n_prefetch;
    أكد(n_prefetch >= 2 && n_prefetch <= HTP_MM_MAX_PREFETCH && (n_prefetch & (n_prefetch - 1)) == 0);

    ثابت size_t dst_row_size  = nb1;
    ثابت size_t act_row_size = nb11;
    ثابت size_t act_stride = mmctx->vtcm_act_stride;

    uint8_t * vtcm_dst_ptr  = mmctx->vtcm_dst + mmctx->vtcm_dst_size_per_thread * ith;
    uint8_t * vtcm_src0_ptr = mmctx->vtcm_src0 + mmctx->vtcm_src0_size_per_thread * ith;
    uint8_t * act_data = mmctx->vtcm_act;

    عائم * tmp = (عائم *) vtcm_dst_ptr;

    ثابت dma_addr_t src0_row = src0->data;

    ثابت uint8_t * restrict act_col = (ثابت uint8_t *) act_data;
    عائم * restrict dst_col          = (عائم *) dst->data;

    ثابت uint32_t tile_size = htp_tensor_get_tile_size(src0->type);
    ثابت uint32_t aligned_tile_size = htp_tensor_get_tile_padded_size(src0->type);

    uint32_t n_k_tiles_w = ne00 / htp_tensor_get_tile_k(src0->type);
    uint32_t tile_row_stride = n_k_tiles_w * tile_size;
    uint32_t tile_row_transfer_size_aligned = n_k_tiles_w * aligned_tile_size;

    uint32_t ct_start = src0_start_row / 32;
    uint32_t ct_end   = (src0_end_row + 31) / 32;

    uint32_t push_ct = ct_start;
    ܐܢ (src0_start_row < src0_end_row) {
        ܐܢ (src2) {
            عائم * vtcm_bias_ptr = (عائم *) mmctx->vtcm_bias + src0_start_row;
            ثابت dma_addr_t src2_addr = src2->data + src0_start_row * حجم(عائم);
            صحيح slice_size = (صحيح)MIN(src0_end_row, ne0) - (صحيح)src0_start_row;
            ܐܢ (slice_size > 0) {
                dma_queue_push(dma_q, dma_make_data(vtcm_bias_ptr, src2_addr),
                               slice_size * حجم(عائم), slice_size * حجم(عائم), slice_size * حجم(عائم), 1);
                dma_queue_pop_nowait(dma_q);
            }
        }
        لكل (uint32_t d = 0; d < n_prefetch && push_ct < ct_end; d++, push_ct++) {
            dma_queue_push(dma_q, dma_make_data(vtcm_src0_ptr + d * tile_row_transfer_size_aligned,
                           src0_row + push_ct * tile_row_stride), aligned_tile_size, tile_size, tile_size, n_k_tiles_w);
        }
    }

    ܐܢ (src0_start_row >= src0_end_row) {
        ܦܢܐ;
    }

    لكل (uint32_t ct = ct_start; ct < ct_end; ct++) {
        ثابت uint8_t * w_tile = (فراغ *) dma_queue_pop(dma_q).dst;

        عائم * dst_ptr = &tmp[ct * 32 - src0_start_row];
        صحيح valid_rows = (صحيح)ne0 - (صحيح)(ct * 32);
        valid_rows = MIN(32, MAX(0, valid_rows));

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, ct);
        mmctx->vec_dot_32x1(ne10, dst_ptr, w_tile, act_col, valid_rows, عدم);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ct);

        ܐܢ (push_ct < ct_end) {
            dma_queue_push(dma_q, dma_make_data(w_tile, src0_row + push_ct * tile_row_stride),
                           aligned_tile_size, tile_size, tile_size, n_k_tiles_w);
            push_ct++;
        }
    }

    صحيح copy_cnt = (صحيح)MIN(src0_end_row, ne0) - (صحيح)src0_start_row;
    ܐܢ (copy_cnt > 0) {
        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, ct_end);
        ܐܢ (src2) {
            hvx_add_f32_uuu((uint8_t *) &dst_col[src0_start_row],
                            (ثابت uint8_t *) tmp,
                            (ثابت uint8_t *) ((ثابت عائم *) mmctx->vtcm_bias + src0_start_row),
                            copy_cnt);
        } ܐܚܪܢܐ {
            hvx_copy_f32_ua((uint8_t *) &dst_col[src0_start_row], (uint8_t *) tmp, copy_cnt);
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ct_end);
    }
}

ساكن فراغ hvx_mm_nx_2d_tiled(unsigned صحيح nth, unsigned صحيح ith, فراغ * data) {
    بنية htp_mm_context * mmctx = data;
    بنية htp_ops_context * octx = mmctx->octx;
    ثابت بنية htp_mm_kernel_params * kparams = (ثابت بنية htp_mm_kernel_params *) octx->kernel_params;
    ثابت uint32_t n_weights = kparams->n_weights;

    ثابت بنية htp_tensor * restrict act = octx->src[n_weights]; /* x */
    ثابت uint32_t ne10 = act->ne[0];
    ثابت uint32_t act_nrows = act->ne[1] * act->ne[2] * act->ne[3];
    ثابت size_t act_stride = mmctx->vtcm_act_stride;

    uint8_t * restrict vtcm_weight_ptr = mmctx->vtcm_src0 + mmctx->vtcm_src0_size_per_thread * ith;
    uint8_t * restrict act_data        = mmctx->vtcm_act;

    بنية htp_thread_trace * tr = &octx->ctx->trace[ith];
    ثابت uint32_t n_prefetch = kparams->n_prefetch;
    أكد(n_prefetch >= 2 && n_prefetch <= HTP_MM_MAX_PREFETCH && (n_prefetch & (n_prefetch - 1)) == 0);

    لكل (uint32_t widx = 0; widx < n_weights; widx++) {
        ثابت بنية htp_tensor * restrict src_w = octx->src[widx];
        ثابت بنية htp_tensor * restrict dst   = octx->dsts[widx];
        ܐܢ (!src_w || !dst) استمر;
        dma_queue * dma_q = octx->ctx->dma[ith];

        ثابت uint32_t tile_size = htp_tensor_get_tile_size(src_w->type);
        ثابت uint32_t aligned_tile_size = htp_tensor_get_tile_padded_size(src_w->type);
        ثابت uint32_t ne00 = src_w->ne[0];
        ثابت uint32_t ne01 = src_w->ne[1];
        ثابت size_t dst_row_size = dst->nb[1];
        ثابت dma_addr_t src_w_row = src_w->data;

        ثابت uint32_t n_k_tiles_w = ne00 / htp_tensor_get_tile_k(src_w->type);
        ثابت uint32_t tile_row_stride = n_k_tiles_w * tile_size;
        ثابت uint32_t tile_row_transfer_size_aligned = n_k_tiles_w * aligned_tile_size;

        uint32_t src0_start_row = 0;
        uint32_t src0_end_row   = ne01;
        ܐܢ (octx->ctx->mdev.count > 1) {
            ثابت bool can_split = htp_tensor_can_row_partition(dst, حجم(عائم));
            ثابت بنية htp_tensor_mdev_range range = htp_tensor_mdev_partition(ne01, can_split ? 32 : 0,
                                                         octx->ctx->mdev.idx, octx->ctx->mdev.count, &octx->ctx->mdev.count_div);
            src0_start_row = range.start;
            src0_end_row   = range.start + range.count;
        }

        ثابت uint32_t nrows = src0_end_row - src0_start_row;
        uint32_t src0_nrows_per_thread = fastdiv(nrows + nth - 1, &octx->n_threads_div);
        src0_nrows_per_thread = hex_round_up(src0_nrows_per_thread, 32);

        ثابت uint32_t start_row = src0_start_row + src0_nrows_per_thread * ith;
        ثابت uint32_t end_row   = MIN(start_row + src0_nrows_per_thread, src0_end_row);
        ܐܢ (start_row >= end_row) استمر;

        uint32_t ct_start = start_row / 32;
        uint32_t ct_end   = (end_row + 31) / 32;

        uint32_t push_ct = ct_start;
        لكل (uint32_t d = 0; d < n_prefetch && push_ct < ct_end; d++, push_ct++) {
            dma_queue_push(dma_q, dma_make_data(vtcm_weight_ptr + d * tile_row_transfer_size_aligned,
                            src_w_row + push_ct * tile_row_stride), aligned_tile_size, tile_size, tile_size, n_k_tiles_w);
        }

        لكل (uint32_t ct = ct_start; ct < ct_end; ct++) {
            ثابت uint8_t * w_tile = (فراغ *) dma_queue_pop(dma_q).dst;
            صحيح valid_rows = (صحيح)dst->ne[0] - (صحيح)(ct * 32);
            valid_rows = MIN(32, MAX(0, valid_rows));

            htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, ct);
            uint32_t ir1 = 0;
            لكل (; ir1 + 1 < act_nrows; ir1 += 2) {
                ثابت uint8_t * restrict act_col0 = (ثابت uint8_t *) (act_data + (ir1+0) * act_stride);
                ثابت uint8_t * restrict act_col1 = (ثابت uint8_t *) (act_data + (ir1+1) * act_stride);

                عائم * restrict dst_row0 = (عائم *) (dst->data + ((ir1+0) * dst_row_size));
                عائم * restrict dst_row1 = (عائم *) (dst->data + ((ir1+1) * dst_row_size));
                عائم * dst_ptr0 = &dst_row0[ct * 32];
                عائم * dst_ptr1 = &dst_row1[ct * 32];

                mmctx->vec_dot_32x2(ne10, dst_ptr0, dst_ptr1, w_tile, act_col0, act_col1, valid_rows, عدم, عدم);
            }

            لكل (; ir1 < act_nrows; ++ir1) {
                ثابت uint8_t * restrict act_col = (ثابت uint8_t *) (act_data + ir1 * act_stride);
                عائم * restrict dst_row = (عائم *) (dst->data + (ir1 * dst_row_size));
                عائم * dst_ptr = &dst_row[ct * 32];
                mmctx->vec_dot_32x1(ne10, dst_ptr, w_tile, act_col, valid_rows, عدم);
            }
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ct);

            ܐܢ (push_ct < ct_end) {
                dma_queue_push(dma_q, dma_make_data(w_tile, src_w_row + push_ct * tile_row_stride),
                                aligned_tile_size, tile_size, tile_size, n_k_tiles_w);
                push_ct++;
            }
        }
    }
}

ساكن فراغ hvx_mm_transfer_act_dma(
    بنية htp_ops_context * octx,
    ثابت بنية htp_mm_kernel_params * kparams,
    ثابت بنية htp_tensor * act,
    uint8_t * dst_base,
    size_t dst_row_size,
    uint32_t m_start,
    uint32_t m_rows
) {
    ܐܢ (m_rows == 0) {
        ܦܢܐ;
    }

    dma_queue * dma_q = octx->ctx->dma[0];
    ثابت uint32_t ne0 = act->ne[0];
    ثابت size_t elem_size = (act->type == HTP_TYPE_F16) ? حجم(__fp16) : حجم(عائم);
    ثابت size_t row_bytes = ne0 * elem_size;
    ثابت size_t act_nb1 = act->nb[1];
    ثابت dma_addr_t act_base = act->data;

    ثابت bool is_contiguous = (act->nb[2] == act->ne[1] * act_nb1) &&
                               (act->nb[3] == act->ne[2] * act->nb[2]);

    ܐܢ (is_contiguous) {
        ثابت dma_addr_t src_addr = act_base + m_start * act_nb1;
        dma_queue_push(dma_q, dma_make_data(dst_base, src_addr),
                       dst_row_size, act_nb1, row_bytes, m_rows);
        dma_queue_pop(dma_q);
    } ܐܚܪܢܐ {
        ثابت uint32_t ne12_ne1 = act->ne[2] * act->ne[1];
        ثابت bool use_fastdiv = kparams->div_ne12_ne1.mp != 0;
        لكل (uint32_t ir = 0; ir < m_rows; ++ir) {
            ثابت uint32_t ir1 = m_start + ir;
            uint32_t i11, i12, i13;
            ܐܢ (use_fastdiv) {
                i13 = fastdiv(ir1, &kparams->div_ne12_ne1);
                ثابت uint32_t rem = ir1 - i13 * ne12_ne1;
                i12 = fastdiv(rem, &kparams->div_ne1);
                i11 = rem - i12 * act->ne[1];
            } ܐܚܪܢܐ {
                i13 = ne12_ne1 ? ir1 / ne12_ne1 : 0;
                ثابت uint32_t rem = ir1 - i13 * ne12_ne1;
                i12 = act->ne[1] ? rem / act->ne[1] : 0;
                i11 = rem - i12 * act->ne[1];
            }
            ثابت dma_addr_t row_src = act_base + (i11 * act->nb[1] +
                                                   i12 * act->nb[2] +
                                                   i13 * act->nb[3]);
            uint8_t * row_dst = dst_base + ir * dst_row_size;
            dma_queue_push(dma_q, dma_make_data(row_dst, row_src),
                           dst_row_size, act_nb1, row_bytes, 1);
            dma_queue_pop(dma_q);
        }
    }

    ܐܢ (dst_row_size > row_bytes) {
        بنية htp_thread_trace * tr = &octx->ctx->trace[0];
        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_A_PREP, (uint16_t) m_start);
        ثابت uint32_t pad_elems = (dst_row_size - row_bytes) / elem_size;
        ܐܢ (elem_size == حجم(عائم)) {
            لكل (uint32_t ir = 0; ir < m_rows; ++ir) {
                hvx_splat_f32_u(dst_base + ir * dst_row_size + row_bytes, 0.0f, pad_elems);
            }
        } ܐܚܪܢܐ {
            لكل (uint32_t ir = 0; ir < m_rows; ++ir) {
                hvx_splat_f16_u(dst_base + ir * dst_row_size + row_bytes, (_Float16) 0.0f, pad_elems);
            }
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_A_PREP, (uint16_t) m_start);
    }
}

#define QUANTIZE_IMPL(name, log_name, kernel_fn, dst_row_size_expr)                                        \
ساكن فراغ name(unsigned صحيح nth, unsigned صحيح ith, فراغ * data) {                                        \
    (فراغ) nth;                                                                                            \
    بنية htp_mm_context * mmctx = data;                                                                  \
    بنية htp_ops_context * octx = mmctx->octx;                                                           \
    ثابت بنية htp_tensor * src = mmctx->act;                                                            \
    ثابت uint32_t ne0 = src->ne[0];                                                                       \
    ثابت uint32_t nrows = mmctx->cur_m_rows ? mmctx->cur_m_rows : mmctx->act_nrows;                       \
    ثابت uint32_t nrows_per_thread = mmctx->n_quant_rows_per_thread;                                      \
                                                                                                           \
    ثابت uint32_t ir_first = nrows_per_thread * ith;                                                      \
    ܐܢ (ir_first >= nrows) {                                                                               \
        ܦܢܐ;                                                                                            \
    }                                                                                                      \
                                                                                                           \
    بنية htp_thread_trace * tr = &octx->ctx->trace[ith];                                                 \
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_A_QUANT, ir_first);                                        \
                                                                                                           \
    uint8_t * restrict dst = mmctx->vtcm_act;                                                              \
    ثابت uint32_t ir_last = MIN(ir_first + nrows_per_thread, nrows);                                      \
    ثابت size_t raw_row_size = mmctx->vtcm_act_raw_stride;                                                \
    ثابت size_t dst_row_size = (dst_row_size_expr);                                                       \
                                                                                                           \
    ثابت uint8_t * restrict src_data = (ثابت uint8_t *) mmctx->vtcm_act_raw + (raw_row_size * ir_first); \
    uint8_t * restrict dst_data = (uint8_t *) dst + (dst_row_size * ir_first);                             \
    kernel_fn(src_data, dst_data, عدم, ne0, ir_last - ir_first, raw_row_size, dst_row_size);              \
                                                                                                           \
    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_A_QUANT, ir_first);                                         \
}

QUANTIZE_IMPL(quantize_f32_q8_0_tiled,     "quantize-f32-q8_0_tiled",     quantize_f32_q8_0_tiled_kernel,     htp_mm_q8_0_tiled_row_size(ne0))
QUANTIZE_IMPL(quantize_f32_q8_1_tiled,     "quantize-f32-q8_1_tiled",     quantize_f32_q8_1_tiled_kernel,     htp_mm_q8_1_tiled_row_size(ne0))
QUANTIZE_IMPL(quantize_f32_q8_1_s16_tiled, "quantize-f32-q8_1_s16_tiled", quantize_f32_q8_1_s16_tiled_kernel, htp_mm_q8_1_tiled_row_size(ne0))
QUANTIZE_IMPL(quantize_f32_f16_tiled,      "quantize-f32-f16_tiled",      quantize_f32_f16_tiled_kernel,      htp_mm_f16_tiled_row_size(ne0))
QUANTIZE_IMPL(quantize_f16_f16_tiled,      "quantize-f16-f16_tiled",      quantize_f16_f16_tiled_kernel,      htp_mm_f16_tiled_row_size(ne0))
QUANTIZE_IMPL(quantize_f32_f32,            "quantize-f32-f32",            quantize_f32_f32_kernel,            mmctx->vtcm_act_stride)
QUANTIZE_IMPL(quantize_f32_f16,            "quantize-f32-f16",            quantize_f32_f16_kernel,            mmctx->vtcm_act_stride)
QUANTIZE_IMPL(quantize_f16_f16,            "quantize-f16-f16",            quantize_f16_f16_kernel,            mmctx->vtcm_act_stride)

ساكن فراغ quantize_f32_q8_0_tiled_block(unsigned صحيح nth, unsigned صحيح ith, فراغ * data) {
    (فراغ) nth;
    بنية htp_mm_context * mmctx = data;
    ܐܢ (mmctx->quant_ib_first[ith] >= mmctx->quant_ib_last[ith]) {
        ܦܢܐ;
    }
    بنية htp_ops_context * octx = mmctx->octx;
    بنية htp_thread_trace * tr = &octx->ctx->trace[ith];
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_A_QUANT, mmctx->quant_ib_first[ith]);

    ثابت بنية htp_tensor * src = mmctx->act;

    quantize_f32_q8_0_tiled_block_kernel(
        (ثابت عائم *) mmctx->vtcm_act_raw,
        mmctx->vtcm_act,
        عدم,
        src->ne[0],
        mmctx->quant_ib_first[ith],
        mmctx->quant_ib_last[ith],
        mmctx->vtcm_act_raw_stride,
        htp_mm_q8_0_tiled_row_size(src->ne[0]),
        mmctx->quant_r[ith],
        mmctx->quant_c[ith]
    );

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_A_QUANT, mmctx->quant_ib_first[ith]);
}

ساكن فراغ quantize_f32_q8_1_tiled_block(unsigned صحيح nth, unsigned صحيح ith, فراغ * data) {
    (فراغ) nth;
    بنية htp_mm_context * mmctx = data;
    ܐܢ (mmctx->quant_ib_first[ith] >= mmctx->quant_ib_last[ith]) {
        ܦܢܐ;
    }
    بنية htp_ops_context * octx = mmctx->octx;
    بنية htp_thread_trace * tr = &octx->ctx->trace[ith];
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_A_QUANT, mmctx->quant_ib_first[ith]);

    ثابت بنية htp_tensor * src = mmctx->act;

    quantize_f32_q8_1_tiled_block_kernel(
        (ثابت عائم *) mmctx->vtcm_act_raw,
        mmctx->vtcm_act,
        عدم,
        src->ne[0],
        mmctx->quant_ib_first[ith],
        mmctx->quant_ib_last[ith],
        mmctx->vtcm_act_raw_stride,
        htp_mm_q8_1_tiled_row_size(src->ne[0]),
        mmctx->quant_r[ith],
        mmctx->quant_c[ith]
    );

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_A_QUANT, mmctx->quant_ib_first[ith]);
}

ساكن فراغ quantize_f32_q8_1_s16_tiled_block(unsigned صحيح nth, unsigned صحيح ith, فراغ * data) {
    (فراغ) nth;
    بنية htp_mm_context * mmctx = data;
    ܐܢ (mmctx->quant_ib_first[ith] >= mmctx->quant_ib_last[ith]) {
        ܦܢܐ;
    }
    بنية htp_ops_context * octx = mmctx->octx;
    بنية htp_thread_trace * tr = &octx->ctx->trace[ith];
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_A_QUANT, mmctx->quant_ib_first[ith]);

    ثابت بنية htp_tensor * src = mmctx->act;

    quantize_f32_q8_1_s16_tiled_block_kernel(
        (ثابت عائم *) mmctx->vtcm_act_raw,
        mmctx->vtcm_act,
        عدم,
        src->ne[0],
        mmctx->quant_ib_first[ith],
        mmctx->quant_ib_last[ith],
        mmctx->vtcm_act_raw_stride,
        htp_mm_q8_1_tiled_row_size(src->ne[0]),
        mmctx->quant_r[ith],
        mmctx->quant_c[ith]
    );

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_A_QUANT, mmctx->quant_ib_first[ith]);
}

ساكن فراغ quantize_f32_f16_tiled_block(unsigned صحيح nth, unsigned صحيح ith, فراغ * data) {
    (فراغ) nth;
    بنية htp_mm_context * mmctx = data;
    ܐܢ (mmctx->quant_ib_first[ith] >= mmctx->quant_ib_last[ith]) {
        ܦܢܐ;
    }
    بنية htp_ops_context * octx = mmctx->octx;
    بنية htp_thread_trace * tr = &octx->ctx->trace[ith];
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_A_QUANT, mmctx->quant_ib_first[ith]);

    ثابت بنية htp_tensor * src = mmctx->act;

    quantize_f32_f16_tiled_block_kernel(
        (ثابت عائم *) mmctx->vtcm_act_raw,
        mmctx->vtcm_act,
        عدم,
        src->ne[0],
        mmctx->quant_ib_first[ith],
        mmctx->quant_ib_last[ith],
        mmctx->vtcm_act_raw_stride,
        htp_mm_f16_tiled_row_size(src->ne[0]),
        mmctx->quant_r[ith],
        mmctx->quant_c[ith]
    );

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_A_QUANT, mmctx->quant_ib_first[ith]);
}

ساكن فراغ quantize_f16_f16_tiled_block(unsigned صحيح nth, unsigned صحيح ith, فراغ * data) {
    (فراغ) nth;
    بنية htp_mm_context * mmctx = data;
    ܐܢ (mmctx->quant_ib_first[ith] >= mmctx->quant_ib_last[ith]) {
        ܦܢܐ;
    }
    بنية htp_ops_context * octx = mmctx->octx;
    بنية htp_thread_trace * tr = &octx->ctx->trace[ith];
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_A_QUANT, mmctx->quant_ib_first[ith]);

    ثابت بنية htp_tensor * src = mmctx->act;

    quantize_f16_f16_tiled_block_kernel(
        (ثابت __fp16 *) mmctx->vtcm_act_raw,
        mmctx->vtcm_act,
        عدم,
        src->ne[0],
        mmctx->quant_ib_first[ith],
        mmctx->quant_ib_last[ith],
        mmctx->vtcm_act_raw_stride,
        htp_mm_f16_tiled_row_size(src->ne[0]),
        mmctx->quant_r[ith],
        mmctx->quant_c[ith]
    );

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_A_QUANT, mmctx->quant_ib_first[ith]);
}

// q8_1 for weight types with offsets (q8_1_s16 for Q2_K), f16_tiled for F16, otherwise q8_0
ساكن مضمن worker_callback_t htp_mm_act_quant_row_func(صحيح weight_type, صحيح act_type) {
    ܐܢ (weight_type == HTP_TYPE_F16) {
        ܦܢܐ (act_type == HTP_TYPE_F16) ? quantize_f16_f16_tiled : quantize_f32_f16_tiled;
    }
    ܐܢ (weight_type == HTP_TYPE_Q2_K) {
        ܦܢܐ quantize_f32_q8_1_s16_tiled;
    }
    ܦܢܐ htp_mm_weight_has_offset(weight_type) ? quantize_f32_q8_1_tiled : quantize_f32_q8_0_tiled;
}

ساكن مضمن worker_callback_t htp_mm_act_quant_block_func(صحيح weight_type, صحيح act_type) {
    ܐܢ (weight_type == HTP_TYPE_F16) {
        ܦܢܐ (act_type == HTP_TYPE_F16) ? quantize_f16_f16_tiled_block : quantize_f32_f16_tiled_block;
    }
    ܐܢ (weight_type == HTP_TYPE_Q2_K) {
        ܦܢܐ quantize_f32_q8_1_s16_tiled_block;
    }
    ܦܢܐ htp_mm_weight_has_offset(weight_type) ? quantize_f32_q8_1_tiled_block : quantize_f32_q8_0_tiled_block;
}

ساكن فراغ hvx_mm_4d_tiled(unsigned صحيح nth, unsigned صحيح ith, فراغ * data) {
    htp_matmul_preamble;

    ثابت uint32_t src0_nrows = mmctx->src0_row_end - mmctx->src0_row_start;
    ثابت uint32_t cur_m_rows = mmctx->cur_m_rows ? mmctx->cur_m_rows : (ne11 * ne12 * ne13);
    ثابت uint32_t cur_m_start = mmctx->cur_m_start;

    ثابت uint32_t src0_start_row  = mmctx->src0_row_start + src0_nrows_per_thread * ith;
    ثابت uint32_t src0_end_row    = MIN(src0_start_row + src0_nrows_per_thread, mmctx->src0_row_end);

    بنية htp_thread_trace * tr = &octx->ctx->trace[ith];

    ثابت بنية htp_mm_kernel_params * kparams = (ثابت بنية htp_mm_kernel_params *) octx->kernel_params;
    ثابت uint32_t n_prefetch = kparams->n_prefetch;
    أكد(n_prefetch >= 2 && n_prefetch <= HTP_MM_MAX_PREFETCH && (n_prefetch & (n_prefetch - 1)) == 0);

    ثابت size_t dst_row_size = nb1;
    ثابت size_t act_stride = mmctx->vtcm_act_stride;

    uint8_t * restrict vtcm_src0_ptr = mmctx->vtcm_src0 + mmctx->vtcm_src0_size_per_thread * ith;
    uint8_t * restrict act_data = mmctx->vtcm_act;

    ثابت uint32_t tile_size = htp_tensor_get_tile_size(src0->type);
    ثابت uint32_t aligned_tile_size = htp_tensor_get_tile_padded_size(src0->type);

    ثابت uint32_t n_k_tiles_w = ne00 / htp_tensor_get_tile_k(src0->type);
    ثابت uint32_t tile_row_stride = n_k_tiles_w * tile_size;
    ثابت uint32_t tile_row_transfer_size_aligned = n_k_tiles_w * aligned_tile_size;
    ثابت uint32_t src0_slice_stride = ((ne01 + 31) / 32) * tile_row_stride;

    ثابت uint32_t ct_start = src0_start_row / 32;
    ثابت uint32_t ct_end   = (src0_end_row + 31) / 32;

    ܐܢ (src0_start_row >= src0_end_row || cur_m_rows == 0) {
        ܦܢܐ;
    }

    ثابت uint32_t total_batches = ne12 * ne13;
    ثابت uint32_t b_start = fastdiv(cur_m_start, &kparams->div_ne1);
    uint32_t b_end = fastdiv(cur_m_start + cur_m_rows + ne11 - 1, &kparams->div_ne1);
    b_end = MIN(b_end, total_batches);

    uint32_t b_grp_start = b_start;
    ܟܡܐ (b_grp_start < b_end) {
        ثابت uint32_t b3 = fastdiv(b_grp_start, &kparams->div_ne12);
        ثابت uint32_t b2 = b_grp_start - b3 * ne12;
        ثابت uint32_t i02 = fastdiv(b2, &kparams->div_r2);
        ثابت uint32_t i03 = fastdiv(b3, &kparams->div_r3);

        uint32_t b_grp_end = b_grp_start + 1;
        ܟܡܐ (b_grp_end < b_end) {
            ثابت uint32_t cur_b3 = fastdiv(b_grp_end, &kparams->div_ne12);
            ثابت uint32_t cur_b2 = b_grp_end - cur_b3 * ne12;
            ثابت uint32_t cur_i02 = fastdiv(cur_b2, &kparams->div_r2);
            ثابت uint32_t cur_i03 = fastdiv(cur_b3, &kparams->div_r3);
            ܐܢ (cur_i02 != i02 || cur_i03 != i03) {
                اكسر;
            }
            b_grp_end++;
        }

        ثابت uint32_t slice_idx = i03 * ne02 + i02;
        ثابت dma_addr_t src0_slice = src0->data + (size_t) slice_idx * src0_slice_stride;

        uint32_t push_ct = ct_start;
        لكل (uint32_t d = 0; d < n_prefetch && push_ct < ct_end; d++, push_ct++) {
            dma_queue_push(dma_q, dma_make_data(vtcm_src0_ptr + d * tile_row_transfer_size_aligned,
                           src0_slice + (size_t) push_ct * tile_row_stride),
                           aligned_tile_size, tile_size, tile_size, n_k_tiles_w);
        }

        لكل (uint32_t ct = ct_start; ct < ct_end; ct++) {
            ثابت uint8_t * w_tile = (فراغ *) dma_queue_pop(dma_q).dst;

            صحيح valid_rows = (صحيح)ne0 - (صحيح)(ct * 32);
            valid_rows = MIN(32, MAX(0, valid_rows));

            htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, ct);
            لكل (uint32_t b = b_grp_start; b < b_grp_end; b++) {
                ثابت uint32_t b_m_start = b * ne11;
                ثابت uint32_t m_first = MAX(cur_m_start, b_m_start);
                ثابت uint32_t m_last  = MIN(cur_m_start + cur_m_rows, b_m_start + ne11);
                ܐܢ (m_first >= m_last) استمر;

                ثابت uint32_t cur_b3 = fastdiv(b, &kparams->div_ne12);
                ثابت uint32_t cur_b2 = b - cur_b3 * ne12;
                uint8_t * dst_batch_base = (uint8_t *) dst->data + (size_t) cur_b2 * nb2 + (size_t) cur_b3 * nb3;

                ثابت uint32_t chunk_m_offset = m_first - cur_m_start;
                ثابت uint32_t dst_m_offset   = m_first - b_m_start;
                ثابت uint32_t batch_nrows    = m_last - m_first;

                uint32_t ir1 = 0;
                لكل (; ir1 + 1 < batch_nrows; ir1 += 2) {
                    ثابت uint8_t * restrict act_col0 = (ثابت uint8_t *) (act_data + (chunk_m_offset + ir1 + 0) * act_stride);
                    ثابت uint8_t * restrict act_col1 = (ثابت uint8_t *) (act_data + (chunk_m_offset + ir1 + 1) * act_stride);
                    عائم * restrict dst_row0 = (عائم *) (dst_batch_base + (dst_m_offset + ir1 + 0) * dst_row_size);
                    عائم * restrict dst_row1 = (عائم *) (dst_batch_base + (dst_m_offset + ir1 + 1) * dst_row_size);
                    عائم * dst_ptr0 = &dst_row0[ct * 32];
                    عائم * dst_ptr1 = &dst_row1[ct * 32];
                    mmctx->vec_dot_32x2(ne10, dst_ptr0, dst_ptr1, w_tile, act_col0, act_col1, valid_rows, عدم, عدم);
                }
                لكل (; ir1 < batch_nrows; ++ir1) {
                    ثابت uint8_t * restrict act_col = (ثابت uint8_t *) (act_data + (chunk_m_offset + ir1) * act_stride);
                    عائم * restrict dst_row = (عائم *) (dst_batch_base + (dst_m_offset + ir1) * dst_row_size);
                    عائم * dst_ptr = &dst_row[ct * 32];
                    mmctx->vec_dot_32x1(ne10, dst_ptr, w_tile, act_col, valid_rows, عدم);
                }
            }
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ct);

            ܐܢ (push_ct < ct_end) {
                dma_queue_push(dma_q, dma_make_data(w_tile, src0_slice + (size_t) push_ct * tile_row_stride),
                               aligned_tile_size, tile_size, tile_size, n_k_tiles_w);
                push_ct++;
            }
        }
        b_grp_start = b_grp_end;
    }
    ܐܢ (src2) {
        hvx_tensor_add_f32_grid(dst, src2, cur_m_start, cur_m_start + cur_m_rows, src0_start_row, src0_end_row, &kparams->div_ne12_ne1, &kparams->div_ne1);
    }
}

ساكن فراغ hvx_mm_2d_flat(unsigned صحيح nth, unsigned صحيح ith, فراغ * data) {
    htp_matmul_preamble;

    ثابت بنية htp_mm_kernel_params * kparams = (ثابت بنية htp_mm_kernel_params *) octx->kernel_params;
    ثابت uint32_t n_prefetch = kparams->n_prefetch;
    أكد(n_prefetch >= 2 && n_prefetch <= HTP_MM_MAX_PREFETCH && (n_prefetch & (n_prefetch - 1)) == 0);
    ثابت uint32_t prefetch_mask = n_prefetch - 1;

    ثابت uint32_t src0_nrows = mmctx->src0_row_end - mmctx->src0_row_start;  // src0 rows
    ثابت uint32_t act_nrows = mmctx->cur_m_rows ? mmctx->cur_m_rows : mmctx->act_nrows;
    ثابت uint32_t cur_m_start = mmctx->cur_m_start;

    ثابت uint32_t src0_start_row  = mmctx->src0_row_start + src0_nrows_per_thread * ith;
    ثابت uint32_t src0_end_row    = MIN(src0_start_row + src0_nrows_per_thread, mmctx->src0_row_end);
    ثابت uint32_t src0_end_row_x2 = src0_start_row + ((src0_end_row - src0_start_row) & ~1U);

    بنية htp_thread_trace * tr = &octx->ctx->trace[ith];

    ثابت size_t dst_row_size  = nb1;
    ثابت size_t src0_row_size = nb01;
    ثابت size_t act_row_size = nb11;

    ثابت size_t src0_stride = mmctx->vtcm_src0_stride;
    ثابت size_t act_stride = mmctx->vtcm_act_stride;

    // Per-thread VTCMs for all tensors
    uint8_t * restrict vtcm_dst_ptr  = mmctx->vtcm_dst  + mmctx->vtcm_dst_size_per_thread  * ith;
    uint8_t * restrict vtcm_src0_ptr = mmctx->vtcm_src0 + mmctx->vtcm_src0_size_per_thread * ith;
    uint8_t * restrict act_data      = mmctx->vtcm_act;

    ثابت dma_addr_t src0_row = src0->data;


    // Prefill vtcm with src0 rows
    ܐܢ (src0_start_row < src0_end_row) {
        لكل (uint32_t ir0 = src0_start_row; ir0 < src0_end_row_x2; ir0 += 2) {
            ثابت صحيح is0 = (ir0 - src0_start_row);
            ܐܢ (is0 >= (صحيح)n_prefetch) {
                اكسر;
            }
            dma_queue_push(dma_q, dma_make_data(vtcm_src0_ptr + is0 * src0_stride, src0_row + ir0 * src0_row_size),
                           src0_stride, src0_row_size, src0_row_size, 2);
        }
    }

    ܐܢ (src0_start_row >= src0_end_row) {
        ܦܢܐ;
    }

    // Process src0 rows
    لكل (uint32_t ir0 = src0_start_row; ir0 < src0_end_row_x2; ir0 += 2) {
        ثابت uint8_t * ss0 = (فراغ *) dma_queue_pop(dma_q).dst;

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, ir0);
        // Process act columns in pairs (2x2 tiling)
        uint32_t ir1 = 0;
        لكل (; ir1 + 1 < act_nrows; ir1 += 2) {
            ثابت uint8_t * restrict act_col0 = (ثابت uint8_t *) (act_data + (ir1+0) * act_stride);
            ثابت uint8_t * restrict act_col1 = (ثابت uint8_t *) (act_data + (ir1+1) * act_stride);
            عائم * restrict dst_row0 = (عائم *) (dst->data + ((cur_m_start + ir1+0) * dst_row_size));
            عائم * restrict dst_row1 = (عائم *) (dst->data + ((cur_m_start + ir1+1) * dst_row_size));
            mmctx->vec_dot_2x2(ne00, &dst_row0[ir0], &dst_row1[ir0], ss0, ss0 + src0_stride, act_col0, act_col1);
        }

        // Handle remaining act rows (fallback to 2x1)
        لكل (; ir1 < act_nrows; ++ir1) {
            ثابت uint8_t * restrict act_col = (ثابت uint8_t *) (act_data + ir1 * act_stride);
            عائم * restrict dst_row          = (عائم *) (dst->data + ((cur_m_start + ir1) * dst_row_size));
            mmctx->vec_dot_2x1(ne00, &dst_row[ir0], ss0, ss0 + src0_stride, act_col);
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ir0);

        // Prefetch next (n + vtcm_nrows) row
        ثابت صحيح pr0 = (ir0 + n_prefetch);
        ثابت صحيح is0 = (pr0 - src0_start_row) & prefetch_mask;
        ܐܢ (pr0 < src0_end_row_x2) {
            dma_queue_push(dma_q, dma_make_data(vtcm_src0_ptr + is0 * src0_stride, src0_row + pr0 * src0_row_size),
                           src0_stride, src0_row_size, src0_row_size, 2);
        }
    }

    // Process the last row (if any)
    ܐܢ (src0_end_row != src0_end_row_x2) {
        uint32_t  ir0 = src0_end_row_x2;
        ثابت صحيح is0 = (ir0 - src0_start_row) & prefetch_mask;
        dma_queue_push(dma_q, dma_make_data(vtcm_src0_ptr + is0 * src0_stride, src0_row + ir0 * src0_row_size),
                       src0_stride, src0_row_size, src0_row_size, 1);
        ثابت uint8_t * ss0 = (فراغ *) dma_queue_pop(dma_q).dst;

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, ir0);
        #pragma unroll(2)
        لكل (uint32_t ir1 = 0; ir1 < act_nrows; ++ir1) {
            ثابت uint8_t * restrict act_col = (ثابت uint8_t *) (act_data + ir1 * act_stride);
            عائم * restrict dst_row          = (عائم *) (dst->data + ((cur_m_start + ir1) * dst_row_size));
            mmctx->vec_dot_1x1(ne00, &dst_row[ir0], ss0, act_col);
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ir0);
    }
    ܐܢ (src2) {
        hvx_tensor_add_f32_grid(dst, src2, cur_m_start, cur_m_start + act_nrows, src0_start_row, src0_end_row, &kparams->div_ne12_ne1, &kparams->div_ne1);
    }
}

ساكن فراغ hvx_mv_2d_flat(unsigned صحيح nth, unsigned صحيح ith, فراغ * data) {
    htp_matmul_preamble;

    ثابت uint32_t src0_nrows = mmctx->src0_row_end - mmctx->src0_row_start;

    ثابت uint32_t src0_start_row  = mmctx->src0_row_start + src0_nrows_per_thread * ith;
    ثابت uint32_t src0_end_row    = MIN(src0_start_row + src0_nrows_per_thread, mmctx->src0_row_end);

    بنية htp_thread_trace * tr = &octx->ctx->trace[ith];

    ثابت size_t dst_row_size  = nb1;
    ثابت size_t src0_row_size = nb01;
    ثابت size_t act_row_size  = nb11;

    ثابت size_t src0_stride = mmctx->vtcm_src0_stride;
    ثابت size_t act_stride  = mmctx->vtcm_act_stride;

    // Per-thread VTCMs for all tensors
    uint8_t * vtcm_dst_ptr  = mmctx->vtcm_dst  + mmctx->vtcm_dst_size_per_thread  * ith;
    uint8_t * vtcm_src0_ptr = mmctx->vtcm_src0 + mmctx->vtcm_src0_size_per_thread * ith;
    uint8_t * act_data      = mmctx->vtcm_act;

    عائم * tmp = (عائم *) vtcm_dst_ptr;

    ثابت dma_addr_t src0_row = src0->data;
    ثابت uint8_t * restrict act_col = (ثابت uint8_t *) act_data;
    عائم * restrict dst_col         = (عائم *) dst->data;

    ثابت uint32_t src0_end_row_x2 = src0_start_row + ((src0_end_row - src0_start_row) & ~1U);

    ثابت بنية htp_mm_kernel_params * kparams = (ثابت بنية htp_mm_kernel_params *) octx->kernel_params;
    ثابت uint32_t n_prefetch = kparams->n_prefetch;
    أكد(n_prefetch >= 2 && n_prefetch <= HTP_MM_MAX_PREFETCH && (n_prefetch & (n_prefetch - 1)) == 0);
    ثابت uint32_t prefetch_mask = n_prefetch - 1;

    // Prefill vtcm with 2x src0 rows
    ܐܢ (src0_start_row < src0_end_row) {
        ܐܢ (src2) {
            عائم * vtcm_bias_ptr = (عائم *) mmctx->vtcm_bias + src0_start_row;
            ثابت dma_addr_t src2_addr = src2->data + src0_start_row * حجم(عائم);
            صحيح slice_size = (صحيح)src0_end_row - (صحيح)src0_start_row;
            ܐܢ (slice_size > 0) {
                dma_queue_push(dma_q, dma_make_data(vtcm_bias_ptr, src2_addr),
                               slice_size * حجم(عائم), slice_size * حجم(عائم), slice_size * حجم(عائم), 1);
                dma_queue_pop_nowait(dma_q);
            }
        }
        لكل (uint32_t ir0 = src0_start_row; ir0 < src0_end_row_x2; ir0 += 2) {
            ثابت uint32_t is0 = (ir0 - src0_start_row);
            ܐܢ (is0 >= n_prefetch) {
                اكسر;
            }
            dma_queue_push(dma_q, dma_make_data(vtcm_src0_ptr + is0 * src0_stride, src0_row + ir0 * src0_row_size),
                           src0_stride, src0_row_size, src0_row_size, 2);
        }
    }


    ܐܢ (src0_start_row >= src0_end_row) {
        ܦܢܐ;
    }

    // Process src0 rows
    لكل (uint32_t ir0 = src0_start_row; ir0 < src0_end_row_x2; ir0 += 2) {
        ثابت uint8_t * ss0 = (فراغ *) dma_queue_pop(dma_q).dst;
        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, ir0);
        mmctx->vec_dot_2x1(ne00, &tmp[ir0 - src0_start_row], ss0, ss0 + src0_stride, act_col);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ir0);

        // Prefetch next (n + vtcm_nrows) row
        ثابت uint32_t pr0 = (ir0 + n_prefetch);
        ثابت uint32_t is0 = (pr0 - src0_start_row) & prefetch_mask;
        ܐܢ (pr0 < src0_end_row_x2) {
            dma_queue_push(dma_q, dma_make_data(vtcm_src0_ptr + is0 * src0_stride, src0_row + pr0 * src0_row_size),
                           src0_stride, src0_row_size, src0_row_size, 2);
        }
    }

    // Process the last row (if any)
    ܐܢ (src0_end_row != src0_end_row_x2) {
        ثابت uint32_t ir0 = src0_end_row_x2;
        ثابت uint32_t is0 = (ir0 - src0_start_row) & prefetch_mask;
        dma_queue_push(dma_q, dma_make_data(vtcm_src0_ptr + is0 * src0_stride, src0_row + ir0 * src0_row_size),
                       src0_stride, src0_row_size, src0_row_size, 1);
        ثابت uint8_t * ss0 = (فراغ *) dma_queue_pop(dma_q).dst;
        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, ir0);
        mmctx->vec_dot_1x1(ne00, &tmp[ir0 - src0_start_row], ss0, act_col);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ir0);
    }

    صحيح copy_cnt = src0_end_row - src0_start_row;
    ܐܢ (copy_cnt > 0) {
        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, src0_end_row);
        ܐܢ (src2) {
            hvx_add_f32_uuu((uint8_t *) &dst_col[src0_start_row],
                            (ثابت uint8_t *) tmp,
                            (ثابت uint8_t *) ((ثابت عائم *) mmctx->vtcm_bias + src0_start_row),
                            copy_cnt);
        } ܐܚܪܢܐ {
            hvx_copy_f32_ua((uint8_t *) &dst_col[src0_start_row], (uint8_t *) tmp, copy_cnt);
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, src0_end_row);
    }
}

ساكن فراغ hvx_mm_4d_flat(unsigned صحيح nth, unsigned صحيح ith, فراغ * data) {
    htp_matmul_preamble;

    ثابت بنية htp_mm_kernel_params * kparams = (ثابت بنية htp_mm_kernel_params *) octx->kernel_params;
    ثابت uint32_t n_prefetch = kparams->n_prefetch;
    أكد(n_prefetch >= 2 && n_prefetch <= HTP_MM_MAX_PREFETCH && (n_prefetch & (n_prefetch - 1)) == 0);
    ثابت uint32_t prefetch_mask = n_prefetch - 1;

    ثابت uint32_t src0_nrows = mmctx->src0_row_end - mmctx->src0_row_start;
    ثابت uint32_t cur_m_rows = mmctx->cur_m_rows ? mmctx->cur_m_rows : (ne11 * ne12 * ne13);
    ثابت uint32_t cur_m_start = mmctx->cur_m_start;

    ثابت uint32_t src0_start_row  = mmctx->src0_row_start + src0_nrows_per_thread * ith;
    ثابت uint32_t src0_end_row    = MIN(src0_start_row + src0_nrows_per_thread, mmctx->src0_row_end);
    ثابت uint32_t src0_end_row_x2 = src0_start_row + ((src0_end_row - src0_start_row) & ~1U);

    بنية htp_thread_trace * tr = &octx->ctx->trace[ith];

    ثابت size_t dst_row_size  = nb1;
    ثابت size_t src0_row_size = nb01;
    ثابت size_t src0_stride = mmctx->vtcm_src0_stride;
    ثابت size_t act_stride  = mmctx->vtcm_act_stride;

    uint8_t * restrict vtcm_src0_ptr = mmctx->vtcm_src0 + mmctx->vtcm_src0_size_per_thread * ith;
    uint8_t * restrict act_data      = mmctx->vtcm_act;


    ܐܢ (src0_start_row >= src0_end_row || cur_m_rows == 0) {
        ܦܢܐ;
    }

    ثابت uint32_t total_batches = ne12 * ne13;
    ثابت uint32_t b_start = fastdiv(cur_m_start, &kparams->div_ne1);
    uint32_t b_end = fastdiv(cur_m_start + cur_m_rows + ne11 - 1, &kparams->div_ne1);
    b_end = MIN(b_end, total_batches);

    uint32_t b_grp_start = b_start;
    ܟܡܐ (b_grp_start < b_end) {
        ثابت uint32_t b3 = fastdiv(b_grp_start, &kparams->div_ne12);
        ثابت uint32_t b2 = b_grp_start - b3 * ne12;
        ثابت uint32_t i02 = fastdiv(b2, &kparams->div_r2);
        ثابت uint32_t i03 = fastdiv(b3, &kparams->div_r3);

        uint32_t b_grp_end = b_grp_start + 1;
        ܟܡܐ (b_grp_end < b_end) {
            ثابت uint32_t cur_b3 = fastdiv(b_grp_end, &kparams->div_ne12);
            ثابت uint32_t cur_b2 = b_grp_end - cur_b3 * ne12;
            ثابت uint32_t cur_i02 = fastdiv(cur_b2, &kparams->div_r2);
            ثابت uint32_t cur_i03 = fastdiv(cur_b3, &kparams->div_r3);
            ܐܢ (cur_i02 != i02 || cur_i03 != i03) {
                اكسر;
            }
            b_grp_end++;
        }

        ثابت dma_addr_t src0_row = src0->data + ((size_t) i02 * nb02 + (size_t) i03 * nb03);

        لكل (uint32_t ir0 = src0_start_row; ir0 < src0_end_row_x2; ir0 += 2) {
            ثابت صحيح is0 = (ir0 - src0_start_row);
            ܐܢ (is0 >= (صحيح)n_prefetch) {
                اكسر;
            }
            dma_queue_push(dma_q, dma_make_data(vtcm_src0_ptr + is0 * src0_stride, src0_row + (size_t) ir0 * src0_row_size),
                           src0_stride, src0_row_size, src0_row_size, 2);
        }

        لكل (uint32_t ir0 = src0_start_row; ir0 < src0_end_row_x2; ir0 += 2) {
            ثابت uint8_t * ss0 = (فراغ *) dma_queue_pop(dma_q).dst;

            htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, ir0);
            لكل (uint32_t b = b_grp_start; b < b_grp_end; b++) {
                ثابت uint32_t b_m_start = b * ne11;
                ثابت uint32_t m_first = MAX(cur_m_start, b_m_start);
                ثابت uint32_t m_last  = MIN(cur_m_start + cur_m_rows, b_m_start + ne11);
                ܐܢ (m_first >= m_last) استمر;

                ثابت uint32_t cur_b3 = fastdiv(b, &kparams->div_ne12);
                ثابت uint32_t cur_b2 = b - cur_b3 * ne12;
                uint8_t * dst_batch_base = (uint8_t *) dst->data + (size_t) cur_b2 * nb2 + (size_t) cur_b3 * nb3;

                ثابت uint32_t chunk_m_offset = m_first - cur_m_start;
                ثابت uint32_t dst_m_offset   = m_first - b_m_start;
                ثابت uint32_t batch_nrows    = m_last - m_first;

                uint32_t ir1 = 0;
                لكل (; ir1 + 1 < batch_nrows; ir1 += 2) {
                    ثابت uint8_t * restrict act_col0 = (ثابت uint8_t *) (act_data + (chunk_m_offset + ir1 + 0) * act_stride);
                    ثابت uint8_t * restrict act_col1 = (ثابت uint8_t *) (act_data + (chunk_m_offset + ir1 + 1) * act_stride);
                    عائم * restrict dst_row0 = (عائم *) (dst_batch_base + (dst_m_offset + ir1 + 0) * dst_row_size);
                    عائم * restrict dst_row1 = (عائم *) (dst_batch_base + (dst_m_offset + ir1 + 1) * dst_row_size);
                    mmctx->vec_dot_2x2(ne00, &dst_row0[ir0], &dst_row1[ir0], ss0, ss0 + src0_stride, act_col0, act_col1);
                }
                لكل (; ir1 < batch_nrows; ++ir1) {
                    ثابت uint8_t * restrict act_col = (ثابت uint8_t *) (act_data + (chunk_m_offset + ir1) * act_stride);
                    عائم * restrict dst_row          = (عائم *) (dst_batch_base + (dst_m_offset + ir1) * dst_row_size);
                    mmctx->vec_dot_2x1(ne00, &dst_row[ir0], ss0, ss0 + src0_stride, act_col);
                }
            }
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ir0);

            ثابت صحيح pr0 = (ir0 + n_prefetch);
            ثابت صحيح is0 = (pr0 - src0_start_row) & prefetch_mask;
            ܐܢ (pr0 < src0_end_row_x2) {
                dma_queue_push(dma_q, dma_make_data(vtcm_src0_ptr + is0 * src0_stride, src0_row + (size_t) pr0 * src0_row_size),
                               src0_stride, src0_row_size, src0_row_size, 2);
            }
        }

        ܐܢ (src0_end_row != src0_end_row_x2) {
            uint32_t ir0 = src0_end_row_x2;
            ثابت صحيح is0 = (ir0 - src0_start_row) & prefetch_mask;
            dma_queue_push(dma_q, dma_make_data(vtcm_src0_ptr + is0 * src0_stride, src0_row + (size_t) ir0 * src0_row_size),
                           src0_stride, src0_row_size, src0_row_size, 1);
            ثابت uint8_t * ss0 = (فراغ *) dma_queue_pop(dma_q).dst;

            htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, ir0);
            لكل (uint32_t b = b_grp_start; b < b_grp_end; b++) {
                ثابت uint32_t b_m_start = b * ne11;
                ثابت uint32_t m_first = MAX(cur_m_start, b_m_start);
                ثابت uint32_t m_last  = MIN(cur_m_start + cur_m_rows, b_m_start + ne11);
                ܐܢ (m_first >= m_last) استمر;

                ثابت uint32_t cur_b3 = fastdiv(b, &kparams->div_ne12);
                ثابت uint32_t cur_b2 = b - cur_b3 * ne12;
                uint8_t * dst_batch_base = (uint8_t *) dst->data + (size_t) cur_b2 * nb2 + (size_t) cur_b3 * nb3;

                ثابت uint32_t chunk_m_offset = m_first - cur_m_start;
                ثابت uint32_t dst_m_offset   = m_first - b_m_start;
                ثابت uint32_t batch_nrows    = m_last - m_first;

                لكل (uint32_t ir1 = 0; ir1 < batch_nrows; ++ir1) {
                    ثابت uint8_t * restrict act_col = (ثابت uint8_t *) (act_data + (chunk_m_offset + ir1) * act_stride);
                    عائم * restrict dst_row          = (عائم *) (dst_batch_base + (dst_m_offset + ir1) * dst_row_size);
                    mmctx->vec_dot_1x1(ne00, &dst_row[ir0], ss0, act_col);
                }
            }
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ir0);
        }

        b_grp_start = b_grp_end;
    }

    ܐܢ (src2) {
        hvx_tensor_add_f32_grid(dst, src2, cur_m_start, cur_m_start + cur_m_rows, src0_start_row, src0_end_row, &kparams->div_ne12_ne1, &kparams->div_ne1);
    }
}

#define MMID_MATRIX_ROW(row_id, i1) matrix_rows[(row_id) * mmctx->mapping_stride + (i1)]

ساكن فراغ hvx_mm_id_tiled(unsigned صحيح nth, unsigned صحيح ith, فراغ * data) {
    htp_matmul_preamble;

    ثابت بنية htp_tensor * restrict ids = octx->src[2];

    ثابت uint32_t src0_nrows      = mmctx->src0_row_end - mmctx->src0_row_start;  // src0 rows per expert
    ثابت uint32_t act_nrows       = ne11;
    ثابت uint32_t src0_start_row  = mmctx->src0_row_start + src0_nrows_per_thread * ith;
    ثابت uint32_t src0_end_row    = MIN(src0_start_row + src0_nrows_per_thread, mmctx->src0_row_end);


    ܐܢ (src0_start_row >= src0_end_row) {
        ܦܢܐ;
    }

    بنية htp_thread_trace * tr = &octx->ctx->trace[ith];

    ثابت بنية htp_mm_kernel_params * kparams = (ثابت بنية htp_mm_kernel_params *) octx->kernel_params;
    ثابت uint32_t n_prefetch = kparams->n_prefetch;
    أكد(n_prefetch >= 2 && n_prefetch <= HTP_MM_MAX_PREFETCH && (n_prefetch & (n_prefetch - 1)) == 0);

    ثابت uint32_t n_ids = ids->ne[0];  // n_expert_used
    ثابت uint32_t n_as  = ne02;        // n_expert

    ثابت uint32_t *                matrix_row_counts = mmctx->matrix_row_counts;
    ثابت بنية mmid_row_mapping * matrix_rows       = mmctx->matrix_rows;

    ثابت size_t dst_row_size  = nb1;
    ثابت size_t act_row_size  = htp_mm_q8_0_tiled_row_size(ne10);

    ثابت size_t act_stride = mmctx->vtcm_act_stride;

    // Per-thread VTCMs for all tensors
    uint8_t * restrict vtcm_src0_ptr = mmctx->vtcm_src0 + mmctx->vtcm_src0_size_per_thread * ith;
    uint8_t * restrict act_data = mmctx->vtcm_act;

    لكل (uint32_t cur_a = 0; cur_a < n_as; ++cur_a) {
        ثابت int32_t cne1 = matrix_row_counts[cur_a];
        ܐܢ (cne1 == 0) {
            استمر;
        }

        ثابت dma_addr_t src0_row = src0->data + cur_a * nb02;

        ثابت uint32_t tile_size = htp_tensor_get_tile_size(src0->type);
        ثابت uint32_t aligned_tile_size = htp_tensor_get_tile_padded_size(src0->type);
        ثابت uint32_t n_k_tiles_w = ne00 / htp_tensor_get_tile_k(src0->type);
        ثابت uint32_t tile_row_stride = n_k_tiles_w * tile_size;
        ثابت uint32_t tile_row_transfer_size_aligned = n_k_tiles_w * aligned_tile_size;

        ثابت uint32_t ct_start = src0_start_row / 32;
        ثابت uint32_t ct_end   = (src0_end_row + 31) / 32;

        uint32_t push_ct = ct_start;
        لكل (uint32_t d = 0; d < n_prefetch && push_ct < ct_end; d++, push_ct++) {
            dma_queue_push(dma_q, dma_make_data(vtcm_src0_ptr + d * tile_row_transfer_size_aligned, src0_row + push_ct * tile_row_stride),
                           aligned_tile_size, tile_size, tile_size, n_k_tiles_w);
        }

        لكل (uint32_t ct = ct_start; ct < ct_end; ct++) {
            ثابت uint8_t * w_tile = (فراغ *) dma_queue_pop(dma_q).dst;

            صحيح valid_rows = (صحيح)ne0 - (صحيح)(ct * 32);
            valid_rows = MIN(32, MAX(0, valid_rows));

            htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, ct);
            لكل (uint32_t cid = 0; cid < cne1; ++cid) {
                بنية mmid_row_mapping row_mapping = MMID_MATRIX_ROW(cur_a, cid);
                ثابت صحيح               rm1         = row_mapping.i1;  // expert idx
                ثابت صحيح               rm2         = row_mapping.i2;  // token idx

                ثابت uint32_t ir1 = fastmodulo(rm1, ne11, &mmctx->mm_div_ne11);        // act row idx
                ثابت uint8_t * restrict act_col = (ثابت uint8_t *) (act_data + (ir1 + rm2 * ne11 + 0) * act_stride);
                عائم * restrict dst_row = (عائم *) (dst->data + (rm1 * nb1 + rm2 * nb2 + 0));

                mmctx->vec_dot_32x1(ne10, &dst_row[ct * 32], w_tile, act_col, valid_rows, عدم);
            }
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ct);

            ܐܢ (push_ct < ct_end) {
                dma_queue_push(dma_q, dma_make_data(w_tile, src0_row + push_ct * tile_row_stride),
                               aligned_tile_size, tile_size, tile_size, n_k_tiles_w);
                push_ct++;
            }
        }
    }
}

ساكن فراغ hvx_mv_id_tiled(unsigned صحيح nth, unsigned صحيح ith, فراغ * data) {
    htp_matmul_preamble;

    ثابت بنية htp_tensor * restrict ids = octx->src[2];

    ثابت uint32_t src0_nrows      = mmctx->src0_row_end - mmctx->src0_row_start;  // src0 rows per expert
    ثابت uint32_t src0_start_row  = mmctx->src0_row_start + src0_nrows_per_thread * ith;
    ثابت uint32_t src0_end_row    = MIN(src0_start_row + src0_nrows_per_thread, mmctx->src0_row_end);


    ܐܢ (src0_start_row >= src0_end_row) {
        ܦܢܐ;
    }

    بنية htp_thread_trace * tr = &octx->ctx->trace[ith];

    ثابت بنية htp_mm_kernel_params * kparams = (ثابت بنية htp_mm_kernel_params *) octx->kernel_params;
    ثابت uint32_t n_prefetch = kparams->n_prefetch;
    أكد(n_prefetch >= 2 && n_prefetch <= HTP_MM_MAX_PREFETCH && (n_prefetch & (n_prefetch - 1)) == 0);

    أكد(ne13 % ne03 == 0);

    ثابت size_t dst_row_size  = nb1;
    ثابت size_t act_row_size = htp_mm_q8_0_tiled_row_size(ne10);

    ثابت uint32_t n_aids = src2->ne[0];  // num activated experts
    ثابت uint32_t n_ids  = ne02;         // num experts

    // Per-thread VTCMs for all tensors
    uint8_t * restrict vtcm_src0_ptr = mmctx->vtcm_src0 + mmctx->vtcm_src0_size_per_thread * ith;
    uint8_t * restrict act_data = mmctx->vtcm_act;

    لكل (uint32_t ie1 = 0; ie1 < n_aids; ++ie1) {  // for each expert
        ثابت int32_t eid = *(ثابت int32_t *) ((ثابت uint8_t *) src2->data + ie1 * src2->nb[0]);
        ܐܢ (eid < 0) {
            استمر;
        }
        أكد(eid < (int32_t) n_ids);

        ثابت dma_addr_t src0_row = src0->data + eid * nb02;
        ثابت uint8_t * restrict act_col = (ثابت uint8_t *) act_data;
        عائم * restrict dst_row          = (عائم *) (dst->data + ie1 * nb1);

        ثابت uint32_t tile_size = htp_tensor_get_tile_size(src0->type);
        ثابت uint32_t aligned_tile_size = htp_tensor_get_tile_padded_size(src0->type);
        ثابت uint32_t n_k_tiles_w = ne00 / htp_tensor_get_tile_k(src0->type);
        ثابت uint32_t tile_row_stride = n_k_tiles_w * tile_size;
        ثابت uint32_t tile_row_transfer_size_aligned = n_k_tiles_w * aligned_tile_size;

        ثابت uint32_t ct_start = src0_start_row / 32;
        ثابت uint32_t ct_end   = (src0_end_row + 31) / 32;

        uint32_t push_ct = ct_start;
        لكل (uint32_t d = 0; d < n_prefetch && push_ct < ct_end; d++, push_ct++) {
            dma_queue_push(dma_q, dma_make_data(vtcm_src0_ptr + d * tile_row_transfer_size_aligned, src0_row + push_ct * tile_row_stride),
                           aligned_tile_size, tile_size, tile_size, n_k_tiles_w);
        }

        لكل (uint32_t ct = ct_start; ct < ct_end; ct++) {
            ثابت uint8_t * w_tile = (فراغ *) dma_queue_pop(dma_q).dst;

            صحيح valid_rows = (صحيح)ne0 - (صحيح)(ct * 32);
            valid_rows = MIN(32, MAX(0, valid_rows));

            htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, ct);
            mmctx->vec_dot_32x1(ne10, &dst_row[ct * 32], w_tile, act_col, valid_rows, عدم);
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ct);

            ܐܢ (push_ct < ct_end) {
                dma_queue_push(dma_q, dma_make_data(w_tile, src0_row + push_ct * tile_row_stride),
                               aligned_tile_size, tile_size, tile_size, n_k_tiles_w);
                push_ct++;
            }
        }
    }
}

ساكن فراغ hvx_mv_id_nx_tiled(unsigned صحيح nth, unsigned صحيح ith, فراغ * data) {
    بنية htp_mm_context * mmctx = (بنية htp_mm_context *) data;
    بنية htp_ops_context * octx = mmctx->octx;
    ثابت بنية htp_mm_kernel_params * kparams = (ثابت بنية htp_mm_kernel_params *) octx->kernel_params;
    ثابت uint32_t n_weights      = kparams->n_weights;
    ثابت بنية htp_tensor * restrict src0 = octx->src[0];
    ثابت بنية htp_tensor * restrict act  = octx->src[n_weights];
    ثابت بنية htp_tensor * restrict ids  = octx->src[n_weights + 1];


    بنية htp_thread_trace * tr = &octx->ctx->trace[ith];

    ثابت uint32_t n_prefetch = kparams->n_prefetch;
    أكد(n_prefetch >= 2 && n_prefetch <= HTP_MM_MAX_PREFETCH && (n_prefetch & (n_prefetch - 1)) == 0);

    ثابت uint32_t n_aids = ids->ne[0];
    ثابت uint32_t n_ids  = src0->ne[2];

    uint8_t * restrict vtcm_src0_ptr = mmctx->vtcm_src0 + mmctx->vtcm_src0_size_per_thread * ith;
    uint8_t * restrict act_data = mmctx->vtcm_act;

    لكل (uint32_t ie1 = 0; ie1 < n_aids; ++ie1) {
        ثابت int32_t eid = *(ثابت int32_t *) ((ثابت uint8_t *) ids->data + ie1 * ids->nb[0]);
        ܐܢ (eid < 0) استمر;
        أكد(eid < (int32_t) n_ids);

        لكل (uint32_t p = 0; p < n_weights; ++p) {
            ثابت بنية htp_tensor * restrict src_w = octx->src[p];
            ثابت بنية htp_tensor * restrict dst   = octx->dsts[p];
            ܐܢ (!src_w || !dst) استمر;
            dma_queue * dma_q = octx->ctx->dma[ith];

            ثابت uint32_t ne01 = src_w->ne[1];
            uint32_t start_row = 0;
            uint32_t end_row   = ne01;
            ܐܢ (octx->ctx->mdev.count > 1) {
                ثابت bool can_split = htp_tensor_can_row_partition(dst, حجم(عائم));
                ثابت بنية htp_tensor_mdev_range range = htp_tensor_mdev_partition(ne01, can_split ? 32 : 0, octx->ctx->mdev.idx, octx->ctx->mdev.count, &octx->ctx->mdev.count_div);
                start_row = range.start;
                end_row   = range.start + range.count;
            }

            ثابت uint32_t nrows = end_row - start_row;
            uint32_t src0_nrows_per_thread = fastdiv(nrows + nth - 1, &octx->n_threads_div);
            src0_nrows_per_thread = hex_round_up(src0_nrows_per_thread, 32);

            ثابت uint32_t src0_start_row = start_row + src0_nrows_per_thread * ith;
            ثابت uint32_t src0_end_row   = MIN(src0_start_row + src0_nrows_per_thread, end_row);
            ܐܢ (src0_start_row >= src0_end_row) استمر;

            ثابت dma_addr_t src0_row = src_w->data + eid * src_w->nb[2];
            ثابت uint8_t * restrict act_col = (ثابت uint8_t *) act_data;
            عائم * restrict dst_row = (عائم *) (dst->data + ie1 * dst->nb[1]);

            ثابت uint32_t tile_size = htp_tensor_get_tile_size(src_w->type);
            ثابت uint32_t aligned_tile_size = htp_tensor_get_tile_padded_size(src_w->type);
            ثابت uint32_t n_k_tiles_w = src_w->ne[0] / htp_tensor_get_tile_k(src_w->type);
            ثابت uint32_t tile_row_stride = n_k_tiles_w * tile_size;
            ثابت uint32_t tile_row_transfer_size_aligned = n_k_tiles_w * aligned_tile_size;

            ثابت uint32_t ct_start = src0_start_row / 32;
            ثابت uint32_t ct_end   = (src0_end_row + 31) / 32;

            uint32_t push_ct = ct_start;
            لكل (uint32_t d = 0; d < n_prefetch && push_ct < ct_end; d++, push_ct++) {
                dma_queue_push(dma_q, dma_make_data(vtcm_src0_ptr + d * tile_row_transfer_size_aligned, src0_row + push_ct * tile_row_stride),
                               aligned_tile_size, tile_size, tile_size, n_k_tiles_w);
            }

            لكل (uint32_t ct = ct_start; ct < ct_end; ct++) {
                ثابت uint8_t * w_tile = (فراغ *) dma_queue_pop(dma_q).dst;

                صحيح valid_rows = (صحيح)dst->ne[0] - (صحيح)(ct * 32);
                valid_rows = MIN(32, MAX(0, valid_rows));

                htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, ct);
                mmctx->vec_dot_32x1(act->ne[0], &dst_row[ct * 32], w_tile, act_col, valid_rows, عدم);
                htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ct);

                ܐܢ (push_ct < ct_end) {
                    dma_queue_push(dma_q, dma_make_data(w_tile, src0_row + push_ct * tile_row_stride),
                                    aligned_tile_size, tile_size, tile_size, n_k_tiles_w);
                    push_ct++;
                }
            }
        }
    }
}

ساكن فراغ hvx_mm_id_nx_tiled(unsigned صحيح nth, unsigned صحيح ith, فراغ * data) {
    بنية htp_mm_context * mmctx = (بنية htp_mm_context *) data;
    بنية htp_ops_context * octx = mmctx->octx;
    ثابت بنية htp_mm_kernel_params * kparams = (ثابت بنية htp_mm_kernel_params *) octx->kernel_params;
    ثابت uint32_t n_weights      = kparams->n_weights;
    ثابت بنية htp_tensor * restrict src0 = octx->src[0];
    ثابت بنية htp_tensor * restrict act  = octx->src[n_weights];
    ثابت بنية htp_tensor * restrict ids  = octx->src[n_weights + 1];


    بنية htp_thread_trace * tr = &octx->ctx->trace[ith];

    ثابت uint32_t n_prefetch = kparams->n_prefetch;
    أكد(n_prefetch >= 2 && n_prefetch <= HTP_MM_MAX_PREFETCH && (n_prefetch & (n_prefetch - 1)) == 0);

    ثابت uint32_t n_as = src0->ne[2];

    ثابت uint32_t * matrix_row_counts = mmctx->matrix_row_counts;
    ثابت بنية mmid_row_mapping * matrix_rows = mmctx->matrix_rows;

    ثابت size_t act_stride = mmctx->vtcm_act_stride;

    uint8_t * restrict vtcm_src0_ptr = mmctx->vtcm_src0 + mmctx->vtcm_src0_size_per_thread * ith;
    uint8_t * restrict act_data = mmctx->vtcm_act;

    لكل (uint32_t cur_a = 0; cur_a < n_as; ++cur_a) {
        ثابت int32_t cne1 = matrix_row_counts[cur_a];
        ܐܢ (cne1 == 0) استمر;

        لكل (uint32_t p = 0; p < n_weights; ++p) {
            ثابت بنية htp_tensor * restrict src_w = octx->src[p];
            ثابت بنية htp_tensor * restrict dst   = octx->dsts[p];
            ܐܢ (!src_w || !dst) استمر;
            dma_queue * dma_q = octx->ctx->dma[ith];

            ثابت uint32_t ne01 = src_w->ne[1];
            uint32_t start_row = 0;
            uint32_t end_row   = ne01;
            ܐܢ (octx->ctx->mdev.count > 1) {
                ثابت bool can_split = htp_tensor_can_row_partition(dst, حجم(عائم));
                ثابت بنية htp_tensor_mdev_range range = htp_tensor_mdev_partition(ne01, can_split ? 32 : 0, octx->ctx->mdev.idx, octx->ctx->mdev.count, &octx->ctx->mdev.count_div);
                start_row = range.start;
                end_row   = range.start + range.count;
            }

            ثابت uint32_t nrows = end_row - start_row;
            uint32_t src0_nrows_per_thread = fastdiv(nrows + nth - 1, &octx->n_threads_div);
            src0_nrows_per_thread = hex_round_up(src0_nrows_per_thread, 32);

            ثابت uint32_t src0_start_row = start_row + src0_nrows_per_thread * ith;
            ثابت uint32_t src0_end_row   = MIN(src0_start_row + src0_nrows_per_thread, end_row);
            ܐܢ (src0_start_row >= src0_end_row) استمر;

            ثابت dma_addr_t src0_row = src_w->data + cur_a * src_w->nb[2];

            ثابت uint32_t tile_size = htp_tensor_get_tile_size(src_w->type);
            ثابت uint32_t aligned_tile_size = htp_tensor_get_tile_padded_size(src_w->type);
            ثابت uint32_t n_k_tiles_w = src_w->ne[0] / htp_tensor_get_tile_k(src_w->type);
            ثابت uint32_t tile_row_stride = n_k_tiles_w * tile_size;
            ثابت uint32_t tile_row_transfer_size_aligned = n_k_tiles_w * aligned_tile_size;

            ثابت uint32_t ct_start = src0_start_row / 32;
            ثابت uint32_t ct_end   = (src0_end_row + 31) / 32;

            uint32_t push_ct = ct_start;
            لكل (uint32_t d = 0; d < n_prefetch && push_ct < ct_end; d++, push_ct++) {
                dma_queue_push(dma_q, dma_make_data(vtcm_src0_ptr + d * tile_row_transfer_size_aligned, src0_row + push_ct * tile_row_stride),
                               aligned_tile_size, tile_size, tile_size, n_k_tiles_w);
            }

            لكل (uint32_t ct = ct_start; ct < ct_end; ct++) {
                ثابت uint8_t * w_tile = (فراغ *) dma_queue_pop(dma_q).dst;

                صحيح valid_rows = (صحيح)dst->ne[0] - (صحيح)(ct * 32);
                valid_rows = MIN(32, MAX(0, valid_rows));

                htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, ct);
                لكل (uint32_t cid = 0; cid < (uint32_t) cne1; ++cid) {
                    بنية mmid_row_mapping row_mapping = MMID_MATRIX_ROW(cur_a, cid);
                    ثابت صحيح rm1 = row_mapping.i1;
                    ثابت صحيح rm2 = row_mapping.i2;

                    ثابت uint32_t ir1 = fastmodulo(rm1, act->ne[1], &mmctx->mm_div_ne11);
                    ثابت uint8_t * restrict act_col = (ثابت uint8_t *) (act_data + (ir1 + rm2 * act->ne[1]) * act_stride);
                    عائم * restrict dst_row = (عائم *) (dst->data + (rm1 * dst->nb[1] + rm2 * dst->nb[2]));

                    mmctx->vec_dot_32x1(act->ne[0], &dst_row[ct * 32], w_tile, act_col, valid_rows, عدم);
                }
                htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ct);

                ܐܢ (push_ct < ct_end) {
                    dma_queue_push(dma_q, dma_make_data(w_tile, src0_row + push_ct * tile_row_stride),
                                    aligned_tile_size, tile_size, tile_size, n_k_tiles_w);
                    push_ct++;
                }
            }
        }
    }
}

ساكن صحيح hvx_mm_init_tiled_vec_dot(بنية htp_mm_context * mmctx, تعداد htp_data_type type) {
    بدل (type) {
        حالة HTP_TYPE_Q4_0:
            mmctx->type         = "q4_0_tiled-f32";
            mmctx->vec_dot_32x1 = tiled_vec_dot_q4_0_32x1;
            mmctx->vec_dot_32x2 = tiled_vec_dot_q4_0_32x2;
            ܦܢܐ 0;
        حالة HTP_TYPE_Q4_1:
        حالة HTP_TYPE_Q4_K:
            mmctx->type         = "q4_1_tiled-f32";
            mmctx->vec_dot_32x1 = tiled_vec_dot_q4_1_32x1;
            mmctx->vec_dot_32x2 = tiled_vec_dot_q4_1_32x2;
            ܦܢܐ 0;
        حالة HTP_TYPE_Q8_0:
            mmctx->type         = "q8_0_tiled-f32";
            mmctx->vec_dot_32x1 = tiled_vec_dot_q8_0_32x1;
            mmctx->vec_dot_32x2 = tiled_vec_dot_q8_0_32x2;
            ܦܢܐ 0;
        حالة HTP_TYPE_Q5_K:
            mmctx->type         = "q5_k_tiled-f32";
            mmctx->vec_dot_32x1 = tiled_vec_dot_q5_k_32x1;
            mmctx->vec_dot_32x2 = tiled_vec_dot_q5_k_32x2;
            ܦܢܐ 0;
        حالة HTP_TYPE_Q6_K:
            mmctx->type         = "q6_k_tiled-f32";
            mmctx->vec_dot_32x1 = tiled_vec_dot_q6_k_32x1;
            mmctx->vec_dot_32x2 = tiled_vec_dot_q6_k_32x2;
            ܦܢܐ 0;
        حالة HTP_TYPE_Q3_K:
            mmctx->type         = "q3_k_tiled-f32";
            mmctx->vec_dot_32x1 = tiled_vec_dot_q3_k_32x1;
            mmctx->vec_dot_32x2 = tiled_vec_dot_q3_k_32x2;
            ܦܢܐ 0;
        حالة HTP_TYPE_Q2_K:
            mmctx->type         = "q2_k_tiled-f32";
            mmctx->vec_dot_32x1 = tiled_vec_dot_q2_k_32x1;
            mmctx->vec_dot_32x2 = tiled_vec_dot_q2_k_32x2;
            ܦܢܐ 0;
        حالة HTP_TYPE_Q2_0:
            mmctx->type         = "q2_0_tiled-f32";
            mmctx->vec_dot_32x1 = tiled_vec_dot_q2_0_32x1;
            mmctx->vec_dot_32x2 = tiled_vec_dot_q2_0_32x2;
            ܦܢܐ 0;
        حالة HTP_TYPE_IQ4_NL:
            mmctx->type         = "iq4nl_tiled-f32";
            mmctx->vec_dot_32x1 = tiled_vec_dot_iq4nl_32x1;
            mmctx->vec_dot_32x2 = tiled_vec_dot_iq4nl_32x2;
            ܦܢܐ 0;
        حالة HTP_TYPE_MXFP4:
            mmctx->type         = "mxfp4_tiled-f32";
            mmctx->vec_dot_32x1 = tiled_vec_dot_mxfp4_32x1;
            mmctx->vec_dot_32x2 = tiled_vec_dot_mxfp4_32x2;
            ܦܢܐ 0;
        حالة HTP_TYPE_Q1_0:
            mmctx->type         = "q1_0_tiled-f32";
            mmctx->vec_dot_32x1 = tiled_vec_dot_q1_0_32x1;
            mmctx->vec_dot_32x2 = tiled_vec_dot_q1_0_32x2;
            ܦܢܐ 0;
        حالة HTP_TYPE_F16:
            mmctx->type         = "f16_tiled-f16";
            mmctx->vec_dot_32x1 = tiled_vec_dot_f16_32x1;
            mmctx->vec_dot_32x2 = tiled_vec_dot_f16_32x2;
            ܦܢܐ 0;
        افتراضي:
            ܦܢܐ -1;
    }
}

ساكن صحيح hvx_mm_matmul(بنية htp_ops_context * octx) {
    htp_matmul_tensors_preamble;

    بنية htp_thread_trace * tr = &octx->ctx->trace[0];
    htp_trace_event_start(tr, HTP_TRACE_EVT_INIT, 0);

    بنية htp_mm_context mmctx_struct = {0};
    بنية htp_mm_context * mmctx = &mmctx_struct;
    mmctx->octx = octx;
    mmctx->act = act;

    ثابت بنية htp_mm_kernel_params * kparams = (ثابت بنية htp_mm_kernel_params *) octx->kernel_params;

    ثابت uint32_t src0_nrows = ne01;
    ثابت uint32_t act_nrows  = ne11 * ne12 * ne13;
    mmctx->act_nrows = act_nrows;

    uint32_t src0_row_start = 0;
    uint32_t src0_row_end   = src0_nrows;

    ܐܢ (octx->ctx->mdev.count > 1) {
        ثابت bool can_split = htp_tensor_can_row_partition(dst, حجم(عائم));
        ثابت بنية htp_tensor_mdev_range range = htp_tensor_mdev_partition(src0_nrows, can_split ? 32 : 0, octx->ctx->mdev.idx, octx->ctx->mdev.count, &octx->ctx->mdev.count_div);
        src0_row_start = range.start;
        src0_row_end   = range.start + range.count;
    }

    ܐܢ (src0_row_start >= src0_row_end) {
        ܦܢܐ HTP_STATUS_OK;
    }

    ثابت uint32_t nrows = src0_row_end - src0_row_start;
    mmctx->src0_row_start = src0_row_start;
    mmctx->src0_row_end   = src0_row_end;

    ثابت bool is_tiled = htp_tensor_is_tiled(src0);

    // Compute src0_nrows_per_thread
    mmctx->src0_nrows_per_thread  = fastdiv(nrows + octx->n_threads - 1, &octx->n_threads_div);
    ܐܢ (is_tiled) {
        mmctx->src0_nrows_per_thread = hex_round_up(mmctx->src0_nrows_per_thread, 32);
    } ܐܚܪܢܐ {
        mmctx->src0_nrows_per_thread += (mmctx->src0_nrows_per_thread & 1); // round up to even
    }

    ثابت size_t src0_row_size = nb01;
    ثابت size_t dst_row_size  = nb1;
    size_t       act_row_size  = nb11;

    ثابت size_t src0_row_size_padded = hex_round_up(src0_row_size, 128);

    worker_callback_t quant_task_func;
    worker_callback_t matmul_job_func;
    uint32_t n_quant_tasks = 1;
    ثابت bool is_batched = (ne12 > 1 || ne13 > 1 || ne02 > 1 || ne03 > 1);
    ܐܢ (is_batched) {
        matmul_job_func = is_tiled ? hvx_mm_4d_tiled : hvx_mm_4d_flat;
    } ܐܚܪܢܐ ܐܢ (act_nrows > 1) {
        matmul_job_func = is_tiled ? hvx_mm_2d_tiled : hvx_mm_2d_flat;
    } ܐܚܪܢܐ {
        matmul_job_func = is_tiled ? hvx_mv_2d_tiled : hvx_mv_2d_flat;
    }

    bool need_quant = true;

    بدل (kparams->kernel_type) {
        حالة HTP_MM_KERNEL_HVX_F16_F16_VTCM:
            quant_task_func        = (act->type == HTP_TYPE_F32) ? quantize_f32_f16 : quantize_f16_f16;
            need_quant             = (act->type == HTP_TYPE_F32);
            mmctx->type            = (act->type == HTP_TYPE_F32) ? "f32-f16" : "f16-f16";
            mmctx->vec_dot_1x1     = flat_vec_dot_f16_f16_aa_1x1;
            mmctx->vec_dot_2x1     = flat_vec_dot_f16_f16_aa_2x1;
            mmctx->vec_dot_2x2     = flat_vec_dot_f16_f16_aa_2x2;
            act_row_size           = kparams->act_row_size;
            اكسر;

        حالة HTP_MM_KERNEL_HVX_F32_F32_VTCM:
            quant_task_func        = عدم;
            need_quant             = false;
            mmctx->type            = "f32-f32";
            mmctx->vec_dot_1x1     = flat_vec_dot_f32_f32_aa_1x1;
            mmctx->vec_dot_2x1     = flat_vec_dot_f32_f32_aa_2x1;
            mmctx->vec_dot_2x2     = flat_vec_dot_f32_f32_aa_2x2;
            act_row_size           = kparams->act_row_size;
            اكسر;

        حالة HTP_MM_KERNEL_HVX_BF16_F32_VTCM:
            quant_task_func        = عدم;
            need_quant             = false;
            mmctx->type            = "bf16-f32";
            mmctx->vec_dot_1x1     = flat_vec_dot_bf16_f32_aa_1x1;
            mmctx->vec_dot_2x1     = flat_vec_dot_bf16_f32_aa_2x1;
            mmctx->vec_dot_2x2     = flat_vec_dot_bf16_f32_aa_2x2;
            act_row_size           = kparams->act_row_size;
            اكسر;

        حالة HTP_MM_KERNEL_HVX_QUANT_BLOCK:
        حالة HTP_MM_KERNEL_HVX_QUANT_ROW:
        افتراضي: {
            ܐܢ (hvx_mm_init_tiled_vec_dot(mmctx, src0->type) != 0) {
                ܦܢܐ HTP_STATUS_NO_SUPPORT;
            }

            ثابت uint32_t qk = QK_Q8_0_TILED;
            ثابت uint32_t nb = (ne10 + qk - 1) / qk;
            ثابت uint32_t total_nb = act_nrows * nb;

            ܐܢ (act_nrows < octx->n_threads && !is_batched) {
                n_quant_tasks = MIN(total_nb, octx->n_threads);
                quant_task_func = htp_mm_act_quant_block_func(src0->type, act->type);
                لكل (uint32_t ith = 0; ith < n_quant_tasks; ++ith) {
                    uint32_t ib_first = (total_nb * ith) / n_quant_tasks;
                    uint32_t ib_last  = (total_nb * (ith + 1)) / n_quant_tasks;
                    mmctx->quant_ib_first[ith] = ib_first;
                    mmctx->quant_ib_last[ith]  = ib_last;
                    mmctx->quant_r[ith]        = ib_first / nb;
                    mmctx->quant_c[ith]        = ib_first % nb;
                }
            } ܐܚܪܢܐ {
                n_quant_tasks = MIN(act_nrows, octx->n_threads);
                quant_task_func = htp_mm_act_quant_row_func(src0->type, act->type);
            }
            act_row_size = kparams->act_row_size;
            اكسر;
        }
    }

    ثابت uint32_t m_chunk = (kparams->m_chunk > 0 && (uint32_t) kparams->m_chunk < act_nrows)
                           ? (uint32_t) kparams->m_chunk : act_nrows;
    ثابت uint32_t m_layout_rows = m_chunk;

    بنية htp_mm_hvx_vtcm_layout L;
    htp_mm_hvx_vtcm_layout_build(&L, kparams->kernel_type, src0->type, ne10, m_layout_rows, octx->n_threads,
                                 dst_row_size, src0_row_size, act_row_size, src2 ? src2->nb[1] : 0, kparams->n_prefetch, false, false);

    ܐܢ (kparams->kernel_type == HTP_MM_KERNEL_HVX_F16_F16_VTCM ||
        kparams->kernel_type == HTP_MM_KERNEL_HVX_F32_F32_VTCM ||
        kparams->kernel_type == HTP_MM_KERNEL_HVX_BF16_F32_VTCM ||
        kparams->kernel_type == HTP_MM_KERNEL_HVX_QUANT_ROW ||
        kparams->kernel_type == HTP_MM_KERNEL_HVX_QUANT_BLOCK) {
        mmctx->vtcm_act_size_per_thread = L.act_bytes;
    } ܐܚܪܢܐ {
        mmctx->vtcm_act_size_per_thread = fastdiv(L.act_bytes, &octx->n_threads_div);
    }

    mmctx->vtcm_src0_size_per_thread = fastdiv(L.src0_bytes, &octx->n_threads_div);
    mmctx->vtcm_dst_size_per_thread  = fastdiv(L.dst_bytes, &octx->n_threads_div);

    ثابت size_t vtcm_size = L.total_bytes;

    FARF(HIGH, "matmul-%s : src0-vtcm-size %zu act-vtcm-size %zu dst-vtcm-size %zu (%zu)\n", mmctx->type,
         L.src0_bytes, L.act_bytes, L.dst_bytes, vtcm_size);

    FARF(HIGH, "matmul-%s : %ux%ux%ux%u * %ux%ux%ux%u-> %ux%ux%ux%u (0x%p, 0x%p, 0x%p)\n", mmctx->type, src0->ne[0],
         src0->ne[1], src0->ne[2], src0->ne[3], act->ne[0], act->ne[1], act->ne[2], act->ne[3], dst->ne[0],
         dst->ne[1], dst->ne[2], dst->ne[3], src0->data, act->data, dst->data);

    ܐܢ (octx->ctx->vtcm_size < vtcm_size) {
        FARF(ERROR, "matmul-%s : current VTCM reservation %zu is too small, needed %zu\n", mmctx->type,
             octx->ctx->vtcm_size, vtcm_size);
        ܦܢܐ HTP_STATUS_VTCM_TOO_SMALL;
    }

    uint8_t * ثابت base = (uint8_t *) octx->ctx->vtcm_base;
    mmctx->vtcm_act      = VTCM_LAYOUT_PTR(uint8_t, base, L.off_act);
    mmctx->vtcm_src0     = VTCM_LAYOUT_PTR(uint8_t, base, L.off_src0);
    mmctx->vtcm_bias     = VTCM_LAYOUT_PTR(uint8_t, base, L.off_bias);
    mmctx->vtcm_dst      = VTCM_LAYOUT_PTR(uint8_t, base, L.off_dst);
    mmctx->vtcm_act_raw  = VTCM_LAYOUT_PTR(uint8_t, base, L.off_act_raw);

    mmctx->vtcm_src0_stride = src0_row_size_padded;
    mmctx->vtcm_act_stride  = act_row_size;
    ܐܢ (kparams->kernel_type == HTP_MM_KERNEL_HVX_QUANT_BLOCK || kparams->kernel_type == HTP_MM_KERNEL_HVX_QUANT_ROW) {
        mmctx->vtcm_act_raw_stride = hex_round_up(ne10 * حجم(عائم), QK_Q8_0_TILED * حجم(عائم));
    } ܐܚܪܢܐ ܐܢ (kparams->kernel_type == HTP_MM_KERNEL_HVX_F16_F16_VTCM) {
        mmctx->vtcm_act_raw_stride = hex_round_up(ne10 * حجم(عائم), 128);
    } ܐܚܪܢܐ {
        mmctx->vtcm_act_raw_stride = 0;
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_INIT, 0);

    ܐܢ (kparams->m_chunk > 0 && (uint32_t) kparams->m_chunk < act_nrows) {
        لكل (uint32_t m_start = 0; m_start < act_nrows; m_start += m_chunk) {
            ثابت uint32_t cur_m_rows = MIN(act_nrows - m_start, m_chunk);
            mmctx->cur_m_start = m_start;
            mmctx->cur_m_rows  = cur_m_rows;

            ܐܢ (need_quant) {
                hvx_mm_transfer_act_dma(octx, kparams, act, mmctx->vtcm_act_raw, mmctx->vtcm_act_raw_stride, m_start, cur_m_rows);

                ثابت uint32_t qk = QK_Q8_0_TILED;
                ثابت uint32_t nb = (ne10 + qk - 1) / qk;
                ثابت uint32_t total_nb = cur_m_rows * nb;
                uint32_t quant_tasks;
                work_queue_func_t q_func;
                ܐܢ (cur_m_rows < octx->n_threads && (kparams->kernel_type == HTP_MM_KERNEL_HVX_QUANT_BLOCK || kparams->kernel_type == HTP_MM_KERNEL_HVX_QUANT_ROW)) {
                    quant_tasks = MIN(total_nb, octx->n_threads);
                    q_func = htp_mm_act_quant_block_func(src0->type, act->type);
                    لكل (uint32_t ith = 0; ith < quant_tasks; ++ith) {
                        uint32_t ib_first = (total_nb * ith) / quant_tasks;
                        uint32_t ib_last  = (total_nb * (ith + 1)) / quant_tasks;
                        mmctx->quant_ib_first[ith] = ib_first;
                        mmctx->quant_ib_last[ith]  = ib_last;
                        mmctx->quant_r[ith]        = ib_first / nb;
                        mmctx->quant_c[ith]        = ib_first % nb;
                    }
                } ܐܚܪܢܐ {
                    quant_tasks = MIN(cur_m_rows, octx->n_threads);
                    q_func = quant_task_func;
                    mmctx->n_quant_rows_per_thread = (cur_m_rows + quant_tasks - 1) / quant_tasks;
                }
                mmctx->n_quant_tasks = quant_tasks;
                work_queue_run(octx->ctx->work_queue, q_func, mmctx, quant_tasks);
            } ܐܚܪܢܐ {
                hvx_mm_transfer_act_dma(octx, kparams, act, mmctx->vtcm_act, mmctx->vtcm_act_stride, m_start, cur_m_rows);
            }

            work_queue_run(octx->ctx->work_queue, matmul_job_func, mmctx, octx->n_threads);
        }
    } ܐܚܪܢܐ {
        mmctx->cur_m_start = 0;
        mmctx->cur_m_rows  = act_nrows;

        ܐܢ (need_quant) {
            hvx_mm_transfer_act_dma(octx, kparams, act, mmctx->vtcm_act_raw, mmctx->vtcm_act_raw_stride, 0, act_nrows);
            mmctx->n_quant_rows_per_thread = (act_nrows + n_quant_tasks - 1) / n_quant_tasks;
            mmctx->n_quant_tasks = n_quant_tasks;
            work_queue_run(octx->ctx->work_queue, quant_task_func, mmctx, n_quant_tasks);
        } ܐܚܪܢܐ {
            hvx_mm_transfer_act_dma(octx, kparams, act, mmctx->vtcm_act, mmctx->vtcm_act_stride, 0, act_nrows);
        }

        work_queue_run(octx->ctx->work_queue, matmul_job_func, mmctx, octx->n_threads);
    }

    ܦܢܐ HTP_STATUS_OK;
}

ساكن فراغ dequantize_tiled_worker_loop(unsigned صحيح n, unsigned صحيح i, فراغ * data) {
    tiled_dequantize_state_t * state = (tiled_dequantize_state_t *) data;
    بنية htp_thread_trace * tr = &state->traces[i];
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_W_DEQUANT, i);
    لكل (unsigned صحيح task_id = i; task_id < (unsigned صحيح) state->n_tasks; task_id += n) {
        صحيح start = task_id * state->n_tiles_per_task;
        صحيح end   = hex_smin(start + state->n_tiles_per_task, state->n_tot_tiles);
        state->task_fn(state, start, end);
    }
    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_W_DEQUANT, i);
}

ساكن فراغ transfer_output_chunk_worker_fn(unsigned صحيح n, unsigned صحيح i, فراغ *data) {
    output_transfer_task_state_t *st = (output_transfer_task_state_t *) data;

    بنية htp_thread_trace * tr = &st->traces[i];

    صحيح start_chunk_idx = i * st->n_chunks_per_task;
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_O_PROC, start_chunk_idx);

    لكل (unsigned صحيح task_id = i; task_id < (unsigned صحيح)st->n_tasks; task_id += n) {
        صحيح    chunk_idx  = task_id * st->n_chunks_per_task;
        size_t chunk_size = hex_smin(st->n_tot_chunks - chunk_idx, st->n_chunks_per_task);

        عائم        *dst      = st->dst      + chunk_idx * st->dst_stride;
        ثابت عائم  *src2     = st->src2     ? (st->src2     + chunk_idx * st->src2_stride) : عدم;
        transfer_output_chunk_fp16_to_fp32(dst, src2, st->vtcm_src, chunk_idx, chunk_size, st->n_cols, st->dst_stride, st->src2_stride, st->dst_cols);
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_O_PROC, start_chunk_idx);
}

تعريفنوع بنية {
    بنية htp_context            * ctx;
    بنية htp_thread_trace       * traces;
    __fp16                        * dst;
    dma_addr_t                      act_dma_addr;
    عائم                         * vtcm_f32_act;
    uint32_t                        n_tasks;
    uint32_t                        n_tot_chunks;
    uint32_t                        n_chunks_per_task;
    uint32_t                        k_block;
    uint32_t                        k_stride;
    uint32_t                        k_valid;
    size_t                          vtcm_f32_act_bytes_per_thread;
    uint32_t                        dma_step_rows;
    uint32_t                        dma_step_rows_shift;
    uint32_t                        act_elem_size;  // 4=F32 src1, 2=F16 src1
} activation_transfer_task_state_t;

تعريفنوع بنية {
    بنية htp_context            * ctx;
    بنية htp_thread_trace       * traces;
    __fp16                        * dst;
    dma_addr_t                      act_dma_addr;
    عائم                         * vtcm_f32_act;
    uint32_t                        n_rows;
    uint32_t                        k_block;
    uint32_t                        k_stride;
    uint32_t                        k_valid;
    uint32_t                        n_col_chunks;
    بنية fastdiv_values           n_threads_div;
    size_t                          vtcm_f32_act_bytes;
    uint32_t                        dma_step_rows;
    uint32_t                        dma_step_rows_shift;
} activation_transfer_col_chunk_state_t;

ساكن فراغ transfer_activation_chunk_fp32_to_fp16_dma_pipelined_col_chunk(
        dma_queue *dma_q,
        __fp16 *restrict vtcm_dst,
        dma_addr_t act_dma_addr,
        uint32_t n_rows,
        uint32_t k_block,
        uint32_t k_stride,
        uint32_t k_chunk_valid,
        uint32_t c_first,
        uint32_t c_len,
        عائم *thread_f32_act,
        بنية htp_thread_trace *tr,
        uint32_t dma_step_rows,
        uint32_t dma_step_rows_shift) {

    ثابت uint32_t R = dma_step_rows;
    ثابت uint32_t n_rows_padded = hex_align_up(n_rows, HTP_MM_HMX_TILE_N_ROWS);

    ثابت uint32_t n_steps = n_rows_padded >> dma_step_rows_shift;

    // Push step 0
    ܐܢ (n_steps > 0 && n_rows > 0) {
        uint32_t nrows_to_fetch = hex_smin(n_rows, R);
        dma_queue_push(dma_q, dma_make_data(thread_f32_act, act_dma_addr + (size_t) c_first * حجم(عائم)),
                       c_len * حجم(عائم), k_stride * حجم(عائم), k_chunk_valid * حجم(عائم), nrows_to_fetch);
    }
    // Push step 1
    ܐܢ (n_steps > 1) {
        uint32_t next_r = R * 1;
        ܐܢ (next_r < n_rows) {
            uint32_t nrows_to_fetch = hex_smin(n_rows - next_r, R);
            عائم *next_buf = thread_f32_act + 1 * R * c_len;
            dma_queue_push(dma_q, dma_make_data(next_buf, act_dma_addr + ((size_t) next_r * k_stride + c_first) * حجم(عائم)),
                           c_len * حجم(عائم), k_stride * حجم(عائم), k_chunk_valid * حجم(عائم), nrows_to_fetch);
        }
    }
    لكل (uint32_t s = 0; s < n_steps; ++s) {
        uint32_t r = s << dma_step_rows_shift;
        عائم *curr_buf = thread_f32_act;

        ܐܢ (r < n_rows) {
            curr_buf = (عائم *) dma_queue_pop(dma_q).dst;
        }

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_A_PREP, r);
        لكل (uint32_t p = 0; p < (R >> 1); ++p) {
            uint32_t row_idx = r + (p << 1);
            عائم *pair_buf = curr_buf + (p << 1) * c_len;
            bool r0_valid = ((row_idx + 0) < n_rows);
            bool r1_valid = ((row_idx + 1) < n_rows);

            transfer_activation_row_pair_fp32_to_fp16_col_chunk(
                vtcm_dst, pair_buf, pair_buf + c_len, row_idx, k_block, c_first, c_len, k_chunk_valid, r0_valid, r1_valid
            );
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_A_PREP, r);

        // Push step s + 2
        uint32_t next_s = s + 2;
        uint32_t next_r = next_s << dma_step_rows_shift;
        ܐܢ (next_r < n_rows) {
            uint32_t nrows_to_fetch = hex_smin(n_rows - next_r, R);
            dma_queue_push(dma_q, dma_make_data(curr_buf, act_dma_addr + ((size_t) next_r * k_stride + c_first) * حجم(عائم)),
                           c_len * حجم(عائم), k_stride * حجم(عائم), k_chunk_valid * حجم(عائم), nrows_to_fetch);
        }
    }
}

ساكن فراغ transfer_activation_chunk_col_chunk_worker_fn(unsigned صحيح n, unsigned صحيح i, فراغ *data) {
    activation_transfer_col_chunk_state_t *st = (activation_transfer_col_chunk_state_t *) data;
    بنية htp_thread_trace * tr = &st->traces[i];

    uint32_t n_blocks = st->k_block / 32;
    uint32_t b_first = fastdiv(n_blocks * i, &st->n_threads_div);
    uint32_t b_last  = fastdiv(n_blocks * (i + 1), &st->n_threads_div);
    uint32_t c_first = b_first * 32;
    uint32_t c_last = b_last * 32;
    uint32_t c_len = c_last - c_first;

    ܐܢ (c_len == 0) {
        ܦܢܐ;
    }

    uint32_t k_chunk_valid = 0;
    ܐܢ (st->k_valid > c_first) {
        k_chunk_valid = hex_smin(st->k_valid, c_last) - c_first;
    }

    __fp16 *dst = st->dst;

    size_t thread_scratch_bytes = hex_align_down(fastdiv(st->vtcm_f32_act_bytes, &st->n_threads_div), 128);
    عائم *thread_f32_act = (عائم *)((حرف *)st->vtcm_f32_act + i * thread_scratch_bytes);

    transfer_activation_chunk_fp32_to_fp16_dma_pipelined_col_chunk(
        st->ctx->dma[i], dst, st->act_dma_addr, st->n_rows, st->k_block, st->k_stride, k_chunk_valid,
        c_first, c_len, thread_f32_act, tr, st->dma_step_rows, st->dma_step_rows_shift
    );
}

ساكن فراغ transfer_activation_chunk_to_fp16_dma_pipelined(
        dma_queue *dma_q,
        __fp16 *restrict vtcm_dst,
        dma_addr_t act_dma_addr,
        uint32_t n_rows,
        uint32_t k_block,
        uint32_t k_stride,
        uint32_t k_valid,
        uint8_t *thread_act,
        uint32_t act_elem_size,
        بنية htp_thread_trace *tr,
        uint32_t dma_step_rows,
        uint32_t dma_step_rows_shift) {

    ثابت uint32_t R = dma_step_rows;
    ثابت uint32_t n_rows_padded = hex_align_up(n_rows, HTP_MM_HMX_TILE_N_ROWS);

    ثابت uint32_t n_steps = n_rows_padded >> dma_step_rows_shift;

    ثابت bool act_is_f16 = (act_elem_size == حجم(__fp16));
    ثابت size_t row_bytes = (size_t) k_block * act_elem_size;  // staging row (DMA dst stride)
    ثابت size_t src_row_bytes = (size_t) k_stride * act_elem_size;  // DDR row (DMA src stride)
    ثابت size_t width_bytes = (size_t) k_valid * act_elem_size;

    // Push step 0
    ܐܢ (n_steps > 0 && n_rows > 0) {
        uint32_t nrows_to_fetch = hex_smin(n_rows, R);
        dma_queue_push(dma_q, dma_make_data(thread_act, act_dma_addr),
                       row_bytes, src_row_bytes, width_bytes, nrows_to_fetch);
    }
    // Push step 1 (if valid)
    ܐܢ (n_steps > 1) {
        uint32_t next_r = R * 1;
        ܐܢ (next_r < n_rows) {
            uint32_t nrows_to_fetch = hex_smin(n_rows - next_r, R);
            uint8_t *next_buf = thread_act + 1 * R * row_bytes;
            dma_queue_push(dma_q, dma_make_data(next_buf, act_dma_addr + (size_t) next_r * src_row_bytes),
                           row_bytes, src_row_bytes, width_bytes, nrows_to_fetch);
        }
    }
    لكل (uint32_t s = 0; s < n_steps; ++s) {
        uint32_t r = s << dma_step_rows_shift;
        uint8_t *curr_buf = thread_act;

        ܐܢ (r < n_rows) {
            curr_buf = (uint8_t *) dma_queue_pop(dma_q).dst;
        }

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_A_PREP, r);
        // Two copies of the pair loop so the type is resolved once per step and the
        // row-pair kernels stay direct (inlinable) calls.
        ܐܢ (act_is_f16) {
            لكل (uint32_t p = 0; p < (R >> 1); ++p) {
                uint32_t row_idx = r + (p << 1);
                ثابت __fp16 *pair_buf = (ثابت __fp16 *) (curr_buf + (p << 1) * row_bytes);
                bool r0_valid = ((row_idx + 0) < n_rows);
                bool r1_valid = ((row_idx + 1) < n_rows);

                transfer_activation_row_pair_f16_to_f16(vtcm_dst, pair_buf, pair_buf + k_block, row_idx, k_block, k_valid, r0_valid, r1_valid);
            }
        } ܐܚܪܢܐ {
            لكل (uint32_t p = 0; p < (R >> 1); ++p) {
                uint32_t row_idx = r + (p << 1);
                ثابت عائم *pair_buf = (ثابت عائم *) (curr_buf + (p << 1) * row_bytes);
                bool r0_valid = ((row_idx + 0) < n_rows);
                bool r1_valid = ((row_idx + 1) < n_rows);

                transfer_activation_row_pair_fp32_to_fp16(vtcm_dst, pair_buf, pair_buf + k_block, row_idx, k_block, k_valid, r0_valid, r1_valid);
            }
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_A_PREP, r);

        // Push step s + 2
        uint32_t next_s = s + 2;
        uint32_t next_r = next_s << dma_step_rows_shift;
        ܐܢ (next_r < n_rows) {
            uint32_t nrows_to_fetch = hex_smin(n_rows - next_r, R);
            dma_queue_push(dma_q, dma_make_data(curr_buf, act_dma_addr + (size_t) next_r * src_row_bytes),
                           row_bytes, src_row_bytes, width_bytes, nrows_to_fetch);
        }
    }
}

ساكن فراغ transfer_activation_chunk_worker_fn(unsigned صحيح n, unsigned صحيح i, فراغ *data) {
    activation_transfer_task_state_t *st = (activation_transfer_task_state_t *) data;

    بنية htp_thread_trace * tr = &st->traces[i];

    لكل (unsigned صحيح task_id = i; task_id < (unsigned صحيح)st->n_tasks; task_id += n) {
        صحيح    chunk_idx  = task_id * st->n_chunks_per_task;
        size_t chunk_size = hex_smin(st->n_tot_chunks - chunk_idx, st->n_chunks_per_task);

        __fp16      *dst = st->dst + chunk_idx * st->k_block;
        ثابت dma_addr_t act_dma_addr = st->act_dma_addr + (size_t) chunk_idx * st->k_stride * st->act_elem_size;

        uint8_t *thread_act = (uint8_t *) st->vtcm_f32_act + i * st->vtcm_f32_act_bytes_per_thread;
        transfer_activation_chunk_to_fp16_dma_pipelined(
            st->ctx->dma[i], dst, act_dma_addr, chunk_size, st->k_block, st->k_stride, st->k_valid, thread_act, st->act_elem_size, tr, st->dma_step_rows, st->dma_step_rows_shift
        );
    }
}

تعريفنوع بنية {
    بنية htp_thread_trace       * traces;
    ثابت بنية mmid_row_mapping * matrix_rows;
    __fp16                        * dst;
    ثابت عائم                   * src;
    uint32_t                        n_tasks;
    uint32_t                        n_tot_chunks;
    uint32_t                        n_chunks_per_task;
    uint32_t                        k_block;
    uint32_t                        cur_a;
    uint32_t                        mapping_stride;
    uint32_t                        ne11;
    بنية fastdiv_values           ne11_div;
    size_t                          nb11;
    size_t                          nb12;
    uint32_t                        start_row;
    uint32_t                        cne1;
    uint32_t                        k_valid;
} activation_transfer_gathered_task_state_t;

تعريفنوع بنية {
    بنية htp_thread_trace       * traces;
    ثابت بنية mmid_row_mapping * matrix_rows;
    ثابت __fp16                  * vtcm_src;
    عائم                         * dst;
    uint32_t                        n_tasks;
    uint32_t                        n_tot_chunks;
    uint32_t                        n_chunks_per_task;
    uint32_t                        n_cols;
    uint32_t                        cur_a;
    uint32_t                        mapping_stride;
    size_t                          dst_nb1;
    size_t                          dst_nb2;
    uint32_t                        start_row;
    uint32_t                        cne1;
} output_transfer_scattered_task_state_t;

ساكن فراغ transfer_activation_chunk_gathered_worker_fn(unsigned صحيح n, unsigned صحيح i, فراغ *data) {
    activation_transfer_gathered_task_state_t *st = data;
    بنية htp_thread_trace * tr = &st->traces[i];
    صحيح chunk_idx      = i;
    صحيح chunk_size     = st->n_chunks_per_task;
    صحيح vtcm_start_row = chunk_idx * chunk_size;
    صحيح start_row      = st->start_row + vtcm_start_row;
    صحيح n_rows         = hex_smin(st->cne1 - start_row, chunk_size);
    ܐܢ (n_rows > 0) {
        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_A_PREP, chunk_idx);
        transfer_activation_chunk_fp32_to_fp16_gathered(
            st->dst, st->src, start_row, vtcm_start_row, n_rows, st->k_block,
            st->matrix_rows, st->cur_a, st->mapping_stride,
            st->ne11, &st->ne11_div, st->nb11, st->nb12, st->cne1, st->k_valid);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_A_PREP, chunk_idx);
    }
}

ساكن فراغ transfer_activation_chunk_gathered_worker_flat_fn(unsigned صحيح n, unsigned صحيح i, فراغ *data) {
    activation_transfer_gathered_task_state_t *st = data;
    بنية htp_thread_trace * tr = &st->traces[i];
    صحيح chunk_idx = i;
    صحيح chunk_size = st->n_chunks_per_task;
    صحيح vtcm_start_row = chunk_idx * chunk_size;
    صحيح start_row = st->start_row + vtcm_start_row;
    صحيح n_rows = hex_smin(st->cne1 - start_row, chunk_size);
    ܐܢ (n_rows > 0) {
        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_A_PREP, chunk_idx);
        transfer_activation_chunk_fp32_to_fp16_gathered_flat(
            st->dst, st->src, start_row, vtcm_start_row, n_rows, st->k_block,
            st->matrix_rows, st->cur_a, st->mapping_stride,
            st->nb12, st->cne1, st->k_valid);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_A_PREP, chunk_idx);
    }
}

ساكن فراغ transfer_output_chunk_scattered_worker_fn(unsigned صحيح n, unsigned صحيح i, فراغ *data) {
    output_transfer_scattered_task_state_t *st = data;
    بنية htp_thread_trace * tr = &st->traces[i];
    صحيح chunk_idx = i;
    صحيح chunk_size = st->n_chunks_per_task;
    صحيح vtcm_start_row = chunk_idx * chunk_size;
    صحيح start_row = st->start_row + vtcm_start_row;
    صحيح n_rows = hex_smin(st->cne1 - start_row, chunk_size);
    ܐܢ (n_rows > 0) {
        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_O_PROC, chunk_idx);
        transfer_output_chunk_fp16_to_fp32_scattered(
            st->dst, st->vtcm_src, start_row, vtcm_start_row, n_rows, st->n_cols,
            st->matrix_rows, st->cur_a, st->mapping_stride,
            st->dst_nb1, st->dst_nb2, st->cne1);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_O_PROC, chunk_idx);
    }
}

// --- HMX Dispatchers & Entry Points ---

ساكن مضمن tiled_dequant_task_fn_t hmx_get_dequant_worker_fn(صحيح weight_type, bool is_tiled) {
    بدل (weight_type) {
        حالة HTP_TYPE_Q4_0:   ܦܢܐ dequantize_tiled_weight_to_fp16_task_q4_0;
        حالة HTP_TYPE_IQ4_NL: ܦܢܐ dequantize_tiled_weight_to_fp16_task_iq4_nl;
        حالة HTP_TYPE_Q4_1:
        حالة HTP_TYPE_Q4_K:   ܦܢܐ dequantize_tiled_weight_to_fp16_task_q4_1;
        حالة HTP_TYPE_MXFP4:  ܦܢܐ dequantize_tiled_weight_to_fp16_task_mxfp4;
        حالة HTP_TYPE_Q8_0:   ܦܢܐ dequantize_tiled_weight_to_fp16_task_q8_0;
        حالة HTP_TYPE_Q5_K:   ܦܢܐ dequantize_tiled_weight_to_fp16_task_q5_k;
        حالة HTP_TYPE_Q6_K:   ܦܢܐ dequantize_tiled_weight_to_fp16_task_q6_k;
        حالة HTP_TYPE_Q3_K:   ܦܢܐ dequantize_tiled_weight_to_fp16_task_q3_k;
        حالة HTP_TYPE_Q2_K:   ܦܢܐ dequantize_tiled_weight_to_fp16_task_q2_k;
        حالة HTP_TYPE_Q1_0:   ܦܢܐ dequantize_tiled_weight_to_fp16_task_q1_0;
        حالة HTP_TYPE_Q2_0:   ܦܢܐ dequantize_tiled_weight_to_fp16_task_q2_0;
        حالة HTP_TYPE_F16:    ܦܢܐ is_tiled ? عدم : convert_f16_weight_to_fp16_tiles_task;
        حالة HTP_TYPE_BF16:   ܦܢܐ convert_bf16_weight_to_fp16_tiles_task;
        حالة HTP_TYPE_F32:    ܦܢܐ quantize_f32_weight_to_fp16_tiles_task;
        افتراضي:              ܦܢܐ عدم;
    }
}

ساكن فراغ dequantize_tiled_weight_chunk_to_fp16_tiles(
        بنية htp_context *ctx, __fp16 *vtcm_dst,
        ثابت فراغ *weight_src_ddr,
        صحيح n_cols, صحيح k_block,
        size_t row_stride, صحيح weight_type,
        صحيح n_k_tiles, بنية fastdiv_values n_k_tiles_div,
        tiled_dequant_task_fn_t dequant_worker_fn, صحيح n_threads) {

    ܐܢ (!dequant_worker_fn) {
        ܦܢܐ;
    }

    أكد(k_block % HTP_MM_HMX_TILE_N_COLS == 0);

    ثابت size_t n_col_tiles = hmx_ceil_div(n_cols, HTP_MM_HMX_TILE_N_COLS);
    ثابت uint32_t n_weight_tiles = k_block / htp_tensor_get_tile_k(weight_type);
    ثابت size_t n_tot_tiles = n_col_tiles * n_weight_tiles;

    size_t n_tiles_per_task = (n_threads == 1) ? n_tot_tiles : hmx_ceil_div(n_tot_tiles, n_threads);

    tiled_dequantize_state_t state;
    state.n_tasks          = (n_tot_tiles + n_tiles_per_task - 1) / n_tiles_per_task;
    state.n_tot_tiles      = n_tot_tiles;
    state.n_tiles_per_task = n_tiles_per_task;
    state.dst              = vtcm_dst;
    state.src              = (ثابت uint8_t *)weight_src_ddr;
    state.task_fn          = dequant_worker_fn;
    state.n_cols           = n_cols;
    state.k_block          = k_block;
    state.row_stride       = row_stride;
    state.weight_type      = weight_type;
    state.n_k_tiles        = n_k_tiles;
    state.n_weight_tiles   = n_weight_tiles;
    state.n_k_tiles_div    = n_k_tiles_div;
    state.traces           = ctx->trace;
    state.ctx              = ctx;

    state.tile_size = htp_tensor_get_tile_size(weight_type);
    state.aligned_tile_size = htp_tensor_get_tile_padded_size(weight_type);

    ܐܢ (state.n_tasks == 1 || n_threads == 1) {
        dequantize_tiled_worker_loop(1, 0, &state);
    } ܐܚܪܢܐ {
        صحيح n_tasks = hex_smin((صحيح) state.n_tasks, n_threads);
        worker_pool_run_func(ctx->worker_pool, dequantize_tiled_worker_loop, &state, n_tasks);
    }
}

تعريفنوع بنية {
    بنية htp_context      * ctx;
    بنية htp_thread_trace * traces;
    عائم                   * dst;
    ثابت __fp16            * vtcm_src;
    ثابت عائم             * src2;
    uint32_t                  n_rows;
    uint32_t                  n_cols;
    uint32_t                  dst_stride;
    uint32_t                  src2_stride;
    uint32_t                  dst_cols;
    بنية fastdiv_values     n_threads_div;
} output_transfer_col_chunk_state_t;

ساكن فراغ transfer_output_chunk_col_chunk_worker_fn(unsigned صحيح n, unsigned صحيح i, فراغ *data) {
    (فراغ) n;
    output_transfer_col_chunk_state_t *st = (output_transfer_col_chunk_state_t *) data;
    بنية htp_thread_trace * tr = &st->traces[i];

    // Round up: the last block is partial when N is not 32-aligned. Its pad
    // columns are dropped by the dst_cols clamp inside the store.
    uint32_t n_blocks = hmx_ceil_div(st->n_cols, 32);
    uint32_t b_first  = fastdiv(n_blocks * i, &st->n_threads_div);
    uint32_t b_last   = fastdiv(n_blocks * (i + 1), &st->n_threads_div);
    uint32_t c_first  = b_first * 32;
    uint32_t c_last   = b_last * 32;
    uint32_t c_len    = c_last - c_first;

    ܐܢ (c_len == 0) ܦܢܐ;

    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_O_PROC, c_first);

    ثابت __fp16 *vtcm_src = st->vtcm_src + b_first * HTP_MM_HMX_TILE_N_ELMS;
    ثابت عائم *src2      = st->src2 ? (st->src2 + c_first) : عدم;
    عائم *dst             = st->dst + c_first;

    صحيح chunk_dst_cols = (صحيح)st->dst_cols - (صحيح)c_first;
    ܐܢ (chunk_dst_cols > 0) {
        transfer_output_chunk_fp16_to_fp32_col_chunk(
            dst, src2, vtcm_src, 0, st->n_rows, c_len, st->n_cols,
            st->dst_stride, st->src2_stride, (uint32_t)chunk_dst_cols
        );
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_O_PROC, c_first);
}

ساكن فراغ transfer_output_chunk_threaded(بنية htp_context *ctx, عائم *dst, ثابت عائم *src2, ثابت __fp16 *vtcm_src,
                                              صحيح n_rows, صحيح n_cols, صحيح dst_stride, uint32_t src2_stride, صحيح dst_cols, صحيح n_threads) {
    ܐܢ (n_rows <= 0) ܦܢܐ;

    uint32_t n_blocks = hmx_ceil_div((uint32_t) n_cols, 32);
    ܐܢ (n_threads > 1 && n_blocks >= (uint32_t)n_threads) {
        بنية fastdiv_values n_threads_div = (n_threads == (صحيح)ctx->n_threads) ? ctx->n_threads_div : init_fastdiv_values(n_threads);
        output_transfer_col_chunk_state_t col_state;
        col_state.dst = dst;
        col_state.src2 = src2;
        col_state.vtcm_src = vtcm_src;
        col_state.n_rows = (uint32_t)n_rows;
        col_state.n_cols = (uint32_t)n_cols;
        col_state.dst_stride = (uint32_t)dst_stride;
        col_state.src2_stride = src2_stride;
        col_state.dst_cols = (uint32_t)dst_cols;
        col_state.n_threads_div = n_threads_div;
        col_state.traces = ctx->trace;
        col_state.ctx = ctx;

        worker_pool_run_func(ctx->worker_pool, transfer_output_chunk_col_chunk_worker_fn, &col_state, n_threads);
        ܦܢܐ;
    }

    size_t n_tot_chunks      = n_rows;
    size_t n_chunks_per_task = (n_threads == 1) ? n_tot_chunks : hmx_ceil_div(n_rows, n_threads);
    n_chunks_per_task        = hex_align_up(n_chunks_per_task, 2);

    صحيح actual_threads = hmx_ceil_div(n_rows, n_chunks_per_task);

    output_transfer_task_state_t state;
    state.n_tasks           = actual_threads;
    state.n_tot_chunks      = n_tot_chunks;
    state.n_chunks_per_task = n_chunks_per_task;
    state.dst               = dst;
    state.src2              = src2;
    state.vtcm_src          = vtcm_src;
    state.n_cols            = n_cols;
    state.dst_stride        = dst_stride;
    state.src2_stride       = src2_stride;
    state.dst_cols          = dst_cols;
    state.traces            = ctx->trace;

    ܐܢ (actual_threads <= 1) {
        transfer_output_chunk_worker_fn(1, 0, &state);
    } ܐܚܪܢܐ {
        worker_pool_run_func(ctx->worker_pool, transfer_output_chunk_worker_fn, &state, actual_threads);
    }
}

بنية activation_transfer_params {
    بنية htp_context *          ctx;
    __fp16 *                      dst;
    dma_addr_t                    act_dma_addr;
    صحيح                           n_rows;
    صحيح                           k_block;
    صحيح                           k_stride;
    صحيح                           n_threads;
    ثابت بنية fastdiv_values * act_threads_div;
    ثابت بنية fastdiv_values * k_div;
    صحيح                           k_valid;
    عائم *                       vtcm_f32_act;
    size_t                        vtcm_f32_act_bytes;
    uint32_t                      act_elem_size;  // 4=F32 src1, 2=F16 src1
};

ساكن فراغ transfer_activation_chunk_threaded(ثابت بنية activation_transfer_params * params) {
    بنية htp_context *          ctx                = params->ctx;
    __fp16 *                      dst                = params->dst;
    ثابت dma_addr_t              act_dma_addr       = params->act_dma_addr;
    صحيح                           n_rows             = params->n_rows;
    صحيح                           k_block            = params->k_block;
    صحيح                           k_stride           = params->k_stride;
    صحيح                           n_threads          = params->n_threads;
    ثابت بنية fastdiv_values * act_threads_div    = params->act_threads_div;
    ثابت بنية fastdiv_values * k_div              = params->k_div;
    صحيح                           k_valid            = params->k_valid;
    عائم *                       vtcm_f32_act       = params->vtcm_f32_act;
    size_t                        vtcm_f32_act_bytes = params->vtcm_f32_act_bytes;
    // element size of the activation rows (4 = F32, 2 = F16).
    ثابت uint32_t                act_elem_size      = params->act_elem_size ? params->act_elem_size : (uint32_t) حجم(عائم);

    ܐܢ (n_rows <= 0) {
        ܦܢܐ;
    }

    ثابت size_t n_tasks = (n_rows + 31) >> 5;
    ܐܢ (act_elem_size == حجم(عائم) && n_threads > 1 && k_block > 32 && n_tasks < (size_t) n_threads) {
        // Calculate step rows parameters for column-chunked dma pipelining
        uint32_t dma_step_rows = 2;
        uint32_t dma_step_rows_shift = 1;
        ܐܢ (vtcm_f32_act_bytes > 0) {
            size_t thread_scratch_bytes = hex_align_down(fastdiv(vtcm_f32_act_bytes, act_threads_div), 128);
            size_t thread_scratch_elements = thread_scratch_bytes / حجم(عائم);
            size_t dma_step_rows_max = fastdiv(thread_scratch_elements / 2, k_div);
            ܐܢ (dma_step_rows_max >= 4) {
                dma_step_rows = 4;
                dma_step_rows_shift = 2;
            }
        }

        activation_transfer_col_chunk_state_t col_state;
        col_state.dst = dst;
        col_state.act_dma_addr = act_dma_addr;
        col_state.n_rows = n_rows;
        col_state.k_block = k_block;
        col_state.k_stride = k_stride;
        col_state.k_valid = k_valid;
        col_state.n_col_chunks = n_threads;
        col_state.n_threads_div = *act_threads_div;
        col_state.vtcm_f32_act = vtcm_f32_act;
        col_state.vtcm_f32_act_bytes = vtcm_f32_act_bytes;
        col_state.traces = ctx->trace;
        col_state.ctx = ctx;
        col_state.dma_step_rows = dma_step_rows;
        col_state.dma_step_rows_shift = dma_step_rows_shift;

        worker_pool_run_func(ctx->worker_pool, transfer_activation_chunk_col_chunk_worker_fn, &col_state, n_threads);
        ܦܢܐ;
    }

    أكد(k_block % HTP_MM_HMX_TILE_N_COLS == 0 && k_stride % HTP_MM_HMX_TILE_N_COLS == 0);

    size_t n_tot_chunks      = n_rows;
    size_t n_chunks_per_task = (n_threads == 1) ? n_tot_chunks : 32;  // must be multiple of 32 to ensure correct destination address

    activation_transfer_task_state_t state;
    state.n_tasks            = (n_threads == 1) ? 1 : hmx_ceil_div(n_tot_chunks, 32);
    state.n_tot_chunks       = n_tot_chunks;
    state.n_chunks_per_task  = n_chunks_per_task;
    state.dst                = dst;
    state.act_dma_addr       = act_dma_addr;
    state.k_block            = k_block;
    state.k_stride           = k_stride;
    state.k_valid            = k_valid;
    state.traces             = ctx->trace;
    state.ctx                = ctx;
    state.vtcm_f32_act       = vtcm_f32_act;
    state.act_elem_size = act_elem_size;

    state.vtcm_f32_act_bytes_per_thread = hex_align_down(fastdiv(vtcm_f32_act_bytes, act_threads_div), 128);

    uint32_t dma_step_rows = 2;
    uint32_t dma_step_rows_shift = 1;
    ܐܢ (state.vtcm_f32_act_bytes_per_thread > 0) {
        size_t thread_scratch_elements = state.vtcm_f32_act_bytes_per_thread / حجم(عائم);
        size_t dma_step_rows_max = fastdiv(thread_scratch_elements / 2, k_div);
        ܐܢ (dma_step_rows_max >= 4) {
            dma_step_rows = 4;
            dma_step_rows_shift = 2;
        }
    }
    state.dma_step_rows      = dma_step_rows;
    state.dma_step_rows_shift = dma_step_rows_shift;

    صحيح active_threads = hex_smin(n_threads, (صحيح)state.n_tasks);
    ܐܢ (state.n_tasks == 1 || n_threads == 1) {
        transfer_activation_chunk_worker_fn(1, 0, &state);
    } ܐܚܪܢܐ {
        worker_pool_run_func(ctx->worker_pool, transfer_activation_chunk_worker_fn, &state, active_threads);
    }
}
// --- Async HMX matmul job (for pipeline overlap) ---

تعريفنوع بنية {
    __fp16 *       output;
    ثابت __fp16 * activation;
    ثابت __fp16 * weight;
    ثابت __fp16 * scales;
    uint32_t       n_row_tiles;
    uint32_t       n_col_tiles;
    uint32_t       n_dot_tiles;
} hmx_matmul_job_t;

ساكن فراغ hmx_matmul_worker_fn(فراغ * data) {
    hmx_matmul_job_t * job = (hmx_matmul_job_t *) data;
    FARF(HIGH, "hmx-mm-job: n_row_tiles %u n_col_tiles %u n_dot_tiles %u", job->n_row_tiles, job->n_col_tiles, job->n_dot_tiles);
    core_dot_chunk_fp16(job->output, job->activation, job->weight, job->scales, job->n_row_tiles, job->n_col_tiles, job->n_dot_tiles);
}

ساكن مضمن فراغ hmx_matmul_job_init(hmx_matmul_job_t * job,
                                       __fp16 *           output,
                                       ثابت __fp16 *     activation,
                                       ثابت __fp16 *     weight,
                                       ثابت __fp16 *     scales,
                                       uint32_t           n_row_tiles,
                                       uint32_t           n_col_tiles,
                                       uint32_t           n_dot_tiles) {
    job->output      = output;
    job->activation  = activation;
    job->weight      = weight;
    job->scales      = scales;
    job->n_row_tiles = n_row_tiles;
    job->n_col_tiles = n_col_tiles;
    job->n_dot_tiles = n_dot_tiles;
}

ساكن صحيح hmx_mm_2d_f32(بنية htp_context *ctx,
                                  dma_queue *weight_dma,
                                  عائم *restrict dst,
                                  dma_addr_t src2_addr,
                                  size_t src2_bytes,
                                  dma_addr_t act_dma_addr,
                                  dma_addr_t weight,
                                  صحيح m, صحيح k, صحيح n,
                                  صحيح act_stride,
                                  uint32_t act_elem_size,  // 4=F32 src1, 2=F16 src1
                                  صحيح weight_stride,
                                  صحيح weight_type,
                                  صحيح k_valid,
                                  صحيح dst_stride,
                                  uint32_t src2_stride,
                                  صحيح dst_cols,
                                  صحيح m_chunk,
                                  صحيح n_chunk,
                                  صحيح pipeline,
                                  صحيح n_threads,
                                  صحيح act_threads,
                                  ثابت بنية fastdiv_values * act_threads_div,
                                  ثابت بنية fastdiv_values * k_div,
                                  صحيح tile_size,
                                  صحيح aligned_tile_size,
                                  صحيح vtcm_size) {
    بنية htp_thread_trace * tr = &ctx->trace[0];
    htp_trace_event_start(tr, HTP_TRACE_EVT_INIT, 0);

    ثابت bool is_tiled = (tile_size > 0);

    ܐܢ (k % 32 != 0) {
        ܦܢܐ -1;
    }
    ܐܢ (!hex_is_aligned(dst, حجم(عائم))) { ܦܢܐ -1; }

    size_t row_stride = htp_tensor_get_row_size(weight_type, k);
    ܐܢ (row_stride == 0) {
        ܦܢܐ -1;
    }

    tiled_dequant_task_fn_t dequant_worker_fn = hmx_get_dequant_worker_fn(weight_type, is_tiled);

    ثابت صحيح n_k_tiles = k / HTP_MM_HMX_TILE_N_COLS;
    ثابت صحيح n_weight_tiles = k / htp_tensor_get_tile_k(weight_type);
    ثابت بنية fastdiv_values n_k_tiles_div = init_fastdiv_values(n_k_tiles);

    ثابت size_t vec_dot_size = k * حجم(__fp16);
    ثابت size_t vtcm_budget  = ctx->vtcm_size;

    ثابت uint32_t dma_dst_stride  = is_tiled ? aligned_tile_size : row_stride;
    ثابت uint32_t dma_src_stride  = is_tiled ? tile_size : weight_stride;
    ثابت uint32_t dma_width_bytes = is_tiled ? tile_size : row_stride;

    size_t m_chunk_n_rows = m_chunk;
    size_t n_chunk_n_cols = n_chunk;
    size_t vtcm_used      = vtcm_size;

    بنية htp_mm_hmx_vtcm_layout L;
    htp_mm_hmx_vtcm_layout_build(&L, HTP_MM_KERNEL_HMX_2D, weight_type, k, m_chunk_n_rows, n_chunk_n_cols, 1, pipeline, act_threads, aligned_tile_size, src2_bytes);

    vtcm_used = L.total_bytes;
    ܐܢ (vtcm_used > vtcm_budget) {
        FARF(ERROR, "hmx-mm-2d-precomputed: VTCM overflow: used %zu budget %zu, m %d k %d n %d mc %zu nc %zu",
             vtcm_used, vtcm_budget, m, k, n, m_chunk_n_rows, n_chunk_n_cols);
        ܦܢܐ -1;
    }

    uint8_t * ثابت base = (uint8_t *) ctx->vtcm_base;
    __fp16  *vtcm_weight_raw[2] = {
        VTCM_LAYOUT_PTR(__fp16, base, L.off_weight[0]),
        VTCM_LAYOUT_PTR_OPTIONAL(__fp16, base, L.off_weight[1], pipeline)
    };
    __fp16  *vtcm_f16_act    = VTCM_LAYOUT_PTR(__fp16, base, L.off_act);
    عائم   *vtcm_f32_act    = VTCM_LAYOUT_PTR(عائم, base, L.off_act_f32);
    __fp16  *vtcm_output     = VTCM_LAYOUT_PTR(__fp16, base, L.off_dst[0]);
    فراغ    *vtcm_scratch0   = VTCM_LAYOUT_PTR(فراغ, base, L.off_scratch[0]);
    فراغ    *vtcm_scratch1   = VTCM_LAYOUT_PTR_OPTIONAL(فراغ, base, L.off_scratch[1], pipeline);
    فراغ    *vtcm_scratch2   = VTCM_LAYOUT_PTR_OPTIONAL(فراغ, base, L.off_dst[1], pipeline);
    __fp16  *vtcm_scales     = VTCM_LAYOUT_PTR(__fp16, base, L.off_scales);

    hmx_init_column_scales(vtcm_scales, Q6_V_vsplat_R(0x3c00));  // scale: 1.0, bias: 0.0 in FP16

    ثابت bool has_bias      = (src2_bytes > 0 && src2_addr != 0);
    عائم   *vtcm_bias       = VTCM_LAYOUT_PTR_OPTIONAL(عائم, base, L.off_bias, has_bias);
    ܐܢ (has_bias) {
        dma_queue_push(weight_dma, dma_make_data(vtcm_bias, src2_addr), hex_align_up(src2_bytes, 128), 0, src2_bytes, 1);
        dma_queue_pop(weight_dma);
    }

    FARF(HIGH, "hmx-mm-2d: m %d k %d n %d wtype %d mc %zu nc %zu vtcm %zu/%zu",
         m, k, n, weight_type, m_chunk_n_rows, n_chunk_n_cols, vtcm_used, vtcm_budget);

    صحيح n_chunk_cnt = hmx_ceil_div(n, n_chunk_n_cols);

    htp_trace_event_stop(tr, HTP_TRACE_EVT_INIT, 0);

    ܐܢ (pipeline) {
        // --- Asynchronous Pipelined Loop ---
        hmx_matmul_job_t job_slots[2];  // persistent double-buffered job descriptors
        فراغ *vtcm_raw_bufs[4] = {
            vtcm_weight_raw[0],
            vtcm_weight_raw[1],
            dequant_worker_fn ? vtcm_weight_raw[0] : vtcm_scratch0,
            dequant_worker_fn ? vtcm_weight_raw[1] : vtcm_scratch1,
        };
        فراغ *vtcm_weight_bufs[4] = {
            dequant_worker_fn ? vtcm_scratch0 : vtcm_weight_raw[0],
            dequant_worker_fn ? vtcm_scratch1 : vtcm_weight_raw[1],
            vtcm_scratch0,
            vtcm_scratch1,
        };
        فراغ *vtcm_output_bufs[2] = { vtcm_output, vtcm_scratch2 };

        لكل (size_t mr = 0; mr < m; mr += m_chunk_n_rows) {
            ثابت size_t n_rows = hex_smin(m - mr, m_chunk_n_rows);

            بنية activation_transfer_params act_params = {
                .ctx = ctx,
                .dst = vtcm_f16_act,
                .act_dma_addr = act_dma_addr + (size_t) mr * act_stride * act_elem_size,  // byte offset (F16/F32)
                .n_rows = (صحيح) n_rows,
                .k_block = k,
                .k_stride = act_stride,
                .n_threads = act_threads,
                .act_threads_div = act_threads_div,
                .k_div = k_div,
                .k_valid = k_valid,
                .vtcm_f32_act = vtcm_f32_act,
                .vtcm_f32_act_bytes = L.act_f32_bytes,
                .act_elem_size = act_elem_size,
            };
            transfer_activation_chunk_threaded(&act_params);

            // Prologue: push A0 and optionally A1 (if n_chunk_cnt > 1)
            ثابت size_t   n_cols_A0 = hex_smin(n - 0 * n_chunk_n_cols, n_chunk_n_cols);
            ثابت uint32_t height_A0 = is_tiled ? hmx_ceil_div(n_cols_A0, 32) * n_weight_tiles : n_cols_A0;
            dma_queue_push(weight_dma, dma_make_data(vtcm_raw_bufs[0], weight),
                           dma_dst_stride, dma_src_stride, dma_width_bytes, height_A0);

            ܐܢ (1 < n_chunk_cnt) {
                ثابت size_t   n_cols_A1 = hex_smin(n - 1 * n_chunk_n_cols, n_chunk_n_cols);
                ثابت uint32_t height_A1 = is_tiled ? hmx_ceil_div(n_cols_A1, 32) * n_weight_tiles : n_cols_A1;
                dma_queue_push(weight_dma, dma_make_data(vtcm_raw_bufs[1], weight + n_chunk_n_cols * weight_stride),
                               dma_dst_stride, dma_src_stride, dma_width_bytes, height_A1);
            }

            // Main loop: pop A_i -> dequantize A_i -> push A_{i+2} -> submit C_i -> wait C_{i-1} and store D_{i-1}
            لكل (صحيح i = 0; i < n_chunk_cnt; ++i) {
                ثابت size_t nc    = i * n_chunk_n_cols;
                ثابت size_t nc_p2 = nc + 2 * n_chunk_n_cols;

                ثابت size_t n_cols    = hex_smin(n - nc, n_chunk_n_cols);
                ثابت size_t n_cols_p2 = hex_smin(n - nc_p2, n_chunk_n_cols);

                // 1. pop A_i
                فراغ * curr_raw = (فراغ *) dma_queue_pop(weight_dma).dst;

                // 2. dequantize A_i
                dequantize_tiled_weight_chunk_to_fp16_tiles(
                    ctx, vtcm_weight_bufs[i % 4], curr_raw,
                    n_cols, k, row_stride, weight_type,
                    n_k_tiles, n_k_tiles_div, dequant_worker_fn, n_threads);

                // 3. push A_{i+2} (if i+2 < n_chunk_cnt)
                ܐܢ (i + 2 < n_chunk_cnt) {
                    ثابت uint32_t height_p2 = is_tiled ? hmx_ceil_div(n_cols_p2, 32) * n_weight_tiles : n_cols_p2;
                    dma_queue_push(weight_dma, dma_make_data(vtcm_raw_bufs[(i + 2) % 4], weight + nc_p2 * weight_stride),
                                   dma_dst_stride, dma_src_stride, dma_width_bytes, height_p2);
                }

                // 4. submit C_i
                hmx_matmul_job_init(&job_slots[i % 2], (__fp16 *) vtcm_output_bufs[i % 2],
                                    (__fp16 *) vtcm_f16_act, (__fp16 *) vtcm_weight_bufs[i % 4],
                                    vtcm_scales, hmx_ceil_div(n_rows, HTP_MM_HMX_TILE_N_ROWS),
                                    hmx_ceil_div(n_cols, HTP_MM_HMX_TILE_N_COLS), k / HTP_MM_HMX_TILE_N_ROWS);
                hmx_queue_push(ctx->hmx_queue, hmx_queue_make_desc(hmx_matmul_worker_fn, &job_slots[i % 2]));

                // 5. wait C_{i-1} and store D_{i-1} (multi-thread HVX, parallel with C_i)
                ܐܢ (i > 0) {
                    hmx_queue_pop(ctx->hmx_queue);
                    ثابت size_t nc_prev = (i - 1) * n_chunk_n_cols;
                    ثابت size_t n_cols_prev = hex_smin(n - nc_prev, n_chunk_n_cols);
                    عائم *output_chunk = dst + (mr * dst_stride + nc_prev);
                    ثابت عائم *bias_chunk = has_bias ? (vtcm_bias + mr * src2_stride + nc_prev) : عدم;
                    صحيح chunk_dst_cols = dst_cols - (صحيح)nc_prev;
                    ܐܢ (chunk_dst_cols > 0) {
                        transfer_output_chunk_threaded(ctx, output_chunk, bias_chunk, vtcm_output_bufs[(i - 1) % 2], n_rows, n_cols_prev, dst_stride, src2_stride, chunk_dst_cols, n_threads);
                    }
                }
            }

            // Epilogue: wait C_{last} and store D_{last}
            hmx_queue_pop(ctx->hmx_queue);
            ثابت size_t nc_last = (n_chunk_cnt - 1) * n_chunk_n_cols;
            ثابت size_t n_cols_last = hex_smin(n - nc_last, n_chunk_n_cols);
            عائم *output_chunk = dst + (mr * dst_stride + nc_last);
            ثابت عائم *bias_chunk = has_bias ? (vtcm_bias + mr * src2_stride + nc_last) : عدم;
            صحيح chunk_dst_cols = dst_cols - (صحيح)nc_last;
            ܐܢ (chunk_dst_cols > 0) {
                transfer_output_chunk_threaded(ctx, output_chunk, bias_chunk, vtcm_output_bufs[(n_chunk_cnt - 1) % 2], n_rows, n_cols_last, dst_stride, src2_stride, chunk_dst_cols, n_threads);
            }
        }
    } ܐܚܪܢܐ {
        // --- Synchronous loop (m <= 32 or fallback) ---
        hmx_matmul_job_t job;
        فراغ *vtcm_raw_bufs[2] = { vtcm_weight_raw[0], dequant_worker_fn ? vtcm_weight_raw[0] : vtcm_scratch0 };
        فراغ *vtcm_weight_bufs[2] = { dequant_worker_fn ? vtcm_scratch0 : vtcm_weight_raw[0], vtcm_scratch0 };

        لكل (size_t mr = 0; mr < m; mr += m_chunk_n_rows) {
            ثابت size_t n_rows = hex_smin(m - mr, m_chunk_n_rows);

            بنية activation_transfer_params act_params = {
                .ctx = ctx,
                .dst = vtcm_f16_act,
                .act_dma_addr = act_dma_addr + (size_t) mr * act_stride * act_elem_size,  // byte offset (F16/F32)
                .n_rows = (صحيح) n_rows,
                .k_block = k,
                .k_stride = act_stride,
                .n_threads = act_threads,
                .act_threads_div = act_threads_div,
                .k_div = k_div,
                .k_valid = k_valid,
                .vtcm_f32_act = vtcm_f32_act,
                .vtcm_f32_act_bytes = L.act_f32_bytes,
                .act_elem_size = act_elem_size,
            };
            transfer_activation_chunk_threaded(&act_params);

            // A0: Pre-fetch the first weight chunk (nc = 0)
            ܐܢ (n > 0) {
                ثابت size_t n_cols = hex_smin(n, n_chunk_n_cols);
                ثابت uint32_t height = is_tiled ? hmx_ceil_div(n_cols, 32) * n_weight_tiles : n_cols;
                dma_queue_push(weight_dma, dma_make_data(vtcm_raw_bufs[0], weight), dma_dst_stride, dma_src_stride, dma_width_bytes, height);
            }

            لكل (صحيح i = 0; i < n_chunk_cnt; ++i) {
                ثابت size_t nc = (size_t) i * n_chunk_n_cols;
                ثابت size_t n_cols = hex_smin(n - nc, n_chunk_n_cols);
                ثابت size_t n_row_tiles = hmx_ceil_div(n_rows, HTP_MM_HMX_TILE_N_ROWS);
                ثابت size_t n_col_tiles = hmx_ceil_div(n_cols, HTP_MM_HMX_TILE_N_COLS);

                // A: Wait for weight DMA
                فراغ * curr_raw = (فراغ *) dma_queue_pop(weight_dma).dst;

                // B: Weight Dequantize (Threaded)
                dequantize_tiled_weight_chunk_to_fp16_tiles(
                    ctx, vtcm_weight_bufs[i % 2], curr_raw,
                    n_cols, k, row_stride, weight_type,
                    n_k_tiles, n_k_tiles_div, dequant_worker_fn, n_threads);

                // Start weight DMA for the next chunk early
                ثابت size_t nc_next = nc + n_chunk_n_cols;
                ܐܢ (nc_next < n) {
                    ثابت size_t n_cols_next = hex_smin(n - nc_next, n_chunk_n_cols);
                    ثابت uint32_t height_next = is_tiled ? hmx_ceil_div(n_cols_next, 32) * n_weight_tiles : n_cols_next;
                    dma_queue_push(weight_dma, dma_make_data(vtcm_raw_bufs[(i + 1) % 2], weight + nc_next * weight_stride), dma_dst_stride, dma_src_stride, dma_width_bytes, height_next);
                }

                // C: HMX Compute (Queue-based)
                hmx_matmul_job_init(&job, vtcm_output, vtcm_f16_act, (__fp16 *) vtcm_weight_bufs[i % 2], vtcm_scales, n_row_tiles, n_col_tiles, k / HTP_MM_HMX_TILE_N_ROWS);
                hmx_queue_push(ctx->hmx_queue, hmx_queue_make_desc(hmx_matmul_worker_fn, &job));
                hmx_queue_pop(ctx->hmx_queue);

                // D: Output Store
                عائم *output_chunk = dst + (mr * dst_stride + nc);
                ثابت عائم *bias_chunk = has_bias ? (vtcm_bias + mr * src2_stride + nc) : عدم;
                صحيح chunk_dst_cols = dst_cols - (صحيح)nc;
                ܐܢ (chunk_dst_cols > 0) {
                    transfer_output_chunk_threaded(ctx, output_chunk, bias_chunk, vtcm_output, n_rows, n_cols, dst_stride, src2_stride, chunk_dst_cols, n_threads);
                }
            }
        }
    }

    ܦܢܐ 0;
}

ساكن صحيح hmx_mm_nx_2d_f32(بنية htp_ops_context * octx, ثابت بنية htp_mm_kernel_params * kparams) {
    بنية htp_context * ctx = octx->ctx;
    بنية htp_thread_trace * tr = &ctx->trace[0];
    htp_trace_event_start(tr, HTP_TRACE_EVT_INIT, 0);

    ثابت uint32_t n_weights = kparams->n_weights;
    ܐܢ (n_weights == 0 || n_weights > HTP_OP_MAX_OUTPUTS) {
        ܦܢܐ HTP_STATUS_INVAL_PARAMS;
    }

    ثابت بنية htp_tensor * restrict src0 = octx->src[0];
    ثابت بنية htp_tensor * restrict act  = octx->src[n_weights];

    ثابت صحيح weight_type = (صحيح) src0->type;
    ثابت صحيح k           = (صحيح) act->ne[0];
    ثابت صحيح k_valid     = (صحيح) act->ne[0];
    ثابت صحيح m           = (صحيح) (act->ne[1] * act->ne[2] * act->ne[3]);
    ثابت uint32_t act_elem_size = (act->type == HTP_TYPE_F16) ? حجم(__fp16) : حجم(عائم);
    ثابت صحيح act_stride  = (صحيح) (act->nb[1] / act_elem_size);
    ثابت dma_addr_t act_dma_addr = act->data;

    ܐܢ (k % 32 != 0) { ܦܢܐ HTP_STATUS_NO_SUPPORT; }

    size_t row_stride = htp_tensor_get_row_size(weight_type, k);
    ܐܢ (row_stride == 0) {
        ܦܢܐ HTP_STATUS_NO_SUPPORT;
    }

    ثابت صحيح n_k_tiles = k / HTP_MM_HMX_TILE_N_COLS;
    ثابت صحيح n_weight_tiles = k / htp_tensor_get_tile_k(weight_type);
    ثابت بنية fastdiv_values n_k_tiles_div = init_fastdiv_values(n_k_tiles);

    ثابت صحيح tile_size       = kparams->tile_size;
    ثابت صحيح aligned_tile_size = kparams->aligned_tile_size;

    ثابت bool is_tiled       = (tile_size > 0);
    tiled_dequant_task_fn_t dequant_worker_fn = hmx_get_dequant_worker_fn(weight_type, is_tiled);
    ثابت size_t vtcm_budget  = ctx->vtcm_size;

    ثابت صحيح m_chunk_n_rows  = kparams->m_chunk;
    ثابت صحيح n_chunk_n_cols  = kparams->n_chunk;
    ثابت صحيح pipeline        = kparams->pipeline;
    ثابت صحيح n_threads       = octx->n_threads;
    ثابت صحيح act_threads     = kparams->n_act_threads;
    ثابت بنية fastdiv_values * act_threads_div = &kparams->div_n_act_threads;
    ثابت بنية fastdiv_values * k_div           = &kparams->div_ne00_padded;

    ثابت uint32_t dma_dst_stride  = is_tiled ? aligned_tile_size : row_stride;
    ثابت uint32_t dma_width_bytes = is_tiled ? tile_size : row_stride;

    بنية htp_mm_hmx_vtcm_layout L;
    htp_mm_hmx_vtcm_layout_build(&L, HTP_MM_KERNEL_HMX_2D, weight_type, k, m_chunk_n_rows, n_chunk_n_cols, 1, pipeline, act_threads, aligned_tile_size, 0);

    ܐܢ (L.total_bytes > vtcm_budget) {
        FARF(ERROR, "hmx-mm-nx-2d: VTCM overflow: used %zu budget %zu, m %d k %d mc %d nc %d",
             L.total_bytes, vtcm_budget, m, k, m_chunk_n_rows, n_chunk_n_cols);
        ܦܢܐ HTP_STATUS_VTCM_TOO_SMALL;
    }

    uint8_t * ثابت base = (uint8_t *) ctx->vtcm_base;
    __fp16  *vtcm_weight_raw[2] = {
        VTCM_LAYOUT_PTR(__fp16, base, L.off_weight[0]),
        VTCM_LAYOUT_PTR_OPTIONAL(__fp16, base, L.off_weight[1], pipeline)
    };
    __fp16  *vtcm_f16_act    = VTCM_LAYOUT_PTR(__fp16, base, L.off_act);
    عائم   *vtcm_f32_act    = VTCM_LAYOUT_PTR(عائم, base, L.off_act_f32);
    __fp16  *vtcm_output     = VTCM_LAYOUT_PTR(__fp16, base, L.off_dst[0]);
    فراغ    *vtcm_scratch0   = VTCM_LAYOUT_PTR(فراغ, base, L.off_scratch[0]);
    فراغ    *vtcm_scratch1   = VTCM_LAYOUT_PTR_OPTIONAL(فراغ, base, L.off_scratch[1], pipeline);
    فراغ    *vtcm_scratch2   = VTCM_LAYOUT_PTR_OPTIONAL(فراغ, base, L.off_dst[1], pipeline);
    __fp16  *vtcm_scales     = VTCM_LAYOUT_PTR(__fp16, base, L.off_scales);

    hmx_init_column_scales(vtcm_scales, Q6_V_vsplat_R(0x3c00));  // scale: 1.0, bias: 0.0 in FP16

    صحيح m_start = 0;
    صحيح m_rows  = m;
    ܐܢ (octx->ctx->mdev.count > 1) {
        bool can_split = htp_tensor_can_row_partition(act, act_elem_size);
        ܐܢ (can_split) {
            لكل (uint32_t p = 0; p < n_weights; ++p) {
                ثابت بنية htp_tensor * restrict dst = octx->dsts[p];
                ܐܢ (!htp_tensor_can_row_partition(dst, حجم(عائم))) {
                    can_split = false;
                    اكسر;
                }
            }
        }
        ثابت بنية htp_tensor_mdev_range range = htp_tensor_mdev_partition((uint32_t) m, can_split ? 1 : 0, octx->ctx->mdev.idx, octx->ctx->mdev.count, &octx->ctx->mdev.count_div);
        m_start = (صحيح) range.start;
        m_rows  = (صحيح) range.count;
    }

    ܐܢ (m_rows == 0) {
        ܦܢܐ HTP_STATUS_OK;
    }

    FARF(HIGH, "hmx-mm-nx-2d: n_weights %u m %d (%d..%d) k %d wtype %d mc %d nc %d vtcm %zu/%zu",
         n_weights, m, m_start, m_start + m_rows, k, weight_type, m_chunk_n_rows, n_chunk_n_cols, L.total_bytes, vtcm_budget);

    htp_trace_event_stop(tr, HTP_TRACE_EVT_INIT, 0);

    ثابت size_t mr_end = (size_t)(m_start + m_rows);

    ܐܢ (pipeline) {
        hmx_matmul_job_t job_slots[2];
        فراغ *vtcm_raw_bufs[4] = {
            vtcm_weight_raw[0],
            vtcm_weight_raw[1],
            dequant_worker_fn ? vtcm_weight_raw[0] : vtcm_scratch0,
            dequant_worker_fn ? vtcm_weight_raw[1] : vtcm_scratch1,
        };
        فراغ *vtcm_weight_bufs[4] = {
            dequant_worker_fn ? vtcm_scratch0 : vtcm_weight_raw[0],
            dequant_worker_fn ? vtcm_scratch1 : vtcm_weight_raw[1],
            vtcm_scratch0,
            vtcm_scratch1,
        };
        فراغ *vtcm_output_bufs[2] = { vtcm_output, vtcm_scratch2 };

        لكل (size_t mr = (size_t) m_start; mr < mr_end; mr += m_chunk_n_rows) {
            ثابت size_t n_rows = hex_smin(mr_end - mr, m_chunk_n_rows);

            بنية activation_transfer_params act_params = {
                .ctx = ctx,
                .dst = vtcm_f16_act,
                .act_dma_addr = act_dma_addr + (size_t) mr * act_stride * act_elem_size,
                .n_rows = (صحيح) n_rows,
                .k_block = k,
                .k_stride = act_stride,
                .n_threads = act_threads,
                .act_threads_div = act_threads_div,
                .k_div = k_div,
                .k_valid = k_valid,
                .vtcm_f32_act = vtcm_f32_act,
                .vtcm_f32_act_bytes = L.act_f32_bytes,
                .act_elem_size = act_elem_size,
            };
            transfer_activation_chunk_threaded(&act_params);

            لكل (uint32_t p = 0; p < n_weights; p++) {
                ثابت بنية htp_tensor * restrict src_w = octx->src[p];
                ثابت بنية htp_tensor * restrict dst   = octx->dsts[p];
                ܐܢ (!src_w || !dst) استمر;

                ثابت dma_addr_t weight       = src_w->data;
                dma_queue * weight_dma      = octx->ctx->dma[0];
                عائم * dst_ptr            = (عائم *) dst->data;
                ثابت size_t n             = src_w->ne[1];
                ܐܢ (n == 0) استمر;
                ثابت size_t weight_stride = src_w->nb[1];
                ثابت size_t dst_stride    = dst->nb[1] / حجم(عائم);
                ثابت صحيح dst_cols         = (صحيح) dst->ne[0];
                ثابت صحيح n_chunk_cnt      = hmx_ceil_div(n, n_chunk_n_cols);

                ثابت uint32_t dma_src_stride = is_tiled ? tile_size : weight_stride;

                ثابت size_t   n_cols_A0 = hex_smin(n - 0 * n_chunk_n_cols, n_chunk_n_cols);
                ثابت uint32_t height_A0 = is_tiled ? hmx_ceil_div(n_cols_A0, 32) * n_weight_tiles : n_cols_A0;
                dma_queue_push(weight_dma, dma_make_data(vtcm_raw_bufs[0], weight),
                               dma_dst_stride, dma_src_stride, dma_width_bytes, height_A0);

                ܐܢ (1 < n_chunk_cnt) {
                    ثابت size_t   n_cols_A1 = hex_smin(n - 1 * n_chunk_n_cols, n_chunk_n_cols);
                    ثابت uint32_t height_A1 = is_tiled ? hmx_ceil_div(n_cols_A1, 32) * n_weight_tiles : n_cols_A1;
                    dma_queue_push(weight_dma, dma_make_data(vtcm_raw_bufs[1], weight + n_chunk_n_cols * weight_stride),
                                   dma_dst_stride, dma_src_stride, dma_width_bytes, height_A1);
                }

                لكل (صحيح i = 0; i < n_chunk_cnt; ++i) {
                    ثابت size_t nc    = i * n_chunk_n_cols;
                    ثابت size_t nc_p2 = nc + 2 * n_chunk_n_cols;

                    ثابت size_t n_cols    = hex_smin(n - nc, n_chunk_n_cols);
                    ثابت size_t n_cols_p2 = hex_smin(n - nc_p2, n_chunk_n_cols);

                    فراغ * curr_raw = (فراغ *) dma_queue_pop(weight_dma).dst;

                    dequantize_tiled_weight_chunk_to_fp16_tiles(
                        ctx, vtcm_weight_bufs[i % 4], curr_raw,
                        n_cols, k, row_stride, weight_type,
                        n_k_tiles, n_k_tiles_div, dequant_worker_fn, n_threads);

                    ܐܢ (i + 2 < n_chunk_cnt) {
                        ثابت uint32_t height_p2 = is_tiled ? hmx_ceil_div(n_cols_p2, 32) * n_weight_tiles : n_cols_p2;
                        dma_queue_push(weight_dma, dma_make_data(vtcm_raw_bufs[(i + 2) % 4], weight + nc_p2 * weight_stride),
                                       dma_dst_stride, dma_src_stride, dma_width_bytes, height_p2);
                    }

                    hmx_matmul_job_init(&job_slots[i % 2], (__fp16 *) vtcm_output_bufs[i % 2],
                                        (__fp16 *) vtcm_f16_act, (__fp16 *) vtcm_weight_bufs[i % 4],
                                        vtcm_scales, hmx_ceil_div(n_rows, HTP_MM_HMX_TILE_N_ROWS),
                                        hmx_ceil_div(n_cols, HTP_MM_HMX_TILE_N_COLS), k / HTP_MM_HMX_TILE_N_ROWS);
                    hmx_queue_push(ctx->hmx_queue, hmx_queue_make_desc(hmx_matmul_worker_fn, &job_slots[i % 2]));

                    ܐܢ (i > 0) {
                        hmx_queue_pop(ctx->hmx_queue);
                        ثابت size_t nc_prev = (i - 1) * n_chunk_n_cols;
                        ثابت size_t n_cols_prev = hex_smin(n - nc_prev, n_chunk_n_cols);
                        عائم *output_chunk = dst_ptr + (mr * dst_stride + nc_prev);
                        صحيح chunk_dst_cols = dst_cols - (صحيح)nc_prev;
                        ܐܢ (chunk_dst_cols > 0) {
                            transfer_output_chunk_threaded(ctx, output_chunk, عدم, vtcm_output_bufs[(i - 1) % 2], n_rows, n_cols_prev, dst_stride, 0, chunk_dst_cols, n_threads);
                        }
                    }
                }

                hmx_queue_pop(ctx->hmx_queue);
                ثابت size_t nc_last = (n_chunk_cnt - 1) * n_chunk_n_cols;
                ثابت size_t n_cols_last = hex_smin(n - nc_last, n_chunk_n_cols);
                عائم *output_chunk = dst_ptr + (mr * dst_stride + nc_last);
                صحيح chunk_dst_cols = dst_cols - (صحيح)nc_last;
                ܐܢ (chunk_dst_cols > 0) {
                    transfer_output_chunk_threaded(ctx, output_chunk, عدم, vtcm_output_bufs[(n_chunk_cnt - 1) % 2], n_rows, n_cols_last, dst_stride, 0, chunk_dst_cols, n_threads);
                }
            }
        }
    } ܐܚܪܢܐ {
        hmx_matmul_job_t job;
        فراغ *vtcm_raw_bufs[2] = { vtcm_weight_raw[0], dequant_worker_fn ? vtcm_weight_raw[0] : vtcm_scratch0 };
        فراغ *vtcm_weight_bufs[2] = { dequant_worker_fn ? vtcm_scratch0 : vtcm_weight_raw[0], vtcm_scratch0 };

        لكل (size_t mr = (size_t) m_start; mr < mr_end; mr += m_chunk_n_rows) {
            ثابت size_t n_rows = hex_smin(mr_end - mr, m_chunk_n_rows);

            بنية activation_transfer_params act_params = {
                .ctx = ctx,
                .dst = vtcm_f16_act,
                .act_dma_addr = act_dma_addr + (size_t) mr * act_stride * act_elem_size,
                .n_rows = (صحيح) n_rows,
                .k_block = k,
                .k_stride = act_stride,
                .n_threads = act_threads,
                .act_threads_div = act_threads_div,
                .k_div = k_div,
                .k_valid = k_valid,
                .vtcm_f32_act = vtcm_f32_act,
                .vtcm_f32_act_bytes = L.act_f32_bytes,
                .act_elem_size = act_elem_size,
            };
            transfer_activation_chunk_threaded(&act_params);

            لكل (uint32_t p = 0; p < n_weights; p++) {
                ثابت بنية htp_tensor * restrict src_w = octx->src[p];
                ثابت بنية htp_tensor * restrict dst   = octx->dsts[p];
                ܐܢ (!src_w || !dst) استمر;

                ثابت dma_addr_t weight       = src_w->data;
                dma_queue * weight_dma      = octx->ctx->dma[0];
                عائم * dst_ptr            = (عائم *) dst->data;
                ثابت size_t n             = src_w->ne[1];
                ܐܢ (n == 0) استمر;
                ثابت size_t weight_stride = src_w->nb[1];
                ثابت size_t dst_stride    = dst->nb[1] / حجم(عائم);
                ثابت صحيح dst_cols         = (صحيح) dst->ne[0];
                ثابت صحيح n_chunk_cnt      = hmx_ceil_div(n, n_chunk_n_cols);

                ثابت uint32_t dma_src_stride = is_tiled ? tile_size : weight_stride;

                ܐܢ (n > 0) {
                    ثابت size_t n_cols = hex_smin(n, n_chunk_n_cols);
                    ثابت uint32_t height = is_tiled ? hmx_ceil_div(n_cols, 32) * n_weight_tiles : n_cols;
                    dma_queue_push(weight_dma, dma_make_data(vtcm_raw_bufs[0], weight), dma_dst_stride, dma_src_stride, dma_width_bytes, height);
                }

                لكل (صحيح i = 0; i < n_chunk_cnt; ++i) {
                    ثابت size_t nc = (size_t) i * n_chunk_n_cols;
                    ثابت size_t n_cols = hex_smin(n - nc, n_chunk_n_cols);
                    ثابت size_t n_row_tiles = hmx_ceil_div(n_rows, HTP_MM_HMX_TILE_N_ROWS);
                    ثابت size_t n_col_tiles = hmx_ceil_div(n_cols, HTP_MM_HMX_TILE_N_COLS);

                    فراغ * curr_raw = (فراغ *) dma_queue_pop(weight_dma).dst;

                    dequantize_tiled_weight_chunk_to_fp16_tiles(
                        ctx, vtcm_weight_bufs[i % 2], curr_raw,
                        n_cols, k, row_stride, weight_type,
                        n_k_tiles, n_k_tiles_div, dequant_worker_fn, n_threads);

                    ثابت size_t nc_next = nc + n_chunk_n_cols;
                    ܐܢ (nc_next < n) {
                        ثابت size_t n_cols_next = hex_smin(n - nc_next, n_chunk_n_cols);
                        ثابت uint32_t height_next = is_tiled ? hmx_ceil_div(n_cols_next, 32) * n_weight_tiles : n_cols_next;
                        dma_queue_push(weight_dma, dma_make_data(vtcm_raw_bufs[(i + 1) % 2], weight + nc_next * weight_stride), dma_dst_stride, dma_src_stride, dma_width_bytes, height_next);
                    }

                    hmx_matmul_job_init(&job, vtcm_output, vtcm_f16_act, (__fp16 *) vtcm_weight_bufs[i % 2], vtcm_scales, n_row_tiles, n_col_tiles, k / HTP_MM_HMX_TILE_N_ROWS);
                    hmx_queue_push(ctx->hmx_queue, hmx_queue_make_desc(hmx_matmul_worker_fn, &job));
                    hmx_queue_pop(ctx->hmx_queue);

                    عائم *output_chunk = dst_ptr + (mr * dst_stride + nc);
                    صحيح chunk_dst_cols = dst_cols - (صحيح)nc;
                    ܐܢ (chunk_dst_cols > 0) {
                        transfer_output_chunk_threaded(ctx, output_chunk, عدم, vtcm_output, n_rows, n_cols, dst_stride, 0, chunk_dst_cols, n_threads);
                    }
                }
            }
        }
    }

    ܦܢܐ HTP_STATUS_OK;
}

ساكن مضمن dma_addr_t hmx_mm_weight_batch_data(ثابت hmx_mm_batched_params_t *params,
                                                         صحيح dst_b2, صحيح dst_b3) {
    ثابت size_t b2_idx = (params->r2 <= 1) ? (size_t) dst_b2 : (size_t) fastdiv((uint32_t) dst_b2, &params->div_r2);
    ثابت size_t b3_idx = (params->r3 <= 1) ? (size_t) dst_b3 : (size_t) fastdiv((uint32_t) dst_b3, &params->div_r3);
    ܦܢܐ params->weight + b2_idx * params->src0_nb2 + b3_idx * params->src0_nb3;
}

ساكن مضمن dma_addr_t hmx_mm_act_batch_addr(ثابت hmx_mm_batched_params_t *params,
                                                صحيح dst_b2, صحيح dst_b3) {
    ܦܢܐ params->act_dma_addr + dst_b2 * params->act_nb2 +
           dst_b3 * params->act_nb3;
}

ساكن مضمن عائم *hmx_mm_dst_batch_ptr(ثابت hmx_mm_batched_params_t *params,
                                              صحيح dst_b2, صحيح dst_b3) {
    ܦܢܐ (عائم *) ((uint8_t *) params->dst +
                      (size_t) dst_b2 * params->dst_nb2 +
                      (size_t) dst_b3 * params->dst_nb3);
}

ساكن صحيح hmx_mm_batched_simple(بنية htp_context *ctx,
                                 ثابت hmx_mm_batched_params_t *params,
                                 صحيح m_chunk, صحيح n_chunk, صحيح pipeline, صحيح n_threads, صحيح act_threads, صحيح vtcm_size,
                                 ثابت بنية fastdiv_values * act_threads_div, ثابت بنية fastdiv_values * k_div) {
    صحيح ret = 0;
    لكل (صحيح b3 = 0; b3 < params->ne13 && ret == 0; ++b3) {
        لكل (صحيح b2 = 0; b2 < params->ne12 && ret == 0; ++b2) {
            dma_addr_t cur_bias_addr = params->bias_addr ? (params->bias_addr +
                                       b2 * params->bias_nb2 +
                                       b3 * params->bias_nb3) : 0;
            ret = hmx_mm_2d_f32(ctx, params->weight_dma, hmx_mm_dst_batch_ptr(params, b2, b3),
                                cur_bias_addr, params->bias_bytes,
                                hmx_mm_act_batch_addr(params, b2, b3),
                                hmx_mm_weight_batch_data(params, b2, b3),
                                params->m, params->k, params->n,
                                params->act_stride, params->act_elem_size, params->weight_stride,
                                params->weight_type, params->k, params->dst_stride, params->bias_stride, params->dst_cols,
                                m_chunk, n_chunk, pipeline, n_threads, act_threads,
                                act_threads_div, k_div, params->tile_size, params->aligned_tile_size, vtcm_size);
        }
    }
    ܦܢܐ ret;
}

ساكن صحيح hmx_mm_batched(بنية htp_context *ctx, ثابت hmx_mm_batched_params_t *params,
                          صحيح m_chunk, صحيح n_chunk, صحيح pipeline, صحيح n_threads, صحيح act_threads,
                          ثابت بنية fastdiv_values * act_threads_div,
                          ثابت بنية fastdiv_values * k_div,
                          صحيح vtcm_size) {
    ثابت size_t row_stride = htp_tensor_get_row_size(params->weight_type, params->k);
    ܐܢ (row_stride == 0) { ܦܢܐ -1; }
    ܐܢ (params->act_stride < params->k || (params->tile_size == 0 && (size_t) params->weight_stride < row_stride) || params->dst_stride < params->dst_cols) { ܦܢܐ -1; }
    ܐܢ (params->ne02 <= 0 || params->ne03 <= 0 || params->ne12 <= 0 || params->ne13 <= 0) { ܦܢܐ -1; }
    ܐܢ (params->ne12 % params->ne02 != 0 || params->ne13 % params->ne03 != 0) { ܦܢܐ -1; }
    // N (the weight row count) does not have to be 32-aligned:
    ܐܢ (params->k % 32 != 0) { ܦܢܐ -1; }
    ܐܢ (!hex_is_aligned(params->dst, حجم(عائم))) { ܦܢܐ -1; }

    ثابت صحيح group_size = params->r2;
    ثابت size_t vtcm_budget  = ctx->vtcm_size;

    // Check if the precomputed parameters are grouped or simple.
    // If simple, or if group_size <= 1, we use simple fallback loop.
    // Grouped path is only valid for flat F16 if group_size > 1 and it fits within VTCM budget.
    bool run_grouped = (params->weight_type == HTP_TYPE_F16 && params->tile_size == 0 && group_size > 1 && (size_t)vtcm_size <= vtcm_budget);
    ܐܢ (!run_grouped) {
        ܦܢܐ hmx_mm_batched_simple(ctx, params, m_chunk, n_chunk, pipeline, n_threads, act_threads, vtcm_size, act_threads_div, k_div);
    }

    بنية htp_thread_trace * tr = &ctx->trace[0];
    htp_trace_event_start(tr, HTP_TRACE_EVT_INIT, 0);

    size_t m_chunk_n_rows = m_chunk;
    size_t n_chunk_n_cols = n_chunk;

    بنية htp_mm_hmx_vtcm_layout L;
    htp_mm_hmx_vtcm_layout_build(&L, HTP_MM_KERNEL_HMX_BATCHED, HTP_TYPE_F16, params->k, m_chunk_n_rows, n_chunk_n_cols, group_size, false, act_threads, 0, params->bias_bytes);

    ܐܢ (L.total_bytes > vtcm_budget) {
        FARF(HIGH, "%s: grouped layout overflowed VTCM, falling back to simple batched loop", __func__);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_INIT, 0);
        ܦܢܐ hmx_mm_batched_simple(ctx, params, m_chunk, n_chunk, pipeline, n_threads, act_threads, vtcm_size, act_threads_div, k_div);
    }

    uint8_t * ثابت base = (uint8_t *) ctx->vtcm_base;
    __fp16  *vtcm_weight     = VTCM_LAYOUT_PTR(__fp16, base, L.off_weight[0]);
    __fp16  *vtcm_f16_act    = VTCM_LAYOUT_PTR(__fp16, base, L.off_act);
    __fp16  *vtcm_output     = VTCM_LAYOUT_PTR(__fp16, base, L.off_dst[0]);
    فراغ    *vtcm_scratch0   = VTCM_LAYOUT_PTR(فراغ, base, L.off_scratch[0]);
    فراغ    *vtcm_scratch1   = VTCM_LAYOUT_PTR(فراغ, base, L.off_scratch[1]);
    __fp16  *vtcm_scales     = VTCM_LAYOUT_PTR(__fp16, base, L.off_scales);
    عائم   *vtcm_f32_act    = VTCM_LAYOUT_PTR(عائم, base, L.off_act_f32);

    ثابت bool has_bias      = (params->bias_bytes > 0 && params->bias_addr != 0);
    عائم   *vtcm_bias       = VTCM_LAYOUT_PTR_OPTIONAL(عائم, base, L.off_bias, has_bias);
    ܐܢ (has_bias) {
        dma_queue_push(params->weight_dma, dma_make_data(vtcm_bias, params->bias_addr), hex_align_up(params->bias_bytes, 128), 0, params->bias_bytes, 1);
        dma_queue_pop(params->weight_dma);
    }

    hmx_init_column_scales(vtcm_scales, Q6_V_vsplat_R(0x3c00));  // scale: 1.0, bias: 0.0 in FP16

    FARF(HIGH, "%s: grouped path m=%d k=%d n=%d group=%d streams=%d mc=%zu nc=%zu vtcm=%zu/%zu",
            __func__, params->m, params->k, params->n, group_size, params->ne13,
            m_chunk_n_rows, n_chunk_n_cols,
            L.total_bytes, vtcm_budget);

    ثابت size_t fp16_row_bytes   = (size_t) params->k * حجم(__fp16);
    ثابت size_t weight_row_bytes = (size_t) params->weight_stride;

    htp_trace_event_stop(tr, HTP_TRACE_EVT_INIT, 0);

    hmx_matmul_job_t job;

    لكل (صحيح b3 = 0; b3 < params->ne13; ++b3) {
        لكل (صحيح b2_base = 0; b2_base < params->ne12; b2_base += group_size) {
            ثابت dma_addr_t weight_group = hmx_mm_weight_batch_data(params, b2_base, b3);
            dma_queue * weight_dma = params->weight_dma;

            لكل (size_t mr = 0; mr < (size_t) params->m; mr += m_chunk_n_rows) {
                ثابت size_t n_rows = hex_smin((size_t) params->m - mr, m_chunk_n_rows);
                ثابت size_t n_row_tiles = hmx_ceil_div((صحيح) n_rows, HTP_MM_HMX_TILE_N_ROWS);

                // Pre-load activations for all heads in the group (once per m_chunk).
                // When the source is strided (permuted Q), use 2D DMA to gather
                // contiguous rows into a VTCM scratch buffer first, then HVX
                // converts from the contiguous VTCM buffer.  This avoids L2 cache
                // thrashing from HVX loads at large strides.
                لكل (صحيح g = 0; g < group_size; ++g) {
                    ثابت dma_addr_t act_dma_addr = hmx_mm_act_batch_addr(params, b2_base + g, b3) +
                                                      (size_t) mr * params->act_stride * params->act_elem_size;
                    __fp16 *vtcm_act_g = vtcm_f16_act + (size_t) g * L.act_head_stride;
                    بنية activation_transfer_params act_params = {
                        .ctx = ctx,
                        .dst = vtcm_act_g,
                        .act_dma_addr = act_dma_addr,
                        .n_rows = (صحيح) n_rows,
                        .k_block = params->k,
                        .k_stride = params->act_stride,
                        .n_threads = act_threads,
                        .act_threads_div = act_threads_div,
                        .k_div = k_div,
                        .k_valid = params->k,
                        .vtcm_f32_act = vtcm_f32_act,
                        .vtcm_f32_act_bytes = L.act_f32_bytes,
                        .act_elem_size = params->act_elem_size,
                    };
                    transfer_activation_chunk_threaded(&act_params);
                }

                // Prologue: Push A0 and A1 (if exists)
                {
                    ثابت size_t n_cols_first = hex_smin((size_t) params->n, n_chunk_n_cols);
                    dma_queue_push(weight_dma, dma_make_data(vtcm_scratch0, weight_group),
                                      fp16_row_bytes, weight_row_bytes, fp16_row_bytes, n_cols_first);
                }
                ܐܢ (n_chunk_n_cols < (size_t) params->n) {
                    ثابت size_t n_cols_second = hex_smin((size_t) params->n - n_chunk_n_cols, n_chunk_n_cols);
                    ثابت dma_addr_t second_weight_chunk = weight_group + n_chunk_n_cols * weight_row_bytes;
                    dma_queue_push(weight_dma, dma_make_data(vtcm_scratch1, second_weight_chunk),
                                      fp16_row_bytes, weight_row_bytes, fp16_row_bytes, n_cols_second);
                }

                لكل (size_t nc = 0; nc < (size_t) params->n; nc += n_chunk_n_cols) {
                    ثابت size_t n_cols      = hex_smin((size_t) params->n - nc, n_chunk_n_cols);
                    ثابت size_t n_col_tiles = hmx_ceil_div((صحيح) n_cols, HTP_MM_HMX_TILE_N_COLS);

                    {
                        فراغ * curr_raw = (فراغ *) dma_queue_pop(weight_dma).dst;

                        ثابت size_t n_cols_tiled = hex_align_up(n_cols, HTP_MM_HMX_TILE_N_COLS);

                        hmx_interleave_rows_to_tiles(vtcm_weight, (ثابت __fp16 *) curr_raw, (uint32_t) n_cols, params->k, params->k, 0, (uint32_t) n_cols_tiled);

                        ثابت size_t nc_next = nc + n_chunk_n_cols * 2;
                        ܐܢ (nc_next < (size_t) params->n) {
                            ثابت size_t n_cols_next = hex_smin((size_t) params->n - nc_next, n_chunk_n_cols);
                            ثابت dma_addr_t next_weight_chunk = weight_group + nc_next * weight_row_bytes;

                            dma_queue_push(weight_dma, dma_make_data(curr_raw, next_weight_chunk),
                                              fp16_row_bytes, weight_row_bytes, fp16_row_bytes, n_cols_next);
                        }
                    }

                    // Reuse the interleaved weight for every q_head in this GQA group
                    لكل (صحيح g = 0; g < group_size; ++g) {
                        {
                            ثابت __fp16 * vtcm_act_g = vtcm_f16_act + (size_t) g * L.act_head_stride;
                            hmx_matmul_job_init(&job, vtcm_output, vtcm_act_g, vtcm_weight, vtcm_scales, n_row_tiles, n_col_tiles, params->k / 32);
                            hmx_queue_push(ctx->hmx_queue, hmx_queue_make_desc(hmx_matmul_worker_fn, &job));
                            hmx_queue_pop(ctx->hmx_queue);
                        }

                        {
                            عائم *output = hmx_mm_dst_batch_ptr(params, b2_base + g, b3) + mr * params->dst_stride + nc;
                            ثابت عائم *bias_chunk = has_bias ? (vtcm_bias + mr * params->bias_stride + nc) : عدم;
                            صحيح chunk_dst_cols = params->dst_cols - (صحيح)nc;
                            ܐܢ (chunk_dst_cols > 0) {
                                transfer_output_chunk_threaded(ctx, output, bias_chunk, vtcm_output, (صحيح) n_rows, (صحيح) n_cols,
                                                               params->dst_stride, params->bias_stride, chunk_dst_cols, n_threads);
                            }
                        }
                    }
                }
            }
        }
    }

    ܦܢܐ 0;
}

ساكن فراغ transfer_activation_chunk_gathered_threaded(
            بنية htp_context *ctx,
            __fp16 *dst,
            ثابت عائم *src,
            صحيح start_row,
            صحيح n_rows,
            صحيح k_block,
            ثابت بنية mmid_row_mapping *matrix_rows,
            صحيح cur_a,
            صحيح mapping_stride,
            صحيح ne11,
            size_t nb11,
            size_t nb12,
            صحيح cne1,
            صحيح n_threads,
            صحيح k_valid) {
    ܐܢ (n_rows <= 0) ܦܢܐ;
    صحيح chunks_per_thread = hmx_ceil_div(n_rows, n_threads);
    chunks_per_thread = hex_align_up(chunks_per_thread, 2);

    صحيح actual_threads = hmx_ceil_div(n_rows, chunks_per_thread);

    activation_transfer_gathered_task_state_t state = {
        .dst               = dst,
        .src               = src,
        .n_tasks           = actual_threads,
        .n_tot_chunks      = n_rows,
        .n_chunks_per_task = chunks_per_thread,
        .k_block           = k_block,
        .matrix_rows       = matrix_rows,
        .cur_a             = cur_a,
        .mapping_stride    = mapping_stride,
        .ne11              = ne11,
        .ne11_div          = ne11 > 1 ? init_fastdiv_values(ne11) : (بنية fastdiv_values){0, 0},
        .nb11              = nb11,
        .nb12              = nb12,
        .start_row         = start_row,
        .cne1              = cne1,
        .k_valid           = k_valid,
        .traces            = ctx->trace,
    };

    worker_callback_t worker_fn = ne11 == 1 ? transfer_activation_chunk_gathered_worker_flat_fn :
                                              transfer_activation_chunk_gathered_worker_fn;

    ܐܢ (actual_threads <= 1) {
        worker_fn(1, 0, &state);
    } ܐܚܪܢܐ {
        worker_pool_run_func(ctx->worker_pool, worker_fn, &state, actual_threads);
    }
}

ساكن فراغ transfer_output_chunk_scattered_threaded(
            بنية htp_context *ctx,
            عائم *dst,
            ثابت __fp16 *vtcm_src,
            صحيح start_row,
            صحيح n_rows,
            صحيح n_cols,
            ثابت بنية mmid_row_mapping *matrix_rows,
            صحيح cur_a,
            صحيح mapping_stride,
            size_t dst_nb1,
            size_t dst_nb2,
            صحيح cne1,
            صحيح n_threads) {
    ܐܢ (n_rows <= 0) ܦܢܐ;
    صحيح chunks_per_thread = hmx_ceil_div(n_rows, n_threads);
    chunks_per_thread = hex_align_up(chunks_per_thread, 2);

    صحيح actual_threads = hmx_ceil_div(n_rows, chunks_per_thread);

    output_transfer_scattered_task_state_t state = {
        .vtcm_src          = vtcm_src,
        .dst               = dst,
        .n_tasks           = actual_threads,
        .n_tot_chunks      = n_rows,
        .n_chunks_per_task = chunks_per_thread,
        .n_cols            = n_cols,
        .matrix_rows       = matrix_rows,
        .cur_a             = cur_a,
        .mapping_stride    = mapping_stride,
        .dst_nb1           = dst_nb1,
        .dst_nb2           = dst_nb2,
        .start_row         = start_row,
        .cne1              = cne1,
        .traces            = ctx->trace,
    };

    ܐܢ (actual_threads <= 1) {
        transfer_output_chunk_scattered_worker_fn(1, 0, &state);
    } ܐܚܪܢܐ {
        worker_pool_run_func(ctx->worker_pool, transfer_output_chunk_scattered_worker_fn, &state, actual_threads);
    }
}

ساكن صحيح hmx_mm_id_2d_f32(بنية htp_context *ctx,
                                         dma_queue *weight_dma,
                                         عائم *restrict dst,
                                         ثابت عائم *activation,
                                         dma_addr_t weight,
                                         صحيح m, صحيح k, صحيح n,
                                         صحيح k_valid,
                                         صحيح ne11,
                                         size_t act_nb1, size_t act_nb2,
                                         size_t dst_nb1, size_t dst_nb2,
                                         صحيح weight_stride,
                                         صحيح weight_type,
                                         صحيح tile_size,
                                         صحيح aligned_tile_size,
                                         ثابت بنية mmid_row_mapping *matrix_rows,
                                         صحيح cur_a,
                                         صحيح mapping_stride,
                                         صحيح m_start,
                                         صحيح m_end,
                                         صحيح n_threads) {
    بنية htp_thread_trace * tr = &ctx->trace[0];
    htp_trace_event_start(tr, HTP_TRACE_EVT_INIT, 0);

    ثابت صحيح cne1 = m;
    ثابت صحيح m_core = m_end - m_start;
    ثابت صحيح m_core_padded = hex_align_up(m_core > 0 ? m_core : 1, 32);

    ܐܢ (k % 32 != 0 || n % 32 != 0) { ܦܢܐ -1; }
    ܐܢ (!hex_is_aligned(dst, VLEN) || !hex_is_aligned(activation, VLEN)) { ܦܢܐ -1; }

    size_t row_stride = htp_tensor_get_row_size(weight_type, k);
    ܐܢ (row_stride == 0) {
        ܦܢܐ -1;
    }

    ثابت صحيح n_k_tiles = k / HTP_MM_HMX_TILE_N_COLS;
    ثابت uint32_t n_weight_tiles = k / htp_tensor_get_tile_k(weight_type);
    ثابت بنية fastdiv_values n_k_tiles_div = init_fastdiv_values(n_k_tiles);

    ثابت size_t vec_dot_size = k * حجم(__fp16);
    ثابت size_t vtcm_budget  = ctx->vtcm_size;
    size_t vtcm_used = 0;

    ثابت bool is_tiled = (tile_size > 0);
    tiled_dequant_task_fn_t dequant_worker_fn = hmx_get_dequant_worker_fn(weight_type, is_tiled);

    ثابت uint32_t dma_dst_stride  = is_tiled ? aligned_tile_size : row_stride;
    ثابت uint32_t dma_src_stride  = is_tiled ? tile_size : weight_stride;
    ثابت uint32_t dma_width_bytes = is_tiled ? tile_size : row_stride;

    ثابت size_t qweight_row_stride = is_tiled ? (size_t)(n_weight_tiles * aligned_tile_size) / 32 : 0;
    ثابت size_t weight_row_stride = is_tiled ? qweight_row_stride : row_stride;

    size_t size_per_n = 0, size_per_m = 0, size_per_mn = 0;
    htp_mm_hmx_get_2d_chunk_costs(weight_type, k, /*pipeline=*/false, aligned_tile_size,
                                  &size_per_n, &size_per_m, &size_per_mn);

    ثابت size_t overhead = htp_mm_hmx_get_2d_overhead(/*pipeline=*/false, /*is_matmul_id=*/true);
    size_t m_chunk_n_rows = 0, n_chunk_n_cols = 0;
    ܐܢ (htp_mm_hmx_compute_chunks(vtcm_budget, overhead, size_per_n, size_per_m, size_per_mn,
                           m_core_padded, n,
                           /*m_block_cost=*/(size_t) n * HTP_MM_HMX_COST_W_DEQUANT,
                           /*n_block_cost=*/(size_t) m_core_padded * HTP_MM_HMX_COST_A_CONVERT, &m_chunk_n_rows, &n_chunk_n_cols, &vtcm_used)) {
        FARF(ERROR, "hmx-mm-id-2d: VTCM too small : m %d k %d n %d budget %zu", m_core_padded, k, n, vtcm_budget);
        ܦܢܐ -1;
    }

    ثابت size_t weight_area_size = hex_align_up(n_chunk_n_cols * weight_row_stride, HTP_MM_HMX_TILE_SIZE);
    ثابت size_t act_area_size    = hex_align_up(m_chunk_n_rows * vec_dot_size, HTP_MM_HMX_TILE_SIZE);
    ثابت size_t output_area_size = hex_align_up(m_chunk_n_rows * n_chunk_n_cols * حجم(__fp16), HTP_MM_HMX_TILE_SIZE);

    size_t scratch0_size = hex_align_up(n_chunk_n_cols * vec_dot_size, HTP_MM_HMX_TILE_SIZE);

    uint8_t *vtcm_ptr      = (uint8_t *) ctx->vtcm_base;
    __fp16  *vtcm_weight   = weight_area_size ? (__fp16 *) vtcm_seq_alloc(&vtcm_ptr, weight_area_size) : عدم;
    __fp16  *vtcm_f16_act  = (__fp16 *) vtcm_seq_alloc(&vtcm_ptr, act_area_size);
    __fp16  *vtcm_output   = (__fp16 *) vtcm_seq_alloc(&vtcm_ptr, output_area_size);
    فراغ    *vtcm_scratch0 = vtcm_seq_alloc(&vtcm_ptr, scratch0_size);
    __fp16  *vtcm_scales   = (__fp16 *) vtcm_seq_alloc(&vtcm_ptr, 256);

    vtcm_used = vtcm_ptr - (uint8_t *) ctx->vtcm_base;
    ܐܢ (vtcm_used > vtcm_budget) {
        FARF(ERROR, "hmx-mm-id-2d: VTCM overflow: used %zu budget %zu", vtcm_used, vtcm_budget);
        ܦܢܐ -1;
    }

    hmx_init_column_scales(vtcm_scales, Q6_V_vsplat_R(0x3c00));

    htp_trace_event_stop(tr, HTP_TRACE_EVT_INIT, 0);

    hmx_matmul_job_t job;
    فراغ *vtcm_raw_bufs[2] = { vtcm_weight, dequant_worker_fn ? vtcm_weight : vtcm_scratch0 };
    فراغ *vtcm_weight_bufs[2] = { dequant_worker_fn ? vtcm_scratch0 : vtcm_weight, vtcm_scratch0 };
    ثابت صحيح n_chunk_cnt = hmx_ceil_div(n, n_chunk_n_cols);

    لكل (size_t mr = (size_t) m_start; mr < (size_t) m_end; mr += m_chunk_n_rows) {
        ثابت size_t n_rows = hex_smin((size_t) m_end - mr, m_chunk_n_rows);
        ثابت size_t n_row_tiles = hmx_ceil_div(n_rows, HTP_MM_HMX_TILE_N_ROWS);

        transfer_activation_chunk_gathered_threaded(
            ctx, vtcm_f16_act, activation, (صحيح) mr, (صحيح) n_rows, k,
            matrix_rows, cur_a, mapping_stride, ne11, act_nb1, act_nb2, cne1, n_threads, k_valid);

        // A0: Pre-fetch the first weight chunk (nc = 0)
        ܐܢ (n > 0) {
            ثابت size_t n_cols = hex_smin((size_t) n, n_chunk_n_cols);
            ثابت uint32_t height = is_tiled ? hmx_ceil_div(n_cols, 32) * n_weight_tiles : n_cols;
            dma_queue_push(weight_dma, dma_make_data(vtcm_raw_bufs[0], weight),
                           dma_dst_stride, dma_src_stride, dma_width_bytes, height);
        }

        لكل (صحيح i = 0; i < n_chunk_cnt; ++i) {
            ثابت size_t nc = (size_t) i * n_chunk_n_cols;
            ثابت size_t n_cols = hex_smin((size_t) n - nc, n_chunk_n_cols);
            ثابت size_t n_col_tiles = hmx_ceil_div(n_cols, HTP_MM_HMX_TILE_N_COLS);

            // A: Wait for weight DMA
            فراغ * curr_raw = (فراغ *) dma_queue_pop(weight_dma).dst;

            // B: Weight Dequantize (Threaded)
            dequantize_tiled_weight_chunk_to_fp16_tiles(
                ctx, vtcm_weight_bufs[i % 2], curr_raw,
                n_cols, k, row_stride, weight_type,
                n_k_tiles, n_k_tiles_div, dequant_worker_fn, n_threads
            );

            // Start weight DMA for the next chunk early
            ثابت size_t nc_next = nc + n_chunk_n_cols;
            ܐܢ (nc_next < (size_t) n) {
                ثابت size_t n_cols_next = hex_smin((size_t) n - nc_next, n_chunk_n_cols);
                ثابت uint32_t height_next = is_tiled ? hmx_ceil_div(n_cols_next, 32) * n_weight_tiles : n_cols_next;
                dma_queue_push(weight_dma, dma_make_data(vtcm_raw_bufs[(i + 1) % 2], weight + nc_next * weight_stride),
                               dma_dst_stride, dma_src_stride, dma_width_bytes, height_next);
            }

            // C: HMX Compute (Queue-based)
            hmx_matmul_job_init(&job, vtcm_output, vtcm_f16_act, (__fp16 *) vtcm_weight_bufs[i % 2], vtcm_scales, n_row_tiles, n_col_tiles, k / HTP_MM_HMX_TILE_N_ROWS);
            hmx_queue_push(ctx->hmx_queue, hmx_queue_make_desc(hmx_matmul_worker_fn, &job));
            hmx_queue_pop(ctx->hmx_queue);

            // D: Output Store
            transfer_output_chunk_scattered_threaded(
                ctx, dst + nc, vtcm_output, (صحيح) mr, (صحيح) n_rows, (صحيح) n_cols,
                matrix_rows, cur_a, mapping_stride, dst_nb1, dst_nb2, cne1, n_threads);
        }
    }

    ܦܢܐ 0;
}

// --- Dispatchers and Public Entry Points ---

ساكن صحيح hmx_mm_op_matmul(بنية htp_ops_context * octx, ثابت بنية htp_mm_kernel_params * kparams) {
    htp_matmul_tensors_preamble;

    صحيح k = (صحيح) src0->ne[0];
    صحيح n = (صحيح) src0->ne[1];
    ثابت صحيح m_total    = (صحيح) act->ne[1];
    ثابت uint32_t act_elem_size = (act->type == HTP_TYPE_F16) ? حجم(__fp16) : حجم(عائم);
    ثابت صحيح act_stride = (صحيح) (act->nb[1] / act_elem_size);

    صحيح m_start = 0;
    صحيح m_rows  = m_total;
    ܐܢ (octx->ctx->mdev.count > 1) {
        bool can_split = htp_tensor_can_row_partition(dst, حجم(عائم)) &&
                         htp_tensor_can_row_partition(act, act_elem_size);
        ܐܢ (src2 && src2->ne[1] > 1 && !htp_tensor_can_row_partition(src2, حجم(عائم))) {
            can_split = false;
        }
        ثابت بنية htp_tensor_mdev_range range = htp_tensor_mdev_partition((uint32_t) m_total, can_split ? 1 : 0, octx->ctx->mdev.idx, octx->ctx->mdev.count, &octx->ctx->mdev.count_div);
        m_start = (صحيح) range.start;
        m_rows  = (صحيح) range.count;
    }

    ܐܢ (m_rows == 0) {
        ܦܢܐ HTP_STATUS_OK;
    }

    dma_addr_t src2_addr = 0;
    size_t src2_bytes = 0;
    uint32_t src2_stride = 0;
    size_t src2_nb2 = 0;
    size_t src2_nb3 = 0;
    ܐܢ (src2) {
        src2_stride = (src2->ne[1] == 1) ? 0 : (uint32_t) (src2->nb[1] / حجم(عائم));
        src2_addr = src2->data + m_start * src2_stride * حجم(عائم);
        src2_bytes = (size_t) kparams->vtcm_bias_size;
        src2_nb2 = (src2->ne[2] == 1) ? 0 : src2->nb[2];
        src2_nb3 = (src2->ne[3] == 1) ? 0 : src2->nb[3];
    }

    ثابت صحيح dst_stride = (صحيح)(dst->nb[1] / حجم(عائم));
    عائم       * dst_ptr = (عائم *) dst->data + m_start * dst_stride;
    // byte offset, act_stride is in activation elements (F16 or F32)
    ثابت dma_addr_t act_addr = act->data + (size_t) m_start * act_stride * act_elem_size;

    صحيح ret = -1;
    ثابت صحيح n_threads = kparams->n_threads;
    ܐܢ (kparams->kernel_type == HTP_MM_KERNEL_HMX_BATCHED) {
        hmx_mm_batched_params_t batch_params = {
            .dst               = dst_ptr,
            .bias_addr         = src2_addr,
            .bias_bytes        = src2_bytes,
            .act_dma_addr      = act_addr,
            .weight            = src0->data,
            .weight_dma        = octx->ctx->dma[0],
            .m                 = m_rows,
            .k                 = k,
            .n                 = n,
            .act_stride        = act_stride,
            .act_elem_size     = act_elem_size,
            .weight_stride     = (صحيح) src0->nb[1],
            .weight_type       = (صحيح) src0->type,
            .dst_stride        = dst_stride,
            .dst_cols          = (صحيح) dst->ne[0],
            .bias_stride       = src2_stride,
            .tile_size         = kparams->tile_size,
            .aligned_tile_size = kparams->aligned_tile_size,
            .ne02              = ne02,
            .ne03              = ne03,
            .ne12              = ne12,
            .ne13              = ne13,
            .src0_nb2          = src0->nb[2],
            .src0_nb3          = src0->nb[3],
            .act_nb2           = act->nb[2],
            .act_nb3           = act->nb[3],
            .dst_nb2           = dst->nb[2],
            .dst_nb3           = dst->nb[3],
            .bias_nb2          = src2_nb2,
            .bias_nb3          = src2_nb3,
            .r2                = (ne02 > 0) ? (ne12 / ne02) : 1,
            .r3                = (ne03 > 0) ? (ne13 / ne03) : 1,
            .div_r2            = kparams->div_r2,
            .div_r3            = kparams->div_r3,
        };
        ret = hmx_mm_batched(octx->ctx, &batch_params,
                             kparams->m_chunk, kparams->n_chunk,
                             kparams->pipeline, n_threads,
                             kparams->n_act_threads,
                             &kparams->div_n_act_threads,
                             &kparams->div_ne00_padded,
                             kparams->vtcm_size);
    } ܐܚܪܢܐ {
        ret = hmx_mm_2d_f32(
            octx->ctx, octx->ctx->dma[0], dst_ptr, src2_addr, src2_bytes,
            act_addr, src0->data,
            m_rows, k, n, act_stride, act_elem_size, (صحيح) src0->nb[1], (صحيح) src0->type, (صحيح) act->ne[0],
            dst_stride, src2_stride, (صحيح)dst->ne[0],
            kparams->m_chunk, kparams->n_chunk, kparams->pipeline, n_threads,
            kparams->n_act_threads,
            &kparams->div_n_act_threads,
            &kparams->div_ne00_padded,
            kparams->tile_size, kparams->aligned_tile_size, kparams->vtcm_size
        );
    }

    ܐܢ (ret != 0) {
        FARF(ERROR, "HMX matmul failed (ret=%d)\n", ret);
        ܦܢܐ HTP_STATUS_INTERNAL_ERR;
    }
    ܦܢܐ HTP_STATUS_OK;
}

ساكن مضمن فراغ htp_mm_tensor_collapse_rows(بنية htp_tensor * c, ثابت بنية htp_tensor * t, uint32_t stride) {
    *c = *t;
    c->ne[1] = t->ne[1] * t->ne[2] * t->ne[3];
    c->ne[2] = 1;
    c->ne[3] = 1;
    c->nb[1] = stride;
    c->nb[2] = c->nb[1] * c->ne[1];
    c->nb[3] = c->nb[2];
}

ساكن صحيح op_matmul_impl(بنية htp_ops_context * octx) {
    ثابت بنية htp_mm_kernel_params * kparams = (ثابت بنية htp_mm_kernel_params *) octx->kernel_params;

    ثابت صحيح status = htp_mm_init_context(octx, kparams);
    ܐܢ (status != HTP_STATUS_OK) {
        ܦܢܐ status;
    }

    ܐܢ (kparams->n_hmx) {
        ܦܢܐ hmx_mm_op_matmul(octx, kparams);
    }

    ܦܢܐ hvx_mm_matmul(octx);
}

صحيح op_matmul(بنية htp_ops_context * octx) {
    ثابت بنية htp_mm_kernel_params * kparams = (ثابت بنية htp_mm_kernel_params *) octx->kernel_params;

    ܐܢ (kparams->collapse) {
        ثابت بنية htp_tensor * act  = octx->src[1];
        ثابت بنية htp_tensor * dst  = octx->dst;
        ثابت uint32_t s_act = (act->ne[1] > 1) ? act->nb[1] : ((act->ne[2] > 1) ? act->nb[2] : act->nb[3]);
        ثابت uint32_t sd    = (dst->ne[1] > 1) ? dst->nb[1] : ((dst->ne[2] > 1) ? dst->nb[2] : dst->nb[3]);
        بنية htp_tensor act_collapsed, dst_collapsed;
        htp_mm_tensor_collapse_rows(&act_collapsed, act, s_act);
        htp_mm_tensor_collapse_rows(&dst_collapsed, dst, sd);
        octx->src[1] = &act_collapsed;
        octx->dst    = &dst_collapsed;

        بنية htp_tensor src2_collapsed;
        ثابت بنية htp_tensor * src2 = octx->src[2];
        ܐܢ (src2 && (src2->ne[1] * src2->ne[2] * src2->ne[3] > 1)) {
            ثابت uint32_t s2 = (src2->ne[1] > 1) ? src2->nb[1] : ((src2->ne[2] > 1) ? src2->nb[2] : src2->nb[3]);
            htp_mm_tensor_collapse_rows(&src2_collapsed, src2, s2);
            octx->src[2] = &src2_collapsed;
        }

        ثابت صحيح status = op_matmul_impl(octx);
        octx->src[1] = act;
        octx->dst    = dst;
        ܐܢ (src2) {
            octx->src[2] = src2;
        }
        ܦܢܐ status;
    }

    ܦܢܐ op_matmul_impl(octx);
}

ساكن صحيح hmx_mm_op_matmul_id(
    بنية htp_ops_context * octx,
    بنية htp_mm_context * mmctx
) {
    ثابت uint32_t *                matrix_row_counts = mmctx->matrix_row_counts;
    ثابت بنية mmid_row_mapping * matrix_rows       = mmctx->matrix_rows;
    htp_matmul_tensors_preamble;
    ثابت بنية htp_mm_kernel_params * kparams = (ثابت بنية htp_mm_kernel_params *) octx->kernel_params;
    ثابت صحيح n_ids = octx->src[2]->ne[0];
    ثابت صحيح n_as  = ne02;

    ثابت bool mdev_split = (octx->ctx->mdev.count > 1) && htp_tensor_can_row_partition(dst, حجم(عائم));
    ܐܢ (octx->ctx->mdev.count > 1 && !mdev_split && octx->ctx->mdev.idx > 0) {
        ܦܢܐ HTP_STATUS_OK;
    }
    uint32_t n_active = 0;
    ܐܢ (mdev_split) {
        لكل (uint32_t a = 0; a < (uint32_t) n_as; ++a) {
            ܐܢ (matrix_row_counts[a] > 0) n_active++;
        }
    }
    ثابت bool expert_split = mdev_split && (n_active >= octx->ctx->mdev.count);

    uint32_t target_dev = 0;
    لكل (uint32_t cur_a = 0; cur_a < (uint32_t) n_as; ++cur_a) {
        ثابت int32_t cne1 = matrix_row_counts[cur_a];
        ܐܢ (cne1 == 0) استمر;

        ثابت صحيح m_padded = hex_align_up(cne1, 32);
        صحيح m_start = 0, m_end = m_padded;
        ܐܢ (expert_split) {
            ثابت bool my_expert = (target_dev == octx->ctx->mdev.idx);
            ܐܢ (++target_dev == octx->ctx->mdev.count) {
                target_dev = 0;
            }
            ܐܢ (!my_expert) استمر;
        } ܐܚܪܢܐ ܐܢ (mdev_split) {
            ثابت bool can_split = (uint32_t) cne1 >= octx->ctx->mdev.count;
            ثابت بنية htp_tensor_mdev_range range = htp_tensor_mdev_partition((uint32_t) m_padded, can_split ? 1 : 0, octx->ctx->mdev.idx, octx->ctx->mdev.count, &octx->ctx->mdev.count_div);
            m_start = (صحيح) range.start;
            m_end   = (صحيح) (range.start + range.count);
        }
        ܐܢ (m_start >= m_end) استمر;

        صحيح ret = hmx_mm_id_2d_f32(octx->ctx, octx->ctx->dma[0], (عائم*) dst->data, (عائم*) act->data,
                                   src0->data + cur_a * nb02,
                                   cne1, ne00, ne01,
                                   ne10,
                                   ne11,
                                   nb11, nb12,
                                   nb1, nb2,
                                   (صحيح) src0->nb[1], (صحيح) src0->type,
                                   kparams->tile_size, kparams->aligned_tile_size,
                                   matrix_rows, cur_a, mmctx->mapping_stride,
                                   m_start, m_end, (صحيح) octx->n_threads);
        ܐܢ (ret != 0) {
            FARF(ERROR, "HMX matmul failed for expert %u, error %d\n", cur_a, ret);
            ܦܢܐ HTP_STATUS_NO_SUPPORT;
        }
    }

    ܦܢܐ HTP_STATUS_OK;
}

ساكن صحيح hvx_mm_matmul_id(
    بنية htp_ops_context * octx,
    بنية htp_mm_context * mmctx,
    work_queue_func_t hvx_mmid_task_func
) {
    htp_matmul_tensors_preamble;
    ثابت uint32_t src0_row_size_padded = mmctx->src0_row_size_padded;
    ثابت uint32_t act_nrows            = mmctx->act_nrows;

    بنية htp_thread_trace * tr = &octx->ctx->trace[0];
    htp_trace_event_start(tr, HTP_TRACE_EVT_INIT, 0);

    ثابت بنية htp_mm_kernel_params * kparams = (ثابت بنية htp_mm_kernel_params *) octx->kernel_params;
    ثابت بنية htp_tensor * restrict ids = octx->src[2];
    ثابت size_t src0_row_size = nb01;

    ثابت uint32_t qk = QK_Q8_0_TILED;
    ثابت uint32_t nb = (ne10 + qk - 1) / qk;
    ثابت uint32_t total_nb = act_nrows * nb;

    work_queue_func_t quant_task_func;
    uint32_t n_quant_tasks = 1;
    ܐܢ (act_nrows < octx->n_threads) {
        n_quant_tasks = MIN(total_nb, octx->n_threads);
        quant_task_func = htp_mm_act_quant_block_func(src0->type, act->type);
        لكل (uint32_t ith = 0; ith < n_quant_tasks; ++ith) {
            uint32_t ib_first = (total_nb * ith) / n_quant_tasks;
            uint32_t ib_last  = (total_nb * (ith + 1)) / n_quant_tasks;
            mmctx->quant_ib_first[ith] = ib_first;
            mmctx->quant_ib_last[ith]  = ib_last;
            mmctx->quant_r[ith]        = ib_first / nb;
            mmctx->quant_c[ith]        = ib_first % nb;
        }
    } ܐܚܪܢܐ {
        n_quant_tasks = MIN(act_nrows, octx->n_threads);
        quant_task_func = htp_mm_act_quant_row_func(src0->type, act->type);
    }
    size_t act_row_size = htp_mm_tiled_act_row_size(src0->type, ne10);

    بنية htp_mm_hvx_vtcm_layout L;
    htp_mm_hvx_vtcm_layout_build(&L, kparams->kernel_type, src0->type, ne10, act_nrows, octx->n_threads,
                                 0, src0_row_size, act_row_size, 0, kparams->n_prefetch, true, false);

    ثابت size_t vtcm_size = L.total_bytes;

    FARF(HIGH, "matmul-id-%s : src0-spad-size %zu act-spad-size %zu bias-spad-size 0 dst-spad-size %zu (%zu)\n", mmctx->type,
         L.src0_bytes, L.act_bytes, L.dst_bytes, vtcm_size);

    FARF(HIGH, "matmul-id-%s : %ux%ux%ux%u * %ux%ux%ux%u (%ux%ux%ux%u) -> %ux%ux%ux%u (0x%p, 0x%p, 0x%p)\n", mmctx->type,
         src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3], act->ne[0], act->ne[1], act->ne[2], act->ne[3],
         ids->ne[0], ids->ne[1], ids->ne[2], ids->ne[3], dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3], src0->data,
         act->data, dst->data);

    // Make sure the reserved vtcm size is sufficient
    ܐܢ (octx->ctx->vtcm_size < vtcm_size) {
        FARF(ERROR, "matmul-id-%s : current VTCM reservation %zu is too small, needed %zu\n", mmctx->type, octx->ctx->vtcm_size, vtcm_size);
        ܦܢܐ HTP_STATUS_VTCM_TOO_SMALL;
    }

    uint8_t * ثابت base = (uint8_t *) octx->ctx->vtcm_base;
    mmctx->vtcm_act      = VTCM_LAYOUT_PTR(uint8_t, base, L.off_act);
    mmctx->vtcm_src0     = VTCM_LAYOUT_PTR(uint8_t, base, L.off_src0);
    mmctx->vtcm_bias     = عدم;
    mmctx->vtcm_dst      = VTCM_LAYOUT_PTR(uint8_t, base, L.off_dst);
    mmctx->vtcm_act_raw  = VTCM_LAYOUT_PTR(uint8_t, base, L.off_act_raw);

    mmctx->vtcm_src0_stride    = src0_row_size_padded;
    mmctx->vtcm_act_stride     = act_row_size;
    mmctx->vtcm_act_raw_stride = hex_round_up(ne10 * حجم(عائم), QK_Q8_0_TILED * حجم(عائم));

    mmctx->vtcm_src0_size_per_thread = fastdiv(L.src0_bytes, &octx->n_threads_div);
    mmctx->vtcm_act_size_per_thread  = L.act_bytes;
    mmctx->vtcm_dst_size_per_thread  = fastdiv(L.dst_bytes, &octx->n_threads_div);

    mmctx->cur_m_start = 0;
    mmctx->cur_m_rows  = act_nrows;

    htp_trace_event_stop(tr, HTP_TRACE_EVT_INIT, 0);

    hvx_mm_transfer_act_dma(octx, kparams, act, mmctx->vtcm_act_raw, mmctx->vtcm_act_raw_stride, 0, act_nrows);

    mmctx->n_quant_rows_per_thread = (act_nrows + n_quant_tasks - 1) / n_quant_tasks;
    mmctx->n_quant_tasks = n_quant_tasks;
    work_queue_run(octx->ctx->work_queue, quant_task_func, mmctx, n_quant_tasks);

    work_queue_run(octx->ctx->work_queue, hvx_mmid_task_func, mmctx, octx->n_threads);

    ܦܢܐ HTP_STATUS_OK;
}

ساكن صحيح hmx_mm_op_matmul_id_nx(
    بنية htp_ops_context * octx,
    بنية htp_mm_context * mmctx
) {
    ثابت uint32_t *                matrix_row_counts = mmctx->matrix_row_counts;
    ثابت بنية mmid_row_mapping * matrix_rows       = mmctx->matrix_rows;
    ثابت بنية htp_mm_kernel_params * kparams = (ثابت بنية htp_mm_kernel_params *) octx->kernel_params;
    ثابت uint32_t n_weights = kparams->n_weights;
    ثابت بنية htp_tensor * restrict src0 = octx->src[0];
    ثابت بنية htp_tensor * restrict act  = octx->src[n_weights];
    ثابت صحيح n_as = src0->ne[2];

    bool mdev_split = (octx->ctx->mdev.count > 1);
    لكل (uint32_t p = 0; p < n_weights && mdev_split; ++p) {
        ثابت بنية htp_tensor * restrict dst = octx->dsts[p];
        mdev_split = htp_tensor_can_row_partition(dst, حجم(عائم));
    }
    ܐܢ (octx->ctx->mdev.count > 1 && !mdev_split && octx->ctx->mdev.idx > 0) {
        ܦܢܐ HTP_STATUS_OK;
    }
    uint32_t n_active = 0;
    ܐܢ (mdev_split) {
        لكل (uint32_t a = 0; a < (uint32_t) n_as; ++a) {
            ܐܢ (matrix_row_counts[a] > 0) n_active++;
        }
    }
    ثابت bool expert_split = mdev_split && (n_active >= octx->ctx->mdev.count);

    uint32_t target_dev = 0;
    لكل (uint32_t cur_a = 0; cur_a < (uint32_t) n_as; ++cur_a) {
        ثابت int32_t cne1 = matrix_row_counts[cur_a];
        ܐܢ (cne1 == 0) استمر;

        ثابت صحيح m_padded = hex_align_up(cne1, 32);
        صحيح m_start = 0, m_end = m_padded;
        ܐܢ (expert_split) {
            ثابت bool my_expert = (target_dev == octx->ctx->mdev.idx);
            ܐܢ (++target_dev == octx->ctx->mdev.count) {
                target_dev = 0;
            }
            ܐܢ (!my_expert) استمر;
        } ܐܚܪܢܐ ܐܢ (mdev_split) {
            ثابت bool can_split = (uint32_t) cne1 >= octx->ctx->mdev.count;
            ثابت بنية htp_tensor_mdev_range range = htp_tensor_mdev_partition((uint32_t) m_padded, can_split ? 1 : 0, octx->ctx->mdev.idx, octx->ctx->mdev.count, &octx->ctx->mdev.count_div);
            m_start = (صحيح) range.start;
            m_end   = (صحيح) (range.start + range.count);
        }
        ܐܢ (m_start >= m_end) استمر;

        لكل (uint32_t p = 0; p < n_weights; ++p) {
            ثابت بنية htp_tensor * restrict src_w = octx->src[p];
            ثابت بنية htp_tensor * restrict dst   = octx->dsts[p];
            ܐܢ (!src_w || !dst) استمر;

            صحيح ret = hmx_mm_id_2d_f32(octx->ctx, octx->ctx->dma[0], (عائم*) dst->data, (عائم*) act->data,
                                       src_w->data + cur_a * src_w->nb[2],
                                       cne1, src_w->ne[0], src_w->ne[1],
                                       act->ne[0],
                                       act->ne[1],
                                       act->nb[1], act->nb[2],
                                       dst->nb[1], dst->nb[2],
                                       (صحيح) src_w->nb[1], (صحيح) src_w->type,
                                       kparams->tile_size, kparams->aligned_tile_size,
                                       matrix_rows, cur_a, mmctx->mapping_stride,
                                       m_start, m_end, (صحيح) octx->n_threads);
            ܐܢ (ret != 0) {
                FARF(ERROR, "HMX matmul ID NX failed for expert %u weight %u, error %d\n", cur_a, p, ret);
                ܦܢܐ HTP_STATUS_NO_SUPPORT;
            }
        }
    }

    ܦܢܐ HTP_STATUS_OK;
}

ساكن صحيح hvx_mm_matmul_id_nx(
    بنية htp_ops_context * octx,
    بنية htp_mm_context * mmctx,
    work_queue_func_t hvx_mmid_task_func
) {
    ثابت uint32_t src0_row_size_padded = mmctx->src0_row_size_padded;
    ثابت uint32_t act_nrows            = mmctx->act_nrows;

    بنية htp_thread_trace * tr = &octx->ctx->trace[0];
    htp_trace_event_start(tr, HTP_TRACE_EVT_INIT, 0);

    ثابت بنية htp_mm_kernel_params * kparams = (ثابت بنية htp_mm_kernel_params *) octx->kernel_params;
    ثابت uint32_t n_weights = kparams->n_weights;
    ثابت بنية htp_tensor * restrict src0 = octx->src[0];
    ثابت بنية htp_tensor * restrict act  = octx->src[n_weights];
    ثابت بنية htp_tensor * restrict ids  = octx->src[n_weights + 1];
    ثابت size_t src0_row_size = src0->nb[1];

    ثابت uint32_t qk = QK_Q8_0_TILED;
    ثابت uint32_t nb = (act->ne[0] + qk - 1) / qk;
    ثابت uint32_t total_nb = act_nrows * nb;

    work_queue_func_t quant_task_func;
    uint32_t n_quant_tasks = 1;
    ܐܢ (act_nrows < octx->n_threads) {
        n_quant_tasks = MIN(total_nb, octx->n_threads);
        quant_task_func = htp_mm_act_quant_block_func(src0->type, act->type);
        لكل (uint32_t ith = 0; ith < n_quant_tasks; ++ith) {
            uint32_t ib_first = (total_nb * ith) / n_quant_tasks;
            uint32_t ib_last  = (total_nb * (ith + 1)) / n_quant_tasks;
            mmctx->quant_ib_first[ith] = ib_first;
            mmctx->quant_ib_last[ith]  = ib_last;
            mmctx->quant_r[ith]        = ib_first / nb;
            mmctx->quant_c[ith]        = ib_first % nb;
        }
    } ܐܚܪܢܐ {
        n_quant_tasks = MIN(act_nrows, octx->n_threads);
        quant_task_func = htp_mm_act_quant_row_func(src0->type, act->type);
    }
    size_t act_row_size = htp_mm_tiled_act_row_size(src0->type, act->ne[0]);

    بنية htp_mm_hvx_vtcm_layout L;
    htp_mm_hvx_vtcm_layout_build(&L, kparams->kernel_type, src0->type, act->ne[0], act_nrows, octx->n_threads,
                                 0, src0_row_size, act_row_size, 0, kparams->n_prefetch, true, false);

    ثابت size_t vtcm_size = L.total_bytes;

    ܐܢ (octx->ctx->vtcm_size < vtcm_size) {
        FARF(ERROR, "matmul-id-nx: current VTCM reservation %zu is too small, needed %zu\n",
             octx->ctx->vtcm_size, vtcm_size);
        ܦܢܐ HTP_STATUS_VTCM_TOO_SMALL;
    }

    uint8_t * ثابت base = (uint8_t *) octx->ctx->vtcm_base;
    mmctx->vtcm_src0     = VTCM_LAYOUT_PTR(uint8_t, base, L.off_src0);
    mmctx->vtcm_act      = VTCM_LAYOUT_PTR(uint8_t, base, L.off_act);
    mmctx->vtcm_dst      = VTCM_LAYOUT_PTR(uint8_t, base, L.off_dst);
    mmctx->vtcm_act_raw  = VTCM_LAYOUT_PTR(uint8_t, base, L.off_act_raw);

    mmctx->vtcm_src0_stride    = 0;
    mmctx->vtcm_act_stride     = act_row_size;
    mmctx->vtcm_act_raw_stride = hex_round_up(act->ne[0] * حجم(عائم), QK_Q8_0_TILED * حجم(عائم));

    mmctx->vtcm_src0_size_per_thread = fastdiv(L.src0_bytes, &octx->n_threads_div);
    mmctx->vtcm_act_size_per_thread  = L.act_bytes;
    mmctx->vtcm_dst_size_per_thread  = fastdiv(L.dst_bytes, &octx->n_threads_div);

    mmctx->cur_m_start = 0;
    mmctx->cur_m_rows  = act_nrows;

    FARF(HIGH, "matmul-id-nx: src0 %d:%d:%d type %s nrows %u, act %d:%d:%d nrows %u, vtcm %zu/%zu, threads %d\n",
         src0->ne[0], src0->ne[1], src0->ne[2], mmctx->type, src0->ne[1],
         act->ne[0], act->ne[1], act->ne[2], act_nrows,
         L.total_bytes, octx->ctx->vtcm_size, octx->n_threads);

    htp_trace_event_stop(tr, HTP_TRACE_EVT_INIT, 0);

    hvx_mm_transfer_act_dma(octx, kparams, act, mmctx->vtcm_act_raw, mmctx->vtcm_act_raw_stride, 0, act_nrows);

    mmctx->n_quant_rows_per_thread = (act_nrows + n_quant_tasks - 1) / n_quant_tasks;
    mmctx->n_quant_tasks           = n_quant_tasks;
    work_queue_run(octx->ctx->work_queue, quant_task_func, mmctx, n_quant_tasks);

    work_queue_run(octx->ctx->work_queue, hvx_mmid_task_func, mmctx, octx->n_threads);

    ܦܢܐ HTP_STATUS_OK;
}

ساكن مضمن فراغ scan_expert_ids_n(
    ثابت بنية htp_tensor * ids,
    ثابت uint32_t n_ids,
    uint32_t n_as,
    uint32_t * counts,
    بنية mmid_row_mapping * matrix_rows,
    uint32_t mapping_stride
) {
    ثابت size_t ids_nb1 = ids->nb[1];
    ثابت uint8_t * ids_data = (ثابت uint8_t *) ids->data;

    لكل (uint32_t iid1 = 0; iid1 < ids->ne[1]; ++iid1) {
        ثابت int32_t * row_ptr = (ثابت int32_t *) (ids_data + iid1 * ids_nb1);
        لكل (uint32_t id = 0; id < n_ids; ++id) {
            ثابت int32_t i02 = row_ptr[id];
            ܐܢ (i02 < 0) {
                استمر;
            }
            أكد(i02 < n_as);

            ܐܢ (matrix_rows) {
                matrix_rows[i02 * mapping_stride + counts[i02]] = (بنية mmid_row_mapping) { id, iid1 };
            }
            counts[i02] += 1;
        }
    }
}

ساكن مضمن فراغ scan_expert_ids(
    ثابت بنية htp_tensor * ids,
    uint32_t n_ids,
    uint32_t n_as,
    uint32_t * counts,
    بنية mmid_row_mapping * matrix_rows,
    uint32_t mapping_stride
) {
    ثابت size_t ids_nb0 = ids->nb[0];

    ܐܢ (ids_nb0 == 4) {
        بدل (n_ids) {
            حالة 8:  scan_expert_ids_n(ids, 8,     n_as, counts, matrix_rows, mapping_stride); اكسر;
            حالة 4:  scan_expert_ids_n(ids, 4,     n_as, counts, matrix_rows, mapping_stride); اكسر;
            حالة 2:  scan_expert_ids_n(ids, 2,     n_as, counts, matrix_rows, mapping_stride); اكسر;
            افتراضي: scan_expert_ids_n(ids, n_ids, n_as, counts, matrix_rows, mapping_stride); اكسر;
        }
    } ܐܚܪܢܐ {
        // Strided fallback
        ثابت size_t ids_nb1 = ids->nb[1];
        ثابت uint8_t * ids_data = (ثابت uint8_t *) ids->data;
        لكل (uint32_t iid1 = 0; iid1 < ids->ne[1]; ++iid1) {
            ثابت int32_t * row_ptr = (ثابت int32_t *) (ids_data + iid1 * ids_nb1);
            لكل (uint32_t id = 0; id < n_ids; ++id) {
                ثابت int32_t i02 = *(ثابت int32_t *) ((ثابت uint8_t *) row_ptr + id * ids_nb0);
                ܐܢ (i02 < 0) {
                    استمر;
                }
                أكد(i02 < n_as);

                ܐܢ (matrix_rows) {
                    matrix_rows[i02 * mapping_stride + counts[i02]] = (بنية mmid_row_mapping) { id, iid1 };
                }
                counts[i02] += 1;
            }
        }
    }
}

صحيح op_matmul_id(بنية htp_ops_context * octx) {
    htp_matmul_tensors_preamble;

    ثابت بنية htp_mm_kernel_params * kparams = (ثابت بنية htp_mm_kernel_params *) octx->kernel_params;
    بنية htp_mm_context mmctx_struct = {0};
    بنية htp_mm_context * mmctx = &mmctx_struct;

    ثابت صحيح status = htp_mm_init_context(octx, kparams);
    ܐܢ (status != HTP_STATUS_OK) {
        ܦܢܐ status;
    }

    بنية htp_thread_trace * tr = &octx->ctx->trace[0];
    htp_trace_event_start(tr, HTP_TRACE_EVT_INIT, 0);

    mmctx->octx = octx;
    mmctx->act = act;

    ثابت بنية htp_tensor * restrict ids = octx->src[2];
    ܐܢ (htp_tensor_is_extended(ids) || htp_tensor_is_extended(act) || htp_tensor_is_extended(dst)) {
        ܦܢܐ HTP_STATUS_NO_SUPPORT;
    }

    ثابت size_t src0_row_size = nb01;
    ثابت size_t dst_row_size  = nb1;

    ثابت size_t src0_row_size_padded = hex_round_up(src0_row_size, 128);

    ثابت uint32_t src0_nrows = ne01;  // per expert
    ثابت uint32_t act_nrows  = ne11 * ne12 * ne13;

    // row groups
    ثابت صحيح n_ids = ids->ne[0];  // n_expert_used
    ثابت صحيح n_as  = ne02;        // n_expert

    uint8_t  * mapping_buf       = octx->ctx->ddr_spad_base;
    uint32_t   mapping_stride    = 1;
    uint32_t * matrix_row_counts = (uint32_t *) mapping_buf;
    بنية mmid_row_mapping * matrix_rows = عدم;

    ܐܢ (act_nrows > 1) {
        ثابت size_t matrix_row_counts_size = n_as * حجم(uint32_t);
        أكد(octx->ctx->ddr_spad_size >= matrix_row_counts_size);

        hex_l2fetch_block((ثابت فراغ *) ids->data, ids->ne[1] * ids->nb[1]);

        memset(matrix_row_counts, 0, matrix_row_counts_size);
        scan_expert_ids(ids, n_ids, n_as, matrix_row_counts, عدم, 0);

        uint32_t max_count = hvx_reduce_max_i32((ثابت uint8_t *) matrix_row_counts, n_as);
        mapping_stride = max_count > 0 ? max_count : 1;

        size_t matrix_row_map_size  = n_as * mapping_stride * حجم(بنية mmid_row_mapping);
        ثابت size_t total_map_size = matrix_row_counts_size + matrix_row_map_size;

        ܐܢ (total_map_size > octx->ctx->ddr_spad_size) {
            mapping_buf = memalign(128, total_map_size);
            ܐܢ (!mapping_buf) {
                ܦܢܐ HTP_STATUS_INTERNAL_ERR;
            }
        }

        matrix_row_counts = (uint32_t *) mapping_buf;
        matrix_rows       = (بنية mmid_row_mapping *) (mapping_buf + matrix_row_counts_size);

        memset(matrix_row_counts, 0, n_as * حجم(uint32_t));
        scan_expert_ids(ids, n_ids, n_as, matrix_row_counts, matrix_rows, mapping_stride);
    }

    mmctx->matrix_row_counts    = matrix_row_counts;
    mmctx->matrix_rows          = matrix_rows;
    mmctx->mapping_stride       = mapping_stride;
    mmctx->mm_div_ne11          = kparams->div_ne1;
    mmctx->src0_row_size_padded = src0_row_size_padded;
    mmctx->act_nrows            = act_nrows;
    mmctx->cur_m_start          = 0;
    mmctx->cur_m_rows           = act_nrows;

    htp_trace_event_stop(tr, HTP_TRACE_EVT_INIT, 0);

    صحيح s;
    ܐܢ (kparams->n_hmx) {
        s = hmx_mm_op_matmul_id(octx, mmctx);
    } ܐܚܪܢܐ {
        uint32_t src0_row_start = 0;
        uint32_t src0_row_end   = src0_nrows;
        ܐܢ (octx->ctx->mdev.count > 1) {
            ثابت bool can_split = htp_tensor_can_row_partition(dst, حجم(عائم));
            ثابت بنية htp_tensor_mdev_range range = htp_tensor_mdev_partition(src0_nrows, can_split ? 32 : 0, octx->ctx->mdev.idx, octx->ctx->mdev.count, &octx->ctx->mdev.count_div);
            src0_row_start = range.start;
            src0_row_end   = range.start + range.count;
        }

        ܐܢ (src0_row_start >= src0_row_end) {
            ܐܢ (mapping_buf != octx->ctx->ddr_spad_base) {
                حرر(mapping_buf);
            }
            ܦܢܐ HTP_STATUS_OK;
        }

        ثابت uint32_t nrows = src0_row_end - src0_row_start;
        mmctx->src0_row_start = src0_row_start;
        mmctx->src0_row_end   = src0_row_end;

        mmctx->src0_nrows_per_thread = fastdiv(nrows + octx->n_threads - 1, &octx->n_threads_div);
        mmctx->src0_nrows_per_thread = hex_round_up(mmctx->src0_nrows_per_thread, 32);

        ܐܢ (hvx_mm_init_tiled_vec_dot(mmctx, src0->type) == 0) {
            s = hvx_mm_matmul_id(octx, mmctx, act_nrows > 1 ? hvx_mm_id_tiled : hvx_mv_id_tiled);
        } ܐܚܪܢܐ {
            s = HTP_STATUS_NO_SUPPORT;
        }
    }

    ܐܢ (mapping_buf != octx->ctx->ddr_spad_base) {
        حرر(mapping_buf);
    }

    ܦܢܐ s;
}

صحيح op_matmul_id_nx(بنية htp_ops_context * octx) {
    ثابت بنية htp_mm_kernel_params * kparams = (ثابت بنية htp_mm_kernel_params *) octx->kernel_params;
    بنية htp_mm_context mmctx_struct = {0};
    بنية htp_mm_context * mmctx = &mmctx_struct;

    ثابت صحيح status = htp_mm_init_context(octx, kparams);
    ܐܢ (status != HTP_STATUS_OK) {
        ܦܢܐ status;
    }

    بنية htp_thread_trace * tr = &octx->ctx->trace[0];
    htp_trace_event_start(tr, HTP_TRACE_EVT_INIT, 0);

    mmctx->octx = octx;
    ثابت uint32_t n_weights = kparams->n_weights;
    ثابت بنية htp_tensor * restrict src0 = octx->src[0];
    ثابت بنية htp_tensor * restrict act  = octx->src[n_weights];
    ثابت بنية htp_tensor * restrict ids  = octx->src[n_weights + 1];
    ܐܢ (htp_tensor_is_extended(ids) || htp_tensor_is_extended(act)) {
        ܦܢܐ HTP_STATUS_NO_SUPPORT;
    }
    لكل (uint32_t p = 0; p < n_weights; p++) {
        ܐܢ (htp_tensor_is_extended(octx->dsts[p])) {
            ܦܢܐ HTP_STATUS_NO_SUPPORT;
        }
    }

    mmctx->act = act;

    ثابت size_t src0_row_size = src0->nb[1];
    ثابت size_t src0_row_size_padded = hex_round_up(src0_row_size, 128);

    ثابت uint32_t act_nrows = act->ne[1] * act->ne[2] * act->ne[3];

    ثابت صحيح n_ids = ids->ne[0];
    ثابت صحيح n_as  = src0->ne[2];

    uint8_t  * mapping_buf       = octx->ctx->ddr_spad_base;
    uint32_t   mapping_stride    = 1;
    uint32_t * matrix_row_counts = (uint32_t *) mapping_buf;
    بنية mmid_row_mapping * matrix_rows = عدم;

    ܐܢ (act_nrows > 1) {
        ثابت size_t matrix_row_counts_size = n_as * حجم(uint32_t);
        أكد(octx->ctx->ddr_spad_size >= matrix_row_counts_size);

        hex_l2fetch_block((ثابت فراغ *) ids->data, ids->ne[1] * ids->nb[1]);

        memset(matrix_row_counts, 0, matrix_row_counts_size);
        scan_expert_ids(ids, n_ids, n_as, matrix_row_counts, عدم, 0);

        uint32_t max_count = hvx_reduce_max_i32((ثابت uint8_t *) matrix_row_counts, n_as);
        mapping_stride = max_count > 0 ? max_count : 1;

        size_t matrix_row_map_size  = n_as * mapping_stride * حجم(بنية mmid_row_mapping);
        ثابت size_t total_map_size = matrix_row_counts_size + matrix_row_map_size;

        ܐܢ (total_map_size > octx->ctx->ddr_spad_size) {
            mapping_buf = memalign(128, total_map_size);
            ܐܢ (!mapping_buf) {
                ܦܢܐ HTP_STATUS_INTERNAL_ERR;
            }
        }

        matrix_row_counts = (uint32_t *) mapping_buf;
        matrix_rows       = (بنية mmid_row_mapping *) (mapping_buf + matrix_row_counts_size);

        memset(matrix_row_counts, 0, n_as * حجم(uint32_t));
        scan_expert_ids(ids, n_ids, n_as, matrix_row_counts, matrix_rows, mapping_stride);
    }

    mmctx->matrix_row_counts    = matrix_row_counts;
    mmctx->matrix_rows          = matrix_rows;
    mmctx->mapping_stride       = mapping_stride;
    mmctx->mm_div_ne11          = kparams->div_ne1;
    mmctx->src0_row_size_padded = src0_row_size_padded;
    mmctx->act_nrows            = act_nrows;
    mmctx->cur_m_start          = 0;
    mmctx->cur_m_rows           = act_nrows;

    htp_trace_event_stop(tr, HTP_TRACE_EVT_INIT, 0);

    صحيح s;
    ܐܢ (kparams->n_hmx) {
        s = hmx_mm_op_matmul_id_nx(octx, mmctx);
    } ܐܚܪܢܐ {
        ܐܢ (hvx_mm_init_tiled_vec_dot(mmctx, src0->type) == 0) {
            s = hvx_mm_matmul_id_nx(octx, mmctx, act_nrows > 1 ? hvx_mm_id_nx_tiled : hvx_mv_id_nx_tiled);
        } ܐܚܪܢܐ {
            s = HTP_STATUS_NO_SUPPORT;
        }
    }

    ܐܢ (mapping_buf != octx->ctx->ddr_spad_base) {
        حرر(mapping_buf);
    }

    ܦܢܐ s;
}
ساكن صحيح op_matmul_nx_impl(بنية htp_ops_context * octx) {
    ثابت بنية htp_mm_kernel_params * kparams = (ثابت بنية htp_mm_kernel_params *) octx->kernel_params;

    ثابت صحيح status = htp_mm_init_context(octx, kparams);
    ܐܢ (status != HTP_STATUS_OK) {
        ܦܢܐ status;
    }

    ܐܢ (kparams->n_hmx) {
        ܦܢܐ hmx_mm_nx_2d_f32(octx, kparams);
    }

    بنية htp_thread_trace * tr = &octx->ctx->trace[0];
    htp_trace_event_start(tr, HTP_TRACE_EVT_INIT, 0);

    ثابت uint32_t n_weights = kparams->n_weights;

    ثابت بنية htp_tensor * restrict src0 = octx->src[0]; // first weight
    ثابت بنية htp_tensor * restrict act  = octx->src[n_weights]; // activation x

    بنية htp_mm_context mmctx_struct = {0};
    بنية htp_mm_context * mmctx = &mmctx_struct;
    mmctx->octx = octx;
    mmctx->act  = act;

    ثابت uint32_t act_nrows = act->ne[1] * act->ne[2] * act->ne[3];
    mmctx->act_nrows   = act_nrows;
    mmctx->cur_m_start = 0;
    mmctx->cur_m_rows  = act_nrows;

    ثابت size_t src0_row_size = src0->nb[1];

    ܐܢ (hvx_mm_init_tiled_vec_dot(mmctx, src0->type) != 0) {
        ܦܢܐ HTP_STATUS_NO_SUPPORT;
    }

    ثابت uint32_t qk = QK_Q8_0_TILED;
    ثابت uint32_t nb = (act->ne[0] + qk - 1) / qk;
    ثابت uint32_t total_nb = act_nrows * nb;

    worker_callback_t quant_task_func;
    uint32_t n_quant_tasks = 1;
    ܐܢ (act_nrows < octx->n_threads) {
        n_quant_tasks = MIN(total_nb, octx->n_threads);
        quant_task_func = htp_mm_act_quant_block_func(src0->type, act->type);
        لكل (uint32_t ith = 0; ith < n_quant_tasks; ++ith) {
            uint32_t ib_first = (total_nb * ith) / n_quant_tasks;
            uint32_t ib_last  = (total_nb * (ith + 1)) / n_quant_tasks;
            mmctx->quant_ib_first[ith] = ib_first;
            mmctx->quant_ib_last[ith]  = ib_last;
            mmctx->quant_r[ith]        = ib_first / nb;
            mmctx->quant_c[ith]        = ib_first % nb;
        }
    } ܐܚܪܢܐ {
        n_quant_tasks = MIN(act_nrows, octx->n_threads);
        quant_task_func = htp_mm_act_quant_row_func(src0->type, act->type);
    }

    ثابت size_t act_row_size = htp_mm_tiled_act_row_size(src0->type, act->ne[0]);

    بنية htp_mm_hvx_vtcm_layout L;
    htp_mm_hvx_vtcm_layout_build(&L, kparams->kernel_type, src0->type, act->ne[0], act_nrows, octx->n_threads,
                                 0, src0_row_size, act_row_size, 0, kparams->n_prefetch, false, true);

    ثابت size_t vtcm_size = L.total_bytes;

    ܐܢ (octx->ctx->vtcm_size < vtcm_size) {
        FARF(ERROR, "matmul-nx: current VTCM reservation %zu is too small, needed %zu\n",
             octx->ctx->vtcm_size, vtcm_size);
        ܦܢܐ HTP_STATUS_VTCM_TOO_SMALL;
    }

    uint8_t * ثابت base = (uint8_t *) octx->ctx->vtcm_base;
    mmctx->vtcm_src0     = VTCM_LAYOUT_PTR(uint8_t, base, L.off_src0);
    mmctx->vtcm_act      = VTCM_LAYOUT_PTR(uint8_t, base, L.off_act);
    mmctx->vtcm_dst      = VTCM_LAYOUT_PTR(uint8_t, base, L.off_dst);
    mmctx->vtcm_act_raw  = VTCM_LAYOUT_PTR(uint8_t, base, L.off_act_raw);

    mmctx->vtcm_src0_stride    = 0;
    mmctx->vtcm_act_stride     = act_row_size;
    mmctx->vtcm_act_raw_stride = hex_round_up(act->ne[0] * حجم(عائم), QK_Q8_0_TILED * حجم(عائم));

    mmctx->vtcm_src0_size_per_thread = fastdiv(L.src0_bytes, &octx->n_threads_div);
    mmctx->vtcm_act_size_per_thread  = L.act_bytes;
    mmctx->vtcm_dst_size_per_thread  = fastdiv(L.dst_bytes, &octx->n_threads_div);

    htp_trace_event_stop(tr, HTP_TRACE_EVT_INIT, 0);

    hvx_mm_transfer_act_dma(octx, kparams, act, mmctx->vtcm_act_raw, mmctx->vtcm_act_raw_stride, 0, act_nrows);

    mmctx->n_quant_rows_per_thread = (act_nrows + n_quant_tasks - 1) / n_quant_tasks;
    mmctx->n_quant_tasks = n_quant_tasks;
    work_queue_run(octx->ctx->work_queue, quant_task_func, mmctx, n_quant_tasks);

    work_queue_run(octx->ctx->work_queue, hvx_mm_nx_2d_tiled, mmctx, octx->n_threads);

    ܦܢܐ HTP_STATUS_OK;
}

صحيح op_matmul_nx(بنية htp_ops_context * octx) {
    ثابت بنية htp_mm_kernel_params * kparams = (ثابت بنية htp_mm_kernel_params *) octx->kernel_params;

    ܐܢ (kparams->collapse) {
        ثابت uint32_t n_weights = kparams->n_weights;
        ثابت بنية htp_tensor * act = octx->src[n_weights];
        ثابت uint32_t s1 = (act->ne[1] > 1) ? act->nb[1] : ((act->ne[2] > 1) ? act->nb[2] : act->nb[3]);
        بنية htp_tensor act_collapsed;
        بنية htp_tensor dsts_collapsed[HTP_OP_MAX_OUTPUTS];
        ثابت بنية htp_tensor * orig_dsts[HTP_OP_MAX_OUTPUTS];

        htp_mm_tensor_collapse_rows(&act_collapsed, act, s1);
        octx->src[n_weights] = &act_collapsed;

        لكل (uint32_t p = 0; p < n_weights; p++) {
            orig_dsts[p] = octx->dsts[p];
            ثابت بنية htp_tensor * d = octx->dsts[p];
            ثابت uint32_t sd = (d->ne[1] > 1) ? d->nb[1] : ((d->ne[2] > 1) ? d->nb[2] : d->nb[3]);
            htp_mm_tensor_collapse_rows(&dsts_collapsed[p], d, sd);
            octx->dsts[p] = &dsts_collapsed[p];
        }

        ثابت صحيح status = op_matmul_nx_impl(octx);

        octx->src[n_weights] = act;
        لكل (uint32_t p = 0; p < n_weights; p++) {
            octx->dsts[p] = orig_dsts[p];
        }
        ܦܢܐ status;
    }

    ܦܢܐ op_matmul_nx_impl(octx);
}
