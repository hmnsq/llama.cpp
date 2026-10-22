#pragma clang diagnostic ignored "-Wgnu-zero-variadic-macro-arguments"
#pragma clang diagnostic ignored "-Wunused-function"
#pragma clang diagnostic ignored "-Wunused-variable"
#pragma clang diagnostic ignored "-Wunused-but-set-variable"

#অন্তর্ভাব <HAP_farf.h>
#অন্তর্ভাব <HAP_perf.h>
#অন্তর্ভাব <HAP_compute_res.h>

#অন্তর্ভাব <math.h>
#অন্তর্ভাব <string.h>
#অন্তর্ভাব <stdatomic.h>

#অন্তর্ভাব "dma-queue.h"
#অন্তর্ভাব "hvx-utils.h"
#অন্তর্ভাব "hvx-dump.h"
#অন্তর্ভাব "hvx-arith.h"
#অন্তর্ভাব "hvx-reduce.h"

#define GGML_COMMON_DECL_C
#অন্তর্ভাব "ggml-common.h"
#অন্তর্ভাব "htp-ctx.h"
#অন্তর্ভাব "htp-ops.h"
#অন্তর্ভাব "htp-tensor.h"
#অন্তর্ভাব "matmul-ops.h"
#অন্তর্ভাব "htp-vtcm.h"

প্রকারনির্দেশ সংরচনা {
    ভিন্ন        *dst;
    dma_addr_t    bias_addr;
    size_t        bias_bytes;
    dma_addr_t    act_dma_addr;
    dma_addr_t    weight;
    dma_queue *   weight_dma;
    পূর্ণাঙ্ক           m;
    পূর্ণাঙ্ক           k;
    পূর্ণাঙ্ক           n;
    পূর্ণাঙ্ক           act_stride;
    uint32_t      act_elem_size;  // 4=F32 src1, 2=F16 src1
    পূর্ণাঙ্ক           weight_stride;
    পূর্ণাঙ্ক           weight_type;
    পূর্ণাঙ্ক           dst_stride;
    পূর্ণাঙ্ক           dst_cols;
    uint32_t      bias_stride;
    পূর্ণাঙ্ক           tile_size;
    পূর্ণাঙ্ক           aligned_tile_size;
    পূর্ণাঙ্ক           ne02;
    পূর্ণাঙ্ক           ne03;
    পূর্ণাঙ্ক           ne12;
    পূর্ণাঙ্ক           ne13;
    size_t        src0_nb2;
    size_t        src0_nb3;
    size_t        act_nb2;
    size_t        act_nb3;
    size_t        bias_nb2;
    size_t        bias_nb3;
    size_t        dst_nb2;
    size_t        dst_nb3;
    পূর্ণাঙ্ক           r2;
    পূর্ণাঙ্ক           r3;
    সংরচনা fastdiv_মানসমূহ div_r2;
    সংরচনা fastdiv_মানসমূহ div_r3;
} hmx_mm_batched_params_t;

স্থির bool htp_matmul_has_extended_weight(ধ্রুব সংরচনা htp_ops_context * octx, uint32_t n_weights) {
    প্রতি (uint32_t i = 0; i < n_weights; ++i) {
        যদি (htp_tensor_is_extended(octx->src[i])) {
            প্রত্যাগম true;
        }
    }
    প্রত্যাগম false;
}

সংরচনা htp_mm_context {
    ধ্রুব বর্ণ * type;
    সংরচনা htp_ops_context * octx;
    ধ্রুব সংরচনা htp_tensor * act;

    শূন্য (*vec_dot_1x1)(ধ্রুব uint32_t n, ভিন্ন * restrict s0,
         ধ্রুব শূন্য * restrict vx0,
         ধ্রুব শূন্য * restrict vy0);

    শূন্য (*vec_dot_2x1)(ধ্রুব uint32_t n, ভিন্ন * restrict s0,
         ধ্রুব শূন্য * restrict vx0, ধ্রুব শূন্য * restrict vx1,
         ধ্রুব শূন্য * restrict vy0);

    শূন্য (*vec_dot_2x2)(ধ্রুব uint32_t n, ভিন্ন * restrict s0, ভিন্ন * restrict s1,
         ধ্রুব শূন্য * restrict vx0, ধ্রুব শূন্য * restrict vx1,
         ধ্রুব শূন্য * restrict vy0, ধ্রুব শূন্য * restrict vy1);

    শূন্য (*vec_dot_32x1)(ধ্রুব uint32_t n, ভিন্ন * restrict s,
         ধ্রুব শূন্য * restrict vx,
         ধ্রুব শূন্য * restrict vy, uint32_t valid_rows,
         ধ্রুব ভিন্ন * restrict sz);

    শূন্য (*vec_dot_32x2)(ধ্রুব uint32_t n, ভিন্ন * restrict s0, ভিন্ন * restrict s1,
         ধ্রুব শূন্য * restrict vx,
         ধ্রুব শূন্য * restrict vy0, ধ্রুব শূন্য * restrict vy1, uint32_t valid_rows,
         ধ্রুব ভিন্ন * restrict sz0, ধ্রুব ভিন্ন * restrict sz1);

    // Precomputed values
    uint32_t src0_nrows_per_thread;
    uint32_t src0_row_start;
    uint32_t src0_row_end;
    uint32_t src0_row_size_padded;
    uint32_t act_nrows;
    uint32_t cur_m_start;
    uint32_t cur_m_rows;

    সংরচনা fastdiv_মানসমূহ mm_div_ne12_ne1;
    সংরচনা fastdiv_মানসমূহ mm_div_ne1;
    সংরচনা fastdiv_মানসমূহ mm_div_r2;
    সংরচনা fastdiv_মানসমূহ mm_div_r3;
    সংরচনা fastdiv_মানসমূহ mm_div_ne11;

    // Precomputed block-parallel quantization values
    uint32_t          quant_ib_first[WORK_QUEUE_MAX_N_THREADS];
    uint32_t          quant_ib_last[WORK_QUEUE_MAX_N_THREADS];
    uint32_t          quant_r[WORK_QUEUE_MAX_N_THREADS];
    uint32_t          quant_c[WORK_QUEUE_MAX_N_THREADS];
    uint32_t          n_quant_tasks;
    uint32_t          n_quant_rows_per_thread;

    // Fields for scattered mapping & HMX support in MUL_MAT_ID
    ধ্রুব uint32_t * matrix_row_counts;
    ধ্রুব সংরচনা mmid_row_mapping * matrix_rows;
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

স্থির পূর্ণাঙ্ক htp_mm_init_context(
    সংরচনা htp_ops_context * octx,
    ধ্রুব সংরচনা htp_mm_kernel_params * kparams
) {
    যদি (!htp_ops_context_set_n_threads(octx, (uint32_t) kparams->n_threads)) {
        প্রত্যাগম HTP_STATUS_INVAL_PARAMS;
    }

    যদি (kparams->n_hmx) {
        যদি (kparams->n_act_threads <= 0 || kparams->n_act_threads > (int32_t) octx->n_threads) {
            প্রত্যাগম HTP_STATUS_INVAL_PARAMS;
        }
    }

    প্রত্যাগম HTP_STATUS_OK;
}

// vdelta control to expand first 32 e8m0 values into 32 uint32 elements
স্থির ধ্রুব uint8_t __attribute__((aligned(128))) expand_x32_e8m0[128] = {
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
স্থির ধ্রুব uint8_t __attribute__((aligned(VLEN))) kvalues_iq4nl_lut[] = {
    0x81, 0, 0x98, 0, 0xAD, 0, 0xBF, 0, 0xCF, 0, 0xDD, 0, 0xEA, 0, 0xF6, 0, 0x01, 0, 0x0D, 0, 0x19, 0, 0x26, 0,
    0x35, 0, 0x45, 0, 0x59, 0, 0x71, 0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0,
    0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0,
    0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0,
    0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0,
};

স্থির ধ্রুব uint8_t __attribute__((aligned(VLEN))) kvalues_mxfp4_lut[] = {
    0,    0, 1,    0, 2,    0, 3, 0, 4, 0, 6, 0, 8, 0, 12, 0, 0, 0, 0xff, 0, 0xfe, 0, 0xfd, 0, 0xfc, 0,
    0xfa, 0, 0xf8, 0, 0xf4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 0,    0, 0,    0, 0,    0, 0,    0,
    0,    0, 0,    0, 0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 0,    0, 0,    0, 0,    0, 0,    0,
    0,    0, 0,    0, 0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 0,    0, 0,    0, 0,    0, 0,    0,
    0,    0, 0,    0, 0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 0,    0, 0,    0, 0,    0,
};

#define htp_matmul_tensors_preamble                                 \
    ধ্রুব সংরচনা htp_tensor * restrict src0 = octx->src[0];         \
    ধ্রুব সংরচনা htp_tensor * restrict src1 = octx->src[1];         \
    ধ্রুব সংরচনা htp_tensor * restrict src2 = octx->src[2];         \
    ধ্রুব সংরচনা htp_tensor * restrict  dst = octx->dst;            \
    ধ্রুব সংরচনা htp_tensor * restrict  act = src1;                 \
                                                                    \
    ধ্রুব uint32_t ne00 = src0->ne[0];                              \
    ধ্রুব uint32_t ne01 = src0->ne[1];                              \
    ধ্রুব uint32_t ne02 = src0->ne[2];                              \
    ধ্রুব uint32_t ne03 = src0->ne[3];                              \
                                                                    \
    ধ্রুব uint32_t ne10 = src1->ne[0];                              \
    ধ্রুব uint32_t ne11 = src1->ne[1];                              \
    ধ্রুব uint32_t ne12 = src1->ne[2];                              \
    ধ্রুব uint32_t ne13 = src1->ne[3];                              \
                                                                    \
    ধ্রুব uint32_t ne20 = src2 ? src2->ne[0] : 0;                   \
    ধ্রুব uint32_t ne21 = src2 ? src2->ne[1] : 0;                   \
    ধ্রুব uint32_t ne22 = src2 ? src2->ne[2] : 0;                   \
    ধ্রুব uint32_t ne23 = src2 ? src2->ne[3] : 0;                   \
                                                                    \
    ধ্রুব uint32_t ne0 = dst->ne[0];                                \
    ধ্রুব uint32_t ne1 = dst->ne[1];                                \
    ধ্রুব uint32_t ne2 = dst->ne[2];                                \
    ধ্রুব uint32_t ne3 = dst->ne[3];                                \
                                                                    \
    ধ্রুব uint32_t nb00 = src0->nb[0];                              \
    ধ্রুব uint32_t nb01 = src0->nb[1];                              \
    ধ্রুব uint32_t nb02 = src0->nb[2];                              \
    ধ্রুব uint32_t nb03 = src0->nb[3];                              \
                                                                    \
    ধ্রুব uint32_t nb10 = src1->nb[0];                              \
    ধ্রুব uint32_t nb11 = src1->nb[1];                              \
    ধ্রুব uint32_t nb12 = src1->nb[2];                              \
    ধ্রুব uint32_t nb13 = src1->nb[3];                              \
                                                                    \
    ধ্রুব uint32_t nb0 = dst->nb[0];                                \
    ধ্রুব uint32_t nb1 = dst->nb[1];                                \
    ধ্রুব uint32_t nb2 = dst->nb[2];                                \
    ধ্রুব uint32_t nb3 = dst->nb[3];

#define htp_matmul_preamble                                         \
    সংরচনা htp_mm_context * mmctx  = data;                          \
    সংরচনা htp_ops_context * octx  = mmctx->octx;                   \
    dma_queue * dma_q              = octx->ctx->dma[ith];           \
    uint32_t src0_nrows_per_thread = mmctx->src0_nrows_per_thread;  \
    htp_matmul_tensors_preamble;




// hvx kernels first: the HMX Q6_K dequantizer reuses unpack_q6_k_group from there
#অন্তর্ভাব "hvx-mm-kernels-tiled.h"
#অন্তর্ভাব "hvx-mm-kernels-float.h"
#অন্তর্ভাব "hmx-mm-kernels-tiled.h"

স্থির শূন্য hvx_mm_2d_tiled(unsigned পূর্ণাঙ্ক nth, unsigned পূর্ণাঙ্ক ith, শূন্য * data) {
    htp_matmul_preamble;

    ধ্রুব uint32_t src0_nrows = mmctx->src0_row_end - mmctx->src0_row_start;
    ধ্রুব uint32_t act_nrows = mmctx->cur_m_rows ? mmctx->cur_m_rows : (ne11 * ne12 * ne13);
    ধ্রুব uint32_t cur_m_start = mmctx->cur_m_start;

    ধ্রুব uint32_t src0_start_row  = mmctx->src0_row_start + src0_nrows_per_thread * ith;
    ধ্রুব uint32_t src0_end_row    = MIN(src0_start_row + src0_nrows_per_thread, mmctx->src0_row_end);

    সংরচনা htp_thread_trace * tr = &octx->ctx->trace[ith];

    ধ্রুব সংরচনা htp_mm_kernel_params * kparams = (ধ্রুব সংরচনা htp_mm_kernel_params *) octx->kernel_params;
    ধ্রুব uint32_t n_prefetch = kparams->n_prefetch;
    প্রতিজ্ঞা(n_prefetch >= 2 && n_prefetch <= HTP_MM_MAX_PREFETCH && (n_prefetch & (n_prefetch - 1)) == 0);

    ধ্রুব size_t dst_row_size  = nb1;
    ধ্রুব size_t act_row_size = nb11;
    ধ্রুব size_t act_stride = mmctx->vtcm_act_stride;
    ধ্রুব size_t src2_stride = src2 ? ((src2->ne[1] == 1) ? 0 : src2->nb[1]) : 0;

    uint8_t * restrict vtcm_dst_ptr  = mmctx->vtcm_dst  + mmctx->vtcm_dst_size_per_thread  * ith;
    uint8_t * restrict vtcm_src0_ptr = mmctx->vtcm_src0 + mmctx->vtcm_src0_size_per_thread * ith;
    uint8_t * restrict act_data = mmctx->vtcm_act;

    ধ্রুব dma_addr_t src0_row = src0->data;

    ধ্রুব uint32_t tile_size = htp_tensor_get_tile_size(src0->type);
    ধ্রুব uint32_t aligned_tile_size = htp_tensor_get_tile_padded_size(src0->type);

    uint32_t n_k_tiles_w = ne00 / htp_tensor_get_tile_k(src0->type);
    uint32_t tile_row_stride = n_k_tiles_w * tile_size;
    uint32_t tile_row_transfer_size_aligned = n_k_tiles_w * aligned_tile_size;

    uint32_t ct_start = src0_start_row / 32;
    uint32_t ct_end   = (src0_end_row + 31) / 32;

    uint32_t ঠেলো_ct = ct_start;
    যদি (src0_start_row < src0_end_row) {
        প্রতি (uint32_t d = 0; d < n_prefetch && ঠেলো_ct < ct_end; d++, ঠেলো_ct++) {
            dma_queue_ঠেলো(dma_q, dma_make_data(vtcm_src0_ptr + d * tile_row_transfer_size_aligned,
                           src0_row + ঠেলো_ct * tile_row_stride), aligned_tile_size, tile_size, tile_size, n_k_tiles_w);
        }
    }

    যদি (src0_start_row >= src0_end_row) {
        প্রত্যাগম;
    }

    প্রতি (uint32_t ct = ct_start; ct < ct_end; ct++) {
        ধ্রুব uint8_t * w_tile = (শূন্য *) dma_queue_বের_করো(dma_q).dst;

        পূর্ণাঙ্ক valid_rows = (পূর্ণাঙ্ক)ne0 - (পূর্ণাঙ্ক)(ct * 32);
        valid_rows = MIN(32, MAX(0, valid_rows));

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, ct);
        uint32_t ir1 = 0;
        প্রতি (; ir1 + 1 < act_nrows; ir1 += 2) {
            ধ্রুব uint8_t * restrict act_col0 = (ধ্রুব uint8_t *) (act_data + (ir1+0) * act_stride);
            ধ্রুব uint8_t * restrict act_col1 = (ধ্রুব uint8_t *) (act_data + (ir1+1) * act_stride);
            ভিন্ন * restrict dst_row0 = (ভিন্ন *) (dst->data + ((cur_m_start + ir1+0) * dst_row_size));
            ভিন্ন * restrict dst_row1 = (ভিন্ন *) (dst->data + ((cur_m_start + ir1+1) * dst_row_size));

            ভিন্ন * dst_ptr0 = &dst_row0[ct * 32];
            ভিন্ন * dst_ptr1 = &dst_row1[ct * 32];

            ধ্রুব ভিন্ন * src2_ptr0 = রিক্ত;
            ধ্রুব ভিন্ন * src2_ptr1 = রিক্ত;
            যদি (src2) {
                ধ্রুব ভিন্ন * restrict src2_row0 = (ধ্রুব ভিন্ন *) ((ধ্রুব uint8_t *) src2->data + ((cur_m_start + ir1+0) * src2_stride));
                ধ্রুব ভিন্ন * restrict src2_row1 = (ধ্রুব ভিন্ন *) ((ধ্রুব uint8_t *) src2->data + ((cur_m_start + ir1+1) * src2_stride));
                src2_ptr0 = &src2_row0[ct * 32];
                src2_ptr1 = &src2_row1[ct * 32];
            }
            mmctx->vec_dot_32x2(ne10, dst_ptr0, dst_ptr1, w_tile, act_col0, act_col1, valid_rows, src2_ptr0, src2_ptr1);
        }

        প্রতি (; ir1 < act_nrows; ++ir1) {
            ধ্রুব uint8_t * restrict act_col = (ধ্রুব uint8_t *) (act_data + ir1 * act_stride);
            ভিন্ন * restrict dst_row          = (ভিন্ন *) (dst->data + ((cur_m_start + ir1) * dst_row_size));
            ভিন্ন * dst_ptr = &dst_row[ct * 32];

            ধ্রুব ভিন্ন * src2_ptr = রিক্ত;
            যদি (src2) {
                ধ্রুব ভিন্ন * restrict src2_row = (ধ্রুব ভিন্ন *) ((ধ্রুব uint8_t *) src2->data + ((cur_m_start + ir1) * src2_stride));
                src2_ptr = &src2_row[ct * 32];
            }
            mmctx->vec_dot_32x1(ne10, dst_ptr, w_tile, act_col, valid_rows, src2_ptr);
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ct);

        যদি (ঠেলো_ct < ct_end) {
            dma_queue_ঠেলো(dma_q, dma_make_data(w_tile, src0_row + ঠেলো_ct * tile_row_stride),
                           aligned_tile_size, tile_size, tile_size, n_k_tiles_w);
            ঠেলো_ct++;
        }
    }
}

স্থির শূন্য hvx_mv_2d_tiled(unsigned পূর্ণাঙ্ক nth, unsigned পূর্ণাঙ্ক ith, শূন্য * data) {
    htp_matmul_preamble;

    ধ্রুব uint32_t src0_nrows = mmctx->src0_row_end - mmctx->src0_row_start;

    ধ্রুব uint32_t src0_start_row  = mmctx->src0_row_start + src0_nrows_per_thread * ith;
    ধ্রুব uint32_t src0_end_row    = MIN(src0_start_row + src0_nrows_per_thread, mmctx->src0_row_end);

    সংরচনা htp_thread_trace * tr = &octx->ctx->trace[ith];

    ধ্রুব সংরচনা htp_mm_kernel_params * kparams = (ধ্রুব সংরচনা htp_mm_kernel_params *) octx->kernel_params;
    ধ্রুব uint32_t n_prefetch = kparams->n_prefetch;
    প্রতিজ্ঞা(n_prefetch >= 2 && n_prefetch <= HTP_MM_MAX_PREFETCH && (n_prefetch & (n_prefetch - 1)) == 0);

    ধ্রুব size_t dst_row_size  = nb1;
    ধ্রুব size_t act_row_size = nb11;
    ধ্রুব size_t act_stride = mmctx->vtcm_act_stride;

    uint8_t * vtcm_dst_ptr  = mmctx->vtcm_dst + mmctx->vtcm_dst_size_per_thread * ith;
    uint8_t * vtcm_src0_ptr = mmctx->vtcm_src0 + mmctx->vtcm_src0_size_per_thread * ith;
    uint8_t * act_data = mmctx->vtcm_act;

    ভিন্ন * tmp = (ভিন্ন *) vtcm_dst_ptr;

    ধ্রুব dma_addr_t src0_row = src0->data;

    ধ্রুব uint8_t * restrict act_col = (ধ্রুব uint8_t *) act_data;
    ভিন্ন * restrict dst_col          = (ভিন্ন *) dst->data;

    ধ্রুব uint32_t tile_size = htp_tensor_get_tile_size(src0->type);
    ধ্রুব uint32_t aligned_tile_size = htp_tensor_get_tile_padded_size(src0->type);

    uint32_t n_k_tiles_w = ne00 / htp_tensor_get_tile_k(src0->type);
    uint32_t tile_row_stride = n_k_tiles_w * tile_size;
    uint32_t tile_row_transfer_size_aligned = n_k_tiles_w * aligned_tile_size;

    uint32_t ct_start = src0_start_row / 32;
    uint32_t ct_end   = (src0_end_row + 31) / 32;

    uint32_t ঠেলো_ct = ct_start;
    যদি (src0_start_row < src0_end_row) {
        যদি (src2) {
            ভিন্ন * vtcm_bias_ptr = (ভিন্ন *) mmctx->vtcm_bias + src0_start_row;
            ধ্রুব dma_addr_t src2_addr = src2->data + src0_start_row * পরিমাণ(ভিন্ন);
            পূর্ণাঙ্ক slice_size = (পূর্ণাঙ্ক)MIN(src0_end_row, ne0) - (পূর্ণাঙ্ক)src0_start_row;
            যদি (slice_size > 0) {
                dma_queue_ঠেলো(dma_q, dma_make_data(vtcm_bias_ptr, src2_addr),
                               slice_size * পরিমাণ(ভিন্ন), slice_size * পরিমাণ(ভিন্ন), slice_size * পরিমাণ(ভিন্ন), 1);
                dma_queue_বের_করো_nowait(dma_q);
            }
        }
        প্রতি (uint32_t d = 0; d < n_prefetch && ঠেলো_ct < ct_end; d++, ঠেলো_ct++) {
            dma_queue_ঠেলো(dma_q, dma_make_data(vtcm_src0_ptr + d * tile_row_transfer_size_aligned,
                           src0_row + ঠেলো_ct * tile_row_stride), aligned_tile_size, tile_size, tile_size, n_k_tiles_w);
        }
    }

    যদি (src0_start_row >= src0_end_row) {
        প্রত্যাগম;
    }

    প্রতি (uint32_t ct = ct_start; ct < ct_end; ct++) {
        ধ্রুব uint8_t * w_tile = (শূন্য *) dma_queue_বের_করো(dma_q).dst;

        ভিন্ন * dst_ptr = &tmp[ct * 32 - src0_start_row];
        পূর্ণাঙ্ক valid_rows = (পূর্ণাঙ্ক)ne0 - (পূর্ণাঙ্ক)(ct * 32);
        valid_rows = MIN(32, MAX(0, valid_rows));

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, ct);
        mmctx->vec_dot_32x1(ne10, dst_ptr, w_tile, act_col, valid_rows, রিক্ত);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ct);

        যদি (ঠেলো_ct < ct_end) {
            dma_queue_ঠেলো(dma_q, dma_make_data(w_tile, src0_row + ঠেলো_ct * tile_row_stride),
                           aligned_tile_size, tile_size, tile_size, n_k_tiles_w);
            ঠেলো_ct++;
        }
    }

    পূর্ণাঙ্ক copy_cnt = (পূর্ণাঙ্ক)MIN(src0_end_row, ne0) - (পূর্ণাঙ্ক)src0_start_row;
    যদি (copy_cnt > 0) {
        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, ct_end);
        যদি (src2) {
            hvx_add_f32_uuu((uint8_t *) &dst_col[src0_start_row],
                            (ধ্রুব uint8_t *) tmp,
                            (ধ্রুব uint8_t *) ((ধ্রুব ভিন্ন *) mmctx->vtcm_bias + src0_start_row),
                            copy_cnt);
        } অন্যথা {
            hvx_copy_f32_ua((uint8_t *) &dst_col[src0_start_row], (uint8_t *) tmp, copy_cnt);
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ct_end);
    }
}

স্থির শূন্য hvx_mm_nx_2d_tiled(unsigned পূর্ণাঙ্ক nth, unsigned পূর্ণাঙ্ক ith, শূন্য * data) {
    সংরচনা htp_mm_context * mmctx = data;
    সংরচনা htp_ops_context * octx = mmctx->octx;
    ধ্রুব সংরচনা htp_mm_kernel_params * kparams = (ধ্রুব সংরচনা htp_mm_kernel_params *) octx->kernel_params;
    ধ্রুব uint32_t n_weights = kparams->n_weights;

    ধ্রুব সংরচনা htp_tensor * restrict act = octx->src[n_weights]; /* x */
    ধ্রুব uint32_t ne10 = act->ne[0];
    ধ্রুব uint32_t act_nrows = act->ne[1] * act->ne[2] * act->ne[3];
    ধ্রুব size_t act_stride = mmctx->vtcm_act_stride;

    uint8_t * restrict vtcm_weight_ptr = mmctx->vtcm_src0 + mmctx->vtcm_src0_size_per_thread * ith;
    uint8_t * restrict act_data        = mmctx->vtcm_act;

    সংরচনা htp_thread_trace * tr = &octx->ctx->trace[ith];
    ধ্রুব uint32_t n_prefetch = kparams->n_prefetch;
    প্রতিজ্ঞা(n_prefetch >= 2 && n_prefetch <= HTP_MM_MAX_PREFETCH && (n_prefetch & (n_prefetch - 1)) == 0);

    প্রতি (uint32_t widx = 0; widx < n_weights; widx++) {
        ধ্রুব সংরচনা htp_tensor * restrict src_w = octx->src[widx];
        ধ্রুব সংরচনা htp_tensor * restrict dst   = octx->dsts[widx];
        যদি (!src_w || !dst) অনুবর্ত;
        dma_queue * dma_q = octx->ctx->dma[ith];

        ধ্রুব uint32_t tile_size = htp_tensor_get_tile_size(src_w->type);
        ধ্রুব uint32_t aligned_tile_size = htp_tensor_get_tile_padded_size(src_w->type);
        ধ্রুব uint32_t ne00 = src_w->ne[0];
        ধ্রুব uint32_t ne01 = src_w->ne[1];
        ধ্রুব size_t dst_row_size = dst->nb[1];
        ধ্রুব dma_addr_t src_w_row = src_w->data;

        ধ্রুব uint32_t n_k_tiles_w = ne00 / htp_tensor_get_tile_k(src_w->type);
        ধ্রুব uint32_t tile_row_stride = n_k_tiles_w * tile_size;
        ধ্রুব uint32_t tile_row_transfer_size_aligned = n_k_tiles_w * aligned_tile_size;

        uint32_t src0_start_row = 0;
        uint32_t src0_end_row   = ne01;
        যদি (octx->ctx->mdev.গুনতি > 1) {
            ধ্রুব bool can_split = htp_tensor_can_row_partition(dst, পরিমাণ(ভিন্ন));
            ধ্রুব সংরচনা htp_tensor_mdev_range range = htp_tensor_mdev_partition(ne01, can_split ? 32 : 0,
                                                         octx->ctx->mdev.idx, octx->ctx->mdev.গুনতি, &octx->ctx->mdev.গুনতি_div);
            src0_start_row = range.start;
            src0_end_row   = range.start + range.গুনতি;
        }

        ধ্রুব uint32_t nrows = src0_end_row - src0_start_row;
        uint32_t src0_nrows_per_thread = fastdiv(nrows + nth - 1, &octx->n_threads_div);
        src0_nrows_per_thread = hex_round_up(src0_nrows_per_thread, 32);

        ধ্রুব uint32_t start_row = src0_start_row + src0_nrows_per_thread * ith;
        ধ্রুব uint32_t end_row   = MIN(start_row + src0_nrows_per_thread, src0_end_row);
        যদি (start_row >= end_row) অনুবর্ত;

        uint32_t ct_start = start_row / 32;
        uint32_t ct_end   = (end_row + 31) / 32;

        uint32_t ঠেলো_ct = ct_start;
        প্রতি (uint32_t d = 0; d < n_prefetch && ঠেলো_ct < ct_end; d++, ঠেলো_ct++) {
            dma_queue_ঠেলো(dma_q, dma_make_data(vtcm_weight_ptr + d * tile_row_transfer_size_aligned,
                            src_w_row + ঠেলো_ct * tile_row_stride), aligned_tile_size, tile_size, tile_size, n_k_tiles_w);
        }

        প্রতি (uint32_t ct = ct_start; ct < ct_end; ct++) {
            ধ্রুব uint8_t * w_tile = (শূন্য *) dma_queue_বের_করো(dma_q).dst;
            পূর্ণাঙ্ক valid_rows = (পূর্ণাঙ্ক)dst->ne[0] - (পূর্ণাঙ্ক)(ct * 32);
            valid_rows = MIN(32, MAX(0, valid_rows));

            htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, ct);
            uint32_t ir1 = 0;
            প্রতি (; ir1 + 1 < act_nrows; ir1 += 2) {
                ধ্রুব uint8_t * restrict act_col0 = (ধ্রুব uint8_t *) (act_data + (ir1+0) * act_stride);
                ধ্রুব uint8_t * restrict act_col1 = (ধ্রুব uint8_t *) (act_data + (ir1+1) * act_stride);

                ভিন্ন * restrict dst_row0 = (ভিন্ন *) (dst->data + ((ir1+0) * dst_row_size));
                ভিন্ন * restrict dst_row1 = (ভিন্ন *) (dst->data + ((ir1+1) * dst_row_size));
                ভিন্ন * dst_ptr0 = &dst_row0[ct * 32];
                ভিন্ন * dst_ptr1 = &dst_row1[ct * 32];

                mmctx->vec_dot_32x2(ne10, dst_ptr0, dst_ptr1, w_tile, act_col0, act_col1, valid_rows, রিক্ত, রিক্ত);
            }

            প্রতি (; ir1 < act_nrows; ++ir1) {
                ধ্রুব uint8_t * restrict act_col = (ধ্রুব uint8_t *) (act_data + ir1 * act_stride);
                ভিন্ন * restrict dst_row = (ভিন্ন *) (dst->data + (ir1 * dst_row_size));
                ভিন্ন * dst_ptr = &dst_row[ct * 32];
                mmctx->vec_dot_32x1(ne10, dst_ptr, w_tile, act_col, valid_rows, রিক্ত);
            }
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ct);

            যদি (ঠেলো_ct < ct_end) {
                dma_queue_ঠেলো(dma_q, dma_make_data(w_tile, src_w_row + ঠেলো_ct * tile_row_stride),
                                aligned_tile_size, tile_size, tile_size, n_k_tiles_w);
                ঠেলো_ct++;
            }
        }
    }
}

স্থির শূন্য hvx_mm_transfer_act_dma(
    সংরচনা htp_ops_context * octx,
    ধ্রুব সংরচনা htp_mm_kernel_params * kparams,
    ধ্রুব সংরচনা htp_tensor * act,
    uint8_t * dst_base,
    size_t dst_row_size,
    uint32_t m_start,
    uint32_t m_rows
) {
    যদি (m_rows == 0) {
        প্রত্যাগম;
    }

    dma_queue * dma_q = octx->ctx->dma[0];
    ধ্রুব uint32_t ne0 = act->ne[0];
    ধ্রুব size_t elem_size = (act->type == HTP_TYPE_F16) ? পরিমাণ(__fp16) : পরিমাণ(ভিন্ন);
    ধ্রুব size_t row_bytes = ne0 * elem_size;
    ধ্রুব size_t act_nb1 = act->nb[1];
    ধ্রুব dma_addr_t act_base = act->data;

    ধ্রুব bool is_contiguous = (act->nb[2] == act->ne[1] * act_nb1) &&
                               (act->nb[3] == act->ne[2] * act->nb[2]);

    যদি (is_contiguous) {
        ধ্রুব dma_addr_t src_addr = act_base + m_start * act_nb1;
        dma_queue_ঠেলো(dma_q, dma_make_data(dst_base, src_addr),
                       dst_row_size, act_nb1, row_bytes, m_rows);
        dma_queue_বের_করো(dma_q);
    } অন্যথা {
        ধ্রুব uint32_t ne12_ne1 = act->ne[2] * act->ne[1];
        ধ্রুব bool use_fastdiv = kparams->div_ne12_ne1.mp != 0;
        প্রতি (uint32_t ir = 0; ir < m_rows; ++ir) {
            ধ্রুব uint32_t ir1 = m_start + ir;
            uint32_t i11, i12, i13;
            যদি (use_fastdiv) {
                i13 = fastdiv(ir1, &kparams->div_ne12_ne1);
                ধ্রুব uint32_t rem = ir1 - i13 * ne12_ne1;
                i12 = fastdiv(rem, &kparams->div_ne1);
                i11 = rem - i12 * act->ne[1];
            } অন্যথা {
                i13 = ne12_ne1 ? ir1 / ne12_ne1 : 0;
                ধ্রুব uint32_t rem = ir1 - i13 * ne12_ne1;
                i12 = act->ne[1] ? rem / act->ne[1] : 0;
                i11 = rem - i12 * act->ne[1];
            }
            ধ্রুব dma_addr_t row_src = act_base + (i11 * act->nb[1] +
                                                   i12 * act->nb[2] +
                                                   i13 * act->nb[3]);
            uint8_t * row_dst = dst_base + ir * dst_row_size;
            dma_queue_ঠেলো(dma_q, dma_make_data(row_dst, row_src),
                           dst_row_size, act_nb1, row_bytes, 1);
            dma_queue_বের_করো(dma_q);
        }
    }

    যদি (dst_row_size > row_bytes) {
        সংরচনা htp_thread_trace * tr = &octx->ctx->trace[0];
        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_A_PREP, (uint16_t) m_start);
        ধ্রুব uint32_t pad_elems = (dst_row_size - row_bytes) / elem_size;
        যদি (elem_size == পরিমাণ(ভিন্ন)) {
            প্রতি (uint32_t ir = 0; ir < m_rows; ++ir) {
                hvx_splat_f32_u(dst_base + ir * dst_row_size + row_bytes, 0.0f, pad_elems);
            }
        } অন্যথা {
            প্রতি (uint32_t ir = 0; ir < m_rows; ++ir) {
                hvx_splat_f16_u(dst_base + ir * dst_row_size + row_bytes, (_Float16) 0.0f, pad_elems);
            }
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_A_PREP, (uint16_t) m_start);
    }
}

#define QUANTIZE_IMPL(name, log_name, kernel_fn, dst_row_size_রাশিমালা)                                        \
স্থির শূন্য name(unsigned পূর্ণাঙ্ক nth, unsigned পূর্ণাঙ্ক ith, শূন্য * data) {                                        \
    (শূন্য) nth;                                                                                            \
    সংরচনা htp_mm_context * mmctx = data;                                                                  \
    সংরচনা htp_ops_context * octx = mmctx->octx;                                                           \
    ধ্রুব সংরচনা htp_tensor * src = mmctx->act;                                                            \
    ধ্রুব uint32_t ne0 = src->ne[0];                                                                       \
    ধ্রুব uint32_t nrows = mmctx->cur_m_rows ? mmctx->cur_m_rows : mmctx->act_nrows;                       \
    ধ্রুব uint32_t nrows_per_thread = mmctx->n_quant_rows_per_thread;                                      \
                                                                                                           \
    ধ্রুব uint32_t ir_first = nrows_per_thread * ith;                                                      \
    যদি (ir_first >= nrows) {                                                                               \
        প্রত্যাগম;                                                                                            \
    }                                                                                                      \
                                                                                                           \
    সংরচনা htp_thread_trace * tr = &octx->ctx->trace[ith];                                                 \
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_A_QUANT, ir_first);                                        \
                                                                                                           \
    uint8_t * restrict dst = mmctx->vtcm_act;                                                              \
    ধ্রুব uint32_t ir_last = MIN(ir_first + nrows_per_thread, nrows);                                      \
    ধ্রুব size_t raw_row_size = mmctx->vtcm_act_raw_stride;                                                \
    ধ্রুব size_t dst_row_size = (dst_row_size_রাশিমালা);                                                       \
                                                                                                           \
    ধ্রুব uint8_t * restrict src_data = (ধ্রুব uint8_t *) mmctx->vtcm_act_raw + (raw_row_size * ir_first); \
    uint8_t * restrict dst_data = (uint8_t *) dst + (dst_row_size * ir_first);                             \
    kernel_fn(src_data, dst_data, রিক্ত, ne0, ir_last - ir_first, raw_row_size, dst_row_size);              \
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

স্থির শূন্য quantize_f32_q8_0_tiled_block(unsigned পূর্ণাঙ্ক nth, unsigned পূর্ণাঙ্ক ith, শূন্য * data) {
    (শূন্য) nth;
    সংরচনা htp_mm_context * mmctx = data;
    যদি (mmctx->quant_ib_first[ith] >= mmctx->quant_ib_last[ith]) {
        প্রত্যাগম;
    }
    সংরচনা htp_ops_context * octx = mmctx->octx;
    সংরচনা htp_thread_trace * tr = &octx->ctx->trace[ith];
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_A_QUANT, mmctx->quant_ib_first[ith]);

    ধ্রুব সংরচনা htp_tensor * src = mmctx->act;

    quantize_f32_q8_0_tiled_block_kernel(
        (ধ্রুব ভিন্ন *) mmctx->vtcm_act_raw,
        mmctx->vtcm_act,
        রিক্ত,
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

স্থির শূন্য quantize_f32_q8_1_tiled_block(unsigned পূর্ণাঙ্ক nth, unsigned পূর্ণাঙ্ক ith, শূন্য * data) {
    (শূন্য) nth;
    সংরচনা htp_mm_context * mmctx = data;
    যদি (mmctx->quant_ib_first[ith] >= mmctx->quant_ib_last[ith]) {
        প্রত্যাগম;
    }
    সংরচনা htp_ops_context * octx = mmctx->octx;
    সংরচনা htp_thread_trace * tr = &octx->ctx->trace[ith];
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_A_QUANT, mmctx->quant_ib_first[ith]);

    ধ্রুব সংরচনা htp_tensor * src = mmctx->act;

    quantize_f32_q8_1_tiled_block_kernel(
        (ধ্রুব ভিন্ন *) mmctx->vtcm_act_raw,
        mmctx->vtcm_act,
        রিক্ত,
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

স্থির শূন্য quantize_f32_q8_1_s16_tiled_block(unsigned পূর্ণাঙ্ক nth, unsigned পূর্ণাঙ্ক ith, শূন্য * data) {
    (শূন্য) nth;
    সংরচনা htp_mm_context * mmctx = data;
    যদি (mmctx->quant_ib_first[ith] >= mmctx->quant_ib_last[ith]) {
        প্রত্যাগম;
    }
    সংরচনা htp_ops_context * octx = mmctx->octx;
    সংরচনা htp_thread_trace * tr = &octx->ctx->trace[ith];
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_A_QUANT, mmctx->quant_ib_first[ith]);

    ধ্রুব সংরচনা htp_tensor * src = mmctx->act;

    quantize_f32_q8_1_s16_tiled_block_kernel(
        (ধ্রুব ভিন্ন *) mmctx->vtcm_act_raw,
        mmctx->vtcm_act,
        রিক্ত,
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

স্থির শূন্য quantize_f32_f16_tiled_block(unsigned পূর্ণাঙ্ক nth, unsigned পূর্ণাঙ্ক ith, শূন্য * data) {
    (শূন্য) nth;
    সংরচনা htp_mm_context * mmctx = data;
    যদি (mmctx->quant_ib_first[ith] >= mmctx->quant_ib_last[ith]) {
        প্রত্যাগম;
    }
    সংরচনা htp_ops_context * octx = mmctx->octx;
    সংরচনা htp_thread_trace * tr = &octx->ctx->trace[ith];
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_A_QUANT, mmctx->quant_ib_first[ith]);

    ধ্রুব সংরচনা htp_tensor * src = mmctx->act;

    quantize_f32_f16_tiled_block_kernel(
        (ধ্রুব ভিন্ন *) mmctx->vtcm_act_raw,
        mmctx->vtcm_act,
        রিক্ত,
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

স্থির শূন্য quantize_f16_f16_tiled_block(unsigned পূর্ণাঙ্ক nth, unsigned পূর্ণাঙ্ক ith, শূন্য * data) {
    (শূন্য) nth;
    সংরচনা htp_mm_context * mmctx = data;
    যদি (mmctx->quant_ib_first[ith] >= mmctx->quant_ib_last[ith]) {
        প্রত্যাগম;
    }
    সংরচনা htp_ops_context * octx = mmctx->octx;
    সংরচনা htp_thread_trace * tr = &octx->ctx->trace[ith];
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_A_QUANT, mmctx->quant_ib_first[ith]);

    ধ্রুব সংরচনা htp_tensor * src = mmctx->act;

    quantize_f16_f16_tiled_block_kernel(
        (ধ্রুব __fp16 *) mmctx->vtcm_act_raw,
        mmctx->vtcm_act,
        রিক্ত,
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
স্থির অন্তঃস্থ worker_callback_t htp_mm_act_quant_row_func(পূর্ণাঙ্ক weight_type, পূর্ণাঙ্ক act_type) {
    যদি (weight_type == HTP_TYPE_F16) {
        প্রত্যাগম (act_type == HTP_TYPE_F16) ? quantize_f16_f16_tiled : quantize_f32_f16_tiled;
    }
    যদি (weight_type == HTP_TYPE_Q2_K) {
        প্রত্যাগম quantize_f32_q8_1_s16_tiled;
    }
    প্রত্যাগম htp_mm_weight_has_offset(weight_type) ? quantize_f32_q8_1_tiled : quantize_f32_q8_0_tiled;
}

স্থির অন্তঃস্থ worker_callback_t htp_mm_act_quant_block_func(পূর্ণাঙ্ক weight_type, পূর্ণাঙ্ক act_type) {
    যদি (weight_type == HTP_TYPE_F16) {
        প্রত্যাগম (act_type == HTP_TYPE_F16) ? quantize_f16_f16_tiled_block : quantize_f32_f16_tiled_block;
    }
    যদি (weight_type == HTP_TYPE_Q2_K) {
        প্রত্যাগম quantize_f32_q8_1_s16_tiled_block;
    }
    প্রত্যাগম htp_mm_weight_has_offset(weight_type) ? quantize_f32_q8_1_tiled_block : quantize_f32_q8_0_tiled_block;
}

স্থির শূন্য hvx_mm_4d_tiled(unsigned পূর্ণাঙ্ক nth, unsigned পূর্ণাঙ্ক ith, শূন্য * data) {
    htp_matmul_preamble;

    ধ্রুব uint32_t src0_nrows = mmctx->src0_row_end - mmctx->src0_row_start;
    ধ্রুব uint32_t cur_m_rows = mmctx->cur_m_rows ? mmctx->cur_m_rows : (ne11 * ne12 * ne13);
    ধ্রুব uint32_t cur_m_start = mmctx->cur_m_start;

    ধ্রুব uint32_t src0_start_row  = mmctx->src0_row_start + src0_nrows_per_thread * ith;
    ধ্রুব uint32_t src0_end_row    = MIN(src0_start_row + src0_nrows_per_thread, mmctx->src0_row_end);

    সংরচনা htp_thread_trace * tr = &octx->ctx->trace[ith];

    ধ্রুব সংরচনা htp_mm_kernel_params * kparams = (ধ্রুব সংরচনা htp_mm_kernel_params *) octx->kernel_params;
    ধ্রুব uint32_t n_prefetch = kparams->n_prefetch;
    প্রতিজ্ঞা(n_prefetch >= 2 && n_prefetch <= HTP_MM_MAX_PREFETCH && (n_prefetch & (n_prefetch - 1)) == 0);

    ধ্রুব size_t dst_row_size = nb1;
    ধ্রুব size_t act_stride = mmctx->vtcm_act_stride;

    uint8_t * restrict vtcm_src0_ptr = mmctx->vtcm_src0 + mmctx->vtcm_src0_size_per_thread * ith;
    uint8_t * restrict act_data = mmctx->vtcm_act;

    ধ্রুব uint32_t tile_size = htp_tensor_get_tile_size(src0->type);
    ধ্রুব uint32_t aligned_tile_size = htp_tensor_get_tile_padded_size(src0->type);

    ধ্রুব uint32_t n_k_tiles_w = ne00 / htp_tensor_get_tile_k(src0->type);
    ধ্রুব uint32_t tile_row_stride = n_k_tiles_w * tile_size;
    ধ্রুব uint32_t tile_row_transfer_size_aligned = n_k_tiles_w * aligned_tile_size;
    ধ্রুব uint32_t src0_slice_stride = ((ne01 + 31) / 32) * tile_row_stride;

    ধ্রুব uint32_t ct_start = src0_start_row / 32;
    ধ্রুব uint32_t ct_end   = (src0_end_row + 31) / 32;

    যদি (src0_start_row >= src0_end_row || cur_m_rows == 0) {
        প্রত্যাগম;
    }

    ধ্রুব uint32_t total_batches = ne12 * ne13;
    ধ্রুব uint32_t b_start = fastdiv(cur_m_start, &kparams->div_ne1);
    uint32_t b_end = fastdiv(cur_m_start + cur_m_rows + ne11 - 1, &kparams->div_ne1);
    b_end = MIN(b_end, total_batches);

    uint32_t b_grp_start = b_start;
    যাবত্ (b_grp_start < b_end) {
        ধ্রুব uint32_t b3 = fastdiv(b_grp_start, &kparams->div_ne12);
        ধ্রুব uint32_t b2 = b_grp_start - b3 * ne12;
        ধ্রুব uint32_t i02 = fastdiv(b2, &kparams->div_r2);
        ধ্রুব uint32_t i03 = fastdiv(b3, &kparams->div_r3);

        uint32_t b_grp_end = b_grp_start + 1;
        যাবত্ (b_grp_end < b_end) {
            ধ্রুব uint32_t cur_b3 = fastdiv(b_grp_end, &kparams->div_ne12);
            ধ্রুব uint32_t cur_b2 = b_grp_end - cur_b3 * ne12;
            ধ্রুব uint32_t cur_i02 = fastdiv(cur_b2, &kparams->div_r2);
            ধ্রুব uint32_t cur_i03 = fastdiv(cur_b3, &kparams->div_r3);
            যদি (cur_i02 != i02 || cur_i03 != i03) {
                ভঙ্গ;
            }
            b_grp_end++;
        }

        ধ্রুব uint32_t slice_idx = i03 * ne02 + i02;
        ধ্রুব dma_addr_t src0_slice = src0->data + (size_t) slice_idx * src0_slice_stride;

        uint32_t ঠেলো_ct = ct_start;
        প্রতি (uint32_t d = 0; d < n_prefetch && ঠেলো_ct < ct_end; d++, ঠেলো_ct++) {
            dma_queue_ঠেলো(dma_q, dma_make_data(vtcm_src0_ptr + d * tile_row_transfer_size_aligned,
                           src0_slice + (size_t) ঠেলো_ct * tile_row_stride),
                           aligned_tile_size, tile_size, tile_size, n_k_tiles_w);
        }

        প্রতি (uint32_t ct = ct_start; ct < ct_end; ct++) {
            ধ্রুব uint8_t * w_tile = (শূন্য *) dma_queue_বের_করো(dma_q).dst;

            পূর্ণাঙ্ক valid_rows = (পূর্ণাঙ্ক)ne0 - (পূর্ণাঙ্ক)(ct * 32);
            valid_rows = MIN(32, MAX(0, valid_rows));

            htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, ct);
            প্রতি (uint32_t b = b_grp_start; b < b_grp_end; b++) {
                ধ্রুব uint32_t b_m_start = b * ne11;
                ধ্রুব uint32_t m_first = MAX(cur_m_start, b_m_start);
                ধ্রুব uint32_t m_last  = MIN(cur_m_start + cur_m_rows, b_m_start + ne11);
                যদি (m_first >= m_last) অনুবর্ত;

                ধ্রুব uint32_t cur_b3 = fastdiv(b, &kparams->div_ne12);
                ধ্রুব uint32_t cur_b2 = b - cur_b3 * ne12;
                uint8_t * dst_batch_base = (uint8_t *) dst->data + (size_t) cur_b2 * nb2 + (size_t) cur_b3 * nb3;

                ধ্রুব uint32_t chunk_m_offset = m_first - cur_m_start;
                ধ্রুব uint32_t dst_m_offset   = m_first - b_m_start;
                ধ্রুব uint32_t batch_nrows    = m_last - m_first;

                uint32_t ir1 = 0;
                প্রতি (; ir1 + 1 < batch_nrows; ir1 += 2) {
                    ধ্রুব uint8_t * restrict act_col0 = (ধ্রুব uint8_t *) (act_data + (chunk_m_offset + ir1 + 0) * act_stride);
                    ধ্রুব uint8_t * restrict act_col1 = (ধ্রুব uint8_t *) (act_data + (chunk_m_offset + ir1 + 1) * act_stride);
                    ভিন্ন * restrict dst_row0 = (ভিন্ন *) (dst_batch_base + (dst_m_offset + ir1 + 0) * dst_row_size);
                    ভিন্ন * restrict dst_row1 = (ভিন্ন *) (dst_batch_base + (dst_m_offset + ir1 + 1) * dst_row_size);
                    ভিন্ন * dst_ptr0 = &dst_row0[ct * 32];
                    ভিন্ন * dst_ptr1 = &dst_row1[ct * 32];
                    mmctx->vec_dot_32x2(ne10, dst_ptr0, dst_ptr1, w_tile, act_col0, act_col1, valid_rows, রিক্ত, রিক্ত);
                }
                প্রতি (; ir1 < batch_nrows; ++ir1) {
                    ধ্রুব uint8_t * restrict act_col = (ধ্রুব uint8_t *) (act_data + (chunk_m_offset + ir1) * act_stride);
                    ভিন্ন * restrict dst_row = (ভিন্ন *) (dst_batch_base + (dst_m_offset + ir1) * dst_row_size);
                    ভিন্ন * dst_ptr = &dst_row[ct * 32];
                    mmctx->vec_dot_32x1(ne10, dst_ptr, w_tile, act_col, valid_rows, রিক্ত);
                }
            }
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ct);

            যদি (ঠেলো_ct < ct_end) {
                dma_queue_ঠেলো(dma_q, dma_make_data(w_tile, src0_slice + (size_t) ঠেলো_ct * tile_row_stride),
                               aligned_tile_size, tile_size, tile_size, n_k_tiles_w);
                ঠেলো_ct++;
            }
        }
        b_grp_start = b_grp_end;
    }
    যদি (src2) {
        hvx_tensor_add_f32_grid(dst, src2, cur_m_start, cur_m_start + cur_m_rows, src0_start_row, src0_end_row, &kparams->div_ne12_ne1, &kparams->div_ne1);
    }
}

স্থির শূন্য hvx_mm_2d_flat(unsigned পূর্ণাঙ্ক nth, unsigned পূর্ণাঙ্ক ith, শূন্য * data) {
    htp_matmul_preamble;

    ধ্রুব সংরচনা htp_mm_kernel_params * kparams = (ধ্রুব সংরচনা htp_mm_kernel_params *) octx->kernel_params;
    ধ্রুব uint32_t n_prefetch = kparams->n_prefetch;
    প্রতিজ্ঞা(n_prefetch >= 2 && n_prefetch <= HTP_MM_MAX_PREFETCH && (n_prefetch & (n_prefetch - 1)) == 0);
    ধ্রুব uint32_t prefetch_mask = n_prefetch - 1;

    ধ্রুব uint32_t src0_nrows = mmctx->src0_row_end - mmctx->src0_row_start;  // src0 rows
    ধ্রুব uint32_t act_nrows = mmctx->cur_m_rows ? mmctx->cur_m_rows : mmctx->act_nrows;
    ধ্রুব uint32_t cur_m_start = mmctx->cur_m_start;

    ধ্রুব uint32_t src0_start_row  = mmctx->src0_row_start + src0_nrows_per_thread * ith;
    ধ্রুব uint32_t src0_end_row    = MIN(src0_start_row + src0_nrows_per_thread, mmctx->src0_row_end);
    ধ্রুব uint32_t src0_end_row_x2 = src0_start_row + ((src0_end_row - src0_start_row) & ~1U);

    সংরচনা htp_thread_trace * tr = &octx->ctx->trace[ith];

    ধ্রুব size_t dst_row_size  = nb1;
    ধ্রুব size_t src0_row_size = nb01;
    ধ্রুব size_t act_row_size = nb11;

    ধ্রুব size_t src0_stride = mmctx->vtcm_src0_stride;
    ধ্রুব size_t act_stride = mmctx->vtcm_act_stride;

    // Per-thread VTCMs for all tensors
    uint8_t * restrict vtcm_dst_ptr  = mmctx->vtcm_dst  + mmctx->vtcm_dst_size_per_thread  * ith;
    uint8_t * restrict vtcm_src0_ptr = mmctx->vtcm_src0 + mmctx->vtcm_src0_size_per_thread * ith;
    uint8_t * restrict act_data      = mmctx->vtcm_act;

    ধ্রুব dma_addr_t src0_row = src0->data;


    // Prefill vtcm with src0 rows
    যদি (src0_start_row < src0_end_row) {
        প্রতি (uint32_t ir0 = src0_start_row; ir0 < src0_end_row_x2; ir0 += 2) {
            ধ্রুব পূর্ণাঙ্ক is0 = (ir0 - src0_start_row);
            যদি (is0 >= (পূর্ণাঙ্ক)n_prefetch) {
                ভঙ্গ;
            }
            dma_queue_ঠেলো(dma_q, dma_make_data(vtcm_src0_ptr + is0 * src0_stride, src0_row + ir0 * src0_row_size),
                           src0_stride, src0_row_size, src0_row_size, 2);
        }
    }

    যদি (src0_start_row >= src0_end_row) {
        প্রত্যাগম;
    }

    // Process src0 rows
    প্রতি (uint32_t ir0 = src0_start_row; ir0 < src0_end_row_x2; ir0 += 2) {
        ধ্রুব uint8_t * ss0 = (শূন্য *) dma_queue_বের_করো(dma_q).dst;

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, ir0);
        // Process act columns in pairs (2x2 tiling)
        uint32_t ir1 = 0;
        প্রতি (; ir1 + 1 < act_nrows; ir1 += 2) {
            ধ্রুব uint8_t * restrict act_col0 = (ধ্রুব uint8_t *) (act_data + (ir1+0) * act_stride);
            ধ্রুব uint8_t * restrict act_col1 = (ধ্রুব uint8_t *) (act_data + (ir1+1) * act_stride);
            ভিন্ন * restrict dst_row0 = (ভিন্ন *) (dst->data + ((cur_m_start + ir1+0) * dst_row_size));
            ভিন্ন * restrict dst_row1 = (ভিন্ন *) (dst->data + ((cur_m_start + ir1+1) * dst_row_size));
            mmctx->vec_dot_2x2(ne00, &dst_row0[ir0], &dst_row1[ir0], ss0, ss0 + src0_stride, act_col0, act_col1);
        }

        // Handle remaining act rows (fallback to 2x1)
        প্রতি (; ir1 < act_nrows; ++ir1) {
            ধ্রুব uint8_t * restrict act_col = (ধ্রুব uint8_t *) (act_data + ir1 * act_stride);
            ভিন্ন * restrict dst_row          = (ভিন্ন *) (dst->data + ((cur_m_start + ir1) * dst_row_size));
            mmctx->vec_dot_2x1(ne00, &dst_row[ir0], ss0, ss0 + src0_stride, act_col);
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ir0);

        // Prefetch next (n + vtcm_nrows) row
        ধ্রুব পূর্ণাঙ্ক pr0 = (ir0 + n_prefetch);
        ধ্রুব পূর্ণাঙ্ক is0 = (pr0 - src0_start_row) & prefetch_mask;
        যদি (pr0 < src0_end_row_x2) {
            dma_queue_ঠেলো(dma_q, dma_make_data(vtcm_src0_ptr + is0 * src0_stride, src0_row + pr0 * src0_row_size),
                           src0_stride, src0_row_size, src0_row_size, 2);
        }
    }

    // Process the last row (if any)
    যদি (src0_end_row != src0_end_row_x2) {
        uint32_t  ir0 = src0_end_row_x2;
        ধ্রুব পূর্ণাঙ্ক is0 = (ir0 - src0_start_row) & prefetch_mask;
        dma_queue_ঠেলো(dma_q, dma_make_data(vtcm_src0_ptr + is0 * src0_stride, src0_row + ir0 * src0_row_size),
                       src0_stride, src0_row_size, src0_row_size, 1);
        ধ্রুব uint8_t * ss0 = (শূন্য *) dma_queue_বের_করো(dma_q).dst;

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, ir0);
        #pragma unroll(2)
        প্রতি (uint32_t ir1 = 0; ir1 < act_nrows; ++ir1) {
            ধ্রুব uint8_t * restrict act_col = (ধ্রুব uint8_t *) (act_data + ir1 * act_stride);
            ভিন্ন * restrict dst_row          = (ভিন্ন *) (dst->data + ((cur_m_start + ir1) * dst_row_size));
            mmctx->vec_dot_1x1(ne00, &dst_row[ir0], ss0, act_col);
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ir0);
    }
    যদি (src2) {
        hvx_tensor_add_f32_grid(dst, src2, cur_m_start, cur_m_start + act_nrows, src0_start_row, src0_end_row, &kparams->div_ne12_ne1, &kparams->div_ne1);
    }
}

স্থির শূন্য hvx_mv_2d_flat(unsigned পূর্ণাঙ্ক nth, unsigned পূর্ণাঙ্ক ith, শূন্য * data) {
    htp_matmul_preamble;

    ধ্রুব uint32_t src0_nrows = mmctx->src0_row_end - mmctx->src0_row_start;

    ধ্রুব uint32_t src0_start_row  = mmctx->src0_row_start + src0_nrows_per_thread * ith;
    ধ্রুব uint32_t src0_end_row    = MIN(src0_start_row + src0_nrows_per_thread, mmctx->src0_row_end);

    সংরচনা htp_thread_trace * tr = &octx->ctx->trace[ith];

    ধ্রুব size_t dst_row_size  = nb1;
    ধ্রুব size_t src0_row_size = nb01;
    ধ্রুব size_t act_row_size  = nb11;

    ধ্রুব size_t src0_stride = mmctx->vtcm_src0_stride;
    ধ্রুব size_t act_stride  = mmctx->vtcm_act_stride;

    // Per-thread VTCMs for all tensors
    uint8_t * vtcm_dst_ptr  = mmctx->vtcm_dst  + mmctx->vtcm_dst_size_per_thread  * ith;
    uint8_t * vtcm_src0_ptr = mmctx->vtcm_src0 + mmctx->vtcm_src0_size_per_thread * ith;
    uint8_t * act_data      = mmctx->vtcm_act;

    ভিন্ন * tmp = (ভিন্ন *) vtcm_dst_ptr;

    ধ্রুব dma_addr_t src0_row = src0->data;
    ধ্রুব uint8_t * restrict act_col = (ধ্রুব uint8_t *) act_data;
    ভিন্ন * restrict dst_col         = (ভিন্ন *) dst->data;

    ধ্রুব uint32_t src0_end_row_x2 = src0_start_row + ((src0_end_row - src0_start_row) & ~1U);

    ধ্রুব সংরচনা htp_mm_kernel_params * kparams = (ধ্রুব সংরচনা htp_mm_kernel_params *) octx->kernel_params;
    ধ্রুব uint32_t n_prefetch = kparams->n_prefetch;
    প্রতিজ্ঞা(n_prefetch >= 2 && n_prefetch <= HTP_MM_MAX_PREFETCH && (n_prefetch & (n_prefetch - 1)) == 0);
    ধ্রুব uint32_t prefetch_mask = n_prefetch - 1;

    // Prefill vtcm with 2x src0 rows
    যদি (src0_start_row < src0_end_row) {
        যদি (src2) {
            ভিন্ন * vtcm_bias_ptr = (ভিন্ন *) mmctx->vtcm_bias + src0_start_row;
            ধ্রুব dma_addr_t src2_addr = src2->data + src0_start_row * পরিমাণ(ভিন্ন);
            পূর্ণাঙ্ক slice_size = (পূর্ণাঙ্ক)src0_end_row - (পূর্ণাঙ্ক)src0_start_row;
            যদি (slice_size > 0) {
                dma_queue_ঠেলো(dma_q, dma_make_data(vtcm_bias_ptr, src2_addr),
                               slice_size * পরিমাণ(ভিন্ন), slice_size * পরিমাণ(ভিন্ন), slice_size * পরিমাণ(ভিন্ন), 1);
                dma_queue_বের_করো_nowait(dma_q);
            }
        }
        প্রতি (uint32_t ir0 = src0_start_row; ir0 < src0_end_row_x2; ir0 += 2) {
            ধ্রুব uint32_t is0 = (ir0 - src0_start_row);
            যদি (is0 >= n_prefetch) {
                ভঙ্গ;
            }
            dma_queue_ঠেলো(dma_q, dma_make_data(vtcm_src0_ptr + is0 * src0_stride, src0_row + ir0 * src0_row_size),
                           src0_stride, src0_row_size, src0_row_size, 2);
        }
    }


    যদি (src0_start_row >= src0_end_row) {
        প্রত্যাগম;
    }

    // Process src0 rows
    প্রতি (uint32_t ir0 = src0_start_row; ir0 < src0_end_row_x2; ir0 += 2) {
        ধ্রুব uint8_t * ss0 = (শূন্য *) dma_queue_বের_করো(dma_q).dst;
        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, ir0);
        mmctx->vec_dot_2x1(ne00, &tmp[ir0 - src0_start_row], ss0, ss0 + src0_stride, act_col);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ir0);

        // Prefetch next (n + vtcm_nrows) row
        ধ্রুব uint32_t pr0 = (ir0 + n_prefetch);
        ধ্রুব uint32_t is0 = (pr0 - src0_start_row) & prefetch_mask;
        যদি (pr0 < src0_end_row_x2) {
            dma_queue_ঠেলো(dma_q, dma_make_data(vtcm_src0_ptr + is0 * src0_stride, src0_row + pr0 * src0_row_size),
                           src0_stride, src0_row_size, src0_row_size, 2);
        }
    }

    // Process the last row (if any)
    যদি (src0_end_row != src0_end_row_x2) {
        ধ্রুব uint32_t ir0 = src0_end_row_x2;
        ধ্রুব uint32_t is0 = (ir0 - src0_start_row) & prefetch_mask;
        dma_queue_ঠেলো(dma_q, dma_make_data(vtcm_src0_ptr + is0 * src0_stride, src0_row + ir0 * src0_row_size),
                       src0_stride, src0_row_size, src0_row_size, 1);
        ধ্রুব uint8_t * ss0 = (শূন্য *) dma_queue_বের_করো(dma_q).dst;
        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, ir0);
        mmctx->vec_dot_1x1(ne00, &tmp[ir0 - src0_start_row], ss0, act_col);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ir0);
    }

    পূর্ণাঙ্ক copy_cnt = src0_end_row - src0_start_row;
    যদি (copy_cnt > 0) {
        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, src0_end_row);
        যদি (src2) {
            hvx_add_f32_uuu((uint8_t *) &dst_col[src0_start_row],
                            (ধ্রুব uint8_t *) tmp,
                            (ধ্রুব uint8_t *) ((ধ্রুব ভিন্ন *) mmctx->vtcm_bias + src0_start_row),
                            copy_cnt);
        } অন্যথা {
            hvx_copy_f32_ua((uint8_t *) &dst_col[src0_start_row], (uint8_t *) tmp, copy_cnt);
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, src0_end_row);
    }
}

স্থির শূন্য hvx_mm_4d_flat(unsigned পূর্ণাঙ্ক nth, unsigned পূর্ণাঙ্ক ith, শূন্য * data) {
    htp_matmul_preamble;

    ধ্রুব সংরচনা htp_mm_kernel_params * kparams = (ধ্রুব সংরচনা htp_mm_kernel_params *) octx->kernel_params;
    ধ্রুব uint32_t n_prefetch = kparams->n_prefetch;
    প্রতিজ্ঞা(n_prefetch >= 2 && n_prefetch <= HTP_MM_MAX_PREFETCH && (n_prefetch & (n_prefetch - 1)) == 0);
    ধ্রুব uint32_t prefetch_mask = n_prefetch - 1;

    ধ্রুব uint32_t src0_nrows = mmctx->src0_row_end - mmctx->src0_row_start;
    ধ্রুব uint32_t cur_m_rows = mmctx->cur_m_rows ? mmctx->cur_m_rows : (ne11 * ne12 * ne13);
    ধ্রুব uint32_t cur_m_start = mmctx->cur_m_start;

    ধ্রুব uint32_t src0_start_row  = mmctx->src0_row_start + src0_nrows_per_thread * ith;
    ধ্রুব uint32_t src0_end_row    = MIN(src0_start_row + src0_nrows_per_thread, mmctx->src0_row_end);
    ধ্রুব uint32_t src0_end_row_x2 = src0_start_row + ((src0_end_row - src0_start_row) & ~1U);

    সংরচনা htp_thread_trace * tr = &octx->ctx->trace[ith];

    ধ্রুব size_t dst_row_size  = nb1;
    ধ্রুব size_t src0_row_size = nb01;
    ধ্রুব size_t src0_stride = mmctx->vtcm_src0_stride;
    ধ্রুব size_t act_stride  = mmctx->vtcm_act_stride;

    uint8_t * restrict vtcm_src0_ptr = mmctx->vtcm_src0 + mmctx->vtcm_src0_size_per_thread * ith;
    uint8_t * restrict act_data      = mmctx->vtcm_act;


    যদি (src0_start_row >= src0_end_row || cur_m_rows == 0) {
        প্রত্যাগম;
    }

    ধ্রুব uint32_t total_batches = ne12 * ne13;
    ধ্রুব uint32_t b_start = fastdiv(cur_m_start, &kparams->div_ne1);
    uint32_t b_end = fastdiv(cur_m_start + cur_m_rows + ne11 - 1, &kparams->div_ne1);
    b_end = MIN(b_end, total_batches);

    uint32_t b_grp_start = b_start;
    যাবত্ (b_grp_start < b_end) {
        ধ্রুব uint32_t b3 = fastdiv(b_grp_start, &kparams->div_ne12);
        ধ্রুব uint32_t b2 = b_grp_start - b3 * ne12;
        ধ্রুব uint32_t i02 = fastdiv(b2, &kparams->div_r2);
        ধ্রুব uint32_t i03 = fastdiv(b3, &kparams->div_r3);

        uint32_t b_grp_end = b_grp_start + 1;
        যাবত্ (b_grp_end < b_end) {
            ধ্রুব uint32_t cur_b3 = fastdiv(b_grp_end, &kparams->div_ne12);
            ধ্রুব uint32_t cur_b2 = b_grp_end - cur_b3 * ne12;
            ধ্রুব uint32_t cur_i02 = fastdiv(cur_b2, &kparams->div_r2);
            ধ্রুব uint32_t cur_i03 = fastdiv(cur_b3, &kparams->div_r3);
            যদি (cur_i02 != i02 || cur_i03 != i03) {
                ভঙ্গ;
            }
            b_grp_end++;
        }

        ধ্রুব dma_addr_t src0_row = src0->data + ((size_t) i02 * nb02 + (size_t) i03 * nb03);

        প্রতি (uint32_t ir0 = src0_start_row; ir0 < src0_end_row_x2; ir0 += 2) {
            ধ্রুব পূর্ণাঙ্ক is0 = (ir0 - src0_start_row);
            যদি (is0 >= (পূর্ণাঙ্ক)n_prefetch) {
                ভঙ্গ;
            }
            dma_queue_ঠেলো(dma_q, dma_make_data(vtcm_src0_ptr + is0 * src0_stride, src0_row + (size_t) ir0 * src0_row_size),
                           src0_stride, src0_row_size, src0_row_size, 2);
        }

        প্রতি (uint32_t ir0 = src0_start_row; ir0 < src0_end_row_x2; ir0 += 2) {
            ধ্রুব uint8_t * ss0 = (শূন্য *) dma_queue_বের_করো(dma_q).dst;

            htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, ir0);
            প্রতি (uint32_t b = b_grp_start; b < b_grp_end; b++) {
                ধ্রুব uint32_t b_m_start = b * ne11;
                ধ্রুব uint32_t m_first = MAX(cur_m_start, b_m_start);
                ধ্রুব uint32_t m_last  = MIN(cur_m_start + cur_m_rows, b_m_start + ne11);
                যদি (m_first >= m_last) অনুবর্ত;

                ধ্রুব uint32_t cur_b3 = fastdiv(b, &kparams->div_ne12);
                ধ্রুব uint32_t cur_b2 = b - cur_b3 * ne12;
                uint8_t * dst_batch_base = (uint8_t *) dst->data + (size_t) cur_b2 * nb2 + (size_t) cur_b3 * nb3;

                ধ্রুব uint32_t chunk_m_offset = m_first - cur_m_start;
                ধ্রুব uint32_t dst_m_offset   = m_first - b_m_start;
                ধ্রুব uint32_t batch_nrows    = m_last - m_first;

                uint32_t ir1 = 0;
                প্রতি (; ir1 + 1 < batch_nrows; ir1 += 2) {
                    ধ্রুব uint8_t * restrict act_col0 = (ধ্রুব uint8_t *) (act_data + (chunk_m_offset + ir1 + 0) * act_stride);
                    ধ্রুব uint8_t * restrict act_col1 = (ধ্রুব uint8_t *) (act_data + (chunk_m_offset + ir1 + 1) * act_stride);
                    ভিন্ন * restrict dst_row0 = (ভিন্ন *) (dst_batch_base + (dst_m_offset + ir1 + 0) * dst_row_size);
                    ভিন্ন * restrict dst_row1 = (ভিন্ন *) (dst_batch_base + (dst_m_offset + ir1 + 1) * dst_row_size);
                    mmctx->vec_dot_2x2(ne00, &dst_row0[ir0], &dst_row1[ir0], ss0, ss0 + src0_stride, act_col0, act_col1);
                }
                প্রতি (; ir1 < batch_nrows; ++ir1) {
                    ধ্রুব uint8_t * restrict act_col = (ধ্রুব uint8_t *) (act_data + (chunk_m_offset + ir1) * act_stride);
                    ভিন্ন * restrict dst_row          = (ভিন্ন *) (dst_batch_base + (dst_m_offset + ir1) * dst_row_size);
                    mmctx->vec_dot_2x1(ne00, &dst_row[ir0], ss0, ss0 + src0_stride, act_col);
                }
            }
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ir0);

            ধ্রুব পূর্ণাঙ্ক pr0 = (ir0 + n_prefetch);
            ধ্রুব পূর্ণাঙ্ক is0 = (pr0 - src0_start_row) & prefetch_mask;
            যদি (pr0 < src0_end_row_x2) {
                dma_queue_ঠেলো(dma_q, dma_make_data(vtcm_src0_ptr + is0 * src0_stride, src0_row + (size_t) pr0 * src0_row_size),
                               src0_stride, src0_row_size, src0_row_size, 2);
            }
        }

        যদি (src0_end_row != src0_end_row_x2) {
            uint32_t ir0 = src0_end_row_x2;
            ধ্রুব পূর্ণাঙ্ক is0 = (ir0 - src0_start_row) & prefetch_mask;
            dma_queue_ঠেলো(dma_q, dma_make_data(vtcm_src0_ptr + is0 * src0_stride, src0_row + (size_t) ir0 * src0_row_size),
                           src0_stride, src0_row_size, src0_row_size, 1);
            ধ্রুব uint8_t * ss0 = (শূন্য *) dma_queue_বের_করো(dma_q).dst;

            htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, ir0);
            প্রতি (uint32_t b = b_grp_start; b < b_grp_end; b++) {
                ধ্রুব uint32_t b_m_start = b * ne11;
                ধ্রুব uint32_t m_first = MAX(cur_m_start, b_m_start);
                ধ্রুব uint32_t m_last  = MIN(cur_m_start + cur_m_rows, b_m_start + ne11);
                যদি (m_first >= m_last) অনুবর্ত;

                ধ্রুব uint32_t cur_b3 = fastdiv(b, &kparams->div_ne12);
                ধ্রুব uint32_t cur_b2 = b - cur_b3 * ne12;
                uint8_t * dst_batch_base = (uint8_t *) dst->data + (size_t) cur_b2 * nb2 + (size_t) cur_b3 * nb3;

                ধ্রুব uint32_t chunk_m_offset = m_first - cur_m_start;
                ধ্রুব uint32_t dst_m_offset   = m_first - b_m_start;
                ধ্রুব uint32_t batch_nrows    = m_last - m_first;

                প্রতি (uint32_t ir1 = 0; ir1 < batch_nrows; ++ir1) {
                    ধ্রুব uint8_t * restrict act_col = (ধ্রুব uint8_t *) (act_data + (chunk_m_offset + ir1) * act_stride);
                    ভিন্ন * restrict dst_row          = (ভিন্ন *) (dst_batch_base + (dst_m_offset + ir1) * dst_row_size);
                    mmctx->vec_dot_1x1(ne00, &dst_row[ir0], ss0, act_col);
                }
            }
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ir0);
        }

        b_grp_start = b_grp_end;
    }

    যদি (src2) {
        hvx_tensor_add_f32_grid(dst, src2, cur_m_start, cur_m_start + cur_m_rows, src0_start_row, src0_end_row, &kparams->div_ne12_ne1, &kparams->div_ne1);
    }
}

#define MMID_MATRIX_ROW(row_id, i1) matrix_rows[(row_id) * mmctx->mapping_stride + (i1)]

স্থির শূন্য hvx_mm_id_tiled(unsigned পূর্ণাঙ্ক nth, unsigned পূর্ণাঙ্ক ith, শূন্য * data) {
    htp_matmul_preamble;

    ধ্রুব সংরচনা htp_tensor * restrict ids = octx->src[2];

    ধ্রুব uint32_t src0_nrows      = mmctx->src0_row_end - mmctx->src0_row_start;  // src0 rows per expert
    ধ্রুব uint32_t act_nrows       = ne11;
    ধ্রুব uint32_t src0_start_row  = mmctx->src0_row_start + src0_nrows_per_thread * ith;
    ধ্রুব uint32_t src0_end_row    = MIN(src0_start_row + src0_nrows_per_thread, mmctx->src0_row_end);


    যদি (src0_start_row >= src0_end_row) {
        প্রত্যাগম;
    }

    সংরচনা htp_thread_trace * tr = &octx->ctx->trace[ith];

    ধ্রুব সংরচনা htp_mm_kernel_params * kparams = (ধ্রুব সংরচনা htp_mm_kernel_params *) octx->kernel_params;
    ধ্রুব uint32_t n_prefetch = kparams->n_prefetch;
    প্রতিজ্ঞা(n_prefetch >= 2 && n_prefetch <= HTP_MM_MAX_PREFETCH && (n_prefetch & (n_prefetch - 1)) == 0);

    ধ্রুব uint32_t n_ids = ids->ne[0];  // n_expert_used
    ধ্রুব uint32_t n_as  = ne02;        // n_expert

    ধ্রুব uint32_t *                matrix_row_counts = mmctx->matrix_row_counts;
    ধ্রুব সংরচনা mmid_row_mapping * matrix_rows       = mmctx->matrix_rows;

    ধ্রুব size_t dst_row_size  = nb1;
    ধ্রুব size_t act_row_size  = htp_mm_q8_0_tiled_row_size(ne10);

    ধ্রুব size_t act_stride = mmctx->vtcm_act_stride;

    // Per-thread VTCMs for all tensors
    uint8_t * restrict vtcm_src0_ptr = mmctx->vtcm_src0 + mmctx->vtcm_src0_size_per_thread * ith;
    uint8_t * restrict act_data = mmctx->vtcm_act;

    প্রতি (uint32_t cur_a = 0; cur_a < n_as; ++cur_a) {
        ধ্রুব int32_t cne1 = matrix_row_counts[cur_a];
        যদি (cne1 == 0) {
            অনুবর্ত;
        }

        ধ্রুব dma_addr_t src0_row = src0->data + cur_a * nb02;

        ধ্রুব uint32_t tile_size = htp_tensor_get_tile_size(src0->type);
        ধ্রুব uint32_t aligned_tile_size = htp_tensor_get_tile_padded_size(src0->type);
        ধ্রুব uint32_t n_k_tiles_w = ne00 / htp_tensor_get_tile_k(src0->type);
        ধ্রুব uint32_t tile_row_stride = n_k_tiles_w * tile_size;
        ধ্রুব uint32_t tile_row_transfer_size_aligned = n_k_tiles_w * aligned_tile_size;

        ধ্রুব uint32_t ct_start = src0_start_row / 32;
        ধ্রুব uint32_t ct_end   = (src0_end_row + 31) / 32;

        uint32_t ঠেলো_ct = ct_start;
        প্রতি (uint32_t d = 0; d < n_prefetch && ঠেলো_ct < ct_end; d++, ঠেলো_ct++) {
            dma_queue_ঠেলো(dma_q, dma_make_data(vtcm_src0_ptr + d * tile_row_transfer_size_aligned, src0_row + ঠেলো_ct * tile_row_stride),
                           aligned_tile_size, tile_size, tile_size, n_k_tiles_w);
        }

        প্রতি (uint32_t ct = ct_start; ct < ct_end; ct++) {
            ধ্রুব uint8_t * w_tile = (শূন্য *) dma_queue_বের_করো(dma_q).dst;

            পূর্ণাঙ্ক valid_rows = (পূর্ণাঙ্ক)ne0 - (পূর্ণাঙ্ক)(ct * 32);
            valid_rows = MIN(32, MAX(0, valid_rows));

            htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, ct);
            প্রতি (uint32_t cid = 0; cid < cne1; ++cid) {
                সংরচনা mmid_row_mapping row_mapping = MMID_MATRIX_ROW(cur_a, cid);
                ধ্রুব পূর্ণাঙ্ক               rm1         = row_mapping.i1;  // expert idx
                ধ্রুব পূর্ণাঙ্ক               rm2         = row_mapping.i2;  // token idx

                ধ্রুব uint32_t ir1 = fastmodulo(rm1, ne11, &mmctx->mm_div_ne11);        // act row idx
                ধ্রুব uint8_t * restrict act_col = (ধ্রুব uint8_t *) (act_data + (ir1 + rm2 * ne11 + 0) * act_stride);
                ভিন্ন * restrict dst_row = (ভিন্ন *) (dst->data + (rm1 * nb1 + rm2 * nb2 + 0));

                mmctx->vec_dot_32x1(ne10, &dst_row[ct * 32], w_tile, act_col, valid_rows, রিক্ত);
            }
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ct);

            যদি (ঠেলো_ct < ct_end) {
                dma_queue_ঠেলো(dma_q, dma_make_data(w_tile, src0_row + ঠেলো_ct * tile_row_stride),
                               aligned_tile_size, tile_size, tile_size, n_k_tiles_w);
                ঠেলো_ct++;
            }
        }
    }
}

স্থির শূন্য hvx_mv_id_tiled(unsigned পূর্ণাঙ্ক nth, unsigned পূর্ণাঙ্ক ith, শূন্য * data) {
    htp_matmul_preamble;

    ধ্রুব সংরচনা htp_tensor * restrict ids = octx->src[2];

    ধ্রুব uint32_t src0_nrows      = mmctx->src0_row_end - mmctx->src0_row_start;  // src0 rows per expert
    ধ্রুব uint32_t src0_start_row  = mmctx->src0_row_start + src0_nrows_per_thread * ith;
    ধ্রুব uint32_t src0_end_row    = MIN(src0_start_row + src0_nrows_per_thread, mmctx->src0_row_end);


    যদি (src0_start_row >= src0_end_row) {
        প্রত্যাগম;
    }

    সংরচনা htp_thread_trace * tr = &octx->ctx->trace[ith];

    ধ্রুব সংরচনা htp_mm_kernel_params * kparams = (ধ্রুব সংরচনা htp_mm_kernel_params *) octx->kernel_params;
    ধ্রুব uint32_t n_prefetch = kparams->n_prefetch;
    প্রতিজ্ঞা(n_prefetch >= 2 && n_prefetch <= HTP_MM_MAX_PREFETCH && (n_prefetch & (n_prefetch - 1)) == 0);

    প্রতিজ্ঞা(ne13 % ne03 == 0);

    ধ্রুব size_t dst_row_size  = nb1;
    ধ্রুব size_t act_row_size = htp_mm_q8_0_tiled_row_size(ne10);

    ধ্রুব uint32_t n_aids = src2->ne[0];  // num activated experts
    ধ্রুব uint32_t n_ids  = ne02;         // num experts

    // Per-thread VTCMs for all tensors
    uint8_t * restrict vtcm_src0_ptr = mmctx->vtcm_src0 + mmctx->vtcm_src0_size_per_thread * ith;
    uint8_t * restrict act_data = mmctx->vtcm_act;

    প্রতি (uint32_t ie1 = 0; ie1 < n_aids; ++ie1) {  // for each expert
        ধ্রুব int32_t eid = *(ধ্রুব int32_t *) ((ধ্রুব uint8_t *) src2->data + ie1 * src2->nb[0]);
        যদি (eid < 0) {
            অনুবর্ত;
        }
        প্রতিজ্ঞা(eid < (int32_t) n_ids);

        ধ্রুব dma_addr_t src0_row = src0->data + eid * nb02;
        ধ্রুব uint8_t * restrict act_col = (ধ্রুব uint8_t *) act_data;
        ভিন্ন * restrict dst_row          = (ভিন্ন *) (dst->data + ie1 * nb1);

        ধ্রুব uint32_t tile_size = htp_tensor_get_tile_size(src0->type);
        ধ্রুব uint32_t aligned_tile_size = htp_tensor_get_tile_padded_size(src0->type);
        ধ্রুব uint32_t n_k_tiles_w = ne00 / htp_tensor_get_tile_k(src0->type);
        ধ্রুব uint32_t tile_row_stride = n_k_tiles_w * tile_size;
        ধ্রুব uint32_t tile_row_transfer_size_aligned = n_k_tiles_w * aligned_tile_size;

        ধ্রুব uint32_t ct_start = src0_start_row / 32;
        ধ্রুব uint32_t ct_end   = (src0_end_row + 31) / 32;

        uint32_t ঠেলো_ct = ct_start;
        প্রতি (uint32_t d = 0; d < n_prefetch && ঠেলো_ct < ct_end; d++, ঠেলো_ct++) {
            dma_queue_ঠেলো(dma_q, dma_make_data(vtcm_src0_ptr + d * tile_row_transfer_size_aligned, src0_row + ঠেলো_ct * tile_row_stride),
                           aligned_tile_size, tile_size, tile_size, n_k_tiles_w);
        }

        প্রতি (uint32_t ct = ct_start; ct < ct_end; ct++) {
            ধ্রুব uint8_t * w_tile = (শূন্য *) dma_queue_বের_করো(dma_q).dst;

            পূর্ণাঙ্ক valid_rows = (পূর্ণাঙ্ক)ne0 - (পূর্ণাঙ্ক)(ct * 32);
            valid_rows = MIN(32, MAX(0, valid_rows));

            htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, ct);
            mmctx->vec_dot_32x1(ne10, &dst_row[ct * 32], w_tile, act_col, valid_rows, রিক্ত);
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ct);

            যদি (ঠেলো_ct < ct_end) {
                dma_queue_ঠেলো(dma_q, dma_make_data(w_tile, src0_row + ঠেলো_ct * tile_row_stride),
                               aligned_tile_size, tile_size, tile_size, n_k_tiles_w);
                ঠেলো_ct++;
            }
        }
    }
}

স্থির শূন্য hvx_mv_id_nx_tiled(unsigned পূর্ণাঙ্ক nth, unsigned পূর্ণাঙ্ক ith, শূন্য * data) {
    সংরচনা htp_mm_context * mmctx = (সংরচনা htp_mm_context *) data;
    সংরচনা htp_ops_context * octx = mmctx->octx;
    ধ্রুব সংরচনা htp_mm_kernel_params * kparams = (ধ্রুব সংরচনা htp_mm_kernel_params *) octx->kernel_params;
    ধ্রুব uint32_t n_weights      = kparams->n_weights;
    ধ্রুব সংরচনা htp_tensor * restrict src0 = octx->src[0];
    ধ্রুব সংরচনা htp_tensor * restrict act  = octx->src[n_weights];
    ধ্রুব সংরচনা htp_tensor * restrict ids  = octx->src[n_weights + 1];


    সংরচনা htp_thread_trace * tr = &octx->ctx->trace[ith];

    ধ্রুব uint32_t n_prefetch = kparams->n_prefetch;
    প্রতিজ্ঞা(n_prefetch >= 2 && n_prefetch <= HTP_MM_MAX_PREFETCH && (n_prefetch & (n_prefetch - 1)) == 0);

    ধ্রুব uint32_t n_aids = ids->ne[0];
    ধ্রুব uint32_t n_ids  = src0->ne[2];

    uint8_t * restrict vtcm_src0_ptr = mmctx->vtcm_src0 + mmctx->vtcm_src0_size_per_thread * ith;
    uint8_t * restrict act_data = mmctx->vtcm_act;

    প্রতি (uint32_t ie1 = 0; ie1 < n_aids; ++ie1) {
        ধ্রুব int32_t eid = *(ধ্রুব int32_t *) ((ধ্রুব uint8_t *) ids->data + ie1 * ids->nb[0]);
        যদি (eid < 0) অনুবর্ত;
        প্রতিজ্ঞা(eid < (int32_t) n_ids);

        প্রতি (uint32_t p = 0; p < n_weights; ++p) {
            ধ্রুব সংরচনা htp_tensor * restrict src_w = octx->src[p];
            ধ্রুব সংরচনা htp_tensor * restrict dst   = octx->dsts[p];
            যদি (!src_w || !dst) অনুবর্ত;
            dma_queue * dma_q = octx->ctx->dma[ith];

            ধ্রুব uint32_t ne01 = src_w->ne[1];
            uint32_t start_row = 0;
            uint32_t end_row   = ne01;
            যদি (octx->ctx->mdev.গুনতি > 1) {
                ধ্রুব bool can_split = htp_tensor_can_row_partition(dst, পরিমাণ(ভিন্ন));
                ধ্রুব সংরচনা htp_tensor_mdev_range range = htp_tensor_mdev_partition(ne01, can_split ? 32 : 0, octx->ctx->mdev.idx, octx->ctx->mdev.গুনতি, &octx->ctx->mdev.গুনতি_div);
                start_row = range.start;
                end_row   = range.start + range.গুনতি;
            }

            ধ্রুব uint32_t nrows = end_row - start_row;
            uint32_t src0_nrows_per_thread = fastdiv(nrows + nth - 1, &octx->n_threads_div);
            src0_nrows_per_thread = hex_round_up(src0_nrows_per_thread, 32);

            ধ্রুব uint32_t src0_start_row = start_row + src0_nrows_per_thread * ith;
            ধ্রুব uint32_t src0_end_row   = MIN(src0_start_row + src0_nrows_per_thread, end_row);
            যদি (src0_start_row >= src0_end_row) অনুবর্ত;

            ধ্রুব dma_addr_t src0_row = src_w->data + eid * src_w->nb[2];
            ধ্রুব uint8_t * restrict act_col = (ধ্রুব uint8_t *) act_data;
            ভিন্ন * restrict dst_row = (ভিন্ন *) (dst->data + ie1 * dst->nb[1]);

            ধ্রুব uint32_t tile_size = htp_tensor_get_tile_size(src_w->type);
            ধ্রুব uint32_t aligned_tile_size = htp_tensor_get_tile_padded_size(src_w->type);
            ধ্রুব uint32_t n_k_tiles_w = src_w->ne[0] / htp_tensor_get_tile_k(src_w->type);
            ধ্রুব uint32_t tile_row_stride = n_k_tiles_w * tile_size;
            ধ্রুব uint32_t tile_row_transfer_size_aligned = n_k_tiles_w * aligned_tile_size;

            ধ্রুব uint32_t ct_start = src0_start_row / 32;
            ধ্রুব uint32_t ct_end   = (src0_end_row + 31) / 32;

            uint32_t ঠেলো_ct = ct_start;
            প্রতি (uint32_t d = 0; d < n_prefetch && ঠেলো_ct < ct_end; d++, ঠেলো_ct++) {
                dma_queue_ঠেলো(dma_q, dma_make_data(vtcm_src0_ptr + d * tile_row_transfer_size_aligned, src0_row + ঠেলো_ct * tile_row_stride),
                               aligned_tile_size, tile_size, tile_size, n_k_tiles_w);
            }

            প্রতি (uint32_t ct = ct_start; ct < ct_end; ct++) {
                ধ্রুব uint8_t * w_tile = (শূন্য *) dma_queue_বের_করো(dma_q).dst;

                পূর্ণাঙ্ক valid_rows = (পূর্ণাঙ্ক)dst->ne[0] - (পূর্ণাঙ্ক)(ct * 32);
                valid_rows = MIN(32, MAX(0, valid_rows));

                htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, ct);
                mmctx->vec_dot_32x1(act->ne[0], &dst_row[ct * 32], w_tile, act_col, valid_rows, রিক্ত);
                htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ct);

                যদি (ঠেলো_ct < ct_end) {
                    dma_queue_ঠেলো(dma_q, dma_make_data(w_tile, src0_row + ঠেলো_ct * tile_row_stride),
                                    aligned_tile_size, tile_size, tile_size, n_k_tiles_w);
                    ঠেলো_ct++;
                }
            }
        }
    }
}

স্থির শূন্য hvx_mm_id_nx_tiled(unsigned পূর্ণাঙ্ক nth, unsigned পূর্ণাঙ্ক ith, শূন্য * data) {
    সংরচনা htp_mm_context * mmctx = (সংরচনা htp_mm_context *) data;
    সংরচনা htp_ops_context * octx = mmctx->octx;
    ধ্রুব সংরচনা htp_mm_kernel_params * kparams = (ধ্রুব সংরচনা htp_mm_kernel_params *) octx->kernel_params;
    ধ্রুব uint32_t n_weights      = kparams->n_weights;
    ধ্রুব সংরচনা htp_tensor * restrict src0 = octx->src[0];
    ধ্রুব সংরচনা htp_tensor * restrict act  = octx->src[n_weights];
    ধ্রুব সংরচনা htp_tensor * restrict ids  = octx->src[n_weights + 1];


    সংরচনা htp_thread_trace * tr = &octx->ctx->trace[ith];

    ধ্রুব uint32_t n_prefetch = kparams->n_prefetch;
    প্রতিজ্ঞা(n_prefetch >= 2 && n_prefetch <= HTP_MM_MAX_PREFETCH && (n_prefetch & (n_prefetch - 1)) == 0);

    ধ্রুব uint32_t n_as = src0->ne[2];

    ধ্রুব uint32_t * matrix_row_counts = mmctx->matrix_row_counts;
    ধ্রুব সংরচনা mmid_row_mapping * matrix_rows = mmctx->matrix_rows;

    ধ্রুব size_t act_stride = mmctx->vtcm_act_stride;

    uint8_t * restrict vtcm_src0_ptr = mmctx->vtcm_src0 + mmctx->vtcm_src0_size_per_thread * ith;
    uint8_t * restrict act_data = mmctx->vtcm_act;

    প্রতি (uint32_t cur_a = 0; cur_a < n_as; ++cur_a) {
        ধ্রুব int32_t cne1 = matrix_row_counts[cur_a];
        যদি (cne1 == 0) অনুবর্ত;

        প্রতি (uint32_t p = 0; p < n_weights; ++p) {
            ধ্রুব সংরচনা htp_tensor * restrict src_w = octx->src[p];
            ধ্রুব সংরচনা htp_tensor * restrict dst   = octx->dsts[p];
            যদি (!src_w || !dst) অনুবর্ত;
            dma_queue * dma_q = octx->ctx->dma[ith];

            ধ্রুব uint32_t ne01 = src_w->ne[1];
            uint32_t start_row = 0;
            uint32_t end_row   = ne01;
            যদি (octx->ctx->mdev.গুনতি > 1) {
                ধ্রুব bool can_split = htp_tensor_can_row_partition(dst, পরিমাণ(ভিন্ন));
                ধ্রুব সংরচনা htp_tensor_mdev_range range = htp_tensor_mdev_partition(ne01, can_split ? 32 : 0, octx->ctx->mdev.idx, octx->ctx->mdev.গুনতি, &octx->ctx->mdev.গুনতি_div);
                start_row = range.start;
                end_row   = range.start + range.গুনতি;
            }

            ধ্রুব uint32_t nrows = end_row - start_row;
            uint32_t src0_nrows_per_thread = fastdiv(nrows + nth - 1, &octx->n_threads_div);
            src0_nrows_per_thread = hex_round_up(src0_nrows_per_thread, 32);

            ধ্রুব uint32_t src0_start_row = start_row + src0_nrows_per_thread * ith;
            ধ্রুব uint32_t src0_end_row   = MIN(src0_start_row + src0_nrows_per_thread, end_row);
            যদি (src0_start_row >= src0_end_row) অনুবর্ত;

            ধ্রুব dma_addr_t src0_row = src_w->data + cur_a * src_w->nb[2];

            ধ্রুব uint32_t tile_size = htp_tensor_get_tile_size(src_w->type);
            ধ্রুব uint32_t aligned_tile_size = htp_tensor_get_tile_padded_size(src_w->type);
            ধ্রুব uint32_t n_k_tiles_w = src_w->ne[0] / htp_tensor_get_tile_k(src_w->type);
            ধ্রুব uint32_t tile_row_stride = n_k_tiles_w * tile_size;
            ধ্রুব uint32_t tile_row_transfer_size_aligned = n_k_tiles_w * aligned_tile_size;

            ধ্রুব uint32_t ct_start = src0_start_row / 32;
            ধ্রুব uint32_t ct_end   = (src0_end_row + 31) / 32;

            uint32_t ঠেলো_ct = ct_start;
            প্রতি (uint32_t d = 0; d < n_prefetch && ঠেলো_ct < ct_end; d++, ঠেলো_ct++) {
                dma_queue_ঠেলো(dma_q, dma_make_data(vtcm_src0_ptr + d * tile_row_transfer_size_aligned, src0_row + ঠেলো_ct * tile_row_stride),
                               aligned_tile_size, tile_size, tile_size, n_k_tiles_w);
            }

            প্রতি (uint32_t ct = ct_start; ct < ct_end; ct++) {
                ধ্রুব uint8_t * w_tile = (শূন্য *) dma_queue_বের_করো(dma_q).dst;

                পূর্ণাঙ্ক valid_rows = (পূর্ণাঙ্ক)dst->ne[0] - (পূর্ণাঙ্ক)(ct * 32);
                valid_rows = MIN(32, MAX(0, valid_rows));

                htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, ct);
                প্রতি (uint32_t cid = 0; cid < (uint32_t) cne1; ++cid) {
                    সংরচনা mmid_row_mapping row_mapping = MMID_MATRIX_ROW(cur_a, cid);
                    ধ্রুব পূর্ণাঙ্ক rm1 = row_mapping.i1;
                    ধ্রুব পূর্ণাঙ্ক rm2 = row_mapping.i2;

                    ধ্রুব uint32_t ir1 = fastmodulo(rm1, act->ne[1], &mmctx->mm_div_ne11);
                    ধ্রুব uint8_t * restrict act_col = (ধ্রুব uint8_t *) (act_data + (ir1 + rm2 * act->ne[1]) * act_stride);
                    ভিন্ন * restrict dst_row = (ভিন্ন *) (dst->data + (rm1 * dst->nb[1] + rm2 * dst->nb[2]));

                    mmctx->vec_dot_32x1(act->ne[0], &dst_row[ct * 32], w_tile, act_col, valid_rows, রিক্ত);
                }
                htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ct);

                যদি (ঠেলো_ct < ct_end) {
                    dma_queue_ঠেলো(dma_q, dma_make_data(w_tile, src0_row + ঠেলো_ct * tile_row_stride),
                                    aligned_tile_size, tile_size, tile_size, n_k_tiles_w);
                    ঠেলো_ct++;
                }
            }
        }
    }
}

স্থির পূর্ণাঙ্ক hvx_mm_init_tiled_vec_dot(সংরচনা htp_mm_context * mmctx, গণনা htp_data_type type) {
    বিকল্প (type) {
        পক্ষ HTP_TYPE_Q4_0:
            mmctx->type         = "q4_0_tiled-f32";
            mmctx->vec_dot_32x1 = tiled_vec_dot_q4_0_32x1;
            mmctx->vec_dot_32x2 = tiled_vec_dot_q4_0_32x2;
            প্রত্যাগম 0;
        পক্ষ HTP_TYPE_Q4_1:
        পক্ষ HTP_TYPE_Q4_K:
            mmctx->type         = "q4_1_tiled-f32";
            mmctx->vec_dot_32x1 = tiled_vec_dot_q4_1_32x1;
            mmctx->vec_dot_32x2 = tiled_vec_dot_q4_1_32x2;
            প্রত্যাগম 0;
        পক্ষ HTP_TYPE_Q8_0:
            mmctx->type         = "q8_0_tiled-f32";
            mmctx->vec_dot_32x1 = tiled_vec_dot_q8_0_32x1;
            mmctx->vec_dot_32x2 = tiled_vec_dot_q8_0_32x2;
            প্রত্যাগম 0;
        পক্ষ HTP_TYPE_Q5_K:
            mmctx->type         = "q5_k_tiled-f32";
            mmctx->vec_dot_32x1 = tiled_vec_dot_q5_k_32x1;
            mmctx->vec_dot_32x2 = tiled_vec_dot_q5_k_32x2;
            প্রত্যাগম 0;
        পক্ষ HTP_TYPE_Q6_K:
            mmctx->type         = "q6_k_tiled-f32";
            mmctx->vec_dot_32x1 = tiled_vec_dot_q6_k_32x1;
            mmctx->vec_dot_32x2 = tiled_vec_dot_q6_k_32x2;
            প্রত্যাগম 0;
        পক্ষ HTP_TYPE_Q3_K:
            mmctx->type         = "q3_k_tiled-f32";
            mmctx->vec_dot_32x1 = tiled_vec_dot_q3_k_32x1;
            mmctx->vec_dot_32x2 = tiled_vec_dot_q3_k_32x2;
            প্রত্যাগম 0;
        পক্ষ HTP_TYPE_Q2_K:
            mmctx->type         = "q2_k_tiled-f32";
            mmctx->vec_dot_32x1 = tiled_vec_dot_q2_k_32x1;
            mmctx->vec_dot_32x2 = tiled_vec_dot_q2_k_32x2;
            প্রত্যাগম 0;
        পক্ষ HTP_TYPE_Q2_0:
            mmctx->type         = "q2_0_tiled-f32";
            mmctx->vec_dot_32x1 = tiled_vec_dot_q2_0_32x1;
            mmctx->vec_dot_32x2 = tiled_vec_dot_q2_0_32x2;
            প্রত্যাগম 0;
        পক্ষ HTP_TYPE_IQ4_NL:
            mmctx->type         = "iq4nl_tiled-f32";
            mmctx->vec_dot_32x1 = tiled_vec_dot_iq4nl_32x1;
            mmctx->vec_dot_32x2 = tiled_vec_dot_iq4nl_32x2;
            প্রত্যাগম 0;
        পক্ষ HTP_TYPE_MXFP4:
            mmctx->type         = "mxfp4_tiled-f32";
            mmctx->vec_dot_32x1 = tiled_vec_dot_mxfp4_32x1;
            mmctx->vec_dot_32x2 = tiled_vec_dot_mxfp4_32x2;
            প্রত্যাগম 0;
        পক্ষ HTP_TYPE_Q1_0:
            mmctx->type         = "q1_0_tiled-f32";
            mmctx->vec_dot_32x1 = tiled_vec_dot_q1_0_32x1;
            mmctx->vec_dot_32x2 = tiled_vec_dot_q1_0_32x2;
            প্রত্যাগম 0;
        পক্ষ HTP_TYPE_F16:
            mmctx->type         = "f16_tiled-f16";
            mmctx->vec_dot_32x1 = tiled_vec_dot_f16_32x1;
            mmctx->vec_dot_32x2 = tiled_vec_dot_f16_32x2;
            প্রত্যাগম 0;
        স্বভাব:
            প্রত্যাগম -1;
    }
}

স্থির পূর্ণাঙ্ক hvx_mm_matmul(সংরচনা htp_ops_context * octx) {
    htp_matmul_tensors_preamble;

    সংরচনা htp_thread_trace * tr = &octx->ctx->trace[0];
    htp_trace_event_start(tr, HTP_TRACE_EVT_INIT, 0);

    সংরচনা htp_mm_context mmctx_struct = {0};
    সংরচনা htp_mm_context * mmctx = &mmctx_struct;
    mmctx->octx = octx;
    mmctx->act = act;

    ধ্রুব সংরচনা htp_mm_kernel_params * kparams = (ধ্রুব সংরচনা htp_mm_kernel_params *) octx->kernel_params;

    ধ্রুব uint32_t src0_nrows = ne01;
    ধ্রুব uint32_t act_nrows  = ne11 * ne12 * ne13;
    mmctx->act_nrows = act_nrows;

    uint32_t src0_row_start = 0;
    uint32_t src0_row_end   = src0_nrows;

    যদি (octx->ctx->mdev.গুনতি > 1) {
        ধ্রুব bool can_split = htp_tensor_can_row_partition(dst, পরিমাণ(ভিন্ন));
        ধ্রুব সংরচনা htp_tensor_mdev_range range = htp_tensor_mdev_partition(src0_nrows, can_split ? 32 : 0, octx->ctx->mdev.idx, octx->ctx->mdev.গুনতি, &octx->ctx->mdev.গুনতি_div);
        src0_row_start = range.start;
        src0_row_end   = range.start + range.গুনতি;
    }

    যদি (src0_row_start >= src0_row_end) {
        প্রত্যাগম HTP_STATUS_OK;
    }

    ধ্রুব uint32_t nrows = src0_row_end - src0_row_start;
    mmctx->src0_row_start = src0_row_start;
    mmctx->src0_row_end   = src0_row_end;

    ধ্রুব bool is_tiled = htp_tensor_is_tiled(src0);

    // Compute src0_nrows_per_thread
    mmctx->src0_nrows_per_thread  = fastdiv(nrows + octx->n_threads - 1, &octx->n_threads_div);
    যদি (is_tiled) {
        mmctx->src0_nrows_per_thread = hex_round_up(mmctx->src0_nrows_per_thread, 32);
    } অন্যথা {
        mmctx->src0_nrows_per_thread += (mmctx->src0_nrows_per_thread & 1); // round up to even
    }

    ধ্রুব size_t src0_row_size = nb01;
    ধ্রুব size_t dst_row_size  = nb1;
    size_t       act_row_size  = nb11;

    ধ্রুব size_t src0_row_size_padded = hex_round_up(src0_row_size, 128);

    worker_callback_t quant_task_func;
    worker_callback_t matmul_job_func;
    uint32_t n_quant_tasks = 1;
    ধ্রুব bool is_batched = (ne12 > 1 || ne13 > 1 || ne02 > 1 || ne03 > 1);
    যদি (is_batched) {
        matmul_job_func = is_tiled ? hvx_mm_4d_tiled : hvx_mm_4d_flat;
    } অন্যথা যদি (act_nrows > 1) {
        matmul_job_func = is_tiled ? hvx_mm_2d_tiled : hvx_mm_2d_flat;
    } অন্যথা {
        matmul_job_func = is_tiled ? hvx_mv_2d_tiled : hvx_mv_2d_flat;
    }

    bool need_quant = true;

    বিকল্প (kparams->kernel_type) {
        পক্ষ HTP_MM_KERNEL_HVX_F16_F16_VTCM:
            quant_task_func        = (act->type == HTP_TYPE_F32) ? quantize_f32_f16 : quantize_f16_f16;
            need_quant             = (act->type == HTP_TYPE_F32);
            mmctx->type            = (act->type == HTP_TYPE_F32) ? "f32-f16" : "f16-f16";
            mmctx->vec_dot_1x1     = flat_vec_dot_f16_f16_aa_1x1;
            mmctx->vec_dot_2x1     = flat_vec_dot_f16_f16_aa_2x1;
            mmctx->vec_dot_2x2     = flat_vec_dot_f16_f16_aa_2x2;
            act_row_size           = kparams->act_row_size;
            ভঙ্গ;

        পক্ষ HTP_MM_KERNEL_HVX_F32_F32_VTCM:
            quant_task_func        = রিক্ত;
            need_quant             = false;
            mmctx->type            = "f32-f32";
            mmctx->vec_dot_1x1     = flat_vec_dot_f32_f32_aa_1x1;
            mmctx->vec_dot_2x1     = flat_vec_dot_f32_f32_aa_2x1;
            mmctx->vec_dot_2x2     = flat_vec_dot_f32_f32_aa_2x2;
            act_row_size           = kparams->act_row_size;
            ভঙ্গ;

        পক্ষ HTP_MM_KERNEL_HVX_BF16_F32_VTCM:
            quant_task_func        = রিক্ত;
            need_quant             = false;
            mmctx->type            = "bf16-f32";
            mmctx->vec_dot_1x1     = flat_vec_dot_bf16_f32_aa_1x1;
            mmctx->vec_dot_2x1     = flat_vec_dot_bf16_f32_aa_2x1;
            mmctx->vec_dot_2x2     = flat_vec_dot_bf16_f32_aa_2x2;
            act_row_size           = kparams->act_row_size;
            ভঙ্গ;

        পক্ষ HTP_MM_KERNEL_HVX_QUANT_BLOCK:
        পক্ষ HTP_MM_KERNEL_HVX_QUANT_ROW:
        স্বভাব: {
            যদি (hvx_mm_init_tiled_vec_dot(mmctx, src0->type) != 0) {
                প্রত্যাগম HTP_STATUS_NO_SUPPORT;
            }

            ধ্রুব uint32_t qk = QK_Q8_0_TILED;
            ধ্রুব uint32_t nb = (ne10 + qk - 1) / qk;
            ধ্রুব uint32_t total_nb = act_nrows * nb;

            যদি (act_nrows < octx->n_threads && !is_batched) {
                n_quant_tasks = MIN(total_nb, octx->n_threads);
                quant_task_func = htp_mm_act_quant_block_func(src0->type, act->type);
                প্রতি (uint32_t ith = 0; ith < n_quant_tasks; ++ith) {
                    uint32_t ib_first = (total_nb * ith) / n_quant_tasks;
                    uint32_t ib_last  = (total_nb * (ith + 1)) / n_quant_tasks;
                    mmctx->quant_ib_first[ith] = ib_first;
                    mmctx->quant_ib_last[ith]  = ib_last;
                    mmctx->quant_r[ith]        = ib_first / nb;
                    mmctx->quant_c[ith]        = ib_first % nb;
                }
            } অন্যথা {
                n_quant_tasks = MIN(act_nrows, octx->n_threads);
                quant_task_func = htp_mm_act_quant_row_func(src0->type, act->type);
            }
            act_row_size = kparams->act_row_size;
            ভঙ্গ;
        }
    }

    ধ্রুব uint32_t m_chunk = (kparams->m_chunk > 0 && (uint32_t) kparams->m_chunk < act_nrows)
                           ? (uint32_t) kparams->m_chunk : act_nrows;
    ধ্রুব uint32_t m_layout_rows = m_chunk;

    সংরচনা htp_mm_hvx_vtcm_layout L;
    htp_mm_hvx_vtcm_layout_build(&L, kparams->kernel_type, src0->type, ne10, m_layout_rows, octx->n_threads,
                                 dst_row_size, src0_row_size, act_row_size, src2 ? src2->nb[1] : 0, kparams->n_prefetch, false, false);

    যদি (kparams->kernel_type == HTP_MM_KERNEL_HVX_F16_F16_VTCM ||
        kparams->kernel_type == HTP_MM_KERNEL_HVX_F32_F32_VTCM ||
        kparams->kernel_type == HTP_MM_KERNEL_HVX_BF16_F32_VTCM ||
        kparams->kernel_type == HTP_MM_KERNEL_HVX_QUANT_ROW ||
        kparams->kernel_type == HTP_MM_KERNEL_HVX_QUANT_BLOCK) {
        mmctx->vtcm_act_size_per_thread = L.act_bytes;
    } অন্যথা {
        mmctx->vtcm_act_size_per_thread = fastdiv(L.act_bytes, &octx->n_threads_div);
    }

    mmctx->vtcm_src0_size_per_thread = fastdiv(L.src0_bytes, &octx->n_threads_div);
    mmctx->vtcm_dst_size_per_thread  = fastdiv(L.dst_bytes, &octx->n_threads_div);

    ধ্রুব size_t vtcm_size = L.total_bytes;

    FARF(উপরের_2, "matmul-%s : src0-vtcm-size %zu act-vtcm-size %zu dst-vtcm-size %zu (%zu)\n", mmctx->type,
         L.src0_bytes, L.act_bytes, L.dst_bytes, vtcm_size);

    FARF(উপরের_2, "matmul-%s : %ux%ux%ux%u * %ux%ux%ux%u-> %ux%ux%ux%u (0x%p, 0x%p, 0x%p)\n", mmctx->type, src0->ne[0],
         src0->ne[1], src0->ne[2], src0->ne[3], act->ne[0], act->ne[1], act->ne[2], act->ne[3], dst->ne[0],
         dst->ne[1], dst->ne[2], dst->ne[3], src0->data, act->data, dst->data);

    যদি (octx->ctx->vtcm_size < vtcm_size) {
        FARF(ERROR, "matmul-%s : current VTCM reservation %zu is too small, needed %zu\n", mmctx->type,
             octx->ctx->vtcm_size, vtcm_size);
        প্রত্যাগম HTP_STATUS_VTCM_TOO_SMALL;
    }

    uint8_t * ধ্রুব base = (uint8_t *) octx->ctx->vtcm_base;
    mmctx->vtcm_act      = VTCM_LAYOUT_PTR(uint8_t, base, L.off_act);
    mmctx->vtcm_src0     = VTCM_LAYOUT_PTR(uint8_t, base, L.off_src0);
    mmctx->vtcm_bias     = VTCM_LAYOUT_PTR(uint8_t, base, L.off_bias);
    mmctx->vtcm_dst      = VTCM_LAYOUT_PTR(uint8_t, base, L.off_dst);
    mmctx->vtcm_act_raw  = VTCM_LAYOUT_PTR(uint8_t, base, L.off_act_raw);

    mmctx->vtcm_src0_stride = src0_row_size_padded;
    mmctx->vtcm_act_stride  = act_row_size;
    যদি (kparams->kernel_type == HTP_MM_KERNEL_HVX_QUANT_BLOCK || kparams->kernel_type == HTP_MM_KERNEL_HVX_QUANT_ROW) {
        mmctx->vtcm_act_raw_stride = hex_round_up(ne10 * পরিমাণ(ভিন্ন), QK_Q8_0_TILED * পরিমাণ(ভিন্ন));
    } অন্যথা যদি (kparams->kernel_type == HTP_MM_KERNEL_HVX_F16_F16_VTCM) {
        mmctx->vtcm_act_raw_stride = hex_round_up(ne10 * পরিমাণ(ভিন্ন), 128);
    } অন্যথা {
        mmctx->vtcm_act_raw_stride = 0;
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_INIT, 0);

    যদি (kparams->m_chunk > 0 && (uint32_t) kparams->m_chunk < act_nrows) {
        প্রতি (uint32_t m_start = 0; m_start < act_nrows; m_start += m_chunk) {
            ধ্রুব uint32_t cur_m_rows = MIN(act_nrows - m_start, m_chunk);
            mmctx->cur_m_start = m_start;
            mmctx->cur_m_rows  = cur_m_rows;

            যদি (need_quant) {
                hvx_mm_transfer_act_dma(octx, kparams, act, mmctx->vtcm_act_raw, mmctx->vtcm_act_raw_stride, m_start, cur_m_rows);

                ধ্রুব uint32_t qk = QK_Q8_0_TILED;
                ধ্রুব uint32_t nb = (ne10 + qk - 1) / qk;
                ধ্রুব uint32_t total_nb = cur_m_rows * nb;
                uint32_t quant_tasks;
                work_queue_func_t q_func;
                যদি (cur_m_rows < octx->n_threads && (kparams->kernel_type == HTP_MM_KERNEL_HVX_QUANT_BLOCK || kparams->kernel_type == HTP_MM_KERNEL_HVX_QUANT_ROW)) {
                    quant_tasks = MIN(total_nb, octx->n_threads);
                    q_func = htp_mm_act_quant_block_func(src0->type, act->type);
                    প্রতি (uint32_t ith = 0; ith < quant_tasks; ++ith) {
                        uint32_t ib_first = (total_nb * ith) / quant_tasks;
                        uint32_t ib_last  = (total_nb * (ith + 1)) / quant_tasks;
                        mmctx->quant_ib_first[ith] = ib_first;
                        mmctx->quant_ib_last[ith]  = ib_last;
                        mmctx->quant_r[ith]        = ib_first / nb;
                        mmctx->quant_c[ith]        = ib_first % nb;
                    }
                } অন্যথা {
                    quant_tasks = MIN(cur_m_rows, octx->n_threads);
                    q_func = quant_task_func;
                    mmctx->n_quant_rows_per_thread = (cur_m_rows + quant_tasks - 1) / quant_tasks;
                }
                mmctx->n_quant_tasks = quant_tasks;
                work_queue_run(octx->ctx->work_queue, q_func, mmctx, quant_tasks);
            } অন্যথা {
                hvx_mm_transfer_act_dma(octx, kparams, act, mmctx->vtcm_act, mmctx->vtcm_act_stride, m_start, cur_m_rows);
            }

            work_queue_run(octx->ctx->work_queue, matmul_job_func, mmctx, octx->n_threads);
        }
    } অন্যথা {
        mmctx->cur_m_start = 0;
        mmctx->cur_m_rows  = act_nrows;

        যদি (need_quant) {
            hvx_mm_transfer_act_dma(octx, kparams, act, mmctx->vtcm_act_raw, mmctx->vtcm_act_raw_stride, 0, act_nrows);
            mmctx->n_quant_rows_per_thread = (act_nrows + n_quant_tasks - 1) / n_quant_tasks;
            mmctx->n_quant_tasks = n_quant_tasks;
            work_queue_run(octx->ctx->work_queue, quant_task_func, mmctx, n_quant_tasks);
        } অন্যথা {
            hvx_mm_transfer_act_dma(octx, kparams, act, mmctx->vtcm_act, mmctx->vtcm_act_stride, 0, act_nrows);
        }

        work_queue_run(octx->ctx->work_queue, matmul_job_func, mmctx, octx->n_threads);
    }

    প্রত্যাগম HTP_STATUS_OK;
}

স্থির শূন্য dequantize_tiled_worker_loop(unsigned পূর্ণাঙ্ক n, unsigned পূর্ণাঙ্ক i, শূন্য * data) {
    tiled_dequantize_state_t * state = (tiled_dequantize_state_t *) data;
    সংরচনা htp_thread_trace * tr = &state->traces[i];
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_W_DEQUANT, i);
    প্রতি (unsigned পূর্ণাঙ্ক task_id = i; task_id < (unsigned পূর্ণাঙ্ক) state->n_tasks; task_id += n) {
        পূর্ণাঙ্ক start = task_id * state->n_tiles_per_task;
        পূর্ণাঙ্ক end   = hex_smin(start + state->n_tiles_per_task, state->n_tot_tiles);
        state->task_fn(state, start, end);
    }
    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_W_DEQUANT, i);
}

স্থির শূন্য transfer_output_chunk_worker_fn(unsigned পূর্ণাঙ্ক n, unsigned পূর্ণাঙ্ক i, শূন্য *data) {
    output_transfer_task_state_t *st = (output_transfer_task_state_t *) data;

    সংরচনা htp_thread_trace * tr = &st->traces[i];

    পূর্ণাঙ্ক start_chunk_idx = i * st->n_chunks_per_task;
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_O_PROC, start_chunk_idx);

    প্রতি (unsigned পূর্ণাঙ্ক task_id = i; task_id < (unsigned পূর্ণাঙ্ক)st->n_tasks; task_id += n) {
        পূর্ণাঙ্ক    chunk_idx  = task_id * st->n_chunks_per_task;
        size_t chunk_size = hex_smin(st->n_tot_chunks - chunk_idx, st->n_chunks_per_task);

        ভিন্ন        *dst      = st->dst      + chunk_idx * st->dst_stride;
        ধ্রুব ভিন্ন  *src2     = st->src2     ? (st->src2     + chunk_idx * st->src2_stride) : রিক্ত;
        transfer_output_chunk_fp16_to_fp32(dst, src2, st->vtcm_src, chunk_idx, chunk_size, st->n_cols, st->dst_stride, st->src2_stride, st->dst_cols);
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_O_PROC, start_chunk_idx);
}

প্রকারনির্দেশ সংরচনা {
    সংরচনা htp_context            * ctx;
    সংরচনা htp_thread_trace       * traces;
    __fp16                        * dst;
    dma_addr_t                      act_dma_addr;
    ভিন্ন                         * vtcm_f32_act;
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

প্রকারনির্দেশ সংরচনা {
    সংরচনা htp_context            * ctx;
    সংরচনা htp_thread_trace       * traces;
    __fp16                        * dst;
    dma_addr_t                      act_dma_addr;
    ভিন্ন                         * vtcm_f32_act;
    uint32_t                        n_rows;
    uint32_t                        k_block;
    uint32_t                        k_stride;
    uint32_t                        k_valid;
    uint32_t                        n_col_chunks;
    সংরচনা fastdiv_মানসমূহ           n_threads_div;
    size_t                          vtcm_f32_act_bytes;
    uint32_t                        dma_step_rows;
    uint32_t                        dma_step_rows_shift;
} activation_transfer_col_chunk_state_t;

স্থির শূন্য transfer_activation_chunk_fp32_to_fp16_dma_pipelined_col_chunk(
        dma_queue *dma_q,
        __fp16 *restrict vtcm_dst,
        dma_addr_t act_dma_addr,
        uint32_t n_rows,
        uint32_t k_block,
        uint32_t k_stride,
        uint32_t k_chunk_valid,
        uint32_t c_first,
        uint32_t c_len,
        ভিন্ন *thread_f32_act,
        সংরচনা htp_thread_trace *tr,
        uint32_t dma_step_rows,
        uint32_t dma_step_rows_shift) {

    ধ্রুব uint32_t R = dma_step_rows;
    ধ্রুব uint32_t n_rows_padded = hex_align_up(n_rows, HTP_MM_HMX_TILE_N_ROWS);

    ধ্রুব uint32_t n_steps = n_rows_padded >> dma_step_rows_shift;

    // Push step 0
    যদি (n_steps > 0 && n_rows > 0) {
        uint32_t nrows_to_fetch = hex_smin(n_rows, R);
        dma_queue_ঠেলো(dma_q, dma_make_data(thread_f32_act, act_dma_addr + (size_t) c_first * পরিমাণ(ভিন্ন)),
                       c_len * পরিমাণ(ভিন্ন), k_stride * পরিমাণ(ভিন্ন), k_chunk_valid * পরিমাণ(ভিন্ন), nrows_to_fetch);
    }
    // Push step 1
    যদি (n_steps > 1) {
        uint32_t next_r = R * 1;
        যদি (next_r < n_rows) {
            uint32_t nrows_to_fetch = hex_smin(n_rows - next_r, R);
            ভিন্ন *next_buf = thread_f32_act + 1 * R * c_len;
            dma_queue_ঠেলো(dma_q, dma_make_data(next_buf, act_dma_addr + ((size_t) next_r * k_stride + c_first) * পরিমাণ(ভিন্ন)),
                           c_len * পরিমাণ(ভিন্ন), k_stride * পরিমাণ(ভিন্ন), k_chunk_valid * পরিমাণ(ভিন্ন), nrows_to_fetch);
        }
    }
    প্রতি (uint32_t s = 0; s < n_steps; ++s) {
        uint32_t r = s << dma_step_rows_shift;
        ভিন্ন *curr_buf = thread_f32_act;

        যদি (r < n_rows) {
            curr_buf = (ভিন্ন *) dma_queue_বের_করো(dma_q).dst;
        }

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_A_PREP, r);
        প্রতি (uint32_t p = 0; p < (R >> 1); ++p) {
            uint32_t row_idx = r + (p << 1);
            ভিন্ন *pair_buf = curr_buf + (p << 1) * c_len;
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
        যদি (next_r < n_rows) {
            uint32_t nrows_to_fetch = hex_smin(n_rows - next_r, R);
            dma_queue_ঠেলো(dma_q, dma_make_data(curr_buf, act_dma_addr + ((size_t) next_r * k_stride + c_first) * পরিমাণ(ভিন্ন)),
                           c_len * পরিমাণ(ভিন্ন), k_stride * পরিমাণ(ভিন্ন), k_chunk_valid * পরিমাণ(ভিন্ন), nrows_to_fetch);
        }
    }
}

স্থির শূন্য transfer_activation_chunk_col_chunk_worker_fn(unsigned পূর্ণাঙ্ক n, unsigned পূর্ণাঙ্ক i, শূন্য *data) {
    activation_transfer_col_chunk_state_t *st = (activation_transfer_col_chunk_state_t *) data;
    সংরচনা htp_thread_trace * tr = &st->traces[i];

    uint32_t n_blocks = st->k_block / 32;
    uint32_t b_first = fastdiv(n_blocks * i, &st->n_threads_div);
    uint32_t b_last  = fastdiv(n_blocks * (i + 1), &st->n_threads_div);
    uint32_t c_first = b_first * 32;
    uint32_t c_last = b_last * 32;
    uint32_t c_len = c_last - c_first;

    যদি (c_len == 0) {
        প্রত্যাগম;
    }

    uint32_t k_chunk_valid = 0;
    যদি (st->k_valid > c_first) {
        k_chunk_valid = hex_smin(st->k_valid, c_last) - c_first;
    }

    __fp16 *dst = st->dst;

    size_t thread_scratch_bytes = hex_align_down(fastdiv(st->vtcm_f32_act_bytes, &st->n_threads_div), 128);
    ভিন্ন *thread_f32_act = (ভিন্ন *)((বর্ণ *)st->vtcm_f32_act + i * thread_scratch_bytes);

    transfer_activation_chunk_fp32_to_fp16_dma_pipelined_col_chunk(
        st->ctx->dma[i], dst, st->act_dma_addr, st->n_rows, st->k_block, st->k_stride, k_chunk_valid,
        c_first, c_len, thread_f32_act, tr, st->dma_step_rows, st->dma_step_rows_shift
    );
}

স্থির শূন্য transfer_activation_chunk_to_fp16_dma_pipelined(
        dma_queue *dma_q,
        __fp16 *restrict vtcm_dst,
        dma_addr_t act_dma_addr,
        uint32_t n_rows,
        uint32_t k_block,
        uint32_t k_stride,
        uint32_t k_valid,
        uint8_t *thread_act,
        uint32_t act_elem_size,
        সংরচনা htp_thread_trace *tr,
        uint32_t dma_step_rows,
        uint32_t dma_step_rows_shift) {

    ধ্রুব uint32_t R = dma_step_rows;
    ধ্রুব uint32_t n_rows_padded = hex_align_up(n_rows, HTP_MM_HMX_TILE_N_ROWS);

    ধ্রুব uint32_t n_steps = n_rows_padded >> dma_step_rows_shift;

    ধ্রুব bool act_is_f16 = (act_elem_size == পরিমাণ(__fp16));
    ধ্রুব size_t row_bytes = (size_t) k_block * act_elem_size;  // staging row (DMA dst stride)
    ধ্রুব size_t src_row_bytes = (size_t) k_stride * act_elem_size;  // DDR row (DMA src stride)
    ধ্রুব size_t width_bytes = (size_t) k_valid * act_elem_size;

    // Push step 0
    যদি (n_steps > 0 && n_rows > 0) {
        uint32_t nrows_to_fetch = hex_smin(n_rows, R);
        dma_queue_ঠেলো(dma_q, dma_make_data(thread_act, act_dma_addr),
                       row_bytes, src_row_bytes, width_bytes, nrows_to_fetch);
    }
    // Push step 1 (if valid)
    যদি (n_steps > 1) {
        uint32_t next_r = R * 1;
        যদি (next_r < n_rows) {
            uint32_t nrows_to_fetch = hex_smin(n_rows - next_r, R);
            uint8_t *next_buf = thread_act + 1 * R * row_bytes;
            dma_queue_ঠেলো(dma_q, dma_make_data(next_buf, act_dma_addr + (size_t) next_r * src_row_bytes),
                           row_bytes, src_row_bytes, width_bytes, nrows_to_fetch);
        }
    }
    প্রতি (uint32_t s = 0; s < n_steps; ++s) {
        uint32_t r = s << dma_step_rows_shift;
        uint8_t *curr_buf = thread_act;

        যদি (r < n_rows) {
            curr_buf = (uint8_t *) dma_queue_বের_করো(dma_q).dst;
        }

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_A_PREP, r);
        // Two copies of the pair loop so the type is resolved once per step and the
        // row-pair kernels stay direct (inlinable) calls.
        যদি (act_is_f16) {
            প্রতি (uint32_t p = 0; p < (R >> 1); ++p) {
                uint32_t row_idx = r + (p << 1);
                ধ্রুব __fp16 *pair_buf = (ধ্রুব __fp16 *) (curr_buf + (p << 1) * row_bytes);
                bool r0_valid = ((row_idx + 0) < n_rows);
                bool r1_valid = ((row_idx + 1) < n_rows);

                transfer_activation_row_pair_f16_to_f16(vtcm_dst, pair_buf, pair_buf + k_block, row_idx, k_block, k_valid, r0_valid, r1_valid);
            }
        } অন্যথা {
            প্রতি (uint32_t p = 0; p < (R >> 1); ++p) {
                uint32_t row_idx = r + (p << 1);
                ধ্রুব ভিন্ন *pair_buf = (ধ্রুব ভিন্ন *) (curr_buf + (p << 1) * row_bytes);
                bool r0_valid = ((row_idx + 0) < n_rows);
                bool r1_valid = ((row_idx + 1) < n_rows);

                transfer_activation_row_pair_fp32_to_fp16(vtcm_dst, pair_buf, pair_buf + k_block, row_idx, k_block, k_valid, r0_valid, r1_valid);
            }
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_A_PREP, r);

        // Push step s + 2
        uint32_t next_s = s + 2;
        uint32_t next_r = next_s << dma_step_rows_shift;
        যদি (next_r < n_rows) {
            uint32_t nrows_to_fetch = hex_smin(n_rows - next_r, R);
            dma_queue_ঠেলো(dma_q, dma_make_data(curr_buf, act_dma_addr + (size_t) next_r * src_row_bytes),
                           row_bytes, src_row_bytes, width_bytes, nrows_to_fetch);
        }
    }
}

স্থির শূন্য transfer_activation_chunk_worker_fn(unsigned পূর্ণাঙ্ক n, unsigned পূর্ণাঙ্ক i, শূন্য *data) {
    activation_transfer_task_state_t *st = (activation_transfer_task_state_t *) data;

    সংরচনা htp_thread_trace * tr = &st->traces[i];

    প্রতি (unsigned পূর্ণাঙ্ক task_id = i; task_id < (unsigned পূর্ণাঙ্ক)st->n_tasks; task_id += n) {
        পূর্ণাঙ্ক    chunk_idx  = task_id * st->n_chunks_per_task;
        size_t chunk_size = hex_smin(st->n_tot_chunks - chunk_idx, st->n_chunks_per_task);

        __fp16      *dst = st->dst + chunk_idx * st->k_block;
        ধ্রুব dma_addr_t act_dma_addr = st->act_dma_addr + (size_t) chunk_idx * st->k_stride * st->act_elem_size;

        uint8_t *thread_act = (uint8_t *) st->vtcm_f32_act + i * st->vtcm_f32_act_bytes_per_thread;
        transfer_activation_chunk_to_fp16_dma_pipelined(
            st->ctx->dma[i], dst, act_dma_addr, chunk_size, st->k_block, st->k_stride, st->k_valid, thread_act, st->act_elem_size, tr, st->dma_step_rows, st->dma_step_rows_shift
        );
    }
}

প্রকারনির্দেশ সংরচনা {
    সংরচনা htp_thread_trace       * traces;
    ধ্রুব সংরচনা mmid_row_mapping * matrix_rows;
    __fp16                        * dst;
    ধ্রুব ভিন্ন                   * src;
    uint32_t                        n_tasks;
    uint32_t                        n_tot_chunks;
    uint32_t                        n_chunks_per_task;
    uint32_t                        k_block;
    uint32_t                        cur_a;
    uint32_t                        mapping_stride;
    uint32_t                        ne11;
    সংরচনা fastdiv_মানসমূহ           ne11_div;
    size_t                          nb11;
    size_t                          nb12;
    uint32_t                        start_row;
    uint32_t                        cne1;
    uint32_t                        k_valid;
} activation_transfer_gathered_task_state_t;

প্রকারনির্দেশ সংরচনা {
    সংরচনা htp_thread_trace       * traces;
    ধ্রুব সংরচনা mmid_row_mapping * matrix_rows;
    ধ্রুব __fp16                  * vtcm_src;
    ভিন্ন                         * dst;
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

স্থির শূন্য transfer_activation_chunk_gathered_worker_fn(unsigned পূর্ণাঙ্ক n, unsigned পূর্ণাঙ্ক i, শূন্য *data) {
    activation_transfer_gathered_task_state_t *st = data;
    সংরচনা htp_thread_trace * tr = &st->traces[i];
    পূর্ণাঙ্ক chunk_idx      = i;
    পূর্ণাঙ্ক chunk_size     = st->n_chunks_per_task;
    পূর্ণাঙ্ক vtcm_start_row = chunk_idx * chunk_size;
    পূর্ণাঙ্ক start_row      = st->start_row + vtcm_start_row;
    পূর্ণাঙ্ক n_rows         = hex_smin(st->cne1 - start_row, chunk_size);
    যদি (n_rows > 0) {
        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_A_PREP, chunk_idx);
        transfer_activation_chunk_fp32_to_fp16_gathered(
            st->dst, st->src, start_row, vtcm_start_row, n_rows, st->k_block,
            st->matrix_rows, st->cur_a, st->mapping_stride,
            st->ne11, &st->ne11_div, st->nb11, st->nb12, st->cne1, st->k_valid);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_A_PREP, chunk_idx);
    }
}

স্থির শূন্য transfer_activation_chunk_gathered_worker_flat_fn(unsigned পূর্ণাঙ্ক n, unsigned পূর্ণাঙ্ক i, শূন্য *data) {
    activation_transfer_gathered_task_state_t *st = data;
    সংরচনা htp_thread_trace * tr = &st->traces[i];
    পূর্ণাঙ্ক chunk_idx = i;
    পূর্ণাঙ্ক chunk_size = st->n_chunks_per_task;
    পূর্ণাঙ্ক vtcm_start_row = chunk_idx * chunk_size;
    পূর্ণাঙ্ক start_row = st->start_row + vtcm_start_row;
    পূর্ণাঙ্ক n_rows = hex_smin(st->cne1 - start_row, chunk_size);
    যদি (n_rows > 0) {
        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_A_PREP, chunk_idx);
        transfer_activation_chunk_fp32_to_fp16_gathered_flat(
            st->dst, st->src, start_row, vtcm_start_row, n_rows, st->k_block,
            st->matrix_rows, st->cur_a, st->mapping_stride,
            st->nb12, st->cne1, st->k_valid);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_A_PREP, chunk_idx);
    }
}

স্থির শূন্য transfer_output_chunk_scattered_worker_fn(unsigned পূর্ণাঙ্ক n, unsigned পূর্ণাঙ্ক i, শূন্য *data) {
    output_transfer_scattered_task_state_t *st = data;
    সংরচনা htp_thread_trace * tr = &st->traces[i];
    পূর্ণাঙ্ক chunk_idx = i;
    পূর্ণাঙ্ক chunk_size = st->n_chunks_per_task;
    পূর্ণাঙ্ক vtcm_start_row = chunk_idx * chunk_size;
    পূর্ণাঙ্ক start_row = st->start_row + vtcm_start_row;
    পূর্ণাঙ্ক n_rows = hex_smin(st->cne1 - start_row, chunk_size);
    যদি (n_rows > 0) {
        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_O_PROC, chunk_idx);
        transfer_output_chunk_fp16_to_fp32_scattered(
            st->dst, st->vtcm_src, start_row, vtcm_start_row, n_rows, st->n_cols,
            st->matrix_rows, st->cur_a, st->mapping_stride,
            st->dst_nb1, st->dst_nb2, st->cne1);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_O_PROC, chunk_idx);
    }
}

// --- HMX Dispatchers & Entry Points ---

স্থির অন্তঃস্থ tiled_dequant_task_fn_t hmx_get_dequant_worker_fn(পূর্ণাঙ্ক weight_type, bool is_tiled) {
    বিকল্প (weight_type) {
        পক্ষ HTP_TYPE_Q4_0:   প্রত্যাগম dequantize_tiled_weight_to_fp16_task_q4_0;
        পক্ষ HTP_TYPE_IQ4_NL: প্রত্যাগম dequantize_tiled_weight_to_fp16_task_iq4_nl;
        পক্ষ HTP_TYPE_Q4_1:
        পক্ষ HTP_TYPE_Q4_K:   প্রত্যাগম dequantize_tiled_weight_to_fp16_task_q4_1;
        পক্ষ HTP_TYPE_MXFP4:  প্রত্যাগম dequantize_tiled_weight_to_fp16_task_mxfp4;
        পক্ষ HTP_TYPE_Q8_0:   প্রত্যাগম dequantize_tiled_weight_to_fp16_task_q8_0;
        পক্ষ HTP_TYPE_Q5_K:   প্রত্যাগম dequantize_tiled_weight_to_fp16_task_q5_k;
        পক্ষ HTP_TYPE_Q6_K:   প্রত্যাগম dequantize_tiled_weight_to_fp16_task_q6_k;
        পক্ষ HTP_TYPE_Q3_K:   প্রত্যাগম dequantize_tiled_weight_to_fp16_task_q3_k;
        পক্ষ HTP_TYPE_Q2_K:   প্রত্যাগম dequantize_tiled_weight_to_fp16_task_q2_k;
        পক্ষ HTP_TYPE_Q1_0:   প্রত্যাগম dequantize_tiled_weight_to_fp16_task_q1_0;
        পক্ষ HTP_TYPE_Q2_0:   প্রত্যাগম dequantize_tiled_weight_to_fp16_task_q2_0;
        পক্ষ HTP_TYPE_F16:    প্রত্যাগম is_tiled ? রিক্ত : convert_f16_weight_to_fp16_tiles_task;
        পক্ষ HTP_TYPE_BF16:   প্রত্যাগম convert_bf16_weight_to_fp16_tiles_task;
        পক্ষ HTP_TYPE_F32:    প্রত্যাগম quantize_f32_weight_to_fp16_tiles_task;
        স্বভাব:              প্রত্যাগম রিক্ত;
    }
}

স্থির শূন্য dequantize_tiled_weight_chunk_to_fp16_tiles(
        সংরচনা htp_context *ctx, __fp16 *vtcm_dst,
        ধ্রুব শূন্য *weight_src_ddr,
        পূর্ণাঙ্ক n_cols, পূর্ণাঙ্ক k_block,
        size_t row_stride, পূর্ণাঙ্ক weight_type,
        পূর্ণাঙ্ক n_k_tiles, সংরচনা fastdiv_মানসমূহ n_k_tiles_div,
        tiled_dequant_task_fn_t dequant_worker_fn, পূর্ণাঙ্ক n_threads) {

    যদি (!dequant_worker_fn) {
        প্রত্যাগম;
    }

    প্রতিজ্ঞা(k_block % HTP_MM_HMX_TILE_N_COLS == 0);

    ধ্রুব size_t n_col_tiles = hmx_ceil_div(n_cols, HTP_MM_HMX_TILE_N_COLS);
    ধ্রুব uint32_t n_weight_tiles = k_block / htp_tensor_get_tile_k(weight_type);
    ধ্রুব size_t n_tot_tiles = n_col_tiles * n_weight_tiles;

    size_t n_tiles_per_task = (n_threads == 1) ? n_tot_tiles : hmx_ceil_div(n_tot_tiles, n_threads);

    tiled_dequantize_state_t state;
    state.n_tasks          = (n_tot_tiles + n_tiles_per_task - 1) / n_tiles_per_task;
    state.n_tot_tiles      = n_tot_tiles;
    state.n_tiles_per_task = n_tiles_per_task;
    state.dst              = vtcm_dst;
    state.src              = (ধ্রুব uint8_t *)weight_src_ddr;
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

    যদি (state.n_tasks == 1 || n_threads == 1) {
        dequantize_tiled_worker_loop(1, 0, &state);
    } অন্যথা {
        পূর্ণাঙ্ক n_tasks = hex_smin((পূর্ণাঙ্ক) state.n_tasks, n_threads);
        worker_pool_run_func(ctx->worker_pool, dequantize_tiled_worker_loop, &state, n_tasks);
    }
}

প্রকারনির্দেশ সংরচনা {
    সংরচনা htp_context      * ctx;
    সংরচনা htp_thread_trace * traces;
    ভিন্ন                   * dst;
    ধ্রুব __fp16            * vtcm_src;
    ধ্রুব ভিন্ন             * src2;
    uint32_t                  n_rows;
    uint32_t                  n_cols;
    uint32_t                  dst_stride;
    uint32_t                  src2_stride;
    uint32_t                  dst_cols;
    সংরচনা fastdiv_মানসমূহ     n_threads_div;
} output_transfer_col_chunk_state_t;

স্থির শূন্য transfer_output_chunk_col_chunk_worker_fn(unsigned পূর্ণাঙ্ক n, unsigned পূর্ণাঙ্ক i, শূন্য *data) {
    (শূন্য) n;
    output_transfer_col_chunk_state_t *st = (output_transfer_col_chunk_state_t *) data;
    সংরচনা htp_thread_trace * tr = &st->traces[i];

    // Round up: the last block is partial when N is not 32-aligned. Its pad
    // columns are dropped by the dst_cols clamp inside the store.
    uint32_t n_blocks = hmx_ceil_div(st->n_cols, 32);
    uint32_t b_first  = fastdiv(n_blocks * i, &st->n_threads_div);
    uint32_t b_last   = fastdiv(n_blocks * (i + 1), &st->n_threads_div);
    uint32_t c_first  = b_first * 32;
    uint32_t c_last   = b_last * 32;
    uint32_t c_len    = c_last - c_first;

    যদি (c_len == 0) প্রত্যাগম;

    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_O_PROC, c_first);

    ধ্রুব __fp16 *vtcm_src = st->vtcm_src + b_first * HTP_MM_HMX_TILE_N_ELMS;
    ধ্রুব ভিন্ন *src2      = st->src2 ? (st->src2 + c_first) : রিক্ত;
    ভিন্ন *dst             = st->dst + c_first;

    পূর্ণাঙ্ক chunk_dst_cols = (পূর্ণাঙ্ক)st->dst_cols - (পূর্ণাঙ্ক)c_first;
    যদি (chunk_dst_cols > 0) {
        transfer_output_chunk_fp16_to_fp32_col_chunk(
            dst, src2, vtcm_src, 0, st->n_rows, c_len, st->n_cols,
            st->dst_stride, st->src2_stride, (uint32_t)chunk_dst_cols
        );
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_O_PROC, c_first);
}

স্থির শূন্য transfer_output_chunk_threaded(সংরচনা htp_context *ctx, ভিন্ন *dst, ধ্রুব ভিন্ন *src2, ধ্রুব __fp16 *vtcm_src,
                                              পূর্ণাঙ্ক n_rows, পূর্ণাঙ্ক n_cols, পূর্ণাঙ্ক dst_stride, uint32_t src2_stride, পূর্ণাঙ্ক dst_cols, পূর্ণাঙ্ক n_threads) {
    যদি (n_rows <= 0) প্রত্যাগম;

    uint32_t n_blocks = hmx_ceil_div((uint32_t) n_cols, 32);
    যদি (n_threads > 1 && n_blocks >= (uint32_t)n_threads) {
        সংরচনা fastdiv_মানসমূহ n_threads_div = (n_threads == (পূর্ণাঙ্ক)ctx->n_threads) ? ctx->n_threads_div : init_fastdiv_মানসমূহ(n_threads);
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
        প্রত্যাগম;
    }

    size_t n_tot_chunks      = n_rows;
    size_t n_chunks_per_task = (n_threads == 1) ? n_tot_chunks : hmx_ceil_div(n_rows, n_threads);
    n_chunks_per_task        = hex_align_up(n_chunks_per_task, 2);

    পূর্ণাঙ্ক actual_threads = hmx_ceil_div(n_rows, n_chunks_per_task);

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

    যদি (actual_threads <= 1) {
        transfer_output_chunk_worker_fn(1, 0, &state);
    } অন্যথা {
        worker_pool_run_func(ctx->worker_pool, transfer_output_chunk_worker_fn, &state, actual_threads);
    }
}

সংরচনা activation_transfer_params {
    সংরচনা htp_context *          ctx;
    __fp16 *                      dst;
    dma_addr_t                    act_dma_addr;
    পূর্ণাঙ্ক                           n_rows;
    পূর্ণাঙ্ক                           k_block;
    পূর্ণাঙ্ক                           k_stride;
    পূর্ণাঙ্ক                           n_threads;
    ধ্রুব সংরচনা fastdiv_মানসমূহ * act_threads_div;
    ধ্রুব সংরচনা fastdiv_মানসমূহ * k_div;
    পূর্ণাঙ্ক                           k_valid;
    ভিন্ন *                       vtcm_f32_act;
    size_t                        vtcm_f32_act_bytes;
    uint32_t                      act_elem_size;  // 4=F32 src1, 2=F16 src1
};

স্থির শূন্য transfer_activation_chunk_threaded(ধ্রুব সংরচনা activation_transfer_params * params) {
    সংরচনা htp_context *          ctx                = params->ctx;
    __fp16 *                      dst                = params->dst;
    ধ্রুব dma_addr_t              act_dma_addr       = params->act_dma_addr;
    পূর্ণাঙ্ক                           n_rows             = params->n_rows;
    পূর্ণাঙ্ক                           k_block            = params->k_block;
    পূর্ণাঙ্ক                           k_stride           = params->k_stride;
    পূর্ণাঙ্ক                           n_threads          = params->n_threads;
    ধ্রুব সংরচনা fastdiv_মানসমূহ * act_threads_div    = params->act_threads_div;
    ধ্রুব সংরচনা fastdiv_মানসমূহ * k_div              = params->k_div;
    পূর্ণাঙ্ক                           k_valid            = params->k_valid;
    ভিন্ন *                       vtcm_f32_act       = params->vtcm_f32_act;
    size_t                        vtcm_f32_act_bytes = params->vtcm_f32_act_bytes;
    // element size of the activation rows (4 = F32, 2 = F16).
    ধ্রুব uint32_t                act_elem_size      = params->act_elem_size ? params->act_elem_size : (uint32_t) পরিমাণ(ভিন্ন);

    যদি (n_rows <= 0) {
        প্রত্যাগম;
    }

    ধ্রুব size_t n_tasks = (n_rows + 31) >> 5;
    যদি (act_elem_size == পরিমাণ(ভিন্ন) && n_threads > 1 && k_block > 32 && n_tasks < (size_t) n_threads) {
        // Calculate step rows parameters for column-chunked dma pipelining
        uint32_t dma_step_rows = 2;
        uint32_t dma_step_rows_shift = 1;
        যদি (vtcm_f32_act_bytes > 0) {
            size_t thread_scratch_bytes = hex_align_down(fastdiv(vtcm_f32_act_bytes, act_threads_div), 128);
            size_t thread_scratch_elements = thread_scratch_bytes / পরিমাণ(ভিন্ন);
            size_t dma_step_rows_max = fastdiv(thread_scratch_elements / 2, k_div);
            যদি (dma_step_rows_max >= 4) {
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
        প্রত্যাগম;
    }

    প্রতিজ্ঞা(k_block % HTP_MM_HMX_TILE_N_COLS == 0 && k_stride % HTP_MM_HMX_TILE_N_COLS == 0);

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
    যদি (state.vtcm_f32_act_bytes_per_thread > 0) {
        size_t thread_scratch_elements = state.vtcm_f32_act_bytes_per_thread / পরিমাণ(ভিন্ন);
        size_t dma_step_rows_max = fastdiv(thread_scratch_elements / 2, k_div);
        যদি (dma_step_rows_max >= 4) {
            dma_step_rows = 4;
            dma_step_rows_shift = 2;
        }
    }
    state.dma_step_rows      = dma_step_rows;
    state.dma_step_rows_shift = dma_step_rows_shift;

    পূর্ণাঙ্ক active_threads = hex_smin(n_threads, (পূর্ণাঙ্ক)state.n_tasks);
    যদি (state.n_tasks == 1 || n_threads == 1) {
        transfer_activation_chunk_worker_fn(1, 0, &state);
    } অন্যথা {
        worker_pool_run_func(ctx->worker_pool, transfer_activation_chunk_worker_fn, &state, active_threads);
    }
}
// --- Async HMX matmul job (for pipeline overlap) ---

প্রকারনির্দেশ সংরচনা {
    __fp16 *       output;
    ধ্রুব __fp16 * activation;
    ধ্রুব __fp16 * weight;
    ধ্রুব __fp16 * scales;
    uint32_t       n_row_tiles;
    uint32_t       n_col_tiles;
    uint32_t       n_dot_tiles;
} hmx_matmul_job_t;

স্থির শূন্য hmx_matmul_worker_fn(শূন্য * data) {
    hmx_matmul_job_t * job = (hmx_matmul_job_t *) data;
    FARF(উপরের_2, "hmx-mm-job: n_row_tiles %u n_col_tiles %u n_dot_tiles %u", job->n_row_tiles, job->n_col_tiles, job->n_dot_tiles);
    core_dot_chunk_fp16(job->output, job->activation, job->weight, job->scales, job->n_row_tiles, job->n_col_tiles, job->n_dot_tiles);
}

স্থির অন্তঃস্থ শূন্য hmx_matmul_job_init(hmx_matmul_job_t * job,
                                       __fp16 *           output,
                                       ধ্রুব __fp16 *     activation,
                                       ধ্রুব __fp16 *     weight,
                                       ধ্রুব __fp16 *     scales,
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

স্থির পূর্ণাঙ্ক hmx_mm_2d_f32(সংরচনা htp_context *ctx,
                                  dma_queue *weight_dma,
                                  ভিন্ন *restrict dst,
                                  dma_addr_t src2_addr,
                                  size_t src2_bytes,
                                  dma_addr_t act_dma_addr,
                                  dma_addr_t weight,
                                  পূর্ণাঙ্ক m, পূর্ণাঙ্ক k, পূর্ণাঙ্ক n,
                                  পূর্ণাঙ্ক act_stride,
                                  uint32_t act_elem_size,  // 4=F32 src1, 2=F16 src1
                                  পূর্ণাঙ্ক weight_stride,
                                  পূর্ণাঙ্ক weight_type,
                                  পূর্ণাঙ্ক k_valid,
                                  পূর্ণাঙ্ক dst_stride,
                                  uint32_t src2_stride,
                                  পূর্ণাঙ্ক dst_cols,
                                  পূর্ণাঙ্ক m_chunk,
                                  পূর্ণাঙ্ক n_chunk,
                                  পূর্ণাঙ্ক pipeline,
                                  পূর্ণাঙ্ক n_threads,
                                  পূর্ণাঙ্ক act_threads,
                                  ধ্রুব সংরচনা fastdiv_মানসমূহ * act_threads_div,
                                  ধ্রুব সংরচনা fastdiv_মানসমূহ * k_div,
                                  পূর্ণাঙ্ক tile_size,
                                  পূর্ণাঙ্ক aligned_tile_size,
                                  পূর্ণাঙ্ক vtcm_size) {
    সংরচনা htp_thread_trace * tr = &ctx->trace[0];
    htp_trace_event_start(tr, HTP_TRACE_EVT_INIT, 0);

    ধ্রুব bool is_tiled = (tile_size > 0);

    যদি (k % 32 != 0) {
        প্রত্যাগম -1;
    }
    যদি (!hex_is_aligned(dst, পরিমাণ(ভিন্ন))) { প্রত্যাগম -1; }

    size_t row_stride = htp_tensor_get_row_size(weight_type, k);
    যদি (row_stride == 0) {
        প্রত্যাগম -1;
    }

    tiled_dequant_task_fn_t dequant_worker_fn = hmx_get_dequant_worker_fn(weight_type, is_tiled);

    ধ্রুব পূর্ণাঙ্ক n_k_tiles = k / HTP_MM_HMX_TILE_N_COLS;
    ধ্রুব পূর্ণাঙ্ক n_weight_tiles = k / htp_tensor_get_tile_k(weight_type);
    ধ্রুব সংরচনা fastdiv_মানসমূহ n_k_tiles_div = init_fastdiv_মানসমূহ(n_k_tiles);

    ধ্রুব size_t vec_dot_size = k * পরিমাণ(__fp16);
    ধ্রুব size_t vtcm_budget  = ctx->vtcm_size;

    ধ্রুব uint32_t dma_dst_stride  = is_tiled ? aligned_tile_size : row_stride;
    ধ্রুব uint32_t dma_src_stride  = is_tiled ? tile_size : weight_stride;
    ধ্রুব uint32_t dma_width_bytes = is_tiled ? tile_size : row_stride;

    size_t m_chunk_n_rows = m_chunk;
    size_t n_chunk_n_cols = n_chunk;
    size_t vtcm_used      = vtcm_size;

    সংরচনা htp_mm_hmx_vtcm_layout L;
    htp_mm_hmx_vtcm_layout_build(&L, HTP_MM_KERNEL_HMX_2D, weight_type, k, m_chunk_n_rows, n_chunk_n_cols, 1, pipeline, act_threads, aligned_tile_size, src2_bytes);

    vtcm_used = L.total_bytes;
    যদি (vtcm_used > vtcm_budget) {
        FARF(ERROR, "hmx-mm-2d-precomputed: VTCM overflow: used %zu budget %zu, m %d k %d n %d mc %zu nc %zu",
             vtcm_used, vtcm_budget, m, k, n, m_chunk_n_rows, n_chunk_n_cols);
        প্রত্যাগম -1;
    }

    uint8_t * ধ্রুব base = (uint8_t *) ctx->vtcm_base;
    __fp16  *vtcm_weight_raw[2] = {
        VTCM_LAYOUT_PTR(__fp16, base, L.off_weight[0]),
        VTCM_LAYOUT_PTR_OPTIONAL(__fp16, base, L.off_weight[1], pipeline)
    };
    __fp16  *vtcm_f16_act    = VTCM_LAYOUT_PTR(__fp16, base, L.off_act);
    ভিন্ন   *vtcm_f32_act    = VTCM_LAYOUT_PTR(ভিন্ন, base, L.off_act_f32);
    __fp16  *vtcm_output     = VTCM_LAYOUT_PTR(__fp16, base, L.off_dst[0]);
    শূন্য    *vtcm_scratch0   = VTCM_LAYOUT_PTR(শূন্য, base, L.off_scratch[0]);
    শূন্য    *vtcm_scratch1   = VTCM_LAYOUT_PTR_OPTIONAL(শূন্য, base, L.off_scratch[1], pipeline);
    শূন্য    *vtcm_scratch2   = VTCM_LAYOUT_PTR_OPTIONAL(শূন্য, base, L.off_dst[1], pipeline);
    __fp16  *vtcm_scales     = VTCM_LAYOUT_PTR(__fp16, base, L.off_scales);

    hmx_init_column_scales(vtcm_scales, Q6_V_vsplat_R(0x3c00));  // scale: 1.0, bias: 0.0 in FP16

    ধ্রুব bool has_bias      = (src2_bytes > 0 && src2_addr != 0);
    ভিন্ন   *vtcm_bias       = VTCM_LAYOUT_PTR_OPTIONAL(ভিন্ন, base, L.off_bias, has_bias);
    যদি (has_bias) {
        dma_queue_ঠেলো(weight_dma, dma_make_data(vtcm_bias, src2_addr), hex_align_up(src2_bytes, 128), 0, src2_bytes, 1);
        dma_queue_বের_করো(weight_dma);
    }

    FARF(উপরের_2, "hmx-mm-2d: m %d k %d n %d wtype %d mc %zu nc %zu vtcm %zu/%zu",
         m, k, n, weight_type, m_chunk_n_rows, n_chunk_n_cols, vtcm_used, vtcm_budget);

    পূর্ণাঙ্ক n_chunk_cnt = hmx_ceil_div(n, n_chunk_n_cols);

    htp_trace_event_stop(tr, HTP_TRACE_EVT_INIT, 0);

    যদি (pipeline) {
        // --- Asynchronous Pipelined Loop ---
        hmx_matmul_job_t job_slots[2];  // persistent double-buffered job descriptors
        শূন্য *vtcm_raw_bufs[4] = {
            vtcm_weight_raw[0],
            vtcm_weight_raw[1],
            dequant_worker_fn ? vtcm_weight_raw[0] : vtcm_scratch0,
            dequant_worker_fn ? vtcm_weight_raw[1] : vtcm_scratch1,
        };
        শূন্য *vtcm_weight_bufs[4] = {
            dequant_worker_fn ? vtcm_scratch0 : vtcm_weight_raw[0],
            dequant_worker_fn ? vtcm_scratch1 : vtcm_weight_raw[1],
            vtcm_scratch0,
            vtcm_scratch1,
        };
        শূন্য *vtcm_output_bufs[2] = { vtcm_output, vtcm_scratch2 };

        প্রতি (size_t mr = 0; mr < m; mr += m_chunk_n_rows) {
            ধ্রুব size_t n_rows = hex_smin(m - mr, m_chunk_n_rows);

            সংরচনা activation_transfer_params act_params = {
                .ctx = ctx,
                .dst = vtcm_f16_act,
                .act_dma_addr = act_dma_addr + (size_t) mr * act_stride * act_elem_size,  // byte offset (F16/F32)
                .n_rows = (পূর্ণাঙ্ক) n_rows,
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
            ধ্রুব size_t   n_cols_A0 = hex_smin(n - 0 * n_chunk_n_cols, n_chunk_n_cols);
            ধ্রুব uint32_t height_A0 = is_tiled ? hmx_ceil_div(n_cols_A0, 32) * n_weight_tiles : n_cols_A0;
            dma_queue_ঠেলো(weight_dma, dma_make_data(vtcm_raw_bufs[0], weight),
                           dma_dst_stride, dma_src_stride, dma_width_bytes, height_A0);

            যদি (1 < n_chunk_cnt) {
                ধ্রুব size_t   n_cols_A1 = hex_smin(n - 1 * n_chunk_n_cols, n_chunk_n_cols);
                ধ্রুব uint32_t height_A1 = is_tiled ? hmx_ceil_div(n_cols_A1, 32) * n_weight_tiles : n_cols_A1;
                dma_queue_ঠেলো(weight_dma, dma_make_data(vtcm_raw_bufs[1], weight + n_chunk_n_cols * weight_stride),
                               dma_dst_stride, dma_src_stride, dma_width_bytes, height_A1);
            }

            // Main loop: pop A_i -> dequantize A_i -> push A_{i+2} -> submit C_i -> wait C_{i-1} and store D_{i-1}
            প্রতি (পূর্ণাঙ্ক i = 0; i < n_chunk_cnt; ++i) {
                ধ্রুব size_t nc    = i * n_chunk_n_cols;
                ধ্রুব size_t nc_p2 = nc + 2 * n_chunk_n_cols;

                ধ্রুব size_t n_cols    = hex_smin(n - nc, n_chunk_n_cols);
                ধ্রুব size_t n_cols_p2 = hex_smin(n - nc_p2, n_chunk_n_cols);

                // 1. pop A_i
                শূন্য * curr_raw = (শূন্য *) dma_queue_বের_করো(weight_dma).dst;

                // 2. dequantize A_i
                dequantize_tiled_weight_chunk_to_fp16_tiles(
                    ctx, vtcm_weight_bufs[i % 4], curr_raw,
                    n_cols, k, row_stride, weight_type,
                    n_k_tiles, n_k_tiles_div, dequant_worker_fn, n_threads);

                // 3. push A_{i+2} (if i+2 < n_chunk_cnt)
                যদি (i + 2 < n_chunk_cnt) {
                    ধ্রুব uint32_t height_p2 = is_tiled ? hmx_ceil_div(n_cols_p2, 32) * n_weight_tiles : n_cols_p2;
                    dma_queue_ঠেলো(weight_dma, dma_make_data(vtcm_raw_bufs[(i + 2) % 4], weight + nc_p2 * weight_stride),
                                   dma_dst_stride, dma_src_stride, dma_width_bytes, height_p2);
                }

                // 4. submit C_i
                hmx_matmul_job_init(&job_slots[i % 2], (__fp16 *) vtcm_output_bufs[i % 2],
                                    (__fp16 *) vtcm_f16_act, (__fp16 *) vtcm_weight_bufs[i % 4],
                                    vtcm_scales, hmx_ceil_div(n_rows, HTP_MM_HMX_TILE_N_ROWS),
                                    hmx_ceil_div(n_cols, HTP_MM_HMX_TILE_N_COLS), k / HTP_MM_HMX_TILE_N_ROWS);
                hmx_queue_ঠেলো(ctx->hmx_queue, hmx_queue_make_desc(hmx_matmul_worker_fn, &job_slots[i % 2]));

                // 5. wait C_{i-1} and store D_{i-1} (multi-thread HVX, parallel with C_i)
                যদি (i > 0) {
                    hmx_queue_বের_করো(ctx->hmx_queue);
                    ধ্রুব size_t nc_prev = (i - 1) * n_chunk_n_cols;
                    ধ্রুব size_t n_cols_prev = hex_smin(n - nc_prev, n_chunk_n_cols);
                    ভিন্ন *output_chunk = dst + (mr * dst_stride + nc_prev);
                    ধ্রুব ভিন্ন *bias_chunk = has_bias ? (vtcm_bias + mr * src2_stride + nc_prev) : রিক্ত;
                    পূর্ণাঙ্ক chunk_dst_cols = dst_cols - (পূর্ণাঙ্ক)nc_prev;
                    যদি (chunk_dst_cols > 0) {
                        transfer_output_chunk_threaded(ctx, output_chunk, bias_chunk, vtcm_output_bufs[(i - 1) % 2], n_rows, n_cols_prev, dst_stride, src2_stride, chunk_dst_cols, n_threads);
                    }
                }
            }

            // Epilogue: wait C_{last} and store D_{last}
            hmx_queue_বের_করো(ctx->hmx_queue);
            ধ্রুব size_t nc_last = (n_chunk_cnt - 1) * n_chunk_n_cols;
            ধ্রুব size_t n_cols_last = hex_smin(n - nc_last, n_chunk_n_cols);
            ভিন্ন *output_chunk = dst + (mr * dst_stride + nc_last);
            ধ্রুব ভিন্ন *bias_chunk = has_bias ? (vtcm_bias + mr * src2_stride + nc_last) : রিক্ত;
            পূর্ণাঙ্ক chunk_dst_cols = dst_cols - (পূর্ণাঙ্ক)nc_last;
            যদি (chunk_dst_cols > 0) {
                transfer_output_chunk_threaded(ctx, output_chunk, bias_chunk, vtcm_output_bufs[(n_chunk_cnt - 1) % 2], n_rows, n_cols_last, dst_stride, src2_stride, chunk_dst_cols, n_threads);
            }
        }
    } অন্যথা {
        // --- Synchronous loop (m <= 32 or fallback) ---
        hmx_matmul_job_t job;
        শূন্য *vtcm_raw_bufs[2] = { vtcm_weight_raw[0], dequant_worker_fn ? vtcm_weight_raw[0] : vtcm_scratch0 };
        শূন্য *vtcm_weight_bufs[2] = { dequant_worker_fn ? vtcm_scratch0 : vtcm_weight_raw[0], vtcm_scratch0 };

        প্রতি (size_t mr = 0; mr < m; mr += m_chunk_n_rows) {
            ধ্রুব size_t n_rows = hex_smin(m - mr, m_chunk_n_rows);

            সংরচনা activation_transfer_params act_params = {
                .ctx = ctx,
                .dst = vtcm_f16_act,
                .act_dma_addr = act_dma_addr + (size_t) mr * act_stride * act_elem_size,  // byte offset (F16/F32)
                .n_rows = (পূর্ণাঙ্ক) n_rows,
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
            যদি (n > 0) {
                ধ্রুব size_t n_cols = hex_smin(n, n_chunk_n_cols);
                ধ্রুব uint32_t height = is_tiled ? hmx_ceil_div(n_cols, 32) * n_weight_tiles : n_cols;
                dma_queue_ঠেলো(weight_dma, dma_make_data(vtcm_raw_bufs[0], weight), dma_dst_stride, dma_src_stride, dma_width_bytes, height);
            }

            প্রতি (পূর্ণাঙ্ক i = 0; i < n_chunk_cnt; ++i) {
                ধ্রুব size_t nc = (size_t) i * n_chunk_n_cols;
                ধ্রুব size_t n_cols = hex_smin(n - nc, n_chunk_n_cols);
                ধ্রুব size_t n_row_tiles = hmx_ceil_div(n_rows, HTP_MM_HMX_TILE_N_ROWS);
                ধ্রুব size_t n_col_tiles = hmx_ceil_div(n_cols, HTP_MM_HMX_TILE_N_COLS);

                // A: Wait for weight DMA
                শূন্য * curr_raw = (শূন্য *) dma_queue_বের_করো(weight_dma).dst;

                // B: Weight Dequantize (Threaded)
                dequantize_tiled_weight_chunk_to_fp16_tiles(
                    ctx, vtcm_weight_bufs[i % 2], curr_raw,
                    n_cols, k, row_stride, weight_type,
                    n_k_tiles, n_k_tiles_div, dequant_worker_fn, n_threads);

                // Start weight DMA for the next chunk early
                ধ্রুব size_t nc_next = nc + n_chunk_n_cols;
                যদি (nc_next < n) {
                    ধ্রুব size_t n_cols_next = hex_smin(n - nc_next, n_chunk_n_cols);
                    ধ্রুব uint32_t height_next = is_tiled ? hmx_ceil_div(n_cols_next, 32) * n_weight_tiles : n_cols_next;
                    dma_queue_ঠেলো(weight_dma, dma_make_data(vtcm_raw_bufs[(i + 1) % 2], weight + nc_next * weight_stride), dma_dst_stride, dma_src_stride, dma_width_bytes, height_next);
                }

                // C: HMX Compute (Queue-based)
                hmx_matmul_job_init(&job, vtcm_output, vtcm_f16_act, (__fp16 *) vtcm_weight_bufs[i % 2], vtcm_scales, n_row_tiles, n_col_tiles, k / HTP_MM_HMX_TILE_N_ROWS);
                hmx_queue_ঠেলো(ctx->hmx_queue, hmx_queue_make_desc(hmx_matmul_worker_fn, &job));
                hmx_queue_বের_করো(ctx->hmx_queue);

                // D: Output Store
                ভিন্ন *output_chunk = dst + (mr * dst_stride + nc);
                ধ্রুব ভিন্ন *bias_chunk = has_bias ? (vtcm_bias + mr * src2_stride + nc) : রিক্ত;
                পূর্ণাঙ্ক chunk_dst_cols = dst_cols - (পূর্ণাঙ্ক)nc;
                যদি (chunk_dst_cols > 0) {
                    transfer_output_chunk_threaded(ctx, output_chunk, bias_chunk, vtcm_output, n_rows, n_cols, dst_stride, src2_stride, chunk_dst_cols, n_threads);
                }
            }
        }
    }

    প্রত্যাগম 0;
}

স্থির পূর্ণাঙ্ক hmx_mm_nx_2d_f32(সংরচনা htp_ops_context * octx, ধ্রুব সংরচনা htp_mm_kernel_params * kparams) {
    সংরচনা htp_context * ctx = octx->ctx;
    সংরচনা htp_thread_trace * tr = &ctx->trace[0];
    htp_trace_event_start(tr, HTP_TRACE_EVT_INIT, 0);

    ধ্রুব uint32_t n_weights = kparams->n_weights;
    যদি (n_weights == 0 || n_weights > HTP_OP_MAX_OUTPUTS) {
        প্রত্যাগম HTP_STATUS_INVAL_PARAMS;
    }

    ধ্রুব সংরচনা htp_tensor * restrict src0 = octx->src[0];
    ধ্রুব সংরচনা htp_tensor * restrict act  = octx->src[n_weights];

    ধ্রুব পূর্ণাঙ্ক weight_type = (পূর্ণাঙ্ক) src0->type;
    ধ্রুব পূর্ণাঙ্ক k           = (পূর্ণাঙ্ক) act->ne[0];
    ধ্রুব পূর্ণাঙ্ক k_valid     = (পূর্ণাঙ্ক) act->ne[0];
    ধ্রুব পূর্ণাঙ্ক m           = (পূর্ণাঙ্ক) (act->ne[1] * act->ne[2] * act->ne[3]);
    ধ্রুব uint32_t act_elem_size = (act->type == HTP_TYPE_F16) ? পরিমাণ(__fp16) : পরিমাণ(ভিন্ন);
    ধ্রুব পূর্ণাঙ্ক act_stride  = (পূর্ণাঙ্ক) (act->nb[1] / act_elem_size);
    ধ্রুব dma_addr_t act_dma_addr = act->data;

    যদি (k % 32 != 0) { প্রত্যাগম HTP_STATUS_NO_SUPPORT; }

    size_t row_stride = htp_tensor_get_row_size(weight_type, k);
    যদি (row_stride == 0) {
        প্রত্যাগম HTP_STATUS_NO_SUPPORT;
    }

    ধ্রুব পূর্ণাঙ্ক n_k_tiles = k / HTP_MM_HMX_TILE_N_COLS;
    ধ্রুব পূর্ণাঙ্ক n_weight_tiles = k / htp_tensor_get_tile_k(weight_type);
    ধ্রুব সংরচনা fastdiv_মানসমূহ n_k_tiles_div = init_fastdiv_মানসমূহ(n_k_tiles);

    ধ্রুব পূর্ণাঙ্ক tile_size       = kparams->tile_size;
    ধ্রুব পূর্ণাঙ্ক aligned_tile_size = kparams->aligned_tile_size;

    ধ্রুব bool is_tiled       = (tile_size > 0);
    tiled_dequant_task_fn_t dequant_worker_fn = hmx_get_dequant_worker_fn(weight_type, is_tiled);
    ধ্রুব size_t vtcm_budget  = ctx->vtcm_size;

    ধ্রুব পূর্ণাঙ্ক m_chunk_n_rows  = kparams->m_chunk;
    ধ্রুব পূর্ণাঙ্ক n_chunk_n_cols  = kparams->n_chunk;
    ধ্রুব পূর্ণাঙ্ক pipeline        = kparams->pipeline;
    ধ্রুব পূর্ণাঙ্ক n_threads       = octx->n_threads;
    ধ্রুব পূর্ণাঙ্ক act_threads     = kparams->n_act_threads;
    ধ্রুব সংরচনা fastdiv_মানসমূহ * act_threads_div = &kparams->div_n_act_threads;
    ধ্রুব সংরচনা fastdiv_মানসমূহ * k_div           = &kparams->div_ne00_padded;

    ধ্রুব uint32_t dma_dst_stride  = is_tiled ? aligned_tile_size : row_stride;
    ধ্রুব uint32_t dma_width_bytes = is_tiled ? tile_size : row_stride;

    সংরচনা htp_mm_hmx_vtcm_layout L;
    htp_mm_hmx_vtcm_layout_build(&L, HTP_MM_KERNEL_HMX_2D, weight_type, k, m_chunk_n_rows, n_chunk_n_cols, 1, pipeline, act_threads, aligned_tile_size, 0);

    যদি (L.total_bytes > vtcm_budget) {
        FARF(ERROR, "hmx-mm-nx-2d: VTCM overflow: used %zu budget %zu, m %d k %d mc %d nc %d",
             L.total_bytes, vtcm_budget, m, k, m_chunk_n_rows, n_chunk_n_cols);
        প্রত্যাগম HTP_STATUS_VTCM_TOO_SMALL;
    }

    uint8_t * ধ্রুব base = (uint8_t *) ctx->vtcm_base;
    __fp16  *vtcm_weight_raw[2] = {
        VTCM_LAYOUT_PTR(__fp16, base, L.off_weight[0]),
        VTCM_LAYOUT_PTR_OPTIONAL(__fp16, base, L.off_weight[1], pipeline)
    };
    __fp16  *vtcm_f16_act    = VTCM_LAYOUT_PTR(__fp16, base, L.off_act);
    ভিন্ন   *vtcm_f32_act    = VTCM_LAYOUT_PTR(ভিন্ন, base, L.off_act_f32);
    __fp16  *vtcm_output     = VTCM_LAYOUT_PTR(__fp16, base, L.off_dst[0]);
    শূন্য    *vtcm_scratch0   = VTCM_LAYOUT_PTR(শূন্য, base, L.off_scratch[0]);
    শূন্য    *vtcm_scratch1   = VTCM_LAYOUT_PTR_OPTIONAL(শূন্য, base, L.off_scratch[1], pipeline);
    শূন্য    *vtcm_scratch2   = VTCM_LAYOUT_PTR_OPTIONAL(শূন্য, base, L.off_dst[1], pipeline);
    __fp16  *vtcm_scales     = VTCM_LAYOUT_PTR(__fp16, base, L.off_scales);

    hmx_init_column_scales(vtcm_scales, Q6_V_vsplat_R(0x3c00));  // scale: 1.0, bias: 0.0 in FP16

    পূর্ণাঙ্ক m_start = 0;
    পূর্ণাঙ্ক m_rows  = m;
    যদি (octx->ctx->mdev.গুনতি > 1) {
        bool can_split = htp_tensor_can_row_partition(act, act_elem_size);
        যদি (can_split) {
            প্রতি (uint32_t p = 0; p < n_weights; ++p) {
                ধ্রুব সংরচনা htp_tensor * restrict dst = octx->dsts[p];
                যদি (!htp_tensor_can_row_partition(dst, পরিমাণ(ভিন্ন))) {
                    can_split = false;
                    ভঙ্গ;
                }
            }
        }
        ধ্রুব সংরচনা htp_tensor_mdev_range range = htp_tensor_mdev_partition((uint32_t) m, can_split ? 1 : 0, octx->ctx->mdev.idx, octx->ctx->mdev.গুনতি, &octx->ctx->mdev.গুনতি_div);
        m_start = (পূর্ণাঙ্ক) range.start;
        m_rows  = (পূর্ণাঙ্ক) range.গুনতি;
    }

    যদি (m_rows == 0) {
        প্রত্যাগম HTP_STATUS_OK;
    }

    FARF(উপরের_2, "hmx-mm-nx-2d: n_weights %u m %d (%d..%d) k %d wtype %d mc %d nc %d vtcm %zu/%zu",
         n_weights, m, m_start, m_start + m_rows, k, weight_type, m_chunk_n_rows, n_chunk_n_cols, L.total_bytes, vtcm_budget);

    htp_trace_event_stop(tr, HTP_TRACE_EVT_INIT, 0);

    ধ্রুব size_t mr_end = (size_t)(m_start + m_rows);

    যদি (pipeline) {
        hmx_matmul_job_t job_slots[2];
        শূন্য *vtcm_raw_bufs[4] = {
            vtcm_weight_raw[0],
            vtcm_weight_raw[1],
            dequant_worker_fn ? vtcm_weight_raw[0] : vtcm_scratch0,
            dequant_worker_fn ? vtcm_weight_raw[1] : vtcm_scratch1,
        };
        শূন্য *vtcm_weight_bufs[4] = {
            dequant_worker_fn ? vtcm_scratch0 : vtcm_weight_raw[0],
            dequant_worker_fn ? vtcm_scratch1 : vtcm_weight_raw[1],
            vtcm_scratch0,
            vtcm_scratch1,
        };
        শূন্য *vtcm_output_bufs[2] = { vtcm_output, vtcm_scratch2 };

        প্রতি (size_t mr = (size_t) m_start; mr < mr_end; mr += m_chunk_n_rows) {
            ধ্রুব size_t n_rows = hex_smin(mr_end - mr, m_chunk_n_rows);

            সংরচনা activation_transfer_params act_params = {
                .ctx = ctx,
                .dst = vtcm_f16_act,
                .act_dma_addr = act_dma_addr + (size_t) mr * act_stride * act_elem_size,
                .n_rows = (পূর্ণাঙ্ক) n_rows,
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

            প্রতি (uint32_t p = 0; p < n_weights; p++) {
                ধ্রুব সংরচনা htp_tensor * restrict src_w = octx->src[p];
                ধ্রুব সংরচনা htp_tensor * restrict dst   = octx->dsts[p];
                যদি (!src_w || !dst) অনুবর্ত;

                ধ্রুব dma_addr_t weight       = src_w->data;
                dma_queue * weight_dma      = octx->ctx->dma[0];
                ভিন্ন * dst_ptr            = (ভিন্ন *) dst->data;
                ধ্রুব size_t n             = src_w->ne[1];
                যদি (n == 0) অনুবর্ত;
                ধ্রুব size_t weight_stride = src_w->nb[1];
                ধ্রুব size_t dst_stride    = dst->nb[1] / পরিমাণ(ভিন্ন);
                ধ্রুব পূর্ণাঙ্ক dst_cols         = (পূর্ণাঙ্ক) dst->ne[0];
                ধ্রুব পূর্ণাঙ্ক n_chunk_cnt      = hmx_ceil_div(n, n_chunk_n_cols);

                ধ্রুব uint32_t dma_src_stride = is_tiled ? tile_size : weight_stride;

                ধ্রুব size_t   n_cols_A0 = hex_smin(n - 0 * n_chunk_n_cols, n_chunk_n_cols);
                ধ্রুব uint32_t height_A0 = is_tiled ? hmx_ceil_div(n_cols_A0, 32) * n_weight_tiles : n_cols_A0;
                dma_queue_ঠেলো(weight_dma, dma_make_data(vtcm_raw_bufs[0], weight),
                               dma_dst_stride, dma_src_stride, dma_width_bytes, height_A0);

                যদি (1 < n_chunk_cnt) {
                    ধ্রুব size_t   n_cols_A1 = hex_smin(n - 1 * n_chunk_n_cols, n_chunk_n_cols);
                    ধ্রুব uint32_t height_A1 = is_tiled ? hmx_ceil_div(n_cols_A1, 32) * n_weight_tiles : n_cols_A1;
                    dma_queue_ঠেলো(weight_dma, dma_make_data(vtcm_raw_bufs[1], weight + n_chunk_n_cols * weight_stride),
                                   dma_dst_stride, dma_src_stride, dma_width_bytes, height_A1);
                }

                প্রতি (পূর্ণাঙ্ক i = 0; i < n_chunk_cnt; ++i) {
                    ধ্রুব size_t nc    = i * n_chunk_n_cols;
                    ধ্রুব size_t nc_p2 = nc + 2 * n_chunk_n_cols;

                    ধ্রুব size_t n_cols    = hex_smin(n - nc, n_chunk_n_cols);
                    ধ্রুব size_t n_cols_p2 = hex_smin(n - nc_p2, n_chunk_n_cols);

                    শূন্য * curr_raw = (শূন্য *) dma_queue_বের_করো(weight_dma).dst;

                    dequantize_tiled_weight_chunk_to_fp16_tiles(
                        ctx, vtcm_weight_bufs[i % 4], curr_raw,
                        n_cols, k, row_stride, weight_type,
                        n_k_tiles, n_k_tiles_div, dequant_worker_fn, n_threads);

                    যদি (i + 2 < n_chunk_cnt) {
                        ধ্রুব uint32_t height_p2 = is_tiled ? hmx_ceil_div(n_cols_p2, 32) * n_weight_tiles : n_cols_p2;
                        dma_queue_ঠেলো(weight_dma, dma_make_data(vtcm_raw_bufs[(i + 2) % 4], weight + nc_p2 * weight_stride),
                                       dma_dst_stride, dma_src_stride, dma_width_bytes, height_p2);
                    }

                    hmx_matmul_job_init(&job_slots[i % 2], (__fp16 *) vtcm_output_bufs[i % 2],
                                        (__fp16 *) vtcm_f16_act, (__fp16 *) vtcm_weight_bufs[i % 4],
                                        vtcm_scales, hmx_ceil_div(n_rows, HTP_MM_HMX_TILE_N_ROWS),
                                        hmx_ceil_div(n_cols, HTP_MM_HMX_TILE_N_COLS), k / HTP_MM_HMX_TILE_N_ROWS);
                    hmx_queue_ঠেলো(ctx->hmx_queue, hmx_queue_make_desc(hmx_matmul_worker_fn, &job_slots[i % 2]));

                    যদি (i > 0) {
                        hmx_queue_বের_করো(ctx->hmx_queue);
                        ধ্রুব size_t nc_prev = (i - 1) * n_chunk_n_cols;
                        ধ্রুব size_t n_cols_prev = hex_smin(n - nc_prev, n_chunk_n_cols);
                        ভিন্ন *output_chunk = dst_ptr + (mr * dst_stride + nc_prev);
                        পূর্ণাঙ্ক chunk_dst_cols = dst_cols - (পূর্ণাঙ্ক)nc_prev;
                        যদি (chunk_dst_cols > 0) {
                            transfer_output_chunk_threaded(ctx, output_chunk, রিক্ত, vtcm_output_bufs[(i - 1) % 2], n_rows, n_cols_prev, dst_stride, 0, chunk_dst_cols, n_threads);
                        }
                    }
                }

                hmx_queue_বের_করো(ctx->hmx_queue);
                ধ্রুব size_t nc_last = (n_chunk_cnt - 1) * n_chunk_n_cols;
                ধ্রুব size_t n_cols_last = hex_smin(n - nc_last, n_chunk_n_cols);
                ভিন্ন *output_chunk = dst_ptr + (mr * dst_stride + nc_last);
                পূর্ণাঙ্ক chunk_dst_cols = dst_cols - (পূর্ণাঙ্ক)nc_last;
                যদি (chunk_dst_cols > 0) {
                    transfer_output_chunk_threaded(ctx, output_chunk, রিক্ত, vtcm_output_bufs[(n_chunk_cnt - 1) % 2], n_rows, n_cols_last, dst_stride, 0, chunk_dst_cols, n_threads);
                }
            }
        }
    } অন্যথা {
        hmx_matmul_job_t job;
        শূন্য *vtcm_raw_bufs[2] = { vtcm_weight_raw[0], dequant_worker_fn ? vtcm_weight_raw[0] : vtcm_scratch0 };
        শূন্য *vtcm_weight_bufs[2] = { dequant_worker_fn ? vtcm_scratch0 : vtcm_weight_raw[0], vtcm_scratch0 };

        প্রতি (size_t mr = (size_t) m_start; mr < mr_end; mr += m_chunk_n_rows) {
            ধ্রুব size_t n_rows = hex_smin(mr_end - mr, m_chunk_n_rows);

            সংরচনা activation_transfer_params act_params = {
                .ctx = ctx,
                .dst = vtcm_f16_act,
                .act_dma_addr = act_dma_addr + (size_t) mr * act_stride * act_elem_size,
                .n_rows = (পূর্ণাঙ্ক) n_rows,
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

            প্রতি (uint32_t p = 0; p < n_weights; p++) {
                ধ্রুব সংরচনা htp_tensor * restrict src_w = octx->src[p];
                ধ্রুব সংরচনা htp_tensor * restrict dst   = octx->dsts[p];
                যদি (!src_w || !dst) অনুবর্ত;

                ধ্রুব dma_addr_t weight       = src_w->data;
                dma_queue * weight_dma      = octx->ctx->dma[0];
                ভিন্ন * dst_ptr            = (ভিন্ন *) dst->data;
                ধ্রুব size_t n             = src_w->ne[1];
                যদি (n == 0) অনুবর্ত;
                ধ্রুব size_t weight_stride = src_w->nb[1];
                ধ্রুব size_t dst_stride    = dst->nb[1] / পরিমাণ(ভিন্ন);
                ধ্রুব পূর্ণাঙ্ক dst_cols         = (পূর্ণাঙ্ক) dst->ne[0];
                ধ্রুব পূর্ণাঙ্ক n_chunk_cnt      = hmx_ceil_div(n, n_chunk_n_cols);

                ধ্রুব uint32_t dma_src_stride = is_tiled ? tile_size : weight_stride;

                যদি (n > 0) {
                    ধ্রুব size_t n_cols = hex_smin(n, n_chunk_n_cols);
                    ধ্রুব uint32_t height = is_tiled ? hmx_ceil_div(n_cols, 32) * n_weight_tiles : n_cols;
                    dma_queue_ঠেলো(weight_dma, dma_make_data(vtcm_raw_bufs[0], weight), dma_dst_stride, dma_src_stride, dma_width_bytes, height);
                }

                প্রতি (পূর্ণাঙ্ক i = 0; i < n_chunk_cnt; ++i) {
                    ধ্রুব size_t nc = (size_t) i * n_chunk_n_cols;
                    ধ্রুব size_t n_cols = hex_smin(n - nc, n_chunk_n_cols);
                    ধ্রুব size_t n_row_tiles = hmx_ceil_div(n_rows, HTP_MM_HMX_TILE_N_ROWS);
                    ধ্রুব size_t n_col_tiles = hmx_ceil_div(n_cols, HTP_MM_HMX_TILE_N_COLS);

                    শূন্য * curr_raw = (শূন্য *) dma_queue_বের_করো(weight_dma).dst;

                    dequantize_tiled_weight_chunk_to_fp16_tiles(
                        ctx, vtcm_weight_bufs[i % 2], curr_raw,
                        n_cols, k, row_stride, weight_type,
                        n_k_tiles, n_k_tiles_div, dequant_worker_fn, n_threads);

                    ধ্রুব size_t nc_next = nc + n_chunk_n_cols;
                    যদি (nc_next < n) {
                        ধ্রুব size_t n_cols_next = hex_smin(n - nc_next, n_chunk_n_cols);
                        ধ্রুব uint32_t height_next = is_tiled ? hmx_ceil_div(n_cols_next, 32) * n_weight_tiles : n_cols_next;
                        dma_queue_ঠেলো(weight_dma, dma_make_data(vtcm_raw_bufs[(i + 1) % 2], weight + nc_next * weight_stride), dma_dst_stride, dma_src_stride, dma_width_bytes, height_next);
                    }

                    hmx_matmul_job_init(&job, vtcm_output, vtcm_f16_act, (__fp16 *) vtcm_weight_bufs[i % 2], vtcm_scales, n_row_tiles, n_col_tiles, k / HTP_MM_HMX_TILE_N_ROWS);
                    hmx_queue_ঠেলো(ctx->hmx_queue, hmx_queue_make_desc(hmx_matmul_worker_fn, &job));
                    hmx_queue_বের_করো(ctx->hmx_queue);

                    ভিন্ন *output_chunk = dst_ptr + (mr * dst_stride + nc);
                    পূর্ণাঙ্ক chunk_dst_cols = dst_cols - (পূর্ণাঙ্ক)nc;
                    যদি (chunk_dst_cols > 0) {
                        transfer_output_chunk_threaded(ctx, output_chunk, রিক্ত, vtcm_output, n_rows, n_cols, dst_stride, 0, chunk_dst_cols, n_threads);
                    }
                }
            }
        }
    }

    প্রত্যাগম HTP_STATUS_OK;
}

স্থির অন্তঃস্থ dma_addr_t hmx_mm_weight_batch_data(ধ্রুব hmx_mm_batched_params_t *params,
                                                         পূর্ণাঙ্ক dst_b2, পূর্ণাঙ্ক dst_b3) {
    ধ্রুব size_t b2_idx = (params->r2 <= 1) ? (size_t) dst_b2 : (size_t) fastdiv((uint32_t) dst_b2, &params->div_r2);
    ধ্রুব size_t b3_idx = (params->r3 <= 1) ? (size_t) dst_b3 : (size_t) fastdiv((uint32_t) dst_b3, &params->div_r3);
    প্রত্যাগম params->weight + b2_idx * params->src0_nb2 + b3_idx * params->src0_nb3;
}

স্থির অন্তঃস্থ dma_addr_t hmx_mm_act_batch_addr(ধ্রুব hmx_mm_batched_params_t *params,
                                                পূর্ণাঙ্ক dst_b2, পূর্ণাঙ্ক dst_b3) {
    প্রত্যাগম params->act_dma_addr + dst_b2 * params->act_nb2 +
           dst_b3 * params->act_nb3;
}

স্থির অন্তঃস্থ ভিন্ন *hmx_mm_dst_batch_ptr(ধ্রুব hmx_mm_batched_params_t *params,
                                              পূর্ণাঙ্ক dst_b2, পূর্ণাঙ্ক dst_b3) {
    প্রত্যাগম (ভিন্ন *) ((uint8_t *) params->dst +
                      (size_t) dst_b2 * params->dst_nb2 +
                      (size_t) dst_b3 * params->dst_nb3);
}

স্থির পূর্ণাঙ্ক hmx_mm_batched_simple(সংরচনা htp_context *ctx,
                                 ধ্রুব hmx_mm_batched_params_t *params,
                                 পূর্ণাঙ্ক m_chunk, পূর্ণাঙ্ক n_chunk, পূর্ণাঙ্ক pipeline, পূর্ণাঙ্ক n_threads, পূর্ণাঙ্ক act_threads, পূর্ণাঙ্ক vtcm_size,
                                 ধ্রুব সংরচনা fastdiv_মানসমূহ * act_threads_div, ধ্রুব সংরচনা fastdiv_মানসমূহ * k_div) {
    পূর্ণাঙ্ক ret = 0;
    প্রতি (পূর্ণাঙ্ক b3 = 0; b3 < params->ne13 && ret == 0; ++b3) {
        প্রতি (পূর্ণাঙ্ক b2 = 0; b2 < params->ne12 && ret == 0; ++b2) {
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
    প্রত্যাগম ret;
}

স্থির পূর্ণাঙ্ক hmx_mm_batched(সংরচনা htp_context *ctx, ধ্রুব hmx_mm_batched_params_t *params,
                          পূর্ণাঙ্ক m_chunk, পূর্ণাঙ্ক n_chunk, পূর্ণাঙ্ক pipeline, পূর্ণাঙ্ক n_threads, পূর্ণাঙ্ক act_threads,
                          ধ্রুব সংরচনা fastdiv_মানসমূহ * act_threads_div,
                          ধ্রুব সংরচনা fastdiv_মানসমূহ * k_div,
                          পূর্ণাঙ্ক vtcm_size) {
    ধ্রুব size_t row_stride = htp_tensor_get_row_size(params->weight_type, params->k);
    যদি (row_stride == 0) { প্রত্যাগম -1; }
    যদি (params->act_stride < params->k || (params->tile_size == 0 && (size_t) params->weight_stride < row_stride) || params->dst_stride < params->dst_cols) { প্রত্যাগম -1; }
    যদি (params->ne02 <= 0 || params->ne03 <= 0 || params->ne12 <= 0 || params->ne13 <= 0) { প্রত্যাগম -1; }
    যদি (params->ne12 % params->ne02 != 0 || params->ne13 % params->ne03 != 0) { প্রত্যাগম -1; }
    // N (the weight row count) does not have to be 32-aligned:
    যদি (params->k % 32 != 0) { প্রত্যাগম -1; }
    যদি (!hex_is_aligned(params->dst, পরিমাণ(ভিন্ন))) { প্রত্যাগম -1; }

    ধ্রুব পূর্ণাঙ্ক group_size = params->r2;
    ধ্রুব size_t vtcm_budget  = ctx->vtcm_size;

    // Check if the precomputed parameters are grouped or simple.
    // If simple, or if group_size <= 1, we use simple fallback loop.
    // Grouped path is only valid for flat F16 if group_size > 1 and it fits within VTCM budget.
    bool run_grouped = (params->weight_type == HTP_TYPE_F16 && params->tile_size == 0 && group_size > 1 && (size_t)vtcm_size <= vtcm_budget);
    যদি (!run_grouped) {
        প্রত্যাগম hmx_mm_batched_simple(ctx, params, m_chunk, n_chunk, pipeline, n_threads, act_threads, vtcm_size, act_threads_div, k_div);
    }

    সংরচনা htp_thread_trace * tr = &ctx->trace[0];
    htp_trace_event_start(tr, HTP_TRACE_EVT_INIT, 0);

    size_t m_chunk_n_rows = m_chunk;
    size_t n_chunk_n_cols = n_chunk;

    সংরচনা htp_mm_hmx_vtcm_layout L;
    htp_mm_hmx_vtcm_layout_build(&L, HTP_MM_KERNEL_HMX_BATCHED, HTP_TYPE_F16, params->k, m_chunk_n_rows, n_chunk_n_cols, group_size, false, act_threads, 0, params->bias_bytes);

    যদি (L.total_bytes > vtcm_budget) {
        FARF(উপরের_2, "%s: grouped layout overflowed VTCM, falling back to simple batched loop", __func__);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_INIT, 0);
        প্রত্যাগম hmx_mm_batched_simple(ctx, params, m_chunk, n_chunk, pipeline, n_threads, act_threads, vtcm_size, act_threads_div, k_div);
    }

    uint8_t * ধ্রুব base = (uint8_t *) ctx->vtcm_base;
    __fp16  *vtcm_weight     = VTCM_LAYOUT_PTR(__fp16, base, L.off_weight[0]);
    __fp16  *vtcm_f16_act    = VTCM_LAYOUT_PTR(__fp16, base, L.off_act);
    __fp16  *vtcm_output     = VTCM_LAYOUT_PTR(__fp16, base, L.off_dst[0]);
    শূন্য    *vtcm_scratch0   = VTCM_LAYOUT_PTR(শূন্য, base, L.off_scratch[0]);
    শূন্য    *vtcm_scratch1   = VTCM_LAYOUT_PTR(শূন্য, base, L.off_scratch[1]);
    __fp16  *vtcm_scales     = VTCM_LAYOUT_PTR(__fp16, base, L.off_scales);
    ভিন্ন   *vtcm_f32_act    = VTCM_LAYOUT_PTR(ভিন্ন, base, L.off_act_f32);

    ধ্রুব bool has_bias      = (params->bias_bytes > 0 && params->bias_addr != 0);
    ভিন্ন   *vtcm_bias       = VTCM_LAYOUT_PTR_OPTIONAL(ভিন্ন, base, L.off_bias, has_bias);
    যদি (has_bias) {
        dma_queue_ঠেলো(params->weight_dma, dma_make_data(vtcm_bias, params->bias_addr), hex_align_up(params->bias_bytes, 128), 0, params->bias_bytes, 1);
        dma_queue_বের_করো(params->weight_dma);
    }

    hmx_init_column_scales(vtcm_scales, Q6_V_vsplat_R(0x3c00));  // scale: 1.0, bias: 0.0 in FP16

    FARF(উপরের_2, "%s: grouped path m=%d k=%d n=%d group=%d streams=%d mc=%zu nc=%zu vtcm=%zu/%zu",
            __func__, params->m, params->k, params->n, group_size, params->ne13,
            m_chunk_n_rows, n_chunk_n_cols,
            L.total_bytes, vtcm_budget);

    ধ্রুব size_t fp16_row_bytes   = (size_t) params->k * পরিমাণ(__fp16);
    ধ্রুব size_t weight_row_bytes = (size_t) params->weight_stride;

    htp_trace_event_stop(tr, HTP_TRACE_EVT_INIT, 0);

    hmx_matmul_job_t job;

    প্রতি (পূর্ণাঙ্ক b3 = 0; b3 < params->ne13; ++b3) {
        প্রতি (পূর্ণাঙ্ক b2_base = 0; b2_base < params->ne12; b2_base += group_size) {
            ধ্রুব dma_addr_t weight_group = hmx_mm_weight_batch_data(params, b2_base, b3);
            dma_queue * weight_dma = params->weight_dma;

            প্রতি (size_t mr = 0; mr < (size_t) params->m; mr += m_chunk_n_rows) {
                ধ্রুব size_t n_rows = hex_smin((size_t) params->m - mr, m_chunk_n_rows);
                ধ্রুব size_t n_row_tiles = hmx_ceil_div((পূর্ণাঙ্ক) n_rows, HTP_MM_HMX_TILE_N_ROWS);

                // Pre-load activations for all heads in the group (once per m_chunk).
                // When the source is strided (permuted Q), use 2D DMA to gather
                // contiguous rows into a VTCM scratch buffer first, then HVX
                // converts from the contiguous VTCM buffer.  This avoids L2 cache
                // thrashing from HVX loads at large strides.
                প্রতি (পূর্ণাঙ্ক g = 0; g < group_size; ++g) {
                    ধ্রুব dma_addr_t act_dma_addr = hmx_mm_act_batch_addr(params, b2_base + g, b3) +
                                                      (size_t) mr * params->act_stride * params->act_elem_size;
                    __fp16 *vtcm_act_g = vtcm_f16_act + (size_t) g * L.act_head_stride;
                    সংরচনা activation_transfer_params act_params = {
                        .ctx = ctx,
                        .dst = vtcm_act_g,
                        .act_dma_addr = act_dma_addr,
                        .n_rows = (পূর্ণাঙ্ক) n_rows,
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
                    ধ্রুব size_t n_cols_first = hex_smin((size_t) params->n, n_chunk_n_cols);
                    dma_queue_ঠেলো(weight_dma, dma_make_data(vtcm_scratch0, weight_group),
                                      fp16_row_bytes, weight_row_bytes, fp16_row_bytes, n_cols_first);
                }
                যদি (n_chunk_n_cols < (size_t) params->n) {
                    ধ্রুব size_t n_cols_second = hex_smin((size_t) params->n - n_chunk_n_cols, n_chunk_n_cols);
                    ধ্রুব dma_addr_t second_weight_chunk = weight_group + n_chunk_n_cols * weight_row_bytes;
                    dma_queue_ঠেলো(weight_dma, dma_make_data(vtcm_scratch1, second_weight_chunk),
                                      fp16_row_bytes, weight_row_bytes, fp16_row_bytes, n_cols_second);
                }

                প্রতি (size_t nc = 0; nc < (size_t) params->n; nc += n_chunk_n_cols) {
                    ধ্রুব size_t n_cols      = hex_smin((size_t) params->n - nc, n_chunk_n_cols);
                    ধ্রুব size_t n_col_tiles = hmx_ceil_div((পূর্ণাঙ্ক) n_cols, HTP_MM_HMX_TILE_N_COLS);

                    {
                        শূন্য * curr_raw = (শূন্য *) dma_queue_বের_করো(weight_dma).dst;

                        ধ্রুব size_t n_cols_tiled = hex_align_up(n_cols, HTP_MM_HMX_TILE_N_COLS);

                        hmx_interleave_rows_to_tiles(vtcm_weight, (ধ্রুব __fp16 *) curr_raw, (uint32_t) n_cols, params->k, params->k, 0, (uint32_t) n_cols_tiled);

                        ধ্রুব size_t nc_next = nc + n_chunk_n_cols * 2;
                        যদি (nc_next < (size_t) params->n) {
                            ধ্রুব size_t n_cols_next = hex_smin((size_t) params->n - nc_next, n_chunk_n_cols);
                            ধ্রুব dma_addr_t next_weight_chunk = weight_group + nc_next * weight_row_bytes;

                            dma_queue_ঠেলো(weight_dma, dma_make_data(curr_raw, next_weight_chunk),
                                              fp16_row_bytes, weight_row_bytes, fp16_row_bytes, n_cols_next);
                        }
                    }

                    // Reuse the interleaved weight for every q_head in this GQA group
                    প্রতি (পূর্ণাঙ্ক g = 0; g < group_size; ++g) {
                        {
                            ধ্রুব __fp16 * vtcm_act_g = vtcm_f16_act + (size_t) g * L.act_head_stride;
                            hmx_matmul_job_init(&job, vtcm_output, vtcm_act_g, vtcm_weight, vtcm_scales, n_row_tiles, n_col_tiles, params->k / 32);
                            hmx_queue_ঠেলো(ctx->hmx_queue, hmx_queue_make_desc(hmx_matmul_worker_fn, &job));
                            hmx_queue_বের_করো(ctx->hmx_queue);
                        }

                        {
                            ভিন্ন *output = hmx_mm_dst_batch_ptr(params, b2_base + g, b3) + mr * params->dst_stride + nc;
                            ধ্রুব ভিন্ন *bias_chunk = has_bias ? (vtcm_bias + mr * params->bias_stride + nc) : রিক্ত;
                            পূর্ণাঙ্ক chunk_dst_cols = params->dst_cols - (পূর্ণাঙ্ক)nc;
                            যদি (chunk_dst_cols > 0) {
                                transfer_output_chunk_threaded(ctx, output, bias_chunk, vtcm_output, (পূর্ণাঙ্ক) n_rows, (পূর্ণাঙ্ক) n_cols,
                                                               params->dst_stride, params->bias_stride, chunk_dst_cols, n_threads);
                            }
                        }
                    }
                }
            }
        }
    }

    প্রত্যাগম 0;
}

স্থির শূন্য transfer_activation_chunk_gathered_threaded(
            সংরচনা htp_context *ctx,
            __fp16 *dst,
            ধ্রুব ভিন্ন *src,
            পূর্ণাঙ্ক start_row,
            পূর্ণাঙ্ক n_rows,
            পূর্ণাঙ্ক k_block,
            ধ্রুব সংরচনা mmid_row_mapping *matrix_rows,
            পূর্ণাঙ্ক cur_a,
            পূর্ণাঙ্ক mapping_stride,
            পূর্ণাঙ্ক ne11,
            size_t nb11,
            size_t nb12,
            পূর্ণাঙ্ক cne1,
            পূর্ণাঙ্ক n_threads,
            পূর্ণাঙ্ক k_valid) {
    যদি (n_rows <= 0) প্রত্যাগম;
    পূর্ণাঙ্ক chunks_per_thread = hmx_ceil_div(n_rows, n_threads);
    chunks_per_thread = hex_align_up(chunks_per_thread, 2);

    পূর্ণাঙ্ক actual_threads = hmx_ceil_div(n_rows, chunks_per_thread);

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
        .ne11_div          = ne11 > 1 ? init_fastdiv_মানসমূহ(ne11) : (সংরচনা fastdiv_মানসমূহ){0, 0},
        .nb11              = nb11,
        .nb12              = nb12,
        .start_row         = start_row,
        .cne1              = cne1,
        .k_valid           = k_valid,
        .traces            = ctx->trace,
    };

    worker_callback_t worker_fn = ne11 == 1 ? transfer_activation_chunk_gathered_worker_flat_fn :
                                              transfer_activation_chunk_gathered_worker_fn;

    যদি (actual_threads <= 1) {
        worker_fn(1, 0, &state);
    } অন্যথা {
        worker_pool_run_func(ctx->worker_pool, worker_fn, &state, actual_threads);
    }
}

স্থির শূন্য transfer_output_chunk_scattered_threaded(
            সংরচনা htp_context *ctx,
            ভিন্ন *dst,
            ধ্রুব __fp16 *vtcm_src,
            পূর্ণাঙ্ক start_row,
            পূর্ণাঙ্ক n_rows,
            পূর্ণাঙ্ক n_cols,
            ধ্রুব সংরচনা mmid_row_mapping *matrix_rows,
            পূর্ণাঙ্ক cur_a,
            পূর্ণাঙ্ক mapping_stride,
            size_t dst_nb1,
            size_t dst_nb2,
            পূর্ণাঙ্ক cne1,
            পূর্ণাঙ্ক n_threads) {
    যদি (n_rows <= 0) প্রত্যাগম;
    পূর্ণাঙ্ক chunks_per_thread = hmx_ceil_div(n_rows, n_threads);
    chunks_per_thread = hex_align_up(chunks_per_thread, 2);

    পূর্ণাঙ্ক actual_threads = hmx_ceil_div(n_rows, chunks_per_thread);

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

    যদি (actual_threads <= 1) {
        transfer_output_chunk_scattered_worker_fn(1, 0, &state);
    } অন্যথা {
        worker_pool_run_func(ctx->worker_pool, transfer_output_chunk_scattered_worker_fn, &state, actual_threads);
    }
}

স্থির পূর্ণাঙ্ক hmx_mm_id_2d_f32(সংরচনা htp_context *ctx,
                                         dma_queue *weight_dma,
                                         ভিন্ন *restrict dst,
                                         ধ্রুব ভিন্ন *activation,
                                         dma_addr_t weight,
                                         পূর্ণাঙ্ক m, পূর্ণাঙ্ক k, পূর্ণাঙ্ক n,
                                         পূর্ণাঙ্ক k_valid,
                                         পূর্ণাঙ্ক ne11,
                                         size_t act_nb1, size_t act_nb2,
                                         size_t dst_nb1, size_t dst_nb2,
                                         পূর্ণাঙ্ক weight_stride,
                                         পূর্ণাঙ্ক weight_type,
                                         পূর্ণাঙ্ক tile_size,
                                         পূর্ণাঙ্ক aligned_tile_size,
                                         ধ্রুব সংরচনা mmid_row_mapping *matrix_rows,
                                         পূর্ণাঙ্ক cur_a,
                                         পূর্ণাঙ্ক mapping_stride,
                                         পূর্ণাঙ্ক m_start,
                                         পূর্ণাঙ্ক m_end,
                                         পূর্ণাঙ্ক n_threads) {
    সংরচনা htp_thread_trace * tr = &ctx->trace[0];
    htp_trace_event_start(tr, HTP_TRACE_EVT_INIT, 0);

    ধ্রুব পূর্ণাঙ্ক cne1 = m;
    ধ্রুব পূর্ণাঙ্ক m_core = m_end - m_start;
    ধ্রুব পূর্ণাঙ্ক m_core_padded = hex_align_up(m_core > 0 ? m_core : 1, 32);

    যদি (k % 32 != 0 || n % 32 != 0) { প্রত্যাগম -1; }
    যদি (!hex_is_aligned(dst, VLEN) || !hex_is_aligned(activation, VLEN)) { প্রত্যাগম -1; }

    size_t row_stride = htp_tensor_get_row_size(weight_type, k);
    যদি (row_stride == 0) {
        প্রত্যাগম -1;
    }

    ধ্রুব পূর্ণাঙ্ক n_k_tiles = k / HTP_MM_HMX_TILE_N_COLS;
    ধ্রুব uint32_t n_weight_tiles = k / htp_tensor_get_tile_k(weight_type);
    ধ্রুব সংরচনা fastdiv_মানসমূহ n_k_tiles_div = init_fastdiv_মানসমূহ(n_k_tiles);

    ধ্রুব size_t vec_dot_size = k * পরিমাণ(__fp16);
    ধ্রুব size_t vtcm_budget  = ctx->vtcm_size;
    size_t vtcm_used = 0;

    ধ্রুব bool is_tiled = (tile_size > 0);
    tiled_dequant_task_fn_t dequant_worker_fn = hmx_get_dequant_worker_fn(weight_type, is_tiled);

    ধ্রুব uint32_t dma_dst_stride  = is_tiled ? aligned_tile_size : row_stride;
    ধ্রুব uint32_t dma_src_stride  = is_tiled ? tile_size : weight_stride;
    ধ্রুব uint32_t dma_width_bytes = is_tiled ? tile_size : row_stride;

    ধ্রুব size_t qweight_row_stride = is_tiled ? (size_t)(n_weight_tiles * aligned_tile_size) / 32 : 0;
    ধ্রুব size_t weight_row_stride = is_tiled ? qweight_row_stride : row_stride;

    size_t size_per_n = 0, size_per_m = 0, size_per_mn = 0;
    htp_mm_hmx_get_2d_chunk_costs(weight_type, k, /*pipeline=*/false, aligned_tile_size,
                                  &size_per_n, &size_per_m, &size_per_mn);

    ধ্রুব size_t overhead = htp_mm_hmx_get_2d_overhead(/*pipeline=*/false, /*is_matmul_id=*/true);
    size_t m_chunk_n_rows = 0, n_chunk_n_cols = 0;
    যদি (htp_mm_hmx_compute_chunks(vtcm_budget, overhead, size_per_n, size_per_m, size_per_mn,
                           m_core_padded, n,
                           /*m_block_cost=*/(size_t) n * HTP_MM_HMX_COST_W_DEQUANT,
                           /*n_block_cost=*/(size_t) m_core_padded * HTP_MM_HMX_COST_A_CONVERT, &m_chunk_n_rows, &n_chunk_n_cols, &vtcm_used)) {
        FARF(ERROR, "hmx-mm-id-2d: VTCM too small : m %d k %d n %d budget %zu", m_core_padded, k, n, vtcm_budget);
        প্রত্যাগম -1;
    }

    ধ্রুব size_t weight_area_size = hex_align_up(n_chunk_n_cols * weight_row_stride, HTP_MM_HMX_TILE_SIZE);
    ধ্রুব size_t act_area_size    = hex_align_up(m_chunk_n_rows * vec_dot_size, HTP_MM_HMX_TILE_SIZE);
    ধ্রুব size_t output_area_size = hex_align_up(m_chunk_n_rows * n_chunk_n_cols * পরিমাণ(__fp16), HTP_MM_HMX_TILE_SIZE);

    size_t scratch0_size = hex_align_up(n_chunk_n_cols * vec_dot_size, HTP_MM_HMX_TILE_SIZE);

    uint8_t *vtcm_ptr      = (uint8_t *) ctx->vtcm_base;
    __fp16  *vtcm_weight   = weight_area_size ? (__fp16 *) vtcm_seq_alloc(&vtcm_ptr, weight_area_size) : রিক্ত;
    __fp16  *vtcm_f16_act  = (__fp16 *) vtcm_seq_alloc(&vtcm_ptr, act_area_size);
    __fp16  *vtcm_output   = (__fp16 *) vtcm_seq_alloc(&vtcm_ptr, output_area_size);
    শূন্য    *vtcm_scratch0 = vtcm_seq_alloc(&vtcm_ptr, scratch0_size);
    __fp16  *vtcm_scales   = (__fp16 *) vtcm_seq_alloc(&vtcm_ptr, 256);

    vtcm_used = vtcm_ptr - (uint8_t *) ctx->vtcm_base;
    যদি (vtcm_used > vtcm_budget) {
        FARF(ERROR, "hmx-mm-id-2d: VTCM overflow: used %zu budget %zu", vtcm_used, vtcm_budget);
        প্রত্যাগম -1;
    }

    hmx_init_column_scales(vtcm_scales, Q6_V_vsplat_R(0x3c00));

    htp_trace_event_stop(tr, HTP_TRACE_EVT_INIT, 0);

    hmx_matmul_job_t job;
    শূন্য *vtcm_raw_bufs[2] = { vtcm_weight, dequant_worker_fn ? vtcm_weight : vtcm_scratch0 };
    শূন্য *vtcm_weight_bufs[2] = { dequant_worker_fn ? vtcm_scratch0 : vtcm_weight, vtcm_scratch0 };
    ধ্রুব পূর্ণাঙ্ক n_chunk_cnt = hmx_ceil_div(n, n_chunk_n_cols);

    প্রতি (size_t mr = (size_t) m_start; mr < (size_t) m_end; mr += m_chunk_n_rows) {
        ধ্রুব size_t n_rows = hex_smin((size_t) m_end - mr, m_chunk_n_rows);
        ধ্রুব size_t n_row_tiles = hmx_ceil_div(n_rows, HTP_MM_HMX_TILE_N_ROWS);

        transfer_activation_chunk_gathered_threaded(
            ctx, vtcm_f16_act, activation, (পূর্ণাঙ্ক) mr, (পূর্ণাঙ্ক) n_rows, k,
            matrix_rows, cur_a, mapping_stride, ne11, act_nb1, act_nb2, cne1, n_threads, k_valid);

        // A0: Pre-fetch the first weight chunk (nc = 0)
        যদি (n > 0) {
            ধ্রুব size_t n_cols = hex_smin((size_t) n, n_chunk_n_cols);
            ধ্রুব uint32_t height = is_tiled ? hmx_ceil_div(n_cols, 32) * n_weight_tiles : n_cols;
            dma_queue_ঠেলো(weight_dma, dma_make_data(vtcm_raw_bufs[0], weight),
                           dma_dst_stride, dma_src_stride, dma_width_bytes, height);
        }

        প্রতি (পূর্ণাঙ্ক i = 0; i < n_chunk_cnt; ++i) {
            ধ্রুব size_t nc = (size_t) i * n_chunk_n_cols;
            ধ্রুব size_t n_cols = hex_smin((size_t) n - nc, n_chunk_n_cols);
            ধ্রুব size_t n_col_tiles = hmx_ceil_div(n_cols, HTP_MM_HMX_TILE_N_COLS);

            // A: Wait for weight DMA
            শূন্য * curr_raw = (শূন্য *) dma_queue_বের_করো(weight_dma).dst;

            // B: Weight Dequantize (Threaded)
            dequantize_tiled_weight_chunk_to_fp16_tiles(
                ctx, vtcm_weight_bufs[i % 2], curr_raw,
                n_cols, k, row_stride, weight_type,
                n_k_tiles, n_k_tiles_div, dequant_worker_fn, n_threads
            );

            // Start weight DMA for the next chunk early
            ধ্রুব size_t nc_next = nc + n_chunk_n_cols;
            যদি (nc_next < (size_t) n) {
                ধ্রুব size_t n_cols_next = hex_smin((size_t) n - nc_next, n_chunk_n_cols);
                ধ্রুব uint32_t height_next = is_tiled ? hmx_ceil_div(n_cols_next, 32) * n_weight_tiles : n_cols_next;
                dma_queue_ঠেলো(weight_dma, dma_make_data(vtcm_raw_bufs[(i + 1) % 2], weight + nc_next * weight_stride),
                               dma_dst_stride, dma_src_stride, dma_width_bytes, height_next);
            }

            // C: HMX Compute (Queue-based)
            hmx_matmul_job_init(&job, vtcm_output, vtcm_f16_act, (__fp16 *) vtcm_weight_bufs[i % 2], vtcm_scales, n_row_tiles, n_col_tiles, k / HTP_MM_HMX_TILE_N_ROWS);
            hmx_queue_ঠেলো(ctx->hmx_queue, hmx_queue_make_desc(hmx_matmul_worker_fn, &job));
            hmx_queue_বের_করো(ctx->hmx_queue);

            // D: Output Store
            transfer_output_chunk_scattered_threaded(
                ctx, dst + nc, vtcm_output, (পূর্ণাঙ্ক) mr, (পূর্ণাঙ্ক) n_rows, (পূর্ণাঙ্ক) n_cols,
                matrix_rows, cur_a, mapping_stride, dst_nb1, dst_nb2, cne1, n_threads);
        }
    }

    প্রত্যাগম 0;
}

// --- Dispatchers and Public Entry Points ---

স্থির পূর্ণাঙ্ক hmx_mm_op_matmul(সংরচনা htp_ops_context * octx, ধ্রুব সংরচনা htp_mm_kernel_params * kparams) {
    htp_matmul_tensors_preamble;

    পূর্ণাঙ্ক k = (পূর্ণাঙ্ক) src0->ne[0];
    পূর্ণাঙ্ক n = (পূর্ণাঙ্ক) src0->ne[1];
    ধ্রুব পূর্ণাঙ্ক m_total    = (পূর্ণাঙ্ক) act->ne[1];
    ধ্রুব uint32_t act_elem_size = (act->type == HTP_TYPE_F16) ? পরিমাণ(__fp16) : পরিমাণ(ভিন্ন);
    ধ্রুব পূর্ণাঙ্ক act_stride = (পূর্ণাঙ্ক) (act->nb[1] / act_elem_size);

    পূর্ণাঙ্ক m_start = 0;
    পূর্ণাঙ্ক m_rows  = m_total;
    যদি (octx->ctx->mdev.গুনতি > 1) {
        bool can_split = htp_tensor_can_row_partition(dst, পরিমাণ(ভিন্ন)) &&
                         htp_tensor_can_row_partition(act, act_elem_size);
        যদি (src2 && src2->ne[1] > 1 && !htp_tensor_can_row_partition(src2, পরিমাণ(ভিন্ন))) {
            can_split = false;
        }
        ধ্রুব সংরচনা htp_tensor_mdev_range range = htp_tensor_mdev_partition((uint32_t) m_total, can_split ? 1 : 0, octx->ctx->mdev.idx, octx->ctx->mdev.গুনতি, &octx->ctx->mdev.গুনতি_div);
        m_start = (পূর্ণাঙ্ক) range.start;
        m_rows  = (পূর্ণাঙ্ক) range.গুনতি;
    }

    যদি (m_rows == 0) {
        প্রত্যাগম HTP_STATUS_OK;
    }

    dma_addr_t src2_addr = 0;
    size_t src2_bytes = 0;
    uint32_t src2_stride = 0;
    size_t src2_nb2 = 0;
    size_t src2_nb3 = 0;
    যদি (src2) {
        src2_stride = (src2->ne[1] == 1) ? 0 : (uint32_t) (src2->nb[1] / পরিমাণ(ভিন্ন));
        src2_addr = src2->data + m_start * src2_stride * পরিমাণ(ভিন্ন);
        src2_bytes = (size_t) kparams->vtcm_bias_size;
        src2_nb2 = (src2->ne[2] == 1) ? 0 : src2->nb[2];
        src2_nb3 = (src2->ne[3] == 1) ? 0 : src2->nb[3];
    }

    ধ্রুব পূর্ণাঙ্ক dst_stride = (পূর্ণাঙ্ক)(dst->nb[1] / পরিমাণ(ভিন্ন));
    ভিন্ন       * dst_ptr = (ভিন্ন *) dst->data + m_start * dst_stride;
    // byte offset, act_stride is in activation elements (F16 or F32)
    ধ্রুব dma_addr_t act_addr = act->data + (size_t) m_start * act_stride * act_elem_size;

    পূর্ণাঙ্ক ret = -1;
    ধ্রুব পূর্ণাঙ্ক n_threads = kparams->n_threads;
    যদি (kparams->kernel_type == HTP_MM_KERNEL_HMX_BATCHED) {
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
            .weight_stride     = (পূর্ণাঙ্ক) src0->nb[1],
            .weight_type       = (পূর্ণাঙ্ক) src0->type,
            .dst_stride        = dst_stride,
            .dst_cols          = (পূর্ণাঙ্ক) dst->ne[0],
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
    } অন্যথা {
        ret = hmx_mm_2d_f32(
            octx->ctx, octx->ctx->dma[0], dst_ptr, src2_addr, src2_bytes,
            act_addr, src0->data,
            m_rows, k, n, act_stride, act_elem_size, (পূর্ণাঙ্ক) src0->nb[1], (পূর্ণাঙ্ক) src0->type, (পূর্ণাঙ্ক) act->ne[0],
            dst_stride, src2_stride, (পূর্ণাঙ্ক)dst->ne[0],
            kparams->m_chunk, kparams->n_chunk, kparams->pipeline, n_threads,
            kparams->n_act_threads,
            &kparams->div_n_act_threads,
            &kparams->div_ne00_padded,
            kparams->tile_size, kparams->aligned_tile_size, kparams->vtcm_size
        );
    }

    যদি (ret != 0) {
        FARF(ERROR, "HMX matmul failed (ret=%d)\n", ret);
        প্রত্যাগম HTP_STATUS_INTERNAL_ERR;
    }
    প্রত্যাগম HTP_STATUS_OK;
}

স্থির অন্তঃস্থ শূন্য htp_mm_tensor_collapse_rows(সংরচনা htp_tensor * c, ধ্রুব সংরচনা htp_tensor * t, uint32_t stride) {
    *c = *t;
    c->ne[1] = t->ne[1] * t->ne[2] * t->ne[3];
    c->ne[2] = 1;
    c->ne[3] = 1;
    c->nb[1] = stride;
    c->nb[2] = c->nb[1] * c->ne[1];
    c->nb[3] = c->nb[2];
}

স্থির পূর্ণাঙ্ক op_matmul_impl(সংরচনা htp_ops_context * octx) {
    ধ্রুব সংরচনা htp_mm_kernel_params * kparams = (ধ্রুব সংরচনা htp_mm_kernel_params *) octx->kernel_params;

    ধ্রুব পূর্ণাঙ্ক status = htp_mm_init_context(octx, kparams);
    যদি (status != HTP_STATUS_OK) {
        প্রত্যাগম status;
    }

    যদি (kparams->n_hmx) {
        প্রত্যাগম hmx_mm_op_matmul(octx, kparams);
    }

    প্রত্যাগম hvx_mm_matmul(octx);
}

পূর্ণাঙ্ক op_matmul(সংরচনা htp_ops_context * octx) {
    ধ্রুব সংরচনা htp_mm_kernel_params * kparams = (ধ্রুব সংরচনা htp_mm_kernel_params *) octx->kernel_params;

    যদি (kparams->collapse) {
        ধ্রুব সংরচনা htp_tensor * act  = octx->src[1];
        ধ্রুব সংরচনা htp_tensor * dst  = octx->dst;
        ধ্রুব uint32_t s_act = (act->ne[1] > 1) ? act->nb[1] : ((act->ne[2] > 1) ? act->nb[2] : act->nb[3]);
        ধ্রুব uint32_t sd    = (dst->ne[1] > 1) ? dst->nb[1] : ((dst->ne[2] > 1) ? dst->nb[2] : dst->nb[3]);
        সংরচনা htp_tensor act_collapsed, dst_collapsed;
        htp_mm_tensor_collapse_rows(&act_collapsed, act, s_act);
        htp_mm_tensor_collapse_rows(&dst_collapsed, dst, sd);
        octx->src[1] = &act_collapsed;
        octx->dst    = &dst_collapsed;

        সংরচনা htp_tensor src2_collapsed;
        ধ্রুব সংরচনা htp_tensor * src2 = octx->src[2];
        যদি (src2 && (src2->ne[1] * src2->ne[2] * src2->ne[3] > 1)) {
            ধ্রুব uint32_t s2 = (src2->ne[1] > 1) ? src2->nb[1] : ((src2->ne[2] > 1) ? src2->nb[2] : src2->nb[3]);
            htp_mm_tensor_collapse_rows(&src2_collapsed, src2, s2);
            octx->src[2] = &src2_collapsed;
        }

        ধ্রুব পূর্ণাঙ্ক status = op_matmul_impl(octx);
        octx->src[1] = act;
        octx->dst    = dst;
        যদি (src2) {
            octx->src[2] = src2;
        }
        প্রত্যাগম status;
    }

    প্রত্যাগম op_matmul_impl(octx);
}

স্থির পূর্ণাঙ্ক hmx_mm_op_matmul_id(
    সংরচনা htp_ops_context * octx,
    সংরচনা htp_mm_context * mmctx
) {
    ধ্রুব uint32_t *                matrix_row_counts = mmctx->matrix_row_counts;
    ধ্রুব সংরচনা mmid_row_mapping * matrix_rows       = mmctx->matrix_rows;
    htp_matmul_tensors_preamble;
    ধ্রুব সংরচনা htp_mm_kernel_params * kparams = (ধ্রুব সংরচনা htp_mm_kernel_params *) octx->kernel_params;
    ধ্রুব পূর্ণাঙ্ক n_ids = octx->src[2]->ne[0];
    ধ্রুব পূর্ণাঙ্ক n_as  = ne02;

    ধ্রুব bool mdev_split = (octx->ctx->mdev.গুনতি > 1) && htp_tensor_can_row_partition(dst, পরিমাণ(ভিন্ন));
    যদি (octx->ctx->mdev.গুনতি > 1 && !mdev_split && octx->ctx->mdev.idx > 0) {
        প্রত্যাগম HTP_STATUS_OK;
    }
    uint32_t n_active = 0;
    যদি (mdev_split) {
        প্রতি (uint32_t a = 0; a < (uint32_t) n_as; ++a) {
            যদি (matrix_row_counts[a] > 0) n_active++;
        }
    }
    ধ্রুব bool expert_split = mdev_split && (n_active >= octx->ctx->mdev.গুনতি);

    uint32_t target_dev = 0;
    প্রতি (uint32_t cur_a = 0; cur_a < (uint32_t) n_as; ++cur_a) {
        ধ্রুব int32_t cne1 = matrix_row_counts[cur_a];
        যদি (cne1 == 0) অনুবর্ত;

        ধ্রুব পূর্ণাঙ্ক m_padded = hex_align_up(cne1, 32);
        পূর্ণাঙ্ক m_start = 0, m_end = m_padded;
        যদি (expert_split) {
            ধ্রুব bool my_expert = (target_dev == octx->ctx->mdev.idx);
            যদি (++target_dev == octx->ctx->mdev.গুনতি) {
                target_dev = 0;
            }
            যদি (!my_expert) অনুবর্ত;
        } অন্যথা যদি (mdev_split) {
            ধ্রুব bool can_split = (uint32_t) cne1 >= octx->ctx->mdev.গুনতি;
            ধ্রুব সংরচনা htp_tensor_mdev_range range = htp_tensor_mdev_partition((uint32_t) m_padded, can_split ? 1 : 0, octx->ctx->mdev.idx, octx->ctx->mdev.গুনতি, &octx->ctx->mdev.গুনতি_div);
            m_start = (পূর্ণাঙ্ক) range.start;
            m_end   = (পূর্ণাঙ্ক) (range.start + range.গুনতি);
        }
        যদি (m_start >= m_end) অনুবর্ত;

        পূর্ণাঙ্ক ret = hmx_mm_id_2d_f32(octx->ctx, octx->ctx->dma[0], (ভিন্ন*) dst->data, (ভিন্ন*) act->data,
                                   src0->data + cur_a * nb02,
                                   cne1, ne00, ne01,
                                   ne10,
                                   ne11,
                                   nb11, nb12,
                                   nb1, nb2,
                                   (পূর্ণাঙ্ক) src0->nb[1], (পূর্ণাঙ্ক) src0->type,
                                   kparams->tile_size, kparams->aligned_tile_size,
                                   matrix_rows, cur_a, mmctx->mapping_stride,
                                   m_start, m_end, (পূর্ণাঙ্ক) octx->n_threads);
        যদি (ret != 0) {
            FARF(ERROR, "HMX matmul failed for expert %u, error %d\n", cur_a, ret);
            প্রত্যাগম HTP_STATUS_NO_SUPPORT;
        }
    }

    প্রত্যাগম HTP_STATUS_OK;
}

স্থির পূর্ণাঙ্ক hvx_mm_matmul_id(
    সংরচনা htp_ops_context * octx,
    সংরচনা htp_mm_context * mmctx,
    work_queue_func_t hvx_mmid_task_func
) {
    htp_matmul_tensors_preamble;
    ধ্রুব uint32_t src0_row_size_padded = mmctx->src0_row_size_padded;
    ধ্রুব uint32_t act_nrows            = mmctx->act_nrows;

    সংরচনা htp_thread_trace * tr = &octx->ctx->trace[0];
    htp_trace_event_start(tr, HTP_TRACE_EVT_INIT, 0);

    ধ্রুব সংরচনা htp_mm_kernel_params * kparams = (ধ্রুব সংরচনা htp_mm_kernel_params *) octx->kernel_params;
    ধ্রুব সংরচনা htp_tensor * restrict ids = octx->src[2];
    ধ্রুব size_t src0_row_size = nb01;

    ধ্রুব uint32_t qk = QK_Q8_0_TILED;
    ধ্রুব uint32_t nb = (ne10 + qk - 1) / qk;
    ধ্রুব uint32_t total_nb = act_nrows * nb;

    work_queue_func_t quant_task_func;
    uint32_t n_quant_tasks = 1;
    যদি (act_nrows < octx->n_threads) {
        n_quant_tasks = MIN(total_nb, octx->n_threads);
        quant_task_func = htp_mm_act_quant_block_func(src0->type, act->type);
        প্রতি (uint32_t ith = 0; ith < n_quant_tasks; ++ith) {
            uint32_t ib_first = (total_nb * ith) / n_quant_tasks;
            uint32_t ib_last  = (total_nb * (ith + 1)) / n_quant_tasks;
            mmctx->quant_ib_first[ith] = ib_first;
            mmctx->quant_ib_last[ith]  = ib_last;
            mmctx->quant_r[ith]        = ib_first / nb;
            mmctx->quant_c[ith]        = ib_first % nb;
        }
    } অন্যথা {
        n_quant_tasks = MIN(act_nrows, octx->n_threads);
        quant_task_func = htp_mm_act_quant_row_func(src0->type, act->type);
    }
    size_t act_row_size = htp_mm_tiled_act_row_size(src0->type, ne10);

    সংরচনা htp_mm_hvx_vtcm_layout L;
    htp_mm_hvx_vtcm_layout_build(&L, kparams->kernel_type, src0->type, ne10, act_nrows, octx->n_threads,
                                 0, src0_row_size, act_row_size, 0, kparams->n_prefetch, true, false);

    ধ্রুব size_t vtcm_size = L.total_bytes;

    FARF(উপরের_2, "matmul-id-%s : src0-spad-size %zu act-spad-size %zu bias-spad-size 0 dst-spad-size %zu (%zu)\n", mmctx->type,
         L.src0_bytes, L.act_bytes, L.dst_bytes, vtcm_size);

    FARF(উপরের_2, "matmul-id-%s : %ux%ux%ux%u * %ux%ux%ux%u (%ux%ux%ux%u) -> %ux%ux%ux%u (0x%p, 0x%p, 0x%p)\n", mmctx->type,
         src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3], act->ne[0], act->ne[1], act->ne[2], act->ne[3],
         ids->ne[0], ids->ne[1], ids->ne[2], ids->ne[3], dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3], src0->data,
         act->data, dst->data);

    // Make sure the reserved vtcm size is sufficient
    যদি (octx->ctx->vtcm_size < vtcm_size) {
        FARF(ERROR, "matmul-id-%s : current VTCM reservation %zu is too small, needed %zu\n", mmctx->type, octx->ctx->vtcm_size, vtcm_size);
        প্রত্যাগম HTP_STATUS_VTCM_TOO_SMALL;
    }

    uint8_t * ধ্রুব base = (uint8_t *) octx->ctx->vtcm_base;
    mmctx->vtcm_act      = VTCM_LAYOUT_PTR(uint8_t, base, L.off_act);
    mmctx->vtcm_src0     = VTCM_LAYOUT_PTR(uint8_t, base, L.off_src0);
    mmctx->vtcm_bias     = রিক্ত;
    mmctx->vtcm_dst      = VTCM_LAYOUT_PTR(uint8_t, base, L.off_dst);
    mmctx->vtcm_act_raw  = VTCM_LAYOUT_PTR(uint8_t, base, L.off_act_raw);

    mmctx->vtcm_src0_stride    = src0_row_size_padded;
    mmctx->vtcm_act_stride     = act_row_size;
    mmctx->vtcm_act_raw_stride = hex_round_up(ne10 * পরিমাণ(ভিন্ন), QK_Q8_0_TILED * পরিমাণ(ভিন্ন));

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

    প্রত্যাগম HTP_STATUS_OK;
}

স্থির পূর্ণাঙ্ক hmx_mm_op_matmul_id_nx(
    সংরচনা htp_ops_context * octx,
    সংরচনা htp_mm_context * mmctx
) {
    ধ্রুব uint32_t *                matrix_row_counts = mmctx->matrix_row_counts;
    ধ্রুব সংরচনা mmid_row_mapping * matrix_rows       = mmctx->matrix_rows;
    ধ্রুব সংরচনা htp_mm_kernel_params * kparams = (ধ্রুব সংরচনা htp_mm_kernel_params *) octx->kernel_params;
    ধ্রুব uint32_t n_weights = kparams->n_weights;
    ধ্রুব সংরচনা htp_tensor * restrict src0 = octx->src[0];
    ধ্রুব সংরচনা htp_tensor * restrict act  = octx->src[n_weights];
    ধ্রুব পূর্ণাঙ্ক n_as = src0->ne[2];

    bool mdev_split = (octx->ctx->mdev.গুনতি > 1);
    প্রতি (uint32_t p = 0; p < n_weights && mdev_split; ++p) {
        ধ্রুব সংরচনা htp_tensor * restrict dst = octx->dsts[p];
        mdev_split = htp_tensor_can_row_partition(dst, পরিমাণ(ভিন্ন));
    }
    যদি (octx->ctx->mdev.গুনতি > 1 && !mdev_split && octx->ctx->mdev.idx > 0) {
        প্রত্যাগম HTP_STATUS_OK;
    }
    uint32_t n_active = 0;
    যদি (mdev_split) {
        প্রতি (uint32_t a = 0; a < (uint32_t) n_as; ++a) {
            যদি (matrix_row_counts[a] > 0) n_active++;
        }
    }
    ধ্রুব bool expert_split = mdev_split && (n_active >= octx->ctx->mdev.গুনতি);

    uint32_t target_dev = 0;
    প্রতি (uint32_t cur_a = 0; cur_a < (uint32_t) n_as; ++cur_a) {
        ধ্রুব int32_t cne1 = matrix_row_counts[cur_a];
        যদি (cne1 == 0) অনুবর্ত;

        ধ্রুব পূর্ণাঙ্ক m_padded = hex_align_up(cne1, 32);
        পূর্ণাঙ্ক m_start = 0, m_end = m_padded;
        যদি (expert_split) {
            ধ্রুব bool my_expert = (target_dev == octx->ctx->mdev.idx);
            যদি (++target_dev == octx->ctx->mdev.গুনতি) {
                target_dev = 0;
            }
            যদি (!my_expert) অনুবর্ত;
        } অন্যথা যদি (mdev_split) {
            ধ্রুব bool can_split = (uint32_t) cne1 >= octx->ctx->mdev.গুনতি;
            ধ্রুব সংরচনা htp_tensor_mdev_range range = htp_tensor_mdev_partition((uint32_t) m_padded, can_split ? 1 : 0, octx->ctx->mdev.idx, octx->ctx->mdev.গুনতি, &octx->ctx->mdev.গুনতি_div);
            m_start = (পূর্ণাঙ্ক) range.start;
            m_end   = (পূর্ণাঙ্ক) (range.start + range.গুনতি);
        }
        যদি (m_start >= m_end) অনুবর্ত;

        প্রতি (uint32_t p = 0; p < n_weights; ++p) {
            ধ্রুব সংরচনা htp_tensor * restrict src_w = octx->src[p];
            ধ্রুব সংরচনা htp_tensor * restrict dst   = octx->dsts[p];
            যদি (!src_w || !dst) অনুবর্ত;

            পূর্ণাঙ্ক ret = hmx_mm_id_2d_f32(octx->ctx, octx->ctx->dma[0], (ভিন্ন*) dst->data, (ভিন্ন*) act->data,
                                       src_w->data + cur_a * src_w->nb[2],
                                       cne1, src_w->ne[0], src_w->ne[1],
                                       act->ne[0],
                                       act->ne[1],
                                       act->nb[1], act->nb[2],
                                       dst->nb[1], dst->nb[2],
                                       (পূর্ণাঙ্ক) src_w->nb[1], (পূর্ণাঙ্ক) src_w->type,
                                       kparams->tile_size, kparams->aligned_tile_size,
                                       matrix_rows, cur_a, mmctx->mapping_stride,
                                       m_start, m_end, (পূর্ণাঙ্ক) octx->n_threads);
            যদি (ret != 0) {
                FARF(ERROR, "HMX matmul ID NX failed for expert %u weight %u, error %d\n", cur_a, p, ret);
                প্রত্যাগম HTP_STATUS_NO_SUPPORT;
            }
        }
    }

    প্রত্যাগম HTP_STATUS_OK;
}

স্থির পূর্ণাঙ্ক hvx_mm_matmul_id_nx(
    সংরচনা htp_ops_context * octx,
    সংরচনা htp_mm_context * mmctx,
    work_queue_func_t hvx_mmid_task_func
) {
    ধ্রুব uint32_t src0_row_size_padded = mmctx->src0_row_size_padded;
    ধ্রুব uint32_t act_nrows            = mmctx->act_nrows;

    সংরচনা htp_thread_trace * tr = &octx->ctx->trace[0];
    htp_trace_event_start(tr, HTP_TRACE_EVT_INIT, 0);

    ধ্রুব সংরচনা htp_mm_kernel_params * kparams = (ধ্রুব সংরচনা htp_mm_kernel_params *) octx->kernel_params;
    ধ্রুব uint32_t n_weights = kparams->n_weights;
    ধ্রুব সংরচনা htp_tensor * restrict src0 = octx->src[0];
    ধ্রুব সংরচনা htp_tensor * restrict act  = octx->src[n_weights];
    ধ্রুব সংরচনা htp_tensor * restrict ids  = octx->src[n_weights + 1];
    ধ্রুব size_t src0_row_size = src0->nb[1];

    ধ্রুব uint32_t qk = QK_Q8_0_TILED;
    ধ্রুব uint32_t nb = (act->ne[0] + qk - 1) / qk;
    ধ্রুব uint32_t total_nb = act_nrows * nb;

    work_queue_func_t quant_task_func;
    uint32_t n_quant_tasks = 1;
    যদি (act_nrows < octx->n_threads) {
        n_quant_tasks = MIN(total_nb, octx->n_threads);
        quant_task_func = htp_mm_act_quant_block_func(src0->type, act->type);
        প্রতি (uint32_t ith = 0; ith < n_quant_tasks; ++ith) {
            uint32_t ib_first = (total_nb * ith) / n_quant_tasks;
            uint32_t ib_last  = (total_nb * (ith + 1)) / n_quant_tasks;
            mmctx->quant_ib_first[ith] = ib_first;
            mmctx->quant_ib_last[ith]  = ib_last;
            mmctx->quant_r[ith]        = ib_first / nb;
            mmctx->quant_c[ith]        = ib_first % nb;
        }
    } অন্যথা {
        n_quant_tasks = MIN(act_nrows, octx->n_threads);
        quant_task_func = htp_mm_act_quant_row_func(src0->type, act->type);
    }
    size_t act_row_size = htp_mm_tiled_act_row_size(src0->type, act->ne[0]);

    সংরচনা htp_mm_hvx_vtcm_layout L;
    htp_mm_hvx_vtcm_layout_build(&L, kparams->kernel_type, src0->type, act->ne[0], act_nrows, octx->n_threads,
                                 0, src0_row_size, act_row_size, 0, kparams->n_prefetch, true, false);

    ধ্রুব size_t vtcm_size = L.total_bytes;

    যদি (octx->ctx->vtcm_size < vtcm_size) {
        FARF(ERROR, "matmul-id-nx: current VTCM reservation %zu is too small, needed %zu\n",
             octx->ctx->vtcm_size, vtcm_size);
        প্রত্যাগম HTP_STATUS_VTCM_TOO_SMALL;
    }

    uint8_t * ধ্রুব base = (uint8_t *) octx->ctx->vtcm_base;
    mmctx->vtcm_src0     = VTCM_LAYOUT_PTR(uint8_t, base, L.off_src0);
    mmctx->vtcm_act      = VTCM_LAYOUT_PTR(uint8_t, base, L.off_act);
    mmctx->vtcm_dst      = VTCM_LAYOUT_PTR(uint8_t, base, L.off_dst);
    mmctx->vtcm_act_raw  = VTCM_LAYOUT_PTR(uint8_t, base, L.off_act_raw);

    mmctx->vtcm_src0_stride    = 0;
    mmctx->vtcm_act_stride     = act_row_size;
    mmctx->vtcm_act_raw_stride = hex_round_up(act->ne[0] * পরিমাণ(ভিন্ন), QK_Q8_0_TILED * পরিমাণ(ভিন্ন));

    mmctx->vtcm_src0_size_per_thread = fastdiv(L.src0_bytes, &octx->n_threads_div);
    mmctx->vtcm_act_size_per_thread  = L.act_bytes;
    mmctx->vtcm_dst_size_per_thread  = fastdiv(L.dst_bytes, &octx->n_threads_div);

    mmctx->cur_m_start = 0;
    mmctx->cur_m_rows  = act_nrows;

    FARF(উপরের_2, "matmul-id-nx: src0 %d:%d:%d type %s nrows %u, act %d:%d:%d nrows %u, vtcm %zu/%zu, threads %d\n",
         src0->ne[0], src0->ne[1], src0->ne[2], mmctx->type, src0->ne[1],
         act->ne[0], act->ne[1], act->ne[2], act_nrows,
         L.total_bytes, octx->ctx->vtcm_size, octx->n_threads);

    htp_trace_event_stop(tr, HTP_TRACE_EVT_INIT, 0);

    hvx_mm_transfer_act_dma(octx, kparams, act, mmctx->vtcm_act_raw, mmctx->vtcm_act_raw_stride, 0, act_nrows);

    mmctx->n_quant_rows_per_thread = (act_nrows + n_quant_tasks - 1) / n_quant_tasks;
    mmctx->n_quant_tasks           = n_quant_tasks;
    work_queue_run(octx->ctx->work_queue, quant_task_func, mmctx, n_quant_tasks);

    work_queue_run(octx->ctx->work_queue, hvx_mmid_task_func, mmctx, octx->n_threads);

    প্রত্যাগম HTP_STATUS_OK;
}

স্থির অন্তঃস্থ শূন্য scan_expert_ids_n(
    ধ্রুব সংরচনা htp_tensor * ids,
    ধ্রুব uint32_t n_ids,
    uint32_t n_as,
    uint32_t * counts,
    সংরচনা mmid_row_mapping * matrix_rows,
    uint32_t mapping_stride
) {
    ধ্রুব size_t ids_nb1 = ids->nb[1];
    ধ্রুব uint8_t * ids_data = (ধ্রুব uint8_t *) ids->data;

    প্রতি (uint32_t iid1 = 0; iid1 < ids->ne[1]; ++iid1) {
        ধ্রুব int32_t * row_ptr = (ধ্রুব int32_t *) (ids_data + iid1 * ids_nb1);
        প্রতি (uint32_t id = 0; id < n_ids; ++id) {
            ধ্রুব int32_t i02 = row_ptr[id];
            যদি (i02 < 0) {
                অনুবর্ত;
            }
            প্রতিজ্ঞা(i02 < n_as);

            যদি (matrix_rows) {
                matrix_rows[i02 * mapping_stride + counts[i02]] = (সংরচনা mmid_row_mapping) { id, iid1 };
            }
            counts[i02] += 1;
        }
    }
}

স্থির অন্তঃস্থ শূন্য scan_expert_ids(
    ধ্রুব সংরচনা htp_tensor * ids,
    uint32_t n_ids,
    uint32_t n_as,
    uint32_t * counts,
    সংরচনা mmid_row_mapping * matrix_rows,
    uint32_t mapping_stride
) {
    ধ্রুব size_t ids_nb0 = ids->nb[0];

    যদি (ids_nb0 == 4) {
        বিকল্প (n_ids) {
            পক্ষ 8:  scan_expert_ids_n(ids, 8,     n_as, counts, matrix_rows, mapping_stride); ভঙ্গ;
            পক্ষ 4:  scan_expert_ids_n(ids, 4,     n_as, counts, matrix_rows, mapping_stride); ভঙ্গ;
            পক্ষ 2:  scan_expert_ids_n(ids, 2,     n_as, counts, matrix_rows, mapping_stride); ভঙ্গ;
            স্বভাব: scan_expert_ids_n(ids, n_ids, n_as, counts, matrix_rows, mapping_stride); ভঙ্গ;
        }
    } অন্যথা {
        // Strided fallback
        ধ্রুব size_t ids_nb1 = ids->nb[1];
        ধ্রুব uint8_t * ids_data = (ধ্রুব uint8_t *) ids->data;
        প্রতি (uint32_t iid1 = 0; iid1 < ids->ne[1]; ++iid1) {
            ধ্রুব int32_t * row_ptr = (ধ্রুব int32_t *) (ids_data + iid1 * ids_nb1);
            প্রতি (uint32_t id = 0; id < n_ids; ++id) {
                ধ্রুব int32_t i02 = *(ধ্রুব int32_t *) ((ধ্রুব uint8_t *) row_ptr + id * ids_nb0);
                যদি (i02 < 0) {
                    অনুবর্ত;
                }
                প্রতিজ্ঞা(i02 < n_as);

                যদি (matrix_rows) {
                    matrix_rows[i02 * mapping_stride + counts[i02]] = (সংরচনা mmid_row_mapping) { id, iid1 };
                }
                counts[i02] += 1;
            }
        }
    }
}

পূর্ণাঙ্ক op_matmul_id(সংরচনা htp_ops_context * octx) {
    htp_matmul_tensors_preamble;

    ধ্রুব সংরচনা htp_mm_kernel_params * kparams = (ধ্রুব সংরচনা htp_mm_kernel_params *) octx->kernel_params;
    সংরচনা htp_mm_context mmctx_struct = {0};
    সংরচনা htp_mm_context * mmctx = &mmctx_struct;

    ধ্রুব পূর্ণাঙ্ক status = htp_mm_init_context(octx, kparams);
    যদি (status != HTP_STATUS_OK) {
        প্রত্যাগম status;
    }

    সংরচনা htp_thread_trace * tr = &octx->ctx->trace[0];
    htp_trace_event_start(tr, HTP_TRACE_EVT_INIT, 0);

    mmctx->octx = octx;
    mmctx->act = act;

    ধ্রুব সংরচনা htp_tensor * restrict ids = octx->src[2];
    যদি (htp_tensor_is_extended(ids) || htp_tensor_is_extended(act) || htp_tensor_is_extended(dst)) {
        প্রত্যাগম HTP_STATUS_NO_SUPPORT;
    }

    ধ্রুব size_t src0_row_size = nb01;
    ধ্রুব size_t dst_row_size  = nb1;

    ধ্রুব size_t src0_row_size_padded = hex_round_up(src0_row_size, 128);

    ধ্রুব uint32_t src0_nrows = ne01;  // per expert
    ধ্রুব uint32_t act_nrows  = ne11 * ne12 * ne13;

    // row groups
    ধ্রুব পূর্ণাঙ্ক n_ids = ids->ne[0];  // n_expert_used
    ধ্রুব পূর্ণাঙ্ক n_as  = ne02;        // n_expert

    uint8_t  * mapping_buf       = octx->ctx->ddr_spad_base;
    uint32_t   mapping_stride    = 1;
    uint32_t * matrix_row_counts = (uint32_t *) mapping_buf;
    সংরচনা mmid_row_mapping * matrix_rows = রিক্ত;

    যদি (act_nrows > 1) {
        ধ্রুব size_t matrix_row_counts_size = n_as * পরিমাণ(uint32_t);
        প্রতিজ্ঞা(octx->ctx->ddr_spad_size >= matrix_row_counts_size);

        hex_l2fetch_block((ধ্রুব শূন্য *) ids->data, ids->ne[1] * ids->nb[1]);

        memset(matrix_row_counts, 0, matrix_row_counts_size);
        scan_expert_ids(ids, n_ids, n_as, matrix_row_counts, রিক্ত, 0);

        uint32_t max_গুনতি = hvx_reduce_max_i32((ধ্রুব uint8_t *) matrix_row_counts, n_as);
        mapping_stride = max_গুনতি > 0 ? max_গুনতি : 1;

        size_t matrix_row_map_size  = n_as * mapping_stride * পরিমাণ(সংরচনা mmid_row_mapping);
        ধ্রুব size_t total_map_size = matrix_row_counts_size + matrix_row_map_size;

        যদি (total_map_size > octx->ctx->ddr_spad_size) {
            mapping_buf = memalign(128, total_map_size);
            যদি (!mapping_buf) {
                প্রত্যাগম HTP_STATUS_INTERNAL_ERR;
            }
        }

        matrix_row_counts = (uint32_t *) mapping_buf;
        matrix_rows       = (সংরচনা mmid_row_mapping *) (mapping_buf + matrix_row_counts_size);

        memset(matrix_row_counts, 0, n_as * পরিমাণ(uint32_t));
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

    পূর্ণাঙ্ক s;
    যদি (kparams->n_hmx) {
        s = hmx_mm_op_matmul_id(octx, mmctx);
    } অন্যথা {
        uint32_t src0_row_start = 0;
        uint32_t src0_row_end   = src0_nrows;
        যদি (octx->ctx->mdev.গুনতি > 1) {
            ধ্রুব bool can_split = htp_tensor_can_row_partition(dst, পরিমাণ(ভিন্ন));
            ধ্রুব সংরচনা htp_tensor_mdev_range range = htp_tensor_mdev_partition(src0_nrows, can_split ? 32 : 0, octx->ctx->mdev.idx, octx->ctx->mdev.গুনতি, &octx->ctx->mdev.গুনতি_div);
            src0_row_start = range.start;
            src0_row_end   = range.start + range.গুনতি;
        }

        যদি (src0_row_start >= src0_row_end) {
            যদি (mapping_buf != octx->ctx->ddr_spad_base) {
                মুক্ত(mapping_buf);
            }
            প্রত্যাগম HTP_STATUS_OK;
        }

        ধ্রুব uint32_t nrows = src0_row_end - src0_row_start;
        mmctx->src0_row_start = src0_row_start;
        mmctx->src0_row_end   = src0_row_end;

        mmctx->src0_nrows_per_thread = fastdiv(nrows + octx->n_threads - 1, &octx->n_threads_div);
        mmctx->src0_nrows_per_thread = hex_round_up(mmctx->src0_nrows_per_thread, 32);

        যদি (hvx_mm_init_tiled_vec_dot(mmctx, src0->type) == 0) {
            s = hvx_mm_matmul_id(octx, mmctx, act_nrows > 1 ? hvx_mm_id_tiled : hvx_mv_id_tiled);
        } অন্যথা {
            s = HTP_STATUS_NO_SUPPORT;
        }
    }

    যদি (mapping_buf != octx->ctx->ddr_spad_base) {
        মুক্ত(mapping_buf);
    }

    প্রত্যাগম s;
}

পূর্ণাঙ্ক op_matmul_id_nx(সংরচনা htp_ops_context * octx) {
    ধ্রুব সংরচনা htp_mm_kernel_params * kparams = (ধ্রুব সংরচনা htp_mm_kernel_params *) octx->kernel_params;
    সংরচনা htp_mm_context mmctx_struct = {0};
    সংরচনা htp_mm_context * mmctx = &mmctx_struct;

    ধ্রুব পূর্ণাঙ্ক status = htp_mm_init_context(octx, kparams);
    যদি (status != HTP_STATUS_OK) {
        প্রত্যাগম status;
    }

    সংরচনা htp_thread_trace * tr = &octx->ctx->trace[0];
    htp_trace_event_start(tr, HTP_TRACE_EVT_INIT, 0);

    mmctx->octx = octx;
    ধ্রুব uint32_t n_weights = kparams->n_weights;
    ধ্রুব সংরচনা htp_tensor * restrict src0 = octx->src[0];
    ধ্রুব সংরচনা htp_tensor * restrict act  = octx->src[n_weights];
    ধ্রুব সংরচনা htp_tensor * restrict ids  = octx->src[n_weights + 1];
    যদি (htp_tensor_is_extended(ids) || htp_tensor_is_extended(act)) {
        প্রত্যাগম HTP_STATUS_NO_SUPPORT;
    }
    প্রতি (uint32_t p = 0; p < n_weights; p++) {
        যদি (htp_tensor_is_extended(octx->dsts[p])) {
            প্রত্যাগম HTP_STATUS_NO_SUPPORT;
        }
    }

    mmctx->act = act;

    ধ্রুব size_t src0_row_size = src0->nb[1];
    ধ্রুব size_t src0_row_size_padded = hex_round_up(src0_row_size, 128);

    ধ্রুব uint32_t act_nrows = act->ne[1] * act->ne[2] * act->ne[3];

    ধ্রুব পূর্ণাঙ্ক n_ids = ids->ne[0];
    ধ্রুব পূর্ণাঙ্ক n_as  = src0->ne[2];

    uint8_t  * mapping_buf       = octx->ctx->ddr_spad_base;
    uint32_t   mapping_stride    = 1;
    uint32_t * matrix_row_counts = (uint32_t *) mapping_buf;
    সংরচনা mmid_row_mapping * matrix_rows = রিক্ত;

    যদি (act_nrows > 1) {
        ধ্রুব size_t matrix_row_counts_size = n_as * পরিমাণ(uint32_t);
        প্রতিজ্ঞা(octx->ctx->ddr_spad_size >= matrix_row_counts_size);

        hex_l2fetch_block((ধ্রুব শূন্য *) ids->data, ids->ne[1] * ids->nb[1]);

        memset(matrix_row_counts, 0, matrix_row_counts_size);
        scan_expert_ids(ids, n_ids, n_as, matrix_row_counts, রিক্ত, 0);

        uint32_t max_গুনতি = hvx_reduce_max_i32((ধ্রুব uint8_t *) matrix_row_counts, n_as);
        mapping_stride = max_গুনতি > 0 ? max_গুনতি : 1;

        size_t matrix_row_map_size  = n_as * mapping_stride * পরিমাণ(সংরচনা mmid_row_mapping);
        ধ্রুব size_t total_map_size = matrix_row_counts_size + matrix_row_map_size;

        যদি (total_map_size > octx->ctx->ddr_spad_size) {
            mapping_buf = memalign(128, total_map_size);
            যদি (!mapping_buf) {
                প্রত্যাগম HTP_STATUS_INTERNAL_ERR;
            }
        }

        matrix_row_counts = (uint32_t *) mapping_buf;
        matrix_rows       = (সংরচনা mmid_row_mapping *) (mapping_buf + matrix_row_counts_size);

        memset(matrix_row_counts, 0, n_as * পরিমাণ(uint32_t));
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

    পূর্ণাঙ্ক s;
    যদি (kparams->n_hmx) {
        s = hmx_mm_op_matmul_id_nx(octx, mmctx);
    } অন্যথা {
        যদি (hvx_mm_init_tiled_vec_dot(mmctx, src0->type) == 0) {
            s = hvx_mm_matmul_id_nx(octx, mmctx, act_nrows > 1 ? hvx_mm_id_nx_tiled : hvx_mv_id_nx_tiled);
        } অন্যথা {
            s = HTP_STATUS_NO_SUPPORT;
        }
    }

    যদি (mapping_buf != octx->ctx->ddr_spad_base) {
        মুক্ত(mapping_buf);
    }

    প্রত্যাগম s;
}
স্থির পূর্ণাঙ্ক op_matmul_nx_impl(সংরচনা htp_ops_context * octx) {
    ধ্রুব সংরচনা htp_mm_kernel_params * kparams = (ধ্রুব সংরচনা htp_mm_kernel_params *) octx->kernel_params;

    ধ্রুব পূর্ণাঙ্ক status = htp_mm_init_context(octx, kparams);
    যদি (status != HTP_STATUS_OK) {
        প্রত্যাগম status;
    }

    যদি (kparams->n_hmx) {
        প্রত্যাগম hmx_mm_nx_2d_f32(octx, kparams);
    }

    সংরচনা htp_thread_trace * tr = &octx->ctx->trace[0];
    htp_trace_event_start(tr, HTP_TRACE_EVT_INIT, 0);

    ধ্রুব uint32_t n_weights = kparams->n_weights;

    ধ্রুব সংরচনা htp_tensor * restrict src0 = octx->src[0]; // first weight
    ধ্রুব সংরচনা htp_tensor * restrict act  = octx->src[n_weights]; // activation x

    সংরচনা htp_mm_context mmctx_struct = {0};
    সংরচনা htp_mm_context * mmctx = &mmctx_struct;
    mmctx->octx = octx;
    mmctx->act  = act;

    ধ্রুব uint32_t act_nrows = act->ne[1] * act->ne[2] * act->ne[3];
    mmctx->act_nrows   = act_nrows;
    mmctx->cur_m_start = 0;
    mmctx->cur_m_rows  = act_nrows;

    ধ্রুব size_t src0_row_size = src0->nb[1];

    যদি (hvx_mm_init_tiled_vec_dot(mmctx, src0->type) != 0) {
        প্রত্যাগম HTP_STATUS_NO_SUPPORT;
    }

    ধ্রুব uint32_t qk = QK_Q8_0_TILED;
    ধ্রুব uint32_t nb = (act->ne[0] + qk - 1) / qk;
    ধ্রুব uint32_t total_nb = act_nrows * nb;

    worker_callback_t quant_task_func;
    uint32_t n_quant_tasks = 1;
    যদি (act_nrows < octx->n_threads) {
        n_quant_tasks = MIN(total_nb, octx->n_threads);
        quant_task_func = htp_mm_act_quant_block_func(src0->type, act->type);
        প্রতি (uint32_t ith = 0; ith < n_quant_tasks; ++ith) {
            uint32_t ib_first = (total_nb * ith) / n_quant_tasks;
            uint32_t ib_last  = (total_nb * (ith + 1)) / n_quant_tasks;
            mmctx->quant_ib_first[ith] = ib_first;
            mmctx->quant_ib_last[ith]  = ib_last;
            mmctx->quant_r[ith]        = ib_first / nb;
            mmctx->quant_c[ith]        = ib_first % nb;
        }
    } অন্যথা {
        n_quant_tasks = MIN(act_nrows, octx->n_threads);
        quant_task_func = htp_mm_act_quant_row_func(src0->type, act->type);
    }

    ধ্রুব size_t act_row_size = htp_mm_tiled_act_row_size(src0->type, act->ne[0]);

    সংরচনা htp_mm_hvx_vtcm_layout L;
    htp_mm_hvx_vtcm_layout_build(&L, kparams->kernel_type, src0->type, act->ne[0], act_nrows, octx->n_threads,
                                 0, src0_row_size, act_row_size, 0, kparams->n_prefetch, false, true);

    ধ্রুব size_t vtcm_size = L.total_bytes;

    যদি (octx->ctx->vtcm_size < vtcm_size) {
        FARF(ERROR, "matmul-nx: current VTCM reservation %zu is too small, needed %zu\n",
             octx->ctx->vtcm_size, vtcm_size);
        প্রত্যাগম HTP_STATUS_VTCM_TOO_SMALL;
    }

    uint8_t * ধ্রুব base = (uint8_t *) octx->ctx->vtcm_base;
    mmctx->vtcm_src0     = VTCM_LAYOUT_PTR(uint8_t, base, L.off_src0);
    mmctx->vtcm_act      = VTCM_LAYOUT_PTR(uint8_t, base, L.off_act);
    mmctx->vtcm_dst      = VTCM_LAYOUT_PTR(uint8_t, base, L.off_dst);
    mmctx->vtcm_act_raw  = VTCM_LAYOUT_PTR(uint8_t, base, L.off_act_raw);

    mmctx->vtcm_src0_stride    = 0;
    mmctx->vtcm_act_stride     = act_row_size;
    mmctx->vtcm_act_raw_stride = hex_round_up(act->ne[0] * পরিমাণ(ভিন্ন), QK_Q8_0_TILED * পরিমাণ(ভিন্ন));

    mmctx->vtcm_src0_size_per_thread = fastdiv(L.src0_bytes, &octx->n_threads_div);
    mmctx->vtcm_act_size_per_thread  = L.act_bytes;
    mmctx->vtcm_dst_size_per_thread  = fastdiv(L.dst_bytes, &octx->n_threads_div);

    htp_trace_event_stop(tr, HTP_TRACE_EVT_INIT, 0);

    hvx_mm_transfer_act_dma(octx, kparams, act, mmctx->vtcm_act_raw, mmctx->vtcm_act_raw_stride, 0, act_nrows);

    mmctx->n_quant_rows_per_thread = (act_nrows + n_quant_tasks - 1) / n_quant_tasks;
    mmctx->n_quant_tasks = n_quant_tasks;
    work_queue_run(octx->ctx->work_queue, quant_task_func, mmctx, n_quant_tasks);

    work_queue_run(octx->ctx->work_queue, hvx_mm_nx_2d_tiled, mmctx, octx->n_threads);

    প্রত্যাগম HTP_STATUS_OK;
}

পূর্ণাঙ্ক op_matmul_nx(সংরচনা htp_ops_context * octx) {
    ধ্রুব সংরচনা htp_mm_kernel_params * kparams = (ধ্রুব সংরচনা htp_mm_kernel_params *) octx->kernel_params;

    যদি (kparams->collapse) {
        ধ্রুব uint32_t n_weights = kparams->n_weights;
        ধ্রুব সংরচনা htp_tensor * act = octx->src[n_weights];
        ধ্রুব uint32_t s1 = (act->ne[1] > 1) ? act->nb[1] : ((act->ne[2] > 1) ? act->nb[2] : act->nb[3]);
        সংরচনা htp_tensor act_collapsed;
        সংরচনা htp_tensor dsts_collapsed[HTP_OP_MAX_OUTPUTS];
        ধ্রুব সংরচনা htp_tensor * orig_dsts[HTP_OP_MAX_OUTPUTS];

        htp_mm_tensor_collapse_rows(&act_collapsed, act, s1);
        octx->src[n_weights] = &act_collapsed;

        প্রতি (uint32_t p = 0; p < n_weights; p++) {
            orig_dsts[p] = octx->dsts[p];
            ধ্রুব সংরচনা htp_tensor * d = octx->dsts[p];
            ধ্রুব uint32_t sd = (d->ne[1] > 1) ? d->nb[1] : ((d->ne[2] > 1) ? d->nb[2] : d->nb[3]);
            htp_mm_tensor_collapse_rows(&dsts_collapsed[p], d, sd);
            octx->dsts[p] = &dsts_collapsed[p];
        }

        ধ্রুব পূর্ণাঙ্ক status = op_matmul_nx_impl(octx);

        octx->src[n_weights] = act;
        প্রতি (uint32_t p = 0; p < n_weights; p++) {
            octx->dsts[p] = orig_dsts[p];
        }
        প্রত্যাগম status;
    }

    প্রত্যাগম op_matmul_nx_impl(octx);
}
