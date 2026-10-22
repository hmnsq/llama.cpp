#pragma clang diagnostic ignored "-Wgnu-zero-variadic-macro-arguments"
#pragma clang diagnostic ignored "-Wunused-function"
#pragma clang diagnostic ignored "-Wunused-variable"
#pragma clang diagnostic ignored "-Wunused-but-set-variable"

#अन्तर्भाव <HAP_farf.h>
#अन्तर्भाव <HAP_perf.h>
#अन्तर्भाव <HAP_compute_res.h>

#अन्तर्भाव <math.h>
#अन्तर्भाव <string.h>
#अन्तर्भाव <stdatomic.h>

#अन्तर्भाव "dma-queue.h"
#अन्तर्भाव "hvx-utils.h"
#अन्तर्भाव "hvx-dump.h"
#अन्तर्भाव "hvx-arith.h"
#अन्तर्भाव "hvx-reduce.h"

#define GGML_COMMON_DECL_C
#अन्तर्भाव "ggml-common.h"
#अन्तर्भाव "htp-ctx.h"
#अन्तर्भाव "htp-ops.h"
#अन्तर्भाव "htp-tensor.h"
#अन्तर्भाव "matmul-ops.h"
#अन्तर्भाव "htp-vtcm.h"

प्रकारनिर्देश संरचना {
    भिन्न        *dst;
    dma_addr_t    bias_addr;
    माप_प्रकार        bias_bytes;
    dma_addr_t    act_dma_addr;
    dma_addr_t    weight;
    dma_queue *   weight_dma;
    पूर्णाङ्क           m;
    पूर्णाङ्क           k;
    पूर्णाङ्क           n;
    पूर्णाङ्क           act_stride;
    uint32_t      act_elem_size;  // 4=F32 src1, 2=F16 src1
    पूर्णाङ्क           weight_stride;
    पूर्णाङ्क           weight_type;
    पूर्णाङ्क           dst_stride;
    पूर्णाङ्क           dst_cols;
    uint32_t      bias_stride;
    पूर्णाङ्क           tile_size;
    पूर्णाङ्क           aligned_tile_size;
    पूर्णाङ्क           ne02;
    पूर्णाङ्क           ne03;
    पूर्णाङ्क           ne12;
    पूर्णाङ्क           ne13;
    माप_प्रकार        src0_nb2;
    माप_प्रकार        src0_nb3;
    माप_प्रकार        act_nb2;
    माप_प्रकार        act_nb3;
    माप_प्रकार        bias_nb2;
    माप_प्रकार        bias_nb3;
    माप_प्रकार        dst_nb2;
    माप_प्रकार        dst_nb3;
    पूर्णाङ्क           r2;
    पूर्णाङ्क           r3;
    संरचना fastdiv_मूल्यावली div_r2;
    संरचना fastdiv_मूल्यावली div_r3;
} hmx_mm_batched_params_t;

स्थिर bool htp_matmul_has_extended_weight(ध्रुव संरचना htp_ops_context * octx, uint32_t n_weights) {
    प्रति (uint32_t i = 0; i < n_weights; ++i) {
        यदि (htp_tensor_is_extended(octx->src[i])) {
            प्रत्यागम true;
        }
    }
    प्रत्यागम false;
}

संरचना htp_mm_context {
    ध्रुव वर्ण * type;
    संरचना htp_ops_context * octx;
    ध्रुव संरचना htp_tensor * act;

    शून्य (*vec_dot_1x1)(ध्रुव uint32_t n, भिन्न * restrict s0,
         ध्रुव शून्य * restrict vx0,
         ध्रुव शून्य * restrict vy0);

    शून्य (*vec_dot_2x1)(ध्रुव uint32_t n, भिन्न * restrict s0,
         ध्रुव शून्य * restrict vx0, ध्रुव शून्य * restrict vx1,
         ध्रुव शून्य * restrict vy0);

    शून्य (*vec_dot_2x2)(ध्रुव uint32_t n, भिन्न * restrict s0, भिन्न * restrict s1,
         ध्रुव शून्य * restrict vx0, ध्रुव शून्य * restrict vx1,
         ध्रुव शून्य * restrict vy0, ध्रुव शून्य * restrict vy1);

    शून्य (*vec_dot_32x1)(ध्रुव uint32_t n, भिन्न * restrict s,
         ध्रुव शून्य * restrict vx,
         ध्रुव शून्य * restrict vy, uint32_t valid_rows,
         ध्रुव भिन्न * restrict sz);

    शून्य (*vec_dot_32x2)(ध्रुव uint32_t n, भिन्न * restrict s0, भिन्न * restrict s1,
         ध्रुव शून्य * restrict vx,
         ध्रुव शून्य * restrict vy0, ध्रुव शून्य * restrict vy1, uint32_t valid_rows,
         ध्रुव भिन्न * restrict sz0, ध्रुव भिन्न * restrict sz1);

    // Precomputed values
    uint32_t src0_nrows_per_thread;
    uint32_t src0_row_start;
    uint32_t src0_row_end;
    uint32_t src0_row_size_padded;
    uint32_t act_nrows;
    uint32_t cur_m_start;
    uint32_t cur_m_rows;

    संरचना fastdiv_मूल्यावली mm_div_ne12_ne1;
    संरचना fastdiv_मूल्यावली mm_div_ne1;
    संरचना fastdiv_मूल्यावली mm_div_r2;
    संरचना fastdiv_मूल्यावली mm_div_r3;
    संरचना fastdiv_मूल्यावली mm_div_ne11;

    // Precomputed block-parallel quantization values
    uint32_t          quant_ib_first[WORK_QUEUE_MAX_N_THREADS];
    uint32_t          quant_ib_last[WORK_QUEUE_MAX_N_THREADS];
    uint32_t          quant_r[WORK_QUEUE_MAX_N_THREADS];
    uint32_t          quant_c[WORK_QUEUE_MAX_N_THREADS];
    uint32_t          n_quant_tasks;
    uint32_t          n_quant_rows_per_thread;

    // Fields for scattered mapping & HMX support in MUL_MAT_ID
    ध्रुव uint32_t * matrix_row_counts;
    ध्रुव संरचना mmid_row_mapping * matrix_rows;
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

स्थिर पूर्णाङ्क htp_mm_init_context(
    संरचना htp_ops_context * octx,
    ध्रुव संरचना htp_mm_kernel_params * kparams
) {
    यदि (!htp_ops_context_set_n_threads(octx, (uint32_t) kparams->n_threads)) {
        प्रत्यागम HTP_STATUS_INVAL_PARAMS;
    }

    यदि (kparams->n_hmx) {
        यदि (kparams->n_act_threads <= 0 || kparams->n_act_threads > (int32_t) octx->n_threads) {
            प्रत्यागम HTP_STATUS_INVAL_PARAMS;
        }
    }

    प्रत्यागम HTP_STATUS_OK;
}

// vdelta control to expand first 32 e8m0 values into 32 uint32 elements
स्थिर ध्रुव uint8_t __attribute__((aligned(128))) expand_x32_e8m0[128] = {
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
स्थिर ध्रुव uint8_t __attribute__((aligned(VLEN))) kvalues_iq4nl_lut[] = {
    0x81, 0, 0x98, 0, 0xAD, 0, 0xBF, 0, 0xCF, 0, 0xDD, 0, 0xEA, 0, 0xF6, 0, 0x01, 0, 0x0D, 0, 0x19, 0, 0x26, 0,
    0x35, 0, 0x45, 0, 0x59, 0, 0x71, 0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0,
    0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0,
    0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0,
    0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0,
};

स्थिर ध्रुव uint8_t __attribute__((aligned(VLEN))) kvalues_mxfp4_lut[] = {
    0,    0, 1,    0, 2,    0, 3, 0, 4, 0, 6, 0, 8, 0, 12, 0, 0, 0, 0xff, 0, 0xfe, 0, 0xfd, 0, 0xfc, 0,
    0xfa, 0, 0xf8, 0, 0xf4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 0,    0, 0,    0, 0,    0, 0,    0,
    0,    0, 0,    0, 0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 0,    0, 0,    0, 0,    0, 0,    0,
    0,    0, 0,    0, 0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 0,    0, 0,    0, 0,    0, 0,    0,
    0,    0, 0,    0, 0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 0,    0, 0,    0, 0,    0,
};

#define htp_matmul_tensors_preamble                                 \
    ध्रुव संरचना htp_tensor * restrict src0 = octx->src[0];         \
    ध्रुव संरचना htp_tensor * restrict src1 = octx->src[1];         \
    ध्रुव संरचना htp_tensor * restrict src2 = octx->src[2];         \
    ध्रुव संरचना htp_tensor * restrict  dst = octx->dst;            \
    ध्रुव संरचना htp_tensor * restrict  act = src1;                 \
                                                                    \
    ध्रुव uint32_t ne00 = src0->ne[0];                              \
    ध्रुव uint32_t ne01 = src0->ne[1];                              \
    ध्रुव uint32_t ne02 = src0->ne[2];                              \
    ध्रुव uint32_t ne03 = src0->ne[3];                              \
                                                                    \
    ध्रुव uint32_t ne10 = src1->ne[0];                              \
    ध्रुव uint32_t ne11 = src1->ne[1];                              \
    ध्रुव uint32_t ne12 = src1->ne[2];                              \
    ध्रुव uint32_t ne13 = src1->ne[3];                              \
                                                                    \
    ध्रुव uint32_t ne20 = src2 ? src2->ne[0] : 0;                   \
    ध्रुव uint32_t ne21 = src2 ? src2->ne[1] : 0;                   \
    ध्रुव uint32_t ne22 = src2 ? src2->ne[2] : 0;                   \
    ध्रुव uint32_t ne23 = src2 ? src2->ne[3] : 0;                   \
                                                                    \
    ध्रुव uint32_t ne0 = dst->ne[0];                                \
    ध्रुव uint32_t ne1 = dst->ne[1];                                \
    ध्रुव uint32_t ne2 = dst->ne[2];                                \
    ध्रुव uint32_t ne3 = dst->ne[3];                                \
                                                                    \
    ध्रुव uint32_t nb00 = src0->nb[0];                              \
    ध्रुव uint32_t nb01 = src0->nb[1];                              \
    ध्रुव uint32_t nb02 = src0->nb[2];                              \
    ध्रुव uint32_t nb03 = src0->nb[3];                              \
                                                                    \
    ध्रुव uint32_t nb10 = src1->nb[0];                              \
    ध्रुव uint32_t nb11 = src1->nb[1];                              \
    ध्रुव uint32_t nb12 = src1->nb[2];                              \
    ध्रुव uint32_t nb13 = src1->nb[3];                              \
                                                                    \
    ध्रुव uint32_t nb0 = dst->nb[0];                                \
    ध्रुव uint32_t nb1 = dst->nb[1];                                \
    ध्रुव uint32_t nb2 = dst->nb[2];                                \
    ध्रुव uint32_t nb3 = dst->nb[3];

#define htp_matmul_preamble                                         \
    संरचना htp_mm_context * mmctx  = data;                          \
    संरचना htp_ops_context * octx  = mmctx->octx;                   \
    dma_queue * dma_q              = octx->ctx->dma[ith];           \
    uint32_t src0_nrows_per_thread = mmctx->src0_nrows_per_thread;  \
    htp_matmul_tensors_preamble;




// hvx kernels first: the HMX Q6_K dequantizer reuses unpack_q6_k_group from there
#अन्तर्भाव "hvx-mm-kernels-tiled.h"
#अन्तर्भाव "hvx-mm-kernels-float.h"
#अन्तर्भाव "hmx-mm-kernels-tiled.h"

स्थिर शून्य hvx_mm_2d_tiled(अचिन्हित पूर्णाङ्क nth, अचिन्हित पूर्णाङ्क ith, शून्य * data) {
    htp_matmul_preamble;

    ध्रुव uint32_t src0_nrows = mmctx->src0_row_end - mmctx->src0_row_start;
    ध्रुव uint32_t act_nrows = mmctx->cur_m_rows ? mmctx->cur_m_rows : (ne11 * ne12 * ne13);
    ध्रुव uint32_t cur_m_start = mmctx->cur_m_start;

    ध्रुव uint32_t src0_start_row  = mmctx->src0_row_start + src0_nrows_per_thread * ith;
    ध्रुव uint32_t src0_end_row    = MIN(src0_start_row + src0_nrows_per_thread, mmctx->src0_row_end);

    संरचना htp_thread_trace * tr = &octx->ctx->trace[ith];

    ध्रुव संरचना htp_mm_kernel_params * kparams = (ध्रुव संरचना htp_mm_kernel_params *) octx->kernel_params;
    ध्रुव uint32_t n_prefetch = kparams->n_prefetch;
    प्रतिज्ञा(n_prefetch >= 2 && n_prefetch <= HTP_MM_MAX_PREFETCH && (n_prefetch & (n_prefetch - 1)) == 0);

    ध्रुव माप_प्रकार dst_row_size  = nb1;
    ध्रुव माप_प्रकार act_row_size = nb11;
    ध्रुव माप_प्रकार act_stride = mmctx->vtcm_act_stride;
    ध्रुव माप_प्रकार src2_stride = src2 ? ((src2->ne[1] == 1) ? 0 : src2->nb[1]) : 0;

    uint8_t * restrict vtcm_dst_ptr  = mmctx->vtcm_dst  + mmctx->vtcm_dst_size_per_thread  * ith;
    uint8_t * restrict vtcm_src0_ptr = mmctx->vtcm_src0 + mmctx->vtcm_src0_size_per_thread * ith;
    uint8_t * restrict act_data = mmctx->vtcm_act;

    ध्रुव dma_addr_t src0_row = src0->data;

    ध्रुव uint32_t tile_size = htp_tensor_get_tile_size(src0->type);
    ध्रुव uint32_t aligned_tile_size = htp_tensor_get_tile_padded_size(src0->type);

    uint32_t n_k_tiles_w = ne00 / htp_tensor_get_tile_k(src0->type);
    uint32_t tile_row_stride = n_k_tiles_w * tile_size;
    uint32_t tile_row_transfer_size_aligned = n_k_tiles_w * aligned_tile_size;

    uint32_t ct_start = src0_start_row / 32;
    uint32_t ct_end   = (src0_end_row + 31) / 32;

    uint32_t धकेलो_ct = ct_start;
    यदि (src0_start_row < src0_end_row) {
        प्रति (uint32_t d = 0; d < n_prefetch && धकेलो_ct < ct_end; d++, धकेलो_ct++) {
            dma_queue_धकेलो(dma_q, dma_make_data(vtcm_src0_ptr + d * tile_row_transfer_size_aligned,
                           src0_row + धकेलो_ct * tile_row_stride), aligned_tile_size, tile_size, tile_size, n_k_tiles_w);
        }
    }

    यदि (src0_start_row >= src0_end_row) {
        प्रत्यागम;
    }

    प्रति (uint32_t ct = ct_start; ct < ct_end; ct++) {
        ध्रुव uint8_t * w_tile = (शून्य *) dma_queue_निकालो(dma_q).dst;

        पूर्णाङ्क valid_rows = (पूर्णाङ्क)ne0 - (पूर्णाङ्क)(ct * 32);
        valid_rows = MIN(32, MAX(0, valid_rows));

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, ct);
        uint32_t ir1 = 0;
        प्रति (; ir1 + 1 < act_nrows; ir1 += 2) {
            ध्रुव uint8_t * restrict act_col0 = (ध्रुव uint8_t *) (act_data + (ir1+0) * act_stride);
            ध्रुव uint8_t * restrict act_col1 = (ध्रुव uint8_t *) (act_data + (ir1+1) * act_stride);
            भिन्न * restrict dst_row0 = (भिन्न *) (dst->data + ((cur_m_start + ir1+0) * dst_row_size));
            भिन्न * restrict dst_row1 = (भिन्न *) (dst->data + ((cur_m_start + ir1+1) * dst_row_size));

            भिन्न * dst_ptr0 = &dst_row0[ct * 32];
            भिन्न * dst_ptr1 = &dst_row1[ct * 32];

            ध्रुव भिन्न * src2_ptr0 = रिक्त;
            ध्रुव भिन्न * src2_ptr1 = रिक्त;
            यदि (src2) {
                ध्रुव भिन्न * restrict src2_row0 = (ध्रुव भिन्न *) ((ध्रुव uint8_t *) src2->data + ((cur_m_start + ir1+0) * src2_stride));
                ध्रुव भिन्न * restrict src2_row1 = (ध्रुव भिन्न *) ((ध्रुव uint8_t *) src2->data + ((cur_m_start + ir1+1) * src2_stride));
                src2_ptr0 = &src2_row0[ct * 32];
                src2_ptr1 = &src2_row1[ct * 32];
            }
            mmctx->vec_dot_32x2(ne10, dst_ptr0, dst_ptr1, w_tile, act_col0, act_col1, valid_rows, src2_ptr0, src2_ptr1);
        }

        प्रति (; ir1 < act_nrows; ++ir1) {
            ध्रुव uint8_t * restrict act_col = (ध्रुव uint8_t *) (act_data + ir1 * act_stride);
            भिन्न * restrict dst_row          = (भिन्न *) (dst->data + ((cur_m_start + ir1) * dst_row_size));
            भिन्न * dst_ptr = &dst_row[ct * 32];

            ध्रुव भिन्न * src2_ptr = रिक्त;
            यदि (src2) {
                ध्रुव भिन्न * restrict src2_row = (ध्रुव भिन्न *) ((ध्रुव uint8_t *) src2->data + ((cur_m_start + ir1) * src2_stride));
                src2_ptr = &src2_row[ct * 32];
            }
            mmctx->vec_dot_32x1(ne10, dst_ptr, w_tile, act_col, valid_rows, src2_ptr);
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ct);

        यदि (धकेलो_ct < ct_end) {
            dma_queue_धकेलो(dma_q, dma_make_data(w_tile, src0_row + धकेलो_ct * tile_row_stride),
                           aligned_tile_size, tile_size, tile_size, n_k_tiles_w);
            धकेलो_ct++;
        }
    }
}

स्थिर शून्य hvx_mv_2d_tiled(अचिन्हित पूर्णाङ्क nth, अचिन्हित पूर्णाङ्क ith, शून्य * data) {
    htp_matmul_preamble;

    ध्रुव uint32_t src0_nrows = mmctx->src0_row_end - mmctx->src0_row_start;

    ध्रुव uint32_t src0_start_row  = mmctx->src0_row_start + src0_nrows_per_thread * ith;
    ध्रुव uint32_t src0_end_row    = MIN(src0_start_row + src0_nrows_per_thread, mmctx->src0_row_end);

    संरचना htp_thread_trace * tr = &octx->ctx->trace[ith];

    ध्रुव संरचना htp_mm_kernel_params * kparams = (ध्रुव संरचना htp_mm_kernel_params *) octx->kernel_params;
    ध्रुव uint32_t n_prefetch = kparams->n_prefetch;
    प्रतिज्ञा(n_prefetch >= 2 && n_prefetch <= HTP_MM_MAX_PREFETCH && (n_prefetch & (n_prefetch - 1)) == 0);

    ध्रुव माप_प्रकार dst_row_size  = nb1;
    ध्रुव माप_प्रकार act_row_size = nb11;
    ध्रुव माप_प्रकार act_stride = mmctx->vtcm_act_stride;

    uint8_t * vtcm_dst_ptr  = mmctx->vtcm_dst + mmctx->vtcm_dst_size_per_thread * ith;
    uint8_t * vtcm_src0_ptr = mmctx->vtcm_src0 + mmctx->vtcm_src0_size_per_thread * ith;
    uint8_t * act_data = mmctx->vtcm_act;

    भिन्न * tmp = (भिन्न *) vtcm_dst_ptr;

    ध्रुव dma_addr_t src0_row = src0->data;

    ध्रुव uint8_t * restrict act_col = (ध्रुव uint8_t *) act_data;
    भिन्न * restrict dst_col          = (भिन्न *) dst->data;

    ध्रुव uint32_t tile_size = htp_tensor_get_tile_size(src0->type);
    ध्रुव uint32_t aligned_tile_size = htp_tensor_get_tile_padded_size(src0->type);

    uint32_t n_k_tiles_w = ne00 / htp_tensor_get_tile_k(src0->type);
    uint32_t tile_row_stride = n_k_tiles_w * tile_size;
    uint32_t tile_row_transfer_size_aligned = n_k_tiles_w * aligned_tile_size;

    uint32_t ct_start = src0_start_row / 32;
    uint32_t ct_end   = (src0_end_row + 31) / 32;

    uint32_t धकेलो_ct = ct_start;
    यदि (src0_start_row < src0_end_row) {
        यदि (src2) {
            भिन्न * vtcm_bias_ptr = (भिन्न *) mmctx->vtcm_bias + src0_start_row;
            ध्रुव dma_addr_t src2_addr = src2->data + src0_start_row * परिमाण(भिन्न);
            पूर्णाङ्क slice_size = (पूर्णाङ्क)MIN(src0_end_row, ne0) - (पूर्णाङ्क)src0_start_row;
            यदि (slice_size > 0) {
                dma_queue_धकेलो(dma_q, dma_make_data(vtcm_bias_ptr, src2_addr),
                               slice_size * परिमाण(भिन्न), slice_size * परिमाण(भिन्न), slice_size * परिमाण(भिन्न), 1);
                dma_queue_निकालो_nowait(dma_q);
            }
        }
        प्रति (uint32_t d = 0; d < n_prefetch && धकेलो_ct < ct_end; d++, धकेलो_ct++) {
            dma_queue_धकेलो(dma_q, dma_make_data(vtcm_src0_ptr + d * tile_row_transfer_size_aligned,
                           src0_row + धकेलो_ct * tile_row_stride), aligned_tile_size, tile_size, tile_size, n_k_tiles_w);
        }
    }

    यदि (src0_start_row >= src0_end_row) {
        प्रत्यागम;
    }

    प्रति (uint32_t ct = ct_start; ct < ct_end; ct++) {
        ध्रुव uint8_t * w_tile = (शून्य *) dma_queue_निकालो(dma_q).dst;

        भिन्न * dst_ptr = &tmp[ct * 32 - src0_start_row];
        पूर्णाङ्क valid_rows = (पूर्णाङ्क)ne0 - (पूर्णाङ्क)(ct * 32);
        valid_rows = MIN(32, MAX(0, valid_rows));

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, ct);
        mmctx->vec_dot_32x1(ne10, dst_ptr, w_tile, act_col, valid_rows, रिक्त);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ct);

        यदि (धकेलो_ct < ct_end) {
            dma_queue_धकेलो(dma_q, dma_make_data(w_tile, src0_row + धकेलो_ct * tile_row_stride),
                           aligned_tile_size, tile_size, tile_size, n_k_tiles_w);
            धकेलो_ct++;
        }
    }

    पूर्णाङ्क copy_cnt = (पूर्णाङ्क)MIN(src0_end_row, ne0) - (पूर्णाङ्क)src0_start_row;
    यदि (copy_cnt > 0) {
        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, ct_end);
        यदि (src2) {
            hvx_add_f32_uuu((uint8_t *) &dst_col[src0_start_row],
                            (ध्रुव uint8_t *) tmp,
                            (ध्रुव uint8_t *) ((ध्रुव भिन्न *) mmctx->vtcm_bias + src0_start_row),
                            copy_cnt);
        } अन्यथा {
            hvx_copy_f32_ua((uint8_t *) &dst_col[src0_start_row], (uint8_t *) tmp, copy_cnt);
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ct_end);
    }
}

स्थिर शून्य hvx_mm_nx_2d_tiled(अचिन्हित पूर्णाङ्क nth, अचिन्हित पूर्णाङ्क ith, शून्य * data) {
    संरचना htp_mm_context * mmctx = data;
    संरचना htp_ops_context * octx = mmctx->octx;
    ध्रुव संरचना htp_mm_kernel_params * kparams = (ध्रुव संरचना htp_mm_kernel_params *) octx->kernel_params;
    ध्रुव uint32_t n_weights = kparams->n_weights;

    ध्रुव संरचना htp_tensor * restrict act = octx->src[n_weights]; /* x */
    ध्रुव uint32_t ne10 = act->ne[0];
    ध्रुव uint32_t act_nrows = act->ne[1] * act->ne[2] * act->ne[3];
    ध्रुव माप_प्रकार act_stride = mmctx->vtcm_act_stride;

    uint8_t * restrict vtcm_weight_ptr = mmctx->vtcm_src0 + mmctx->vtcm_src0_size_per_thread * ith;
    uint8_t * restrict act_data        = mmctx->vtcm_act;

    संरचना htp_thread_trace * tr = &octx->ctx->trace[ith];
    ध्रुव uint32_t n_prefetch = kparams->n_prefetch;
    प्रतिज्ञा(n_prefetch >= 2 && n_prefetch <= HTP_MM_MAX_PREFETCH && (n_prefetch & (n_prefetch - 1)) == 0);

    प्रति (uint32_t widx = 0; widx < n_weights; widx++) {
        ध्रुव संरचना htp_tensor * restrict src_w = octx->src[widx];
        ध्रुव संरचना htp_tensor * restrict dst   = octx->dsts[widx];
        यदि (!src_w || !dst) अनुवर्त;
        dma_queue * dma_q = octx->ctx->dma[ith];

        ध्रुव uint32_t tile_size = htp_tensor_get_tile_size(src_w->type);
        ध्रुव uint32_t aligned_tile_size = htp_tensor_get_tile_padded_size(src_w->type);
        ध्रुव uint32_t ne00 = src_w->ne[0];
        ध्रुव uint32_t ne01 = src_w->ne[1];
        ध्रुव माप_प्रकार dst_row_size = dst->nb[1];
        ध्रुव dma_addr_t src_w_row = src_w->data;

        ध्रुव uint32_t n_k_tiles_w = ne00 / htp_tensor_get_tile_k(src_w->type);
        ध्रुव uint32_t tile_row_stride = n_k_tiles_w * tile_size;
        ध्रुव uint32_t tile_row_transfer_size_aligned = n_k_tiles_w * aligned_tile_size;

        uint32_t src0_start_row = 0;
        uint32_t src0_end_row   = ne01;
        यदि (octx->ctx->mdev.गिनती > 1) {
            ध्रुव bool can_split = htp_tensor_can_row_partition(dst, परिमाण(भिन्न));
            ध्रुव संरचना htp_tensor_mdev_range range = htp_tensor_mdev_partition(ne01, can_split ? 32 : 0,
                                                         octx->ctx->mdev.idx, octx->ctx->mdev.गिनती, &octx->ctx->mdev.गिनती_div);
            src0_start_row = range.start;
            src0_end_row   = range.start + range.गिनती;
        }

        ध्रुव uint32_t nrows = src0_end_row - src0_start_row;
        uint32_t src0_nrows_per_thread = fastdiv(nrows + nth - 1, &octx->n_threads_div);
        src0_nrows_per_thread = hex_round_up(src0_nrows_per_thread, 32);

        ध्रुव uint32_t start_row = src0_start_row + src0_nrows_per_thread * ith;
        ध्रुव uint32_t end_row   = MIN(start_row + src0_nrows_per_thread, src0_end_row);
        यदि (start_row >= end_row) अनुवर्त;

        uint32_t ct_start = start_row / 32;
        uint32_t ct_end   = (end_row + 31) / 32;

        uint32_t धकेलो_ct = ct_start;
        प्रति (uint32_t d = 0; d < n_prefetch && धकेलो_ct < ct_end; d++, धकेलो_ct++) {
            dma_queue_धकेलो(dma_q, dma_make_data(vtcm_weight_ptr + d * tile_row_transfer_size_aligned,
                            src_w_row + धकेलो_ct * tile_row_stride), aligned_tile_size, tile_size, tile_size, n_k_tiles_w);
        }

        प्रति (uint32_t ct = ct_start; ct < ct_end; ct++) {
            ध्रुव uint8_t * w_tile = (शून्य *) dma_queue_निकालो(dma_q).dst;
            पूर्णाङ्क valid_rows = (पूर्णाङ्क)dst->ne[0] - (पूर्णाङ्क)(ct * 32);
            valid_rows = MIN(32, MAX(0, valid_rows));

            htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, ct);
            uint32_t ir1 = 0;
            प्रति (; ir1 + 1 < act_nrows; ir1 += 2) {
                ध्रुव uint8_t * restrict act_col0 = (ध्रुव uint8_t *) (act_data + (ir1+0) * act_stride);
                ध्रुव uint8_t * restrict act_col1 = (ध्रुव uint8_t *) (act_data + (ir1+1) * act_stride);

                भिन्न * restrict dst_row0 = (भिन्न *) (dst->data + ((ir1+0) * dst_row_size));
                भिन्न * restrict dst_row1 = (भिन्न *) (dst->data + ((ir1+1) * dst_row_size));
                भिन्न * dst_ptr0 = &dst_row0[ct * 32];
                भिन्न * dst_ptr1 = &dst_row1[ct * 32];

                mmctx->vec_dot_32x2(ne10, dst_ptr0, dst_ptr1, w_tile, act_col0, act_col1, valid_rows, रिक्त, रिक्त);
            }

            प्रति (; ir1 < act_nrows; ++ir1) {
                ध्रुव uint8_t * restrict act_col = (ध्रुव uint8_t *) (act_data + ir1 * act_stride);
                भिन्न * restrict dst_row = (भिन्न *) (dst->data + (ir1 * dst_row_size));
                भिन्न * dst_ptr = &dst_row[ct * 32];
                mmctx->vec_dot_32x1(ne10, dst_ptr, w_tile, act_col, valid_rows, रिक्त);
            }
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ct);

            यदि (धकेलो_ct < ct_end) {
                dma_queue_धकेलो(dma_q, dma_make_data(w_tile, src_w_row + धकेलो_ct * tile_row_stride),
                                aligned_tile_size, tile_size, tile_size, n_k_tiles_w);
                धकेलो_ct++;
            }
        }
    }
}

स्थिर शून्य hvx_mm_transfer_act_dma(
    संरचना htp_ops_context * octx,
    ध्रुव संरचना htp_mm_kernel_params * kparams,
    ध्रुव संरचना htp_tensor * act,
    uint8_t * dst_base,
    माप_प्रकार dst_row_size,
    uint32_t m_start,
    uint32_t m_rows
) {
    यदि (m_rows == 0) {
        प्रत्यागम;
    }

    dma_queue * dma_q = octx->ctx->dma[0];
    ध्रुव uint32_t ne0 = act->ne[0];
    ध्रुव माप_प्रकार elem_size = (act->type == HTP_TYPE_F16) ? परिमाण(__fp16) : परिमाण(भिन्न);
    ध्रुव माप_प्रकार row_bytes = ne0 * elem_size;
    ध्रुव माप_प्रकार act_nb1 = act->nb[1];
    ध्रुव dma_addr_t act_base = act->data;

    ध्रुव bool is_contiguous = (act->nb[2] == act->ne[1] * act_nb1) &&
                               (act->nb[3] == act->ne[2] * act->nb[2]);

    यदि (is_contiguous) {
        ध्रुव dma_addr_t src_addr = act_base + m_start * act_nb1;
        dma_queue_धकेलो(dma_q, dma_make_data(dst_base, src_addr),
                       dst_row_size, act_nb1, row_bytes, m_rows);
        dma_queue_निकालो(dma_q);
    } अन्यथा {
        ध्रुव uint32_t ne12_ne1 = act->ne[2] * act->ne[1];
        ध्रुव bool use_fastdiv = kparams->div_ne12_ne1.mp != 0;
        प्रति (uint32_t ir = 0; ir < m_rows; ++ir) {
            ध्रुव uint32_t ir1 = m_start + ir;
            uint32_t i11, i12, i13;
            यदि (use_fastdiv) {
                i13 = fastdiv(ir1, &kparams->div_ne12_ne1);
                ध्रुव uint32_t rem = ir1 - i13 * ne12_ne1;
                i12 = fastdiv(rem, &kparams->div_ne1);
                i11 = rem - i12 * act->ne[1];
            } अन्यथा {
                i13 = ne12_ne1 ? ir1 / ne12_ne1 : 0;
                ध्रुव uint32_t rem = ir1 - i13 * ne12_ne1;
                i12 = act->ne[1] ? rem / act->ne[1] : 0;
                i11 = rem - i12 * act->ne[1];
            }
            ध्रुव dma_addr_t row_src = act_base + (i11 * act->nb[1] +
                                                   i12 * act->nb[2] +
                                                   i13 * act->nb[3]);
            uint8_t * row_dst = dst_base + ir * dst_row_size;
            dma_queue_धकेलो(dma_q, dma_make_data(row_dst, row_src),
                           dst_row_size, act_nb1, row_bytes, 1);
            dma_queue_निकालो(dma_q);
        }
    }

    यदि (dst_row_size > row_bytes) {
        संरचना htp_thread_trace * tr = &octx->ctx->trace[0];
        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_A_PREP, (uint16_t) m_start);
        ध्रुव uint32_t pad_elems = (dst_row_size - row_bytes) / elem_size;
        यदि (elem_size == परिमाण(भिन्न)) {
            प्रति (uint32_t ir = 0; ir < m_rows; ++ir) {
                hvx_splat_f32_u(dst_base + ir * dst_row_size + row_bytes, 0.0f, pad_elems);
            }
        } अन्यथा {
            प्रति (uint32_t ir = 0; ir < m_rows; ++ir) {
                hvx_splat_f16_u(dst_base + ir * dst_row_size + row_bytes, (_Float16) 0.0f, pad_elems);
            }
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_A_PREP, (uint16_t) m_start);
    }
}

#define QUANTIZE_IMPL(name, log_name, kernel_fn, dst_row_size_व्यंजक)                                        \
स्थिर शून्य name(अचिन्हित पूर्णाङ्क nth, अचिन्हित पूर्णाङ्क ith, शून्य * data) {                                        \
    (शून्य) nth;                                                                                            \
    संरचना htp_mm_context * mmctx = data;                                                                  \
    संरचना htp_ops_context * octx = mmctx->octx;                                                           \
    ध्रुव संरचना htp_tensor * src = mmctx->act;                                                            \
    ध्रुव uint32_t ne0 = src->ne[0];                                                                       \
    ध्रुव uint32_t nrows = mmctx->cur_m_rows ? mmctx->cur_m_rows : mmctx->act_nrows;                       \
    ध्रुव uint32_t nrows_per_thread = mmctx->n_quant_rows_per_thread;                                      \
                                                                                                           \
    ध्रुव uint32_t ir_first = nrows_per_thread * ith;                                                      \
    यदि (ir_first >= nrows) {                                                                               \
        प्रत्यागम;                                                                                            \
    }                                                                                                      \
                                                                                                           \
    संरचना htp_thread_trace * tr = &octx->ctx->trace[ith];                                                 \
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_A_QUANT, ir_first);                                        \
                                                                                                           \
    uint8_t * restrict dst = mmctx->vtcm_act;                                                              \
    ध्रुव uint32_t ir_last = MIN(ir_first + nrows_per_thread, nrows);                                      \
    ध्रुव माप_प्रकार raw_row_size = mmctx->vtcm_act_raw_stride;                                                \
    ध्रुव माप_प्रकार dst_row_size = (dst_row_size_व्यंजक);                                                       \
                                                                                                           \
    ध्रुव uint8_t * restrict src_data = (ध्रुव uint8_t *) mmctx->vtcm_act_raw + (raw_row_size * ir_first); \
    uint8_t * restrict dst_data = (uint8_t *) dst + (dst_row_size * ir_first);                             \
    kernel_fn(src_data, dst_data, रिक्त, ne0, ir_last - ir_first, raw_row_size, dst_row_size);              \
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

स्थिर शून्य quantize_f32_q8_0_tiled_block(अचिन्हित पूर्णाङ्क nth, अचिन्हित पूर्णाङ्क ith, शून्य * data) {
    (शून्य) nth;
    संरचना htp_mm_context * mmctx = data;
    यदि (mmctx->quant_ib_first[ith] >= mmctx->quant_ib_last[ith]) {
        प्रत्यागम;
    }
    संरचना htp_ops_context * octx = mmctx->octx;
    संरचना htp_thread_trace * tr = &octx->ctx->trace[ith];
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_A_QUANT, mmctx->quant_ib_first[ith]);

    ध्रुव संरचना htp_tensor * src = mmctx->act;

    quantize_f32_q8_0_tiled_block_kernel(
        (ध्रुव भिन्न *) mmctx->vtcm_act_raw,
        mmctx->vtcm_act,
        रिक्त,
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

स्थिर शून्य quantize_f32_q8_1_tiled_block(अचिन्हित पूर्णाङ्क nth, अचिन्हित पूर्णाङ्क ith, शून्य * data) {
    (शून्य) nth;
    संरचना htp_mm_context * mmctx = data;
    यदि (mmctx->quant_ib_first[ith] >= mmctx->quant_ib_last[ith]) {
        प्रत्यागम;
    }
    संरचना htp_ops_context * octx = mmctx->octx;
    संरचना htp_thread_trace * tr = &octx->ctx->trace[ith];
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_A_QUANT, mmctx->quant_ib_first[ith]);

    ध्रुव संरचना htp_tensor * src = mmctx->act;

    quantize_f32_q8_1_tiled_block_kernel(
        (ध्रुव भिन्न *) mmctx->vtcm_act_raw,
        mmctx->vtcm_act,
        रिक्त,
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

स्थिर शून्य quantize_f32_q8_1_s16_tiled_block(अचिन्हित पूर्णाङ्क nth, अचिन्हित पूर्णाङ्क ith, शून्य * data) {
    (शून्य) nth;
    संरचना htp_mm_context * mmctx = data;
    यदि (mmctx->quant_ib_first[ith] >= mmctx->quant_ib_last[ith]) {
        प्रत्यागम;
    }
    संरचना htp_ops_context * octx = mmctx->octx;
    संरचना htp_thread_trace * tr = &octx->ctx->trace[ith];
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_A_QUANT, mmctx->quant_ib_first[ith]);

    ध्रुव संरचना htp_tensor * src = mmctx->act;

    quantize_f32_q8_1_s16_tiled_block_kernel(
        (ध्रुव भिन्न *) mmctx->vtcm_act_raw,
        mmctx->vtcm_act,
        रिक्त,
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

स्थिर शून्य quantize_f32_f16_tiled_block(अचिन्हित पूर्णाङ्क nth, अचिन्हित पूर्णाङ्क ith, शून्य * data) {
    (शून्य) nth;
    संरचना htp_mm_context * mmctx = data;
    यदि (mmctx->quant_ib_first[ith] >= mmctx->quant_ib_last[ith]) {
        प्रत्यागम;
    }
    संरचना htp_ops_context * octx = mmctx->octx;
    संरचना htp_thread_trace * tr = &octx->ctx->trace[ith];
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_A_QUANT, mmctx->quant_ib_first[ith]);

    ध्रुव संरचना htp_tensor * src = mmctx->act;

    quantize_f32_f16_tiled_block_kernel(
        (ध्रुव भिन्न *) mmctx->vtcm_act_raw,
        mmctx->vtcm_act,
        रिक्त,
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

स्थिर शून्य quantize_f16_f16_tiled_block(अचिन्हित पूर्णाङ्क nth, अचिन्हित पूर्णाङ्क ith, शून्य * data) {
    (शून्य) nth;
    संरचना htp_mm_context * mmctx = data;
    यदि (mmctx->quant_ib_first[ith] >= mmctx->quant_ib_last[ith]) {
        प्रत्यागम;
    }
    संरचना htp_ops_context * octx = mmctx->octx;
    संरचना htp_thread_trace * tr = &octx->ctx->trace[ith];
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_A_QUANT, mmctx->quant_ib_first[ith]);

    ध्रुव संरचना htp_tensor * src = mmctx->act;

    quantize_f16_f16_tiled_block_kernel(
        (ध्रुव __fp16 *) mmctx->vtcm_act_raw,
        mmctx->vtcm_act,
        रिक्त,
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
स्थिर अन्तःस्थ worker_callback_t htp_mm_act_quant_row_func(पूर्णाङ्क weight_type, पूर्णाङ्क act_type) {
    यदि (weight_type == HTP_TYPE_F16) {
        प्रत्यागम (act_type == HTP_TYPE_F16) ? quantize_f16_f16_tiled : quantize_f32_f16_tiled;
    }
    यदि (weight_type == HTP_TYPE_Q2_K) {
        प्रत्यागम quantize_f32_q8_1_s16_tiled;
    }
    प्रत्यागम htp_mm_weight_has_offset(weight_type) ? quantize_f32_q8_1_tiled : quantize_f32_q8_0_tiled;
}

स्थिर अन्तःस्थ worker_callback_t htp_mm_act_quant_block_func(पूर्णाङ्क weight_type, पूर्णाङ्क act_type) {
    यदि (weight_type == HTP_TYPE_F16) {
        प्रत्यागम (act_type == HTP_TYPE_F16) ? quantize_f16_f16_tiled_block : quantize_f32_f16_tiled_block;
    }
    यदि (weight_type == HTP_TYPE_Q2_K) {
        प्रत्यागम quantize_f32_q8_1_s16_tiled_block;
    }
    प्रत्यागम htp_mm_weight_has_offset(weight_type) ? quantize_f32_q8_1_tiled_block : quantize_f32_q8_0_tiled_block;
}

स्थिर शून्य hvx_mm_4d_tiled(अचिन्हित पूर्णाङ्क nth, अचिन्हित पूर्णाङ्क ith, शून्य * data) {
    htp_matmul_preamble;

    ध्रुव uint32_t src0_nrows = mmctx->src0_row_end - mmctx->src0_row_start;
    ध्रुव uint32_t cur_m_rows = mmctx->cur_m_rows ? mmctx->cur_m_rows : (ne11 * ne12 * ne13);
    ध्रुव uint32_t cur_m_start = mmctx->cur_m_start;

    ध्रुव uint32_t src0_start_row  = mmctx->src0_row_start + src0_nrows_per_thread * ith;
    ध्रुव uint32_t src0_end_row    = MIN(src0_start_row + src0_nrows_per_thread, mmctx->src0_row_end);

    संरचना htp_thread_trace * tr = &octx->ctx->trace[ith];

    ध्रुव संरचना htp_mm_kernel_params * kparams = (ध्रुव संरचना htp_mm_kernel_params *) octx->kernel_params;
    ध्रुव uint32_t n_prefetch = kparams->n_prefetch;
    प्रतिज्ञा(n_prefetch >= 2 && n_prefetch <= HTP_MM_MAX_PREFETCH && (n_prefetch & (n_prefetch - 1)) == 0);

    ध्रुव माप_प्रकार dst_row_size = nb1;
    ध्रुव माप_प्रकार act_stride = mmctx->vtcm_act_stride;

    uint8_t * restrict vtcm_src0_ptr = mmctx->vtcm_src0 + mmctx->vtcm_src0_size_per_thread * ith;
    uint8_t * restrict act_data = mmctx->vtcm_act;

    ध्रुव uint32_t tile_size = htp_tensor_get_tile_size(src0->type);
    ध्रुव uint32_t aligned_tile_size = htp_tensor_get_tile_padded_size(src0->type);

    ध्रुव uint32_t n_k_tiles_w = ne00 / htp_tensor_get_tile_k(src0->type);
    ध्रुव uint32_t tile_row_stride = n_k_tiles_w * tile_size;
    ध्रुव uint32_t tile_row_transfer_size_aligned = n_k_tiles_w * aligned_tile_size;
    ध्रुव uint32_t src0_slice_stride = ((ne01 + 31) / 32) * tile_row_stride;

    ध्रुव uint32_t ct_start = src0_start_row / 32;
    ध्रुव uint32_t ct_end   = (src0_end_row + 31) / 32;

    यदि (src0_start_row >= src0_end_row || cur_m_rows == 0) {
        प्रत्यागम;
    }

    ध्रुव uint32_t total_batches = ne12 * ne13;
    ध्रुव uint32_t b_start = fastdiv(cur_m_start, &kparams->div_ne1);
    uint32_t b_end = fastdiv(cur_m_start + cur_m_rows + ne11 - 1, &kparams->div_ne1);
    b_end = MIN(b_end, total_batches);

    uint32_t b_grp_start = b_start;
    यावत् (b_grp_start < b_end) {
        ध्रुव uint32_t b3 = fastdiv(b_grp_start, &kparams->div_ne12);
        ध्रुव uint32_t b2 = b_grp_start - b3 * ne12;
        ध्रुव uint32_t i02 = fastdiv(b2, &kparams->div_r2);
        ध्रुव uint32_t i03 = fastdiv(b3, &kparams->div_r3);

        uint32_t b_grp_end = b_grp_start + 1;
        यावत् (b_grp_end < b_end) {
            ध्रुव uint32_t cur_b3 = fastdiv(b_grp_end, &kparams->div_ne12);
            ध्रुव uint32_t cur_b2 = b_grp_end - cur_b3 * ne12;
            ध्रुव uint32_t cur_i02 = fastdiv(cur_b2, &kparams->div_r2);
            ध्रुव uint32_t cur_i03 = fastdiv(cur_b3, &kparams->div_r3);
            यदि (cur_i02 != i02 || cur_i03 != i03) {
                भङ्ग;
            }
            b_grp_end++;
        }

        ध्रुव uint32_t slice_idx = i03 * ne02 + i02;
        ध्रुव dma_addr_t src0_slice = src0->data + (माप_प्रकार) slice_idx * src0_slice_stride;

        uint32_t धकेलो_ct = ct_start;
        प्रति (uint32_t d = 0; d < n_prefetch && धकेलो_ct < ct_end; d++, धकेलो_ct++) {
            dma_queue_धकेलो(dma_q, dma_make_data(vtcm_src0_ptr + d * tile_row_transfer_size_aligned,
                           src0_slice + (माप_प्रकार) धकेलो_ct * tile_row_stride),
                           aligned_tile_size, tile_size, tile_size, n_k_tiles_w);
        }

        प्रति (uint32_t ct = ct_start; ct < ct_end; ct++) {
            ध्रुव uint8_t * w_tile = (शून्य *) dma_queue_निकालो(dma_q).dst;

            पूर्णाङ्क valid_rows = (पूर्णाङ्क)ne0 - (पूर्णाङ्क)(ct * 32);
            valid_rows = MIN(32, MAX(0, valid_rows));

            htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, ct);
            प्रति (uint32_t b = b_grp_start; b < b_grp_end; b++) {
                ध्रुव uint32_t b_m_start = b * ne11;
                ध्रुव uint32_t m_first = MAX(cur_m_start, b_m_start);
                ध्रुव uint32_t m_last  = MIN(cur_m_start + cur_m_rows, b_m_start + ne11);
                यदि (m_first >= m_last) अनुवर्त;

                ध्रुव uint32_t cur_b3 = fastdiv(b, &kparams->div_ne12);
                ध्रुव uint32_t cur_b2 = b - cur_b3 * ne12;
                uint8_t * dst_batch_base = (uint8_t *) dst->data + (माप_प्रकार) cur_b2 * nb2 + (माप_प्रकार) cur_b3 * nb3;

                ध्रुव uint32_t chunk_m_offset = m_first - cur_m_start;
                ध्रुव uint32_t dst_m_offset   = m_first - b_m_start;
                ध्रुव uint32_t batch_nrows    = m_last - m_first;

                uint32_t ir1 = 0;
                प्रति (; ir1 + 1 < batch_nrows; ir1 += 2) {
                    ध्रुव uint8_t * restrict act_col0 = (ध्रुव uint8_t *) (act_data + (chunk_m_offset + ir1 + 0) * act_stride);
                    ध्रुव uint8_t * restrict act_col1 = (ध्रुव uint8_t *) (act_data + (chunk_m_offset + ir1 + 1) * act_stride);
                    भिन्न * restrict dst_row0 = (भिन्न *) (dst_batch_base + (dst_m_offset + ir1 + 0) * dst_row_size);
                    भिन्न * restrict dst_row1 = (भिन्न *) (dst_batch_base + (dst_m_offset + ir1 + 1) * dst_row_size);
                    भिन्न * dst_ptr0 = &dst_row0[ct * 32];
                    भिन्न * dst_ptr1 = &dst_row1[ct * 32];
                    mmctx->vec_dot_32x2(ne10, dst_ptr0, dst_ptr1, w_tile, act_col0, act_col1, valid_rows, रिक्त, रिक्त);
                }
                प्रति (; ir1 < batch_nrows; ++ir1) {
                    ध्रुव uint8_t * restrict act_col = (ध्रुव uint8_t *) (act_data + (chunk_m_offset + ir1) * act_stride);
                    भिन्न * restrict dst_row = (भिन्न *) (dst_batch_base + (dst_m_offset + ir1) * dst_row_size);
                    भिन्न * dst_ptr = &dst_row[ct * 32];
                    mmctx->vec_dot_32x1(ne10, dst_ptr, w_tile, act_col, valid_rows, रिक्त);
                }
            }
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ct);

            यदि (धकेलो_ct < ct_end) {
                dma_queue_धकेलो(dma_q, dma_make_data(w_tile, src0_slice + (माप_प्रकार) धकेलो_ct * tile_row_stride),
                               aligned_tile_size, tile_size, tile_size, n_k_tiles_w);
                धकेलो_ct++;
            }
        }
        b_grp_start = b_grp_end;
    }
    यदि (src2) {
        hvx_tensor_add_f32_grid(dst, src2, cur_m_start, cur_m_start + cur_m_rows, src0_start_row, src0_end_row, &kparams->div_ne12_ne1, &kparams->div_ne1);
    }
}

स्थिर शून्य hvx_mm_2d_flat(अचिन्हित पूर्णाङ्क nth, अचिन्हित पूर्णाङ्क ith, शून्य * data) {
    htp_matmul_preamble;

    ध्रुव संरचना htp_mm_kernel_params * kparams = (ध्रुव संरचना htp_mm_kernel_params *) octx->kernel_params;
    ध्रुव uint32_t n_prefetch = kparams->n_prefetch;
    प्रतिज्ञा(n_prefetch >= 2 && n_prefetch <= HTP_MM_MAX_PREFETCH && (n_prefetch & (n_prefetch - 1)) == 0);
    ध्रुव uint32_t prefetch_mask = n_prefetch - 1;

    ध्रुव uint32_t src0_nrows = mmctx->src0_row_end - mmctx->src0_row_start;  // src0 rows
    ध्रुव uint32_t act_nrows = mmctx->cur_m_rows ? mmctx->cur_m_rows : mmctx->act_nrows;
    ध्रुव uint32_t cur_m_start = mmctx->cur_m_start;

    ध्रुव uint32_t src0_start_row  = mmctx->src0_row_start + src0_nrows_per_thread * ith;
    ध्रुव uint32_t src0_end_row    = MIN(src0_start_row + src0_nrows_per_thread, mmctx->src0_row_end);
    ध्रुव uint32_t src0_end_row_x2 = src0_start_row + ((src0_end_row - src0_start_row) & ~1U);

    संरचना htp_thread_trace * tr = &octx->ctx->trace[ith];

    ध्रुव माप_प्रकार dst_row_size  = nb1;
    ध्रुव माप_प्रकार src0_row_size = nb01;
    ध्रुव माप_प्रकार act_row_size = nb11;

    ध्रुव माप_प्रकार src0_stride = mmctx->vtcm_src0_stride;
    ध्रुव माप_प्रकार act_stride = mmctx->vtcm_act_stride;

    // Per-thread VTCMs for all tensors
    uint8_t * restrict vtcm_dst_ptr  = mmctx->vtcm_dst  + mmctx->vtcm_dst_size_per_thread  * ith;
    uint8_t * restrict vtcm_src0_ptr = mmctx->vtcm_src0 + mmctx->vtcm_src0_size_per_thread * ith;
    uint8_t * restrict act_data      = mmctx->vtcm_act;

    ध्रुव dma_addr_t src0_row = src0->data;


    // Prefill vtcm with src0 rows
    यदि (src0_start_row < src0_end_row) {
        प्रति (uint32_t ir0 = src0_start_row; ir0 < src0_end_row_x2; ir0 += 2) {
            ध्रुव पूर्णाङ्क is0 = (ir0 - src0_start_row);
            यदि (is0 >= (पूर्णाङ्क)n_prefetch) {
                भङ्ग;
            }
            dma_queue_धकेलो(dma_q, dma_make_data(vtcm_src0_ptr + is0 * src0_stride, src0_row + ir0 * src0_row_size),
                           src0_stride, src0_row_size, src0_row_size, 2);
        }
    }

    यदि (src0_start_row >= src0_end_row) {
        प्रत्यागम;
    }

    // Process src0 rows
    प्रति (uint32_t ir0 = src0_start_row; ir0 < src0_end_row_x2; ir0 += 2) {
        ध्रुव uint8_t * ss0 = (शून्य *) dma_queue_निकालो(dma_q).dst;

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, ir0);
        // Process act columns in pairs (2x2 tiling)
        uint32_t ir1 = 0;
        प्रति (; ir1 + 1 < act_nrows; ir1 += 2) {
            ध्रुव uint8_t * restrict act_col0 = (ध्रुव uint8_t *) (act_data + (ir1+0) * act_stride);
            ध्रुव uint8_t * restrict act_col1 = (ध्रुव uint8_t *) (act_data + (ir1+1) * act_stride);
            भिन्न * restrict dst_row0 = (भिन्न *) (dst->data + ((cur_m_start + ir1+0) * dst_row_size));
            भिन्न * restrict dst_row1 = (भिन्न *) (dst->data + ((cur_m_start + ir1+1) * dst_row_size));
            mmctx->vec_dot_2x2(ne00, &dst_row0[ir0], &dst_row1[ir0], ss0, ss0 + src0_stride, act_col0, act_col1);
        }

        // Handle remaining act rows (fallback to 2x1)
        प्रति (; ir1 < act_nrows; ++ir1) {
            ध्रुव uint8_t * restrict act_col = (ध्रुव uint8_t *) (act_data + ir1 * act_stride);
            भिन्न * restrict dst_row          = (भिन्न *) (dst->data + ((cur_m_start + ir1) * dst_row_size));
            mmctx->vec_dot_2x1(ne00, &dst_row[ir0], ss0, ss0 + src0_stride, act_col);
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ir0);

        // Prefetch next (n + vtcm_nrows) row
        ध्रुव पूर्णाङ्क pr0 = (ir0 + n_prefetch);
        ध्रुव पूर्णाङ्क is0 = (pr0 - src0_start_row) & prefetch_mask;
        यदि (pr0 < src0_end_row_x2) {
            dma_queue_धकेलो(dma_q, dma_make_data(vtcm_src0_ptr + is0 * src0_stride, src0_row + pr0 * src0_row_size),
                           src0_stride, src0_row_size, src0_row_size, 2);
        }
    }

    // Process the last row (if any)
    यदि (src0_end_row != src0_end_row_x2) {
        uint32_t  ir0 = src0_end_row_x2;
        ध्रुव पूर्णाङ्क is0 = (ir0 - src0_start_row) & prefetch_mask;
        dma_queue_धकेलो(dma_q, dma_make_data(vtcm_src0_ptr + is0 * src0_stride, src0_row + ir0 * src0_row_size),
                       src0_stride, src0_row_size, src0_row_size, 1);
        ध्रुव uint8_t * ss0 = (शून्य *) dma_queue_निकालो(dma_q).dst;

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, ir0);
        #pragma unroll(2)
        प्रति (uint32_t ir1 = 0; ir1 < act_nrows; ++ir1) {
            ध्रुव uint8_t * restrict act_col = (ध्रुव uint8_t *) (act_data + ir1 * act_stride);
            भिन्न * restrict dst_row          = (भिन्न *) (dst->data + ((cur_m_start + ir1) * dst_row_size));
            mmctx->vec_dot_1x1(ne00, &dst_row[ir0], ss0, act_col);
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ir0);
    }
    यदि (src2) {
        hvx_tensor_add_f32_grid(dst, src2, cur_m_start, cur_m_start + act_nrows, src0_start_row, src0_end_row, &kparams->div_ne12_ne1, &kparams->div_ne1);
    }
}

स्थिर शून्य hvx_mv_2d_flat(अचिन्हित पूर्णाङ्क nth, अचिन्हित पूर्णाङ्क ith, शून्य * data) {
    htp_matmul_preamble;

    ध्रुव uint32_t src0_nrows = mmctx->src0_row_end - mmctx->src0_row_start;

    ध्रुव uint32_t src0_start_row  = mmctx->src0_row_start + src0_nrows_per_thread * ith;
    ध्रुव uint32_t src0_end_row    = MIN(src0_start_row + src0_nrows_per_thread, mmctx->src0_row_end);

    संरचना htp_thread_trace * tr = &octx->ctx->trace[ith];

    ध्रुव माप_प्रकार dst_row_size  = nb1;
    ध्रुव माप_प्रकार src0_row_size = nb01;
    ध्रुव माप_प्रकार act_row_size  = nb11;

    ध्रुव माप_प्रकार src0_stride = mmctx->vtcm_src0_stride;
    ध्रुव माप_प्रकार act_stride  = mmctx->vtcm_act_stride;

    // Per-thread VTCMs for all tensors
    uint8_t * vtcm_dst_ptr  = mmctx->vtcm_dst  + mmctx->vtcm_dst_size_per_thread  * ith;
    uint8_t * vtcm_src0_ptr = mmctx->vtcm_src0 + mmctx->vtcm_src0_size_per_thread * ith;
    uint8_t * act_data      = mmctx->vtcm_act;

    भिन्न * tmp = (भिन्न *) vtcm_dst_ptr;

    ध्रुव dma_addr_t src0_row = src0->data;
    ध्रुव uint8_t * restrict act_col = (ध्रुव uint8_t *) act_data;
    भिन्न * restrict dst_col         = (भिन्न *) dst->data;

    ध्रुव uint32_t src0_end_row_x2 = src0_start_row + ((src0_end_row - src0_start_row) & ~1U);

    ध्रुव संरचना htp_mm_kernel_params * kparams = (ध्रुव संरचना htp_mm_kernel_params *) octx->kernel_params;
    ध्रुव uint32_t n_prefetch = kparams->n_prefetch;
    प्रतिज्ञा(n_prefetch >= 2 && n_prefetch <= HTP_MM_MAX_PREFETCH && (n_prefetch & (n_prefetch - 1)) == 0);
    ध्रुव uint32_t prefetch_mask = n_prefetch - 1;

    // Prefill vtcm with 2x src0 rows
    यदि (src0_start_row < src0_end_row) {
        यदि (src2) {
            भिन्न * vtcm_bias_ptr = (भिन्न *) mmctx->vtcm_bias + src0_start_row;
            ध्रुव dma_addr_t src2_addr = src2->data + src0_start_row * परिमाण(भिन्न);
            पूर्णाङ्क slice_size = (पूर्णाङ्क)src0_end_row - (पूर्णाङ्क)src0_start_row;
            यदि (slice_size > 0) {
                dma_queue_धकेलो(dma_q, dma_make_data(vtcm_bias_ptr, src2_addr),
                               slice_size * परिमाण(भिन्न), slice_size * परिमाण(भिन्न), slice_size * परिमाण(भिन्न), 1);
                dma_queue_निकालो_nowait(dma_q);
            }
        }
        प्रति (uint32_t ir0 = src0_start_row; ir0 < src0_end_row_x2; ir0 += 2) {
            ध्रुव uint32_t is0 = (ir0 - src0_start_row);
            यदि (is0 >= n_prefetch) {
                भङ्ग;
            }
            dma_queue_धकेलो(dma_q, dma_make_data(vtcm_src0_ptr + is0 * src0_stride, src0_row + ir0 * src0_row_size),
                           src0_stride, src0_row_size, src0_row_size, 2);
        }
    }


    यदि (src0_start_row >= src0_end_row) {
        प्रत्यागम;
    }

    // Process src0 rows
    प्रति (uint32_t ir0 = src0_start_row; ir0 < src0_end_row_x2; ir0 += 2) {
        ध्रुव uint8_t * ss0 = (शून्य *) dma_queue_निकालो(dma_q).dst;
        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, ir0);
        mmctx->vec_dot_2x1(ne00, &tmp[ir0 - src0_start_row], ss0, ss0 + src0_stride, act_col);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ir0);

        // Prefetch next (n + vtcm_nrows) row
        ध्रुव uint32_t pr0 = (ir0 + n_prefetch);
        ध्रुव uint32_t is0 = (pr0 - src0_start_row) & prefetch_mask;
        यदि (pr0 < src0_end_row_x2) {
            dma_queue_धकेलो(dma_q, dma_make_data(vtcm_src0_ptr + is0 * src0_stride, src0_row + pr0 * src0_row_size),
                           src0_stride, src0_row_size, src0_row_size, 2);
        }
    }

    // Process the last row (if any)
    यदि (src0_end_row != src0_end_row_x2) {
        ध्रुव uint32_t ir0 = src0_end_row_x2;
        ध्रुव uint32_t is0 = (ir0 - src0_start_row) & prefetch_mask;
        dma_queue_धकेलो(dma_q, dma_make_data(vtcm_src0_ptr + is0 * src0_stride, src0_row + ir0 * src0_row_size),
                       src0_stride, src0_row_size, src0_row_size, 1);
        ध्रुव uint8_t * ss0 = (शून्य *) dma_queue_निकालो(dma_q).dst;
        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, ir0);
        mmctx->vec_dot_1x1(ne00, &tmp[ir0 - src0_start_row], ss0, act_col);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ir0);
    }

    पूर्णाङ्क copy_cnt = src0_end_row - src0_start_row;
    यदि (copy_cnt > 0) {
        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, src0_end_row);
        यदि (src2) {
            hvx_add_f32_uuu((uint8_t *) &dst_col[src0_start_row],
                            (ध्रुव uint8_t *) tmp,
                            (ध्रुव uint8_t *) ((ध्रुव भिन्न *) mmctx->vtcm_bias + src0_start_row),
                            copy_cnt);
        } अन्यथा {
            hvx_copy_f32_ua((uint8_t *) &dst_col[src0_start_row], (uint8_t *) tmp, copy_cnt);
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, src0_end_row);
    }
}

स्थिर शून्य hvx_mm_4d_flat(अचिन्हित पूर्णाङ्क nth, अचिन्हित पूर्णाङ्क ith, शून्य * data) {
    htp_matmul_preamble;

    ध्रुव संरचना htp_mm_kernel_params * kparams = (ध्रुव संरचना htp_mm_kernel_params *) octx->kernel_params;
    ध्रुव uint32_t n_prefetch = kparams->n_prefetch;
    प्रतिज्ञा(n_prefetch >= 2 && n_prefetch <= HTP_MM_MAX_PREFETCH && (n_prefetch & (n_prefetch - 1)) == 0);
    ध्रुव uint32_t prefetch_mask = n_prefetch - 1;

    ध्रुव uint32_t src0_nrows = mmctx->src0_row_end - mmctx->src0_row_start;
    ध्रुव uint32_t cur_m_rows = mmctx->cur_m_rows ? mmctx->cur_m_rows : (ne11 * ne12 * ne13);
    ध्रुव uint32_t cur_m_start = mmctx->cur_m_start;

    ध्रुव uint32_t src0_start_row  = mmctx->src0_row_start + src0_nrows_per_thread * ith;
    ध्रुव uint32_t src0_end_row    = MIN(src0_start_row + src0_nrows_per_thread, mmctx->src0_row_end);
    ध्रुव uint32_t src0_end_row_x2 = src0_start_row + ((src0_end_row - src0_start_row) & ~1U);

    संरचना htp_thread_trace * tr = &octx->ctx->trace[ith];

    ध्रुव माप_प्रकार dst_row_size  = nb1;
    ध्रुव माप_प्रकार src0_row_size = nb01;
    ध्रुव माप_प्रकार src0_stride = mmctx->vtcm_src0_stride;
    ध्रुव माप_प्रकार act_stride  = mmctx->vtcm_act_stride;

    uint8_t * restrict vtcm_src0_ptr = mmctx->vtcm_src0 + mmctx->vtcm_src0_size_per_thread * ith;
    uint8_t * restrict act_data      = mmctx->vtcm_act;


    यदि (src0_start_row >= src0_end_row || cur_m_rows == 0) {
        प्रत्यागम;
    }

    ध्रुव uint32_t total_batches = ne12 * ne13;
    ध्रुव uint32_t b_start = fastdiv(cur_m_start, &kparams->div_ne1);
    uint32_t b_end = fastdiv(cur_m_start + cur_m_rows + ne11 - 1, &kparams->div_ne1);
    b_end = MIN(b_end, total_batches);

    uint32_t b_grp_start = b_start;
    यावत् (b_grp_start < b_end) {
        ध्रुव uint32_t b3 = fastdiv(b_grp_start, &kparams->div_ne12);
        ध्रुव uint32_t b2 = b_grp_start - b3 * ne12;
        ध्रुव uint32_t i02 = fastdiv(b2, &kparams->div_r2);
        ध्रुव uint32_t i03 = fastdiv(b3, &kparams->div_r3);

        uint32_t b_grp_end = b_grp_start + 1;
        यावत् (b_grp_end < b_end) {
            ध्रुव uint32_t cur_b3 = fastdiv(b_grp_end, &kparams->div_ne12);
            ध्रुव uint32_t cur_b2 = b_grp_end - cur_b3 * ne12;
            ध्रुव uint32_t cur_i02 = fastdiv(cur_b2, &kparams->div_r2);
            ध्रुव uint32_t cur_i03 = fastdiv(cur_b3, &kparams->div_r3);
            यदि (cur_i02 != i02 || cur_i03 != i03) {
                भङ्ग;
            }
            b_grp_end++;
        }

        ध्रुव dma_addr_t src0_row = src0->data + ((माप_प्रकार) i02 * nb02 + (माप_प्रकार) i03 * nb03);

        प्रति (uint32_t ir0 = src0_start_row; ir0 < src0_end_row_x2; ir0 += 2) {
            ध्रुव पूर्णाङ्क is0 = (ir0 - src0_start_row);
            यदि (is0 >= (पूर्णाङ्क)n_prefetch) {
                भङ्ग;
            }
            dma_queue_धकेलो(dma_q, dma_make_data(vtcm_src0_ptr + is0 * src0_stride, src0_row + (माप_प्रकार) ir0 * src0_row_size),
                           src0_stride, src0_row_size, src0_row_size, 2);
        }

        प्रति (uint32_t ir0 = src0_start_row; ir0 < src0_end_row_x2; ir0 += 2) {
            ध्रुव uint8_t * ss0 = (शून्य *) dma_queue_निकालो(dma_q).dst;

            htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, ir0);
            प्रति (uint32_t b = b_grp_start; b < b_grp_end; b++) {
                ध्रुव uint32_t b_m_start = b * ne11;
                ध्रुव uint32_t m_first = MAX(cur_m_start, b_m_start);
                ध्रुव uint32_t m_last  = MIN(cur_m_start + cur_m_rows, b_m_start + ne11);
                यदि (m_first >= m_last) अनुवर्त;

                ध्रुव uint32_t cur_b3 = fastdiv(b, &kparams->div_ne12);
                ध्रुव uint32_t cur_b2 = b - cur_b3 * ne12;
                uint8_t * dst_batch_base = (uint8_t *) dst->data + (माप_प्रकार) cur_b2 * nb2 + (माप_प्रकार) cur_b3 * nb3;

                ध्रुव uint32_t chunk_m_offset = m_first - cur_m_start;
                ध्रुव uint32_t dst_m_offset   = m_first - b_m_start;
                ध्रुव uint32_t batch_nrows    = m_last - m_first;

                uint32_t ir1 = 0;
                प्रति (; ir1 + 1 < batch_nrows; ir1 += 2) {
                    ध्रुव uint8_t * restrict act_col0 = (ध्रुव uint8_t *) (act_data + (chunk_m_offset + ir1 + 0) * act_stride);
                    ध्रुव uint8_t * restrict act_col1 = (ध्रुव uint8_t *) (act_data + (chunk_m_offset + ir1 + 1) * act_stride);
                    भिन्न * restrict dst_row0 = (भिन्न *) (dst_batch_base + (dst_m_offset + ir1 + 0) * dst_row_size);
                    भिन्न * restrict dst_row1 = (भिन्न *) (dst_batch_base + (dst_m_offset + ir1 + 1) * dst_row_size);
                    mmctx->vec_dot_2x2(ne00, &dst_row0[ir0], &dst_row1[ir0], ss0, ss0 + src0_stride, act_col0, act_col1);
                }
                प्रति (; ir1 < batch_nrows; ++ir1) {
                    ध्रुव uint8_t * restrict act_col = (ध्रुव uint8_t *) (act_data + (chunk_m_offset + ir1) * act_stride);
                    भिन्न * restrict dst_row          = (भिन्न *) (dst_batch_base + (dst_m_offset + ir1) * dst_row_size);
                    mmctx->vec_dot_2x1(ne00, &dst_row[ir0], ss0, ss0 + src0_stride, act_col);
                }
            }
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ir0);

            ध्रुव पूर्णाङ्क pr0 = (ir0 + n_prefetch);
            ध्रुव पूर्णाङ्क is0 = (pr0 - src0_start_row) & prefetch_mask;
            यदि (pr0 < src0_end_row_x2) {
                dma_queue_धकेलो(dma_q, dma_make_data(vtcm_src0_ptr + is0 * src0_stride, src0_row + (माप_प्रकार) pr0 * src0_row_size),
                               src0_stride, src0_row_size, src0_row_size, 2);
            }
        }

        यदि (src0_end_row != src0_end_row_x2) {
            uint32_t ir0 = src0_end_row_x2;
            ध्रुव पूर्णाङ्क is0 = (ir0 - src0_start_row) & prefetch_mask;
            dma_queue_धकेलो(dma_q, dma_make_data(vtcm_src0_ptr + is0 * src0_stride, src0_row + (माप_प्रकार) ir0 * src0_row_size),
                           src0_stride, src0_row_size, src0_row_size, 1);
            ध्रुव uint8_t * ss0 = (शून्य *) dma_queue_निकालो(dma_q).dst;

            htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, ir0);
            प्रति (uint32_t b = b_grp_start; b < b_grp_end; b++) {
                ध्रुव uint32_t b_m_start = b * ne11;
                ध्रुव uint32_t m_first = MAX(cur_m_start, b_m_start);
                ध्रुव uint32_t m_last  = MIN(cur_m_start + cur_m_rows, b_m_start + ne11);
                यदि (m_first >= m_last) अनुवर्त;

                ध्रुव uint32_t cur_b3 = fastdiv(b, &kparams->div_ne12);
                ध्रुव uint32_t cur_b2 = b - cur_b3 * ne12;
                uint8_t * dst_batch_base = (uint8_t *) dst->data + (माप_प्रकार) cur_b2 * nb2 + (माप_प्रकार) cur_b3 * nb3;

                ध्रुव uint32_t chunk_m_offset = m_first - cur_m_start;
                ध्रुव uint32_t dst_m_offset   = m_first - b_m_start;
                ध्रुव uint32_t batch_nrows    = m_last - m_first;

                प्रति (uint32_t ir1 = 0; ir1 < batch_nrows; ++ir1) {
                    ध्रुव uint8_t * restrict act_col = (ध्रुव uint8_t *) (act_data + (chunk_m_offset + ir1) * act_stride);
                    भिन्न * restrict dst_row          = (भिन्न *) (dst_batch_base + (dst_m_offset + ir1) * dst_row_size);
                    mmctx->vec_dot_1x1(ne00, &dst_row[ir0], ss0, act_col);
                }
            }
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ir0);
        }

        b_grp_start = b_grp_end;
    }

    यदि (src2) {
        hvx_tensor_add_f32_grid(dst, src2, cur_m_start, cur_m_start + cur_m_rows, src0_start_row, src0_end_row, &kparams->div_ne12_ne1, &kparams->div_ne1);
    }
}

#define MMID_MATRIX_ROW(row_id, i1) matrix_rows[(row_id) * mmctx->mapping_stride + (i1)]

स्थिर शून्य hvx_mm_id_tiled(अचिन्हित पूर्णाङ्क nth, अचिन्हित पूर्णाङ्क ith, शून्य * data) {
    htp_matmul_preamble;

    ध्रुव संरचना htp_tensor * restrict ids = octx->src[2];

    ध्रुव uint32_t src0_nrows      = mmctx->src0_row_end - mmctx->src0_row_start;  // src0 rows per expert
    ध्रुव uint32_t act_nrows       = ne11;
    ध्रुव uint32_t src0_start_row  = mmctx->src0_row_start + src0_nrows_per_thread * ith;
    ध्रुव uint32_t src0_end_row    = MIN(src0_start_row + src0_nrows_per_thread, mmctx->src0_row_end);


    यदि (src0_start_row >= src0_end_row) {
        प्रत्यागम;
    }

    संरचना htp_thread_trace * tr = &octx->ctx->trace[ith];

    ध्रुव संरचना htp_mm_kernel_params * kparams = (ध्रुव संरचना htp_mm_kernel_params *) octx->kernel_params;
    ध्रुव uint32_t n_prefetch = kparams->n_prefetch;
    प्रतिज्ञा(n_prefetch >= 2 && n_prefetch <= HTP_MM_MAX_PREFETCH && (n_prefetch & (n_prefetch - 1)) == 0);

    ध्रुव uint32_t n_ids = ids->ne[0];  // n_expert_used
    ध्रुव uint32_t n_as  = ne02;        // n_expert

    ध्रुव uint32_t *                matrix_row_counts = mmctx->matrix_row_counts;
    ध्रुव संरचना mmid_row_mapping * matrix_rows       = mmctx->matrix_rows;

    ध्रुव माप_प्रकार dst_row_size  = nb1;
    ध्रुव माप_प्रकार act_row_size  = htp_mm_q8_0_tiled_row_size(ne10);

    ध्रुव माप_प्रकार act_stride = mmctx->vtcm_act_stride;

    // Per-thread VTCMs for all tensors
    uint8_t * restrict vtcm_src0_ptr = mmctx->vtcm_src0 + mmctx->vtcm_src0_size_per_thread * ith;
    uint8_t * restrict act_data = mmctx->vtcm_act;

    प्रति (uint32_t cur_a = 0; cur_a < n_as; ++cur_a) {
        ध्रुव int32_t cne1 = matrix_row_counts[cur_a];
        यदि (cne1 == 0) {
            अनुवर्त;
        }

        ध्रुव dma_addr_t src0_row = src0->data + cur_a * nb02;

        ध्रुव uint32_t tile_size = htp_tensor_get_tile_size(src0->type);
        ध्रुव uint32_t aligned_tile_size = htp_tensor_get_tile_padded_size(src0->type);
        ध्रुव uint32_t n_k_tiles_w = ne00 / htp_tensor_get_tile_k(src0->type);
        ध्रुव uint32_t tile_row_stride = n_k_tiles_w * tile_size;
        ध्रुव uint32_t tile_row_transfer_size_aligned = n_k_tiles_w * aligned_tile_size;

        ध्रुव uint32_t ct_start = src0_start_row / 32;
        ध्रुव uint32_t ct_end   = (src0_end_row + 31) / 32;

        uint32_t धकेलो_ct = ct_start;
        प्रति (uint32_t d = 0; d < n_prefetch && धकेलो_ct < ct_end; d++, धकेलो_ct++) {
            dma_queue_धकेलो(dma_q, dma_make_data(vtcm_src0_ptr + d * tile_row_transfer_size_aligned, src0_row + धकेलो_ct * tile_row_stride),
                           aligned_tile_size, tile_size, tile_size, n_k_tiles_w);
        }

        प्रति (uint32_t ct = ct_start; ct < ct_end; ct++) {
            ध्रुव uint8_t * w_tile = (शून्य *) dma_queue_निकालो(dma_q).dst;

            पूर्णाङ्क valid_rows = (पूर्णाङ्क)ne0 - (पूर्णाङ्क)(ct * 32);
            valid_rows = MIN(32, MAX(0, valid_rows));

            htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, ct);
            प्रति (uint32_t cid = 0; cid < cne1; ++cid) {
                संरचना mmid_row_mapping row_mapping = MMID_MATRIX_ROW(cur_a, cid);
                ध्रुव पूर्णाङ्क               rm1         = row_mapping.i1;  // expert idx
                ध्रुव पूर्णाङ्क               rm2         = row_mapping.i2;  // token idx

                ध्रुव uint32_t ir1 = fastmodulo(rm1, ne11, &mmctx->mm_div_ne11);        // act row idx
                ध्रुव uint8_t * restrict act_col = (ध्रुव uint8_t *) (act_data + (ir1 + rm2 * ne11 + 0) * act_stride);
                भिन्न * restrict dst_row = (भिन्न *) (dst->data + (rm1 * nb1 + rm2 * nb2 + 0));

                mmctx->vec_dot_32x1(ne10, &dst_row[ct * 32], w_tile, act_col, valid_rows, रिक्त);
            }
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ct);

            यदि (धकेलो_ct < ct_end) {
                dma_queue_धकेलो(dma_q, dma_make_data(w_tile, src0_row + धकेलो_ct * tile_row_stride),
                               aligned_tile_size, tile_size, tile_size, n_k_tiles_w);
                धकेलो_ct++;
            }
        }
    }
}

स्थिर शून्य hvx_mv_id_tiled(अचिन्हित पूर्णाङ्क nth, अचिन्हित पूर्णाङ्क ith, शून्य * data) {
    htp_matmul_preamble;

    ध्रुव संरचना htp_tensor * restrict ids = octx->src[2];

    ध्रुव uint32_t src0_nrows      = mmctx->src0_row_end - mmctx->src0_row_start;  // src0 rows per expert
    ध्रुव uint32_t src0_start_row  = mmctx->src0_row_start + src0_nrows_per_thread * ith;
    ध्रुव uint32_t src0_end_row    = MIN(src0_start_row + src0_nrows_per_thread, mmctx->src0_row_end);


    यदि (src0_start_row >= src0_end_row) {
        प्रत्यागम;
    }

    संरचना htp_thread_trace * tr = &octx->ctx->trace[ith];

    ध्रुव संरचना htp_mm_kernel_params * kparams = (ध्रुव संरचना htp_mm_kernel_params *) octx->kernel_params;
    ध्रुव uint32_t n_prefetch = kparams->n_prefetch;
    प्रतिज्ञा(n_prefetch >= 2 && n_prefetch <= HTP_MM_MAX_PREFETCH && (n_prefetch & (n_prefetch - 1)) == 0);

    प्रतिज्ञा(ne13 % ne03 == 0);

    ध्रुव माप_प्रकार dst_row_size  = nb1;
    ध्रुव माप_प्रकार act_row_size = htp_mm_q8_0_tiled_row_size(ne10);

    ध्रुव uint32_t n_aids = src2->ne[0];  // num activated experts
    ध्रुव uint32_t n_ids  = ne02;         // num experts

    // Per-thread VTCMs for all tensors
    uint8_t * restrict vtcm_src0_ptr = mmctx->vtcm_src0 + mmctx->vtcm_src0_size_per_thread * ith;
    uint8_t * restrict act_data = mmctx->vtcm_act;

    प्रति (uint32_t ie1 = 0; ie1 < n_aids; ++ie1) {  // for each expert
        ध्रुव int32_t eid = *(ध्रुव int32_t *) ((ध्रुव uint8_t *) src2->data + ie1 * src2->nb[0]);
        यदि (eid < 0) {
            अनुवर्त;
        }
        प्रतिज्ञा(eid < (int32_t) n_ids);

        ध्रुव dma_addr_t src0_row = src0->data + eid * nb02;
        ध्रुव uint8_t * restrict act_col = (ध्रुव uint8_t *) act_data;
        भिन्न * restrict dst_row          = (भिन्न *) (dst->data + ie1 * nb1);

        ध्रुव uint32_t tile_size = htp_tensor_get_tile_size(src0->type);
        ध्रुव uint32_t aligned_tile_size = htp_tensor_get_tile_padded_size(src0->type);
        ध्रुव uint32_t n_k_tiles_w = ne00 / htp_tensor_get_tile_k(src0->type);
        ध्रुव uint32_t tile_row_stride = n_k_tiles_w * tile_size;
        ध्रुव uint32_t tile_row_transfer_size_aligned = n_k_tiles_w * aligned_tile_size;

        ध्रुव uint32_t ct_start = src0_start_row / 32;
        ध्रुव uint32_t ct_end   = (src0_end_row + 31) / 32;

        uint32_t धकेलो_ct = ct_start;
        प्रति (uint32_t d = 0; d < n_prefetch && धकेलो_ct < ct_end; d++, धकेलो_ct++) {
            dma_queue_धकेलो(dma_q, dma_make_data(vtcm_src0_ptr + d * tile_row_transfer_size_aligned, src0_row + धकेलो_ct * tile_row_stride),
                           aligned_tile_size, tile_size, tile_size, n_k_tiles_w);
        }

        प्रति (uint32_t ct = ct_start; ct < ct_end; ct++) {
            ध्रुव uint8_t * w_tile = (शून्य *) dma_queue_निकालो(dma_q).dst;

            पूर्णाङ्क valid_rows = (पूर्णाङ्क)ne0 - (पूर्णाङ्क)(ct * 32);
            valid_rows = MIN(32, MAX(0, valid_rows));

            htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, ct);
            mmctx->vec_dot_32x1(ne10, &dst_row[ct * 32], w_tile, act_col, valid_rows, रिक्त);
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ct);

            यदि (धकेलो_ct < ct_end) {
                dma_queue_धकेलो(dma_q, dma_make_data(w_tile, src0_row + धकेलो_ct * tile_row_stride),
                               aligned_tile_size, tile_size, tile_size, n_k_tiles_w);
                धकेलो_ct++;
            }
        }
    }
}

स्थिर शून्य hvx_mv_id_nx_tiled(अचिन्हित पूर्णाङ्क nth, अचिन्हित पूर्णाङ्क ith, शून्य * data) {
    संरचना htp_mm_context * mmctx = (संरचना htp_mm_context *) data;
    संरचना htp_ops_context * octx = mmctx->octx;
    ध्रुव संरचना htp_mm_kernel_params * kparams = (ध्रुव संरचना htp_mm_kernel_params *) octx->kernel_params;
    ध्रुव uint32_t n_weights      = kparams->n_weights;
    ध्रुव संरचना htp_tensor * restrict src0 = octx->src[0];
    ध्रुव संरचना htp_tensor * restrict act  = octx->src[n_weights];
    ध्रुव संरचना htp_tensor * restrict ids  = octx->src[n_weights + 1];


    संरचना htp_thread_trace * tr = &octx->ctx->trace[ith];

    ध्रुव uint32_t n_prefetch = kparams->n_prefetch;
    प्रतिज्ञा(n_prefetch >= 2 && n_prefetch <= HTP_MM_MAX_PREFETCH && (n_prefetch & (n_prefetch - 1)) == 0);

    ध्रुव uint32_t n_aids = ids->ne[0];
    ध्रुव uint32_t n_ids  = src0->ne[2];

    uint8_t * restrict vtcm_src0_ptr = mmctx->vtcm_src0 + mmctx->vtcm_src0_size_per_thread * ith;
    uint8_t * restrict act_data = mmctx->vtcm_act;

    प्रति (uint32_t ie1 = 0; ie1 < n_aids; ++ie1) {
        ध्रुव int32_t eid = *(ध्रुव int32_t *) ((ध्रुव uint8_t *) ids->data + ie1 * ids->nb[0]);
        यदि (eid < 0) अनुवर्त;
        प्रतिज्ञा(eid < (int32_t) n_ids);

        प्रति (uint32_t p = 0; p < n_weights; ++p) {
            ध्रुव संरचना htp_tensor * restrict src_w = octx->src[p];
            ध्रुव संरचना htp_tensor * restrict dst   = octx->dsts[p];
            यदि (!src_w || !dst) अनुवर्त;
            dma_queue * dma_q = octx->ctx->dma[ith];

            ध्रुव uint32_t ne01 = src_w->ne[1];
            uint32_t start_row = 0;
            uint32_t end_row   = ne01;
            यदि (octx->ctx->mdev.गिनती > 1) {
                ध्रुव bool can_split = htp_tensor_can_row_partition(dst, परिमाण(भिन्न));
                ध्रुव संरचना htp_tensor_mdev_range range = htp_tensor_mdev_partition(ne01, can_split ? 32 : 0, octx->ctx->mdev.idx, octx->ctx->mdev.गिनती, &octx->ctx->mdev.गिनती_div);
                start_row = range.start;
                end_row   = range.start + range.गिनती;
            }

            ध्रुव uint32_t nrows = end_row - start_row;
            uint32_t src0_nrows_per_thread = fastdiv(nrows + nth - 1, &octx->n_threads_div);
            src0_nrows_per_thread = hex_round_up(src0_nrows_per_thread, 32);

            ध्रुव uint32_t src0_start_row = start_row + src0_nrows_per_thread * ith;
            ध्रुव uint32_t src0_end_row   = MIN(src0_start_row + src0_nrows_per_thread, end_row);
            यदि (src0_start_row >= src0_end_row) अनुवर्त;

            ध्रुव dma_addr_t src0_row = src_w->data + eid * src_w->nb[2];
            ध्रुव uint8_t * restrict act_col = (ध्रुव uint8_t *) act_data;
            भिन्न * restrict dst_row = (भिन्न *) (dst->data + ie1 * dst->nb[1]);

            ध्रुव uint32_t tile_size = htp_tensor_get_tile_size(src_w->type);
            ध्रुव uint32_t aligned_tile_size = htp_tensor_get_tile_padded_size(src_w->type);
            ध्रुव uint32_t n_k_tiles_w = src_w->ne[0] / htp_tensor_get_tile_k(src_w->type);
            ध्रुव uint32_t tile_row_stride = n_k_tiles_w * tile_size;
            ध्रुव uint32_t tile_row_transfer_size_aligned = n_k_tiles_w * aligned_tile_size;

            ध्रुव uint32_t ct_start = src0_start_row / 32;
            ध्रुव uint32_t ct_end   = (src0_end_row + 31) / 32;

            uint32_t धकेलो_ct = ct_start;
            प्रति (uint32_t d = 0; d < n_prefetch && धकेलो_ct < ct_end; d++, धकेलो_ct++) {
                dma_queue_धकेलो(dma_q, dma_make_data(vtcm_src0_ptr + d * tile_row_transfer_size_aligned, src0_row + धकेलो_ct * tile_row_stride),
                               aligned_tile_size, tile_size, tile_size, n_k_tiles_w);
            }

            प्रति (uint32_t ct = ct_start; ct < ct_end; ct++) {
                ध्रुव uint8_t * w_tile = (शून्य *) dma_queue_निकालो(dma_q).dst;

                पूर्णाङ्क valid_rows = (पूर्णाङ्क)dst->ne[0] - (पूर्णाङ्क)(ct * 32);
                valid_rows = MIN(32, MAX(0, valid_rows));

                htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, ct);
                mmctx->vec_dot_32x1(act->ne[0], &dst_row[ct * 32], w_tile, act_col, valid_rows, रिक्त);
                htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ct);

                यदि (धकेलो_ct < ct_end) {
                    dma_queue_धकेलो(dma_q, dma_make_data(w_tile, src0_row + धकेलो_ct * tile_row_stride),
                                    aligned_tile_size, tile_size, tile_size, n_k_tiles_w);
                    धकेलो_ct++;
                }
            }
        }
    }
}

स्थिर शून्य hvx_mm_id_nx_tiled(अचिन्हित पूर्णाङ्क nth, अचिन्हित पूर्णाङ्क ith, शून्य * data) {
    संरचना htp_mm_context * mmctx = (संरचना htp_mm_context *) data;
    संरचना htp_ops_context * octx = mmctx->octx;
    ध्रुव संरचना htp_mm_kernel_params * kparams = (ध्रुव संरचना htp_mm_kernel_params *) octx->kernel_params;
    ध्रुव uint32_t n_weights      = kparams->n_weights;
    ध्रुव संरचना htp_tensor * restrict src0 = octx->src[0];
    ध्रुव संरचना htp_tensor * restrict act  = octx->src[n_weights];
    ध्रुव संरचना htp_tensor * restrict ids  = octx->src[n_weights + 1];


    संरचना htp_thread_trace * tr = &octx->ctx->trace[ith];

    ध्रुव uint32_t n_prefetch = kparams->n_prefetch;
    प्रतिज्ञा(n_prefetch >= 2 && n_prefetch <= HTP_MM_MAX_PREFETCH && (n_prefetch & (n_prefetch - 1)) == 0);

    ध्रुव uint32_t n_as = src0->ne[2];

    ध्रुव uint32_t * matrix_row_counts = mmctx->matrix_row_counts;
    ध्रुव संरचना mmid_row_mapping * matrix_rows = mmctx->matrix_rows;

    ध्रुव माप_प्रकार act_stride = mmctx->vtcm_act_stride;

    uint8_t * restrict vtcm_src0_ptr = mmctx->vtcm_src0 + mmctx->vtcm_src0_size_per_thread * ith;
    uint8_t * restrict act_data = mmctx->vtcm_act;

    प्रति (uint32_t cur_a = 0; cur_a < n_as; ++cur_a) {
        ध्रुव int32_t cne1 = matrix_row_counts[cur_a];
        यदि (cne1 == 0) अनुवर्त;

        प्रति (uint32_t p = 0; p < n_weights; ++p) {
            ध्रुव संरचना htp_tensor * restrict src_w = octx->src[p];
            ध्रुव संरचना htp_tensor * restrict dst   = octx->dsts[p];
            यदि (!src_w || !dst) अनुवर्त;
            dma_queue * dma_q = octx->ctx->dma[ith];

            ध्रुव uint32_t ne01 = src_w->ne[1];
            uint32_t start_row = 0;
            uint32_t end_row   = ne01;
            यदि (octx->ctx->mdev.गिनती > 1) {
                ध्रुव bool can_split = htp_tensor_can_row_partition(dst, परिमाण(भिन्न));
                ध्रुव संरचना htp_tensor_mdev_range range = htp_tensor_mdev_partition(ne01, can_split ? 32 : 0, octx->ctx->mdev.idx, octx->ctx->mdev.गिनती, &octx->ctx->mdev.गिनती_div);
                start_row = range.start;
                end_row   = range.start + range.गिनती;
            }

            ध्रुव uint32_t nrows = end_row - start_row;
            uint32_t src0_nrows_per_thread = fastdiv(nrows + nth - 1, &octx->n_threads_div);
            src0_nrows_per_thread = hex_round_up(src0_nrows_per_thread, 32);

            ध्रुव uint32_t src0_start_row = start_row + src0_nrows_per_thread * ith;
            ध्रुव uint32_t src0_end_row   = MIN(src0_start_row + src0_nrows_per_thread, end_row);
            यदि (src0_start_row >= src0_end_row) अनुवर्त;

            ध्रुव dma_addr_t src0_row = src_w->data + cur_a * src_w->nb[2];

            ध्रुव uint32_t tile_size = htp_tensor_get_tile_size(src_w->type);
            ध्रुव uint32_t aligned_tile_size = htp_tensor_get_tile_padded_size(src_w->type);
            ध्रुव uint32_t n_k_tiles_w = src_w->ne[0] / htp_tensor_get_tile_k(src_w->type);
            ध्रुव uint32_t tile_row_stride = n_k_tiles_w * tile_size;
            ध्रुव uint32_t tile_row_transfer_size_aligned = n_k_tiles_w * aligned_tile_size;

            ध्रुव uint32_t ct_start = src0_start_row / 32;
            ध्रुव uint32_t ct_end   = (src0_end_row + 31) / 32;

            uint32_t धकेलो_ct = ct_start;
            प्रति (uint32_t d = 0; d < n_prefetch && धकेलो_ct < ct_end; d++, धकेलो_ct++) {
                dma_queue_धकेलो(dma_q, dma_make_data(vtcm_src0_ptr + d * tile_row_transfer_size_aligned, src0_row + धकेलो_ct * tile_row_stride),
                               aligned_tile_size, tile_size, tile_size, n_k_tiles_w);
            }

            प्रति (uint32_t ct = ct_start; ct < ct_end; ct++) {
                ध्रुव uint8_t * w_tile = (शून्य *) dma_queue_निकालो(dma_q).dst;

                पूर्णाङ्क valid_rows = (पूर्णाङ्क)dst->ne[0] - (पूर्णाङ्क)(ct * 32);
                valid_rows = MIN(32, MAX(0, valid_rows));

                htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, ct);
                प्रति (uint32_t cid = 0; cid < (uint32_t) cne1; ++cid) {
                    संरचना mmid_row_mapping row_mapping = MMID_MATRIX_ROW(cur_a, cid);
                    ध्रुव पूर्णाङ्क rm1 = row_mapping.i1;
                    ध्रुव पूर्णाङ्क rm2 = row_mapping.i2;

                    ध्रुव uint32_t ir1 = fastmodulo(rm1, act->ne[1], &mmctx->mm_div_ne11);
                    ध्रुव uint8_t * restrict act_col = (ध्रुव uint8_t *) (act_data + (ir1 + rm2 * act->ne[1]) * act_stride);
                    भिन्न * restrict dst_row = (भिन्न *) (dst->data + (rm1 * dst->nb[1] + rm2 * dst->nb[2]));

                    mmctx->vec_dot_32x1(act->ne[0], &dst_row[ct * 32], w_tile, act_col, valid_rows, रिक्त);
                }
                htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ct);

                यदि (धकेलो_ct < ct_end) {
                    dma_queue_धकेलो(dma_q, dma_make_data(w_tile, src0_row + धकेलो_ct * tile_row_stride),
                                    aligned_tile_size, tile_size, tile_size, n_k_tiles_w);
                    धकेलो_ct++;
                }
            }
        }
    }
}

स्थिर पूर्णाङ्क hvx_mm_init_tiled_vec_dot(संरचना htp_mm_context * mmctx, गणना htp_data_type type) {
    विकल्प (type) {
        पक्ष HTP_TYPE_Q4_0:
            mmctx->type         = "q4_0_tiled-f32";
            mmctx->vec_dot_32x1 = tiled_vec_dot_q4_0_32x1;
            mmctx->vec_dot_32x2 = tiled_vec_dot_q4_0_32x2;
            प्रत्यागम 0;
        पक्ष HTP_TYPE_Q4_1:
        पक्ष HTP_TYPE_Q4_K:
            mmctx->type         = "q4_1_tiled-f32";
            mmctx->vec_dot_32x1 = tiled_vec_dot_q4_1_32x1;
            mmctx->vec_dot_32x2 = tiled_vec_dot_q4_1_32x2;
            प्रत्यागम 0;
        पक्ष HTP_TYPE_Q8_0:
            mmctx->type         = "q8_0_tiled-f32";
            mmctx->vec_dot_32x1 = tiled_vec_dot_q8_0_32x1;
            mmctx->vec_dot_32x2 = tiled_vec_dot_q8_0_32x2;
            प्रत्यागम 0;
        पक्ष HTP_TYPE_Q5_K:
            mmctx->type         = "q5_k_tiled-f32";
            mmctx->vec_dot_32x1 = tiled_vec_dot_q5_k_32x1;
            mmctx->vec_dot_32x2 = tiled_vec_dot_q5_k_32x2;
            प्रत्यागम 0;
        पक्ष HTP_TYPE_Q6_K:
            mmctx->type         = "q6_k_tiled-f32";
            mmctx->vec_dot_32x1 = tiled_vec_dot_q6_k_32x1;
            mmctx->vec_dot_32x2 = tiled_vec_dot_q6_k_32x2;
            प्रत्यागम 0;
        पक्ष HTP_TYPE_Q3_K:
            mmctx->type         = "q3_k_tiled-f32";
            mmctx->vec_dot_32x1 = tiled_vec_dot_q3_k_32x1;
            mmctx->vec_dot_32x2 = tiled_vec_dot_q3_k_32x2;
            प्रत्यागम 0;
        पक्ष HTP_TYPE_Q2_K:
            mmctx->type         = "q2_k_tiled-f32";
            mmctx->vec_dot_32x1 = tiled_vec_dot_q2_k_32x1;
            mmctx->vec_dot_32x2 = tiled_vec_dot_q2_k_32x2;
            प्रत्यागम 0;
        पक्ष HTP_TYPE_Q2_0:
            mmctx->type         = "q2_0_tiled-f32";
            mmctx->vec_dot_32x1 = tiled_vec_dot_q2_0_32x1;
            mmctx->vec_dot_32x2 = tiled_vec_dot_q2_0_32x2;
            प्रत्यागम 0;
        पक्ष HTP_TYPE_IQ4_NL:
            mmctx->type         = "iq4nl_tiled-f32";
            mmctx->vec_dot_32x1 = tiled_vec_dot_iq4nl_32x1;
            mmctx->vec_dot_32x2 = tiled_vec_dot_iq4nl_32x2;
            प्रत्यागम 0;
        पक्ष HTP_TYPE_MXFP4:
            mmctx->type         = "mxfp4_tiled-f32";
            mmctx->vec_dot_32x1 = tiled_vec_dot_mxfp4_32x1;
            mmctx->vec_dot_32x2 = tiled_vec_dot_mxfp4_32x2;
            प्रत्यागम 0;
        पक्ष HTP_TYPE_Q1_0:
            mmctx->type         = "q1_0_tiled-f32";
            mmctx->vec_dot_32x1 = tiled_vec_dot_q1_0_32x1;
            mmctx->vec_dot_32x2 = tiled_vec_dot_q1_0_32x2;
            प्रत्यागम 0;
        पक्ष HTP_TYPE_F16:
            mmctx->type         = "f16_tiled-f16";
            mmctx->vec_dot_32x1 = tiled_vec_dot_f16_32x1;
            mmctx->vec_dot_32x2 = tiled_vec_dot_f16_32x2;
            प्रत्यागम 0;
        स्वभाव:
            प्रत्यागम -1;
    }
}

स्थिर पूर्णाङ्क hvx_mm_matmul(संरचना htp_ops_context * octx) {
    htp_matmul_tensors_preamble;

    संरचना htp_thread_trace * tr = &octx->ctx->trace[0];
    htp_trace_event_start(tr, HTP_TRACE_EVT_INIT, 0);

    संरचना htp_mm_context mmctx_struct = {0};
    संरचना htp_mm_context * mmctx = &mmctx_struct;
    mmctx->octx = octx;
    mmctx->act = act;

    ध्रुव संरचना htp_mm_kernel_params * kparams = (ध्रुव संरचना htp_mm_kernel_params *) octx->kernel_params;

    ध्रुव uint32_t src0_nrows = ne01;
    ध्रुव uint32_t act_nrows  = ne11 * ne12 * ne13;
    mmctx->act_nrows = act_nrows;

    uint32_t src0_row_start = 0;
    uint32_t src0_row_end   = src0_nrows;

    यदि (octx->ctx->mdev.गिनती > 1) {
        ध्रुव bool can_split = htp_tensor_can_row_partition(dst, परिमाण(भिन्न));
        ध्रुव संरचना htp_tensor_mdev_range range = htp_tensor_mdev_partition(src0_nrows, can_split ? 32 : 0, octx->ctx->mdev.idx, octx->ctx->mdev.गिनती, &octx->ctx->mdev.गिनती_div);
        src0_row_start = range.start;
        src0_row_end   = range.start + range.गिनती;
    }

    यदि (src0_row_start >= src0_row_end) {
        प्रत्यागम HTP_STATUS_OK;
    }

    ध्रुव uint32_t nrows = src0_row_end - src0_row_start;
    mmctx->src0_row_start = src0_row_start;
    mmctx->src0_row_end   = src0_row_end;

    ध्रुव bool is_tiled = htp_tensor_is_tiled(src0);

    // Compute src0_nrows_per_thread
    mmctx->src0_nrows_per_thread  = fastdiv(nrows + octx->n_threads - 1, &octx->n_threads_div);
    यदि (is_tiled) {
        mmctx->src0_nrows_per_thread = hex_round_up(mmctx->src0_nrows_per_thread, 32);
    } अन्यथा {
        mmctx->src0_nrows_per_thread += (mmctx->src0_nrows_per_thread & 1); // round up to even
    }

    ध्रुव माप_प्रकार src0_row_size = nb01;
    ध्रुव माप_प्रकार dst_row_size  = nb1;
    माप_प्रकार       act_row_size  = nb11;

    ध्रुव माप_प्रकार src0_row_size_padded = hex_round_up(src0_row_size, 128);

    worker_callback_t quant_task_func;
    worker_callback_t matmul_job_func;
    uint32_t n_quant_tasks = 1;
    ध्रुव bool is_batched = (ne12 > 1 || ne13 > 1 || ne02 > 1 || ne03 > 1);
    यदि (is_batched) {
        matmul_job_func = is_tiled ? hvx_mm_4d_tiled : hvx_mm_4d_flat;
    } अन्यथा यदि (act_nrows > 1) {
        matmul_job_func = is_tiled ? hvx_mm_2d_tiled : hvx_mm_2d_flat;
    } अन्यथा {
        matmul_job_func = is_tiled ? hvx_mv_2d_tiled : hvx_mv_2d_flat;
    }

    bool need_quant = true;

    विकल्प (kparams->kernel_type) {
        पक्ष HTP_MM_KERNEL_HVX_F16_F16_VTCM:
            quant_task_func        = (act->type == HTP_TYPE_F32) ? quantize_f32_f16 : quantize_f16_f16;
            need_quant             = (act->type == HTP_TYPE_F32);
            mmctx->type            = (act->type == HTP_TYPE_F32) ? "f32-f16" : "f16-f16";
            mmctx->vec_dot_1x1     = flat_vec_dot_f16_f16_aa_1x1;
            mmctx->vec_dot_2x1     = flat_vec_dot_f16_f16_aa_2x1;
            mmctx->vec_dot_2x2     = flat_vec_dot_f16_f16_aa_2x2;
            act_row_size           = kparams->act_row_size;
            भङ्ग;

        पक्ष HTP_MM_KERNEL_HVX_F32_F32_VTCM:
            quant_task_func        = रिक्त;
            need_quant             = false;
            mmctx->type            = "f32-f32";
            mmctx->vec_dot_1x1     = flat_vec_dot_f32_f32_aa_1x1;
            mmctx->vec_dot_2x1     = flat_vec_dot_f32_f32_aa_2x1;
            mmctx->vec_dot_2x2     = flat_vec_dot_f32_f32_aa_2x2;
            act_row_size           = kparams->act_row_size;
            भङ्ग;

        पक्ष HTP_MM_KERNEL_HVX_BF16_F32_VTCM:
            quant_task_func        = रिक्त;
            need_quant             = false;
            mmctx->type            = "bf16-f32";
            mmctx->vec_dot_1x1     = flat_vec_dot_bf16_f32_aa_1x1;
            mmctx->vec_dot_2x1     = flat_vec_dot_bf16_f32_aa_2x1;
            mmctx->vec_dot_2x2     = flat_vec_dot_bf16_f32_aa_2x2;
            act_row_size           = kparams->act_row_size;
            भङ्ग;

        पक्ष HTP_MM_KERNEL_HVX_QUANT_BLOCK:
        पक्ष HTP_MM_KERNEL_HVX_QUANT_ROW:
        स्वभाव: {
            यदि (hvx_mm_init_tiled_vec_dot(mmctx, src0->type) != 0) {
                प्रत्यागम HTP_STATUS_NO_SUPPORT;
            }

            ध्रुव uint32_t qk = QK_Q8_0_TILED;
            ध्रुव uint32_t nb = (ne10 + qk - 1) / qk;
            ध्रुव uint32_t total_nb = act_nrows * nb;

            यदि (act_nrows < octx->n_threads && !is_batched) {
                n_quant_tasks = MIN(total_nb, octx->n_threads);
                quant_task_func = htp_mm_act_quant_block_func(src0->type, act->type);
                प्रति (uint32_t ith = 0; ith < n_quant_tasks; ++ith) {
                    uint32_t ib_first = (total_nb * ith) / n_quant_tasks;
                    uint32_t ib_last  = (total_nb * (ith + 1)) / n_quant_tasks;
                    mmctx->quant_ib_first[ith] = ib_first;
                    mmctx->quant_ib_last[ith]  = ib_last;
                    mmctx->quant_r[ith]        = ib_first / nb;
                    mmctx->quant_c[ith]        = ib_first % nb;
                }
            } अन्यथा {
                n_quant_tasks = MIN(act_nrows, octx->n_threads);
                quant_task_func = htp_mm_act_quant_row_func(src0->type, act->type);
            }
            act_row_size = kparams->act_row_size;
            भङ्ग;
        }
    }

    ध्रुव uint32_t m_chunk = (kparams->m_chunk > 0 && (uint32_t) kparams->m_chunk < act_nrows)
                           ? (uint32_t) kparams->m_chunk : act_nrows;
    ध्रुव uint32_t m_layout_rows = m_chunk;

    संरचना htp_mm_hvx_vtcm_layout L;
    htp_mm_hvx_vtcm_layout_build(&L, kparams->kernel_type, src0->type, ne10, m_layout_rows, octx->n_threads,
                                 dst_row_size, src0_row_size, act_row_size, src2 ? src2->nb[1] : 0, kparams->n_prefetch, false, false);

    यदि (kparams->kernel_type == HTP_MM_KERNEL_HVX_F16_F16_VTCM ||
        kparams->kernel_type == HTP_MM_KERNEL_HVX_F32_F32_VTCM ||
        kparams->kernel_type == HTP_MM_KERNEL_HVX_BF16_F32_VTCM ||
        kparams->kernel_type == HTP_MM_KERNEL_HVX_QUANT_ROW ||
        kparams->kernel_type == HTP_MM_KERNEL_HVX_QUANT_BLOCK) {
        mmctx->vtcm_act_size_per_thread = L.act_bytes;
    } अन्यथा {
        mmctx->vtcm_act_size_per_thread = fastdiv(L.act_bytes, &octx->n_threads_div);
    }

    mmctx->vtcm_src0_size_per_thread = fastdiv(L.src0_bytes, &octx->n_threads_div);
    mmctx->vtcm_dst_size_per_thread  = fastdiv(L.dst_bytes, &octx->n_threads_div);

    ध्रुव माप_प्रकार vtcm_size = L.total_bytes;

    FARF(ऊपरी_2, "matmul-%s : src0-vtcm-size %zu act-vtcm-size %zu dst-vtcm-size %zu (%zu)\n", mmctx->type,
         L.src0_bytes, L.act_bytes, L.dst_bytes, vtcm_size);

    FARF(ऊपरी_2, "matmul-%s : %ux%ux%ux%u * %ux%ux%ux%u-> %ux%ux%ux%u (0x%p, 0x%p, 0x%p)\n", mmctx->type, src0->ne[0],
         src0->ne[1], src0->ne[2], src0->ne[3], act->ne[0], act->ne[1], act->ne[2], act->ne[3], dst->ne[0],
         dst->ne[1], dst->ne[2], dst->ne[3], src0->data, act->data, dst->data);

    यदि (octx->ctx->vtcm_size < vtcm_size) {
        FARF(ERROR, "matmul-%s : current VTCM reservation %zu is too small, needed %zu\n", mmctx->type,
             octx->ctx->vtcm_size, vtcm_size);
        प्रत्यागम HTP_STATUS_VTCM_TOO_SMALL;
    }

    uint8_t * ध्रुव base = (uint8_t *) octx->ctx->vtcm_base;
    mmctx->vtcm_act      = VTCM_LAYOUT_PTR(uint8_t, base, L.off_act);
    mmctx->vtcm_src0     = VTCM_LAYOUT_PTR(uint8_t, base, L.off_src0);
    mmctx->vtcm_bias     = VTCM_LAYOUT_PTR(uint8_t, base, L.off_bias);
    mmctx->vtcm_dst      = VTCM_LAYOUT_PTR(uint8_t, base, L.off_dst);
    mmctx->vtcm_act_raw  = VTCM_LAYOUT_PTR(uint8_t, base, L.off_act_raw);

    mmctx->vtcm_src0_stride = src0_row_size_padded;
    mmctx->vtcm_act_stride  = act_row_size;
    यदि (kparams->kernel_type == HTP_MM_KERNEL_HVX_QUANT_BLOCK || kparams->kernel_type == HTP_MM_KERNEL_HVX_QUANT_ROW) {
        mmctx->vtcm_act_raw_stride = hex_round_up(ne10 * परिमाण(भिन्न), QK_Q8_0_TILED * परिमाण(भिन्न));
    } अन्यथा यदि (kparams->kernel_type == HTP_MM_KERNEL_HVX_F16_F16_VTCM) {
        mmctx->vtcm_act_raw_stride = hex_round_up(ne10 * परिमाण(भिन्न), 128);
    } अन्यथा {
        mmctx->vtcm_act_raw_stride = 0;
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_INIT, 0);

    यदि (kparams->m_chunk > 0 && (uint32_t) kparams->m_chunk < act_nrows) {
        प्रति (uint32_t m_start = 0; m_start < act_nrows; m_start += m_chunk) {
            ध्रुव uint32_t cur_m_rows = MIN(act_nrows - m_start, m_chunk);
            mmctx->cur_m_start = m_start;
            mmctx->cur_m_rows  = cur_m_rows;

            यदि (need_quant) {
                hvx_mm_transfer_act_dma(octx, kparams, act, mmctx->vtcm_act_raw, mmctx->vtcm_act_raw_stride, m_start, cur_m_rows);

                ध्रुव uint32_t qk = QK_Q8_0_TILED;
                ध्रुव uint32_t nb = (ne10 + qk - 1) / qk;
                ध्रुव uint32_t total_nb = cur_m_rows * nb;
                uint32_t quant_tasks;
                work_queue_func_t q_func;
                यदि (cur_m_rows < octx->n_threads && (kparams->kernel_type == HTP_MM_KERNEL_HVX_QUANT_BLOCK || kparams->kernel_type == HTP_MM_KERNEL_HVX_QUANT_ROW)) {
                    quant_tasks = MIN(total_nb, octx->n_threads);
                    q_func = htp_mm_act_quant_block_func(src0->type, act->type);
                    प्रति (uint32_t ith = 0; ith < quant_tasks; ++ith) {
                        uint32_t ib_first = (total_nb * ith) / quant_tasks;
                        uint32_t ib_last  = (total_nb * (ith + 1)) / quant_tasks;
                        mmctx->quant_ib_first[ith] = ib_first;
                        mmctx->quant_ib_last[ith]  = ib_last;
                        mmctx->quant_r[ith]        = ib_first / nb;
                        mmctx->quant_c[ith]        = ib_first % nb;
                    }
                } अन्यथा {
                    quant_tasks = MIN(cur_m_rows, octx->n_threads);
                    q_func = quant_task_func;
                    mmctx->n_quant_rows_per_thread = (cur_m_rows + quant_tasks - 1) / quant_tasks;
                }
                mmctx->n_quant_tasks = quant_tasks;
                work_queue_run(octx->ctx->work_queue, q_func, mmctx, quant_tasks);
            } अन्यथा {
                hvx_mm_transfer_act_dma(octx, kparams, act, mmctx->vtcm_act, mmctx->vtcm_act_stride, m_start, cur_m_rows);
            }

            work_queue_run(octx->ctx->work_queue, matmul_job_func, mmctx, octx->n_threads);
        }
    } अन्यथा {
        mmctx->cur_m_start = 0;
        mmctx->cur_m_rows  = act_nrows;

        यदि (need_quant) {
            hvx_mm_transfer_act_dma(octx, kparams, act, mmctx->vtcm_act_raw, mmctx->vtcm_act_raw_stride, 0, act_nrows);
            mmctx->n_quant_rows_per_thread = (act_nrows + n_quant_tasks - 1) / n_quant_tasks;
            mmctx->n_quant_tasks = n_quant_tasks;
            work_queue_run(octx->ctx->work_queue, quant_task_func, mmctx, n_quant_tasks);
        } अन्यथा {
            hvx_mm_transfer_act_dma(octx, kparams, act, mmctx->vtcm_act, mmctx->vtcm_act_stride, 0, act_nrows);
        }

        work_queue_run(octx->ctx->work_queue, matmul_job_func, mmctx, octx->n_threads);
    }

    प्रत्यागम HTP_STATUS_OK;
}

स्थिर शून्य dequantize_tiled_worker_loop(अचिन्हित पूर्णाङ्क n, अचिन्हित पूर्णाङ्क i, शून्य * data) {
    tiled_dequantize_state_t * state = (tiled_dequantize_state_t *) data;
    संरचना htp_thread_trace * tr = &state->traces[i];
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_W_DEQUANT, i);
    प्रति (अचिन्हित पूर्णाङ्क task_id = i; task_id < (अचिन्हित पूर्णाङ्क) state->n_tasks; task_id += n) {
        पूर्णाङ्क start = task_id * state->n_tiles_per_task;
        पूर्णाङ्क end   = hex_smin(start + state->n_tiles_per_task, state->n_tot_tiles);
        state->task_fn(state, start, end);
    }
    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_W_DEQUANT, i);
}

स्थिर शून्य transfer_output_chunk_worker_fn(अचिन्हित पूर्णाङ्क n, अचिन्हित पूर्णाङ्क i, शून्य *data) {
    output_transfer_task_state_t *st = (output_transfer_task_state_t *) data;

    संरचना htp_thread_trace * tr = &st->traces[i];

    पूर्णाङ्क start_chunk_idx = i * st->n_chunks_per_task;
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_O_PROC, start_chunk_idx);

    प्रति (अचिन्हित पूर्णाङ्क task_id = i; task_id < (अचिन्हित पूर्णाङ्क)st->n_tasks; task_id += n) {
        पूर्णाङ्क    chunk_idx  = task_id * st->n_chunks_per_task;
        माप_प्रकार chunk_size = hex_smin(st->n_tot_chunks - chunk_idx, st->n_chunks_per_task);

        भिन्न        *dst      = st->dst      + chunk_idx * st->dst_stride;
        ध्रुव भिन्न  *src2     = st->src2     ? (st->src2     + chunk_idx * st->src2_stride) : रिक्त;
        transfer_output_chunk_fp16_to_fp32(dst, src2, st->vtcm_src, chunk_idx, chunk_size, st->n_cols, st->dst_stride, st->src2_stride, st->dst_cols);
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_O_PROC, start_chunk_idx);
}

प्रकारनिर्देश संरचना {
    संरचना htp_context            * ctx;
    संरचना htp_thread_trace       * traces;
    __fp16                        * dst;
    dma_addr_t                      act_dma_addr;
    भिन्न                         * vtcm_f32_act;
    uint32_t                        n_tasks;
    uint32_t                        n_tot_chunks;
    uint32_t                        n_chunks_per_task;
    uint32_t                        k_block;
    uint32_t                        k_stride;
    uint32_t                        k_valid;
    माप_प्रकार                          vtcm_f32_act_bytes_per_thread;
    uint32_t                        dma_step_rows;
    uint32_t                        dma_step_rows_shift;
    uint32_t                        act_elem_size;  // 4=F32 src1, 2=F16 src1
} activation_transfer_task_state_t;

प्रकारनिर्देश संरचना {
    संरचना htp_context            * ctx;
    संरचना htp_thread_trace       * traces;
    __fp16                        * dst;
    dma_addr_t                      act_dma_addr;
    भिन्न                         * vtcm_f32_act;
    uint32_t                        n_rows;
    uint32_t                        k_block;
    uint32_t                        k_stride;
    uint32_t                        k_valid;
    uint32_t                        n_col_chunks;
    संरचना fastdiv_मूल्यावली           n_threads_div;
    माप_प्रकार                          vtcm_f32_act_bytes;
    uint32_t                        dma_step_rows;
    uint32_t                        dma_step_rows_shift;
} activation_transfer_col_chunk_state_t;

स्थिर शून्य transfer_activation_chunk_fp32_to_fp16_dma_pipelined_col_chunk(
        dma_queue *dma_q,
        __fp16 *restrict vtcm_dst,
        dma_addr_t act_dma_addr,
        uint32_t n_rows,
        uint32_t k_block,
        uint32_t k_stride,
        uint32_t k_chunk_valid,
        uint32_t c_first,
        uint32_t c_len,
        भिन्न *thread_f32_act,
        संरचना htp_thread_trace *tr,
        uint32_t dma_step_rows,
        uint32_t dma_step_rows_shift) {

    ध्रुव uint32_t R = dma_step_rows;
    ध्रुव uint32_t n_rows_padded = hex_align_up(n_rows, HTP_MM_HMX_TILE_N_ROWS);

    ध्रुव uint32_t n_steps = n_rows_padded >> dma_step_rows_shift;

    // Push step 0
    यदि (n_steps > 0 && n_rows > 0) {
        uint32_t nrows_to_fetch = hex_smin(n_rows, R);
        dma_queue_धकेलो(dma_q, dma_make_data(thread_f32_act, act_dma_addr + (माप_प्रकार) c_first * परिमाण(भिन्न)),
                       c_len * परिमाण(भिन्न), k_stride * परिमाण(भिन्न), k_chunk_valid * परिमाण(भिन्न), nrows_to_fetch);
    }
    // Push step 1
    यदि (n_steps > 1) {
        uint32_t next_r = R * 1;
        यदि (next_r < n_rows) {
            uint32_t nrows_to_fetch = hex_smin(n_rows - next_r, R);
            भिन्न *next_buf = thread_f32_act + 1 * R * c_len;
            dma_queue_धकेलो(dma_q, dma_make_data(next_buf, act_dma_addr + ((माप_प्रकार) next_r * k_stride + c_first) * परिमाण(भिन्न)),
                           c_len * परिमाण(भिन्न), k_stride * परिमाण(भिन्न), k_chunk_valid * परिमाण(भिन्न), nrows_to_fetch);
        }
    }
    प्रति (uint32_t s = 0; s < n_steps; ++s) {
        uint32_t r = s << dma_step_rows_shift;
        भिन्न *curr_buf = thread_f32_act;

        यदि (r < n_rows) {
            curr_buf = (भिन्न *) dma_queue_निकालो(dma_q).dst;
        }

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_A_PREP, r);
        प्रति (uint32_t p = 0; p < (R >> 1); ++p) {
            uint32_t row_idx = r + (p << 1);
            भिन्न *pair_buf = curr_buf + (p << 1) * c_len;
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
        यदि (next_r < n_rows) {
            uint32_t nrows_to_fetch = hex_smin(n_rows - next_r, R);
            dma_queue_धकेलो(dma_q, dma_make_data(curr_buf, act_dma_addr + ((माप_प्रकार) next_r * k_stride + c_first) * परिमाण(भिन्न)),
                           c_len * परिमाण(भिन्न), k_stride * परिमाण(भिन्न), k_chunk_valid * परिमाण(भिन्न), nrows_to_fetch);
        }
    }
}

स्थिर शून्य transfer_activation_chunk_col_chunk_worker_fn(अचिन्हित पूर्णाङ्क n, अचिन्हित पूर्णाङ्क i, शून्य *data) {
    activation_transfer_col_chunk_state_t *st = (activation_transfer_col_chunk_state_t *) data;
    संरचना htp_thread_trace * tr = &st->traces[i];

    uint32_t n_blocks = st->k_block / 32;
    uint32_t b_first = fastdiv(n_blocks * i, &st->n_threads_div);
    uint32_t b_last  = fastdiv(n_blocks * (i + 1), &st->n_threads_div);
    uint32_t c_first = b_first * 32;
    uint32_t c_last = b_last * 32;
    uint32_t c_len = c_last - c_first;

    यदि (c_len == 0) {
        प्रत्यागम;
    }

    uint32_t k_chunk_valid = 0;
    यदि (st->k_valid > c_first) {
        k_chunk_valid = hex_smin(st->k_valid, c_last) - c_first;
    }

    __fp16 *dst = st->dst;

    माप_प्रकार thread_scratch_bytes = hex_align_down(fastdiv(st->vtcm_f32_act_bytes, &st->n_threads_div), 128);
    भिन्न *thread_f32_act = (भिन्न *)((वर्ण *)st->vtcm_f32_act + i * thread_scratch_bytes);

    transfer_activation_chunk_fp32_to_fp16_dma_pipelined_col_chunk(
        st->ctx->dma[i], dst, st->act_dma_addr, st->n_rows, st->k_block, st->k_stride, k_chunk_valid,
        c_first, c_len, thread_f32_act, tr, st->dma_step_rows, st->dma_step_rows_shift
    );
}

स्थिर शून्य transfer_activation_chunk_to_fp16_dma_pipelined(
        dma_queue *dma_q,
        __fp16 *restrict vtcm_dst,
        dma_addr_t act_dma_addr,
        uint32_t n_rows,
        uint32_t k_block,
        uint32_t k_stride,
        uint32_t k_valid,
        uint8_t *thread_act,
        uint32_t act_elem_size,
        संरचना htp_thread_trace *tr,
        uint32_t dma_step_rows,
        uint32_t dma_step_rows_shift) {

    ध्रुव uint32_t R = dma_step_rows;
    ध्रुव uint32_t n_rows_padded = hex_align_up(n_rows, HTP_MM_HMX_TILE_N_ROWS);

    ध्रुव uint32_t n_steps = n_rows_padded >> dma_step_rows_shift;

    ध्रुव bool act_is_f16 = (act_elem_size == परिमाण(__fp16));
    ध्रुव माप_प्रकार row_bytes = (माप_प्रकार) k_block * act_elem_size;  // staging row (DMA dst stride)
    ध्रुव माप_प्रकार src_row_bytes = (माप_प्रकार) k_stride * act_elem_size;  // DDR row (DMA src stride)
    ध्रुव माप_प्रकार width_bytes = (माप_प्रकार) k_valid * act_elem_size;

    // Push step 0
    यदि (n_steps > 0 && n_rows > 0) {
        uint32_t nrows_to_fetch = hex_smin(n_rows, R);
        dma_queue_धकेलो(dma_q, dma_make_data(thread_act, act_dma_addr),
                       row_bytes, src_row_bytes, width_bytes, nrows_to_fetch);
    }
    // Push step 1 (if valid)
    यदि (n_steps > 1) {
        uint32_t next_r = R * 1;
        यदि (next_r < n_rows) {
            uint32_t nrows_to_fetch = hex_smin(n_rows - next_r, R);
            uint8_t *next_buf = thread_act + 1 * R * row_bytes;
            dma_queue_धकेलो(dma_q, dma_make_data(next_buf, act_dma_addr + (माप_प्रकार) next_r * src_row_bytes),
                           row_bytes, src_row_bytes, width_bytes, nrows_to_fetch);
        }
    }
    प्रति (uint32_t s = 0; s < n_steps; ++s) {
        uint32_t r = s << dma_step_rows_shift;
        uint8_t *curr_buf = thread_act;

        यदि (r < n_rows) {
            curr_buf = (uint8_t *) dma_queue_निकालो(dma_q).dst;
        }

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_A_PREP, r);
        // Two copies of the pair loop so the type is resolved once per step and the
        // row-pair kernels stay direct (inlinable) calls.
        यदि (act_is_f16) {
            प्रति (uint32_t p = 0; p < (R >> 1); ++p) {
                uint32_t row_idx = r + (p << 1);
                ध्रुव __fp16 *pair_buf = (ध्रुव __fp16 *) (curr_buf + (p << 1) * row_bytes);
                bool r0_valid = ((row_idx + 0) < n_rows);
                bool r1_valid = ((row_idx + 1) < n_rows);

                transfer_activation_row_pair_f16_to_f16(vtcm_dst, pair_buf, pair_buf + k_block, row_idx, k_block, k_valid, r0_valid, r1_valid);
            }
        } अन्यथा {
            प्रति (uint32_t p = 0; p < (R >> 1); ++p) {
                uint32_t row_idx = r + (p << 1);
                ध्रुव भिन्न *pair_buf = (ध्रुव भिन्न *) (curr_buf + (p << 1) * row_bytes);
                bool r0_valid = ((row_idx + 0) < n_rows);
                bool r1_valid = ((row_idx + 1) < n_rows);

                transfer_activation_row_pair_fp32_to_fp16(vtcm_dst, pair_buf, pair_buf + k_block, row_idx, k_block, k_valid, r0_valid, r1_valid);
            }
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_A_PREP, r);

        // Push step s + 2
        uint32_t next_s = s + 2;
        uint32_t next_r = next_s << dma_step_rows_shift;
        यदि (next_r < n_rows) {
            uint32_t nrows_to_fetch = hex_smin(n_rows - next_r, R);
            dma_queue_धकेलो(dma_q, dma_make_data(curr_buf, act_dma_addr + (माप_प्रकार) next_r * src_row_bytes),
                           row_bytes, src_row_bytes, width_bytes, nrows_to_fetch);
        }
    }
}

स्थिर शून्य transfer_activation_chunk_worker_fn(अचिन्हित पूर्णाङ्क n, अचिन्हित पूर्णाङ्क i, शून्य *data) {
    activation_transfer_task_state_t *st = (activation_transfer_task_state_t *) data;

    संरचना htp_thread_trace * tr = &st->traces[i];

    प्रति (अचिन्हित पूर्णाङ्क task_id = i; task_id < (अचिन्हित पूर्णाङ्क)st->n_tasks; task_id += n) {
        पूर्णाङ्क    chunk_idx  = task_id * st->n_chunks_per_task;
        माप_प्रकार chunk_size = hex_smin(st->n_tot_chunks - chunk_idx, st->n_chunks_per_task);

        __fp16      *dst = st->dst + chunk_idx * st->k_block;
        ध्रुव dma_addr_t act_dma_addr = st->act_dma_addr + (माप_प्रकार) chunk_idx * st->k_stride * st->act_elem_size;

        uint8_t *thread_act = (uint8_t *) st->vtcm_f32_act + i * st->vtcm_f32_act_bytes_per_thread;
        transfer_activation_chunk_to_fp16_dma_pipelined(
            st->ctx->dma[i], dst, act_dma_addr, chunk_size, st->k_block, st->k_stride, st->k_valid, thread_act, st->act_elem_size, tr, st->dma_step_rows, st->dma_step_rows_shift
        );
    }
}

प्रकारनिर्देश संरचना {
    संरचना htp_thread_trace       * traces;
    ध्रुव संरचना mmid_row_mapping * matrix_rows;
    __fp16                        * dst;
    ध्रुव भिन्न                   * src;
    uint32_t                        n_tasks;
    uint32_t                        n_tot_chunks;
    uint32_t                        n_chunks_per_task;
    uint32_t                        k_block;
    uint32_t                        cur_a;
    uint32_t                        mapping_stride;
    uint32_t                        ne11;
    संरचना fastdiv_मूल्यावली           ne11_div;
    माप_प्रकार                          nb11;
    माप_प्रकार                          nb12;
    uint32_t                        start_row;
    uint32_t                        cne1;
    uint32_t                        k_valid;
} activation_transfer_gathered_task_state_t;

प्रकारनिर्देश संरचना {
    संरचना htp_thread_trace       * traces;
    ध्रुव संरचना mmid_row_mapping * matrix_rows;
    ध्रुव __fp16                  * vtcm_src;
    भिन्न                         * dst;
    uint32_t                        n_tasks;
    uint32_t                        n_tot_chunks;
    uint32_t                        n_chunks_per_task;
    uint32_t                        n_cols;
    uint32_t                        cur_a;
    uint32_t                        mapping_stride;
    माप_प्रकार                          dst_nb1;
    माप_प्रकार                          dst_nb2;
    uint32_t                        start_row;
    uint32_t                        cne1;
} output_transfer_scattered_task_state_t;

स्थिर शून्य transfer_activation_chunk_gathered_worker_fn(अचिन्हित पूर्णाङ्क n, अचिन्हित पूर्णाङ्क i, शून्य *data) {
    activation_transfer_gathered_task_state_t *st = data;
    संरचना htp_thread_trace * tr = &st->traces[i];
    पूर्णाङ्क chunk_idx      = i;
    पूर्णाङ्क chunk_size     = st->n_chunks_per_task;
    पूर्णाङ्क vtcm_start_row = chunk_idx * chunk_size;
    पूर्णाङ्क start_row      = st->start_row + vtcm_start_row;
    पूर्णाङ्क n_rows         = hex_smin(st->cne1 - start_row, chunk_size);
    यदि (n_rows > 0) {
        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_A_PREP, chunk_idx);
        transfer_activation_chunk_fp32_to_fp16_gathered(
            st->dst, st->src, start_row, vtcm_start_row, n_rows, st->k_block,
            st->matrix_rows, st->cur_a, st->mapping_stride,
            st->ne11, &st->ne11_div, st->nb11, st->nb12, st->cne1, st->k_valid);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_A_PREP, chunk_idx);
    }
}

स्थिर शून्य transfer_activation_chunk_gathered_worker_flat_fn(अचिन्हित पूर्णाङ्क n, अचिन्हित पूर्णाङ्क i, शून्य *data) {
    activation_transfer_gathered_task_state_t *st = data;
    संरचना htp_thread_trace * tr = &st->traces[i];
    पूर्णाङ्क chunk_idx = i;
    पूर्णाङ्क chunk_size = st->n_chunks_per_task;
    पूर्णाङ्क vtcm_start_row = chunk_idx * chunk_size;
    पूर्णाङ्क start_row = st->start_row + vtcm_start_row;
    पूर्णाङ्क n_rows = hex_smin(st->cne1 - start_row, chunk_size);
    यदि (n_rows > 0) {
        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_A_PREP, chunk_idx);
        transfer_activation_chunk_fp32_to_fp16_gathered_flat(
            st->dst, st->src, start_row, vtcm_start_row, n_rows, st->k_block,
            st->matrix_rows, st->cur_a, st->mapping_stride,
            st->nb12, st->cne1, st->k_valid);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_A_PREP, chunk_idx);
    }
}

स्थिर शून्य transfer_output_chunk_scattered_worker_fn(अचिन्हित पूर्णाङ्क n, अचिन्हित पूर्णाङ्क i, शून्य *data) {
    output_transfer_scattered_task_state_t *st = data;
    संरचना htp_thread_trace * tr = &st->traces[i];
    पूर्णाङ्क chunk_idx = i;
    पूर्णाङ्क chunk_size = st->n_chunks_per_task;
    पूर्णाङ्क vtcm_start_row = chunk_idx * chunk_size;
    पूर्णाङ्क start_row = st->start_row + vtcm_start_row;
    पूर्णाङ्क n_rows = hex_smin(st->cne1 - start_row, chunk_size);
    यदि (n_rows > 0) {
        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_O_PROC, chunk_idx);
        transfer_output_chunk_fp16_to_fp32_scattered(
            st->dst, st->vtcm_src, start_row, vtcm_start_row, n_rows, st->n_cols,
            st->matrix_rows, st->cur_a, st->mapping_stride,
            st->dst_nb1, st->dst_nb2, st->cne1);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_O_PROC, chunk_idx);
    }
}

// --- HMX Dispatchers & Entry Points ---

स्थिर अन्तःस्थ tiled_dequant_task_fn_t hmx_get_dequant_worker_fn(पूर्णाङ्क weight_type, bool is_tiled) {
    विकल्प (weight_type) {
        पक्ष HTP_TYPE_Q4_0:   प्रत्यागम dequantize_tiled_weight_to_fp16_task_q4_0;
        पक्ष HTP_TYPE_IQ4_NL: प्रत्यागम dequantize_tiled_weight_to_fp16_task_iq4_nl;
        पक्ष HTP_TYPE_Q4_1:
        पक्ष HTP_TYPE_Q4_K:   प्रत्यागम dequantize_tiled_weight_to_fp16_task_q4_1;
        पक्ष HTP_TYPE_MXFP4:  प्रत्यागम dequantize_tiled_weight_to_fp16_task_mxfp4;
        पक्ष HTP_TYPE_Q8_0:   प्रत्यागम dequantize_tiled_weight_to_fp16_task_q8_0;
        पक्ष HTP_TYPE_Q5_K:   प्रत्यागम dequantize_tiled_weight_to_fp16_task_q5_k;
        पक्ष HTP_TYPE_Q6_K:   प्रत्यागम dequantize_tiled_weight_to_fp16_task_q6_k;
        पक्ष HTP_TYPE_Q3_K:   प्रत्यागम dequantize_tiled_weight_to_fp16_task_q3_k;
        पक्ष HTP_TYPE_Q2_K:   प्रत्यागम dequantize_tiled_weight_to_fp16_task_q2_k;
        पक्ष HTP_TYPE_Q1_0:   प्रत्यागम dequantize_tiled_weight_to_fp16_task_q1_0;
        पक्ष HTP_TYPE_Q2_0:   प्रत्यागम dequantize_tiled_weight_to_fp16_task_q2_0;
        पक्ष HTP_TYPE_F16:    प्रत्यागम is_tiled ? रिक्त : convert_f16_weight_to_fp16_tiles_task;
        पक्ष HTP_TYPE_BF16:   प्रत्यागम convert_bf16_weight_to_fp16_tiles_task;
        पक्ष HTP_TYPE_F32:    प्रत्यागम quantize_f32_weight_to_fp16_tiles_task;
        स्वभाव:              प्रत्यागम रिक्त;
    }
}

स्थिर शून्य dequantize_tiled_weight_chunk_to_fp16_tiles(
        संरचना htp_context *ctx, __fp16 *vtcm_dst,
        ध्रुव शून्य *weight_src_ddr,
        पूर्णाङ्क n_cols, पूर्णाङ्क k_block,
        माप_प्रकार row_stride, पूर्णाङ्क weight_type,
        पूर्णाङ्क n_k_tiles, संरचना fastdiv_मूल्यावली n_k_tiles_div,
        tiled_dequant_task_fn_t dequant_worker_fn, पूर्णाङ्क n_threads) {

    यदि (!dequant_worker_fn) {
        प्रत्यागम;
    }

    प्रतिज्ञा(k_block % HTP_MM_HMX_TILE_N_COLS == 0);

    ध्रुव माप_प्रकार n_col_tiles = hmx_ceil_div(n_cols, HTP_MM_HMX_TILE_N_COLS);
    ध्रुव uint32_t n_weight_tiles = k_block / htp_tensor_get_tile_k(weight_type);
    ध्रुव माप_प्रकार n_tot_tiles = n_col_tiles * n_weight_tiles;

    माप_प्रकार n_tiles_per_task = (n_threads == 1) ? n_tot_tiles : hmx_ceil_div(n_tot_tiles, n_threads);

    tiled_dequantize_state_t state;
    state.n_tasks          = (n_tot_tiles + n_tiles_per_task - 1) / n_tiles_per_task;
    state.n_tot_tiles      = n_tot_tiles;
    state.n_tiles_per_task = n_tiles_per_task;
    state.dst              = vtcm_dst;
    state.src              = (ध्रुव uint8_t *)weight_src_ddr;
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

    यदि (state.n_tasks == 1 || n_threads == 1) {
        dequantize_tiled_worker_loop(1, 0, &state);
    } अन्यथा {
        पूर्णाङ्क n_tasks = hex_smin((पूर्णाङ्क) state.n_tasks, n_threads);
        worker_pool_run_func(ctx->worker_pool, dequantize_tiled_worker_loop, &state, n_tasks);
    }
}

प्रकारनिर्देश संरचना {
    संरचना htp_context      * ctx;
    संरचना htp_thread_trace * traces;
    भिन्न                   * dst;
    ध्रुव __fp16            * vtcm_src;
    ध्रुव भिन्न             * src2;
    uint32_t                  n_rows;
    uint32_t                  n_cols;
    uint32_t                  dst_stride;
    uint32_t                  src2_stride;
    uint32_t                  dst_cols;
    संरचना fastdiv_मूल्यावली     n_threads_div;
} output_transfer_col_chunk_state_t;

स्थिर शून्य transfer_output_chunk_col_chunk_worker_fn(अचिन्हित पूर्णाङ्क n, अचिन्हित पूर्णाङ्क i, शून्य *data) {
    (शून्य) n;
    output_transfer_col_chunk_state_t *st = (output_transfer_col_chunk_state_t *) data;
    संरचना htp_thread_trace * tr = &st->traces[i];

    // Round up: the last block is partial when N is not 32-aligned. Its pad
    // columns are dropped by the dst_cols clamp inside the store.
    uint32_t n_blocks = hmx_ceil_div(st->n_cols, 32);
    uint32_t b_first  = fastdiv(n_blocks * i, &st->n_threads_div);
    uint32_t b_last   = fastdiv(n_blocks * (i + 1), &st->n_threads_div);
    uint32_t c_first  = b_first * 32;
    uint32_t c_last   = b_last * 32;
    uint32_t c_len    = c_last - c_first;

    यदि (c_len == 0) प्रत्यागम;

    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_O_PROC, c_first);

    ध्रुव __fp16 *vtcm_src = st->vtcm_src + b_first * HTP_MM_HMX_TILE_N_ELMS;
    ध्रुव भिन्न *src2      = st->src2 ? (st->src2 + c_first) : रिक्त;
    भिन्न *dst             = st->dst + c_first;

    पूर्णाङ्क chunk_dst_cols = (पूर्णाङ्क)st->dst_cols - (पूर्णाङ्क)c_first;
    यदि (chunk_dst_cols > 0) {
        transfer_output_chunk_fp16_to_fp32_col_chunk(
            dst, src2, vtcm_src, 0, st->n_rows, c_len, st->n_cols,
            st->dst_stride, st->src2_stride, (uint32_t)chunk_dst_cols
        );
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_O_PROC, c_first);
}

स्थिर शून्य transfer_output_chunk_threaded(संरचना htp_context *ctx, भिन्न *dst, ध्रुव भिन्न *src2, ध्रुव __fp16 *vtcm_src,
                                              पूर्णाङ्क n_rows, पूर्णाङ्क n_cols, पूर्णाङ्क dst_stride, uint32_t src2_stride, पूर्णाङ्क dst_cols, पूर्णाङ्क n_threads) {
    यदि (n_rows <= 0) प्रत्यागम;

    uint32_t n_blocks = hmx_ceil_div((uint32_t) n_cols, 32);
    यदि (n_threads > 1 && n_blocks >= (uint32_t)n_threads) {
        संरचना fastdiv_मूल्यावली n_threads_div = (n_threads == (पूर्णाङ्क)ctx->n_threads) ? ctx->n_threads_div : init_fastdiv_मूल्यावली(n_threads);
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
        प्रत्यागम;
    }

    माप_प्रकार n_tot_chunks      = n_rows;
    माप_प्रकार n_chunks_per_task = (n_threads == 1) ? n_tot_chunks : hmx_ceil_div(n_rows, n_threads);
    n_chunks_per_task        = hex_align_up(n_chunks_per_task, 2);

    पूर्णाङ्क actual_threads = hmx_ceil_div(n_rows, n_chunks_per_task);

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

    यदि (actual_threads <= 1) {
        transfer_output_chunk_worker_fn(1, 0, &state);
    } अन्यथा {
        worker_pool_run_func(ctx->worker_pool, transfer_output_chunk_worker_fn, &state, actual_threads);
    }
}

संरचना activation_transfer_params {
    संरचना htp_context *          ctx;
    __fp16 *                      dst;
    dma_addr_t                    act_dma_addr;
    पूर्णाङ्क                           n_rows;
    पूर्णाङ्क                           k_block;
    पूर्णाङ्क                           k_stride;
    पूर्णाङ्क                           n_threads;
    ध्रुव संरचना fastdiv_मूल्यावली * act_threads_div;
    ध्रुव संरचना fastdiv_मूल्यावली * k_div;
    पूर्णाङ्क                           k_valid;
    भिन्न *                       vtcm_f32_act;
    माप_प्रकार                        vtcm_f32_act_bytes;
    uint32_t                      act_elem_size;  // 4=F32 src1, 2=F16 src1
};

स्थिर शून्य transfer_activation_chunk_threaded(ध्रुव संरचना activation_transfer_params * params) {
    संरचना htp_context *          ctx                = params->ctx;
    __fp16 *                      dst                = params->dst;
    ध्रुव dma_addr_t              act_dma_addr       = params->act_dma_addr;
    पूर्णाङ्क                           n_rows             = params->n_rows;
    पूर्णाङ्क                           k_block            = params->k_block;
    पूर्णाङ्क                           k_stride           = params->k_stride;
    पूर्णाङ्क                           n_threads          = params->n_threads;
    ध्रुव संरचना fastdiv_मूल्यावली * act_threads_div    = params->act_threads_div;
    ध्रुव संरचना fastdiv_मूल्यावली * k_div              = params->k_div;
    पूर्णाङ्क                           k_valid            = params->k_valid;
    भिन्न *                       vtcm_f32_act       = params->vtcm_f32_act;
    माप_प्रकार                        vtcm_f32_act_bytes = params->vtcm_f32_act_bytes;
    // element size of the activation rows (4 = F32, 2 = F16).
    ध्रुव uint32_t                act_elem_size      = params->act_elem_size ? params->act_elem_size : (uint32_t) परिमाण(भिन्न);

    यदि (n_rows <= 0) {
        प्रत्यागम;
    }

    ध्रुव माप_प्रकार n_tasks = (n_rows + 31) >> 5;
    यदि (act_elem_size == परिमाण(भिन्न) && n_threads > 1 && k_block > 32 && n_tasks < (माप_प्रकार) n_threads) {
        // Calculate step rows parameters for column-chunked dma pipelining
        uint32_t dma_step_rows = 2;
        uint32_t dma_step_rows_shift = 1;
        यदि (vtcm_f32_act_bytes > 0) {
            माप_प्रकार thread_scratch_bytes = hex_align_down(fastdiv(vtcm_f32_act_bytes, act_threads_div), 128);
            माप_प्रकार thread_scratch_elements = thread_scratch_bytes / परिमाण(भिन्न);
            माप_प्रकार dma_step_rows_max = fastdiv(thread_scratch_elements / 2, k_div);
            यदि (dma_step_rows_max >= 4) {
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
        प्रत्यागम;
    }

    प्रतिज्ञा(k_block % HTP_MM_HMX_TILE_N_COLS == 0 && k_stride % HTP_MM_HMX_TILE_N_COLS == 0);

    माप_प्रकार n_tot_chunks      = n_rows;
    माप_प्रकार n_chunks_per_task = (n_threads == 1) ? n_tot_chunks : 32;  // must be multiple of 32 to ensure correct destination address

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
    यदि (state.vtcm_f32_act_bytes_per_thread > 0) {
        माप_प्रकार thread_scratch_elements = state.vtcm_f32_act_bytes_per_thread / परिमाण(भिन्न);
        माप_प्रकार dma_step_rows_max = fastdiv(thread_scratch_elements / 2, k_div);
        यदि (dma_step_rows_max >= 4) {
            dma_step_rows = 4;
            dma_step_rows_shift = 2;
        }
    }
    state.dma_step_rows      = dma_step_rows;
    state.dma_step_rows_shift = dma_step_rows_shift;

    पूर्णाङ्क active_threads = hex_smin(n_threads, (पूर्णाङ्क)state.n_tasks);
    यदि (state.n_tasks == 1 || n_threads == 1) {
        transfer_activation_chunk_worker_fn(1, 0, &state);
    } अन्यथा {
        worker_pool_run_func(ctx->worker_pool, transfer_activation_chunk_worker_fn, &state, active_threads);
    }
}
// --- Async HMX matmul job (for pipeline overlap) ---

प्रकारनिर्देश संरचना {
    __fp16 *       output;
    ध्रुव __fp16 * activation;
    ध्रुव __fp16 * weight;
    ध्रुव __fp16 * scales;
    uint32_t       n_row_tiles;
    uint32_t       n_col_tiles;
    uint32_t       n_dot_tiles;
} hmx_matmul_job_t;

स्थिर शून्य hmx_matmul_worker_fn(शून्य * data) {
    hmx_matmul_job_t * job = (hmx_matmul_job_t *) data;
    FARF(ऊपरी_2, "hmx-mm-job: n_row_tiles %u n_col_tiles %u n_dot_tiles %u", job->n_row_tiles, job->n_col_tiles, job->n_dot_tiles);
    core_dot_chunk_fp16(job->output, job->activation, job->weight, job->scales, job->n_row_tiles, job->n_col_tiles, job->n_dot_tiles);
}

स्थिर अन्तःस्थ शून्य hmx_matmul_job_init(hmx_matmul_job_t * job,
                                       __fp16 *           output,
                                       ध्रुव __fp16 *     activation,
                                       ध्रुव __fp16 *     weight,
                                       ध्रुव __fp16 *     scales,
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

स्थिर पूर्णाङ्क hmx_mm_2d_f32(संरचना htp_context *ctx,
                                  dma_queue *weight_dma,
                                  भिन्न *restrict dst,
                                  dma_addr_t src2_addr,
                                  माप_प्रकार src2_bytes,
                                  dma_addr_t act_dma_addr,
                                  dma_addr_t weight,
                                  पूर्णाङ्क m, पूर्णाङ्क k, पूर्णाङ्क n,
                                  पूर्णाङ्क act_stride,
                                  uint32_t act_elem_size,  // 4=F32 src1, 2=F16 src1
                                  पूर्णाङ्क weight_stride,
                                  पूर्णाङ्क weight_type,
                                  पूर्णाङ्क k_valid,
                                  पूर्णाङ्क dst_stride,
                                  uint32_t src2_stride,
                                  पूर्णाङ्क dst_cols,
                                  पूर्णाङ्क m_chunk,
                                  पूर्णाङ्क n_chunk,
                                  पूर्णाङ्क pipeline,
                                  पूर्णाङ्क n_threads,
                                  पूर्णाङ्क act_threads,
                                  ध्रुव संरचना fastdiv_मूल्यावली * act_threads_div,
                                  ध्रुव संरचना fastdiv_मूल्यावली * k_div,
                                  पूर्णाङ्क tile_size,
                                  पूर्णाङ्क aligned_tile_size,
                                  पूर्णाङ्क vtcm_size) {
    संरचना htp_thread_trace * tr = &ctx->trace[0];
    htp_trace_event_start(tr, HTP_TRACE_EVT_INIT, 0);

    ध्रुव bool is_tiled = (tile_size > 0);

    यदि (k % 32 != 0) {
        प्रत्यागम -1;
    }
    यदि (!hex_is_aligned(dst, परिमाण(भिन्न))) { प्रत्यागम -1; }

    माप_प्रकार row_stride = htp_tensor_get_row_size(weight_type, k);
    यदि (row_stride == 0) {
        प्रत्यागम -1;
    }

    tiled_dequant_task_fn_t dequant_worker_fn = hmx_get_dequant_worker_fn(weight_type, is_tiled);

    ध्रुव पूर्णाङ्क n_k_tiles = k / HTP_MM_HMX_TILE_N_COLS;
    ध्रुव पूर्णाङ्क n_weight_tiles = k / htp_tensor_get_tile_k(weight_type);
    ध्रुव संरचना fastdiv_मूल्यावली n_k_tiles_div = init_fastdiv_मूल्यावली(n_k_tiles);

    ध्रुव माप_प्रकार vec_dot_size = k * परिमाण(__fp16);
    ध्रुव माप_प्रकार vtcm_budget  = ctx->vtcm_size;

    ध्रुव uint32_t dma_dst_stride  = is_tiled ? aligned_tile_size : row_stride;
    ध्रुव uint32_t dma_src_stride  = is_tiled ? tile_size : weight_stride;
    ध्रुव uint32_t dma_width_bytes = is_tiled ? tile_size : row_stride;

    माप_प्रकार m_chunk_n_rows = m_chunk;
    माप_प्रकार n_chunk_n_cols = n_chunk;
    माप_प्रकार vtcm_used      = vtcm_size;

    संरचना htp_mm_hmx_vtcm_layout L;
    htp_mm_hmx_vtcm_layout_build(&L, HTP_MM_KERNEL_HMX_2D, weight_type, k, m_chunk_n_rows, n_chunk_n_cols, 1, pipeline, act_threads, aligned_tile_size, src2_bytes);

    vtcm_used = L.total_bytes;
    यदि (vtcm_used > vtcm_budget) {
        FARF(ERROR, "hmx-mm-2d-precomputed: VTCM overflow: used %zu budget %zu, m %d k %d n %d mc %zu nc %zu",
             vtcm_used, vtcm_budget, m, k, n, m_chunk_n_rows, n_chunk_n_cols);
        प्रत्यागम -1;
    }

    uint8_t * ध्रुव base = (uint8_t *) ctx->vtcm_base;
    __fp16  *vtcm_weight_raw[2] = {
        VTCM_LAYOUT_PTR(__fp16, base, L.off_weight[0]),
        VTCM_LAYOUT_PTR_OPTIONAL(__fp16, base, L.off_weight[1], pipeline)
    };
    __fp16  *vtcm_f16_act    = VTCM_LAYOUT_PTR(__fp16, base, L.off_act);
    भिन्न   *vtcm_f32_act    = VTCM_LAYOUT_PTR(भिन्न, base, L.off_act_f32);
    __fp16  *vtcm_output     = VTCM_LAYOUT_PTR(__fp16, base, L.off_dst[0]);
    शून्य    *vtcm_scratch0   = VTCM_LAYOUT_PTR(शून्य, base, L.off_scratch[0]);
    शून्य    *vtcm_scratch1   = VTCM_LAYOUT_PTR_OPTIONAL(शून्य, base, L.off_scratch[1], pipeline);
    शून्य    *vtcm_scratch2   = VTCM_LAYOUT_PTR_OPTIONAL(शून्य, base, L.off_dst[1], pipeline);
    __fp16  *vtcm_scales     = VTCM_LAYOUT_PTR(__fp16, base, L.off_scales);

    hmx_init_column_scales(vtcm_scales, Q6_V_vsplat_R(0x3c00));  // scale: 1.0, bias: 0.0 in FP16

    ध्रुव bool has_bias      = (src2_bytes > 0 && src2_addr != 0);
    भिन्न   *vtcm_bias       = VTCM_LAYOUT_PTR_OPTIONAL(भिन्न, base, L.off_bias, has_bias);
    यदि (has_bias) {
        dma_queue_धकेलो(weight_dma, dma_make_data(vtcm_bias, src2_addr), hex_align_up(src2_bytes, 128), 0, src2_bytes, 1);
        dma_queue_निकालो(weight_dma);
    }

    FARF(ऊपरी_2, "hmx-mm-2d: m %d k %d n %d wtype %d mc %zu nc %zu vtcm %zu/%zu",
         m, k, n, weight_type, m_chunk_n_rows, n_chunk_n_cols, vtcm_used, vtcm_budget);

    पूर्णाङ्क n_chunk_cnt = hmx_ceil_div(n, n_chunk_n_cols);

    htp_trace_event_stop(tr, HTP_TRACE_EVT_INIT, 0);

    यदि (pipeline) {
        // --- Asynchronous Pipelined Loop ---
        hmx_matmul_job_t job_slots[2];  // persistent double-buffered job descriptors
        शून्य *vtcm_raw_bufs[4] = {
            vtcm_weight_raw[0],
            vtcm_weight_raw[1],
            dequant_worker_fn ? vtcm_weight_raw[0] : vtcm_scratch0,
            dequant_worker_fn ? vtcm_weight_raw[1] : vtcm_scratch1,
        };
        शून्य *vtcm_weight_bufs[4] = {
            dequant_worker_fn ? vtcm_scratch0 : vtcm_weight_raw[0],
            dequant_worker_fn ? vtcm_scratch1 : vtcm_weight_raw[1],
            vtcm_scratch0,
            vtcm_scratch1,
        };
        शून्य *vtcm_output_bufs[2] = { vtcm_output, vtcm_scratch2 };

        प्रति (माप_प्रकार mr = 0; mr < m; mr += m_chunk_n_rows) {
            ध्रुव माप_प्रकार n_rows = hex_smin(m - mr, m_chunk_n_rows);

            संरचना activation_transfer_params act_params = {
                .ctx = ctx,
                .dst = vtcm_f16_act,
                .act_dma_addr = act_dma_addr + (माप_प्रकार) mr * act_stride * act_elem_size,  // byte offset (F16/F32)
                .n_rows = (पूर्णाङ्क) n_rows,
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
            ध्रुव माप_प्रकार   n_cols_A0 = hex_smin(n - 0 * n_chunk_n_cols, n_chunk_n_cols);
            ध्रुव uint32_t height_A0 = is_tiled ? hmx_ceil_div(n_cols_A0, 32) * n_weight_tiles : n_cols_A0;
            dma_queue_धकेलो(weight_dma, dma_make_data(vtcm_raw_bufs[0], weight),
                           dma_dst_stride, dma_src_stride, dma_width_bytes, height_A0);

            यदि (1 < n_chunk_cnt) {
                ध्रुव माप_प्रकार   n_cols_A1 = hex_smin(n - 1 * n_chunk_n_cols, n_chunk_n_cols);
                ध्रुव uint32_t height_A1 = is_tiled ? hmx_ceil_div(n_cols_A1, 32) * n_weight_tiles : n_cols_A1;
                dma_queue_धकेलो(weight_dma, dma_make_data(vtcm_raw_bufs[1], weight + n_chunk_n_cols * weight_stride),
                               dma_dst_stride, dma_src_stride, dma_width_bytes, height_A1);
            }

            // Main loop: pop A_i -> dequantize A_i -> push A_{i+2} -> submit C_i -> wait C_{i-1} and store D_{i-1}
            प्रति (पूर्णाङ्क i = 0; i < n_chunk_cnt; ++i) {
                ध्रुव माप_प्रकार nc    = i * n_chunk_n_cols;
                ध्रुव माप_प्रकार nc_p2 = nc + 2 * n_chunk_n_cols;

                ध्रुव माप_प्रकार n_cols    = hex_smin(n - nc, n_chunk_n_cols);
                ध्रुव माप_प्रकार n_cols_p2 = hex_smin(n - nc_p2, n_chunk_n_cols);

                // 1. pop A_i
                शून्य * curr_raw = (शून्य *) dma_queue_निकालो(weight_dma).dst;

                // 2. dequantize A_i
                dequantize_tiled_weight_chunk_to_fp16_tiles(
                    ctx, vtcm_weight_bufs[i % 4], curr_raw,
                    n_cols, k, row_stride, weight_type,
                    n_k_tiles, n_k_tiles_div, dequant_worker_fn, n_threads);

                // 3. push A_{i+2} (if i+2 < n_chunk_cnt)
                यदि (i + 2 < n_chunk_cnt) {
                    ध्रुव uint32_t height_p2 = is_tiled ? hmx_ceil_div(n_cols_p2, 32) * n_weight_tiles : n_cols_p2;
                    dma_queue_धकेलो(weight_dma, dma_make_data(vtcm_raw_bufs[(i + 2) % 4], weight + nc_p2 * weight_stride),
                                   dma_dst_stride, dma_src_stride, dma_width_bytes, height_p2);
                }

                // 4. submit C_i
                hmx_matmul_job_init(&job_slots[i % 2], (__fp16 *) vtcm_output_bufs[i % 2],
                                    (__fp16 *) vtcm_f16_act, (__fp16 *) vtcm_weight_bufs[i % 4],
                                    vtcm_scales, hmx_ceil_div(n_rows, HTP_MM_HMX_TILE_N_ROWS),
                                    hmx_ceil_div(n_cols, HTP_MM_HMX_TILE_N_COLS), k / HTP_MM_HMX_TILE_N_ROWS);
                hmx_queue_धकेलो(ctx->hmx_queue, hmx_queue_make_desc(hmx_matmul_worker_fn, &job_slots[i % 2]));

                // 5. wait C_{i-1} and store D_{i-1} (multi-thread HVX, parallel with C_i)
                यदि (i > 0) {
                    hmx_queue_निकालो(ctx->hmx_queue);
                    ध्रुव माप_प्रकार nc_prev = (i - 1) * n_chunk_n_cols;
                    ध्रुव माप_प्रकार n_cols_prev = hex_smin(n - nc_prev, n_chunk_n_cols);
                    भिन्न *output_chunk = dst + (mr * dst_stride + nc_prev);
                    ध्रुव भिन्न *bias_chunk = has_bias ? (vtcm_bias + mr * src2_stride + nc_prev) : रिक्त;
                    पूर्णाङ्क chunk_dst_cols = dst_cols - (पूर्णाङ्क)nc_prev;
                    यदि (chunk_dst_cols > 0) {
                        transfer_output_chunk_threaded(ctx, output_chunk, bias_chunk, vtcm_output_bufs[(i - 1) % 2], n_rows, n_cols_prev, dst_stride, src2_stride, chunk_dst_cols, n_threads);
                    }
                }
            }

            // Epilogue: wait C_{last} and store D_{last}
            hmx_queue_निकालो(ctx->hmx_queue);
            ध्रुव माप_प्रकार nc_last = (n_chunk_cnt - 1) * n_chunk_n_cols;
            ध्रुव माप_प्रकार n_cols_last = hex_smin(n - nc_last, n_chunk_n_cols);
            भिन्न *output_chunk = dst + (mr * dst_stride + nc_last);
            ध्रुव भिन्न *bias_chunk = has_bias ? (vtcm_bias + mr * src2_stride + nc_last) : रिक्त;
            पूर्णाङ्क chunk_dst_cols = dst_cols - (पूर्णाङ्क)nc_last;
            यदि (chunk_dst_cols > 0) {
                transfer_output_chunk_threaded(ctx, output_chunk, bias_chunk, vtcm_output_bufs[(n_chunk_cnt - 1) % 2], n_rows, n_cols_last, dst_stride, src2_stride, chunk_dst_cols, n_threads);
            }
        }
    } अन्यथा {
        // --- Synchronous loop (m <= 32 or fallback) ---
        hmx_matmul_job_t job;
        शून्य *vtcm_raw_bufs[2] = { vtcm_weight_raw[0], dequant_worker_fn ? vtcm_weight_raw[0] : vtcm_scratch0 };
        शून्य *vtcm_weight_bufs[2] = { dequant_worker_fn ? vtcm_scratch0 : vtcm_weight_raw[0], vtcm_scratch0 };

        प्रति (माप_प्रकार mr = 0; mr < m; mr += m_chunk_n_rows) {
            ध्रुव माप_प्रकार n_rows = hex_smin(m - mr, m_chunk_n_rows);

            संरचना activation_transfer_params act_params = {
                .ctx = ctx,
                .dst = vtcm_f16_act,
                .act_dma_addr = act_dma_addr + (माप_प्रकार) mr * act_stride * act_elem_size,  // byte offset (F16/F32)
                .n_rows = (पूर्णाङ्क) n_rows,
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
            यदि (n > 0) {
                ध्रुव माप_प्रकार n_cols = hex_smin(n, n_chunk_n_cols);
                ध्रुव uint32_t height = is_tiled ? hmx_ceil_div(n_cols, 32) * n_weight_tiles : n_cols;
                dma_queue_धकेलो(weight_dma, dma_make_data(vtcm_raw_bufs[0], weight), dma_dst_stride, dma_src_stride, dma_width_bytes, height);
            }

            प्रति (पूर्णाङ्क i = 0; i < n_chunk_cnt; ++i) {
                ध्रुव माप_प्रकार nc = (माप_प्रकार) i * n_chunk_n_cols;
                ध्रुव माप_प्रकार n_cols = hex_smin(n - nc, n_chunk_n_cols);
                ध्रुव माप_प्रकार n_row_tiles = hmx_ceil_div(n_rows, HTP_MM_HMX_TILE_N_ROWS);
                ध्रुव माप_प्रकार n_col_tiles = hmx_ceil_div(n_cols, HTP_MM_HMX_TILE_N_COLS);

                // A: Wait for weight DMA
                शून्य * curr_raw = (शून्य *) dma_queue_निकालो(weight_dma).dst;

                // B: Weight Dequantize (Threaded)
                dequantize_tiled_weight_chunk_to_fp16_tiles(
                    ctx, vtcm_weight_bufs[i % 2], curr_raw,
                    n_cols, k, row_stride, weight_type,
                    n_k_tiles, n_k_tiles_div, dequant_worker_fn, n_threads);

                // Start weight DMA for the next chunk early
                ध्रुव माप_प्रकार nc_next = nc + n_chunk_n_cols;
                यदि (nc_next < n) {
                    ध्रुव माप_प्रकार n_cols_next = hex_smin(n - nc_next, n_chunk_n_cols);
                    ध्रुव uint32_t height_next = is_tiled ? hmx_ceil_div(n_cols_next, 32) * n_weight_tiles : n_cols_next;
                    dma_queue_धकेलो(weight_dma, dma_make_data(vtcm_raw_bufs[(i + 1) % 2], weight + nc_next * weight_stride), dma_dst_stride, dma_src_stride, dma_width_bytes, height_next);
                }

                // C: HMX Compute (Queue-based)
                hmx_matmul_job_init(&job, vtcm_output, vtcm_f16_act, (__fp16 *) vtcm_weight_bufs[i % 2], vtcm_scales, n_row_tiles, n_col_tiles, k / HTP_MM_HMX_TILE_N_ROWS);
                hmx_queue_धकेलो(ctx->hmx_queue, hmx_queue_make_desc(hmx_matmul_worker_fn, &job));
                hmx_queue_निकालो(ctx->hmx_queue);

                // D: Output Store
                भिन्न *output_chunk = dst + (mr * dst_stride + nc);
                ध्रुव भिन्न *bias_chunk = has_bias ? (vtcm_bias + mr * src2_stride + nc) : रिक्त;
                पूर्णाङ्क chunk_dst_cols = dst_cols - (पूर्णाङ्क)nc;
                यदि (chunk_dst_cols > 0) {
                    transfer_output_chunk_threaded(ctx, output_chunk, bias_chunk, vtcm_output, n_rows, n_cols, dst_stride, src2_stride, chunk_dst_cols, n_threads);
                }
            }
        }
    }

    प्रत्यागम 0;
}

स्थिर पूर्णाङ्क hmx_mm_nx_2d_f32(संरचना htp_ops_context * octx, ध्रुव संरचना htp_mm_kernel_params * kparams) {
    संरचना htp_context * ctx = octx->ctx;
    संरचना htp_thread_trace * tr = &ctx->trace[0];
    htp_trace_event_start(tr, HTP_TRACE_EVT_INIT, 0);

    ध्रुव uint32_t n_weights = kparams->n_weights;
    यदि (n_weights == 0 || n_weights > HTP_OP_MAX_OUTPUTS) {
        प्रत्यागम HTP_STATUS_INVAL_PARAMS;
    }

    ध्रुव संरचना htp_tensor * restrict src0 = octx->src[0];
    ध्रुव संरचना htp_tensor * restrict act  = octx->src[n_weights];

    ध्रुव पूर्णाङ्क weight_type = (पूर्णाङ्क) src0->type;
    ध्रुव पूर्णाङ्क k           = (पूर्णाङ्क) act->ne[0];
    ध्रुव पूर्णाङ्क k_valid     = (पूर्णाङ्क) act->ne[0];
    ध्रुव पूर्णाङ्क m           = (पूर्णाङ्क) (act->ne[1] * act->ne[2] * act->ne[3]);
    ध्रुव uint32_t act_elem_size = (act->type == HTP_TYPE_F16) ? परिमाण(__fp16) : परिमाण(भिन्न);
    ध्रुव पूर्णाङ्क act_stride  = (पूर्णाङ्क) (act->nb[1] / act_elem_size);
    ध्रुव dma_addr_t act_dma_addr = act->data;

    यदि (k % 32 != 0) { प्रत्यागम HTP_STATUS_NO_SUPPORT; }

    माप_प्रकार row_stride = htp_tensor_get_row_size(weight_type, k);
    यदि (row_stride == 0) {
        प्रत्यागम HTP_STATUS_NO_SUPPORT;
    }

    ध्रुव पूर्णाङ्क n_k_tiles = k / HTP_MM_HMX_TILE_N_COLS;
    ध्रुव पूर्णाङ्क n_weight_tiles = k / htp_tensor_get_tile_k(weight_type);
    ध्रुव संरचना fastdiv_मूल्यावली n_k_tiles_div = init_fastdiv_मूल्यावली(n_k_tiles);

    ध्रुव पूर्णाङ्क tile_size       = kparams->tile_size;
    ध्रुव पूर्णाङ्क aligned_tile_size = kparams->aligned_tile_size;

    ध्रुव bool is_tiled       = (tile_size > 0);
    tiled_dequant_task_fn_t dequant_worker_fn = hmx_get_dequant_worker_fn(weight_type, is_tiled);
    ध्रुव माप_प्रकार vtcm_budget  = ctx->vtcm_size;

    ध्रुव पूर्णाङ्क m_chunk_n_rows  = kparams->m_chunk;
    ध्रुव पूर्णाङ्क n_chunk_n_cols  = kparams->n_chunk;
    ध्रुव पूर्णाङ्क pipeline        = kparams->pipeline;
    ध्रुव पूर्णाङ्क n_threads       = octx->n_threads;
    ध्रुव पूर्णाङ्क act_threads     = kparams->n_act_threads;
    ध्रुव संरचना fastdiv_मूल्यावली * act_threads_div = &kparams->div_n_act_threads;
    ध्रुव संरचना fastdiv_मूल्यावली * k_div           = &kparams->div_ne00_padded;

    ध्रुव uint32_t dma_dst_stride  = is_tiled ? aligned_tile_size : row_stride;
    ध्रुव uint32_t dma_width_bytes = is_tiled ? tile_size : row_stride;

    संरचना htp_mm_hmx_vtcm_layout L;
    htp_mm_hmx_vtcm_layout_build(&L, HTP_MM_KERNEL_HMX_2D, weight_type, k, m_chunk_n_rows, n_chunk_n_cols, 1, pipeline, act_threads, aligned_tile_size, 0);

    यदि (L.total_bytes > vtcm_budget) {
        FARF(ERROR, "hmx-mm-nx-2d: VTCM overflow: used %zu budget %zu, m %d k %d mc %d nc %d",
             L.total_bytes, vtcm_budget, m, k, m_chunk_n_rows, n_chunk_n_cols);
        प्रत्यागम HTP_STATUS_VTCM_TOO_SMALL;
    }

    uint8_t * ध्रुव base = (uint8_t *) ctx->vtcm_base;
    __fp16  *vtcm_weight_raw[2] = {
        VTCM_LAYOUT_PTR(__fp16, base, L.off_weight[0]),
        VTCM_LAYOUT_PTR_OPTIONAL(__fp16, base, L.off_weight[1], pipeline)
    };
    __fp16  *vtcm_f16_act    = VTCM_LAYOUT_PTR(__fp16, base, L.off_act);
    भिन्न   *vtcm_f32_act    = VTCM_LAYOUT_PTR(भिन्न, base, L.off_act_f32);
    __fp16  *vtcm_output     = VTCM_LAYOUT_PTR(__fp16, base, L.off_dst[0]);
    शून्य    *vtcm_scratch0   = VTCM_LAYOUT_PTR(शून्य, base, L.off_scratch[0]);
    शून्य    *vtcm_scratch1   = VTCM_LAYOUT_PTR_OPTIONAL(शून्य, base, L.off_scratch[1], pipeline);
    शून्य    *vtcm_scratch2   = VTCM_LAYOUT_PTR_OPTIONAL(शून्य, base, L.off_dst[1], pipeline);
    __fp16  *vtcm_scales     = VTCM_LAYOUT_PTR(__fp16, base, L.off_scales);

    hmx_init_column_scales(vtcm_scales, Q6_V_vsplat_R(0x3c00));  // scale: 1.0, bias: 0.0 in FP16

    पूर्णाङ्क m_start = 0;
    पूर्णाङ्क m_rows  = m;
    यदि (octx->ctx->mdev.गिनती > 1) {
        bool can_split = htp_tensor_can_row_partition(act, act_elem_size);
        यदि (can_split) {
            प्रति (uint32_t p = 0; p < n_weights; ++p) {
                ध्रुव संरचना htp_tensor * restrict dst = octx->dsts[p];
                यदि (!htp_tensor_can_row_partition(dst, परिमाण(भिन्न))) {
                    can_split = false;
                    भङ्ग;
                }
            }
        }
        ध्रुव संरचना htp_tensor_mdev_range range = htp_tensor_mdev_partition((uint32_t) m, can_split ? 1 : 0, octx->ctx->mdev.idx, octx->ctx->mdev.गिनती, &octx->ctx->mdev.गिनती_div);
        m_start = (पूर्णाङ्क) range.start;
        m_rows  = (पूर्णाङ्क) range.गिनती;
    }

    यदि (m_rows == 0) {
        प्रत्यागम HTP_STATUS_OK;
    }

    FARF(ऊपरी_2, "hmx-mm-nx-2d: n_weights %u m %d (%d..%d) k %d wtype %d mc %d nc %d vtcm %zu/%zu",
         n_weights, m, m_start, m_start + m_rows, k, weight_type, m_chunk_n_rows, n_chunk_n_cols, L.total_bytes, vtcm_budget);

    htp_trace_event_stop(tr, HTP_TRACE_EVT_INIT, 0);

    ध्रुव माप_प्रकार mr_end = (माप_प्रकार)(m_start + m_rows);

    यदि (pipeline) {
        hmx_matmul_job_t job_slots[2];
        शून्य *vtcm_raw_bufs[4] = {
            vtcm_weight_raw[0],
            vtcm_weight_raw[1],
            dequant_worker_fn ? vtcm_weight_raw[0] : vtcm_scratch0,
            dequant_worker_fn ? vtcm_weight_raw[1] : vtcm_scratch1,
        };
        शून्य *vtcm_weight_bufs[4] = {
            dequant_worker_fn ? vtcm_scratch0 : vtcm_weight_raw[0],
            dequant_worker_fn ? vtcm_scratch1 : vtcm_weight_raw[1],
            vtcm_scratch0,
            vtcm_scratch1,
        };
        शून्य *vtcm_output_bufs[2] = { vtcm_output, vtcm_scratch2 };

        प्रति (माप_प्रकार mr = (माप_प्रकार) m_start; mr < mr_end; mr += m_chunk_n_rows) {
            ध्रुव माप_प्रकार n_rows = hex_smin(mr_end - mr, m_chunk_n_rows);

            संरचना activation_transfer_params act_params = {
                .ctx = ctx,
                .dst = vtcm_f16_act,
                .act_dma_addr = act_dma_addr + (माप_प्रकार) mr * act_stride * act_elem_size,
                .n_rows = (पूर्णाङ्क) n_rows,
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

            प्रति (uint32_t p = 0; p < n_weights; p++) {
                ध्रुव संरचना htp_tensor * restrict src_w = octx->src[p];
                ध्रुव संरचना htp_tensor * restrict dst   = octx->dsts[p];
                यदि (!src_w || !dst) अनुवर्त;

                ध्रुव dma_addr_t weight       = src_w->data;
                dma_queue * weight_dma      = octx->ctx->dma[0];
                भिन्न * dst_ptr            = (भिन्न *) dst->data;
                ध्रुव माप_प्रकार n             = src_w->ne[1];
                यदि (n == 0) अनुवर्त;
                ध्रुव माप_प्रकार weight_stride = src_w->nb[1];
                ध्रुव माप_प्रकार dst_stride    = dst->nb[1] / परिमाण(भिन्न);
                ध्रुव पूर्णाङ्क dst_cols         = (पूर्णाङ्क) dst->ne[0];
                ध्रुव पूर्णाङ्क n_chunk_cnt      = hmx_ceil_div(n, n_chunk_n_cols);

                ध्रुव uint32_t dma_src_stride = is_tiled ? tile_size : weight_stride;

                ध्रुव माप_प्रकार   n_cols_A0 = hex_smin(n - 0 * n_chunk_n_cols, n_chunk_n_cols);
                ध्रुव uint32_t height_A0 = is_tiled ? hmx_ceil_div(n_cols_A0, 32) * n_weight_tiles : n_cols_A0;
                dma_queue_धकेलो(weight_dma, dma_make_data(vtcm_raw_bufs[0], weight),
                               dma_dst_stride, dma_src_stride, dma_width_bytes, height_A0);

                यदि (1 < n_chunk_cnt) {
                    ध्रुव माप_प्रकार   n_cols_A1 = hex_smin(n - 1 * n_chunk_n_cols, n_chunk_n_cols);
                    ध्रुव uint32_t height_A1 = is_tiled ? hmx_ceil_div(n_cols_A1, 32) * n_weight_tiles : n_cols_A1;
                    dma_queue_धकेलो(weight_dma, dma_make_data(vtcm_raw_bufs[1], weight + n_chunk_n_cols * weight_stride),
                                   dma_dst_stride, dma_src_stride, dma_width_bytes, height_A1);
                }

                प्रति (पूर्णाङ्क i = 0; i < n_chunk_cnt; ++i) {
                    ध्रुव माप_प्रकार nc    = i * n_chunk_n_cols;
                    ध्रुव माप_प्रकार nc_p2 = nc + 2 * n_chunk_n_cols;

                    ध्रुव माप_प्रकार n_cols    = hex_smin(n - nc, n_chunk_n_cols);
                    ध्रुव माप_प्रकार n_cols_p2 = hex_smin(n - nc_p2, n_chunk_n_cols);

                    शून्य * curr_raw = (शून्य *) dma_queue_निकालो(weight_dma).dst;

                    dequantize_tiled_weight_chunk_to_fp16_tiles(
                        ctx, vtcm_weight_bufs[i % 4], curr_raw,
                        n_cols, k, row_stride, weight_type,
                        n_k_tiles, n_k_tiles_div, dequant_worker_fn, n_threads);

                    यदि (i + 2 < n_chunk_cnt) {
                        ध्रुव uint32_t height_p2 = is_tiled ? hmx_ceil_div(n_cols_p2, 32) * n_weight_tiles : n_cols_p2;
                        dma_queue_धकेलो(weight_dma, dma_make_data(vtcm_raw_bufs[(i + 2) % 4], weight + nc_p2 * weight_stride),
                                       dma_dst_stride, dma_src_stride, dma_width_bytes, height_p2);
                    }

                    hmx_matmul_job_init(&job_slots[i % 2], (__fp16 *) vtcm_output_bufs[i % 2],
                                        (__fp16 *) vtcm_f16_act, (__fp16 *) vtcm_weight_bufs[i % 4],
                                        vtcm_scales, hmx_ceil_div(n_rows, HTP_MM_HMX_TILE_N_ROWS),
                                        hmx_ceil_div(n_cols, HTP_MM_HMX_TILE_N_COLS), k / HTP_MM_HMX_TILE_N_ROWS);
                    hmx_queue_धकेलो(ctx->hmx_queue, hmx_queue_make_desc(hmx_matmul_worker_fn, &job_slots[i % 2]));

                    यदि (i > 0) {
                        hmx_queue_निकालो(ctx->hmx_queue);
                        ध्रुव माप_प्रकार nc_prev = (i - 1) * n_chunk_n_cols;
                        ध्रुव माप_प्रकार n_cols_prev = hex_smin(n - nc_prev, n_chunk_n_cols);
                        भिन्न *output_chunk = dst_ptr + (mr * dst_stride + nc_prev);
                        पूर्णाङ्क chunk_dst_cols = dst_cols - (पूर्णाङ्क)nc_prev;
                        यदि (chunk_dst_cols > 0) {
                            transfer_output_chunk_threaded(ctx, output_chunk, रिक्त, vtcm_output_bufs[(i - 1) % 2], n_rows, n_cols_prev, dst_stride, 0, chunk_dst_cols, n_threads);
                        }
                    }
                }

                hmx_queue_निकालो(ctx->hmx_queue);
                ध्रुव माप_प्रकार nc_last = (n_chunk_cnt - 1) * n_chunk_n_cols;
                ध्रुव माप_प्रकार n_cols_last = hex_smin(n - nc_last, n_chunk_n_cols);
                भिन्न *output_chunk = dst_ptr + (mr * dst_stride + nc_last);
                पूर्णाङ्क chunk_dst_cols = dst_cols - (पूर्णाङ्क)nc_last;
                यदि (chunk_dst_cols > 0) {
                    transfer_output_chunk_threaded(ctx, output_chunk, रिक्त, vtcm_output_bufs[(n_chunk_cnt - 1) % 2], n_rows, n_cols_last, dst_stride, 0, chunk_dst_cols, n_threads);
                }
            }
        }
    } अन्यथा {
        hmx_matmul_job_t job;
        शून्य *vtcm_raw_bufs[2] = { vtcm_weight_raw[0], dequant_worker_fn ? vtcm_weight_raw[0] : vtcm_scratch0 };
        शून्य *vtcm_weight_bufs[2] = { dequant_worker_fn ? vtcm_scratch0 : vtcm_weight_raw[0], vtcm_scratch0 };

        प्रति (माप_प्रकार mr = (माप_प्रकार) m_start; mr < mr_end; mr += m_chunk_n_rows) {
            ध्रुव माप_प्रकार n_rows = hex_smin(mr_end - mr, m_chunk_n_rows);

            संरचना activation_transfer_params act_params = {
                .ctx = ctx,
                .dst = vtcm_f16_act,
                .act_dma_addr = act_dma_addr + (माप_प्रकार) mr * act_stride * act_elem_size,
                .n_rows = (पूर्णाङ्क) n_rows,
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

            प्रति (uint32_t p = 0; p < n_weights; p++) {
                ध्रुव संरचना htp_tensor * restrict src_w = octx->src[p];
                ध्रुव संरचना htp_tensor * restrict dst   = octx->dsts[p];
                यदि (!src_w || !dst) अनुवर्त;

                ध्रुव dma_addr_t weight       = src_w->data;
                dma_queue * weight_dma      = octx->ctx->dma[0];
                भिन्न * dst_ptr            = (भिन्न *) dst->data;
                ध्रुव माप_प्रकार n             = src_w->ne[1];
                यदि (n == 0) अनुवर्त;
                ध्रुव माप_प्रकार weight_stride = src_w->nb[1];
                ध्रुव माप_प्रकार dst_stride    = dst->nb[1] / परिमाण(भिन्न);
                ध्रुव पूर्णाङ्क dst_cols         = (पूर्णाङ्क) dst->ne[0];
                ध्रुव पूर्णाङ्क n_chunk_cnt      = hmx_ceil_div(n, n_chunk_n_cols);

                ध्रुव uint32_t dma_src_stride = is_tiled ? tile_size : weight_stride;

                यदि (n > 0) {
                    ध्रुव माप_प्रकार n_cols = hex_smin(n, n_chunk_n_cols);
                    ध्रुव uint32_t height = is_tiled ? hmx_ceil_div(n_cols, 32) * n_weight_tiles : n_cols;
                    dma_queue_धकेलो(weight_dma, dma_make_data(vtcm_raw_bufs[0], weight), dma_dst_stride, dma_src_stride, dma_width_bytes, height);
                }

                प्रति (पूर्णाङ्क i = 0; i < n_chunk_cnt; ++i) {
                    ध्रुव माप_प्रकार nc = (माप_प्रकार) i * n_chunk_n_cols;
                    ध्रुव माप_प्रकार n_cols = hex_smin(n - nc, n_chunk_n_cols);
                    ध्रुव माप_प्रकार n_row_tiles = hmx_ceil_div(n_rows, HTP_MM_HMX_TILE_N_ROWS);
                    ध्रुव माप_प्रकार n_col_tiles = hmx_ceil_div(n_cols, HTP_MM_HMX_TILE_N_COLS);

                    शून्य * curr_raw = (शून्य *) dma_queue_निकालो(weight_dma).dst;

                    dequantize_tiled_weight_chunk_to_fp16_tiles(
                        ctx, vtcm_weight_bufs[i % 2], curr_raw,
                        n_cols, k, row_stride, weight_type,
                        n_k_tiles, n_k_tiles_div, dequant_worker_fn, n_threads);

                    ध्रुव माप_प्रकार nc_next = nc + n_chunk_n_cols;
                    यदि (nc_next < n) {
                        ध्रुव माप_प्रकार n_cols_next = hex_smin(n - nc_next, n_chunk_n_cols);
                        ध्रुव uint32_t height_next = is_tiled ? hmx_ceil_div(n_cols_next, 32) * n_weight_tiles : n_cols_next;
                        dma_queue_धकेलो(weight_dma, dma_make_data(vtcm_raw_bufs[(i + 1) % 2], weight + nc_next * weight_stride), dma_dst_stride, dma_src_stride, dma_width_bytes, height_next);
                    }

                    hmx_matmul_job_init(&job, vtcm_output, vtcm_f16_act, (__fp16 *) vtcm_weight_bufs[i % 2], vtcm_scales, n_row_tiles, n_col_tiles, k / HTP_MM_HMX_TILE_N_ROWS);
                    hmx_queue_धकेलो(ctx->hmx_queue, hmx_queue_make_desc(hmx_matmul_worker_fn, &job));
                    hmx_queue_निकालो(ctx->hmx_queue);

                    भिन्न *output_chunk = dst_ptr + (mr * dst_stride + nc);
                    पूर्णाङ्क chunk_dst_cols = dst_cols - (पूर्णाङ्क)nc;
                    यदि (chunk_dst_cols > 0) {
                        transfer_output_chunk_threaded(ctx, output_chunk, रिक्त, vtcm_output, n_rows, n_cols, dst_stride, 0, chunk_dst_cols, n_threads);
                    }
                }
            }
        }
    }

    प्रत्यागम HTP_STATUS_OK;
}

स्थिर अन्तःस्थ dma_addr_t hmx_mm_weight_batch_data(ध्रुव hmx_mm_batched_params_t *params,
                                                         पूर्णाङ्क dst_b2, पूर्णाङ्क dst_b3) {
    ध्रुव माप_प्रकार b2_idx = (params->r2 <= 1) ? (माप_प्रकार) dst_b2 : (माप_प्रकार) fastdiv((uint32_t) dst_b2, &params->div_r2);
    ध्रुव माप_प्रकार b3_idx = (params->r3 <= 1) ? (माप_प्रकार) dst_b3 : (माप_प्रकार) fastdiv((uint32_t) dst_b3, &params->div_r3);
    प्रत्यागम params->weight + b2_idx * params->src0_nb2 + b3_idx * params->src0_nb3;
}

स्थिर अन्तःस्थ dma_addr_t hmx_mm_act_batch_addr(ध्रुव hmx_mm_batched_params_t *params,
                                                पूर्णाङ्क dst_b2, पूर्णाङ्क dst_b3) {
    प्रत्यागम params->act_dma_addr + dst_b2 * params->act_nb2 +
           dst_b3 * params->act_nb3;
}

स्थिर अन्तःस्थ भिन्न *hmx_mm_dst_batch_ptr(ध्रुव hmx_mm_batched_params_t *params,
                                              पूर्णाङ्क dst_b2, पूर्णाङ्क dst_b3) {
    प्रत्यागम (भिन्न *) ((uint8_t *) params->dst +
                      (माप_प्रकार) dst_b2 * params->dst_nb2 +
                      (माप_प्रकार) dst_b3 * params->dst_nb3);
}

स्थिर पूर्णाङ्क hmx_mm_batched_simple(संरचना htp_context *ctx,
                                 ध्रुव hmx_mm_batched_params_t *params,
                                 पूर्णाङ्क m_chunk, पूर्णाङ्क n_chunk, पूर्णाङ्क pipeline, पूर्णाङ्क n_threads, पूर्णाङ्क act_threads, पूर्णाङ्क vtcm_size,
                                 ध्रुव संरचना fastdiv_मूल्यावली * act_threads_div, ध्रुव संरचना fastdiv_मूल्यावली * k_div) {
    पूर्णाङ्क ret = 0;
    प्रति (पूर्णाङ्क b3 = 0; b3 < params->ne13 && ret == 0; ++b3) {
        प्रति (पूर्णाङ्क b2 = 0; b2 < params->ne12 && ret == 0; ++b2) {
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
    प्रत्यागम ret;
}

स्थिर पूर्णाङ्क hmx_mm_batched(संरचना htp_context *ctx, ध्रुव hmx_mm_batched_params_t *params,
                          पूर्णाङ्क m_chunk, पूर्णाङ्क n_chunk, पूर्णाङ्क pipeline, पूर्णाङ्क n_threads, पूर्णाङ्क act_threads,
                          ध्रुव संरचना fastdiv_मूल्यावली * act_threads_div,
                          ध्रुव संरचना fastdiv_मूल्यावली * k_div,
                          पूर्णाङ्क vtcm_size) {
    ध्रुव माप_प्रकार row_stride = htp_tensor_get_row_size(params->weight_type, params->k);
    यदि (row_stride == 0) { प्रत्यागम -1; }
    यदि (params->act_stride < params->k || (params->tile_size == 0 && (माप_प्रकार) params->weight_stride < row_stride) || params->dst_stride < params->dst_cols) { प्रत्यागम -1; }
    यदि (params->ne02 <= 0 || params->ne03 <= 0 || params->ne12 <= 0 || params->ne13 <= 0) { प्रत्यागम -1; }
    यदि (params->ne12 % params->ne02 != 0 || params->ne13 % params->ne03 != 0) { प्रत्यागम -1; }
    // N (the weight row count) does not have to be 32-aligned:
    यदि (params->k % 32 != 0) { प्रत्यागम -1; }
    यदि (!hex_is_aligned(params->dst, परिमाण(भिन्न))) { प्रत्यागम -1; }

    ध्रुव पूर्णाङ्क group_size = params->r2;
    ध्रुव माप_प्रकार vtcm_budget  = ctx->vtcm_size;

    // Check if the precomputed parameters are grouped or simple.
    // If simple, or if group_size <= 1, we use simple fallback loop.
    // Grouped path is only valid for flat F16 if group_size > 1 and it fits within VTCM budget.
    bool run_grouped = (params->weight_type == HTP_TYPE_F16 && params->tile_size == 0 && group_size > 1 && (माप_प्रकार)vtcm_size <= vtcm_budget);
    यदि (!run_grouped) {
        प्रत्यागम hmx_mm_batched_simple(ctx, params, m_chunk, n_chunk, pipeline, n_threads, act_threads, vtcm_size, act_threads_div, k_div);
    }

    संरचना htp_thread_trace * tr = &ctx->trace[0];
    htp_trace_event_start(tr, HTP_TRACE_EVT_INIT, 0);

    माप_प्रकार m_chunk_n_rows = m_chunk;
    माप_प्रकार n_chunk_n_cols = n_chunk;

    संरचना htp_mm_hmx_vtcm_layout L;
    htp_mm_hmx_vtcm_layout_build(&L, HTP_MM_KERNEL_HMX_BATCHED, HTP_TYPE_F16, params->k, m_chunk_n_rows, n_chunk_n_cols, group_size, false, act_threads, 0, params->bias_bytes);

    यदि (L.total_bytes > vtcm_budget) {
        FARF(ऊपरी_2, "%s: grouped layout overflowed VTCM, falling back to simple batched loop", __func__);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_INIT, 0);
        प्रत्यागम hmx_mm_batched_simple(ctx, params, m_chunk, n_chunk, pipeline, n_threads, act_threads, vtcm_size, act_threads_div, k_div);
    }

    uint8_t * ध्रुव base = (uint8_t *) ctx->vtcm_base;
    __fp16  *vtcm_weight     = VTCM_LAYOUT_PTR(__fp16, base, L.off_weight[0]);
    __fp16  *vtcm_f16_act    = VTCM_LAYOUT_PTR(__fp16, base, L.off_act);
    __fp16  *vtcm_output     = VTCM_LAYOUT_PTR(__fp16, base, L.off_dst[0]);
    शून्य    *vtcm_scratch0   = VTCM_LAYOUT_PTR(शून्य, base, L.off_scratch[0]);
    शून्य    *vtcm_scratch1   = VTCM_LAYOUT_PTR(शून्य, base, L.off_scratch[1]);
    __fp16  *vtcm_scales     = VTCM_LAYOUT_PTR(__fp16, base, L.off_scales);
    भिन्न   *vtcm_f32_act    = VTCM_LAYOUT_PTR(भिन्न, base, L.off_act_f32);

    ध्रुव bool has_bias      = (params->bias_bytes > 0 && params->bias_addr != 0);
    भिन्न   *vtcm_bias       = VTCM_LAYOUT_PTR_OPTIONAL(भिन्न, base, L.off_bias, has_bias);
    यदि (has_bias) {
        dma_queue_धकेलो(params->weight_dma, dma_make_data(vtcm_bias, params->bias_addr), hex_align_up(params->bias_bytes, 128), 0, params->bias_bytes, 1);
        dma_queue_निकालो(params->weight_dma);
    }

    hmx_init_column_scales(vtcm_scales, Q6_V_vsplat_R(0x3c00));  // scale: 1.0, bias: 0.0 in FP16

    FARF(ऊपरी_2, "%s: grouped path m=%d k=%d n=%d group=%d streams=%d mc=%zu nc=%zu vtcm=%zu/%zu",
            __func__, params->m, params->k, params->n, group_size, params->ne13,
            m_chunk_n_rows, n_chunk_n_cols,
            L.total_bytes, vtcm_budget);

    ध्रुव माप_प्रकार fp16_row_bytes   = (माप_प्रकार) params->k * परिमाण(__fp16);
    ध्रुव माप_प्रकार weight_row_bytes = (माप_प्रकार) params->weight_stride;

    htp_trace_event_stop(tr, HTP_TRACE_EVT_INIT, 0);

    hmx_matmul_job_t job;

    प्रति (पूर्णाङ्क b3 = 0; b3 < params->ne13; ++b3) {
        प्रति (पूर्णाङ्क b2_base = 0; b2_base < params->ne12; b2_base += group_size) {
            ध्रुव dma_addr_t weight_group = hmx_mm_weight_batch_data(params, b2_base, b3);
            dma_queue * weight_dma = params->weight_dma;

            प्रति (माप_प्रकार mr = 0; mr < (माप_प्रकार) params->m; mr += m_chunk_n_rows) {
                ध्रुव माप_प्रकार n_rows = hex_smin((माप_प्रकार) params->m - mr, m_chunk_n_rows);
                ध्रुव माप_प्रकार n_row_tiles = hmx_ceil_div((पूर्णाङ्क) n_rows, HTP_MM_HMX_TILE_N_ROWS);

                // Pre-load activations for all heads in the group (once per m_chunk).
                // When the source is strided (permuted Q), use 2D DMA to gather
                // contiguous rows into a VTCM scratch buffer first, then HVX
                // converts from the contiguous VTCM buffer.  This avoids L2 cache
                // thrashing from HVX loads at large strides.
                प्रति (पूर्णाङ्क g = 0; g < group_size; ++g) {
                    ध्रुव dma_addr_t act_dma_addr = hmx_mm_act_batch_addr(params, b2_base + g, b3) +
                                                      (माप_प्रकार) mr * params->act_stride * params->act_elem_size;
                    __fp16 *vtcm_act_g = vtcm_f16_act + (माप_प्रकार) g * L.act_head_stride;
                    संरचना activation_transfer_params act_params = {
                        .ctx = ctx,
                        .dst = vtcm_act_g,
                        .act_dma_addr = act_dma_addr,
                        .n_rows = (पूर्णाङ्क) n_rows,
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
                    ध्रुव माप_प्रकार n_cols_first = hex_smin((माप_प्रकार) params->n, n_chunk_n_cols);
                    dma_queue_धकेलो(weight_dma, dma_make_data(vtcm_scratch0, weight_group),
                                      fp16_row_bytes, weight_row_bytes, fp16_row_bytes, n_cols_first);
                }
                यदि (n_chunk_n_cols < (माप_प्रकार) params->n) {
                    ध्रुव माप_प्रकार n_cols_second = hex_smin((माप_प्रकार) params->n - n_chunk_n_cols, n_chunk_n_cols);
                    ध्रुव dma_addr_t second_weight_chunk = weight_group + n_chunk_n_cols * weight_row_bytes;
                    dma_queue_धकेलो(weight_dma, dma_make_data(vtcm_scratch1, second_weight_chunk),
                                      fp16_row_bytes, weight_row_bytes, fp16_row_bytes, n_cols_second);
                }

                प्रति (माप_प्रकार nc = 0; nc < (माप_प्रकार) params->n; nc += n_chunk_n_cols) {
                    ध्रुव माप_प्रकार n_cols      = hex_smin((माप_प्रकार) params->n - nc, n_chunk_n_cols);
                    ध्रुव माप_प्रकार n_col_tiles = hmx_ceil_div((पूर्णाङ्क) n_cols, HTP_MM_HMX_TILE_N_COLS);

                    {
                        शून्य * curr_raw = (शून्य *) dma_queue_निकालो(weight_dma).dst;

                        ध्रुव माप_प्रकार n_cols_tiled = hex_align_up(n_cols, HTP_MM_HMX_TILE_N_COLS);

                        hmx_interleave_rows_to_tiles(vtcm_weight, (ध्रुव __fp16 *) curr_raw, (uint32_t) n_cols, params->k, params->k, 0, (uint32_t) n_cols_tiled);

                        ध्रुव माप_प्रकार nc_next = nc + n_chunk_n_cols * 2;
                        यदि (nc_next < (माप_प्रकार) params->n) {
                            ध्रुव माप_प्रकार n_cols_next = hex_smin((माप_प्रकार) params->n - nc_next, n_chunk_n_cols);
                            ध्रुव dma_addr_t next_weight_chunk = weight_group + nc_next * weight_row_bytes;

                            dma_queue_धकेलो(weight_dma, dma_make_data(curr_raw, next_weight_chunk),
                                              fp16_row_bytes, weight_row_bytes, fp16_row_bytes, n_cols_next);
                        }
                    }

                    // Reuse the interleaved weight for every q_head in this GQA group
                    प्रति (पूर्णाङ्क g = 0; g < group_size; ++g) {
                        {
                            ध्रुव __fp16 * vtcm_act_g = vtcm_f16_act + (माप_प्रकार) g * L.act_head_stride;
                            hmx_matmul_job_init(&job, vtcm_output, vtcm_act_g, vtcm_weight, vtcm_scales, n_row_tiles, n_col_tiles, params->k / 32);
                            hmx_queue_धकेलो(ctx->hmx_queue, hmx_queue_make_desc(hmx_matmul_worker_fn, &job));
                            hmx_queue_निकालो(ctx->hmx_queue);
                        }

                        {
                            भिन्न *output = hmx_mm_dst_batch_ptr(params, b2_base + g, b3) + mr * params->dst_stride + nc;
                            ध्रुव भिन्न *bias_chunk = has_bias ? (vtcm_bias + mr * params->bias_stride + nc) : रिक्त;
                            पूर्णाङ्क chunk_dst_cols = params->dst_cols - (पूर्णाङ्क)nc;
                            यदि (chunk_dst_cols > 0) {
                                transfer_output_chunk_threaded(ctx, output, bias_chunk, vtcm_output, (पूर्णाङ्क) n_rows, (पूर्णाङ्क) n_cols,
                                                               params->dst_stride, params->bias_stride, chunk_dst_cols, n_threads);
                            }
                        }
                    }
                }
            }
        }
    }

    प्रत्यागम 0;
}

स्थिर शून्य transfer_activation_chunk_gathered_threaded(
            संरचना htp_context *ctx,
            __fp16 *dst,
            ध्रुव भिन्न *src,
            पूर्णाङ्क start_row,
            पूर्णाङ्क n_rows,
            पूर्णाङ्क k_block,
            ध्रुव संरचना mmid_row_mapping *matrix_rows,
            पूर्णाङ्क cur_a,
            पूर्णाङ्क mapping_stride,
            पूर्णाङ्क ne11,
            माप_प्रकार nb11,
            माप_प्रकार nb12,
            पूर्णाङ्क cne1,
            पूर्णाङ्क n_threads,
            पूर्णाङ्क k_valid) {
    यदि (n_rows <= 0) प्रत्यागम;
    पूर्णाङ्क chunks_per_thread = hmx_ceil_div(n_rows, n_threads);
    chunks_per_thread = hex_align_up(chunks_per_thread, 2);

    पूर्णाङ्क actual_threads = hmx_ceil_div(n_rows, chunks_per_thread);

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
        .ne11_div          = ne11 > 1 ? init_fastdiv_मूल्यावली(ne11) : (संरचना fastdiv_मूल्यावली){0, 0},
        .nb11              = nb11,
        .nb12              = nb12,
        .start_row         = start_row,
        .cne1              = cne1,
        .k_valid           = k_valid,
        .traces            = ctx->trace,
    };

    worker_callback_t worker_fn = ne11 == 1 ? transfer_activation_chunk_gathered_worker_flat_fn :
                                              transfer_activation_chunk_gathered_worker_fn;

    यदि (actual_threads <= 1) {
        worker_fn(1, 0, &state);
    } अन्यथा {
        worker_pool_run_func(ctx->worker_pool, worker_fn, &state, actual_threads);
    }
}

स्थिर शून्य transfer_output_chunk_scattered_threaded(
            संरचना htp_context *ctx,
            भिन्न *dst,
            ध्रुव __fp16 *vtcm_src,
            पूर्णाङ्क start_row,
            पूर्णाङ्क n_rows,
            पूर्णाङ्क n_cols,
            ध्रुव संरचना mmid_row_mapping *matrix_rows,
            पूर्णाङ्क cur_a,
            पूर्णाङ्क mapping_stride,
            माप_प्रकार dst_nb1,
            माप_प्रकार dst_nb2,
            पूर्णाङ्क cne1,
            पूर्णाङ्क n_threads) {
    यदि (n_rows <= 0) प्रत्यागम;
    पूर्णाङ्क chunks_per_thread = hmx_ceil_div(n_rows, n_threads);
    chunks_per_thread = hex_align_up(chunks_per_thread, 2);

    पूर्णाङ्क actual_threads = hmx_ceil_div(n_rows, chunks_per_thread);

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

    यदि (actual_threads <= 1) {
        transfer_output_chunk_scattered_worker_fn(1, 0, &state);
    } अन्यथा {
        worker_pool_run_func(ctx->worker_pool, transfer_output_chunk_scattered_worker_fn, &state, actual_threads);
    }
}

स्थिर पूर्णाङ्क hmx_mm_id_2d_f32(संरचना htp_context *ctx,
                                         dma_queue *weight_dma,
                                         भिन्न *restrict dst,
                                         ध्रुव भिन्न *activation,
                                         dma_addr_t weight,
                                         पूर्णाङ्क m, पूर्णाङ्क k, पूर्णाङ्क n,
                                         पूर्णाङ्क k_valid,
                                         पूर्णाङ्क ne11,
                                         माप_प्रकार act_nb1, माप_प्रकार act_nb2,
                                         माप_प्रकार dst_nb1, माप_प्रकार dst_nb2,
                                         पूर्णाङ्क weight_stride,
                                         पूर्णाङ्क weight_type,
                                         पूर्णाङ्क tile_size,
                                         पूर्णाङ्क aligned_tile_size,
                                         ध्रुव संरचना mmid_row_mapping *matrix_rows,
                                         पूर्णाङ्क cur_a,
                                         पूर्णाङ्क mapping_stride,
                                         पूर्णाङ्क m_start,
                                         पूर्णाङ्क m_end,
                                         पूर्णाङ्क n_threads) {
    संरचना htp_thread_trace * tr = &ctx->trace[0];
    htp_trace_event_start(tr, HTP_TRACE_EVT_INIT, 0);

    ध्रुव पूर्णाङ्क cne1 = m;
    ध्रुव पूर्णाङ्क m_core = m_end - m_start;
    ध्रुव पूर्णाङ्क m_core_padded = hex_align_up(m_core > 0 ? m_core : 1, 32);

    यदि (k % 32 != 0 || n % 32 != 0) { प्रत्यागम -1; }
    यदि (!hex_is_aligned(dst, VLEN) || !hex_is_aligned(activation, VLEN)) { प्रत्यागम -1; }

    माप_प्रकार row_stride = htp_tensor_get_row_size(weight_type, k);
    यदि (row_stride == 0) {
        प्रत्यागम -1;
    }

    ध्रुव पूर्णाङ्क n_k_tiles = k / HTP_MM_HMX_TILE_N_COLS;
    ध्रुव uint32_t n_weight_tiles = k / htp_tensor_get_tile_k(weight_type);
    ध्रुव संरचना fastdiv_मूल्यावली n_k_tiles_div = init_fastdiv_मूल्यावली(n_k_tiles);

    ध्रुव माप_प्रकार vec_dot_size = k * परिमाण(__fp16);
    ध्रुव माप_प्रकार vtcm_budget  = ctx->vtcm_size;
    माप_प्रकार vtcm_used = 0;

    ध्रुव bool is_tiled = (tile_size > 0);
    tiled_dequant_task_fn_t dequant_worker_fn = hmx_get_dequant_worker_fn(weight_type, is_tiled);

    ध्रुव uint32_t dma_dst_stride  = is_tiled ? aligned_tile_size : row_stride;
    ध्रुव uint32_t dma_src_stride  = is_tiled ? tile_size : weight_stride;
    ध्रुव uint32_t dma_width_bytes = is_tiled ? tile_size : row_stride;

    ध्रुव माप_प्रकार qweight_row_stride = is_tiled ? (माप_प्रकार)(n_weight_tiles * aligned_tile_size) / 32 : 0;
    ध्रुव माप_प्रकार weight_row_stride = is_tiled ? qweight_row_stride : row_stride;

    माप_प्रकार size_per_n = 0, size_per_m = 0, size_per_mn = 0;
    htp_mm_hmx_get_2d_chunk_costs(weight_type, k, /*pipeline=*/false, aligned_tile_size,
                                  &size_per_n, &size_per_m, &size_per_mn);

    ध्रुव माप_प्रकार overhead = htp_mm_hmx_get_2d_overhead(/*pipeline=*/false, /*is_matmul_id=*/true);
    माप_प्रकार m_chunk_n_rows = 0, n_chunk_n_cols = 0;
    यदि (htp_mm_hmx_compute_chunks(vtcm_budget, overhead, size_per_n, size_per_m, size_per_mn,
                           m_core_padded, n,
                           /*m_block_cost=*/(माप_प्रकार) n * HTP_MM_HMX_COST_W_DEQUANT,
                           /*n_block_cost=*/(माप_प्रकार) m_core_padded * HTP_MM_HMX_COST_A_CONVERT, &m_chunk_n_rows, &n_chunk_n_cols, &vtcm_used)) {
        FARF(ERROR, "hmx-mm-id-2d: VTCM too small : m %d k %d n %d budget %zu", m_core_padded, k, n, vtcm_budget);
        प्रत्यागम -1;
    }

    ध्रुव माप_प्रकार weight_area_size = hex_align_up(n_chunk_n_cols * weight_row_stride, HTP_MM_HMX_TILE_SIZE);
    ध्रुव माप_प्रकार act_area_size    = hex_align_up(m_chunk_n_rows * vec_dot_size, HTP_MM_HMX_TILE_SIZE);
    ध्रुव माप_प्रकार output_area_size = hex_align_up(m_chunk_n_rows * n_chunk_n_cols * परिमाण(__fp16), HTP_MM_HMX_TILE_SIZE);

    माप_प्रकार scratch0_size = hex_align_up(n_chunk_n_cols * vec_dot_size, HTP_MM_HMX_TILE_SIZE);

    uint8_t *vtcm_ptr      = (uint8_t *) ctx->vtcm_base;
    __fp16  *vtcm_weight   = weight_area_size ? (__fp16 *) vtcm_seq_alloc(&vtcm_ptr, weight_area_size) : रिक्त;
    __fp16  *vtcm_f16_act  = (__fp16 *) vtcm_seq_alloc(&vtcm_ptr, act_area_size);
    __fp16  *vtcm_output   = (__fp16 *) vtcm_seq_alloc(&vtcm_ptr, output_area_size);
    शून्य    *vtcm_scratch0 = vtcm_seq_alloc(&vtcm_ptr, scratch0_size);
    __fp16  *vtcm_scales   = (__fp16 *) vtcm_seq_alloc(&vtcm_ptr, 256);

    vtcm_used = vtcm_ptr - (uint8_t *) ctx->vtcm_base;
    यदि (vtcm_used > vtcm_budget) {
        FARF(ERROR, "hmx-mm-id-2d: VTCM overflow: used %zu budget %zu", vtcm_used, vtcm_budget);
        प्रत्यागम -1;
    }

    hmx_init_column_scales(vtcm_scales, Q6_V_vsplat_R(0x3c00));

    htp_trace_event_stop(tr, HTP_TRACE_EVT_INIT, 0);

    hmx_matmul_job_t job;
    शून्य *vtcm_raw_bufs[2] = { vtcm_weight, dequant_worker_fn ? vtcm_weight : vtcm_scratch0 };
    शून्य *vtcm_weight_bufs[2] = { dequant_worker_fn ? vtcm_scratch0 : vtcm_weight, vtcm_scratch0 };
    ध्रुव पूर्णाङ्क n_chunk_cnt = hmx_ceil_div(n, n_chunk_n_cols);

    प्रति (माप_प्रकार mr = (माप_प्रकार) m_start; mr < (माप_प्रकार) m_end; mr += m_chunk_n_rows) {
        ध्रुव माप_प्रकार n_rows = hex_smin((माप_प्रकार) m_end - mr, m_chunk_n_rows);
        ध्रुव माप_प्रकार n_row_tiles = hmx_ceil_div(n_rows, HTP_MM_HMX_TILE_N_ROWS);

        transfer_activation_chunk_gathered_threaded(
            ctx, vtcm_f16_act, activation, (पूर्णाङ्क) mr, (पूर्णाङ्क) n_rows, k,
            matrix_rows, cur_a, mapping_stride, ne11, act_nb1, act_nb2, cne1, n_threads, k_valid);

        // A0: Pre-fetch the first weight chunk (nc = 0)
        यदि (n > 0) {
            ध्रुव माप_प्रकार n_cols = hex_smin((माप_प्रकार) n, n_chunk_n_cols);
            ध्रुव uint32_t height = is_tiled ? hmx_ceil_div(n_cols, 32) * n_weight_tiles : n_cols;
            dma_queue_धकेलो(weight_dma, dma_make_data(vtcm_raw_bufs[0], weight),
                           dma_dst_stride, dma_src_stride, dma_width_bytes, height);
        }

        प्रति (पूर्णाङ्क i = 0; i < n_chunk_cnt; ++i) {
            ध्रुव माप_प्रकार nc = (माप_प्रकार) i * n_chunk_n_cols;
            ध्रुव माप_प्रकार n_cols = hex_smin((माप_प्रकार) n - nc, n_chunk_n_cols);
            ध्रुव माप_प्रकार n_col_tiles = hmx_ceil_div(n_cols, HTP_MM_HMX_TILE_N_COLS);

            // A: Wait for weight DMA
            शून्य * curr_raw = (शून्य *) dma_queue_निकालो(weight_dma).dst;

            // B: Weight Dequantize (Threaded)
            dequantize_tiled_weight_chunk_to_fp16_tiles(
                ctx, vtcm_weight_bufs[i % 2], curr_raw,
                n_cols, k, row_stride, weight_type,
                n_k_tiles, n_k_tiles_div, dequant_worker_fn, n_threads
            );

            // Start weight DMA for the next chunk early
            ध्रुव माप_प्रकार nc_next = nc + n_chunk_n_cols;
            यदि (nc_next < (माप_प्रकार) n) {
                ध्रुव माप_प्रकार n_cols_next = hex_smin((माप_प्रकार) n - nc_next, n_chunk_n_cols);
                ध्रुव uint32_t height_next = is_tiled ? hmx_ceil_div(n_cols_next, 32) * n_weight_tiles : n_cols_next;
                dma_queue_धकेलो(weight_dma, dma_make_data(vtcm_raw_bufs[(i + 1) % 2], weight + nc_next * weight_stride),
                               dma_dst_stride, dma_src_stride, dma_width_bytes, height_next);
            }

            // C: HMX Compute (Queue-based)
            hmx_matmul_job_init(&job, vtcm_output, vtcm_f16_act, (__fp16 *) vtcm_weight_bufs[i % 2], vtcm_scales, n_row_tiles, n_col_tiles, k / HTP_MM_HMX_TILE_N_ROWS);
            hmx_queue_धकेलो(ctx->hmx_queue, hmx_queue_make_desc(hmx_matmul_worker_fn, &job));
            hmx_queue_निकालो(ctx->hmx_queue);

            // D: Output Store
            transfer_output_chunk_scattered_threaded(
                ctx, dst + nc, vtcm_output, (पूर्णाङ्क) mr, (पूर्णाङ्क) n_rows, (पूर्णाङ्क) n_cols,
                matrix_rows, cur_a, mapping_stride, dst_nb1, dst_nb2, cne1, n_threads);
        }
    }

    प्रत्यागम 0;
}

// --- Dispatchers and Public Entry Points ---

स्थिर पूर्णाङ्क hmx_mm_op_matmul(संरचना htp_ops_context * octx, ध्रुव संरचना htp_mm_kernel_params * kparams) {
    htp_matmul_tensors_preamble;

    पूर्णाङ्क k = (पूर्णाङ्क) src0->ne[0];
    पूर्णाङ्क n = (पूर्णाङ्क) src0->ne[1];
    ध्रुव पूर्णाङ्क m_total    = (पूर्णाङ्क) act->ne[1];
    ध्रुव uint32_t act_elem_size = (act->type == HTP_TYPE_F16) ? परिमाण(__fp16) : परिमाण(भिन्न);
    ध्रुव पूर्णाङ्क act_stride = (पूर्णाङ्क) (act->nb[1] / act_elem_size);

    पूर्णाङ्क m_start = 0;
    पूर्णाङ्क m_rows  = m_total;
    यदि (octx->ctx->mdev.गिनती > 1) {
        bool can_split = htp_tensor_can_row_partition(dst, परिमाण(भिन्न)) &&
                         htp_tensor_can_row_partition(act, act_elem_size);
        यदि (src2 && src2->ne[1] > 1 && !htp_tensor_can_row_partition(src2, परिमाण(भिन्न))) {
            can_split = false;
        }
        ध्रुव संरचना htp_tensor_mdev_range range = htp_tensor_mdev_partition((uint32_t) m_total, can_split ? 1 : 0, octx->ctx->mdev.idx, octx->ctx->mdev.गिनती, &octx->ctx->mdev.गिनती_div);
        m_start = (पूर्णाङ्क) range.start;
        m_rows  = (पूर्णाङ्क) range.गिनती;
    }

    यदि (m_rows == 0) {
        प्रत्यागम HTP_STATUS_OK;
    }

    dma_addr_t src2_addr = 0;
    माप_प्रकार src2_bytes = 0;
    uint32_t src2_stride = 0;
    माप_प्रकार src2_nb2 = 0;
    माप_प्रकार src2_nb3 = 0;
    यदि (src2) {
        src2_stride = (src2->ne[1] == 1) ? 0 : (uint32_t) (src2->nb[1] / परिमाण(भिन्न));
        src2_addr = src2->data + m_start * src2_stride * परिमाण(भिन्न);
        src2_bytes = (माप_प्रकार) kparams->vtcm_bias_size;
        src2_nb2 = (src2->ne[2] == 1) ? 0 : src2->nb[2];
        src2_nb3 = (src2->ne[3] == 1) ? 0 : src2->nb[3];
    }

    ध्रुव पूर्णाङ्क dst_stride = (पूर्णाङ्क)(dst->nb[1] / परिमाण(भिन्न));
    भिन्न       * dst_ptr = (भिन्न *) dst->data + m_start * dst_stride;
    // byte offset, act_stride is in activation elements (F16 or F32)
    ध्रुव dma_addr_t act_addr = act->data + (माप_प्रकार) m_start * act_stride * act_elem_size;

    पूर्णाङ्क ret = -1;
    ध्रुव पूर्णाङ्क n_threads = kparams->n_threads;
    यदि (kparams->kernel_type == HTP_MM_KERNEL_HMX_BATCHED) {
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
            .weight_stride     = (पूर्णाङ्क) src0->nb[1],
            .weight_type       = (पूर्णाङ्क) src0->type,
            .dst_stride        = dst_stride,
            .dst_cols          = (पूर्णाङ्क) dst->ne[0],
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
    } अन्यथा {
        ret = hmx_mm_2d_f32(
            octx->ctx, octx->ctx->dma[0], dst_ptr, src2_addr, src2_bytes,
            act_addr, src0->data,
            m_rows, k, n, act_stride, act_elem_size, (पूर्णाङ्क) src0->nb[1], (पूर्णाङ्क) src0->type, (पूर्णाङ्क) act->ne[0],
            dst_stride, src2_stride, (पूर्णाङ्क)dst->ne[0],
            kparams->m_chunk, kparams->n_chunk, kparams->pipeline, n_threads,
            kparams->n_act_threads,
            &kparams->div_n_act_threads,
            &kparams->div_ne00_padded,
            kparams->tile_size, kparams->aligned_tile_size, kparams->vtcm_size
        );
    }

    यदि (ret != 0) {
        FARF(ERROR, "HMX matmul failed (ret=%d)\n", ret);
        प्रत्यागम HTP_STATUS_INTERNAL_ERR;
    }
    प्रत्यागम HTP_STATUS_OK;
}

स्थिर अन्तःस्थ शून्य htp_mm_tensor_collapse_rows(संरचना htp_tensor * c, ध्रुव संरचना htp_tensor * t, uint32_t stride) {
    *c = *t;
    c->ne[1] = t->ne[1] * t->ne[2] * t->ne[3];
    c->ne[2] = 1;
    c->ne[3] = 1;
    c->nb[1] = stride;
    c->nb[2] = c->nb[1] * c->ne[1];
    c->nb[3] = c->nb[2];
}

स्थिर पूर्णाङ्क op_matmul_impl(संरचना htp_ops_context * octx) {
    ध्रुव संरचना htp_mm_kernel_params * kparams = (ध्रुव संरचना htp_mm_kernel_params *) octx->kernel_params;

    ध्रुव पूर्णाङ्क status = htp_mm_init_context(octx, kparams);
    यदि (status != HTP_STATUS_OK) {
        प्रत्यागम status;
    }

    यदि (kparams->n_hmx) {
        प्रत्यागम hmx_mm_op_matmul(octx, kparams);
    }

    प्रत्यागम hvx_mm_matmul(octx);
}

पूर्णाङ्क op_matmul(संरचना htp_ops_context * octx) {
    ध्रुव संरचना htp_mm_kernel_params * kparams = (ध्रुव संरचना htp_mm_kernel_params *) octx->kernel_params;

    यदि (kparams->collapse) {
        ध्रुव संरचना htp_tensor * act  = octx->src[1];
        ध्रुव संरचना htp_tensor * dst  = octx->dst;
        ध्रुव uint32_t s_act = (act->ne[1] > 1) ? act->nb[1] : ((act->ne[2] > 1) ? act->nb[2] : act->nb[3]);
        ध्रुव uint32_t sd    = (dst->ne[1] > 1) ? dst->nb[1] : ((dst->ne[2] > 1) ? dst->nb[2] : dst->nb[3]);
        संरचना htp_tensor act_collapsed, dst_collapsed;
        htp_mm_tensor_collapse_rows(&act_collapsed, act, s_act);
        htp_mm_tensor_collapse_rows(&dst_collapsed, dst, sd);
        octx->src[1] = &act_collapsed;
        octx->dst    = &dst_collapsed;

        संरचना htp_tensor src2_collapsed;
        ध्रुव संरचना htp_tensor * src2 = octx->src[2];
        यदि (src2 && (src2->ne[1] * src2->ne[2] * src2->ne[3] > 1)) {
            ध्रुव uint32_t s2 = (src2->ne[1] > 1) ? src2->nb[1] : ((src2->ne[2] > 1) ? src2->nb[2] : src2->nb[3]);
            htp_mm_tensor_collapse_rows(&src2_collapsed, src2, s2);
            octx->src[2] = &src2_collapsed;
        }

        ध्रुव पूर्णाङ्क status = op_matmul_impl(octx);
        octx->src[1] = act;
        octx->dst    = dst;
        यदि (src2) {
            octx->src[2] = src2;
        }
        प्रत्यागम status;
    }

    प्रत्यागम op_matmul_impl(octx);
}

स्थिर पूर्णाङ्क hmx_mm_op_matmul_id(
    संरचना htp_ops_context * octx,
    संरचना htp_mm_context * mmctx
) {
    ध्रुव uint32_t *                matrix_row_counts = mmctx->matrix_row_counts;
    ध्रुव संरचना mmid_row_mapping * matrix_rows       = mmctx->matrix_rows;
    htp_matmul_tensors_preamble;
    ध्रुव संरचना htp_mm_kernel_params * kparams = (ध्रुव संरचना htp_mm_kernel_params *) octx->kernel_params;
    ध्रुव पूर्णाङ्क n_ids = octx->src[2]->ne[0];
    ध्रुव पूर्णाङ्क n_as  = ne02;

    ध्रुव bool mdev_split = (octx->ctx->mdev.गिनती > 1) && htp_tensor_can_row_partition(dst, परिमाण(भिन्न));
    यदि (octx->ctx->mdev.गिनती > 1 && !mdev_split && octx->ctx->mdev.idx > 0) {
        प्रत्यागम HTP_STATUS_OK;
    }
    uint32_t n_active = 0;
    यदि (mdev_split) {
        प्रति (uint32_t a = 0; a < (uint32_t) n_as; ++a) {
            यदि (matrix_row_counts[a] > 0) n_active++;
        }
    }
    ध्रुव bool expert_split = mdev_split && (n_active >= octx->ctx->mdev.गिनती);

    uint32_t target_dev = 0;
    प्रति (uint32_t cur_a = 0; cur_a < (uint32_t) n_as; ++cur_a) {
        ध्रुव int32_t cne1 = matrix_row_counts[cur_a];
        यदि (cne1 == 0) अनुवर्त;

        ध्रुव पूर्णाङ्क m_padded = hex_align_up(cne1, 32);
        पूर्णाङ्क m_start = 0, m_end = m_padded;
        यदि (expert_split) {
            ध्रुव bool my_expert = (target_dev == octx->ctx->mdev.idx);
            यदि (++target_dev == octx->ctx->mdev.गिनती) {
                target_dev = 0;
            }
            यदि (!my_expert) अनुवर्त;
        } अन्यथा यदि (mdev_split) {
            ध्रुव bool can_split = (uint32_t) cne1 >= octx->ctx->mdev.गिनती;
            ध्रुव संरचना htp_tensor_mdev_range range = htp_tensor_mdev_partition((uint32_t) m_padded, can_split ? 1 : 0, octx->ctx->mdev.idx, octx->ctx->mdev.गिनती, &octx->ctx->mdev.गिनती_div);
            m_start = (पूर्णाङ्क) range.start;
            m_end   = (पूर्णाङ्क) (range.start + range.गिनती);
        }
        यदि (m_start >= m_end) अनुवर्त;

        पूर्णाङ्क ret = hmx_mm_id_2d_f32(octx->ctx, octx->ctx->dma[0], (भिन्न*) dst->data, (भिन्न*) act->data,
                                   src0->data + cur_a * nb02,
                                   cne1, ne00, ne01,
                                   ne10,
                                   ne11,
                                   nb11, nb12,
                                   nb1, nb2,
                                   (पूर्णाङ्क) src0->nb[1], (पूर्णाङ्क) src0->type,
                                   kparams->tile_size, kparams->aligned_tile_size,
                                   matrix_rows, cur_a, mmctx->mapping_stride,
                                   m_start, m_end, (पूर्णाङ्क) octx->n_threads);
        यदि (ret != 0) {
            FARF(ERROR, "HMX matmul failed for expert %u, error %d\n", cur_a, ret);
            प्रत्यागम HTP_STATUS_NO_SUPPORT;
        }
    }

    प्रत्यागम HTP_STATUS_OK;
}

स्थिर पूर्णाङ्क hvx_mm_matmul_id(
    संरचना htp_ops_context * octx,
    संरचना htp_mm_context * mmctx,
    work_queue_func_t hvx_mmid_task_func
) {
    htp_matmul_tensors_preamble;
    ध्रुव uint32_t src0_row_size_padded = mmctx->src0_row_size_padded;
    ध्रुव uint32_t act_nrows            = mmctx->act_nrows;

    संरचना htp_thread_trace * tr = &octx->ctx->trace[0];
    htp_trace_event_start(tr, HTP_TRACE_EVT_INIT, 0);

    ध्रुव संरचना htp_mm_kernel_params * kparams = (ध्रुव संरचना htp_mm_kernel_params *) octx->kernel_params;
    ध्रुव संरचना htp_tensor * restrict ids = octx->src[2];
    ध्रुव माप_प्रकार src0_row_size = nb01;

    ध्रुव uint32_t qk = QK_Q8_0_TILED;
    ध्रुव uint32_t nb = (ne10 + qk - 1) / qk;
    ध्रुव uint32_t total_nb = act_nrows * nb;

    work_queue_func_t quant_task_func;
    uint32_t n_quant_tasks = 1;
    यदि (act_nrows < octx->n_threads) {
        n_quant_tasks = MIN(total_nb, octx->n_threads);
        quant_task_func = htp_mm_act_quant_block_func(src0->type, act->type);
        प्रति (uint32_t ith = 0; ith < n_quant_tasks; ++ith) {
            uint32_t ib_first = (total_nb * ith) / n_quant_tasks;
            uint32_t ib_last  = (total_nb * (ith + 1)) / n_quant_tasks;
            mmctx->quant_ib_first[ith] = ib_first;
            mmctx->quant_ib_last[ith]  = ib_last;
            mmctx->quant_r[ith]        = ib_first / nb;
            mmctx->quant_c[ith]        = ib_first % nb;
        }
    } अन्यथा {
        n_quant_tasks = MIN(act_nrows, octx->n_threads);
        quant_task_func = htp_mm_act_quant_row_func(src0->type, act->type);
    }
    माप_प्रकार act_row_size = htp_mm_tiled_act_row_size(src0->type, ne10);

    संरचना htp_mm_hvx_vtcm_layout L;
    htp_mm_hvx_vtcm_layout_build(&L, kparams->kernel_type, src0->type, ne10, act_nrows, octx->n_threads,
                                 0, src0_row_size, act_row_size, 0, kparams->n_prefetch, true, false);

    ध्रुव माप_प्रकार vtcm_size = L.total_bytes;

    FARF(ऊपरी_2, "matmul-id-%s : src0-spad-size %zu act-spad-size %zu bias-spad-size 0 dst-spad-size %zu (%zu)\n", mmctx->type,
         L.src0_bytes, L.act_bytes, L.dst_bytes, vtcm_size);

    FARF(ऊपरी_2, "matmul-id-%s : %ux%ux%ux%u * %ux%ux%ux%u (%ux%ux%ux%u) -> %ux%ux%ux%u (0x%p, 0x%p, 0x%p)\n", mmctx->type,
         src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3], act->ne[0], act->ne[1], act->ne[2], act->ne[3],
         ids->ne[0], ids->ne[1], ids->ne[2], ids->ne[3], dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3], src0->data,
         act->data, dst->data);

    // Make sure the reserved vtcm size is sufficient
    यदि (octx->ctx->vtcm_size < vtcm_size) {
        FARF(ERROR, "matmul-id-%s : current VTCM reservation %zu is too small, needed %zu\n", mmctx->type, octx->ctx->vtcm_size, vtcm_size);
        प्रत्यागम HTP_STATUS_VTCM_TOO_SMALL;
    }

    uint8_t * ध्रुव base = (uint8_t *) octx->ctx->vtcm_base;
    mmctx->vtcm_act      = VTCM_LAYOUT_PTR(uint8_t, base, L.off_act);
    mmctx->vtcm_src0     = VTCM_LAYOUT_PTR(uint8_t, base, L.off_src0);
    mmctx->vtcm_bias     = रिक्त;
    mmctx->vtcm_dst      = VTCM_LAYOUT_PTR(uint8_t, base, L.off_dst);
    mmctx->vtcm_act_raw  = VTCM_LAYOUT_PTR(uint8_t, base, L.off_act_raw);

    mmctx->vtcm_src0_stride    = src0_row_size_padded;
    mmctx->vtcm_act_stride     = act_row_size;
    mmctx->vtcm_act_raw_stride = hex_round_up(ne10 * परिमाण(भिन्न), QK_Q8_0_TILED * परिमाण(भिन्न));

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

    प्रत्यागम HTP_STATUS_OK;
}

स्थिर पूर्णाङ्क hmx_mm_op_matmul_id_nx(
    संरचना htp_ops_context * octx,
    संरचना htp_mm_context * mmctx
) {
    ध्रुव uint32_t *                matrix_row_counts = mmctx->matrix_row_counts;
    ध्रुव संरचना mmid_row_mapping * matrix_rows       = mmctx->matrix_rows;
    ध्रुव संरचना htp_mm_kernel_params * kparams = (ध्रुव संरचना htp_mm_kernel_params *) octx->kernel_params;
    ध्रुव uint32_t n_weights = kparams->n_weights;
    ध्रुव संरचना htp_tensor * restrict src0 = octx->src[0];
    ध्रुव संरचना htp_tensor * restrict act  = octx->src[n_weights];
    ध्रुव पूर्णाङ्क n_as = src0->ne[2];

    bool mdev_split = (octx->ctx->mdev.गिनती > 1);
    प्रति (uint32_t p = 0; p < n_weights && mdev_split; ++p) {
        ध्रुव संरचना htp_tensor * restrict dst = octx->dsts[p];
        mdev_split = htp_tensor_can_row_partition(dst, परिमाण(भिन्न));
    }
    यदि (octx->ctx->mdev.गिनती > 1 && !mdev_split && octx->ctx->mdev.idx > 0) {
        प्रत्यागम HTP_STATUS_OK;
    }
    uint32_t n_active = 0;
    यदि (mdev_split) {
        प्रति (uint32_t a = 0; a < (uint32_t) n_as; ++a) {
            यदि (matrix_row_counts[a] > 0) n_active++;
        }
    }
    ध्रुव bool expert_split = mdev_split && (n_active >= octx->ctx->mdev.गिनती);

    uint32_t target_dev = 0;
    प्रति (uint32_t cur_a = 0; cur_a < (uint32_t) n_as; ++cur_a) {
        ध्रुव int32_t cne1 = matrix_row_counts[cur_a];
        यदि (cne1 == 0) अनुवर्त;

        ध्रुव पूर्णाङ्क m_padded = hex_align_up(cne1, 32);
        पूर्णाङ्क m_start = 0, m_end = m_padded;
        यदि (expert_split) {
            ध्रुव bool my_expert = (target_dev == octx->ctx->mdev.idx);
            यदि (++target_dev == octx->ctx->mdev.गिनती) {
                target_dev = 0;
            }
            यदि (!my_expert) अनुवर्त;
        } अन्यथा यदि (mdev_split) {
            ध्रुव bool can_split = (uint32_t) cne1 >= octx->ctx->mdev.गिनती;
            ध्रुव संरचना htp_tensor_mdev_range range = htp_tensor_mdev_partition((uint32_t) m_padded, can_split ? 1 : 0, octx->ctx->mdev.idx, octx->ctx->mdev.गिनती, &octx->ctx->mdev.गिनती_div);
            m_start = (पूर्णाङ्क) range.start;
            m_end   = (पूर्णाङ्क) (range.start + range.गिनती);
        }
        यदि (m_start >= m_end) अनुवर्त;

        प्रति (uint32_t p = 0; p < n_weights; ++p) {
            ध्रुव संरचना htp_tensor * restrict src_w = octx->src[p];
            ध्रुव संरचना htp_tensor * restrict dst   = octx->dsts[p];
            यदि (!src_w || !dst) अनुवर्त;

            पूर्णाङ्क ret = hmx_mm_id_2d_f32(octx->ctx, octx->ctx->dma[0], (भिन्न*) dst->data, (भिन्न*) act->data,
                                       src_w->data + cur_a * src_w->nb[2],
                                       cne1, src_w->ne[0], src_w->ne[1],
                                       act->ne[0],
                                       act->ne[1],
                                       act->nb[1], act->nb[2],
                                       dst->nb[1], dst->nb[2],
                                       (पूर्णाङ्क) src_w->nb[1], (पूर्णाङ्क) src_w->type,
                                       kparams->tile_size, kparams->aligned_tile_size,
                                       matrix_rows, cur_a, mmctx->mapping_stride,
                                       m_start, m_end, (पूर्णाङ्क) octx->n_threads);
            यदि (ret != 0) {
                FARF(ERROR, "HMX matmul ID NX failed for expert %u weight %u, error %d\n", cur_a, p, ret);
                प्रत्यागम HTP_STATUS_NO_SUPPORT;
            }
        }
    }

    प्रत्यागम HTP_STATUS_OK;
}

स्थिर पूर्णाङ्क hvx_mm_matmul_id_nx(
    संरचना htp_ops_context * octx,
    संरचना htp_mm_context * mmctx,
    work_queue_func_t hvx_mmid_task_func
) {
    ध्रुव uint32_t src0_row_size_padded = mmctx->src0_row_size_padded;
    ध्रुव uint32_t act_nrows            = mmctx->act_nrows;

    संरचना htp_thread_trace * tr = &octx->ctx->trace[0];
    htp_trace_event_start(tr, HTP_TRACE_EVT_INIT, 0);

    ध्रुव संरचना htp_mm_kernel_params * kparams = (ध्रुव संरचना htp_mm_kernel_params *) octx->kernel_params;
    ध्रुव uint32_t n_weights = kparams->n_weights;
    ध्रुव संरचना htp_tensor * restrict src0 = octx->src[0];
    ध्रुव संरचना htp_tensor * restrict act  = octx->src[n_weights];
    ध्रुव संरचना htp_tensor * restrict ids  = octx->src[n_weights + 1];
    ध्रुव माप_प्रकार src0_row_size = src0->nb[1];

    ध्रुव uint32_t qk = QK_Q8_0_TILED;
    ध्रुव uint32_t nb = (act->ne[0] + qk - 1) / qk;
    ध्रुव uint32_t total_nb = act_nrows * nb;

    work_queue_func_t quant_task_func;
    uint32_t n_quant_tasks = 1;
    यदि (act_nrows < octx->n_threads) {
        n_quant_tasks = MIN(total_nb, octx->n_threads);
        quant_task_func = htp_mm_act_quant_block_func(src0->type, act->type);
        प्रति (uint32_t ith = 0; ith < n_quant_tasks; ++ith) {
            uint32_t ib_first = (total_nb * ith) / n_quant_tasks;
            uint32_t ib_last  = (total_nb * (ith + 1)) / n_quant_tasks;
            mmctx->quant_ib_first[ith] = ib_first;
            mmctx->quant_ib_last[ith]  = ib_last;
            mmctx->quant_r[ith]        = ib_first / nb;
            mmctx->quant_c[ith]        = ib_first % nb;
        }
    } अन्यथा {
        n_quant_tasks = MIN(act_nrows, octx->n_threads);
        quant_task_func = htp_mm_act_quant_row_func(src0->type, act->type);
    }
    माप_प्रकार act_row_size = htp_mm_tiled_act_row_size(src0->type, act->ne[0]);

    संरचना htp_mm_hvx_vtcm_layout L;
    htp_mm_hvx_vtcm_layout_build(&L, kparams->kernel_type, src0->type, act->ne[0], act_nrows, octx->n_threads,
                                 0, src0_row_size, act_row_size, 0, kparams->n_prefetch, true, false);

    ध्रुव माप_प्रकार vtcm_size = L.total_bytes;

    यदि (octx->ctx->vtcm_size < vtcm_size) {
        FARF(ERROR, "matmul-id-nx: current VTCM reservation %zu is too small, needed %zu\n",
             octx->ctx->vtcm_size, vtcm_size);
        प्रत्यागम HTP_STATUS_VTCM_TOO_SMALL;
    }

    uint8_t * ध्रुव base = (uint8_t *) octx->ctx->vtcm_base;
    mmctx->vtcm_src0     = VTCM_LAYOUT_PTR(uint8_t, base, L.off_src0);
    mmctx->vtcm_act      = VTCM_LAYOUT_PTR(uint8_t, base, L.off_act);
    mmctx->vtcm_dst      = VTCM_LAYOUT_PTR(uint8_t, base, L.off_dst);
    mmctx->vtcm_act_raw  = VTCM_LAYOUT_PTR(uint8_t, base, L.off_act_raw);

    mmctx->vtcm_src0_stride    = 0;
    mmctx->vtcm_act_stride     = act_row_size;
    mmctx->vtcm_act_raw_stride = hex_round_up(act->ne[0] * परिमाण(भिन्न), QK_Q8_0_TILED * परिमाण(भिन्न));

    mmctx->vtcm_src0_size_per_thread = fastdiv(L.src0_bytes, &octx->n_threads_div);
    mmctx->vtcm_act_size_per_thread  = L.act_bytes;
    mmctx->vtcm_dst_size_per_thread  = fastdiv(L.dst_bytes, &octx->n_threads_div);

    mmctx->cur_m_start = 0;
    mmctx->cur_m_rows  = act_nrows;

    FARF(ऊपरी_2, "matmul-id-nx: src0 %d:%d:%d type %s nrows %u, act %d:%d:%d nrows %u, vtcm %zu/%zu, threads %d\n",
         src0->ne[0], src0->ne[1], src0->ne[2], mmctx->type, src0->ne[1],
         act->ne[0], act->ne[1], act->ne[2], act_nrows,
         L.total_bytes, octx->ctx->vtcm_size, octx->n_threads);

    htp_trace_event_stop(tr, HTP_TRACE_EVT_INIT, 0);

    hvx_mm_transfer_act_dma(octx, kparams, act, mmctx->vtcm_act_raw, mmctx->vtcm_act_raw_stride, 0, act_nrows);

    mmctx->n_quant_rows_per_thread = (act_nrows + n_quant_tasks - 1) / n_quant_tasks;
    mmctx->n_quant_tasks           = n_quant_tasks;
    work_queue_run(octx->ctx->work_queue, quant_task_func, mmctx, n_quant_tasks);

    work_queue_run(octx->ctx->work_queue, hvx_mmid_task_func, mmctx, octx->n_threads);

    प्रत्यागम HTP_STATUS_OK;
}

स्थिर अन्तःस्थ शून्य scan_expert_ids_n(
    ध्रुव संरचना htp_tensor * ids,
    ध्रुव uint32_t n_ids,
    uint32_t n_as,
    uint32_t * counts,
    संरचना mmid_row_mapping * matrix_rows,
    uint32_t mapping_stride
) {
    ध्रुव माप_प्रकार ids_nb1 = ids->nb[1];
    ध्रुव uint8_t * ids_data = (ध्रुव uint8_t *) ids->data;

    प्रति (uint32_t iid1 = 0; iid1 < ids->ne[1]; ++iid1) {
        ध्रुव int32_t * row_ptr = (ध्रुव int32_t *) (ids_data + iid1 * ids_nb1);
        प्रति (uint32_t id = 0; id < n_ids; ++id) {
            ध्रुव int32_t i02 = row_ptr[id];
            यदि (i02 < 0) {
                अनुवर्त;
            }
            प्रतिज्ञा(i02 < n_as);

            यदि (matrix_rows) {
                matrix_rows[i02 * mapping_stride + counts[i02]] = (संरचना mmid_row_mapping) { id, iid1 };
            }
            counts[i02] += 1;
        }
    }
}

स्थिर अन्तःस्थ शून्य scan_expert_ids(
    ध्रुव संरचना htp_tensor * ids,
    uint32_t n_ids,
    uint32_t n_as,
    uint32_t * counts,
    संरचना mmid_row_mapping * matrix_rows,
    uint32_t mapping_stride
) {
    ध्रुव माप_प्रकार ids_nb0 = ids->nb[0];

    यदि (ids_nb0 == 4) {
        विकल्प (n_ids) {
            पक्ष 8:  scan_expert_ids_n(ids, 8,     n_as, counts, matrix_rows, mapping_stride); भङ्ग;
            पक्ष 4:  scan_expert_ids_n(ids, 4,     n_as, counts, matrix_rows, mapping_stride); भङ्ग;
            पक्ष 2:  scan_expert_ids_n(ids, 2,     n_as, counts, matrix_rows, mapping_stride); भङ्ग;
            स्वभाव: scan_expert_ids_n(ids, n_ids, n_as, counts, matrix_rows, mapping_stride); भङ्ग;
        }
    } अन्यथा {
        // Strided fallback
        ध्रुव माप_प्रकार ids_nb1 = ids->nb[1];
        ध्रुव uint8_t * ids_data = (ध्रुव uint8_t *) ids->data;
        प्रति (uint32_t iid1 = 0; iid1 < ids->ne[1]; ++iid1) {
            ध्रुव int32_t * row_ptr = (ध्रुव int32_t *) (ids_data + iid1 * ids_nb1);
            प्रति (uint32_t id = 0; id < n_ids; ++id) {
                ध्रुव int32_t i02 = *(ध्रुव int32_t *) ((ध्रुव uint8_t *) row_ptr + id * ids_nb0);
                यदि (i02 < 0) {
                    अनुवर्त;
                }
                प्रतिज्ञा(i02 < n_as);

                यदि (matrix_rows) {
                    matrix_rows[i02 * mapping_stride + counts[i02]] = (संरचना mmid_row_mapping) { id, iid1 };
                }
                counts[i02] += 1;
            }
        }
    }
}

पूर्णाङ्क op_matmul_id(संरचना htp_ops_context * octx) {
    htp_matmul_tensors_preamble;

    ध्रुव संरचना htp_mm_kernel_params * kparams = (ध्रुव संरचना htp_mm_kernel_params *) octx->kernel_params;
    संरचना htp_mm_context mmctx_struct = {0};
    संरचना htp_mm_context * mmctx = &mmctx_struct;

    ध्रुव पूर्णाङ्क status = htp_mm_init_context(octx, kparams);
    यदि (status != HTP_STATUS_OK) {
        प्रत्यागम status;
    }

    संरचना htp_thread_trace * tr = &octx->ctx->trace[0];
    htp_trace_event_start(tr, HTP_TRACE_EVT_INIT, 0);

    mmctx->octx = octx;
    mmctx->act = act;

    ध्रुव संरचना htp_tensor * restrict ids = octx->src[2];
    यदि (htp_tensor_is_extended(ids) || htp_tensor_is_extended(act) || htp_tensor_is_extended(dst)) {
        प्रत्यागम HTP_STATUS_NO_SUPPORT;
    }

    ध्रुव माप_प्रकार src0_row_size = nb01;
    ध्रुव माप_प्रकार dst_row_size  = nb1;

    ध्रुव माप_प्रकार src0_row_size_padded = hex_round_up(src0_row_size, 128);

    ध्रुव uint32_t src0_nrows = ne01;  // per expert
    ध्रुव uint32_t act_nrows  = ne11 * ne12 * ne13;

    // row groups
    ध्रुव पूर्णाङ्क n_ids = ids->ne[0];  // n_expert_used
    ध्रुव पूर्णाङ्क n_as  = ne02;        // n_expert

    uint8_t  * mapping_buf       = octx->ctx->ddr_spad_base;
    uint32_t   mapping_stride    = 1;
    uint32_t * matrix_row_counts = (uint32_t *) mapping_buf;
    संरचना mmid_row_mapping * matrix_rows = रिक्त;

    यदि (act_nrows > 1) {
        ध्रुव माप_प्रकार matrix_row_counts_size = n_as * परिमाण(uint32_t);
        प्रतिज्ञा(octx->ctx->ddr_spad_size >= matrix_row_counts_size);

        hex_l2fetch_block((ध्रुव शून्य *) ids->data, ids->ne[1] * ids->nb[1]);

        स_रखो(matrix_row_counts, 0, matrix_row_counts_size);
        scan_expert_ids(ids, n_ids, n_as, matrix_row_counts, रिक्त, 0);

        uint32_t max_गिनती = hvx_reduce_max_i32((ध्रुव uint8_t *) matrix_row_counts, n_as);
        mapping_stride = max_गिनती > 0 ? max_गिनती : 1;

        माप_प्रकार matrix_row_map_size  = n_as * mapping_stride * परिमाण(संरचना mmid_row_mapping);
        ध्रुव माप_प्रकार total_map_size = matrix_row_counts_size + matrix_row_map_size;

        यदि (total_map_size > octx->ctx->ddr_spad_size) {
            mapping_buf = memalign(128, total_map_size);
            यदि (!mapping_buf) {
                प्रत्यागम HTP_STATUS_INTERNAL_ERR;
            }
        }

        matrix_row_counts = (uint32_t *) mapping_buf;
        matrix_rows       = (संरचना mmid_row_mapping *) (mapping_buf + matrix_row_counts_size);

        स_रखो(matrix_row_counts, 0, n_as * परिमाण(uint32_t));
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

    पूर्णाङ्क s;
    यदि (kparams->n_hmx) {
        s = hmx_mm_op_matmul_id(octx, mmctx);
    } अन्यथा {
        uint32_t src0_row_start = 0;
        uint32_t src0_row_end   = src0_nrows;
        यदि (octx->ctx->mdev.गिनती > 1) {
            ध्रुव bool can_split = htp_tensor_can_row_partition(dst, परिमाण(भिन्न));
            ध्रुव संरचना htp_tensor_mdev_range range = htp_tensor_mdev_partition(src0_nrows, can_split ? 32 : 0, octx->ctx->mdev.idx, octx->ctx->mdev.गिनती, &octx->ctx->mdev.गिनती_div);
            src0_row_start = range.start;
            src0_row_end   = range.start + range.गिनती;
        }

        यदि (src0_row_start >= src0_row_end) {
            यदि (mapping_buf != octx->ctx->ddr_spad_base) {
                मुक्त(mapping_buf);
            }
            प्रत्यागम HTP_STATUS_OK;
        }

        ध्रुव uint32_t nrows = src0_row_end - src0_row_start;
        mmctx->src0_row_start = src0_row_start;
        mmctx->src0_row_end   = src0_row_end;

        mmctx->src0_nrows_per_thread = fastdiv(nrows + octx->n_threads - 1, &octx->n_threads_div);
        mmctx->src0_nrows_per_thread = hex_round_up(mmctx->src0_nrows_per_thread, 32);

        यदि (hvx_mm_init_tiled_vec_dot(mmctx, src0->type) == 0) {
            s = hvx_mm_matmul_id(octx, mmctx, act_nrows > 1 ? hvx_mm_id_tiled : hvx_mv_id_tiled);
        } अन्यथा {
            s = HTP_STATUS_NO_SUPPORT;
        }
    }

    यदि (mapping_buf != octx->ctx->ddr_spad_base) {
        मुक्त(mapping_buf);
    }

    प्रत्यागम s;
}

पूर्णाङ्क op_matmul_id_nx(संरचना htp_ops_context * octx) {
    ध्रुव संरचना htp_mm_kernel_params * kparams = (ध्रुव संरचना htp_mm_kernel_params *) octx->kernel_params;
    संरचना htp_mm_context mmctx_struct = {0};
    संरचना htp_mm_context * mmctx = &mmctx_struct;

    ध्रुव पूर्णाङ्क status = htp_mm_init_context(octx, kparams);
    यदि (status != HTP_STATUS_OK) {
        प्रत्यागम status;
    }

    संरचना htp_thread_trace * tr = &octx->ctx->trace[0];
    htp_trace_event_start(tr, HTP_TRACE_EVT_INIT, 0);

    mmctx->octx = octx;
    ध्रुव uint32_t n_weights = kparams->n_weights;
    ध्रुव संरचना htp_tensor * restrict src0 = octx->src[0];
    ध्रुव संरचना htp_tensor * restrict act  = octx->src[n_weights];
    ध्रुव संरचना htp_tensor * restrict ids  = octx->src[n_weights + 1];
    यदि (htp_tensor_is_extended(ids) || htp_tensor_is_extended(act)) {
        प्रत्यागम HTP_STATUS_NO_SUPPORT;
    }
    प्रति (uint32_t p = 0; p < n_weights; p++) {
        यदि (htp_tensor_is_extended(octx->dsts[p])) {
            प्रत्यागम HTP_STATUS_NO_SUPPORT;
        }
    }

    mmctx->act = act;

    ध्रुव माप_प्रकार src0_row_size = src0->nb[1];
    ध्रुव माप_प्रकार src0_row_size_padded = hex_round_up(src0_row_size, 128);

    ध्रुव uint32_t act_nrows = act->ne[1] * act->ne[2] * act->ne[3];

    ध्रुव पूर्णाङ्क n_ids = ids->ne[0];
    ध्रुव पूर्णाङ्क n_as  = src0->ne[2];

    uint8_t  * mapping_buf       = octx->ctx->ddr_spad_base;
    uint32_t   mapping_stride    = 1;
    uint32_t * matrix_row_counts = (uint32_t *) mapping_buf;
    संरचना mmid_row_mapping * matrix_rows = रिक्त;

    यदि (act_nrows > 1) {
        ध्रुव माप_प्रकार matrix_row_counts_size = n_as * परिमाण(uint32_t);
        प्रतिज्ञा(octx->ctx->ddr_spad_size >= matrix_row_counts_size);

        hex_l2fetch_block((ध्रुव शून्य *) ids->data, ids->ne[1] * ids->nb[1]);

        स_रखो(matrix_row_counts, 0, matrix_row_counts_size);
        scan_expert_ids(ids, n_ids, n_as, matrix_row_counts, रिक्त, 0);

        uint32_t max_गिनती = hvx_reduce_max_i32((ध्रुव uint8_t *) matrix_row_counts, n_as);
        mapping_stride = max_गिनती > 0 ? max_गिनती : 1;

        माप_प्रकार matrix_row_map_size  = n_as * mapping_stride * परिमाण(संरचना mmid_row_mapping);
        ध्रुव माप_प्रकार total_map_size = matrix_row_counts_size + matrix_row_map_size;

        यदि (total_map_size > octx->ctx->ddr_spad_size) {
            mapping_buf = memalign(128, total_map_size);
            यदि (!mapping_buf) {
                प्रत्यागम HTP_STATUS_INTERNAL_ERR;
            }
        }

        matrix_row_counts = (uint32_t *) mapping_buf;
        matrix_rows       = (संरचना mmid_row_mapping *) (mapping_buf + matrix_row_counts_size);

        स_रखो(matrix_row_counts, 0, n_as * परिमाण(uint32_t));
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

    पूर्णाङ्क s;
    यदि (kparams->n_hmx) {
        s = hmx_mm_op_matmul_id_nx(octx, mmctx);
    } अन्यथा {
        यदि (hvx_mm_init_tiled_vec_dot(mmctx, src0->type) == 0) {
            s = hvx_mm_matmul_id_nx(octx, mmctx, act_nrows > 1 ? hvx_mm_id_nx_tiled : hvx_mv_id_nx_tiled);
        } अन्यथा {
            s = HTP_STATUS_NO_SUPPORT;
        }
    }

    यदि (mapping_buf != octx->ctx->ddr_spad_base) {
        मुक्त(mapping_buf);
    }

    प्रत्यागम s;
}
स्थिर पूर्णाङ्क op_matmul_nx_impl(संरचना htp_ops_context * octx) {
    ध्रुव संरचना htp_mm_kernel_params * kparams = (ध्रुव संरचना htp_mm_kernel_params *) octx->kernel_params;

    ध्रुव पूर्णाङ्क status = htp_mm_init_context(octx, kparams);
    यदि (status != HTP_STATUS_OK) {
        प्रत्यागम status;
    }

    यदि (kparams->n_hmx) {
        प्रत्यागम hmx_mm_nx_2d_f32(octx, kparams);
    }

    संरचना htp_thread_trace * tr = &octx->ctx->trace[0];
    htp_trace_event_start(tr, HTP_TRACE_EVT_INIT, 0);

    ध्रुव uint32_t n_weights = kparams->n_weights;

    ध्रुव संरचना htp_tensor * restrict src0 = octx->src[0]; // first weight
    ध्रुव संरचना htp_tensor * restrict act  = octx->src[n_weights]; // activation x

    संरचना htp_mm_context mmctx_struct = {0};
    संरचना htp_mm_context * mmctx = &mmctx_struct;
    mmctx->octx = octx;
    mmctx->act  = act;

    ध्रुव uint32_t act_nrows = act->ne[1] * act->ne[2] * act->ne[3];
    mmctx->act_nrows   = act_nrows;
    mmctx->cur_m_start = 0;
    mmctx->cur_m_rows  = act_nrows;

    ध्रुव माप_प्रकार src0_row_size = src0->nb[1];

    यदि (hvx_mm_init_tiled_vec_dot(mmctx, src0->type) != 0) {
        प्रत्यागम HTP_STATUS_NO_SUPPORT;
    }

    ध्रुव uint32_t qk = QK_Q8_0_TILED;
    ध्रुव uint32_t nb = (act->ne[0] + qk - 1) / qk;
    ध्रुव uint32_t total_nb = act_nrows * nb;

    worker_callback_t quant_task_func;
    uint32_t n_quant_tasks = 1;
    यदि (act_nrows < octx->n_threads) {
        n_quant_tasks = MIN(total_nb, octx->n_threads);
        quant_task_func = htp_mm_act_quant_block_func(src0->type, act->type);
        प्रति (uint32_t ith = 0; ith < n_quant_tasks; ++ith) {
            uint32_t ib_first = (total_nb * ith) / n_quant_tasks;
            uint32_t ib_last  = (total_nb * (ith + 1)) / n_quant_tasks;
            mmctx->quant_ib_first[ith] = ib_first;
            mmctx->quant_ib_last[ith]  = ib_last;
            mmctx->quant_r[ith]        = ib_first / nb;
            mmctx->quant_c[ith]        = ib_first % nb;
        }
    } अन्यथा {
        n_quant_tasks = MIN(act_nrows, octx->n_threads);
        quant_task_func = htp_mm_act_quant_row_func(src0->type, act->type);
    }

    ध्रुव माप_प्रकार act_row_size = htp_mm_tiled_act_row_size(src0->type, act->ne[0]);

    संरचना htp_mm_hvx_vtcm_layout L;
    htp_mm_hvx_vtcm_layout_build(&L, kparams->kernel_type, src0->type, act->ne[0], act_nrows, octx->n_threads,
                                 0, src0_row_size, act_row_size, 0, kparams->n_prefetch, false, true);

    ध्रुव माप_प्रकार vtcm_size = L.total_bytes;

    यदि (octx->ctx->vtcm_size < vtcm_size) {
        FARF(ERROR, "matmul-nx: current VTCM reservation %zu is too small, needed %zu\n",
             octx->ctx->vtcm_size, vtcm_size);
        प्रत्यागम HTP_STATUS_VTCM_TOO_SMALL;
    }

    uint8_t * ध्रुव base = (uint8_t *) octx->ctx->vtcm_base;
    mmctx->vtcm_src0     = VTCM_LAYOUT_PTR(uint8_t, base, L.off_src0);
    mmctx->vtcm_act      = VTCM_LAYOUT_PTR(uint8_t, base, L.off_act);
    mmctx->vtcm_dst      = VTCM_LAYOUT_PTR(uint8_t, base, L.off_dst);
    mmctx->vtcm_act_raw  = VTCM_LAYOUT_PTR(uint8_t, base, L.off_act_raw);

    mmctx->vtcm_src0_stride    = 0;
    mmctx->vtcm_act_stride     = act_row_size;
    mmctx->vtcm_act_raw_stride = hex_round_up(act->ne[0] * परिमाण(भिन्न), QK_Q8_0_TILED * परिमाण(भिन्न));

    mmctx->vtcm_src0_size_per_thread = fastdiv(L.src0_bytes, &octx->n_threads_div);
    mmctx->vtcm_act_size_per_thread  = L.act_bytes;
    mmctx->vtcm_dst_size_per_thread  = fastdiv(L.dst_bytes, &octx->n_threads_div);

    htp_trace_event_stop(tr, HTP_TRACE_EVT_INIT, 0);

    hvx_mm_transfer_act_dma(octx, kparams, act, mmctx->vtcm_act_raw, mmctx->vtcm_act_raw_stride, 0, act_nrows);

    mmctx->n_quant_rows_per_thread = (act_nrows + n_quant_tasks - 1) / n_quant_tasks;
    mmctx->n_quant_tasks = n_quant_tasks;
    work_queue_run(octx->ctx->work_queue, quant_task_func, mmctx, n_quant_tasks);

    work_queue_run(octx->ctx->work_queue, hvx_mm_nx_2d_tiled, mmctx, octx->n_threads);

    प्रत्यागम HTP_STATUS_OK;
}

पूर्णाङ्क op_matmul_nx(संरचना htp_ops_context * octx) {
    ध्रुव संरचना htp_mm_kernel_params * kparams = (ध्रुव संरचना htp_mm_kernel_params *) octx->kernel_params;

    यदि (kparams->collapse) {
        ध्रुव uint32_t n_weights = kparams->n_weights;
        ध्रुव संरचना htp_tensor * act = octx->src[n_weights];
        ध्रुव uint32_t s1 = (act->ne[1] > 1) ? act->nb[1] : ((act->ne[2] > 1) ? act->nb[2] : act->nb[3]);
        संरचना htp_tensor act_collapsed;
        संरचना htp_tensor dsts_collapsed[HTP_OP_MAX_OUTPUTS];
        ध्रुव संरचना htp_tensor * orig_dsts[HTP_OP_MAX_OUTPUTS];

        htp_mm_tensor_collapse_rows(&act_collapsed, act, s1);
        octx->src[n_weights] = &act_collapsed;

        प्रति (uint32_t p = 0; p < n_weights; p++) {
            orig_dsts[p] = octx->dsts[p];
            ध्रुव संरचना htp_tensor * d = octx->dsts[p];
            ध्रुव uint32_t sd = (d->ne[1] > 1) ? d->nb[1] : ((d->ne[2] > 1) ? d->nb[2] : d->nb[3]);
            htp_mm_tensor_collapse_rows(&dsts_collapsed[p], d, sd);
            octx->dsts[p] = &dsts_collapsed[p];
        }

        ध्रुव पूर्णाङ्क status = op_matmul_nx_impl(octx);

        octx->src[n_weights] = act;
        प्रति (uint32_t p = 0; p < n_weights; p++) {
            octx->dsts[p] = orig_dsts[p];
        }
        प्रत्यागम status;
    }

    प्रत्यागम op_matmul_nx_impl(octx);
}
