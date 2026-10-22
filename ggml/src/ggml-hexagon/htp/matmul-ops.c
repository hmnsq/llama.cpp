#आशय clang diagnostic ignored "-Wgnu-zero-variadic-macro-arguments"
#आशय clang diagnostic ignored "-Wunused-function"
#आशय clang diagnostic ignored "-Wunused-variable"
#आशय clang diagnostic ignored "-Wunused-but-set-variable"

#समावेश <HAP_farf.h>
#समावेश <HAP_perf.h>
#समावेश <HAP_compute_res.h>

#समावेश <math.h>
#समावेश <string.h>
#समावेश <stdatomic.h>

#समावेश "dma-queue.h"
#समावेश "hvx-utils.h"
#समावेश "hvx-dump.h"
#समावेश "hvx-arith.h"
#समावेश "hvx-reduce.h"

#घोषणा GGML_COMMON_DECL_C
#समावेश "ggml-common.h"
#समावेश "htp-ctx.h"
#समावेश "htp-ops.h"
#समावेश "htp-tensor.h"
#समावेश "matmul-ops.h"
#समावेश "htp-vtcm.h"

प्रकार काष्ठा {
    भग्न        *dst;
    dma_addr_t    bias_addr;
    माप_प्रकार        bias_bytes;
    dma_addr_t    act_dma_addr;
    dma_addr_t    weight;
    dma_queue *   weight_dma;
    पूर्णांक           m;
    पूर्णांक           k;
    पूर्णांक           n;
    पूर्णांक           act_stride;
    uint32_t      act_elem_आकार;  // 4=F32 src1, 2=F16 src1
    पूर्णांक           weight_stride;
    पूर्णांक           weight_type;
    पूर्णांक           dst_stride;
    पूर्णांक           dst_cols;
    uint32_t      bias_stride;
    पूर्णांक           tile_आकार;
    पूर्णांक           aligned_tile_आकार;
    पूर्णांक           ne02;
    पूर्णांक           ne03;
    पूर्णांक           ne12;
    पूर्णांक           ne13;
    माप_प्रकार        src0_nb2;
    माप_प्रकार        src0_nb3;
    माप_प्रकार        act_nb2;
    माप_प्रकार        act_nb3;
    माप_प्रकार        bias_nb2;
    माप_प्रकार        bias_nb3;
    माप_प्रकार        dst_nb2;
    माप_प्रकार        dst_nb3;
    पूर्णांक           r2;
    पूर्णांक           r3;
    काष्ठा fastdiv_मूल्यावली div_r2;
    काष्ठा fastdiv_मूल्यावली div_r3;
} hmx_mm_batched_params_t;

अटल bool htp_matmul_has_extended_weight(स्थिर काष्ठा htp_ops_context * octx, uint32_t n_weights) {
    क्रम (uint32_t i = 0; i < n_weights; ++i) {
        अगर (htp_tensor_is_extended(octx->src[i])) {
            वापस true;
        }
    }
    वापस false;
}

काष्ठा htp_mm_context {
    स्थिर अक्षर * type;
    काष्ठा htp_ops_context * octx;
    स्थिर काष्ठा htp_tensor * act;

    व्योम (*vec_dot_1x1)(स्थिर uint32_t n, भग्न * restrict s0,
         स्थिर व्योम * restrict vx0,
         स्थिर व्योम * restrict vy0);

    व्योम (*vec_dot_2x1)(स्थिर uint32_t n, भग्न * restrict s0,
         स्थिर व्योम * restrict vx0, स्थिर व्योम * restrict vx1,
         स्थिर व्योम * restrict vy0);

    व्योम (*vec_dot_2x2)(स्थिर uint32_t n, भग्न * restrict s0, भग्न * restrict s1,
         स्थिर व्योम * restrict vx0, स्थिर व्योम * restrict vx1,
         स्थिर व्योम * restrict vy0, स्थिर व्योम * restrict vy1);

    व्योम (*vec_dot_32x1)(स्थिर uint32_t n, भग्न * restrict s,
         स्थिर व्योम * restrict vx,
         स्थिर व्योम * restrict vy, uint32_t valid_rows,
         स्थिर भग्न * restrict sz);

    व्योम (*vec_dot_32x2)(स्थिर uint32_t n, भग्न * restrict s0, भग्न * restrict s1,
         स्थिर व्योम * restrict vx,
         स्थिर व्योम * restrict vy0, स्थिर व्योम * restrict vy1, uint32_t valid_rows,
         स्थिर भग्न * restrict sz0, स्थिर भग्न * restrict sz1);

    // Precomputed values
    uint32_t src0_nrows_per_thread;
    uint32_t src0_row_start;
    uint32_t src0_row_end;
    uint32_t src0_row_आकार_padded;
    uint32_t act_nrows;
    uint32_t cur_m_start;
    uint32_t cur_m_rows;

    काष्ठा fastdiv_मूल्यावली mm_div_ne12_ne1;
    काष्ठा fastdiv_मूल्यावली mm_div_ne1;
    काष्ठा fastdiv_मूल्यावली mm_div_r2;
    काष्ठा fastdiv_मूल्यावली mm_div_r3;
    काष्ठा fastdiv_मूल्यावली mm_div_ne11;

    // Precomputed block-parallel quantization values
    uint32_t          quant_ib_first[WORK_QUEUE_MAX_N_THREADS];
    uint32_t          quant_ib_last[WORK_QUEUE_MAX_N_THREADS];
    uint32_t          quant_r[WORK_QUEUE_MAX_N_THREADS];
    uint32_t          quant_c[WORK_QUEUE_MAX_N_THREADS];
    uint32_t          n_quant_tasks;
    uint32_t          n_quant_rows_per_thread;

    // Fields for scattered mapping & HMX support in MUL_MAT_ID
    स्थिर uint32_t * matrix_row_counts;
    स्थिर काष्ठा mmid_row_mapping * matrix_rows;
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
    uint32_t vtcm_src0_आकार_per_thread;
    uint32_t vtcm_act_आकार_per_thread;
    uint32_t vtcm_dst_आकार_per_thread;
};

अटल पूर्णांक htp_mm_init_context(
    काष्ठा htp_ops_context * octx,
    स्थिर काष्ठा htp_mm_kernel_params * kparams
) {
    अगर (!htp_ops_context_set_n_threads(octx, (uint32_t) kparams->n_threads)) {
        वापस HTP_STATUS_INVAL_PARAMS;
    }

    अगर (kparams->n_hmx) {
        अगर (kparams->n_act_threads <= 0 || kparams->n_act_threads > (int32_t) octx->n_threads) {
            वापस HTP_STATUS_INVAL_PARAMS;
        }
    }

    वापस HTP_STATUS_OK;
}

// vdelta control to expand first 32 e8m0 values into 32 uint32 elements
अटल स्थिर uint8_t __attribute__((aligned(128))) expand_x32_e8m0[128] = {
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
अटल स्थिर uint8_t __attribute__((aligned(VLEN))) kvalues_iq4nl_lut[] = {
    0x81, 0, 0x98, 0, 0xAD, 0, 0xBF, 0, 0xCF, 0, 0xDD, 0, 0xEA, 0, 0xF6, 0, 0x01, 0, 0x0D, 0, 0x19, 0, 0x26, 0,
    0x35, 0, 0x45, 0, 0x59, 0, 0x71, 0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0,
    0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0,
    0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0,
    0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0,
};

अटल स्थिर uint8_t __attribute__((aligned(VLEN))) kvalues_mxfp4_lut[] = {
    0,    0, 1,    0, 2,    0, 3, 0, 4, 0, 6, 0, 8, 0, 12, 0, 0, 0, 0xff, 0, 0xfe, 0, 0xfd, 0, 0xfc, 0,
    0xfa, 0, 0xf8, 0, 0xf4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 0,    0, 0,    0, 0,    0, 0,    0,
    0,    0, 0,    0, 0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 0,    0, 0,    0, 0,    0, 0,    0,
    0,    0, 0,    0, 0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 0,    0, 0,    0, 0,    0, 0,    0,
    0,    0, 0,    0, 0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 0,    0, 0,    0, 0,    0,
};

#घोषणा htp_matmul_tensors_preamble                                 \
    स्थिर काष्ठा htp_tensor * restrict src0 = octx->src[0];         \
    स्थिर काष्ठा htp_tensor * restrict src1 = octx->src[1];         \
    स्थिर काष्ठा htp_tensor * restrict src2 = octx->src[2];         \
    स्थिर काष्ठा htp_tensor * restrict  dst = octx->dst;            \
    स्थिर काष्ठा htp_tensor * restrict  act = src1;                 \
                                                                    \
    स्थिर uint32_t ne00 = src0->ne[0];                              \
    स्थिर uint32_t ne01 = src0->ne[1];                              \
    स्थिर uint32_t ne02 = src0->ne[2];                              \
    स्थिर uint32_t ne03 = src0->ne[3];                              \
                                                                    \
    स्थिर uint32_t ne10 = src1->ne[0];                              \
    स्थिर uint32_t ne11 = src1->ne[1];                              \
    स्थिर uint32_t ne12 = src1->ne[2];                              \
    स्थिर uint32_t ne13 = src1->ne[3];                              \
                                                                    \
    स्थिर uint32_t ne20 = src2 ? src2->ne[0] : 0;                   \
    स्थिर uint32_t ne21 = src2 ? src2->ne[1] : 0;                   \
    स्थिर uint32_t ne22 = src2 ? src2->ne[2] : 0;                   \
    स्थिर uint32_t ne23 = src2 ? src2->ne[3] : 0;                   \
                                                                    \
    स्थिर uint32_t ne0 = dst->ne[0];                                \
    स्थिर uint32_t ne1 = dst->ne[1];                                \
    स्थिर uint32_t ne2 = dst->ne[2];                                \
    स्थिर uint32_t ne3 = dst->ne[3];                                \
                                                                    \
    स्थिर uint32_t nb00 = src0->nb[0];                              \
    स्थिर uint32_t nb01 = src0->nb[1];                              \
    स्थिर uint32_t nb02 = src0->nb[2];                              \
    स्थिर uint32_t nb03 = src0->nb[3];                              \
                                                                    \
    स्थिर uint32_t nb10 = src1->nb[0];                              \
    स्थिर uint32_t nb11 = src1->nb[1];                              \
    स्थिर uint32_t nb12 = src1->nb[2];                              \
    स्थिर uint32_t nb13 = src1->nb[3];                              \
                                                                    \
    स्थिर uint32_t nb0 = dst->nb[0];                                \
    स्थिर uint32_t nb1 = dst->nb[1];                                \
    स्थिर uint32_t nb2 = dst->nb[2];                                \
    स्थिर uint32_t nb3 = dst->nb[3];

#घोषणा htp_matmul_preamble                                         \
    काष्ठा htp_mm_context * mmctx  = तथ्य;                          \
    काष्ठा htp_ops_context * octx  = mmctx->octx;                   \
    dma_queue * dma_q              = octx->ctx->dma[ith];           \
    uint32_t src0_nrows_per_thread = mmctx->src0_nrows_per_thread;  \
    htp_matmul_tensors_preamble;




// hvx kernels first: the HMX Q6_K dequantizer reuses unpack_q6_k_group from there
#समावेश "hvx-mm-kernels-tiled.h"
#समावेश "hvx-mm-kernels-float.h"
#समावेश "hmx-mm-kernels-tiled.h"

अटल व्योम hvx_mm_2d_tiled(अचिन्हित पूर्णांक nth, अचिन्हित पूर्णांक ith, व्योम * तथ्य) {
    htp_matmul_preamble;

    स्थिर uint32_t src0_nrows = mmctx->src0_row_end - mmctx->src0_row_start;
    स्थिर uint32_t act_nrows = mmctx->cur_m_rows ? mmctx->cur_m_rows : (ne11 * ne12 * ne13);
    स्थिर uint32_t cur_m_start = mmctx->cur_m_start;

    स्थिर uint32_t src0_start_row  = mmctx->src0_row_start + src0_nrows_per_thread * ith;
    स्थिर uint32_t src0_end_row    = MIN(src0_start_row + src0_nrows_per_thread, mmctx->src0_row_end);

    काष्ठा htp_thread_trace * tr = &octx->ctx->trace[ith];

    स्थिर काष्ठा htp_mm_kernel_params * kparams = (स्थिर काष्ठा htp_mm_kernel_params *) octx->kernel_params;
    स्थिर uint32_t n_prefetch = kparams->n_prefetch;
    निश्चित(n_prefetch >= 2 && n_prefetch <= HTP_MM_MAX_PREFETCH && (n_prefetch & (n_prefetch - 1)) == 0);

    स्थिर माप_प्रकार dst_row_आकार  = nb1;
    स्थिर माप_प्रकार act_row_आकार = nb11;
    स्थिर माप_प्रकार act_stride = mmctx->vtcm_act_stride;
    स्थिर माप_प्रकार src2_stride = src2 ? ((src2->ne[1] == 1) ? 0 : src2->nb[1]) : 0;

    uint8_t * restrict vtcm_dst_ptr  = mmctx->vtcm_dst  + mmctx->vtcm_dst_आकार_per_thread  * ith;
    uint8_t * restrict vtcm_src0_ptr = mmctx->vtcm_src0 + mmctx->vtcm_src0_आकार_per_thread * ith;
    uint8_t * restrict act_तथ्य = mmctx->vtcm_act;

    स्थिर dma_addr_t src0_row = src0->तथ्य;

    स्थिर uint32_t tile_आकार = htp_tensor_get_tile_आकार(src0->type);
    स्थिर uint32_t aligned_tile_आकार = htp_tensor_get_tile_padded_आकार(src0->type);

    uint32_t n_k_tiles_w = ne00 / htp_tensor_get_tile_k(src0->type);
    uint32_t tile_row_stride = n_k_tiles_w * tile_आकार;
    uint32_t tile_row_transfer_आकार_aligned = n_k_tiles_w * aligned_tile_आकार;

    uint32_t ct_start = src0_start_row / 32;
    uint32_t ct_end   = (src0_end_row + 31) / 32;

    uint32_t धकेलो_ct = ct_start;
    अगर (src0_start_row < src0_end_row) {
        क्रम (uint32_t d = 0; d < n_prefetch && धकेलो_ct < ct_end; d++, धकेलो_ct++) {
            dma_queue_धकेलो(dma_q, dma_make_तथ्य(vtcm_src0_ptr + d * tile_row_transfer_आकार_aligned,
                           src0_row + धकेलो_ct * tile_row_stride), aligned_tile_आकार, tile_आकार, tile_आकार, n_k_tiles_w);
        }
    }

    अगर (src0_start_row >= src0_end_row) {
        वापस;
    }

    क्रम (uint32_t ct = ct_start; ct < ct_end; ct++) {
        स्थिर uint8_t * w_tile = (व्योम *) dma_queue_निकालो(dma_q).dst;

        पूर्णांक valid_rows = (पूर्णांक)ne0 - (पूर्णांक)(ct * 32);
        valid_rows = MIN(32, MAX(0, valid_rows));

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, ct);
        uint32_t ir1 = 0;
        क्रम (; ir1 + 1 < act_nrows; ir1 += 2) {
            स्थिर uint8_t * restrict act_col0 = (स्थिर uint8_t *) (act_तथ्य + (ir1+0) * act_stride);
            स्थिर uint8_t * restrict act_col1 = (स्थिर uint8_t *) (act_तथ्य + (ir1+1) * act_stride);
            भग्न * restrict dst_row0 = (भग्न *) (dst->तथ्य + ((cur_m_start + ir1+0) * dst_row_आकार));
            भग्न * restrict dst_row1 = (भग्न *) (dst->तथ्य + ((cur_m_start + ir1+1) * dst_row_आकार));

            भग्न * dst_ptr0 = &dst_row0[ct * 32];
            भग्न * dst_ptr1 = &dst_row1[ct * 32];

            स्थिर भग्न * src2_ptr0 = शून्य;
            स्थिर भग्न * src2_ptr1 = शून्य;
            अगर (src2) {
                स्थिर भग्न * restrict src2_row0 = (स्थिर भग्न *) ((स्थिर uint8_t *) src2->तथ्य + ((cur_m_start + ir1+0) * src2_stride));
                स्थिर भग्न * restrict src2_row1 = (स्थिर भग्न *) ((स्थिर uint8_t *) src2->तथ्य + ((cur_m_start + ir1+1) * src2_stride));
                src2_ptr0 = &src2_row0[ct * 32];
                src2_ptr1 = &src2_row1[ct * 32];
            }
            mmctx->vec_dot_32x2(ne10, dst_ptr0, dst_ptr1, w_tile, act_col0, act_col1, valid_rows, src2_ptr0, src2_ptr1);
        }

        क्रम (; ir1 < act_nrows; ++ir1) {
            स्थिर uint8_t * restrict act_col = (स्थिर uint8_t *) (act_तथ्य + ir1 * act_stride);
            भग्न * restrict dst_row          = (भग्न *) (dst->तथ्य + ((cur_m_start + ir1) * dst_row_आकार));
            भग्न * dst_ptr = &dst_row[ct * 32];

            स्थिर भग्न * src2_ptr = शून्य;
            अगर (src2) {
                स्थिर भग्न * restrict src2_row = (स्थिर भग्न *) ((स्थिर uint8_t *) src2->तथ्य + ((cur_m_start + ir1) * src2_stride));
                src2_ptr = &src2_row[ct * 32];
            }
            mmctx->vec_dot_32x1(ne10, dst_ptr, w_tile, act_col, valid_rows, src2_ptr);
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ct);

        अगर (धकेलो_ct < ct_end) {
            dma_queue_धकेलो(dma_q, dma_make_तथ्य(w_tile, src0_row + धकेलो_ct * tile_row_stride),
                           aligned_tile_आकार, tile_आकार, tile_आकार, n_k_tiles_w);
            धकेलो_ct++;
        }
    }
}

अटल व्योम hvx_mv_2d_tiled(अचिन्हित पूर्णांक nth, अचिन्हित पूर्णांक ith, व्योम * तथ्य) {
    htp_matmul_preamble;

    स्थिर uint32_t src0_nrows = mmctx->src0_row_end - mmctx->src0_row_start;

    स्थिर uint32_t src0_start_row  = mmctx->src0_row_start + src0_nrows_per_thread * ith;
    स्थिर uint32_t src0_end_row    = MIN(src0_start_row + src0_nrows_per_thread, mmctx->src0_row_end);

    काष्ठा htp_thread_trace * tr = &octx->ctx->trace[ith];

    स्थिर काष्ठा htp_mm_kernel_params * kparams = (स्थिर काष्ठा htp_mm_kernel_params *) octx->kernel_params;
    स्थिर uint32_t n_prefetch = kparams->n_prefetch;
    निश्चित(n_prefetch >= 2 && n_prefetch <= HTP_MM_MAX_PREFETCH && (n_prefetch & (n_prefetch - 1)) == 0);

    स्थिर माप_प्रकार dst_row_आकार  = nb1;
    स्थिर माप_प्रकार act_row_आकार = nb11;
    स्थिर माप_प्रकार act_stride = mmctx->vtcm_act_stride;

    uint8_t * vtcm_dst_ptr  = mmctx->vtcm_dst + mmctx->vtcm_dst_आकार_per_thread * ith;
    uint8_t * vtcm_src0_ptr = mmctx->vtcm_src0 + mmctx->vtcm_src0_आकार_per_thread * ith;
    uint8_t * act_तथ्य = mmctx->vtcm_act;

    भग्न * tmp = (भग्न *) vtcm_dst_ptr;

    स्थिर dma_addr_t src0_row = src0->तथ्य;

    स्थिर uint8_t * restrict act_col = (स्थिर uint8_t *) act_तथ्य;
    भग्न * restrict dst_col          = (भग्न *) dst->तथ्य;

    स्थिर uint32_t tile_आकार = htp_tensor_get_tile_आकार(src0->type);
    स्थिर uint32_t aligned_tile_आकार = htp_tensor_get_tile_padded_आकार(src0->type);

    uint32_t n_k_tiles_w = ne00 / htp_tensor_get_tile_k(src0->type);
    uint32_t tile_row_stride = n_k_tiles_w * tile_आकार;
    uint32_t tile_row_transfer_आकार_aligned = n_k_tiles_w * aligned_tile_आकार;

    uint32_t ct_start = src0_start_row / 32;
    uint32_t ct_end   = (src0_end_row + 31) / 32;

    uint32_t धकेलो_ct = ct_start;
    अगर (src0_start_row < src0_end_row) {
        अगर (src2) {
            भग्न * vtcm_bias_ptr = (भग्न *) mmctx->vtcm_bias + src0_start_row;
            स्थिर dma_addr_t src2_addr = src2->तथ्य + src0_start_row * माप(भग्न);
            पूर्णांक slice_आकार = (पूर्णांक)MIN(src0_end_row, ne0) - (पूर्णांक)src0_start_row;
            अगर (slice_आकार > 0) {
                dma_queue_धकेलो(dma_q, dma_make_तथ्य(vtcm_bias_ptr, src2_addr),
                               slice_आकार * माप(भग्न), slice_आकार * माप(भग्न), slice_आकार * माप(भग्न), 1);
                dma_queue_निकालो_nowait(dma_q);
            }
        }
        क्रम (uint32_t d = 0; d < n_prefetch && धकेलो_ct < ct_end; d++, धकेलो_ct++) {
            dma_queue_धकेलो(dma_q, dma_make_तथ्य(vtcm_src0_ptr + d * tile_row_transfer_आकार_aligned,
                           src0_row + धकेलो_ct * tile_row_stride), aligned_tile_आकार, tile_आकार, tile_आकार, n_k_tiles_w);
        }
    }

    अगर (src0_start_row >= src0_end_row) {
        वापस;
    }

    क्रम (uint32_t ct = ct_start; ct < ct_end; ct++) {
        स्थिर uint8_t * w_tile = (व्योम *) dma_queue_निकालो(dma_q).dst;

        भग्न * dst_ptr = &tmp[ct * 32 - src0_start_row];
        पूर्णांक valid_rows = (पूर्णांक)ne0 - (पूर्णांक)(ct * 32);
        valid_rows = MIN(32, MAX(0, valid_rows));

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, ct);
        mmctx->vec_dot_32x1(ne10, dst_ptr, w_tile, act_col, valid_rows, शून्य);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ct);

        अगर (धकेलो_ct < ct_end) {
            dma_queue_धकेलो(dma_q, dma_make_तथ्य(w_tile, src0_row + धकेलो_ct * tile_row_stride),
                           aligned_tile_आकार, tile_आकार, tile_आकार, n_k_tiles_w);
            धकेलो_ct++;
        }
    }

    पूर्णांक copy_cnt = (पूर्णांक)MIN(src0_end_row, ne0) - (पूर्णांक)src0_start_row;
    अगर (copy_cnt > 0) {
        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, ct_end);
        अगर (src2) {
            hvx_add_f32_uuu((uint8_t *) &dst_col[src0_start_row],
                            (स्थिर uint8_t *) tmp,
                            (स्थिर uint8_t *) ((स्थिर भग्न *) mmctx->vtcm_bias + src0_start_row),
                            copy_cnt);
        } अन्यथा {
            hvx_copy_f32_ua((uint8_t *) &dst_col[src0_start_row], (uint8_t *) tmp, copy_cnt);
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ct_end);
    }
}

अटल व्योम hvx_mm_nx_2d_tiled(अचिन्हित पूर्णांक nth, अचिन्हित पूर्णांक ith, व्योम * तथ्य) {
    काष्ठा htp_mm_context * mmctx = तथ्य;
    काष्ठा htp_ops_context * octx = mmctx->octx;
    स्थिर काष्ठा htp_mm_kernel_params * kparams = (स्थिर काष्ठा htp_mm_kernel_params *) octx->kernel_params;
    स्थिर uint32_t n_weights = kparams->n_weights;

    स्थिर काष्ठा htp_tensor * restrict act = octx->src[n_weights]; /* x */
    स्थिर uint32_t ne10 = act->ne[0];
    स्थिर uint32_t act_nrows = act->ne[1] * act->ne[2] * act->ne[3];
    स्थिर माप_प्रकार act_stride = mmctx->vtcm_act_stride;

    uint8_t * restrict vtcm_weight_ptr = mmctx->vtcm_src0 + mmctx->vtcm_src0_आकार_per_thread * ith;
    uint8_t * restrict act_तथ्य        = mmctx->vtcm_act;

    काष्ठा htp_thread_trace * tr = &octx->ctx->trace[ith];
    स्थिर uint32_t n_prefetch = kparams->n_prefetch;
    निश्चित(n_prefetch >= 2 && n_prefetch <= HTP_MM_MAX_PREFETCH && (n_prefetch & (n_prefetch - 1)) == 0);

    क्रम (uint32_t widx = 0; widx < n_weights; widx++) {
        स्थिर काष्ठा htp_tensor * restrict src_w = octx->src[widx];
        स्थिर काष्ठा htp_tensor * restrict dst   = octx->dsts[widx];
        अगर (!src_w || !dst) जारी;
        dma_queue * dma_q = octx->ctx->dma[ith];

        स्थिर uint32_t tile_आकार = htp_tensor_get_tile_आकार(src_w->type);
        स्थिर uint32_t aligned_tile_आकार = htp_tensor_get_tile_padded_आकार(src_w->type);
        स्थिर uint32_t ne00 = src_w->ne[0];
        स्थिर uint32_t ne01 = src_w->ne[1];
        स्थिर माप_प्रकार dst_row_आकार = dst->nb[1];
        स्थिर dma_addr_t src_w_row = src_w->तथ्य;

        स्थिर uint32_t n_k_tiles_w = ne00 / htp_tensor_get_tile_k(src_w->type);
        स्थिर uint32_t tile_row_stride = n_k_tiles_w * tile_आकार;
        स्थिर uint32_t tile_row_transfer_आकार_aligned = n_k_tiles_w * aligned_tile_आकार;

        uint32_t src0_start_row = 0;
        uint32_t src0_end_row   = ne01;
        अगर (octx->ctx->mdev.गिनती > 1) {
            स्थिर bool can_split = htp_tensor_can_row_partition(dst, माप(भग्न));
            स्थिर काष्ठा htp_tensor_mdev_range range = htp_tensor_mdev_partition(ne01, can_split ? 32 : 0,
                                                         octx->ctx->mdev.idx, octx->ctx->mdev.गिनती, &octx->ctx->mdev.गिनती_div);
            src0_start_row = range.start;
            src0_end_row   = range.start + range.गिनती;
        }

        स्थिर uint32_t nrows = src0_end_row - src0_start_row;
        uint32_t src0_nrows_per_thread = fastdiv(nrows + nth - 1, &octx->n_threads_div);
        src0_nrows_per_thread = hex_round_up(src0_nrows_per_thread, 32);

        स्थिर uint32_t start_row = src0_start_row + src0_nrows_per_thread * ith;
        स्थिर uint32_t end_row   = MIN(start_row + src0_nrows_per_thread, src0_end_row);
        अगर (start_row >= end_row) जारी;

        uint32_t ct_start = start_row / 32;
        uint32_t ct_end   = (end_row + 31) / 32;

        uint32_t धकेलो_ct = ct_start;
        क्रम (uint32_t d = 0; d < n_prefetch && धकेलो_ct < ct_end; d++, धकेलो_ct++) {
            dma_queue_धकेलो(dma_q, dma_make_तथ्य(vtcm_weight_ptr + d * tile_row_transfer_आकार_aligned,
                            src_w_row + धकेलो_ct * tile_row_stride), aligned_tile_आकार, tile_आकार, tile_आकार, n_k_tiles_w);
        }

        क्रम (uint32_t ct = ct_start; ct < ct_end; ct++) {
            स्थिर uint8_t * w_tile = (व्योम *) dma_queue_निकालो(dma_q).dst;
            पूर्णांक valid_rows = (पूर्णांक)dst->ne[0] - (पूर्णांक)(ct * 32);
            valid_rows = MIN(32, MAX(0, valid_rows));

            htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, ct);
            uint32_t ir1 = 0;
            क्रम (; ir1 + 1 < act_nrows; ir1 += 2) {
                स्थिर uint8_t * restrict act_col0 = (स्थिर uint8_t *) (act_तथ्य + (ir1+0) * act_stride);
                स्थिर uint8_t * restrict act_col1 = (स्थिर uint8_t *) (act_तथ्य + (ir1+1) * act_stride);

                भग्न * restrict dst_row0 = (भग्न *) (dst->तथ्य + ((ir1+0) * dst_row_आकार));
                भग्न * restrict dst_row1 = (भग्न *) (dst->तथ्य + ((ir1+1) * dst_row_आकार));
                भग्न * dst_ptr0 = &dst_row0[ct * 32];
                भग्न * dst_ptr1 = &dst_row1[ct * 32];

                mmctx->vec_dot_32x2(ne10, dst_ptr0, dst_ptr1, w_tile, act_col0, act_col1, valid_rows, शून्य, शून्य);
            }

            क्रम (; ir1 < act_nrows; ++ir1) {
                स्थिर uint8_t * restrict act_col = (स्थिर uint8_t *) (act_तथ्य + ir1 * act_stride);
                भग्न * restrict dst_row = (भग्न *) (dst->तथ्य + (ir1 * dst_row_आकार));
                भग्न * dst_ptr = &dst_row[ct * 32];
                mmctx->vec_dot_32x1(ne10, dst_ptr, w_tile, act_col, valid_rows, शून्य);
            }
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ct);

            अगर (धकेलो_ct < ct_end) {
                dma_queue_धकेलो(dma_q, dma_make_तथ्य(w_tile, src_w_row + धकेलो_ct * tile_row_stride),
                                aligned_tile_आकार, tile_आकार, tile_आकार, n_k_tiles_w);
                धकेलो_ct++;
            }
        }
    }
}

अटल व्योम hvx_mm_transfer_act_dma(
    काष्ठा htp_ops_context * octx,
    स्थिर काष्ठा htp_mm_kernel_params * kparams,
    स्थिर काष्ठा htp_tensor * act,
    uint8_t * dst_base,
    माप_प्रकार dst_row_आकार,
    uint32_t m_start,
    uint32_t m_rows
) {
    अगर (m_rows == 0) {
        वापस;
    }

    dma_queue * dma_q = octx->ctx->dma[0];
    स्थिर uint32_t ne0 = act->ne[0];
    स्थिर माप_प्रकार elem_आकार = (act->type == HTP_TYPE_F16) ? माप(__fp16) : माप(भग्न);
    स्थिर माप_प्रकार row_bytes = ne0 * elem_आकार;
    स्थिर माप_प्रकार act_nb1 = act->nb[1];
    स्थिर dma_addr_t act_base = act->तथ्य;

    स्थिर bool is_contiguous = (act->nb[2] == act->ne[1] * act_nb1) &&
                               (act->nb[3] == act->ne[2] * act->nb[2]);

    अगर (is_contiguous) {
        स्थिर dma_addr_t src_addr = act_base + m_start * act_nb1;
        dma_queue_धकेलो(dma_q, dma_make_तथ्य(dst_base, src_addr),
                       dst_row_आकार, act_nb1, row_bytes, m_rows);
        dma_queue_निकालो(dma_q);
    } अन्यथा {
        स्थिर uint32_t ne12_ne1 = act->ne[2] * act->ne[1];
        स्थिर bool use_fastdiv = kparams->div_ne12_ne1.mp != 0;
        क्रम (uint32_t ir = 0; ir < m_rows; ++ir) {
            स्थिर uint32_t ir1 = m_start + ir;
            uint32_t i11, i12, i13;
            अगर (use_fastdiv) {
                i13 = fastdiv(ir1, &kparams->div_ne12_ne1);
                स्थिर uint32_t rem = ir1 - i13 * ne12_ne1;
                i12 = fastdiv(rem, &kparams->div_ne1);
                i11 = rem - i12 * act->ne[1];
            } अन्यथा {
                i13 = ne12_ne1 ? ir1 / ne12_ne1 : 0;
                स्थिर uint32_t rem = ir1 - i13 * ne12_ne1;
                i12 = act->ne[1] ? rem / act->ne[1] : 0;
                i11 = rem - i12 * act->ne[1];
            }
            स्थिर dma_addr_t row_src = act_base + (i11 * act->nb[1] +
                                                   i12 * act->nb[2] +
                                                   i13 * act->nb[3]);
            uint8_t * row_dst = dst_base + ir * dst_row_आकार;
            dma_queue_धकेलो(dma_q, dma_make_तथ्य(row_dst, row_src),
                           dst_row_आकार, act_nb1, row_bytes, 1);
            dma_queue_निकालो(dma_q);
        }
    }

    अगर (dst_row_आकार > row_bytes) {
        काष्ठा htp_thread_trace * tr = &octx->ctx->trace[0];
        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_A_PREP, (uint16_t) m_start);
        स्थिर uint32_t pad_elems = (dst_row_आकार - row_bytes) / elem_आकार;
        अगर (elem_आकार == माप(भग्न)) {
            क्रम (uint32_t ir = 0; ir < m_rows; ++ir) {
                hvx_splat_f32_u(dst_base + ir * dst_row_आकार + row_bytes, 0.0f, pad_elems);
            }
        } अन्यथा {
            क्रम (uint32_t ir = 0; ir < m_rows; ++ir) {
                hvx_splat_f16_u(dst_base + ir * dst_row_आकार + row_bytes, (_Float16) 0.0f, pad_elems);
            }
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_A_PREP, (uint16_t) m_start);
    }
}

#घोषणा QUANTIZE_IMPL(name, log_name, kernel_fn, dst_row_आकार_व्यंजक)                                        \
अटल व्योम name(अचिन्हित पूर्णांक nth, अचिन्हित पूर्णांक ith, व्योम * तथ्य) {                                        \
    (व्योम) nth;                                                                                            \
    काष्ठा htp_mm_context * mmctx = तथ्य;                                                                  \
    काष्ठा htp_ops_context * octx = mmctx->octx;                                                           \
    स्थिर काष्ठा htp_tensor * src = mmctx->act;                                                            \
    स्थिर uint32_t ne0 = src->ne[0];                                                                       \
    स्थिर uint32_t nrows = mmctx->cur_m_rows ? mmctx->cur_m_rows : mmctx->act_nrows;                       \
    स्थिर uint32_t nrows_per_thread = mmctx->n_quant_rows_per_thread;                                      \
                                                                                                           \
    स्थिर uint32_t ir_first = nrows_per_thread * ith;                                                      \
    अगर (ir_first >= nrows) {                                                                               \
        वापस;                                                                                            \
    }                                                                                                      \
                                                                                                           \
    काष्ठा htp_thread_trace * tr = &octx->ctx->trace[ith];                                                 \
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_A_QUANT, ir_first);                                        \
                                                                                                           \
    uint8_t * restrict dst = mmctx->vtcm_act;                                                              \
    स्थिर uint32_t ir_last = MIN(ir_first + nrows_per_thread, nrows);                                      \
    स्थिर माप_प्रकार raw_row_आकार = mmctx->vtcm_act_raw_stride;                                                \
    स्थिर माप_प्रकार dst_row_आकार = (dst_row_आकार_व्यंजक);                                                       \
                                                                                                           \
    स्थिर uint8_t * restrict src_तथ्य = (स्थिर uint8_t *) mmctx->vtcm_act_raw + (raw_row_आकार * ir_first); \
    uint8_t * restrict dst_तथ्य = (uint8_t *) dst + (dst_row_आकार * ir_first);                             \
    kernel_fn(src_तथ्य, dst_तथ्य, शून्य, ne0, ir_last - ir_first, raw_row_आकार, dst_row_आकार);              \
                                                                                                           \
    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_A_QUANT, ir_first);                                         \
}

QUANTIZE_IMPL(quantize_f32_q8_0_tiled,     "quantize-f32-q8_0_tiled",     quantize_f32_q8_0_tiled_kernel,     htp_mm_q8_0_tiled_row_आकार(ne0))
QUANTIZE_IMPL(quantize_f32_q8_1_tiled,     "quantize-f32-q8_1_tiled",     quantize_f32_q8_1_tiled_kernel,     htp_mm_q8_1_tiled_row_आकार(ne0))
QUANTIZE_IMPL(quantize_f32_q8_1_s16_tiled, "quantize-f32-q8_1_s16_tiled", quantize_f32_q8_1_s16_tiled_kernel, htp_mm_q8_1_tiled_row_आकार(ne0))
QUANTIZE_IMPL(quantize_f32_f16_tiled,      "quantize-f32-f16_tiled",      quantize_f32_f16_tiled_kernel,      htp_mm_f16_tiled_row_आकार(ne0))
QUANTIZE_IMPL(quantize_f16_f16_tiled,      "quantize-f16-f16_tiled",      quantize_f16_f16_tiled_kernel,      htp_mm_f16_tiled_row_आकार(ne0))
QUANTIZE_IMPL(quantize_f32_f32,            "quantize-f32-f32",            quantize_f32_f32_kernel,            mmctx->vtcm_act_stride)
QUANTIZE_IMPL(quantize_f32_f16,            "quantize-f32-f16",            quantize_f32_f16_kernel,            mmctx->vtcm_act_stride)
QUANTIZE_IMPL(quantize_f16_f16,            "quantize-f16-f16",            quantize_f16_f16_kernel,            mmctx->vtcm_act_stride)

अटल व्योम quantize_f32_q8_0_tiled_block(अचिन्हित पूर्णांक nth, अचिन्हित पूर्णांक ith, व्योम * तथ्य) {
    (व्योम) nth;
    काष्ठा htp_mm_context * mmctx = तथ्य;
    अगर (mmctx->quant_ib_first[ith] >= mmctx->quant_ib_last[ith]) {
        वापस;
    }
    काष्ठा htp_ops_context * octx = mmctx->octx;
    काष्ठा htp_thread_trace * tr = &octx->ctx->trace[ith];
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_A_QUANT, mmctx->quant_ib_first[ith]);

    स्थिर काष्ठा htp_tensor * src = mmctx->act;

    quantize_f32_q8_0_tiled_block_kernel(
        (स्थिर भग्न *) mmctx->vtcm_act_raw,
        mmctx->vtcm_act,
        शून्य,
        src->ne[0],
        mmctx->quant_ib_first[ith],
        mmctx->quant_ib_last[ith],
        mmctx->vtcm_act_raw_stride,
        htp_mm_q8_0_tiled_row_आकार(src->ne[0]),
        mmctx->quant_r[ith],
        mmctx->quant_c[ith]
    );

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_A_QUANT, mmctx->quant_ib_first[ith]);
}

अटल व्योम quantize_f32_q8_1_tiled_block(अचिन्हित पूर्णांक nth, अचिन्हित पूर्णांक ith, व्योम * तथ्य) {
    (व्योम) nth;
    काष्ठा htp_mm_context * mmctx = तथ्य;
    अगर (mmctx->quant_ib_first[ith] >= mmctx->quant_ib_last[ith]) {
        वापस;
    }
    काष्ठा htp_ops_context * octx = mmctx->octx;
    काष्ठा htp_thread_trace * tr = &octx->ctx->trace[ith];
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_A_QUANT, mmctx->quant_ib_first[ith]);

    स्थिर काष्ठा htp_tensor * src = mmctx->act;

    quantize_f32_q8_1_tiled_block_kernel(
        (स्थिर भग्न *) mmctx->vtcm_act_raw,
        mmctx->vtcm_act,
        शून्य,
        src->ne[0],
        mmctx->quant_ib_first[ith],
        mmctx->quant_ib_last[ith],
        mmctx->vtcm_act_raw_stride,
        htp_mm_q8_1_tiled_row_आकार(src->ne[0]),
        mmctx->quant_r[ith],
        mmctx->quant_c[ith]
    );

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_A_QUANT, mmctx->quant_ib_first[ith]);
}

अटल व्योम quantize_f32_q8_1_s16_tiled_block(अचिन्हित पूर्णांक nth, अचिन्हित पूर्णांक ith, व्योम * तथ्य) {
    (व्योम) nth;
    काष्ठा htp_mm_context * mmctx = तथ्य;
    अगर (mmctx->quant_ib_first[ith] >= mmctx->quant_ib_last[ith]) {
        वापस;
    }
    काष्ठा htp_ops_context * octx = mmctx->octx;
    काष्ठा htp_thread_trace * tr = &octx->ctx->trace[ith];
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_A_QUANT, mmctx->quant_ib_first[ith]);

    स्थिर काष्ठा htp_tensor * src = mmctx->act;

    quantize_f32_q8_1_s16_tiled_block_kernel(
        (स्थिर भग्न *) mmctx->vtcm_act_raw,
        mmctx->vtcm_act,
        शून्य,
        src->ne[0],
        mmctx->quant_ib_first[ith],
        mmctx->quant_ib_last[ith],
        mmctx->vtcm_act_raw_stride,
        htp_mm_q8_1_tiled_row_आकार(src->ne[0]),
        mmctx->quant_r[ith],
        mmctx->quant_c[ith]
    );

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_A_QUANT, mmctx->quant_ib_first[ith]);
}

अटल व्योम quantize_f32_f16_tiled_block(अचिन्हित पूर्णांक nth, अचिन्हित पूर्णांक ith, व्योम * तथ्य) {
    (व्योम) nth;
    काष्ठा htp_mm_context * mmctx = तथ्य;
    अगर (mmctx->quant_ib_first[ith] >= mmctx->quant_ib_last[ith]) {
        वापस;
    }
    काष्ठा htp_ops_context * octx = mmctx->octx;
    काष्ठा htp_thread_trace * tr = &octx->ctx->trace[ith];
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_A_QUANT, mmctx->quant_ib_first[ith]);

    स्थिर काष्ठा htp_tensor * src = mmctx->act;

    quantize_f32_f16_tiled_block_kernel(
        (स्थिर भग्न *) mmctx->vtcm_act_raw,
        mmctx->vtcm_act,
        शून्य,
        src->ne[0],
        mmctx->quant_ib_first[ith],
        mmctx->quant_ib_last[ith],
        mmctx->vtcm_act_raw_stride,
        htp_mm_f16_tiled_row_आकार(src->ne[0]),
        mmctx->quant_r[ith],
        mmctx->quant_c[ith]
    );

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_A_QUANT, mmctx->quant_ib_first[ith]);
}

अटल व्योम quantize_f16_f16_tiled_block(अचिन्हित पूर्णांक nth, अचिन्हित पूर्णांक ith, व्योम * तथ्य) {
    (व्योम) nth;
    काष्ठा htp_mm_context * mmctx = तथ्य;
    अगर (mmctx->quant_ib_first[ith] >= mmctx->quant_ib_last[ith]) {
        वापस;
    }
    काष्ठा htp_ops_context * octx = mmctx->octx;
    काष्ठा htp_thread_trace * tr = &octx->ctx->trace[ith];
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_A_QUANT, mmctx->quant_ib_first[ith]);

    स्थिर काष्ठा htp_tensor * src = mmctx->act;

    quantize_f16_f16_tiled_block_kernel(
        (स्थिर __fp16 *) mmctx->vtcm_act_raw,
        mmctx->vtcm_act,
        शून्य,
        src->ne[0],
        mmctx->quant_ib_first[ith],
        mmctx->quant_ib_last[ith],
        mmctx->vtcm_act_raw_stride,
        htp_mm_f16_tiled_row_आकार(src->ne[0]),
        mmctx->quant_r[ith],
        mmctx->quant_c[ith]
    );

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_A_QUANT, mmctx->quant_ib_first[ith]);
}

// q8_1 for weight types with offsets (q8_1_s16 for Q2_K), f16_tiled for F16, otherwise q8_0
अटल अंतरभूत worker_callback_t htp_mm_act_quant_row_func(पूर्णांक weight_type, पूर्णांक act_type) {
    अगर (weight_type == HTP_TYPE_F16) {
        वापस (act_type == HTP_TYPE_F16) ? quantize_f16_f16_tiled : quantize_f32_f16_tiled;
    }
    अगर (weight_type == HTP_TYPE_Q2_K) {
        वापस quantize_f32_q8_1_s16_tiled;
    }
    वापस htp_mm_weight_has_offset(weight_type) ? quantize_f32_q8_1_tiled : quantize_f32_q8_0_tiled;
}

अटल अंतरभूत worker_callback_t htp_mm_act_quant_block_func(पूर्णांक weight_type, पूर्णांक act_type) {
    अगर (weight_type == HTP_TYPE_F16) {
        वापस (act_type == HTP_TYPE_F16) ? quantize_f16_f16_tiled_block : quantize_f32_f16_tiled_block;
    }
    अगर (weight_type == HTP_TYPE_Q2_K) {
        वापस quantize_f32_q8_1_s16_tiled_block;
    }
    वापस htp_mm_weight_has_offset(weight_type) ? quantize_f32_q8_1_tiled_block : quantize_f32_q8_0_tiled_block;
}

अटल व्योम hvx_mm_4d_tiled(अचिन्हित पूर्णांक nth, अचिन्हित पूर्णांक ith, व्योम * तथ्य) {
    htp_matmul_preamble;

    स्थिर uint32_t src0_nrows = mmctx->src0_row_end - mmctx->src0_row_start;
    स्थिर uint32_t cur_m_rows = mmctx->cur_m_rows ? mmctx->cur_m_rows : (ne11 * ne12 * ne13);
    स्थिर uint32_t cur_m_start = mmctx->cur_m_start;

    स्थिर uint32_t src0_start_row  = mmctx->src0_row_start + src0_nrows_per_thread * ith;
    स्थिर uint32_t src0_end_row    = MIN(src0_start_row + src0_nrows_per_thread, mmctx->src0_row_end);

    काष्ठा htp_thread_trace * tr = &octx->ctx->trace[ith];

    स्थिर काष्ठा htp_mm_kernel_params * kparams = (स्थिर काष्ठा htp_mm_kernel_params *) octx->kernel_params;
    स्थिर uint32_t n_prefetch = kparams->n_prefetch;
    निश्चित(n_prefetch >= 2 && n_prefetch <= HTP_MM_MAX_PREFETCH && (n_prefetch & (n_prefetch - 1)) == 0);

    स्थिर माप_प्रकार dst_row_आकार = nb1;
    स्थिर माप_प्रकार act_stride = mmctx->vtcm_act_stride;

    uint8_t * restrict vtcm_src0_ptr = mmctx->vtcm_src0 + mmctx->vtcm_src0_आकार_per_thread * ith;
    uint8_t * restrict act_तथ्य = mmctx->vtcm_act;

    स्थिर uint32_t tile_आकार = htp_tensor_get_tile_आकार(src0->type);
    स्थिर uint32_t aligned_tile_आकार = htp_tensor_get_tile_padded_आकार(src0->type);

    स्थिर uint32_t n_k_tiles_w = ne00 / htp_tensor_get_tile_k(src0->type);
    स्थिर uint32_t tile_row_stride = n_k_tiles_w * tile_आकार;
    स्थिर uint32_t tile_row_transfer_आकार_aligned = n_k_tiles_w * aligned_tile_आकार;
    स्थिर uint32_t src0_slice_stride = ((ne01 + 31) / 32) * tile_row_stride;

    स्थिर uint32_t ct_start = src0_start_row / 32;
    स्थिर uint32_t ct_end   = (src0_end_row + 31) / 32;

    अगर (src0_start_row >= src0_end_row || cur_m_rows == 0) {
        वापस;
    }

    स्थिर uint32_t योग_batches = ne12 * ne13;
    स्थिर uint32_t b_start = fastdiv(cur_m_start, &kparams->div_ne1);
    uint32_t b_end = fastdiv(cur_m_start + cur_m_rows + ne11 - 1, &kparams->div_ne1);
    b_end = MIN(b_end, योग_batches);

    uint32_t b_grp_start = b_start;
    जबतक (b_grp_start < b_end) {
        स्थिर uint32_t b3 = fastdiv(b_grp_start, &kparams->div_ne12);
        स्थिर uint32_t b2 = b_grp_start - b3 * ne12;
        स्थिर uint32_t i02 = fastdiv(b2, &kparams->div_r2);
        स्थिर uint32_t i03 = fastdiv(b3, &kparams->div_r3);

        uint32_t b_grp_end = b_grp_start + 1;
        जबतक (b_grp_end < b_end) {
            स्थिर uint32_t cur_b3 = fastdiv(b_grp_end, &kparams->div_ne12);
            स्थिर uint32_t cur_b2 = b_grp_end - cur_b3 * ne12;
            स्थिर uint32_t cur_i02 = fastdiv(cur_b2, &kparams->div_r2);
            स्थिर uint32_t cur_i03 = fastdiv(cur_b3, &kparams->div_r3);
            अगर (cur_i02 != i02 || cur_i03 != i03) {
                अवरोध;
            }
            b_grp_end++;
        }

        स्थिर uint32_t slice_idx = i03 * ne02 + i02;
        स्थिर dma_addr_t src0_slice = src0->तथ्य + (माप_प्रकार) slice_idx * src0_slice_stride;

        uint32_t धकेलो_ct = ct_start;
        क्रम (uint32_t d = 0; d < n_prefetch && धकेलो_ct < ct_end; d++, धकेलो_ct++) {
            dma_queue_धकेलो(dma_q, dma_make_तथ्य(vtcm_src0_ptr + d * tile_row_transfer_आकार_aligned,
                           src0_slice + (माप_प्रकार) धकेलो_ct * tile_row_stride),
                           aligned_tile_आकार, tile_आकार, tile_आकार, n_k_tiles_w);
        }

        क्रम (uint32_t ct = ct_start; ct < ct_end; ct++) {
            स्थिर uint8_t * w_tile = (व्योम *) dma_queue_निकालो(dma_q).dst;

            पूर्णांक valid_rows = (पूर्णांक)ne0 - (पूर्णांक)(ct * 32);
            valid_rows = MIN(32, MAX(0, valid_rows));

            htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, ct);
            क्रम (uint32_t b = b_grp_start; b < b_grp_end; b++) {
                स्थिर uint32_t b_m_start = b * ne11;
                स्थिर uint32_t m_first = MAX(cur_m_start, b_m_start);
                स्थिर uint32_t m_last  = MIN(cur_m_start + cur_m_rows, b_m_start + ne11);
                अगर (m_first >= m_last) जारी;

                स्थिर uint32_t cur_b3 = fastdiv(b, &kparams->div_ne12);
                स्थिर uint32_t cur_b2 = b - cur_b3 * ne12;
                uint8_t * dst_batch_base = (uint8_t *) dst->तथ्य + (माप_प्रकार) cur_b2 * nb2 + (माप_प्रकार) cur_b3 * nb3;

                स्थिर uint32_t chunk_m_offset = m_first - cur_m_start;
                स्थिर uint32_t dst_m_offset   = m_first - b_m_start;
                स्थिर uint32_t batch_nrows    = m_last - m_first;

                uint32_t ir1 = 0;
                क्रम (; ir1 + 1 < batch_nrows; ir1 += 2) {
                    स्थिर uint8_t * restrict act_col0 = (स्थिर uint8_t *) (act_तथ्य + (chunk_m_offset + ir1 + 0) * act_stride);
                    स्थिर uint8_t * restrict act_col1 = (स्थिर uint8_t *) (act_तथ्य + (chunk_m_offset + ir1 + 1) * act_stride);
                    भग्न * restrict dst_row0 = (भग्न *) (dst_batch_base + (dst_m_offset + ir1 + 0) * dst_row_आकार);
                    भग्न * restrict dst_row1 = (भग्न *) (dst_batch_base + (dst_m_offset + ir1 + 1) * dst_row_आकार);
                    भग्न * dst_ptr0 = &dst_row0[ct * 32];
                    भग्न * dst_ptr1 = &dst_row1[ct * 32];
                    mmctx->vec_dot_32x2(ne10, dst_ptr0, dst_ptr1, w_tile, act_col0, act_col1, valid_rows, शून्य, शून्य);
                }
                क्रम (; ir1 < batch_nrows; ++ir1) {
                    स्थिर uint8_t * restrict act_col = (स्थिर uint8_t *) (act_तथ्य + (chunk_m_offset + ir1) * act_stride);
                    भग्न * restrict dst_row = (भग्न *) (dst_batch_base + (dst_m_offset + ir1) * dst_row_आकार);
                    भग्न * dst_ptr = &dst_row[ct * 32];
                    mmctx->vec_dot_32x1(ne10, dst_ptr, w_tile, act_col, valid_rows, शून्य);
                }
            }
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ct);

            अगर (धकेलो_ct < ct_end) {
                dma_queue_धकेलो(dma_q, dma_make_तथ्य(w_tile, src0_slice + (माप_प्रकार) धकेलो_ct * tile_row_stride),
                               aligned_tile_आकार, tile_आकार, tile_आकार, n_k_tiles_w);
                धकेलो_ct++;
            }
        }
        b_grp_start = b_grp_end;
    }
    अगर (src2) {
        hvx_tensor_add_f32_grid(dst, src2, cur_m_start, cur_m_start + cur_m_rows, src0_start_row, src0_end_row, &kparams->div_ne12_ne1, &kparams->div_ne1);
    }
}

अटल व्योम hvx_mm_2d_flat(अचिन्हित पूर्णांक nth, अचिन्हित पूर्णांक ith, व्योम * तथ्य) {
    htp_matmul_preamble;

    स्थिर काष्ठा htp_mm_kernel_params * kparams = (स्थिर काष्ठा htp_mm_kernel_params *) octx->kernel_params;
    स्थिर uint32_t n_prefetch = kparams->n_prefetch;
    निश्चित(n_prefetch >= 2 && n_prefetch <= HTP_MM_MAX_PREFETCH && (n_prefetch & (n_prefetch - 1)) == 0);
    स्थिर uint32_t prefetch_mask = n_prefetch - 1;

    स्थिर uint32_t src0_nrows = mmctx->src0_row_end - mmctx->src0_row_start;  // src0 rows
    स्थिर uint32_t act_nrows = mmctx->cur_m_rows ? mmctx->cur_m_rows : mmctx->act_nrows;
    स्थिर uint32_t cur_m_start = mmctx->cur_m_start;

    स्थिर uint32_t src0_start_row  = mmctx->src0_row_start + src0_nrows_per_thread * ith;
    स्थिर uint32_t src0_end_row    = MIN(src0_start_row + src0_nrows_per_thread, mmctx->src0_row_end);
    स्थिर uint32_t src0_end_row_x2 = src0_start_row + ((src0_end_row - src0_start_row) & ~1U);

    काष्ठा htp_thread_trace * tr = &octx->ctx->trace[ith];

    स्थिर माप_प्रकार dst_row_आकार  = nb1;
    स्थिर माप_प्रकार src0_row_आकार = nb01;
    स्थिर माप_प्रकार act_row_आकार = nb11;

    स्थिर माप_प्रकार src0_stride = mmctx->vtcm_src0_stride;
    स्थिर माप_प्रकार act_stride = mmctx->vtcm_act_stride;

    // Per-thread VTCMs for all tensors
    uint8_t * restrict vtcm_dst_ptr  = mmctx->vtcm_dst  + mmctx->vtcm_dst_आकार_per_thread  * ith;
    uint8_t * restrict vtcm_src0_ptr = mmctx->vtcm_src0 + mmctx->vtcm_src0_आकार_per_thread * ith;
    uint8_t * restrict act_तथ्य      = mmctx->vtcm_act;

    स्थिर dma_addr_t src0_row = src0->तथ्य;


    // Prefill vtcm with src0 rows
    अगर (src0_start_row < src0_end_row) {
        क्रम (uint32_t ir0 = src0_start_row; ir0 < src0_end_row_x2; ir0 += 2) {
            स्थिर पूर्णांक is0 = (ir0 - src0_start_row);
            अगर (is0 >= (पूर्णांक)n_prefetch) {
                अवरोध;
            }
            dma_queue_धकेलो(dma_q, dma_make_तथ्य(vtcm_src0_ptr + is0 * src0_stride, src0_row + ir0 * src0_row_आकार),
                           src0_stride, src0_row_आकार, src0_row_आकार, 2);
        }
    }

    अगर (src0_start_row >= src0_end_row) {
        वापस;
    }

    // Process src0 rows
    क्रम (uint32_t ir0 = src0_start_row; ir0 < src0_end_row_x2; ir0 += 2) {
        स्थिर uint8_t * ss0 = (व्योम *) dma_queue_निकालो(dma_q).dst;

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, ir0);
        // Process act columns in pairs (2x2 tiling)
        uint32_t ir1 = 0;
        क्रम (; ir1 + 1 < act_nrows; ir1 += 2) {
            स्थिर uint8_t * restrict act_col0 = (स्थिर uint8_t *) (act_तथ्य + (ir1+0) * act_stride);
            स्थिर uint8_t * restrict act_col1 = (स्थिर uint8_t *) (act_तथ्य + (ir1+1) * act_stride);
            भग्न * restrict dst_row0 = (भग्न *) (dst->तथ्य + ((cur_m_start + ir1+0) * dst_row_आकार));
            भग्न * restrict dst_row1 = (भग्न *) (dst->तथ्य + ((cur_m_start + ir1+1) * dst_row_आकार));
            mmctx->vec_dot_2x2(ne00, &dst_row0[ir0], &dst_row1[ir0], ss0, ss0 + src0_stride, act_col0, act_col1);
        }

        // Handle remaining act rows (fallback to 2x1)
        क्रम (; ir1 < act_nrows; ++ir1) {
            स्थिर uint8_t * restrict act_col = (स्थिर uint8_t *) (act_तथ्य + ir1 * act_stride);
            भग्न * restrict dst_row          = (भग्न *) (dst->तथ्य + ((cur_m_start + ir1) * dst_row_आकार));
            mmctx->vec_dot_2x1(ne00, &dst_row[ir0], ss0, ss0 + src0_stride, act_col);
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ir0);

        // Prefetch next (n + vtcm_nrows) row
        स्थिर पूर्णांक pr0 = (ir0 + n_prefetch);
        स्थिर पूर्णांक is0 = (pr0 - src0_start_row) & prefetch_mask;
        अगर (pr0 < src0_end_row_x2) {
            dma_queue_धकेलो(dma_q, dma_make_तथ्य(vtcm_src0_ptr + is0 * src0_stride, src0_row + pr0 * src0_row_आकार),
                           src0_stride, src0_row_आकार, src0_row_आकार, 2);
        }
    }

    // Process the last row (if any)
    अगर (src0_end_row != src0_end_row_x2) {
        uint32_t  ir0 = src0_end_row_x2;
        स्थिर पूर्णांक is0 = (ir0 - src0_start_row) & prefetch_mask;
        dma_queue_धकेलो(dma_q, dma_make_तथ्य(vtcm_src0_ptr + is0 * src0_stride, src0_row + ir0 * src0_row_आकार),
                       src0_stride, src0_row_आकार, src0_row_आकार, 1);
        स्थिर uint8_t * ss0 = (व्योम *) dma_queue_निकालो(dma_q).dst;

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, ir0);
        #आशय unroll(2)
        क्रम (uint32_t ir1 = 0; ir1 < act_nrows; ++ir1) {
            स्थिर uint8_t * restrict act_col = (स्थिर uint8_t *) (act_तथ्य + ir1 * act_stride);
            भग्न * restrict dst_row          = (भग्न *) (dst->तथ्य + ((cur_m_start + ir1) * dst_row_आकार));
            mmctx->vec_dot_1x1(ne00, &dst_row[ir0], ss0, act_col);
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ir0);
    }
    अगर (src2) {
        hvx_tensor_add_f32_grid(dst, src2, cur_m_start, cur_m_start + act_nrows, src0_start_row, src0_end_row, &kparams->div_ne12_ne1, &kparams->div_ne1);
    }
}

अटल व्योम hvx_mv_2d_flat(अचिन्हित पूर्णांक nth, अचिन्हित पूर्णांक ith, व्योम * तथ्य) {
    htp_matmul_preamble;

    स्थिर uint32_t src0_nrows = mmctx->src0_row_end - mmctx->src0_row_start;

    स्थिर uint32_t src0_start_row  = mmctx->src0_row_start + src0_nrows_per_thread * ith;
    स्थिर uint32_t src0_end_row    = MIN(src0_start_row + src0_nrows_per_thread, mmctx->src0_row_end);

    काष्ठा htp_thread_trace * tr = &octx->ctx->trace[ith];

    स्थिर माप_प्रकार dst_row_आकार  = nb1;
    स्थिर माप_प्रकार src0_row_आकार = nb01;
    स्थिर माप_प्रकार act_row_आकार  = nb11;

    स्थिर माप_प्रकार src0_stride = mmctx->vtcm_src0_stride;
    स्थिर माप_प्रकार act_stride  = mmctx->vtcm_act_stride;

    // Per-thread VTCMs for all tensors
    uint8_t * vtcm_dst_ptr  = mmctx->vtcm_dst  + mmctx->vtcm_dst_आकार_per_thread  * ith;
    uint8_t * vtcm_src0_ptr = mmctx->vtcm_src0 + mmctx->vtcm_src0_आकार_per_thread * ith;
    uint8_t * act_तथ्य      = mmctx->vtcm_act;

    भग्न * tmp = (भग्न *) vtcm_dst_ptr;

    स्थिर dma_addr_t src0_row = src0->तथ्य;
    स्थिर uint8_t * restrict act_col = (स्थिर uint8_t *) act_तथ्य;
    भग्न * restrict dst_col         = (भग्न *) dst->तथ्य;

    स्थिर uint32_t src0_end_row_x2 = src0_start_row + ((src0_end_row - src0_start_row) & ~1U);

    स्थिर काष्ठा htp_mm_kernel_params * kparams = (स्थिर काष्ठा htp_mm_kernel_params *) octx->kernel_params;
    स्थिर uint32_t n_prefetch = kparams->n_prefetch;
    निश्चित(n_prefetch >= 2 && n_prefetch <= HTP_MM_MAX_PREFETCH && (n_prefetch & (n_prefetch - 1)) == 0);
    स्थिर uint32_t prefetch_mask = n_prefetch - 1;

    // Prefill vtcm with 2x src0 rows
    अगर (src0_start_row < src0_end_row) {
        अगर (src2) {
            भग्न * vtcm_bias_ptr = (भग्न *) mmctx->vtcm_bias + src0_start_row;
            स्थिर dma_addr_t src2_addr = src2->तथ्य + src0_start_row * माप(भग्न);
            पूर्णांक slice_आकार = (पूर्णांक)src0_end_row - (पूर्णांक)src0_start_row;
            अगर (slice_आकार > 0) {
                dma_queue_धकेलो(dma_q, dma_make_तथ्य(vtcm_bias_ptr, src2_addr),
                               slice_आकार * माप(भग्न), slice_आकार * माप(भग्न), slice_आकार * माप(भग्न), 1);
                dma_queue_निकालो_nowait(dma_q);
            }
        }
        क्रम (uint32_t ir0 = src0_start_row; ir0 < src0_end_row_x2; ir0 += 2) {
            स्थिर uint32_t is0 = (ir0 - src0_start_row);
            अगर (is0 >= n_prefetch) {
                अवरोध;
            }
            dma_queue_धकेलो(dma_q, dma_make_तथ्य(vtcm_src0_ptr + is0 * src0_stride, src0_row + ir0 * src0_row_आकार),
                           src0_stride, src0_row_आकार, src0_row_आकार, 2);
        }
    }


    अगर (src0_start_row >= src0_end_row) {
        वापस;
    }

    // Process src0 rows
    क्रम (uint32_t ir0 = src0_start_row; ir0 < src0_end_row_x2; ir0 += 2) {
        स्थिर uint8_t * ss0 = (व्योम *) dma_queue_निकालो(dma_q).dst;
        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, ir0);
        mmctx->vec_dot_2x1(ne00, &tmp[ir0 - src0_start_row], ss0, ss0 + src0_stride, act_col);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ir0);

        // Prefetch next (n + vtcm_nrows) row
        स्थिर uint32_t pr0 = (ir0 + n_prefetch);
        स्थिर uint32_t is0 = (pr0 - src0_start_row) & prefetch_mask;
        अगर (pr0 < src0_end_row_x2) {
            dma_queue_धकेलो(dma_q, dma_make_तथ्य(vtcm_src0_ptr + is0 * src0_stride, src0_row + pr0 * src0_row_आकार),
                           src0_stride, src0_row_आकार, src0_row_आकार, 2);
        }
    }

    // Process the last row (if any)
    अगर (src0_end_row != src0_end_row_x2) {
        स्थिर uint32_t ir0 = src0_end_row_x2;
        स्थिर uint32_t is0 = (ir0 - src0_start_row) & prefetch_mask;
        dma_queue_धकेलो(dma_q, dma_make_तथ्य(vtcm_src0_ptr + is0 * src0_stride, src0_row + ir0 * src0_row_आकार),
                       src0_stride, src0_row_आकार, src0_row_आकार, 1);
        स्थिर uint8_t * ss0 = (व्योम *) dma_queue_निकालो(dma_q).dst;
        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, ir0);
        mmctx->vec_dot_1x1(ne00, &tmp[ir0 - src0_start_row], ss0, act_col);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ir0);
    }

    पूर्णांक copy_cnt = src0_end_row - src0_start_row;
    अगर (copy_cnt > 0) {
        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, src0_end_row);
        अगर (src2) {
            hvx_add_f32_uuu((uint8_t *) &dst_col[src0_start_row],
                            (स्थिर uint8_t *) tmp,
                            (स्थिर uint8_t *) ((स्थिर भग्न *) mmctx->vtcm_bias + src0_start_row),
                            copy_cnt);
        } अन्यथा {
            hvx_copy_f32_ua((uint8_t *) &dst_col[src0_start_row], (uint8_t *) tmp, copy_cnt);
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, src0_end_row);
    }
}

अटल व्योम hvx_mm_4d_flat(अचिन्हित पूर्णांक nth, अचिन्हित पूर्णांक ith, व्योम * तथ्य) {
    htp_matmul_preamble;

    स्थिर काष्ठा htp_mm_kernel_params * kparams = (स्थिर काष्ठा htp_mm_kernel_params *) octx->kernel_params;
    स्थिर uint32_t n_prefetch = kparams->n_prefetch;
    निश्चित(n_prefetch >= 2 && n_prefetch <= HTP_MM_MAX_PREFETCH && (n_prefetch & (n_prefetch - 1)) == 0);
    स्थिर uint32_t prefetch_mask = n_prefetch - 1;

    स्थिर uint32_t src0_nrows = mmctx->src0_row_end - mmctx->src0_row_start;
    स्थिर uint32_t cur_m_rows = mmctx->cur_m_rows ? mmctx->cur_m_rows : (ne11 * ne12 * ne13);
    स्थिर uint32_t cur_m_start = mmctx->cur_m_start;

    स्थिर uint32_t src0_start_row  = mmctx->src0_row_start + src0_nrows_per_thread * ith;
    स्थिर uint32_t src0_end_row    = MIN(src0_start_row + src0_nrows_per_thread, mmctx->src0_row_end);
    स्थिर uint32_t src0_end_row_x2 = src0_start_row + ((src0_end_row - src0_start_row) & ~1U);

    काष्ठा htp_thread_trace * tr = &octx->ctx->trace[ith];

    स्थिर माप_प्रकार dst_row_आकार  = nb1;
    स्थिर माप_प्रकार src0_row_आकार = nb01;
    स्थिर माप_प्रकार src0_stride = mmctx->vtcm_src0_stride;
    स्थिर माप_प्रकार act_stride  = mmctx->vtcm_act_stride;

    uint8_t * restrict vtcm_src0_ptr = mmctx->vtcm_src0 + mmctx->vtcm_src0_आकार_per_thread * ith;
    uint8_t * restrict act_तथ्य      = mmctx->vtcm_act;


    अगर (src0_start_row >= src0_end_row || cur_m_rows == 0) {
        वापस;
    }

    स्थिर uint32_t योग_batches = ne12 * ne13;
    स्थिर uint32_t b_start = fastdiv(cur_m_start, &kparams->div_ne1);
    uint32_t b_end = fastdiv(cur_m_start + cur_m_rows + ne11 - 1, &kparams->div_ne1);
    b_end = MIN(b_end, योग_batches);

    uint32_t b_grp_start = b_start;
    जबतक (b_grp_start < b_end) {
        स्थिर uint32_t b3 = fastdiv(b_grp_start, &kparams->div_ne12);
        स्थिर uint32_t b2 = b_grp_start - b3 * ne12;
        स्थिर uint32_t i02 = fastdiv(b2, &kparams->div_r2);
        स्थिर uint32_t i03 = fastdiv(b3, &kparams->div_r3);

        uint32_t b_grp_end = b_grp_start + 1;
        जबतक (b_grp_end < b_end) {
            स्थिर uint32_t cur_b3 = fastdiv(b_grp_end, &kparams->div_ne12);
            स्थिर uint32_t cur_b2 = b_grp_end - cur_b3 * ne12;
            स्थिर uint32_t cur_i02 = fastdiv(cur_b2, &kparams->div_r2);
            स्थिर uint32_t cur_i03 = fastdiv(cur_b3, &kparams->div_r3);
            अगर (cur_i02 != i02 || cur_i03 != i03) {
                अवरोध;
            }
            b_grp_end++;
        }

        स्थिर dma_addr_t src0_row = src0->तथ्य + ((माप_प्रकार) i02 * nb02 + (माप_प्रकार) i03 * nb03);

        क्रम (uint32_t ir0 = src0_start_row; ir0 < src0_end_row_x2; ir0 += 2) {
            स्थिर पूर्णांक is0 = (ir0 - src0_start_row);
            अगर (is0 >= (पूर्णांक)n_prefetch) {
                अवरोध;
            }
            dma_queue_धकेलो(dma_q, dma_make_तथ्य(vtcm_src0_ptr + is0 * src0_stride, src0_row + (माप_प्रकार) ir0 * src0_row_आकार),
                           src0_stride, src0_row_आकार, src0_row_आकार, 2);
        }

        क्रम (uint32_t ir0 = src0_start_row; ir0 < src0_end_row_x2; ir0 += 2) {
            स्थिर uint8_t * ss0 = (व्योम *) dma_queue_निकालो(dma_q).dst;

            htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, ir0);
            क्रम (uint32_t b = b_grp_start; b < b_grp_end; b++) {
                स्थिर uint32_t b_m_start = b * ne11;
                स्थिर uint32_t m_first = MAX(cur_m_start, b_m_start);
                स्थिर uint32_t m_last  = MIN(cur_m_start + cur_m_rows, b_m_start + ne11);
                अगर (m_first >= m_last) जारी;

                स्थिर uint32_t cur_b3 = fastdiv(b, &kparams->div_ne12);
                स्थिर uint32_t cur_b2 = b - cur_b3 * ne12;
                uint8_t * dst_batch_base = (uint8_t *) dst->तथ्य + (माप_प्रकार) cur_b2 * nb2 + (माप_प्रकार) cur_b3 * nb3;

                स्थिर uint32_t chunk_m_offset = m_first - cur_m_start;
                स्थिर uint32_t dst_m_offset   = m_first - b_m_start;
                स्थिर uint32_t batch_nrows    = m_last - m_first;

                uint32_t ir1 = 0;
                क्रम (; ir1 + 1 < batch_nrows; ir1 += 2) {
                    स्थिर uint8_t * restrict act_col0 = (स्थिर uint8_t *) (act_तथ्य + (chunk_m_offset + ir1 + 0) * act_stride);
                    स्थिर uint8_t * restrict act_col1 = (स्थिर uint8_t *) (act_तथ्य + (chunk_m_offset + ir1 + 1) * act_stride);
                    भग्न * restrict dst_row0 = (भग्न *) (dst_batch_base + (dst_m_offset + ir1 + 0) * dst_row_आकार);
                    भग्न * restrict dst_row1 = (भग्न *) (dst_batch_base + (dst_m_offset + ir1 + 1) * dst_row_आकार);
                    mmctx->vec_dot_2x2(ne00, &dst_row0[ir0], &dst_row1[ir0], ss0, ss0 + src0_stride, act_col0, act_col1);
                }
                क्रम (; ir1 < batch_nrows; ++ir1) {
                    स्थिर uint8_t * restrict act_col = (स्थिर uint8_t *) (act_तथ्य + (chunk_m_offset + ir1) * act_stride);
                    भग्न * restrict dst_row          = (भग्न *) (dst_batch_base + (dst_m_offset + ir1) * dst_row_आकार);
                    mmctx->vec_dot_2x1(ne00, &dst_row[ir0], ss0, ss0 + src0_stride, act_col);
                }
            }
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ir0);

            स्थिर पूर्णांक pr0 = (ir0 + n_prefetch);
            स्थिर पूर्णांक is0 = (pr0 - src0_start_row) & prefetch_mask;
            अगर (pr0 < src0_end_row_x2) {
                dma_queue_धकेलो(dma_q, dma_make_तथ्य(vtcm_src0_ptr + is0 * src0_stride, src0_row + (माप_प्रकार) pr0 * src0_row_आकार),
                               src0_stride, src0_row_आकार, src0_row_आकार, 2);
            }
        }

        अगर (src0_end_row != src0_end_row_x2) {
            uint32_t ir0 = src0_end_row_x2;
            स्थिर पूर्णांक is0 = (ir0 - src0_start_row) & prefetch_mask;
            dma_queue_धकेलो(dma_q, dma_make_तथ्य(vtcm_src0_ptr + is0 * src0_stride, src0_row + (माप_प्रकार) ir0 * src0_row_आकार),
                           src0_stride, src0_row_आकार, src0_row_आकार, 1);
            स्थिर uint8_t * ss0 = (व्योम *) dma_queue_निकालो(dma_q).dst;

            htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, ir0);
            क्रम (uint32_t b = b_grp_start; b < b_grp_end; b++) {
                स्थिर uint32_t b_m_start = b * ne11;
                स्थिर uint32_t m_first = MAX(cur_m_start, b_m_start);
                स्थिर uint32_t m_last  = MIN(cur_m_start + cur_m_rows, b_m_start + ne11);
                अगर (m_first >= m_last) जारी;

                स्थिर uint32_t cur_b3 = fastdiv(b, &kparams->div_ne12);
                स्थिर uint32_t cur_b2 = b - cur_b3 * ne12;
                uint8_t * dst_batch_base = (uint8_t *) dst->तथ्य + (माप_प्रकार) cur_b2 * nb2 + (माप_प्रकार) cur_b3 * nb3;

                स्थिर uint32_t chunk_m_offset = m_first - cur_m_start;
                स्थिर uint32_t dst_m_offset   = m_first - b_m_start;
                स्थिर uint32_t batch_nrows    = m_last - m_first;

                क्रम (uint32_t ir1 = 0; ir1 < batch_nrows; ++ir1) {
                    स्थिर uint8_t * restrict act_col = (स्थिर uint8_t *) (act_तथ्य + (chunk_m_offset + ir1) * act_stride);
                    भग्न * restrict dst_row          = (भग्न *) (dst_batch_base + (dst_m_offset + ir1) * dst_row_आकार);
                    mmctx->vec_dot_1x1(ne00, &dst_row[ir0], ss0, act_col);
                }
            }
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ir0);
        }

        b_grp_start = b_grp_end;
    }

    अगर (src2) {
        hvx_tensor_add_f32_grid(dst, src2, cur_m_start, cur_m_start + cur_m_rows, src0_start_row, src0_end_row, &kparams->div_ne12_ne1, &kparams->div_ne1);
    }
}

#घोषणा MMID_MATRIX_ROW(row_id, i1) matrix_rows[(row_id) * mmctx->mapping_stride + (i1)]

अटल व्योम hvx_mm_id_tiled(अचिन्हित पूर्णांक nth, अचिन्हित पूर्णांक ith, व्योम * तथ्य) {
    htp_matmul_preamble;

    स्थिर काष्ठा htp_tensor * restrict ids = octx->src[2];

    स्थिर uint32_t src0_nrows      = mmctx->src0_row_end - mmctx->src0_row_start;  // src0 rows per expert
    स्थिर uint32_t act_nrows       = ne11;
    स्थिर uint32_t src0_start_row  = mmctx->src0_row_start + src0_nrows_per_thread * ith;
    स्थिर uint32_t src0_end_row    = MIN(src0_start_row + src0_nrows_per_thread, mmctx->src0_row_end);


    अगर (src0_start_row >= src0_end_row) {
        वापस;
    }

    काष्ठा htp_thread_trace * tr = &octx->ctx->trace[ith];

    स्थिर काष्ठा htp_mm_kernel_params * kparams = (स्थिर काष्ठा htp_mm_kernel_params *) octx->kernel_params;
    स्थिर uint32_t n_prefetch = kparams->n_prefetch;
    निश्चित(n_prefetch >= 2 && n_prefetch <= HTP_MM_MAX_PREFETCH && (n_prefetch & (n_prefetch - 1)) == 0);

    स्थिर uint32_t n_ids = ids->ne[0];  // n_expert_used
    स्थिर uint32_t n_as  = ne02;        // n_expert

    स्थिर uint32_t *                matrix_row_counts = mmctx->matrix_row_counts;
    स्थिर काष्ठा mmid_row_mapping * matrix_rows       = mmctx->matrix_rows;

    स्थिर माप_प्रकार dst_row_आकार  = nb1;
    स्थिर माप_प्रकार act_row_आकार  = htp_mm_q8_0_tiled_row_आकार(ne10);

    स्थिर माप_प्रकार act_stride = mmctx->vtcm_act_stride;

    // Per-thread VTCMs for all tensors
    uint8_t * restrict vtcm_src0_ptr = mmctx->vtcm_src0 + mmctx->vtcm_src0_आकार_per_thread * ith;
    uint8_t * restrict act_तथ्य = mmctx->vtcm_act;

    क्रम (uint32_t cur_a = 0; cur_a < n_as; ++cur_a) {
        स्थिर int32_t cne1 = matrix_row_counts[cur_a];
        अगर (cne1 == 0) {
            जारी;
        }

        स्थिर dma_addr_t src0_row = src0->तथ्य + cur_a * nb02;

        स्थिर uint32_t tile_आकार = htp_tensor_get_tile_आकार(src0->type);
        स्थिर uint32_t aligned_tile_आकार = htp_tensor_get_tile_padded_आकार(src0->type);
        स्थिर uint32_t n_k_tiles_w = ne00 / htp_tensor_get_tile_k(src0->type);
        स्थिर uint32_t tile_row_stride = n_k_tiles_w * tile_आकार;
        स्थिर uint32_t tile_row_transfer_आकार_aligned = n_k_tiles_w * aligned_tile_आकार;

        स्थिर uint32_t ct_start = src0_start_row / 32;
        स्थिर uint32_t ct_end   = (src0_end_row + 31) / 32;

        uint32_t धकेलो_ct = ct_start;
        क्रम (uint32_t d = 0; d < n_prefetch && धकेलो_ct < ct_end; d++, धकेलो_ct++) {
            dma_queue_धकेलो(dma_q, dma_make_तथ्य(vtcm_src0_ptr + d * tile_row_transfer_आकार_aligned, src0_row + धकेलो_ct * tile_row_stride),
                           aligned_tile_आकार, tile_आकार, tile_आकार, n_k_tiles_w);
        }

        क्रम (uint32_t ct = ct_start; ct < ct_end; ct++) {
            स्थिर uint8_t * w_tile = (व्योम *) dma_queue_निकालो(dma_q).dst;

            पूर्णांक valid_rows = (पूर्णांक)ne0 - (पूर्णांक)(ct * 32);
            valid_rows = MIN(32, MAX(0, valid_rows));

            htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, ct);
            क्रम (uint32_t cid = 0; cid < cne1; ++cid) {
                काष्ठा mmid_row_mapping row_mapping = MMID_MATRIX_ROW(cur_a, cid);
                स्थिर पूर्णांक               rm1         = row_mapping.i1;  // expert idx
                स्थिर पूर्णांक               rm2         = row_mapping.i2;  // token idx

                स्थिर uint32_t ir1 = fastmodulo(rm1, ne11, &mmctx->mm_div_ne11);        // act row idx
                स्थिर uint8_t * restrict act_col = (स्थिर uint8_t *) (act_तथ्य + (ir1 + rm2 * ne11 + 0) * act_stride);
                भग्न * restrict dst_row = (भग्न *) (dst->तथ्य + (rm1 * nb1 + rm2 * nb2 + 0));

                mmctx->vec_dot_32x1(ne10, &dst_row[ct * 32], w_tile, act_col, valid_rows, शून्य);
            }
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ct);

            अगर (धकेलो_ct < ct_end) {
                dma_queue_धकेलो(dma_q, dma_make_तथ्य(w_tile, src0_row + धकेलो_ct * tile_row_stride),
                               aligned_tile_आकार, tile_आकार, tile_आकार, n_k_tiles_w);
                धकेलो_ct++;
            }
        }
    }
}

अटल व्योम hvx_mv_id_tiled(अचिन्हित पूर्णांक nth, अचिन्हित पूर्णांक ith, व्योम * तथ्य) {
    htp_matmul_preamble;

    स्थिर काष्ठा htp_tensor * restrict ids = octx->src[2];

    स्थिर uint32_t src0_nrows      = mmctx->src0_row_end - mmctx->src0_row_start;  // src0 rows per expert
    स्थिर uint32_t src0_start_row  = mmctx->src0_row_start + src0_nrows_per_thread * ith;
    स्थिर uint32_t src0_end_row    = MIN(src0_start_row + src0_nrows_per_thread, mmctx->src0_row_end);


    अगर (src0_start_row >= src0_end_row) {
        वापस;
    }

    काष्ठा htp_thread_trace * tr = &octx->ctx->trace[ith];

    स्थिर काष्ठा htp_mm_kernel_params * kparams = (स्थिर काष्ठा htp_mm_kernel_params *) octx->kernel_params;
    स्थिर uint32_t n_prefetch = kparams->n_prefetch;
    निश्चित(n_prefetch >= 2 && n_prefetch <= HTP_MM_MAX_PREFETCH && (n_prefetch & (n_prefetch - 1)) == 0);

    निश्चित(ne13 % ne03 == 0);

    स्थिर माप_प्रकार dst_row_आकार  = nb1;
    स्थिर माप_प्रकार act_row_आकार = htp_mm_q8_0_tiled_row_आकार(ne10);

    स्थिर uint32_t n_aids = src2->ne[0];  // num activated experts
    स्थिर uint32_t n_ids  = ne02;         // num experts

    // Per-thread VTCMs for all tensors
    uint8_t * restrict vtcm_src0_ptr = mmctx->vtcm_src0 + mmctx->vtcm_src0_आकार_per_thread * ith;
    uint8_t * restrict act_तथ्य = mmctx->vtcm_act;

    क्रम (uint32_t ie1 = 0; ie1 < n_aids; ++ie1) {  // for each expert
        स्थिर int32_t eid = *(स्थिर int32_t *) ((स्थिर uint8_t *) src2->तथ्य + ie1 * src2->nb[0]);
        अगर (eid < 0) {
            जारी;
        }
        निश्चित(eid < (int32_t) n_ids);

        स्थिर dma_addr_t src0_row = src0->तथ्य + eid * nb02;
        स्थिर uint8_t * restrict act_col = (स्थिर uint8_t *) act_तथ्य;
        भग्न * restrict dst_row          = (भग्न *) (dst->तथ्य + ie1 * nb1);

        स्थिर uint32_t tile_आकार = htp_tensor_get_tile_आकार(src0->type);
        स्थिर uint32_t aligned_tile_आकार = htp_tensor_get_tile_padded_आकार(src0->type);
        स्थिर uint32_t n_k_tiles_w = ne00 / htp_tensor_get_tile_k(src0->type);
        स्थिर uint32_t tile_row_stride = n_k_tiles_w * tile_आकार;
        स्थिर uint32_t tile_row_transfer_आकार_aligned = n_k_tiles_w * aligned_tile_आकार;

        स्थिर uint32_t ct_start = src0_start_row / 32;
        स्थिर uint32_t ct_end   = (src0_end_row + 31) / 32;

        uint32_t धकेलो_ct = ct_start;
        क्रम (uint32_t d = 0; d < n_prefetch && धकेलो_ct < ct_end; d++, धकेलो_ct++) {
            dma_queue_धकेलो(dma_q, dma_make_तथ्य(vtcm_src0_ptr + d * tile_row_transfer_आकार_aligned, src0_row + धकेलो_ct * tile_row_stride),
                           aligned_tile_आकार, tile_आकार, tile_आकार, n_k_tiles_w);
        }

        क्रम (uint32_t ct = ct_start; ct < ct_end; ct++) {
            स्थिर uint8_t * w_tile = (व्योम *) dma_queue_निकालो(dma_q).dst;

            पूर्णांक valid_rows = (पूर्णांक)ne0 - (पूर्णांक)(ct * 32);
            valid_rows = MIN(32, MAX(0, valid_rows));

            htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, ct);
            mmctx->vec_dot_32x1(ne10, &dst_row[ct * 32], w_tile, act_col, valid_rows, शून्य);
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ct);

            अगर (धकेलो_ct < ct_end) {
                dma_queue_धकेलो(dma_q, dma_make_तथ्य(w_tile, src0_row + धकेलो_ct * tile_row_stride),
                               aligned_tile_आकार, tile_आकार, tile_आकार, n_k_tiles_w);
                धकेलो_ct++;
            }
        }
    }
}

अटल व्योम hvx_mv_id_nx_tiled(अचिन्हित पूर्णांक nth, अचिन्हित पूर्णांक ith, व्योम * तथ्य) {
    काष्ठा htp_mm_context * mmctx = (काष्ठा htp_mm_context *) तथ्य;
    काष्ठा htp_ops_context * octx = mmctx->octx;
    स्थिर काष्ठा htp_mm_kernel_params * kparams = (स्थिर काष्ठा htp_mm_kernel_params *) octx->kernel_params;
    स्थिर uint32_t n_weights      = kparams->n_weights;
    स्थिर काष्ठा htp_tensor * restrict src0 = octx->src[0];
    स्थिर काष्ठा htp_tensor * restrict act  = octx->src[n_weights];
    स्थिर काष्ठा htp_tensor * restrict ids  = octx->src[n_weights + 1];


    काष्ठा htp_thread_trace * tr = &octx->ctx->trace[ith];

    स्थिर uint32_t n_prefetch = kparams->n_prefetch;
    निश्चित(n_prefetch >= 2 && n_prefetch <= HTP_MM_MAX_PREFETCH && (n_prefetch & (n_prefetch - 1)) == 0);

    स्थिर uint32_t n_aids = ids->ne[0];
    स्थिर uint32_t n_ids  = src0->ne[2];

    uint8_t * restrict vtcm_src0_ptr = mmctx->vtcm_src0 + mmctx->vtcm_src0_आकार_per_thread * ith;
    uint8_t * restrict act_तथ्य = mmctx->vtcm_act;

    क्रम (uint32_t ie1 = 0; ie1 < n_aids; ++ie1) {
        स्थिर int32_t eid = *(स्थिर int32_t *) ((स्थिर uint8_t *) ids->तथ्य + ie1 * ids->nb[0]);
        अगर (eid < 0) जारी;
        निश्चित(eid < (int32_t) n_ids);

        क्रम (uint32_t p = 0; p < n_weights; ++p) {
            स्थिर काष्ठा htp_tensor * restrict src_w = octx->src[p];
            स्थिर काष्ठा htp_tensor * restrict dst   = octx->dsts[p];
            अगर (!src_w || !dst) जारी;
            dma_queue * dma_q = octx->ctx->dma[ith];

            स्थिर uint32_t ne01 = src_w->ne[1];
            uint32_t start_row = 0;
            uint32_t end_row   = ne01;
            अगर (octx->ctx->mdev.गिनती > 1) {
                स्थिर bool can_split = htp_tensor_can_row_partition(dst, माप(भग्न));
                स्थिर काष्ठा htp_tensor_mdev_range range = htp_tensor_mdev_partition(ne01, can_split ? 32 : 0, octx->ctx->mdev.idx, octx->ctx->mdev.गिनती, &octx->ctx->mdev.गिनती_div);
                start_row = range.start;
                end_row   = range.start + range.गिनती;
            }

            स्थिर uint32_t nrows = end_row - start_row;
            uint32_t src0_nrows_per_thread = fastdiv(nrows + nth - 1, &octx->n_threads_div);
            src0_nrows_per_thread = hex_round_up(src0_nrows_per_thread, 32);

            स्थिर uint32_t src0_start_row = start_row + src0_nrows_per_thread * ith;
            स्थिर uint32_t src0_end_row   = MIN(src0_start_row + src0_nrows_per_thread, end_row);
            अगर (src0_start_row >= src0_end_row) जारी;

            स्थिर dma_addr_t src0_row = src_w->तथ्य + eid * src_w->nb[2];
            स्थिर uint8_t * restrict act_col = (स्थिर uint8_t *) act_तथ्य;
            भग्न * restrict dst_row = (भग्न *) (dst->तथ्य + ie1 * dst->nb[1]);

            स्थिर uint32_t tile_आकार = htp_tensor_get_tile_आकार(src_w->type);
            स्थिर uint32_t aligned_tile_आकार = htp_tensor_get_tile_padded_आकार(src_w->type);
            स्थिर uint32_t n_k_tiles_w = src_w->ne[0] / htp_tensor_get_tile_k(src_w->type);
            स्थिर uint32_t tile_row_stride = n_k_tiles_w * tile_आकार;
            स्थिर uint32_t tile_row_transfer_आकार_aligned = n_k_tiles_w * aligned_tile_आकार;

            स्थिर uint32_t ct_start = src0_start_row / 32;
            स्थिर uint32_t ct_end   = (src0_end_row + 31) / 32;

            uint32_t धकेलो_ct = ct_start;
            क्रम (uint32_t d = 0; d < n_prefetch && धकेलो_ct < ct_end; d++, धकेलो_ct++) {
                dma_queue_धकेलो(dma_q, dma_make_तथ्य(vtcm_src0_ptr + d * tile_row_transfer_आकार_aligned, src0_row + धकेलो_ct * tile_row_stride),
                               aligned_tile_आकार, tile_आकार, tile_आकार, n_k_tiles_w);
            }

            क्रम (uint32_t ct = ct_start; ct < ct_end; ct++) {
                स्थिर uint8_t * w_tile = (व्योम *) dma_queue_निकालो(dma_q).dst;

                पूर्णांक valid_rows = (पूर्णांक)dst->ne[0] - (पूर्णांक)(ct * 32);
                valid_rows = MIN(32, MAX(0, valid_rows));

                htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, ct);
                mmctx->vec_dot_32x1(act->ne[0], &dst_row[ct * 32], w_tile, act_col, valid_rows, शून्य);
                htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ct);

                अगर (धकेलो_ct < ct_end) {
                    dma_queue_धकेलो(dma_q, dma_make_तथ्य(w_tile, src0_row + धकेलो_ct * tile_row_stride),
                                    aligned_tile_आकार, tile_आकार, tile_आकार, n_k_tiles_w);
                    धकेलो_ct++;
                }
            }
        }
    }
}

अटल व्योम hvx_mm_id_nx_tiled(अचिन्हित पूर्णांक nth, अचिन्हित पूर्णांक ith, व्योम * तथ्य) {
    काष्ठा htp_mm_context * mmctx = (काष्ठा htp_mm_context *) तथ्य;
    काष्ठा htp_ops_context * octx = mmctx->octx;
    स्थिर काष्ठा htp_mm_kernel_params * kparams = (स्थिर काष्ठा htp_mm_kernel_params *) octx->kernel_params;
    स्थिर uint32_t n_weights      = kparams->n_weights;
    स्थिर काष्ठा htp_tensor * restrict src0 = octx->src[0];
    स्थिर काष्ठा htp_tensor * restrict act  = octx->src[n_weights];
    स्थिर काष्ठा htp_tensor * restrict ids  = octx->src[n_weights + 1];


    काष्ठा htp_thread_trace * tr = &octx->ctx->trace[ith];

    स्थिर uint32_t n_prefetch = kparams->n_prefetch;
    निश्चित(n_prefetch >= 2 && n_prefetch <= HTP_MM_MAX_PREFETCH && (n_prefetch & (n_prefetch - 1)) == 0);

    स्थिर uint32_t n_as = src0->ne[2];

    स्थिर uint32_t * matrix_row_counts = mmctx->matrix_row_counts;
    स्थिर काष्ठा mmid_row_mapping * matrix_rows = mmctx->matrix_rows;

    स्थिर माप_प्रकार act_stride = mmctx->vtcm_act_stride;

    uint8_t * restrict vtcm_src0_ptr = mmctx->vtcm_src0 + mmctx->vtcm_src0_आकार_per_thread * ith;
    uint8_t * restrict act_तथ्य = mmctx->vtcm_act;

    क्रम (uint32_t cur_a = 0; cur_a < n_as; ++cur_a) {
        स्थिर int32_t cne1 = matrix_row_counts[cur_a];
        अगर (cne1 == 0) जारी;

        क्रम (uint32_t p = 0; p < n_weights; ++p) {
            स्थिर काष्ठा htp_tensor * restrict src_w = octx->src[p];
            स्थिर काष्ठा htp_tensor * restrict dst   = octx->dsts[p];
            अगर (!src_w || !dst) जारी;
            dma_queue * dma_q = octx->ctx->dma[ith];

            स्थिर uint32_t ne01 = src_w->ne[1];
            uint32_t start_row = 0;
            uint32_t end_row   = ne01;
            अगर (octx->ctx->mdev.गिनती > 1) {
                स्थिर bool can_split = htp_tensor_can_row_partition(dst, माप(भग्न));
                स्थिर काष्ठा htp_tensor_mdev_range range = htp_tensor_mdev_partition(ne01, can_split ? 32 : 0, octx->ctx->mdev.idx, octx->ctx->mdev.गिनती, &octx->ctx->mdev.गिनती_div);
                start_row = range.start;
                end_row   = range.start + range.गिनती;
            }

            स्थिर uint32_t nrows = end_row - start_row;
            uint32_t src0_nrows_per_thread = fastdiv(nrows + nth - 1, &octx->n_threads_div);
            src0_nrows_per_thread = hex_round_up(src0_nrows_per_thread, 32);

            स्थिर uint32_t src0_start_row = start_row + src0_nrows_per_thread * ith;
            स्थिर uint32_t src0_end_row   = MIN(src0_start_row + src0_nrows_per_thread, end_row);
            अगर (src0_start_row >= src0_end_row) जारी;

            स्थिर dma_addr_t src0_row = src_w->तथ्य + cur_a * src_w->nb[2];

            स्थिर uint32_t tile_आकार = htp_tensor_get_tile_आकार(src_w->type);
            स्थिर uint32_t aligned_tile_आकार = htp_tensor_get_tile_padded_आकार(src_w->type);
            स्थिर uint32_t n_k_tiles_w = src_w->ne[0] / htp_tensor_get_tile_k(src_w->type);
            स्थिर uint32_t tile_row_stride = n_k_tiles_w * tile_आकार;
            स्थिर uint32_t tile_row_transfer_आकार_aligned = n_k_tiles_w * aligned_tile_आकार;

            स्थिर uint32_t ct_start = src0_start_row / 32;
            स्थिर uint32_t ct_end   = (src0_end_row + 31) / 32;

            uint32_t धकेलो_ct = ct_start;
            क्रम (uint32_t d = 0; d < n_prefetch && धकेलो_ct < ct_end; d++, धकेलो_ct++) {
                dma_queue_धकेलो(dma_q, dma_make_तथ्य(vtcm_src0_ptr + d * tile_row_transfer_आकार_aligned, src0_row + धकेलो_ct * tile_row_stride),
                               aligned_tile_आकार, tile_आकार, tile_आकार, n_k_tiles_w);
            }

            क्रम (uint32_t ct = ct_start; ct < ct_end; ct++) {
                स्थिर uint8_t * w_tile = (व्योम *) dma_queue_निकालो(dma_q).dst;

                पूर्णांक valid_rows = (पूर्णांक)dst->ne[0] - (पूर्णांक)(ct * 32);
                valid_rows = MIN(32, MAX(0, valid_rows));

                htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, ct);
                क्रम (uint32_t cid = 0; cid < (uint32_t) cne1; ++cid) {
                    काष्ठा mmid_row_mapping row_mapping = MMID_MATRIX_ROW(cur_a, cid);
                    स्थिर पूर्णांक rm1 = row_mapping.i1;
                    स्थिर पूर्णांक rm2 = row_mapping.i2;

                    स्थिर uint32_t ir1 = fastmodulo(rm1, act->ne[1], &mmctx->mm_div_ne11);
                    स्थिर uint8_t * restrict act_col = (स्थिर uint8_t *) (act_तथ्य + (ir1 + rm2 * act->ne[1]) * act_stride);
                    भग्न * restrict dst_row = (भग्न *) (dst->तथ्य + (rm1 * dst->nb[1] + rm2 * dst->nb[2]));

                    mmctx->vec_dot_32x1(act->ne[0], &dst_row[ct * 32], w_tile, act_col, valid_rows, शून्य);
                }
                htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ct);

                अगर (धकेलो_ct < ct_end) {
                    dma_queue_धकेलो(dma_q, dma_make_तथ्य(w_tile, src0_row + धकेलो_ct * tile_row_stride),
                                    aligned_tile_आकार, tile_आकार, tile_आकार, n_k_tiles_w);
                    धकेलो_ct++;
                }
            }
        }
    }
}

अटल पूर्णांक hvx_mm_init_tiled_vec_dot(काष्ठा htp_mm_context * mmctx, क्रमागत htp_तथ्य_type type) {
    चयन (type) {
        हाल HTP_TYPE_Q4_0:
            mmctx->type         = "q4_0_tiled-f32";
            mmctx->vec_dot_32x1 = tiled_vec_dot_q4_0_32x1;
            mmctx->vec_dot_32x2 = tiled_vec_dot_q4_0_32x2;
            वापस 0;
        हाल HTP_TYPE_Q4_1:
        हाल HTP_TYPE_Q4_K:
            mmctx->type         = "q4_1_tiled-f32";
            mmctx->vec_dot_32x1 = tiled_vec_dot_q4_1_32x1;
            mmctx->vec_dot_32x2 = tiled_vec_dot_q4_1_32x2;
            वापस 0;
        हाल HTP_TYPE_Q8_0:
            mmctx->type         = "q8_0_tiled-f32";
            mmctx->vec_dot_32x1 = tiled_vec_dot_q8_0_32x1;
            mmctx->vec_dot_32x2 = tiled_vec_dot_q8_0_32x2;
            वापस 0;
        हाल HTP_TYPE_Q5_K:
            mmctx->type         = "q5_k_tiled-f32";
            mmctx->vec_dot_32x1 = tiled_vec_dot_q5_k_32x1;
            mmctx->vec_dot_32x2 = tiled_vec_dot_q5_k_32x2;
            वापस 0;
        हाल HTP_TYPE_Q6_K:
            mmctx->type         = "q6_k_tiled-f32";
            mmctx->vec_dot_32x1 = tiled_vec_dot_q6_k_32x1;
            mmctx->vec_dot_32x2 = tiled_vec_dot_q6_k_32x2;
            वापस 0;
        हाल HTP_TYPE_Q3_K:
            mmctx->type         = "q3_k_tiled-f32";
            mmctx->vec_dot_32x1 = tiled_vec_dot_q3_k_32x1;
            mmctx->vec_dot_32x2 = tiled_vec_dot_q3_k_32x2;
            वापस 0;
        हाल HTP_TYPE_Q2_K:
            mmctx->type         = "q2_k_tiled-f32";
            mmctx->vec_dot_32x1 = tiled_vec_dot_q2_k_32x1;
            mmctx->vec_dot_32x2 = tiled_vec_dot_q2_k_32x2;
            वापस 0;
        हाल HTP_TYPE_Q2_0:
            mmctx->type         = "q2_0_tiled-f32";
            mmctx->vec_dot_32x1 = tiled_vec_dot_q2_0_32x1;
            mmctx->vec_dot_32x2 = tiled_vec_dot_q2_0_32x2;
            वापस 0;
        हाल HTP_TYPE_IQ4_NL:
            mmctx->type         = "iq4nl_tiled-f32";
            mmctx->vec_dot_32x1 = tiled_vec_dot_iq4nl_32x1;
            mmctx->vec_dot_32x2 = tiled_vec_dot_iq4nl_32x2;
            वापस 0;
        हाल HTP_TYPE_MXFP4:
            mmctx->type         = "mxfp4_tiled-f32";
            mmctx->vec_dot_32x1 = tiled_vec_dot_mxfp4_32x1;
            mmctx->vec_dot_32x2 = tiled_vec_dot_mxfp4_32x2;
            वापस 0;
        हाल HTP_TYPE_Q1_0:
            mmctx->type         = "q1_0_tiled-f32";
            mmctx->vec_dot_32x1 = tiled_vec_dot_q1_0_32x1;
            mmctx->vec_dot_32x2 = tiled_vec_dot_q1_0_32x2;
            वापस 0;
        हाल HTP_TYPE_F16:
            mmctx->type         = "f16_tiled-f16";
            mmctx->vec_dot_32x1 = tiled_vec_dot_f16_32x1;
            mmctx->vec_dot_32x2 = tiled_vec_dot_f16_32x2;
            वापस 0;
        शेष:
            वापस -1;
    }
}

अटल पूर्णांक hvx_mm_matmul(काष्ठा htp_ops_context * octx) {
    htp_matmul_tensors_preamble;

    काष्ठा htp_thread_trace * tr = &octx->ctx->trace[0];
    htp_trace_event_start(tr, HTP_TRACE_EVT_INIT, 0);

    काष्ठा htp_mm_context mmctx_struct = {0};
    काष्ठा htp_mm_context * mmctx = &mmctx_struct;
    mmctx->octx = octx;
    mmctx->act = act;

    स्थिर काष्ठा htp_mm_kernel_params * kparams = (स्थिर काष्ठा htp_mm_kernel_params *) octx->kernel_params;

    स्थिर uint32_t src0_nrows = ne01;
    स्थिर uint32_t act_nrows  = ne11 * ne12 * ne13;
    mmctx->act_nrows = act_nrows;

    uint32_t src0_row_start = 0;
    uint32_t src0_row_end   = src0_nrows;

    अगर (octx->ctx->mdev.गिनती > 1) {
        स्थिर bool can_split = htp_tensor_can_row_partition(dst, माप(भग्न));
        स्थिर काष्ठा htp_tensor_mdev_range range = htp_tensor_mdev_partition(src0_nrows, can_split ? 32 : 0, octx->ctx->mdev.idx, octx->ctx->mdev.गिनती, &octx->ctx->mdev.गिनती_div);
        src0_row_start = range.start;
        src0_row_end   = range.start + range.गिनती;
    }

    अगर (src0_row_start >= src0_row_end) {
        वापस HTP_STATUS_OK;
    }

    स्थिर uint32_t nrows = src0_row_end - src0_row_start;
    mmctx->src0_row_start = src0_row_start;
    mmctx->src0_row_end   = src0_row_end;

    स्थिर bool is_tiled = htp_tensor_is_tiled(src0);

    // Compute src0_nrows_per_thread
    mmctx->src0_nrows_per_thread  = fastdiv(nrows + octx->n_threads - 1, &octx->n_threads_div);
    अगर (is_tiled) {
        mmctx->src0_nrows_per_thread = hex_round_up(mmctx->src0_nrows_per_thread, 32);
    } अन्यथा {
        mmctx->src0_nrows_per_thread += (mmctx->src0_nrows_per_thread & 1); // round up to even
    }

    स्थिर माप_प्रकार src0_row_आकार = nb01;
    स्थिर माप_प्रकार dst_row_आकार  = nb1;
    माप_प्रकार       act_row_आकार  = nb11;

    स्थिर माप_प्रकार src0_row_आकार_padded = hex_round_up(src0_row_आकार, 128);

    worker_callback_t quant_task_func;
    worker_callback_t matmul_job_func;
    uint32_t n_quant_tasks = 1;
    स्थिर bool is_batched = (ne12 > 1 || ne13 > 1 || ne02 > 1 || ne03 > 1);
    अगर (is_batched) {
        matmul_job_func = is_tiled ? hvx_mm_4d_tiled : hvx_mm_4d_flat;
    } अन्यथा अगर (act_nrows > 1) {
        matmul_job_func = is_tiled ? hvx_mm_2d_tiled : hvx_mm_2d_flat;
    } अन्यथा {
        matmul_job_func = is_tiled ? hvx_mv_2d_tiled : hvx_mv_2d_flat;
    }

    bool need_quant = true;

    चयन (kparams->kernel_type) {
        हाल HTP_MM_KERNEL_HVX_F16_F16_VTCM:
            quant_task_func        = (act->type == HTP_TYPE_F32) ? quantize_f32_f16 : quantize_f16_f16;
            need_quant             = (act->type == HTP_TYPE_F32);
            mmctx->type            = (act->type == HTP_TYPE_F32) ? "f32-f16" : "f16-f16";
            mmctx->vec_dot_1x1     = flat_vec_dot_f16_f16_aa_1x1;
            mmctx->vec_dot_2x1     = flat_vec_dot_f16_f16_aa_2x1;
            mmctx->vec_dot_2x2     = flat_vec_dot_f16_f16_aa_2x2;
            act_row_आकार           = kparams->act_row_आकार;
            अवरोध;

        हाल HTP_MM_KERNEL_HVX_F32_F32_VTCM:
            quant_task_func        = शून्य;
            need_quant             = false;
            mmctx->type            = "f32-f32";
            mmctx->vec_dot_1x1     = flat_vec_dot_f32_f32_aa_1x1;
            mmctx->vec_dot_2x1     = flat_vec_dot_f32_f32_aa_2x1;
            mmctx->vec_dot_2x2     = flat_vec_dot_f32_f32_aa_2x2;
            act_row_आकार           = kparams->act_row_आकार;
            अवरोध;

        हाल HTP_MM_KERNEL_HVX_BF16_F32_VTCM:
            quant_task_func        = शून्य;
            need_quant             = false;
            mmctx->type            = "bf16-f32";
            mmctx->vec_dot_1x1     = flat_vec_dot_bf16_f32_aa_1x1;
            mmctx->vec_dot_2x1     = flat_vec_dot_bf16_f32_aa_2x1;
            mmctx->vec_dot_2x2     = flat_vec_dot_bf16_f32_aa_2x2;
            act_row_आकार           = kparams->act_row_आकार;
            अवरोध;

        हाल HTP_MM_KERNEL_HVX_QUANT_BLOCK:
        हाल HTP_MM_KERNEL_HVX_QUANT_ROW:
        शेष: {
            अगर (hvx_mm_init_tiled_vec_dot(mmctx, src0->type) != 0) {
                वापस HTP_STATUS_NO_SUPPORT;
            }

            स्थिर uint32_t qk = QK_Q8_0_TILED;
            स्थिर uint32_t nb = (ne10 + qk - 1) / qk;
            स्थिर uint32_t योग_nb = act_nrows * nb;

            अगर (act_nrows < octx->n_threads && !is_batched) {
                n_quant_tasks = MIN(योग_nb, octx->n_threads);
                quant_task_func = htp_mm_act_quant_block_func(src0->type, act->type);
                क्रम (uint32_t ith = 0; ith < n_quant_tasks; ++ith) {
                    uint32_t ib_first = (योग_nb * ith) / n_quant_tasks;
                    uint32_t ib_last  = (योग_nb * (ith + 1)) / n_quant_tasks;
                    mmctx->quant_ib_first[ith] = ib_first;
                    mmctx->quant_ib_last[ith]  = ib_last;
                    mmctx->quant_r[ith]        = ib_first / nb;
                    mmctx->quant_c[ith]        = ib_first % nb;
                }
            } अन्यथा {
                n_quant_tasks = MIN(act_nrows, octx->n_threads);
                quant_task_func = htp_mm_act_quant_row_func(src0->type, act->type);
            }
            act_row_आकार = kparams->act_row_आकार;
            अवरोध;
        }
    }

    स्थिर uint32_t m_chunk = (kparams->m_chunk > 0 && (uint32_t) kparams->m_chunk < act_nrows)
                           ? (uint32_t) kparams->m_chunk : act_nrows;
    स्थिर uint32_t m_layout_rows = m_chunk;

    काष्ठा htp_mm_hvx_vtcm_layout L;
    htp_mm_hvx_vtcm_layout_build(&L, kparams->kernel_type, src0->type, ne10, m_layout_rows, octx->n_threads,
                                 dst_row_आकार, src0_row_आकार, act_row_आकार, src2 ? src2->nb[1] : 0, kparams->n_prefetch, false, false);

    अगर (kparams->kernel_type == HTP_MM_KERNEL_HVX_F16_F16_VTCM ||
        kparams->kernel_type == HTP_MM_KERNEL_HVX_F32_F32_VTCM ||
        kparams->kernel_type == HTP_MM_KERNEL_HVX_BF16_F32_VTCM ||
        kparams->kernel_type == HTP_MM_KERNEL_HVX_QUANT_ROW ||
        kparams->kernel_type == HTP_MM_KERNEL_HVX_QUANT_BLOCK) {
        mmctx->vtcm_act_आकार_per_thread = L.act_bytes;
    } अन्यथा {
        mmctx->vtcm_act_आकार_per_thread = fastdiv(L.act_bytes, &octx->n_threads_div);
    }

    mmctx->vtcm_src0_आकार_per_thread = fastdiv(L.src0_bytes, &octx->n_threads_div);
    mmctx->vtcm_dst_आकार_per_thread  = fastdiv(L.dst_bytes, &octx->n_threads_div);

    स्थिर माप_प्रकार vtcm_आकार = L.योग_bytes;

    FARF(ऊपरी_2, "matmul-%s : src0-vtcm-size %zu act-vtcm-size %zu dst-vtcm-size %zu (%zu)\n", mmctx->type,
         L.src0_bytes, L.act_bytes, L.dst_bytes, vtcm_आकार);

    FARF(ऊपरी_2, "matmul-%s : %ux%ux%ux%u * %ux%ux%ux%u-> %ux%ux%ux%u (0x%p, 0x%p, 0x%p)\n", mmctx->type, src0->ne[0],
         src0->ne[1], src0->ne[2], src0->ne[3], act->ne[0], act->ne[1], act->ne[2], act->ne[3], dst->ne[0],
         dst->ne[1], dst->ne[2], dst->ne[3], src0->तथ्य, act->तथ्य, dst->तथ्य);

    अगर (octx->ctx->vtcm_आकार < vtcm_आकार) {
        FARF(ERROR, "matmul-%s : current VTCM reservation %zu is too small, needed %zu\n", mmctx->type,
             octx->ctx->vtcm_आकार, vtcm_आकार);
        वापस HTP_STATUS_VTCM_TOO_SMALL;
    }

    uint8_t * स्थिर base = (uint8_t *) octx->ctx->vtcm_base;
    mmctx->vtcm_act      = VTCM_LAYOUT_PTR(uint8_t, base, L.off_act);
    mmctx->vtcm_src0     = VTCM_LAYOUT_PTR(uint8_t, base, L.off_src0);
    mmctx->vtcm_bias     = VTCM_LAYOUT_PTR(uint8_t, base, L.off_bias);
    mmctx->vtcm_dst      = VTCM_LAYOUT_PTR(uint8_t, base, L.off_dst);
    mmctx->vtcm_act_raw  = VTCM_LAYOUT_PTR(uint8_t, base, L.off_act_raw);

    mmctx->vtcm_src0_stride = src0_row_आकार_padded;
    mmctx->vtcm_act_stride  = act_row_आकार;
    अगर (kparams->kernel_type == HTP_MM_KERNEL_HVX_QUANT_BLOCK || kparams->kernel_type == HTP_MM_KERNEL_HVX_QUANT_ROW) {
        mmctx->vtcm_act_raw_stride = hex_round_up(ne10 * माप(भग्न), QK_Q8_0_TILED * माप(भग्न));
    } अन्यथा अगर (kparams->kernel_type == HTP_MM_KERNEL_HVX_F16_F16_VTCM) {
        mmctx->vtcm_act_raw_stride = hex_round_up(ne10 * माप(भग्न), 128);
    } अन्यथा {
        mmctx->vtcm_act_raw_stride = 0;
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_INIT, 0);

    अगर (kparams->m_chunk > 0 && (uint32_t) kparams->m_chunk < act_nrows) {
        क्रम (uint32_t m_start = 0; m_start < act_nrows; m_start += m_chunk) {
            स्थिर uint32_t cur_m_rows = MIN(act_nrows - m_start, m_chunk);
            mmctx->cur_m_start = m_start;
            mmctx->cur_m_rows  = cur_m_rows;

            अगर (need_quant) {
                hvx_mm_transfer_act_dma(octx, kparams, act, mmctx->vtcm_act_raw, mmctx->vtcm_act_raw_stride, m_start, cur_m_rows);

                स्थिर uint32_t qk = QK_Q8_0_TILED;
                स्थिर uint32_t nb = (ne10 + qk - 1) / qk;
                स्थिर uint32_t योग_nb = cur_m_rows * nb;
                uint32_t quant_tasks;
                work_queue_func_t q_func;
                अगर (cur_m_rows < octx->n_threads && (kparams->kernel_type == HTP_MM_KERNEL_HVX_QUANT_BLOCK || kparams->kernel_type == HTP_MM_KERNEL_HVX_QUANT_ROW)) {
                    quant_tasks = MIN(योग_nb, octx->n_threads);
                    q_func = htp_mm_act_quant_block_func(src0->type, act->type);
                    क्रम (uint32_t ith = 0; ith < quant_tasks; ++ith) {
                        uint32_t ib_first = (योग_nb * ith) / quant_tasks;
                        uint32_t ib_last  = (योग_nb * (ith + 1)) / quant_tasks;
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

        अगर (need_quant) {
            hvx_mm_transfer_act_dma(octx, kparams, act, mmctx->vtcm_act_raw, mmctx->vtcm_act_raw_stride, 0, act_nrows);
            mmctx->n_quant_rows_per_thread = (act_nrows + n_quant_tasks - 1) / n_quant_tasks;
            mmctx->n_quant_tasks = n_quant_tasks;
            work_queue_run(octx->ctx->work_queue, quant_task_func, mmctx, n_quant_tasks);
        } अन्यथा {
            hvx_mm_transfer_act_dma(octx, kparams, act, mmctx->vtcm_act, mmctx->vtcm_act_stride, 0, act_nrows);
        }

        work_queue_run(octx->ctx->work_queue, matmul_job_func, mmctx, octx->n_threads);
    }

    वापस HTP_STATUS_OK;
}

अटल व्योम dequantize_tiled_worker_loop(अचिन्हित पूर्णांक n, अचिन्हित पूर्णांक i, व्योम * तथ्य) {
    tiled_dequantize_state_t * state = (tiled_dequantize_state_t *) तथ्य;
    काष्ठा htp_thread_trace * tr = &state->traces[i];
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_W_DEQUANT, i);
    क्रम (अचिन्हित पूर्णांक task_id = i; task_id < (अचिन्हित पूर्णांक) state->n_tasks; task_id += n) {
        पूर्णांक start = task_id * state->n_tiles_per_task;
        पूर्णांक end   = hex_smin(start + state->n_tiles_per_task, state->n_tot_tiles);
        state->task_fn(state, start, end);
    }
    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_W_DEQUANT, i);
}

अटल व्योम transfer_output_chunk_worker_fn(अचिन्हित पूर्णांक n, अचिन्हित पूर्णांक i, व्योम *तथ्य) {
    output_transfer_task_state_t *st = (output_transfer_task_state_t *) तथ्य;

    काष्ठा htp_thread_trace * tr = &st->traces[i];

    पूर्णांक start_chunk_idx = i * st->n_chunks_per_task;
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_O_PROC, start_chunk_idx);

    क्रम (अचिन्हित पूर्णांक task_id = i; task_id < (अचिन्हित पूर्णांक)st->n_tasks; task_id += n) {
        पूर्णांक    chunk_idx  = task_id * st->n_chunks_per_task;
        माप_प्रकार chunk_आकार = hex_smin(st->n_tot_chunks - chunk_idx, st->n_chunks_per_task);

        भग्न        *dst      = st->dst      + chunk_idx * st->dst_stride;
        स्थिर भग्न  *src2     = st->src2     ? (st->src2     + chunk_idx * st->src2_stride) : शून्य;
        transfer_output_chunk_fp16_to_fp32(dst, src2, st->vtcm_src, chunk_idx, chunk_आकार, st->n_cols, st->dst_stride, st->src2_stride, st->dst_cols);
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_O_PROC, start_chunk_idx);
}

प्रकार काष्ठा {
    काष्ठा htp_context            * ctx;
    काष्ठा htp_thread_trace       * traces;
    __fp16                        * dst;
    dma_addr_t                      act_dma_addr;
    भग्न                         * vtcm_f32_act;
    uint32_t                        n_tasks;
    uint32_t                        n_tot_chunks;
    uint32_t                        n_chunks_per_task;
    uint32_t                        k_block;
    uint32_t                        k_stride;
    uint32_t                        k_valid;
    माप_प्रकार                          vtcm_f32_act_bytes_per_thread;
    uint32_t                        dma_step_rows;
    uint32_t                        dma_step_rows_shift;
    uint32_t                        act_elem_आकार;  // 4=F32 src1, 2=F16 src1
} activation_transfer_task_state_t;

प्रकार काष्ठा {
    काष्ठा htp_context            * ctx;
    काष्ठा htp_thread_trace       * traces;
    __fp16                        * dst;
    dma_addr_t                      act_dma_addr;
    भग्न                         * vtcm_f32_act;
    uint32_t                        n_rows;
    uint32_t                        k_block;
    uint32_t                        k_stride;
    uint32_t                        k_valid;
    uint32_t                        n_col_chunks;
    काष्ठा fastdiv_मूल्यावली           n_threads_div;
    माप_प्रकार                          vtcm_f32_act_bytes;
    uint32_t                        dma_step_rows;
    uint32_t                        dma_step_rows_shift;
} activation_transfer_col_chunk_state_t;

अटल व्योम transfer_activation_chunk_fp32_to_fp16_dma_pipelined_col_chunk(
        dma_queue *dma_q,
        __fp16 *restrict vtcm_dst,
        dma_addr_t act_dma_addr,
        uint32_t n_rows,
        uint32_t k_block,
        uint32_t k_stride,
        uint32_t k_chunk_valid,
        uint32_t c_first,
        uint32_t c_len,
        भग्न *thread_f32_act,
        काष्ठा htp_thread_trace *tr,
        uint32_t dma_step_rows,
        uint32_t dma_step_rows_shift) {

    स्थिर uint32_t R = dma_step_rows;
    स्थिर uint32_t n_rows_padded = hex_align_up(n_rows, HTP_MM_HMX_TILE_N_ROWS);

    स्थिर uint32_t n_steps = n_rows_padded >> dma_step_rows_shift;

    // Push step 0
    अगर (n_steps > 0 && n_rows > 0) {
        uint32_t nrows_to_fetch = hex_smin(n_rows, R);
        dma_queue_धकेलो(dma_q, dma_make_तथ्य(thread_f32_act, act_dma_addr + (माप_प्रकार) c_first * माप(भग्न)),
                       c_len * माप(भग्न), k_stride * माप(भग्न), k_chunk_valid * माप(भग्न), nrows_to_fetch);
    }
    // Push step 1
    अगर (n_steps > 1) {
        uint32_t अग्र_r = R * 1;
        अगर (अग्र_r < n_rows) {
            uint32_t nrows_to_fetch = hex_smin(n_rows - अग्र_r, R);
            भग्न *अग्र_buf = thread_f32_act + 1 * R * c_len;
            dma_queue_धकेलो(dma_q, dma_make_तथ्य(अग्र_buf, act_dma_addr + ((माप_प्रकार) अग्र_r * k_stride + c_first) * माप(भग्न)),
                           c_len * माप(भग्न), k_stride * माप(भग्न), k_chunk_valid * माप(भग्न), nrows_to_fetch);
        }
    }
    क्रम (uint32_t s = 0; s < n_steps; ++s) {
        uint32_t r = s << dma_step_rows_shift;
        भग्न *curr_buf = thread_f32_act;

        अगर (r < n_rows) {
            curr_buf = (भग्न *) dma_queue_निकालो(dma_q).dst;
        }

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_A_PREP, r);
        क्रम (uint32_t p = 0; p < (R >> 1); ++p) {
            uint32_t row_idx = r + (p << 1);
            भग्न *pair_buf = curr_buf + (p << 1) * c_len;
            bool r0_valid = ((row_idx + 0) < n_rows);
            bool r1_valid = ((row_idx + 1) < n_rows);

            transfer_activation_row_pair_fp32_to_fp16_col_chunk(
                vtcm_dst, pair_buf, pair_buf + c_len, row_idx, k_block, c_first, c_len, k_chunk_valid, r0_valid, r1_valid
            );
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_A_PREP, r);

        // Push step s + 2
        uint32_t अग्र_s = s + 2;
        uint32_t अग्र_r = अग्र_s << dma_step_rows_shift;
        अगर (अग्र_r < n_rows) {
            uint32_t nrows_to_fetch = hex_smin(n_rows - अग्र_r, R);
            dma_queue_धकेलो(dma_q, dma_make_तथ्य(curr_buf, act_dma_addr + ((माप_प्रकार) अग्र_r * k_stride + c_first) * माप(भग्न)),
                           c_len * माप(भग्न), k_stride * माप(भग्न), k_chunk_valid * माप(भग्न), nrows_to_fetch);
        }
    }
}

अटल व्योम transfer_activation_chunk_col_chunk_worker_fn(अचिन्हित पूर्णांक n, अचिन्हित पूर्णांक i, व्योम *तथ्य) {
    activation_transfer_col_chunk_state_t *st = (activation_transfer_col_chunk_state_t *) तथ्य;
    काष्ठा htp_thread_trace * tr = &st->traces[i];

    uint32_t n_blocks = st->k_block / 32;
    uint32_t b_first = fastdiv(n_blocks * i, &st->n_threads_div);
    uint32_t b_last  = fastdiv(n_blocks * (i + 1), &st->n_threads_div);
    uint32_t c_first = b_first * 32;
    uint32_t c_last = b_last * 32;
    uint32_t c_len = c_last - c_first;

    अगर (c_len == 0) {
        वापस;
    }

    uint32_t k_chunk_valid = 0;
    अगर (st->k_valid > c_first) {
        k_chunk_valid = hex_smin(st->k_valid, c_last) - c_first;
    }

    __fp16 *dst = st->dst;

    माप_प्रकार thread_scratch_bytes = hex_align_down(fastdiv(st->vtcm_f32_act_bytes, &st->n_threads_div), 128);
    भग्न *thread_f32_act = (भग्न *)((अक्षर *)st->vtcm_f32_act + i * thread_scratch_bytes);

    transfer_activation_chunk_fp32_to_fp16_dma_pipelined_col_chunk(
        st->ctx->dma[i], dst, st->act_dma_addr, st->n_rows, st->k_block, st->k_stride, k_chunk_valid,
        c_first, c_len, thread_f32_act, tr, st->dma_step_rows, st->dma_step_rows_shift
    );
}

अटल व्योम transfer_activation_chunk_to_fp16_dma_pipelined(
        dma_queue *dma_q,
        __fp16 *restrict vtcm_dst,
        dma_addr_t act_dma_addr,
        uint32_t n_rows,
        uint32_t k_block,
        uint32_t k_stride,
        uint32_t k_valid,
        uint8_t *thread_act,
        uint32_t act_elem_आकार,
        काष्ठा htp_thread_trace *tr,
        uint32_t dma_step_rows,
        uint32_t dma_step_rows_shift) {

    स्थिर uint32_t R = dma_step_rows;
    स्थिर uint32_t n_rows_padded = hex_align_up(n_rows, HTP_MM_HMX_TILE_N_ROWS);

    स्थिर uint32_t n_steps = n_rows_padded >> dma_step_rows_shift;

    स्थिर bool act_is_f16 = (act_elem_आकार == माप(__fp16));
    स्थिर माप_प्रकार row_bytes = (माप_प्रकार) k_block * act_elem_आकार;  // staging row (DMA dst stride)
    स्थिर माप_प्रकार src_row_bytes = (माप_प्रकार) k_stride * act_elem_आकार;  // DDR row (DMA src stride)
    स्थिर माप_प्रकार width_bytes = (माप_प्रकार) k_valid * act_elem_आकार;

    // Push step 0
    अगर (n_steps > 0 && n_rows > 0) {
        uint32_t nrows_to_fetch = hex_smin(n_rows, R);
        dma_queue_धकेलो(dma_q, dma_make_तथ्य(thread_act, act_dma_addr),
                       row_bytes, src_row_bytes, width_bytes, nrows_to_fetch);
    }
    // Push step 1 (if valid)
    अगर (n_steps > 1) {
        uint32_t अग्र_r = R * 1;
        अगर (अग्र_r < n_rows) {
            uint32_t nrows_to_fetch = hex_smin(n_rows - अग्र_r, R);
            uint8_t *अग्र_buf = thread_act + 1 * R * row_bytes;
            dma_queue_धकेलो(dma_q, dma_make_तथ्य(अग्र_buf, act_dma_addr + (माप_प्रकार) अग्र_r * src_row_bytes),
                           row_bytes, src_row_bytes, width_bytes, nrows_to_fetch);
        }
    }
    क्रम (uint32_t s = 0; s < n_steps; ++s) {
        uint32_t r = s << dma_step_rows_shift;
        uint8_t *curr_buf = thread_act;

        अगर (r < n_rows) {
            curr_buf = (uint8_t *) dma_queue_निकालो(dma_q).dst;
        }

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_A_PREP, r);
        // Two copies of the pair loop so the type is resolved once per step and the
        // row-pair kernels stay direct (inlinable) calls.
        अगर (act_is_f16) {
            क्रम (uint32_t p = 0; p < (R >> 1); ++p) {
                uint32_t row_idx = r + (p << 1);
                स्थिर __fp16 *pair_buf = (स्थिर __fp16 *) (curr_buf + (p << 1) * row_bytes);
                bool r0_valid = ((row_idx + 0) < n_rows);
                bool r1_valid = ((row_idx + 1) < n_rows);

                transfer_activation_row_pair_f16_to_f16(vtcm_dst, pair_buf, pair_buf + k_block, row_idx, k_block, k_valid, r0_valid, r1_valid);
            }
        } अन्यथा {
            क्रम (uint32_t p = 0; p < (R >> 1); ++p) {
                uint32_t row_idx = r + (p << 1);
                स्थिर भग्न *pair_buf = (स्थिर भग्न *) (curr_buf + (p << 1) * row_bytes);
                bool r0_valid = ((row_idx + 0) < n_rows);
                bool r1_valid = ((row_idx + 1) < n_rows);

                transfer_activation_row_pair_fp32_to_fp16(vtcm_dst, pair_buf, pair_buf + k_block, row_idx, k_block, k_valid, r0_valid, r1_valid);
            }
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_A_PREP, r);

        // Push step s + 2
        uint32_t अग्र_s = s + 2;
        uint32_t अग्र_r = अग्र_s << dma_step_rows_shift;
        अगर (अग्र_r < n_rows) {
            uint32_t nrows_to_fetch = hex_smin(n_rows - अग्र_r, R);
            dma_queue_धकेलो(dma_q, dma_make_तथ्य(curr_buf, act_dma_addr + (माप_प्रकार) अग्र_r * src_row_bytes),
                           row_bytes, src_row_bytes, width_bytes, nrows_to_fetch);
        }
    }
}

अटल व्योम transfer_activation_chunk_worker_fn(अचिन्हित पूर्णांक n, अचिन्हित पूर्णांक i, व्योम *तथ्य) {
    activation_transfer_task_state_t *st = (activation_transfer_task_state_t *) तथ्य;

    काष्ठा htp_thread_trace * tr = &st->traces[i];

    क्रम (अचिन्हित पूर्णांक task_id = i; task_id < (अचिन्हित पूर्णांक)st->n_tasks; task_id += n) {
        पूर्णांक    chunk_idx  = task_id * st->n_chunks_per_task;
        माप_प्रकार chunk_आकार = hex_smin(st->n_tot_chunks - chunk_idx, st->n_chunks_per_task);

        __fp16      *dst = st->dst + chunk_idx * st->k_block;
        स्थिर dma_addr_t act_dma_addr = st->act_dma_addr + (माप_प्रकार) chunk_idx * st->k_stride * st->act_elem_आकार;

        uint8_t *thread_act = (uint8_t *) st->vtcm_f32_act + i * st->vtcm_f32_act_bytes_per_thread;
        transfer_activation_chunk_to_fp16_dma_pipelined(
            st->ctx->dma[i], dst, act_dma_addr, chunk_आकार, st->k_block, st->k_stride, st->k_valid, thread_act, st->act_elem_आकार, tr, st->dma_step_rows, st->dma_step_rows_shift
        );
    }
}

प्रकार काष्ठा {
    काष्ठा htp_thread_trace       * traces;
    स्थिर काष्ठा mmid_row_mapping * matrix_rows;
    __fp16                        * dst;
    स्थिर भग्न                   * src;
    uint32_t                        n_tasks;
    uint32_t                        n_tot_chunks;
    uint32_t                        n_chunks_per_task;
    uint32_t                        k_block;
    uint32_t                        cur_a;
    uint32_t                        mapping_stride;
    uint32_t                        ne11;
    काष्ठा fastdiv_मूल्यावली           ne11_div;
    माप_प्रकार                          nb11;
    माप_प्रकार                          nb12;
    uint32_t                        start_row;
    uint32_t                        cne1;
    uint32_t                        k_valid;
} activation_transfer_gathered_task_state_t;

प्रकार काष्ठा {
    काष्ठा htp_thread_trace       * traces;
    स्थिर काष्ठा mmid_row_mapping * matrix_rows;
    स्थिर __fp16                  * vtcm_src;
    भग्न                         * dst;
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

अटल व्योम transfer_activation_chunk_gathered_worker_fn(अचिन्हित पूर्णांक n, अचिन्हित पूर्णांक i, व्योम *तथ्य) {
    activation_transfer_gathered_task_state_t *st = तथ्य;
    काष्ठा htp_thread_trace * tr = &st->traces[i];
    पूर्णांक chunk_idx      = i;
    पूर्णांक chunk_आकार     = st->n_chunks_per_task;
    पूर्णांक vtcm_start_row = chunk_idx * chunk_आकार;
    पूर्णांक start_row      = st->start_row + vtcm_start_row;
    पूर्णांक n_rows         = hex_smin(st->cne1 - start_row, chunk_आकार);
    अगर (n_rows > 0) {
        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_A_PREP, chunk_idx);
        transfer_activation_chunk_fp32_to_fp16_gathered(
            st->dst, st->src, start_row, vtcm_start_row, n_rows, st->k_block,
            st->matrix_rows, st->cur_a, st->mapping_stride,
            st->ne11, &st->ne11_div, st->nb11, st->nb12, st->cne1, st->k_valid);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_A_PREP, chunk_idx);
    }
}

अटल व्योम transfer_activation_chunk_gathered_worker_flat_fn(अचिन्हित पूर्णांक n, अचिन्हित पूर्णांक i, व्योम *तथ्य) {
    activation_transfer_gathered_task_state_t *st = तथ्य;
    काष्ठा htp_thread_trace * tr = &st->traces[i];
    पूर्णांक chunk_idx = i;
    पूर्णांक chunk_आकार = st->n_chunks_per_task;
    पूर्णांक vtcm_start_row = chunk_idx * chunk_आकार;
    पूर्णांक start_row = st->start_row + vtcm_start_row;
    पूर्णांक n_rows = hex_smin(st->cne1 - start_row, chunk_आकार);
    अगर (n_rows > 0) {
        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_A_PREP, chunk_idx);
        transfer_activation_chunk_fp32_to_fp16_gathered_flat(
            st->dst, st->src, start_row, vtcm_start_row, n_rows, st->k_block,
            st->matrix_rows, st->cur_a, st->mapping_stride,
            st->nb12, st->cne1, st->k_valid);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_A_PREP, chunk_idx);
    }
}

अटल व्योम transfer_output_chunk_scattered_worker_fn(अचिन्हित पूर्णांक n, अचिन्हित पूर्णांक i, व्योम *तथ्य) {
    output_transfer_scattered_task_state_t *st = तथ्य;
    काष्ठा htp_thread_trace * tr = &st->traces[i];
    पूर्णांक chunk_idx = i;
    पूर्णांक chunk_आकार = st->n_chunks_per_task;
    पूर्णांक vtcm_start_row = chunk_idx * chunk_आकार;
    पूर्णांक start_row = st->start_row + vtcm_start_row;
    पूर्णांक n_rows = hex_smin(st->cne1 - start_row, chunk_आकार);
    अगर (n_rows > 0) {
        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_O_PROC, chunk_idx);
        transfer_output_chunk_fp16_to_fp32_scattered(
            st->dst, st->vtcm_src, start_row, vtcm_start_row, n_rows, st->n_cols,
            st->matrix_rows, st->cur_a, st->mapping_stride,
            st->dst_nb1, st->dst_nb2, st->cne1);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_O_PROC, chunk_idx);
    }
}

// --- HMX Dispatchers & Entry Points ---

अटल अंतरभूत tiled_dequant_task_fn_t hmx_get_dequant_worker_fn(पूर्णांक weight_type, bool is_tiled) {
    चयन (weight_type) {
        हाल HTP_TYPE_Q4_0:   वापस dequantize_tiled_weight_to_fp16_task_q4_0;
        हाल HTP_TYPE_IQ4_NL: वापस dequantize_tiled_weight_to_fp16_task_iq4_nl;
        हाल HTP_TYPE_Q4_1:
        हाल HTP_TYPE_Q4_K:   वापस dequantize_tiled_weight_to_fp16_task_q4_1;
        हाल HTP_TYPE_MXFP4:  वापस dequantize_tiled_weight_to_fp16_task_mxfp4;
        हाल HTP_TYPE_Q8_0:   वापस dequantize_tiled_weight_to_fp16_task_q8_0;
        हाल HTP_TYPE_Q5_K:   वापस dequantize_tiled_weight_to_fp16_task_q5_k;
        हाल HTP_TYPE_Q6_K:   वापस dequantize_tiled_weight_to_fp16_task_q6_k;
        हाल HTP_TYPE_Q3_K:   वापस dequantize_tiled_weight_to_fp16_task_q3_k;
        हाल HTP_TYPE_Q2_K:   वापस dequantize_tiled_weight_to_fp16_task_q2_k;
        हाल HTP_TYPE_Q1_0:   वापस dequantize_tiled_weight_to_fp16_task_q1_0;
        हाल HTP_TYPE_Q2_0:   वापस dequantize_tiled_weight_to_fp16_task_q2_0;
        हाल HTP_TYPE_F16:    वापस is_tiled ? शून्य : convert_f16_weight_to_fp16_tiles_task;
        हाल HTP_TYPE_BF16:   वापस convert_bf16_weight_to_fp16_tiles_task;
        हाल HTP_TYPE_F32:    वापस quantize_f32_weight_to_fp16_tiles_task;
        शेष:              वापस शून्य;
    }
}

अटल व्योम dequantize_tiled_weight_chunk_to_fp16_tiles(
        काष्ठा htp_context *ctx, __fp16 *vtcm_dst,
        स्थिर व्योम *weight_src_ddr,
        पूर्णांक n_cols, पूर्णांक k_block,
        माप_प्रकार row_stride, पूर्णांक weight_type,
        पूर्णांक n_k_tiles, काष्ठा fastdiv_मूल्यावली n_k_tiles_div,
        tiled_dequant_task_fn_t dequant_worker_fn, पूर्णांक n_threads) {

    अगर (!dequant_worker_fn) {
        वापस;
    }

    निश्चित(k_block % HTP_MM_HMX_TILE_N_COLS == 0);

    स्थिर माप_प्रकार n_col_tiles = hmx_ceil_div(n_cols, HTP_MM_HMX_TILE_N_COLS);
    स्थिर uint32_t n_weight_tiles = k_block / htp_tensor_get_tile_k(weight_type);
    स्थिर माप_प्रकार n_tot_tiles = n_col_tiles * n_weight_tiles;

    माप_प्रकार n_tiles_per_task = (n_threads == 1) ? n_tot_tiles : hmx_ceil_div(n_tot_tiles, n_threads);

    tiled_dequantize_state_t state;
    state.n_tasks          = (n_tot_tiles + n_tiles_per_task - 1) / n_tiles_per_task;
    state.n_tot_tiles      = n_tot_tiles;
    state.n_tiles_per_task = n_tiles_per_task;
    state.dst              = vtcm_dst;
    state.src              = (स्थिर uint8_t *)weight_src_ddr;
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

    state.tile_आकार = htp_tensor_get_tile_आकार(weight_type);
    state.aligned_tile_आकार = htp_tensor_get_tile_padded_आकार(weight_type);

    अगर (state.n_tasks == 1 || n_threads == 1) {
        dequantize_tiled_worker_loop(1, 0, &state);
    } अन्यथा {
        पूर्णांक n_tasks = hex_smin((पूर्णांक) state.n_tasks, n_threads);
        worker_pool_run_func(ctx->worker_pool, dequantize_tiled_worker_loop, &state, n_tasks);
    }
}

प्रकार काष्ठा {
    काष्ठा htp_context      * ctx;
    काष्ठा htp_thread_trace * traces;
    भग्न                   * dst;
    स्थिर __fp16            * vtcm_src;
    स्थिर भग्न             * src2;
    uint32_t                  n_rows;
    uint32_t                  n_cols;
    uint32_t                  dst_stride;
    uint32_t                  src2_stride;
    uint32_t                  dst_cols;
    काष्ठा fastdiv_मूल्यावली     n_threads_div;
} output_transfer_col_chunk_state_t;

अटल व्योम transfer_output_chunk_col_chunk_worker_fn(अचिन्हित पूर्णांक n, अचिन्हित पूर्णांक i, व्योम *तथ्य) {
    (व्योम) n;
    output_transfer_col_chunk_state_t *st = (output_transfer_col_chunk_state_t *) तथ्य;
    काष्ठा htp_thread_trace * tr = &st->traces[i];

    // Round up: the last block is partial when N is not 32-aligned. Its pad
    // columns are dropped by the dst_cols clamp inside the store.
    uint32_t n_blocks = hmx_ceil_div(st->n_cols, 32);
    uint32_t b_first  = fastdiv(n_blocks * i, &st->n_threads_div);
    uint32_t b_last   = fastdiv(n_blocks * (i + 1), &st->n_threads_div);
    uint32_t c_first  = b_first * 32;
    uint32_t c_last   = b_last * 32;
    uint32_t c_len    = c_last - c_first;

    अगर (c_len == 0) वापस;

    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_O_PROC, c_first);

    स्थिर __fp16 *vtcm_src = st->vtcm_src + b_first * HTP_MM_HMX_TILE_N_ELMS;
    स्थिर भग्न *src2      = st->src2 ? (st->src2 + c_first) : शून्य;
    भग्न *dst             = st->dst + c_first;

    पूर्णांक chunk_dst_cols = (पूर्णांक)st->dst_cols - (पूर्णांक)c_first;
    अगर (chunk_dst_cols > 0) {
        transfer_output_chunk_fp16_to_fp32_col_chunk(
            dst, src2, vtcm_src, 0, st->n_rows, c_len, st->n_cols,
            st->dst_stride, st->src2_stride, (uint32_t)chunk_dst_cols
        );
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_O_PROC, c_first);
}

अटल व्योम transfer_output_chunk_threaded(काष्ठा htp_context *ctx, भग्न *dst, स्थिर भग्न *src2, स्थिर __fp16 *vtcm_src,
                                              पूर्णांक n_rows, पूर्णांक n_cols, पूर्णांक dst_stride, uint32_t src2_stride, पूर्णांक dst_cols, पूर्णांक n_threads) {
    अगर (n_rows <= 0) वापस;

    uint32_t n_blocks = hmx_ceil_div((uint32_t) n_cols, 32);
    अगर (n_threads > 1 && n_blocks >= (uint32_t)n_threads) {
        काष्ठा fastdiv_मूल्यावली n_threads_div = (n_threads == (पूर्णांक)ctx->n_threads) ? ctx->n_threads_div : init_fastdiv_मूल्यावली(n_threads);
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
        वापस;
    }

    माप_प्रकार n_tot_chunks      = n_rows;
    माप_प्रकार n_chunks_per_task = (n_threads == 1) ? n_tot_chunks : hmx_ceil_div(n_rows, n_threads);
    n_chunks_per_task        = hex_align_up(n_chunks_per_task, 2);

    पूर्णांक actual_threads = hmx_ceil_div(n_rows, n_chunks_per_task);

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

    अगर (actual_threads <= 1) {
        transfer_output_chunk_worker_fn(1, 0, &state);
    } अन्यथा {
        worker_pool_run_func(ctx->worker_pool, transfer_output_chunk_worker_fn, &state, actual_threads);
    }
}

काष्ठा activation_transfer_params {
    काष्ठा htp_context *          ctx;
    __fp16 *                      dst;
    dma_addr_t                    act_dma_addr;
    पूर्णांक                           n_rows;
    पूर्णांक                           k_block;
    पूर्णांक                           k_stride;
    पूर्णांक                           n_threads;
    स्थिर काष्ठा fastdiv_मूल्यावली * act_threads_div;
    स्थिर काष्ठा fastdiv_मूल्यावली * k_div;
    पूर्णांक                           k_valid;
    भग्न *                       vtcm_f32_act;
    माप_प्रकार                        vtcm_f32_act_bytes;
    uint32_t                      act_elem_आकार;  // 4=F32 src1, 2=F16 src1
};

अटल व्योम transfer_activation_chunk_threaded(स्थिर काष्ठा activation_transfer_params * params) {
    काष्ठा htp_context *          ctx                = params->ctx;
    __fp16 *                      dst                = params->dst;
    स्थिर dma_addr_t              act_dma_addr       = params->act_dma_addr;
    पूर्णांक                           n_rows             = params->n_rows;
    पूर्णांक                           k_block            = params->k_block;
    पूर्णांक                           k_stride           = params->k_stride;
    पूर्णांक                           n_threads          = params->n_threads;
    स्थिर काष्ठा fastdiv_मूल्यावली * act_threads_div    = params->act_threads_div;
    स्थिर काष्ठा fastdiv_मूल्यावली * k_div              = params->k_div;
    पूर्णांक                           k_valid            = params->k_valid;
    भग्न *                       vtcm_f32_act       = params->vtcm_f32_act;
    माप_प्रकार                        vtcm_f32_act_bytes = params->vtcm_f32_act_bytes;
    // element size of the activation rows (4 = F32, 2 = F16).
    स्थिर uint32_t                act_elem_आकार      = params->act_elem_आकार ? params->act_elem_आकार : (uint32_t) माप(भग्न);

    अगर (n_rows <= 0) {
        वापस;
    }

    स्थिर माप_प्रकार n_tasks = (n_rows + 31) >> 5;
    अगर (act_elem_आकार == माप(भग्न) && n_threads > 1 && k_block > 32 && n_tasks < (माप_प्रकार) n_threads) {
        // Calculate step rows parameters for column-chunked dma pipelining
        uint32_t dma_step_rows = 2;
        uint32_t dma_step_rows_shift = 1;
        अगर (vtcm_f32_act_bytes > 0) {
            माप_प्रकार thread_scratch_bytes = hex_align_down(fastdiv(vtcm_f32_act_bytes, act_threads_div), 128);
            माप_प्रकार thread_scratch_elements = thread_scratch_bytes / माप(भग्न);
            माप_प्रकार dma_step_rows_max = fastdiv(thread_scratch_elements / 2, k_div);
            अगर (dma_step_rows_max >= 4) {
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
        वापस;
    }

    निश्चित(k_block % HTP_MM_HMX_TILE_N_COLS == 0 && k_stride % HTP_MM_HMX_TILE_N_COLS == 0);

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
    state.act_elem_आकार = act_elem_आकार;

    state.vtcm_f32_act_bytes_per_thread = hex_align_down(fastdiv(vtcm_f32_act_bytes, act_threads_div), 128);

    uint32_t dma_step_rows = 2;
    uint32_t dma_step_rows_shift = 1;
    अगर (state.vtcm_f32_act_bytes_per_thread > 0) {
        माप_प्रकार thread_scratch_elements = state.vtcm_f32_act_bytes_per_thread / माप(भग्न);
        माप_प्रकार dma_step_rows_max = fastdiv(thread_scratch_elements / 2, k_div);
        अगर (dma_step_rows_max >= 4) {
            dma_step_rows = 4;
            dma_step_rows_shift = 2;
        }
    }
    state.dma_step_rows      = dma_step_rows;
    state.dma_step_rows_shift = dma_step_rows_shift;

    पूर्णांक active_threads = hex_smin(n_threads, (पूर्णांक)state.n_tasks);
    अगर (state.n_tasks == 1 || n_threads == 1) {
        transfer_activation_chunk_worker_fn(1, 0, &state);
    } अन्यथा {
        worker_pool_run_func(ctx->worker_pool, transfer_activation_chunk_worker_fn, &state, active_threads);
    }
}
// --- Async HMX matmul job (for pipeline overlap) ---

प्रकार काष्ठा {
    __fp16 *       output;
    स्थिर __fp16 * activation;
    स्थिर __fp16 * weight;
    स्थिर __fp16 * scales;
    uint32_t       n_row_tiles;
    uint32_t       n_col_tiles;
    uint32_t       n_dot_tiles;
} hmx_matmul_job_t;

अटल व्योम hmx_matmul_worker_fn(व्योम * तथ्य) {
    hmx_matmul_job_t * job = (hmx_matmul_job_t *) तथ्य;
    FARF(ऊपरी_2, "hmx-mm-job: n_row_tiles %u n_col_tiles %u n_dot_tiles %u", job->n_row_tiles, job->n_col_tiles, job->n_dot_tiles);
    core_dot_chunk_fp16(job->output, job->activation, job->weight, job->scales, job->n_row_tiles, job->n_col_tiles, job->n_dot_tiles);
}

अटल अंतरभूत व्योम hmx_matmul_job_init(hmx_matmul_job_t * job,
                                       __fp16 *           output,
                                       स्थिर __fp16 *     activation,
                                       स्थिर __fp16 *     weight,
                                       स्थिर __fp16 *     scales,
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

अटल पूर्णांक hmx_mm_2d_f32(काष्ठा htp_context *ctx,
                                  dma_queue *weight_dma,
                                  भग्न *restrict dst,
                                  dma_addr_t src2_addr,
                                  माप_प्रकार src2_bytes,
                                  dma_addr_t act_dma_addr,
                                  dma_addr_t weight,
                                  पूर्णांक m, पूर्णांक k, पूर्णांक n,
                                  पूर्णांक act_stride,
                                  uint32_t act_elem_आकार,  // 4=F32 src1, 2=F16 src1
                                  पूर्णांक weight_stride,
                                  पूर्णांक weight_type,
                                  पूर्णांक k_valid,
                                  पूर्णांक dst_stride,
                                  uint32_t src2_stride,
                                  पूर्णांक dst_cols,
                                  पूर्णांक m_chunk,
                                  पूर्णांक n_chunk,
                                  पूर्णांक pipeline,
                                  पूर्णांक n_threads,
                                  पूर्णांक act_threads,
                                  स्थिर काष्ठा fastdiv_मूल्यावली * act_threads_div,
                                  स्थिर काष्ठा fastdiv_मूल्यावली * k_div,
                                  पूर्णांक tile_आकार,
                                  पूर्णांक aligned_tile_आकार,
                                  पूर्णांक vtcm_आकार) {
    काष्ठा htp_thread_trace * tr = &ctx->trace[0];
    htp_trace_event_start(tr, HTP_TRACE_EVT_INIT, 0);

    स्थिर bool is_tiled = (tile_आकार > 0);

    अगर (k % 32 != 0) {
        वापस -1;
    }
    अगर (!hex_is_aligned(dst, माप(भग्न))) { वापस -1; }

    माप_प्रकार row_stride = htp_tensor_get_row_आकार(weight_type, k);
    अगर (row_stride == 0) {
        वापस -1;
    }

    tiled_dequant_task_fn_t dequant_worker_fn = hmx_get_dequant_worker_fn(weight_type, is_tiled);

    स्थिर पूर्णांक n_k_tiles = k / HTP_MM_HMX_TILE_N_COLS;
    स्थिर पूर्णांक n_weight_tiles = k / htp_tensor_get_tile_k(weight_type);
    स्थिर काष्ठा fastdiv_मूल्यावली n_k_tiles_div = init_fastdiv_मूल्यावली(n_k_tiles);

    स्थिर माप_प्रकार vec_dot_आकार = k * माप(__fp16);
    स्थिर माप_प्रकार vtcm_budget  = ctx->vtcm_आकार;

    स्थिर uint32_t dma_dst_stride  = is_tiled ? aligned_tile_आकार : row_stride;
    स्थिर uint32_t dma_src_stride  = is_tiled ? tile_आकार : weight_stride;
    स्थिर uint32_t dma_width_bytes = is_tiled ? tile_आकार : row_stride;

    माप_प्रकार m_chunk_n_rows = m_chunk;
    माप_प्रकार n_chunk_n_cols = n_chunk;
    माप_प्रकार vtcm_used      = vtcm_आकार;

    काष्ठा htp_mm_hmx_vtcm_layout L;
    htp_mm_hmx_vtcm_layout_build(&L, HTP_MM_KERNEL_HMX_2D, weight_type, k, m_chunk_n_rows, n_chunk_n_cols, 1, pipeline, act_threads, aligned_tile_आकार, src2_bytes);

    vtcm_used = L.योग_bytes;
    अगर (vtcm_used > vtcm_budget) {
        FARF(ERROR, "hmx-mm-2d-precomputed: VTCM overflow: used %zu budget %zu, m %d k %d n %d mc %zu nc %zu",
             vtcm_used, vtcm_budget, m, k, n, m_chunk_n_rows, n_chunk_n_cols);
        वापस -1;
    }

    uint8_t * स्थिर base = (uint8_t *) ctx->vtcm_base;
    __fp16  *vtcm_weight_raw[2] = {
        VTCM_LAYOUT_PTR(__fp16, base, L.off_weight[0]),
        VTCM_LAYOUT_PTR_OPTIONAL(__fp16, base, L.off_weight[1], pipeline)
    };
    __fp16  *vtcm_f16_act    = VTCM_LAYOUT_PTR(__fp16, base, L.off_act);
    भग्न   *vtcm_f32_act    = VTCM_LAYOUT_PTR(भग्न, base, L.off_act_f32);
    __fp16  *vtcm_output     = VTCM_LAYOUT_PTR(__fp16, base, L.off_dst[0]);
    व्योम    *vtcm_scratch0   = VTCM_LAYOUT_PTR(व्योम, base, L.off_scratch[0]);
    व्योम    *vtcm_scratch1   = VTCM_LAYOUT_PTR_OPTIONAL(व्योम, base, L.off_scratch[1], pipeline);
    व्योम    *vtcm_scratch2   = VTCM_LAYOUT_PTR_OPTIONAL(व्योम, base, L.off_dst[1], pipeline);
    __fp16  *vtcm_scales     = VTCM_LAYOUT_PTR(__fp16, base, L.off_scales);

    hmx_init_column_scales(vtcm_scales, Q6_V_vsplat_R(0x3c00));  // scale: 1.0, bias: 0.0 in FP16

    स्थिर bool has_bias      = (src2_bytes > 0 && src2_addr != 0);
    भग्न   *vtcm_bias       = VTCM_LAYOUT_PTR_OPTIONAL(भग्न, base, L.off_bias, has_bias);
    अगर (has_bias) {
        dma_queue_धकेलो(weight_dma, dma_make_तथ्य(vtcm_bias, src2_addr), hex_align_up(src2_bytes, 128), 0, src2_bytes, 1);
        dma_queue_निकालो(weight_dma);
    }

    FARF(ऊपरी_2, "hmx-mm-2d: m %d k %d n %d wtype %d mc %zu nc %zu vtcm %zu/%zu",
         m, k, n, weight_type, m_chunk_n_rows, n_chunk_n_cols, vtcm_used, vtcm_budget);

    पूर्णांक n_chunk_cnt = hmx_ceil_div(n, n_chunk_n_cols);

    htp_trace_event_stop(tr, HTP_TRACE_EVT_INIT, 0);

    अगर (pipeline) {
        // --- Asynchronous Pipelined Loop ---
        hmx_matmul_job_t job_slots[2];  // persistent double-buffered job descriptors
        व्योम *vtcm_raw_bufs[4] = {
            vtcm_weight_raw[0],
            vtcm_weight_raw[1],
            dequant_worker_fn ? vtcm_weight_raw[0] : vtcm_scratch0,
            dequant_worker_fn ? vtcm_weight_raw[1] : vtcm_scratch1,
        };
        व्योम *vtcm_weight_bufs[4] = {
            dequant_worker_fn ? vtcm_scratch0 : vtcm_weight_raw[0],
            dequant_worker_fn ? vtcm_scratch1 : vtcm_weight_raw[1],
            vtcm_scratch0,
            vtcm_scratch1,
        };
        व्योम *vtcm_output_bufs[2] = { vtcm_output, vtcm_scratch2 };

        क्रम (माप_प्रकार mr = 0; mr < m; mr += m_chunk_n_rows) {
            स्थिर माप_प्रकार n_rows = hex_smin(m - mr, m_chunk_n_rows);

            काष्ठा activation_transfer_params act_params = {
                .ctx = ctx,
                .dst = vtcm_f16_act,
                .act_dma_addr = act_dma_addr + (माप_प्रकार) mr * act_stride * act_elem_आकार,  // byte offset (F16/F32)
                .n_rows = (पूर्णांक) n_rows,
                .k_block = k,
                .k_stride = act_stride,
                .n_threads = act_threads,
                .act_threads_div = act_threads_div,
                .k_div = k_div,
                .k_valid = k_valid,
                .vtcm_f32_act = vtcm_f32_act,
                .vtcm_f32_act_bytes = L.act_f32_bytes,
                .act_elem_आकार = act_elem_आकार,
            };
            transfer_activation_chunk_threaded(&act_params);

            // Prologue: push A0 and optionally A1 (if n_chunk_cnt > 1)
            स्थिर माप_प्रकार   n_cols_A0 = hex_smin(n - 0 * n_chunk_n_cols, n_chunk_n_cols);
            स्थिर uint32_t height_A0 = is_tiled ? hmx_ceil_div(n_cols_A0, 32) * n_weight_tiles : n_cols_A0;
            dma_queue_धकेलो(weight_dma, dma_make_तथ्य(vtcm_raw_bufs[0], weight),
                           dma_dst_stride, dma_src_stride, dma_width_bytes, height_A0);

            अगर (1 < n_chunk_cnt) {
                स्थिर माप_प्रकार   n_cols_A1 = hex_smin(n - 1 * n_chunk_n_cols, n_chunk_n_cols);
                स्थिर uint32_t height_A1 = is_tiled ? hmx_ceil_div(n_cols_A1, 32) * n_weight_tiles : n_cols_A1;
                dma_queue_धकेलो(weight_dma, dma_make_तथ्य(vtcm_raw_bufs[1], weight + n_chunk_n_cols * weight_stride),
                               dma_dst_stride, dma_src_stride, dma_width_bytes, height_A1);
            }

            // Main loop: pop A_i -> dequantize A_i -> push A_{i+2} -> submit C_i -> wait C_{i-1} and store D_{i-1}
            क्रम (पूर्णांक i = 0; i < n_chunk_cnt; ++i) {
                स्थिर माप_प्रकार nc    = i * n_chunk_n_cols;
                स्थिर माप_प्रकार nc_p2 = nc + 2 * n_chunk_n_cols;

                स्थिर माप_प्रकार n_cols    = hex_smin(n - nc, n_chunk_n_cols);
                स्थिर माप_प्रकार n_cols_p2 = hex_smin(n - nc_p2, n_chunk_n_cols);

                // 1. pop A_i
                व्योम * curr_raw = (व्योम *) dma_queue_निकालो(weight_dma).dst;

                // 2. dequantize A_i
                dequantize_tiled_weight_chunk_to_fp16_tiles(
                    ctx, vtcm_weight_bufs[i % 4], curr_raw,
                    n_cols, k, row_stride, weight_type,
                    n_k_tiles, n_k_tiles_div, dequant_worker_fn, n_threads);

                // 3. push A_{i+2} (if i+2 < n_chunk_cnt)
                अगर (i + 2 < n_chunk_cnt) {
                    स्थिर uint32_t height_p2 = is_tiled ? hmx_ceil_div(n_cols_p2, 32) * n_weight_tiles : n_cols_p2;
                    dma_queue_धकेलो(weight_dma, dma_make_तथ्य(vtcm_raw_bufs[(i + 2) % 4], weight + nc_p2 * weight_stride),
                                   dma_dst_stride, dma_src_stride, dma_width_bytes, height_p2);
                }

                // 4. submit C_i
                hmx_matmul_job_init(&job_slots[i % 2], (__fp16 *) vtcm_output_bufs[i % 2],
                                    (__fp16 *) vtcm_f16_act, (__fp16 *) vtcm_weight_bufs[i % 4],
                                    vtcm_scales, hmx_ceil_div(n_rows, HTP_MM_HMX_TILE_N_ROWS),
                                    hmx_ceil_div(n_cols, HTP_MM_HMX_TILE_N_COLS), k / HTP_MM_HMX_TILE_N_ROWS);
                hmx_queue_धकेलो(ctx->hmx_queue, hmx_queue_make_desc(hmx_matmul_worker_fn, &job_slots[i % 2]));

                // 5. wait C_{i-1} and store D_{i-1} (multi-thread HVX, parallel with C_i)
                अगर (i > 0) {
                    hmx_queue_निकालो(ctx->hmx_queue);
                    स्थिर माप_प्रकार nc_पूर्व = (i - 1) * n_chunk_n_cols;
                    स्थिर माप_प्रकार n_cols_पूर्व = hex_smin(n - nc_पूर्व, n_chunk_n_cols);
                    भग्न *output_chunk = dst + (mr * dst_stride + nc_पूर्व);
                    स्थिर भग्न *bias_chunk = has_bias ? (vtcm_bias + mr * src2_stride + nc_पूर्व) : शून्य;
                    पूर्णांक chunk_dst_cols = dst_cols - (पूर्णांक)nc_पूर्व;
                    अगर (chunk_dst_cols > 0) {
                        transfer_output_chunk_threaded(ctx, output_chunk, bias_chunk, vtcm_output_bufs[(i - 1) % 2], n_rows, n_cols_पूर्व, dst_stride, src2_stride, chunk_dst_cols, n_threads);
                    }
                }
            }

            // Epilogue: wait C_{last} and store D_{last}
            hmx_queue_निकालो(ctx->hmx_queue);
            स्थिर माप_प्रकार nc_last = (n_chunk_cnt - 1) * n_chunk_n_cols;
            स्थिर माप_प्रकार n_cols_last = hex_smin(n - nc_last, n_chunk_n_cols);
            भग्न *output_chunk = dst + (mr * dst_stride + nc_last);
            स्थिर भग्न *bias_chunk = has_bias ? (vtcm_bias + mr * src2_stride + nc_last) : शून्य;
            पूर्णांक chunk_dst_cols = dst_cols - (पूर्णांक)nc_last;
            अगर (chunk_dst_cols > 0) {
                transfer_output_chunk_threaded(ctx, output_chunk, bias_chunk, vtcm_output_bufs[(n_chunk_cnt - 1) % 2], n_rows, n_cols_last, dst_stride, src2_stride, chunk_dst_cols, n_threads);
            }
        }
    } अन्यथा {
        // --- Synchronous loop (m <= 32 or fallback) ---
        hmx_matmul_job_t job;
        व्योम *vtcm_raw_bufs[2] = { vtcm_weight_raw[0], dequant_worker_fn ? vtcm_weight_raw[0] : vtcm_scratch0 };
        व्योम *vtcm_weight_bufs[2] = { dequant_worker_fn ? vtcm_scratch0 : vtcm_weight_raw[0], vtcm_scratch0 };

        क्रम (माप_प्रकार mr = 0; mr < m; mr += m_chunk_n_rows) {
            स्थिर माप_प्रकार n_rows = hex_smin(m - mr, m_chunk_n_rows);

            काष्ठा activation_transfer_params act_params = {
                .ctx = ctx,
                .dst = vtcm_f16_act,
                .act_dma_addr = act_dma_addr + (माप_प्रकार) mr * act_stride * act_elem_आकार,  // byte offset (F16/F32)
                .n_rows = (पूर्णांक) n_rows,
                .k_block = k,
                .k_stride = act_stride,
                .n_threads = act_threads,
                .act_threads_div = act_threads_div,
                .k_div = k_div,
                .k_valid = k_valid,
                .vtcm_f32_act = vtcm_f32_act,
                .vtcm_f32_act_bytes = L.act_f32_bytes,
                .act_elem_आकार = act_elem_आकार,
            };
            transfer_activation_chunk_threaded(&act_params);

            // A0: Pre-fetch the first weight chunk (nc = 0)
            अगर (n > 0) {
                स्थिर माप_प्रकार n_cols = hex_smin(n, n_chunk_n_cols);
                स्थिर uint32_t height = is_tiled ? hmx_ceil_div(n_cols, 32) * n_weight_tiles : n_cols;
                dma_queue_धकेलो(weight_dma, dma_make_तथ्य(vtcm_raw_bufs[0], weight), dma_dst_stride, dma_src_stride, dma_width_bytes, height);
            }

            क्रम (पूर्णांक i = 0; i < n_chunk_cnt; ++i) {
                स्थिर माप_प्रकार nc = (माप_प्रकार) i * n_chunk_n_cols;
                स्थिर माप_प्रकार n_cols = hex_smin(n - nc, n_chunk_n_cols);
                स्थिर माप_प्रकार n_row_tiles = hmx_ceil_div(n_rows, HTP_MM_HMX_TILE_N_ROWS);
                स्थिर माप_प्रकार n_col_tiles = hmx_ceil_div(n_cols, HTP_MM_HMX_TILE_N_COLS);

                // A: Wait for weight DMA
                व्योम * curr_raw = (व्योम *) dma_queue_निकालो(weight_dma).dst;

                // B: Weight Dequantize (Threaded)
                dequantize_tiled_weight_chunk_to_fp16_tiles(
                    ctx, vtcm_weight_bufs[i % 2], curr_raw,
                    n_cols, k, row_stride, weight_type,
                    n_k_tiles, n_k_tiles_div, dequant_worker_fn, n_threads);

                // Start weight DMA for the next chunk early
                स्थिर माप_प्रकार nc_अग्र = nc + n_chunk_n_cols;
                अगर (nc_अग्र < n) {
                    स्थिर माप_प्रकार n_cols_अग्र = hex_smin(n - nc_अग्र, n_chunk_n_cols);
                    स्थिर uint32_t height_अग्र = is_tiled ? hmx_ceil_div(n_cols_अग्र, 32) * n_weight_tiles : n_cols_अग्र;
                    dma_queue_धकेलो(weight_dma, dma_make_तथ्य(vtcm_raw_bufs[(i + 1) % 2], weight + nc_अग्र * weight_stride), dma_dst_stride, dma_src_stride, dma_width_bytes, height_अग्र);
                }

                // C: HMX Compute (Queue-based)
                hmx_matmul_job_init(&job, vtcm_output, vtcm_f16_act, (__fp16 *) vtcm_weight_bufs[i % 2], vtcm_scales, n_row_tiles, n_col_tiles, k / HTP_MM_HMX_TILE_N_ROWS);
                hmx_queue_धकेलो(ctx->hmx_queue, hmx_queue_make_desc(hmx_matmul_worker_fn, &job));
                hmx_queue_निकालो(ctx->hmx_queue);

                // D: Output Store
                भग्न *output_chunk = dst + (mr * dst_stride + nc);
                स्थिर भग्न *bias_chunk = has_bias ? (vtcm_bias + mr * src2_stride + nc) : शून्य;
                पूर्णांक chunk_dst_cols = dst_cols - (पूर्णांक)nc;
                अगर (chunk_dst_cols > 0) {
                    transfer_output_chunk_threaded(ctx, output_chunk, bias_chunk, vtcm_output, n_rows, n_cols, dst_stride, src2_stride, chunk_dst_cols, n_threads);
                }
            }
        }
    }

    वापस 0;
}

अटल पूर्णांक hmx_mm_nx_2d_f32(काष्ठा htp_ops_context * octx, स्थिर काष्ठा htp_mm_kernel_params * kparams) {
    काष्ठा htp_context * ctx = octx->ctx;
    काष्ठा htp_thread_trace * tr = &ctx->trace[0];
    htp_trace_event_start(tr, HTP_TRACE_EVT_INIT, 0);

    स्थिर uint32_t n_weights = kparams->n_weights;
    अगर (n_weights == 0 || n_weights > HTP_OP_MAX_OUTPUTS) {
        वापस HTP_STATUS_INVAL_PARAMS;
    }

    स्थिर काष्ठा htp_tensor * restrict src0 = octx->src[0];
    स्थिर काष्ठा htp_tensor * restrict act  = octx->src[n_weights];

    स्थिर पूर्णांक weight_type = (पूर्णांक) src0->type;
    स्थिर पूर्णांक k           = (पूर्णांक) act->ne[0];
    स्थिर पूर्णांक k_valid     = (पूर्णांक) act->ne[0];
    स्थिर पूर्णांक m           = (पूर्णांक) (act->ne[1] * act->ne[2] * act->ne[3]);
    स्थिर uint32_t act_elem_आकार = (act->type == HTP_TYPE_F16) ? माप(__fp16) : माप(भग्न);
    स्थिर पूर्णांक act_stride  = (पूर्णांक) (act->nb[1] / act_elem_आकार);
    स्थिर dma_addr_t act_dma_addr = act->तथ्य;

    अगर (k % 32 != 0) { वापस HTP_STATUS_NO_SUPPORT; }

    माप_प्रकार row_stride = htp_tensor_get_row_आकार(weight_type, k);
    अगर (row_stride == 0) {
        वापस HTP_STATUS_NO_SUPPORT;
    }

    स्थिर पूर्णांक n_k_tiles = k / HTP_MM_HMX_TILE_N_COLS;
    स्थिर पूर्णांक n_weight_tiles = k / htp_tensor_get_tile_k(weight_type);
    स्थिर काष्ठा fastdiv_मूल्यावली n_k_tiles_div = init_fastdiv_मूल्यावली(n_k_tiles);

    स्थिर पूर्णांक tile_आकार       = kparams->tile_आकार;
    स्थिर पूर्णांक aligned_tile_आकार = kparams->aligned_tile_आकार;

    स्थिर bool is_tiled       = (tile_आकार > 0);
    tiled_dequant_task_fn_t dequant_worker_fn = hmx_get_dequant_worker_fn(weight_type, is_tiled);
    स्थिर माप_प्रकार vtcm_budget  = ctx->vtcm_आकार;

    स्थिर पूर्णांक m_chunk_n_rows  = kparams->m_chunk;
    स्थिर पूर्णांक n_chunk_n_cols  = kparams->n_chunk;
    स्थिर पूर्णांक pipeline        = kparams->pipeline;
    स्थिर पूर्णांक n_threads       = octx->n_threads;
    स्थिर पूर्णांक act_threads     = kparams->n_act_threads;
    स्थिर काष्ठा fastdiv_मूल्यावली * act_threads_div = &kparams->div_n_act_threads;
    स्थिर काष्ठा fastdiv_मूल्यावली * k_div           = &kparams->div_ne00_padded;

    स्थिर uint32_t dma_dst_stride  = is_tiled ? aligned_tile_आकार : row_stride;
    स्थिर uint32_t dma_width_bytes = is_tiled ? tile_आकार : row_stride;

    काष्ठा htp_mm_hmx_vtcm_layout L;
    htp_mm_hmx_vtcm_layout_build(&L, HTP_MM_KERNEL_HMX_2D, weight_type, k, m_chunk_n_rows, n_chunk_n_cols, 1, pipeline, act_threads, aligned_tile_आकार, 0);

    अगर (L.योग_bytes > vtcm_budget) {
        FARF(ERROR, "hmx-mm-nx-2d: VTCM overflow: used %zu budget %zu, m %d k %d mc %d nc %d",
             L.योग_bytes, vtcm_budget, m, k, m_chunk_n_rows, n_chunk_n_cols);
        वापस HTP_STATUS_VTCM_TOO_SMALL;
    }

    uint8_t * स्थिर base = (uint8_t *) ctx->vtcm_base;
    __fp16  *vtcm_weight_raw[2] = {
        VTCM_LAYOUT_PTR(__fp16, base, L.off_weight[0]),
        VTCM_LAYOUT_PTR_OPTIONAL(__fp16, base, L.off_weight[1], pipeline)
    };
    __fp16  *vtcm_f16_act    = VTCM_LAYOUT_PTR(__fp16, base, L.off_act);
    भग्न   *vtcm_f32_act    = VTCM_LAYOUT_PTR(भग्न, base, L.off_act_f32);
    __fp16  *vtcm_output     = VTCM_LAYOUT_PTR(__fp16, base, L.off_dst[0]);
    व्योम    *vtcm_scratch0   = VTCM_LAYOUT_PTR(व्योम, base, L.off_scratch[0]);
    व्योम    *vtcm_scratch1   = VTCM_LAYOUT_PTR_OPTIONAL(व्योम, base, L.off_scratch[1], pipeline);
    व्योम    *vtcm_scratch2   = VTCM_LAYOUT_PTR_OPTIONAL(व्योम, base, L.off_dst[1], pipeline);
    __fp16  *vtcm_scales     = VTCM_LAYOUT_PTR(__fp16, base, L.off_scales);

    hmx_init_column_scales(vtcm_scales, Q6_V_vsplat_R(0x3c00));  // scale: 1.0, bias: 0.0 in FP16

    पूर्णांक m_start = 0;
    पूर्णांक m_rows  = m;
    अगर (octx->ctx->mdev.गिनती > 1) {
        bool can_split = htp_tensor_can_row_partition(act, act_elem_आकार);
        अगर (can_split) {
            क्रम (uint32_t p = 0; p < n_weights; ++p) {
                स्थिर काष्ठा htp_tensor * restrict dst = octx->dsts[p];
                अगर (!htp_tensor_can_row_partition(dst, माप(भग्न))) {
                    can_split = false;
                    अवरोध;
                }
            }
        }
        स्थिर काष्ठा htp_tensor_mdev_range range = htp_tensor_mdev_partition((uint32_t) m, can_split ? 1 : 0, octx->ctx->mdev.idx, octx->ctx->mdev.गिनती, &octx->ctx->mdev.गिनती_div);
        m_start = (पूर्णांक) range.start;
        m_rows  = (पूर्णांक) range.गिनती;
    }

    अगर (m_rows == 0) {
        वापस HTP_STATUS_OK;
    }

    FARF(ऊपरी_2, "hmx-mm-nx-2d: n_weights %u m %d (%d..%d) k %d wtype %d mc %d nc %d vtcm %zu/%zu",
         n_weights, m, m_start, m_start + m_rows, k, weight_type, m_chunk_n_rows, n_chunk_n_cols, L.योग_bytes, vtcm_budget);

    htp_trace_event_stop(tr, HTP_TRACE_EVT_INIT, 0);

    स्थिर माप_प्रकार mr_end = (माप_प्रकार)(m_start + m_rows);

    अगर (pipeline) {
        hmx_matmul_job_t job_slots[2];
        व्योम *vtcm_raw_bufs[4] = {
            vtcm_weight_raw[0],
            vtcm_weight_raw[1],
            dequant_worker_fn ? vtcm_weight_raw[0] : vtcm_scratch0,
            dequant_worker_fn ? vtcm_weight_raw[1] : vtcm_scratch1,
        };
        व्योम *vtcm_weight_bufs[4] = {
            dequant_worker_fn ? vtcm_scratch0 : vtcm_weight_raw[0],
            dequant_worker_fn ? vtcm_scratch1 : vtcm_weight_raw[1],
            vtcm_scratch0,
            vtcm_scratch1,
        };
        व्योम *vtcm_output_bufs[2] = { vtcm_output, vtcm_scratch2 };

        क्रम (माप_प्रकार mr = (माप_प्रकार) m_start; mr < mr_end; mr += m_chunk_n_rows) {
            स्थिर माप_प्रकार n_rows = hex_smin(mr_end - mr, m_chunk_n_rows);

            काष्ठा activation_transfer_params act_params = {
                .ctx = ctx,
                .dst = vtcm_f16_act,
                .act_dma_addr = act_dma_addr + (माप_प्रकार) mr * act_stride * act_elem_आकार,
                .n_rows = (पूर्णांक) n_rows,
                .k_block = k,
                .k_stride = act_stride,
                .n_threads = act_threads,
                .act_threads_div = act_threads_div,
                .k_div = k_div,
                .k_valid = k_valid,
                .vtcm_f32_act = vtcm_f32_act,
                .vtcm_f32_act_bytes = L.act_f32_bytes,
                .act_elem_आकार = act_elem_आकार,
            };
            transfer_activation_chunk_threaded(&act_params);

            क्रम (uint32_t p = 0; p < n_weights; p++) {
                स्थिर काष्ठा htp_tensor * restrict src_w = octx->src[p];
                स्थिर काष्ठा htp_tensor * restrict dst   = octx->dsts[p];
                अगर (!src_w || !dst) जारी;

                स्थिर dma_addr_t weight       = src_w->तथ्य;
                dma_queue * weight_dma      = octx->ctx->dma[0];
                भग्न * dst_ptr            = (भग्न *) dst->तथ्य;
                स्थिर माप_प्रकार n             = src_w->ne[1];
                अगर (n == 0) जारी;
                स्थिर माप_प्रकार weight_stride = src_w->nb[1];
                स्थिर माप_प्रकार dst_stride    = dst->nb[1] / माप(भग्न);
                स्थिर पूर्णांक dst_cols         = (पूर्णांक) dst->ne[0];
                स्थिर पूर्णांक n_chunk_cnt      = hmx_ceil_div(n, n_chunk_n_cols);

                स्थिर uint32_t dma_src_stride = is_tiled ? tile_आकार : weight_stride;

                स्थिर माप_प्रकार   n_cols_A0 = hex_smin(n - 0 * n_chunk_n_cols, n_chunk_n_cols);
                स्थिर uint32_t height_A0 = is_tiled ? hmx_ceil_div(n_cols_A0, 32) * n_weight_tiles : n_cols_A0;
                dma_queue_धकेलो(weight_dma, dma_make_तथ्य(vtcm_raw_bufs[0], weight),
                               dma_dst_stride, dma_src_stride, dma_width_bytes, height_A0);

                अगर (1 < n_chunk_cnt) {
                    स्थिर माप_प्रकार   n_cols_A1 = hex_smin(n - 1 * n_chunk_n_cols, n_chunk_n_cols);
                    स्थिर uint32_t height_A1 = is_tiled ? hmx_ceil_div(n_cols_A1, 32) * n_weight_tiles : n_cols_A1;
                    dma_queue_धकेलो(weight_dma, dma_make_तथ्य(vtcm_raw_bufs[1], weight + n_chunk_n_cols * weight_stride),
                                   dma_dst_stride, dma_src_stride, dma_width_bytes, height_A1);
                }

                क्रम (पूर्णांक i = 0; i < n_chunk_cnt; ++i) {
                    स्थिर माप_प्रकार nc    = i * n_chunk_n_cols;
                    स्थिर माप_प्रकार nc_p2 = nc + 2 * n_chunk_n_cols;

                    स्थिर माप_प्रकार n_cols    = hex_smin(n - nc, n_chunk_n_cols);
                    स्थिर माप_प्रकार n_cols_p2 = hex_smin(n - nc_p2, n_chunk_n_cols);

                    व्योम * curr_raw = (व्योम *) dma_queue_निकालो(weight_dma).dst;

                    dequantize_tiled_weight_chunk_to_fp16_tiles(
                        ctx, vtcm_weight_bufs[i % 4], curr_raw,
                        n_cols, k, row_stride, weight_type,
                        n_k_tiles, n_k_tiles_div, dequant_worker_fn, n_threads);

                    अगर (i + 2 < n_chunk_cnt) {
                        स्थिर uint32_t height_p2 = is_tiled ? hmx_ceil_div(n_cols_p2, 32) * n_weight_tiles : n_cols_p2;
                        dma_queue_धकेलो(weight_dma, dma_make_तथ्य(vtcm_raw_bufs[(i + 2) % 4], weight + nc_p2 * weight_stride),
                                       dma_dst_stride, dma_src_stride, dma_width_bytes, height_p2);
                    }

                    hmx_matmul_job_init(&job_slots[i % 2], (__fp16 *) vtcm_output_bufs[i % 2],
                                        (__fp16 *) vtcm_f16_act, (__fp16 *) vtcm_weight_bufs[i % 4],
                                        vtcm_scales, hmx_ceil_div(n_rows, HTP_MM_HMX_TILE_N_ROWS),
                                        hmx_ceil_div(n_cols, HTP_MM_HMX_TILE_N_COLS), k / HTP_MM_HMX_TILE_N_ROWS);
                    hmx_queue_धकेलो(ctx->hmx_queue, hmx_queue_make_desc(hmx_matmul_worker_fn, &job_slots[i % 2]));

                    अगर (i > 0) {
                        hmx_queue_निकालो(ctx->hmx_queue);
                        स्थिर माप_प्रकार nc_पूर्व = (i - 1) * n_chunk_n_cols;
                        स्थिर माप_प्रकार n_cols_पूर्व = hex_smin(n - nc_पूर्व, n_chunk_n_cols);
                        भग्न *output_chunk = dst_ptr + (mr * dst_stride + nc_पूर्व);
                        पूर्णांक chunk_dst_cols = dst_cols - (पूर्णांक)nc_पूर्व;
                        अगर (chunk_dst_cols > 0) {
                            transfer_output_chunk_threaded(ctx, output_chunk, शून्य, vtcm_output_bufs[(i - 1) % 2], n_rows, n_cols_पूर्व, dst_stride, 0, chunk_dst_cols, n_threads);
                        }
                    }
                }

                hmx_queue_निकालो(ctx->hmx_queue);
                स्थिर माप_प्रकार nc_last = (n_chunk_cnt - 1) * n_chunk_n_cols;
                स्थिर माप_प्रकार n_cols_last = hex_smin(n - nc_last, n_chunk_n_cols);
                भग्न *output_chunk = dst_ptr + (mr * dst_stride + nc_last);
                पूर्णांक chunk_dst_cols = dst_cols - (पूर्णांक)nc_last;
                अगर (chunk_dst_cols > 0) {
                    transfer_output_chunk_threaded(ctx, output_chunk, शून्य, vtcm_output_bufs[(n_chunk_cnt - 1) % 2], n_rows, n_cols_last, dst_stride, 0, chunk_dst_cols, n_threads);
                }
            }
        }
    } अन्यथा {
        hmx_matmul_job_t job;
        व्योम *vtcm_raw_bufs[2] = { vtcm_weight_raw[0], dequant_worker_fn ? vtcm_weight_raw[0] : vtcm_scratch0 };
        व्योम *vtcm_weight_bufs[2] = { dequant_worker_fn ? vtcm_scratch0 : vtcm_weight_raw[0], vtcm_scratch0 };

        क्रम (माप_प्रकार mr = (माप_प्रकार) m_start; mr < mr_end; mr += m_chunk_n_rows) {
            स्थिर माप_प्रकार n_rows = hex_smin(mr_end - mr, m_chunk_n_rows);

            काष्ठा activation_transfer_params act_params = {
                .ctx = ctx,
                .dst = vtcm_f16_act,
                .act_dma_addr = act_dma_addr + (माप_प्रकार) mr * act_stride * act_elem_आकार,
                .n_rows = (पूर्णांक) n_rows,
                .k_block = k,
                .k_stride = act_stride,
                .n_threads = act_threads,
                .act_threads_div = act_threads_div,
                .k_div = k_div,
                .k_valid = k_valid,
                .vtcm_f32_act = vtcm_f32_act,
                .vtcm_f32_act_bytes = L.act_f32_bytes,
                .act_elem_आकार = act_elem_आकार,
            };
            transfer_activation_chunk_threaded(&act_params);

            क्रम (uint32_t p = 0; p < n_weights; p++) {
                स्थिर काष्ठा htp_tensor * restrict src_w = octx->src[p];
                स्थिर काष्ठा htp_tensor * restrict dst   = octx->dsts[p];
                अगर (!src_w || !dst) जारी;

                स्थिर dma_addr_t weight       = src_w->तथ्य;
                dma_queue * weight_dma      = octx->ctx->dma[0];
                भग्न * dst_ptr            = (भग्न *) dst->तथ्य;
                स्थिर माप_प्रकार n             = src_w->ne[1];
                अगर (n == 0) जारी;
                स्थिर माप_प्रकार weight_stride = src_w->nb[1];
                स्थिर माप_प्रकार dst_stride    = dst->nb[1] / माप(भग्न);
                स्थिर पूर्णांक dst_cols         = (पूर्णांक) dst->ne[0];
                स्थिर पूर्णांक n_chunk_cnt      = hmx_ceil_div(n, n_chunk_n_cols);

                स्थिर uint32_t dma_src_stride = is_tiled ? tile_आकार : weight_stride;

                अगर (n > 0) {
                    स्थिर माप_प्रकार n_cols = hex_smin(n, n_chunk_n_cols);
                    स्थिर uint32_t height = is_tiled ? hmx_ceil_div(n_cols, 32) * n_weight_tiles : n_cols;
                    dma_queue_धकेलो(weight_dma, dma_make_तथ्य(vtcm_raw_bufs[0], weight), dma_dst_stride, dma_src_stride, dma_width_bytes, height);
                }

                क्रम (पूर्णांक i = 0; i < n_chunk_cnt; ++i) {
                    स्थिर माप_प्रकार nc = (माप_प्रकार) i * n_chunk_n_cols;
                    स्थिर माप_प्रकार n_cols = hex_smin(n - nc, n_chunk_n_cols);
                    स्थिर माप_प्रकार n_row_tiles = hmx_ceil_div(n_rows, HTP_MM_HMX_TILE_N_ROWS);
                    स्थिर माप_प्रकार n_col_tiles = hmx_ceil_div(n_cols, HTP_MM_HMX_TILE_N_COLS);

                    व्योम * curr_raw = (व्योम *) dma_queue_निकालो(weight_dma).dst;

                    dequantize_tiled_weight_chunk_to_fp16_tiles(
                        ctx, vtcm_weight_bufs[i % 2], curr_raw,
                        n_cols, k, row_stride, weight_type,
                        n_k_tiles, n_k_tiles_div, dequant_worker_fn, n_threads);

                    स्थिर माप_प्रकार nc_अग्र = nc + n_chunk_n_cols;
                    अगर (nc_अग्र < n) {
                        स्थिर माप_प्रकार n_cols_अग्र = hex_smin(n - nc_अग्र, n_chunk_n_cols);
                        स्थिर uint32_t height_अग्र = is_tiled ? hmx_ceil_div(n_cols_अग्र, 32) * n_weight_tiles : n_cols_अग्र;
                        dma_queue_धकेलो(weight_dma, dma_make_तथ्य(vtcm_raw_bufs[(i + 1) % 2], weight + nc_अग्र * weight_stride), dma_dst_stride, dma_src_stride, dma_width_bytes, height_अग्र);
                    }

                    hmx_matmul_job_init(&job, vtcm_output, vtcm_f16_act, (__fp16 *) vtcm_weight_bufs[i % 2], vtcm_scales, n_row_tiles, n_col_tiles, k / HTP_MM_HMX_TILE_N_ROWS);
                    hmx_queue_धकेलो(ctx->hmx_queue, hmx_queue_make_desc(hmx_matmul_worker_fn, &job));
                    hmx_queue_निकालो(ctx->hmx_queue);

                    भग्न *output_chunk = dst_ptr + (mr * dst_stride + nc);
                    पूर्णांक chunk_dst_cols = dst_cols - (पूर्णांक)nc;
                    अगर (chunk_dst_cols > 0) {
                        transfer_output_chunk_threaded(ctx, output_chunk, शून्य, vtcm_output, n_rows, n_cols, dst_stride, 0, chunk_dst_cols, n_threads);
                    }
                }
            }
        }
    }

    वापस HTP_STATUS_OK;
}

अटल अंतरभूत dma_addr_t hmx_mm_weight_batch_तथ्य(स्थिर hmx_mm_batched_params_t *params,
                                                         पूर्णांक dst_b2, पूर्णांक dst_b3) {
    स्थिर माप_प्रकार b2_idx = (params->r2 <= 1) ? (माप_प्रकार) dst_b2 : (माप_प्रकार) fastdiv((uint32_t) dst_b2, &params->div_r2);
    स्थिर माप_प्रकार b3_idx = (params->r3 <= 1) ? (माप_प्रकार) dst_b3 : (माप_प्रकार) fastdiv((uint32_t) dst_b3, &params->div_r3);
    वापस params->weight + b2_idx * params->src0_nb2 + b3_idx * params->src0_nb3;
}

अटल अंतरभूत dma_addr_t hmx_mm_act_batch_addr(स्थिर hmx_mm_batched_params_t *params,
                                                पूर्णांक dst_b2, पूर्णांक dst_b3) {
    वापस params->act_dma_addr + dst_b2 * params->act_nb2 +
           dst_b3 * params->act_nb3;
}

अटल अंतरभूत भग्न *hmx_mm_dst_batch_ptr(स्थिर hmx_mm_batched_params_t *params,
                                              पूर्णांक dst_b2, पूर्णांक dst_b3) {
    वापस (भग्न *) ((uint8_t *) params->dst +
                      (माप_प्रकार) dst_b2 * params->dst_nb2 +
                      (माप_प्रकार) dst_b3 * params->dst_nb3);
}

अटल पूर्णांक hmx_mm_batched_simple(काष्ठा htp_context *ctx,
                                 स्थिर hmx_mm_batched_params_t *params,
                                 पूर्णांक m_chunk, पूर्णांक n_chunk, पूर्णांक pipeline, पूर्णांक n_threads, पूर्णांक act_threads, पूर्णांक vtcm_आकार,
                                 स्थिर काष्ठा fastdiv_मूल्यावली * act_threads_div, स्थिर काष्ठा fastdiv_मूल्यावली * k_div) {
    पूर्णांक ret = 0;
    क्रम (पूर्णांक b3 = 0; b3 < params->ne13 && ret == 0; ++b3) {
        क्रम (पूर्णांक b2 = 0; b2 < params->ne12 && ret == 0; ++b2) {
            dma_addr_t cur_bias_addr = params->bias_addr ? (params->bias_addr +
                                       b2 * params->bias_nb2 +
                                       b3 * params->bias_nb3) : 0;
            ret = hmx_mm_2d_f32(ctx, params->weight_dma, hmx_mm_dst_batch_ptr(params, b2, b3),
                                cur_bias_addr, params->bias_bytes,
                                hmx_mm_act_batch_addr(params, b2, b3),
                                hmx_mm_weight_batch_तथ्य(params, b2, b3),
                                params->m, params->k, params->n,
                                params->act_stride, params->act_elem_आकार, params->weight_stride,
                                params->weight_type, params->k, params->dst_stride, params->bias_stride, params->dst_cols,
                                m_chunk, n_chunk, pipeline, n_threads, act_threads,
                                act_threads_div, k_div, params->tile_आकार, params->aligned_tile_आकार, vtcm_आकार);
        }
    }
    वापस ret;
}

अटल पूर्णांक hmx_mm_batched(काष्ठा htp_context *ctx, स्थिर hmx_mm_batched_params_t *params,
                          पूर्णांक m_chunk, पूर्णांक n_chunk, पूर्णांक pipeline, पूर्णांक n_threads, पूर्णांक act_threads,
                          स्थिर काष्ठा fastdiv_मूल्यावली * act_threads_div,
                          स्थिर काष्ठा fastdiv_मूल्यावली * k_div,
                          पूर्णांक vtcm_आकार) {
    स्थिर माप_प्रकार row_stride = htp_tensor_get_row_आकार(params->weight_type, params->k);
    अगर (row_stride == 0) { वापस -1; }
    अगर (params->act_stride < params->k || (params->tile_आकार == 0 && (माप_प्रकार) params->weight_stride < row_stride) || params->dst_stride < params->dst_cols) { वापस -1; }
    अगर (params->ne02 <= 0 || params->ne03 <= 0 || params->ne12 <= 0 || params->ne13 <= 0) { वापस -1; }
    अगर (params->ne12 % params->ne02 != 0 || params->ne13 % params->ne03 != 0) { वापस -1; }
    // N (the weight row count) does not have to be 32-aligned:
    अगर (params->k % 32 != 0) { वापस -1; }
    अगर (!hex_is_aligned(params->dst, माप(भग्न))) { वापस -1; }

    स्थिर पूर्णांक group_आकार = params->r2;
    स्थिर माप_प्रकार vtcm_budget  = ctx->vtcm_आकार;

    // Check if the precomputed parameters are grouped or simple.
    // If simple, or if group_size <= 1, we use simple fallback loop.
    // Grouped path is only valid for flat F16 if group_size > 1 and it fits within VTCM budget.
    bool run_grouped = (params->weight_type == HTP_TYPE_F16 && params->tile_आकार == 0 && group_आकार > 1 && (माप_प्रकार)vtcm_आकार <= vtcm_budget);
    अगर (!run_grouped) {
        वापस hmx_mm_batched_simple(ctx, params, m_chunk, n_chunk, pipeline, n_threads, act_threads, vtcm_आकार, act_threads_div, k_div);
    }

    काष्ठा htp_thread_trace * tr = &ctx->trace[0];
    htp_trace_event_start(tr, HTP_TRACE_EVT_INIT, 0);

    माप_प्रकार m_chunk_n_rows = m_chunk;
    माप_प्रकार n_chunk_n_cols = n_chunk;

    काष्ठा htp_mm_hmx_vtcm_layout L;
    htp_mm_hmx_vtcm_layout_build(&L, HTP_MM_KERNEL_HMX_BATCHED, HTP_TYPE_F16, params->k, m_chunk_n_rows, n_chunk_n_cols, group_आकार, false, act_threads, 0, params->bias_bytes);

    अगर (L.योग_bytes > vtcm_budget) {
        FARF(ऊपरी_2, "%s: grouped layout overflowed VTCM, falling back to simple batched loop", __func__);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_INIT, 0);
        वापस hmx_mm_batched_simple(ctx, params, m_chunk, n_chunk, pipeline, n_threads, act_threads, vtcm_आकार, act_threads_div, k_div);
    }

    uint8_t * स्थिर base = (uint8_t *) ctx->vtcm_base;
    __fp16  *vtcm_weight     = VTCM_LAYOUT_PTR(__fp16, base, L.off_weight[0]);
    __fp16  *vtcm_f16_act    = VTCM_LAYOUT_PTR(__fp16, base, L.off_act);
    __fp16  *vtcm_output     = VTCM_LAYOUT_PTR(__fp16, base, L.off_dst[0]);
    व्योम    *vtcm_scratch0   = VTCM_LAYOUT_PTR(व्योम, base, L.off_scratch[0]);
    व्योम    *vtcm_scratch1   = VTCM_LAYOUT_PTR(व्योम, base, L.off_scratch[1]);
    __fp16  *vtcm_scales     = VTCM_LAYOUT_PTR(__fp16, base, L.off_scales);
    भग्न   *vtcm_f32_act    = VTCM_LAYOUT_PTR(भग्न, base, L.off_act_f32);

    स्थिर bool has_bias      = (params->bias_bytes > 0 && params->bias_addr != 0);
    भग्न   *vtcm_bias       = VTCM_LAYOUT_PTR_OPTIONAL(भग्न, base, L.off_bias, has_bias);
    अगर (has_bias) {
        dma_queue_धकेलो(params->weight_dma, dma_make_तथ्य(vtcm_bias, params->bias_addr), hex_align_up(params->bias_bytes, 128), 0, params->bias_bytes, 1);
        dma_queue_निकालो(params->weight_dma);
    }

    hmx_init_column_scales(vtcm_scales, Q6_V_vsplat_R(0x3c00));  // scale: 1.0, bias: 0.0 in FP16

    FARF(ऊपरी_2, "%s: grouped path m=%d k=%d n=%d group=%d streams=%d mc=%zu nc=%zu vtcm=%zu/%zu",
            __func__, params->m, params->k, params->n, group_आकार, params->ne13,
            m_chunk_n_rows, n_chunk_n_cols,
            L.योग_bytes, vtcm_budget);

    स्थिर माप_प्रकार fp16_row_bytes   = (माप_प्रकार) params->k * माप(__fp16);
    स्थिर माप_प्रकार weight_row_bytes = (माप_प्रकार) params->weight_stride;

    htp_trace_event_stop(tr, HTP_TRACE_EVT_INIT, 0);

    hmx_matmul_job_t job;

    क्रम (पूर्णांक b3 = 0; b3 < params->ne13; ++b3) {
        क्रम (पूर्णांक b2_base = 0; b2_base < params->ne12; b2_base += group_आकार) {
            स्थिर dma_addr_t weight_group = hmx_mm_weight_batch_तथ्य(params, b2_base, b3);
            dma_queue * weight_dma = params->weight_dma;

            क्रम (माप_प्रकार mr = 0; mr < (माप_प्रकार) params->m; mr += m_chunk_n_rows) {
                स्थिर माप_प्रकार n_rows = hex_smin((माप_प्रकार) params->m - mr, m_chunk_n_rows);
                स्थिर माप_प्रकार n_row_tiles = hmx_ceil_div((पूर्णांक) n_rows, HTP_MM_HMX_TILE_N_ROWS);

                // Pre-load activations for all heads in the group (once per m_chunk).
                // When the source is strided (permuted Q), use 2D DMA to gather
                // contiguous rows into a VTCM scratch buffer first, then HVX
                // converts from the contiguous VTCM buffer.  This avoids L2 cache
                // thrashing from HVX loads at large strides.
                क्रम (पूर्णांक g = 0; g < group_आकार; ++g) {
                    स्थिर dma_addr_t act_dma_addr = hmx_mm_act_batch_addr(params, b2_base + g, b3) +
                                                      (माप_प्रकार) mr * params->act_stride * params->act_elem_आकार;
                    __fp16 *vtcm_act_g = vtcm_f16_act + (माप_प्रकार) g * L.act_शीर्ष_stride;
                    काष्ठा activation_transfer_params act_params = {
                        .ctx = ctx,
                        .dst = vtcm_act_g,
                        .act_dma_addr = act_dma_addr,
                        .n_rows = (पूर्णांक) n_rows,
                        .k_block = params->k,
                        .k_stride = params->act_stride,
                        .n_threads = act_threads,
                        .act_threads_div = act_threads_div,
                        .k_div = k_div,
                        .k_valid = params->k,
                        .vtcm_f32_act = vtcm_f32_act,
                        .vtcm_f32_act_bytes = L.act_f32_bytes,
                        .act_elem_आकार = params->act_elem_आकार,
                    };
                    transfer_activation_chunk_threaded(&act_params);
                }

                // Prologue: Push A0 and A1 (if exists)
                {
                    स्थिर माप_प्रकार n_cols_first = hex_smin((माप_प्रकार) params->n, n_chunk_n_cols);
                    dma_queue_धकेलो(weight_dma, dma_make_तथ्य(vtcm_scratch0, weight_group),
                                      fp16_row_bytes, weight_row_bytes, fp16_row_bytes, n_cols_first);
                }
                अगर (n_chunk_n_cols < (माप_प्रकार) params->n) {
                    स्थिर माप_प्रकार n_cols_second = hex_smin((माप_प्रकार) params->n - n_chunk_n_cols, n_chunk_n_cols);
                    स्थिर dma_addr_t second_weight_chunk = weight_group + n_chunk_n_cols * weight_row_bytes;
                    dma_queue_धकेलो(weight_dma, dma_make_तथ्य(vtcm_scratch1, second_weight_chunk),
                                      fp16_row_bytes, weight_row_bytes, fp16_row_bytes, n_cols_second);
                }

                क्रम (माप_प्रकार nc = 0; nc < (माप_प्रकार) params->n; nc += n_chunk_n_cols) {
                    स्थिर माप_प्रकार n_cols      = hex_smin((माप_प्रकार) params->n - nc, n_chunk_n_cols);
                    स्थिर माप_प्रकार n_col_tiles = hmx_ceil_div((पूर्णांक) n_cols, HTP_MM_HMX_TILE_N_COLS);

                    {
                        व्योम * curr_raw = (व्योम *) dma_queue_निकालो(weight_dma).dst;

                        स्थिर माप_प्रकार n_cols_tiled = hex_align_up(n_cols, HTP_MM_HMX_TILE_N_COLS);

                        hmx_interleave_rows_to_tiles(vtcm_weight, (स्थिर __fp16 *) curr_raw, (uint32_t) n_cols, params->k, params->k, 0, (uint32_t) n_cols_tiled);

                        स्थिर माप_प्रकार nc_अग्र = nc + n_chunk_n_cols * 2;
                        अगर (nc_अग्र < (माप_प्रकार) params->n) {
                            स्थिर माप_प्रकार n_cols_अग्र = hex_smin((माप_प्रकार) params->n - nc_अग्र, n_chunk_n_cols);
                            स्थिर dma_addr_t अग्र_weight_chunk = weight_group + nc_अग्र * weight_row_bytes;

                            dma_queue_धकेलो(weight_dma, dma_make_तथ्य(curr_raw, अग्र_weight_chunk),
                                              fp16_row_bytes, weight_row_bytes, fp16_row_bytes, n_cols_अग्र);
                        }
                    }

                    // Reuse the interleaved weight for every q_head in this GQA group
                    क्रम (पूर्णांक g = 0; g < group_आकार; ++g) {
                        {
                            स्थिर __fp16 * vtcm_act_g = vtcm_f16_act + (माप_प्रकार) g * L.act_शीर्ष_stride;
                            hmx_matmul_job_init(&job, vtcm_output, vtcm_act_g, vtcm_weight, vtcm_scales, n_row_tiles, n_col_tiles, params->k / 32);
                            hmx_queue_धकेलो(ctx->hmx_queue, hmx_queue_make_desc(hmx_matmul_worker_fn, &job));
                            hmx_queue_निकालो(ctx->hmx_queue);
                        }

                        {
                            भग्न *output = hmx_mm_dst_batch_ptr(params, b2_base + g, b3) + mr * params->dst_stride + nc;
                            स्थिर भग्न *bias_chunk = has_bias ? (vtcm_bias + mr * params->bias_stride + nc) : शून्य;
                            पूर्णांक chunk_dst_cols = params->dst_cols - (पूर्णांक)nc;
                            अगर (chunk_dst_cols > 0) {
                                transfer_output_chunk_threaded(ctx, output, bias_chunk, vtcm_output, (पूर्णांक) n_rows, (पूर्णांक) n_cols,
                                                               params->dst_stride, params->bias_stride, chunk_dst_cols, n_threads);
                            }
                        }
                    }
                }
            }
        }
    }

    वापस 0;
}

अटल व्योम transfer_activation_chunk_gathered_threaded(
            काष्ठा htp_context *ctx,
            __fp16 *dst,
            स्थिर भग्न *src,
            पूर्णांक start_row,
            पूर्णांक n_rows,
            पूर्णांक k_block,
            स्थिर काष्ठा mmid_row_mapping *matrix_rows,
            पूर्णांक cur_a,
            पूर्णांक mapping_stride,
            पूर्णांक ne11,
            माप_प्रकार nb11,
            माप_प्रकार nb12,
            पूर्णांक cne1,
            पूर्णांक n_threads,
            पूर्णांक k_valid) {
    अगर (n_rows <= 0) वापस;
    पूर्णांक chunks_per_thread = hmx_ceil_div(n_rows, n_threads);
    chunks_per_thread = hex_align_up(chunks_per_thread, 2);

    पूर्णांक actual_threads = hmx_ceil_div(n_rows, chunks_per_thread);

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
        .ne11_div          = ne11 > 1 ? init_fastdiv_मूल्यावली(ne11) : (काष्ठा fastdiv_मूल्यावली){0, 0},
        .nb11              = nb11,
        .nb12              = nb12,
        .start_row         = start_row,
        .cne1              = cne1,
        .k_valid           = k_valid,
        .traces            = ctx->trace,
    };

    worker_callback_t worker_fn = ne11 == 1 ? transfer_activation_chunk_gathered_worker_flat_fn :
                                              transfer_activation_chunk_gathered_worker_fn;

    अगर (actual_threads <= 1) {
        worker_fn(1, 0, &state);
    } अन्यथा {
        worker_pool_run_func(ctx->worker_pool, worker_fn, &state, actual_threads);
    }
}

अटल व्योम transfer_output_chunk_scattered_threaded(
            काष्ठा htp_context *ctx,
            भग्न *dst,
            स्थिर __fp16 *vtcm_src,
            पूर्णांक start_row,
            पूर्णांक n_rows,
            पूर्णांक n_cols,
            स्थिर काष्ठा mmid_row_mapping *matrix_rows,
            पूर्णांक cur_a,
            पूर्णांक mapping_stride,
            माप_प्रकार dst_nb1,
            माप_प्रकार dst_nb2,
            पूर्णांक cne1,
            पूर्णांक n_threads) {
    अगर (n_rows <= 0) वापस;
    पूर्णांक chunks_per_thread = hmx_ceil_div(n_rows, n_threads);
    chunks_per_thread = hex_align_up(chunks_per_thread, 2);

    पूर्णांक actual_threads = hmx_ceil_div(n_rows, chunks_per_thread);

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

    अगर (actual_threads <= 1) {
        transfer_output_chunk_scattered_worker_fn(1, 0, &state);
    } अन्यथा {
        worker_pool_run_func(ctx->worker_pool, transfer_output_chunk_scattered_worker_fn, &state, actual_threads);
    }
}

अटल पूर्णांक hmx_mm_id_2d_f32(काष्ठा htp_context *ctx,
                                         dma_queue *weight_dma,
                                         भग्न *restrict dst,
                                         स्थिर भग्न *activation,
                                         dma_addr_t weight,
                                         पूर्णांक m, पूर्णांक k, पूर्णांक n,
                                         पूर्णांक k_valid,
                                         पूर्णांक ne11,
                                         माप_प्रकार act_nb1, माप_प्रकार act_nb2,
                                         माप_प्रकार dst_nb1, माप_प्रकार dst_nb2,
                                         पूर्णांक weight_stride,
                                         पूर्णांक weight_type,
                                         पूर्णांक tile_आकार,
                                         पूर्णांक aligned_tile_आकार,
                                         स्थिर काष्ठा mmid_row_mapping *matrix_rows,
                                         पूर्णांक cur_a,
                                         पूर्णांक mapping_stride,
                                         पूर्णांक m_start,
                                         पूर्णांक m_end,
                                         पूर्णांक n_threads) {
    काष्ठा htp_thread_trace * tr = &ctx->trace[0];
    htp_trace_event_start(tr, HTP_TRACE_EVT_INIT, 0);

    स्थिर पूर्णांक cne1 = m;
    स्थिर पूर्णांक m_core = m_end - m_start;
    स्थिर पूर्णांक m_core_padded = hex_align_up(m_core > 0 ? m_core : 1, 32);

    अगर (k % 32 != 0 || n % 32 != 0) { वापस -1; }
    अगर (!hex_is_aligned(dst, VLEN) || !hex_is_aligned(activation, VLEN)) { वापस -1; }

    माप_प्रकार row_stride = htp_tensor_get_row_आकार(weight_type, k);
    अगर (row_stride == 0) {
        वापस -1;
    }

    स्थिर पूर्णांक n_k_tiles = k / HTP_MM_HMX_TILE_N_COLS;
    स्थिर uint32_t n_weight_tiles = k / htp_tensor_get_tile_k(weight_type);
    स्थिर काष्ठा fastdiv_मूल्यावली n_k_tiles_div = init_fastdiv_मूल्यावली(n_k_tiles);

    स्थिर माप_प्रकार vec_dot_आकार = k * माप(__fp16);
    स्थिर माप_प्रकार vtcm_budget  = ctx->vtcm_आकार;
    माप_प्रकार vtcm_used = 0;

    स्थिर bool is_tiled = (tile_आकार > 0);
    tiled_dequant_task_fn_t dequant_worker_fn = hmx_get_dequant_worker_fn(weight_type, is_tiled);

    स्थिर uint32_t dma_dst_stride  = is_tiled ? aligned_tile_आकार : row_stride;
    स्थिर uint32_t dma_src_stride  = is_tiled ? tile_आकार : weight_stride;
    स्थिर uint32_t dma_width_bytes = is_tiled ? tile_आकार : row_stride;

    स्थिर माप_प्रकार qweight_row_stride = is_tiled ? (माप_प्रकार)(n_weight_tiles * aligned_tile_आकार) / 32 : 0;
    स्थिर माप_प्रकार weight_row_stride = is_tiled ? qweight_row_stride : row_stride;

    माप_प्रकार आकार_per_n = 0, आकार_per_m = 0, आकार_per_mn = 0;
    htp_mm_hmx_get_2d_chunk_costs(weight_type, k, /*pipeline=*/false, aligned_tile_आकार,
                                  &आकार_per_n, &आकार_per_m, &आकार_per_mn);

    स्थिर माप_प्रकार overhead = htp_mm_hmx_get_2d_overhead(/*pipeline=*/false, /*is_matmul_id=*/true);
    माप_प्रकार m_chunk_n_rows = 0, n_chunk_n_cols = 0;
    अगर (htp_mm_hmx_compute_chunks(vtcm_budget, overhead, आकार_per_n, आकार_per_m, आकार_per_mn,
                           m_core_padded, n,
                           /*m_block_cost=*/(माप_प्रकार) n * HTP_MM_HMX_COST_W_DEQUANT,
                           /*n_block_cost=*/(माप_प्रकार) m_core_padded * HTP_MM_HMX_COST_A_CONVERT, &m_chunk_n_rows, &n_chunk_n_cols, &vtcm_used)) {
        FARF(ERROR, "hmx-mm-id-2d: VTCM too small : m %d k %d n %d budget %zu", m_core_padded, k, n, vtcm_budget);
        वापस -1;
    }

    स्थिर माप_प्रकार weight_area_आकार = hex_align_up(n_chunk_n_cols * weight_row_stride, HTP_MM_HMX_TILE_आकार);
    स्थिर माप_प्रकार act_area_आकार    = hex_align_up(m_chunk_n_rows * vec_dot_आकार, HTP_MM_HMX_TILE_आकार);
    स्थिर माप_प्रकार output_area_आकार = hex_align_up(m_chunk_n_rows * n_chunk_n_cols * माप(__fp16), HTP_MM_HMX_TILE_आकार);

    माप_प्रकार scratch0_आकार = hex_align_up(n_chunk_n_cols * vec_dot_आकार, HTP_MM_HMX_TILE_आकार);

    uint8_t *vtcm_ptr      = (uint8_t *) ctx->vtcm_base;
    __fp16  *vtcm_weight   = weight_area_आकार ? (__fp16 *) vtcm_seq_alloc(&vtcm_ptr, weight_area_आकार) : शून्य;
    __fp16  *vtcm_f16_act  = (__fp16 *) vtcm_seq_alloc(&vtcm_ptr, act_area_आकार);
    __fp16  *vtcm_output   = (__fp16 *) vtcm_seq_alloc(&vtcm_ptr, output_area_आकार);
    व्योम    *vtcm_scratch0 = vtcm_seq_alloc(&vtcm_ptr, scratch0_आकार);
    __fp16  *vtcm_scales   = (__fp16 *) vtcm_seq_alloc(&vtcm_ptr, 256);

    vtcm_used = vtcm_ptr - (uint8_t *) ctx->vtcm_base;
    अगर (vtcm_used > vtcm_budget) {
        FARF(ERROR, "hmx-mm-id-2d: VTCM overflow: used %zu budget %zu", vtcm_used, vtcm_budget);
        वापस -1;
    }

    hmx_init_column_scales(vtcm_scales, Q6_V_vsplat_R(0x3c00));

    htp_trace_event_stop(tr, HTP_TRACE_EVT_INIT, 0);

    hmx_matmul_job_t job;
    व्योम *vtcm_raw_bufs[2] = { vtcm_weight, dequant_worker_fn ? vtcm_weight : vtcm_scratch0 };
    व्योम *vtcm_weight_bufs[2] = { dequant_worker_fn ? vtcm_scratch0 : vtcm_weight, vtcm_scratch0 };
    स्थिर पूर्णांक n_chunk_cnt = hmx_ceil_div(n, n_chunk_n_cols);

    क्रम (माप_प्रकार mr = (माप_प्रकार) m_start; mr < (माप_प्रकार) m_end; mr += m_chunk_n_rows) {
        स्थिर माप_प्रकार n_rows = hex_smin((माप_प्रकार) m_end - mr, m_chunk_n_rows);
        स्थिर माप_प्रकार n_row_tiles = hmx_ceil_div(n_rows, HTP_MM_HMX_TILE_N_ROWS);

        transfer_activation_chunk_gathered_threaded(
            ctx, vtcm_f16_act, activation, (पूर्णांक) mr, (पूर्णांक) n_rows, k,
            matrix_rows, cur_a, mapping_stride, ne11, act_nb1, act_nb2, cne1, n_threads, k_valid);

        // A0: Pre-fetch the first weight chunk (nc = 0)
        अगर (n > 0) {
            स्थिर माप_प्रकार n_cols = hex_smin((माप_प्रकार) n, n_chunk_n_cols);
            स्थिर uint32_t height = is_tiled ? hmx_ceil_div(n_cols, 32) * n_weight_tiles : n_cols;
            dma_queue_धकेलो(weight_dma, dma_make_तथ्य(vtcm_raw_bufs[0], weight),
                           dma_dst_stride, dma_src_stride, dma_width_bytes, height);
        }

        क्रम (पूर्णांक i = 0; i < n_chunk_cnt; ++i) {
            स्थिर माप_प्रकार nc = (माप_प्रकार) i * n_chunk_n_cols;
            स्थिर माप_प्रकार n_cols = hex_smin((माप_प्रकार) n - nc, n_chunk_n_cols);
            स्थिर माप_प्रकार n_col_tiles = hmx_ceil_div(n_cols, HTP_MM_HMX_TILE_N_COLS);

            // A: Wait for weight DMA
            व्योम * curr_raw = (व्योम *) dma_queue_निकालो(weight_dma).dst;

            // B: Weight Dequantize (Threaded)
            dequantize_tiled_weight_chunk_to_fp16_tiles(
                ctx, vtcm_weight_bufs[i % 2], curr_raw,
                n_cols, k, row_stride, weight_type,
                n_k_tiles, n_k_tiles_div, dequant_worker_fn, n_threads
            );

            // Start weight DMA for the next chunk early
            स्थिर माप_प्रकार nc_अग्र = nc + n_chunk_n_cols;
            अगर (nc_अग्र < (माप_प्रकार) n) {
                स्थिर माप_प्रकार n_cols_अग्र = hex_smin((माप_प्रकार) n - nc_अग्र, n_chunk_n_cols);
                स्थिर uint32_t height_अग्र = is_tiled ? hmx_ceil_div(n_cols_अग्र, 32) * n_weight_tiles : n_cols_अग्र;
                dma_queue_धकेलो(weight_dma, dma_make_तथ्य(vtcm_raw_bufs[(i + 1) % 2], weight + nc_अग्र * weight_stride),
                               dma_dst_stride, dma_src_stride, dma_width_bytes, height_अग्र);
            }

            // C: HMX Compute (Queue-based)
            hmx_matmul_job_init(&job, vtcm_output, vtcm_f16_act, (__fp16 *) vtcm_weight_bufs[i % 2], vtcm_scales, n_row_tiles, n_col_tiles, k / HTP_MM_HMX_TILE_N_ROWS);
            hmx_queue_धकेलो(ctx->hmx_queue, hmx_queue_make_desc(hmx_matmul_worker_fn, &job));
            hmx_queue_निकालो(ctx->hmx_queue);

            // D: Output Store
            transfer_output_chunk_scattered_threaded(
                ctx, dst + nc, vtcm_output, (पूर्णांक) mr, (पूर्णांक) n_rows, (पूर्णांक) n_cols,
                matrix_rows, cur_a, mapping_stride, dst_nb1, dst_nb2, cne1, n_threads);
        }
    }

    वापस 0;
}

// --- Dispatchers and Public Entry Points ---

अटल पूर्णांक hmx_mm_op_matmul(काष्ठा htp_ops_context * octx, स्थिर काष्ठा htp_mm_kernel_params * kparams) {
    htp_matmul_tensors_preamble;

    पूर्णांक k = (पूर्णांक) src0->ne[0];
    पूर्णांक n = (पूर्णांक) src0->ne[1];
    स्थिर पूर्णांक m_योग    = (पूर्णांक) act->ne[1];
    स्थिर uint32_t act_elem_आकार = (act->type == HTP_TYPE_F16) ? माप(__fp16) : माप(भग्न);
    स्थिर पूर्णांक act_stride = (पूर्णांक) (act->nb[1] / act_elem_आकार);

    पूर्णांक m_start = 0;
    पूर्णांक m_rows  = m_योग;
    अगर (octx->ctx->mdev.गिनती > 1) {
        bool can_split = htp_tensor_can_row_partition(dst, माप(भग्न)) &&
                         htp_tensor_can_row_partition(act, act_elem_आकार);
        अगर (src2 && src2->ne[1] > 1 && !htp_tensor_can_row_partition(src2, माप(भग्न))) {
            can_split = false;
        }
        स्थिर काष्ठा htp_tensor_mdev_range range = htp_tensor_mdev_partition((uint32_t) m_योग, can_split ? 1 : 0, octx->ctx->mdev.idx, octx->ctx->mdev.गिनती, &octx->ctx->mdev.गिनती_div);
        m_start = (पूर्णांक) range.start;
        m_rows  = (पूर्णांक) range.गिनती;
    }

    अगर (m_rows == 0) {
        वापस HTP_STATUS_OK;
    }

    dma_addr_t src2_addr = 0;
    माप_प्रकार src2_bytes = 0;
    uint32_t src2_stride = 0;
    माप_प्रकार src2_nb2 = 0;
    माप_प्रकार src2_nb3 = 0;
    अगर (src2) {
        src2_stride = (src2->ne[1] == 1) ? 0 : (uint32_t) (src2->nb[1] / माप(भग्न));
        src2_addr = src2->तथ्य + m_start * src2_stride * माप(भग्न);
        src2_bytes = (माप_प्रकार) kparams->vtcm_bias_आकार;
        src2_nb2 = (src2->ne[2] == 1) ? 0 : src2->nb[2];
        src2_nb3 = (src2->ne[3] == 1) ? 0 : src2->nb[3];
    }

    स्थिर पूर्णांक dst_stride = (पूर्णांक)(dst->nb[1] / माप(भग्न));
    भग्न       * dst_ptr = (भग्न *) dst->तथ्य + m_start * dst_stride;
    // byte offset, act_stride is in activation elements (F16 or F32)
    स्थिर dma_addr_t act_addr = act->तथ्य + (माप_प्रकार) m_start * act_stride * act_elem_आकार;

    पूर्णांक ret = -1;
    स्थिर पूर्णांक n_threads = kparams->n_threads;
    अगर (kparams->kernel_type == HTP_MM_KERNEL_HMX_BATCHED) {
        hmx_mm_batched_params_t batch_params = {
            .dst               = dst_ptr,
            .bias_addr         = src2_addr,
            .bias_bytes        = src2_bytes,
            .act_dma_addr      = act_addr,
            .weight            = src0->तथ्य,
            .weight_dma        = octx->ctx->dma[0],
            .m                 = m_rows,
            .k                 = k,
            .n                 = n,
            .act_stride        = act_stride,
            .act_elem_आकार     = act_elem_आकार,
            .weight_stride     = (पूर्णांक) src0->nb[1],
            .weight_type       = (पूर्णांक) src0->type,
            .dst_stride        = dst_stride,
            .dst_cols          = (पूर्णांक) dst->ne[0],
            .bias_stride       = src2_stride,
            .tile_आकार         = kparams->tile_आकार,
            .aligned_tile_आकार = kparams->aligned_tile_आकार,
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
                             kparams->vtcm_आकार);
    } अन्यथा {
        ret = hmx_mm_2d_f32(
            octx->ctx, octx->ctx->dma[0], dst_ptr, src2_addr, src2_bytes,
            act_addr, src0->तथ्य,
            m_rows, k, n, act_stride, act_elem_आकार, (पूर्णांक) src0->nb[1], (पूर्णांक) src0->type, (पूर्णांक) act->ne[0],
            dst_stride, src2_stride, (पूर्णांक)dst->ne[0],
            kparams->m_chunk, kparams->n_chunk, kparams->pipeline, n_threads,
            kparams->n_act_threads,
            &kparams->div_n_act_threads,
            &kparams->div_ne00_padded,
            kparams->tile_आकार, kparams->aligned_tile_आकार, kparams->vtcm_आकार
        );
    }

    अगर (ret != 0) {
        FARF(ERROR, "HMX matmul failed (ret=%d)\n", ret);
        वापस HTP_STATUS_INTERNAL_ERR;
    }
    वापस HTP_STATUS_OK;
}

अटल अंतरभूत व्योम htp_mm_tensor_collapse_rows(काष्ठा htp_tensor * c, स्थिर काष्ठा htp_tensor * t, uint32_t stride) {
    *c = *t;
    c->ne[1] = t->ne[1] * t->ne[2] * t->ne[3];
    c->ne[2] = 1;
    c->ne[3] = 1;
    c->nb[1] = stride;
    c->nb[2] = c->nb[1] * c->ne[1];
    c->nb[3] = c->nb[2];
}

अटल पूर्णांक op_matmul_impl(काष्ठा htp_ops_context * octx) {
    स्थिर काष्ठा htp_mm_kernel_params * kparams = (स्थिर काष्ठा htp_mm_kernel_params *) octx->kernel_params;

    स्थिर पूर्णांक status = htp_mm_init_context(octx, kparams);
    अगर (status != HTP_STATUS_OK) {
        वापस status;
    }

    अगर (kparams->n_hmx) {
        वापस hmx_mm_op_matmul(octx, kparams);
    }

    वापस hvx_mm_matmul(octx);
}

पूर्णांक op_matmul(काष्ठा htp_ops_context * octx) {
    स्थिर काष्ठा htp_mm_kernel_params * kparams = (स्थिर काष्ठा htp_mm_kernel_params *) octx->kernel_params;

    अगर (kparams->collapse) {
        स्थिर काष्ठा htp_tensor * act  = octx->src[1];
        स्थिर काष्ठा htp_tensor * dst  = octx->dst;
        स्थिर uint32_t s_act = (act->ne[1] > 1) ? act->nb[1] : ((act->ne[2] > 1) ? act->nb[2] : act->nb[3]);
        स्थिर uint32_t sd    = (dst->ne[1] > 1) ? dst->nb[1] : ((dst->ne[2] > 1) ? dst->nb[2] : dst->nb[3]);
        काष्ठा htp_tensor act_collapsed, dst_collapsed;
        htp_mm_tensor_collapse_rows(&act_collapsed, act, s_act);
        htp_mm_tensor_collapse_rows(&dst_collapsed, dst, sd);
        octx->src[1] = &act_collapsed;
        octx->dst    = &dst_collapsed;

        काष्ठा htp_tensor src2_collapsed;
        स्थिर काष्ठा htp_tensor * src2 = octx->src[2];
        अगर (src2 && (src2->ne[1] * src2->ne[2] * src2->ne[3] > 1)) {
            स्थिर uint32_t s2 = (src2->ne[1] > 1) ? src2->nb[1] : ((src2->ne[2] > 1) ? src2->nb[2] : src2->nb[3]);
            htp_mm_tensor_collapse_rows(&src2_collapsed, src2, s2);
            octx->src[2] = &src2_collapsed;
        }

        स्थिर पूर्णांक status = op_matmul_impl(octx);
        octx->src[1] = act;
        octx->dst    = dst;
        अगर (src2) {
            octx->src[2] = src2;
        }
        वापस status;
    }

    वापस op_matmul_impl(octx);
}

अटल पूर्णांक hmx_mm_op_matmul_id(
    काष्ठा htp_ops_context * octx,
    काष्ठा htp_mm_context * mmctx
) {
    स्थिर uint32_t *                matrix_row_counts = mmctx->matrix_row_counts;
    स्थिर काष्ठा mmid_row_mapping * matrix_rows       = mmctx->matrix_rows;
    htp_matmul_tensors_preamble;
    स्थिर काष्ठा htp_mm_kernel_params * kparams = (स्थिर काष्ठा htp_mm_kernel_params *) octx->kernel_params;
    स्थिर पूर्णांक n_ids = octx->src[2]->ne[0];
    स्थिर पूर्णांक n_as  = ne02;

    स्थिर bool mdev_split = (octx->ctx->mdev.गिनती > 1) && htp_tensor_can_row_partition(dst, माप(भग्न));
    अगर (octx->ctx->mdev.गिनती > 1 && !mdev_split && octx->ctx->mdev.idx > 0) {
        वापस HTP_STATUS_OK;
    }
    uint32_t n_active = 0;
    अगर (mdev_split) {
        क्रम (uint32_t a = 0; a < (uint32_t) n_as; ++a) {
            अगर (matrix_row_counts[a] > 0) n_active++;
        }
    }
    स्थिर bool expert_split = mdev_split && (n_active >= octx->ctx->mdev.गिनती);

    uint32_t target_dev = 0;
    क्रम (uint32_t cur_a = 0; cur_a < (uint32_t) n_as; ++cur_a) {
        स्थिर int32_t cne1 = matrix_row_counts[cur_a];
        अगर (cne1 == 0) जारी;

        स्थिर पूर्णांक m_padded = hex_align_up(cne1, 32);
        पूर्णांक m_start = 0, m_end = m_padded;
        अगर (expert_split) {
            स्थिर bool my_expert = (target_dev == octx->ctx->mdev.idx);
            अगर (++target_dev == octx->ctx->mdev.गिनती) {
                target_dev = 0;
            }
            अगर (!my_expert) जारी;
        } अन्यथा अगर (mdev_split) {
            स्थिर bool can_split = (uint32_t) cne1 >= octx->ctx->mdev.गिनती;
            स्थिर काष्ठा htp_tensor_mdev_range range = htp_tensor_mdev_partition((uint32_t) m_padded, can_split ? 1 : 0, octx->ctx->mdev.idx, octx->ctx->mdev.गिनती, &octx->ctx->mdev.गिनती_div);
            m_start = (पूर्णांक) range.start;
            m_end   = (पूर्णांक) (range.start + range.गिनती);
        }
        अगर (m_start >= m_end) जारी;

        पूर्णांक ret = hmx_mm_id_2d_f32(octx->ctx, octx->ctx->dma[0], (भग्न*) dst->तथ्य, (भग्न*) act->तथ्य,
                                   src0->तथ्य + cur_a * nb02,
                                   cne1, ne00, ne01,
                                   ne10,
                                   ne11,
                                   nb11, nb12,
                                   nb1, nb2,
                                   (पूर्णांक) src0->nb[1], (पूर्णांक) src0->type,
                                   kparams->tile_आकार, kparams->aligned_tile_आकार,
                                   matrix_rows, cur_a, mmctx->mapping_stride,
                                   m_start, m_end, (पूर्णांक) octx->n_threads);
        अगर (ret != 0) {
            FARF(ERROR, "HMX matmul failed for expert %u, error %d\n", cur_a, ret);
            वापस HTP_STATUS_NO_SUPPORT;
        }
    }

    वापस HTP_STATUS_OK;
}

अटल पूर्णांक hvx_mm_matmul_id(
    काष्ठा htp_ops_context * octx,
    काष्ठा htp_mm_context * mmctx,
    work_queue_func_t hvx_mmid_task_func
) {
    htp_matmul_tensors_preamble;
    स्थिर uint32_t src0_row_आकार_padded = mmctx->src0_row_आकार_padded;
    स्थिर uint32_t act_nrows            = mmctx->act_nrows;

    काष्ठा htp_thread_trace * tr = &octx->ctx->trace[0];
    htp_trace_event_start(tr, HTP_TRACE_EVT_INIT, 0);

    स्थिर काष्ठा htp_mm_kernel_params * kparams = (स्थिर काष्ठा htp_mm_kernel_params *) octx->kernel_params;
    स्थिर काष्ठा htp_tensor * restrict ids = octx->src[2];
    स्थिर माप_प्रकार src0_row_आकार = nb01;

    स्थिर uint32_t qk = QK_Q8_0_TILED;
    स्थिर uint32_t nb = (ne10 + qk - 1) / qk;
    स्थिर uint32_t योग_nb = act_nrows * nb;

    work_queue_func_t quant_task_func;
    uint32_t n_quant_tasks = 1;
    अगर (act_nrows < octx->n_threads) {
        n_quant_tasks = MIN(योग_nb, octx->n_threads);
        quant_task_func = htp_mm_act_quant_block_func(src0->type, act->type);
        क्रम (uint32_t ith = 0; ith < n_quant_tasks; ++ith) {
            uint32_t ib_first = (योग_nb * ith) / n_quant_tasks;
            uint32_t ib_last  = (योग_nb * (ith + 1)) / n_quant_tasks;
            mmctx->quant_ib_first[ith] = ib_first;
            mmctx->quant_ib_last[ith]  = ib_last;
            mmctx->quant_r[ith]        = ib_first / nb;
            mmctx->quant_c[ith]        = ib_first % nb;
        }
    } अन्यथा {
        n_quant_tasks = MIN(act_nrows, octx->n_threads);
        quant_task_func = htp_mm_act_quant_row_func(src0->type, act->type);
    }
    माप_प्रकार act_row_आकार = htp_mm_tiled_act_row_आकार(src0->type, ne10);

    काष्ठा htp_mm_hvx_vtcm_layout L;
    htp_mm_hvx_vtcm_layout_build(&L, kparams->kernel_type, src0->type, ne10, act_nrows, octx->n_threads,
                                 0, src0_row_आकार, act_row_आकार, 0, kparams->n_prefetch, true, false);

    स्थिर माप_प्रकार vtcm_आकार = L.योग_bytes;

    FARF(ऊपरी_2, "matmul-id-%s : src0-spad-size %zu act-spad-size %zu bias-spad-size 0 dst-spad-size %zu (%zu)\n", mmctx->type,
         L.src0_bytes, L.act_bytes, L.dst_bytes, vtcm_आकार);

    FARF(ऊपरी_2, "matmul-id-%s : %ux%ux%ux%u * %ux%ux%ux%u (%ux%ux%ux%u) -> %ux%ux%ux%u (0x%p, 0x%p, 0x%p)\n", mmctx->type,
         src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3], act->ne[0], act->ne[1], act->ne[2], act->ne[3],
         ids->ne[0], ids->ne[1], ids->ne[2], ids->ne[3], dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3], src0->तथ्य,
         act->तथ्य, dst->तथ्य);

    // Make sure the reserved vtcm size is sufficient
    अगर (octx->ctx->vtcm_आकार < vtcm_आकार) {
        FARF(ERROR, "matmul-id-%s : current VTCM reservation %zu is too small, needed %zu\n", mmctx->type, octx->ctx->vtcm_आकार, vtcm_आकार);
        वापस HTP_STATUS_VTCM_TOO_SMALL;
    }

    uint8_t * स्थिर base = (uint8_t *) octx->ctx->vtcm_base;
    mmctx->vtcm_act      = VTCM_LAYOUT_PTR(uint8_t, base, L.off_act);
    mmctx->vtcm_src0     = VTCM_LAYOUT_PTR(uint8_t, base, L.off_src0);
    mmctx->vtcm_bias     = शून्य;
    mmctx->vtcm_dst      = VTCM_LAYOUT_PTR(uint8_t, base, L.off_dst);
    mmctx->vtcm_act_raw  = VTCM_LAYOUT_PTR(uint8_t, base, L.off_act_raw);

    mmctx->vtcm_src0_stride    = src0_row_आकार_padded;
    mmctx->vtcm_act_stride     = act_row_आकार;
    mmctx->vtcm_act_raw_stride = hex_round_up(ne10 * माप(भग्न), QK_Q8_0_TILED * माप(भग्न));

    mmctx->vtcm_src0_आकार_per_thread = fastdiv(L.src0_bytes, &octx->n_threads_div);
    mmctx->vtcm_act_आकार_per_thread  = L.act_bytes;
    mmctx->vtcm_dst_आकार_per_thread  = fastdiv(L.dst_bytes, &octx->n_threads_div);

    mmctx->cur_m_start = 0;
    mmctx->cur_m_rows  = act_nrows;

    htp_trace_event_stop(tr, HTP_TRACE_EVT_INIT, 0);

    hvx_mm_transfer_act_dma(octx, kparams, act, mmctx->vtcm_act_raw, mmctx->vtcm_act_raw_stride, 0, act_nrows);

    mmctx->n_quant_rows_per_thread = (act_nrows + n_quant_tasks - 1) / n_quant_tasks;
    mmctx->n_quant_tasks = n_quant_tasks;
    work_queue_run(octx->ctx->work_queue, quant_task_func, mmctx, n_quant_tasks);

    work_queue_run(octx->ctx->work_queue, hvx_mmid_task_func, mmctx, octx->n_threads);

    वापस HTP_STATUS_OK;
}

अटल पूर्णांक hmx_mm_op_matmul_id_nx(
    काष्ठा htp_ops_context * octx,
    काष्ठा htp_mm_context * mmctx
) {
    स्थिर uint32_t *                matrix_row_counts = mmctx->matrix_row_counts;
    स्थिर काष्ठा mmid_row_mapping * matrix_rows       = mmctx->matrix_rows;
    स्थिर काष्ठा htp_mm_kernel_params * kparams = (स्थिर काष्ठा htp_mm_kernel_params *) octx->kernel_params;
    स्थिर uint32_t n_weights = kparams->n_weights;
    स्थिर काष्ठा htp_tensor * restrict src0 = octx->src[0];
    स्थिर काष्ठा htp_tensor * restrict act  = octx->src[n_weights];
    स्थिर पूर्णांक n_as = src0->ne[2];

    bool mdev_split = (octx->ctx->mdev.गिनती > 1);
    क्रम (uint32_t p = 0; p < n_weights && mdev_split; ++p) {
        स्थिर काष्ठा htp_tensor * restrict dst = octx->dsts[p];
        mdev_split = htp_tensor_can_row_partition(dst, माप(भग्न));
    }
    अगर (octx->ctx->mdev.गिनती > 1 && !mdev_split && octx->ctx->mdev.idx > 0) {
        वापस HTP_STATUS_OK;
    }
    uint32_t n_active = 0;
    अगर (mdev_split) {
        क्रम (uint32_t a = 0; a < (uint32_t) n_as; ++a) {
            अगर (matrix_row_counts[a] > 0) n_active++;
        }
    }
    स्थिर bool expert_split = mdev_split && (n_active >= octx->ctx->mdev.गिनती);

    uint32_t target_dev = 0;
    क्रम (uint32_t cur_a = 0; cur_a < (uint32_t) n_as; ++cur_a) {
        स्थिर int32_t cne1 = matrix_row_counts[cur_a];
        अगर (cne1 == 0) जारी;

        स्थिर पूर्णांक m_padded = hex_align_up(cne1, 32);
        पूर्णांक m_start = 0, m_end = m_padded;
        अगर (expert_split) {
            स्थिर bool my_expert = (target_dev == octx->ctx->mdev.idx);
            अगर (++target_dev == octx->ctx->mdev.गिनती) {
                target_dev = 0;
            }
            अगर (!my_expert) जारी;
        } अन्यथा अगर (mdev_split) {
            स्थिर bool can_split = (uint32_t) cne1 >= octx->ctx->mdev.गिनती;
            स्थिर काष्ठा htp_tensor_mdev_range range = htp_tensor_mdev_partition((uint32_t) m_padded, can_split ? 1 : 0, octx->ctx->mdev.idx, octx->ctx->mdev.गिनती, &octx->ctx->mdev.गिनती_div);
            m_start = (पूर्णांक) range.start;
            m_end   = (पूर्णांक) (range.start + range.गिनती);
        }
        अगर (m_start >= m_end) जारी;

        क्रम (uint32_t p = 0; p < n_weights; ++p) {
            स्थिर काष्ठा htp_tensor * restrict src_w = octx->src[p];
            स्थिर काष्ठा htp_tensor * restrict dst   = octx->dsts[p];
            अगर (!src_w || !dst) जारी;

            पूर्णांक ret = hmx_mm_id_2d_f32(octx->ctx, octx->ctx->dma[0], (भग्न*) dst->तथ्य, (भग्न*) act->तथ्य,
                                       src_w->तथ्य + cur_a * src_w->nb[2],
                                       cne1, src_w->ne[0], src_w->ne[1],
                                       act->ne[0],
                                       act->ne[1],
                                       act->nb[1], act->nb[2],
                                       dst->nb[1], dst->nb[2],
                                       (पूर्णांक) src_w->nb[1], (पूर्णांक) src_w->type,
                                       kparams->tile_आकार, kparams->aligned_tile_आकार,
                                       matrix_rows, cur_a, mmctx->mapping_stride,
                                       m_start, m_end, (पूर्णांक) octx->n_threads);
            अगर (ret != 0) {
                FARF(ERROR, "HMX matmul ID NX failed for expert %u weight %u, error %d\n", cur_a, p, ret);
                वापस HTP_STATUS_NO_SUPPORT;
            }
        }
    }

    वापस HTP_STATUS_OK;
}

अटल पूर्णांक hvx_mm_matmul_id_nx(
    काष्ठा htp_ops_context * octx,
    काष्ठा htp_mm_context * mmctx,
    work_queue_func_t hvx_mmid_task_func
) {
    स्थिर uint32_t src0_row_आकार_padded = mmctx->src0_row_आकार_padded;
    स्थिर uint32_t act_nrows            = mmctx->act_nrows;

    काष्ठा htp_thread_trace * tr = &octx->ctx->trace[0];
    htp_trace_event_start(tr, HTP_TRACE_EVT_INIT, 0);

    स्थिर काष्ठा htp_mm_kernel_params * kparams = (स्थिर काष्ठा htp_mm_kernel_params *) octx->kernel_params;
    स्थिर uint32_t n_weights = kparams->n_weights;
    स्थिर काष्ठा htp_tensor * restrict src0 = octx->src[0];
    स्थिर काष्ठा htp_tensor * restrict act  = octx->src[n_weights];
    स्थिर काष्ठा htp_tensor * restrict ids  = octx->src[n_weights + 1];
    स्थिर माप_प्रकार src0_row_आकार = src0->nb[1];

    स्थिर uint32_t qk = QK_Q8_0_TILED;
    स्थिर uint32_t nb = (act->ne[0] + qk - 1) / qk;
    स्थिर uint32_t योग_nb = act_nrows * nb;

    work_queue_func_t quant_task_func;
    uint32_t n_quant_tasks = 1;
    अगर (act_nrows < octx->n_threads) {
        n_quant_tasks = MIN(योग_nb, octx->n_threads);
        quant_task_func = htp_mm_act_quant_block_func(src0->type, act->type);
        क्रम (uint32_t ith = 0; ith < n_quant_tasks; ++ith) {
            uint32_t ib_first = (योग_nb * ith) / n_quant_tasks;
            uint32_t ib_last  = (योग_nb * (ith + 1)) / n_quant_tasks;
            mmctx->quant_ib_first[ith] = ib_first;
            mmctx->quant_ib_last[ith]  = ib_last;
            mmctx->quant_r[ith]        = ib_first / nb;
            mmctx->quant_c[ith]        = ib_first % nb;
        }
    } अन्यथा {
        n_quant_tasks = MIN(act_nrows, octx->n_threads);
        quant_task_func = htp_mm_act_quant_row_func(src0->type, act->type);
    }
    माप_प्रकार act_row_आकार = htp_mm_tiled_act_row_आकार(src0->type, act->ne[0]);

    काष्ठा htp_mm_hvx_vtcm_layout L;
    htp_mm_hvx_vtcm_layout_build(&L, kparams->kernel_type, src0->type, act->ne[0], act_nrows, octx->n_threads,
                                 0, src0_row_आकार, act_row_आकार, 0, kparams->n_prefetch, true, false);

    स्थिर माप_प्रकार vtcm_आकार = L.योग_bytes;

    अगर (octx->ctx->vtcm_आकार < vtcm_आकार) {
        FARF(ERROR, "matmul-id-nx: current VTCM reservation %zu is too small, needed %zu\n",
             octx->ctx->vtcm_आकार, vtcm_आकार);
        वापस HTP_STATUS_VTCM_TOO_SMALL;
    }

    uint8_t * स्थिर base = (uint8_t *) octx->ctx->vtcm_base;
    mmctx->vtcm_src0     = VTCM_LAYOUT_PTR(uint8_t, base, L.off_src0);
    mmctx->vtcm_act      = VTCM_LAYOUT_PTR(uint8_t, base, L.off_act);
    mmctx->vtcm_dst      = VTCM_LAYOUT_PTR(uint8_t, base, L.off_dst);
    mmctx->vtcm_act_raw  = VTCM_LAYOUT_PTR(uint8_t, base, L.off_act_raw);

    mmctx->vtcm_src0_stride    = 0;
    mmctx->vtcm_act_stride     = act_row_आकार;
    mmctx->vtcm_act_raw_stride = hex_round_up(act->ne[0] * माप(भग्न), QK_Q8_0_TILED * माप(भग्न));

    mmctx->vtcm_src0_आकार_per_thread = fastdiv(L.src0_bytes, &octx->n_threads_div);
    mmctx->vtcm_act_आकार_per_thread  = L.act_bytes;
    mmctx->vtcm_dst_आकार_per_thread  = fastdiv(L.dst_bytes, &octx->n_threads_div);

    mmctx->cur_m_start = 0;
    mmctx->cur_m_rows  = act_nrows;

    FARF(ऊपरी_2, "matmul-id-nx: src0 %d:%d:%d type %s nrows %u, act %d:%d:%d nrows %u, vtcm %zu/%zu, threads %d\n",
         src0->ne[0], src0->ne[1], src0->ne[2], mmctx->type, src0->ne[1],
         act->ne[0], act->ne[1], act->ne[2], act_nrows,
         L.योग_bytes, octx->ctx->vtcm_आकार, octx->n_threads);

    htp_trace_event_stop(tr, HTP_TRACE_EVT_INIT, 0);

    hvx_mm_transfer_act_dma(octx, kparams, act, mmctx->vtcm_act_raw, mmctx->vtcm_act_raw_stride, 0, act_nrows);

    mmctx->n_quant_rows_per_thread = (act_nrows + n_quant_tasks - 1) / n_quant_tasks;
    mmctx->n_quant_tasks           = n_quant_tasks;
    work_queue_run(octx->ctx->work_queue, quant_task_func, mmctx, n_quant_tasks);

    work_queue_run(octx->ctx->work_queue, hvx_mmid_task_func, mmctx, octx->n_threads);

    वापस HTP_STATUS_OK;
}

अटल अंतरभूत व्योम scan_expert_ids_n(
    स्थिर काष्ठा htp_tensor * ids,
    स्थिर uint32_t n_ids,
    uint32_t n_as,
    uint32_t * counts,
    काष्ठा mmid_row_mapping * matrix_rows,
    uint32_t mapping_stride
) {
    स्थिर माप_प्रकार ids_nb1 = ids->nb[1];
    स्थिर uint8_t * ids_तथ्य = (स्थिर uint8_t *) ids->तथ्य;

    क्रम (uint32_t iid1 = 0; iid1 < ids->ne[1]; ++iid1) {
        स्थिर int32_t * row_ptr = (स्थिर int32_t *) (ids_तथ्य + iid1 * ids_nb1);
        क्रम (uint32_t id = 0; id < n_ids; ++id) {
            स्थिर int32_t i02 = row_ptr[id];
            अगर (i02 < 0) {
                जारी;
            }
            निश्चित(i02 < n_as);

            अगर (matrix_rows) {
                matrix_rows[i02 * mapping_stride + counts[i02]] = (काष्ठा mmid_row_mapping) { id, iid1 };
            }
            counts[i02] += 1;
        }
    }
}

अटल अंतरभूत व्योम scan_expert_ids(
    स्थिर काष्ठा htp_tensor * ids,
    uint32_t n_ids,
    uint32_t n_as,
    uint32_t * counts,
    काष्ठा mmid_row_mapping * matrix_rows,
    uint32_t mapping_stride
) {
    स्थिर माप_प्रकार ids_nb0 = ids->nb[0];

    अगर (ids_nb0 == 4) {
        चयन (n_ids) {
            हाल 8:  scan_expert_ids_n(ids, 8,     n_as, counts, matrix_rows, mapping_stride); अवरोध;
            हाल 4:  scan_expert_ids_n(ids, 4,     n_as, counts, matrix_rows, mapping_stride); अवरोध;
            हाल 2:  scan_expert_ids_n(ids, 2,     n_as, counts, matrix_rows, mapping_stride); अवरोध;
            शेष: scan_expert_ids_n(ids, n_ids, n_as, counts, matrix_rows, mapping_stride); अवरोध;
        }
    } अन्यथा {
        // Strided fallback
        स्थिर माप_प्रकार ids_nb1 = ids->nb[1];
        स्थिर uint8_t * ids_तथ्य = (स्थिर uint8_t *) ids->तथ्य;
        क्रम (uint32_t iid1 = 0; iid1 < ids->ne[1]; ++iid1) {
            स्थिर int32_t * row_ptr = (स्थिर int32_t *) (ids_तथ्य + iid1 * ids_nb1);
            क्रम (uint32_t id = 0; id < n_ids; ++id) {
                स्थिर int32_t i02 = *(स्थिर int32_t *) ((स्थिर uint8_t *) row_ptr + id * ids_nb0);
                अगर (i02 < 0) {
                    जारी;
                }
                निश्चित(i02 < n_as);

                अगर (matrix_rows) {
                    matrix_rows[i02 * mapping_stride + counts[i02]] = (काष्ठा mmid_row_mapping) { id, iid1 };
                }
                counts[i02] += 1;
            }
        }
    }
}

पूर्णांक op_matmul_id(काष्ठा htp_ops_context * octx) {
    htp_matmul_tensors_preamble;

    स्थिर काष्ठा htp_mm_kernel_params * kparams = (स्थिर काष्ठा htp_mm_kernel_params *) octx->kernel_params;
    काष्ठा htp_mm_context mmctx_struct = {0};
    काष्ठा htp_mm_context * mmctx = &mmctx_struct;

    स्थिर पूर्णांक status = htp_mm_init_context(octx, kparams);
    अगर (status != HTP_STATUS_OK) {
        वापस status;
    }

    काष्ठा htp_thread_trace * tr = &octx->ctx->trace[0];
    htp_trace_event_start(tr, HTP_TRACE_EVT_INIT, 0);

    mmctx->octx = octx;
    mmctx->act = act;

    स्थिर काष्ठा htp_tensor * restrict ids = octx->src[2];
    अगर (htp_tensor_is_extended(ids) || htp_tensor_is_extended(act) || htp_tensor_is_extended(dst)) {
        वापस HTP_STATUS_NO_SUPPORT;
    }

    स्थिर माप_प्रकार src0_row_आकार = nb01;
    स्थिर माप_प्रकार dst_row_आकार  = nb1;

    स्थिर माप_प्रकार src0_row_आकार_padded = hex_round_up(src0_row_आकार, 128);

    स्थिर uint32_t src0_nrows = ne01;  // per expert
    स्थिर uint32_t act_nrows  = ne11 * ne12 * ne13;

    // row groups
    स्थिर पूर्णांक n_ids = ids->ne[0];  // n_expert_used
    स्थिर पूर्णांक n_as  = ne02;        // n_expert

    uint8_t  * mapping_buf       = octx->ctx->ddr_spad_base;
    uint32_t   mapping_stride    = 1;
    uint32_t * matrix_row_counts = (uint32_t *) mapping_buf;
    काष्ठा mmid_row_mapping * matrix_rows = शून्य;

    अगर (act_nrows > 1) {
        स्थिर माप_प्रकार matrix_row_counts_आकार = n_as * माप(uint32_t);
        निश्चित(octx->ctx->ddr_spad_आकार >= matrix_row_counts_आकार);

        hex_l2fetch_block((स्थिर व्योम *) ids->तथ्य, ids->ne[1] * ids->nb[1]);

        स_रखो(matrix_row_counts, 0, matrix_row_counts_आकार);
        scan_expert_ids(ids, n_ids, n_as, matrix_row_counts, शून्य, 0);

        uint32_t max_गिनती = hvx_reduce_max_i32((स्थिर uint8_t *) matrix_row_counts, n_as);
        mapping_stride = max_गिनती > 0 ? max_गिनती : 1;

        माप_प्रकार matrix_row_map_आकार  = n_as * mapping_stride * माप(काष्ठा mmid_row_mapping);
        स्थिर माप_प्रकार योग_map_आकार = matrix_row_counts_आकार + matrix_row_map_आकार;

        अगर (योग_map_आकार > octx->ctx->ddr_spad_आकार) {
            mapping_buf = memalign(128, योग_map_आकार);
            अगर (!mapping_buf) {
                वापस HTP_STATUS_INTERNAL_ERR;
            }
        }

        matrix_row_counts = (uint32_t *) mapping_buf;
        matrix_rows       = (काष्ठा mmid_row_mapping *) (mapping_buf + matrix_row_counts_आकार);

        स_रखो(matrix_row_counts, 0, n_as * माप(uint32_t));
        scan_expert_ids(ids, n_ids, n_as, matrix_row_counts, matrix_rows, mapping_stride);
    }

    mmctx->matrix_row_counts    = matrix_row_counts;
    mmctx->matrix_rows          = matrix_rows;
    mmctx->mapping_stride       = mapping_stride;
    mmctx->mm_div_ne11          = kparams->div_ne1;
    mmctx->src0_row_आकार_padded = src0_row_आकार_padded;
    mmctx->act_nrows            = act_nrows;
    mmctx->cur_m_start          = 0;
    mmctx->cur_m_rows           = act_nrows;

    htp_trace_event_stop(tr, HTP_TRACE_EVT_INIT, 0);

    पूर्णांक s;
    अगर (kparams->n_hmx) {
        s = hmx_mm_op_matmul_id(octx, mmctx);
    } अन्यथा {
        uint32_t src0_row_start = 0;
        uint32_t src0_row_end   = src0_nrows;
        अगर (octx->ctx->mdev.गिनती > 1) {
            स्थिर bool can_split = htp_tensor_can_row_partition(dst, माप(भग्न));
            स्थिर काष्ठा htp_tensor_mdev_range range = htp_tensor_mdev_partition(src0_nrows, can_split ? 32 : 0, octx->ctx->mdev.idx, octx->ctx->mdev.गिनती, &octx->ctx->mdev.गिनती_div);
            src0_row_start = range.start;
            src0_row_end   = range.start + range.गिनती;
        }

        अगर (src0_row_start >= src0_row_end) {
            अगर (mapping_buf != octx->ctx->ddr_spad_base) {
                मुक्त(mapping_buf);
            }
            वापस HTP_STATUS_OK;
        }

        स्थिर uint32_t nrows = src0_row_end - src0_row_start;
        mmctx->src0_row_start = src0_row_start;
        mmctx->src0_row_end   = src0_row_end;

        mmctx->src0_nrows_per_thread = fastdiv(nrows + octx->n_threads - 1, &octx->n_threads_div);
        mmctx->src0_nrows_per_thread = hex_round_up(mmctx->src0_nrows_per_thread, 32);

        अगर (hvx_mm_init_tiled_vec_dot(mmctx, src0->type) == 0) {
            s = hvx_mm_matmul_id(octx, mmctx, act_nrows > 1 ? hvx_mm_id_tiled : hvx_mv_id_tiled);
        } अन्यथा {
            s = HTP_STATUS_NO_SUPPORT;
        }
    }

    अगर (mapping_buf != octx->ctx->ddr_spad_base) {
        मुक्त(mapping_buf);
    }

    वापस s;
}

पूर्णांक op_matmul_id_nx(काष्ठा htp_ops_context * octx) {
    स्थिर काष्ठा htp_mm_kernel_params * kparams = (स्थिर काष्ठा htp_mm_kernel_params *) octx->kernel_params;
    काष्ठा htp_mm_context mmctx_struct = {0};
    काष्ठा htp_mm_context * mmctx = &mmctx_struct;

    स्थिर पूर्णांक status = htp_mm_init_context(octx, kparams);
    अगर (status != HTP_STATUS_OK) {
        वापस status;
    }

    काष्ठा htp_thread_trace * tr = &octx->ctx->trace[0];
    htp_trace_event_start(tr, HTP_TRACE_EVT_INIT, 0);

    mmctx->octx = octx;
    स्थिर uint32_t n_weights = kparams->n_weights;
    स्थिर काष्ठा htp_tensor * restrict src0 = octx->src[0];
    स्थिर काष्ठा htp_tensor * restrict act  = octx->src[n_weights];
    स्थिर काष्ठा htp_tensor * restrict ids  = octx->src[n_weights + 1];
    अगर (htp_tensor_is_extended(ids) || htp_tensor_is_extended(act)) {
        वापस HTP_STATUS_NO_SUPPORT;
    }
    क्रम (uint32_t p = 0; p < n_weights; p++) {
        अगर (htp_tensor_is_extended(octx->dsts[p])) {
            वापस HTP_STATUS_NO_SUPPORT;
        }
    }

    mmctx->act = act;

    स्थिर माप_प्रकार src0_row_आकार = src0->nb[1];
    स्थिर माप_प्रकार src0_row_आकार_padded = hex_round_up(src0_row_आकार, 128);

    स्थिर uint32_t act_nrows = act->ne[1] * act->ne[2] * act->ne[3];

    स्थिर पूर्णांक n_ids = ids->ne[0];
    स्थिर पूर्णांक n_as  = src0->ne[2];

    uint8_t  * mapping_buf       = octx->ctx->ddr_spad_base;
    uint32_t   mapping_stride    = 1;
    uint32_t * matrix_row_counts = (uint32_t *) mapping_buf;
    काष्ठा mmid_row_mapping * matrix_rows = शून्य;

    अगर (act_nrows > 1) {
        स्थिर माप_प्रकार matrix_row_counts_आकार = n_as * माप(uint32_t);
        निश्चित(octx->ctx->ddr_spad_आकार >= matrix_row_counts_आकार);

        hex_l2fetch_block((स्थिर व्योम *) ids->तथ्य, ids->ne[1] * ids->nb[1]);

        स_रखो(matrix_row_counts, 0, matrix_row_counts_आकार);
        scan_expert_ids(ids, n_ids, n_as, matrix_row_counts, शून्य, 0);

        uint32_t max_गिनती = hvx_reduce_max_i32((स्थिर uint8_t *) matrix_row_counts, n_as);
        mapping_stride = max_गिनती > 0 ? max_गिनती : 1;

        माप_प्रकार matrix_row_map_आकार  = n_as * mapping_stride * माप(काष्ठा mmid_row_mapping);
        स्थिर माप_प्रकार योग_map_आकार = matrix_row_counts_आकार + matrix_row_map_आकार;

        अगर (योग_map_आकार > octx->ctx->ddr_spad_आकार) {
            mapping_buf = memalign(128, योग_map_आकार);
            अगर (!mapping_buf) {
                वापस HTP_STATUS_INTERNAL_ERR;
            }
        }

        matrix_row_counts = (uint32_t *) mapping_buf;
        matrix_rows       = (काष्ठा mmid_row_mapping *) (mapping_buf + matrix_row_counts_आकार);

        स_रखो(matrix_row_counts, 0, n_as * माप(uint32_t));
        scan_expert_ids(ids, n_ids, n_as, matrix_row_counts, matrix_rows, mapping_stride);
    }

    mmctx->matrix_row_counts    = matrix_row_counts;
    mmctx->matrix_rows          = matrix_rows;
    mmctx->mapping_stride       = mapping_stride;
    mmctx->mm_div_ne11          = kparams->div_ne1;
    mmctx->src0_row_आकार_padded = src0_row_आकार_padded;
    mmctx->act_nrows            = act_nrows;
    mmctx->cur_m_start          = 0;
    mmctx->cur_m_rows           = act_nrows;

    htp_trace_event_stop(tr, HTP_TRACE_EVT_INIT, 0);

    पूर्णांक s;
    अगर (kparams->n_hmx) {
        s = hmx_mm_op_matmul_id_nx(octx, mmctx);
    } अन्यथा {
        अगर (hvx_mm_init_tiled_vec_dot(mmctx, src0->type) == 0) {
            s = hvx_mm_matmul_id_nx(octx, mmctx, act_nrows > 1 ? hvx_mm_id_nx_tiled : hvx_mv_id_nx_tiled);
        } अन्यथा {
            s = HTP_STATUS_NO_SUPPORT;
        }
    }

    अगर (mapping_buf != octx->ctx->ddr_spad_base) {
        मुक्त(mapping_buf);
    }

    वापस s;
}
अटल पूर्णांक op_matmul_nx_impl(काष्ठा htp_ops_context * octx) {
    स्थिर काष्ठा htp_mm_kernel_params * kparams = (स्थिर काष्ठा htp_mm_kernel_params *) octx->kernel_params;

    स्थिर पूर्णांक status = htp_mm_init_context(octx, kparams);
    अगर (status != HTP_STATUS_OK) {
        वापस status;
    }

    अगर (kparams->n_hmx) {
        वापस hmx_mm_nx_2d_f32(octx, kparams);
    }

    काष्ठा htp_thread_trace * tr = &octx->ctx->trace[0];
    htp_trace_event_start(tr, HTP_TRACE_EVT_INIT, 0);

    स्थिर uint32_t n_weights = kparams->n_weights;

    स्थिर काष्ठा htp_tensor * restrict src0 = octx->src[0]; // first weight
    स्थिर काष्ठा htp_tensor * restrict act  = octx->src[n_weights]; // activation x

    काष्ठा htp_mm_context mmctx_struct = {0};
    काष्ठा htp_mm_context * mmctx = &mmctx_struct;
    mmctx->octx = octx;
    mmctx->act  = act;

    स्थिर uint32_t act_nrows = act->ne[1] * act->ne[2] * act->ne[3];
    mmctx->act_nrows   = act_nrows;
    mmctx->cur_m_start = 0;
    mmctx->cur_m_rows  = act_nrows;

    स्थिर माप_प्रकार src0_row_आकार = src0->nb[1];

    अगर (hvx_mm_init_tiled_vec_dot(mmctx, src0->type) != 0) {
        वापस HTP_STATUS_NO_SUPPORT;
    }

    स्थिर uint32_t qk = QK_Q8_0_TILED;
    स्थिर uint32_t nb = (act->ne[0] + qk - 1) / qk;
    स्थिर uint32_t योग_nb = act_nrows * nb;

    worker_callback_t quant_task_func;
    uint32_t n_quant_tasks = 1;
    अगर (act_nrows < octx->n_threads) {
        n_quant_tasks = MIN(योग_nb, octx->n_threads);
        quant_task_func = htp_mm_act_quant_block_func(src0->type, act->type);
        क्रम (uint32_t ith = 0; ith < n_quant_tasks; ++ith) {
            uint32_t ib_first = (योग_nb * ith) / n_quant_tasks;
            uint32_t ib_last  = (योग_nb * (ith + 1)) / n_quant_tasks;
            mmctx->quant_ib_first[ith] = ib_first;
            mmctx->quant_ib_last[ith]  = ib_last;
            mmctx->quant_r[ith]        = ib_first / nb;
            mmctx->quant_c[ith]        = ib_first % nb;
        }
    } अन्यथा {
        n_quant_tasks = MIN(act_nrows, octx->n_threads);
        quant_task_func = htp_mm_act_quant_row_func(src0->type, act->type);
    }

    स्थिर माप_प्रकार act_row_आकार = htp_mm_tiled_act_row_आकार(src0->type, act->ne[0]);

    काष्ठा htp_mm_hvx_vtcm_layout L;
    htp_mm_hvx_vtcm_layout_build(&L, kparams->kernel_type, src0->type, act->ne[0], act_nrows, octx->n_threads,
                                 0, src0_row_आकार, act_row_आकार, 0, kparams->n_prefetch, false, true);

    स्थिर माप_प्रकार vtcm_आकार = L.योग_bytes;

    अगर (octx->ctx->vtcm_आकार < vtcm_आकार) {
        FARF(ERROR, "matmul-nx: current VTCM reservation %zu is too small, needed %zu\n",
             octx->ctx->vtcm_आकार, vtcm_आकार);
        वापस HTP_STATUS_VTCM_TOO_SMALL;
    }

    uint8_t * स्थिर base = (uint8_t *) octx->ctx->vtcm_base;
    mmctx->vtcm_src0     = VTCM_LAYOUT_PTR(uint8_t, base, L.off_src0);
    mmctx->vtcm_act      = VTCM_LAYOUT_PTR(uint8_t, base, L.off_act);
    mmctx->vtcm_dst      = VTCM_LAYOUT_PTR(uint8_t, base, L.off_dst);
    mmctx->vtcm_act_raw  = VTCM_LAYOUT_PTR(uint8_t, base, L.off_act_raw);

    mmctx->vtcm_src0_stride    = 0;
    mmctx->vtcm_act_stride     = act_row_आकार;
    mmctx->vtcm_act_raw_stride = hex_round_up(act->ne[0] * माप(भग्न), QK_Q8_0_TILED * माप(भग्न));

    mmctx->vtcm_src0_आकार_per_thread = fastdiv(L.src0_bytes, &octx->n_threads_div);
    mmctx->vtcm_act_आकार_per_thread  = L.act_bytes;
    mmctx->vtcm_dst_आकार_per_thread  = fastdiv(L.dst_bytes, &octx->n_threads_div);

    htp_trace_event_stop(tr, HTP_TRACE_EVT_INIT, 0);

    hvx_mm_transfer_act_dma(octx, kparams, act, mmctx->vtcm_act_raw, mmctx->vtcm_act_raw_stride, 0, act_nrows);

    mmctx->n_quant_rows_per_thread = (act_nrows + n_quant_tasks - 1) / n_quant_tasks;
    mmctx->n_quant_tasks = n_quant_tasks;
    work_queue_run(octx->ctx->work_queue, quant_task_func, mmctx, n_quant_tasks);

    work_queue_run(octx->ctx->work_queue, hvx_mm_nx_2d_tiled, mmctx, octx->n_threads);

    वापस HTP_STATUS_OK;
}

पूर्णांक op_matmul_nx(काष्ठा htp_ops_context * octx) {
    स्थिर काष्ठा htp_mm_kernel_params * kparams = (स्थिर काष्ठा htp_mm_kernel_params *) octx->kernel_params;

    अगर (kparams->collapse) {
        स्थिर uint32_t n_weights = kparams->n_weights;
        स्थिर काष्ठा htp_tensor * act = octx->src[n_weights];
        स्थिर uint32_t s1 = (act->ne[1] > 1) ? act->nb[1] : ((act->ne[2] > 1) ? act->nb[2] : act->nb[3]);
        काष्ठा htp_tensor act_collapsed;
        काष्ठा htp_tensor dsts_collapsed[HTP_OP_MAX_OUTPUTS];
        स्थिर काष्ठा htp_tensor * orig_dsts[HTP_OP_MAX_OUTPUTS];

        htp_mm_tensor_collapse_rows(&act_collapsed, act, s1);
        octx->src[n_weights] = &act_collapsed;

        क्रम (uint32_t p = 0; p < n_weights; p++) {
            orig_dsts[p] = octx->dsts[p];
            स्थिर काष्ठा htp_tensor * d = octx->dsts[p];
            स्थिर uint32_t sd = (d->ne[1] > 1) ? d->nb[1] : ((d->ne[2] > 1) ? d->nb[2] : d->nb[3]);
            htp_mm_tensor_collapse_rows(&dsts_collapsed[p], d, sd);
            octx->dsts[p] = &dsts_collapsed[p];
        }

        स्थिर पूर्णांक status = op_matmul_nx_impl(octx);

        octx->src[n_weights] = act;
        क्रम (uint32_t p = 0; p < n_weights; p++) {
            octx->dsts[p] = orig_dsts[p];
        }
        वापस status;
    }

    वापस op_matmul_nx_impl(octx);
}
