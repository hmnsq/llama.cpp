#pragma clang diagnostic ignored "-Wgnu-zero-variadic-macro-arguments"
#pragma clang diagnostic ignored "-Wunused-function"
#pragma clang diagnostic ignored "-Wunused-variable"
#pragma clang diagnostic ignored "-Wunused-but-set-variable"

#ଅନ୍ତର୍ଭାଵ <HAP_farf.h>
#ଅନ୍ତର୍ଭାଵ <HAP_perf.h>
#ଅନ୍ତର୍ଭାଵ <HAP_compute_res.h>

#ଅନ୍ତର୍ଭାଵ <math.h>
#ଅନ୍ତର୍ଭାଵ <string.h>
#ଅନ୍ତର୍ଭାଵ <stdatomic.h>

#ଅନ୍ତର୍ଭାଵ "dma-queue.h"
#ଅନ୍ତର୍ଭାଵ "hvx-utils.h"
#ଅନ୍ତର୍ଭାଵ "hvx-dump.h"
#ଅନ୍ତର୍ଭାଵ "hvx-arith.h"
#ଅନ୍ତର୍ଭାଵ "hvx-reduce.h"

#define GGML_COMMON_DECL_C
#ଅନ୍ତର୍ଭାଵ "ggml-common.h"
#ଅନ୍ତର୍ଭାଵ "htp-ctx.h"
#ଅନ୍ତର୍ଭାଵ "htp-ops.h"
#ଅନ୍ତର୍ଭାଵ "htp-tensor.h"
#ଅନ୍ତର୍ଭାଵ "matmul-ops.h"
#ଅନ୍ତର୍ଭାଵ "htp-vtcm.h"

ପ୍ରକାରବ୍ୟାଖ୍ୟା ସଂରଚନା {
    ଦଶମିକ        *dst;
    dma_addr_t    bias_addr;
    size_t        bias_bytes;
    dma_addr_t    act_dma_addr;
    dma_addr_t    weight;
    dma_queue *   weight_dma;
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା           m;
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା           k;
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା           n;
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା           act_stride;
    uint32_t      act_elem_size;  // 4=F32 src1, 2=F16 src1
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା           weight_stride;
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା           weight_type;
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା           dst_stride;
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା           dst_cols;
    uint32_t      bias_stride;
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା           tile_size;
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା           aligned_tile_size;
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା           ne02;
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା           ne03;
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା           ne12;
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା           ne13;
    size_t        src0_nb2;
    size_t        src0_nb3;
    size_t        act_nb2;
    size_t        act_nb3;
    size_t        bias_nb2;
    size_t        bias_nb3;
    size_t        dst_nb2;
    size_t        dst_nb3;
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା           r2;
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା           r3;
    ସଂରଚନା fastdiv_ମୂଲ୍ୟଗୁଡ଼ିକ div_r2;
    ସଂରଚନା fastdiv_ମୂଲ୍ୟଗୁଡ଼ିକ div_r3;
} hmx_mm_batched_params_t;

ସ୍ଥିର bool htp_matmul_has_extended_weight(ଅଚଳ ସଂରଚନା htp_ops_context * octx, uint32_t n_weights) {
    ପାଇଁ (uint32_t i = 0; i < n_weights; ++i) {
        ଯଦି (htp_tensor_is_extended(octx->src[i])) {
            ଫେରନ୍ତୁ true;
        }
    }
    ଫେରନ୍ତୁ false;
}

ସଂରଚନା htp_mm_context {
    ଅଚଳ ଅକ୍ଷର * type;
    ସଂରଚନା htp_ops_context * octx;
    ଅଚଳ ସଂରଚନା htp_tensor * act;

    ଶୂନ୍ୟ (*vec_dot_1x1)(ଅଚଳ uint32_t n, ଦଶମିକ * restrict s0,
         ଅଚଳ ଶୂନ୍ୟ * restrict vx0,
         ଅଚଳ ଶୂନ୍ୟ * restrict vy0);

    ଶୂନ୍ୟ (*vec_dot_2x1)(ଅଚଳ uint32_t n, ଦଶମିକ * restrict s0,
         ଅଚଳ ଶୂନ୍ୟ * restrict vx0, ଅଚଳ ଶୂନ୍ୟ * restrict vx1,
         ଅଚଳ ଶୂନ୍ୟ * restrict vy0);

    ଶୂନ୍ୟ (*vec_dot_2x2)(ଅଚଳ uint32_t n, ଦଶମିକ * restrict s0, ଦଶମିକ * restrict s1,
         ଅଚଳ ଶୂନ୍ୟ * restrict vx0, ଅଚଳ ଶୂନ୍ୟ * restrict vx1,
         ଅଚଳ ଶୂନ୍ୟ * restrict vy0, ଅଚଳ ଶୂନ୍ୟ * restrict vy1);

    ଶୂନ୍ୟ (*vec_dot_32x1)(ଅଚଳ uint32_t n, ଦଶମିକ * restrict s,
         ଅଚଳ ଶୂନ୍ୟ * restrict vx,
         ଅଚଳ ଶୂନ୍ୟ * restrict vy, uint32_t valid_rows,
         ଅଚଳ ଦଶମିକ * restrict sz);

    ଶୂନ୍ୟ (*vec_dot_32x2)(ଅଚଳ uint32_t n, ଦଶମିକ * restrict s0, ଦଶମିକ * restrict s1,
         ଅଚଳ ଶୂନ୍ୟ * restrict vx,
         ଅଚଳ ଶୂନ୍ୟ * restrict vy0, ଅଚଳ ଶୂନ୍ୟ * restrict vy1, uint32_t valid_rows,
         ଅଚଳ ଦଶମିକ * restrict sz0, ଅଚଳ ଦଶମିକ * restrict sz1);

    // Precomputed values
    uint32_t src0_nrows_per_thread;
    uint32_t src0_row_start;
    uint32_t src0_row_end;
    uint32_t src0_row_size_padded;
    uint32_t act_nrows;
    uint32_t cur_m_start;
    uint32_t cur_m_rows;

    ସଂରଚନା fastdiv_ମୂଲ୍ୟଗୁଡ଼ିକ mm_div_ne12_ne1;
    ସଂରଚନା fastdiv_ମୂଲ୍ୟଗୁଡ଼ିକ mm_div_ne1;
    ସଂରଚନା fastdiv_ମୂଲ୍ୟଗୁଡ଼ିକ mm_div_r2;
    ସଂରଚନା fastdiv_ମୂଲ୍ୟଗୁଡ଼ିକ mm_div_r3;
    ସଂରଚନା fastdiv_ମୂଲ୍ୟଗୁଡ଼ିକ mm_div_ne11;

    // Precomputed block-parallel quantization values
    uint32_t          quant_ib_first[WORK_QUEUE_MAX_N_THREADS];
    uint32_t          quant_ib_last[WORK_QUEUE_MAX_N_THREADS];
    uint32_t          quant_r[WORK_QUEUE_MAX_N_THREADS];
    uint32_t          quant_c[WORK_QUEUE_MAX_N_THREADS];
    uint32_t          n_quant_tasks;
    uint32_t          n_quant_rows_per_thread;

    // Fields for scattered mapping & HMX support in MUL_MAT_ID
    ଅଚଳ uint32_t * matrix_row_counts;
    ଅଚଳ ସଂରଚନା mmid_row_mapping * matrix_rows;
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

ସ୍ଥିର ପୂର୍ଣ୍ଣସଂଖ୍ୟା htp_mm_init_context(
    ସଂରଚନା htp_ops_context * octx,
    ଅଚଳ ସଂରଚନା htp_mm_kernel_params * kparams
) {
    ଯଦି (!htp_ops_context_set_n_threads(octx, (uint32_t) kparams->n_threads)) {
        ଫେରନ୍ତୁ HTP_STATUS_INVAL_PARAMS;
    }

    ଯଦି (kparams->n_hmx) {
        ଯଦି (kparams->n_act_threads <= 0 || kparams->n_act_threads > (int32_t) octx->n_threads) {
            ଫେରନ୍ତୁ HTP_STATUS_INVAL_PARAMS;
        }
    }

    ଫେରନ୍ତୁ HTP_STATUS_OK;
}

// vdelta control to expand first 32 e8m0 values into 32 uint32 elements
ସ୍ଥିର ଅଚଳ uint8_t __attribute__((aligned(128))) expand_x32_e8m0[128] = {
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
ସ୍ଥିର ଅଚଳ uint8_t __attribute__((aligned(VLEN))) kvalues_iq4nl_lut[] = {
    0x81, 0, 0x98, 0, 0xAD, 0, 0xBF, 0, 0xCF, 0, 0xDD, 0, 0xEA, 0, 0xF6, 0, 0x01, 0, 0x0D, 0, 0x19, 0, 0x26, 0,
    0x35, 0, 0x45, 0, 0x59, 0, 0x71, 0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0,
    0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0,
    0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0,
    0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0,
};

ସ୍ଥିର ଅଚଳ uint8_t __attribute__((aligned(VLEN))) kvalues_mxfp4_lut[] = {
    0,    0, 1,    0, 2,    0, 3, 0, 4, 0, 6, 0, 8, 0, 12, 0, 0, 0, 0xff, 0, 0xfe, 0, 0xfd, 0, 0xfc, 0,
    0xfa, 0, 0xf8, 0, 0xf4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 0,    0, 0,    0, 0,    0, 0,    0,
    0,    0, 0,    0, 0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 0,    0, 0,    0, 0,    0, 0,    0,
    0,    0, 0,    0, 0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 0,    0, 0,    0, 0,    0, 0,    0,
    0,    0, 0,    0, 0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 0,    0, 0,    0, 0,    0,
};

#define htp_matmul_tensors_preamble                                 \
    ଅଚଳ ସଂରଚନା htp_tensor * restrict src0 = octx->src[0];         \
    ଅଚଳ ସଂରଚନା htp_tensor * restrict src1 = octx->src[1];         \
    ଅଚଳ ସଂରଚନା htp_tensor * restrict src2 = octx->src[2];         \
    ଅଚଳ ସଂରଚନା htp_tensor * restrict  dst = octx->dst;            \
    ଅଚଳ ସଂରଚନା htp_tensor * restrict  act = src1;                 \
                                                                    \
    ଅଚଳ uint32_t ne00 = src0->ne[0];                              \
    ଅଚଳ uint32_t ne01 = src0->ne[1];                              \
    ଅଚଳ uint32_t ne02 = src0->ne[2];                              \
    ଅଚଳ uint32_t ne03 = src0->ne[3];                              \
                                                                    \
    ଅଚଳ uint32_t ne10 = src1->ne[0];                              \
    ଅଚଳ uint32_t ne11 = src1->ne[1];                              \
    ଅଚଳ uint32_t ne12 = src1->ne[2];                              \
    ଅଚଳ uint32_t ne13 = src1->ne[3];                              \
                                                                    \
    ଅଚଳ uint32_t ne20 = src2 ? src2->ne[0] : 0;                   \
    ଅଚଳ uint32_t ne21 = src2 ? src2->ne[1] : 0;                   \
    ଅଚଳ uint32_t ne22 = src2 ? src2->ne[2] : 0;                   \
    ଅଚଳ uint32_t ne23 = src2 ? src2->ne[3] : 0;                   \
                                                                    \
    ଅଚଳ uint32_t ne0 = dst->ne[0];                                \
    ଅଚଳ uint32_t ne1 = dst->ne[1];                                \
    ଅଚଳ uint32_t ne2 = dst->ne[2];                                \
    ଅଚଳ uint32_t ne3 = dst->ne[3];                                \
                                                                    \
    ଅଚଳ uint32_t nb00 = src0->nb[0];                              \
    ଅଚଳ uint32_t nb01 = src0->nb[1];                              \
    ଅଚଳ uint32_t nb02 = src0->nb[2];                              \
    ଅଚଳ uint32_t nb03 = src0->nb[3];                              \
                                                                    \
    ଅଚଳ uint32_t nb10 = src1->nb[0];                              \
    ଅଚଳ uint32_t nb11 = src1->nb[1];                              \
    ଅଚଳ uint32_t nb12 = src1->nb[2];                              \
    ଅଚଳ uint32_t nb13 = src1->nb[3];                              \
                                                                    \
    ଅଚଳ uint32_t nb0 = dst->nb[0];                                \
    ଅଚଳ uint32_t nb1 = dst->nb[1];                                \
    ଅଚଳ uint32_t nb2 = dst->nb[2];                                \
    ଅଚଳ uint32_t nb3 = dst->nb[3];

#define htp_matmul_preamble                                         \
    ସଂରଚନା htp_mm_context * mmctx  = data;                          \
    ସଂରଚନା htp_ops_context * octx  = mmctx->octx;                   \
    dma_queue * dma_q              = octx->ctx->dma[ith];           \
    uint32_t src0_nrows_per_thread = mmctx->src0_nrows_per_thread;  \
    htp_matmul_tensors_preamble;




// hvx kernels first: the HMX Q6_K dequantizer reuses unpack_q6_k_group from there
#ଅନ୍ତର୍ଭାଵ "hvx-mm-kernels-tiled.h"
#ଅନ୍ତର୍ଭାଵ "hvx-mm-kernels-float.h"
#ଅନ୍ତର୍ଭାଵ "hmx-mm-kernels-tiled.h"

ସ୍ଥିର ଶୂନ୍ୟ hvx_mm_2d_tiled(unsigned ପୂର୍ଣ୍ଣସଂଖ୍ୟା nth, unsigned ପୂର୍ଣ୍ଣସଂଖ୍ୟା ith, ଶୂନ୍ୟ * data) {
    htp_matmul_preamble;

    ଅଚଳ uint32_t src0_nrows = mmctx->src0_row_end - mmctx->src0_row_start;
    ଅଚଳ uint32_t act_nrows = mmctx->cur_m_rows ? mmctx->cur_m_rows : (ne11 * ne12 * ne13);
    ଅଚଳ uint32_t cur_m_start = mmctx->cur_m_start;

    ଅଚଳ uint32_t src0_start_row  = mmctx->src0_row_start + src0_nrows_per_thread * ith;
    ଅଚଳ uint32_t src0_end_row    = MIN(src0_start_row + src0_nrows_per_thread, mmctx->src0_row_end);

    ସଂରଚନା htp_thread_trace * tr = &octx->ctx->trace[ith];

    ଅଚଳ ସଂରଚନା htp_mm_kernel_params * kparams = (ଅଚଳ ସଂରଚନା htp_mm_kernel_params *) octx->kernel_params;
    ଅଚଳ uint32_t n_prefetch = kparams->n_prefetch;
    ପ୍ରତିଜ୍ଞା(n_prefetch >= 2 && n_prefetch <= HTP_MM_MAX_PREFETCH && (n_prefetch & (n_prefetch - 1)) == 0);

    ଅଚଳ size_t dst_row_size  = nb1;
    ଅଚଳ size_t act_row_size = nb11;
    ଅଚଳ size_t act_stride = mmctx->vtcm_act_stride;
    ଅଚଳ size_t src2_stride = src2 ? ((src2->ne[1] == 1) ? 0 : src2->nb[1]) : 0;

    uint8_t * restrict vtcm_dst_ptr  = mmctx->vtcm_dst  + mmctx->vtcm_dst_size_per_thread  * ith;
    uint8_t * restrict vtcm_src0_ptr = mmctx->vtcm_src0 + mmctx->vtcm_src0_size_per_thread * ith;
    uint8_t * restrict act_data = mmctx->vtcm_act;

    ଅଚଳ dma_addr_t src0_row = src0->data;

    ଅଚଳ uint32_t tile_size = htp_tensor_get_tile_size(src0->type);
    ଅଚଳ uint32_t aligned_tile_size = htp_tensor_get_tile_padded_size(src0->type);

    uint32_t n_k_tiles_w = ne00 / htp_tensor_get_tile_k(src0->type);
    uint32_t tile_row_stride = n_k_tiles_w * tile_size;
    uint32_t tile_row_transfer_size_aligned = n_k_tiles_w * aligned_tile_size;

    uint32_t ct_start = src0_start_row / 32;
    uint32_t ct_end   = (src0_end_row + 31) / 32;

    uint32_t ଠେଲ_ct = ct_start;
    ଯଦି (src0_start_row < src0_end_row) {
        ପାଇଁ (uint32_t d = 0; d < n_prefetch && ଠେଲ_ct < ct_end; d++, ଠେଲ_ct++) {
            dma_queue_ଠେଲ(dma_q, dma_make_data(vtcm_src0_ptr + d * tile_row_transfer_size_aligned,
                           src0_row + ଠେଲ_ct * tile_row_stride), aligned_tile_size, tile_size, tile_size, n_k_tiles_w);
        }
    }

    ଯଦି (src0_start_row >= src0_end_row) {
        ଫେରନ୍ତୁ;
    }

    ପାଇଁ (uint32_t ct = ct_start; ct < ct_end; ct++) {
        ଅଚଳ uint8_t * w_tile = (ଶୂନ୍ୟ *) dma_queue_କାଢ଼(dma_q).dst;

        ପୂର୍ଣ୍ଣସଂଖ୍ୟା valid_rows = (ପୂର୍ଣ୍ଣସଂଖ୍ୟା)ne0 - (ପୂର୍ଣ୍ଣସଂଖ୍ୟା)(ct * 32);
        valid_rows = MIN(32, MAX(0, valid_rows));

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, ct);
        uint32_t ir1 = 0;
        ପାଇଁ (; ir1 + 1 < act_nrows; ir1 += 2) {
            ଅଚଳ uint8_t * restrict act_col0 = (ଅଚଳ uint8_t *) (act_data + (ir1+0) * act_stride);
            ଅଚଳ uint8_t * restrict act_col1 = (ଅଚଳ uint8_t *) (act_data + (ir1+1) * act_stride);
            ଦଶମିକ * restrict dst_row0 = (ଦଶମିକ *) (dst->data + ((cur_m_start + ir1+0) * dst_row_size));
            ଦଶମିକ * restrict dst_row1 = (ଦଶମିକ *) (dst->data + ((cur_m_start + ir1+1) * dst_row_size));

            ଦଶମିକ * dst_ptr0 = &dst_row0[ct * 32];
            ଦଶମିକ * dst_ptr1 = &dst_row1[ct * 32];

            ଅଚଳ ଦଶମିକ * src2_ptr0 = ଖାଲି;
            ଅଚଳ ଦଶମିକ * src2_ptr1 = ଖାଲି;
            ଯଦି (src2) {
                ଅଚଳ ଦଶମିକ * restrict src2_row0 = (ଅଚଳ ଦଶମିକ *) ((ଅଚଳ uint8_t *) src2->data + ((cur_m_start + ir1+0) * src2_stride));
                ଅଚଳ ଦଶମିକ * restrict src2_row1 = (ଅଚଳ ଦଶମିକ *) ((ଅଚଳ uint8_t *) src2->data + ((cur_m_start + ir1+1) * src2_stride));
                src2_ptr0 = &src2_row0[ct * 32];
                src2_ptr1 = &src2_row1[ct * 32];
            }
            mmctx->vec_dot_32x2(ne10, dst_ptr0, dst_ptr1, w_tile, act_col0, act_col1, valid_rows, src2_ptr0, src2_ptr1);
        }

        ପାଇଁ (; ir1 < act_nrows; ++ir1) {
            ଅଚଳ uint8_t * restrict act_col = (ଅଚଳ uint8_t *) (act_data + ir1 * act_stride);
            ଦଶମିକ * restrict dst_row          = (ଦଶମିକ *) (dst->data + ((cur_m_start + ir1) * dst_row_size));
            ଦଶମିକ * dst_ptr = &dst_row[ct * 32];

            ଅଚଳ ଦଶମିକ * src2_ptr = ଖାଲି;
            ଯଦି (src2) {
                ଅଚଳ ଦଶମିକ * restrict src2_row = (ଅଚଳ ଦଶମିକ *) ((ଅଚଳ uint8_t *) src2->data + ((cur_m_start + ir1) * src2_stride));
                src2_ptr = &src2_row[ct * 32];
            }
            mmctx->vec_dot_32x1(ne10, dst_ptr, w_tile, act_col, valid_rows, src2_ptr);
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ct);

        ଯଦି (ଠେଲ_ct < ct_end) {
            dma_queue_ଠେଲ(dma_q, dma_make_data(w_tile, src0_row + ଠେଲ_ct * tile_row_stride),
                           aligned_tile_size, tile_size, tile_size, n_k_tiles_w);
            ଠେଲ_ct++;
        }
    }
}

ସ୍ଥିର ଶୂନ୍ୟ hvx_mv_2d_tiled(unsigned ପୂର୍ଣ୍ଣସଂଖ୍ୟା nth, unsigned ପୂର୍ଣ୍ଣସଂଖ୍ୟା ith, ଶୂନ୍ୟ * data) {
    htp_matmul_preamble;

    ଅଚଳ uint32_t src0_nrows = mmctx->src0_row_end - mmctx->src0_row_start;

    ଅଚଳ uint32_t src0_start_row  = mmctx->src0_row_start + src0_nrows_per_thread * ith;
    ଅଚଳ uint32_t src0_end_row    = MIN(src0_start_row + src0_nrows_per_thread, mmctx->src0_row_end);

    ସଂରଚନା htp_thread_trace * tr = &octx->ctx->trace[ith];

    ଅଚଳ ସଂରଚନା htp_mm_kernel_params * kparams = (ଅଚଳ ସଂରଚନା htp_mm_kernel_params *) octx->kernel_params;
    ଅଚଳ uint32_t n_prefetch = kparams->n_prefetch;
    ପ୍ରତିଜ୍ଞା(n_prefetch >= 2 && n_prefetch <= HTP_MM_MAX_PREFETCH && (n_prefetch & (n_prefetch - 1)) == 0);

    ଅଚଳ size_t dst_row_size  = nb1;
    ଅଚଳ size_t act_row_size = nb11;
    ଅଚଳ size_t act_stride = mmctx->vtcm_act_stride;

    uint8_t * vtcm_dst_ptr  = mmctx->vtcm_dst + mmctx->vtcm_dst_size_per_thread * ith;
    uint8_t * vtcm_src0_ptr = mmctx->vtcm_src0 + mmctx->vtcm_src0_size_per_thread * ith;
    uint8_t * act_data = mmctx->vtcm_act;

    ଦଶମିକ * tmp = (ଦଶମିକ *) vtcm_dst_ptr;

    ଅଚଳ dma_addr_t src0_row = src0->data;

    ଅଚଳ uint8_t * restrict act_col = (ଅଚଳ uint8_t *) act_data;
    ଦଶମିକ * restrict dst_col          = (ଦଶମିକ *) dst->data;

    ଅଚଳ uint32_t tile_size = htp_tensor_get_tile_size(src0->type);
    ଅଚଳ uint32_t aligned_tile_size = htp_tensor_get_tile_padded_size(src0->type);

    uint32_t n_k_tiles_w = ne00 / htp_tensor_get_tile_k(src0->type);
    uint32_t tile_row_stride = n_k_tiles_w * tile_size;
    uint32_t tile_row_transfer_size_aligned = n_k_tiles_w * aligned_tile_size;

    uint32_t ct_start = src0_start_row / 32;
    uint32_t ct_end   = (src0_end_row + 31) / 32;

    uint32_t ଠେଲ_ct = ct_start;
    ଯଦି (src0_start_row < src0_end_row) {
        ଯଦି (src2) {
            ଦଶମିକ * vtcm_bias_ptr = (ଦଶମିକ *) mmctx->vtcm_bias + src0_start_row;
            ଅଚଳ dma_addr_t src2_addr = src2->data + src0_start_row * ଆକାର(ଦଶମିକ);
            ପୂର୍ଣ୍ଣସଂଖ୍ୟା slice_size = (ପୂର୍ଣ୍ଣସଂଖ୍ୟା)MIN(src0_end_row, ne0) - (ପୂର୍ଣ୍ଣସଂଖ୍ୟା)src0_start_row;
            ଯଦି (slice_size > 0) {
                dma_queue_ଠେଲ(dma_q, dma_make_data(vtcm_bias_ptr, src2_addr),
                               slice_size * ଆକାର(ଦଶମିକ), slice_size * ଆକାର(ଦଶମିକ), slice_size * ଆକାର(ଦଶମିକ), 1);
                dma_queue_କାଢ଼_nowait(dma_q);
            }
        }
        ପାଇଁ (uint32_t d = 0; d < n_prefetch && ଠେଲ_ct < ct_end; d++, ଠେଲ_ct++) {
            dma_queue_ଠେଲ(dma_q, dma_make_data(vtcm_src0_ptr + d * tile_row_transfer_size_aligned,
                           src0_row + ଠେଲ_ct * tile_row_stride), aligned_tile_size, tile_size, tile_size, n_k_tiles_w);
        }
    }

    ଯଦି (src0_start_row >= src0_end_row) {
        ଫେରନ୍ତୁ;
    }

    ପାଇଁ (uint32_t ct = ct_start; ct < ct_end; ct++) {
        ଅଚଳ uint8_t * w_tile = (ଶୂନ୍ୟ *) dma_queue_କାଢ଼(dma_q).dst;

        ଦଶମିକ * dst_ptr = &tmp[ct * 32 - src0_start_row];
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା valid_rows = (ପୂର୍ଣ୍ଣସଂଖ୍ୟା)ne0 - (ପୂର୍ଣ୍ଣସଂଖ୍ୟା)(ct * 32);
        valid_rows = MIN(32, MAX(0, valid_rows));

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, ct);
        mmctx->vec_dot_32x1(ne10, dst_ptr, w_tile, act_col, valid_rows, ଖାଲି);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ct);

        ଯଦି (ଠେଲ_ct < ct_end) {
            dma_queue_ଠେଲ(dma_q, dma_make_data(w_tile, src0_row + ଠେଲ_ct * tile_row_stride),
                           aligned_tile_size, tile_size, tile_size, n_k_tiles_w);
            ଠେଲ_ct++;
        }
    }

    ପୂର୍ଣ୍ଣସଂଖ୍ୟା copy_cnt = (ପୂର୍ଣ୍ଣସଂଖ୍ୟା)MIN(src0_end_row, ne0) - (ପୂର୍ଣ୍ଣସଂଖ୍ୟା)src0_start_row;
    ଯଦି (copy_cnt > 0) {
        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, ct_end);
        ଯଦି (src2) {
            hvx_add_f32_uuu((uint8_t *) &dst_col[src0_start_row],
                            (ଅଚଳ uint8_t *) tmp,
                            (ଅଚଳ uint8_t *) ((ଅଚଳ ଦଶମିକ *) mmctx->vtcm_bias + src0_start_row),
                            copy_cnt);
        } ଅନ୍ୟଥା {
            hvx_copy_f32_ua((uint8_t *) &dst_col[src0_start_row], (uint8_t *) tmp, copy_cnt);
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ct_end);
    }
}

ସ୍ଥିର ଶୂନ୍ୟ hvx_mm_nx_2d_tiled(unsigned ପୂର୍ଣ୍ଣସଂଖ୍ୟା nth, unsigned ପୂର୍ଣ୍ଣସଂଖ୍ୟା ith, ଶୂନ୍ୟ * data) {
    ସଂରଚନା htp_mm_context * mmctx = data;
    ସଂରଚନା htp_ops_context * octx = mmctx->octx;
    ଅଚଳ ସଂରଚନା htp_mm_kernel_params * kparams = (ଅଚଳ ସଂରଚନା htp_mm_kernel_params *) octx->kernel_params;
    ଅଚଳ uint32_t n_weights = kparams->n_weights;

    ଅଚଳ ସଂରଚନା htp_tensor * restrict act = octx->src[n_weights]; /* x */
    ଅଚଳ uint32_t ne10 = act->ne[0];
    ଅଚଳ uint32_t act_nrows = act->ne[1] * act->ne[2] * act->ne[3];
    ଅଚଳ size_t act_stride = mmctx->vtcm_act_stride;

    uint8_t * restrict vtcm_weight_ptr = mmctx->vtcm_src0 + mmctx->vtcm_src0_size_per_thread * ith;
    uint8_t * restrict act_data        = mmctx->vtcm_act;

    ସଂରଚନା htp_thread_trace * tr = &octx->ctx->trace[ith];
    ଅଚଳ uint32_t n_prefetch = kparams->n_prefetch;
    ପ୍ରତିଜ୍ଞା(n_prefetch >= 2 && n_prefetch <= HTP_MM_MAX_PREFETCH && (n_prefetch & (n_prefetch - 1)) == 0);

    ପାଇଁ (uint32_t widx = 0; widx < n_weights; widx++) {
        ଅଚଳ ସଂରଚନା htp_tensor * restrict src_w = octx->src[widx];
        ଅଚଳ ସଂରଚନା htp_tensor * restrict dst   = octx->dsts[widx];
        ଯଦି (!src_w || !dst) ଜାରି;
        dma_queue * dma_q = octx->ctx->dma[ith];

        ଅଚଳ uint32_t tile_size = htp_tensor_get_tile_size(src_w->type);
        ଅଚଳ uint32_t aligned_tile_size = htp_tensor_get_tile_padded_size(src_w->type);
        ଅଚଳ uint32_t ne00 = src_w->ne[0];
        ଅଚଳ uint32_t ne01 = src_w->ne[1];
        ଅଚଳ size_t dst_row_size = dst->nb[1];
        ଅଚଳ dma_addr_t src_w_row = src_w->data;

        ଅଚଳ uint32_t n_k_tiles_w = ne00 / htp_tensor_get_tile_k(src_w->type);
        ଅଚଳ uint32_t tile_row_stride = n_k_tiles_w * tile_size;
        ଅଚଳ uint32_t tile_row_transfer_size_aligned = n_k_tiles_w * aligned_tile_size;

        uint32_t src0_start_row = 0;
        uint32_t src0_end_row   = ne01;
        ଯଦି (octx->ctx->mdev.ଗଣନା_2 > 1) {
            ଅଚଳ bool can_split = htp_tensor_can_row_partition(dst, ଆକାର(ଦଶମିକ));
            ଅଚଳ ସଂରଚନା htp_tensor_mdev_range range = htp_tensor_mdev_partition(ne01, can_split ? 32 : 0,
                                                         octx->ctx->mdev.idx, octx->ctx->mdev.ଗଣନା_2, &octx->ctx->mdev.ଗଣନା_div);
            src0_start_row = range.start;
            src0_end_row   = range.start + range.ଗଣନା_2;
        }

        ଅଚଳ uint32_t nrows = src0_end_row - src0_start_row;
        uint32_t src0_nrows_per_thread = fastdiv(nrows + nth - 1, &octx->n_threads_div);
        src0_nrows_per_thread = hex_round_up(src0_nrows_per_thread, 32);

        ଅଚଳ uint32_t start_row = src0_start_row + src0_nrows_per_thread * ith;
        ଅଚଳ uint32_t end_row   = MIN(start_row + src0_nrows_per_thread, src0_end_row);
        ଯଦି (start_row >= end_row) ଜାରି;

        uint32_t ct_start = start_row / 32;
        uint32_t ct_end   = (end_row + 31) / 32;

        uint32_t ଠେଲ_ct = ct_start;
        ପାଇଁ (uint32_t d = 0; d < n_prefetch && ଠେଲ_ct < ct_end; d++, ଠେଲ_ct++) {
            dma_queue_ଠେଲ(dma_q, dma_make_data(vtcm_weight_ptr + d * tile_row_transfer_size_aligned,
                            src_w_row + ଠେଲ_ct * tile_row_stride), aligned_tile_size, tile_size, tile_size, n_k_tiles_w);
        }

        ପାଇଁ (uint32_t ct = ct_start; ct < ct_end; ct++) {
            ଅଚଳ uint8_t * w_tile = (ଶୂନ୍ୟ *) dma_queue_କାଢ଼(dma_q).dst;
            ପୂର୍ଣ୍ଣସଂଖ୍ୟା valid_rows = (ପୂର୍ଣ୍ଣସଂଖ୍ୟା)dst->ne[0] - (ପୂର୍ଣ୍ଣସଂଖ୍ୟା)(ct * 32);
            valid_rows = MIN(32, MAX(0, valid_rows));

            htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, ct);
            uint32_t ir1 = 0;
            ପାଇଁ (; ir1 + 1 < act_nrows; ir1 += 2) {
                ଅଚଳ uint8_t * restrict act_col0 = (ଅଚଳ uint8_t *) (act_data + (ir1+0) * act_stride);
                ଅଚଳ uint8_t * restrict act_col1 = (ଅଚଳ uint8_t *) (act_data + (ir1+1) * act_stride);

                ଦଶମିକ * restrict dst_row0 = (ଦଶମିକ *) (dst->data + ((ir1+0) * dst_row_size));
                ଦଶମିକ * restrict dst_row1 = (ଦଶମିକ *) (dst->data + ((ir1+1) * dst_row_size));
                ଦଶମିକ * dst_ptr0 = &dst_row0[ct * 32];
                ଦଶମିକ * dst_ptr1 = &dst_row1[ct * 32];

                mmctx->vec_dot_32x2(ne10, dst_ptr0, dst_ptr1, w_tile, act_col0, act_col1, valid_rows, ଖାଲି, ଖାଲି);
            }

            ପାଇଁ (; ir1 < act_nrows; ++ir1) {
                ଅଚଳ uint8_t * restrict act_col = (ଅଚଳ uint8_t *) (act_data + ir1 * act_stride);
                ଦଶମିକ * restrict dst_row = (ଦଶମିକ *) (dst->data + (ir1 * dst_row_size));
                ଦଶମିକ * dst_ptr = &dst_row[ct * 32];
                mmctx->vec_dot_32x1(ne10, dst_ptr, w_tile, act_col, valid_rows, ଖାଲି);
            }
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ct);

            ଯଦି (ଠେଲ_ct < ct_end) {
                dma_queue_ଠେଲ(dma_q, dma_make_data(w_tile, src_w_row + ଠେଲ_ct * tile_row_stride),
                                aligned_tile_size, tile_size, tile_size, n_k_tiles_w);
                ଠେଲ_ct++;
            }
        }
    }
}

ସ୍ଥିର ଶୂନ୍ୟ hvx_mm_transfer_act_dma(
    ସଂରଚନା htp_ops_context * octx,
    ଅଚଳ ସଂରଚନା htp_mm_kernel_params * kparams,
    ଅଚଳ ସଂରଚନା htp_tensor * act,
    uint8_t * dst_base,
    size_t dst_row_size,
    uint32_t m_start,
    uint32_t m_rows
) {
    ଯଦି (m_rows == 0) {
        ଫେରନ୍ତୁ;
    }

    dma_queue * dma_q = octx->ctx->dma[0];
    ଅଚଳ uint32_t ne0 = act->ne[0];
    ଅଚଳ size_t elem_size = (act->type == HTP_TYPE_F16) ? ଆକାର(__fp16) : ଆକାର(ଦଶମିକ);
    ଅଚଳ size_t row_bytes = ne0 * elem_size;
    ଅଚଳ size_t act_nb1 = act->nb[1];
    ଅଚଳ dma_addr_t act_base = act->data;

    ଅଚଳ bool is_contiguous = (act->nb[2] == act->ne[1] * act_nb1) &&
                               (act->nb[3] == act->ne[2] * act->nb[2]);

    ଯଦି (is_contiguous) {
        ଅଚଳ dma_addr_t src_addr = act_base + m_start * act_nb1;
        dma_queue_ଠେଲ(dma_q, dma_make_data(dst_base, src_addr),
                       dst_row_size, act_nb1, row_bytes, m_rows);
        dma_queue_କାଢ଼(dma_q);
    } ଅନ୍ୟଥା {
        ଅଚଳ uint32_t ne12_ne1 = act->ne[2] * act->ne[1];
        ଅଚଳ bool use_fastdiv = kparams->div_ne12_ne1.mp != 0;
        ପାଇଁ (uint32_t ir = 0; ir < m_rows; ++ir) {
            ଅଚଳ uint32_t ir1 = m_start + ir;
            uint32_t i11, i12, i13;
            ଯଦି (use_fastdiv) {
                i13 = fastdiv(ir1, &kparams->div_ne12_ne1);
                ଅଚଳ uint32_t rem = ir1 - i13 * ne12_ne1;
                i12 = fastdiv(rem, &kparams->div_ne1);
                i11 = rem - i12 * act->ne[1];
            } ଅନ୍ୟଥା {
                i13 = ne12_ne1 ? ir1 / ne12_ne1 : 0;
                ଅଚଳ uint32_t rem = ir1 - i13 * ne12_ne1;
                i12 = act->ne[1] ? rem / act->ne[1] : 0;
                i11 = rem - i12 * act->ne[1];
            }
            ଅଚଳ dma_addr_t row_src = act_base + (i11 * act->nb[1] +
                                                   i12 * act->nb[2] +
                                                   i13 * act->nb[3]);
            uint8_t * row_dst = dst_base + ir * dst_row_size;
            dma_queue_ଠେଲ(dma_q, dma_make_data(row_dst, row_src),
                           dst_row_size, act_nb1, row_bytes, 1);
            dma_queue_କାଢ଼(dma_q);
        }
    }

    ଯଦି (dst_row_size > row_bytes) {
        ସଂରଚନା htp_thread_trace * tr = &octx->ctx->trace[0];
        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_A_PREP, (uint16_t) m_start);
        ଅଚଳ uint32_t pad_elems = (dst_row_size - row_bytes) / elem_size;
        ଯଦି (elem_size == ଆକାର(ଦଶମିକ)) {
            ପାଇଁ (uint32_t ir = 0; ir < m_rows; ++ir) {
                hvx_splat_f32_u(dst_base + ir * dst_row_size + row_bytes, 0.0f, pad_elems);
            }
        } ଅନ୍ୟଥା {
            ପାଇଁ (uint32_t ir = 0; ir < m_rows; ++ir) {
                hvx_splat_f16_u(dst_base + ir * dst_row_size + row_bytes, (_Float16) 0.0f, pad_elems);
            }
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_A_PREP, (uint16_t) m_start);
    }
}

#define QUANTIZE_IMPL(name, log_name, kernel_fn, dst_row_size_ପଦସମଷ୍ଟି)                                        \
ସ୍ଥିର ଶୂନ୍ୟ name(unsigned ପୂର୍ଣ୍ଣସଂଖ୍ୟା nth, unsigned ପୂର୍ଣ୍ଣସଂଖ୍ୟା ith, ଶୂନ୍ୟ * data) {                                        \
    (ଶୂନ୍ୟ) nth;                                                                                            \
    ସଂରଚନା htp_mm_context * mmctx = data;                                                                  \
    ସଂରଚନା htp_ops_context * octx = mmctx->octx;                                                           \
    ଅଚଳ ସଂରଚନା htp_tensor * src = mmctx->act;                                                            \
    ଅଚଳ uint32_t ne0 = src->ne[0];                                                                       \
    ଅଚଳ uint32_t nrows = mmctx->cur_m_rows ? mmctx->cur_m_rows : mmctx->act_nrows;                       \
    ଅଚଳ uint32_t nrows_per_thread = mmctx->n_quant_rows_per_thread;                                      \
                                                                                                           \
    ଅଚଳ uint32_t ir_first = nrows_per_thread * ith;                                                      \
    ଯଦି (ir_first >= nrows) {                                                                               \
        ଫେରନ୍ତୁ;                                                                                            \
    }                                                                                                      \
                                                                                                           \
    ସଂରଚନା htp_thread_trace * tr = &octx->ctx->trace[ith];                                                 \
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_A_QUANT, ir_first);                                        \
                                                                                                           \
    uint8_t * restrict dst = mmctx->vtcm_act;                                                              \
    ଅଚଳ uint32_t ir_last = MIN(ir_first + nrows_per_thread, nrows);                                      \
    ଅଚଳ size_t raw_row_size = mmctx->vtcm_act_raw_stride;                                                \
    ଅଚଳ size_t dst_row_size = (dst_row_size_ପଦସମଷ୍ଟି);                                                       \
                                                                                                           \
    ଅଚଳ uint8_t * restrict src_data = (ଅଚଳ uint8_t *) mmctx->vtcm_act_raw + (raw_row_size * ir_first); \
    uint8_t * restrict dst_data = (uint8_t *) dst + (dst_row_size * ir_first);                             \
    kernel_fn(src_data, dst_data, ଖାଲି, ne0, ir_last - ir_first, raw_row_size, dst_row_size);              \
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

ସ୍ଥିର ଶୂନ୍ୟ quantize_f32_q8_0_tiled_block(unsigned ପୂର୍ଣ୍ଣସଂଖ୍ୟା nth, unsigned ପୂର୍ଣ୍ଣସଂଖ୍ୟା ith, ଶୂନ୍ୟ * data) {
    (ଶୂନ୍ୟ) nth;
    ସଂରଚନା htp_mm_context * mmctx = data;
    ଯଦି (mmctx->quant_ib_first[ith] >= mmctx->quant_ib_last[ith]) {
        ଫେରନ୍ତୁ;
    }
    ସଂରଚନା htp_ops_context * octx = mmctx->octx;
    ସଂରଚନା htp_thread_trace * tr = &octx->ctx->trace[ith];
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_A_QUANT, mmctx->quant_ib_first[ith]);

    ଅଚଳ ସଂରଚନା htp_tensor * src = mmctx->act;

    quantize_f32_q8_0_tiled_block_kernel(
        (ଅଚଳ ଦଶମିକ *) mmctx->vtcm_act_raw,
        mmctx->vtcm_act,
        ଖାଲି,
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

ସ୍ଥିର ଶୂନ୍ୟ quantize_f32_q8_1_tiled_block(unsigned ପୂର୍ଣ୍ଣସଂଖ୍ୟା nth, unsigned ପୂର୍ଣ୍ଣସଂଖ୍ୟା ith, ଶୂନ୍ୟ * data) {
    (ଶୂନ୍ୟ) nth;
    ସଂରଚନା htp_mm_context * mmctx = data;
    ଯଦି (mmctx->quant_ib_first[ith] >= mmctx->quant_ib_last[ith]) {
        ଫେରନ୍ତୁ;
    }
    ସଂରଚନା htp_ops_context * octx = mmctx->octx;
    ସଂରଚନା htp_thread_trace * tr = &octx->ctx->trace[ith];
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_A_QUANT, mmctx->quant_ib_first[ith]);

    ଅଚଳ ସଂରଚନା htp_tensor * src = mmctx->act;

    quantize_f32_q8_1_tiled_block_kernel(
        (ଅଚଳ ଦଶମିକ *) mmctx->vtcm_act_raw,
        mmctx->vtcm_act,
        ଖାଲି,
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

ସ୍ଥିର ଶୂନ୍ୟ quantize_f32_q8_1_s16_tiled_block(unsigned ପୂର୍ଣ୍ଣସଂଖ୍ୟା nth, unsigned ପୂର୍ଣ୍ଣସଂଖ୍ୟା ith, ଶୂନ୍ୟ * data) {
    (ଶୂନ୍ୟ) nth;
    ସଂରଚନା htp_mm_context * mmctx = data;
    ଯଦି (mmctx->quant_ib_first[ith] >= mmctx->quant_ib_last[ith]) {
        ଫେରନ୍ତୁ;
    }
    ସଂରଚନା htp_ops_context * octx = mmctx->octx;
    ସଂରଚନା htp_thread_trace * tr = &octx->ctx->trace[ith];
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_A_QUANT, mmctx->quant_ib_first[ith]);

    ଅଚଳ ସଂରଚନା htp_tensor * src = mmctx->act;

    quantize_f32_q8_1_s16_tiled_block_kernel(
        (ଅଚଳ ଦଶମିକ *) mmctx->vtcm_act_raw,
        mmctx->vtcm_act,
        ଖାଲି,
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

ସ୍ଥିର ଶୂନ୍ୟ quantize_f32_f16_tiled_block(unsigned ପୂର୍ଣ୍ଣସଂଖ୍ୟା nth, unsigned ପୂର୍ଣ୍ଣସଂଖ୍ୟା ith, ଶୂନ୍ୟ * data) {
    (ଶୂନ୍ୟ) nth;
    ସଂରଚନା htp_mm_context * mmctx = data;
    ଯଦି (mmctx->quant_ib_first[ith] >= mmctx->quant_ib_last[ith]) {
        ଫେରନ୍ତୁ;
    }
    ସଂରଚନା htp_ops_context * octx = mmctx->octx;
    ସଂରଚନା htp_thread_trace * tr = &octx->ctx->trace[ith];
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_A_QUANT, mmctx->quant_ib_first[ith]);

    ଅଚଳ ସଂରଚନା htp_tensor * src = mmctx->act;

    quantize_f32_f16_tiled_block_kernel(
        (ଅଚଳ ଦଶମିକ *) mmctx->vtcm_act_raw,
        mmctx->vtcm_act,
        ଖାଲି,
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

ସ୍ଥିର ଶୂନ୍ୟ quantize_f16_f16_tiled_block(unsigned ପୂର୍ଣ୍ଣସଂଖ୍ୟା nth, unsigned ପୂର୍ଣ୍ଣସଂଖ୍ୟା ith, ଶୂନ୍ୟ * data) {
    (ଶୂନ୍ୟ) nth;
    ସଂରଚନା htp_mm_context * mmctx = data;
    ଯଦି (mmctx->quant_ib_first[ith] >= mmctx->quant_ib_last[ith]) {
        ଫେରନ୍ତୁ;
    }
    ସଂରଚନା htp_ops_context * octx = mmctx->octx;
    ସଂରଚନା htp_thread_trace * tr = &octx->ctx->trace[ith];
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_A_QUANT, mmctx->quant_ib_first[ith]);

    ଅଚଳ ସଂରଚନା htp_tensor * src = mmctx->act;

    quantize_f16_f16_tiled_block_kernel(
        (ଅଚଳ __fp16 *) mmctx->vtcm_act_raw,
        mmctx->vtcm_act,
        ଖାଲି,
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
ସ୍ଥିର ଧାଡ଼ିଭିତରେ worker_callback_t htp_mm_act_quant_row_func(ପୂର୍ଣ୍ଣସଂଖ୍ୟା weight_type, ପୂର୍ଣ୍ଣସଂଖ୍ୟା act_type) {
    ଯଦି (weight_type == HTP_TYPE_F16) {
        ଫେରନ୍ତୁ (act_type == HTP_TYPE_F16) ? quantize_f16_f16_tiled : quantize_f32_f16_tiled;
    }
    ଯଦି (weight_type == HTP_TYPE_Q2_K) {
        ଫେରନ୍ତୁ quantize_f32_q8_1_s16_tiled;
    }
    ଫେରନ୍ତୁ htp_mm_weight_has_offset(weight_type) ? quantize_f32_q8_1_tiled : quantize_f32_q8_0_tiled;
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ worker_callback_t htp_mm_act_quant_block_func(ପୂର୍ଣ୍ଣସଂଖ୍ୟା weight_type, ପୂର୍ଣ୍ଣସଂଖ୍ୟା act_type) {
    ଯଦି (weight_type == HTP_TYPE_F16) {
        ଫେରନ୍ତୁ (act_type == HTP_TYPE_F16) ? quantize_f16_f16_tiled_block : quantize_f32_f16_tiled_block;
    }
    ଯଦି (weight_type == HTP_TYPE_Q2_K) {
        ଫେରନ୍ତୁ quantize_f32_q8_1_s16_tiled_block;
    }
    ଫେରନ୍ତୁ htp_mm_weight_has_offset(weight_type) ? quantize_f32_q8_1_tiled_block : quantize_f32_q8_0_tiled_block;
}

ସ୍ଥିର ଶୂନ୍ୟ hvx_mm_4d_tiled(unsigned ପୂର୍ଣ୍ଣସଂଖ୍ୟା nth, unsigned ପୂର୍ଣ୍ଣସଂଖ୍ୟା ith, ଶୂନ୍ୟ * data) {
    htp_matmul_preamble;

    ଅଚଳ uint32_t src0_nrows = mmctx->src0_row_end - mmctx->src0_row_start;
    ଅଚଳ uint32_t cur_m_rows = mmctx->cur_m_rows ? mmctx->cur_m_rows : (ne11 * ne12 * ne13);
    ଅଚଳ uint32_t cur_m_start = mmctx->cur_m_start;

    ଅଚଳ uint32_t src0_start_row  = mmctx->src0_row_start + src0_nrows_per_thread * ith;
    ଅଚଳ uint32_t src0_end_row    = MIN(src0_start_row + src0_nrows_per_thread, mmctx->src0_row_end);

    ସଂରଚନା htp_thread_trace * tr = &octx->ctx->trace[ith];

    ଅଚଳ ସଂରଚନା htp_mm_kernel_params * kparams = (ଅଚଳ ସଂରଚନା htp_mm_kernel_params *) octx->kernel_params;
    ଅଚଳ uint32_t n_prefetch = kparams->n_prefetch;
    ପ୍ରତିଜ୍ଞା(n_prefetch >= 2 && n_prefetch <= HTP_MM_MAX_PREFETCH && (n_prefetch & (n_prefetch - 1)) == 0);

    ଅଚଳ size_t dst_row_size = nb1;
    ଅଚଳ size_t act_stride = mmctx->vtcm_act_stride;

    uint8_t * restrict vtcm_src0_ptr = mmctx->vtcm_src0 + mmctx->vtcm_src0_size_per_thread * ith;
    uint8_t * restrict act_data = mmctx->vtcm_act;

    ଅଚଳ uint32_t tile_size = htp_tensor_get_tile_size(src0->type);
    ଅଚଳ uint32_t aligned_tile_size = htp_tensor_get_tile_padded_size(src0->type);

    ଅଚଳ uint32_t n_k_tiles_w = ne00 / htp_tensor_get_tile_k(src0->type);
    ଅଚଳ uint32_t tile_row_stride = n_k_tiles_w * tile_size;
    ଅଚଳ uint32_t tile_row_transfer_size_aligned = n_k_tiles_w * aligned_tile_size;
    ଅଚଳ uint32_t src0_slice_stride = ((ne01 + 31) / 32) * tile_row_stride;

    ଅଚଳ uint32_t ct_start = src0_start_row / 32;
    ଅଚଳ uint32_t ct_end   = (src0_end_row + 31) / 32;

    ଯଦି (src0_start_row >= src0_end_row || cur_m_rows == 0) {
        ଫେରନ୍ତୁ;
    }

    ଅଚଳ uint32_t total_batches = ne12 * ne13;
    ଅଚଳ uint32_t b_start = fastdiv(cur_m_start, &kparams->div_ne1);
    uint32_t b_end = fastdiv(cur_m_start + cur_m_rows + ne11 - 1, &kparams->div_ne1);
    b_end = MIN(b_end, total_batches);

    uint32_t b_grp_start = b_start;
    ଯେତେବେଳେ (b_grp_start < b_end) {
        ଅଚଳ uint32_t b3 = fastdiv(b_grp_start, &kparams->div_ne12);
        ଅଚଳ uint32_t b2 = b_grp_start - b3 * ne12;
        ଅଚଳ uint32_t i02 = fastdiv(b2, &kparams->div_r2);
        ଅଚଳ uint32_t i03 = fastdiv(b3, &kparams->div_r3);

        uint32_t b_grp_end = b_grp_start + 1;
        ଯେତେବେଳେ (b_grp_end < b_end) {
            ଅଚଳ uint32_t cur_b3 = fastdiv(b_grp_end, &kparams->div_ne12);
            ଅଚଳ uint32_t cur_b2 = b_grp_end - cur_b3 * ne12;
            ଅଚଳ uint32_t cur_i02 = fastdiv(cur_b2, &kparams->div_r2);
            ଅଚଳ uint32_t cur_i03 = fastdiv(cur_b3, &kparams->div_r3);
            ଯଦି (cur_i02 != i02 || cur_i03 != i03) {
                ଭାଙ୍ଗ;
            }
            b_grp_end++;
        }

        ଅଚଳ uint32_t slice_idx = i03 * ne02 + i02;
        ଅଚଳ dma_addr_t src0_slice = src0->data + (size_t) slice_idx * src0_slice_stride;

        uint32_t ଠେଲ_ct = ct_start;
        ପାଇଁ (uint32_t d = 0; d < n_prefetch && ଠେଲ_ct < ct_end; d++, ଠେଲ_ct++) {
            dma_queue_ଠେଲ(dma_q, dma_make_data(vtcm_src0_ptr + d * tile_row_transfer_size_aligned,
                           src0_slice + (size_t) ଠେଲ_ct * tile_row_stride),
                           aligned_tile_size, tile_size, tile_size, n_k_tiles_w);
        }

        ପାଇଁ (uint32_t ct = ct_start; ct < ct_end; ct++) {
            ଅଚଳ uint8_t * w_tile = (ଶୂନ୍ୟ *) dma_queue_କାଢ଼(dma_q).dst;

            ପୂର୍ଣ୍ଣସଂଖ୍ୟା valid_rows = (ପୂର୍ଣ୍ଣସଂଖ୍ୟା)ne0 - (ପୂର୍ଣ୍ଣସଂଖ୍ୟା)(ct * 32);
            valid_rows = MIN(32, MAX(0, valid_rows));

            htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, ct);
            ପାଇଁ (uint32_t b = b_grp_start; b < b_grp_end; b++) {
                ଅଚଳ uint32_t b_m_start = b * ne11;
                ଅଚଳ uint32_t m_first = MAX(cur_m_start, b_m_start);
                ଅଚଳ uint32_t m_last  = MIN(cur_m_start + cur_m_rows, b_m_start + ne11);
                ଯଦି (m_first >= m_last) ଜାରି;

                ଅଚଳ uint32_t cur_b3 = fastdiv(b, &kparams->div_ne12);
                ଅଚଳ uint32_t cur_b2 = b - cur_b3 * ne12;
                uint8_t * dst_batch_base = (uint8_t *) dst->data + (size_t) cur_b2 * nb2 + (size_t) cur_b3 * nb3;

                ଅଚଳ uint32_t chunk_m_offset = m_first - cur_m_start;
                ଅଚଳ uint32_t dst_m_offset   = m_first - b_m_start;
                ଅଚଳ uint32_t batch_nrows    = m_last - m_first;

                uint32_t ir1 = 0;
                ପାଇଁ (; ir1 + 1 < batch_nrows; ir1 += 2) {
                    ଅଚଳ uint8_t * restrict act_col0 = (ଅଚଳ uint8_t *) (act_data + (chunk_m_offset + ir1 + 0) * act_stride);
                    ଅଚଳ uint8_t * restrict act_col1 = (ଅଚଳ uint8_t *) (act_data + (chunk_m_offset + ir1 + 1) * act_stride);
                    ଦଶମିକ * restrict dst_row0 = (ଦଶମିକ *) (dst_batch_base + (dst_m_offset + ir1 + 0) * dst_row_size);
                    ଦଶମିକ * restrict dst_row1 = (ଦଶମିକ *) (dst_batch_base + (dst_m_offset + ir1 + 1) * dst_row_size);
                    ଦଶମିକ * dst_ptr0 = &dst_row0[ct * 32];
                    ଦଶମିକ * dst_ptr1 = &dst_row1[ct * 32];
                    mmctx->vec_dot_32x2(ne10, dst_ptr0, dst_ptr1, w_tile, act_col0, act_col1, valid_rows, ଖାଲି, ଖାଲି);
                }
                ପାଇଁ (; ir1 < batch_nrows; ++ir1) {
                    ଅଚଳ uint8_t * restrict act_col = (ଅଚଳ uint8_t *) (act_data + (chunk_m_offset + ir1) * act_stride);
                    ଦଶମିକ * restrict dst_row = (ଦଶମିକ *) (dst_batch_base + (dst_m_offset + ir1) * dst_row_size);
                    ଦଶମିକ * dst_ptr = &dst_row[ct * 32];
                    mmctx->vec_dot_32x1(ne10, dst_ptr, w_tile, act_col, valid_rows, ଖାଲି);
                }
            }
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ct);

            ଯଦି (ଠେଲ_ct < ct_end) {
                dma_queue_ଠେଲ(dma_q, dma_make_data(w_tile, src0_slice + (size_t) ଠେଲ_ct * tile_row_stride),
                               aligned_tile_size, tile_size, tile_size, n_k_tiles_w);
                ଠେଲ_ct++;
            }
        }
        b_grp_start = b_grp_end;
    }
    ଯଦି (src2) {
        hvx_tensor_add_f32_grid(dst, src2, cur_m_start, cur_m_start + cur_m_rows, src0_start_row, src0_end_row, &kparams->div_ne12_ne1, &kparams->div_ne1);
    }
}

ସ୍ଥିର ଶୂନ୍ୟ hvx_mm_2d_flat(unsigned ପୂର୍ଣ୍ଣସଂଖ୍ୟା nth, unsigned ପୂର୍ଣ୍ଣସଂଖ୍ୟା ith, ଶୂନ୍ୟ * data) {
    htp_matmul_preamble;

    ଅଚଳ ସଂରଚନା htp_mm_kernel_params * kparams = (ଅଚଳ ସଂରଚନା htp_mm_kernel_params *) octx->kernel_params;
    ଅଚଳ uint32_t n_prefetch = kparams->n_prefetch;
    ପ୍ରତିଜ୍ଞା(n_prefetch >= 2 && n_prefetch <= HTP_MM_MAX_PREFETCH && (n_prefetch & (n_prefetch - 1)) == 0);
    ଅଚଳ uint32_t prefetch_mask = n_prefetch - 1;

    ଅଚଳ uint32_t src0_nrows = mmctx->src0_row_end - mmctx->src0_row_start;  // src0 rows
    ଅଚଳ uint32_t act_nrows = mmctx->cur_m_rows ? mmctx->cur_m_rows : mmctx->act_nrows;
    ଅଚଳ uint32_t cur_m_start = mmctx->cur_m_start;

    ଅଚଳ uint32_t src0_start_row  = mmctx->src0_row_start + src0_nrows_per_thread * ith;
    ଅଚଳ uint32_t src0_end_row    = MIN(src0_start_row + src0_nrows_per_thread, mmctx->src0_row_end);
    ଅଚଳ uint32_t src0_end_row_x2 = src0_start_row + ((src0_end_row - src0_start_row) & ~1U);

    ସଂରଚନା htp_thread_trace * tr = &octx->ctx->trace[ith];

    ଅଚଳ size_t dst_row_size  = nb1;
    ଅଚଳ size_t src0_row_size = nb01;
    ଅଚଳ size_t act_row_size = nb11;

    ଅଚଳ size_t src0_stride = mmctx->vtcm_src0_stride;
    ଅଚଳ size_t act_stride = mmctx->vtcm_act_stride;

    // Per-thread VTCMs for all tensors
    uint8_t * restrict vtcm_dst_ptr  = mmctx->vtcm_dst  + mmctx->vtcm_dst_size_per_thread  * ith;
    uint8_t * restrict vtcm_src0_ptr = mmctx->vtcm_src0 + mmctx->vtcm_src0_size_per_thread * ith;
    uint8_t * restrict act_data      = mmctx->vtcm_act;

    ଅଚଳ dma_addr_t src0_row = src0->data;


    // Prefill vtcm with src0 rows
    ଯଦି (src0_start_row < src0_end_row) {
        ପାଇଁ (uint32_t ir0 = src0_start_row; ir0 < src0_end_row_x2; ir0 += 2) {
            ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା is0 = (ir0 - src0_start_row);
            ଯଦି (is0 >= (ପୂର୍ଣ୍ଣସଂଖ୍ୟା)n_prefetch) {
                ଭାଙ୍ଗ;
            }
            dma_queue_ଠେଲ(dma_q, dma_make_data(vtcm_src0_ptr + is0 * src0_stride, src0_row + ir0 * src0_row_size),
                           src0_stride, src0_row_size, src0_row_size, 2);
        }
    }

    ଯଦି (src0_start_row >= src0_end_row) {
        ଫେରନ୍ତୁ;
    }

    // Process src0 rows
    ପାଇଁ (uint32_t ir0 = src0_start_row; ir0 < src0_end_row_x2; ir0 += 2) {
        ଅଚଳ uint8_t * ss0 = (ଶୂନ୍ୟ *) dma_queue_କାଢ଼(dma_q).dst;

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, ir0);
        // Process act columns in pairs (2x2 tiling)
        uint32_t ir1 = 0;
        ପାଇଁ (; ir1 + 1 < act_nrows; ir1 += 2) {
            ଅଚଳ uint8_t * restrict act_col0 = (ଅଚଳ uint8_t *) (act_data + (ir1+0) * act_stride);
            ଅଚଳ uint8_t * restrict act_col1 = (ଅଚଳ uint8_t *) (act_data + (ir1+1) * act_stride);
            ଦଶମିକ * restrict dst_row0 = (ଦଶମିକ *) (dst->data + ((cur_m_start + ir1+0) * dst_row_size));
            ଦଶମିକ * restrict dst_row1 = (ଦଶମିକ *) (dst->data + ((cur_m_start + ir1+1) * dst_row_size));
            mmctx->vec_dot_2x2(ne00, &dst_row0[ir0], &dst_row1[ir0], ss0, ss0 + src0_stride, act_col0, act_col1);
        }

        // Handle remaining act rows (fallback to 2x1)
        ପାଇଁ (; ir1 < act_nrows; ++ir1) {
            ଅଚଳ uint8_t * restrict act_col = (ଅଚଳ uint8_t *) (act_data + ir1 * act_stride);
            ଦଶମିକ * restrict dst_row          = (ଦଶମିକ *) (dst->data + ((cur_m_start + ir1) * dst_row_size));
            mmctx->vec_dot_2x1(ne00, &dst_row[ir0], ss0, ss0 + src0_stride, act_col);
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ir0);

        // Prefetch next (n + vtcm_nrows) row
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା pr0 = (ir0 + n_prefetch);
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା is0 = (pr0 - src0_start_row) & prefetch_mask;
        ଯଦି (pr0 < src0_end_row_x2) {
            dma_queue_ଠେଲ(dma_q, dma_make_data(vtcm_src0_ptr + is0 * src0_stride, src0_row + pr0 * src0_row_size),
                           src0_stride, src0_row_size, src0_row_size, 2);
        }
    }

    // Process the last row (if any)
    ଯଦି (src0_end_row != src0_end_row_x2) {
        uint32_t  ir0 = src0_end_row_x2;
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା is0 = (ir0 - src0_start_row) & prefetch_mask;
        dma_queue_ଠେଲ(dma_q, dma_make_data(vtcm_src0_ptr + is0 * src0_stride, src0_row + ir0 * src0_row_size),
                       src0_stride, src0_row_size, src0_row_size, 1);
        ଅଚଳ uint8_t * ss0 = (ଶୂନ୍ୟ *) dma_queue_କାଢ଼(dma_q).dst;

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, ir0);
        #pragma unroll(2)
        ପାଇଁ (uint32_t ir1 = 0; ir1 < act_nrows; ++ir1) {
            ଅଚଳ uint8_t * restrict act_col = (ଅଚଳ uint8_t *) (act_data + ir1 * act_stride);
            ଦଶମିକ * restrict dst_row          = (ଦଶମିକ *) (dst->data + ((cur_m_start + ir1) * dst_row_size));
            mmctx->vec_dot_1x1(ne00, &dst_row[ir0], ss0, act_col);
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ir0);
    }
    ଯଦି (src2) {
        hvx_tensor_add_f32_grid(dst, src2, cur_m_start, cur_m_start + act_nrows, src0_start_row, src0_end_row, &kparams->div_ne12_ne1, &kparams->div_ne1);
    }
}

ସ୍ଥିର ଶୂନ୍ୟ hvx_mv_2d_flat(unsigned ପୂର୍ଣ୍ଣସଂଖ୍ୟା nth, unsigned ପୂର୍ଣ୍ଣସଂଖ୍ୟା ith, ଶୂନ୍ୟ * data) {
    htp_matmul_preamble;

    ଅଚଳ uint32_t src0_nrows = mmctx->src0_row_end - mmctx->src0_row_start;

    ଅଚଳ uint32_t src0_start_row  = mmctx->src0_row_start + src0_nrows_per_thread * ith;
    ଅଚଳ uint32_t src0_end_row    = MIN(src0_start_row + src0_nrows_per_thread, mmctx->src0_row_end);

    ସଂରଚନା htp_thread_trace * tr = &octx->ctx->trace[ith];

    ଅଚଳ size_t dst_row_size  = nb1;
    ଅଚଳ size_t src0_row_size = nb01;
    ଅଚଳ size_t act_row_size  = nb11;

    ଅଚଳ size_t src0_stride = mmctx->vtcm_src0_stride;
    ଅଚଳ size_t act_stride  = mmctx->vtcm_act_stride;

    // Per-thread VTCMs for all tensors
    uint8_t * vtcm_dst_ptr  = mmctx->vtcm_dst  + mmctx->vtcm_dst_size_per_thread  * ith;
    uint8_t * vtcm_src0_ptr = mmctx->vtcm_src0 + mmctx->vtcm_src0_size_per_thread * ith;
    uint8_t * act_data      = mmctx->vtcm_act;

    ଦଶମିକ * tmp = (ଦଶମିକ *) vtcm_dst_ptr;

    ଅଚଳ dma_addr_t src0_row = src0->data;
    ଅଚଳ uint8_t * restrict act_col = (ଅଚଳ uint8_t *) act_data;
    ଦଶମିକ * restrict dst_col         = (ଦଶମିକ *) dst->data;

    ଅଚଳ uint32_t src0_end_row_x2 = src0_start_row + ((src0_end_row - src0_start_row) & ~1U);

    ଅଚଳ ସଂରଚନା htp_mm_kernel_params * kparams = (ଅଚଳ ସଂରଚନା htp_mm_kernel_params *) octx->kernel_params;
    ଅଚଳ uint32_t n_prefetch = kparams->n_prefetch;
    ପ୍ରତିଜ୍ଞା(n_prefetch >= 2 && n_prefetch <= HTP_MM_MAX_PREFETCH && (n_prefetch & (n_prefetch - 1)) == 0);
    ଅଚଳ uint32_t prefetch_mask = n_prefetch - 1;

    // Prefill vtcm with 2x src0 rows
    ଯଦି (src0_start_row < src0_end_row) {
        ଯଦି (src2) {
            ଦଶମିକ * vtcm_bias_ptr = (ଦଶମିକ *) mmctx->vtcm_bias + src0_start_row;
            ଅଚଳ dma_addr_t src2_addr = src2->data + src0_start_row * ଆକାର(ଦଶମିକ);
            ପୂର୍ଣ୍ଣସଂଖ୍ୟା slice_size = (ପୂର୍ଣ୍ଣସଂଖ୍ୟା)src0_end_row - (ପୂର୍ଣ୍ଣସଂଖ୍ୟା)src0_start_row;
            ଯଦି (slice_size > 0) {
                dma_queue_ଠେଲ(dma_q, dma_make_data(vtcm_bias_ptr, src2_addr),
                               slice_size * ଆକାର(ଦଶମିକ), slice_size * ଆକାର(ଦଶମିକ), slice_size * ଆକାର(ଦଶମିକ), 1);
                dma_queue_କାଢ଼_nowait(dma_q);
            }
        }
        ପାଇଁ (uint32_t ir0 = src0_start_row; ir0 < src0_end_row_x2; ir0 += 2) {
            ଅଚଳ uint32_t is0 = (ir0 - src0_start_row);
            ଯଦି (is0 >= n_prefetch) {
                ଭାଙ୍ଗ;
            }
            dma_queue_ଠେଲ(dma_q, dma_make_data(vtcm_src0_ptr + is0 * src0_stride, src0_row + ir0 * src0_row_size),
                           src0_stride, src0_row_size, src0_row_size, 2);
        }
    }


    ଯଦି (src0_start_row >= src0_end_row) {
        ଫେରନ୍ତୁ;
    }

    // Process src0 rows
    ପାଇଁ (uint32_t ir0 = src0_start_row; ir0 < src0_end_row_x2; ir0 += 2) {
        ଅଚଳ uint8_t * ss0 = (ଶୂନ୍ୟ *) dma_queue_କାଢ଼(dma_q).dst;
        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, ir0);
        mmctx->vec_dot_2x1(ne00, &tmp[ir0 - src0_start_row], ss0, ss0 + src0_stride, act_col);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ir0);

        // Prefetch next (n + vtcm_nrows) row
        ଅଚଳ uint32_t pr0 = (ir0 + n_prefetch);
        ଅଚଳ uint32_t is0 = (pr0 - src0_start_row) & prefetch_mask;
        ଯଦି (pr0 < src0_end_row_x2) {
            dma_queue_ଠେଲ(dma_q, dma_make_data(vtcm_src0_ptr + is0 * src0_stride, src0_row + pr0 * src0_row_size),
                           src0_stride, src0_row_size, src0_row_size, 2);
        }
    }

    // Process the last row (if any)
    ଯଦି (src0_end_row != src0_end_row_x2) {
        ଅଚଳ uint32_t ir0 = src0_end_row_x2;
        ଅଚଳ uint32_t is0 = (ir0 - src0_start_row) & prefetch_mask;
        dma_queue_ଠେଲ(dma_q, dma_make_data(vtcm_src0_ptr + is0 * src0_stride, src0_row + ir0 * src0_row_size),
                       src0_stride, src0_row_size, src0_row_size, 1);
        ଅଚଳ uint8_t * ss0 = (ଶୂନ୍ୟ *) dma_queue_କାଢ଼(dma_q).dst;
        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, ir0);
        mmctx->vec_dot_1x1(ne00, &tmp[ir0 - src0_start_row], ss0, act_col);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ir0);
    }

    ପୂର୍ଣ୍ଣସଂଖ୍ୟା copy_cnt = src0_end_row - src0_start_row;
    ଯଦି (copy_cnt > 0) {
        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, src0_end_row);
        ଯଦି (src2) {
            hvx_add_f32_uuu((uint8_t *) &dst_col[src0_start_row],
                            (ଅଚଳ uint8_t *) tmp,
                            (ଅଚଳ uint8_t *) ((ଅଚଳ ଦଶମିକ *) mmctx->vtcm_bias + src0_start_row),
                            copy_cnt);
        } ଅନ୍ୟଥା {
            hvx_copy_f32_ua((uint8_t *) &dst_col[src0_start_row], (uint8_t *) tmp, copy_cnt);
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, src0_end_row);
    }
}

ସ୍ଥିର ଶୂନ୍ୟ hvx_mm_4d_flat(unsigned ପୂର୍ଣ୍ଣସଂଖ୍ୟା nth, unsigned ପୂର୍ଣ୍ଣସଂଖ୍ୟା ith, ଶୂନ୍ୟ * data) {
    htp_matmul_preamble;

    ଅଚଳ ସଂରଚନା htp_mm_kernel_params * kparams = (ଅଚଳ ସଂରଚନା htp_mm_kernel_params *) octx->kernel_params;
    ଅଚଳ uint32_t n_prefetch = kparams->n_prefetch;
    ପ୍ରତିଜ୍ଞା(n_prefetch >= 2 && n_prefetch <= HTP_MM_MAX_PREFETCH && (n_prefetch & (n_prefetch - 1)) == 0);
    ଅଚଳ uint32_t prefetch_mask = n_prefetch - 1;

    ଅଚଳ uint32_t src0_nrows = mmctx->src0_row_end - mmctx->src0_row_start;
    ଅଚଳ uint32_t cur_m_rows = mmctx->cur_m_rows ? mmctx->cur_m_rows : (ne11 * ne12 * ne13);
    ଅଚଳ uint32_t cur_m_start = mmctx->cur_m_start;

    ଅଚଳ uint32_t src0_start_row  = mmctx->src0_row_start + src0_nrows_per_thread * ith;
    ଅଚଳ uint32_t src0_end_row    = MIN(src0_start_row + src0_nrows_per_thread, mmctx->src0_row_end);
    ଅଚଳ uint32_t src0_end_row_x2 = src0_start_row + ((src0_end_row - src0_start_row) & ~1U);

    ସଂରଚନା htp_thread_trace * tr = &octx->ctx->trace[ith];

    ଅଚଳ size_t dst_row_size  = nb1;
    ଅଚଳ size_t src0_row_size = nb01;
    ଅଚଳ size_t src0_stride = mmctx->vtcm_src0_stride;
    ଅଚଳ size_t act_stride  = mmctx->vtcm_act_stride;

    uint8_t * restrict vtcm_src0_ptr = mmctx->vtcm_src0 + mmctx->vtcm_src0_size_per_thread * ith;
    uint8_t * restrict act_data      = mmctx->vtcm_act;


    ଯଦି (src0_start_row >= src0_end_row || cur_m_rows == 0) {
        ଫେରନ୍ତୁ;
    }

    ଅଚଳ uint32_t total_batches = ne12 * ne13;
    ଅଚଳ uint32_t b_start = fastdiv(cur_m_start, &kparams->div_ne1);
    uint32_t b_end = fastdiv(cur_m_start + cur_m_rows + ne11 - 1, &kparams->div_ne1);
    b_end = MIN(b_end, total_batches);

    uint32_t b_grp_start = b_start;
    ଯେତେବେଳେ (b_grp_start < b_end) {
        ଅଚଳ uint32_t b3 = fastdiv(b_grp_start, &kparams->div_ne12);
        ଅଚଳ uint32_t b2 = b_grp_start - b3 * ne12;
        ଅଚଳ uint32_t i02 = fastdiv(b2, &kparams->div_r2);
        ଅଚଳ uint32_t i03 = fastdiv(b3, &kparams->div_r3);

        uint32_t b_grp_end = b_grp_start + 1;
        ଯେତେବେଳେ (b_grp_end < b_end) {
            ଅଚଳ uint32_t cur_b3 = fastdiv(b_grp_end, &kparams->div_ne12);
            ଅଚଳ uint32_t cur_b2 = b_grp_end - cur_b3 * ne12;
            ଅଚଳ uint32_t cur_i02 = fastdiv(cur_b2, &kparams->div_r2);
            ଅଚଳ uint32_t cur_i03 = fastdiv(cur_b3, &kparams->div_r3);
            ଯଦି (cur_i02 != i02 || cur_i03 != i03) {
                ଭାଙ୍ଗ;
            }
            b_grp_end++;
        }

        ଅଚଳ dma_addr_t src0_row = src0->data + ((size_t) i02 * nb02 + (size_t) i03 * nb03);

        ପାଇଁ (uint32_t ir0 = src0_start_row; ir0 < src0_end_row_x2; ir0 += 2) {
            ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା is0 = (ir0 - src0_start_row);
            ଯଦି (is0 >= (ପୂର୍ଣ୍ଣସଂଖ୍ୟା)n_prefetch) {
                ଭାଙ୍ଗ;
            }
            dma_queue_ଠେଲ(dma_q, dma_make_data(vtcm_src0_ptr + is0 * src0_stride, src0_row + (size_t) ir0 * src0_row_size),
                           src0_stride, src0_row_size, src0_row_size, 2);
        }

        ପାଇଁ (uint32_t ir0 = src0_start_row; ir0 < src0_end_row_x2; ir0 += 2) {
            ଅଚଳ uint8_t * ss0 = (ଶୂନ୍ୟ *) dma_queue_କାଢ଼(dma_q).dst;

            htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, ir0);
            ପାଇଁ (uint32_t b = b_grp_start; b < b_grp_end; b++) {
                ଅଚଳ uint32_t b_m_start = b * ne11;
                ଅଚଳ uint32_t m_first = MAX(cur_m_start, b_m_start);
                ଅଚଳ uint32_t m_last  = MIN(cur_m_start + cur_m_rows, b_m_start + ne11);
                ଯଦି (m_first >= m_last) ଜାରି;

                ଅଚଳ uint32_t cur_b3 = fastdiv(b, &kparams->div_ne12);
                ଅଚଳ uint32_t cur_b2 = b - cur_b3 * ne12;
                uint8_t * dst_batch_base = (uint8_t *) dst->data + (size_t) cur_b2 * nb2 + (size_t) cur_b3 * nb3;

                ଅଚଳ uint32_t chunk_m_offset = m_first - cur_m_start;
                ଅଚଳ uint32_t dst_m_offset   = m_first - b_m_start;
                ଅଚଳ uint32_t batch_nrows    = m_last - m_first;

                uint32_t ir1 = 0;
                ପାଇଁ (; ir1 + 1 < batch_nrows; ir1 += 2) {
                    ଅଚଳ uint8_t * restrict act_col0 = (ଅଚଳ uint8_t *) (act_data + (chunk_m_offset + ir1 + 0) * act_stride);
                    ଅଚଳ uint8_t * restrict act_col1 = (ଅଚଳ uint8_t *) (act_data + (chunk_m_offset + ir1 + 1) * act_stride);
                    ଦଶମିକ * restrict dst_row0 = (ଦଶମିକ *) (dst_batch_base + (dst_m_offset + ir1 + 0) * dst_row_size);
                    ଦଶମିକ * restrict dst_row1 = (ଦଶମିକ *) (dst_batch_base + (dst_m_offset + ir1 + 1) * dst_row_size);
                    mmctx->vec_dot_2x2(ne00, &dst_row0[ir0], &dst_row1[ir0], ss0, ss0 + src0_stride, act_col0, act_col1);
                }
                ପାଇଁ (; ir1 < batch_nrows; ++ir1) {
                    ଅଚଳ uint8_t * restrict act_col = (ଅଚଳ uint8_t *) (act_data + (chunk_m_offset + ir1) * act_stride);
                    ଦଶମିକ * restrict dst_row          = (ଦଶମିକ *) (dst_batch_base + (dst_m_offset + ir1) * dst_row_size);
                    mmctx->vec_dot_2x1(ne00, &dst_row[ir0], ss0, ss0 + src0_stride, act_col);
                }
            }
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ir0);

            ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା pr0 = (ir0 + n_prefetch);
            ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା is0 = (pr0 - src0_start_row) & prefetch_mask;
            ଯଦି (pr0 < src0_end_row_x2) {
                dma_queue_ଠେଲ(dma_q, dma_make_data(vtcm_src0_ptr + is0 * src0_stride, src0_row + (size_t) pr0 * src0_row_size),
                               src0_stride, src0_row_size, src0_row_size, 2);
            }
        }

        ଯଦି (src0_end_row != src0_end_row_x2) {
            uint32_t ir0 = src0_end_row_x2;
            ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା is0 = (ir0 - src0_start_row) & prefetch_mask;
            dma_queue_ଠେଲ(dma_q, dma_make_data(vtcm_src0_ptr + is0 * src0_stride, src0_row + (size_t) ir0 * src0_row_size),
                           src0_stride, src0_row_size, src0_row_size, 1);
            ଅଚଳ uint8_t * ss0 = (ଶୂନ୍ୟ *) dma_queue_କାଢ଼(dma_q).dst;

            htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, ir0);
            ପାଇଁ (uint32_t b = b_grp_start; b < b_grp_end; b++) {
                ଅଚଳ uint32_t b_m_start = b * ne11;
                ଅଚଳ uint32_t m_first = MAX(cur_m_start, b_m_start);
                ଅଚଳ uint32_t m_last  = MIN(cur_m_start + cur_m_rows, b_m_start + ne11);
                ଯଦି (m_first >= m_last) ଜାରି;

                ଅଚଳ uint32_t cur_b3 = fastdiv(b, &kparams->div_ne12);
                ଅଚଳ uint32_t cur_b2 = b - cur_b3 * ne12;
                uint8_t * dst_batch_base = (uint8_t *) dst->data + (size_t) cur_b2 * nb2 + (size_t) cur_b3 * nb3;

                ଅଚଳ uint32_t chunk_m_offset = m_first - cur_m_start;
                ଅଚଳ uint32_t dst_m_offset   = m_first - b_m_start;
                ଅଚଳ uint32_t batch_nrows    = m_last - m_first;

                ପାଇଁ (uint32_t ir1 = 0; ir1 < batch_nrows; ++ir1) {
                    ଅଚଳ uint8_t * restrict act_col = (ଅଚଳ uint8_t *) (act_data + (chunk_m_offset + ir1) * act_stride);
                    ଦଶମିକ * restrict dst_row          = (ଦଶମିକ *) (dst_batch_base + (dst_m_offset + ir1) * dst_row_size);
                    mmctx->vec_dot_1x1(ne00, &dst_row[ir0], ss0, act_col);
                }
            }
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ir0);
        }

        b_grp_start = b_grp_end;
    }

    ଯଦି (src2) {
        hvx_tensor_add_f32_grid(dst, src2, cur_m_start, cur_m_start + cur_m_rows, src0_start_row, src0_end_row, &kparams->div_ne12_ne1, &kparams->div_ne1);
    }
}

#define MMID_MATRIX_ROW(row_id, i1) matrix_rows[(row_id) * mmctx->mapping_stride + (i1)]

ସ୍ଥିର ଶୂନ୍ୟ hvx_mm_id_tiled(unsigned ପୂର୍ଣ୍ଣସଂଖ୍ୟା nth, unsigned ପୂର୍ଣ୍ଣସଂଖ୍ୟା ith, ଶୂନ୍ୟ * data) {
    htp_matmul_preamble;

    ଅଚଳ ସଂରଚନା htp_tensor * restrict ids = octx->src[2];

    ଅଚଳ uint32_t src0_nrows      = mmctx->src0_row_end - mmctx->src0_row_start;  // src0 rows per expert
    ଅଚଳ uint32_t act_nrows       = ne11;
    ଅଚଳ uint32_t src0_start_row  = mmctx->src0_row_start + src0_nrows_per_thread * ith;
    ଅଚଳ uint32_t src0_end_row    = MIN(src0_start_row + src0_nrows_per_thread, mmctx->src0_row_end);


    ଯଦି (src0_start_row >= src0_end_row) {
        ଫେରନ୍ତୁ;
    }

    ସଂରଚନା htp_thread_trace * tr = &octx->ctx->trace[ith];

    ଅଚଳ ସଂରଚନା htp_mm_kernel_params * kparams = (ଅଚଳ ସଂରଚନା htp_mm_kernel_params *) octx->kernel_params;
    ଅଚଳ uint32_t n_prefetch = kparams->n_prefetch;
    ପ୍ରତିଜ୍ଞା(n_prefetch >= 2 && n_prefetch <= HTP_MM_MAX_PREFETCH && (n_prefetch & (n_prefetch - 1)) == 0);

    ଅଚଳ uint32_t n_ids = ids->ne[0];  // n_expert_used
    ଅଚଳ uint32_t n_as  = ne02;        // n_expert

    ଅଚଳ uint32_t *                matrix_row_counts = mmctx->matrix_row_counts;
    ଅଚଳ ସଂରଚନା mmid_row_mapping * matrix_rows       = mmctx->matrix_rows;

    ଅଚଳ size_t dst_row_size  = nb1;
    ଅଚଳ size_t act_row_size  = htp_mm_q8_0_tiled_row_size(ne10);

    ଅଚଳ size_t act_stride = mmctx->vtcm_act_stride;

    // Per-thread VTCMs for all tensors
    uint8_t * restrict vtcm_src0_ptr = mmctx->vtcm_src0 + mmctx->vtcm_src0_size_per_thread * ith;
    uint8_t * restrict act_data = mmctx->vtcm_act;

    ପାଇଁ (uint32_t cur_a = 0; cur_a < n_as; ++cur_a) {
        ଅଚଳ int32_t cne1 = matrix_row_counts[cur_a];
        ଯଦି (cne1 == 0) {
            ଜାରି;
        }

        ଅଚଳ dma_addr_t src0_row = src0->data + cur_a * nb02;

        ଅଚଳ uint32_t tile_size = htp_tensor_get_tile_size(src0->type);
        ଅଚଳ uint32_t aligned_tile_size = htp_tensor_get_tile_padded_size(src0->type);
        ଅଚଳ uint32_t n_k_tiles_w = ne00 / htp_tensor_get_tile_k(src0->type);
        ଅଚଳ uint32_t tile_row_stride = n_k_tiles_w * tile_size;
        ଅଚଳ uint32_t tile_row_transfer_size_aligned = n_k_tiles_w * aligned_tile_size;

        ଅଚଳ uint32_t ct_start = src0_start_row / 32;
        ଅଚଳ uint32_t ct_end   = (src0_end_row + 31) / 32;

        uint32_t ଠେଲ_ct = ct_start;
        ପାଇଁ (uint32_t d = 0; d < n_prefetch && ଠେଲ_ct < ct_end; d++, ଠେଲ_ct++) {
            dma_queue_ଠେଲ(dma_q, dma_make_data(vtcm_src0_ptr + d * tile_row_transfer_size_aligned, src0_row + ଠେଲ_ct * tile_row_stride),
                           aligned_tile_size, tile_size, tile_size, n_k_tiles_w);
        }

        ପାଇଁ (uint32_t ct = ct_start; ct < ct_end; ct++) {
            ଅଚଳ uint8_t * w_tile = (ଶୂନ୍ୟ *) dma_queue_କାଢ଼(dma_q).dst;

            ପୂର୍ଣ୍ଣସଂଖ୍ୟା valid_rows = (ପୂର୍ଣ୍ଣସଂଖ୍ୟା)ne0 - (ପୂର୍ଣ୍ଣସଂଖ୍ୟା)(ct * 32);
            valid_rows = MIN(32, MAX(0, valid_rows));

            htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, ct);
            ପାଇଁ (uint32_t cid = 0; cid < cne1; ++cid) {
                ସଂରଚନା mmid_row_mapping row_mapping = MMID_MATRIX_ROW(cur_a, cid);
                ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା               rm1         = row_mapping.i1;  // expert idx
                ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା               rm2         = row_mapping.i2;  // token idx

                ଅଚଳ uint32_t ir1 = fastmodulo(rm1, ne11, &mmctx->mm_div_ne11);        // act row idx
                ଅଚଳ uint8_t * restrict act_col = (ଅଚଳ uint8_t *) (act_data + (ir1 + rm2 * ne11 + 0) * act_stride);
                ଦଶମିକ * restrict dst_row = (ଦଶମିକ *) (dst->data + (rm1 * nb1 + rm2 * nb2 + 0));

                mmctx->vec_dot_32x1(ne10, &dst_row[ct * 32], w_tile, act_col, valid_rows, ଖାଲି);
            }
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ct);

            ଯଦି (ଠେଲ_ct < ct_end) {
                dma_queue_ଠେଲ(dma_q, dma_make_data(w_tile, src0_row + ଠେଲ_ct * tile_row_stride),
                               aligned_tile_size, tile_size, tile_size, n_k_tiles_w);
                ଠେଲ_ct++;
            }
        }
    }
}

ସ୍ଥିର ଶୂନ୍ୟ hvx_mv_id_tiled(unsigned ପୂର୍ଣ୍ଣସଂଖ୍ୟା nth, unsigned ପୂର୍ଣ୍ଣସଂଖ୍ୟା ith, ଶୂନ୍ୟ * data) {
    htp_matmul_preamble;

    ଅଚଳ ସଂରଚନା htp_tensor * restrict ids = octx->src[2];

    ଅଚଳ uint32_t src0_nrows      = mmctx->src0_row_end - mmctx->src0_row_start;  // src0 rows per expert
    ଅଚଳ uint32_t src0_start_row  = mmctx->src0_row_start + src0_nrows_per_thread * ith;
    ଅଚଳ uint32_t src0_end_row    = MIN(src0_start_row + src0_nrows_per_thread, mmctx->src0_row_end);


    ଯଦି (src0_start_row >= src0_end_row) {
        ଫେରନ୍ତୁ;
    }

    ସଂରଚନା htp_thread_trace * tr = &octx->ctx->trace[ith];

    ଅଚଳ ସଂରଚନା htp_mm_kernel_params * kparams = (ଅଚଳ ସଂରଚନା htp_mm_kernel_params *) octx->kernel_params;
    ଅଚଳ uint32_t n_prefetch = kparams->n_prefetch;
    ପ୍ରତିଜ୍ଞା(n_prefetch >= 2 && n_prefetch <= HTP_MM_MAX_PREFETCH && (n_prefetch & (n_prefetch - 1)) == 0);

    ପ୍ରତିଜ୍ଞା(ne13 % ne03 == 0);

    ଅଚଳ size_t dst_row_size  = nb1;
    ଅଚଳ size_t act_row_size = htp_mm_q8_0_tiled_row_size(ne10);

    ଅଚଳ uint32_t n_aids = src2->ne[0];  // num activated experts
    ଅଚଳ uint32_t n_ids  = ne02;         // num experts

    // Per-thread VTCMs for all tensors
    uint8_t * restrict vtcm_src0_ptr = mmctx->vtcm_src0 + mmctx->vtcm_src0_size_per_thread * ith;
    uint8_t * restrict act_data = mmctx->vtcm_act;

    ପାଇଁ (uint32_t ie1 = 0; ie1 < n_aids; ++ie1) {  // for each expert
        ଅଚଳ int32_t eid = *(ଅଚଳ int32_t *) ((ଅଚଳ uint8_t *) src2->data + ie1 * src2->nb[0]);
        ଯଦି (eid < 0) {
            ଜାରି;
        }
        ପ୍ରତିଜ୍ଞା(eid < (int32_t) n_ids);

        ଅଚଳ dma_addr_t src0_row = src0->data + eid * nb02;
        ଅଚଳ uint8_t * restrict act_col = (ଅଚଳ uint8_t *) act_data;
        ଦଶମିକ * restrict dst_row          = (ଦଶମିକ *) (dst->data + ie1 * nb1);

        ଅଚଳ uint32_t tile_size = htp_tensor_get_tile_size(src0->type);
        ଅଚଳ uint32_t aligned_tile_size = htp_tensor_get_tile_padded_size(src0->type);
        ଅଚଳ uint32_t n_k_tiles_w = ne00 / htp_tensor_get_tile_k(src0->type);
        ଅଚଳ uint32_t tile_row_stride = n_k_tiles_w * tile_size;
        ଅଚଳ uint32_t tile_row_transfer_size_aligned = n_k_tiles_w * aligned_tile_size;

        ଅଚଳ uint32_t ct_start = src0_start_row / 32;
        ଅଚଳ uint32_t ct_end   = (src0_end_row + 31) / 32;

        uint32_t ଠେଲ_ct = ct_start;
        ପାଇଁ (uint32_t d = 0; d < n_prefetch && ଠେଲ_ct < ct_end; d++, ଠେଲ_ct++) {
            dma_queue_ଠେଲ(dma_q, dma_make_data(vtcm_src0_ptr + d * tile_row_transfer_size_aligned, src0_row + ଠେଲ_ct * tile_row_stride),
                           aligned_tile_size, tile_size, tile_size, n_k_tiles_w);
        }

        ପାଇଁ (uint32_t ct = ct_start; ct < ct_end; ct++) {
            ଅଚଳ uint8_t * w_tile = (ଶୂନ୍ୟ *) dma_queue_କାଢ଼(dma_q).dst;

            ପୂର୍ଣ୍ଣସଂଖ୍ୟା valid_rows = (ପୂର୍ଣ୍ଣସଂଖ୍ୟା)ne0 - (ପୂର୍ଣ୍ଣସଂଖ୍ୟା)(ct * 32);
            valid_rows = MIN(32, MAX(0, valid_rows));

            htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, ct);
            mmctx->vec_dot_32x1(ne10, &dst_row[ct * 32], w_tile, act_col, valid_rows, ଖାଲି);
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ct);

            ଯଦି (ଠେଲ_ct < ct_end) {
                dma_queue_ଠେଲ(dma_q, dma_make_data(w_tile, src0_row + ଠେଲ_ct * tile_row_stride),
                               aligned_tile_size, tile_size, tile_size, n_k_tiles_w);
                ଠେଲ_ct++;
            }
        }
    }
}

ସ୍ଥିର ଶୂନ୍ୟ hvx_mv_id_nx_tiled(unsigned ପୂର୍ଣ୍ଣସଂଖ୍ୟା nth, unsigned ପୂର୍ଣ୍ଣସଂଖ୍ୟା ith, ଶୂନ୍ୟ * data) {
    ସଂରଚନା htp_mm_context * mmctx = (ସଂରଚନା htp_mm_context *) data;
    ସଂରଚନା htp_ops_context * octx = mmctx->octx;
    ଅଚଳ ସଂରଚନା htp_mm_kernel_params * kparams = (ଅଚଳ ସଂରଚନା htp_mm_kernel_params *) octx->kernel_params;
    ଅଚଳ uint32_t n_weights      = kparams->n_weights;
    ଅଚଳ ସଂରଚନା htp_tensor * restrict src0 = octx->src[0];
    ଅଚଳ ସଂରଚନା htp_tensor * restrict act  = octx->src[n_weights];
    ଅଚଳ ସଂରଚନା htp_tensor * restrict ids  = octx->src[n_weights + 1];


    ସଂରଚନା htp_thread_trace * tr = &octx->ctx->trace[ith];

    ଅଚଳ uint32_t n_prefetch = kparams->n_prefetch;
    ପ୍ରତିଜ୍ଞା(n_prefetch >= 2 && n_prefetch <= HTP_MM_MAX_PREFETCH && (n_prefetch & (n_prefetch - 1)) == 0);

    ଅଚଳ uint32_t n_aids = ids->ne[0];
    ଅଚଳ uint32_t n_ids  = src0->ne[2];

    uint8_t * restrict vtcm_src0_ptr = mmctx->vtcm_src0 + mmctx->vtcm_src0_size_per_thread * ith;
    uint8_t * restrict act_data = mmctx->vtcm_act;

    ପାଇଁ (uint32_t ie1 = 0; ie1 < n_aids; ++ie1) {
        ଅଚଳ int32_t eid = *(ଅଚଳ int32_t *) ((ଅଚଳ uint8_t *) ids->data + ie1 * ids->nb[0]);
        ଯଦି (eid < 0) ଜାରି;
        ପ୍ରତିଜ୍ଞା(eid < (int32_t) n_ids);

        ପାଇଁ (uint32_t p = 0; p < n_weights; ++p) {
            ଅଚଳ ସଂରଚନା htp_tensor * restrict src_w = octx->src[p];
            ଅଚଳ ସଂରଚନା htp_tensor * restrict dst   = octx->dsts[p];
            ଯଦି (!src_w || !dst) ଜାରି;
            dma_queue * dma_q = octx->ctx->dma[ith];

            ଅଚଳ uint32_t ne01 = src_w->ne[1];
            uint32_t start_row = 0;
            uint32_t end_row   = ne01;
            ଯଦି (octx->ctx->mdev.ଗଣନା_2 > 1) {
                ଅଚଳ bool can_split = htp_tensor_can_row_partition(dst, ଆକାର(ଦଶମିକ));
                ଅଚଳ ସଂରଚନା htp_tensor_mdev_range range = htp_tensor_mdev_partition(ne01, can_split ? 32 : 0, octx->ctx->mdev.idx, octx->ctx->mdev.ଗଣନା_2, &octx->ctx->mdev.ଗଣନା_div);
                start_row = range.start;
                end_row   = range.start + range.ଗଣନା_2;
            }

            ଅଚଳ uint32_t nrows = end_row - start_row;
            uint32_t src0_nrows_per_thread = fastdiv(nrows + nth - 1, &octx->n_threads_div);
            src0_nrows_per_thread = hex_round_up(src0_nrows_per_thread, 32);

            ଅଚଳ uint32_t src0_start_row = start_row + src0_nrows_per_thread * ith;
            ଅଚଳ uint32_t src0_end_row   = MIN(src0_start_row + src0_nrows_per_thread, end_row);
            ଯଦି (src0_start_row >= src0_end_row) ଜାରି;

            ଅଚଳ dma_addr_t src0_row = src_w->data + eid * src_w->nb[2];
            ଅଚଳ uint8_t * restrict act_col = (ଅଚଳ uint8_t *) act_data;
            ଦଶମିକ * restrict dst_row = (ଦଶମିକ *) (dst->data + ie1 * dst->nb[1]);

            ଅଚଳ uint32_t tile_size = htp_tensor_get_tile_size(src_w->type);
            ଅଚଳ uint32_t aligned_tile_size = htp_tensor_get_tile_padded_size(src_w->type);
            ଅଚଳ uint32_t n_k_tiles_w = src_w->ne[0] / htp_tensor_get_tile_k(src_w->type);
            ଅଚଳ uint32_t tile_row_stride = n_k_tiles_w * tile_size;
            ଅଚଳ uint32_t tile_row_transfer_size_aligned = n_k_tiles_w * aligned_tile_size;

            ଅଚଳ uint32_t ct_start = src0_start_row / 32;
            ଅଚଳ uint32_t ct_end   = (src0_end_row + 31) / 32;

            uint32_t ଠେଲ_ct = ct_start;
            ପାଇଁ (uint32_t d = 0; d < n_prefetch && ଠେଲ_ct < ct_end; d++, ଠେଲ_ct++) {
                dma_queue_ଠେଲ(dma_q, dma_make_data(vtcm_src0_ptr + d * tile_row_transfer_size_aligned, src0_row + ଠେଲ_ct * tile_row_stride),
                               aligned_tile_size, tile_size, tile_size, n_k_tiles_w);
            }

            ପାଇଁ (uint32_t ct = ct_start; ct < ct_end; ct++) {
                ଅଚଳ uint8_t * w_tile = (ଶୂନ୍ୟ *) dma_queue_କାଢ଼(dma_q).dst;

                ପୂର୍ଣ୍ଣସଂଖ୍ୟା valid_rows = (ପୂର୍ଣ୍ଣସଂଖ୍ୟା)dst->ne[0] - (ପୂର୍ଣ୍ଣସଂଖ୍ୟା)(ct * 32);
                valid_rows = MIN(32, MAX(0, valid_rows));

                htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, ct);
                mmctx->vec_dot_32x1(act->ne[0], &dst_row[ct * 32], w_tile, act_col, valid_rows, ଖାଲି);
                htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ct);

                ଯଦି (ଠେଲ_ct < ct_end) {
                    dma_queue_ଠେଲ(dma_q, dma_make_data(w_tile, src0_row + ଠେଲ_ct * tile_row_stride),
                                    aligned_tile_size, tile_size, tile_size, n_k_tiles_w);
                    ଠେଲ_ct++;
                }
            }
        }
    }
}

ସ୍ଥିର ଶୂନ୍ୟ hvx_mm_id_nx_tiled(unsigned ପୂର୍ଣ୍ଣସଂଖ୍ୟା nth, unsigned ପୂର୍ଣ୍ଣସଂଖ୍ୟା ith, ଶୂନ୍ୟ * data) {
    ସଂରଚନା htp_mm_context * mmctx = (ସଂରଚନା htp_mm_context *) data;
    ସଂରଚନା htp_ops_context * octx = mmctx->octx;
    ଅଚଳ ସଂରଚନା htp_mm_kernel_params * kparams = (ଅଚଳ ସଂରଚନା htp_mm_kernel_params *) octx->kernel_params;
    ଅଚଳ uint32_t n_weights      = kparams->n_weights;
    ଅଚଳ ସଂରଚନା htp_tensor * restrict src0 = octx->src[0];
    ଅଚଳ ସଂରଚନା htp_tensor * restrict act  = octx->src[n_weights];
    ଅଚଳ ସଂରଚନା htp_tensor * restrict ids  = octx->src[n_weights + 1];


    ସଂରଚନା htp_thread_trace * tr = &octx->ctx->trace[ith];

    ଅଚଳ uint32_t n_prefetch = kparams->n_prefetch;
    ପ୍ରତିଜ୍ଞା(n_prefetch >= 2 && n_prefetch <= HTP_MM_MAX_PREFETCH && (n_prefetch & (n_prefetch - 1)) == 0);

    ଅଚଳ uint32_t n_as = src0->ne[2];

    ଅଚଳ uint32_t * matrix_row_counts = mmctx->matrix_row_counts;
    ଅଚଳ ସଂରଚନା mmid_row_mapping * matrix_rows = mmctx->matrix_rows;

    ଅଚଳ size_t act_stride = mmctx->vtcm_act_stride;

    uint8_t * restrict vtcm_src0_ptr = mmctx->vtcm_src0 + mmctx->vtcm_src0_size_per_thread * ith;
    uint8_t * restrict act_data = mmctx->vtcm_act;

    ପାଇଁ (uint32_t cur_a = 0; cur_a < n_as; ++cur_a) {
        ଅଚଳ int32_t cne1 = matrix_row_counts[cur_a];
        ଯଦି (cne1 == 0) ଜାରି;

        ପାଇଁ (uint32_t p = 0; p < n_weights; ++p) {
            ଅଚଳ ସଂରଚନା htp_tensor * restrict src_w = octx->src[p];
            ଅଚଳ ସଂରଚନା htp_tensor * restrict dst   = octx->dsts[p];
            ଯଦି (!src_w || !dst) ଜାରି;
            dma_queue * dma_q = octx->ctx->dma[ith];

            ଅଚଳ uint32_t ne01 = src_w->ne[1];
            uint32_t start_row = 0;
            uint32_t end_row   = ne01;
            ଯଦି (octx->ctx->mdev.ଗଣନା_2 > 1) {
                ଅଚଳ bool can_split = htp_tensor_can_row_partition(dst, ଆକାର(ଦଶମିକ));
                ଅଚଳ ସଂରଚନା htp_tensor_mdev_range range = htp_tensor_mdev_partition(ne01, can_split ? 32 : 0, octx->ctx->mdev.idx, octx->ctx->mdev.ଗଣନା_2, &octx->ctx->mdev.ଗଣନା_div);
                start_row = range.start;
                end_row   = range.start + range.ଗଣନା_2;
            }

            ଅଚଳ uint32_t nrows = end_row - start_row;
            uint32_t src0_nrows_per_thread = fastdiv(nrows + nth - 1, &octx->n_threads_div);
            src0_nrows_per_thread = hex_round_up(src0_nrows_per_thread, 32);

            ଅଚଳ uint32_t src0_start_row = start_row + src0_nrows_per_thread * ith;
            ଅଚଳ uint32_t src0_end_row   = MIN(src0_start_row + src0_nrows_per_thread, end_row);
            ଯଦି (src0_start_row >= src0_end_row) ଜାରି;

            ଅଚଳ dma_addr_t src0_row = src_w->data + cur_a * src_w->nb[2];

            ଅଚଳ uint32_t tile_size = htp_tensor_get_tile_size(src_w->type);
            ଅଚଳ uint32_t aligned_tile_size = htp_tensor_get_tile_padded_size(src_w->type);
            ଅଚଳ uint32_t n_k_tiles_w = src_w->ne[0] / htp_tensor_get_tile_k(src_w->type);
            ଅଚଳ uint32_t tile_row_stride = n_k_tiles_w * tile_size;
            ଅଚଳ uint32_t tile_row_transfer_size_aligned = n_k_tiles_w * aligned_tile_size;

            ଅଚଳ uint32_t ct_start = src0_start_row / 32;
            ଅଚଳ uint32_t ct_end   = (src0_end_row + 31) / 32;

            uint32_t ଠେଲ_ct = ct_start;
            ପାଇଁ (uint32_t d = 0; d < n_prefetch && ଠେଲ_ct < ct_end; d++, ଠେଲ_ct++) {
                dma_queue_ଠେଲ(dma_q, dma_make_data(vtcm_src0_ptr + d * tile_row_transfer_size_aligned, src0_row + ଠେଲ_ct * tile_row_stride),
                               aligned_tile_size, tile_size, tile_size, n_k_tiles_w);
            }

            ପାଇଁ (uint32_t ct = ct_start; ct < ct_end; ct++) {
                ଅଚଳ uint8_t * w_tile = (ଶୂନ୍ୟ *) dma_queue_କାଢ଼(dma_q).dst;

                ପୂର୍ଣ୍ଣସଂଖ୍ୟା valid_rows = (ପୂର୍ଣ୍ଣସଂଖ୍ୟା)dst->ne[0] - (ପୂର୍ଣ୍ଣସଂଖ୍ୟା)(ct * 32);
                valid_rows = MIN(32, MAX(0, valid_rows));

                htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, ct);
                ପାଇଁ (uint32_t cid = 0; cid < (uint32_t) cne1; ++cid) {
                    ସଂରଚନା mmid_row_mapping row_mapping = MMID_MATRIX_ROW(cur_a, cid);
                    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା rm1 = row_mapping.i1;
                    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା rm2 = row_mapping.i2;

                    ଅଚଳ uint32_t ir1 = fastmodulo(rm1, act->ne[1], &mmctx->mm_div_ne11);
                    ଅଚଳ uint8_t * restrict act_col = (ଅଚଳ uint8_t *) (act_data + (ir1 + rm2 * act->ne[1]) * act_stride);
                    ଦଶମିକ * restrict dst_row = (ଦଶମିକ *) (dst->data + (rm1 * dst->nb[1] + rm2 * dst->nb[2]));

                    mmctx->vec_dot_32x1(act->ne[0], &dst_row[ct * 32], w_tile, act_col, valid_rows, ଖାଲି);
                }
                htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ct);

                ଯଦି (ଠେଲ_ct < ct_end) {
                    dma_queue_ଠେଲ(dma_q, dma_make_data(w_tile, src0_row + ଠେଲ_ct * tile_row_stride),
                                    aligned_tile_size, tile_size, tile_size, n_k_tiles_w);
                    ଠେଲ_ct++;
                }
            }
        }
    }
}

ସ୍ଥିର ପୂର୍ଣ୍ଣସଂଖ୍ୟା hvx_mm_init_tiled_vec_dot(ସଂରଚନା htp_mm_context * mmctx, ଗଣନା htp_data_type type) {
    ବାଛ (type) {
        ସ୍ଥିତି HTP_TYPE_Q4_0:
            mmctx->type         = "q4_0_tiled-f32";
            mmctx->vec_dot_32x1 = tiled_vec_dot_q4_0_32x1;
            mmctx->vec_dot_32x2 = tiled_vec_dot_q4_0_32x2;
            ଫେରନ୍ତୁ 0;
        ସ୍ଥିତି HTP_TYPE_Q4_1:
        ସ୍ଥିତି HTP_TYPE_Q4_K:
            mmctx->type         = "q4_1_tiled-f32";
            mmctx->vec_dot_32x1 = tiled_vec_dot_q4_1_32x1;
            mmctx->vec_dot_32x2 = tiled_vec_dot_q4_1_32x2;
            ଫେରନ୍ତୁ 0;
        ସ୍ଥିତି HTP_TYPE_Q8_0:
            mmctx->type         = "q8_0_tiled-f32";
            mmctx->vec_dot_32x1 = tiled_vec_dot_q8_0_32x1;
            mmctx->vec_dot_32x2 = tiled_vec_dot_q8_0_32x2;
            ଫେରନ୍ତୁ 0;
        ସ୍ଥିତି HTP_TYPE_Q5_K:
            mmctx->type         = "q5_k_tiled-f32";
            mmctx->vec_dot_32x1 = tiled_vec_dot_q5_k_32x1;
            mmctx->vec_dot_32x2 = tiled_vec_dot_q5_k_32x2;
            ଫେରନ୍ତୁ 0;
        ସ୍ଥିତି HTP_TYPE_Q6_K:
            mmctx->type         = "q6_k_tiled-f32";
            mmctx->vec_dot_32x1 = tiled_vec_dot_q6_k_32x1;
            mmctx->vec_dot_32x2 = tiled_vec_dot_q6_k_32x2;
            ଫେରନ୍ତୁ 0;
        ସ୍ଥିତି HTP_TYPE_Q3_K:
            mmctx->type         = "q3_k_tiled-f32";
            mmctx->vec_dot_32x1 = tiled_vec_dot_q3_k_32x1;
            mmctx->vec_dot_32x2 = tiled_vec_dot_q3_k_32x2;
            ଫେରନ୍ତୁ 0;
        ସ୍ଥିତି HTP_TYPE_Q2_K:
            mmctx->type         = "q2_k_tiled-f32";
            mmctx->vec_dot_32x1 = tiled_vec_dot_q2_k_32x1;
            mmctx->vec_dot_32x2 = tiled_vec_dot_q2_k_32x2;
            ଫେରନ୍ତୁ 0;
        ସ୍ଥିତି HTP_TYPE_Q2_0:
            mmctx->type         = "q2_0_tiled-f32";
            mmctx->vec_dot_32x1 = tiled_vec_dot_q2_0_32x1;
            mmctx->vec_dot_32x2 = tiled_vec_dot_q2_0_32x2;
            ଫେରନ୍ତୁ 0;
        ସ୍ଥିତି HTP_TYPE_IQ4_NL:
            mmctx->type         = "iq4nl_tiled-f32";
            mmctx->vec_dot_32x1 = tiled_vec_dot_iq4nl_32x1;
            mmctx->vec_dot_32x2 = tiled_vec_dot_iq4nl_32x2;
            ଫେରନ୍ତୁ 0;
        ସ୍ଥିତି HTP_TYPE_MXFP4:
            mmctx->type         = "mxfp4_tiled-f32";
            mmctx->vec_dot_32x1 = tiled_vec_dot_mxfp4_32x1;
            mmctx->vec_dot_32x2 = tiled_vec_dot_mxfp4_32x2;
            ଫେରନ୍ତୁ 0;
        ସ୍ଥିତି HTP_TYPE_Q1_0:
            mmctx->type         = "q1_0_tiled-f32";
            mmctx->vec_dot_32x1 = tiled_vec_dot_q1_0_32x1;
            mmctx->vec_dot_32x2 = tiled_vec_dot_q1_0_32x2;
            ଫେରନ୍ତୁ 0;
        ସ୍ଥିତି HTP_TYPE_F16:
            mmctx->type         = "f16_tiled-f16";
            mmctx->vec_dot_32x1 = tiled_vec_dot_f16_32x1;
            mmctx->vec_dot_32x2 = tiled_vec_dot_f16_32x2;
            ଫେରନ୍ତୁ 0;
        ପୂର୍ବନିର୍ଦ୍ଧାରିତ:
            ଫେରନ୍ତୁ -1;
    }
}

ସ୍ଥିର ପୂର୍ଣ୍ଣସଂଖ୍ୟା hvx_mm_matmul(ସଂରଚନା htp_ops_context * octx) {
    htp_matmul_tensors_preamble;

    ସଂରଚନା htp_thread_trace * tr = &octx->ctx->trace[0];
    htp_trace_event_start(tr, HTP_TRACE_EVT_INIT, 0);

    ସଂରଚନା htp_mm_context mmctx_struct = {0};
    ସଂରଚନା htp_mm_context * mmctx = &mmctx_struct;
    mmctx->octx = octx;
    mmctx->act = act;

    ଅଚଳ ସଂରଚନା htp_mm_kernel_params * kparams = (ଅଚଳ ସଂରଚନା htp_mm_kernel_params *) octx->kernel_params;

    ଅଚଳ uint32_t src0_nrows = ne01;
    ଅଚଳ uint32_t act_nrows  = ne11 * ne12 * ne13;
    mmctx->act_nrows = act_nrows;

    uint32_t src0_row_start = 0;
    uint32_t src0_row_end   = src0_nrows;

    ଯଦି (octx->ctx->mdev.ଗଣନା_2 > 1) {
        ଅଚଳ bool can_split = htp_tensor_can_row_partition(dst, ଆକାର(ଦଶମିକ));
        ଅଚଳ ସଂରଚନା htp_tensor_mdev_range range = htp_tensor_mdev_partition(src0_nrows, can_split ? 32 : 0, octx->ctx->mdev.idx, octx->ctx->mdev.ଗଣନା_2, &octx->ctx->mdev.ଗଣନା_div);
        src0_row_start = range.start;
        src0_row_end   = range.start + range.ଗଣନା_2;
    }

    ଯଦି (src0_row_start >= src0_row_end) {
        ଫେରନ୍ତୁ HTP_STATUS_OK;
    }

    ଅଚଳ uint32_t nrows = src0_row_end - src0_row_start;
    mmctx->src0_row_start = src0_row_start;
    mmctx->src0_row_end   = src0_row_end;

    ଅଚଳ bool is_tiled = htp_tensor_is_tiled(src0);

    // Compute src0_nrows_per_thread
    mmctx->src0_nrows_per_thread  = fastdiv(nrows + octx->n_threads - 1, &octx->n_threads_div);
    ଯଦି (is_tiled) {
        mmctx->src0_nrows_per_thread = hex_round_up(mmctx->src0_nrows_per_thread, 32);
    } ଅନ୍ୟଥା {
        mmctx->src0_nrows_per_thread += (mmctx->src0_nrows_per_thread & 1); // round up to even
    }

    ଅଚଳ size_t src0_row_size = nb01;
    ଅଚଳ size_t dst_row_size  = nb1;
    size_t       act_row_size  = nb11;

    ଅଚଳ size_t src0_row_size_padded = hex_round_up(src0_row_size, 128);

    worker_callback_t quant_task_func;
    worker_callback_t matmul_job_func;
    uint32_t n_quant_tasks = 1;
    ଅଚଳ bool is_batched = (ne12 > 1 || ne13 > 1 || ne02 > 1 || ne03 > 1);
    ଯଦି (is_batched) {
        matmul_job_func = is_tiled ? hvx_mm_4d_tiled : hvx_mm_4d_flat;
    } ଅନ୍ୟଥା ଯଦି (act_nrows > 1) {
        matmul_job_func = is_tiled ? hvx_mm_2d_tiled : hvx_mm_2d_flat;
    } ଅନ୍ୟଥା {
        matmul_job_func = is_tiled ? hvx_mv_2d_tiled : hvx_mv_2d_flat;
    }

    bool need_quant = true;

    ବାଛ (kparams->kernel_type) {
        ସ୍ଥିତି HTP_MM_KERNEL_HVX_F16_F16_VTCM:
            quant_task_func        = (act->type == HTP_TYPE_F32) ? quantize_f32_f16 : quantize_f16_f16;
            need_quant             = (act->type == HTP_TYPE_F32);
            mmctx->type            = (act->type == HTP_TYPE_F32) ? "f32-f16" : "f16-f16";
            mmctx->vec_dot_1x1     = flat_vec_dot_f16_f16_aa_1x1;
            mmctx->vec_dot_2x1     = flat_vec_dot_f16_f16_aa_2x1;
            mmctx->vec_dot_2x2     = flat_vec_dot_f16_f16_aa_2x2;
            act_row_size           = kparams->act_row_size;
            ଭାଙ୍ଗ;

        ସ୍ଥିତି HTP_MM_KERNEL_HVX_F32_F32_VTCM:
            quant_task_func        = ଖାଲି;
            need_quant             = false;
            mmctx->type            = "f32-f32";
            mmctx->vec_dot_1x1     = flat_vec_dot_f32_f32_aa_1x1;
            mmctx->vec_dot_2x1     = flat_vec_dot_f32_f32_aa_2x1;
            mmctx->vec_dot_2x2     = flat_vec_dot_f32_f32_aa_2x2;
            act_row_size           = kparams->act_row_size;
            ଭାଙ୍ଗ;

        ସ୍ଥିତି HTP_MM_KERNEL_HVX_BF16_F32_VTCM:
            quant_task_func        = ଖାଲି;
            need_quant             = false;
            mmctx->type            = "bf16-f32";
            mmctx->vec_dot_1x1     = flat_vec_dot_bf16_f32_aa_1x1;
            mmctx->vec_dot_2x1     = flat_vec_dot_bf16_f32_aa_2x1;
            mmctx->vec_dot_2x2     = flat_vec_dot_bf16_f32_aa_2x2;
            act_row_size           = kparams->act_row_size;
            ଭାଙ୍ଗ;

        ସ୍ଥିତି HTP_MM_KERNEL_HVX_QUANT_BLOCK:
        ସ୍ଥିତି HTP_MM_KERNEL_HVX_QUANT_ROW:
        ପୂର୍ବନିର୍ଦ୍ଧାରିତ: {
            ଯଦି (hvx_mm_init_tiled_vec_dot(mmctx, src0->type) != 0) {
                ଫେରନ୍ତୁ HTP_STATUS_NO_SUPPORT;
            }

            ଅଚଳ uint32_t qk = QK_Q8_0_TILED;
            ଅଚଳ uint32_t nb = (ne10 + qk - 1) / qk;
            ଅଚଳ uint32_t total_nb = act_nrows * nb;

            ଯଦି (act_nrows < octx->n_threads && !is_batched) {
                n_quant_tasks = MIN(total_nb, octx->n_threads);
                quant_task_func = htp_mm_act_quant_block_func(src0->type, act->type);
                ପାଇଁ (uint32_t ith = 0; ith < n_quant_tasks; ++ith) {
                    uint32_t ib_first = (total_nb * ith) / n_quant_tasks;
                    uint32_t ib_last  = (total_nb * (ith + 1)) / n_quant_tasks;
                    mmctx->quant_ib_first[ith] = ib_first;
                    mmctx->quant_ib_last[ith]  = ib_last;
                    mmctx->quant_r[ith]        = ib_first / nb;
                    mmctx->quant_c[ith]        = ib_first % nb;
                }
            } ଅନ୍ୟଥା {
                n_quant_tasks = MIN(act_nrows, octx->n_threads);
                quant_task_func = htp_mm_act_quant_row_func(src0->type, act->type);
            }
            act_row_size = kparams->act_row_size;
            ଭାଙ୍ଗ;
        }
    }

    ଅଚଳ uint32_t m_chunk = (kparams->m_chunk > 0 && (uint32_t) kparams->m_chunk < act_nrows)
                           ? (uint32_t) kparams->m_chunk : act_nrows;
    ଅଚଳ uint32_t m_layout_rows = m_chunk;

    ସଂରଚନା htp_mm_hvx_vtcm_layout L;
    htp_mm_hvx_vtcm_layout_build(&L, kparams->kernel_type, src0->type, ne10, m_layout_rows, octx->n_threads,
                                 dst_row_size, src0_row_size, act_row_size, src2 ? src2->nb[1] : 0, kparams->n_prefetch, false, false);

    ଯଦି (kparams->kernel_type == HTP_MM_KERNEL_HVX_F16_F16_VTCM ||
        kparams->kernel_type == HTP_MM_KERNEL_HVX_F32_F32_VTCM ||
        kparams->kernel_type == HTP_MM_KERNEL_HVX_BF16_F32_VTCM ||
        kparams->kernel_type == HTP_MM_KERNEL_HVX_QUANT_ROW ||
        kparams->kernel_type == HTP_MM_KERNEL_HVX_QUANT_BLOCK) {
        mmctx->vtcm_act_size_per_thread = L.act_bytes;
    } ଅନ୍ୟଥା {
        mmctx->vtcm_act_size_per_thread = fastdiv(L.act_bytes, &octx->n_threads_div);
    }

    mmctx->vtcm_src0_size_per_thread = fastdiv(L.src0_bytes, &octx->n_threads_div);
    mmctx->vtcm_dst_size_per_thread  = fastdiv(L.dst_bytes, &octx->n_threads_div);

    ଅଚଳ size_t vtcm_size = L.total_bytes;

    FARF(ଉପର_2, "matmul-%s : src0-vtcm-size %zu act-vtcm-size %zu dst-vtcm-size %zu (%zu)\n", mmctx->type,
         L.src0_bytes, L.act_bytes, L.dst_bytes, vtcm_size);

    FARF(ଉପର_2, "matmul-%s : %ux%ux%ux%u * %ux%ux%ux%u-> %ux%ux%ux%u (0x%p, 0x%p, 0x%p)\n", mmctx->type, src0->ne[0],
         src0->ne[1], src0->ne[2], src0->ne[3], act->ne[0], act->ne[1], act->ne[2], act->ne[3], dst->ne[0],
         dst->ne[1], dst->ne[2], dst->ne[3], src0->data, act->data, dst->data);

    ଯଦି (octx->ctx->vtcm_size < vtcm_size) {
        FARF(ERROR, "matmul-%s : current VTCM reservation %zu is too small, needed %zu\n", mmctx->type,
             octx->ctx->vtcm_size, vtcm_size);
        ଫେରନ୍ତୁ HTP_STATUS_VTCM_TOO_SMALL;
    }

    uint8_t * ଅଚଳ base = (uint8_t *) octx->ctx->vtcm_base;
    mmctx->vtcm_act      = VTCM_LAYOUT_PTR(uint8_t, base, L.off_act);
    mmctx->vtcm_src0     = VTCM_LAYOUT_PTR(uint8_t, base, L.off_src0);
    mmctx->vtcm_bias     = VTCM_LAYOUT_PTR(uint8_t, base, L.off_bias);
    mmctx->vtcm_dst      = VTCM_LAYOUT_PTR(uint8_t, base, L.off_dst);
    mmctx->vtcm_act_raw  = VTCM_LAYOUT_PTR(uint8_t, base, L.off_act_raw);

    mmctx->vtcm_src0_stride = src0_row_size_padded;
    mmctx->vtcm_act_stride  = act_row_size;
    ଯଦି (kparams->kernel_type == HTP_MM_KERNEL_HVX_QUANT_BLOCK || kparams->kernel_type == HTP_MM_KERNEL_HVX_QUANT_ROW) {
        mmctx->vtcm_act_raw_stride = hex_round_up(ne10 * ଆକାର(ଦଶମିକ), QK_Q8_0_TILED * ଆକାର(ଦଶମିକ));
    } ଅନ୍ୟଥା ଯଦି (kparams->kernel_type == HTP_MM_KERNEL_HVX_F16_F16_VTCM) {
        mmctx->vtcm_act_raw_stride = hex_round_up(ne10 * ଆକାର(ଦଶମିକ), 128);
    } ଅନ୍ୟଥା {
        mmctx->vtcm_act_raw_stride = 0;
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_INIT, 0);

    ଯଦି (kparams->m_chunk > 0 && (uint32_t) kparams->m_chunk < act_nrows) {
        ପାଇଁ (uint32_t m_start = 0; m_start < act_nrows; m_start += m_chunk) {
            ଅଚଳ uint32_t cur_m_rows = MIN(act_nrows - m_start, m_chunk);
            mmctx->cur_m_start = m_start;
            mmctx->cur_m_rows  = cur_m_rows;

            ଯଦି (need_quant) {
                hvx_mm_transfer_act_dma(octx, kparams, act, mmctx->vtcm_act_raw, mmctx->vtcm_act_raw_stride, m_start, cur_m_rows);

                ଅଚଳ uint32_t qk = QK_Q8_0_TILED;
                ଅଚଳ uint32_t nb = (ne10 + qk - 1) / qk;
                ଅଚଳ uint32_t total_nb = cur_m_rows * nb;
                uint32_t quant_tasks;
                work_queue_func_t q_func;
                ଯଦି (cur_m_rows < octx->n_threads && (kparams->kernel_type == HTP_MM_KERNEL_HVX_QUANT_BLOCK || kparams->kernel_type == HTP_MM_KERNEL_HVX_QUANT_ROW)) {
                    quant_tasks = MIN(total_nb, octx->n_threads);
                    q_func = htp_mm_act_quant_block_func(src0->type, act->type);
                    ପାଇଁ (uint32_t ith = 0; ith < quant_tasks; ++ith) {
                        uint32_t ib_first = (total_nb * ith) / quant_tasks;
                        uint32_t ib_last  = (total_nb * (ith + 1)) / quant_tasks;
                        mmctx->quant_ib_first[ith] = ib_first;
                        mmctx->quant_ib_last[ith]  = ib_last;
                        mmctx->quant_r[ith]        = ib_first / nb;
                        mmctx->quant_c[ith]        = ib_first % nb;
                    }
                } ଅନ୍ୟଥା {
                    quant_tasks = MIN(cur_m_rows, octx->n_threads);
                    q_func = quant_task_func;
                    mmctx->n_quant_rows_per_thread = (cur_m_rows + quant_tasks - 1) / quant_tasks;
                }
                mmctx->n_quant_tasks = quant_tasks;
                work_queue_run(octx->ctx->work_queue, q_func, mmctx, quant_tasks);
            } ଅନ୍ୟଥା {
                hvx_mm_transfer_act_dma(octx, kparams, act, mmctx->vtcm_act, mmctx->vtcm_act_stride, m_start, cur_m_rows);
            }

            work_queue_run(octx->ctx->work_queue, matmul_job_func, mmctx, octx->n_threads);
        }
    } ଅନ୍ୟଥା {
        mmctx->cur_m_start = 0;
        mmctx->cur_m_rows  = act_nrows;

        ଯଦି (need_quant) {
            hvx_mm_transfer_act_dma(octx, kparams, act, mmctx->vtcm_act_raw, mmctx->vtcm_act_raw_stride, 0, act_nrows);
            mmctx->n_quant_rows_per_thread = (act_nrows + n_quant_tasks - 1) / n_quant_tasks;
            mmctx->n_quant_tasks = n_quant_tasks;
            work_queue_run(octx->ctx->work_queue, quant_task_func, mmctx, n_quant_tasks);
        } ଅନ୍ୟଥା {
            hvx_mm_transfer_act_dma(octx, kparams, act, mmctx->vtcm_act, mmctx->vtcm_act_stride, 0, act_nrows);
        }

        work_queue_run(octx->ctx->work_queue, matmul_job_func, mmctx, octx->n_threads);
    }

    ଫେରନ୍ତୁ HTP_STATUS_OK;
}

ସ୍ଥିର ଶୂନ୍ୟ dequantize_tiled_worker_loop(unsigned ପୂର୍ଣ୍ଣସଂଖ୍ୟା n, unsigned ପୂର୍ଣ୍ଣସଂଖ୍ୟା i, ଶୂନ୍ୟ * data) {
    tiled_dequantize_state_t * state = (tiled_dequantize_state_t *) data;
    ସଂରଚନା htp_thread_trace * tr = &state->traces[i];
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_W_DEQUANT, i);
    ପାଇଁ (unsigned ପୂର୍ଣ୍ଣସଂଖ୍ୟା task_id = i; task_id < (unsigned ପୂର୍ଣ୍ଣସଂଖ୍ୟା) state->n_tasks; task_id += n) {
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା start = task_id * state->n_tiles_per_task;
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା end   = hex_smin(start + state->n_tiles_per_task, state->n_tot_tiles);
        state->task_fn(state, start, end);
    }
    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_W_DEQUANT, i);
}

ସ୍ଥିର ଶୂନ୍ୟ transfer_output_chunk_worker_fn(unsigned ପୂର୍ଣ୍ଣସଂଖ୍ୟା n, unsigned ପୂର୍ଣ୍ଣସଂଖ୍ୟା i, ଶୂନ୍ୟ *data) {
    output_transfer_task_state_t *st = (output_transfer_task_state_t *) data;

    ସଂରଚନା htp_thread_trace * tr = &st->traces[i];

    ପୂର୍ଣ୍ଣସଂଖ୍ୟା start_chunk_idx = i * st->n_chunks_per_task;
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_O_PROC, start_chunk_idx);

    ପାଇଁ (unsigned ପୂର୍ଣ୍ଣସଂଖ୍ୟା task_id = i; task_id < (unsigned ପୂର୍ଣ୍ଣସଂଖ୍ୟା)st->n_tasks; task_id += n) {
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା    chunk_idx  = task_id * st->n_chunks_per_task;
        size_t chunk_size = hex_smin(st->n_tot_chunks - chunk_idx, st->n_chunks_per_task);

        ଦଶମିକ        *dst      = st->dst      + chunk_idx * st->dst_stride;
        ଅଚଳ ଦଶମିକ  *src2     = st->src2     ? (st->src2     + chunk_idx * st->src2_stride) : ଖାଲି;
        transfer_output_chunk_fp16_to_fp32(dst, src2, st->vtcm_src, chunk_idx, chunk_size, st->n_cols, st->dst_stride, st->src2_stride, st->dst_cols);
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_O_PROC, start_chunk_idx);
}

ପ୍ରକାରବ୍ୟାଖ୍ୟା ସଂରଚନା {
    ସଂରଚନା htp_context            * ctx;
    ସଂରଚନା htp_thread_trace       * traces;
    __fp16                        * dst;
    dma_addr_t                      act_dma_addr;
    ଦଶମିକ                         * vtcm_f32_act;
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

ପ୍ରକାରବ୍ୟାଖ୍ୟା ସଂରଚନା {
    ସଂରଚନା htp_context            * ctx;
    ସଂରଚନା htp_thread_trace       * traces;
    __fp16                        * dst;
    dma_addr_t                      act_dma_addr;
    ଦଶମିକ                         * vtcm_f32_act;
    uint32_t                        n_rows;
    uint32_t                        k_block;
    uint32_t                        k_stride;
    uint32_t                        k_valid;
    uint32_t                        n_col_chunks;
    ସଂରଚନା fastdiv_ମୂଲ୍ୟଗୁଡ଼ିକ           n_threads_div;
    size_t                          vtcm_f32_act_bytes;
    uint32_t                        dma_step_rows;
    uint32_t                        dma_step_rows_shift;
} activation_transfer_col_chunk_state_t;

ସ୍ଥିର ଶୂନ୍ୟ transfer_activation_chunk_fp32_to_fp16_dma_pipelined_col_chunk(
        dma_queue *dma_q,
        __fp16 *restrict vtcm_dst,
        dma_addr_t act_dma_addr,
        uint32_t n_rows,
        uint32_t k_block,
        uint32_t k_stride,
        uint32_t k_chunk_valid,
        uint32_t c_first,
        uint32_t c_len,
        ଦଶମିକ *thread_f32_act,
        ସଂରଚନା htp_thread_trace *tr,
        uint32_t dma_step_rows,
        uint32_t dma_step_rows_shift) {

    ଅଚଳ uint32_t R = dma_step_rows;
    ଅଚଳ uint32_t n_rows_padded = hex_align_up(n_rows, HTP_MM_HMX_TILE_N_ROWS);

    ଅଚଳ uint32_t n_steps = n_rows_padded >> dma_step_rows_shift;

    // Push step 0
    ଯଦି (n_steps > 0 && n_rows > 0) {
        uint32_t nrows_to_fetch = hex_smin(n_rows, R);
        dma_queue_ଠେଲ(dma_q, dma_make_data(thread_f32_act, act_dma_addr + (size_t) c_first * ଆକାର(ଦଶମିକ)),
                       c_len * ଆକାର(ଦଶମିକ), k_stride * ଆକାର(ଦଶମିକ), k_chunk_valid * ଆକାର(ଦଶମିକ), nrows_to_fetch);
    }
    // Push step 1
    ଯଦି (n_steps > 1) {
        uint32_t next_r = R * 1;
        ଯଦି (next_r < n_rows) {
            uint32_t nrows_to_fetch = hex_smin(n_rows - next_r, R);
            ଦଶମିକ *next_buf = thread_f32_act + 1 * R * c_len;
            dma_queue_ଠେଲ(dma_q, dma_make_data(next_buf, act_dma_addr + ((size_t) next_r * k_stride + c_first) * ଆକାର(ଦଶମିକ)),
                           c_len * ଆକାର(ଦଶମିକ), k_stride * ଆକାର(ଦଶମିକ), k_chunk_valid * ଆକାର(ଦଶମିକ), nrows_to_fetch);
        }
    }
    ପାଇଁ (uint32_t s = 0; s < n_steps; ++s) {
        uint32_t r = s << dma_step_rows_shift;
        ଦଶମିକ *curr_buf = thread_f32_act;

        ଯଦି (r < n_rows) {
            curr_buf = (ଦଶମିକ *) dma_queue_କାଢ଼(dma_q).dst;
        }

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_A_PREP, r);
        ପାଇଁ (uint32_t p = 0; p < (R >> 1); ++p) {
            uint32_t row_idx = r + (p << 1);
            ଦଶମିକ *pair_buf = curr_buf + (p << 1) * c_len;
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
        ଯଦି (next_r < n_rows) {
            uint32_t nrows_to_fetch = hex_smin(n_rows - next_r, R);
            dma_queue_ଠେଲ(dma_q, dma_make_data(curr_buf, act_dma_addr + ((size_t) next_r * k_stride + c_first) * ଆକାର(ଦଶମିକ)),
                           c_len * ଆକାର(ଦଶମିକ), k_stride * ଆକାର(ଦଶମିକ), k_chunk_valid * ଆକାର(ଦଶମିକ), nrows_to_fetch);
        }
    }
}

ସ୍ଥିର ଶୂନ୍ୟ transfer_activation_chunk_col_chunk_worker_fn(unsigned ପୂର୍ଣ୍ଣସଂଖ୍ୟା n, unsigned ପୂର୍ଣ୍ଣସଂଖ୍ୟା i, ଶୂନ୍ୟ *data) {
    activation_transfer_col_chunk_state_t *st = (activation_transfer_col_chunk_state_t *) data;
    ସଂରଚନା htp_thread_trace * tr = &st->traces[i];

    uint32_t n_blocks = st->k_block / 32;
    uint32_t b_first = fastdiv(n_blocks * i, &st->n_threads_div);
    uint32_t b_last  = fastdiv(n_blocks * (i + 1), &st->n_threads_div);
    uint32_t c_first = b_first * 32;
    uint32_t c_last = b_last * 32;
    uint32_t c_len = c_last - c_first;

    ଯଦି (c_len == 0) {
        ଫେରନ୍ତୁ;
    }

    uint32_t k_chunk_valid = 0;
    ଯଦି (st->k_valid > c_first) {
        k_chunk_valid = hex_smin(st->k_valid, c_last) - c_first;
    }

    __fp16 *dst = st->dst;

    size_t thread_scratch_bytes = hex_align_down(fastdiv(st->vtcm_f32_act_bytes, &st->n_threads_div), 128);
    ଦଶମିକ *thread_f32_act = (ଦଶମିକ *)((ଅକ୍ଷର *)st->vtcm_f32_act + i * thread_scratch_bytes);

    transfer_activation_chunk_fp32_to_fp16_dma_pipelined_col_chunk(
        st->ctx->dma[i], dst, st->act_dma_addr, st->n_rows, st->k_block, st->k_stride, k_chunk_valid,
        c_first, c_len, thread_f32_act, tr, st->dma_step_rows, st->dma_step_rows_shift
    );
}

ସ୍ଥିର ଶୂନ୍ୟ transfer_activation_chunk_to_fp16_dma_pipelined(
        dma_queue *dma_q,
        __fp16 *restrict vtcm_dst,
        dma_addr_t act_dma_addr,
        uint32_t n_rows,
        uint32_t k_block,
        uint32_t k_stride,
        uint32_t k_valid,
        uint8_t *thread_act,
        uint32_t act_elem_size,
        ସଂରଚନା htp_thread_trace *tr,
        uint32_t dma_step_rows,
        uint32_t dma_step_rows_shift) {

    ଅଚଳ uint32_t R = dma_step_rows;
    ଅଚଳ uint32_t n_rows_padded = hex_align_up(n_rows, HTP_MM_HMX_TILE_N_ROWS);

    ଅଚଳ uint32_t n_steps = n_rows_padded >> dma_step_rows_shift;

    ଅଚଳ bool act_is_f16 = (act_elem_size == ଆକାର(__fp16));
    ଅଚଳ size_t row_bytes = (size_t) k_block * act_elem_size;  // staging row (DMA dst stride)
    ଅଚଳ size_t src_row_bytes = (size_t) k_stride * act_elem_size;  // DDR row (DMA src stride)
    ଅଚଳ size_t width_bytes = (size_t) k_valid * act_elem_size;

    // Push step 0
    ଯଦି (n_steps > 0 && n_rows > 0) {
        uint32_t nrows_to_fetch = hex_smin(n_rows, R);
        dma_queue_ଠେଲ(dma_q, dma_make_data(thread_act, act_dma_addr),
                       row_bytes, src_row_bytes, width_bytes, nrows_to_fetch);
    }
    // Push step 1 (if valid)
    ଯଦି (n_steps > 1) {
        uint32_t next_r = R * 1;
        ଯଦି (next_r < n_rows) {
            uint32_t nrows_to_fetch = hex_smin(n_rows - next_r, R);
            uint8_t *next_buf = thread_act + 1 * R * row_bytes;
            dma_queue_ଠେଲ(dma_q, dma_make_data(next_buf, act_dma_addr + (size_t) next_r * src_row_bytes),
                           row_bytes, src_row_bytes, width_bytes, nrows_to_fetch);
        }
    }
    ପାଇଁ (uint32_t s = 0; s < n_steps; ++s) {
        uint32_t r = s << dma_step_rows_shift;
        uint8_t *curr_buf = thread_act;

        ଯଦି (r < n_rows) {
            curr_buf = (uint8_t *) dma_queue_କାଢ଼(dma_q).dst;
        }

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_A_PREP, r);
        // Two copies of the pair loop so the type is resolved once per step and the
        // row-pair kernels stay direct (inlinable) calls.
        ଯଦି (act_is_f16) {
            ପାଇଁ (uint32_t p = 0; p < (R >> 1); ++p) {
                uint32_t row_idx = r + (p << 1);
                ଅଚଳ __fp16 *pair_buf = (ଅଚଳ __fp16 *) (curr_buf + (p << 1) * row_bytes);
                bool r0_valid = ((row_idx + 0) < n_rows);
                bool r1_valid = ((row_idx + 1) < n_rows);

                transfer_activation_row_pair_f16_to_f16(vtcm_dst, pair_buf, pair_buf + k_block, row_idx, k_block, k_valid, r0_valid, r1_valid);
            }
        } ଅନ୍ୟଥା {
            ପାଇଁ (uint32_t p = 0; p < (R >> 1); ++p) {
                uint32_t row_idx = r + (p << 1);
                ଅଚଳ ଦଶମିକ *pair_buf = (ଅଚଳ ଦଶମିକ *) (curr_buf + (p << 1) * row_bytes);
                bool r0_valid = ((row_idx + 0) < n_rows);
                bool r1_valid = ((row_idx + 1) < n_rows);

                transfer_activation_row_pair_fp32_to_fp16(vtcm_dst, pair_buf, pair_buf + k_block, row_idx, k_block, k_valid, r0_valid, r1_valid);
            }
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_A_PREP, r);

        // Push step s + 2
        uint32_t next_s = s + 2;
        uint32_t next_r = next_s << dma_step_rows_shift;
        ଯଦି (next_r < n_rows) {
            uint32_t nrows_to_fetch = hex_smin(n_rows - next_r, R);
            dma_queue_ଠେଲ(dma_q, dma_make_data(curr_buf, act_dma_addr + (size_t) next_r * src_row_bytes),
                           row_bytes, src_row_bytes, width_bytes, nrows_to_fetch);
        }
    }
}

ସ୍ଥିର ଶୂନ୍ୟ transfer_activation_chunk_worker_fn(unsigned ପୂର୍ଣ୍ଣସଂଖ୍ୟା n, unsigned ପୂର୍ଣ୍ଣସଂଖ୍ୟା i, ଶୂନ୍ୟ *data) {
    activation_transfer_task_state_t *st = (activation_transfer_task_state_t *) data;

    ସଂରଚନା htp_thread_trace * tr = &st->traces[i];

    ପାଇଁ (unsigned ପୂର୍ଣ୍ଣସଂଖ୍ୟା task_id = i; task_id < (unsigned ପୂର୍ଣ୍ଣସଂଖ୍ୟା)st->n_tasks; task_id += n) {
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା    chunk_idx  = task_id * st->n_chunks_per_task;
        size_t chunk_size = hex_smin(st->n_tot_chunks - chunk_idx, st->n_chunks_per_task);

        __fp16      *dst = st->dst + chunk_idx * st->k_block;
        ଅଚଳ dma_addr_t act_dma_addr = st->act_dma_addr + (size_t) chunk_idx * st->k_stride * st->act_elem_size;

        uint8_t *thread_act = (uint8_t *) st->vtcm_f32_act + i * st->vtcm_f32_act_bytes_per_thread;
        transfer_activation_chunk_to_fp16_dma_pipelined(
            st->ctx->dma[i], dst, act_dma_addr, chunk_size, st->k_block, st->k_stride, st->k_valid, thread_act, st->act_elem_size, tr, st->dma_step_rows, st->dma_step_rows_shift
        );
    }
}

ପ୍ରକାରବ୍ୟାଖ୍ୟା ସଂରଚନା {
    ସଂରଚନା htp_thread_trace       * traces;
    ଅଚଳ ସଂରଚନା mmid_row_mapping * matrix_rows;
    __fp16                        * dst;
    ଅଚଳ ଦଶମିକ                   * src;
    uint32_t                        n_tasks;
    uint32_t                        n_tot_chunks;
    uint32_t                        n_chunks_per_task;
    uint32_t                        k_block;
    uint32_t                        cur_a;
    uint32_t                        mapping_stride;
    uint32_t                        ne11;
    ସଂରଚନା fastdiv_ମୂଲ୍ୟଗୁଡ଼ିକ           ne11_div;
    size_t                          nb11;
    size_t                          nb12;
    uint32_t                        start_row;
    uint32_t                        cne1;
    uint32_t                        k_valid;
} activation_transfer_gathered_task_state_t;

ପ୍ରକାରବ୍ୟାଖ୍ୟା ସଂରଚନା {
    ସଂରଚନା htp_thread_trace       * traces;
    ଅଚଳ ସଂରଚନା mmid_row_mapping * matrix_rows;
    ଅଚଳ __fp16                  * vtcm_src;
    ଦଶମିକ                         * dst;
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

ସ୍ଥିର ଶୂନ୍ୟ transfer_activation_chunk_gathered_worker_fn(unsigned ପୂର୍ଣ୍ଣସଂଖ୍ୟା n, unsigned ପୂର୍ଣ୍ଣସଂଖ୍ୟା i, ଶୂନ୍ୟ *data) {
    activation_transfer_gathered_task_state_t *st = data;
    ସଂରଚନା htp_thread_trace * tr = &st->traces[i];
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା chunk_idx      = i;
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା chunk_size     = st->n_chunks_per_task;
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା vtcm_start_row = chunk_idx * chunk_size;
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା start_row      = st->start_row + vtcm_start_row;
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା n_rows         = hex_smin(st->cne1 - start_row, chunk_size);
    ଯଦି (n_rows > 0) {
        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_A_PREP, chunk_idx);
        transfer_activation_chunk_fp32_to_fp16_gathered(
            st->dst, st->src, start_row, vtcm_start_row, n_rows, st->k_block,
            st->matrix_rows, st->cur_a, st->mapping_stride,
            st->ne11, &st->ne11_div, st->nb11, st->nb12, st->cne1, st->k_valid);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_A_PREP, chunk_idx);
    }
}

ସ୍ଥିର ଶୂନ୍ୟ transfer_activation_chunk_gathered_worker_flat_fn(unsigned ପୂର୍ଣ୍ଣସଂଖ୍ୟା n, unsigned ପୂର୍ଣ୍ଣସଂଖ୍ୟା i, ଶୂନ୍ୟ *data) {
    activation_transfer_gathered_task_state_t *st = data;
    ସଂରଚନା htp_thread_trace * tr = &st->traces[i];
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା chunk_idx = i;
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା chunk_size = st->n_chunks_per_task;
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା vtcm_start_row = chunk_idx * chunk_size;
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା start_row = st->start_row + vtcm_start_row;
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା n_rows = hex_smin(st->cne1 - start_row, chunk_size);
    ଯଦି (n_rows > 0) {
        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_A_PREP, chunk_idx);
        transfer_activation_chunk_fp32_to_fp16_gathered_flat(
            st->dst, st->src, start_row, vtcm_start_row, n_rows, st->k_block,
            st->matrix_rows, st->cur_a, st->mapping_stride,
            st->nb12, st->cne1, st->k_valid);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_A_PREP, chunk_idx);
    }
}

ସ୍ଥିର ଶୂନ୍ୟ transfer_output_chunk_scattered_worker_fn(unsigned ପୂର୍ଣ୍ଣସଂଖ୍ୟା n, unsigned ପୂର୍ଣ୍ଣସଂଖ୍ୟା i, ଶୂନ୍ୟ *data) {
    output_transfer_scattered_task_state_t *st = data;
    ସଂରଚନା htp_thread_trace * tr = &st->traces[i];
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା chunk_idx = i;
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା chunk_size = st->n_chunks_per_task;
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା vtcm_start_row = chunk_idx * chunk_size;
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା start_row = st->start_row + vtcm_start_row;
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା n_rows = hex_smin(st->cne1 - start_row, chunk_size);
    ଯଦି (n_rows > 0) {
        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_O_PROC, chunk_idx);
        transfer_output_chunk_fp16_to_fp32_scattered(
            st->dst, st->vtcm_src, start_row, vtcm_start_row, n_rows, st->n_cols,
            st->matrix_rows, st->cur_a, st->mapping_stride,
            st->dst_nb1, st->dst_nb2, st->cne1);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_O_PROC, chunk_idx);
    }
}

// --- HMX Dispatchers & Entry Points ---

ସ୍ଥିର ଧାଡ଼ିଭିତରେ tiled_dequant_task_fn_t hmx_get_dequant_worker_fn(ପୂର୍ଣ୍ଣସଂଖ୍ୟା weight_type, bool is_tiled) {
    ବାଛ (weight_type) {
        ସ୍ଥିତି HTP_TYPE_Q4_0:   ଫେରନ୍ତୁ dequantize_tiled_weight_to_fp16_task_q4_0;
        ସ୍ଥିତି HTP_TYPE_IQ4_NL: ଫେରନ୍ତୁ dequantize_tiled_weight_to_fp16_task_iq4_nl;
        ସ୍ଥିତି HTP_TYPE_Q4_1:
        ସ୍ଥିତି HTP_TYPE_Q4_K:   ଫେରନ୍ତୁ dequantize_tiled_weight_to_fp16_task_q4_1;
        ସ୍ଥିତି HTP_TYPE_MXFP4:  ଫେରନ୍ତୁ dequantize_tiled_weight_to_fp16_task_mxfp4;
        ସ୍ଥିତି HTP_TYPE_Q8_0:   ଫେରନ୍ତୁ dequantize_tiled_weight_to_fp16_task_q8_0;
        ସ୍ଥିତି HTP_TYPE_Q5_K:   ଫେରନ୍ତୁ dequantize_tiled_weight_to_fp16_task_q5_k;
        ସ୍ଥିତି HTP_TYPE_Q6_K:   ଫେରନ୍ତୁ dequantize_tiled_weight_to_fp16_task_q6_k;
        ସ୍ଥିତି HTP_TYPE_Q3_K:   ଫେରନ୍ତୁ dequantize_tiled_weight_to_fp16_task_q3_k;
        ସ୍ଥିତି HTP_TYPE_Q2_K:   ଫେରନ୍ତୁ dequantize_tiled_weight_to_fp16_task_q2_k;
        ସ୍ଥିତି HTP_TYPE_Q1_0:   ଫେରନ୍ତୁ dequantize_tiled_weight_to_fp16_task_q1_0;
        ସ୍ଥିତି HTP_TYPE_Q2_0:   ଫେରନ୍ତୁ dequantize_tiled_weight_to_fp16_task_q2_0;
        ସ୍ଥିତି HTP_TYPE_F16:    ଫେରନ୍ତୁ is_tiled ? ଖାଲି : convert_f16_weight_to_fp16_tiles_task;
        ସ୍ଥିତି HTP_TYPE_BF16:   ଫେରନ୍ତୁ convert_bf16_weight_to_fp16_tiles_task;
        ସ୍ଥିତି HTP_TYPE_F32:    ଫେରନ୍ତୁ quantize_f32_weight_to_fp16_tiles_task;
        ପୂର୍ବନିର୍ଦ୍ଧାରିତ:              ଫେରନ୍ତୁ ଖାଲି;
    }
}

ସ୍ଥିର ଶୂନ୍ୟ dequantize_tiled_weight_chunk_to_fp16_tiles(
        ସଂରଚନା htp_context *ctx, __fp16 *vtcm_dst,
        ଅଚଳ ଶୂନ୍ୟ *weight_src_ddr,
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା n_cols, ପୂର୍ଣ୍ଣସଂଖ୍ୟା k_block,
        size_t row_stride, ପୂର୍ଣ୍ଣସଂଖ୍ୟା weight_type,
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା n_k_tiles, ସଂରଚନା fastdiv_ମୂଲ୍ୟଗୁଡ଼ିକ n_k_tiles_div,
        tiled_dequant_task_fn_t dequant_worker_fn, ପୂର୍ଣ୍ଣସଂଖ୍ୟା n_threads) {

    ଯଦି (!dequant_worker_fn) {
        ଫେରନ୍ତୁ;
    }

    ପ୍ରତିଜ୍ଞା(k_block % HTP_MM_HMX_TILE_N_COLS == 0);

    ଅଚଳ size_t n_col_tiles = hmx_ceil_div(n_cols, HTP_MM_HMX_TILE_N_COLS);
    ଅଚଳ uint32_t n_weight_tiles = k_block / htp_tensor_get_tile_k(weight_type);
    ଅଚଳ size_t n_tot_tiles = n_col_tiles * n_weight_tiles;

    size_t n_tiles_per_task = (n_threads == 1) ? n_tot_tiles : hmx_ceil_div(n_tot_tiles, n_threads);

    tiled_dequantize_state_t state;
    state.n_tasks          = (n_tot_tiles + n_tiles_per_task - 1) / n_tiles_per_task;
    state.n_tot_tiles      = n_tot_tiles;
    state.n_tiles_per_task = n_tiles_per_task;
    state.dst              = vtcm_dst;
    state.src              = (ଅଚଳ uint8_t *)weight_src_ddr;
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

    ଯଦି (state.n_tasks == 1 || n_threads == 1) {
        dequantize_tiled_worker_loop(1, 0, &state);
    } ଅନ୍ୟଥା {
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା n_tasks = hex_smin((ପୂର୍ଣ୍ଣସଂଖ୍ୟା) state.n_tasks, n_threads);
        worker_pool_run_func(ctx->worker_pool, dequantize_tiled_worker_loop, &state, n_tasks);
    }
}

ପ୍ରକାରବ୍ୟାଖ୍ୟା ସଂରଚନା {
    ସଂରଚନା htp_context      * ctx;
    ସଂରଚନା htp_thread_trace * traces;
    ଦଶମିକ                   * dst;
    ଅଚଳ __fp16            * vtcm_src;
    ଅଚଳ ଦଶମିକ             * src2;
    uint32_t                  n_rows;
    uint32_t                  n_cols;
    uint32_t                  dst_stride;
    uint32_t                  src2_stride;
    uint32_t                  dst_cols;
    ସଂରଚନା fastdiv_ମୂଲ୍ୟଗୁଡ଼ିକ     n_threads_div;
} output_transfer_col_chunk_state_t;

ସ୍ଥିର ଶୂନ୍ୟ transfer_output_chunk_col_chunk_worker_fn(unsigned ପୂର୍ଣ୍ଣସଂଖ୍ୟା n, unsigned ପୂର୍ଣ୍ଣସଂଖ୍ୟା i, ଶୂନ୍ୟ *data) {
    (ଶୂନ୍ୟ) n;
    output_transfer_col_chunk_state_t *st = (output_transfer_col_chunk_state_t *) data;
    ସଂରଚନା htp_thread_trace * tr = &st->traces[i];

    // Round up: the last block is partial when N is not 32-aligned. Its pad
    // columns are dropped by the dst_cols clamp inside the store.
    uint32_t n_blocks = hmx_ceil_div(st->n_cols, 32);
    uint32_t b_first  = fastdiv(n_blocks * i, &st->n_threads_div);
    uint32_t b_last   = fastdiv(n_blocks * (i + 1), &st->n_threads_div);
    uint32_t c_first  = b_first * 32;
    uint32_t c_last   = b_last * 32;
    uint32_t c_len    = c_last - c_first;

    ଯଦି (c_len == 0) ଫେରନ୍ତୁ;

    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_O_PROC, c_first);

    ଅଚଳ __fp16 *vtcm_src = st->vtcm_src + b_first * HTP_MM_HMX_TILE_N_ELMS;
    ଅଚଳ ଦଶମିକ *src2      = st->src2 ? (st->src2 + c_first) : ଖାଲି;
    ଦଶମିକ *dst             = st->dst + c_first;

    ପୂର୍ଣ୍ଣସଂଖ୍ୟା chunk_dst_cols = (ପୂର୍ଣ୍ଣସଂଖ୍ୟା)st->dst_cols - (ପୂର୍ଣ୍ଣସଂଖ୍ୟା)c_first;
    ଯଦି (chunk_dst_cols > 0) {
        transfer_output_chunk_fp16_to_fp32_col_chunk(
            dst, src2, vtcm_src, 0, st->n_rows, c_len, st->n_cols,
            st->dst_stride, st->src2_stride, (uint32_t)chunk_dst_cols
        );
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_O_PROC, c_first);
}

ସ୍ଥିର ଶୂନ୍ୟ transfer_output_chunk_threaded(ସଂରଚନା htp_context *ctx, ଦଶମିକ *dst, ଅଚଳ ଦଶମିକ *src2, ଅଚଳ __fp16 *vtcm_src,
                                              ପୂର୍ଣ୍ଣସଂଖ୍ୟା n_rows, ପୂର୍ଣ୍ଣସଂଖ୍ୟା n_cols, ପୂର୍ଣ୍ଣସଂଖ୍ୟା dst_stride, uint32_t src2_stride, ପୂର୍ଣ୍ଣସଂଖ୍ୟା dst_cols, ପୂର୍ଣ୍ଣସଂଖ୍ୟା n_threads) {
    ଯଦି (n_rows <= 0) ଫେରନ୍ତୁ;

    uint32_t n_blocks = hmx_ceil_div((uint32_t) n_cols, 32);
    ଯଦି (n_threads > 1 && n_blocks >= (uint32_t)n_threads) {
        ସଂରଚନା fastdiv_ମୂଲ୍ୟଗୁଡ଼ିକ n_threads_div = (n_threads == (ପୂର୍ଣ୍ଣସଂଖ୍ୟା)ctx->n_threads) ? ctx->n_threads_div : init_fastdiv_ମୂଲ୍ୟଗୁଡ଼ିକ(n_threads);
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
        ଫେରନ୍ତୁ;
    }

    size_t n_tot_chunks      = n_rows;
    size_t n_chunks_per_task = (n_threads == 1) ? n_tot_chunks : hmx_ceil_div(n_rows, n_threads);
    n_chunks_per_task        = hex_align_up(n_chunks_per_task, 2);

    ପୂର୍ଣ୍ଣସଂଖ୍ୟା actual_threads = hmx_ceil_div(n_rows, n_chunks_per_task);

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

    ଯଦି (actual_threads <= 1) {
        transfer_output_chunk_worker_fn(1, 0, &state);
    } ଅନ୍ୟଥା {
        worker_pool_run_func(ctx->worker_pool, transfer_output_chunk_worker_fn, &state, actual_threads);
    }
}

ସଂରଚନା activation_transfer_params {
    ସଂରଚନା htp_context *          ctx;
    __fp16 *                      dst;
    dma_addr_t                    act_dma_addr;
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା                           n_rows;
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା                           k_block;
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା                           k_stride;
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା                           n_threads;
    ଅଚଳ ସଂରଚନା fastdiv_ମୂଲ୍ୟଗୁଡ଼ିକ * act_threads_div;
    ଅଚଳ ସଂରଚନା fastdiv_ମୂଲ୍ୟଗୁଡ଼ିକ * k_div;
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା                           k_valid;
    ଦଶମିକ *                       vtcm_f32_act;
    size_t                        vtcm_f32_act_bytes;
    uint32_t                      act_elem_size;  // 4=F32 src1, 2=F16 src1
};

ସ୍ଥିର ଶୂନ୍ୟ transfer_activation_chunk_threaded(ଅଚଳ ସଂରଚନା activation_transfer_params * params) {
    ସଂରଚନା htp_context *          ctx                = params->ctx;
    __fp16 *                      dst                = params->dst;
    ଅଚଳ dma_addr_t              act_dma_addr       = params->act_dma_addr;
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା                           n_rows             = params->n_rows;
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା                           k_block            = params->k_block;
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା                           k_stride           = params->k_stride;
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା                           n_threads          = params->n_threads;
    ଅଚଳ ସଂରଚନା fastdiv_ମୂଲ୍ୟଗୁଡ଼ିକ * act_threads_div    = params->act_threads_div;
    ଅଚଳ ସଂରଚନା fastdiv_ମୂଲ୍ୟଗୁଡ଼ିକ * k_div              = params->k_div;
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା                           k_valid            = params->k_valid;
    ଦଶମିକ *                       vtcm_f32_act       = params->vtcm_f32_act;
    size_t                        vtcm_f32_act_bytes = params->vtcm_f32_act_bytes;
    // element size of the activation rows (4 = F32, 2 = F16).
    ଅଚଳ uint32_t                act_elem_size      = params->act_elem_size ? params->act_elem_size : (uint32_t) ଆକାର(ଦଶମିକ);

    ଯଦି (n_rows <= 0) {
        ଫେରନ୍ତୁ;
    }

    ଅଚଳ size_t n_tasks = (n_rows + 31) >> 5;
    ଯଦି (act_elem_size == ଆକାର(ଦଶମିକ) && n_threads > 1 && k_block > 32 && n_tasks < (size_t) n_threads) {
        // Calculate step rows parameters for column-chunked dma pipelining
        uint32_t dma_step_rows = 2;
        uint32_t dma_step_rows_shift = 1;
        ଯଦି (vtcm_f32_act_bytes > 0) {
            size_t thread_scratch_bytes = hex_align_down(fastdiv(vtcm_f32_act_bytes, act_threads_div), 128);
            size_t thread_scratch_elements = thread_scratch_bytes / ଆକାର(ଦଶମିକ);
            size_t dma_step_rows_max = fastdiv(thread_scratch_elements / 2, k_div);
            ଯଦି (dma_step_rows_max >= 4) {
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
        ଫେରନ୍ତୁ;
    }

    ପ୍ରତିଜ୍ଞା(k_block % HTP_MM_HMX_TILE_N_COLS == 0 && k_stride % HTP_MM_HMX_TILE_N_COLS == 0);

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
    ଯଦି (state.vtcm_f32_act_bytes_per_thread > 0) {
        size_t thread_scratch_elements = state.vtcm_f32_act_bytes_per_thread / ଆକାର(ଦଶମିକ);
        size_t dma_step_rows_max = fastdiv(thread_scratch_elements / 2, k_div);
        ଯଦି (dma_step_rows_max >= 4) {
            dma_step_rows = 4;
            dma_step_rows_shift = 2;
        }
    }
    state.dma_step_rows      = dma_step_rows;
    state.dma_step_rows_shift = dma_step_rows_shift;

    ପୂର୍ଣ୍ଣସଂଖ୍ୟା active_threads = hex_smin(n_threads, (ପୂର୍ଣ୍ଣସଂଖ୍ୟା)state.n_tasks);
    ଯଦି (state.n_tasks == 1 || n_threads == 1) {
        transfer_activation_chunk_worker_fn(1, 0, &state);
    } ଅନ୍ୟଥା {
        worker_pool_run_func(ctx->worker_pool, transfer_activation_chunk_worker_fn, &state, active_threads);
    }
}
// --- Async HMX matmul job (for pipeline overlap) ---

ପ୍ରକାରବ୍ୟାଖ୍ୟା ସଂରଚନା {
    __fp16 *       output;
    ଅଚଳ __fp16 * activation;
    ଅଚଳ __fp16 * weight;
    ଅଚଳ __fp16 * scales;
    uint32_t       n_row_tiles;
    uint32_t       n_col_tiles;
    uint32_t       n_dot_tiles;
} hmx_matmul_job_t;

ସ୍ଥିର ଶୂନ୍ୟ hmx_matmul_worker_fn(ଶୂନ୍ୟ * data) {
    hmx_matmul_job_t * job = (hmx_matmul_job_t *) data;
    FARF(ଉପର_2, "hmx-mm-job: n_row_tiles %u n_col_tiles %u n_dot_tiles %u", job->n_row_tiles, job->n_col_tiles, job->n_dot_tiles);
    core_dot_chunk_fp16(job->output, job->activation, job->weight, job->scales, job->n_row_tiles, job->n_col_tiles, job->n_dot_tiles);
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ hmx_matmul_job_init(hmx_matmul_job_t * job,
                                       __fp16 *           output,
                                       ଅଚଳ __fp16 *     activation,
                                       ଅଚଳ __fp16 *     weight,
                                       ଅଚଳ __fp16 *     scales,
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

ସ୍ଥିର ପୂର୍ଣ୍ଣସଂଖ୍ୟା hmx_mm_2d_f32(ସଂରଚନା htp_context *ctx,
                                  dma_queue *weight_dma,
                                  ଦଶମିକ *restrict dst,
                                  dma_addr_t src2_addr,
                                  size_t src2_bytes,
                                  dma_addr_t act_dma_addr,
                                  dma_addr_t weight,
                                  ପୂର୍ଣ୍ଣସଂଖ୍ୟା m, ପୂର୍ଣ୍ଣସଂଖ୍ୟା k, ପୂର୍ଣ୍ଣସଂଖ୍ୟା n,
                                  ପୂର୍ଣ୍ଣସଂଖ୍ୟା act_stride,
                                  uint32_t act_elem_size,  // 4=F32 src1, 2=F16 src1
                                  ପୂର୍ଣ୍ଣସଂଖ୍ୟା weight_stride,
                                  ପୂର୍ଣ୍ଣସଂଖ୍ୟା weight_type,
                                  ପୂର୍ଣ୍ଣସଂଖ୍ୟା k_valid,
                                  ପୂର୍ଣ୍ଣସଂଖ୍ୟା dst_stride,
                                  uint32_t src2_stride,
                                  ପୂର୍ଣ୍ଣସଂଖ୍ୟା dst_cols,
                                  ପୂର୍ଣ୍ଣସଂଖ୍ୟା m_chunk,
                                  ପୂର୍ଣ୍ଣସଂଖ୍ୟା n_chunk,
                                  ପୂର୍ଣ୍ଣସଂଖ୍ୟା pipeline,
                                  ପୂର୍ଣ୍ଣସଂଖ୍ୟା n_threads,
                                  ପୂର୍ଣ୍ଣସଂଖ୍ୟା act_threads,
                                  ଅଚଳ ସଂରଚନା fastdiv_ମୂଲ୍ୟଗୁଡ଼ିକ * act_threads_div,
                                  ଅଚଳ ସଂରଚନା fastdiv_ମୂଲ୍ୟଗୁଡ଼ିକ * k_div,
                                  ପୂର୍ଣ୍ଣସଂଖ୍ୟା tile_size,
                                  ପୂର୍ଣ୍ଣସଂଖ୍ୟା aligned_tile_size,
                                  ପୂର୍ଣ୍ଣସଂଖ୍ୟା vtcm_size) {
    ସଂରଚନା htp_thread_trace * tr = &ctx->trace[0];
    htp_trace_event_start(tr, HTP_TRACE_EVT_INIT, 0);

    ଅଚଳ bool is_tiled = (tile_size > 0);

    ଯଦି (k % 32 != 0) {
        ଫେରନ୍ତୁ -1;
    }
    ଯଦି (!hex_is_aligned(dst, ଆକାର(ଦଶମିକ))) { ଫେରନ୍ତୁ -1; }

    size_t row_stride = htp_tensor_get_row_size(weight_type, k);
    ଯଦି (row_stride == 0) {
        ଫେରନ୍ତୁ -1;
    }

    tiled_dequant_task_fn_t dequant_worker_fn = hmx_get_dequant_worker_fn(weight_type, is_tiled);

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା n_k_tiles = k / HTP_MM_HMX_TILE_N_COLS;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା n_weight_tiles = k / htp_tensor_get_tile_k(weight_type);
    ଅଚଳ ସଂରଚନା fastdiv_ମୂଲ୍ୟଗୁଡ଼ିକ n_k_tiles_div = init_fastdiv_ମୂଲ୍ୟଗୁଡ଼ିକ(n_k_tiles);

    ଅଚଳ size_t vec_dot_size = k * ଆକାର(__fp16);
    ଅଚଳ size_t vtcm_budget  = ctx->vtcm_size;

    ଅଚଳ uint32_t dma_dst_stride  = is_tiled ? aligned_tile_size : row_stride;
    ଅଚଳ uint32_t dma_src_stride  = is_tiled ? tile_size : weight_stride;
    ଅଚଳ uint32_t dma_width_bytes = is_tiled ? tile_size : row_stride;

    size_t m_chunk_n_rows = m_chunk;
    size_t n_chunk_n_cols = n_chunk;
    size_t vtcm_used      = vtcm_size;

    ସଂରଚନା htp_mm_hmx_vtcm_layout L;
    htp_mm_hmx_vtcm_layout_build(&L, HTP_MM_KERNEL_HMX_2D, weight_type, k, m_chunk_n_rows, n_chunk_n_cols, 1, pipeline, act_threads, aligned_tile_size, src2_bytes);

    vtcm_used = L.total_bytes;
    ଯଦି (vtcm_used > vtcm_budget) {
        FARF(ERROR, "hmx-mm-2d-precomputed: VTCM overflow: used %zu budget %zu, m %d k %d n %d mc %zu nc %zu",
             vtcm_used, vtcm_budget, m, k, n, m_chunk_n_rows, n_chunk_n_cols);
        ଫେରନ୍ତୁ -1;
    }

    uint8_t * ଅଚଳ base = (uint8_t *) ctx->vtcm_base;
    __fp16  *vtcm_weight_raw[2] = {
        VTCM_LAYOUT_PTR(__fp16, base, L.off_weight[0]),
        VTCM_LAYOUT_PTR_OPTIONAL(__fp16, base, L.off_weight[1], pipeline)
    };
    __fp16  *vtcm_f16_act    = VTCM_LAYOUT_PTR(__fp16, base, L.off_act);
    ଦଶମିକ   *vtcm_f32_act    = VTCM_LAYOUT_PTR(ଦଶମିକ, base, L.off_act_f32);
    __fp16  *vtcm_output     = VTCM_LAYOUT_PTR(__fp16, base, L.off_dst[0]);
    ଶୂନ୍ୟ    *vtcm_scratch0   = VTCM_LAYOUT_PTR(ଶୂନ୍ୟ, base, L.off_scratch[0]);
    ଶୂନ୍ୟ    *vtcm_scratch1   = VTCM_LAYOUT_PTR_OPTIONAL(ଶୂନ୍ୟ, base, L.off_scratch[1], pipeline);
    ଶୂନ୍ୟ    *vtcm_scratch2   = VTCM_LAYOUT_PTR_OPTIONAL(ଶୂନ୍ୟ, base, L.off_dst[1], pipeline);
    __fp16  *vtcm_scales     = VTCM_LAYOUT_PTR(__fp16, base, L.off_scales);

    hmx_init_column_scales(vtcm_scales, Q6_V_vsplat_R(0x3c00));  // scale: 1.0, bias: 0.0 in FP16

    ଅଚଳ bool has_bias      = (src2_bytes > 0 && src2_addr != 0);
    ଦଶମିକ   *vtcm_bias       = VTCM_LAYOUT_PTR_OPTIONAL(ଦଶମିକ, base, L.off_bias, has_bias);
    ଯଦି (has_bias) {
        dma_queue_ଠେଲ(weight_dma, dma_make_data(vtcm_bias, src2_addr), hex_align_up(src2_bytes, 128), 0, src2_bytes, 1);
        dma_queue_କାଢ଼(weight_dma);
    }

    FARF(ଉପର_2, "hmx-mm-2d: m %d k %d n %d wtype %d mc %zu nc %zu vtcm %zu/%zu",
         m, k, n, weight_type, m_chunk_n_rows, n_chunk_n_cols, vtcm_used, vtcm_budget);

    ପୂର୍ଣ୍ଣସଂଖ୍ୟା n_chunk_cnt = hmx_ceil_div(n, n_chunk_n_cols);

    htp_trace_event_stop(tr, HTP_TRACE_EVT_INIT, 0);

    ଯଦି (pipeline) {
        // --- Asynchronous Pipelined Loop ---
        hmx_matmul_job_t job_slots[2];  // persistent double-buffered job descriptors
        ଶୂନ୍ୟ *vtcm_raw_bufs[4] = {
            vtcm_weight_raw[0],
            vtcm_weight_raw[1],
            dequant_worker_fn ? vtcm_weight_raw[0] : vtcm_scratch0,
            dequant_worker_fn ? vtcm_weight_raw[1] : vtcm_scratch1,
        };
        ଶୂନ୍ୟ *vtcm_weight_bufs[4] = {
            dequant_worker_fn ? vtcm_scratch0 : vtcm_weight_raw[0],
            dequant_worker_fn ? vtcm_scratch1 : vtcm_weight_raw[1],
            vtcm_scratch0,
            vtcm_scratch1,
        };
        ଶୂନ୍ୟ *vtcm_output_bufs[2] = { vtcm_output, vtcm_scratch2 };

        ପାଇଁ (size_t mr = 0; mr < m; mr += m_chunk_n_rows) {
            ଅଚଳ size_t n_rows = hex_smin(m - mr, m_chunk_n_rows);

            ସଂରଚନା activation_transfer_params act_params = {
                .ctx = ctx,
                .dst = vtcm_f16_act,
                .act_dma_addr = act_dma_addr + (size_t) mr * act_stride * act_elem_size,  // byte offset (F16/F32)
                .n_rows = (ପୂର୍ଣ୍ଣସଂଖ୍ୟା) n_rows,
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
            ଅଚଳ size_t   n_cols_A0 = hex_smin(n - 0 * n_chunk_n_cols, n_chunk_n_cols);
            ଅଚଳ uint32_t height_A0 = is_tiled ? hmx_ceil_div(n_cols_A0, 32) * n_weight_tiles : n_cols_A0;
            dma_queue_ଠେଲ(weight_dma, dma_make_data(vtcm_raw_bufs[0], weight),
                           dma_dst_stride, dma_src_stride, dma_width_bytes, height_A0);

            ଯଦି (1 < n_chunk_cnt) {
                ଅଚଳ size_t   n_cols_A1 = hex_smin(n - 1 * n_chunk_n_cols, n_chunk_n_cols);
                ଅଚଳ uint32_t height_A1 = is_tiled ? hmx_ceil_div(n_cols_A1, 32) * n_weight_tiles : n_cols_A1;
                dma_queue_ଠେଲ(weight_dma, dma_make_data(vtcm_raw_bufs[1], weight + n_chunk_n_cols * weight_stride),
                               dma_dst_stride, dma_src_stride, dma_width_bytes, height_A1);
            }

            // Main loop: pop A_i -> dequantize A_i -> push A_{i+2} -> submit C_i -> wait C_{i-1} and store D_{i-1}
            ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < n_chunk_cnt; ++i) {
                ଅଚଳ size_t nc    = i * n_chunk_n_cols;
                ଅଚଳ size_t nc_p2 = nc + 2 * n_chunk_n_cols;

                ଅଚଳ size_t n_cols    = hex_smin(n - nc, n_chunk_n_cols);
                ଅଚଳ size_t n_cols_p2 = hex_smin(n - nc_p2, n_chunk_n_cols);

                // 1. pop A_i
                ଶୂନ୍ୟ * curr_raw = (ଶୂନ୍ୟ *) dma_queue_କାଢ଼(weight_dma).dst;

                // 2. dequantize A_i
                dequantize_tiled_weight_chunk_to_fp16_tiles(
                    ctx, vtcm_weight_bufs[i % 4], curr_raw,
                    n_cols, k, row_stride, weight_type,
                    n_k_tiles, n_k_tiles_div, dequant_worker_fn, n_threads);

                // 3. push A_{i+2} (if i+2 < n_chunk_cnt)
                ଯଦି (i + 2 < n_chunk_cnt) {
                    ଅଚଳ uint32_t height_p2 = is_tiled ? hmx_ceil_div(n_cols_p2, 32) * n_weight_tiles : n_cols_p2;
                    dma_queue_ଠେଲ(weight_dma, dma_make_data(vtcm_raw_bufs[(i + 2) % 4], weight + nc_p2 * weight_stride),
                                   dma_dst_stride, dma_src_stride, dma_width_bytes, height_p2);
                }

                // 4. submit C_i
                hmx_matmul_job_init(&job_slots[i % 2], (__fp16 *) vtcm_output_bufs[i % 2],
                                    (__fp16 *) vtcm_f16_act, (__fp16 *) vtcm_weight_bufs[i % 4],
                                    vtcm_scales, hmx_ceil_div(n_rows, HTP_MM_HMX_TILE_N_ROWS),
                                    hmx_ceil_div(n_cols, HTP_MM_HMX_TILE_N_COLS), k / HTP_MM_HMX_TILE_N_ROWS);
                hmx_queue_ଠେଲ(ctx->hmx_queue, hmx_queue_make_desc(hmx_matmul_worker_fn, &job_slots[i % 2]));

                // 5. wait C_{i-1} and store D_{i-1} (multi-thread HVX, parallel with C_i)
                ଯଦି (i > 0) {
                    hmx_queue_କାଢ଼(ctx->hmx_queue);
                    ଅଚଳ size_t nc_prev = (i - 1) * n_chunk_n_cols;
                    ଅଚଳ size_t n_cols_prev = hex_smin(n - nc_prev, n_chunk_n_cols);
                    ଦଶମିକ *output_chunk = dst + (mr * dst_stride + nc_prev);
                    ଅଚଳ ଦଶମିକ *bias_chunk = has_bias ? (vtcm_bias + mr * src2_stride + nc_prev) : ଖାଲି;
                    ପୂର୍ଣ୍ଣସଂଖ୍ୟା chunk_dst_cols = dst_cols - (ପୂର୍ଣ୍ଣସଂଖ୍ୟା)nc_prev;
                    ଯଦି (chunk_dst_cols > 0) {
                        transfer_output_chunk_threaded(ctx, output_chunk, bias_chunk, vtcm_output_bufs[(i - 1) % 2], n_rows, n_cols_prev, dst_stride, src2_stride, chunk_dst_cols, n_threads);
                    }
                }
            }

            // Epilogue: wait C_{last} and store D_{last}
            hmx_queue_କାଢ଼(ctx->hmx_queue);
            ଅଚଳ size_t nc_last = (n_chunk_cnt - 1) * n_chunk_n_cols;
            ଅଚଳ size_t n_cols_last = hex_smin(n - nc_last, n_chunk_n_cols);
            ଦଶମିକ *output_chunk = dst + (mr * dst_stride + nc_last);
            ଅଚଳ ଦଶମିକ *bias_chunk = has_bias ? (vtcm_bias + mr * src2_stride + nc_last) : ଖାଲି;
            ପୂର୍ଣ୍ଣସଂଖ୍ୟା chunk_dst_cols = dst_cols - (ପୂର୍ଣ୍ଣସଂଖ୍ୟା)nc_last;
            ଯଦି (chunk_dst_cols > 0) {
                transfer_output_chunk_threaded(ctx, output_chunk, bias_chunk, vtcm_output_bufs[(n_chunk_cnt - 1) % 2], n_rows, n_cols_last, dst_stride, src2_stride, chunk_dst_cols, n_threads);
            }
        }
    } ଅନ୍ୟଥା {
        // --- Synchronous loop (m <= 32 or fallback) ---
        hmx_matmul_job_t job;
        ଶୂନ୍ୟ *vtcm_raw_bufs[2] = { vtcm_weight_raw[0], dequant_worker_fn ? vtcm_weight_raw[0] : vtcm_scratch0 };
        ଶୂନ୍ୟ *vtcm_weight_bufs[2] = { dequant_worker_fn ? vtcm_scratch0 : vtcm_weight_raw[0], vtcm_scratch0 };

        ପାଇଁ (size_t mr = 0; mr < m; mr += m_chunk_n_rows) {
            ଅଚଳ size_t n_rows = hex_smin(m - mr, m_chunk_n_rows);

            ସଂରଚନା activation_transfer_params act_params = {
                .ctx = ctx,
                .dst = vtcm_f16_act,
                .act_dma_addr = act_dma_addr + (size_t) mr * act_stride * act_elem_size,  // byte offset (F16/F32)
                .n_rows = (ପୂର୍ଣ୍ଣସଂଖ୍ୟା) n_rows,
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
            ଯଦି (n > 0) {
                ଅଚଳ size_t n_cols = hex_smin(n, n_chunk_n_cols);
                ଅଚଳ uint32_t height = is_tiled ? hmx_ceil_div(n_cols, 32) * n_weight_tiles : n_cols;
                dma_queue_ଠେଲ(weight_dma, dma_make_data(vtcm_raw_bufs[0], weight), dma_dst_stride, dma_src_stride, dma_width_bytes, height);
            }

            ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < n_chunk_cnt; ++i) {
                ଅଚଳ size_t nc = (size_t) i * n_chunk_n_cols;
                ଅଚଳ size_t n_cols = hex_smin(n - nc, n_chunk_n_cols);
                ଅଚଳ size_t n_row_tiles = hmx_ceil_div(n_rows, HTP_MM_HMX_TILE_N_ROWS);
                ଅଚଳ size_t n_col_tiles = hmx_ceil_div(n_cols, HTP_MM_HMX_TILE_N_COLS);

                // A: Wait for weight DMA
                ଶୂନ୍ୟ * curr_raw = (ଶୂନ୍ୟ *) dma_queue_କାଢ଼(weight_dma).dst;

                // B: Weight Dequantize (Threaded)
                dequantize_tiled_weight_chunk_to_fp16_tiles(
                    ctx, vtcm_weight_bufs[i % 2], curr_raw,
                    n_cols, k, row_stride, weight_type,
                    n_k_tiles, n_k_tiles_div, dequant_worker_fn, n_threads);

                // Start weight DMA for the next chunk early
                ଅଚଳ size_t nc_next = nc + n_chunk_n_cols;
                ଯଦି (nc_next < n) {
                    ଅଚଳ size_t n_cols_next = hex_smin(n - nc_next, n_chunk_n_cols);
                    ଅଚଳ uint32_t height_next = is_tiled ? hmx_ceil_div(n_cols_next, 32) * n_weight_tiles : n_cols_next;
                    dma_queue_ଠେଲ(weight_dma, dma_make_data(vtcm_raw_bufs[(i + 1) % 2], weight + nc_next * weight_stride), dma_dst_stride, dma_src_stride, dma_width_bytes, height_next);
                }

                // C: HMX Compute (Queue-based)
                hmx_matmul_job_init(&job, vtcm_output, vtcm_f16_act, (__fp16 *) vtcm_weight_bufs[i % 2], vtcm_scales, n_row_tiles, n_col_tiles, k / HTP_MM_HMX_TILE_N_ROWS);
                hmx_queue_ଠେଲ(ctx->hmx_queue, hmx_queue_make_desc(hmx_matmul_worker_fn, &job));
                hmx_queue_କାଢ଼(ctx->hmx_queue);

                // D: Output Store
                ଦଶମିକ *output_chunk = dst + (mr * dst_stride + nc);
                ଅଚଳ ଦଶମିକ *bias_chunk = has_bias ? (vtcm_bias + mr * src2_stride + nc) : ଖାଲି;
                ପୂର୍ଣ୍ଣସଂଖ୍ୟା chunk_dst_cols = dst_cols - (ପୂର୍ଣ୍ଣସଂଖ୍ୟା)nc;
                ଯଦି (chunk_dst_cols > 0) {
                    transfer_output_chunk_threaded(ctx, output_chunk, bias_chunk, vtcm_output, n_rows, n_cols, dst_stride, src2_stride, chunk_dst_cols, n_threads);
                }
            }
        }
    }

    ଫେରନ୍ତୁ 0;
}

ସ୍ଥିର ପୂର୍ଣ୍ଣସଂଖ୍ୟା hmx_mm_nx_2d_f32(ସଂରଚନା htp_ops_context * octx, ଅଚଳ ସଂରଚନା htp_mm_kernel_params * kparams) {
    ସଂରଚନା htp_context * ctx = octx->ctx;
    ସଂରଚନା htp_thread_trace * tr = &ctx->trace[0];
    htp_trace_event_start(tr, HTP_TRACE_EVT_INIT, 0);

    ଅଚଳ uint32_t n_weights = kparams->n_weights;
    ଯଦି (n_weights == 0 || n_weights > HTP_OP_MAX_OUTPUTS) {
        ଫେରନ୍ତୁ HTP_STATUS_INVAL_PARAMS;
    }

    ଅଚଳ ସଂରଚନା htp_tensor * restrict src0 = octx->src[0];
    ଅଚଳ ସଂରଚନା htp_tensor * restrict act  = octx->src[n_weights];

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା weight_type = (ପୂର୍ଣ୍ଣସଂଖ୍ୟା) src0->type;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା k           = (ପୂର୍ଣ୍ଣସଂଖ୍ୟା) act->ne[0];
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା k_valid     = (ପୂର୍ଣ୍ଣସଂଖ୍ୟା) act->ne[0];
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା m           = (ପୂର୍ଣ୍ଣସଂଖ୍ୟା) (act->ne[1] * act->ne[2] * act->ne[3]);
    ଅଚଳ uint32_t act_elem_size = (act->type == HTP_TYPE_F16) ? ଆକାର(__fp16) : ଆକାର(ଦଶମିକ);
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା act_stride  = (ପୂର୍ଣ୍ଣସଂଖ୍ୟା) (act->nb[1] / act_elem_size);
    ଅଚଳ dma_addr_t act_dma_addr = act->data;

    ଯଦି (k % 32 != 0) { ଫେରନ୍ତୁ HTP_STATUS_NO_SUPPORT; }

    size_t row_stride = htp_tensor_get_row_size(weight_type, k);
    ଯଦି (row_stride == 0) {
        ଫେରନ୍ତୁ HTP_STATUS_NO_SUPPORT;
    }

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା n_k_tiles = k / HTP_MM_HMX_TILE_N_COLS;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା n_weight_tiles = k / htp_tensor_get_tile_k(weight_type);
    ଅଚଳ ସଂରଚନା fastdiv_ମୂଲ୍ୟଗୁଡ଼ିକ n_k_tiles_div = init_fastdiv_ମୂଲ୍ୟଗୁଡ଼ିକ(n_k_tiles);

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା tile_size       = kparams->tile_size;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା aligned_tile_size = kparams->aligned_tile_size;

    ଅଚଳ bool is_tiled       = (tile_size > 0);
    tiled_dequant_task_fn_t dequant_worker_fn = hmx_get_dequant_worker_fn(weight_type, is_tiled);
    ଅଚଳ size_t vtcm_budget  = ctx->vtcm_size;

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା m_chunk_n_rows  = kparams->m_chunk;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା n_chunk_n_cols  = kparams->n_chunk;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା pipeline        = kparams->pipeline;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା n_threads       = octx->n_threads;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା act_threads     = kparams->n_act_threads;
    ଅଚଳ ସଂରଚନା fastdiv_ମୂଲ୍ୟଗୁଡ଼ିକ * act_threads_div = &kparams->div_n_act_threads;
    ଅଚଳ ସଂରଚନା fastdiv_ମୂଲ୍ୟଗୁଡ଼ିକ * k_div           = &kparams->div_ne00_padded;

    ଅଚଳ uint32_t dma_dst_stride  = is_tiled ? aligned_tile_size : row_stride;
    ଅଚଳ uint32_t dma_width_bytes = is_tiled ? tile_size : row_stride;

    ସଂରଚନା htp_mm_hmx_vtcm_layout L;
    htp_mm_hmx_vtcm_layout_build(&L, HTP_MM_KERNEL_HMX_2D, weight_type, k, m_chunk_n_rows, n_chunk_n_cols, 1, pipeline, act_threads, aligned_tile_size, 0);

    ଯଦି (L.total_bytes > vtcm_budget) {
        FARF(ERROR, "hmx-mm-nx-2d: VTCM overflow: used %zu budget %zu, m %d k %d mc %d nc %d",
             L.total_bytes, vtcm_budget, m, k, m_chunk_n_rows, n_chunk_n_cols);
        ଫେରନ୍ତୁ HTP_STATUS_VTCM_TOO_SMALL;
    }

    uint8_t * ଅଚଳ base = (uint8_t *) ctx->vtcm_base;
    __fp16  *vtcm_weight_raw[2] = {
        VTCM_LAYOUT_PTR(__fp16, base, L.off_weight[0]),
        VTCM_LAYOUT_PTR_OPTIONAL(__fp16, base, L.off_weight[1], pipeline)
    };
    __fp16  *vtcm_f16_act    = VTCM_LAYOUT_PTR(__fp16, base, L.off_act);
    ଦଶମିକ   *vtcm_f32_act    = VTCM_LAYOUT_PTR(ଦଶମିକ, base, L.off_act_f32);
    __fp16  *vtcm_output     = VTCM_LAYOUT_PTR(__fp16, base, L.off_dst[0]);
    ଶୂନ୍ୟ    *vtcm_scratch0   = VTCM_LAYOUT_PTR(ଶୂନ୍ୟ, base, L.off_scratch[0]);
    ଶୂନ୍ୟ    *vtcm_scratch1   = VTCM_LAYOUT_PTR_OPTIONAL(ଶୂନ୍ୟ, base, L.off_scratch[1], pipeline);
    ଶୂନ୍ୟ    *vtcm_scratch2   = VTCM_LAYOUT_PTR_OPTIONAL(ଶୂନ୍ୟ, base, L.off_dst[1], pipeline);
    __fp16  *vtcm_scales     = VTCM_LAYOUT_PTR(__fp16, base, L.off_scales);

    hmx_init_column_scales(vtcm_scales, Q6_V_vsplat_R(0x3c00));  // scale: 1.0, bias: 0.0 in FP16

    ପୂର୍ଣ୍ଣସଂଖ୍ୟା m_start = 0;
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା m_rows  = m;
    ଯଦି (octx->ctx->mdev.ଗଣନା_2 > 1) {
        bool can_split = htp_tensor_can_row_partition(act, act_elem_size);
        ଯଦି (can_split) {
            ପାଇଁ (uint32_t p = 0; p < n_weights; ++p) {
                ଅଚଳ ସଂରଚନା htp_tensor * restrict dst = octx->dsts[p];
                ଯଦି (!htp_tensor_can_row_partition(dst, ଆକାର(ଦଶମିକ))) {
                    can_split = false;
                    ଭାଙ୍ଗ;
                }
            }
        }
        ଅଚଳ ସଂରଚନା htp_tensor_mdev_range range = htp_tensor_mdev_partition((uint32_t) m, can_split ? 1 : 0, octx->ctx->mdev.idx, octx->ctx->mdev.ଗଣନା_2, &octx->ctx->mdev.ଗଣନା_div);
        m_start = (ପୂର୍ଣ୍ଣସଂଖ୍ୟା) range.start;
        m_rows  = (ପୂର୍ଣ୍ଣସଂଖ୍ୟା) range.ଗଣନା_2;
    }

    ଯଦି (m_rows == 0) {
        ଫେରନ୍ତୁ HTP_STATUS_OK;
    }

    FARF(ଉପର_2, "hmx-mm-nx-2d: n_weights %u m %d (%d..%d) k %d wtype %d mc %d nc %d vtcm %zu/%zu",
         n_weights, m, m_start, m_start + m_rows, k, weight_type, m_chunk_n_rows, n_chunk_n_cols, L.total_bytes, vtcm_budget);

    htp_trace_event_stop(tr, HTP_TRACE_EVT_INIT, 0);

    ଅଚଳ size_t mr_end = (size_t)(m_start + m_rows);

    ଯଦି (pipeline) {
        hmx_matmul_job_t job_slots[2];
        ଶୂନ୍ୟ *vtcm_raw_bufs[4] = {
            vtcm_weight_raw[0],
            vtcm_weight_raw[1],
            dequant_worker_fn ? vtcm_weight_raw[0] : vtcm_scratch0,
            dequant_worker_fn ? vtcm_weight_raw[1] : vtcm_scratch1,
        };
        ଶୂନ୍ୟ *vtcm_weight_bufs[4] = {
            dequant_worker_fn ? vtcm_scratch0 : vtcm_weight_raw[0],
            dequant_worker_fn ? vtcm_scratch1 : vtcm_weight_raw[1],
            vtcm_scratch0,
            vtcm_scratch1,
        };
        ଶୂନ୍ୟ *vtcm_output_bufs[2] = { vtcm_output, vtcm_scratch2 };

        ପାଇଁ (size_t mr = (size_t) m_start; mr < mr_end; mr += m_chunk_n_rows) {
            ଅଚଳ size_t n_rows = hex_smin(mr_end - mr, m_chunk_n_rows);

            ସଂରଚନା activation_transfer_params act_params = {
                .ctx = ctx,
                .dst = vtcm_f16_act,
                .act_dma_addr = act_dma_addr + (size_t) mr * act_stride * act_elem_size,
                .n_rows = (ପୂର୍ଣ୍ଣସଂଖ୍ୟା) n_rows,
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

            ପାଇଁ (uint32_t p = 0; p < n_weights; p++) {
                ଅଚଳ ସଂରଚନା htp_tensor * restrict src_w = octx->src[p];
                ଅଚଳ ସଂରଚନା htp_tensor * restrict dst   = octx->dsts[p];
                ଯଦି (!src_w || !dst) ଜାରି;

                ଅଚଳ dma_addr_t weight       = src_w->data;
                dma_queue * weight_dma      = octx->ctx->dma[0];
                ଦଶମିକ * dst_ptr            = (ଦଶମିକ *) dst->data;
                ଅଚଳ size_t n             = src_w->ne[1];
                ଯଦି (n == 0) ଜାରି;
                ଅଚଳ size_t weight_stride = src_w->nb[1];
                ଅଚଳ size_t dst_stride    = dst->nb[1] / ଆକାର(ଦଶମିକ);
                ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା dst_cols         = (ପୂର୍ଣ୍ଣସଂଖ୍ୟା) dst->ne[0];
                ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା n_chunk_cnt      = hmx_ceil_div(n, n_chunk_n_cols);

                ଅଚଳ uint32_t dma_src_stride = is_tiled ? tile_size : weight_stride;

                ଅଚଳ size_t   n_cols_A0 = hex_smin(n - 0 * n_chunk_n_cols, n_chunk_n_cols);
                ଅଚଳ uint32_t height_A0 = is_tiled ? hmx_ceil_div(n_cols_A0, 32) * n_weight_tiles : n_cols_A0;
                dma_queue_ଠେଲ(weight_dma, dma_make_data(vtcm_raw_bufs[0], weight),
                               dma_dst_stride, dma_src_stride, dma_width_bytes, height_A0);

                ଯଦି (1 < n_chunk_cnt) {
                    ଅଚଳ size_t   n_cols_A1 = hex_smin(n - 1 * n_chunk_n_cols, n_chunk_n_cols);
                    ଅଚଳ uint32_t height_A1 = is_tiled ? hmx_ceil_div(n_cols_A1, 32) * n_weight_tiles : n_cols_A1;
                    dma_queue_ଠେଲ(weight_dma, dma_make_data(vtcm_raw_bufs[1], weight + n_chunk_n_cols * weight_stride),
                                   dma_dst_stride, dma_src_stride, dma_width_bytes, height_A1);
                }

                ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < n_chunk_cnt; ++i) {
                    ଅଚଳ size_t nc    = i * n_chunk_n_cols;
                    ଅଚଳ size_t nc_p2 = nc + 2 * n_chunk_n_cols;

                    ଅଚଳ size_t n_cols    = hex_smin(n - nc, n_chunk_n_cols);
                    ଅଚଳ size_t n_cols_p2 = hex_smin(n - nc_p2, n_chunk_n_cols);

                    ଶୂନ୍ୟ * curr_raw = (ଶୂନ୍ୟ *) dma_queue_କାଢ଼(weight_dma).dst;

                    dequantize_tiled_weight_chunk_to_fp16_tiles(
                        ctx, vtcm_weight_bufs[i % 4], curr_raw,
                        n_cols, k, row_stride, weight_type,
                        n_k_tiles, n_k_tiles_div, dequant_worker_fn, n_threads);

                    ଯଦି (i + 2 < n_chunk_cnt) {
                        ଅଚଳ uint32_t height_p2 = is_tiled ? hmx_ceil_div(n_cols_p2, 32) * n_weight_tiles : n_cols_p2;
                        dma_queue_ଠେଲ(weight_dma, dma_make_data(vtcm_raw_bufs[(i + 2) % 4], weight + nc_p2 * weight_stride),
                                       dma_dst_stride, dma_src_stride, dma_width_bytes, height_p2);
                    }

                    hmx_matmul_job_init(&job_slots[i % 2], (__fp16 *) vtcm_output_bufs[i % 2],
                                        (__fp16 *) vtcm_f16_act, (__fp16 *) vtcm_weight_bufs[i % 4],
                                        vtcm_scales, hmx_ceil_div(n_rows, HTP_MM_HMX_TILE_N_ROWS),
                                        hmx_ceil_div(n_cols, HTP_MM_HMX_TILE_N_COLS), k / HTP_MM_HMX_TILE_N_ROWS);
                    hmx_queue_ଠେଲ(ctx->hmx_queue, hmx_queue_make_desc(hmx_matmul_worker_fn, &job_slots[i % 2]));

                    ଯଦି (i > 0) {
                        hmx_queue_କାଢ଼(ctx->hmx_queue);
                        ଅଚଳ size_t nc_prev = (i - 1) * n_chunk_n_cols;
                        ଅଚଳ size_t n_cols_prev = hex_smin(n - nc_prev, n_chunk_n_cols);
                        ଦଶମିକ *output_chunk = dst_ptr + (mr * dst_stride + nc_prev);
                        ପୂର୍ଣ୍ଣସଂଖ୍ୟା chunk_dst_cols = dst_cols - (ପୂର୍ଣ୍ଣସଂଖ୍ୟା)nc_prev;
                        ଯଦି (chunk_dst_cols > 0) {
                            transfer_output_chunk_threaded(ctx, output_chunk, ଖାଲି, vtcm_output_bufs[(i - 1) % 2], n_rows, n_cols_prev, dst_stride, 0, chunk_dst_cols, n_threads);
                        }
                    }
                }

                hmx_queue_କାଢ଼(ctx->hmx_queue);
                ଅଚଳ size_t nc_last = (n_chunk_cnt - 1) * n_chunk_n_cols;
                ଅଚଳ size_t n_cols_last = hex_smin(n - nc_last, n_chunk_n_cols);
                ଦଶମିକ *output_chunk = dst_ptr + (mr * dst_stride + nc_last);
                ପୂର୍ଣ୍ଣସଂଖ୍ୟା chunk_dst_cols = dst_cols - (ପୂର୍ଣ୍ଣସଂଖ୍ୟା)nc_last;
                ଯଦି (chunk_dst_cols > 0) {
                    transfer_output_chunk_threaded(ctx, output_chunk, ଖାଲି, vtcm_output_bufs[(n_chunk_cnt - 1) % 2], n_rows, n_cols_last, dst_stride, 0, chunk_dst_cols, n_threads);
                }
            }
        }
    } ଅନ୍ୟଥା {
        hmx_matmul_job_t job;
        ଶୂନ୍ୟ *vtcm_raw_bufs[2] = { vtcm_weight_raw[0], dequant_worker_fn ? vtcm_weight_raw[0] : vtcm_scratch0 };
        ଶୂନ୍ୟ *vtcm_weight_bufs[2] = { dequant_worker_fn ? vtcm_scratch0 : vtcm_weight_raw[0], vtcm_scratch0 };

        ପାଇଁ (size_t mr = (size_t) m_start; mr < mr_end; mr += m_chunk_n_rows) {
            ଅଚଳ size_t n_rows = hex_smin(mr_end - mr, m_chunk_n_rows);

            ସଂରଚନା activation_transfer_params act_params = {
                .ctx = ctx,
                .dst = vtcm_f16_act,
                .act_dma_addr = act_dma_addr + (size_t) mr * act_stride * act_elem_size,
                .n_rows = (ପୂର୍ଣ୍ଣସଂଖ୍ୟା) n_rows,
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

            ପାଇଁ (uint32_t p = 0; p < n_weights; p++) {
                ଅଚଳ ସଂରଚନା htp_tensor * restrict src_w = octx->src[p];
                ଅଚଳ ସଂରଚନା htp_tensor * restrict dst   = octx->dsts[p];
                ଯଦି (!src_w || !dst) ଜାରି;

                ଅଚଳ dma_addr_t weight       = src_w->data;
                dma_queue * weight_dma      = octx->ctx->dma[0];
                ଦଶମିକ * dst_ptr            = (ଦଶମିକ *) dst->data;
                ଅଚଳ size_t n             = src_w->ne[1];
                ଯଦି (n == 0) ଜାରି;
                ଅଚଳ size_t weight_stride = src_w->nb[1];
                ଅଚଳ size_t dst_stride    = dst->nb[1] / ଆକାର(ଦଶମିକ);
                ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା dst_cols         = (ପୂର୍ଣ୍ଣସଂଖ୍ୟା) dst->ne[0];
                ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା n_chunk_cnt      = hmx_ceil_div(n, n_chunk_n_cols);

                ଅଚଳ uint32_t dma_src_stride = is_tiled ? tile_size : weight_stride;

                ଯଦି (n > 0) {
                    ଅଚଳ size_t n_cols = hex_smin(n, n_chunk_n_cols);
                    ଅଚଳ uint32_t height = is_tiled ? hmx_ceil_div(n_cols, 32) * n_weight_tiles : n_cols;
                    dma_queue_ଠେଲ(weight_dma, dma_make_data(vtcm_raw_bufs[0], weight), dma_dst_stride, dma_src_stride, dma_width_bytes, height);
                }

                ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < n_chunk_cnt; ++i) {
                    ଅଚଳ size_t nc = (size_t) i * n_chunk_n_cols;
                    ଅଚଳ size_t n_cols = hex_smin(n - nc, n_chunk_n_cols);
                    ଅଚଳ size_t n_row_tiles = hmx_ceil_div(n_rows, HTP_MM_HMX_TILE_N_ROWS);
                    ଅଚଳ size_t n_col_tiles = hmx_ceil_div(n_cols, HTP_MM_HMX_TILE_N_COLS);

                    ଶୂନ୍ୟ * curr_raw = (ଶୂନ୍ୟ *) dma_queue_କାଢ଼(weight_dma).dst;

                    dequantize_tiled_weight_chunk_to_fp16_tiles(
                        ctx, vtcm_weight_bufs[i % 2], curr_raw,
                        n_cols, k, row_stride, weight_type,
                        n_k_tiles, n_k_tiles_div, dequant_worker_fn, n_threads);

                    ଅଚଳ size_t nc_next = nc + n_chunk_n_cols;
                    ଯଦି (nc_next < n) {
                        ଅଚଳ size_t n_cols_next = hex_smin(n - nc_next, n_chunk_n_cols);
                        ଅଚଳ uint32_t height_next = is_tiled ? hmx_ceil_div(n_cols_next, 32) * n_weight_tiles : n_cols_next;
                        dma_queue_ଠେଲ(weight_dma, dma_make_data(vtcm_raw_bufs[(i + 1) % 2], weight + nc_next * weight_stride), dma_dst_stride, dma_src_stride, dma_width_bytes, height_next);
                    }

                    hmx_matmul_job_init(&job, vtcm_output, vtcm_f16_act, (__fp16 *) vtcm_weight_bufs[i % 2], vtcm_scales, n_row_tiles, n_col_tiles, k / HTP_MM_HMX_TILE_N_ROWS);
                    hmx_queue_ଠେଲ(ctx->hmx_queue, hmx_queue_make_desc(hmx_matmul_worker_fn, &job));
                    hmx_queue_କାଢ଼(ctx->hmx_queue);

                    ଦଶମିକ *output_chunk = dst_ptr + (mr * dst_stride + nc);
                    ପୂର୍ଣ୍ଣସଂଖ୍ୟା chunk_dst_cols = dst_cols - (ପୂର୍ଣ୍ଣସଂଖ୍ୟା)nc;
                    ଯଦି (chunk_dst_cols > 0) {
                        transfer_output_chunk_threaded(ctx, output_chunk, ଖାଲି, vtcm_output, n_rows, n_cols, dst_stride, 0, chunk_dst_cols, n_threads);
                    }
                }
            }
        }
    }

    ଫେରନ୍ତୁ HTP_STATUS_OK;
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ dma_addr_t hmx_mm_weight_batch_data(ଅଚଳ hmx_mm_batched_params_t *params,
                                                         ପୂର୍ଣ୍ଣସଂଖ୍ୟା dst_b2, ପୂର୍ଣ୍ଣସଂଖ୍ୟା dst_b3) {
    ଅଚଳ size_t b2_idx = (params->r2 <= 1) ? (size_t) dst_b2 : (size_t) fastdiv((uint32_t) dst_b2, &params->div_r2);
    ଅଚଳ size_t b3_idx = (params->r3 <= 1) ? (size_t) dst_b3 : (size_t) fastdiv((uint32_t) dst_b3, &params->div_r3);
    ଫେରନ୍ତୁ params->weight + b2_idx * params->src0_nb2 + b3_idx * params->src0_nb3;
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ dma_addr_t hmx_mm_act_batch_addr(ଅଚଳ hmx_mm_batched_params_t *params,
                                                ପୂର୍ଣ୍ଣସଂଖ୍ୟା dst_b2, ପୂର୍ଣ୍ଣସଂଖ୍ୟା dst_b3) {
    ଫେରନ୍ତୁ params->act_dma_addr + dst_b2 * params->act_nb2 +
           dst_b3 * params->act_nb3;
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଦଶମିକ *hmx_mm_dst_batch_ptr(ଅଚଳ hmx_mm_batched_params_t *params,
                                              ପୂର୍ଣ୍ଣସଂଖ୍ୟା dst_b2, ପୂର୍ଣ୍ଣସଂଖ୍ୟା dst_b3) {
    ଫେରନ୍ତୁ (ଦଶମିକ *) ((uint8_t *) params->dst +
                      (size_t) dst_b2 * params->dst_nb2 +
                      (size_t) dst_b3 * params->dst_nb3);
}

ସ୍ଥିର ପୂର୍ଣ୍ଣସଂଖ୍ୟା hmx_mm_batched_simple(ସଂରଚନା htp_context *ctx,
                                 ଅଚଳ hmx_mm_batched_params_t *params,
                                 ପୂର୍ଣ୍ଣସଂଖ୍ୟା m_chunk, ପୂର୍ଣ୍ଣସଂଖ୍ୟା n_chunk, ପୂର୍ଣ୍ଣସଂଖ୍ୟା pipeline, ପୂର୍ଣ୍ଣସଂଖ୍ୟା n_threads, ପୂର୍ଣ୍ଣସଂଖ୍ୟା act_threads, ପୂର୍ଣ୍ଣସଂଖ୍ୟା vtcm_size,
                                 ଅଚଳ ସଂରଚନା fastdiv_ମୂଲ୍ୟଗୁଡ଼ିକ * act_threads_div, ଅଚଳ ସଂରଚନା fastdiv_ମୂଲ୍ୟଗୁଡ଼ିକ * k_div) {
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା ret = 0;
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା b3 = 0; b3 < params->ne13 && ret == 0; ++b3) {
        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା b2 = 0; b2 < params->ne12 && ret == 0; ++b2) {
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
    ଫେରନ୍ତୁ ret;
}

ସ୍ଥିର ପୂର୍ଣ୍ଣସଂଖ୍ୟା hmx_mm_batched(ସଂରଚନା htp_context *ctx, ଅଚଳ hmx_mm_batched_params_t *params,
                          ପୂର୍ଣ୍ଣସଂଖ୍ୟା m_chunk, ପୂର୍ଣ୍ଣସଂଖ୍ୟା n_chunk, ପୂର୍ଣ୍ଣସଂଖ୍ୟା pipeline, ପୂର୍ଣ୍ଣସଂଖ୍ୟା n_threads, ପୂର୍ଣ୍ଣସଂଖ୍ୟା act_threads,
                          ଅଚଳ ସଂରଚନା fastdiv_ମୂଲ୍ୟଗୁଡ଼ିକ * act_threads_div,
                          ଅଚଳ ସଂରଚନା fastdiv_ମୂଲ୍ୟଗୁଡ଼ିକ * k_div,
                          ପୂର୍ଣ୍ଣସଂଖ୍ୟା vtcm_size) {
    ଅଚଳ size_t row_stride = htp_tensor_get_row_size(params->weight_type, params->k);
    ଯଦି (row_stride == 0) { ଫେରନ୍ତୁ -1; }
    ଯଦି (params->act_stride < params->k || (params->tile_size == 0 && (size_t) params->weight_stride < row_stride) || params->dst_stride < params->dst_cols) { ଫେରନ୍ତୁ -1; }
    ଯଦି (params->ne02 <= 0 || params->ne03 <= 0 || params->ne12 <= 0 || params->ne13 <= 0) { ଫେରନ୍ତୁ -1; }
    ଯଦି (params->ne12 % params->ne02 != 0 || params->ne13 % params->ne03 != 0) { ଫେରନ୍ତୁ -1; }
    // N (the weight row count) does not have to be 32-aligned:
    ଯଦି (params->k % 32 != 0) { ଫେରନ୍ତୁ -1; }
    ଯଦି (!hex_is_aligned(params->dst, ଆକାର(ଦଶମିକ))) { ଫେରନ୍ତୁ -1; }

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା group_size = params->r2;
    ଅଚଳ size_t vtcm_budget  = ctx->vtcm_size;

    // Check if the precomputed parameters are grouped or simple.
    // If simple, or if group_size <= 1, we use simple fallback loop.
    // Grouped path is only valid for flat F16 if group_size > 1 and it fits within VTCM budget.
    bool run_grouped = (params->weight_type == HTP_TYPE_F16 && params->tile_size == 0 && group_size > 1 && (size_t)vtcm_size <= vtcm_budget);
    ଯଦି (!run_grouped) {
        ଫେରନ୍ତୁ hmx_mm_batched_simple(ctx, params, m_chunk, n_chunk, pipeline, n_threads, act_threads, vtcm_size, act_threads_div, k_div);
    }

    ସଂରଚନା htp_thread_trace * tr = &ctx->trace[0];
    htp_trace_event_start(tr, HTP_TRACE_EVT_INIT, 0);

    size_t m_chunk_n_rows = m_chunk;
    size_t n_chunk_n_cols = n_chunk;

    ସଂରଚନା htp_mm_hmx_vtcm_layout L;
    htp_mm_hmx_vtcm_layout_build(&L, HTP_MM_KERNEL_HMX_BATCHED, HTP_TYPE_F16, params->k, m_chunk_n_rows, n_chunk_n_cols, group_size, false, act_threads, 0, params->bias_bytes);

    ଯଦି (L.total_bytes > vtcm_budget) {
        FARF(ଉପର_2, "%s: grouped layout overflowed VTCM, falling back to simple batched loop", __func__);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_INIT, 0);
        ଫେରନ୍ତୁ hmx_mm_batched_simple(ctx, params, m_chunk, n_chunk, pipeline, n_threads, act_threads, vtcm_size, act_threads_div, k_div);
    }

    uint8_t * ଅଚଳ base = (uint8_t *) ctx->vtcm_base;
    __fp16  *vtcm_weight     = VTCM_LAYOUT_PTR(__fp16, base, L.off_weight[0]);
    __fp16  *vtcm_f16_act    = VTCM_LAYOUT_PTR(__fp16, base, L.off_act);
    __fp16  *vtcm_output     = VTCM_LAYOUT_PTR(__fp16, base, L.off_dst[0]);
    ଶୂନ୍ୟ    *vtcm_scratch0   = VTCM_LAYOUT_PTR(ଶୂନ୍ୟ, base, L.off_scratch[0]);
    ଶୂନ୍ୟ    *vtcm_scratch1   = VTCM_LAYOUT_PTR(ଶୂନ୍ୟ, base, L.off_scratch[1]);
    __fp16  *vtcm_scales     = VTCM_LAYOUT_PTR(__fp16, base, L.off_scales);
    ଦଶମିକ   *vtcm_f32_act    = VTCM_LAYOUT_PTR(ଦଶମିକ, base, L.off_act_f32);

    ଅଚଳ bool has_bias      = (params->bias_bytes > 0 && params->bias_addr != 0);
    ଦଶମିକ   *vtcm_bias       = VTCM_LAYOUT_PTR_OPTIONAL(ଦଶମିକ, base, L.off_bias, has_bias);
    ଯଦି (has_bias) {
        dma_queue_ଠେଲ(params->weight_dma, dma_make_data(vtcm_bias, params->bias_addr), hex_align_up(params->bias_bytes, 128), 0, params->bias_bytes, 1);
        dma_queue_କାଢ଼(params->weight_dma);
    }

    hmx_init_column_scales(vtcm_scales, Q6_V_vsplat_R(0x3c00));  // scale: 1.0, bias: 0.0 in FP16

    FARF(ଉପର_2, "%s: grouped path m=%d k=%d n=%d group=%d streams=%d mc=%zu nc=%zu vtcm=%zu/%zu",
            __func__, params->m, params->k, params->n, group_size, params->ne13,
            m_chunk_n_rows, n_chunk_n_cols,
            L.total_bytes, vtcm_budget);

    ଅଚଳ size_t fp16_row_bytes   = (size_t) params->k * ଆକାର(__fp16);
    ଅଚଳ size_t weight_row_bytes = (size_t) params->weight_stride;

    htp_trace_event_stop(tr, HTP_TRACE_EVT_INIT, 0);

    hmx_matmul_job_t job;

    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା b3 = 0; b3 < params->ne13; ++b3) {
        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା b2_base = 0; b2_base < params->ne12; b2_base += group_size) {
            ଅଚଳ dma_addr_t weight_group = hmx_mm_weight_batch_data(params, b2_base, b3);
            dma_queue * weight_dma = params->weight_dma;

            ପାଇଁ (size_t mr = 0; mr < (size_t) params->m; mr += m_chunk_n_rows) {
                ଅଚଳ size_t n_rows = hex_smin((size_t) params->m - mr, m_chunk_n_rows);
                ଅଚଳ size_t n_row_tiles = hmx_ceil_div((ପୂର୍ଣ୍ଣସଂଖ୍ୟା) n_rows, HTP_MM_HMX_TILE_N_ROWS);

                // Pre-load activations for all heads in the group (once per m_chunk).
                // When the source is strided (permuted Q), use 2D DMA to gather
                // contiguous rows into a VTCM scratch buffer first, then HVX
                // converts from the contiguous VTCM buffer.  This avoids L2 cache
                // thrashing from HVX loads at large strides.
                ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା g = 0; g < group_size; ++g) {
                    ଅଚଳ dma_addr_t act_dma_addr = hmx_mm_act_batch_addr(params, b2_base + g, b3) +
                                                      (size_t) mr * params->act_stride * params->act_elem_size;
                    __fp16 *vtcm_act_g = vtcm_f16_act + (size_t) g * L.act_head_stride;
                    ସଂରଚନା activation_transfer_params act_params = {
                        .ctx = ctx,
                        .dst = vtcm_act_g,
                        .act_dma_addr = act_dma_addr,
                        .n_rows = (ପୂର୍ଣ୍ଣସଂଖ୍ୟା) n_rows,
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
                    ଅଚଳ size_t n_cols_first = hex_smin((size_t) params->n, n_chunk_n_cols);
                    dma_queue_ଠେଲ(weight_dma, dma_make_data(vtcm_scratch0, weight_group),
                                      fp16_row_bytes, weight_row_bytes, fp16_row_bytes, n_cols_first);
                }
                ଯଦି (n_chunk_n_cols < (size_t) params->n) {
                    ଅଚଳ size_t n_cols_second = hex_smin((size_t) params->n - n_chunk_n_cols, n_chunk_n_cols);
                    ଅଚଳ dma_addr_t second_weight_chunk = weight_group + n_chunk_n_cols * weight_row_bytes;
                    dma_queue_ଠେଲ(weight_dma, dma_make_data(vtcm_scratch1, second_weight_chunk),
                                      fp16_row_bytes, weight_row_bytes, fp16_row_bytes, n_cols_second);
                }

                ପାଇଁ (size_t nc = 0; nc < (size_t) params->n; nc += n_chunk_n_cols) {
                    ଅଚଳ size_t n_cols      = hex_smin((size_t) params->n - nc, n_chunk_n_cols);
                    ଅଚଳ size_t n_col_tiles = hmx_ceil_div((ପୂର୍ଣ୍ଣସଂଖ୍ୟା) n_cols, HTP_MM_HMX_TILE_N_COLS);

                    {
                        ଶୂନ୍ୟ * curr_raw = (ଶୂନ୍ୟ *) dma_queue_କାଢ଼(weight_dma).dst;

                        ଅଚଳ size_t n_cols_tiled = hex_align_up(n_cols, HTP_MM_HMX_TILE_N_COLS);

                        hmx_interleave_rows_to_tiles(vtcm_weight, (ଅଚଳ __fp16 *) curr_raw, (uint32_t) n_cols, params->k, params->k, 0, (uint32_t) n_cols_tiled);

                        ଅଚଳ size_t nc_next = nc + n_chunk_n_cols * 2;
                        ଯଦି (nc_next < (size_t) params->n) {
                            ଅଚଳ size_t n_cols_next = hex_smin((size_t) params->n - nc_next, n_chunk_n_cols);
                            ଅଚଳ dma_addr_t next_weight_chunk = weight_group + nc_next * weight_row_bytes;

                            dma_queue_ଠେଲ(weight_dma, dma_make_data(curr_raw, next_weight_chunk),
                                              fp16_row_bytes, weight_row_bytes, fp16_row_bytes, n_cols_next);
                        }
                    }

                    // Reuse the interleaved weight for every q_head in this GQA group
                    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା g = 0; g < group_size; ++g) {
                        {
                            ଅଚଳ __fp16 * vtcm_act_g = vtcm_f16_act + (size_t) g * L.act_head_stride;
                            hmx_matmul_job_init(&job, vtcm_output, vtcm_act_g, vtcm_weight, vtcm_scales, n_row_tiles, n_col_tiles, params->k / 32);
                            hmx_queue_ଠେଲ(ctx->hmx_queue, hmx_queue_make_desc(hmx_matmul_worker_fn, &job));
                            hmx_queue_କାଢ଼(ctx->hmx_queue);
                        }

                        {
                            ଦଶମିକ *output = hmx_mm_dst_batch_ptr(params, b2_base + g, b3) + mr * params->dst_stride + nc;
                            ଅଚଳ ଦଶମିକ *bias_chunk = has_bias ? (vtcm_bias + mr * params->bias_stride + nc) : ଖାଲି;
                            ପୂର୍ଣ୍ଣସଂଖ୍ୟା chunk_dst_cols = params->dst_cols - (ପୂର୍ଣ୍ଣସଂଖ୍ୟା)nc;
                            ଯଦି (chunk_dst_cols > 0) {
                                transfer_output_chunk_threaded(ctx, output, bias_chunk, vtcm_output, (ପୂର୍ଣ୍ଣସଂଖ୍ୟା) n_rows, (ପୂର୍ଣ୍ଣସଂଖ୍ୟା) n_cols,
                                                               params->dst_stride, params->bias_stride, chunk_dst_cols, n_threads);
                            }
                        }
                    }
                }
            }
        }
    }

    ଫେରନ୍ତୁ 0;
}

ସ୍ଥିର ଶୂନ୍ୟ transfer_activation_chunk_gathered_threaded(
            ସଂରଚନା htp_context *ctx,
            __fp16 *dst,
            ଅଚଳ ଦଶମିକ *src,
            ପୂର୍ଣ୍ଣସଂଖ୍ୟା start_row,
            ପୂର୍ଣ୍ଣସଂଖ୍ୟା n_rows,
            ପୂର୍ଣ୍ଣସଂଖ୍ୟା k_block,
            ଅଚଳ ସଂରଚନା mmid_row_mapping *matrix_rows,
            ପୂର୍ଣ୍ଣସଂଖ୍ୟା cur_a,
            ପୂର୍ଣ୍ଣସଂଖ୍ୟା mapping_stride,
            ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne11,
            size_t nb11,
            size_t nb12,
            ପୂର୍ଣ୍ଣସଂଖ୍ୟା cne1,
            ପୂର୍ଣ୍ଣସଂଖ୍ୟା n_threads,
            ପୂର୍ଣ୍ଣସଂଖ୍ୟା k_valid) {
    ଯଦି (n_rows <= 0) ଫେରନ୍ତୁ;
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା chunks_per_thread = hmx_ceil_div(n_rows, n_threads);
    chunks_per_thread = hex_align_up(chunks_per_thread, 2);

    ପୂର୍ଣ୍ଣସଂଖ୍ୟା actual_threads = hmx_ceil_div(n_rows, chunks_per_thread);

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
        .ne11_div          = ne11 > 1 ? init_fastdiv_ମୂଲ୍ୟଗୁଡ଼ିକ(ne11) : (ସଂରଚନା fastdiv_ମୂଲ୍ୟଗୁଡ଼ିକ){0, 0},
        .nb11              = nb11,
        .nb12              = nb12,
        .start_row         = start_row,
        .cne1              = cne1,
        .k_valid           = k_valid,
        .traces            = ctx->trace,
    };

    worker_callback_t worker_fn = ne11 == 1 ? transfer_activation_chunk_gathered_worker_flat_fn :
                                              transfer_activation_chunk_gathered_worker_fn;

    ଯଦି (actual_threads <= 1) {
        worker_fn(1, 0, &state);
    } ଅନ୍ୟଥା {
        worker_pool_run_func(ctx->worker_pool, worker_fn, &state, actual_threads);
    }
}

ସ୍ଥିର ଶୂନ୍ୟ transfer_output_chunk_scattered_threaded(
            ସଂରଚନା htp_context *ctx,
            ଦଶମିକ *dst,
            ଅଚଳ __fp16 *vtcm_src,
            ପୂର୍ଣ୍ଣସଂଖ୍ୟା start_row,
            ପୂର୍ଣ୍ଣସଂଖ୍ୟା n_rows,
            ପୂର୍ଣ୍ଣସଂଖ୍ୟା n_cols,
            ଅଚଳ ସଂରଚନା mmid_row_mapping *matrix_rows,
            ପୂର୍ଣ୍ଣସଂଖ୍ୟା cur_a,
            ପୂର୍ଣ୍ଣସଂଖ୍ୟା mapping_stride,
            size_t dst_nb1,
            size_t dst_nb2,
            ପୂର୍ଣ୍ଣସଂଖ୍ୟା cne1,
            ପୂର୍ଣ୍ଣସଂଖ୍ୟା n_threads) {
    ଯଦି (n_rows <= 0) ଫେରନ୍ତୁ;
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା chunks_per_thread = hmx_ceil_div(n_rows, n_threads);
    chunks_per_thread = hex_align_up(chunks_per_thread, 2);

    ପୂର୍ଣ୍ଣସଂଖ୍ୟା actual_threads = hmx_ceil_div(n_rows, chunks_per_thread);

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

    ଯଦି (actual_threads <= 1) {
        transfer_output_chunk_scattered_worker_fn(1, 0, &state);
    } ଅନ୍ୟଥା {
        worker_pool_run_func(ctx->worker_pool, transfer_output_chunk_scattered_worker_fn, &state, actual_threads);
    }
}

ସ୍ଥିର ପୂର୍ଣ୍ଣସଂଖ୍ୟା hmx_mm_id_2d_f32(ସଂରଚନା htp_context *ctx,
                                         dma_queue *weight_dma,
                                         ଦଶମିକ *restrict dst,
                                         ଅଚଳ ଦଶମିକ *activation,
                                         dma_addr_t weight,
                                         ପୂର୍ଣ୍ଣସଂଖ୍ୟା m, ପୂର୍ଣ୍ଣସଂଖ୍ୟା k, ପୂର୍ଣ୍ଣସଂଖ୍ୟା n,
                                         ପୂର୍ଣ୍ଣସଂଖ୍ୟା k_valid,
                                         ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne11,
                                         size_t act_nb1, size_t act_nb2,
                                         size_t dst_nb1, size_t dst_nb2,
                                         ପୂର୍ଣ୍ଣସଂଖ୍ୟା weight_stride,
                                         ପୂର୍ଣ୍ଣସଂଖ୍ୟା weight_type,
                                         ପୂର୍ଣ୍ଣସଂଖ୍ୟା tile_size,
                                         ପୂର୍ଣ୍ଣସଂଖ୍ୟା aligned_tile_size,
                                         ଅଚଳ ସଂରଚନା mmid_row_mapping *matrix_rows,
                                         ପୂର୍ଣ୍ଣସଂଖ୍ୟା cur_a,
                                         ପୂର୍ଣ୍ଣସଂଖ୍ୟା mapping_stride,
                                         ପୂର୍ଣ୍ଣସଂଖ୍ୟା m_start,
                                         ପୂର୍ଣ୍ଣସଂଖ୍ୟା m_end,
                                         ପୂର୍ଣ୍ଣସଂଖ୍ୟା n_threads) {
    ସଂରଚନା htp_thread_trace * tr = &ctx->trace[0];
    htp_trace_event_start(tr, HTP_TRACE_EVT_INIT, 0);

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା cne1 = m;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା m_core = m_end - m_start;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା m_core_padded = hex_align_up(m_core > 0 ? m_core : 1, 32);

    ଯଦି (k % 32 != 0 || n % 32 != 0) { ଫେରନ୍ତୁ -1; }
    ଯଦି (!hex_is_aligned(dst, VLEN) || !hex_is_aligned(activation, VLEN)) { ଫେରନ୍ତୁ -1; }

    size_t row_stride = htp_tensor_get_row_size(weight_type, k);
    ଯଦି (row_stride == 0) {
        ଫେରନ୍ତୁ -1;
    }

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା n_k_tiles = k / HTP_MM_HMX_TILE_N_COLS;
    ଅଚଳ uint32_t n_weight_tiles = k / htp_tensor_get_tile_k(weight_type);
    ଅଚଳ ସଂରଚନା fastdiv_ମୂଲ୍ୟଗୁଡ଼ିକ n_k_tiles_div = init_fastdiv_ମୂଲ୍ୟଗୁଡ଼ିକ(n_k_tiles);

    ଅଚଳ size_t vec_dot_size = k * ଆକାର(__fp16);
    ଅଚଳ size_t vtcm_budget  = ctx->vtcm_size;
    size_t vtcm_used = 0;

    ଅଚଳ bool is_tiled = (tile_size > 0);
    tiled_dequant_task_fn_t dequant_worker_fn = hmx_get_dequant_worker_fn(weight_type, is_tiled);

    ଅଚଳ uint32_t dma_dst_stride  = is_tiled ? aligned_tile_size : row_stride;
    ଅଚଳ uint32_t dma_src_stride  = is_tiled ? tile_size : weight_stride;
    ଅଚଳ uint32_t dma_width_bytes = is_tiled ? tile_size : row_stride;

    ଅଚଳ size_t qweight_row_stride = is_tiled ? (size_t)(n_weight_tiles * aligned_tile_size) / 32 : 0;
    ଅଚଳ size_t weight_row_stride = is_tiled ? qweight_row_stride : row_stride;

    size_t size_per_n = 0, size_per_m = 0, size_per_mn = 0;
    htp_mm_hmx_get_2d_chunk_costs(weight_type, k, /*pipeline=*/false, aligned_tile_size,
                                  &size_per_n, &size_per_m, &size_per_mn);

    ଅଚଳ size_t overhead = htp_mm_hmx_get_2d_overhead(/*pipeline=*/false, /*is_matmul_id=*/true);
    size_t m_chunk_n_rows = 0, n_chunk_n_cols = 0;
    ଯଦି (htp_mm_hmx_compute_chunks(vtcm_budget, overhead, size_per_n, size_per_m, size_per_mn,
                           m_core_padded, n,
                           /*m_block_cost=*/(size_t) n * HTP_MM_HMX_COST_W_DEQUANT,
                           /*n_block_cost=*/(size_t) m_core_padded * HTP_MM_HMX_COST_A_CONVERT, &m_chunk_n_rows, &n_chunk_n_cols, &vtcm_used)) {
        FARF(ERROR, "hmx-mm-id-2d: VTCM too small : m %d k %d n %d budget %zu", m_core_padded, k, n, vtcm_budget);
        ଫେରନ୍ତୁ -1;
    }

    ଅଚଳ size_t weight_area_size = hex_align_up(n_chunk_n_cols * weight_row_stride, HTP_MM_HMX_TILE_SIZE);
    ଅଚଳ size_t act_area_size    = hex_align_up(m_chunk_n_rows * vec_dot_size, HTP_MM_HMX_TILE_SIZE);
    ଅଚଳ size_t output_area_size = hex_align_up(m_chunk_n_rows * n_chunk_n_cols * ଆକାର(__fp16), HTP_MM_HMX_TILE_SIZE);

    size_t scratch0_size = hex_align_up(n_chunk_n_cols * vec_dot_size, HTP_MM_HMX_TILE_SIZE);

    uint8_t *vtcm_ptr      = (uint8_t *) ctx->vtcm_base;
    __fp16  *vtcm_weight   = weight_area_size ? (__fp16 *) vtcm_seq_alloc(&vtcm_ptr, weight_area_size) : ଖାଲି;
    __fp16  *vtcm_f16_act  = (__fp16 *) vtcm_seq_alloc(&vtcm_ptr, act_area_size);
    __fp16  *vtcm_output   = (__fp16 *) vtcm_seq_alloc(&vtcm_ptr, output_area_size);
    ଶୂନ୍ୟ    *vtcm_scratch0 = vtcm_seq_alloc(&vtcm_ptr, scratch0_size);
    __fp16  *vtcm_scales   = (__fp16 *) vtcm_seq_alloc(&vtcm_ptr, 256);

    vtcm_used = vtcm_ptr - (uint8_t *) ctx->vtcm_base;
    ଯଦି (vtcm_used > vtcm_budget) {
        FARF(ERROR, "hmx-mm-id-2d: VTCM overflow: used %zu budget %zu", vtcm_used, vtcm_budget);
        ଫେରନ୍ତୁ -1;
    }

    hmx_init_column_scales(vtcm_scales, Q6_V_vsplat_R(0x3c00));

    htp_trace_event_stop(tr, HTP_TRACE_EVT_INIT, 0);

    hmx_matmul_job_t job;
    ଶୂନ୍ୟ *vtcm_raw_bufs[2] = { vtcm_weight, dequant_worker_fn ? vtcm_weight : vtcm_scratch0 };
    ଶୂନ୍ୟ *vtcm_weight_bufs[2] = { dequant_worker_fn ? vtcm_scratch0 : vtcm_weight, vtcm_scratch0 };
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା n_chunk_cnt = hmx_ceil_div(n, n_chunk_n_cols);

    ପାଇଁ (size_t mr = (size_t) m_start; mr < (size_t) m_end; mr += m_chunk_n_rows) {
        ଅଚଳ size_t n_rows = hex_smin((size_t) m_end - mr, m_chunk_n_rows);
        ଅଚଳ size_t n_row_tiles = hmx_ceil_div(n_rows, HTP_MM_HMX_TILE_N_ROWS);

        transfer_activation_chunk_gathered_threaded(
            ctx, vtcm_f16_act, activation, (ପୂର୍ଣ୍ଣସଂଖ୍ୟା) mr, (ପୂର୍ଣ୍ଣସଂଖ୍ୟା) n_rows, k,
            matrix_rows, cur_a, mapping_stride, ne11, act_nb1, act_nb2, cne1, n_threads, k_valid);

        // A0: Pre-fetch the first weight chunk (nc = 0)
        ଯଦି (n > 0) {
            ଅଚଳ size_t n_cols = hex_smin((size_t) n, n_chunk_n_cols);
            ଅଚଳ uint32_t height = is_tiled ? hmx_ceil_div(n_cols, 32) * n_weight_tiles : n_cols;
            dma_queue_ଠେଲ(weight_dma, dma_make_data(vtcm_raw_bufs[0], weight),
                           dma_dst_stride, dma_src_stride, dma_width_bytes, height);
        }

        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < n_chunk_cnt; ++i) {
            ଅଚଳ size_t nc = (size_t) i * n_chunk_n_cols;
            ଅଚଳ size_t n_cols = hex_smin((size_t) n - nc, n_chunk_n_cols);
            ଅଚଳ size_t n_col_tiles = hmx_ceil_div(n_cols, HTP_MM_HMX_TILE_N_COLS);

            // A: Wait for weight DMA
            ଶୂନ୍ୟ * curr_raw = (ଶୂନ୍ୟ *) dma_queue_କାଢ଼(weight_dma).dst;

            // B: Weight Dequantize (Threaded)
            dequantize_tiled_weight_chunk_to_fp16_tiles(
                ctx, vtcm_weight_bufs[i % 2], curr_raw,
                n_cols, k, row_stride, weight_type,
                n_k_tiles, n_k_tiles_div, dequant_worker_fn, n_threads
            );

            // Start weight DMA for the next chunk early
            ଅଚଳ size_t nc_next = nc + n_chunk_n_cols;
            ଯଦି (nc_next < (size_t) n) {
                ଅଚଳ size_t n_cols_next = hex_smin((size_t) n - nc_next, n_chunk_n_cols);
                ଅଚଳ uint32_t height_next = is_tiled ? hmx_ceil_div(n_cols_next, 32) * n_weight_tiles : n_cols_next;
                dma_queue_ଠେଲ(weight_dma, dma_make_data(vtcm_raw_bufs[(i + 1) % 2], weight + nc_next * weight_stride),
                               dma_dst_stride, dma_src_stride, dma_width_bytes, height_next);
            }

            // C: HMX Compute (Queue-based)
            hmx_matmul_job_init(&job, vtcm_output, vtcm_f16_act, (__fp16 *) vtcm_weight_bufs[i % 2], vtcm_scales, n_row_tiles, n_col_tiles, k / HTP_MM_HMX_TILE_N_ROWS);
            hmx_queue_ଠେଲ(ctx->hmx_queue, hmx_queue_make_desc(hmx_matmul_worker_fn, &job));
            hmx_queue_କାଢ଼(ctx->hmx_queue);

            // D: Output Store
            transfer_output_chunk_scattered_threaded(
                ctx, dst + nc, vtcm_output, (ପୂର୍ଣ୍ଣସଂଖ୍ୟା) mr, (ପୂର୍ଣ୍ଣସଂଖ୍ୟା) n_rows, (ପୂର୍ଣ୍ଣସଂଖ୍ୟା) n_cols,
                matrix_rows, cur_a, mapping_stride, dst_nb1, dst_nb2, cne1, n_threads);
        }
    }

    ଫେରନ୍ତୁ 0;
}

// --- Dispatchers and Public Entry Points ---

ସ୍ଥିର ପୂର୍ଣ୍ଣସଂଖ୍ୟା hmx_mm_op_matmul(ସଂରଚନା htp_ops_context * octx, ଅଚଳ ସଂରଚନା htp_mm_kernel_params * kparams) {
    htp_matmul_tensors_preamble;

    ପୂର୍ଣ୍ଣସଂଖ୍ୟା k = (ପୂର୍ଣ୍ଣସଂଖ୍ୟା) src0->ne[0];
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା n = (ପୂର୍ଣ୍ଣସଂଖ୍ୟା) src0->ne[1];
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା m_total    = (ପୂର୍ଣ୍ଣସଂଖ୍ୟା) act->ne[1];
    ଅଚଳ uint32_t act_elem_size = (act->type == HTP_TYPE_F16) ? ଆକାର(__fp16) : ଆକାର(ଦଶମିକ);
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା act_stride = (ପୂର୍ଣ୍ଣସଂଖ୍ୟା) (act->nb[1] / act_elem_size);

    ପୂର୍ଣ୍ଣସଂଖ୍ୟା m_start = 0;
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା m_rows  = m_total;
    ଯଦି (octx->ctx->mdev.ଗଣନା_2 > 1) {
        bool can_split = htp_tensor_can_row_partition(dst, ଆକାର(ଦଶମିକ)) &&
                         htp_tensor_can_row_partition(act, act_elem_size);
        ଯଦି (src2 && src2->ne[1] > 1 && !htp_tensor_can_row_partition(src2, ଆକାର(ଦଶମିକ))) {
            can_split = false;
        }
        ଅଚଳ ସଂରଚନା htp_tensor_mdev_range range = htp_tensor_mdev_partition((uint32_t) m_total, can_split ? 1 : 0, octx->ctx->mdev.idx, octx->ctx->mdev.ଗଣନା_2, &octx->ctx->mdev.ଗଣନା_div);
        m_start = (ପୂର୍ଣ୍ଣସଂଖ୍ୟା) range.start;
        m_rows  = (ପୂର୍ଣ୍ଣସଂଖ୍ୟା) range.ଗଣନା_2;
    }

    ଯଦି (m_rows == 0) {
        ଫେରନ୍ତୁ HTP_STATUS_OK;
    }

    dma_addr_t src2_addr = 0;
    size_t src2_bytes = 0;
    uint32_t src2_stride = 0;
    size_t src2_nb2 = 0;
    size_t src2_nb3 = 0;
    ଯଦି (src2) {
        src2_stride = (src2->ne[1] == 1) ? 0 : (uint32_t) (src2->nb[1] / ଆକାର(ଦଶମିକ));
        src2_addr = src2->data + m_start * src2_stride * ଆକାର(ଦଶମିକ);
        src2_bytes = (size_t) kparams->vtcm_bias_size;
        src2_nb2 = (src2->ne[2] == 1) ? 0 : src2->nb[2];
        src2_nb3 = (src2->ne[3] == 1) ? 0 : src2->nb[3];
    }

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା dst_stride = (ପୂର୍ଣ୍ଣସଂଖ୍ୟା)(dst->nb[1] / ଆକାର(ଦଶମିକ));
    ଦଶମିକ       * dst_ptr = (ଦଶମିକ *) dst->data + m_start * dst_stride;
    // byte offset, act_stride is in activation elements (F16 or F32)
    ଅଚଳ dma_addr_t act_addr = act->data + (size_t) m_start * act_stride * act_elem_size;

    ପୂର୍ଣ୍ଣସଂଖ୍ୟା ret = -1;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା n_threads = kparams->n_threads;
    ଯଦି (kparams->kernel_type == HTP_MM_KERNEL_HMX_BATCHED) {
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
            .weight_stride     = (ପୂର୍ଣ୍ଣସଂଖ୍ୟା) src0->nb[1],
            .weight_type       = (ପୂର୍ଣ୍ଣସଂଖ୍ୟା) src0->type,
            .dst_stride        = dst_stride,
            .dst_cols          = (ପୂର୍ଣ୍ଣସଂଖ୍ୟା) dst->ne[0],
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
    } ଅନ୍ୟଥା {
        ret = hmx_mm_2d_f32(
            octx->ctx, octx->ctx->dma[0], dst_ptr, src2_addr, src2_bytes,
            act_addr, src0->data,
            m_rows, k, n, act_stride, act_elem_size, (ପୂର୍ଣ୍ଣସଂଖ୍ୟା) src0->nb[1], (ପୂର୍ଣ୍ଣସଂଖ୍ୟା) src0->type, (ପୂର୍ଣ୍ଣସଂଖ୍ୟା) act->ne[0],
            dst_stride, src2_stride, (ପୂର୍ଣ୍ଣସଂଖ୍ୟା)dst->ne[0],
            kparams->m_chunk, kparams->n_chunk, kparams->pipeline, n_threads,
            kparams->n_act_threads,
            &kparams->div_n_act_threads,
            &kparams->div_ne00_padded,
            kparams->tile_size, kparams->aligned_tile_size, kparams->vtcm_size
        );
    }

    ଯଦି (ret != 0) {
        FARF(ERROR, "HMX matmul failed (ret=%d)\n", ret);
        ଫେରନ୍ତୁ HTP_STATUS_INTERNAL_ERR;
    }
    ଫେରନ୍ତୁ HTP_STATUS_OK;
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ htp_mm_tensor_collapse_rows(ସଂରଚନା htp_tensor * c, ଅଚଳ ସଂରଚନା htp_tensor * t, uint32_t stride) {
    *c = *t;
    c->ne[1] = t->ne[1] * t->ne[2] * t->ne[3];
    c->ne[2] = 1;
    c->ne[3] = 1;
    c->nb[1] = stride;
    c->nb[2] = c->nb[1] * c->ne[1];
    c->nb[3] = c->nb[2];
}

ସ୍ଥିର ପୂର୍ଣ୍ଣସଂଖ୍ୟା op_matmul_impl(ସଂରଚନା htp_ops_context * octx) {
    ଅଚଳ ସଂରଚନା htp_mm_kernel_params * kparams = (ଅଚଳ ସଂରଚନା htp_mm_kernel_params *) octx->kernel_params;

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା status = htp_mm_init_context(octx, kparams);
    ଯଦି (status != HTP_STATUS_OK) {
        ଫେରନ୍ତୁ status;
    }

    ଯଦି (kparams->n_hmx) {
        ଫେରନ୍ତୁ hmx_mm_op_matmul(octx, kparams);
    }

    ଫେରନ୍ତୁ hvx_mm_matmul(octx);
}

ପୂର୍ଣ୍ଣସଂଖ୍ୟା op_matmul(ସଂରଚନା htp_ops_context * octx) {
    ଅଚଳ ସଂରଚନା htp_mm_kernel_params * kparams = (ଅଚଳ ସଂରଚନା htp_mm_kernel_params *) octx->kernel_params;

    ଯଦି (kparams->collapse) {
        ଅଚଳ ସଂରଚନା htp_tensor * act  = octx->src[1];
        ଅଚଳ ସଂରଚନା htp_tensor * dst  = octx->dst;
        ଅଚଳ uint32_t s_act = (act->ne[1] > 1) ? act->nb[1] : ((act->ne[2] > 1) ? act->nb[2] : act->nb[3]);
        ଅଚଳ uint32_t sd    = (dst->ne[1] > 1) ? dst->nb[1] : ((dst->ne[2] > 1) ? dst->nb[2] : dst->nb[3]);
        ସଂରଚନା htp_tensor act_collapsed, dst_collapsed;
        htp_mm_tensor_collapse_rows(&act_collapsed, act, s_act);
        htp_mm_tensor_collapse_rows(&dst_collapsed, dst, sd);
        octx->src[1] = &act_collapsed;
        octx->dst    = &dst_collapsed;

        ସଂରଚନା htp_tensor src2_collapsed;
        ଅଚଳ ସଂରଚନା htp_tensor * src2 = octx->src[2];
        ଯଦି (src2 && (src2->ne[1] * src2->ne[2] * src2->ne[3] > 1)) {
            ଅଚଳ uint32_t s2 = (src2->ne[1] > 1) ? src2->nb[1] : ((src2->ne[2] > 1) ? src2->nb[2] : src2->nb[3]);
            htp_mm_tensor_collapse_rows(&src2_collapsed, src2, s2);
            octx->src[2] = &src2_collapsed;
        }

        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା status = op_matmul_impl(octx);
        octx->src[1] = act;
        octx->dst    = dst;
        ଯଦି (src2) {
            octx->src[2] = src2;
        }
        ଫେରନ୍ତୁ status;
    }

    ଫେରନ୍ତୁ op_matmul_impl(octx);
}

ସ୍ଥିର ପୂର୍ଣ୍ଣସଂଖ୍ୟା hmx_mm_op_matmul_id(
    ସଂରଚନା htp_ops_context * octx,
    ସଂରଚନା htp_mm_context * mmctx
) {
    ଅଚଳ uint32_t *                matrix_row_counts = mmctx->matrix_row_counts;
    ଅଚଳ ସଂରଚନା mmid_row_mapping * matrix_rows       = mmctx->matrix_rows;
    htp_matmul_tensors_preamble;
    ଅଚଳ ସଂରଚନା htp_mm_kernel_params * kparams = (ଅଚଳ ସଂରଚନା htp_mm_kernel_params *) octx->kernel_params;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା n_ids = octx->src[2]->ne[0];
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା n_as  = ne02;

    ଅଚଳ bool mdev_split = (octx->ctx->mdev.ଗଣନା_2 > 1) && htp_tensor_can_row_partition(dst, ଆକାର(ଦଶମିକ));
    ଯଦି (octx->ctx->mdev.ଗଣନା_2 > 1 && !mdev_split && octx->ctx->mdev.idx > 0) {
        ଫେରନ୍ତୁ HTP_STATUS_OK;
    }
    uint32_t n_active = 0;
    ଯଦି (mdev_split) {
        ପାଇଁ (uint32_t a = 0; a < (uint32_t) n_as; ++a) {
            ଯଦି (matrix_row_counts[a] > 0) n_active++;
        }
    }
    ଅଚଳ bool expert_split = mdev_split && (n_active >= octx->ctx->mdev.ଗଣନା_2);

    uint32_t target_dev = 0;
    ପାଇଁ (uint32_t cur_a = 0; cur_a < (uint32_t) n_as; ++cur_a) {
        ଅଚଳ int32_t cne1 = matrix_row_counts[cur_a];
        ଯଦି (cne1 == 0) ଜାରି;

        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା m_padded = hex_align_up(cne1, 32);
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା m_start = 0, m_end = m_padded;
        ଯଦି (expert_split) {
            ଅଚଳ bool my_expert = (target_dev == octx->ctx->mdev.idx);
            ଯଦି (++target_dev == octx->ctx->mdev.ଗଣନା_2) {
                target_dev = 0;
            }
            ଯଦି (!my_expert) ଜାରି;
        } ଅନ୍ୟଥା ଯଦି (mdev_split) {
            ଅଚଳ bool can_split = (uint32_t) cne1 >= octx->ctx->mdev.ଗଣନା_2;
            ଅଚଳ ସଂରଚନା htp_tensor_mdev_range range = htp_tensor_mdev_partition((uint32_t) m_padded, can_split ? 1 : 0, octx->ctx->mdev.idx, octx->ctx->mdev.ଗଣନା_2, &octx->ctx->mdev.ଗଣନା_div);
            m_start = (ପୂର୍ଣ୍ଣସଂଖ୍ୟା) range.start;
            m_end   = (ପୂର୍ଣ୍ଣସଂଖ୍ୟା) (range.start + range.ଗଣନା_2);
        }
        ଯଦି (m_start >= m_end) ଜାରି;

        ପୂର୍ଣ୍ଣସଂଖ୍ୟା ret = hmx_mm_id_2d_f32(octx->ctx, octx->ctx->dma[0], (ଦଶମିକ*) dst->data, (ଦଶମିକ*) act->data,
                                   src0->data + cur_a * nb02,
                                   cne1, ne00, ne01,
                                   ne10,
                                   ne11,
                                   nb11, nb12,
                                   nb1, nb2,
                                   (ପୂର୍ଣ୍ଣସଂଖ୍ୟା) src0->nb[1], (ପୂର୍ଣ୍ଣସଂଖ୍ୟା) src0->type,
                                   kparams->tile_size, kparams->aligned_tile_size,
                                   matrix_rows, cur_a, mmctx->mapping_stride,
                                   m_start, m_end, (ପୂର୍ଣ୍ଣସଂଖ୍ୟା) octx->n_threads);
        ଯଦି (ret != 0) {
            FARF(ERROR, "HMX matmul failed for expert %u, error %d\n", cur_a, ret);
            ଫେରନ୍ତୁ HTP_STATUS_NO_SUPPORT;
        }
    }

    ଫେରନ୍ତୁ HTP_STATUS_OK;
}

ସ୍ଥିର ପୂର୍ଣ୍ଣସଂଖ୍ୟା hvx_mm_matmul_id(
    ସଂରଚନା htp_ops_context * octx,
    ସଂରଚନା htp_mm_context * mmctx,
    work_queue_func_t hvx_mmid_task_func
) {
    htp_matmul_tensors_preamble;
    ଅଚଳ uint32_t src0_row_size_padded = mmctx->src0_row_size_padded;
    ଅଚଳ uint32_t act_nrows            = mmctx->act_nrows;

    ସଂରଚନା htp_thread_trace * tr = &octx->ctx->trace[0];
    htp_trace_event_start(tr, HTP_TRACE_EVT_INIT, 0);

    ଅଚଳ ସଂରଚନା htp_mm_kernel_params * kparams = (ଅଚଳ ସଂରଚନା htp_mm_kernel_params *) octx->kernel_params;
    ଅଚଳ ସଂରଚନା htp_tensor * restrict ids = octx->src[2];
    ଅଚଳ size_t src0_row_size = nb01;

    ଅଚଳ uint32_t qk = QK_Q8_0_TILED;
    ଅଚଳ uint32_t nb = (ne10 + qk - 1) / qk;
    ଅଚଳ uint32_t total_nb = act_nrows * nb;

    work_queue_func_t quant_task_func;
    uint32_t n_quant_tasks = 1;
    ଯଦି (act_nrows < octx->n_threads) {
        n_quant_tasks = MIN(total_nb, octx->n_threads);
        quant_task_func = htp_mm_act_quant_block_func(src0->type, act->type);
        ପାଇଁ (uint32_t ith = 0; ith < n_quant_tasks; ++ith) {
            uint32_t ib_first = (total_nb * ith) / n_quant_tasks;
            uint32_t ib_last  = (total_nb * (ith + 1)) / n_quant_tasks;
            mmctx->quant_ib_first[ith] = ib_first;
            mmctx->quant_ib_last[ith]  = ib_last;
            mmctx->quant_r[ith]        = ib_first / nb;
            mmctx->quant_c[ith]        = ib_first % nb;
        }
    } ଅନ୍ୟଥା {
        n_quant_tasks = MIN(act_nrows, octx->n_threads);
        quant_task_func = htp_mm_act_quant_row_func(src0->type, act->type);
    }
    size_t act_row_size = htp_mm_tiled_act_row_size(src0->type, ne10);

    ସଂରଚନା htp_mm_hvx_vtcm_layout L;
    htp_mm_hvx_vtcm_layout_build(&L, kparams->kernel_type, src0->type, ne10, act_nrows, octx->n_threads,
                                 0, src0_row_size, act_row_size, 0, kparams->n_prefetch, true, false);

    ଅଚଳ size_t vtcm_size = L.total_bytes;

    FARF(ଉପର_2, "matmul-id-%s : src0-spad-size %zu act-spad-size %zu bias-spad-size 0 dst-spad-size %zu (%zu)\n", mmctx->type,
         L.src0_bytes, L.act_bytes, L.dst_bytes, vtcm_size);

    FARF(ଉପର_2, "matmul-id-%s : %ux%ux%ux%u * %ux%ux%ux%u (%ux%ux%ux%u) -> %ux%ux%ux%u (0x%p, 0x%p, 0x%p)\n", mmctx->type,
         src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3], act->ne[0], act->ne[1], act->ne[2], act->ne[3],
         ids->ne[0], ids->ne[1], ids->ne[2], ids->ne[3], dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3], src0->data,
         act->data, dst->data);

    // Make sure the reserved vtcm size is sufficient
    ଯଦି (octx->ctx->vtcm_size < vtcm_size) {
        FARF(ERROR, "matmul-id-%s : current VTCM reservation %zu is too small, needed %zu\n", mmctx->type, octx->ctx->vtcm_size, vtcm_size);
        ଫେରନ୍ତୁ HTP_STATUS_VTCM_TOO_SMALL;
    }

    uint8_t * ଅଚଳ base = (uint8_t *) octx->ctx->vtcm_base;
    mmctx->vtcm_act      = VTCM_LAYOUT_PTR(uint8_t, base, L.off_act);
    mmctx->vtcm_src0     = VTCM_LAYOUT_PTR(uint8_t, base, L.off_src0);
    mmctx->vtcm_bias     = ଖାଲି;
    mmctx->vtcm_dst      = VTCM_LAYOUT_PTR(uint8_t, base, L.off_dst);
    mmctx->vtcm_act_raw  = VTCM_LAYOUT_PTR(uint8_t, base, L.off_act_raw);

    mmctx->vtcm_src0_stride    = src0_row_size_padded;
    mmctx->vtcm_act_stride     = act_row_size;
    mmctx->vtcm_act_raw_stride = hex_round_up(ne10 * ଆକାର(ଦଶମିକ), QK_Q8_0_TILED * ଆକାର(ଦଶମିକ));

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

    ଫେରନ୍ତୁ HTP_STATUS_OK;
}

ସ୍ଥିର ପୂର୍ଣ୍ଣସଂଖ୍ୟା hmx_mm_op_matmul_id_nx(
    ସଂରଚନା htp_ops_context * octx,
    ସଂରଚନା htp_mm_context * mmctx
) {
    ଅଚଳ uint32_t *                matrix_row_counts = mmctx->matrix_row_counts;
    ଅଚଳ ସଂରଚନା mmid_row_mapping * matrix_rows       = mmctx->matrix_rows;
    ଅଚଳ ସଂରଚନା htp_mm_kernel_params * kparams = (ଅଚଳ ସଂରଚନା htp_mm_kernel_params *) octx->kernel_params;
    ଅଚଳ uint32_t n_weights = kparams->n_weights;
    ଅଚଳ ସଂରଚନା htp_tensor * restrict src0 = octx->src[0];
    ଅଚଳ ସଂରଚନା htp_tensor * restrict act  = octx->src[n_weights];
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା n_as = src0->ne[2];

    bool mdev_split = (octx->ctx->mdev.ଗଣନା_2 > 1);
    ପାଇଁ (uint32_t p = 0; p < n_weights && mdev_split; ++p) {
        ଅଚଳ ସଂରଚନା htp_tensor * restrict dst = octx->dsts[p];
        mdev_split = htp_tensor_can_row_partition(dst, ଆକାର(ଦଶମିକ));
    }
    ଯଦି (octx->ctx->mdev.ଗଣନା_2 > 1 && !mdev_split && octx->ctx->mdev.idx > 0) {
        ଫେରନ୍ତୁ HTP_STATUS_OK;
    }
    uint32_t n_active = 0;
    ଯଦି (mdev_split) {
        ପାଇଁ (uint32_t a = 0; a < (uint32_t) n_as; ++a) {
            ଯଦି (matrix_row_counts[a] > 0) n_active++;
        }
    }
    ଅଚଳ bool expert_split = mdev_split && (n_active >= octx->ctx->mdev.ଗଣନା_2);

    uint32_t target_dev = 0;
    ପାଇଁ (uint32_t cur_a = 0; cur_a < (uint32_t) n_as; ++cur_a) {
        ଅଚଳ int32_t cne1 = matrix_row_counts[cur_a];
        ଯଦି (cne1 == 0) ଜାରି;

        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା m_padded = hex_align_up(cne1, 32);
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା m_start = 0, m_end = m_padded;
        ଯଦି (expert_split) {
            ଅଚଳ bool my_expert = (target_dev == octx->ctx->mdev.idx);
            ଯଦି (++target_dev == octx->ctx->mdev.ଗଣନା_2) {
                target_dev = 0;
            }
            ଯଦି (!my_expert) ଜାରି;
        } ଅନ୍ୟଥା ଯଦି (mdev_split) {
            ଅଚଳ bool can_split = (uint32_t) cne1 >= octx->ctx->mdev.ଗଣନା_2;
            ଅଚଳ ସଂରଚନା htp_tensor_mdev_range range = htp_tensor_mdev_partition((uint32_t) m_padded, can_split ? 1 : 0, octx->ctx->mdev.idx, octx->ctx->mdev.ଗଣନା_2, &octx->ctx->mdev.ଗଣନା_div);
            m_start = (ପୂର୍ଣ୍ଣସଂଖ୍ୟା) range.start;
            m_end   = (ପୂର୍ଣ୍ଣସଂଖ୍ୟା) (range.start + range.ଗଣନା_2);
        }
        ଯଦି (m_start >= m_end) ଜାରି;

        ପାଇଁ (uint32_t p = 0; p < n_weights; ++p) {
            ଅଚଳ ସଂରଚନା htp_tensor * restrict src_w = octx->src[p];
            ଅଚଳ ସଂରଚନା htp_tensor * restrict dst   = octx->dsts[p];
            ଯଦି (!src_w || !dst) ଜାରି;

            ପୂର୍ଣ୍ଣସଂଖ୍ୟା ret = hmx_mm_id_2d_f32(octx->ctx, octx->ctx->dma[0], (ଦଶମିକ*) dst->data, (ଦଶମିକ*) act->data,
                                       src_w->data + cur_a * src_w->nb[2],
                                       cne1, src_w->ne[0], src_w->ne[1],
                                       act->ne[0],
                                       act->ne[1],
                                       act->nb[1], act->nb[2],
                                       dst->nb[1], dst->nb[2],
                                       (ପୂର୍ଣ୍ଣସଂଖ୍ୟା) src_w->nb[1], (ପୂର୍ଣ୍ଣସଂଖ୍ୟା) src_w->type,
                                       kparams->tile_size, kparams->aligned_tile_size,
                                       matrix_rows, cur_a, mmctx->mapping_stride,
                                       m_start, m_end, (ପୂର୍ଣ୍ଣସଂଖ୍ୟା) octx->n_threads);
            ଯଦି (ret != 0) {
                FARF(ERROR, "HMX matmul ID NX failed for expert %u weight %u, error %d\n", cur_a, p, ret);
                ଫେରନ୍ତୁ HTP_STATUS_NO_SUPPORT;
            }
        }
    }

    ଫେରନ୍ତୁ HTP_STATUS_OK;
}

ସ୍ଥିର ପୂର୍ଣ୍ଣସଂଖ୍ୟା hvx_mm_matmul_id_nx(
    ସଂରଚନା htp_ops_context * octx,
    ସଂରଚନା htp_mm_context * mmctx,
    work_queue_func_t hvx_mmid_task_func
) {
    ଅଚଳ uint32_t src0_row_size_padded = mmctx->src0_row_size_padded;
    ଅଚଳ uint32_t act_nrows            = mmctx->act_nrows;

    ସଂରଚନା htp_thread_trace * tr = &octx->ctx->trace[0];
    htp_trace_event_start(tr, HTP_TRACE_EVT_INIT, 0);

    ଅଚଳ ସଂରଚନା htp_mm_kernel_params * kparams = (ଅଚଳ ସଂରଚନା htp_mm_kernel_params *) octx->kernel_params;
    ଅଚଳ uint32_t n_weights = kparams->n_weights;
    ଅଚଳ ସଂରଚନା htp_tensor * restrict src0 = octx->src[0];
    ଅଚଳ ସଂରଚନା htp_tensor * restrict act  = octx->src[n_weights];
    ଅଚଳ ସଂରଚନା htp_tensor * restrict ids  = octx->src[n_weights + 1];
    ଅଚଳ size_t src0_row_size = src0->nb[1];

    ଅଚଳ uint32_t qk = QK_Q8_0_TILED;
    ଅଚଳ uint32_t nb = (act->ne[0] + qk - 1) / qk;
    ଅଚଳ uint32_t total_nb = act_nrows * nb;

    work_queue_func_t quant_task_func;
    uint32_t n_quant_tasks = 1;
    ଯଦି (act_nrows < octx->n_threads) {
        n_quant_tasks = MIN(total_nb, octx->n_threads);
        quant_task_func = htp_mm_act_quant_block_func(src0->type, act->type);
        ପାଇଁ (uint32_t ith = 0; ith < n_quant_tasks; ++ith) {
            uint32_t ib_first = (total_nb * ith) / n_quant_tasks;
            uint32_t ib_last  = (total_nb * (ith + 1)) / n_quant_tasks;
            mmctx->quant_ib_first[ith] = ib_first;
            mmctx->quant_ib_last[ith]  = ib_last;
            mmctx->quant_r[ith]        = ib_first / nb;
            mmctx->quant_c[ith]        = ib_first % nb;
        }
    } ଅନ୍ୟଥା {
        n_quant_tasks = MIN(act_nrows, octx->n_threads);
        quant_task_func = htp_mm_act_quant_row_func(src0->type, act->type);
    }
    size_t act_row_size = htp_mm_tiled_act_row_size(src0->type, act->ne[0]);

    ସଂରଚନା htp_mm_hvx_vtcm_layout L;
    htp_mm_hvx_vtcm_layout_build(&L, kparams->kernel_type, src0->type, act->ne[0], act_nrows, octx->n_threads,
                                 0, src0_row_size, act_row_size, 0, kparams->n_prefetch, true, false);

    ଅଚଳ size_t vtcm_size = L.total_bytes;

    ଯଦି (octx->ctx->vtcm_size < vtcm_size) {
        FARF(ERROR, "matmul-id-nx: current VTCM reservation %zu is too small, needed %zu\n",
             octx->ctx->vtcm_size, vtcm_size);
        ଫେରନ୍ତୁ HTP_STATUS_VTCM_TOO_SMALL;
    }

    uint8_t * ଅଚଳ base = (uint8_t *) octx->ctx->vtcm_base;
    mmctx->vtcm_src0     = VTCM_LAYOUT_PTR(uint8_t, base, L.off_src0);
    mmctx->vtcm_act      = VTCM_LAYOUT_PTR(uint8_t, base, L.off_act);
    mmctx->vtcm_dst      = VTCM_LAYOUT_PTR(uint8_t, base, L.off_dst);
    mmctx->vtcm_act_raw  = VTCM_LAYOUT_PTR(uint8_t, base, L.off_act_raw);

    mmctx->vtcm_src0_stride    = 0;
    mmctx->vtcm_act_stride     = act_row_size;
    mmctx->vtcm_act_raw_stride = hex_round_up(act->ne[0] * ଆକାର(ଦଶମିକ), QK_Q8_0_TILED * ଆକାର(ଦଶମିକ));

    mmctx->vtcm_src0_size_per_thread = fastdiv(L.src0_bytes, &octx->n_threads_div);
    mmctx->vtcm_act_size_per_thread  = L.act_bytes;
    mmctx->vtcm_dst_size_per_thread  = fastdiv(L.dst_bytes, &octx->n_threads_div);

    mmctx->cur_m_start = 0;
    mmctx->cur_m_rows  = act_nrows;

    FARF(ଉପର_2, "matmul-id-nx: src0 %d:%d:%d type %s nrows %u, act %d:%d:%d nrows %u, vtcm %zu/%zu, threads %d\n",
         src0->ne[0], src0->ne[1], src0->ne[2], mmctx->type, src0->ne[1],
         act->ne[0], act->ne[1], act->ne[2], act_nrows,
         L.total_bytes, octx->ctx->vtcm_size, octx->n_threads);

    htp_trace_event_stop(tr, HTP_TRACE_EVT_INIT, 0);

    hvx_mm_transfer_act_dma(octx, kparams, act, mmctx->vtcm_act_raw, mmctx->vtcm_act_raw_stride, 0, act_nrows);

    mmctx->n_quant_rows_per_thread = (act_nrows + n_quant_tasks - 1) / n_quant_tasks;
    mmctx->n_quant_tasks           = n_quant_tasks;
    work_queue_run(octx->ctx->work_queue, quant_task_func, mmctx, n_quant_tasks);

    work_queue_run(octx->ctx->work_queue, hvx_mmid_task_func, mmctx, octx->n_threads);

    ଫେରନ୍ତୁ HTP_STATUS_OK;
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ scan_expert_ids_n(
    ଅଚଳ ସଂରଚନା htp_tensor * ids,
    ଅଚଳ uint32_t n_ids,
    uint32_t n_as,
    uint32_t * counts,
    ସଂରଚନା mmid_row_mapping * matrix_rows,
    uint32_t mapping_stride
) {
    ଅଚଳ size_t ids_nb1 = ids->nb[1];
    ଅଚଳ uint8_t * ids_data = (ଅଚଳ uint8_t *) ids->data;

    ପାଇଁ (uint32_t iid1 = 0; iid1 < ids->ne[1]; ++iid1) {
        ଅଚଳ int32_t * row_ptr = (ଅଚଳ int32_t *) (ids_data + iid1 * ids_nb1);
        ପାଇଁ (uint32_t id = 0; id < n_ids; ++id) {
            ଅଚଳ int32_t i02 = row_ptr[id];
            ଯଦି (i02 < 0) {
                ଜାରି;
            }
            ପ୍ରତିଜ୍ଞା(i02 < n_as);

            ଯଦି (matrix_rows) {
                matrix_rows[i02 * mapping_stride + counts[i02]] = (ସଂରଚନା mmid_row_mapping) { id, iid1 };
            }
            counts[i02] += 1;
        }
    }
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ scan_expert_ids(
    ଅଚଳ ସଂରଚନା htp_tensor * ids,
    uint32_t n_ids,
    uint32_t n_as,
    uint32_t * counts,
    ସଂରଚନା mmid_row_mapping * matrix_rows,
    uint32_t mapping_stride
) {
    ଅଚଳ size_t ids_nb0 = ids->nb[0];

    ଯଦି (ids_nb0 == 4) {
        ବାଛ (n_ids) {
            ସ୍ଥିତି 8:  scan_expert_ids_n(ids, 8,     n_as, counts, matrix_rows, mapping_stride); ଭାଙ୍ଗ;
            ସ୍ଥିତି 4:  scan_expert_ids_n(ids, 4,     n_as, counts, matrix_rows, mapping_stride); ଭାଙ୍ଗ;
            ସ୍ଥିତି 2:  scan_expert_ids_n(ids, 2,     n_as, counts, matrix_rows, mapping_stride); ଭାଙ୍ଗ;
            ପୂର୍ବନିର୍ଦ୍ଧାରିତ: scan_expert_ids_n(ids, n_ids, n_as, counts, matrix_rows, mapping_stride); ଭାଙ୍ଗ;
        }
    } ଅନ୍ୟଥା {
        // Strided fallback
        ଅଚଳ size_t ids_nb1 = ids->nb[1];
        ଅଚଳ uint8_t * ids_data = (ଅଚଳ uint8_t *) ids->data;
        ପାଇଁ (uint32_t iid1 = 0; iid1 < ids->ne[1]; ++iid1) {
            ଅଚଳ int32_t * row_ptr = (ଅଚଳ int32_t *) (ids_data + iid1 * ids_nb1);
            ପାଇଁ (uint32_t id = 0; id < n_ids; ++id) {
                ଅଚଳ int32_t i02 = *(ଅଚଳ int32_t *) ((ଅଚଳ uint8_t *) row_ptr + id * ids_nb0);
                ଯଦି (i02 < 0) {
                    ଜାରି;
                }
                ପ୍ରତିଜ୍ଞା(i02 < n_as);

                ଯଦି (matrix_rows) {
                    matrix_rows[i02 * mapping_stride + counts[i02]] = (ସଂରଚନା mmid_row_mapping) { id, iid1 };
                }
                counts[i02] += 1;
            }
        }
    }
}

ପୂର୍ଣ୍ଣସଂଖ୍ୟା op_matmul_id(ସଂରଚନା htp_ops_context * octx) {
    htp_matmul_tensors_preamble;

    ଅଚଳ ସଂରଚନା htp_mm_kernel_params * kparams = (ଅଚଳ ସଂରଚନା htp_mm_kernel_params *) octx->kernel_params;
    ସଂରଚନା htp_mm_context mmctx_struct = {0};
    ସଂରଚନା htp_mm_context * mmctx = &mmctx_struct;

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା status = htp_mm_init_context(octx, kparams);
    ଯଦି (status != HTP_STATUS_OK) {
        ଫେରନ୍ତୁ status;
    }

    ସଂରଚନା htp_thread_trace * tr = &octx->ctx->trace[0];
    htp_trace_event_start(tr, HTP_TRACE_EVT_INIT, 0);

    mmctx->octx = octx;
    mmctx->act = act;

    ଅଚଳ ସଂରଚନା htp_tensor * restrict ids = octx->src[2];
    ଯଦି (htp_tensor_is_extended(ids) || htp_tensor_is_extended(act) || htp_tensor_is_extended(dst)) {
        ଫେରନ୍ତୁ HTP_STATUS_NO_SUPPORT;
    }

    ଅଚଳ size_t src0_row_size = nb01;
    ଅଚଳ size_t dst_row_size  = nb1;

    ଅଚଳ size_t src0_row_size_padded = hex_round_up(src0_row_size, 128);

    ଅଚଳ uint32_t src0_nrows = ne01;  // per expert
    ଅଚଳ uint32_t act_nrows  = ne11 * ne12 * ne13;

    // row groups
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା n_ids = ids->ne[0];  // n_expert_used
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା n_as  = ne02;        // n_expert

    uint8_t  * mapping_buf       = octx->ctx->ddr_spad_base;
    uint32_t   mapping_stride    = 1;
    uint32_t * matrix_row_counts = (uint32_t *) mapping_buf;
    ସଂରଚନା mmid_row_mapping * matrix_rows = ଖାଲି;

    ଯଦି (act_nrows > 1) {
        ଅଚଳ size_t matrix_row_counts_size = n_as * ଆକାର(uint32_t);
        ପ୍ରତିଜ୍ଞା(octx->ctx->ddr_spad_size >= matrix_row_counts_size);

        hex_l2fetch_block((ଅଚଳ ଶୂନ୍ୟ *) ids->data, ids->ne[1] * ids->nb[1]);

        memset(matrix_row_counts, 0, matrix_row_counts_size);
        scan_expert_ids(ids, n_ids, n_as, matrix_row_counts, ଖାଲି, 0);

        uint32_t max_ଗଣନା = hvx_reduce_max_i32((ଅଚଳ uint8_t *) matrix_row_counts, n_as);
        mapping_stride = max_ଗଣନା > 0 ? max_ଗଣନା : 1;

        size_t matrix_row_map_size  = n_as * mapping_stride * ଆକାର(ସଂରଚନା mmid_row_mapping);
        ଅଚଳ size_t total_map_size = matrix_row_counts_size + matrix_row_map_size;

        ଯଦି (total_map_size > octx->ctx->ddr_spad_size) {
            mapping_buf = memalign(128, total_map_size);
            ଯଦି (!mapping_buf) {
                ଫେରନ୍ତୁ HTP_STATUS_INTERNAL_ERR;
            }
        }

        matrix_row_counts = (uint32_t *) mapping_buf;
        matrix_rows       = (ସଂରଚନା mmid_row_mapping *) (mapping_buf + matrix_row_counts_size);

        memset(matrix_row_counts, 0, n_as * ଆକାର(uint32_t));
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

    ପୂର୍ଣ୍ଣସଂଖ୍ୟା s;
    ଯଦି (kparams->n_hmx) {
        s = hmx_mm_op_matmul_id(octx, mmctx);
    } ଅନ୍ୟଥା {
        uint32_t src0_row_start = 0;
        uint32_t src0_row_end   = src0_nrows;
        ଯଦି (octx->ctx->mdev.ଗଣନା_2 > 1) {
            ଅଚଳ bool can_split = htp_tensor_can_row_partition(dst, ଆକାର(ଦଶମିକ));
            ଅଚଳ ସଂରଚନା htp_tensor_mdev_range range = htp_tensor_mdev_partition(src0_nrows, can_split ? 32 : 0, octx->ctx->mdev.idx, octx->ctx->mdev.ଗଣନା_2, &octx->ctx->mdev.ଗଣନା_div);
            src0_row_start = range.start;
            src0_row_end   = range.start + range.ଗଣନା_2;
        }

        ଯଦି (src0_row_start >= src0_row_end) {
            ଯଦି (mapping_buf != octx->ctx->ddr_spad_base) {
                ମୁକ୍ତ(mapping_buf);
            }
            ଫେରନ୍ତୁ HTP_STATUS_OK;
        }

        ଅଚଳ uint32_t nrows = src0_row_end - src0_row_start;
        mmctx->src0_row_start = src0_row_start;
        mmctx->src0_row_end   = src0_row_end;

        mmctx->src0_nrows_per_thread = fastdiv(nrows + octx->n_threads - 1, &octx->n_threads_div);
        mmctx->src0_nrows_per_thread = hex_round_up(mmctx->src0_nrows_per_thread, 32);

        ଯଦି (hvx_mm_init_tiled_vec_dot(mmctx, src0->type) == 0) {
            s = hvx_mm_matmul_id(octx, mmctx, act_nrows > 1 ? hvx_mm_id_tiled : hvx_mv_id_tiled);
        } ଅନ୍ୟଥା {
            s = HTP_STATUS_NO_SUPPORT;
        }
    }

    ଯଦି (mapping_buf != octx->ctx->ddr_spad_base) {
        ମୁକ୍ତ(mapping_buf);
    }

    ଫେରନ୍ତୁ s;
}

ପୂର୍ଣ୍ଣସଂଖ୍ୟା op_matmul_id_nx(ସଂରଚନା htp_ops_context * octx) {
    ଅଚଳ ସଂରଚନା htp_mm_kernel_params * kparams = (ଅଚଳ ସଂରଚନା htp_mm_kernel_params *) octx->kernel_params;
    ସଂରଚନା htp_mm_context mmctx_struct = {0};
    ସଂରଚନା htp_mm_context * mmctx = &mmctx_struct;

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା status = htp_mm_init_context(octx, kparams);
    ଯଦି (status != HTP_STATUS_OK) {
        ଫେରନ୍ତୁ status;
    }

    ସଂରଚନା htp_thread_trace * tr = &octx->ctx->trace[0];
    htp_trace_event_start(tr, HTP_TRACE_EVT_INIT, 0);

    mmctx->octx = octx;
    ଅଚଳ uint32_t n_weights = kparams->n_weights;
    ଅଚଳ ସଂରଚନା htp_tensor * restrict src0 = octx->src[0];
    ଅଚଳ ସଂରଚନା htp_tensor * restrict act  = octx->src[n_weights];
    ଅଚଳ ସଂରଚନା htp_tensor * restrict ids  = octx->src[n_weights + 1];
    ଯଦି (htp_tensor_is_extended(ids) || htp_tensor_is_extended(act)) {
        ଫେରନ୍ତୁ HTP_STATUS_NO_SUPPORT;
    }
    ପାଇଁ (uint32_t p = 0; p < n_weights; p++) {
        ଯଦି (htp_tensor_is_extended(octx->dsts[p])) {
            ଫେରନ୍ତୁ HTP_STATUS_NO_SUPPORT;
        }
    }

    mmctx->act = act;

    ଅଚଳ size_t src0_row_size = src0->nb[1];
    ଅଚଳ size_t src0_row_size_padded = hex_round_up(src0_row_size, 128);

    ଅଚଳ uint32_t act_nrows = act->ne[1] * act->ne[2] * act->ne[3];

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା n_ids = ids->ne[0];
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା n_as  = src0->ne[2];

    uint8_t  * mapping_buf       = octx->ctx->ddr_spad_base;
    uint32_t   mapping_stride    = 1;
    uint32_t * matrix_row_counts = (uint32_t *) mapping_buf;
    ସଂରଚନା mmid_row_mapping * matrix_rows = ଖାଲି;

    ଯଦି (act_nrows > 1) {
        ଅଚଳ size_t matrix_row_counts_size = n_as * ଆକାର(uint32_t);
        ପ୍ରତିଜ୍ଞା(octx->ctx->ddr_spad_size >= matrix_row_counts_size);

        hex_l2fetch_block((ଅଚଳ ଶୂନ୍ୟ *) ids->data, ids->ne[1] * ids->nb[1]);

        memset(matrix_row_counts, 0, matrix_row_counts_size);
        scan_expert_ids(ids, n_ids, n_as, matrix_row_counts, ଖାଲି, 0);

        uint32_t max_ଗଣନା = hvx_reduce_max_i32((ଅଚଳ uint8_t *) matrix_row_counts, n_as);
        mapping_stride = max_ଗଣନା > 0 ? max_ଗଣନା : 1;

        size_t matrix_row_map_size  = n_as * mapping_stride * ଆକାର(ସଂରଚନା mmid_row_mapping);
        ଅଚଳ size_t total_map_size = matrix_row_counts_size + matrix_row_map_size;

        ଯଦି (total_map_size > octx->ctx->ddr_spad_size) {
            mapping_buf = memalign(128, total_map_size);
            ଯଦି (!mapping_buf) {
                ଫେରନ୍ତୁ HTP_STATUS_INTERNAL_ERR;
            }
        }

        matrix_row_counts = (uint32_t *) mapping_buf;
        matrix_rows       = (ସଂରଚନା mmid_row_mapping *) (mapping_buf + matrix_row_counts_size);

        memset(matrix_row_counts, 0, n_as * ଆକାର(uint32_t));
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

    ପୂର୍ଣ୍ଣସଂଖ୍ୟା s;
    ଯଦି (kparams->n_hmx) {
        s = hmx_mm_op_matmul_id_nx(octx, mmctx);
    } ଅନ୍ୟଥା {
        ଯଦି (hvx_mm_init_tiled_vec_dot(mmctx, src0->type) == 0) {
            s = hvx_mm_matmul_id_nx(octx, mmctx, act_nrows > 1 ? hvx_mm_id_nx_tiled : hvx_mv_id_nx_tiled);
        } ଅନ୍ୟଥା {
            s = HTP_STATUS_NO_SUPPORT;
        }
    }

    ଯଦି (mapping_buf != octx->ctx->ddr_spad_base) {
        ମୁକ୍ତ(mapping_buf);
    }

    ଫେରନ୍ତୁ s;
}
ସ୍ଥିର ପୂର୍ଣ୍ଣସଂଖ୍ୟା op_matmul_nx_impl(ସଂରଚନା htp_ops_context * octx) {
    ଅଚଳ ସଂରଚନା htp_mm_kernel_params * kparams = (ଅଚଳ ସଂରଚନା htp_mm_kernel_params *) octx->kernel_params;

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା status = htp_mm_init_context(octx, kparams);
    ଯଦି (status != HTP_STATUS_OK) {
        ଫେରନ୍ତୁ status;
    }

    ଯଦି (kparams->n_hmx) {
        ଫେରନ୍ତୁ hmx_mm_nx_2d_f32(octx, kparams);
    }

    ସଂରଚନା htp_thread_trace * tr = &octx->ctx->trace[0];
    htp_trace_event_start(tr, HTP_TRACE_EVT_INIT, 0);

    ଅଚଳ uint32_t n_weights = kparams->n_weights;

    ଅଚଳ ସଂରଚନା htp_tensor * restrict src0 = octx->src[0]; // first weight
    ଅଚଳ ସଂରଚନା htp_tensor * restrict act  = octx->src[n_weights]; // activation x

    ସଂରଚନା htp_mm_context mmctx_struct = {0};
    ସଂରଚନା htp_mm_context * mmctx = &mmctx_struct;
    mmctx->octx = octx;
    mmctx->act  = act;

    ଅଚଳ uint32_t act_nrows = act->ne[1] * act->ne[2] * act->ne[3];
    mmctx->act_nrows   = act_nrows;
    mmctx->cur_m_start = 0;
    mmctx->cur_m_rows  = act_nrows;

    ଅଚଳ size_t src0_row_size = src0->nb[1];

    ଯଦି (hvx_mm_init_tiled_vec_dot(mmctx, src0->type) != 0) {
        ଫେରନ୍ତୁ HTP_STATUS_NO_SUPPORT;
    }

    ଅଚଳ uint32_t qk = QK_Q8_0_TILED;
    ଅଚଳ uint32_t nb = (act->ne[0] + qk - 1) / qk;
    ଅଚଳ uint32_t total_nb = act_nrows * nb;

    worker_callback_t quant_task_func;
    uint32_t n_quant_tasks = 1;
    ଯଦି (act_nrows < octx->n_threads) {
        n_quant_tasks = MIN(total_nb, octx->n_threads);
        quant_task_func = htp_mm_act_quant_block_func(src0->type, act->type);
        ପାଇଁ (uint32_t ith = 0; ith < n_quant_tasks; ++ith) {
            uint32_t ib_first = (total_nb * ith) / n_quant_tasks;
            uint32_t ib_last  = (total_nb * (ith + 1)) / n_quant_tasks;
            mmctx->quant_ib_first[ith] = ib_first;
            mmctx->quant_ib_last[ith]  = ib_last;
            mmctx->quant_r[ith]        = ib_first / nb;
            mmctx->quant_c[ith]        = ib_first % nb;
        }
    } ଅନ୍ୟଥା {
        n_quant_tasks = MIN(act_nrows, octx->n_threads);
        quant_task_func = htp_mm_act_quant_row_func(src0->type, act->type);
    }

    ଅଚଳ size_t act_row_size = htp_mm_tiled_act_row_size(src0->type, act->ne[0]);

    ସଂରଚନା htp_mm_hvx_vtcm_layout L;
    htp_mm_hvx_vtcm_layout_build(&L, kparams->kernel_type, src0->type, act->ne[0], act_nrows, octx->n_threads,
                                 0, src0_row_size, act_row_size, 0, kparams->n_prefetch, false, true);

    ଅଚଳ size_t vtcm_size = L.total_bytes;

    ଯଦି (octx->ctx->vtcm_size < vtcm_size) {
        FARF(ERROR, "matmul-nx: current VTCM reservation %zu is too small, needed %zu\n",
             octx->ctx->vtcm_size, vtcm_size);
        ଫେରନ୍ତୁ HTP_STATUS_VTCM_TOO_SMALL;
    }

    uint8_t * ଅଚଳ base = (uint8_t *) octx->ctx->vtcm_base;
    mmctx->vtcm_src0     = VTCM_LAYOUT_PTR(uint8_t, base, L.off_src0);
    mmctx->vtcm_act      = VTCM_LAYOUT_PTR(uint8_t, base, L.off_act);
    mmctx->vtcm_dst      = VTCM_LAYOUT_PTR(uint8_t, base, L.off_dst);
    mmctx->vtcm_act_raw  = VTCM_LAYOUT_PTR(uint8_t, base, L.off_act_raw);

    mmctx->vtcm_src0_stride    = 0;
    mmctx->vtcm_act_stride     = act_row_size;
    mmctx->vtcm_act_raw_stride = hex_round_up(act->ne[0] * ଆକାର(ଦଶମିକ), QK_Q8_0_TILED * ଆକାର(ଦଶମିକ));

    mmctx->vtcm_src0_size_per_thread = fastdiv(L.src0_bytes, &octx->n_threads_div);
    mmctx->vtcm_act_size_per_thread  = L.act_bytes;
    mmctx->vtcm_dst_size_per_thread  = fastdiv(L.dst_bytes, &octx->n_threads_div);

    htp_trace_event_stop(tr, HTP_TRACE_EVT_INIT, 0);

    hvx_mm_transfer_act_dma(octx, kparams, act, mmctx->vtcm_act_raw, mmctx->vtcm_act_raw_stride, 0, act_nrows);

    mmctx->n_quant_rows_per_thread = (act_nrows + n_quant_tasks - 1) / n_quant_tasks;
    mmctx->n_quant_tasks = n_quant_tasks;
    work_queue_run(octx->ctx->work_queue, quant_task_func, mmctx, n_quant_tasks);

    work_queue_run(octx->ctx->work_queue, hvx_mm_nx_2d_tiled, mmctx, octx->n_threads);

    ଫେରନ୍ତୁ HTP_STATUS_OK;
}

ପୂର୍ଣ୍ଣସଂଖ୍ୟା op_matmul_nx(ସଂରଚନା htp_ops_context * octx) {
    ଅଚଳ ସଂରଚନା htp_mm_kernel_params * kparams = (ଅଚଳ ସଂରଚନା htp_mm_kernel_params *) octx->kernel_params;

    ଯଦି (kparams->collapse) {
        ଅଚଳ uint32_t n_weights = kparams->n_weights;
        ଅଚଳ ସଂରଚନା htp_tensor * act = octx->src[n_weights];
        ଅଚଳ uint32_t s1 = (act->ne[1] > 1) ? act->nb[1] : ((act->ne[2] > 1) ? act->nb[2] : act->nb[3]);
        ସଂରଚନା htp_tensor act_collapsed;
        ସଂରଚନା htp_tensor dsts_collapsed[HTP_OP_MAX_OUTPUTS];
        ଅଚଳ ସଂରଚନା htp_tensor * orig_dsts[HTP_OP_MAX_OUTPUTS];

        htp_mm_tensor_collapse_rows(&act_collapsed, act, s1);
        octx->src[n_weights] = &act_collapsed;

        ପାଇଁ (uint32_t p = 0; p < n_weights; p++) {
            orig_dsts[p] = octx->dsts[p];
            ଅଚଳ ସଂରଚନା htp_tensor * d = octx->dsts[p];
            ଅଚଳ uint32_t sd = (d->ne[1] > 1) ? d->nb[1] : ((d->ne[2] > 1) ? d->nb[2] : d->nb[3]);
            htp_mm_tensor_collapse_rows(&dsts_collapsed[p], d, sd);
            octx->dsts[p] = &dsts_collapsed[p];
        }

        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା status = op_matmul_nx_impl(octx);

        octx->src[n_weights] = act;
        ପାଇଁ (uint32_t p = 0; p < n_weights; p++) {
            octx->dsts[p] = orig_dsts[p];
        }
        ଫେରନ୍ତୁ status;
    }

    ଫେରନ୍ତୁ op_matmul_nx_impl(octx);
}
