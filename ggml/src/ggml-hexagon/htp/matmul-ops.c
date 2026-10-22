#निर्देशः clang diagnostic ignored "-Wgnu-zero-variadic-macro-arguments"
#निर्देशः clang diagnostic ignored "-Wunused-function"
#निर्देशः clang diagnostic ignored "-Wunused-variable"
#निर्देशः clang diagnostic ignored "-Wunused-but-set-variable"

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

#परिभाषय GGML_COMMON_DECL_C
#अन्तर्भाव "ggml-common.h"
#अन्तर्भाव "htp-ctx.h"
#अन्तर्भाव "htp-ops.h"
#अन्तर्भाव "htp-tensor.h"
#अन्तर्भाव "matmul-ops.h"
#अन्तर्भाव "htp-vtcm.h"

प्रकारनिर्देश संरचना {
    भिन्न        *dst;
    dma_addr_t    bias_addr;
    size_t        bias_अष्टकानि;
    dma_addr_t    act_dma_addr;
    dma_addr_t    weight;
    dma_queue *   weight_dma;
    पूर्णाङ्क           m;
    पूर्णाङ्क           k;
    पूर्णाङ्क           n;
    पूर्णाङ्क           act_stride;
    uint32_t      act_elem_आकार;  // 4=F32 src1, 2=F16 src1
    पूर्णाङ्क           weight_stride;
    पूर्णाङ्क           weight_type;
    पूर्णाङ्क           dst_stride;
    पूर्णाङ्क           dst_स्तम्भसङ्ख्या;
    uint32_t      bias_stride;
    पूर्णाङ्क           tile_आकार;
    पूर्णाङ्क           aligned_tile_आकार;
    पूर्णाङ्क           ne02;
    पूर्णाङ्क           ne03;
    पूर्णाङ्क           ne12;
    पूर्णाङ्क           ne13;
    size_t        src0_nb2;
    size_t        src0_nb3;
    size_t        act_nb2;
    size_t        act_nb3;
    size_t        bias_nb2;
    size_t        bias_nb3;
    size_t        dst_nb2;
    size_t        dst_nb3;
    पूर्णाङ्क           r2;
    पूर्णाङ्क           r3;
    संरचना fastdiv_मानानि div_r2;
    संरचना fastdiv_मानानि div_r3;
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

    शून्य (*vec_dot_1x1)(ध्रुव uint32_t n, भिन्न * सीमित s0,
         ध्रुव शून्य * सीमित vx0,
         ध्रुव शून्य * सीमित vy0);

    शून्य (*vec_dot_2x1)(ध्रुव uint32_t n, भिन्न * सीमित s0,
         ध्रुव शून्य * सीमित vx0, ध्रुव शून्य * सीमित vx1,
         ध्रुव शून्य * सीमित vy0);

    शून्य (*vec_dot_2x2)(ध्रुव uint32_t n, भिन्न * सीमित s0, भिन्न * सीमित s1,
         ध्रुव शून्य * सीमित vx0, ध्रुव शून्य * सीमित vx1,
         ध्रुव शून्य * सीमित vy0, ध्रुव शून्य * सीमित vy1);

    शून्य (*vec_dot_32x1)(ध्रुव uint32_t n, भिन्न * सीमित s,
         ध्रुव शून्य * सीमित vx,
         ध्रुव शून्य * सीमित vy, uint32_t valid_पङ्क्तिकाः,
         ध्रुव भिन्न * सीमित sz);

    शून्य (*vec_dot_32x2)(ध्रुव uint32_t n, भिन्न * सीमित s0, भिन्न * सीमित s1,
         ध्रुव शून्य * सीमित vx,
         ध्रुव शून्य * सीमित vy0, ध्रुव शून्य * सीमित vy1, uint32_t valid_पङ्क्तिकाः,
         ध्रुव भिन्न * सीमित sz0, ध्रुव भिन्न * सीमित sz1);

    // Precomputed values
    uint32_t src0_nrows_प्रत्येकम्_thread;
    uint32_t src0_row_start;
    uint32_t src0_row_अन्तः;
    uint32_t src0_row_आकार_padded;
    uint32_t act_nrows;
    uint32_t cur_m_start;
    uint32_t cur_m_पङ्क्तिकाः;

    संरचना fastdiv_मानानि mm_div_ne12_ne1;
    संरचना fastdiv_मानानि mm_div_ne1;
    संरचना fastdiv_मानानि mm_div_r2;
    संरचना fastdiv_मानानि mm_div_r3;
    संरचना fastdiv_मानानि mm_div_ne11;

    // Precomputed block-parallel quantization values
    uint32_t          quant_ib_first[WORK_QUEUE_MAX_N_तन्तवः];
    uint32_t          quant_ib_चरमः[WORK_QUEUE_MAX_N_तन्तवः];
    uint32_t          quant_r[WORK_QUEUE_MAX_N_तन्तवः];
    uint32_t          quant_c[WORK_QUEUE_MAX_N_तन्तवः];
    uint32_t          n_quant_tasks;
    uint32_t          n_quant_पङ्क्तिकाः_प्रत्येकम्_thread;

    // Fields for scattered mapping & HMX support in MUL_MAT_ID
    ध्रुव uint32_t * matrix_row_counts;
    ध्रुव संरचना mmid_row_mapping * matrix_पङ्क्तिकाः;
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
    uint32_t vtcm_src0_आकार_प्रत्येकम्_thread;
    uint32_t vtcm_act_आकार_प्रत्येकम्_thread;
    uint32_t vtcm_dst_आकार_प्रत्येकम्_thread;
};

स्थिर पूर्णाङ्क htp_mm_init_context(
    संरचना htp_ops_context * octx,
    ध्रुव संरचना htp_mm_kernel_params * kparams
) {
    यदि (!htp_ops_context_समुच्चयः_n_तन्तवः(octx, (uint32_t) kparams->n_तन्तवः)) {
        प्रत्यागम HTP_स्थितिः_INVAL_PARAMS;
    }

    यदि (kparams->n_hmx) {
        यदि (kparams->n_act_तन्तवः <= 0 || kparams->n_act_तन्तवः > (int32_t) octx->n_तन्तवः) {
            प्रत्यागम HTP_स्थितिः_INVAL_PARAMS;
        }
    }

    प्रत्यागम HTP_स्थितिः_OK;
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

#परिभाषय htp_matmul_tensors_preamble                                 \
    ध्रुव संरचना htp_tensor * सीमित src0 = octx->src[0];         \
    ध्रुव संरचना htp_tensor * सीमित src1 = octx->src[1];         \
    ध्रुव संरचना htp_tensor * सीमित src2 = octx->src[2];         \
    ध्रुव संरचना htp_tensor * सीमित  dst = octx->dst;            \
    ध्रुव संरचना htp_tensor * सीमित  act = src1;                 \
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

#परिभाषय htp_matmul_preamble                                         \
    संरचना htp_mm_context * mmctx  = दत्त;                          \
    संरचना htp_ops_context * octx  = mmctx->octx;                   \
    dma_queue * dma_q              = octx->ctx->dma[ith];           \
    uint32_t src0_nrows_प्रत्येकम्_thread = mmctx->src0_nrows_प्रत्येकम्_thread;  \
    htp_matmul_tensors_preamble;




// hvx kernels first: the HMX Q6_K dequantizer reuses unpack_q6_k_group from there
#अन्तर्भाव "hvx-mm-kernels-tiled.h"
#अन्तर्भाव "hvx-mm-kernels-float.h"
#अन्तर्भाव "hmx-mm-kernels-tiled.h"

स्थिर शून्य hvx_mm_2d_tiled(अचिह्नित पूर्णाङ्क nth, अचिह्नित पूर्णाङ्क ith, शून्य * दत्त) {
    htp_matmul_preamble;

    ध्रुव uint32_t src0_nrows = mmctx->src0_row_अन्तः - mmctx->src0_row_start;
    ध्रुव uint32_t act_nrows = mmctx->cur_m_पङ्क्तिकाः ? mmctx->cur_m_पङ्क्तिकाः : (ne11 * ne12 * ne13);
    ध्रुव uint32_t cur_m_start = mmctx->cur_m_start;

    ध्रुव uint32_t src0_start_row  = mmctx->src0_row_start + src0_nrows_प्रत्येकम्_thread * ith;
    ध्रुव uint32_t src0_अन्तः_row    = MIN(src0_start_row + src0_nrows_प्रत्येकम्_thread, mmctx->src0_row_अन्तः);

    संरचना htp_thread_trace * tr = &octx->ctx->trace[ith];

    ध्रुव संरचना htp_mm_kernel_params * kparams = (ध्रुव संरचना htp_mm_kernel_params *) octx->kernel_params;
    ध्रुव uint32_t n_prefetch = kparams->n_prefetch;
    प्रतिज्ञा(n_prefetch >= 2 && n_prefetch <= HTP_MM_MAX_PREFETCH && (n_prefetch & (n_prefetch - 1)) == 0);

    ध्रुव size_t dst_row_आकार  = nb1;
    ध्रुव size_t act_row_आकार = nb11;
    ध्रुव size_t act_stride = mmctx->vtcm_act_stride;
    ध्रुव size_t src2_stride = src2 ? ((src2->ne[1] == 1) ? 0 : src2->nb[1]) : 0;

    uint8_t * सीमित vtcm_dst_ptr  = mmctx->vtcm_dst  + mmctx->vtcm_dst_आकार_प्रत्येकम्_thread  * ith;
    uint8_t * सीमित vtcm_src0_ptr = mmctx->vtcm_src0 + mmctx->vtcm_src0_आकार_प्रत्येकम्_thread * ith;
    uint8_t * सीमित act_दत्त = mmctx->vtcm_act;

    ध्रुव dma_addr_t src0_row = src0->दत्त;

    ध्रुव uint32_t tile_आकार = htp_tensor_get_tile_आकार(src0->type);
    ध्रुव uint32_t aligned_tile_आकार = htp_tensor_get_tile_padded_आकार(src0->type);

    uint32_t n_k_tiles_w = ne00 / htp_tensor_get_tile_k(src0->type);
    uint32_t tile_row_stride = n_k_tiles_w * tile_आकार;
    uint32_t tile_row_transfer_आकार_aligned = n_k_tiles_w * aligned_tile_आकार;

    uint32_t ct_start = src0_start_row / 32;
    uint32_t ct_अन्तः   = (src0_अन्तः_row + 31) / 32;

    uint32_t प्रक्षिप_ct = ct_start;
    यदि (src0_start_row < src0_अन्तः_row) {
        प्रति (uint32_t d = 0; d < n_prefetch && प्रक्षिप_ct < ct_अन्तः; d++, प्रक्षिप_ct++) {
            dma_queue_प्रक्षिप(dma_q, dma_make_दत्त(vtcm_src0_ptr + d * tile_row_transfer_आकार_aligned,
                           src0_row + प्रक्षिप_ct * tile_row_stride), aligned_tile_आकार, tile_आकार, tile_आकार, n_k_tiles_w);
        }
    }

    यदि (src0_start_row >= src0_अन्तः_row) {
        प्रत्यागम;
    }

    प्रति (uint32_t ct = ct_start; ct < ct_अन्तः; ct++) {
        ध्रुव uint8_t * w_tile = (शून्य *) dma_queue_निष्कासय(dma_q).dst;

        पूर्णाङ्क valid_पङ्क्तिकाः = (पूर्णाङ्क)ne0 - (पूर्णाङ्क)(ct * 32);
        valid_पङ्क्तिकाः = MIN(32, MAX(0, valid_पङ्क्तिकाः));

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, ct);
        uint32_t ir1 = 0;
        प्रति (; ir1 + 1 < act_nrows; ir1 += 2) {
            ध्रुव uint8_t * सीमित act_col0 = (ध्रुव uint8_t *) (act_दत्त + (ir1+0) * act_stride);
            ध्रुव uint8_t * सीमित act_col1 = (ध्रुव uint8_t *) (act_दत्त + (ir1+1) * act_stride);
            भिन्न * सीमित dst_row0 = (भिन्न *) (dst->दत्त + ((cur_m_start + ir1+0) * dst_row_आकार));
            भिन्न * सीमित dst_row1 = (भिन्न *) (dst->दत्त + ((cur_m_start + ir1+1) * dst_row_आकार));

            भिन्न * dst_ptr0 = &dst_row0[ct * 32];
            भिन्न * dst_ptr1 = &dst_row1[ct * 32];

            ध्रुव भिन्न * src2_ptr0 = रिक्त;
            ध्रुव भिन्न * src2_ptr1 = रिक्त;
            यदि (src2) {
                ध्रुव भिन्न * सीमित src2_row0 = (ध्रुव भिन्न *) ((ध्रुव uint8_t *) src2->दत्त + ((cur_m_start + ir1+0) * src2_stride));
                ध्रुव भिन्न * सीमित src2_row1 = (ध्रुव भिन्न *) ((ध्रुव uint8_t *) src2->दत्त + ((cur_m_start + ir1+1) * src2_stride));
                src2_ptr0 = &src2_row0[ct * 32];
                src2_ptr1 = &src2_row1[ct * 32];
            }
            mmctx->vec_dot_32x2(ne10, dst_ptr0, dst_ptr1, w_tile, act_col0, act_col1, valid_पङ्क्तिकाः, src2_ptr0, src2_ptr1);
        }

        प्रति (; ir1 < act_nrows; ++ir1) {
            ध्रुव uint8_t * सीमित act_col = (ध्रुव uint8_t *) (act_दत्त + ir1 * act_stride);
            भिन्न * सीमित dst_row          = (भिन्न *) (dst->दत्त + ((cur_m_start + ir1) * dst_row_आकार));
            भिन्न * dst_ptr = &dst_row[ct * 32];

            ध्रुव भिन्न * src2_ptr = रिक्त;
            यदि (src2) {
                ध्रुव भिन्न * सीमित src2_row = (ध्रुव भिन्न *) ((ध्रुव uint8_t *) src2->दत्त + ((cur_m_start + ir1) * src2_stride));
                src2_ptr = &src2_row[ct * 32];
            }
            mmctx->vec_dot_32x1(ne10, dst_ptr, w_tile, act_col, valid_पङ्क्तिकाः, src2_ptr);
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ct);

        यदि (प्रक्षिप_ct < ct_अन्तः) {
            dma_queue_प्रक्षिप(dma_q, dma_make_दत्त(w_tile, src0_row + प्रक्षिप_ct * tile_row_stride),
                           aligned_tile_आकार, tile_आकार, tile_आकार, n_k_tiles_w);
            प्रक्षिप_ct++;
        }
    }
}

स्थिर शून्य hvx_mv_2d_tiled(अचिह्नित पूर्णाङ्क nth, अचिह्नित पूर्णाङ्क ith, शून्य * दत्त) {
    htp_matmul_preamble;

    ध्रुव uint32_t src0_nrows = mmctx->src0_row_अन्तः - mmctx->src0_row_start;

    ध्रुव uint32_t src0_start_row  = mmctx->src0_row_start + src0_nrows_प्रत्येकम्_thread * ith;
    ध्रुव uint32_t src0_अन्तः_row    = MIN(src0_start_row + src0_nrows_प्रत्येकम्_thread, mmctx->src0_row_अन्तः);

    संरचना htp_thread_trace * tr = &octx->ctx->trace[ith];

    ध्रुव संरचना htp_mm_kernel_params * kparams = (ध्रुव संरचना htp_mm_kernel_params *) octx->kernel_params;
    ध्रुव uint32_t n_prefetch = kparams->n_prefetch;
    प्रतिज्ञा(n_prefetch >= 2 && n_prefetch <= HTP_MM_MAX_PREFETCH && (n_prefetch & (n_prefetch - 1)) == 0);

    ध्रुव size_t dst_row_आकार  = nb1;
    ध्रुव size_t act_row_आकार = nb11;
    ध्रुव size_t act_stride = mmctx->vtcm_act_stride;

    uint8_t * vtcm_dst_ptr  = mmctx->vtcm_dst + mmctx->vtcm_dst_आकार_प्रत्येकम्_thread * ith;
    uint8_t * vtcm_src0_ptr = mmctx->vtcm_src0 + mmctx->vtcm_src0_आकार_प्रत्येकम्_thread * ith;
    uint8_t * act_दत्त = mmctx->vtcm_act;

    भिन्न * tmp = (भिन्न *) vtcm_dst_ptr;

    ध्रुव dma_addr_t src0_row = src0->दत्त;

    ध्रुव uint8_t * सीमित act_col = (ध्रुव uint8_t *) act_दत्त;
    भिन्न * सीमित dst_col          = (भिन्न *) dst->दत्त;

    ध्रुव uint32_t tile_आकार = htp_tensor_get_tile_आकार(src0->type);
    ध्रुव uint32_t aligned_tile_आकार = htp_tensor_get_tile_padded_आकार(src0->type);

    uint32_t n_k_tiles_w = ne00 / htp_tensor_get_tile_k(src0->type);
    uint32_t tile_row_stride = n_k_tiles_w * tile_आकार;
    uint32_t tile_row_transfer_आकार_aligned = n_k_tiles_w * aligned_tile_आकार;

    uint32_t ct_start = src0_start_row / 32;
    uint32_t ct_अन्तः   = (src0_अन्तः_row + 31) / 32;

    uint32_t प्रक्षिप_ct = ct_start;
    यदि (src0_start_row < src0_अन्तः_row) {
        यदि (src2) {
            भिन्न * vtcm_bias_ptr = (भिन्न *) mmctx->vtcm_bias + src0_start_row;
            ध्रुव dma_addr_t src2_addr = src2->दत्त + src0_start_row * परिमाण(भिन्न);
            पूर्णाङ्क खण्डिका_आकार = (पूर्णाङ्क)MIN(src0_अन्तः_row, ne0) - (पूर्णाङ्क)src0_start_row;
            यदि (खण्डिका_आकार > 0) {
                dma_queue_प्रक्षिप(dma_q, dma_make_दत्त(vtcm_bias_ptr, src2_addr),
                               खण्डिका_आकार * परिमाण(भिन्न), खण्डिका_आकार * परिमाण(भिन्न), खण्डिका_आकार * परिमाण(भिन्न), 1);
                dma_queue_निष्कासय_nowait(dma_q);
            }
        }
        प्रति (uint32_t d = 0; d < n_prefetch && प्रक्षिप_ct < ct_अन्तः; d++, प्रक्षिप_ct++) {
            dma_queue_प्रक्षिप(dma_q, dma_make_दत्त(vtcm_src0_ptr + d * tile_row_transfer_आकार_aligned,
                           src0_row + प्रक्षिप_ct * tile_row_stride), aligned_tile_आकार, tile_आकार, tile_आकार, n_k_tiles_w);
        }
    }

    यदि (src0_start_row >= src0_अन्तः_row) {
        प्रत्यागम;
    }

    प्रति (uint32_t ct = ct_start; ct < ct_अन्तः; ct++) {
        ध्रुव uint8_t * w_tile = (शून्य *) dma_queue_निष्कासय(dma_q).dst;

        भिन्न * dst_ptr = &tmp[ct * 32 - src0_start_row];
        पूर्णाङ्क valid_पङ्क्तिकाः = (पूर्णाङ्क)ne0 - (पूर्णाङ्क)(ct * 32);
        valid_पङ्क्तिकाः = MIN(32, MAX(0, valid_पङ्क्तिकाः));

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, ct);
        mmctx->vec_dot_32x1(ne10, dst_ptr, w_tile, act_col, valid_पङ्क्तिकाः, रिक्त);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ct);

        यदि (प्रक्षिप_ct < ct_अन्तः) {
            dma_queue_प्रक्षिप(dma_q, dma_make_दत्त(w_tile, src0_row + प्रक्षिप_ct * tile_row_stride),
                           aligned_tile_आकार, tile_आकार, tile_आकार, n_k_tiles_w);
            प्रक्षिप_ct++;
        }
    }

    पूर्णाङ्क copy_cnt = (पूर्णाङ्क)MIN(src0_अन्तः_row, ne0) - (पूर्णाङ्क)src0_start_row;
    यदि (copy_cnt > 0) {
        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, ct_अन्तः);
        यदि (src2) {
            hvx_सङ्कलय_f32_uuu((uint8_t *) &dst_col[src0_start_row],
                            (ध्रुव uint8_t *) tmp,
                            (ध्रुव uint8_t *) ((ध्रुव भिन्न *) mmctx->vtcm_bias + src0_start_row),
                            copy_cnt);
        } अन्यथा {
            hvx_copy_f32_ua((uint8_t *) &dst_col[src0_start_row], (uint8_t *) tmp, copy_cnt);
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ct_अन्तः);
    }
}

स्थिर शून्य hvx_mm_nx_2d_tiled(अचिह्नित पूर्णाङ्क nth, अचिह्नित पूर्णाङ्क ith, शून्य * दत्त) {
    संरचना htp_mm_context * mmctx = दत्त;
    संरचना htp_ops_context * octx = mmctx->octx;
    ध्रुव संरचना htp_mm_kernel_params * kparams = (ध्रुव संरचना htp_mm_kernel_params *) octx->kernel_params;
    ध्रुव uint32_t n_weights = kparams->n_weights;

    ध्रुव संरचना htp_tensor * सीमित act = octx->src[n_weights]; /* x */
    ध्रुव uint32_t ne10 = act->ne[0];
    ध्रुव uint32_t act_nrows = act->ne[1] * act->ne[2] * act->ne[3];
    ध्रुव size_t act_stride = mmctx->vtcm_act_stride;

    uint8_t * सीमित vtcm_weight_ptr = mmctx->vtcm_src0 + mmctx->vtcm_src0_आकार_प्रत्येकम्_thread * ith;
    uint8_t * सीमित act_दत्त        = mmctx->vtcm_act;

    संरचना htp_thread_trace * tr = &octx->ctx->trace[ith];
    ध्रुव uint32_t n_prefetch = kparams->n_prefetch;
    प्रतिज्ञा(n_prefetch >= 2 && n_prefetch <= HTP_MM_MAX_PREFETCH && (n_prefetch & (n_prefetch - 1)) == 0);

    प्रति (uint32_t widx = 0; widx < n_weights; widx++) {
        ध्रुव संरचना htp_tensor * सीमित src_w = octx->src[widx];
        ध्रुव संरचना htp_tensor * सीमित dst   = octx->dsts[widx];
        यदि (!src_w || !dst) अनुवर्त;
        dma_queue * dma_q = octx->ctx->dma[ith];

        ध्रुव uint32_t tile_आकार = htp_tensor_get_tile_आकार(src_w->type);
        ध्रुव uint32_t aligned_tile_आकार = htp_tensor_get_tile_padded_आकार(src_w->type);
        ध्रुव uint32_t ne00 = src_w->ne[0];
        ध्रुव uint32_t ne01 = src_w->ne[1];
        ध्रुव size_t dst_row_आकार = dst->nb[1];
        ध्रुव dma_addr_t src_w_row = src_w->दत्त;

        ध्रुव uint32_t n_k_tiles_w = ne00 / htp_tensor_get_tile_k(src_w->type);
        ध्रुव uint32_t tile_row_stride = n_k_tiles_w * tile_आकार;
        ध्रुव uint32_t tile_row_transfer_आकार_aligned = n_k_tiles_w * aligned_tile_आकार;

        uint32_t src0_start_row = 0;
        uint32_t src0_अन्तः_row   = ne01;
        यदि (octx->ctx->mdev.गणनम् > 1) {
            ध्रुव bool can_विभज = htp_tensor_can_row_partition(dst, परिमाण(भिन्न));
            ध्रुव संरचना htp_tensor_mdev_range range = htp_tensor_mdev_partition(ne01, can_विभज ? 32 : 0,
                                                         octx->ctx->mdev.idx, octx->ctx->mdev.गणनम्, &octx->ctx->mdev.गणनम्_div);
            src0_start_row = range.start;
            src0_अन्तः_row   = range.start + range.गणनम्;
        }

        ध्रुव uint32_t nrows = src0_अन्तः_row - src0_start_row;
        uint32_t src0_nrows_प्रत्येकम्_thread = fastdiv(nrows + nth - 1, &octx->n_तन्तवः_div);
        src0_nrows_प्रत्येकम्_thread = hex_आवर्तनम्_ऊर्ध्वम्(src0_nrows_प्रत्येकम्_thread, 32);

        ध्रुव uint32_t start_row = src0_start_row + src0_nrows_प्रत्येकम्_thread * ith;
        ध्रुव uint32_t अन्तः_row   = MIN(start_row + src0_nrows_प्रत्येकम्_thread, src0_अन्तः_row);
        यदि (start_row >= अन्तः_row) अनुवर्त;

        uint32_t ct_start = start_row / 32;
        uint32_t ct_अन्तः   = (अन्तः_row + 31) / 32;

        uint32_t प्रक्षिप_ct = ct_start;
        प्रति (uint32_t d = 0; d < n_prefetch && प्रक्षिप_ct < ct_अन्तः; d++, प्रक्षिप_ct++) {
            dma_queue_प्रक्षिप(dma_q, dma_make_दत्त(vtcm_weight_ptr + d * tile_row_transfer_आकार_aligned,
                            src_w_row + प्रक्षिप_ct * tile_row_stride), aligned_tile_आकार, tile_आकार, tile_आकार, n_k_tiles_w);
        }

        प्रति (uint32_t ct = ct_start; ct < ct_अन्तः; ct++) {
            ध्रुव uint8_t * w_tile = (शून्य *) dma_queue_निष्कासय(dma_q).dst;
            पूर्णाङ्क valid_पङ्क्तिकाः = (पूर्णाङ्क)dst->ne[0] - (पूर्णाङ्क)(ct * 32);
            valid_पङ्क्तिकाः = MIN(32, MAX(0, valid_पङ्क्तिकाः));

            htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, ct);
            uint32_t ir1 = 0;
            प्रति (; ir1 + 1 < act_nrows; ir1 += 2) {
                ध्रुव uint8_t * सीमित act_col0 = (ध्रुव uint8_t *) (act_दत्त + (ir1+0) * act_stride);
                ध्रुव uint8_t * सीमित act_col1 = (ध्रुव uint8_t *) (act_दत्त + (ir1+1) * act_stride);

                भिन्न * सीमित dst_row0 = (भिन्न *) (dst->दत्त + ((ir1+0) * dst_row_आकार));
                भिन्न * सीमित dst_row1 = (भिन्न *) (dst->दत्त + ((ir1+1) * dst_row_आकार));
                भिन्न * dst_ptr0 = &dst_row0[ct * 32];
                भिन्न * dst_ptr1 = &dst_row1[ct * 32];

                mmctx->vec_dot_32x2(ne10, dst_ptr0, dst_ptr1, w_tile, act_col0, act_col1, valid_पङ्क्तिकाः, रिक्त, रिक्त);
            }

            प्रति (; ir1 < act_nrows; ++ir1) {
                ध्रुव uint8_t * सीमित act_col = (ध्रुव uint8_t *) (act_दत्त + ir1 * act_stride);
                भिन्न * सीमित dst_row = (भिन्न *) (dst->दत्त + (ir1 * dst_row_आकार));
                भिन्न * dst_ptr = &dst_row[ct * 32];
                mmctx->vec_dot_32x1(ne10, dst_ptr, w_tile, act_col, valid_पङ्क्तिकाः, रिक्त);
            }
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ct);

            यदि (प्रक्षिप_ct < ct_अन्तः) {
                dma_queue_प्रक्षिप(dma_q, dma_make_दत्त(w_tile, src_w_row + प्रक्षिप_ct * tile_row_stride),
                                aligned_tile_आकार, tile_आकार, tile_आकार, n_k_tiles_w);
                प्रक्षिप_ct++;
            }
        }
    }
}

स्थिर शून्य hvx_mm_transfer_act_dma(
    संरचना htp_ops_context * octx,
    ध्रुव संरचना htp_mm_kernel_params * kparams,
    ध्रुव संरचना htp_tensor * act,
    uint8_t * dst_आधारः,
    size_t dst_row_आकार,
    uint32_t m_start,
    uint32_t m_पङ्क्तिकाः
) {
    यदि (m_पङ्क्तिकाः == 0) {
        प्रत्यागम;
    }

    dma_queue * dma_q = octx->ctx->dma[0];
    ध्रुव uint32_t ne0 = act->ne[0];
    ध्रुव size_t elem_आकार = (act->type == HTP_TYPE_F16) ? परिमाण(__fp16) : परिमाण(भिन्न);
    ध्रुव size_t row_अष्टकानि = ne0 * elem_आकार;
    ध्रुव size_t act_nb1 = act->nb[1];
    ध्रुव dma_addr_t act_आधारः = act->दत्त;

    ध्रुव bool is_contiguous = (act->nb[2] == act->ne[1] * act_nb1) &&
                               (act->nb[3] == act->ne[2] * act->nb[2]);

    यदि (is_contiguous) {
        ध्रुव dma_addr_t src_addr = act_आधारः + m_start * act_nb1;
        dma_queue_प्रक्षिप(dma_q, dma_make_दत्त(dst_आधारः, src_addr),
                       dst_row_आकार, act_nb1, row_अष्टकानि, m_पङ्क्तिकाः);
        dma_queue_निष्कासय(dma_q);
    } अन्यथा {
        ध्रुव uint32_t ne12_ne1 = act->ne[2] * act->ne[1];
        ध्रुव bool use_fastdiv = kparams->div_ne12_ne1.mp != 0;
        प्रति (uint32_t ir = 0; ir < m_पङ्क्तिकाः; ++ir) {
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
            ध्रुव dma_addr_t row_src = act_आधारः + (i11 * act->nb[1] +
                                                   i12 * act->nb[2] +
                                                   i13 * act->nb[3]);
            uint8_t * row_dst = dst_आधारः + ir * dst_row_आकार;
            dma_queue_प्रक्षिप(dma_q, dma_make_दत्त(row_dst, row_src),
                           dst_row_आकार, act_nb1, row_अष्टकानि, 1);
            dma_queue_निष्कासय(dma_q);
        }
    }

    यदि (dst_row_आकार > row_अष्टकानि) {
        संरचना htp_thread_trace * tr = &octx->ctx->trace[0];
        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_A_PREP, (uint16_t) m_start);
        ध्रुव uint32_t pad_elems = (dst_row_आकार - row_अष्टकानि) / elem_आकार;
        यदि (elem_आकार == परिमाण(भिन्न)) {
            प्रति (uint32_t ir = 0; ir < m_पङ्क्तिकाः; ++ir) {
                hvx_splat_f32_u(dst_आधारः + ir * dst_row_आकार + row_अष्टकानि, 0.0f, pad_elems);
            }
        } अन्यथा {
            प्रति (uint32_t ir = 0; ir < m_पङ्क्तिकाः; ++ir) {
                hvx_splat_f16_u(dst_आधारः + ir * dst_row_आकार + row_अष्टकानि, (_Float16) 0.0f, pad_elems);
            }
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_A_PREP, (uint16_t) m_start);
    }
}

#परिभाषय QUANTIZE_IMPL(name, log_name, kernel_fn, dst_row_आकार_व्यञ्जकम्)                                        \
स्थिर शून्य name(अचिह्नित पूर्णाङ्क nth, अचिह्नित पूर्णाङ्क ith, शून्य * दत्त) {                                        \
    (शून्य) nth;                                                                                            \
    संरचना htp_mm_context * mmctx = दत्त;                                                                  \
    संरचना htp_ops_context * octx = mmctx->octx;                                                           \
    ध्रुव संरचना htp_tensor * src = mmctx->act;                                                            \
    ध्रुव uint32_t ne0 = src->ne[0];                                                                       \
    ध्रुव uint32_t nrows = mmctx->cur_m_पङ्क्तिकाः ? mmctx->cur_m_पङ्क्तिकाः : mmctx->act_nrows;                       \
    ध्रुव uint32_t nrows_प्रत्येकम्_thread = mmctx->n_quant_पङ्क्तिकाः_प्रत्येकम्_thread;                                      \
                                                                                                           \
    ध्रुव uint32_t ir_first = nrows_प्रत्येकम्_thread * ith;                                                      \
    यदि (ir_first >= nrows) {                                                                               \
        प्रत्यागम;                                                                                            \
    }                                                                                                      \
                                                                                                           \
    संरचना htp_thread_trace * tr = &octx->ctx->trace[ith];                                                 \
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_A_QUANT, ir_first);                                        \
                                                                                                           \
    uint8_t * सीमित dst = mmctx->vtcm_act;                                                              \
    ध्रुव uint32_t ir_चरमः = MIN(ir_first + nrows_प्रत्येकम्_thread, nrows);                                      \
    ध्रुव size_t raw_row_आकार = mmctx->vtcm_act_raw_stride;                                                \
    ध्रुव size_t dst_row_आकार = (dst_row_आकार_व्यञ्जकम्);                                                       \
                                                                                                           \
    ध्रुव uint8_t * सीमित src_दत्त = (ध्रुव uint8_t *) mmctx->vtcm_act_raw + (raw_row_आकार * ir_first); \
    uint8_t * सीमित dst_दत्त = (uint8_t *) dst + (dst_row_आकार * ir_first);                             \
    kernel_fn(src_दत्त, dst_दत्त, रिक्त, ne0, ir_चरमः - ir_first, raw_row_आकार, dst_row_आकार);              \
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

स्थिर शून्य quantize_f32_q8_0_tiled_खण्डम्(अचिह्नित पूर्णाङ्क nth, अचिह्नित पूर्णाङ्क ith, शून्य * दत्त) {
    (शून्य) nth;
    संरचना htp_mm_context * mmctx = दत्त;
    यदि (mmctx->quant_ib_first[ith] >= mmctx->quant_ib_चरमः[ith]) {
        प्रत्यागम;
    }
    संरचना htp_ops_context * octx = mmctx->octx;
    संरचना htp_thread_trace * tr = &octx->ctx->trace[ith];
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_A_QUANT, mmctx->quant_ib_first[ith]);

    ध्रुव संरचना htp_tensor * src = mmctx->act;

    quantize_f32_q8_0_tiled_खण्डम्_kernel(
        (ध्रुव भिन्न *) mmctx->vtcm_act_raw,
        mmctx->vtcm_act,
        रिक्त,
        src->ne[0],
        mmctx->quant_ib_first[ith],
        mmctx->quant_ib_चरमः[ith],
        mmctx->vtcm_act_raw_stride,
        htp_mm_q8_0_tiled_row_आकार(src->ne[0]),
        mmctx->quant_r[ith],
        mmctx->quant_c[ith]
    );

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_A_QUANT, mmctx->quant_ib_first[ith]);
}

स्थिर शून्य quantize_f32_q8_1_tiled_खण्डम्(अचिह्नित पूर्णाङ्क nth, अचिह्नित पूर्णाङ्क ith, शून्य * दत्त) {
    (शून्य) nth;
    संरचना htp_mm_context * mmctx = दत्त;
    यदि (mmctx->quant_ib_first[ith] >= mmctx->quant_ib_चरमः[ith]) {
        प्रत्यागम;
    }
    संरचना htp_ops_context * octx = mmctx->octx;
    संरचना htp_thread_trace * tr = &octx->ctx->trace[ith];
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_A_QUANT, mmctx->quant_ib_first[ith]);

    ध्रुव संरचना htp_tensor * src = mmctx->act;

    quantize_f32_q8_1_tiled_खण्डम्_kernel(
        (ध्रुव भिन्न *) mmctx->vtcm_act_raw,
        mmctx->vtcm_act,
        रिक्त,
        src->ne[0],
        mmctx->quant_ib_first[ith],
        mmctx->quant_ib_चरमः[ith],
        mmctx->vtcm_act_raw_stride,
        htp_mm_q8_1_tiled_row_आकार(src->ne[0]),
        mmctx->quant_r[ith],
        mmctx->quant_c[ith]
    );

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_A_QUANT, mmctx->quant_ib_first[ith]);
}

स्थिर शून्य quantize_f32_q8_1_s16_tiled_खण्डम्(अचिह्नित पूर्णाङ्क nth, अचिह्नित पूर्णाङ्क ith, शून्य * दत्त) {
    (शून्य) nth;
    संरचना htp_mm_context * mmctx = दत्त;
    यदि (mmctx->quant_ib_first[ith] >= mmctx->quant_ib_चरमः[ith]) {
        प्रत्यागम;
    }
    संरचना htp_ops_context * octx = mmctx->octx;
    संरचना htp_thread_trace * tr = &octx->ctx->trace[ith];
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_A_QUANT, mmctx->quant_ib_first[ith]);

    ध्रुव संरचना htp_tensor * src = mmctx->act;

    quantize_f32_q8_1_s16_tiled_खण्डम्_kernel(
        (ध्रुव भिन्न *) mmctx->vtcm_act_raw,
        mmctx->vtcm_act,
        रिक्त,
        src->ne[0],
        mmctx->quant_ib_first[ith],
        mmctx->quant_ib_चरमः[ith],
        mmctx->vtcm_act_raw_stride,
        htp_mm_q8_1_tiled_row_आकार(src->ne[0]),
        mmctx->quant_r[ith],
        mmctx->quant_c[ith]
    );

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_A_QUANT, mmctx->quant_ib_first[ith]);
}

स्थिर शून्य quantize_f32_f16_tiled_खण्डम्(अचिह्नित पूर्णाङ्क nth, अचिह्नित पूर्णाङ्क ith, शून्य * दत्त) {
    (शून्य) nth;
    संरचना htp_mm_context * mmctx = दत्त;
    यदि (mmctx->quant_ib_first[ith] >= mmctx->quant_ib_चरमः[ith]) {
        प्रत्यागम;
    }
    संरचना htp_ops_context * octx = mmctx->octx;
    संरचना htp_thread_trace * tr = &octx->ctx->trace[ith];
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_A_QUANT, mmctx->quant_ib_first[ith]);

    ध्रुव संरचना htp_tensor * src = mmctx->act;

    quantize_f32_f16_tiled_खण्डम्_kernel(
        (ध्रुव भिन्न *) mmctx->vtcm_act_raw,
        mmctx->vtcm_act,
        रिक्त,
        src->ne[0],
        mmctx->quant_ib_first[ith],
        mmctx->quant_ib_चरमः[ith],
        mmctx->vtcm_act_raw_stride,
        htp_mm_f16_tiled_row_आकार(src->ne[0]),
        mmctx->quant_r[ith],
        mmctx->quant_c[ith]
    );

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_A_QUANT, mmctx->quant_ib_first[ith]);
}

स्थिर शून्य quantize_f16_f16_tiled_खण्डम्(अचिह्नित पूर्णाङ्क nth, अचिह्नित पूर्णाङ्क ith, शून्य * दत्त) {
    (शून्य) nth;
    संरचना htp_mm_context * mmctx = दत्त;
    यदि (mmctx->quant_ib_first[ith] >= mmctx->quant_ib_चरमः[ith]) {
        प्रत्यागम;
    }
    संरचना htp_ops_context * octx = mmctx->octx;
    संरचना htp_thread_trace * tr = &octx->ctx->trace[ith];
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_A_QUANT, mmctx->quant_ib_first[ith]);

    ध्रुव संरचना htp_tensor * src = mmctx->act;

    quantize_f16_f16_tiled_खण्डम्_kernel(
        (ध्रुव __fp16 *) mmctx->vtcm_act_raw,
        mmctx->vtcm_act,
        रिक्त,
        src->ne[0],
        mmctx->quant_ib_first[ith],
        mmctx->quant_ib_चरमः[ith],
        mmctx->vtcm_act_raw_stride,
        htp_mm_f16_tiled_row_आकार(src->ne[0]),
        mmctx->quant_r[ith],
        mmctx->quant_c[ith]
    );

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_A_QUANT, mmctx->quant_ib_first[ith]);
}

// q8_1 for weight types with offsets (q8_1_s16 for Q2_K), f16_tiled for F16, otherwise q8_0
स्थिर अन्तःस्थ कार्यकर्ता_callback_t htp_mm_act_quant_row_func(पूर्णाङ्क weight_type, पूर्णाङ्क act_type) {
    यदि (weight_type == HTP_TYPE_F16) {
        प्रत्यागम (act_type == HTP_TYPE_F16) ? quantize_f16_f16_tiled : quantize_f32_f16_tiled;
    }
    यदि (weight_type == HTP_TYPE_Q2_K) {
        प्रत्यागम quantize_f32_q8_1_s16_tiled;
    }
    प्रत्यागम htp_mm_weight_has_विस्थापनम्(weight_type) ? quantize_f32_q8_1_tiled : quantize_f32_q8_0_tiled;
}

स्थिर अन्तःस्थ कार्यकर्ता_callback_t htp_mm_act_quant_खण्डम्_func(पूर्णाङ्क weight_type, पूर्णाङ्क act_type) {
    यदि (weight_type == HTP_TYPE_F16) {
        प्रत्यागम (act_type == HTP_TYPE_F16) ? quantize_f16_f16_tiled_खण्डम् : quantize_f32_f16_tiled_खण्डम्;
    }
    यदि (weight_type == HTP_TYPE_Q2_K) {
        प्रत्यागम quantize_f32_q8_1_s16_tiled_खण्डम्;
    }
    प्रत्यागम htp_mm_weight_has_विस्थापनम्(weight_type) ? quantize_f32_q8_1_tiled_खण्डम् : quantize_f32_q8_0_tiled_खण्डम्;
}

स्थिर शून्य hvx_mm_4d_tiled(अचिह्नित पूर्णाङ्क nth, अचिह्नित पूर्णाङ्क ith, शून्य * दत्त) {
    htp_matmul_preamble;

    ध्रुव uint32_t src0_nrows = mmctx->src0_row_अन्तः - mmctx->src0_row_start;
    ध्रुव uint32_t cur_m_पङ्क्तिकाः = mmctx->cur_m_पङ्क्तिकाः ? mmctx->cur_m_पङ्क्तिकाः : (ne11 * ne12 * ne13);
    ध्रुव uint32_t cur_m_start = mmctx->cur_m_start;

    ध्रुव uint32_t src0_start_row  = mmctx->src0_row_start + src0_nrows_प्रत्येकम्_thread * ith;
    ध्रुव uint32_t src0_अन्तः_row    = MIN(src0_start_row + src0_nrows_प्रत्येकम्_thread, mmctx->src0_row_अन्तः);

    संरचना htp_thread_trace * tr = &octx->ctx->trace[ith];

    ध्रुव संरचना htp_mm_kernel_params * kparams = (ध्रुव संरचना htp_mm_kernel_params *) octx->kernel_params;
    ध्रुव uint32_t n_prefetch = kparams->n_prefetch;
    प्रतिज्ञा(n_prefetch >= 2 && n_prefetch <= HTP_MM_MAX_PREFETCH && (n_prefetch & (n_prefetch - 1)) == 0);

    ध्रुव size_t dst_row_आकार = nb1;
    ध्रुव size_t act_stride = mmctx->vtcm_act_stride;

    uint8_t * सीमित vtcm_src0_ptr = mmctx->vtcm_src0 + mmctx->vtcm_src0_आकार_प्रत्येकम्_thread * ith;
    uint8_t * सीमित act_दत्त = mmctx->vtcm_act;

    ध्रुव uint32_t tile_आकार = htp_tensor_get_tile_आकार(src0->type);
    ध्रुव uint32_t aligned_tile_आकार = htp_tensor_get_tile_padded_आकार(src0->type);

    ध्रुव uint32_t n_k_tiles_w = ne00 / htp_tensor_get_tile_k(src0->type);
    ध्रुव uint32_t tile_row_stride = n_k_tiles_w * tile_आकार;
    ध्रुव uint32_t tile_row_transfer_आकार_aligned = n_k_tiles_w * aligned_tile_आकार;
    ध्रुव uint32_t src0_खण्डिका_stride = ((ne01 + 31) / 32) * tile_row_stride;

    ध्रुव uint32_t ct_start = src0_start_row / 32;
    ध्रुव uint32_t ct_अन्तः   = (src0_अन्तः_row + 31) / 32;

    यदि (src0_start_row >= src0_अन्तः_row || cur_m_पङ्क्तिकाः == 0) {
        प्रत्यागम;
    }

    ध्रुव uint32_t योग_batches = ne12 * ne13;
    ध्रुव uint32_t b_start = fastdiv(cur_m_start, &kparams->div_ne1);
    uint32_t b_अन्तः = fastdiv(cur_m_start + cur_m_पङ्क्तिकाः + ne11 - 1, &kparams->div_ne1);
    b_अन्तः = MIN(b_अन्तः, योग_batches);

    uint32_t b_grp_start = b_start;
    यावत् (b_grp_start < b_अन्तः) {
        ध्रुव uint32_t b3 = fastdiv(b_grp_start, &kparams->div_ne12);
        ध्रुव uint32_t b2 = b_grp_start - b3 * ne12;
        ध्रुव uint32_t i02 = fastdiv(b2, &kparams->div_r2);
        ध्रुव uint32_t i03 = fastdiv(b3, &kparams->div_r3);

        uint32_t b_grp_अन्तः = b_grp_start + 1;
        यावत् (b_grp_अन्तः < b_अन्तः) {
            ध्रुव uint32_t cur_b3 = fastdiv(b_grp_अन्तः, &kparams->div_ne12);
            ध्रुव uint32_t cur_b2 = b_grp_अन्तः - cur_b3 * ne12;
            ध्रुव uint32_t cur_i02 = fastdiv(cur_b2, &kparams->div_r2);
            ध्रुव uint32_t cur_i03 = fastdiv(cur_b3, &kparams->div_r3);
            यदि (cur_i02 != i02 || cur_i03 != i03) {
                भङ्ग;
            }
            b_grp_अन्तः++;
        }

        ध्रुव uint32_t खण्डिका_idx = i03 * ne02 + i02;
        ध्रुव dma_addr_t src0_खण्डिका = src0->दत्त + (size_t) खण्डिका_idx * src0_खण्डिका_stride;

        uint32_t प्रक्षिप_ct = ct_start;
        प्रति (uint32_t d = 0; d < n_prefetch && प्रक्षिप_ct < ct_अन्तः; d++, प्रक्षिप_ct++) {
            dma_queue_प्रक्षिप(dma_q, dma_make_दत्त(vtcm_src0_ptr + d * tile_row_transfer_आकार_aligned,
                           src0_खण्डिका + (size_t) प्रक्षिप_ct * tile_row_stride),
                           aligned_tile_आकार, tile_आकार, tile_आकार, n_k_tiles_w);
        }

        प्रति (uint32_t ct = ct_start; ct < ct_अन्तः; ct++) {
            ध्रुव uint8_t * w_tile = (शून्य *) dma_queue_निष्कासय(dma_q).dst;

            पूर्णाङ्क valid_पङ्क्तिकाः = (पूर्णाङ्क)ne0 - (पूर्णाङ्क)(ct * 32);
            valid_पङ्क्तिकाः = MIN(32, MAX(0, valid_पङ्क्तिकाः));

            htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, ct);
            प्रति (uint32_t b = b_grp_start; b < b_grp_अन्तः; b++) {
                ध्रुव uint32_t b_m_start = b * ne11;
                ध्रुव uint32_t m_first = MAX(cur_m_start, b_m_start);
                ध्रुव uint32_t m_चरमः  = MIN(cur_m_start + cur_m_पङ्क्तिकाः, b_m_start + ne11);
                यदि (m_first >= m_चरमः) अनुवर्त;

                ध्रुव uint32_t cur_b3 = fastdiv(b, &kparams->div_ne12);
                ध्रुव uint32_t cur_b2 = b - cur_b3 * ne12;
                uint8_t * dst_batch_आधारः = (uint8_t *) dst->दत्त + (size_t) cur_b2 * nb2 + (size_t) cur_b3 * nb3;

                ध्रुव uint32_t chunk_m_विस्थापनम् = m_first - cur_m_start;
                ध्रुव uint32_t dst_m_विस्थापनम्   = m_first - b_m_start;
                ध्रुव uint32_t batch_nrows    = m_चरमः - m_first;

                uint32_t ir1 = 0;
                प्रति (; ir1 + 1 < batch_nrows; ir1 += 2) {
                    ध्रुव uint8_t * सीमित act_col0 = (ध्रुव uint8_t *) (act_दत्त + (chunk_m_विस्थापनम् + ir1 + 0) * act_stride);
                    ध्रुव uint8_t * सीमित act_col1 = (ध्रुव uint8_t *) (act_दत्त + (chunk_m_विस्थापनम् + ir1 + 1) * act_stride);
                    भिन्न * सीमित dst_row0 = (भिन्न *) (dst_batch_आधारः + (dst_m_विस्थापनम् + ir1 + 0) * dst_row_आकार);
                    भिन्न * सीमित dst_row1 = (भिन्न *) (dst_batch_आधारः + (dst_m_विस्थापनम् + ir1 + 1) * dst_row_आकार);
                    भिन्न * dst_ptr0 = &dst_row0[ct * 32];
                    भिन्न * dst_ptr1 = &dst_row1[ct * 32];
                    mmctx->vec_dot_32x2(ne10, dst_ptr0, dst_ptr1, w_tile, act_col0, act_col1, valid_पङ्क्तिकाः, रिक्त, रिक्त);
                }
                प्रति (; ir1 < batch_nrows; ++ir1) {
                    ध्रुव uint8_t * सीमित act_col = (ध्रुव uint8_t *) (act_दत्त + (chunk_m_विस्थापनम् + ir1) * act_stride);
                    भिन्न * सीमित dst_row = (भिन्न *) (dst_batch_आधारः + (dst_m_विस्थापनम् + ir1) * dst_row_आकार);
                    भिन्न * dst_ptr = &dst_row[ct * 32];
                    mmctx->vec_dot_32x1(ne10, dst_ptr, w_tile, act_col, valid_पङ्क्तिकाः, रिक्त);
                }
            }
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ct);

            यदि (प्रक्षिप_ct < ct_अन्तः) {
                dma_queue_प्रक्षिप(dma_q, dma_make_दत्त(w_tile, src0_खण्डिका + (size_t) प्रक्षिप_ct * tile_row_stride),
                               aligned_tile_आकार, tile_आकार, tile_आकार, n_k_tiles_w);
                प्रक्षिप_ct++;
            }
        }
        b_grp_start = b_grp_अन्तः;
    }
    यदि (src2) {
        hvx_tensor_सङ्कलय_f32_grid(dst, src2, cur_m_start, cur_m_start + cur_m_पङ्क्तिकाः, src0_start_row, src0_अन्तः_row, &kparams->div_ne12_ne1, &kparams->div_ne1);
    }
}

स्थिर शून्य hvx_mm_2d_flat(अचिह्नित पूर्णाङ्क nth, अचिह्नित पूर्णाङ्क ith, शून्य * दत्त) {
    htp_matmul_preamble;

    ध्रुव संरचना htp_mm_kernel_params * kparams = (ध्रुव संरचना htp_mm_kernel_params *) octx->kernel_params;
    ध्रुव uint32_t n_prefetch = kparams->n_prefetch;
    प्रतिज्ञा(n_prefetch >= 2 && n_prefetch <= HTP_MM_MAX_PREFETCH && (n_prefetch & (n_prefetch - 1)) == 0);
    ध्रुव uint32_t prefetch_आवरणम् = n_prefetch - 1;

    ध्रुव uint32_t src0_nrows = mmctx->src0_row_अन्तः - mmctx->src0_row_start;  // src0 rows
    ध्रुव uint32_t act_nrows = mmctx->cur_m_पङ्क्तिकाः ? mmctx->cur_m_पङ्क्तिकाः : mmctx->act_nrows;
    ध्रुव uint32_t cur_m_start = mmctx->cur_m_start;

    ध्रुव uint32_t src0_start_row  = mmctx->src0_row_start + src0_nrows_प्रत्येकम्_thread * ith;
    ध्रुव uint32_t src0_अन्तः_row    = MIN(src0_start_row + src0_nrows_प्रत्येकम्_thread, mmctx->src0_row_अन्तः);
    ध्रुव uint32_t src0_अन्तः_row_x2 = src0_start_row + ((src0_अन्तः_row - src0_start_row) & ~1U);

    संरचना htp_thread_trace * tr = &octx->ctx->trace[ith];

    ध्रुव size_t dst_row_आकार  = nb1;
    ध्रुव size_t src0_row_आकार = nb01;
    ध्रुव size_t act_row_आकार = nb11;

    ध्रुव size_t src0_stride = mmctx->vtcm_src0_stride;
    ध्रुव size_t act_stride = mmctx->vtcm_act_stride;

    // Per-thread VTCMs for all tensors
    uint8_t * सीमित vtcm_dst_ptr  = mmctx->vtcm_dst  + mmctx->vtcm_dst_आकार_प्रत्येकम्_thread  * ith;
    uint8_t * सीमित vtcm_src0_ptr = mmctx->vtcm_src0 + mmctx->vtcm_src0_आकार_प्रत्येकम्_thread * ith;
    uint8_t * सीमित act_दत्त      = mmctx->vtcm_act;

    ध्रुव dma_addr_t src0_row = src0->दत्त;


    // Prefill vtcm with src0 rows
    यदि (src0_start_row < src0_अन्तः_row) {
        प्रति (uint32_t ir0 = src0_start_row; ir0 < src0_अन्तः_row_x2; ir0 += 2) {
            ध्रुव पूर्णाङ्क is0 = (ir0 - src0_start_row);
            यदि (is0 >= (पूर्णाङ्क)n_prefetch) {
                भङ्ग;
            }
            dma_queue_प्रक्षिप(dma_q, dma_make_दत्त(vtcm_src0_ptr + is0 * src0_stride, src0_row + ir0 * src0_row_आकार),
                           src0_stride, src0_row_आकार, src0_row_आकार, 2);
        }
    }

    यदि (src0_start_row >= src0_अन्तः_row) {
        प्रत्यागम;
    }

    // Process src0 rows
    प्रति (uint32_t ir0 = src0_start_row; ir0 < src0_अन्तः_row_x2; ir0 += 2) {
        ध्रुव uint8_t * ss0 = (शून्य *) dma_queue_निष्कासय(dma_q).dst;

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, ir0);
        // Process act columns in pairs (2x2 tiling)
        uint32_t ir1 = 0;
        प्रति (; ir1 + 1 < act_nrows; ir1 += 2) {
            ध्रुव uint8_t * सीमित act_col0 = (ध्रुव uint8_t *) (act_दत्त + (ir1+0) * act_stride);
            ध्रुव uint8_t * सीमित act_col1 = (ध्रुव uint8_t *) (act_दत्त + (ir1+1) * act_stride);
            भिन्न * सीमित dst_row0 = (भिन्न *) (dst->दत्त + ((cur_m_start + ir1+0) * dst_row_आकार));
            भिन्न * सीमित dst_row1 = (भिन्न *) (dst->दत्त + ((cur_m_start + ir1+1) * dst_row_आकार));
            mmctx->vec_dot_2x2(ne00, &dst_row0[ir0], &dst_row1[ir0], ss0, ss0 + src0_stride, act_col0, act_col1);
        }

        // Handle remaining act rows (fallback to 2x1)
        प्रति (; ir1 < act_nrows; ++ir1) {
            ध्रुव uint8_t * सीमित act_col = (ध्रुव uint8_t *) (act_दत्त + ir1 * act_stride);
            भिन्न * सीमित dst_row          = (भिन्न *) (dst->दत्त + ((cur_m_start + ir1) * dst_row_आकार));
            mmctx->vec_dot_2x1(ne00, &dst_row[ir0], ss0, ss0 + src0_stride, act_col);
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ir0);

        // Prefetch next (n + vtcm_nrows) row
        ध्रुव पूर्णाङ्क pr0 = (ir0 + n_prefetch);
        ध्रुव पूर्णाङ्क is0 = (pr0 - src0_start_row) & prefetch_आवरणम्;
        यदि (pr0 < src0_अन्तः_row_x2) {
            dma_queue_प्रक्षिप(dma_q, dma_make_दत्त(vtcm_src0_ptr + is0 * src0_stride, src0_row + pr0 * src0_row_आकार),
                           src0_stride, src0_row_आकार, src0_row_आकार, 2);
        }
    }

    // Process the last row (if any)
    यदि (src0_अन्तः_row != src0_अन्तः_row_x2) {
        uint32_t  ir0 = src0_अन्तः_row_x2;
        ध्रुव पूर्णाङ्क is0 = (ir0 - src0_start_row) & prefetch_आवरणम्;
        dma_queue_प्रक्षिप(dma_q, dma_make_दत्त(vtcm_src0_ptr + is0 * src0_stride, src0_row + ir0 * src0_row_आकार),
                       src0_stride, src0_row_आकार, src0_row_आकार, 1);
        ध्रुव uint8_t * ss0 = (शून्य *) dma_queue_निष्कासय(dma_q).dst;

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, ir0);
        #निर्देशः unroll(2)
        प्रति (uint32_t ir1 = 0; ir1 < act_nrows; ++ir1) {
            ध्रुव uint8_t * सीमित act_col = (ध्रुव uint8_t *) (act_दत्त + ir1 * act_stride);
            भिन्न * सीमित dst_row          = (भिन्न *) (dst->दत्त + ((cur_m_start + ir1) * dst_row_आकार));
            mmctx->vec_dot_1x1(ne00, &dst_row[ir0], ss0, act_col);
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ir0);
    }
    यदि (src2) {
        hvx_tensor_सङ्कलय_f32_grid(dst, src2, cur_m_start, cur_m_start + act_nrows, src0_start_row, src0_अन्तः_row, &kparams->div_ne12_ne1, &kparams->div_ne1);
    }
}

स्थिर शून्य hvx_mv_2d_flat(अचिह्नित पूर्णाङ्क nth, अचिह्नित पूर्णाङ्क ith, शून्य * दत्त) {
    htp_matmul_preamble;

    ध्रुव uint32_t src0_nrows = mmctx->src0_row_अन्तः - mmctx->src0_row_start;

    ध्रुव uint32_t src0_start_row  = mmctx->src0_row_start + src0_nrows_प्रत्येकम्_thread * ith;
    ध्रुव uint32_t src0_अन्तः_row    = MIN(src0_start_row + src0_nrows_प्रत्येकम्_thread, mmctx->src0_row_अन्तः);

    संरचना htp_thread_trace * tr = &octx->ctx->trace[ith];

    ध्रुव size_t dst_row_आकार  = nb1;
    ध्रुव size_t src0_row_आकार = nb01;
    ध्रुव size_t act_row_आकार  = nb11;

    ध्रुव size_t src0_stride = mmctx->vtcm_src0_stride;
    ध्रुव size_t act_stride  = mmctx->vtcm_act_stride;

    // Per-thread VTCMs for all tensors
    uint8_t * vtcm_dst_ptr  = mmctx->vtcm_dst  + mmctx->vtcm_dst_आकार_प्रत्येकम्_thread  * ith;
    uint8_t * vtcm_src0_ptr = mmctx->vtcm_src0 + mmctx->vtcm_src0_आकार_प्रत्येकम्_thread * ith;
    uint8_t * act_दत्त      = mmctx->vtcm_act;

    भिन्न * tmp = (भिन्न *) vtcm_dst_ptr;

    ध्रुव dma_addr_t src0_row = src0->दत्त;
    ध्रुव uint8_t * सीमित act_col = (ध्रुव uint8_t *) act_दत्त;
    भिन्न * सीमित dst_col         = (भिन्न *) dst->दत्त;

    ध्रुव uint32_t src0_अन्तः_row_x2 = src0_start_row + ((src0_अन्तः_row - src0_start_row) & ~1U);

    ध्रुव संरचना htp_mm_kernel_params * kparams = (ध्रुव संरचना htp_mm_kernel_params *) octx->kernel_params;
    ध्रुव uint32_t n_prefetch = kparams->n_prefetch;
    प्रतिज्ञा(n_prefetch >= 2 && n_prefetch <= HTP_MM_MAX_PREFETCH && (n_prefetch & (n_prefetch - 1)) == 0);
    ध्रुव uint32_t prefetch_आवरणम् = n_prefetch - 1;

    // Prefill vtcm with 2x src0 rows
    यदि (src0_start_row < src0_अन्तः_row) {
        यदि (src2) {
            भिन्न * vtcm_bias_ptr = (भिन्न *) mmctx->vtcm_bias + src0_start_row;
            ध्रुव dma_addr_t src2_addr = src2->दत्त + src0_start_row * परिमाण(भिन्न);
            पूर्णाङ्क खण्डिका_आकार = (पूर्णाङ्क)src0_अन्तः_row - (पूर्णाङ्क)src0_start_row;
            यदि (खण्डिका_आकार > 0) {
                dma_queue_प्रक्षिप(dma_q, dma_make_दत्त(vtcm_bias_ptr, src2_addr),
                               खण्डिका_आकार * परिमाण(भिन्न), खण्डिका_आकार * परिमाण(भिन्न), खण्डिका_आकार * परिमाण(भिन्न), 1);
                dma_queue_निष्कासय_nowait(dma_q);
            }
        }
        प्रति (uint32_t ir0 = src0_start_row; ir0 < src0_अन्तः_row_x2; ir0 += 2) {
            ध्रुव uint32_t is0 = (ir0 - src0_start_row);
            यदि (is0 >= n_prefetch) {
                भङ्ग;
            }
            dma_queue_प्रक्षिप(dma_q, dma_make_दत्त(vtcm_src0_ptr + is0 * src0_stride, src0_row + ir0 * src0_row_आकार),
                           src0_stride, src0_row_आकार, src0_row_आकार, 2);
        }
    }


    यदि (src0_start_row >= src0_अन्तः_row) {
        प्रत्यागम;
    }

    // Process src0 rows
    प्रति (uint32_t ir0 = src0_start_row; ir0 < src0_अन्तः_row_x2; ir0 += 2) {
        ध्रुव uint8_t * ss0 = (शून्य *) dma_queue_निष्कासय(dma_q).dst;
        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, ir0);
        mmctx->vec_dot_2x1(ne00, &tmp[ir0 - src0_start_row], ss0, ss0 + src0_stride, act_col);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ir0);

        // Prefetch next (n + vtcm_nrows) row
        ध्रुव uint32_t pr0 = (ir0 + n_prefetch);
        ध्रुव uint32_t is0 = (pr0 - src0_start_row) & prefetch_आवरणम्;
        यदि (pr0 < src0_अन्तः_row_x2) {
            dma_queue_प्रक्षिप(dma_q, dma_make_दत्त(vtcm_src0_ptr + is0 * src0_stride, src0_row + pr0 * src0_row_आकार),
                           src0_stride, src0_row_आकार, src0_row_आकार, 2);
        }
    }

    // Process the last row (if any)
    यदि (src0_अन्तः_row != src0_अन्तः_row_x2) {
        ध्रुव uint32_t ir0 = src0_अन्तः_row_x2;
        ध्रुव uint32_t is0 = (ir0 - src0_start_row) & prefetch_आवरणम्;
        dma_queue_प्रक्षिप(dma_q, dma_make_दत्त(vtcm_src0_ptr + is0 * src0_stride, src0_row + ir0 * src0_row_आकार),
                       src0_stride, src0_row_आकार, src0_row_आकार, 1);
        ध्रुव uint8_t * ss0 = (शून्य *) dma_queue_निष्कासय(dma_q).dst;
        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, ir0);
        mmctx->vec_dot_1x1(ne00, &tmp[ir0 - src0_start_row], ss0, act_col);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ir0);
    }

    पूर्णाङ्क copy_cnt = src0_अन्तः_row - src0_start_row;
    यदि (copy_cnt > 0) {
        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, src0_अन्तः_row);
        यदि (src2) {
            hvx_सङ्कलय_f32_uuu((uint8_t *) &dst_col[src0_start_row],
                            (ध्रुव uint8_t *) tmp,
                            (ध्रुव uint8_t *) ((ध्रुव भिन्न *) mmctx->vtcm_bias + src0_start_row),
                            copy_cnt);
        } अन्यथा {
            hvx_copy_f32_ua((uint8_t *) &dst_col[src0_start_row], (uint8_t *) tmp, copy_cnt);
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, src0_अन्तः_row);
    }
}

स्थिर शून्य hvx_mm_4d_flat(अचिह्नित पूर्णाङ्क nth, अचिह्नित पूर्णाङ्क ith, शून्य * दत्त) {
    htp_matmul_preamble;

    ध्रुव संरचना htp_mm_kernel_params * kparams = (ध्रुव संरचना htp_mm_kernel_params *) octx->kernel_params;
    ध्रुव uint32_t n_prefetch = kparams->n_prefetch;
    प्रतिज्ञा(n_prefetch >= 2 && n_prefetch <= HTP_MM_MAX_PREFETCH && (n_prefetch & (n_prefetch - 1)) == 0);
    ध्रुव uint32_t prefetch_आवरणम् = n_prefetch - 1;

    ध्रुव uint32_t src0_nrows = mmctx->src0_row_अन्तः - mmctx->src0_row_start;
    ध्रुव uint32_t cur_m_पङ्क्तिकाः = mmctx->cur_m_पङ्क्तिकाः ? mmctx->cur_m_पङ्क्तिकाः : (ne11 * ne12 * ne13);
    ध्रुव uint32_t cur_m_start = mmctx->cur_m_start;

    ध्रुव uint32_t src0_start_row  = mmctx->src0_row_start + src0_nrows_प्रत्येकम्_thread * ith;
    ध्रुव uint32_t src0_अन्तः_row    = MIN(src0_start_row + src0_nrows_प्रत्येकम्_thread, mmctx->src0_row_अन्तः);
    ध्रुव uint32_t src0_अन्तः_row_x2 = src0_start_row + ((src0_अन्तः_row - src0_start_row) & ~1U);

    संरचना htp_thread_trace * tr = &octx->ctx->trace[ith];

    ध्रुव size_t dst_row_आकार  = nb1;
    ध्रुव size_t src0_row_आकार = nb01;
    ध्रुव size_t src0_stride = mmctx->vtcm_src0_stride;
    ध्रुव size_t act_stride  = mmctx->vtcm_act_stride;

    uint8_t * सीमित vtcm_src0_ptr = mmctx->vtcm_src0 + mmctx->vtcm_src0_आकार_प्रत्येकम्_thread * ith;
    uint8_t * सीमित act_दत्त      = mmctx->vtcm_act;


    यदि (src0_start_row >= src0_अन्तः_row || cur_m_पङ्क्तिकाः == 0) {
        प्रत्यागम;
    }

    ध्रुव uint32_t योग_batches = ne12 * ne13;
    ध्रुव uint32_t b_start = fastdiv(cur_m_start, &kparams->div_ne1);
    uint32_t b_अन्तः = fastdiv(cur_m_start + cur_m_पङ्क्तिकाः + ne11 - 1, &kparams->div_ne1);
    b_अन्तः = MIN(b_अन्तः, योग_batches);

    uint32_t b_grp_start = b_start;
    यावत् (b_grp_start < b_अन्तः) {
        ध्रुव uint32_t b3 = fastdiv(b_grp_start, &kparams->div_ne12);
        ध्रुव uint32_t b2 = b_grp_start - b3 * ne12;
        ध्रुव uint32_t i02 = fastdiv(b2, &kparams->div_r2);
        ध्रुव uint32_t i03 = fastdiv(b3, &kparams->div_r3);

        uint32_t b_grp_अन्तः = b_grp_start + 1;
        यावत् (b_grp_अन्तः < b_अन्तः) {
            ध्रुव uint32_t cur_b3 = fastdiv(b_grp_अन्तः, &kparams->div_ne12);
            ध्रुव uint32_t cur_b2 = b_grp_अन्तः - cur_b3 * ne12;
            ध्रुव uint32_t cur_i02 = fastdiv(cur_b2, &kparams->div_r2);
            ध्रुव uint32_t cur_i03 = fastdiv(cur_b3, &kparams->div_r3);
            यदि (cur_i02 != i02 || cur_i03 != i03) {
                भङ्ग;
            }
            b_grp_अन्तः++;
        }

        ध्रुव dma_addr_t src0_row = src0->दत्त + ((size_t) i02 * nb02 + (size_t) i03 * nb03);

        प्रति (uint32_t ir0 = src0_start_row; ir0 < src0_अन्तः_row_x2; ir0 += 2) {
            ध्रुव पूर्णाङ्क is0 = (ir0 - src0_start_row);
            यदि (is0 >= (पूर्णाङ्क)n_prefetch) {
                भङ्ग;
            }
            dma_queue_प्रक्षिप(dma_q, dma_make_दत्त(vtcm_src0_ptr + is0 * src0_stride, src0_row + (size_t) ir0 * src0_row_आकार),
                           src0_stride, src0_row_आकार, src0_row_आकार, 2);
        }

        प्रति (uint32_t ir0 = src0_start_row; ir0 < src0_अन्तः_row_x2; ir0 += 2) {
            ध्रुव uint8_t * ss0 = (शून्य *) dma_queue_निष्कासय(dma_q).dst;

            htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, ir0);
            प्रति (uint32_t b = b_grp_start; b < b_grp_अन्तः; b++) {
                ध्रुव uint32_t b_m_start = b * ne11;
                ध्रुव uint32_t m_first = MAX(cur_m_start, b_m_start);
                ध्रुव uint32_t m_चरमः  = MIN(cur_m_start + cur_m_पङ्क्तिकाः, b_m_start + ne11);
                यदि (m_first >= m_चरमः) अनुवर्त;

                ध्रुव uint32_t cur_b3 = fastdiv(b, &kparams->div_ne12);
                ध्रुव uint32_t cur_b2 = b - cur_b3 * ne12;
                uint8_t * dst_batch_आधारः = (uint8_t *) dst->दत्त + (size_t) cur_b2 * nb2 + (size_t) cur_b3 * nb3;

                ध्रुव uint32_t chunk_m_विस्थापनम् = m_first - cur_m_start;
                ध्रुव uint32_t dst_m_विस्थापनम्   = m_first - b_m_start;
                ध्रुव uint32_t batch_nrows    = m_चरमः - m_first;

                uint32_t ir1 = 0;
                प्रति (; ir1 + 1 < batch_nrows; ir1 += 2) {
                    ध्रुव uint8_t * सीमित act_col0 = (ध्रुव uint8_t *) (act_दत्त + (chunk_m_विस्थापनम् + ir1 + 0) * act_stride);
                    ध्रुव uint8_t * सीमित act_col1 = (ध्रुव uint8_t *) (act_दत्त + (chunk_m_विस्थापनम् + ir1 + 1) * act_stride);
                    भिन्न * सीमित dst_row0 = (भिन्न *) (dst_batch_आधारः + (dst_m_विस्थापनम् + ir1 + 0) * dst_row_आकार);
                    भिन्न * सीमित dst_row1 = (भिन्न *) (dst_batch_आधारः + (dst_m_विस्थापनम् + ir1 + 1) * dst_row_आकार);
                    mmctx->vec_dot_2x2(ne00, &dst_row0[ir0], &dst_row1[ir0], ss0, ss0 + src0_stride, act_col0, act_col1);
                }
                प्रति (; ir1 < batch_nrows; ++ir1) {
                    ध्रुव uint8_t * सीमित act_col = (ध्रुव uint8_t *) (act_दत्त + (chunk_m_विस्थापनम् + ir1) * act_stride);
                    भिन्न * सीमित dst_row          = (भिन्न *) (dst_batch_आधारः + (dst_m_विस्थापनम् + ir1) * dst_row_आकार);
                    mmctx->vec_dot_2x1(ne00, &dst_row[ir0], ss0, ss0 + src0_stride, act_col);
                }
            }
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ir0);

            ध्रुव पूर्णाङ्क pr0 = (ir0 + n_prefetch);
            ध्रुव पूर्णाङ्क is0 = (pr0 - src0_start_row) & prefetch_आवरणम्;
            यदि (pr0 < src0_अन्तः_row_x2) {
                dma_queue_प्रक्षिप(dma_q, dma_make_दत्त(vtcm_src0_ptr + is0 * src0_stride, src0_row + (size_t) pr0 * src0_row_आकार),
                               src0_stride, src0_row_आकार, src0_row_आकार, 2);
            }
        }

        यदि (src0_अन्तः_row != src0_अन्तः_row_x2) {
            uint32_t ir0 = src0_अन्तः_row_x2;
            ध्रुव पूर्णाङ्क is0 = (ir0 - src0_start_row) & prefetch_आवरणम्;
            dma_queue_प्रक्षिप(dma_q, dma_make_दत्त(vtcm_src0_ptr + is0 * src0_stride, src0_row + (size_t) ir0 * src0_row_आकार),
                           src0_stride, src0_row_आकार, src0_row_आकार, 1);
            ध्रुव uint8_t * ss0 = (शून्य *) dma_queue_निष्कासय(dma_q).dst;

            htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, ir0);
            प्रति (uint32_t b = b_grp_start; b < b_grp_अन्तः; b++) {
                ध्रुव uint32_t b_m_start = b * ne11;
                ध्रुव uint32_t m_first = MAX(cur_m_start, b_m_start);
                ध्रुव uint32_t m_चरमः  = MIN(cur_m_start + cur_m_पङ्क्तिकाः, b_m_start + ne11);
                यदि (m_first >= m_चरमः) अनुवर्त;

                ध्रुव uint32_t cur_b3 = fastdiv(b, &kparams->div_ne12);
                ध्रुव uint32_t cur_b2 = b - cur_b3 * ne12;
                uint8_t * dst_batch_आधारः = (uint8_t *) dst->दत्त + (size_t) cur_b2 * nb2 + (size_t) cur_b3 * nb3;

                ध्रुव uint32_t chunk_m_विस्थापनम् = m_first - cur_m_start;
                ध्रुव uint32_t dst_m_विस्थापनम्   = m_first - b_m_start;
                ध्रुव uint32_t batch_nrows    = m_चरमः - m_first;

                प्रति (uint32_t ir1 = 0; ir1 < batch_nrows; ++ir1) {
                    ध्रुव uint8_t * सीमित act_col = (ध्रुव uint8_t *) (act_दत्त + (chunk_m_विस्थापनम् + ir1) * act_stride);
                    भिन्न * सीमित dst_row          = (भिन्न *) (dst_batch_आधारः + (dst_m_विस्थापनम् + ir1) * dst_row_आकार);
                    mmctx->vec_dot_1x1(ne00, &dst_row[ir0], ss0, act_col);
                }
            }
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ir0);
        }

        b_grp_start = b_grp_अन्तः;
    }

    यदि (src2) {
        hvx_tensor_सङ्कलय_f32_grid(dst, src2, cur_m_start, cur_m_start + cur_m_पङ्क्तिकाः, src0_start_row, src0_अन्तः_row, &kparams->div_ne12_ne1, &kparams->div_ne1);
    }
}

#परिभाषय MMID_MATRIX_ROW(row_id, i1) matrix_पङ्क्तिकाः[(row_id) * mmctx->mapping_stride + (i1)]

स्थिर शून्य hvx_mm_id_tiled(अचिह्नित पूर्णाङ्क nth, अचिह्नित पूर्णाङ्क ith, शून्य * दत्त) {
    htp_matmul_preamble;

    ध्रुव संरचना htp_tensor * सीमित ids = octx->src[2];

    ध्रुव uint32_t src0_nrows      = mmctx->src0_row_अन्तः - mmctx->src0_row_start;  // src0 rows per expert
    ध्रुव uint32_t act_nrows       = ne11;
    ध्रुव uint32_t src0_start_row  = mmctx->src0_row_start + src0_nrows_प्रत्येकम्_thread * ith;
    ध्रुव uint32_t src0_अन्तः_row    = MIN(src0_start_row + src0_nrows_प्रत्येकम्_thread, mmctx->src0_row_अन्तः);


    यदि (src0_start_row >= src0_अन्तः_row) {
        प्रत्यागम;
    }

    संरचना htp_thread_trace * tr = &octx->ctx->trace[ith];

    ध्रुव संरचना htp_mm_kernel_params * kparams = (ध्रुव संरचना htp_mm_kernel_params *) octx->kernel_params;
    ध्रुव uint32_t n_prefetch = kparams->n_prefetch;
    प्रतिज्ञा(n_prefetch >= 2 && n_prefetch <= HTP_MM_MAX_PREFETCH && (n_prefetch & (n_prefetch - 1)) == 0);

    ध्रुव uint32_t n_ids = ids->ne[0];  // n_expert_used
    ध्रुव uint32_t n_as  = ne02;        // n_expert

    ध्रुव uint32_t *                matrix_row_counts = mmctx->matrix_row_counts;
    ध्रुव संरचना mmid_row_mapping * matrix_पङ्क्तिकाः       = mmctx->matrix_पङ्क्तिकाः;

    ध्रुव size_t dst_row_आकार  = nb1;
    ध्रुव size_t act_row_आकार  = htp_mm_q8_0_tiled_row_आकार(ne10);

    ध्रुव size_t act_stride = mmctx->vtcm_act_stride;

    // Per-thread VTCMs for all tensors
    uint8_t * सीमित vtcm_src0_ptr = mmctx->vtcm_src0 + mmctx->vtcm_src0_आकार_प्रत्येकम्_thread * ith;
    uint8_t * सीमित act_दत्त = mmctx->vtcm_act;

    प्रति (uint32_t cur_a = 0; cur_a < n_as; ++cur_a) {
        ध्रुव int32_t cne1 = matrix_row_counts[cur_a];
        यदि (cne1 == 0) {
            अनुवर्त;
        }

        ध्रुव dma_addr_t src0_row = src0->दत्त + cur_a * nb02;

        ध्रुव uint32_t tile_आकार = htp_tensor_get_tile_आकार(src0->type);
        ध्रुव uint32_t aligned_tile_आकार = htp_tensor_get_tile_padded_आकार(src0->type);
        ध्रुव uint32_t n_k_tiles_w = ne00 / htp_tensor_get_tile_k(src0->type);
        ध्रुव uint32_t tile_row_stride = n_k_tiles_w * tile_आकार;
        ध्रुव uint32_t tile_row_transfer_आकार_aligned = n_k_tiles_w * aligned_tile_आकार;

        ध्रुव uint32_t ct_start = src0_start_row / 32;
        ध्रुव uint32_t ct_अन्तः   = (src0_अन्तः_row + 31) / 32;

        uint32_t प्रक्षिप_ct = ct_start;
        प्रति (uint32_t d = 0; d < n_prefetch && प्रक्षिप_ct < ct_अन्तः; d++, प्रक्षिप_ct++) {
            dma_queue_प्रक्षिप(dma_q, dma_make_दत्त(vtcm_src0_ptr + d * tile_row_transfer_आकार_aligned, src0_row + प्रक्षिप_ct * tile_row_stride),
                           aligned_tile_आकार, tile_आकार, tile_आकार, n_k_tiles_w);
        }

        प्रति (uint32_t ct = ct_start; ct < ct_अन्तः; ct++) {
            ध्रुव uint8_t * w_tile = (शून्य *) dma_queue_निष्कासय(dma_q).dst;

            पूर्णाङ्क valid_पङ्क्तिकाः = (पूर्णाङ्क)ne0 - (पूर्णाङ्क)(ct * 32);
            valid_पङ्क्तिकाः = MIN(32, MAX(0, valid_पङ्क्तिकाः));

            htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, ct);
            प्रति (uint32_t cid = 0; cid < cne1; ++cid) {
                संरचना mmid_row_mapping row_mapping = MMID_MATRIX_ROW(cur_a, cid);
                ध्रुव पूर्णाङ्क               rm1         = row_mapping.i1;  // expert idx
                ध्रुव पूर्णाङ्क               rm2         = row_mapping.i2;  // token idx

                ध्रुव uint32_t ir1 = fastmodulo(rm1, ne11, &mmctx->mm_div_ne11);        // act row idx
                ध्रुव uint8_t * सीमित act_col = (ध्रुव uint8_t *) (act_दत्त + (ir1 + rm2 * ne11 + 0) * act_stride);
                भिन्न * सीमित dst_row = (भिन्न *) (dst->दत्त + (rm1 * nb1 + rm2 * nb2 + 0));

                mmctx->vec_dot_32x1(ne10, &dst_row[ct * 32], w_tile, act_col, valid_पङ्क्तिकाः, रिक्त);
            }
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ct);

            यदि (प्रक्षिप_ct < ct_अन्तः) {
                dma_queue_प्रक्षिप(dma_q, dma_make_दत्त(w_tile, src0_row + प्रक्षिप_ct * tile_row_stride),
                               aligned_tile_आकार, tile_आकार, tile_आकार, n_k_tiles_w);
                प्रक्षिप_ct++;
            }
        }
    }
}

स्थिर शून्य hvx_mv_id_tiled(अचिह्नित पूर्णाङ्क nth, अचिह्नित पूर्णाङ्क ith, शून्य * दत्त) {
    htp_matmul_preamble;

    ध्रुव संरचना htp_tensor * सीमित ids = octx->src[2];

    ध्रुव uint32_t src0_nrows      = mmctx->src0_row_अन्तः - mmctx->src0_row_start;  // src0 rows per expert
    ध्रुव uint32_t src0_start_row  = mmctx->src0_row_start + src0_nrows_प्रत्येकम्_thread * ith;
    ध्रुव uint32_t src0_अन्तः_row    = MIN(src0_start_row + src0_nrows_प्रत्येकम्_thread, mmctx->src0_row_अन्तः);


    यदि (src0_start_row >= src0_अन्तः_row) {
        प्रत्यागम;
    }

    संरचना htp_thread_trace * tr = &octx->ctx->trace[ith];

    ध्रुव संरचना htp_mm_kernel_params * kparams = (ध्रुव संरचना htp_mm_kernel_params *) octx->kernel_params;
    ध्रुव uint32_t n_prefetch = kparams->n_prefetch;
    प्रतिज्ञा(n_prefetch >= 2 && n_prefetch <= HTP_MM_MAX_PREFETCH && (n_prefetch & (n_prefetch - 1)) == 0);

    प्रतिज्ञा(ne13 % ne03 == 0);

    ध्रुव size_t dst_row_आकार  = nb1;
    ध्रुव size_t act_row_आकार = htp_mm_q8_0_tiled_row_आकार(ne10);

    ध्रुव uint32_t n_aids = src2->ne[0];  // num activated experts
    ध्रुव uint32_t n_ids  = ne02;         // num experts

    // Per-thread VTCMs for all tensors
    uint8_t * सीमित vtcm_src0_ptr = mmctx->vtcm_src0 + mmctx->vtcm_src0_आकार_प्रत्येकम्_thread * ith;
    uint8_t * सीमित act_दत्त = mmctx->vtcm_act;

    प्रति (uint32_t ie1 = 0; ie1 < n_aids; ++ie1) {  // for each expert
        ध्रुव int32_t eid = *(ध्रुव int32_t *) ((ध्रुव uint8_t *) src2->दत्त + ie1 * src2->nb[0]);
        यदि (eid < 0) {
            अनुवर्त;
        }
        प्रतिज्ञा(eid < (int32_t) n_ids);

        ध्रुव dma_addr_t src0_row = src0->दत्त + eid * nb02;
        ध्रुव uint8_t * सीमित act_col = (ध्रुव uint8_t *) act_दत्त;
        भिन्न * सीमित dst_row          = (भिन्न *) (dst->दत्त + ie1 * nb1);

        ध्रुव uint32_t tile_आकार = htp_tensor_get_tile_आकार(src0->type);
        ध्रुव uint32_t aligned_tile_आकार = htp_tensor_get_tile_padded_आकार(src0->type);
        ध्रुव uint32_t n_k_tiles_w = ne00 / htp_tensor_get_tile_k(src0->type);
        ध्रुव uint32_t tile_row_stride = n_k_tiles_w * tile_आकार;
        ध्रुव uint32_t tile_row_transfer_आकार_aligned = n_k_tiles_w * aligned_tile_आकार;

        ध्रुव uint32_t ct_start = src0_start_row / 32;
        ध्रुव uint32_t ct_अन्तः   = (src0_अन्तः_row + 31) / 32;

        uint32_t प्रक्षिप_ct = ct_start;
        प्रति (uint32_t d = 0; d < n_prefetch && प्रक्षिप_ct < ct_अन्तः; d++, प्रक्षिप_ct++) {
            dma_queue_प्रक्षिप(dma_q, dma_make_दत्त(vtcm_src0_ptr + d * tile_row_transfer_आकार_aligned, src0_row + प्रक्षिप_ct * tile_row_stride),
                           aligned_tile_आकार, tile_आकार, tile_आकार, n_k_tiles_w);
        }

        प्रति (uint32_t ct = ct_start; ct < ct_अन्तः; ct++) {
            ध्रुव uint8_t * w_tile = (शून्य *) dma_queue_निष्कासय(dma_q).dst;

            पूर्णाङ्क valid_पङ्क्तिकाः = (पूर्णाङ्क)ne0 - (पूर्णाङ्क)(ct * 32);
            valid_पङ्क्तिकाः = MIN(32, MAX(0, valid_पङ्क्तिकाः));

            htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, ct);
            mmctx->vec_dot_32x1(ne10, &dst_row[ct * 32], w_tile, act_col, valid_पङ्क्तिकाः, रिक्त);
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ct);

            यदि (प्रक्षिप_ct < ct_अन्तः) {
                dma_queue_प्रक्षिप(dma_q, dma_make_दत्त(w_tile, src0_row + प्रक्षिप_ct * tile_row_stride),
                               aligned_tile_आकार, tile_आकार, tile_आकार, n_k_tiles_w);
                प्रक्षिप_ct++;
            }
        }
    }
}

स्थिर शून्य hvx_mv_id_nx_tiled(अचिह्नित पूर्णाङ्क nth, अचिह्नित पूर्णाङ्क ith, शून्य * दत्त) {
    संरचना htp_mm_context * mmctx = (संरचना htp_mm_context *) दत्त;
    संरचना htp_ops_context * octx = mmctx->octx;
    ध्रुव संरचना htp_mm_kernel_params * kparams = (ध्रुव संरचना htp_mm_kernel_params *) octx->kernel_params;
    ध्रुव uint32_t n_weights      = kparams->n_weights;
    ध्रुव संरचना htp_tensor * सीमित src0 = octx->src[0];
    ध्रुव संरचना htp_tensor * सीमित act  = octx->src[n_weights];
    ध्रुव संरचना htp_tensor * सीमित ids  = octx->src[n_weights + 1];


    संरचना htp_thread_trace * tr = &octx->ctx->trace[ith];

    ध्रुव uint32_t n_prefetch = kparams->n_prefetch;
    प्रतिज्ञा(n_prefetch >= 2 && n_prefetch <= HTP_MM_MAX_PREFETCH && (n_prefetch & (n_prefetch - 1)) == 0);

    ध्रुव uint32_t n_aids = ids->ne[0];
    ध्रुव uint32_t n_ids  = src0->ne[2];

    uint8_t * सीमित vtcm_src0_ptr = mmctx->vtcm_src0 + mmctx->vtcm_src0_आकार_प्रत्येकम्_thread * ith;
    uint8_t * सीमित act_दत्त = mmctx->vtcm_act;

    प्रति (uint32_t ie1 = 0; ie1 < n_aids; ++ie1) {
        ध्रुव int32_t eid = *(ध्रुव int32_t *) ((ध्रुव uint8_t *) ids->दत्त + ie1 * ids->nb[0]);
        यदि (eid < 0) अनुवर्त;
        प्रतिज्ञा(eid < (int32_t) n_ids);

        प्रति (uint32_t p = 0; p < n_weights; ++p) {
            ध्रुव संरचना htp_tensor * सीमित src_w = octx->src[p];
            ध्रुव संरचना htp_tensor * सीमित dst   = octx->dsts[p];
            यदि (!src_w || !dst) अनुवर्त;
            dma_queue * dma_q = octx->ctx->dma[ith];

            ध्रुव uint32_t ne01 = src_w->ne[1];
            uint32_t start_row = 0;
            uint32_t अन्तः_row   = ne01;
            यदि (octx->ctx->mdev.गणनम् > 1) {
                ध्रुव bool can_विभज = htp_tensor_can_row_partition(dst, परिमाण(भिन्न));
                ध्रुव संरचना htp_tensor_mdev_range range = htp_tensor_mdev_partition(ne01, can_विभज ? 32 : 0, octx->ctx->mdev.idx, octx->ctx->mdev.गणनम्, &octx->ctx->mdev.गणनम्_div);
                start_row = range.start;
                अन्तः_row   = range.start + range.गणनम्;
            }

            ध्रुव uint32_t nrows = अन्तः_row - start_row;
            uint32_t src0_nrows_प्रत्येकम्_thread = fastdiv(nrows + nth - 1, &octx->n_तन्तवः_div);
            src0_nrows_प्रत्येकम्_thread = hex_आवर्तनम्_ऊर्ध्वम्(src0_nrows_प्रत्येकम्_thread, 32);

            ध्रुव uint32_t src0_start_row = start_row + src0_nrows_प्रत्येकम्_thread * ith;
            ध्रुव uint32_t src0_अन्तः_row   = MIN(src0_start_row + src0_nrows_प्रत्येकम्_thread, अन्तः_row);
            यदि (src0_start_row >= src0_अन्तः_row) अनुवर्त;

            ध्रुव dma_addr_t src0_row = src_w->दत्त + eid * src_w->nb[2];
            ध्रुव uint8_t * सीमित act_col = (ध्रुव uint8_t *) act_दत्त;
            भिन्न * सीमित dst_row = (भिन्न *) (dst->दत्त + ie1 * dst->nb[1]);

            ध्रुव uint32_t tile_आकार = htp_tensor_get_tile_आकार(src_w->type);
            ध्रुव uint32_t aligned_tile_आकार = htp_tensor_get_tile_padded_आकार(src_w->type);
            ध्रुव uint32_t n_k_tiles_w = src_w->ne[0] / htp_tensor_get_tile_k(src_w->type);
            ध्रुव uint32_t tile_row_stride = n_k_tiles_w * tile_आकार;
            ध्रुव uint32_t tile_row_transfer_आकार_aligned = n_k_tiles_w * aligned_tile_आकार;

            ध्रुव uint32_t ct_start = src0_start_row / 32;
            ध्रुव uint32_t ct_अन्तः   = (src0_अन्तः_row + 31) / 32;

            uint32_t प्रक्षिप_ct = ct_start;
            प्रति (uint32_t d = 0; d < n_prefetch && प्रक्षिप_ct < ct_अन्तः; d++, प्रक्षिप_ct++) {
                dma_queue_प्रक्षिप(dma_q, dma_make_दत्त(vtcm_src0_ptr + d * tile_row_transfer_आकार_aligned, src0_row + प्रक्षिप_ct * tile_row_stride),
                               aligned_tile_आकार, tile_आकार, tile_आकार, n_k_tiles_w);
            }

            प्रति (uint32_t ct = ct_start; ct < ct_अन्तः; ct++) {
                ध्रुव uint8_t * w_tile = (शून्य *) dma_queue_निष्कासय(dma_q).dst;

                पूर्णाङ्क valid_पङ्क्तिकाः = (पूर्णाङ्क)dst->ne[0] - (पूर्णाङ्क)(ct * 32);
                valid_पङ्क्तिकाः = MIN(32, MAX(0, valid_पङ्क्तिकाः));

                htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, ct);
                mmctx->vec_dot_32x1(act->ne[0], &dst_row[ct * 32], w_tile, act_col, valid_पङ्क्तिकाः, रिक्त);
                htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ct);

                यदि (प्रक्षिप_ct < ct_अन्तः) {
                    dma_queue_प्रक्षिप(dma_q, dma_make_दत्त(w_tile, src0_row + प्रक्षिप_ct * tile_row_stride),
                                    aligned_tile_आकार, tile_आकार, tile_आकार, n_k_tiles_w);
                    प्रक्षिप_ct++;
                }
            }
        }
    }
}

स्थिर शून्य hvx_mm_id_nx_tiled(अचिह्नित पूर्णाङ्क nth, अचिह्नित पूर्णाङ्क ith, शून्य * दत्त) {
    संरचना htp_mm_context * mmctx = (संरचना htp_mm_context *) दत्त;
    संरचना htp_ops_context * octx = mmctx->octx;
    ध्रुव संरचना htp_mm_kernel_params * kparams = (ध्रुव संरचना htp_mm_kernel_params *) octx->kernel_params;
    ध्रुव uint32_t n_weights      = kparams->n_weights;
    ध्रुव संरचना htp_tensor * सीमित src0 = octx->src[0];
    ध्रुव संरचना htp_tensor * सीमित act  = octx->src[n_weights];
    ध्रुव संरचना htp_tensor * सीमित ids  = octx->src[n_weights + 1];


    संरचना htp_thread_trace * tr = &octx->ctx->trace[ith];

    ध्रुव uint32_t n_prefetch = kparams->n_prefetch;
    प्रतिज्ञा(n_prefetch >= 2 && n_prefetch <= HTP_MM_MAX_PREFETCH && (n_prefetch & (n_prefetch - 1)) == 0);

    ध्रुव uint32_t n_as = src0->ne[2];

    ध्रुव uint32_t * matrix_row_counts = mmctx->matrix_row_counts;
    ध्रुव संरचना mmid_row_mapping * matrix_पङ्क्तिकाः = mmctx->matrix_पङ्क्तिकाः;

    ध्रुव size_t act_stride = mmctx->vtcm_act_stride;

    uint8_t * सीमित vtcm_src0_ptr = mmctx->vtcm_src0 + mmctx->vtcm_src0_आकार_प्रत्येकम्_thread * ith;
    uint8_t * सीमित act_दत्त = mmctx->vtcm_act;

    प्रति (uint32_t cur_a = 0; cur_a < n_as; ++cur_a) {
        ध्रुव int32_t cne1 = matrix_row_counts[cur_a];
        यदि (cne1 == 0) अनुवर्त;

        प्रति (uint32_t p = 0; p < n_weights; ++p) {
            ध्रुव संरचना htp_tensor * सीमित src_w = octx->src[p];
            ध्रुव संरचना htp_tensor * सीमित dst   = octx->dsts[p];
            यदि (!src_w || !dst) अनुवर्त;
            dma_queue * dma_q = octx->ctx->dma[ith];

            ध्रुव uint32_t ne01 = src_w->ne[1];
            uint32_t start_row = 0;
            uint32_t अन्तः_row   = ne01;
            यदि (octx->ctx->mdev.गणनम् > 1) {
                ध्रुव bool can_विभज = htp_tensor_can_row_partition(dst, परिमाण(भिन्न));
                ध्रुव संरचना htp_tensor_mdev_range range = htp_tensor_mdev_partition(ne01, can_विभज ? 32 : 0, octx->ctx->mdev.idx, octx->ctx->mdev.गणनम्, &octx->ctx->mdev.गणनम्_div);
                start_row = range.start;
                अन्तः_row   = range.start + range.गणनम्;
            }

            ध्रुव uint32_t nrows = अन्तः_row - start_row;
            uint32_t src0_nrows_प्रत्येकम्_thread = fastdiv(nrows + nth - 1, &octx->n_तन्तवः_div);
            src0_nrows_प्रत्येकम्_thread = hex_आवर्तनम्_ऊर्ध्वम्(src0_nrows_प्रत्येकम्_thread, 32);

            ध्रुव uint32_t src0_start_row = start_row + src0_nrows_प्रत्येकम्_thread * ith;
            ध्रुव uint32_t src0_अन्तः_row   = MIN(src0_start_row + src0_nrows_प्रत्येकम्_thread, अन्तः_row);
            यदि (src0_start_row >= src0_अन्तः_row) अनुवर्त;

            ध्रुव dma_addr_t src0_row = src_w->दत्त + cur_a * src_w->nb[2];

            ध्रुव uint32_t tile_आकार = htp_tensor_get_tile_आकार(src_w->type);
            ध्रुव uint32_t aligned_tile_आकार = htp_tensor_get_tile_padded_आकार(src_w->type);
            ध्रुव uint32_t n_k_tiles_w = src_w->ne[0] / htp_tensor_get_tile_k(src_w->type);
            ध्रुव uint32_t tile_row_stride = n_k_tiles_w * tile_आकार;
            ध्रुव uint32_t tile_row_transfer_आकार_aligned = n_k_tiles_w * aligned_tile_आकार;

            ध्रुव uint32_t ct_start = src0_start_row / 32;
            ध्रुव uint32_t ct_अन्तः   = (src0_अन्तः_row + 31) / 32;

            uint32_t प्रक्षिप_ct = ct_start;
            प्रति (uint32_t d = 0; d < n_prefetch && प्रक्षिप_ct < ct_अन्तः; d++, प्रक्षिप_ct++) {
                dma_queue_प्रक्षिप(dma_q, dma_make_दत्त(vtcm_src0_ptr + d * tile_row_transfer_आकार_aligned, src0_row + प्रक्षिप_ct * tile_row_stride),
                               aligned_tile_आकार, tile_आकार, tile_आकार, n_k_tiles_w);
            }

            प्रति (uint32_t ct = ct_start; ct < ct_अन्तः; ct++) {
                ध्रुव uint8_t * w_tile = (शून्य *) dma_queue_निष्कासय(dma_q).dst;

                पूर्णाङ्क valid_पङ्क्तिकाः = (पूर्णाङ्क)dst->ne[0] - (पूर्णाङ्क)(ct * 32);
                valid_पङ्क्तिकाः = MIN(32, MAX(0, valid_पङ्क्तिकाः));

                htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, ct);
                प्रति (uint32_t cid = 0; cid < (uint32_t) cne1; ++cid) {
                    संरचना mmid_row_mapping row_mapping = MMID_MATRIX_ROW(cur_a, cid);
                    ध्रुव पूर्णाङ्क rm1 = row_mapping.i1;
                    ध्रुव पूर्णाङ्क rm2 = row_mapping.i2;

                    ध्रुव uint32_t ir1 = fastmodulo(rm1, act->ne[1], &mmctx->mm_div_ne11);
                    ध्रुव uint8_t * सीमित act_col = (ध्रुव uint8_t *) (act_दत्त + (ir1 + rm2 * act->ne[1]) * act_stride);
                    भिन्न * सीमित dst_row = (भिन्न *) (dst->दत्त + (rm1 * dst->nb[1] + rm2 * dst->nb[2]));

                    mmctx->vec_dot_32x1(act->ne[0], &dst_row[ct * 32], w_tile, act_col, valid_पङ्क्तिकाः, रिक्त);
                }
                htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ct);

                यदि (प्रक्षिप_ct < ct_अन्तः) {
                    dma_queue_प्रक्षिप(dma_q, dma_make_दत्त(w_tile, src0_row + प्रक्षिप_ct * tile_row_stride),
                                    aligned_tile_आकार, tile_आकार, tile_आकार, n_k_tiles_w);
                    प्रक्षिप_ct++;
                }
            }
        }
    }
}

स्थिर पूर्णाङ्क hvx_mm_init_tiled_vec_dot(संरचना htp_mm_context * mmctx, गणना htp_दत्त_type type) {
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
    uint32_t src0_row_अन्तः   = src0_nrows;

    यदि (octx->ctx->mdev.गणनम् > 1) {
        ध्रुव bool can_विभज = htp_tensor_can_row_partition(dst, परिमाण(भिन्न));
        ध्रुव संरचना htp_tensor_mdev_range range = htp_tensor_mdev_partition(src0_nrows, can_विभज ? 32 : 0, octx->ctx->mdev.idx, octx->ctx->mdev.गणनम्, &octx->ctx->mdev.गणनम्_div);
        src0_row_start = range.start;
        src0_row_अन्तः   = range.start + range.गणनम्;
    }

    यदि (src0_row_start >= src0_row_अन्तः) {
        प्रत्यागम HTP_स्थितिः_OK;
    }

    ध्रुव uint32_t nrows = src0_row_अन्तः - src0_row_start;
    mmctx->src0_row_start = src0_row_start;
    mmctx->src0_row_अन्तः   = src0_row_अन्तः;

    ध्रुव bool is_tiled = htp_tensor_is_tiled(src0);

    // Compute src0_nrows_per_thread
    mmctx->src0_nrows_प्रत्येकम्_thread  = fastdiv(nrows + octx->n_तन्तवः - 1, &octx->n_तन्तवः_div);
    यदि (is_tiled) {
        mmctx->src0_nrows_प्रत्येकम्_thread = hex_आवर्तनम्_ऊर्ध्वम्(mmctx->src0_nrows_प्रत्येकम्_thread, 32);
    } अन्यथा {
        mmctx->src0_nrows_प्रत्येकम्_thread += (mmctx->src0_nrows_प्रत्येकम्_thread & 1); // round up to even
    }

    ध्रुव size_t src0_row_आकार = nb01;
    ध्रुव size_t dst_row_आकार  = nb1;
    size_t       act_row_आकार  = nb11;

    ध्रुव size_t src0_row_आकार_padded = hex_आवर्तनम्_ऊर्ध्वम्(src0_row_आकार, 128);

    कार्यकर्ता_callback_t quant_task_func;
    कार्यकर्ता_callback_t matmul_job_func;
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
            act_row_आकार           = kparams->act_row_आकार;
            भङ्ग;

        पक्ष HTP_MM_KERNEL_HVX_F32_F32_VTCM:
            quant_task_func        = रिक्त;
            need_quant             = false;
            mmctx->type            = "f32-f32";
            mmctx->vec_dot_1x1     = flat_vec_dot_f32_f32_aa_1x1;
            mmctx->vec_dot_2x1     = flat_vec_dot_f32_f32_aa_2x1;
            mmctx->vec_dot_2x2     = flat_vec_dot_f32_f32_aa_2x2;
            act_row_आकार           = kparams->act_row_आकार;
            भङ्ग;

        पक्ष HTP_MM_KERNEL_HVX_BF16_F32_VTCM:
            quant_task_func        = रिक्त;
            need_quant             = false;
            mmctx->type            = "bf16-f32";
            mmctx->vec_dot_1x1     = flat_vec_dot_bf16_f32_aa_1x1;
            mmctx->vec_dot_2x1     = flat_vec_dot_bf16_f32_aa_2x1;
            mmctx->vec_dot_2x2     = flat_vec_dot_bf16_f32_aa_2x2;
            act_row_आकार           = kparams->act_row_आकार;
            भङ्ग;

        पक्ष HTP_MM_KERNEL_HVX_QUANT_खण्डम्:
        पक्ष HTP_MM_KERNEL_HVX_QUANT_ROW:
        स्वभाव: {
            यदि (hvx_mm_init_tiled_vec_dot(mmctx, src0->type) != 0) {
                प्रत्यागम HTP_स्थितिः_NO_SUPPORT;
            }

            ध्रुव uint32_t qk = QK_Q8_0_TILED;
            ध्रुव uint32_t nb = (ne10 + qk - 1) / qk;
            ध्रुव uint32_t योग_nb = act_nrows * nb;

            यदि (act_nrows < octx->n_तन्तवः && !is_batched) {
                n_quant_tasks = MIN(योग_nb, octx->n_तन्तवः);
                quant_task_func = htp_mm_act_quant_खण्डम्_func(src0->type, act->type);
                प्रति (uint32_t ith = 0; ith < n_quant_tasks; ++ith) {
                    uint32_t ib_first = (योग_nb * ith) / n_quant_tasks;
                    uint32_t ib_चरमः  = (योग_nb * (ith + 1)) / n_quant_tasks;
                    mmctx->quant_ib_first[ith] = ib_first;
                    mmctx->quant_ib_चरमः[ith]  = ib_चरमः;
                    mmctx->quant_r[ith]        = ib_first / nb;
                    mmctx->quant_c[ith]        = ib_first % nb;
                }
            } अन्यथा {
                n_quant_tasks = MIN(act_nrows, octx->n_तन्तवः);
                quant_task_func = htp_mm_act_quant_row_func(src0->type, act->type);
            }
            act_row_आकार = kparams->act_row_आकार;
            भङ्ग;
        }
    }

    ध्रुव uint32_t m_chunk = (kparams->m_chunk > 0 && (uint32_t) kparams->m_chunk < act_nrows)
                           ? (uint32_t) kparams->m_chunk : act_nrows;
    ध्रुव uint32_t m_layout_पङ्क्तिकाः = m_chunk;

    संरचना htp_mm_hvx_vtcm_layout L;
    htp_mm_hvx_vtcm_layout_build(&L, kparams->kernel_type, src0->type, ne10, m_layout_पङ्क्तिकाः, octx->n_तन्तवः,
                                 dst_row_आकार, src0_row_आकार, act_row_आकार, src2 ? src2->nb[1] : 0, kparams->n_prefetch, false, false);

    यदि (kparams->kernel_type == HTP_MM_KERNEL_HVX_F16_F16_VTCM ||
        kparams->kernel_type == HTP_MM_KERNEL_HVX_F32_F32_VTCM ||
        kparams->kernel_type == HTP_MM_KERNEL_HVX_BF16_F32_VTCM ||
        kparams->kernel_type == HTP_MM_KERNEL_HVX_QUANT_ROW ||
        kparams->kernel_type == HTP_MM_KERNEL_HVX_QUANT_खण्डम्) {
        mmctx->vtcm_act_आकार_प्रत्येकम्_thread = L.act_अष्टकानि;
    } अन्यथा {
        mmctx->vtcm_act_आकार_प्रत्येकम्_thread = fastdiv(L.act_अष्टकानि, &octx->n_तन्तवः_div);
    }

    mmctx->vtcm_src0_आकार_प्रत्येकम्_thread = fastdiv(L.src0_अष्टकानि, &octx->n_तन्तवः_div);
    mmctx->vtcm_dst_आकार_प्रत्येकम्_thread  = fastdiv(L.dst_अष्टकानि, &octx->n_तन्तवः_div);

    ध्रुव size_t vtcm_आकार = L.योग_अष्टकानि;

    FARF(उच्चः_2, "matmul-%s : src0-vtcm-size %zu act-vtcm-size %zu dst-vtcm-size %zu (%zu)\n", mmctx->type,
         L.src0_अष्टकानि, L.act_अष्टकानि, L.dst_अष्टकानि, vtcm_आकार);

    FARF(उच्चः_2, "matmul-%s : %ux%ux%ux%u * %ux%ux%ux%u-> %ux%ux%ux%u (0x%p, 0x%p, 0x%p)\n", mmctx->type, src0->ne[0],
         src0->ne[1], src0->ne[2], src0->ne[3], act->ne[0], act->ne[1], act->ne[2], act->ne[3], dst->ne[0],
         dst->ne[1], dst->ne[2], dst->ne[3], src0->दत्त, act->दत्त, dst->दत्त);

    यदि (octx->ctx->vtcm_आकार < vtcm_आकार) {
        FARF(ERROR, "matmul-%s : current VTCM reservation %zu is too small, needed %zu\n", mmctx->type,
             octx->ctx->vtcm_आकार, vtcm_आकार);
        प्रत्यागम HTP_स्थितिः_VTCM_TOO_SMALL;
    }

    uint8_t * ध्रुव आधारः = (uint8_t *) octx->ctx->vtcm_आधारः;
    mmctx->vtcm_act      = VTCM_LAYOUT_PTR(uint8_t, आधारः, L.off_act);
    mmctx->vtcm_src0     = VTCM_LAYOUT_PTR(uint8_t, आधारः, L.off_src0);
    mmctx->vtcm_bias     = VTCM_LAYOUT_PTR(uint8_t, आधारः, L.off_bias);
    mmctx->vtcm_dst      = VTCM_LAYOUT_PTR(uint8_t, आधारः, L.off_dst);
    mmctx->vtcm_act_raw  = VTCM_LAYOUT_PTR(uint8_t, आधारः, L.off_act_raw);

    mmctx->vtcm_src0_stride = src0_row_आकार_padded;
    mmctx->vtcm_act_stride  = act_row_आकार;
    यदि (kparams->kernel_type == HTP_MM_KERNEL_HVX_QUANT_खण्डम् || kparams->kernel_type == HTP_MM_KERNEL_HVX_QUANT_ROW) {
        mmctx->vtcm_act_raw_stride = hex_आवर्तनम्_ऊर्ध्वम्(ne10 * परिमाण(भिन्न), QK_Q8_0_TILED * परिमाण(भिन्न));
    } अन्यथा यदि (kparams->kernel_type == HTP_MM_KERNEL_HVX_F16_F16_VTCM) {
        mmctx->vtcm_act_raw_stride = hex_आवर्तनम्_ऊर्ध्वम्(ne10 * परिमाण(भिन्न), 128);
    } अन्यथा {
        mmctx->vtcm_act_raw_stride = 0;
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_INIT, 0);

    यदि (kparams->m_chunk > 0 && (uint32_t) kparams->m_chunk < act_nrows) {
        प्रति (uint32_t m_start = 0; m_start < act_nrows; m_start += m_chunk) {
            ध्रुव uint32_t cur_m_पङ्क्तिकाः = MIN(act_nrows - m_start, m_chunk);
            mmctx->cur_m_start = m_start;
            mmctx->cur_m_पङ्क्तिकाः  = cur_m_पङ्क्तिकाः;

            यदि (need_quant) {
                hvx_mm_transfer_act_dma(octx, kparams, act, mmctx->vtcm_act_raw, mmctx->vtcm_act_raw_stride, m_start, cur_m_पङ्क्तिकाः);

                ध्रुव uint32_t qk = QK_Q8_0_TILED;
                ध्रुव uint32_t nb = (ne10 + qk - 1) / qk;
                ध्रुव uint32_t योग_nb = cur_m_पङ्क्तिकाः * nb;
                uint32_t quant_tasks;
                work_queue_func_t q_func;
                यदि (cur_m_पङ्क्तिकाः < octx->n_तन्तवः && (kparams->kernel_type == HTP_MM_KERNEL_HVX_QUANT_खण्डम् || kparams->kernel_type == HTP_MM_KERNEL_HVX_QUANT_ROW)) {
                    quant_tasks = MIN(योग_nb, octx->n_तन्तवः);
                    q_func = htp_mm_act_quant_खण्डम्_func(src0->type, act->type);
                    प्रति (uint32_t ith = 0; ith < quant_tasks; ++ith) {
                        uint32_t ib_first = (योग_nb * ith) / quant_tasks;
                        uint32_t ib_चरमः  = (योग_nb * (ith + 1)) / quant_tasks;
                        mmctx->quant_ib_first[ith] = ib_first;
                        mmctx->quant_ib_चरमः[ith]  = ib_चरमः;
                        mmctx->quant_r[ith]        = ib_first / nb;
                        mmctx->quant_c[ith]        = ib_first % nb;
                    }
                } अन्यथा {
                    quant_tasks = MIN(cur_m_पङ्क्तिकाः, octx->n_तन्तवः);
                    q_func = quant_task_func;
                    mmctx->n_quant_पङ्क्तिकाः_प्रत्येकम्_thread = (cur_m_पङ्क्तिकाः + quant_tasks - 1) / quant_tasks;
                }
                mmctx->n_quant_tasks = quant_tasks;
                work_queue_सततम्(octx->ctx->work_queue, q_func, mmctx, quant_tasks);
            } अन्यथा {
                hvx_mm_transfer_act_dma(octx, kparams, act, mmctx->vtcm_act, mmctx->vtcm_act_stride, m_start, cur_m_पङ्क्तिकाः);
            }

            work_queue_सततम्(octx->ctx->work_queue, matmul_job_func, mmctx, octx->n_तन्तवः);
        }
    } अन्यथा {
        mmctx->cur_m_start = 0;
        mmctx->cur_m_पङ्क्तिकाः  = act_nrows;

        यदि (need_quant) {
            hvx_mm_transfer_act_dma(octx, kparams, act, mmctx->vtcm_act_raw, mmctx->vtcm_act_raw_stride, 0, act_nrows);
            mmctx->n_quant_पङ्क्तिकाः_प्रत्येकम्_thread = (act_nrows + n_quant_tasks - 1) / n_quant_tasks;
            mmctx->n_quant_tasks = n_quant_tasks;
            work_queue_सततम्(octx->ctx->work_queue, quant_task_func, mmctx, n_quant_tasks);
        } अन्यथा {
            hvx_mm_transfer_act_dma(octx, kparams, act, mmctx->vtcm_act, mmctx->vtcm_act_stride, 0, act_nrows);
        }

        work_queue_सततम्(octx->ctx->work_queue, matmul_job_func, mmctx, octx->n_तन्तवः);
    }

    प्रत्यागम HTP_स्थितिः_OK;
}

स्थिर शून्य dequantize_tiled_कार्यकर्ता_loop(अचिह्नित पूर्णाङ्क n, अचिह्नित पूर्णाङ्क i, शून्य * दत्त) {
    tiled_dequantize_state_t * state = (tiled_dequantize_state_t *) दत्त;
    संरचना htp_thread_trace * tr = &state->traces[i];
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_W_DEQUANT, i);
    प्रति (अचिह्नित पूर्णाङ्क task_id = i; task_id < (अचिह्नित पूर्णाङ्क) state->n_tasks; task_id += n) {
        पूर्णाङ्क start = task_id * state->n_tiles_प्रत्येकम्_task;
        पूर्णाङ्क अन्तः_2   = hex_smin(start + state->n_tiles_प्रत्येकम्_task, state->n_tot_tiles);
        state->task_fn(state, start, अन्तः_2);
    }
    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_W_DEQUANT, i);
}

स्थिर शून्य transfer_output_chunk_कार्यकर्ता_fn(अचिह्नित पूर्णाङ्क n, अचिह्नित पूर्णाङ्क i, शून्य *दत्त) {
    output_transfer_task_state_t *st = (output_transfer_task_state_t *) दत्त;

    संरचना htp_thread_trace * tr = &st->traces[i];

    पूर्णाङ्क start_chunk_idx = i * st->n_chunks_प्रत्येकम्_task;
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_O_PROC, start_chunk_idx);

    प्रति (अचिह्नित पूर्णाङ्क task_id = i; task_id < (अचिह्नित पूर्णाङ्क)st->n_tasks; task_id += n) {
        पूर्णाङ्क    chunk_idx  = task_id * st->n_chunks_प्रत्येकम्_task;
        size_t chunk_आकार = hex_smin(st->n_tot_chunks - chunk_idx, st->n_chunks_प्रत्येकम्_task);

        भिन्न        *dst      = st->dst      + chunk_idx * st->dst_stride;
        ध्रुव भिन्न  *src2     = st->src2     ? (st->src2     + chunk_idx * st->src2_stride) : रिक्त;
        transfer_output_chunk_fp16_गन्तव्यम्_fp32(dst, src2, st->vtcm_src, chunk_idx, chunk_आकार, st->n_स्तम्भसङ्ख्या, st->dst_stride, st->src2_stride, st->dst_स्तम्भसङ्ख्या);
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
    uint32_t                        n_chunks_प्रत्येकम्_task;
    uint32_t                        k_खण्डम्;
    uint32_t                        k_stride;
    uint32_t                        k_valid;
    size_t                          vtcm_f32_act_अष्टकानि_प्रत्येकम्_thread;
    uint32_t                        dma_चरणः_पङ्क्तिकाः;
    uint32_t                        dma_चरणः_पङ्क्तिकाः_सरणम्;
    uint32_t                        act_elem_आकार;  // 4=F32 src1, 2=F16 src1
} activation_transfer_task_state_t;

प्रकारनिर्देश संरचना {
    संरचना htp_context            * ctx;
    संरचना htp_thread_trace       * traces;
    __fp16                        * dst;
    dma_addr_t                      act_dma_addr;
    भिन्न                         * vtcm_f32_act;
    uint32_t                        n_पङ्क्तिकाः;
    uint32_t                        k_खण्डम्;
    uint32_t                        k_stride;
    uint32_t                        k_valid;
    uint32_t                        n_col_chunks;
    संरचना fastdiv_मानानि           n_तन्तवः_div;
    size_t                          vtcm_f32_act_अष्टकानि;
    uint32_t                        dma_चरणः_पङ्क्तिकाः;
    uint32_t                        dma_चरणः_पङ्क्तिकाः_सरणम्;
} activation_transfer_col_chunk_state_t;

स्थिर शून्य transfer_activation_chunk_fp32_गन्तव्यम्_fp16_dma_pipelined_col_chunk(
        dma_queue *dma_q,
        __fp16 *सीमित vtcm_dst,
        dma_addr_t act_dma_addr,
        uint32_t n_पङ्क्तिकाः,
        uint32_t k_खण्डम्,
        uint32_t k_stride,
        uint32_t k_chunk_valid,
        uint32_t c_first,
        uint32_t c_len,
        भिन्न *thread_f32_act,
        संरचना htp_thread_trace *tr,
        uint32_t dma_चरणः_पङ्क्तिकाः,
        uint32_t dma_चरणः_पङ्क्तिकाः_सरणम्) {

    ध्रुव uint32_t R = dma_चरणः_पङ्क्तिकाः;
    ध्रुव uint32_t n_पङ्क्तिकाः_padded = hex_align_ऊर्ध्वम्(n_पङ्क्तिकाः, HTP_MM_HMX_TILE_N_पङ्क्तिकाः);

    ध्रुव uint32_t n_चरणाः = n_पङ्क्तिकाः_padded >> dma_चरणः_पङ्क्तिकाः_सरणम्;

    // Push step 0
    यदि (n_चरणाः > 0 && n_पङ्क्तिकाः > 0) {
        uint32_t nrows_गन्तव्यम्_fetch = hex_smin(n_पङ्क्तिकाः, R);
        dma_queue_प्रक्षिप(dma_q, dma_make_दत्त(thread_f32_act, act_dma_addr + (size_t) c_first * परिमाण(भिन्न)),
                       c_len * परिमाण(भिन्न), k_stride * परिमाण(भिन्न), k_chunk_valid * परिमाण(भिन्न), nrows_गन्तव्यम्_fetch);
    }
    // Push step 1
    यदि (n_चरणाः > 1) {
        uint32_t अग्र_r = R * 1;
        यदि (अग्र_r < n_पङ्क्तिकाः) {
            uint32_t nrows_गन्तव्यम्_fetch = hex_smin(n_पङ्क्तिकाः - अग्र_r, R);
            भिन्न *अग्र_buf = thread_f32_act + 1 * R * c_len;
            dma_queue_प्रक्षिप(dma_q, dma_make_दत्त(अग्र_buf, act_dma_addr + ((size_t) अग्र_r * k_stride + c_first) * परिमाण(भिन्न)),
                           c_len * परिमाण(भिन्न), k_stride * परिमाण(भिन्न), k_chunk_valid * परिमाण(भिन्न), nrows_गन्तव्यम्_fetch);
        }
    }
    प्रति (uint32_t s = 0; s < n_चरणाः; ++s) {
        uint32_t r = s << dma_चरणः_पङ्क्तिकाः_सरणम्;
        भिन्न *curr_buf = thread_f32_act;

        यदि (r < n_पङ्क्तिकाः) {
            curr_buf = (भिन्न *) dma_queue_निष्कासय(dma_q).dst;
        }

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_A_PREP, r);
        प्रति (uint32_t p = 0; p < (R >> 1); ++p) {
            uint32_t row_idx = r + (p << 1);
            भिन्न *pair_buf = curr_buf + (p << 1) * c_len;
            bool r0_valid = ((row_idx + 0) < n_पङ्क्तिकाः);
            bool r1_valid = ((row_idx + 1) < n_पङ्क्तिकाः);

            transfer_activation_row_pair_fp32_गन्तव्यम्_fp16_col_chunk(
                vtcm_dst, pair_buf, pair_buf + c_len, row_idx, k_खण्डम्, c_first, c_len, k_chunk_valid, r0_valid, r1_valid
            );
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_A_PREP, r);

        // Push step s + 2
        uint32_t अग्र_s = s + 2;
        uint32_t अग्र_r = अग्र_s << dma_चरणः_पङ्क्तिकाः_सरणम्;
        यदि (अग्र_r < n_पङ्क्तिकाः) {
            uint32_t nrows_गन्तव्यम्_fetch = hex_smin(n_पङ्क्तिकाः - अग्र_r, R);
            dma_queue_प्रक्षिप(dma_q, dma_make_दत्त(curr_buf, act_dma_addr + ((size_t) अग्र_r * k_stride + c_first) * परिमाण(भिन्न)),
                           c_len * परिमाण(भिन्न), k_stride * परिमाण(भिन्न), k_chunk_valid * परिमाण(भिन्न), nrows_गन्तव्यम्_fetch);
        }
    }
}

स्थिर शून्य transfer_activation_chunk_col_chunk_कार्यकर्ता_fn(अचिह्नित पूर्णाङ्क n, अचिह्नित पूर्णाङ्क i, शून्य *दत्त) {
    activation_transfer_col_chunk_state_t *st = (activation_transfer_col_chunk_state_t *) दत्त;
    संरचना htp_thread_trace * tr = &st->traces[i];

    uint32_t n_खण्डानि = st->k_खण्डम् / 32;
    uint32_t b_first = fastdiv(n_खण्डानि * i, &st->n_तन्तवः_div);
    uint32_t b_चरमः  = fastdiv(n_खण्डानि * (i + 1), &st->n_तन्तवः_div);
    uint32_t c_first = b_first * 32;
    uint32_t c_चरमः = b_चरमः * 32;
    uint32_t c_len = c_चरमः - c_first;

    यदि (c_len == 0) {
        प्रत्यागम;
    }

    uint32_t k_chunk_valid = 0;
    यदि (st->k_valid > c_first) {
        k_chunk_valid = hex_smin(st->k_valid, c_चरमः) - c_first;
    }

    __fp16 *dst = st->dst;

    size_t thread_scratch_अष्टकानि = hex_align_अधस्तात्(fastdiv(st->vtcm_f32_act_अष्टकानि, &st->n_तन्तवः_div), 128);
    भिन्न *thread_f32_act = (भिन्न *)((वर्ण *)st->vtcm_f32_act + i * thread_scratch_अष्टकानि);

    transfer_activation_chunk_fp32_गन्तव्यम्_fp16_dma_pipelined_col_chunk(
        st->ctx->dma[i], dst, st->act_dma_addr, st->n_पङ्क्तिकाः, st->k_खण्डम्, st->k_stride, k_chunk_valid,
        c_first, c_len, thread_f32_act, tr, st->dma_चरणः_पङ्क्तिकाः, st->dma_चरणः_पङ्क्तिकाः_सरणम्
    );
}

स्थिर शून्य transfer_activation_chunk_गन्तव्यम्_fp16_dma_pipelined(
        dma_queue *dma_q,
        __fp16 *सीमित vtcm_dst,
        dma_addr_t act_dma_addr,
        uint32_t n_पङ्क्तिकाः,
        uint32_t k_खण्डम्,
        uint32_t k_stride,
        uint32_t k_valid,
        uint8_t *thread_act,
        uint32_t act_elem_आकार,
        संरचना htp_thread_trace *tr,
        uint32_t dma_चरणः_पङ्क्तिकाः,
        uint32_t dma_चरणः_पङ्क्तिकाः_सरणम्) {

    ध्रुव uint32_t R = dma_चरणः_पङ्क्तिकाः;
    ध्रुव uint32_t n_पङ्क्तिकाः_padded = hex_align_ऊर्ध्वम्(n_पङ्क्तिकाः, HTP_MM_HMX_TILE_N_पङ्क्तिकाः);

    ध्रुव uint32_t n_चरणाः = n_पङ्क्तिकाः_padded >> dma_चरणः_पङ्क्तिकाः_सरणम्;

    ध्रुव bool act_is_f16 = (act_elem_आकार == परिमाण(__fp16));
    ध्रुव size_t row_अष्टकानि = (size_t) k_खण्डम् * act_elem_आकार;  // staging row (DMA dst stride)
    ध्रुव size_t src_row_अष्टकानि = (size_t) k_stride * act_elem_आकार;  // DDR row (DMA src stride)
    ध्रुव size_t विस्तारः_अष्टकानि = (size_t) k_valid * act_elem_आकार;

    // Push step 0
    यदि (n_चरणाः > 0 && n_पङ्क्तिकाः > 0) {
        uint32_t nrows_गन्तव्यम्_fetch = hex_smin(n_पङ्क्तिकाः, R);
        dma_queue_प्रक्षिप(dma_q, dma_make_दत्त(thread_act, act_dma_addr),
                       row_अष्टकानि, src_row_अष्टकानि, विस्तारः_अष्टकानि, nrows_गन्तव्यम्_fetch);
    }
    // Push step 1 (if valid)
    यदि (n_चरणाः > 1) {
        uint32_t अग्र_r = R * 1;
        यदि (अग्र_r < n_पङ्क्तिकाः) {
            uint32_t nrows_गन्तव्यम्_fetch = hex_smin(n_पङ्क्तिकाः - अग्र_r, R);
            uint8_t *अग्र_buf = thread_act + 1 * R * row_अष्टकानि;
            dma_queue_प्रक्षिप(dma_q, dma_make_दत्त(अग्र_buf, act_dma_addr + (size_t) अग्र_r * src_row_अष्टकानि),
                           row_अष्टकानि, src_row_अष्टकानि, विस्तारः_अष्टकानि, nrows_गन्तव्यम्_fetch);
        }
    }
    प्रति (uint32_t s = 0; s < n_चरणाः; ++s) {
        uint32_t r = s << dma_चरणः_पङ्क्तिकाः_सरणम्;
        uint8_t *curr_buf = thread_act;

        यदि (r < n_पङ्क्तिकाः) {
            curr_buf = (uint8_t *) dma_queue_निष्कासय(dma_q).dst;
        }

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_A_PREP, r);
        // Two copies of the pair loop so the type is resolved once per step and the
        // row-pair kernels stay direct (inlinable) calls.
        यदि (act_is_f16) {
            प्रति (uint32_t p = 0; p < (R >> 1); ++p) {
                uint32_t row_idx = r + (p << 1);
                ध्रुव __fp16 *pair_buf = (ध्रुव __fp16 *) (curr_buf + (p << 1) * row_अष्टकानि);
                bool r0_valid = ((row_idx + 0) < n_पङ्क्तिकाः);
                bool r1_valid = ((row_idx + 1) < n_पङ्क्तिकाः);

                transfer_activation_row_pair_f16_गन्तव्यम्_f16(vtcm_dst, pair_buf, pair_buf + k_खण्डम्, row_idx, k_खण्डम्, k_valid, r0_valid, r1_valid);
            }
        } अन्यथा {
            प्रति (uint32_t p = 0; p < (R >> 1); ++p) {
                uint32_t row_idx = r + (p << 1);
                ध्रुव भिन्न *pair_buf = (ध्रुव भिन्न *) (curr_buf + (p << 1) * row_अष्टकानि);
                bool r0_valid = ((row_idx + 0) < n_पङ्क्तिकाः);
                bool r1_valid = ((row_idx + 1) < n_पङ्क्तिकाः);

                transfer_activation_row_pair_fp32_गन्तव्यम्_fp16(vtcm_dst, pair_buf, pair_buf + k_खण्डम्, row_idx, k_खण्डम्, k_valid, r0_valid, r1_valid);
            }
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_A_PREP, r);

        // Push step s + 2
        uint32_t अग्र_s = s + 2;
        uint32_t अग्र_r = अग्र_s << dma_चरणः_पङ्क्तिकाः_सरणम्;
        यदि (अग्र_r < n_पङ्क्तिकाः) {
            uint32_t nrows_गन्तव्यम्_fetch = hex_smin(n_पङ्क्तिकाः - अग्र_r, R);
            dma_queue_प्रक्षिप(dma_q, dma_make_दत्त(curr_buf, act_dma_addr + (size_t) अग्र_r * src_row_अष्टकानि),
                           row_अष्टकानि, src_row_अष्टकानि, विस्तारः_अष्टकानि, nrows_गन्तव्यम्_fetch);
        }
    }
}

स्थिर शून्य transfer_activation_chunk_कार्यकर्ता_fn(अचिह्नित पूर्णाङ्क n, अचिह्नित पूर्णाङ्क i, शून्य *दत्त) {
    activation_transfer_task_state_t *st = (activation_transfer_task_state_t *) दत्त;

    संरचना htp_thread_trace * tr = &st->traces[i];

    प्रति (अचिह्नित पूर्णाङ्क task_id = i; task_id < (अचिह्नित पूर्णाङ्क)st->n_tasks; task_id += n) {
        पूर्णाङ्क    chunk_idx  = task_id * st->n_chunks_प्रत्येकम्_task;
        size_t chunk_आकार = hex_smin(st->n_tot_chunks - chunk_idx, st->n_chunks_प्रत्येकम्_task);

        __fp16      *dst = st->dst + chunk_idx * st->k_खण्डम्;
        ध्रुव dma_addr_t act_dma_addr = st->act_dma_addr + (size_t) chunk_idx * st->k_stride * st->act_elem_आकार;

        uint8_t *thread_act = (uint8_t *) st->vtcm_f32_act + i * st->vtcm_f32_act_अष्टकानि_प्रत्येकम्_thread;
        transfer_activation_chunk_गन्तव्यम्_fp16_dma_pipelined(
            st->ctx->dma[i], dst, act_dma_addr, chunk_आकार, st->k_खण्डम्, st->k_stride, st->k_valid, thread_act, st->act_elem_आकार, tr, st->dma_चरणः_पङ्क्तिकाः, st->dma_चरणः_पङ्क्तिकाः_सरणम्
        );
    }
}

प्रकारनिर्देश संरचना {
    संरचना htp_thread_trace       * traces;
    ध्रुव संरचना mmid_row_mapping * matrix_पङ्क्तिकाः;
    __fp16                        * dst;
    ध्रुव भिन्न                   * src;
    uint32_t                        n_tasks;
    uint32_t                        n_tot_chunks;
    uint32_t                        n_chunks_प्रत्येकम्_task;
    uint32_t                        k_खण्डम्;
    uint32_t                        cur_a;
    uint32_t                        mapping_stride;
    uint32_t                        ne11;
    संरचना fastdiv_मानानि           ne11_div;
    size_t                          nb11;
    size_t                          nb12;
    uint32_t                        start_row;
    uint32_t                        cne1;
    uint32_t                        k_valid;
} activation_transfer_gathered_task_state_t;

प्रकारनिर्देश संरचना {
    संरचना htp_thread_trace       * traces;
    ध्रुव संरचना mmid_row_mapping * matrix_पङ्क्तिकाः;
    ध्रुव __fp16                  * vtcm_src;
    भिन्न                         * dst;
    uint32_t                        n_tasks;
    uint32_t                        n_tot_chunks;
    uint32_t                        n_chunks_प्रत्येकम्_task;
    uint32_t                        n_स्तम्भसङ्ख्या;
    uint32_t                        cur_a;
    uint32_t                        mapping_stride;
    size_t                          dst_nb1;
    size_t                          dst_nb2;
    uint32_t                        start_row;
    uint32_t                        cne1;
} output_transfer_scattered_task_state_t;

स्थिर शून्य transfer_activation_chunk_gathered_कार्यकर्ता_fn(अचिह्नित पूर्णाङ्क n, अचिह्नित पूर्णाङ्क i, शून्य *दत्त) {
    activation_transfer_gathered_task_state_t *st = दत्त;
    संरचना htp_thread_trace * tr = &st->traces[i];
    पूर्णाङ्क chunk_idx      = i;
    पूर्णाङ्क chunk_आकार     = st->n_chunks_प्रत्येकम्_task;
    पूर्णाङ्क vtcm_start_row = chunk_idx * chunk_आकार;
    पूर्णाङ्क start_row      = st->start_row + vtcm_start_row;
    पूर्णाङ्क n_पङ्क्तिकाः         = hex_smin(st->cne1 - start_row, chunk_आकार);
    यदि (n_पङ्क्तिकाः > 0) {
        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_A_PREP, chunk_idx);
        transfer_activation_chunk_fp32_गन्तव्यम्_fp16_gathered(
            st->dst, st->src, start_row, vtcm_start_row, n_पङ्क्तिकाः, st->k_खण्डम्,
            st->matrix_पङ्क्तिकाः, st->cur_a, st->mapping_stride,
            st->ne11, &st->ne11_div, st->nb11, st->nb12, st->cne1, st->k_valid);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_A_PREP, chunk_idx);
    }
}

स्थिर शून्य transfer_activation_chunk_gathered_कार्यकर्ता_flat_fn(अचिह्नित पूर्णाङ्क n, अचिह्नित पूर्णाङ्क i, शून्य *दत्त) {
    activation_transfer_gathered_task_state_t *st = दत्त;
    संरचना htp_thread_trace * tr = &st->traces[i];
    पूर्णाङ्क chunk_idx = i;
    पूर्णाङ्क chunk_आकार = st->n_chunks_प्रत्येकम्_task;
    पूर्णाङ्क vtcm_start_row = chunk_idx * chunk_आकार;
    पूर्णाङ्क start_row = st->start_row + vtcm_start_row;
    पूर्णाङ्क n_पङ्क्तिकाः = hex_smin(st->cne1 - start_row, chunk_आकार);
    यदि (n_पङ्क्तिकाः > 0) {
        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_A_PREP, chunk_idx);
        transfer_activation_chunk_fp32_गन्तव्यम्_fp16_gathered_flat(
            st->dst, st->src, start_row, vtcm_start_row, n_पङ्क्तिकाः, st->k_खण्डम्,
            st->matrix_पङ्क्तिकाः, st->cur_a, st->mapping_stride,
            st->nb12, st->cne1, st->k_valid);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_A_PREP, chunk_idx);
    }
}

स्थिर शून्य transfer_output_chunk_scattered_कार्यकर्ता_fn(अचिह्नित पूर्णाङ्क n, अचिह्नित पूर्णाङ्क i, शून्य *दत्त) {
    output_transfer_scattered_task_state_t *st = दत्त;
    संरचना htp_thread_trace * tr = &st->traces[i];
    पूर्णाङ्क chunk_idx = i;
    पूर्णाङ्क chunk_आकार = st->n_chunks_प्रत्येकम्_task;
    पूर्णाङ्क vtcm_start_row = chunk_idx * chunk_आकार;
    पूर्णाङ्क start_row = st->start_row + vtcm_start_row;
    पूर्णाङ्क n_पङ्क्तिकाः = hex_smin(st->cne1 - start_row, chunk_आकार);
    यदि (n_पङ्क्तिकाः > 0) {
        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_O_PROC, chunk_idx);
        transfer_output_chunk_fp16_गन्तव्यम्_fp32_scattered(
            st->dst, st->vtcm_src, start_row, vtcm_start_row, n_पङ्क्तिकाः, st->n_स्तम्भसङ्ख्या,
            st->matrix_पङ्क्तिकाः, st->cur_a, st->mapping_stride,
            st->dst_nb1, st->dst_nb2, st->cne1);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_O_PROC, chunk_idx);
    }
}

// --- HMX Dispatchers & Entry Points ---

स्थिर अन्तःस्थ tiled_dequant_task_fn_t hmx_get_dequant_कार्यकर्ता_fn(पूर्णाङ्क weight_type, bool is_tiled) {
    विकल्प (weight_type) {
        पक्ष HTP_TYPE_Q4_0:   प्रत्यागम dequantize_tiled_weight_गन्तव्यम्_fp16_task_q4_0;
        पक्ष HTP_TYPE_IQ4_NL: प्रत्यागम dequantize_tiled_weight_गन्तव्यम्_fp16_task_iq4_nl;
        पक्ष HTP_TYPE_Q4_1:
        पक्ष HTP_TYPE_Q4_K:   प्रत्यागम dequantize_tiled_weight_गन्तव्यम्_fp16_task_q4_1;
        पक्ष HTP_TYPE_MXFP4:  प्रत्यागम dequantize_tiled_weight_गन्तव्यम्_fp16_task_mxfp4;
        पक्ष HTP_TYPE_Q8_0:   प्रत्यागम dequantize_tiled_weight_गन्तव्यम्_fp16_task_q8_0;
        पक्ष HTP_TYPE_Q5_K:   प्रत्यागम dequantize_tiled_weight_गन्तव्यम्_fp16_task_q5_k;
        पक्ष HTP_TYPE_Q6_K:   प्रत्यागम dequantize_tiled_weight_गन्तव्यम्_fp16_task_q6_k;
        पक्ष HTP_TYPE_Q3_K:   प्रत्यागम dequantize_tiled_weight_गन्तव्यम्_fp16_task_q3_k;
        पक्ष HTP_TYPE_Q2_K:   प्रत्यागम dequantize_tiled_weight_गन्तव्यम्_fp16_task_q2_k;
        पक्ष HTP_TYPE_Q1_0:   प्रत्यागम dequantize_tiled_weight_गन्तव्यम्_fp16_task_q1_0;
        पक्ष HTP_TYPE_Q2_0:   प्रत्यागम dequantize_tiled_weight_गन्तव्यम्_fp16_task_q2_0;
        पक्ष HTP_TYPE_F16:    प्रत्यागम is_tiled ? रिक्त : convert_f16_weight_गन्तव्यम्_fp16_tiles_task;
        पक्ष HTP_TYPE_BF16:   प्रत्यागम convert_bf16_weight_गन्तव्यम्_fp16_tiles_task;
        पक्ष HTP_TYPE_F32:    प्रत्यागम quantize_f32_weight_गन्तव्यम्_fp16_tiles_task;
        स्वभाव:              प्रत्यागम रिक्त;
    }
}

स्थिर शून्य dequantize_tiled_weight_chunk_गन्तव्यम्_fp16_tiles(
        संरचना htp_context *ctx, __fp16 *vtcm_dst,
        ध्रुव शून्य *weight_src_ddr,
        पूर्णाङ्क n_स्तम्भसङ्ख्या, पूर्णाङ्क k_खण्डम्,
        size_t row_stride, पूर्णाङ्क weight_type,
        पूर्णाङ्क n_k_tiles, संरचना fastdiv_मानानि n_k_tiles_div,
        tiled_dequant_task_fn_t dequant_कार्यकर्ता_fn, पूर्णाङ्क n_तन्तवः) {

    यदि (!dequant_कार्यकर्ता_fn) {
        प्रत्यागम;
    }

    प्रतिज्ञा(k_खण्डम् % HTP_MM_HMX_TILE_N_स्तम्भसङ्ख्या == 0);

    ध्रुव size_t n_col_tiles = hmx_ceil_div(n_स्तम्भसङ्ख्या, HTP_MM_HMX_TILE_N_स्तम्भसङ्ख्या);
    ध्रुव uint32_t n_weight_tiles = k_खण्डम् / htp_tensor_get_tile_k(weight_type);
    ध्रुव size_t n_tot_tiles = n_col_tiles * n_weight_tiles;

    size_t n_tiles_प्रत्येकम्_task = (n_तन्तवः == 1) ? n_tot_tiles : hmx_ceil_div(n_tot_tiles, n_तन्तवः);

    tiled_dequantize_state_t state;
    state.n_tasks          = (n_tot_tiles + n_tiles_प्रत्येकम्_task - 1) / n_tiles_प्रत्येकम्_task;
    state.n_tot_tiles      = n_tot_tiles;
    state.n_tiles_प्रत्येकम्_task = n_tiles_प्रत्येकम्_task;
    state.dst              = vtcm_dst;
    state.src              = (ध्रुव uint8_t *)weight_src_ddr;
    state.task_fn          = dequant_कार्यकर्ता_fn;
    state.n_स्तम्भसङ्ख्या           = n_स्तम्भसङ्ख्या;
    state.k_खण्डम्          = k_खण्डम्;
    state.row_stride       = row_stride;
    state.weight_type      = weight_type;
    state.n_k_tiles        = n_k_tiles;
    state.n_weight_tiles   = n_weight_tiles;
    state.n_k_tiles_div    = n_k_tiles_div;
    state.traces           = ctx->trace;
    state.ctx              = ctx;

    state.tile_आकार = htp_tensor_get_tile_आकार(weight_type);
    state.aligned_tile_आकार = htp_tensor_get_tile_padded_आकार(weight_type);

    यदि (state.n_tasks == 1 || n_तन्तवः == 1) {
        dequantize_tiled_कार्यकर्ता_loop(1, 0, &state);
    } अन्यथा {
        पूर्णाङ्क n_tasks = hex_smin((पूर्णाङ्क) state.n_tasks, n_तन्तवः);
        कार्यकर्ता_pool_सततम्_func(ctx->कार्यकर्ता_pool, dequantize_tiled_कार्यकर्ता_loop, &state, n_tasks);
    }
}

प्रकारनिर्देश संरचना {
    संरचना htp_context      * ctx;
    संरचना htp_thread_trace * traces;
    भिन्न                   * dst;
    ध्रुव __fp16            * vtcm_src;
    ध्रुव भिन्न             * src2;
    uint32_t                  n_पङ्क्तिकाः;
    uint32_t                  n_स्तम्भसङ्ख्या;
    uint32_t                  dst_stride;
    uint32_t                  src2_stride;
    uint32_t                  dst_स्तम्भसङ्ख्या;
    संरचना fastdiv_मानानि     n_तन्तवः_div;
} output_transfer_col_chunk_state_t;

स्थिर शून्य transfer_output_chunk_col_chunk_कार्यकर्ता_fn(अचिह्नित पूर्णाङ्क n, अचिह्नित पूर्णाङ्क i, शून्य *दत्त) {
    (शून्य) n;
    output_transfer_col_chunk_state_t *st = (output_transfer_col_chunk_state_t *) दत्त;
    संरचना htp_thread_trace * tr = &st->traces[i];

    // Round up: the last block is partial when N is not 32-aligned. Its pad
    // columns are dropped by the dst_cols clamp inside the store.
    uint32_t n_खण्डानि = hmx_ceil_div(st->n_स्तम्भसङ्ख्या, 32);
    uint32_t b_first  = fastdiv(n_खण्डानि * i, &st->n_तन्तवः_div);
    uint32_t b_चरमः   = fastdiv(n_खण्डानि * (i + 1), &st->n_तन्तवः_div);
    uint32_t c_first  = b_first * 32;
    uint32_t c_चरमः   = b_चरमः * 32;
    uint32_t c_len    = c_चरमः - c_first;

    यदि (c_len == 0) प्रत्यागम;

    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_O_PROC, c_first);

    ध्रुव __fp16 *vtcm_src = st->vtcm_src + b_first * HTP_MM_HMX_TILE_N_ELMS;
    ध्रुव भिन्न *src2      = st->src2 ? (st->src2 + c_first) : रिक्त;
    भिन्न *dst             = st->dst + c_first;

    पूर्णाङ्क chunk_dst_स्तम्भसङ्ख्या = (पूर्णाङ्क)st->dst_स्तम्भसङ्ख्या - (पूर्णाङ्क)c_first;
    यदि (chunk_dst_स्तम्भसङ्ख्या > 0) {
        transfer_output_chunk_fp16_गन्तव्यम्_fp32_col_chunk(
            dst, src2, vtcm_src, 0, st->n_पङ्क्तिकाः, c_len, st->n_स्तम्भसङ्ख्या,
            st->dst_stride, st->src2_stride, (uint32_t)chunk_dst_स्तम्भसङ्ख्या
        );
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_O_PROC, c_first);
}

स्थिर शून्य transfer_output_chunk_threaded(संरचना htp_context *ctx, भिन्न *dst, ध्रुव भिन्न *src2, ध्रुव __fp16 *vtcm_src,
                                              पूर्णाङ्क n_पङ्क्तिकाः, पूर्णाङ्क n_स्तम्भसङ्ख्या, पूर्णाङ्क dst_stride, uint32_t src2_stride, पूर्णाङ्क dst_स्तम्भसङ्ख्या, पूर्णाङ्क n_तन्तवः) {
    यदि (n_पङ्क्तिकाः <= 0) प्रत्यागम;

    uint32_t n_खण्डानि = hmx_ceil_div((uint32_t) n_स्तम्भसङ्ख्या, 32);
    यदि (n_तन्तवः > 1 && n_खण्डानि >= (uint32_t)n_तन्तवः) {
        संरचना fastdiv_मानानि n_तन्तवः_div = (n_तन्तवः == (पूर्णाङ्क)ctx->n_तन्तवः) ? ctx->n_तन्तवः_div : init_fastdiv_मानानि(n_तन्तवः);
        output_transfer_col_chunk_state_t col_state;
        col_state.dst = dst;
        col_state.src2 = src2;
        col_state.vtcm_src = vtcm_src;
        col_state.n_पङ्क्तिकाः = (uint32_t)n_पङ्क्तिकाः;
        col_state.n_स्तम्भसङ्ख्या = (uint32_t)n_स्तम्भसङ्ख्या;
        col_state.dst_stride = (uint32_t)dst_stride;
        col_state.src2_stride = src2_stride;
        col_state.dst_स्तम्भसङ्ख्या = (uint32_t)dst_स्तम्भसङ्ख्या;
        col_state.n_तन्तवः_div = n_तन्तवः_div;
        col_state.traces = ctx->trace;
        col_state.ctx = ctx;

        कार्यकर्ता_pool_सततम्_func(ctx->कार्यकर्ता_pool, transfer_output_chunk_col_chunk_कार्यकर्ता_fn, &col_state, n_तन्तवः);
        प्रत्यागम;
    }

    size_t n_tot_chunks      = n_पङ्क्तिकाः;
    size_t n_chunks_प्रत्येकम्_task = (n_तन्तवः == 1) ? n_tot_chunks : hmx_ceil_div(n_पङ्क्तिकाः, n_तन्तवः);
    n_chunks_प्रत्येकम्_task        = hex_align_ऊर्ध्वम्(n_chunks_प्रत्येकम्_task, 2);

    पूर्णाङ्क actual_तन्तवः = hmx_ceil_div(n_पङ्क्तिकाः, n_chunks_प्रत्येकम्_task);

    output_transfer_task_state_t state;
    state.n_tasks           = actual_तन्तवः;
    state.n_tot_chunks      = n_tot_chunks;
    state.n_chunks_प्रत्येकम्_task = n_chunks_प्रत्येकम्_task;
    state.dst               = dst;
    state.src2              = src2;
    state.vtcm_src          = vtcm_src;
    state.n_स्तम्भसङ्ख्या            = n_स्तम्भसङ्ख्या;
    state.dst_stride        = dst_stride;
    state.src2_stride       = src2_stride;
    state.dst_स्तम्भसङ्ख्या          = dst_स्तम्भसङ्ख्या;
    state.traces            = ctx->trace;

    यदि (actual_तन्तवः <= 1) {
        transfer_output_chunk_कार्यकर्ता_fn(1, 0, &state);
    } अन्यथा {
        कार्यकर्ता_pool_सततम्_func(ctx->कार्यकर्ता_pool, transfer_output_chunk_कार्यकर्ता_fn, &state, actual_तन्तवः);
    }
}

संरचना activation_transfer_params {
    संरचना htp_context *          ctx;
    __fp16 *                      dst;
    dma_addr_t                    act_dma_addr;
    पूर्णाङ्क                           n_पङ्क्तिकाः;
    पूर्णाङ्क                           k_खण्डम्;
    पूर्णाङ्क                           k_stride;
    पूर्णाङ्क                           n_तन्तवः;
    ध्रुव संरचना fastdiv_मानानि * act_तन्तवः_div;
    ध्रुव संरचना fastdiv_मानानि * k_div;
    पूर्णाङ्क                           k_valid;
    भिन्न *                       vtcm_f32_act;
    size_t                        vtcm_f32_act_अष्टकानि;
    uint32_t                      act_elem_आकार;  // 4=F32 src1, 2=F16 src1
};

स्थिर शून्य transfer_activation_chunk_threaded(ध्रुव संरचना activation_transfer_params * params) {
    संरचना htp_context *          ctx                = params->ctx;
    __fp16 *                      dst                = params->dst;
    ध्रुव dma_addr_t              act_dma_addr       = params->act_dma_addr;
    पूर्णाङ्क                           n_पङ्क्तिकाः             = params->n_पङ्क्तिकाः;
    पूर्णाङ्क                           k_खण्डम्            = params->k_खण्डम्;
    पूर्णाङ्क                           k_stride           = params->k_stride;
    पूर्णाङ्क                           n_तन्तवः          = params->n_तन्तवः;
    ध्रुव संरचना fastdiv_मानानि * act_तन्तवः_div    = params->act_तन्तवः_div;
    ध्रुव संरचना fastdiv_मानानि * k_div              = params->k_div;
    पूर्णाङ्क                           k_valid            = params->k_valid;
    भिन्न *                       vtcm_f32_act       = params->vtcm_f32_act;
    size_t                        vtcm_f32_act_अष्टकानि = params->vtcm_f32_act_अष्टकानि;
    // element size of the activation rows (4 = F32, 2 = F16).
    ध्रुव uint32_t                act_elem_आकार      = params->act_elem_आकार ? params->act_elem_आकार : (uint32_t) परिमाण(भिन्न);

    यदि (n_पङ्क्तिकाः <= 0) {
        प्रत्यागम;
    }

    ध्रुव size_t n_tasks = (n_पङ्क्तिकाः + 31) >> 5;
    यदि (act_elem_आकार == परिमाण(भिन्न) && n_तन्तवः > 1 && k_खण्डम् > 32 && n_tasks < (size_t) n_तन्तवः) {
        // Calculate step rows parameters for column-chunked dma pipelining
        uint32_t dma_चरणः_पङ्क्तिकाः = 2;
        uint32_t dma_चरणः_पङ्क्तिकाः_सरणम् = 1;
        यदि (vtcm_f32_act_अष्टकानि > 0) {
            size_t thread_scratch_अष्टकानि = hex_align_अधस्तात्(fastdiv(vtcm_f32_act_अष्टकानि, act_तन्तवः_div), 128);
            size_t thread_scratch_elements = thread_scratch_अष्टकानि / परिमाण(भिन्न);
            size_t dma_चरणः_पङ्क्तिकाः_max = fastdiv(thread_scratch_elements / 2, k_div);
            यदि (dma_चरणः_पङ्क्तिकाः_max >= 4) {
                dma_चरणः_पङ्क्तिकाः = 4;
                dma_चरणः_पङ्क्तिकाः_सरणम् = 2;
            }
        }

        activation_transfer_col_chunk_state_t col_state;
        col_state.dst = dst;
        col_state.act_dma_addr = act_dma_addr;
        col_state.n_पङ्क्तिकाः = n_पङ्क्तिकाः;
        col_state.k_खण्डम् = k_खण्डम्;
        col_state.k_stride = k_stride;
        col_state.k_valid = k_valid;
        col_state.n_col_chunks = n_तन्तवः;
        col_state.n_तन्तवः_div = *act_तन्तवः_div;
        col_state.vtcm_f32_act = vtcm_f32_act;
        col_state.vtcm_f32_act_अष्टकानि = vtcm_f32_act_अष्टकानि;
        col_state.traces = ctx->trace;
        col_state.ctx = ctx;
        col_state.dma_चरणः_पङ्क्तिकाः = dma_चरणः_पङ्क्तिकाः;
        col_state.dma_चरणः_पङ्क्तिकाः_सरणम् = dma_चरणः_पङ्क्तिकाः_सरणम्;

        कार्यकर्ता_pool_सततम्_func(ctx->कार्यकर्ता_pool, transfer_activation_chunk_col_chunk_कार्यकर्ता_fn, &col_state, n_तन्तवः);
        प्रत्यागम;
    }

    प्रतिज्ञा(k_खण्डम् % HTP_MM_HMX_TILE_N_स्तम्भसङ्ख्या == 0 && k_stride % HTP_MM_HMX_TILE_N_स्तम्भसङ्ख्या == 0);

    size_t n_tot_chunks      = n_पङ्क्तिकाः;
    size_t n_chunks_प्रत्येकम्_task = (n_तन्तवः == 1) ? n_tot_chunks : 32;  // must be multiple of 32 to ensure correct destination address

    activation_transfer_task_state_t state;
    state.n_tasks            = (n_तन्तवः == 1) ? 1 : hmx_ceil_div(n_tot_chunks, 32);
    state.n_tot_chunks       = n_tot_chunks;
    state.n_chunks_प्रत्येकम्_task  = n_chunks_प्रत्येकम्_task;
    state.dst                = dst;
    state.act_dma_addr       = act_dma_addr;
    state.k_खण्डम्            = k_खण्डम्;
    state.k_stride           = k_stride;
    state.k_valid            = k_valid;
    state.traces             = ctx->trace;
    state.ctx                = ctx;
    state.vtcm_f32_act       = vtcm_f32_act;
    state.act_elem_आकार = act_elem_आकार;

    state.vtcm_f32_act_अष्टकानि_प्रत्येकम्_thread = hex_align_अधस्तात्(fastdiv(vtcm_f32_act_अष्टकानि, act_तन्तवः_div), 128);

    uint32_t dma_चरणः_पङ्क्तिकाः = 2;
    uint32_t dma_चरणः_पङ्क्तिकाः_सरणम् = 1;
    यदि (state.vtcm_f32_act_अष्टकानि_प्रत्येकम्_thread > 0) {
        size_t thread_scratch_elements = state.vtcm_f32_act_अष्टकानि_प्रत्येकम्_thread / परिमाण(भिन्न);
        size_t dma_चरणः_पङ्क्तिकाः_max = fastdiv(thread_scratch_elements / 2, k_div);
        यदि (dma_चरणः_पङ्क्तिकाः_max >= 4) {
            dma_चरणः_पङ्क्तिकाः = 4;
            dma_चरणः_पङ्क्तिकाः_सरणम् = 2;
        }
    }
    state.dma_चरणः_पङ्क्तिकाः      = dma_चरणः_पङ्क्तिकाः;
    state.dma_चरणः_पङ्क्तिकाः_सरणम् = dma_चरणः_पङ्क्तिकाः_सरणम्;

    पूर्णाङ्क सक्रिय_तन्तवः = hex_smin(n_तन्तवः, (पूर्णाङ्क)state.n_tasks);
    यदि (state.n_tasks == 1 || n_तन्तवः == 1) {
        transfer_activation_chunk_कार्यकर्ता_fn(1, 0, &state);
    } अन्यथा {
        कार्यकर्ता_pool_सततम्_func(ctx->कार्यकर्ता_pool, transfer_activation_chunk_कार्यकर्ता_fn, &state, सक्रिय_तन्तवः);
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

स्थिर शून्य hmx_matmul_कार्यकर्ता_fn(शून्य * दत्त) {
    hmx_matmul_job_t * job = (hmx_matmul_job_t *) दत्त;
    FARF(उच्चः_2, "hmx-mm-job: n_row_tiles %u n_col_tiles %u n_dot_tiles %u", job->n_row_tiles, job->n_col_tiles, job->n_dot_tiles);
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
                                  भिन्न *सीमित dst,
                                  dma_addr_t src2_addr,
                                  size_t src2_अष्टकानि,
                                  dma_addr_t act_dma_addr,
                                  dma_addr_t weight,
                                  पूर्णाङ्क m, पूर्णाङ्क k, पूर्णाङ्क n,
                                  पूर्णाङ्क act_stride,
                                  uint32_t act_elem_आकार,  // 4=F32 src1, 2=F16 src1
                                  पूर्णाङ्क weight_stride,
                                  पूर्णाङ्क weight_type,
                                  पूर्णाङ्क k_valid,
                                  पूर्णाङ्क dst_stride,
                                  uint32_t src2_stride,
                                  पूर्णाङ्क dst_स्तम्भसङ्ख्या,
                                  पूर्णाङ्क m_chunk,
                                  पूर्णाङ्क n_chunk,
                                  पूर्णाङ्क pipeline,
                                  पूर्णाङ्क n_तन्तवः,
                                  पूर्णाङ्क act_तन्तवः,
                                  ध्रुव संरचना fastdiv_मानानि * act_तन्तवः_div,
                                  ध्रुव संरचना fastdiv_मानानि * k_div,
                                  पूर्णाङ्क tile_आकार,
                                  पूर्णाङ्क aligned_tile_आकार,
                                  पूर्णाङ्क vtcm_आकार) {
    संरचना htp_thread_trace * tr = &ctx->trace[0];
    htp_trace_event_start(tr, HTP_TRACE_EVT_INIT, 0);

    ध्रुव bool is_tiled = (tile_आकार > 0);

    यदि (k % 32 != 0) {
        प्रत्यागम -1;
    }
    यदि (!hex_is_aligned(dst, परिमाण(भिन्न))) { प्रत्यागम -1; }

    size_t row_stride = htp_tensor_get_row_आकार(weight_type, k);
    यदि (row_stride == 0) {
        प्रत्यागम -1;
    }

    tiled_dequant_task_fn_t dequant_कार्यकर्ता_fn = hmx_get_dequant_कार्यकर्ता_fn(weight_type, is_tiled);

    ध्रुव पूर्णाङ्क n_k_tiles = k / HTP_MM_HMX_TILE_N_स्तम्भसङ्ख्या;
    ध्रुव पूर्णाङ्क n_weight_tiles = k / htp_tensor_get_tile_k(weight_type);
    ध्रुव संरचना fastdiv_मानानि n_k_tiles_div = init_fastdiv_मानानि(n_k_tiles);

    ध्रुव size_t vec_dot_आकार = k * परिमाण(__fp16);
    ध्रुव size_t vtcm_budget  = ctx->vtcm_आकार;

    ध्रुव uint32_t dma_dst_stride  = is_tiled ? aligned_tile_आकार : row_stride;
    ध्रुव uint32_t dma_src_stride  = is_tiled ? tile_आकार : weight_stride;
    ध्रुव uint32_t dma_विस्तारः_अष्टकानि = is_tiled ? tile_आकार : row_stride;

    size_t m_chunk_n_पङ्क्तिकाः = m_chunk;
    size_t n_chunk_n_स्तम्भसङ्ख्या = n_chunk;
    size_t vtcm_प्रयुक्तम्      = vtcm_आकार;

    संरचना htp_mm_hmx_vtcm_layout L;
    htp_mm_hmx_vtcm_layout_build(&L, HTP_MM_KERNEL_HMX_2D, weight_type, k, m_chunk_n_पङ्क्तिकाः, n_chunk_n_स्तम्भसङ्ख्या, 1, pipeline, act_तन्तवः, aligned_tile_आकार, src2_अष्टकानि);

    vtcm_प्रयुक्तम् = L.योग_अष्टकानि;
    यदि (vtcm_प्रयुक्तम् > vtcm_budget) {
        FARF(ERROR, "hmx-mm-2d-precomputed: VTCM overflow: used %zu budget %zu, m %d k %d n %d mc %zu nc %zu",
             vtcm_प्रयुक्तम्, vtcm_budget, m, k, n, m_chunk_n_पङ्क्तिकाः, n_chunk_n_स्तम्भसङ्ख्या);
        प्रत्यागम -1;
    }

    uint8_t * ध्रुव आधारः = (uint8_t *) ctx->vtcm_आधारः;
    __fp16  *vtcm_weight_raw[2] = {
        VTCM_LAYOUT_PTR(__fp16, आधारः, L.off_weight[0]),
        VTCM_LAYOUT_PTR_OPTIONAL(__fp16, आधारः, L.off_weight[1], pipeline)
    };
    __fp16  *vtcm_f16_act    = VTCM_LAYOUT_PTR(__fp16, आधारः, L.off_act);
    भिन्न   *vtcm_f32_act    = VTCM_LAYOUT_PTR(भिन्न, आधारः, L.off_act_f32);
    __fp16  *vtcm_output     = VTCM_LAYOUT_PTR(__fp16, आधारः, L.off_dst[0]);
    शून्य    *vtcm_scratch0   = VTCM_LAYOUT_PTR(शून्य, आधारः, L.off_scratch[0]);
    शून्य    *vtcm_scratch1   = VTCM_LAYOUT_PTR_OPTIONAL(शून्य, आधारः, L.off_scratch[1], pipeline);
    शून्य    *vtcm_scratch2   = VTCM_LAYOUT_PTR_OPTIONAL(शून्य, आधारः, L.off_dst[1], pipeline);
    __fp16  *vtcm_scales     = VTCM_LAYOUT_PTR(__fp16, आधारः, L.off_scales);

    hmx_init_column_scales(vtcm_scales, Q6_V_vsplat_R(0x3c00));  // scale: 1.0, bias: 0.0 in FP16

    ध्रुव bool has_bias      = (src2_अष्टकानि > 0 && src2_addr != 0);
    भिन्न   *vtcm_bias       = VTCM_LAYOUT_PTR_OPTIONAL(भिन्न, आधारः, L.off_bias, has_bias);
    यदि (has_bias) {
        dma_queue_प्रक्षिप(weight_dma, dma_make_दत्त(vtcm_bias, src2_addr), hex_align_ऊर्ध्वम्(src2_अष्टकानि, 128), 0, src2_अष्टकानि, 1);
        dma_queue_निष्कासय(weight_dma);
    }

    FARF(उच्चः_2, "hmx-mm-2d: m %d k %d n %d wtype %d mc %zu nc %zu vtcm %zu/%zu",
         m, k, n, weight_type, m_chunk_n_पङ्क्तिकाः, n_chunk_n_स्तम्भसङ्ख्या, vtcm_प्रयुक्तम्, vtcm_budget);

    पूर्णाङ्क n_chunk_cnt = hmx_ceil_div(n, n_chunk_n_स्तम्भसङ्ख्या);

    htp_trace_event_stop(tr, HTP_TRACE_EVT_INIT, 0);

    यदि (pipeline) {
        // --- Asynchronous Pipelined Loop ---
        hmx_matmul_job_t job_slots[2];  // persistent double-buffered job descriptors
        शून्य *vtcm_raw_bufs[4] = {
            vtcm_weight_raw[0],
            vtcm_weight_raw[1],
            dequant_कार्यकर्ता_fn ? vtcm_weight_raw[0] : vtcm_scratch0,
            dequant_कार्यकर्ता_fn ? vtcm_weight_raw[1] : vtcm_scratch1,
        };
        शून्य *vtcm_weight_bufs[4] = {
            dequant_कार्यकर्ता_fn ? vtcm_scratch0 : vtcm_weight_raw[0],
            dequant_कार्यकर्ता_fn ? vtcm_scratch1 : vtcm_weight_raw[1],
            vtcm_scratch0,
            vtcm_scratch1,
        };
        शून्य *vtcm_output_bufs[2] = { vtcm_output, vtcm_scratch2 };

        प्रति (size_t mr = 0; mr < m; mr += m_chunk_n_पङ्क्तिकाः) {
            ध्रुव size_t n_पङ्क्तिकाः = hex_smin(m - mr, m_chunk_n_पङ्क्तिकाः);

            संरचना activation_transfer_params act_params = {
                .ctx = ctx,
                .dst = vtcm_f16_act,
                .act_dma_addr = act_dma_addr + (size_t) mr * act_stride * act_elem_आकार,  // byte offset (F16/F32)
                .n_पङ्क्तिकाः = (पूर्णाङ्क) n_पङ्क्तिकाः,
                .k_खण्डम् = k,
                .k_stride = act_stride,
                .n_तन्तवः = act_तन्तवः,
                .act_तन्तवः_div = act_तन्तवः_div,
                .k_div = k_div,
                .k_valid = k_valid,
                .vtcm_f32_act = vtcm_f32_act,
                .vtcm_f32_act_अष्टकानि = L.act_f32_अष्टकानि,
                .act_elem_आकार = act_elem_आकार,
            };
            transfer_activation_chunk_threaded(&act_params);

            // Prologue: push A0 and optionally A1 (if n_chunk_cnt > 1)
            ध्रुव size_t   n_स्तम्भसङ्ख्या_A0 = hex_smin(n - 0 * n_chunk_n_स्तम्भसङ्ख्या, n_chunk_n_स्तम्भसङ्ख्या);
            ध्रुव uint32_t height_A0 = is_tiled ? hmx_ceil_div(n_स्तम्भसङ्ख्या_A0, 32) * n_weight_tiles : n_स्तम्भसङ्ख्या_A0;
            dma_queue_प्रक्षिप(weight_dma, dma_make_दत्त(vtcm_raw_bufs[0], weight),
                           dma_dst_stride, dma_src_stride, dma_विस्तारः_अष्टकानि, height_A0);

            यदि (1 < n_chunk_cnt) {
                ध्रुव size_t   n_स्तम्भसङ्ख्या_A1 = hex_smin(n - 1 * n_chunk_n_स्तम्भसङ्ख्या, n_chunk_n_स्तम्भसङ्ख्या);
                ध्रुव uint32_t height_A1 = is_tiled ? hmx_ceil_div(n_स्तम्भसङ्ख्या_A1, 32) * n_weight_tiles : n_स्तम्भसङ्ख्या_A1;
                dma_queue_प्रक्षिप(weight_dma, dma_make_दत्त(vtcm_raw_bufs[1], weight + n_chunk_n_स्तम्भसङ्ख्या * weight_stride),
                               dma_dst_stride, dma_src_stride, dma_विस्तारः_अष्टकानि, height_A1);
            }

            // Main loop: pop A_i -> dequantize A_i -> push A_{i+2} -> submit C_i -> wait C_{i-1} and store D_{i-1}
            प्रति (पूर्णाङ्क i = 0; i < n_chunk_cnt; ++i) {
                ध्रुव size_t nc    = i * n_chunk_n_स्तम्भसङ्ख्या;
                ध्रुव size_t nc_p2 = nc + 2 * n_chunk_n_स्तम्भसङ्ख्या;

                ध्रुव size_t n_स्तम्भसङ्ख्या    = hex_smin(n - nc, n_chunk_n_स्तम्भसङ्ख्या);
                ध्रुव size_t n_स्तम्भसङ्ख्या_p2 = hex_smin(n - nc_p2, n_chunk_n_स्तम्भसङ्ख्या);

                // 1. pop A_i
                शून्य * curr_raw = (शून्य *) dma_queue_निष्कासय(weight_dma).dst;

                // 2. dequantize A_i
                dequantize_tiled_weight_chunk_गन्तव्यम्_fp16_tiles(
                    ctx, vtcm_weight_bufs[i % 4], curr_raw,
                    n_स्तम्भसङ्ख्या, k, row_stride, weight_type,
                    n_k_tiles, n_k_tiles_div, dequant_कार्यकर्ता_fn, n_तन्तवः);

                // 3. push A_{i+2} (if i+2 < n_chunk_cnt)
                यदि (i + 2 < n_chunk_cnt) {
                    ध्रुव uint32_t height_p2 = is_tiled ? hmx_ceil_div(n_स्तम्भसङ्ख्या_p2, 32) * n_weight_tiles : n_स्तम्भसङ्ख्या_p2;
                    dma_queue_प्रक्षिप(weight_dma, dma_make_दत्त(vtcm_raw_bufs[(i + 2) % 4], weight + nc_p2 * weight_stride),
                                   dma_dst_stride, dma_src_stride, dma_विस्तारः_अष्टकानि, height_p2);
                }

                // 4. submit C_i
                hmx_matmul_job_init(&job_slots[i % 2], (__fp16 *) vtcm_output_bufs[i % 2],
                                    (__fp16 *) vtcm_f16_act, (__fp16 *) vtcm_weight_bufs[i % 4],
                                    vtcm_scales, hmx_ceil_div(n_पङ्क्तिकाः, HTP_MM_HMX_TILE_N_पङ्क्तिकाः),
                                    hmx_ceil_div(n_स्तम्भसङ्ख्या, HTP_MM_HMX_TILE_N_स्तम्भसङ्ख्या), k / HTP_MM_HMX_TILE_N_पङ्क्तिकाः);
                hmx_queue_प्रक्षिप(ctx->hmx_queue, hmx_queue_make_desc(hmx_matmul_कार्यकर्ता_fn, &job_slots[i % 2]));

                // 5. wait C_{i-1} and store D_{i-1} (multi-thread HVX, parallel with C_i)
                यदि (i > 0) {
                    hmx_queue_निष्कासय(ctx->hmx_queue);
                    ध्रुव size_t nc_पूर्व = (i - 1) * n_chunk_n_स्तम्भसङ्ख्या;
                    ध्रुव size_t n_स्तम्भसङ्ख्या_पूर्व = hex_smin(n - nc_पूर्व, n_chunk_n_स्तम्भसङ्ख्या);
                    भिन्न *output_chunk = dst + (mr * dst_stride + nc_पूर्व);
                    ध्रुव भिन्न *bias_chunk = has_bias ? (vtcm_bias + mr * src2_stride + nc_पूर्व) : रिक्त;
                    पूर्णाङ्क chunk_dst_स्तम्भसङ्ख्या = dst_स्तम्भसङ्ख्या - (पूर्णाङ्क)nc_पूर्व;
                    यदि (chunk_dst_स्तम्भसङ्ख्या > 0) {
                        transfer_output_chunk_threaded(ctx, output_chunk, bias_chunk, vtcm_output_bufs[(i - 1) % 2], n_पङ्क्तिकाः, n_स्तम्भसङ्ख्या_पूर्व, dst_stride, src2_stride, chunk_dst_स्तम्भसङ्ख्या, n_तन्तवः);
                    }
                }
            }

            // Epilogue: wait C_{last} and store D_{last}
            hmx_queue_निष्कासय(ctx->hmx_queue);
            ध्रुव size_t nc_चरमः = (n_chunk_cnt - 1) * n_chunk_n_स्तम्भसङ्ख्या;
            ध्रुव size_t n_स्तम्भसङ्ख्या_चरमः = hex_smin(n - nc_चरमः, n_chunk_n_स्तम्भसङ्ख्या);
            भिन्न *output_chunk = dst + (mr * dst_stride + nc_चरमः);
            ध्रुव भिन्न *bias_chunk = has_bias ? (vtcm_bias + mr * src2_stride + nc_चरमः) : रिक्त;
            पूर्णाङ्क chunk_dst_स्तम्भसङ्ख्या = dst_स्तम्भसङ्ख्या - (पूर्णाङ्क)nc_चरमः;
            यदि (chunk_dst_स्तम्भसङ्ख्या > 0) {
                transfer_output_chunk_threaded(ctx, output_chunk, bias_chunk, vtcm_output_bufs[(n_chunk_cnt - 1) % 2], n_पङ्क्तिकाः, n_स्तम्भसङ्ख्या_चरमः, dst_stride, src2_stride, chunk_dst_स्तम्भसङ्ख्या, n_तन्तवः);
            }
        }
    } अन्यथा {
        // --- Synchronous loop (m <= 32 or fallback) ---
        hmx_matmul_job_t job;
        शून्य *vtcm_raw_bufs[2] = { vtcm_weight_raw[0], dequant_कार्यकर्ता_fn ? vtcm_weight_raw[0] : vtcm_scratch0 };
        शून्य *vtcm_weight_bufs[2] = { dequant_कार्यकर्ता_fn ? vtcm_scratch0 : vtcm_weight_raw[0], vtcm_scratch0 };

        प्रति (size_t mr = 0; mr < m; mr += m_chunk_n_पङ्क्तिकाः) {
            ध्रुव size_t n_पङ्क्तिकाः = hex_smin(m - mr, m_chunk_n_पङ्क्तिकाः);

            संरचना activation_transfer_params act_params = {
                .ctx = ctx,
                .dst = vtcm_f16_act,
                .act_dma_addr = act_dma_addr + (size_t) mr * act_stride * act_elem_आकार,  // byte offset (F16/F32)
                .n_पङ्क्तिकाः = (पूर्णाङ्क) n_पङ्क्तिकाः,
                .k_खण्डम् = k,
                .k_stride = act_stride,
                .n_तन्तवः = act_तन्तवः,
                .act_तन्तवः_div = act_तन्तवः_div,
                .k_div = k_div,
                .k_valid = k_valid,
                .vtcm_f32_act = vtcm_f32_act,
                .vtcm_f32_act_अष्टकानि = L.act_f32_अष्टकानि,
                .act_elem_आकार = act_elem_आकार,
            };
            transfer_activation_chunk_threaded(&act_params);

            // A0: Pre-fetch the first weight chunk (nc = 0)
            यदि (n > 0) {
                ध्रुव size_t n_स्तम्भसङ्ख्या = hex_smin(n, n_chunk_n_स्तम्भसङ्ख्या);
                ध्रुव uint32_t height = is_tiled ? hmx_ceil_div(n_स्तम्भसङ्ख्या, 32) * n_weight_tiles : n_स्तम्भसङ्ख्या;
                dma_queue_प्रक्षिप(weight_dma, dma_make_दत्त(vtcm_raw_bufs[0], weight), dma_dst_stride, dma_src_stride, dma_विस्तारः_अष्टकानि, height);
            }

            प्रति (पूर्णाङ्क i = 0; i < n_chunk_cnt; ++i) {
                ध्रुव size_t nc = (size_t) i * n_chunk_n_स्तम्भसङ्ख्या;
                ध्रुव size_t n_स्तम्भसङ्ख्या = hex_smin(n - nc, n_chunk_n_स्तम्भसङ्ख्या);
                ध्रुव size_t n_row_tiles = hmx_ceil_div(n_पङ्क्तिकाः, HTP_MM_HMX_TILE_N_पङ्क्तिकाः);
                ध्रुव size_t n_col_tiles = hmx_ceil_div(n_स्तम्भसङ्ख्या, HTP_MM_HMX_TILE_N_स्तम्भसङ्ख्या);

                // A: Wait for weight DMA
                शून्य * curr_raw = (शून्य *) dma_queue_निष्कासय(weight_dma).dst;

                // B: Weight Dequantize (Threaded)
                dequantize_tiled_weight_chunk_गन्तव्यम्_fp16_tiles(
                    ctx, vtcm_weight_bufs[i % 2], curr_raw,
                    n_स्तम्भसङ्ख्या, k, row_stride, weight_type,
                    n_k_tiles, n_k_tiles_div, dequant_कार्यकर्ता_fn, n_तन्तवः);

                // Start weight DMA for the next chunk early
                ध्रुव size_t nc_अग्र = nc + n_chunk_n_स्तम्भसङ्ख्या;
                यदि (nc_अग्र < n) {
                    ध्रुव size_t n_स्तम्भसङ्ख्या_अग्र = hex_smin(n - nc_अग्र, n_chunk_n_स्तम्भसङ्ख्या);
                    ध्रुव uint32_t height_अग्र = is_tiled ? hmx_ceil_div(n_स्तम्भसङ्ख्या_अग्र, 32) * n_weight_tiles : n_स्तम्भसङ्ख्या_अग्र;
                    dma_queue_प्रक्षिप(weight_dma, dma_make_दत्त(vtcm_raw_bufs[(i + 1) % 2], weight + nc_अग्र * weight_stride), dma_dst_stride, dma_src_stride, dma_विस्तारः_अष्टकानि, height_अग्र);
                }

                // C: HMX Compute (Queue-based)
                hmx_matmul_job_init(&job, vtcm_output, vtcm_f16_act, (__fp16 *) vtcm_weight_bufs[i % 2], vtcm_scales, n_row_tiles, n_col_tiles, k / HTP_MM_HMX_TILE_N_पङ्क्तिकाः);
                hmx_queue_प्रक्षिप(ctx->hmx_queue, hmx_queue_make_desc(hmx_matmul_कार्यकर्ता_fn, &job));
                hmx_queue_निष्कासय(ctx->hmx_queue);

                // D: Output Store
                भिन्न *output_chunk = dst + (mr * dst_stride + nc);
                ध्रुव भिन्न *bias_chunk = has_bias ? (vtcm_bias + mr * src2_stride + nc) : रिक्त;
                पूर्णाङ्क chunk_dst_स्तम्भसङ्ख्या = dst_स्तम्भसङ्ख्या - (पूर्णाङ्क)nc;
                यदि (chunk_dst_स्तम्भसङ्ख्या > 0) {
                    transfer_output_chunk_threaded(ctx, output_chunk, bias_chunk, vtcm_output, n_पङ्क्तिकाः, n_स्तम्भसङ्ख्या, dst_stride, src2_stride, chunk_dst_स्तम्भसङ्ख्या, n_तन्तवः);
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
        प्रत्यागम HTP_स्थितिः_INVAL_PARAMS;
    }

    ध्रुव संरचना htp_tensor * सीमित src0 = octx->src[0];
    ध्रुव संरचना htp_tensor * सीमित act  = octx->src[n_weights];

    ध्रुव पूर्णाङ्क weight_type = (पूर्णाङ्क) src0->type;
    ध्रुव पूर्णाङ्क k           = (पूर्णाङ्क) act->ne[0];
    ध्रुव पूर्णाङ्क k_valid     = (पूर्णाङ्क) act->ne[0];
    ध्रुव पूर्णाङ्क m           = (पूर्णाङ्क) (act->ne[1] * act->ne[2] * act->ne[3]);
    ध्रुव uint32_t act_elem_आकार = (act->type == HTP_TYPE_F16) ? परिमाण(__fp16) : परिमाण(भिन्न);
    ध्रुव पूर्णाङ्क act_stride  = (पूर्णाङ्क) (act->nb[1] / act_elem_आकार);
    ध्रुव dma_addr_t act_dma_addr = act->दत्त;

    यदि (k % 32 != 0) { प्रत्यागम HTP_स्थितिः_NO_SUPPORT; }

    size_t row_stride = htp_tensor_get_row_आकार(weight_type, k);
    यदि (row_stride == 0) {
        प्रत्यागम HTP_स्थितिः_NO_SUPPORT;
    }

    ध्रुव पूर्णाङ्क n_k_tiles = k / HTP_MM_HMX_TILE_N_स्तम्भसङ्ख्या;
    ध्रुव पूर्णाङ्क n_weight_tiles = k / htp_tensor_get_tile_k(weight_type);
    ध्रुव संरचना fastdiv_मानानि n_k_tiles_div = init_fastdiv_मानानि(n_k_tiles);

    ध्रुव पूर्णाङ्क tile_आकार       = kparams->tile_आकार;
    ध्रुव पूर्णाङ्क aligned_tile_आकार = kparams->aligned_tile_आकार;

    ध्रुव bool is_tiled       = (tile_आकार > 0);
    tiled_dequant_task_fn_t dequant_कार्यकर्ता_fn = hmx_get_dequant_कार्यकर्ता_fn(weight_type, is_tiled);
    ध्रुव size_t vtcm_budget  = ctx->vtcm_आकार;

    ध्रुव पूर्णाङ्क m_chunk_n_पङ्क्तिकाः  = kparams->m_chunk;
    ध्रुव पूर्णाङ्क n_chunk_n_स्तम्भसङ्ख्या  = kparams->n_chunk;
    ध्रुव पूर्णाङ्क pipeline        = kparams->pipeline;
    ध्रुव पूर्णाङ्क n_तन्तवः       = octx->n_तन्तवः;
    ध्रुव पूर्णाङ्क act_तन्तवः     = kparams->n_act_तन्तवः;
    ध्रुव संरचना fastdiv_मानानि * act_तन्तवः_div = &kparams->div_n_act_तन्तवः;
    ध्रुव संरचना fastdiv_मानानि * k_div           = &kparams->div_ne00_padded;

    ध्रुव uint32_t dma_dst_stride  = is_tiled ? aligned_tile_आकार : row_stride;
    ध्रुव uint32_t dma_विस्तारः_अष्टकानि = is_tiled ? tile_आकार : row_stride;

    संरचना htp_mm_hmx_vtcm_layout L;
    htp_mm_hmx_vtcm_layout_build(&L, HTP_MM_KERNEL_HMX_2D, weight_type, k, m_chunk_n_पङ्क्तिकाः, n_chunk_n_स्तम्भसङ्ख्या, 1, pipeline, act_तन्तवः, aligned_tile_आकार, 0);

    यदि (L.योग_अष्टकानि > vtcm_budget) {
        FARF(ERROR, "hmx-mm-nx-2d: VTCM overflow: used %zu budget %zu, m %d k %d mc %d nc %d",
             L.योग_अष्टकानि, vtcm_budget, m, k, m_chunk_n_पङ्क्तिकाः, n_chunk_n_स्तम्भसङ्ख्या);
        प्रत्यागम HTP_स्थितिः_VTCM_TOO_SMALL;
    }

    uint8_t * ध्रुव आधारः = (uint8_t *) ctx->vtcm_आधारः;
    __fp16  *vtcm_weight_raw[2] = {
        VTCM_LAYOUT_PTR(__fp16, आधारः, L.off_weight[0]),
        VTCM_LAYOUT_PTR_OPTIONAL(__fp16, आधारः, L.off_weight[1], pipeline)
    };
    __fp16  *vtcm_f16_act    = VTCM_LAYOUT_PTR(__fp16, आधारः, L.off_act);
    भिन्न   *vtcm_f32_act    = VTCM_LAYOUT_PTR(भिन्न, आधारः, L.off_act_f32);
    __fp16  *vtcm_output     = VTCM_LAYOUT_PTR(__fp16, आधारः, L.off_dst[0]);
    शून्य    *vtcm_scratch0   = VTCM_LAYOUT_PTR(शून्य, आधारः, L.off_scratch[0]);
    शून्य    *vtcm_scratch1   = VTCM_LAYOUT_PTR_OPTIONAL(शून्य, आधारः, L.off_scratch[1], pipeline);
    शून्य    *vtcm_scratch2   = VTCM_LAYOUT_PTR_OPTIONAL(शून्य, आधारः, L.off_dst[1], pipeline);
    __fp16  *vtcm_scales     = VTCM_LAYOUT_PTR(__fp16, आधारः, L.off_scales);

    hmx_init_column_scales(vtcm_scales, Q6_V_vsplat_R(0x3c00));  // scale: 1.0, bias: 0.0 in FP16

    पूर्णाङ्क m_start = 0;
    पूर्णाङ्क m_पङ्क्तिकाः  = m;
    यदि (octx->ctx->mdev.गणनम् > 1) {
        bool can_विभज = htp_tensor_can_row_partition(act, act_elem_आकार);
        यदि (can_विभज) {
            प्रति (uint32_t p = 0; p < n_weights; ++p) {
                ध्रुव संरचना htp_tensor * सीमित dst = octx->dsts[p];
                यदि (!htp_tensor_can_row_partition(dst, परिमाण(भिन्न))) {
                    can_विभज = false;
                    भङ्ग;
                }
            }
        }
        ध्रुव संरचना htp_tensor_mdev_range range = htp_tensor_mdev_partition((uint32_t) m, can_विभज ? 1 : 0, octx->ctx->mdev.idx, octx->ctx->mdev.गणनम्, &octx->ctx->mdev.गणनम्_div);
        m_start = (पूर्णाङ्क) range.start;
        m_पङ्क्तिकाः  = (पूर्णाङ्क) range.गणनम्;
    }

    यदि (m_पङ्क्तिकाः == 0) {
        प्रत्यागम HTP_स्थितिः_OK;
    }

    FARF(उच्चः_2, "hmx-mm-nx-2d: n_weights %u m %d (%d..%d) k %d wtype %d mc %d nc %d vtcm %zu/%zu",
         n_weights, m, m_start, m_start + m_पङ्क्तिकाः, k, weight_type, m_chunk_n_पङ्क्तिकाः, n_chunk_n_स्तम्भसङ्ख्या, L.योग_अष्टकानि, vtcm_budget);

    htp_trace_event_stop(tr, HTP_TRACE_EVT_INIT, 0);

    ध्रुव size_t mr_अन्तः = (size_t)(m_start + m_पङ्क्तिकाः);

    यदि (pipeline) {
        hmx_matmul_job_t job_slots[2];
        शून्य *vtcm_raw_bufs[4] = {
            vtcm_weight_raw[0],
            vtcm_weight_raw[1],
            dequant_कार्यकर्ता_fn ? vtcm_weight_raw[0] : vtcm_scratch0,
            dequant_कार्यकर्ता_fn ? vtcm_weight_raw[1] : vtcm_scratch1,
        };
        शून्य *vtcm_weight_bufs[4] = {
            dequant_कार्यकर्ता_fn ? vtcm_scratch0 : vtcm_weight_raw[0],
            dequant_कार्यकर्ता_fn ? vtcm_scratch1 : vtcm_weight_raw[1],
            vtcm_scratch0,
            vtcm_scratch1,
        };
        शून्य *vtcm_output_bufs[2] = { vtcm_output, vtcm_scratch2 };

        प्रति (size_t mr = (size_t) m_start; mr < mr_अन्तः; mr += m_chunk_n_पङ्क्तिकाः) {
            ध्रुव size_t n_पङ्क्तिकाः = hex_smin(mr_अन्तः - mr, m_chunk_n_पङ्क्तिकाः);

            संरचना activation_transfer_params act_params = {
                .ctx = ctx,
                .dst = vtcm_f16_act,
                .act_dma_addr = act_dma_addr + (size_t) mr * act_stride * act_elem_आकार,
                .n_पङ्क्तिकाः = (पूर्णाङ्क) n_पङ्क्तिकाः,
                .k_खण्डम् = k,
                .k_stride = act_stride,
                .n_तन्तवः = act_तन्तवः,
                .act_तन्तवः_div = act_तन्तवः_div,
                .k_div = k_div,
                .k_valid = k_valid,
                .vtcm_f32_act = vtcm_f32_act,
                .vtcm_f32_act_अष्टकानि = L.act_f32_अष्टकानि,
                .act_elem_आकार = act_elem_आकार,
            };
            transfer_activation_chunk_threaded(&act_params);

            प्रति (uint32_t p = 0; p < n_weights; p++) {
                ध्रुव संरचना htp_tensor * सीमित src_w = octx->src[p];
                ध्रुव संरचना htp_tensor * सीमित dst   = octx->dsts[p];
                यदि (!src_w || !dst) अनुवर्त;

                ध्रुव dma_addr_t weight       = src_w->दत्त;
                dma_queue * weight_dma      = octx->ctx->dma[0];
                भिन्न * dst_ptr            = (भिन्न *) dst->दत्त;
                ध्रुव size_t n             = src_w->ne[1];
                यदि (n == 0) अनुवर्त;
                ध्रुव size_t weight_stride = src_w->nb[1];
                ध्रुव size_t dst_stride    = dst->nb[1] / परिमाण(भिन्न);
                ध्रुव पूर्णाङ्क dst_स्तम्भसङ्ख्या         = (पूर्णाङ्क) dst->ne[0];
                ध्रुव पूर्णाङ्क n_chunk_cnt      = hmx_ceil_div(n, n_chunk_n_स्तम्भसङ्ख्या);

                ध्रुव uint32_t dma_src_stride = is_tiled ? tile_आकार : weight_stride;

                ध्रुव size_t   n_स्तम्भसङ्ख्या_A0 = hex_smin(n - 0 * n_chunk_n_स्तम्भसङ्ख्या, n_chunk_n_स्तम्भसङ्ख्या);
                ध्रुव uint32_t height_A0 = is_tiled ? hmx_ceil_div(n_स्तम्भसङ्ख्या_A0, 32) * n_weight_tiles : n_स्तम्भसङ्ख्या_A0;
                dma_queue_प्रक्षिप(weight_dma, dma_make_दत्त(vtcm_raw_bufs[0], weight),
                               dma_dst_stride, dma_src_stride, dma_विस्तारः_अष्टकानि, height_A0);

                यदि (1 < n_chunk_cnt) {
                    ध्रुव size_t   n_स्तम्भसङ्ख्या_A1 = hex_smin(n - 1 * n_chunk_n_स्तम्भसङ्ख्या, n_chunk_n_स्तम्भसङ्ख्या);
                    ध्रुव uint32_t height_A1 = is_tiled ? hmx_ceil_div(n_स्तम्भसङ्ख्या_A1, 32) * n_weight_tiles : n_स्तम्भसङ्ख्या_A1;
                    dma_queue_प्रक्षिप(weight_dma, dma_make_दत्त(vtcm_raw_bufs[1], weight + n_chunk_n_स्तम्भसङ्ख्या * weight_stride),
                                   dma_dst_stride, dma_src_stride, dma_विस्तारः_अष्टकानि, height_A1);
                }

                प्रति (पूर्णाङ्क i = 0; i < n_chunk_cnt; ++i) {
                    ध्रुव size_t nc    = i * n_chunk_n_स्तम्भसङ्ख्या;
                    ध्रुव size_t nc_p2 = nc + 2 * n_chunk_n_स्तम्भसङ्ख्या;

                    ध्रुव size_t n_स्तम्भसङ्ख्या    = hex_smin(n - nc, n_chunk_n_स्तम्भसङ्ख्या);
                    ध्रुव size_t n_स्तम्भसङ्ख्या_p2 = hex_smin(n - nc_p2, n_chunk_n_स्तम्भसङ्ख्या);

                    शून्य * curr_raw = (शून्य *) dma_queue_निष्कासय(weight_dma).dst;

                    dequantize_tiled_weight_chunk_गन्तव्यम्_fp16_tiles(
                        ctx, vtcm_weight_bufs[i % 4], curr_raw,
                        n_स्तम्भसङ्ख्या, k, row_stride, weight_type,
                        n_k_tiles, n_k_tiles_div, dequant_कार्यकर्ता_fn, n_तन्तवः);

                    यदि (i + 2 < n_chunk_cnt) {
                        ध्रुव uint32_t height_p2 = is_tiled ? hmx_ceil_div(n_स्तम्भसङ्ख्या_p2, 32) * n_weight_tiles : n_स्तम्भसङ्ख्या_p2;
                        dma_queue_प्रक्षिप(weight_dma, dma_make_दत्त(vtcm_raw_bufs[(i + 2) % 4], weight + nc_p2 * weight_stride),
                                       dma_dst_stride, dma_src_stride, dma_विस्तारः_अष्टकानि, height_p2);
                    }

                    hmx_matmul_job_init(&job_slots[i % 2], (__fp16 *) vtcm_output_bufs[i % 2],
                                        (__fp16 *) vtcm_f16_act, (__fp16 *) vtcm_weight_bufs[i % 4],
                                        vtcm_scales, hmx_ceil_div(n_पङ्क्तिकाः, HTP_MM_HMX_TILE_N_पङ्क्तिकाः),
                                        hmx_ceil_div(n_स्तम्भसङ्ख्या, HTP_MM_HMX_TILE_N_स्तम्भसङ्ख्या), k / HTP_MM_HMX_TILE_N_पङ्क्तिकाः);
                    hmx_queue_प्रक्षिप(ctx->hmx_queue, hmx_queue_make_desc(hmx_matmul_कार्यकर्ता_fn, &job_slots[i % 2]));

                    यदि (i > 0) {
                        hmx_queue_निष्कासय(ctx->hmx_queue);
                        ध्रुव size_t nc_पूर्व = (i - 1) * n_chunk_n_स्तम्भसङ्ख्या;
                        ध्रुव size_t n_स्तम्भसङ्ख्या_पूर्व = hex_smin(n - nc_पूर्व, n_chunk_n_स्तम्भसङ्ख्या);
                        भिन्न *output_chunk = dst_ptr + (mr * dst_stride + nc_पूर्व);
                        पूर्णाङ्क chunk_dst_स्तम्भसङ्ख्या = dst_स्तम्भसङ्ख्या - (पूर्णाङ्क)nc_पूर्व;
                        यदि (chunk_dst_स्तम्भसङ्ख्या > 0) {
                            transfer_output_chunk_threaded(ctx, output_chunk, रिक्त, vtcm_output_bufs[(i - 1) % 2], n_पङ्क्तिकाः, n_स्तम्भसङ्ख्या_पूर्व, dst_stride, 0, chunk_dst_स्तम्भसङ्ख्या, n_तन्तवः);
                        }
                    }
                }

                hmx_queue_निष्कासय(ctx->hmx_queue);
                ध्रुव size_t nc_चरमः = (n_chunk_cnt - 1) * n_chunk_n_स्तम्भसङ्ख्या;
                ध्रुव size_t n_स्तम्भसङ्ख्या_चरमः = hex_smin(n - nc_चरमः, n_chunk_n_स्तम्भसङ्ख्या);
                भिन्न *output_chunk = dst_ptr + (mr * dst_stride + nc_चरमः);
                पूर्णाङ्क chunk_dst_स्तम्भसङ्ख्या = dst_स्तम्भसङ्ख्या - (पूर्णाङ्क)nc_चरमः;
                यदि (chunk_dst_स्तम्भसङ्ख्या > 0) {
                    transfer_output_chunk_threaded(ctx, output_chunk, रिक्त, vtcm_output_bufs[(n_chunk_cnt - 1) % 2], n_पङ्क्तिकाः, n_स्तम्भसङ्ख्या_चरमः, dst_stride, 0, chunk_dst_स्तम्भसङ्ख्या, n_तन्तवः);
                }
            }
        }
    } अन्यथा {
        hmx_matmul_job_t job;
        शून्य *vtcm_raw_bufs[2] = { vtcm_weight_raw[0], dequant_कार्यकर्ता_fn ? vtcm_weight_raw[0] : vtcm_scratch0 };
        शून्य *vtcm_weight_bufs[2] = { dequant_कार्यकर्ता_fn ? vtcm_scratch0 : vtcm_weight_raw[0], vtcm_scratch0 };

        प्रति (size_t mr = (size_t) m_start; mr < mr_अन्तः; mr += m_chunk_n_पङ्क्तिकाः) {
            ध्रुव size_t n_पङ्क्तिकाः = hex_smin(mr_अन्तः - mr, m_chunk_n_पङ्क्तिकाः);

            संरचना activation_transfer_params act_params = {
                .ctx = ctx,
                .dst = vtcm_f16_act,
                .act_dma_addr = act_dma_addr + (size_t) mr * act_stride * act_elem_आकार,
                .n_पङ्क्तिकाः = (पूर्णाङ्क) n_पङ्क्तिकाः,
                .k_खण्डम् = k,
                .k_stride = act_stride,
                .n_तन्तवः = act_तन्तवः,
                .act_तन्तवः_div = act_तन्तवः_div,
                .k_div = k_div,
                .k_valid = k_valid,
                .vtcm_f32_act = vtcm_f32_act,
                .vtcm_f32_act_अष्टकानि = L.act_f32_अष्टकानि,
                .act_elem_आकार = act_elem_आकार,
            };
            transfer_activation_chunk_threaded(&act_params);

            प्रति (uint32_t p = 0; p < n_weights; p++) {
                ध्रुव संरचना htp_tensor * सीमित src_w = octx->src[p];
                ध्रुव संरचना htp_tensor * सीमित dst   = octx->dsts[p];
                यदि (!src_w || !dst) अनुवर्त;

                ध्रुव dma_addr_t weight       = src_w->दत्त;
                dma_queue * weight_dma      = octx->ctx->dma[0];
                भिन्न * dst_ptr            = (भिन्न *) dst->दत्त;
                ध्रुव size_t n             = src_w->ne[1];
                यदि (n == 0) अनुवर्त;
                ध्रुव size_t weight_stride = src_w->nb[1];
                ध्रुव size_t dst_stride    = dst->nb[1] / परिमाण(भिन्न);
                ध्रुव पूर्णाङ्क dst_स्तम्भसङ्ख्या         = (पूर्णाङ्क) dst->ne[0];
                ध्रुव पूर्णाङ्क n_chunk_cnt      = hmx_ceil_div(n, n_chunk_n_स्तम्भसङ्ख्या);

                ध्रुव uint32_t dma_src_stride = is_tiled ? tile_आकार : weight_stride;

                यदि (n > 0) {
                    ध्रुव size_t n_स्तम्भसङ्ख्या = hex_smin(n, n_chunk_n_स्तम्भसङ्ख्या);
                    ध्रुव uint32_t height = is_tiled ? hmx_ceil_div(n_स्तम्भसङ्ख्या, 32) * n_weight_tiles : n_स्तम्भसङ्ख्या;
                    dma_queue_प्रक्षिप(weight_dma, dma_make_दत्त(vtcm_raw_bufs[0], weight), dma_dst_stride, dma_src_stride, dma_विस्तारः_अष्टकानि, height);
                }

                प्रति (पूर्णाङ्क i = 0; i < n_chunk_cnt; ++i) {
                    ध्रुव size_t nc = (size_t) i * n_chunk_n_स्तम्भसङ्ख्या;
                    ध्रुव size_t n_स्तम्भसङ्ख्या = hex_smin(n - nc, n_chunk_n_स्तम्भसङ्ख्या);
                    ध्रुव size_t n_row_tiles = hmx_ceil_div(n_पङ्क्तिकाः, HTP_MM_HMX_TILE_N_पङ्क्तिकाः);
                    ध्रुव size_t n_col_tiles = hmx_ceil_div(n_स्तम्भसङ्ख्या, HTP_MM_HMX_TILE_N_स्तम्भसङ्ख्या);

                    शून्य * curr_raw = (शून्य *) dma_queue_निष्कासय(weight_dma).dst;

                    dequantize_tiled_weight_chunk_गन्तव्यम्_fp16_tiles(
                        ctx, vtcm_weight_bufs[i % 2], curr_raw,
                        n_स्तम्भसङ्ख्या, k, row_stride, weight_type,
                        n_k_tiles, n_k_tiles_div, dequant_कार्यकर्ता_fn, n_तन्तवः);

                    ध्रुव size_t nc_अग्र = nc + n_chunk_n_स्तम्भसङ्ख्या;
                    यदि (nc_अग्र < n) {
                        ध्रुव size_t n_स्तम्भसङ्ख्या_अग्र = hex_smin(n - nc_अग्र, n_chunk_n_स्तम्भसङ्ख्या);
                        ध्रुव uint32_t height_अग्र = is_tiled ? hmx_ceil_div(n_स्तम्भसङ्ख्या_अग्र, 32) * n_weight_tiles : n_स्तम्भसङ्ख्या_अग्र;
                        dma_queue_प्रक्षिप(weight_dma, dma_make_दत्त(vtcm_raw_bufs[(i + 1) % 2], weight + nc_अग्र * weight_stride), dma_dst_stride, dma_src_stride, dma_विस्तारः_अष्टकानि, height_अग्र);
                    }

                    hmx_matmul_job_init(&job, vtcm_output, vtcm_f16_act, (__fp16 *) vtcm_weight_bufs[i % 2], vtcm_scales, n_row_tiles, n_col_tiles, k / HTP_MM_HMX_TILE_N_पङ्क्तिकाः);
                    hmx_queue_प्रक्षिप(ctx->hmx_queue, hmx_queue_make_desc(hmx_matmul_कार्यकर्ता_fn, &job));
                    hmx_queue_निष्कासय(ctx->hmx_queue);

                    भिन्न *output_chunk = dst_ptr + (mr * dst_stride + nc);
                    पूर्णाङ्क chunk_dst_स्तम्भसङ्ख्या = dst_स्तम्भसङ्ख्या - (पूर्णाङ्क)nc;
                    यदि (chunk_dst_स्तम्भसङ्ख्या > 0) {
                        transfer_output_chunk_threaded(ctx, output_chunk, रिक्त, vtcm_output, n_पङ्क्तिकाः, n_स्तम्भसङ्ख्या, dst_stride, 0, chunk_dst_स्तम्भसङ्ख्या, n_तन्तवः);
                    }
                }
            }
        }
    }

    प्रत्यागम HTP_स्थितिः_OK;
}

स्थिर अन्तःस्थ dma_addr_t hmx_mm_weight_batch_दत्त(ध्रुव hmx_mm_batched_params_t *params,
                                                         पूर्णाङ्क dst_b2, पूर्णाङ्क dst_b3) {
    ध्रुव size_t b2_idx = (params->r2 <= 1) ? (size_t) dst_b2 : (size_t) fastdiv((uint32_t) dst_b2, &params->div_r2);
    ध्रुव size_t b3_idx = (params->r3 <= 1) ? (size_t) dst_b3 : (size_t) fastdiv((uint32_t) dst_b3, &params->div_r3);
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
                      (size_t) dst_b2 * params->dst_nb2 +
                      (size_t) dst_b3 * params->dst_nb3);
}

स्थिर पूर्णाङ्क hmx_mm_batched_simple(संरचना htp_context *ctx,
                                 ध्रुव hmx_mm_batched_params_t *params,
                                 पूर्णाङ्क m_chunk, पूर्णाङ्क n_chunk, पूर्णाङ्क pipeline, पूर्णाङ्क n_तन्तवः, पूर्णाङ्क act_तन्तवः, पूर्णाङ्क vtcm_आकार,
                                 ध्रुव संरचना fastdiv_मानानि * act_तन्तवः_div, ध्रुव संरचना fastdiv_मानानि * k_div) {
    पूर्णाङ्क ret = 0;
    प्रति (पूर्णाङ्क b3 = 0; b3 < params->ne13 && ret == 0; ++b3) {
        प्रति (पूर्णाङ्क b2 = 0; b2 < params->ne12 && ret == 0; ++b2) {
            dma_addr_t cur_bias_addr = params->bias_addr ? (params->bias_addr +
                                       b2 * params->bias_nb2 +
                                       b3 * params->bias_nb3) : 0;
            ret = hmx_mm_2d_f32(ctx, params->weight_dma, hmx_mm_dst_batch_ptr(params, b2, b3),
                                cur_bias_addr, params->bias_अष्टकानि,
                                hmx_mm_act_batch_addr(params, b2, b3),
                                hmx_mm_weight_batch_दत्त(params, b2, b3),
                                params->m, params->k, params->n,
                                params->act_stride, params->act_elem_आकार, params->weight_stride,
                                params->weight_type, params->k, params->dst_stride, params->bias_stride, params->dst_स्तम्भसङ्ख्या,
                                m_chunk, n_chunk, pipeline, n_तन्तवः, act_तन्तवः,
                                act_तन्तवः_div, k_div, params->tile_आकार, params->aligned_tile_आकार, vtcm_आकार);
        }
    }
    प्रत्यागम ret;
}

स्थिर पूर्णाङ्क hmx_mm_batched(संरचना htp_context *ctx, ध्रुव hmx_mm_batched_params_t *params,
                          पूर्णाङ्क m_chunk, पूर्णाङ्क n_chunk, पूर्णाङ्क pipeline, पूर्णाङ्क n_तन्तवः, पूर्णाङ्क act_तन्तवः,
                          ध्रुव संरचना fastdiv_मानानि * act_तन्तवः_div,
                          ध्रुव संरचना fastdiv_मानानि * k_div,
                          पूर्णाङ्क vtcm_आकार) {
    ध्रुव size_t row_stride = htp_tensor_get_row_आकार(params->weight_type, params->k);
    यदि (row_stride == 0) { प्रत्यागम -1; }
    यदि (params->act_stride < params->k || (params->tile_आकार == 0 && (size_t) params->weight_stride < row_stride) || params->dst_stride < params->dst_स्तम्भसङ्ख्या) { प्रत्यागम -1; }
    यदि (params->ne02 <= 0 || params->ne03 <= 0 || params->ne12 <= 0 || params->ne13 <= 0) { प्रत्यागम -1; }
    यदि (params->ne12 % params->ne02 != 0 || params->ne13 % params->ne03 != 0) { प्रत्यागम -1; }
    // N (the weight row count) does not have to be 32-aligned:
    यदि (params->k % 32 != 0) { प्रत्यागम -1; }
    यदि (!hex_is_aligned(params->dst, परिमाण(भिन्न))) { प्रत्यागम -1; }

    ध्रुव पूर्णाङ्क गणः_आकार = params->r2;
    ध्रुव size_t vtcm_budget  = ctx->vtcm_आकार;

    // Check if the precomputed parameters are grouped or simple.
    // If simple, or if group_size <= 1, we use simple fallback loop.
    // Grouped path is only valid for flat F16 if group_size > 1 and it fits within VTCM budget.
    bool सततम्_grouped = (params->weight_type == HTP_TYPE_F16 && params->tile_आकार == 0 && गणः_आकार > 1 && (size_t)vtcm_आकार <= vtcm_budget);
    यदि (!सततम्_grouped) {
        प्रत्यागम hmx_mm_batched_simple(ctx, params, m_chunk, n_chunk, pipeline, n_तन्तवः, act_तन्तवः, vtcm_आकार, act_तन्तवः_div, k_div);
    }

    संरचना htp_thread_trace * tr = &ctx->trace[0];
    htp_trace_event_start(tr, HTP_TRACE_EVT_INIT, 0);

    size_t m_chunk_n_पङ्क्तिकाः = m_chunk;
    size_t n_chunk_n_स्तम्भसङ्ख्या = n_chunk;

    संरचना htp_mm_hmx_vtcm_layout L;
    htp_mm_hmx_vtcm_layout_build(&L, HTP_MM_KERNEL_HMX_BATCHED, HTP_TYPE_F16, params->k, m_chunk_n_पङ्क्तिकाः, n_chunk_n_स्तम्भसङ्ख्या, गणः_आकार, false, act_तन्तवः, 0, params->bias_अष्टकानि);

    यदि (L.योग_अष्टकानि > vtcm_budget) {
        FARF(उच्चः_2, "%s: grouped layout overflowed VTCM, falling back to simple batched loop", __func__);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_INIT, 0);
        प्रत्यागम hmx_mm_batched_simple(ctx, params, m_chunk, n_chunk, pipeline, n_तन्तवः, act_तन्तवः, vtcm_आकार, act_तन्तवः_div, k_div);
    }

    uint8_t * ध्रुव आधारः = (uint8_t *) ctx->vtcm_आधारः;
    __fp16  *vtcm_weight     = VTCM_LAYOUT_PTR(__fp16, आधारः, L.off_weight[0]);
    __fp16  *vtcm_f16_act    = VTCM_LAYOUT_PTR(__fp16, आधारः, L.off_act);
    __fp16  *vtcm_output     = VTCM_LAYOUT_PTR(__fp16, आधारः, L.off_dst[0]);
    शून्य    *vtcm_scratch0   = VTCM_LAYOUT_PTR(शून्य, आधारः, L.off_scratch[0]);
    शून्य    *vtcm_scratch1   = VTCM_LAYOUT_PTR(शून्य, आधारः, L.off_scratch[1]);
    __fp16  *vtcm_scales     = VTCM_LAYOUT_PTR(__fp16, आधारः, L.off_scales);
    भिन्न   *vtcm_f32_act    = VTCM_LAYOUT_PTR(भिन्न, आधारः, L.off_act_f32);

    ध्रुव bool has_bias      = (params->bias_अष्टकानि > 0 && params->bias_addr != 0);
    भिन्न   *vtcm_bias       = VTCM_LAYOUT_PTR_OPTIONAL(भिन्न, आधारः, L.off_bias, has_bias);
    यदि (has_bias) {
        dma_queue_प्रक्षिप(params->weight_dma, dma_make_दत्त(vtcm_bias, params->bias_addr), hex_align_ऊर्ध्वम्(params->bias_अष्टकानि, 128), 0, params->bias_अष्टकानि, 1);
        dma_queue_निष्कासय(params->weight_dma);
    }

    hmx_init_column_scales(vtcm_scales, Q6_V_vsplat_R(0x3c00));  // scale: 1.0, bias: 0.0 in FP16

    FARF(उच्चः_2, "%s: grouped path m=%d k=%d n=%d group=%d streams=%d mc=%zu nc=%zu vtcm=%zu/%zu",
            __func__, params->m, params->k, params->n, गणः_आकार, params->ne13,
            m_chunk_n_पङ्क्तिकाः, n_chunk_n_स्तम्भसङ्ख्या,
            L.योग_अष्टकानि, vtcm_budget);

    ध्रुव size_t fp16_row_अष्टकानि   = (size_t) params->k * परिमाण(__fp16);
    ध्रुव size_t weight_row_अष्टकानि = (size_t) params->weight_stride;

    htp_trace_event_stop(tr, HTP_TRACE_EVT_INIT, 0);

    hmx_matmul_job_t job;

    प्रति (पूर्णाङ्क b3 = 0; b3 < params->ne13; ++b3) {
        प्रति (पूर्णाङ्क b2_आधारः = 0; b2_आधारः < params->ne12; b2_आधारः += गणः_आकार) {
            ध्रुव dma_addr_t weight_गणः = hmx_mm_weight_batch_दत्त(params, b2_आधारः, b3);
            dma_queue * weight_dma = params->weight_dma;

            प्रति (size_t mr = 0; mr < (size_t) params->m; mr += m_chunk_n_पङ्क्तिकाः) {
                ध्रुव size_t n_पङ्क्तिकाः = hex_smin((size_t) params->m - mr, m_chunk_n_पङ्क्तिकाः);
                ध्रुव size_t n_row_tiles = hmx_ceil_div((पूर्णाङ्क) n_पङ्क्तिकाः, HTP_MM_HMX_TILE_N_पङ्क्तिकाः);

                // Pre-load activations for all heads in the group (once per m_chunk).
                // When the source is strided (permuted Q), use 2D DMA to gather
                // contiguous rows into a VTCM scratch buffer first, then HVX
                // converts from the contiguous VTCM buffer.  This avoids L2 cache
                // thrashing from HVX loads at large strides.
                प्रति (पूर्णाङ्क g = 0; g < गणः_आकार; ++g) {
                    ध्रुव dma_addr_t act_dma_addr = hmx_mm_act_batch_addr(params, b2_आधारः + g, b3) +
                                                      (size_t) mr * params->act_stride * params->act_elem_आकार;
                    __fp16 *vtcm_act_g = vtcm_f16_act + (size_t) g * L.act_शीर्ष_stride;
                    संरचना activation_transfer_params act_params = {
                        .ctx = ctx,
                        .dst = vtcm_act_g,
                        .act_dma_addr = act_dma_addr,
                        .n_पङ्क्तिकाः = (पूर्णाङ्क) n_पङ्क्तिकाः,
                        .k_खण्डम् = params->k,
                        .k_stride = params->act_stride,
                        .n_तन्तवः = act_तन्तवः,
                        .act_तन्तवः_div = act_तन्तवः_div,
                        .k_div = k_div,
                        .k_valid = params->k,
                        .vtcm_f32_act = vtcm_f32_act,
                        .vtcm_f32_act_अष्टकानि = L.act_f32_अष्टकानि,
                        .act_elem_आकार = params->act_elem_आकार,
                    };
                    transfer_activation_chunk_threaded(&act_params);
                }

                // Prologue: Push A0 and A1 (if exists)
                {
                    ध्रुव size_t n_स्तम्भसङ्ख्या_first = hex_smin((size_t) params->n, n_chunk_n_स्तम्भसङ्ख्या);
                    dma_queue_प्रक्षिप(weight_dma, dma_make_दत्त(vtcm_scratch0, weight_गणः),
                                      fp16_row_अष्टकानि, weight_row_अष्टकानि, fp16_row_अष्टकानि, n_स्तम्भसङ्ख्या_first);
                }
                यदि (n_chunk_n_स्तम्भसङ्ख्या < (size_t) params->n) {
                    ध्रुव size_t n_स्तम्भसङ्ख्या_second = hex_smin((size_t) params->n - n_chunk_n_स्तम्भसङ्ख्या, n_chunk_n_स्तम्भसङ्ख्या);
                    ध्रुव dma_addr_t second_weight_chunk = weight_गणः + n_chunk_n_स्तम्भसङ्ख्या * weight_row_अष्टकानि;
                    dma_queue_प्रक्षिप(weight_dma, dma_make_दत्त(vtcm_scratch1, second_weight_chunk),
                                      fp16_row_अष्टकानि, weight_row_अष्टकानि, fp16_row_अष्टकानि, n_स्तम्भसङ्ख्या_second);
                }

                प्रति (size_t nc = 0; nc < (size_t) params->n; nc += n_chunk_n_स्तम्भसङ्ख्या) {
                    ध्रुव size_t n_स्तम्भसङ्ख्या      = hex_smin((size_t) params->n - nc, n_chunk_n_स्तम्भसङ्ख्या);
                    ध्रुव size_t n_col_tiles = hmx_ceil_div((पूर्णाङ्क) n_स्तम्भसङ्ख्या, HTP_MM_HMX_TILE_N_स्तम्भसङ्ख्या);

                    {
                        शून्य * curr_raw = (शून्य *) dma_queue_निष्कासय(weight_dma).dst;

                        ध्रुव size_t n_स्तम्भसङ्ख्या_tiled = hex_align_ऊर्ध्वम्(n_स्तम्भसङ्ख्या, HTP_MM_HMX_TILE_N_स्तम्भसङ्ख्या);

                        hmx_interleave_पङ्क्तिकाः_गन्तव्यम्_tiles(vtcm_weight, (ध्रुव __fp16 *) curr_raw, (uint32_t) n_स्तम्भसङ्ख्या, params->k, params->k, 0, (uint32_t) n_स्तम्भसङ्ख्या_tiled);

                        ध्रुव size_t nc_अग्र = nc + n_chunk_n_स्तम्भसङ्ख्या * 2;
                        यदि (nc_अग्र < (size_t) params->n) {
                            ध्रुव size_t n_स्तम्भसङ्ख्या_अग्र = hex_smin((size_t) params->n - nc_अग्र, n_chunk_n_स्तम्भसङ्ख्या);
                            ध्रुव dma_addr_t अग्र_weight_chunk = weight_गणः + nc_अग्र * weight_row_अष्टकानि;

                            dma_queue_प्रक्षिप(weight_dma, dma_make_दत्त(curr_raw, अग्र_weight_chunk),
                                              fp16_row_अष्टकानि, weight_row_अष्टकानि, fp16_row_अष्टकानि, n_स्तम्भसङ्ख्या_अग्र);
                        }
                    }

                    // Reuse the interleaved weight for every q_head in this GQA group
                    प्रति (पूर्णाङ्क g = 0; g < गणः_आकार; ++g) {
                        {
                            ध्रुव __fp16 * vtcm_act_g = vtcm_f16_act + (size_t) g * L.act_शीर्ष_stride;
                            hmx_matmul_job_init(&job, vtcm_output, vtcm_act_g, vtcm_weight, vtcm_scales, n_row_tiles, n_col_tiles, params->k / 32);
                            hmx_queue_प्रक्षिप(ctx->hmx_queue, hmx_queue_make_desc(hmx_matmul_कार्यकर्ता_fn, &job));
                            hmx_queue_निष्कासय(ctx->hmx_queue);
                        }

                        {
                            भिन्न *output = hmx_mm_dst_batch_ptr(params, b2_आधारः + g, b3) + mr * params->dst_stride + nc;
                            ध्रुव भिन्न *bias_chunk = has_bias ? (vtcm_bias + mr * params->bias_stride + nc) : रिक्त;
                            पूर्णाङ्क chunk_dst_स्तम्भसङ्ख्या = params->dst_स्तम्भसङ्ख्या - (पूर्णाङ्क)nc;
                            यदि (chunk_dst_स्तम्भसङ्ख्या > 0) {
                                transfer_output_chunk_threaded(ctx, output, bias_chunk, vtcm_output, (पूर्णाङ्क) n_पङ्क्तिकाः, (पूर्णाङ्क) n_स्तम्भसङ्ख्या,
                                                               params->dst_stride, params->bias_stride, chunk_dst_स्तम्भसङ्ख्या, n_तन्तवः);
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
            पूर्णाङ्क n_पङ्क्तिकाः,
            पूर्णाङ्क k_खण्डम्,
            ध्रुव संरचना mmid_row_mapping *matrix_पङ्क्तिकाः,
            पूर्णाङ्क cur_a,
            पूर्णाङ्क mapping_stride,
            पूर्णाङ्क ne11,
            size_t nb11,
            size_t nb12,
            पूर्णाङ्क cne1,
            पूर्णाङ्क n_तन्तवः,
            पूर्णाङ्क k_valid) {
    यदि (n_पङ्क्तिकाः <= 0) प्रत्यागम;
    पूर्णाङ्क chunks_प्रत्येकम्_thread = hmx_ceil_div(n_पङ्क्तिकाः, n_तन्तवः);
    chunks_प्रत्येकम्_thread = hex_align_ऊर्ध्वम्(chunks_प्रत्येकम्_thread, 2);

    पूर्णाङ्क actual_तन्तवः = hmx_ceil_div(n_पङ्क्तिकाः, chunks_प्रत्येकम्_thread);

    activation_transfer_gathered_task_state_t state = {
        .dst               = dst,
        .src               = src,
        .n_tasks           = actual_तन्तवः,
        .n_tot_chunks      = n_पङ्क्तिकाः,
        .n_chunks_प्रत्येकम्_task = chunks_प्रत्येकम्_thread,
        .k_खण्डम्           = k_खण्डम्,
        .matrix_पङ्क्तिकाः       = matrix_पङ्क्तिकाः,
        .cur_a             = cur_a,
        .mapping_stride    = mapping_stride,
        .ne11              = ne11,
        .ne11_div          = ne11 > 1 ? init_fastdiv_मानानि(ne11) : (संरचना fastdiv_मानानि){0, 0},
        .nb11              = nb11,
        .nb12              = nb12,
        .start_row         = start_row,
        .cne1              = cne1,
        .k_valid           = k_valid,
        .traces            = ctx->trace,
    };

    कार्यकर्ता_callback_t कार्यकर्ता_fn = ne11 == 1 ? transfer_activation_chunk_gathered_कार्यकर्ता_flat_fn :
                                              transfer_activation_chunk_gathered_कार्यकर्ता_fn;

    यदि (actual_तन्तवः <= 1) {
        कार्यकर्ता_fn(1, 0, &state);
    } अन्यथा {
        कार्यकर्ता_pool_सततम्_func(ctx->कार्यकर्ता_pool, कार्यकर्ता_fn, &state, actual_तन्तवः);
    }
}

स्थिर शून्य transfer_output_chunk_scattered_threaded(
            संरचना htp_context *ctx,
            भिन्न *dst,
            ध्रुव __fp16 *vtcm_src,
            पूर्णाङ्क start_row,
            पूर्णाङ्क n_पङ्क्तिकाः,
            पूर्णाङ्क n_स्तम्भसङ्ख्या,
            ध्रुव संरचना mmid_row_mapping *matrix_पङ्क्तिकाः,
            पूर्णाङ्क cur_a,
            पूर्णाङ्क mapping_stride,
            size_t dst_nb1,
            size_t dst_nb2,
            पूर्णाङ्क cne1,
            पूर्णाङ्क n_तन्तवः) {
    यदि (n_पङ्क्तिकाः <= 0) प्रत्यागम;
    पूर्णाङ्क chunks_प्रत्येकम्_thread = hmx_ceil_div(n_पङ्क्तिकाः, n_तन्तवः);
    chunks_प्रत्येकम्_thread = hex_align_ऊर्ध्वम्(chunks_प्रत्येकम्_thread, 2);

    पूर्णाङ्क actual_तन्तवः = hmx_ceil_div(n_पङ्क्तिकाः, chunks_प्रत्येकम्_thread);

    output_transfer_scattered_task_state_t state = {
        .vtcm_src          = vtcm_src,
        .dst               = dst,
        .n_tasks           = actual_तन्तवः,
        .n_tot_chunks      = n_पङ्क्तिकाः,
        .n_chunks_प्रत्येकम्_task = chunks_प्रत्येकम्_thread,
        .n_स्तम्भसङ्ख्या            = n_स्तम्भसङ्ख्या,
        .matrix_पङ्क्तिकाः       = matrix_पङ्क्तिकाः,
        .cur_a             = cur_a,
        .mapping_stride    = mapping_stride,
        .dst_nb1           = dst_nb1,
        .dst_nb2           = dst_nb2,
        .start_row         = start_row,
        .cne1              = cne1,
        .traces            = ctx->trace,
    };

    यदि (actual_तन्तवः <= 1) {
        transfer_output_chunk_scattered_कार्यकर्ता_fn(1, 0, &state);
    } अन्यथा {
        कार्यकर्ता_pool_सततम्_func(ctx->कार्यकर्ता_pool, transfer_output_chunk_scattered_कार्यकर्ता_fn, &state, actual_तन्तवः);
    }
}

स्थिर पूर्णाङ्क hmx_mm_id_2d_f32(संरचना htp_context *ctx,
                                         dma_queue *weight_dma,
                                         भिन्न *सीमित dst,
                                         ध्रुव भिन्न *activation,
                                         dma_addr_t weight,
                                         पूर्णाङ्क m, पूर्णाङ्क k, पूर्णाङ्क n,
                                         पूर्णाङ्क k_valid,
                                         पूर्णाङ्क ne11,
                                         size_t act_nb1, size_t act_nb2,
                                         size_t dst_nb1, size_t dst_nb2,
                                         पूर्णाङ्क weight_stride,
                                         पूर्णाङ्क weight_type,
                                         पूर्णाङ्क tile_आकार,
                                         पूर्णाङ्क aligned_tile_आकार,
                                         ध्रुव संरचना mmid_row_mapping *matrix_पङ्क्तिकाः,
                                         पूर्णाङ्क cur_a,
                                         पूर्णाङ्क mapping_stride,
                                         पूर्णाङ्क m_start,
                                         पूर्णाङ्क m_अन्तः,
                                         पूर्णाङ्क n_तन्तवः) {
    संरचना htp_thread_trace * tr = &ctx->trace[0];
    htp_trace_event_start(tr, HTP_TRACE_EVT_INIT, 0);

    ध्रुव पूर्णाङ्क cne1 = m;
    ध्रुव पूर्णाङ्क m_core = m_अन्तः - m_start;
    ध्रुव पूर्णाङ्क m_core_padded = hex_align_ऊर्ध्वम्(m_core > 0 ? m_core : 1, 32);

    यदि (k % 32 != 0 || n % 32 != 0) { प्रत्यागम -1; }
    यदि (!hex_is_aligned(dst, VLEN) || !hex_is_aligned(activation, VLEN)) { प्रत्यागम -1; }

    size_t row_stride = htp_tensor_get_row_आकार(weight_type, k);
    यदि (row_stride == 0) {
        प्रत्यागम -1;
    }

    ध्रुव पूर्णाङ्क n_k_tiles = k / HTP_MM_HMX_TILE_N_स्तम्भसङ्ख्या;
    ध्रुव uint32_t n_weight_tiles = k / htp_tensor_get_tile_k(weight_type);
    ध्रुव संरचना fastdiv_मानानि n_k_tiles_div = init_fastdiv_मानानि(n_k_tiles);

    ध्रुव size_t vec_dot_आकार = k * परिमाण(__fp16);
    ध्रुव size_t vtcm_budget  = ctx->vtcm_आकार;
    size_t vtcm_प्रयुक्तम् = 0;

    ध्रुव bool is_tiled = (tile_आकार > 0);
    tiled_dequant_task_fn_t dequant_कार्यकर्ता_fn = hmx_get_dequant_कार्यकर्ता_fn(weight_type, is_tiled);

    ध्रुव uint32_t dma_dst_stride  = is_tiled ? aligned_tile_आकार : row_stride;
    ध्रुव uint32_t dma_src_stride  = is_tiled ? tile_आकार : weight_stride;
    ध्रुव uint32_t dma_विस्तारः_अष्टकानि = is_tiled ? tile_आकार : row_stride;

    ध्रुव size_t qweight_row_stride = is_tiled ? (size_t)(n_weight_tiles * aligned_tile_आकार) / 32 : 0;
    ध्रुव size_t weight_row_stride = is_tiled ? qweight_row_stride : row_stride;

    size_t आकार_प्रत्येकम्_n = 0, आकार_प्रत्येकम्_m = 0, आकार_प्रत्येकम्_mn = 0;
    htp_mm_hmx_get_2d_chunk_costs(weight_type, k, /*pipeline=*/false, aligned_tile_आकार,
                                  &आकार_प्रत्येकम्_n, &आकार_प्रत्येकम्_m, &आकार_प्रत्येकम्_mn);

    ध्रुव size_t overhead = htp_mm_hmx_get_2d_overhead(/*pipeline=*/false, /*is_matmul_id=*/true);
    size_t m_chunk_n_पङ्क्तिकाः = 0, n_chunk_n_स्तम्भसङ्ख्या = 0;
    यदि (htp_mm_hmx_compute_chunks(vtcm_budget, overhead, आकार_प्रत्येकम्_n, आकार_प्रत्येकम्_m, आकार_प्रत्येकम्_mn,
                           m_core_padded, n,
                           /*m_block_cost=*/(size_t) n * HTP_MM_HMX_COST_W_DEQUANT,
                           /*n_block_cost=*/(size_t) m_core_padded * HTP_MM_HMX_COST_A_CONVERT, &m_chunk_n_पङ्क्तिकाः, &n_chunk_n_स्तम्भसङ्ख्या, &vtcm_प्रयुक्तम्)) {
        FARF(ERROR, "hmx-mm-id-2d: VTCM too small : m %d k %d n %d budget %zu", m_core_padded, k, n, vtcm_budget);
        प्रत्यागम -1;
    }

    ध्रुव size_t weight_area_आकार = hex_align_ऊर्ध्वम्(n_chunk_n_स्तम्भसङ्ख्या * weight_row_stride, HTP_MM_HMX_TILE_आकार);
    ध्रुव size_t act_area_आकार    = hex_align_ऊर्ध्वम्(m_chunk_n_पङ्क्तिकाः * vec_dot_आकार, HTP_MM_HMX_TILE_आकार);
    ध्रुव size_t output_area_आकार = hex_align_ऊर्ध्वम्(m_chunk_n_पङ्क्तिकाः * n_chunk_n_स्तम्भसङ्ख्या * परिमाण(__fp16), HTP_MM_HMX_TILE_आकार);

    size_t scratch0_आकार = hex_align_ऊर्ध्वम्(n_chunk_n_स्तम्भसङ्ख्या * vec_dot_आकार, HTP_MM_HMX_TILE_आकार);

    uint8_t *vtcm_ptr      = (uint8_t *) ctx->vtcm_आधारः;
    __fp16  *vtcm_weight   = weight_area_आकार ? (__fp16 *) vtcm_seq_alloc(&vtcm_ptr, weight_area_आकार) : रिक्त;
    __fp16  *vtcm_f16_act  = (__fp16 *) vtcm_seq_alloc(&vtcm_ptr, act_area_आकार);
    __fp16  *vtcm_output   = (__fp16 *) vtcm_seq_alloc(&vtcm_ptr, output_area_आकार);
    शून्य    *vtcm_scratch0 = vtcm_seq_alloc(&vtcm_ptr, scratch0_आकार);
    __fp16  *vtcm_scales   = (__fp16 *) vtcm_seq_alloc(&vtcm_ptr, 256);

    vtcm_प्रयुक्तम् = vtcm_ptr - (uint8_t *) ctx->vtcm_आधारः;
    यदि (vtcm_प्रयुक्तम् > vtcm_budget) {
        FARF(ERROR, "hmx-mm-id-2d: VTCM overflow: used %zu budget %zu", vtcm_प्रयुक्तम्, vtcm_budget);
        प्रत्यागम -1;
    }

    hmx_init_column_scales(vtcm_scales, Q6_V_vsplat_R(0x3c00));

    htp_trace_event_stop(tr, HTP_TRACE_EVT_INIT, 0);

    hmx_matmul_job_t job;
    शून्य *vtcm_raw_bufs[2] = { vtcm_weight, dequant_कार्यकर्ता_fn ? vtcm_weight : vtcm_scratch0 };
    शून्य *vtcm_weight_bufs[2] = { dequant_कार्यकर्ता_fn ? vtcm_scratch0 : vtcm_weight, vtcm_scratch0 };
    ध्रुव पूर्णाङ्क n_chunk_cnt = hmx_ceil_div(n, n_chunk_n_स्तम्भसङ्ख्या);

    प्रति (size_t mr = (size_t) m_start; mr < (size_t) m_अन्तः; mr += m_chunk_n_पङ्क्तिकाः) {
        ध्रुव size_t n_पङ्क्तिकाः = hex_smin((size_t) m_अन्तः - mr, m_chunk_n_पङ्क्तिकाः);
        ध्रुव size_t n_row_tiles = hmx_ceil_div(n_पङ्क्तिकाः, HTP_MM_HMX_TILE_N_पङ्क्तिकाः);

        transfer_activation_chunk_gathered_threaded(
            ctx, vtcm_f16_act, activation, (पूर्णाङ्क) mr, (पूर्णाङ्क) n_पङ्क्तिकाः, k,
            matrix_पङ्क्तिकाः, cur_a, mapping_stride, ne11, act_nb1, act_nb2, cne1, n_तन्तवः, k_valid);

        // A0: Pre-fetch the first weight chunk (nc = 0)
        यदि (n > 0) {
            ध्रुव size_t n_स्तम्भसङ्ख्या = hex_smin((size_t) n, n_chunk_n_स्तम्भसङ्ख्या);
            ध्रुव uint32_t height = is_tiled ? hmx_ceil_div(n_स्तम्भसङ्ख्या, 32) * n_weight_tiles : n_स्तम्भसङ्ख्या;
            dma_queue_प्रक्षिप(weight_dma, dma_make_दत्त(vtcm_raw_bufs[0], weight),
                           dma_dst_stride, dma_src_stride, dma_विस्तारः_अष्टकानि, height);
        }

        प्रति (पूर्णाङ्क i = 0; i < n_chunk_cnt; ++i) {
            ध्रुव size_t nc = (size_t) i * n_chunk_n_स्तम्भसङ्ख्या;
            ध्रुव size_t n_स्तम्भसङ्ख्या = hex_smin((size_t) n - nc, n_chunk_n_स्तम्भसङ्ख्या);
            ध्रुव size_t n_col_tiles = hmx_ceil_div(n_स्तम्भसङ्ख्या, HTP_MM_HMX_TILE_N_स्तम्भसङ्ख्या);

            // A: Wait for weight DMA
            शून्य * curr_raw = (शून्य *) dma_queue_निष्कासय(weight_dma).dst;

            // B: Weight Dequantize (Threaded)
            dequantize_tiled_weight_chunk_गन्तव्यम्_fp16_tiles(
                ctx, vtcm_weight_bufs[i % 2], curr_raw,
                n_स्तम्भसङ्ख्या, k, row_stride, weight_type,
                n_k_tiles, n_k_tiles_div, dequant_कार्यकर्ता_fn, n_तन्तवः
            );

            // Start weight DMA for the next chunk early
            ध्रुव size_t nc_अग्र = nc + n_chunk_n_स्तम्भसङ्ख्या;
            यदि (nc_अग्र < (size_t) n) {
                ध्रुव size_t n_स्तम्भसङ्ख्या_अग्र = hex_smin((size_t) n - nc_अग्र, n_chunk_n_स्तम्भसङ्ख्या);
                ध्रुव uint32_t height_अग्र = is_tiled ? hmx_ceil_div(n_स्तम्भसङ्ख्या_अग्र, 32) * n_weight_tiles : n_स्तम्भसङ्ख्या_अग्र;
                dma_queue_प्रक्षिप(weight_dma, dma_make_दत्त(vtcm_raw_bufs[(i + 1) % 2], weight + nc_अग्र * weight_stride),
                               dma_dst_stride, dma_src_stride, dma_विस्तारः_अष्टकानि, height_अग्र);
            }

            // C: HMX Compute (Queue-based)
            hmx_matmul_job_init(&job, vtcm_output, vtcm_f16_act, (__fp16 *) vtcm_weight_bufs[i % 2], vtcm_scales, n_row_tiles, n_col_tiles, k / HTP_MM_HMX_TILE_N_पङ्क्तिकाः);
            hmx_queue_प्रक्षिप(ctx->hmx_queue, hmx_queue_make_desc(hmx_matmul_कार्यकर्ता_fn, &job));
            hmx_queue_निष्कासय(ctx->hmx_queue);

            // D: Output Store
            transfer_output_chunk_scattered_threaded(
                ctx, dst + nc, vtcm_output, (पूर्णाङ्क) mr, (पूर्णाङ्क) n_पङ्क्तिकाः, (पूर्णाङ्क) n_स्तम्भसङ्ख्या,
                matrix_पङ्क्तिकाः, cur_a, mapping_stride, dst_nb1, dst_nb2, cne1, n_तन्तवः);
        }
    }

    प्रत्यागम 0;
}

// --- Dispatchers and Public Entry Points ---

स्थिर पूर्णाङ्क hmx_mm_op_matmul(संरचना htp_ops_context * octx, ध्रुव संरचना htp_mm_kernel_params * kparams) {
    htp_matmul_tensors_preamble;

    पूर्णाङ्क k = (पूर्णाङ्क) src0->ne[0];
    पूर्णाङ्क n = (पूर्णाङ्क) src0->ne[1];
    ध्रुव पूर्णाङ्क m_योग    = (पूर्णाङ्क) act->ne[1];
    ध्रुव uint32_t act_elem_आकार = (act->type == HTP_TYPE_F16) ? परिमाण(__fp16) : परिमाण(भिन्न);
    ध्रुव पूर्णाङ्क act_stride = (पूर्णाङ्क) (act->nb[1] / act_elem_आकार);

    पूर्णाङ्क m_start = 0;
    पूर्णाङ्क m_पङ्क्तिकाः  = m_योग;
    यदि (octx->ctx->mdev.गणनम् > 1) {
        bool can_विभज = htp_tensor_can_row_partition(dst, परिमाण(भिन्न)) &&
                         htp_tensor_can_row_partition(act, act_elem_आकार);
        यदि (src2 && src2->ne[1] > 1 && !htp_tensor_can_row_partition(src2, परिमाण(भिन्न))) {
            can_विभज = false;
        }
        ध्रुव संरचना htp_tensor_mdev_range range = htp_tensor_mdev_partition((uint32_t) m_योग, can_विभज ? 1 : 0, octx->ctx->mdev.idx, octx->ctx->mdev.गणनम्, &octx->ctx->mdev.गणनम्_div);
        m_start = (पूर्णाङ्क) range.start;
        m_पङ्क्तिकाः  = (पूर्णाङ्क) range.गणनम्;
    }

    यदि (m_पङ्क्तिकाः == 0) {
        प्रत्यागम HTP_स्थितिः_OK;
    }

    dma_addr_t src2_addr = 0;
    size_t src2_अष्टकानि = 0;
    uint32_t src2_stride = 0;
    size_t src2_nb2 = 0;
    size_t src2_nb3 = 0;
    यदि (src2) {
        src2_stride = (src2->ne[1] == 1) ? 0 : (uint32_t) (src2->nb[1] / परिमाण(भिन्न));
        src2_addr = src2->दत्त + m_start * src2_stride * परिमाण(भिन्न);
        src2_अष्टकानि = (size_t) kparams->vtcm_bias_आकार;
        src2_nb2 = (src2->ne[2] == 1) ? 0 : src2->nb[2];
        src2_nb3 = (src2->ne[3] == 1) ? 0 : src2->nb[3];
    }

    ध्रुव पूर्णाङ्क dst_stride = (पूर्णाङ्क)(dst->nb[1] / परिमाण(भिन्न));
    भिन्न       * dst_ptr = (भिन्न *) dst->दत्त + m_start * dst_stride;
    // byte offset, act_stride is in activation elements (F16 or F32)
    ध्रुव dma_addr_t act_addr = act->दत्त + (size_t) m_start * act_stride * act_elem_आकार;

    पूर्णाङ्क ret = -1;
    ध्रुव पूर्णाङ्क n_तन्तवः = kparams->n_तन्तवः;
    यदि (kparams->kernel_type == HTP_MM_KERNEL_HMX_BATCHED) {
        hmx_mm_batched_params_t batch_params = {
            .dst               = dst_ptr,
            .bias_addr         = src2_addr,
            .bias_अष्टकानि        = src2_अष्टकानि,
            .act_dma_addr      = act_addr,
            .weight            = src0->दत्त,
            .weight_dma        = octx->ctx->dma[0],
            .m                 = m_पङ्क्तिकाः,
            .k                 = k,
            .n                 = n,
            .act_stride        = act_stride,
            .act_elem_आकार     = act_elem_आकार,
            .weight_stride     = (पूर्णाङ्क) src0->nb[1],
            .weight_type       = (पूर्णाङ्क) src0->type,
            .dst_stride        = dst_stride,
            .dst_स्तम्भसङ्ख्या          = (पूर्णाङ्क) dst->ne[0],
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
                             kparams->pipeline, n_तन्तवः,
                             kparams->n_act_तन्तवः,
                             &kparams->div_n_act_तन्तवः,
                             &kparams->div_ne00_padded,
                             kparams->vtcm_आकार);
    } अन्यथा {
        ret = hmx_mm_2d_f32(
            octx->ctx, octx->ctx->dma[0], dst_ptr, src2_addr, src2_अष्टकानि,
            act_addr, src0->दत्त,
            m_पङ्क्तिकाः, k, n, act_stride, act_elem_आकार, (पूर्णाङ्क) src0->nb[1], (पूर्णाङ्क) src0->type, (पूर्णाङ्क) act->ne[0],
            dst_stride, src2_stride, (पूर्णाङ्क)dst->ne[0],
            kparams->m_chunk, kparams->n_chunk, kparams->pipeline, n_तन्तवः,
            kparams->n_act_तन्तवः,
            &kparams->div_n_act_तन्तवः,
            &kparams->div_ne00_padded,
            kparams->tile_आकार, kparams->aligned_tile_आकार, kparams->vtcm_आकार
        );
    }

    यदि (ret != 0) {
        FARF(ERROR, "HMX matmul failed (ret=%d)\n", ret);
        प्रत्यागम HTP_स्थितिः_आन्तरिकम्_ERR;
    }
    प्रत्यागम HTP_स्थितिः_OK;
}

स्थिर अन्तःस्थ शून्य htp_mm_tensor_collapse_पङ्क्तिकाः(संरचना htp_tensor * c, ध्रुव संरचना htp_tensor * t, uint32_t stride) {
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

    ध्रुव पूर्णाङ्क स्थितिः = htp_mm_init_context(octx, kparams);
    यदि (स्थितिः != HTP_स्थितिः_OK) {
        प्रत्यागम स्थितिः;
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
        htp_mm_tensor_collapse_पङ्क्तिकाः(&act_collapsed, act, s_act);
        htp_mm_tensor_collapse_पङ्क्तिकाः(&dst_collapsed, dst, sd);
        octx->src[1] = &act_collapsed;
        octx->dst    = &dst_collapsed;

        संरचना htp_tensor src2_collapsed;
        ध्रुव संरचना htp_tensor * src2 = octx->src[2];
        यदि (src2 && (src2->ne[1] * src2->ne[2] * src2->ne[3] > 1)) {
            ध्रुव uint32_t s2 = (src2->ne[1] > 1) ? src2->nb[1] : ((src2->ne[2] > 1) ? src2->nb[2] : src2->nb[3]);
            htp_mm_tensor_collapse_पङ्क्तिकाः(&src2_collapsed, src2, s2);
            octx->src[2] = &src2_collapsed;
        }

        ध्रुव पूर्णाङ्क स्थितिः = op_matmul_impl(octx);
        octx->src[1] = act;
        octx->dst    = dst;
        यदि (src2) {
            octx->src[2] = src2;
        }
        प्रत्यागम स्थितिः;
    }

    प्रत्यागम op_matmul_impl(octx);
}

स्थिर पूर्णाङ्क hmx_mm_op_matmul_id(
    संरचना htp_ops_context * octx,
    संरचना htp_mm_context * mmctx
) {
    ध्रुव uint32_t *                matrix_row_counts = mmctx->matrix_row_counts;
    ध्रुव संरचना mmid_row_mapping * matrix_पङ्क्तिकाः       = mmctx->matrix_पङ्क्तिकाः;
    htp_matmul_tensors_preamble;
    ध्रुव संरचना htp_mm_kernel_params * kparams = (ध्रुव संरचना htp_mm_kernel_params *) octx->kernel_params;
    ध्रुव पूर्णाङ्क n_ids = octx->src[2]->ne[0];
    ध्रुव पूर्णाङ्क n_as  = ne02;

    ध्रुव bool mdev_विभज = (octx->ctx->mdev.गणनम् > 1) && htp_tensor_can_row_partition(dst, परिमाण(भिन्न));
    यदि (octx->ctx->mdev.गणनम् > 1 && !mdev_विभज && octx->ctx->mdev.idx > 0) {
        प्रत्यागम HTP_स्थितिः_OK;
    }
    uint32_t n_सक्रिय = 0;
    यदि (mdev_विभज) {
        प्रति (uint32_t a = 0; a < (uint32_t) n_as; ++a) {
            यदि (matrix_row_counts[a] > 0) n_सक्रिय++;
        }
    }
    ध्रुव bool expert_विभज = mdev_विभज && (n_सक्रिय >= octx->ctx->mdev.गणनम्);

    uint32_t target_dev = 0;
    प्रति (uint32_t cur_a = 0; cur_a < (uint32_t) n_as; ++cur_a) {
        ध्रुव int32_t cne1 = matrix_row_counts[cur_a];
        यदि (cne1 == 0) अनुवर्त;

        ध्रुव पूर्णाङ्क m_padded = hex_align_ऊर्ध्वम्(cne1, 32);
        पूर्णाङ्क m_start = 0, m_अन्तः = m_padded;
        यदि (expert_विभज) {
            ध्रुव bool my_expert = (target_dev == octx->ctx->mdev.idx);
            यदि (++target_dev == octx->ctx->mdev.गणनम्) {
                target_dev = 0;
            }
            यदि (!my_expert) अनुवर्त;
        } अन्यथा यदि (mdev_विभज) {
            ध्रुव bool can_विभज = (uint32_t) cne1 >= octx->ctx->mdev.गणनम्;
            ध्रुव संरचना htp_tensor_mdev_range range = htp_tensor_mdev_partition((uint32_t) m_padded, can_विभज ? 1 : 0, octx->ctx->mdev.idx, octx->ctx->mdev.गणनम्, &octx->ctx->mdev.गणनम्_div);
            m_start = (पूर्णाङ्क) range.start;
            m_अन्तः   = (पूर्णाङ्क) (range.start + range.गणनम्);
        }
        यदि (m_start >= m_अन्तः) अनुवर्त;

        पूर्णाङ्क ret = hmx_mm_id_2d_f32(octx->ctx, octx->ctx->dma[0], (भिन्न*) dst->दत्त, (भिन्न*) act->दत्त,
                                   src0->दत्त + cur_a * nb02,
                                   cne1, ne00, ne01,
                                   ne10,
                                   ne11,
                                   nb11, nb12,
                                   nb1, nb2,
                                   (पूर्णाङ्क) src0->nb[1], (पूर्णाङ्क) src0->type,
                                   kparams->tile_आकार, kparams->aligned_tile_आकार,
                                   matrix_पङ्क्तिकाः, cur_a, mmctx->mapping_stride,
                                   m_start, m_अन्तः, (पूर्णाङ्क) octx->n_तन्तवः);
        यदि (ret != 0) {
            FARF(ERROR, "HMX matmul failed for expert %u, error %d\n", cur_a, ret);
            प्रत्यागम HTP_स्थितिः_NO_SUPPORT;
        }
    }

    प्रत्यागम HTP_स्थितिः_OK;
}

स्थिर पूर्णाङ्क hvx_mm_matmul_id(
    संरचना htp_ops_context * octx,
    संरचना htp_mm_context * mmctx,
    work_queue_func_t hvx_mmid_task_func
) {
    htp_matmul_tensors_preamble;
    ध्रुव uint32_t src0_row_आकार_padded = mmctx->src0_row_आकार_padded;
    ध्रुव uint32_t act_nrows            = mmctx->act_nrows;

    संरचना htp_thread_trace * tr = &octx->ctx->trace[0];
    htp_trace_event_start(tr, HTP_TRACE_EVT_INIT, 0);

    ध्रुव संरचना htp_mm_kernel_params * kparams = (ध्रुव संरचना htp_mm_kernel_params *) octx->kernel_params;
    ध्रुव संरचना htp_tensor * सीमित ids = octx->src[2];
    ध्रुव size_t src0_row_आकार = nb01;

    ध्रुव uint32_t qk = QK_Q8_0_TILED;
    ध्रुव uint32_t nb = (ne10 + qk - 1) / qk;
    ध्रुव uint32_t योग_nb = act_nrows * nb;

    work_queue_func_t quant_task_func;
    uint32_t n_quant_tasks = 1;
    यदि (act_nrows < octx->n_तन्तवः) {
        n_quant_tasks = MIN(योग_nb, octx->n_तन्तवः);
        quant_task_func = htp_mm_act_quant_खण्डम्_func(src0->type, act->type);
        प्रति (uint32_t ith = 0; ith < n_quant_tasks; ++ith) {
            uint32_t ib_first = (योग_nb * ith) / n_quant_tasks;
            uint32_t ib_चरमः  = (योग_nb * (ith + 1)) / n_quant_tasks;
            mmctx->quant_ib_first[ith] = ib_first;
            mmctx->quant_ib_चरमः[ith]  = ib_चरमः;
            mmctx->quant_r[ith]        = ib_first / nb;
            mmctx->quant_c[ith]        = ib_first % nb;
        }
    } अन्यथा {
        n_quant_tasks = MIN(act_nrows, octx->n_तन्तवः);
        quant_task_func = htp_mm_act_quant_row_func(src0->type, act->type);
    }
    size_t act_row_आकार = htp_mm_tiled_act_row_आकार(src0->type, ne10);

    संरचना htp_mm_hvx_vtcm_layout L;
    htp_mm_hvx_vtcm_layout_build(&L, kparams->kernel_type, src0->type, ne10, act_nrows, octx->n_तन्तवः,
                                 0, src0_row_आकार, act_row_आकार, 0, kparams->n_prefetch, true, false);

    ध्रुव size_t vtcm_आकार = L.योग_अष्टकानि;

    FARF(उच्चः_2, "matmul-id-%s : src0-spad-size %zu act-spad-size %zu bias-spad-size 0 dst-spad-size %zu (%zu)\n", mmctx->type,
         L.src0_अष्टकानि, L.act_अष्टकानि, L.dst_अष्टकानि, vtcm_आकार);

    FARF(उच्चः_2, "matmul-id-%s : %ux%ux%ux%u * %ux%ux%ux%u (%ux%ux%ux%u) -> %ux%ux%ux%u (0x%p, 0x%p, 0x%p)\n", mmctx->type,
         src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3], act->ne[0], act->ne[1], act->ne[2], act->ne[3],
         ids->ne[0], ids->ne[1], ids->ne[2], ids->ne[3], dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3], src0->दत्त,
         act->दत्त, dst->दत्त);

    // Make sure the reserved vtcm size is sufficient
    यदि (octx->ctx->vtcm_आकार < vtcm_आकार) {
        FARF(ERROR, "matmul-id-%s : current VTCM reservation %zu is too small, needed %zu\n", mmctx->type, octx->ctx->vtcm_आकार, vtcm_आकार);
        प्रत्यागम HTP_स्थितिः_VTCM_TOO_SMALL;
    }

    uint8_t * ध्रुव आधारः = (uint8_t *) octx->ctx->vtcm_आधारः;
    mmctx->vtcm_act      = VTCM_LAYOUT_PTR(uint8_t, आधारः, L.off_act);
    mmctx->vtcm_src0     = VTCM_LAYOUT_PTR(uint8_t, आधारः, L.off_src0);
    mmctx->vtcm_bias     = रिक्त;
    mmctx->vtcm_dst      = VTCM_LAYOUT_PTR(uint8_t, आधारः, L.off_dst);
    mmctx->vtcm_act_raw  = VTCM_LAYOUT_PTR(uint8_t, आधारः, L.off_act_raw);

    mmctx->vtcm_src0_stride    = src0_row_आकार_padded;
    mmctx->vtcm_act_stride     = act_row_आकार;
    mmctx->vtcm_act_raw_stride = hex_आवर्तनम्_ऊर्ध्वम्(ne10 * परिमाण(भिन्न), QK_Q8_0_TILED * परिमाण(भिन्न));

    mmctx->vtcm_src0_आकार_प्रत्येकम्_thread = fastdiv(L.src0_अष्टकानि, &octx->n_तन्तवः_div);
    mmctx->vtcm_act_आकार_प्रत्येकम्_thread  = L.act_अष्टकानि;
    mmctx->vtcm_dst_आकार_प्रत्येकम्_thread  = fastdiv(L.dst_अष्टकानि, &octx->n_तन्तवः_div);

    mmctx->cur_m_start = 0;
    mmctx->cur_m_पङ्क्तिकाः  = act_nrows;

    htp_trace_event_stop(tr, HTP_TRACE_EVT_INIT, 0);

    hvx_mm_transfer_act_dma(octx, kparams, act, mmctx->vtcm_act_raw, mmctx->vtcm_act_raw_stride, 0, act_nrows);

    mmctx->n_quant_पङ्क्तिकाः_प्रत्येकम्_thread = (act_nrows + n_quant_tasks - 1) / n_quant_tasks;
    mmctx->n_quant_tasks = n_quant_tasks;
    work_queue_सततम्(octx->ctx->work_queue, quant_task_func, mmctx, n_quant_tasks);

    work_queue_सततम्(octx->ctx->work_queue, hvx_mmid_task_func, mmctx, octx->n_तन्तवः);

    प्रत्यागम HTP_स्थितिः_OK;
}

स्थिर पूर्णाङ्क hmx_mm_op_matmul_id_nx(
    संरचना htp_ops_context * octx,
    संरचना htp_mm_context * mmctx
) {
    ध्रुव uint32_t *                matrix_row_counts = mmctx->matrix_row_counts;
    ध्रुव संरचना mmid_row_mapping * matrix_पङ्क्तिकाः       = mmctx->matrix_पङ्क्तिकाः;
    ध्रुव संरचना htp_mm_kernel_params * kparams = (ध्रुव संरचना htp_mm_kernel_params *) octx->kernel_params;
    ध्रुव uint32_t n_weights = kparams->n_weights;
    ध्रुव संरचना htp_tensor * सीमित src0 = octx->src[0];
    ध्रुव संरचना htp_tensor * सीमित act  = octx->src[n_weights];
    ध्रुव पूर्णाङ्क n_as = src0->ne[2];

    bool mdev_विभज = (octx->ctx->mdev.गणनम् > 1);
    प्रति (uint32_t p = 0; p < n_weights && mdev_विभज; ++p) {
        ध्रुव संरचना htp_tensor * सीमित dst = octx->dsts[p];
        mdev_विभज = htp_tensor_can_row_partition(dst, परिमाण(भिन्न));
    }
    यदि (octx->ctx->mdev.गणनम् > 1 && !mdev_विभज && octx->ctx->mdev.idx > 0) {
        प्रत्यागम HTP_स्थितिः_OK;
    }
    uint32_t n_सक्रिय = 0;
    यदि (mdev_विभज) {
        प्रति (uint32_t a = 0; a < (uint32_t) n_as; ++a) {
            यदि (matrix_row_counts[a] > 0) n_सक्रिय++;
        }
    }
    ध्रुव bool expert_विभज = mdev_विभज && (n_सक्रिय >= octx->ctx->mdev.गणनम्);

    uint32_t target_dev = 0;
    प्रति (uint32_t cur_a = 0; cur_a < (uint32_t) n_as; ++cur_a) {
        ध्रुव int32_t cne1 = matrix_row_counts[cur_a];
        यदि (cne1 == 0) अनुवर्त;

        ध्रुव पूर्णाङ्क m_padded = hex_align_ऊर्ध्वम्(cne1, 32);
        पूर्णाङ्क m_start = 0, m_अन्तः = m_padded;
        यदि (expert_विभज) {
            ध्रुव bool my_expert = (target_dev == octx->ctx->mdev.idx);
            यदि (++target_dev == octx->ctx->mdev.गणनम्) {
                target_dev = 0;
            }
            यदि (!my_expert) अनुवर्त;
        } अन्यथा यदि (mdev_विभज) {
            ध्रुव bool can_विभज = (uint32_t) cne1 >= octx->ctx->mdev.गणनम्;
            ध्रुव संरचना htp_tensor_mdev_range range = htp_tensor_mdev_partition((uint32_t) m_padded, can_विभज ? 1 : 0, octx->ctx->mdev.idx, octx->ctx->mdev.गणनम्, &octx->ctx->mdev.गणनम्_div);
            m_start = (पूर्णाङ्क) range.start;
            m_अन्तः   = (पूर्णाङ्क) (range.start + range.गणनम्);
        }
        यदि (m_start >= m_अन्तः) अनुवर्त;

        प्रति (uint32_t p = 0; p < n_weights; ++p) {
            ध्रुव संरचना htp_tensor * सीमित src_w = octx->src[p];
            ध्रुव संरचना htp_tensor * सीमित dst   = octx->dsts[p];
            यदि (!src_w || !dst) अनुवर्त;

            पूर्णाङ्क ret = hmx_mm_id_2d_f32(octx->ctx, octx->ctx->dma[0], (भिन्न*) dst->दत्त, (भिन्न*) act->दत्त,
                                       src_w->दत्त + cur_a * src_w->nb[2],
                                       cne1, src_w->ne[0], src_w->ne[1],
                                       act->ne[0],
                                       act->ne[1],
                                       act->nb[1], act->nb[2],
                                       dst->nb[1], dst->nb[2],
                                       (पूर्णाङ्क) src_w->nb[1], (पूर्णाङ्क) src_w->type,
                                       kparams->tile_आकार, kparams->aligned_tile_आकार,
                                       matrix_पङ्क्तिकाः, cur_a, mmctx->mapping_stride,
                                       m_start, m_अन्तः, (पूर्णाङ्क) octx->n_तन्तवः);
            यदि (ret != 0) {
                FARF(ERROR, "HMX matmul ID NX failed for expert %u weight %u, error %d\n", cur_a, p, ret);
                प्रत्यागम HTP_स्थितिः_NO_SUPPORT;
            }
        }
    }

    प्रत्यागम HTP_स्थितिः_OK;
}

स्थिर पूर्णाङ्क hvx_mm_matmul_id_nx(
    संरचना htp_ops_context * octx,
    संरचना htp_mm_context * mmctx,
    work_queue_func_t hvx_mmid_task_func
) {
    ध्रुव uint32_t src0_row_आकार_padded = mmctx->src0_row_आकार_padded;
    ध्रुव uint32_t act_nrows            = mmctx->act_nrows;

    संरचना htp_thread_trace * tr = &octx->ctx->trace[0];
    htp_trace_event_start(tr, HTP_TRACE_EVT_INIT, 0);

    ध्रुव संरचना htp_mm_kernel_params * kparams = (ध्रुव संरचना htp_mm_kernel_params *) octx->kernel_params;
    ध्रुव uint32_t n_weights = kparams->n_weights;
    ध्रुव संरचना htp_tensor * सीमित src0 = octx->src[0];
    ध्रुव संरचना htp_tensor * सीमित act  = octx->src[n_weights];
    ध्रुव संरचना htp_tensor * सीमित ids  = octx->src[n_weights + 1];
    ध्रुव size_t src0_row_आकार = src0->nb[1];

    ध्रुव uint32_t qk = QK_Q8_0_TILED;
    ध्रुव uint32_t nb = (act->ne[0] + qk - 1) / qk;
    ध्रुव uint32_t योग_nb = act_nrows * nb;

    work_queue_func_t quant_task_func;
    uint32_t n_quant_tasks = 1;
    यदि (act_nrows < octx->n_तन्तवः) {
        n_quant_tasks = MIN(योग_nb, octx->n_तन्तवः);
        quant_task_func = htp_mm_act_quant_खण्डम्_func(src0->type, act->type);
        प्रति (uint32_t ith = 0; ith < n_quant_tasks; ++ith) {
            uint32_t ib_first = (योग_nb * ith) / n_quant_tasks;
            uint32_t ib_चरमः  = (योग_nb * (ith + 1)) / n_quant_tasks;
            mmctx->quant_ib_first[ith] = ib_first;
            mmctx->quant_ib_चरमः[ith]  = ib_चरमः;
            mmctx->quant_r[ith]        = ib_first / nb;
            mmctx->quant_c[ith]        = ib_first % nb;
        }
    } अन्यथा {
        n_quant_tasks = MIN(act_nrows, octx->n_तन्तवः);
        quant_task_func = htp_mm_act_quant_row_func(src0->type, act->type);
    }
    size_t act_row_आकार = htp_mm_tiled_act_row_आकार(src0->type, act->ne[0]);

    संरचना htp_mm_hvx_vtcm_layout L;
    htp_mm_hvx_vtcm_layout_build(&L, kparams->kernel_type, src0->type, act->ne[0], act_nrows, octx->n_तन्तवः,
                                 0, src0_row_आकार, act_row_आकार, 0, kparams->n_prefetch, true, false);

    ध्रुव size_t vtcm_आकार = L.योग_अष्टकानि;

    यदि (octx->ctx->vtcm_आकार < vtcm_आकार) {
        FARF(ERROR, "matmul-id-nx: current VTCM reservation %zu is too small, needed %zu\n",
             octx->ctx->vtcm_आकार, vtcm_आकार);
        प्रत्यागम HTP_स्थितिः_VTCM_TOO_SMALL;
    }

    uint8_t * ध्रुव आधारः = (uint8_t *) octx->ctx->vtcm_आधारः;
    mmctx->vtcm_src0     = VTCM_LAYOUT_PTR(uint8_t, आधारः, L.off_src0);
    mmctx->vtcm_act      = VTCM_LAYOUT_PTR(uint8_t, आधारः, L.off_act);
    mmctx->vtcm_dst      = VTCM_LAYOUT_PTR(uint8_t, आधारः, L.off_dst);
    mmctx->vtcm_act_raw  = VTCM_LAYOUT_PTR(uint8_t, आधारः, L.off_act_raw);

    mmctx->vtcm_src0_stride    = 0;
    mmctx->vtcm_act_stride     = act_row_आकार;
    mmctx->vtcm_act_raw_stride = hex_आवर्तनम्_ऊर्ध्वम्(act->ne[0] * परिमाण(भिन्न), QK_Q8_0_TILED * परिमाण(भिन्न));

    mmctx->vtcm_src0_आकार_प्रत्येकम्_thread = fastdiv(L.src0_अष्टकानि, &octx->n_तन्तवः_div);
    mmctx->vtcm_act_आकार_प्रत्येकम्_thread  = L.act_अष्टकानि;
    mmctx->vtcm_dst_आकार_प्रत्येकम्_thread  = fastdiv(L.dst_अष्टकानि, &octx->n_तन्तवः_div);

    mmctx->cur_m_start = 0;
    mmctx->cur_m_पङ्क्तिकाः  = act_nrows;

    FARF(उच्चः_2, "matmul-id-nx: src0 %d:%d:%d type %s nrows %u, act %d:%d:%d nrows %u, vtcm %zu/%zu, threads %d\n",
         src0->ne[0], src0->ne[1], src0->ne[2], mmctx->type, src0->ne[1],
         act->ne[0], act->ne[1], act->ne[2], act_nrows,
         L.योग_अष्टकानि, octx->ctx->vtcm_आकार, octx->n_तन्तवः);

    htp_trace_event_stop(tr, HTP_TRACE_EVT_INIT, 0);

    hvx_mm_transfer_act_dma(octx, kparams, act, mmctx->vtcm_act_raw, mmctx->vtcm_act_raw_stride, 0, act_nrows);

    mmctx->n_quant_पङ्क्तिकाः_प्रत्येकम्_thread = (act_nrows + n_quant_tasks - 1) / n_quant_tasks;
    mmctx->n_quant_tasks           = n_quant_tasks;
    work_queue_सततम्(octx->ctx->work_queue, quant_task_func, mmctx, n_quant_tasks);

    work_queue_सततम्(octx->ctx->work_queue, hvx_mmid_task_func, mmctx, octx->n_तन्तवः);

    प्रत्यागम HTP_स्थितिः_OK;
}

स्थिर अन्तःस्थ शून्य अवलोकनम्_expert_ids_n(
    ध्रुव संरचना htp_tensor * ids,
    ध्रुव uint32_t n_ids,
    uint32_t n_as,
    uint32_t * counts,
    संरचना mmid_row_mapping * matrix_पङ्क्तिकाः,
    uint32_t mapping_stride
) {
    ध्रुव size_t ids_nb1 = ids->nb[1];
    ध्रुव uint8_t * ids_दत्त = (ध्रुव uint8_t *) ids->दत्त;

    प्रति (uint32_t iid1 = 0; iid1 < ids->ne[1]; ++iid1) {
        ध्रुव int32_t * row_ptr = (ध्रुव int32_t *) (ids_दत्त + iid1 * ids_nb1);
        प्रति (uint32_t id = 0; id < n_ids; ++id) {
            ध्रुव int32_t i02 = row_ptr[id];
            यदि (i02 < 0) {
                अनुवर्त;
            }
            प्रतिज्ञा(i02 < n_as);

            यदि (matrix_पङ्क्तिकाः) {
                matrix_पङ्क्तिकाः[i02 * mapping_stride + counts[i02]] = (संरचना mmid_row_mapping) { id, iid1 };
            }
            counts[i02] += 1;
        }
    }
}

स्थिर अन्तःस्थ शून्य अवलोकनम्_expert_ids(
    ध्रुव संरचना htp_tensor * ids,
    uint32_t n_ids,
    uint32_t n_as,
    uint32_t * counts,
    संरचना mmid_row_mapping * matrix_पङ्क्तिकाः,
    uint32_t mapping_stride
) {
    ध्रुव size_t ids_nb0 = ids->nb[0];

    यदि (ids_nb0 == 4) {
        विकल्प (n_ids) {
            पक्ष 8:  अवलोकनम्_expert_ids_n(ids, 8,     n_as, counts, matrix_पङ्क्तिकाः, mapping_stride); भङ्ग;
            पक्ष 4:  अवलोकनम्_expert_ids_n(ids, 4,     n_as, counts, matrix_पङ्क्तिकाः, mapping_stride); भङ्ग;
            पक्ष 2:  अवलोकनम्_expert_ids_n(ids, 2,     n_as, counts, matrix_पङ्क्तिकाः, mapping_stride); भङ्ग;
            स्वभाव: अवलोकनम्_expert_ids_n(ids, n_ids, n_as, counts, matrix_पङ्क्तिकाः, mapping_stride); भङ्ग;
        }
    } अन्यथा {
        // Strided fallback
        ध्रुव size_t ids_nb1 = ids->nb[1];
        ध्रुव uint8_t * ids_दत्त = (ध्रुव uint8_t *) ids->दत्त;
        प्रति (uint32_t iid1 = 0; iid1 < ids->ne[1]; ++iid1) {
            ध्रुव int32_t * row_ptr = (ध्रुव int32_t *) (ids_दत्त + iid1 * ids_nb1);
            प्रति (uint32_t id = 0; id < n_ids; ++id) {
                ध्रुव int32_t i02 = *(ध्रुव int32_t *) ((ध्रुव uint8_t *) row_ptr + id * ids_nb0);
                यदि (i02 < 0) {
                    अनुवर्त;
                }
                प्रतिज्ञा(i02 < n_as);

                यदि (matrix_पङ्क्तिकाः) {
                    matrix_पङ्क्तिकाः[i02 * mapping_stride + counts[i02]] = (संरचना mmid_row_mapping) { id, iid1 };
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

    ध्रुव पूर्णाङ्क स्थितिः = htp_mm_init_context(octx, kparams);
    यदि (स्थितिः != HTP_स्थितिः_OK) {
        प्रत्यागम स्थितिः;
    }

    संरचना htp_thread_trace * tr = &octx->ctx->trace[0];
    htp_trace_event_start(tr, HTP_TRACE_EVT_INIT, 0);

    mmctx->octx = octx;
    mmctx->act = act;

    ध्रुव संरचना htp_tensor * सीमित ids = octx->src[2];
    यदि (htp_tensor_is_extended(ids) || htp_tensor_is_extended(act) || htp_tensor_is_extended(dst)) {
        प्रत्यागम HTP_स्थितिः_NO_SUPPORT;
    }

    ध्रुव size_t src0_row_आकार = nb01;
    ध्रुव size_t dst_row_आकार  = nb1;

    ध्रुव size_t src0_row_आकार_padded = hex_आवर्तनम्_ऊर्ध्वम्(src0_row_आकार, 128);

    ध्रुव uint32_t src0_nrows = ne01;  // per expert
    ध्रुव uint32_t act_nrows  = ne11 * ne12 * ne13;

    // row groups
    ध्रुव पूर्णाङ्क n_ids = ids->ne[0];  // n_expert_used
    ध्रुव पूर्णाङ्क n_as  = ne02;        // n_expert

    uint8_t  * mapping_buf       = octx->ctx->ddr_spad_आधारः;
    uint32_t   mapping_stride    = 1;
    uint32_t * matrix_row_counts = (uint32_t *) mapping_buf;
    संरचना mmid_row_mapping * matrix_पङ्क्तिकाः = रिक्त;

    यदि (act_nrows > 1) {
        ध्रुव size_t matrix_row_counts_आकार = n_as * परिमाण(uint32_t);
        प्रतिज्ञा(octx->ctx->ddr_spad_आकार >= matrix_row_counts_आकार);

        hex_l2fetch_खण्डम्((ध्रुव शून्य *) ids->दत्त, ids->ne[1] * ids->nb[1]);

        स्मृतिं_पूरय(matrix_row_counts, 0, matrix_row_counts_आकार);
        अवलोकनम्_expert_ids(ids, n_ids, n_as, matrix_row_counts, रिक्त, 0);

        uint32_t max_गणनम् = hvx_reduce_max_i32((ध्रुव uint8_t *) matrix_row_counts, n_as);
        mapping_stride = max_गणनम् > 0 ? max_गणनम् : 1;

        size_t matrix_row_map_आकार  = n_as * mapping_stride * परिमाण(संरचना mmid_row_mapping);
        ध्रुव size_t योग_map_आकार = matrix_row_counts_आकार + matrix_row_map_आकार;

        यदि (योग_map_आकार > octx->ctx->ddr_spad_आकार) {
            mapping_buf = memalign(128, योग_map_आकार);
            यदि (!mapping_buf) {
                प्रत्यागम HTP_स्थितिः_आन्तरिकम्_ERR;
            }
        }

        matrix_row_counts = (uint32_t *) mapping_buf;
        matrix_पङ्क्तिकाः       = (संरचना mmid_row_mapping *) (mapping_buf + matrix_row_counts_आकार);

        स्मृतिं_पूरय(matrix_row_counts, 0, n_as * परिमाण(uint32_t));
        अवलोकनम्_expert_ids(ids, n_ids, n_as, matrix_row_counts, matrix_पङ्क्तिकाः, mapping_stride);
    }

    mmctx->matrix_row_counts    = matrix_row_counts;
    mmctx->matrix_पङ्क्तिकाः          = matrix_पङ्क्तिकाः;
    mmctx->mapping_stride       = mapping_stride;
    mmctx->mm_div_ne11          = kparams->div_ne1;
    mmctx->src0_row_आकार_padded = src0_row_आकार_padded;
    mmctx->act_nrows            = act_nrows;
    mmctx->cur_m_start          = 0;
    mmctx->cur_m_पङ्क्तिकाः           = act_nrows;

    htp_trace_event_stop(tr, HTP_TRACE_EVT_INIT, 0);

    पूर्णाङ्क s;
    यदि (kparams->n_hmx) {
        s = hmx_mm_op_matmul_id(octx, mmctx);
    } अन्यथा {
        uint32_t src0_row_start = 0;
        uint32_t src0_row_अन्तः   = src0_nrows;
        यदि (octx->ctx->mdev.गणनम् > 1) {
            ध्रुव bool can_विभज = htp_tensor_can_row_partition(dst, परिमाण(भिन्न));
            ध्रुव संरचना htp_tensor_mdev_range range = htp_tensor_mdev_partition(src0_nrows, can_विभज ? 32 : 0, octx->ctx->mdev.idx, octx->ctx->mdev.गणनम्, &octx->ctx->mdev.गणनम्_div);
            src0_row_start = range.start;
            src0_row_अन्तः   = range.start + range.गणनम्;
        }

        यदि (src0_row_start >= src0_row_अन्तः) {
            यदि (mapping_buf != octx->ctx->ddr_spad_आधारः) {
                मुक्त(mapping_buf);
            }
            प्रत्यागम HTP_स्थितिः_OK;
        }

        ध्रुव uint32_t nrows = src0_row_अन्तः - src0_row_start;
        mmctx->src0_row_start = src0_row_start;
        mmctx->src0_row_अन्तः   = src0_row_अन्तः;

        mmctx->src0_nrows_प्रत्येकम्_thread = fastdiv(nrows + octx->n_तन्तवः - 1, &octx->n_तन्तवः_div);
        mmctx->src0_nrows_प्रत्येकम्_thread = hex_आवर्तनम्_ऊर्ध्वम्(mmctx->src0_nrows_प्रत्येकम्_thread, 32);

        यदि (hvx_mm_init_tiled_vec_dot(mmctx, src0->type) == 0) {
            s = hvx_mm_matmul_id(octx, mmctx, act_nrows > 1 ? hvx_mm_id_tiled : hvx_mv_id_tiled);
        } अन्यथा {
            s = HTP_स्थितिः_NO_SUPPORT;
        }
    }

    यदि (mapping_buf != octx->ctx->ddr_spad_आधारः) {
        मुक्त(mapping_buf);
    }

    प्रत्यागम s;
}

पूर्णाङ्क op_matmul_id_nx(संरचना htp_ops_context * octx) {
    ध्रुव संरचना htp_mm_kernel_params * kparams = (ध्रुव संरचना htp_mm_kernel_params *) octx->kernel_params;
    संरचना htp_mm_context mmctx_struct = {0};
    संरचना htp_mm_context * mmctx = &mmctx_struct;

    ध्रुव पूर्णाङ्क स्थितिः = htp_mm_init_context(octx, kparams);
    यदि (स्थितिः != HTP_स्थितिः_OK) {
        प्रत्यागम स्थितिः;
    }

    संरचना htp_thread_trace * tr = &octx->ctx->trace[0];
    htp_trace_event_start(tr, HTP_TRACE_EVT_INIT, 0);

    mmctx->octx = octx;
    ध्रुव uint32_t n_weights = kparams->n_weights;
    ध्रुव संरचना htp_tensor * सीमित src0 = octx->src[0];
    ध्रुव संरचना htp_tensor * सीमित act  = octx->src[n_weights];
    ध्रुव संरचना htp_tensor * सीमित ids  = octx->src[n_weights + 1];
    यदि (htp_tensor_is_extended(ids) || htp_tensor_is_extended(act)) {
        प्रत्यागम HTP_स्थितिः_NO_SUPPORT;
    }
    प्रति (uint32_t p = 0; p < n_weights; p++) {
        यदि (htp_tensor_is_extended(octx->dsts[p])) {
            प्रत्यागम HTP_स्थितिः_NO_SUPPORT;
        }
    }

    mmctx->act = act;

    ध्रुव size_t src0_row_आकार = src0->nb[1];
    ध्रुव size_t src0_row_आकार_padded = hex_आवर्तनम्_ऊर्ध्वम्(src0_row_आकार, 128);

    ध्रुव uint32_t act_nrows = act->ne[1] * act->ne[2] * act->ne[3];

    ध्रुव पूर्णाङ्क n_ids = ids->ne[0];
    ध्रुव पूर्णाङ्क n_as  = src0->ne[2];

    uint8_t  * mapping_buf       = octx->ctx->ddr_spad_आधारः;
    uint32_t   mapping_stride    = 1;
    uint32_t * matrix_row_counts = (uint32_t *) mapping_buf;
    संरचना mmid_row_mapping * matrix_पङ्क्तिकाः = रिक्त;

    यदि (act_nrows > 1) {
        ध्रुव size_t matrix_row_counts_आकार = n_as * परिमाण(uint32_t);
        प्रतिज्ञा(octx->ctx->ddr_spad_आकार >= matrix_row_counts_आकार);

        hex_l2fetch_खण्डम्((ध्रुव शून्य *) ids->दत्त, ids->ne[1] * ids->nb[1]);

        स्मृतिं_पूरय(matrix_row_counts, 0, matrix_row_counts_आकार);
        अवलोकनम्_expert_ids(ids, n_ids, n_as, matrix_row_counts, रिक्त, 0);

        uint32_t max_गणनम् = hvx_reduce_max_i32((ध्रुव uint8_t *) matrix_row_counts, n_as);
        mapping_stride = max_गणनम् > 0 ? max_गणनम् : 1;

        size_t matrix_row_map_आकार  = n_as * mapping_stride * परिमाण(संरचना mmid_row_mapping);
        ध्रुव size_t योग_map_आकार = matrix_row_counts_आकार + matrix_row_map_आकार;

        यदि (योग_map_आकार > octx->ctx->ddr_spad_आकार) {
            mapping_buf = memalign(128, योग_map_आकार);
            यदि (!mapping_buf) {
                प्रत्यागम HTP_स्थितिः_आन्तरिकम्_ERR;
            }
        }

        matrix_row_counts = (uint32_t *) mapping_buf;
        matrix_पङ्क्तिकाः       = (संरचना mmid_row_mapping *) (mapping_buf + matrix_row_counts_आकार);

        स्मृतिं_पूरय(matrix_row_counts, 0, n_as * परिमाण(uint32_t));
        अवलोकनम्_expert_ids(ids, n_ids, n_as, matrix_row_counts, matrix_पङ्क्तिकाः, mapping_stride);
    }

    mmctx->matrix_row_counts    = matrix_row_counts;
    mmctx->matrix_पङ्क्तिकाः          = matrix_पङ्क्तिकाः;
    mmctx->mapping_stride       = mapping_stride;
    mmctx->mm_div_ne11          = kparams->div_ne1;
    mmctx->src0_row_आकार_padded = src0_row_आकार_padded;
    mmctx->act_nrows            = act_nrows;
    mmctx->cur_m_start          = 0;
    mmctx->cur_m_पङ्क्तिकाः           = act_nrows;

    htp_trace_event_stop(tr, HTP_TRACE_EVT_INIT, 0);

    पूर्णाङ्क s;
    यदि (kparams->n_hmx) {
        s = hmx_mm_op_matmul_id_nx(octx, mmctx);
    } अन्यथा {
        यदि (hvx_mm_init_tiled_vec_dot(mmctx, src0->type) == 0) {
            s = hvx_mm_matmul_id_nx(octx, mmctx, act_nrows > 1 ? hvx_mm_id_nx_tiled : hvx_mv_id_nx_tiled);
        } अन्यथा {
            s = HTP_स्थितिः_NO_SUPPORT;
        }
    }

    यदि (mapping_buf != octx->ctx->ddr_spad_आधारः) {
        मुक्त(mapping_buf);
    }

    प्रत्यागम s;
}
स्थिर पूर्णाङ्क op_matmul_nx_impl(संरचना htp_ops_context * octx) {
    ध्रुव संरचना htp_mm_kernel_params * kparams = (ध्रुव संरचना htp_mm_kernel_params *) octx->kernel_params;

    ध्रुव पूर्णाङ्क स्थितिः = htp_mm_init_context(octx, kparams);
    यदि (स्थितिः != HTP_स्थितिः_OK) {
        प्रत्यागम स्थितिः;
    }

    यदि (kparams->n_hmx) {
        प्रत्यागम hmx_mm_nx_2d_f32(octx, kparams);
    }

    संरचना htp_thread_trace * tr = &octx->ctx->trace[0];
    htp_trace_event_start(tr, HTP_TRACE_EVT_INIT, 0);

    ध्रुव uint32_t n_weights = kparams->n_weights;

    ध्रुव संरचना htp_tensor * सीमित src0 = octx->src[0]; // first weight
    ध्रुव संरचना htp_tensor * सीमित act  = octx->src[n_weights]; // activation x

    संरचना htp_mm_context mmctx_struct = {0};
    संरचना htp_mm_context * mmctx = &mmctx_struct;
    mmctx->octx = octx;
    mmctx->act  = act;

    ध्रुव uint32_t act_nrows = act->ne[1] * act->ne[2] * act->ne[3];
    mmctx->act_nrows   = act_nrows;
    mmctx->cur_m_start = 0;
    mmctx->cur_m_पङ्क्तिकाः  = act_nrows;

    ध्रुव size_t src0_row_आकार = src0->nb[1];

    यदि (hvx_mm_init_tiled_vec_dot(mmctx, src0->type) != 0) {
        प्रत्यागम HTP_स्थितिः_NO_SUPPORT;
    }

    ध्रुव uint32_t qk = QK_Q8_0_TILED;
    ध्रुव uint32_t nb = (act->ne[0] + qk - 1) / qk;
    ध्रुव uint32_t योग_nb = act_nrows * nb;

    कार्यकर्ता_callback_t quant_task_func;
    uint32_t n_quant_tasks = 1;
    यदि (act_nrows < octx->n_तन्तवः) {
        n_quant_tasks = MIN(योग_nb, octx->n_तन्तवः);
        quant_task_func = htp_mm_act_quant_खण्डम्_func(src0->type, act->type);
        प्रति (uint32_t ith = 0; ith < n_quant_tasks; ++ith) {
            uint32_t ib_first = (योग_nb * ith) / n_quant_tasks;
            uint32_t ib_चरमः  = (योग_nb * (ith + 1)) / n_quant_tasks;
            mmctx->quant_ib_first[ith] = ib_first;
            mmctx->quant_ib_चरमः[ith]  = ib_चरमः;
            mmctx->quant_r[ith]        = ib_first / nb;
            mmctx->quant_c[ith]        = ib_first % nb;
        }
    } अन्यथा {
        n_quant_tasks = MIN(act_nrows, octx->n_तन्तवः);
        quant_task_func = htp_mm_act_quant_row_func(src0->type, act->type);
    }

    ध्रुव size_t act_row_आकार = htp_mm_tiled_act_row_आकार(src0->type, act->ne[0]);

    संरचना htp_mm_hvx_vtcm_layout L;
    htp_mm_hvx_vtcm_layout_build(&L, kparams->kernel_type, src0->type, act->ne[0], act_nrows, octx->n_तन्तवः,
                                 0, src0_row_आकार, act_row_आकार, 0, kparams->n_prefetch, false, true);

    ध्रुव size_t vtcm_आकार = L.योग_अष्टकानि;

    यदि (octx->ctx->vtcm_आकार < vtcm_आकार) {
        FARF(ERROR, "matmul-nx: current VTCM reservation %zu is too small, needed %zu\n",
             octx->ctx->vtcm_आकार, vtcm_आकार);
        प्रत्यागम HTP_स्थितिः_VTCM_TOO_SMALL;
    }

    uint8_t * ध्रुव आधारः = (uint8_t *) octx->ctx->vtcm_आधारः;
    mmctx->vtcm_src0     = VTCM_LAYOUT_PTR(uint8_t, आधारः, L.off_src0);
    mmctx->vtcm_act      = VTCM_LAYOUT_PTR(uint8_t, आधारः, L.off_act);
    mmctx->vtcm_dst      = VTCM_LAYOUT_PTR(uint8_t, आधारः, L.off_dst);
    mmctx->vtcm_act_raw  = VTCM_LAYOUT_PTR(uint8_t, आधारः, L.off_act_raw);

    mmctx->vtcm_src0_stride    = 0;
    mmctx->vtcm_act_stride     = act_row_आकार;
    mmctx->vtcm_act_raw_stride = hex_आवर्तनम्_ऊर्ध्वम्(act->ne[0] * परिमाण(भिन्न), QK_Q8_0_TILED * परिमाण(भिन्न));

    mmctx->vtcm_src0_आकार_प्रत्येकम्_thread = fastdiv(L.src0_अष्टकानि, &octx->n_तन्तवः_div);
    mmctx->vtcm_act_आकार_प्रत्येकम्_thread  = L.act_अष्टकानि;
    mmctx->vtcm_dst_आकार_प्रत्येकम्_thread  = fastdiv(L.dst_अष्टकानि, &octx->n_तन्तवः_div);

    htp_trace_event_stop(tr, HTP_TRACE_EVT_INIT, 0);

    hvx_mm_transfer_act_dma(octx, kparams, act, mmctx->vtcm_act_raw, mmctx->vtcm_act_raw_stride, 0, act_nrows);

    mmctx->n_quant_पङ्क्तिकाः_प्रत्येकम्_thread = (act_nrows + n_quant_tasks - 1) / n_quant_tasks;
    mmctx->n_quant_tasks = n_quant_tasks;
    work_queue_सततम्(octx->ctx->work_queue, quant_task_func, mmctx, n_quant_tasks);

    work_queue_सततम्(octx->ctx->work_queue, hvx_mm_nx_2d_tiled, mmctx, octx->n_तन्तवः);

    प्रत्यागम HTP_स्थितिः_OK;
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

        htp_mm_tensor_collapse_पङ्क्तिकाः(&act_collapsed, act, s1);
        octx->src[n_weights] = &act_collapsed;

        प्रति (uint32_t p = 0; p < n_weights; p++) {
            orig_dsts[p] = octx->dsts[p];
            ध्रुव संरचना htp_tensor * d = octx->dsts[p];
            ध्रुव uint32_t sd = (d->ne[1] > 1) ? d->nb[1] : ((d->ne[2] > 1) ? d->nb[2] : d->nb[3]);
            htp_mm_tensor_collapse_पङ्क्तिकाः(&dsts_collapsed[p], d, sd);
            octx->dsts[p] = &dsts_collapsed[p];
        }

        ध्रुव पूर्णाङ्क स्थितिः = op_matmul_nx_impl(octx);

        octx->src[n_weights] = act;
        प्रति (uint32_t p = 0; p < n_weights; p++) {
            octx->dsts[p] = orig_dsts[p];
        }
        प्रत्यागम स्थितिः;
    }

    प्रत्यागम op_matmul_nx_impl(octx);
}
