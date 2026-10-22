#సూచన clang diagnostic ignored "-Wgnu-zero-variadic-macro-arguments"
#సూచన clang diagnostic ignored "-Wunused-function"
#సూచన clang diagnostic ignored "-Wunused-variable"
#సూచన clang diagnostic ignored "-Wunused-but-set-variable"

#అన్తర్భావ <HAP_farf.h>
#అన్తర్భావ <HAP_perf.h>
#అన్తర్భావ <HAP_compute_res.h>

#అన్తర్భావ <math.h>
#అన్తర్భావ <string.h>
#అన్తర్భావ <stdatomic.h>

#అన్తర్భావ "dma-queue.h"
#అన్తర్భావ "hvx-utils.h"
#అన్తర్భావ "hvx-dump.h"
#అన్తర్భావ "hvx-arith.h"
#అన్తర్భావ "hvx-reduce.h"

#నిర్వచించు GGML_COMMON_DECL_C
#అన్తర్భావ "ggml-common.h"
#అన్తర్భావ "htp-ctx.h"
#అన్తర్భావ "htp-ops.h"
#అన్తర్భావ "htp-tensor.h"
#అన్తర్భావ "matmul-ops.h"
#అన్తర్భావ "htp-vtcm.h"

రకనిర్వచనం నిర్మాణం {
    దశాంశం        *dst;
    dma_addr_t    bias_addr;
    size_t        bias_బైట్లు;
    dma_addr_t    act_dma_addr;
    dma_addr_t    బరువు;
    dma_వరుస *   బరువు_dma;
    పూర్ణాంకం           m;
    పూర్ణాంకం           k;
    పూర్ణాంకం           n;
    పూర్ణాంకం           act_stride;
    uint32_t      act_elem_కొలత;  // 4=F32 src1, 2=F16 src1
    పూర్ణాంకం           బరువు_stride;
    పూర్ణాంకం           బరువు_type;
    పూర్ణాంకం           dst_stride;
    పూర్ణాంకం           dst_నిలువుల_సంఖ్య;
    uint32_t      bias_stride;
    పూర్ణాంకం           tile_కొలత;
    పూర్ణాంకం           aligned_tile_కొలత;
    పూర్ణాంకం           ne02;
    పూర్ణాంకం           ne03;
    పూర్ణాంకం           ne12;
    పూర్ణాంకం           ne13;
    size_t        src0_nb2;
    size_t        src0_nb3;
    size_t        act_nb2;
    size_t        act_nb3;
    size_t        bias_nb2;
    size_t        bias_nb3;
    size_t        dst_nb2;
    size_t        dst_nb3;
    పూర్ణాంకం           r2;
    పూర్ణాంకం           r3;
    నిర్మాణం fastdiv_విలువలు div_r2;
    నిర్మాణం fastdiv_విలువలు div_r3;
} hmx_mm_batched_params_t;

స్థిర bool htp_matmul_has_extended_బరువు(స్థిరరాశి నిర్మాణం htp_ops_context * octx, uint32_t n_weights) {
    కొరకు (uint32_t i = 0; i < n_weights; ++i) {
        ఒకవేళ (htp_tensor_is_extended(octx->src[i])) {
            తిరిగి true;
        }
    }
    తిరిగి false;
}

నిర్మాణం htp_mm_context {
    స్థిరరాశి అక్షరం * type;
    నిర్మాణం htp_ops_context * octx;
    స్థిరరాశి నిర్మాణం htp_tensor * act;

    శూన్య (*vec_dot_1x1)(స్థిరరాశి uint32_t n, దశాంశం * పరిమిత s0,
         స్థిరరాశి శూన్య * పరిమిత vx0,
         స్థిరరాశి శూన్య * పరిమిత vy0);

    శూన్య (*vec_dot_2x1)(స్థిరరాశి uint32_t n, దశాంశం * పరిమిత s0,
         స్థిరరాశి శూన్య * పరిమిత vx0, స్థిరరాశి శూన్య * పరిమిత vx1,
         స్థిరరాశి శూన్య * పరిమిత vy0);

    శూన్య (*vec_dot_2x2)(స్థిరరాశి uint32_t n, దశాంశం * పరిమిత s0, దశాంశం * పరిమిత s1,
         స్థిరరాశి శూన్య * పరిమిత vx0, స్థిరరాశి శూన్య * పరిమిత vx1,
         స్థిరరాశి శూన్య * పరిమిత vy0, స్థిరరాశి శూన్య * పరిమిత vy1);

    శూన్య (*vec_dot_32x1)(స్థిరరాశి uint32_t n, దశాంశం * పరిమిత s,
         స్థిరరాశి శూన్య * పరిమిత vx,
         స్థిరరాశి శూన్య * పరిమిత vy, uint32_t valid_అడ్డువరుసలు,
         స్థిరరాశి దశాంశం * పరిమిత sz);

    శూన్య (*vec_dot_32x2)(స్థిరరాశి uint32_t n, దశాంశం * పరిమిత s0, దశాంశం * పరిమిత s1,
         స్థిరరాశి శూన్య * పరిమిత vx,
         స్థిరరాశి శూన్య * పరిమిత vy0, స్థిరరాశి శూన్య * పరిమిత vy1, uint32_t valid_అడ్డువరుసలు,
         స్థిరరాశి దశాంశం * పరిమిత sz0, స్థిరరాశి దశాంశం * పరిమిత sz1);

    // Precomputed values
    uint32_t src0_nrows_చొప్పున_thread;
    uint32_t src0_అడ్డువరుస_ప్రారంభం;
    uint32_t src0_అడ్డువరుస_చివర;
    uint32_t src0_అడ్డువరుస_కొలత_padded;
    uint32_t act_nrows;
    uint32_t cur_m_ప్రారంభం;
    uint32_t cur_m_అడ్డువరుసలు;

    నిర్మాణం fastdiv_విలువలు mm_div_ne12_ne1;
    నిర్మాణం fastdiv_విలువలు mm_div_ne1;
    నిర్మాణం fastdiv_విలువలు mm_div_r2;
    నిర్మాణం fastdiv_విలువలు mm_div_r3;
    నిర్మాణం fastdiv_విలువలు mm_div_ne11;

    // Precomputed block-parallel quantization values
    uint32_t          quant_ib_మొదటి[WORK_వరుస_MAX_N_దారాలు];
    uint32_t          quant_ib_చివరి[WORK_వరుస_MAX_N_దారాలు];
    uint32_t          quant_r[WORK_వరుస_MAX_N_దారాలు];
    uint32_t          quant_c[WORK_వరుస_MAX_N_దారాలు];
    uint32_t          n_quant_tasks;
    uint32_t          n_quant_అడ్డువరుసలు_చొప్పున_thread;

    // Fields for scattered mapping & HMX support in MUL_MAT_ID
    స్థిరరాశి uint32_t * మాతృక_అడ్డువరుస_counts;
    స్థిరరాశి నిర్మాణం mmid_అడ్డువరుస_mapping * మాతృక_అడ్డువరుసలు;
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
    uint32_t vtcm_src0_కొలత_చొప్పున_thread;
    uint32_t vtcm_act_కొలత_చొప్పున_thread;
    uint32_t vtcm_dst_కొలత_చొప్పున_thread;
};

స్థిర పూర్ణాంకం htp_mm_init_context(
    నిర్మాణం htp_ops_context * octx,
    స్థిరరాశి నిర్మాణం htp_mm_kernel_params * kparams
) {
    ఒకవేళ (!htp_ops_context_సమితి_n_దారాలు(octx, (uint32_t) kparams->n_దారాలు)) {
        తిరిగి HTP_స్థితి_INVAL_PARAMS;
    }

    ఒకవేళ (kparams->n_hmx) {
        ఒకవేళ (kparams->n_act_దారాలు <= 0 || kparams->n_act_దారాలు > (int32_t) octx->n_దారాలు) {
            తిరిగి HTP_స్థితి_INVAL_PARAMS;
        }
    }

    తిరిగి HTP_స్థితి_OK;
}

// vdelta control to expand first 32 e8m0 values into 32 uint32 elements
స్థిర స్థిరరాశి uint8_t __attribute__((aligned(128))) expand_x32_e8m0[128] = {
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
స్థిర స్థిరరాశి uint8_t __attribute__((aligned(VLEN))) kvalues_iq4nl_lut[] = {
    0x81, 0, 0x98, 0, 0xAD, 0, 0xBF, 0, 0xCF, 0, 0xDD, 0, 0xEA, 0, 0xF6, 0, 0x01, 0, 0x0D, 0, 0x19, 0, 0x26, 0,
    0x35, 0, 0x45, 0, 0x59, 0, 0x71, 0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0,
    0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0,
    0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0,
    0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0,
};

స్థిర స్థిరరాశి uint8_t __attribute__((aligned(VLEN))) kvalues_mxfp4_lut[] = {
    0,    0, 1,    0, 2,    0, 3, 0, 4, 0, 6, 0, 8, 0, 12, 0, 0, 0, 0xff, 0, 0xfe, 0, 0xfd, 0, 0xfc, 0,
    0xfa, 0, 0xf8, 0, 0xf4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 0,    0, 0,    0, 0,    0, 0,    0,
    0,    0, 0,    0, 0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 0,    0, 0,    0, 0,    0, 0,    0,
    0,    0, 0,    0, 0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 0,    0, 0,    0, 0,    0, 0,    0,
    0,    0, 0,    0, 0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 0,    0, 0,    0, 0,    0,
};

#నిర్వచించు htp_matmul_tensors_preamble                                 \
    స్థిరరాశి నిర్మాణం htp_tensor * పరిమిత src0 = octx->src[0];         \
    స్థిరరాశి నిర్మాణం htp_tensor * పరిమిత src1 = octx->src[1];         \
    స్థిరరాశి నిర్మాణం htp_tensor * పరిమిత src2 = octx->src[2];         \
    స్థిరరాశి నిర్మాణం htp_tensor * పరిమిత  dst = octx->dst;            \
    స్థిరరాశి నిర్మాణం htp_tensor * పరిమిత  act = src1;                 \
                                                                    \
    స్థిరరాశి uint32_t ne00 = src0->ne[0];                              \
    స్థిరరాశి uint32_t ne01 = src0->ne[1];                              \
    స్థిరరాశి uint32_t ne02 = src0->ne[2];                              \
    స్థిరరాశి uint32_t ne03 = src0->ne[3];                              \
                                                                    \
    స్థిరరాశి uint32_t ne10 = src1->ne[0];                              \
    స్థిరరాశి uint32_t ne11 = src1->ne[1];                              \
    స్థిరరాశి uint32_t ne12 = src1->ne[2];                              \
    స్థిరరాశి uint32_t ne13 = src1->ne[3];                              \
                                                                    \
    స్థిరరాశి uint32_t ne20 = src2 ? src2->ne[0] : 0;                   \
    స్థిరరాశి uint32_t ne21 = src2 ? src2->ne[1] : 0;                   \
    స్థిరరాశి uint32_t ne22 = src2 ? src2->ne[2] : 0;                   \
    స్థిరరాశి uint32_t ne23 = src2 ? src2->ne[3] : 0;                   \
                                                                    \
    స్థిరరాశి uint32_t ne0 = dst->ne[0];                                \
    స్థిరరాశి uint32_t ne1 = dst->ne[1];                                \
    స్థిరరాశి uint32_t ne2 = dst->ne[2];                                \
    స్థిరరాశి uint32_t ne3 = dst->ne[3];                                \
                                                                    \
    స్థిరరాశి uint32_t nb00 = src0->nb[0];                              \
    స్థిరరాశి uint32_t nb01 = src0->nb[1];                              \
    స్థిరరాశి uint32_t nb02 = src0->nb[2];                              \
    స్థిరరాశి uint32_t nb03 = src0->nb[3];                              \
                                                                    \
    స్థిరరాశి uint32_t nb10 = src1->nb[0];                              \
    స్థిరరాశి uint32_t nb11 = src1->nb[1];                              \
    స్థిరరాశి uint32_t nb12 = src1->nb[2];                              \
    స్థిరరాశి uint32_t nb13 = src1->nb[3];                              \
                                                                    \
    స్థిరరాశి uint32_t nb0 = dst->nb[0];                                \
    స్థిరరాశి uint32_t nb1 = dst->nb[1];                                \
    స్థిరరాశి uint32_t nb2 = dst->nb[2];                                \
    స్థిరరాశి uint32_t nb3 = dst->nb[3];

#నిర్వచించు htp_matmul_preamble                                         \
    నిర్మాణం htp_mm_context * mmctx  = దత్తాంశం;                          \
    నిర్మాణం htp_ops_context * octx  = mmctx->octx;                   \
    dma_వరుస * dma_q              = octx->ctx->dma[ith];           \
    uint32_t src0_nrows_చొప్పున_thread = mmctx->src0_nrows_చొప్పున_thread;  \
    htp_matmul_tensors_preamble;




// hvx kernels first: the HMX Q6_K dequantizer reuses unpack_q6_k_group from there
#అన్తర్భావ "hvx-mm-kernels-tiled.h"
#అన్తర్భావ "hvx-mm-kernels-float.h"
#అన్తర్భావ "hmx-mm-kernels-tiled.h"

స్థిర శూన్య hvx_mm_2d_tiled(గుర్తులేని పూర్ణాంకం nth, గుర్తులేని పూర్ణాంకం ith, శూన్య * దత్తాంశం) {
    htp_matmul_preamble;

    స్థిరరాశి uint32_t src0_nrows = mmctx->src0_అడ్డువరుస_చివర - mmctx->src0_అడ్డువరుస_ప్రారంభం;
    స్థిరరాశి uint32_t act_nrows = mmctx->cur_m_అడ్డువరుసలు ? mmctx->cur_m_అడ్డువరుసలు : (ne11 * ne12 * ne13);
    స్థిరరాశి uint32_t cur_m_ప్రారంభం = mmctx->cur_m_ప్రారంభం;

    స్థిరరాశి uint32_t src0_ప్రారంభం_అడ్డువరుస  = mmctx->src0_అడ్డువరుస_ప్రారంభం + src0_nrows_చొప్పున_thread * ith;
    స్థిరరాశి uint32_t src0_చివర_అడ్డువరుస    = MIN(src0_ప్రారంభం_అడ్డువరుస + src0_nrows_చొప్పున_thread, mmctx->src0_అడ్డువరుస_చివర);

    నిర్మాణం htp_thread_trace * tr = &octx->ctx->trace[ith];

    స్థిరరాశి నిర్మాణం htp_mm_kernel_params * kparams = (స్థిరరాశి నిర్మాణం htp_mm_kernel_params *) octx->kernel_params;
    స్థిరరాశి uint32_t n_prefetch = kparams->n_prefetch;
    ప్రతిజ్ఞా(n_prefetch >= 2 && n_prefetch <= HTP_MM_MAX_PREFETCH && (n_prefetch & (n_prefetch - 1)) == 0);

    స్థిరరాశి size_t dst_అడ్డువరుస_కొలత  = nb1;
    స్థిరరాశి size_t act_అడ్డువరుస_కొలత = nb11;
    స్థిరరాశి size_t act_stride = mmctx->vtcm_act_stride;
    స్థిరరాశి size_t src2_stride = src2 ? ((src2->ne[1] == 1) ? 0 : src2->nb[1]) : 0;

    uint8_t * పరిమిత vtcm_dst_ptr  = mmctx->vtcm_dst  + mmctx->vtcm_dst_కొలత_చొప్పున_thread  * ith;
    uint8_t * పరిమిత vtcm_src0_ptr = mmctx->vtcm_src0 + mmctx->vtcm_src0_కొలత_చొప్పున_thread * ith;
    uint8_t * పరిమిత act_దత్తాంశం = mmctx->vtcm_act;

    స్థిరరాశి dma_addr_t src0_అడ్డువరుస = src0->దత్తాంశం;

    స్థిరరాశి uint32_t tile_కొలత = htp_tensor_get_tile_కొలత(src0->type);
    స్థిరరాశి uint32_t aligned_tile_కొలత = htp_tensor_get_tile_padded_కొలత(src0->type);

    uint32_t n_k_tiles_w = ne00 / htp_tensor_get_tile_k(src0->type);
    uint32_t tile_అడ్డువరుస_stride = n_k_tiles_w * tile_కొలత;
    uint32_t tile_అడ్డువరుస_transfer_కొలత_aligned = n_k_tiles_w * aligned_tile_కొలత;

    uint32_t ct_ప్రారంభం = src0_ప్రారంభం_అడ్డువరుస / 32;
    uint32_t ct_చివర   = (src0_చివర_అడ్డువరుస + 31) / 32;

    uint32_t నెట్టు_ct = ct_ప్రారంభం;
    ఒకవేళ (src0_ప్రారంభం_అడ్డువరుస < src0_చివర_అడ్డువరుస) {
        కొరకు (uint32_t d = 0; d < n_prefetch && నెట్టు_ct < ct_చివర; d++, నెట్టు_ct++) {
            dma_వరుస_నెట్టు(dma_q, dma_make_దత్తాంశం(vtcm_src0_ptr + d * tile_అడ్డువరుస_transfer_కొలత_aligned,
                           src0_అడ్డువరుస + నెట్టు_ct * tile_అడ్డువరుస_stride), aligned_tile_కొలత, tile_కొలత, tile_కొలత, n_k_tiles_w);
        }
    }

    ఒకవేళ (src0_ప్రారంభం_అడ్డువరుస >= src0_చివర_అడ్డువరుస) {
        తిరిగి;
    }

    కొరకు (uint32_t ct = ct_ప్రారంభం; ct < ct_చివర; ct++) {
        స్థిరరాశి uint8_t * w_tile = (శూన్య *) dma_వరుస_తీయి(dma_q).dst;

        పూర్ణాంకం valid_అడ్డువరుసలు = (పూర్ణాంకం)ne0 - (పూర్ణాంకం)(ct * 32);
        valid_అడ్డువరుసలు = MIN(32, MAX(0, valid_అడ్డువరుసలు));

        htp_trace_event_ప్రారంభం(tr, HTP_TRACE_EVT_HVX_COMP, ct);
        uint32_t ir1 = 0;
        కొరకు (; ir1 + 1 < act_nrows; ir1 += 2) {
            స్థిరరాశి uint8_t * పరిమిత act_col0 = (స్థిరరాశి uint8_t *) (act_దత్తాంశం + (ir1+0) * act_stride);
            స్థిరరాశి uint8_t * పరిమిత act_col1 = (స్థిరరాశి uint8_t *) (act_దత్తాంశం + (ir1+1) * act_stride);
            దశాంశం * పరిమిత dst_row0 = (దశాంశం *) (dst->దత్తాంశం + ((cur_m_ప్రారంభం + ir1+0) * dst_అడ్డువరుస_కొలత));
            దశాంశం * పరిమిత dst_row1 = (దశాంశం *) (dst->దత్తాంశం + ((cur_m_ప్రారంభం + ir1+1) * dst_అడ్డువరుస_కొలత));

            దశాంశం * dst_ptr0 = &dst_row0[ct * 32];
            దశాంశం * dst_ptr1 = &dst_row1[ct * 32];

            స్థిరరాశి దశాంశం * src2_ptr0 = ఖాళీ;
            స్థిరరాశి దశాంశం * src2_ptr1 = ఖాళీ;
            ఒకవేళ (src2) {
                స్థిరరాశి దశాంశం * పరిమిత src2_row0 = (స్థిరరాశి దశాంశం *) ((స్థిరరాశి uint8_t *) src2->దత్తాంశం + ((cur_m_ప్రారంభం + ir1+0) * src2_stride));
                స్థిరరాశి దశాంశం * పరిమిత src2_row1 = (స్థిరరాశి దశాంశం *) ((స్థిరరాశి uint8_t *) src2->దత్తాంశం + ((cur_m_ప్రారంభం + ir1+1) * src2_stride));
                src2_ptr0 = &src2_row0[ct * 32];
                src2_ptr1 = &src2_row1[ct * 32];
            }
            mmctx->vec_dot_32x2(ne10, dst_ptr0, dst_ptr1, w_tile, act_col0, act_col1, valid_అడ్డువరుసలు, src2_ptr0, src2_ptr1);
        }

        కొరకు (; ir1 < act_nrows; ++ir1) {
            స్థిరరాశి uint8_t * పరిమిత act_నిలువు = (స్థిరరాశి uint8_t *) (act_దత్తాంశం + ir1 * act_stride);
            దశాంశం * పరిమిత dst_అడ్డువరుస          = (దశాంశం *) (dst->దత్తాంశం + ((cur_m_ప్రారంభం + ir1) * dst_అడ్డువరుస_కొలత));
            దశాంశం * dst_ptr = &dst_అడ్డువరుస[ct * 32];

            స్థిరరాశి దశాంశం * src2_ptr = ఖాళీ;
            ఒకవేళ (src2) {
                స్థిరరాశి దశాంశం * పరిమిత src2_అడ్డువరుస = (స్థిరరాశి దశాంశం *) ((స్థిరరాశి uint8_t *) src2->దత్తాంశం + ((cur_m_ప్రారంభం + ir1) * src2_stride));
                src2_ptr = &src2_అడ్డువరుస[ct * 32];
            }
            mmctx->vec_dot_32x1(ne10, dst_ptr, w_tile, act_నిలువు, valid_అడ్డువరుసలు, src2_ptr);
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ct);

        ఒకవేళ (నెట్టు_ct < ct_చివర) {
            dma_వరుస_నెట్టు(dma_q, dma_make_దత్తాంశం(w_tile, src0_అడ్డువరుస + నెట్టు_ct * tile_అడ్డువరుస_stride),
                           aligned_tile_కొలత, tile_కొలత, tile_కొలత, n_k_tiles_w);
            నెట్టు_ct++;
        }
    }
}

స్థిర శూన్య hvx_mv_2d_tiled(గుర్తులేని పూర్ణాంకం nth, గుర్తులేని పూర్ణాంకం ith, శూన్య * దత్తాంశం) {
    htp_matmul_preamble;

    స్థిరరాశి uint32_t src0_nrows = mmctx->src0_అడ్డువరుస_చివర - mmctx->src0_అడ్డువరుస_ప్రారంభం;

    స్థిరరాశి uint32_t src0_ప్రారంభం_అడ్డువరుస  = mmctx->src0_అడ్డువరుస_ప్రారంభం + src0_nrows_చొప్పున_thread * ith;
    స్థిరరాశి uint32_t src0_చివర_అడ్డువరుస    = MIN(src0_ప్రారంభం_అడ్డువరుస + src0_nrows_చొప్పున_thread, mmctx->src0_అడ్డువరుస_చివర);

    నిర్మాణం htp_thread_trace * tr = &octx->ctx->trace[ith];

    స్థిరరాశి నిర్మాణం htp_mm_kernel_params * kparams = (స్థిరరాశి నిర్మాణం htp_mm_kernel_params *) octx->kernel_params;
    స్థిరరాశి uint32_t n_prefetch = kparams->n_prefetch;
    ప్రతిజ్ఞా(n_prefetch >= 2 && n_prefetch <= HTP_MM_MAX_PREFETCH && (n_prefetch & (n_prefetch - 1)) == 0);

    స్థిరరాశి size_t dst_అడ్డువరుస_కొలత  = nb1;
    స్థిరరాశి size_t act_అడ్డువరుస_కొలత = nb11;
    స్థిరరాశి size_t act_stride = mmctx->vtcm_act_stride;

    uint8_t * vtcm_dst_ptr  = mmctx->vtcm_dst + mmctx->vtcm_dst_కొలత_చొప్పున_thread * ith;
    uint8_t * vtcm_src0_ptr = mmctx->vtcm_src0 + mmctx->vtcm_src0_కొలత_చొప్పున_thread * ith;
    uint8_t * act_దత్తాంశం = mmctx->vtcm_act;

    దశాంశం * tmp = (దశాంశం *) vtcm_dst_ptr;

    స్థిరరాశి dma_addr_t src0_అడ్డువరుస = src0->దత్తాంశం;

    స్థిరరాశి uint8_t * పరిమిత act_నిలువు = (స్థిరరాశి uint8_t *) act_దత్తాంశం;
    దశాంశం * పరిమిత dst_నిలువు          = (దశాంశం *) dst->దత్తాంశం;

    స్థిరరాశి uint32_t tile_కొలత = htp_tensor_get_tile_కొలత(src0->type);
    స్థిరరాశి uint32_t aligned_tile_కొలత = htp_tensor_get_tile_padded_కొలత(src0->type);

    uint32_t n_k_tiles_w = ne00 / htp_tensor_get_tile_k(src0->type);
    uint32_t tile_అడ్డువరుస_stride = n_k_tiles_w * tile_కొలత;
    uint32_t tile_అడ్డువరుస_transfer_కొలత_aligned = n_k_tiles_w * aligned_tile_కొలత;

    uint32_t ct_ప్రారంభం = src0_ప్రారంభం_అడ్డువరుస / 32;
    uint32_t ct_చివర   = (src0_చివర_అడ్డువరుస + 31) / 32;

    uint32_t నెట్టు_ct = ct_ప్రారంభం;
    ఒకవేళ (src0_ప్రారంభం_అడ్డువరుస < src0_చివర_అడ్డువరుస) {
        ఒకవేళ (src2) {
            దశాంశం * vtcm_bias_ptr = (దశాంశం *) mmctx->vtcm_bias + src0_ప్రారంభం_అడ్డువరుస;
            స్థిరరాశి dma_addr_t src2_addr = src2->దత్తాంశం + src0_ప్రారంభం_అడ్డువరుస * పరిమాణం(దశాంశం);
            పూర్ణాంకం ముక్క_కొలత = (పూర్ణాంకం)MIN(src0_చివర_అడ్డువరుస, ne0) - (పూర్ణాంకం)src0_ప్రారంభం_అడ్డువరుస;
            ఒకవేళ (ముక్క_కొలత > 0) {
                dma_వరుస_నెట్టు(dma_q, dma_make_దత్తాంశం(vtcm_bias_ptr, src2_addr),
                               ముక్క_కొలత * పరిమాణం(దశాంశం), ముక్క_కొలత * పరిమాణం(దశాంశం), ముక్క_కొలత * పరిమాణం(దశాంశం), 1);
                dma_వరుస_తీయి_nowait(dma_q);
            }
        }
        కొరకు (uint32_t d = 0; d < n_prefetch && నెట్టు_ct < ct_చివర; d++, నెట్టు_ct++) {
            dma_వరుస_నెట్టు(dma_q, dma_make_దత్తాంశం(vtcm_src0_ptr + d * tile_అడ్డువరుస_transfer_కొలత_aligned,
                           src0_అడ్డువరుస + నెట్టు_ct * tile_అడ్డువరుస_stride), aligned_tile_కొలత, tile_కొలత, tile_కొలత, n_k_tiles_w);
        }
    }

    ఒకవేళ (src0_ప్రారంభం_అడ్డువరుస >= src0_చివర_అడ్డువరుస) {
        తిరిగి;
    }

    కొరకు (uint32_t ct = ct_ప్రారంభం; ct < ct_చివర; ct++) {
        స్థిరరాశి uint8_t * w_tile = (శూన్య *) dma_వరుస_తీయి(dma_q).dst;

        దశాంశం * dst_ptr = &tmp[ct * 32 - src0_ప్రారంభం_అడ్డువరుస];
        పూర్ణాంకం valid_అడ్డువరుసలు = (పూర్ణాంకం)ne0 - (పూర్ణాంకం)(ct * 32);
        valid_అడ్డువరుసలు = MIN(32, MAX(0, valid_అడ్డువరుసలు));

        htp_trace_event_ప్రారంభం(tr, HTP_TRACE_EVT_HVX_COMP, ct);
        mmctx->vec_dot_32x1(ne10, dst_ptr, w_tile, act_నిలువు, valid_అడ్డువరుసలు, ఖాళీ);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ct);

        ఒకవేళ (నెట్టు_ct < ct_చివర) {
            dma_వరుస_నెట్టు(dma_q, dma_make_దత్తాంశం(w_tile, src0_అడ్డువరుస + నెట్టు_ct * tile_అడ్డువరుస_stride),
                           aligned_tile_కొలత, tile_కొలత, tile_కొలత, n_k_tiles_w);
            నెట్టు_ct++;
        }
    }

    పూర్ణాంకం copy_cnt = (పూర్ణాంకం)MIN(src0_చివర_అడ్డువరుస, ne0) - (పూర్ణాంకం)src0_ప్రారంభం_అడ్డువరుస;
    ఒకవేళ (copy_cnt > 0) {
        htp_trace_event_ప్రారంభం(tr, HTP_TRACE_EVT_HVX_COMP, ct_చివర);
        ఒకవేళ (src2) {
            hvx_కూడు_f32_uuu((uint8_t *) &dst_నిలువు[src0_ప్రారంభం_అడ్డువరుస],
                            (స్థిరరాశి uint8_t *) tmp,
                            (స్థిరరాశి uint8_t *) ((స్థిరరాశి దశాంశం *) mmctx->vtcm_bias + src0_ప్రారంభం_అడ్డువరుస),
                            copy_cnt);
        } లేకపోతే {
            hvx_copy_f32_ua((uint8_t *) &dst_నిలువు[src0_ప్రారంభం_అడ్డువరుస], (uint8_t *) tmp, copy_cnt);
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ct_చివర);
    }
}

స్థిర శూన్య hvx_mm_nx_2d_tiled(గుర్తులేని పూర్ణాంకం nth, గుర్తులేని పూర్ణాంకం ith, శూన్య * దత్తాంశం) {
    నిర్మాణం htp_mm_context * mmctx = దత్తాంశం;
    నిర్మాణం htp_ops_context * octx = mmctx->octx;
    స్థిరరాశి నిర్మాణం htp_mm_kernel_params * kparams = (స్థిరరాశి నిర్మాణం htp_mm_kernel_params *) octx->kernel_params;
    స్థిరరాశి uint32_t n_weights = kparams->n_weights;

    స్థిరరాశి నిర్మాణం htp_tensor * పరిమిత act = octx->src[n_weights]; /* x */
    స్థిరరాశి uint32_t ne10 = act->ne[0];
    స్థిరరాశి uint32_t act_nrows = act->ne[1] * act->ne[2] * act->ne[3];
    స్థిరరాశి size_t act_stride = mmctx->vtcm_act_stride;

    uint8_t * పరిమిత vtcm_బరువు_ptr = mmctx->vtcm_src0 + mmctx->vtcm_src0_కొలత_చొప్పున_thread * ith;
    uint8_t * పరిమిత act_దత్తాంశం        = mmctx->vtcm_act;

    నిర్మాణం htp_thread_trace * tr = &octx->ctx->trace[ith];
    స్థిరరాశి uint32_t n_prefetch = kparams->n_prefetch;
    ప్రతిజ్ఞా(n_prefetch >= 2 && n_prefetch <= HTP_MM_MAX_PREFETCH && (n_prefetch & (n_prefetch - 1)) == 0);

    కొరకు (uint32_t widx = 0; widx < n_weights; widx++) {
        స్థిరరాశి నిర్మాణం htp_tensor * పరిమిత src_w = octx->src[widx];
        స్థిరరాశి నిర్మాణం htp_tensor * పరిమిత dst   = octx->dsts[widx];
        ఒకవేళ (!src_w || !dst) కొనసాగు;
        dma_వరుస * dma_q = octx->ctx->dma[ith];

        స్థిరరాశి uint32_t tile_కొలత = htp_tensor_get_tile_కొలత(src_w->type);
        స్థిరరాశి uint32_t aligned_tile_కొలత = htp_tensor_get_tile_padded_కొలత(src_w->type);
        స్థిరరాశి uint32_t ne00 = src_w->ne[0];
        స్థిరరాశి uint32_t ne01 = src_w->ne[1];
        స్థిరరాశి size_t dst_అడ్డువరుస_కొలత = dst->nb[1];
        స్థిరరాశి dma_addr_t src_w_అడ్డువరుస = src_w->దత్తాంశం;

        స్థిరరాశి uint32_t n_k_tiles_w = ne00 / htp_tensor_get_tile_k(src_w->type);
        స్థిరరాశి uint32_t tile_అడ్డువరుస_stride = n_k_tiles_w * tile_కొలత;
        స్థిరరాశి uint32_t tile_అడ్డువరుస_transfer_కొలత_aligned = n_k_tiles_w * aligned_tile_కొలత;

        uint32_t src0_ప్రారంభం_అడ్డువరుస = 0;
        uint32_t src0_చివర_అడ్డువరుస   = ne01;
        ఒకవేళ (octx->ctx->mdev.లెక్క > 1) {
            స్థిరరాశి bool can_విభజించు = htp_tensor_can_అడ్డువరుస_విభజన(dst, పరిమాణం(దశాంశం));
            స్థిరరాశి నిర్మాణం htp_tensor_mdev_range range = htp_tensor_mdev_విభజన(ne01, can_విభజించు ? 32 : 0,
                                                         octx->ctx->mdev.idx, octx->ctx->mdev.లెక్క, &octx->ctx->mdev.లెక్క_div);
            src0_ప్రారంభం_అడ్డువరుస = range.ప్రారంభం;
            src0_చివర_అడ్డువరుస   = range.ప్రారంభం + range.లెక్క;
        }

        స్థిరరాశి uint32_t nrows = src0_చివర_అడ్డువరుస - src0_ప్రారంభం_అడ్డువరుస;
        uint32_t src0_nrows_చొప్పున_thread = fastdiv(nrows + nth - 1, &octx->n_దారాలు_div);
        src0_nrows_చొప్పున_thread = hex_విడత_పైకి(src0_nrows_చొప్పున_thread, 32);

        స్థిరరాశి uint32_t ప్రారంభం_అడ్డువరుస = src0_ప్రారంభం_అడ్డువరుస + src0_nrows_చొప్పున_thread * ith;
        స్థిరరాశి uint32_t చివర_అడ్డువరుస   = MIN(ప్రారంభం_అడ్డువరుస + src0_nrows_చొప్పున_thread, src0_చివర_అడ్డువరుస);
        ఒకవేళ (ప్రారంభం_అడ్డువరుస >= చివర_అడ్డువరుస) కొనసాగు;

        uint32_t ct_ప్రారంభం = ప్రారంభం_అడ్డువరుస / 32;
        uint32_t ct_చివర   = (చివర_అడ్డువరుస + 31) / 32;

        uint32_t నెట్టు_ct = ct_ప్రారంభం;
        కొరకు (uint32_t d = 0; d < n_prefetch && నెట్టు_ct < ct_చివర; d++, నెట్టు_ct++) {
            dma_వరుస_నెట్టు(dma_q, dma_make_దత్తాంశం(vtcm_బరువు_ptr + d * tile_అడ్డువరుస_transfer_కొలత_aligned,
                            src_w_అడ్డువరుస + నెట్టు_ct * tile_అడ్డువరుస_stride), aligned_tile_కొలత, tile_కొలత, tile_కొలత, n_k_tiles_w);
        }

        కొరకు (uint32_t ct = ct_ప్రారంభం; ct < ct_చివర; ct++) {
            స్థిరరాశి uint8_t * w_tile = (శూన్య *) dma_వరుస_తీయి(dma_q).dst;
            పూర్ణాంకం valid_అడ్డువరుసలు = (పూర్ణాంకం)dst->ne[0] - (పూర్ణాంకం)(ct * 32);
            valid_అడ్డువరుసలు = MIN(32, MAX(0, valid_అడ్డువరుసలు));

            htp_trace_event_ప్రారంభం(tr, HTP_TRACE_EVT_HVX_COMP, ct);
            uint32_t ir1 = 0;
            కొరకు (; ir1 + 1 < act_nrows; ir1 += 2) {
                స్థిరరాశి uint8_t * పరిమిత act_col0 = (స్థిరరాశి uint8_t *) (act_దత్తాంశం + (ir1+0) * act_stride);
                స్థిరరాశి uint8_t * పరిమిత act_col1 = (స్థిరరాశి uint8_t *) (act_దత్తాంశం + (ir1+1) * act_stride);

                దశాంశం * పరిమిత dst_row0 = (దశాంశం *) (dst->దత్తాంశం + ((ir1+0) * dst_అడ్డువరుస_కొలత));
                దశాంశం * పరిమిత dst_row1 = (దశాంశం *) (dst->దత్తాంశం + ((ir1+1) * dst_అడ్డువరుస_కొలత));
                దశాంశం * dst_ptr0 = &dst_row0[ct * 32];
                దశాంశం * dst_ptr1 = &dst_row1[ct * 32];

                mmctx->vec_dot_32x2(ne10, dst_ptr0, dst_ptr1, w_tile, act_col0, act_col1, valid_అడ్డువరుసలు, ఖాళీ, ఖాళీ);
            }

            కొరకు (; ir1 < act_nrows; ++ir1) {
                స్థిరరాశి uint8_t * పరిమిత act_నిలువు = (స్థిరరాశి uint8_t *) (act_దత్తాంశం + ir1 * act_stride);
                దశాంశం * పరిమిత dst_అడ్డువరుస = (దశాంశం *) (dst->దత్తాంశం + (ir1 * dst_అడ్డువరుస_కొలత));
                దశాంశం * dst_ptr = &dst_అడ్డువరుస[ct * 32];
                mmctx->vec_dot_32x1(ne10, dst_ptr, w_tile, act_నిలువు, valid_అడ్డువరుసలు, ఖాళీ);
            }
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ct);

            ఒకవేళ (నెట్టు_ct < ct_చివర) {
                dma_వరుస_నెట్టు(dma_q, dma_make_దత్తాంశం(w_tile, src_w_అడ్డువరుస + నెట్టు_ct * tile_అడ్డువరుస_stride),
                                aligned_tile_కొలత, tile_కొలత, tile_కొలత, n_k_tiles_w);
                నెట్టు_ct++;
            }
        }
    }
}

స్థిర శూన్య hvx_mm_transfer_act_dma(
    నిర్మాణం htp_ops_context * octx,
    స్థిరరాశి నిర్మాణం htp_mm_kernel_params * kparams,
    స్థిరరాశి నిర్మాణం htp_tensor * act,
    uint8_t * dst_పునాది,
    size_t dst_అడ్డువరుస_కొలత,
    uint32_t m_ప్రారంభం,
    uint32_t m_అడ్డువరుసలు
) {
    ఒకవేళ (m_అడ్డువరుసలు == 0) {
        తిరిగి;
    }

    dma_వరుస * dma_q = octx->ctx->dma[0];
    స్థిరరాశి uint32_t ne0 = act->ne[0];
    స్థిరరాశి size_t elem_కొలత = (act->type == HTP_TYPE_F16) ? పరిమాణం(__fp16) : పరిమాణం(దశాంశం);
    స్థిరరాశి size_t అడ్డువరుస_బైట్లు = ne0 * elem_కొలత;
    స్థిరరాశి size_t act_nb1 = act->nb[1];
    స్థిరరాశి dma_addr_t act_పునాది = act->దత్తాంశం;

    స్థిరరాశి bool is_contiguous = (act->nb[2] == act->ne[1] * act_nb1) &&
                               (act->nb[3] == act->ne[2] * act->nb[2]);

    ఒకవేళ (is_contiguous) {
        స్థిరరాశి dma_addr_t src_addr = act_పునాది + m_ప్రారంభం * act_nb1;
        dma_వరుస_నెట్టు(dma_q, dma_make_దత్తాంశం(dst_పునాది, src_addr),
                       dst_అడ్డువరుస_కొలత, act_nb1, అడ్డువరుస_బైట్లు, m_అడ్డువరుసలు);
        dma_వరుస_తీయి(dma_q);
    } లేకపోతే {
        స్థిరరాశి uint32_t ne12_ne1 = act->ne[2] * act->ne[1];
        స్థిరరాశి bool use_fastdiv = kparams->div_ne12_ne1.mp != 0;
        కొరకు (uint32_t ir = 0; ir < m_అడ్డువరుసలు; ++ir) {
            స్థిరరాశి uint32_t ir1 = m_ప్రారంభం + ir;
            uint32_t i11, i12, i13;
            ఒకవేళ (use_fastdiv) {
                i13 = fastdiv(ir1, &kparams->div_ne12_ne1);
                స్థిరరాశి uint32_t rem = ir1 - i13 * ne12_ne1;
                i12 = fastdiv(rem, &kparams->div_ne1);
                i11 = rem - i12 * act->ne[1];
            } లేకపోతే {
                i13 = ne12_ne1 ? ir1 / ne12_ne1 : 0;
                స్థిరరాశి uint32_t rem = ir1 - i13 * ne12_ne1;
                i12 = act->ne[1] ? rem / act->ne[1] : 0;
                i11 = rem - i12 * act->ne[1];
            }
            స్థిరరాశి dma_addr_t అడ్డువరుస_src = act_పునాది + (i11 * act->nb[1] +
                                                   i12 * act->nb[2] +
                                                   i13 * act->nb[3]);
            uint8_t * అడ్డువరుస_dst = dst_పునాది + ir * dst_అడ్డువరుస_కొలత;
            dma_వరుస_నెట్టు(dma_q, dma_make_దత్తాంశం(అడ్డువరుస_dst, అడ్డువరుస_src),
                           dst_అడ్డువరుస_కొలత, act_nb1, అడ్డువరుస_బైట్లు, 1);
            dma_వరుస_తీయి(dma_q);
        }
    }

    ఒకవేళ (dst_అడ్డువరుస_కొలత > అడ్డువరుస_బైట్లు) {
        నిర్మాణం htp_thread_trace * tr = &octx->ctx->trace[0];
        htp_trace_event_ప్రారంభం(tr, HTP_TRACE_EVT_HVX_A_PREP, (uint16_t) m_ప్రారంభం);
        స్థిరరాశి uint32_t pad_elems = (dst_అడ్డువరుస_కొలత - అడ్డువరుస_బైట్లు) / elem_కొలత;
        ఒకవేళ (elem_కొలత == పరిమాణం(దశాంశం)) {
            కొరకు (uint32_t ir = 0; ir < m_అడ్డువరుసలు; ++ir) {
                hvx_splat_f32_u(dst_పునాది + ir * dst_అడ్డువరుస_కొలత + అడ్డువరుస_బైట్లు, 0.0f, pad_elems);
            }
        } లేకపోతే {
            కొరకు (uint32_t ir = 0; ir < m_అడ్డువరుసలు; ++ir) {
                hvx_splat_f16_u(dst_పునాది + ir * dst_అడ్డువరుస_కొలత + అడ్డువరుస_బైట్లు, (_Float16) 0.0f, pad_elems);
            }
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_A_PREP, (uint16_t) m_ప్రారంభం);
    }
}

#నిర్వచించు QUANTIZE_IMPL(name, log_name, kernel_fn, dst_అడ్డువరుస_కొలత_సమాసం)                                        \
స్థిర శూన్య name(గుర్తులేని పూర్ణాంకం nth, గుర్తులేని పూర్ణాంకం ith, శూన్య * దత్తాంశం) {                                        \
    (శూన్య) nth;                                                                                            \
    నిర్మాణం htp_mm_context * mmctx = దత్తాంశం;                                                                  \
    నిర్మాణం htp_ops_context * octx = mmctx->octx;                                                           \
    స్థిరరాశి నిర్మాణం htp_tensor * src = mmctx->act;                                                            \
    స్థిరరాశి uint32_t ne0 = src->ne[0];                                                                       \
    స్థిరరాశి uint32_t nrows = mmctx->cur_m_అడ్డువరుసలు ? mmctx->cur_m_అడ్డువరుసలు : mmctx->act_nrows;                       \
    స్థిరరాశి uint32_t nrows_చొప్పున_thread = mmctx->n_quant_అడ్డువరుసలు_చొప్పున_thread;                                      \
                                                                                                           \
    స్థిరరాశి uint32_t ir_మొదటి = nrows_చొప్పున_thread * ith;                                                      \
    ఒకవేళ (ir_మొదటి >= nrows) {                                                                               \
        తిరిగి;                                                                                            \
    }                                                                                                      \
                                                                                                           \
    నిర్మాణం htp_thread_trace * tr = &octx->ctx->trace[ith];                                                 \
    htp_trace_event_ప్రారంభం(tr, HTP_TRACE_EVT_HVX_A_QUANT, ir_మొదటి);                                        \
                                                                                                           \
    uint8_t * పరిమిత dst = mmctx->vtcm_act;                                                              \
    స్థిరరాశి uint32_t ir_చివరి = MIN(ir_మొదటి + nrows_చొప్పున_thread, nrows);                                      \
    స్థిరరాశి size_t raw_అడ్డువరుస_కొలత = mmctx->vtcm_act_raw_stride;                                                \
    స్థిరరాశి size_t dst_అడ్డువరుస_కొలత = (dst_అడ్డువరుస_కొలత_సమాసం);                                                       \
                                                                                                           \
    స్థిరరాశి uint8_t * పరిమిత src_దత్తాంశం = (స్థిరరాశి uint8_t *) mmctx->vtcm_act_raw + (raw_అడ్డువరుస_కొలత * ir_మొదటి); \
    uint8_t * పరిమిత dst_దత్తాంశం = (uint8_t *) dst + (dst_అడ్డువరుస_కొలత * ir_మొదటి);                             \
    kernel_fn(src_దత్తాంశం, dst_దత్తాంశం, ఖాళీ, ne0, ir_చివరి - ir_మొదటి, raw_అడ్డువరుస_కొలత, dst_అడ్డువరుస_కొలత);              \
                                                                                                           \
    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_A_QUANT, ir_మొదటి);                                         \
}

QUANTIZE_IMPL(quantize_f32_q8_0_tiled,     "quantize-f32-q8_0_tiled",     quantize_f32_q8_0_tiled_kernel,     htp_mm_q8_0_tiled_అడ్డువరుస_కొలత(ne0))
QUANTIZE_IMPL(quantize_f32_q8_1_tiled,     "quantize-f32-q8_1_tiled",     quantize_f32_q8_1_tiled_kernel,     htp_mm_q8_1_tiled_అడ్డువరుస_కొలత(ne0))
QUANTIZE_IMPL(quantize_f32_q8_1_s16_tiled, "quantize-f32-q8_1_s16_tiled", quantize_f32_q8_1_s16_tiled_kernel, htp_mm_q8_1_tiled_అడ్డువరుస_కొలత(ne0))
QUANTIZE_IMPL(quantize_f32_f16_tiled,      "quantize-f32-f16_tiled",      quantize_f32_f16_tiled_kernel,      htp_mm_f16_tiled_అడ్డువరుస_కొలత(ne0))
QUANTIZE_IMPL(quantize_f16_f16_tiled,      "quantize-f16-f16_tiled",      quantize_f16_f16_tiled_kernel,      htp_mm_f16_tiled_అడ్డువరుస_కొలత(ne0))
QUANTIZE_IMPL(quantize_f32_f32,            "quantize-f32-f32",            quantize_f32_f32_kernel,            mmctx->vtcm_act_stride)
QUANTIZE_IMPL(quantize_f32_f16,            "quantize-f32-f16",            quantize_f32_f16_kernel,            mmctx->vtcm_act_stride)
QUANTIZE_IMPL(quantize_f16_f16,            "quantize-f16-f16",            quantize_f16_f16_kernel,            mmctx->vtcm_act_stride)

స్థిర శూన్య quantize_f32_q8_0_tiled_ఖండం(గుర్తులేని పూర్ణాంకం nth, గుర్తులేని పూర్ణాంకం ith, శూన్య * దత్తాంశం) {
    (శూన్య) nth;
    నిర్మాణం htp_mm_context * mmctx = దత్తాంశం;
    ఒకవేళ (mmctx->quant_ib_మొదటి[ith] >= mmctx->quant_ib_చివరి[ith]) {
        తిరిగి;
    }
    నిర్మాణం htp_ops_context * octx = mmctx->octx;
    నిర్మాణం htp_thread_trace * tr = &octx->ctx->trace[ith];
    htp_trace_event_ప్రారంభం(tr, HTP_TRACE_EVT_HVX_A_QUANT, mmctx->quant_ib_మొదటి[ith]);

    స్థిరరాశి నిర్మాణం htp_tensor * src = mmctx->act;

    quantize_f32_q8_0_tiled_ఖండం_kernel(
        (స్థిరరాశి దశాంశం *) mmctx->vtcm_act_raw,
        mmctx->vtcm_act,
        ఖాళీ,
        src->ne[0],
        mmctx->quant_ib_మొదటి[ith],
        mmctx->quant_ib_చివరి[ith],
        mmctx->vtcm_act_raw_stride,
        htp_mm_q8_0_tiled_అడ్డువరుస_కొలత(src->ne[0]),
        mmctx->quant_r[ith],
        mmctx->quant_c[ith]
    );

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_A_QUANT, mmctx->quant_ib_మొదటి[ith]);
}

స్థిర శూన్య quantize_f32_q8_1_tiled_ఖండం(గుర్తులేని పూర్ణాంకం nth, గుర్తులేని పూర్ణాంకం ith, శూన్య * దత్తాంశం) {
    (శూన్య) nth;
    నిర్మాణం htp_mm_context * mmctx = దత్తాంశం;
    ఒకవేళ (mmctx->quant_ib_మొదటి[ith] >= mmctx->quant_ib_చివరి[ith]) {
        తిరిగి;
    }
    నిర్మాణం htp_ops_context * octx = mmctx->octx;
    నిర్మాణం htp_thread_trace * tr = &octx->ctx->trace[ith];
    htp_trace_event_ప్రారంభం(tr, HTP_TRACE_EVT_HVX_A_QUANT, mmctx->quant_ib_మొదటి[ith]);

    స్థిరరాశి నిర్మాణం htp_tensor * src = mmctx->act;

    quantize_f32_q8_1_tiled_ఖండం_kernel(
        (స్థిరరాశి దశాంశం *) mmctx->vtcm_act_raw,
        mmctx->vtcm_act,
        ఖాళీ,
        src->ne[0],
        mmctx->quant_ib_మొదటి[ith],
        mmctx->quant_ib_చివరి[ith],
        mmctx->vtcm_act_raw_stride,
        htp_mm_q8_1_tiled_అడ్డువరుస_కొలత(src->ne[0]),
        mmctx->quant_r[ith],
        mmctx->quant_c[ith]
    );

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_A_QUANT, mmctx->quant_ib_మొదటి[ith]);
}

స్థిర శూన్య quantize_f32_q8_1_s16_tiled_ఖండం(గుర్తులేని పూర్ణాంకం nth, గుర్తులేని పూర్ణాంకం ith, శూన్య * దత్తాంశం) {
    (శూన్య) nth;
    నిర్మాణం htp_mm_context * mmctx = దత్తాంశం;
    ఒకవేళ (mmctx->quant_ib_మొదటి[ith] >= mmctx->quant_ib_చివరి[ith]) {
        తిరిగి;
    }
    నిర్మాణం htp_ops_context * octx = mmctx->octx;
    నిర్మాణం htp_thread_trace * tr = &octx->ctx->trace[ith];
    htp_trace_event_ప్రారంభం(tr, HTP_TRACE_EVT_HVX_A_QUANT, mmctx->quant_ib_మొదటి[ith]);

    స్థిరరాశి నిర్మాణం htp_tensor * src = mmctx->act;

    quantize_f32_q8_1_s16_tiled_ఖండం_kernel(
        (స్థిరరాశి దశాంశం *) mmctx->vtcm_act_raw,
        mmctx->vtcm_act,
        ఖాళీ,
        src->ne[0],
        mmctx->quant_ib_మొదటి[ith],
        mmctx->quant_ib_చివరి[ith],
        mmctx->vtcm_act_raw_stride,
        htp_mm_q8_1_tiled_అడ్డువరుస_కొలత(src->ne[0]),
        mmctx->quant_r[ith],
        mmctx->quant_c[ith]
    );

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_A_QUANT, mmctx->quant_ib_మొదటి[ith]);
}

స్థిర శూన్య quantize_f32_f16_tiled_ఖండం(గుర్తులేని పూర్ణాంకం nth, గుర్తులేని పూర్ణాంకం ith, శూన్య * దత్తాంశం) {
    (శూన్య) nth;
    నిర్మాణం htp_mm_context * mmctx = దత్తాంశం;
    ఒకవేళ (mmctx->quant_ib_మొదటి[ith] >= mmctx->quant_ib_చివరి[ith]) {
        తిరిగి;
    }
    నిర్మాణం htp_ops_context * octx = mmctx->octx;
    నిర్మాణం htp_thread_trace * tr = &octx->ctx->trace[ith];
    htp_trace_event_ప్రారంభం(tr, HTP_TRACE_EVT_HVX_A_QUANT, mmctx->quant_ib_మొదటి[ith]);

    స్థిరరాశి నిర్మాణం htp_tensor * src = mmctx->act;

    quantize_f32_f16_tiled_ఖండం_kernel(
        (స్థిరరాశి దశాంశం *) mmctx->vtcm_act_raw,
        mmctx->vtcm_act,
        ఖాళీ,
        src->ne[0],
        mmctx->quant_ib_మొదటి[ith],
        mmctx->quant_ib_చివరి[ith],
        mmctx->vtcm_act_raw_stride,
        htp_mm_f16_tiled_అడ్డువరుస_కొలత(src->ne[0]),
        mmctx->quant_r[ith],
        mmctx->quant_c[ith]
    );

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_A_QUANT, mmctx->quant_ib_మొదటి[ith]);
}

స్థిర శూన్య quantize_f16_f16_tiled_ఖండం(గుర్తులేని పూర్ణాంకం nth, గుర్తులేని పూర్ణాంకం ith, శూన్య * దత్తాంశం) {
    (శూన్య) nth;
    నిర్మాణం htp_mm_context * mmctx = దత్తాంశం;
    ఒకవేళ (mmctx->quant_ib_మొదటి[ith] >= mmctx->quant_ib_చివరి[ith]) {
        తిరిగి;
    }
    నిర్మాణం htp_ops_context * octx = mmctx->octx;
    నిర్మాణం htp_thread_trace * tr = &octx->ctx->trace[ith];
    htp_trace_event_ప్రారంభం(tr, HTP_TRACE_EVT_HVX_A_QUANT, mmctx->quant_ib_మొదటి[ith]);

    స్థిరరాశి నిర్మాణం htp_tensor * src = mmctx->act;

    quantize_f16_f16_tiled_ఖండం_kernel(
        (స్థిరరాశి __fp16 *) mmctx->vtcm_act_raw,
        mmctx->vtcm_act,
        ఖాళీ,
        src->ne[0],
        mmctx->quant_ib_మొదటి[ith],
        mmctx->quant_ib_చివరి[ith],
        mmctx->vtcm_act_raw_stride,
        htp_mm_f16_tiled_అడ్డువరుస_కొలత(src->ne[0]),
        mmctx->quant_r[ith],
        mmctx->quant_c[ith]
    );

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_A_QUANT, mmctx->quant_ib_మొదటి[ith]);
}

// q8_1 for weight types with offsets (q8_1_s16 for Q2_K), f16_tiled for F16, otherwise q8_0
స్థిర అంతఃవరుస కార్మికుడు_callback_t htp_mm_act_quant_అడ్డువరుస_func(పూర్ణాంకం బరువు_type, పూర్ణాంకం act_type) {
    ఒకవేళ (బరువు_type == HTP_TYPE_F16) {
        తిరిగి (act_type == HTP_TYPE_F16) ? quantize_f16_f16_tiled : quantize_f32_f16_tiled;
    }
    ఒకవేళ (బరువు_type == HTP_TYPE_Q2_K) {
        తిరిగి quantize_f32_q8_1_s16_tiled;
    }
    తిరిగి htp_mm_బరువు_has_స్థానభ్రంశం(బరువు_type) ? quantize_f32_q8_1_tiled : quantize_f32_q8_0_tiled;
}

స్థిర అంతఃవరుస కార్మికుడు_callback_t htp_mm_act_quant_ఖండం_func(పూర్ణాంకం బరువు_type, పూర్ణాంకం act_type) {
    ఒకవేళ (బరువు_type == HTP_TYPE_F16) {
        తిరిగి (act_type == HTP_TYPE_F16) ? quantize_f16_f16_tiled_ఖండం : quantize_f32_f16_tiled_ఖండం;
    }
    ఒకవేళ (బరువు_type == HTP_TYPE_Q2_K) {
        తిరిగి quantize_f32_q8_1_s16_tiled_ఖండం;
    }
    తిరిగి htp_mm_బరువు_has_స్థానభ్రంశం(బరువు_type) ? quantize_f32_q8_1_tiled_ఖండం : quantize_f32_q8_0_tiled_ఖండం;
}

స్థిర శూన్య hvx_mm_4d_tiled(గుర్తులేని పూర్ణాంకం nth, గుర్తులేని పూర్ణాంకం ith, శూన్య * దత్తాంశం) {
    htp_matmul_preamble;

    స్థిరరాశి uint32_t src0_nrows = mmctx->src0_అడ్డువరుస_చివర - mmctx->src0_అడ్డువరుస_ప్రారంభం;
    స్థిరరాశి uint32_t cur_m_అడ్డువరుసలు = mmctx->cur_m_అడ్డువరుసలు ? mmctx->cur_m_అడ్డువరుసలు : (ne11 * ne12 * ne13);
    స్థిరరాశి uint32_t cur_m_ప్రారంభం = mmctx->cur_m_ప్రారంభం;

    స్థిరరాశి uint32_t src0_ప్రారంభం_అడ్డువరుస  = mmctx->src0_అడ్డువరుస_ప్రారంభం + src0_nrows_చొప్పున_thread * ith;
    స్థిరరాశి uint32_t src0_చివర_అడ్డువరుస    = MIN(src0_ప్రారంభం_అడ్డువరుస + src0_nrows_చొప్పున_thread, mmctx->src0_అడ్డువరుస_చివర);

    నిర్మాణం htp_thread_trace * tr = &octx->ctx->trace[ith];

    స్థిరరాశి నిర్మాణం htp_mm_kernel_params * kparams = (స్థిరరాశి నిర్మాణం htp_mm_kernel_params *) octx->kernel_params;
    స్థిరరాశి uint32_t n_prefetch = kparams->n_prefetch;
    ప్రతిజ్ఞా(n_prefetch >= 2 && n_prefetch <= HTP_MM_MAX_PREFETCH && (n_prefetch & (n_prefetch - 1)) == 0);

    స్థిరరాశి size_t dst_అడ్డువరుస_కొలత = nb1;
    స్థిరరాశి size_t act_stride = mmctx->vtcm_act_stride;

    uint8_t * పరిమిత vtcm_src0_ptr = mmctx->vtcm_src0 + mmctx->vtcm_src0_కొలత_చొప్పున_thread * ith;
    uint8_t * పరిమిత act_దత్తాంశం = mmctx->vtcm_act;

    స్థిరరాశి uint32_t tile_కొలత = htp_tensor_get_tile_కొలత(src0->type);
    స్థిరరాశి uint32_t aligned_tile_కొలత = htp_tensor_get_tile_padded_కొలత(src0->type);

    స్థిరరాశి uint32_t n_k_tiles_w = ne00 / htp_tensor_get_tile_k(src0->type);
    స్థిరరాశి uint32_t tile_అడ్డువరుస_stride = n_k_tiles_w * tile_కొలత;
    స్థిరరాశి uint32_t tile_అడ్డువరుస_transfer_కొలత_aligned = n_k_tiles_w * aligned_tile_కొలత;
    స్థిరరాశి uint32_t src0_ముక్క_stride = ((ne01 + 31) / 32) * tile_అడ్డువరుస_stride;

    స్థిరరాశి uint32_t ct_ప్రారంభం = src0_ప్రారంభం_అడ్డువరుస / 32;
    స్థిరరాశి uint32_t ct_చివర   = (src0_చివర_అడ్డువరుస + 31) / 32;

    ఒకవేళ (src0_ప్రారంభం_అడ్డువరుస >= src0_చివర_అడ్డువరుస || cur_m_అడ్డువరుసలు == 0) {
        తిరిగి;
    }

    స్థిరరాశి uint32_t మొత్తం_batches = ne12 * ne13;
    స్థిరరాశి uint32_t b_ప్రారంభం = fastdiv(cur_m_ప్రారంభం, &kparams->div_ne1);
    uint32_t b_చివర = fastdiv(cur_m_ప్రారంభం + cur_m_అడ్డువరుసలు + ne11 - 1, &kparams->div_ne1);
    b_చివర = MIN(b_చివర, మొత్తం_batches);

    uint32_t b_grp_ప్రారంభం = b_ప్రారంభం;
    అయితే (b_grp_ప్రారంభం < b_చివర) {
        స్థిరరాశి uint32_t b3 = fastdiv(b_grp_ప్రారంభం, &kparams->div_ne12);
        స్థిరరాశి uint32_t b2 = b_grp_ప్రారంభం - b3 * ne12;
        స్థిరరాశి uint32_t i02 = fastdiv(b2, &kparams->div_r2);
        స్థిరరాశి uint32_t i03 = fastdiv(b3, &kparams->div_r3);

        uint32_t b_grp_చివర = b_grp_ప్రారంభం + 1;
        అయితే (b_grp_చివర < b_చివర) {
            స్థిరరాశి uint32_t cur_b3 = fastdiv(b_grp_చివర, &kparams->div_ne12);
            స్థిరరాశి uint32_t cur_b2 = b_grp_చివర - cur_b3 * ne12;
            స్థిరరాశి uint32_t cur_i02 = fastdiv(cur_b2, &kparams->div_r2);
            స్థిరరాశి uint32_t cur_i03 = fastdiv(cur_b3, &kparams->div_r3);
            ఒకవేళ (cur_i02 != i02 || cur_i03 != i03) {
                ఆపు;
            }
            b_grp_చివర++;
        }

        స్థిరరాశి uint32_t ముక్క_idx = i03 * ne02 + i02;
        స్థిరరాశి dma_addr_t src0_ముక్క = src0->దత్తాంశం + (size_t) ముక్క_idx * src0_ముక్క_stride;

        uint32_t నెట్టు_ct = ct_ప్రారంభం;
        కొరకు (uint32_t d = 0; d < n_prefetch && నెట్టు_ct < ct_చివర; d++, నెట్టు_ct++) {
            dma_వరుస_నెట్టు(dma_q, dma_make_దత్తాంశం(vtcm_src0_ptr + d * tile_అడ్డువరుస_transfer_కొలత_aligned,
                           src0_ముక్క + (size_t) నెట్టు_ct * tile_అడ్డువరుస_stride),
                           aligned_tile_కొలత, tile_కొలత, tile_కొలత, n_k_tiles_w);
        }

        కొరకు (uint32_t ct = ct_ప్రారంభం; ct < ct_చివర; ct++) {
            స్థిరరాశి uint8_t * w_tile = (శూన్య *) dma_వరుస_తీయి(dma_q).dst;

            పూర్ణాంకం valid_అడ్డువరుసలు = (పూర్ణాంకం)ne0 - (పూర్ణాంకం)(ct * 32);
            valid_అడ్డువరుసలు = MIN(32, MAX(0, valid_అడ్డువరుసలు));

            htp_trace_event_ప్రారంభం(tr, HTP_TRACE_EVT_HVX_COMP, ct);
            కొరకు (uint32_t b = b_grp_ప్రారంభం; b < b_grp_చివర; b++) {
                స్థిరరాశి uint32_t b_m_ప్రారంభం = b * ne11;
                స్థిరరాశి uint32_t m_మొదటి = MAX(cur_m_ప్రారంభం, b_m_ప్రారంభం);
                స్థిరరాశి uint32_t m_చివరి  = MIN(cur_m_ప్రారంభం + cur_m_అడ్డువరుసలు, b_m_ప్రారంభం + ne11);
                ఒకవేళ (m_మొదటి >= m_చివరి) కొనసాగు;

                స్థిరరాశి uint32_t cur_b3 = fastdiv(b, &kparams->div_ne12);
                స్థిరరాశి uint32_t cur_b2 = b - cur_b3 * ne12;
                uint8_t * dst_batch_పునాది = (uint8_t *) dst->దత్తాంశం + (size_t) cur_b2 * nb2 + (size_t) cur_b3 * nb3;

                స్థిరరాశి uint32_t chunk_m_స్థానభ్రంశం = m_మొదటి - cur_m_ప్రారంభం;
                స్థిరరాశి uint32_t dst_m_స్థానభ్రంశం   = m_మొదటి - b_m_ప్రారంభం;
                స్థిరరాశి uint32_t batch_nrows    = m_చివరి - m_మొదటి;

                uint32_t ir1 = 0;
                కొరకు (; ir1 + 1 < batch_nrows; ir1 += 2) {
                    స్థిరరాశి uint8_t * పరిమిత act_col0 = (స్థిరరాశి uint8_t *) (act_దత్తాంశం + (chunk_m_స్థానభ్రంశం + ir1 + 0) * act_stride);
                    స్థిరరాశి uint8_t * పరిమిత act_col1 = (స్థిరరాశి uint8_t *) (act_దత్తాంశం + (chunk_m_స్థానభ్రంశం + ir1 + 1) * act_stride);
                    దశాంశం * పరిమిత dst_row0 = (దశాంశం *) (dst_batch_పునాది + (dst_m_స్థానభ్రంశం + ir1 + 0) * dst_అడ్డువరుస_కొలత);
                    దశాంశం * పరిమిత dst_row1 = (దశాంశం *) (dst_batch_పునాది + (dst_m_స్థానభ్రంశం + ir1 + 1) * dst_అడ్డువరుస_కొలత);
                    దశాంశం * dst_ptr0 = &dst_row0[ct * 32];
                    దశాంశం * dst_ptr1 = &dst_row1[ct * 32];
                    mmctx->vec_dot_32x2(ne10, dst_ptr0, dst_ptr1, w_tile, act_col0, act_col1, valid_అడ్డువరుసలు, ఖాళీ, ఖాళీ);
                }
                కొరకు (; ir1 < batch_nrows; ++ir1) {
                    స్థిరరాశి uint8_t * పరిమిత act_నిలువు = (స్థిరరాశి uint8_t *) (act_దత్తాంశం + (chunk_m_స్థానభ్రంశం + ir1) * act_stride);
                    దశాంశం * పరిమిత dst_అడ్డువరుస = (దశాంశం *) (dst_batch_పునాది + (dst_m_స్థానభ్రంశం + ir1) * dst_అడ్డువరుస_కొలత);
                    దశాంశం * dst_ptr = &dst_అడ్డువరుస[ct * 32];
                    mmctx->vec_dot_32x1(ne10, dst_ptr, w_tile, act_నిలువు, valid_అడ్డువరుసలు, ఖాళీ);
                }
            }
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ct);

            ఒకవేళ (నెట్టు_ct < ct_చివర) {
                dma_వరుస_నెట్టు(dma_q, dma_make_దత్తాంశం(w_tile, src0_ముక్క + (size_t) నెట్టు_ct * tile_అడ్డువరుస_stride),
                               aligned_tile_కొలత, tile_కొలత, tile_కొలత, n_k_tiles_w);
                నెట్టు_ct++;
            }
        }
        b_grp_ప్రారంభం = b_grp_చివర;
    }
    ఒకవేళ (src2) {
        hvx_tensor_కూడు_f32_grid(dst, src2, cur_m_ప్రారంభం, cur_m_ప్రారంభం + cur_m_అడ్డువరుసలు, src0_ప్రారంభం_అడ్డువరుస, src0_చివర_అడ్డువరుస, &kparams->div_ne12_ne1, &kparams->div_ne1);
    }
}

స్థిర శూన్య hvx_mm_2d_flat(గుర్తులేని పూర్ణాంకం nth, గుర్తులేని పూర్ణాంకం ith, శూన్య * దత్తాంశం) {
    htp_matmul_preamble;

    స్థిరరాశి నిర్మాణం htp_mm_kernel_params * kparams = (స్థిరరాశి నిర్మాణం htp_mm_kernel_params *) octx->kernel_params;
    స్థిరరాశి uint32_t n_prefetch = kparams->n_prefetch;
    ప్రతిజ్ఞా(n_prefetch >= 2 && n_prefetch <= HTP_MM_MAX_PREFETCH && (n_prefetch & (n_prefetch - 1)) == 0);
    స్థిరరాశి uint32_t prefetch_ముసుగు = n_prefetch - 1;

    స్థిరరాశి uint32_t src0_nrows = mmctx->src0_అడ్డువరుస_చివర - mmctx->src0_అడ్డువరుస_ప్రారంభం;  // src0 rows
    స్థిరరాశి uint32_t act_nrows = mmctx->cur_m_అడ్డువరుసలు ? mmctx->cur_m_అడ్డువరుసలు : mmctx->act_nrows;
    స్థిరరాశి uint32_t cur_m_ప్రారంభం = mmctx->cur_m_ప్రారంభం;

    స్థిరరాశి uint32_t src0_ప్రారంభం_అడ్డువరుస  = mmctx->src0_అడ్డువరుస_ప్రారంభం + src0_nrows_చొప్పున_thread * ith;
    స్థిరరాశి uint32_t src0_చివర_అడ్డువరుస    = MIN(src0_ప్రారంభం_అడ్డువరుస + src0_nrows_చొప్పున_thread, mmctx->src0_అడ్డువరుస_చివర);
    స్థిరరాశి uint32_t src0_చివర_అడ్డువరుస_x2 = src0_ప్రారంభం_అడ్డువరుస + ((src0_చివర_అడ్డువరుస - src0_ప్రారంభం_అడ్డువరుస) & ~1U);

    నిర్మాణం htp_thread_trace * tr = &octx->ctx->trace[ith];

    స్థిరరాశి size_t dst_అడ్డువరుస_కొలత  = nb1;
    స్థిరరాశి size_t src0_అడ్డువరుస_కొలత = nb01;
    స్థిరరాశి size_t act_అడ్డువరుస_కొలత = nb11;

    స్థిరరాశి size_t src0_stride = mmctx->vtcm_src0_stride;
    స్థిరరాశి size_t act_stride = mmctx->vtcm_act_stride;

    // Per-thread VTCMs for all tensors
    uint8_t * పరిమిత vtcm_dst_ptr  = mmctx->vtcm_dst  + mmctx->vtcm_dst_కొలత_చొప్పున_thread  * ith;
    uint8_t * పరిమిత vtcm_src0_ptr = mmctx->vtcm_src0 + mmctx->vtcm_src0_కొలత_చొప్పున_thread * ith;
    uint8_t * పరిమిత act_దత్తాంశం      = mmctx->vtcm_act;

    స్థిరరాశి dma_addr_t src0_అడ్డువరుస = src0->దత్తాంశం;


    // Prefill vtcm with src0 rows
    ఒకవేళ (src0_ప్రారంభం_అడ్డువరుస < src0_చివర_అడ్డువరుస) {
        కొరకు (uint32_t ir0 = src0_ప్రారంభం_అడ్డువరుస; ir0 < src0_చివర_అడ్డువరుస_x2; ir0 += 2) {
            స్థిరరాశి పూర్ణాంకం is0 = (ir0 - src0_ప్రారంభం_అడ్డువరుస);
            ఒకవేళ (is0 >= (పూర్ణాంకం)n_prefetch) {
                ఆపు;
            }
            dma_వరుస_నెట్టు(dma_q, dma_make_దత్తాంశం(vtcm_src0_ptr + is0 * src0_stride, src0_అడ్డువరుస + ir0 * src0_అడ్డువరుస_కొలత),
                           src0_stride, src0_అడ్డువరుస_కొలత, src0_అడ్డువరుస_కొలత, 2);
        }
    }

    ఒకవేళ (src0_ప్రారంభం_అడ్డువరుస >= src0_చివర_అడ్డువరుస) {
        తిరిగి;
    }

    // Process src0 rows
    కొరకు (uint32_t ir0 = src0_ప్రారంభం_అడ్డువరుస; ir0 < src0_చివర_అడ్డువరుస_x2; ir0 += 2) {
        స్థిరరాశి uint8_t * ss0 = (శూన్య *) dma_వరుస_తీయి(dma_q).dst;

        htp_trace_event_ప్రారంభం(tr, HTP_TRACE_EVT_HVX_COMP, ir0);
        // Process act columns in pairs (2x2 tiling)
        uint32_t ir1 = 0;
        కొరకు (; ir1 + 1 < act_nrows; ir1 += 2) {
            స్థిరరాశి uint8_t * పరిమిత act_col0 = (స్థిరరాశి uint8_t *) (act_దత్తాంశం + (ir1+0) * act_stride);
            స్థిరరాశి uint8_t * పరిమిత act_col1 = (స్థిరరాశి uint8_t *) (act_దత్తాంశం + (ir1+1) * act_stride);
            దశాంశం * పరిమిత dst_row0 = (దశాంశం *) (dst->దత్తాంశం + ((cur_m_ప్రారంభం + ir1+0) * dst_అడ్డువరుస_కొలత));
            దశాంశం * పరిమిత dst_row1 = (దశాంశం *) (dst->దత్తాంశం + ((cur_m_ప్రారంభం + ir1+1) * dst_అడ్డువరుస_కొలత));
            mmctx->vec_dot_2x2(ne00, &dst_row0[ir0], &dst_row1[ir0], ss0, ss0 + src0_stride, act_col0, act_col1);
        }

        // Handle remaining act rows (fallback to 2x1)
        కొరకు (; ir1 < act_nrows; ++ir1) {
            స్థిరరాశి uint8_t * పరిమిత act_నిలువు = (స్థిరరాశి uint8_t *) (act_దత్తాంశం + ir1 * act_stride);
            దశాంశం * పరిమిత dst_అడ్డువరుస          = (దశాంశం *) (dst->దత్తాంశం + ((cur_m_ప్రారంభం + ir1) * dst_అడ్డువరుస_కొలత));
            mmctx->vec_dot_2x1(ne00, &dst_అడ్డువరుస[ir0], ss0, ss0 + src0_stride, act_నిలువు);
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ir0);

        // Prefetch next (n + vtcm_nrows) row
        స్థిరరాశి పూర్ణాంకం pr0 = (ir0 + n_prefetch);
        స్థిరరాశి పూర్ణాంకం is0 = (pr0 - src0_ప్రారంభం_అడ్డువరుస) & prefetch_ముసుగు;
        ఒకవేళ (pr0 < src0_చివర_అడ్డువరుస_x2) {
            dma_వరుస_నెట్టు(dma_q, dma_make_దత్తాంశం(vtcm_src0_ptr + is0 * src0_stride, src0_అడ్డువరుస + pr0 * src0_అడ్డువరుస_కొలత),
                           src0_stride, src0_అడ్డువరుస_కొలత, src0_అడ్డువరుస_కొలత, 2);
        }
    }

    // Process the last row (if any)
    ఒకవేళ (src0_చివర_అడ్డువరుస != src0_చివర_అడ్డువరుస_x2) {
        uint32_t  ir0 = src0_చివర_అడ్డువరుస_x2;
        స్థిరరాశి పూర్ణాంకం is0 = (ir0 - src0_ప్రారంభం_అడ్డువరుస) & prefetch_ముసుగు;
        dma_వరుస_నెట్టు(dma_q, dma_make_దత్తాంశం(vtcm_src0_ptr + is0 * src0_stride, src0_అడ్డువరుస + ir0 * src0_అడ్డువరుస_కొలత),
                       src0_stride, src0_అడ్డువరుస_కొలత, src0_అడ్డువరుస_కొలత, 1);
        స్థిరరాశి uint8_t * ss0 = (శూన్య *) dma_వరుస_తీయి(dma_q).dst;

        htp_trace_event_ప్రారంభం(tr, HTP_TRACE_EVT_HVX_COMP, ir0);
        #సూచన unroll(2)
        కొరకు (uint32_t ir1 = 0; ir1 < act_nrows; ++ir1) {
            స్థిరరాశి uint8_t * పరిమిత act_నిలువు = (స్థిరరాశి uint8_t *) (act_దత్తాంశం + ir1 * act_stride);
            దశాంశం * పరిమిత dst_అడ్డువరుస          = (దశాంశం *) (dst->దత్తాంశం + ((cur_m_ప్రారంభం + ir1) * dst_అడ్డువరుస_కొలత));
            mmctx->vec_dot_1x1(ne00, &dst_అడ్డువరుస[ir0], ss0, act_నిలువు);
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ir0);
    }
    ఒకవేళ (src2) {
        hvx_tensor_కూడు_f32_grid(dst, src2, cur_m_ప్రారంభం, cur_m_ప్రారంభం + act_nrows, src0_ప్రారంభం_అడ్డువరుస, src0_చివర_అడ్డువరుస, &kparams->div_ne12_ne1, &kparams->div_ne1);
    }
}

స్థిర శూన్య hvx_mv_2d_flat(గుర్తులేని పూర్ణాంకం nth, గుర్తులేని పూర్ణాంకం ith, శూన్య * దత్తాంశం) {
    htp_matmul_preamble;

    స్థిరరాశి uint32_t src0_nrows = mmctx->src0_అడ్డువరుస_చివర - mmctx->src0_అడ్డువరుస_ప్రారంభం;

    స్థిరరాశి uint32_t src0_ప్రారంభం_అడ్డువరుస  = mmctx->src0_అడ్డువరుస_ప్రారంభం + src0_nrows_చొప్పున_thread * ith;
    స్థిరరాశి uint32_t src0_చివర_అడ్డువరుస    = MIN(src0_ప్రారంభం_అడ్డువరుస + src0_nrows_చొప్పున_thread, mmctx->src0_అడ్డువరుస_చివర);

    నిర్మాణం htp_thread_trace * tr = &octx->ctx->trace[ith];

    స్థిరరాశి size_t dst_అడ్డువరుస_కొలత  = nb1;
    స్థిరరాశి size_t src0_అడ్డువరుస_కొలత = nb01;
    స్థిరరాశి size_t act_అడ్డువరుస_కొలత  = nb11;

    స్థిరరాశి size_t src0_stride = mmctx->vtcm_src0_stride;
    స్థిరరాశి size_t act_stride  = mmctx->vtcm_act_stride;

    // Per-thread VTCMs for all tensors
    uint8_t * vtcm_dst_ptr  = mmctx->vtcm_dst  + mmctx->vtcm_dst_కొలత_చొప్పున_thread  * ith;
    uint8_t * vtcm_src0_ptr = mmctx->vtcm_src0 + mmctx->vtcm_src0_కొలత_చొప్పున_thread * ith;
    uint8_t * act_దత్తాంశం      = mmctx->vtcm_act;

    దశాంశం * tmp = (దశాంశం *) vtcm_dst_ptr;

    స్థిరరాశి dma_addr_t src0_అడ్డువరుస = src0->దత్తాంశం;
    స్థిరరాశి uint8_t * పరిమిత act_నిలువు = (స్థిరరాశి uint8_t *) act_దత్తాంశం;
    దశాంశం * పరిమిత dst_నిలువు         = (దశాంశం *) dst->దత్తాంశం;

    స్థిరరాశి uint32_t src0_చివర_అడ్డువరుస_x2 = src0_ప్రారంభం_అడ్డువరుస + ((src0_చివర_అడ్డువరుస - src0_ప్రారంభం_అడ్డువరుస) & ~1U);

    స్థిరరాశి నిర్మాణం htp_mm_kernel_params * kparams = (స్థిరరాశి నిర్మాణం htp_mm_kernel_params *) octx->kernel_params;
    స్థిరరాశి uint32_t n_prefetch = kparams->n_prefetch;
    ప్రతిజ్ఞా(n_prefetch >= 2 && n_prefetch <= HTP_MM_MAX_PREFETCH && (n_prefetch & (n_prefetch - 1)) == 0);
    స్థిరరాశి uint32_t prefetch_ముసుగు = n_prefetch - 1;

    // Prefill vtcm with 2x src0 rows
    ఒకవేళ (src0_ప్రారంభం_అడ్డువరుస < src0_చివర_అడ్డువరుస) {
        ఒకవేళ (src2) {
            దశాంశం * vtcm_bias_ptr = (దశాంశం *) mmctx->vtcm_bias + src0_ప్రారంభం_అడ్డువరుస;
            స్థిరరాశి dma_addr_t src2_addr = src2->దత్తాంశం + src0_ప్రారంభం_అడ్డువరుస * పరిమాణం(దశాంశం);
            పూర్ణాంకం ముక్క_కొలత = (పూర్ణాంకం)src0_చివర_అడ్డువరుస - (పూర్ణాంకం)src0_ప్రారంభం_అడ్డువరుస;
            ఒకవేళ (ముక్క_కొలత > 0) {
                dma_వరుస_నెట్టు(dma_q, dma_make_దత్తాంశం(vtcm_bias_ptr, src2_addr),
                               ముక్క_కొలత * పరిమాణం(దశాంశం), ముక్క_కొలత * పరిమాణం(దశాంశం), ముక్క_కొలత * పరిమాణం(దశాంశం), 1);
                dma_వరుస_తీయి_nowait(dma_q);
            }
        }
        కొరకు (uint32_t ir0 = src0_ప్రారంభం_అడ్డువరుస; ir0 < src0_చివర_అడ్డువరుస_x2; ir0 += 2) {
            స్థిరరాశి uint32_t is0 = (ir0 - src0_ప్రారంభం_అడ్డువరుస);
            ఒకవేళ (is0 >= n_prefetch) {
                ఆపు;
            }
            dma_వరుస_నెట్టు(dma_q, dma_make_దత్తాంశం(vtcm_src0_ptr + is0 * src0_stride, src0_అడ్డువరుస + ir0 * src0_అడ్డువరుస_కొలత),
                           src0_stride, src0_అడ్డువరుస_కొలత, src0_అడ్డువరుస_కొలత, 2);
        }
    }


    ఒకవేళ (src0_ప్రారంభం_అడ్డువరుస >= src0_చివర_అడ్డువరుస) {
        తిరిగి;
    }

    // Process src0 rows
    కొరకు (uint32_t ir0 = src0_ప్రారంభం_అడ్డువరుస; ir0 < src0_చివర_అడ్డువరుస_x2; ir0 += 2) {
        స్థిరరాశి uint8_t * ss0 = (శూన్య *) dma_వరుస_తీయి(dma_q).dst;
        htp_trace_event_ప్రారంభం(tr, HTP_TRACE_EVT_HVX_COMP, ir0);
        mmctx->vec_dot_2x1(ne00, &tmp[ir0 - src0_ప్రారంభం_అడ్డువరుస], ss0, ss0 + src0_stride, act_నిలువు);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ir0);

        // Prefetch next (n + vtcm_nrows) row
        స్థిరరాశి uint32_t pr0 = (ir0 + n_prefetch);
        స్థిరరాశి uint32_t is0 = (pr0 - src0_ప్రారంభం_అడ్డువరుస) & prefetch_ముసుగు;
        ఒకవేళ (pr0 < src0_చివర_అడ్డువరుస_x2) {
            dma_వరుస_నెట్టు(dma_q, dma_make_దత్తాంశం(vtcm_src0_ptr + is0 * src0_stride, src0_అడ్డువరుస + pr0 * src0_అడ్డువరుస_కొలత),
                           src0_stride, src0_అడ్డువరుస_కొలత, src0_అడ్డువరుస_కొలత, 2);
        }
    }

    // Process the last row (if any)
    ఒకవేళ (src0_చివర_అడ్డువరుస != src0_చివర_అడ్డువరుస_x2) {
        స్థిరరాశి uint32_t ir0 = src0_చివర_అడ్డువరుస_x2;
        స్థిరరాశి uint32_t is0 = (ir0 - src0_ప్రారంభం_అడ్డువరుస) & prefetch_ముసుగు;
        dma_వరుస_నెట్టు(dma_q, dma_make_దత్తాంశం(vtcm_src0_ptr + is0 * src0_stride, src0_అడ్డువరుస + ir0 * src0_అడ్డువరుస_కొలత),
                       src0_stride, src0_అడ్డువరుస_కొలత, src0_అడ్డువరుస_కొలత, 1);
        స్థిరరాశి uint8_t * ss0 = (శూన్య *) dma_వరుస_తీయి(dma_q).dst;
        htp_trace_event_ప్రారంభం(tr, HTP_TRACE_EVT_HVX_COMP, ir0);
        mmctx->vec_dot_1x1(ne00, &tmp[ir0 - src0_ప్రారంభం_అడ్డువరుస], ss0, act_నిలువు);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ir0);
    }

    పూర్ణాంకం copy_cnt = src0_చివర_అడ్డువరుస - src0_ప్రారంభం_అడ్డువరుస;
    ఒకవేళ (copy_cnt > 0) {
        htp_trace_event_ప్రారంభం(tr, HTP_TRACE_EVT_HVX_COMP, src0_చివర_అడ్డువరుస);
        ఒకవేళ (src2) {
            hvx_కూడు_f32_uuu((uint8_t *) &dst_నిలువు[src0_ప్రారంభం_అడ్డువరుస],
                            (స్థిరరాశి uint8_t *) tmp,
                            (స్థిరరాశి uint8_t *) ((స్థిరరాశి దశాంశం *) mmctx->vtcm_bias + src0_ప్రారంభం_అడ్డువరుస),
                            copy_cnt);
        } లేకపోతే {
            hvx_copy_f32_ua((uint8_t *) &dst_నిలువు[src0_ప్రారంభం_అడ్డువరుస], (uint8_t *) tmp, copy_cnt);
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, src0_చివర_అడ్డువరుస);
    }
}

స్థిర శూన్య hvx_mm_4d_flat(గుర్తులేని పూర్ణాంకం nth, గుర్తులేని పూర్ణాంకం ith, శూన్య * దత్తాంశం) {
    htp_matmul_preamble;

    స్థిరరాశి నిర్మాణం htp_mm_kernel_params * kparams = (స్థిరరాశి నిర్మాణం htp_mm_kernel_params *) octx->kernel_params;
    స్థిరరాశి uint32_t n_prefetch = kparams->n_prefetch;
    ప్రతిజ్ఞా(n_prefetch >= 2 && n_prefetch <= HTP_MM_MAX_PREFETCH && (n_prefetch & (n_prefetch - 1)) == 0);
    స్థిరరాశి uint32_t prefetch_ముసుగు = n_prefetch - 1;

    స్థిరరాశి uint32_t src0_nrows = mmctx->src0_అడ్డువరుస_చివర - mmctx->src0_అడ్డువరుస_ప్రారంభం;
    స్థిరరాశి uint32_t cur_m_అడ్డువరుసలు = mmctx->cur_m_అడ్డువరుసలు ? mmctx->cur_m_అడ్డువరుసలు : (ne11 * ne12 * ne13);
    స్థిరరాశి uint32_t cur_m_ప్రారంభం = mmctx->cur_m_ప్రారంభం;

    స్థిరరాశి uint32_t src0_ప్రారంభం_అడ్డువరుస  = mmctx->src0_అడ్డువరుస_ప్రారంభం + src0_nrows_చొప్పున_thread * ith;
    స్థిరరాశి uint32_t src0_చివర_అడ్డువరుస    = MIN(src0_ప్రారంభం_అడ్డువరుస + src0_nrows_చొప్పున_thread, mmctx->src0_అడ్డువరుస_చివర);
    స్థిరరాశి uint32_t src0_చివర_అడ్డువరుస_x2 = src0_ప్రారంభం_అడ్డువరుస + ((src0_చివర_అడ్డువరుస - src0_ప్రారంభం_అడ్డువరుస) & ~1U);

    నిర్మాణం htp_thread_trace * tr = &octx->ctx->trace[ith];

    స్థిరరాశి size_t dst_అడ్డువరుస_కొలత  = nb1;
    స్థిరరాశి size_t src0_అడ్డువరుస_కొలత = nb01;
    స్థిరరాశి size_t src0_stride = mmctx->vtcm_src0_stride;
    స్థిరరాశి size_t act_stride  = mmctx->vtcm_act_stride;

    uint8_t * పరిమిత vtcm_src0_ptr = mmctx->vtcm_src0 + mmctx->vtcm_src0_కొలత_చొప్పున_thread * ith;
    uint8_t * పరిమిత act_దత్తాంశం      = mmctx->vtcm_act;


    ఒకవేళ (src0_ప్రారంభం_అడ్డువరుస >= src0_చివర_అడ్డువరుస || cur_m_అడ్డువరుసలు == 0) {
        తిరిగి;
    }

    స్థిరరాశి uint32_t మొత్తం_batches = ne12 * ne13;
    స్థిరరాశి uint32_t b_ప్రారంభం = fastdiv(cur_m_ప్రారంభం, &kparams->div_ne1);
    uint32_t b_చివర = fastdiv(cur_m_ప్రారంభం + cur_m_అడ్డువరుసలు + ne11 - 1, &kparams->div_ne1);
    b_చివర = MIN(b_చివర, మొత్తం_batches);

    uint32_t b_grp_ప్రారంభం = b_ప్రారంభం;
    అయితే (b_grp_ప్రారంభం < b_చివర) {
        స్థిరరాశి uint32_t b3 = fastdiv(b_grp_ప్రారంభం, &kparams->div_ne12);
        స్థిరరాశి uint32_t b2 = b_grp_ప్రారంభం - b3 * ne12;
        స్థిరరాశి uint32_t i02 = fastdiv(b2, &kparams->div_r2);
        స్థిరరాశి uint32_t i03 = fastdiv(b3, &kparams->div_r3);

        uint32_t b_grp_చివర = b_grp_ప్రారంభం + 1;
        అయితే (b_grp_చివర < b_చివర) {
            స్థిరరాశి uint32_t cur_b3 = fastdiv(b_grp_చివర, &kparams->div_ne12);
            స్థిరరాశి uint32_t cur_b2 = b_grp_చివర - cur_b3 * ne12;
            స్థిరరాశి uint32_t cur_i02 = fastdiv(cur_b2, &kparams->div_r2);
            స్థిరరాశి uint32_t cur_i03 = fastdiv(cur_b3, &kparams->div_r3);
            ఒకవేళ (cur_i02 != i02 || cur_i03 != i03) {
                ఆపు;
            }
            b_grp_చివర++;
        }

        స్థిరరాశి dma_addr_t src0_అడ్డువరుస = src0->దత్తాంశం + ((size_t) i02 * nb02 + (size_t) i03 * nb03);

        కొరకు (uint32_t ir0 = src0_ప్రారంభం_అడ్డువరుస; ir0 < src0_చివర_అడ్డువరుస_x2; ir0 += 2) {
            స్థిరరాశి పూర్ణాంకం is0 = (ir0 - src0_ప్రారంభం_అడ్డువరుస);
            ఒకవేళ (is0 >= (పూర్ణాంకం)n_prefetch) {
                ఆపు;
            }
            dma_వరుస_నెట్టు(dma_q, dma_make_దత్తాంశం(vtcm_src0_ptr + is0 * src0_stride, src0_అడ్డువరుస + (size_t) ir0 * src0_అడ్డువరుస_కొలత),
                           src0_stride, src0_అడ్డువరుస_కొలత, src0_అడ్డువరుస_కొలత, 2);
        }

        కొరకు (uint32_t ir0 = src0_ప్రారంభం_అడ్డువరుస; ir0 < src0_చివర_అడ్డువరుస_x2; ir0 += 2) {
            స్థిరరాశి uint8_t * ss0 = (శూన్య *) dma_వరుస_తీయి(dma_q).dst;

            htp_trace_event_ప్రారంభం(tr, HTP_TRACE_EVT_HVX_COMP, ir0);
            కొరకు (uint32_t b = b_grp_ప్రారంభం; b < b_grp_చివర; b++) {
                స్థిరరాశి uint32_t b_m_ప్రారంభం = b * ne11;
                స్థిరరాశి uint32_t m_మొదటి = MAX(cur_m_ప్రారంభం, b_m_ప్రారంభం);
                స్థిరరాశి uint32_t m_చివరి  = MIN(cur_m_ప్రారంభం + cur_m_అడ్డువరుసలు, b_m_ప్రారంభం + ne11);
                ఒకవేళ (m_మొదటి >= m_చివరి) కొనసాగు;

                స్థిరరాశి uint32_t cur_b3 = fastdiv(b, &kparams->div_ne12);
                స్థిరరాశి uint32_t cur_b2 = b - cur_b3 * ne12;
                uint8_t * dst_batch_పునాది = (uint8_t *) dst->దత్తాంశం + (size_t) cur_b2 * nb2 + (size_t) cur_b3 * nb3;

                స్థిరరాశి uint32_t chunk_m_స్థానభ్రంశం = m_మొదటి - cur_m_ప్రారంభం;
                స్థిరరాశి uint32_t dst_m_స్థానభ్రంశం   = m_మొదటి - b_m_ప్రారంభం;
                స్థిరరాశి uint32_t batch_nrows    = m_చివరి - m_మొదటి;

                uint32_t ir1 = 0;
                కొరకు (; ir1 + 1 < batch_nrows; ir1 += 2) {
                    స్థిరరాశి uint8_t * పరిమిత act_col0 = (స్థిరరాశి uint8_t *) (act_దత్తాంశం + (chunk_m_స్థానభ్రంశం + ir1 + 0) * act_stride);
                    స్థిరరాశి uint8_t * పరిమిత act_col1 = (స్థిరరాశి uint8_t *) (act_దత్తాంశం + (chunk_m_స్థానభ్రంశం + ir1 + 1) * act_stride);
                    దశాంశం * పరిమిత dst_row0 = (దశాంశం *) (dst_batch_పునాది + (dst_m_స్థానభ్రంశం + ir1 + 0) * dst_అడ్డువరుస_కొలత);
                    దశాంశం * పరిమిత dst_row1 = (దశాంశం *) (dst_batch_పునాది + (dst_m_స్థానభ్రంశం + ir1 + 1) * dst_అడ్డువరుస_కొలత);
                    mmctx->vec_dot_2x2(ne00, &dst_row0[ir0], &dst_row1[ir0], ss0, ss0 + src0_stride, act_col0, act_col1);
                }
                కొరకు (; ir1 < batch_nrows; ++ir1) {
                    స్థిరరాశి uint8_t * పరిమిత act_నిలువు = (స్థిరరాశి uint8_t *) (act_దత్తాంశం + (chunk_m_స్థానభ్రంశం + ir1) * act_stride);
                    దశాంశం * పరిమిత dst_అడ్డువరుస          = (దశాంశం *) (dst_batch_పునాది + (dst_m_స్థానభ్రంశం + ir1) * dst_అడ్డువరుస_కొలత);
                    mmctx->vec_dot_2x1(ne00, &dst_అడ్డువరుస[ir0], ss0, ss0 + src0_stride, act_నిలువు);
                }
            }
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ir0);

            స్థిరరాశి పూర్ణాంకం pr0 = (ir0 + n_prefetch);
            స్థిరరాశి పూర్ణాంకం is0 = (pr0 - src0_ప్రారంభం_అడ్డువరుస) & prefetch_ముసుగు;
            ఒకవేళ (pr0 < src0_చివర_అడ్డువరుస_x2) {
                dma_వరుస_నెట్టు(dma_q, dma_make_దత్తాంశం(vtcm_src0_ptr + is0 * src0_stride, src0_అడ్డువరుస + (size_t) pr0 * src0_అడ్డువరుస_కొలత),
                               src0_stride, src0_అడ్డువరుస_కొలత, src0_అడ్డువరుస_కొలత, 2);
            }
        }

        ఒకవేళ (src0_చివర_అడ్డువరుస != src0_చివర_అడ్డువరుస_x2) {
            uint32_t ir0 = src0_చివర_అడ్డువరుస_x2;
            స్థిరరాశి పూర్ణాంకం is0 = (ir0 - src0_ప్రారంభం_అడ్డువరుస) & prefetch_ముసుగు;
            dma_వరుస_నెట్టు(dma_q, dma_make_దత్తాంశం(vtcm_src0_ptr + is0 * src0_stride, src0_అడ్డువరుస + (size_t) ir0 * src0_అడ్డువరుస_కొలత),
                           src0_stride, src0_అడ్డువరుస_కొలత, src0_అడ్డువరుస_కొలత, 1);
            స్థిరరాశి uint8_t * ss0 = (శూన్య *) dma_వరుస_తీయి(dma_q).dst;

            htp_trace_event_ప్రారంభం(tr, HTP_TRACE_EVT_HVX_COMP, ir0);
            కొరకు (uint32_t b = b_grp_ప్రారంభం; b < b_grp_చివర; b++) {
                స్థిరరాశి uint32_t b_m_ప్రారంభం = b * ne11;
                స్థిరరాశి uint32_t m_మొదటి = MAX(cur_m_ప్రారంభం, b_m_ప్రారంభం);
                స్థిరరాశి uint32_t m_చివరి  = MIN(cur_m_ప్రారంభం + cur_m_అడ్డువరుసలు, b_m_ప్రారంభం + ne11);
                ఒకవేళ (m_మొదటి >= m_చివరి) కొనసాగు;

                స్థిరరాశి uint32_t cur_b3 = fastdiv(b, &kparams->div_ne12);
                స్థిరరాశి uint32_t cur_b2 = b - cur_b3 * ne12;
                uint8_t * dst_batch_పునాది = (uint8_t *) dst->దత్తాంశం + (size_t) cur_b2 * nb2 + (size_t) cur_b3 * nb3;

                స్థిరరాశి uint32_t chunk_m_స్థానభ్రంశం = m_మొదటి - cur_m_ప్రారంభం;
                స్థిరరాశి uint32_t dst_m_స్థానభ్రంశం   = m_మొదటి - b_m_ప్రారంభం;
                స్థిరరాశి uint32_t batch_nrows    = m_చివరి - m_మొదటి;

                కొరకు (uint32_t ir1 = 0; ir1 < batch_nrows; ++ir1) {
                    స్థిరరాశి uint8_t * పరిమిత act_నిలువు = (స్థిరరాశి uint8_t *) (act_దత్తాంశం + (chunk_m_స్థానభ్రంశం + ir1) * act_stride);
                    దశాంశం * పరిమిత dst_అడ్డువరుస          = (దశాంశం *) (dst_batch_పునాది + (dst_m_స్థానభ్రంశం + ir1) * dst_అడ్డువరుస_కొలత);
                    mmctx->vec_dot_1x1(ne00, &dst_అడ్డువరుస[ir0], ss0, act_నిలువు);
                }
            }
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ir0);
        }

        b_grp_ప్రారంభం = b_grp_చివర;
    }

    ఒకవేళ (src2) {
        hvx_tensor_కూడు_f32_grid(dst, src2, cur_m_ప్రారంభం, cur_m_ప్రారంభం + cur_m_అడ్డువరుసలు, src0_ప్రారంభం_అడ్డువరుస, src0_చివర_అడ్డువరుస, &kparams->div_ne12_ne1, &kparams->div_ne1);
    }
}

#నిర్వచించు MMID_మాతృక_అడ్డువరుస(అడ్డువరుస_id, i1) మాతృక_అడ్డువరుసలు[(అడ్డువరుస_id) * mmctx->mapping_stride + (i1)]

స్థిర శూన్య hvx_mm_id_tiled(గుర్తులేని పూర్ణాంకం nth, గుర్తులేని పూర్ణాంకం ith, శూన్య * దత్తాంశం) {
    htp_matmul_preamble;

    స్థిరరాశి నిర్మాణం htp_tensor * పరిమిత ids = octx->src[2];

    స్థిరరాశి uint32_t src0_nrows      = mmctx->src0_అడ్డువరుస_చివర - mmctx->src0_అడ్డువరుస_ప్రారంభం;  // src0 rows per expert
    స్థిరరాశి uint32_t act_nrows       = ne11;
    స్థిరరాశి uint32_t src0_ప్రారంభం_అడ్డువరుస  = mmctx->src0_అడ్డువరుస_ప్రారంభం + src0_nrows_చొప్పున_thread * ith;
    స్థిరరాశి uint32_t src0_చివర_అడ్డువరుస    = MIN(src0_ప్రారంభం_అడ్డువరుస + src0_nrows_చొప్పున_thread, mmctx->src0_అడ్డువరుస_చివర);


    ఒకవేళ (src0_ప్రారంభం_అడ్డువరుస >= src0_చివర_అడ్డువరుస) {
        తిరిగి;
    }

    నిర్మాణం htp_thread_trace * tr = &octx->ctx->trace[ith];

    స్థిరరాశి నిర్మాణం htp_mm_kernel_params * kparams = (స్థిరరాశి నిర్మాణం htp_mm_kernel_params *) octx->kernel_params;
    స్థిరరాశి uint32_t n_prefetch = kparams->n_prefetch;
    ప్రతిజ్ఞా(n_prefetch >= 2 && n_prefetch <= HTP_MM_MAX_PREFETCH && (n_prefetch & (n_prefetch - 1)) == 0);

    స్థిరరాశి uint32_t n_ids = ids->ne[0];  // n_expert_used
    స్థిరరాశి uint32_t n_as  = ne02;        // n_expert

    స్థిరరాశి uint32_t *                మాతృక_అడ్డువరుస_counts = mmctx->మాతృక_అడ్డువరుస_counts;
    స్థిరరాశి నిర్మాణం mmid_అడ్డువరుస_mapping * మాతృక_అడ్డువరుసలు       = mmctx->మాతృక_అడ్డువరుసలు;

    స్థిరరాశి size_t dst_అడ్డువరుస_కొలత  = nb1;
    స్థిరరాశి size_t act_అడ్డువరుస_కొలత  = htp_mm_q8_0_tiled_అడ్డువరుస_కొలత(ne10);

    స్థిరరాశి size_t act_stride = mmctx->vtcm_act_stride;

    // Per-thread VTCMs for all tensors
    uint8_t * పరిమిత vtcm_src0_ptr = mmctx->vtcm_src0 + mmctx->vtcm_src0_కొలత_చొప్పున_thread * ith;
    uint8_t * పరిమిత act_దత్తాంశం = mmctx->vtcm_act;

    కొరకు (uint32_t cur_a = 0; cur_a < n_as; ++cur_a) {
        స్థిరరాశి int32_t cne1 = మాతృక_అడ్డువరుస_counts[cur_a];
        ఒకవేళ (cne1 == 0) {
            కొనసాగు;
        }

        స్థిరరాశి dma_addr_t src0_అడ్డువరుస = src0->దత్తాంశం + cur_a * nb02;

        స్థిరరాశి uint32_t tile_కొలత = htp_tensor_get_tile_కొలత(src0->type);
        స్థిరరాశి uint32_t aligned_tile_కొలత = htp_tensor_get_tile_padded_కొలత(src0->type);
        స్థిరరాశి uint32_t n_k_tiles_w = ne00 / htp_tensor_get_tile_k(src0->type);
        స్థిరరాశి uint32_t tile_అడ్డువరుస_stride = n_k_tiles_w * tile_కొలత;
        స్థిరరాశి uint32_t tile_అడ్డువరుస_transfer_కొలత_aligned = n_k_tiles_w * aligned_tile_కొలత;

        స్థిరరాశి uint32_t ct_ప్రారంభం = src0_ప్రారంభం_అడ్డువరుస / 32;
        స్థిరరాశి uint32_t ct_చివర   = (src0_చివర_అడ్డువరుస + 31) / 32;

        uint32_t నెట్టు_ct = ct_ప్రారంభం;
        కొరకు (uint32_t d = 0; d < n_prefetch && నెట్టు_ct < ct_చివర; d++, నెట్టు_ct++) {
            dma_వరుస_నెట్టు(dma_q, dma_make_దత్తాంశం(vtcm_src0_ptr + d * tile_అడ్డువరుస_transfer_కొలత_aligned, src0_అడ్డువరుస + నెట్టు_ct * tile_అడ్డువరుస_stride),
                           aligned_tile_కొలత, tile_కొలత, tile_కొలత, n_k_tiles_w);
        }

        కొరకు (uint32_t ct = ct_ప్రారంభం; ct < ct_చివర; ct++) {
            స్థిరరాశి uint8_t * w_tile = (శూన్య *) dma_వరుస_తీయి(dma_q).dst;

            పూర్ణాంకం valid_అడ్డువరుసలు = (పూర్ణాంకం)ne0 - (పూర్ణాంకం)(ct * 32);
            valid_అడ్డువరుసలు = MIN(32, MAX(0, valid_అడ్డువరుసలు));

            htp_trace_event_ప్రారంభం(tr, HTP_TRACE_EVT_HVX_COMP, ct);
            కొరకు (uint32_t cid = 0; cid < cne1; ++cid) {
                నిర్మాణం mmid_అడ్డువరుస_mapping అడ్డువరుస_mapping = MMID_మాతృక_అడ్డువరుస(cur_a, cid);
                స్థిరరాశి పూర్ణాంకం               rm1         = అడ్డువరుస_mapping.i1;  // expert idx
                స్థిరరాశి పూర్ణాంకం               rm2         = అడ్డువరుస_mapping.i2;  // token idx

                స్థిరరాశి uint32_t ir1 = fastmodulo(rm1, ne11, &mmctx->mm_div_ne11);        // act row idx
                స్థిరరాశి uint8_t * పరిమిత act_నిలువు = (స్థిరరాశి uint8_t *) (act_దత్తాంశం + (ir1 + rm2 * ne11 + 0) * act_stride);
                దశాంశం * పరిమిత dst_అడ్డువరుస = (దశాంశం *) (dst->దత్తాంశం + (rm1 * nb1 + rm2 * nb2 + 0));

                mmctx->vec_dot_32x1(ne10, &dst_అడ్డువరుస[ct * 32], w_tile, act_నిలువు, valid_అడ్డువరుసలు, ఖాళీ);
            }
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ct);

            ఒకవేళ (నెట్టు_ct < ct_చివర) {
                dma_వరుస_నెట్టు(dma_q, dma_make_దత్తాంశం(w_tile, src0_అడ్డువరుస + నెట్టు_ct * tile_అడ్డువరుస_stride),
                               aligned_tile_కొలత, tile_కొలత, tile_కొలత, n_k_tiles_w);
                నెట్టు_ct++;
            }
        }
    }
}

స్థిర శూన్య hvx_mv_id_tiled(గుర్తులేని పూర్ణాంకం nth, గుర్తులేని పూర్ణాంకం ith, శూన్య * దత్తాంశం) {
    htp_matmul_preamble;

    స్థిరరాశి నిర్మాణం htp_tensor * పరిమిత ids = octx->src[2];

    స్థిరరాశి uint32_t src0_nrows      = mmctx->src0_అడ్డువరుస_చివర - mmctx->src0_అడ్డువరుస_ప్రారంభం;  // src0 rows per expert
    స్థిరరాశి uint32_t src0_ప్రారంభం_అడ్డువరుస  = mmctx->src0_అడ్డువరుస_ప్రారంభం + src0_nrows_చొప్పున_thread * ith;
    స్థిరరాశి uint32_t src0_చివర_అడ్డువరుస    = MIN(src0_ప్రారంభం_అడ్డువరుస + src0_nrows_చొప్పున_thread, mmctx->src0_అడ్డువరుస_చివర);


    ఒకవేళ (src0_ప్రారంభం_అడ్డువరుస >= src0_చివర_అడ్డువరుస) {
        తిరిగి;
    }

    నిర్మాణం htp_thread_trace * tr = &octx->ctx->trace[ith];

    స్థిరరాశి నిర్మాణం htp_mm_kernel_params * kparams = (స్థిరరాశి నిర్మాణం htp_mm_kernel_params *) octx->kernel_params;
    స్థిరరాశి uint32_t n_prefetch = kparams->n_prefetch;
    ప్రతిజ్ఞా(n_prefetch >= 2 && n_prefetch <= HTP_MM_MAX_PREFETCH && (n_prefetch & (n_prefetch - 1)) == 0);

    ప్రతిజ్ఞా(ne13 % ne03 == 0);

    స్థిరరాశి size_t dst_అడ్డువరుస_కొలత  = nb1;
    స్థిరరాశి size_t act_అడ్డువరుస_కొలత = htp_mm_q8_0_tiled_అడ్డువరుస_కొలత(ne10);

    స్థిరరాశి uint32_t n_aids = src2->ne[0];  // num activated experts
    స్థిరరాశి uint32_t n_ids  = ne02;         // num experts

    // Per-thread VTCMs for all tensors
    uint8_t * పరిమిత vtcm_src0_ptr = mmctx->vtcm_src0 + mmctx->vtcm_src0_కొలత_చొప్పున_thread * ith;
    uint8_t * పరిమిత act_దత్తాంశం = mmctx->vtcm_act;

    కొరకు (uint32_t ie1 = 0; ie1 < n_aids; ++ie1) {  // for each expert
        స్థిరరాశి int32_t eid = *(స్థిరరాశి int32_t *) ((స్థిరరాశి uint8_t *) src2->దత్తాంశం + ie1 * src2->nb[0]);
        ఒకవేళ (eid < 0) {
            కొనసాగు;
        }
        ప్రతిజ్ఞా(eid < (int32_t) n_ids);

        స్థిరరాశి dma_addr_t src0_అడ్డువరుస = src0->దత్తాంశం + eid * nb02;
        స్థిరరాశి uint8_t * పరిమిత act_నిలువు = (స్థిరరాశి uint8_t *) act_దత్తాంశం;
        దశాంశం * పరిమిత dst_అడ్డువరుస          = (దశాంశం *) (dst->దత్తాంశం + ie1 * nb1);

        స్థిరరాశి uint32_t tile_కొలత = htp_tensor_get_tile_కొలత(src0->type);
        స్థిరరాశి uint32_t aligned_tile_కొలత = htp_tensor_get_tile_padded_కొలత(src0->type);
        స్థిరరాశి uint32_t n_k_tiles_w = ne00 / htp_tensor_get_tile_k(src0->type);
        స్థిరరాశి uint32_t tile_అడ్డువరుస_stride = n_k_tiles_w * tile_కొలత;
        స్థిరరాశి uint32_t tile_అడ్డువరుస_transfer_కొలత_aligned = n_k_tiles_w * aligned_tile_కొలత;

        స్థిరరాశి uint32_t ct_ప్రారంభం = src0_ప్రారంభం_అడ్డువరుస / 32;
        స్థిరరాశి uint32_t ct_చివర   = (src0_చివర_అడ్డువరుస + 31) / 32;

        uint32_t నెట్టు_ct = ct_ప్రారంభం;
        కొరకు (uint32_t d = 0; d < n_prefetch && నెట్టు_ct < ct_చివర; d++, నెట్టు_ct++) {
            dma_వరుస_నెట్టు(dma_q, dma_make_దత్తాంశం(vtcm_src0_ptr + d * tile_అడ్డువరుస_transfer_కొలత_aligned, src0_అడ్డువరుస + నెట్టు_ct * tile_అడ్డువరుస_stride),
                           aligned_tile_కొలత, tile_కొలత, tile_కొలత, n_k_tiles_w);
        }

        కొరకు (uint32_t ct = ct_ప్రారంభం; ct < ct_చివర; ct++) {
            స్థిరరాశి uint8_t * w_tile = (శూన్య *) dma_వరుస_తీయి(dma_q).dst;

            పూర్ణాంకం valid_అడ్డువరుసలు = (పూర్ణాంకం)ne0 - (పూర్ణాంకం)(ct * 32);
            valid_అడ్డువరుసలు = MIN(32, MAX(0, valid_అడ్డువరుసలు));

            htp_trace_event_ప్రారంభం(tr, HTP_TRACE_EVT_HVX_COMP, ct);
            mmctx->vec_dot_32x1(ne10, &dst_అడ్డువరుస[ct * 32], w_tile, act_నిలువు, valid_అడ్డువరుసలు, ఖాళీ);
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ct);

            ఒకవేళ (నెట్టు_ct < ct_చివర) {
                dma_వరుస_నెట్టు(dma_q, dma_make_దత్తాంశం(w_tile, src0_అడ్డువరుస + నెట్టు_ct * tile_అడ్డువరుస_stride),
                               aligned_tile_కొలత, tile_కొలత, tile_కొలత, n_k_tiles_w);
                నెట్టు_ct++;
            }
        }
    }
}

స్థిర శూన్య hvx_mv_id_nx_tiled(గుర్తులేని పూర్ణాంకం nth, గుర్తులేని పూర్ణాంకం ith, శూన్య * దత్తాంశం) {
    నిర్మాణం htp_mm_context * mmctx = (నిర్మాణం htp_mm_context *) దత్తాంశం;
    నిర్మాణం htp_ops_context * octx = mmctx->octx;
    స్థిరరాశి నిర్మాణం htp_mm_kernel_params * kparams = (స్థిరరాశి నిర్మాణం htp_mm_kernel_params *) octx->kernel_params;
    స్థిరరాశి uint32_t n_weights      = kparams->n_weights;
    స్థిరరాశి నిర్మాణం htp_tensor * పరిమిత src0 = octx->src[0];
    స్థిరరాశి నిర్మాణం htp_tensor * పరిమిత act  = octx->src[n_weights];
    స్థిరరాశి నిర్మాణం htp_tensor * పరిమిత ids  = octx->src[n_weights + 1];


    నిర్మాణం htp_thread_trace * tr = &octx->ctx->trace[ith];

    స్థిరరాశి uint32_t n_prefetch = kparams->n_prefetch;
    ప్రతిజ్ఞా(n_prefetch >= 2 && n_prefetch <= HTP_MM_MAX_PREFETCH && (n_prefetch & (n_prefetch - 1)) == 0);

    స్థిరరాశి uint32_t n_aids = ids->ne[0];
    స్థిరరాశి uint32_t n_ids  = src0->ne[2];

    uint8_t * పరిమిత vtcm_src0_ptr = mmctx->vtcm_src0 + mmctx->vtcm_src0_కొలత_చొప్పున_thread * ith;
    uint8_t * పరిమిత act_దత్తాంశం = mmctx->vtcm_act;

    కొరకు (uint32_t ie1 = 0; ie1 < n_aids; ++ie1) {
        స్థిరరాశి int32_t eid = *(స్థిరరాశి int32_t *) ((స్థిరరాశి uint8_t *) ids->దత్తాంశం + ie1 * ids->nb[0]);
        ఒకవేళ (eid < 0) కొనసాగు;
        ప్రతిజ్ఞా(eid < (int32_t) n_ids);

        కొరకు (uint32_t p = 0; p < n_weights; ++p) {
            స్థిరరాశి నిర్మాణం htp_tensor * పరిమిత src_w = octx->src[p];
            స్థిరరాశి నిర్మాణం htp_tensor * పరిమిత dst   = octx->dsts[p];
            ఒకవేళ (!src_w || !dst) కొనసాగు;
            dma_వరుస * dma_q = octx->ctx->dma[ith];

            స్థిరరాశి uint32_t ne01 = src_w->ne[1];
            uint32_t ప్రారంభం_అడ్డువరుస = 0;
            uint32_t చివర_అడ్డువరుస   = ne01;
            ఒకవేళ (octx->ctx->mdev.లెక్క > 1) {
                స్థిరరాశి bool can_విభజించు = htp_tensor_can_అడ్డువరుస_విభజన(dst, పరిమాణం(దశాంశం));
                స్థిరరాశి నిర్మాణం htp_tensor_mdev_range range = htp_tensor_mdev_విభజన(ne01, can_విభజించు ? 32 : 0, octx->ctx->mdev.idx, octx->ctx->mdev.లెక్క, &octx->ctx->mdev.లెక్క_div);
                ప్రారంభం_అడ్డువరుస = range.ప్రారంభం;
                చివర_అడ్డువరుస   = range.ప్రారంభం + range.లెక్క;
            }

            స్థిరరాశి uint32_t nrows = చివర_అడ్డువరుస - ప్రారంభం_అడ్డువరుస;
            uint32_t src0_nrows_చొప్పున_thread = fastdiv(nrows + nth - 1, &octx->n_దారాలు_div);
            src0_nrows_చొప్పున_thread = hex_విడత_పైకి(src0_nrows_చొప్పున_thread, 32);

            స్థిరరాశి uint32_t src0_ప్రారంభం_అడ్డువరుస = ప్రారంభం_అడ్డువరుస + src0_nrows_చొప్పున_thread * ith;
            స్థిరరాశి uint32_t src0_చివర_అడ్డువరుస   = MIN(src0_ప్రారంభం_అడ్డువరుస + src0_nrows_చొప్పున_thread, చివర_అడ్డువరుస);
            ఒకవేళ (src0_ప్రారంభం_అడ్డువరుస >= src0_చివర_అడ్డువరుస) కొనసాగు;

            స్థిరరాశి dma_addr_t src0_అడ్డువరుస = src_w->దత్తాంశం + eid * src_w->nb[2];
            స్థిరరాశి uint8_t * పరిమిత act_నిలువు = (స్థిరరాశి uint8_t *) act_దత్తాంశం;
            దశాంశం * పరిమిత dst_అడ్డువరుస = (దశాంశం *) (dst->దత్తాంశం + ie1 * dst->nb[1]);

            స్థిరరాశి uint32_t tile_కొలత = htp_tensor_get_tile_కొలత(src_w->type);
            స్థిరరాశి uint32_t aligned_tile_కొలత = htp_tensor_get_tile_padded_కొలత(src_w->type);
            స్థిరరాశి uint32_t n_k_tiles_w = src_w->ne[0] / htp_tensor_get_tile_k(src_w->type);
            స్థిరరాశి uint32_t tile_అడ్డువరుస_stride = n_k_tiles_w * tile_కొలత;
            స్థిరరాశి uint32_t tile_అడ్డువరుస_transfer_కొలత_aligned = n_k_tiles_w * aligned_tile_కొలత;

            స్థిరరాశి uint32_t ct_ప్రారంభం = src0_ప్రారంభం_అడ్డువరుస / 32;
            స్థిరరాశి uint32_t ct_చివర   = (src0_చివర_అడ్డువరుస + 31) / 32;

            uint32_t నెట్టు_ct = ct_ప్రారంభం;
            కొరకు (uint32_t d = 0; d < n_prefetch && నెట్టు_ct < ct_చివర; d++, నెట్టు_ct++) {
                dma_వరుస_నెట్టు(dma_q, dma_make_దత్తాంశం(vtcm_src0_ptr + d * tile_అడ్డువరుస_transfer_కొలత_aligned, src0_అడ్డువరుస + నెట్టు_ct * tile_అడ్డువరుస_stride),
                               aligned_tile_కొలత, tile_కొలత, tile_కొలత, n_k_tiles_w);
            }

            కొరకు (uint32_t ct = ct_ప్రారంభం; ct < ct_చివర; ct++) {
                స్థిరరాశి uint8_t * w_tile = (శూన్య *) dma_వరుస_తీయి(dma_q).dst;

                పూర్ణాంకం valid_అడ్డువరుసలు = (పూర్ణాంకం)dst->ne[0] - (పూర్ణాంకం)(ct * 32);
                valid_అడ్డువరుసలు = MIN(32, MAX(0, valid_అడ్డువరుసలు));

                htp_trace_event_ప్రారంభం(tr, HTP_TRACE_EVT_HVX_COMP, ct);
                mmctx->vec_dot_32x1(act->ne[0], &dst_అడ్డువరుస[ct * 32], w_tile, act_నిలువు, valid_అడ్డువరుసలు, ఖాళీ);
                htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ct);

                ఒకవేళ (నెట్టు_ct < ct_చివర) {
                    dma_వరుస_నెట్టు(dma_q, dma_make_దత్తాంశం(w_tile, src0_అడ్డువరుస + నెట్టు_ct * tile_అడ్డువరుస_stride),
                                    aligned_tile_కొలత, tile_కొలత, tile_కొలత, n_k_tiles_w);
                    నెట్టు_ct++;
                }
            }
        }
    }
}

స్థిర శూన్య hvx_mm_id_nx_tiled(గుర్తులేని పూర్ణాంకం nth, గుర్తులేని పూర్ణాంకం ith, శూన్య * దత్తాంశం) {
    నిర్మాణం htp_mm_context * mmctx = (నిర్మాణం htp_mm_context *) దత్తాంశం;
    నిర్మాణం htp_ops_context * octx = mmctx->octx;
    స్థిరరాశి నిర్మాణం htp_mm_kernel_params * kparams = (స్థిరరాశి నిర్మాణం htp_mm_kernel_params *) octx->kernel_params;
    స్థిరరాశి uint32_t n_weights      = kparams->n_weights;
    స్థిరరాశి నిర్మాణం htp_tensor * పరిమిత src0 = octx->src[0];
    స్థిరరాశి నిర్మాణం htp_tensor * పరిమిత act  = octx->src[n_weights];
    స్థిరరాశి నిర్మాణం htp_tensor * పరిమిత ids  = octx->src[n_weights + 1];


    నిర్మాణం htp_thread_trace * tr = &octx->ctx->trace[ith];

    స్థిరరాశి uint32_t n_prefetch = kparams->n_prefetch;
    ప్రతిజ్ఞా(n_prefetch >= 2 && n_prefetch <= HTP_MM_MAX_PREFETCH && (n_prefetch & (n_prefetch - 1)) == 0);

    స్థిరరాశి uint32_t n_as = src0->ne[2];

    స్థిరరాశి uint32_t * మాతృక_అడ్డువరుస_counts = mmctx->మాతృక_అడ్డువరుస_counts;
    స్థిరరాశి నిర్మాణం mmid_అడ్డువరుస_mapping * మాతృక_అడ్డువరుసలు = mmctx->మాతృక_అడ్డువరుసలు;

    స్థిరరాశి size_t act_stride = mmctx->vtcm_act_stride;

    uint8_t * పరిమిత vtcm_src0_ptr = mmctx->vtcm_src0 + mmctx->vtcm_src0_కొలత_చొప్పున_thread * ith;
    uint8_t * పరిమిత act_దత్తాంశం = mmctx->vtcm_act;

    కొరకు (uint32_t cur_a = 0; cur_a < n_as; ++cur_a) {
        స్థిరరాశి int32_t cne1 = మాతృక_అడ్డువరుస_counts[cur_a];
        ఒకవేళ (cne1 == 0) కొనసాగు;

        కొరకు (uint32_t p = 0; p < n_weights; ++p) {
            స్థిరరాశి నిర్మాణం htp_tensor * పరిమిత src_w = octx->src[p];
            స్థిరరాశి నిర్మాణం htp_tensor * పరిమిత dst   = octx->dsts[p];
            ఒకవేళ (!src_w || !dst) కొనసాగు;
            dma_వరుస * dma_q = octx->ctx->dma[ith];

            స్థిరరాశి uint32_t ne01 = src_w->ne[1];
            uint32_t ప్రారంభం_అడ్డువరుస = 0;
            uint32_t చివర_అడ్డువరుస   = ne01;
            ఒకవేళ (octx->ctx->mdev.లెక్క > 1) {
                స్థిరరాశి bool can_విభజించు = htp_tensor_can_అడ్డువరుస_విభజన(dst, పరిమాణం(దశాంశం));
                స్థిరరాశి నిర్మాణం htp_tensor_mdev_range range = htp_tensor_mdev_విభజన(ne01, can_విభజించు ? 32 : 0, octx->ctx->mdev.idx, octx->ctx->mdev.లెక్క, &octx->ctx->mdev.లెక్క_div);
                ప్రారంభం_అడ్డువరుస = range.ప్రారంభం;
                చివర_అడ్డువరుస   = range.ప్రారంభం + range.లెక్క;
            }

            స్థిరరాశి uint32_t nrows = చివర_అడ్డువరుస - ప్రారంభం_అడ్డువరుస;
            uint32_t src0_nrows_చొప్పున_thread = fastdiv(nrows + nth - 1, &octx->n_దారాలు_div);
            src0_nrows_చొప్పున_thread = hex_విడత_పైకి(src0_nrows_చొప్పున_thread, 32);

            స్థిరరాశి uint32_t src0_ప్రారంభం_అడ్డువరుస = ప్రారంభం_అడ్డువరుస + src0_nrows_చొప్పున_thread * ith;
            స్థిరరాశి uint32_t src0_చివర_అడ్డువరుస   = MIN(src0_ప్రారంభం_అడ్డువరుస + src0_nrows_చొప్పున_thread, చివర_అడ్డువరుస);
            ఒకవేళ (src0_ప్రారంభం_అడ్డువరుస >= src0_చివర_అడ్డువరుస) కొనసాగు;

            స్థిరరాశి dma_addr_t src0_అడ్డువరుస = src_w->దత్తాంశం + cur_a * src_w->nb[2];

            స్థిరరాశి uint32_t tile_కొలత = htp_tensor_get_tile_కొలత(src_w->type);
            స్థిరరాశి uint32_t aligned_tile_కొలత = htp_tensor_get_tile_padded_కొలత(src_w->type);
            స్థిరరాశి uint32_t n_k_tiles_w = src_w->ne[0] / htp_tensor_get_tile_k(src_w->type);
            స్థిరరాశి uint32_t tile_అడ్డువరుస_stride = n_k_tiles_w * tile_కొలత;
            స్థిరరాశి uint32_t tile_అడ్డువరుస_transfer_కొలత_aligned = n_k_tiles_w * aligned_tile_కొలత;

            స్థిరరాశి uint32_t ct_ప్రారంభం = src0_ప్రారంభం_అడ్డువరుస / 32;
            స్థిరరాశి uint32_t ct_చివర   = (src0_చివర_అడ్డువరుస + 31) / 32;

            uint32_t నెట్టు_ct = ct_ప్రారంభం;
            కొరకు (uint32_t d = 0; d < n_prefetch && నెట్టు_ct < ct_చివర; d++, నెట్టు_ct++) {
                dma_వరుస_నెట్టు(dma_q, dma_make_దత్తాంశం(vtcm_src0_ptr + d * tile_అడ్డువరుస_transfer_కొలత_aligned, src0_అడ్డువరుస + నెట్టు_ct * tile_అడ్డువరుస_stride),
                               aligned_tile_కొలత, tile_కొలత, tile_కొలత, n_k_tiles_w);
            }

            కొరకు (uint32_t ct = ct_ప్రారంభం; ct < ct_చివర; ct++) {
                స్థిరరాశి uint8_t * w_tile = (శూన్య *) dma_వరుస_తీయి(dma_q).dst;

                పూర్ణాంకం valid_అడ్డువరుసలు = (పూర్ణాంకం)dst->ne[0] - (పూర్ణాంకం)(ct * 32);
                valid_అడ్డువరుసలు = MIN(32, MAX(0, valid_అడ్డువరుసలు));

                htp_trace_event_ప్రారంభం(tr, HTP_TRACE_EVT_HVX_COMP, ct);
                కొరకు (uint32_t cid = 0; cid < (uint32_t) cne1; ++cid) {
                    నిర్మాణం mmid_అడ్డువరుస_mapping అడ్డువరుస_mapping = MMID_మాతృక_అడ్డువరుస(cur_a, cid);
                    స్థిరరాశి పూర్ణాంకం rm1 = అడ్డువరుస_mapping.i1;
                    స్థిరరాశి పూర్ణాంకం rm2 = అడ్డువరుస_mapping.i2;

                    స్థిరరాశి uint32_t ir1 = fastmodulo(rm1, act->ne[1], &mmctx->mm_div_ne11);
                    స్థిరరాశి uint8_t * పరిమిత act_నిలువు = (స్థిరరాశి uint8_t *) (act_దత్తాంశం + (ir1 + rm2 * act->ne[1]) * act_stride);
                    దశాంశం * పరిమిత dst_అడ్డువరుస = (దశాంశం *) (dst->దత్తాంశం + (rm1 * dst->nb[1] + rm2 * dst->nb[2]));

                    mmctx->vec_dot_32x1(act->ne[0], &dst_అడ్డువరుస[ct * 32], w_tile, act_నిలువు, valid_అడ్డువరుసలు, ఖాళీ);
                }
                htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ct);

                ఒకవేళ (నెట్టు_ct < ct_చివర) {
                    dma_వరుస_నెట్టు(dma_q, dma_make_దత్తాంశం(w_tile, src0_అడ్డువరుస + నెట్టు_ct * tile_అడ్డువరుస_stride),
                                    aligned_tile_కొలత, tile_కొలత, tile_కొలత, n_k_tiles_w);
                    నెట్టు_ct++;
                }
            }
        }
    }
}

స్థిర పూర్ణాంకం hvx_mm_init_tiled_vec_dot(నిర్మాణం htp_mm_context * mmctx, గణన htp_దత్తాంశం_type type) {
    ఎంపిక (type) {
        స్థితి HTP_TYPE_Q4_0:
            mmctx->type         = "q4_0_tiled-f32";
            mmctx->vec_dot_32x1 = tiled_vec_dot_q4_0_32x1;
            mmctx->vec_dot_32x2 = tiled_vec_dot_q4_0_32x2;
            తిరిగి 0;
        స్థితి HTP_TYPE_Q4_1:
        స్థితి HTP_TYPE_Q4_K:
            mmctx->type         = "q4_1_tiled-f32";
            mmctx->vec_dot_32x1 = tiled_vec_dot_q4_1_32x1;
            mmctx->vec_dot_32x2 = tiled_vec_dot_q4_1_32x2;
            తిరిగి 0;
        స్థితి HTP_TYPE_Q8_0:
            mmctx->type         = "q8_0_tiled-f32";
            mmctx->vec_dot_32x1 = tiled_vec_dot_q8_0_32x1;
            mmctx->vec_dot_32x2 = tiled_vec_dot_q8_0_32x2;
            తిరిగి 0;
        స్థితి HTP_TYPE_Q5_K:
            mmctx->type         = "q5_k_tiled-f32";
            mmctx->vec_dot_32x1 = tiled_vec_dot_q5_k_32x1;
            mmctx->vec_dot_32x2 = tiled_vec_dot_q5_k_32x2;
            తిరిగి 0;
        స్థితి HTP_TYPE_Q6_K:
            mmctx->type         = "q6_k_tiled-f32";
            mmctx->vec_dot_32x1 = tiled_vec_dot_q6_k_32x1;
            mmctx->vec_dot_32x2 = tiled_vec_dot_q6_k_32x2;
            తిరిగి 0;
        స్థితి HTP_TYPE_Q3_K:
            mmctx->type         = "q3_k_tiled-f32";
            mmctx->vec_dot_32x1 = tiled_vec_dot_q3_k_32x1;
            mmctx->vec_dot_32x2 = tiled_vec_dot_q3_k_32x2;
            తిరిగి 0;
        స్థితి HTP_TYPE_Q2_K:
            mmctx->type         = "q2_k_tiled-f32";
            mmctx->vec_dot_32x1 = tiled_vec_dot_q2_k_32x1;
            mmctx->vec_dot_32x2 = tiled_vec_dot_q2_k_32x2;
            తిరిగి 0;
        స్థితి HTP_TYPE_Q2_0:
            mmctx->type         = "q2_0_tiled-f32";
            mmctx->vec_dot_32x1 = tiled_vec_dot_q2_0_32x1;
            mmctx->vec_dot_32x2 = tiled_vec_dot_q2_0_32x2;
            తిరిగి 0;
        స్థితి HTP_TYPE_IQ4_NL:
            mmctx->type         = "iq4nl_tiled-f32";
            mmctx->vec_dot_32x1 = tiled_vec_dot_iq4nl_32x1;
            mmctx->vec_dot_32x2 = tiled_vec_dot_iq4nl_32x2;
            తిరిగి 0;
        స్థితి HTP_TYPE_MXFP4:
            mmctx->type         = "mxfp4_tiled-f32";
            mmctx->vec_dot_32x1 = tiled_vec_dot_mxfp4_32x1;
            mmctx->vec_dot_32x2 = tiled_vec_dot_mxfp4_32x2;
            తిరిగి 0;
        స్థితి HTP_TYPE_Q1_0:
            mmctx->type         = "q1_0_tiled-f32";
            mmctx->vec_dot_32x1 = tiled_vec_dot_q1_0_32x1;
            mmctx->vec_dot_32x2 = tiled_vec_dot_q1_0_32x2;
            తిరిగి 0;
        స్థితి HTP_TYPE_F16:
            mmctx->type         = "f16_tiled-f16";
            mmctx->vec_dot_32x1 = tiled_vec_dot_f16_32x1;
            mmctx->vec_dot_32x2 = tiled_vec_dot_f16_32x2;
            తిరిగి 0;
        అప్రమేయ:
            తిరిగి -1;
    }
}

స్థిర పూర్ణాంకం hvx_mm_matmul(నిర్మాణం htp_ops_context * octx) {
    htp_matmul_tensors_preamble;

    నిర్మాణం htp_thread_trace * tr = &octx->ctx->trace[0];
    htp_trace_event_ప్రారంభం(tr, HTP_TRACE_EVT_INIT, 0);

    నిర్మాణం htp_mm_context mmctx_struct = {0};
    నిర్మాణం htp_mm_context * mmctx = &mmctx_struct;
    mmctx->octx = octx;
    mmctx->act = act;

    స్థిరరాశి నిర్మాణం htp_mm_kernel_params * kparams = (స్థిరరాశి నిర్మాణం htp_mm_kernel_params *) octx->kernel_params;

    స్థిరరాశి uint32_t src0_nrows = ne01;
    స్థిరరాశి uint32_t act_nrows  = ne11 * ne12 * ne13;
    mmctx->act_nrows = act_nrows;

    uint32_t src0_అడ్డువరుస_ప్రారంభం = 0;
    uint32_t src0_అడ్డువరుస_చివర   = src0_nrows;

    ఒకవేళ (octx->ctx->mdev.లెక్క > 1) {
        స్థిరరాశి bool can_విభజించు = htp_tensor_can_అడ్డువరుస_విభజన(dst, పరిమాణం(దశాంశం));
        స్థిరరాశి నిర్మాణం htp_tensor_mdev_range range = htp_tensor_mdev_విభజన(src0_nrows, can_విభజించు ? 32 : 0, octx->ctx->mdev.idx, octx->ctx->mdev.లెక్క, &octx->ctx->mdev.లెక్క_div);
        src0_అడ్డువరుస_ప్రారంభం = range.ప్రారంభం;
        src0_అడ్డువరుస_చివర   = range.ప్రారంభం + range.లెక్క;
    }

    ఒకవేళ (src0_అడ్డువరుస_ప్రారంభం >= src0_అడ్డువరుస_చివర) {
        తిరిగి HTP_స్థితి_OK;
    }

    స్థిరరాశి uint32_t nrows = src0_అడ్డువరుస_చివర - src0_అడ్డువరుస_ప్రారంభం;
    mmctx->src0_అడ్డువరుస_ప్రారంభం = src0_అడ్డువరుస_ప్రారంభం;
    mmctx->src0_అడ్డువరుస_చివర   = src0_అడ్డువరుస_చివర;

    స్థిరరాశి bool is_tiled = htp_tensor_is_tiled(src0);

    // Compute src0_nrows_per_thread
    mmctx->src0_nrows_చొప్పున_thread  = fastdiv(nrows + octx->n_దారాలు - 1, &octx->n_దారాలు_div);
    ఒకవేళ (is_tiled) {
        mmctx->src0_nrows_చొప్పున_thread = hex_విడత_పైకి(mmctx->src0_nrows_చొప్పున_thread, 32);
    } లేకపోతే {
        mmctx->src0_nrows_చొప్పున_thread += (mmctx->src0_nrows_చొప్పున_thread & 1); // round up to even
    }

    స్థిరరాశి size_t src0_అడ్డువరుస_కొలత = nb01;
    స్థిరరాశి size_t dst_అడ్డువరుస_కొలత  = nb1;
    size_t       act_అడ్డువరుస_కొలత  = nb11;

    స్థిరరాశి size_t src0_అడ్డువరుస_కొలత_padded = hex_విడత_పైకి(src0_అడ్డువరుస_కొలత, 128);

    కార్మికుడు_callback_t quant_task_func;
    కార్మికుడు_callback_t matmul_పని_func;
    uint32_t n_quant_tasks = 1;
    స్థిరరాశి bool is_batched = (ne12 > 1 || ne13 > 1 || ne02 > 1 || ne03 > 1);
    ఒకవేళ (is_batched) {
        matmul_పని_func = is_tiled ? hvx_mm_4d_tiled : hvx_mm_4d_flat;
    } లేకపోతే ఒకవేళ (act_nrows > 1) {
        matmul_పని_func = is_tiled ? hvx_mm_2d_tiled : hvx_mm_2d_flat;
    } లేకపోతే {
        matmul_పని_func = is_tiled ? hvx_mv_2d_tiled : hvx_mv_2d_flat;
    }

    bool need_quant = true;

    ఎంపిక (kparams->kernel_type) {
        స్థితి HTP_MM_KERNEL_HVX_F16_F16_VTCM:
            quant_task_func        = (act->type == HTP_TYPE_F32) ? quantize_f32_f16 : quantize_f16_f16;
            need_quant             = (act->type == HTP_TYPE_F32);
            mmctx->type            = (act->type == HTP_TYPE_F32) ? "f32-f16" : "f16-f16";
            mmctx->vec_dot_1x1     = flat_vec_dot_f16_f16_aa_1x1;
            mmctx->vec_dot_2x1     = flat_vec_dot_f16_f16_aa_2x1;
            mmctx->vec_dot_2x2     = flat_vec_dot_f16_f16_aa_2x2;
            act_అడ్డువరుస_కొలత           = kparams->act_అడ్డువరుస_కొలత;
            ఆపు;

        స్థితి HTP_MM_KERNEL_HVX_F32_F32_VTCM:
            quant_task_func        = ఖాళీ;
            need_quant             = false;
            mmctx->type            = "f32-f32";
            mmctx->vec_dot_1x1     = flat_vec_dot_f32_f32_aa_1x1;
            mmctx->vec_dot_2x1     = flat_vec_dot_f32_f32_aa_2x1;
            mmctx->vec_dot_2x2     = flat_vec_dot_f32_f32_aa_2x2;
            act_అడ్డువరుస_కొలత           = kparams->act_అడ్డువరుస_కొలత;
            ఆపు;

        స్థితి HTP_MM_KERNEL_HVX_BF16_F32_VTCM:
            quant_task_func        = ఖాళీ;
            need_quant             = false;
            mmctx->type            = "bf16-f32";
            mmctx->vec_dot_1x1     = flat_vec_dot_bf16_f32_aa_1x1;
            mmctx->vec_dot_2x1     = flat_vec_dot_bf16_f32_aa_2x1;
            mmctx->vec_dot_2x2     = flat_vec_dot_bf16_f32_aa_2x2;
            act_అడ్డువరుస_కొలత           = kparams->act_అడ్డువరుస_కొలత;
            ఆపు;

        స్థితి HTP_MM_KERNEL_HVX_QUANT_ఖండం:
        స్థితి HTP_MM_KERNEL_HVX_QUANT_అడ్డువరుస:
        అప్రమేయ: {
            ఒకవేళ (hvx_mm_init_tiled_vec_dot(mmctx, src0->type) != 0) {
                తిరిగి HTP_స్థితి_NO_SUPPORT;
            }

            స్థిరరాశి uint32_t qk = QK_Q8_0_TILED;
            స్థిరరాశి uint32_t nb = (ne10 + qk - 1) / qk;
            స్థిరరాశి uint32_t మొత్తం_nb = act_nrows * nb;

            ఒకవేళ (act_nrows < octx->n_దారాలు && !is_batched) {
                n_quant_tasks = MIN(మొత్తం_nb, octx->n_దారాలు);
                quant_task_func = htp_mm_act_quant_ఖండం_func(src0->type, act->type);
                కొరకు (uint32_t ith = 0; ith < n_quant_tasks; ++ith) {
                    uint32_t ib_మొదటి = (మొత్తం_nb * ith) / n_quant_tasks;
                    uint32_t ib_చివరి  = (మొత్తం_nb * (ith + 1)) / n_quant_tasks;
                    mmctx->quant_ib_మొదటి[ith] = ib_మొదటి;
                    mmctx->quant_ib_చివరి[ith]  = ib_చివరి;
                    mmctx->quant_r[ith]        = ib_మొదటి / nb;
                    mmctx->quant_c[ith]        = ib_మొదటి % nb;
                }
            } లేకపోతే {
                n_quant_tasks = MIN(act_nrows, octx->n_దారాలు);
                quant_task_func = htp_mm_act_quant_అడ్డువరుస_func(src0->type, act->type);
            }
            act_అడ్డువరుస_కొలత = kparams->act_అడ్డువరుస_కొలత;
            ఆపు;
        }
    }

    స్థిరరాశి uint32_t m_chunk = (kparams->m_chunk > 0 && (uint32_t) kparams->m_chunk < act_nrows)
                           ? (uint32_t) kparams->m_chunk : act_nrows;
    స్థిరరాశి uint32_t m_layout_అడ్డువరుసలు = m_chunk;

    నిర్మాణం htp_mm_hvx_vtcm_layout L;
    htp_mm_hvx_vtcm_layout_build(&L, kparams->kernel_type, src0->type, ne10, m_layout_అడ్డువరుసలు, octx->n_దారాలు,
                                 dst_అడ్డువరుస_కొలత, src0_అడ్డువరుస_కొలత, act_అడ్డువరుస_కొలత, src2 ? src2->nb[1] : 0, kparams->n_prefetch, false, false);

    ఒకవేళ (kparams->kernel_type == HTP_MM_KERNEL_HVX_F16_F16_VTCM ||
        kparams->kernel_type == HTP_MM_KERNEL_HVX_F32_F32_VTCM ||
        kparams->kernel_type == HTP_MM_KERNEL_HVX_BF16_F32_VTCM ||
        kparams->kernel_type == HTP_MM_KERNEL_HVX_QUANT_అడ్డువరుస ||
        kparams->kernel_type == HTP_MM_KERNEL_HVX_QUANT_ఖండం) {
        mmctx->vtcm_act_కొలత_చొప్పున_thread = L.act_బైట్లు;
    } లేకపోతే {
        mmctx->vtcm_act_కొలత_చొప్పున_thread = fastdiv(L.act_బైట్లు, &octx->n_దారాలు_div);
    }

    mmctx->vtcm_src0_కొలత_చొప్పున_thread = fastdiv(L.src0_బైట్లు, &octx->n_దారాలు_div);
    mmctx->vtcm_dst_కొలత_చొప్పున_thread  = fastdiv(L.dst_బైట్లు, &octx->n_దారాలు_div);

    స్థిరరాశి size_t vtcm_కొలత = L.మొత్తం_బైట్లు;

    FARF(ఎగువ_2, "matmul-%s : src0-vtcm-size %zu act-vtcm-size %zu dst-vtcm-size %zu (%zu)\n", mmctx->type,
         L.src0_బైట్లు, L.act_బైట్లు, L.dst_బైట్లు, vtcm_కొలత);

    FARF(ఎగువ_2, "matmul-%s : %ux%ux%ux%u * %ux%ux%ux%u-> %ux%ux%ux%u (0x%p, 0x%p, 0x%p)\n", mmctx->type, src0->ne[0],
         src0->ne[1], src0->ne[2], src0->ne[3], act->ne[0], act->ne[1], act->ne[2], act->ne[3], dst->ne[0],
         dst->ne[1], dst->ne[2], dst->ne[3], src0->దత్తాంశం, act->దత్తాంశం, dst->దత్తాంశం);

    ఒకవేళ (octx->ctx->vtcm_కొలత < vtcm_కొలత) {
        FARF(ERROR, "matmul-%s : current VTCM reservation %zu is too small, needed %zu\n", mmctx->type,
             octx->ctx->vtcm_కొలత, vtcm_కొలత);
        తిరిగి HTP_స్థితి_VTCM_TOO_SMALL;
    }

    uint8_t * స్థిరరాశి పునాది = (uint8_t *) octx->ctx->vtcm_పునాది;
    mmctx->vtcm_act      = VTCM_LAYOUT_PTR(uint8_t, పునాది, L.off_act);
    mmctx->vtcm_src0     = VTCM_LAYOUT_PTR(uint8_t, పునాది, L.off_src0);
    mmctx->vtcm_bias     = VTCM_LAYOUT_PTR(uint8_t, పునాది, L.off_bias);
    mmctx->vtcm_dst      = VTCM_LAYOUT_PTR(uint8_t, పునాది, L.off_dst);
    mmctx->vtcm_act_raw  = VTCM_LAYOUT_PTR(uint8_t, పునాది, L.off_act_raw);

    mmctx->vtcm_src0_stride = src0_అడ్డువరుస_కొలత_padded;
    mmctx->vtcm_act_stride  = act_అడ్డువరుస_కొలత;
    ఒకవేళ (kparams->kernel_type == HTP_MM_KERNEL_HVX_QUANT_ఖండం || kparams->kernel_type == HTP_MM_KERNEL_HVX_QUANT_అడ్డువరుస) {
        mmctx->vtcm_act_raw_stride = hex_విడత_పైకి(ne10 * పరిమాణం(దశాంశం), QK_Q8_0_TILED * పరిమాణం(దశాంశం));
    } లేకపోతే ఒకవేళ (kparams->kernel_type == HTP_MM_KERNEL_HVX_F16_F16_VTCM) {
        mmctx->vtcm_act_raw_stride = hex_విడత_పైకి(ne10 * పరిమాణం(దశాంశం), 128);
    } లేకపోతే {
        mmctx->vtcm_act_raw_stride = 0;
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_INIT, 0);

    ఒకవేళ (kparams->m_chunk > 0 && (uint32_t) kparams->m_chunk < act_nrows) {
        కొరకు (uint32_t m_ప్రారంభం = 0; m_ప్రారంభం < act_nrows; m_ప్రారంభం += m_chunk) {
            స్థిరరాశి uint32_t cur_m_అడ్డువరుసలు = MIN(act_nrows - m_ప్రారంభం, m_chunk);
            mmctx->cur_m_ప్రారంభం = m_ప్రారంభం;
            mmctx->cur_m_అడ్డువరుసలు  = cur_m_అడ్డువరుసలు;

            ఒకవేళ (need_quant) {
                hvx_mm_transfer_act_dma(octx, kparams, act, mmctx->vtcm_act_raw, mmctx->vtcm_act_raw_stride, m_ప్రారంభం, cur_m_అడ్డువరుసలు);

                స్థిరరాశి uint32_t qk = QK_Q8_0_TILED;
                స్థిరరాశి uint32_t nb = (ne10 + qk - 1) / qk;
                స్థిరరాశి uint32_t మొత్తం_nb = cur_m_అడ్డువరుసలు * nb;
                uint32_t quant_tasks;
                work_వరుస_func_t q_func;
                ఒకవేళ (cur_m_అడ్డువరుసలు < octx->n_దారాలు && (kparams->kernel_type == HTP_MM_KERNEL_HVX_QUANT_ఖండం || kparams->kernel_type == HTP_MM_KERNEL_HVX_QUANT_అడ్డువరుస)) {
                    quant_tasks = MIN(మొత్తం_nb, octx->n_దారాలు);
                    q_func = htp_mm_act_quant_ఖండం_func(src0->type, act->type);
                    కొరకు (uint32_t ith = 0; ith < quant_tasks; ++ith) {
                        uint32_t ib_మొదటి = (మొత్తం_nb * ith) / quant_tasks;
                        uint32_t ib_చివరి  = (మొత్తం_nb * (ith + 1)) / quant_tasks;
                        mmctx->quant_ib_మొదటి[ith] = ib_మొదటి;
                        mmctx->quant_ib_చివరి[ith]  = ib_చివరి;
                        mmctx->quant_r[ith]        = ib_మొదటి / nb;
                        mmctx->quant_c[ith]        = ib_మొదటి % nb;
                    }
                } లేకపోతే {
                    quant_tasks = MIN(cur_m_అడ్డువరుసలు, octx->n_దారాలు);
                    q_func = quant_task_func;
                    mmctx->n_quant_అడ్డువరుసలు_చొప్పున_thread = (cur_m_అడ్డువరుసలు + quant_tasks - 1) / quant_tasks;
                }
                mmctx->n_quant_tasks = quant_tasks;
                work_వరుస_నిరంతరం(octx->ctx->work_వరుస, q_func, mmctx, quant_tasks);
            } లేకపోతే {
                hvx_mm_transfer_act_dma(octx, kparams, act, mmctx->vtcm_act, mmctx->vtcm_act_stride, m_ప్రారంభం, cur_m_అడ్డువరుసలు);
            }

            work_వరుస_నిరంతరం(octx->ctx->work_వరుస, matmul_పని_func, mmctx, octx->n_దారాలు);
        }
    } లేకపోతే {
        mmctx->cur_m_ప్రారంభం = 0;
        mmctx->cur_m_అడ్డువరుసలు  = act_nrows;

        ఒకవేళ (need_quant) {
            hvx_mm_transfer_act_dma(octx, kparams, act, mmctx->vtcm_act_raw, mmctx->vtcm_act_raw_stride, 0, act_nrows);
            mmctx->n_quant_అడ్డువరుసలు_చొప్పున_thread = (act_nrows + n_quant_tasks - 1) / n_quant_tasks;
            mmctx->n_quant_tasks = n_quant_tasks;
            work_వరుస_నిరంతరం(octx->ctx->work_వరుస, quant_task_func, mmctx, n_quant_tasks);
        } లేకపోతే {
            hvx_mm_transfer_act_dma(octx, kparams, act, mmctx->vtcm_act, mmctx->vtcm_act_stride, 0, act_nrows);
        }

        work_వరుస_నిరంతరం(octx->ctx->work_వరుస, matmul_పని_func, mmctx, octx->n_దారాలు);
    }

    తిరిగి HTP_స్థితి_OK;
}

స్థిర శూన్య dequantize_tiled_కార్మికుడు_loop(గుర్తులేని పూర్ణాంకం n, గుర్తులేని పూర్ణాంకం i, శూన్య * దత్తాంశం) {
    tiled_dequantize_state_t * state = (tiled_dequantize_state_t *) దత్తాంశం;
    నిర్మాణం htp_thread_trace * tr = &state->traces[i];
    htp_trace_event_ప్రారంభం(tr, HTP_TRACE_EVT_HVX_W_DEQUANT, i);
    కొరకు (గుర్తులేని పూర్ణాంకం task_id = i; task_id < (గుర్తులేని పూర్ణాంకం) state->n_tasks; task_id += n) {
        పూర్ణాంకం ప్రారంభం = task_id * state->n_tiles_చొప్పున_task;
        పూర్ణాంకం చివర   = hex_smin(ప్రారంభం + state->n_tiles_చొప్పున_task, state->n_tot_tiles);
        state->task_fn(state, ప్రారంభం, చివర);
    }
    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_W_DEQUANT, i);
}

స్థిర శూన్య transfer_output_chunk_కార్మికుడు_fn(గుర్తులేని పూర్ణాంకం n, గుర్తులేని పూర్ణాంకం i, శూన్య *దత్తాంశం) {
    output_transfer_task_state_t *st = (output_transfer_task_state_t *) దత్తాంశం;

    నిర్మాణం htp_thread_trace * tr = &st->traces[i];

    పూర్ణాంకం ప్రారంభం_chunk_idx = i * st->n_chunks_చొప్పున_task;
    htp_trace_event_ప్రారంభం(tr, HTP_TRACE_EVT_HVX_O_PROC, ప్రారంభం_chunk_idx);

    కొరకు (గుర్తులేని పూర్ణాంకం task_id = i; task_id < (గుర్తులేని పూర్ణాంకం)st->n_tasks; task_id += n) {
        పూర్ణాంకం    chunk_idx  = task_id * st->n_chunks_చొప్పున_task;
        size_t chunk_కొలత = hex_smin(st->n_tot_chunks - chunk_idx, st->n_chunks_చొప్పున_task);

        దశాంశం        *dst      = st->dst      + chunk_idx * st->dst_stride;
        స్థిరరాశి దశాంశం  *src2     = st->src2     ? (st->src2     + chunk_idx * st->src2_stride) : ఖాళీ;
        transfer_output_chunk_fp16_గమ్యం_fp32(dst, src2, st->vtcm_src, chunk_idx, chunk_కొలత, st->n_నిలువుల_సంఖ్య, st->dst_stride, st->src2_stride, st->dst_నిలువుల_సంఖ్య);
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_O_PROC, ప్రారంభం_chunk_idx);
}

రకనిర్వచనం నిర్మాణం {
    నిర్మాణం htp_context            * ctx;
    నిర్మాణం htp_thread_trace       * traces;
    __fp16                        * dst;
    dma_addr_t                      act_dma_addr;
    దశాంశం                         * vtcm_f32_act;
    uint32_t                        n_tasks;
    uint32_t                        n_tot_chunks;
    uint32_t                        n_chunks_చొప్పున_task;
    uint32_t                        k_ఖండం;
    uint32_t                        k_stride;
    uint32_t                        k_valid;
    size_t                          vtcm_f32_act_బైట్లు_చొప్పున_thread;
    uint32_t                        dma_అడుగు_అడ్డువరుసలు;
    uint32_t                        dma_అడుగు_అడ్డువరుసలు_జరుపు;
    uint32_t                        act_elem_కొలత;  // 4=F32 src1, 2=F16 src1
} activation_transfer_task_state_t;

రకనిర్వచనం నిర్మాణం {
    నిర్మాణం htp_context            * ctx;
    నిర్మాణం htp_thread_trace       * traces;
    __fp16                        * dst;
    dma_addr_t                      act_dma_addr;
    దశాంశం                         * vtcm_f32_act;
    uint32_t                        n_అడ్డువరుసలు;
    uint32_t                        k_ఖండం;
    uint32_t                        k_stride;
    uint32_t                        k_valid;
    uint32_t                        n_నిలువు_chunks;
    నిర్మాణం fastdiv_విలువలు           n_దారాలు_div;
    size_t                          vtcm_f32_act_బైట్లు;
    uint32_t                        dma_అడుగు_అడ్డువరుసలు;
    uint32_t                        dma_అడుగు_అడ్డువరుసలు_జరుపు;
} activation_transfer_నిలువు_chunk_state_t;

స్థిర శూన్య transfer_activation_chunk_fp32_గమ్యం_fp16_dma_pipelined_నిలువు_chunk(
        dma_వరుస *dma_q,
        __fp16 *పరిమిత vtcm_dst,
        dma_addr_t act_dma_addr,
        uint32_t n_అడ్డువరుసలు,
        uint32_t k_ఖండం,
        uint32_t k_stride,
        uint32_t k_chunk_valid,
        uint32_t c_మొదటి,
        uint32_t c_len,
        దశాంశం *thread_f32_act,
        నిర్మాణం htp_thread_trace *tr,
        uint32_t dma_అడుగు_అడ్డువరుసలు,
        uint32_t dma_అడుగు_అడ్డువరుసలు_జరుపు) {

    స్థిరరాశి uint32_t R = dma_అడుగు_అడ్డువరుసలు;
    స్థిరరాశి uint32_t n_అడ్డువరుసలు_padded = hex_align_పైకి(n_అడ్డువరుసలు, HTP_MM_HMX_TILE_N_అడ్డువరుసలు);

    స్థిరరాశి uint32_t n_అడుగులు = n_అడ్డువరుసలు_padded >> dma_అడుగు_అడ్డువరుసలు_జరుపు;

    // Push step 0
    ఒకవేళ (n_అడుగులు > 0 && n_అడ్డువరుసలు > 0) {
        uint32_t nrows_గమ్యం_fetch = hex_smin(n_అడ్డువరుసలు, R);
        dma_వరుస_నెట్టు(dma_q, dma_make_దత్తాంశం(thread_f32_act, act_dma_addr + (size_t) c_మొదటి * పరిమాణం(దశాంశం)),
                       c_len * పరిమాణం(దశాంశం), k_stride * పరిమాణం(దశాంశం), k_chunk_valid * పరిమాణం(దశాంశం), nrows_గమ్యం_fetch);
    }
    // Push step 1
    ఒకవేళ (n_అడుగులు > 1) {
        uint32_t తదుపరి_r = R * 1;
        ఒకవేళ (తదుపరి_r < n_అడ్డువరుసలు) {
            uint32_t nrows_గమ్యం_fetch = hex_smin(n_అడ్డువరుసలు - తదుపరి_r, R);
            దశాంశం *తదుపరి_buf = thread_f32_act + 1 * R * c_len;
            dma_వరుస_నెట్టు(dma_q, dma_make_దత్తాంశం(తదుపరి_buf, act_dma_addr + ((size_t) తదుపరి_r * k_stride + c_మొదటి) * పరిమాణం(దశాంశం)),
                           c_len * పరిమాణం(దశాంశం), k_stride * పరిమాణం(దశాంశం), k_chunk_valid * పరిమాణం(దశాంశం), nrows_గమ్యం_fetch);
        }
    }
    కొరకు (uint32_t s = 0; s < n_అడుగులు; ++s) {
        uint32_t r = s << dma_అడుగు_అడ్డువరుసలు_జరుపు;
        దశాంశం *curr_buf = thread_f32_act;

        ఒకవేళ (r < n_అడ్డువరుసలు) {
            curr_buf = (దశాంశం *) dma_వరుస_తీయి(dma_q).dst;
        }

        htp_trace_event_ప్రారంభం(tr, HTP_TRACE_EVT_HVX_A_PREP, r);
        కొరకు (uint32_t p = 0; p < (R >> 1); ++p) {
            uint32_t అడ్డువరుస_idx = r + (p << 1);
            దశాంశం *pair_buf = curr_buf + (p << 1) * c_len;
            bool r0_valid = ((అడ్డువరుస_idx + 0) < n_అడ్డువరుసలు);
            bool r1_valid = ((అడ్డువరుస_idx + 1) < n_అడ్డువరుసలు);

            transfer_activation_అడ్డువరుస_pair_fp32_గమ్యం_fp16_నిలువు_chunk(
                vtcm_dst, pair_buf, pair_buf + c_len, అడ్డువరుస_idx, k_ఖండం, c_మొదటి, c_len, k_chunk_valid, r0_valid, r1_valid
            );
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_A_PREP, r);

        // Push step s + 2
        uint32_t తదుపరి_s = s + 2;
        uint32_t తదుపరి_r = తదుపరి_s << dma_అడుగు_అడ్డువరుసలు_జరుపు;
        ఒకవేళ (తదుపరి_r < n_అడ్డువరుసలు) {
            uint32_t nrows_గమ్యం_fetch = hex_smin(n_అడ్డువరుసలు - తదుపరి_r, R);
            dma_వరుస_నెట్టు(dma_q, dma_make_దత్తాంశం(curr_buf, act_dma_addr + ((size_t) తదుపరి_r * k_stride + c_మొదటి) * పరిమాణం(దశాంశం)),
                           c_len * పరిమాణం(దశాంశం), k_stride * పరిమాణం(దశాంశం), k_chunk_valid * పరిమాణం(దశాంశం), nrows_గమ్యం_fetch);
        }
    }
}

స్థిర శూన్య transfer_activation_chunk_నిలువు_chunk_కార్మికుడు_fn(గుర్తులేని పూర్ణాంకం n, గుర్తులేని పూర్ణాంకం i, శూన్య *దత్తాంశం) {
    activation_transfer_నిలువు_chunk_state_t *st = (activation_transfer_నిలువు_chunk_state_t *) దత్తాంశం;
    నిర్మాణం htp_thread_trace * tr = &st->traces[i];

    uint32_t n_ఖండాలు = st->k_ఖండం / 32;
    uint32_t b_మొదటి = fastdiv(n_ఖండాలు * i, &st->n_దారాలు_div);
    uint32_t b_చివరి  = fastdiv(n_ఖండాలు * (i + 1), &st->n_దారాలు_div);
    uint32_t c_మొదటి = b_మొదటి * 32;
    uint32_t c_చివరి = b_చివరి * 32;
    uint32_t c_len = c_చివరి - c_మొదటి;

    ఒకవేళ (c_len == 0) {
        తిరిగి;
    }

    uint32_t k_chunk_valid = 0;
    ఒకవేళ (st->k_valid > c_మొదటి) {
        k_chunk_valid = hex_smin(st->k_valid, c_చివరి) - c_మొదటి;
    }

    __fp16 *dst = st->dst;

    size_t thread_scratch_బైట్లు = hex_align_కిందికి(fastdiv(st->vtcm_f32_act_బైట్లు, &st->n_దారాలు_div), 128);
    దశాంశం *thread_f32_act = (దశాంశం *)((అక్షరం *)st->vtcm_f32_act + i * thread_scratch_బైట్లు);

    transfer_activation_chunk_fp32_గమ్యం_fp16_dma_pipelined_నిలువు_chunk(
        st->ctx->dma[i], dst, st->act_dma_addr, st->n_అడ్డువరుసలు, st->k_ఖండం, st->k_stride, k_chunk_valid,
        c_మొదటి, c_len, thread_f32_act, tr, st->dma_అడుగు_అడ్డువరుసలు, st->dma_అడుగు_అడ్డువరుసలు_జరుపు
    );
}

స్థిర శూన్య transfer_activation_chunk_గమ్యం_fp16_dma_pipelined(
        dma_వరుస *dma_q,
        __fp16 *పరిమిత vtcm_dst,
        dma_addr_t act_dma_addr,
        uint32_t n_అడ్డువరుసలు,
        uint32_t k_ఖండం,
        uint32_t k_stride,
        uint32_t k_valid,
        uint8_t *thread_act,
        uint32_t act_elem_కొలత,
        నిర్మాణం htp_thread_trace *tr,
        uint32_t dma_అడుగు_అడ్డువరుసలు,
        uint32_t dma_అడుగు_అడ్డువరుసలు_జరుపు) {

    స్థిరరాశి uint32_t R = dma_అడుగు_అడ్డువరుసలు;
    స్థిరరాశి uint32_t n_అడ్డువరుసలు_padded = hex_align_పైకి(n_అడ్డువరుసలు, HTP_MM_HMX_TILE_N_అడ్డువరుసలు);

    స్థిరరాశి uint32_t n_అడుగులు = n_అడ్డువరుసలు_padded >> dma_అడుగు_అడ్డువరుసలు_జరుపు;

    స్థిరరాశి bool act_is_f16 = (act_elem_కొలత == పరిమాణం(__fp16));
    స్థిరరాశి size_t అడ్డువరుస_బైట్లు = (size_t) k_ఖండం * act_elem_కొలత;  // staging row (DMA dst stride)
    స్థిరరాశి size_t src_అడ్డువరుస_బైట్లు = (size_t) k_stride * act_elem_కొలత;  // DDR row (DMA src stride)
    స్థిరరాశి size_t వెడల్పు_బైట్లు = (size_t) k_valid * act_elem_కొలత;

    // Push step 0
    ఒకవేళ (n_అడుగులు > 0 && n_అడ్డువరుసలు > 0) {
        uint32_t nrows_గమ్యం_fetch = hex_smin(n_అడ్డువరుసలు, R);
        dma_వరుస_నెట్టు(dma_q, dma_make_దత్తాంశం(thread_act, act_dma_addr),
                       అడ్డువరుస_బైట్లు, src_అడ్డువరుస_బైట్లు, వెడల్పు_బైట్లు, nrows_గమ్యం_fetch);
    }
    // Push step 1 (if valid)
    ఒకవేళ (n_అడుగులు > 1) {
        uint32_t తదుపరి_r = R * 1;
        ఒకవేళ (తదుపరి_r < n_అడ్డువరుసలు) {
            uint32_t nrows_గమ్యం_fetch = hex_smin(n_అడ్డువరుసలు - తదుపరి_r, R);
            uint8_t *తదుపరి_buf = thread_act + 1 * R * అడ్డువరుస_బైట్లు;
            dma_వరుస_నెట్టు(dma_q, dma_make_దత్తాంశం(తదుపరి_buf, act_dma_addr + (size_t) తదుపరి_r * src_అడ్డువరుస_బైట్లు),
                           అడ్డువరుస_బైట్లు, src_అడ్డువరుస_బైట్లు, వెడల్పు_బైట్లు, nrows_గమ్యం_fetch);
        }
    }
    కొరకు (uint32_t s = 0; s < n_అడుగులు; ++s) {
        uint32_t r = s << dma_అడుగు_అడ్డువరుసలు_జరుపు;
        uint8_t *curr_buf = thread_act;

        ఒకవేళ (r < n_అడ్డువరుసలు) {
            curr_buf = (uint8_t *) dma_వరుస_తీయి(dma_q).dst;
        }

        htp_trace_event_ప్రారంభం(tr, HTP_TRACE_EVT_HVX_A_PREP, r);
        // Two copies of the pair loop so the type is resolved once per step and the
        // row-pair kernels stay direct (inlinable) calls.
        ఒకవేళ (act_is_f16) {
            కొరకు (uint32_t p = 0; p < (R >> 1); ++p) {
                uint32_t అడ్డువరుస_idx = r + (p << 1);
                స్థిరరాశి __fp16 *pair_buf = (స్థిరరాశి __fp16 *) (curr_buf + (p << 1) * అడ్డువరుస_బైట్లు);
                bool r0_valid = ((అడ్డువరుస_idx + 0) < n_అడ్డువరుసలు);
                bool r1_valid = ((అడ్డువరుస_idx + 1) < n_అడ్డువరుసలు);

                transfer_activation_అడ్డువరుస_pair_f16_గమ్యం_f16(vtcm_dst, pair_buf, pair_buf + k_ఖండం, అడ్డువరుస_idx, k_ఖండం, k_valid, r0_valid, r1_valid);
            }
        } లేకపోతే {
            కొరకు (uint32_t p = 0; p < (R >> 1); ++p) {
                uint32_t అడ్డువరుస_idx = r + (p << 1);
                స్థిరరాశి దశాంశం *pair_buf = (స్థిరరాశి దశాంశం *) (curr_buf + (p << 1) * అడ్డువరుస_బైట్లు);
                bool r0_valid = ((అడ్డువరుస_idx + 0) < n_అడ్డువరుసలు);
                bool r1_valid = ((అడ్డువరుస_idx + 1) < n_అడ్డువరుసలు);

                transfer_activation_అడ్డువరుస_pair_fp32_గమ్యం_fp16(vtcm_dst, pair_buf, pair_buf + k_ఖండం, అడ్డువరుస_idx, k_ఖండం, k_valid, r0_valid, r1_valid);
            }
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_A_PREP, r);

        // Push step s + 2
        uint32_t తదుపరి_s = s + 2;
        uint32_t తదుపరి_r = తదుపరి_s << dma_అడుగు_అడ్డువరుసలు_జరుపు;
        ఒకవేళ (తదుపరి_r < n_అడ్డువరుసలు) {
            uint32_t nrows_గమ్యం_fetch = hex_smin(n_అడ్డువరుసలు - తదుపరి_r, R);
            dma_వరుస_నెట్టు(dma_q, dma_make_దత్తాంశం(curr_buf, act_dma_addr + (size_t) తదుపరి_r * src_అడ్డువరుస_బైట్లు),
                           అడ్డువరుస_బైట్లు, src_అడ్డువరుస_బైట్లు, వెడల్పు_బైట్లు, nrows_గమ్యం_fetch);
        }
    }
}

స్థిర శూన్య transfer_activation_chunk_కార్మికుడు_fn(గుర్తులేని పూర్ణాంకం n, గుర్తులేని పూర్ణాంకం i, శూన్య *దత్తాంశం) {
    activation_transfer_task_state_t *st = (activation_transfer_task_state_t *) దత్తాంశం;

    నిర్మాణం htp_thread_trace * tr = &st->traces[i];

    కొరకు (గుర్తులేని పూర్ణాంకం task_id = i; task_id < (గుర్తులేని పూర్ణాంకం)st->n_tasks; task_id += n) {
        పూర్ణాంకం    chunk_idx  = task_id * st->n_chunks_చొప్పున_task;
        size_t chunk_కొలత = hex_smin(st->n_tot_chunks - chunk_idx, st->n_chunks_చొప్పున_task);

        __fp16      *dst = st->dst + chunk_idx * st->k_ఖండం;
        స్థిరరాశి dma_addr_t act_dma_addr = st->act_dma_addr + (size_t) chunk_idx * st->k_stride * st->act_elem_కొలత;

        uint8_t *thread_act = (uint8_t *) st->vtcm_f32_act + i * st->vtcm_f32_act_బైట్లు_చొప్పున_thread;
        transfer_activation_chunk_గమ్యం_fp16_dma_pipelined(
            st->ctx->dma[i], dst, act_dma_addr, chunk_కొలత, st->k_ఖండం, st->k_stride, st->k_valid, thread_act, st->act_elem_కొలత, tr, st->dma_అడుగు_అడ్డువరుసలు, st->dma_అడుగు_అడ్డువరుసలు_జరుపు
        );
    }
}

రకనిర్వచనం నిర్మాణం {
    నిర్మాణం htp_thread_trace       * traces;
    స్థిరరాశి నిర్మాణం mmid_అడ్డువరుస_mapping * మాతృక_అడ్డువరుసలు;
    __fp16                        * dst;
    స్థిరరాశి దశాంశం                   * src;
    uint32_t                        n_tasks;
    uint32_t                        n_tot_chunks;
    uint32_t                        n_chunks_చొప్పున_task;
    uint32_t                        k_ఖండం;
    uint32_t                        cur_a;
    uint32_t                        mapping_stride;
    uint32_t                        ne11;
    నిర్మాణం fastdiv_విలువలు           ne11_div;
    size_t                          nb11;
    size_t                          nb12;
    uint32_t                        ప్రారంభం_అడ్డువరుస;
    uint32_t                        cne1;
    uint32_t                        k_valid;
} activation_transfer_gathered_task_state_t;

రకనిర్వచనం నిర్మాణం {
    నిర్మాణం htp_thread_trace       * traces;
    స్థిరరాశి నిర్మాణం mmid_అడ్డువరుస_mapping * మాతృక_అడ్డువరుసలు;
    స్థిరరాశి __fp16                  * vtcm_src;
    దశాంశం                         * dst;
    uint32_t                        n_tasks;
    uint32_t                        n_tot_chunks;
    uint32_t                        n_chunks_చొప్పున_task;
    uint32_t                        n_నిలువుల_సంఖ్య;
    uint32_t                        cur_a;
    uint32_t                        mapping_stride;
    size_t                          dst_nb1;
    size_t                          dst_nb2;
    uint32_t                        ప్రారంభం_అడ్డువరుస;
    uint32_t                        cne1;
} output_transfer_scattered_task_state_t;

స్థిర శూన్య transfer_activation_chunk_gathered_కార్మికుడు_fn(గుర్తులేని పూర్ణాంకం n, గుర్తులేని పూర్ణాంకం i, శూన్య *దత్తాంశం) {
    activation_transfer_gathered_task_state_t *st = దత్తాంశం;
    నిర్మాణం htp_thread_trace * tr = &st->traces[i];
    పూర్ణాంకం chunk_idx      = i;
    పూర్ణాంకం chunk_కొలత     = st->n_chunks_చొప్పున_task;
    పూర్ణాంకం vtcm_ప్రారంభం_అడ్డువరుస = chunk_idx * chunk_కొలత;
    పూర్ణాంకం ప్రారంభం_అడ్డువరుస      = st->ప్రారంభం_అడ్డువరుస + vtcm_ప్రారంభం_అడ్డువరుస;
    పూర్ణాంకం n_అడ్డువరుసలు         = hex_smin(st->cne1 - ప్రారంభం_అడ్డువరుస, chunk_కొలత);
    ఒకవేళ (n_అడ్డువరుసలు > 0) {
        htp_trace_event_ప్రారంభం(tr, HTP_TRACE_EVT_HVX_A_PREP, chunk_idx);
        transfer_activation_chunk_fp32_గమ్యం_fp16_gathered(
            st->dst, st->src, ప్రారంభం_అడ్డువరుస, vtcm_ప్రారంభం_అడ్డువరుస, n_అడ్డువరుసలు, st->k_ఖండం,
            st->మాతృక_అడ్డువరుసలు, st->cur_a, st->mapping_stride,
            st->ne11, &st->ne11_div, st->nb11, st->nb12, st->cne1, st->k_valid);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_A_PREP, chunk_idx);
    }
}

స్థిర శూన్య transfer_activation_chunk_gathered_కార్మికుడు_flat_fn(గుర్తులేని పూర్ణాంకం n, గుర్తులేని పూర్ణాంకం i, శూన్య *దత్తాంశం) {
    activation_transfer_gathered_task_state_t *st = దత్తాంశం;
    నిర్మాణం htp_thread_trace * tr = &st->traces[i];
    పూర్ణాంకం chunk_idx = i;
    పూర్ణాంకం chunk_కొలత = st->n_chunks_చొప్పున_task;
    పూర్ణాంకం vtcm_ప్రారంభం_అడ్డువరుస = chunk_idx * chunk_కొలత;
    పూర్ణాంకం ప్రారంభం_అడ్డువరుస = st->ప్రారంభం_అడ్డువరుస + vtcm_ప్రారంభం_అడ్డువరుస;
    పూర్ణాంకం n_అడ్డువరుసలు = hex_smin(st->cne1 - ప్రారంభం_అడ్డువరుస, chunk_కొలత);
    ఒకవేళ (n_అడ్డువరుసలు > 0) {
        htp_trace_event_ప్రారంభం(tr, HTP_TRACE_EVT_HVX_A_PREP, chunk_idx);
        transfer_activation_chunk_fp32_గమ్యం_fp16_gathered_flat(
            st->dst, st->src, ప్రారంభం_అడ్డువరుస, vtcm_ప్రారంభం_అడ్డువరుస, n_అడ్డువరుసలు, st->k_ఖండం,
            st->మాతృక_అడ్డువరుసలు, st->cur_a, st->mapping_stride,
            st->nb12, st->cne1, st->k_valid);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_A_PREP, chunk_idx);
    }
}

స్థిర శూన్య transfer_output_chunk_scattered_కార్మికుడు_fn(గుర్తులేని పూర్ణాంకం n, గుర్తులేని పూర్ణాంకం i, శూన్య *దత్తాంశం) {
    output_transfer_scattered_task_state_t *st = దత్తాంశం;
    నిర్మాణం htp_thread_trace * tr = &st->traces[i];
    పూర్ణాంకం chunk_idx = i;
    పూర్ణాంకం chunk_కొలత = st->n_chunks_చొప్పున_task;
    పూర్ణాంకం vtcm_ప్రారంభం_అడ్డువరుస = chunk_idx * chunk_కొలత;
    పూర్ణాంకం ప్రారంభం_అడ్డువరుస = st->ప్రారంభం_అడ్డువరుస + vtcm_ప్రారంభం_అడ్డువరుస;
    పూర్ణాంకం n_అడ్డువరుసలు = hex_smin(st->cne1 - ప్రారంభం_అడ్డువరుస, chunk_కొలత);
    ఒకవేళ (n_అడ్డువరుసలు > 0) {
        htp_trace_event_ప్రారంభం(tr, HTP_TRACE_EVT_HVX_O_PROC, chunk_idx);
        transfer_output_chunk_fp16_గమ్యం_fp32_scattered(
            st->dst, st->vtcm_src, ప్రారంభం_అడ్డువరుస, vtcm_ప్రారంభం_అడ్డువరుస, n_అడ్డువరుసలు, st->n_నిలువుల_సంఖ్య,
            st->మాతృక_అడ్డువరుసలు, st->cur_a, st->mapping_stride,
            st->dst_nb1, st->dst_nb2, st->cne1);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_O_PROC, chunk_idx);
    }
}

// --- HMX Dispatchers & Entry Points ---

స్థిర అంతఃవరుస tiled_dequant_task_fn_t hmx_get_dequant_కార్మికుడు_fn(పూర్ణాంకం బరువు_type, bool is_tiled) {
    ఎంపిక (బరువు_type) {
        స్థితి HTP_TYPE_Q4_0:   తిరిగి dequantize_tiled_బరువు_గమ్యం_fp16_task_q4_0;
        స్థితి HTP_TYPE_IQ4_NL: తిరిగి dequantize_tiled_బరువు_గమ్యం_fp16_task_iq4_nl;
        స్థితి HTP_TYPE_Q4_1:
        స్థితి HTP_TYPE_Q4_K:   తిరిగి dequantize_tiled_బరువు_గమ్యం_fp16_task_q4_1;
        స్థితి HTP_TYPE_MXFP4:  తిరిగి dequantize_tiled_బరువు_గమ్యం_fp16_task_mxfp4;
        స్థితి HTP_TYPE_Q8_0:   తిరిగి dequantize_tiled_బరువు_గమ్యం_fp16_task_q8_0;
        స్థితి HTP_TYPE_Q5_K:   తిరిగి dequantize_tiled_బరువు_గమ్యం_fp16_task_q5_k;
        స్థితి HTP_TYPE_Q6_K:   తిరిగి dequantize_tiled_బరువు_గమ్యం_fp16_task_q6_k;
        స్థితి HTP_TYPE_Q3_K:   తిరిగి dequantize_tiled_బరువు_గమ్యం_fp16_task_q3_k;
        స్థితి HTP_TYPE_Q2_K:   తిరిగి dequantize_tiled_బరువు_గమ్యం_fp16_task_q2_k;
        స్థితి HTP_TYPE_Q1_0:   తిరిగి dequantize_tiled_బరువు_గమ్యం_fp16_task_q1_0;
        స్థితి HTP_TYPE_Q2_0:   తిరిగి dequantize_tiled_బరువు_గమ్యం_fp16_task_q2_0;
        స్థితి HTP_TYPE_F16:    తిరిగి is_tiled ? ఖాళీ : convert_f16_బరువు_గమ్యం_fp16_tiles_task;
        స్థితి HTP_TYPE_BF16:   తిరిగి convert_bf16_బరువు_గమ్యం_fp16_tiles_task;
        స్థితి HTP_TYPE_F32:    తిరిగి quantize_f32_బరువు_గమ్యం_fp16_tiles_task;
        అప్రమేయ:              తిరిగి ఖాళీ;
    }
}

స్థిర శూన్య dequantize_tiled_బరువు_chunk_గమ్యం_fp16_tiles(
        నిర్మాణం htp_context *ctx, __fp16 *vtcm_dst,
        స్థిరరాశి శూన్య *బరువు_src_ddr,
        పూర్ణాంకం n_నిలువుల_సంఖ్య, పూర్ణాంకం k_ఖండం,
        size_t అడ్డువరుస_stride, పూర్ణాంకం బరువు_type,
        పూర్ణాంకం n_k_tiles, నిర్మాణం fastdiv_విలువలు n_k_tiles_div,
        tiled_dequant_task_fn_t dequant_కార్మికుడు_fn, పూర్ణాంకం n_దారాలు) {

    ఒకవేళ (!dequant_కార్మికుడు_fn) {
        తిరిగి;
    }

    ప్రతిజ్ఞా(k_ఖండం % HTP_MM_HMX_TILE_N_నిలువుల_సంఖ్య == 0);

    స్థిరరాశి size_t n_నిలువు_tiles = hmx_ceil_div(n_నిలువుల_సంఖ్య, HTP_MM_HMX_TILE_N_నిలువుల_సంఖ్య);
    స్థిరరాశి uint32_t n_బరువు_tiles = k_ఖండం / htp_tensor_get_tile_k(బరువు_type);
    స్థిరరాశి size_t n_tot_tiles = n_నిలువు_tiles * n_బరువు_tiles;

    size_t n_tiles_చొప్పున_task = (n_దారాలు == 1) ? n_tot_tiles : hmx_ceil_div(n_tot_tiles, n_దారాలు);

    tiled_dequantize_state_t state;
    state.n_tasks          = (n_tot_tiles + n_tiles_చొప్పున_task - 1) / n_tiles_చొప్పున_task;
    state.n_tot_tiles      = n_tot_tiles;
    state.n_tiles_చొప్పున_task = n_tiles_చొప్పున_task;
    state.dst              = vtcm_dst;
    state.src              = (స్థిరరాశి uint8_t *)బరువు_src_ddr;
    state.task_fn          = dequant_కార్మికుడు_fn;
    state.n_నిలువుల_సంఖ్య           = n_నిలువుల_సంఖ్య;
    state.k_ఖండం          = k_ఖండం;
    state.అడ్డువరుస_stride       = అడ్డువరుస_stride;
    state.బరువు_type      = బరువు_type;
    state.n_k_tiles        = n_k_tiles;
    state.n_బరువు_tiles   = n_బరువు_tiles;
    state.n_k_tiles_div    = n_k_tiles_div;
    state.traces           = ctx->trace;
    state.ctx              = ctx;

    state.tile_కొలత = htp_tensor_get_tile_కొలత(బరువు_type);
    state.aligned_tile_కొలత = htp_tensor_get_tile_padded_కొలత(బరువు_type);

    ఒకవేళ (state.n_tasks == 1 || n_దారాలు == 1) {
        dequantize_tiled_కార్మికుడు_loop(1, 0, &state);
    } లేకపోతే {
        పూర్ణాంకం n_tasks = hex_smin((పూర్ణాంకం) state.n_tasks, n_దారాలు);
        కార్మికుడు_pool_నిరంతరం_func(ctx->కార్మికుడు_pool, dequantize_tiled_కార్మికుడు_loop, &state, n_tasks);
    }
}

రకనిర్వచనం నిర్మాణం {
    నిర్మాణం htp_context      * ctx;
    నిర్మాణం htp_thread_trace * traces;
    దశాంశం                   * dst;
    స్థిరరాశి __fp16            * vtcm_src;
    స్థిరరాశి దశాంశం             * src2;
    uint32_t                  n_అడ్డువరుసలు;
    uint32_t                  n_నిలువుల_సంఖ్య;
    uint32_t                  dst_stride;
    uint32_t                  src2_stride;
    uint32_t                  dst_నిలువుల_సంఖ్య;
    నిర్మాణం fastdiv_విలువలు     n_దారాలు_div;
} output_transfer_నిలువు_chunk_state_t;

స్థిర శూన్య transfer_output_chunk_నిలువు_chunk_కార్మికుడు_fn(గుర్తులేని పూర్ణాంకం n, గుర్తులేని పూర్ణాంకం i, శూన్య *దత్తాంశం) {
    (శూన్య) n;
    output_transfer_నిలువు_chunk_state_t *st = (output_transfer_నిలువు_chunk_state_t *) దత్తాంశం;
    నిర్మాణం htp_thread_trace * tr = &st->traces[i];

    // Round up: the last block is partial when N is not 32-aligned. Its pad
    // columns are dropped by the dst_cols clamp inside the store.
    uint32_t n_ఖండాలు = hmx_ceil_div(st->n_నిలువుల_సంఖ్య, 32);
    uint32_t b_మొదటి  = fastdiv(n_ఖండాలు * i, &st->n_దారాలు_div);
    uint32_t b_చివరి   = fastdiv(n_ఖండాలు * (i + 1), &st->n_దారాలు_div);
    uint32_t c_మొదటి  = b_మొదటి * 32;
    uint32_t c_చివరి   = b_చివరి * 32;
    uint32_t c_len    = c_చివరి - c_మొదటి;

    ఒకవేళ (c_len == 0) తిరిగి;

    htp_trace_event_ప్రారంభం(tr, HTP_TRACE_EVT_HVX_O_PROC, c_మొదటి);

    స్థిరరాశి __fp16 *vtcm_src = st->vtcm_src + b_మొదటి * HTP_MM_HMX_TILE_N_ELMS;
    స్థిరరాశి దశాంశం *src2      = st->src2 ? (st->src2 + c_మొదటి) : ఖాళీ;
    దశాంశం *dst             = st->dst + c_మొదటి;

    పూర్ణాంకం chunk_dst_నిలువుల_సంఖ్య = (పూర్ణాంకం)st->dst_నిలువుల_సంఖ్య - (పూర్ణాంకం)c_మొదటి;
    ఒకవేళ (chunk_dst_నిలువుల_సంఖ్య > 0) {
        transfer_output_chunk_fp16_గమ్యం_fp32_నిలువు_chunk(
            dst, src2, vtcm_src, 0, st->n_అడ్డువరుసలు, c_len, st->n_నిలువుల_సంఖ్య,
            st->dst_stride, st->src2_stride, (uint32_t)chunk_dst_నిలువుల_సంఖ్య
        );
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_O_PROC, c_మొదటి);
}

స్థిర శూన్య transfer_output_chunk_threaded(నిర్మాణం htp_context *ctx, దశాంశం *dst, స్థిరరాశి దశాంశం *src2, స్థిరరాశి __fp16 *vtcm_src,
                                              పూర్ణాంకం n_అడ్డువరుసలు, పూర్ణాంకం n_నిలువుల_సంఖ్య, పూర్ణాంకం dst_stride, uint32_t src2_stride, పూర్ణాంకం dst_నిలువుల_సంఖ్య, పూర్ణాంకం n_దారాలు) {
    ఒకవేళ (n_అడ్డువరుసలు <= 0) తిరిగి;

    uint32_t n_ఖండాలు = hmx_ceil_div((uint32_t) n_నిలువుల_సంఖ్య, 32);
    ఒకవేళ (n_దారాలు > 1 && n_ఖండాలు >= (uint32_t)n_దారాలు) {
        నిర్మాణం fastdiv_విలువలు n_దారాలు_div = (n_దారాలు == (పూర్ణాంకం)ctx->n_దారాలు) ? ctx->n_దారాలు_div : init_fastdiv_విలువలు(n_దారాలు);
        output_transfer_నిలువు_chunk_state_t నిలువు_state;
        నిలువు_state.dst = dst;
        నిలువు_state.src2 = src2;
        నిలువు_state.vtcm_src = vtcm_src;
        నిలువు_state.n_అడ్డువరుసలు = (uint32_t)n_అడ్డువరుసలు;
        నిలువు_state.n_నిలువుల_సంఖ్య = (uint32_t)n_నిలువుల_సంఖ్య;
        నిలువు_state.dst_stride = (uint32_t)dst_stride;
        నిలువు_state.src2_stride = src2_stride;
        నిలువు_state.dst_నిలువుల_సంఖ్య = (uint32_t)dst_నిలువుల_సంఖ్య;
        నిలువు_state.n_దారాలు_div = n_దారాలు_div;
        నిలువు_state.traces = ctx->trace;
        నిలువు_state.ctx = ctx;

        కార్మికుడు_pool_నిరంతరం_func(ctx->కార్మికుడు_pool, transfer_output_chunk_నిలువు_chunk_కార్మికుడు_fn, &నిలువు_state, n_దారాలు);
        తిరిగి;
    }

    size_t n_tot_chunks      = n_అడ్డువరుసలు;
    size_t n_chunks_చొప్పున_task = (n_దారాలు == 1) ? n_tot_chunks : hmx_ceil_div(n_అడ్డువరుసలు, n_దారాలు);
    n_chunks_చొప్పున_task        = hex_align_పైకి(n_chunks_చొప్పున_task, 2);

    పూర్ణాంకం actual_దారాలు = hmx_ceil_div(n_అడ్డువరుసలు, n_chunks_చొప్పున_task);

    output_transfer_task_state_t state;
    state.n_tasks           = actual_దారాలు;
    state.n_tot_chunks      = n_tot_chunks;
    state.n_chunks_చొప్పున_task = n_chunks_చొప్పున_task;
    state.dst               = dst;
    state.src2              = src2;
    state.vtcm_src          = vtcm_src;
    state.n_నిలువుల_సంఖ్య            = n_నిలువుల_సంఖ్య;
    state.dst_stride        = dst_stride;
    state.src2_stride       = src2_stride;
    state.dst_నిలువుల_సంఖ్య          = dst_నిలువుల_సంఖ్య;
    state.traces            = ctx->trace;

    ఒకవేళ (actual_దారాలు <= 1) {
        transfer_output_chunk_కార్మికుడు_fn(1, 0, &state);
    } లేకపోతే {
        కార్మికుడు_pool_నిరంతరం_func(ctx->కార్మికుడు_pool, transfer_output_chunk_కార్మికుడు_fn, &state, actual_దారాలు);
    }
}

నిర్మాణం activation_transfer_params {
    నిర్మాణం htp_context *          ctx;
    __fp16 *                      dst;
    dma_addr_t                    act_dma_addr;
    పూర్ణాంకం                           n_అడ్డువరుసలు;
    పూర్ణాంకం                           k_ఖండం;
    పూర్ణాంకం                           k_stride;
    పూర్ణాంకం                           n_దారాలు;
    స్థిరరాశి నిర్మాణం fastdiv_విలువలు * act_దారాలు_div;
    స్థిరరాశి నిర్మాణం fastdiv_విలువలు * k_div;
    పూర్ణాంకం                           k_valid;
    దశాంశం *                       vtcm_f32_act;
    size_t                        vtcm_f32_act_బైట్లు;
    uint32_t                      act_elem_కొలత;  // 4=F32 src1, 2=F16 src1
};

స్థిర శూన్య transfer_activation_chunk_threaded(స్థిరరాశి నిర్మాణం activation_transfer_params * params) {
    నిర్మాణం htp_context *          ctx                = params->ctx;
    __fp16 *                      dst                = params->dst;
    స్థిరరాశి dma_addr_t              act_dma_addr       = params->act_dma_addr;
    పూర్ణాంకం                           n_అడ్డువరుసలు             = params->n_అడ్డువరుసలు;
    పూర్ణాంకం                           k_ఖండం            = params->k_ఖండం;
    పూర్ణాంకం                           k_stride           = params->k_stride;
    పూర్ణాంకం                           n_దారాలు          = params->n_దారాలు;
    స్థిరరాశి నిర్మాణం fastdiv_విలువలు * act_దారాలు_div    = params->act_దారాలు_div;
    స్థిరరాశి నిర్మాణం fastdiv_విలువలు * k_div              = params->k_div;
    పూర్ణాంకం                           k_valid            = params->k_valid;
    దశాంశం *                       vtcm_f32_act       = params->vtcm_f32_act;
    size_t                        vtcm_f32_act_బైట్లు = params->vtcm_f32_act_బైట్లు;
    // element size of the activation rows (4 = F32, 2 = F16).
    స్థిరరాశి uint32_t                act_elem_కొలత      = params->act_elem_కొలత ? params->act_elem_కొలత : (uint32_t) పరిమాణం(దశాంశం);

    ఒకవేళ (n_అడ్డువరుసలు <= 0) {
        తిరిగి;
    }

    స్థిరరాశి size_t n_tasks = (n_అడ్డువరుసలు + 31) >> 5;
    ఒకవేళ (act_elem_కొలత == పరిమాణం(దశాంశం) && n_దారాలు > 1 && k_ఖండం > 32 && n_tasks < (size_t) n_దారాలు) {
        // Calculate step rows parameters for column-chunked dma pipelining
        uint32_t dma_అడుగు_అడ్డువరుసలు = 2;
        uint32_t dma_అడుగు_అడ్డువరుసలు_జరుపు = 1;
        ఒకవేళ (vtcm_f32_act_బైట్లు > 0) {
            size_t thread_scratch_బైట్లు = hex_align_కిందికి(fastdiv(vtcm_f32_act_బైట్లు, act_దారాలు_div), 128);
            size_t thread_scratch_elements = thread_scratch_బైట్లు / పరిమాణం(దశాంశం);
            size_t dma_అడుగు_అడ్డువరుసలు_max = fastdiv(thread_scratch_elements / 2, k_div);
            ఒకవేళ (dma_అడుగు_అడ్డువరుసలు_max >= 4) {
                dma_అడుగు_అడ్డువరుసలు = 4;
                dma_అడుగు_అడ్డువరుసలు_జరుపు = 2;
            }
        }

        activation_transfer_నిలువు_chunk_state_t నిలువు_state;
        నిలువు_state.dst = dst;
        నిలువు_state.act_dma_addr = act_dma_addr;
        నిలువు_state.n_అడ్డువరుసలు = n_అడ్డువరుసలు;
        నిలువు_state.k_ఖండం = k_ఖండం;
        నిలువు_state.k_stride = k_stride;
        నిలువు_state.k_valid = k_valid;
        నిలువు_state.n_నిలువు_chunks = n_దారాలు;
        నిలువు_state.n_దారాలు_div = *act_దారాలు_div;
        నిలువు_state.vtcm_f32_act = vtcm_f32_act;
        నిలువు_state.vtcm_f32_act_బైట్లు = vtcm_f32_act_బైట్లు;
        నిలువు_state.traces = ctx->trace;
        నిలువు_state.ctx = ctx;
        నిలువు_state.dma_అడుగు_అడ్డువరుసలు = dma_అడుగు_అడ్డువరుసలు;
        నిలువు_state.dma_అడుగు_అడ్డువరుసలు_జరుపు = dma_అడుగు_అడ్డువరుసలు_జరుపు;

        కార్మికుడు_pool_నిరంతరం_func(ctx->కార్మికుడు_pool, transfer_activation_chunk_నిలువు_chunk_కార్మికుడు_fn, &నిలువు_state, n_దారాలు);
        తిరిగి;
    }

    ప్రతిజ్ఞా(k_ఖండం % HTP_MM_HMX_TILE_N_నిలువుల_సంఖ్య == 0 && k_stride % HTP_MM_HMX_TILE_N_నిలువుల_సంఖ్య == 0);

    size_t n_tot_chunks      = n_అడ్డువరుసలు;
    size_t n_chunks_చొప్పున_task = (n_దారాలు == 1) ? n_tot_chunks : 32;  // must be multiple of 32 to ensure correct destination address

    activation_transfer_task_state_t state;
    state.n_tasks            = (n_దారాలు == 1) ? 1 : hmx_ceil_div(n_tot_chunks, 32);
    state.n_tot_chunks       = n_tot_chunks;
    state.n_chunks_చొప్పున_task  = n_chunks_చొప్పున_task;
    state.dst                = dst;
    state.act_dma_addr       = act_dma_addr;
    state.k_ఖండం            = k_ఖండం;
    state.k_stride           = k_stride;
    state.k_valid            = k_valid;
    state.traces             = ctx->trace;
    state.ctx                = ctx;
    state.vtcm_f32_act       = vtcm_f32_act;
    state.act_elem_కొలత = act_elem_కొలత;

    state.vtcm_f32_act_బైట్లు_చొప్పున_thread = hex_align_కిందికి(fastdiv(vtcm_f32_act_బైట్లు, act_దారాలు_div), 128);

    uint32_t dma_అడుగు_అడ్డువరుసలు = 2;
    uint32_t dma_అడుగు_అడ్డువరుసలు_జరుపు = 1;
    ఒకవేళ (state.vtcm_f32_act_బైట్లు_చొప్పున_thread > 0) {
        size_t thread_scratch_elements = state.vtcm_f32_act_బైట్లు_చొప్పున_thread / పరిమాణం(దశాంశం);
        size_t dma_అడుగు_అడ్డువరుసలు_max = fastdiv(thread_scratch_elements / 2, k_div);
        ఒకవేళ (dma_అడుగు_అడ్డువరుసలు_max >= 4) {
            dma_అడుగు_అడ్డువరుసలు = 4;
            dma_అడుగు_అడ్డువరుసలు_జరుపు = 2;
        }
    }
    state.dma_అడుగు_అడ్డువరుసలు      = dma_అడుగు_అడ్డువరుసలు;
    state.dma_అడుగు_అడ్డువరుసలు_జరుపు = dma_అడుగు_అడ్డువరుసలు_జరుపు;

    పూర్ణాంకం చురుకైన_దారాలు = hex_smin(n_దారాలు, (పూర్ణాంకం)state.n_tasks);
    ఒకవేళ (state.n_tasks == 1 || n_దారాలు == 1) {
        transfer_activation_chunk_కార్మికుడు_fn(1, 0, &state);
    } లేకపోతే {
        కార్మికుడు_pool_నిరంతరం_func(ctx->కార్మికుడు_pool, transfer_activation_chunk_కార్మికుడు_fn, &state, చురుకైన_దారాలు);
    }
}
// --- Async HMX matmul job (for pipeline overlap) ---

రకనిర్వచనం నిర్మాణం {
    __fp16 *       output;
    స్థిరరాశి __fp16 * activation;
    స్థిరరాశి __fp16 * బరువు;
    స్థిరరాశి __fp16 * scales;
    uint32_t       n_అడ్డువరుస_tiles;
    uint32_t       n_నిలువు_tiles;
    uint32_t       n_dot_tiles;
} hmx_matmul_పని_t;

స్థిర శూన్య hmx_matmul_కార్మికుడు_fn(శూన్య * దత్తాంశం) {
    hmx_matmul_పని_t * పని = (hmx_matmul_పని_t *) దత్తాంశం;
    FARF(ఎగువ_2, "hmx-mm-job: n_row_tiles %u n_col_tiles %u n_dot_tiles %u", పని->n_అడ్డువరుస_tiles, పని->n_నిలువు_tiles, పని->n_dot_tiles);
    core_dot_chunk_fp16(పని->output, పని->activation, పని->బరువు, పని->scales, పని->n_అడ్డువరుస_tiles, పని->n_నిలువు_tiles, పని->n_dot_tiles);
}

స్థిర అంతఃవరుస శూన్య hmx_matmul_పని_init(hmx_matmul_పని_t * పని,
                                       __fp16 *           output,
                                       స్థిరరాశి __fp16 *     activation,
                                       స్థిరరాశి __fp16 *     బరువు,
                                       స్థిరరాశి __fp16 *     scales,
                                       uint32_t           n_అడ్డువరుస_tiles,
                                       uint32_t           n_నిలువు_tiles,
                                       uint32_t           n_dot_tiles) {
    పని->output      = output;
    పని->activation  = activation;
    పని->బరువు      = బరువు;
    పని->scales      = scales;
    పని->n_అడ్డువరుస_tiles = n_అడ్డువరుస_tiles;
    పని->n_నిలువు_tiles = n_నిలువు_tiles;
    పని->n_dot_tiles = n_dot_tiles;
}

స్థిర పూర్ణాంకం hmx_mm_2d_f32(నిర్మాణం htp_context *ctx,
                                  dma_వరుస *బరువు_dma,
                                  దశాంశం *పరిమిత dst,
                                  dma_addr_t src2_addr,
                                  size_t src2_బైట్లు,
                                  dma_addr_t act_dma_addr,
                                  dma_addr_t బరువు,
                                  పూర్ణాంకం m, పూర్ణాంకం k, పూర్ణాంకం n,
                                  పూర్ణాంకం act_stride,
                                  uint32_t act_elem_కొలత,  // 4=F32 src1, 2=F16 src1
                                  పూర్ణాంకం బరువు_stride,
                                  పూర్ణాంకం బరువు_type,
                                  పూర్ణాంకం k_valid,
                                  పూర్ణాంకం dst_stride,
                                  uint32_t src2_stride,
                                  పూర్ణాంకం dst_నిలువుల_సంఖ్య,
                                  పూర్ణాంకం m_chunk,
                                  పూర్ణాంకం n_chunk,
                                  పూర్ణాంకం pipeline,
                                  పూర్ణాంకం n_దారాలు,
                                  పూర్ణాంకం act_దారాలు,
                                  స్థిరరాశి నిర్మాణం fastdiv_విలువలు * act_దారాలు_div,
                                  స్థిరరాశి నిర్మాణం fastdiv_విలువలు * k_div,
                                  పూర్ణాంకం tile_కొలత,
                                  పూర్ణాంకం aligned_tile_కొలత,
                                  పూర్ణాంకం vtcm_కొలత) {
    నిర్మాణం htp_thread_trace * tr = &ctx->trace[0];
    htp_trace_event_ప్రారంభం(tr, HTP_TRACE_EVT_INIT, 0);

    స్థిరరాశి bool is_tiled = (tile_కొలత > 0);

    ఒకవేళ (k % 32 != 0) {
        తిరిగి -1;
    }
    ఒకవేళ (!hex_is_aligned(dst, పరిమాణం(దశాంశం))) { తిరిగి -1; }

    size_t అడ్డువరుస_stride = htp_tensor_get_అడ్డువరుస_కొలత(బరువు_type, k);
    ఒకవేళ (అడ్డువరుస_stride == 0) {
        తిరిగి -1;
    }

    tiled_dequant_task_fn_t dequant_కార్మికుడు_fn = hmx_get_dequant_కార్మికుడు_fn(బరువు_type, is_tiled);

    స్థిరరాశి పూర్ణాంకం n_k_tiles = k / HTP_MM_HMX_TILE_N_నిలువుల_సంఖ్య;
    స్థిరరాశి పూర్ణాంకం n_బరువు_tiles = k / htp_tensor_get_tile_k(బరువు_type);
    స్థిరరాశి నిర్మాణం fastdiv_విలువలు n_k_tiles_div = init_fastdiv_విలువలు(n_k_tiles);

    స్థిరరాశి size_t vec_dot_కొలత = k * పరిమాణం(__fp16);
    స్థిరరాశి size_t vtcm_budget  = ctx->vtcm_కొలత;

    స్థిరరాశి uint32_t dma_dst_stride  = is_tiled ? aligned_tile_కొలత : అడ్డువరుస_stride;
    స్థిరరాశి uint32_t dma_src_stride  = is_tiled ? tile_కొలత : బరువు_stride;
    స్థిరరాశి uint32_t dma_వెడల్పు_బైట్లు = is_tiled ? tile_కొలత : అడ్డువరుస_stride;

    size_t m_chunk_n_అడ్డువరుసలు = m_chunk;
    size_t n_chunk_n_నిలువుల_సంఖ్య = n_chunk;
    size_t vtcm_వాడిన      = vtcm_కొలత;

    నిర్మాణం htp_mm_hmx_vtcm_layout L;
    htp_mm_hmx_vtcm_layout_build(&L, HTP_MM_KERNEL_HMX_2D, బరువు_type, k, m_chunk_n_అడ్డువరుసలు, n_chunk_n_నిలువుల_సంఖ్య, 1, pipeline, act_దారాలు, aligned_tile_కొలత, src2_బైట్లు);

    vtcm_వాడిన = L.మొత్తం_బైట్లు;
    ఒకవేళ (vtcm_వాడిన > vtcm_budget) {
        FARF(ERROR, "hmx-mm-2d-precomputed: VTCM overflow: used %zu budget %zu, m %d k %d n %d mc %zu nc %zu",
             vtcm_వాడిన, vtcm_budget, m, k, n, m_chunk_n_అడ్డువరుసలు, n_chunk_n_నిలువుల_సంఖ్య);
        తిరిగి -1;
    }

    uint8_t * స్థిరరాశి పునాది = (uint8_t *) ctx->vtcm_పునాది;
    __fp16  *vtcm_బరువు_raw[2] = {
        VTCM_LAYOUT_PTR(__fp16, పునాది, L.off_బరువు[0]),
        VTCM_LAYOUT_PTR_OPTIONAL(__fp16, పునాది, L.off_బరువు[1], pipeline)
    };
    __fp16  *vtcm_f16_act    = VTCM_LAYOUT_PTR(__fp16, పునాది, L.off_act);
    దశాంశం   *vtcm_f32_act    = VTCM_LAYOUT_PTR(దశాంశం, పునాది, L.off_act_f32);
    __fp16  *vtcm_output     = VTCM_LAYOUT_PTR(__fp16, పునాది, L.off_dst[0]);
    శూన్య    *vtcm_scratch0   = VTCM_LAYOUT_PTR(శూన్య, పునాది, L.off_scratch[0]);
    శూన్య    *vtcm_scratch1   = VTCM_LAYOUT_PTR_OPTIONAL(శూన్య, పునాది, L.off_scratch[1], pipeline);
    శూన్య    *vtcm_scratch2   = VTCM_LAYOUT_PTR_OPTIONAL(శూన్య, పునాది, L.off_dst[1], pipeline);
    __fp16  *vtcm_scales     = VTCM_LAYOUT_PTR(__fp16, పునాది, L.off_scales);

    hmx_init_స్తంభం_scales(vtcm_scales, Q6_V_vsplat_R(0x3c00));  // scale: 1.0, bias: 0.0 in FP16

    స్థిరరాశి bool has_bias      = (src2_బైట్లు > 0 && src2_addr != 0);
    దశాంశం   *vtcm_bias       = VTCM_LAYOUT_PTR_OPTIONAL(దశాంశం, పునాది, L.off_bias, has_bias);
    ఒకవేళ (has_bias) {
        dma_వరుస_నెట్టు(బరువు_dma, dma_make_దత్తాంశం(vtcm_bias, src2_addr), hex_align_పైకి(src2_బైట్లు, 128), 0, src2_బైట్లు, 1);
        dma_వరుస_తీయి(బరువు_dma);
    }

    FARF(ఎగువ_2, "hmx-mm-2d: m %d k %d n %d wtype %d mc %zu nc %zu vtcm %zu/%zu",
         m, k, n, బరువు_type, m_chunk_n_అడ్డువరుసలు, n_chunk_n_నిలువుల_సంఖ్య, vtcm_వాడిన, vtcm_budget);

    పూర్ణాంకం n_chunk_cnt = hmx_ceil_div(n, n_chunk_n_నిలువుల_సంఖ్య);

    htp_trace_event_stop(tr, HTP_TRACE_EVT_INIT, 0);

    ఒకవేళ (pipeline) {
        // --- Asynchronous Pipelined Loop ---
        hmx_matmul_పని_t పని_slots[2];  // persistent double-buffered job descriptors
        శూన్య *vtcm_raw_bufs[4] = {
            vtcm_బరువు_raw[0],
            vtcm_బరువు_raw[1],
            dequant_కార్మికుడు_fn ? vtcm_బరువు_raw[0] : vtcm_scratch0,
            dequant_కార్మికుడు_fn ? vtcm_బరువు_raw[1] : vtcm_scratch1,
        };
        శూన్య *vtcm_బరువు_bufs[4] = {
            dequant_కార్మికుడు_fn ? vtcm_scratch0 : vtcm_బరువు_raw[0],
            dequant_కార్మికుడు_fn ? vtcm_scratch1 : vtcm_బరువు_raw[1],
            vtcm_scratch0,
            vtcm_scratch1,
        };
        శూన్య *vtcm_output_bufs[2] = { vtcm_output, vtcm_scratch2 };

        కొరకు (size_t mr = 0; mr < m; mr += m_chunk_n_అడ్డువరుసలు) {
            స్థిరరాశి size_t n_అడ్డువరుసలు = hex_smin(m - mr, m_chunk_n_అడ్డువరుసలు);

            నిర్మాణం activation_transfer_params act_params = {
                .ctx = ctx,
                .dst = vtcm_f16_act,
                .act_dma_addr = act_dma_addr + (size_t) mr * act_stride * act_elem_కొలత,  // byte offset (F16/F32)
                .n_అడ్డువరుసలు = (పూర్ణాంకం) n_అడ్డువరుసలు,
                .k_ఖండం = k,
                .k_stride = act_stride,
                .n_దారాలు = act_దారాలు,
                .act_దారాలు_div = act_దారాలు_div,
                .k_div = k_div,
                .k_valid = k_valid,
                .vtcm_f32_act = vtcm_f32_act,
                .vtcm_f32_act_బైట్లు = L.act_f32_బైట్లు,
                .act_elem_కొలత = act_elem_కొలత,
            };
            transfer_activation_chunk_threaded(&act_params);

            // Prologue: push A0 and optionally A1 (if n_chunk_cnt > 1)
            స్థిరరాశి size_t   n_నిలువుల_సంఖ్య_A0 = hex_smin(n - 0 * n_chunk_n_నిలువుల_సంఖ్య, n_chunk_n_నిలువుల_సంఖ్య);
            స్థిరరాశి uint32_t ఎత్తు_A0 = is_tiled ? hmx_ceil_div(n_నిలువుల_సంఖ్య_A0, 32) * n_బరువు_tiles : n_నిలువుల_సంఖ్య_A0;
            dma_వరుస_నెట్టు(బరువు_dma, dma_make_దత్తాంశం(vtcm_raw_bufs[0], బరువు),
                           dma_dst_stride, dma_src_stride, dma_వెడల్పు_బైట్లు, ఎత్తు_A0);

            ఒకవేళ (1 < n_chunk_cnt) {
                స్థిరరాశి size_t   n_నిలువుల_సంఖ్య_A1 = hex_smin(n - 1 * n_chunk_n_నిలువుల_సంఖ్య, n_chunk_n_నిలువుల_సంఖ్య);
                స్థిరరాశి uint32_t ఎత్తు_A1 = is_tiled ? hmx_ceil_div(n_నిలువుల_సంఖ్య_A1, 32) * n_బరువు_tiles : n_నిలువుల_సంఖ్య_A1;
                dma_వరుస_నెట్టు(బరువు_dma, dma_make_దత్తాంశం(vtcm_raw_bufs[1], బరువు + n_chunk_n_నిలువుల_సంఖ్య * బరువు_stride),
                               dma_dst_stride, dma_src_stride, dma_వెడల్పు_బైట్లు, ఎత్తు_A1);
            }

            // Main loop: pop A_i -> dequantize A_i -> push A_{i+2} -> submit C_i -> wait C_{i-1} and store D_{i-1}
            కొరకు (పూర్ణాంకం i = 0; i < n_chunk_cnt; ++i) {
                స్థిరరాశి size_t nc    = i * n_chunk_n_నిలువుల_సంఖ్య;
                స్థిరరాశి size_t nc_p2 = nc + 2 * n_chunk_n_నిలువుల_సంఖ్య;

                స్థిరరాశి size_t n_నిలువుల_సంఖ్య    = hex_smin(n - nc, n_chunk_n_నిలువుల_సంఖ్య);
                స్థిరరాశి size_t n_నిలువుల_సంఖ్య_p2 = hex_smin(n - nc_p2, n_chunk_n_నిలువుల_సంఖ్య);

                // 1. pop A_i
                శూన్య * curr_raw = (శూన్య *) dma_వరుస_తీయి(బరువు_dma).dst;

                // 2. dequantize A_i
                dequantize_tiled_బరువు_chunk_గమ్యం_fp16_tiles(
                    ctx, vtcm_బరువు_bufs[i % 4], curr_raw,
                    n_నిలువుల_సంఖ్య, k, అడ్డువరుస_stride, బరువు_type,
                    n_k_tiles, n_k_tiles_div, dequant_కార్మికుడు_fn, n_దారాలు);

                // 3. push A_{i+2} (if i+2 < n_chunk_cnt)
                ఒకవేళ (i + 2 < n_chunk_cnt) {
                    స్థిరరాశి uint32_t ఎత్తు_p2 = is_tiled ? hmx_ceil_div(n_నిలువుల_సంఖ్య_p2, 32) * n_బరువు_tiles : n_నిలువుల_సంఖ్య_p2;
                    dma_వరుస_నెట్టు(బరువు_dma, dma_make_దత్తాంశం(vtcm_raw_bufs[(i + 2) % 4], బరువు + nc_p2 * బరువు_stride),
                                   dma_dst_stride, dma_src_stride, dma_వెడల్పు_బైట్లు, ఎత్తు_p2);
                }

                // 4. submit C_i
                hmx_matmul_పని_init(&పని_slots[i % 2], (__fp16 *) vtcm_output_bufs[i % 2],
                                    (__fp16 *) vtcm_f16_act, (__fp16 *) vtcm_బరువు_bufs[i % 4],
                                    vtcm_scales, hmx_ceil_div(n_అడ్డువరుసలు, HTP_MM_HMX_TILE_N_అడ్డువరుసలు),
                                    hmx_ceil_div(n_నిలువుల_సంఖ్య, HTP_MM_HMX_TILE_N_నిలువుల_సంఖ్య), k / HTP_MM_HMX_TILE_N_అడ్డువరుసలు);
                hmx_వరుస_నెట్టు(ctx->hmx_వరుస, hmx_వరుస_make_desc(hmx_matmul_కార్మికుడు_fn, &పని_slots[i % 2]));

                // 5. wait C_{i-1} and store D_{i-1} (multi-thread HVX, parallel with C_i)
                ఒకవేళ (i > 0) {
                    hmx_వరుస_తీయి(ctx->hmx_వరుస);
                    స్థిరరాశి size_t nc_prev = (i - 1) * n_chunk_n_నిలువుల_సంఖ్య;
                    స్థిరరాశి size_t n_నిలువుల_సంఖ్య_prev = hex_smin(n - nc_prev, n_chunk_n_నిలువుల_సంఖ్య);
                    దశాంశం *output_chunk = dst + (mr * dst_stride + nc_prev);
                    స్థిరరాశి దశాంశం *bias_chunk = has_bias ? (vtcm_bias + mr * src2_stride + nc_prev) : ఖాళీ;
                    పూర్ణాంకం chunk_dst_నిలువుల_సంఖ్య = dst_నిలువుల_సంఖ్య - (పూర్ణాంకం)nc_prev;
                    ఒకవేళ (chunk_dst_నిలువుల_సంఖ్య > 0) {
                        transfer_output_chunk_threaded(ctx, output_chunk, bias_chunk, vtcm_output_bufs[(i - 1) % 2], n_అడ్డువరుసలు, n_నిలువుల_సంఖ్య_prev, dst_stride, src2_stride, chunk_dst_నిలువుల_సంఖ్య, n_దారాలు);
                    }
                }
            }

            // Epilogue: wait C_{last} and store D_{last}
            hmx_వరుస_తీయి(ctx->hmx_వరుస);
            స్థిరరాశి size_t nc_చివరి = (n_chunk_cnt - 1) * n_chunk_n_నిలువుల_సంఖ్య;
            స్థిరరాశి size_t n_నిలువుల_సంఖ్య_చివరి = hex_smin(n - nc_చివరి, n_chunk_n_నిలువుల_సంఖ్య);
            దశాంశం *output_chunk = dst + (mr * dst_stride + nc_చివరి);
            స్థిరరాశి దశాంశం *bias_chunk = has_bias ? (vtcm_bias + mr * src2_stride + nc_చివరి) : ఖాళీ;
            పూర్ణాంకం chunk_dst_నిలువుల_సంఖ్య = dst_నిలువుల_సంఖ్య - (పూర్ణాంకం)nc_చివరి;
            ఒకవేళ (chunk_dst_నిలువుల_సంఖ్య > 0) {
                transfer_output_chunk_threaded(ctx, output_chunk, bias_chunk, vtcm_output_bufs[(n_chunk_cnt - 1) % 2], n_అడ్డువరుసలు, n_నిలువుల_సంఖ్య_చివరి, dst_stride, src2_stride, chunk_dst_నిలువుల_సంఖ్య, n_దారాలు);
            }
        }
    } లేకపోతే {
        // --- Synchronous loop (m <= 32 or fallback) ---
        hmx_matmul_పని_t పని;
        శూన్య *vtcm_raw_bufs[2] = { vtcm_బరువు_raw[0], dequant_కార్మికుడు_fn ? vtcm_బరువు_raw[0] : vtcm_scratch0 };
        శూన్య *vtcm_బరువు_bufs[2] = { dequant_కార్మికుడు_fn ? vtcm_scratch0 : vtcm_బరువు_raw[0], vtcm_scratch0 };

        కొరకు (size_t mr = 0; mr < m; mr += m_chunk_n_అడ్డువరుసలు) {
            స్థిరరాశి size_t n_అడ్డువరుసలు = hex_smin(m - mr, m_chunk_n_అడ్డువరుసలు);

            నిర్మాణం activation_transfer_params act_params = {
                .ctx = ctx,
                .dst = vtcm_f16_act,
                .act_dma_addr = act_dma_addr + (size_t) mr * act_stride * act_elem_కొలత,  // byte offset (F16/F32)
                .n_అడ్డువరుసలు = (పూర్ణాంకం) n_అడ్డువరుసలు,
                .k_ఖండం = k,
                .k_stride = act_stride,
                .n_దారాలు = act_దారాలు,
                .act_దారాలు_div = act_దారాలు_div,
                .k_div = k_div,
                .k_valid = k_valid,
                .vtcm_f32_act = vtcm_f32_act,
                .vtcm_f32_act_బైట్లు = L.act_f32_బైట్లు,
                .act_elem_కొలత = act_elem_కొలత,
            };
            transfer_activation_chunk_threaded(&act_params);

            // A0: Pre-fetch the first weight chunk (nc = 0)
            ఒకవేళ (n > 0) {
                స్థిరరాశి size_t n_నిలువుల_సంఖ్య = hex_smin(n, n_chunk_n_నిలువుల_సంఖ్య);
                స్థిరరాశి uint32_t ఎత్తు = is_tiled ? hmx_ceil_div(n_నిలువుల_సంఖ్య, 32) * n_బరువు_tiles : n_నిలువుల_సంఖ్య;
                dma_వరుస_నెట్టు(బరువు_dma, dma_make_దత్తాంశం(vtcm_raw_bufs[0], బరువు), dma_dst_stride, dma_src_stride, dma_వెడల్పు_బైట్లు, ఎత్తు);
            }

            కొరకు (పూర్ణాంకం i = 0; i < n_chunk_cnt; ++i) {
                స్థిరరాశి size_t nc = (size_t) i * n_chunk_n_నిలువుల_సంఖ్య;
                స్థిరరాశి size_t n_నిలువుల_సంఖ్య = hex_smin(n - nc, n_chunk_n_నిలువుల_సంఖ్య);
                స్థిరరాశి size_t n_అడ్డువరుస_tiles = hmx_ceil_div(n_అడ్డువరుసలు, HTP_MM_HMX_TILE_N_అడ్డువరుసలు);
                స్థిరరాశి size_t n_నిలువు_tiles = hmx_ceil_div(n_నిలువుల_సంఖ్య, HTP_MM_HMX_TILE_N_నిలువుల_సంఖ్య);

                // A: Wait for weight DMA
                శూన్య * curr_raw = (శూన్య *) dma_వరుస_తీయి(బరువు_dma).dst;

                // B: Weight Dequantize (Threaded)
                dequantize_tiled_బరువు_chunk_గమ్యం_fp16_tiles(
                    ctx, vtcm_బరువు_bufs[i % 2], curr_raw,
                    n_నిలువుల_సంఖ్య, k, అడ్డువరుస_stride, బరువు_type,
                    n_k_tiles, n_k_tiles_div, dequant_కార్మికుడు_fn, n_దారాలు);

                // Start weight DMA for the next chunk early
                స్థిరరాశి size_t nc_తదుపరి = nc + n_chunk_n_నిలువుల_సంఖ్య;
                ఒకవేళ (nc_తదుపరి < n) {
                    స్థిరరాశి size_t n_నిలువుల_సంఖ్య_తదుపరి = hex_smin(n - nc_తదుపరి, n_chunk_n_నిలువుల_సంఖ్య);
                    స్థిరరాశి uint32_t ఎత్తు_తదుపరి = is_tiled ? hmx_ceil_div(n_నిలువుల_సంఖ్య_తదుపరి, 32) * n_బరువు_tiles : n_నిలువుల_సంఖ్య_తదుపరి;
                    dma_వరుస_నెట్టు(బరువు_dma, dma_make_దత్తాంశం(vtcm_raw_bufs[(i + 1) % 2], బరువు + nc_తదుపరి * బరువు_stride), dma_dst_stride, dma_src_stride, dma_వెడల్పు_బైట్లు, ఎత్తు_తదుపరి);
                }

                // C: HMX Compute (Queue-based)
                hmx_matmul_పని_init(&పని, vtcm_output, vtcm_f16_act, (__fp16 *) vtcm_బరువు_bufs[i % 2], vtcm_scales, n_అడ్డువరుస_tiles, n_నిలువు_tiles, k / HTP_MM_HMX_TILE_N_అడ్డువరుసలు);
                hmx_వరుస_నెట్టు(ctx->hmx_వరుస, hmx_వరుస_make_desc(hmx_matmul_కార్మికుడు_fn, &పని));
                hmx_వరుస_తీయి(ctx->hmx_వరుస);

                // D: Output Store
                దశాంశం *output_chunk = dst + (mr * dst_stride + nc);
                స్థిరరాశి దశాంశం *bias_chunk = has_bias ? (vtcm_bias + mr * src2_stride + nc) : ఖాళీ;
                పూర్ణాంకం chunk_dst_నిలువుల_సంఖ్య = dst_నిలువుల_సంఖ్య - (పూర్ణాంకం)nc;
                ఒకవేళ (chunk_dst_నిలువుల_సంఖ్య > 0) {
                    transfer_output_chunk_threaded(ctx, output_chunk, bias_chunk, vtcm_output, n_అడ్డువరుసలు, n_నిలువుల_సంఖ్య, dst_stride, src2_stride, chunk_dst_నిలువుల_సంఖ్య, n_దారాలు);
                }
            }
        }
    }

    తిరిగి 0;
}

స్థిర పూర్ణాంకం hmx_mm_nx_2d_f32(నిర్మాణం htp_ops_context * octx, స్థిరరాశి నిర్మాణం htp_mm_kernel_params * kparams) {
    నిర్మాణం htp_context * ctx = octx->ctx;
    నిర్మాణం htp_thread_trace * tr = &ctx->trace[0];
    htp_trace_event_ప్రారంభం(tr, HTP_TRACE_EVT_INIT, 0);

    స్థిరరాశి uint32_t n_weights = kparams->n_weights;
    ఒకవేళ (n_weights == 0 || n_weights > HTP_OP_MAX_OUTPUTS) {
        తిరిగి HTP_స్థితి_INVAL_PARAMS;
    }

    స్థిరరాశి నిర్మాణం htp_tensor * పరిమిత src0 = octx->src[0];
    స్థిరరాశి నిర్మాణం htp_tensor * పరిమిత act  = octx->src[n_weights];

    స్థిరరాశి పూర్ణాంకం బరువు_type = (పూర్ణాంకం) src0->type;
    స్థిరరాశి పూర్ణాంకం k           = (పూర్ణాంకం) act->ne[0];
    స్థిరరాశి పూర్ణాంకం k_valid     = (పూర్ణాంకం) act->ne[0];
    స్థిరరాశి పూర్ణాంకం m           = (పూర్ణాంకం) (act->ne[1] * act->ne[2] * act->ne[3]);
    స్థిరరాశి uint32_t act_elem_కొలత = (act->type == HTP_TYPE_F16) ? పరిమాణం(__fp16) : పరిమాణం(దశాంశం);
    స్థిరరాశి పూర్ణాంకం act_stride  = (పూర్ణాంకం) (act->nb[1] / act_elem_కొలత);
    స్థిరరాశి dma_addr_t act_dma_addr = act->దత్తాంశం;

    ఒకవేళ (k % 32 != 0) { తిరిగి HTP_స్థితి_NO_SUPPORT; }

    size_t అడ్డువరుస_stride = htp_tensor_get_అడ్డువరుస_కొలత(బరువు_type, k);
    ఒకవేళ (అడ్డువరుస_stride == 0) {
        తిరిగి HTP_స్థితి_NO_SUPPORT;
    }

    స్థిరరాశి పూర్ణాంకం n_k_tiles = k / HTP_MM_HMX_TILE_N_నిలువుల_సంఖ్య;
    స్థిరరాశి పూర్ణాంకం n_బరువు_tiles = k / htp_tensor_get_tile_k(బరువు_type);
    స్థిరరాశి నిర్మాణం fastdiv_విలువలు n_k_tiles_div = init_fastdiv_విలువలు(n_k_tiles);

    స్థిరరాశి పూర్ణాంకం tile_కొలత       = kparams->tile_కొలత;
    స్థిరరాశి పూర్ణాంకం aligned_tile_కొలత = kparams->aligned_tile_కొలత;

    స్థిరరాశి bool is_tiled       = (tile_కొలత > 0);
    tiled_dequant_task_fn_t dequant_కార్మికుడు_fn = hmx_get_dequant_కార్మికుడు_fn(బరువు_type, is_tiled);
    స్థిరరాశి size_t vtcm_budget  = ctx->vtcm_కొలత;

    స్థిరరాశి పూర్ణాంకం m_chunk_n_అడ్డువరుసలు  = kparams->m_chunk;
    స్థిరరాశి పూర్ణాంకం n_chunk_n_నిలువుల_సంఖ్య  = kparams->n_chunk;
    స్థిరరాశి పూర్ణాంకం pipeline        = kparams->pipeline;
    స్థిరరాశి పూర్ణాంకం n_దారాలు       = octx->n_దారాలు;
    స్థిరరాశి పూర్ణాంకం act_దారాలు     = kparams->n_act_దారాలు;
    స్థిరరాశి నిర్మాణం fastdiv_విలువలు * act_దారాలు_div = &kparams->div_n_act_దారాలు;
    స్థిరరాశి నిర్మాణం fastdiv_విలువలు * k_div           = &kparams->div_ne00_padded;

    స్థిరరాశి uint32_t dma_dst_stride  = is_tiled ? aligned_tile_కొలత : అడ్డువరుస_stride;
    స్థిరరాశి uint32_t dma_వెడల్పు_బైట్లు = is_tiled ? tile_కొలత : అడ్డువరుస_stride;

    నిర్మాణం htp_mm_hmx_vtcm_layout L;
    htp_mm_hmx_vtcm_layout_build(&L, HTP_MM_KERNEL_HMX_2D, బరువు_type, k, m_chunk_n_అడ్డువరుసలు, n_chunk_n_నిలువుల_సంఖ్య, 1, pipeline, act_దారాలు, aligned_tile_కొలత, 0);

    ఒకవేళ (L.మొత్తం_బైట్లు > vtcm_budget) {
        FARF(ERROR, "hmx-mm-nx-2d: VTCM overflow: used %zu budget %zu, m %d k %d mc %d nc %d",
             L.మొత్తం_బైట్లు, vtcm_budget, m, k, m_chunk_n_అడ్డువరుసలు, n_chunk_n_నిలువుల_సంఖ్య);
        తిరిగి HTP_స్థితి_VTCM_TOO_SMALL;
    }

    uint8_t * స్థిరరాశి పునాది = (uint8_t *) ctx->vtcm_పునాది;
    __fp16  *vtcm_బరువు_raw[2] = {
        VTCM_LAYOUT_PTR(__fp16, పునాది, L.off_బరువు[0]),
        VTCM_LAYOUT_PTR_OPTIONAL(__fp16, పునాది, L.off_బరువు[1], pipeline)
    };
    __fp16  *vtcm_f16_act    = VTCM_LAYOUT_PTR(__fp16, పునాది, L.off_act);
    దశాంశం   *vtcm_f32_act    = VTCM_LAYOUT_PTR(దశాంశం, పునాది, L.off_act_f32);
    __fp16  *vtcm_output     = VTCM_LAYOUT_PTR(__fp16, పునాది, L.off_dst[0]);
    శూన్య    *vtcm_scratch0   = VTCM_LAYOUT_PTR(శూన్య, పునాది, L.off_scratch[0]);
    శూన్య    *vtcm_scratch1   = VTCM_LAYOUT_PTR_OPTIONAL(శూన్య, పునాది, L.off_scratch[1], pipeline);
    శూన్య    *vtcm_scratch2   = VTCM_LAYOUT_PTR_OPTIONAL(శూన్య, పునాది, L.off_dst[1], pipeline);
    __fp16  *vtcm_scales     = VTCM_LAYOUT_PTR(__fp16, పునాది, L.off_scales);

    hmx_init_స్తంభం_scales(vtcm_scales, Q6_V_vsplat_R(0x3c00));  // scale: 1.0, bias: 0.0 in FP16

    పూర్ణాంకం m_ప్రారంభం = 0;
    పూర్ణాంకం m_అడ్డువరుసలు  = m;
    ఒకవేళ (octx->ctx->mdev.లెక్క > 1) {
        bool can_విభజించు = htp_tensor_can_అడ్డువరుస_విభజన(act, act_elem_కొలత);
        ఒకవేళ (can_విభజించు) {
            కొరకు (uint32_t p = 0; p < n_weights; ++p) {
                స్థిరరాశి నిర్మాణం htp_tensor * పరిమిత dst = octx->dsts[p];
                ఒకవేళ (!htp_tensor_can_అడ్డువరుస_విభజన(dst, పరిమాణం(దశాంశం))) {
                    can_విభజించు = false;
                    ఆపు;
                }
            }
        }
        స్థిరరాశి నిర్మాణం htp_tensor_mdev_range range = htp_tensor_mdev_విభజన((uint32_t) m, can_విభజించు ? 1 : 0, octx->ctx->mdev.idx, octx->ctx->mdev.లెక్క, &octx->ctx->mdev.లెక్క_div);
        m_ప్రారంభం = (పూర్ణాంకం) range.ప్రారంభం;
        m_అడ్డువరుసలు  = (పూర్ణాంకం) range.లెక్క;
    }

    ఒకవేళ (m_అడ్డువరుసలు == 0) {
        తిరిగి HTP_స్థితి_OK;
    }

    FARF(ఎగువ_2, "hmx-mm-nx-2d: n_weights %u m %d (%d..%d) k %d wtype %d mc %d nc %d vtcm %zu/%zu",
         n_weights, m, m_ప్రారంభం, m_ప్రారంభం + m_అడ్డువరుసలు, k, బరువు_type, m_chunk_n_అడ్డువరుసలు, n_chunk_n_నిలువుల_సంఖ్య, L.మొత్తం_బైట్లు, vtcm_budget);

    htp_trace_event_stop(tr, HTP_TRACE_EVT_INIT, 0);

    స్థిరరాశి size_t mr_చివర = (size_t)(m_ప్రారంభం + m_అడ్డువరుసలు);

    ఒకవేళ (pipeline) {
        hmx_matmul_పని_t పని_slots[2];
        శూన్య *vtcm_raw_bufs[4] = {
            vtcm_బరువు_raw[0],
            vtcm_బరువు_raw[1],
            dequant_కార్మికుడు_fn ? vtcm_బరువు_raw[0] : vtcm_scratch0,
            dequant_కార్మికుడు_fn ? vtcm_బరువు_raw[1] : vtcm_scratch1,
        };
        శూన్య *vtcm_బరువు_bufs[4] = {
            dequant_కార్మికుడు_fn ? vtcm_scratch0 : vtcm_బరువు_raw[0],
            dequant_కార్మికుడు_fn ? vtcm_scratch1 : vtcm_బరువు_raw[1],
            vtcm_scratch0,
            vtcm_scratch1,
        };
        శూన్య *vtcm_output_bufs[2] = { vtcm_output, vtcm_scratch2 };

        కొరకు (size_t mr = (size_t) m_ప్రారంభం; mr < mr_చివర; mr += m_chunk_n_అడ్డువరుసలు) {
            స్థిరరాశి size_t n_అడ్డువరుసలు = hex_smin(mr_చివర - mr, m_chunk_n_అడ్డువరుసలు);

            నిర్మాణం activation_transfer_params act_params = {
                .ctx = ctx,
                .dst = vtcm_f16_act,
                .act_dma_addr = act_dma_addr + (size_t) mr * act_stride * act_elem_కొలత,
                .n_అడ్డువరుసలు = (పూర్ణాంకం) n_అడ్డువరుసలు,
                .k_ఖండం = k,
                .k_stride = act_stride,
                .n_దారాలు = act_దారాలు,
                .act_దారాలు_div = act_దారాలు_div,
                .k_div = k_div,
                .k_valid = k_valid,
                .vtcm_f32_act = vtcm_f32_act,
                .vtcm_f32_act_బైట్లు = L.act_f32_బైట్లు,
                .act_elem_కొలత = act_elem_కొలత,
            };
            transfer_activation_chunk_threaded(&act_params);

            కొరకు (uint32_t p = 0; p < n_weights; p++) {
                స్థిరరాశి నిర్మాణం htp_tensor * పరిమిత src_w = octx->src[p];
                స్థిరరాశి నిర్మాణం htp_tensor * పరిమిత dst   = octx->dsts[p];
                ఒకవేళ (!src_w || !dst) కొనసాగు;

                స్థిరరాశి dma_addr_t బరువు       = src_w->దత్తాంశం;
                dma_వరుస * బరువు_dma      = octx->ctx->dma[0];
                దశాంశం * dst_ptr            = (దశాంశం *) dst->దత్తాంశం;
                స్థిరరాశి size_t n             = src_w->ne[1];
                ఒకవేళ (n == 0) కొనసాగు;
                స్థిరరాశి size_t బరువు_stride = src_w->nb[1];
                స్థిరరాశి size_t dst_stride    = dst->nb[1] / పరిమాణం(దశాంశం);
                స్థిరరాశి పూర్ణాంకం dst_నిలువుల_సంఖ్య         = (పూర్ణాంకం) dst->ne[0];
                స్థిరరాశి పూర్ణాంకం n_chunk_cnt      = hmx_ceil_div(n, n_chunk_n_నిలువుల_సంఖ్య);

                స్థిరరాశి uint32_t dma_src_stride = is_tiled ? tile_కొలత : బరువు_stride;

                స్థిరరాశి size_t   n_నిలువుల_సంఖ్య_A0 = hex_smin(n - 0 * n_chunk_n_నిలువుల_సంఖ్య, n_chunk_n_నిలువుల_సంఖ్య);
                స్థిరరాశి uint32_t ఎత్తు_A0 = is_tiled ? hmx_ceil_div(n_నిలువుల_సంఖ్య_A0, 32) * n_బరువు_tiles : n_నిలువుల_సంఖ్య_A0;
                dma_వరుస_నెట్టు(బరువు_dma, dma_make_దత్తాంశం(vtcm_raw_bufs[0], బరువు),
                               dma_dst_stride, dma_src_stride, dma_వెడల్పు_బైట్లు, ఎత్తు_A0);

                ఒకవేళ (1 < n_chunk_cnt) {
                    స్థిరరాశి size_t   n_నిలువుల_సంఖ్య_A1 = hex_smin(n - 1 * n_chunk_n_నిలువుల_సంఖ్య, n_chunk_n_నిలువుల_సంఖ్య);
                    స్థిరరాశి uint32_t ఎత్తు_A1 = is_tiled ? hmx_ceil_div(n_నిలువుల_సంఖ్య_A1, 32) * n_బరువు_tiles : n_నిలువుల_సంఖ్య_A1;
                    dma_వరుస_నెట్టు(బరువు_dma, dma_make_దత్తాంశం(vtcm_raw_bufs[1], బరువు + n_chunk_n_నిలువుల_సంఖ్య * బరువు_stride),
                                   dma_dst_stride, dma_src_stride, dma_వెడల్పు_బైట్లు, ఎత్తు_A1);
                }

                కొరకు (పూర్ణాంకం i = 0; i < n_chunk_cnt; ++i) {
                    స్థిరరాశి size_t nc    = i * n_chunk_n_నిలువుల_సంఖ్య;
                    స్థిరరాశి size_t nc_p2 = nc + 2 * n_chunk_n_నిలువుల_సంఖ్య;

                    స్థిరరాశి size_t n_నిలువుల_సంఖ్య    = hex_smin(n - nc, n_chunk_n_నిలువుల_సంఖ్య);
                    స్థిరరాశి size_t n_నిలువుల_సంఖ్య_p2 = hex_smin(n - nc_p2, n_chunk_n_నిలువుల_సంఖ్య);

                    శూన్య * curr_raw = (శూన్య *) dma_వరుస_తీయి(బరువు_dma).dst;

                    dequantize_tiled_బరువు_chunk_గమ్యం_fp16_tiles(
                        ctx, vtcm_బరువు_bufs[i % 4], curr_raw,
                        n_నిలువుల_సంఖ్య, k, అడ్డువరుస_stride, బరువు_type,
                        n_k_tiles, n_k_tiles_div, dequant_కార్మికుడు_fn, n_దారాలు);

                    ఒకవేళ (i + 2 < n_chunk_cnt) {
                        స్థిరరాశి uint32_t ఎత్తు_p2 = is_tiled ? hmx_ceil_div(n_నిలువుల_సంఖ్య_p2, 32) * n_బరువు_tiles : n_నిలువుల_సంఖ్య_p2;
                        dma_వరుస_నెట్టు(బరువు_dma, dma_make_దత్తాంశం(vtcm_raw_bufs[(i + 2) % 4], బరువు + nc_p2 * బరువు_stride),
                                       dma_dst_stride, dma_src_stride, dma_వెడల్పు_బైట్లు, ఎత్తు_p2);
                    }

                    hmx_matmul_పని_init(&పని_slots[i % 2], (__fp16 *) vtcm_output_bufs[i % 2],
                                        (__fp16 *) vtcm_f16_act, (__fp16 *) vtcm_బరువు_bufs[i % 4],
                                        vtcm_scales, hmx_ceil_div(n_అడ్డువరుసలు, HTP_MM_HMX_TILE_N_అడ్డువరుసలు),
                                        hmx_ceil_div(n_నిలువుల_సంఖ్య, HTP_MM_HMX_TILE_N_నిలువుల_సంఖ్య), k / HTP_MM_HMX_TILE_N_అడ్డువరుసలు);
                    hmx_వరుస_నెట్టు(ctx->hmx_వరుస, hmx_వరుస_make_desc(hmx_matmul_కార్మికుడు_fn, &పని_slots[i % 2]));

                    ఒకవేళ (i > 0) {
                        hmx_వరుస_తీయి(ctx->hmx_వరుస);
                        స్థిరరాశి size_t nc_prev = (i - 1) * n_chunk_n_నిలువుల_సంఖ్య;
                        స్థిరరాశి size_t n_నిలువుల_సంఖ్య_prev = hex_smin(n - nc_prev, n_chunk_n_నిలువుల_సంఖ్య);
                        దశాంశం *output_chunk = dst_ptr + (mr * dst_stride + nc_prev);
                        పూర్ణాంకం chunk_dst_నిలువుల_సంఖ్య = dst_నిలువుల_సంఖ్య - (పూర్ణాంకం)nc_prev;
                        ఒకవేళ (chunk_dst_నిలువుల_సంఖ్య > 0) {
                            transfer_output_chunk_threaded(ctx, output_chunk, ఖాళీ, vtcm_output_bufs[(i - 1) % 2], n_అడ్డువరుసలు, n_నిలువుల_సంఖ్య_prev, dst_stride, 0, chunk_dst_నిలువుల_సంఖ్య, n_దారాలు);
                        }
                    }
                }

                hmx_వరుస_తీయి(ctx->hmx_వరుస);
                స్థిరరాశి size_t nc_చివరి = (n_chunk_cnt - 1) * n_chunk_n_నిలువుల_సంఖ్య;
                స్థిరరాశి size_t n_నిలువుల_సంఖ్య_చివరి = hex_smin(n - nc_చివరి, n_chunk_n_నిలువుల_సంఖ్య);
                దశాంశం *output_chunk = dst_ptr + (mr * dst_stride + nc_చివరి);
                పూర్ణాంకం chunk_dst_నిలువుల_సంఖ్య = dst_నిలువుల_సంఖ్య - (పూర్ణాంకం)nc_చివరి;
                ఒకవేళ (chunk_dst_నిలువుల_సంఖ్య > 0) {
                    transfer_output_chunk_threaded(ctx, output_chunk, ఖాళీ, vtcm_output_bufs[(n_chunk_cnt - 1) % 2], n_అడ్డువరుసలు, n_నిలువుల_సంఖ్య_చివరి, dst_stride, 0, chunk_dst_నిలువుల_సంఖ్య, n_దారాలు);
                }
            }
        }
    } లేకపోతే {
        hmx_matmul_పని_t పని;
        శూన్య *vtcm_raw_bufs[2] = { vtcm_బరువు_raw[0], dequant_కార్మికుడు_fn ? vtcm_బరువు_raw[0] : vtcm_scratch0 };
        శూన్య *vtcm_బరువు_bufs[2] = { dequant_కార్మికుడు_fn ? vtcm_scratch0 : vtcm_బరువు_raw[0], vtcm_scratch0 };

        కొరకు (size_t mr = (size_t) m_ప్రారంభం; mr < mr_చివర; mr += m_chunk_n_అడ్డువరుసలు) {
            స్థిరరాశి size_t n_అడ్డువరుసలు = hex_smin(mr_చివర - mr, m_chunk_n_అడ్డువరుసలు);

            నిర్మాణం activation_transfer_params act_params = {
                .ctx = ctx,
                .dst = vtcm_f16_act,
                .act_dma_addr = act_dma_addr + (size_t) mr * act_stride * act_elem_కొలత,
                .n_అడ్డువరుసలు = (పూర్ణాంకం) n_అడ్డువరుసలు,
                .k_ఖండం = k,
                .k_stride = act_stride,
                .n_దారాలు = act_దారాలు,
                .act_దారాలు_div = act_దారాలు_div,
                .k_div = k_div,
                .k_valid = k_valid,
                .vtcm_f32_act = vtcm_f32_act,
                .vtcm_f32_act_బైట్లు = L.act_f32_బైట్లు,
                .act_elem_కొలత = act_elem_కొలత,
            };
            transfer_activation_chunk_threaded(&act_params);

            కొరకు (uint32_t p = 0; p < n_weights; p++) {
                స్థిరరాశి నిర్మాణం htp_tensor * పరిమిత src_w = octx->src[p];
                స్థిరరాశి నిర్మాణం htp_tensor * పరిమిత dst   = octx->dsts[p];
                ఒకవేళ (!src_w || !dst) కొనసాగు;

                స్థిరరాశి dma_addr_t బరువు       = src_w->దత్తాంశం;
                dma_వరుస * బరువు_dma      = octx->ctx->dma[0];
                దశాంశం * dst_ptr            = (దశాంశం *) dst->దత్తాంశం;
                స్థిరరాశి size_t n             = src_w->ne[1];
                ఒకవేళ (n == 0) కొనసాగు;
                స్థిరరాశి size_t బరువు_stride = src_w->nb[1];
                స్థిరరాశి size_t dst_stride    = dst->nb[1] / పరిమాణం(దశాంశం);
                స్థిరరాశి పూర్ణాంకం dst_నిలువుల_సంఖ్య         = (పూర్ణాంకం) dst->ne[0];
                స్థిరరాశి పూర్ణాంకం n_chunk_cnt      = hmx_ceil_div(n, n_chunk_n_నిలువుల_సంఖ్య);

                స్థిరరాశి uint32_t dma_src_stride = is_tiled ? tile_కొలత : బరువు_stride;

                ఒకవేళ (n > 0) {
                    స్థిరరాశి size_t n_నిలువుల_సంఖ్య = hex_smin(n, n_chunk_n_నిలువుల_సంఖ్య);
                    స్థిరరాశి uint32_t ఎత్తు = is_tiled ? hmx_ceil_div(n_నిలువుల_సంఖ్య, 32) * n_బరువు_tiles : n_నిలువుల_సంఖ్య;
                    dma_వరుస_నెట్టు(బరువు_dma, dma_make_దత్తాంశం(vtcm_raw_bufs[0], బరువు), dma_dst_stride, dma_src_stride, dma_వెడల్పు_బైట్లు, ఎత్తు);
                }

                కొరకు (పూర్ణాంకం i = 0; i < n_chunk_cnt; ++i) {
                    స్థిరరాశి size_t nc = (size_t) i * n_chunk_n_నిలువుల_సంఖ్య;
                    స్థిరరాశి size_t n_నిలువుల_సంఖ్య = hex_smin(n - nc, n_chunk_n_నిలువుల_సంఖ్య);
                    స్థిరరాశి size_t n_అడ్డువరుస_tiles = hmx_ceil_div(n_అడ్డువరుసలు, HTP_MM_HMX_TILE_N_అడ్డువరుసలు);
                    స్థిరరాశి size_t n_నిలువు_tiles = hmx_ceil_div(n_నిలువుల_సంఖ్య, HTP_MM_HMX_TILE_N_నిలువుల_సంఖ్య);

                    శూన్య * curr_raw = (శూన్య *) dma_వరుస_తీయి(బరువు_dma).dst;

                    dequantize_tiled_బరువు_chunk_గమ్యం_fp16_tiles(
                        ctx, vtcm_బరువు_bufs[i % 2], curr_raw,
                        n_నిలువుల_సంఖ్య, k, అడ్డువరుస_stride, బరువు_type,
                        n_k_tiles, n_k_tiles_div, dequant_కార్మికుడు_fn, n_దారాలు);

                    స్థిరరాశి size_t nc_తదుపరి = nc + n_chunk_n_నిలువుల_సంఖ్య;
                    ఒకవేళ (nc_తదుపరి < n) {
                        స్థిరరాశి size_t n_నిలువుల_సంఖ్య_తదుపరి = hex_smin(n - nc_తదుపరి, n_chunk_n_నిలువుల_సంఖ్య);
                        స్థిరరాశి uint32_t ఎత్తు_తదుపరి = is_tiled ? hmx_ceil_div(n_నిలువుల_సంఖ్య_తదుపరి, 32) * n_బరువు_tiles : n_నిలువుల_సంఖ్య_తదుపరి;
                        dma_వరుస_నెట్టు(బరువు_dma, dma_make_దత్తాంశం(vtcm_raw_bufs[(i + 1) % 2], బరువు + nc_తదుపరి * బరువు_stride), dma_dst_stride, dma_src_stride, dma_వెడల్పు_బైట్లు, ఎత్తు_తదుపరి);
                    }

                    hmx_matmul_పని_init(&పని, vtcm_output, vtcm_f16_act, (__fp16 *) vtcm_బరువు_bufs[i % 2], vtcm_scales, n_అడ్డువరుస_tiles, n_నిలువు_tiles, k / HTP_MM_HMX_TILE_N_అడ్డువరుసలు);
                    hmx_వరుస_నెట్టు(ctx->hmx_వరుస, hmx_వరుస_make_desc(hmx_matmul_కార్మికుడు_fn, &పని));
                    hmx_వరుస_తీయి(ctx->hmx_వరుస);

                    దశాంశం *output_chunk = dst_ptr + (mr * dst_stride + nc);
                    పూర్ణాంకం chunk_dst_నిలువుల_సంఖ్య = dst_నిలువుల_సంఖ్య - (పూర్ణాంకం)nc;
                    ఒకవేళ (chunk_dst_నిలువుల_సంఖ్య > 0) {
                        transfer_output_chunk_threaded(ctx, output_chunk, ఖాళీ, vtcm_output, n_అడ్డువరుసలు, n_నిలువుల_సంఖ్య, dst_stride, 0, chunk_dst_నిలువుల_సంఖ్య, n_దారాలు);
                    }
                }
            }
        }
    }

    తిరిగి HTP_స్థితి_OK;
}

స్థిర అంతఃవరుస dma_addr_t hmx_mm_బరువు_batch_దత్తాంశం(స్థిరరాశి hmx_mm_batched_params_t *params,
                                                         పూర్ణాంకం dst_b2, పూర్ణాంకం dst_b3) {
    స్థిరరాశి size_t b2_idx = (params->r2 <= 1) ? (size_t) dst_b2 : (size_t) fastdiv((uint32_t) dst_b2, &params->div_r2);
    స్థిరరాశి size_t b3_idx = (params->r3 <= 1) ? (size_t) dst_b3 : (size_t) fastdiv((uint32_t) dst_b3, &params->div_r3);
    తిరిగి params->బరువు + b2_idx * params->src0_nb2 + b3_idx * params->src0_nb3;
}

స్థిర అంతఃవరుస dma_addr_t hmx_mm_act_batch_addr(స్థిరరాశి hmx_mm_batched_params_t *params,
                                                పూర్ణాంకం dst_b2, పూర్ణాంకం dst_b3) {
    తిరిగి params->act_dma_addr + dst_b2 * params->act_nb2 +
           dst_b3 * params->act_nb3;
}

స్థిర అంతఃవరుస దశాంశం *hmx_mm_dst_batch_ptr(స్థిరరాశి hmx_mm_batched_params_t *params,
                                              పూర్ణాంకం dst_b2, పూర్ణాంకం dst_b3) {
    తిరిగి (దశాంశం *) ((uint8_t *) params->dst +
                      (size_t) dst_b2 * params->dst_nb2 +
                      (size_t) dst_b3 * params->dst_nb3);
}

స్థిర పూర్ణాంకం hmx_mm_batched_simple(నిర్మాణం htp_context *ctx,
                                 స్థిరరాశి hmx_mm_batched_params_t *params,
                                 పూర్ణాంకం m_chunk, పూర్ణాంకం n_chunk, పూర్ణాంకం pipeline, పూర్ణాంకం n_దారాలు, పూర్ణాంకం act_దారాలు, పూర్ణాంకం vtcm_కొలత,
                                 స్థిరరాశి నిర్మాణం fastdiv_విలువలు * act_దారాలు_div, స్థిరరాశి నిర్మాణం fastdiv_విలువలు * k_div) {
    పూర్ణాంకం ret = 0;
    కొరకు (పూర్ణాంకం b3 = 0; b3 < params->ne13 && ret == 0; ++b3) {
        కొరకు (పూర్ణాంకం b2 = 0; b2 < params->ne12 && ret == 0; ++b2) {
            dma_addr_t cur_bias_addr = params->bias_addr ? (params->bias_addr +
                                       b2 * params->bias_nb2 +
                                       b3 * params->bias_nb3) : 0;
            ret = hmx_mm_2d_f32(ctx, params->బరువు_dma, hmx_mm_dst_batch_ptr(params, b2, b3),
                                cur_bias_addr, params->bias_బైట్లు,
                                hmx_mm_act_batch_addr(params, b2, b3),
                                hmx_mm_బరువు_batch_దత్తాంశం(params, b2, b3),
                                params->m, params->k, params->n,
                                params->act_stride, params->act_elem_కొలత, params->బరువు_stride,
                                params->బరువు_type, params->k, params->dst_stride, params->bias_stride, params->dst_నిలువుల_సంఖ్య,
                                m_chunk, n_chunk, pipeline, n_దారాలు, act_దారాలు,
                                act_దారాలు_div, k_div, params->tile_కొలత, params->aligned_tile_కొలత, vtcm_కొలత);
        }
    }
    తిరిగి ret;
}

స్థిర పూర్ణాంకం hmx_mm_batched(నిర్మాణం htp_context *ctx, స్థిరరాశి hmx_mm_batched_params_t *params,
                          పూర్ణాంకం m_chunk, పూర్ణాంకం n_chunk, పూర్ణాంకం pipeline, పూర్ణాంకం n_దారాలు, పూర్ణాంకం act_దారాలు,
                          స్థిరరాశి నిర్మాణం fastdiv_విలువలు * act_దారాలు_div,
                          స్థిరరాశి నిర్మాణం fastdiv_విలువలు * k_div,
                          పూర్ణాంకం vtcm_కొలత) {
    స్థిరరాశి size_t అడ్డువరుస_stride = htp_tensor_get_అడ్డువరుస_కొలత(params->బరువు_type, params->k);
    ఒకవేళ (అడ్డువరుస_stride == 0) { తిరిగి -1; }
    ఒకవేళ (params->act_stride < params->k || (params->tile_కొలత == 0 && (size_t) params->బరువు_stride < అడ్డువరుస_stride) || params->dst_stride < params->dst_నిలువుల_సంఖ్య) { తిరిగి -1; }
    ఒకవేళ (params->ne02 <= 0 || params->ne03 <= 0 || params->ne12 <= 0 || params->ne13 <= 0) { తిరిగి -1; }
    ఒకవేళ (params->ne12 % params->ne02 != 0 || params->ne13 % params->ne03 != 0) { తిరిగి -1; }
    // N (the weight row count) does not have to be 32-aligned:
    ఒకవేళ (params->k % 32 != 0) { తిరిగి -1; }
    ఒకవేళ (!hex_is_aligned(params->dst, పరిమాణం(దశాంశం))) { తిరిగి -1; }

    స్థిరరాశి పూర్ణాంకం సమూహం_కొలత = params->r2;
    స్థిరరాశి size_t vtcm_budget  = ctx->vtcm_కొలత;

    // Check if the precomputed parameters are grouped or simple.
    // If simple, or if group_size <= 1, we use simple fallback loop.
    // Grouped path is only valid for flat F16 if group_size > 1 and it fits within VTCM budget.
    bool నిరంతరం_grouped = (params->బరువు_type == HTP_TYPE_F16 && params->tile_కొలత == 0 && సమూహం_కొలత > 1 && (size_t)vtcm_కొలత <= vtcm_budget);
    ఒకవేళ (!నిరంతరం_grouped) {
        తిరిగి hmx_mm_batched_simple(ctx, params, m_chunk, n_chunk, pipeline, n_దారాలు, act_దారాలు, vtcm_కొలత, act_దారాలు_div, k_div);
    }

    నిర్మాణం htp_thread_trace * tr = &ctx->trace[0];
    htp_trace_event_ప్రారంభం(tr, HTP_TRACE_EVT_INIT, 0);

    size_t m_chunk_n_అడ్డువరుసలు = m_chunk;
    size_t n_chunk_n_నిలువుల_సంఖ్య = n_chunk;

    నిర్మాణం htp_mm_hmx_vtcm_layout L;
    htp_mm_hmx_vtcm_layout_build(&L, HTP_MM_KERNEL_HMX_BATCHED, HTP_TYPE_F16, params->k, m_chunk_n_అడ్డువరుసలు, n_chunk_n_నిలువుల_సంఖ్య, సమూహం_కొలత, false, act_దారాలు, 0, params->bias_బైట్లు);

    ఒకవేళ (L.మొత్తం_బైట్లు > vtcm_budget) {
        FARF(ఎగువ_2, "%s: grouped layout overflowed VTCM, falling back to simple batched loop", __func__);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_INIT, 0);
        తిరిగి hmx_mm_batched_simple(ctx, params, m_chunk, n_chunk, pipeline, n_దారాలు, act_దారాలు, vtcm_కొలత, act_దారాలు_div, k_div);
    }

    uint8_t * స్థిరరాశి పునాది = (uint8_t *) ctx->vtcm_పునాది;
    __fp16  *vtcm_బరువు     = VTCM_LAYOUT_PTR(__fp16, పునాది, L.off_బరువు[0]);
    __fp16  *vtcm_f16_act    = VTCM_LAYOUT_PTR(__fp16, పునాది, L.off_act);
    __fp16  *vtcm_output     = VTCM_LAYOUT_PTR(__fp16, పునాది, L.off_dst[0]);
    శూన్య    *vtcm_scratch0   = VTCM_LAYOUT_PTR(శూన్య, పునాది, L.off_scratch[0]);
    శూన్య    *vtcm_scratch1   = VTCM_LAYOUT_PTR(శూన్య, పునాది, L.off_scratch[1]);
    __fp16  *vtcm_scales     = VTCM_LAYOUT_PTR(__fp16, పునాది, L.off_scales);
    దశాంశం   *vtcm_f32_act    = VTCM_LAYOUT_PTR(దశాంశం, పునాది, L.off_act_f32);

    స్థిరరాశి bool has_bias      = (params->bias_బైట్లు > 0 && params->bias_addr != 0);
    దశాంశం   *vtcm_bias       = VTCM_LAYOUT_PTR_OPTIONAL(దశాంశం, పునాది, L.off_bias, has_bias);
    ఒకవేళ (has_bias) {
        dma_వరుస_నెట్టు(params->బరువు_dma, dma_make_దత్తాంశం(vtcm_bias, params->bias_addr), hex_align_పైకి(params->bias_బైట్లు, 128), 0, params->bias_బైట్లు, 1);
        dma_వరుస_తీయి(params->బరువు_dma);
    }

    hmx_init_స్తంభం_scales(vtcm_scales, Q6_V_vsplat_R(0x3c00));  // scale: 1.0, bias: 0.0 in FP16

    FARF(ఎగువ_2, "%s: grouped path m=%d k=%d n=%d group=%d streams=%d mc=%zu nc=%zu vtcm=%zu/%zu",
            __func__, params->m, params->k, params->n, సమూహం_కొలత, params->ne13,
            m_chunk_n_అడ్డువరుసలు, n_chunk_n_నిలువుల_సంఖ్య,
            L.మొత్తం_బైట్లు, vtcm_budget);

    స్థిరరాశి size_t fp16_అడ్డువరుస_బైట్లు   = (size_t) params->k * పరిమాణం(__fp16);
    స్థిరరాశి size_t బరువు_అడ్డువరుస_బైట్లు = (size_t) params->బరువు_stride;

    htp_trace_event_stop(tr, HTP_TRACE_EVT_INIT, 0);

    hmx_matmul_పని_t పని;

    కొరకు (పూర్ణాంకం b3 = 0; b3 < params->ne13; ++b3) {
        కొరకు (పూర్ణాంకం b2_పునాది = 0; b2_పునాది < params->ne12; b2_పునాది += సమూహం_కొలత) {
            స్థిరరాశి dma_addr_t బరువు_సమూహం = hmx_mm_బరువు_batch_దత్తాంశం(params, b2_పునాది, b3);
            dma_వరుస * బరువు_dma = params->బరువు_dma;

            కొరకు (size_t mr = 0; mr < (size_t) params->m; mr += m_chunk_n_అడ్డువరుసలు) {
                స్థిరరాశి size_t n_అడ్డువరుసలు = hex_smin((size_t) params->m - mr, m_chunk_n_అడ్డువరుసలు);
                స్థిరరాశి size_t n_అడ్డువరుస_tiles = hmx_ceil_div((పూర్ణాంకం) n_అడ్డువరుసలు, HTP_MM_HMX_TILE_N_అడ్డువరుసలు);

                // Pre-load activations for all heads in the group (once per m_chunk).
                // When the source is strided (permuted Q), use 2D DMA to gather
                // contiguous rows into a VTCM scratch buffer first, then HVX
                // converts from the contiguous VTCM buffer.  This avoids L2 cache
                // thrashing from HVX loads at large strides.
                కొరకు (పూర్ణాంకం g = 0; g < సమూహం_కొలత; ++g) {
                    స్థిరరాశి dma_addr_t act_dma_addr = hmx_mm_act_batch_addr(params, b2_పునాది + g, b3) +
                                                      (size_t) mr * params->act_stride * params->act_elem_కొలత;
                    __fp16 *vtcm_act_g = vtcm_f16_act + (size_t) g * L.act_తల_stride;
                    నిర్మాణం activation_transfer_params act_params = {
                        .ctx = ctx,
                        .dst = vtcm_act_g,
                        .act_dma_addr = act_dma_addr,
                        .n_అడ్డువరుసలు = (పూర్ణాంకం) n_అడ్డువరుసలు,
                        .k_ఖండం = params->k,
                        .k_stride = params->act_stride,
                        .n_దారాలు = act_దారాలు,
                        .act_దారాలు_div = act_దారాలు_div,
                        .k_div = k_div,
                        .k_valid = params->k,
                        .vtcm_f32_act = vtcm_f32_act,
                        .vtcm_f32_act_బైట్లు = L.act_f32_బైట్లు,
                        .act_elem_కొలత = params->act_elem_కొలత,
                    };
                    transfer_activation_chunk_threaded(&act_params);
                }

                // Prologue: Push A0 and A1 (if exists)
                {
                    స్థిరరాశి size_t n_నిలువుల_సంఖ్య_మొదటి = hex_smin((size_t) params->n, n_chunk_n_నిలువుల_సంఖ్య);
                    dma_వరుస_నెట్టు(బరువు_dma, dma_make_దత్తాంశం(vtcm_scratch0, బరువు_సమూహం),
                                      fp16_అడ్డువరుస_బైట్లు, బరువు_అడ్డువరుస_బైట్లు, fp16_అడ్డువరుస_బైట్లు, n_నిలువుల_సంఖ్య_మొదటి);
                }
                ఒకవేళ (n_chunk_n_నిలువుల_సంఖ్య < (size_t) params->n) {
                    స్థిరరాశి size_t n_నిలువుల_సంఖ్య_రెండవ = hex_smin((size_t) params->n - n_chunk_n_నిలువుల_సంఖ్య, n_chunk_n_నిలువుల_సంఖ్య);
                    స్థిరరాశి dma_addr_t రెండవ_బరువు_chunk = బరువు_సమూహం + n_chunk_n_నిలువుల_సంఖ్య * బరువు_అడ్డువరుస_బైట్లు;
                    dma_వరుస_నెట్టు(బరువు_dma, dma_make_దత్తాంశం(vtcm_scratch1, రెండవ_బరువు_chunk),
                                      fp16_అడ్డువరుస_బైట్లు, బరువు_అడ్డువరుస_బైట్లు, fp16_అడ్డువరుస_బైట్లు, n_నిలువుల_సంఖ్య_రెండవ);
                }

                కొరకు (size_t nc = 0; nc < (size_t) params->n; nc += n_chunk_n_నిలువుల_సంఖ్య) {
                    స్థిరరాశి size_t n_నిలువుల_సంఖ్య      = hex_smin((size_t) params->n - nc, n_chunk_n_నిలువుల_సంఖ్య);
                    స్థిరరాశి size_t n_నిలువు_tiles = hmx_ceil_div((పూర్ణాంకం) n_నిలువుల_సంఖ్య, HTP_MM_HMX_TILE_N_నిలువుల_సంఖ్య);

                    {
                        శూన్య * curr_raw = (శూన్య *) dma_వరుస_తీయి(బరువు_dma).dst;

                        స్థిరరాశి size_t n_నిలువుల_సంఖ్య_tiled = hex_align_పైకి(n_నిలువుల_సంఖ్య, HTP_MM_HMX_TILE_N_నిలువుల_సంఖ్య);

                        hmx_interleave_అడ్డువరుసలు_గమ్యం_tiles(vtcm_బరువు, (స్థిరరాశి __fp16 *) curr_raw, (uint32_t) n_నిలువుల_సంఖ్య, params->k, params->k, 0, (uint32_t) n_నిలువుల_సంఖ్య_tiled);

                        స్థిరరాశి size_t nc_తదుపరి = nc + n_chunk_n_నిలువుల_సంఖ్య * 2;
                        ఒకవేళ (nc_తదుపరి < (size_t) params->n) {
                            స్థిరరాశి size_t n_నిలువుల_సంఖ్య_తదుపరి = hex_smin((size_t) params->n - nc_తదుపరి, n_chunk_n_నిలువుల_సంఖ్య);
                            స్థిరరాశి dma_addr_t తదుపరి_బరువు_chunk = బరువు_సమూహం + nc_తదుపరి * బరువు_అడ్డువరుస_బైట్లు;

                            dma_వరుస_నెట్టు(బరువు_dma, dma_make_దత్తాంశం(curr_raw, తదుపరి_బరువు_chunk),
                                              fp16_అడ్డువరుస_బైట్లు, బరువు_అడ్డువరుస_బైట్లు, fp16_అడ్డువరుస_బైట్లు, n_నిలువుల_సంఖ్య_తదుపరి);
                        }
                    }

                    // Reuse the interleaved weight for every q_head in this GQA group
                    కొరకు (పూర్ణాంకం g = 0; g < సమూహం_కొలత; ++g) {
                        {
                            స్థిరరాశి __fp16 * vtcm_act_g = vtcm_f16_act + (size_t) g * L.act_తల_stride;
                            hmx_matmul_పని_init(&పని, vtcm_output, vtcm_act_g, vtcm_బరువు, vtcm_scales, n_అడ్డువరుస_tiles, n_నిలువు_tiles, params->k / 32);
                            hmx_వరుస_నెట్టు(ctx->hmx_వరుస, hmx_వరుస_make_desc(hmx_matmul_కార్మికుడు_fn, &పని));
                            hmx_వరుస_తీయి(ctx->hmx_వరుస);
                        }

                        {
                            దశాంశం *output = hmx_mm_dst_batch_ptr(params, b2_పునాది + g, b3) + mr * params->dst_stride + nc;
                            స్థిరరాశి దశాంశం *bias_chunk = has_bias ? (vtcm_bias + mr * params->bias_stride + nc) : ఖాళీ;
                            పూర్ణాంకం chunk_dst_నిలువుల_సంఖ్య = params->dst_నిలువుల_సంఖ్య - (పూర్ణాంకం)nc;
                            ఒకవేళ (chunk_dst_నిలువుల_సంఖ్య > 0) {
                                transfer_output_chunk_threaded(ctx, output, bias_chunk, vtcm_output, (పూర్ణాంకం) n_అడ్డువరుసలు, (పూర్ణాంకం) n_నిలువుల_సంఖ్య,
                                                               params->dst_stride, params->bias_stride, chunk_dst_నిలువుల_సంఖ్య, n_దారాలు);
                            }
                        }
                    }
                }
            }
        }
    }

    తిరిగి 0;
}

స్థిర శూన్య transfer_activation_chunk_gathered_threaded(
            నిర్మాణం htp_context *ctx,
            __fp16 *dst,
            స్థిరరాశి దశాంశం *src,
            పూర్ణాంకం ప్రారంభం_అడ్డువరుస,
            పూర్ణాంకం n_అడ్డువరుసలు,
            పూర్ణాంకం k_ఖండం,
            స్థిరరాశి నిర్మాణం mmid_అడ్డువరుస_mapping *మాతృక_అడ్డువరుసలు,
            పూర్ణాంకం cur_a,
            పూర్ణాంకం mapping_stride,
            పూర్ణాంకం ne11,
            size_t nb11,
            size_t nb12,
            పూర్ణాంకం cne1,
            పూర్ణాంకం n_దారాలు,
            పూర్ణాంకం k_valid) {
    ఒకవేళ (n_అడ్డువరుసలు <= 0) తిరిగి;
    పూర్ణాంకం chunks_చొప్పున_thread = hmx_ceil_div(n_అడ్డువరుసలు, n_దారాలు);
    chunks_చొప్పున_thread = hex_align_పైకి(chunks_చొప్పున_thread, 2);

    పూర్ణాంకం actual_దారాలు = hmx_ceil_div(n_అడ్డువరుసలు, chunks_చొప్పున_thread);

    activation_transfer_gathered_task_state_t state = {
        .dst               = dst,
        .src               = src,
        .n_tasks           = actual_దారాలు,
        .n_tot_chunks      = n_అడ్డువరుసలు,
        .n_chunks_చొప్పున_task = chunks_చొప్పున_thread,
        .k_ఖండం           = k_ఖండం,
        .మాతృక_అడ్డువరుసలు       = మాతృక_అడ్డువరుసలు,
        .cur_a             = cur_a,
        .mapping_stride    = mapping_stride,
        .ne11              = ne11,
        .ne11_div          = ne11 > 1 ? init_fastdiv_విలువలు(ne11) : (నిర్మాణం fastdiv_విలువలు){0, 0},
        .nb11              = nb11,
        .nb12              = nb12,
        .ప్రారంభం_అడ్డువరుస         = ప్రారంభం_అడ్డువరుస,
        .cne1              = cne1,
        .k_valid           = k_valid,
        .traces            = ctx->trace,
    };

    కార్మికుడు_callback_t కార్మికుడు_fn = ne11 == 1 ? transfer_activation_chunk_gathered_కార్మికుడు_flat_fn :
                                              transfer_activation_chunk_gathered_కార్మికుడు_fn;

    ఒకవేళ (actual_దారాలు <= 1) {
        కార్మికుడు_fn(1, 0, &state);
    } లేకపోతే {
        కార్మికుడు_pool_నిరంతరం_func(ctx->కార్మికుడు_pool, కార్మికుడు_fn, &state, actual_దారాలు);
    }
}

స్థిర శూన్య transfer_output_chunk_scattered_threaded(
            నిర్మాణం htp_context *ctx,
            దశాంశం *dst,
            స్థిరరాశి __fp16 *vtcm_src,
            పూర్ణాంకం ప్రారంభం_అడ్డువరుస,
            పూర్ణాంకం n_అడ్డువరుసలు,
            పూర్ణాంకం n_నిలువుల_సంఖ్య,
            స్థిరరాశి నిర్మాణం mmid_అడ్డువరుస_mapping *మాతృక_అడ్డువరుసలు,
            పూర్ణాంకం cur_a,
            పూర్ణాంకం mapping_stride,
            size_t dst_nb1,
            size_t dst_nb2,
            పూర్ణాంకం cne1,
            పూర్ణాంకం n_దారాలు) {
    ఒకవేళ (n_అడ్డువరుసలు <= 0) తిరిగి;
    పూర్ణాంకం chunks_చొప్పున_thread = hmx_ceil_div(n_అడ్డువరుసలు, n_దారాలు);
    chunks_చొప్పున_thread = hex_align_పైకి(chunks_చొప్పున_thread, 2);

    పూర్ణాంకం actual_దారాలు = hmx_ceil_div(n_అడ్డువరుసలు, chunks_చొప్పున_thread);

    output_transfer_scattered_task_state_t state = {
        .vtcm_src          = vtcm_src,
        .dst               = dst,
        .n_tasks           = actual_దారాలు,
        .n_tot_chunks      = n_అడ్డువరుసలు,
        .n_chunks_చొప్పున_task = chunks_చొప్పున_thread,
        .n_నిలువుల_సంఖ్య            = n_నిలువుల_సంఖ్య,
        .మాతృక_అడ్డువరుసలు       = మాతృక_అడ్డువరుసలు,
        .cur_a             = cur_a,
        .mapping_stride    = mapping_stride,
        .dst_nb1           = dst_nb1,
        .dst_nb2           = dst_nb2,
        .ప్రారంభం_అడ్డువరుస         = ప్రారంభం_అడ్డువరుస,
        .cne1              = cne1,
        .traces            = ctx->trace,
    };

    ఒకవేళ (actual_దారాలు <= 1) {
        transfer_output_chunk_scattered_కార్మికుడు_fn(1, 0, &state);
    } లేకపోతే {
        కార్మికుడు_pool_నిరంతరం_func(ctx->కార్మికుడు_pool, transfer_output_chunk_scattered_కార్మికుడు_fn, &state, actual_దారాలు);
    }
}

స్థిర పూర్ణాంకం hmx_mm_id_2d_f32(నిర్మాణం htp_context *ctx,
                                         dma_వరుస *బరువు_dma,
                                         దశాంశం *పరిమిత dst,
                                         స్థిరరాశి దశాంశం *activation,
                                         dma_addr_t బరువు,
                                         పూర్ణాంకం m, పూర్ణాంకం k, పూర్ణాంకం n,
                                         పూర్ణాంకం k_valid,
                                         పూర్ణాంకం ne11,
                                         size_t act_nb1, size_t act_nb2,
                                         size_t dst_nb1, size_t dst_nb2,
                                         పూర్ణాంకం బరువు_stride,
                                         పూర్ణాంకం బరువు_type,
                                         పూర్ణాంకం tile_కొలత,
                                         పూర్ణాంకం aligned_tile_కొలత,
                                         స్థిరరాశి నిర్మాణం mmid_అడ్డువరుస_mapping *మాతృక_అడ్డువరుసలు,
                                         పూర్ణాంకం cur_a,
                                         పూర్ణాంకం mapping_stride,
                                         పూర్ణాంకం m_ప్రారంభం,
                                         పూర్ణాంకం m_చివర,
                                         పూర్ణాంకం n_దారాలు) {
    నిర్మాణం htp_thread_trace * tr = &ctx->trace[0];
    htp_trace_event_ప్రారంభం(tr, HTP_TRACE_EVT_INIT, 0);

    స్థిరరాశి పూర్ణాంకం cne1 = m;
    స్థిరరాశి పూర్ణాంకం m_core = m_చివర - m_ప్రారంభం;
    స్థిరరాశి పూర్ణాంకం m_core_padded = hex_align_పైకి(m_core > 0 ? m_core : 1, 32);

    ఒకవేళ (k % 32 != 0 || n % 32 != 0) { తిరిగి -1; }
    ఒకవేళ (!hex_is_aligned(dst, VLEN) || !hex_is_aligned(activation, VLEN)) { తిరిగి -1; }

    size_t అడ్డువరుస_stride = htp_tensor_get_అడ్డువరుస_కొలత(బరువు_type, k);
    ఒకవేళ (అడ్డువరుస_stride == 0) {
        తిరిగి -1;
    }

    స్థిరరాశి పూర్ణాంకం n_k_tiles = k / HTP_MM_HMX_TILE_N_నిలువుల_సంఖ్య;
    స్థిరరాశి uint32_t n_బరువు_tiles = k / htp_tensor_get_tile_k(బరువు_type);
    స్థిరరాశి నిర్మాణం fastdiv_విలువలు n_k_tiles_div = init_fastdiv_విలువలు(n_k_tiles);

    స్థిరరాశి size_t vec_dot_కొలత = k * పరిమాణం(__fp16);
    స్థిరరాశి size_t vtcm_budget  = ctx->vtcm_కొలత;
    size_t vtcm_వాడిన = 0;

    స్థిరరాశి bool is_tiled = (tile_కొలత > 0);
    tiled_dequant_task_fn_t dequant_కార్మికుడు_fn = hmx_get_dequant_కార్మికుడు_fn(బరువు_type, is_tiled);

    స్థిరరాశి uint32_t dma_dst_stride  = is_tiled ? aligned_tile_కొలత : అడ్డువరుస_stride;
    స్థిరరాశి uint32_t dma_src_stride  = is_tiled ? tile_కొలత : బరువు_stride;
    స్థిరరాశి uint32_t dma_వెడల్పు_బైట్లు = is_tiled ? tile_కొలత : అడ్డువరుస_stride;

    స్థిరరాశి size_t qweight_అడ్డువరుస_stride = is_tiled ? (size_t)(n_బరువు_tiles * aligned_tile_కొలత) / 32 : 0;
    స్థిరరాశి size_t బరువు_అడ్డువరుస_stride = is_tiled ? qweight_అడ్డువరుస_stride : అడ్డువరుస_stride;

    size_t కొలత_చొప్పున_n = 0, కొలత_చొప్పున_m = 0, కొలత_చొప్పున_mn = 0;
    htp_mm_hmx_get_2d_chunk_costs(బరువు_type, k, /*pipeline=*/false, aligned_tile_కొలత,
                                  &కొలత_చొప్పున_n, &కొలత_చొప్పున_m, &కొలత_చొప్పున_mn);

    స్థిరరాశి size_t overhead = htp_mm_hmx_get_2d_overhead(/*pipeline=*/false, /*is_matmul_id=*/true);
    size_t m_chunk_n_అడ్డువరుసలు = 0, n_chunk_n_నిలువుల_సంఖ్య = 0;
    ఒకవేళ (htp_mm_hmx_compute_chunks(vtcm_budget, overhead, కొలత_చొప్పున_n, కొలత_చొప్పున_m, కొలత_చొప్పున_mn,
                           m_core_padded, n,
                           /*m_block_cost=*/(size_t) n * HTP_MM_HMX_ఖర్చు_W_DEQUANT,
                           /*n_block_cost=*/(size_t) m_core_padded * HTP_MM_HMX_ఖర్చు_A_CONVERT, &m_chunk_n_అడ్డువరుసలు, &n_chunk_n_నిలువుల_సంఖ్య, &vtcm_వాడిన)) {
        FARF(ERROR, "hmx-mm-id-2d: VTCM too small : m %d k %d n %d budget %zu", m_core_padded, k, n, vtcm_budget);
        తిరిగి -1;
    }

    స్థిరరాశి size_t బరువు_area_కొలత = hex_align_పైకి(n_chunk_n_నిలువుల_సంఖ్య * బరువు_అడ్డువరుస_stride, HTP_MM_HMX_TILE_కొలత);
    స్థిరరాశి size_t act_area_కొలత    = hex_align_పైకి(m_chunk_n_అడ్డువరుసలు * vec_dot_కొలత, HTP_MM_HMX_TILE_కొలత);
    స్థిరరాశి size_t output_area_కొలత = hex_align_పైకి(m_chunk_n_అడ్డువరుసలు * n_chunk_n_నిలువుల_సంఖ్య * పరిమాణం(__fp16), HTP_MM_HMX_TILE_కొలత);

    size_t scratch0_కొలత = hex_align_పైకి(n_chunk_n_నిలువుల_సంఖ్య * vec_dot_కొలత, HTP_MM_HMX_TILE_కొలత);

    uint8_t *vtcm_ptr      = (uint8_t *) ctx->vtcm_పునాది;
    __fp16  *vtcm_బరువు   = బరువు_area_కొలత ? (__fp16 *) vtcm_seq_alloc(&vtcm_ptr, బరువు_area_కొలత) : ఖాళీ;
    __fp16  *vtcm_f16_act  = (__fp16 *) vtcm_seq_alloc(&vtcm_ptr, act_area_కొలత);
    __fp16  *vtcm_output   = (__fp16 *) vtcm_seq_alloc(&vtcm_ptr, output_area_కొలత);
    శూన్య    *vtcm_scratch0 = vtcm_seq_alloc(&vtcm_ptr, scratch0_కొలత);
    __fp16  *vtcm_scales   = (__fp16 *) vtcm_seq_alloc(&vtcm_ptr, 256);

    vtcm_వాడిన = vtcm_ptr - (uint8_t *) ctx->vtcm_పునాది;
    ఒకవేళ (vtcm_వాడిన > vtcm_budget) {
        FARF(ERROR, "hmx-mm-id-2d: VTCM overflow: used %zu budget %zu", vtcm_వాడిన, vtcm_budget);
        తిరిగి -1;
    }

    hmx_init_స్తంభం_scales(vtcm_scales, Q6_V_vsplat_R(0x3c00));

    htp_trace_event_stop(tr, HTP_TRACE_EVT_INIT, 0);

    hmx_matmul_పని_t పని;
    శూన్య *vtcm_raw_bufs[2] = { vtcm_బరువు, dequant_కార్మికుడు_fn ? vtcm_బరువు : vtcm_scratch0 };
    శూన్య *vtcm_బరువు_bufs[2] = { dequant_కార్మికుడు_fn ? vtcm_scratch0 : vtcm_బరువు, vtcm_scratch0 };
    స్థిరరాశి పూర్ణాంకం n_chunk_cnt = hmx_ceil_div(n, n_chunk_n_నిలువుల_సంఖ్య);

    కొరకు (size_t mr = (size_t) m_ప్రారంభం; mr < (size_t) m_చివర; mr += m_chunk_n_అడ్డువరుసలు) {
        స్థిరరాశి size_t n_అడ్డువరుసలు = hex_smin((size_t) m_చివర - mr, m_chunk_n_అడ్డువరుసలు);
        స్థిరరాశి size_t n_అడ్డువరుస_tiles = hmx_ceil_div(n_అడ్డువరుసలు, HTP_MM_HMX_TILE_N_అడ్డువరుసలు);

        transfer_activation_chunk_gathered_threaded(
            ctx, vtcm_f16_act, activation, (పూర్ణాంకం) mr, (పూర్ణాంకం) n_అడ్డువరుసలు, k,
            మాతృక_అడ్డువరుసలు, cur_a, mapping_stride, ne11, act_nb1, act_nb2, cne1, n_దారాలు, k_valid);

        // A0: Pre-fetch the first weight chunk (nc = 0)
        ఒకవేళ (n > 0) {
            స్థిరరాశి size_t n_నిలువుల_సంఖ్య = hex_smin((size_t) n, n_chunk_n_నిలువుల_సంఖ్య);
            స్థిరరాశి uint32_t ఎత్తు = is_tiled ? hmx_ceil_div(n_నిలువుల_సంఖ్య, 32) * n_బరువు_tiles : n_నిలువుల_సంఖ్య;
            dma_వరుస_నెట్టు(బరువు_dma, dma_make_దత్తాంశం(vtcm_raw_bufs[0], బరువు),
                           dma_dst_stride, dma_src_stride, dma_వెడల్పు_బైట్లు, ఎత్తు);
        }

        కొరకు (పూర్ణాంకం i = 0; i < n_chunk_cnt; ++i) {
            స్థిరరాశి size_t nc = (size_t) i * n_chunk_n_నిలువుల_సంఖ్య;
            స్థిరరాశి size_t n_నిలువుల_సంఖ్య = hex_smin((size_t) n - nc, n_chunk_n_నిలువుల_సంఖ్య);
            స్థిరరాశి size_t n_నిలువు_tiles = hmx_ceil_div(n_నిలువుల_సంఖ్య, HTP_MM_HMX_TILE_N_నిలువుల_సంఖ్య);

            // A: Wait for weight DMA
            శూన్య * curr_raw = (శూన్య *) dma_వరుస_తీయి(బరువు_dma).dst;

            // B: Weight Dequantize (Threaded)
            dequantize_tiled_బరువు_chunk_గమ్యం_fp16_tiles(
                ctx, vtcm_బరువు_bufs[i % 2], curr_raw,
                n_నిలువుల_సంఖ్య, k, అడ్డువరుస_stride, బరువు_type,
                n_k_tiles, n_k_tiles_div, dequant_కార్మికుడు_fn, n_దారాలు
            );

            // Start weight DMA for the next chunk early
            స్థిరరాశి size_t nc_తదుపరి = nc + n_chunk_n_నిలువుల_సంఖ్య;
            ఒకవేళ (nc_తదుపరి < (size_t) n) {
                స్థిరరాశి size_t n_నిలువుల_సంఖ్య_తదుపరి = hex_smin((size_t) n - nc_తదుపరి, n_chunk_n_నిలువుల_సంఖ్య);
                స్థిరరాశి uint32_t ఎత్తు_తదుపరి = is_tiled ? hmx_ceil_div(n_నిలువుల_సంఖ్య_తదుపరి, 32) * n_బరువు_tiles : n_నిలువుల_సంఖ్య_తదుపరి;
                dma_వరుస_నెట్టు(బరువు_dma, dma_make_దత్తాంశం(vtcm_raw_bufs[(i + 1) % 2], బరువు + nc_తదుపరి * బరువు_stride),
                               dma_dst_stride, dma_src_stride, dma_వెడల్పు_బైట్లు, ఎత్తు_తదుపరి);
            }

            // C: HMX Compute (Queue-based)
            hmx_matmul_పని_init(&పని, vtcm_output, vtcm_f16_act, (__fp16 *) vtcm_బరువు_bufs[i % 2], vtcm_scales, n_అడ్డువరుస_tiles, n_నిలువు_tiles, k / HTP_MM_HMX_TILE_N_అడ్డువరుసలు);
            hmx_వరుస_నెట్టు(ctx->hmx_వరుస, hmx_వరుస_make_desc(hmx_matmul_కార్మికుడు_fn, &పని));
            hmx_వరుస_తీయి(ctx->hmx_వరుస);

            // D: Output Store
            transfer_output_chunk_scattered_threaded(
                ctx, dst + nc, vtcm_output, (పూర్ణాంకం) mr, (పూర్ణాంకం) n_అడ్డువరుసలు, (పూర్ణాంకం) n_నిలువుల_సంఖ్య,
                మాతృక_అడ్డువరుసలు, cur_a, mapping_stride, dst_nb1, dst_nb2, cne1, n_దారాలు);
        }
    }

    తిరిగి 0;
}

// --- Dispatchers and Public Entry Points ---

స్థిర పూర్ణాంకం hmx_mm_op_matmul(నిర్మాణం htp_ops_context * octx, స్థిరరాశి నిర్మాణం htp_mm_kernel_params * kparams) {
    htp_matmul_tensors_preamble;

    పూర్ణాంకం k = (పూర్ణాంకం) src0->ne[0];
    పూర్ణాంకం n = (పూర్ణాంకం) src0->ne[1];
    స్థిరరాశి పూర్ణాంకం m_మొత్తం    = (పూర్ణాంకం) act->ne[1];
    స్థిరరాశి uint32_t act_elem_కొలత = (act->type == HTP_TYPE_F16) ? పరిమాణం(__fp16) : పరిమాణం(దశాంశం);
    స్థిరరాశి పూర్ణాంకం act_stride = (పూర్ణాంకం) (act->nb[1] / act_elem_కొలత);

    పూర్ణాంకం m_ప్రారంభం = 0;
    పూర్ణాంకం m_అడ్డువరుసలు  = m_మొత్తం;
    ఒకవేళ (octx->ctx->mdev.లెక్క > 1) {
        bool can_విభజించు = htp_tensor_can_అడ్డువరుస_విభజన(dst, పరిమాణం(దశాంశం)) &&
                         htp_tensor_can_అడ్డువరుస_విభజన(act, act_elem_కొలత);
        ఒకవేళ (src2 && src2->ne[1] > 1 && !htp_tensor_can_అడ్డువరుస_విభజన(src2, పరిమాణం(దశాంశం))) {
            can_విభజించు = false;
        }
        స్థిరరాశి నిర్మాణం htp_tensor_mdev_range range = htp_tensor_mdev_విభజన((uint32_t) m_మొత్తం, can_విభజించు ? 1 : 0, octx->ctx->mdev.idx, octx->ctx->mdev.లెక్క, &octx->ctx->mdev.లెక్క_div);
        m_ప్రారంభం = (పూర్ణాంకం) range.ప్రారంభం;
        m_అడ్డువరుసలు  = (పూర్ణాంకం) range.లెక్క;
    }

    ఒకవేళ (m_అడ్డువరుసలు == 0) {
        తిరిగి HTP_స్థితి_OK;
    }

    dma_addr_t src2_addr = 0;
    size_t src2_బైట్లు = 0;
    uint32_t src2_stride = 0;
    size_t src2_nb2 = 0;
    size_t src2_nb3 = 0;
    ఒకవేళ (src2) {
        src2_stride = (src2->ne[1] == 1) ? 0 : (uint32_t) (src2->nb[1] / పరిమాణం(దశాంశం));
        src2_addr = src2->దత్తాంశం + m_ప్రారంభం * src2_stride * పరిమాణం(దశాంశం);
        src2_బైట్లు = (size_t) kparams->vtcm_bias_కొలత;
        src2_nb2 = (src2->ne[2] == 1) ? 0 : src2->nb[2];
        src2_nb3 = (src2->ne[3] == 1) ? 0 : src2->nb[3];
    }

    స్థిరరాశి పూర్ణాంకం dst_stride = (పూర్ణాంకం)(dst->nb[1] / పరిమాణం(దశాంశం));
    దశాంశం       * dst_ptr = (దశాంశం *) dst->దత్తాంశం + m_ప్రారంభం * dst_stride;
    // byte offset, act_stride is in activation elements (F16 or F32)
    స్థిరరాశి dma_addr_t act_addr = act->దత్తాంశం + (size_t) m_ప్రారంభం * act_stride * act_elem_కొలత;

    పూర్ణాంకం ret = -1;
    స్థిరరాశి పూర్ణాంకం n_దారాలు = kparams->n_దారాలు;
    ఒకవేళ (kparams->kernel_type == HTP_MM_KERNEL_HMX_BATCHED) {
        hmx_mm_batched_params_t batch_params = {
            .dst               = dst_ptr,
            .bias_addr         = src2_addr,
            .bias_బైట్లు        = src2_బైట్లు,
            .act_dma_addr      = act_addr,
            .బరువు            = src0->దత్తాంశం,
            .బరువు_dma        = octx->ctx->dma[0],
            .m                 = m_అడ్డువరుసలు,
            .k                 = k,
            .n                 = n,
            .act_stride        = act_stride,
            .act_elem_కొలత     = act_elem_కొలత,
            .బరువు_stride     = (పూర్ణాంకం) src0->nb[1],
            .బరువు_type       = (పూర్ణాంకం) src0->type,
            .dst_stride        = dst_stride,
            .dst_నిలువుల_సంఖ్య          = (పూర్ణాంకం) dst->ne[0],
            .bias_stride       = src2_stride,
            .tile_కొలత         = kparams->tile_కొలత,
            .aligned_tile_కొలత = kparams->aligned_tile_కొలత,
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
                             kparams->pipeline, n_దారాలు,
                             kparams->n_act_దారాలు,
                             &kparams->div_n_act_దారాలు,
                             &kparams->div_ne00_padded,
                             kparams->vtcm_కొలత);
    } లేకపోతే {
        ret = hmx_mm_2d_f32(
            octx->ctx, octx->ctx->dma[0], dst_ptr, src2_addr, src2_బైట్లు,
            act_addr, src0->దత్తాంశం,
            m_అడ్డువరుసలు, k, n, act_stride, act_elem_కొలత, (పూర్ణాంకం) src0->nb[1], (పూర్ణాంకం) src0->type, (పూర్ణాంకం) act->ne[0],
            dst_stride, src2_stride, (పూర్ణాంకం)dst->ne[0],
            kparams->m_chunk, kparams->n_chunk, kparams->pipeline, n_దారాలు,
            kparams->n_act_దారాలు,
            &kparams->div_n_act_దారాలు,
            &kparams->div_ne00_padded,
            kparams->tile_కొలత, kparams->aligned_tile_కొలత, kparams->vtcm_కొలత
        );
    }

    ఒకవేళ (ret != 0) {
        FARF(ERROR, "HMX matmul failed (ret=%d)\n", ret);
        తిరిగి HTP_స్థితి_అంతర్గత_ERR;
    }
    తిరిగి HTP_స్థితి_OK;
}

స్థిర అంతఃవరుస శూన్య htp_mm_tensor_collapse_అడ్డువరుసలు(నిర్మాణం htp_tensor * c, స్థిరరాశి నిర్మాణం htp_tensor * t, uint32_t stride) {
    *c = *t;
    c->ne[1] = t->ne[1] * t->ne[2] * t->ne[3];
    c->ne[2] = 1;
    c->ne[3] = 1;
    c->nb[1] = stride;
    c->nb[2] = c->nb[1] * c->ne[1];
    c->nb[3] = c->nb[2];
}

స్థిర పూర్ణాంకం op_matmul_impl(నిర్మాణం htp_ops_context * octx) {
    స్థిరరాశి నిర్మాణం htp_mm_kernel_params * kparams = (స్థిరరాశి నిర్మాణం htp_mm_kernel_params *) octx->kernel_params;

    స్థిరరాశి పూర్ణాంకం స్థితి_2 = htp_mm_init_context(octx, kparams);
    ఒకవేళ (స్థితి_2 != HTP_స్థితి_OK) {
        తిరిగి స్థితి_2;
    }

    ఒకవేళ (kparams->n_hmx) {
        తిరిగి hmx_mm_op_matmul(octx, kparams);
    }

    తిరిగి hvx_mm_matmul(octx);
}

పూర్ణాంకం op_matmul(నిర్మాణం htp_ops_context * octx) {
    స్థిరరాశి నిర్మాణం htp_mm_kernel_params * kparams = (స్థిరరాశి నిర్మాణం htp_mm_kernel_params *) octx->kernel_params;

    ఒకవేళ (kparams->collapse) {
        స్థిరరాశి నిర్మాణం htp_tensor * act  = octx->src[1];
        స్థిరరాశి నిర్మాణం htp_tensor * dst  = octx->dst;
        స్థిరరాశి uint32_t s_act = (act->ne[1] > 1) ? act->nb[1] : ((act->ne[2] > 1) ? act->nb[2] : act->nb[3]);
        స్థిరరాశి uint32_t sd    = (dst->ne[1] > 1) ? dst->nb[1] : ((dst->ne[2] > 1) ? dst->nb[2] : dst->nb[3]);
        నిర్మాణం htp_tensor act_collapsed, dst_collapsed;
        htp_mm_tensor_collapse_అడ్డువరుసలు(&act_collapsed, act, s_act);
        htp_mm_tensor_collapse_అడ్డువరుసలు(&dst_collapsed, dst, sd);
        octx->src[1] = &act_collapsed;
        octx->dst    = &dst_collapsed;

        నిర్మాణం htp_tensor src2_collapsed;
        స్థిరరాశి నిర్మాణం htp_tensor * src2 = octx->src[2];
        ఒకవేళ (src2 && (src2->ne[1] * src2->ne[2] * src2->ne[3] > 1)) {
            స్థిరరాశి uint32_t s2 = (src2->ne[1] > 1) ? src2->nb[1] : ((src2->ne[2] > 1) ? src2->nb[2] : src2->nb[3]);
            htp_mm_tensor_collapse_అడ్డువరుసలు(&src2_collapsed, src2, s2);
            octx->src[2] = &src2_collapsed;
        }

        స్థిరరాశి పూర్ణాంకం స్థితి_2 = op_matmul_impl(octx);
        octx->src[1] = act;
        octx->dst    = dst;
        ఒకవేళ (src2) {
            octx->src[2] = src2;
        }
        తిరిగి స్థితి_2;
    }

    తిరిగి op_matmul_impl(octx);
}

స్థిర పూర్ణాంకం hmx_mm_op_matmul_id(
    నిర్మాణం htp_ops_context * octx,
    నిర్మాణం htp_mm_context * mmctx
) {
    స్థిరరాశి uint32_t *                మాతృక_అడ్డువరుస_counts = mmctx->మాతృక_అడ్డువరుస_counts;
    స్థిరరాశి నిర్మాణం mmid_అడ్డువరుస_mapping * మాతృక_అడ్డువరుసలు       = mmctx->మాతృక_అడ్డువరుసలు;
    htp_matmul_tensors_preamble;
    స్థిరరాశి నిర్మాణం htp_mm_kernel_params * kparams = (స్థిరరాశి నిర్మాణం htp_mm_kernel_params *) octx->kernel_params;
    స్థిరరాశి పూర్ణాంకం n_ids = octx->src[2]->ne[0];
    స్థిరరాశి పూర్ణాంకం n_as  = ne02;

    స్థిరరాశి bool mdev_విభజించు = (octx->ctx->mdev.లెక్క > 1) && htp_tensor_can_అడ్డువరుస_విభజన(dst, పరిమాణం(దశాంశం));
    ఒకవేళ (octx->ctx->mdev.లెక్క > 1 && !mdev_విభజించు && octx->ctx->mdev.idx > 0) {
        తిరిగి HTP_స్థితి_OK;
    }
    uint32_t n_చురుకైన = 0;
    ఒకవేళ (mdev_విభజించు) {
        కొరకు (uint32_t a = 0; a < (uint32_t) n_as; ++a) {
            ఒకవేళ (మాతృక_అడ్డువరుస_counts[a] > 0) n_చురుకైన++;
        }
    }
    స్థిరరాశి bool expert_విభజించు = mdev_విభజించు && (n_చురుకైన >= octx->ctx->mdev.లెక్క);

    uint32_t లక్ష్యం_dev = 0;
    కొరకు (uint32_t cur_a = 0; cur_a < (uint32_t) n_as; ++cur_a) {
        స్థిరరాశి int32_t cne1 = మాతృక_అడ్డువరుస_counts[cur_a];
        ఒకవేళ (cne1 == 0) కొనసాగు;

        స్థిరరాశి పూర్ణాంకం m_padded = hex_align_పైకి(cne1, 32);
        పూర్ణాంకం m_ప్రారంభం = 0, m_చివర = m_padded;
        ఒకవేళ (expert_విభజించు) {
            స్థిరరాశి bool my_expert = (లక్ష్యం_dev == octx->ctx->mdev.idx);
            ఒకవేళ (++లక్ష్యం_dev == octx->ctx->mdev.లెక్క) {
                లక్ష్యం_dev = 0;
            }
            ఒకవేళ (!my_expert) కొనసాగు;
        } లేకపోతే ఒకవేళ (mdev_విభజించు) {
            స్థిరరాశి bool can_విభజించు = (uint32_t) cne1 >= octx->ctx->mdev.లెక్క;
            స్థిరరాశి నిర్మాణం htp_tensor_mdev_range range = htp_tensor_mdev_విభజన((uint32_t) m_padded, can_విభజించు ? 1 : 0, octx->ctx->mdev.idx, octx->ctx->mdev.లెక్క, &octx->ctx->mdev.లెక్క_div);
            m_ప్రారంభం = (పూర్ణాంకం) range.ప్రారంభం;
            m_చివర   = (పూర్ణాంకం) (range.ప్రారంభం + range.లెక్క);
        }
        ఒకవేళ (m_ప్రారంభం >= m_చివర) కొనసాగు;

        పూర్ణాంకం ret = hmx_mm_id_2d_f32(octx->ctx, octx->ctx->dma[0], (దశాంశం*) dst->దత్తాంశం, (దశాంశం*) act->దత్తాంశం,
                                   src0->దత్తాంశం + cur_a * nb02,
                                   cne1, ne00, ne01,
                                   ne10,
                                   ne11,
                                   nb11, nb12,
                                   nb1, nb2,
                                   (పూర్ణాంకం) src0->nb[1], (పూర్ణాంకం) src0->type,
                                   kparams->tile_కొలత, kparams->aligned_tile_కొలత,
                                   మాతృక_అడ్డువరుసలు, cur_a, mmctx->mapping_stride,
                                   m_ప్రారంభం, m_చివర, (పూర్ణాంకం) octx->n_దారాలు);
        ఒకవేళ (ret != 0) {
            FARF(ERROR, "HMX matmul failed for expert %u, error %d\n", cur_a, ret);
            తిరిగి HTP_స్థితి_NO_SUPPORT;
        }
    }

    తిరిగి HTP_స్థితి_OK;
}

స్థిర పూర్ణాంకం hvx_mm_matmul_id(
    నిర్మాణం htp_ops_context * octx,
    నిర్మాణం htp_mm_context * mmctx,
    work_వరుస_func_t hvx_mmid_task_func
) {
    htp_matmul_tensors_preamble;
    స్థిరరాశి uint32_t src0_అడ్డువరుస_కొలత_padded = mmctx->src0_అడ్డువరుస_కొలత_padded;
    స్థిరరాశి uint32_t act_nrows            = mmctx->act_nrows;

    నిర్మాణం htp_thread_trace * tr = &octx->ctx->trace[0];
    htp_trace_event_ప్రారంభం(tr, HTP_TRACE_EVT_INIT, 0);

    స్థిరరాశి నిర్మాణం htp_mm_kernel_params * kparams = (స్థిరరాశి నిర్మాణం htp_mm_kernel_params *) octx->kernel_params;
    స్థిరరాశి నిర్మాణం htp_tensor * పరిమిత ids = octx->src[2];
    స్థిరరాశి size_t src0_అడ్డువరుస_కొలత = nb01;

    స్థిరరాశి uint32_t qk = QK_Q8_0_TILED;
    స్థిరరాశి uint32_t nb = (ne10 + qk - 1) / qk;
    స్థిరరాశి uint32_t మొత్తం_nb = act_nrows * nb;

    work_వరుస_func_t quant_task_func;
    uint32_t n_quant_tasks = 1;
    ఒకవేళ (act_nrows < octx->n_దారాలు) {
        n_quant_tasks = MIN(మొత్తం_nb, octx->n_దారాలు);
        quant_task_func = htp_mm_act_quant_ఖండం_func(src0->type, act->type);
        కొరకు (uint32_t ith = 0; ith < n_quant_tasks; ++ith) {
            uint32_t ib_మొదటి = (మొత్తం_nb * ith) / n_quant_tasks;
            uint32_t ib_చివరి  = (మొత్తం_nb * (ith + 1)) / n_quant_tasks;
            mmctx->quant_ib_మొదటి[ith] = ib_మొదటి;
            mmctx->quant_ib_చివరి[ith]  = ib_చివరి;
            mmctx->quant_r[ith]        = ib_మొదటి / nb;
            mmctx->quant_c[ith]        = ib_మొదటి % nb;
        }
    } లేకపోతే {
        n_quant_tasks = MIN(act_nrows, octx->n_దారాలు);
        quant_task_func = htp_mm_act_quant_అడ్డువరుస_func(src0->type, act->type);
    }
    size_t act_అడ్డువరుస_కొలత = htp_mm_tiled_act_అడ్డువరుస_కొలత(src0->type, ne10);

    నిర్మాణం htp_mm_hvx_vtcm_layout L;
    htp_mm_hvx_vtcm_layout_build(&L, kparams->kernel_type, src0->type, ne10, act_nrows, octx->n_దారాలు,
                                 0, src0_అడ్డువరుస_కొలత, act_అడ్డువరుస_కొలత, 0, kparams->n_prefetch, true, false);

    స్థిరరాశి size_t vtcm_కొలత = L.మొత్తం_బైట్లు;

    FARF(ఎగువ_2, "matmul-id-%s : src0-spad-size %zu act-spad-size %zu bias-spad-size 0 dst-spad-size %zu (%zu)\n", mmctx->type,
         L.src0_బైట్లు, L.act_బైట్లు, L.dst_బైట్లు, vtcm_కొలత);

    FARF(ఎగువ_2, "matmul-id-%s : %ux%ux%ux%u * %ux%ux%ux%u (%ux%ux%ux%u) -> %ux%ux%ux%u (0x%p, 0x%p, 0x%p)\n", mmctx->type,
         src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3], act->ne[0], act->ne[1], act->ne[2], act->ne[3],
         ids->ne[0], ids->ne[1], ids->ne[2], ids->ne[3], dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3], src0->దత్తాంశం,
         act->దత్తాంశం, dst->దత్తాంశం);

    // Make sure the reserved vtcm size is sufficient
    ఒకవేళ (octx->ctx->vtcm_కొలత < vtcm_కొలత) {
        FARF(ERROR, "matmul-id-%s : current VTCM reservation %zu is too small, needed %zu\n", mmctx->type, octx->ctx->vtcm_కొలత, vtcm_కొలత);
        తిరిగి HTP_స్థితి_VTCM_TOO_SMALL;
    }

    uint8_t * స్థిరరాశి పునాది = (uint8_t *) octx->ctx->vtcm_పునాది;
    mmctx->vtcm_act      = VTCM_LAYOUT_PTR(uint8_t, పునాది, L.off_act);
    mmctx->vtcm_src0     = VTCM_LAYOUT_PTR(uint8_t, పునాది, L.off_src0);
    mmctx->vtcm_bias     = ఖాళీ;
    mmctx->vtcm_dst      = VTCM_LAYOUT_PTR(uint8_t, పునాది, L.off_dst);
    mmctx->vtcm_act_raw  = VTCM_LAYOUT_PTR(uint8_t, పునాది, L.off_act_raw);

    mmctx->vtcm_src0_stride    = src0_అడ్డువరుస_కొలత_padded;
    mmctx->vtcm_act_stride     = act_అడ్డువరుస_కొలత;
    mmctx->vtcm_act_raw_stride = hex_విడత_పైకి(ne10 * పరిమాణం(దశాంశం), QK_Q8_0_TILED * పరిమాణం(దశాంశం));

    mmctx->vtcm_src0_కొలత_చొప్పున_thread = fastdiv(L.src0_బైట్లు, &octx->n_దారాలు_div);
    mmctx->vtcm_act_కొలత_చొప్పున_thread  = L.act_బైట్లు;
    mmctx->vtcm_dst_కొలత_చొప్పున_thread  = fastdiv(L.dst_బైట్లు, &octx->n_దారాలు_div);

    mmctx->cur_m_ప్రారంభం = 0;
    mmctx->cur_m_అడ్డువరుసలు  = act_nrows;

    htp_trace_event_stop(tr, HTP_TRACE_EVT_INIT, 0);

    hvx_mm_transfer_act_dma(octx, kparams, act, mmctx->vtcm_act_raw, mmctx->vtcm_act_raw_stride, 0, act_nrows);

    mmctx->n_quant_అడ్డువరుసలు_చొప్పున_thread = (act_nrows + n_quant_tasks - 1) / n_quant_tasks;
    mmctx->n_quant_tasks = n_quant_tasks;
    work_వరుస_నిరంతరం(octx->ctx->work_వరుస, quant_task_func, mmctx, n_quant_tasks);

    work_వరుస_నిరంతరం(octx->ctx->work_వరుస, hvx_mmid_task_func, mmctx, octx->n_దారాలు);

    తిరిగి HTP_స్థితి_OK;
}

స్థిర పూర్ణాంకం hmx_mm_op_matmul_id_nx(
    నిర్మాణం htp_ops_context * octx,
    నిర్మాణం htp_mm_context * mmctx
) {
    స్థిరరాశి uint32_t *                మాతృక_అడ్డువరుస_counts = mmctx->మాతృక_అడ్డువరుస_counts;
    స్థిరరాశి నిర్మాణం mmid_అడ్డువరుస_mapping * మాతృక_అడ్డువరుసలు       = mmctx->మాతృక_అడ్డువరుసలు;
    స్థిరరాశి నిర్మాణం htp_mm_kernel_params * kparams = (స్థిరరాశి నిర్మాణం htp_mm_kernel_params *) octx->kernel_params;
    స్థిరరాశి uint32_t n_weights = kparams->n_weights;
    స్థిరరాశి నిర్మాణం htp_tensor * పరిమిత src0 = octx->src[0];
    స్థిరరాశి నిర్మాణం htp_tensor * పరిమిత act  = octx->src[n_weights];
    స్థిరరాశి పూర్ణాంకం n_as = src0->ne[2];

    bool mdev_విభజించు = (octx->ctx->mdev.లెక్క > 1);
    కొరకు (uint32_t p = 0; p < n_weights && mdev_విభజించు; ++p) {
        స్థిరరాశి నిర్మాణం htp_tensor * పరిమిత dst = octx->dsts[p];
        mdev_విభజించు = htp_tensor_can_అడ్డువరుస_విభజన(dst, పరిమాణం(దశాంశం));
    }
    ఒకవేళ (octx->ctx->mdev.లెక్క > 1 && !mdev_విభజించు && octx->ctx->mdev.idx > 0) {
        తిరిగి HTP_స్థితి_OK;
    }
    uint32_t n_చురుకైన = 0;
    ఒకవేళ (mdev_విభజించు) {
        కొరకు (uint32_t a = 0; a < (uint32_t) n_as; ++a) {
            ఒకవేళ (మాతృక_అడ్డువరుస_counts[a] > 0) n_చురుకైన++;
        }
    }
    స్థిరరాశి bool expert_విభజించు = mdev_విభజించు && (n_చురుకైన >= octx->ctx->mdev.లెక్క);

    uint32_t లక్ష్యం_dev = 0;
    కొరకు (uint32_t cur_a = 0; cur_a < (uint32_t) n_as; ++cur_a) {
        స్థిరరాశి int32_t cne1 = మాతృక_అడ్డువరుస_counts[cur_a];
        ఒకవేళ (cne1 == 0) కొనసాగు;

        స్థిరరాశి పూర్ణాంకం m_padded = hex_align_పైకి(cne1, 32);
        పూర్ణాంకం m_ప్రారంభం = 0, m_చివర = m_padded;
        ఒకవేళ (expert_విభజించు) {
            స్థిరరాశి bool my_expert = (లక్ష్యం_dev == octx->ctx->mdev.idx);
            ఒకవేళ (++లక్ష్యం_dev == octx->ctx->mdev.లెక్క) {
                లక్ష్యం_dev = 0;
            }
            ఒకవేళ (!my_expert) కొనసాగు;
        } లేకపోతే ఒకవేళ (mdev_విభజించు) {
            స్థిరరాశి bool can_విభజించు = (uint32_t) cne1 >= octx->ctx->mdev.లెక్క;
            స్థిరరాశి నిర్మాణం htp_tensor_mdev_range range = htp_tensor_mdev_విభజన((uint32_t) m_padded, can_విభజించు ? 1 : 0, octx->ctx->mdev.idx, octx->ctx->mdev.లెక్క, &octx->ctx->mdev.లెక్క_div);
            m_ప్రారంభం = (పూర్ణాంకం) range.ప్రారంభం;
            m_చివర   = (పూర్ణాంకం) (range.ప్రారంభం + range.లెక్క);
        }
        ఒకవేళ (m_ప్రారంభం >= m_చివర) కొనసాగు;

        కొరకు (uint32_t p = 0; p < n_weights; ++p) {
            స్థిరరాశి నిర్మాణం htp_tensor * పరిమిత src_w = octx->src[p];
            స్థిరరాశి నిర్మాణం htp_tensor * పరిమిత dst   = octx->dsts[p];
            ఒకవేళ (!src_w || !dst) కొనసాగు;

            పూర్ణాంకం ret = hmx_mm_id_2d_f32(octx->ctx, octx->ctx->dma[0], (దశాంశం*) dst->దత్తాంశం, (దశాంశం*) act->దత్తాంశం,
                                       src_w->దత్తాంశం + cur_a * src_w->nb[2],
                                       cne1, src_w->ne[0], src_w->ne[1],
                                       act->ne[0],
                                       act->ne[1],
                                       act->nb[1], act->nb[2],
                                       dst->nb[1], dst->nb[2],
                                       (పూర్ణాంకం) src_w->nb[1], (పూర్ణాంకం) src_w->type,
                                       kparams->tile_కొలత, kparams->aligned_tile_కొలత,
                                       మాతృక_అడ్డువరుసలు, cur_a, mmctx->mapping_stride,
                                       m_ప్రారంభం, m_చివర, (పూర్ణాంకం) octx->n_దారాలు);
            ఒకవేళ (ret != 0) {
                FARF(ERROR, "HMX matmul ID NX failed for expert %u weight %u, error %d\n", cur_a, p, ret);
                తిరిగి HTP_స్థితి_NO_SUPPORT;
            }
        }
    }

    తిరిగి HTP_స్థితి_OK;
}

స్థిర పూర్ణాంకం hvx_mm_matmul_id_nx(
    నిర్మాణం htp_ops_context * octx,
    నిర్మాణం htp_mm_context * mmctx,
    work_వరుస_func_t hvx_mmid_task_func
) {
    స్థిరరాశి uint32_t src0_అడ్డువరుస_కొలత_padded = mmctx->src0_అడ్డువరుస_కొలత_padded;
    స్థిరరాశి uint32_t act_nrows            = mmctx->act_nrows;

    నిర్మాణం htp_thread_trace * tr = &octx->ctx->trace[0];
    htp_trace_event_ప్రారంభం(tr, HTP_TRACE_EVT_INIT, 0);

    స్థిరరాశి నిర్మాణం htp_mm_kernel_params * kparams = (స్థిరరాశి నిర్మాణం htp_mm_kernel_params *) octx->kernel_params;
    స్థిరరాశి uint32_t n_weights = kparams->n_weights;
    స్థిరరాశి నిర్మాణం htp_tensor * పరిమిత src0 = octx->src[0];
    స్థిరరాశి నిర్మాణం htp_tensor * పరిమిత act  = octx->src[n_weights];
    స్థిరరాశి నిర్మాణం htp_tensor * పరిమిత ids  = octx->src[n_weights + 1];
    స్థిరరాశి size_t src0_అడ్డువరుస_కొలత = src0->nb[1];

    స్థిరరాశి uint32_t qk = QK_Q8_0_TILED;
    స్థిరరాశి uint32_t nb = (act->ne[0] + qk - 1) / qk;
    స్థిరరాశి uint32_t మొత్తం_nb = act_nrows * nb;

    work_వరుస_func_t quant_task_func;
    uint32_t n_quant_tasks = 1;
    ఒకవేళ (act_nrows < octx->n_దారాలు) {
        n_quant_tasks = MIN(మొత్తం_nb, octx->n_దారాలు);
        quant_task_func = htp_mm_act_quant_ఖండం_func(src0->type, act->type);
        కొరకు (uint32_t ith = 0; ith < n_quant_tasks; ++ith) {
            uint32_t ib_మొదటి = (మొత్తం_nb * ith) / n_quant_tasks;
            uint32_t ib_చివరి  = (మొత్తం_nb * (ith + 1)) / n_quant_tasks;
            mmctx->quant_ib_మొదటి[ith] = ib_మొదటి;
            mmctx->quant_ib_చివరి[ith]  = ib_చివరి;
            mmctx->quant_r[ith]        = ib_మొదటి / nb;
            mmctx->quant_c[ith]        = ib_మొదటి % nb;
        }
    } లేకపోతే {
        n_quant_tasks = MIN(act_nrows, octx->n_దారాలు);
        quant_task_func = htp_mm_act_quant_అడ్డువరుస_func(src0->type, act->type);
    }
    size_t act_అడ్డువరుస_కొలత = htp_mm_tiled_act_అడ్డువరుస_కొలత(src0->type, act->ne[0]);

    నిర్మాణం htp_mm_hvx_vtcm_layout L;
    htp_mm_hvx_vtcm_layout_build(&L, kparams->kernel_type, src0->type, act->ne[0], act_nrows, octx->n_దారాలు,
                                 0, src0_అడ్డువరుస_కొలత, act_అడ్డువరుస_కొలత, 0, kparams->n_prefetch, true, false);

    స్థిరరాశి size_t vtcm_కొలత = L.మొత్తం_బైట్లు;

    ఒకవేళ (octx->ctx->vtcm_కొలత < vtcm_కొలత) {
        FARF(ERROR, "matmul-id-nx: current VTCM reservation %zu is too small, needed %zu\n",
             octx->ctx->vtcm_కొలత, vtcm_కొలత);
        తిరిగి HTP_స్థితి_VTCM_TOO_SMALL;
    }

    uint8_t * స్థిరరాశి పునాది = (uint8_t *) octx->ctx->vtcm_పునాది;
    mmctx->vtcm_src0     = VTCM_LAYOUT_PTR(uint8_t, పునాది, L.off_src0);
    mmctx->vtcm_act      = VTCM_LAYOUT_PTR(uint8_t, పునాది, L.off_act);
    mmctx->vtcm_dst      = VTCM_LAYOUT_PTR(uint8_t, పునాది, L.off_dst);
    mmctx->vtcm_act_raw  = VTCM_LAYOUT_PTR(uint8_t, పునాది, L.off_act_raw);

    mmctx->vtcm_src0_stride    = 0;
    mmctx->vtcm_act_stride     = act_అడ్డువరుస_కొలత;
    mmctx->vtcm_act_raw_stride = hex_విడత_పైకి(act->ne[0] * పరిమాణం(దశాంశం), QK_Q8_0_TILED * పరిమాణం(దశాంశం));

    mmctx->vtcm_src0_కొలత_చొప్పున_thread = fastdiv(L.src0_బైట్లు, &octx->n_దారాలు_div);
    mmctx->vtcm_act_కొలత_చొప్పున_thread  = L.act_బైట్లు;
    mmctx->vtcm_dst_కొలత_చొప్పున_thread  = fastdiv(L.dst_బైట్లు, &octx->n_దారాలు_div);

    mmctx->cur_m_ప్రారంభం = 0;
    mmctx->cur_m_అడ్డువరుసలు  = act_nrows;

    FARF(ఎగువ_2, "matmul-id-nx: src0 %d:%d:%d type %s nrows %u, act %d:%d:%d nrows %u, vtcm %zu/%zu, threads %d\n",
         src0->ne[0], src0->ne[1], src0->ne[2], mmctx->type, src0->ne[1],
         act->ne[0], act->ne[1], act->ne[2], act_nrows,
         L.మొత్తం_బైట్లు, octx->ctx->vtcm_కొలత, octx->n_దారాలు);

    htp_trace_event_stop(tr, HTP_TRACE_EVT_INIT, 0);

    hvx_mm_transfer_act_dma(octx, kparams, act, mmctx->vtcm_act_raw, mmctx->vtcm_act_raw_stride, 0, act_nrows);

    mmctx->n_quant_అడ్డువరుసలు_చొప్పున_thread = (act_nrows + n_quant_tasks - 1) / n_quant_tasks;
    mmctx->n_quant_tasks           = n_quant_tasks;
    work_వరుస_నిరంతరం(octx->ctx->work_వరుస, quant_task_func, mmctx, n_quant_tasks);

    work_వరుస_నిరంతరం(octx->ctx->work_వరుస, hvx_mmid_task_func, mmctx, octx->n_దారాలు);

    తిరిగి HTP_స్థితి_OK;
}

స్థిర అంతఃవరుస శూన్య పరిశీలన_expert_ids_n(
    స్థిరరాశి నిర్మాణం htp_tensor * ids,
    స్థిరరాశి uint32_t n_ids,
    uint32_t n_as,
    uint32_t * counts,
    నిర్మాణం mmid_అడ్డువరుస_mapping * మాతృక_అడ్డువరుసలు,
    uint32_t mapping_stride
) {
    స్థిరరాశి size_t ids_nb1 = ids->nb[1];
    స్థిరరాశి uint8_t * ids_దత్తాంశం = (స్థిరరాశి uint8_t *) ids->దత్తాంశం;

    కొరకు (uint32_t iid1 = 0; iid1 < ids->ne[1]; ++iid1) {
        స్థిరరాశి int32_t * అడ్డువరుస_ptr = (స్థిరరాశి int32_t *) (ids_దత్తాంశం + iid1 * ids_nb1);
        కొరకు (uint32_t id = 0; id < n_ids; ++id) {
            స్థిరరాశి int32_t i02 = అడ్డువరుస_ptr[id];
            ఒకవేళ (i02 < 0) {
                కొనసాగు;
            }
            ప్రతిజ్ఞా(i02 < n_as);

            ఒకవేళ (మాతృక_అడ్డువరుసలు) {
                మాతృక_అడ్డువరుసలు[i02 * mapping_stride + counts[i02]] = (నిర్మాణం mmid_అడ్డువరుస_mapping) { id, iid1 };
            }
            counts[i02] += 1;
        }
    }
}

స్థిర అంతఃవరుస శూన్య పరిశీలన_expert_ids(
    స్థిరరాశి నిర్మాణం htp_tensor * ids,
    uint32_t n_ids,
    uint32_t n_as,
    uint32_t * counts,
    నిర్మాణం mmid_అడ్డువరుస_mapping * మాతృక_అడ్డువరుసలు,
    uint32_t mapping_stride
) {
    స్థిరరాశి size_t ids_nb0 = ids->nb[0];

    ఒకవేళ (ids_nb0 == 4) {
        ఎంపిక (n_ids) {
            స్థితి 8:  పరిశీలన_expert_ids_n(ids, 8,     n_as, counts, మాతృక_అడ్డువరుసలు, mapping_stride); ఆపు;
            స్థితి 4:  పరిశీలన_expert_ids_n(ids, 4,     n_as, counts, మాతృక_అడ్డువరుసలు, mapping_stride); ఆపు;
            స్థితి 2:  పరిశీలన_expert_ids_n(ids, 2,     n_as, counts, మాతృక_అడ్డువరుసలు, mapping_stride); ఆపు;
            అప్రమేయ: పరిశీలన_expert_ids_n(ids, n_ids, n_as, counts, మాతృక_అడ్డువరుసలు, mapping_stride); ఆపు;
        }
    } లేకపోతే {
        // Strided fallback
        స్థిరరాశి size_t ids_nb1 = ids->nb[1];
        స్థిరరాశి uint8_t * ids_దత్తాంశం = (స్థిరరాశి uint8_t *) ids->దత్తాంశం;
        కొరకు (uint32_t iid1 = 0; iid1 < ids->ne[1]; ++iid1) {
            స్థిరరాశి int32_t * అడ్డువరుస_ptr = (స్థిరరాశి int32_t *) (ids_దత్తాంశం + iid1 * ids_nb1);
            కొరకు (uint32_t id = 0; id < n_ids; ++id) {
                స్థిరరాశి int32_t i02 = *(స్థిరరాశి int32_t *) ((స్థిరరాశి uint8_t *) అడ్డువరుస_ptr + id * ids_nb0);
                ఒకవేళ (i02 < 0) {
                    కొనసాగు;
                }
                ప్రతిజ్ఞా(i02 < n_as);

                ఒకవేళ (మాతృక_అడ్డువరుసలు) {
                    మాతృక_అడ్డువరుసలు[i02 * mapping_stride + counts[i02]] = (నిర్మాణం mmid_అడ్డువరుస_mapping) { id, iid1 };
                }
                counts[i02] += 1;
            }
        }
    }
}

పూర్ణాంకం op_matmul_id(నిర్మాణం htp_ops_context * octx) {
    htp_matmul_tensors_preamble;

    స్థిరరాశి నిర్మాణం htp_mm_kernel_params * kparams = (స్థిరరాశి నిర్మాణం htp_mm_kernel_params *) octx->kernel_params;
    నిర్మాణం htp_mm_context mmctx_struct = {0};
    నిర్మాణం htp_mm_context * mmctx = &mmctx_struct;

    స్థిరరాశి పూర్ణాంకం స్థితి_2 = htp_mm_init_context(octx, kparams);
    ఒకవేళ (స్థితి_2 != HTP_స్థితి_OK) {
        తిరిగి స్థితి_2;
    }

    నిర్మాణం htp_thread_trace * tr = &octx->ctx->trace[0];
    htp_trace_event_ప్రారంభం(tr, HTP_TRACE_EVT_INIT, 0);

    mmctx->octx = octx;
    mmctx->act = act;

    స్థిరరాశి నిర్మాణం htp_tensor * పరిమిత ids = octx->src[2];
    ఒకవేళ (htp_tensor_is_extended(ids) || htp_tensor_is_extended(act) || htp_tensor_is_extended(dst)) {
        తిరిగి HTP_స్థితి_NO_SUPPORT;
    }

    స్థిరరాశి size_t src0_అడ్డువరుస_కొలత = nb01;
    స్థిరరాశి size_t dst_అడ్డువరుస_కొలత  = nb1;

    స్థిరరాశి size_t src0_అడ్డువరుస_కొలత_padded = hex_విడత_పైకి(src0_అడ్డువరుస_కొలత, 128);

    స్థిరరాశి uint32_t src0_nrows = ne01;  // per expert
    స్థిరరాశి uint32_t act_nrows  = ne11 * ne12 * ne13;

    // row groups
    స్థిరరాశి పూర్ణాంకం n_ids = ids->ne[0];  // n_expert_used
    స్థిరరాశి పూర్ణాంకం n_as  = ne02;        // n_expert

    uint8_t  * mapping_buf       = octx->ctx->ddr_spad_పునాది;
    uint32_t   mapping_stride    = 1;
    uint32_t * మాతృక_అడ్డువరుస_counts = (uint32_t *) mapping_buf;
    నిర్మాణం mmid_అడ్డువరుస_mapping * మాతృక_అడ్డువరుసలు = ఖాళీ;

    ఒకవేళ (act_nrows > 1) {
        స్థిరరాశి size_t మాతృక_అడ్డువరుస_counts_కొలత = n_as * పరిమాణం(uint32_t);
        ప్రతిజ్ఞా(octx->ctx->ddr_spad_కొలత >= మాతృక_అడ్డువరుస_counts_కొలత);

        hex_l2fetch_ఖండం((స్థిరరాశి శూన్య *) ids->దత్తాంశం, ids->ne[1] * ids->nb[1]);

        memset(మాతృక_అడ్డువరుస_counts, 0, మాతృక_అడ్డువరుస_counts_కొలత);
        పరిశీలన_expert_ids(ids, n_ids, n_as, మాతృక_అడ్డువరుస_counts, ఖాళీ, 0);

        uint32_t max_లెక్క = hvx_reduce_max_i32((స్థిరరాశి uint8_t *) మాతృక_అడ్డువరుస_counts, n_as);
        mapping_stride = max_లెక్క > 0 ? max_లెక్క : 1;

        size_t మాతృక_అడ్డువరుస_map_కొలత  = n_as * mapping_stride * పరిమాణం(నిర్మాణం mmid_అడ్డువరుస_mapping);
        స్థిరరాశి size_t మొత్తం_map_కొలత = మాతృక_అడ్డువరుస_counts_కొలత + మాతృక_అడ్డువరుస_map_కొలత;

        ఒకవేళ (మొత్తం_map_కొలత > octx->ctx->ddr_spad_కొలత) {
            mapping_buf = memalign(128, మొత్తం_map_కొలత);
            ఒకవేళ (!mapping_buf) {
                తిరిగి HTP_స్థితి_అంతర్గత_ERR;
            }
        }

        మాతృక_అడ్డువరుస_counts = (uint32_t *) mapping_buf;
        మాతృక_అడ్డువరుసలు       = (నిర్మాణం mmid_అడ్డువరుస_mapping *) (mapping_buf + మాతృక_అడ్డువరుస_counts_కొలత);

        memset(మాతృక_అడ్డువరుస_counts, 0, n_as * పరిమాణం(uint32_t));
        పరిశీలన_expert_ids(ids, n_ids, n_as, మాతృక_అడ్డువరుస_counts, మాతృక_అడ్డువరుసలు, mapping_stride);
    }

    mmctx->మాతృక_అడ్డువరుస_counts    = మాతృక_అడ్డువరుస_counts;
    mmctx->మాతృక_అడ్డువరుసలు          = మాతృక_అడ్డువరుసలు;
    mmctx->mapping_stride       = mapping_stride;
    mmctx->mm_div_ne11          = kparams->div_ne1;
    mmctx->src0_అడ్డువరుస_కొలత_padded = src0_అడ్డువరుస_కొలత_padded;
    mmctx->act_nrows            = act_nrows;
    mmctx->cur_m_ప్రారంభం          = 0;
    mmctx->cur_m_అడ్డువరుసలు           = act_nrows;

    htp_trace_event_stop(tr, HTP_TRACE_EVT_INIT, 0);

    పూర్ణాంకం s;
    ఒకవేళ (kparams->n_hmx) {
        s = hmx_mm_op_matmul_id(octx, mmctx);
    } లేకపోతే {
        uint32_t src0_అడ్డువరుస_ప్రారంభం = 0;
        uint32_t src0_అడ్డువరుస_చివర   = src0_nrows;
        ఒకవేళ (octx->ctx->mdev.లెక్క > 1) {
            స్థిరరాశి bool can_విభజించు = htp_tensor_can_అడ్డువరుస_విభజన(dst, పరిమాణం(దశాంశం));
            స్థిరరాశి నిర్మాణం htp_tensor_mdev_range range = htp_tensor_mdev_విభజన(src0_nrows, can_విభజించు ? 32 : 0, octx->ctx->mdev.idx, octx->ctx->mdev.లెక్క, &octx->ctx->mdev.లెక్క_div);
            src0_అడ్డువరుస_ప్రారంభం = range.ప్రారంభం;
            src0_అడ్డువరుస_చివర   = range.ప్రారంభం + range.లెక్క;
        }

        ఒకవేళ (src0_అడ్డువరుస_ప్రారంభం >= src0_అడ్డువరుస_చివర) {
            ఒకవేళ (mapping_buf != octx->ctx->ddr_spad_పునాది) {
                విడుదల(mapping_buf);
            }
            తిరిగి HTP_స్థితి_OK;
        }

        స్థిరరాశి uint32_t nrows = src0_అడ్డువరుస_చివర - src0_అడ్డువరుస_ప్రారంభం;
        mmctx->src0_అడ్డువరుస_ప్రారంభం = src0_అడ్డువరుస_ప్రారంభం;
        mmctx->src0_అడ్డువరుస_చివర   = src0_అడ్డువరుస_చివర;

        mmctx->src0_nrows_చొప్పున_thread = fastdiv(nrows + octx->n_దారాలు - 1, &octx->n_దారాలు_div);
        mmctx->src0_nrows_చొప్పున_thread = hex_విడత_పైకి(mmctx->src0_nrows_చొప్పున_thread, 32);

        ఒకవేళ (hvx_mm_init_tiled_vec_dot(mmctx, src0->type) == 0) {
            s = hvx_mm_matmul_id(octx, mmctx, act_nrows > 1 ? hvx_mm_id_tiled : hvx_mv_id_tiled);
        } లేకపోతే {
            s = HTP_స్థితి_NO_SUPPORT;
        }
    }

    ఒకవేళ (mapping_buf != octx->ctx->ddr_spad_పునాది) {
        విడుదల(mapping_buf);
    }

    తిరిగి s;
}

పూర్ణాంకం op_matmul_id_nx(నిర్మాణం htp_ops_context * octx) {
    స్థిరరాశి నిర్మాణం htp_mm_kernel_params * kparams = (స్థిరరాశి నిర్మాణం htp_mm_kernel_params *) octx->kernel_params;
    నిర్మాణం htp_mm_context mmctx_struct = {0};
    నిర్మాణం htp_mm_context * mmctx = &mmctx_struct;

    స్థిరరాశి పూర్ణాంకం స్థితి_2 = htp_mm_init_context(octx, kparams);
    ఒకవేళ (స్థితి_2 != HTP_స్థితి_OK) {
        తిరిగి స్థితి_2;
    }

    నిర్మాణం htp_thread_trace * tr = &octx->ctx->trace[0];
    htp_trace_event_ప్రారంభం(tr, HTP_TRACE_EVT_INIT, 0);

    mmctx->octx = octx;
    స్థిరరాశి uint32_t n_weights = kparams->n_weights;
    స్థిరరాశి నిర్మాణం htp_tensor * పరిమిత src0 = octx->src[0];
    స్థిరరాశి నిర్మాణం htp_tensor * పరిమిత act  = octx->src[n_weights];
    స్థిరరాశి నిర్మాణం htp_tensor * పరిమిత ids  = octx->src[n_weights + 1];
    ఒకవేళ (htp_tensor_is_extended(ids) || htp_tensor_is_extended(act)) {
        తిరిగి HTP_స్థితి_NO_SUPPORT;
    }
    కొరకు (uint32_t p = 0; p < n_weights; p++) {
        ఒకవేళ (htp_tensor_is_extended(octx->dsts[p])) {
            తిరిగి HTP_స్థితి_NO_SUPPORT;
        }
    }

    mmctx->act = act;

    స్థిరరాశి size_t src0_అడ్డువరుస_కొలత = src0->nb[1];
    స్థిరరాశి size_t src0_అడ్డువరుస_కొలత_padded = hex_విడత_పైకి(src0_అడ్డువరుస_కొలత, 128);

    స్థిరరాశి uint32_t act_nrows = act->ne[1] * act->ne[2] * act->ne[3];

    స్థిరరాశి పూర్ణాంకం n_ids = ids->ne[0];
    స్థిరరాశి పూర్ణాంకం n_as  = src0->ne[2];

    uint8_t  * mapping_buf       = octx->ctx->ddr_spad_పునాది;
    uint32_t   mapping_stride    = 1;
    uint32_t * మాతృక_అడ్డువరుస_counts = (uint32_t *) mapping_buf;
    నిర్మాణం mmid_అడ్డువరుస_mapping * మాతృక_అడ్డువరుసలు = ఖాళీ;

    ఒకవేళ (act_nrows > 1) {
        స్థిరరాశి size_t మాతృక_అడ్డువరుస_counts_కొలత = n_as * పరిమాణం(uint32_t);
        ప్రతిజ్ఞా(octx->ctx->ddr_spad_కొలత >= మాతృక_అడ్డువరుస_counts_కొలత);

        hex_l2fetch_ఖండం((స్థిరరాశి శూన్య *) ids->దత్తాంశం, ids->ne[1] * ids->nb[1]);

        memset(మాతృక_అడ్డువరుస_counts, 0, మాతృక_అడ్డువరుస_counts_కొలత);
        పరిశీలన_expert_ids(ids, n_ids, n_as, మాతృక_అడ్డువరుస_counts, ఖాళీ, 0);

        uint32_t max_లెక్క = hvx_reduce_max_i32((స్థిరరాశి uint8_t *) మాతృక_అడ్డువరుస_counts, n_as);
        mapping_stride = max_లెక్క > 0 ? max_లెక్క : 1;

        size_t మాతృక_అడ్డువరుస_map_కొలత  = n_as * mapping_stride * పరిమాణం(నిర్మాణం mmid_అడ్డువరుస_mapping);
        స్థిరరాశి size_t మొత్తం_map_కొలత = మాతృక_అడ్డువరుస_counts_కొలత + మాతృక_అడ్డువరుస_map_కొలత;

        ఒకవేళ (మొత్తం_map_కొలత > octx->ctx->ddr_spad_కొలత) {
            mapping_buf = memalign(128, మొత్తం_map_కొలత);
            ఒకవేళ (!mapping_buf) {
                తిరిగి HTP_స్థితి_అంతర్గత_ERR;
            }
        }

        మాతృక_అడ్డువరుస_counts = (uint32_t *) mapping_buf;
        మాతృక_అడ్డువరుసలు       = (నిర్మాణం mmid_అడ్డువరుస_mapping *) (mapping_buf + మాతృక_అడ్డువరుస_counts_కొలత);

        memset(మాతృక_అడ్డువరుస_counts, 0, n_as * పరిమాణం(uint32_t));
        పరిశీలన_expert_ids(ids, n_ids, n_as, మాతృక_అడ్డువరుస_counts, మాతృక_అడ్డువరుసలు, mapping_stride);
    }

    mmctx->మాతృక_అడ్డువరుస_counts    = మాతృక_అడ్డువరుస_counts;
    mmctx->మాతృక_అడ్డువరుసలు          = మాతృక_అడ్డువరుసలు;
    mmctx->mapping_stride       = mapping_stride;
    mmctx->mm_div_ne11          = kparams->div_ne1;
    mmctx->src0_అడ్డువరుస_కొలత_padded = src0_అడ్డువరుస_కొలత_padded;
    mmctx->act_nrows            = act_nrows;
    mmctx->cur_m_ప్రారంభం          = 0;
    mmctx->cur_m_అడ్డువరుసలు           = act_nrows;

    htp_trace_event_stop(tr, HTP_TRACE_EVT_INIT, 0);

    పూర్ణాంకం s;
    ఒకవేళ (kparams->n_hmx) {
        s = hmx_mm_op_matmul_id_nx(octx, mmctx);
    } లేకపోతే {
        ఒకవేళ (hvx_mm_init_tiled_vec_dot(mmctx, src0->type) == 0) {
            s = hvx_mm_matmul_id_nx(octx, mmctx, act_nrows > 1 ? hvx_mm_id_nx_tiled : hvx_mv_id_nx_tiled);
        } లేకపోతే {
            s = HTP_స్థితి_NO_SUPPORT;
        }
    }

    ఒకవేళ (mapping_buf != octx->ctx->ddr_spad_పునాది) {
        విడుదల(mapping_buf);
    }

    తిరిగి s;
}
స్థిర పూర్ణాంకం op_matmul_nx_impl(నిర్మాణం htp_ops_context * octx) {
    స్థిరరాశి నిర్మాణం htp_mm_kernel_params * kparams = (స్థిరరాశి నిర్మాణం htp_mm_kernel_params *) octx->kernel_params;

    స్థిరరాశి పూర్ణాంకం స్థితి_2 = htp_mm_init_context(octx, kparams);
    ఒకవేళ (స్థితి_2 != HTP_స్థితి_OK) {
        తిరిగి స్థితి_2;
    }

    ఒకవేళ (kparams->n_hmx) {
        తిరిగి hmx_mm_nx_2d_f32(octx, kparams);
    }

    నిర్మాణం htp_thread_trace * tr = &octx->ctx->trace[0];
    htp_trace_event_ప్రారంభం(tr, HTP_TRACE_EVT_INIT, 0);

    స్థిరరాశి uint32_t n_weights = kparams->n_weights;

    స్థిరరాశి నిర్మాణం htp_tensor * పరిమిత src0 = octx->src[0]; // first weight
    స్థిరరాశి నిర్మాణం htp_tensor * పరిమిత act  = octx->src[n_weights]; // activation x

    నిర్మాణం htp_mm_context mmctx_struct = {0};
    నిర్మాణం htp_mm_context * mmctx = &mmctx_struct;
    mmctx->octx = octx;
    mmctx->act  = act;

    స్థిరరాశి uint32_t act_nrows = act->ne[1] * act->ne[2] * act->ne[3];
    mmctx->act_nrows   = act_nrows;
    mmctx->cur_m_ప్రారంభం = 0;
    mmctx->cur_m_అడ్డువరుసలు  = act_nrows;

    స్థిరరాశి size_t src0_అడ్డువరుస_కొలత = src0->nb[1];

    ఒకవేళ (hvx_mm_init_tiled_vec_dot(mmctx, src0->type) != 0) {
        తిరిగి HTP_స్థితి_NO_SUPPORT;
    }

    స్థిరరాశి uint32_t qk = QK_Q8_0_TILED;
    స్థిరరాశి uint32_t nb = (act->ne[0] + qk - 1) / qk;
    స్థిరరాశి uint32_t మొత్తం_nb = act_nrows * nb;

    కార్మికుడు_callback_t quant_task_func;
    uint32_t n_quant_tasks = 1;
    ఒకవేళ (act_nrows < octx->n_దారాలు) {
        n_quant_tasks = MIN(మొత్తం_nb, octx->n_దారాలు);
        quant_task_func = htp_mm_act_quant_ఖండం_func(src0->type, act->type);
        కొరకు (uint32_t ith = 0; ith < n_quant_tasks; ++ith) {
            uint32_t ib_మొదటి = (మొత్తం_nb * ith) / n_quant_tasks;
            uint32_t ib_చివరి  = (మొత్తం_nb * (ith + 1)) / n_quant_tasks;
            mmctx->quant_ib_మొదటి[ith] = ib_మొదటి;
            mmctx->quant_ib_చివరి[ith]  = ib_చివరి;
            mmctx->quant_r[ith]        = ib_మొదటి / nb;
            mmctx->quant_c[ith]        = ib_మొదటి % nb;
        }
    } లేకపోతే {
        n_quant_tasks = MIN(act_nrows, octx->n_దారాలు);
        quant_task_func = htp_mm_act_quant_అడ్డువరుస_func(src0->type, act->type);
    }

    స్థిరరాశి size_t act_అడ్డువరుస_కొలత = htp_mm_tiled_act_అడ్డువరుస_కొలత(src0->type, act->ne[0]);

    నిర్మాణం htp_mm_hvx_vtcm_layout L;
    htp_mm_hvx_vtcm_layout_build(&L, kparams->kernel_type, src0->type, act->ne[0], act_nrows, octx->n_దారాలు,
                                 0, src0_అడ్డువరుస_కొలత, act_అడ్డువరుస_కొలత, 0, kparams->n_prefetch, false, true);

    స్థిరరాశి size_t vtcm_కొలత = L.మొత్తం_బైట్లు;

    ఒకవేళ (octx->ctx->vtcm_కొలత < vtcm_కొలత) {
        FARF(ERROR, "matmul-nx: current VTCM reservation %zu is too small, needed %zu\n",
             octx->ctx->vtcm_కొలత, vtcm_కొలత);
        తిరిగి HTP_స్థితి_VTCM_TOO_SMALL;
    }

    uint8_t * స్థిరరాశి పునాది = (uint8_t *) octx->ctx->vtcm_పునాది;
    mmctx->vtcm_src0     = VTCM_LAYOUT_PTR(uint8_t, పునాది, L.off_src0);
    mmctx->vtcm_act      = VTCM_LAYOUT_PTR(uint8_t, పునాది, L.off_act);
    mmctx->vtcm_dst      = VTCM_LAYOUT_PTR(uint8_t, పునాది, L.off_dst);
    mmctx->vtcm_act_raw  = VTCM_LAYOUT_PTR(uint8_t, పునాది, L.off_act_raw);

    mmctx->vtcm_src0_stride    = 0;
    mmctx->vtcm_act_stride     = act_అడ్డువరుస_కొలత;
    mmctx->vtcm_act_raw_stride = hex_విడత_పైకి(act->ne[0] * పరిమాణం(దశాంశం), QK_Q8_0_TILED * పరిమాణం(దశాంశం));

    mmctx->vtcm_src0_కొలత_చొప్పున_thread = fastdiv(L.src0_బైట్లు, &octx->n_దారాలు_div);
    mmctx->vtcm_act_కొలత_చొప్పున_thread  = L.act_బైట్లు;
    mmctx->vtcm_dst_కొలత_చొప్పున_thread  = fastdiv(L.dst_బైట్లు, &octx->n_దారాలు_div);

    htp_trace_event_stop(tr, HTP_TRACE_EVT_INIT, 0);

    hvx_mm_transfer_act_dma(octx, kparams, act, mmctx->vtcm_act_raw, mmctx->vtcm_act_raw_stride, 0, act_nrows);

    mmctx->n_quant_అడ్డువరుసలు_చొప్పున_thread = (act_nrows + n_quant_tasks - 1) / n_quant_tasks;
    mmctx->n_quant_tasks = n_quant_tasks;
    work_వరుస_నిరంతరం(octx->ctx->work_వరుస, quant_task_func, mmctx, n_quant_tasks);

    work_వరుస_నిరంతరం(octx->ctx->work_వరుస, hvx_mm_nx_2d_tiled, mmctx, octx->n_దారాలు);

    తిరిగి HTP_స్థితి_OK;
}

పూర్ణాంకం op_matmul_nx(నిర్మాణం htp_ops_context * octx) {
    స్థిరరాశి నిర్మాణం htp_mm_kernel_params * kparams = (స్థిరరాశి నిర్మాణం htp_mm_kernel_params *) octx->kernel_params;

    ఒకవేళ (kparams->collapse) {
        స్థిరరాశి uint32_t n_weights = kparams->n_weights;
        స్థిరరాశి నిర్మాణం htp_tensor * act = octx->src[n_weights];
        స్థిరరాశి uint32_t s1 = (act->ne[1] > 1) ? act->nb[1] : ((act->ne[2] > 1) ? act->nb[2] : act->nb[3]);
        నిర్మాణం htp_tensor act_collapsed;
        నిర్మాణం htp_tensor dsts_collapsed[HTP_OP_MAX_OUTPUTS];
        స్థిరరాశి నిర్మాణం htp_tensor * orig_dsts[HTP_OP_MAX_OUTPUTS];

        htp_mm_tensor_collapse_అడ్డువరుసలు(&act_collapsed, act, s1);
        octx->src[n_weights] = &act_collapsed;

        కొరకు (uint32_t p = 0; p < n_weights; p++) {
            orig_dsts[p] = octx->dsts[p];
            స్థిరరాశి నిర్మాణం htp_tensor * d = octx->dsts[p];
            స్థిరరాశి uint32_t sd = (d->ne[1] > 1) ? d->nb[1] : ((d->ne[2] > 1) ? d->nb[2] : d->nb[3]);
            htp_mm_tensor_collapse_అడ్డువరుసలు(&dsts_collapsed[p], d, sd);
            octx->dsts[p] = &dsts_collapsed[p];
        }

        స్థిరరాశి పూర్ణాంకం స్థితి_2 = op_matmul_nx_impl(octx);

        octx->src[n_weights] = act;
        కొరకు (uint32_t p = 0; p < n_weights; p++) {
            octx->dsts[p] = orig_dsts[p];
        }
        తిరిగి స్థితి_2;
    }

    తిరిగి op_matmul_nx_impl(octx);
}
