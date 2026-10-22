#సూచన clang diagnostic ignored "-Wunused-variable"
#సూచన clang diagnostic ignored "-Wunused-function"
#సూచన clang diagnostic ignored "-Wunused-but-set-variable"

#అన్తర్భావ <HAP_farf.h>
#అన్తర్భావ <HAP_perf.h>

#అన్తర్భావ <math.h>
#అన్తర్భావ <string.h>

#అన్తర్భావ "dma-queue.h"
#అన్తర్భావ "hex-fastdiv.h"
#అన్తర్భావ "hvx-exp.h"
#అన్తర్భావ "hvx-sigmoid.h"
#అన్తర్భావ "hvx-utils.h"
#అన్తర్భావ "unary-ops.h"

#నిర్వచించు GGML_COMMON_DECL_C
#అన్తర్భావ "ggml-common.h"
#అన్తర్భావ "htp-ctx.h"
#అన్తర్భావ "htp-ops.h"
#అన్తర్భావ "htp-tensor.h"
#అన్తర్భావ "htp-vtcm.h"
#అన్తర్భావ "hex-profile.h"

నిర్మాణం htp_unary_context;

రకనిర్వచనం శూన్య (*unary_compute_fn_t)(స్థిరరాశి శూన్య * పరిమిత src,
                                   శూన్య * పరిమిత dst,
                                   uint32_t అంకె_అడ్డువరుసలు_2,
                                   స్థిరరాశి నిర్మాణం htp_unary_context * uctx);

రకనిర్వచనం శూన్య (*unary_rms_norm_mul_compute_fn_t)(స్థిరరాశి శూన్య * పరిమిత src,
                                                స్థిరరాశి శూన్య * పరిమిత బరువు,
                                                శూన్య * పరిమిత dst,
                                                uint32_t అంకె_అడ్డువరుసలు_2,
                                                స్థిరరాశి నిర్మాణం htp_unary_context * uctx);

రకనిర్వచనం శూన్య (*unary_tri_compute_fn_t)(స్థిరరాశి శూన్య * పరిమిత src,
                                       శూన్య * పరిమిత dst,
                                       uint32_t అంకె_అడ్డువరుసలు_2,
                                       uint32_t ir,
                                       స్థిరరాశి నిర్మాణం htp_unary_context * uctx);

రకనిర్వచనం శూన్య (*unary_tile_compute_fn_t)(శూన్య * పరిమిత dst,
                                        స్థిరరాశి శూన్య * పరిమిత src,
                                        uint32_t tw,
                                        స్థిరరాశి నిర్మాణం htp_unary_context * uctx);

రకనిర్వచనం శూన్య (*unary_tiled_tri_compute_fn_t)(స్థిరరాశి శూన్య * పరిమిత src,
                                             శూన్య * పరిమిత dst,
                                             uint32_t tile_elems,
                                             uint32_t నిలువు_ప్రారంభం,
                                             uint32_t i01,
                                             uint32_t ne0,
                                             int32_t ttype);

నిర్మాణం htp_unary_context {
    నిర్మాణం htp_ops_context * octx;
    స్థిరరాశి నిర్మాణం htp_unary_kernel_params * kparams;

    శూన్య *                    compute;

    dma_addr_t                దత్తాంశం_src0;
    dma_addr_t                దత్తాంశం_src1;            // weight/scale tensor for RMS_NORM_MUL
    dma_addr_t                దత్తాంశం_dst;

    size_t                    src0_దత్తాంశం_అడ్డువరుస_కొలత;   // actual data bytes per row
    size_t                    src1_దత్తాంశం_అడ్డువరుస_కొలత;
    size_t                    dst_దత్తాంశం_అడ్డువరుస_కొలత;    // actual data bytes per row

    size_t                    src0_అడ్డువరుస_కొలత_aligned;
    size_t                    src1_అడ్డువరుస_కొలత_aligned;
    size_t                    dst_అడ్డువరుస_కొలత_aligned;

    size_t                    src0_vtcm_half_కొలత;
    size_t                    src1_vtcm_half_కొలత;
    size_t                    dst_vtcm_half_కొలత;

    uint32_t                  ఖండం;
    uint32_t                  src0_nrows;
    uint32_t                  src0_nrows_చొప్పున_thread;
    uint32_t                  అడ్డువరుస_ప్రారంభం;
    uint32_t                  nc;
    uint32_t                  నిలువు_tile;             // tiled mode
    bool                      ప్రసారం_బరువు;

    uint8_t *                 vtcm_src0;
    uint8_t *                 vtcm_src1;
    uint8_t *                 vtcm_dst;

    size_t                    vtcm_src0_కొలత_చొప్పున_thread;
    size_t                    vtcm_src1_కొలత_చొప్పున_thread;
    size_t                    vtcm_dst_కొలత_చొప్పున_thread;
};

// Convert flat row index to DDR byte offset using the tensor's actual strides.
// ir = i1 + ne1*(i2 + ne2*i3)  =>  offset = i1*nb1 + i2*nb2 + i3*nb3
స్థిర అంతఃవరుస size_t unary_అడ్డువరుస_స్థానభ్రంశం(uint32_t ir,
                                      uint32_t ne1, uint32_t ne2,
                                      స్థిరరాశి నిర్మాణం fastdiv_విలువలు * div_ne1,
                                      స్థిరరాశి నిర్మాణం fastdiv_విలువలు * div_ne2,
                                      స్థిరరాశి నిర్మాణం fastdiv_విలువలు * div_ne12,
                                      size_t nb1, size_t nb2, size_t nb3) {
    స్థిరరాశి uint32_t i1 = fastmodulo(ir, ne1, div_ne1);
    స్థిరరాశి uint32_t ir_div_ne1 = fastdiv(ir, div_ne1);
    స్థిరరాశి uint32_t i2 = fastmodulo(ir_div_ne1, ne2, div_ne2);
    స్థిరరాశి uint32_t i3 = fastdiv(ir, div_ne12);
    తిరిగి i1 * nb1 + i2 * nb2 + i3 * nb3;
}

// Safe DMA block size from row `ir`: clamp to the tighter dim-1 slice
// boundary of src and dst so the nb1 stride stays valid for all rows.
స్థిర అంతఃవరుస uint32_t unary_ఖండం_కొలత(uint32_t ir,
                                        uint32_t చివర_అడ్డువరుస,
                                        uint32_t ఖండం,
                                        bool src_contig,
                                        bool dst_contig,
                                        uint32_t ne1,
                                        స్థిరరాశి నిర్మాణం fastdiv_విలువలు * div_ne1) {
    uint32_t హద్దు = MIN(ఖండం, చివర_అడ్డువరుస - ir);

    ఒకవేళ (!src_contig || !dst_contig) {
        స్థిరరాశి uint32_t ముక్క_చివర = (fastdiv(ir, div_ne1) + 1) * ne1;
        హద్దు = MIN(హద్దు, ముక్క_చివర - ir);
    }

    తిరిగి హద్దు;
}

#నిర్వచించు htp_unary_preamble            \
    స్థిరరాశి uint32_t ne00 = src->ne[0]; \
    స్థిరరాశి uint32_t ne01 = src->ne[1]; \
    స్థిరరాశి uint32_t ne02 = src->ne[2]; \
    స్థిరరాశి uint32_t ne03 = src->ne[3]; \
                                      \
    స్థిరరాశి uint32_t ne0 = dst->ne[0];  \
    స్థిరరాశి uint32_t ne1 = dst->ne[1];  \
    స్థిరరాశి uint32_t ne2 = dst->ne[2];  \
    స్థిరరాశి uint32_t ne3 = dst->ne[3];  \
                                      \
    స్థిరరాశి uint32_t nb00 = src->nb[0]; \
    స్థిరరాశి uint32_t nb01 = src->nb[1]; \
    స్థిరరాశి uint32_t nb02 = src->nb[2]; \
    స్థిరరాశి uint32_t nb03 = src->nb[3]; \
                                      \
    స్థిరరాశి uint32_t nb0 = dst->nb[0];  \
    స్థిరరాశి uint32_t nb1 = dst->nb[1];  \
    స్థిరరాశి uint32_t nb2 = dst->nb[2];  \
    స్థిరరాశి uint32_t nb3 = dst->nb[3];

#నిర్వచించు htp_unary_op_preamble                                         \
    int32_t * op_params = uctx->octx->op_params;                      \
    స్థిరరాశి uint32_t ne0 = uctx->nc;                                    \
    స్థిరరాశి size_t src0_అడ్డువరుస_కొలత_aligned = uctx->src0_అడ్డువరుస_కొలత_aligned; \
    స్థిరరాశి size_t dst_అడ్డువరుస_కొలత_aligned = uctx->dst_అడ్డువరుస_కొలత_aligned;

స్థిర శూన్య scale_f32(స్థిరరాశి శూన్య * పరిమిత src,
                      శూన్య * పరిమిత dst,
                      స్థిరరాశి uint32_t అంకె_అడ్డువరుసలు_2,
                      స్థిరరాశి నిర్మాణం htp_unary_context * uctx) {
    htp_unary_op_preamble;
    దశాంశం scale = 0.f;
    దశాంశం bias  = 0.f;
    జ్ఞాపకం_నకలు(&scale, &op_params[0], పరిమాణం(దశాంశం));
    జ్ఞాపకం_నకలు(&bias,  &op_params[1], పరిమాణం(దశాంశం));

    కొరకు (uint32_t ir = 0; ir < అంకె_అడ్డువరుసలు_2; ir++) {
        స్థిరరాశి uint8_t * పరిమిత src_local = (స్థిరరాశి uint8_t *)src + (ir * src0_అడ్డువరుస_కొలత_aligned);
        uint8_t * పరిమిత dst_local       = (uint8_t *)dst + (ir * dst_అడ్డువరుస_కొలత_aligned);

        hvx_scale_స్థానభ్రంశం_f32_aa((uint8_t *) dst_local, (స్థిరరాశి uint8_t *) src_local, ne0, scale, bias);
    }
}

స్థిర శూన్య clamp_f32(స్థిరరాశి శూన్య * పరిమిత src,
                      శూన్య * పరిమిత dst,
                      స్థిరరాశి uint32_t అంకె_అడ్డువరుసలు_2,
                      స్థిరరాశి నిర్మాణం htp_unary_context * uctx) {
    htp_unary_op_preamble;
    దశాంశం min = 0.f;
    దశాంశం max = 0.f;
    జ్ఞాపకం_నకలు(&min, &op_params[0], పరిమాణం(దశాంశం));
    జ్ఞాపకం_నకలు(&max, &op_params[1], పరిమాణం(దశాంశం));

    కొరకు (uint32_t ir = 0; ir < అంకె_అడ్డువరుసలు_2; ir++) {
        స్థిరరాశి uint8_t * పరిమిత src_local = (స్థిరరాశి uint8_t *)src + (ir * src0_అడ్డువరుస_కొలత_aligned);
        uint8_t * పరిమిత dst_local       = (uint8_t *)dst + (ir * dst_అడ్డువరుస_కొలత_aligned);

        hvx_clamp_scalar_f32(dst_local, src_local, min, max, ne0);
    }
}

స్థిర శూన్య leaky_relu_f32(స్థిరరాశి శూన్య * పరిమిత src,
                           శూన్య * పరిమిత dst,
                           స్థిరరాశి uint32_t అంకె_అడ్డువరుసలు_2,
                           స్థిరరాశి నిర్మాణం htp_unary_context * uctx) {
    htp_unary_op_preamble;
    దశాంశం negative_వాలు = 0.f;
    జ్ఞాపకం_నకలు(&negative_వాలు, &op_params[0], పరిమాణం(దశాంశం));

    కొరకు (uint32_t ir = 0; ir < అంకె_అడ్డువరుసలు_2; ir++) {
        స్థిరరాశి uint8_t * పరిమిత src_local = (స్థిరరాశి uint8_t *)src + (ir * src0_అడ్డువరుస_కొలత_aligned);
        uint8_t * పరిమిత dst_local       = (uint8_t *)dst + (ir * dst_అడ్డువరుస_కొలత_aligned);

        hvx_leaky_relu_scalar_f32(dst_local, src_local, negative_వాలు, ne0);
    }
}

స్థిర శూన్య rms_norm_f32(స్థిరరాశి శూన్య * పరిమిత src,
                         శూన్య * పరిమిత dst,
                         స్థిరరాశి uint32_t అంకె_అడ్డువరుసలు_2,
                         స్థిరరాశి నిర్మాణం htp_unary_context * uctx) {
    htp_unary_op_preamble;
    దశాంశం epsilon = 0.f;
    జ్ఞాపకం_నకలు(&epsilon, op_params, పరిమాణం(దశాంశం));

    కొరకు (uint32_t ir = 0; ir < అంకె_అడ్డువరుసలు_2; ir++) {
        స్థిరరాశి uint8_t * పరిమిత src_local = (స్థిరరాశి uint8_t *)src + (ir * src0_అడ్డువరుస_కొలత_aligned);
        uint8_t * పరిమిత dst_local       = (uint8_t *)dst + (ir * dst_అడ్డువరుస_కొలత_aligned);

        hvx_fast_rms_norm_f32((స్థిరరాశి uint8_t *) src_local, (uint8_t *) dst_local, ne0, epsilon);
    }
}

స్థిర శూన్య rms_norm_mul_f32(స్థిరరాశి శూన్య * పరిమిత src,
                             స్థిరరాశి శూన్య * పరిమిత బరువు,
                             శూన్య * పరిమిత dst,
                             స్థిరరాశి uint32_t అంకె_అడ్డువరుసలు_2,
                             స్థిరరాశి నిర్మాణం htp_unary_context * uctx) {
    htp_unary_op_preamble;
    దశాంశం epsilon = 0.f;
    జ్ఞాపకం_నకలు(&epsilon, op_params, పరిమాణం(దశాంశం));

    కొరకు (uint32_t ir = 0; ir < అంకె_అడ్డువరుసలు_2; ir++) {
        స్థిరరాశి uint8_t * పరిమిత src_local = (స్థిరరాశి uint8_t *)src + (ir * src0_అడ్డువరుస_కొలత_aligned);
        స్థిరరాశి uint8_t * పరిమిత w_local   = (స్థిరరాశి uint8_t *)బరువు + (uctx->ప్రసారం_బరువు ? 0 : ir * uctx->src1_అడ్డువరుస_కొలత_aligned);
        uint8_t * పరిమిత dst_local       = (uint8_t *)dst + (ir * dst_అడ్డువరుస_కొలత_aligned);

        hvx_fast_rms_norm_mul_f32(src_local, w_local, dst_local, ne0, epsilon);
    }
}

స్థిర శూన్య norm_f32(స్థిరరాశి శూన్య * పరిమిత src,
                     శూన్య * పరిమిత dst,
                     స్థిరరాశి uint32_t అంకె_అడ్డువరుసలు_2,
                     స్థిరరాశి నిర్మాణం htp_unary_context * uctx) {
    htp_unary_op_preamble;
    దశాంశం epsilon = 0.f;
    జ్ఞాపకం_నకలు(&epsilon, op_params, పరిమాణం(దశాంశం));

    కొరకు (uint32_t ir = 0; ir < అంకె_అడ్డువరుసలు_2; ir++) {
        స్థిరరాశి uint8_t * పరిమిత src_local = (స్థిరరాశి uint8_t *)src + (ir * src0_అడ్డువరుస_కొలత_aligned);
        uint8_t * పరిమిత dst_local       = (uint8_t *)dst + (ir * dst_అడ్డువరుస_కొలత_aligned);

        hvx_fast_norm_f32((స్థిరరాశి uint8_t *) src_local, (uint8_t *) dst_local, ne0, epsilon);
    }
}

స్థిర శూన్య sqr_f32(స్థిరరాశి శూన్య * పరిమిత src,
                    శూన్య * పరిమిత dst,
                    స్థిరరాశి uint32_t అంకె_అడ్డువరుసలు_2,
                    స్థిరరాశి నిర్మాణం htp_unary_context * uctx) {
    htp_unary_op_preamble;

    కొరకు (uint32_t ir = 0; ir < అంకె_అడ్డువరుసలు_2; ir++) {
        స్థిరరాశి uint8_t * పరిమిత src_local = (స్థిరరాశి uint8_t *)src + (ir * src0_అడ్డువరుస_కొలత_aligned);
        uint8_t * పరిమిత dst_local       = (uint8_t *)dst + (ir * dst_అడ్డువరుస_కొలత_aligned);

        hvx_sqr_f32_aa((uint8_t *) dst_local, (స్థిరరాశి uint8_t *) src_local, ne0);
    }
}

స్థిర శూన్య sqrt_f32(స్థిరరాశి శూన్య * పరిమిత src,
                     శూన్య * పరిమిత dst,
                     స్థిరరాశి uint32_t అంకె_అడ్డువరుసలు_2,
                     స్థిరరాశి నిర్మాణం htp_unary_context * uctx) {
    htp_unary_op_preamble;

    కొరకు (uint32_t ir = 0; ir < అంకె_అడ్డువరుసలు_2; ir++) {
        స్థిరరాశి uint8_t * పరిమిత src_local = (స్థిరరాశి uint8_t *)src + (ir * src0_అడ్డువరుస_కొలత_aligned);
        uint8_t * పరిమిత dst_local       = (uint8_t *)dst + (ir * dst_అడ్డువరుస_కొలత_aligned);

        hvx_sqrt_f32_aa((uint8_t *) dst_local, (స్థిరరాశి uint8_t *) src_local, ne0);
    }
}

స్థిర శూన్య scale_f16(స్థిరరాశి శూన్య * పరిమిత src,
                      శూన్య * పరిమిత dst,
                      స్థిరరాశి uint32_t అంకె_అడ్డువరుసలు_2,
                      స్థిరరాశి నిర్మాణం htp_unary_context * uctx) {
    htp_unary_op_preamble;
    దశాంశం scale = 0.f;
    దశాంశం bias  = 0.f;
    జ్ఞాపకం_నకలు(&scale, &op_params[0], పరిమాణం(దశాంశం));
    జ్ఞాపకం_నకలు(&bias,  &op_params[1], పరిమాణం(దశాంశం));

    కొరకు (uint32_t ir = 0; ir < అంకె_అడ్డువరుసలు_2; ir++) {
        స్థిరరాశి uint8_t * పరిమిత src_local = (స్థిరరాశి uint8_t *)src + (ir * src0_అడ్డువరుస_కొలత_aligned);
        uint8_t * పరిమిత dst_local       = (uint8_t *)dst + (ir * dst_అడ్డువరుస_కొలత_aligned);

        hvx_scale_స్థానభ్రంశం_f16_aa((uint8_t *) dst_local, (స్థిరరాశి uint8_t *) src_local, ne0, scale, bias);
    }
}

స్థిర శూన్య clamp_f16(స్థిరరాశి శూన్య * పరిమిత src,
                      శూన్య * పరిమిత dst,
                      స్థిరరాశి uint32_t అంకె_అడ్డువరుసలు_2,
                      స్థిరరాశి నిర్మాణం htp_unary_context * uctx) {
    htp_unary_op_preamble;
    దశాంశం min = 0.f;
    దశాంశం max = 0.f;
    జ్ఞాపకం_నకలు(&min, &op_params[0], పరిమాణం(దశాంశం));
    జ్ఞాపకం_నకలు(&max, &op_params[1], పరిమాణం(దశాంశం));

    కొరకు (uint32_t ir = 0; ir < అంకె_అడ్డువరుసలు_2; ir++) {
        స్థిరరాశి uint8_t * పరిమిత src_local = (స్థిరరాశి uint8_t *)src + (ir * src0_అడ్డువరుస_కొలత_aligned);
        uint8_t * పరిమిత dst_local       = (uint8_t *)dst + (ir * dst_అడ్డువరుస_కొలత_aligned);

        hvx_clamp_scalar_f16(dst_local, src_local, (_Float16) min, (_Float16) max, ne0);
    }
}

స్థిర శూన్య rms_norm_f16(స్థిరరాశి శూన్య * పరిమిత src,
                         శూన్య * పరిమిత dst,
                         స్థిరరాశి uint32_t అంకె_అడ్డువరుసలు_2,
                         స్థిరరాశి నిర్మాణం htp_unary_context * uctx) {
    htp_unary_op_preamble;
    దశాంశం epsilon = 0.f;
    జ్ఞాపకం_నకలు(&epsilon, op_params, పరిమాణం(దశాంశం));

    కొరకు (uint32_t ir = 0; ir < అంకె_అడ్డువరుసలు_2; ir++) {
        స్థిరరాశి uint8_t * పరిమిత src_local = (స్థిరరాశి uint8_t *)src + (ir * src0_అడ్డువరుస_కొలత_aligned);
        uint8_t * పరిమిత dst_local       = (uint8_t *)dst + (ir * dst_అడ్డువరుస_కొలత_aligned);

        hvx_fast_rms_norm_f16((స్థిరరాశి uint8_t *) src_local, (uint8_t *) dst_local, ne0, epsilon);
    }
}

స్థిర శూన్య norm_f16(స్థిరరాశి శూన్య * పరిమిత src,
                     శూన్య * పరిమిత dst,
                     స్థిరరాశి uint32_t అంకె_అడ్డువరుసలు_2,
                     స్థిరరాశి నిర్మాణం htp_unary_context * uctx) {
    htp_unary_op_preamble;
    దశాంశం epsilon = 0.f;
    జ్ఞాపకం_నకలు(&epsilon, op_params, పరిమాణం(దశాంశం));

    కొరకు (uint32_t ir = 0; ir < అంకె_అడ్డువరుసలు_2; ir++) {
        స్థిరరాశి uint8_t * పరిమిత src_local = (స్థిరరాశి uint8_t *)src + (ir * src0_అడ్డువరుస_కొలత_aligned);
        uint8_t * పరిమిత dst_local       = (uint8_t *)dst + (ir * dst_అడ్డువరుస_కొలత_aligned);

        hvx_fast_norm_f16((స్థిరరాశి uint8_t *) src_local, (uint8_t *) dst_local, ne0, epsilon);
    }
}

స్థిర శూన్య sqr_f16(స్థిరరాశి శూన్య * పరిమిత src,
                    శూన్య * పరిమిత dst,
                    స్థిరరాశి uint32_t అంకె_అడ్డువరుసలు_2,
                    స్థిరరాశి నిర్మాణం htp_unary_context * uctx) {
    htp_unary_op_preamble;

    కొరకు (uint32_t ir = 0; ir < అంకె_అడ్డువరుసలు_2; ir++) {
        స్థిరరాశి uint8_t * పరిమిత src_local = (స్థిరరాశి uint8_t *)src + (ir * src0_అడ్డువరుస_కొలత_aligned);
        uint8_t * పరిమిత dst_local       = (uint8_t *)dst + (ir * dst_అడ్డువరుస_కొలత_aligned);

        hvx_sqr_f16_aa((uint8_t *) dst_local, (స్థిరరాశి uint8_t *) src_local, ne0);
    }
}

స్థిర శూన్య sqrt_f16(స్థిరరాశి శూన్య * పరిమిత src,
                     శూన్య * పరిమిత dst,
                     స్థిరరాశి uint32_t అంకె_అడ్డువరుసలు_2,
                     స్థిరరాశి నిర్మాణం htp_unary_context * uctx) {
    htp_unary_op_preamble;

    కొరకు (uint32_t ir = 0; ir < అంకె_అడ్డువరుసలు_2; ir++) {
        స్థిరరాశి uint8_t * పరిమిత src_local = (స్థిరరాశి uint8_t *)src + (ir * src0_అడ్డువరుస_కొలత_aligned);
        uint8_t * పరిమిత dst_local       = (uint8_t *)dst + (ir * dst_అడ్డువరుస_కొలత_aligned);

        hvx_sqrt_f16_aa((uint8_t *) dst_local, (స్థిరరాశి uint8_t *) src_local, ne0);
    }
}

స్థిర శూన్య abs_f16(స్థిరరాశి శూన్య * పరిమిత src,
                    శూన్య * పరిమిత dst,
                    స్థిరరాశి uint32_t అంకె_అడ్డువరుసలు_2,
                    స్థిరరాశి నిర్మాణం htp_unary_context * uctx) {
    htp_unary_op_preamble;

    కొరకు (uint32_t ir = 0; ir < అంకె_అడ్డువరుసలు_2; ir++) {
        స్థిరరాశి uint8_t * పరిమిత src_local = (స్థిరరాశి uint8_t *)src + (ir * src0_అడ్డువరుస_కొలత_aligned);
        uint8_t * పరిమిత dst_local       = (uint8_t *)dst + (ir * dst_అడ్డువరుస_కొలత_aligned);

        hvx_abs_f16_aa((uint8_t *) dst_local, (స్థిరరాశి uint8_t *) src_local, ne0);
    }
}

స్థిర శూన్య log_f16(స్థిరరాశి శూన్య * పరిమిత src,
                    శూన్య * పరిమిత dst,
                    స్థిరరాశి uint32_t అంకె_అడ్డువరుసలు_2,
                    స్థిరరాశి నిర్మాణం htp_unary_context * uctx) {
    htp_unary_op_preamble;

    కొరకు (uint32_t ir = 0; ir < అంకె_అడ్డువరుసలు_2; ir++) {
        స్థిరరాశి uint8_t * పరిమిత src_local = (స్థిరరాశి uint8_t *)src + (ir * src0_అడ్డువరుస_కొలత_aligned);
        uint8_t * పరిమిత dst_local       = (uint8_t *)dst + (ir * dst_అడ్డువరుస_కొలత_aligned);

        hvx_log_f16_aa((uint8_t *) dst_local, (స్థిరరాశి uint8_t *) src_local, ne0);
    }
}

స్థిర శూన్య అడుగు_f16(స్థిరరాశి శూన్య * పరిమిత src,
                     శూన్య * పరిమిత dst,
                     స్థిరరాశి uint32_t అంకె_అడ్డువరుసలు_2,
                     స్థిరరాశి నిర్మాణం htp_unary_context * uctx) {
    htp_unary_op_preamble;

    కొరకు (uint32_t ir = 0; ir < అంకె_అడ్డువరుసలు_2; ir++) {
        స్థిరరాశి uint8_t * పరిమిత src_local = (స్థిరరాశి uint8_t *)src + (ir * src0_అడ్డువరుస_కొలత_aligned);
        uint8_t * పరిమిత dst_local       = (uint8_t *)dst + (ir * dst_అడ్డువరుస_కొలత_aligned);

        hvx_అడుగు_f16_aa((uint8_t *) dst_local, (స్థిరరాశి uint8_t *) src_local, ne0);
    }
}

స్థిర శూన్య l2_norm_f16(స్థిరరాశి శూన్య * పరిమిత src,
                        శూన్య * పరిమిత dst,
                        స్థిరరాశి uint32_t అంకె_అడ్డువరుసలు_2,
                        స్థిరరాశి నిర్మాణం htp_unary_context * uctx) {
    htp_unary_op_preamble;
    దశాంశం epsilon = 0.f;
    జ్ఞాపకం_నకలు(&epsilon, op_params, పరిమాణం(దశాంశం));

    కొరకు (uint32_t ir = 0; ir < అంకె_అడ్డువరుసలు_2; ir++) {
        స్థిరరాశి uint8_t * పరిమిత src_f = (స్థిరరాశి uint8_t *)src + (ir * src0_అడ్డువరుస_కొలత_aligned);
        uint8_t * పరిమిత dst_f       = (uint8_t *)dst + (ir * dst_అడ్డువరుస_కొలత_aligned);

        hvx_fast_l2_norm_f16((స్థిరరాశి uint8_t *)src_f, (uint8_t *)dst_f, ne0, epsilon);
    }
}

స్థిర శూన్య neg_f32(స్థిరరాశి శూన్య * పరిమిత src,
                    శూన్య * పరిమిత dst,
                    స్థిరరాశి uint32_t అంకె_అడ్డువరుసలు_2,
                    స్థిరరాశి నిర్మాణం htp_unary_context * uctx) {
    htp_unary_op_preamble;

    కొరకు (uint32_t ir = 0; ir < అంకె_అడ్డువరుసలు_2; ir++) {
        స్థిరరాశి uint8_t * పరిమిత src_local = (స్థిరరాశి uint8_t *)src + (ir * src0_అడ్డువరుస_కొలత_aligned);
        uint8_t * పరిమిత dst_local       = (uint8_t *)dst + (ir * dst_అడ్డువరుస_కొలత_aligned);

        hvx_scale_f32_aa(dst_local, src_local, ne0, -1.0f);
    }
}

స్థిర శూన్య exp_f32(స్థిరరాశి శూన్య * పరిమిత src,
                    శూన్య * పరిమిత dst,
                    స్థిరరాశి uint32_t అంకె_అడ్డువరుసలు_2,
                    స్థిరరాశి నిర్మాణం htp_unary_context * uctx) {
    htp_unary_op_preamble;

    కొరకు (uint32_t ir = 0; ir < అంకె_అడ్డువరుసలు_2; ir++) {
        స్థిరరాశి uint8_t * పరిమిత src_local = (స్థిరరాశి uint8_t *)src + (ir * src0_అడ్డువరుస_కొలత_aligned);
        uint8_t * పరిమిత dst_local       = (uint8_t *)dst + (ir * dst_అడ్డువరుస_కొలత_aligned);

        hvx_exp_f32(dst_local, src_local, ne0, false);
    }
}

స్థిర శూన్య sigmoid_f32(స్థిరరాశి శూన్య * పరిమిత src,
                        శూన్య * పరిమిత dst,
                        స్థిరరాశి uint32_t అంకె_అడ్డువరుసలు_2,
                        స్థిరరాశి నిర్మాణం htp_unary_context * uctx) {
    htp_unary_op_preamble;

    కొరకు (uint32_t ir = 0; ir < అంకె_అడ్డువరుసలు_2; ir++) {
        స్థిరరాశి uint8_t * పరిమిత src_local = (స్థిరరాశి uint8_t *)src + (ir * src0_అడ్డువరుస_కొలత_aligned);
        uint8_t * పరిమిత dst_local       = (uint8_t *)dst + (ir * dst_అడ్డువరుస_కొలత_aligned);

        hvx_sigmoid_f32_aa(dst_local, src_local, ne0);
    }
}

// silu(x) = x * sigmoid(x)
స్థిర శూన్య silu_f32(స్థిరరాశి శూన్య * పరిమిత src,
                     శూన్య * పరిమిత dst,
                     స్థిరరాశి uint32_t అంకె_అడ్డువరుసలు_2,
                     స్థిరరాశి నిర్మాణం htp_unary_context * uctx) {
    htp_unary_op_preamble;

    కొరకు (uint32_t ir = 0; ir < అంకె_అడ్డువరుసలు_2; ir++) {
        స్థిరరాశి uint8_t * పరిమిత src_local = (స్థిరరాశి uint8_t *)src + (ir * src0_అడ్డువరుస_కొలత_aligned);
        uint8_t * పరిమిత dst_local       = (uint8_t *)dst + (ir * dst_అడ్డువరుస_కొలత_aligned);

        hvx_sigmoid_f32_aa(dst_local, src_local, ne0);
        hvx_mul_f32_aaa(dst_local, src_local, dst_local, ne0);
    }
}

// GELU uses the tanh approximation.
స్థిర __attribute__((noinline)) HVX_సదిశ hvx_vec_gelu_f32(HVX_సదిశ x) {
    స్థిరరాశి HVX_సదిశ half = hvx_vec_splat_f32(0.5f);
    స్థిరరాశి HVX_సదిశ one  = hvx_vec_splat_f32(1.0f);

    HVX_సదిశ inner = hvx_vec_mul_f32_f32(x, x);
    inner = hvx_vec_mul_f32_f32(inner, hvx_vec_splat_f32(0.044715f));
    inner = hvx_vec_కూడు_f32_f32(inner, one);
    inner = hvx_vec_mul_f32_f32(inner, x);
    inner = hvx_vec_mul_f32_f32(inner, hvx_vec_splat_f32(0.7978845608028654f));

    తిరిగి hvx_vec_mul_f32_f32(hvx_vec_mul_f32_f32(half, x),
                               hvx_vec_కూడు_f32_f32(one, hvx_vec_tanh_f32(inner)));
}

స్థిర అంతఃవరుస శూన్య hvx_gelu_f32_aa(uint8_t * పరిమిత dst, స్థిరరాశి uint8_t * పరిమిత src, uint32_t n) {
    ప్రతిజ్ఞా((గుర్తులేని దీర్ఘ) dst % 128 == 0);
    ప్రతిజ్ఞా((గుర్తులేని దీర్ఘ) src % 128 == 0);

    HVX_సదిశ * పరిమిత vdst = (HVX_సదిశ *) dst;
    HVX_సదిశ * పరిమిత vsrc = (HVX_సదిశ *) src;

    స్థిరరాశి uint32_t nvec = n / VLEN_FP32;
    స్థిరరాశి uint32_t nloe = n % VLEN_FP32;

    uint32_t i = 0;
    _Pragma("unroll(4)")
    కొరకు (; i < nvec; i++) {
        vdst[i] = hvx_vec_gelu_f32(vsrc[i]);
    }
    ఒకవేళ (nloe) {
        hvx_vec_store_a(&vdst[i], nloe * పరిమాణం(దశాంశం), hvx_vec_gelu_f32(vsrc[i]));
    }
}

// GELU_QUICK uses x * sigmoid(1.702 * x).
స్థిర __attribute__((noinline)) HVX_సదిశ hvx_vec_gelu_శీఘ్ర_f32(HVX_సదిశ x) {
    స్థిరరాశి HVX_సదిశ one     = hvx_vec_splat_f32(1.0f);
    స్థిరరాశి HVX_సదిశ max_exp = hvx_vec_splat_f32(87.0f);
    స్థిరరాశి HVX_సదిశ min_exp = hvx_vec_splat_f32(-87.0f);
    స్థిరరాశి HVX_సదిశ scaled  = hvx_vec_mul_f32_f32(x, hvx_vec_splat_f32(1.702f));
    స్థిరరాశి HVX_సదిశ sigmoid = hvx_vec_fast_sigmoid_f32_guard(scaled, one, max_exp, min_exp);

    తిరిగి hvx_vec_mul_f32_f32(x, sigmoid);
}

స్థిర అంతఃవరుస శూన్య hvx_gelu_శీఘ్ర_f32_aa(uint8_t * పరిమిత dst, స్థిరరాశి uint8_t * పరిమిత src, uint32_t n) {
    ప్రతిజ్ఞా((గుర్తులేని దీర్ఘ) dst % 128 == 0);
    ప్రతిజ్ఞా((గుర్తులేని దీర్ఘ) src % 128 == 0);

    HVX_సదిశ * పరిమిత vdst = (HVX_సదిశ *) dst;
    HVX_సదిశ * పరిమిత vsrc = (HVX_సదిశ *) src;

    స్థిరరాశి uint32_t nvec = n / VLEN_FP32;
    స్థిరరాశి uint32_t nloe = n % VLEN_FP32;

    uint32_t i = 0;
    _Pragma("unroll(4)")
    కొరకు (; i < nvec; i++) {
        vdst[i] = hvx_vec_gelu_శీఘ్ర_f32(vsrc[i]);
    }
    ఒకవేళ (nloe) {
        hvx_vec_store_a(&vdst[i], nloe * పరిమాణం(దశాంశం), hvx_vec_gelu_శీఘ్ర_f32(vsrc[i]));
    }
}

స్థిర శూన్య gelu_f32(స్థిరరాశి శూన్య * పరిమిత src,
                     శూన్య * పరిమిత dst,
                     స్థిరరాశి uint32_t అంకె_అడ్డువరుసలు_2,
                     స్థిరరాశి నిర్మాణం htp_unary_context * uctx) {
    htp_unary_op_preamble;

    కొరకు (uint32_t ir = 0; ir < అంకె_అడ్డువరుసలు_2; ir++) {
        స్థిరరాశి uint8_t * పరిమిత src_local = (స్థిరరాశి uint8_t *)src + (ir * src0_అడ్డువరుస_కొలత_aligned);
        uint8_t * పరిమిత dst_local       = (uint8_t *)dst + (ir * dst_అడ్డువరుస_కొలత_aligned);

        hvx_gelu_f32_aa(dst_local, src_local, ne0);
    }
}

స్థిర శూన్య gelu_శీఘ్ర_f32(స్థిరరాశి శూన్య * పరిమిత src,
                           శూన్య * పరిమిత dst,
                           స్థిరరాశి uint32_t అంకె_అడ్డువరుసలు_2,
                           స్థిరరాశి నిర్మాణం htp_unary_context * uctx) {
    htp_unary_op_preamble;

    కొరకు (uint32_t ir = 0; ir < అంకె_అడ్డువరుసలు_2; ir++) {
        స్థిరరాశి uint8_t * పరిమిత src_local = (స్థిరరాశి uint8_t *)src + (ir * src0_అడ్డువరుస_కొలత_aligned);
        uint8_t * పరిమిత dst_local       = (uint8_t *)dst + (ir * dst_అడ్డువరుస_కొలత_aligned);

        hvx_gelu_శీఘ్ర_f32_aa(dst_local, src_local, ne0);
    }
}

స్థిర శూన్య gelu_erf_f32(స్థిరరాశి శూన్య * పరిమిత src,
                         శూన్య * పరిమిత dst,
                         స్థిరరాశి uint32_t అంకె_అడ్డువరుసలు_2,
                         స్థిరరాశి నిర్మాణం htp_unary_context * uctx) {
    htp_unary_op_preamble;

    కొరకు (uint32_t ir = 0; ir < అంకె_అడ్డువరుసలు_2; ir++) {
        స్థిరరాశి uint8_t * పరిమిత src_local = (స్థిరరాశి uint8_t *) src + (ir * src0_అడ్డువరుస_కొలత_aligned);
        uint8_t * పరిమిత dst_local       = (uint8_t *) dst + (ir * dst_అడ్డువరుస_కొలత_aligned);

        hvx_gelu_erf_f32_aa(dst_local, src_local, ne0);
    }
}

స్థిర శూన్య tri_f32(స్థిరరాశి శూన్య * పరిమిత src,
                    శూన్య * పరిమిత dst,
                    స్థిరరాశి uint32_t అంకె_అడ్డువరుసలు_2,
                    స్థిరరాశి uint32_t ir,
                    స్థిరరాశి నిర్మాణం htp_unary_context * uctx) {
    htp_unary_op_preamble;
    స్థిరరాశి int32_t ttype = op_params[0];
    స్థిరరాశి HVX_సదిశ zero = hvx_vec_splat_f32(0.0f);
    స్థిరరాశి uint32_t nvec  = ne0 / VLEN_FP32;
    స్థిరరాశి uint32_t nloe  = ne0 % VLEN_FP32;

    స్థిరరాశి uint32_t ne01 = uctx->octx->src[0]->ne[1];

    కొరకు (uint32_t b = 0; b < అంకె_అడ్డువరుసలు_2; b++) {
        స్థిరరాశి uint32_t abs_అడ్డువరుస = ir + b;
        స్థిరరాశి uint32_t i01     = abs_అడ్డువరుస % ne01;

        స్థిరరాశి HVX_సదిశ * పరిమిత v_src = (స్థిరరాశి HVX_సదిశ *) ((స్థిరరాశి uint8_t *) src + b * src0_అడ్డువరుస_కొలత_aligned);
        HVX_సదిశ * పరిమిత v_dst       = (HVX_సదిశ *) ((uint8_t *) dst + b * dst_అడ్డువరుస_కొలత_aligned);

        uint32_t boundary;
        పూర్ణాంకం      keep_ఎడమ;
        ఎంపిక (ttype) {
            స్థితి 0: boundary = i01;     keep_ఎడమ = 0; ఆపు;  // keep col >= row
            స్థితి 1: boundary = i01 + 1; keep_ఎడమ = 0; ఆపు;  // keep col > row
            స్థితి 2: boundary = i01 + 1; keep_ఎడమ = 1; ఆపు;  // keep col <= row
            స్థితి 3: boundary = i01;     keep_ఎడమ = 1; ఆపు;  // keep col < row
            అప్రమేయ: boundary = 0; keep_ఎడమ = 0; ఆపు;
        }
        ఒకవేళ (boundary > ne0) boundary = ne0;

        // Full HVX vectors - each starts at a 128-byte aligned offset
        కొరకు (uint32_t i = 0; i < nvec; i++) {
            స్థిరరాశి uint32_t vec_ప్రారంభం = i * VLEN_FP32;
            స్థిరరాశి uint32_t vec_చివర   = vec_ప్రారంభం + VLEN_FP32;
            ఒకవేళ (keep_ఎడమ) {
                ఒకవేళ (vec_చివర <= boundary) {
                    v_dst[i] = v_src[i];
                } లేకపోతే ఒకవేళ (vec_ప్రారంభం >= boundary) {
                    v_dst[i] = zero;
                } లేకపోతే {
                    HVX_సదిశ_Pred ముసుగు = Q6_Q_vsetq_R((boundary - vec_ప్రారంభం) * పరిమాణం(దశాంశం));
                    v_dst[i]            = Q6_V_vmux_QVV(ముసుగు, v_src[i], zero);
                }
            } లేకపోతే {
                ఒకవేళ (vec_చివర <= boundary) {
                    v_dst[i] = zero;
                } లేకపోతే ఒకవేళ (vec_ప్రారంభం >= boundary) {
                    v_dst[i] = v_src[i];
                } లేకపోతే {
                    HVX_సదిశ_Pred ముసుగు = Q6_Q_vsetq_R((boundary - vec_ప్రారంభం) * పరిమాణం(దశాంశం));
                    v_dst[i]            = Q6_V_vmux_QVV(ముసుగు, zero, v_src[i]);
                }
            }
        }

        // Tail elements (row_elems not a multiple of VLEN_FP32)
        ఒకవేళ (nloe > 0) {
            స్థిరరాశి uint32_t abs_ప్రారంభం = nvec * VLEN_FP32;
            స్థిరరాశి uint32_t abs_చివర   = abs_ప్రారంభం + nloe;
            HVX_సదిశ     తోక_val;
            ఒకవేళ (keep_ఎడమ) {
                ఒకవేళ (abs_చివర <= boundary) {
                    తోక_val = v_src[nvec];
                } లేకపోతే ఒకవేళ (abs_ప్రారంభం >= boundary) {
                    తోక_val = zero;
                } లేకపోతే {
                    HVX_సదిశ_Pred ముసుగు = Q6_Q_vsetq_R((boundary - abs_ప్రారంభం) * పరిమాణం(దశాంశం));
                    తోక_val            = Q6_V_vmux_QVV(ముసుగు, v_src[nvec], zero);
                }
            } లేకపోతే {
                ఒకవేళ (abs_చివర <= boundary) {
                    తోక_val = zero;
                } లేకపోతే ఒకవేళ (abs_ప్రారంభం >= boundary) {
                    తోక_val = v_src[nvec];
                } లేకపోతే {
                    HVX_సదిశ_Pred ముసుగు = Q6_Q_vsetq_R((boundary - abs_ప్రారంభం) * పరిమాణం(దశాంశం));
                    తోక_val            = Q6_V_vmux_QVV(ముసుగు, zero, v_src[nvec]);
                }
            }
            hvx_vec_store_a(&v_dst[nvec], nloe * పరిమాణం(దశాంశం), తోక_val);
        }
    }
}

// softplus(x) = log(1 + exp(x))
// Match CPU reference: return x for x > 20 to avoid overflow.
స్థిర అంతఃవరుస శూన్య hvx_softplus_f32_aa(uint8_t * పరిమిత dst, స్థిరరాశి uint8_t * పరిమిత src, uint32_t n) {
    స్థిరరాశి HVX_సదిశ * పరిమిత v_src = (స్థిరరాశి HVX_సదిశ *) src;
    HVX_సదిశ * పరిమిత v_dst       = (HVX_సదిశ *) dst;

    స్థిరరాశి HVX_సదిశ twenty = hvx_vec_splat_f32(20.0f);
    స్థిరరాశి HVX_సదిశ one    = hvx_vec_splat_f32(1.0f);

    స్థిరరాశి uint32_t nvec = n / VLEN_FP32;
    స్థిరరాశి uint32_t nloe = n % VLEN_FP32;

    #సూచన unroll(4)
    కొరకు (uint32_t i = 0; i < nvec; i++) {
        HVX_సదిశ x              = v_src[i];
        HVX_సదిశ_Pred pred_gt20  = Q6_Q_vcmp_gt_VsfVsf(x, twenty);
        HVX_సదిశ expx           = hvx_vec_exp_f32(x);
        HVX_సదిశ logv           = hvx_vec_log_f32(hvx_vec_కూడు_f32_f32(one, expx));
        v_dst[i]                  = Q6_V_vmux_QVV(pred_gt20, x, logv);
    }

    ఒకవేళ (nloe > 0) {
        HVX_సదిశ x             = v_src[nvec];
        HVX_సదిశ_Pred pred_gt20 = Q6_Q_vcmp_gt_VsfVsf(x, twenty);
        HVX_సదిశ expx          = hvx_vec_exp_f32(x);
        HVX_సదిశ logv          = hvx_vec_log_f32(hvx_vec_కూడు_f32_f32(one, expx));
        HVX_సదిశ sp            = Q6_V_vmux_QVV(pred_gt20, x, logv);
        hvx_vec_store_a((శూన్య *) &v_dst[nvec], nloe * పరిమాణం(దశాంశం), sp);
    }
}

స్థిర శూన్య softplus_f32(స్థిరరాశి శూన్య * పరిమిత src,
                         శూన్య * పరిమిత dst,
                         స్థిరరాశి uint32_t అంకె_అడ్డువరుసలు_2,
                         స్థిరరాశి నిర్మాణం htp_unary_context * uctx) {
    htp_unary_op_preamble;

    కొరకు (uint32_t ir = 0; ir < అంకె_అడ్డువరుసలు_2; ir++) {
        స్థిరరాశి uint8_t * పరిమిత src_local = (స్థిరరాశి uint8_t *)src + (ir * src0_అడ్డువరుస_కొలత_aligned);
        uint8_t * పరిమిత dst_local       = (uint8_t *)dst + (ir * dst_అడ్డువరుస_కొలత_aligned);

        hvx_softplus_f32_aa(dst_local, src_local, ne0);
    }
}

స్థిర శూన్య l2_norm_f32(స్థిరరాశి శూన్య * పరిమిత src,
                        శూన్య * పరిమిత dst,
                        స్థిరరాశి uint32_t అంకె_అడ్డువరుసలు_2,
                        స్థిరరాశి నిర్మాణం htp_unary_context * uctx) {
    htp_unary_op_preamble;
    దశాంశం epsilon = 0.f;
    జ్ఞాపకం_నకలు(&epsilon, op_params, పరిమాణం(దశాంశం));

    కొరకు (uint32_t ir = 0; ir < అంకె_అడ్డువరుసలు_2; ir++) {
        స్థిరరాశి uint8_t * పరిమిత src_f = (స్థిరరాశి uint8_t *)src + (ir * src0_అడ్డువరుస_కొలత_aligned);
        uint8_t * పరిమిత dst_f       = (uint8_t *)dst + (ir * dst_అడ్డువరుస_కొలత_aligned);

        hvx_fast_l2_norm_f32((స్థిరరాశి uint8_t *)src_f, (uint8_t *)dst_f, ne0, epsilon);
    }
}

స్థిర అంతఃవరుస HVX_సదిశ_Pair hvx_fwht_stage_pair(HVX_సదిశ v0, HVX_సదిశ v1, int32_t b) {
    HVX_సదిశ_Pair p = Q6_W_vshuff_VVR(v1, v0, b);
    HVX_సదిశ s = Q6_Vqf32_vadd_Vqf32Vqf32(Q6_V_lo_W(p), Q6_V_hi_W(p));
    HVX_సదిశ d = Q6_Vqf32_vsub_Vqf32Vqf32(Q6_V_lo_W(p), Q6_V_hi_W(p));
    తిరిగి Q6_W_vshuff_VVR(d, s, b);
}

స్థిర అంతఃవరుస HVX_సదిశ_Pair hvx_fwht32_pair(HVX_సదిశ v0_sf, HVX_సదిశ v1_sf, HVX_సదిశ v_scale) {
    HVX_సదిశ v0 = Q6_Vqf32_vmpy_VsfVsf(v0_sf, v_scale);
    HVX_సదిశ v1 = Q6_Vqf32_vmpy_VsfVsf(v1_sf, v_scale);
    HVX_సదిశ_Pair p;
    p = hvx_fwht_stage_pair(v0, v1, 4);
    p = hvx_fwht_stage_pair(Q6_V_lo_W(p), Q6_V_hi_W(p), 8);
    p = hvx_fwht_stage_pair(Q6_V_lo_W(p), Q6_V_hi_W(p), 16);
    p = hvx_fwht_stage_pair(Q6_V_lo_W(p), Q6_V_hi_W(p), 32);
    p = hvx_fwht_stage_pair(Q6_V_lo_W(p), Q6_V_hi_W(p), 64);
    తిరిగి p;
}

స్థిర అంతఃవరుస శూన్య hvx_fwht_store32(HVX_సదిశ * పరిమిత v_dst, HVX_సదిశ_Pair p) {
    v_dst[0] = Q6_Vsf_equals_Vqf32(Q6_V_lo_W(p));
}

స్థిర అంతఃవరుస శూన్య hvx_fwht_store64(HVX_సదిశ * పరిమిత v_dst, HVX_సదిశ_Pair p) {
    HVX_సదిశ u = Q6_V_lo_W(p);
    HVX_సదిశ v = Q6_V_hi_W(p);
    v_dst[0] = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vqf32(u, v));
    v_dst[1] = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vsub_Vqf32Vqf32(u, v));
}

స్థిర అంతఃవరుస శూన్య hvx_fwht_తోక_f32(HVX_సదిశ * పరిమిత v_dst, uint32_t nvec) {
    uint32_t len_vec = 2;
    కొరకు (; len_vec * 2 < nvec; len_vec <<= 2) {
        స్థిరరాశి uint32_t అడుగు = len_vec * 4;
        కొరకు (uint32_t i = 0; i < nvec; i += అడుగు) {
            కొరకు (uint32_t j = 0; j < len_vec; j++) {
                HVX_సదిశ a0 = v_dst[i + j];
                HVX_సదిశ a1 = v_dst[i + j + len_vec];
                HVX_సదిశ a2 = v_dst[i + j + 2 * len_vec];
                HVX_సదిశ a3 = v_dst[i + j + 3 * len_vec];

                HVX_సదిశ t0 = Q6_Vqf32_vadd_VsfVsf(a0, a1);
                HVX_సదిశ t1 = Q6_Vqf32_vsub_VsfVsf(a0, a1);
                HVX_సదిశ t2 = Q6_Vqf32_vadd_VsfVsf(a2, a3);
                HVX_సదిశ t3 = Q6_Vqf32_vsub_VsfVsf(a2, a3);

                v_dst[i + j]               = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vqf32(t0, t2));
                v_dst[i + j + len_vec]     = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vqf32(t1, t3));
                v_dst[i + j + 2 * len_vec] = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vsub_Vqf32Vqf32(t0, t2));
                v_dst[i + j + 3 * len_vec] = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vsub_Vqf32Vqf32(t1, t3));
            }
        }
    }

    ఒకవేళ (len_vec < nvec) {
        కొరకు (uint32_t i = 0; i < nvec; i += 2 * len_vec) {
            కొరకు (uint32_t j = 0; j < len_vec; j++) {
                HVX_సదిశ u = v_dst[i + j];
                HVX_సదిశ v = v_dst[i + len_vec + j];
                v_dst[i + j]           = hvx_vec_కూడు_f32_f32(u, v);
                v_dst[i + len_vec + j] = hvx_vec_sub_f32_f32(u, v);
            }
        }
    }
}

స్థిర అంతఃవరుస శూన్య hvx_fwht_అడ్డువరుస_f32(స్థిరరాశి uint8_t * పరిమిత src, uint8_t * పరిమిత dst, uint32_t n, HVX_సదిశ v_scale) {
    స్థిరరాశి HVX_సదిశ * పరిమిత v_src = (స్థిరరాశి HVX_సదిశ *) src;
    HVX_సదిశ * పరిమిత v_dst       = (HVX_సదిశ *) dst;
    స్థిరరాశి uint32_t nvec               = n / VLEN_FP32;

    ఒకవేళ (nvec == 1) {
        HVX_సదిశ_Pair p = hvx_fwht32_pair(v_src[0], v_src[0], v_scale);
        hvx_fwht_store32(v_dst, p);
        తిరిగి;
    }

    కొరకు (uint32_t i = 0; i < nvec; i += 2) {
        HVX_సదిశ_Pair p = hvx_fwht32_pair(v_src[i], v_src[i + 1], v_scale);
        hvx_fwht_store64(&v_dst[i], p);
    }

    hvx_fwht_తోక_f32(v_dst, nvec);
}

స్థిర అంతఃవరుస శూన్య hvx_fwht_అడ్డువరుస_f16(స్థిరరాశి uint8_t * పరిమిత src, uint8_t * పరిమిత dst, uint32_t n, HVX_సదిశ v_scale) {
    స్థిరరాశి HVX_సదిశ * పరిమిత v_src = (స్థిరరాశి HVX_సదిశ *) src;
    HVX_సదిశ * పరిమిత v_dst       = (HVX_సదిశ *) dst;
    స్థిరరాశి uint32_t nvec               = n / VLEN_FP32;

    ఒకవేళ (nvec == 1) {
        HVX_సదిశ v0 = Q6_V_lo_W(hvx_vec_f16_గమ్యం_f32(v_src[0]));
        HVX_సదిశ_Pair p = hvx_fwht32_pair(v0, v0, v_scale);
        hvx_fwht_store32(v_dst, p);
        తిరిగి;
    }

    స్థిరరాశి uint32_t nvec_f16 = n / VLEN_FP16;
    కొరకు (uint32_t i = 0; i < nvec_f16; i++) {
        HVX_సదిశ_Pair f32p = hvx_vec_f16_గమ్యం_f32(v_src[i]);
        HVX_సదిశ_Pair p    = hvx_fwht32_pair(Q6_V_lo_W(f32p), Q6_V_hi_W(f32p), v_scale);
        hvx_fwht_store64(&v_dst[2 * i], p);
    }

    hvx_fwht_తోక_f32(v_dst, nvec);
}

స్థిర శూన్య hadamard_f32(స్థిరరాశి శూన్య * పరిమిత src,
                         శూన్య * పరిమిత dst,
                         స్థిరరాశి uint32_t అంకె_అడ్డువరుసలు_2,
                         స్థిరరాశి నిర్మాణం htp_unary_context * uctx) {
    htp_unary_op_preamble;
    స్థిరరాశి HVX_సదిశ v_scale = hvx_vec_splat_f32(1.0f / sqrtf((దశాంశం) ne0));

    కొరకు (uint32_t ir = 0; ir < అంకె_అడ్డువరుసలు_2; ir++) {
        స్థిరరాశి uint8_t * పరిమిత src_అడ్డువరుస = (స్థిరరాశి uint8_t *) src + (ir * src0_అడ్డువరుస_కొలత_aligned);
        uint8_t * పరిమిత dst_అడ్డువరుస       = (uint8_t *) dst + (ir * dst_అడ్డువరుస_కొలత_aligned);

        hvx_fwht_అడ్డువరుస_f32(src_అడ్డువరుస, dst_అడ్డువరుస, ne0, v_scale);
    }
}

స్థిర శూన్య hadamard_f16(స్థిరరాశి శూన్య * పరిమిత src,
                         శూన్య * పరిమిత dst,
                         స్థిరరాశి uint32_t అంకె_అడ్డువరుసలు_2,
                         స్థిరరాశి నిర్మాణం htp_unary_context * uctx) {
    htp_unary_op_preamble;
    స్థిరరాశి HVX_సదిశ v_scale = hvx_vec_splat_f32(1.0f / sqrtf((దశాంశం) ne0));

    కొరకు (uint32_t ir = 0; ir < అంకె_అడ్డువరుసలు_2; ir++) {
        స్థిరరాశి uint8_t * పరిమిత src_అడ్డువరుస = (స్థిరరాశి uint8_t *) src + (ir * src0_అడ్డువరుస_కొలత_aligned);
        uint8_t * పరిమిత dst_అడ్డువరుస       = (uint8_t *) dst + (ir * dst_అడ్డువరుస_కొలత_aligned);

        hvx_fwht_అడ్డువరుస_f16(src_అడ్డువరుస, dst_అడ్డువరుస, ne0, v_scale);
    }
}

స్థిర శూన్య tanh_f32(స్థిరరాశి శూన్య * పరిమిత src,
                     శూన్య * పరిమిత dst,
                     స్థిరరాశి uint32_t అంకె_అడ్డువరుసలు_2,
                     స్థిరరాశి నిర్మాణం htp_unary_context * uctx) {
    htp_unary_op_preamble;

    కొరకు (uint32_t ir = 0; ir < అంకె_అడ్డువరుసలు_2; ir++) {
        స్థిరరాశి uint8_t * పరిమిత src_local = (స్థిరరాశి uint8_t *)src + (ir * src0_అడ్డువరుస_కొలత_aligned);
        uint8_t * పరిమిత dst_local       = (uint8_t *)dst + (ir * dst_అడ్డువరుస_కొలత_aligned);

        hvx_tanh_f32_aa(dst_local, src_local, ne0);
    }
}

స్థిర శూన్య abs_f32(స్థిరరాశి శూన్య * పరిమిత src,
                    శూన్య * పరిమిత dst,
                    స్థిరరాశి uint32_t అంకె_అడ్డువరుసలు_2,
                    స్థిరరాశి నిర్మాణం htp_unary_context * uctx) {
    htp_unary_op_preamble;

    కొరకు (uint32_t ir = 0; ir < అంకె_అడ్డువరుసలు_2; ir++) {
        స్థిరరాశి uint8_t * పరిమిత src_local = (స్థిరరాశి uint8_t *)src + (ir * src0_అడ్డువరుస_కొలత_aligned);
        uint8_t * పరిమిత dst_local       = (uint8_t *)dst + (ir * dst_అడ్డువరుస_కొలత_aligned);

        hvx_abs_f32_aa(dst_local, src_local, ne0);
    }
}

స్థిర శూన్య relu_f32(స్థిరరాశి శూన్య * పరిమిత src,
                     శూన్య * పరిమిత dst,
                     స్థిరరాశి uint32_t అంకె_అడ్డువరుసలు_2,
                     స్థిరరాశి నిర్మాణం htp_unary_context * uctx) {
    htp_unary_op_preamble;

    కొరకు (uint32_t ir = 0; ir < అంకె_అడ్డువరుసలు_2; ir++) {
        స్థిరరాశి uint8_t * పరిమిత src_local = (స్థిరరాశి uint8_t *)src + (ir * src0_అడ్డువరుస_కొలత_aligned);
        uint8_t * పరిమిత dst_local       = (uint8_t *)dst + (ir * dst_అడ్డువరుస_కొలత_aligned);

        hvx_max_scalar_f32(dst_local, src_local, 0.0f, ne0);
    }
}

స్థిర శూన్య అడుగు_f32(స్థిరరాశి శూన్య * పరిమిత src,
                     శూన్య * పరిమిత dst,
                     స్థిరరాశి uint32_t అంకె_అడ్డువరుసలు_2,
                     స్థిరరాశి నిర్మాణం htp_unary_context * uctx) {
    htp_unary_op_preamble;

    కొరకు (uint32_t ir = 0; ir < అంకె_అడ్డువరుసలు_2; ir++) {
        స్థిరరాశి uint8_t * పరిమిత src_local = (స్థిరరాశి uint8_t *)src + (ir * src0_అడ్డువరుస_కొలత_aligned);
        uint8_t * పరిమిత dst_local       = (uint8_t *)dst + (ir * dst_అడ్డువరుస_కొలత_aligned);

        hvx_అడుగు_f32_aa(dst_local, src_local, ne0);
    }
}

స్థిర శూన్య log_f32(స్థిరరాశి శూన్య * పరిమిత src,
                    శూన్య * పరిమిత dst,
                    స్థిరరాశి uint32_t అంకె_అడ్డువరుసలు_2,
                    స్థిరరాశి నిర్మాణం htp_unary_context * uctx) {
    htp_unary_op_preamble;

    కొరకు (uint32_t ir = 0; ir < అంకె_అడ్డువరుసలు_2; ir++) {
        స్థిరరాశి uint8_t * పరిమిత src_local = (స్థిరరాశి uint8_t *)src + (ir * src0_అడ్డువరుస_కొలత_aligned);
        uint8_t * పరిమిత dst_local       = (uint8_t *)dst + (ir * dst_అడ్డువరుస_కొలత_aligned);

        hvx_log_f32_aa(dst_local, src_local, ne0);
    }
}

#// Pointwise unary ops on one column tile in VTCM.
స్థిర శూన్య tile_scale_f32(శూన్య * పరిమిత dst, స్థిరరాశి శూన్య * పరిమిత src, uint32_t tw, స్థిరరాశి నిర్మాణం htp_unary_context * uctx) {
    దశాంశం scale = 0.f;
    దశాంశం bias  = 0.f;
    జ్ఞాపకం_నకలు(&scale, &uctx->octx->op_params[0], పరిమాణం(దశాంశం));
    జ్ఞాపకం_నకలు(&bias,  &uctx->octx->op_params[1], పరిమాణం(దశాంశం));
    hvx_scale_స్థానభ్రంశం_f32_aa((uint8_t *) dst, (స్థిరరాశి uint8_t *) src, tw, scale, bias);
}

స్థిర శూన్య tile_clamp_f32(శూన్య * పరిమిత dst, స్థిరరాశి శూన్య * పరిమిత src, uint32_t tw, స్థిరరాశి నిర్మాణం htp_unary_context * uctx) {
    దశాంశం min = 0.f;
    దశాంశం max = 0.f;
    జ్ఞాపకం_నకలు(&min, &uctx->octx->op_params[0], పరిమాణం(దశాంశం));
    జ్ఞాపకం_నకలు(&max, &uctx->octx->op_params[1], పరిమాణం(దశాంశం));
    hvx_clamp_scalar_f32((uint8_t *) dst, (స్థిరరాశి uint8_t *) src, min, max, tw);
}

స్థిర శూన్య tile_leaky_relu_f32(శూన్య * పరిమిత dst, స్థిరరాశి శూన్య * పరిమిత src, uint32_t tw, స్థిరరాశి నిర్మాణం htp_unary_context * uctx) {
    దశాంశం negative_వాలు = 0.f;
    జ్ఞాపకం_నకలు(&negative_వాలు, &uctx->octx->op_params[0], పరిమాణం(దశాంశం));
    hvx_leaky_relu_scalar_f32((uint8_t *) dst, (స్థిరరాశి uint8_t *) src, negative_వాలు, tw);
}

స్థిర శూన్య tile_sqr_f32(శూన్య * పరిమిత dst, స్థిరరాశి శూన్య * పరిమిత src, uint32_t tw, స్థిరరాశి నిర్మాణం htp_unary_context * uctx) {
    (శూన్య) uctx;
    hvx_sqr_f32_aa((uint8_t *) dst, (స్థిరరాశి uint8_t *) src, tw);
}

స్థిర శూన్య tile_sqrt_f32(శూన్య * పరిమిత dst, స్థిరరాశి శూన్య * పరిమిత src, uint32_t tw, స్థిరరాశి నిర్మాణం htp_unary_context * uctx) {
    (శూన్య) uctx;
    hvx_sqrt_f32_aa((uint8_t *) dst, (స్థిరరాశి uint8_t *) src, tw);
}

స్థిర శూన్య tile_neg_f32(శూన్య * పరిమిత dst, స్థిరరాశి శూన్య * పరిమిత src, uint32_t tw, స్థిరరాశి నిర్మాణం htp_unary_context * uctx) {
    (శూన్య) uctx;
    hvx_scale_f32_aa((uint8_t *) dst, (స్థిరరాశి uint8_t *) src, tw, -1.0f);
}

స్థిర శూన్య tile_exp_f32(శూన్య * పరిమిత dst, స్థిరరాశి శూన్య * పరిమిత src, uint32_t tw, స్థిరరాశి నిర్మాణం htp_unary_context * uctx) {
    (శూన్య) uctx;
    hvx_exp_f32((uint8_t *) dst, (స్థిరరాశి uint8_t *) src, tw, false);
}

స్థిర శూన్య tile_sigmoid_f32(శూన్య * పరిమిత dst, స్థిరరాశి శూన్య * పరిమిత src, uint32_t tw, స్థిరరాశి నిర్మాణం htp_unary_context * uctx) {
    (శూన్య) uctx;
    hvx_sigmoid_f32_aa((uint8_t *) dst, (స్థిరరాశి uint8_t *) src, tw);
}

స్థిర శూన్య tile_silu_f32(శూన్య * పరిమిత dst, స్థిరరాశి శూన్య * పరిమిత src, uint32_t tw, స్థిరరాశి నిర్మాణం htp_unary_context * uctx) {
    (శూన్య) uctx;
    hvx_sigmoid_f32_aa((uint8_t *) dst, (స్థిరరాశి uint8_t *) src, tw);
    hvx_mul_f32_aaa((uint8_t *) dst, (స్థిరరాశి uint8_t *) src, (uint8_t *) dst, tw);
}

స్థిర శూన్య tile_gelu_f32(శూన్య * పరిమిత dst, స్థిరరాశి శూన్య * పరిమిత src, uint32_t tw, స్థిరరాశి నిర్మాణం htp_unary_context * uctx) {
    (శూన్య) uctx;
    hvx_gelu_f32_aa((uint8_t *) dst, (స్థిరరాశి uint8_t *) src, tw);
}

స్థిర శూన్య tile_gelu_శీఘ్ర_f32(శూన్య * పరిమిత dst, స్థిరరాశి శూన్య * పరిమిత src, uint32_t tw, స్థిరరాశి నిర్మాణం htp_unary_context * uctx) {
    (శూన్య) uctx;
    hvx_gelu_శీఘ్ర_f32_aa((uint8_t *) dst, (స్థిరరాశి uint8_t *) src, tw);
}

స్థిర శూన్య tile_gelu_erf_f32(శూన్య * పరిమిత dst, స్థిరరాశి శూన్య * పరిమిత src, uint32_t tw, స్థిరరాశి నిర్మాణం htp_unary_context * uctx) {
    (శూన్య) uctx;
    hvx_gelu_erf_f32_aa((uint8_t *) dst, (స్థిరరాశి uint8_t *) src, tw);
}

స్థిర శూన్య tile_softplus_f32(శూన్య * పరిమిత dst, స్థిరరాశి శూన్య * పరిమిత src, uint32_t tw, స్థిరరాశి నిర్మాణం htp_unary_context * uctx) {
    (శూన్య) uctx;
    hvx_softplus_f32_aa((uint8_t *) dst, (స్థిరరాశి uint8_t *) src, tw);
}

స్థిర శూన్య tile_tanh_f32(శూన్య * పరిమిత dst, స్థిరరాశి శూన్య * పరిమిత src, uint32_t tw, స్థిరరాశి నిర్మాణం htp_unary_context * uctx) {
    (శూన్య) uctx;
    hvx_tanh_f32_aa((uint8_t *) dst, (స్థిరరాశి uint8_t *) src, tw);
}

స్థిర శూన్య tile_abs_f32(శూన్య * పరిమిత dst, స్థిరరాశి శూన్య * పరిమిత src, uint32_t tw, స్థిరరాశి నిర్మాణం htp_unary_context * uctx) {
    (శూన్య) uctx;
    hvx_abs_f32_aa((uint8_t *) dst, (స్థిరరాశి uint8_t *) src, tw);
}

స్థిర శూన్య tile_log_f32(శూన్య * పరిమిత dst, స్థిరరాశి శూన్య * పరిమిత src, uint32_t tw, స్థిరరాశి నిర్మాణం htp_unary_context * uctx) {
    (శూన్య) uctx;
    hvx_log_f32_aa((uint8_t *) dst, (స్థిరరాశి uint8_t *) src, tw);
}

స్థిర శూన్య tile_relu_f32(శూన్య * పరిమిత dst, స్థిరరాశి శూన్య * పరిమిత src, uint32_t tw, స్థిరరాశి నిర్మాణం htp_unary_context * uctx) {
    (శూన్య) uctx;
    hvx_max_scalar_f32((uint8_t *) dst, (స్థిరరాశి uint8_t *) src, 0.0f, tw);
}

స్థిర శూన్య tile_అడుగు_f32(శూన్య * పరిమిత dst, స్థిరరాశి శూన్య * పరిమిత src, uint32_t tw, స్థిరరాశి నిర్మాణం htp_unary_context * uctx) {
    (శూన్య) uctx;
    hvx_అడుగు_f32_aa((uint8_t *) dst, (స్థిరరాశి uint8_t *) src, tw);
}

స్థిర శూన్య tri_apply_tile_f32(స్థిరరాశి శూన్య * పరిమిత src, శూన్య * పరిమిత dst,
                               uint32_t tile_elems, uint32_t నిలువు_ప్రారంభం, uint32_t i01,
                               uint32_t ne0, int32_t ttype) {
    స్థిరరాశి HVX_సదిశ * పరిమిత v_src = (స్థిరరాశి HVX_సదిశ *) src;
    HVX_సదిశ * పరిమిత v_dst       = (HVX_సదిశ *) dst;
    స్థిరరాశి HVX_సదిశ zero = hvx_vec_splat_f32(0.0f);

    uint32_t boundary;
    పూర్ణాంకం      keep_ఎడమ;
    ఎంపిక (ttype) {
        స్థితి 0: boundary = i01;     keep_ఎడమ = 0; ఆపు;
        స్థితి 1: boundary = i01 + 1; keep_ఎడమ = 0; ఆపు;
        స్థితి 2: boundary = i01 + 1; keep_ఎడమ = 1; ఆపు;
        స్థితి 3: boundary = i01;     keep_ఎడమ = 1; ఆపు;
        అప్రమేయ: boundary = 0; keep_ఎడమ = 0; ఆపు;
    }
    ఒకవేళ (boundary > ne0) boundary = ne0;

    స్థిరరాశి uint32_t nvec = tile_elems / VLEN_FP32;
    స్థిరరాశి uint32_t nloe = tile_elems % VLEN_FP32;

    కొరకు (uint32_t i = 0; i < nvec; i++) {
        స్థిరరాశి uint32_t abs_ప్రారంభం = నిలువు_ప్రారంభం + i * VLEN_FP32;
        స్థిరరాశి uint32_t abs_చివర   = abs_ప్రారంభం + VLEN_FP32;
        ఒకవేళ (keep_ఎడమ) {
            ఒకవేళ (abs_చివర <= boundary) {
                v_dst[i] = v_src[i];
            } లేకపోతే ఒకవేళ (abs_ప్రారంభం >= boundary) {
                v_dst[i] = zero;
            } లేకపోతే {
                HVX_సదిశ_Pred ముసుగు = Q6_Q_vsetq_R((boundary - abs_ప్రారంభం) * పరిమాణం(దశాంశం));
                v_dst[i]            = Q6_V_vmux_QVV(ముసుగు, v_src[i], zero);
            }
        } లేకపోతే {
            ఒకవేళ (abs_చివర <= boundary) {
                v_dst[i] = zero;
            } లేకపోతే ఒకవేళ (abs_ప్రారంభం >= boundary) {
                v_dst[i] = v_src[i];
            } లేకపోతే {
                HVX_సదిశ_Pred ముసుగు = Q6_Q_vsetq_R((boundary - abs_ప్రారంభం) * పరిమాణం(దశాంశం));
                v_dst[i]            = Q6_V_vmux_QVV(ముసుగు, zero, v_src[i]);
            }
        }
    }

    ఒకవేళ (nloe > 0) {
        స్థిరరాశి uint32_t abs_ప్రారంభం = నిలువు_ప్రారంభం + nvec * VLEN_FP32;
        స్థిరరాశి uint32_t abs_చివర   = abs_ప్రారంభం + nloe;
        HVX_సదిశ     తోక_val;
        ఒకవేళ (keep_ఎడమ) {
            ఒకవేళ (abs_చివర <= boundary) {
                తోక_val = v_src[nvec];
            } లేకపోతే ఒకవేళ (abs_ప్రారంభం >= boundary) {
                తోక_val = zero;
            } లేకపోతే {
                HVX_సదిశ_Pred ముసుగు = Q6_Q_vsetq_R((boundary - abs_ప్రారంభం) * పరిమాణం(దశాంశం));
                తోక_val            = Q6_V_vmux_QVV(ముసుగు, v_src[nvec], zero);
            }
        } లేకపోతే {
            ఒకవేళ (abs_చివర <= boundary) {
                తోక_val = zero;
            } లేకపోతే ఒకవేళ (abs_ప్రారంభం >= boundary) {
                తోక_val = v_src[nvec];
            } లేకపోతే {
                HVX_సదిశ_Pred ముసుగు = Q6_Q_vsetq_R((boundary - abs_ప్రారంభం) * పరిమాణం(దశాంశం));
                తోక_val            = Q6_V_vmux_QVV(ముసుగు, zero, v_src[nvec]);
            }
        }
        hvx_vec_store_a(&v_dst[nvec], nloe * పరిమాణం(దశాంశం), తోక_val);
    }
}

// 1. Standard row-block unary task (F32 and F16).
స్థిర శూన్య unary_thread_అడ్డువరుస_ఖండం(గుర్తులేని పూర్ణాంకం nth, గుర్తులేని పూర్ణాంకం ith, శూన్య * దత్తాంశం) {
    (శూన్య) nth;
    స్థిరరాశి నిర్మాణం htp_unary_context * uctx = (స్థిరరాశి నిర్మాణం htp_unary_context *) దత్తాంశం;
    నిర్మాణం htp_ops_context * octx = uctx->octx;
    స్థిరరాశి నిర్మాణం htp_tensor * src = octx->src[0];
    స్థిరరాశి నిర్మాణం htp_tensor * dst = octx->dst;
    నిర్మాణం htp_thread_trace * tr = &octx->ctx->trace[ith];

    htp_unary_preamble;

    స్థిరరాశి uint32_t src0_nrows_చొప్పున_thread = uctx->src0_nrows_చొప్పున_thread;
    స్థిరరాశి size_t src0_దత్తాంశం_అడ్డువరుస_కొలత = uctx->src0_దత్తాంశం_అడ్డువరుస_కొలత;
    స్థిరరాశి size_t dst_దత్తాంశం_అడ్డువరుస_కొలత  = uctx->dst_దత్తాంశం_అడ్డువరుస_కొలత;
    స్థిరరాశి size_t src0_అడ్డువరుస_కొలత_aligned = uctx->src0_అడ్డువరుస_కొలత_aligned;
    స్థిరరాశి size_t dst_అడ్డువరుస_కొలత_aligned  = uctx->dst_అడ్డువరుస_కొలత_aligned;

    స్థిరరాశి uint32_t src0_nrows = uctx->src0_nrows;
    స్థిరరాశి uint32_t src0_ప్రారంభం_అడ్డువరుస = uctx->అడ్డువరుస_ప్రారంభం + src0_nrows_చొప్పున_thread * ith;
    స్థిరరాశి uint32_t src0_చివర_అడ్డువరుస   = MIN(src0_ప్రారంభం_అడ్డువరుస + src0_nrows_చొప్పున_thread, uctx->అడ్డువరుస_ప్రారంభం + src0_nrows);

    ఒకవేళ (src0_ప్రారంభం_అడ్డువరుస >= src0_చివర_అడ్డువరుస) {
        తిరిగి;
    }

    స్థిరరాశి dma_addr_t దత్తాంశం_src = uctx->దత్తాంశం_src0;
    స్థిరరాశి dma_addr_t దత్తాంశం_dst = uctx->దత్తాంశం_dst;

    uint8_t * src0_vtcm_దత్తాంశం = uctx->vtcm_src0 + (ith * uctx->vtcm_src0_కొలత_చొప్పున_thread);
    uint8_t * dst_vtcm_దత్తాంశం  = uctx->vtcm_dst + (ith * uctx->vtcm_dst_కొలత_చొప్పున_thread);

    స్థిరరాశి size_t src0_vtcm_half_కొలత = uctx->src0_vtcm_half_కొలత;
    స్థిరరాశి size_t dst_vtcm_half_కొలత  = uctx->dst_vtcm_half_కొలత;

    స్థిరరాశి bool src0_contig = (nb02 == (size_t)ne01 * nb01) &&
                             (nb03 == (size_t)ne02 * nb02);
    స్థిరరాశి bool dst_contig  = (nb2  == (size_t)ne1  * nb1)  &&
                             (nb3  == (size_t)ne2  * nb2);

    స్థిరరాశి నిర్మాణం fastdiv_విలువలు * div_ne01  = &uctx->kparams->div_ne01;
    స్థిరరాశి నిర్మాణం fastdiv_విలువలు * div_ne02  = &uctx->kparams->div_ne02;
    స్థిరరాశి నిర్మాణం fastdiv_విలువలు * div_ne012 = &uctx->kparams->div_ne012;

    స్థిరరాశి uint32_t src0_max_ఖండం = src0_contig ? uctx->ఖండం : MIN((uint32_t)uctx->ఖండం, ne01);
    స్థిరరాశి uint32_t dst_max_ఖండం  = dst_contig  ? uctx->ఖండం : MIN((uint32_t)uctx->ఖండం, ne1);
    స్థిరరాశి uint32_t ఖండం_2 = MIN(src0_max_ఖండం, dst_max_ఖండం);
    ఒకవేళ (ఖండం_2 == 0) {
        FARF(ERROR, "unary-row-block : current VTCM reservation %zu is too small, needed at least %zu\n",
             uctx->vtcm_src0_కొలత_చొప్పున_thread, src0_అడ్డువరుస_కొలత_aligned);
        తిరిగి;
    }

    dma_వరుస * dma_q = octx->ctx->dma[ith];

    కొరకు (uint32_t ir = src0_ప్రారంభం_అడ్డువరుస, vtcm_idx = 0; ir < src0_చివర_అడ్డువరుస && vtcm_idx < 2; vtcm_idx++) {
        స్థిరరాశి uint32_t ఖండం_కొలత_3 = unary_ఖండం_కొలత(ir, src0_చివర_అడ్డువరుస, ఖండం_2, src0_contig, dst_contig,
                                                     ne01, div_ne01);

        dma_వరుస_నెట్టు(dma_q,
            dma_make_దత్తాంశం(దత్తాంశం_dst, dst_vtcm_దత్తాంశం + (vtcm_idx * dst_vtcm_half_కొలత)),
            nb1, dst_అడ్డువరుస_కొలత_aligned, dst_దత్తాంశం_అడ్డువరుస_కొలత, 0);

        స్థిరరాశి size_t src0_off = src0_contig ? (ir * nb01) :
            unary_అడ్డువరుస_స్థానభ్రంశం(ir, ne01, ne02, div_ne01, div_ne02, div_ne012, nb01, nb02, nb03);
        dma_వరుస_నెట్టు(dma_q,
            dma_make_దత్తాంశం(src0_vtcm_దత్తాంశం + (vtcm_idx * src0_vtcm_half_కొలత), దత్తాంశం_src + src0_off),
            src0_అడ్డువరుస_కొలత_aligned, nb01, src0_దత్తాంశం_అడ్డువరుస_కొలత, ఖండం_కొలత_3);

        ir += ఖండం_కొలత_3;
    }

    unary_compute_fn_t compute = (unary_compute_fn_t) uctx->compute;

    కొరకు (uint32_t ir = src0_ప్రారంభం_అడ్డువరుస; ir < src0_చివర_అడ్డువరుస; ) {
        స్థిరరాశి uint32_t ఖండం_కొలత_3 = unary_ఖండం_కొలత(ir, src0_చివర_అడ్డువరుస, ఖండం_2, src0_contig, dst_contig,
                                                     ne01, div_ne01);

        శూన్య * dst_vtcm  = (శూన్య *) (uintptr_t) dma_వరుస_తీయి(dma_q).src;
        శూన్య * src0_vtcm = (శూన్య *) (uintptr_t) dma_వరుస_తీయి(dma_q).dst;

        htp_trace_event_ప్రారంభం(tr, HTP_TRACE_EVT_HVX_COMP, ir);
        compute(src0_vtcm, dst_vtcm, ఖండం_కొలత_3, uctx);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ir);

        స్థిరరాశి size_t dst_off = dst_contig ? (ir * nb1) :
            unary_అడ్డువరుస_స్థానభ్రంశం(ir, ne1, ne2, div_ne01, div_ne02, div_ne012, nb1, nb2, nb3);
        dma_వరుస_నెట్టు(dma_q,
            dma_make_దత్తాంశం(దత్తాంశం_dst + dst_off, dst_vtcm),
            nb1, dst_అడ్డువరుస_కొలత_aligned, dst_దత్తాంశం_అడ్డువరుస_కొలత, ఖండం_కొలత_3);

        స్థిరరాశి uint32_t తదుపరి_ir = ir + ఖండం_కొలత_3;
        ఒకవేళ (తదుపరి_ir < src0_చివర_అడ్డువరుస) {
            స్థిరరాశి uint32_t తదుపరి_ఖండం_కొలత = unary_ఖండం_కొలత(తదుపరి_ir, src0_చివర_అడ్డువరుస, ఖండం_2, src0_contig,
                                                              dst_contig, ne01, div_ne01);
            స్థిరరాశి uint32_t pref_ir = తదుపరి_ir + తదుపరి_ఖండం_కొలత;
            ఒకవేళ (pref_ir < src0_చివర_అడ్డువరుస) {
                స్థిరరాశి uint32_t pref_ఖండం_కొలత = unary_ఖండం_కొలత(pref_ir, src0_చివర_అడ్డువరుస, ఖండం_2, src0_contig,
                                                                  dst_contig, ne01, div_ne01);
                స్థిరరాశి size_t src0_pref_off = src0_contig ? (pref_ir * nb01) :
                    unary_అడ్డువరుస_స్థానభ్రంశం(pref_ir, ne01, ne02, div_ne01, div_ne02, div_ne012, nb01, nb02, nb03);
                dma_వరుస_నెట్టు(dma_q,
                    dma_make_దత్తాంశం(src0_vtcm, దత్తాంశం_src + src0_pref_off),
                    src0_అడ్డువరుస_కొలత_aligned, nb01, src0_దత్తాంశం_అడ్డువరుస_కొలత, pref_ఖండం_కొలత);
            }
        }
        ir += ఖండం_కొలత_3;
    }

    dma_వరుస_flush(dma_q);
}

// 2. RMS_NORM_MUL row-block task with weight buffer.
స్థిర శూన్య unary_thread_rms_norm_mul_f32(గుర్తులేని పూర్ణాంకం nth, గుర్తులేని పూర్ణాంకం ith, శూన్య * దత్తాంశం) {
    (శూన్య) nth;
    స్థిరరాశి నిర్మాణం htp_unary_context * uctx = (స్థిరరాశి నిర్మాణం htp_unary_context *) దత్తాంశం;
    నిర్మాణం htp_ops_context * octx = uctx->octx;
    స్థిరరాశి నిర్మాణం htp_tensor * src = octx->src[0];
    స్థిరరాశి నిర్మాణం htp_tensor * dst = octx->dst;
    నిర్మాణం htp_thread_trace * tr = &octx->ctx->trace[ith];

    htp_unary_preamble;

    స్థిరరాశి uint32_t src0_nrows_చొప్పున_thread = uctx->src0_nrows_చొప్పున_thread;
    స్థిరరాశి size_t src0_దత్తాంశం_అడ్డువరుస_కొలత = uctx->src0_దత్తాంశం_అడ్డువరుస_కొలత;
    స్థిరరాశి size_t dst_దత్తాంశం_అడ్డువరుస_కొలత  = uctx->dst_దత్తాంశం_అడ్డువరుస_కొలత;
    స్థిరరాశి size_t src0_అడ్డువరుస_కొలత_aligned = uctx->src0_అడ్డువరుస_కొలత_aligned;
    స్థిరరాశి size_t dst_అడ్డువరుస_కొలత_aligned  = uctx->dst_అడ్డువరుస_కొలత_aligned;

    స్థిరరాశి uint32_t src0_nrows = uctx->src0_nrows;
    స్థిరరాశి uint32_t src0_ప్రారంభం_అడ్డువరుస = uctx->అడ్డువరుస_ప్రారంభం + src0_nrows_చొప్పున_thread * ith;
    స్థిరరాశి uint32_t src0_చివర_అడ్డువరుస   = MIN(src0_ప్రారంభం_అడ్డువరుస + src0_nrows_చొప్పున_thread, uctx->అడ్డువరుస_ప్రారంభం + src0_nrows);

    ఒకవేళ (src0_ప్రారంభం_అడ్డువరుస >= src0_చివర_అడ్డువరుస) {
        తిరిగి;
    }

    స్థిరరాశి dma_addr_t దత్తాంశం_src  = uctx->దత్తాంశం_src0;
    స్థిరరాశి dma_addr_t దత్తాంశం_src1 = uctx->దత్తాంశం_src1;
    స్థిరరాశి dma_addr_t దత్తాంశం_dst  = uctx->దత్తాంశం_dst;

    స్థిరరాశి నిర్మాణం htp_tensor * src1 = octx->src[1];
    స్థిరరాశి uint32_t nb11 = src1->nb[1];
    స్థిరరాశి uint32_t nb12 = src1->nb[2];
    స్థిరరాశి uint32_t nb13 = src1->nb[3];
    స్థిరరాశి uint32_t nb11_bc = (src1->ne[1] > 1) ? nb11 : 0;
    స్థిరరాశి uint32_t nb12_bc = (src1->ne[2] > 1) ? nb12 : 0;
    స్థిరరాశి uint32_t nb13_bc = (src1->ne[3] > 1) ? nb13 : 0;
    స్థిరరాశి bool src1_contig = ((nb12 == (size_t)ne01 * nb11) && (nb13 == (size_t)ne02 * nb12));

    uint8_t * src0_vtcm_దత్తాంశం = uctx->vtcm_src0 + (ith * uctx->vtcm_src0_కొలత_చొప్పున_thread);
    uint8_t * src1_vtcm_దత్తాంశం = uctx->vtcm_src1 ? (uctx->vtcm_src1 + (ith * uctx->vtcm_src1_కొలత_చొప్పున_thread)) : ఖాళీ;
    uint8_t * dst_vtcm_దత్తాంశం  = uctx->vtcm_dst + (ith * uctx->vtcm_dst_కొలత_చొప్పున_thread);

    స్థిరరాశి size_t src0_vtcm_half_కొలత = uctx->src0_vtcm_half_కొలత;
    స్థిరరాశి size_t src1_vtcm_half_కొలత = uctx->src1_vtcm_half_కొలత;
    స్థిరరాశి size_t dst_vtcm_half_కొలత  = uctx->dst_vtcm_half_కొలత;

    స్థిరరాశి bool src0_contig = (nb02 == (size_t)ne01 * nb01) &&
                             (nb03 == (size_t)ne02 * nb02);
    స్థిరరాశి bool dst_contig  = (nb2  == (size_t)ne1  * nb1)  &&
                             (nb3  == (size_t)ne2  * nb2);

    స్థిరరాశి నిర్మాణం fastdiv_విలువలు * div_ne01  = &uctx->kparams->div_ne01;
    స్థిరరాశి నిర్మాణం fastdiv_విలువలు * div_ne02  = &uctx->kparams->div_ne02;
    స్థిరరాశి నిర్మాణం fastdiv_విలువలు * div_ne012 = &uctx->kparams->div_ne012;

    స్థిరరాశి bool src1_needs_అడ్డువరుస_clip = !uctx->ప్రసారం_బరువు && !src1_contig;
    స్థిరరాశి bool ఖండం_src0_contig = src0_contig && !src1_needs_అడ్డువరుస_clip;
    స్థిరరాశి bool ఖండం_dst_contig  = dst_contig  && !src1_needs_అడ్డువరుస_clip;

    స్థిరరాశి uint32_t src0_max_ఖండం = ఖండం_src0_contig ? uctx->ఖండం : MIN((uint32_t)uctx->ఖండం, ne01);
    స్థిరరాశి uint32_t dst_max_ఖండం  = ఖండం_dst_contig  ? uctx->ఖండం : MIN((uint32_t)uctx->ఖండం, ne1);
    స్థిరరాశి uint32_t ఖండం_2 = MIN(src0_max_ఖండం, dst_max_ఖండం);
    ఒకవేళ (ఖండం_2 == 0) {
        FARF(ERROR, "unary-rms-norm-mul : current VTCM reservation %zu is too small, needed at least %zu\n",
             uctx->vtcm_src0_కొలత_చొప్పున_thread, src0_అడ్డువరుస_కొలత_aligned);
        తిరిగి;
    }

    dma_వరుస * dma_q = octx->ctx->dma[ith];

    ఒకవేళ (uctx->ప్రసారం_బరువు) {
        dma_వరుస_నెట్టు(dma_q, dma_make_దత్తాంశం(src1_vtcm_దత్తాంశం, దత్తాంశం_src1),
                       uctx->src1_అడ్డువరుస_కొలత_aligned, 0, uctx->src1_దత్తాంశం_అడ్డువరుస_కొలత, 1);
        dma_వరుస_flush(dma_q);
    }

    కొరకు (uint32_t ir = src0_ప్రారంభం_అడ్డువరుస, vtcm_idx = 0; ir < src0_చివర_అడ్డువరుస && vtcm_idx < 2; vtcm_idx++) {
        స్థిరరాశి uint32_t ఖండం_కొలత_3 = unary_ఖండం_కొలత(ir, src0_చివర_అడ్డువరుస, ఖండం_2, ఖండం_src0_contig, ఖండం_dst_contig,
                                                     ne01, div_ne01);

        dma_వరుస_నెట్టు(dma_q,
            dma_make_దత్తాంశం(దత్తాంశం_dst, dst_vtcm_దత్తాంశం + (vtcm_idx * dst_vtcm_half_కొలత)),
            nb1, dst_అడ్డువరుస_కొలత_aligned, dst_దత్తాంశం_అడ్డువరుస_కొలత, 0);

        స్థిరరాశి size_t src0_off = src0_contig ? (ir * nb01) :
            unary_అడ్డువరుస_స్థానభ్రంశం(ir, ne01, ne02, div_ne01, div_ne02, div_ne012, nb01, nb02, nb03);
        dma_వరుస_నెట్టు(dma_q,
            dma_make_దత్తాంశం(src0_vtcm_దత్తాంశం + (vtcm_idx * src0_vtcm_half_కొలత), దత్తాంశం_src + src0_off),
            src0_అడ్డువరుస_కొలత_aligned, nb01, src0_దత్తాంశం_అడ్డువరుస_కొలత, ఖండం_కొలత_3);

        ఒకవేళ (!uctx->ప్రసారం_బరువు) {
            స్థిరరాశి size_t src1_off = src1_contig ? (ir * nb11) :
                unary_అడ్డువరుస_స్థానభ్రంశం(ir, ne01, ne02, div_ne01, div_ne02, div_ne012, nb11_bc, nb12_bc, nb13_bc);
            dma_వరుస_నెట్టు(dma_q,
                dma_make_దత్తాంశం(src1_vtcm_దత్తాంశం + (vtcm_idx * src1_vtcm_half_కొలత), దత్తాంశం_src1 + src1_off),
                uctx->src1_అడ్డువరుస_కొలత_aligned, nb11, uctx->src1_దత్తాంశం_అడ్డువరుస_కొలత, ఖండం_కొలత_3);
        }

        ir += ఖండం_కొలత_3;
    }

    unary_rms_norm_mul_compute_fn_t compute = (unary_rms_norm_mul_compute_fn_t) uctx->compute;

    కొరకు (uint32_t ir = src0_ప్రారంభం_అడ్డువరుస; ir < src0_చివర_అడ్డువరుస; ) {
        స్థిరరాశి uint32_t ఖండం_కొలత_3 = unary_ఖండం_కొలత(ir, src0_చివర_అడ్డువరుస, ఖండం_2, ఖండం_src0_contig, ఖండం_dst_contig,
                                                     ne01, div_ne01);

        శూన్య * dst_vtcm  = (శూన్య *) (uintptr_t) dma_వరుస_తీయి(dma_q).src;
        శూన్య * src0_vtcm = (శూన్య *) (uintptr_t) dma_వరుస_తీయి(dma_q).dst;
        శూన్య * src1_vtcm = ఖాళీ;
        ఒకవేళ (!uctx->ప్రసారం_బరువు) {
            src1_vtcm = (శూన్య *) (uintptr_t) dma_వరుస_తీయి(dma_q).dst;
        }

        htp_trace_event_ప్రారంభం(tr, HTP_TRACE_EVT_HVX_COMP, ir);
        స్థిరరాశి శూన్య * w = uctx->ప్రసారం_బరువు ? (స్థిరరాశి శూన్య *) src1_vtcm_దత్తాంశం : src1_vtcm;
        compute(src0_vtcm, w, dst_vtcm, ఖండం_కొలత_3, uctx);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ir);

        స్థిరరాశి size_t dst_off = dst_contig ? (ir * nb1) :
            unary_అడ్డువరుస_స్థానభ్రంశం(ir, ne1, ne2, div_ne01, div_ne02, div_ne012, nb1, nb2, nb3);
        dma_వరుస_నెట్టు(dma_q,
            dma_make_దత్తాంశం(దత్తాంశం_dst + dst_off, dst_vtcm),
            nb1, dst_అడ్డువరుస_కొలత_aligned, dst_దత్తాంశం_అడ్డువరుస_కొలత, ఖండం_కొలత_3);

        స్థిరరాశి uint32_t తదుపరి_ir = ir + ఖండం_కొలత_3;
        ఒకవేళ (తదుపరి_ir < src0_చివర_అడ్డువరుస) {
            స్థిరరాశి uint32_t తదుపరి_ఖండం_కొలత = unary_ఖండం_కొలత(తదుపరి_ir, src0_చివర_అడ్డువరుస, ఖండం_2, ఖండం_src0_contig,
                                                              ఖండం_dst_contig, ne01, div_ne01);
            స్థిరరాశి uint32_t pref_ir = తదుపరి_ir + తదుపరి_ఖండం_కొలత;
            ఒకవేళ (pref_ir < src0_చివర_అడ్డువరుస) {
                స్థిరరాశి uint32_t pref_ఖండం_కొలత = unary_ఖండం_కొలత(pref_ir, src0_చివర_అడ్డువరుస, ఖండం_2, ఖండం_src0_contig,
                                                                  ఖండం_dst_contig, ne01, div_ne01);
                స్థిరరాశి size_t src0_pref_off = src0_contig ? (pref_ir * nb01) :
                    unary_అడ్డువరుస_స్థానభ్రంశం(pref_ir, ne01, ne02, div_ne01, div_ne02, div_ne012, nb01, nb02, nb03);
                dma_వరుస_నెట్టు(dma_q,
                    dma_make_దత్తాంశం(src0_vtcm, దత్తాంశం_src + src0_pref_off),
                    src0_అడ్డువరుస_కొలత_aligned, nb01, src0_దత్తాంశం_అడ్డువరుస_కొలత, pref_ఖండం_కొలత);

                ఒకవేళ (!uctx->ప్రసారం_బరువు) {
                    స్థిరరాశి size_t src1_pref_off = src1_contig ? (pref_ir * nb11) :
                        unary_అడ్డువరుస_స్థానభ్రంశం(pref_ir, ne01, ne02, div_ne01, div_ne02, div_ne012, nb11_bc, nb12_bc,
                                          nb13_bc);
                    dma_వరుస_నెట్టు(dma_q,
                        dma_make_దత్తాంశం(src1_vtcm, దత్తాంశం_src1 + src1_pref_off),
                        uctx->src1_అడ్డువరుస_కొలత_aligned, nb11, uctx->src1_దత్తాంశం_అడ్డువరుస_కొలత, pref_ఖండం_కొలత);
                }
            }
        }
        ir += ఖండం_కొలత_3;
    }

    dma_వరుస_flush(dma_q);
}

// 3. TRI row-block task with row index ir.
స్థిర శూన్య unary_thread_tri_f32(గుర్తులేని పూర్ణాంకం nth, గుర్తులేని పూర్ణాంకం ith, శూన్య * దత్తాంశం) {
    (శూన్య) nth;
    స్థిరరాశి నిర్మాణం htp_unary_context * uctx = (స్థిరరాశి నిర్మాణం htp_unary_context *) దత్తాంశం;
    నిర్మాణం htp_ops_context * octx = uctx->octx;
    స్థిరరాశి నిర్మాణం htp_tensor * src = octx->src[0];
    స్థిరరాశి నిర్మాణం htp_tensor * dst = octx->dst;
    నిర్మాణం htp_thread_trace * tr = &octx->ctx->trace[ith];

    htp_unary_preamble;

    స్థిరరాశి uint32_t src0_nrows_చొప్పున_thread = uctx->src0_nrows_చొప్పున_thread;
    స్థిరరాశి size_t src0_దత్తాంశం_అడ్డువరుస_కొలత = uctx->src0_దత్తాంశం_అడ్డువరుస_కొలత;
    స్థిరరాశి size_t dst_దత్తాంశం_అడ్డువరుస_కొలత  = uctx->dst_దత్తాంశం_అడ్డువరుస_కొలత;
    స్థిరరాశి size_t src0_అడ్డువరుస_కొలత_aligned = uctx->src0_అడ్డువరుస_కొలత_aligned;
    స్థిరరాశి size_t dst_అడ్డువరుస_కొలత_aligned  = uctx->dst_అడ్డువరుస_కొలత_aligned;

    స్థిరరాశి uint32_t src0_nrows = uctx->src0_nrows;
    స్థిరరాశి uint32_t src0_ప్రారంభం_అడ్డువరుస = uctx->అడ్డువరుస_ప్రారంభం + src0_nrows_చొప్పున_thread * ith;
    స్థిరరాశి uint32_t src0_చివర_అడ్డువరుస   = MIN(src0_ప్రారంభం_అడ్డువరుస + src0_nrows_చొప్పున_thread, uctx->అడ్డువరుస_ప్రారంభం + src0_nrows);

    ఒకవేళ (src0_ప్రారంభం_అడ్డువరుస >= src0_చివర_అడ్డువరుస) {
        తిరిగి;
    }

    స్థిరరాశి dma_addr_t దత్తాంశం_src = uctx->దత్తాంశం_src0;
    స్థిరరాశి dma_addr_t దత్తాంశం_dst = uctx->దత్తాంశం_dst;

    uint8_t * src0_vtcm_దత్తాంశం = uctx->vtcm_src0 + (ith * uctx->vtcm_src0_కొలత_చొప్పున_thread);
    uint8_t * dst_vtcm_దత్తాంశం  = uctx->vtcm_dst + (ith * uctx->vtcm_dst_కొలత_చొప్పున_thread);

    స్థిరరాశి size_t src0_vtcm_half_కొలత = uctx->src0_vtcm_half_కొలత;
    స్థిరరాశి size_t dst_vtcm_half_కొలత  = uctx->dst_vtcm_half_కొలత;

    స్థిరరాశి bool src0_contig = (nb02 == (size_t)ne01 * nb01) &&
                             (nb03 == (size_t)ne02 * nb02);
    స్థిరరాశి bool dst_contig  = (nb2  == (size_t)ne1  * nb1)  &&
                             (nb3  == (size_t)ne2  * nb2);

    స్థిరరాశి నిర్మాణం fastdiv_విలువలు * div_ne01  = &uctx->kparams->div_ne01;
    స్థిరరాశి నిర్మాణం fastdiv_విలువలు * div_ne02  = &uctx->kparams->div_ne02;
    స్థిరరాశి నిర్మాణం fastdiv_విలువలు * div_ne012 = &uctx->kparams->div_ne012;

    స్థిరరాశి uint32_t src0_max_ఖండం = src0_contig ? uctx->ఖండం : MIN((uint32_t)uctx->ఖండం, ne01);
    స్థిరరాశి uint32_t dst_max_ఖండం  = dst_contig  ? uctx->ఖండం : MIN((uint32_t)uctx->ఖండం, ne1);
    స్థిరరాశి uint32_t ఖండం_2 = MIN(src0_max_ఖండం, dst_max_ఖండం);
    ఒకవేళ (ఖండం_2 == 0) {
        FARF(ERROR, "unary-tri : current VTCM reservation %zu is too small, needed at least %zu\n",
             uctx->vtcm_src0_కొలత_చొప్పున_thread, src0_అడ్డువరుస_కొలత_aligned);
        తిరిగి;
    }

    dma_వరుస * dma_q = octx->ctx->dma[ith];

    కొరకు (uint32_t ir = src0_ప్రారంభం_అడ్డువరుస, vtcm_idx = 0; ir < src0_చివర_అడ్డువరుస && vtcm_idx < 2; vtcm_idx++) {
        స్థిరరాశి uint32_t ఖండం_కొలత_3 = unary_ఖండం_కొలత(ir, src0_చివర_అడ్డువరుస, ఖండం_2, src0_contig, dst_contig,
                                                     ne01, div_ne01);

        dma_వరుస_నెట్టు(dma_q,
            dma_make_దత్తాంశం(దత్తాంశం_dst, dst_vtcm_దత్తాంశం + (vtcm_idx * dst_vtcm_half_కొలత)),
            nb1, dst_అడ్డువరుస_కొలత_aligned, dst_దత్తాంశం_అడ్డువరుస_కొలత, 0);

        స్థిరరాశి size_t src0_off = src0_contig ? (ir * nb01) :
            unary_అడ్డువరుస_స్థానభ్రంశం(ir, ne01, ne02, div_ne01, div_ne02, div_ne012, nb01, nb02, nb03);
        dma_వరుస_నెట్టు(dma_q,
            dma_make_దత్తాంశం(src0_vtcm_దత్తాంశం + (vtcm_idx * src0_vtcm_half_కొలత), దత్తాంశం_src + src0_off),
            src0_అడ్డువరుస_కొలత_aligned, nb01, src0_దత్తాంశం_అడ్డువరుస_కొలత, ఖండం_కొలత_3);

        ir += ఖండం_కొలత_3;
    }

    unary_tri_compute_fn_t compute = (unary_tri_compute_fn_t) uctx->compute;

    కొరకు (uint32_t ir = src0_ప్రారంభం_అడ్డువరుస; ir < src0_చివర_అడ్డువరుస; ) {
        స్థిరరాశి uint32_t ఖండం_కొలత_3 = unary_ఖండం_కొలత(ir, src0_చివర_అడ్డువరుస, ఖండం_2, src0_contig, dst_contig,
                                                     ne01, div_ne01);

        శూన్య * dst_vtcm  = (శూన్య *) (uintptr_t) dma_వరుస_తీయి(dma_q).src;
        శూన్య * src0_vtcm = (శూన్య *) (uintptr_t) dma_వరుస_తీయి(dma_q).dst;

        htp_trace_event_ప్రారంభం(tr, HTP_TRACE_EVT_HVX_COMP, ir);
        compute(src0_vtcm, dst_vtcm, ఖండం_కొలత_3, ir, uctx);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ir);

        స్థిరరాశి size_t dst_off = dst_contig ? (ir * nb1) :
            unary_అడ్డువరుస_స్థానభ్రంశం(ir, ne1, ne2, div_ne01, div_ne02, div_ne012, nb1, nb2, nb3);
        dma_వరుస_నెట్టు(dma_q,
            dma_make_దత్తాంశం(దత్తాంశం_dst + dst_off, dst_vtcm),
            nb1, dst_అడ్డువరుస_కొలత_aligned, dst_దత్తాంశం_అడ్డువరుస_కొలత, ఖండం_కొలత_3);

        స్థిరరాశి uint32_t తదుపరి_ir = ir + ఖండం_కొలత_3;
        ఒకవేళ (తదుపరి_ir < src0_చివర_అడ్డువరుస) {
            స్థిరరాశి uint32_t తదుపరి_ఖండం_కొలత = unary_ఖండం_కొలత(తదుపరి_ir, src0_చివర_అడ్డువరుస, ఖండం_2, src0_contig,
                                                              dst_contig, ne01, div_ne01);
            స్థిరరాశి uint32_t pref_ir = తదుపరి_ir + తదుపరి_ఖండం_కొలత;
            ఒకవేళ (pref_ir < src0_చివర_అడ్డువరుస) {
                స్థిరరాశి uint32_t pref_ఖండం_కొలత = unary_ఖండం_కొలత(pref_ir, src0_చివర_అడ్డువరుస, ఖండం_2, src0_contig,
                                                                  dst_contig, ne01, div_ne01);
                స్థిరరాశి size_t src0_pref_off = src0_contig ? (pref_ir * nb01) :
                    unary_అడ్డువరుస_స్థానభ్రంశం(pref_ir, ne01, ne02, div_ne01, div_ne02, div_ne012, nb01, nb02, nb03);
                dma_వరుస_నెట్టు(dma_q,
                    dma_make_దత్తాంశం(src0_vtcm, దత్తాంశం_src + src0_pref_off),
                    src0_అడ్డువరుస_కొలత_aligned, nb01, src0_దత్తాంశం_అడ్డువరుస_కొలత, pref_ఖండం_కొలత);
            }
        }
        ir += ఖండం_కొలత_3;
    }

    dma_వరుస_flush(dma_q);
}

// 4. Pointwise tiled unary task.
స్థిర శూన్య unary_thread_tiled(గుర్తులేని పూర్ణాంకం nth, గుర్తులేని పూర్ణాంకం ith, శూన్య * దత్తాంశం) {
    (శూన్య) nth;
    స్థిరరాశి నిర్మాణం htp_unary_context * uctx = (స్థిరరాశి నిర్మాణం htp_unary_context *) దత్తాంశం;
    నిర్మాణం htp_ops_context * octx = uctx->octx;
    స్థిరరాశి నిర్మాణం htp_tensor * src = octx->src[0];
    స్థిరరాశి నిర్మాణం htp_tensor * dst = octx->dst;
    నిర్మాణం htp_thread_trace * tr = &octx->ctx->trace[ith];

    htp_unary_preamble;

    స్థిరరాశి uint32_t src0_nrows_చొప్పున_thread = uctx->src0_nrows_చొప్పున_thread;
    స్థిరరాశి uint32_t నిలువు_tile              = uctx->నిలువు_tile;

    స్థిరరాశి uint32_t src0_nrows     = uctx->src0_nrows;
    స్థిరరాశి uint32_t src0_ప్రారంభం_అడ్డువరుస = uctx->అడ్డువరుస_ప్రారంభం + src0_nrows_చొప్పున_thread * ith;
    స్థిరరాశి uint32_t src0_చివర_అడ్డువరుస   = MIN(src0_ప్రారంభం_అడ్డువరుస + src0_nrows_చొప్పున_thread, uctx->అడ్డువరుస_ప్రారంభం + src0_nrows);

    ఒకవేళ (src0_ప్రారంభం_అడ్డువరుస >= src0_చివర_అడ్డువరుస) {
        తిరిగి;
    }

    స్థిరరాశి dma_addr_t దత్తాంశం_src = uctx->దత్తాంశం_src0;
    స్థిరరాశి dma_addr_t దత్తాంశం_dst = uctx->దత్తాంశం_dst;

    uint8_t * src0_vtcm_దత్తాంశం = uctx->vtcm_src0 + (ith * uctx->vtcm_src0_కొలత_చొప్పున_thread);
    uint8_t * dst_vtcm_దత్తాంశం  = uctx->vtcm_dst + (ith * uctx->vtcm_dst_కొలత_చొప్పున_thread);

    స్థిరరాశి size_t src0_half = uctx->src0_vtcm_half_కొలత;
    స్థిరరాశి size_t dst_half  = uctx->dst_vtcm_half_కొలత;

    dma_వరుస * dma_q = octx->ctx->dma[ith];

    స్థిరరాశి నిర్మాణం fastdiv_విలువలు * div_ne01  = &uctx->kparams->div_ne01;
    స్థిరరాశి నిర్మాణం fastdiv_విలువలు * div_ne02  = &uctx->kparams->div_ne02;
    స్థిరరాశి నిర్మాణం fastdiv_విలువలు * div_ne012 = &uctx->kparams->div_ne012;
    స్థిరరాశి నిర్మాణం fastdiv_విలువలు * div_tpr   = &uctx->kparams->div_tpr;

    స్థిరరాశి uint32_t tiles_చొప్పున_అడ్డువరుస = (ne0 + నిలువు_tile - 1) / నిలువు_tile;

    స్థిరరాశి bool src0_contig = (nb02 == (size_t)ne01 * nb01) &&
                             (nb03 == (size_t)ne02 * nb02);
    స్థిరరాశి bool dst_contig  = (nb2  == (size_t)ne1  * nb1)  &&
                             (nb3  == (size_t)ne2  * nb2);

    స్థిరరాశి uint32_t మొత్తం_tiles = (src0_చివర_అడ్డువరుస - src0_ప్రారంభం_అడ్డువరుస) * tiles_చొప్పున_అడ్డువరుస;

    కొరకు (uint32_t t = 0, vtcm_idx = 0; t < మొత్తం_tiles && vtcm_idx < 2; t++, vtcm_idx++) {
        స్థిరరాశి uint32_t అడ్డువరుస  = src0_ప్రారంభం_అడ్డువరుస + t / tiles_చొప్పున_అడ్డువరుస;
        స్థిరరాశి uint32_t నిలువు  = (t % tiles_చొప్పున_అడ్డువరుస) * నిలువు_tile;
        స్థిరరాశి uint32_t tw   = MIN(నిలువు_tile, ne0 - నిలువు);
        స్థిరరాశి size_t   tb   = (size_t) tw * పరిమాణం(దశాంశం);
        స్థిరరాశి size_t   soff = (src0_contig ? (అడ్డువరుస * nb01) :
                               unary_అడ్డువరుస_స్థానభ్రంశం(అడ్డువరుస, ne01, ne02, div_ne01, div_ne02, div_ne012, nb01, nb02, nb03)) +
                               (size_t) నిలువు * పరిమాణం(దశాంశం);

        dma_వరుస_నెట్టు(dma_q, dma_make_దత్తాంశం(దత్తాంశం_dst, dst_vtcm_దత్తాంశం + (vtcm_idx * dst_half)), 0, 0, 0, 0);
        dma_వరుస_నెట్టు(dma_q, dma_make_దత్తాంశం(src0_vtcm_దత్తాంశం + (vtcm_idx * src0_half), దత్తాంశం_src + soff), tb, tb, tb, 1);
    }

    unary_tile_compute_fn_t compute = (unary_tile_compute_fn_t) uctx->compute;

    uint32_t అడ్డువరుస = src0_ప్రారంభం_అడ్డువరుస;
    uint32_t నిలువు = 0;
    uint32_t tile_లోపల_అడ్డువరుస = 0;

    uint32_t prow = src0_ప్రారంభం_అడ్డువరుస + fastdiv(2, div_tpr);
    uint32_t pcol = fastmodulo(2, tiles_చొప్పున_అడ్డువరుస, div_tpr) * నిలువు_tile;
    uint32_t ptile_లోపల_అడ్డువరుస = fastmodulo(2, tiles_చొప్పున_అడ్డువరుస, div_tpr);

    కొరకు (uint32_t t = 0; t < మొత్తం_tiles; t++) {
        శూన్య * dst_vtcm = (శూన్య *) (uintptr_t) dma_వరుస_తీయి(dma_q).src;
        శూన్య * src_vtcm = (శూన్య *) (uintptr_t) dma_వరుస_తీయి(dma_q).dst;

        స్థిరరాశి uint32_t tw = MIN(నిలువు_tile, ne0 - నిలువు);

        htp_trace_event_ప్రారంభం(tr, HTP_TRACE_EVT_HVX_COMP, t);
        compute(dst_vtcm, src_vtcm, tw, uctx);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, t);

        స్థిరరాశి size_t doff = (dst_contig ? (అడ్డువరుస * nb1) :
                             unary_అడ్డువరుస_స్థానభ్రంశం(అడ్డువరుస, ne1, ne2, div_ne01, div_ne02, div_ne012, nb1, nb2, nb3)) +
                             (size_t) నిలువు * పరిమాణం(దశాంశం);
        స్థిరరాశి size_t tb   = (size_t) tw * పరిమాణం(దశాంశం);
        dma_వరుస_నెట్టు(dma_q, dma_make_దత్తాంశం(దత్తాంశం_dst + doff, dst_vtcm), tb, tb, tb, 1);

        స్థిరరాశి uint32_t pt = t + 2;
        ఒకవేళ (pt < మొత్తం_tiles) {
            స్థిరరాశి uint32_t ptw  = MIN(నిలువు_tile, ne0 - pcol);
            స్థిరరాశి size_t   ptb  = (size_t) ptw * పరిమాణం(దశాంశం);
            స్థిరరాశి size_t   psoff = (src0_contig ? (prow * nb01) :
                                    unary_అడ్డువరుస_స్థానభ్రంశం(prow, ne01, ne02, div_ne01, div_ne02, div_ne012, nb01, nb02,
                                                     nb03)) +
                                   (size_t) pcol * పరిమాణం(దశాంశం);
            dma_వరుస_నెట్టు(dma_q, dma_make_దత్తాంశం(src_vtcm, దత్తాంశం_src + psoff), ptb, ptb, ptb, 1);
        }

        tile_లోపల_అడ్డువరుస++;
        నిలువు += నిలువు_tile;
        ఒకవేళ (tile_లోపల_అడ్డువరుస == tiles_చొప్పున_అడ్డువరుస) {
            tile_లోపల_అడ్డువరుస = 0;
            నిలువు = 0;
            అడ్డువరుస++;
        }

        ptile_లోపల_అడ్డువరుస++;
        pcol += నిలువు_tile;
        ఒకవేళ (ptile_లోపల_అడ్డువరుస == tiles_చొప్పున_అడ్డువరుస) {
            ptile_లోపల_అడ్డువరుస = 0;
            pcol = 0;
            prow++;
        }
    }

    dma_వరుస_flush(dma_q);
}

// 5. TRI tiled task.
స్థిర శూన్య unary_thread_tiled_tri_f32(గుర్తులేని పూర్ణాంకం nth, గుర్తులేని పూర్ణాంకం ith, శూన్య * దత్తాంశం) {
    (శూన్య) nth;
    స్థిరరాశి నిర్మాణం htp_unary_context * uctx = (స్థిరరాశి నిర్మాణం htp_unary_context *) దత్తాంశం;
    నిర్మాణం htp_ops_context * octx = uctx->octx;
    స్థిరరాశి నిర్మాణం htp_tensor * src = octx->src[0];
    స్థిరరాశి నిర్మాణం htp_tensor * dst = octx->dst;
    నిర్మాణం htp_thread_trace * tr = &octx->ctx->trace[ith];

    htp_unary_preamble;

    స్థిరరాశి uint32_t src0_nrows_చొప్పున_thread = uctx->src0_nrows_చొప్పున_thread;
    స్థిరరాశి int32_t * op_params            = octx->op_params;
    స్థిరరాశి uint32_t నిలువు_tile              = uctx->నిలువు_tile;

    స్థిరరాశి uint32_t src0_nrows     = uctx->src0_nrows;
    స్థిరరాశి uint32_t src0_ప్రారంభం_అడ్డువరుస = uctx->అడ్డువరుస_ప్రారంభం + src0_nrows_చొప్పున_thread * ith;
    స్థిరరాశి uint32_t src0_చివర_అడ్డువరుస   = MIN(src0_ప్రారంభం_అడ్డువరుస + src0_nrows_చొప్పున_thread, uctx->అడ్డువరుస_ప్రారంభం + src0_nrows);

    ఒకవేళ (src0_ప్రారంభం_అడ్డువరుస >= src0_చివర_అడ్డువరుస) {
        తిరిగి;
    }

    స్థిరరాశి dma_addr_t దత్తాంశం_src = uctx->దత్తాంశం_src0;
    స్థిరరాశి dma_addr_t దత్తాంశం_dst = uctx->దత్తాంశం_dst;

    uint8_t * src0_vtcm_దత్తాంశం = uctx->vtcm_src0 + (ith * uctx->vtcm_src0_కొలత_చొప్పున_thread);
    uint8_t * dst_vtcm_దత్తాంశం  = uctx->vtcm_dst + (ith * uctx->vtcm_dst_కొలత_చొప్పున_thread);

    స్థిరరాశి size_t src0_half = uctx->src0_vtcm_half_కొలత;
    స్థిరరాశి size_t dst_half  = uctx->dst_vtcm_half_కొలత;

    dma_వరుస * dma_q = octx->ctx->dma[ith];

    స్థిరరాశి నిర్మాణం fastdiv_విలువలు * div_ne01  = &uctx->kparams->div_ne01;
    స్థిరరాశి నిర్మాణం fastdiv_విలువలు * div_ne02  = &uctx->kparams->div_ne02;
    స్థిరరాశి నిర్మాణం fastdiv_విలువలు * div_ne012 = &uctx->kparams->div_ne012;
    స్థిరరాశి నిర్మాణం fastdiv_విలువలు * div_tpr   = &uctx->kparams->div_tpr;

    స్థిరరాశి uint32_t tiles_చొప్పున_అడ్డువరుస = (ne0 + నిలువు_tile - 1) / నిలువు_tile;
    స్థిరరాశి int32_t  tri_ttype     = op_params[0];

    స్థిరరాశి bool src0_contig = (nb02 == (size_t)ne01 * nb01) &&
                             (nb03 == (size_t)ne02 * nb02);
    స్థిరరాశి bool dst_contig  = (nb2  == (size_t)ne1  * nb1)  &&
                             (nb3  == (size_t)ne2  * nb2);

    స్థిరరాశి uint32_t మొత్తం_tiles = (src0_చివర_అడ్డువరుస - src0_ప్రారంభం_అడ్డువరుస) * tiles_చొప్పున_అడ్డువరుస;

    కొరకు (uint32_t t = 0, vtcm_idx = 0; t < మొత్తం_tiles && vtcm_idx < 2; t++, vtcm_idx++) {
        స్థిరరాశి uint32_t అడ్డువరుస  = src0_ప్రారంభం_అడ్డువరుస + t / tiles_చొప్పున_అడ్డువరుస;
        స్థిరరాశి uint32_t నిలువు  = (t % tiles_చొప్పున_అడ్డువరుస) * నిలువు_tile;
        స్థిరరాశి uint32_t tw   = MIN(నిలువు_tile, ne0 - నిలువు);
        స్థిరరాశి size_t   tb   = (size_t) tw * పరిమాణం(దశాంశం);
        స్థిరరాశి size_t   soff = (src0_contig ? (అడ్డువరుస * nb01) :
                               unary_అడ్డువరుస_స్థానభ్రంశం(అడ్డువరుస, ne01, ne02, div_ne01, div_ne02, div_ne012, nb01, nb02, nb03)) +
                               (size_t) నిలువు * పరిమాణం(దశాంశం);

        dma_వరుస_నెట్టు(dma_q, dma_make_దత్తాంశం(దత్తాంశం_dst, dst_vtcm_దత్తాంశం + (vtcm_idx * dst_half)), 0, 0, 0, 0);
        dma_వరుస_నెట్టు(dma_q, dma_make_దత్తాంశం(src0_vtcm_దత్తాంశం + (vtcm_idx * src0_half), దత్తాంశం_src + soff), tb, tb, tb, 1);
    }

    unary_tiled_tri_compute_fn_t compute = (unary_tiled_tri_compute_fn_t) uctx->compute;

    uint32_t అడ్డువరుస = src0_ప్రారంభం_అడ్డువరుస;
    uint32_t నిలువు = 0;
    uint32_t tile_లోపల_అడ్డువరుస = 0;
    uint32_t i01 = fastmodulo(అడ్డువరుస, ne01, div_ne01);

    uint32_t prow = src0_ప్రారంభం_అడ్డువరుస + fastdiv(2, div_tpr);
    uint32_t pcol = fastmodulo(2, tiles_చొప్పున_అడ్డువరుస, div_tpr) * నిలువు_tile;
    uint32_t ptile_లోపల_అడ్డువరుస = fastmodulo(2, tiles_చొప్పున_అడ్డువరుస, div_tpr);

    కొరకు (uint32_t t = 0; t < మొత్తం_tiles; t++) {
        శూన్య * dst_vtcm = (శూన్య *) (uintptr_t) dma_వరుస_తీయి(dma_q).src;
        శూన్య * src_vtcm = (శూన్య *) (uintptr_t) dma_వరుస_తీయి(dma_q).dst;

        స్థిరరాశి uint32_t tw = MIN(నిలువు_tile, ne0 - నిలువు);

        htp_trace_event_ప్రారంభం(tr, HTP_TRACE_EVT_HVX_COMP, t);
        compute(src_vtcm, dst_vtcm, tw, నిలువు, i01, ne0, tri_ttype);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, t);

        స్థిరరాశి size_t doff = (dst_contig ? (అడ్డువరుస * nb1) :
                             unary_అడ్డువరుస_స్థానభ్రంశం(అడ్డువరుస, ne1, ne2, div_ne01, div_ne02, div_ne012, nb1, nb2, nb3)) +
                             (size_t) నిలువు * పరిమాణం(దశాంశం);
        స్థిరరాశి size_t tb   = (size_t) tw * పరిమాణం(దశాంశం);
        dma_వరుస_నెట్టు(dma_q, dma_make_దత్తాంశం(దత్తాంశం_dst + doff, dst_vtcm), tb, tb, tb, 1);

        స్థిరరాశి uint32_t pt = t + 2;
        ఒకవేళ (pt < మొత్తం_tiles) {
            స్థిరరాశి uint32_t ptw  = MIN(నిలువు_tile, ne0 - pcol);
            స్థిరరాశి size_t   ptb  = (size_t) ptw * పరిమాణం(దశాంశం);
            స్థిరరాశి size_t   psoff = (src0_contig ? (prow * nb01) :
                                    unary_అడ్డువరుస_స్థానభ్రంశం(prow, ne01, ne02, div_ne01, div_ne02, div_ne012, nb01, nb02,
                                                     nb03)) +
                                   (size_t) pcol * పరిమాణం(దశాంశం);
            dma_వరుస_నెట్టు(dma_q, dma_make_దత్తాంశం(src_vtcm, దత్తాంశం_src + psoff), ptb, ptb, ptb, 1);
        }

        tile_లోపల_అడ్డువరుస++;
        నిలువు += నిలువు_tile;
        ఒకవేళ (tile_లోపల_అడ్డువరుస == tiles_చొప్పున_అడ్డువరుస) {
            tile_లోపల_అడ్డువరుస = 0;
            నిలువు = 0;
            అడ్డువరుస++;
            i01++;
            ఒకవేళ (i01 == ne01) {
                i01 = 0;
            }
        }

        ptile_లోపల_అడ్డువరుస++;
        pcol += నిలువు_tile;
        ఒకవేళ (ptile_లోపల_అడ్డువరుస == tiles_చొప్పున_అడ్డువరుస) {
            ptile_లోపల_అడ్డువరుస = 0;
            pcol = 0;
            prow++;
        }
    }

    dma_వరుస_flush(dma_q);
}

స్థిర పూర్ణాంకం execute_op_unary(నిర్మాణం htp_ops_context * octx) {
    పూర్ణాంకం err = HTP_స్థితి_OK;

    స్థిరరాశి నిర్మాణం htp_tensor * src0 = octx->src[0];
    స్థిరరాశి నిర్మాణం htp_tensor * dst  = octx->dst;

    స్థిరరాశి bool is_f16 = (src0->type == HTP_TYPE_F16);

    స్థిరరాశి అక్షరం * op_type = ఖాళీ;

    ఎంపిక (octx->op) {
        స్థితి HTP_OP_NORM:            op_type = is_f16 ? "norm-f16"     : "norm-f32";         ఆపు;
        స్థితి HTP_OP_RMS_NORM:        op_type = is_f16 ? "rmsnorm-f16"  : "rmsnorm-f32";      ఆపు;
        స్థితి HTP_OP_RMS_NORM_MUL:    op_type = "rmsnorm-mul-f32";                            ఆపు;
        స్థితి HTP_OP_SCALE:           op_type = is_f16 ? "scale-f16"    : "scale-f32";        ఆపు;
        స్థితి HTP_OP_CLAMP:           op_type = is_f16 ? "clamp-f16"    : "clamp-f32";        ఆపు;
        స్థితి HTP_OP_LEAKY_RELU:      op_type = "leaky-relu-f32";                             ఆపు;
        స్థితి HTP_OP_SQR:             op_type = is_f16 ? "sqr-f16"      : "sqr-f32";          ఆపు;
        స్థితి HTP_OP_SQRT:            op_type = is_f16 ? "sqrt-f16"     : "sqrt-f32";         ఆపు;
        స్థితి HTP_OP_UNARY_NEG:       op_type = "neg-f32";                                    ఆపు;
        స్థితి HTP_OP_UNARY_EXP:       op_type = "exp-f32";                                    ఆపు;
        స్థితి HTP_OP_UNARY_SIGMOID:   op_type = "sigmoid-f32";                                ఆపు;
        స్థితి HTP_OP_UNARY_SILU:      op_type = "silu-f32";                                   ఆపు;
        స్థితి HTP_OP_UNARY_GELU:      op_type = "gelu-f32";                                   ఆపు;
        స్థితి HTP_OP_UNARY_GELU_శీఘ్ర: op_type = "gelu-quick-f32";                             ఆపు;
        స్థితి HTP_OP_UNARY_GELU_ERF:  op_type = "gelu-erf-f32";                               ఆపు;
        స్థితి HTP_OP_UNARY_SOFTPLUS:  op_type = "softplus-f32";                               ఆపు;
        స్థితి HTP_OP_UNARY_TANH:      op_type = "tanh-f32";                                   ఆపు;
        స్థితి HTP_OP_UNARY_ABS:       op_type = is_f16 ? "abs-f16"      : "abs-f32";          ఆపు;
        స్థితి HTP_OP_UNARY_LOG:       op_type = is_f16 ? "log-f16"      : "log-f32";          ఆపు;
        స్థితి HTP_OP_UNARY_RELU:      op_type = "relu-f32";                                   ఆపు;
        స్థితి HTP_OP_UNARY_అడుగు:      op_type = is_f16 ? "step-f16"     : "step-f32";         ఆపు;
        స్థితి HTP_OP_L2_NORM:         op_type = is_f16 ? "l2norm-f16"   : "l2norm-f32";       ఆపు;
        స్థితి HTP_OP_TRI:             op_type = "tri-f32";                                    ఆపు;
        స్థితి HTP_OP_HADAMARD:        op_type = is_f16 ? "hadamard-f16" : "hadamard-f32";     ఆపు;
        అప్రమేయ:
            FARF(ERROR, "Unsupported unary Op %u\n", octx->op);
            తిరిగి HTP_స్థితి_NO_SUPPORT;
    }

    // F16 only has row-block kernels for this subset of ops (see the dispatch switch
    // below) - reject everything else up front, before touching kparams/VTCM.
    ఒకవేళ (is_f16) {
        ఎంపిక (octx->op) {
            స్థితి HTP_OP_NORM:
            స్థితి HTP_OP_RMS_NORM:
            స్థితి HTP_OP_SCALE:
            స్థితి HTP_OP_CLAMP:
            స్థితి HTP_OP_SQR:
            స్థితి HTP_OP_SQRT:
            స్థితి HTP_OP_L2_NORM:
            స్థితి HTP_OP_UNARY_ABS:
            స్థితి HTP_OP_UNARY_LOG:
            స్థితి HTP_OP_UNARY_అడుగు:
            స్థితి HTP_OP_HADAMARD:
                ఆపు;
            అప్రమేయ:
                FARF(ERROR, "unary-%s: not supported for F16\n", op_type);
                తిరిగి HTP_స్థితి_NO_SUPPORT;
        }
    }

    స్థిరరాశి నిర్మాణం htp_unary_kernel_params * kparams = (స్థిరరాశి నిర్మాణం htp_unary_kernel_params *) octx->kernel_params;

    ఒకవేళ (!htp_ops_context_సమితి_n_దారాలు(octx, kparams->n_దారాలు)) {
        తిరిగి HTP_స్థితి_INVAL_PARAMS;
    }

    స్థిరరాశి uint32_t src0_nrows = src0->ne[1] * src0->ne[2] * src0->ne[3];
    స్థిరరాశి size_t src0_elem_కొలత = is_f16 ? పరిమాణం(_Float16) : పరిమాణం(దశాంశం);
    స్థిరరాశి size_t dst_elem_కొలత  = (dst->type == HTP_TYPE_F16) ? పరిమాణం(_Float16) : పరిమాణం(దశాంశం);
    స్థిరరాశి size_t src0_దత్తాంశం_అడ్డువరుస_కొలత = src0->ne[0] * src0_elem_కొలత;
    స్థిరరాశి size_t dst_దత్తాంశం_అడ్డువరుస_కొలత  = dst->ne[0]  * dst_elem_కొలత;

    uint32_t అడ్డువరుస_ప్రారంభం = 0;
    uint32_t nrows     = src0_nrows;

    ఒకవేళ (octx->ctx->mdev.లెక్క > 1) {
        uint32_t అడ్డువరుసలు_చొప్పున_chunk = 0;
        htp_tensor_mdev_అడ్డువరుసలు_చొప్పున_chunk(dst, (uint32_t) dst_elem_కొలత, (uint32_t) dst_దత్తాంశం_అడ్డువరుస_కొలత, &అడ్డువరుసలు_చొప్పున_chunk);
        స్థిరరాశి నిర్మాణం htp_tensor_mdev_range range = htp_tensor_mdev_విభజన(src0_nrows, అడ్డువరుసలు_చొప్పున_chunk, octx->ctx->mdev.idx, octx->ctx->mdev.లెక్క, &octx->ctx->mdev.లెక్క_div);
        అడ్డువరుస_ప్రారంభం = range.ప్రారంభం;
        nrows     = range.లెక్క;
    }

    ఒకవేళ (nrows == 0) {
        తిరిగి HTP_స్థితి_OK;
    }

    స్థిరరాశి uint32_t n_దారాలు = octx->n_దారాలు;

    స్థిరరాశి size_t src0_అడ్డువరుస_కొలత_aligned = kparams->src0_అడ్డువరుస_కొలత_aligned;
    స్థిరరాశి size_t dst_అడ్డువరుస_కొలత_aligned  = kparams->dst_అడ్డువరుస_కొలత_aligned;

    // Always 0 for F16 - htp_unary_vtcm_layout_build() keeps F16 on the row-block path,
    // since only F32 has unary_task_f32_tiled_* kernels.
    స్థిరరాశి uint32_t నిలువు_tile = kparams->నిలువు_tile;

    size_t src1_దత్తాంశం_అడ్డువరుస_కొలత = 0;
    size_t src1_అడ్డువరుస_కొలత_aligned = kparams->src1_అడ్డువరుస_కొలత_aligned;
    bool ప్రసారం_బరువు = kparams->ప్రసారం_బరువు;
    స్థిరరాశి నిర్మాణం htp_tensor * src1 = ఖాళీ;

    // RMS_NORM_MUL fusion is F32-only (its weight tensor is always F32; see
    // try_fuse_node()'s type guard), so this never triggers when is_f16 is true.
    ఒకవేళ (octx->op == HTP_OP_RMS_NORM_MUL) {
        src1 = octx->src[1];
        src1_దత్తాంశం_అడ్డువరుస_కొలత = src1->ne[0] * పరిమాణం(దశాంశం);
    }

    ఒకవేళ (octx->ctx->vtcm_కొలత < (size_t)kparams->vtcm_కొలత) {
        FARF(ERROR, "unary-%s : current VTCM reservation %zu is too small, needed %zu\n", op_type, octx->ctx->vtcm_కొలత, (size_t)kparams->vtcm_కొలత);
        తిరిగి HTP_స్థితి_VTCM_TOO_SMALL;
    }

    octx->src0_spad.src = ఖాళీ;
    octx->src1_spad.src = ఖాళీ;
    octx->dst_spad.src  = ఖాళీ;

    FARF(ఎగువ_2, "%s: (%ux%ux%ux%u) -> (%ux%ux%ux%u) : src0-vtcm-size %u src1-vtcm-size %u dst-vtcm-size %u\n", op_type,
         src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3], dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3],
         kparams->vtcm_src0_కొలత, kparams->vtcm_src1_కొలత, kparams->vtcm_dst_కొలత);

    uint8_t * స్థిరరాశి పునాది = (uint8_t *) octx->ctx->vtcm_పునాది;
    నిర్మాణం htp_unary_context uctx = {
        .octx                  = octx,
        .kparams               = kparams,
        .src0_nrows_చొప్పున_thread = fastdiv(nrows + n_దారాలు - 1, &octx->n_దారాలు_div),
        .src0_nrows            = nrows,
        .అడ్డువరుస_ప్రారంభం             = అడ్డువరుస_ప్రారంభం,

        .దత్తాంశం_src0             = src0->దత్తాంశం,
        .దత్తాంశం_src1             = (octx->op == HTP_OP_RMS_NORM_MUL) ? src1->దత్తాంశం : 0,
        .దత్తాంశం_dst              = dst->దత్తాంశం,

        .src0_దత్తాంశం_అడ్డువరుస_కొలత    = src0_దత్తాంశం_అడ్డువరుస_కొలత,
        .src1_దత్తాంశం_అడ్డువరుస_కొలత    = src1_దత్తాంశం_అడ్డువరుస_కొలత,
        .dst_దత్తాంశం_అడ్డువరుస_కొలత     = dst_దత్తాంశం_అడ్డువరుస_కొలత,

        .src0_అడ్డువరుస_కొలత_aligned = src0_అడ్డువరుస_కొలత_aligned,
        .src1_అడ్డువరుస_కొలత_aligned = src1_అడ్డువరుస_కొలత_aligned,
        .dst_అడ్డువరుస_కొలత_aligned  = dst_అడ్డువరుస_కొలత_aligned,

        .src0_vtcm_half_కొలత   = kparams->vtcm_src0_కొలత_చొప్పున_thread / 2,
        .src1_vtcm_half_కొలత   = (octx->op == HTP_OP_RMS_NORM_MUL) ? (kparams->vtcm_src1_కొలత_చొప్పున_thread / (ప్రసారం_బరువు ? 1 : 2)) : 0,
        .dst_vtcm_half_కొలత    = kparams->vtcm_dst_కొలత_చొప్పున_thread / 2,

        .ఖండం                 = kparams->ఖండం,
        .nc                    = src0->ne[0],
        .నిలువు_tile              = నిలువు_tile,
        .ప్రసారం_బరువు      = ప్రసారం_బరువు,

        .vtcm_src0             = VTCM_LAYOUT_PTR(uint8_t, పునాది, 0),
        .vtcm_src1             = VTCM_LAYOUT_PTR_OPTIONAL(uint8_t, పునాది, kparams->vtcm_src0_కొలత, kparams->vtcm_src1_కొలత > 0),
        .vtcm_dst              = VTCM_LAYOUT_PTR(uint8_t, పునాది, kparams->vtcm_src0_కొలత + kparams->vtcm_src1_కొలత),

        .vtcm_src0_కొలత_చొప్పున_thread = kparams->vtcm_src0_కొలత_చొప్పున_thread,
        .vtcm_src1_కొలత_చొప్పున_thread = kparams->vtcm_src1_కొలత_చొప్పున_thread,
        .vtcm_dst_కొలత_చొప్పున_thread  = kparams->vtcm_dst_కొలత_చొప్పున_thread,
    };

    FARF(ఎగువ_2, "%s: %s mode (col_tile %u)\n", op_type, నిలువు_tile ? "tiled" : "row-block", నిలువు_tile);

    కార్మికుడు_callback_t task_func = ఖాళీ;
    శూన్య * compute_func = ఖాళీ;

    ఒకవేళ (నిలువు_tile) {
        task_func = unary_thread_tiled;
        ఎంపిక (octx->op) {
            స్థితి HTP_OP_SCALE:           compute_func = (శూన్య *) tile_scale_f32;          ఆపు;
            స్థితి HTP_OP_CLAMP:           compute_func = (శూన్య *) tile_clamp_f32;          ఆపు;
            స్థితి HTP_OP_LEAKY_RELU:      compute_func = (శూన్య *) tile_leaky_relu_f32;     ఆపు;
            స్థితి HTP_OP_SQR:             compute_func = (శూన్య *) tile_sqr_f32;            ఆపు;
            స్థితి HTP_OP_SQRT:            compute_func = (శూన్య *) tile_sqrt_f32;           ఆపు;
            స్థితి HTP_OP_UNARY_NEG:       compute_func = (శూన్య *) tile_neg_f32;            ఆపు;
            స్థితి HTP_OP_UNARY_EXP:       compute_func = (శూన్య *) tile_exp_f32;            ఆపు;
            స్థితి HTP_OP_UNARY_SIGMOID:   compute_func = (శూన్య *) tile_sigmoid_f32;        ఆపు;
            స్థితి HTP_OP_UNARY_SILU:      compute_func = (శూన్య *) tile_silu_f32;           ఆపు;
            స్థితి HTP_OP_UNARY_GELU:      compute_func = (శూన్య *) tile_gelu_f32;           ఆపు;
            స్థితి HTP_OP_UNARY_GELU_శీఘ్ర: compute_func = (శూన్య *) tile_gelu_శీఘ్ర_f32;     ఆపు;
            స్థితి HTP_OP_UNARY_GELU_ERF:  compute_func = (శూన్య *) tile_gelu_erf_f32;       ఆపు;
            స్థితి HTP_OP_UNARY_SOFTPLUS:  compute_func = (శూన్య *) tile_softplus_f32;       ఆపు;
            స్థితి HTP_OP_UNARY_TANH:      compute_func = (శూన్య *) tile_tanh_f32;           ఆపు;
            స్థితి HTP_OP_UNARY_ABS:       compute_func = (శూన్య *) tile_abs_f32;            ఆపు;
            స్థితి HTP_OP_UNARY_LOG:       compute_func = (శూన్య *) tile_log_f32;            ఆపు;
            స్థితి HTP_OP_UNARY_RELU:      compute_func = (శూన్య *) tile_relu_f32;           ఆపు;
            స్థితి HTP_OP_UNARY_అడుగు:      compute_func = (శూన్య *) tile_అడుగు_f32;           ఆపు;
            స్థితి HTP_OP_TRI:
                task_func    = unary_thread_tiled_tri_f32;
                compute_func = (శూన్య *) tri_apply_tile_f32;
                ఆపు;
            అప్రమేయ:                     ఆపు;
        }
    } లేకపోతే ఒకవేళ (is_f16) {
        task_func = unary_thread_అడ్డువరుస_ఖండం;
        ఎంపిక (octx->op) {
            స్థితి HTP_OP_NORM:            compute_func = (శూన్య *) norm_f16;                ఆపు;
            స్థితి HTP_OP_RMS_NORM:        compute_func = (శూన్య *) rms_norm_f16;            ఆపు;
            స్థితి HTP_OP_SCALE:           compute_func = (శూన్య *) scale_f16;               ఆపు;
            స్థితి HTP_OP_CLAMP:           compute_func = (శూన్య *) clamp_f16;               ఆపు;
            స్థితి HTP_OP_SQR:             compute_func = (శూన్య *) sqr_f16;                 ఆపు;
            స్థితి HTP_OP_SQRT:            compute_func = (శూన్య *) sqrt_f16;                ఆపు;
            స్థితి HTP_OP_L2_NORM:         compute_func = (శూన్య *) l2_norm_f16;             ఆపు;
            స్థితి HTP_OP_UNARY_ABS:       compute_func = (శూన్య *) abs_f16;                 ఆపు;
            స్థితి HTP_OP_UNARY_LOG:       compute_func = (శూన్య *) log_f16;                 ఆపు;
            స్థితి HTP_OP_UNARY_అడుగు:      compute_func = (శూన్య *) అడుగు_f16;                ఆపు;
            స్థితి HTP_OP_HADAMARD:        compute_func = (శూన్య *) hadamard_f16;            ఆపు;
            అప్రమేయ:                     ఆపు;
        }
    } లేకపోతే {
        task_func = unary_thread_అడ్డువరుస_ఖండం;
        ఎంపిక (octx->op) {
            స్థితి HTP_OP_NORM:            compute_func = (శూన్య *) norm_f32;                ఆపు;
            స్థితి HTP_OP_RMS_NORM:        compute_func = (శూన్య *) rms_norm_f32;            ఆపు;
            స్థితి HTP_OP_RMS_NORM_MUL:
                task_func    = unary_thread_rms_norm_mul_f32;
                compute_func = (శూన్య *) rms_norm_mul_f32;
                ఆపు;
            స్థితి HTP_OP_SCALE:           compute_func = (శూన్య *) scale_f32;               ఆపు;
            స్థితి HTP_OP_CLAMP:           compute_func = (శూన్య *) clamp_f32;               ఆపు;
            స్థితి HTP_OP_LEAKY_RELU:      compute_func = (శూన్య *) leaky_relu_f32;          ఆపు;
            స్థితి HTP_OP_SQR:             compute_func = (శూన్య *) sqr_f32;                 ఆపు;
            స్థితి HTP_OP_SQRT:            compute_func = (శూన్య *) sqrt_f32;                ఆపు;
            స్థితి HTP_OP_UNARY_NEG:       compute_func = (శూన్య *) neg_f32;                 ఆపు;
            స్థితి HTP_OP_UNARY_EXP:       compute_func = (శూన్య *) exp_f32;                 ఆపు;
            స్థితి HTP_OP_UNARY_SIGMOID:   compute_func = (శూన్య *) sigmoid_f32;             ఆపు;
            స్థితి HTP_OP_UNARY_SILU:      compute_func = (శూన్య *) silu_f32;                ఆపు;
            స్థితి HTP_OP_UNARY_GELU:      compute_func = (శూన్య *) gelu_f32;                ఆపు;
            స్థితి HTP_OP_UNARY_GELU_శీఘ్ర: compute_func = (శూన్య *) gelu_శీఘ్ర_f32;          ఆపు;
            స్థితి HTP_OP_UNARY_GELU_ERF:  compute_func = (శూన్య *) gelu_erf_f32;            ఆపు;
            స్థితి HTP_OP_UNARY_SOFTPLUS:  compute_func = (శూన్య *) softplus_f32;            ఆపు;
            స్థితి HTP_OP_UNARY_TANH:      compute_func = (శూన్య *) tanh_f32;                ఆపు;
            స్థితి HTP_OP_UNARY_ABS:       compute_func = (శూన్య *) abs_f32;                 ఆపు;
            స్థితి HTP_OP_UNARY_LOG:       compute_func = (శూన్య *) log_f32;                 ఆపు;
            స్థితి HTP_OP_UNARY_RELU:      compute_func = (శూన్య *) relu_f32;                ఆపు;
            స్థితి HTP_OP_UNARY_అడుగు:      compute_func = (శూన్య *) అడుగు_f32;                ఆపు;
            స్థితి HTP_OP_L2_NORM:         compute_func = (శూన్య *) l2_norm_f32;             ఆపు;
            స్థితి HTP_OP_HADAMARD:        compute_func = (శూన్య *) hadamard_f32;            ఆపు;
            స్థితి HTP_OP_TRI:
                task_func    = unary_thread_tri_f32;
                compute_func = (శూన్య *) tri_f32;
                ఆపు;
            అప్రమేయ:                     ఆపు;
        }
    }

    ఒకవేళ (!task_func || !compute_func) {
        FARF(ERROR, "execute_op_unary: task function is NULL for op %d\n", octx->op);
        తిరిగి HTP_స్థితి_NO_SUPPORT;
    }

    uctx.compute = compute_func;
    work_వరుస_నిరంతరం(octx->ctx->work_వరుస, task_func, &uctx, n_దారాలు);

    తిరిగి err;
}

పూర్ణాంకం op_unary(నిర్మాణం htp_ops_context * octx) {
    ఎంపిక (octx->src[0]->type) {
        స్థితి HTP_TYPE_F32:
        స్థితి HTP_TYPE_F16:
            తిరిగి execute_op_unary(octx);

        అప్రమేయ:
            తిరిగి HTP_స్థితి_NO_SUPPORT;
    }
}
