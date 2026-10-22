#आशय clang diagnostic ignored "-Wunused-variable"
#आशय clang diagnostic ignored "-Wunused-function"
#आशय clang diagnostic ignored "-Wunused-but-set-variable"

#समावेश <HAP_farf.h>
#समावेश <HAP_perf.h>

#समावेश <math.h>
#समावेश <string.h>

#समावेश "dma-queue.h"
#समावेश "hex-fastdiv.h"
#समावेश "hvx-exp.h"
#समावेश "hvx-sigmoid.h"
#समावेश "hvx-utils.h"
#समावेश "unary-ops.h"

#घोषणा GGML_COMMON_DECL_C
#समावेश "ggml-common.h"
#समावेश "htp-ctx.h"
#समावेश "htp-ops.h"
#समावेश "htp-tensor.h"
#समावेश "htp-vtcm.h"
#समावेश "hex-profile.h"

काष्ठा htp_unary_context;

प्रकार व्योम (*unary_compute_fn_t)(स्थिर व्योम * restrict src,
                                   व्योम * restrict dst,
                                   uint32_t अंक_rows,
                                   स्थिर काष्ठा htp_unary_context * uctx);

प्रकार व्योम (*unary_rms_norm_mul_compute_fn_t)(स्थिर व्योम * restrict src,
                                                स्थिर व्योम * restrict weight,
                                                व्योम * restrict dst,
                                                uint32_t अंक_rows,
                                                स्थिर काष्ठा htp_unary_context * uctx);

प्रकार व्योम (*unary_tri_compute_fn_t)(स्थिर व्योम * restrict src,
                                       व्योम * restrict dst,
                                       uint32_t अंक_rows,
                                       uint32_t ir,
                                       स्थिर काष्ठा htp_unary_context * uctx);

प्रकार व्योम (*unary_tile_compute_fn_t)(व्योम * restrict dst,
                                        स्थिर व्योम * restrict src,
                                        uint32_t tw,
                                        स्थिर काष्ठा htp_unary_context * uctx);

प्रकार व्योम (*unary_tiled_tri_compute_fn_t)(स्थिर व्योम * restrict src,
                                             व्योम * restrict dst,
                                             uint32_t tile_elems,
                                             uint32_t col_start,
                                             uint32_t i01,
                                             uint32_t ne0,
                                             int32_t ttype);

काष्ठा htp_unary_context {
    काष्ठा htp_ops_context * octx;
    स्थिर काष्ठा htp_unary_kernel_params * kparams;

    व्योम *                    compute;

    dma_addr_t                तथ्य_src0;
    dma_addr_t                तथ्य_src1;            // weight/scale tensor for RMS_NORM_MUL
    dma_addr_t                तथ्य_dst;

    माप_प्रकार                    src0_तथ्य_row_आकार;   // actual data bytes per row
    माप_प्रकार                    src1_तथ्य_row_आकार;
    माप_प्रकार                    dst_तथ्य_row_आकार;    // actual data bytes per row

    माप_प्रकार                    src0_row_आकार_aligned;
    माप_प्रकार                    src1_row_आकार_aligned;
    माप_प्रकार                    dst_row_आकार_aligned;

    माप_प्रकार                    src0_vtcm_half_आकार;
    माप_प्रकार                    src1_vtcm_half_आकार;
    माप_प्रकार                    dst_vtcm_half_आकार;

    uint32_t                  block;
    uint32_t                  src0_nrows;
    uint32_t                  src0_nrows_per_thread;
    uint32_t                  row_start;
    uint32_t                  nc;
    uint32_t                  col_tile;             // tiled mode
    bool                      broadcast_weight;

    uint8_t *                 vtcm_src0;
    uint8_t *                 vtcm_src1;
    uint8_t *                 vtcm_dst;

    माप_प्रकार                    vtcm_src0_आकार_per_thread;
    माप_प्रकार                    vtcm_src1_आकार_per_thread;
    माप_प्रकार                    vtcm_dst_आकार_per_thread;
};

// Convert flat row index to DDR byte offset using the tensor's actual strides.
// ir = i1 + ne1*(i2 + ne2*i3)  =>  offset = i1*nb1 + i2*nb2 + i3*nb3
अटल अंतरभूत माप_प्रकार unary_row_offset(uint32_t ir,
                                      uint32_t ne1, uint32_t ne2,
                                      स्थिर काष्ठा fastdiv_मूल्यावली * div_ne1,
                                      स्थिर काष्ठा fastdiv_मूल्यावली * div_ne2,
                                      स्थिर काष्ठा fastdiv_मूल्यावली * div_ne12,
                                      माप_प्रकार nb1, माप_प्रकार nb2, माप_प्रकार nb3) {
    स्थिर uint32_t i1 = fastmodulo(ir, ne1, div_ne1);
    स्थिर uint32_t ir_div_ne1 = fastdiv(ir, div_ne1);
    स्थिर uint32_t i2 = fastmodulo(ir_div_ne1, ne2, div_ne2);
    स्थिर uint32_t i3 = fastdiv(ir, div_ne12);
    वापस i1 * nb1 + i2 * nb2 + i3 * nb3;
}

// Safe DMA block size from row `ir`: clamp to the tighter dim-1 slice
// boundary of src and dst so the nb1 stride stays valid for all rows.
अटल अंतरभूत uint32_t unary_block_आकार(uint32_t ir,
                                        uint32_t end_row,
                                        uint32_t block,
                                        bool src_contig,
                                        bool dst_contig,
                                        uint32_t ne1,
                                        स्थिर काष्ठा fastdiv_मूल्यावली * div_ne1) {
    uint32_t limit = MIN(block, end_row - ir);

    अगर (!src_contig || !dst_contig) {
        स्थिर uint32_t slice_end = (fastdiv(ir, div_ne1) + 1) * ne1;
        limit = MIN(limit, slice_end - ir);
    }

    वापस limit;
}

#घोषणा htp_unary_preamble            \
    स्थिर uint32_t ne00 = src->ne[0]; \
    स्थिर uint32_t ne01 = src->ne[1]; \
    स्थिर uint32_t ne02 = src->ne[2]; \
    स्थिर uint32_t ne03 = src->ne[3]; \
                                      \
    स्थिर uint32_t ne0 = dst->ne[0];  \
    स्थिर uint32_t ne1 = dst->ne[1];  \
    स्थिर uint32_t ne2 = dst->ne[2];  \
    स्थिर uint32_t ne3 = dst->ne[3];  \
                                      \
    स्थिर uint32_t nb00 = src->nb[0]; \
    स्थिर uint32_t nb01 = src->nb[1]; \
    स्थिर uint32_t nb02 = src->nb[2]; \
    स्थिर uint32_t nb03 = src->nb[3]; \
                                      \
    स्थिर uint32_t nb0 = dst->nb[0];  \
    स्थिर uint32_t nb1 = dst->nb[1];  \
    स्थिर uint32_t nb2 = dst->nb[2];  \
    स्थिर uint32_t nb3 = dst->nb[3];

#घोषणा htp_unary_op_preamble                                         \
    int32_t * op_params = uctx->octx->op_params;                      \
    स्थिर uint32_t ne0 = uctx->nc;                                    \
    स्थिर माप_प्रकार src0_row_आकार_aligned = uctx->src0_row_आकार_aligned; \
    स्थिर माप_प्रकार dst_row_आकार_aligned = uctx->dst_row_आकार_aligned;

अटल व्योम scale_f32(स्थिर व्योम * restrict src,
                      व्योम * restrict dst,
                      स्थिर uint32_t अंक_rows,
                      स्थिर काष्ठा htp_unary_context * uctx) {
    htp_unary_op_preamble;
    भग्न scale = 0.f;
    भग्न bias  = 0.f;
    स_नकल(&scale, &op_params[0], माप(भग्न));
    स_नकल(&bias,  &op_params[1], माप(भग्न));

    क्रम (uint32_t ir = 0; ir < अंक_rows; ir++) {
        स्थिर uint8_t * restrict src_local = (स्थिर uint8_t *)src + (ir * src0_row_आकार_aligned);
        uint8_t * restrict dst_local       = (uint8_t *)dst + (ir * dst_row_आकार_aligned);

        hvx_scale_offset_f32_aa((uint8_t *) dst_local, (स्थिर uint8_t *) src_local, ne0, scale, bias);
    }
}

अटल व्योम clamp_f32(स्थिर व्योम * restrict src,
                      व्योम * restrict dst,
                      स्थिर uint32_t अंक_rows,
                      स्थिर काष्ठा htp_unary_context * uctx) {
    htp_unary_op_preamble;
    भग्न min = 0.f;
    भग्न max = 0.f;
    स_नकल(&min, &op_params[0], माप(भग्न));
    स_नकल(&max, &op_params[1], माप(भग्न));

    क्रम (uint32_t ir = 0; ir < अंक_rows; ir++) {
        स्थिर uint8_t * restrict src_local = (स्थिर uint8_t *)src + (ir * src0_row_आकार_aligned);
        uint8_t * restrict dst_local       = (uint8_t *)dst + (ir * dst_row_आकार_aligned);

        hvx_clamp_scalar_f32(dst_local, src_local, min, max, ne0);
    }
}

अटल व्योम leaky_relu_f32(स्थिर व्योम * restrict src,
                           व्योम * restrict dst,
                           स्थिर uint32_t अंक_rows,
                           स्थिर काष्ठा htp_unary_context * uctx) {
    htp_unary_op_preamble;
    भग्न negative_slope = 0.f;
    स_नकल(&negative_slope, &op_params[0], माप(भग्न));

    क्रम (uint32_t ir = 0; ir < अंक_rows; ir++) {
        स्थिर uint8_t * restrict src_local = (स्थिर uint8_t *)src + (ir * src0_row_आकार_aligned);
        uint8_t * restrict dst_local       = (uint8_t *)dst + (ir * dst_row_आकार_aligned);

        hvx_leaky_relu_scalar_f32(dst_local, src_local, negative_slope, ne0);
    }
}

अटल व्योम rms_norm_f32(स्थिर व्योम * restrict src,
                         व्योम * restrict dst,
                         स्थिर uint32_t अंक_rows,
                         स्थिर काष्ठा htp_unary_context * uctx) {
    htp_unary_op_preamble;
    भग्न epsilon = 0.f;
    स_नकल(&epsilon, op_params, माप(भग्न));

    क्रम (uint32_t ir = 0; ir < अंक_rows; ir++) {
        स्थिर uint8_t * restrict src_local = (स्थिर uint8_t *)src + (ir * src0_row_आकार_aligned);
        uint8_t * restrict dst_local       = (uint8_t *)dst + (ir * dst_row_आकार_aligned);

        hvx_fast_rms_norm_f32((स्थिर uint8_t *) src_local, (uint8_t *) dst_local, ne0, epsilon);
    }
}

अटल व्योम rms_norm_mul_f32(स्थिर व्योम * restrict src,
                             स्थिर व्योम * restrict weight,
                             व्योम * restrict dst,
                             स्थिर uint32_t अंक_rows,
                             स्थिर काष्ठा htp_unary_context * uctx) {
    htp_unary_op_preamble;
    भग्न epsilon = 0.f;
    स_नकल(&epsilon, op_params, माप(भग्न));

    क्रम (uint32_t ir = 0; ir < अंक_rows; ir++) {
        स्थिर uint8_t * restrict src_local = (स्थिर uint8_t *)src + (ir * src0_row_आकार_aligned);
        स्थिर uint8_t * restrict w_local   = (स्थिर uint8_t *)weight + (uctx->broadcast_weight ? 0 : ir * uctx->src1_row_आकार_aligned);
        uint8_t * restrict dst_local       = (uint8_t *)dst + (ir * dst_row_आकार_aligned);

        hvx_fast_rms_norm_mul_f32(src_local, w_local, dst_local, ne0, epsilon);
    }
}

अटल व्योम norm_f32(स्थिर व्योम * restrict src,
                     व्योम * restrict dst,
                     स्थिर uint32_t अंक_rows,
                     स्थिर काष्ठा htp_unary_context * uctx) {
    htp_unary_op_preamble;
    भग्न epsilon = 0.f;
    स_नकल(&epsilon, op_params, माप(भग्न));

    क्रम (uint32_t ir = 0; ir < अंक_rows; ir++) {
        स्थिर uint8_t * restrict src_local = (स्थिर uint8_t *)src + (ir * src0_row_आकार_aligned);
        uint8_t * restrict dst_local       = (uint8_t *)dst + (ir * dst_row_आकार_aligned);

        hvx_fast_norm_f32((स्थिर uint8_t *) src_local, (uint8_t *) dst_local, ne0, epsilon);
    }
}

अटल व्योम sqr_f32(स्थिर व्योम * restrict src,
                    व्योम * restrict dst,
                    स्थिर uint32_t अंक_rows,
                    स्थिर काष्ठा htp_unary_context * uctx) {
    htp_unary_op_preamble;

    क्रम (uint32_t ir = 0; ir < अंक_rows; ir++) {
        स्थिर uint8_t * restrict src_local = (स्थिर uint8_t *)src + (ir * src0_row_आकार_aligned);
        uint8_t * restrict dst_local       = (uint8_t *)dst + (ir * dst_row_आकार_aligned);

        hvx_sqr_f32_aa((uint8_t *) dst_local, (स्थिर uint8_t *) src_local, ne0);
    }
}

अटल व्योम sqrt_f32(स्थिर व्योम * restrict src,
                     व्योम * restrict dst,
                     स्थिर uint32_t अंक_rows,
                     स्थिर काष्ठा htp_unary_context * uctx) {
    htp_unary_op_preamble;

    क्रम (uint32_t ir = 0; ir < अंक_rows; ir++) {
        स्थिर uint8_t * restrict src_local = (स्थिर uint8_t *)src + (ir * src0_row_आकार_aligned);
        uint8_t * restrict dst_local       = (uint8_t *)dst + (ir * dst_row_आकार_aligned);

        hvx_sqrt_f32_aa((uint8_t *) dst_local, (स्थिर uint8_t *) src_local, ne0);
    }
}

अटल व्योम scale_f16(स्थिर व्योम * restrict src,
                      व्योम * restrict dst,
                      स्थिर uint32_t अंक_rows,
                      स्थिर काष्ठा htp_unary_context * uctx) {
    htp_unary_op_preamble;
    भग्न scale = 0.f;
    भग्न bias  = 0.f;
    स_नकल(&scale, &op_params[0], माप(भग्न));
    स_नकल(&bias,  &op_params[1], माप(भग्न));

    क्रम (uint32_t ir = 0; ir < अंक_rows; ir++) {
        स्थिर uint8_t * restrict src_local = (स्थिर uint8_t *)src + (ir * src0_row_आकार_aligned);
        uint8_t * restrict dst_local       = (uint8_t *)dst + (ir * dst_row_आकार_aligned);

        hvx_scale_offset_f16_aa((uint8_t *) dst_local, (स्थिर uint8_t *) src_local, ne0, scale, bias);
    }
}

अटल व्योम clamp_f16(स्थिर व्योम * restrict src,
                      व्योम * restrict dst,
                      स्थिर uint32_t अंक_rows,
                      स्थिर काष्ठा htp_unary_context * uctx) {
    htp_unary_op_preamble;
    भग्न min = 0.f;
    भग्न max = 0.f;
    स_नकल(&min, &op_params[0], माप(भग्न));
    स_नकल(&max, &op_params[1], माप(भग्न));

    क्रम (uint32_t ir = 0; ir < अंक_rows; ir++) {
        स्थिर uint8_t * restrict src_local = (स्थिर uint8_t *)src + (ir * src0_row_आकार_aligned);
        uint8_t * restrict dst_local       = (uint8_t *)dst + (ir * dst_row_आकार_aligned);

        hvx_clamp_scalar_f16(dst_local, src_local, (_Float16) min, (_Float16) max, ne0);
    }
}

अटल व्योम rms_norm_f16(स्थिर व्योम * restrict src,
                         व्योम * restrict dst,
                         स्थिर uint32_t अंक_rows,
                         स्थिर काष्ठा htp_unary_context * uctx) {
    htp_unary_op_preamble;
    भग्न epsilon = 0.f;
    स_नकल(&epsilon, op_params, माप(भग्न));

    क्रम (uint32_t ir = 0; ir < अंक_rows; ir++) {
        स्थिर uint8_t * restrict src_local = (स्थिर uint8_t *)src + (ir * src0_row_आकार_aligned);
        uint8_t * restrict dst_local       = (uint8_t *)dst + (ir * dst_row_आकार_aligned);

        hvx_fast_rms_norm_f16((स्थिर uint8_t *) src_local, (uint8_t *) dst_local, ne0, epsilon);
    }
}

अटल व्योम norm_f16(स्थिर व्योम * restrict src,
                     व्योम * restrict dst,
                     स्थिर uint32_t अंक_rows,
                     स्थिर काष्ठा htp_unary_context * uctx) {
    htp_unary_op_preamble;
    भग्न epsilon = 0.f;
    स_नकल(&epsilon, op_params, माप(भग्न));

    क्रम (uint32_t ir = 0; ir < अंक_rows; ir++) {
        स्थिर uint8_t * restrict src_local = (स्थिर uint8_t *)src + (ir * src0_row_आकार_aligned);
        uint8_t * restrict dst_local       = (uint8_t *)dst + (ir * dst_row_आकार_aligned);

        hvx_fast_norm_f16((स्थिर uint8_t *) src_local, (uint8_t *) dst_local, ne0, epsilon);
    }
}

अटल व्योम sqr_f16(स्थिर व्योम * restrict src,
                    व्योम * restrict dst,
                    स्थिर uint32_t अंक_rows,
                    स्थिर काष्ठा htp_unary_context * uctx) {
    htp_unary_op_preamble;

    क्रम (uint32_t ir = 0; ir < अंक_rows; ir++) {
        स्थिर uint8_t * restrict src_local = (स्थिर uint8_t *)src + (ir * src0_row_आकार_aligned);
        uint8_t * restrict dst_local       = (uint8_t *)dst + (ir * dst_row_आकार_aligned);

        hvx_sqr_f16_aa((uint8_t *) dst_local, (स्थिर uint8_t *) src_local, ne0);
    }
}

अटल व्योम sqrt_f16(स्थिर व्योम * restrict src,
                     व्योम * restrict dst,
                     स्थिर uint32_t अंक_rows,
                     स्थिर काष्ठा htp_unary_context * uctx) {
    htp_unary_op_preamble;

    क्रम (uint32_t ir = 0; ir < अंक_rows; ir++) {
        स्थिर uint8_t * restrict src_local = (स्थिर uint8_t *)src + (ir * src0_row_आकार_aligned);
        uint8_t * restrict dst_local       = (uint8_t *)dst + (ir * dst_row_आकार_aligned);

        hvx_sqrt_f16_aa((uint8_t *) dst_local, (स्थिर uint8_t *) src_local, ne0);
    }
}

अटल व्योम abs_f16(स्थिर व्योम * restrict src,
                    व्योम * restrict dst,
                    स्थिर uint32_t अंक_rows,
                    स्थिर काष्ठा htp_unary_context * uctx) {
    htp_unary_op_preamble;

    क्रम (uint32_t ir = 0; ir < अंक_rows; ir++) {
        स्थिर uint8_t * restrict src_local = (स्थिर uint8_t *)src + (ir * src0_row_आकार_aligned);
        uint8_t * restrict dst_local       = (uint8_t *)dst + (ir * dst_row_आकार_aligned);

        hvx_abs_f16_aa((uint8_t *) dst_local, (स्थिर uint8_t *) src_local, ne0);
    }
}

अटल व्योम log_f16(स्थिर व्योम * restrict src,
                    व्योम * restrict dst,
                    स्थिर uint32_t अंक_rows,
                    स्थिर काष्ठा htp_unary_context * uctx) {
    htp_unary_op_preamble;

    क्रम (uint32_t ir = 0; ir < अंक_rows; ir++) {
        स्थिर uint8_t * restrict src_local = (स्थिर uint8_t *)src + (ir * src0_row_आकार_aligned);
        uint8_t * restrict dst_local       = (uint8_t *)dst + (ir * dst_row_आकार_aligned);

        hvx_log_f16_aa((uint8_t *) dst_local, (स्थिर uint8_t *) src_local, ne0);
    }
}

अटल व्योम step_f16(स्थिर व्योम * restrict src,
                     व्योम * restrict dst,
                     स्थिर uint32_t अंक_rows,
                     स्थिर काष्ठा htp_unary_context * uctx) {
    htp_unary_op_preamble;

    क्रम (uint32_t ir = 0; ir < अंक_rows; ir++) {
        स्थिर uint8_t * restrict src_local = (स्थिर uint8_t *)src + (ir * src0_row_आकार_aligned);
        uint8_t * restrict dst_local       = (uint8_t *)dst + (ir * dst_row_आकार_aligned);

        hvx_step_f16_aa((uint8_t *) dst_local, (स्थिर uint8_t *) src_local, ne0);
    }
}

अटल व्योम l2_norm_f16(स्थिर व्योम * restrict src,
                        व्योम * restrict dst,
                        स्थिर uint32_t अंक_rows,
                        स्थिर काष्ठा htp_unary_context * uctx) {
    htp_unary_op_preamble;
    भग्न epsilon = 0.f;
    स_नकल(&epsilon, op_params, माप(भग्न));

    क्रम (uint32_t ir = 0; ir < अंक_rows; ir++) {
        स्थिर uint8_t * restrict src_f = (स्थिर uint8_t *)src + (ir * src0_row_आकार_aligned);
        uint8_t * restrict dst_f       = (uint8_t *)dst + (ir * dst_row_आकार_aligned);

        hvx_fast_l2_norm_f16((स्थिर uint8_t *)src_f, (uint8_t *)dst_f, ne0, epsilon);
    }
}

अटल व्योम neg_f32(स्थिर व्योम * restrict src,
                    व्योम * restrict dst,
                    स्थिर uint32_t अंक_rows,
                    स्थिर काष्ठा htp_unary_context * uctx) {
    htp_unary_op_preamble;

    क्रम (uint32_t ir = 0; ir < अंक_rows; ir++) {
        स्थिर uint8_t * restrict src_local = (स्थिर uint8_t *)src + (ir * src0_row_आकार_aligned);
        uint8_t * restrict dst_local       = (uint8_t *)dst + (ir * dst_row_आकार_aligned);

        hvx_scale_f32_aa(dst_local, src_local, ne0, -1.0f);
    }
}

अटल व्योम exp_f32(स्थिर व्योम * restrict src,
                    व्योम * restrict dst,
                    स्थिर uint32_t अंक_rows,
                    स्थिर काष्ठा htp_unary_context * uctx) {
    htp_unary_op_preamble;

    क्रम (uint32_t ir = 0; ir < अंक_rows; ir++) {
        स्थिर uint8_t * restrict src_local = (स्थिर uint8_t *)src + (ir * src0_row_आकार_aligned);
        uint8_t * restrict dst_local       = (uint8_t *)dst + (ir * dst_row_आकार_aligned);

        hvx_exp_f32(dst_local, src_local, ne0, false);
    }
}

अटल व्योम sigmoid_f32(स्थिर व्योम * restrict src,
                        व्योम * restrict dst,
                        स्थिर uint32_t अंक_rows,
                        स्थिर काष्ठा htp_unary_context * uctx) {
    htp_unary_op_preamble;

    क्रम (uint32_t ir = 0; ir < अंक_rows; ir++) {
        स्थिर uint8_t * restrict src_local = (स्थिर uint8_t *)src + (ir * src0_row_आकार_aligned);
        uint8_t * restrict dst_local       = (uint8_t *)dst + (ir * dst_row_आकार_aligned);

        hvx_sigmoid_f32_aa(dst_local, src_local, ne0);
    }
}

// silu(x) = x * sigmoid(x)
अटल व्योम silu_f32(स्थिर व्योम * restrict src,
                     व्योम * restrict dst,
                     स्थिर uint32_t अंक_rows,
                     स्थिर काष्ठा htp_unary_context * uctx) {
    htp_unary_op_preamble;

    क्रम (uint32_t ir = 0; ir < अंक_rows; ir++) {
        स्थिर uint8_t * restrict src_local = (स्थिर uint8_t *)src + (ir * src0_row_आकार_aligned);
        uint8_t * restrict dst_local       = (uint8_t *)dst + (ir * dst_row_आकार_aligned);

        hvx_sigmoid_f32_aa(dst_local, src_local, ne0);
        hvx_mul_f32_aaa(dst_local, src_local, dst_local, ne0);
    }
}

// GELU uses the tanh approximation.
अटल __attribute__((noinline)) HVX_Vector hvx_vec_gelu_f32(HVX_Vector x) {
    स्थिर HVX_Vector half = hvx_vec_splat_f32(0.5f);
    स्थिर HVX_Vector one  = hvx_vec_splat_f32(1.0f);

    HVX_Vector inner = hvx_vec_mul_f32_f32(x, x);
    inner = hvx_vec_mul_f32_f32(inner, hvx_vec_splat_f32(0.044715f));
    inner = hvx_vec_add_f32_f32(inner, one);
    inner = hvx_vec_mul_f32_f32(inner, x);
    inner = hvx_vec_mul_f32_f32(inner, hvx_vec_splat_f32(0.7978845608028654f));

    वापस hvx_vec_mul_f32_f32(hvx_vec_mul_f32_f32(half, x),
                               hvx_vec_add_f32_f32(one, hvx_vec_tanh_f32(inner)));
}

अटल अंतरभूत व्योम hvx_gelu_f32_aa(uint8_t * restrict dst, स्थिर uint8_t * restrict src, uint32_t n) {
    निश्चित((अचिन्हित दीर्घ) dst % 128 == 0);
    निश्चित((अचिन्हित दीर्घ) src % 128 == 0);

    HVX_Vector * restrict vdst = (HVX_Vector *) dst;
    HVX_Vector * restrict vsrc = (HVX_Vector *) src;

    स्थिर uint32_t nvec = n / VLEN_FP32;
    स्थिर uint32_t nloe = n % VLEN_FP32;

    uint32_t i = 0;
    _Pragma("unroll(4)")
    क्रम (; i < nvec; i++) {
        vdst[i] = hvx_vec_gelu_f32(vsrc[i]);
    }
    अगर (nloe) {
        hvx_vec_store_a(&vdst[i], nloe * माप(भग्न), hvx_vec_gelu_f32(vsrc[i]));
    }
}

// GELU_QUICK uses x * sigmoid(1.702 * x).
अटल __attribute__((noinline)) HVX_Vector hvx_vec_gelu_quick_f32(HVX_Vector x) {
    स्थिर HVX_Vector one     = hvx_vec_splat_f32(1.0f);
    स्थिर HVX_Vector max_exp = hvx_vec_splat_f32(87.0f);
    स्थिर HVX_Vector min_exp = hvx_vec_splat_f32(-87.0f);
    स्थिर HVX_Vector scaled  = hvx_vec_mul_f32_f32(x, hvx_vec_splat_f32(1.702f));
    स्थिर HVX_Vector sigmoid = hvx_vec_fast_sigmoid_f32_guard(scaled, one, max_exp, min_exp);

    वापस hvx_vec_mul_f32_f32(x, sigmoid);
}

अटल अंतरभूत व्योम hvx_gelu_quick_f32_aa(uint8_t * restrict dst, स्थिर uint8_t * restrict src, uint32_t n) {
    निश्चित((अचिन्हित दीर्घ) dst % 128 == 0);
    निश्चित((अचिन्हित दीर्घ) src % 128 == 0);

    HVX_Vector * restrict vdst = (HVX_Vector *) dst;
    HVX_Vector * restrict vsrc = (HVX_Vector *) src;

    स्थिर uint32_t nvec = n / VLEN_FP32;
    स्थिर uint32_t nloe = n % VLEN_FP32;

    uint32_t i = 0;
    _Pragma("unroll(4)")
    क्रम (; i < nvec; i++) {
        vdst[i] = hvx_vec_gelu_quick_f32(vsrc[i]);
    }
    अगर (nloe) {
        hvx_vec_store_a(&vdst[i], nloe * माप(भग्न), hvx_vec_gelu_quick_f32(vsrc[i]));
    }
}

अटल व्योम gelu_f32(स्थिर व्योम * restrict src,
                     व्योम * restrict dst,
                     स्थिर uint32_t अंक_rows,
                     स्थिर काष्ठा htp_unary_context * uctx) {
    htp_unary_op_preamble;

    क्रम (uint32_t ir = 0; ir < अंक_rows; ir++) {
        स्थिर uint8_t * restrict src_local = (स्थिर uint8_t *)src + (ir * src0_row_आकार_aligned);
        uint8_t * restrict dst_local       = (uint8_t *)dst + (ir * dst_row_आकार_aligned);

        hvx_gelu_f32_aa(dst_local, src_local, ne0);
    }
}

अटल व्योम gelu_quick_f32(स्थिर व्योम * restrict src,
                           व्योम * restrict dst,
                           स्थिर uint32_t अंक_rows,
                           स्थिर काष्ठा htp_unary_context * uctx) {
    htp_unary_op_preamble;

    क्रम (uint32_t ir = 0; ir < अंक_rows; ir++) {
        स्थिर uint8_t * restrict src_local = (स्थिर uint8_t *)src + (ir * src0_row_आकार_aligned);
        uint8_t * restrict dst_local       = (uint8_t *)dst + (ir * dst_row_आकार_aligned);

        hvx_gelu_quick_f32_aa(dst_local, src_local, ne0);
    }
}

अटल व्योम gelu_erf_f32(स्थिर व्योम * restrict src,
                         व्योम * restrict dst,
                         स्थिर uint32_t अंक_rows,
                         स्थिर काष्ठा htp_unary_context * uctx) {
    htp_unary_op_preamble;

    क्रम (uint32_t ir = 0; ir < अंक_rows; ir++) {
        स्थिर uint8_t * restrict src_local = (स्थिर uint8_t *) src + (ir * src0_row_आकार_aligned);
        uint8_t * restrict dst_local       = (uint8_t *) dst + (ir * dst_row_आकार_aligned);

        hvx_gelu_erf_f32_aa(dst_local, src_local, ne0);
    }
}

अटल व्योम tri_f32(स्थिर व्योम * restrict src,
                    व्योम * restrict dst,
                    स्थिर uint32_t अंक_rows,
                    स्थिर uint32_t ir,
                    स्थिर काष्ठा htp_unary_context * uctx) {
    htp_unary_op_preamble;
    स्थिर int32_t ttype = op_params[0];
    स्थिर HVX_Vector zero = hvx_vec_splat_f32(0.0f);
    स्थिर uint32_t nvec  = ne0 / VLEN_FP32;
    स्थिर uint32_t nloe  = ne0 % VLEN_FP32;

    स्थिर uint32_t ne01 = uctx->octx->src[0]->ne[1];

    क्रम (uint32_t b = 0; b < अंक_rows; b++) {
        स्थिर uint32_t abs_row = ir + b;
        स्थिर uint32_t i01     = abs_row % ne01;

        स्थिर HVX_Vector * restrict v_src = (स्थिर HVX_Vector *) ((स्थिर uint8_t *) src + b * src0_row_आकार_aligned);
        HVX_Vector * restrict v_dst       = (HVX_Vector *) ((uint8_t *) dst + b * dst_row_आकार_aligned);

        uint32_t boundary;
        पूर्णांक      keep_बायाँ;
        चयन (ttype) {
            हाल 0: boundary = i01;     keep_बायाँ = 0; अवरोध;  // keep col >= row
            हाल 1: boundary = i01 + 1; keep_बायाँ = 0; अवरोध;  // keep col > row
            हाल 2: boundary = i01 + 1; keep_बायाँ = 1; अवरोध;  // keep col <= row
            हाल 3: boundary = i01;     keep_बायाँ = 1; अवरोध;  // keep col < row
            शेष: boundary = 0; keep_बायाँ = 0; अवरोध;
        }
        अगर (boundary > ne0) boundary = ne0;

        // Full HVX vectors - each starts at a 128-byte aligned offset
        क्रम (uint32_t i = 0; i < nvec; i++) {
            स्थिर uint32_t vec_start = i * VLEN_FP32;
            स्थिर uint32_t vec_end   = vec_start + VLEN_FP32;
            अगर (keep_बायाँ) {
                अगर (vec_end <= boundary) {
                    v_dst[i] = v_src[i];
                } अन्यथा अगर (vec_start >= boundary) {
                    v_dst[i] = zero;
                } अन्यथा {
                    HVX_VectorPred mask = Q6_Q_vsetq_R((boundary - vec_start) * माप(भग्न));
                    v_dst[i]            = Q6_V_vmux_QVV(mask, v_src[i], zero);
                }
            } अन्यथा {
                अगर (vec_end <= boundary) {
                    v_dst[i] = zero;
                } अन्यथा अगर (vec_start >= boundary) {
                    v_dst[i] = v_src[i];
                } अन्यथा {
                    HVX_VectorPred mask = Q6_Q_vsetq_R((boundary - vec_start) * माप(भग्न));
                    v_dst[i]            = Q6_V_vmux_QVV(mask, zero, v_src[i]);
                }
            }
        }

        // Tail elements (row_elems not a multiple of VLEN_FP32)
        अगर (nloe > 0) {
            स्थिर uint32_t abs_start = nvec * VLEN_FP32;
            स्थिर uint32_t abs_end   = abs_start + nloe;
            HVX_Vector     पुच्छ_val;
            अगर (keep_बायाँ) {
                अगर (abs_end <= boundary) {
                    पुच्छ_val = v_src[nvec];
                } अन्यथा अगर (abs_start >= boundary) {
                    पुच्छ_val = zero;
                } अन्यथा {
                    HVX_VectorPred mask = Q6_Q_vsetq_R((boundary - abs_start) * माप(भग्न));
                    पुच्छ_val            = Q6_V_vmux_QVV(mask, v_src[nvec], zero);
                }
            } अन्यथा {
                अगर (abs_end <= boundary) {
                    पुच्छ_val = zero;
                } अन्यथा अगर (abs_start >= boundary) {
                    पुच्छ_val = v_src[nvec];
                } अन्यथा {
                    HVX_VectorPred mask = Q6_Q_vsetq_R((boundary - abs_start) * माप(भग्न));
                    पुच्छ_val            = Q6_V_vmux_QVV(mask, zero, v_src[nvec]);
                }
            }
            hvx_vec_store_a(&v_dst[nvec], nloe * माप(भग्न), पुच्छ_val);
        }
    }
}

// softplus(x) = log(1 + exp(x))
// Match CPU reference: return x for x > 20 to avoid overflow.
अटल अंतरभूत व्योम hvx_softplus_f32_aa(uint8_t * restrict dst, स्थिर uint8_t * restrict src, uint32_t n) {
    स्थिर HVX_Vector * restrict v_src = (स्थिर HVX_Vector *) src;
    HVX_Vector * restrict v_dst       = (HVX_Vector *) dst;

    स्थिर HVX_Vector twenty = hvx_vec_splat_f32(20.0f);
    स्थिर HVX_Vector one    = hvx_vec_splat_f32(1.0f);

    स्थिर uint32_t nvec = n / VLEN_FP32;
    स्थिर uint32_t nloe = n % VLEN_FP32;

    #आशय unroll(4)
    क्रम (uint32_t i = 0; i < nvec; i++) {
        HVX_Vector x              = v_src[i];
        HVX_VectorPred pred_gt20  = Q6_Q_vcmp_gt_VsfVsf(x, twenty);
        HVX_Vector expx           = hvx_vec_exp_f32(x);
        HVX_Vector logv           = hvx_vec_log_f32(hvx_vec_add_f32_f32(one, expx));
        v_dst[i]                  = Q6_V_vmux_QVV(pred_gt20, x, logv);
    }

    अगर (nloe > 0) {
        HVX_Vector x             = v_src[nvec];
        HVX_VectorPred pred_gt20 = Q6_Q_vcmp_gt_VsfVsf(x, twenty);
        HVX_Vector expx          = hvx_vec_exp_f32(x);
        HVX_Vector logv          = hvx_vec_log_f32(hvx_vec_add_f32_f32(one, expx));
        HVX_Vector sp            = Q6_V_vmux_QVV(pred_gt20, x, logv);
        hvx_vec_store_a((व्योम *) &v_dst[nvec], nloe * माप(भग्न), sp);
    }
}

अटल व्योम softplus_f32(स्थिर व्योम * restrict src,
                         व्योम * restrict dst,
                         स्थिर uint32_t अंक_rows,
                         स्थिर काष्ठा htp_unary_context * uctx) {
    htp_unary_op_preamble;

    क्रम (uint32_t ir = 0; ir < अंक_rows; ir++) {
        स्थिर uint8_t * restrict src_local = (स्थिर uint8_t *)src + (ir * src0_row_आकार_aligned);
        uint8_t * restrict dst_local       = (uint8_t *)dst + (ir * dst_row_आकार_aligned);

        hvx_softplus_f32_aa(dst_local, src_local, ne0);
    }
}

अटल व्योम l2_norm_f32(स्थिर व्योम * restrict src,
                        व्योम * restrict dst,
                        स्थिर uint32_t अंक_rows,
                        स्थिर काष्ठा htp_unary_context * uctx) {
    htp_unary_op_preamble;
    भग्न epsilon = 0.f;
    स_नकल(&epsilon, op_params, माप(भग्न));

    क्रम (uint32_t ir = 0; ir < अंक_rows; ir++) {
        स्थिर uint8_t * restrict src_f = (स्थिर uint8_t *)src + (ir * src0_row_आकार_aligned);
        uint8_t * restrict dst_f       = (uint8_t *)dst + (ir * dst_row_आकार_aligned);

        hvx_fast_l2_norm_f32((स्थिर uint8_t *)src_f, (uint8_t *)dst_f, ne0, epsilon);
    }
}

अटल अंतरभूत HVX_VectorPair hvx_fwht_stage_pair(HVX_Vector v0, HVX_Vector v1, int32_t b) {
    HVX_VectorPair p = Q6_W_vshuff_VVR(v1, v0, b);
    HVX_Vector s = Q6_Vqf32_vadd_Vqf32Vqf32(Q6_V_lo_W(p), Q6_V_hi_W(p));
    HVX_Vector d = Q6_Vqf32_vsub_Vqf32Vqf32(Q6_V_lo_W(p), Q6_V_hi_W(p));
    वापस Q6_W_vshuff_VVR(d, s, b);
}

अटल अंतरभूत HVX_VectorPair hvx_fwht32_pair(HVX_Vector v0_sf, HVX_Vector v1_sf, HVX_Vector v_scale) {
    HVX_Vector v0 = Q6_Vqf32_vmpy_VsfVsf(v0_sf, v_scale);
    HVX_Vector v1 = Q6_Vqf32_vmpy_VsfVsf(v1_sf, v_scale);
    HVX_VectorPair p;
    p = hvx_fwht_stage_pair(v0, v1, 4);
    p = hvx_fwht_stage_pair(Q6_V_lo_W(p), Q6_V_hi_W(p), 8);
    p = hvx_fwht_stage_pair(Q6_V_lo_W(p), Q6_V_hi_W(p), 16);
    p = hvx_fwht_stage_pair(Q6_V_lo_W(p), Q6_V_hi_W(p), 32);
    p = hvx_fwht_stage_pair(Q6_V_lo_W(p), Q6_V_hi_W(p), 64);
    वापस p;
}

अटल अंतरभूत व्योम hvx_fwht_store32(HVX_Vector * restrict v_dst, HVX_VectorPair p) {
    v_dst[0] = Q6_Vsf_equals_Vqf32(Q6_V_lo_W(p));
}

अटल अंतरभूत व्योम hvx_fwht_store64(HVX_Vector * restrict v_dst, HVX_VectorPair p) {
    HVX_Vector u = Q6_V_lo_W(p);
    HVX_Vector v = Q6_V_hi_W(p);
    v_dst[0] = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vqf32(u, v));
    v_dst[1] = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vsub_Vqf32Vqf32(u, v));
}

अटल अंतरभूत व्योम hvx_fwht_पुच्छ_f32(HVX_Vector * restrict v_dst, uint32_t nvec) {
    uint32_t len_vec = 2;
    क्रम (; len_vec * 2 < nvec; len_vec <<= 2) {
        स्थिर uint32_t step = len_vec * 4;
        क्रम (uint32_t i = 0; i < nvec; i += step) {
            क्रम (uint32_t j = 0; j < len_vec; j++) {
                HVX_Vector a0 = v_dst[i + j];
                HVX_Vector a1 = v_dst[i + j + len_vec];
                HVX_Vector a2 = v_dst[i + j + 2 * len_vec];
                HVX_Vector a3 = v_dst[i + j + 3 * len_vec];

                HVX_Vector t0 = Q6_Vqf32_vadd_VsfVsf(a0, a1);
                HVX_Vector t1 = Q6_Vqf32_vsub_VsfVsf(a0, a1);
                HVX_Vector t2 = Q6_Vqf32_vadd_VsfVsf(a2, a3);
                HVX_Vector t3 = Q6_Vqf32_vsub_VsfVsf(a2, a3);

                v_dst[i + j]               = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vqf32(t0, t2));
                v_dst[i + j + len_vec]     = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vqf32(t1, t3));
                v_dst[i + j + 2 * len_vec] = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vsub_Vqf32Vqf32(t0, t2));
                v_dst[i + j + 3 * len_vec] = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vsub_Vqf32Vqf32(t1, t3));
            }
        }
    }

    अगर (len_vec < nvec) {
        क्रम (uint32_t i = 0; i < nvec; i += 2 * len_vec) {
            क्रम (uint32_t j = 0; j < len_vec; j++) {
                HVX_Vector u = v_dst[i + j];
                HVX_Vector v = v_dst[i + len_vec + j];
                v_dst[i + j]           = hvx_vec_add_f32_f32(u, v);
                v_dst[i + len_vec + j] = hvx_vec_sub_f32_f32(u, v);
            }
        }
    }
}

अटल अंतरभूत व्योम hvx_fwht_row_f32(स्थिर uint8_t * restrict src, uint8_t * restrict dst, uint32_t n, HVX_Vector v_scale) {
    स्थिर HVX_Vector * restrict v_src = (स्थिर HVX_Vector *) src;
    HVX_Vector * restrict v_dst       = (HVX_Vector *) dst;
    स्थिर uint32_t nvec               = n / VLEN_FP32;

    अगर (nvec == 1) {
        HVX_VectorPair p = hvx_fwht32_pair(v_src[0], v_src[0], v_scale);
        hvx_fwht_store32(v_dst, p);
        वापस;
    }

    क्रम (uint32_t i = 0; i < nvec; i += 2) {
        HVX_VectorPair p = hvx_fwht32_pair(v_src[i], v_src[i + 1], v_scale);
        hvx_fwht_store64(&v_dst[i], p);
    }

    hvx_fwht_पुच्छ_f32(v_dst, nvec);
}

अटल अंतरभूत व्योम hvx_fwht_row_f16(स्थिर uint8_t * restrict src, uint8_t * restrict dst, uint32_t n, HVX_Vector v_scale) {
    स्थिर HVX_Vector * restrict v_src = (स्थिर HVX_Vector *) src;
    HVX_Vector * restrict v_dst       = (HVX_Vector *) dst;
    स्थिर uint32_t nvec               = n / VLEN_FP32;

    अगर (nvec == 1) {
        HVX_Vector v0 = Q6_V_lo_W(hvx_vec_f16_to_f32(v_src[0]));
        HVX_VectorPair p = hvx_fwht32_pair(v0, v0, v_scale);
        hvx_fwht_store32(v_dst, p);
        वापस;
    }

    स्थिर uint32_t nvec_f16 = n / VLEN_FP16;
    क्रम (uint32_t i = 0; i < nvec_f16; i++) {
        HVX_VectorPair f32p = hvx_vec_f16_to_f32(v_src[i]);
        HVX_VectorPair p    = hvx_fwht32_pair(Q6_V_lo_W(f32p), Q6_V_hi_W(f32p), v_scale);
        hvx_fwht_store64(&v_dst[2 * i], p);
    }

    hvx_fwht_पुच्छ_f32(v_dst, nvec);
}

अटल व्योम hadamard_f32(स्थिर व्योम * restrict src,
                         व्योम * restrict dst,
                         स्थिर uint32_t अंक_rows,
                         स्थिर काष्ठा htp_unary_context * uctx) {
    htp_unary_op_preamble;
    स्थिर HVX_Vector v_scale = hvx_vec_splat_f32(1.0f / sqrtf((भग्न) ne0));

    क्रम (uint32_t ir = 0; ir < अंक_rows; ir++) {
        स्थिर uint8_t * restrict src_row = (स्थिर uint8_t *) src + (ir * src0_row_आकार_aligned);
        uint8_t * restrict dst_row       = (uint8_t *) dst + (ir * dst_row_आकार_aligned);

        hvx_fwht_row_f32(src_row, dst_row, ne0, v_scale);
    }
}

अटल व्योम hadamard_f16(स्थिर व्योम * restrict src,
                         व्योम * restrict dst,
                         स्थिर uint32_t अंक_rows,
                         स्थिर काष्ठा htp_unary_context * uctx) {
    htp_unary_op_preamble;
    स्थिर HVX_Vector v_scale = hvx_vec_splat_f32(1.0f / sqrtf((भग्न) ne0));

    क्रम (uint32_t ir = 0; ir < अंक_rows; ir++) {
        स्थिर uint8_t * restrict src_row = (स्थिर uint8_t *) src + (ir * src0_row_आकार_aligned);
        uint8_t * restrict dst_row       = (uint8_t *) dst + (ir * dst_row_आकार_aligned);

        hvx_fwht_row_f16(src_row, dst_row, ne0, v_scale);
    }
}

अटल व्योम tanh_f32(स्थिर व्योम * restrict src,
                     व्योम * restrict dst,
                     स्थिर uint32_t अंक_rows,
                     स्थिर काष्ठा htp_unary_context * uctx) {
    htp_unary_op_preamble;

    क्रम (uint32_t ir = 0; ir < अंक_rows; ir++) {
        स्थिर uint8_t * restrict src_local = (स्थिर uint8_t *)src + (ir * src0_row_आकार_aligned);
        uint8_t * restrict dst_local       = (uint8_t *)dst + (ir * dst_row_आकार_aligned);

        hvx_tanh_f32_aa(dst_local, src_local, ne0);
    }
}

अटल व्योम abs_f32(स्थिर व्योम * restrict src,
                    व्योम * restrict dst,
                    स्थिर uint32_t अंक_rows,
                    स्थिर काष्ठा htp_unary_context * uctx) {
    htp_unary_op_preamble;

    क्रम (uint32_t ir = 0; ir < अंक_rows; ir++) {
        स्थिर uint8_t * restrict src_local = (स्थिर uint8_t *)src + (ir * src0_row_आकार_aligned);
        uint8_t * restrict dst_local       = (uint8_t *)dst + (ir * dst_row_आकार_aligned);

        hvx_abs_f32_aa(dst_local, src_local, ne0);
    }
}

अटल व्योम relu_f32(स्थिर व्योम * restrict src,
                     व्योम * restrict dst,
                     स्थिर uint32_t अंक_rows,
                     स्थिर काष्ठा htp_unary_context * uctx) {
    htp_unary_op_preamble;

    क्रम (uint32_t ir = 0; ir < अंक_rows; ir++) {
        स्थिर uint8_t * restrict src_local = (स्थिर uint8_t *)src + (ir * src0_row_आकार_aligned);
        uint8_t * restrict dst_local       = (uint8_t *)dst + (ir * dst_row_आकार_aligned);

        hvx_max_scalar_f32(dst_local, src_local, 0.0f, ne0);
    }
}

अटल व्योम step_f32(स्थिर व्योम * restrict src,
                     व्योम * restrict dst,
                     स्थिर uint32_t अंक_rows,
                     स्थिर काष्ठा htp_unary_context * uctx) {
    htp_unary_op_preamble;

    क्रम (uint32_t ir = 0; ir < अंक_rows; ir++) {
        स्थिर uint8_t * restrict src_local = (स्थिर uint8_t *)src + (ir * src0_row_आकार_aligned);
        uint8_t * restrict dst_local       = (uint8_t *)dst + (ir * dst_row_आकार_aligned);

        hvx_step_f32_aa(dst_local, src_local, ne0);
    }
}

अटल व्योम log_f32(स्थिर व्योम * restrict src,
                    व्योम * restrict dst,
                    स्थिर uint32_t अंक_rows,
                    स्थिर काष्ठा htp_unary_context * uctx) {
    htp_unary_op_preamble;

    क्रम (uint32_t ir = 0; ir < अंक_rows; ir++) {
        स्थिर uint8_t * restrict src_local = (स्थिर uint8_t *)src + (ir * src0_row_आकार_aligned);
        uint8_t * restrict dst_local       = (uint8_t *)dst + (ir * dst_row_आकार_aligned);

        hvx_log_f32_aa(dst_local, src_local, ne0);
    }
}

#// Pointwise unary ops on one column tile in VTCM.
अटल व्योम tile_scale_f32(व्योम * restrict dst, स्थिर व्योम * restrict src, uint32_t tw, स्थिर काष्ठा htp_unary_context * uctx) {
    भग्न scale = 0.f;
    भग्न bias  = 0.f;
    स_नकल(&scale, &uctx->octx->op_params[0], माप(भग्न));
    स_नकल(&bias,  &uctx->octx->op_params[1], माप(भग्न));
    hvx_scale_offset_f32_aa((uint8_t *) dst, (स्थिर uint8_t *) src, tw, scale, bias);
}

अटल व्योम tile_clamp_f32(व्योम * restrict dst, स्थिर व्योम * restrict src, uint32_t tw, स्थिर काष्ठा htp_unary_context * uctx) {
    भग्न min = 0.f;
    भग्न max = 0.f;
    स_नकल(&min, &uctx->octx->op_params[0], माप(भग्न));
    स_नकल(&max, &uctx->octx->op_params[1], माप(भग्न));
    hvx_clamp_scalar_f32((uint8_t *) dst, (स्थिर uint8_t *) src, min, max, tw);
}

अटल व्योम tile_leaky_relu_f32(व्योम * restrict dst, स्थिर व्योम * restrict src, uint32_t tw, स्थिर काष्ठा htp_unary_context * uctx) {
    भग्न negative_slope = 0.f;
    स_नकल(&negative_slope, &uctx->octx->op_params[0], माप(भग्न));
    hvx_leaky_relu_scalar_f32((uint8_t *) dst, (स्थिर uint8_t *) src, negative_slope, tw);
}

अटल व्योम tile_sqr_f32(व्योम * restrict dst, स्थिर व्योम * restrict src, uint32_t tw, स्थिर काष्ठा htp_unary_context * uctx) {
    (व्योम) uctx;
    hvx_sqr_f32_aa((uint8_t *) dst, (स्थिर uint8_t *) src, tw);
}

अटल व्योम tile_sqrt_f32(व्योम * restrict dst, स्थिर व्योम * restrict src, uint32_t tw, स्थिर काष्ठा htp_unary_context * uctx) {
    (व्योम) uctx;
    hvx_sqrt_f32_aa((uint8_t *) dst, (स्थिर uint8_t *) src, tw);
}

अटल व्योम tile_neg_f32(व्योम * restrict dst, स्थिर व्योम * restrict src, uint32_t tw, स्थिर काष्ठा htp_unary_context * uctx) {
    (व्योम) uctx;
    hvx_scale_f32_aa((uint8_t *) dst, (स्थिर uint8_t *) src, tw, -1.0f);
}

अटल व्योम tile_exp_f32(व्योम * restrict dst, स्थिर व्योम * restrict src, uint32_t tw, स्थिर काष्ठा htp_unary_context * uctx) {
    (व्योम) uctx;
    hvx_exp_f32((uint8_t *) dst, (स्थिर uint8_t *) src, tw, false);
}

अटल व्योम tile_sigmoid_f32(व्योम * restrict dst, स्थिर व्योम * restrict src, uint32_t tw, स्थिर काष्ठा htp_unary_context * uctx) {
    (व्योम) uctx;
    hvx_sigmoid_f32_aa((uint8_t *) dst, (स्थिर uint8_t *) src, tw);
}

अटल व्योम tile_silu_f32(व्योम * restrict dst, स्थिर व्योम * restrict src, uint32_t tw, स्थिर काष्ठा htp_unary_context * uctx) {
    (व्योम) uctx;
    hvx_sigmoid_f32_aa((uint8_t *) dst, (स्थिर uint8_t *) src, tw);
    hvx_mul_f32_aaa((uint8_t *) dst, (स्थिर uint8_t *) src, (uint8_t *) dst, tw);
}

अटल व्योम tile_gelu_f32(व्योम * restrict dst, स्थिर व्योम * restrict src, uint32_t tw, स्थिर काष्ठा htp_unary_context * uctx) {
    (व्योम) uctx;
    hvx_gelu_f32_aa((uint8_t *) dst, (स्थिर uint8_t *) src, tw);
}

अटल व्योम tile_gelu_quick_f32(व्योम * restrict dst, स्थिर व्योम * restrict src, uint32_t tw, स्थिर काष्ठा htp_unary_context * uctx) {
    (व्योम) uctx;
    hvx_gelu_quick_f32_aa((uint8_t *) dst, (स्थिर uint8_t *) src, tw);
}

अटल व्योम tile_gelu_erf_f32(व्योम * restrict dst, स्थिर व्योम * restrict src, uint32_t tw, स्थिर काष्ठा htp_unary_context * uctx) {
    (व्योम) uctx;
    hvx_gelu_erf_f32_aa((uint8_t *) dst, (स्थिर uint8_t *) src, tw);
}

अटल व्योम tile_softplus_f32(व्योम * restrict dst, स्थिर व्योम * restrict src, uint32_t tw, स्थिर काष्ठा htp_unary_context * uctx) {
    (व्योम) uctx;
    hvx_softplus_f32_aa((uint8_t *) dst, (स्थिर uint8_t *) src, tw);
}

अटल व्योम tile_tanh_f32(व्योम * restrict dst, स्थिर व्योम * restrict src, uint32_t tw, स्थिर काष्ठा htp_unary_context * uctx) {
    (व्योम) uctx;
    hvx_tanh_f32_aa((uint8_t *) dst, (स्थिर uint8_t *) src, tw);
}

अटल व्योम tile_abs_f32(व्योम * restrict dst, स्थिर व्योम * restrict src, uint32_t tw, स्थिर काष्ठा htp_unary_context * uctx) {
    (व्योम) uctx;
    hvx_abs_f32_aa((uint8_t *) dst, (स्थिर uint8_t *) src, tw);
}

अटल व्योम tile_log_f32(व्योम * restrict dst, स्थिर व्योम * restrict src, uint32_t tw, स्थिर काष्ठा htp_unary_context * uctx) {
    (व्योम) uctx;
    hvx_log_f32_aa((uint8_t *) dst, (स्थिर uint8_t *) src, tw);
}

अटल व्योम tile_relu_f32(व्योम * restrict dst, स्थिर व्योम * restrict src, uint32_t tw, स्थिर काष्ठा htp_unary_context * uctx) {
    (व्योम) uctx;
    hvx_max_scalar_f32((uint8_t *) dst, (स्थिर uint8_t *) src, 0.0f, tw);
}

अटल व्योम tile_step_f32(व्योम * restrict dst, स्थिर व्योम * restrict src, uint32_t tw, स्थिर काष्ठा htp_unary_context * uctx) {
    (व्योम) uctx;
    hvx_step_f32_aa((uint8_t *) dst, (स्थिर uint8_t *) src, tw);
}

अटल व्योम tri_apply_tile_f32(स्थिर व्योम * restrict src, व्योम * restrict dst,
                               uint32_t tile_elems, uint32_t col_start, uint32_t i01,
                               uint32_t ne0, int32_t ttype) {
    स्थिर HVX_Vector * restrict v_src = (स्थिर HVX_Vector *) src;
    HVX_Vector * restrict v_dst       = (HVX_Vector *) dst;
    स्थिर HVX_Vector zero = hvx_vec_splat_f32(0.0f);

    uint32_t boundary;
    पूर्णांक      keep_बायाँ;
    चयन (ttype) {
        हाल 0: boundary = i01;     keep_बायाँ = 0; अवरोध;
        हाल 1: boundary = i01 + 1; keep_बायाँ = 0; अवरोध;
        हाल 2: boundary = i01 + 1; keep_बायाँ = 1; अवरोध;
        हाल 3: boundary = i01;     keep_बायाँ = 1; अवरोध;
        शेष: boundary = 0; keep_बायाँ = 0; अवरोध;
    }
    अगर (boundary > ne0) boundary = ne0;

    स्थिर uint32_t nvec = tile_elems / VLEN_FP32;
    स्थिर uint32_t nloe = tile_elems % VLEN_FP32;

    क्रम (uint32_t i = 0; i < nvec; i++) {
        स्थिर uint32_t abs_start = col_start + i * VLEN_FP32;
        स्थिर uint32_t abs_end   = abs_start + VLEN_FP32;
        अगर (keep_बायाँ) {
            अगर (abs_end <= boundary) {
                v_dst[i] = v_src[i];
            } अन्यथा अगर (abs_start >= boundary) {
                v_dst[i] = zero;
            } अन्यथा {
                HVX_VectorPred mask = Q6_Q_vsetq_R((boundary - abs_start) * माप(भग्न));
                v_dst[i]            = Q6_V_vmux_QVV(mask, v_src[i], zero);
            }
        } अन्यथा {
            अगर (abs_end <= boundary) {
                v_dst[i] = zero;
            } अन्यथा अगर (abs_start >= boundary) {
                v_dst[i] = v_src[i];
            } अन्यथा {
                HVX_VectorPred mask = Q6_Q_vsetq_R((boundary - abs_start) * माप(भग्न));
                v_dst[i]            = Q6_V_vmux_QVV(mask, zero, v_src[i]);
            }
        }
    }

    अगर (nloe > 0) {
        स्थिर uint32_t abs_start = col_start + nvec * VLEN_FP32;
        स्थिर uint32_t abs_end   = abs_start + nloe;
        HVX_Vector     पुच्छ_val;
        अगर (keep_बायाँ) {
            अगर (abs_end <= boundary) {
                पुच्छ_val = v_src[nvec];
            } अन्यथा अगर (abs_start >= boundary) {
                पुच्छ_val = zero;
            } अन्यथा {
                HVX_VectorPred mask = Q6_Q_vsetq_R((boundary - abs_start) * माप(भग्न));
                पुच्छ_val            = Q6_V_vmux_QVV(mask, v_src[nvec], zero);
            }
        } अन्यथा {
            अगर (abs_end <= boundary) {
                पुच्छ_val = zero;
            } अन्यथा अगर (abs_start >= boundary) {
                पुच्छ_val = v_src[nvec];
            } अन्यथा {
                HVX_VectorPred mask = Q6_Q_vsetq_R((boundary - abs_start) * माप(भग्न));
                पुच्छ_val            = Q6_V_vmux_QVV(mask, zero, v_src[nvec]);
            }
        }
        hvx_vec_store_a(&v_dst[nvec], nloe * माप(भग्न), पुच्छ_val);
    }
}

// 1. Standard row-block unary task (F32 and F16).
अटल व्योम unary_thread_row_block(अचिन्हित पूर्णांक nth, अचिन्हित पूर्णांक ith, व्योम * तथ्य) {
    (व्योम) nth;
    स्थिर काष्ठा htp_unary_context * uctx = (स्थिर काष्ठा htp_unary_context *) तथ्य;
    काष्ठा htp_ops_context * octx = uctx->octx;
    स्थिर काष्ठा htp_tensor * src = octx->src[0];
    स्थिर काष्ठा htp_tensor * dst = octx->dst;
    काष्ठा htp_thread_trace * tr = &octx->ctx->trace[ith];

    htp_unary_preamble;

    स्थिर uint32_t src0_nrows_per_thread = uctx->src0_nrows_per_thread;
    स्थिर माप_प्रकार src0_तथ्य_row_आकार = uctx->src0_तथ्य_row_आकार;
    स्थिर माप_प्रकार dst_तथ्य_row_आकार  = uctx->dst_तथ्य_row_आकार;
    स्थिर माप_प्रकार src0_row_आकार_aligned = uctx->src0_row_आकार_aligned;
    स्थिर माप_प्रकार dst_row_आकार_aligned  = uctx->dst_row_आकार_aligned;

    स्थिर uint32_t src0_nrows = uctx->src0_nrows;
    स्थिर uint32_t src0_start_row = uctx->row_start + src0_nrows_per_thread * ith;
    स्थिर uint32_t src0_end_row   = MIN(src0_start_row + src0_nrows_per_thread, uctx->row_start + src0_nrows);

    अगर (src0_start_row >= src0_end_row) {
        वापस;
    }

    स्थिर dma_addr_t तथ्य_src = uctx->तथ्य_src0;
    स्थिर dma_addr_t तथ्य_dst = uctx->तथ्य_dst;

    uint8_t * src0_vtcm_तथ्य = uctx->vtcm_src0 + (ith * uctx->vtcm_src0_आकार_per_thread);
    uint8_t * dst_vtcm_तथ्य  = uctx->vtcm_dst + (ith * uctx->vtcm_dst_आकार_per_thread);

    स्थिर माप_प्रकार src0_vtcm_half_आकार = uctx->src0_vtcm_half_आकार;
    स्थिर माप_प्रकार dst_vtcm_half_आकार  = uctx->dst_vtcm_half_आकार;

    स्थिर bool src0_contig = (nb02 == (माप_प्रकार)ne01 * nb01) &&
                             (nb03 == (माप_प्रकार)ne02 * nb02);
    स्थिर bool dst_contig  = (nb2  == (माप_प्रकार)ne1  * nb1)  &&
                             (nb3  == (माप_प्रकार)ne2  * nb2);

    स्थिर काष्ठा fastdiv_मूल्यावली * div_ne01  = &uctx->kparams->div_ne01;
    स्थिर काष्ठा fastdiv_मूल्यावली * div_ne02  = &uctx->kparams->div_ne02;
    स्थिर काष्ठा fastdiv_मूल्यावली * div_ne012 = &uctx->kparams->div_ne012;

    स्थिर uint32_t src0_max_block = src0_contig ? uctx->block : MIN((uint32_t)uctx->block, ne01);
    स्थिर uint32_t dst_max_block  = dst_contig  ? uctx->block : MIN((uint32_t)uctx->block, ne1);
    स्थिर uint32_t BLOCK = MIN(src0_max_block, dst_max_block);
    अगर (BLOCK == 0) {
        FARF(ERROR, "unary-row-block : current VTCM reservation %zu is too small, needed at least %zu\n",
             uctx->vtcm_src0_आकार_per_thread, src0_row_आकार_aligned);
        वापस;
    }

    dma_queue * dma_q = octx->ctx->dma[ith];

    क्रम (uint32_t ir = src0_start_row, vtcm_idx = 0; ir < src0_end_row && vtcm_idx < 2; vtcm_idx++) {
        स्थिर uint32_t block_आकार_2 = unary_block_आकार(ir, src0_end_row, BLOCK, src0_contig, dst_contig,
                                                     ne01, div_ne01);

        dma_queue_धकेलो(dma_q,
            dma_make_तथ्य(तथ्य_dst, dst_vtcm_तथ्य + (vtcm_idx * dst_vtcm_half_आकार)),
            nb1, dst_row_आकार_aligned, dst_तथ्य_row_आकार, 0);

        स्थिर माप_प्रकार src0_off = src0_contig ? (ir * nb01) :
            unary_row_offset(ir, ne01, ne02, div_ne01, div_ne02, div_ne012, nb01, nb02, nb03);
        dma_queue_धकेलो(dma_q,
            dma_make_तथ्य(src0_vtcm_तथ्य + (vtcm_idx * src0_vtcm_half_आकार), तथ्य_src + src0_off),
            src0_row_आकार_aligned, nb01, src0_तथ्य_row_आकार, block_आकार_2);

        ir += block_आकार_2;
    }

    unary_compute_fn_t compute = (unary_compute_fn_t) uctx->compute;

    क्रम (uint32_t ir = src0_start_row; ir < src0_end_row; ) {
        स्थिर uint32_t block_आकार_2 = unary_block_आकार(ir, src0_end_row, BLOCK, src0_contig, dst_contig,
                                                     ne01, div_ne01);

        व्योम * dst_vtcm  = (व्योम *) (uintptr_t) dma_queue_निकालो(dma_q).src;
        व्योम * src0_vtcm = (व्योम *) (uintptr_t) dma_queue_निकालो(dma_q).dst;

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, ir);
        compute(src0_vtcm, dst_vtcm, block_आकार_2, uctx);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ir);

        स्थिर माप_प्रकार dst_off = dst_contig ? (ir * nb1) :
            unary_row_offset(ir, ne1, ne2, div_ne01, div_ne02, div_ne012, nb1, nb2, nb3);
        dma_queue_धकेलो(dma_q,
            dma_make_तथ्य(तथ्य_dst + dst_off, dst_vtcm),
            nb1, dst_row_आकार_aligned, dst_तथ्य_row_आकार, block_आकार_2);

        स्थिर uint32_t अग्र_ir = ir + block_आकार_2;
        अगर (अग्र_ir < src0_end_row) {
            स्थिर uint32_t अग्र_block_आकार = unary_block_आकार(अग्र_ir, src0_end_row, BLOCK, src0_contig,
                                                              dst_contig, ne01, div_ne01);
            स्थिर uint32_t pref_ir = अग्र_ir + अग्र_block_आकार;
            अगर (pref_ir < src0_end_row) {
                स्थिर uint32_t pref_block_आकार = unary_block_आकार(pref_ir, src0_end_row, BLOCK, src0_contig,
                                                                  dst_contig, ne01, div_ne01);
                स्थिर माप_प्रकार src0_pref_off = src0_contig ? (pref_ir * nb01) :
                    unary_row_offset(pref_ir, ne01, ne02, div_ne01, div_ne02, div_ne012, nb01, nb02, nb03);
                dma_queue_धकेलो(dma_q,
                    dma_make_तथ्य(src0_vtcm, तथ्य_src + src0_pref_off),
                    src0_row_आकार_aligned, nb01, src0_तथ्य_row_आकार, pref_block_आकार);
            }
        }
        ir += block_आकार_2;
    }

    dma_queue_flush(dma_q);
}

// 2. RMS_NORM_MUL row-block task with weight buffer.
अटल व्योम unary_thread_rms_norm_mul_f32(अचिन्हित पूर्णांक nth, अचिन्हित पूर्णांक ith, व्योम * तथ्य) {
    (व्योम) nth;
    स्थिर काष्ठा htp_unary_context * uctx = (स्थिर काष्ठा htp_unary_context *) तथ्य;
    काष्ठा htp_ops_context * octx = uctx->octx;
    स्थिर काष्ठा htp_tensor * src = octx->src[0];
    स्थिर काष्ठा htp_tensor * dst = octx->dst;
    काष्ठा htp_thread_trace * tr = &octx->ctx->trace[ith];

    htp_unary_preamble;

    स्थिर uint32_t src0_nrows_per_thread = uctx->src0_nrows_per_thread;
    स्थिर माप_प्रकार src0_तथ्य_row_आकार = uctx->src0_तथ्य_row_आकार;
    स्थिर माप_प्रकार dst_तथ्य_row_आकार  = uctx->dst_तथ्य_row_आकार;
    स्थिर माप_प्रकार src0_row_आकार_aligned = uctx->src0_row_आकार_aligned;
    स्थिर माप_प्रकार dst_row_आकार_aligned  = uctx->dst_row_आकार_aligned;

    स्थिर uint32_t src0_nrows = uctx->src0_nrows;
    स्थिर uint32_t src0_start_row = uctx->row_start + src0_nrows_per_thread * ith;
    स्थिर uint32_t src0_end_row   = MIN(src0_start_row + src0_nrows_per_thread, uctx->row_start + src0_nrows);

    अगर (src0_start_row >= src0_end_row) {
        वापस;
    }

    स्थिर dma_addr_t तथ्य_src  = uctx->तथ्य_src0;
    स्थिर dma_addr_t तथ्य_src1 = uctx->तथ्य_src1;
    स्थिर dma_addr_t तथ्य_dst  = uctx->तथ्य_dst;

    स्थिर काष्ठा htp_tensor * src1 = octx->src[1];
    स्थिर uint32_t nb11 = src1->nb[1];
    स्थिर uint32_t nb12 = src1->nb[2];
    स्थिर uint32_t nb13 = src1->nb[3];
    स्थिर uint32_t nb11_bc = (src1->ne[1] > 1) ? nb11 : 0;
    स्थिर uint32_t nb12_bc = (src1->ne[2] > 1) ? nb12 : 0;
    स्थिर uint32_t nb13_bc = (src1->ne[3] > 1) ? nb13 : 0;
    स्थिर bool src1_contig = ((nb12 == (माप_प्रकार)ne01 * nb11) && (nb13 == (माप_प्रकार)ne02 * nb12));

    uint8_t * src0_vtcm_तथ्य = uctx->vtcm_src0 + (ith * uctx->vtcm_src0_आकार_per_thread);
    uint8_t * src1_vtcm_तथ्य = uctx->vtcm_src1 ? (uctx->vtcm_src1 + (ith * uctx->vtcm_src1_आकार_per_thread)) : शून्य;
    uint8_t * dst_vtcm_तथ्य  = uctx->vtcm_dst + (ith * uctx->vtcm_dst_आकार_per_thread);

    स्थिर माप_प्रकार src0_vtcm_half_आकार = uctx->src0_vtcm_half_आकार;
    स्थिर माप_प्रकार src1_vtcm_half_आकार = uctx->src1_vtcm_half_आकार;
    स्थिर माप_प्रकार dst_vtcm_half_आकार  = uctx->dst_vtcm_half_आकार;

    स्थिर bool src0_contig = (nb02 == (माप_प्रकार)ne01 * nb01) &&
                             (nb03 == (माप_प्रकार)ne02 * nb02);
    स्थिर bool dst_contig  = (nb2  == (माप_प्रकार)ne1  * nb1)  &&
                             (nb3  == (माप_प्रकार)ne2  * nb2);

    स्थिर काष्ठा fastdiv_मूल्यावली * div_ne01  = &uctx->kparams->div_ne01;
    स्थिर काष्ठा fastdiv_मूल्यावली * div_ne02  = &uctx->kparams->div_ne02;
    स्थिर काष्ठा fastdiv_मूल्यावली * div_ne012 = &uctx->kparams->div_ne012;

    स्थिर bool src1_needs_row_clip = !uctx->broadcast_weight && !src1_contig;
    स्थिर bool block_src0_contig = src0_contig && !src1_needs_row_clip;
    स्थिर bool block_dst_contig  = dst_contig  && !src1_needs_row_clip;

    स्थिर uint32_t src0_max_block = block_src0_contig ? uctx->block : MIN((uint32_t)uctx->block, ne01);
    स्थिर uint32_t dst_max_block  = block_dst_contig  ? uctx->block : MIN((uint32_t)uctx->block, ne1);
    स्थिर uint32_t BLOCK = MIN(src0_max_block, dst_max_block);
    अगर (BLOCK == 0) {
        FARF(ERROR, "unary-rms-norm-mul : current VTCM reservation %zu is too small, needed at least %zu\n",
             uctx->vtcm_src0_आकार_per_thread, src0_row_आकार_aligned);
        वापस;
    }

    dma_queue * dma_q = octx->ctx->dma[ith];

    अगर (uctx->broadcast_weight) {
        dma_queue_धकेलो(dma_q, dma_make_तथ्य(src1_vtcm_तथ्य, तथ्य_src1),
                       uctx->src1_row_आकार_aligned, 0, uctx->src1_तथ्य_row_आकार, 1);
        dma_queue_flush(dma_q);
    }

    क्रम (uint32_t ir = src0_start_row, vtcm_idx = 0; ir < src0_end_row && vtcm_idx < 2; vtcm_idx++) {
        स्थिर uint32_t block_आकार_2 = unary_block_आकार(ir, src0_end_row, BLOCK, block_src0_contig, block_dst_contig,
                                                     ne01, div_ne01);

        dma_queue_धकेलो(dma_q,
            dma_make_तथ्य(तथ्य_dst, dst_vtcm_तथ्य + (vtcm_idx * dst_vtcm_half_आकार)),
            nb1, dst_row_आकार_aligned, dst_तथ्य_row_आकार, 0);

        स्थिर माप_प्रकार src0_off = src0_contig ? (ir * nb01) :
            unary_row_offset(ir, ne01, ne02, div_ne01, div_ne02, div_ne012, nb01, nb02, nb03);
        dma_queue_धकेलो(dma_q,
            dma_make_तथ्य(src0_vtcm_तथ्य + (vtcm_idx * src0_vtcm_half_आकार), तथ्य_src + src0_off),
            src0_row_आकार_aligned, nb01, src0_तथ्य_row_आकार, block_आकार_2);

        अगर (!uctx->broadcast_weight) {
            स्थिर माप_प्रकार src1_off = src1_contig ? (ir * nb11) :
                unary_row_offset(ir, ne01, ne02, div_ne01, div_ne02, div_ne012, nb11_bc, nb12_bc, nb13_bc);
            dma_queue_धकेलो(dma_q,
                dma_make_तथ्य(src1_vtcm_तथ्य + (vtcm_idx * src1_vtcm_half_आकार), तथ्य_src1 + src1_off),
                uctx->src1_row_आकार_aligned, nb11, uctx->src1_तथ्य_row_आकार, block_आकार_2);
        }

        ir += block_आकार_2;
    }

    unary_rms_norm_mul_compute_fn_t compute = (unary_rms_norm_mul_compute_fn_t) uctx->compute;

    क्रम (uint32_t ir = src0_start_row; ir < src0_end_row; ) {
        स्थिर uint32_t block_आकार_2 = unary_block_आकार(ir, src0_end_row, BLOCK, block_src0_contig, block_dst_contig,
                                                     ne01, div_ne01);

        व्योम * dst_vtcm  = (व्योम *) (uintptr_t) dma_queue_निकालो(dma_q).src;
        व्योम * src0_vtcm = (व्योम *) (uintptr_t) dma_queue_निकालो(dma_q).dst;
        व्योम * src1_vtcm = शून्य;
        अगर (!uctx->broadcast_weight) {
            src1_vtcm = (व्योम *) (uintptr_t) dma_queue_निकालो(dma_q).dst;
        }

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, ir);
        स्थिर व्योम * w = uctx->broadcast_weight ? (स्थिर व्योम *) src1_vtcm_तथ्य : src1_vtcm;
        compute(src0_vtcm, w, dst_vtcm, block_आकार_2, uctx);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ir);

        स्थिर माप_प्रकार dst_off = dst_contig ? (ir * nb1) :
            unary_row_offset(ir, ne1, ne2, div_ne01, div_ne02, div_ne012, nb1, nb2, nb3);
        dma_queue_धकेलो(dma_q,
            dma_make_तथ्य(तथ्य_dst + dst_off, dst_vtcm),
            nb1, dst_row_आकार_aligned, dst_तथ्य_row_आकार, block_आकार_2);

        स्थिर uint32_t अग्र_ir = ir + block_आकार_2;
        अगर (अग्र_ir < src0_end_row) {
            स्थिर uint32_t अग्र_block_आकार = unary_block_आकार(अग्र_ir, src0_end_row, BLOCK, block_src0_contig,
                                                              block_dst_contig, ne01, div_ne01);
            स्थिर uint32_t pref_ir = अग्र_ir + अग्र_block_आकार;
            अगर (pref_ir < src0_end_row) {
                स्थिर uint32_t pref_block_आकार = unary_block_आकार(pref_ir, src0_end_row, BLOCK, block_src0_contig,
                                                                  block_dst_contig, ne01, div_ne01);
                स्थिर माप_प्रकार src0_pref_off = src0_contig ? (pref_ir * nb01) :
                    unary_row_offset(pref_ir, ne01, ne02, div_ne01, div_ne02, div_ne012, nb01, nb02, nb03);
                dma_queue_धकेलो(dma_q,
                    dma_make_तथ्य(src0_vtcm, तथ्य_src + src0_pref_off),
                    src0_row_आकार_aligned, nb01, src0_तथ्य_row_आकार, pref_block_आकार);

                अगर (!uctx->broadcast_weight) {
                    स्थिर माप_प्रकार src1_pref_off = src1_contig ? (pref_ir * nb11) :
                        unary_row_offset(pref_ir, ne01, ne02, div_ne01, div_ne02, div_ne012, nb11_bc, nb12_bc,
                                          nb13_bc);
                    dma_queue_धकेलो(dma_q,
                        dma_make_तथ्य(src1_vtcm, तथ्य_src1 + src1_pref_off),
                        uctx->src1_row_आकार_aligned, nb11, uctx->src1_तथ्य_row_आकार, pref_block_आकार);
                }
            }
        }
        ir += block_आकार_2;
    }

    dma_queue_flush(dma_q);
}

// 3. TRI row-block task with row index ir.
अटल व्योम unary_thread_tri_f32(अचिन्हित पूर्णांक nth, अचिन्हित पूर्णांक ith, व्योम * तथ्य) {
    (व्योम) nth;
    स्थिर काष्ठा htp_unary_context * uctx = (स्थिर काष्ठा htp_unary_context *) तथ्य;
    काष्ठा htp_ops_context * octx = uctx->octx;
    स्थिर काष्ठा htp_tensor * src = octx->src[0];
    स्थिर काष्ठा htp_tensor * dst = octx->dst;
    काष्ठा htp_thread_trace * tr = &octx->ctx->trace[ith];

    htp_unary_preamble;

    स्थिर uint32_t src0_nrows_per_thread = uctx->src0_nrows_per_thread;
    स्थिर माप_प्रकार src0_तथ्य_row_आकार = uctx->src0_तथ्य_row_आकार;
    स्थिर माप_प्रकार dst_तथ्य_row_आकार  = uctx->dst_तथ्य_row_आकार;
    स्थिर माप_प्रकार src0_row_आकार_aligned = uctx->src0_row_आकार_aligned;
    स्थिर माप_प्रकार dst_row_आकार_aligned  = uctx->dst_row_आकार_aligned;

    स्थिर uint32_t src0_nrows = uctx->src0_nrows;
    स्थिर uint32_t src0_start_row = uctx->row_start + src0_nrows_per_thread * ith;
    स्थिर uint32_t src0_end_row   = MIN(src0_start_row + src0_nrows_per_thread, uctx->row_start + src0_nrows);

    अगर (src0_start_row >= src0_end_row) {
        वापस;
    }

    स्थिर dma_addr_t तथ्य_src = uctx->तथ्य_src0;
    स्थिर dma_addr_t तथ्य_dst = uctx->तथ्य_dst;

    uint8_t * src0_vtcm_तथ्य = uctx->vtcm_src0 + (ith * uctx->vtcm_src0_आकार_per_thread);
    uint8_t * dst_vtcm_तथ्य  = uctx->vtcm_dst + (ith * uctx->vtcm_dst_आकार_per_thread);

    स्थिर माप_प्रकार src0_vtcm_half_आकार = uctx->src0_vtcm_half_आकार;
    स्थिर माप_प्रकार dst_vtcm_half_आकार  = uctx->dst_vtcm_half_आकार;

    स्थिर bool src0_contig = (nb02 == (माप_प्रकार)ne01 * nb01) &&
                             (nb03 == (माप_प्रकार)ne02 * nb02);
    स्थिर bool dst_contig  = (nb2  == (माप_प्रकार)ne1  * nb1)  &&
                             (nb3  == (माप_प्रकार)ne2  * nb2);

    स्थिर काष्ठा fastdiv_मूल्यावली * div_ne01  = &uctx->kparams->div_ne01;
    स्थिर काष्ठा fastdiv_मूल्यावली * div_ne02  = &uctx->kparams->div_ne02;
    स्थिर काष्ठा fastdiv_मूल्यावली * div_ne012 = &uctx->kparams->div_ne012;

    स्थिर uint32_t src0_max_block = src0_contig ? uctx->block : MIN((uint32_t)uctx->block, ne01);
    स्थिर uint32_t dst_max_block  = dst_contig  ? uctx->block : MIN((uint32_t)uctx->block, ne1);
    स्थिर uint32_t BLOCK = MIN(src0_max_block, dst_max_block);
    अगर (BLOCK == 0) {
        FARF(ERROR, "unary-tri : current VTCM reservation %zu is too small, needed at least %zu\n",
             uctx->vtcm_src0_आकार_per_thread, src0_row_आकार_aligned);
        वापस;
    }

    dma_queue * dma_q = octx->ctx->dma[ith];

    क्रम (uint32_t ir = src0_start_row, vtcm_idx = 0; ir < src0_end_row && vtcm_idx < 2; vtcm_idx++) {
        स्थिर uint32_t block_आकार_2 = unary_block_आकार(ir, src0_end_row, BLOCK, src0_contig, dst_contig,
                                                     ne01, div_ne01);

        dma_queue_धकेलो(dma_q,
            dma_make_तथ्य(तथ्य_dst, dst_vtcm_तथ्य + (vtcm_idx * dst_vtcm_half_आकार)),
            nb1, dst_row_आकार_aligned, dst_तथ्य_row_आकार, 0);

        स्थिर माप_प्रकार src0_off = src0_contig ? (ir * nb01) :
            unary_row_offset(ir, ne01, ne02, div_ne01, div_ne02, div_ne012, nb01, nb02, nb03);
        dma_queue_धकेलो(dma_q,
            dma_make_तथ्य(src0_vtcm_तथ्य + (vtcm_idx * src0_vtcm_half_आकार), तथ्य_src + src0_off),
            src0_row_आकार_aligned, nb01, src0_तथ्य_row_आकार, block_आकार_2);

        ir += block_आकार_2;
    }

    unary_tri_compute_fn_t compute = (unary_tri_compute_fn_t) uctx->compute;

    क्रम (uint32_t ir = src0_start_row; ir < src0_end_row; ) {
        स्थिर uint32_t block_आकार_2 = unary_block_आकार(ir, src0_end_row, BLOCK, src0_contig, dst_contig,
                                                     ne01, div_ne01);

        व्योम * dst_vtcm  = (व्योम *) (uintptr_t) dma_queue_निकालो(dma_q).src;
        व्योम * src0_vtcm = (व्योम *) (uintptr_t) dma_queue_निकालो(dma_q).dst;

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, ir);
        compute(src0_vtcm, dst_vtcm, block_आकार_2, ir, uctx);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ir);

        स्थिर माप_प्रकार dst_off = dst_contig ? (ir * nb1) :
            unary_row_offset(ir, ne1, ne2, div_ne01, div_ne02, div_ne012, nb1, nb2, nb3);
        dma_queue_धकेलो(dma_q,
            dma_make_तथ्य(तथ्य_dst + dst_off, dst_vtcm),
            nb1, dst_row_आकार_aligned, dst_तथ्य_row_आकार, block_आकार_2);

        स्थिर uint32_t अग्र_ir = ir + block_आकार_2;
        अगर (अग्र_ir < src0_end_row) {
            स्थिर uint32_t अग्र_block_आकार = unary_block_आकार(अग्र_ir, src0_end_row, BLOCK, src0_contig,
                                                              dst_contig, ne01, div_ne01);
            स्थिर uint32_t pref_ir = अग्र_ir + अग्र_block_आकार;
            अगर (pref_ir < src0_end_row) {
                स्थिर uint32_t pref_block_आकार = unary_block_आकार(pref_ir, src0_end_row, BLOCK, src0_contig,
                                                                  dst_contig, ne01, div_ne01);
                स्थिर माप_प्रकार src0_pref_off = src0_contig ? (pref_ir * nb01) :
                    unary_row_offset(pref_ir, ne01, ne02, div_ne01, div_ne02, div_ne012, nb01, nb02, nb03);
                dma_queue_धकेलो(dma_q,
                    dma_make_तथ्य(src0_vtcm, तथ्य_src + src0_pref_off),
                    src0_row_आकार_aligned, nb01, src0_तथ्य_row_आकार, pref_block_आकार);
            }
        }
        ir += block_आकार_2;
    }

    dma_queue_flush(dma_q);
}

// 4. Pointwise tiled unary task.
अटल व्योम unary_thread_tiled(अचिन्हित पूर्णांक nth, अचिन्हित पूर्णांक ith, व्योम * तथ्य) {
    (व्योम) nth;
    स्थिर काष्ठा htp_unary_context * uctx = (स्थिर काष्ठा htp_unary_context *) तथ्य;
    काष्ठा htp_ops_context * octx = uctx->octx;
    स्थिर काष्ठा htp_tensor * src = octx->src[0];
    स्थिर काष्ठा htp_tensor * dst = octx->dst;
    काष्ठा htp_thread_trace * tr = &octx->ctx->trace[ith];

    htp_unary_preamble;

    स्थिर uint32_t src0_nrows_per_thread = uctx->src0_nrows_per_thread;
    स्थिर uint32_t col_tile              = uctx->col_tile;

    स्थिर uint32_t src0_nrows     = uctx->src0_nrows;
    स्थिर uint32_t src0_start_row = uctx->row_start + src0_nrows_per_thread * ith;
    स्थिर uint32_t src0_end_row   = MIN(src0_start_row + src0_nrows_per_thread, uctx->row_start + src0_nrows);

    अगर (src0_start_row >= src0_end_row) {
        वापस;
    }

    स्थिर dma_addr_t तथ्य_src = uctx->तथ्य_src0;
    स्थिर dma_addr_t तथ्य_dst = uctx->तथ्य_dst;

    uint8_t * src0_vtcm_तथ्य = uctx->vtcm_src0 + (ith * uctx->vtcm_src0_आकार_per_thread);
    uint8_t * dst_vtcm_तथ्य  = uctx->vtcm_dst + (ith * uctx->vtcm_dst_आकार_per_thread);

    स्थिर माप_प्रकार src0_half = uctx->src0_vtcm_half_आकार;
    स्थिर माप_प्रकार dst_half  = uctx->dst_vtcm_half_आकार;

    dma_queue * dma_q = octx->ctx->dma[ith];

    स्थिर काष्ठा fastdiv_मूल्यावली * div_ne01  = &uctx->kparams->div_ne01;
    स्थिर काष्ठा fastdiv_मूल्यावली * div_ne02  = &uctx->kparams->div_ne02;
    स्थिर काष्ठा fastdiv_मूल्यावली * div_ne012 = &uctx->kparams->div_ne012;
    स्थिर काष्ठा fastdiv_मूल्यावली * div_tpr   = &uctx->kparams->div_tpr;

    स्थिर uint32_t tiles_per_row = (ne0 + col_tile - 1) / col_tile;

    स्थिर bool src0_contig = (nb02 == (माप_प्रकार)ne01 * nb01) &&
                             (nb03 == (माप_प्रकार)ne02 * nb02);
    स्थिर bool dst_contig  = (nb2  == (माप_प्रकार)ne1  * nb1)  &&
                             (nb3  == (माप_प्रकार)ne2  * nb2);

    स्थिर uint32_t योग_tiles = (src0_end_row - src0_start_row) * tiles_per_row;

    क्रम (uint32_t t = 0, vtcm_idx = 0; t < योग_tiles && vtcm_idx < 2; t++, vtcm_idx++) {
        स्थिर uint32_t row  = src0_start_row + t / tiles_per_row;
        स्थिर uint32_t col  = (t % tiles_per_row) * col_tile;
        स्थिर uint32_t tw   = MIN(col_tile, ne0 - col);
        स्थिर माप_प्रकार   tb   = (माप_प्रकार) tw * माप(भग्न);
        स्थिर माप_प्रकार   soff = (src0_contig ? (row * nb01) :
                               unary_row_offset(row, ne01, ne02, div_ne01, div_ne02, div_ne012, nb01, nb02, nb03)) +
                               (माप_प्रकार) col * माप(भग्न);

        dma_queue_धकेलो(dma_q, dma_make_तथ्य(तथ्य_dst, dst_vtcm_तथ्य + (vtcm_idx * dst_half)), 0, 0, 0, 0);
        dma_queue_धकेलो(dma_q, dma_make_तथ्य(src0_vtcm_तथ्य + (vtcm_idx * src0_half), तथ्य_src + soff), tb, tb, tb, 1);
    }

    unary_tile_compute_fn_t compute = (unary_tile_compute_fn_t) uctx->compute;

    uint32_t row = src0_start_row;
    uint32_t col = 0;
    uint32_t tile_in_row = 0;

    uint32_t prow = src0_start_row + fastdiv(2, div_tpr);
    uint32_t pcol = fastmodulo(2, tiles_per_row, div_tpr) * col_tile;
    uint32_t ptile_in_row = fastmodulo(2, tiles_per_row, div_tpr);

    क्रम (uint32_t t = 0; t < योग_tiles; t++) {
        व्योम * dst_vtcm = (व्योम *) (uintptr_t) dma_queue_निकालो(dma_q).src;
        व्योम * src_vtcm = (व्योम *) (uintptr_t) dma_queue_निकालो(dma_q).dst;

        स्थिर uint32_t tw = MIN(col_tile, ne0 - col);

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, t);
        compute(dst_vtcm, src_vtcm, tw, uctx);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, t);

        स्थिर माप_प्रकार doff = (dst_contig ? (row * nb1) :
                             unary_row_offset(row, ne1, ne2, div_ne01, div_ne02, div_ne012, nb1, nb2, nb3)) +
                             (माप_प्रकार) col * माप(भग्न);
        स्थिर माप_प्रकार tb   = (माप_प्रकार) tw * माप(भग्न);
        dma_queue_धकेलो(dma_q, dma_make_तथ्य(तथ्य_dst + doff, dst_vtcm), tb, tb, tb, 1);

        स्थिर uint32_t pt = t + 2;
        अगर (pt < योग_tiles) {
            स्थिर uint32_t ptw  = MIN(col_tile, ne0 - pcol);
            स्थिर माप_प्रकार   ptb  = (माप_प्रकार) ptw * माप(भग्न);
            स्थिर माप_प्रकार   psoff = (src0_contig ? (prow * nb01) :
                                    unary_row_offset(prow, ne01, ne02, div_ne01, div_ne02, div_ne012, nb01, nb02,
                                                     nb03)) +
                                   (माप_प्रकार) pcol * माप(भग्न);
            dma_queue_धकेलो(dma_q, dma_make_तथ्य(src_vtcm, तथ्य_src + psoff), ptb, ptb, ptb, 1);
        }

        tile_in_row++;
        col += col_tile;
        अगर (tile_in_row == tiles_per_row) {
            tile_in_row = 0;
            col = 0;
            row++;
        }

        ptile_in_row++;
        pcol += col_tile;
        अगर (ptile_in_row == tiles_per_row) {
            ptile_in_row = 0;
            pcol = 0;
            prow++;
        }
    }

    dma_queue_flush(dma_q);
}

// 5. TRI tiled task.
अटल व्योम unary_thread_tiled_tri_f32(अचिन्हित पूर्णांक nth, अचिन्हित पूर्णांक ith, व्योम * तथ्य) {
    (व्योम) nth;
    स्थिर काष्ठा htp_unary_context * uctx = (स्थिर काष्ठा htp_unary_context *) तथ्य;
    काष्ठा htp_ops_context * octx = uctx->octx;
    स्थिर काष्ठा htp_tensor * src = octx->src[0];
    स्थिर काष्ठा htp_tensor * dst = octx->dst;
    काष्ठा htp_thread_trace * tr = &octx->ctx->trace[ith];

    htp_unary_preamble;

    स्थिर uint32_t src0_nrows_per_thread = uctx->src0_nrows_per_thread;
    स्थिर int32_t * op_params            = octx->op_params;
    स्थिर uint32_t col_tile              = uctx->col_tile;

    स्थिर uint32_t src0_nrows     = uctx->src0_nrows;
    स्थिर uint32_t src0_start_row = uctx->row_start + src0_nrows_per_thread * ith;
    स्थिर uint32_t src0_end_row   = MIN(src0_start_row + src0_nrows_per_thread, uctx->row_start + src0_nrows);

    अगर (src0_start_row >= src0_end_row) {
        वापस;
    }

    स्थिर dma_addr_t तथ्य_src = uctx->तथ्य_src0;
    स्थिर dma_addr_t तथ्य_dst = uctx->तथ्य_dst;

    uint8_t * src0_vtcm_तथ्य = uctx->vtcm_src0 + (ith * uctx->vtcm_src0_आकार_per_thread);
    uint8_t * dst_vtcm_तथ्य  = uctx->vtcm_dst + (ith * uctx->vtcm_dst_आकार_per_thread);

    स्थिर माप_प्रकार src0_half = uctx->src0_vtcm_half_आकार;
    स्थिर माप_प्रकार dst_half  = uctx->dst_vtcm_half_आकार;

    dma_queue * dma_q = octx->ctx->dma[ith];

    स्थिर काष्ठा fastdiv_मूल्यावली * div_ne01  = &uctx->kparams->div_ne01;
    स्थिर काष्ठा fastdiv_मूल्यावली * div_ne02  = &uctx->kparams->div_ne02;
    स्थिर काष्ठा fastdiv_मूल्यावली * div_ne012 = &uctx->kparams->div_ne012;
    स्थिर काष्ठा fastdiv_मूल्यावली * div_tpr   = &uctx->kparams->div_tpr;

    स्थिर uint32_t tiles_per_row = (ne0 + col_tile - 1) / col_tile;
    स्थिर int32_t  tri_ttype     = op_params[0];

    स्थिर bool src0_contig = (nb02 == (माप_प्रकार)ne01 * nb01) &&
                             (nb03 == (माप_प्रकार)ne02 * nb02);
    स्थिर bool dst_contig  = (nb2  == (माप_प्रकार)ne1  * nb1)  &&
                             (nb3  == (माप_प्रकार)ne2  * nb2);

    स्थिर uint32_t योग_tiles = (src0_end_row - src0_start_row) * tiles_per_row;

    क्रम (uint32_t t = 0, vtcm_idx = 0; t < योग_tiles && vtcm_idx < 2; t++, vtcm_idx++) {
        स्थिर uint32_t row  = src0_start_row + t / tiles_per_row;
        स्थिर uint32_t col  = (t % tiles_per_row) * col_tile;
        स्थिर uint32_t tw   = MIN(col_tile, ne0 - col);
        स्थिर माप_प्रकार   tb   = (माप_प्रकार) tw * माप(भग्न);
        स्थिर माप_प्रकार   soff = (src0_contig ? (row * nb01) :
                               unary_row_offset(row, ne01, ne02, div_ne01, div_ne02, div_ne012, nb01, nb02, nb03)) +
                               (माप_प्रकार) col * माप(भग्न);

        dma_queue_धकेलो(dma_q, dma_make_तथ्य(तथ्य_dst, dst_vtcm_तथ्य + (vtcm_idx * dst_half)), 0, 0, 0, 0);
        dma_queue_धकेलो(dma_q, dma_make_तथ्य(src0_vtcm_तथ्य + (vtcm_idx * src0_half), तथ्य_src + soff), tb, tb, tb, 1);
    }

    unary_tiled_tri_compute_fn_t compute = (unary_tiled_tri_compute_fn_t) uctx->compute;

    uint32_t row = src0_start_row;
    uint32_t col = 0;
    uint32_t tile_in_row = 0;
    uint32_t i01 = fastmodulo(row, ne01, div_ne01);

    uint32_t prow = src0_start_row + fastdiv(2, div_tpr);
    uint32_t pcol = fastmodulo(2, tiles_per_row, div_tpr) * col_tile;
    uint32_t ptile_in_row = fastmodulo(2, tiles_per_row, div_tpr);

    क्रम (uint32_t t = 0; t < योग_tiles; t++) {
        व्योम * dst_vtcm = (व्योम *) (uintptr_t) dma_queue_निकालो(dma_q).src;
        व्योम * src_vtcm = (व्योम *) (uintptr_t) dma_queue_निकालो(dma_q).dst;

        स्थिर uint32_t tw = MIN(col_tile, ne0 - col);

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, t);
        compute(src_vtcm, dst_vtcm, tw, col, i01, ne0, tri_ttype);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, t);

        स्थिर माप_प्रकार doff = (dst_contig ? (row * nb1) :
                             unary_row_offset(row, ne1, ne2, div_ne01, div_ne02, div_ne012, nb1, nb2, nb3)) +
                             (माप_प्रकार) col * माप(भग्न);
        स्थिर माप_प्रकार tb   = (माप_प्रकार) tw * माप(भग्न);
        dma_queue_धकेलो(dma_q, dma_make_तथ्य(तथ्य_dst + doff, dst_vtcm), tb, tb, tb, 1);

        स्थिर uint32_t pt = t + 2;
        अगर (pt < योग_tiles) {
            स्थिर uint32_t ptw  = MIN(col_tile, ne0 - pcol);
            स्थिर माप_प्रकार   ptb  = (माप_प्रकार) ptw * माप(भग्न);
            स्थिर माप_प्रकार   psoff = (src0_contig ? (prow * nb01) :
                                    unary_row_offset(prow, ne01, ne02, div_ne01, div_ne02, div_ne012, nb01, nb02,
                                                     nb03)) +
                                   (माप_प्रकार) pcol * माप(भग्न);
            dma_queue_धकेलो(dma_q, dma_make_तथ्य(src_vtcm, तथ्य_src + psoff), ptb, ptb, ptb, 1);
        }

        tile_in_row++;
        col += col_tile;
        अगर (tile_in_row == tiles_per_row) {
            tile_in_row = 0;
            col = 0;
            row++;
            i01++;
            अगर (i01 == ne01) {
                i01 = 0;
            }
        }

        ptile_in_row++;
        pcol += col_tile;
        अगर (ptile_in_row == tiles_per_row) {
            ptile_in_row = 0;
            pcol = 0;
            prow++;
        }
    }

    dma_queue_flush(dma_q);
}

अटल पूर्णांक execute_op_unary(काष्ठा htp_ops_context * octx) {
    पूर्णांक err = HTP_STATUS_OK;

    स्थिर काष्ठा htp_tensor * src0 = octx->src[0];
    स्थिर काष्ठा htp_tensor * dst  = octx->dst;

    स्थिर bool is_f16 = (src0->type == HTP_TYPE_F16);

    स्थिर अक्षर * op_type = शून्य;

    चयन (octx->op) {
        हाल HTP_OP_NORM:            op_type = is_f16 ? "norm-f16"     : "norm-f32";         अवरोध;
        हाल HTP_OP_RMS_NORM:        op_type = is_f16 ? "rmsnorm-f16"  : "rmsnorm-f32";      अवरोध;
        हाल HTP_OP_RMS_NORM_MUL:    op_type = "rmsnorm-mul-f32";                            अवरोध;
        हाल HTP_OP_SCALE:           op_type = is_f16 ? "scale-f16"    : "scale-f32";        अवरोध;
        हाल HTP_OP_CLAMP:           op_type = is_f16 ? "clamp-f16"    : "clamp-f32";        अवरोध;
        हाल HTP_OP_LEAKY_RELU:      op_type = "leaky-relu-f32";                             अवरोध;
        हाल HTP_OP_SQR:             op_type = is_f16 ? "sqr-f16"      : "sqr-f32";          अवरोध;
        हाल HTP_OP_SQRT:            op_type = is_f16 ? "sqrt-f16"     : "sqrt-f32";         अवरोध;
        हाल HTP_OP_UNARY_NEG:       op_type = "neg-f32";                                    अवरोध;
        हाल HTP_OP_UNARY_EXP:       op_type = "exp-f32";                                    अवरोध;
        हाल HTP_OP_UNARY_SIGMOID:   op_type = "sigmoid-f32";                                अवरोध;
        हाल HTP_OP_UNARY_SILU:      op_type = "silu-f32";                                   अवरोध;
        हाल HTP_OP_UNARY_GELU:      op_type = "gelu-f32";                                   अवरोध;
        हाल HTP_OP_UNARY_GELU_QUICK: op_type = "gelu-quick-f32";                             अवरोध;
        हाल HTP_OP_UNARY_GELU_ERF:  op_type = "gelu-erf-f32";                               अवरोध;
        हाल HTP_OP_UNARY_SOFTPLUS:  op_type = "softplus-f32";                               अवरोध;
        हाल HTP_OP_UNARY_TANH:      op_type = "tanh-f32";                                   अवरोध;
        हाल HTP_OP_UNARY_ABS:       op_type = is_f16 ? "abs-f16"      : "abs-f32";          अवरोध;
        हाल HTP_OP_UNARY_LOG:       op_type = is_f16 ? "log-f16"      : "log-f32";          अवरोध;
        हाल HTP_OP_UNARY_RELU:      op_type = "relu-f32";                                   अवरोध;
        हाल HTP_OP_UNARY_STEP:      op_type = is_f16 ? "step-f16"     : "step-f32";         अवरोध;
        हाल HTP_OP_L2_NORM:         op_type = is_f16 ? "l2norm-f16"   : "l2norm-f32";       अवरोध;
        हाल HTP_OP_TRI:             op_type = "tri-f32";                                    अवरोध;
        हाल HTP_OP_HADAMARD:        op_type = is_f16 ? "hadamard-f16" : "hadamard-f32";     अवरोध;
        शेष:
            FARF(ERROR, "Unsupported unary Op %u\n", octx->op);
            वापस HTP_STATUS_NO_SUPPORT;
    }

    // F16 only has row-block kernels for this subset of ops (see the dispatch switch
    // below) - reject everything else up front, before touching kparams/VTCM.
    अगर (is_f16) {
        चयन (octx->op) {
            हाल HTP_OP_NORM:
            हाल HTP_OP_RMS_NORM:
            हाल HTP_OP_SCALE:
            हाल HTP_OP_CLAMP:
            हाल HTP_OP_SQR:
            हाल HTP_OP_SQRT:
            हाल HTP_OP_L2_NORM:
            हाल HTP_OP_UNARY_ABS:
            हाल HTP_OP_UNARY_LOG:
            हाल HTP_OP_UNARY_STEP:
            हाल HTP_OP_HADAMARD:
                अवरोध;
            शेष:
                FARF(ERROR, "unary-%s: not supported for F16\n", op_type);
                वापस HTP_STATUS_NO_SUPPORT;
        }
    }

    स्थिर काष्ठा htp_unary_kernel_params * kparams = (स्थिर काष्ठा htp_unary_kernel_params *) octx->kernel_params;

    अगर (!htp_ops_context_set_n_threads(octx, kparams->n_threads)) {
        वापस HTP_STATUS_INVAL_PARAMS;
    }

    स्थिर uint32_t src0_nrows = src0->ne[1] * src0->ne[2] * src0->ne[3];
    स्थिर माप_प्रकार src0_elem_आकार = is_f16 ? माप(_Float16) : माप(भग्न);
    स्थिर माप_प्रकार dst_elem_आकार  = (dst->type == HTP_TYPE_F16) ? माप(_Float16) : माप(भग्न);
    स्थिर माप_प्रकार src0_तथ्य_row_आकार = src0->ne[0] * src0_elem_आकार;
    स्थिर माप_प्रकार dst_तथ्य_row_आकार  = dst->ne[0]  * dst_elem_आकार;

    uint32_t row_start = 0;
    uint32_t nrows     = src0_nrows;

    अगर (octx->ctx->mdev.गिनती > 1) {
        uint32_t rows_per_chunk = 0;
        htp_tensor_mdev_rows_per_chunk(dst, (uint32_t) dst_elem_आकार, (uint32_t) dst_तथ्य_row_आकार, &rows_per_chunk);
        स्थिर काष्ठा htp_tensor_mdev_range range = htp_tensor_mdev_partition(src0_nrows, rows_per_chunk, octx->ctx->mdev.idx, octx->ctx->mdev.गिनती, &octx->ctx->mdev.गिनती_div);
        row_start = range.start;
        nrows     = range.गिनती;
    }

    अगर (nrows == 0) {
        वापस HTP_STATUS_OK;
    }

    स्थिर uint32_t n_threads = octx->n_threads;

    स्थिर माप_प्रकार src0_row_आकार_aligned = kparams->src0_row_आकार_aligned;
    स्थिर माप_प्रकार dst_row_आकार_aligned  = kparams->dst_row_आकार_aligned;

    // Always 0 for F16 - htp_unary_vtcm_layout_build() keeps F16 on the row-block path,
    // since only F32 has unary_task_f32_tiled_* kernels.
    स्थिर uint32_t col_tile = kparams->col_tile;

    माप_प्रकार src1_तथ्य_row_आकार = 0;
    माप_प्रकार src1_row_आकार_aligned = kparams->src1_row_आकार_aligned;
    bool broadcast_weight = kparams->broadcast_weight;
    स्थिर काष्ठा htp_tensor * src1 = शून्य;

    // RMS_NORM_MUL fusion is F32-only (its weight tensor is always F32; see
    // try_fuse_node()'s type guard), so this never triggers when is_f16 is true.
    अगर (octx->op == HTP_OP_RMS_NORM_MUL) {
        src1 = octx->src[1];
        src1_तथ्य_row_आकार = src1->ne[0] * माप(भग्न);
    }

    अगर (octx->ctx->vtcm_आकार < (माप_प्रकार)kparams->vtcm_आकार) {
        FARF(ERROR, "unary-%s : current VTCM reservation %zu is too small, needed %zu\n", op_type, octx->ctx->vtcm_आकार, (माप_प्रकार)kparams->vtcm_आकार);
        वापस HTP_STATUS_VTCM_TOO_SMALL;
    }

    octx->src0_spad.src = शून्य;
    octx->src1_spad.src = शून्य;
    octx->dst_spad.src  = शून्य;

    FARF(ऊपरी_2, "%s: (%ux%ux%ux%u) -> (%ux%ux%ux%u) : src0-vtcm-size %u src1-vtcm-size %u dst-vtcm-size %u\n", op_type,
         src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3], dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3],
         kparams->vtcm_src0_आकार, kparams->vtcm_src1_आकार, kparams->vtcm_dst_आकार);

    uint8_t * स्थिर base = (uint8_t *) octx->ctx->vtcm_base;
    काष्ठा htp_unary_context uctx = {
        .octx                  = octx,
        .kparams               = kparams,
        .src0_nrows_per_thread = fastdiv(nrows + n_threads - 1, &octx->n_threads_div),
        .src0_nrows            = nrows,
        .row_start             = row_start,

        .तथ्य_src0             = src0->तथ्य,
        .तथ्य_src1             = (octx->op == HTP_OP_RMS_NORM_MUL) ? src1->तथ्य : 0,
        .तथ्य_dst              = dst->तथ्य,

        .src0_तथ्य_row_आकार    = src0_तथ्य_row_आकार,
        .src1_तथ्य_row_आकार    = src1_तथ्य_row_आकार,
        .dst_तथ्य_row_आकार     = dst_तथ्य_row_आकार,

        .src0_row_आकार_aligned = src0_row_आकार_aligned,
        .src1_row_आकार_aligned = src1_row_आकार_aligned,
        .dst_row_आकार_aligned  = dst_row_आकार_aligned,

        .src0_vtcm_half_आकार   = kparams->vtcm_src0_आकार_per_thread / 2,
        .src1_vtcm_half_आकार   = (octx->op == HTP_OP_RMS_NORM_MUL) ? (kparams->vtcm_src1_आकार_per_thread / (broadcast_weight ? 1 : 2)) : 0,
        .dst_vtcm_half_आकार    = kparams->vtcm_dst_आकार_per_thread / 2,

        .block                 = kparams->block,
        .nc                    = src0->ne[0],
        .col_tile              = col_tile,
        .broadcast_weight      = broadcast_weight,

        .vtcm_src0             = VTCM_LAYOUT_PTR(uint8_t, base, 0),
        .vtcm_src1             = VTCM_LAYOUT_PTR_OPTIONAL(uint8_t, base, kparams->vtcm_src0_आकार, kparams->vtcm_src1_आकार > 0),
        .vtcm_dst              = VTCM_LAYOUT_PTR(uint8_t, base, kparams->vtcm_src0_आकार + kparams->vtcm_src1_आकार),

        .vtcm_src0_आकार_per_thread = kparams->vtcm_src0_आकार_per_thread,
        .vtcm_src1_आकार_per_thread = kparams->vtcm_src1_आकार_per_thread,
        .vtcm_dst_आकार_per_thread  = kparams->vtcm_dst_आकार_per_thread,
    };

    FARF(ऊपरी_2, "%s: %s mode (col_tile %u)\n", op_type, col_tile ? "tiled" : "row-block", col_tile);

    worker_callback_t task_func = शून्य;
    व्योम * compute_func = शून्य;

    अगर (col_tile) {
        task_func = unary_thread_tiled;
        चयन (octx->op) {
            हाल HTP_OP_SCALE:           compute_func = (व्योम *) tile_scale_f32;          अवरोध;
            हाल HTP_OP_CLAMP:           compute_func = (व्योम *) tile_clamp_f32;          अवरोध;
            हाल HTP_OP_LEAKY_RELU:      compute_func = (व्योम *) tile_leaky_relu_f32;     अवरोध;
            हाल HTP_OP_SQR:             compute_func = (व्योम *) tile_sqr_f32;            अवरोध;
            हाल HTP_OP_SQRT:            compute_func = (व्योम *) tile_sqrt_f32;           अवरोध;
            हाल HTP_OP_UNARY_NEG:       compute_func = (व्योम *) tile_neg_f32;            अवरोध;
            हाल HTP_OP_UNARY_EXP:       compute_func = (व्योम *) tile_exp_f32;            अवरोध;
            हाल HTP_OP_UNARY_SIGMOID:   compute_func = (व्योम *) tile_sigmoid_f32;        अवरोध;
            हाल HTP_OP_UNARY_SILU:      compute_func = (व्योम *) tile_silu_f32;           अवरोध;
            हाल HTP_OP_UNARY_GELU:      compute_func = (व्योम *) tile_gelu_f32;           अवरोध;
            हाल HTP_OP_UNARY_GELU_QUICK: compute_func = (व्योम *) tile_gelu_quick_f32;     अवरोध;
            हाल HTP_OP_UNARY_GELU_ERF:  compute_func = (व्योम *) tile_gelu_erf_f32;       अवरोध;
            हाल HTP_OP_UNARY_SOFTPLUS:  compute_func = (व्योम *) tile_softplus_f32;       अवरोध;
            हाल HTP_OP_UNARY_TANH:      compute_func = (व्योम *) tile_tanh_f32;           अवरोध;
            हाल HTP_OP_UNARY_ABS:       compute_func = (व्योम *) tile_abs_f32;            अवरोध;
            हाल HTP_OP_UNARY_LOG:       compute_func = (व्योम *) tile_log_f32;            अवरोध;
            हाल HTP_OP_UNARY_RELU:      compute_func = (व्योम *) tile_relu_f32;           अवरोध;
            हाल HTP_OP_UNARY_STEP:      compute_func = (व्योम *) tile_step_f32;           अवरोध;
            हाल HTP_OP_TRI:
                task_func    = unary_thread_tiled_tri_f32;
                compute_func = (व्योम *) tri_apply_tile_f32;
                अवरोध;
            शेष:                     अवरोध;
        }
    } अन्यथा अगर (is_f16) {
        task_func = unary_thread_row_block;
        चयन (octx->op) {
            हाल HTP_OP_NORM:            compute_func = (व्योम *) norm_f16;                अवरोध;
            हाल HTP_OP_RMS_NORM:        compute_func = (व्योम *) rms_norm_f16;            अवरोध;
            हाल HTP_OP_SCALE:           compute_func = (व्योम *) scale_f16;               अवरोध;
            हाल HTP_OP_CLAMP:           compute_func = (व्योम *) clamp_f16;               अवरोध;
            हाल HTP_OP_SQR:             compute_func = (व्योम *) sqr_f16;                 अवरोध;
            हाल HTP_OP_SQRT:            compute_func = (व्योम *) sqrt_f16;                अवरोध;
            हाल HTP_OP_L2_NORM:         compute_func = (व्योम *) l2_norm_f16;             अवरोध;
            हाल HTP_OP_UNARY_ABS:       compute_func = (व्योम *) abs_f16;                 अवरोध;
            हाल HTP_OP_UNARY_LOG:       compute_func = (व्योम *) log_f16;                 अवरोध;
            हाल HTP_OP_UNARY_STEP:      compute_func = (व्योम *) step_f16;                अवरोध;
            हाल HTP_OP_HADAMARD:        compute_func = (व्योम *) hadamard_f16;            अवरोध;
            शेष:                     अवरोध;
        }
    } अन्यथा {
        task_func = unary_thread_row_block;
        चयन (octx->op) {
            हाल HTP_OP_NORM:            compute_func = (व्योम *) norm_f32;                अवरोध;
            हाल HTP_OP_RMS_NORM:        compute_func = (व्योम *) rms_norm_f32;            अवरोध;
            हाल HTP_OP_RMS_NORM_MUL:
                task_func    = unary_thread_rms_norm_mul_f32;
                compute_func = (व्योम *) rms_norm_mul_f32;
                अवरोध;
            हाल HTP_OP_SCALE:           compute_func = (व्योम *) scale_f32;               अवरोध;
            हाल HTP_OP_CLAMP:           compute_func = (व्योम *) clamp_f32;               अवरोध;
            हाल HTP_OP_LEAKY_RELU:      compute_func = (व्योम *) leaky_relu_f32;          अवरोध;
            हाल HTP_OP_SQR:             compute_func = (व्योम *) sqr_f32;                 अवरोध;
            हाल HTP_OP_SQRT:            compute_func = (व्योम *) sqrt_f32;                अवरोध;
            हाल HTP_OP_UNARY_NEG:       compute_func = (व्योम *) neg_f32;                 अवरोध;
            हाल HTP_OP_UNARY_EXP:       compute_func = (व्योम *) exp_f32;                 अवरोध;
            हाल HTP_OP_UNARY_SIGMOID:   compute_func = (व्योम *) sigmoid_f32;             अवरोध;
            हाल HTP_OP_UNARY_SILU:      compute_func = (व्योम *) silu_f32;                अवरोध;
            हाल HTP_OP_UNARY_GELU:      compute_func = (व्योम *) gelu_f32;                अवरोध;
            हाल HTP_OP_UNARY_GELU_QUICK: compute_func = (व्योम *) gelu_quick_f32;          अवरोध;
            हाल HTP_OP_UNARY_GELU_ERF:  compute_func = (व्योम *) gelu_erf_f32;            अवरोध;
            हाल HTP_OP_UNARY_SOFTPLUS:  compute_func = (व्योम *) softplus_f32;            अवरोध;
            हाल HTP_OP_UNARY_TANH:      compute_func = (व्योम *) tanh_f32;                अवरोध;
            हाल HTP_OP_UNARY_ABS:       compute_func = (व्योम *) abs_f32;                 अवरोध;
            हाल HTP_OP_UNARY_LOG:       compute_func = (व्योम *) log_f32;                 अवरोध;
            हाल HTP_OP_UNARY_RELU:      compute_func = (व्योम *) relu_f32;                अवरोध;
            हाल HTP_OP_UNARY_STEP:      compute_func = (व्योम *) step_f32;                अवरोध;
            हाल HTP_OP_L2_NORM:         compute_func = (व्योम *) l2_norm_f32;             अवरोध;
            हाल HTP_OP_HADAMARD:        compute_func = (व्योम *) hadamard_f32;            अवरोध;
            हाल HTP_OP_TRI:
                task_func    = unary_thread_tri_f32;
                compute_func = (व्योम *) tri_f32;
                अवरोध;
            शेष:                     अवरोध;
        }
    }

    अगर (!task_func || !compute_func) {
        FARF(ERROR, "execute_op_unary: task function is NULL for op %d\n", octx->op);
        वापस HTP_STATUS_NO_SUPPORT;
    }

    uctx.compute = compute_func;
    work_queue_run(octx->ctx->work_queue, task_func, &uctx, n_threads);

    वापस err;
}

पूर्णांक op_unary(काष्ठा htp_ops_context * octx) {
    चयन (octx->src[0]->type) {
        हाल HTP_TYPE_F32:
        हाल HTP_TYPE_F16:
            वापस execute_op_unary(octx);

        शेष:
            वापस HTP_STATUS_NO_SUPPORT;
    }
}
