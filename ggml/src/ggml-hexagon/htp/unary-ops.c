#pragma clang diagnostic ignored "-Wunused-variable"
#pragma clang diagnostic ignored "-Wunused-function"
#pragma clang diagnostic ignored "-Wunused-but-set-variable"

#અન્તર્ભાવ <HAP_farf.h>
#અન્તર્ભાવ <HAP_perf.h>

#અન્તર્ભાવ <math.h>
#અન્તર્ભાવ <string.h>

#અન્તર્ભાવ "dma-queue.h"
#અન્તર્ભાવ "hex-fastdiv.h"
#અન્તર્ભાવ "hvx-exp.h"
#અન્તર્ભાવ "hvx-sigmoid.h"
#અન્તર્ભાવ "hvx-utils.h"
#અન્તર્ભાવ "unary-ops.h"

#define GGML_COMMON_DECL_C
#અન્તર્ભાવ "ggml-common.h"
#અન્તર્ભાવ "htp-ctx.h"
#અન્તર્ભાવ "htp-ops.h"
#અન્તર્ભાવ "htp-tensor.h"
#અન્તર્ભાવ "htp-vtcm.h"
#અન્તર્ભાવ "hex-profile.h"

સંરચના htp_unary_context;

પ્રકારવ્યાખ્યા શૂન્ય (*unary_compute_fn_t)(અચલ શૂન્ય * restrict src,
                                   શૂન્ય * restrict dst,
                                   uint32_t અંક_rows,
                                   અચલ સંરચના htp_unary_context * uctx);

પ્રકારવ્યાખ્યા શૂન્ય (*unary_rms_norm_mul_compute_fn_t)(અચલ શૂન્ય * restrict src,
                                                અચલ શૂન્ય * restrict weight,
                                                શૂન્ય * restrict dst,
                                                uint32_t અંક_rows,
                                                અચલ સંરચના htp_unary_context * uctx);

પ્રકારવ્યાખ્યા શૂન્ય (*unary_tri_compute_fn_t)(અચલ શૂન્ય * restrict src,
                                       શૂન્ય * restrict dst,
                                       uint32_t અંક_rows,
                                       uint32_t ir,
                                       અચલ સંરચના htp_unary_context * uctx);

પ્રકારવ્યાખ્યા શૂન્ય (*unary_tile_compute_fn_t)(શૂન્ય * restrict dst,
                                        અચલ શૂન્ય * restrict src,
                                        uint32_t tw,
                                        અચલ સંરચના htp_unary_context * uctx);

પ્રકારવ્યાખ્યા શૂન્ય (*unary_tiled_tri_compute_fn_t)(અચલ શૂન્ય * restrict src,
                                             શૂન્ય * restrict dst,
                                             uint32_t tile_elems,
                                             uint32_t col_start,
                                             uint32_t i01,
                                             uint32_t ne0,
                                             int32_t ttype);

સંરચના htp_unary_context {
    સંરચના htp_ops_context * octx;
    અચલ સંરચના htp_unary_kernel_params * kparams;

    શૂન્ય *                    compute;

    dma_addr_t                data_src0;
    dma_addr_t                data_src1;            // weight/scale tensor for RMS_NORM_MUL
    dma_addr_t                data_dst;

    size_t                    src0_data_row_size;   // actual data bytes per row
    size_t                    src1_data_row_size;
    size_t                    dst_data_row_size;    // actual data bytes per row

    size_t                    src0_row_size_aligned;
    size_t                    src1_row_size_aligned;
    size_t                    dst_row_size_aligned;

    size_t                    src0_vtcm_half_size;
    size_t                    src1_vtcm_half_size;
    size_t                    dst_vtcm_half_size;

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

    size_t                    vtcm_src0_size_per_thread;
    size_t                    vtcm_src1_size_per_thread;
    size_t                    vtcm_dst_size_per_thread;
};

// Convert flat row index to DDR byte offset using the tensor's actual strides.
// ir = i1 + ne1*(i2 + ne2*i3)  =>  offset = i1*nb1 + i2*nb2 + i3*nb3
સ્થિર અંતઃરેખ size_t unary_row_offset(uint32_t ir,
                                      uint32_t ne1, uint32_t ne2,
                                      અચલ સંરચના fastdiv_કિંમતો * div_ne1,
                                      અચલ સંરચના fastdiv_કિંમતો * div_ne2,
                                      અચલ સંરચના fastdiv_કિંમતો * div_ne12,
                                      size_t nb1, size_t nb2, size_t nb3) {
    અચલ uint32_t i1 = fastmodulo(ir, ne1, div_ne1);
    અચલ uint32_t ir_div_ne1 = fastdiv(ir, div_ne1);
    અચલ uint32_t i2 = fastmodulo(ir_div_ne1, ne2, div_ne2);
    અચલ uint32_t i3 = fastdiv(ir, div_ne12);
    પાછા i1 * nb1 + i2 * nb2 + i3 * nb3;
}

// Safe DMA block size from row `ir`: clamp to the tighter dim-1 slice
// boundary of src and dst so the nb1 stride stays valid for all rows.
સ્થિર અંતઃરેખ uint32_t unary_block_size(uint32_t ir,
                                        uint32_t end_row,
                                        uint32_t block,
                                        bool src_contig,
                                        bool dst_contig,
                                        uint32_t ne1,
                                        અચલ સંરચના fastdiv_કિંમતો * div_ne1) {
    uint32_t limit = MIN(block, end_row - ir);

    જો (!src_contig || !dst_contig) {
        અચલ uint32_t slice_end = (fastdiv(ir, div_ne1) + 1) * ne1;
        limit = MIN(limit, slice_end - ir);
    }

    પાછા limit;
}

#define htp_unary_preamble            \
    અચલ uint32_t ne00 = src->ne[0]; \
    અચલ uint32_t ne01 = src->ne[1]; \
    અચલ uint32_t ne02 = src->ne[2]; \
    અચલ uint32_t ne03 = src->ne[3]; \
                                      \
    અચલ uint32_t ne0 = dst->ne[0];  \
    અચલ uint32_t ne1 = dst->ne[1];  \
    અચલ uint32_t ne2 = dst->ne[2];  \
    અચલ uint32_t ne3 = dst->ne[3];  \
                                      \
    અચલ uint32_t nb00 = src->nb[0]; \
    અચલ uint32_t nb01 = src->nb[1]; \
    અચલ uint32_t nb02 = src->nb[2]; \
    અચલ uint32_t nb03 = src->nb[3]; \
                                      \
    અચલ uint32_t nb0 = dst->nb[0];  \
    અચલ uint32_t nb1 = dst->nb[1];  \
    અચલ uint32_t nb2 = dst->nb[2];  \
    અચલ uint32_t nb3 = dst->nb[3];

#define htp_unary_op_preamble                                         \
    int32_t * op_params = uctx->octx->op_params;                      \
    અચલ uint32_t ne0 = uctx->nc;                                    \
    અચલ size_t src0_row_size_aligned = uctx->src0_row_size_aligned; \
    અચલ size_t dst_row_size_aligned = uctx->dst_row_size_aligned;

સ્થિર શૂન્ય scale_f32(અચલ શૂન્ય * restrict src,
                      શૂન્ય * restrict dst,
                      અચલ uint32_t અંક_rows,
                      અચલ સંરચના htp_unary_context * uctx) {
    htp_unary_op_preamble;
    દશાંશ scale = 0.f;
    દશાંશ bias  = 0.f;
    memcpy(&scale, &op_params[0], કદ(દશાંશ));
    memcpy(&bias,  &op_params[1], કદ(દશાંશ));

    માટે (uint32_t ir = 0; ir < અંક_rows; ir++) {
        અચલ uint8_t * restrict src_local = (અચલ uint8_t *)src + (ir * src0_row_size_aligned);
        uint8_t * restrict dst_local       = (uint8_t *)dst + (ir * dst_row_size_aligned);

        hvx_scale_offset_f32_aa((uint8_t *) dst_local, (અચલ uint8_t *) src_local, ne0, scale, bias);
    }
}

સ્થિર શૂન્ય clamp_f32(અચલ શૂન્ય * restrict src,
                      શૂન્ય * restrict dst,
                      અચલ uint32_t અંક_rows,
                      અચલ સંરચના htp_unary_context * uctx) {
    htp_unary_op_preamble;
    દશાંશ min = 0.f;
    દશાંશ max = 0.f;
    memcpy(&min, &op_params[0], કદ(દશાંશ));
    memcpy(&max, &op_params[1], કદ(દશાંશ));

    માટે (uint32_t ir = 0; ir < અંક_rows; ir++) {
        અચલ uint8_t * restrict src_local = (અચલ uint8_t *)src + (ir * src0_row_size_aligned);
        uint8_t * restrict dst_local       = (uint8_t *)dst + (ir * dst_row_size_aligned);

        hvx_clamp_scalar_f32(dst_local, src_local, min, max, ne0);
    }
}

સ્થિર શૂન્ય leaky_relu_f32(અચલ શૂન્ય * restrict src,
                           શૂન્ય * restrict dst,
                           અચલ uint32_t અંક_rows,
                           અચલ સંરચના htp_unary_context * uctx) {
    htp_unary_op_preamble;
    દશાંશ negative_slope = 0.f;
    memcpy(&negative_slope, &op_params[0], કદ(દશાંશ));

    માટે (uint32_t ir = 0; ir < અંક_rows; ir++) {
        અચલ uint8_t * restrict src_local = (અચલ uint8_t *)src + (ir * src0_row_size_aligned);
        uint8_t * restrict dst_local       = (uint8_t *)dst + (ir * dst_row_size_aligned);

        hvx_leaky_relu_scalar_f32(dst_local, src_local, negative_slope, ne0);
    }
}

સ્થિર શૂન્ય rms_norm_f32(અચલ શૂન્ય * restrict src,
                         શૂન્ય * restrict dst,
                         અચલ uint32_t અંક_rows,
                         અચલ સંરચના htp_unary_context * uctx) {
    htp_unary_op_preamble;
    દશાંશ epsilon = 0.f;
    memcpy(&epsilon, op_params, કદ(દશાંશ));

    માટે (uint32_t ir = 0; ir < અંક_rows; ir++) {
        અચલ uint8_t * restrict src_local = (અચલ uint8_t *)src + (ir * src0_row_size_aligned);
        uint8_t * restrict dst_local       = (uint8_t *)dst + (ir * dst_row_size_aligned);

        hvx_fast_rms_norm_f32((અચલ uint8_t *) src_local, (uint8_t *) dst_local, ne0, epsilon);
    }
}

સ્થિર શૂન્ય rms_norm_mul_f32(અચલ શૂન્ય * restrict src,
                             અચલ શૂન્ય * restrict weight,
                             શૂન્ય * restrict dst,
                             અચલ uint32_t અંક_rows,
                             અચલ સંરચના htp_unary_context * uctx) {
    htp_unary_op_preamble;
    દશાંશ epsilon = 0.f;
    memcpy(&epsilon, op_params, કદ(દશાંશ));

    માટે (uint32_t ir = 0; ir < અંક_rows; ir++) {
        અચલ uint8_t * restrict src_local = (અચલ uint8_t *)src + (ir * src0_row_size_aligned);
        અચલ uint8_t * restrict w_local   = (અચલ uint8_t *)weight + (uctx->broadcast_weight ? 0 : ir * uctx->src1_row_size_aligned);
        uint8_t * restrict dst_local       = (uint8_t *)dst + (ir * dst_row_size_aligned);

        hvx_fast_rms_norm_mul_f32(src_local, w_local, dst_local, ne0, epsilon);
    }
}

સ્થિર શૂન્ય norm_f32(અચલ શૂન્ય * restrict src,
                     શૂન્ય * restrict dst,
                     અચલ uint32_t અંક_rows,
                     અચલ સંરચના htp_unary_context * uctx) {
    htp_unary_op_preamble;
    દશાંશ epsilon = 0.f;
    memcpy(&epsilon, op_params, કદ(દશાંશ));

    માટે (uint32_t ir = 0; ir < અંક_rows; ir++) {
        અચલ uint8_t * restrict src_local = (અચલ uint8_t *)src + (ir * src0_row_size_aligned);
        uint8_t * restrict dst_local       = (uint8_t *)dst + (ir * dst_row_size_aligned);

        hvx_fast_norm_f32((અચલ uint8_t *) src_local, (uint8_t *) dst_local, ne0, epsilon);
    }
}

સ્થિર શૂન્ય sqr_f32(અચલ શૂન્ય * restrict src,
                    શૂન્ય * restrict dst,
                    અચલ uint32_t અંક_rows,
                    અચલ સંરચના htp_unary_context * uctx) {
    htp_unary_op_preamble;

    માટે (uint32_t ir = 0; ir < અંક_rows; ir++) {
        અચલ uint8_t * restrict src_local = (અચલ uint8_t *)src + (ir * src0_row_size_aligned);
        uint8_t * restrict dst_local       = (uint8_t *)dst + (ir * dst_row_size_aligned);

        hvx_sqr_f32_aa((uint8_t *) dst_local, (અચલ uint8_t *) src_local, ne0);
    }
}

સ્થિર શૂન્ય sqrt_f32(અચલ શૂન્ય * restrict src,
                     શૂન્ય * restrict dst,
                     અચલ uint32_t અંક_rows,
                     અચલ સંરચના htp_unary_context * uctx) {
    htp_unary_op_preamble;

    માટે (uint32_t ir = 0; ir < અંક_rows; ir++) {
        અચલ uint8_t * restrict src_local = (અચલ uint8_t *)src + (ir * src0_row_size_aligned);
        uint8_t * restrict dst_local       = (uint8_t *)dst + (ir * dst_row_size_aligned);

        hvx_sqrt_f32_aa((uint8_t *) dst_local, (અચલ uint8_t *) src_local, ne0);
    }
}

સ્થિર શૂન્ય scale_f16(અચલ શૂન્ય * restrict src,
                      શૂન્ય * restrict dst,
                      અચલ uint32_t અંક_rows,
                      અચલ સંરચના htp_unary_context * uctx) {
    htp_unary_op_preamble;
    દશાંશ scale = 0.f;
    દશાંશ bias  = 0.f;
    memcpy(&scale, &op_params[0], કદ(દશાંશ));
    memcpy(&bias,  &op_params[1], કદ(દશાંશ));

    માટે (uint32_t ir = 0; ir < અંક_rows; ir++) {
        અચલ uint8_t * restrict src_local = (અચલ uint8_t *)src + (ir * src0_row_size_aligned);
        uint8_t * restrict dst_local       = (uint8_t *)dst + (ir * dst_row_size_aligned);

        hvx_scale_offset_f16_aa((uint8_t *) dst_local, (અચલ uint8_t *) src_local, ne0, scale, bias);
    }
}

સ્થિર શૂન્ય clamp_f16(અચલ શૂન્ય * restrict src,
                      શૂન્ય * restrict dst,
                      અચલ uint32_t અંક_rows,
                      અચલ સંરચના htp_unary_context * uctx) {
    htp_unary_op_preamble;
    દશાંશ min = 0.f;
    દશાંશ max = 0.f;
    memcpy(&min, &op_params[0], કદ(દશાંશ));
    memcpy(&max, &op_params[1], કદ(દશાંશ));

    માટે (uint32_t ir = 0; ir < અંક_rows; ir++) {
        અચલ uint8_t * restrict src_local = (અચલ uint8_t *)src + (ir * src0_row_size_aligned);
        uint8_t * restrict dst_local       = (uint8_t *)dst + (ir * dst_row_size_aligned);

        hvx_clamp_scalar_f16(dst_local, src_local, (_Float16) min, (_Float16) max, ne0);
    }
}

સ્થિર શૂન્ય rms_norm_f16(અચલ શૂન્ય * restrict src,
                         શૂન્ય * restrict dst,
                         અચલ uint32_t અંક_rows,
                         અચલ સંરચના htp_unary_context * uctx) {
    htp_unary_op_preamble;
    દશાંશ epsilon = 0.f;
    memcpy(&epsilon, op_params, કદ(દશાંશ));

    માટે (uint32_t ir = 0; ir < અંક_rows; ir++) {
        અચલ uint8_t * restrict src_local = (અચલ uint8_t *)src + (ir * src0_row_size_aligned);
        uint8_t * restrict dst_local       = (uint8_t *)dst + (ir * dst_row_size_aligned);

        hvx_fast_rms_norm_f16((અચલ uint8_t *) src_local, (uint8_t *) dst_local, ne0, epsilon);
    }
}

સ્થિર શૂન્ય norm_f16(અચલ શૂન્ય * restrict src,
                     શૂન્ય * restrict dst,
                     અચલ uint32_t અંક_rows,
                     અચલ સંરચના htp_unary_context * uctx) {
    htp_unary_op_preamble;
    દશાંશ epsilon = 0.f;
    memcpy(&epsilon, op_params, કદ(દશાંશ));

    માટે (uint32_t ir = 0; ir < અંક_rows; ir++) {
        અચલ uint8_t * restrict src_local = (અચલ uint8_t *)src + (ir * src0_row_size_aligned);
        uint8_t * restrict dst_local       = (uint8_t *)dst + (ir * dst_row_size_aligned);

        hvx_fast_norm_f16((અચલ uint8_t *) src_local, (uint8_t *) dst_local, ne0, epsilon);
    }
}

સ્થિર શૂન્ય sqr_f16(અચલ શૂન્ય * restrict src,
                    શૂન્ય * restrict dst,
                    અચલ uint32_t અંક_rows,
                    અચલ સંરચના htp_unary_context * uctx) {
    htp_unary_op_preamble;

    માટે (uint32_t ir = 0; ir < અંક_rows; ir++) {
        અચલ uint8_t * restrict src_local = (અચલ uint8_t *)src + (ir * src0_row_size_aligned);
        uint8_t * restrict dst_local       = (uint8_t *)dst + (ir * dst_row_size_aligned);

        hvx_sqr_f16_aa((uint8_t *) dst_local, (અચલ uint8_t *) src_local, ne0);
    }
}

સ્થિર શૂન્ય sqrt_f16(અચલ શૂન્ય * restrict src,
                     શૂન્ય * restrict dst,
                     અચલ uint32_t અંક_rows,
                     અચલ સંરચના htp_unary_context * uctx) {
    htp_unary_op_preamble;

    માટે (uint32_t ir = 0; ir < અંક_rows; ir++) {
        અચલ uint8_t * restrict src_local = (અચલ uint8_t *)src + (ir * src0_row_size_aligned);
        uint8_t * restrict dst_local       = (uint8_t *)dst + (ir * dst_row_size_aligned);

        hvx_sqrt_f16_aa((uint8_t *) dst_local, (અચલ uint8_t *) src_local, ne0);
    }
}

સ્થિર શૂન્ય abs_f16(અચલ શૂન્ય * restrict src,
                    શૂન્ય * restrict dst,
                    અચલ uint32_t અંક_rows,
                    અચલ સંરચના htp_unary_context * uctx) {
    htp_unary_op_preamble;

    માટે (uint32_t ir = 0; ir < અંક_rows; ir++) {
        અચલ uint8_t * restrict src_local = (અચલ uint8_t *)src + (ir * src0_row_size_aligned);
        uint8_t * restrict dst_local       = (uint8_t *)dst + (ir * dst_row_size_aligned);

        hvx_abs_f16_aa((uint8_t *) dst_local, (અચલ uint8_t *) src_local, ne0);
    }
}

સ્થિર શૂન્ય log_f16(અચલ શૂન્ય * restrict src,
                    શૂન્ય * restrict dst,
                    અચલ uint32_t અંક_rows,
                    અચલ સંરચના htp_unary_context * uctx) {
    htp_unary_op_preamble;

    માટે (uint32_t ir = 0; ir < અંક_rows; ir++) {
        અચલ uint8_t * restrict src_local = (અચલ uint8_t *)src + (ir * src0_row_size_aligned);
        uint8_t * restrict dst_local       = (uint8_t *)dst + (ir * dst_row_size_aligned);

        hvx_log_f16_aa((uint8_t *) dst_local, (અચલ uint8_t *) src_local, ne0);
    }
}

સ્થિર શૂન્ય step_f16(અચલ શૂન્ય * restrict src,
                     શૂન્ય * restrict dst,
                     અચલ uint32_t અંક_rows,
                     અચલ સંરચના htp_unary_context * uctx) {
    htp_unary_op_preamble;

    માટે (uint32_t ir = 0; ir < અંક_rows; ir++) {
        અચલ uint8_t * restrict src_local = (અચલ uint8_t *)src + (ir * src0_row_size_aligned);
        uint8_t * restrict dst_local       = (uint8_t *)dst + (ir * dst_row_size_aligned);

        hvx_step_f16_aa((uint8_t *) dst_local, (અચલ uint8_t *) src_local, ne0);
    }
}

સ્થિર શૂન્ય l2_norm_f16(અચલ શૂન્ય * restrict src,
                        શૂન્ય * restrict dst,
                        અચલ uint32_t અંક_rows,
                        અચલ સંરચના htp_unary_context * uctx) {
    htp_unary_op_preamble;
    દશાંશ epsilon = 0.f;
    memcpy(&epsilon, op_params, કદ(દશાંશ));

    માટે (uint32_t ir = 0; ir < અંક_rows; ir++) {
        અચલ uint8_t * restrict src_f = (અચલ uint8_t *)src + (ir * src0_row_size_aligned);
        uint8_t * restrict dst_f       = (uint8_t *)dst + (ir * dst_row_size_aligned);

        hvx_fast_l2_norm_f16((અચલ uint8_t *)src_f, (uint8_t *)dst_f, ne0, epsilon);
    }
}

સ્થિર શૂન્ય neg_f32(અચલ શૂન્ય * restrict src,
                    શૂન્ય * restrict dst,
                    અચલ uint32_t અંક_rows,
                    અચલ સંરચના htp_unary_context * uctx) {
    htp_unary_op_preamble;

    માટે (uint32_t ir = 0; ir < અંક_rows; ir++) {
        અચલ uint8_t * restrict src_local = (અચલ uint8_t *)src + (ir * src0_row_size_aligned);
        uint8_t * restrict dst_local       = (uint8_t *)dst + (ir * dst_row_size_aligned);

        hvx_scale_f32_aa(dst_local, src_local, ne0, -1.0f);
    }
}

સ્થિર શૂન્ય exp_f32(અચલ શૂન્ય * restrict src,
                    શૂન્ય * restrict dst,
                    અચલ uint32_t અંક_rows,
                    અચલ સંરચના htp_unary_context * uctx) {
    htp_unary_op_preamble;

    માટે (uint32_t ir = 0; ir < અંક_rows; ir++) {
        અચલ uint8_t * restrict src_local = (અચલ uint8_t *)src + (ir * src0_row_size_aligned);
        uint8_t * restrict dst_local       = (uint8_t *)dst + (ir * dst_row_size_aligned);

        hvx_exp_f32(dst_local, src_local, ne0, false);
    }
}

સ્થિર શૂન્ય sigmoid_f32(અચલ શૂન્ય * restrict src,
                        શૂન્ય * restrict dst,
                        અચલ uint32_t અંક_rows,
                        અચલ સંરચના htp_unary_context * uctx) {
    htp_unary_op_preamble;

    માટે (uint32_t ir = 0; ir < અંક_rows; ir++) {
        અચલ uint8_t * restrict src_local = (અચલ uint8_t *)src + (ir * src0_row_size_aligned);
        uint8_t * restrict dst_local       = (uint8_t *)dst + (ir * dst_row_size_aligned);

        hvx_sigmoid_f32_aa(dst_local, src_local, ne0);
    }
}

// silu(x) = x * sigmoid(x)
સ્થિર શૂન્ય silu_f32(અચલ શૂન્ય * restrict src,
                     શૂન્ય * restrict dst,
                     અચલ uint32_t અંક_rows,
                     અચલ સંરચના htp_unary_context * uctx) {
    htp_unary_op_preamble;

    માટે (uint32_t ir = 0; ir < અંક_rows; ir++) {
        અચલ uint8_t * restrict src_local = (અચલ uint8_t *)src + (ir * src0_row_size_aligned);
        uint8_t * restrict dst_local       = (uint8_t *)dst + (ir * dst_row_size_aligned);

        hvx_sigmoid_f32_aa(dst_local, src_local, ne0);
        hvx_mul_f32_aaa(dst_local, src_local, dst_local, ne0);
    }
}

// GELU uses the tanh approximation.
સ્થિર __attribute__((noinline)) HVX_Vector hvx_vec_gelu_f32(HVX_Vector x) {
    અચલ HVX_Vector half = hvx_vec_splat_f32(0.5f);
    અચલ HVX_Vector one  = hvx_vec_splat_f32(1.0f);

    HVX_Vector inner = hvx_vec_mul_f32_f32(x, x);
    inner = hvx_vec_mul_f32_f32(inner, hvx_vec_splat_f32(0.044715f));
    inner = hvx_vec_add_f32_f32(inner, one);
    inner = hvx_vec_mul_f32_f32(inner, x);
    inner = hvx_vec_mul_f32_f32(inner, hvx_vec_splat_f32(0.7978845608028654f));

    પાછા hvx_vec_mul_f32_f32(hvx_vec_mul_f32_f32(half, x),
                               hvx_vec_add_f32_f32(one, hvx_vec_tanh_f32(inner)));
}

સ્થિર અંતઃરેખ શૂન્ય hvx_gelu_f32_aa(uint8_t * restrict dst, અચલ uint8_t * restrict src, uint32_t n) {
    પ્રતિજ્ઞા((unsigned long) dst % 128 == 0);
    પ્રતિજ્ઞા((unsigned long) src % 128 == 0);

    HVX_Vector * restrict vdst = (HVX_Vector *) dst;
    HVX_Vector * restrict vsrc = (HVX_Vector *) src;

    અચલ uint32_t nvec = n / VLEN_FP32;
    અચલ uint32_t nloe = n % VLEN_FP32;

    uint32_t i = 0;
    _Pragma("unroll(4)")
    માટે (; i < nvec; i++) {
        vdst[i] = hvx_vec_gelu_f32(vsrc[i]);
    }
    જો (nloe) {
        hvx_vec_store_a(&vdst[i], nloe * કદ(દશાંશ), hvx_vec_gelu_f32(vsrc[i]));
    }
}

// GELU_QUICK uses x * sigmoid(1.702 * x).
સ્થિર __attribute__((noinline)) HVX_Vector hvx_vec_gelu_quick_f32(HVX_Vector x) {
    અચલ HVX_Vector one     = hvx_vec_splat_f32(1.0f);
    અચલ HVX_Vector max_exp = hvx_vec_splat_f32(87.0f);
    અચલ HVX_Vector min_exp = hvx_vec_splat_f32(-87.0f);
    અચલ HVX_Vector scaled  = hvx_vec_mul_f32_f32(x, hvx_vec_splat_f32(1.702f));
    અચલ HVX_Vector sigmoid = hvx_vec_fast_sigmoid_f32_guard(scaled, one, max_exp, min_exp);

    પાછા hvx_vec_mul_f32_f32(x, sigmoid);
}

સ્થિર અંતઃરેખ શૂન્ય hvx_gelu_quick_f32_aa(uint8_t * restrict dst, અચલ uint8_t * restrict src, uint32_t n) {
    પ્રતિજ્ઞા((unsigned long) dst % 128 == 0);
    પ્રતિજ્ઞા((unsigned long) src % 128 == 0);

    HVX_Vector * restrict vdst = (HVX_Vector *) dst;
    HVX_Vector * restrict vsrc = (HVX_Vector *) src;

    અચલ uint32_t nvec = n / VLEN_FP32;
    અચલ uint32_t nloe = n % VLEN_FP32;

    uint32_t i = 0;
    _Pragma("unroll(4)")
    માટે (; i < nvec; i++) {
        vdst[i] = hvx_vec_gelu_quick_f32(vsrc[i]);
    }
    જો (nloe) {
        hvx_vec_store_a(&vdst[i], nloe * કદ(દશાંશ), hvx_vec_gelu_quick_f32(vsrc[i]));
    }
}

સ્થિર શૂન્ય gelu_f32(અચલ શૂન્ય * restrict src,
                     શૂન્ય * restrict dst,
                     અચલ uint32_t અંક_rows,
                     અચલ સંરચના htp_unary_context * uctx) {
    htp_unary_op_preamble;

    માટે (uint32_t ir = 0; ir < અંક_rows; ir++) {
        અચલ uint8_t * restrict src_local = (અચલ uint8_t *)src + (ir * src0_row_size_aligned);
        uint8_t * restrict dst_local       = (uint8_t *)dst + (ir * dst_row_size_aligned);

        hvx_gelu_f32_aa(dst_local, src_local, ne0);
    }
}

સ્થિર શૂન્ય gelu_quick_f32(અચલ શૂન્ય * restrict src,
                           શૂન્ય * restrict dst,
                           અચલ uint32_t અંક_rows,
                           અચલ સંરચના htp_unary_context * uctx) {
    htp_unary_op_preamble;

    માટે (uint32_t ir = 0; ir < અંક_rows; ir++) {
        અચલ uint8_t * restrict src_local = (અચલ uint8_t *)src + (ir * src0_row_size_aligned);
        uint8_t * restrict dst_local       = (uint8_t *)dst + (ir * dst_row_size_aligned);

        hvx_gelu_quick_f32_aa(dst_local, src_local, ne0);
    }
}

સ્થિર શૂન્ય gelu_erf_f32(અચલ શૂન્ય * restrict src,
                         શૂન્ય * restrict dst,
                         અચલ uint32_t અંક_rows,
                         અચલ સંરચના htp_unary_context * uctx) {
    htp_unary_op_preamble;

    માટે (uint32_t ir = 0; ir < અંક_rows; ir++) {
        અચલ uint8_t * restrict src_local = (અચલ uint8_t *) src + (ir * src0_row_size_aligned);
        uint8_t * restrict dst_local       = (uint8_t *) dst + (ir * dst_row_size_aligned);

        hvx_gelu_erf_f32_aa(dst_local, src_local, ne0);
    }
}

સ્થિર શૂન્ય tri_f32(અચલ શૂન્ય * restrict src,
                    શૂન્ય * restrict dst,
                    અચલ uint32_t અંક_rows,
                    અચલ uint32_t ir,
                    અચલ સંરચના htp_unary_context * uctx) {
    htp_unary_op_preamble;
    અચલ int32_t ttype = op_params[0];
    અચલ HVX_Vector zero = hvx_vec_splat_f32(0.0f);
    અચલ uint32_t nvec  = ne0 / VLEN_FP32;
    અચલ uint32_t nloe  = ne0 % VLEN_FP32;

    અચલ uint32_t ne01 = uctx->octx->src[0]->ne[1];

    માટે (uint32_t b = 0; b < અંક_rows; b++) {
        અચલ uint32_t abs_row = ir + b;
        અચલ uint32_t i01     = abs_row % ne01;

        અચલ HVX_Vector * restrict v_src = (અચલ HVX_Vector *) ((અચલ uint8_t *) src + b * src0_row_size_aligned);
        HVX_Vector * restrict v_dst       = (HVX_Vector *) ((uint8_t *) dst + b * dst_row_size_aligned);

        uint32_t boundary;
        પૂર્ણાંક      keep_ડાબું;
        પસંદગી (ttype) {
            સ્થિતિ 0: boundary = i01;     keep_ડાબું = 0; તોડો;  // keep col >= row
            સ્થિતિ 1: boundary = i01 + 1; keep_ડાબું = 0; તોડો;  // keep col > row
            સ્થિતિ 2: boundary = i01 + 1; keep_ડાબું = 1; તોડો;  // keep col <= row
            સ્થિતિ 3: boundary = i01;     keep_ડાબું = 1; તોડો;  // keep col < row
            મૂળભૂત: boundary = 0; keep_ડાબું = 0; તોડો;
        }
        જો (boundary > ne0) boundary = ne0;

        // Full HVX vectors - each starts at a 128-byte aligned offset
        માટે (uint32_t i = 0; i < nvec; i++) {
            અચલ uint32_t vec_start = i * VLEN_FP32;
            અચલ uint32_t vec_end   = vec_start + VLEN_FP32;
            જો (keep_ડાબું) {
                જો (vec_end <= boundary) {
                    v_dst[i] = v_src[i];
                } નહીંતર જો (vec_start >= boundary) {
                    v_dst[i] = zero;
                } નહીંતર {
                    HVX_VectorPred mask = Q6_Q_vsetq_R((boundary - vec_start) * કદ(દશાંશ));
                    v_dst[i]            = Q6_V_vmux_QVV(mask, v_src[i], zero);
                }
            } નહીંતર {
                જો (vec_end <= boundary) {
                    v_dst[i] = zero;
                } નહીંતર જો (vec_start >= boundary) {
                    v_dst[i] = v_src[i];
                } નહીંતર {
                    HVX_VectorPred mask = Q6_Q_vsetq_R((boundary - vec_start) * કદ(દશાંશ));
                    v_dst[i]            = Q6_V_vmux_QVV(mask, zero, v_src[i]);
                }
            }
        }

        // Tail elements (row_elems not a multiple of VLEN_FP32)
        જો (nloe > 0) {
            અચલ uint32_t abs_start = nvec * VLEN_FP32;
            અચલ uint32_t abs_end   = abs_start + nloe;
            HVX_Vector     tail_val;
            જો (keep_ડાબું) {
                જો (abs_end <= boundary) {
                    tail_val = v_src[nvec];
                } નહીંતર જો (abs_start >= boundary) {
                    tail_val = zero;
                } નહીંતર {
                    HVX_VectorPred mask = Q6_Q_vsetq_R((boundary - abs_start) * કદ(દશાંશ));
                    tail_val            = Q6_V_vmux_QVV(mask, v_src[nvec], zero);
                }
            } નહીંતર {
                જો (abs_end <= boundary) {
                    tail_val = zero;
                } નહીંતર જો (abs_start >= boundary) {
                    tail_val = v_src[nvec];
                } નહીંતર {
                    HVX_VectorPred mask = Q6_Q_vsetq_R((boundary - abs_start) * કદ(દશાંશ));
                    tail_val            = Q6_V_vmux_QVV(mask, zero, v_src[nvec]);
                }
            }
            hvx_vec_store_a(&v_dst[nvec], nloe * કદ(દશાંશ), tail_val);
        }
    }
}

// softplus(x) = log(1 + exp(x))
// Match CPU reference: return x for x > 20 to avoid overflow.
સ્થિર અંતઃરેખ શૂન્ય hvx_softplus_f32_aa(uint8_t * restrict dst, અચલ uint8_t * restrict src, uint32_t n) {
    અચલ HVX_Vector * restrict v_src = (અચલ HVX_Vector *) src;
    HVX_Vector * restrict v_dst       = (HVX_Vector *) dst;

    અચલ HVX_Vector twenty = hvx_vec_splat_f32(20.0f);
    અચલ HVX_Vector one    = hvx_vec_splat_f32(1.0f);

    અચલ uint32_t nvec = n / VLEN_FP32;
    અચલ uint32_t nloe = n % VLEN_FP32;

    #pragma unroll(4)
    માટે (uint32_t i = 0; i < nvec; i++) {
        HVX_Vector x              = v_src[i];
        HVX_VectorPred pred_gt20  = Q6_Q_vcmp_gt_VsfVsf(x, twenty);
        HVX_Vector expx           = hvx_vec_exp_f32(x);
        HVX_Vector logv           = hvx_vec_log_f32(hvx_vec_add_f32_f32(one, expx));
        v_dst[i]                  = Q6_V_vmux_QVV(pred_gt20, x, logv);
    }

    જો (nloe > 0) {
        HVX_Vector x             = v_src[nvec];
        HVX_VectorPred pred_gt20 = Q6_Q_vcmp_gt_VsfVsf(x, twenty);
        HVX_Vector expx          = hvx_vec_exp_f32(x);
        HVX_Vector logv          = hvx_vec_log_f32(hvx_vec_add_f32_f32(one, expx));
        HVX_Vector sp            = Q6_V_vmux_QVV(pred_gt20, x, logv);
        hvx_vec_store_a((શૂન્ય *) &v_dst[nvec], nloe * કદ(દશાંશ), sp);
    }
}

સ્થિર શૂન્ય softplus_f32(અચલ શૂન્ય * restrict src,
                         શૂન્ય * restrict dst,
                         અચલ uint32_t અંક_rows,
                         અચલ સંરચના htp_unary_context * uctx) {
    htp_unary_op_preamble;

    માટે (uint32_t ir = 0; ir < અંક_rows; ir++) {
        અચલ uint8_t * restrict src_local = (અચલ uint8_t *)src + (ir * src0_row_size_aligned);
        uint8_t * restrict dst_local       = (uint8_t *)dst + (ir * dst_row_size_aligned);

        hvx_softplus_f32_aa(dst_local, src_local, ne0);
    }
}

સ્થિર શૂન્ય l2_norm_f32(અચલ શૂન્ય * restrict src,
                        શૂન્ય * restrict dst,
                        અચલ uint32_t અંક_rows,
                        અચલ સંરચના htp_unary_context * uctx) {
    htp_unary_op_preamble;
    દશાંશ epsilon = 0.f;
    memcpy(&epsilon, op_params, કદ(દશાંશ));

    માટે (uint32_t ir = 0; ir < અંક_rows; ir++) {
        અચલ uint8_t * restrict src_f = (અચલ uint8_t *)src + (ir * src0_row_size_aligned);
        uint8_t * restrict dst_f       = (uint8_t *)dst + (ir * dst_row_size_aligned);

        hvx_fast_l2_norm_f32((અચલ uint8_t *)src_f, (uint8_t *)dst_f, ne0, epsilon);
    }
}

સ્થિર અંતઃરેખ HVX_VectorPair hvx_fwht_stage_pair(HVX_Vector v0, HVX_Vector v1, int32_t b) {
    HVX_VectorPair p = Q6_W_vshuff_VVR(v1, v0, b);
    HVX_Vector s = Q6_Vqf32_vadd_Vqf32Vqf32(Q6_V_lo_W(p), Q6_V_hi_W(p));
    HVX_Vector d = Q6_Vqf32_vsub_Vqf32Vqf32(Q6_V_lo_W(p), Q6_V_hi_W(p));
    પાછા Q6_W_vshuff_VVR(d, s, b);
}

સ્થિર અંતઃરેખ HVX_VectorPair hvx_fwht32_pair(HVX_Vector v0_sf, HVX_Vector v1_sf, HVX_Vector v_scale) {
    HVX_Vector v0 = Q6_Vqf32_vmpy_VsfVsf(v0_sf, v_scale);
    HVX_Vector v1 = Q6_Vqf32_vmpy_VsfVsf(v1_sf, v_scale);
    HVX_VectorPair p;
    p = hvx_fwht_stage_pair(v0, v1, 4);
    p = hvx_fwht_stage_pair(Q6_V_lo_W(p), Q6_V_hi_W(p), 8);
    p = hvx_fwht_stage_pair(Q6_V_lo_W(p), Q6_V_hi_W(p), 16);
    p = hvx_fwht_stage_pair(Q6_V_lo_W(p), Q6_V_hi_W(p), 32);
    p = hvx_fwht_stage_pair(Q6_V_lo_W(p), Q6_V_hi_W(p), 64);
    પાછા p;
}

સ્થિર અંતઃરેખ શૂન્ય hvx_fwht_store32(HVX_Vector * restrict v_dst, HVX_VectorPair p) {
    v_dst[0] = Q6_Vsf_equals_Vqf32(Q6_V_lo_W(p));
}

સ્થિર અંતઃરેખ શૂન્ય hvx_fwht_store64(HVX_Vector * restrict v_dst, HVX_VectorPair p) {
    HVX_Vector u = Q6_V_lo_W(p);
    HVX_Vector v = Q6_V_hi_W(p);
    v_dst[0] = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vqf32(u, v));
    v_dst[1] = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vsub_Vqf32Vqf32(u, v));
}

સ્થિર અંતઃરેખ શૂન્ય hvx_fwht_tail_f32(HVX_Vector * restrict v_dst, uint32_t nvec) {
    uint32_t len_vec = 2;
    માટે (; len_vec * 2 < nvec; len_vec <<= 2) {
        અચલ uint32_t step = len_vec * 4;
        માટે (uint32_t i = 0; i < nvec; i += step) {
            માટે (uint32_t j = 0; j < len_vec; j++) {
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

    જો (len_vec < nvec) {
        માટે (uint32_t i = 0; i < nvec; i += 2 * len_vec) {
            માટે (uint32_t j = 0; j < len_vec; j++) {
                HVX_Vector u = v_dst[i + j];
                HVX_Vector v = v_dst[i + len_vec + j];
                v_dst[i + j]           = hvx_vec_add_f32_f32(u, v);
                v_dst[i + len_vec + j] = hvx_vec_sub_f32_f32(u, v);
            }
        }
    }
}

સ્થિર અંતઃરેખ શૂન્ય hvx_fwht_row_f32(અચલ uint8_t * restrict src, uint8_t * restrict dst, uint32_t n, HVX_Vector v_scale) {
    અચલ HVX_Vector * restrict v_src = (અચલ HVX_Vector *) src;
    HVX_Vector * restrict v_dst       = (HVX_Vector *) dst;
    અચલ uint32_t nvec               = n / VLEN_FP32;

    જો (nvec == 1) {
        HVX_VectorPair p = hvx_fwht32_pair(v_src[0], v_src[0], v_scale);
        hvx_fwht_store32(v_dst, p);
        પાછા;
    }

    માટે (uint32_t i = 0; i < nvec; i += 2) {
        HVX_VectorPair p = hvx_fwht32_pair(v_src[i], v_src[i + 1], v_scale);
        hvx_fwht_store64(&v_dst[i], p);
    }

    hvx_fwht_tail_f32(v_dst, nvec);
}

સ્થિર અંતઃરેખ શૂન્ય hvx_fwht_row_f16(અચલ uint8_t * restrict src, uint8_t * restrict dst, uint32_t n, HVX_Vector v_scale) {
    અચલ HVX_Vector * restrict v_src = (અચલ HVX_Vector *) src;
    HVX_Vector * restrict v_dst       = (HVX_Vector *) dst;
    અચલ uint32_t nvec               = n / VLEN_FP32;

    જો (nvec == 1) {
        HVX_Vector v0 = Q6_V_lo_W(hvx_vec_f16_to_f32(v_src[0]));
        HVX_VectorPair p = hvx_fwht32_pair(v0, v0, v_scale);
        hvx_fwht_store32(v_dst, p);
        પાછા;
    }

    અચલ uint32_t nvec_f16 = n / VLEN_FP16;
    માટે (uint32_t i = 0; i < nvec_f16; i++) {
        HVX_VectorPair f32p = hvx_vec_f16_to_f32(v_src[i]);
        HVX_VectorPair p    = hvx_fwht32_pair(Q6_V_lo_W(f32p), Q6_V_hi_W(f32p), v_scale);
        hvx_fwht_store64(&v_dst[2 * i], p);
    }

    hvx_fwht_tail_f32(v_dst, nvec);
}

સ્થિર શૂન્ય hadamard_f32(અચલ શૂન્ય * restrict src,
                         શૂન્ય * restrict dst,
                         અચલ uint32_t અંક_rows,
                         અચલ સંરચના htp_unary_context * uctx) {
    htp_unary_op_preamble;
    અચલ HVX_Vector v_scale = hvx_vec_splat_f32(1.0f / sqrtf((દશાંશ) ne0));

    માટે (uint32_t ir = 0; ir < અંક_rows; ir++) {
        અચલ uint8_t * restrict src_row = (અચલ uint8_t *) src + (ir * src0_row_size_aligned);
        uint8_t * restrict dst_row       = (uint8_t *) dst + (ir * dst_row_size_aligned);

        hvx_fwht_row_f32(src_row, dst_row, ne0, v_scale);
    }
}

સ્થિર શૂન્ય hadamard_f16(અચલ શૂન્ય * restrict src,
                         શૂન્ય * restrict dst,
                         અચલ uint32_t અંક_rows,
                         અચલ સંરચના htp_unary_context * uctx) {
    htp_unary_op_preamble;
    અચલ HVX_Vector v_scale = hvx_vec_splat_f32(1.0f / sqrtf((દશાંશ) ne0));

    માટે (uint32_t ir = 0; ir < અંક_rows; ir++) {
        અચલ uint8_t * restrict src_row = (અચલ uint8_t *) src + (ir * src0_row_size_aligned);
        uint8_t * restrict dst_row       = (uint8_t *) dst + (ir * dst_row_size_aligned);

        hvx_fwht_row_f16(src_row, dst_row, ne0, v_scale);
    }
}

સ્થિર શૂન્ય tanh_f32(અચલ શૂન્ય * restrict src,
                     શૂન્ય * restrict dst,
                     અચલ uint32_t અંક_rows,
                     અચલ સંરચના htp_unary_context * uctx) {
    htp_unary_op_preamble;

    માટે (uint32_t ir = 0; ir < અંક_rows; ir++) {
        અચલ uint8_t * restrict src_local = (અચલ uint8_t *)src + (ir * src0_row_size_aligned);
        uint8_t * restrict dst_local       = (uint8_t *)dst + (ir * dst_row_size_aligned);

        hvx_tanh_f32_aa(dst_local, src_local, ne0);
    }
}

સ્થિર શૂન્ય abs_f32(અચલ શૂન્ય * restrict src,
                    શૂન્ય * restrict dst,
                    અચલ uint32_t અંક_rows,
                    અચલ સંરચના htp_unary_context * uctx) {
    htp_unary_op_preamble;

    માટે (uint32_t ir = 0; ir < અંક_rows; ir++) {
        અચલ uint8_t * restrict src_local = (અચલ uint8_t *)src + (ir * src0_row_size_aligned);
        uint8_t * restrict dst_local       = (uint8_t *)dst + (ir * dst_row_size_aligned);

        hvx_abs_f32_aa(dst_local, src_local, ne0);
    }
}

સ્થિર શૂન્ય relu_f32(અચલ શૂન્ય * restrict src,
                     શૂન્ય * restrict dst,
                     અચલ uint32_t અંક_rows,
                     અચલ સંરચના htp_unary_context * uctx) {
    htp_unary_op_preamble;

    માટે (uint32_t ir = 0; ir < અંક_rows; ir++) {
        અચલ uint8_t * restrict src_local = (અચલ uint8_t *)src + (ir * src0_row_size_aligned);
        uint8_t * restrict dst_local       = (uint8_t *)dst + (ir * dst_row_size_aligned);

        hvx_max_scalar_f32(dst_local, src_local, 0.0f, ne0);
    }
}

સ્થિર શૂન્ય step_f32(અચલ શૂન્ય * restrict src,
                     શૂન્ય * restrict dst,
                     અચલ uint32_t અંક_rows,
                     અચલ સંરચના htp_unary_context * uctx) {
    htp_unary_op_preamble;

    માટે (uint32_t ir = 0; ir < અંક_rows; ir++) {
        અચલ uint8_t * restrict src_local = (અચલ uint8_t *)src + (ir * src0_row_size_aligned);
        uint8_t * restrict dst_local       = (uint8_t *)dst + (ir * dst_row_size_aligned);

        hvx_step_f32_aa(dst_local, src_local, ne0);
    }
}

સ્થિર શૂન્ય log_f32(અચલ શૂન્ય * restrict src,
                    શૂન્ય * restrict dst,
                    અચલ uint32_t અંક_rows,
                    અચલ સંરચના htp_unary_context * uctx) {
    htp_unary_op_preamble;

    માટે (uint32_t ir = 0; ir < અંક_rows; ir++) {
        અચલ uint8_t * restrict src_local = (અચલ uint8_t *)src + (ir * src0_row_size_aligned);
        uint8_t * restrict dst_local       = (uint8_t *)dst + (ir * dst_row_size_aligned);

        hvx_log_f32_aa(dst_local, src_local, ne0);
    }
}

#// Pointwise unary ops on one column tile in VTCM.
સ્થિર શૂન્ય tile_scale_f32(શૂન્ય * restrict dst, અચલ શૂન્ય * restrict src, uint32_t tw, અચલ સંરચના htp_unary_context * uctx) {
    દશાંશ scale = 0.f;
    દશાંશ bias  = 0.f;
    memcpy(&scale, &uctx->octx->op_params[0], કદ(દશાંશ));
    memcpy(&bias,  &uctx->octx->op_params[1], કદ(દશાંશ));
    hvx_scale_offset_f32_aa((uint8_t *) dst, (અચલ uint8_t *) src, tw, scale, bias);
}

સ્થિર શૂન્ય tile_clamp_f32(શૂન્ય * restrict dst, અચલ શૂન્ય * restrict src, uint32_t tw, અચલ સંરચના htp_unary_context * uctx) {
    દશાંશ min = 0.f;
    દશાંશ max = 0.f;
    memcpy(&min, &uctx->octx->op_params[0], કદ(દશાંશ));
    memcpy(&max, &uctx->octx->op_params[1], કદ(દશાંશ));
    hvx_clamp_scalar_f32((uint8_t *) dst, (અચલ uint8_t *) src, min, max, tw);
}

સ્થિર શૂન્ય tile_leaky_relu_f32(શૂન્ય * restrict dst, અચલ શૂન્ય * restrict src, uint32_t tw, અચલ સંરચના htp_unary_context * uctx) {
    દશાંશ negative_slope = 0.f;
    memcpy(&negative_slope, &uctx->octx->op_params[0], કદ(દશાંશ));
    hvx_leaky_relu_scalar_f32((uint8_t *) dst, (અચલ uint8_t *) src, negative_slope, tw);
}

સ્થિર શૂન્ય tile_sqr_f32(શૂન્ય * restrict dst, અચલ શૂન્ય * restrict src, uint32_t tw, અચલ સંરચના htp_unary_context * uctx) {
    (શૂન્ય) uctx;
    hvx_sqr_f32_aa((uint8_t *) dst, (અચલ uint8_t *) src, tw);
}

સ્થિર શૂન્ય tile_sqrt_f32(શૂન્ય * restrict dst, અચલ શૂન્ય * restrict src, uint32_t tw, અચલ સંરચના htp_unary_context * uctx) {
    (શૂન્ય) uctx;
    hvx_sqrt_f32_aa((uint8_t *) dst, (અચલ uint8_t *) src, tw);
}

સ્થિર શૂન્ય tile_neg_f32(શૂન્ય * restrict dst, અચલ શૂન્ય * restrict src, uint32_t tw, અચલ સંરચના htp_unary_context * uctx) {
    (શૂન્ય) uctx;
    hvx_scale_f32_aa((uint8_t *) dst, (અચલ uint8_t *) src, tw, -1.0f);
}

સ્થિર શૂન્ય tile_exp_f32(શૂન્ય * restrict dst, અચલ શૂન્ય * restrict src, uint32_t tw, અચલ સંરચના htp_unary_context * uctx) {
    (શૂન્ય) uctx;
    hvx_exp_f32((uint8_t *) dst, (અચલ uint8_t *) src, tw, false);
}

સ્થિર શૂન્ય tile_sigmoid_f32(શૂન્ય * restrict dst, અચલ શૂન્ય * restrict src, uint32_t tw, અચલ સંરચના htp_unary_context * uctx) {
    (શૂન્ય) uctx;
    hvx_sigmoid_f32_aa((uint8_t *) dst, (અચલ uint8_t *) src, tw);
}

સ્થિર શૂન્ય tile_silu_f32(શૂન્ય * restrict dst, અચલ શૂન્ય * restrict src, uint32_t tw, અચલ સંરચના htp_unary_context * uctx) {
    (શૂન્ય) uctx;
    hvx_sigmoid_f32_aa((uint8_t *) dst, (અચલ uint8_t *) src, tw);
    hvx_mul_f32_aaa((uint8_t *) dst, (અચલ uint8_t *) src, (uint8_t *) dst, tw);
}

સ્થિર શૂન્ય tile_gelu_f32(શૂન્ય * restrict dst, અચલ શૂન્ય * restrict src, uint32_t tw, અચલ સંરચના htp_unary_context * uctx) {
    (શૂન્ય) uctx;
    hvx_gelu_f32_aa((uint8_t *) dst, (અચલ uint8_t *) src, tw);
}

સ્થિર શૂન્ય tile_gelu_quick_f32(શૂન્ય * restrict dst, અચલ શૂન્ય * restrict src, uint32_t tw, અચલ સંરચના htp_unary_context * uctx) {
    (શૂન્ય) uctx;
    hvx_gelu_quick_f32_aa((uint8_t *) dst, (અચલ uint8_t *) src, tw);
}

સ્થિર શૂન્ય tile_gelu_erf_f32(શૂન્ય * restrict dst, અચલ શૂન્ય * restrict src, uint32_t tw, અચલ સંરચના htp_unary_context * uctx) {
    (શૂન્ય) uctx;
    hvx_gelu_erf_f32_aa((uint8_t *) dst, (અચલ uint8_t *) src, tw);
}

સ્થિર શૂન્ય tile_softplus_f32(શૂન્ય * restrict dst, અચલ શૂન્ય * restrict src, uint32_t tw, અચલ સંરચના htp_unary_context * uctx) {
    (શૂન્ય) uctx;
    hvx_softplus_f32_aa((uint8_t *) dst, (અચલ uint8_t *) src, tw);
}

સ્થિર શૂન્ય tile_tanh_f32(શૂન્ય * restrict dst, અચલ શૂન્ય * restrict src, uint32_t tw, અચલ સંરચના htp_unary_context * uctx) {
    (શૂન્ય) uctx;
    hvx_tanh_f32_aa((uint8_t *) dst, (અચલ uint8_t *) src, tw);
}

સ્થિર શૂન્ય tile_abs_f32(શૂન્ય * restrict dst, અચલ શૂન્ય * restrict src, uint32_t tw, અચલ સંરચના htp_unary_context * uctx) {
    (શૂન્ય) uctx;
    hvx_abs_f32_aa((uint8_t *) dst, (અચલ uint8_t *) src, tw);
}

સ્થિર શૂન્ય tile_log_f32(શૂન્ય * restrict dst, અચલ શૂન્ય * restrict src, uint32_t tw, અચલ સંરચના htp_unary_context * uctx) {
    (શૂન્ય) uctx;
    hvx_log_f32_aa((uint8_t *) dst, (અચલ uint8_t *) src, tw);
}

સ્થિર શૂન્ય tile_relu_f32(શૂન્ય * restrict dst, અચલ શૂન્ય * restrict src, uint32_t tw, અચલ સંરચના htp_unary_context * uctx) {
    (શૂન્ય) uctx;
    hvx_max_scalar_f32((uint8_t *) dst, (અચલ uint8_t *) src, 0.0f, tw);
}

સ્થિર શૂન્ય tile_step_f32(શૂન્ય * restrict dst, અચલ શૂન્ય * restrict src, uint32_t tw, અચલ સંરચના htp_unary_context * uctx) {
    (શૂન્ય) uctx;
    hvx_step_f32_aa((uint8_t *) dst, (અચલ uint8_t *) src, tw);
}

સ્થિર શૂન્ય tri_apply_tile_f32(અચલ શૂન્ય * restrict src, શૂન્ય * restrict dst,
                               uint32_t tile_elems, uint32_t col_start, uint32_t i01,
                               uint32_t ne0, int32_t ttype) {
    અચલ HVX_Vector * restrict v_src = (અચલ HVX_Vector *) src;
    HVX_Vector * restrict v_dst       = (HVX_Vector *) dst;
    અચલ HVX_Vector zero = hvx_vec_splat_f32(0.0f);

    uint32_t boundary;
    પૂર્ણાંક      keep_ડાબું;
    પસંદગી (ttype) {
        સ્થિતિ 0: boundary = i01;     keep_ડાબું = 0; તોડો;
        સ્થિતિ 1: boundary = i01 + 1; keep_ડાબું = 0; તોડો;
        સ્થિતિ 2: boundary = i01 + 1; keep_ડાબું = 1; તોડો;
        સ્થિતિ 3: boundary = i01;     keep_ડાબું = 1; તોડો;
        મૂળભૂત: boundary = 0; keep_ડાબું = 0; તોડો;
    }
    જો (boundary > ne0) boundary = ne0;

    અચલ uint32_t nvec = tile_elems / VLEN_FP32;
    અચલ uint32_t nloe = tile_elems % VLEN_FP32;

    માટે (uint32_t i = 0; i < nvec; i++) {
        અચલ uint32_t abs_start = col_start + i * VLEN_FP32;
        અચલ uint32_t abs_end   = abs_start + VLEN_FP32;
        જો (keep_ડાબું) {
            જો (abs_end <= boundary) {
                v_dst[i] = v_src[i];
            } નહીંતર જો (abs_start >= boundary) {
                v_dst[i] = zero;
            } નહીંતર {
                HVX_VectorPred mask = Q6_Q_vsetq_R((boundary - abs_start) * કદ(દશાંશ));
                v_dst[i]            = Q6_V_vmux_QVV(mask, v_src[i], zero);
            }
        } નહીંતર {
            જો (abs_end <= boundary) {
                v_dst[i] = zero;
            } નહીંતર જો (abs_start >= boundary) {
                v_dst[i] = v_src[i];
            } નહીંતર {
                HVX_VectorPred mask = Q6_Q_vsetq_R((boundary - abs_start) * કદ(દશાંશ));
                v_dst[i]            = Q6_V_vmux_QVV(mask, zero, v_src[i]);
            }
        }
    }

    જો (nloe > 0) {
        અચલ uint32_t abs_start = col_start + nvec * VLEN_FP32;
        અચલ uint32_t abs_end   = abs_start + nloe;
        HVX_Vector     tail_val;
        જો (keep_ડાબું) {
            જો (abs_end <= boundary) {
                tail_val = v_src[nvec];
            } નહીંતર જો (abs_start >= boundary) {
                tail_val = zero;
            } નહીંતર {
                HVX_VectorPred mask = Q6_Q_vsetq_R((boundary - abs_start) * કદ(દશાંશ));
                tail_val            = Q6_V_vmux_QVV(mask, v_src[nvec], zero);
            }
        } નહીંતર {
            જો (abs_end <= boundary) {
                tail_val = zero;
            } નહીંતર જો (abs_start >= boundary) {
                tail_val = v_src[nvec];
            } નહીંતર {
                HVX_VectorPred mask = Q6_Q_vsetq_R((boundary - abs_start) * કદ(દશાંશ));
                tail_val            = Q6_V_vmux_QVV(mask, zero, v_src[nvec]);
            }
        }
        hvx_vec_store_a(&v_dst[nvec], nloe * કદ(દશાંશ), tail_val);
    }
}

// 1. Standard row-block unary task (F32 and F16).
સ્થિર શૂન્ય unary_thread_row_block(unsigned પૂર્ણાંક nth, unsigned પૂર્ણાંક ith, શૂન્ય * data) {
    (શૂન્ય) nth;
    અચલ સંરચના htp_unary_context * uctx = (અચલ સંરચના htp_unary_context *) data;
    સંરચના htp_ops_context * octx = uctx->octx;
    અચલ સંરચના htp_tensor * src = octx->src[0];
    અચલ સંરચના htp_tensor * dst = octx->dst;
    સંરચના htp_thread_trace * tr = &octx->ctx->trace[ith];

    htp_unary_preamble;

    અચલ uint32_t src0_nrows_per_thread = uctx->src0_nrows_per_thread;
    અચલ size_t src0_data_row_size = uctx->src0_data_row_size;
    અચલ size_t dst_data_row_size  = uctx->dst_data_row_size;
    અચલ size_t src0_row_size_aligned = uctx->src0_row_size_aligned;
    અચલ size_t dst_row_size_aligned  = uctx->dst_row_size_aligned;

    અચલ uint32_t src0_nrows = uctx->src0_nrows;
    અચલ uint32_t src0_start_row = uctx->row_start + src0_nrows_per_thread * ith;
    અચલ uint32_t src0_end_row   = MIN(src0_start_row + src0_nrows_per_thread, uctx->row_start + src0_nrows);

    જો (src0_start_row >= src0_end_row) {
        પાછા;
    }

    અચલ dma_addr_t data_src = uctx->data_src0;
    અચલ dma_addr_t data_dst = uctx->data_dst;

    uint8_t * src0_vtcm_data = uctx->vtcm_src0 + (ith * uctx->vtcm_src0_size_per_thread);
    uint8_t * dst_vtcm_data  = uctx->vtcm_dst + (ith * uctx->vtcm_dst_size_per_thread);

    અચલ size_t src0_vtcm_half_size = uctx->src0_vtcm_half_size;
    અચલ size_t dst_vtcm_half_size  = uctx->dst_vtcm_half_size;

    અચલ bool src0_contig = (nb02 == (size_t)ne01 * nb01) &&
                             (nb03 == (size_t)ne02 * nb02);
    અચલ bool dst_contig  = (nb2  == (size_t)ne1  * nb1)  &&
                             (nb3  == (size_t)ne2  * nb2);

    અચલ સંરચના fastdiv_કિંમતો * div_ne01  = &uctx->kparams->div_ne01;
    અચલ સંરચના fastdiv_કિંમતો * div_ne02  = &uctx->kparams->div_ne02;
    અચલ સંરચના fastdiv_કિંમતો * div_ne012 = &uctx->kparams->div_ne012;

    અચલ uint32_t src0_max_block = src0_contig ? uctx->block : MIN((uint32_t)uctx->block, ne01);
    અચલ uint32_t dst_max_block  = dst_contig  ? uctx->block : MIN((uint32_t)uctx->block, ne1);
    અચલ uint32_t BLOCK = MIN(src0_max_block, dst_max_block);
    જો (BLOCK == 0) {
        FARF(ERROR, "unary-row-block : current VTCM reservation %zu is too small, needed at least %zu\n",
             uctx->vtcm_src0_size_per_thread, src0_row_size_aligned);
        પાછા;
    }

    dma_queue * dma_q = octx->ctx->dma[ith];

    માટે (uint32_t ir = src0_start_row, vtcm_idx = 0; ir < src0_end_row && vtcm_idx < 2; vtcm_idx++) {
        અચલ uint32_t block_size = unary_block_size(ir, src0_end_row, BLOCK, src0_contig, dst_contig,
                                                     ne01, div_ne01);

        dma_queue_ધકેલો(dma_q,
            dma_make_data(data_dst, dst_vtcm_data + (vtcm_idx * dst_vtcm_half_size)),
            nb1, dst_row_size_aligned, dst_data_row_size, 0);

        અચલ size_t src0_off = src0_contig ? (ir * nb01) :
            unary_row_offset(ir, ne01, ne02, div_ne01, div_ne02, div_ne012, nb01, nb02, nb03);
        dma_queue_ધકેલો(dma_q,
            dma_make_data(src0_vtcm_data + (vtcm_idx * src0_vtcm_half_size), data_src + src0_off),
            src0_row_size_aligned, nb01, src0_data_row_size, block_size);

        ir += block_size;
    }

    unary_compute_fn_t compute = (unary_compute_fn_t) uctx->compute;

    માટે (uint32_t ir = src0_start_row; ir < src0_end_row; ) {
        અચલ uint32_t block_size = unary_block_size(ir, src0_end_row, BLOCK, src0_contig, dst_contig,
                                                     ne01, div_ne01);

        શૂન્ય * dst_vtcm  = (શૂન્ય *) (uintptr_t) dma_queue_કાઢો(dma_q).src;
        શૂન્ય * src0_vtcm = (શૂન્ય *) (uintptr_t) dma_queue_કાઢો(dma_q).dst;

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, ir);
        compute(src0_vtcm, dst_vtcm, block_size, uctx);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ir);

        અચલ size_t dst_off = dst_contig ? (ir * nb1) :
            unary_row_offset(ir, ne1, ne2, div_ne01, div_ne02, div_ne012, nb1, nb2, nb3);
        dma_queue_ધકેલો(dma_q,
            dma_make_data(data_dst + dst_off, dst_vtcm),
            nb1, dst_row_size_aligned, dst_data_row_size, block_size);

        અચલ uint32_t next_ir = ir + block_size;
        જો (next_ir < src0_end_row) {
            અચલ uint32_t next_block_size = unary_block_size(next_ir, src0_end_row, BLOCK, src0_contig,
                                                              dst_contig, ne01, div_ne01);
            અચલ uint32_t pref_ir = next_ir + next_block_size;
            જો (pref_ir < src0_end_row) {
                અચલ uint32_t pref_block_size = unary_block_size(pref_ir, src0_end_row, BLOCK, src0_contig,
                                                                  dst_contig, ne01, div_ne01);
                અચલ size_t src0_pref_off = src0_contig ? (pref_ir * nb01) :
                    unary_row_offset(pref_ir, ne01, ne02, div_ne01, div_ne02, div_ne012, nb01, nb02, nb03);
                dma_queue_ધકેલો(dma_q,
                    dma_make_data(src0_vtcm, data_src + src0_pref_off),
                    src0_row_size_aligned, nb01, src0_data_row_size, pref_block_size);
            }
        }
        ir += block_size;
    }

    dma_queue_flush(dma_q);
}

// 2. RMS_NORM_MUL row-block task with weight buffer.
સ્થિર શૂન્ય unary_thread_rms_norm_mul_f32(unsigned પૂર્ણાંક nth, unsigned પૂર્ણાંક ith, શૂન્ય * data) {
    (શૂન્ય) nth;
    અચલ સંરચના htp_unary_context * uctx = (અચલ સંરચના htp_unary_context *) data;
    સંરચના htp_ops_context * octx = uctx->octx;
    અચલ સંરચના htp_tensor * src = octx->src[0];
    અચલ સંરચના htp_tensor * dst = octx->dst;
    સંરચના htp_thread_trace * tr = &octx->ctx->trace[ith];

    htp_unary_preamble;

    અચલ uint32_t src0_nrows_per_thread = uctx->src0_nrows_per_thread;
    અચલ size_t src0_data_row_size = uctx->src0_data_row_size;
    અચલ size_t dst_data_row_size  = uctx->dst_data_row_size;
    અચલ size_t src0_row_size_aligned = uctx->src0_row_size_aligned;
    અચલ size_t dst_row_size_aligned  = uctx->dst_row_size_aligned;

    અચલ uint32_t src0_nrows = uctx->src0_nrows;
    અચલ uint32_t src0_start_row = uctx->row_start + src0_nrows_per_thread * ith;
    અચલ uint32_t src0_end_row   = MIN(src0_start_row + src0_nrows_per_thread, uctx->row_start + src0_nrows);

    જો (src0_start_row >= src0_end_row) {
        પાછા;
    }

    અચલ dma_addr_t data_src  = uctx->data_src0;
    અચલ dma_addr_t data_src1 = uctx->data_src1;
    અચલ dma_addr_t data_dst  = uctx->data_dst;

    અચલ સંરચના htp_tensor * src1 = octx->src[1];
    અચલ uint32_t nb11 = src1->nb[1];
    અચલ uint32_t nb12 = src1->nb[2];
    અચલ uint32_t nb13 = src1->nb[3];
    અચલ uint32_t nb11_bc = (src1->ne[1] > 1) ? nb11 : 0;
    અચલ uint32_t nb12_bc = (src1->ne[2] > 1) ? nb12 : 0;
    અચલ uint32_t nb13_bc = (src1->ne[3] > 1) ? nb13 : 0;
    અચલ bool src1_contig = ((nb12 == (size_t)ne01 * nb11) && (nb13 == (size_t)ne02 * nb12));

    uint8_t * src0_vtcm_data = uctx->vtcm_src0 + (ith * uctx->vtcm_src0_size_per_thread);
    uint8_t * src1_vtcm_data = uctx->vtcm_src1 ? (uctx->vtcm_src1 + (ith * uctx->vtcm_src1_size_per_thread)) : ખાલી;
    uint8_t * dst_vtcm_data  = uctx->vtcm_dst + (ith * uctx->vtcm_dst_size_per_thread);

    અચલ size_t src0_vtcm_half_size = uctx->src0_vtcm_half_size;
    અચલ size_t src1_vtcm_half_size = uctx->src1_vtcm_half_size;
    અચલ size_t dst_vtcm_half_size  = uctx->dst_vtcm_half_size;

    અચલ bool src0_contig = (nb02 == (size_t)ne01 * nb01) &&
                             (nb03 == (size_t)ne02 * nb02);
    અચલ bool dst_contig  = (nb2  == (size_t)ne1  * nb1)  &&
                             (nb3  == (size_t)ne2  * nb2);

    અચલ સંરચના fastdiv_કિંમતો * div_ne01  = &uctx->kparams->div_ne01;
    અચલ સંરચના fastdiv_કિંમતો * div_ne02  = &uctx->kparams->div_ne02;
    અચલ સંરચના fastdiv_કિંમતો * div_ne012 = &uctx->kparams->div_ne012;

    અચલ bool src1_needs_row_clip = !uctx->broadcast_weight && !src1_contig;
    અચલ bool block_src0_contig = src0_contig && !src1_needs_row_clip;
    અચલ bool block_dst_contig  = dst_contig  && !src1_needs_row_clip;

    અચલ uint32_t src0_max_block = block_src0_contig ? uctx->block : MIN((uint32_t)uctx->block, ne01);
    અચલ uint32_t dst_max_block  = block_dst_contig  ? uctx->block : MIN((uint32_t)uctx->block, ne1);
    અચલ uint32_t BLOCK = MIN(src0_max_block, dst_max_block);
    જો (BLOCK == 0) {
        FARF(ERROR, "unary-rms-norm-mul : current VTCM reservation %zu is too small, needed at least %zu\n",
             uctx->vtcm_src0_size_per_thread, src0_row_size_aligned);
        પાછા;
    }

    dma_queue * dma_q = octx->ctx->dma[ith];

    જો (uctx->broadcast_weight) {
        dma_queue_ધકેલો(dma_q, dma_make_data(src1_vtcm_data, data_src1),
                       uctx->src1_row_size_aligned, 0, uctx->src1_data_row_size, 1);
        dma_queue_flush(dma_q);
    }

    માટે (uint32_t ir = src0_start_row, vtcm_idx = 0; ir < src0_end_row && vtcm_idx < 2; vtcm_idx++) {
        અચલ uint32_t block_size = unary_block_size(ir, src0_end_row, BLOCK, block_src0_contig, block_dst_contig,
                                                     ne01, div_ne01);

        dma_queue_ધકેલો(dma_q,
            dma_make_data(data_dst, dst_vtcm_data + (vtcm_idx * dst_vtcm_half_size)),
            nb1, dst_row_size_aligned, dst_data_row_size, 0);

        અચલ size_t src0_off = src0_contig ? (ir * nb01) :
            unary_row_offset(ir, ne01, ne02, div_ne01, div_ne02, div_ne012, nb01, nb02, nb03);
        dma_queue_ધકેલો(dma_q,
            dma_make_data(src0_vtcm_data + (vtcm_idx * src0_vtcm_half_size), data_src + src0_off),
            src0_row_size_aligned, nb01, src0_data_row_size, block_size);

        જો (!uctx->broadcast_weight) {
            અચલ size_t src1_off = src1_contig ? (ir * nb11) :
                unary_row_offset(ir, ne01, ne02, div_ne01, div_ne02, div_ne012, nb11_bc, nb12_bc, nb13_bc);
            dma_queue_ધકેલો(dma_q,
                dma_make_data(src1_vtcm_data + (vtcm_idx * src1_vtcm_half_size), data_src1 + src1_off),
                uctx->src1_row_size_aligned, nb11, uctx->src1_data_row_size, block_size);
        }

        ir += block_size;
    }

    unary_rms_norm_mul_compute_fn_t compute = (unary_rms_norm_mul_compute_fn_t) uctx->compute;

    માટે (uint32_t ir = src0_start_row; ir < src0_end_row; ) {
        અચલ uint32_t block_size = unary_block_size(ir, src0_end_row, BLOCK, block_src0_contig, block_dst_contig,
                                                     ne01, div_ne01);

        શૂન્ય * dst_vtcm  = (શૂન્ય *) (uintptr_t) dma_queue_કાઢો(dma_q).src;
        શૂન્ય * src0_vtcm = (શૂન્ય *) (uintptr_t) dma_queue_કાઢો(dma_q).dst;
        શૂન્ય * src1_vtcm = ખાલી;
        જો (!uctx->broadcast_weight) {
            src1_vtcm = (શૂન્ય *) (uintptr_t) dma_queue_કાઢો(dma_q).dst;
        }

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, ir);
        અચલ શૂન્ય * w = uctx->broadcast_weight ? (અચલ શૂન્ય *) src1_vtcm_data : src1_vtcm;
        compute(src0_vtcm, w, dst_vtcm, block_size, uctx);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ir);

        અચલ size_t dst_off = dst_contig ? (ir * nb1) :
            unary_row_offset(ir, ne1, ne2, div_ne01, div_ne02, div_ne012, nb1, nb2, nb3);
        dma_queue_ધકેલો(dma_q,
            dma_make_data(data_dst + dst_off, dst_vtcm),
            nb1, dst_row_size_aligned, dst_data_row_size, block_size);

        અચલ uint32_t next_ir = ir + block_size;
        જો (next_ir < src0_end_row) {
            અચલ uint32_t next_block_size = unary_block_size(next_ir, src0_end_row, BLOCK, block_src0_contig,
                                                              block_dst_contig, ne01, div_ne01);
            અચલ uint32_t pref_ir = next_ir + next_block_size;
            જો (pref_ir < src0_end_row) {
                અચલ uint32_t pref_block_size = unary_block_size(pref_ir, src0_end_row, BLOCK, block_src0_contig,
                                                                  block_dst_contig, ne01, div_ne01);
                અચલ size_t src0_pref_off = src0_contig ? (pref_ir * nb01) :
                    unary_row_offset(pref_ir, ne01, ne02, div_ne01, div_ne02, div_ne012, nb01, nb02, nb03);
                dma_queue_ધકેલો(dma_q,
                    dma_make_data(src0_vtcm, data_src + src0_pref_off),
                    src0_row_size_aligned, nb01, src0_data_row_size, pref_block_size);

                જો (!uctx->broadcast_weight) {
                    અચલ size_t src1_pref_off = src1_contig ? (pref_ir * nb11) :
                        unary_row_offset(pref_ir, ne01, ne02, div_ne01, div_ne02, div_ne012, nb11_bc, nb12_bc,
                                          nb13_bc);
                    dma_queue_ધકેલો(dma_q,
                        dma_make_data(src1_vtcm, data_src1 + src1_pref_off),
                        uctx->src1_row_size_aligned, nb11, uctx->src1_data_row_size, pref_block_size);
                }
            }
        }
        ir += block_size;
    }

    dma_queue_flush(dma_q);
}

// 3. TRI row-block task with row index ir.
સ્થિર શૂન્ય unary_thread_tri_f32(unsigned પૂર્ણાંક nth, unsigned પૂર્ણાંક ith, શૂન્ય * data) {
    (શૂન્ય) nth;
    અચલ સંરચના htp_unary_context * uctx = (અચલ સંરચના htp_unary_context *) data;
    સંરચના htp_ops_context * octx = uctx->octx;
    અચલ સંરચના htp_tensor * src = octx->src[0];
    અચલ સંરચના htp_tensor * dst = octx->dst;
    સંરચના htp_thread_trace * tr = &octx->ctx->trace[ith];

    htp_unary_preamble;

    અચલ uint32_t src0_nrows_per_thread = uctx->src0_nrows_per_thread;
    અચલ size_t src0_data_row_size = uctx->src0_data_row_size;
    અચલ size_t dst_data_row_size  = uctx->dst_data_row_size;
    અચલ size_t src0_row_size_aligned = uctx->src0_row_size_aligned;
    અચલ size_t dst_row_size_aligned  = uctx->dst_row_size_aligned;

    અચલ uint32_t src0_nrows = uctx->src0_nrows;
    અચલ uint32_t src0_start_row = uctx->row_start + src0_nrows_per_thread * ith;
    અચલ uint32_t src0_end_row   = MIN(src0_start_row + src0_nrows_per_thread, uctx->row_start + src0_nrows);

    જો (src0_start_row >= src0_end_row) {
        પાછા;
    }

    અચલ dma_addr_t data_src = uctx->data_src0;
    અચલ dma_addr_t data_dst = uctx->data_dst;

    uint8_t * src0_vtcm_data = uctx->vtcm_src0 + (ith * uctx->vtcm_src0_size_per_thread);
    uint8_t * dst_vtcm_data  = uctx->vtcm_dst + (ith * uctx->vtcm_dst_size_per_thread);

    અચલ size_t src0_vtcm_half_size = uctx->src0_vtcm_half_size;
    અચલ size_t dst_vtcm_half_size  = uctx->dst_vtcm_half_size;

    અચલ bool src0_contig = (nb02 == (size_t)ne01 * nb01) &&
                             (nb03 == (size_t)ne02 * nb02);
    અચલ bool dst_contig  = (nb2  == (size_t)ne1  * nb1)  &&
                             (nb3  == (size_t)ne2  * nb2);

    અચલ સંરચના fastdiv_કિંમતો * div_ne01  = &uctx->kparams->div_ne01;
    અચલ સંરચના fastdiv_કિંમતો * div_ne02  = &uctx->kparams->div_ne02;
    અચલ સંરચના fastdiv_કિંમતો * div_ne012 = &uctx->kparams->div_ne012;

    અચલ uint32_t src0_max_block = src0_contig ? uctx->block : MIN((uint32_t)uctx->block, ne01);
    અચલ uint32_t dst_max_block  = dst_contig  ? uctx->block : MIN((uint32_t)uctx->block, ne1);
    અચલ uint32_t BLOCK = MIN(src0_max_block, dst_max_block);
    જો (BLOCK == 0) {
        FARF(ERROR, "unary-tri : current VTCM reservation %zu is too small, needed at least %zu\n",
             uctx->vtcm_src0_size_per_thread, src0_row_size_aligned);
        પાછા;
    }

    dma_queue * dma_q = octx->ctx->dma[ith];

    માટે (uint32_t ir = src0_start_row, vtcm_idx = 0; ir < src0_end_row && vtcm_idx < 2; vtcm_idx++) {
        અચલ uint32_t block_size = unary_block_size(ir, src0_end_row, BLOCK, src0_contig, dst_contig,
                                                     ne01, div_ne01);

        dma_queue_ધકેલો(dma_q,
            dma_make_data(data_dst, dst_vtcm_data + (vtcm_idx * dst_vtcm_half_size)),
            nb1, dst_row_size_aligned, dst_data_row_size, 0);

        અચલ size_t src0_off = src0_contig ? (ir * nb01) :
            unary_row_offset(ir, ne01, ne02, div_ne01, div_ne02, div_ne012, nb01, nb02, nb03);
        dma_queue_ધકેલો(dma_q,
            dma_make_data(src0_vtcm_data + (vtcm_idx * src0_vtcm_half_size), data_src + src0_off),
            src0_row_size_aligned, nb01, src0_data_row_size, block_size);

        ir += block_size;
    }

    unary_tri_compute_fn_t compute = (unary_tri_compute_fn_t) uctx->compute;

    માટે (uint32_t ir = src0_start_row; ir < src0_end_row; ) {
        અચલ uint32_t block_size = unary_block_size(ir, src0_end_row, BLOCK, src0_contig, dst_contig,
                                                     ne01, div_ne01);

        શૂન્ય * dst_vtcm  = (શૂન્ય *) (uintptr_t) dma_queue_કાઢો(dma_q).src;
        શૂન્ય * src0_vtcm = (શૂન્ય *) (uintptr_t) dma_queue_કાઢો(dma_q).dst;

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, ir);
        compute(src0_vtcm, dst_vtcm, block_size, ir, uctx);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ir);

        અચલ size_t dst_off = dst_contig ? (ir * nb1) :
            unary_row_offset(ir, ne1, ne2, div_ne01, div_ne02, div_ne012, nb1, nb2, nb3);
        dma_queue_ધકેલો(dma_q,
            dma_make_data(data_dst + dst_off, dst_vtcm),
            nb1, dst_row_size_aligned, dst_data_row_size, block_size);

        અચલ uint32_t next_ir = ir + block_size;
        જો (next_ir < src0_end_row) {
            અચલ uint32_t next_block_size = unary_block_size(next_ir, src0_end_row, BLOCK, src0_contig,
                                                              dst_contig, ne01, div_ne01);
            અચલ uint32_t pref_ir = next_ir + next_block_size;
            જો (pref_ir < src0_end_row) {
                અચલ uint32_t pref_block_size = unary_block_size(pref_ir, src0_end_row, BLOCK, src0_contig,
                                                                  dst_contig, ne01, div_ne01);
                અચલ size_t src0_pref_off = src0_contig ? (pref_ir * nb01) :
                    unary_row_offset(pref_ir, ne01, ne02, div_ne01, div_ne02, div_ne012, nb01, nb02, nb03);
                dma_queue_ધકેલો(dma_q,
                    dma_make_data(src0_vtcm, data_src + src0_pref_off),
                    src0_row_size_aligned, nb01, src0_data_row_size, pref_block_size);
            }
        }
        ir += block_size;
    }

    dma_queue_flush(dma_q);
}

// 4. Pointwise tiled unary task.
સ્થિર શૂન્ય unary_thread_tiled(unsigned પૂર્ણાંક nth, unsigned પૂર્ણાંક ith, શૂન્ય * data) {
    (શૂન્ય) nth;
    અચલ સંરચના htp_unary_context * uctx = (અચલ સંરચના htp_unary_context *) data;
    સંરચના htp_ops_context * octx = uctx->octx;
    અચલ સંરચના htp_tensor * src = octx->src[0];
    અચલ સંરચના htp_tensor * dst = octx->dst;
    સંરચના htp_thread_trace * tr = &octx->ctx->trace[ith];

    htp_unary_preamble;

    અચલ uint32_t src0_nrows_per_thread = uctx->src0_nrows_per_thread;
    અચલ uint32_t col_tile              = uctx->col_tile;

    અચલ uint32_t src0_nrows     = uctx->src0_nrows;
    અચલ uint32_t src0_start_row = uctx->row_start + src0_nrows_per_thread * ith;
    અચલ uint32_t src0_end_row   = MIN(src0_start_row + src0_nrows_per_thread, uctx->row_start + src0_nrows);

    જો (src0_start_row >= src0_end_row) {
        પાછા;
    }

    અચલ dma_addr_t data_src = uctx->data_src0;
    અચલ dma_addr_t data_dst = uctx->data_dst;

    uint8_t * src0_vtcm_data = uctx->vtcm_src0 + (ith * uctx->vtcm_src0_size_per_thread);
    uint8_t * dst_vtcm_data  = uctx->vtcm_dst + (ith * uctx->vtcm_dst_size_per_thread);

    અચલ size_t src0_half = uctx->src0_vtcm_half_size;
    અચલ size_t dst_half  = uctx->dst_vtcm_half_size;

    dma_queue * dma_q = octx->ctx->dma[ith];

    અચલ સંરચના fastdiv_કિંમતો * div_ne01  = &uctx->kparams->div_ne01;
    અચલ સંરચના fastdiv_કિંમતો * div_ne02  = &uctx->kparams->div_ne02;
    અચલ સંરચના fastdiv_કિંમતો * div_ne012 = &uctx->kparams->div_ne012;
    અચલ સંરચના fastdiv_કિંમતો * div_tpr   = &uctx->kparams->div_tpr;

    અચલ uint32_t tiles_per_row = (ne0 + col_tile - 1) / col_tile;

    અચલ bool src0_contig = (nb02 == (size_t)ne01 * nb01) &&
                             (nb03 == (size_t)ne02 * nb02);
    અચલ bool dst_contig  = (nb2  == (size_t)ne1  * nb1)  &&
                             (nb3  == (size_t)ne2  * nb2);

    અચલ uint32_t total_tiles = (src0_end_row - src0_start_row) * tiles_per_row;

    માટે (uint32_t t = 0, vtcm_idx = 0; t < total_tiles && vtcm_idx < 2; t++, vtcm_idx++) {
        અચલ uint32_t row  = src0_start_row + t / tiles_per_row;
        અચલ uint32_t col  = (t % tiles_per_row) * col_tile;
        અચલ uint32_t tw   = MIN(col_tile, ne0 - col);
        અચલ size_t   tb   = (size_t) tw * કદ(દશાંશ);
        અચલ size_t   soff = (src0_contig ? (row * nb01) :
                               unary_row_offset(row, ne01, ne02, div_ne01, div_ne02, div_ne012, nb01, nb02, nb03)) +
                               (size_t) col * કદ(દશાંશ);

        dma_queue_ધકેલો(dma_q, dma_make_data(data_dst, dst_vtcm_data + (vtcm_idx * dst_half)), 0, 0, 0, 0);
        dma_queue_ધકેલો(dma_q, dma_make_data(src0_vtcm_data + (vtcm_idx * src0_half), data_src + soff), tb, tb, tb, 1);
    }

    unary_tile_compute_fn_t compute = (unary_tile_compute_fn_t) uctx->compute;

    uint32_t row = src0_start_row;
    uint32_t col = 0;
    uint32_t tile_in_row = 0;

    uint32_t prow = src0_start_row + fastdiv(2, div_tpr);
    uint32_t pcol = fastmodulo(2, tiles_per_row, div_tpr) * col_tile;
    uint32_t ptile_in_row = fastmodulo(2, tiles_per_row, div_tpr);

    માટે (uint32_t t = 0; t < total_tiles; t++) {
        શૂન્ય * dst_vtcm = (શૂન્ય *) (uintptr_t) dma_queue_કાઢો(dma_q).src;
        શૂન્ય * src_vtcm = (શૂન્ય *) (uintptr_t) dma_queue_કાઢો(dma_q).dst;

        અચલ uint32_t tw = MIN(col_tile, ne0 - col);

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, t);
        compute(dst_vtcm, src_vtcm, tw, uctx);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, t);

        અચલ size_t doff = (dst_contig ? (row * nb1) :
                             unary_row_offset(row, ne1, ne2, div_ne01, div_ne02, div_ne012, nb1, nb2, nb3)) +
                             (size_t) col * કદ(દશાંશ);
        અચલ size_t tb   = (size_t) tw * કદ(દશાંશ);
        dma_queue_ધકેલો(dma_q, dma_make_data(data_dst + doff, dst_vtcm), tb, tb, tb, 1);

        અચલ uint32_t pt = t + 2;
        જો (pt < total_tiles) {
            અચલ uint32_t ptw  = MIN(col_tile, ne0 - pcol);
            અચલ size_t   ptb  = (size_t) ptw * કદ(દશાંશ);
            અચલ size_t   psoff = (src0_contig ? (prow * nb01) :
                                    unary_row_offset(prow, ne01, ne02, div_ne01, div_ne02, div_ne012, nb01, nb02,
                                                     nb03)) +
                                   (size_t) pcol * કદ(દશાંશ);
            dma_queue_ધકેલો(dma_q, dma_make_data(src_vtcm, data_src + psoff), ptb, ptb, ptb, 1);
        }

        tile_in_row++;
        col += col_tile;
        જો (tile_in_row == tiles_per_row) {
            tile_in_row = 0;
            col = 0;
            row++;
        }

        ptile_in_row++;
        pcol += col_tile;
        જો (ptile_in_row == tiles_per_row) {
            ptile_in_row = 0;
            pcol = 0;
            prow++;
        }
    }

    dma_queue_flush(dma_q);
}

// 5. TRI tiled task.
સ્થિર શૂન્ય unary_thread_tiled_tri_f32(unsigned પૂર્ણાંક nth, unsigned પૂર્ણાંક ith, શૂન્ય * data) {
    (શૂન્ય) nth;
    અચલ સંરચના htp_unary_context * uctx = (અચલ સંરચના htp_unary_context *) data;
    સંરચના htp_ops_context * octx = uctx->octx;
    અચલ સંરચના htp_tensor * src = octx->src[0];
    અચલ સંરચના htp_tensor * dst = octx->dst;
    સંરચના htp_thread_trace * tr = &octx->ctx->trace[ith];

    htp_unary_preamble;

    અચલ uint32_t src0_nrows_per_thread = uctx->src0_nrows_per_thread;
    અચલ int32_t * op_params            = octx->op_params;
    અચલ uint32_t col_tile              = uctx->col_tile;

    અચલ uint32_t src0_nrows     = uctx->src0_nrows;
    અચલ uint32_t src0_start_row = uctx->row_start + src0_nrows_per_thread * ith;
    અચલ uint32_t src0_end_row   = MIN(src0_start_row + src0_nrows_per_thread, uctx->row_start + src0_nrows);

    જો (src0_start_row >= src0_end_row) {
        પાછા;
    }

    અચલ dma_addr_t data_src = uctx->data_src0;
    અચલ dma_addr_t data_dst = uctx->data_dst;

    uint8_t * src0_vtcm_data = uctx->vtcm_src0 + (ith * uctx->vtcm_src0_size_per_thread);
    uint8_t * dst_vtcm_data  = uctx->vtcm_dst + (ith * uctx->vtcm_dst_size_per_thread);

    અચલ size_t src0_half = uctx->src0_vtcm_half_size;
    અચલ size_t dst_half  = uctx->dst_vtcm_half_size;

    dma_queue * dma_q = octx->ctx->dma[ith];

    અચલ સંરચના fastdiv_કિંમતો * div_ne01  = &uctx->kparams->div_ne01;
    અચલ સંરચના fastdiv_કિંમતો * div_ne02  = &uctx->kparams->div_ne02;
    અચલ સંરચના fastdiv_કિંમતો * div_ne012 = &uctx->kparams->div_ne012;
    અચલ સંરચના fastdiv_કિંમતો * div_tpr   = &uctx->kparams->div_tpr;

    અચલ uint32_t tiles_per_row = (ne0 + col_tile - 1) / col_tile;
    અચલ int32_t  tri_ttype     = op_params[0];

    અચલ bool src0_contig = (nb02 == (size_t)ne01 * nb01) &&
                             (nb03 == (size_t)ne02 * nb02);
    અચલ bool dst_contig  = (nb2  == (size_t)ne1  * nb1)  &&
                             (nb3  == (size_t)ne2  * nb2);

    અચલ uint32_t total_tiles = (src0_end_row - src0_start_row) * tiles_per_row;

    માટે (uint32_t t = 0, vtcm_idx = 0; t < total_tiles && vtcm_idx < 2; t++, vtcm_idx++) {
        અચલ uint32_t row  = src0_start_row + t / tiles_per_row;
        અચલ uint32_t col  = (t % tiles_per_row) * col_tile;
        અચલ uint32_t tw   = MIN(col_tile, ne0 - col);
        અચલ size_t   tb   = (size_t) tw * કદ(દશાંશ);
        અચલ size_t   soff = (src0_contig ? (row * nb01) :
                               unary_row_offset(row, ne01, ne02, div_ne01, div_ne02, div_ne012, nb01, nb02, nb03)) +
                               (size_t) col * કદ(દશાંશ);

        dma_queue_ધકેલો(dma_q, dma_make_data(data_dst, dst_vtcm_data + (vtcm_idx * dst_half)), 0, 0, 0, 0);
        dma_queue_ધકેલો(dma_q, dma_make_data(src0_vtcm_data + (vtcm_idx * src0_half), data_src + soff), tb, tb, tb, 1);
    }

    unary_tiled_tri_compute_fn_t compute = (unary_tiled_tri_compute_fn_t) uctx->compute;

    uint32_t row = src0_start_row;
    uint32_t col = 0;
    uint32_t tile_in_row = 0;
    uint32_t i01 = fastmodulo(row, ne01, div_ne01);

    uint32_t prow = src0_start_row + fastdiv(2, div_tpr);
    uint32_t pcol = fastmodulo(2, tiles_per_row, div_tpr) * col_tile;
    uint32_t ptile_in_row = fastmodulo(2, tiles_per_row, div_tpr);

    માટે (uint32_t t = 0; t < total_tiles; t++) {
        શૂન્ય * dst_vtcm = (શૂન્ય *) (uintptr_t) dma_queue_કાઢો(dma_q).src;
        શૂન્ય * src_vtcm = (શૂન્ય *) (uintptr_t) dma_queue_કાઢો(dma_q).dst;

        અચલ uint32_t tw = MIN(col_tile, ne0 - col);

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, t);
        compute(src_vtcm, dst_vtcm, tw, col, i01, ne0, tri_ttype);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, t);

        અચલ size_t doff = (dst_contig ? (row * nb1) :
                             unary_row_offset(row, ne1, ne2, div_ne01, div_ne02, div_ne012, nb1, nb2, nb3)) +
                             (size_t) col * કદ(દશાંશ);
        અચલ size_t tb   = (size_t) tw * કદ(દશાંશ);
        dma_queue_ધકેલો(dma_q, dma_make_data(data_dst + doff, dst_vtcm), tb, tb, tb, 1);

        અચલ uint32_t pt = t + 2;
        જો (pt < total_tiles) {
            અચલ uint32_t ptw  = MIN(col_tile, ne0 - pcol);
            અચલ size_t   ptb  = (size_t) ptw * કદ(દશાંશ);
            અચલ size_t   psoff = (src0_contig ? (prow * nb01) :
                                    unary_row_offset(prow, ne01, ne02, div_ne01, div_ne02, div_ne012, nb01, nb02,
                                                     nb03)) +
                                   (size_t) pcol * કદ(દશાંશ);
            dma_queue_ધકેલો(dma_q, dma_make_data(src_vtcm, data_src + psoff), ptb, ptb, ptb, 1);
        }

        tile_in_row++;
        col += col_tile;
        જો (tile_in_row == tiles_per_row) {
            tile_in_row = 0;
            col = 0;
            row++;
            i01++;
            જો (i01 == ne01) {
                i01 = 0;
            }
        }

        ptile_in_row++;
        pcol += col_tile;
        જો (ptile_in_row == tiles_per_row) {
            ptile_in_row = 0;
            pcol = 0;
            prow++;
        }
    }

    dma_queue_flush(dma_q);
}

સ્થિર પૂર્ણાંક execute_op_unary(સંરચના htp_ops_context * octx) {
    પૂર્ણાંક err = HTP_STATUS_OK;

    અચલ સંરચના htp_tensor * src0 = octx->src[0];
    અચલ સંરચના htp_tensor * dst  = octx->dst;

    અચલ bool is_f16 = (src0->type == HTP_TYPE_F16);

    અચલ અક્ષર * op_type = ખાલી;

    પસંદગી (octx->op) {
        સ્થિતિ HTP_OP_NORM:            op_type = is_f16 ? "norm-f16"     : "norm-f32";         તોડો;
        સ્થિતિ HTP_OP_RMS_NORM:        op_type = is_f16 ? "rmsnorm-f16"  : "rmsnorm-f32";      તોડો;
        સ્થિતિ HTP_OP_RMS_NORM_MUL:    op_type = "rmsnorm-mul-f32";                            તોડો;
        સ્થિતિ HTP_OP_SCALE:           op_type = is_f16 ? "scale-f16"    : "scale-f32";        તોડો;
        સ્થિતિ HTP_OP_CLAMP:           op_type = is_f16 ? "clamp-f16"    : "clamp-f32";        તોડો;
        સ્થિતિ HTP_OP_LEAKY_RELU:      op_type = "leaky-relu-f32";                             તોડો;
        સ્થિતિ HTP_OP_SQR:             op_type = is_f16 ? "sqr-f16"      : "sqr-f32";          તોડો;
        સ્થિતિ HTP_OP_SQRT:            op_type = is_f16 ? "sqrt-f16"     : "sqrt-f32";         તોડો;
        સ્થિતિ HTP_OP_UNARY_NEG:       op_type = "neg-f32";                                    તોડો;
        સ્થિતિ HTP_OP_UNARY_EXP:       op_type = "exp-f32";                                    તોડો;
        સ્થિતિ HTP_OP_UNARY_SIGMOID:   op_type = "sigmoid-f32";                                તોડો;
        સ્થિતિ HTP_OP_UNARY_SILU:      op_type = "silu-f32";                                   તોડો;
        સ્થિતિ HTP_OP_UNARY_GELU:      op_type = "gelu-f32";                                   તોડો;
        સ્થિતિ HTP_OP_UNARY_GELU_QUICK: op_type = "gelu-quick-f32";                             તોડો;
        સ્થિતિ HTP_OP_UNARY_GELU_ERF:  op_type = "gelu-erf-f32";                               તોડો;
        સ્થિતિ HTP_OP_UNARY_SOFTPLUS:  op_type = "softplus-f32";                               તોડો;
        સ્થિતિ HTP_OP_UNARY_TANH:      op_type = "tanh-f32";                                   તોડો;
        સ્થિતિ HTP_OP_UNARY_ABS:       op_type = is_f16 ? "abs-f16"      : "abs-f32";          તોડો;
        સ્થિતિ HTP_OP_UNARY_LOG:       op_type = is_f16 ? "log-f16"      : "log-f32";          તોડો;
        સ્થિતિ HTP_OP_UNARY_RELU:      op_type = "relu-f32";                                   તોડો;
        સ્થિતિ HTP_OP_UNARY_STEP:      op_type = is_f16 ? "step-f16"     : "step-f32";         તોડો;
        સ્થિતિ HTP_OP_L2_NORM:         op_type = is_f16 ? "l2norm-f16"   : "l2norm-f32";       તોડો;
        સ્થિતિ HTP_OP_TRI:             op_type = "tri-f32";                                    તોડો;
        સ્થિતિ HTP_OP_HADAMARD:        op_type = is_f16 ? "hadamard-f16" : "hadamard-f32";     તોડો;
        મૂળભૂત:
            FARF(ERROR, "Unsupported unary Op %u\n", octx->op);
            પાછા HTP_STATUS_NO_SUPPORT;
    }

    // F16 only has row-block kernels for this subset of ops (see the dispatch switch
    // below) - reject everything else up front, before touching kparams/VTCM.
    જો (is_f16) {
        પસંદગી (octx->op) {
            સ્થિતિ HTP_OP_NORM:
            સ્થિતિ HTP_OP_RMS_NORM:
            સ્થિતિ HTP_OP_SCALE:
            સ્થિતિ HTP_OP_CLAMP:
            સ્થિતિ HTP_OP_SQR:
            સ્થિતિ HTP_OP_SQRT:
            સ્થિતિ HTP_OP_L2_NORM:
            સ્થિતિ HTP_OP_UNARY_ABS:
            સ્થિતિ HTP_OP_UNARY_LOG:
            સ્થિતિ HTP_OP_UNARY_STEP:
            સ્થિતિ HTP_OP_HADAMARD:
                તોડો;
            મૂળભૂત:
                FARF(ERROR, "unary-%s: not supported for F16\n", op_type);
                પાછા HTP_STATUS_NO_SUPPORT;
        }
    }

    અચલ સંરચના htp_unary_kernel_params * kparams = (અચલ સંરચના htp_unary_kernel_params *) octx->kernel_params;

    જો (!htp_ops_context_set_n_threads(octx, kparams->n_threads)) {
        પાછા HTP_STATUS_INVAL_PARAMS;
    }

    અચલ uint32_t src0_nrows = src0->ne[1] * src0->ne[2] * src0->ne[3];
    અચલ size_t src0_elem_size = is_f16 ? કદ(_Float16) : કદ(દશાંશ);
    અચલ size_t dst_elem_size  = (dst->type == HTP_TYPE_F16) ? કદ(_Float16) : કદ(દશાંશ);
    અચલ size_t src0_data_row_size = src0->ne[0] * src0_elem_size;
    અચલ size_t dst_data_row_size  = dst->ne[0]  * dst_elem_size;

    uint32_t row_start = 0;
    uint32_t nrows     = src0_nrows;

    જો (octx->ctx->mdev.ગણતરી > 1) {
        uint32_t rows_per_chunk = 0;
        htp_tensor_mdev_rows_per_chunk(dst, (uint32_t) dst_elem_size, (uint32_t) dst_data_row_size, &rows_per_chunk);
        અચલ સંરચના htp_tensor_mdev_range range = htp_tensor_mdev_partition(src0_nrows, rows_per_chunk, octx->ctx->mdev.idx, octx->ctx->mdev.ગણતરી, &octx->ctx->mdev.ગણતરી_div);
        row_start = range.start;
        nrows     = range.ગણતરી;
    }

    જો (nrows == 0) {
        પાછા HTP_STATUS_OK;
    }

    અચલ uint32_t n_threads = octx->n_threads;

    અચલ size_t src0_row_size_aligned = kparams->src0_row_size_aligned;
    અચલ size_t dst_row_size_aligned  = kparams->dst_row_size_aligned;

    // Always 0 for F16 - htp_unary_vtcm_layout_build() keeps F16 on the row-block path,
    // since only F32 has unary_task_f32_tiled_* kernels.
    અચલ uint32_t col_tile = kparams->col_tile;

    size_t src1_data_row_size = 0;
    size_t src1_row_size_aligned = kparams->src1_row_size_aligned;
    bool broadcast_weight = kparams->broadcast_weight;
    અચલ સંરચના htp_tensor * src1 = ખાલી;

    // RMS_NORM_MUL fusion is F32-only (its weight tensor is always F32; see
    // try_fuse_node()'s type guard), so this never triggers when is_f16 is true.
    જો (octx->op == HTP_OP_RMS_NORM_MUL) {
        src1 = octx->src[1];
        src1_data_row_size = src1->ne[0] * કદ(દશાંશ);
    }

    જો (octx->ctx->vtcm_size < (size_t)kparams->vtcm_size) {
        FARF(ERROR, "unary-%s : current VTCM reservation %zu is too small, needed %zu\n", op_type, octx->ctx->vtcm_size, (size_t)kparams->vtcm_size);
        પાછા HTP_STATUS_VTCM_TOO_SMALL;
    }

    octx->src0_spad.src = ખાલી;
    octx->src1_spad.src = ખાલી;
    octx->dst_spad.src  = ખાલી;

    FARF(ઉપલું_2, "%s: (%ux%ux%ux%u) -> (%ux%ux%ux%u) : src0-vtcm-size %u src1-vtcm-size %u dst-vtcm-size %u\n", op_type,
         src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3], dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3],
         kparams->vtcm_src0_size, kparams->vtcm_src1_size, kparams->vtcm_dst_size);

    uint8_t * અચલ base = (uint8_t *) octx->ctx->vtcm_base;
    સંરચના htp_unary_context uctx = {
        .octx                  = octx,
        .kparams               = kparams,
        .src0_nrows_per_thread = fastdiv(nrows + n_threads - 1, &octx->n_threads_div),
        .src0_nrows            = nrows,
        .row_start             = row_start,

        .data_src0             = src0->data,
        .data_src1             = (octx->op == HTP_OP_RMS_NORM_MUL) ? src1->data : 0,
        .data_dst              = dst->data,

        .src0_data_row_size    = src0_data_row_size,
        .src1_data_row_size    = src1_data_row_size,
        .dst_data_row_size     = dst_data_row_size,

        .src0_row_size_aligned = src0_row_size_aligned,
        .src1_row_size_aligned = src1_row_size_aligned,
        .dst_row_size_aligned  = dst_row_size_aligned,

        .src0_vtcm_half_size   = kparams->vtcm_src0_size_per_thread / 2,
        .src1_vtcm_half_size   = (octx->op == HTP_OP_RMS_NORM_MUL) ? (kparams->vtcm_src1_size_per_thread / (broadcast_weight ? 1 : 2)) : 0,
        .dst_vtcm_half_size    = kparams->vtcm_dst_size_per_thread / 2,

        .block                 = kparams->block,
        .nc                    = src0->ne[0],
        .col_tile              = col_tile,
        .broadcast_weight      = broadcast_weight,

        .vtcm_src0             = VTCM_LAYOUT_PTR(uint8_t, base, 0),
        .vtcm_src1             = VTCM_LAYOUT_PTR_OPTIONAL(uint8_t, base, kparams->vtcm_src0_size, kparams->vtcm_src1_size > 0),
        .vtcm_dst              = VTCM_LAYOUT_PTR(uint8_t, base, kparams->vtcm_src0_size + kparams->vtcm_src1_size),

        .vtcm_src0_size_per_thread = kparams->vtcm_src0_size_per_thread,
        .vtcm_src1_size_per_thread = kparams->vtcm_src1_size_per_thread,
        .vtcm_dst_size_per_thread  = kparams->vtcm_dst_size_per_thread,
    };

    FARF(ઉપલું_2, "%s: %s mode (col_tile %u)\n", op_type, col_tile ? "tiled" : "row-block", col_tile);

    worker_callback_t task_func = ખાલી;
    શૂન્ય * compute_func = ખાલી;

    જો (col_tile) {
        task_func = unary_thread_tiled;
        પસંદગી (octx->op) {
            સ્થિતિ HTP_OP_SCALE:           compute_func = (શૂન્ય *) tile_scale_f32;          તોડો;
            સ્થિતિ HTP_OP_CLAMP:           compute_func = (શૂન્ય *) tile_clamp_f32;          તોડો;
            સ્થિતિ HTP_OP_LEAKY_RELU:      compute_func = (શૂન્ય *) tile_leaky_relu_f32;     તોડો;
            સ્થિતિ HTP_OP_SQR:             compute_func = (શૂન્ય *) tile_sqr_f32;            તોડો;
            સ્થિતિ HTP_OP_SQRT:            compute_func = (શૂન્ય *) tile_sqrt_f32;           તોડો;
            સ્થિતિ HTP_OP_UNARY_NEG:       compute_func = (શૂન્ય *) tile_neg_f32;            તોડો;
            સ્થિતિ HTP_OP_UNARY_EXP:       compute_func = (શૂન્ય *) tile_exp_f32;            તોડો;
            સ્થિતિ HTP_OP_UNARY_SIGMOID:   compute_func = (શૂન્ય *) tile_sigmoid_f32;        તોડો;
            સ્થિતિ HTP_OP_UNARY_SILU:      compute_func = (શૂન્ય *) tile_silu_f32;           તોડો;
            સ્થિતિ HTP_OP_UNARY_GELU:      compute_func = (શૂન્ય *) tile_gelu_f32;           તોડો;
            સ્થિતિ HTP_OP_UNARY_GELU_QUICK: compute_func = (શૂન્ય *) tile_gelu_quick_f32;     તોડો;
            સ્થિતિ HTP_OP_UNARY_GELU_ERF:  compute_func = (શૂન્ય *) tile_gelu_erf_f32;       તોડો;
            સ્થિતિ HTP_OP_UNARY_SOFTPLUS:  compute_func = (શૂન્ય *) tile_softplus_f32;       તોડો;
            સ્થિતિ HTP_OP_UNARY_TANH:      compute_func = (શૂન્ય *) tile_tanh_f32;           તોડો;
            સ્થિતિ HTP_OP_UNARY_ABS:       compute_func = (શૂન્ય *) tile_abs_f32;            તોડો;
            સ્થિતિ HTP_OP_UNARY_LOG:       compute_func = (શૂન્ય *) tile_log_f32;            તોડો;
            સ્થિતિ HTP_OP_UNARY_RELU:      compute_func = (શૂન્ય *) tile_relu_f32;           તોડો;
            સ્થિતિ HTP_OP_UNARY_STEP:      compute_func = (શૂન્ય *) tile_step_f32;           તોડો;
            સ્થિતિ HTP_OP_TRI:
                task_func    = unary_thread_tiled_tri_f32;
                compute_func = (શૂન્ય *) tri_apply_tile_f32;
                તોડો;
            મૂળભૂત:                     તોડો;
        }
    } નહીંતર જો (is_f16) {
        task_func = unary_thread_row_block;
        પસંદગી (octx->op) {
            સ્થિતિ HTP_OP_NORM:            compute_func = (શૂન્ય *) norm_f16;                તોડો;
            સ્થિતિ HTP_OP_RMS_NORM:        compute_func = (શૂન્ય *) rms_norm_f16;            તોડો;
            સ્થિતિ HTP_OP_SCALE:           compute_func = (શૂન્ય *) scale_f16;               તોડો;
            સ્થિતિ HTP_OP_CLAMP:           compute_func = (શૂન્ય *) clamp_f16;               તોડો;
            સ્થિતિ HTP_OP_SQR:             compute_func = (શૂન્ય *) sqr_f16;                 તોડો;
            સ્થિતિ HTP_OP_SQRT:            compute_func = (શૂન્ય *) sqrt_f16;                તોડો;
            સ્થિતિ HTP_OP_L2_NORM:         compute_func = (શૂન્ય *) l2_norm_f16;             તોડો;
            સ્થિતિ HTP_OP_UNARY_ABS:       compute_func = (શૂન્ય *) abs_f16;                 તોડો;
            સ્થિતિ HTP_OP_UNARY_LOG:       compute_func = (શૂન્ય *) log_f16;                 તોડો;
            સ્થિતિ HTP_OP_UNARY_STEP:      compute_func = (શૂન્ય *) step_f16;                તોડો;
            સ્થિતિ HTP_OP_HADAMARD:        compute_func = (શૂન્ય *) hadamard_f16;            તોડો;
            મૂળભૂત:                     તોડો;
        }
    } નહીંતર {
        task_func = unary_thread_row_block;
        પસંદગી (octx->op) {
            સ્થિતિ HTP_OP_NORM:            compute_func = (શૂન્ય *) norm_f32;                તોડો;
            સ્થિતિ HTP_OP_RMS_NORM:        compute_func = (શૂન્ય *) rms_norm_f32;            તોડો;
            સ્થિતિ HTP_OP_RMS_NORM_MUL:
                task_func    = unary_thread_rms_norm_mul_f32;
                compute_func = (શૂન્ય *) rms_norm_mul_f32;
                તોડો;
            સ્થિતિ HTP_OP_SCALE:           compute_func = (શૂન્ય *) scale_f32;               તોડો;
            સ્થિતિ HTP_OP_CLAMP:           compute_func = (શૂન્ય *) clamp_f32;               તોડો;
            સ્થિતિ HTP_OP_LEAKY_RELU:      compute_func = (શૂન્ય *) leaky_relu_f32;          તોડો;
            સ્થિતિ HTP_OP_SQR:             compute_func = (શૂન્ય *) sqr_f32;                 તોડો;
            સ્થિતિ HTP_OP_SQRT:            compute_func = (શૂન્ય *) sqrt_f32;                તોડો;
            સ્થિતિ HTP_OP_UNARY_NEG:       compute_func = (શૂન્ય *) neg_f32;                 તોડો;
            સ્થિતિ HTP_OP_UNARY_EXP:       compute_func = (શૂન્ય *) exp_f32;                 તોડો;
            સ્થિતિ HTP_OP_UNARY_SIGMOID:   compute_func = (શૂન્ય *) sigmoid_f32;             તોડો;
            સ્થિતિ HTP_OP_UNARY_SILU:      compute_func = (શૂન્ય *) silu_f32;                તોડો;
            સ્થિતિ HTP_OP_UNARY_GELU:      compute_func = (શૂન્ય *) gelu_f32;                તોડો;
            સ્થિતિ HTP_OP_UNARY_GELU_QUICK: compute_func = (શૂન્ય *) gelu_quick_f32;          તોડો;
            સ્થિતિ HTP_OP_UNARY_GELU_ERF:  compute_func = (શૂન્ય *) gelu_erf_f32;            તોડો;
            સ્થિતિ HTP_OP_UNARY_SOFTPLUS:  compute_func = (શૂન્ય *) softplus_f32;            તોડો;
            સ્થિતિ HTP_OP_UNARY_TANH:      compute_func = (શૂન્ય *) tanh_f32;                તોડો;
            સ્થિતિ HTP_OP_UNARY_ABS:       compute_func = (શૂન્ય *) abs_f32;                 તોડો;
            સ્થિતિ HTP_OP_UNARY_LOG:       compute_func = (શૂન્ય *) log_f32;                 તોડો;
            સ્થિતિ HTP_OP_UNARY_RELU:      compute_func = (શૂન્ય *) relu_f32;                તોડો;
            સ્થિતિ HTP_OP_UNARY_STEP:      compute_func = (શૂન્ય *) step_f32;                તોડો;
            સ્થિતિ HTP_OP_L2_NORM:         compute_func = (શૂન્ય *) l2_norm_f32;             તોડો;
            સ્થિતિ HTP_OP_HADAMARD:        compute_func = (શૂન્ય *) hadamard_f32;            તોડો;
            સ્થિતિ HTP_OP_TRI:
                task_func    = unary_thread_tri_f32;
                compute_func = (શૂન્ય *) tri_f32;
                તોડો;
            મૂળભૂત:                     તોડો;
        }
    }

    જો (!task_func || !compute_func) {
        FARF(ERROR, "execute_op_unary: task function is NULL for op %d\n", octx->op);
        પાછા HTP_STATUS_NO_SUPPORT;
    }

    uctx.compute = compute_func;
    work_queue_run(octx->ctx->work_queue, task_func, &uctx, n_threads);

    પાછા err;
}

પૂર્ણાંક op_unary(સંરચના htp_ops_context * octx) {
    પસંદગી (octx->src[0]->type) {
        સ્થિતિ HTP_TYPE_F32:
        સ્થિતિ HTP_TYPE_F16:
            પાછા execute_op_unary(octx);

        મૂળભૂત:
            પાછા HTP_STATUS_NO_SUPPORT;
    }
}
