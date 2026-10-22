#pragma clang diagnostic ignored "-Wunused-variable"
#pragma clang diagnostic ignored "-Wunused-function"
#pragma clang diagnostic ignored "-Wunused-but-set-variable"

#ضمن <HAP_farf.h>
#ضمن <HAP_perf.h>

#ضمن <math.h>
#ضمن <string.h>

#ضمن "dma-queue.h"
#ضمن "hex-fastdiv.h"
#ضمن "hvx-exp.h"
#ضمن "hvx-sigmoid.h"
#ضمن "hvx-utils.h"
#ضمن "unary-ops.h"

#define GGML_COMMON_DECL_C
#ضمن "ggml-common.h"
#ضمن "htp-ctx.h"
#ضمن "htp-ops.h"
#ضمن "htp-tensor.h"
#ضمن "htp-vtcm.h"
#ضمن "hex-profile.h"

بنية htp_unary_context;

تعريفنوع فراغ (*unary_compute_fn_t)(ثابت فراغ * restrict src,
                                   فراغ * restrict dst,
                                   uint32_t ژمارە_rows,
                                   ثابت بنية htp_unary_context * uctx);

تعريفنوع فراغ (*unary_rms_norm_mul_compute_fn_t)(ثابت فراغ * restrict src,
                                                ثابت فراغ * restrict weight,
                                                فراغ * restrict dst,
                                                uint32_t ژمارە_rows,
                                                ثابت بنية htp_unary_context * uctx);

تعريفنوع فراغ (*unary_tri_compute_fn_t)(ثابت فراغ * restrict src,
                                       فراغ * restrict dst,
                                       uint32_t ژمارە_rows,
                                       uint32_t ir,
                                       ثابت بنية htp_unary_context * uctx);

تعريفنوع فراغ (*unary_tile_compute_fn_t)(فراغ * restrict dst,
                                        ثابت فراغ * restrict src,
                                        uint32_t tw,
                                        ثابت بنية htp_unary_context * uctx);

تعريفنوع فراغ (*unary_tiled_tri_compute_fn_t)(ثابت فراغ * restrict src,
                                             فراغ * restrict dst,
                                             uint32_t tile_elems,
                                             uint32_t col_start,
                                             uint32_t i01,
                                             uint32_t ne0,
                                             int32_t ttype);

بنية htp_unary_context {
    بنية htp_ops_context * octx;
    ثابت بنية htp_unary_kernel_params * kparams;

    فراغ *                    compute;

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
ساكن مضمن size_t unary_row_offset(uint32_t ir,
                                      uint32_t ne1, uint32_t ne2,
                                      ثابت بنية fastdiv_بەهاکان * div_ne1,
                                      ثابت بنية fastdiv_بەهاکان * div_ne2,
                                      ثابت بنية fastdiv_بەهاکان * div_ne12,
                                      size_t nb1, size_t nb2, size_t nb3) {
    ثابت uint32_t i1 = fastmodulo(ir, ne1, div_ne1);
    ثابت uint32_t ir_div_ne1 = fastdiv(ir, div_ne1);
    ثابت uint32_t i2 = fastmodulo(ir_div_ne1, ne2, div_ne2);
    ثابت uint32_t i3 = fastdiv(ir, div_ne12);
    أرجع i1 * nb1 + i2 * nb2 + i3 * nb3;
}

// Safe DMA block size from row `ir`: clamp to the tighter dim-1 slice
// boundary of src and dst so the nb1 stride stays valid for all rows.
ساكن مضمن uint32_t unary_block_size(uint32_t ir,
                                        uint32_t end_row,
                                        uint32_t block,
                                        bool src_contig,
                                        bool dst_contig,
                                        uint32_t ne1,
                                        ثابت بنية fastdiv_بەهاکان * div_ne1) {
    uint32_t limit = MIN(block, end_row - ir);

    إذا (!src_contig || !dst_contig) {
        ثابت uint32_t slice_end = (fastdiv(ir, div_ne1) + 1) * ne1;
        limit = MIN(limit, slice_end - ir);
    }

    أرجع limit;
}

#define htp_unary_preamble            \
    ثابت uint32_t ne00 = src->ne[0]; \
    ثابت uint32_t ne01 = src->ne[1]; \
    ثابت uint32_t ne02 = src->ne[2]; \
    ثابت uint32_t ne03 = src->ne[3]; \
                                      \
    ثابت uint32_t ne0 = dst->ne[0];  \
    ثابت uint32_t ne1 = dst->ne[1];  \
    ثابت uint32_t ne2 = dst->ne[2];  \
    ثابت uint32_t ne3 = dst->ne[3];  \
                                      \
    ثابت uint32_t nb00 = src->nb[0]; \
    ثابت uint32_t nb01 = src->nb[1]; \
    ثابت uint32_t nb02 = src->nb[2]; \
    ثابت uint32_t nb03 = src->nb[3]; \
                                      \
    ثابت uint32_t nb0 = dst->nb[0];  \
    ثابت uint32_t nb1 = dst->nb[1];  \
    ثابت uint32_t nb2 = dst->nb[2];  \
    ثابت uint32_t nb3 = dst->nb[3];

#define htp_unary_op_preamble                                         \
    int32_t * op_params = uctx->octx->op_params;                      \
    ثابت uint32_t ne0 = uctx->nc;                                    \
    ثابت size_t src0_row_size_aligned = uctx->src0_row_size_aligned; \
    ثابت size_t dst_row_size_aligned = uctx->dst_row_size_aligned;

ساكن فراغ scale_f32(ثابت فراغ * restrict src,
                      فراغ * restrict dst,
                      ثابت uint32_t ژمارە_rows,
                      ثابت بنية htp_unary_context * uctx) {
    htp_unary_op_preamble;
    عائم scale = 0.f;
    عائم bias  = 0.f;
    memcpy(&scale, &op_params[0], حجم(عائم));
    memcpy(&bias,  &op_params[1], حجم(عائم));

    لكل (uint32_t ir = 0; ir < ژمارە_rows; ir++) {
        ثابت uint8_t * restrict src_local = (ثابت uint8_t *)src + (ir * src0_row_size_aligned);
        uint8_t * restrict dst_local       = (uint8_t *)dst + (ir * dst_row_size_aligned);

        hvx_scale_offset_f32_aa((uint8_t *) dst_local, (ثابت uint8_t *) src_local, ne0, scale, bias);
    }
}

ساكن فراغ clamp_f32(ثابت فراغ * restrict src,
                      فراغ * restrict dst,
                      ثابت uint32_t ژمارە_rows,
                      ثابت بنية htp_unary_context * uctx) {
    htp_unary_op_preamble;
    عائم min = 0.f;
    عائم max = 0.f;
    memcpy(&min, &op_params[0], حجم(عائم));
    memcpy(&max, &op_params[1], حجم(عائم));

    لكل (uint32_t ir = 0; ir < ژمارە_rows; ir++) {
        ثابت uint8_t * restrict src_local = (ثابت uint8_t *)src + (ir * src0_row_size_aligned);
        uint8_t * restrict dst_local       = (uint8_t *)dst + (ir * dst_row_size_aligned);

        hvx_clamp_scalar_f32(dst_local, src_local, min, max, ne0);
    }
}

ساكن فراغ leaky_relu_f32(ثابت فراغ * restrict src,
                           فراغ * restrict dst,
                           ثابت uint32_t ژمارە_rows,
                           ثابت بنية htp_unary_context * uctx) {
    htp_unary_op_preamble;
    عائم negative_slope = 0.f;
    memcpy(&negative_slope, &op_params[0], حجم(عائم));

    لكل (uint32_t ir = 0; ir < ژمارە_rows; ir++) {
        ثابت uint8_t * restrict src_local = (ثابت uint8_t *)src + (ir * src0_row_size_aligned);
        uint8_t * restrict dst_local       = (uint8_t *)dst + (ir * dst_row_size_aligned);

        hvx_leaky_relu_scalar_f32(dst_local, src_local, negative_slope, ne0);
    }
}

ساكن فراغ rms_norm_f32(ثابت فراغ * restrict src,
                         فراغ * restrict dst,
                         ثابت uint32_t ژمارە_rows,
                         ثابت بنية htp_unary_context * uctx) {
    htp_unary_op_preamble;
    عائم epsilon = 0.f;
    memcpy(&epsilon, op_params, حجم(عائم));

    لكل (uint32_t ir = 0; ir < ژمارە_rows; ir++) {
        ثابت uint8_t * restrict src_local = (ثابت uint8_t *)src + (ir * src0_row_size_aligned);
        uint8_t * restrict dst_local       = (uint8_t *)dst + (ir * dst_row_size_aligned);

        hvx_fast_rms_norm_f32((ثابت uint8_t *) src_local, (uint8_t *) dst_local, ne0, epsilon);
    }
}

ساكن فراغ rms_norm_mul_f32(ثابت فراغ * restrict src,
                             ثابت فراغ * restrict weight,
                             فراغ * restrict dst,
                             ثابت uint32_t ژمارە_rows,
                             ثابت بنية htp_unary_context * uctx) {
    htp_unary_op_preamble;
    عائم epsilon = 0.f;
    memcpy(&epsilon, op_params, حجم(عائم));

    لكل (uint32_t ir = 0; ir < ژمارە_rows; ir++) {
        ثابت uint8_t * restrict src_local = (ثابت uint8_t *)src + (ir * src0_row_size_aligned);
        ثابت uint8_t * restrict w_local   = (ثابت uint8_t *)weight + (uctx->broadcast_weight ? 0 : ir * uctx->src1_row_size_aligned);
        uint8_t * restrict dst_local       = (uint8_t *)dst + (ir * dst_row_size_aligned);

        hvx_fast_rms_norm_mul_f32(src_local, w_local, dst_local, ne0, epsilon);
    }
}

ساكن فراغ norm_f32(ثابت فراغ * restrict src,
                     فراغ * restrict dst,
                     ثابت uint32_t ژمارە_rows,
                     ثابت بنية htp_unary_context * uctx) {
    htp_unary_op_preamble;
    عائم epsilon = 0.f;
    memcpy(&epsilon, op_params, حجم(عائم));

    لكل (uint32_t ir = 0; ir < ژمارە_rows; ir++) {
        ثابت uint8_t * restrict src_local = (ثابت uint8_t *)src + (ir * src0_row_size_aligned);
        uint8_t * restrict dst_local       = (uint8_t *)dst + (ir * dst_row_size_aligned);

        hvx_fast_norm_f32((ثابت uint8_t *) src_local, (uint8_t *) dst_local, ne0, epsilon);
    }
}

ساكن فراغ sqr_f32(ثابت فراغ * restrict src,
                    فراغ * restrict dst,
                    ثابت uint32_t ژمارە_rows,
                    ثابت بنية htp_unary_context * uctx) {
    htp_unary_op_preamble;

    لكل (uint32_t ir = 0; ir < ژمارە_rows; ir++) {
        ثابت uint8_t * restrict src_local = (ثابت uint8_t *)src + (ir * src0_row_size_aligned);
        uint8_t * restrict dst_local       = (uint8_t *)dst + (ir * dst_row_size_aligned);

        hvx_sqr_f32_aa((uint8_t *) dst_local, (ثابت uint8_t *) src_local, ne0);
    }
}

ساكن فراغ sqrt_f32(ثابت فراغ * restrict src,
                     فراغ * restrict dst,
                     ثابت uint32_t ژمارە_rows,
                     ثابت بنية htp_unary_context * uctx) {
    htp_unary_op_preamble;

    لكل (uint32_t ir = 0; ir < ژمارە_rows; ir++) {
        ثابت uint8_t * restrict src_local = (ثابت uint8_t *)src + (ir * src0_row_size_aligned);
        uint8_t * restrict dst_local       = (uint8_t *)dst + (ir * dst_row_size_aligned);

        hvx_sqrt_f32_aa((uint8_t *) dst_local, (ثابت uint8_t *) src_local, ne0);
    }
}

ساكن فراغ scale_f16(ثابت فراغ * restrict src,
                      فراغ * restrict dst,
                      ثابت uint32_t ژمارە_rows,
                      ثابت بنية htp_unary_context * uctx) {
    htp_unary_op_preamble;
    عائم scale = 0.f;
    عائم bias  = 0.f;
    memcpy(&scale, &op_params[0], حجم(عائم));
    memcpy(&bias,  &op_params[1], حجم(عائم));

    لكل (uint32_t ir = 0; ir < ژمارە_rows; ir++) {
        ثابت uint8_t * restrict src_local = (ثابت uint8_t *)src + (ir * src0_row_size_aligned);
        uint8_t * restrict dst_local       = (uint8_t *)dst + (ir * dst_row_size_aligned);

        hvx_scale_offset_f16_aa((uint8_t *) dst_local, (ثابت uint8_t *) src_local, ne0, scale, bias);
    }
}

ساكن فراغ clamp_f16(ثابت فراغ * restrict src,
                      فراغ * restrict dst,
                      ثابت uint32_t ژمارە_rows,
                      ثابت بنية htp_unary_context * uctx) {
    htp_unary_op_preamble;
    عائم min = 0.f;
    عائم max = 0.f;
    memcpy(&min, &op_params[0], حجم(عائم));
    memcpy(&max, &op_params[1], حجم(عائم));

    لكل (uint32_t ir = 0; ir < ژمارە_rows; ir++) {
        ثابت uint8_t * restrict src_local = (ثابت uint8_t *)src + (ir * src0_row_size_aligned);
        uint8_t * restrict dst_local       = (uint8_t *)dst + (ir * dst_row_size_aligned);

        hvx_clamp_scalar_f16(dst_local, src_local, (_Float16) min, (_Float16) max, ne0);
    }
}

ساكن فراغ rms_norm_f16(ثابت فراغ * restrict src,
                         فراغ * restrict dst,
                         ثابت uint32_t ژمارە_rows,
                         ثابت بنية htp_unary_context * uctx) {
    htp_unary_op_preamble;
    عائم epsilon = 0.f;
    memcpy(&epsilon, op_params, حجم(عائم));

    لكل (uint32_t ir = 0; ir < ژمارە_rows; ir++) {
        ثابت uint8_t * restrict src_local = (ثابت uint8_t *)src + (ir * src0_row_size_aligned);
        uint8_t * restrict dst_local       = (uint8_t *)dst + (ir * dst_row_size_aligned);

        hvx_fast_rms_norm_f16((ثابت uint8_t *) src_local, (uint8_t *) dst_local, ne0, epsilon);
    }
}

ساكن فراغ norm_f16(ثابت فراغ * restrict src,
                     فراغ * restrict dst,
                     ثابت uint32_t ژمارە_rows,
                     ثابت بنية htp_unary_context * uctx) {
    htp_unary_op_preamble;
    عائم epsilon = 0.f;
    memcpy(&epsilon, op_params, حجم(عائم));

    لكل (uint32_t ir = 0; ir < ژمارە_rows; ir++) {
        ثابت uint8_t * restrict src_local = (ثابت uint8_t *)src + (ir * src0_row_size_aligned);
        uint8_t * restrict dst_local       = (uint8_t *)dst + (ir * dst_row_size_aligned);

        hvx_fast_norm_f16((ثابت uint8_t *) src_local, (uint8_t *) dst_local, ne0, epsilon);
    }
}

ساكن فراغ sqr_f16(ثابت فراغ * restrict src,
                    فراغ * restrict dst,
                    ثابت uint32_t ژمارە_rows,
                    ثابت بنية htp_unary_context * uctx) {
    htp_unary_op_preamble;

    لكل (uint32_t ir = 0; ir < ژمارە_rows; ir++) {
        ثابت uint8_t * restrict src_local = (ثابت uint8_t *)src + (ir * src0_row_size_aligned);
        uint8_t * restrict dst_local       = (uint8_t *)dst + (ir * dst_row_size_aligned);

        hvx_sqr_f16_aa((uint8_t *) dst_local, (ثابت uint8_t *) src_local, ne0);
    }
}

ساكن فراغ sqrt_f16(ثابت فراغ * restrict src,
                     فراغ * restrict dst,
                     ثابت uint32_t ژمارە_rows,
                     ثابت بنية htp_unary_context * uctx) {
    htp_unary_op_preamble;

    لكل (uint32_t ir = 0; ir < ژمارە_rows; ir++) {
        ثابت uint8_t * restrict src_local = (ثابت uint8_t *)src + (ir * src0_row_size_aligned);
        uint8_t * restrict dst_local       = (uint8_t *)dst + (ir * dst_row_size_aligned);

        hvx_sqrt_f16_aa((uint8_t *) dst_local, (ثابت uint8_t *) src_local, ne0);
    }
}

ساكن فراغ abs_f16(ثابت فراغ * restrict src,
                    فراغ * restrict dst,
                    ثابت uint32_t ژمارە_rows,
                    ثابت بنية htp_unary_context * uctx) {
    htp_unary_op_preamble;

    لكل (uint32_t ir = 0; ir < ژمارە_rows; ir++) {
        ثابت uint8_t * restrict src_local = (ثابت uint8_t *)src + (ir * src0_row_size_aligned);
        uint8_t * restrict dst_local       = (uint8_t *)dst + (ir * dst_row_size_aligned);

        hvx_abs_f16_aa((uint8_t *) dst_local, (ثابت uint8_t *) src_local, ne0);
    }
}

ساكن فراغ log_f16(ثابت فراغ * restrict src,
                    فراغ * restrict dst,
                    ثابت uint32_t ژمارە_rows,
                    ثابت بنية htp_unary_context * uctx) {
    htp_unary_op_preamble;

    لكل (uint32_t ir = 0; ir < ژمارە_rows; ir++) {
        ثابت uint8_t * restrict src_local = (ثابت uint8_t *)src + (ir * src0_row_size_aligned);
        uint8_t * restrict dst_local       = (uint8_t *)dst + (ir * dst_row_size_aligned);

        hvx_log_f16_aa((uint8_t *) dst_local, (ثابت uint8_t *) src_local, ne0);
    }
}

ساكن فراغ step_f16(ثابت فراغ * restrict src,
                     فراغ * restrict dst,
                     ثابت uint32_t ژمارە_rows,
                     ثابت بنية htp_unary_context * uctx) {
    htp_unary_op_preamble;

    لكل (uint32_t ir = 0; ir < ژمارە_rows; ir++) {
        ثابت uint8_t * restrict src_local = (ثابت uint8_t *)src + (ir * src0_row_size_aligned);
        uint8_t * restrict dst_local       = (uint8_t *)dst + (ir * dst_row_size_aligned);

        hvx_step_f16_aa((uint8_t *) dst_local, (ثابت uint8_t *) src_local, ne0);
    }
}

ساكن فراغ l2_norm_f16(ثابت فراغ * restrict src,
                        فراغ * restrict dst,
                        ثابت uint32_t ژمارە_rows,
                        ثابت بنية htp_unary_context * uctx) {
    htp_unary_op_preamble;
    عائم epsilon = 0.f;
    memcpy(&epsilon, op_params, حجم(عائم));

    لكل (uint32_t ir = 0; ir < ژمارە_rows; ir++) {
        ثابت uint8_t * restrict src_f = (ثابت uint8_t *)src + (ir * src0_row_size_aligned);
        uint8_t * restrict dst_f       = (uint8_t *)dst + (ir * dst_row_size_aligned);

        hvx_fast_l2_norm_f16((ثابت uint8_t *)src_f, (uint8_t *)dst_f, ne0, epsilon);
    }
}

ساكن فراغ neg_f32(ثابت فراغ * restrict src,
                    فراغ * restrict dst,
                    ثابت uint32_t ژمارە_rows,
                    ثابت بنية htp_unary_context * uctx) {
    htp_unary_op_preamble;

    لكل (uint32_t ir = 0; ir < ژمارە_rows; ir++) {
        ثابت uint8_t * restrict src_local = (ثابت uint8_t *)src + (ir * src0_row_size_aligned);
        uint8_t * restrict dst_local       = (uint8_t *)dst + (ir * dst_row_size_aligned);

        hvx_scale_f32_aa(dst_local, src_local, ne0, -1.0f);
    }
}

ساكن فراغ exp_f32(ثابت فراغ * restrict src,
                    فراغ * restrict dst,
                    ثابت uint32_t ژمارە_rows,
                    ثابت بنية htp_unary_context * uctx) {
    htp_unary_op_preamble;

    لكل (uint32_t ir = 0; ir < ژمارە_rows; ir++) {
        ثابت uint8_t * restrict src_local = (ثابت uint8_t *)src + (ir * src0_row_size_aligned);
        uint8_t * restrict dst_local       = (uint8_t *)dst + (ir * dst_row_size_aligned);

        hvx_exp_f32(dst_local, src_local, ne0, false);
    }
}

ساكن فراغ sigmoid_f32(ثابت فراغ * restrict src,
                        فراغ * restrict dst,
                        ثابت uint32_t ژمارە_rows,
                        ثابت بنية htp_unary_context * uctx) {
    htp_unary_op_preamble;

    لكل (uint32_t ir = 0; ir < ژمارە_rows; ir++) {
        ثابت uint8_t * restrict src_local = (ثابت uint8_t *)src + (ir * src0_row_size_aligned);
        uint8_t * restrict dst_local       = (uint8_t *)dst + (ir * dst_row_size_aligned);

        hvx_sigmoid_f32_aa(dst_local, src_local, ne0);
    }
}

// silu(x) = x * sigmoid(x)
ساكن فراغ silu_f32(ثابت فراغ * restrict src,
                     فراغ * restrict dst,
                     ثابت uint32_t ژمارە_rows,
                     ثابت بنية htp_unary_context * uctx) {
    htp_unary_op_preamble;

    لكل (uint32_t ir = 0; ir < ژمارە_rows; ir++) {
        ثابت uint8_t * restrict src_local = (ثابت uint8_t *)src + (ir * src0_row_size_aligned);
        uint8_t * restrict dst_local       = (uint8_t *)dst + (ir * dst_row_size_aligned);

        hvx_sigmoid_f32_aa(dst_local, src_local, ne0);
        hvx_mul_f32_aaa(dst_local, src_local, dst_local, ne0);
    }
}

// GELU uses the tanh approximation.
ساكن __attribute__((noinline)) HVX_Vector hvx_vec_gelu_f32(HVX_Vector x) {
    ثابت HVX_Vector half = hvx_vec_splat_f32(0.5f);
    ثابت HVX_Vector one  = hvx_vec_splat_f32(1.0f);

    HVX_Vector inner = hvx_vec_mul_f32_f32(x, x);
    inner = hvx_vec_mul_f32_f32(inner, hvx_vec_splat_f32(0.044715f));
    inner = hvx_vec_add_f32_f32(inner, one);
    inner = hvx_vec_mul_f32_f32(inner, x);
    inner = hvx_vec_mul_f32_f32(inner, hvx_vec_splat_f32(0.7978845608028654f));

    أرجع hvx_vec_mul_f32_f32(hvx_vec_mul_f32_f32(half, x),
                               hvx_vec_add_f32_f32(one, hvx_vec_tanh_f32(inner)));
}

ساكن مضمن فراغ hvx_gelu_f32_aa(uint8_t * restrict dst, ثابت uint8_t * restrict src, uint32_t n) {
    أكد((unsigned long) dst % 128 == 0);
    أكد((unsigned long) src % 128 == 0);

    HVX_Vector * restrict vdst = (HVX_Vector *) dst;
    HVX_Vector * restrict vsrc = (HVX_Vector *) src;

    ثابت uint32_t nvec = n / VLEN_FP32;
    ثابت uint32_t nloe = n % VLEN_FP32;

    uint32_t i = 0;
    _Pragma("unroll(4)")
    لكل (; i < nvec; i++) {
        vdst[i] = hvx_vec_gelu_f32(vsrc[i]);
    }
    إذا (nloe) {
        hvx_vec_store_a(&vdst[i], nloe * حجم(عائم), hvx_vec_gelu_f32(vsrc[i]));
    }
}

// GELU_QUICK uses x * sigmoid(1.702 * x).
ساكن __attribute__((noinline)) HVX_Vector hvx_vec_gelu_quick_f32(HVX_Vector x) {
    ثابت HVX_Vector one     = hvx_vec_splat_f32(1.0f);
    ثابت HVX_Vector max_exp = hvx_vec_splat_f32(87.0f);
    ثابت HVX_Vector min_exp = hvx_vec_splat_f32(-87.0f);
    ثابت HVX_Vector scaled  = hvx_vec_mul_f32_f32(x, hvx_vec_splat_f32(1.702f));
    ثابت HVX_Vector sigmoid = hvx_vec_fast_sigmoid_f32_guard(scaled, one, max_exp, min_exp);

    أرجع hvx_vec_mul_f32_f32(x, sigmoid);
}

ساكن مضمن فراغ hvx_gelu_quick_f32_aa(uint8_t * restrict dst, ثابت uint8_t * restrict src, uint32_t n) {
    أكد((unsigned long) dst % 128 == 0);
    أكد((unsigned long) src % 128 == 0);

    HVX_Vector * restrict vdst = (HVX_Vector *) dst;
    HVX_Vector * restrict vsrc = (HVX_Vector *) src;

    ثابت uint32_t nvec = n / VLEN_FP32;
    ثابت uint32_t nloe = n % VLEN_FP32;

    uint32_t i = 0;
    _Pragma("unroll(4)")
    لكل (; i < nvec; i++) {
        vdst[i] = hvx_vec_gelu_quick_f32(vsrc[i]);
    }
    إذا (nloe) {
        hvx_vec_store_a(&vdst[i], nloe * حجم(عائم), hvx_vec_gelu_quick_f32(vsrc[i]));
    }
}

ساكن فراغ gelu_f32(ثابت فراغ * restrict src,
                     فراغ * restrict dst,
                     ثابت uint32_t ژمارە_rows,
                     ثابت بنية htp_unary_context * uctx) {
    htp_unary_op_preamble;

    لكل (uint32_t ir = 0; ir < ژمارە_rows; ir++) {
        ثابت uint8_t * restrict src_local = (ثابت uint8_t *)src + (ir * src0_row_size_aligned);
        uint8_t * restrict dst_local       = (uint8_t *)dst + (ir * dst_row_size_aligned);

        hvx_gelu_f32_aa(dst_local, src_local, ne0);
    }
}

ساكن فراغ gelu_quick_f32(ثابت فراغ * restrict src,
                           فراغ * restrict dst,
                           ثابت uint32_t ژمارە_rows,
                           ثابت بنية htp_unary_context * uctx) {
    htp_unary_op_preamble;

    لكل (uint32_t ir = 0; ir < ژمارە_rows; ir++) {
        ثابت uint8_t * restrict src_local = (ثابت uint8_t *)src + (ir * src0_row_size_aligned);
        uint8_t * restrict dst_local       = (uint8_t *)dst + (ir * dst_row_size_aligned);

        hvx_gelu_quick_f32_aa(dst_local, src_local, ne0);
    }
}

ساكن فراغ gelu_erf_f32(ثابت فراغ * restrict src,
                         فراغ * restrict dst,
                         ثابت uint32_t ژمارە_rows,
                         ثابت بنية htp_unary_context * uctx) {
    htp_unary_op_preamble;

    لكل (uint32_t ir = 0; ir < ژمارە_rows; ir++) {
        ثابت uint8_t * restrict src_local = (ثابت uint8_t *) src + (ir * src0_row_size_aligned);
        uint8_t * restrict dst_local       = (uint8_t *) dst + (ir * dst_row_size_aligned);

        hvx_gelu_erf_f32_aa(dst_local, src_local, ne0);
    }
}

ساكن فراغ tri_f32(ثابت فراغ * restrict src,
                    فراغ * restrict dst,
                    ثابت uint32_t ژمارە_rows,
                    ثابت uint32_t ir,
                    ثابت بنية htp_unary_context * uctx) {
    htp_unary_op_preamble;
    ثابت int32_t ttype = op_params[0];
    ثابت HVX_Vector zero = hvx_vec_splat_f32(0.0f);
    ثابت uint32_t nvec  = ne0 / VLEN_FP32;
    ثابت uint32_t nloe  = ne0 % VLEN_FP32;

    ثابت uint32_t ne01 = uctx->octx->src[0]->ne[1];

    لكل (uint32_t b = 0; b < ژمارە_rows; b++) {
        ثابت uint32_t abs_row = ir + b;
        ثابت uint32_t i01     = abs_row % ne01;

        ثابت HVX_Vector * restrict v_src = (ثابت HVX_Vector *) ((ثابت uint8_t *) src + b * src0_row_size_aligned);
        HVX_Vector * restrict v_dst       = (HVX_Vector *) ((uint8_t *) dst + b * dst_row_size_aligned);

        uint32_t boundary;
        صحيح      keep_چەپ;
        بدل (ttype) {
            حالة 0: boundary = i01;     keep_چەپ = 0; اكسر;  // keep col >= row
            حالة 1: boundary = i01 + 1; keep_چەپ = 0; اكسر;  // keep col > row
            حالة 2: boundary = i01 + 1; keep_چەپ = 1; اكسر;  // keep col <= row
            حالة 3: boundary = i01;     keep_چەپ = 1; اكسر;  // keep col < row
            افتراضي: boundary = 0; keep_چەپ = 0; اكسر;
        }
        إذا (boundary > ne0) boundary = ne0;

        // Full HVX vectors - each starts at a 128-byte aligned offset
        لكل (uint32_t i = 0; i < nvec; i++) {
            ثابت uint32_t vec_start = i * VLEN_FP32;
            ثابت uint32_t vec_end   = vec_start + VLEN_FP32;
            إذا (keep_چەپ) {
                إذا (vec_end <= boundary) {
                    v_dst[i] = v_src[i];
                } وإلا إذا (vec_start >= boundary) {
                    v_dst[i] = zero;
                } وإلا {
                    HVX_VectorPred mask = Q6_Q_vsetq_R((boundary - vec_start) * حجم(عائم));
                    v_dst[i]            = Q6_V_vmux_QVV(mask, v_src[i], zero);
                }
            } وإلا {
                إذا (vec_end <= boundary) {
                    v_dst[i] = zero;
                } وإلا إذا (vec_start >= boundary) {
                    v_dst[i] = v_src[i];
                } وإلا {
                    HVX_VectorPred mask = Q6_Q_vsetq_R((boundary - vec_start) * حجم(عائم));
                    v_dst[i]            = Q6_V_vmux_QVV(mask, zero, v_src[i]);
                }
            }
        }

        // Tail elements (row_elems not a multiple of VLEN_FP32)
        إذا (nloe > 0) {
            ثابت uint32_t abs_start = nvec * VLEN_FP32;
            ثابت uint32_t abs_end   = abs_start + nloe;
            HVX_Vector     tail_val;
            إذا (keep_چەپ) {
                إذا (abs_end <= boundary) {
                    tail_val = v_src[nvec];
                } وإلا إذا (abs_start >= boundary) {
                    tail_val = zero;
                } وإلا {
                    HVX_VectorPred mask = Q6_Q_vsetq_R((boundary - abs_start) * حجم(عائم));
                    tail_val            = Q6_V_vmux_QVV(mask, v_src[nvec], zero);
                }
            } وإلا {
                إذا (abs_end <= boundary) {
                    tail_val = zero;
                } وإلا إذا (abs_start >= boundary) {
                    tail_val = v_src[nvec];
                } وإلا {
                    HVX_VectorPred mask = Q6_Q_vsetq_R((boundary - abs_start) * حجم(عائم));
                    tail_val            = Q6_V_vmux_QVV(mask, zero, v_src[nvec]);
                }
            }
            hvx_vec_store_a(&v_dst[nvec], nloe * حجم(عائم), tail_val);
        }
    }
}

// softplus(x) = log(1 + exp(x))
// Match CPU reference: return x for x > 20 to avoid overflow.
ساكن مضمن فراغ hvx_softplus_f32_aa(uint8_t * restrict dst, ثابت uint8_t * restrict src, uint32_t n) {
    ثابت HVX_Vector * restrict v_src = (ثابت HVX_Vector *) src;
    HVX_Vector * restrict v_dst       = (HVX_Vector *) dst;

    ثابت HVX_Vector twenty = hvx_vec_splat_f32(20.0f);
    ثابت HVX_Vector one    = hvx_vec_splat_f32(1.0f);

    ثابت uint32_t nvec = n / VLEN_FP32;
    ثابت uint32_t nloe = n % VLEN_FP32;

    #pragma unroll(4)
    لكل (uint32_t i = 0; i < nvec; i++) {
        HVX_Vector x              = v_src[i];
        HVX_VectorPred pred_gt20  = Q6_Q_vcmp_gt_VsfVsf(x, twenty);
        HVX_Vector expx           = hvx_vec_exp_f32(x);
        HVX_Vector logv           = hvx_vec_log_f32(hvx_vec_add_f32_f32(one, expx));
        v_dst[i]                  = Q6_V_vmux_QVV(pred_gt20, x, logv);
    }

    إذا (nloe > 0) {
        HVX_Vector x             = v_src[nvec];
        HVX_VectorPred pred_gt20 = Q6_Q_vcmp_gt_VsfVsf(x, twenty);
        HVX_Vector expx          = hvx_vec_exp_f32(x);
        HVX_Vector logv          = hvx_vec_log_f32(hvx_vec_add_f32_f32(one, expx));
        HVX_Vector sp            = Q6_V_vmux_QVV(pred_gt20, x, logv);
        hvx_vec_store_a((فراغ *) &v_dst[nvec], nloe * حجم(عائم), sp);
    }
}

ساكن فراغ softplus_f32(ثابت فراغ * restrict src,
                         فراغ * restrict dst,
                         ثابت uint32_t ژمارە_rows,
                         ثابت بنية htp_unary_context * uctx) {
    htp_unary_op_preamble;

    لكل (uint32_t ir = 0; ir < ژمارە_rows; ir++) {
        ثابت uint8_t * restrict src_local = (ثابت uint8_t *)src + (ir * src0_row_size_aligned);
        uint8_t * restrict dst_local       = (uint8_t *)dst + (ir * dst_row_size_aligned);

        hvx_softplus_f32_aa(dst_local, src_local, ne0);
    }
}

ساكن فراغ l2_norm_f32(ثابت فراغ * restrict src,
                        فراغ * restrict dst,
                        ثابت uint32_t ژمارە_rows,
                        ثابت بنية htp_unary_context * uctx) {
    htp_unary_op_preamble;
    عائم epsilon = 0.f;
    memcpy(&epsilon, op_params, حجم(عائم));

    لكل (uint32_t ir = 0; ir < ژمارە_rows; ir++) {
        ثابت uint8_t * restrict src_f = (ثابت uint8_t *)src + (ir * src0_row_size_aligned);
        uint8_t * restrict dst_f       = (uint8_t *)dst + (ir * dst_row_size_aligned);

        hvx_fast_l2_norm_f32((ثابت uint8_t *)src_f, (uint8_t *)dst_f, ne0, epsilon);
    }
}

ساكن مضمن HVX_VectorPair hvx_fwht_stage_pair(HVX_Vector v0, HVX_Vector v1, int32_t b) {
    HVX_VectorPair p = Q6_W_vshuff_VVR(v1, v0, b);
    HVX_Vector s = Q6_Vqf32_vadd_Vqf32Vqf32(Q6_V_lo_W(p), Q6_V_hi_W(p));
    HVX_Vector d = Q6_Vqf32_vsub_Vqf32Vqf32(Q6_V_lo_W(p), Q6_V_hi_W(p));
    أرجع Q6_W_vshuff_VVR(d, s, b);
}

ساكن مضمن HVX_VectorPair hvx_fwht32_pair(HVX_Vector v0_sf, HVX_Vector v1_sf, HVX_Vector v_scale) {
    HVX_Vector v0 = Q6_Vqf32_vmpy_VsfVsf(v0_sf, v_scale);
    HVX_Vector v1 = Q6_Vqf32_vmpy_VsfVsf(v1_sf, v_scale);
    HVX_VectorPair p;
    p = hvx_fwht_stage_pair(v0, v1, 4);
    p = hvx_fwht_stage_pair(Q6_V_lo_W(p), Q6_V_hi_W(p), 8);
    p = hvx_fwht_stage_pair(Q6_V_lo_W(p), Q6_V_hi_W(p), 16);
    p = hvx_fwht_stage_pair(Q6_V_lo_W(p), Q6_V_hi_W(p), 32);
    p = hvx_fwht_stage_pair(Q6_V_lo_W(p), Q6_V_hi_W(p), 64);
    أرجع p;
}

ساكن مضمن فراغ hvx_fwht_store32(HVX_Vector * restrict v_dst, HVX_VectorPair p) {
    v_dst[0] = Q6_Vsf_equals_Vqf32(Q6_V_lo_W(p));
}

ساكن مضمن فراغ hvx_fwht_store64(HVX_Vector * restrict v_dst, HVX_VectorPair p) {
    HVX_Vector u = Q6_V_lo_W(p);
    HVX_Vector v = Q6_V_hi_W(p);
    v_dst[0] = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vqf32(u, v));
    v_dst[1] = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vsub_Vqf32Vqf32(u, v));
}

ساكن مضمن فراغ hvx_fwht_tail_f32(HVX_Vector * restrict v_dst, uint32_t nvec) {
    uint32_t len_vec = 2;
    لكل (; len_vec * 2 < nvec; len_vec <<= 2) {
        ثابت uint32_t step = len_vec * 4;
        لكل (uint32_t i = 0; i < nvec; i += step) {
            لكل (uint32_t j = 0; j < len_vec; j++) {
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

    إذا (len_vec < nvec) {
        لكل (uint32_t i = 0; i < nvec; i += 2 * len_vec) {
            لكل (uint32_t j = 0; j < len_vec; j++) {
                HVX_Vector u = v_dst[i + j];
                HVX_Vector v = v_dst[i + len_vec + j];
                v_dst[i + j]           = hvx_vec_add_f32_f32(u, v);
                v_dst[i + len_vec + j] = hvx_vec_sub_f32_f32(u, v);
            }
        }
    }
}

ساكن مضمن فراغ hvx_fwht_row_f32(ثابت uint8_t * restrict src, uint8_t * restrict dst, uint32_t n, HVX_Vector v_scale) {
    ثابت HVX_Vector * restrict v_src = (ثابت HVX_Vector *) src;
    HVX_Vector * restrict v_dst       = (HVX_Vector *) dst;
    ثابت uint32_t nvec               = n / VLEN_FP32;

    إذا (nvec == 1) {
        HVX_VectorPair p = hvx_fwht32_pair(v_src[0], v_src[0], v_scale);
        hvx_fwht_store32(v_dst, p);
        أرجع;
    }

    لكل (uint32_t i = 0; i < nvec; i += 2) {
        HVX_VectorPair p = hvx_fwht32_pair(v_src[i], v_src[i + 1], v_scale);
        hvx_fwht_store64(&v_dst[i], p);
    }

    hvx_fwht_tail_f32(v_dst, nvec);
}

ساكن مضمن فراغ hvx_fwht_row_f16(ثابت uint8_t * restrict src, uint8_t * restrict dst, uint32_t n, HVX_Vector v_scale) {
    ثابت HVX_Vector * restrict v_src = (ثابت HVX_Vector *) src;
    HVX_Vector * restrict v_dst       = (HVX_Vector *) dst;
    ثابت uint32_t nvec               = n / VLEN_FP32;

    إذا (nvec == 1) {
        HVX_Vector v0 = Q6_V_lo_W(hvx_vec_f16_to_f32(v_src[0]));
        HVX_VectorPair p = hvx_fwht32_pair(v0, v0, v_scale);
        hvx_fwht_store32(v_dst, p);
        أرجع;
    }

    ثابت uint32_t nvec_f16 = n / VLEN_FP16;
    لكل (uint32_t i = 0; i < nvec_f16; i++) {
        HVX_VectorPair f32p = hvx_vec_f16_to_f32(v_src[i]);
        HVX_VectorPair p    = hvx_fwht32_pair(Q6_V_lo_W(f32p), Q6_V_hi_W(f32p), v_scale);
        hvx_fwht_store64(&v_dst[2 * i], p);
    }

    hvx_fwht_tail_f32(v_dst, nvec);
}

ساكن فراغ hadamard_f32(ثابت فراغ * restrict src,
                         فراغ * restrict dst,
                         ثابت uint32_t ژمارە_rows,
                         ثابت بنية htp_unary_context * uctx) {
    htp_unary_op_preamble;
    ثابت HVX_Vector v_scale = hvx_vec_splat_f32(1.0f / sqrtf((عائم) ne0));

    لكل (uint32_t ir = 0; ir < ژمارە_rows; ir++) {
        ثابت uint8_t * restrict src_row = (ثابت uint8_t *) src + (ir * src0_row_size_aligned);
        uint8_t * restrict dst_row       = (uint8_t *) dst + (ir * dst_row_size_aligned);

        hvx_fwht_row_f32(src_row, dst_row, ne0, v_scale);
    }
}

ساكن فراغ hadamard_f16(ثابت فراغ * restrict src,
                         فراغ * restrict dst,
                         ثابت uint32_t ژمارە_rows,
                         ثابت بنية htp_unary_context * uctx) {
    htp_unary_op_preamble;
    ثابت HVX_Vector v_scale = hvx_vec_splat_f32(1.0f / sqrtf((عائم) ne0));

    لكل (uint32_t ir = 0; ir < ژمارە_rows; ir++) {
        ثابت uint8_t * restrict src_row = (ثابت uint8_t *) src + (ir * src0_row_size_aligned);
        uint8_t * restrict dst_row       = (uint8_t *) dst + (ir * dst_row_size_aligned);

        hvx_fwht_row_f16(src_row, dst_row, ne0, v_scale);
    }
}

ساكن فراغ tanh_f32(ثابت فراغ * restrict src,
                     فراغ * restrict dst,
                     ثابت uint32_t ژمارە_rows,
                     ثابت بنية htp_unary_context * uctx) {
    htp_unary_op_preamble;

    لكل (uint32_t ir = 0; ir < ژمارە_rows; ir++) {
        ثابت uint8_t * restrict src_local = (ثابت uint8_t *)src + (ir * src0_row_size_aligned);
        uint8_t * restrict dst_local       = (uint8_t *)dst + (ir * dst_row_size_aligned);

        hvx_tanh_f32_aa(dst_local, src_local, ne0);
    }
}

ساكن فراغ abs_f32(ثابت فراغ * restrict src,
                    فراغ * restrict dst,
                    ثابت uint32_t ژمارە_rows,
                    ثابت بنية htp_unary_context * uctx) {
    htp_unary_op_preamble;

    لكل (uint32_t ir = 0; ir < ژمارە_rows; ir++) {
        ثابت uint8_t * restrict src_local = (ثابت uint8_t *)src + (ir * src0_row_size_aligned);
        uint8_t * restrict dst_local       = (uint8_t *)dst + (ir * dst_row_size_aligned);

        hvx_abs_f32_aa(dst_local, src_local, ne0);
    }
}

ساكن فراغ relu_f32(ثابت فراغ * restrict src,
                     فراغ * restrict dst,
                     ثابت uint32_t ژمارە_rows,
                     ثابت بنية htp_unary_context * uctx) {
    htp_unary_op_preamble;

    لكل (uint32_t ir = 0; ir < ژمارە_rows; ir++) {
        ثابت uint8_t * restrict src_local = (ثابت uint8_t *)src + (ir * src0_row_size_aligned);
        uint8_t * restrict dst_local       = (uint8_t *)dst + (ir * dst_row_size_aligned);

        hvx_max_scalar_f32(dst_local, src_local, 0.0f, ne0);
    }
}

ساكن فراغ step_f32(ثابت فراغ * restrict src,
                     فراغ * restrict dst,
                     ثابت uint32_t ژمارە_rows,
                     ثابت بنية htp_unary_context * uctx) {
    htp_unary_op_preamble;

    لكل (uint32_t ir = 0; ir < ژمارە_rows; ir++) {
        ثابت uint8_t * restrict src_local = (ثابت uint8_t *)src + (ir * src0_row_size_aligned);
        uint8_t * restrict dst_local       = (uint8_t *)dst + (ir * dst_row_size_aligned);

        hvx_step_f32_aa(dst_local, src_local, ne0);
    }
}

ساكن فراغ log_f32(ثابت فراغ * restrict src,
                    فراغ * restrict dst,
                    ثابت uint32_t ژمارە_rows,
                    ثابت بنية htp_unary_context * uctx) {
    htp_unary_op_preamble;

    لكل (uint32_t ir = 0; ir < ژمارە_rows; ir++) {
        ثابت uint8_t * restrict src_local = (ثابت uint8_t *)src + (ir * src0_row_size_aligned);
        uint8_t * restrict dst_local       = (uint8_t *)dst + (ir * dst_row_size_aligned);

        hvx_log_f32_aa(dst_local, src_local, ne0);
    }
}

#// Pointwise unary ops on one column tile in VTCM.
ساكن فراغ tile_scale_f32(فراغ * restrict dst, ثابت فراغ * restrict src, uint32_t tw, ثابت بنية htp_unary_context * uctx) {
    عائم scale = 0.f;
    عائم bias  = 0.f;
    memcpy(&scale, &uctx->octx->op_params[0], حجم(عائم));
    memcpy(&bias,  &uctx->octx->op_params[1], حجم(عائم));
    hvx_scale_offset_f32_aa((uint8_t *) dst, (ثابت uint8_t *) src, tw, scale, bias);
}

ساكن فراغ tile_clamp_f32(فراغ * restrict dst, ثابت فراغ * restrict src, uint32_t tw, ثابت بنية htp_unary_context * uctx) {
    عائم min = 0.f;
    عائم max = 0.f;
    memcpy(&min, &uctx->octx->op_params[0], حجم(عائم));
    memcpy(&max, &uctx->octx->op_params[1], حجم(عائم));
    hvx_clamp_scalar_f32((uint8_t *) dst, (ثابت uint8_t *) src, min, max, tw);
}

ساكن فراغ tile_leaky_relu_f32(فراغ * restrict dst, ثابت فراغ * restrict src, uint32_t tw, ثابت بنية htp_unary_context * uctx) {
    عائم negative_slope = 0.f;
    memcpy(&negative_slope, &uctx->octx->op_params[0], حجم(عائم));
    hvx_leaky_relu_scalar_f32((uint8_t *) dst, (ثابت uint8_t *) src, negative_slope, tw);
}

ساكن فراغ tile_sqr_f32(فراغ * restrict dst, ثابت فراغ * restrict src, uint32_t tw, ثابت بنية htp_unary_context * uctx) {
    (فراغ) uctx;
    hvx_sqr_f32_aa((uint8_t *) dst, (ثابت uint8_t *) src, tw);
}

ساكن فراغ tile_sqrt_f32(فراغ * restrict dst, ثابت فراغ * restrict src, uint32_t tw, ثابت بنية htp_unary_context * uctx) {
    (فراغ) uctx;
    hvx_sqrt_f32_aa((uint8_t *) dst, (ثابت uint8_t *) src, tw);
}

ساكن فراغ tile_neg_f32(فراغ * restrict dst, ثابت فراغ * restrict src, uint32_t tw, ثابت بنية htp_unary_context * uctx) {
    (فراغ) uctx;
    hvx_scale_f32_aa((uint8_t *) dst, (ثابت uint8_t *) src, tw, -1.0f);
}

ساكن فراغ tile_exp_f32(فراغ * restrict dst, ثابت فراغ * restrict src, uint32_t tw, ثابت بنية htp_unary_context * uctx) {
    (فراغ) uctx;
    hvx_exp_f32((uint8_t *) dst, (ثابت uint8_t *) src, tw, false);
}

ساكن فراغ tile_sigmoid_f32(فراغ * restrict dst, ثابت فراغ * restrict src, uint32_t tw, ثابت بنية htp_unary_context * uctx) {
    (فراغ) uctx;
    hvx_sigmoid_f32_aa((uint8_t *) dst, (ثابت uint8_t *) src, tw);
}

ساكن فراغ tile_silu_f32(فراغ * restrict dst, ثابت فراغ * restrict src, uint32_t tw, ثابت بنية htp_unary_context * uctx) {
    (فراغ) uctx;
    hvx_sigmoid_f32_aa((uint8_t *) dst, (ثابت uint8_t *) src, tw);
    hvx_mul_f32_aaa((uint8_t *) dst, (ثابت uint8_t *) src, (uint8_t *) dst, tw);
}

ساكن فراغ tile_gelu_f32(فراغ * restrict dst, ثابت فراغ * restrict src, uint32_t tw, ثابت بنية htp_unary_context * uctx) {
    (فراغ) uctx;
    hvx_gelu_f32_aa((uint8_t *) dst, (ثابت uint8_t *) src, tw);
}

ساكن فراغ tile_gelu_quick_f32(فراغ * restrict dst, ثابت فراغ * restrict src, uint32_t tw, ثابت بنية htp_unary_context * uctx) {
    (فراغ) uctx;
    hvx_gelu_quick_f32_aa((uint8_t *) dst, (ثابت uint8_t *) src, tw);
}

ساكن فراغ tile_gelu_erf_f32(فراغ * restrict dst, ثابت فراغ * restrict src, uint32_t tw, ثابت بنية htp_unary_context * uctx) {
    (فراغ) uctx;
    hvx_gelu_erf_f32_aa((uint8_t *) dst, (ثابت uint8_t *) src, tw);
}

ساكن فراغ tile_softplus_f32(فراغ * restrict dst, ثابت فراغ * restrict src, uint32_t tw, ثابت بنية htp_unary_context * uctx) {
    (فراغ) uctx;
    hvx_softplus_f32_aa((uint8_t *) dst, (ثابت uint8_t *) src, tw);
}

ساكن فراغ tile_tanh_f32(فراغ * restrict dst, ثابت فراغ * restrict src, uint32_t tw, ثابت بنية htp_unary_context * uctx) {
    (فراغ) uctx;
    hvx_tanh_f32_aa((uint8_t *) dst, (ثابت uint8_t *) src, tw);
}

ساكن فراغ tile_abs_f32(فراغ * restrict dst, ثابت فراغ * restrict src, uint32_t tw, ثابت بنية htp_unary_context * uctx) {
    (فراغ) uctx;
    hvx_abs_f32_aa((uint8_t *) dst, (ثابت uint8_t *) src, tw);
}

ساكن فراغ tile_log_f32(فراغ * restrict dst, ثابت فراغ * restrict src, uint32_t tw, ثابت بنية htp_unary_context * uctx) {
    (فراغ) uctx;
    hvx_log_f32_aa((uint8_t *) dst, (ثابت uint8_t *) src, tw);
}

ساكن فراغ tile_relu_f32(فراغ * restrict dst, ثابت فراغ * restrict src, uint32_t tw, ثابت بنية htp_unary_context * uctx) {
    (فراغ) uctx;
    hvx_max_scalar_f32((uint8_t *) dst, (ثابت uint8_t *) src, 0.0f, tw);
}

ساكن فراغ tile_step_f32(فراغ * restrict dst, ثابت فراغ * restrict src, uint32_t tw, ثابت بنية htp_unary_context * uctx) {
    (فراغ) uctx;
    hvx_step_f32_aa((uint8_t *) dst, (ثابت uint8_t *) src, tw);
}

ساكن فراغ tri_apply_tile_f32(ثابت فراغ * restrict src, فراغ * restrict dst,
                               uint32_t tile_elems, uint32_t col_start, uint32_t i01,
                               uint32_t ne0, int32_t ttype) {
    ثابت HVX_Vector * restrict v_src = (ثابت HVX_Vector *) src;
    HVX_Vector * restrict v_dst       = (HVX_Vector *) dst;
    ثابت HVX_Vector zero = hvx_vec_splat_f32(0.0f);

    uint32_t boundary;
    صحيح      keep_چەپ;
    بدل (ttype) {
        حالة 0: boundary = i01;     keep_چەپ = 0; اكسر;
        حالة 1: boundary = i01 + 1; keep_چەپ = 0; اكسر;
        حالة 2: boundary = i01 + 1; keep_چەپ = 1; اكسر;
        حالة 3: boundary = i01;     keep_چەپ = 1; اكسر;
        افتراضي: boundary = 0; keep_چەپ = 0; اكسر;
    }
    إذا (boundary > ne0) boundary = ne0;

    ثابت uint32_t nvec = tile_elems / VLEN_FP32;
    ثابت uint32_t nloe = tile_elems % VLEN_FP32;

    لكل (uint32_t i = 0; i < nvec; i++) {
        ثابت uint32_t abs_start = col_start + i * VLEN_FP32;
        ثابت uint32_t abs_end   = abs_start + VLEN_FP32;
        إذا (keep_چەپ) {
            إذا (abs_end <= boundary) {
                v_dst[i] = v_src[i];
            } وإلا إذا (abs_start >= boundary) {
                v_dst[i] = zero;
            } وإلا {
                HVX_VectorPred mask = Q6_Q_vsetq_R((boundary - abs_start) * حجم(عائم));
                v_dst[i]            = Q6_V_vmux_QVV(mask, v_src[i], zero);
            }
        } وإلا {
            إذا (abs_end <= boundary) {
                v_dst[i] = zero;
            } وإلا إذا (abs_start >= boundary) {
                v_dst[i] = v_src[i];
            } وإلا {
                HVX_VectorPred mask = Q6_Q_vsetq_R((boundary - abs_start) * حجم(عائم));
                v_dst[i]            = Q6_V_vmux_QVV(mask, zero, v_src[i]);
            }
        }
    }

    إذا (nloe > 0) {
        ثابت uint32_t abs_start = col_start + nvec * VLEN_FP32;
        ثابت uint32_t abs_end   = abs_start + nloe;
        HVX_Vector     tail_val;
        إذا (keep_چەپ) {
            إذا (abs_end <= boundary) {
                tail_val = v_src[nvec];
            } وإلا إذا (abs_start >= boundary) {
                tail_val = zero;
            } وإلا {
                HVX_VectorPred mask = Q6_Q_vsetq_R((boundary - abs_start) * حجم(عائم));
                tail_val            = Q6_V_vmux_QVV(mask, v_src[nvec], zero);
            }
        } وإلا {
            إذا (abs_end <= boundary) {
                tail_val = zero;
            } وإلا إذا (abs_start >= boundary) {
                tail_val = v_src[nvec];
            } وإلا {
                HVX_VectorPred mask = Q6_Q_vsetq_R((boundary - abs_start) * حجم(عائم));
                tail_val            = Q6_V_vmux_QVV(mask, zero, v_src[nvec]);
            }
        }
        hvx_vec_store_a(&v_dst[nvec], nloe * حجم(عائم), tail_val);
    }
}

// 1. Standard row-block unary task (F32 and F16).
ساكن فراغ unary_thread_row_block(unsigned صحيح nth, unsigned صحيح ith, فراغ * data) {
    (فراغ) nth;
    ثابت بنية htp_unary_context * uctx = (ثابت بنية htp_unary_context *) data;
    بنية htp_ops_context * octx = uctx->octx;
    ثابت بنية htp_tensor * src = octx->src[0];
    ثابت بنية htp_tensor * dst = octx->dst;
    بنية htp_thread_trace * tr = &octx->ctx->trace[ith];

    htp_unary_preamble;

    ثابت uint32_t src0_nrows_per_thread = uctx->src0_nrows_per_thread;
    ثابت size_t src0_data_row_size = uctx->src0_data_row_size;
    ثابت size_t dst_data_row_size  = uctx->dst_data_row_size;
    ثابت size_t src0_row_size_aligned = uctx->src0_row_size_aligned;
    ثابت size_t dst_row_size_aligned  = uctx->dst_row_size_aligned;

    ثابت uint32_t src0_nrows = uctx->src0_nrows;
    ثابت uint32_t src0_start_row = uctx->row_start + src0_nrows_per_thread * ith;
    ثابت uint32_t src0_end_row   = MIN(src0_start_row + src0_nrows_per_thread, uctx->row_start + src0_nrows);

    إذا (src0_start_row >= src0_end_row) {
        أرجع;
    }

    ثابت dma_addr_t data_src = uctx->data_src0;
    ثابت dma_addr_t data_dst = uctx->data_dst;

    uint8_t * src0_vtcm_data = uctx->vtcm_src0 + (ith * uctx->vtcm_src0_size_per_thread);
    uint8_t * dst_vtcm_data  = uctx->vtcm_dst + (ith * uctx->vtcm_dst_size_per_thread);

    ثابت size_t src0_vtcm_half_size = uctx->src0_vtcm_half_size;
    ثابت size_t dst_vtcm_half_size  = uctx->dst_vtcm_half_size;

    ثابت bool src0_contig = (nb02 == (size_t)ne01 * nb01) &&
                             (nb03 == (size_t)ne02 * nb02);
    ثابت bool dst_contig  = (nb2  == (size_t)ne1  * nb1)  &&
                             (nb3  == (size_t)ne2  * nb2);

    ثابت بنية fastdiv_بەهاکان * div_ne01  = &uctx->kparams->div_ne01;
    ثابت بنية fastdiv_بەهاکان * div_ne02  = &uctx->kparams->div_ne02;
    ثابت بنية fastdiv_بەهاکان * div_ne012 = &uctx->kparams->div_ne012;

    ثابت uint32_t src0_max_block = src0_contig ? uctx->block : MIN((uint32_t)uctx->block, ne01);
    ثابت uint32_t dst_max_block  = dst_contig  ? uctx->block : MIN((uint32_t)uctx->block, ne1);
    ثابت uint32_t BLOCK = MIN(src0_max_block, dst_max_block);
    إذا (BLOCK == 0) {
        FARF(ERROR, "unary-row-block : current VTCM reservation %zu is too small, needed at least %zu\n",
             uctx->vtcm_src0_size_per_thread, src0_row_size_aligned);
        أرجع;
    }

    dma_queue * dma_q = octx->ctx->dma[ith];

    لكل (uint32_t ir = src0_start_row, vtcm_idx = 0; ir < src0_end_row && vtcm_idx < 2; vtcm_idx++) {
        ثابت uint32_t block_size = unary_block_size(ir, src0_end_row, BLOCK, src0_contig, dst_contig,
                                                     ne01, div_ne01);

        dma_queue_پاڵنان(dma_q,
            dma_make_data(data_dst, dst_vtcm_data + (vtcm_idx * dst_vtcm_half_size)),
            nb1, dst_row_size_aligned, dst_data_row_size, 0);

        ثابت size_t src0_off = src0_contig ? (ir * nb01) :
            unary_row_offset(ir, ne01, ne02, div_ne01, div_ne02, div_ne012, nb01, nb02, nb03);
        dma_queue_پاڵنان(dma_q,
            dma_make_data(src0_vtcm_data + (vtcm_idx * src0_vtcm_half_size), data_src + src0_off),
            src0_row_size_aligned, nb01, src0_data_row_size, block_size);

        ir += block_size;
    }

    unary_compute_fn_t compute = (unary_compute_fn_t) uctx->compute;

    لكل (uint32_t ir = src0_start_row; ir < src0_end_row; ) {
        ثابت uint32_t block_size = unary_block_size(ir, src0_end_row, BLOCK, src0_contig, dst_contig,
                                                     ne01, div_ne01);

        فراغ * dst_vtcm  = (فراغ *) (uintptr_t) dma_queue_دەرهێنان(dma_q).src;
        فراغ * src0_vtcm = (فراغ *) (uintptr_t) dma_queue_دەرهێنان(dma_q).dst;

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, ir);
        compute(src0_vtcm, dst_vtcm, block_size, uctx);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ir);

        ثابت size_t dst_off = dst_contig ? (ir * nb1) :
            unary_row_offset(ir, ne1, ne2, div_ne01, div_ne02, div_ne012, nb1, nb2, nb3);
        dma_queue_پاڵنان(dma_q,
            dma_make_data(data_dst + dst_off, dst_vtcm),
            nb1, dst_row_size_aligned, dst_data_row_size, block_size);

        ثابت uint32_t next_ir = ir + block_size;
        إذا (next_ir < src0_end_row) {
            ثابت uint32_t next_block_size = unary_block_size(next_ir, src0_end_row, BLOCK, src0_contig,
                                                              dst_contig, ne01, div_ne01);
            ثابت uint32_t pref_ir = next_ir + next_block_size;
            إذا (pref_ir < src0_end_row) {
                ثابت uint32_t pref_block_size = unary_block_size(pref_ir, src0_end_row, BLOCK, src0_contig,
                                                                  dst_contig, ne01, div_ne01);
                ثابت size_t src0_pref_off = src0_contig ? (pref_ir * nb01) :
                    unary_row_offset(pref_ir, ne01, ne02, div_ne01, div_ne02, div_ne012, nb01, nb02, nb03);
                dma_queue_پاڵنان(dma_q,
                    dma_make_data(src0_vtcm, data_src + src0_pref_off),
                    src0_row_size_aligned, nb01, src0_data_row_size, pref_block_size);
            }
        }
        ir += block_size;
    }

    dma_queue_flush(dma_q);
}

// 2. RMS_NORM_MUL row-block task with weight buffer.
ساكن فراغ unary_thread_rms_norm_mul_f32(unsigned صحيح nth, unsigned صحيح ith, فراغ * data) {
    (فراغ) nth;
    ثابت بنية htp_unary_context * uctx = (ثابت بنية htp_unary_context *) data;
    بنية htp_ops_context * octx = uctx->octx;
    ثابت بنية htp_tensor * src = octx->src[0];
    ثابت بنية htp_tensor * dst = octx->dst;
    بنية htp_thread_trace * tr = &octx->ctx->trace[ith];

    htp_unary_preamble;

    ثابت uint32_t src0_nrows_per_thread = uctx->src0_nrows_per_thread;
    ثابت size_t src0_data_row_size = uctx->src0_data_row_size;
    ثابت size_t dst_data_row_size  = uctx->dst_data_row_size;
    ثابت size_t src0_row_size_aligned = uctx->src0_row_size_aligned;
    ثابت size_t dst_row_size_aligned  = uctx->dst_row_size_aligned;

    ثابت uint32_t src0_nrows = uctx->src0_nrows;
    ثابت uint32_t src0_start_row = uctx->row_start + src0_nrows_per_thread * ith;
    ثابت uint32_t src0_end_row   = MIN(src0_start_row + src0_nrows_per_thread, uctx->row_start + src0_nrows);

    إذا (src0_start_row >= src0_end_row) {
        أرجع;
    }

    ثابت dma_addr_t data_src  = uctx->data_src0;
    ثابت dma_addr_t data_src1 = uctx->data_src1;
    ثابت dma_addr_t data_dst  = uctx->data_dst;

    ثابت بنية htp_tensor * src1 = octx->src[1];
    ثابت uint32_t nb11 = src1->nb[1];
    ثابت uint32_t nb12 = src1->nb[2];
    ثابت uint32_t nb13 = src1->nb[3];
    ثابت uint32_t nb11_bc = (src1->ne[1] > 1) ? nb11 : 0;
    ثابت uint32_t nb12_bc = (src1->ne[2] > 1) ? nb12 : 0;
    ثابت uint32_t nb13_bc = (src1->ne[3] > 1) ? nb13 : 0;
    ثابت bool src1_contig = ((nb12 == (size_t)ne01 * nb11) && (nb13 == (size_t)ne02 * nb12));

    uint8_t * src0_vtcm_data = uctx->vtcm_src0 + (ith * uctx->vtcm_src0_size_per_thread);
    uint8_t * src1_vtcm_data = uctx->vtcm_src1 ? (uctx->vtcm_src1 + (ith * uctx->vtcm_src1_size_per_thread)) : عدم;
    uint8_t * dst_vtcm_data  = uctx->vtcm_dst + (ith * uctx->vtcm_dst_size_per_thread);

    ثابت size_t src0_vtcm_half_size = uctx->src0_vtcm_half_size;
    ثابت size_t src1_vtcm_half_size = uctx->src1_vtcm_half_size;
    ثابت size_t dst_vtcm_half_size  = uctx->dst_vtcm_half_size;

    ثابت bool src0_contig = (nb02 == (size_t)ne01 * nb01) &&
                             (nb03 == (size_t)ne02 * nb02);
    ثابت bool dst_contig  = (nb2  == (size_t)ne1  * nb1)  &&
                             (nb3  == (size_t)ne2  * nb2);

    ثابت بنية fastdiv_بەهاکان * div_ne01  = &uctx->kparams->div_ne01;
    ثابت بنية fastdiv_بەهاکان * div_ne02  = &uctx->kparams->div_ne02;
    ثابت بنية fastdiv_بەهاکان * div_ne012 = &uctx->kparams->div_ne012;

    ثابت bool src1_needs_row_clip = !uctx->broadcast_weight && !src1_contig;
    ثابت bool block_src0_contig = src0_contig && !src1_needs_row_clip;
    ثابت bool block_dst_contig  = dst_contig  && !src1_needs_row_clip;

    ثابت uint32_t src0_max_block = block_src0_contig ? uctx->block : MIN((uint32_t)uctx->block, ne01);
    ثابت uint32_t dst_max_block  = block_dst_contig  ? uctx->block : MIN((uint32_t)uctx->block, ne1);
    ثابت uint32_t BLOCK = MIN(src0_max_block, dst_max_block);
    إذا (BLOCK == 0) {
        FARF(ERROR, "unary-rms-norm-mul : current VTCM reservation %zu is too small, needed at least %zu\n",
             uctx->vtcm_src0_size_per_thread, src0_row_size_aligned);
        أرجع;
    }

    dma_queue * dma_q = octx->ctx->dma[ith];

    إذا (uctx->broadcast_weight) {
        dma_queue_پاڵنان(dma_q, dma_make_data(src1_vtcm_data, data_src1),
                       uctx->src1_row_size_aligned, 0, uctx->src1_data_row_size, 1);
        dma_queue_flush(dma_q);
    }

    لكل (uint32_t ir = src0_start_row, vtcm_idx = 0; ir < src0_end_row && vtcm_idx < 2; vtcm_idx++) {
        ثابت uint32_t block_size = unary_block_size(ir, src0_end_row, BLOCK, block_src0_contig, block_dst_contig,
                                                     ne01, div_ne01);

        dma_queue_پاڵنان(dma_q,
            dma_make_data(data_dst, dst_vtcm_data + (vtcm_idx * dst_vtcm_half_size)),
            nb1, dst_row_size_aligned, dst_data_row_size, 0);

        ثابت size_t src0_off = src0_contig ? (ir * nb01) :
            unary_row_offset(ir, ne01, ne02, div_ne01, div_ne02, div_ne012, nb01, nb02, nb03);
        dma_queue_پاڵنان(dma_q,
            dma_make_data(src0_vtcm_data + (vtcm_idx * src0_vtcm_half_size), data_src + src0_off),
            src0_row_size_aligned, nb01, src0_data_row_size, block_size);

        إذا (!uctx->broadcast_weight) {
            ثابت size_t src1_off = src1_contig ? (ir * nb11) :
                unary_row_offset(ir, ne01, ne02, div_ne01, div_ne02, div_ne012, nb11_bc, nb12_bc, nb13_bc);
            dma_queue_پاڵنان(dma_q,
                dma_make_data(src1_vtcm_data + (vtcm_idx * src1_vtcm_half_size), data_src1 + src1_off),
                uctx->src1_row_size_aligned, nb11, uctx->src1_data_row_size, block_size);
        }

        ir += block_size;
    }

    unary_rms_norm_mul_compute_fn_t compute = (unary_rms_norm_mul_compute_fn_t) uctx->compute;

    لكل (uint32_t ir = src0_start_row; ir < src0_end_row; ) {
        ثابت uint32_t block_size = unary_block_size(ir, src0_end_row, BLOCK, block_src0_contig, block_dst_contig,
                                                     ne01, div_ne01);

        فراغ * dst_vtcm  = (فراغ *) (uintptr_t) dma_queue_دەرهێنان(dma_q).src;
        فراغ * src0_vtcm = (فراغ *) (uintptr_t) dma_queue_دەرهێنان(dma_q).dst;
        فراغ * src1_vtcm = عدم;
        إذا (!uctx->broadcast_weight) {
            src1_vtcm = (فراغ *) (uintptr_t) dma_queue_دەرهێنان(dma_q).dst;
        }

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, ir);
        ثابت فراغ * w = uctx->broadcast_weight ? (ثابت فراغ *) src1_vtcm_data : src1_vtcm;
        compute(src0_vtcm, w, dst_vtcm, block_size, uctx);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ir);

        ثابت size_t dst_off = dst_contig ? (ir * nb1) :
            unary_row_offset(ir, ne1, ne2, div_ne01, div_ne02, div_ne012, nb1, nb2, nb3);
        dma_queue_پاڵنان(dma_q,
            dma_make_data(data_dst + dst_off, dst_vtcm),
            nb1, dst_row_size_aligned, dst_data_row_size, block_size);

        ثابت uint32_t next_ir = ir + block_size;
        إذا (next_ir < src0_end_row) {
            ثابت uint32_t next_block_size = unary_block_size(next_ir, src0_end_row, BLOCK, block_src0_contig,
                                                              block_dst_contig, ne01, div_ne01);
            ثابت uint32_t pref_ir = next_ir + next_block_size;
            إذا (pref_ir < src0_end_row) {
                ثابت uint32_t pref_block_size = unary_block_size(pref_ir, src0_end_row, BLOCK, block_src0_contig,
                                                                  block_dst_contig, ne01, div_ne01);
                ثابت size_t src0_pref_off = src0_contig ? (pref_ir * nb01) :
                    unary_row_offset(pref_ir, ne01, ne02, div_ne01, div_ne02, div_ne012, nb01, nb02, nb03);
                dma_queue_پاڵنان(dma_q,
                    dma_make_data(src0_vtcm, data_src + src0_pref_off),
                    src0_row_size_aligned, nb01, src0_data_row_size, pref_block_size);

                إذا (!uctx->broadcast_weight) {
                    ثابت size_t src1_pref_off = src1_contig ? (pref_ir * nb11) :
                        unary_row_offset(pref_ir, ne01, ne02, div_ne01, div_ne02, div_ne012, nb11_bc, nb12_bc,
                                          nb13_bc);
                    dma_queue_پاڵنان(dma_q,
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
ساكن فراغ unary_thread_tri_f32(unsigned صحيح nth, unsigned صحيح ith, فراغ * data) {
    (فراغ) nth;
    ثابت بنية htp_unary_context * uctx = (ثابت بنية htp_unary_context *) data;
    بنية htp_ops_context * octx = uctx->octx;
    ثابت بنية htp_tensor * src = octx->src[0];
    ثابت بنية htp_tensor * dst = octx->dst;
    بنية htp_thread_trace * tr = &octx->ctx->trace[ith];

    htp_unary_preamble;

    ثابت uint32_t src0_nrows_per_thread = uctx->src0_nrows_per_thread;
    ثابت size_t src0_data_row_size = uctx->src0_data_row_size;
    ثابت size_t dst_data_row_size  = uctx->dst_data_row_size;
    ثابت size_t src0_row_size_aligned = uctx->src0_row_size_aligned;
    ثابت size_t dst_row_size_aligned  = uctx->dst_row_size_aligned;

    ثابت uint32_t src0_nrows = uctx->src0_nrows;
    ثابت uint32_t src0_start_row = uctx->row_start + src0_nrows_per_thread * ith;
    ثابت uint32_t src0_end_row   = MIN(src0_start_row + src0_nrows_per_thread, uctx->row_start + src0_nrows);

    إذا (src0_start_row >= src0_end_row) {
        أرجع;
    }

    ثابت dma_addr_t data_src = uctx->data_src0;
    ثابت dma_addr_t data_dst = uctx->data_dst;

    uint8_t * src0_vtcm_data = uctx->vtcm_src0 + (ith * uctx->vtcm_src0_size_per_thread);
    uint8_t * dst_vtcm_data  = uctx->vtcm_dst + (ith * uctx->vtcm_dst_size_per_thread);

    ثابت size_t src0_vtcm_half_size = uctx->src0_vtcm_half_size;
    ثابت size_t dst_vtcm_half_size  = uctx->dst_vtcm_half_size;

    ثابت bool src0_contig = (nb02 == (size_t)ne01 * nb01) &&
                             (nb03 == (size_t)ne02 * nb02);
    ثابت bool dst_contig  = (nb2  == (size_t)ne1  * nb1)  &&
                             (nb3  == (size_t)ne2  * nb2);

    ثابت بنية fastdiv_بەهاکان * div_ne01  = &uctx->kparams->div_ne01;
    ثابت بنية fastdiv_بەهاکان * div_ne02  = &uctx->kparams->div_ne02;
    ثابت بنية fastdiv_بەهاکان * div_ne012 = &uctx->kparams->div_ne012;

    ثابت uint32_t src0_max_block = src0_contig ? uctx->block : MIN((uint32_t)uctx->block, ne01);
    ثابت uint32_t dst_max_block  = dst_contig  ? uctx->block : MIN((uint32_t)uctx->block, ne1);
    ثابت uint32_t BLOCK = MIN(src0_max_block, dst_max_block);
    إذا (BLOCK == 0) {
        FARF(ERROR, "unary-tri : current VTCM reservation %zu is too small, needed at least %zu\n",
             uctx->vtcm_src0_size_per_thread, src0_row_size_aligned);
        أرجع;
    }

    dma_queue * dma_q = octx->ctx->dma[ith];

    لكل (uint32_t ir = src0_start_row, vtcm_idx = 0; ir < src0_end_row && vtcm_idx < 2; vtcm_idx++) {
        ثابت uint32_t block_size = unary_block_size(ir, src0_end_row, BLOCK, src0_contig, dst_contig,
                                                     ne01, div_ne01);

        dma_queue_پاڵنان(dma_q,
            dma_make_data(data_dst, dst_vtcm_data + (vtcm_idx * dst_vtcm_half_size)),
            nb1, dst_row_size_aligned, dst_data_row_size, 0);

        ثابت size_t src0_off = src0_contig ? (ir * nb01) :
            unary_row_offset(ir, ne01, ne02, div_ne01, div_ne02, div_ne012, nb01, nb02, nb03);
        dma_queue_پاڵنان(dma_q,
            dma_make_data(src0_vtcm_data + (vtcm_idx * src0_vtcm_half_size), data_src + src0_off),
            src0_row_size_aligned, nb01, src0_data_row_size, block_size);

        ir += block_size;
    }

    unary_tri_compute_fn_t compute = (unary_tri_compute_fn_t) uctx->compute;

    لكل (uint32_t ir = src0_start_row; ir < src0_end_row; ) {
        ثابت uint32_t block_size = unary_block_size(ir, src0_end_row, BLOCK, src0_contig, dst_contig,
                                                     ne01, div_ne01);

        فراغ * dst_vtcm  = (فراغ *) (uintptr_t) dma_queue_دەرهێنان(dma_q).src;
        فراغ * src0_vtcm = (فراغ *) (uintptr_t) dma_queue_دەرهێنان(dma_q).dst;

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, ir);
        compute(src0_vtcm, dst_vtcm, block_size, ir, uctx);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ir);

        ثابت size_t dst_off = dst_contig ? (ir * nb1) :
            unary_row_offset(ir, ne1, ne2, div_ne01, div_ne02, div_ne012, nb1, nb2, nb3);
        dma_queue_پاڵنان(dma_q,
            dma_make_data(data_dst + dst_off, dst_vtcm),
            nb1, dst_row_size_aligned, dst_data_row_size, block_size);

        ثابت uint32_t next_ir = ir + block_size;
        إذا (next_ir < src0_end_row) {
            ثابت uint32_t next_block_size = unary_block_size(next_ir, src0_end_row, BLOCK, src0_contig,
                                                              dst_contig, ne01, div_ne01);
            ثابت uint32_t pref_ir = next_ir + next_block_size;
            إذا (pref_ir < src0_end_row) {
                ثابت uint32_t pref_block_size = unary_block_size(pref_ir, src0_end_row, BLOCK, src0_contig,
                                                                  dst_contig, ne01, div_ne01);
                ثابت size_t src0_pref_off = src0_contig ? (pref_ir * nb01) :
                    unary_row_offset(pref_ir, ne01, ne02, div_ne01, div_ne02, div_ne012, nb01, nb02, nb03);
                dma_queue_پاڵنان(dma_q,
                    dma_make_data(src0_vtcm, data_src + src0_pref_off),
                    src0_row_size_aligned, nb01, src0_data_row_size, pref_block_size);
            }
        }
        ir += block_size;
    }

    dma_queue_flush(dma_q);
}

// 4. Pointwise tiled unary task.
ساكن فراغ unary_thread_tiled(unsigned صحيح nth, unsigned صحيح ith, فراغ * data) {
    (فراغ) nth;
    ثابت بنية htp_unary_context * uctx = (ثابت بنية htp_unary_context *) data;
    بنية htp_ops_context * octx = uctx->octx;
    ثابت بنية htp_tensor * src = octx->src[0];
    ثابت بنية htp_tensor * dst = octx->dst;
    بنية htp_thread_trace * tr = &octx->ctx->trace[ith];

    htp_unary_preamble;

    ثابت uint32_t src0_nrows_per_thread = uctx->src0_nrows_per_thread;
    ثابت uint32_t col_tile              = uctx->col_tile;

    ثابت uint32_t src0_nrows     = uctx->src0_nrows;
    ثابت uint32_t src0_start_row = uctx->row_start + src0_nrows_per_thread * ith;
    ثابت uint32_t src0_end_row   = MIN(src0_start_row + src0_nrows_per_thread, uctx->row_start + src0_nrows);

    إذا (src0_start_row >= src0_end_row) {
        أرجع;
    }

    ثابت dma_addr_t data_src = uctx->data_src0;
    ثابت dma_addr_t data_dst = uctx->data_dst;

    uint8_t * src0_vtcm_data = uctx->vtcm_src0 + (ith * uctx->vtcm_src0_size_per_thread);
    uint8_t * dst_vtcm_data  = uctx->vtcm_dst + (ith * uctx->vtcm_dst_size_per_thread);

    ثابت size_t src0_half = uctx->src0_vtcm_half_size;
    ثابت size_t dst_half  = uctx->dst_vtcm_half_size;

    dma_queue * dma_q = octx->ctx->dma[ith];

    ثابت بنية fastdiv_بەهاکان * div_ne01  = &uctx->kparams->div_ne01;
    ثابت بنية fastdiv_بەهاکان * div_ne02  = &uctx->kparams->div_ne02;
    ثابت بنية fastdiv_بەهاکان * div_ne012 = &uctx->kparams->div_ne012;
    ثابت بنية fastdiv_بەهاکان * div_tpr   = &uctx->kparams->div_tpr;

    ثابت uint32_t tiles_per_row = (ne0 + col_tile - 1) / col_tile;

    ثابت bool src0_contig = (nb02 == (size_t)ne01 * nb01) &&
                             (nb03 == (size_t)ne02 * nb02);
    ثابت bool dst_contig  = (nb2  == (size_t)ne1  * nb1)  &&
                             (nb3  == (size_t)ne2  * nb2);

    ثابت uint32_t total_tiles = (src0_end_row - src0_start_row) * tiles_per_row;

    لكل (uint32_t t = 0, vtcm_idx = 0; t < total_tiles && vtcm_idx < 2; t++, vtcm_idx++) {
        ثابت uint32_t row  = src0_start_row + t / tiles_per_row;
        ثابت uint32_t col  = (t % tiles_per_row) * col_tile;
        ثابت uint32_t tw   = MIN(col_tile, ne0 - col);
        ثابت size_t   tb   = (size_t) tw * حجم(عائم);
        ثابت size_t   soff = (src0_contig ? (row * nb01) :
                               unary_row_offset(row, ne01, ne02, div_ne01, div_ne02, div_ne012, nb01, nb02, nb03)) +
                               (size_t) col * حجم(عائم);

        dma_queue_پاڵنان(dma_q, dma_make_data(data_dst, dst_vtcm_data + (vtcm_idx * dst_half)), 0, 0, 0, 0);
        dma_queue_پاڵنان(dma_q, dma_make_data(src0_vtcm_data + (vtcm_idx * src0_half), data_src + soff), tb, tb, tb, 1);
    }

    unary_tile_compute_fn_t compute = (unary_tile_compute_fn_t) uctx->compute;

    uint32_t row = src0_start_row;
    uint32_t col = 0;
    uint32_t tile_in_row = 0;

    uint32_t prow = src0_start_row + fastdiv(2, div_tpr);
    uint32_t pcol = fastmodulo(2, tiles_per_row, div_tpr) * col_tile;
    uint32_t ptile_in_row = fastmodulo(2, tiles_per_row, div_tpr);

    لكل (uint32_t t = 0; t < total_tiles; t++) {
        فراغ * dst_vtcm = (فراغ *) (uintptr_t) dma_queue_دەرهێنان(dma_q).src;
        فراغ * src_vtcm = (فراغ *) (uintptr_t) dma_queue_دەرهێنان(dma_q).dst;

        ثابت uint32_t tw = MIN(col_tile, ne0 - col);

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, t);
        compute(dst_vtcm, src_vtcm, tw, uctx);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, t);

        ثابت size_t doff = (dst_contig ? (row * nb1) :
                             unary_row_offset(row, ne1, ne2, div_ne01, div_ne02, div_ne012, nb1, nb2, nb3)) +
                             (size_t) col * حجم(عائم);
        ثابت size_t tb   = (size_t) tw * حجم(عائم);
        dma_queue_پاڵنان(dma_q, dma_make_data(data_dst + doff, dst_vtcm), tb, tb, tb, 1);

        ثابت uint32_t pt = t + 2;
        إذا (pt < total_tiles) {
            ثابت uint32_t ptw  = MIN(col_tile, ne0 - pcol);
            ثابت size_t   ptb  = (size_t) ptw * حجم(عائم);
            ثابت size_t   psoff = (src0_contig ? (prow * nb01) :
                                    unary_row_offset(prow, ne01, ne02, div_ne01, div_ne02, div_ne012, nb01, nb02,
                                                     nb03)) +
                                   (size_t) pcol * حجم(عائم);
            dma_queue_پاڵنان(dma_q, dma_make_data(src_vtcm, data_src + psoff), ptb, ptb, ptb, 1);
        }

        tile_in_row++;
        col += col_tile;
        إذا (tile_in_row == tiles_per_row) {
            tile_in_row = 0;
            col = 0;
            row++;
        }

        ptile_in_row++;
        pcol += col_tile;
        إذا (ptile_in_row == tiles_per_row) {
            ptile_in_row = 0;
            pcol = 0;
            prow++;
        }
    }

    dma_queue_flush(dma_q);
}

// 5. TRI tiled task.
ساكن فراغ unary_thread_tiled_tri_f32(unsigned صحيح nth, unsigned صحيح ith, فراغ * data) {
    (فراغ) nth;
    ثابت بنية htp_unary_context * uctx = (ثابت بنية htp_unary_context *) data;
    بنية htp_ops_context * octx = uctx->octx;
    ثابت بنية htp_tensor * src = octx->src[0];
    ثابت بنية htp_tensor * dst = octx->dst;
    بنية htp_thread_trace * tr = &octx->ctx->trace[ith];

    htp_unary_preamble;

    ثابت uint32_t src0_nrows_per_thread = uctx->src0_nrows_per_thread;
    ثابت int32_t * op_params            = octx->op_params;
    ثابت uint32_t col_tile              = uctx->col_tile;

    ثابت uint32_t src0_nrows     = uctx->src0_nrows;
    ثابت uint32_t src0_start_row = uctx->row_start + src0_nrows_per_thread * ith;
    ثابت uint32_t src0_end_row   = MIN(src0_start_row + src0_nrows_per_thread, uctx->row_start + src0_nrows);

    إذا (src0_start_row >= src0_end_row) {
        أرجع;
    }

    ثابت dma_addr_t data_src = uctx->data_src0;
    ثابت dma_addr_t data_dst = uctx->data_dst;

    uint8_t * src0_vtcm_data = uctx->vtcm_src0 + (ith * uctx->vtcm_src0_size_per_thread);
    uint8_t * dst_vtcm_data  = uctx->vtcm_dst + (ith * uctx->vtcm_dst_size_per_thread);

    ثابت size_t src0_half = uctx->src0_vtcm_half_size;
    ثابت size_t dst_half  = uctx->dst_vtcm_half_size;

    dma_queue * dma_q = octx->ctx->dma[ith];

    ثابت بنية fastdiv_بەهاکان * div_ne01  = &uctx->kparams->div_ne01;
    ثابت بنية fastdiv_بەهاکان * div_ne02  = &uctx->kparams->div_ne02;
    ثابت بنية fastdiv_بەهاکان * div_ne012 = &uctx->kparams->div_ne012;
    ثابت بنية fastdiv_بەهاکان * div_tpr   = &uctx->kparams->div_tpr;

    ثابت uint32_t tiles_per_row = (ne0 + col_tile - 1) / col_tile;
    ثابت int32_t  tri_ttype     = op_params[0];

    ثابت bool src0_contig = (nb02 == (size_t)ne01 * nb01) &&
                             (nb03 == (size_t)ne02 * nb02);
    ثابت bool dst_contig  = (nb2  == (size_t)ne1  * nb1)  &&
                             (nb3  == (size_t)ne2  * nb2);

    ثابت uint32_t total_tiles = (src0_end_row - src0_start_row) * tiles_per_row;

    لكل (uint32_t t = 0, vtcm_idx = 0; t < total_tiles && vtcm_idx < 2; t++, vtcm_idx++) {
        ثابت uint32_t row  = src0_start_row + t / tiles_per_row;
        ثابت uint32_t col  = (t % tiles_per_row) * col_tile;
        ثابت uint32_t tw   = MIN(col_tile, ne0 - col);
        ثابت size_t   tb   = (size_t) tw * حجم(عائم);
        ثابت size_t   soff = (src0_contig ? (row * nb01) :
                               unary_row_offset(row, ne01, ne02, div_ne01, div_ne02, div_ne012, nb01, nb02, nb03)) +
                               (size_t) col * حجم(عائم);

        dma_queue_پاڵنان(dma_q, dma_make_data(data_dst, dst_vtcm_data + (vtcm_idx * dst_half)), 0, 0, 0, 0);
        dma_queue_پاڵنان(dma_q, dma_make_data(src0_vtcm_data + (vtcm_idx * src0_half), data_src + soff), tb, tb, tb, 1);
    }

    unary_tiled_tri_compute_fn_t compute = (unary_tiled_tri_compute_fn_t) uctx->compute;

    uint32_t row = src0_start_row;
    uint32_t col = 0;
    uint32_t tile_in_row = 0;
    uint32_t i01 = fastmodulo(row, ne01, div_ne01);

    uint32_t prow = src0_start_row + fastdiv(2, div_tpr);
    uint32_t pcol = fastmodulo(2, tiles_per_row, div_tpr) * col_tile;
    uint32_t ptile_in_row = fastmodulo(2, tiles_per_row, div_tpr);

    لكل (uint32_t t = 0; t < total_tiles; t++) {
        فراغ * dst_vtcm = (فراغ *) (uintptr_t) dma_queue_دەرهێنان(dma_q).src;
        فراغ * src_vtcm = (فراغ *) (uintptr_t) dma_queue_دەرهێنان(dma_q).dst;

        ثابت uint32_t tw = MIN(col_tile, ne0 - col);

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, t);
        compute(src_vtcm, dst_vtcm, tw, col, i01, ne0, tri_ttype);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, t);

        ثابت size_t doff = (dst_contig ? (row * nb1) :
                             unary_row_offset(row, ne1, ne2, div_ne01, div_ne02, div_ne012, nb1, nb2, nb3)) +
                             (size_t) col * حجم(عائم);
        ثابت size_t tb   = (size_t) tw * حجم(عائم);
        dma_queue_پاڵنان(dma_q, dma_make_data(data_dst + doff, dst_vtcm), tb, tb, tb, 1);

        ثابت uint32_t pt = t + 2;
        إذا (pt < total_tiles) {
            ثابت uint32_t ptw  = MIN(col_tile, ne0 - pcol);
            ثابت size_t   ptb  = (size_t) ptw * حجم(عائم);
            ثابت size_t   psoff = (src0_contig ? (prow * nb01) :
                                    unary_row_offset(prow, ne01, ne02, div_ne01, div_ne02, div_ne012, nb01, nb02,
                                                     nb03)) +
                                   (size_t) pcol * حجم(عائم);
            dma_queue_پاڵنان(dma_q, dma_make_data(src_vtcm, data_src + psoff), ptb, ptb, ptb, 1);
        }

        tile_in_row++;
        col += col_tile;
        إذا (tile_in_row == tiles_per_row) {
            tile_in_row = 0;
            col = 0;
            row++;
            i01++;
            إذا (i01 == ne01) {
                i01 = 0;
            }
        }

        ptile_in_row++;
        pcol += col_tile;
        إذا (ptile_in_row == tiles_per_row) {
            ptile_in_row = 0;
            pcol = 0;
            prow++;
        }
    }

    dma_queue_flush(dma_q);
}

ساكن صحيح execute_op_unary(بنية htp_ops_context * octx) {
    صحيح err = HTP_STATUS_OK;

    ثابت بنية htp_tensor * src0 = octx->src[0];
    ثابت بنية htp_tensor * dst  = octx->dst;

    ثابت bool is_f16 = (src0->type == HTP_TYPE_F16);

    ثابت حرف * op_type = عدم;

    بدل (octx->op) {
        حالة HTP_OP_NORM:            op_type = is_f16 ? "norm-f16"     : "norm-f32";         اكسر;
        حالة HTP_OP_RMS_NORM:        op_type = is_f16 ? "rmsnorm-f16"  : "rmsnorm-f32";      اكسر;
        حالة HTP_OP_RMS_NORM_MUL:    op_type = "rmsnorm-mul-f32";                            اكسر;
        حالة HTP_OP_SCALE:           op_type = is_f16 ? "scale-f16"    : "scale-f32";        اكسر;
        حالة HTP_OP_CLAMP:           op_type = is_f16 ? "clamp-f16"    : "clamp-f32";        اكسر;
        حالة HTP_OP_LEAKY_RELU:      op_type = "leaky-relu-f32";                             اكسر;
        حالة HTP_OP_SQR:             op_type = is_f16 ? "sqr-f16"      : "sqr-f32";          اكسر;
        حالة HTP_OP_SQRT:            op_type = is_f16 ? "sqrt-f16"     : "sqrt-f32";         اكسر;
        حالة HTP_OP_UNARY_NEG:       op_type = "neg-f32";                                    اكسر;
        حالة HTP_OP_UNARY_EXP:       op_type = "exp-f32";                                    اكسر;
        حالة HTP_OP_UNARY_SIGMOID:   op_type = "sigmoid-f32";                                اكسر;
        حالة HTP_OP_UNARY_SILU:      op_type = "silu-f32";                                   اكسر;
        حالة HTP_OP_UNARY_GELU:      op_type = "gelu-f32";                                   اكسر;
        حالة HTP_OP_UNARY_GELU_QUICK: op_type = "gelu-quick-f32";                             اكسر;
        حالة HTP_OP_UNARY_GELU_ERF:  op_type = "gelu-erf-f32";                               اكسر;
        حالة HTP_OP_UNARY_SOFTPLUS:  op_type = "softplus-f32";                               اكسر;
        حالة HTP_OP_UNARY_TANH:      op_type = "tanh-f32";                                   اكسر;
        حالة HTP_OP_UNARY_ABS:       op_type = is_f16 ? "abs-f16"      : "abs-f32";          اكسر;
        حالة HTP_OP_UNARY_LOG:       op_type = is_f16 ? "log-f16"      : "log-f32";          اكسر;
        حالة HTP_OP_UNARY_RELU:      op_type = "relu-f32";                                   اكسر;
        حالة HTP_OP_UNARY_STEP:      op_type = is_f16 ? "step-f16"     : "step-f32";         اكسر;
        حالة HTP_OP_L2_NORM:         op_type = is_f16 ? "l2norm-f16"   : "l2norm-f32";       اكسر;
        حالة HTP_OP_TRI:             op_type = "tri-f32";                                    اكسر;
        حالة HTP_OP_HADAMARD:        op_type = is_f16 ? "hadamard-f16" : "hadamard-f32";     اكسر;
        افتراضي:
            FARF(ERROR, "Unsupported unary Op %u\n", octx->op);
            أرجع HTP_STATUS_NO_SUPPORT;
    }

    // F16 only has row-block kernels for this subset of ops (see the dispatch switch
    // below) - reject everything else up front, before touching kparams/VTCM.
    إذا (is_f16) {
        بدل (octx->op) {
            حالة HTP_OP_NORM:
            حالة HTP_OP_RMS_NORM:
            حالة HTP_OP_SCALE:
            حالة HTP_OP_CLAMP:
            حالة HTP_OP_SQR:
            حالة HTP_OP_SQRT:
            حالة HTP_OP_L2_NORM:
            حالة HTP_OP_UNARY_ABS:
            حالة HTP_OP_UNARY_LOG:
            حالة HTP_OP_UNARY_STEP:
            حالة HTP_OP_HADAMARD:
                اكسر;
            افتراضي:
                FARF(ERROR, "unary-%s: not supported for F16\n", op_type);
                أرجع HTP_STATUS_NO_SUPPORT;
        }
    }

    ثابت بنية htp_unary_kernel_params * kparams = (ثابت بنية htp_unary_kernel_params *) octx->kernel_params;

    إذا (!htp_ops_context_set_n_threads(octx, kparams->n_threads)) {
        أرجع HTP_STATUS_INVAL_PARAMS;
    }

    ثابت uint32_t src0_nrows = src0->ne[1] * src0->ne[2] * src0->ne[3];
    ثابت size_t src0_elem_size = is_f16 ? حجم(_Float16) : حجم(عائم);
    ثابت size_t dst_elem_size  = (dst->type == HTP_TYPE_F16) ? حجم(_Float16) : حجم(عائم);
    ثابت size_t src0_data_row_size = src0->ne[0] * src0_elem_size;
    ثابت size_t dst_data_row_size  = dst->ne[0]  * dst_elem_size;

    uint32_t row_start = 0;
    uint32_t nrows     = src0_nrows;

    إذا (octx->ctx->mdev.ژماردن > 1) {
        uint32_t rows_per_chunk = 0;
        htp_tensor_mdev_rows_per_chunk(dst, (uint32_t) dst_elem_size, (uint32_t) dst_data_row_size, &rows_per_chunk);
        ثابت بنية htp_tensor_mdev_range range = htp_tensor_mdev_partition(src0_nrows, rows_per_chunk, octx->ctx->mdev.idx, octx->ctx->mdev.ژماردن, &octx->ctx->mdev.ژماردن_div);
        row_start = range.start;
        nrows     = range.ژماردن;
    }

    إذا (nrows == 0) {
        أرجع HTP_STATUS_OK;
    }

    ثابت uint32_t n_threads = octx->n_threads;

    ثابت size_t src0_row_size_aligned = kparams->src0_row_size_aligned;
    ثابت size_t dst_row_size_aligned  = kparams->dst_row_size_aligned;

    // Always 0 for F16 - htp_unary_vtcm_layout_build() keeps F16 on the row-block path,
    // since only F32 has unary_task_f32_tiled_* kernels.
    ثابت uint32_t col_tile = kparams->col_tile;

    size_t src1_data_row_size = 0;
    size_t src1_row_size_aligned = kparams->src1_row_size_aligned;
    bool broadcast_weight = kparams->broadcast_weight;
    ثابت بنية htp_tensor * src1 = عدم;

    // RMS_NORM_MUL fusion is F32-only (its weight tensor is always F32; see
    // try_fuse_node()'s type guard), so this never triggers when is_f16 is true.
    إذا (octx->op == HTP_OP_RMS_NORM_MUL) {
        src1 = octx->src[1];
        src1_data_row_size = src1->ne[0] * حجم(عائم);
    }

    إذا (octx->ctx->vtcm_size < (size_t)kparams->vtcm_size) {
        FARF(ERROR, "unary-%s : current VTCM reservation %zu is too small, needed %zu\n", op_type, octx->ctx->vtcm_size, (size_t)kparams->vtcm_size);
        أرجع HTP_STATUS_VTCM_TOO_SMALL;
    }

    octx->src0_spad.src = عدم;
    octx->src1_spad.src = عدم;
    octx->dst_spad.src  = عدم;

    FARF(بەرز_2, "%s: (%ux%ux%ux%u) -> (%ux%ux%ux%u) : src0-vtcm-size %u src1-vtcm-size %u dst-vtcm-size %u\n", op_type,
         src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3], dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3],
         kparams->vtcm_src0_size, kparams->vtcm_src1_size, kparams->vtcm_dst_size);

    uint8_t * ثابت base = (uint8_t *) octx->ctx->vtcm_base;
    بنية htp_unary_context uctx = {
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

    FARF(بەرز_2, "%s: %s mode (col_tile %u)\n", op_type, col_tile ? "tiled" : "row-block", col_tile);

    worker_callback_t task_func = عدم;
    فراغ * compute_func = عدم;

    إذا (col_tile) {
        task_func = unary_thread_tiled;
        بدل (octx->op) {
            حالة HTP_OP_SCALE:           compute_func = (فراغ *) tile_scale_f32;          اكسر;
            حالة HTP_OP_CLAMP:           compute_func = (فراغ *) tile_clamp_f32;          اكسر;
            حالة HTP_OP_LEAKY_RELU:      compute_func = (فراغ *) tile_leaky_relu_f32;     اكسر;
            حالة HTP_OP_SQR:             compute_func = (فراغ *) tile_sqr_f32;            اكسر;
            حالة HTP_OP_SQRT:            compute_func = (فراغ *) tile_sqrt_f32;           اكسر;
            حالة HTP_OP_UNARY_NEG:       compute_func = (فراغ *) tile_neg_f32;            اكسر;
            حالة HTP_OP_UNARY_EXP:       compute_func = (فراغ *) tile_exp_f32;            اكسر;
            حالة HTP_OP_UNARY_SIGMOID:   compute_func = (فراغ *) tile_sigmoid_f32;        اكسر;
            حالة HTP_OP_UNARY_SILU:      compute_func = (فراغ *) tile_silu_f32;           اكسر;
            حالة HTP_OP_UNARY_GELU:      compute_func = (فراغ *) tile_gelu_f32;           اكسر;
            حالة HTP_OP_UNARY_GELU_QUICK: compute_func = (فراغ *) tile_gelu_quick_f32;     اكسر;
            حالة HTP_OP_UNARY_GELU_ERF:  compute_func = (فراغ *) tile_gelu_erf_f32;       اكسر;
            حالة HTP_OP_UNARY_SOFTPLUS:  compute_func = (فراغ *) tile_softplus_f32;       اكسر;
            حالة HTP_OP_UNARY_TANH:      compute_func = (فراغ *) tile_tanh_f32;           اكسر;
            حالة HTP_OP_UNARY_ABS:       compute_func = (فراغ *) tile_abs_f32;            اكسر;
            حالة HTP_OP_UNARY_LOG:       compute_func = (فراغ *) tile_log_f32;            اكسر;
            حالة HTP_OP_UNARY_RELU:      compute_func = (فراغ *) tile_relu_f32;           اكسر;
            حالة HTP_OP_UNARY_STEP:      compute_func = (فراغ *) tile_step_f32;           اكسر;
            حالة HTP_OP_TRI:
                task_func    = unary_thread_tiled_tri_f32;
                compute_func = (فراغ *) tri_apply_tile_f32;
                اكسر;
            افتراضي:                     اكسر;
        }
    } وإلا إذا (is_f16) {
        task_func = unary_thread_row_block;
        بدل (octx->op) {
            حالة HTP_OP_NORM:            compute_func = (فراغ *) norm_f16;                اكسر;
            حالة HTP_OP_RMS_NORM:        compute_func = (فراغ *) rms_norm_f16;            اكسر;
            حالة HTP_OP_SCALE:           compute_func = (فراغ *) scale_f16;               اكسر;
            حالة HTP_OP_CLAMP:           compute_func = (فراغ *) clamp_f16;               اكسر;
            حالة HTP_OP_SQR:             compute_func = (فراغ *) sqr_f16;                 اكسر;
            حالة HTP_OP_SQRT:            compute_func = (فراغ *) sqrt_f16;                اكسر;
            حالة HTP_OP_L2_NORM:         compute_func = (فراغ *) l2_norm_f16;             اكسر;
            حالة HTP_OP_UNARY_ABS:       compute_func = (فراغ *) abs_f16;                 اكسر;
            حالة HTP_OP_UNARY_LOG:       compute_func = (فراغ *) log_f16;                 اكسر;
            حالة HTP_OP_UNARY_STEP:      compute_func = (فراغ *) step_f16;                اكسر;
            حالة HTP_OP_HADAMARD:        compute_func = (فراغ *) hadamard_f16;            اكسر;
            افتراضي:                     اكسر;
        }
    } وإلا {
        task_func = unary_thread_row_block;
        بدل (octx->op) {
            حالة HTP_OP_NORM:            compute_func = (فراغ *) norm_f32;                اكسر;
            حالة HTP_OP_RMS_NORM:        compute_func = (فراغ *) rms_norm_f32;            اكسر;
            حالة HTP_OP_RMS_NORM_MUL:
                task_func    = unary_thread_rms_norm_mul_f32;
                compute_func = (فراغ *) rms_norm_mul_f32;
                اكسر;
            حالة HTP_OP_SCALE:           compute_func = (فراغ *) scale_f32;               اكسر;
            حالة HTP_OP_CLAMP:           compute_func = (فراغ *) clamp_f32;               اكسر;
            حالة HTP_OP_LEAKY_RELU:      compute_func = (فراغ *) leaky_relu_f32;          اكسر;
            حالة HTP_OP_SQR:             compute_func = (فراغ *) sqr_f32;                 اكسر;
            حالة HTP_OP_SQRT:            compute_func = (فراغ *) sqrt_f32;                اكسر;
            حالة HTP_OP_UNARY_NEG:       compute_func = (فراغ *) neg_f32;                 اكسر;
            حالة HTP_OP_UNARY_EXP:       compute_func = (فراغ *) exp_f32;                 اكسر;
            حالة HTP_OP_UNARY_SIGMOID:   compute_func = (فراغ *) sigmoid_f32;             اكسر;
            حالة HTP_OP_UNARY_SILU:      compute_func = (فراغ *) silu_f32;                اكسر;
            حالة HTP_OP_UNARY_GELU:      compute_func = (فراغ *) gelu_f32;                اكسر;
            حالة HTP_OP_UNARY_GELU_QUICK: compute_func = (فراغ *) gelu_quick_f32;          اكسر;
            حالة HTP_OP_UNARY_GELU_ERF:  compute_func = (فراغ *) gelu_erf_f32;            اكسر;
            حالة HTP_OP_UNARY_SOFTPLUS:  compute_func = (فراغ *) softplus_f32;            اكسر;
            حالة HTP_OP_UNARY_TANH:      compute_func = (فراغ *) tanh_f32;                اكسر;
            حالة HTP_OP_UNARY_ABS:       compute_func = (فراغ *) abs_f32;                 اكسر;
            حالة HTP_OP_UNARY_LOG:       compute_func = (فراغ *) log_f32;                 اكسر;
            حالة HTP_OP_UNARY_RELU:      compute_func = (فراغ *) relu_f32;                اكسر;
            حالة HTP_OP_UNARY_STEP:      compute_func = (فراغ *) step_f32;                اكسر;
            حالة HTP_OP_L2_NORM:         compute_func = (فراغ *) l2_norm_f32;             اكسر;
            حالة HTP_OP_HADAMARD:        compute_func = (فراغ *) hadamard_f32;            اكسر;
            حالة HTP_OP_TRI:
                task_func    = unary_thread_tri_f32;
                compute_func = (فراغ *) tri_f32;
                اكسر;
            افتراضي:                     اكسر;
        }
    }

    إذا (!task_func || !compute_func) {
        FARF(ERROR, "execute_op_unary: task function is NULL for op %d\n", octx->op);
        أرجع HTP_STATUS_NO_SUPPORT;
    }

    uctx.compute = compute_func;
    work_queue_run(octx->ctx->work_queue, task_func, &uctx, n_threads);

    أرجع err;
}

صحيح op_unary(بنية htp_ops_context * octx) {
    بدل (octx->src[0]->type) {
        حالة HTP_TYPE_F32:
        حالة HTP_TYPE_F16:
            أرجع execute_op_unary(octx);

        افتراضي:
            أرجع HTP_STATUS_NO_SUPPORT;
    }
}
