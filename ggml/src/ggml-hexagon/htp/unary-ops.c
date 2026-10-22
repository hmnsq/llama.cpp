#pragma clang diagnostic ignored "-Wunused-variable"
#pragma clang diagnostic ignored "-Wunused-function"
#pragma clang diagnostic ignored "-Wunused-but-set-variable"

#شامل <HAP_farf.h>
#شامل <HAP_perf.h>

#شامل <math.h>
#شامل <string.h>

#شامل "dma-queue.h"
#شامل "hex-fastdiv.h"
#شامل "hvx-exp.h"
#شامل "hvx-sigmoid.h"
#شامل "hvx-utils.h"
#شامل "unary-ops.h"

#define GGML_COMMON_DECL_C
#شامل "ggml-common.h"
#شامل "htp-ctx.h"
#شامل "htp-ops.h"
#شامل "htp-tensor.h"
#شامل "htp-vtcm.h"
#شامل "hex-profile.h"

ساختار htp_unary_context;

تعریفنوع تهی (*unary_compute_fn_t)(ثابت تهی * restrict src,
                                   تهی * restrict dst,
                                   uint32_t رقم_rows,
                                   ثابت ساختار htp_unary_context * uctx);

تعریفنوع تهی (*unary_rms_norm_mul_compute_fn_t)(ثابت تهی * restrict src,
                                                ثابت تهی * restrict weight,
                                                تهی * restrict dst,
                                                uint32_t رقم_rows,
                                                ثابت ساختار htp_unary_context * uctx);

تعریفنوع تهی (*unary_tri_compute_fn_t)(ثابت تهی * restrict src,
                                       تهی * restrict dst,
                                       uint32_t رقم_rows,
                                       uint32_t ir,
                                       ثابت ساختار htp_unary_context * uctx);

تعریفنوع تهی (*unary_tile_compute_fn_t)(تهی * restrict dst,
                                        ثابت تهی * restrict src,
                                        uint32_t tw,
                                        ثابت ساختار htp_unary_context * uctx);

تعریفنوع تهی (*unary_tiled_tri_compute_fn_t)(ثابت تهی * restrict src,
                                             تهی * restrict dst,
                                             uint32_t tile_elems,
                                             uint32_t col_start,
                                             uint32_t i01,
                                             uint32_t ne0,
                                             int32_t ttype);

ساختار htp_unary_context {
    ساختار htp_ops_context * octx;
    ثابت ساختار htp_unary_kernel_params * kparams;

    تهی *                    compute;

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
ایستا درونخطی size_t unary_row_offset(uint32_t ir,
                                      uint32_t ne1, uint32_t ne2,
                                      ثابت ساختار fastdiv_مقادیر * div_ne1,
                                      ثابت ساختار fastdiv_مقادیر * div_ne2,
                                      ثابت ساختار fastdiv_مقادیر * div_ne12,
                                      size_t nb1, size_t nb2, size_t nb3) {
    ثابت uint32_t i1 = fastmodulo(ir, ne1, div_ne1);
    ثابت uint32_t ir_div_ne1 = fastdiv(ir, div_ne1);
    ثابت uint32_t i2 = fastmodulo(ir_div_ne1, ne2, div_ne2);
    ثابت uint32_t i3 = fastdiv(ir, div_ne12);
    بازگشت i1 * nb1 + i2 * nb2 + i3 * nb3;
}

// Safe DMA block size from row `ir`: clamp to the tighter dim-1 slice
// boundary of src and dst so the nb1 stride stays valid for all rows.
ایستا درونخطی uint32_t unary_block_size(uint32_t ir,
                                        uint32_t end_row,
                                        uint32_t block,
                                        bool src_contig,
                                        bool dst_contig,
                                        uint32_t ne1,
                                        ثابت ساختار fastdiv_مقادیر * div_ne1) {
    uint32_t limit = MIN(block, end_row - ir);

    اگر (!src_contig || !dst_contig) {
        ثابت uint32_t slice_end = (fastdiv(ir, div_ne1) + 1) * ne1;
        limit = MIN(limit, slice_end - ir);
    }

    بازگشت limit;
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

ایستا تهی scale_f32(ثابت تهی * restrict src,
                      تهی * restrict dst,
                      ثابت uint32_t رقم_rows,
                      ثابت ساختار htp_unary_context * uctx) {
    htp_unary_op_preamble;
    اعشاری scale = 0.f;
    اعشاری bias  = 0.f;
    memcpy(&scale, &op_params[0], اندازه(اعشاری));
    memcpy(&bias,  &op_params[1], اندازه(اعشاری));

    برای (uint32_t ir = 0; ir < رقم_rows; ir++) {
        ثابت uint8_t * restrict src_local = (ثابت uint8_t *)src + (ir * src0_row_size_aligned);
        uint8_t * restrict dst_local       = (uint8_t *)dst + (ir * dst_row_size_aligned);

        hvx_scale_offset_f32_aa((uint8_t *) dst_local, (ثابت uint8_t *) src_local, ne0, scale, bias);
    }
}

ایستا تهی clamp_f32(ثابت تهی * restrict src,
                      تهی * restrict dst,
                      ثابت uint32_t رقم_rows,
                      ثابت ساختار htp_unary_context * uctx) {
    htp_unary_op_preamble;
    اعشاری min = 0.f;
    اعشاری max = 0.f;
    memcpy(&min, &op_params[0], اندازه(اعشاری));
    memcpy(&max, &op_params[1], اندازه(اعشاری));

    برای (uint32_t ir = 0; ir < رقم_rows; ir++) {
        ثابت uint8_t * restrict src_local = (ثابت uint8_t *)src + (ir * src0_row_size_aligned);
        uint8_t * restrict dst_local       = (uint8_t *)dst + (ir * dst_row_size_aligned);

        hvx_clamp_scalar_f32(dst_local, src_local, min, max, ne0);
    }
}

ایستا تهی leaky_relu_f32(ثابت تهی * restrict src,
                           تهی * restrict dst,
                           ثابت uint32_t رقم_rows,
                           ثابت ساختار htp_unary_context * uctx) {
    htp_unary_op_preamble;
    اعشاری negative_slope = 0.f;
    memcpy(&negative_slope, &op_params[0], اندازه(اعشاری));

    برای (uint32_t ir = 0; ir < رقم_rows; ir++) {
        ثابت uint8_t * restrict src_local = (ثابت uint8_t *)src + (ir * src0_row_size_aligned);
        uint8_t * restrict dst_local       = (uint8_t *)dst + (ir * dst_row_size_aligned);

        hvx_leaky_relu_scalar_f32(dst_local, src_local, negative_slope, ne0);
    }
}

ایستا تهی rms_norm_f32(ثابت تهی * restrict src,
                         تهی * restrict dst,
                         ثابت uint32_t رقم_rows,
                         ثابت ساختار htp_unary_context * uctx) {
    htp_unary_op_preamble;
    اعشاری epsilon = 0.f;
    memcpy(&epsilon, op_params, اندازه(اعشاری));

    برای (uint32_t ir = 0; ir < رقم_rows; ir++) {
        ثابت uint8_t * restrict src_local = (ثابت uint8_t *)src + (ir * src0_row_size_aligned);
        uint8_t * restrict dst_local       = (uint8_t *)dst + (ir * dst_row_size_aligned);

        hvx_fast_rms_norm_f32((ثابت uint8_t *) src_local, (uint8_t *) dst_local, ne0, epsilon);
    }
}

ایستا تهی rms_norm_mul_f32(ثابت تهی * restrict src,
                             ثابت تهی * restrict weight,
                             تهی * restrict dst,
                             ثابت uint32_t رقم_rows,
                             ثابت ساختار htp_unary_context * uctx) {
    htp_unary_op_preamble;
    اعشاری epsilon = 0.f;
    memcpy(&epsilon, op_params, اندازه(اعشاری));

    برای (uint32_t ir = 0; ir < رقم_rows; ir++) {
        ثابت uint8_t * restrict src_local = (ثابت uint8_t *)src + (ir * src0_row_size_aligned);
        ثابت uint8_t * restrict w_local   = (ثابت uint8_t *)weight + (uctx->broadcast_weight ? 0 : ir * uctx->src1_row_size_aligned);
        uint8_t * restrict dst_local       = (uint8_t *)dst + (ir * dst_row_size_aligned);

        hvx_fast_rms_norm_mul_f32(src_local, w_local, dst_local, ne0, epsilon);
    }
}

ایستا تهی norm_f32(ثابت تهی * restrict src,
                     تهی * restrict dst,
                     ثابت uint32_t رقم_rows,
                     ثابت ساختار htp_unary_context * uctx) {
    htp_unary_op_preamble;
    اعشاری epsilon = 0.f;
    memcpy(&epsilon, op_params, اندازه(اعشاری));

    برای (uint32_t ir = 0; ir < رقم_rows; ir++) {
        ثابت uint8_t * restrict src_local = (ثابت uint8_t *)src + (ir * src0_row_size_aligned);
        uint8_t * restrict dst_local       = (uint8_t *)dst + (ir * dst_row_size_aligned);

        hvx_fast_norm_f32((ثابت uint8_t *) src_local, (uint8_t *) dst_local, ne0, epsilon);
    }
}

ایستا تهی sqr_f32(ثابت تهی * restrict src,
                    تهی * restrict dst,
                    ثابت uint32_t رقم_rows,
                    ثابت ساختار htp_unary_context * uctx) {
    htp_unary_op_preamble;

    برای (uint32_t ir = 0; ir < رقم_rows; ir++) {
        ثابت uint8_t * restrict src_local = (ثابت uint8_t *)src + (ir * src0_row_size_aligned);
        uint8_t * restrict dst_local       = (uint8_t *)dst + (ir * dst_row_size_aligned);

        hvx_sqr_f32_aa((uint8_t *) dst_local, (ثابت uint8_t *) src_local, ne0);
    }
}

ایستا تهی sqrt_f32(ثابت تهی * restrict src,
                     تهی * restrict dst,
                     ثابت uint32_t رقم_rows,
                     ثابت ساختار htp_unary_context * uctx) {
    htp_unary_op_preamble;

    برای (uint32_t ir = 0; ir < رقم_rows; ir++) {
        ثابت uint8_t * restrict src_local = (ثابت uint8_t *)src + (ir * src0_row_size_aligned);
        uint8_t * restrict dst_local       = (uint8_t *)dst + (ir * dst_row_size_aligned);

        hvx_sqrt_f32_aa((uint8_t *) dst_local, (ثابت uint8_t *) src_local, ne0);
    }
}

ایستا تهی scale_f16(ثابت تهی * restrict src,
                      تهی * restrict dst,
                      ثابت uint32_t رقم_rows,
                      ثابت ساختار htp_unary_context * uctx) {
    htp_unary_op_preamble;
    اعشاری scale = 0.f;
    اعشاری bias  = 0.f;
    memcpy(&scale, &op_params[0], اندازه(اعشاری));
    memcpy(&bias,  &op_params[1], اندازه(اعشاری));

    برای (uint32_t ir = 0; ir < رقم_rows; ir++) {
        ثابت uint8_t * restrict src_local = (ثابت uint8_t *)src + (ir * src0_row_size_aligned);
        uint8_t * restrict dst_local       = (uint8_t *)dst + (ir * dst_row_size_aligned);

        hvx_scale_offset_f16_aa((uint8_t *) dst_local, (ثابت uint8_t *) src_local, ne0, scale, bias);
    }
}

ایستا تهی clamp_f16(ثابت تهی * restrict src,
                      تهی * restrict dst,
                      ثابت uint32_t رقم_rows,
                      ثابت ساختار htp_unary_context * uctx) {
    htp_unary_op_preamble;
    اعشاری min = 0.f;
    اعشاری max = 0.f;
    memcpy(&min, &op_params[0], اندازه(اعشاری));
    memcpy(&max, &op_params[1], اندازه(اعشاری));

    برای (uint32_t ir = 0; ir < رقم_rows; ir++) {
        ثابت uint8_t * restrict src_local = (ثابت uint8_t *)src + (ir * src0_row_size_aligned);
        uint8_t * restrict dst_local       = (uint8_t *)dst + (ir * dst_row_size_aligned);

        hvx_clamp_scalar_f16(dst_local, src_local, (_Float16) min, (_Float16) max, ne0);
    }
}

ایستا تهی rms_norm_f16(ثابت تهی * restrict src,
                         تهی * restrict dst,
                         ثابت uint32_t رقم_rows,
                         ثابت ساختار htp_unary_context * uctx) {
    htp_unary_op_preamble;
    اعشاری epsilon = 0.f;
    memcpy(&epsilon, op_params, اندازه(اعشاری));

    برای (uint32_t ir = 0; ir < رقم_rows; ir++) {
        ثابت uint8_t * restrict src_local = (ثابت uint8_t *)src + (ir * src0_row_size_aligned);
        uint8_t * restrict dst_local       = (uint8_t *)dst + (ir * dst_row_size_aligned);

        hvx_fast_rms_norm_f16((ثابت uint8_t *) src_local, (uint8_t *) dst_local, ne0, epsilon);
    }
}

ایستا تهی norm_f16(ثابت تهی * restrict src,
                     تهی * restrict dst,
                     ثابت uint32_t رقم_rows,
                     ثابت ساختار htp_unary_context * uctx) {
    htp_unary_op_preamble;
    اعشاری epsilon = 0.f;
    memcpy(&epsilon, op_params, اندازه(اعشاری));

    برای (uint32_t ir = 0; ir < رقم_rows; ir++) {
        ثابت uint8_t * restrict src_local = (ثابت uint8_t *)src + (ir * src0_row_size_aligned);
        uint8_t * restrict dst_local       = (uint8_t *)dst + (ir * dst_row_size_aligned);

        hvx_fast_norm_f16((ثابت uint8_t *) src_local, (uint8_t *) dst_local, ne0, epsilon);
    }
}

ایستا تهی sqr_f16(ثابت تهی * restrict src,
                    تهی * restrict dst,
                    ثابت uint32_t رقم_rows,
                    ثابت ساختار htp_unary_context * uctx) {
    htp_unary_op_preamble;

    برای (uint32_t ir = 0; ir < رقم_rows; ir++) {
        ثابت uint8_t * restrict src_local = (ثابت uint8_t *)src + (ir * src0_row_size_aligned);
        uint8_t * restrict dst_local       = (uint8_t *)dst + (ir * dst_row_size_aligned);

        hvx_sqr_f16_aa((uint8_t *) dst_local, (ثابت uint8_t *) src_local, ne0);
    }
}

ایستا تهی sqrt_f16(ثابت تهی * restrict src,
                     تهی * restrict dst,
                     ثابت uint32_t رقم_rows,
                     ثابت ساختار htp_unary_context * uctx) {
    htp_unary_op_preamble;

    برای (uint32_t ir = 0; ir < رقم_rows; ir++) {
        ثابت uint8_t * restrict src_local = (ثابت uint8_t *)src + (ir * src0_row_size_aligned);
        uint8_t * restrict dst_local       = (uint8_t *)dst + (ir * dst_row_size_aligned);

        hvx_sqrt_f16_aa((uint8_t *) dst_local, (ثابت uint8_t *) src_local, ne0);
    }
}

ایستا تهی abs_f16(ثابت تهی * restrict src,
                    تهی * restrict dst,
                    ثابت uint32_t رقم_rows,
                    ثابت ساختار htp_unary_context * uctx) {
    htp_unary_op_preamble;

    برای (uint32_t ir = 0; ir < رقم_rows; ir++) {
        ثابت uint8_t * restrict src_local = (ثابت uint8_t *)src + (ir * src0_row_size_aligned);
        uint8_t * restrict dst_local       = (uint8_t *)dst + (ir * dst_row_size_aligned);

        hvx_abs_f16_aa((uint8_t *) dst_local, (ثابت uint8_t *) src_local, ne0);
    }
}

ایستا تهی log_f16(ثابت تهی * restrict src,
                    تهی * restrict dst,
                    ثابت uint32_t رقم_rows,
                    ثابت ساختار htp_unary_context * uctx) {
    htp_unary_op_preamble;

    برای (uint32_t ir = 0; ir < رقم_rows; ir++) {
        ثابت uint8_t * restrict src_local = (ثابت uint8_t *)src + (ir * src0_row_size_aligned);
        uint8_t * restrict dst_local       = (uint8_t *)dst + (ir * dst_row_size_aligned);

        hvx_log_f16_aa((uint8_t *) dst_local, (ثابت uint8_t *) src_local, ne0);
    }
}

ایستا تهی step_f16(ثابت تهی * restrict src,
                     تهی * restrict dst,
                     ثابت uint32_t رقم_rows,
                     ثابت ساختار htp_unary_context * uctx) {
    htp_unary_op_preamble;

    برای (uint32_t ir = 0; ir < رقم_rows; ir++) {
        ثابت uint8_t * restrict src_local = (ثابت uint8_t *)src + (ir * src0_row_size_aligned);
        uint8_t * restrict dst_local       = (uint8_t *)dst + (ir * dst_row_size_aligned);

        hvx_step_f16_aa((uint8_t *) dst_local, (ثابت uint8_t *) src_local, ne0);
    }
}

ایستا تهی l2_norm_f16(ثابت تهی * restrict src,
                        تهی * restrict dst,
                        ثابت uint32_t رقم_rows,
                        ثابت ساختار htp_unary_context * uctx) {
    htp_unary_op_preamble;
    اعشاری epsilon = 0.f;
    memcpy(&epsilon, op_params, اندازه(اعشاری));

    برای (uint32_t ir = 0; ir < رقم_rows; ir++) {
        ثابت uint8_t * restrict src_f = (ثابت uint8_t *)src + (ir * src0_row_size_aligned);
        uint8_t * restrict dst_f       = (uint8_t *)dst + (ir * dst_row_size_aligned);

        hvx_fast_l2_norm_f16((ثابت uint8_t *)src_f, (uint8_t *)dst_f, ne0, epsilon);
    }
}

ایستا تهی neg_f32(ثابت تهی * restrict src,
                    تهی * restrict dst,
                    ثابت uint32_t رقم_rows,
                    ثابت ساختار htp_unary_context * uctx) {
    htp_unary_op_preamble;

    برای (uint32_t ir = 0; ir < رقم_rows; ir++) {
        ثابت uint8_t * restrict src_local = (ثابت uint8_t *)src + (ir * src0_row_size_aligned);
        uint8_t * restrict dst_local       = (uint8_t *)dst + (ir * dst_row_size_aligned);

        hvx_scale_f32_aa(dst_local, src_local, ne0, -1.0f);
    }
}

ایستا تهی exp_f32(ثابت تهی * restrict src,
                    تهی * restrict dst,
                    ثابت uint32_t رقم_rows,
                    ثابت ساختار htp_unary_context * uctx) {
    htp_unary_op_preamble;

    برای (uint32_t ir = 0; ir < رقم_rows; ir++) {
        ثابت uint8_t * restrict src_local = (ثابت uint8_t *)src + (ir * src0_row_size_aligned);
        uint8_t * restrict dst_local       = (uint8_t *)dst + (ir * dst_row_size_aligned);

        hvx_exp_f32(dst_local, src_local, ne0, false);
    }
}

ایستا تهی sigmoid_f32(ثابت تهی * restrict src,
                        تهی * restrict dst,
                        ثابت uint32_t رقم_rows,
                        ثابت ساختار htp_unary_context * uctx) {
    htp_unary_op_preamble;

    برای (uint32_t ir = 0; ir < رقم_rows; ir++) {
        ثابت uint8_t * restrict src_local = (ثابت uint8_t *)src + (ir * src0_row_size_aligned);
        uint8_t * restrict dst_local       = (uint8_t *)dst + (ir * dst_row_size_aligned);

        hvx_sigmoid_f32_aa(dst_local, src_local, ne0);
    }
}

// silu(x) = x * sigmoid(x)
ایستا تهی silu_f32(ثابت تهی * restrict src,
                     تهی * restrict dst,
                     ثابت uint32_t رقم_rows,
                     ثابت ساختار htp_unary_context * uctx) {
    htp_unary_op_preamble;

    برای (uint32_t ir = 0; ir < رقم_rows; ir++) {
        ثابت uint8_t * restrict src_local = (ثابت uint8_t *)src + (ir * src0_row_size_aligned);
        uint8_t * restrict dst_local       = (uint8_t *)dst + (ir * dst_row_size_aligned);

        hvx_sigmoid_f32_aa(dst_local, src_local, ne0);
        hvx_mul_f32_aaa(dst_local, src_local, dst_local, ne0);
    }
}

// GELU uses the tanh approximation.
ایستا __attribute__((noinline)) HVX_Vector hvx_vec_gelu_f32(HVX_Vector x) {
    ثابت HVX_Vector half = hvx_vec_splat_f32(0.5f);
    ثابت HVX_Vector one  = hvx_vec_splat_f32(1.0f);

    HVX_Vector inner = hvx_vec_mul_f32_f32(x, x);
    inner = hvx_vec_mul_f32_f32(inner, hvx_vec_splat_f32(0.044715f));
    inner = hvx_vec_add_f32_f32(inner, one);
    inner = hvx_vec_mul_f32_f32(inner, x);
    inner = hvx_vec_mul_f32_f32(inner, hvx_vec_splat_f32(0.7978845608028654f));

    بازگشت hvx_vec_mul_f32_f32(hvx_vec_mul_f32_f32(half, x),
                               hvx_vec_add_f32_f32(one, hvx_vec_tanh_f32(inner)));
}

ایستا درونخطی تهی hvx_gelu_f32_aa(uint8_t * restrict dst, ثابت uint8_t * restrict src, uint32_t n) {
    أكد((unsigned long) dst % 128 == 0);
    أكد((unsigned long) src % 128 == 0);

    HVX_Vector * restrict vdst = (HVX_Vector *) dst;
    HVX_Vector * restrict vsrc = (HVX_Vector *) src;

    ثابت uint32_t nvec = n / VLEN_FP32;
    ثابت uint32_t nloe = n % VLEN_FP32;

    uint32_t i = 0;
    _Pragma("unroll(4)")
    برای (; i < nvec; i++) {
        vdst[i] = hvx_vec_gelu_f32(vsrc[i]);
    }
    اگر (nloe) {
        hvx_vec_store_a(&vdst[i], nloe * اندازه(اعشاری), hvx_vec_gelu_f32(vsrc[i]));
    }
}

// GELU_QUICK uses x * sigmoid(1.702 * x).
ایستا __attribute__((noinline)) HVX_Vector hvx_vec_gelu_quick_f32(HVX_Vector x) {
    ثابت HVX_Vector one     = hvx_vec_splat_f32(1.0f);
    ثابت HVX_Vector max_exp = hvx_vec_splat_f32(87.0f);
    ثابت HVX_Vector min_exp = hvx_vec_splat_f32(-87.0f);
    ثابت HVX_Vector scaled  = hvx_vec_mul_f32_f32(x, hvx_vec_splat_f32(1.702f));
    ثابت HVX_Vector sigmoid = hvx_vec_fast_sigmoid_f32_guard(scaled, one, max_exp, min_exp);

    بازگشت hvx_vec_mul_f32_f32(x, sigmoid);
}

ایستا درونخطی تهی hvx_gelu_quick_f32_aa(uint8_t * restrict dst, ثابت uint8_t * restrict src, uint32_t n) {
    أكد((unsigned long) dst % 128 == 0);
    أكد((unsigned long) src % 128 == 0);

    HVX_Vector * restrict vdst = (HVX_Vector *) dst;
    HVX_Vector * restrict vsrc = (HVX_Vector *) src;

    ثابت uint32_t nvec = n / VLEN_FP32;
    ثابت uint32_t nloe = n % VLEN_FP32;

    uint32_t i = 0;
    _Pragma("unroll(4)")
    برای (; i < nvec; i++) {
        vdst[i] = hvx_vec_gelu_quick_f32(vsrc[i]);
    }
    اگر (nloe) {
        hvx_vec_store_a(&vdst[i], nloe * اندازه(اعشاری), hvx_vec_gelu_quick_f32(vsrc[i]));
    }
}

ایستا تهی gelu_f32(ثابت تهی * restrict src,
                     تهی * restrict dst,
                     ثابت uint32_t رقم_rows,
                     ثابت ساختار htp_unary_context * uctx) {
    htp_unary_op_preamble;

    برای (uint32_t ir = 0; ir < رقم_rows; ir++) {
        ثابت uint8_t * restrict src_local = (ثابت uint8_t *)src + (ir * src0_row_size_aligned);
        uint8_t * restrict dst_local       = (uint8_t *)dst + (ir * dst_row_size_aligned);

        hvx_gelu_f32_aa(dst_local, src_local, ne0);
    }
}

ایستا تهی gelu_quick_f32(ثابت تهی * restrict src,
                           تهی * restrict dst,
                           ثابت uint32_t رقم_rows,
                           ثابت ساختار htp_unary_context * uctx) {
    htp_unary_op_preamble;

    برای (uint32_t ir = 0; ir < رقم_rows; ir++) {
        ثابت uint8_t * restrict src_local = (ثابت uint8_t *)src + (ir * src0_row_size_aligned);
        uint8_t * restrict dst_local       = (uint8_t *)dst + (ir * dst_row_size_aligned);

        hvx_gelu_quick_f32_aa(dst_local, src_local, ne0);
    }
}

ایستا تهی gelu_erf_f32(ثابت تهی * restrict src,
                         تهی * restrict dst,
                         ثابت uint32_t رقم_rows,
                         ثابت ساختار htp_unary_context * uctx) {
    htp_unary_op_preamble;

    برای (uint32_t ir = 0; ir < رقم_rows; ir++) {
        ثابت uint8_t * restrict src_local = (ثابت uint8_t *) src + (ir * src0_row_size_aligned);
        uint8_t * restrict dst_local       = (uint8_t *) dst + (ir * dst_row_size_aligned);

        hvx_gelu_erf_f32_aa(dst_local, src_local, ne0);
    }
}

ایستا تهی tri_f32(ثابت تهی * restrict src,
                    تهی * restrict dst,
                    ثابت uint32_t رقم_rows,
                    ثابت uint32_t ir,
                    ثابت ساختار htp_unary_context * uctx) {
    htp_unary_op_preamble;
    ثابت int32_t ttype = op_params[0];
    ثابت HVX_Vector zero = hvx_vec_splat_f32(0.0f);
    ثابت uint32_t nvec  = ne0 / VLEN_FP32;
    ثابت uint32_t nloe  = ne0 % VLEN_FP32;

    ثابت uint32_t ne01 = uctx->octx->src[0]->ne[1];

    برای (uint32_t b = 0; b < رقم_rows; b++) {
        ثابت uint32_t abs_row = ir + b;
        ثابت uint32_t i01     = abs_row % ne01;

        ثابت HVX_Vector * restrict v_src = (ثابت HVX_Vector *) ((ثابت uint8_t *) src + b * src0_row_size_aligned);
        HVX_Vector * restrict v_dst       = (HVX_Vector *) ((uint8_t *) dst + b * dst_row_size_aligned);

        uint32_t boundary;
        صحیح      keep_چپ;
        انتخاب (ttype) {
            حالت 0: boundary = i01;     keep_چپ = 0; بشکن;  // keep col >= row
            حالت 1: boundary = i01 + 1; keep_چپ = 0; بشکن;  // keep col > row
            حالت 2: boundary = i01 + 1; keep_چپ = 1; بشکن;  // keep col <= row
            حالت 3: boundary = i01;     keep_چپ = 1; بشکن;  // keep col < row
            پیشفرض: boundary = 0; keep_چپ = 0; بشکن;
        }
        اگر (boundary > ne0) boundary = ne0;

        // Full HVX vectors - each starts at a 128-byte aligned offset
        برای (uint32_t i = 0; i < nvec; i++) {
            ثابت uint32_t vec_start = i * VLEN_FP32;
            ثابت uint32_t vec_end   = vec_start + VLEN_FP32;
            اگر (keep_چپ) {
                اگر (vec_end <= boundary) {
                    v_dst[i] = v_src[i];
                } وگرنه اگر (vec_start >= boundary) {
                    v_dst[i] = zero;
                } وگرنه {
                    HVX_VectorPred mask = Q6_Q_vsetq_R((boundary - vec_start) * اندازه(اعشاری));
                    v_dst[i]            = Q6_V_vmux_QVV(mask, v_src[i], zero);
                }
            } وگرنه {
                اگر (vec_end <= boundary) {
                    v_dst[i] = zero;
                } وگرنه اگر (vec_start >= boundary) {
                    v_dst[i] = v_src[i];
                } وگرنه {
                    HVX_VectorPred mask = Q6_Q_vsetq_R((boundary - vec_start) * اندازه(اعشاری));
                    v_dst[i]            = Q6_V_vmux_QVV(mask, zero, v_src[i]);
                }
            }
        }

        // Tail elements (row_elems not a multiple of VLEN_FP32)
        اگر (nloe > 0) {
            ثابت uint32_t abs_start = nvec * VLEN_FP32;
            ثابت uint32_t abs_end   = abs_start + nloe;
            HVX_Vector     tail_val;
            اگر (keep_چپ) {
                اگر (abs_end <= boundary) {
                    tail_val = v_src[nvec];
                } وگرنه اگر (abs_start >= boundary) {
                    tail_val = zero;
                } وگرنه {
                    HVX_VectorPred mask = Q6_Q_vsetq_R((boundary - abs_start) * اندازه(اعشاری));
                    tail_val            = Q6_V_vmux_QVV(mask, v_src[nvec], zero);
                }
            } وگرنه {
                اگر (abs_end <= boundary) {
                    tail_val = zero;
                } وگرنه اگر (abs_start >= boundary) {
                    tail_val = v_src[nvec];
                } وگرنه {
                    HVX_VectorPred mask = Q6_Q_vsetq_R((boundary - abs_start) * اندازه(اعشاری));
                    tail_val            = Q6_V_vmux_QVV(mask, zero, v_src[nvec]);
                }
            }
            hvx_vec_store_a(&v_dst[nvec], nloe * اندازه(اعشاری), tail_val);
        }
    }
}

// softplus(x) = log(1 + exp(x))
// Match CPU reference: return x for x > 20 to avoid overflow.
ایستا درونخطی تهی hvx_softplus_f32_aa(uint8_t * restrict dst, ثابت uint8_t * restrict src, uint32_t n) {
    ثابت HVX_Vector * restrict v_src = (ثابت HVX_Vector *) src;
    HVX_Vector * restrict v_dst       = (HVX_Vector *) dst;

    ثابت HVX_Vector twenty = hvx_vec_splat_f32(20.0f);
    ثابت HVX_Vector one    = hvx_vec_splat_f32(1.0f);

    ثابت uint32_t nvec = n / VLEN_FP32;
    ثابت uint32_t nloe = n % VLEN_FP32;

    #pragma unroll(4)
    برای (uint32_t i = 0; i < nvec; i++) {
        HVX_Vector x              = v_src[i];
        HVX_VectorPred pred_gt20  = Q6_Q_vcmp_gt_VsfVsf(x, twenty);
        HVX_Vector expx           = hvx_vec_exp_f32(x);
        HVX_Vector logv           = hvx_vec_log_f32(hvx_vec_add_f32_f32(one, expx));
        v_dst[i]                  = Q6_V_vmux_QVV(pred_gt20, x, logv);
    }

    اگر (nloe > 0) {
        HVX_Vector x             = v_src[nvec];
        HVX_VectorPred pred_gt20 = Q6_Q_vcmp_gt_VsfVsf(x, twenty);
        HVX_Vector expx          = hvx_vec_exp_f32(x);
        HVX_Vector logv          = hvx_vec_log_f32(hvx_vec_add_f32_f32(one, expx));
        HVX_Vector sp            = Q6_V_vmux_QVV(pred_gt20, x, logv);
        hvx_vec_store_a((تهی *) &v_dst[nvec], nloe * اندازه(اعشاری), sp);
    }
}

ایستا تهی softplus_f32(ثابت تهی * restrict src,
                         تهی * restrict dst,
                         ثابت uint32_t رقم_rows,
                         ثابت ساختار htp_unary_context * uctx) {
    htp_unary_op_preamble;

    برای (uint32_t ir = 0; ir < رقم_rows; ir++) {
        ثابت uint8_t * restrict src_local = (ثابت uint8_t *)src + (ir * src0_row_size_aligned);
        uint8_t * restrict dst_local       = (uint8_t *)dst + (ir * dst_row_size_aligned);

        hvx_softplus_f32_aa(dst_local, src_local, ne0);
    }
}

ایستا تهی l2_norm_f32(ثابت تهی * restrict src,
                        تهی * restrict dst,
                        ثابت uint32_t رقم_rows,
                        ثابت ساختار htp_unary_context * uctx) {
    htp_unary_op_preamble;
    اعشاری epsilon = 0.f;
    memcpy(&epsilon, op_params, اندازه(اعشاری));

    برای (uint32_t ir = 0; ir < رقم_rows; ir++) {
        ثابت uint8_t * restrict src_f = (ثابت uint8_t *)src + (ir * src0_row_size_aligned);
        uint8_t * restrict dst_f       = (uint8_t *)dst + (ir * dst_row_size_aligned);

        hvx_fast_l2_norm_f32((ثابت uint8_t *)src_f, (uint8_t *)dst_f, ne0, epsilon);
    }
}

ایستا درونخطی HVX_VectorPair hvx_fwht_stage_pair(HVX_Vector v0, HVX_Vector v1, int32_t b) {
    HVX_VectorPair p = Q6_W_vshuff_VVR(v1, v0, b);
    HVX_Vector s = Q6_Vqf32_vadd_Vqf32Vqf32(Q6_V_lo_W(p), Q6_V_hi_W(p));
    HVX_Vector d = Q6_Vqf32_vsub_Vqf32Vqf32(Q6_V_lo_W(p), Q6_V_hi_W(p));
    بازگشت Q6_W_vshuff_VVR(d, s, b);
}

ایستا درونخطی HVX_VectorPair hvx_fwht32_pair(HVX_Vector v0_sf, HVX_Vector v1_sf, HVX_Vector v_scale) {
    HVX_Vector v0 = Q6_Vqf32_vmpy_VsfVsf(v0_sf, v_scale);
    HVX_Vector v1 = Q6_Vqf32_vmpy_VsfVsf(v1_sf, v_scale);
    HVX_VectorPair p;
    p = hvx_fwht_stage_pair(v0, v1, 4);
    p = hvx_fwht_stage_pair(Q6_V_lo_W(p), Q6_V_hi_W(p), 8);
    p = hvx_fwht_stage_pair(Q6_V_lo_W(p), Q6_V_hi_W(p), 16);
    p = hvx_fwht_stage_pair(Q6_V_lo_W(p), Q6_V_hi_W(p), 32);
    p = hvx_fwht_stage_pair(Q6_V_lo_W(p), Q6_V_hi_W(p), 64);
    بازگشت p;
}

ایستا درونخطی تهی hvx_fwht_store32(HVX_Vector * restrict v_dst, HVX_VectorPair p) {
    v_dst[0] = Q6_Vsf_equals_Vqf32(Q6_V_lo_W(p));
}

ایستا درونخطی تهی hvx_fwht_store64(HVX_Vector * restrict v_dst, HVX_VectorPair p) {
    HVX_Vector u = Q6_V_lo_W(p);
    HVX_Vector v = Q6_V_hi_W(p);
    v_dst[0] = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vqf32(u, v));
    v_dst[1] = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vsub_Vqf32Vqf32(u, v));
}

ایستا درونخطی تهی hvx_fwht_tail_f32(HVX_Vector * restrict v_dst, uint32_t nvec) {
    uint32_t len_vec = 2;
    برای (; len_vec * 2 < nvec; len_vec <<= 2) {
        ثابت uint32_t step = len_vec * 4;
        برای (uint32_t i = 0; i < nvec; i += step) {
            برای (uint32_t j = 0; j < len_vec; j++) {
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

    اگر (len_vec < nvec) {
        برای (uint32_t i = 0; i < nvec; i += 2 * len_vec) {
            برای (uint32_t j = 0; j < len_vec; j++) {
                HVX_Vector u = v_dst[i + j];
                HVX_Vector v = v_dst[i + len_vec + j];
                v_dst[i + j]           = hvx_vec_add_f32_f32(u, v);
                v_dst[i + len_vec + j] = hvx_vec_sub_f32_f32(u, v);
            }
        }
    }
}

ایستا درونخطی تهی hvx_fwht_row_f32(ثابت uint8_t * restrict src, uint8_t * restrict dst, uint32_t n, HVX_Vector v_scale) {
    ثابت HVX_Vector * restrict v_src = (ثابت HVX_Vector *) src;
    HVX_Vector * restrict v_dst       = (HVX_Vector *) dst;
    ثابت uint32_t nvec               = n / VLEN_FP32;

    اگر (nvec == 1) {
        HVX_VectorPair p = hvx_fwht32_pair(v_src[0], v_src[0], v_scale);
        hvx_fwht_store32(v_dst, p);
        بازگشت;
    }

    برای (uint32_t i = 0; i < nvec; i += 2) {
        HVX_VectorPair p = hvx_fwht32_pair(v_src[i], v_src[i + 1], v_scale);
        hvx_fwht_store64(&v_dst[i], p);
    }

    hvx_fwht_tail_f32(v_dst, nvec);
}

ایستا درونخطی تهی hvx_fwht_row_f16(ثابت uint8_t * restrict src, uint8_t * restrict dst, uint32_t n, HVX_Vector v_scale) {
    ثابت HVX_Vector * restrict v_src = (ثابت HVX_Vector *) src;
    HVX_Vector * restrict v_dst       = (HVX_Vector *) dst;
    ثابت uint32_t nvec               = n / VLEN_FP32;

    اگر (nvec == 1) {
        HVX_Vector v0 = Q6_V_lo_W(hvx_vec_f16_to_f32(v_src[0]));
        HVX_VectorPair p = hvx_fwht32_pair(v0, v0, v_scale);
        hvx_fwht_store32(v_dst, p);
        بازگشت;
    }

    ثابت uint32_t nvec_f16 = n / VLEN_FP16;
    برای (uint32_t i = 0; i < nvec_f16; i++) {
        HVX_VectorPair f32p = hvx_vec_f16_to_f32(v_src[i]);
        HVX_VectorPair p    = hvx_fwht32_pair(Q6_V_lo_W(f32p), Q6_V_hi_W(f32p), v_scale);
        hvx_fwht_store64(&v_dst[2 * i], p);
    }

    hvx_fwht_tail_f32(v_dst, nvec);
}

ایستا تهی hadamard_f32(ثابت تهی * restrict src,
                         تهی * restrict dst,
                         ثابت uint32_t رقم_rows,
                         ثابت ساختار htp_unary_context * uctx) {
    htp_unary_op_preamble;
    ثابت HVX_Vector v_scale = hvx_vec_splat_f32(1.0f / sqrtf((اعشاری) ne0));

    برای (uint32_t ir = 0; ir < رقم_rows; ir++) {
        ثابت uint8_t * restrict src_row = (ثابت uint8_t *) src + (ir * src0_row_size_aligned);
        uint8_t * restrict dst_row       = (uint8_t *) dst + (ir * dst_row_size_aligned);

        hvx_fwht_row_f32(src_row, dst_row, ne0, v_scale);
    }
}

ایستا تهی hadamard_f16(ثابت تهی * restrict src,
                         تهی * restrict dst,
                         ثابت uint32_t رقم_rows,
                         ثابت ساختار htp_unary_context * uctx) {
    htp_unary_op_preamble;
    ثابت HVX_Vector v_scale = hvx_vec_splat_f32(1.0f / sqrtf((اعشاری) ne0));

    برای (uint32_t ir = 0; ir < رقم_rows; ir++) {
        ثابت uint8_t * restrict src_row = (ثابت uint8_t *) src + (ir * src0_row_size_aligned);
        uint8_t * restrict dst_row       = (uint8_t *) dst + (ir * dst_row_size_aligned);

        hvx_fwht_row_f16(src_row, dst_row, ne0, v_scale);
    }
}

ایستا تهی tanh_f32(ثابت تهی * restrict src,
                     تهی * restrict dst,
                     ثابت uint32_t رقم_rows,
                     ثابت ساختار htp_unary_context * uctx) {
    htp_unary_op_preamble;

    برای (uint32_t ir = 0; ir < رقم_rows; ir++) {
        ثابت uint8_t * restrict src_local = (ثابت uint8_t *)src + (ir * src0_row_size_aligned);
        uint8_t * restrict dst_local       = (uint8_t *)dst + (ir * dst_row_size_aligned);

        hvx_tanh_f32_aa(dst_local, src_local, ne0);
    }
}

ایستا تهی abs_f32(ثابت تهی * restrict src,
                    تهی * restrict dst,
                    ثابت uint32_t رقم_rows,
                    ثابت ساختار htp_unary_context * uctx) {
    htp_unary_op_preamble;

    برای (uint32_t ir = 0; ir < رقم_rows; ir++) {
        ثابت uint8_t * restrict src_local = (ثابت uint8_t *)src + (ir * src0_row_size_aligned);
        uint8_t * restrict dst_local       = (uint8_t *)dst + (ir * dst_row_size_aligned);

        hvx_abs_f32_aa(dst_local, src_local, ne0);
    }
}

ایستا تهی relu_f32(ثابت تهی * restrict src,
                     تهی * restrict dst,
                     ثابت uint32_t رقم_rows,
                     ثابت ساختار htp_unary_context * uctx) {
    htp_unary_op_preamble;

    برای (uint32_t ir = 0; ir < رقم_rows; ir++) {
        ثابت uint8_t * restrict src_local = (ثابت uint8_t *)src + (ir * src0_row_size_aligned);
        uint8_t * restrict dst_local       = (uint8_t *)dst + (ir * dst_row_size_aligned);

        hvx_max_scalar_f32(dst_local, src_local, 0.0f, ne0);
    }
}

ایستا تهی step_f32(ثابت تهی * restrict src,
                     تهی * restrict dst,
                     ثابت uint32_t رقم_rows,
                     ثابت ساختار htp_unary_context * uctx) {
    htp_unary_op_preamble;

    برای (uint32_t ir = 0; ir < رقم_rows; ir++) {
        ثابت uint8_t * restrict src_local = (ثابت uint8_t *)src + (ir * src0_row_size_aligned);
        uint8_t * restrict dst_local       = (uint8_t *)dst + (ir * dst_row_size_aligned);

        hvx_step_f32_aa(dst_local, src_local, ne0);
    }
}

ایستا تهی log_f32(ثابت تهی * restrict src,
                    تهی * restrict dst,
                    ثابت uint32_t رقم_rows,
                    ثابت ساختار htp_unary_context * uctx) {
    htp_unary_op_preamble;

    برای (uint32_t ir = 0; ir < رقم_rows; ir++) {
        ثابت uint8_t * restrict src_local = (ثابت uint8_t *)src + (ir * src0_row_size_aligned);
        uint8_t * restrict dst_local       = (uint8_t *)dst + (ir * dst_row_size_aligned);

        hvx_log_f32_aa(dst_local, src_local, ne0);
    }
}

#// Pointwise unary ops on one column tile in VTCM.
ایستا تهی tile_scale_f32(تهی * restrict dst, ثابت تهی * restrict src, uint32_t tw, ثابت ساختار htp_unary_context * uctx) {
    اعشاری scale = 0.f;
    اعشاری bias  = 0.f;
    memcpy(&scale, &uctx->octx->op_params[0], اندازه(اعشاری));
    memcpy(&bias,  &uctx->octx->op_params[1], اندازه(اعشاری));
    hvx_scale_offset_f32_aa((uint8_t *) dst, (ثابت uint8_t *) src, tw, scale, bias);
}

ایستا تهی tile_clamp_f32(تهی * restrict dst, ثابت تهی * restrict src, uint32_t tw, ثابت ساختار htp_unary_context * uctx) {
    اعشاری min = 0.f;
    اعشاری max = 0.f;
    memcpy(&min, &uctx->octx->op_params[0], اندازه(اعشاری));
    memcpy(&max, &uctx->octx->op_params[1], اندازه(اعشاری));
    hvx_clamp_scalar_f32((uint8_t *) dst, (ثابت uint8_t *) src, min, max, tw);
}

ایستا تهی tile_leaky_relu_f32(تهی * restrict dst, ثابت تهی * restrict src, uint32_t tw, ثابت ساختار htp_unary_context * uctx) {
    اعشاری negative_slope = 0.f;
    memcpy(&negative_slope, &uctx->octx->op_params[0], اندازه(اعشاری));
    hvx_leaky_relu_scalar_f32((uint8_t *) dst, (ثابت uint8_t *) src, negative_slope, tw);
}

ایستا تهی tile_sqr_f32(تهی * restrict dst, ثابت تهی * restrict src, uint32_t tw, ثابت ساختار htp_unary_context * uctx) {
    (تهی) uctx;
    hvx_sqr_f32_aa((uint8_t *) dst, (ثابت uint8_t *) src, tw);
}

ایستا تهی tile_sqrt_f32(تهی * restrict dst, ثابت تهی * restrict src, uint32_t tw, ثابت ساختار htp_unary_context * uctx) {
    (تهی) uctx;
    hvx_sqrt_f32_aa((uint8_t *) dst, (ثابت uint8_t *) src, tw);
}

ایستا تهی tile_neg_f32(تهی * restrict dst, ثابت تهی * restrict src, uint32_t tw, ثابت ساختار htp_unary_context * uctx) {
    (تهی) uctx;
    hvx_scale_f32_aa((uint8_t *) dst, (ثابت uint8_t *) src, tw, -1.0f);
}

ایستا تهی tile_exp_f32(تهی * restrict dst, ثابت تهی * restrict src, uint32_t tw, ثابت ساختار htp_unary_context * uctx) {
    (تهی) uctx;
    hvx_exp_f32((uint8_t *) dst, (ثابت uint8_t *) src, tw, false);
}

ایستا تهی tile_sigmoid_f32(تهی * restrict dst, ثابت تهی * restrict src, uint32_t tw, ثابت ساختار htp_unary_context * uctx) {
    (تهی) uctx;
    hvx_sigmoid_f32_aa((uint8_t *) dst, (ثابت uint8_t *) src, tw);
}

ایستا تهی tile_silu_f32(تهی * restrict dst, ثابت تهی * restrict src, uint32_t tw, ثابت ساختار htp_unary_context * uctx) {
    (تهی) uctx;
    hvx_sigmoid_f32_aa((uint8_t *) dst, (ثابت uint8_t *) src, tw);
    hvx_mul_f32_aaa((uint8_t *) dst, (ثابت uint8_t *) src, (uint8_t *) dst, tw);
}

ایستا تهی tile_gelu_f32(تهی * restrict dst, ثابت تهی * restrict src, uint32_t tw, ثابت ساختار htp_unary_context * uctx) {
    (تهی) uctx;
    hvx_gelu_f32_aa((uint8_t *) dst, (ثابت uint8_t *) src, tw);
}

ایستا تهی tile_gelu_quick_f32(تهی * restrict dst, ثابت تهی * restrict src, uint32_t tw, ثابت ساختار htp_unary_context * uctx) {
    (تهی) uctx;
    hvx_gelu_quick_f32_aa((uint8_t *) dst, (ثابت uint8_t *) src, tw);
}

ایستا تهی tile_gelu_erf_f32(تهی * restrict dst, ثابت تهی * restrict src, uint32_t tw, ثابت ساختار htp_unary_context * uctx) {
    (تهی) uctx;
    hvx_gelu_erf_f32_aa((uint8_t *) dst, (ثابت uint8_t *) src, tw);
}

ایستا تهی tile_softplus_f32(تهی * restrict dst, ثابت تهی * restrict src, uint32_t tw, ثابت ساختار htp_unary_context * uctx) {
    (تهی) uctx;
    hvx_softplus_f32_aa((uint8_t *) dst, (ثابت uint8_t *) src, tw);
}

ایستا تهی tile_tanh_f32(تهی * restrict dst, ثابت تهی * restrict src, uint32_t tw, ثابت ساختار htp_unary_context * uctx) {
    (تهی) uctx;
    hvx_tanh_f32_aa((uint8_t *) dst, (ثابت uint8_t *) src, tw);
}

ایستا تهی tile_abs_f32(تهی * restrict dst, ثابت تهی * restrict src, uint32_t tw, ثابت ساختار htp_unary_context * uctx) {
    (تهی) uctx;
    hvx_abs_f32_aa((uint8_t *) dst, (ثابت uint8_t *) src, tw);
}

ایستا تهی tile_log_f32(تهی * restrict dst, ثابت تهی * restrict src, uint32_t tw, ثابت ساختار htp_unary_context * uctx) {
    (تهی) uctx;
    hvx_log_f32_aa((uint8_t *) dst, (ثابت uint8_t *) src, tw);
}

ایستا تهی tile_relu_f32(تهی * restrict dst, ثابت تهی * restrict src, uint32_t tw, ثابت ساختار htp_unary_context * uctx) {
    (تهی) uctx;
    hvx_max_scalar_f32((uint8_t *) dst, (ثابت uint8_t *) src, 0.0f, tw);
}

ایستا تهی tile_step_f32(تهی * restrict dst, ثابت تهی * restrict src, uint32_t tw, ثابت ساختار htp_unary_context * uctx) {
    (تهی) uctx;
    hvx_step_f32_aa((uint8_t *) dst, (ثابت uint8_t *) src, tw);
}

ایستا تهی tri_apply_tile_f32(ثابت تهی * restrict src, تهی * restrict dst,
                               uint32_t tile_elems, uint32_t col_start, uint32_t i01,
                               uint32_t ne0, int32_t ttype) {
    ثابت HVX_Vector * restrict v_src = (ثابت HVX_Vector *) src;
    HVX_Vector * restrict v_dst       = (HVX_Vector *) dst;
    ثابت HVX_Vector zero = hvx_vec_splat_f32(0.0f);

    uint32_t boundary;
    صحیح      keep_چپ;
    انتخاب (ttype) {
        حالت 0: boundary = i01;     keep_چپ = 0; بشکن;
        حالت 1: boundary = i01 + 1; keep_چپ = 0; بشکن;
        حالت 2: boundary = i01 + 1; keep_چپ = 1; بشکن;
        حالت 3: boundary = i01;     keep_چپ = 1; بشکن;
        پیشفرض: boundary = 0; keep_چپ = 0; بشکن;
    }
    اگر (boundary > ne0) boundary = ne0;

    ثابت uint32_t nvec = tile_elems / VLEN_FP32;
    ثابت uint32_t nloe = tile_elems % VLEN_FP32;

    برای (uint32_t i = 0; i < nvec; i++) {
        ثابت uint32_t abs_start = col_start + i * VLEN_FP32;
        ثابت uint32_t abs_end   = abs_start + VLEN_FP32;
        اگر (keep_چپ) {
            اگر (abs_end <= boundary) {
                v_dst[i] = v_src[i];
            } وگرنه اگر (abs_start >= boundary) {
                v_dst[i] = zero;
            } وگرنه {
                HVX_VectorPred mask = Q6_Q_vsetq_R((boundary - abs_start) * اندازه(اعشاری));
                v_dst[i]            = Q6_V_vmux_QVV(mask, v_src[i], zero);
            }
        } وگرنه {
            اگر (abs_end <= boundary) {
                v_dst[i] = zero;
            } وگرنه اگر (abs_start >= boundary) {
                v_dst[i] = v_src[i];
            } وگرنه {
                HVX_VectorPred mask = Q6_Q_vsetq_R((boundary - abs_start) * اندازه(اعشاری));
                v_dst[i]            = Q6_V_vmux_QVV(mask, zero, v_src[i]);
            }
        }
    }

    اگر (nloe > 0) {
        ثابت uint32_t abs_start = col_start + nvec * VLEN_FP32;
        ثابت uint32_t abs_end   = abs_start + nloe;
        HVX_Vector     tail_val;
        اگر (keep_چپ) {
            اگر (abs_end <= boundary) {
                tail_val = v_src[nvec];
            } وگرنه اگر (abs_start >= boundary) {
                tail_val = zero;
            } وگرنه {
                HVX_VectorPred mask = Q6_Q_vsetq_R((boundary - abs_start) * اندازه(اعشاری));
                tail_val            = Q6_V_vmux_QVV(mask, v_src[nvec], zero);
            }
        } وگرنه {
            اگر (abs_end <= boundary) {
                tail_val = zero;
            } وگرنه اگر (abs_start >= boundary) {
                tail_val = v_src[nvec];
            } وگرنه {
                HVX_VectorPred mask = Q6_Q_vsetq_R((boundary - abs_start) * اندازه(اعشاری));
                tail_val            = Q6_V_vmux_QVV(mask, zero, v_src[nvec]);
            }
        }
        hvx_vec_store_a(&v_dst[nvec], nloe * اندازه(اعشاری), tail_val);
    }
}

// 1. Standard row-block unary task (F32 and F16).
ایستا تهی unary_thread_row_block(unsigned صحیح nth, unsigned صحیح ith, تهی * data) {
    (تهی) nth;
    ثابت ساختار htp_unary_context * uctx = (ثابت ساختار htp_unary_context *) data;
    ساختار htp_ops_context * octx = uctx->octx;
    ثابت ساختار htp_tensor * src = octx->src[0];
    ثابت ساختار htp_tensor * dst = octx->dst;
    ساختار htp_thread_trace * tr = &octx->ctx->trace[ith];

    htp_unary_preamble;

    ثابت uint32_t src0_nrows_per_thread = uctx->src0_nrows_per_thread;
    ثابت size_t src0_data_row_size = uctx->src0_data_row_size;
    ثابت size_t dst_data_row_size  = uctx->dst_data_row_size;
    ثابت size_t src0_row_size_aligned = uctx->src0_row_size_aligned;
    ثابت size_t dst_row_size_aligned  = uctx->dst_row_size_aligned;

    ثابت uint32_t src0_nrows = uctx->src0_nrows;
    ثابت uint32_t src0_start_row = uctx->row_start + src0_nrows_per_thread * ith;
    ثابت uint32_t src0_end_row   = MIN(src0_start_row + src0_nrows_per_thread, uctx->row_start + src0_nrows);

    اگر (src0_start_row >= src0_end_row) {
        بازگشت;
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

    ثابت ساختار fastdiv_مقادیر * div_ne01  = &uctx->kparams->div_ne01;
    ثابت ساختار fastdiv_مقادیر * div_ne02  = &uctx->kparams->div_ne02;
    ثابت ساختار fastdiv_مقادیر * div_ne012 = &uctx->kparams->div_ne012;

    ثابت uint32_t src0_max_block = src0_contig ? uctx->block : MIN((uint32_t)uctx->block, ne01);
    ثابت uint32_t dst_max_block  = dst_contig  ? uctx->block : MIN((uint32_t)uctx->block, ne1);
    ثابت uint32_t BLOCK = MIN(src0_max_block, dst_max_block);
    اگر (BLOCK == 0) {
        FARF(ERROR, "unary-row-block : current VTCM reservation %zu is too small, needed at least %zu\n",
             uctx->vtcm_src0_size_per_thread, src0_row_size_aligned);
        بازگشت;
    }

    dma_queue * dma_q = octx->ctx->dma[ith];

    برای (uint32_t ir = src0_start_row, vtcm_idx = 0; ir < src0_end_row && vtcm_idx < 2; vtcm_idx++) {
        ثابت uint32_t block_size = unary_block_size(ir, src0_end_row, BLOCK, src0_contig, dst_contig,
                                                     ne01, div_ne01);

        dma_queue_افزودن(dma_q,
            dma_make_data(data_dst, dst_vtcm_data + (vtcm_idx * dst_vtcm_half_size)),
            nb1, dst_row_size_aligned, dst_data_row_size, 0);

        ثابت size_t src0_off = src0_contig ? (ir * nb01) :
            unary_row_offset(ir, ne01, ne02, div_ne01, div_ne02, div_ne012, nb01, nb02, nb03);
        dma_queue_افزودن(dma_q,
            dma_make_data(src0_vtcm_data + (vtcm_idx * src0_vtcm_half_size), data_src + src0_off),
            src0_row_size_aligned, nb01, src0_data_row_size, block_size);

        ir += block_size;
    }

    unary_compute_fn_t compute = (unary_compute_fn_t) uctx->compute;

    برای (uint32_t ir = src0_start_row; ir < src0_end_row; ) {
        ثابت uint32_t block_size = unary_block_size(ir, src0_end_row, BLOCK, src0_contig, dst_contig,
                                                     ne01, div_ne01);

        تهی * dst_vtcm  = (تهی *) (uintptr_t) dma_queue_برداشت(dma_q).src;
        تهی * src0_vtcm = (تهی *) (uintptr_t) dma_queue_برداشت(dma_q).dst;

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, ir);
        compute(src0_vtcm, dst_vtcm, block_size, uctx);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ir);

        ثابت size_t dst_off = dst_contig ? (ir * nb1) :
            unary_row_offset(ir, ne1, ne2, div_ne01, div_ne02, div_ne012, nb1, nb2, nb3);
        dma_queue_افزودن(dma_q,
            dma_make_data(data_dst + dst_off, dst_vtcm),
            nb1, dst_row_size_aligned, dst_data_row_size, block_size);

        ثابت uint32_t next_ir = ir + block_size;
        اگر (next_ir < src0_end_row) {
            ثابت uint32_t next_block_size = unary_block_size(next_ir, src0_end_row, BLOCK, src0_contig,
                                                              dst_contig, ne01, div_ne01);
            ثابت uint32_t pref_ir = next_ir + next_block_size;
            اگر (pref_ir < src0_end_row) {
                ثابت uint32_t pref_block_size = unary_block_size(pref_ir, src0_end_row, BLOCK, src0_contig,
                                                                  dst_contig, ne01, div_ne01);
                ثابت size_t src0_pref_off = src0_contig ? (pref_ir * nb01) :
                    unary_row_offset(pref_ir, ne01, ne02, div_ne01, div_ne02, div_ne012, nb01, nb02, nb03);
                dma_queue_افزودن(dma_q,
                    dma_make_data(src0_vtcm, data_src + src0_pref_off),
                    src0_row_size_aligned, nb01, src0_data_row_size, pref_block_size);
            }
        }
        ir += block_size;
    }

    dma_queue_flush(dma_q);
}

// 2. RMS_NORM_MUL row-block task with weight buffer.
ایستا تهی unary_thread_rms_norm_mul_f32(unsigned صحیح nth, unsigned صحیح ith, تهی * data) {
    (تهی) nth;
    ثابت ساختار htp_unary_context * uctx = (ثابت ساختار htp_unary_context *) data;
    ساختار htp_ops_context * octx = uctx->octx;
    ثابت ساختار htp_tensor * src = octx->src[0];
    ثابت ساختار htp_tensor * dst = octx->dst;
    ساختار htp_thread_trace * tr = &octx->ctx->trace[ith];

    htp_unary_preamble;

    ثابت uint32_t src0_nrows_per_thread = uctx->src0_nrows_per_thread;
    ثابت size_t src0_data_row_size = uctx->src0_data_row_size;
    ثابت size_t dst_data_row_size  = uctx->dst_data_row_size;
    ثابت size_t src0_row_size_aligned = uctx->src0_row_size_aligned;
    ثابت size_t dst_row_size_aligned  = uctx->dst_row_size_aligned;

    ثابت uint32_t src0_nrows = uctx->src0_nrows;
    ثابت uint32_t src0_start_row = uctx->row_start + src0_nrows_per_thread * ith;
    ثابت uint32_t src0_end_row   = MIN(src0_start_row + src0_nrows_per_thread, uctx->row_start + src0_nrows);

    اگر (src0_start_row >= src0_end_row) {
        بازگشت;
    }

    ثابت dma_addr_t data_src  = uctx->data_src0;
    ثابت dma_addr_t data_src1 = uctx->data_src1;
    ثابت dma_addr_t data_dst  = uctx->data_dst;

    ثابت ساختار htp_tensor * src1 = octx->src[1];
    ثابت uint32_t nb11 = src1->nb[1];
    ثابت uint32_t nb12 = src1->nb[2];
    ثابت uint32_t nb13 = src1->nb[3];
    ثابت uint32_t nb11_bc = (src1->ne[1] > 1) ? nb11 : 0;
    ثابت uint32_t nb12_bc = (src1->ne[2] > 1) ? nb12 : 0;
    ثابت uint32_t nb13_bc = (src1->ne[3] > 1) ? nb13 : 0;
    ثابت bool src1_contig = ((nb12 == (size_t)ne01 * nb11) && (nb13 == (size_t)ne02 * nb12));

    uint8_t * src0_vtcm_data = uctx->vtcm_src0 + (ith * uctx->vtcm_src0_size_per_thread);
    uint8_t * src1_vtcm_data = uctx->vtcm_src1 ? (uctx->vtcm_src1 + (ith * uctx->vtcm_src1_size_per_thread)) : پوچ;
    uint8_t * dst_vtcm_data  = uctx->vtcm_dst + (ith * uctx->vtcm_dst_size_per_thread);

    ثابت size_t src0_vtcm_half_size = uctx->src0_vtcm_half_size;
    ثابت size_t src1_vtcm_half_size = uctx->src1_vtcm_half_size;
    ثابت size_t dst_vtcm_half_size  = uctx->dst_vtcm_half_size;

    ثابت bool src0_contig = (nb02 == (size_t)ne01 * nb01) &&
                             (nb03 == (size_t)ne02 * nb02);
    ثابت bool dst_contig  = (nb2  == (size_t)ne1  * nb1)  &&
                             (nb3  == (size_t)ne2  * nb2);

    ثابت ساختار fastdiv_مقادیر * div_ne01  = &uctx->kparams->div_ne01;
    ثابت ساختار fastdiv_مقادیر * div_ne02  = &uctx->kparams->div_ne02;
    ثابت ساختار fastdiv_مقادیر * div_ne012 = &uctx->kparams->div_ne012;

    ثابت bool src1_needs_row_clip = !uctx->broadcast_weight && !src1_contig;
    ثابت bool block_src0_contig = src0_contig && !src1_needs_row_clip;
    ثابت bool block_dst_contig  = dst_contig  && !src1_needs_row_clip;

    ثابت uint32_t src0_max_block = block_src0_contig ? uctx->block : MIN((uint32_t)uctx->block, ne01);
    ثابت uint32_t dst_max_block  = block_dst_contig  ? uctx->block : MIN((uint32_t)uctx->block, ne1);
    ثابت uint32_t BLOCK = MIN(src0_max_block, dst_max_block);
    اگر (BLOCK == 0) {
        FARF(ERROR, "unary-rms-norm-mul : current VTCM reservation %zu is too small, needed at least %zu\n",
             uctx->vtcm_src0_size_per_thread, src0_row_size_aligned);
        بازگشت;
    }

    dma_queue * dma_q = octx->ctx->dma[ith];

    اگر (uctx->broadcast_weight) {
        dma_queue_افزودن(dma_q, dma_make_data(src1_vtcm_data, data_src1),
                       uctx->src1_row_size_aligned, 0, uctx->src1_data_row_size, 1);
        dma_queue_flush(dma_q);
    }

    برای (uint32_t ir = src0_start_row, vtcm_idx = 0; ir < src0_end_row && vtcm_idx < 2; vtcm_idx++) {
        ثابت uint32_t block_size = unary_block_size(ir, src0_end_row, BLOCK, block_src0_contig, block_dst_contig,
                                                     ne01, div_ne01);

        dma_queue_افزودن(dma_q,
            dma_make_data(data_dst, dst_vtcm_data + (vtcm_idx * dst_vtcm_half_size)),
            nb1, dst_row_size_aligned, dst_data_row_size, 0);

        ثابت size_t src0_off = src0_contig ? (ir * nb01) :
            unary_row_offset(ir, ne01, ne02, div_ne01, div_ne02, div_ne012, nb01, nb02, nb03);
        dma_queue_افزودن(dma_q,
            dma_make_data(src0_vtcm_data + (vtcm_idx * src0_vtcm_half_size), data_src + src0_off),
            src0_row_size_aligned, nb01, src0_data_row_size, block_size);

        اگر (!uctx->broadcast_weight) {
            ثابت size_t src1_off = src1_contig ? (ir * nb11) :
                unary_row_offset(ir, ne01, ne02, div_ne01, div_ne02, div_ne012, nb11_bc, nb12_bc, nb13_bc);
            dma_queue_افزودن(dma_q,
                dma_make_data(src1_vtcm_data + (vtcm_idx * src1_vtcm_half_size), data_src1 + src1_off),
                uctx->src1_row_size_aligned, nb11, uctx->src1_data_row_size, block_size);
        }

        ir += block_size;
    }

    unary_rms_norm_mul_compute_fn_t compute = (unary_rms_norm_mul_compute_fn_t) uctx->compute;

    برای (uint32_t ir = src0_start_row; ir < src0_end_row; ) {
        ثابت uint32_t block_size = unary_block_size(ir, src0_end_row, BLOCK, block_src0_contig, block_dst_contig,
                                                     ne01, div_ne01);

        تهی * dst_vtcm  = (تهی *) (uintptr_t) dma_queue_برداشت(dma_q).src;
        تهی * src0_vtcm = (تهی *) (uintptr_t) dma_queue_برداشت(dma_q).dst;
        تهی * src1_vtcm = پوچ;
        اگر (!uctx->broadcast_weight) {
            src1_vtcm = (تهی *) (uintptr_t) dma_queue_برداشت(dma_q).dst;
        }

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, ir);
        ثابت تهی * w = uctx->broadcast_weight ? (ثابت تهی *) src1_vtcm_data : src1_vtcm;
        compute(src0_vtcm, w, dst_vtcm, block_size, uctx);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ir);

        ثابت size_t dst_off = dst_contig ? (ir * nb1) :
            unary_row_offset(ir, ne1, ne2, div_ne01, div_ne02, div_ne012, nb1, nb2, nb3);
        dma_queue_افزودن(dma_q,
            dma_make_data(data_dst + dst_off, dst_vtcm),
            nb1, dst_row_size_aligned, dst_data_row_size, block_size);

        ثابت uint32_t next_ir = ir + block_size;
        اگر (next_ir < src0_end_row) {
            ثابت uint32_t next_block_size = unary_block_size(next_ir, src0_end_row, BLOCK, block_src0_contig,
                                                              block_dst_contig, ne01, div_ne01);
            ثابت uint32_t pref_ir = next_ir + next_block_size;
            اگر (pref_ir < src0_end_row) {
                ثابت uint32_t pref_block_size = unary_block_size(pref_ir, src0_end_row, BLOCK, block_src0_contig,
                                                                  block_dst_contig, ne01, div_ne01);
                ثابت size_t src0_pref_off = src0_contig ? (pref_ir * nb01) :
                    unary_row_offset(pref_ir, ne01, ne02, div_ne01, div_ne02, div_ne012, nb01, nb02, nb03);
                dma_queue_افزودن(dma_q,
                    dma_make_data(src0_vtcm, data_src + src0_pref_off),
                    src0_row_size_aligned, nb01, src0_data_row_size, pref_block_size);

                اگر (!uctx->broadcast_weight) {
                    ثابت size_t src1_pref_off = src1_contig ? (pref_ir * nb11) :
                        unary_row_offset(pref_ir, ne01, ne02, div_ne01, div_ne02, div_ne012, nb11_bc, nb12_bc,
                                          nb13_bc);
                    dma_queue_افزودن(dma_q,
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
ایستا تهی unary_thread_tri_f32(unsigned صحیح nth, unsigned صحیح ith, تهی * data) {
    (تهی) nth;
    ثابت ساختار htp_unary_context * uctx = (ثابت ساختار htp_unary_context *) data;
    ساختار htp_ops_context * octx = uctx->octx;
    ثابت ساختار htp_tensor * src = octx->src[0];
    ثابت ساختار htp_tensor * dst = octx->dst;
    ساختار htp_thread_trace * tr = &octx->ctx->trace[ith];

    htp_unary_preamble;

    ثابت uint32_t src0_nrows_per_thread = uctx->src0_nrows_per_thread;
    ثابت size_t src0_data_row_size = uctx->src0_data_row_size;
    ثابت size_t dst_data_row_size  = uctx->dst_data_row_size;
    ثابت size_t src0_row_size_aligned = uctx->src0_row_size_aligned;
    ثابت size_t dst_row_size_aligned  = uctx->dst_row_size_aligned;

    ثابت uint32_t src0_nrows = uctx->src0_nrows;
    ثابت uint32_t src0_start_row = uctx->row_start + src0_nrows_per_thread * ith;
    ثابت uint32_t src0_end_row   = MIN(src0_start_row + src0_nrows_per_thread, uctx->row_start + src0_nrows);

    اگر (src0_start_row >= src0_end_row) {
        بازگشت;
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

    ثابت ساختار fastdiv_مقادیر * div_ne01  = &uctx->kparams->div_ne01;
    ثابت ساختار fastdiv_مقادیر * div_ne02  = &uctx->kparams->div_ne02;
    ثابت ساختار fastdiv_مقادیر * div_ne012 = &uctx->kparams->div_ne012;

    ثابت uint32_t src0_max_block = src0_contig ? uctx->block : MIN((uint32_t)uctx->block, ne01);
    ثابت uint32_t dst_max_block  = dst_contig  ? uctx->block : MIN((uint32_t)uctx->block, ne1);
    ثابت uint32_t BLOCK = MIN(src0_max_block, dst_max_block);
    اگر (BLOCK == 0) {
        FARF(ERROR, "unary-tri : current VTCM reservation %zu is too small, needed at least %zu\n",
             uctx->vtcm_src0_size_per_thread, src0_row_size_aligned);
        بازگشت;
    }

    dma_queue * dma_q = octx->ctx->dma[ith];

    برای (uint32_t ir = src0_start_row, vtcm_idx = 0; ir < src0_end_row && vtcm_idx < 2; vtcm_idx++) {
        ثابت uint32_t block_size = unary_block_size(ir, src0_end_row, BLOCK, src0_contig, dst_contig,
                                                     ne01, div_ne01);

        dma_queue_افزودن(dma_q,
            dma_make_data(data_dst, dst_vtcm_data + (vtcm_idx * dst_vtcm_half_size)),
            nb1, dst_row_size_aligned, dst_data_row_size, 0);

        ثابت size_t src0_off = src0_contig ? (ir * nb01) :
            unary_row_offset(ir, ne01, ne02, div_ne01, div_ne02, div_ne012, nb01, nb02, nb03);
        dma_queue_افزودن(dma_q,
            dma_make_data(src0_vtcm_data + (vtcm_idx * src0_vtcm_half_size), data_src + src0_off),
            src0_row_size_aligned, nb01, src0_data_row_size, block_size);

        ir += block_size;
    }

    unary_tri_compute_fn_t compute = (unary_tri_compute_fn_t) uctx->compute;

    برای (uint32_t ir = src0_start_row; ir < src0_end_row; ) {
        ثابت uint32_t block_size = unary_block_size(ir, src0_end_row, BLOCK, src0_contig, dst_contig,
                                                     ne01, div_ne01);

        تهی * dst_vtcm  = (تهی *) (uintptr_t) dma_queue_برداشت(dma_q).src;
        تهی * src0_vtcm = (تهی *) (uintptr_t) dma_queue_برداشت(dma_q).dst;

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, ir);
        compute(src0_vtcm, dst_vtcm, block_size, ir, uctx);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ir);

        ثابت size_t dst_off = dst_contig ? (ir * nb1) :
            unary_row_offset(ir, ne1, ne2, div_ne01, div_ne02, div_ne012, nb1, nb2, nb3);
        dma_queue_افزودن(dma_q,
            dma_make_data(data_dst + dst_off, dst_vtcm),
            nb1, dst_row_size_aligned, dst_data_row_size, block_size);

        ثابت uint32_t next_ir = ir + block_size;
        اگر (next_ir < src0_end_row) {
            ثابت uint32_t next_block_size = unary_block_size(next_ir, src0_end_row, BLOCK, src0_contig,
                                                              dst_contig, ne01, div_ne01);
            ثابت uint32_t pref_ir = next_ir + next_block_size;
            اگر (pref_ir < src0_end_row) {
                ثابت uint32_t pref_block_size = unary_block_size(pref_ir, src0_end_row, BLOCK, src0_contig,
                                                                  dst_contig, ne01, div_ne01);
                ثابت size_t src0_pref_off = src0_contig ? (pref_ir * nb01) :
                    unary_row_offset(pref_ir, ne01, ne02, div_ne01, div_ne02, div_ne012, nb01, nb02, nb03);
                dma_queue_افزودن(dma_q,
                    dma_make_data(src0_vtcm, data_src + src0_pref_off),
                    src0_row_size_aligned, nb01, src0_data_row_size, pref_block_size);
            }
        }
        ir += block_size;
    }

    dma_queue_flush(dma_q);
}

// 4. Pointwise tiled unary task.
ایستا تهی unary_thread_tiled(unsigned صحیح nth, unsigned صحیح ith, تهی * data) {
    (تهی) nth;
    ثابت ساختار htp_unary_context * uctx = (ثابت ساختار htp_unary_context *) data;
    ساختار htp_ops_context * octx = uctx->octx;
    ثابت ساختار htp_tensor * src = octx->src[0];
    ثابت ساختار htp_tensor * dst = octx->dst;
    ساختار htp_thread_trace * tr = &octx->ctx->trace[ith];

    htp_unary_preamble;

    ثابت uint32_t src0_nrows_per_thread = uctx->src0_nrows_per_thread;
    ثابت uint32_t col_tile              = uctx->col_tile;

    ثابت uint32_t src0_nrows     = uctx->src0_nrows;
    ثابت uint32_t src0_start_row = uctx->row_start + src0_nrows_per_thread * ith;
    ثابت uint32_t src0_end_row   = MIN(src0_start_row + src0_nrows_per_thread, uctx->row_start + src0_nrows);

    اگر (src0_start_row >= src0_end_row) {
        بازگشت;
    }

    ثابت dma_addr_t data_src = uctx->data_src0;
    ثابت dma_addr_t data_dst = uctx->data_dst;

    uint8_t * src0_vtcm_data = uctx->vtcm_src0 + (ith * uctx->vtcm_src0_size_per_thread);
    uint8_t * dst_vtcm_data  = uctx->vtcm_dst + (ith * uctx->vtcm_dst_size_per_thread);

    ثابت size_t src0_half = uctx->src0_vtcm_half_size;
    ثابت size_t dst_half  = uctx->dst_vtcm_half_size;

    dma_queue * dma_q = octx->ctx->dma[ith];

    ثابت ساختار fastdiv_مقادیر * div_ne01  = &uctx->kparams->div_ne01;
    ثابت ساختار fastdiv_مقادیر * div_ne02  = &uctx->kparams->div_ne02;
    ثابت ساختار fastdiv_مقادیر * div_ne012 = &uctx->kparams->div_ne012;
    ثابت ساختار fastdiv_مقادیر * div_tpr   = &uctx->kparams->div_tpr;

    ثابت uint32_t tiles_per_row = (ne0 + col_tile - 1) / col_tile;

    ثابت bool src0_contig = (nb02 == (size_t)ne01 * nb01) &&
                             (nb03 == (size_t)ne02 * nb02);
    ثابت bool dst_contig  = (nb2  == (size_t)ne1  * nb1)  &&
                             (nb3  == (size_t)ne2  * nb2);

    ثابت uint32_t total_tiles = (src0_end_row - src0_start_row) * tiles_per_row;

    برای (uint32_t t = 0, vtcm_idx = 0; t < total_tiles && vtcm_idx < 2; t++, vtcm_idx++) {
        ثابت uint32_t row  = src0_start_row + t / tiles_per_row;
        ثابت uint32_t col  = (t % tiles_per_row) * col_tile;
        ثابت uint32_t tw   = MIN(col_tile, ne0 - col);
        ثابت size_t   tb   = (size_t) tw * اندازه(اعشاری);
        ثابت size_t   soff = (src0_contig ? (row * nb01) :
                               unary_row_offset(row, ne01, ne02, div_ne01, div_ne02, div_ne012, nb01, nb02, nb03)) +
                               (size_t) col * اندازه(اعشاری);

        dma_queue_افزودن(dma_q, dma_make_data(data_dst, dst_vtcm_data + (vtcm_idx * dst_half)), 0, 0, 0, 0);
        dma_queue_افزودن(dma_q, dma_make_data(src0_vtcm_data + (vtcm_idx * src0_half), data_src + soff), tb, tb, tb, 1);
    }

    unary_tile_compute_fn_t compute = (unary_tile_compute_fn_t) uctx->compute;

    uint32_t row = src0_start_row;
    uint32_t col = 0;
    uint32_t tile_in_row = 0;

    uint32_t prow = src0_start_row + fastdiv(2, div_tpr);
    uint32_t pcol = fastmodulo(2, tiles_per_row, div_tpr) * col_tile;
    uint32_t ptile_in_row = fastmodulo(2, tiles_per_row, div_tpr);

    برای (uint32_t t = 0; t < total_tiles; t++) {
        تهی * dst_vtcm = (تهی *) (uintptr_t) dma_queue_برداشت(dma_q).src;
        تهی * src_vtcm = (تهی *) (uintptr_t) dma_queue_برداشت(dma_q).dst;

        ثابت uint32_t tw = MIN(col_tile, ne0 - col);

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, t);
        compute(dst_vtcm, src_vtcm, tw, uctx);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, t);

        ثابت size_t doff = (dst_contig ? (row * nb1) :
                             unary_row_offset(row, ne1, ne2, div_ne01, div_ne02, div_ne012, nb1, nb2, nb3)) +
                             (size_t) col * اندازه(اعشاری);
        ثابت size_t tb   = (size_t) tw * اندازه(اعشاری);
        dma_queue_افزودن(dma_q, dma_make_data(data_dst + doff, dst_vtcm), tb, tb, tb, 1);

        ثابت uint32_t pt = t + 2;
        اگر (pt < total_tiles) {
            ثابت uint32_t ptw  = MIN(col_tile, ne0 - pcol);
            ثابت size_t   ptb  = (size_t) ptw * اندازه(اعشاری);
            ثابت size_t   psoff = (src0_contig ? (prow * nb01) :
                                    unary_row_offset(prow, ne01, ne02, div_ne01, div_ne02, div_ne012, nb01, nb02,
                                                     nb03)) +
                                   (size_t) pcol * اندازه(اعشاری);
            dma_queue_افزودن(dma_q, dma_make_data(src_vtcm, data_src + psoff), ptb, ptb, ptb, 1);
        }

        tile_in_row++;
        col += col_tile;
        اگر (tile_in_row == tiles_per_row) {
            tile_in_row = 0;
            col = 0;
            row++;
        }

        ptile_in_row++;
        pcol += col_tile;
        اگر (ptile_in_row == tiles_per_row) {
            ptile_in_row = 0;
            pcol = 0;
            prow++;
        }
    }

    dma_queue_flush(dma_q);
}

// 5. TRI tiled task.
ایستا تهی unary_thread_tiled_tri_f32(unsigned صحیح nth, unsigned صحیح ith, تهی * data) {
    (تهی) nth;
    ثابت ساختار htp_unary_context * uctx = (ثابت ساختار htp_unary_context *) data;
    ساختار htp_ops_context * octx = uctx->octx;
    ثابت ساختار htp_tensor * src = octx->src[0];
    ثابت ساختار htp_tensor * dst = octx->dst;
    ساختار htp_thread_trace * tr = &octx->ctx->trace[ith];

    htp_unary_preamble;

    ثابت uint32_t src0_nrows_per_thread = uctx->src0_nrows_per_thread;
    ثابت int32_t * op_params            = octx->op_params;
    ثابت uint32_t col_tile              = uctx->col_tile;

    ثابت uint32_t src0_nrows     = uctx->src0_nrows;
    ثابت uint32_t src0_start_row = uctx->row_start + src0_nrows_per_thread * ith;
    ثابت uint32_t src0_end_row   = MIN(src0_start_row + src0_nrows_per_thread, uctx->row_start + src0_nrows);

    اگر (src0_start_row >= src0_end_row) {
        بازگشت;
    }

    ثابت dma_addr_t data_src = uctx->data_src0;
    ثابت dma_addr_t data_dst = uctx->data_dst;

    uint8_t * src0_vtcm_data = uctx->vtcm_src0 + (ith * uctx->vtcm_src0_size_per_thread);
    uint8_t * dst_vtcm_data  = uctx->vtcm_dst + (ith * uctx->vtcm_dst_size_per_thread);

    ثابت size_t src0_half = uctx->src0_vtcm_half_size;
    ثابت size_t dst_half  = uctx->dst_vtcm_half_size;

    dma_queue * dma_q = octx->ctx->dma[ith];

    ثابت ساختار fastdiv_مقادیر * div_ne01  = &uctx->kparams->div_ne01;
    ثابت ساختار fastdiv_مقادیر * div_ne02  = &uctx->kparams->div_ne02;
    ثابت ساختار fastdiv_مقادیر * div_ne012 = &uctx->kparams->div_ne012;
    ثابت ساختار fastdiv_مقادیر * div_tpr   = &uctx->kparams->div_tpr;

    ثابت uint32_t tiles_per_row = (ne0 + col_tile - 1) / col_tile;
    ثابت int32_t  tri_ttype     = op_params[0];

    ثابت bool src0_contig = (nb02 == (size_t)ne01 * nb01) &&
                             (nb03 == (size_t)ne02 * nb02);
    ثابت bool dst_contig  = (nb2  == (size_t)ne1  * nb1)  &&
                             (nb3  == (size_t)ne2  * nb2);

    ثابت uint32_t total_tiles = (src0_end_row - src0_start_row) * tiles_per_row;

    برای (uint32_t t = 0, vtcm_idx = 0; t < total_tiles && vtcm_idx < 2; t++, vtcm_idx++) {
        ثابت uint32_t row  = src0_start_row + t / tiles_per_row;
        ثابت uint32_t col  = (t % tiles_per_row) * col_tile;
        ثابت uint32_t tw   = MIN(col_tile, ne0 - col);
        ثابت size_t   tb   = (size_t) tw * اندازه(اعشاری);
        ثابت size_t   soff = (src0_contig ? (row * nb01) :
                               unary_row_offset(row, ne01, ne02, div_ne01, div_ne02, div_ne012, nb01, nb02, nb03)) +
                               (size_t) col * اندازه(اعشاری);

        dma_queue_افزودن(dma_q, dma_make_data(data_dst, dst_vtcm_data + (vtcm_idx * dst_half)), 0, 0, 0, 0);
        dma_queue_افزودن(dma_q, dma_make_data(src0_vtcm_data + (vtcm_idx * src0_half), data_src + soff), tb, tb, tb, 1);
    }

    unary_tiled_tri_compute_fn_t compute = (unary_tiled_tri_compute_fn_t) uctx->compute;

    uint32_t row = src0_start_row;
    uint32_t col = 0;
    uint32_t tile_in_row = 0;
    uint32_t i01 = fastmodulo(row, ne01, div_ne01);

    uint32_t prow = src0_start_row + fastdiv(2, div_tpr);
    uint32_t pcol = fastmodulo(2, tiles_per_row, div_tpr) * col_tile;
    uint32_t ptile_in_row = fastmodulo(2, tiles_per_row, div_tpr);

    برای (uint32_t t = 0; t < total_tiles; t++) {
        تهی * dst_vtcm = (تهی *) (uintptr_t) dma_queue_برداشت(dma_q).src;
        تهی * src_vtcm = (تهی *) (uintptr_t) dma_queue_برداشت(dma_q).dst;

        ثابت uint32_t tw = MIN(col_tile, ne0 - col);

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, t);
        compute(src_vtcm, dst_vtcm, tw, col, i01, ne0, tri_ttype);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, t);

        ثابت size_t doff = (dst_contig ? (row * nb1) :
                             unary_row_offset(row, ne1, ne2, div_ne01, div_ne02, div_ne012, nb1, nb2, nb3)) +
                             (size_t) col * اندازه(اعشاری);
        ثابت size_t tb   = (size_t) tw * اندازه(اعشاری);
        dma_queue_افزودن(dma_q, dma_make_data(data_dst + doff, dst_vtcm), tb, tb, tb, 1);

        ثابت uint32_t pt = t + 2;
        اگر (pt < total_tiles) {
            ثابت uint32_t ptw  = MIN(col_tile, ne0 - pcol);
            ثابت size_t   ptb  = (size_t) ptw * اندازه(اعشاری);
            ثابت size_t   psoff = (src0_contig ? (prow * nb01) :
                                    unary_row_offset(prow, ne01, ne02, div_ne01, div_ne02, div_ne012, nb01, nb02,
                                                     nb03)) +
                                   (size_t) pcol * اندازه(اعشاری);
            dma_queue_افزودن(dma_q, dma_make_data(src_vtcm, data_src + psoff), ptb, ptb, ptb, 1);
        }

        tile_in_row++;
        col += col_tile;
        اگر (tile_in_row == tiles_per_row) {
            tile_in_row = 0;
            col = 0;
            row++;
            i01++;
            اگر (i01 == ne01) {
                i01 = 0;
            }
        }

        ptile_in_row++;
        pcol += col_tile;
        اگر (ptile_in_row == tiles_per_row) {
            ptile_in_row = 0;
            pcol = 0;
            prow++;
        }
    }

    dma_queue_flush(dma_q);
}

ایستا صحیح execute_op_unary(ساختار htp_ops_context * octx) {
    صحیح err = HTP_STATUS_OK;

    ثابت ساختار htp_tensor * src0 = octx->src[0];
    ثابت ساختار htp_tensor * dst  = octx->dst;

    ثابت bool is_f16 = (src0->type == HTP_TYPE_F16);

    ثابت نویسه * op_type = پوچ;

    انتخاب (octx->op) {
        حالت HTP_OP_NORM:            op_type = is_f16 ? "norm-f16"     : "norm-f32";         بشکن;
        حالت HTP_OP_RMS_NORM:        op_type = is_f16 ? "rmsnorm-f16"  : "rmsnorm-f32";      بشکن;
        حالت HTP_OP_RMS_NORM_MUL:    op_type = "rmsnorm-mul-f32";                            بشکن;
        حالت HTP_OP_SCALE:           op_type = is_f16 ? "scale-f16"    : "scale-f32";        بشکن;
        حالت HTP_OP_CLAMP:           op_type = is_f16 ? "clamp-f16"    : "clamp-f32";        بشکن;
        حالت HTP_OP_LEAKY_RELU:      op_type = "leaky-relu-f32";                             بشکن;
        حالت HTP_OP_SQR:             op_type = is_f16 ? "sqr-f16"      : "sqr-f32";          بشکن;
        حالت HTP_OP_SQRT:            op_type = is_f16 ? "sqrt-f16"     : "sqrt-f32";         بشکن;
        حالت HTP_OP_UNARY_NEG:       op_type = "neg-f32";                                    بشکن;
        حالت HTP_OP_UNARY_EXP:       op_type = "exp-f32";                                    بشکن;
        حالت HTP_OP_UNARY_SIGMOID:   op_type = "sigmoid-f32";                                بشکن;
        حالت HTP_OP_UNARY_SILU:      op_type = "silu-f32";                                   بشکن;
        حالت HTP_OP_UNARY_GELU:      op_type = "gelu-f32";                                   بشکن;
        حالت HTP_OP_UNARY_GELU_QUICK: op_type = "gelu-quick-f32";                             بشکن;
        حالت HTP_OP_UNARY_GELU_ERF:  op_type = "gelu-erf-f32";                               بشکن;
        حالت HTP_OP_UNARY_SOFTPLUS:  op_type = "softplus-f32";                               بشکن;
        حالت HTP_OP_UNARY_TANH:      op_type = "tanh-f32";                                   بشکن;
        حالت HTP_OP_UNARY_ABS:       op_type = is_f16 ? "abs-f16"      : "abs-f32";          بشکن;
        حالت HTP_OP_UNARY_LOG:       op_type = is_f16 ? "log-f16"      : "log-f32";          بشکن;
        حالت HTP_OP_UNARY_RELU:      op_type = "relu-f32";                                   بشکن;
        حالت HTP_OP_UNARY_STEP:      op_type = is_f16 ? "step-f16"     : "step-f32";         بشکن;
        حالت HTP_OP_L2_NORM:         op_type = is_f16 ? "l2norm-f16"   : "l2norm-f32";       بشکن;
        حالت HTP_OP_TRI:             op_type = "tri-f32";                                    بشکن;
        حالت HTP_OP_HADAMARD:        op_type = is_f16 ? "hadamard-f16" : "hadamard-f32";     بشکن;
        پیشفرض:
            FARF(ERROR, "Unsupported unary Op %u\n", octx->op);
            بازگشت HTP_STATUS_NO_SUPPORT;
    }

    // F16 only has row-block kernels for this subset of ops (see the dispatch switch
    // below) - reject everything else up front, before touching kparams/VTCM.
    اگر (is_f16) {
        انتخاب (octx->op) {
            حالت HTP_OP_NORM:
            حالت HTP_OP_RMS_NORM:
            حالت HTP_OP_SCALE:
            حالت HTP_OP_CLAMP:
            حالت HTP_OP_SQR:
            حالت HTP_OP_SQRT:
            حالت HTP_OP_L2_NORM:
            حالت HTP_OP_UNARY_ABS:
            حالت HTP_OP_UNARY_LOG:
            حالت HTP_OP_UNARY_STEP:
            حالت HTP_OP_HADAMARD:
                بشکن;
            پیشفرض:
                FARF(ERROR, "unary-%s: not supported for F16\n", op_type);
                بازگشت HTP_STATUS_NO_SUPPORT;
        }
    }

    ثابت ساختار htp_unary_kernel_params * kparams = (ثابت ساختار htp_unary_kernel_params *) octx->kernel_params;

    اگر (!htp_ops_context_set_n_threads(octx, kparams->n_threads)) {
        بازگشت HTP_STATUS_INVAL_PARAMS;
    }

    ثابت uint32_t src0_nrows = src0->ne[1] * src0->ne[2] * src0->ne[3];
    ثابت size_t src0_elem_size = is_f16 ? اندازه(_Float16) : اندازه(اعشاری);
    ثابت size_t dst_elem_size  = (dst->type == HTP_TYPE_F16) ? اندازه(_Float16) : اندازه(اعشاری);
    ثابت size_t src0_data_row_size = src0->ne[0] * src0_elem_size;
    ثابت size_t dst_data_row_size  = dst->ne[0]  * dst_elem_size;

    uint32_t row_start = 0;
    uint32_t nrows     = src0_nrows;

    اگر (octx->ctx->mdev.شمار > 1) {
        uint32_t rows_per_chunk = 0;
        htp_tensor_mdev_rows_per_chunk(dst, (uint32_t) dst_elem_size, (uint32_t) dst_data_row_size, &rows_per_chunk);
        ثابت ساختار htp_tensor_mdev_range range = htp_tensor_mdev_partition(src0_nrows, rows_per_chunk, octx->ctx->mdev.idx, octx->ctx->mdev.شمار, &octx->ctx->mdev.شمار_div);
        row_start = range.start;
        nrows     = range.شمار;
    }

    اگر (nrows == 0) {
        بازگشت HTP_STATUS_OK;
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
    ثابت ساختار htp_tensor * src1 = پوچ;

    // RMS_NORM_MUL fusion is F32-only (its weight tensor is always F32; see
    // try_fuse_node()'s type guard), so this never triggers when is_f16 is true.
    اگر (octx->op == HTP_OP_RMS_NORM_MUL) {
        src1 = octx->src[1];
        src1_data_row_size = src1->ne[0] * اندازه(اعشاری);
    }

    اگر (octx->ctx->vtcm_size < (size_t)kparams->vtcm_size) {
        FARF(ERROR, "unary-%s : current VTCM reservation %zu is too small, needed %zu\n", op_type, octx->ctx->vtcm_size, (size_t)kparams->vtcm_size);
        بازگشت HTP_STATUS_VTCM_TOO_SMALL;
    }

    octx->src0_spad.src = پوچ;
    octx->src1_spad.src = پوچ;
    octx->dst_spad.src  = پوچ;

    FARF(بالا_2, "%s: (%ux%ux%ux%u) -> (%ux%ux%ux%u) : src0-vtcm-size %u src1-vtcm-size %u dst-vtcm-size %u\n", op_type,
         src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3], dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3],
         kparams->vtcm_src0_size, kparams->vtcm_src1_size, kparams->vtcm_dst_size);

    uint8_t * ثابت base = (uint8_t *) octx->ctx->vtcm_base;
    ساختار htp_unary_context uctx = {
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

    FARF(بالا_2, "%s: %s mode (col_tile %u)\n", op_type, col_tile ? "tiled" : "row-block", col_tile);

    worker_callback_t task_func = پوچ;
    تهی * compute_func = پوچ;

    اگر (col_tile) {
        task_func = unary_thread_tiled;
        انتخاب (octx->op) {
            حالت HTP_OP_SCALE:           compute_func = (تهی *) tile_scale_f32;          بشکن;
            حالت HTP_OP_CLAMP:           compute_func = (تهی *) tile_clamp_f32;          بشکن;
            حالت HTP_OP_LEAKY_RELU:      compute_func = (تهی *) tile_leaky_relu_f32;     بشکن;
            حالت HTP_OP_SQR:             compute_func = (تهی *) tile_sqr_f32;            بشکن;
            حالت HTP_OP_SQRT:            compute_func = (تهی *) tile_sqrt_f32;           بشکن;
            حالت HTP_OP_UNARY_NEG:       compute_func = (تهی *) tile_neg_f32;            بشکن;
            حالت HTP_OP_UNARY_EXP:       compute_func = (تهی *) tile_exp_f32;            بشکن;
            حالت HTP_OP_UNARY_SIGMOID:   compute_func = (تهی *) tile_sigmoid_f32;        بشکن;
            حالت HTP_OP_UNARY_SILU:      compute_func = (تهی *) tile_silu_f32;           بشکن;
            حالت HTP_OP_UNARY_GELU:      compute_func = (تهی *) tile_gelu_f32;           بشکن;
            حالت HTP_OP_UNARY_GELU_QUICK: compute_func = (تهی *) tile_gelu_quick_f32;     بشکن;
            حالت HTP_OP_UNARY_GELU_ERF:  compute_func = (تهی *) tile_gelu_erf_f32;       بشکن;
            حالت HTP_OP_UNARY_SOFTPLUS:  compute_func = (تهی *) tile_softplus_f32;       بشکن;
            حالت HTP_OP_UNARY_TANH:      compute_func = (تهی *) tile_tanh_f32;           بشکن;
            حالت HTP_OP_UNARY_ABS:       compute_func = (تهی *) tile_abs_f32;            بشکن;
            حالت HTP_OP_UNARY_LOG:       compute_func = (تهی *) tile_log_f32;            بشکن;
            حالت HTP_OP_UNARY_RELU:      compute_func = (تهی *) tile_relu_f32;           بشکن;
            حالت HTP_OP_UNARY_STEP:      compute_func = (تهی *) tile_step_f32;           بشکن;
            حالت HTP_OP_TRI:
                task_func    = unary_thread_tiled_tri_f32;
                compute_func = (تهی *) tri_apply_tile_f32;
                بشکن;
            پیشفرض:                     بشکن;
        }
    } وگرنه اگر (is_f16) {
        task_func = unary_thread_row_block;
        انتخاب (octx->op) {
            حالت HTP_OP_NORM:            compute_func = (تهی *) norm_f16;                بشکن;
            حالت HTP_OP_RMS_NORM:        compute_func = (تهی *) rms_norm_f16;            بشکن;
            حالت HTP_OP_SCALE:           compute_func = (تهی *) scale_f16;               بشکن;
            حالت HTP_OP_CLAMP:           compute_func = (تهی *) clamp_f16;               بشکن;
            حالت HTP_OP_SQR:             compute_func = (تهی *) sqr_f16;                 بشکن;
            حالت HTP_OP_SQRT:            compute_func = (تهی *) sqrt_f16;                بشکن;
            حالت HTP_OP_L2_NORM:         compute_func = (تهی *) l2_norm_f16;             بشکن;
            حالت HTP_OP_UNARY_ABS:       compute_func = (تهی *) abs_f16;                 بشکن;
            حالت HTP_OP_UNARY_LOG:       compute_func = (تهی *) log_f16;                 بشکن;
            حالت HTP_OP_UNARY_STEP:      compute_func = (تهی *) step_f16;                بشکن;
            حالت HTP_OP_HADAMARD:        compute_func = (تهی *) hadamard_f16;            بشکن;
            پیشفرض:                     بشکن;
        }
    } وگرنه {
        task_func = unary_thread_row_block;
        انتخاب (octx->op) {
            حالت HTP_OP_NORM:            compute_func = (تهی *) norm_f32;                بشکن;
            حالت HTP_OP_RMS_NORM:        compute_func = (تهی *) rms_norm_f32;            بشکن;
            حالت HTP_OP_RMS_NORM_MUL:
                task_func    = unary_thread_rms_norm_mul_f32;
                compute_func = (تهی *) rms_norm_mul_f32;
                بشکن;
            حالت HTP_OP_SCALE:           compute_func = (تهی *) scale_f32;               بشکن;
            حالت HTP_OP_CLAMP:           compute_func = (تهی *) clamp_f32;               بشکن;
            حالت HTP_OP_LEAKY_RELU:      compute_func = (تهی *) leaky_relu_f32;          بشکن;
            حالت HTP_OP_SQR:             compute_func = (تهی *) sqr_f32;                 بشکن;
            حالت HTP_OP_SQRT:            compute_func = (تهی *) sqrt_f32;                بشکن;
            حالت HTP_OP_UNARY_NEG:       compute_func = (تهی *) neg_f32;                 بشکن;
            حالت HTP_OP_UNARY_EXP:       compute_func = (تهی *) exp_f32;                 بشکن;
            حالت HTP_OP_UNARY_SIGMOID:   compute_func = (تهی *) sigmoid_f32;             بشکن;
            حالت HTP_OP_UNARY_SILU:      compute_func = (تهی *) silu_f32;                بشکن;
            حالت HTP_OP_UNARY_GELU:      compute_func = (تهی *) gelu_f32;                بشکن;
            حالت HTP_OP_UNARY_GELU_QUICK: compute_func = (تهی *) gelu_quick_f32;          بشکن;
            حالت HTP_OP_UNARY_GELU_ERF:  compute_func = (تهی *) gelu_erf_f32;            بشکن;
            حالت HTP_OP_UNARY_SOFTPLUS:  compute_func = (تهی *) softplus_f32;            بشکن;
            حالت HTP_OP_UNARY_TANH:      compute_func = (تهی *) tanh_f32;                بشکن;
            حالت HTP_OP_UNARY_ABS:       compute_func = (تهی *) abs_f32;                 بشکن;
            حالت HTP_OP_UNARY_LOG:       compute_func = (تهی *) log_f32;                 بشکن;
            حالت HTP_OP_UNARY_RELU:      compute_func = (تهی *) relu_f32;                بشکن;
            حالت HTP_OP_UNARY_STEP:      compute_func = (تهی *) step_f32;                بشکن;
            حالت HTP_OP_L2_NORM:         compute_func = (تهی *) l2_norm_f32;             بشکن;
            حالت HTP_OP_HADAMARD:        compute_func = (تهی *) hadamard_f32;            بشکن;
            حالت HTP_OP_TRI:
                task_func    = unary_thread_tri_f32;
                compute_func = (تهی *) tri_f32;
                بشکن;
            پیشفرض:                     بشکن;
        }
    }

    اگر (!task_func || !compute_func) {
        FARF(ERROR, "execute_op_unary: task function is NULL for op %d\n", octx->op);
        بازگشت HTP_STATUS_NO_SUPPORT;
    }

    uctx.compute = compute_func;
    work_queue_run(octx->ctx->work_queue, task_func, &uctx, n_threads);

    بازگشت err;
}

صحیح op_unary(ساختار htp_ops_context * octx) {
    انتخاب (octx->src[0]->type) {
        حالت HTP_TYPE_F32:
        حالت HTP_TYPE_F16:
            بازگشت execute_op_unary(octx);

        پیشفرض:
            بازگشت HTP_STATUS_NO_SUPPORT;
    }
}
