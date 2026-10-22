#لارښوونه clang diagnostic ignored "-Wunused-variable"
#لارښوونه clang diagnostic ignored "-Wunused-function"
#لارښوونه clang diagnostic ignored "-Wunused-but-set-variable"

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

#تعریف_کړه GGML_COMMON_DECL_C
#ضمن "ggml-common.h"
#ضمن "htp-ctx.h"
#ضمن "htp-ops.h"
#ضمن "htp-tensor.h"
#ضمن "htp-vtcm.h"
#ضمن "hex-profile.h"

بنية htp_unary_context;

تعريفنوع فراغ (*unary_compute_fn_t)(ثابت فراغ * محدود src,
                                   فراغ * محدود dst,
                                   uint32_t عدد_لیکې_2,
                                   ثابت بنية htp_unary_context * uctx);

تعريفنوع فراغ (*unary_rms_norm_mul_compute_fn_t)(ثابت فراغ * محدود src,
                                                ثابت فراغ * محدود وزن,
                                                فراغ * محدود dst,
                                                uint32_t عدد_لیکې_2,
                                                ثابت بنية htp_unary_context * uctx);

تعريفنوع فراغ (*unary_tri_compute_fn_t)(ثابت فراغ * محدود src,
                                       فراغ * محدود dst,
                                       uint32_t عدد_لیکې_2,
                                       uint32_t ir,
                                       ثابت بنية htp_unary_context * uctx);

تعريفنوع فراغ (*unary_tile_compute_fn_t)(فراغ * محدود dst,
                                        ثابت فراغ * محدود src,
                                        uint32_t tw,
                                        ثابت بنية htp_unary_context * uctx);

تعريفنوع فراغ (*unary_tiled_tri_compute_fn_t)(ثابت فراغ * محدود src,
                                             فراغ * محدود dst,
                                             uint32_t tile_elems,
                                             uint32_t ستون_پیل,
                                             uint32_t i01,
                                             uint32_t ne0,
                                             int32_t ttype);

بنية htp_unary_context {
    بنية htp_ops_context * octx;
    ثابت بنية htp_unary_kernel_params * kparams;

    فراغ *                    compute;

    dma_addr_t                معلومات_src0;
    dma_addr_t                معلومات_src1;            // weight/scale tensor for RMS_NORM_MUL
    dma_addr_t                معلومات_dst;

    size_t                    src0_معلومات_لیکه_اندازه;   // actual data bytes per row
    size_t                    src1_معلومات_لیکه_اندازه;
    size_t                    dst_معلومات_لیکه_اندازه;    // actual data bytes per row

    size_t                    src0_لیکه_اندازه_aligned;
    size_t                    src1_لیکه_اندازه_aligned;
    size_t                    dst_لیکه_اندازه_aligned;

    size_t                    src0_vtcm_half_اندازه;
    size_t                    src1_vtcm_half_اندازه;
    size_t                    dst_vtcm_half_اندازه;

    uint32_t                  بلاک;
    uint32_t                  src0_nrows;
    uint32_t                  src0_nrows_فی_thread;
    uint32_t                  لیکه_پیل;
    uint32_t                  nc;
    uint32_t                  ستون_tile;             // tiled mode
    bool                      خپرونه_وزن;

    uint8_t *                 vtcm_src0;
    uint8_t *                 vtcm_src1;
    uint8_t *                 vtcm_dst;

    size_t                    vtcm_src0_اندازه_فی_thread;
    size_t                    vtcm_src1_اندازه_فی_thread;
    size_t                    vtcm_dst_اندازه_فی_thread;
};

// Convert flat row index to DDR byte offset using the tensor's actual strides.
// ir = i1 + ne1*(i2 + ne2*i3)  =>  offset = i1*nb1 + i2*nb2 + i3*nb3
ساكن مضمن size_t unary_لیکه_دننه_واټن(uint32_t ir,
                                      uint32_t ne1, uint32_t ne2,
                                      ثابت بنية fastdiv_ارزښتونه * div_ne1,
                                      ثابت بنية fastdiv_ارزښتونه * div_ne2,
                                      ثابت بنية fastdiv_ارزښتونه * div_ne12,
                                      size_t nb1, size_t nb2, size_t nb3) {
    ثابت uint32_t i1 = fastmodulo(ir, ne1, div_ne1);
    ثابت uint32_t ir_div_ne1 = fastdiv(ir, div_ne1);
    ثابت uint32_t i2 = fastmodulo(ir_div_ne1, ne2, div_ne2);
    ثابت uint32_t i3 = fastdiv(ir, div_ne12);
    أرجع i1 * nb1 + i2 * nb2 + i3 * nb3;
}

// Safe DMA block size from row `ir`: clamp to the tighter dim-1 slice
// boundary of src and dst so the nb1 stride stays valid for all rows.
ساكن مضمن uint32_t unary_بلاک_اندازه(uint32_t ir,
                                        uint32_t پای_ټکی_لیکه,
                                        uint32_t بلاک,
                                        bool src_contig,
                                        bool dst_contig,
                                        uint32_t ne1,
                                        ثابت بنية fastdiv_ارزښتونه * div_ne1) {
    uint32_t حد = MIN(بلاک, پای_ټکی_لیکه - ir);

    إذا (!src_contig || !dst_contig) {
        ثابت uint32_t ټوټه_پای_ټکی = (fastdiv(ir, div_ne1) + 1) * ne1;
        حد = MIN(حد, ټوټه_پای_ټکی - ir);
    }

    أرجع حد;
}

#تعریف_کړه htp_unary_preamble            \
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

#تعریف_کړه htp_unary_op_preamble                                         \
    int32_t * op_params = uctx->octx->op_params;                      \
    ثابت uint32_t ne0 = uctx->nc;                                    \
    ثابت size_t src0_لیکه_اندازه_aligned = uctx->src0_لیکه_اندازه_aligned; \
    ثابت size_t dst_لیکه_اندازه_aligned = uctx->dst_لیکه_اندازه_aligned;

ساكن فراغ scale_f32(ثابت فراغ * محدود src,
                      فراغ * محدود dst,
                      ثابت uint32_t عدد_لیکې_2,
                      ثابت بنية htp_unary_context * uctx) {
    htp_unary_op_preamble;
    عائم scale = 0.f;
    عائم bias  = 0.f;
    حافظه_نقل(&scale, &op_params[0], حجم(عائم));
    حافظه_نقل(&bias,  &op_params[1], حجم(عائم));

    لكل (uint32_t ir = 0; ir < عدد_لیکې_2; ir++) {
        ثابت uint8_t * محدود src_local = (ثابت uint8_t *)src + (ir * src0_لیکه_اندازه_aligned);
        uint8_t * محدود dst_local       = (uint8_t *)dst + (ir * dst_لیکه_اندازه_aligned);

        hvx_scale_دننه_واټن_f32_aa((uint8_t *) dst_local, (ثابت uint8_t *) src_local, ne0, scale, bias);
    }
}

ساكن فراغ clamp_f32(ثابت فراغ * محدود src,
                      فراغ * محدود dst,
                      ثابت uint32_t عدد_لیکې_2,
                      ثابت بنية htp_unary_context * uctx) {
    htp_unary_op_preamble;
    عائم min = 0.f;
    عائم max = 0.f;
    حافظه_نقل(&min, &op_params[0], حجم(عائم));
    حافظه_نقل(&max, &op_params[1], حجم(عائم));

    لكل (uint32_t ir = 0; ir < عدد_لیکې_2; ir++) {
        ثابت uint8_t * محدود src_local = (ثابت uint8_t *)src + (ir * src0_لیکه_اندازه_aligned);
        uint8_t * محدود dst_local       = (uint8_t *)dst + (ir * dst_لیکه_اندازه_aligned);

        hvx_clamp_scalar_f32(dst_local, src_local, min, max, ne0);
    }
}

ساكن فراغ leaky_relu_f32(ثابت فراغ * محدود src,
                           فراغ * محدود dst,
                           ثابت uint32_t عدد_لیکې_2,
                           ثابت بنية htp_unary_context * uctx) {
    htp_unary_op_preamble;
    عائم negative_میلان = 0.f;
    حافظه_نقل(&negative_میلان, &op_params[0], حجم(عائم));

    لكل (uint32_t ir = 0; ir < عدد_لیکې_2; ir++) {
        ثابت uint8_t * محدود src_local = (ثابت uint8_t *)src + (ir * src0_لیکه_اندازه_aligned);
        uint8_t * محدود dst_local       = (uint8_t *)dst + (ir * dst_لیکه_اندازه_aligned);

        hvx_leaky_relu_scalar_f32(dst_local, src_local, negative_میلان, ne0);
    }
}

ساكن فراغ rms_norm_f32(ثابت فراغ * محدود src,
                         فراغ * محدود dst,
                         ثابت uint32_t عدد_لیکې_2,
                         ثابت بنية htp_unary_context * uctx) {
    htp_unary_op_preamble;
    عائم epsilon = 0.f;
    حافظه_نقل(&epsilon, op_params, حجم(عائم));

    لكل (uint32_t ir = 0; ir < عدد_لیکې_2; ir++) {
        ثابت uint8_t * محدود src_local = (ثابت uint8_t *)src + (ir * src0_لیکه_اندازه_aligned);
        uint8_t * محدود dst_local       = (uint8_t *)dst + (ir * dst_لیکه_اندازه_aligned);

        hvx_fast_rms_norm_f32((ثابت uint8_t *) src_local, (uint8_t *) dst_local, ne0, epsilon);
    }
}

ساكن فراغ rms_norm_mul_f32(ثابت فراغ * محدود src,
                             ثابت فراغ * محدود وزن,
                             فراغ * محدود dst,
                             ثابت uint32_t عدد_لیکې_2,
                             ثابت بنية htp_unary_context * uctx) {
    htp_unary_op_preamble;
    عائم epsilon = 0.f;
    حافظه_نقل(&epsilon, op_params, حجم(عائم));

    لكل (uint32_t ir = 0; ir < عدد_لیکې_2; ir++) {
        ثابت uint8_t * محدود src_local = (ثابت uint8_t *)src + (ir * src0_لیکه_اندازه_aligned);
        ثابت uint8_t * محدود w_local   = (ثابت uint8_t *)وزن + (uctx->خپرونه_وزن ? 0 : ir * uctx->src1_لیکه_اندازه_aligned);
        uint8_t * محدود dst_local       = (uint8_t *)dst + (ir * dst_لیکه_اندازه_aligned);

        hvx_fast_rms_norm_mul_f32(src_local, w_local, dst_local, ne0, epsilon);
    }
}

ساكن فراغ norm_f32(ثابت فراغ * محدود src,
                     فراغ * محدود dst,
                     ثابت uint32_t عدد_لیکې_2,
                     ثابت بنية htp_unary_context * uctx) {
    htp_unary_op_preamble;
    عائم epsilon = 0.f;
    حافظه_نقل(&epsilon, op_params, حجم(عائم));

    لكل (uint32_t ir = 0; ir < عدد_لیکې_2; ir++) {
        ثابت uint8_t * محدود src_local = (ثابت uint8_t *)src + (ir * src0_لیکه_اندازه_aligned);
        uint8_t * محدود dst_local       = (uint8_t *)dst + (ir * dst_لیکه_اندازه_aligned);

        hvx_fast_norm_f32((ثابت uint8_t *) src_local, (uint8_t *) dst_local, ne0, epsilon);
    }
}

ساكن فراغ sqr_f32(ثابت فراغ * محدود src,
                    فراغ * محدود dst,
                    ثابت uint32_t عدد_لیکې_2,
                    ثابت بنية htp_unary_context * uctx) {
    htp_unary_op_preamble;

    لكل (uint32_t ir = 0; ir < عدد_لیکې_2; ir++) {
        ثابت uint8_t * محدود src_local = (ثابت uint8_t *)src + (ir * src0_لیکه_اندازه_aligned);
        uint8_t * محدود dst_local       = (uint8_t *)dst + (ir * dst_لیکه_اندازه_aligned);

        hvx_sqr_f32_aa((uint8_t *) dst_local, (ثابت uint8_t *) src_local, ne0);
    }
}

ساكن فراغ sqrt_f32(ثابت فراغ * محدود src,
                     فراغ * محدود dst,
                     ثابت uint32_t عدد_لیکې_2,
                     ثابت بنية htp_unary_context * uctx) {
    htp_unary_op_preamble;

    لكل (uint32_t ir = 0; ir < عدد_لیکې_2; ir++) {
        ثابت uint8_t * محدود src_local = (ثابت uint8_t *)src + (ir * src0_لیکه_اندازه_aligned);
        uint8_t * محدود dst_local       = (uint8_t *)dst + (ir * dst_لیکه_اندازه_aligned);

        hvx_sqrt_f32_aa((uint8_t *) dst_local, (ثابت uint8_t *) src_local, ne0);
    }
}

ساكن فراغ scale_f16(ثابت فراغ * محدود src,
                      فراغ * محدود dst,
                      ثابت uint32_t عدد_لیکې_2,
                      ثابت بنية htp_unary_context * uctx) {
    htp_unary_op_preamble;
    عائم scale = 0.f;
    عائم bias  = 0.f;
    حافظه_نقل(&scale, &op_params[0], حجم(عائم));
    حافظه_نقل(&bias,  &op_params[1], حجم(عائم));

    لكل (uint32_t ir = 0; ir < عدد_لیکې_2; ir++) {
        ثابت uint8_t * محدود src_local = (ثابت uint8_t *)src + (ir * src0_لیکه_اندازه_aligned);
        uint8_t * محدود dst_local       = (uint8_t *)dst + (ir * dst_لیکه_اندازه_aligned);

        hvx_scale_دننه_واټن_f16_aa((uint8_t *) dst_local, (ثابت uint8_t *) src_local, ne0, scale, bias);
    }
}

ساكن فراغ clamp_f16(ثابت فراغ * محدود src,
                      فراغ * محدود dst,
                      ثابت uint32_t عدد_لیکې_2,
                      ثابت بنية htp_unary_context * uctx) {
    htp_unary_op_preamble;
    عائم min = 0.f;
    عائم max = 0.f;
    حافظه_نقل(&min, &op_params[0], حجم(عائم));
    حافظه_نقل(&max, &op_params[1], حجم(عائم));

    لكل (uint32_t ir = 0; ir < عدد_لیکې_2; ir++) {
        ثابت uint8_t * محدود src_local = (ثابت uint8_t *)src + (ir * src0_لیکه_اندازه_aligned);
        uint8_t * محدود dst_local       = (uint8_t *)dst + (ir * dst_لیکه_اندازه_aligned);

        hvx_clamp_scalar_f16(dst_local, src_local, (_Float16) min, (_Float16) max, ne0);
    }
}

ساكن فراغ rms_norm_f16(ثابت فراغ * محدود src,
                         فراغ * محدود dst,
                         ثابت uint32_t عدد_لیکې_2,
                         ثابت بنية htp_unary_context * uctx) {
    htp_unary_op_preamble;
    عائم epsilon = 0.f;
    حافظه_نقل(&epsilon, op_params, حجم(عائم));

    لكل (uint32_t ir = 0; ir < عدد_لیکې_2; ir++) {
        ثابت uint8_t * محدود src_local = (ثابت uint8_t *)src + (ir * src0_لیکه_اندازه_aligned);
        uint8_t * محدود dst_local       = (uint8_t *)dst + (ir * dst_لیکه_اندازه_aligned);

        hvx_fast_rms_norm_f16((ثابت uint8_t *) src_local, (uint8_t *) dst_local, ne0, epsilon);
    }
}

ساكن فراغ norm_f16(ثابت فراغ * محدود src,
                     فراغ * محدود dst,
                     ثابت uint32_t عدد_لیکې_2,
                     ثابت بنية htp_unary_context * uctx) {
    htp_unary_op_preamble;
    عائم epsilon = 0.f;
    حافظه_نقل(&epsilon, op_params, حجم(عائم));

    لكل (uint32_t ir = 0; ir < عدد_لیکې_2; ir++) {
        ثابت uint8_t * محدود src_local = (ثابت uint8_t *)src + (ir * src0_لیکه_اندازه_aligned);
        uint8_t * محدود dst_local       = (uint8_t *)dst + (ir * dst_لیکه_اندازه_aligned);

        hvx_fast_norm_f16((ثابت uint8_t *) src_local, (uint8_t *) dst_local, ne0, epsilon);
    }
}

ساكن فراغ sqr_f16(ثابت فراغ * محدود src,
                    فراغ * محدود dst,
                    ثابت uint32_t عدد_لیکې_2,
                    ثابت بنية htp_unary_context * uctx) {
    htp_unary_op_preamble;

    لكل (uint32_t ir = 0; ir < عدد_لیکې_2; ir++) {
        ثابت uint8_t * محدود src_local = (ثابت uint8_t *)src + (ir * src0_لیکه_اندازه_aligned);
        uint8_t * محدود dst_local       = (uint8_t *)dst + (ir * dst_لیکه_اندازه_aligned);

        hvx_sqr_f16_aa((uint8_t *) dst_local, (ثابت uint8_t *) src_local, ne0);
    }
}

ساكن فراغ sqrt_f16(ثابت فراغ * محدود src,
                     فراغ * محدود dst,
                     ثابت uint32_t عدد_لیکې_2,
                     ثابت بنية htp_unary_context * uctx) {
    htp_unary_op_preamble;

    لكل (uint32_t ir = 0; ir < عدد_لیکې_2; ir++) {
        ثابت uint8_t * محدود src_local = (ثابت uint8_t *)src + (ir * src0_لیکه_اندازه_aligned);
        uint8_t * محدود dst_local       = (uint8_t *)dst + (ir * dst_لیکه_اندازه_aligned);

        hvx_sqrt_f16_aa((uint8_t *) dst_local, (ثابت uint8_t *) src_local, ne0);
    }
}

ساكن فراغ abs_f16(ثابت فراغ * محدود src,
                    فراغ * محدود dst,
                    ثابت uint32_t عدد_لیکې_2,
                    ثابت بنية htp_unary_context * uctx) {
    htp_unary_op_preamble;

    لكل (uint32_t ir = 0; ir < عدد_لیکې_2; ir++) {
        ثابت uint8_t * محدود src_local = (ثابت uint8_t *)src + (ir * src0_لیکه_اندازه_aligned);
        uint8_t * محدود dst_local       = (uint8_t *)dst + (ir * dst_لیکه_اندازه_aligned);

        hvx_abs_f16_aa((uint8_t *) dst_local, (ثابت uint8_t *) src_local, ne0);
    }
}

ساكن فراغ log_f16(ثابت فراغ * محدود src,
                    فراغ * محدود dst,
                    ثابت uint32_t عدد_لیکې_2,
                    ثابت بنية htp_unary_context * uctx) {
    htp_unary_op_preamble;

    لكل (uint32_t ir = 0; ir < عدد_لیکې_2; ir++) {
        ثابت uint8_t * محدود src_local = (ثابت uint8_t *)src + (ir * src0_لیکه_اندازه_aligned);
        uint8_t * محدود dst_local       = (uint8_t *)dst + (ir * dst_لیکه_اندازه_aligned);

        hvx_log_f16_aa((uint8_t *) dst_local, (ثابت uint8_t *) src_local, ne0);
    }
}

ساكن فراغ ګام_f16(ثابت فراغ * محدود src,
                     فراغ * محدود dst,
                     ثابت uint32_t عدد_لیکې_2,
                     ثابت بنية htp_unary_context * uctx) {
    htp_unary_op_preamble;

    لكل (uint32_t ir = 0; ir < عدد_لیکې_2; ir++) {
        ثابت uint8_t * محدود src_local = (ثابت uint8_t *)src + (ir * src0_لیکه_اندازه_aligned);
        uint8_t * محدود dst_local       = (uint8_t *)dst + (ir * dst_لیکه_اندازه_aligned);

        hvx_ګام_f16_aa((uint8_t *) dst_local, (ثابت uint8_t *) src_local, ne0);
    }
}

ساكن فراغ l2_norm_f16(ثابت فراغ * محدود src,
                        فراغ * محدود dst,
                        ثابت uint32_t عدد_لیکې_2,
                        ثابت بنية htp_unary_context * uctx) {
    htp_unary_op_preamble;
    عائم epsilon = 0.f;
    حافظه_نقل(&epsilon, op_params, حجم(عائم));

    لكل (uint32_t ir = 0; ir < عدد_لیکې_2; ir++) {
        ثابت uint8_t * محدود src_f = (ثابت uint8_t *)src + (ir * src0_لیکه_اندازه_aligned);
        uint8_t * محدود dst_f       = (uint8_t *)dst + (ir * dst_لیکه_اندازه_aligned);

        hvx_fast_l2_norm_f16((ثابت uint8_t *)src_f, (uint8_t *)dst_f, ne0, epsilon);
    }
}

ساكن فراغ neg_f32(ثابت فراغ * محدود src,
                    فراغ * محدود dst,
                    ثابت uint32_t عدد_لیکې_2,
                    ثابت بنية htp_unary_context * uctx) {
    htp_unary_op_preamble;

    لكل (uint32_t ir = 0; ir < عدد_لیکې_2; ir++) {
        ثابت uint8_t * محدود src_local = (ثابت uint8_t *)src + (ir * src0_لیکه_اندازه_aligned);
        uint8_t * محدود dst_local       = (uint8_t *)dst + (ir * dst_لیکه_اندازه_aligned);

        hvx_scale_f32_aa(dst_local, src_local, ne0, -1.0f);
    }
}

ساكن فراغ exp_f32(ثابت فراغ * محدود src,
                    فراغ * محدود dst,
                    ثابت uint32_t عدد_لیکې_2,
                    ثابت بنية htp_unary_context * uctx) {
    htp_unary_op_preamble;

    لكل (uint32_t ir = 0; ir < عدد_لیکې_2; ir++) {
        ثابت uint8_t * محدود src_local = (ثابت uint8_t *)src + (ir * src0_لیکه_اندازه_aligned);
        uint8_t * محدود dst_local       = (uint8_t *)dst + (ir * dst_لیکه_اندازه_aligned);

        hvx_exp_f32(dst_local, src_local, ne0, false);
    }
}

ساكن فراغ sigmoid_f32(ثابت فراغ * محدود src,
                        فراغ * محدود dst,
                        ثابت uint32_t عدد_لیکې_2,
                        ثابت بنية htp_unary_context * uctx) {
    htp_unary_op_preamble;

    لكل (uint32_t ir = 0; ir < عدد_لیکې_2; ir++) {
        ثابت uint8_t * محدود src_local = (ثابت uint8_t *)src + (ir * src0_لیکه_اندازه_aligned);
        uint8_t * محدود dst_local       = (uint8_t *)dst + (ir * dst_لیکه_اندازه_aligned);

        hvx_sigmoid_f32_aa(dst_local, src_local, ne0);
    }
}

// silu(x) = x * sigmoid(x)
ساكن فراغ silu_f32(ثابت فراغ * محدود src,
                     فراغ * محدود dst,
                     ثابت uint32_t عدد_لیکې_2,
                     ثابت بنية htp_unary_context * uctx) {
    htp_unary_op_preamble;

    لكل (uint32_t ir = 0; ir < عدد_لیکې_2; ir++) {
        ثابت uint8_t * محدود src_local = (ثابت uint8_t *)src + (ir * src0_لیکه_اندازه_aligned);
        uint8_t * محدود dst_local       = (uint8_t *)dst + (ir * dst_لیکه_اندازه_aligned);

        hvx_sigmoid_f32_aa(dst_local, src_local, ne0);
        hvx_mul_f32_aaa(dst_local, src_local, dst_local, ne0);
    }
}

// GELU uses the tanh approximation.
ساكن __attribute__((noinline)) HVX_ویکټور hvx_vec_gelu_f32(HVX_ویکټور x) {
    ثابت HVX_ویکټور half = hvx_vec_splat_f32(0.5f);
    ثابت HVX_ویکټور one  = hvx_vec_splat_f32(1.0f);

    HVX_ویکټور inner = hvx_vec_mul_f32_f32(x, x);
    inner = hvx_vec_mul_f32_f32(inner, hvx_vec_splat_f32(0.044715f));
    inner = hvx_vec_جمع_کړه_f32_f32(inner, one);
    inner = hvx_vec_mul_f32_f32(inner, x);
    inner = hvx_vec_mul_f32_f32(inner, hvx_vec_splat_f32(0.7978845608028654f));

    أرجع hvx_vec_mul_f32_f32(hvx_vec_mul_f32_f32(half, x),
                               hvx_vec_جمع_کړه_f32_f32(one, hvx_vec_tanh_f32(inner)));
}

ساكن مضمن فراغ hvx_gelu_f32_aa(uint8_t * محدود dst, ثابت uint8_t * محدود src, uint32_t n) {
    أكد((بینښه اوږد) dst % 128 == 0);
    أكد((بینښه اوږد) src % 128 == 0);

    HVX_ویکټور * محدود vdst = (HVX_ویکټور *) dst;
    HVX_ویکټور * محدود vsrc = (HVX_ویکټور *) src;

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
ساكن __attribute__((noinline)) HVX_ویکټور hvx_vec_gelu_چټک_f32(HVX_ویکټور x) {
    ثابت HVX_ویکټور one     = hvx_vec_splat_f32(1.0f);
    ثابت HVX_ویکټور max_exp = hvx_vec_splat_f32(87.0f);
    ثابت HVX_ویکټور min_exp = hvx_vec_splat_f32(-87.0f);
    ثابت HVX_ویکټور scaled  = hvx_vec_mul_f32_f32(x, hvx_vec_splat_f32(1.702f));
    ثابت HVX_ویکټور sigmoid = hvx_vec_fast_sigmoid_f32_guard(scaled, one, max_exp, min_exp);

    أرجع hvx_vec_mul_f32_f32(x, sigmoid);
}

ساكن مضمن فراغ hvx_gelu_چټک_f32_aa(uint8_t * محدود dst, ثابت uint8_t * محدود src, uint32_t n) {
    أكد((بینښه اوږد) dst % 128 == 0);
    أكد((بینښه اوږد) src % 128 == 0);

    HVX_ویکټور * محدود vdst = (HVX_ویکټور *) dst;
    HVX_ویکټور * محدود vsrc = (HVX_ویکټور *) src;

    ثابت uint32_t nvec = n / VLEN_FP32;
    ثابت uint32_t nloe = n % VLEN_FP32;

    uint32_t i = 0;
    _Pragma("unroll(4)")
    لكل (; i < nvec; i++) {
        vdst[i] = hvx_vec_gelu_چټک_f32(vsrc[i]);
    }
    إذا (nloe) {
        hvx_vec_store_a(&vdst[i], nloe * حجم(عائم), hvx_vec_gelu_چټک_f32(vsrc[i]));
    }
}

ساكن فراغ gelu_f32(ثابت فراغ * محدود src,
                     فراغ * محدود dst,
                     ثابت uint32_t عدد_لیکې_2,
                     ثابت بنية htp_unary_context * uctx) {
    htp_unary_op_preamble;

    لكل (uint32_t ir = 0; ir < عدد_لیکې_2; ir++) {
        ثابت uint8_t * محدود src_local = (ثابت uint8_t *)src + (ir * src0_لیکه_اندازه_aligned);
        uint8_t * محدود dst_local       = (uint8_t *)dst + (ir * dst_لیکه_اندازه_aligned);

        hvx_gelu_f32_aa(dst_local, src_local, ne0);
    }
}

ساكن فراغ gelu_چټک_f32(ثابت فراغ * محدود src,
                           فراغ * محدود dst,
                           ثابت uint32_t عدد_لیکې_2,
                           ثابت بنية htp_unary_context * uctx) {
    htp_unary_op_preamble;

    لكل (uint32_t ir = 0; ir < عدد_لیکې_2; ir++) {
        ثابت uint8_t * محدود src_local = (ثابت uint8_t *)src + (ir * src0_لیکه_اندازه_aligned);
        uint8_t * محدود dst_local       = (uint8_t *)dst + (ir * dst_لیکه_اندازه_aligned);

        hvx_gelu_چټک_f32_aa(dst_local, src_local, ne0);
    }
}

ساكن فراغ gelu_erf_f32(ثابت فراغ * محدود src,
                         فراغ * محدود dst,
                         ثابت uint32_t عدد_لیکې_2,
                         ثابت بنية htp_unary_context * uctx) {
    htp_unary_op_preamble;

    لكل (uint32_t ir = 0; ir < عدد_لیکې_2; ir++) {
        ثابت uint8_t * محدود src_local = (ثابت uint8_t *) src + (ir * src0_لیکه_اندازه_aligned);
        uint8_t * محدود dst_local       = (uint8_t *) dst + (ir * dst_لیکه_اندازه_aligned);

        hvx_gelu_erf_f32_aa(dst_local, src_local, ne0);
    }
}

ساكن فراغ tri_f32(ثابت فراغ * محدود src,
                    فراغ * محدود dst,
                    ثابت uint32_t عدد_لیکې_2,
                    ثابت uint32_t ir,
                    ثابت بنية htp_unary_context * uctx) {
    htp_unary_op_preamble;
    ثابت int32_t ttype = op_params[0];
    ثابت HVX_ویکټور zero = hvx_vec_splat_f32(0.0f);
    ثابت uint32_t nvec  = ne0 / VLEN_FP32;
    ثابت uint32_t nloe  = ne0 % VLEN_FP32;

    ثابت uint32_t ne01 = uctx->octx->src[0]->ne[1];

    لكل (uint32_t b = 0; b < عدد_لیکې_2; b++) {
        ثابت uint32_t abs_لیکه = ir + b;
        ثابت uint32_t i01     = abs_لیکه % ne01;

        ثابت HVX_ویکټور * محدود v_src = (ثابت HVX_ویکټور *) ((ثابت uint8_t *) src + b * src0_لیکه_اندازه_aligned);
        HVX_ویکټور * محدود v_dst       = (HVX_ویکټور *) ((uint8_t *) dst + b * dst_لیکه_اندازه_aligned);

        uint32_t boundary;
        صحيح      keep_کیڼ;
        بدل (ttype) {
            حالة 0: boundary = i01;     keep_کیڼ = 0; اكسر;  // keep col >= row
            حالة 1: boundary = i01 + 1; keep_کیڼ = 0; اكسر;  // keep col > row
            حالة 2: boundary = i01 + 1; keep_کیڼ = 1; اكسر;  // keep col <= row
            حالة 3: boundary = i01;     keep_کیڼ = 1; اكسر;  // keep col < row
            افتراضي: boundary = 0; keep_کیڼ = 0; اكسر;
        }
        إذا (boundary > ne0) boundary = ne0;

        // Full HVX vectors - each starts at a 128-byte aligned offset
        لكل (uint32_t i = 0; i < nvec; i++) {
            ثابت uint32_t vec_پیل = i * VLEN_FP32;
            ثابت uint32_t vec_پای_ټکی   = vec_پیل + VLEN_FP32;
            إذا (keep_کیڼ) {
                إذا (vec_پای_ټکی <= boundary) {
                    v_dst[i] = v_src[i];
                } وإلا إذا (vec_پیل >= boundary) {
                    v_dst[i] = zero;
                } وإلا {
                    HVX_ویکټور_Pred ماسک = Q6_Q_vsetq_R((boundary - vec_پیل) * حجم(عائم));
                    v_dst[i]            = Q6_V_vmux_QVV(ماسک, v_src[i], zero);
                }
            } وإلا {
                إذا (vec_پای_ټکی <= boundary) {
                    v_dst[i] = zero;
                } وإلا إذا (vec_پیل >= boundary) {
                    v_dst[i] = v_src[i];
                } وإلا {
                    HVX_ویکټور_Pred ماسک = Q6_Q_vsetq_R((boundary - vec_پیل) * حجم(عائم));
                    v_dst[i]            = Q6_V_vmux_QVV(ماسک, zero, v_src[i]);
                }
            }
        }

        // Tail elements (row_elems not a multiple of VLEN_FP32)
        إذا (nloe > 0) {
            ثابت uint32_t abs_پیل = nvec * VLEN_FP32;
            ثابت uint32_t abs_پای_ټکی   = abs_پیل + nloe;
            HVX_ویکټور     لکۍ_val;
            إذا (keep_کیڼ) {
                إذا (abs_پای_ټکی <= boundary) {
                    لکۍ_val = v_src[nvec];
                } وإلا إذا (abs_پیل >= boundary) {
                    لکۍ_val = zero;
                } وإلا {
                    HVX_ویکټور_Pred ماسک = Q6_Q_vsetq_R((boundary - abs_پیل) * حجم(عائم));
                    لکۍ_val            = Q6_V_vmux_QVV(ماسک, v_src[nvec], zero);
                }
            } وإلا {
                إذا (abs_پای_ټکی <= boundary) {
                    لکۍ_val = zero;
                } وإلا إذا (abs_پیل >= boundary) {
                    لکۍ_val = v_src[nvec];
                } وإلا {
                    HVX_ویکټور_Pred ماسک = Q6_Q_vsetq_R((boundary - abs_پیل) * حجم(عائم));
                    لکۍ_val            = Q6_V_vmux_QVV(ماسک, zero, v_src[nvec]);
                }
            }
            hvx_vec_store_a(&v_dst[nvec], nloe * حجم(عائم), لکۍ_val);
        }
    }
}

// softplus(x) = log(1 + exp(x))
// Match CPU reference: return x for x > 20 to avoid overflow.
ساكن مضمن فراغ hvx_softplus_f32_aa(uint8_t * محدود dst, ثابت uint8_t * محدود src, uint32_t n) {
    ثابت HVX_ویکټور * محدود v_src = (ثابت HVX_ویکټور *) src;
    HVX_ویکټور * محدود v_dst       = (HVX_ویکټور *) dst;

    ثابت HVX_ویکټور twenty = hvx_vec_splat_f32(20.0f);
    ثابت HVX_ویکټور one    = hvx_vec_splat_f32(1.0f);

    ثابت uint32_t nvec = n / VLEN_FP32;
    ثابت uint32_t nloe = n % VLEN_FP32;

    #لارښوونه unroll(4)
    لكل (uint32_t i = 0; i < nvec; i++) {
        HVX_ویکټور x              = v_src[i];
        HVX_ویکټور_Pred pred_gt20  = Q6_Q_vcmp_gt_VsfVsf(x, twenty);
        HVX_ویکټور expx           = hvx_vec_exp_f32(x);
        HVX_ویکټور logv           = hvx_vec_log_f32(hvx_vec_جمع_کړه_f32_f32(one, expx));
        v_dst[i]                  = Q6_V_vmux_QVV(pred_gt20, x, logv);
    }

    إذا (nloe > 0) {
        HVX_ویکټور x             = v_src[nvec];
        HVX_ویکټور_Pred pred_gt20 = Q6_Q_vcmp_gt_VsfVsf(x, twenty);
        HVX_ویکټور expx          = hvx_vec_exp_f32(x);
        HVX_ویکټور logv          = hvx_vec_log_f32(hvx_vec_جمع_کړه_f32_f32(one, expx));
        HVX_ویکټور sp            = Q6_V_vmux_QVV(pred_gt20, x, logv);
        hvx_vec_store_a((فراغ *) &v_dst[nvec], nloe * حجم(عائم), sp);
    }
}

ساكن فراغ softplus_f32(ثابت فراغ * محدود src,
                         فراغ * محدود dst,
                         ثابت uint32_t عدد_لیکې_2,
                         ثابت بنية htp_unary_context * uctx) {
    htp_unary_op_preamble;

    لكل (uint32_t ir = 0; ir < عدد_لیکې_2; ir++) {
        ثابت uint8_t * محدود src_local = (ثابت uint8_t *)src + (ir * src0_لیکه_اندازه_aligned);
        uint8_t * محدود dst_local       = (uint8_t *)dst + (ir * dst_لیکه_اندازه_aligned);

        hvx_softplus_f32_aa(dst_local, src_local, ne0);
    }
}

ساكن فراغ l2_norm_f32(ثابت فراغ * محدود src,
                        فراغ * محدود dst,
                        ثابت uint32_t عدد_لیکې_2,
                        ثابت بنية htp_unary_context * uctx) {
    htp_unary_op_preamble;
    عائم epsilon = 0.f;
    حافظه_نقل(&epsilon, op_params, حجم(عائم));

    لكل (uint32_t ir = 0; ir < عدد_لیکې_2; ir++) {
        ثابت uint8_t * محدود src_f = (ثابت uint8_t *)src + (ir * src0_لیکه_اندازه_aligned);
        uint8_t * محدود dst_f       = (uint8_t *)dst + (ir * dst_لیکه_اندازه_aligned);

        hvx_fast_l2_norm_f32((ثابت uint8_t *)src_f, (uint8_t *)dst_f, ne0, epsilon);
    }
}

ساكن مضمن HVX_ویکټور_Pair hvx_fwht_stage_pair(HVX_ویکټور v0, HVX_ویکټور v1, int32_t b) {
    HVX_ویکټور_Pair p = Q6_W_vshuff_VVR(v1, v0, b);
    HVX_ویکټور s = Q6_Vqf32_vadd_Vqf32Vqf32(Q6_V_lo_W(p), Q6_V_hi_W(p));
    HVX_ویکټور d = Q6_Vqf32_vsub_Vqf32Vqf32(Q6_V_lo_W(p), Q6_V_hi_W(p));
    أرجع Q6_W_vshuff_VVR(d, s, b);
}

ساكن مضمن HVX_ویکټور_Pair hvx_fwht32_pair(HVX_ویکټور v0_sf, HVX_ویکټور v1_sf, HVX_ویکټور v_scale) {
    HVX_ویکټور v0 = Q6_Vqf32_vmpy_VsfVsf(v0_sf, v_scale);
    HVX_ویکټور v1 = Q6_Vqf32_vmpy_VsfVsf(v1_sf, v_scale);
    HVX_ویکټور_Pair p;
    p = hvx_fwht_stage_pair(v0, v1, 4);
    p = hvx_fwht_stage_pair(Q6_V_lo_W(p), Q6_V_hi_W(p), 8);
    p = hvx_fwht_stage_pair(Q6_V_lo_W(p), Q6_V_hi_W(p), 16);
    p = hvx_fwht_stage_pair(Q6_V_lo_W(p), Q6_V_hi_W(p), 32);
    p = hvx_fwht_stage_pair(Q6_V_lo_W(p), Q6_V_hi_W(p), 64);
    أرجع p;
}

ساكن مضمن فراغ hvx_fwht_store32(HVX_ویکټور * محدود v_dst, HVX_ویکټور_Pair p) {
    v_dst[0] = Q6_Vsf_equals_Vqf32(Q6_V_lo_W(p));
}

ساكن مضمن فراغ hvx_fwht_store64(HVX_ویکټور * محدود v_dst, HVX_ویکټور_Pair p) {
    HVX_ویکټور u = Q6_V_lo_W(p);
    HVX_ویکټور v = Q6_V_hi_W(p);
    v_dst[0] = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vqf32(u, v));
    v_dst[1] = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vsub_Vqf32Vqf32(u, v));
}

ساكن مضمن فراغ hvx_fwht_لکۍ_f32(HVX_ویکټور * محدود v_dst, uint32_t nvec) {
    uint32_t len_vec = 2;
    لكل (; len_vec * 2 < nvec; len_vec <<= 2) {
        ثابت uint32_t ګام = len_vec * 4;
        لكل (uint32_t i = 0; i < nvec; i += ګام) {
            لكل (uint32_t j = 0; j < len_vec; j++) {
                HVX_ویکټور a0 = v_dst[i + j];
                HVX_ویکټور a1 = v_dst[i + j + len_vec];
                HVX_ویکټور a2 = v_dst[i + j + 2 * len_vec];
                HVX_ویکټور a3 = v_dst[i + j + 3 * len_vec];

                HVX_ویکټور t0 = Q6_Vqf32_vadd_VsfVsf(a0, a1);
                HVX_ویکټور t1 = Q6_Vqf32_vsub_VsfVsf(a0, a1);
                HVX_ویکټور t2 = Q6_Vqf32_vadd_VsfVsf(a2, a3);
                HVX_ویکټور t3 = Q6_Vqf32_vsub_VsfVsf(a2, a3);

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
                HVX_ویکټور u = v_dst[i + j];
                HVX_ویکټور v = v_dst[i + len_vec + j];
                v_dst[i + j]           = hvx_vec_جمع_کړه_f32_f32(u, v);
                v_dst[i + len_vec + j] = hvx_vec_sub_f32_f32(u, v);
            }
        }
    }
}

ساكن مضمن فراغ hvx_fwht_لیکه_f32(ثابت uint8_t * محدود src, uint8_t * محدود dst, uint32_t n, HVX_ویکټور v_scale) {
    ثابت HVX_ویکټور * محدود v_src = (ثابت HVX_ویکټور *) src;
    HVX_ویکټور * محدود v_dst       = (HVX_ویکټور *) dst;
    ثابت uint32_t nvec               = n / VLEN_FP32;

    إذا (nvec == 1) {
        HVX_ویکټور_Pair p = hvx_fwht32_pair(v_src[0], v_src[0], v_scale);
        hvx_fwht_store32(v_dst, p);
        أرجع;
    }

    لكل (uint32_t i = 0; i < nvec; i += 2) {
        HVX_ویکټور_Pair p = hvx_fwht32_pair(v_src[i], v_src[i + 1], v_scale);
        hvx_fwht_store64(&v_dst[i], p);
    }

    hvx_fwht_لکۍ_f32(v_dst, nvec);
}

ساكن مضمن فراغ hvx_fwht_لیکه_f16(ثابت uint8_t * محدود src, uint8_t * محدود dst, uint32_t n, HVX_ویکټور v_scale) {
    ثابت HVX_ویکټور * محدود v_src = (ثابت HVX_ویکټور *) src;
    HVX_ویکټور * محدود v_dst       = (HVX_ویکټور *) dst;
    ثابت uint32_t nvec               = n / VLEN_FP32;

    إذا (nvec == 1) {
        HVX_ویکټور v0 = Q6_V_lo_W(hvx_vec_f16_منزل_f32(v_src[0]));
        HVX_ویکټور_Pair p = hvx_fwht32_pair(v0, v0, v_scale);
        hvx_fwht_store32(v_dst, p);
        أرجع;
    }

    ثابت uint32_t nvec_f16 = n / VLEN_FP16;
    لكل (uint32_t i = 0; i < nvec_f16; i++) {
        HVX_ویکټور_Pair f32p = hvx_vec_f16_منزل_f32(v_src[i]);
        HVX_ویکټور_Pair p    = hvx_fwht32_pair(Q6_V_lo_W(f32p), Q6_V_hi_W(f32p), v_scale);
        hvx_fwht_store64(&v_dst[2 * i], p);
    }

    hvx_fwht_لکۍ_f32(v_dst, nvec);
}

ساكن فراغ hadamard_f32(ثابت فراغ * محدود src,
                         فراغ * محدود dst,
                         ثابت uint32_t عدد_لیکې_2,
                         ثابت بنية htp_unary_context * uctx) {
    htp_unary_op_preamble;
    ثابت HVX_ویکټور v_scale = hvx_vec_splat_f32(1.0f / sqrtf((عائم) ne0));

    لكل (uint32_t ir = 0; ir < عدد_لیکې_2; ir++) {
        ثابت uint8_t * محدود src_لیکه = (ثابت uint8_t *) src + (ir * src0_لیکه_اندازه_aligned);
        uint8_t * محدود dst_لیکه       = (uint8_t *) dst + (ir * dst_لیکه_اندازه_aligned);

        hvx_fwht_لیکه_f32(src_لیکه, dst_لیکه, ne0, v_scale);
    }
}

ساكن فراغ hadamard_f16(ثابت فراغ * محدود src,
                         فراغ * محدود dst,
                         ثابت uint32_t عدد_لیکې_2,
                         ثابت بنية htp_unary_context * uctx) {
    htp_unary_op_preamble;
    ثابت HVX_ویکټور v_scale = hvx_vec_splat_f32(1.0f / sqrtf((عائم) ne0));

    لكل (uint32_t ir = 0; ir < عدد_لیکې_2; ir++) {
        ثابت uint8_t * محدود src_لیکه = (ثابت uint8_t *) src + (ir * src0_لیکه_اندازه_aligned);
        uint8_t * محدود dst_لیکه       = (uint8_t *) dst + (ir * dst_لیکه_اندازه_aligned);

        hvx_fwht_لیکه_f16(src_لیکه, dst_لیکه, ne0, v_scale);
    }
}

ساكن فراغ tanh_f32(ثابت فراغ * محدود src,
                     فراغ * محدود dst,
                     ثابت uint32_t عدد_لیکې_2,
                     ثابت بنية htp_unary_context * uctx) {
    htp_unary_op_preamble;

    لكل (uint32_t ir = 0; ir < عدد_لیکې_2; ir++) {
        ثابت uint8_t * محدود src_local = (ثابت uint8_t *)src + (ir * src0_لیکه_اندازه_aligned);
        uint8_t * محدود dst_local       = (uint8_t *)dst + (ir * dst_لیکه_اندازه_aligned);

        hvx_tanh_f32_aa(dst_local, src_local, ne0);
    }
}

ساكن فراغ abs_f32(ثابت فراغ * محدود src,
                    فراغ * محدود dst,
                    ثابت uint32_t عدد_لیکې_2,
                    ثابت بنية htp_unary_context * uctx) {
    htp_unary_op_preamble;

    لكل (uint32_t ir = 0; ir < عدد_لیکې_2; ir++) {
        ثابت uint8_t * محدود src_local = (ثابت uint8_t *)src + (ir * src0_لیکه_اندازه_aligned);
        uint8_t * محدود dst_local       = (uint8_t *)dst + (ir * dst_لیکه_اندازه_aligned);

        hvx_abs_f32_aa(dst_local, src_local, ne0);
    }
}

ساكن فراغ relu_f32(ثابت فراغ * محدود src,
                     فراغ * محدود dst,
                     ثابت uint32_t عدد_لیکې_2,
                     ثابت بنية htp_unary_context * uctx) {
    htp_unary_op_preamble;

    لكل (uint32_t ir = 0; ir < عدد_لیکې_2; ir++) {
        ثابت uint8_t * محدود src_local = (ثابت uint8_t *)src + (ir * src0_لیکه_اندازه_aligned);
        uint8_t * محدود dst_local       = (uint8_t *)dst + (ir * dst_لیکه_اندازه_aligned);

        hvx_max_scalar_f32(dst_local, src_local, 0.0f, ne0);
    }
}

ساكن فراغ ګام_f32(ثابت فراغ * محدود src,
                     فراغ * محدود dst,
                     ثابت uint32_t عدد_لیکې_2,
                     ثابت بنية htp_unary_context * uctx) {
    htp_unary_op_preamble;

    لكل (uint32_t ir = 0; ir < عدد_لیکې_2; ir++) {
        ثابت uint8_t * محدود src_local = (ثابت uint8_t *)src + (ir * src0_لیکه_اندازه_aligned);
        uint8_t * محدود dst_local       = (uint8_t *)dst + (ir * dst_لیکه_اندازه_aligned);

        hvx_ګام_f32_aa(dst_local, src_local, ne0);
    }
}

ساكن فراغ log_f32(ثابت فراغ * محدود src,
                    فراغ * محدود dst,
                    ثابت uint32_t عدد_لیکې_2,
                    ثابت بنية htp_unary_context * uctx) {
    htp_unary_op_preamble;

    لكل (uint32_t ir = 0; ir < عدد_لیکې_2; ir++) {
        ثابت uint8_t * محدود src_local = (ثابت uint8_t *)src + (ir * src0_لیکه_اندازه_aligned);
        uint8_t * محدود dst_local       = (uint8_t *)dst + (ir * dst_لیکه_اندازه_aligned);

        hvx_log_f32_aa(dst_local, src_local, ne0);
    }
}

#// Pointwise unary ops on one column tile in VTCM.
ساكن فراغ tile_scale_f32(فراغ * محدود dst, ثابت فراغ * محدود src, uint32_t tw, ثابت بنية htp_unary_context * uctx) {
    عائم scale = 0.f;
    عائم bias  = 0.f;
    حافظه_نقل(&scale, &uctx->octx->op_params[0], حجم(عائم));
    حافظه_نقل(&bias,  &uctx->octx->op_params[1], حجم(عائم));
    hvx_scale_دننه_واټن_f32_aa((uint8_t *) dst, (ثابت uint8_t *) src, tw, scale, bias);
}

ساكن فراغ tile_clamp_f32(فراغ * محدود dst, ثابت فراغ * محدود src, uint32_t tw, ثابت بنية htp_unary_context * uctx) {
    عائم min = 0.f;
    عائم max = 0.f;
    حافظه_نقل(&min, &uctx->octx->op_params[0], حجم(عائم));
    حافظه_نقل(&max, &uctx->octx->op_params[1], حجم(عائم));
    hvx_clamp_scalar_f32((uint8_t *) dst, (ثابت uint8_t *) src, min, max, tw);
}

ساكن فراغ tile_leaky_relu_f32(فراغ * محدود dst, ثابت فراغ * محدود src, uint32_t tw, ثابت بنية htp_unary_context * uctx) {
    عائم negative_میلان = 0.f;
    حافظه_نقل(&negative_میلان, &uctx->octx->op_params[0], حجم(عائم));
    hvx_leaky_relu_scalar_f32((uint8_t *) dst, (ثابت uint8_t *) src, negative_میلان, tw);
}

ساكن فراغ tile_sqr_f32(فراغ * محدود dst, ثابت فراغ * محدود src, uint32_t tw, ثابت بنية htp_unary_context * uctx) {
    (فراغ) uctx;
    hvx_sqr_f32_aa((uint8_t *) dst, (ثابت uint8_t *) src, tw);
}

ساكن فراغ tile_sqrt_f32(فراغ * محدود dst, ثابت فراغ * محدود src, uint32_t tw, ثابت بنية htp_unary_context * uctx) {
    (فراغ) uctx;
    hvx_sqrt_f32_aa((uint8_t *) dst, (ثابت uint8_t *) src, tw);
}

ساكن فراغ tile_neg_f32(فراغ * محدود dst, ثابت فراغ * محدود src, uint32_t tw, ثابت بنية htp_unary_context * uctx) {
    (فراغ) uctx;
    hvx_scale_f32_aa((uint8_t *) dst, (ثابت uint8_t *) src, tw, -1.0f);
}

ساكن فراغ tile_exp_f32(فراغ * محدود dst, ثابت فراغ * محدود src, uint32_t tw, ثابت بنية htp_unary_context * uctx) {
    (فراغ) uctx;
    hvx_exp_f32((uint8_t *) dst, (ثابت uint8_t *) src, tw, false);
}

ساكن فراغ tile_sigmoid_f32(فراغ * محدود dst, ثابت فراغ * محدود src, uint32_t tw, ثابت بنية htp_unary_context * uctx) {
    (فراغ) uctx;
    hvx_sigmoid_f32_aa((uint8_t *) dst, (ثابت uint8_t *) src, tw);
}

ساكن فراغ tile_silu_f32(فراغ * محدود dst, ثابت فراغ * محدود src, uint32_t tw, ثابت بنية htp_unary_context * uctx) {
    (فراغ) uctx;
    hvx_sigmoid_f32_aa((uint8_t *) dst, (ثابت uint8_t *) src, tw);
    hvx_mul_f32_aaa((uint8_t *) dst, (ثابت uint8_t *) src, (uint8_t *) dst, tw);
}

ساكن فراغ tile_gelu_f32(فراغ * محدود dst, ثابت فراغ * محدود src, uint32_t tw, ثابت بنية htp_unary_context * uctx) {
    (فراغ) uctx;
    hvx_gelu_f32_aa((uint8_t *) dst, (ثابت uint8_t *) src, tw);
}

ساكن فراغ tile_gelu_چټک_f32(فراغ * محدود dst, ثابت فراغ * محدود src, uint32_t tw, ثابت بنية htp_unary_context * uctx) {
    (فراغ) uctx;
    hvx_gelu_چټک_f32_aa((uint8_t *) dst, (ثابت uint8_t *) src, tw);
}

ساكن فراغ tile_gelu_erf_f32(فراغ * محدود dst, ثابت فراغ * محدود src, uint32_t tw, ثابت بنية htp_unary_context * uctx) {
    (فراغ) uctx;
    hvx_gelu_erf_f32_aa((uint8_t *) dst, (ثابت uint8_t *) src, tw);
}

ساكن فراغ tile_softplus_f32(فراغ * محدود dst, ثابت فراغ * محدود src, uint32_t tw, ثابت بنية htp_unary_context * uctx) {
    (فراغ) uctx;
    hvx_softplus_f32_aa((uint8_t *) dst, (ثابت uint8_t *) src, tw);
}

ساكن فراغ tile_tanh_f32(فراغ * محدود dst, ثابت فراغ * محدود src, uint32_t tw, ثابت بنية htp_unary_context * uctx) {
    (فراغ) uctx;
    hvx_tanh_f32_aa((uint8_t *) dst, (ثابت uint8_t *) src, tw);
}

ساكن فراغ tile_abs_f32(فراغ * محدود dst, ثابت فراغ * محدود src, uint32_t tw, ثابت بنية htp_unary_context * uctx) {
    (فراغ) uctx;
    hvx_abs_f32_aa((uint8_t *) dst, (ثابت uint8_t *) src, tw);
}

ساكن فراغ tile_log_f32(فراغ * محدود dst, ثابت فراغ * محدود src, uint32_t tw, ثابت بنية htp_unary_context * uctx) {
    (فراغ) uctx;
    hvx_log_f32_aa((uint8_t *) dst, (ثابت uint8_t *) src, tw);
}

ساكن فراغ tile_relu_f32(فراغ * محدود dst, ثابت فراغ * محدود src, uint32_t tw, ثابت بنية htp_unary_context * uctx) {
    (فراغ) uctx;
    hvx_max_scalar_f32((uint8_t *) dst, (ثابت uint8_t *) src, 0.0f, tw);
}

ساكن فراغ tile_ګام_f32(فراغ * محدود dst, ثابت فراغ * محدود src, uint32_t tw, ثابت بنية htp_unary_context * uctx) {
    (فراغ) uctx;
    hvx_ګام_f32_aa((uint8_t *) dst, (ثابت uint8_t *) src, tw);
}

ساكن فراغ tri_apply_tile_f32(ثابت فراغ * محدود src, فراغ * محدود dst,
                               uint32_t tile_elems, uint32_t ستون_پیل, uint32_t i01,
                               uint32_t ne0, int32_t ttype) {
    ثابت HVX_ویکټور * محدود v_src = (ثابت HVX_ویکټور *) src;
    HVX_ویکټور * محدود v_dst       = (HVX_ویکټور *) dst;
    ثابت HVX_ویکټور zero = hvx_vec_splat_f32(0.0f);

    uint32_t boundary;
    صحيح      keep_کیڼ;
    بدل (ttype) {
        حالة 0: boundary = i01;     keep_کیڼ = 0; اكسر;
        حالة 1: boundary = i01 + 1; keep_کیڼ = 0; اكسر;
        حالة 2: boundary = i01 + 1; keep_کیڼ = 1; اكسر;
        حالة 3: boundary = i01;     keep_کیڼ = 1; اكسر;
        افتراضي: boundary = 0; keep_کیڼ = 0; اكسر;
    }
    إذا (boundary > ne0) boundary = ne0;

    ثابت uint32_t nvec = tile_elems / VLEN_FP32;
    ثابت uint32_t nloe = tile_elems % VLEN_FP32;

    لكل (uint32_t i = 0; i < nvec; i++) {
        ثابت uint32_t abs_پیل = ستون_پیل + i * VLEN_FP32;
        ثابت uint32_t abs_پای_ټکی   = abs_پیل + VLEN_FP32;
        إذا (keep_کیڼ) {
            إذا (abs_پای_ټکی <= boundary) {
                v_dst[i] = v_src[i];
            } وإلا إذا (abs_پیل >= boundary) {
                v_dst[i] = zero;
            } وإلا {
                HVX_ویکټور_Pred ماسک = Q6_Q_vsetq_R((boundary - abs_پیل) * حجم(عائم));
                v_dst[i]            = Q6_V_vmux_QVV(ماسک, v_src[i], zero);
            }
        } وإلا {
            إذا (abs_پای_ټکی <= boundary) {
                v_dst[i] = zero;
            } وإلا إذا (abs_پیل >= boundary) {
                v_dst[i] = v_src[i];
            } وإلا {
                HVX_ویکټور_Pred ماسک = Q6_Q_vsetq_R((boundary - abs_پیل) * حجم(عائم));
                v_dst[i]            = Q6_V_vmux_QVV(ماسک, zero, v_src[i]);
            }
        }
    }

    إذا (nloe > 0) {
        ثابت uint32_t abs_پیل = ستون_پیل + nvec * VLEN_FP32;
        ثابت uint32_t abs_پای_ټکی   = abs_پیل + nloe;
        HVX_ویکټور     لکۍ_val;
        إذا (keep_کیڼ) {
            إذا (abs_پای_ټکی <= boundary) {
                لکۍ_val = v_src[nvec];
            } وإلا إذا (abs_پیل >= boundary) {
                لکۍ_val = zero;
            } وإلا {
                HVX_ویکټور_Pred ماسک = Q6_Q_vsetq_R((boundary - abs_پیل) * حجم(عائم));
                لکۍ_val            = Q6_V_vmux_QVV(ماسک, v_src[nvec], zero);
            }
        } وإلا {
            إذا (abs_پای_ټکی <= boundary) {
                لکۍ_val = zero;
            } وإلا إذا (abs_پیل >= boundary) {
                لکۍ_val = v_src[nvec];
            } وإلا {
                HVX_ویکټور_Pred ماسک = Q6_Q_vsetq_R((boundary - abs_پیل) * حجم(عائم));
                لکۍ_val            = Q6_V_vmux_QVV(ماسک, zero, v_src[nvec]);
            }
        }
        hvx_vec_store_a(&v_dst[nvec], nloe * حجم(عائم), لکۍ_val);
    }
}

// 1. Standard row-block unary task (F32 and F16).
ساكن فراغ unary_thread_لیکه_بلاک(بینښه صحيح nth, بینښه صحيح ith, فراغ * معلومات) {
    (فراغ) nth;
    ثابت بنية htp_unary_context * uctx = (ثابت بنية htp_unary_context *) معلومات;
    بنية htp_ops_context * octx = uctx->octx;
    ثابت بنية htp_tensor * src = octx->src[0];
    ثابت بنية htp_tensor * dst = octx->dst;
    بنية htp_thread_trace * tr = &octx->ctx->trace[ith];

    htp_unary_preamble;

    ثابت uint32_t src0_nrows_فی_thread = uctx->src0_nrows_فی_thread;
    ثابت size_t src0_معلومات_لیکه_اندازه = uctx->src0_معلومات_لیکه_اندازه;
    ثابت size_t dst_معلومات_لیکه_اندازه  = uctx->dst_معلومات_لیکه_اندازه;
    ثابت size_t src0_لیکه_اندازه_aligned = uctx->src0_لیکه_اندازه_aligned;
    ثابت size_t dst_لیکه_اندازه_aligned  = uctx->dst_لیکه_اندازه_aligned;

    ثابت uint32_t src0_nrows = uctx->src0_nrows;
    ثابت uint32_t src0_پیل_لیکه = uctx->لیکه_پیل + src0_nrows_فی_thread * ith;
    ثابت uint32_t src0_پای_ټکی_لیکه   = MIN(src0_پیل_لیکه + src0_nrows_فی_thread, uctx->لیکه_پیل + src0_nrows);

    إذا (src0_پیل_لیکه >= src0_پای_ټکی_لیکه) {
        أرجع;
    }

    ثابت dma_addr_t معلومات_src = uctx->معلومات_src0;
    ثابت dma_addr_t معلومات_dst = uctx->معلومات_dst;

    uint8_t * src0_vtcm_معلومات = uctx->vtcm_src0 + (ith * uctx->vtcm_src0_اندازه_فی_thread);
    uint8_t * dst_vtcm_معلومات  = uctx->vtcm_dst + (ith * uctx->vtcm_dst_اندازه_فی_thread);

    ثابت size_t src0_vtcm_half_اندازه = uctx->src0_vtcm_half_اندازه;
    ثابت size_t dst_vtcm_half_اندازه  = uctx->dst_vtcm_half_اندازه;

    ثابت bool src0_contig = (nb02 == (size_t)ne01 * nb01) &&
                             (nb03 == (size_t)ne02 * nb02);
    ثابت bool dst_contig  = (nb2  == (size_t)ne1  * nb1)  &&
                             (nb3  == (size_t)ne2  * nb2);

    ثابت بنية fastdiv_ارزښتونه * div_ne01  = &uctx->kparams->div_ne01;
    ثابت بنية fastdiv_ارزښتونه * div_ne02  = &uctx->kparams->div_ne02;
    ثابت بنية fastdiv_ارزښتونه * div_ne012 = &uctx->kparams->div_ne012;

    ثابت uint32_t src0_max_بلاک = src0_contig ? uctx->بلاک : MIN((uint32_t)uctx->بلاک, ne01);
    ثابت uint32_t dst_max_بلاک  = dst_contig  ? uctx->بلاک : MIN((uint32_t)uctx->بلاک, ne1);
    ثابت uint32_t بلاک_2 = MIN(src0_max_بلاک, dst_max_بلاک);
    إذا (بلاک_2 == 0) {
        FARF(ERROR, "unary-row-block : current VTCM reservation %zu is too small, needed at least %zu\n",
             uctx->vtcm_src0_اندازه_فی_thread, src0_لیکه_اندازه_aligned);
        أرجع;
    }

    dma_کتار * dma_q = octx->ctx->dma[ith];

    لكل (uint32_t ir = src0_پیل_لیکه, vtcm_idx = 0; ir < src0_پای_ټکی_لیکه && vtcm_idx < 2; vtcm_idx++) {
        ثابت uint32_t بلاک_اندازه_3 = unary_بلاک_اندازه(ir, src0_پای_ټکی_لیکه, بلاک_2, src0_contig, dst_contig,
                                                     ne01, div_ne01);

        dma_کتار_ټېل(dma_q,
            dma_make_معلومات(معلومات_dst, dst_vtcm_معلومات + (vtcm_idx * dst_vtcm_half_اندازه)),
            nb1, dst_لیکه_اندازه_aligned, dst_معلومات_لیکه_اندازه, 0);

        ثابت size_t src0_off = src0_contig ? (ir * nb01) :
            unary_لیکه_دننه_واټن(ir, ne01, ne02, div_ne01, div_ne02, div_ne012, nb01, nb02, nb03);
        dma_کتار_ټېل(dma_q,
            dma_make_معلومات(src0_vtcm_معلومات + (vtcm_idx * src0_vtcm_half_اندازه), معلومات_src + src0_off),
            src0_لیکه_اندازه_aligned, nb01, src0_معلومات_لیکه_اندازه, بلاک_اندازه_3);

        ir += بلاک_اندازه_3;
    }

    unary_compute_fn_t compute = (unary_compute_fn_t) uctx->compute;

    لكل (uint32_t ir = src0_پیل_لیکه; ir < src0_پای_ټکی_لیکه; ) {
        ثابت uint32_t بلاک_اندازه_3 = unary_بلاک_اندازه(ir, src0_پای_ټکی_لیکه, بلاک_2, src0_contig, dst_contig,
                                                     ne01, div_ne01);

        فراغ * dst_vtcm  = (فراغ *) (uintptr_t) dma_کتار_راوباسه(dma_q).src;
        فراغ * src0_vtcm = (فراغ *) (uintptr_t) dma_کتار_راوباسه(dma_q).dst;

        htp_trace_event_پیل(tr, HTP_TRACE_EVT_HVX_COMP, ir);
        compute(src0_vtcm, dst_vtcm, بلاک_اندازه_3, uctx);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ir);

        ثابت size_t dst_off = dst_contig ? (ir * nb1) :
            unary_لیکه_دننه_واټن(ir, ne1, ne2, div_ne01, div_ne02, div_ne012, nb1, nb2, nb3);
        dma_کتار_ټېل(dma_q,
            dma_make_معلومات(معلومات_dst + dst_off, dst_vtcm),
            nb1, dst_لیکه_اندازه_aligned, dst_معلومات_لیکه_اندازه, بلاک_اندازه_3);

        ثابت uint32_t بل_ir = ir + بلاک_اندازه_3;
        إذا (بل_ir < src0_پای_ټکی_لیکه) {
            ثابت uint32_t بل_بلاک_اندازه = unary_بلاک_اندازه(بل_ir, src0_پای_ټکی_لیکه, بلاک_2, src0_contig,
                                                              dst_contig, ne01, div_ne01);
            ثابت uint32_t pref_ir = بل_ir + بل_بلاک_اندازه;
            إذا (pref_ir < src0_پای_ټکی_لیکه) {
                ثابت uint32_t pref_بلاک_اندازه = unary_بلاک_اندازه(pref_ir, src0_پای_ټکی_لیکه, بلاک_2, src0_contig,
                                                                  dst_contig, ne01, div_ne01);
                ثابت size_t src0_pref_off = src0_contig ? (pref_ir * nb01) :
                    unary_لیکه_دننه_واټن(pref_ir, ne01, ne02, div_ne01, div_ne02, div_ne012, nb01, nb02, nb03);
                dma_کتار_ټېل(dma_q,
                    dma_make_معلومات(src0_vtcm, معلومات_src + src0_pref_off),
                    src0_لیکه_اندازه_aligned, nb01, src0_معلومات_لیکه_اندازه, pref_بلاک_اندازه);
            }
        }
        ir += بلاک_اندازه_3;
    }

    dma_کتار_flush(dma_q);
}

// 2. RMS_NORM_MUL row-block task with weight buffer.
ساكن فراغ unary_thread_rms_norm_mul_f32(بینښه صحيح nth, بینښه صحيح ith, فراغ * معلومات) {
    (فراغ) nth;
    ثابت بنية htp_unary_context * uctx = (ثابت بنية htp_unary_context *) معلومات;
    بنية htp_ops_context * octx = uctx->octx;
    ثابت بنية htp_tensor * src = octx->src[0];
    ثابت بنية htp_tensor * dst = octx->dst;
    بنية htp_thread_trace * tr = &octx->ctx->trace[ith];

    htp_unary_preamble;

    ثابت uint32_t src0_nrows_فی_thread = uctx->src0_nrows_فی_thread;
    ثابت size_t src0_معلومات_لیکه_اندازه = uctx->src0_معلومات_لیکه_اندازه;
    ثابت size_t dst_معلومات_لیکه_اندازه  = uctx->dst_معلومات_لیکه_اندازه;
    ثابت size_t src0_لیکه_اندازه_aligned = uctx->src0_لیکه_اندازه_aligned;
    ثابت size_t dst_لیکه_اندازه_aligned  = uctx->dst_لیکه_اندازه_aligned;

    ثابت uint32_t src0_nrows = uctx->src0_nrows;
    ثابت uint32_t src0_پیل_لیکه = uctx->لیکه_پیل + src0_nrows_فی_thread * ith;
    ثابت uint32_t src0_پای_ټکی_لیکه   = MIN(src0_پیل_لیکه + src0_nrows_فی_thread, uctx->لیکه_پیل + src0_nrows);

    إذا (src0_پیل_لیکه >= src0_پای_ټکی_لیکه) {
        أرجع;
    }

    ثابت dma_addr_t معلومات_src  = uctx->معلومات_src0;
    ثابت dma_addr_t معلومات_src1 = uctx->معلومات_src1;
    ثابت dma_addr_t معلومات_dst  = uctx->معلومات_dst;

    ثابت بنية htp_tensor * src1 = octx->src[1];
    ثابت uint32_t nb11 = src1->nb[1];
    ثابت uint32_t nb12 = src1->nb[2];
    ثابت uint32_t nb13 = src1->nb[3];
    ثابت uint32_t nb11_bc = (src1->ne[1] > 1) ? nb11 : 0;
    ثابت uint32_t nb12_bc = (src1->ne[2] > 1) ? nb12 : 0;
    ثابت uint32_t nb13_bc = (src1->ne[3] > 1) ? nb13 : 0;
    ثابت bool src1_contig = ((nb12 == (size_t)ne01 * nb11) && (nb13 == (size_t)ne02 * nb12));

    uint8_t * src0_vtcm_معلومات = uctx->vtcm_src0 + (ith * uctx->vtcm_src0_اندازه_فی_thread);
    uint8_t * src1_vtcm_معلومات = uctx->vtcm_src1 ? (uctx->vtcm_src1 + (ith * uctx->vtcm_src1_اندازه_فی_thread)) : عدم;
    uint8_t * dst_vtcm_معلومات  = uctx->vtcm_dst + (ith * uctx->vtcm_dst_اندازه_فی_thread);

    ثابت size_t src0_vtcm_half_اندازه = uctx->src0_vtcm_half_اندازه;
    ثابت size_t src1_vtcm_half_اندازه = uctx->src1_vtcm_half_اندازه;
    ثابت size_t dst_vtcm_half_اندازه  = uctx->dst_vtcm_half_اندازه;

    ثابت bool src0_contig = (nb02 == (size_t)ne01 * nb01) &&
                             (nb03 == (size_t)ne02 * nb02);
    ثابت bool dst_contig  = (nb2  == (size_t)ne1  * nb1)  &&
                             (nb3  == (size_t)ne2  * nb2);

    ثابت بنية fastdiv_ارزښتونه * div_ne01  = &uctx->kparams->div_ne01;
    ثابت بنية fastdiv_ارزښتونه * div_ne02  = &uctx->kparams->div_ne02;
    ثابت بنية fastdiv_ارزښتونه * div_ne012 = &uctx->kparams->div_ne012;

    ثابت bool src1_needs_لیکه_clip = !uctx->خپرونه_وزن && !src1_contig;
    ثابت bool بلاک_src0_contig = src0_contig && !src1_needs_لیکه_clip;
    ثابت bool بلاک_dst_contig  = dst_contig  && !src1_needs_لیکه_clip;

    ثابت uint32_t src0_max_بلاک = بلاک_src0_contig ? uctx->بلاک : MIN((uint32_t)uctx->بلاک, ne01);
    ثابت uint32_t dst_max_بلاک  = بلاک_dst_contig  ? uctx->بلاک : MIN((uint32_t)uctx->بلاک, ne1);
    ثابت uint32_t بلاک_2 = MIN(src0_max_بلاک, dst_max_بلاک);
    إذا (بلاک_2 == 0) {
        FARF(ERROR, "unary-rms-norm-mul : current VTCM reservation %zu is too small, needed at least %zu\n",
             uctx->vtcm_src0_اندازه_فی_thread, src0_لیکه_اندازه_aligned);
        أرجع;
    }

    dma_کتار * dma_q = octx->ctx->dma[ith];

    إذا (uctx->خپرونه_وزن) {
        dma_کتار_ټېل(dma_q, dma_make_معلومات(src1_vtcm_معلومات, معلومات_src1),
                       uctx->src1_لیکه_اندازه_aligned, 0, uctx->src1_معلومات_لیکه_اندازه, 1);
        dma_کتار_flush(dma_q);
    }

    لكل (uint32_t ir = src0_پیل_لیکه, vtcm_idx = 0; ir < src0_پای_ټکی_لیکه && vtcm_idx < 2; vtcm_idx++) {
        ثابت uint32_t بلاک_اندازه_3 = unary_بلاک_اندازه(ir, src0_پای_ټکی_لیکه, بلاک_2, بلاک_src0_contig, بلاک_dst_contig,
                                                     ne01, div_ne01);

        dma_کتار_ټېل(dma_q,
            dma_make_معلومات(معلومات_dst, dst_vtcm_معلومات + (vtcm_idx * dst_vtcm_half_اندازه)),
            nb1, dst_لیکه_اندازه_aligned, dst_معلومات_لیکه_اندازه, 0);

        ثابت size_t src0_off = src0_contig ? (ir * nb01) :
            unary_لیکه_دننه_واټن(ir, ne01, ne02, div_ne01, div_ne02, div_ne012, nb01, nb02, nb03);
        dma_کتار_ټېل(dma_q,
            dma_make_معلومات(src0_vtcm_معلومات + (vtcm_idx * src0_vtcm_half_اندازه), معلومات_src + src0_off),
            src0_لیکه_اندازه_aligned, nb01, src0_معلومات_لیکه_اندازه, بلاک_اندازه_3);

        إذا (!uctx->خپرونه_وزن) {
            ثابت size_t src1_off = src1_contig ? (ir * nb11) :
                unary_لیکه_دننه_واټن(ir, ne01, ne02, div_ne01, div_ne02, div_ne012, nb11_bc, nb12_bc, nb13_bc);
            dma_کتار_ټېل(dma_q,
                dma_make_معلومات(src1_vtcm_معلومات + (vtcm_idx * src1_vtcm_half_اندازه), معلومات_src1 + src1_off),
                uctx->src1_لیکه_اندازه_aligned, nb11, uctx->src1_معلومات_لیکه_اندازه, بلاک_اندازه_3);
        }

        ir += بلاک_اندازه_3;
    }

    unary_rms_norm_mul_compute_fn_t compute = (unary_rms_norm_mul_compute_fn_t) uctx->compute;

    لكل (uint32_t ir = src0_پیل_لیکه; ir < src0_پای_ټکی_لیکه; ) {
        ثابت uint32_t بلاک_اندازه_3 = unary_بلاک_اندازه(ir, src0_پای_ټکی_لیکه, بلاک_2, بلاک_src0_contig, بلاک_dst_contig,
                                                     ne01, div_ne01);

        فراغ * dst_vtcm  = (فراغ *) (uintptr_t) dma_کتار_راوباسه(dma_q).src;
        فراغ * src0_vtcm = (فراغ *) (uintptr_t) dma_کتار_راوباسه(dma_q).dst;
        فراغ * src1_vtcm = عدم;
        إذا (!uctx->خپرونه_وزن) {
            src1_vtcm = (فراغ *) (uintptr_t) dma_کتار_راوباسه(dma_q).dst;
        }

        htp_trace_event_پیل(tr, HTP_TRACE_EVT_HVX_COMP, ir);
        ثابت فراغ * w = uctx->خپرونه_وزن ? (ثابت فراغ *) src1_vtcm_معلومات : src1_vtcm;
        compute(src0_vtcm, w, dst_vtcm, بلاک_اندازه_3, uctx);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ir);

        ثابت size_t dst_off = dst_contig ? (ir * nb1) :
            unary_لیکه_دننه_واټن(ir, ne1, ne2, div_ne01, div_ne02, div_ne012, nb1, nb2, nb3);
        dma_کتار_ټېل(dma_q,
            dma_make_معلومات(معلومات_dst + dst_off, dst_vtcm),
            nb1, dst_لیکه_اندازه_aligned, dst_معلومات_لیکه_اندازه, بلاک_اندازه_3);

        ثابت uint32_t بل_ir = ir + بلاک_اندازه_3;
        إذا (بل_ir < src0_پای_ټکی_لیکه) {
            ثابت uint32_t بل_بلاک_اندازه = unary_بلاک_اندازه(بل_ir, src0_پای_ټکی_لیکه, بلاک_2, بلاک_src0_contig,
                                                              بلاک_dst_contig, ne01, div_ne01);
            ثابت uint32_t pref_ir = بل_ir + بل_بلاک_اندازه;
            إذا (pref_ir < src0_پای_ټکی_لیکه) {
                ثابت uint32_t pref_بلاک_اندازه = unary_بلاک_اندازه(pref_ir, src0_پای_ټکی_لیکه, بلاک_2, بلاک_src0_contig,
                                                                  بلاک_dst_contig, ne01, div_ne01);
                ثابت size_t src0_pref_off = src0_contig ? (pref_ir * nb01) :
                    unary_لیکه_دننه_واټن(pref_ir, ne01, ne02, div_ne01, div_ne02, div_ne012, nb01, nb02, nb03);
                dma_کتار_ټېل(dma_q,
                    dma_make_معلومات(src0_vtcm, معلومات_src + src0_pref_off),
                    src0_لیکه_اندازه_aligned, nb01, src0_معلومات_لیکه_اندازه, pref_بلاک_اندازه);

                إذا (!uctx->خپرونه_وزن) {
                    ثابت size_t src1_pref_off = src1_contig ? (pref_ir * nb11) :
                        unary_لیکه_دننه_واټن(pref_ir, ne01, ne02, div_ne01, div_ne02, div_ne012, nb11_bc, nb12_bc,
                                          nb13_bc);
                    dma_کتار_ټېل(dma_q,
                        dma_make_معلومات(src1_vtcm, معلومات_src1 + src1_pref_off),
                        uctx->src1_لیکه_اندازه_aligned, nb11, uctx->src1_معلومات_لیکه_اندازه, pref_بلاک_اندازه);
                }
            }
        }
        ir += بلاک_اندازه_3;
    }

    dma_کتار_flush(dma_q);
}

// 3. TRI row-block task with row index ir.
ساكن فراغ unary_thread_tri_f32(بینښه صحيح nth, بینښه صحيح ith, فراغ * معلومات) {
    (فراغ) nth;
    ثابت بنية htp_unary_context * uctx = (ثابت بنية htp_unary_context *) معلومات;
    بنية htp_ops_context * octx = uctx->octx;
    ثابت بنية htp_tensor * src = octx->src[0];
    ثابت بنية htp_tensor * dst = octx->dst;
    بنية htp_thread_trace * tr = &octx->ctx->trace[ith];

    htp_unary_preamble;

    ثابت uint32_t src0_nrows_فی_thread = uctx->src0_nrows_فی_thread;
    ثابت size_t src0_معلومات_لیکه_اندازه = uctx->src0_معلومات_لیکه_اندازه;
    ثابت size_t dst_معلومات_لیکه_اندازه  = uctx->dst_معلومات_لیکه_اندازه;
    ثابت size_t src0_لیکه_اندازه_aligned = uctx->src0_لیکه_اندازه_aligned;
    ثابت size_t dst_لیکه_اندازه_aligned  = uctx->dst_لیکه_اندازه_aligned;

    ثابت uint32_t src0_nrows = uctx->src0_nrows;
    ثابت uint32_t src0_پیل_لیکه = uctx->لیکه_پیل + src0_nrows_فی_thread * ith;
    ثابت uint32_t src0_پای_ټکی_لیکه   = MIN(src0_پیل_لیکه + src0_nrows_فی_thread, uctx->لیکه_پیل + src0_nrows);

    إذا (src0_پیل_لیکه >= src0_پای_ټکی_لیکه) {
        أرجع;
    }

    ثابت dma_addr_t معلومات_src = uctx->معلومات_src0;
    ثابت dma_addr_t معلومات_dst = uctx->معلومات_dst;

    uint8_t * src0_vtcm_معلومات = uctx->vtcm_src0 + (ith * uctx->vtcm_src0_اندازه_فی_thread);
    uint8_t * dst_vtcm_معلومات  = uctx->vtcm_dst + (ith * uctx->vtcm_dst_اندازه_فی_thread);

    ثابت size_t src0_vtcm_half_اندازه = uctx->src0_vtcm_half_اندازه;
    ثابت size_t dst_vtcm_half_اندازه  = uctx->dst_vtcm_half_اندازه;

    ثابت bool src0_contig = (nb02 == (size_t)ne01 * nb01) &&
                             (nb03 == (size_t)ne02 * nb02);
    ثابت bool dst_contig  = (nb2  == (size_t)ne1  * nb1)  &&
                             (nb3  == (size_t)ne2  * nb2);

    ثابت بنية fastdiv_ارزښتونه * div_ne01  = &uctx->kparams->div_ne01;
    ثابت بنية fastdiv_ارزښتونه * div_ne02  = &uctx->kparams->div_ne02;
    ثابت بنية fastdiv_ارزښتونه * div_ne012 = &uctx->kparams->div_ne012;

    ثابت uint32_t src0_max_بلاک = src0_contig ? uctx->بلاک : MIN((uint32_t)uctx->بلاک, ne01);
    ثابت uint32_t dst_max_بلاک  = dst_contig  ? uctx->بلاک : MIN((uint32_t)uctx->بلاک, ne1);
    ثابت uint32_t بلاک_2 = MIN(src0_max_بلاک, dst_max_بلاک);
    إذا (بلاک_2 == 0) {
        FARF(ERROR, "unary-tri : current VTCM reservation %zu is too small, needed at least %zu\n",
             uctx->vtcm_src0_اندازه_فی_thread, src0_لیکه_اندازه_aligned);
        أرجع;
    }

    dma_کتار * dma_q = octx->ctx->dma[ith];

    لكل (uint32_t ir = src0_پیل_لیکه, vtcm_idx = 0; ir < src0_پای_ټکی_لیکه && vtcm_idx < 2; vtcm_idx++) {
        ثابت uint32_t بلاک_اندازه_3 = unary_بلاک_اندازه(ir, src0_پای_ټکی_لیکه, بلاک_2, src0_contig, dst_contig,
                                                     ne01, div_ne01);

        dma_کتار_ټېل(dma_q,
            dma_make_معلومات(معلومات_dst, dst_vtcm_معلومات + (vtcm_idx * dst_vtcm_half_اندازه)),
            nb1, dst_لیکه_اندازه_aligned, dst_معلومات_لیکه_اندازه, 0);

        ثابت size_t src0_off = src0_contig ? (ir * nb01) :
            unary_لیکه_دننه_واټن(ir, ne01, ne02, div_ne01, div_ne02, div_ne012, nb01, nb02, nb03);
        dma_کتار_ټېل(dma_q,
            dma_make_معلومات(src0_vtcm_معلومات + (vtcm_idx * src0_vtcm_half_اندازه), معلومات_src + src0_off),
            src0_لیکه_اندازه_aligned, nb01, src0_معلومات_لیکه_اندازه, بلاک_اندازه_3);

        ir += بلاک_اندازه_3;
    }

    unary_tri_compute_fn_t compute = (unary_tri_compute_fn_t) uctx->compute;

    لكل (uint32_t ir = src0_پیل_لیکه; ir < src0_پای_ټکی_لیکه; ) {
        ثابت uint32_t بلاک_اندازه_3 = unary_بلاک_اندازه(ir, src0_پای_ټکی_لیکه, بلاک_2, src0_contig, dst_contig,
                                                     ne01, div_ne01);

        فراغ * dst_vtcm  = (فراغ *) (uintptr_t) dma_کتار_راوباسه(dma_q).src;
        فراغ * src0_vtcm = (فراغ *) (uintptr_t) dma_کتار_راوباسه(dma_q).dst;

        htp_trace_event_پیل(tr, HTP_TRACE_EVT_HVX_COMP, ir);
        compute(src0_vtcm, dst_vtcm, بلاک_اندازه_3, ir, uctx);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ir);

        ثابت size_t dst_off = dst_contig ? (ir * nb1) :
            unary_لیکه_دننه_واټن(ir, ne1, ne2, div_ne01, div_ne02, div_ne012, nb1, nb2, nb3);
        dma_کتار_ټېل(dma_q,
            dma_make_معلومات(معلومات_dst + dst_off, dst_vtcm),
            nb1, dst_لیکه_اندازه_aligned, dst_معلومات_لیکه_اندازه, بلاک_اندازه_3);

        ثابت uint32_t بل_ir = ir + بلاک_اندازه_3;
        إذا (بل_ir < src0_پای_ټکی_لیکه) {
            ثابت uint32_t بل_بلاک_اندازه = unary_بلاک_اندازه(بل_ir, src0_پای_ټکی_لیکه, بلاک_2, src0_contig,
                                                              dst_contig, ne01, div_ne01);
            ثابت uint32_t pref_ir = بل_ir + بل_بلاک_اندازه;
            إذا (pref_ir < src0_پای_ټکی_لیکه) {
                ثابت uint32_t pref_بلاک_اندازه = unary_بلاک_اندازه(pref_ir, src0_پای_ټکی_لیکه, بلاک_2, src0_contig,
                                                                  dst_contig, ne01, div_ne01);
                ثابت size_t src0_pref_off = src0_contig ? (pref_ir * nb01) :
                    unary_لیکه_دننه_واټن(pref_ir, ne01, ne02, div_ne01, div_ne02, div_ne012, nb01, nb02, nb03);
                dma_کتار_ټېل(dma_q,
                    dma_make_معلومات(src0_vtcm, معلومات_src + src0_pref_off),
                    src0_لیکه_اندازه_aligned, nb01, src0_معلومات_لیکه_اندازه, pref_بلاک_اندازه);
            }
        }
        ir += بلاک_اندازه_3;
    }

    dma_کتار_flush(dma_q);
}

// 4. Pointwise tiled unary task.
ساكن فراغ unary_thread_tiled(بینښه صحيح nth, بینښه صحيح ith, فراغ * معلومات) {
    (فراغ) nth;
    ثابت بنية htp_unary_context * uctx = (ثابت بنية htp_unary_context *) معلومات;
    بنية htp_ops_context * octx = uctx->octx;
    ثابت بنية htp_tensor * src = octx->src[0];
    ثابت بنية htp_tensor * dst = octx->dst;
    بنية htp_thread_trace * tr = &octx->ctx->trace[ith];

    htp_unary_preamble;

    ثابت uint32_t src0_nrows_فی_thread = uctx->src0_nrows_فی_thread;
    ثابت uint32_t ستون_tile              = uctx->ستون_tile;

    ثابت uint32_t src0_nrows     = uctx->src0_nrows;
    ثابت uint32_t src0_پیل_لیکه = uctx->لیکه_پیل + src0_nrows_فی_thread * ith;
    ثابت uint32_t src0_پای_ټکی_لیکه   = MIN(src0_پیل_لیکه + src0_nrows_فی_thread, uctx->لیکه_پیل + src0_nrows);

    إذا (src0_پیل_لیکه >= src0_پای_ټکی_لیکه) {
        أرجع;
    }

    ثابت dma_addr_t معلومات_src = uctx->معلومات_src0;
    ثابت dma_addr_t معلومات_dst = uctx->معلومات_dst;

    uint8_t * src0_vtcm_معلومات = uctx->vtcm_src0 + (ith * uctx->vtcm_src0_اندازه_فی_thread);
    uint8_t * dst_vtcm_معلومات  = uctx->vtcm_dst + (ith * uctx->vtcm_dst_اندازه_فی_thread);

    ثابت size_t src0_half = uctx->src0_vtcm_half_اندازه;
    ثابت size_t dst_half  = uctx->dst_vtcm_half_اندازه;

    dma_کتار * dma_q = octx->ctx->dma[ith];

    ثابت بنية fastdiv_ارزښتونه * div_ne01  = &uctx->kparams->div_ne01;
    ثابت بنية fastdiv_ارزښتونه * div_ne02  = &uctx->kparams->div_ne02;
    ثابت بنية fastdiv_ارزښتونه * div_ne012 = &uctx->kparams->div_ne012;
    ثابت بنية fastdiv_ارزښتونه * div_tpr   = &uctx->kparams->div_tpr;

    ثابت uint32_t tiles_فی_لیکه = (ne0 + ستون_tile - 1) / ستون_tile;

    ثابت bool src0_contig = (nb02 == (size_t)ne01 * nb01) &&
                             (nb03 == (size_t)ne02 * nb02);
    ثابت bool dst_contig  = (nb2  == (size_t)ne1  * nb1)  &&
                             (nb3  == (size_t)ne2  * nb2);

    ثابت uint32_t ټول_tiles = (src0_پای_ټکی_لیکه - src0_پیل_لیکه) * tiles_فی_لیکه;

    لكل (uint32_t t = 0, vtcm_idx = 0; t < ټول_tiles && vtcm_idx < 2; t++, vtcm_idx++) {
        ثابت uint32_t لیکه  = src0_پیل_لیکه + t / tiles_فی_لیکه;
        ثابت uint32_t ستون  = (t % tiles_فی_لیکه) * ستون_tile;
        ثابت uint32_t tw   = MIN(ستون_tile, ne0 - ستون);
        ثابت size_t   tb   = (size_t) tw * حجم(عائم);
        ثابت size_t   soff = (src0_contig ? (لیکه * nb01) :
                               unary_لیکه_دننه_واټن(لیکه, ne01, ne02, div_ne01, div_ne02, div_ne012, nb01, nb02, nb03)) +
                               (size_t) ستون * حجم(عائم);

        dma_کتار_ټېل(dma_q, dma_make_معلومات(معلومات_dst, dst_vtcm_معلومات + (vtcm_idx * dst_half)), 0, 0, 0, 0);
        dma_کتار_ټېل(dma_q, dma_make_معلومات(src0_vtcm_معلومات + (vtcm_idx * src0_half), معلومات_src + soff), tb, tb, tb, 1);
    }

    unary_tile_compute_fn_t compute = (unary_tile_compute_fn_t) uctx->compute;

    uint32_t لیکه = src0_پیل_لیکه;
    uint32_t ستون = 0;
    uint32_t tile_دننه_لیکه = 0;

    uint32_t prow = src0_پیل_لیکه + fastdiv(2, div_tpr);
    uint32_t pcol = fastmodulo(2, tiles_فی_لیکه, div_tpr) * ستون_tile;
    uint32_t ptile_دننه_لیکه = fastmodulo(2, tiles_فی_لیکه, div_tpr);

    لكل (uint32_t t = 0; t < ټول_tiles; t++) {
        فراغ * dst_vtcm = (فراغ *) (uintptr_t) dma_کتار_راوباسه(dma_q).src;
        فراغ * src_vtcm = (فراغ *) (uintptr_t) dma_کتار_راوباسه(dma_q).dst;

        ثابت uint32_t tw = MIN(ستون_tile, ne0 - ستون);

        htp_trace_event_پیل(tr, HTP_TRACE_EVT_HVX_COMP, t);
        compute(dst_vtcm, src_vtcm, tw, uctx);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, t);

        ثابت size_t doff = (dst_contig ? (لیکه * nb1) :
                             unary_لیکه_دننه_واټن(لیکه, ne1, ne2, div_ne01, div_ne02, div_ne012, nb1, nb2, nb3)) +
                             (size_t) ستون * حجم(عائم);
        ثابت size_t tb   = (size_t) tw * حجم(عائم);
        dma_کتار_ټېل(dma_q, dma_make_معلومات(معلومات_dst + doff, dst_vtcm), tb, tb, tb, 1);

        ثابت uint32_t pt = t + 2;
        إذا (pt < ټول_tiles) {
            ثابت uint32_t ptw  = MIN(ستون_tile, ne0 - pcol);
            ثابت size_t   ptb  = (size_t) ptw * حجم(عائم);
            ثابت size_t   psoff = (src0_contig ? (prow * nb01) :
                                    unary_لیکه_دننه_واټن(prow, ne01, ne02, div_ne01, div_ne02, div_ne012, nb01, nb02,
                                                     nb03)) +
                                   (size_t) pcol * حجم(عائم);
            dma_کتار_ټېل(dma_q, dma_make_معلومات(src_vtcm, معلومات_src + psoff), ptb, ptb, ptb, 1);
        }

        tile_دننه_لیکه++;
        ستون += ستون_tile;
        إذا (tile_دننه_لیکه == tiles_فی_لیکه) {
            tile_دننه_لیکه = 0;
            ستون = 0;
            لیکه++;
        }

        ptile_دننه_لیکه++;
        pcol += ستون_tile;
        إذا (ptile_دننه_لیکه == tiles_فی_لیکه) {
            ptile_دننه_لیکه = 0;
            pcol = 0;
            prow++;
        }
    }

    dma_کتار_flush(dma_q);
}

// 5. TRI tiled task.
ساكن فراغ unary_thread_tiled_tri_f32(بینښه صحيح nth, بینښه صحيح ith, فراغ * معلومات) {
    (فراغ) nth;
    ثابت بنية htp_unary_context * uctx = (ثابت بنية htp_unary_context *) معلومات;
    بنية htp_ops_context * octx = uctx->octx;
    ثابت بنية htp_tensor * src = octx->src[0];
    ثابت بنية htp_tensor * dst = octx->dst;
    بنية htp_thread_trace * tr = &octx->ctx->trace[ith];

    htp_unary_preamble;

    ثابت uint32_t src0_nrows_فی_thread = uctx->src0_nrows_فی_thread;
    ثابت int32_t * op_params            = octx->op_params;
    ثابت uint32_t ستون_tile              = uctx->ستون_tile;

    ثابت uint32_t src0_nrows     = uctx->src0_nrows;
    ثابت uint32_t src0_پیل_لیکه = uctx->لیکه_پیل + src0_nrows_فی_thread * ith;
    ثابت uint32_t src0_پای_ټکی_لیکه   = MIN(src0_پیل_لیکه + src0_nrows_فی_thread, uctx->لیکه_پیل + src0_nrows);

    إذا (src0_پیل_لیکه >= src0_پای_ټکی_لیکه) {
        أرجع;
    }

    ثابت dma_addr_t معلومات_src = uctx->معلومات_src0;
    ثابت dma_addr_t معلومات_dst = uctx->معلومات_dst;

    uint8_t * src0_vtcm_معلومات = uctx->vtcm_src0 + (ith * uctx->vtcm_src0_اندازه_فی_thread);
    uint8_t * dst_vtcm_معلومات  = uctx->vtcm_dst + (ith * uctx->vtcm_dst_اندازه_فی_thread);

    ثابت size_t src0_half = uctx->src0_vtcm_half_اندازه;
    ثابت size_t dst_half  = uctx->dst_vtcm_half_اندازه;

    dma_کتار * dma_q = octx->ctx->dma[ith];

    ثابت بنية fastdiv_ارزښتونه * div_ne01  = &uctx->kparams->div_ne01;
    ثابت بنية fastdiv_ارزښتونه * div_ne02  = &uctx->kparams->div_ne02;
    ثابت بنية fastdiv_ارزښتونه * div_ne012 = &uctx->kparams->div_ne012;
    ثابت بنية fastdiv_ارزښتونه * div_tpr   = &uctx->kparams->div_tpr;

    ثابت uint32_t tiles_فی_لیکه = (ne0 + ستون_tile - 1) / ستون_tile;
    ثابت int32_t  tri_ttype     = op_params[0];

    ثابت bool src0_contig = (nb02 == (size_t)ne01 * nb01) &&
                             (nb03 == (size_t)ne02 * nb02);
    ثابت bool dst_contig  = (nb2  == (size_t)ne1  * nb1)  &&
                             (nb3  == (size_t)ne2  * nb2);

    ثابت uint32_t ټول_tiles = (src0_پای_ټکی_لیکه - src0_پیل_لیکه) * tiles_فی_لیکه;

    لكل (uint32_t t = 0, vtcm_idx = 0; t < ټول_tiles && vtcm_idx < 2; t++, vtcm_idx++) {
        ثابت uint32_t لیکه  = src0_پیل_لیکه + t / tiles_فی_لیکه;
        ثابت uint32_t ستون  = (t % tiles_فی_لیکه) * ستون_tile;
        ثابت uint32_t tw   = MIN(ستون_tile, ne0 - ستون);
        ثابت size_t   tb   = (size_t) tw * حجم(عائم);
        ثابت size_t   soff = (src0_contig ? (لیکه * nb01) :
                               unary_لیکه_دننه_واټن(لیکه, ne01, ne02, div_ne01, div_ne02, div_ne012, nb01, nb02, nb03)) +
                               (size_t) ستون * حجم(عائم);

        dma_کتار_ټېل(dma_q, dma_make_معلومات(معلومات_dst, dst_vtcm_معلومات + (vtcm_idx * dst_half)), 0, 0, 0, 0);
        dma_کتار_ټېل(dma_q, dma_make_معلومات(src0_vtcm_معلومات + (vtcm_idx * src0_half), معلومات_src + soff), tb, tb, tb, 1);
    }

    unary_tiled_tri_compute_fn_t compute = (unary_tiled_tri_compute_fn_t) uctx->compute;

    uint32_t لیکه = src0_پیل_لیکه;
    uint32_t ستون = 0;
    uint32_t tile_دننه_لیکه = 0;
    uint32_t i01 = fastmodulo(لیکه, ne01, div_ne01);

    uint32_t prow = src0_پیل_لیکه + fastdiv(2, div_tpr);
    uint32_t pcol = fastmodulo(2, tiles_فی_لیکه, div_tpr) * ستون_tile;
    uint32_t ptile_دننه_لیکه = fastmodulo(2, tiles_فی_لیکه, div_tpr);

    لكل (uint32_t t = 0; t < ټول_tiles; t++) {
        فراغ * dst_vtcm = (فراغ *) (uintptr_t) dma_کتار_راوباسه(dma_q).src;
        فراغ * src_vtcm = (فراغ *) (uintptr_t) dma_کتار_راوباسه(dma_q).dst;

        ثابت uint32_t tw = MIN(ستون_tile, ne0 - ستون);

        htp_trace_event_پیل(tr, HTP_TRACE_EVT_HVX_COMP, t);
        compute(src_vtcm, dst_vtcm, tw, ستون, i01, ne0, tri_ttype);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, t);

        ثابت size_t doff = (dst_contig ? (لیکه * nb1) :
                             unary_لیکه_دننه_واټن(لیکه, ne1, ne2, div_ne01, div_ne02, div_ne012, nb1, nb2, nb3)) +
                             (size_t) ستون * حجم(عائم);
        ثابت size_t tb   = (size_t) tw * حجم(عائم);
        dma_کتار_ټېل(dma_q, dma_make_معلومات(معلومات_dst + doff, dst_vtcm), tb, tb, tb, 1);

        ثابت uint32_t pt = t + 2;
        إذا (pt < ټول_tiles) {
            ثابت uint32_t ptw  = MIN(ستون_tile, ne0 - pcol);
            ثابت size_t   ptb  = (size_t) ptw * حجم(عائم);
            ثابت size_t   psoff = (src0_contig ? (prow * nb01) :
                                    unary_لیکه_دننه_واټن(prow, ne01, ne02, div_ne01, div_ne02, div_ne012, nb01, nb02,
                                                     nb03)) +
                                   (size_t) pcol * حجم(عائم);
            dma_کتار_ټېل(dma_q, dma_make_معلومات(src_vtcm, معلومات_src + psoff), ptb, ptb, ptb, 1);
        }

        tile_دننه_لیکه++;
        ستون += ستون_tile;
        إذا (tile_دننه_لیکه == tiles_فی_لیکه) {
            tile_دننه_لیکه = 0;
            ستون = 0;
            لیکه++;
            i01++;
            إذا (i01 == ne01) {
                i01 = 0;
            }
        }

        ptile_دننه_لیکه++;
        pcol += ستون_tile;
        إذا (ptile_دننه_لیکه == tiles_فی_لیکه) {
            ptile_دننه_لیکه = 0;
            pcol = 0;
            prow++;
        }
    }

    dma_کتار_flush(dma_q);
}

ساكن صحيح execute_op_unary(بنية htp_ops_context * octx) {
    صحيح err = HTP_حالت_OK;

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
        حالة HTP_OP_UNARY_GELU_چټک: op_type = "gelu-quick-f32";                             اكسر;
        حالة HTP_OP_UNARY_GELU_ERF:  op_type = "gelu-erf-f32";                               اكسر;
        حالة HTP_OP_UNARY_SOFTPLUS:  op_type = "softplus-f32";                               اكسر;
        حالة HTP_OP_UNARY_TANH:      op_type = "tanh-f32";                                   اكسر;
        حالة HTP_OP_UNARY_ABS:       op_type = is_f16 ? "abs-f16"      : "abs-f32";          اكسر;
        حالة HTP_OP_UNARY_LOG:       op_type = is_f16 ? "log-f16"      : "log-f32";          اكسر;
        حالة HTP_OP_UNARY_RELU:      op_type = "relu-f32";                                   اكسر;
        حالة HTP_OP_UNARY_ګام:      op_type = is_f16 ? "step-f16"     : "step-f32";         اكسر;
        حالة HTP_OP_L2_NORM:         op_type = is_f16 ? "l2norm-f16"   : "l2norm-f32";       اكسر;
        حالة HTP_OP_TRI:             op_type = "tri-f32";                                    اكسر;
        حالة HTP_OP_HADAMARD:        op_type = is_f16 ? "hadamard-f16" : "hadamard-f32";     اكسر;
        افتراضي:
            FARF(ERROR, "Unsupported unary Op %u\n", octx->op);
            أرجع HTP_حالت_NO_SUPPORT;
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
            حالة HTP_OP_UNARY_ګام:
            حالة HTP_OP_HADAMARD:
                اكسر;
            افتراضي:
                FARF(ERROR, "unary-%s: not supported for F16\n", op_type);
                أرجع HTP_حالت_NO_SUPPORT;
        }
    }

    ثابت بنية htp_unary_kernel_params * kparams = (ثابت بنية htp_unary_kernel_params *) octx->kernel_params;

    إذا (!htp_ops_context_سېټ_n_تارونه(octx, kparams->n_تارونه)) {
        أرجع HTP_حالت_INVAL_PARAMS;
    }

    ثابت uint32_t src0_nrows = src0->ne[1] * src0->ne[2] * src0->ne[3];
    ثابت size_t src0_elem_اندازه = is_f16 ? حجم(_Float16) : حجم(عائم);
    ثابت size_t dst_elem_اندازه  = (dst->type == HTP_TYPE_F16) ? حجم(_Float16) : حجم(عائم);
    ثابت size_t src0_معلومات_لیکه_اندازه = src0->ne[0] * src0_elem_اندازه;
    ثابت size_t dst_معلومات_لیکه_اندازه  = dst->ne[0]  * dst_elem_اندازه;

    uint32_t لیکه_پیل = 0;
    uint32_t nrows     = src0_nrows;

    إذا (octx->ctx->mdev.شمېر > 1) {
        uint32_t لیکې_فی_chunk = 0;
        htp_tensor_mdev_لیکې_فی_chunk(dst, (uint32_t) dst_elem_اندازه, (uint32_t) dst_معلومات_لیکه_اندازه, &لیکې_فی_chunk);
        ثابت بنية htp_tensor_mdev_range range = htp_tensor_mdev_برخه_کول(src0_nrows, لیکې_فی_chunk, octx->ctx->mdev.idx, octx->ctx->mdev.شمېر, &octx->ctx->mdev.شمېر_div);
        لیکه_پیل = range.پیل;
        nrows     = range.شمېر;
    }

    إذا (nrows == 0) {
        أرجع HTP_حالت_OK;
    }

    ثابت uint32_t n_تارونه = octx->n_تارونه;

    ثابت size_t src0_لیکه_اندازه_aligned = kparams->src0_لیکه_اندازه_aligned;
    ثابت size_t dst_لیکه_اندازه_aligned  = kparams->dst_لیکه_اندازه_aligned;

    // Always 0 for F16 - htp_unary_vtcm_layout_build() keeps F16 on the row-block path,
    // since only F32 has unary_task_f32_tiled_* kernels.
    ثابت uint32_t ستون_tile = kparams->ستون_tile;

    size_t src1_معلومات_لیکه_اندازه = 0;
    size_t src1_لیکه_اندازه_aligned = kparams->src1_لیکه_اندازه_aligned;
    bool خپرونه_وزن = kparams->خپرونه_وزن;
    ثابت بنية htp_tensor * src1 = عدم;

    // RMS_NORM_MUL fusion is F32-only (its weight tensor is always F32; see
    // try_fuse_node()'s type guard), so this never triggers when is_f16 is true.
    إذا (octx->op == HTP_OP_RMS_NORM_MUL) {
        src1 = octx->src[1];
        src1_معلومات_لیکه_اندازه = src1->ne[0] * حجم(عائم);
    }

    إذا (octx->ctx->vtcm_اندازه < (size_t)kparams->vtcm_اندازه) {
        FARF(ERROR, "unary-%s : current VTCM reservation %zu is too small, needed %zu\n", op_type, octx->ctx->vtcm_اندازه, (size_t)kparams->vtcm_اندازه);
        أرجع HTP_حالت_VTCM_TOO_SMALL;
    }

    octx->src0_spad.src = عدم;
    octx->src1_spad.src = عدم;
    octx->dst_spad.src  = عدم;

    FARF(لوړ_2, "%s: (%ux%ux%ux%u) -> (%ux%ux%ux%u) : src0-vtcm-size %u src1-vtcm-size %u dst-vtcm-size %u\n", op_type,
         src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3], dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3],
         kparams->vtcm_src0_اندازه, kparams->vtcm_src1_اندازه, kparams->vtcm_dst_اندازه);

    uint8_t * ثابت بنسټ = (uint8_t *) octx->ctx->vtcm_بنسټ;
    بنية htp_unary_context uctx = {
        .octx                  = octx,
        .kparams               = kparams,
        .src0_nrows_فی_thread = fastdiv(nrows + n_تارونه - 1, &octx->n_تارونه_div),
        .src0_nrows            = nrows,
        .لیکه_پیل             = لیکه_پیل,

        .معلومات_src0             = src0->معلومات,
        .معلومات_src1             = (octx->op == HTP_OP_RMS_NORM_MUL) ? src1->معلومات : 0,
        .معلومات_dst              = dst->معلومات,

        .src0_معلومات_لیکه_اندازه    = src0_معلومات_لیکه_اندازه,
        .src1_معلومات_لیکه_اندازه    = src1_معلومات_لیکه_اندازه,
        .dst_معلومات_لیکه_اندازه     = dst_معلومات_لیکه_اندازه,

        .src0_لیکه_اندازه_aligned = src0_لیکه_اندازه_aligned,
        .src1_لیکه_اندازه_aligned = src1_لیکه_اندازه_aligned,
        .dst_لیکه_اندازه_aligned  = dst_لیکه_اندازه_aligned,

        .src0_vtcm_half_اندازه   = kparams->vtcm_src0_اندازه_فی_thread / 2,
        .src1_vtcm_half_اندازه   = (octx->op == HTP_OP_RMS_NORM_MUL) ? (kparams->vtcm_src1_اندازه_فی_thread / (خپرونه_وزن ? 1 : 2)) : 0,
        .dst_vtcm_half_اندازه    = kparams->vtcm_dst_اندازه_فی_thread / 2,

        .بلاک                 = kparams->بلاک,
        .nc                    = src0->ne[0],
        .ستون_tile              = ستون_tile,
        .خپرونه_وزن      = خپرونه_وزن,

        .vtcm_src0             = VTCM_LAYOUT_PTR(uint8_t, بنسټ, 0),
        .vtcm_src1             = VTCM_LAYOUT_PTR_OPTIONAL(uint8_t, بنسټ, kparams->vtcm_src0_اندازه, kparams->vtcm_src1_اندازه > 0),
        .vtcm_dst              = VTCM_LAYOUT_PTR(uint8_t, بنسټ, kparams->vtcm_src0_اندازه + kparams->vtcm_src1_اندازه),

        .vtcm_src0_اندازه_فی_thread = kparams->vtcm_src0_اندازه_فی_thread,
        .vtcm_src1_اندازه_فی_thread = kparams->vtcm_src1_اندازه_فی_thread,
        .vtcm_dst_اندازه_فی_thread  = kparams->vtcm_dst_اندازه_فی_thread,
    };

    FARF(لوړ_2, "%s: %s mode (col_tile %u)\n", op_type, ستون_tile ? "tiled" : "row-block", ستون_tile);

    کارګر_callback_t task_func = عدم;
    فراغ * compute_func = عدم;

    إذا (ستون_tile) {
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
            حالة HTP_OP_UNARY_GELU_چټک: compute_func = (فراغ *) tile_gelu_چټک_f32;     اكسر;
            حالة HTP_OP_UNARY_GELU_ERF:  compute_func = (فراغ *) tile_gelu_erf_f32;       اكسر;
            حالة HTP_OP_UNARY_SOFTPLUS:  compute_func = (فراغ *) tile_softplus_f32;       اكسر;
            حالة HTP_OP_UNARY_TANH:      compute_func = (فراغ *) tile_tanh_f32;           اكسر;
            حالة HTP_OP_UNARY_ABS:       compute_func = (فراغ *) tile_abs_f32;            اكسر;
            حالة HTP_OP_UNARY_LOG:       compute_func = (فراغ *) tile_log_f32;            اكسر;
            حالة HTP_OP_UNARY_RELU:      compute_func = (فراغ *) tile_relu_f32;           اكسر;
            حالة HTP_OP_UNARY_ګام:      compute_func = (فراغ *) tile_ګام_f32;           اكسر;
            حالة HTP_OP_TRI:
                task_func    = unary_thread_tiled_tri_f32;
                compute_func = (فراغ *) tri_apply_tile_f32;
                اكسر;
            افتراضي:                     اكسر;
        }
    } وإلا إذا (is_f16) {
        task_func = unary_thread_لیکه_بلاک;
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
            حالة HTP_OP_UNARY_ګام:      compute_func = (فراغ *) ګام_f16;                اكسر;
            حالة HTP_OP_HADAMARD:        compute_func = (فراغ *) hadamard_f16;            اكسر;
            افتراضي:                     اكسر;
        }
    } وإلا {
        task_func = unary_thread_لیکه_بلاک;
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
            حالة HTP_OP_UNARY_GELU_چټک: compute_func = (فراغ *) gelu_چټک_f32;          اكسر;
            حالة HTP_OP_UNARY_GELU_ERF:  compute_func = (فراغ *) gelu_erf_f32;            اكسر;
            حالة HTP_OP_UNARY_SOFTPLUS:  compute_func = (فراغ *) softplus_f32;            اكسر;
            حالة HTP_OP_UNARY_TANH:      compute_func = (فراغ *) tanh_f32;                اكسر;
            حالة HTP_OP_UNARY_ABS:       compute_func = (فراغ *) abs_f32;                 اكسر;
            حالة HTP_OP_UNARY_LOG:       compute_func = (فراغ *) log_f32;                 اكسر;
            حالة HTP_OP_UNARY_RELU:      compute_func = (فراغ *) relu_f32;                اكسر;
            حالة HTP_OP_UNARY_ګام:      compute_func = (فراغ *) ګام_f32;                اكسر;
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
        أرجع HTP_حالت_NO_SUPPORT;
    }

    uctx.compute = compute_func;
    work_کتار_پرلپسې(octx->ctx->work_کتار, task_func, &uctx, n_تارونه);

    أرجع err;
}

صحيح op_unary(بنية htp_ops_context * octx) {
    بدل (octx->src[0]->type) {
        حالة HTP_TYPE_F32:
        حالة HTP_TYPE_F16:
            أرجع execute_op_unary(octx);

        افتراضي:
            أرجع HTP_حالت_NO_SUPPORT;
    }
}
