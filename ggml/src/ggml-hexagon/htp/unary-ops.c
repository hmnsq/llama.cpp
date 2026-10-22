#ਹਦਾਇਤ clang diagnostic ignored "-Wunused-variable"
#ਹਦਾਇਤ clang diagnostic ignored "-Wunused-function"
#ਹਦਾਇਤ clang diagnostic ignored "-Wunused-but-set-variable"

#ਅਨ੍ਤਰ੍ਭਾਵ <HAP_farf.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <HAP_perf.h>

#ਅਨ੍ਤਰ੍ਭਾਵ <math.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <string.h>

#ਅਨ੍ਤਰ੍ਭਾਵ "dma-queue.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "hex-fastdiv.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "hvx-exp.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "hvx-sigmoid.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "hvx-utils.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "unary-ops.h"

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_COMMON_DECL_C
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-common.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "htp-ctx.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "htp-ops.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "htp-tensor.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "htp-vtcm.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "hex-profile.h"

ਬਣਤਰ htp_unary_context;

ਕਿਸਮਪਰਿਭਾਸ਼ਾ ਖਾਲੀ (*unary_compute_fn_t)(ਅਚਲ ਖਾਲੀ * ਸੀਮਿਤ src,
                                   ਖਾਲੀ * ਸੀਮਿਤ dst,
                                   uint32_t ਅੰਕ_ਪੰਕਤੀਆਂ_2,
                                   ਅਚਲ ਬਣਤਰ htp_unary_context * uctx);

ਕਿਸਮਪਰਿਭਾਸ਼ਾ ਖਾਲੀ (*unary_rms_norm_mul_compute_fn_t)(ਅਚਲ ਖਾਲੀ * ਸੀਮਿਤ src,
                                                ਅਚਲ ਖਾਲੀ * ਸੀਮਿਤ ਭਾਰ,
                                                ਖਾਲੀ * ਸੀਮਿਤ dst,
                                                uint32_t ਅੰਕ_ਪੰਕਤੀਆਂ_2,
                                                ਅਚਲ ਬਣਤਰ htp_unary_context * uctx);

ਕਿਸਮਪਰਿਭਾਸ਼ਾ ਖਾਲੀ (*unary_tri_compute_fn_t)(ਅਚਲ ਖਾਲੀ * ਸੀਮਿਤ src,
                                       ਖਾਲੀ * ਸੀਮਿਤ dst,
                                       uint32_t ਅੰਕ_ਪੰਕਤੀਆਂ_2,
                                       uint32_t ir,
                                       ਅਚਲ ਬਣਤਰ htp_unary_context * uctx);

ਕਿਸਮਪਰਿਭਾਸ਼ਾ ਖਾਲੀ (*unary_tile_compute_fn_t)(ਖਾਲੀ * ਸੀਮਿਤ dst,
                                        ਅਚਲ ਖਾਲੀ * ਸੀਮਿਤ src,
                                        uint32_t tw,
                                        ਅਚਲ ਬਣਤਰ htp_unary_context * uctx);

ਕਿਸਮਪਰਿਭਾਸ਼ਾ ਖਾਲੀ (*unary_tiled_tri_compute_fn_t)(ਅਚਲ ਖਾਲੀ * ਸੀਮਿਤ src,
                                             ਖਾਲੀ * ਸੀਮਿਤ dst,
                                             uint32_t tile_elems,
                                             uint32_t ਖਾਨਾ_ਸ਼ੁਰੂ,
                                             uint32_t i01,
                                             uint32_t ne0,
                                             int32_t ttype);

ਬਣਤਰ htp_unary_context {
    ਬਣਤਰ htp_ops_context * octx;
    ਅਚਲ ਬਣਤਰ htp_unary_kernel_params * kparams;

    ਖਾਲੀ *                    compute;

    dma_addr_t                ਅੰਕੜੇ_src0;
    dma_addr_t                ਅੰਕੜੇ_src1;            // weight/scale tensor for RMS_NORM_MUL
    dma_addr_t                ਅੰਕੜੇ_dst;

    size_t                    src0_ਅੰਕੜੇ_ਪੰਕਤੀ_ਮਾਪ;   // actual data bytes per row
    size_t                    src1_ਅੰਕੜੇ_ਪੰਕਤੀ_ਮਾਪ;
    size_t                    dst_ਅੰਕੜੇ_ਪੰਕਤੀ_ਮਾਪ;    // actual data bytes per row

    size_t                    src0_ਪੰਕਤੀ_ਮਾਪ_aligned;
    size_t                    src1_ਪੰਕਤੀ_ਮਾਪ_aligned;
    size_t                    dst_ਪੰਕਤੀ_ਮਾਪ_aligned;

    size_t                    src0_vtcm_half_ਮਾਪ;
    size_t                    src1_vtcm_half_ਮਾਪ;
    size_t                    dst_vtcm_half_ਮਾਪ;

    uint32_t                  ਖੰਡ;
    uint32_t                  src0_nrows;
    uint32_t                  src0_nrows_ਪ੍ਰਤੀ_thread;
    uint32_t                  ਪੰਕਤੀ_ਸ਼ੁਰੂ;
    uint32_t                  nc;
    uint32_t                  ਖਾਨਾ_tile;             // tiled mode
    bool                      ਪ੍ਰਸਾਰਣ_ਭਾਰ;

    uint8_t *                 vtcm_src0;
    uint8_t *                 vtcm_src1;
    uint8_t *                 vtcm_dst;

    size_t                    vtcm_src0_ਮਾਪ_ਪ੍ਰਤੀ_thread;
    size_t                    vtcm_src1_ਮਾਪ_ਪ੍ਰਤੀ_thread;
    size_t                    vtcm_dst_ਮਾਪ_ਪ੍ਰਤੀ_thread;
};

// Convert flat row index to DDR byte offset using the tensor's actual strides.
// ir = i1 + ne1*(i2 + ne2*i3)  =>  offset = i1*nb1 + i2*nb2 + i3*nb3
ਸਥਿਰ ਸਤਰਅੰਦਰ size_t unary_ਪੰਕਤੀ_ਵਿੱਥ(uint32_t ir,
                                      uint32_t ne1, uint32_t ne2,
                                      ਅਚਲ ਬਣਤਰ fastdiv_ਮੁੱਲਾਂ * div_ne1,
                                      ਅਚਲ ਬਣਤਰ fastdiv_ਮੁੱਲਾਂ * div_ne2,
                                      ਅਚਲ ਬਣਤਰ fastdiv_ਮੁੱਲਾਂ * div_ne12,
                                      size_t nb1, size_t nb2, size_t nb3) {
    ਅਚਲ uint32_t i1 = fastmodulo(ir, ne1, div_ne1);
    ਅਚਲ uint32_t ir_div_ne1 = fastdiv(ir, div_ne1);
    ਅਚਲ uint32_t i2 = fastmodulo(ir_div_ne1, ne2, div_ne2);
    ਅਚਲ uint32_t i3 = fastdiv(ir, div_ne12);
    ਵਾਪਸ i1 * nb1 + i2 * nb2 + i3 * nb3;
}

// Safe DMA block size from row `ir`: clamp to the tighter dim-1 slice
// boundary of src and dst so the nb1 stride stays valid for all rows.
ਸਥਿਰ ਸਤਰਅੰਦਰ uint32_t unary_ਖੰਡ_ਮਾਪ(uint32_t ir,
                                        uint32_t ਸਿਰਾ_ਪੰਕਤੀ,
                                        uint32_t ਖੰਡ,
                                        bool src_contig,
                                        bool dst_contig,
                                        uint32_t ne1,
                                        ਅਚਲ ਬਣਤਰ fastdiv_ਮੁੱਲਾਂ * div_ne1) {
    uint32_t ਹੱਦਬੰਦੀ = MIN(ਖੰਡ, ਸਿਰਾ_ਪੰਕਤੀ - ir);

    ਜੇ (!src_contig || !dst_contig) {
        ਅਚਲ uint32_t ਟੁਕੜਾ_ਸਿਰਾ = (fastdiv(ir, div_ne1) + 1) * ne1;
        ਹੱਦਬੰਦੀ = MIN(ਹੱਦਬੰਦੀ, ਟੁਕੜਾ_ਸਿਰਾ - ir);
    }

    ਵਾਪਸ ਹੱਦਬੰਦੀ;
}

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ htp_unary_preamble            \
    ਅਚਲ uint32_t ne00 = src->ne[0]; \
    ਅਚਲ uint32_t ne01 = src->ne[1]; \
    ਅਚਲ uint32_t ne02 = src->ne[2]; \
    ਅਚਲ uint32_t ne03 = src->ne[3]; \
                                      \
    ਅਚਲ uint32_t ne0 = dst->ne[0];  \
    ਅਚਲ uint32_t ne1 = dst->ne[1];  \
    ਅਚਲ uint32_t ne2 = dst->ne[2];  \
    ਅਚਲ uint32_t ne3 = dst->ne[3];  \
                                      \
    ਅਚਲ uint32_t nb00 = src->nb[0]; \
    ਅਚਲ uint32_t nb01 = src->nb[1]; \
    ਅਚਲ uint32_t nb02 = src->nb[2]; \
    ਅਚਲ uint32_t nb03 = src->nb[3]; \
                                      \
    ਅਚਲ uint32_t nb0 = dst->nb[0];  \
    ਅਚਲ uint32_t nb1 = dst->nb[1];  \
    ਅਚਲ uint32_t nb2 = dst->nb[2];  \
    ਅਚਲ uint32_t nb3 = dst->nb[3];

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ htp_unary_op_preamble                                         \
    int32_t * op_params = uctx->octx->op_params;                      \
    ਅਚਲ uint32_t ne0 = uctx->nc;                                    \
    ਅਚਲ size_t src0_ਪੰਕਤੀ_ਮਾਪ_aligned = uctx->src0_ਪੰਕਤੀ_ਮਾਪ_aligned; \
    ਅਚਲ size_t dst_ਪੰਕਤੀ_ਮਾਪ_aligned = uctx->dst_ਪੰਕਤੀ_ਮਾਪ_aligned;

ਸਥਿਰ ਖਾਲੀ scale_f32(ਅਚਲ ਖਾਲੀ * ਸੀਮਿਤ src,
                      ਖਾਲੀ * ਸੀਮਿਤ dst,
                      ਅਚਲ uint32_t ਅੰਕ_ਪੰਕਤੀਆਂ_2,
                      ਅਚਲ ਬਣਤਰ htp_unary_context * uctx) {
    htp_unary_op_preamble;
    ਦਸ਼ਮਲਵ scale = 0.f;
    ਦਸ਼ਮਲਵ bias  = 0.f;
    ਸਮ੍ਰਿਤੀ_ਨਕਲ(&scale, &op_params[0], ਆਕਾਰ(ਦਸ਼ਮਲਵ));
    ਸਮ੍ਰਿਤੀ_ਨਕਲ(&bias,  &op_params[1], ਆਕਾਰ(ਦਸ਼ਮਲਵ));

    ਲਈ (uint32_t ir = 0; ir < ਅੰਕ_ਪੰਕਤੀਆਂ_2; ir++) {
        ਅਚਲ uint8_t * ਸੀਮਿਤ src_local = (ਅਚਲ uint8_t *)src + (ir * src0_ਪੰਕਤੀ_ਮਾਪ_aligned);
        uint8_t * ਸੀਮਿਤ dst_local       = (uint8_t *)dst + (ir * dst_ਪੰਕਤੀ_ਮਾਪ_aligned);

        hvx_scale_ਵਿੱਥ_f32_aa((uint8_t *) dst_local, (ਅਚਲ uint8_t *) src_local, ne0, scale, bias);
    }
}

ਸਥਿਰ ਖਾਲੀ clamp_f32(ਅਚਲ ਖਾਲੀ * ਸੀਮਿਤ src,
                      ਖਾਲੀ * ਸੀਮਿਤ dst,
                      ਅਚਲ uint32_t ਅੰਕ_ਪੰਕਤੀਆਂ_2,
                      ਅਚਲ ਬਣਤਰ htp_unary_context * uctx) {
    htp_unary_op_preamble;
    ਦਸ਼ਮਲਵ min = 0.f;
    ਦਸ਼ਮਲਵ max = 0.f;
    ਸਮ੍ਰਿਤੀ_ਨਕਲ(&min, &op_params[0], ਆਕਾਰ(ਦਸ਼ਮਲਵ));
    ਸਮ੍ਰਿਤੀ_ਨਕਲ(&max, &op_params[1], ਆਕਾਰ(ਦਸ਼ਮਲਵ));

    ਲਈ (uint32_t ir = 0; ir < ਅੰਕ_ਪੰਕਤੀਆਂ_2; ir++) {
        ਅਚਲ uint8_t * ਸੀਮਿਤ src_local = (ਅਚਲ uint8_t *)src + (ir * src0_ਪੰਕਤੀ_ਮਾਪ_aligned);
        uint8_t * ਸੀਮਿਤ dst_local       = (uint8_t *)dst + (ir * dst_ਪੰਕਤੀ_ਮਾਪ_aligned);

        hvx_clamp_scalar_f32(dst_local, src_local, min, max, ne0);
    }
}

ਸਥਿਰ ਖਾਲੀ leaky_relu_f32(ਅਚਲ ਖਾਲੀ * ਸੀਮਿਤ src,
                           ਖਾਲੀ * ਸੀਮਿਤ dst,
                           ਅਚਲ uint32_t ਅੰਕ_ਪੰਕਤੀਆਂ_2,
                           ਅਚਲ ਬਣਤਰ htp_unary_context * uctx) {
    htp_unary_op_preamble;
    ਦਸ਼ਮਲਵ negative_ਢਲਾਨ = 0.f;
    ਸਮ੍ਰਿਤੀ_ਨਕਲ(&negative_ਢਲਾਨ, &op_params[0], ਆਕਾਰ(ਦਸ਼ਮਲਵ));

    ਲਈ (uint32_t ir = 0; ir < ਅੰਕ_ਪੰਕਤੀਆਂ_2; ir++) {
        ਅਚਲ uint8_t * ਸੀਮਿਤ src_local = (ਅਚਲ uint8_t *)src + (ir * src0_ਪੰਕਤੀ_ਮਾਪ_aligned);
        uint8_t * ਸੀਮਿਤ dst_local       = (uint8_t *)dst + (ir * dst_ਪੰਕਤੀ_ਮਾਪ_aligned);

        hvx_leaky_relu_scalar_f32(dst_local, src_local, negative_ਢਲਾਨ, ne0);
    }
}

ਸਥਿਰ ਖਾਲੀ rms_norm_f32(ਅਚਲ ਖਾਲੀ * ਸੀਮਿਤ src,
                         ਖਾਲੀ * ਸੀਮਿਤ dst,
                         ਅਚਲ uint32_t ਅੰਕ_ਪੰਕਤੀਆਂ_2,
                         ਅਚਲ ਬਣਤਰ htp_unary_context * uctx) {
    htp_unary_op_preamble;
    ਦਸ਼ਮਲਵ epsilon = 0.f;
    ਸਮ੍ਰਿਤੀ_ਨਕਲ(&epsilon, op_params, ਆਕਾਰ(ਦਸ਼ਮਲਵ));

    ਲਈ (uint32_t ir = 0; ir < ਅੰਕ_ਪੰਕਤੀਆਂ_2; ir++) {
        ਅਚਲ uint8_t * ਸੀਮਿਤ src_local = (ਅਚਲ uint8_t *)src + (ir * src0_ਪੰਕਤੀ_ਮਾਪ_aligned);
        uint8_t * ਸੀਮਿਤ dst_local       = (uint8_t *)dst + (ir * dst_ਪੰਕਤੀ_ਮਾਪ_aligned);

        hvx_fast_rms_norm_f32((ਅਚਲ uint8_t *) src_local, (uint8_t *) dst_local, ne0, epsilon);
    }
}

ਸਥਿਰ ਖਾਲੀ rms_norm_mul_f32(ਅਚਲ ਖਾਲੀ * ਸੀਮਿਤ src,
                             ਅਚਲ ਖਾਲੀ * ਸੀਮਿਤ ਭਾਰ,
                             ਖਾਲੀ * ਸੀਮਿਤ dst,
                             ਅਚਲ uint32_t ਅੰਕ_ਪੰਕਤੀਆਂ_2,
                             ਅਚਲ ਬਣਤਰ htp_unary_context * uctx) {
    htp_unary_op_preamble;
    ਦਸ਼ਮਲਵ epsilon = 0.f;
    ਸਮ੍ਰਿਤੀ_ਨਕਲ(&epsilon, op_params, ਆਕਾਰ(ਦਸ਼ਮਲਵ));

    ਲਈ (uint32_t ir = 0; ir < ਅੰਕ_ਪੰਕਤੀਆਂ_2; ir++) {
        ਅਚਲ uint8_t * ਸੀਮਿਤ src_local = (ਅਚਲ uint8_t *)src + (ir * src0_ਪੰਕਤੀ_ਮਾਪ_aligned);
        ਅਚਲ uint8_t * ਸੀਮਿਤ w_local   = (ਅਚਲ uint8_t *)ਭਾਰ + (uctx->ਪ੍ਰਸਾਰਣ_ਭਾਰ ? 0 : ir * uctx->src1_ਪੰਕਤੀ_ਮਾਪ_aligned);
        uint8_t * ਸੀਮਿਤ dst_local       = (uint8_t *)dst + (ir * dst_ਪੰਕਤੀ_ਮਾਪ_aligned);

        hvx_fast_rms_norm_mul_f32(src_local, w_local, dst_local, ne0, epsilon);
    }
}

ਸਥਿਰ ਖਾਲੀ norm_f32(ਅਚਲ ਖਾਲੀ * ਸੀਮਿਤ src,
                     ਖਾਲੀ * ਸੀਮਿਤ dst,
                     ਅਚਲ uint32_t ਅੰਕ_ਪੰਕਤੀਆਂ_2,
                     ਅਚਲ ਬਣਤਰ htp_unary_context * uctx) {
    htp_unary_op_preamble;
    ਦਸ਼ਮਲਵ epsilon = 0.f;
    ਸਮ੍ਰਿਤੀ_ਨਕਲ(&epsilon, op_params, ਆਕਾਰ(ਦਸ਼ਮਲਵ));

    ਲਈ (uint32_t ir = 0; ir < ਅੰਕ_ਪੰਕਤੀਆਂ_2; ir++) {
        ਅਚਲ uint8_t * ਸੀਮਿਤ src_local = (ਅਚਲ uint8_t *)src + (ir * src0_ਪੰਕਤੀ_ਮਾਪ_aligned);
        uint8_t * ਸੀਮਿਤ dst_local       = (uint8_t *)dst + (ir * dst_ਪੰਕਤੀ_ਮਾਪ_aligned);

        hvx_fast_norm_f32((ਅਚਲ uint8_t *) src_local, (uint8_t *) dst_local, ne0, epsilon);
    }
}

ਸਥਿਰ ਖਾਲੀ sqr_f32(ਅਚਲ ਖਾਲੀ * ਸੀਮਿਤ src,
                    ਖਾਲੀ * ਸੀਮਿਤ dst,
                    ਅਚਲ uint32_t ਅੰਕ_ਪੰਕਤੀਆਂ_2,
                    ਅਚਲ ਬਣਤਰ htp_unary_context * uctx) {
    htp_unary_op_preamble;

    ਲਈ (uint32_t ir = 0; ir < ਅੰਕ_ਪੰਕਤੀਆਂ_2; ir++) {
        ਅਚਲ uint8_t * ਸੀਮਿਤ src_local = (ਅਚਲ uint8_t *)src + (ir * src0_ਪੰਕਤੀ_ਮਾਪ_aligned);
        uint8_t * ਸੀਮਿਤ dst_local       = (uint8_t *)dst + (ir * dst_ਪੰਕਤੀ_ਮਾਪ_aligned);

        hvx_sqr_f32_aa((uint8_t *) dst_local, (ਅਚਲ uint8_t *) src_local, ne0);
    }
}

ਸਥਿਰ ਖਾਲੀ sqrt_f32(ਅਚਲ ਖਾਲੀ * ਸੀਮਿਤ src,
                     ਖਾਲੀ * ਸੀਮਿਤ dst,
                     ਅਚਲ uint32_t ਅੰਕ_ਪੰਕਤੀਆਂ_2,
                     ਅਚਲ ਬਣਤਰ htp_unary_context * uctx) {
    htp_unary_op_preamble;

    ਲਈ (uint32_t ir = 0; ir < ਅੰਕ_ਪੰਕਤੀਆਂ_2; ir++) {
        ਅਚਲ uint8_t * ਸੀਮਿਤ src_local = (ਅਚਲ uint8_t *)src + (ir * src0_ਪੰਕਤੀ_ਮਾਪ_aligned);
        uint8_t * ਸੀਮਿਤ dst_local       = (uint8_t *)dst + (ir * dst_ਪੰਕਤੀ_ਮਾਪ_aligned);

        hvx_sqrt_f32_aa((uint8_t *) dst_local, (ਅਚਲ uint8_t *) src_local, ne0);
    }
}

ਸਥਿਰ ਖਾਲੀ scale_f16(ਅਚਲ ਖਾਲੀ * ਸੀਮਿਤ src,
                      ਖਾਲੀ * ਸੀਮਿਤ dst,
                      ਅਚਲ uint32_t ਅੰਕ_ਪੰਕਤੀਆਂ_2,
                      ਅਚਲ ਬਣਤਰ htp_unary_context * uctx) {
    htp_unary_op_preamble;
    ਦਸ਼ਮਲਵ scale = 0.f;
    ਦਸ਼ਮਲਵ bias  = 0.f;
    ਸਮ੍ਰਿਤੀ_ਨਕਲ(&scale, &op_params[0], ਆਕਾਰ(ਦਸ਼ਮਲਵ));
    ਸਮ੍ਰਿਤੀ_ਨਕਲ(&bias,  &op_params[1], ਆਕਾਰ(ਦਸ਼ਮਲਵ));

    ਲਈ (uint32_t ir = 0; ir < ਅੰਕ_ਪੰਕਤੀਆਂ_2; ir++) {
        ਅਚਲ uint8_t * ਸੀਮਿਤ src_local = (ਅਚਲ uint8_t *)src + (ir * src0_ਪੰਕਤੀ_ਮਾਪ_aligned);
        uint8_t * ਸੀਮਿਤ dst_local       = (uint8_t *)dst + (ir * dst_ਪੰਕਤੀ_ਮਾਪ_aligned);

        hvx_scale_ਵਿੱਥ_f16_aa((uint8_t *) dst_local, (ਅਚਲ uint8_t *) src_local, ne0, scale, bias);
    }
}

ਸਥਿਰ ਖਾਲੀ clamp_f16(ਅਚਲ ਖਾਲੀ * ਸੀਮਿਤ src,
                      ਖਾਲੀ * ਸੀਮਿਤ dst,
                      ਅਚਲ uint32_t ਅੰਕ_ਪੰਕਤੀਆਂ_2,
                      ਅਚਲ ਬਣਤਰ htp_unary_context * uctx) {
    htp_unary_op_preamble;
    ਦਸ਼ਮਲਵ min = 0.f;
    ਦਸ਼ਮਲਵ max = 0.f;
    ਸਮ੍ਰਿਤੀ_ਨਕਲ(&min, &op_params[0], ਆਕਾਰ(ਦਸ਼ਮਲਵ));
    ਸਮ੍ਰਿਤੀ_ਨਕਲ(&max, &op_params[1], ਆਕਾਰ(ਦਸ਼ਮਲਵ));

    ਲਈ (uint32_t ir = 0; ir < ਅੰਕ_ਪੰਕਤੀਆਂ_2; ir++) {
        ਅਚਲ uint8_t * ਸੀਮਿਤ src_local = (ਅਚਲ uint8_t *)src + (ir * src0_ਪੰਕਤੀ_ਮਾਪ_aligned);
        uint8_t * ਸੀਮਿਤ dst_local       = (uint8_t *)dst + (ir * dst_ਪੰਕਤੀ_ਮਾਪ_aligned);

        hvx_clamp_scalar_f16(dst_local, src_local, (_Float16) min, (_Float16) max, ne0);
    }
}

ਸਥਿਰ ਖਾਲੀ rms_norm_f16(ਅਚਲ ਖਾਲੀ * ਸੀਮਿਤ src,
                         ਖਾਲੀ * ਸੀਮਿਤ dst,
                         ਅਚਲ uint32_t ਅੰਕ_ਪੰਕਤੀਆਂ_2,
                         ਅਚਲ ਬਣਤਰ htp_unary_context * uctx) {
    htp_unary_op_preamble;
    ਦਸ਼ਮਲਵ epsilon = 0.f;
    ਸਮ੍ਰਿਤੀ_ਨਕਲ(&epsilon, op_params, ਆਕਾਰ(ਦਸ਼ਮਲਵ));

    ਲਈ (uint32_t ir = 0; ir < ਅੰਕ_ਪੰਕਤੀਆਂ_2; ir++) {
        ਅਚਲ uint8_t * ਸੀਮਿਤ src_local = (ਅਚਲ uint8_t *)src + (ir * src0_ਪੰਕਤੀ_ਮਾਪ_aligned);
        uint8_t * ਸੀਮਿਤ dst_local       = (uint8_t *)dst + (ir * dst_ਪੰਕਤੀ_ਮਾਪ_aligned);

        hvx_fast_rms_norm_f16((ਅਚਲ uint8_t *) src_local, (uint8_t *) dst_local, ne0, epsilon);
    }
}

ਸਥਿਰ ਖਾਲੀ norm_f16(ਅਚਲ ਖਾਲੀ * ਸੀਮਿਤ src,
                     ਖਾਲੀ * ਸੀਮਿਤ dst,
                     ਅਚਲ uint32_t ਅੰਕ_ਪੰਕਤੀਆਂ_2,
                     ਅਚਲ ਬਣਤਰ htp_unary_context * uctx) {
    htp_unary_op_preamble;
    ਦਸ਼ਮਲਵ epsilon = 0.f;
    ਸਮ੍ਰਿਤੀ_ਨਕਲ(&epsilon, op_params, ਆਕਾਰ(ਦਸ਼ਮਲਵ));

    ਲਈ (uint32_t ir = 0; ir < ਅੰਕ_ਪੰਕਤੀਆਂ_2; ir++) {
        ਅਚਲ uint8_t * ਸੀਮਿਤ src_local = (ਅਚਲ uint8_t *)src + (ir * src0_ਪੰਕਤੀ_ਮਾਪ_aligned);
        uint8_t * ਸੀਮਿਤ dst_local       = (uint8_t *)dst + (ir * dst_ਪੰਕਤੀ_ਮਾਪ_aligned);

        hvx_fast_norm_f16((ਅਚਲ uint8_t *) src_local, (uint8_t *) dst_local, ne0, epsilon);
    }
}

ਸਥਿਰ ਖਾਲੀ sqr_f16(ਅਚਲ ਖਾਲੀ * ਸੀਮਿਤ src,
                    ਖਾਲੀ * ਸੀਮਿਤ dst,
                    ਅਚਲ uint32_t ਅੰਕ_ਪੰਕਤੀਆਂ_2,
                    ਅਚਲ ਬਣਤਰ htp_unary_context * uctx) {
    htp_unary_op_preamble;

    ਲਈ (uint32_t ir = 0; ir < ਅੰਕ_ਪੰਕਤੀਆਂ_2; ir++) {
        ਅਚਲ uint8_t * ਸੀਮਿਤ src_local = (ਅਚਲ uint8_t *)src + (ir * src0_ਪੰਕਤੀ_ਮਾਪ_aligned);
        uint8_t * ਸੀਮਿਤ dst_local       = (uint8_t *)dst + (ir * dst_ਪੰਕਤੀ_ਮਾਪ_aligned);

        hvx_sqr_f16_aa((uint8_t *) dst_local, (ਅਚਲ uint8_t *) src_local, ne0);
    }
}

ਸਥਿਰ ਖਾਲੀ sqrt_f16(ਅਚਲ ਖਾਲੀ * ਸੀਮਿਤ src,
                     ਖਾਲੀ * ਸੀਮਿਤ dst,
                     ਅਚਲ uint32_t ਅੰਕ_ਪੰਕਤੀਆਂ_2,
                     ਅਚਲ ਬਣਤਰ htp_unary_context * uctx) {
    htp_unary_op_preamble;

    ਲਈ (uint32_t ir = 0; ir < ਅੰਕ_ਪੰਕਤੀਆਂ_2; ir++) {
        ਅਚਲ uint8_t * ਸੀਮਿਤ src_local = (ਅਚਲ uint8_t *)src + (ir * src0_ਪੰਕਤੀ_ਮਾਪ_aligned);
        uint8_t * ਸੀਮਿਤ dst_local       = (uint8_t *)dst + (ir * dst_ਪੰਕਤੀ_ਮਾਪ_aligned);

        hvx_sqrt_f16_aa((uint8_t *) dst_local, (ਅਚਲ uint8_t *) src_local, ne0);
    }
}

ਸਥਿਰ ਖਾਲੀ abs_f16(ਅਚਲ ਖਾਲੀ * ਸੀਮਿਤ src,
                    ਖਾਲੀ * ਸੀਮਿਤ dst,
                    ਅਚਲ uint32_t ਅੰਕ_ਪੰਕਤੀਆਂ_2,
                    ਅਚਲ ਬਣਤਰ htp_unary_context * uctx) {
    htp_unary_op_preamble;

    ਲਈ (uint32_t ir = 0; ir < ਅੰਕ_ਪੰਕਤੀਆਂ_2; ir++) {
        ਅਚਲ uint8_t * ਸੀਮਿਤ src_local = (ਅਚਲ uint8_t *)src + (ir * src0_ਪੰਕਤੀ_ਮਾਪ_aligned);
        uint8_t * ਸੀਮਿਤ dst_local       = (uint8_t *)dst + (ir * dst_ਪੰਕਤੀ_ਮਾਪ_aligned);

        hvx_abs_f16_aa((uint8_t *) dst_local, (ਅਚਲ uint8_t *) src_local, ne0);
    }
}

ਸਥਿਰ ਖਾਲੀ log_f16(ਅਚਲ ਖਾਲੀ * ਸੀਮਿਤ src,
                    ਖਾਲੀ * ਸੀਮਿਤ dst,
                    ਅਚਲ uint32_t ਅੰਕ_ਪੰਕਤੀਆਂ_2,
                    ਅਚਲ ਬਣਤਰ htp_unary_context * uctx) {
    htp_unary_op_preamble;

    ਲਈ (uint32_t ir = 0; ir < ਅੰਕ_ਪੰਕਤੀਆਂ_2; ir++) {
        ਅਚਲ uint8_t * ਸੀਮਿਤ src_local = (ਅਚਲ uint8_t *)src + (ir * src0_ਪੰਕਤੀ_ਮਾਪ_aligned);
        uint8_t * ਸੀਮਿਤ dst_local       = (uint8_t *)dst + (ir * dst_ਪੰਕਤੀ_ਮਾਪ_aligned);

        hvx_log_f16_aa((uint8_t *) dst_local, (ਅਚਲ uint8_t *) src_local, ne0);
    }
}

ਸਥਿਰ ਖਾਲੀ ਕਦਮ_f16(ਅਚਲ ਖਾਲੀ * ਸੀਮਿਤ src,
                     ਖਾਲੀ * ਸੀਮਿਤ dst,
                     ਅਚਲ uint32_t ਅੰਕ_ਪੰਕਤੀਆਂ_2,
                     ਅਚਲ ਬਣਤਰ htp_unary_context * uctx) {
    htp_unary_op_preamble;

    ਲਈ (uint32_t ir = 0; ir < ਅੰਕ_ਪੰਕਤੀਆਂ_2; ir++) {
        ਅਚਲ uint8_t * ਸੀਮਿਤ src_local = (ਅਚਲ uint8_t *)src + (ir * src0_ਪੰਕਤੀ_ਮਾਪ_aligned);
        uint8_t * ਸੀਮਿਤ dst_local       = (uint8_t *)dst + (ir * dst_ਪੰਕਤੀ_ਮਾਪ_aligned);

        hvx_ਕਦਮ_f16_aa((uint8_t *) dst_local, (ਅਚਲ uint8_t *) src_local, ne0);
    }
}

ਸਥਿਰ ਖਾਲੀ l2_norm_f16(ਅਚਲ ਖਾਲੀ * ਸੀਮਿਤ src,
                        ਖਾਲੀ * ਸੀਮਿਤ dst,
                        ਅਚਲ uint32_t ਅੰਕ_ਪੰਕਤੀਆਂ_2,
                        ਅਚਲ ਬਣਤਰ htp_unary_context * uctx) {
    htp_unary_op_preamble;
    ਦਸ਼ਮਲਵ epsilon = 0.f;
    ਸਮ੍ਰਿਤੀ_ਨਕਲ(&epsilon, op_params, ਆਕਾਰ(ਦਸ਼ਮਲਵ));

    ਲਈ (uint32_t ir = 0; ir < ਅੰਕ_ਪੰਕਤੀਆਂ_2; ir++) {
        ਅਚਲ uint8_t * ਸੀਮਿਤ src_f = (ਅਚਲ uint8_t *)src + (ir * src0_ਪੰਕਤੀ_ਮਾਪ_aligned);
        uint8_t * ਸੀਮਿਤ dst_f       = (uint8_t *)dst + (ir * dst_ਪੰਕਤੀ_ਮਾਪ_aligned);

        hvx_fast_l2_norm_f16((ਅਚਲ uint8_t *)src_f, (uint8_t *)dst_f, ne0, epsilon);
    }
}

ਸਥਿਰ ਖਾਲੀ neg_f32(ਅਚਲ ਖਾਲੀ * ਸੀਮਿਤ src,
                    ਖਾਲੀ * ਸੀਮਿਤ dst,
                    ਅਚਲ uint32_t ਅੰਕ_ਪੰਕਤੀਆਂ_2,
                    ਅਚਲ ਬਣਤਰ htp_unary_context * uctx) {
    htp_unary_op_preamble;

    ਲਈ (uint32_t ir = 0; ir < ਅੰਕ_ਪੰਕਤੀਆਂ_2; ir++) {
        ਅਚਲ uint8_t * ਸੀਮਿਤ src_local = (ਅਚਲ uint8_t *)src + (ir * src0_ਪੰਕਤੀ_ਮਾਪ_aligned);
        uint8_t * ਸੀਮਿਤ dst_local       = (uint8_t *)dst + (ir * dst_ਪੰਕਤੀ_ਮਾਪ_aligned);

        hvx_scale_f32_aa(dst_local, src_local, ne0, -1.0f);
    }
}

ਸਥਿਰ ਖਾਲੀ exp_f32(ਅਚਲ ਖਾਲੀ * ਸੀਮਿਤ src,
                    ਖਾਲੀ * ਸੀਮਿਤ dst,
                    ਅਚਲ uint32_t ਅੰਕ_ਪੰਕਤੀਆਂ_2,
                    ਅਚਲ ਬਣਤਰ htp_unary_context * uctx) {
    htp_unary_op_preamble;

    ਲਈ (uint32_t ir = 0; ir < ਅੰਕ_ਪੰਕਤੀਆਂ_2; ir++) {
        ਅਚਲ uint8_t * ਸੀਮਿਤ src_local = (ਅਚਲ uint8_t *)src + (ir * src0_ਪੰਕਤੀ_ਮਾਪ_aligned);
        uint8_t * ਸੀਮਿਤ dst_local       = (uint8_t *)dst + (ir * dst_ਪੰਕਤੀ_ਮਾਪ_aligned);

        hvx_exp_f32(dst_local, src_local, ne0, false);
    }
}

ਸਥਿਰ ਖਾਲੀ sigmoid_f32(ਅਚਲ ਖਾਲੀ * ਸੀਮਿਤ src,
                        ਖਾਲੀ * ਸੀਮਿਤ dst,
                        ਅਚਲ uint32_t ਅੰਕ_ਪੰਕਤੀਆਂ_2,
                        ਅਚਲ ਬਣਤਰ htp_unary_context * uctx) {
    htp_unary_op_preamble;

    ਲਈ (uint32_t ir = 0; ir < ਅੰਕ_ਪੰਕਤੀਆਂ_2; ir++) {
        ਅਚਲ uint8_t * ਸੀਮਿਤ src_local = (ਅਚਲ uint8_t *)src + (ir * src0_ਪੰਕਤੀ_ਮਾਪ_aligned);
        uint8_t * ਸੀਮਿਤ dst_local       = (uint8_t *)dst + (ir * dst_ਪੰਕਤੀ_ਮਾਪ_aligned);

        hvx_sigmoid_f32_aa(dst_local, src_local, ne0);
    }
}

// silu(x) = x * sigmoid(x)
ਸਥਿਰ ਖਾਲੀ silu_f32(ਅਚਲ ਖਾਲੀ * ਸੀਮਿਤ src,
                     ਖਾਲੀ * ਸੀਮਿਤ dst,
                     ਅਚਲ uint32_t ਅੰਕ_ਪੰਕਤੀਆਂ_2,
                     ਅਚਲ ਬਣਤਰ htp_unary_context * uctx) {
    htp_unary_op_preamble;

    ਲਈ (uint32_t ir = 0; ir < ਅੰਕ_ਪੰਕਤੀਆਂ_2; ir++) {
        ਅਚਲ uint8_t * ਸੀਮਿਤ src_local = (ਅਚਲ uint8_t *)src + (ir * src0_ਪੰਕਤੀ_ਮਾਪ_aligned);
        uint8_t * ਸੀਮਿਤ dst_local       = (uint8_t *)dst + (ir * dst_ਪੰਕਤੀ_ਮਾਪ_aligned);

        hvx_sigmoid_f32_aa(dst_local, src_local, ne0);
        hvx_mul_f32_aaa(dst_local, src_local, dst_local, ne0);
    }
}

// GELU uses the tanh approximation.
ਸਥਿਰ __attribute__((noinline)) HVX_ਸਦਿਸ਼ hvx_vec_gelu_f32(HVX_ਸਦਿਸ਼ x) {
    ਅਚਲ HVX_ਸਦਿਸ਼ half = hvx_vec_splat_f32(0.5f);
    ਅਚਲ HVX_ਸਦਿਸ਼ one  = hvx_vec_splat_f32(1.0f);

    HVX_ਸਦਿਸ਼ inner = hvx_vec_mul_f32_f32(x, x);
    inner = hvx_vec_mul_f32_f32(inner, hvx_vec_splat_f32(0.044715f));
    inner = hvx_vec_ਜੋੜੋ_f32_f32(inner, one);
    inner = hvx_vec_mul_f32_f32(inner, x);
    inner = hvx_vec_mul_f32_f32(inner, hvx_vec_splat_f32(0.7978845608028654f));

    ਵਾਪਸ hvx_vec_mul_f32_f32(hvx_vec_mul_f32_f32(half, x),
                               hvx_vec_ਜੋੜੋ_f32_f32(one, hvx_vec_tanh_f32(inner)));
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ hvx_gelu_f32_aa(uint8_t * ਸੀਮਿਤ dst, ਅਚਲ uint8_t * ਸੀਮਿਤ src, uint32_t n) {
    ਪ੍ਰਤਿਜ੍ਞਾ((ਬਿਨਾਨਿਸ਼ਾਨੀ ਲੰਮਾ) dst % 128 == 0);
    ਪ੍ਰਤਿਜ੍ਞਾ((ਬਿਨਾਨਿਸ਼ਾਨੀ ਲੰਮਾ) src % 128 == 0);

    HVX_ਸਦਿਸ਼ * ਸੀਮਿਤ vdst = (HVX_ਸਦਿਸ਼ *) dst;
    HVX_ਸਦਿਸ਼ * ਸੀਮਿਤ vsrc = (HVX_ਸਦਿਸ਼ *) src;

    ਅਚਲ uint32_t nvec = n / VLEN_FP32;
    ਅਚਲ uint32_t nloe = n % VLEN_FP32;

    uint32_t i = 0;
    _Pragma("unroll(4)")
    ਲਈ (; i < nvec; i++) {
        vdst[i] = hvx_vec_gelu_f32(vsrc[i]);
    }
    ਜੇ (nloe) {
        hvx_vec_store_a(&vdst[i], nloe * ਆਕਾਰ(ਦਸ਼ਮਲਵ), hvx_vec_gelu_f32(vsrc[i]));
    }
}

// GELU_QUICK uses x * sigmoid(1.702 * x).
ਸਥਿਰ __attribute__((noinline)) HVX_ਸਦਿਸ਼ hvx_vec_gelu_ਤੇਜ਼_f32(HVX_ਸਦਿਸ਼ x) {
    ਅਚਲ HVX_ਸਦਿਸ਼ one     = hvx_vec_splat_f32(1.0f);
    ਅਚਲ HVX_ਸਦਿਸ਼ max_exp = hvx_vec_splat_f32(87.0f);
    ਅਚਲ HVX_ਸਦਿਸ਼ min_exp = hvx_vec_splat_f32(-87.0f);
    ਅਚਲ HVX_ਸਦਿਸ਼ scaled  = hvx_vec_mul_f32_f32(x, hvx_vec_splat_f32(1.702f));
    ਅਚਲ HVX_ਸਦਿਸ਼ sigmoid = hvx_vec_fast_sigmoid_f32_guard(scaled, one, max_exp, min_exp);

    ਵਾਪਸ hvx_vec_mul_f32_f32(x, sigmoid);
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ hvx_gelu_ਤੇਜ਼_f32_aa(uint8_t * ਸੀਮਿਤ dst, ਅਚਲ uint8_t * ਸੀਮਿਤ src, uint32_t n) {
    ਪ੍ਰਤਿਜ੍ਞਾ((ਬਿਨਾਨਿਸ਼ਾਨੀ ਲੰਮਾ) dst % 128 == 0);
    ਪ੍ਰਤਿਜ੍ਞਾ((ਬਿਨਾਨਿਸ਼ਾਨੀ ਲੰਮਾ) src % 128 == 0);

    HVX_ਸਦਿਸ਼ * ਸੀਮਿਤ vdst = (HVX_ਸਦਿਸ਼ *) dst;
    HVX_ਸਦਿਸ਼ * ਸੀਮਿਤ vsrc = (HVX_ਸਦਿਸ਼ *) src;

    ਅਚਲ uint32_t nvec = n / VLEN_FP32;
    ਅਚਲ uint32_t nloe = n % VLEN_FP32;

    uint32_t i = 0;
    _Pragma("unroll(4)")
    ਲਈ (; i < nvec; i++) {
        vdst[i] = hvx_vec_gelu_ਤੇਜ਼_f32(vsrc[i]);
    }
    ਜੇ (nloe) {
        hvx_vec_store_a(&vdst[i], nloe * ਆਕਾਰ(ਦਸ਼ਮਲਵ), hvx_vec_gelu_ਤੇਜ਼_f32(vsrc[i]));
    }
}

ਸਥਿਰ ਖਾਲੀ gelu_f32(ਅਚਲ ਖਾਲੀ * ਸੀਮਿਤ src,
                     ਖਾਲੀ * ਸੀਮਿਤ dst,
                     ਅਚਲ uint32_t ਅੰਕ_ਪੰਕਤੀਆਂ_2,
                     ਅਚਲ ਬਣਤਰ htp_unary_context * uctx) {
    htp_unary_op_preamble;

    ਲਈ (uint32_t ir = 0; ir < ਅੰਕ_ਪੰਕਤੀਆਂ_2; ir++) {
        ਅਚਲ uint8_t * ਸੀਮਿਤ src_local = (ਅਚਲ uint8_t *)src + (ir * src0_ਪੰਕਤੀ_ਮਾਪ_aligned);
        uint8_t * ਸੀਮਿਤ dst_local       = (uint8_t *)dst + (ir * dst_ਪੰਕਤੀ_ਮਾਪ_aligned);

        hvx_gelu_f32_aa(dst_local, src_local, ne0);
    }
}

ਸਥਿਰ ਖਾਲੀ gelu_ਤੇਜ਼_f32(ਅਚਲ ਖਾਲੀ * ਸੀਮਿਤ src,
                           ਖਾਲੀ * ਸੀਮਿਤ dst,
                           ਅਚਲ uint32_t ਅੰਕ_ਪੰਕਤੀਆਂ_2,
                           ਅਚਲ ਬਣਤਰ htp_unary_context * uctx) {
    htp_unary_op_preamble;

    ਲਈ (uint32_t ir = 0; ir < ਅੰਕ_ਪੰਕਤੀਆਂ_2; ir++) {
        ਅਚਲ uint8_t * ਸੀਮਿਤ src_local = (ਅਚਲ uint8_t *)src + (ir * src0_ਪੰਕਤੀ_ਮਾਪ_aligned);
        uint8_t * ਸੀਮਿਤ dst_local       = (uint8_t *)dst + (ir * dst_ਪੰਕਤੀ_ਮਾਪ_aligned);

        hvx_gelu_ਤੇਜ਼_f32_aa(dst_local, src_local, ne0);
    }
}

ਸਥਿਰ ਖਾਲੀ gelu_erf_f32(ਅਚਲ ਖਾਲੀ * ਸੀਮਿਤ src,
                         ਖਾਲੀ * ਸੀਮਿਤ dst,
                         ਅਚਲ uint32_t ਅੰਕ_ਪੰਕਤੀਆਂ_2,
                         ਅਚਲ ਬਣਤਰ htp_unary_context * uctx) {
    htp_unary_op_preamble;

    ਲਈ (uint32_t ir = 0; ir < ਅੰਕ_ਪੰਕਤੀਆਂ_2; ir++) {
        ਅਚਲ uint8_t * ਸੀਮਿਤ src_local = (ਅਚਲ uint8_t *) src + (ir * src0_ਪੰਕਤੀ_ਮਾਪ_aligned);
        uint8_t * ਸੀਮਿਤ dst_local       = (uint8_t *) dst + (ir * dst_ਪੰਕਤੀ_ਮਾਪ_aligned);

        hvx_gelu_erf_f32_aa(dst_local, src_local, ne0);
    }
}

ਸਥਿਰ ਖਾਲੀ tri_f32(ਅਚਲ ਖਾਲੀ * ਸੀਮਿਤ src,
                    ਖਾਲੀ * ਸੀਮਿਤ dst,
                    ਅਚਲ uint32_t ਅੰਕ_ਪੰਕਤੀਆਂ_2,
                    ਅਚਲ uint32_t ir,
                    ਅਚਲ ਬਣਤਰ htp_unary_context * uctx) {
    htp_unary_op_preamble;
    ਅਚਲ int32_t ttype = op_params[0];
    ਅਚਲ HVX_ਸਦਿਸ਼ zero = hvx_vec_splat_f32(0.0f);
    ਅਚਲ uint32_t nvec  = ne0 / VLEN_FP32;
    ਅਚਲ uint32_t nloe  = ne0 % VLEN_FP32;

    ਅਚਲ uint32_t ne01 = uctx->octx->src[0]->ne[1];

    ਲਈ (uint32_t b = 0; b < ਅੰਕ_ਪੰਕਤੀਆਂ_2; b++) {
        ਅਚਲ uint32_t abs_ਪੰਕਤੀ = ir + b;
        ਅਚਲ uint32_t i01     = abs_ਪੰਕਤੀ % ne01;

        ਅਚਲ HVX_ਸਦਿਸ਼ * ਸੀਮਿਤ v_src = (ਅਚਲ HVX_ਸਦਿਸ਼ *) ((ਅਚਲ uint8_t *) src + b * src0_ਪੰਕਤੀ_ਮਾਪ_aligned);
        HVX_ਸਦਿਸ਼ * ਸੀਮਿਤ v_dst       = (HVX_ਸਦਿਸ਼ *) ((uint8_t *) dst + b * dst_ਪੰਕਤੀ_ਮਾਪ_aligned);

        uint32_t boundary;
        ਪੂਰਨ      keep_ਖੱਬਾ;
        ਚੋਣ (ttype) {
            ਹਾਲਤ 0: boundary = i01;     keep_ਖੱਬਾ = 0; ਰੋਕੋ;  // keep col >= row
            ਹਾਲਤ 1: boundary = i01 + 1; keep_ਖੱਬਾ = 0; ਰੋਕੋ;  // keep col > row
            ਹਾਲਤ 2: boundary = i01 + 1; keep_ਖੱਬਾ = 1; ਰੋਕੋ;  // keep col <= row
            ਹਾਲਤ 3: boundary = i01;     keep_ਖੱਬਾ = 1; ਰੋਕੋ;  // keep col < row
            ਮੂਲ: boundary = 0; keep_ਖੱਬਾ = 0; ਰੋਕੋ;
        }
        ਜੇ (boundary > ne0) boundary = ne0;

        // Full HVX vectors - each starts at a 128-byte aligned offset
        ਲਈ (uint32_t i = 0; i < nvec; i++) {
            ਅਚਲ uint32_t vec_ਸ਼ੁਰੂ = i * VLEN_FP32;
            ਅਚਲ uint32_t vec_ਸਿਰਾ   = vec_ਸ਼ੁਰੂ + VLEN_FP32;
            ਜੇ (keep_ਖੱਬਾ) {
                ਜੇ (vec_ਸਿਰਾ <= boundary) {
                    v_dst[i] = v_src[i];
                } ਨਹੀਂਤਾਂ ਜੇ (vec_ਸ਼ੁਰੂ >= boundary) {
                    v_dst[i] = zero;
                } ਨਹੀਂਤਾਂ {
                    HVX_ਸਦਿਸ਼_Pred ਨਕਾਬ = Q6_Q_vsetq_R((boundary - vec_ਸ਼ੁਰੂ) * ਆਕਾਰ(ਦਸ਼ਮਲਵ));
                    v_dst[i]            = Q6_V_vmux_QVV(ਨਕਾਬ, v_src[i], zero);
                }
            } ਨਹੀਂਤਾਂ {
                ਜੇ (vec_ਸਿਰਾ <= boundary) {
                    v_dst[i] = zero;
                } ਨਹੀਂਤਾਂ ਜੇ (vec_ਸ਼ੁਰੂ >= boundary) {
                    v_dst[i] = v_src[i];
                } ਨਹੀਂਤਾਂ {
                    HVX_ਸਦਿਸ਼_Pred ਨਕਾਬ = Q6_Q_vsetq_R((boundary - vec_ਸ਼ੁਰੂ) * ਆਕਾਰ(ਦਸ਼ਮਲਵ));
                    v_dst[i]            = Q6_V_vmux_QVV(ਨਕਾਬ, zero, v_src[i]);
                }
            }
        }

        // Tail elements (row_elems not a multiple of VLEN_FP32)
        ਜੇ (nloe > 0) {
            ਅਚਲ uint32_t abs_ਸ਼ੁਰੂ = nvec * VLEN_FP32;
            ਅਚਲ uint32_t abs_ਸਿਰਾ   = abs_ਸ਼ੁਰੂ + nloe;
            HVX_ਸਦਿਸ਼     ਪੂਛ_val;
            ਜੇ (keep_ਖੱਬਾ) {
                ਜੇ (abs_ਸਿਰਾ <= boundary) {
                    ਪੂਛ_val = v_src[nvec];
                } ਨਹੀਂਤਾਂ ਜੇ (abs_ਸ਼ੁਰੂ >= boundary) {
                    ਪੂਛ_val = zero;
                } ਨਹੀਂਤਾਂ {
                    HVX_ਸਦਿਸ਼_Pred ਨਕਾਬ = Q6_Q_vsetq_R((boundary - abs_ਸ਼ੁਰੂ) * ਆਕਾਰ(ਦਸ਼ਮਲਵ));
                    ਪੂਛ_val            = Q6_V_vmux_QVV(ਨਕਾਬ, v_src[nvec], zero);
                }
            } ਨਹੀਂਤਾਂ {
                ਜੇ (abs_ਸਿਰਾ <= boundary) {
                    ਪੂਛ_val = zero;
                } ਨਹੀਂਤਾਂ ਜੇ (abs_ਸ਼ੁਰੂ >= boundary) {
                    ਪੂਛ_val = v_src[nvec];
                } ਨਹੀਂਤਾਂ {
                    HVX_ਸਦਿਸ਼_Pred ਨਕਾਬ = Q6_Q_vsetq_R((boundary - abs_ਸ਼ੁਰੂ) * ਆਕਾਰ(ਦਸ਼ਮਲਵ));
                    ਪੂਛ_val            = Q6_V_vmux_QVV(ਨਕਾਬ, zero, v_src[nvec]);
                }
            }
            hvx_vec_store_a(&v_dst[nvec], nloe * ਆਕਾਰ(ਦਸ਼ਮਲਵ), ਪੂਛ_val);
        }
    }
}

// softplus(x) = log(1 + exp(x))
// Match CPU reference: return x for x > 20 to avoid overflow.
ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ hvx_softplus_f32_aa(uint8_t * ਸੀਮਿਤ dst, ਅਚਲ uint8_t * ਸੀਮਿਤ src, uint32_t n) {
    ਅਚਲ HVX_ਸਦਿਸ਼ * ਸੀਮਿਤ v_src = (ਅਚਲ HVX_ਸਦਿਸ਼ *) src;
    HVX_ਸਦਿਸ਼ * ਸੀਮਿਤ v_dst       = (HVX_ਸਦਿਸ਼ *) dst;

    ਅਚਲ HVX_ਸਦਿਸ਼ twenty = hvx_vec_splat_f32(20.0f);
    ਅਚਲ HVX_ਸਦਿਸ਼ one    = hvx_vec_splat_f32(1.0f);

    ਅਚਲ uint32_t nvec = n / VLEN_FP32;
    ਅਚਲ uint32_t nloe = n % VLEN_FP32;

    #ਹਦਾਇਤ unroll(4)
    ਲਈ (uint32_t i = 0; i < nvec; i++) {
        HVX_ਸਦਿਸ਼ x              = v_src[i];
        HVX_ਸਦਿਸ਼_Pred pred_gt20  = Q6_Q_vcmp_gt_VsfVsf(x, twenty);
        HVX_ਸਦਿਸ਼ expx           = hvx_vec_exp_f32(x);
        HVX_ਸਦਿਸ਼ logv           = hvx_vec_log_f32(hvx_vec_ਜੋੜੋ_f32_f32(one, expx));
        v_dst[i]                  = Q6_V_vmux_QVV(pred_gt20, x, logv);
    }

    ਜੇ (nloe > 0) {
        HVX_ਸਦਿਸ਼ x             = v_src[nvec];
        HVX_ਸਦਿਸ਼_Pred pred_gt20 = Q6_Q_vcmp_gt_VsfVsf(x, twenty);
        HVX_ਸਦਿਸ਼ expx          = hvx_vec_exp_f32(x);
        HVX_ਸਦਿਸ਼ logv          = hvx_vec_log_f32(hvx_vec_ਜੋੜੋ_f32_f32(one, expx));
        HVX_ਸਦਿਸ਼ sp            = Q6_V_vmux_QVV(pred_gt20, x, logv);
        hvx_vec_store_a((ਖਾਲੀ *) &v_dst[nvec], nloe * ਆਕਾਰ(ਦਸ਼ਮਲਵ), sp);
    }
}

ਸਥਿਰ ਖਾਲੀ softplus_f32(ਅਚਲ ਖਾਲੀ * ਸੀਮਿਤ src,
                         ਖਾਲੀ * ਸੀਮਿਤ dst,
                         ਅਚਲ uint32_t ਅੰਕ_ਪੰਕਤੀਆਂ_2,
                         ਅਚਲ ਬਣਤਰ htp_unary_context * uctx) {
    htp_unary_op_preamble;

    ਲਈ (uint32_t ir = 0; ir < ਅੰਕ_ਪੰਕਤੀਆਂ_2; ir++) {
        ਅਚਲ uint8_t * ਸੀਮਿਤ src_local = (ਅਚਲ uint8_t *)src + (ir * src0_ਪੰਕਤੀ_ਮਾਪ_aligned);
        uint8_t * ਸੀਮਿਤ dst_local       = (uint8_t *)dst + (ir * dst_ਪੰਕਤੀ_ਮਾਪ_aligned);

        hvx_softplus_f32_aa(dst_local, src_local, ne0);
    }
}

ਸਥਿਰ ਖਾਲੀ l2_norm_f32(ਅਚਲ ਖਾਲੀ * ਸੀਮਿਤ src,
                        ਖਾਲੀ * ਸੀਮਿਤ dst,
                        ਅਚਲ uint32_t ਅੰਕ_ਪੰਕਤੀਆਂ_2,
                        ਅਚਲ ਬਣਤਰ htp_unary_context * uctx) {
    htp_unary_op_preamble;
    ਦਸ਼ਮਲਵ epsilon = 0.f;
    ਸਮ੍ਰਿਤੀ_ਨਕਲ(&epsilon, op_params, ਆਕਾਰ(ਦਸ਼ਮਲਵ));

    ਲਈ (uint32_t ir = 0; ir < ਅੰਕ_ਪੰਕਤੀਆਂ_2; ir++) {
        ਅਚਲ uint8_t * ਸੀਮਿਤ src_f = (ਅਚਲ uint8_t *)src + (ir * src0_ਪੰਕਤੀ_ਮਾਪ_aligned);
        uint8_t * ਸੀਮਿਤ dst_f       = (uint8_t *)dst + (ir * dst_ਪੰਕਤੀ_ਮਾਪ_aligned);

        hvx_fast_l2_norm_f32((ਅਚਲ uint8_t *)src_f, (uint8_t *)dst_f, ne0, epsilon);
    }
}

ਸਥਿਰ ਸਤਰਅੰਦਰ HVX_ਸਦਿਸ਼_Pair hvx_fwht_stage_pair(HVX_ਸਦਿਸ਼ v0, HVX_ਸਦਿਸ਼ v1, int32_t b) {
    HVX_ਸਦਿਸ਼_Pair p = Q6_W_vshuff_VVR(v1, v0, b);
    HVX_ਸਦਿਸ਼ s = Q6_Vqf32_vadd_Vqf32Vqf32(Q6_V_lo_W(p), Q6_V_hi_W(p));
    HVX_ਸਦਿਸ਼ d = Q6_Vqf32_vsub_Vqf32Vqf32(Q6_V_lo_W(p), Q6_V_hi_W(p));
    ਵਾਪਸ Q6_W_vshuff_VVR(d, s, b);
}

ਸਥਿਰ ਸਤਰਅੰਦਰ HVX_ਸਦਿਸ਼_Pair hvx_fwht32_pair(HVX_ਸਦਿਸ਼ v0_sf, HVX_ਸਦਿਸ਼ v1_sf, HVX_ਸਦਿਸ਼ v_scale) {
    HVX_ਸਦਿਸ਼ v0 = Q6_Vqf32_vmpy_VsfVsf(v0_sf, v_scale);
    HVX_ਸਦਿਸ਼ v1 = Q6_Vqf32_vmpy_VsfVsf(v1_sf, v_scale);
    HVX_ਸਦਿਸ਼_Pair p;
    p = hvx_fwht_stage_pair(v0, v1, 4);
    p = hvx_fwht_stage_pair(Q6_V_lo_W(p), Q6_V_hi_W(p), 8);
    p = hvx_fwht_stage_pair(Q6_V_lo_W(p), Q6_V_hi_W(p), 16);
    p = hvx_fwht_stage_pair(Q6_V_lo_W(p), Q6_V_hi_W(p), 32);
    p = hvx_fwht_stage_pair(Q6_V_lo_W(p), Q6_V_hi_W(p), 64);
    ਵਾਪਸ p;
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ hvx_fwht_store32(HVX_ਸਦਿਸ਼ * ਸੀਮਿਤ v_dst, HVX_ਸਦਿਸ਼_Pair p) {
    v_dst[0] = Q6_Vsf_equals_Vqf32(Q6_V_lo_W(p));
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ hvx_fwht_store64(HVX_ਸਦਿਸ਼ * ਸੀਮਿਤ v_dst, HVX_ਸਦਿਸ਼_Pair p) {
    HVX_ਸਦਿਸ਼ u = Q6_V_lo_W(p);
    HVX_ਸਦਿਸ਼ v = Q6_V_hi_W(p);
    v_dst[0] = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vqf32(u, v));
    v_dst[1] = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vsub_Vqf32Vqf32(u, v));
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ hvx_fwht_ਪੂਛ_f32(HVX_ਸਦਿਸ਼ * ਸੀਮਿਤ v_dst, uint32_t nvec) {
    uint32_t len_vec = 2;
    ਲਈ (; len_vec * 2 < nvec; len_vec <<= 2) {
        ਅਚਲ uint32_t ਕਦਮ = len_vec * 4;
        ਲਈ (uint32_t i = 0; i < nvec; i += ਕਦਮ) {
            ਲਈ (uint32_t j = 0; j < len_vec; j++) {
                HVX_ਸਦਿਸ਼ a0 = v_dst[i + j];
                HVX_ਸਦਿਸ਼ a1 = v_dst[i + j + len_vec];
                HVX_ਸਦਿਸ਼ a2 = v_dst[i + j + 2 * len_vec];
                HVX_ਸਦਿਸ਼ a3 = v_dst[i + j + 3 * len_vec];

                HVX_ਸਦਿਸ਼ t0 = Q6_Vqf32_vadd_VsfVsf(a0, a1);
                HVX_ਸਦਿਸ਼ t1 = Q6_Vqf32_vsub_VsfVsf(a0, a1);
                HVX_ਸਦਿਸ਼ t2 = Q6_Vqf32_vadd_VsfVsf(a2, a3);
                HVX_ਸਦਿਸ਼ t3 = Q6_Vqf32_vsub_VsfVsf(a2, a3);

                v_dst[i + j]               = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vqf32(t0, t2));
                v_dst[i + j + len_vec]     = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vqf32(t1, t3));
                v_dst[i + j + 2 * len_vec] = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vsub_Vqf32Vqf32(t0, t2));
                v_dst[i + j + 3 * len_vec] = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vsub_Vqf32Vqf32(t1, t3));
            }
        }
    }

    ਜੇ (len_vec < nvec) {
        ਲਈ (uint32_t i = 0; i < nvec; i += 2 * len_vec) {
            ਲਈ (uint32_t j = 0; j < len_vec; j++) {
                HVX_ਸਦਿਸ਼ u = v_dst[i + j];
                HVX_ਸਦਿਸ਼ v = v_dst[i + len_vec + j];
                v_dst[i + j]           = hvx_vec_ਜੋੜੋ_f32_f32(u, v);
                v_dst[i + len_vec + j] = hvx_vec_sub_f32_f32(u, v);
            }
        }
    }
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ hvx_fwht_ਪੰਕਤੀ_f32(ਅਚਲ uint8_t * ਸੀਮਿਤ src, uint8_t * ਸੀਮਿਤ dst, uint32_t n, HVX_ਸਦਿਸ਼ v_scale) {
    ਅਚਲ HVX_ਸਦਿਸ਼ * ਸੀਮਿਤ v_src = (ਅਚਲ HVX_ਸਦਿਸ਼ *) src;
    HVX_ਸਦਿਸ਼ * ਸੀਮਿਤ v_dst       = (HVX_ਸਦਿਸ਼ *) dst;
    ਅਚਲ uint32_t nvec               = n / VLEN_FP32;

    ਜੇ (nvec == 1) {
        HVX_ਸਦਿਸ਼_Pair p = hvx_fwht32_pair(v_src[0], v_src[0], v_scale);
        hvx_fwht_store32(v_dst, p);
        ਵਾਪਸ;
    }

    ਲਈ (uint32_t i = 0; i < nvec; i += 2) {
        HVX_ਸਦਿਸ਼_Pair p = hvx_fwht32_pair(v_src[i], v_src[i + 1], v_scale);
        hvx_fwht_store64(&v_dst[i], p);
    }

    hvx_fwht_ਪੂਛ_f32(v_dst, nvec);
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ hvx_fwht_ਪੰਕਤੀ_f16(ਅਚਲ uint8_t * ਸੀਮਿਤ src, uint8_t * ਸੀਮਿਤ dst, uint32_t n, HVX_ਸਦਿਸ਼ v_scale) {
    ਅਚਲ HVX_ਸਦਿਸ਼ * ਸੀਮਿਤ v_src = (ਅਚਲ HVX_ਸਦਿਸ਼ *) src;
    HVX_ਸਦਿਸ਼ * ਸੀਮਿਤ v_dst       = (HVX_ਸਦਿਸ਼ *) dst;
    ਅਚਲ uint32_t nvec               = n / VLEN_FP32;

    ਜੇ (nvec == 1) {
        HVX_ਸਦਿਸ਼ v0 = Q6_V_lo_W(hvx_vec_f16_ਮੰਜ਼ਿਲ_f32(v_src[0]));
        HVX_ਸਦਿਸ਼_Pair p = hvx_fwht32_pair(v0, v0, v_scale);
        hvx_fwht_store32(v_dst, p);
        ਵਾਪਸ;
    }

    ਅਚਲ uint32_t nvec_f16 = n / VLEN_FP16;
    ਲਈ (uint32_t i = 0; i < nvec_f16; i++) {
        HVX_ਸਦਿਸ਼_Pair f32p = hvx_vec_f16_ਮੰਜ਼ਿਲ_f32(v_src[i]);
        HVX_ਸਦਿਸ਼_Pair p    = hvx_fwht32_pair(Q6_V_lo_W(f32p), Q6_V_hi_W(f32p), v_scale);
        hvx_fwht_store64(&v_dst[2 * i], p);
    }

    hvx_fwht_ਪੂਛ_f32(v_dst, nvec);
}

ਸਥਿਰ ਖਾਲੀ hadamard_f32(ਅਚਲ ਖਾਲੀ * ਸੀਮਿਤ src,
                         ਖਾਲੀ * ਸੀਮਿਤ dst,
                         ਅਚਲ uint32_t ਅੰਕ_ਪੰਕਤੀਆਂ_2,
                         ਅਚਲ ਬਣਤਰ htp_unary_context * uctx) {
    htp_unary_op_preamble;
    ਅਚਲ HVX_ਸਦਿਸ਼ v_scale = hvx_vec_splat_f32(1.0f / sqrtf((ਦਸ਼ਮਲਵ) ne0));

    ਲਈ (uint32_t ir = 0; ir < ਅੰਕ_ਪੰਕਤੀਆਂ_2; ir++) {
        ਅਚਲ uint8_t * ਸੀਮਿਤ src_ਪੰਕਤੀ = (ਅਚਲ uint8_t *) src + (ir * src0_ਪੰਕਤੀ_ਮਾਪ_aligned);
        uint8_t * ਸੀਮਿਤ dst_ਪੰਕਤੀ       = (uint8_t *) dst + (ir * dst_ਪੰਕਤੀ_ਮਾਪ_aligned);

        hvx_fwht_ਪੰਕਤੀ_f32(src_ਪੰਕਤੀ, dst_ਪੰਕਤੀ, ne0, v_scale);
    }
}

ਸਥਿਰ ਖਾਲੀ hadamard_f16(ਅਚਲ ਖਾਲੀ * ਸੀਮਿਤ src,
                         ਖਾਲੀ * ਸੀਮਿਤ dst,
                         ਅਚਲ uint32_t ਅੰਕ_ਪੰਕਤੀਆਂ_2,
                         ਅਚਲ ਬਣਤਰ htp_unary_context * uctx) {
    htp_unary_op_preamble;
    ਅਚਲ HVX_ਸਦਿਸ਼ v_scale = hvx_vec_splat_f32(1.0f / sqrtf((ਦਸ਼ਮਲਵ) ne0));

    ਲਈ (uint32_t ir = 0; ir < ਅੰਕ_ਪੰਕਤੀਆਂ_2; ir++) {
        ਅਚਲ uint8_t * ਸੀਮਿਤ src_ਪੰਕਤੀ = (ਅਚਲ uint8_t *) src + (ir * src0_ਪੰਕਤੀ_ਮਾਪ_aligned);
        uint8_t * ਸੀਮਿਤ dst_ਪੰਕਤੀ       = (uint8_t *) dst + (ir * dst_ਪੰਕਤੀ_ਮਾਪ_aligned);

        hvx_fwht_ਪੰਕਤੀ_f16(src_ਪੰਕਤੀ, dst_ਪੰਕਤੀ, ne0, v_scale);
    }
}

ਸਥਿਰ ਖਾਲੀ tanh_f32(ਅਚਲ ਖਾਲੀ * ਸੀਮਿਤ src,
                     ਖਾਲੀ * ਸੀਮਿਤ dst,
                     ਅਚਲ uint32_t ਅੰਕ_ਪੰਕਤੀਆਂ_2,
                     ਅਚਲ ਬਣਤਰ htp_unary_context * uctx) {
    htp_unary_op_preamble;

    ਲਈ (uint32_t ir = 0; ir < ਅੰਕ_ਪੰਕਤੀਆਂ_2; ir++) {
        ਅਚਲ uint8_t * ਸੀਮਿਤ src_local = (ਅਚਲ uint8_t *)src + (ir * src0_ਪੰਕਤੀ_ਮਾਪ_aligned);
        uint8_t * ਸੀਮਿਤ dst_local       = (uint8_t *)dst + (ir * dst_ਪੰਕਤੀ_ਮਾਪ_aligned);

        hvx_tanh_f32_aa(dst_local, src_local, ne0);
    }
}

ਸਥਿਰ ਖਾਲੀ abs_f32(ਅਚਲ ਖਾਲੀ * ਸੀਮਿਤ src,
                    ਖਾਲੀ * ਸੀਮਿਤ dst,
                    ਅਚਲ uint32_t ਅੰਕ_ਪੰਕਤੀਆਂ_2,
                    ਅਚਲ ਬਣਤਰ htp_unary_context * uctx) {
    htp_unary_op_preamble;

    ਲਈ (uint32_t ir = 0; ir < ਅੰਕ_ਪੰਕਤੀਆਂ_2; ir++) {
        ਅਚਲ uint8_t * ਸੀਮਿਤ src_local = (ਅਚਲ uint8_t *)src + (ir * src0_ਪੰਕਤੀ_ਮਾਪ_aligned);
        uint8_t * ਸੀਮਿਤ dst_local       = (uint8_t *)dst + (ir * dst_ਪੰਕਤੀ_ਮਾਪ_aligned);

        hvx_abs_f32_aa(dst_local, src_local, ne0);
    }
}

ਸਥਿਰ ਖਾਲੀ relu_f32(ਅਚਲ ਖਾਲੀ * ਸੀਮਿਤ src,
                     ਖਾਲੀ * ਸੀਮਿਤ dst,
                     ਅਚਲ uint32_t ਅੰਕ_ਪੰਕਤੀਆਂ_2,
                     ਅਚਲ ਬਣਤਰ htp_unary_context * uctx) {
    htp_unary_op_preamble;

    ਲਈ (uint32_t ir = 0; ir < ਅੰਕ_ਪੰਕਤੀਆਂ_2; ir++) {
        ਅਚਲ uint8_t * ਸੀਮਿਤ src_local = (ਅਚਲ uint8_t *)src + (ir * src0_ਪੰਕਤੀ_ਮਾਪ_aligned);
        uint8_t * ਸੀਮਿਤ dst_local       = (uint8_t *)dst + (ir * dst_ਪੰਕਤੀ_ਮਾਪ_aligned);

        hvx_max_scalar_f32(dst_local, src_local, 0.0f, ne0);
    }
}

ਸਥਿਰ ਖਾਲੀ ਕਦਮ_f32(ਅਚਲ ਖਾਲੀ * ਸੀਮਿਤ src,
                     ਖਾਲੀ * ਸੀਮਿਤ dst,
                     ਅਚਲ uint32_t ਅੰਕ_ਪੰਕਤੀਆਂ_2,
                     ਅਚਲ ਬਣਤਰ htp_unary_context * uctx) {
    htp_unary_op_preamble;

    ਲਈ (uint32_t ir = 0; ir < ਅੰਕ_ਪੰਕਤੀਆਂ_2; ir++) {
        ਅਚਲ uint8_t * ਸੀਮਿਤ src_local = (ਅਚਲ uint8_t *)src + (ir * src0_ਪੰਕਤੀ_ਮਾਪ_aligned);
        uint8_t * ਸੀਮਿਤ dst_local       = (uint8_t *)dst + (ir * dst_ਪੰਕਤੀ_ਮਾਪ_aligned);

        hvx_ਕਦਮ_f32_aa(dst_local, src_local, ne0);
    }
}

ਸਥਿਰ ਖਾਲੀ log_f32(ਅਚਲ ਖਾਲੀ * ਸੀਮਿਤ src,
                    ਖਾਲੀ * ਸੀਮਿਤ dst,
                    ਅਚਲ uint32_t ਅੰਕ_ਪੰਕਤੀਆਂ_2,
                    ਅਚਲ ਬਣਤਰ htp_unary_context * uctx) {
    htp_unary_op_preamble;

    ਲਈ (uint32_t ir = 0; ir < ਅੰਕ_ਪੰਕਤੀਆਂ_2; ir++) {
        ਅਚਲ uint8_t * ਸੀਮਿਤ src_local = (ਅਚਲ uint8_t *)src + (ir * src0_ਪੰਕਤੀ_ਮਾਪ_aligned);
        uint8_t * ਸੀਮਿਤ dst_local       = (uint8_t *)dst + (ir * dst_ਪੰਕਤੀ_ਮਾਪ_aligned);

        hvx_log_f32_aa(dst_local, src_local, ne0);
    }
}

#// Pointwise unary ops on one column tile in VTCM.
ਸਥਿਰ ਖਾਲੀ tile_scale_f32(ਖਾਲੀ * ਸੀਮਿਤ dst, ਅਚਲ ਖਾਲੀ * ਸੀਮਿਤ src, uint32_t tw, ਅਚਲ ਬਣਤਰ htp_unary_context * uctx) {
    ਦਸ਼ਮਲਵ scale = 0.f;
    ਦਸ਼ਮਲਵ bias  = 0.f;
    ਸਮ੍ਰਿਤੀ_ਨਕਲ(&scale, &uctx->octx->op_params[0], ਆਕਾਰ(ਦਸ਼ਮਲਵ));
    ਸਮ੍ਰਿਤੀ_ਨਕਲ(&bias,  &uctx->octx->op_params[1], ਆਕਾਰ(ਦਸ਼ਮਲਵ));
    hvx_scale_ਵਿੱਥ_f32_aa((uint8_t *) dst, (ਅਚਲ uint8_t *) src, tw, scale, bias);
}

ਸਥਿਰ ਖਾਲੀ tile_clamp_f32(ਖਾਲੀ * ਸੀਮਿਤ dst, ਅਚਲ ਖਾਲੀ * ਸੀਮਿਤ src, uint32_t tw, ਅਚਲ ਬਣਤਰ htp_unary_context * uctx) {
    ਦਸ਼ਮਲਵ min = 0.f;
    ਦਸ਼ਮਲਵ max = 0.f;
    ਸਮ੍ਰਿਤੀ_ਨਕਲ(&min, &uctx->octx->op_params[0], ਆਕਾਰ(ਦਸ਼ਮਲਵ));
    ਸਮ੍ਰਿਤੀ_ਨਕਲ(&max, &uctx->octx->op_params[1], ਆਕਾਰ(ਦਸ਼ਮਲਵ));
    hvx_clamp_scalar_f32((uint8_t *) dst, (ਅਚਲ uint8_t *) src, min, max, tw);
}

ਸਥਿਰ ਖਾਲੀ tile_leaky_relu_f32(ਖਾਲੀ * ਸੀਮਿਤ dst, ਅਚਲ ਖਾਲੀ * ਸੀਮਿਤ src, uint32_t tw, ਅਚਲ ਬਣਤਰ htp_unary_context * uctx) {
    ਦਸ਼ਮਲਵ negative_ਢਲਾਨ = 0.f;
    ਸਮ੍ਰਿਤੀ_ਨਕਲ(&negative_ਢਲਾਨ, &uctx->octx->op_params[0], ਆਕਾਰ(ਦਸ਼ਮਲਵ));
    hvx_leaky_relu_scalar_f32((uint8_t *) dst, (ਅਚਲ uint8_t *) src, negative_ਢਲਾਨ, tw);
}

ਸਥਿਰ ਖਾਲੀ tile_sqr_f32(ਖਾਲੀ * ਸੀਮਿਤ dst, ਅਚਲ ਖਾਲੀ * ਸੀਮਿਤ src, uint32_t tw, ਅਚਲ ਬਣਤਰ htp_unary_context * uctx) {
    (ਖਾਲੀ) uctx;
    hvx_sqr_f32_aa((uint8_t *) dst, (ਅਚਲ uint8_t *) src, tw);
}

ਸਥਿਰ ਖਾਲੀ tile_sqrt_f32(ਖਾਲੀ * ਸੀਮਿਤ dst, ਅਚਲ ਖਾਲੀ * ਸੀਮਿਤ src, uint32_t tw, ਅਚਲ ਬਣਤਰ htp_unary_context * uctx) {
    (ਖਾਲੀ) uctx;
    hvx_sqrt_f32_aa((uint8_t *) dst, (ਅਚਲ uint8_t *) src, tw);
}

ਸਥਿਰ ਖਾਲੀ tile_neg_f32(ਖਾਲੀ * ਸੀਮਿਤ dst, ਅਚਲ ਖਾਲੀ * ਸੀਮਿਤ src, uint32_t tw, ਅਚਲ ਬਣਤਰ htp_unary_context * uctx) {
    (ਖਾਲੀ) uctx;
    hvx_scale_f32_aa((uint8_t *) dst, (ਅਚਲ uint8_t *) src, tw, -1.0f);
}

ਸਥਿਰ ਖਾਲੀ tile_exp_f32(ਖਾਲੀ * ਸੀਮਿਤ dst, ਅਚਲ ਖਾਲੀ * ਸੀਮਿਤ src, uint32_t tw, ਅਚਲ ਬਣਤਰ htp_unary_context * uctx) {
    (ਖਾਲੀ) uctx;
    hvx_exp_f32((uint8_t *) dst, (ਅਚਲ uint8_t *) src, tw, false);
}

ਸਥਿਰ ਖਾਲੀ tile_sigmoid_f32(ਖਾਲੀ * ਸੀਮਿਤ dst, ਅਚਲ ਖਾਲੀ * ਸੀਮਿਤ src, uint32_t tw, ਅਚਲ ਬਣਤਰ htp_unary_context * uctx) {
    (ਖਾਲੀ) uctx;
    hvx_sigmoid_f32_aa((uint8_t *) dst, (ਅਚਲ uint8_t *) src, tw);
}

ਸਥਿਰ ਖਾਲੀ tile_silu_f32(ਖਾਲੀ * ਸੀਮਿਤ dst, ਅਚਲ ਖਾਲੀ * ਸੀਮਿਤ src, uint32_t tw, ਅਚਲ ਬਣਤਰ htp_unary_context * uctx) {
    (ਖਾਲੀ) uctx;
    hvx_sigmoid_f32_aa((uint8_t *) dst, (ਅਚਲ uint8_t *) src, tw);
    hvx_mul_f32_aaa((uint8_t *) dst, (ਅਚਲ uint8_t *) src, (uint8_t *) dst, tw);
}

ਸਥਿਰ ਖਾਲੀ tile_gelu_f32(ਖਾਲੀ * ਸੀਮਿਤ dst, ਅਚਲ ਖਾਲੀ * ਸੀਮਿਤ src, uint32_t tw, ਅਚਲ ਬਣਤਰ htp_unary_context * uctx) {
    (ਖਾਲੀ) uctx;
    hvx_gelu_f32_aa((uint8_t *) dst, (ਅਚਲ uint8_t *) src, tw);
}

ਸਥਿਰ ਖਾਲੀ tile_gelu_ਤੇਜ਼_f32(ਖਾਲੀ * ਸੀਮਿਤ dst, ਅਚਲ ਖਾਲੀ * ਸੀਮਿਤ src, uint32_t tw, ਅਚਲ ਬਣਤਰ htp_unary_context * uctx) {
    (ਖਾਲੀ) uctx;
    hvx_gelu_ਤੇਜ਼_f32_aa((uint8_t *) dst, (ਅਚਲ uint8_t *) src, tw);
}

ਸਥਿਰ ਖਾਲੀ tile_gelu_erf_f32(ਖਾਲੀ * ਸੀਮਿਤ dst, ਅਚਲ ਖਾਲੀ * ਸੀਮਿਤ src, uint32_t tw, ਅਚਲ ਬਣਤਰ htp_unary_context * uctx) {
    (ਖਾਲੀ) uctx;
    hvx_gelu_erf_f32_aa((uint8_t *) dst, (ਅਚਲ uint8_t *) src, tw);
}

ਸਥਿਰ ਖਾਲੀ tile_softplus_f32(ਖਾਲੀ * ਸੀਮਿਤ dst, ਅਚਲ ਖਾਲੀ * ਸੀਮਿਤ src, uint32_t tw, ਅਚਲ ਬਣਤਰ htp_unary_context * uctx) {
    (ਖਾਲੀ) uctx;
    hvx_softplus_f32_aa((uint8_t *) dst, (ਅਚਲ uint8_t *) src, tw);
}

ਸਥਿਰ ਖਾਲੀ tile_tanh_f32(ਖਾਲੀ * ਸੀਮਿਤ dst, ਅਚਲ ਖਾਲੀ * ਸੀਮਿਤ src, uint32_t tw, ਅਚਲ ਬਣਤਰ htp_unary_context * uctx) {
    (ਖਾਲੀ) uctx;
    hvx_tanh_f32_aa((uint8_t *) dst, (ਅਚਲ uint8_t *) src, tw);
}

ਸਥਿਰ ਖਾਲੀ tile_abs_f32(ਖਾਲੀ * ਸੀਮਿਤ dst, ਅਚਲ ਖਾਲੀ * ਸੀਮਿਤ src, uint32_t tw, ਅਚਲ ਬਣਤਰ htp_unary_context * uctx) {
    (ਖਾਲੀ) uctx;
    hvx_abs_f32_aa((uint8_t *) dst, (ਅਚਲ uint8_t *) src, tw);
}

ਸਥਿਰ ਖਾਲੀ tile_log_f32(ਖਾਲੀ * ਸੀਮਿਤ dst, ਅਚਲ ਖਾਲੀ * ਸੀਮਿਤ src, uint32_t tw, ਅਚਲ ਬਣਤਰ htp_unary_context * uctx) {
    (ਖਾਲੀ) uctx;
    hvx_log_f32_aa((uint8_t *) dst, (ਅਚਲ uint8_t *) src, tw);
}

ਸਥਿਰ ਖਾਲੀ tile_relu_f32(ਖਾਲੀ * ਸੀਮਿਤ dst, ਅਚਲ ਖਾਲੀ * ਸੀਮਿਤ src, uint32_t tw, ਅਚਲ ਬਣਤਰ htp_unary_context * uctx) {
    (ਖਾਲੀ) uctx;
    hvx_max_scalar_f32((uint8_t *) dst, (ਅਚਲ uint8_t *) src, 0.0f, tw);
}

ਸਥਿਰ ਖਾਲੀ tile_ਕਦਮ_f32(ਖਾਲੀ * ਸੀਮਿਤ dst, ਅਚਲ ਖਾਲੀ * ਸੀਮਿਤ src, uint32_t tw, ਅਚਲ ਬਣਤਰ htp_unary_context * uctx) {
    (ਖਾਲੀ) uctx;
    hvx_ਕਦਮ_f32_aa((uint8_t *) dst, (ਅਚਲ uint8_t *) src, tw);
}

ਸਥਿਰ ਖਾਲੀ tri_apply_tile_f32(ਅਚਲ ਖਾਲੀ * ਸੀਮਿਤ src, ਖਾਲੀ * ਸੀਮਿਤ dst,
                               uint32_t tile_elems, uint32_t ਖਾਨਾ_ਸ਼ੁਰੂ, uint32_t i01,
                               uint32_t ne0, int32_t ttype) {
    ਅਚਲ HVX_ਸਦਿਸ਼ * ਸੀਮਿਤ v_src = (ਅਚਲ HVX_ਸਦਿਸ਼ *) src;
    HVX_ਸਦਿਸ਼ * ਸੀਮਿਤ v_dst       = (HVX_ਸਦਿਸ਼ *) dst;
    ਅਚਲ HVX_ਸਦਿਸ਼ zero = hvx_vec_splat_f32(0.0f);

    uint32_t boundary;
    ਪੂਰਨ      keep_ਖੱਬਾ;
    ਚੋਣ (ttype) {
        ਹਾਲਤ 0: boundary = i01;     keep_ਖੱਬਾ = 0; ਰੋਕੋ;
        ਹਾਲਤ 1: boundary = i01 + 1; keep_ਖੱਬਾ = 0; ਰੋਕੋ;
        ਹਾਲਤ 2: boundary = i01 + 1; keep_ਖੱਬਾ = 1; ਰੋਕੋ;
        ਹਾਲਤ 3: boundary = i01;     keep_ਖੱਬਾ = 1; ਰੋਕੋ;
        ਮੂਲ: boundary = 0; keep_ਖੱਬਾ = 0; ਰੋਕੋ;
    }
    ਜੇ (boundary > ne0) boundary = ne0;

    ਅਚਲ uint32_t nvec = tile_elems / VLEN_FP32;
    ਅਚਲ uint32_t nloe = tile_elems % VLEN_FP32;

    ਲਈ (uint32_t i = 0; i < nvec; i++) {
        ਅਚਲ uint32_t abs_ਸ਼ੁਰੂ = ਖਾਨਾ_ਸ਼ੁਰੂ + i * VLEN_FP32;
        ਅਚਲ uint32_t abs_ਸਿਰਾ   = abs_ਸ਼ੁਰੂ + VLEN_FP32;
        ਜੇ (keep_ਖੱਬਾ) {
            ਜੇ (abs_ਸਿਰਾ <= boundary) {
                v_dst[i] = v_src[i];
            } ਨਹੀਂਤਾਂ ਜੇ (abs_ਸ਼ੁਰੂ >= boundary) {
                v_dst[i] = zero;
            } ਨਹੀਂਤਾਂ {
                HVX_ਸਦਿਸ਼_Pred ਨਕਾਬ = Q6_Q_vsetq_R((boundary - abs_ਸ਼ੁਰੂ) * ਆਕਾਰ(ਦਸ਼ਮਲਵ));
                v_dst[i]            = Q6_V_vmux_QVV(ਨਕਾਬ, v_src[i], zero);
            }
        } ਨਹੀਂਤਾਂ {
            ਜੇ (abs_ਸਿਰਾ <= boundary) {
                v_dst[i] = zero;
            } ਨਹੀਂਤਾਂ ਜੇ (abs_ਸ਼ੁਰੂ >= boundary) {
                v_dst[i] = v_src[i];
            } ਨਹੀਂਤਾਂ {
                HVX_ਸਦਿਸ਼_Pred ਨਕਾਬ = Q6_Q_vsetq_R((boundary - abs_ਸ਼ੁਰੂ) * ਆਕਾਰ(ਦਸ਼ਮਲਵ));
                v_dst[i]            = Q6_V_vmux_QVV(ਨਕਾਬ, zero, v_src[i]);
            }
        }
    }

    ਜੇ (nloe > 0) {
        ਅਚਲ uint32_t abs_ਸ਼ੁਰੂ = ਖਾਨਾ_ਸ਼ੁਰੂ + nvec * VLEN_FP32;
        ਅਚਲ uint32_t abs_ਸਿਰਾ   = abs_ਸ਼ੁਰੂ + nloe;
        HVX_ਸਦਿਸ਼     ਪੂਛ_val;
        ਜੇ (keep_ਖੱਬਾ) {
            ਜੇ (abs_ਸਿਰਾ <= boundary) {
                ਪੂਛ_val = v_src[nvec];
            } ਨਹੀਂਤਾਂ ਜੇ (abs_ਸ਼ੁਰੂ >= boundary) {
                ਪੂਛ_val = zero;
            } ਨਹੀਂਤਾਂ {
                HVX_ਸਦਿਸ਼_Pred ਨਕਾਬ = Q6_Q_vsetq_R((boundary - abs_ਸ਼ੁਰੂ) * ਆਕਾਰ(ਦਸ਼ਮਲਵ));
                ਪੂਛ_val            = Q6_V_vmux_QVV(ਨਕਾਬ, v_src[nvec], zero);
            }
        } ਨਹੀਂਤਾਂ {
            ਜੇ (abs_ਸਿਰਾ <= boundary) {
                ਪੂਛ_val = zero;
            } ਨਹੀਂਤਾਂ ਜੇ (abs_ਸ਼ੁਰੂ >= boundary) {
                ਪੂਛ_val = v_src[nvec];
            } ਨਹੀਂਤਾਂ {
                HVX_ਸਦਿਸ਼_Pred ਨਕਾਬ = Q6_Q_vsetq_R((boundary - abs_ਸ਼ੁਰੂ) * ਆਕਾਰ(ਦਸ਼ਮਲਵ));
                ਪੂਛ_val            = Q6_V_vmux_QVV(ਨਕਾਬ, zero, v_src[nvec]);
            }
        }
        hvx_vec_store_a(&v_dst[nvec], nloe * ਆਕਾਰ(ਦਸ਼ਮਲਵ), ਪੂਛ_val);
    }
}

// 1. Standard row-block unary task (F32 and F16).
ਸਥਿਰ ਖਾਲੀ unary_thread_ਪੰਕਤੀ_ਖੰਡ(ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ nth, ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ ith, ਖਾਲੀ * ਅੰਕੜੇ) {
    (ਖਾਲੀ) nth;
    ਅਚਲ ਬਣਤਰ htp_unary_context * uctx = (ਅਚਲ ਬਣਤਰ htp_unary_context *) ਅੰਕੜੇ;
    ਬਣਤਰ htp_ops_context * octx = uctx->octx;
    ਅਚਲ ਬਣਤਰ htp_tensor * src = octx->src[0];
    ਅਚਲ ਬਣਤਰ htp_tensor * dst = octx->dst;
    ਬਣਤਰ htp_thread_trace * tr = &octx->ctx->trace[ith];

    htp_unary_preamble;

    ਅਚਲ uint32_t src0_nrows_ਪ੍ਰਤੀ_thread = uctx->src0_nrows_ਪ੍ਰਤੀ_thread;
    ਅਚਲ size_t src0_ਅੰਕੜੇ_ਪੰਕਤੀ_ਮਾਪ = uctx->src0_ਅੰਕੜੇ_ਪੰਕਤੀ_ਮਾਪ;
    ਅਚਲ size_t dst_ਅੰਕੜੇ_ਪੰਕਤੀ_ਮਾਪ  = uctx->dst_ਅੰਕੜੇ_ਪੰਕਤੀ_ਮਾਪ;
    ਅਚਲ size_t src0_ਪੰਕਤੀ_ਮਾਪ_aligned = uctx->src0_ਪੰਕਤੀ_ਮਾਪ_aligned;
    ਅਚਲ size_t dst_ਪੰਕਤੀ_ਮਾਪ_aligned  = uctx->dst_ਪੰਕਤੀ_ਮਾਪ_aligned;

    ਅਚਲ uint32_t src0_nrows = uctx->src0_nrows;
    ਅਚਲ uint32_t src0_ਸ਼ੁਰੂ_ਪੰਕਤੀ = uctx->ਪੰਕਤੀ_ਸ਼ੁਰੂ + src0_nrows_ਪ੍ਰਤੀ_thread * ith;
    ਅਚਲ uint32_t src0_ਸਿਰਾ_ਪੰਕਤੀ   = MIN(src0_ਸ਼ੁਰੂ_ਪੰਕਤੀ + src0_nrows_ਪ੍ਰਤੀ_thread, uctx->ਪੰਕਤੀ_ਸ਼ੁਰੂ + src0_nrows);

    ਜੇ (src0_ਸ਼ੁਰੂ_ਪੰਕਤੀ >= src0_ਸਿਰਾ_ਪੰਕਤੀ) {
        ਵਾਪਸ;
    }

    ਅਚਲ dma_addr_t ਅੰਕੜੇ_src = uctx->ਅੰਕੜੇ_src0;
    ਅਚਲ dma_addr_t ਅੰਕੜੇ_dst = uctx->ਅੰਕੜੇ_dst;

    uint8_t * src0_vtcm_ਅੰਕੜੇ = uctx->vtcm_src0 + (ith * uctx->vtcm_src0_ਮਾਪ_ਪ੍ਰਤੀ_thread);
    uint8_t * dst_vtcm_ਅੰਕੜੇ  = uctx->vtcm_dst + (ith * uctx->vtcm_dst_ਮਾਪ_ਪ੍ਰਤੀ_thread);

    ਅਚਲ size_t src0_vtcm_half_ਮਾਪ = uctx->src0_vtcm_half_ਮਾਪ;
    ਅਚਲ size_t dst_vtcm_half_ਮਾਪ  = uctx->dst_vtcm_half_ਮਾਪ;

    ਅਚਲ bool src0_contig = (nb02 == (size_t)ne01 * nb01) &&
                             (nb03 == (size_t)ne02 * nb02);
    ਅਚਲ bool dst_contig  = (nb2  == (size_t)ne1  * nb1)  &&
                             (nb3  == (size_t)ne2  * nb2);

    ਅਚਲ ਬਣਤਰ fastdiv_ਮੁੱਲਾਂ * div_ne01  = &uctx->kparams->div_ne01;
    ਅਚਲ ਬਣਤਰ fastdiv_ਮੁੱਲਾਂ * div_ne02  = &uctx->kparams->div_ne02;
    ਅਚਲ ਬਣਤਰ fastdiv_ਮੁੱਲਾਂ * div_ne012 = &uctx->kparams->div_ne012;

    ਅਚਲ uint32_t src0_max_ਖੰਡ = src0_contig ? uctx->ਖੰਡ : MIN((uint32_t)uctx->ਖੰਡ, ne01);
    ਅਚਲ uint32_t dst_max_ਖੰਡ  = dst_contig  ? uctx->ਖੰਡ : MIN((uint32_t)uctx->ਖੰਡ, ne1);
    ਅਚਲ uint32_t ਖੰਡ_2 = MIN(src0_max_ਖੰਡ, dst_max_ਖੰਡ);
    ਜੇ (ਖੰਡ_2 == 0) {
        FARF(ERROR, "unary-row-block : current VTCM reservation %zu is too small, needed at least %zu\n",
             uctx->vtcm_src0_ਮਾਪ_ਪ੍ਰਤੀ_thread, src0_ਪੰਕਤੀ_ਮਾਪ_aligned);
        ਵਾਪਸ;
    }

    dma_ਕਤਾਰ * dma_q = octx->ctx->dma[ith];

    ਲਈ (uint32_t ir = src0_ਸ਼ੁਰੂ_ਪੰਕਤੀ, vtcm_idx = 0; ir < src0_ਸਿਰਾ_ਪੰਕਤੀ && vtcm_idx < 2; vtcm_idx++) {
        ਅਚਲ uint32_t ਖੰਡ_ਮਾਪ_3 = unary_ਖੰਡ_ਮਾਪ(ir, src0_ਸਿਰਾ_ਪੰਕਤੀ, ਖੰਡ_2, src0_contig, dst_contig,
                                                     ne01, div_ne01);

        dma_ਕਤਾਰ_ਧੱਕੋ(dma_q,
            dma_make_ਅੰਕੜੇ(ਅੰਕੜੇ_dst, dst_vtcm_ਅੰਕੜੇ + (vtcm_idx * dst_vtcm_half_ਮਾਪ)),
            nb1, dst_ਪੰਕਤੀ_ਮਾਪ_aligned, dst_ਅੰਕੜੇ_ਪੰਕਤੀ_ਮਾਪ, 0);

        ਅਚਲ size_t src0_off = src0_contig ? (ir * nb01) :
            unary_ਪੰਕਤੀ_ਵਿੱਥ(ir, ne01, ne02, div_ne01, div_ne02, div_ne012, nb01, nb02, nb03);
        dma_ਕਤਾਰ_ਧੱਕੋ(dma_q,
            dma_make_ਅੰਕੜੇ(src0_vtcm_ਅੰਕੜੇ + (vtcm_idx * src0_vtcm_half_ਮਾਪ), ਅੰਕੜੇ_src + src0_off),
            src0_ਪੰਕਤੀ_ਮਾਪ_aligned, nb01, src0_ਅੰਕੜੇ_ਪੰਕਤੀ_ਮਾਪ, ਖੰਡ_ਮਾਪ_3);

        ir += ਖੰਡ_ਮਾਪ_3;
    }

    unary_compute_fn_t compute = (unary_compute_fn_t) uctx->compute;

    ਲਈ (uint32_t ir = src0_ਸ਼ੁਰੂ_ਪੰਕਤੀ; ir < src0_ਸਿਰਾ_ਪੰਕਤੀ; ) {
        ਅਚਲ uint32_t ਖੰਡ_ਮਾਪ_3 = unary_ਖੰਡ_ਮਾਪ(ir, src0_ਸਿਰਾ_ਪੰਕਤੀ, ਖੰਡ_2, src0_contig, dst_contig,
                                                     ne01, div_ne01);

        ਖਾਲੀ * dst_vtcm  = (ਖਾਲੀ *) (uintptr_t) dma_ਕਤਾਰ_ਕੱਢੋ(dma_q).src;
        ਖਾਲੀ * src0_vtcm = (ਖਾਲੀ *) (uintptr_t) dma_ਕਤਾਰ_ਕੱਢੋ(dma_q).dst;

        htp_trace_event_ਸ਼ੁਰੂ(tr, HTP_TRACE_EVT_HVX_COMP, ir);
        compute(src0_vtcm, dst_vtcm, ਖੰਡ_ਮਾਪ_3, uctx);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ir);

        ਅਚਲ size_t dst_off = dst_contig ? (ir * nb1) :
            unary_ਪੰਕਤੀ_ਵਿੱਥ(ir, ne1, ne2, div_ne01, div_ne02, div_ne012, nb1, nb2, nb3);
        dma_ਕਤਾਰ_ਧੱਕੋ(dma_q,
            dma_make_ਅੰਕੜੇ(ਅੰਕੜੇ_dst + dst_off, dst_vtcm),
            nb1, dst_ਪੰਕਤੀ_ਮਾਪ_aligned, dst_ਅੰਕੜੇ_ਪੰਕਤੀ_ਮਾਪ, ਖੰਡ_ਮਾਪ_3);

        ਅਚਲ uint32_t ਅਗਲਾ_ir = ir + ਖੰਡ_ਮਾਪ_3;
        ਜੇ (ਅਗਲਾ_ir < src0_ਸਿਰਾ_ਪੰਕਤੀ) {
            ਅਚਲ uint32_t ਅਗਲਾ_ਖੰਡ_ਮਾਪ = unary_ਖੰਡ_ਮਾਪ(ਅਗਲਾ_ir, src0_ਸਿਰਾ_ਪੰਕਤੀ, ਖੰਡ_2, src0_contig,
                                                              dst_contig, ne01, div_ne01);
            ਅਚਲ uint32_t pref_ir = ਅਗਲਾ_ir + ਅਗਲਾ_ਖੰਡ_ਮਾਪ;
            ਜੇ (pref_ir < src0_ਸਿਰਾ_ਪੰਕਤੀ) {
                ਅਚਲ uint32_t pref_ਖੰਡ_ਮਾਪ = unary_ਖੰਡ_ਮਾਪ(pref_ir, src0_ਸਿਰਾ_ਪੰਕਤੀ, ਖੰਡ_2, src0_contig,
                                                                  dst_contig, ne01, div_ne01);
                ਅਚਲ size_t src0_pref_off = src0_contig ? (pref_ir * nb01) :
                    unary_ਪੰਕਤੀ_ਵਿੱਥ(pref_ir, ne01, ne02, div_ne01, div_ne02, div_ne012, nb01, nb02, nb03);
                dma_ਕਤਾਰ_ਧੱਕੋ(dma_q,
                    dma_make_ਅੰਕੜੇ(src0_vtcm, ਅੰਕੜੇ_src + src0_pref_off),
                    src0_ਪੰਕਤੀ_ਮਾਪ_aligned, nb01, src0_ਅੰਕੜੇ_ਪੰਕਤੀ_ਮਾਪ, pref_ਖੰਡ_ਮਾਪ);
            }
        }
        ir += ਖੰਡ_ਮਾਪ_3;
    }

    dma_ਕਤਾਰ_flush(dma_q);
}

// 2. RMS_NORM_MUL row-block task with weight buffer.
ਸਥਿਰ ਖਾਲੀ unary_thread_rms_norm_mul_f32(ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ nth, ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ ith, ਖਾਲੀ * ਅੰਕੜੇ) {
    (ਖਾਲੀ) nth;
    ਅਚਲ ਬਣਤਰ htp_unary_context * uctx = (ਅਚਲ ਬਣਤਰ htp_unary_context *) ਅੰਕੜੇ;
    ਬਣਤਰ htp_ops_context * octx = uctx->octx;
    ਅਚਲ ਬਣਤਰ htp_tensor * src = octx->src[0];
    ਅਚਲ ਬਣਤਰ htp_tensor * dst = octx->dst;
    ਬਣਤਰ htp_thread_trace * tr = &octx->ctx->trace[ith];

    htp_unary_preamble;

    ਅਚਲ uint32_t src0_nrows_ਪ੍ਰਤੀ_thread = uctx->src0_nrows_ਪ੍ਰਤੀ_thread;
    ਅਚਲ size_t src0_ਅੰਕੜੇ_ਪੰਕਤੀ_ਮਾਪ = uctx->src0_ਅੰਕੜੇ_ਪੰਕਤੀ_ਮਾਪ;
    ਅਚਲ size_t dst_ਅੰਕੜੇ_ਪੰਕਤੀ_ਮਾਪ  = uctx->dst_ਅੰਕੜੇ_ਪੰਕਤੀ_ਮਾਪ;
    ਅਚਲ size_t src0_ਪੰਕਤੀ_ਮਾਪ_aligned = uctx->src0_ਪੰਕਤੀ_ਮਾਪ_aligned;
    ਅਚਲ size_t dst_ਪੰਕਤੀ_ਮਾਪ_aligned  = uctx->dst_ਪੰਕਤੀ_ਮਾਪ_aligned;

    ਅਚਲ uint32_t src0_nrows = uctx->src0_nrows;
    ਅਚਲ uint32_t src0_ਸ਼ੁਰੂ_ਪੰਕਤੀ = uctx->ਪੰਕਤੀ_ਸ਼ੁਰੂ + src0_nrows_ਪ੍ਰਤੀ_thread * ith;
    ਅਚਲ uint32_t src0_ਸਿਰਾ_ਪੰਕਤੀ   = MIN(src0_ਸ਼ੁਰੂ_ਪੰਕਤੀ + src0_nrows_ਪ੍ਰਤੀ_thread, uctx->ਪੰਕਤੀ_ਸ਼ੁਰੂ + src0_nrows);

    ਜੇ (src0_ਸ਼ੁਰੂ_ਪੰਕਤੀ >= src0_ਸਿਰਾ_ਪੰਕਤੀ) {
        ਵਾਪਸ;
    }

    ਅਚਲ dma_addr_t ਅੰਕੜੇ_src  = uctx->ਅੰਕੜੇ_src0;
    ਅਚਲ dma_addr_t ਅੰਕੜੇ_src1 = uctx->ਅੰਕੜੇ_src1;
    ਅਚਲ dma_addr_t ਅੰਕੜੇ_dst  = uctx->ਅੰਕੜੇ_dst;

    ਅਚਲ ਬਣਤਰ htp_tensor * src1 = octx->src[1];
    ਅਚਲ uint32_t nb11 = src1->nb[1];
    ਅਚਲ uint32_t nb12 = src1->nb[2];
    ਅਚਲ uint32_t nb13 = src1->nb[3];
    ਅਚਲ uint32_t nb11_bc = (src1->ne[1] > 1) ? nb11 : 0;
    ਅਚਲ uint32_t nb12_bc = (src1->ne[2] > 1) ? nb12 : 0;
    ਅਚਲ uint32_t nb13_bc = (src1->ne[3] > 1) ? nb13 : 0;
    ਅਚਲ bool src1_contig = ((nb12 == (size_t)ne01 * nb11) && (nb13 == (size_t)ne02 * nb12));

    uint8_t * src0_vtcm_ਅੰਕੜੇ = uctx->vtcm_src0 + (ith * uctx->vtcm_src0_ਮਾਪ_ਪ੍ਰਤੀ_thread);
    uint8_t * src1_vtcm_ਅੰਕੜੇ = uctx->vtcm_src1 ? (uctx->vtcm_src1 + (ith * uctx->vtcm_src1_ਮਾਪ_ਪ੍ਰਤੀ_thread)) : ਨਿਰਾ;
    uint8_t * dst_vtcm_ਅੰਕੜੇ  = uctx->vtcm_dst + (ith * uctx->vtcm_dst_ਮਾਪ_ਪ੍ਰਤੀ_thread);

    ਅਚਲ size_t src0_vtcm_half_ਮਾਪ = uctx->src0_vtcm_half_ਮਾਪ;
    ਅਚਲ size_t src1_vtcm_half_ਮਾਪ = uctx->src1_vtcm_half_ਮਾਪ;
    ਅਚਲ size_t dst_vtcm_half_ਮਾਪ  = uctx->dst_vtcm_half_ਮਾਪ;

    ਅਚਲ bool src0_contig = (nb02 == (size_t)ne01 * nb01) &&
                             (nb03 == (size_t)ne02 * nb02);
    ਅਚਲ bool dst_contig  = (nb2  == (size_t)ne1  * nb1)  &&
                             (nb3  == (size_t)ne2  * nb2);

    ਅਚਲ ਬਣਤਰ fastdiv_ਮੁੱਲਾਂ * div_ne01  = &uctx->kparams->div_ne01;
    ਅਚਲ ਬਣਤਰ fastdiv_ਮੁੱਲਾਂ * div_ne02  = &uctx->kparams->div_ne02;
    ਅਚਲ ਬਣਤਰ fastdiv_ਮੁੱਲਾਂ * div_ne012 = &uctx->kparams->div_ne012;

    ਅਚਲ bool src1_needs_ਪੰਕਤੀ_clip = !uctx->ਪ੍ਰਸਾਰਣ_ਭਾਰ && !src1_contig;
    ਅਚਲ bool ਖੰਡ_src0_contig = src0_contig && !src1_needs_ਪੰਕਤੀ_clip;
    ਅਚਲ bool ਖੰਡ_dst_contig  = dst_contig  && !src1_needs_ਪੰਕਤੀ_clip;

    ਅਚਲ uint32_t src0_max_ਖੰਡ = ਖੰਡ_src0_contig ? uctx->ਖੰਡ : MIN((uint32_t)uctx->ਖੰਡ, ne01);
    ਅਚਲ uint32_t dst_max_ਖੰਡ  = ਖੰਡ_dst_contig  ? uctx->ਖੰਡ : MIN((uint32_t)uctx->ਖੰਡ, ne1);
    ਅਚਲ uint32_t ਖੰਡ_2 = MIN(src0_max_ਖੰਡ, dst_max_ਖੰਡ);
    ਜੇ (ਖੰਡ_2 == 0) {
        FARF(ERROR, "unary-rms-norm-mul : current VTCM reservation %zu is too small, needed at least %zu\n",
             uctx->vtcm_src0_ਮਾਪ_ਪ੍ਰਤੀ_thread, src0_ਪੰਕਤੀ_ਮਾਪ_aligned);
        ਵਾਪਸ;
    }

    dma_ਕਤਾਰ * dma_q = octx->ctx->dma[ith];

    ਜੇ (uctx->ਪ੍ਰਸਾਰਣ_ਭਾਰ) {
        dma_ਕਤਾਰ_ਧੱਕੋ(dma_q, dma_make_ਅੰਕੜੇ(src1_vtcm_ਅੰਕੜੇ, ਅੰਕੜੇ_src1),
                       uctx->src1_ਪੰਕਤੀ_ਮਾਪ_aligned, 0, uctx->src1_ਅੰਕੜੇ_ਪੰਕਤੀ_ਮਾਪ, 1);
        dma_ਕਤਾਰ_flush(dma_q);
    }

    ਲਈ (uint32_t ir = src0_ਸ਼ੁਰੂ_ਪੰਕਤੀ, vtcm_idx = 0; ir < src0_ਸਿਰਾ_ਪੰਕਤੀ && vtcm_idx < 2; vtcm_idx++) {
        ਅਚਲ uint32_t ਖੰਡ_ਮਾਪ_3 = unary_ਖੰਡ_ਮਾਪ(ir, src0_ਸਿਰਾ_ਪੰਕਤੀ, ਖੰਡ_2, ਖੰਡ_src0_contig, ਖੰਡ_dst_contig,
                                                     ne01, div_ne01);

        dma_ਕਤਾਰ_ਧੱਕੋ(dma_q,
            dma_make_ਅੰਕੜੇ(ਅੰਕੜੇ_dst, dst_vtcm_ਅੰਕੜੇ + (vtcm_idx * dst_vtcm_half_ਮਾਪ)),
            nb1, dst_ਪੰਕਤੀ_ਮਾਪ_aligned, dst_ਅੰਕੜੇ_ਪੰਕਤੀ_ਮਾਪ, 0);

        ਅਚਲ size_t src0_off = src0_contig ? (ir * nb01) :
            unary_ਪੰਕਤੀ_ਵਿੱਥ(ir, ne01, ne02, div_ne01, div_ne02, div_ne012, nb01, nb02, nb03);
        dma_ਕਤਾਰ_ਧੱਕੋ(dma_q,
            dma_make_ਅੰਕੜੇ(src0_vtcm_ਅੰਕੜੇ + (vtcm_idx * src0_vtcm_half_ਮਾਪ), ਅੰਕੜੇ_src + src0_off),
            src0_ਪੰਕਤੀ_ਮਾਪ_aligned, nb01, src0_ਅੰਕੜੇ_ਪੰਕਤੀ_ਮਾਪ, ਖੰਡ_ਮਾਪ_3);

        ਜੇ (!uctx->ਪ੍ਰਸਾਰਣ_ਭਾਰ) {
            ਅਚਲ size_t src1_off = src1_contig ? (ir * nb11) :
                unary_ਪੰਕਤੀ_ਵਿੱਥ(ir, ne01, ne02, div_ne01, div_ne02, div_ne012, nb11_bc, nb12_bc, nb13_bc);
            dma_ਕਤਾਰ_ਧੱਕੋ(dma_q,
                dma_make_ਅੰਕੜੇ(src1_vtcm_ਅੰਕੜੇ + (vtcm_idx * src1_vtcm_half_ਮਾਪ), ਅੰਕੜੇ_src1 + src1_off),
                uctx->src1_ਪੰਕਤੀ_ਮਾਪ_aligned, nb11, uctx->src1_ਅੰਕੜੇ_ਪੰਕਤੀ_ਮਾਪ, ਖੰਡ_ਮਾਪ_3);
        }

        ir += ਖੰਡ_ਮਾਪ_3;
    }

    unary_rms_norm_mul_compute_fn_t compute = (unary_rms_norm_mul_compute_fn_t) uctx->compute;

    ਲਈ (uint32_t ir = src0_ਸ਼ੁਰੂ_ਪੰਕਤੀ; ir < src0_ਸਿਰਾ_ਪੰਕਤੀ; ) {
        ਅਚਲ uint32_t ਖੰਡ_ਮਾਪ_3 = unary_ਖੰਡ_ਮਾਪ(ir, src0_ਸਿਰਾ_ਪੰਕਤੀ, ਖੰਡ_2, ਖੰਡ_src0_contig, ਖੰਡ_dst_contig,
                                                     ne01, div_ne01);

        ਖਾਲੀ * dst_vtcm  = (ਖਾਲੀ *) (uintptr_t) dma_ਕਤਾਰ_ਕੱਢੋ(dma_q).src;
        ਖਾਲੀ * src0_vtcm = (ਖਾਲੀ *) (uintptr_t) dma_ਕਤਾਰ_ਕੱਢੋ(dma_q).dst;
        ਖਾਲੀ * src1_vtcm = ਨਿਰਾ;
        ਜੇ (!uctx->ਪ੍ਰਸਾਰਣ_ਭਾਰ) {
            src1_vtcm = (ਖਾਲੀ *) (uintptr_t) dma_ਕਤਾਰ_ਕੱਢੋ(dma_q).dst;
        }

        htp_trace_event_ਸ਼ੁਰੂ(tr, HTP_TRACE_EVT_HVX_COMP, ir);
        ਅਚਲ ਖਾਲੀ * w = uctx->ਪ੍ਰਸਾਰਣ_ਭਾਰ ? (ਅਚਲ ਖਾਲੀ *) src1_vtcm_ਅੰਕੜੇ : src1_vtcm;
        compute(src0_vtcm, w, dst_vtcm, ਖੰਡ_ਮਾਪ_3, uctx);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ir);

        ਅਚਲ size_t dst_off = dst_contig ? (ir * nb1) :
            unary_ਪੰਕਤੀ_ਵਿੱਥ(ir, ne1, ne2, div_ne01, div_ne02, div_ne012, nb1, nb2, nb3);
        dma_ਕਤਾਰ_ਧੱਕੋ(dma_q,
            dma_make_ਅੰਕੜੇ(ਅੰਕੜੇ_dst + dst_off, dst_vtcm),
            nb1, dst_ਪੰਕਤੀ_ਮਾਪ_aligned, dst_ਅੰਕੜੇ_ਪੰਕਤੀ_ਮਾਪ, ਖੰਡ_ਮਾਪ_3);

        ਅਚਲ uint32_t ਅਗਲਾ_ir = ir + ਖੰਡ_ਮਾਪ_3;
        ਜੇ (ਅਗਲਾ_ir < src0_ਸਿਰਾ_ਪੰਕਤੀ) {
            ਅਚਲ uint32_t ਅਗਲਾ_ਖੰਡ_ਮਾਪ = unary_ਖੰਡ_ਮਾਪ(ਅਗਲਾ_ir, src0_ਸਿਰਾ_ਪੰਕਤੀ, ਖੰਡ_2, ਖੰਡ_src0_contig,
                                                              ਖੰਡ_dst_contig, ne01, div_ne01);
            ਅਚਲ uint32_t pref_ir = ਅਗਲਾ_ir + ਅਗਲਾ_ਖੰਡ_ਮਾਪ;
            ਜੇ (pref_ir < src0_ਸਿਰਾ_ਪੰਕਤੀ) {
                ਅਚਲ uint32_t pref_ਖੰਡ_ਮਾਪ = unary_ਖੰਡ_ਮਾਪ(pref_ir, src0_ਸਿਰਾ_ਪੰਕਤੀ, ਖੰਡ_2, ਖੰਡ_src0_contig,
                                                                  ਖੰਡ_dst_contig, ne01, div_ne01);
                ਅਚਲ size_t src0_pref_off = src0_contig ? (pref_ir * nb01) :
                    unary_ਪੰਕਤੀ_ਵਿੱਥ(pref_ir, ne01, ne02, div_ne01, div_ne02, div_ne012, nb01, nb02, nb03);
                dma_ਕਤਾਰ_ਧੱਕੋ(dma_q,
                    dma_make_ਅੰਕੜੇ(src0_vtcm, ਅੰਕੜੇ_src + src0_pref_off),
                    src0_ਪੰਕਤੀ_ਮਾਪ_aligned, nb01, src0_ਅੰਕੜੇ_ਪੰਕਤੀ_ਮਾਪ, pref_ਖੰਡ_ਮਾਪ);

                ਜੇ (!uctx->ਪ੍ਰਸਾਰਣ_ਭਾਰ) {
                    ਅਚਲ size_t src1_pref_off = src1_contig ? (pref_ir * nb11) :
                        unary_ਪੰਕਤੀ_ਵਿੱਥ(pref_ir, ne01, ne02, div_ne01, div_ne02, div_ne012, nb11_bc, nb12_bc,
                                          nb13_bc);
                    dma_ਕਤਾਰ_ਧੱਕੋ(dma_q,
                        dma_make_ਅੰਕੜੇ(src1_vtcm, ਅੰਕੜੇ_src1 + src1_pref_off),
                        uctx->src1_ਪੰਕਤੀ_ਮਾਪ_aligned, nb11, uctx->src1_ਅੰਕੜੇ_ਪੰਕਤੀ_ਮਾਪ, pref_ਖੰਡ_ਮਾਪ);
                }
            }
        }
        ir += ਖੰਡ_ਮਾਪ_3;
    }

    dma_ਕਤਾਰ_flush(dma_q);
}

// 3. TRI row-block task with row index ir.
ਸਥਿਰ ਖਾਲੀ unary_thread_tri_f32(ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ nth, ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ ith, ਖਾਲੀ * ਅੰਕੜੇ) {
    (ਖਾਲੀ) nth;
    ਅਚਲ ਬਣਤਰ htp_unary_context * uctx = (ਅਚਲ ਬਣਤਰ htp_unary_context *) ਅੰਕੜੇ;
    ਬਣਤਰ htp_ops_context * octx = uctx->octx;
    ਅਚਲ ਬਣਤਰ htp_tensor * src = octx->src[0];
    ਅਚਲ ਬਣਤਰ htp_tensor * dst = octx->dst;
    ਬਣਤਰ htp_thread_trace * tr = &octx->ctx->trace[ith];

    htp_unary_preamble;

    ਅਚਲ uint32_t src0_nrows_ਪ੍ਰਤੀ_thread = uctx->src0_nrows_ਪ੍ਰਤੀ_thread;
    ਅਚਲ size_t src0_ਅੰਕੜੇ_ਪੰਕਤੀ_ਮਾਪ = uctx->src0_ਅੰਕੜੇ_ਪੰਕਤੀ_ਮਾਪ;
    ਅਚਲ size_t dst_ਅੰਕੜੇ_ਪੰਕਤੀ_ਮਾਪ  = uctx->dst_ਅੰਕੜੇ_ਪੰਕਤੀ_ਮਾਪ;
    ਅਚਲ size_t src0_ਪੰਕਤੀ_ਮਾਪ_aligned = uctx->src0_ਪੰਕਤੀ_ਮਾਪ_aligned;
    ਅਚਲ size_t dst_ਪੰਕਤੀ_ਮਾਪ_aligned  = uctx->dst_ਪੰਕਤੀ_ਮਾਪ_aligned;

    ਅਚਲ uint32_t src0_nrows = uctx->src0_nrows;
    ਅਚਲ uint32_t src0_ਸ਼ੁਰੂ_ਪੰਕਤੀ = uctx->ਪੰਕਤੀ_ਸ਼ੁਰੂ + src0_nrows_ਪ੍ਰਤੀ_thread * ith;
    ਅਚਲ uint32_t src0_ਸਿਰਾ_ਪੰਕਤੀ   = MIN(src0_ਸ਼ੁਰੂ_ਪੰਕਤੀ + src0_nrows_ਪ੍ਰਤੀ_thread, uctx->ਪੰਕਤੀ_ਸ਼ੁਰੂ + src0_nrows);

    ਜੇ (src0_ਸ਼ੁਰੂ_ਪੰਕਤੀ >= src0_ਸਿਰਾ_ਪੰਕਤੀ) {
        ਵਾਪਸ;
    }

    ਅਚਲ dma_addr_t ਅੰਕੜੇ_src = uctx->ਅੰਕੜੇ_src0;
    ਅਚਲ dma_addr_t ਅੰਕੜੇ_dst = uctx->ਅੰਕੜੇ_dst;

    uint8_t * src0_vtcm_ਅੰਕੜੇ = uctx->vtcm_src0 + (ith * uctx->vtcm_src0_ਮਾਪ_ਪ੍ਰਤੀ_thread);
    uint8_t * dst_vtcm_ਅੰਕੜੇ  = uctx->vtcm_dst + (ith * uctx->vtcm_dst_ਮਾਪ_ਪ੍ਰਤੀ_thread);

    ਅਚਲ size_t src0_vtcm_half_ਮਾਪ = uctx->src0_vtcm_half_ਮਾਪ;
    ਅਚਲ size_t dst_vtcm_half_ਮਾਪ  = uctx->dst_vtcm_half_ਮਾਪ;

    ਅਚਲ bool src0_contig = (nb02 == (size_t)ne01 * nb01) &&
                             (nb03 == (size_t)ne02 * nb02);
    ਅਚਲ bool dst_contig  = (nb2  == (size_t)ne1  * nb1)  &&
                             (nb3  == (size_t)ne2  * nb2);

    ਅਚਲ ਬਣਤਰ fastdiv_ਮੁੱਲਾਂ * div_ne01  = &uctx->kparams->div_ne01;
    ਅਚਲ ਬਣਤਰ fastdiv_ਮੁੱਲਾਂ * div_ne02  = &uctx->kparams->div_ne02;
    ਅਚਲ ਬਣਤਰ fastdiv_ਮੁੱਲਾਂ * div_ne012 = &uctx->kparams->div_ne012;

    ਅਚਲ uint32_t src0_max_ਖੰਡ = src0_contig ? uctx->ਖੰਡ : MIN((uint32_t)uctx->ਖੰਡ, ne01);
    ਅਚਲ uint32_t dst_max_ਖੰਡ  = dst_contig  ? uctx->ਖੰਡ : MIN((uint32_t)uctx->ਖੰਡ, ne1);
    ਅਚਲ uint32_t ਖੰਡ_2 = MIN(src0_max_ਖੰਡ, dst_max_ਖੰਡ);
    ਜੇ (ਖੰਡ_2 == 0) {
        FARF(ERROR, "unary-tri : current VTCM reservation %zu is too small, needed at least %zu\n",
             uctx->vtcm_src0_ਮਾਪ_ਪ੍ਰਤੀ_thread, src0_ਪੰਕਤੀ_ਮਾਪ_aligned);
        ਵਾਪਸ;
    }

    dma_ਕਤਾਰ * dma_q = octx->ctx->dma[ith];

    ਲਈ (uint32_t ir = src0_ਸ਼ੁਰੂ_ਪੰਕਤੀ, vtcm_idx = 0; ir < src0_ਸਿਰਾ_ਪੰਕਤੀ && vtcm_idx < 2; vtcm_idx++) {
        ਅਚਲ uint32_t ਖੰਡ_ਮਾਪ_3 = unary_ਖੰਡ_ਮਾਪ(ir, src0_ਸਿਰਾ_ਪੰਕਤੀ, ਖੰਡ_2, src0_contig, dst_contig,
                                                     ne01, div_ne01);

        dma_ਕਤਾਰ_ਧੱਕੋ(dma_q,
            dma_make_ਅੰਕੜੇ(ਅੰਕੜੇ_dst, dst_vtcm_ਅੰਕੜੇ + (vtcm_idx * dst_vtcm_half_ਮਾਪ)),
            nb1, dst_ਪੰਕਤੀ_ਮਾਪ_aligned, dst_ਅੰਕੜੇ_ਪੰਕਤੀ_ਮਾਪ, 0);

        ਅਚਲ size_t src0_off = src0_contig ? (ir * nb01) :
            unary_ਪੰਕਤੀ_ਵਿੱਥ(ir, ne01, ne02, div_ne01, div_ne02, div_ne012, nb01, nb02, nb03);
        dma_ਕਤਾਰ_ਧੱਕੋ(dma_q,
            dma_make_ਅੰਕੜੇ(src0_vtcm_ਅੰਕੜੇ + (vtcm_idx * src0_vtcm_half_ਮਾਪ), ਅੰਕੜੇ_src + src0_off),
            src0_ਪੰਕਤੀ_ਮਾਪ_aligned, nb01, src0_ਅੰਕੜੇ_ਪੰਕਤੀ_ਮਾਪ, ਖੰਡ_ਮਾਪ_3);

        ir += ਖੰਡ_ਮਾਪ_3;
    }

    unary_tri_compute_fn_t compute = (unary_tri_compute_fn_t) uctx->compute;

    ਲਈ (uint32_t ir = src0_ਸ਼ੁਰੂ_ਪੰਕਤੀ; ir < src0_ਸਿਰਾ_ਪੰਕਤੀ; ) {
        ਅਚਲ uint32_t ਖੰਡ_ਮਾਪ_3 = unary_ਖੰਡ_ਮਾਪ(ir, src0_ਸਿਰਾ_ਪੰਕਤੀ, ਖੰਡ_2, src0_contig, dst_contig,
                                                     ne01, div_ne01);

        ਖਾਲੀ * dst_vtcm  = (ਖਾਲੀ *) (uintptr_t) dma_ਕਤਾਰ_ਕੱਢੋ(dma_q).src;
        ਖਾਲੀ * src0_vtcm = (ਖਾਲੀ *) (uintptr_t) dma_ਕਤਾਰ_ਕੱਢੋ(dma_q).dst;

        htp_trace_event_ਸ਼ੁਰੂ(tr, HTP_TRACE_EVT_HVX_COMP, ir);
        compute(src0_vtcm, dst_vtcm, ਖੰਡ_ਮਾਪ_3, ir, uctx);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ir);

        ਅਚਲ size_t dst_off = dst_contig ? (ir * nb1) :
            unary_ਪੰਕਤੀ_ਵਿੱਥ(ir, ne1, ne2, div_ne01, div_ne02, div_ne012, nb1, nb2, nb3);
        dma_ਕਤਾਰ_ਧੱਕੋ(dma_q,
            dma_make_ਅੰਕੜੇ(ਅੰਕੜੇ_dst + dst_off, dst_vtcm),
            nb1, dst_ਪੰਕਤੀ_ਮਾਪ_aligned, dst_ਅੰਕੜੇ_ਪੰਕਤੀ_ਮਾਪ, ਖੰਡ_ਮਾਪ_3);

        ਅਚਲ uint32_t ਅਗਲਾ_ir = ir + ਖੰਡ_ਮਾਪ_3;
        ਜੇ (ਅਗਲਾ_ir < src0_ਸਿਰਾ_ਪੰਕਤੀ) {
            ਅਚਲ uint32_t ਅਗਲਾ_ਖੰਡ_ਮਾਪ = unary_ਖੰਡ_ਮਾਪ(ਅਗਲਾ_ir, src0_ਸਿਰਾ_ਪੰਕਤੀ, ਖੰਡ_2, src0_contig,
                                                              dst_contig, ne01, div_ne01);
            ਅਚਲ uint32_t pref_ir = ਅਗਲਾ_ir + ਅਗਲਾ_ਖੰਡ_ਮਾਪ;
            ਜੇ (pref_ir < src0_ਸਿਰਾ_ਪੰਕਤੀ) {
                ਅਚਲ uint32_t pref_ਖੰਡ_ਮਾਪ = unary_ਖੰਡ_ਮਾਪ(pref_ir, src0_ਸਿਰਾ_ਪੰਕਤੀ, ਖੰਡ_2, src0_contig,
                                                                  dst_contig, ne01, div_ne01);
                ਅਚਲ size_t src0_pref_off = src0_contig ? (pref_ir * nb01) :
                    unary_ਪੰਕਤੀ_ਵਿੱਥ(pref_ir, ne01, ne02, div_ne01, div_ne02, div_ne012, nb01, nb02, nb03);
                dma_ਕਤਾਰ_ਧੱਕੋ(dma_q,
                    dma_make_ਅੰਕੜੇ(src0_vtcm, ਅੰਕੜੇ_src + src0_pref_off),
                    src0_ਪੰਕਤੀ_ਮਾਪ_aligned, nb01, src0_ਅੰਕੜੇ_ਪੰਕਤੀ_ਮਾਪ, pref_ਖੰਡ_ਮਾਪ);
            }
        }
        ir += ਖੰਡ_ਮਾਪ_3;
    }

    dma_ਕਤਾਰ_flush(dma_q);
}

// 4. Pointwise tiled unary task.
ਸਥਿਰ ਖਾਲੀ unary_thread_tiled(ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ nth, ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ ith, ਖਾਲੀ * ਅੰਕੜੇ) {
    (ਖਾਲੀ) nth;
    ਅਚਲ ਬਣਤਰ htp_unary_context * uctx = (ਅਚਲ ਬਣਤਰ htp_unary_context *) ਅੰਕੜੇ;
    ਬਣਤਰ htp_ops_context * octx = uctx->octx;
    ਅਚਲ ਬਣਤਰ htp_tensor * src = octx->src[0];
    ਅਚਲ ਬਣਤਰ htp_tensor * dst = octx->dst;
    ਬਣਤਰ htp_thread_trace * tr = &octx->ctx->trace[ith];

    htp_unary_preamble;

    ਅਚਲ uint32_t src0_nrows_ਪ੍ਰਤੀ_thread = uctx->src0_nrows_ਪ੍ਰਤੀ_thread;
    ਅਚਲ uint32_t ਖਾਨਾ_tile              = uctx->ਖਾਨਾ_tile;

    ਅਚਲ uint32_t src0_nrows     = uctx->src0_nrows;
    ਅਚਲ uint32_t src0_ਸ਼ੁਰੂ_ਪੰਕਤੀ = uctx->ਪੰਕਤੀ_ਸ਼ੁਰੂ + src0_nrows_ਪ੍ਰਤੀ_thread * ith;
    ਅਚਲ uint32_t src0_ਸਿਰਾ_ਪੰਕਤੀ   = MIN(src0_ਸ਼ੁਰੂ_ਪੰਕਤੀ + src0_nrows_ਪ੍ਰਤੀ_thread, uctx->ਪੰਕਤੀ_ਸ਼ੁਰੂ + src0_nrows);

    ਜੇ (src0_ਸ਼ੁਰੂ_ਪੰਕਤੀ >= src0_ਸਿਰਾ_ਪੰਕਤੀ) {
        ਵਾਪਸ;
    }

    ਅਚਲ dma_addr_t ਅੰਕੜੇ_src = uctx->ਅੰਕੜੇ_src0;
    ਅਚਲ dma_addr_t ਅੰਕੜੇ_dst = uctx->ਅੰਕੜੇ_dst;

    uint8_t * src0_vtcm_ਅੰਕੜੇ = uctx->vtcm_src0 + (ith * uctx->vtcm_src0_ਮਾਪ_ਪ੍ਰਤੀ_thread);
    uint8_t * dst_vtcm_ਅੰਕੜੇ  = uctx->vtcm_dst + (ith * uctx->vtcm_dst_ਮਾਪ_ਪ੍ਰਤੀ_thread);

    ਅਚਲ size_t src0_half = uctx->src0_vtcm_half_ਮਾਪ;
    ਅਚਲ size_t dst_half  = uctx->dst_vtcm_half_ਮਾਪ;

    dma_ਕਤਾਰ * dma_q = octx->ctx->dma[ith];

    ਅਚਲ ਬਣਤਰ fastdiv_ਮੁੱਲਾਂ * div_ne01  = &uctx->kparams->div_ne01;
    ਅਚਲ ਬਣਤਰ fastdiv_ਮੁੱਲਾਂ * div_ne02  = &uctx->kparams->div_ne02;
    ਅਚਲ ਬਣਤਰ fastdiv_ਮੁੱਲਾਂ * div_ne012 = &uctx->kparams->div_ne012;
    ਅਚਲ ਬਣਤਰ fastdiv_ਮੁੱਲਾਂ * div_tpr   = &uctx->kparams->div_tpr;

    ਅਚਲ uint32_t tiles_ਪ੍ਰਤੀ_ਪੰਕਤੀ = (ne0 + ਖਾਨਾ_tile - 1) / ਖਾਨਾ_tile;

    ਅਚਲ bool src0_contig = (nb02 == (size_t)ne01 * nb01) &&
                             (nb03 == (size_t)ne02 * nb02);
    ਅਚਲ bool dst_contig  = (nb2  == (size_t)ne1  * nb1)  &&
                             (nb3  == (size_t)ne2  * nb2);

    ਅਚਲ uint32_t ਕੁੱਲ_tiles = (src0_ਸਿਰਾ_ਪੰਕਤੀ - src0_ਸ਼ੁਰੂ_ਪੰਕਤੀ) * tiles_ਪ੍ਰਤੀ_ਪੰਕਤੀ;

    ਲਈ (uint32_t t = 0, vtcm_idx = 0; t < ਕੁੱਲ_tiles && vtcm_idx < 2; t++, vtcm_idx++) {
        ਅਚਲ uint32_t ਪੰਕਤੀ  = src0_ਸ਼ੁਰੂ_ਪੰਕਤੀ + t / tiles_ਪ੍ਰਤੀ_ਪੰਕਤੀ;
        ਅਚਲ uint32_t ਖਾਨਾ  = (t % tiles_ਪ੍ਰਤੀ_ਪੰਕਤੀ) * ਖਾਨਾ_tile;
        ਅਚਲ uint32_t tw   = MIN(ਖਾਨਾ_tile, ne0 - ਖਾਨਾ);
        ਅਚਲ size_t   tb   = (size_t) tw * ਆਕਾਰ(ਦਸ਼ਮਲਵ);
        ਅਚਲ size_t   soff = (src0_contig ? (ਪੰਕਤੀ * nb01) :
                               unary_ਪੰਕਤੀ_ਵਿੱਥ(ਪੰਕਤੀ, ne01, ne02, div_ne01, div_ne02, div_ne012, nb01, nb02, nb03)) +
                               (size_t) ਖਾਨਾ * ਆਕਾਰ(ਦਸ਼ਮਲਵ);

        dma_ਕਤਾਰ_ਧੱਕੋ(dma_q, dma_make_ਅੰਕੜੇ(ਅੰਕੜੇ_dst, dst_vtcm_ਅੰਕੜੇ + (vtcm_idx * dst_half)), 0, 0, 0, 0);
        dma_ਕਤਾਰ_ਧੱਕੋ(dma_q, dma_make_ਅੰਕੜੇ(src0_vtcm_ਅੰਕੜੇ + (vtcm_idx * src0_half), ਅੰਕੜੇ_src + soff), tb, tb, tb, 1);
    }

    unary_tile_compute_fn_t compute = (unary_tile_compute_fn_t) uctx->compute;

    uint32_t ਪੰਕਤੀ = src0_ਸ਼ੁਰੂ_ਪੰਕਤੀ;
    uint32_t ਖਾਨਾ = 0;
    uint32_t tile_ਵਿੱਚ_ਪੰਕਤੀ = 0;

    uint32_t prow = src0_ਸ਼ੁਰੂ_ਪੰਕਤੀ + fastdiv(2, div_tpr);
    uint32_t pcol = fastmodulo(2, tiles_ਪ੍ਰਤੀ_ਪੰਕਤੀ, div_tpr) * ਖਾਨਾ_tile;
    uint32_t ptile_ਵਿੱਚ_ਪੰਕਤੀ = fastmodulo(2, tiles_ਪ੍ਰਤੀ_ਪੰਕਤੀ, div_tpr);

    ਲਈ (uint32_t t = 0; t < ਕੁੱਲ_tiles; t++) {
        ਖਾਲੀ * dst_vtcm = (ਖਾਲੀ *) (uintptr_t) dma_ਕਤਾਰ_ਕੱਢੋ(dma_q).src;
        ਖਾਲੀ * src_vtcm = (ਖਾਲੀ *) (uintptr_t) dma_ਕਤਾਰ_ਕੱਢੋ(dma_q).dst;

        ਅਚਲ uint32_t tw = MIN(ਖਾਨਾ_tile, ne0 - ਖਾਨਾ);

        htp_trace_event_ਸ਼ੁਰੂ(tr, HTP_TRACE_EVT_HVX_COMP, t);
        compute(dst_vtcm, src_vtcm, tw, uctx);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, t);

        ਅਚਲ size_t doff = (dst_contig ? (ਪੰਕਤੀ * nb1) :
                             unary_ਪੰਕਤੀ_ਵਿੱਥ(ਪੰਕਤੀ, ne1, ne2, div_ne01, div_ne02, div_ne012, nb1, nb2, nb3)) +
                             (size_t) ਖਾਨਾ * ਆਕਾਰ(ਦਸ਼ਮਲਵ);
        ਅਚਲ size_t tb   = (size_t) tw * ਆਕਾਰ(ਦਸ਼ਮਲਵ);
        dma_ਕਤਾਰ_ਧੱਕੋ(dma_q, dma_make_ਅੰਕੜੇ(ਅੰਕੜੇ_dst + doff, dst_vtcm), tb, tb, tb, 1);

        ਅਚਲ uint32_t pt = t + 2;
        ਜੇ (pt < ਕੁੱਲ_tiles) {
            ਅਚਲ uint32_t ptw  = MIN(ਖਾਨਾ_tile, ne0 - pcol);
            ਅਚਲ size_t   ptb  = (size_t) ptw * ਆਕਾਰ(ਦਸ਼ਮਲਵ);
            ਅਚਲ size_t   psoff = (src0_contig ? (prow * nb01) :
                                    unary_ਪੰਕਤੀ_ਵਿੱਥ(prow, ne01, ne02, div_ne01, div_ne02, div_ne012, nb01, nb02,
                                                     nb03)) +
                                   (size_t) pcol * ਆਕਾਰ(ਦਸ਼ਮਲਵ);
            dma_ਕਤਾਰ_ਧੱਕੋ(dma_q, dma_make_ਅੰਕੜੇ(src_vtcm, ਅੰਕੜੇ_src + psoff), ptb, ptb, ptb, 1);
        }

        tile_ਵਿੱਚ_ਪੰਕਤੀ++;
        ਖਾਨਾ += ਖਾਨਾ_tile;
        ਜੇ (tile_ਵਿੱਚ_ਪੰਕਤੀ == tiles_ਪ੍ਰਤੀ_ਪੰਕਤੀ) {
            tile_ਵਿੱਚ_ਪੰਕਤੀ = 0;
            ਖਾਨਾ = 0;
            ਪੰਕਤੀ++;
        }

        ptile_ਵਿੱਚ_ਪੰਕਤੀ++;
        pcol += ਖਾਨਾ_tile;
        ਜੇ (ptile_ਵਿੱਚ_ਪੰਕਤੀ == tiles_ਪ੍ਰਤੀ_ਪੰਕਤੀ) {
            ptile_ਵਿੱਚ_ਪੰਕਤੀ = 0;
            pcol = 0;
            prow++;
        }
    }

    dma_ਕਤਾਰ_flush(dma_q);
}

// 5. TRI tiled task.
ਸਥਿਰ ਖਾਲੀ unary_thread_tiled_tri_f32(ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ nth, ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ ith, ਖਾਲੀ * ਅੰਕੜੇ) {
    (ਖਾਲੀ) nth;
    ਅਚਲ ਬਣਤਰ htp_unary_context * uctx = (ਅਚਲ ਬਣਤਰ htp_unary_context *) ਅੰਕੜੇ;
    ਬਣਤਰ htp_ops_context * octx = uctx->octx;
    ਅਚਲ ਬਣਤਰ htp_tensor * src = octx->src[0];
    ਅਚਲ ਬਣਤਰ htp_tensor * dst = octx->dst;
    ਬਣਤਰ htp_thread_trace * tr = &octx->ctx->trace[ith];

    htp_unary_preamble;

    ਅਚਲ uint32_t src0_nrows_ਪ੍ਰਤੀ_thread = uctx->src0_nrows_ਪ੍ਰਤੀ_thread;
    ਅਚਲ int32_t * op_params            = octx->op_params;
    ਅਚਲ uint32_t ਖਾਨਾ_tile              = uctx->ਖਾਨਾ_tile;

    ਅਚਲ uint32_t src0_nrows     = uctx->src0_nrows;
    ਅਚਲ uint32_t src0_ਸ਼ੁਰੂ_ਪੰਕਤੀ = uctx->ਪੰਕਤੀ_ਸ਼ੁਰੂ + src0_nrows_ਪ੍ਰਤੀ_thread * ith;
    ਅਚਲ uint32_t src0_ਸਿਰਾ_ਪੰਕਤੀ   = MIN(src0_ਸ਼ੁਰੂ_ਪੰਕਤੀ + src0_nrows_ਪ੍ਰਤੀ_thread, uctx->ਪੰਕਤੀ_ਸ਼ੁਰੂ + src0_nrows);

    ਜੇ (src0_ਸ਼ੁਰੂ_ਪੰਕਤੀ >= src0_ਸਿਰਾ_ਪੰਕਤੀ) {
        ਵਾਪਸ;
    }

    ਅਚਲ dma_addr_t ਅੰਕੜੇ_src = uctx->ਅੰਕੜੇ_src0;
    ਅਚਲ dma_addr_t ਅੰਕੜੇ_dst = uctx->ਅੰਕੜੇ_dst;

    uint8_t * src0_vtcm_ਅੰਕੜੇ = uctx->vtcm_src0 + (ith * uctx->vtcm_src0_ਮਾਪ_ਪ੍ਰਤੀ_thread);
    uint8_t * dst_vtcm_ਅੰਕੜੇ  = uctx->vtcm_dst + (ith * uctx->vtcm_dst_ਮਾਪ_ਪ੍ਰਤੀ_thread);

    ਅਚਲ size_t src0_half = uctx->src0_vtcm_half_ਮਾਪ;
    ਅਚਲ size_t dst_half  = uctx->dst_vtcm_half_ਮਾਪ;

    dma_ਕਤਾਰ * dma_q = octx->ctx->dma[ith];

    ਅਚਲ ਬਣਤਰ fastdiv_ਮੁੱਲਾਂ * div_ne01  = &uctx->kparams->div_ne01;
    ਅਚਲ ਬਣਤਰ fastdiv_ਮੁੱਲਾਂ * div_ne02  = &uctx->kparams->div_ne02;
    ਅਚਲ ਬਣਤਰ fastdiv_ਮੁੱਲਾਂ * div_ne012 = &uctx->kparams->div_ne012;
    ਅਚਲ ਬਣਤਰ fastdiv_ਮੁੱਲਾਂ * div_tpr   = &uctx->kparams->div_tpr;

    ਅਚਲ uint32_t tiles_ਪ੍ਰਤੀ_ਪੰਕਤੀ = (ne0 + ਖਾਨਾ_tile - 1) / ਖਾਨਾ_tile;
    ਅਚਲ int32_t  tri_ttype     = op_params[0];

    ਅਚਲ bool src0_contig = (nb02 == (size_t)ne01 * nb01) &&
                             (nb03 == (size_t)ne02 * nb02);
    ਅਚਲ bool dst_contig  = (nb2  == (size_t)ne1  * nb1)  &&
                             (nb3  == (size_t)ne2  * nb2);

    ਅਚਲ uint32_t ਕੁੱਲ_tiles = (src0_ਸਿਰਾ_ਪੰਕਤੀ - src0_ਸ਼ੁਰੂ_ਪੰਕਤੀ) * tiles_ਪ੍ਰਤੀ_ਪੰਕਤੀ;

    ਲਈ (uint32_t t = 0, vtcm_idx = 0; t < ਕੁੱਲ_tiles && vtcm_idx < 2; t++, vtcm_idx++) {
        ਅਚਲ uint32_t ਪੰਕਤੀ  = src0_ਸ਼ੁਰੂ_ਪੰਕਤੀ + t / tiles_ਪ੍ਰਤੀ_ਪੰਕਤੀ;
        ਅਚਲ uint32_t ਖਾਨਾ  = (t % tiles_ਪ੍ਰਤੀ_ਪੰਕਤੀ) * ਖਾਨਾ_tile;
        ਅਚਲ uint32_t tw   = MIN(ਖਾਨਾ_tile, ne0 - ਖਾਨਾ);
        ਅਚਲ size_t   tb   = (size_t) tw * ਆਕਾਰ(ਦਸ਼ਮਲਵ);
        ਅਚਲ size_t   soff = (src0_contig ? (ਪੰਕਤੀ * nb01) :
                               unary_ਪੰਕਤੀ_ਵਿੱਥ(ਪੰਕਤੀ, ne01, ne02, div_ne01, div_ne02, div_ne012, nb01, nb02, nb03)) +
                               (size_t) ਖਾਨਾ * ਆਕਾਰ(ਦਸ਼ਮਲਵ);

        dma_ਕਤਾਰ_ਧੱਕੋ(dma_q, dma_make_ਅੰਕੜੇ(ਅੰਕੜੇ_dst, dst_vtcm_ਅੰਕੜੇ + (vtcm_idx * dst_half)), 0, 0, 0, 0);
        dma_ਕਤਾਰ_ਧੱਕੋ(dma_q, dma_make_ਅੰਕੜੇ(src0_vtcm_ਅੰਕੜੇ + (vtcm_idx * src0_half), ਅੰਕੜੇ_src + soff), tb, tb, tb, 1);
    }

    unary_tiled_tri_compute_fn_t compute = (unary_tiled_tri_compute_fn_t) uctx->compute;

    uint32_t ਪੰਕਤੀ = src0_ਸ਼ੁਰੂ_ਪੰਕਤੀ;
    uint32_t ਖਾਨਾ = 0;
    uint32_t tile_ਵਿੱਚ_ਪੰਕਤੀ = 0;
    uint32_t i01 = fastmodulo(ਪੰਕਤੀ, ne01, div_ne01);

    uint32_t prow = src0_ਸ਼ੁਰੂ_ਪੰਕਤੀ + fastdiv(2, div_tpr);
    uint32_t pcol = fastmodulo(2, tiles_ਪ੍ਰਤੀ_ਪੰਕਤੀ, div_tpr) * ਖਾਨਾ_tile;
    uint32_t ptile_ਵਿੱਚ_ਪੰਕਤੀ = fastmodulo(2, tiles_ਪ੍ਰਤੀ_ਪੰਕਤੀ, div_tpr);

    ਲਈ (uint32_t t = 0; t < ਕੁੱਲ_tiles; t++) {
        ਖਾਲੀ * dst_vtcm = (ਖਾਲੀ *) (uintptr_t) dma_ਕਤਾਰ_ਕੱਢੋ(dma_q).src;
        ਖਾਲੀ * src_vtcm = (ਖਾਲੀ *) (uintptr_t) dma_ਕਤਾਰ_ਕੱਢੋ(dma_q).dst;

        ਅਚਲ uint32_t tw = MIN(ਖਾਨਾ_tile, ne0 - ਖਾਨਾ);

        htp_trace_event_ਸ਼ੁਰੂ(tr, HTP_TRACE_EVT_HVX_COMP, t);
        compute(src_vtcm, dst_vtcm, tw, ਖਾਨਾ, i01, ne0, tri_ttype);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, t);

        ਅਚਲ size_t doff = (dst_contig ? (ਪੰਕਤੀ * nb1) :
                             unary_ਪੰਕਤੀ_ਵਿੱਥ(ਪੰਕਤੀ, ne1, ne2, div_ne01, div_ne02, div_ne012, nb1, nb2, nb3)) +
                             (size_t) ਖਾਨਾ * ਆਕਾਰ(ਦਸ਼ਮਲਵ);
        ਅਚਲ size_t tb   = (size_t) tw * ਆਕਾਰ(ਦਸ਼ਮਲਵ);
        dma_ਕਤਾਰ_ਧੱਕੋ(dma_q, dma_make_ਅੰਕੜੇ(ਅੰਕੜੇ_dst + doff, dst_vtcm), tb, tb, tb, 1);

        ਅਚਲ uint32_t pt = t + 2;
        ਜੇ (pt < ਕੁੱਲ_tiles) {
            ਅਚਲ uint32_t ptw  = MIN(ਖਾਨਾ_tile, ne0 - pcol);
            ਅਚਲ size_t   ptb  = (size_t) ptw * ਆਕਾਰ(ਦਸ਼ਮਲਵ);
            ਅਚਲ size_t   psoff = (src0_contig ? (prow * nb01) :
                                    unary_ਪੰਕਤੀ_ਵਿੱਥ(prow, ne01, ne02, div_ne01, div_ne02, div_ne012, nb01, nb02,
                                                     nb03)) +
                                   (size_t) pcol * ਆਕਾਰ(ਦਸ਼ਮਲਵ);
            dma_ਕਤਾਰ_ਧੱਕੋ(dma_q, dma_make_ਅੰਕੜੇ(src_vtcm, ਅੰਕੜੇ_src + psoff), ptb, ptb, ptb, 1);
        }

        tile_ਵਿੱਚ_ਪੰਕਤੀ++;
        ਖਾਨਾ += ਖਾਨਾ_tile;
        ਜੇ (tile_ਵਿੱਚ_ਪੰਕਤੀ == tiles_ਪ੍ਰਤੀ_ਪੰਕਤੀ) {
            tile_ਵਿੱਚ_ਪੰਕਤੀ = 0;
            ਖਾਨਾ = 0;
            ਪੰਕਤੀ++;
            i01++;
            ਜੇ (i01 == ne01) {
                i01 = 0;
            }
        }

        ptile_ਵਿੱਚ_ਪੰਕਤੀ++;
        pcol += ਖਾਨਾ_tile;
        ਜੇ (ptile_ਵਿੱਚ_ਪੰਕਤੀ == tiles_ਪ੍ਰਤੀ_ਪੰਕਤੀ) {
            ptile_ਵਿੱਚ_ਪੰਕਤੀ = 0;
            pcol = 0;
            prow++;
        }
    }

    dma_ਕਤਾਰ_flush(dma_q);
}

ਸਥਿਰ ਪੂਰਨ execute_op_unary(ਬਣਤਰ htp_ops_context * octx) {
    ਪੂਰਨ err = HTP_ਸਥਿਤੀ_OK;

    ਅਚਲ ਬਣਤਰ htp_tensor * src0 = octx->src[0];
    ਅਚਲ ਬਣਤਰ htp_tensor * dst  = octx->dst;

    ਅਚਲ bool is_f16 = (src0->type == HTP_TYPE_F16);

    ਅਚਲ ਅੱਖਰ * op_type = ਨਿਰਾ;

    ਚੋਣ (octx->op) {
        ਹਾਲਤ HTP_OP_NORM:            op_type = is_f16 ? "norm-f16"     : "norm-f32";         ਰੋਕੋ;
        ਹਾਲਤ HTP_OP_RMS_NORM:        op_type = is_f16 ? "rmsnorm-f16"  : "rmsnorm-f32";      ਰੋਕੋ;
        ਹਾਲਤ HTP_OP_RMS_NORM_MUL:    op_type = "rmsnorm-mul-f32";                            ਰੋਕੋ;
        ਹਾਲਤ HTP_OP_SCALE:           op_type = is_f16 ? "scale-f16"    : "scale-f32";        ਰੋਕੋ;
        ਹਾਲਤ HTP_OP_CLAMP:           op_type = is_f16 ? "clamp-f16"    : "clamp-f32";        ਰੋਕੋ;
        ਹਾਲਤ HTP_OP_LEAKY_RELU:      op_type = "leaky-relu-f32";                             ਰੋਕੋ;
        ਹਾਲਤ HTP_OP_SQR:             op_type = is_f16 ? "sqr-f16"      : "sqr-f32";          ਰੋਕੋ;
        ਹਾਲਤ HTP_OP_SQRT:            op_type = is_f16 ? "sqrt-f16"     : "sqrt-f32";         ਰੋਕੋ;
        ਹਾਲਤ HTP_OP_UNARY_NEG:       op_type = "neg-f32";                                    ਰੋਕੋ;
        ਹਾਲਤ HTP_OP_UNARY_EXP:       op_type = "exp-f32";                                    ਰੋਕੋ;
        ਹਾਲਤ HTP_OP_UNARY_SIGMOID:   op_type = "sigmoid-f32";                                ਰੋਕੋ;
        ਹਾਲਤ HTP_OP_UNARY_SILU:      op_type = "silu-f32";                                   ਰੋਕੋ;
        ਹਾਲਤ HTP_OP_UNARY_GELU:      op_type = "gelu-f32";                                   ਰੋਕੋ;
        ਹਾਲਤ HTP_OP_UNARY_GELU_ਤੇਜ਼: op_type = "gelu-quick-f32";                             ਰੋਕੋ;
        ਹਾਲਤ HTP_OP_UNARY_GELU_ERF:  op_type = "gelu-erf-f32";                               ਰੋਕੋ;
        ਹਾਲਤ HTP_OP_UNARY_SOFTPLUS:  op_type = "softplus-f32";                               ਰੋਕੋ;
        ਹਾਲਤ HTP_OP_UNARY_TANH:      op_type = "tanh-f32";                                   ਰੋਕੋ;
        ਹਾਲਤ HTP_OP_UNARY_ABS:       op_type = is_f16 ? "abs-f16"      : "abs-f32";          ਰੋਕੋ;
        ਹਾਲਤ HTP_OP_UNARY_LOG:       op_type = is_f16 ? "log-f16"      : "log-f32";          ਰੋਕੋ;
        ਹਾਲਤ HTP_OP_UNARY_RELU:      op_type = "relu-f32";                                   ਰੋਕੋ;
        ਹਾਲਤ HTP_OP_UNARY_ਕਦਮ:      op_type = is_f16 ? "step-f16"     : "step-f32";         ਰੋਕੋ;
        ਹਾਲਤ HTP_OP_L2_NORM:         op_type = is_f16 ? "l2norm-f16"   : "l2norm-f32";       ਰੋਕੋ;
        ਹਾਲਤ HTP_OP_TRI:             op_type = "tri-f32";                                    ਰੋਕੋ;
        ਹਾਲਤ HTP_OP_HADAMARD:        op_type = is_f16 ? "hadamard-f16" : "hadamard-f32";     ਰੋਕੋ;
        ਮੂਲ:
            FARF(ERROR, "Unsupported unary Op %u\n", octx->op);
            ਵਾਪਸ HTP_ਸਥਿਤੀ_NO_SUPPORT;
    }

    // F16 only has row-block kernels for this subset of ops (see the dispatch switch
    // below) - reject everything else up front, before touching kparams/VTCM.
    ਜੇ (is_f16) {
        ਚੋਣ (octx->op) {
            ਹਾਲਤ HTP_OP_NORM:
            ਹਾਲਤ HTP_OP_RMS_NORM:
            ਹਾਲਤ HTP_OP_SCALE:
            ਹਾਲਤ HTP_OP_CLAMP:
            ਹਾਲਤ HTP_OP_SQR:
            ਹਾਲਤ HTP_OP_SQRT:
            ਹਾਲਤ HTP_OP_L2_NORM:
            ਹਾਲਤ HTP_OP_UNARY_ABS:
            ਹਾਲਤ HTP_OP_UNARY_LOG:
            ਹਾਲਤ HTP_OP_UNARY_ਕਦਮ:
            ਹਾਲਤ HTP_OP_HADAMARD:
                ਰੋਕੋ;
            ਮੂਲ:
                FARF(ERROR, "unary-%s: not supported for F16\n", op_type);
                ਵਾਪਸ HTP_ਸਥਿਤੀ_NO_SUPPORT;
        }
    }

    ਅਚਲ ਬਣਤਰ htp_unary_kernel_params * kparams = (ਅਚਲ ਬਣਤਰ htp_unary_kernel_params *) octx->kernel_params;

    ਜੇ (!htp_ops_context_ਸੈੱਟ_n_ਧਾਗੇ(octx, kparams->n_ਧਾਗੇ)) {
        ਵਾਪਸ HTP_ਸਥਿਤੀ_INVAL_PARAMS;
    }

    ਅਚਲ uint32_t src0_nrows = src0->ne[1] * src0->ne[2] * src0->ne[3];
    ਅਚਲ size_t src0_elem_ਮਾਪ = is_f16 ? ਆਕਾਰ(_Float16) : ਆਕਾਰ(ਦਸ਼ਮਲਵ);
    ਅਚਲ size_t dst_elem_ਮਾਪ  = (dst->type == HTP_TYPE_F16) ? ਆਕਾਰ(_Float16) : ਆਕਾਰ(ਦਸ਼ਮਲਵ);
    ਅਚਲ size_t src0_ਅੰਕੜੇ_ਪੰਕਤੀ_ਮਾਪ = src0->ne[0] * src0_elem_ਮਾਪ;
    ਅਚਲ size_t dst_ਅੰਕੜੇ_ਪੰਕਤੀ_ਮਾਪ  = dst->ne[0]  * dst_elem_ਮਾਪ;

    uint32_t ਪੰਕਤੀ_ਸ਼ੁਰੂ = 0;
    uint32_t nrows     = src0_nrows;

    ਜੇ (octx->ctx->mdev.ਗਿਣਤੀ_2 > 1) {
        uint32_t ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_chunk = 0;
        htp_tensor_mdev_ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_chunk(dst, (uint32_t) dst_elem_ਮਾਪ, (uint32_t) dst_ਅੰਕੜੇ_ਪੰਕਤੀ_ਮਾਪ, &ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_chunk);
        ਅਚਲ ਬਣਤਰ htp_tensor_mdev_range range = htp_tensor_mdev_ਹਿੱਸੇ_ਕਰੋ(src0_nrows, ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_chunk, octx->ctx->mdev.idx, octx->ctx->mdev.ਗਿਣਤੀ_2, &octx->ctx->mdev.ਗਿਣਤੀ_div);
        ਪੰਕਤੀ_ਸ਼ੁਰੂ = range.ਸ਼ੁਰੂ;
        nrows     = range.ਗਿਣਤੀ_2;
    }

    ਜੇ (nrows == 0) {
        ਵਾਪਸ HTP_ਸਥਿਤੀ_OK;
    }

    ਅਚਲ uint32_t n_ਧਾਗੇ = octx->n_ਧਾਗੇ;

    ਅਚਲ size_t src0_ਪੰਕਤੀ_ਮਾਪ_aligned = kparams->src0_ਪੰਕਤੀ_ਮਾਪ_aligned;
    ਅਚਲ size_t dst_ਪੰਕਤੀ_ਮਾਪ_aligned  = kparams->dst_ਪੰਕਤੀ_ਮਾਪ_aligned;

    // Always 0 for F16 - htp_unary_vtcm_layout_build() keeps F16 on the row-block path,
    // since only F32 has unary_task_f32_tiled_* kernels.
    ਅਚਲ uint32_t ਖਾਨਾ_tile = kparams->ਖਾਨਾ_tile;

    size_t src1_ਅੰਕੜੇ_ਪੰਕਤੀ_ਮਾਪ = 0;
    size_t src1_ਪੰਕਤੀ_ਮਾਪ_aligned = kparams->src1_ਪੰਕਤੀ_ਮਾਪ_aligned;
    bool ਪ੍ਰਸਾਰਣ_ਭਾਰ = kparams->ਪ੍ਰਸਾਰਣ_ਭਾਰ;
    ਅਚਲ ਬਣਤਰ htp_tensor * src1 = ਨਿਰਾ;

    // RMS_NORM_MUL fusion is F32-only (its weight tensor is always F32; see
    // try_fuse_node()'s type guard), so this never triggers when is_f16 is true.
    ਜੇ (octx->op == HTP_OP_RMS_NORM_MUL) {
        src1 = octx->src[1];
        src1_ਅੰਕੜੇ_ਪੰਕਤੀ_ਮਾਪ = src1->ne[0] * ਆਕਾਰ(ਦਸ਼ਮਲਵ);
    }

    ਜੇ (octx->ctx->vtcm_ਮਾਪ < (size_t)kparams->vtcm_ਮਾਪ) {
        FARF(ERROR, "unary-%s : current VTCM reservation %zu is too small, needed %zu\n", op_type, octx->ctx->vtcm_ਮਾਪ, (size_t)kparams->vtcm_ਮਾਪ);
        ਵਾਪਸ HTP_ਸਥਿਤੀ_VTCM_TOO_SMALL;
    }

    octx->src0_spad.src = ਨਿਰਾ;
    octx->src1_spad.src = ਨਿਰਾ;
    octx->dst_spad.src  = ਨਿਰਾ;

    FARF(ਉੱਚਾ_2, "%s: (%ux%ux%ux%u) -> (%ux%ux%ux%u) : src0-vtcm-size %u src1-vtcm-size %u dst-vtcm-size %u\n", op_type,
         src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3], dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3],
         kparams->vtcm_src0_ਮਾਪ, kparams->vtcm_src1_ਮਾਪ, kparams->vtcm_dst_ਮਾਪ);

    uint8_t * ਅਚਲ ਅਧਾਰ = (uint8_t *) octx->ctx->vtcm_ਅਧਾਰ;
    ਬਣਤਰ htp_unary_context uctx = {
        .octx                  = octx,
        .kparams               = kparams,
        .src0_nrows_ਪ੍ਰਤੀ_thread = fastdiv(nrows + n_ਧਾਗੇ - 1, &octx->n_ਧਾਗੇ_div),
        .src0_nrows            = nrows,
        .ਪੰਕਤੀ_ਸ਼ੁਰੂ             = ਪੰਕਤੀ_ਸ਼ੁਰੂ,

        .ਅੰਕੜੇ_src0             = src0->ਅੰਕੜੇ,
        .ਅੰਕੜੇ_src1             = (octx->op == HTP_OP_RMS_NORM_MUL) ? src1->ਅੰਕੜੇ : 0,
        .ਅੰਕੜੇ_dst              = dst->ਅੰਕੜੇ,

        .src0_ਅੰਕੜੇ_ਪੰਕਤੀ_ਮਾਪ    = src0_ਅੰਕੜੇ_ਪੰਕਤੀ_ਮਾਪ,
        .src1_ਅੰਕੜੇ_ਪੰਕਤੀ_ਮਾਪ    = src1_ਅੰਕੜੇ_ਪੰਕਤੀ_ਮਾਪ,
        .dst_ਅੰਕੜੇ_ਪੰਕਤੀ_ਮਾਪ     = dst_ਅੰਕੜੇ_ਪੰਕਤੀ_ਮਾਪ,

        .src0_ਪੰਕਤੀ_ਮਾਪ_aligned = src0_ਪੰਕਤੀ_ਮਾਪ_aligned,
        .src1_ਪੰਕਤੀ_ਮਾਪ_aligned = src1_ਪੰਕਤੀ_ਮਾਪ_aligned,
        .dst_ਪੰਕਤੀ_ਮਾਪ_aligned  = dst_ਪੰਕਤੀ_ਮਾਪ_aligned,

        .src0_vtcm_half_ਮਾਪ   = kparams->vtcm_src0_ਮਾਪ_ਪ੍ਰਤੀ_thread / 2,
        .src1_vtcm_half_ਮਾਪ   = (octx->op == HTP_OP_RMS_NORM_MUL) ? (kparams->vtcm_src1_ਮਾਪ_ਪ੍ਰਤੀ_thread / (ਪ੍ਰਸਾਰਣ_ਭਾਰ ? 1 : 2)) : 0,
        .dst_vtcm_half_ਮਾਪ    = kparams->vtcm_dst_ਮਾਪ_ਪ੍ਰਤੀ_thread / 2,

        .ਖੰਡ                 = kparams->ਖੰਡ,
        .nc                    = src0->ne[0],
        .ਖਾਨਾ_tile              = ਖਾਨਾ_tile,
        .ਪ੍ਰਸਾਰਣ_ਭਾਰ      = ਪ੍ਰਸਾਰਣ_ਭਾਰ,

        .vtcm_src0             = VTCM_LAYOUT_PTR(uint8_t, ਅਧਾਰ, 0),
        .vtcm_src1             = VTCM_LAYOUT_PTR_OPTIONAL(uint8_t, ਅਧਾਰ, kparams->vtcm_src0_ਮਾਪ, kparams->vtcm_src1_ਮਾਪ > 0),
        .vtcm_dst              = VTCM_LAYOUT_PTR(uint8_t, ਅਧਾਰ, kparams->vtcm_src0_ਮਾਪ + kparams->vtcm_src1_ਮਾਪ),

        .vtcm_src0_ਮਾਪ_ਪ੍ਰਤੀ_thread = kparams->vtcm_src0_ਮਾਪ_ਪ੍ਰਤੀ_thread,
        .vtcm_src1_ਮਾਪ_ਪ੍ਰਤੀ_thread = kparams->vtcm_src1_ਮਾਪ_ਪ੍ਰਤੀ_thread,
        .vtcm_dst_ਮਾਪ_ਪ੍ਰਤੀ_thread  = kparams->vtcm_dst_ਮਾਪ_ਪ੍ਰਤੀ_thread,
    };

    FARF(ਉੱਚਾ_2, "%s: %s mode (col_tile %u)\n", op_type, ਖਾਨਾ_tile ? "tiled" : "row-block", ਖਾਨਾ_tile);

    ਕਾਮਾ_callback_t task_func = ਨਿਰਾ;
    ਖਾਲੀ * compute_func = ਨਿਰਾ;

    ਜੇ (ਖਾਨਾ_tile) {
        task_func = unary_thread_tiled;
        ਚੋਣ (octx->op) {
            ਹਾਲਤ HTP_OP_SCALE:           compute_func = (ਖਾਲੀ *) tile_scale_f32;          ਰੋਕੋ;
            ਹਾਲਤ HTP_OP_CLAMP:           compute_func = (ਖਾਲੀ *) tile_clamp_f32;          ਰੋਕੋ;
            ਹਾਲਤ HTP_OP_LEAKY_RELU:      compute_func = (ਖਾਲੀ *) tile_leaky_relu_f32;     ਰੋਕੋ;
            ਹਾਲਤ HTP_OP_SQR:             compute_func = (ਖਾਲੀ *) tile_sqr_f32;            ਰੋਕੋ;
            ਹਾਲਤ HTP_OP_SQRT:            compute_func = (ਖਾਲੀ *) tile_sqrt_f32;           ਰੋਕੋ;
            ਹਾਲਤ HTP_OP_UNARY_NEG:       compute_func = (ਖਾਲੀ *) tile_neg_f32;            ਰੋਕੋ;
            ਹਾਲਤ HTP_OP_UNARY_EXP:       compute_func = (ਖਾਲੀ *) tile_exp_f32;            ਰੋਕੋ;
            ਹਾਲਤ HTP_OP_UNARY_SIGMOID:   compute_func = (ਖਾਲੀ *) tile_sigmoid_f32;        ਰੋਕੋ;
            ਹਾਲਤ HTP_OP_UNARY_SILU:      compute_func = (ਖਾਲੀ *) tile_silu_f32;           ਰੋਕੋ;
            ਹਾਲਤ HTP_OP_UNARY_GELU:      compute_func = (ਖਾਲੀ *) tile_gelu_f32;           ਰੋਕੋ;
            ਹਾਲਤ HTP_OP_UNARY_GELU_ਤੇਜ਼: compute_func = (ਖਾਲੀ *) tile_gelu_ਤੇਜ਼_f32;     ਰੋਕੋ;
            ਹਾਲਤ HTP_OP_UNARY_GELU_ERF:  compute_func = (ਖਾਲੀ *) tile_gelu_erf_f32;       ਰੋਕੋ;
            ਹਾਲਤ HTP_OP_UNARY_SOFTPLUS:  compute_func = (ਖਾਲੀ *) tile_softplus_f32;       ਰੋਕੋ;
            ਹਾਲਤ HTP_OP_UNARY_TANH:      compute_func = (ਖਾਲੀ *) tile_tanh_f32;           ਰੋਕੋ;
            ਹਾਲਤ HTP_OP_UNARY_ABS:       compute_func = (ਖਾਲੀ *) tile_abs_f32;            ਰੋਕੋ;
            ਹਾਲਤ HTP_OP_UNARY_LOG:       compute_func = (ਖਾਲੀ *) tile_log_f32;            ਰੋਕੋ;
            ਹਾਲਤ HTP_OP_UNARY_RELU:      compute_func = (ਖਾਲੀ *) tile_relu_f32;           ਰੋਕੋ;
            ਹਾਲਤ HTP_OP_UNARY_ਕਦਮ:      compute_func = (ਖਾਲੀ *) tile_ਕਦਮ_f32;           ਰੋਕੋ;
            ਹਾਲਤ HTP_OP_TRI:
                task_func    = unary_thread_tiled_tri_f32;
                compute_func = (ਖਾਲੀ *) tri_apply_tile_f32;
                ਰੋਕੋ;
            ਮੂਲ:                     ਰੋਕੋ;
        }
    } ਨਹੀਂਤਾਂ ਜੇ (is_f16) {
        task_func = unary_thread_ਪੰਕਤੀ_ਖੰਡ;
        ਚੋਣ (octx->op) {
            ਹਾਲਤ HTP_OP_NORM:            compute_func = (ਖਾਲੀ *) norm_f16;                ਰੋਕੋ;
            ਹਾਲਤ HTP_OP_RMS_NORM:        compute_func = (ਖਾਲੀ *) rms_norm_f16;            ਰੋਕੋ;
            ਹਾਲਤ HTP_OP_SCALE:           compute_func = (ਖਾਲੀ *) scale_f16;               ਰੋਕੋ;
            ਹਾਲਤ HTP_OP_CLAMP:           compute_func = (ਖਾਲੀ *) clamp_f16;               ਰੋਕੋ;
            ਹਾਲਤ HTP_OP_SQR:             compute_func = (ਖਾਲੀ *) sqr_f16;                 ਰੋਕੋ;
            ਹਾਲਤ HTP_OP_SQRT:            compute_func = (ਖਾਲੀ *) sqrt_f16;                ਰੋਕੋ;
            ਹਾਲਤ HTP_OP_L2_NORM:         compute_func = (ਖਾਲੀ *) l2_norm_f16;             ਰੋਕੋ;
            ਹਾਲਤ HTP_OP_UNARY_ABS:       compute_func = (ਖਾਲੀ *) abs_f16;                 ਰੋਕੋ;
            ਹਾਲਤ HTP_OP_UNARY_LOG:       compute_func = (ਖਾਲੀ *) log_f16;                 ਰੋਕੋ;
            ਹਾਲਤ HTP_OP_UNARY_ਕਦਮ:      compute_func = (ਖਾਲੀ *) ਕਦਮ_f16;                ਰੋਕੋ;
            ਹਾਲਤ HTP_OP_HADAMARD:        compute_func = (ਖਾਲੀ *) hadamard_f16;            ਰੋਕੋ;
            ਮੂਲ:                     ਰੋਕੋ;
        }
    } ਨਹੀਂਤਾਂ {
        task_func = unary_thread_ਪੰਕਤੀ_ਖੰਡ;
        ਚੋਣ (octx->op) {
            ਹਾਲਤ HTP_OP_NORM:            compute_func = (ਖਾਲੀ *) norm_f32;                ਰੋਕੋ;
            ਹਾਲਤ HTP_OP_RMS_NORM:        compute_func = (ਖਾਲੀ *) rms_norm_f32;            ਰੋਕੋ;
            ਹਾਲਤ HTP_OP_RMS_NORM_MUL:
                task_func    = unary_thread_rms_norm_mul_f32;
                compute_func = (ਖਾਲੀ *) rms_norm_mul_f32;
                ਰੋਕੋ;
            ਹਾਲਤ HTP_OP_SCALE:           compute_func = (ਖਾਲੀ *) scale_f32;               ਰੋਕੋ;
            ਹਾਲਤ HTP_OP_CLAMP:           compute_func = (ਖਾਲੀ *) clamp_f32;               ਰੋਕੋ;
            ਹਾਲਤ HTP_OP_LEAKY_RELU:      compute_func = (ਖਾਲੀ *) leaky_relu_f32;          ਰੋਕੋ;
            ਹਾਲਤ HTP_OP_SQR:             compute_func = (ਖਾਲੀ *) sqr_f32;                 ਰੋਕੋ;
            ਹਾਲਤ HTP_OP_SQRT:            compute_func = (ਖਾਲੀ *) sqrt_f32;                ਰੋਕੋ;
            ਹਾਲਤ HTP_OP_UNARY_NEG:       compute_func = (ਖਾਲੀ *) neg_f32;                 ਰੋਕੋ;
            ਹਾਲਤ HTP_OP_UNARY_EXP:       compute_func = (ਖਾਲੀ *) exp_f32;                 ਰੋਕੋ;
            ਹਾਲਤ HTP_OP_UNARY_SIGMOID:   compute_func = (ਖਾਲੀ *) sigmoid_f32;             ਰੋਕੋ;
            ਹਾਲਤ HTP_OP_UNARY_SILU:      compute_func = (ਖਾਲੀ *) silu_f32;                ਰੋਕੋ;
            ਹਾਲਤ HTP_OP_UNARY_GELU:      compute_func = (ਖਾਲੀ *) gelu_f32;                ਰੋਕੋ;
            ਹਾਲਤ HTP_OP_UNARY_GELU_ਤੇਜ਼: compute_func = (ਖਾਲੀ *) gelu_ਤੇਜ਼_f32;          ਰੋਕੋ;
            ਹਾਲਤ HTP_OP_UNARY_GELU_ERF:  compute_func = (ਖਾਲੀ *) gelu_erf_f32;            ਰੋਕੋ;
            ਹਾਲਤ HTP_OP_UNARY_SOFTPLUS:  compute_func = (ਖਾਲੀ *) softplus_f32;            ਰੋਕੋ;
            ਹਾਲਤ HTP_OP_UNARY_TANH:      compute_func = (ਖਾਲੀ *) tanh_f32;                ਰੋਕੋ;
            ਹਾਲਤ HTP_OP_UNARY_ABS:       compute_func = (ਖਾਲੀ *) abs_f32;                 ਰੋਕੋ;
            ਹਾਲਤ HTP_OP_UNARY_LOG:       compute_func = (ਖਾਲੀ *) log_f32;                 ਰੋਕੋ;
            ਹਾਲਤ HTP_OP_UNARY_RELU:      compute_func = (ਖਾਲੀ *) relu_f32;                ਰੋਕੋ;
            ਹਾਲਤ HTP_OP_UNARY_ਕਦਮ:      compute_func = (ਖਾਲੀ *) ਕਦਮ_f32;                ਰੋਕੋ;
            ਹਾਲਤ HTP_OP_L2_NORM:         compute_func = (ਖਾਲੀ *) l2_norm_f32;             ਰੋਕੋ;
            ਹਾਲਤ HTP_OP_HADAMARD:        compute_func = (ਖਾਲੀ *) hadamard_f32;            ਰੋਕੋ;
            ਹਾਲਤ HTP_OP_TRI:
                task_func    = unary_thread_tri_f32;
                compute_func = (ਖਾਲੀ *) tri_f32;
                ਰੋਕੋ;
            ਮੂਲ:                     ਰੋਕੋ;
        }
    }

    ਜੇ (!task_func || !compute_func) {
        FARF(ERROR, "execute_op_unary: task function is NULL for op %d\n", octx->op);
        ਵਾਪਸ HTP_ਸਥਿਤੀ_NO_SUPPORT;
    }

    uctx.compute = compute_func;
    work_ਕਤਾਰ_ਲੜੀਵਾਰ(octx->ctx->work_ਕਤਾਰ, task_func, &uctx, n_ਧਾਗੇ);

    ਵਾਪਸ err;
}

ਪੂਰਨ op_unary(ਬਣਤਰ htp_ops_context * octx) {
    ਚੋਣ (octx->src[0]->type) {
        ਹਾਲਤ HTP_TYPE_F32:
        ਹਾਲਤ HTP_TYPE_F16:
            ਵਾਪਸ execute_op_unary(octx);

        ਮੂਲ:
            ਵਾਪਸ HTP_ਸਥਿਤੀ_NO_SUPPORT;
    }
}
