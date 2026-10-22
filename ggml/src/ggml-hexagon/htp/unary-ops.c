#ಸೂಚನೆ clang diagnostic ignored "-Wunused-variable"
#ಸೂಚನೆ clang diagnostic ignored "-Wunused-function"
#ಸೂಚನೆ clang diagnostic ignored "-Wunused-but-set-variable"

#ಅನ್ತರ್ಭಾವ <HAP_farf.h>
#ಅನ್ತರ್ಭಾವ <HAP_perf.h>

#ಅನ್ತರ್ಭಾವ <math.h>
#ಅನ್ತರ್ಭಾವ <string.h>

#ಅನ್ತರ್ಭಾವ "dma-queue.h"
#ಅನ್ತರ್ಭಾವ "hex-fastdiv.h"
#ಅನ್ತರ್ಭಾವ "hvx-exp.h"
#ಅನ್ತರ್ಭಾವ "hvx-sigmoid.h"
#ಅನ್ತರ್ಭಾವ "hvx-utils.h"
#ಅನ್ತರ್ಭಾವ "unary-ops.h"

#ವ್ಯಾಖ್ಯಾನಿಸು GGML_COMMON_DECL_C
#ಅನ್ತರ್ಭಾವ "ggml-common.h"
#ಅನ್ತರ್ಭಾವ "htp-ctx.h"
#ಅನ್ತರ್ಭಾವ "htp-ops.h"
#ಅನ್ತರ್ಭಾವ "htp-tensor.h"
#ಅನ್ತರ್ಭಾವ "htp-vtcm.h"
#ಅನ್ತರ್ಭಾವ "hex-profile.h"

ರಚನೆ htp_unary_context;

ಪ್ರಕಾರವ್ಯಾಖ್ಯೆ ಶೂನ್ಯ (*unary_compute_fn_t)(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * ಸೀಮಿತ src,
                                   ಶೂನ್ಯ * ಸೀಮಿತ dst,
                                   uint32_t ಅಂಕೆ_ಅಡ್ಡಸಾಲುಗಳು_2,
                                   ಸ್ಥಿರಾಂಕ ರಚನೆ htp_unary_context * uctx);

ಪ್ರಕಾರವ್ಯಾಖ್ಯೆ ಶೂನ್ಯ (*unary_rms_norm_mul_compute_fn_t)(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * ಸೀಮಿತ src,
                                                ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * ಸೀಮಿತ ತೂಕ,
                                                ಶೂನ್ಯ * ಸೀಮಿತ dst,
                                                uint32_t ಅಂಕೆ_ಅಡ್ಡಸಾಲುಗಳು_2,
                                                ಸ್ಥಿರಾಂಕ ರಚನೆ htp_unary_context * uctx);

ಪ್ರಕಾರವ್ಯಾಖ್ಯೆ ಶೂನ್ಯ (*unary_tri_compute_fn_t)(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * ಸೀಮಿತ src,
                                       ಶೂನ್ಯ * ಸೀಮಿತ dst,
                                       uint32_t ಅಂಕೆ_ಅಡ್ಡಸಾಲುಗಳು_2,
                                       uint32_t ir,
                                       ಸ್ಥಿರಾಂಕ ರಚನೆ htp_unary_context * uctx);

ಪ್ರಕಾರವ್ಯಾಖ್ಯೆ ಶೂನ್ಯ (*unary_tile_compute_fn_t)(ಶೂನ್ಯ * ಸೀಮಿತ dst,
                                        ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * ಸೀಮಿತ src,
                                        uint32_t tw,
                                        ಸ್ಥಿರಾಂಕ ರಚನೆ htp_unary_context * uctx);

ಪ್ರಕಾರವ್ಯಾಖ್ಯೆ ಶೂನ್ಯ (*unary_tiled_tri_compute_fn_t)(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * ಸೀಮಿತ src,
                                             ಶೂನ್ಯ * ಸೀಮಿತ dst,
                                             uint32_t tile_elems,
                                             uint32_t ಲಂಬಸಾಲು_ಆರಂಭ,
                                             uint32_t i01,
                                             uint32_t ne0,
                                             int32_t ttype);

ರಚನೆ htp_unary_context {
    ರಚನೆ htp_ops_context * octx;
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_unary_kernel_params * kparams;

    ಶೂನ್ಯ *                    compute;

    dma_addr_t                ದತ್ತಾಂಶ_src0;
    dma_addr_t                ದತ್ತಾಂಶ_src1;            // weight/scale tensor for RMS_NORM_MUL
    dma_addr_t                ದತ್ತಾಂಶ_dst;

    size_t                    src0_ದತ್ತಾಂಶ_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ;   // actual data bytes per row
    size_t                    src1_ದತ್ತಾಂಶ_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ;
    size_t                    dst_ದತ್ತಾಂಶ_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ;    // actual data bytes per row

    size_t                    src0_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned;
    size_t                    src1_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned;
    size_t                    dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned;

    size_t                    src0_vtcm_half_ಗಾತ್ರ;
    size_t                    src1_vtcm_half_ಗಾತ್ರ;
    size_t                    dst_vtcm_half_ಗಾತ್ರ;

    uint32_t                  ಖಂಡ;
    uint32_t                  src0_nrows;
    uint32_t                  src0_nrows_ಪ್ರತಿ_thread;
    uint32_t                  ಅಡ್ಡಸಾಲು_ಆರಂಭ;
    uint32_t                  nc;
    uint32_t                  ಲಂಬಸಾಲು_tile;             // tiled mode
    bool                      ಪ್ರಸಾರ_ತೂಕ;

    uint8_t *                 vtcm_src0;
    uint8_t *                 vtcm_src1;
    uint8_t *                 vtcm_dst;

    size_t                    vtcm_src0_ಗಾತ್ರ_ಪ್ರತಿ_thread;
    size_t                    vtcm_src1_ಗಾತ್ರ_ಪ್ರತಿ_thread;
    size_t                    vtcm_dst_ಗಾತ್ರ_ಪ್ರತಿ_thread;
};

// Convert flat row index to DDR byte offset using the tensor's actual strides.
// ir = i1 + ne1*(i2 + ne2*i3)  =>  offset = i1*nb1 + i2*nb2 + i3*nb3
ಸ್ಥಿರ ಸಾಲೊಳಗೆ size_t unary_ಅಡ್ಡಸಾಲು_ಸ್ಥಾನಾಂತರ(uint32_t ir,
                                      uint32_t ne1, uint32_t ne2,
                                      ಸ್ಥಿರಾಂಕ ರಚನೆ fastdiv_ಮೌಲ್ಯಗಳು * div_ne1,
                                      ಸ್ಥಿರಾಂಕ ರಚನೆ fastdiv_ಮೌಲ್ಯಗಳು * div_ne2,
                                      ಸ್ಥಿರಾಂಕ ರಚನೆ fastdiv_ಮೌಲ್ಯಗಳು * div_ne12,
                                      size_t nb1, size_t nb2, size_t nb3) {
    ಸ್ಥಿರಾಂಕ uint32_t i1 = fastmodulo(ir, ne1, div_ne1);
    ಸ್ಥಿರಾಂಕ uint32_t ir_div_ne1 = fastdiv(ir, div_ne1);
    ಸ್ಥಿರಾಂಕ uint32_t i2 = fastmodulo(ir_div_ne1, ne2, div_ne2);
    ಸ್ಥಿರಾಂಕ uint32_t i3 = fastdiv(ir, div_ne12);
    ಹಿಂತಿರುಗಿ i1 * nb1 + i2 * nb2 + i3 * nb3;
}

// Safe DMA block size from row `ir`: clamp to the tighter dim-1 slice
// boundary of src and dst so the nb1 stride stays valid for all rows.
ಸ್ಥಿರ ಸಾಲೊಳಗೆ uint32_t unary_ಖಂಡ_ಗಾತ್ರ(uint32_t ir,
                                        uint32_t ಕೊನೆ_ಅಡ್ಡಸಾಲು,
                                        uint32_t ಖಂಡ,
                                        bool src_contig,
                                        bool dst_contig,
                                        uint32_t ne1,
                                        ಸ್ಥಿರಾಂಕ ರಚನೆ fastdiv_ಮೌಲ್ಯಗಳು * div_ne1) {
    uint32_t ಎಲ್ಲೆ = MIN(ಖಂಡ, ಕೊನೆ_ಅಡ್ಡಸಾಲು - ir);

    ಆದರೆ (!src_contig || !dst_contig) {
        ಸ್ಥಿರಾಂಕ uint32_t ತುಂಡು_ಕೊನೆ = (fastdiv(ir, div_ne1) + 1) * ne1;
        ಎಲ್ಲೆ = MIN(ಎಲ್ಲೆ, ತುಂಡು_ಕೊನೆ - ir);
    }

    ಹಿಂತಿರುಗಿ ಎಲ್ಲೆ;
}

#ವ್ಯಾಖ್ಯಾನಿಸು htp_unary_preamble            \
    ಸ್ಥಿರಾಂಕ uint32_t ne00 = src->ne[0]; \
    ಸ್ಥಿರಾಂಕ uint32_t ne01 = src->ne[1]; \
    ಸ್ಥಿರಾಂಕ uint32_t ne02 = src->ne[2]; \
    ಸ್ಥಿರಾಂಕ uint32_t ne03 = src->ne[3]; \
                                      \
    ಸ್ಥಿರಾಂಕ uint32_t ne0 = dst->ne[0];  \
    ಸ್ಥಿರಾಂಕ uint32_t ne1 = dst->ne[1];  \
    ಸ್ಥಿರಾಂಕ uint32_t ne2 = dst->ne[2];  \
    ಸ್ಥಿರಾಂಕ uint32_t ne3 = dst->ne[3];  \
                                      \
    ಸ್ಥಿರಾಂಕ uint32_t nb00 = src->nb[0]; \
    ಸ್ಥಿರಾಂಕ uint32_t nb01 = src->nb[1]; \
    ಸ್ಥಿರಾಂಕ uint32_t nb02 = src->nb[2]; \
    ಸ್ಥಿರಾಂಕ uint32_t nb03 = src->nb[3]; \
                                      \
    ಸ್ಥಿರಾಂಕ uint32_t nb0 = dst->nb[0];  \
    ಸ್ಥಿರಾಂಕ uint32_t nb1 = dst->nb[1];  \
    ಸ್ಥಿರಾಂಕ uint32_t nb2 = dst->nb[2];  \
    ಸ್ಥಿರಾಂಕ uint32_t nb3 = dst->nb[3];

#ವ್ಯಾಖ್ಯಾನಿಸು htp_unary_op_preamble                                         \
    int32_t * op_params = uctx->octx->op_params;                      \
    ಸ್ಥಿರಾಂಕ uint32_t ne0 = uctx->nc;                                    \
    ಸ್ಥಿರಾಂಕ size_t src0_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned = uctx->src0_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned; \
    ಸ್ಥಿರಾಂಕ size_t dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned = uctx->dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned;

ಸ್ಥಿರ ಶೂನ್ಯ scale_f32(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * ಸೀಮಿತ src,
                      ಶೂನ್ಯ * ಸೀಮಿತ dst,
                      ಸ್ಥಿರಾಂಕ uint32_t ಅಂಕೆ_ಅಡ್ಡಸಾಲುಗಳು_2,
                      ಸ್ಥಿರಾಂಕ ರಚನೆ htp_unary_context * uctx) {
    htp_unary_op_preamble;
    ದಶಮಾಂಶ scale = 0.f;
    ದಶಮಾಂಶ bias  = 0.f;
    ಸ್ಮರಣೆ_ನಕಲು(&scale, &op_params[0], ಗಾತ್ರ(ದಶಮಾಂಶ));
    ಸ್ಮರಣೆ_ನಕಲು(&bias,  &op_params[1], ಗಾತ್ರ(ದಶಮಾಂಶ));

    ಗಾಗಿ (uint32_t ir = 0; ir < ಅಂಕೆ_ಅಡ್ಡಸಾಲುಗಳು_2; ir++) {
        ಸ್ಥಿರಾಂಕ uint8_t * ಸೀಮಿತ src_local = (ಸ್ಥಿರಾಂಕ uint8_t *)src + (ir * src0_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned);
        uint8_t * ಸೀಮಿತ dst_local       = (uint8_t *)dst + (ir * dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned);

        hvx_scale_ಸ್ಥಾನಾಂತರ_f32_aa((uint8_t *) dst_local, (ಸ್ಥಿರಾಂಕ uint8_t *) src_local, ne0, scale, bias);
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ clamp_f32(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * ಸೀಮಿತ src,
                      ಶೂನ್ಯ * ಸೀಮಿತ dst,
                      ಸ್ಥಿರಾಂಕ uint32_t ಅಂಕೆ_ಅಡ್ಡಸಾಲುಗಳು_2,
                      ಸ್ಥಿರಾಂಕ ರಚನೆ htp_unary_context * uctx) {
    htp_unary_op_preamble;
    ದಶಮಾಂಶ min = 0.f;
    ದಶಮಾಂಶ max = 0.f;
    ಸ್ಮರಣೆ_ನಕಲು(&min, &op_params[0], ಗಾತ್ರ(ದಶಮಾಂಶ));
    ಸ್ಮರಣೆ_ನಕಲು(&max, &op_params[1], ಗಾತ್ರ(ದಶಮಾಂಶ));

    ಗಾಗಿ (uint32_t ir = 0; ir < ಅಂಕೆ_ಅಡ್ಡಸಾಲುಗಳು_2; ir++) {
        ಸ್ಥಿರಾಂಕ uint8_t * ಸೀಮಿತ src_local = (ಸ್ಥಿರಾಂಕ uint8_t *)src + (ir * src0_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned);
        uint8_t * ಸೀಮಿತ dst_local       = (uint8_t *)dst + (ir * dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned);

        hvx_clamp_scalar_f32(dst_local, src_local, min, max, ne0);
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ leaky_relu_f32(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * ಸೀಮಿತ src,
                           ಶೂನ್ಯ * ಸೀಮಿತ dst,
                           ಸ್ಥಿರಾಂಕ uint32_t ಅಂಕೆ_ಅಡ್ಡಸಾಲುಗಳು_2,
                           ಸ್ಥಿರಾಂಕ ರಚನೆ htp_unary_context * uctx) {
    htp_unary_op_preamble;
    ದಶಮಾಂಶ negative_ಇಳಿಜಾರು = 0.f;
    ಸ್ಮರಣೆ_ನಕಲು(&negative_ಇಳಿಜಾರು, &op_params[0], ಗಾತ್ರ(ದಶಮಾಂಶ));

    ಗಾಗಿ (uint32_t ir = 0; ir < ಅಂಕೆ_ಅಡ್ಡಸಾಲುಗಳು_2; ir++) {
        ಸ್ಥಿರಾಂಕ uint8_t * ಸೀಮಿತ src_local = (ಸ್ಥಿರಾಂಕ uint8_t *)src + (ir * src0_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned);
        uint8_t * ಸೀಮಿತ dst_local       = (uint8_t *)dst + (ir * dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned);

        hvx_leaky_relu_scalar_f32(dst_local, src_local, negative_ಇಳಿಜಾರು, ne0);
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ rms_norm_f32(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * ಸೀಮಿತ src,
                         ಶೂನ್ಯ * ಸೀಮಿತ dst,
                         ಸ್ಥಿರಾಂಕ uint32_t ಅಂಕೆ_ಅಡ್ಡಸಾಲುಗಳು_2,
                         ಸ್ಥಿರಾಂಕ ರಚನೆ htp_unary_context * uctx) {
    htp_unary_op_preamble;
    ದಶಮಾಂಶ epsilon = 0.f;
    ಸ್ಮರಣೆ_ನಕಲು(&epsilon, op_params, ಗಾತ್ರ(ದಶಮಾಂಶ));

    ಗಾಗಿ (uint32_t ir = 0; ir < ಅಂಕೆ_ಅಡ್ಡಸಾಲುಗಳು_2; ir++) {
        ಸ್ಥಿರಾಂಕ uint8_t * ಸೀಮಿತ src_local = (ಸ್ಥಿರಾಂಕ uint8_t *)src + (ir * src0_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned);
        uint8_t * ಸೀಮಿತ dst_local       = (uint8_t *)dst + (ir * dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned);

        hvx_fast_rms_norm_f32((ಸ್ಥಿರಾಂಕ uint8_t *) src_local, (uint8_t *) dst_local, ne0, epsilon);
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ rms_norm_mul_f32(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * ಸೀಮಿತ src,
                             ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * ಸೀಮಿತ ತೂಕ,
                             ಶೂನ್ಯ * ಸೀಮಿತ dst,
                             ಸ್ಥಿರಾಂಕ uint32_t ಅಂಕೆ_ಅಡ್ಡಸಾಲುಗಳು_2,
                             ಸ್ಥಿರಾಂಕ ರಚನೆ htp_unary_context * uctx) {
    htp_unary_op_preamble;
    ದಶಮಾಂಶ epsilon = 0.f;
    ಸ್ಮರಣೆ_ನಕಲು(&epsilon, op_params, ಗಾತ್ರ(ದಶಮಾಂಶ));

    ಗಾಗಿ (uint32_t ir = 0; ir < ಅಂಕೆ_ಅಡ್ಡಸಾಲುಗಳು_2; ir++) {
        ಸ್ಥಿರಾಂಕ uint8_t * ಸೀಮಿತ src_local = (ಸ್ಥಿರಾಂಕ uint8_t *)src + (ir * src0_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned);
        ಸ್ಥಿರಾಂಕ uint8_t * ಸೀಮಿತ w_local   = (ಸ್ಥಿರಾಂಕ uint8_t *)ತೂಕ + (uctx->ಪ್ರಸಾರ_ತೂಕ ? 0 : ir * uctx->src1_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned);
        uint8_t * ಸೀಮಿತ dst_local       = (uint8_t *)dst + (ir * dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned);

        hvx_fast_rms_norm_mul_f32(src_local, w_local, dst_local, ne0, epsilon);
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ norm_f32(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * ಸೀಮಿತ src,
                     ಶೂನ್ಯ * ಸೀಮಿತ dst,
                     ಸ್ಥಿರಾಂಕ uint32_t ಅಂಕೆ_ಅಡ್ಡಸಾಲುಗಳು_2,
                     ಸ್ಥಿರಾಂಕ ರಚನೆ htp_unary_context * uctx) {
    htp_unary_op_preamble;
    ದಶಮಾಂಶ epsilon = 0.f;
    ಸ್ಮರಣೆ_ನಕಲು(&epsilon, op_params, ಗಾತ್ರ(ದಶಮಾಂಶ));

    ಗಾಗಿ (uint32_t ir = 0; ir < ಅಂಕೆ_ಅಡ್ಡಸಾಲುಗಳು_2; ir++) {
        ಸ್ಥಿರಾಂಕ uint8_t * ಸೀಮಿತ src_local = (ಸ್ಥಿರಾಂಕ uint8_t *)src + (ir * src0_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned);
        uint8_t * ಸೀಮಿತ dst_local       = (uint8_t *)dst + (ir * dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned);

        hvx_fast_norm_f32((ಸ್ಥಿರಾಂಕ uint8_t *) src_local, (uint8_t *) dst_local, ne0, epsilon);
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ sqr_f32(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * ಸೀಮಿತ src,
                    ಶೂನ್ಯ * ಸೀಮಿತ dst,
                    ಸ್ಥಿರಾಂಕ uint32_t ಅಂಕೆ_ಅಡ್ಡಸಾಲುಗಳು_2,
                    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_unary_context * uctx) {
    htp_unary_op_preamble;

    ಗಾಗಿ (uint32_t ir = 0; ir < ಅಂಕೆ_ಅಡ್ಡಸಾಲುಗಳು_2; ir++) {
        ಸ್ಥಿರಾಂಕ uint8_t * ಸೀಮಿತ src_local = (ಸ್ಥಿರಾಂಕ uint8_t *)src + (ir * src0_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned);
        uint8_t * ಸೀಮಿತ dst_local       = (uint8_t *)dst + (ir * dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned);

        hvx_sqr_f32_aa((uint8_t *) dst_local, (ಸ್ಥಿರಾಂಕ uint8_t *) src_local, ne0);
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ sqrt_f32(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * ಸೀಮಿತ src,
                     ಶೂನ್ಯ * ಸೀಮಿತ dst,
                     ಸ್ಥಿರಾಂಕ uint32_t ಅಂಕೆ_ಅಡ್ಡಸಾಲುಗಳು_2,
                     ಸ್ಥಿರಾಂಕ ರಚನೆ htp_unary_context * uctx) {
    htp_unary_op_preamble;

    ಗಾಗಿ (uint32_t ir = 0; ir < ಅಂಕೆ_ಅಡ್ಡಸಾಲುಗಳು_2; ir++) {
        ಸ್ಥಿರಾಂಕ uint8_t * ಸೀಮಿತ src_local = (ಸ್ಥಿರಾಂಕ uint8_t *)src + (ir * src0_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned);
        uint8_t * ಸೀಮಿತ dst_local       = (uint8_t *)dst + (ir * dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned);

        hvx_sqrt_f32_aa((uint8_t *) dst_local, (ಸ್ಥಿರಾಂಕ uint8_t *) src_local, ne0);
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ scale_f16(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * ಸೀಮಿತ src,
                      ಶೂನ್ಯ * ಸೀಮಿತ dst,
                      ಸ್ಥಿರಾಂಕ uint32_t ಅಂಕೆ_ಅಡ್ಡಸಾಲುಗಳು_2,
                      ಸ್ಥಿರಾಂಕ ರಚನೆ htp_unary_context * uctx) {
    htp_unary_op_preamble;
    ದಶಮಾಂಶ scale = 0.f;
    ದಶಮಾಂಶ bias  = 0.f;
    ಸ್ಮರಣೆ_ನಕಲು(&scale, &op_params[0], ಗಾತ್ರ(ದಶಮಾಂಶ));
    ಸ್ಮರಣೆ_ನಕಲು(&bias,  &op_params[1], ಗಾತ್ರ(ದಶಮಾಂಶ));

    ಗಾಗಿ (uint32_t ir = 0; ir < ಅಂಕೆ_ಅಡ್ಡಸಾಲುಗಳು_2; ir++) {
        ಸ್ಥಿರಾಂಕ uint8_t * ಸೀಮಿತ src_local = (ಸ್ಥಿರಾಂಕ uint8_t *)src + (ir * src0_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned);
        uint8_t * ಸೀಮಿತ dst_local       = (uint8_t *)dst + (ir * dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned);

        hvx_scale_ಸ್ಥಾನಾಂತರ_f16_aa((uint8_t *) dst_local, (ಸ್ಥಿರಾಂಕ uint8_t *) src_local, ne0, scale, bias);
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ clamp_f16(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * ಸೀಮಿತ src,
                      ಶೂನ್ಯ * ಸೀಮಿತ dst,
                      ಸ್ಥಿರಾಂಕ uint32_t ಅಂಕೆ_ಅಡ್ಡಸಾಲುಗಳು_2,
                      ಸ್ಥಿರಾಂಕ ರಚನೆ htp_unary_context * uctx) {
    htp_unary_op_preamble;
    ದಶಮಾಂಶ min = 0.f;
    ದಶಮಾಂಶ max = 0.f;
    ಸ್ಮರಣೆ_ನಕಲು(&min, &op_params[0], ಗಾತ್ರ(ದಶಮಾಂಶ));
    ಸ್ಮರಣೆ_ನಕಲು(&max, &op_params[1], ಗಾತ್ರ(ದಶಮಾಂಶ));

    ಗಾಗಿ (uint32_t ir = 0; ir < ಅಂಕೆ_ಅಡ್ಡಸಾಲುಗಳು_2; ir++) {
        ಸ್ಥಿರಾಂಕ uint8_t * ಸೀಮಿತ src_local = (ಸ್ಥಿರಾಂಕ uint8_t *)src + (ir * src0_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned);
        uint8_t * ಸೀಮಿತ dst_local       = (uint8_t *)dst + (ir * dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned);

        hvx_clamp_scalar_f16(dst_local, src_local, (_Float16) min, (_Float16) max, ne0);
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ rms_norm_f16(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * ಸೀಮಿತ src,
                         ಶೂನ್ಯ * ಸೀಮಿತ dst,
                         ಸ್ಥಿರಾಂಕ uint32_t ಅಂಕೆ_ಅಡ್ಡಸಾಲುಗಳು_2,
                         ಸ್ಥಿರಾಂಕ ರಚನೆ htp_unary_context * uctx) {
    htp_unary_op_preamble;
    ದಶಮಾಂಶ epsilon = 0.f;
    ಸ್ಮರಣೆ_ನಕಲು(&epsilon, op_params, ಗಾತ್ರ(ದಶಮಾಂಶ));

    ಗಾಗಿ (uint32_t ir = 0; ir < ಅಂಕೆ_ಅಡ್ಡಸಾಲುಗಳು_2; ir++) {
        ಸ್ಥಿರಾಂಕ uint8_t * ಸೀಮಿತ src_local = (ಸ್ಥಿರಾಂಕ uint8_t *)src + (ir * src0_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned);
        uint8_t * ಸೀಮಿತ dst_local       = (uint8_t *)dst + (ir * dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned);

        hvx_fast_rms_norm_f16((ಸ್ಥಿರಾಂಕ uint8_t *) src_local, (uint8_t *) dst_local, ne0, epsilon);
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ norm_f16(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * ಸೀಮಿತ src,
                     ಶೂನ್ಯ * ಸೀಮಿತ dst,
                     ಸ್ಥಿರಾಂಕ uint32_t ಅಂಕೆ_ಅಡ್ಡಸಾಲುಗಳು_2,
                     ಸ್ಥಿರಾಂಕ ರಚನೆ htp_unary_context * uctx) {
    htp_unary_op_preamble;
    ದಶಮಾಂಶ epsilon = 0.f;
    ಸ್ಮರಣೆ_ನಕಲು(&epsilon, op_params, ಗಾತ್ರ(ದಶಮಾಂಶ));

    ಗಾಗಿ (uint32_t ir = 0; ir < ಅಂಕೆ_ಅಡ್ಡಸಾಲುಗಳು_2; ir++) {
        ಸ್ಥಿರಾಂಕ uint8_t * ಸೀಮಿತ src_local = (ಸ್ಥಿರಾಂಕ uint8_t *)src + (ir * src0_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned);
        uint8_t * ಸೀಮಿತ dst_local       = (uint8_t *)dst + (ir * dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned);

        hvx_fast_norm_f16((ಸ್ಥಿರಾಂಕ uint8_t *) src_local, (uint8_t *) dst_local, ne0, epsilon);
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ sqr_f16(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * ಸೀಮಿತ src,
                    ಶೂನ್ಯ * ಸೀಮಿತ dst,
                    ಸ್ಥಿರಾಂಕ uint32_t ಅಂಕೆ_ಅಡ್ಡಸಾಲುಗಳು_2,
                    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_unary_context * uctx) {
    htp_unary_op_preamble;

    ಗಾಗಿ (uint32_t ir = 0; ir < ಅಂಕೆ_ಅಡ್ಡಸಾಲುಗಳು_2; ir++) {
        ಸ್ಥಿರಾಂಕ uint8_t * ಸೀಮಿತ src_local = (ಸ್ಥಿರಾಂಕ uint8_t *)src + (ir * src0_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned);
        uint8_t * ಸೀಮಿತ dst_local       = (uint8_t *)dst + (ir * dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned);

        hvx_sqr_f16_aa((uint8_t *) dst_local, (ಸ್ಥಿರಾಂಕ uint8_t *) src_local, ne0);
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ sqrt_f16(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * ಸೀಮಿತ src,
                     ಶೂನ್ಯ * ಸೀಮಿತ dst,
                     ಸ್ಥಿರಾಂಕ uint32_t ಅಂಕೆ_ಅಡ್ಡಸಾಲುಗಳು_2,
                     ಸ್ಥಿರಾಂಕ ರಚನೆ htp_unary_context * uctx) {
    htp_unary_op_preamble;

    ಗಾಗಿ (uint32_t ir = 0; ir < ಅಂಕೆ_ಅಡ್ಡಸಾಲುಗಳು_2; ir++) {
        ಸ್ಥಿರಾಂಕ uint8_t * ಸೀಮಿತ src_local = (ಸ್ಥಿರಾಂಕ uint8_t *)src + (ir * src0_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned);
        uint8_t * ಸೀಮಿತ dst_local       = (uint8_t *)dst + (ir * dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned);

        hvx_sqrt_f16_aa((uint8_t *) dst_local, (ಸ್ಥಿರಾಂಕ uint8_t *) src_local, ne0);
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ abs_f16(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * ಸೀಮಿತ src,
                    ಶೂನ್ಯ * ಸೀಮಿತ dst,
                    ಸ್ಥಿರಾಂಕ uint32_t ಅಂಕೆ_ಅಡ್ಡಸಾಲುಗಳು_2,
                    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_unary_context * uctx) {
    htp_unary_op_preamble;

    ಗಾಗಿ (uint32_t ir = 0; ir < ಅಂಕೆ_ಅಡ್ಡಸಾಲುಗಳು_2; ir++) {
        ಸ್ಥಿರಾಂಕ uint8_t * ಸೀಮಿತ src_local = (ಸ್ಥಿರಾಂಕ uint8_t *)src + (ir * src0_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned);
        uint8_t * ಸೀಮಿತ dst_local       = (uint8_t *)dst + (ir * dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned);

        hvx_abs_f16_aa((uint8_t *) dst_local, (ಸ್ಥಿರಾಂಕ uint8_t *) src_local, ne0);
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ log_f16(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * ಸೀಮಿತ src,
                    ಶೂನ್ಯ * ಸೀಮಿತ dst,
                    ಸ್ಥಿರಾಂಕ uint32_t ಅಂಕೆ_ಅಡ್ಡಸಾಲುಗಳು_2,
                    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_unary_context * uctx) {
    htp_unary_op_preamble;

    ಗಾಗಿ (uint32_t ir = 0; ir < ಅಂಕೆ_ಅಡ್ಡಸಾಲುಗಳು_2; ir++) {
        ಸ್ಥಿರಾಂಕ uint8_t * ಸೀಮಿತ src_local = (ಸ್ಥಿರಾಂಕ uint8_t *)src + (ir * src0_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned);
        uint8_t * ಸೀಮಿತ dst_local       = (uint8_t *)dst + (ir * dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned);

        hvx_log_f16_aa((uint8_t *) dst_local, (ಸ್ಥಿರಾಂಕ uint8_t *) src_local, ne0);
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ ಹೆಜ್ಜೆ_f16(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * ಸೀಮಿತ src,
                     ಶೂನ್ಯ * ಸೀಮಿತ dst,
                     ಸ್ಥಿರಾಂಕ uint32_t ಅಂಕೆ_ಅಡ್ಡಸಾಲುಗಳು_2,
                     ಸ್ಥಿರಾಂಕ ರಚನೆ htp_unary_context * uctx) {
    htp_unary_op_preamble;

    ಗಾಗಿ (uint32_t ir = 0; ir < ಅಂಕೆ_ಅಡ್ಡಸಾಲುಗಳು_2; ir++) {
        ಸ್ಥಿರಾಂಕ uint8_t * ಸೀಮಿತ src_local = (ಸ್ಥಿರಾಂಕ uint8_t *)src + (ir * src0_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned);
        uint8_t * ಸೀಮಿತ dst_local       = (uint8_t *)dst + (ir * dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned);

        hvx_ಹೆಜ್ಜೆ_f16_aa((uint8_t *) dst_local, (ಸ್ಥಿರಾಂಕ uint8_t *) src_local, ne0);
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ l2_norm_f16(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * ಸೀಮಿತ src,
                        ಶೂನ್ಯ * ಸೀಮಿತ dst,
                        ಸ್ಥಿರಾಂಕ uint32_t ಅಂಕೆ_ಅಡ್ಡಸಾಲುಗಳು_2,
                        ಸ್ಥಿರಾಂಕ ರಚನೆ htp_unary_context * uctx) {
    htp_unary_op_preamble;
    ದಶಮಾಂಶ epsilon = 0.f;
    ಸ್ಮರಣೆ_ನಕಲು(&epsilon, op_params, ಗಾತ್ರ(ದಶಮಾಂಶ));

    ಗಾಗಿ (uint32_t ir = 0; ir < ಅಂಕೆ_ಅಡ್ಡಸಾಲುಗಳು_2; ir++) {
        ಸ್ಥಿರಾಂಕ uint8_t * ಸೀಮಿತ src_f = (ಸ್ಥಿರಾಂಕ uint8_t *)src + (ir * src0_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned);
        uint8_t * ಸೀಮಿತ dst_f       = (uint8_t *)dst + (ir * dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned);

        hvx_fast_l2_norm_f16((ಸ್ಥಿರಾಂಕ uint8_t *)src_f, (uint8_t *)dst_f, ne0, epsilon);
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ neg_f32(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * ಸೀಮಿತ src,
                    ಶೂನ್ಯ * ಸೀಮಿತ dst,
                    ಸ್ಥಿರಾಂಕ uint32_t ಅಂಕೆ_ಅಡ್ಡಸಾಲುಗಳು_2,
                    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_unary_context * uctx) {
    htp_unary_op_preamble;

    ಗಾಗಿ (uint32_t ir = 0; ir < ಅಂಕೆ_ಅಡ್ಡಸಾಲುಗಳು_2; ir++) {
        ಸ್ಥಿರಾಂಕ uint8_t * ಸೀಮಿತ src_local = (ಸ್ಥಿರಾಂಕ uint8_t *)src + (ir * src0_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned);
        uint8_t * ಸೀಮಿತ dst_local       = (uint8_t *)dst + (ir * dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned);

        hvx_scale_f32_aa(dst_local, src_local, ne0, -1.0f);
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ exp_f32(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * ಸೀಮಿತ src,
                    ಶೂನ್ಯ * ಸೀಮಿತ dst,
                    ಸ್ಥಿರಾಂಕ uint32_t ಅಂಕೆ_ಅಡ್ಡಸಾಲುಗಳು_2,
                    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_unary_context * uctx) {
    htp_unary_op_preamble;

    ಗಾಗಿ (uint32_t ir = 0; ir < ಅಂಕೆ_ಅಡ್ಡಸಾಲುಗಳು_2; ir++) {
        ಸ್ಥಿರಾಂಕ uint8_t * ಸೀಮಿತ src_local = (ಸ್ಥಿರಾಂಕ uint8_t *)src + (ir * src0_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned);
        uint8_t * ಸೀಮಿತ dst_local       = (uint8_t *)dst + (ir * dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned);

        hvx_exp_f32(dst_local, src_local, ne0, false);
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ sigmoid_f32(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * ಸೀಮಿತ src,
                        ಶೂನ್ಯ * ಸೀಮಿತ dst,
                        ಸ್ಥಿರಾಂಕ uint32_t ಅಂಕೆ_ಅಡ್ಡಸಾಲುಗಳು_2,
                        ಸ್ಥಿರಾಂಕ ರಚನೆ htp_unary_context * uctx) {
    htp_unary_op_preamble;

    ಗಾಗಿ (uint32_t ir = 0; ir < ಅಂಕೆ_ಅಡ್ಡಸಾಲುಗಳು_2; ir++) {
        ಸ್ಥಿರಾಂಕ uint8_t * ಸೀಮಿತ src_local = (ಸ್ಥಿರಾಂಕ uint8_t *)src + (ir * src0_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned);
        uint8_t * ಸೀಮಿತ dst_local       = (uint8_t *)dst + (ir * dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned);

        hvx_sigmoid_f32_aa(dst_local, src_local, ne0);
    }
}

// silu(x) = x * sigmoid(x)
ಸ್ಥಿರ ಶೂನ್ಯ silu_f32(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * ಸೀಮಿತ src,
                     ಶೂನ್ಯ * ಸೀಮಿತ dst,
                     ಸ್ಥಿರಾಂಕ uint32_t ಅಂಕೆ_ಅಡ್ಡಸಾಲುಗಳು_2,
                     ಸ್ಥಿರಾಂಕ ರಚನೆ htp_unary_context * uctx) {
    htp_unary_op_preamble;

    ಗಾಗಿ (uint32_t ir = 0; ir < ಅಂಕೆ_ಅಡ್ಡಸಾಲುಗಳು_2; ir++) {
        ಸ್ಥಿರಾಂಕ uint8_t * ಸೀಮಿತ src_local = (ಸ್ಥಿರಾಂಕ uint8_t *)src + (ir * src0_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned);
        uint8_t * ಸೀಮಿತ dst_local       = (uint8_t *)dst + (ir * dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned);

        hvx_sigmoid_f32_aa(dst_local, src_local, ne0);
        hvx_mul_f32_aaa(dst_local, src_local, dst_local, ne0);
    }
}

// GELU uses the tanh approximation.
ಸ್ಥಿರ __attribute__((noinline)) HVX_ಸದಿಶ hvx_vec_gelu_f32(HVX_ಸದಿಶ x) {
    ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ half = hvx_vec_splat_f32(0.5f);
    ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ one  = hvx_vec_splat_f32(1.0f);

    HVX_ಸದಿಶ inner = hvx_vec_mul_f32_f32(x, x);
    inner = hvx_vec_mul_f32_f32(inner, hvx_vec_splat_f32(0.044715f));
    inner = hvx_vec_ಕೂಡಿಸು_f32_f32(inner, one);
    inner = hvx_vec_mul_f32_f32(inner, x);
    inner = hvx_vec_mul_f32_f32(inner, hvx_vec_splat_f32(0.7978845608028654f));

    ಹಿಂತಿರುಗಿ hvx_vec_mul_f32_f32(hvx_vec_mul_f32_f32(half, x),
                               hvx_vec_ಕೂಡಿಸು_f32_f32(one, hvx_vec_tanh_f32(inner)));
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ hvx_gelu_f32_aa(uint8_t * ಸೀಮಿತ dst, ಸ್ಥಿರಾಂಕ uint8_t * ಸೀಮಿತ src, uint32_t n) {
    ಪ್ರತಿಜ್ಞಾ((ಚಿಹ್ನೆರಹಿತ ದೀರ್ಘ) dst % 128 == 0);
    ಪ್ರತಿಜ್ಞಾ((ಚಿಹ್ನೆರಹಿತ ದೀರ್ಘ) src % 128 == 0);

    HVX_ಸದಿಶ * ಸೀಮಿತ vdst = (HVX_ಸದಿಶ *) dst;
    HVX_ಸದಿಶ * ಸೀಮಿತ vsrc = (HVX_ಸದಿಶ *) src;

    ಸ್ಥಿರಾಂಕ uint32_t nvec = n / VLEN_FP32;
    ಸ್ಥಿರಾಂಕ uint32_t nloe = n % VLEN_FP32;

    uint32_t i = 0;
    _Pragma("unroll(4)")
    ಗಾಗಿ (; i < nvec; i++) {
        vdst[i] = hvx_vec_gelu_f32(vsrc[i]);
    }
    ಆದರೆ (nloe) {
        hvx_vec_store_a(&vdst[i], nloe * ಗಾತ್ರ(ದಶಮಾಂಶ), hvx_vec_gelu_f32(vsrc[i]));
    }
}

// GELU_QUICK uses x * sigmoid(1.702 * x).
ಸ್ಥಿರ __attribute__((noinline)) HVX_ಸದಿಶ hvx_vec_gelu_ಶೀಘ್ರ_f32(HVX_ಸದಿಶ x) {
    ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ one     = hvx_vec_splat_f32(1.0f);
    ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ max_exp = hvx_vec_splat_f32(87.0f);
    ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ min_exp = hvx_vec_splat_f32(-87.0f);
    ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ scaled  = hvx_vec_mul_f32_f32(x, hvx_vec_splat_f32(1.702f));
    ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ sigmoid = hvx_vec_fast_sigmoid_f32_guard(scaled, one, max_exp, min_exp);

    ಹಿಂತಿರುಗಿ hvx_vec_mul_f32_f32(x, sigmoid);
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ hvx_gelu_ಶೀಘ್ರ_f32_aa(uint8_t * ಸೀಮಿತ dst, ಸ್ಥಿರಾಂಕ uint8_t * ಸೀಮಿತ src, uint32_t n) {
    ಪ್ರತಿಜ್ಞಾ((ಚಿಹ್ನೆರಹಿತ ದೀರ್ಘ) dst % 128 == 0);
    ಪ್ರತಿಜ್ಞಾ((ಚಿಹ್ನೆರಹಿತ ದೀರ್ಘ) src % 128 == 0);

    HVX_ಸದಿಶ * ಸೀಮಿತ vdst = (HVX_ಸದಿಶ *) dst;
    HVX_ಸದಿಶ * ಸೀಮಿತ vsrc = (HVX_ಸದಿಶ *) src;

    ಸ್ಥಿರಾಂಕ uint32_t nvec = n / VLEN_FP32;
    ಸ್ಥಿರಾಂಕ uint32_t nloe = n % VLEN_FP32;

    uint32_t i = 0;
    _Pragma("unroll(4)")
    ಗಾಗಿ (; i < nvec; i++) {
        vdst[i] = hvx_vec_gelu_ಶೀಘ್ರ_f32(vsrc[i]);
    }
    ಆದರೆ (nloe) {
        hvx_vec_store_a(&vdst[i], nloe * ಗಾತ್ರ(ದಶಮಾಂಶ), hvx_vec_gelu_ಶೀಘ್ರ_f32(vsrc[i]));
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ gelu_f32(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * ಸೀಮಿತ src,
                     ಶೂನ್ಯ * ಸೀಮಿತ dst,
                     ಸ್ಥಿರಾಂಕ uint32_t ಅಂಕೆ_ಅಡ್ಡಸಾಲುಗಳು_2,
                     ಸ್ಥಿರಾಂಕ ರಚನೆ htp_unary_context * uctx) {
    htp_unary_op_preamble;

    ಗಾಗಿ (uint32_t ir = 0; ir < ಅಂಕೆ_ಅಡ್ಡಸಾಲುಗಳು_2; ir++) {
        ಸ್ಥಿರಾಂಕ uint8_t * ಸೀಮಿತ src_local = (ಸ್ಥಿರಾಂಕ uint8_t *)src + (ir * src0_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned);
        uint8_t * ಸೀಮಿತ dst_local       = (uint8_t *)dst + (ir * dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned);

        hvx_gelu_f32_aa(dst_local, src_local, ne0);
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ gelu_ಶೀಘ್ರ_f32(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * ಸೀಮಿತ src,
                           ಶೂನ್ಯ * ಸೀಮಿತ dst,
                           ಸ್ಥಿರಾಂಕ uint32_t ಅಂಕೆ_ಅಡ್ಡಸಾಲುಗಳು_2,
                           ಸ್ಥಿರಾಂಕ ರಚನೆ htp_unary_context * uctx) {
    htp_unary_op_preamble;

    ಗಾಗಿ (uint32_t ir = 0; ir < ಅಂಕೆ_ಅಡ್ಡಸಾಲುಗಳು_2; ir++) {
        ಸ್ಥಿರಾಂಕ uint8_t * ಸೀಮಿತ src_local = (ಸ್ಥಿರಾಂಕ uint8_t *)src + (ir * src0_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned);
        uint8_t * ಸೀಮಿತ dst_local       = (uint8_t *)dst + (ir * dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned);

        hvx_gelu_ಶೀಘ್ರ_f32_aa(dst_local, src_local, ne0);
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ gelu_erf_f32(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * ಸೀಮಿತ src,
                         ಶೂನ್ಯ * ಸೀಮಿತ dst,
                         ಸ್ಥಿರಾಂಕ uint32_t ಅಂಕೆ_ಅಡ್ಡಸಾಲುಗಳು_2,
                         ಸ್ಥಿರಾಂಕ ರಚನೆ htp_unary_context * uctx) {
    htp_unary_op_preamble;

    ಗಾಗಿ (uint32_t ir = 0; ir < ಅಂಕೆ_ಅಡ್ಡಸಾಲುಗಳು_2; ir++) {
        ಸ್ಥಿರಾಂಕ uint8_t * ಸೀಮಿತ src_local = (ಸ್ಥಿರಾಂಕ uint8_t *) src + (ir * src0_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned);
        uint8_t * ಸೀಮಿತ dst_local       = (uint8_t *) dst + (ir * dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned);

        hvx_gelu_erf_f32_aa(dst_local, src_local, ne0);
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ tri_f32(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * ಸೀಮಿತ src,
                    ಶೂನ್ಯ * ಸೀಮಿತ dst,
                    ಸ್ಥಿರಾಂಕ uint32_t ಅಂಕೆ_ಅಡ್ಡಸಾಲುಗಳು_2,
                    ಸ್ಥಿರಾಂಕ uint32_t ir,
                    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_unary_context * uctx) {
    htp_unary_op_preamble;
    ಸ್ಥಿರಾಂಕ int32_t ttype = op_params[0];
    ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ zero = hvx_vec_splat_f32(0.0f);
    ಸ್ಥಿರಾಂಕ uint32_t nvec  = ne0 / VLEN_FP32;
    ಸ್ಥಿರಾಂಕ uint32_t nloe  = ne0 % VLEN_FP32;

    ಸ್ಥಿರಾಂಕ uint32_t ne01 = uctx->octx->src[0]->ne[1];

    ಗಾಗಿ (uint32_t b = 0; b < ಅಂಕೆ_ಅಡ್ಡಸಾಲುಗಳು_2; b++) {
        ಸ್ಥಿರಾಂಕ uint32_t abs_ಅಡ್ಡಸಾಲು = ir + b;
        ಸ್ಥಿರಾಂಕ uint32_t i01     = abs_ಅಡ್ಡಸಾಲು % ne01;

        ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ * ಸೀಮಿತ v_src = (ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ *) ((ಸ್ಥಿರಾಂಕ uint8_t *) src + b * src0_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned);
        HVX_ಸದಿಶ * ಸೀಮಿತ v_dst       = (HVX_ಸದಿಶ *) ((uint8_t *) dst + b * dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned);

        uint32_t boundary;
        ಪೂರ್ಣಾಂಕ      keep_ಎಡ;
        ಆಯ್ಕೆ (ttype) {
            ಸ್ಥಿತಿ 0: boundary = i01;     keep_ಎಡ = 0; ನಿಲ್ಲಿಸು;  // keep col >= row
            ಸ್ಥಿತಿ 1: boundary = i01 + 1; keep_ಎಡ = 0; ನಿಲ್ಲಿಸು;  // keep col > row
            ಸ್ಥಿತಿ 2: boundary = i01 + 1; keep_ಎಡ = 1; ನಿಲ್ಲಿಸು;  // keep col <= row
            ಸ್ಥಿತಿ 3: boundary = i01;     keep_ಎಡ = 1; ನಿಲ್ಲಿಸು;  // keep col < row
            ಪೂರ್ವನಿಯೋಜಿತ: boundary = 0; keep_ಎಡ = 0; ನಿಲ್ಲಿಸು;
        }
        ಆದರೆ (boundary > ne0) boundary = ne0;

        // Full HVX vectors - each starts at a 128-byte aligned offset
        ಗಾಗಿ (uint32_t i = 0; i < nvec; i++) {
            ಸ್ಥಿರಾಂಕ uint32_t vec_ಆರಂಭ = i * VLEN_FP32;
            ಸ್ಥಿರಾಂಕ uint32_t vec_ಕೊನೆ   = vec_ಆರಂಭ + VLEN_FP32;
            ಆದರೆ (keep_ಎಡ) {
                ಆದರೆ (vec_ಕೊನೆ <= boundary) {
                    v_dst[i] = v_src[i];
                } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (vec_ಆರಂಭ >= boundary) {
                    v_dst[i] = zero;
                } ಇಲ್ಲದಿದ್ದರೆ {
                    HVX_ಸದಿಶ_Pred ಮುಖವಾಡ = Q6_Q_vsetq_R((boundary - vec_ಆರಂಭ) * ಗಾತ್ರ(ದಶಮಾಂಶ));
                    v_dst[i]            = Q6_V_vmux_QVV(ಮುಖವಾಡ, v_src[i], zero);
                }
            } ಇಲ್ಲದಿದ್ದರೆ {
                ಆದರೆ (vec_ಕೊನೆ <= boundary) {
                    v_dst[i] = zero;
                } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (vec_ಆರಂಭ >= boundary) {
                    v_dst[i] = v_src[i];
                } ಇಲ್ಲದಿದ್ದರೆ {
                    HVX_ಸದಿಶ_Pred ಮುಖವಾಡ = Q6_Q_vsetq_R((boundary - vec_ಆರಂಭ) * ಗಾತ್ರ(ದಶಮಾಂಶ));
                    v_dst[i]            = Q6_V_vmux_QVV(ಮುಖವಾಡ, zero, v_src[i]);
                }
            }
        }

        // Tail elements (row_elems not a multiple of VLEN_FP32)
        ಆದರೆ (nloe > 0) {
            ಸ್ಥಿರಾಂಕ uint32_t abs_ಆರಂಭ = nvec * VLEN_FP32;
            ಸ್ಥಿರಾಂಕ uint32_t abs_ಕೊನೆ   = abs_ಆರಂಭ + nloe;
            HVX_ಸದಿಶ     ಬಾಲ_val;
            ಆದರೆ (keep_ಎಡ) {
                ಆದರೆ (abs_ಕೊನೆ <= boundary) {
                    ಬಾಲ_val = v_src[nvec];
                } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (abs_ಆರಂಭ >= boundary) {
                    ಬಾಲ_val = zero;
                } ಇಲ್ಲದಿದ್ದರೆ {
                    HVX_ಸದಿಶ_Pred ಮುಖವಾಡ = Q6_Q_vsetq_R((boundary - abs_ಆರಂಭ) * ಗಾತ್ರ(ದಶಮಾಂಶ));
                    ಬಾಲ_val            = Q6_V_vmux_QVV(ಮುಖವಾಡ, v_src[nvec], zero);
                }
            } ಇಲ್ಲದಿದ್ದರೆ {
                ಆದರೆ (abs_ಕೊನೆ <= boundary) {
                    ಬಾಲ_val = zero;
                } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (abs_ಆರಂಭ >= boundary) {
                    ಬಾಲ_val = v_src[nvec];
                } ಇಲ್ಲದಿದ್ದರೆ {
                    HVX_ಸದಿಶ_Pred ಮುಖವಾಡ = Q6_Q_vsetq_R((boundary - abs_ಆರಂಭ) * ಗಾತ್ರ(ದಶಮಾಂಶ));
                    ಬಾಲ_val            = Q6_V_vmux_QVV(ಮುಖವಾಡ, zero, v_src[nvec]);
                }
            }
            hvx_vec_store_a(&v_dst[nvec], nloe * ಗಾತ್ರ(ದಶಮಾಂಶ), ಬಾಲ_val);
        }
    }
}

// softplus(x) = log(1 + exp(x))
// Match CPU reference: return x for x > 20 to avoid overflow.
ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ hvx_softplus_f32_aa(uint8_t * ಸೀಮಿತ dst, ಸ್ಥಿರಾಂಕ uint8_t * ಸೀಮಿತ src, uint32_t n) {
    ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ * ಸೀಮಿತ v_src = (ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ *) src;
    HVX_ಸದಿಶ * ಸೀಮಿತ v_dst       = (HVX_ಸದಿಶ *) dst;

    ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ twenty = hvx_vec_splat_f32(20.0f);
    ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ one    = hvx_vec_splat_f32(1.0f);

    ಸ್ಥಿರಾಂಕ uint32_t nvec = n / VLEN_FP32;
    ಸ್ಥಿರಾಂಕ uint32_t nloe = n % VLEN_FP32;

    #ಸೂಚನೆ unroll(4)
    ಗಾಗಿ (uint32_t i = 0; i < nvec; i++) {
        HVX_ಸದಿಶ x              = v_src[i];
        HVX_ಸದಿಶ_Pred pred_gt20  = Q6_Q_vcmp_gt_VsfVsf(x, twenty);
        HVX_ಸದಿಶ expx           = hvx_vec_exp_f32(x);
        HVX_ಸದಿಶ logv           = hvx_vec_log_f32(hvx_vec_ಕೂಡಿಸು_f32_f32(one, expx));
        v_dst[i]                  = Q6_V_vmux_QVV(pred_gt20, x, logv);
    }

    ಆದರೆ (nloe > 0) {
        HVX_ಸದಿಶ x             = v_src[nvec];
        HVX_ಸದಿಶ_Pred pred_gt20 = Q6_Q_vcmp_gt_VsfVsf(x, twenty);
        HVX_ಸದಿಶ expx          = hvx_vec_exp_f32(x);
        HVX_ಸದಿಶ logv          = hvx_vec_log_f32(hvx_vec_ಕೂಡಿಸು_f32_f32(one, expx));
        HVX_ಸದಿಶ sp            = Q6_V_vmux_QVV(pred_gt20, x, logv);
        hvx_vec_store_a((ಶೂನ್ಯ *) &v_dst[nvec], nloe * ಗಾತ್ರ(ದಶಮಾಂಶ), sp);
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ softplus_f32(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * ಸೀಮಿತ src,
                         ಶೂನ್ಯ * ಸೀಮಿತ dst,
                         ಸ್ಥಿರಾಂಕ uint32_t ಅಂಕೆ_ಅಡ್ಡಸಾಲುಗಳು_2,
                         ಸ್ಥಿರಾಂಕ ರಚನೆ htp_unary_context * uctx) {
    htp_unary_op_preamble;

    ಗಾಗಿ (uint32_t ir = 0; ir < ಅಂಕೆ_ಅಡ್ಡಸಾಲುಗಳು_2; ir++) {
        ಸ್ಥಿರಾಂಕ uint8_t * ಸೀಮಿತ src_local = (ಸ್ಥಿರಾಂಕ uint8_t *)src + (ir * src0_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned);
        uint8_t * ಸೀಮಿತ dst_local       = (uint8_t *)dst + (ir * dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned);

        hvx_softplus_f32_aa(dst_local, src_local, ne0);
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ l2_norm_f32(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * ಸೀಮಿತ src,
                        ಶೂನ್ಯ * ಸೀಮಿತ dst,
                        ಸ್ಥಿರಾಂಕ uint32_t ಅಂಕೆ_ಅಡ್ಡಸಾಲುಗಳು_2,
                        ಸ್ಥಿರಾಂಕ ರಚನೆ htp_unary_context * uctx) {
    htp_unary_op_preamble;
    ದಶಮಾಂಶ epsilon = 0.f;
    ಸ್ಮರಣೆ_ನಕಲು(&epsilon, op_params, ಗಾತ್ರ(ದಶಮಾಂಶ));

    ಗಾಗಿ (uint32_t ir = 0; ir < ಅಂಕೆ_ಅಡ್ಡಸಾಲುಗಳು_2; ir++) {
        ಸ್ಥಿರಾಂಕ uint8_t * ಸೀಮಿತ src_f = (ಸ್ಥಿರಾಂಕ uint8_t *)src + (ir * src0_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned);
        uint8_t * ಸೀಮಿತ dst_f       = (uint8_t *)dst + (ir * dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned);

        hvx_fast_l2_norm_f32((ಸ್ಥಿರಾಂಕ uint8_t *)src_f, (uint8_t *)dst_f, ne0, epsilon);
    }
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ HVX_ಸದಿಶ_Pair hvx_fwht_stage_pair(HVX_ಸದಿಶ v0, HVX_ಸದಿಶ v1, int32_t b) {
    HVX_ಸದಿಶ_Pair p = Q6_W_vshuff_VVR(v1, v0, b);
    HVX_ಸದಿಶ s = Q6_Vqf32_vadd_Vqf32Vqf32(Q6_V_lo_W(p), Q6_V_hi_W(p));
    HVX_ಸದಿಶ d = Q6_Vqf32_vsub_Vqf32Vqf32(Q6_V_lo_W(p), Q6_V_hi_W(p));
    ಹಿಂತಿರುಗಿ Q6_W_vshuff_VVR(d, s, b);
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ HVX_ಸದಿಶ_Pair hvx_fwht32_pair(HVX_ಸದಿಶ v0_sf, HVX_ಸದಿಶ v1_sf, HVX_ಸದಿಶ v_scale) {
    HVX_ಸದಿಶ v0 = Q6_Vqf32_vmpy_VsfVsf(v0_sf, v_scale);
    HVX_ಸದಿಶ v1 = Q6_Vqf32_vmpy_VsfVsf(v1_sf, v_scale);
    HVX_ಸದಿಶ_Pair p;
    p = hvx_fwht_stage_pair(v0, v1, 4);
    p = hvx_fwht_stage_pair(Q6_V_lo_W(p), Q6_V_hi_W(p), 8);
    p = hvx_fwht_stage_pair(Q6_V_lo_W(p), Q6_V_hi_W(p), 16);
    p = hvx_fwht_stage_pair(Q6_V_lo_W(p), Q6_V_hi_W(p), 32);
    p = hvx_fwht_stage_pair(Q6_V_lo_W(p), Q6_V_hi_W(p), 64);
    ಹಿಂತಿರುಗಿ p;
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ hvx_fwht_store32(HVX_ಸದಿಶ * ಸೀಮಿತ v_dst, HVX_ಸದಿಶ_Pair p) {
    v_dst[0] = Q6_Vsf_equals_Vqf32(Q6_V_lo_W(p));
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ hvx_fwht_store64(HVX_ಸದಿಶ * ಸೀಮಿತ v_dst, HVX_ಸದಿಶ_Pair p) {
    HVX_ಸದಿಶ u = Q6_V_lo_W(p);
    HVX_ಸದಿಶ v = Q6_V_hi_W(p);
    v_dst[0] = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vqf32(u, v));
    v_dst[1] = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vsub_Vqf32Vqf32(u, v));
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ hvx_fwht_ಬಾಲ_f32(HVX_ಸದಿಶ * ಸೀಮಿತ v_dst, uint32_t nvec) {
    uint32_t len_vec = 2;
    ಗಾಗಿ (; len_vec * 2 < nvec; len_vec <<= 2) {
        ಸ್ಥಿರಾಂಕ uint32_t ಹೆಜ್ಜೆ = len_vec * 4;
        ಗಾಗಿ (uint32_t i = 0; i < nvec; i += ಹೆಜ್ಜೆ) {
            ಗಾಗಿ (uint32_t j = 0; j < len_vec; j++) {
                HVX_ಸದಿಶ a0 = v_dst[i + j];
                HVX_ಸದಿಶ a1 = v_dst[i + j + len_vec];
                HVX_ಸದಿಶ a2 = v_dst[i + j + 2 * len_vec];
                HVX_ಸದಿಶ a3 = v_dst[i + j + 3 * len_vec];

                HVX_ಸದಿಶ t0 = Q6_Vqf32_vadd_VsfVsf(a0, a1);
                HVX_ಸದಿಶ t1 = Q6_Vqf32_vsub_VsfVsf(a0, a1);
                HVX_ಸದಿಶ t2 = Q6_Vqf32_vadd_VsfVsf(a2, a3);
                HVX_ಸದಿಶ t3 = Q6_Vqf32_vsub_VsfVsf(a2, a3);

                v_dst[i + j]               = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vqf32(t0, t2));
                v_dst[i + j + len_vec]     = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vqf32(t1, t3));
                v_dst[i + j + 2 * len_vec] = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vsub_Vqf32Vqf32(t0, t2));
                v_dst[i + j + 3 * len_vec] = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vsub_Vqf32Vqf32(t1, t3));
            }
        }
    }

    ಆದರೆ (len_vec < nvec) {
        ಗಾಗಿ (uint32_t i = 0; i < nvec; i += 2 * len_vec) {
            ಗಾಗಿ (uint32_t j = 0; j < len_vec; j++) {
                HVX_ಸದಿಶ u = v_dst[i + j];
                HVX_ಸದಿಶ v = v_dst[i + len_vec + j];
                v_dst[i + j]           = hvx_vec_ಕೂಡಿಸು_f32_f32(u, v);
                v_dst[i + len_vec + j] = hvx_vec_sub_f32_f32(u, v);
            }
        }
    }
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ hvx_fwht_ಅಡ್ಡಸಾಲು_f32(ಸ್ಥಿರಾಂಕ uint8_t * ಸೀಮಿತ src, uint8_t * ಸೀಮಿತ dst, uint32_t n, HVX_ಸದಿಶ v_scale) {
    ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ * ಸೀಮಿತ v_src = (ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ *) src;
    HVX_ಸದಿಶ * ಸೀಮಿತ v_dst       = (HVX_ಸದಿಶ *) dst;
    ಸ್ಥಿರಾಂಕ uint32_t nvec               = n / VLEN_FP32;

    ಆದರೆ (nvec == 1) {
        HVX_ಸದಿಶ_Pair p = hvx_fwht32_pair(v_src[0], v_src[0], v_scale);
        hvx_fwht_store32(v_dst, p);
        ಹಿಂತಿರುಗಿ;
    }

    ಗಾಗಿ (uint32_t i = 0; i < nvec; i += 2) {
        HVX_ಸದಿಶ_Pair p = hvx_fwht32_pair(v_src[i], v_src[i + 1], v_scale);
        hvx_fwht_store64(&v_dst[i], p);
    }

    hvx_fwht_ಬಾಲ_f32(v_dst, nvec);
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ hvx_fwht_ಅಡ್ಡಸಾಲು_f16(ಸ್ಥಿರಾಂಕ uint8_t * ಸೀಮಿತ src, uint8_t * ಸೀಮಿತ dst, uint32_t n, HVX_ಸದಿಶ v_scale) {
    ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ * ಸೀಮಿತ v_src = (ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ *) src;
    HVX_ಸದಿಶ * ಸೀಮಿತ v_dst       = (HVX_ಸದಿಶ *) dst;
    ಸ್ಥಿರಾಂಕ uint32_t nvec               = n / VLEN_FP32;

    ಆದರೆ (nvec == 1) {
        HVX_ಸದಿಶ v0 = Q6_V_lo_W(hvx_vec_f16_ಗಮ್ಯ_f32(v_src[0]));
        HVX_ಸದಿಶ_Pair p = hvx_fwht32_pair(v0, v0, v_scale);
        hvx_fwht_store32(v_dst, p);
        ಹಿಂತಿರುಗಿ;
    }

    ಸ್ಥಿರಾಂಕ uint32_t nvec_f16 = n / VLEN_FP16;
    ಗಾಗಿ (uint32_t i = 0; i < nvec_f16; i++) {
        HVX_ಸದಿಶ_Pair f32p = hvx_vec_f16_ಗಮ್ಯ_f32(v_src[i]);
        HVX_ಸದಿಶ_Pair p    = hvx_fwht32_pair(Q6_V_lo_W(f32p), Q6_V_hi_W(f32p), v_scale);
        hvx_fwht_store64(&v_dst[2 * i], p);
    }

    hvx_fwht_ಬಾಲ_f32(v_dst, nvec);
}

ಸ್ಥಿರ ಶೂನ್ಯ hadamard_f32(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * ಸೀಮಿತ src,
                         ಶೂನ್ಯ * ಸೀಮಿತ dst,
                         ಸ್ಥಿರಾಂಕ uint32_t ಅಂಕೆ_ಅಡ್ಡಸಾಲುಗಳು_2,
                         ಸ್ಥಿರಾಂಕ ರಚನೆ htp_unary_context * uctx) {
    htp_unary_op_preamble;
    ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ v_scale = hvx_vec_splat_f32(1.0f / sqrtf((ದಶಮಾಂಶ) ne0));

    ಗಾಗಿ (uint32_t ir = 0; ir < ಅಂಕೆ_ಅಡ್ಡಸಾಲುಗಳು_2; ir++) {
        ಸ್ಥಿರಾಂಕ uint8_t * ಸೀಮಿತ src_ಅಡ್ಡಸಾಲು = (ಸ್ಥಿರಾಂಕ uint8_t *) src + (ir * src0_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned);
        uint8_t * ಸೀಮಿತ dst_ಅಡ್ಡಸಾಲು       = (uint8_t *) dst + (ir * dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned);

        hvx_fwht_ಅಡ್ಡಸಾಲು_f32(src_ಅಡ್ಡಸಾಲು, dst_ಅಡ್ಡಸಾಲು, ne0, v_scale);
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ hadamard_f16(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * ಸೀಮಿತ src,
                         ಶೂನ್ಯ * ಸೀಮಿತ dst,
                         ಸ್ಥಿರಾಂಕ uint32_t ಅಂಕೆ_ಅಡ್ಡಸಾಲುಗಳು_2,
                         ಸ್ಥಿರಾಂಕ ರಚನೆ htp_unary_context * uctx) {
    htp_unary_op_preamble;
    ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ v_scale = hvx_vec_splat_f32(1.0f / sqrtf((ದಶಮಾಂಶ) ne0));

    ಗಾಗಿ (uint32_t ir = 0; ir < ಅಂಕೆ_ಅಡ್ಡಸಾಲುಗಳು_2; ir++) {
        ಸ್ಥಿರಾಂಕ uint8_t * ಸೀಮಿತ src_ಅಡ್ಡಸಾಲು = (ಸ್ಥಿರಾಂಕ uint8_t *) src + (ir * src0_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned);
        uint8_t * ಸೀಮಿತ dst_ಅಡ್ಡಸಾಲು       = (uint8_t *) dst + (ir * dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned);

        hvx_fwht_ಅಡ್ಡಸಾಲು_f16(src_ಅಡ್ಡಸಾಲು, dst_ಅಡ್ಡಸಾಲು, ne0, v_scale);
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ tanh_f32(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * ಸೀಮಿತ src,
                     ಶೂನ್ಯ * ಸೀಮಿತ dst,
                     ಸ್ಥಿರಾಂಕ uint32_t ಅಂಕೆ_ಅಡ್ಡಸಾಲುಗಳು_2,
                     ಸ್ಥಿರಾಂಕ ರಚನೆ htp_unary_context * uctx) {
    htp_unary_op_preamble;

    ಗಾಗಿ (uint32_t ir = 0; ir < ಅಂಕೆ_ಅಡ್ಡಸಾಲುಗಳು_2; ir++) {
        ಸ್ಥಿರಾಂಕ uint8_t * ಸೀಮಿತ src_local = (ಸ್ಥಿರಾಂಕ uint8_t *)src + (ir * src0_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned);
        uint8_t * ಸೀಮಿತ dst_local       = (uint8_t *)dst + (ir * dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned);

        hvx_tanh_f32_aa(dst_local, src_local, ne0);
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ abs_f32(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * ಸೀಮಿತ src,
                    ಶೂನ್ಯ * ಸೀಮಿತ dst,
                    ಸ್ಥಿರಾಂಕ uint32_t ಅಂಕೆ_ಅಡ್ಡಸಾಲುಗಳು_2,
                    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_unary_context * uctx) {
    htp_unary_op_preamble;

    ಗಾಗಿ (uint32_t ir = 0; ir < ಅಂಕೆ_ಅಡ್ಡಸಾಲುಗಳು_2; ir++) {
        ಸ್ಥಿರಾಂಕ uint8_t * ಸೀಮಿತ src_local = (ಸ್ಥಿರಾಂಕ uint8_t *)src + (ir * src0_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned);
        uint8_t * ಸೀಮಿತ dst_local       = (uint8_t *)dst + (ir * dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned);

        hvx_abs_f32_aa(dst_local, src_local, ne0);
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ relu_f32(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * ಸೀಮಿತ src,
                     ಶೂನ್ಯ * ಸೀಮಿತ dst,
                     ಸ್ಥಿರಾಂಕ uint32_t ಅಂಕೆ_ಅಡ್ಡಸಾಲುಗಳು_2,
                     ಸ್ಥಿರಾಂಕ ರಚನೆ htp_unary_context * uctx) {
    htp_unary_op_preamble;

    ಗಾಗಿ (uint32_t ir = 0; ir < ಅಂಕೆ_ಅಡ್ಡಸಾಲುಗಳು_2; ir++) {
        ಸ್ಥಿರಾಂಕ uint8_t * ಸೀಮಿತ src_local = (ಸ್ಥಿರಾಂಕ uint8_t *)src + (ir * src0_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned);
        uint8_t * ಸೀಮಿತ dst_local       = (uint8_t *)dst + (ir * dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned);

        hvx_max_scalar_f32(dst_local, src_local, 0.0f, ne0);
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ ಹೆಜ್ಜೆ_f32(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * ಸೀಮಿತ src,
                     ಶೂನ್ಯ * ಸೀಮಿತ dst,
                     ಸ್ಥಿರಾಂಕ uint32_t ಅಂಕೆ_ಅಡ್ಡಸಾಲುಗಳು_2,
                     ಸ್ಥಿರಾಂಕ ರಚನೆ htp_unary_context * uctx) {
    htp_unary_op_preamble;

    ಗಾಗಿ (uint32_t ir = 0; ir < ಅಂಕೆ_ಅಡ್ಡಸಾಲುಗಳು_2; ir++) {
        ಸ್ಥಿರಾಂಕ uint8_t * ಸೀಮಿತ src_local = (ಸ್ಥಿರಾಂಕ uint8_t *)src + (ir * src0_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned);
        uint8_t * ಸೀಮಿತ dst_local       = (uint8_t *)dst + (ir * dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned);

        hvx_ಹೆಜ್ಜೆ_f32_aa(dst_local, src_local, ne0);
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ log_f32(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * ಸೀಮಿತ src,
                    ಶೂನ್ಯ * ಸೀಮಿತ dst,
                    ಸ್ಥಿರಾಂಕ uint32_t ಅಂಕೆ_ಅಡ್ಡಸಾಲುಗಳು_2,
                    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_unary_context * uctx) {
    htp_unary_op_preamble;

    ಗಾಗಿ (uint32_t ir = 0; ir < ಅಂಕೆ_ಅಡ್ಡಸಾಲುಗಳು_2; ir++) {
        ಸ್ಥಿರಾಂಕ uint8_t * ಸೀಮಿತ src_local = (ಸ್ಥಿರಾಂಕ uint8_t *)src + (ir * src0_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned);
        uint8_t * ಸೀಮಿತ dst_local       = (uint8_t *)dst + (ir * dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned);

        hvx_log_f32_aa(dst_local, src_local, ne0);
    }
}

#// Pointwise unary ops on one column tile in VTCM.
ಸ್ಥಿರ ಶೂನ್ಯ tile_scale_f32(ಶೂನ್ಯ * ಸೀಮಿತ dst, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * ಸೀಮಿತ src, uint32_t tw, ಸ್ಥಿರಾಂಕ ರಚನೆ htp_unary_context * uctx) {
    ದಶಮಾಂಶ scale = 0.f;
    ದಶಮಾಂಶ bias  = 0.f;
    ಸ್ಮರಣೆ_ನಕಲು(&scale, &uctx->octx->op_params[0], ಗಾತ್ರ(ದಶಮಾಂಶ));
    ಸ್ಮರಣೆ_ನಕಲು(&bias,  &uctx->octx->op_params[1], ಗಾತ್ರ(ದಶಮಾಂಶ));
    hvx_scale_ಸ್ಥಾನಾಂತರ_f32_aa((uint8_t *) dst, (ಸ್ಥಿರಾಂಕ uint8_t *) src, tw, scale, bias);
}

ಸ್ಥಿರ ಶೂನ್ಯ tile_clamp_f32(ಶೂನ್ಯ * ಸೀಮಿತ dst, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * ಸೀಮಿತ src, uint32_t tw, ಸ್ಥಿರಾಂಕ ರಚನೆ htp_unary_context * uctx) {
    ದಶಮಾಂಶ min = 0.f;
    ದಶಮಾಂಶ max = 0.f;
    ಸ್ಮರಣೆ_ನಕಲು(&min, &uctx->octx->op_params[0], ಗಾತ್ರ(ದಶಮಾಂಶ));
    ಸ್ಮರಣೆ_ನಕಲು(&max, &uctx->octx->op_params[1], ಗಾತ್ರ(ದಶಮಾಂಶ));
    hvx_clamp_scalar_f32((uint8_t *) dst, (ಸ್ಥಿರಾಂಕ uint8_t *) src, min, max, tw);
}

ಸ್ಥಿರ ಶೂನ್ಯ tile_leaky_relu_f32(ಶೂನ್ಯ * ಸೀಮಿತ dst, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * ಸೀಮಿತ src, uint32_t tw, ಸ್ಥಿರಾಂಕ ರಚನೆ htp_unary_context * uctx) {
    ದಶಮಾಂಶ negative_ಇಳಿಜಾರು = 0.f;
    ಸ್ಮರಣೆ_ನಕಲು(&negative_ಇಳಿಜಾರು, &uctx->octx->op_params[0], ಗಾತ್ರ(ದಶಮಾಂಶ));
    hvx_leaky_relu_scalar_f32((uint8_t *) dst, (ಸ್ಥಿರಾಂಕ uint8_t *) src, negative_ಇಳಿಜಾರು, tw);
}

ಸ್ಥಿರ ಶೂನ್ಯ tile_sqr_f32(ಶೂನ್ಯ * ಸೀಮಿತ dst, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * ಸೀಮಿತ src, uint32_t tw, ಸ್ಥಿರಾಂಕ ರಚನೆ htp_unary_context * uctx) {
    (ಶೂನ್ಯ) uctx;
    hvx_sqr_f32_aa((uint8_t *) dst, (ಸ್ಥಿರಾಂಕ uint8_t *) src, tw);
}

ಸ್ಥಿರ ಶೂನ್ಯ tile_sqrt_f32(ಶೂನ್ಯ * ಸೀಮಿತ dst, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * ಸೀಮಿತ src, uint32_t tw, ಸ್ಥಿರಾಂಕ ರಚನೆ htp_unary_context * uctx) {
    (ಶೂನ್ಯ) uctx;
    hvx_sqrt_f32_aa((uint8_t *) dst, (ಸ್ಥಿರಾಂಕ uint8_t *) src, tw);
}

ಸ್ಥಿರ ಶೂನ್ಯ tile_neg_f32(ಶೂನ್ಯ * ಸೀಮಿತ dst, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * ಸೀಮಿತ src, uint32_t tw, ಸ್ಥಿರಾಂಕ ರಚನೆ htp_unary_context * uctx) {
    (ಶೂನ್ಯ) uctx;
    hvx_scale_f32_aa((uint8_t *) dst, (ಸ್ಥಿರಾಂಕ uint8_t *) src, tw, -1.0f);
}

ಸ್ಥಿರ ಶೂನ್ಯ tile_exp_f32(ಶೂನ್ಯ * ಸೀಮಿತ dst, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * ಸೀಮಿತ src, uint32_t tw, ಸ್ಥಿರಾಂಕ ರಚನೆ htp_unary_context * uctx) {
    (ಶೂನ್ಯ) uctx;
    hvx_exp_f32((uint8_t *) dst, (ಸ್ಥಿರಾಂಕ uint8_t *) src, tw, false);
}

ಸ್ಥಿರ ಶೂನ್ಯ tile_sigmoid_f32(ಶೂನ್ಯ * ಸೀಮಿತ dst, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * ಸೀಮಿತ src, uint32_t tw, ಸ್ಥಿರಾಂಕ ರಚನೆ htp_unary_context * uctx) {
    (ಶೂನ್ಯ) uctx;
    hvx_sigmoid_f32_aa((uint8_t *) dst, (ಸ್ಥಿರಾಂಕ uint8_t *) src, tw);
}

ಸ್ಥಿರ ಶೂನ್ಯ tile_silu_f32(ಶೂನ್ಯ * ಸೀಮಿತ dst, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * ಸೀಮಿತ src, uint32_t tw, ಸ್ಥಿರಾಂಕ ರಚನೆ htp_unary_context * uctx) {
    (ಶೂನ್ಯ) uctx;
    hvx_sigmoid_f32_aa((uint8_t *) dst, (ಸ್ಥಿರಾಂಕ uint8_t *) src, tw);
    hvx_mul_f32_aaa((uint8_t *) dst, (ಸ್ಥಿರಾಂಕ uint8_t *) src, (uint8_t *) dst, tw);
}

ಸ್ಥಿರ ಶೂನ್ಯ tile_gelu_f32(ಶೂನ್ಯ * ಸೀಮಿತ dst, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * ಸೀಮಿತ src, uint32_t tw, ಸ್ಥಿರಾಂಕ ರಚನೆ htp_unary_context * uctx) {
    (ಶೂನ್ಯ) uctx;
    hvx_gelu_f32_aa((uint8_t *) dst, (ಸ್ಥಿರಾಂಕ uint8_t *) src, tw);
}

ಸ್ಥಿರ ಶೂನ್ಯ tile_gelu_ಶೀಘ್ರ_f32(ಶೂನ್ಯ * ಸೀಮಿತ dst, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * ಸೀಮಿತ src, uint32_t tw, ಸ್ಥಿರಾಂಕ ರಚನೆ htp_unary_context * uctx) {
    (ಶೂನ್ಯ) uctx;
    hvx_gelu_ಶೀಘ್ರ_f32_aa((uint8_t *) dst, (ಸ್ಥಿರಾಂಕ uint8_t *) src, tw);
}

ಸ್ಥಿರ ಶೂನ್ಯ tile_gelu_erf_f32(ಶೂನ್ಯ * ಸೀಮಿತ dst, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * ಸೀಮಿತ src, uint32_t tw, ಸ್ಥಿರಾಂಕ ರಚನೆ htp_unary_context * uctx) {
    (ಶೂನ್ಯ) uctx;
    hvx_gelu_erf_f32_aa((uint8_t *) dst, (ಸ್ಥಿರಾಂಕ uint8_t *) src, tw);
}

ಸ್ಥಿರ ಶೂನ್ಯ tile_softplus_f32(ಶೂನ್ಯ * ಸೀಮಿತ dst, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * ಸೀಮಿತ src, uint32_t tw, ಸ್ಥಿರಾಂಕ ರಚನೆ htp_unary_context * uctx) {
    (ಶೂನ್ಯ) uctx;
    hvx_softplus_f32_aa((uint8_t *) dst, (ಸ್ಥಿರಾಂಕ uint8_t *) src, tw);
}

ಸ್ಥಿರ ಶೂನ್ಯ tile_tanh_f32(ಶೂನ್ಯ * ಸೀಮಿತ dst, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * ಸೀಮಿತ src, uint32_t tw, ಸ್ಥಿರಾಂಕ ರಚನೆ htp_unary_context * uctx) {
    (ಶೂನ್ಯ) uctx;
    hvx_tanh_f32_aa((uint8_t *) dst, (ಸ್ಥಿರಾಂಕ uint8_t *) src, tw);
}

ಸ್ಥಿರ ಶೂನ್ಯ tile_abs_f32(ಶೂನ್ಯ * ಸೀಮಿತ dst, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * ಸೀಮಿತ src, uint32_t tw, ಸ್ಥಿರಾಂಕ ರಚನೆ htp_unary_context * uctx) {
    (ಶೂನ್ಯ) uctx;
    hvx_abs_f32_aa((uint8_t *) dst, (ಸ್ಥಿರಾಂಕ uint8_t *) src, tw);
}

ಸ್ಥಿರ ಶೂನ್ಯ tile_log_f32(ಶೂನ್ಯ * ಸೀಮಿತ dst, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * ಸೀಮಿತ src, uint32_t tw, ಸ್ಥಿರಾಂಕ ರಚನೆ htp_unary_context * uctx) {
    (ಶೂನ್ಯ) uctx;
    hvx_log_f32_aa((uint8_t *) dst, (ಸ್ಥಿರಾಂಕ uint8_t *) src, tw);
}

ಸ್ಥಿರ ಶೂನ್ಯ tile_relu_f32(ಶೂನ್ಯ * ಸೀಮಿತ dst, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * ಸೀಮಿತ src, uint32_t tw, ಸ್ಥಿರಾಂಕ ರಚನೆ htp_unary_context * uctx) {
    (ಶೂನ್ಯ) uctx;
    hvx_max_scalar_f32((uint8_t *) dst, (ಸ್ಥಿರಾಂಕ uint8_t *) src, 0.0f, tw);
}

ಸ್ಥಿರ ಶೂನ್ಯ tile_ಹೆಜ್ಜೆ_f32(ಶೂನ್ಯ * ಸೀಮಿತ dst, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * ಸೀಮಿತ src, uint32_t tw, ಸ್ಥಿರಾಂಕ ರಚನೆ htp_unary_context * uctx) {
    (ಶೂನ್ಯ) uctx;
    hvx_ಹೆಜ್ಜೆ_f32_aa((uint8_t *) dst, (ಸ್ಥಿರಾಂಕ uint8_t *) src, tw);
}

ಸ್ಥಿರ ಶೂನ್ಯ tri_apply_tile_f32(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * ಸೀಮಿತ src, ಶೂನ್ಯ * ಸೀಮಿತ dst,
                               uint32_t tile_elems, uint32_t ಲಂಬಸಾಲು_ಆರಂಭ, uint32_t i01,
                               uint32_t ne0, int32_t ttype) {
    ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ * ಸೀಮಿತ v_src = (ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ *) src;
    HVX_ಸದಿಶ * ಸೀಮಿತ v_dst       = (HVX_ಸದಿಶ *) dst;
    ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ zero = hvx_vec_splat_f32(0.0f);

    uint32_t boundary;
    ಪೂರ್ಣಾಂಕ      keep_ಎಡ;
    ಆಯ್ಕೆ (ttype) {
        ಸ್ಥಿತಿ 0: boundary = i01;     keep_ಎಡ = 0; ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ 1: boundary = i01 + 1; keep_ಎಡ = 0; ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ 2: boundary = i01 + 1; keep_ಎಡ = 1; ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ 3: boundary = i01;     keep_ಎಡ = 1; ನಿಲ್ಲಿಸು;
        ಪೂರ್ವನಿಯೋಜಿತ: boundary = 0; keep_ಎಡ = 0; ನಿಲ್ಲಿಸು;
    }
    ಆದರೆ (boundary > ne0) boundary = ne0;

    ಸ್ಥಿರಾಂಕ uint32_t nvec = tile_elems / VLEN_FP32;
    ಸ್ಥಿರಾಂಕ uint32_t nloe = tile_elems % VLEN_FP32;

    ಗಾಗಿ (uint32_t i = 0; i < nvec; i++) {
        ಸ್ಥಿರಾಂಕ uint32_t abs_ಆರಂಭ = ಲಂಬಸಾಲು_ಆರಂಭ + i * VLEN_FP32;
        ಸ್ಥಿರಾಂಕ uint32_t abs_ಕೊನೆ   = abs_ಆರಂಭ + VLEN_FP32;
        ಆದರೆ (keep_ಎಡ) {
            ಆದರೆ (abs_ಕೊನೆ <= boundary) {
                v_dst[i] = v_src[i];
            } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (abs_ಆರಂಭ >= boundary) {
                v_dst[i] = zero;
            } ಇಲ್ಲದಿದ್ದರೆ {
                HVX_ಸದಿಶ_Pred ಮುಖವಾಡ = Q6_Q_vsetq_R((boundary - abs_ಆರಂಭ) * ಗಾತ್ರ(ದಶಮಾಂಶ));
                v_dst[i]            = Q6_V_vmux_QVV(ಮುಖವಾಡ, v_src[i], zero);
            }
        } ಇಲ್ಲದಿದ್ದರೆ {
            ಆದರೆ (abs_ಕೊನೆ <= boundary) {
                v_dst[i] = zero;
            } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (abs_ಆರಂಭ >= boundary) {
                v_dst[i] = v_src[i];
            } ಇಲ್ಲದಿದ್ದರೆ {
                HVX_ಸದಿಶ_Pred ಮುಖವಾಡ = Q6_Q_vsetq_R((boundary - abs_ಆರಂಭ) * ಗಾತ್ರ(ದಶಮಾಂಶ));
                v_dst[i]            = Q6_V_vmux_QVV(ಮುಖವಾಡ, zero, v_src[i]);
            }
        }
    }

    ಆದರೆ (nloe > 0) {
        ಸ್ಥಿರಾಂಕ uint32_t abs_ಆರಂಭ = ಲಂಬಸಾಲು_ಆರಂಭ + nvec * VLEN_FP32;
        ಸ್ಥಿರಾಂಕ uint32_t abs_ಕೊನೆ   = abs_ಆರಂಭ + nloe;
        HVX_ಸದಿಶ     ಬಾಲ_val;
        ಆದರೆ (keep_ಎಡ) {
            ಆದರೆ (abs_ಕೊನೆ <= boundary) {
                ಬಾಲ_val = v_src[nvec];
            } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (abs_ಆರಂಭ >= boundary) {
                ಬಾಲ_val = zero;
            } ಇಲ್ಲದಿದ್ದರೆ {
                HVX_ಸದಿಶ_Pred ಮುಖವಾಡ = Q6_Q_vsetq_R((boundary - abs_ಆರಂಭ) * ಗಾತ್ರ(ದಶಮಾಂಶ));
                ಬಾಲ_val            = Q6_V_vmux_QVV(ಮುಖವಾಡ, v_src[nvec], zero);
            }
        } ಇಲ್ಲದಿದ್ದರೆ {
            ಆದರೆ (abs_ಕೊನೆ <= boundary) {
                ಬಾಲ_val = zero;
            } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (abs_ಆರಂಭ >= boundary) {
                ಬಾಲ_val = v_src[nvec];
            } ಇಲ್ಲದಿದ್ದರೆ {
                HVX_ಸದಿಶ_Pred ಮುಖವಾಡ = Q6_Q_vsetq_R((boundary - abs_ಆರಂಭ) * ಗಾತ್ರ(ದಶಮಾಂಶ));
                ಬಾಲ_val            = Q6_V_vmux_QVV(ಮುಖವಾಡ, zero, v_src[nvec]);
            }
        }
        hvx_vec_store_a(&v_dst[nvec], nloe * ಗಾತ್ರ(ದಶಮಾಂಶ), ಬಾಲ_val);
    }
}

// 1. Standard row-block unary task (F32 and F16).
ಸ್ಥಿರ ಶೂನ್ಯ unary_thread_ಅಡ್ಡಸಾಲು_ಖಂಡ(ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ nth, ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ ith, ಶೂನ್ಯ * ದತ್ತಾಂಶ) {
    (ಶೂನ್ಯ) nth;
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_unary_context * uctx = (ಸ್ಥಿರಾಂಕ ರಚನೆ htp_unary_context *) ದತ್ತಾಂಶ;
    ರಚನೆ htp_ops_context * octx = uctx->octx;
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * src = octx->src[0];
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * dst = octx->dst;
    ರಚನೆ htp_thread_trace * tr = &octx->ctx->trace[ith];

    htp_unary_preamble;

    ಸ್ಥಿರಾಂಕ uint32_t src0_nrows_ಪ್ರತಿ_thread = uctx->src0_nrows_ಪ್ರತಿ_thread;
    ಸ್ಥಿರಾಂಕ size_t src0_ದತ್ತಾಂಶ_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ = uctx->src0_ದತ್ತಾಂಶ_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ;
    ಸ್ಥಿರಾಂಕ size_t dst_ದತ್ತಾಂಶ_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ  = uctx->dst_ದತ್ತಾಂಶ_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ;
    ಸ್ಥಿರಾಂಕ size_t src0_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned = uctx->src0_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned;
    ಸ್ಥಿರಾಂಕ size_t dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned  = uctx->dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned;

    ಸ್ಥಿರಾಂಕ uint32_t src0_nrows = uctx->src0_nrows;
    ಸ್ಥಿರಾಂಕ uint32_t src0_ಆರಂಭ_ಅಡ್ಡಸಾಲು = uctx->ಅಡ್ಡಸಾಲು_ಆರಂಭ + src0_nrows_ಪ್ರತಿ_thread * ith;
    ಸ್ಥಿರಾಂಕ uint32_t src0_ಕೊನೆ_ಅಡ್ಡಸಾಲು   = MIN(src0_ಆರಂಭ_ಅಡ್ಡಸಾಲು + src0_nrows_ಪ್ರತಿ_thread, uctx->ಅಡ್ಡಸಾಲು_ಆರಂಭ + src0_nrows);

    ಆದರೆ (src0_ಆರಂಭ_ಅಡ್ಡಸಾಲು >= src0_ಕೊನೆ_ಅಡ್ಡಸಾಲು) {
        ಹಿಂತಿರುಗಿ;
    }

    ಸ್ಥಿರಾಂಕ dma_addr_t ದತ್ತಾಂಶ_src = uctx->ದತ್ತಾಂಶ_src0;
    ಸ್ಥಿರಾಂಕ dma_addr_t ದತ್ತಾಂಶ_dst = uctx->ದತ್ತಾಂಶ_dst;

    uint8_t * src0_vtcm_ದತ್ತಾಂಶ = uctx->vtcm_src0 + (ith * uctx->vtcm_src0_ಗಾತ್ರ_ಪ್ರತಿ_thread);
    uint8_t * dst_vtcm_ದತ್ತಾಂಶ  = uctx->vtcm_dst + (ith * uctx->vtcm_dst_ಗಾತ್ರ_ಪ್ರತಿ_thread);

    ಸ್ಥಿರಾಂಕ size_t src0_vtcm_half_ಗಾತ್ರ = uctx->src0_vtcm_half_ಗಾತ್ರ;
    ಸ್ಥಿರಾಂಕ size_t dst_vtcm_half_ಗಾತ್ರ  = uctx->dst_vtcm_half_ಗಾತ್ರ;

    ಸ್ಥಿರಾಂಕ bool src0_contig = (nb02 == (size_t)ne01 * nb01) &&
                             (nb03 == (size_t)ne02 * nb02);
    ಸ್ಥಿರಾಂಕ bool dst_contig  = (nb2  == (size_t)ne1  * nb1)  &&
                             (nb3  == (size_t)ne2  * nb2);

    ಸ್ಥಿರಾಂಕ ರಚನೆ fastdiv_ಮೌಲ್ಯಗಳು * div_ne01  = &uctx->kparams->div_ne01;
    ಸ್ಥಿರಾಂಕ ರಚನೆ fastdiv_ಮೌಲ್ಯಗಳು * div_ne02  = &uctx->kparams->div_ne02;
    ಸ್ಥಿರಾಂಕ ರಚನೆ fastdiv_ಮೌಲ್ಯಗಳು * div_ne012 = &uctx->kparams->div_ne012;

    ಸ್ಥಿರಾಂಕ uint32_t src0_max_ಖಂಡ = src0_contig ? uctx->ಖಂಡ : MIN((uint32_t)uctx->ಖಂಡ, ne01);
    ಸ್ಥಿರಾಂಕ uint32_t dst_max_ಖಂಡ  = dst_contig  ? uctx->ಖಂಡ : MIN((uint32_t)uctx->ಖಂಡ, ne1);
    ಸ್ಥಿರಾಂಕ uint32_t ಖಂಡ_2 = MIN(src0_max_ಖಂಡ, dst_max_ಖಂಡ);
    ಆದರೆ (ಖಂಡ_2 == 0) {
        FARF(ERROR, "unary-row-block : current VTCM reservation %zu is too small, needed at least %zu\n",
             uctx->vtcm_src0_ಗಾತ್ರ_ಪ್ರತಿ_thread, src0_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned);
        ಹಿಂತಿರುಗಿ;
    }

    dma_ಸರತಿ * dma_q = octx->ctx->dma[ith];

    ಗಾಗಿ (uint32_t ir = src0_ಆರಂಭ_ಅಡ್ಡಸಾಲು, vtcm_idx = 0; ir < src0_ಕೊನೆ_ಅಡ್ಡಸಾಲು && vtcm_idx < 2; vtcm_idx++) {
        ಸ್ಥಿರಾಂಕ uint32_t ಖಂಡ_ಗಾತ್ರ_3 = unary_ಖಂಡ_ಗಾತ್ರ(ir, src0_ಕೊನೆ_ಅಡ್ಡಸಾಲು, ಖಂಡ_2, src0_contig, dst_contig,
                                                     ne01, div_ne01);

        dma_ಸರತಿ_ತಳ್ಳು(dma_q,
            dma_make_ದತ್ತಾಂಶ(ದತ್ತಾಂಶ_dst, dst_vtcm_ದತ್ತಾಂಶ + (vtcm_idx * dst_vtcm_half_ಗಾತ್ರ)),
            nb1, dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned, dst_ದತ್ತಾಂಶ_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ, 0);

        ಸ್ಥಿರಾಂಕ size_t src0_off = src0_contig ? (ir * nb01) :
            unary_ಅಡ್ಡಸಾಲು_ಸ್ಥಾನಾಂತರ(ir, ne01, ne02, div_ne01, div_ne02, div_ne012, nb01, nb02, nb03);
        dma_ಸರತಿ_ತಳ್ಳು(dma_q,
            dma_make_ದತ್ತಾಂಶ(src0_vtcm_ದತ್ತಾಂಶ + (vtcm_idx * src0_vtcm_half_ಗಾತ್ರ), ದತ್ತಾಂಶ_src + src0_off),
            src0_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned, nb01, src0_ದತ್ತಾಂಶ_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ, ಖಂಡ_ಗಾತ್ರ_3);

        ir += ಖಂಡ_ಗಾತ್ರ_3;
    }

    unary_compute_fn_t compute = (unary_compute_fn_t) uctx->compute;

    ಗಾಗಿ (uint32_t ir = src0_ಆರಂಭ_ಅಡ್ಡಸಾಲು; ir < src0_ಕೊನೆ_ಅಡ್ಡಸಾಲು; ) {
        ಸ್ಥಿರಾಂಕ uint32_t ಖಂಡ_ಗಾತ್ರ_3 = unary_ಖಂಡ_ಗಾತ್ರ(ir, src0_ಕೊನೆ_ಅಡ್ಡಸಾಲು, ಖಂಡ_2, src0_contig, dst_contig,
                                                     ne01, div_ne01);

        ಶೂನ್ಯ * dst_vtcm  = (ಶೂನ್ಯ *) (uintptr_t) dma_ಸರತಿ_ತೆಗೆ(dma_q).src;
        ಶೂನ್ಯ * src0_vtcm = (ಶೂನ್ಯ *) (uintptr_t) dma_ಸರತಿ_ತೆಗೆ(dma_q).dst;

        htp_trace_event_ಆರಂಭ(tr, HTP_TRACE_EVT_HVX_COMP, ir);
        compute(src0_vtcm, dst_vtcm, ಖಂಡ_ಗಾತ್ರ_3, uctx);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ir);

        ಸ್ಥಿರಾಂಕ size_t dst_off = dst_contig ? (ir * nb1) :
            unary_ಅಡ್ಡಸಾಲು_ಸ್ಥಾನಾಂತರ(ir, ne1, ne2, div_ne01, div_ne02, div_ne012, nb1, nb2, nb3);
        dma_ಸರತಿ_ತಳ್ಳು(dma_q,
            dma_make_ದತ್ತಾಂಶ(ದತ್ತಾಂಶ_dst + dst_off, dst_vtcm),
            nb1, dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned, dst_ದತ್ತಾಂಶ_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ, ಖಂಡ_ಗಾತ್ರ_3);

        ಸ್ಥಿರಾಂಕ uint32_t ಮುಂದಿನ_ir = ir + ಖಂಡ_ಗಾತ್ರ_3;
        ಆದರೆ (ಮುಂದಿನ_ir < src0_ಕೊನೆ_ಅಡ್ಡಸಾಲು) {
            ಸ್ಥಿರಾಂಕ uint32_t ಮುಂದಿನ_ಖಂಡ_ಗಾತ್ರ = unary_ಖಂಡ_ಗಾತ್ರ(ಮುಂದಿನ_ir, src0_ಕೊನೆ_ಅಡ್ಡಸಾಲು, ಖಂಡ_2, src0_contig,
                                                              dst_contig, ne01, div_ne01);
            ಸ್ಥಿರಾಂಕ uint32_t pref_ir = ಮುಂದಿನ_ir + ಮುಂದಿನ_ಖಂಡ_ಗಾತ್ರ;
            ಆದರೆ (pref_ir < src0_ಕೊನೆ_ಅಡ್ಡಸಾಲು) {
                ಸ್ಥಿರಾಂಕ uint32_t pref_ಖಂಡ_ಗಾತ್ರ = unary_ಖಂಡ_ಗಾತ್ರ(pref_ir, src0_ಕೊನೆ_ಅಡ್ಡಸಾಲು, ಖಂಡ_2, src0_contig,
                                                                  dst_contig, ne01, div_ne01);
                ಸ್ಥಿರಾಂಕ size_t src0_pref_off = src0_contig ? (pref_ir * nb01) :
                    unary_ಅಡ್ಡಸಾಲು_ಸ್ಥಾನಾಂತರ(pref_ir, ne01, ne02, div_ne01, div_ne02, div_ne012, nb01, nb02, nb03);
                dma_ಸರತಿ_ತಳ್ಳು(dma_q,
                    dma_make_ದತ್ತಾಂಶ(src0_vtcm, ದತ್ತಾಂಶ_src + src0_pref_off),
                    src0_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned, nb01, src0_ದತ್ತಾಂಶ_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ, pref_ಖಂಡ_ಗಾತ್ರ);
            }
        }
        ir += ಖಂಡ_ಗಾತ್ರ_3;
    }

    dma_ಸರತಿ_flush(dma_q);
}

// 2. RMS_NORM_MUL row-block task with weight buffer.
ಸ್ಥಿರ ಶೂನ್ಯ unary_thread_rms_norm_mul_f32(ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ nth, ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ ith, ಶೂನ್ಯ * ದತ್ತಾಂಶ) {
    (ಶೂನ್ಯ) nth;
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_unary_context * uctx = (ಸ್ಥಿರಾಂಕ ರಚನೆ htp_unary_context *) ದತ್ತಾಂಶ;
    ರಚನೆ htp_ops_context * octx = uctx->octx;
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * src = octx->src[0];
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * dst = octx->dst;
    ರಚನೆ htp_thread_trace * tr = &octx->ctx->trace[ith];

    htp_unary_preamble;

    ಸ್ಥಿರಾಂಕ uint32_t src0_nrows_ಪ್ರತಿ_thread = uctx->src0_nrows_ಪ್ರತಿ_thread;
    ಸ್ಥಿರಾಂಕ size_t src0_ದತ್ತಾಂಶ_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ = uctx->src0_ದತ್ತಾಂಶ_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ;
    ಸ್ಥಿರಾಂಕ size_t dst_ದತ್ತಾಂಶ_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ  = uctx->dst_ದತ್ತಾಂಶ_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ;
    ಸ್ಥಿರಾಂಕ size_t src0_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned = uctx->src0_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned;
    ಸ್ಥಿರಾಂಕ size_t dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned  = uctx->dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned;

    ಸ್ಥಿರಾಂಕ uint32_t src0_nrows = uctx->src0_nrows;
    ಸ್ಥಿರಾಂಕ uint32_t src0_ಆರಂಭ_ಅಡ್ಡಸಾಲು = uctx->ಅಡ್ಡಸಾಲು_ಆರಂಭ + src0_nrows_ಪ್ರತಿ_thread * ith;
    ಸ್ಥಿರಾಂಕ uint32_t src0_ಕೊನೆ_ಅಡ್ಡಸಾಲು   = MIN(src0_ಆರಂಭ_ಅಡ್ಡಸಾಲು + src0_nrows_ಪ್ರತಿ_thread, uctx->ಅಡ್ಡಸಾಲು_ಆರಂಭ + src0_nrows);

    ಆದರೆ (src0_ಆರಂಭ_ಅಡ್ಡಸಾಲು >= src0_ಕೊನೆ_ಅಡ್ಡಸಾಲು) {
        ಹಿಂತಿರುಗಿ;
    }

    ಸ್ಥಿರಾಂಕ dma_addr_t ದತ್ತಾಂಶ_src  = uctx->ದತ್ತಾಂಶ_src0;
    ಸ್ಥಿರಾಂಕ dma_addr_t ದತ್ತಾಂಶ_src1 = uctx->ದತ್ತಾಂಶ_src1;
    ಸ್ಥಿರಾಂಕ dma_addr_t ದತ್ತಾಂಶ_dst  = uctx->ದತ್ತಾಂಶ_dst;

    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * src1 = octx->src[1];
    ಸ್ಥಿರಾಂಕ uint32_t nb11 = src1->nb[1];
    ಸ್ಥಿರಾಂಕ uint32_t nb12 = src1->nb[2];
    ಸ್ಥಿರಾಂಕ uint32_t nb13 = src1->nb[3];
    ಸ್ಥಿರಾಂಕ uint32_t nb11_bc = (src1->ne[1] > 1) ? nb11 : 0;
    ಸ್ಥಿರಾಂಕ uint32_t nb12_bc = (src1->ne[2] > 1) ? nb12 : 0;
    ಸ್ಥಿರಾಂಕ uint32_t nb13_bc = (src1->ne[3] > 1) ? nb13 : 0;
    ಸ್ಥಿರಾಂಕ bool src1_contig = ((nb12 == (size_t)ne01 * nb11) && (nb13 == (size_t)ne02 * nb12));

    uint8_t * src0_vtcm_ದತ್ತಾಂಶ = uctx->vtcm_src0 + (ith * uctx->vtcm_src0_ಗಾತ್ರ_ಪ್ರತಿ_thread);
    uint8_t * src1_vtcm_ದತ್ತಾಂಶ = uctx->vtcm_src1 ? (uctx->vtcm_src1 + (ith * uctx->vtcm_src1_ಗಾತ್ರ_ಪ್ರತಿ_thread)) : ಖಾಲಿ;
    uint8_t * dst_vtcm_ದತ್ತಾಂಶ  = uctx->vtcm_dst + (ith * uctx->vtcm_dst_ಗಾತ್ರ_ಪ್ರತಿ_thread);

    ಸ್ಥಿರಾಂಕ size_t src0_vtcm_half_ಗಾತ್ರ = uctx->src0_vtcm_half_ಗಾತ್ರ;
    ಸ್ಥಿರಾಂಕ size_t src1_vtcm_half_ಗಾತ್ರ = uctx->src1_vtcm_half_ಗಾತ್ರ;
    ಸ್ಥಿರಾಂಕ size_t dst_vtcm_half_ಗಾತ್ರ  = uctx->dst_vtcm_half_ಗಾತ್ರ;

    ಸ್ಥಿರಾಂಕ bool src0_contig = (nb02 == (size_t)ne01 * nb01) &&
                             (nb03 == (size_t)ne02 * nb02);
    ಸ್ಥಿರಾಂಕ bool dst_contig  = (nb2  == (size_t)ne1  * nb1)  &&
                             (nb3  == (size_t)ne2  * nb2);

    ಸ್ಥಿರಾಂಕ ರಚನೆ fastdiv_ಮೌಲ್ಯಗಳು * div_ne01  = &uctx->kparams->div_ne01;
    ಸ್ಥಿರಾಂಕ ರಚನೆ fastdiv_ಮೌಲ್ಯಗಳು * div_ne02  = &uctx->kparams->div_ne02;
    ಸ್ಥಿರಾಂಕ ರಚನೆ fastdiv_ಮೌಲ್ಯಗಳು * div_ne012 = &uctx->kparams->div_ne012;

    ಸ್ಥಿರಾಂಕ bool src1_needs_ಅಡ್ಡಸಾಲು_clip = !uctx->ಪ್ರಸಾರ_ತೂಕ && !src1_contig;
    ಸ್ಥಿರಾಂಕ bool ಖಂಡ_src0_contig = src0_contig && !src1_needs_ಅಡ್ಡಸಾಲು_clip;
    ಸ್ಥಿರಾಂಕ bool ಖಂಡ_dst_contig  = dst_contig  && !src1_needs_ಅಡ್ಡಸಾಲು_clip;

    ಸ್ಥಿರಾಂಕ uint32_t src0_max_ಖಂಡ = ಖಂಡ_src0_contig ? uctx->ಖಂಡ : MIN((uint32_t)uctx->ಖಂಡ, ne01);
    ಸ್ಥಿರಾಂಕ uint32_t dst_max_ಖಂಡ  = ಖಂಡ_dst_contig  ? uctx->ಖಂಡ : MIN((uint32_t)uctx->ಖಂಡ, ne1);
    ಸ್ಥಿರಾಂಕ uint32_t ಖಂಡ_2 = MIN(src0_max_ಖಂಡ, dst_max_ಖಂಡ);
    ಆದರೆ (ಖಂಡ_2 == 0) {
        FARF(ERROR, "unary-rms-norm-mul : current VTCM reservation %zu is too small, needed at least %zu\n",
             uctx->vtcm_src0_ಗಾತ್ರ_ಪ್ರತಿ_thread, src0_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned);
        ಹಿಂತಿರುಗಿ;
    }

    dma_ಸರತಿ * dma_q = octx->ctx->dma[ith];

    ಆದರೆ (uctx->ಪ್ರಸಾರ_ತೂಕ) {
        dma_ಸರತಿ_ತಳ್ಳು(dma_q, dma_make_ದತ್ತಾಂಶ(src1_vtcm_ದತ್ತಾಂಶ, ದತ್ತಾಂಶ_src1),
                       uctx->src1_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned, 0, uctx->src1_ದತ್ತಾಂಶ_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ, 1);
        dma_ಸರತಿ_flush(dma_q);
    }

    ಗಾಗಿ (uint32_t ir = src0_ಆರಂಭ_ಅಡ್ಡಸಾಲು, vtcm_idx = 0; ir < src0_ಕೊನೆ_ಅಡ್ಡಸಾಲು && vtcm_idx < 2; vtcm_idx++) {
        ಸ್ಥಿರಾಂಕ uint32_t ಖಂಡ_ಗಾತ್ರ_3 = unary_ಖಂಡ_ಗಾತ್ರ(ir, src0_ಕೊನೆ_ಅಡ್ಡಸಾಲು, ಖಂಡ_2, ಖಂಡ_src0_contig, ಖಂಡ_dst_contig,
                                                     ne01, div_ne01);

        dma_ಸರತಿ_ತಳ್ಳು(dma_q,
            dma_make_ದತ್ತಾಂಶ(ದತ್ತಾಂಶ_dst, dst_vtcm_ದತ್ತಾಂಶ + (vtcm_idx * dst_vtcm_half_ಗಾತ್ರ)),
            nb1, dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned, dst_ದತ್ತಾಂಶ_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ, 0);

        ಸ್ಥಿರಾಂಕ size_t src0_off = src0_contig ? (ir * nb01) :
            unary_ಅಡ್ಡಸಾಲು_ಸ್ಥಾನಾಂತರ(ir, ne01, ne02, div_ne01, div_ne02, div_ne012, nb01, nb02, nb03);
        dma_ಸರತಿ_ತಳ್ಳು(dma_q,
            dma_make_ದತ್ತಾಂಶ(src0_vtcm_ದತ್ತಾಂಶ + (vtcm_idx * src0_vtcm_half_ಗಾತ್ರ), ದತ್ತಾಂಶ_src + src0_off),
            src0_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned, nb01, src0_ದತ್ತಾಂಶ_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ, ಖಂಡ_ಗಾತ್ರ_3);

        ಆದರೆ (!uctx->ಪ್ರಸಾರ_ತೂಕ) {
            ಸ್ಥಿರಾಂಕ size_t src1_off = src1_contig ? (ir * nb11) :
                unary_ಅಡ್ಡಸಾಲು_ಸ್ಥಾನಾಂತರ(ir, ne01, ne02, div_ne01, div_ne02, div_ne012, nb11_bc, nb12_bc, nb13_bc);
            dma_ಸರತಿ_ತಳ್ಳು(dma_q,
                dma_make_ದತ್ತಾಂಶ(src1_vtcm_ದತ್ತಾಂಶ + (vtcm_idx * src1_vtcm_half_ಗಾತ್ರ), ದತ್ತಾಂಶ_src1 + src1_off),
                uctx->src1_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned, nb11, uctx->src1_ದತ್ತಾಂಶ_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ, ಖಂಡ_ಗಾತ್ರ_3);
        }

        ir += ಖಂಡ_ಗಾತ್ರ_3;
    }

    unary_rms_norm_mul_compute_fn_t compute = (unary_rms_norm_mul_compute_fn_t) uctx->compute;

    ಗಾಗಿ (uint32_t ir = src0_ಆರಂಭ_ಅಡ್ಡಸಾಲು; ir < src0_ಕೊನೆ_ಅಡ್ಡಸಾಲು; ) {
        ಸ್ಥಿರಾಂಕ uint32_t ಖಂಡ_ಗಾತ್ರ_3 = unary_ಖಂಡ_ಗಾತ್ರ(ir, src0_ಕೊನೆ_ಅಡ್ಡಸಾಲು, ಖಂಡ_2, ಖಂಡ_src0_contig, ಖಂಡ_dst_contig,
                                                     ne01, div_ne01);

        ಶೂನ್ಯ * dst_vtcm  = (ಶೂನ್ಯ *) (uintptr_t) dma_ಸರತಿ_ತೆಗೆ(dma_q).src;
        ಶೂನ್ಯ * src0_vtcm = (ಶೂನ್ಯ *) (uintptr_t) dma_ಸರತಿ_ತೆಗೆ(dma_q).dst;
        ಶೂನ್ಯ * src1_vtcm = ಖಾಲಿ;
        ಆದರೆ (!uctx->ಪ್ರಸಾರ_ತೂಕ) {
            src1_vtcm = (ಶೂನ್ಯ *) (uintptr_t) dma_ಸರತಿ_ತೆಗೆ(dma_q).dst;
        }

        htp_trace_event_ಆರಂಭ(tr, HTP_TRACE_EVT_HVX_COMP, ir);
        ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * w = uctx->ಪ್ರಸಾರ_ತೂಕ ? (ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *) src1_vtcm_ದತ್ತಾಂಶ : src1_vtcm;
        compute(src0_vtcm, w, dst_vtcm, ಖಂಡ_ಗಾತ್ರ_3, uctx);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ir);

        ಸ್ಥಿರಾಂಕ size_t dst_off = dst_contig ? (ir * nb1) :
            unary_ಅಡ್ಡಸಾಲು_ಸ್ಥಾನಾಂತರ(ir, ne1, ne2, div_ne01, div_ne02, div_ne012, nb1, nb2, nb3);
        dma_ಸರತಿ_ತಳ್ಳು(dma_q,
            dma_make_ದತ್ತಾಂಶ(ದತ್ತಾಂಶ_dst + dst_off, dst_vtcm),
            nb1, dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned, dst_ದತ್ತಾಂಶ_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ, ಖಂಡ_ಗಾತ್ರ_3);

        ಸ್ಥಿರಾಂಕ uint32_t ಮುಂದಿನ_ir = ir + ಖಂಡ_ಗಾತ್ರ_3;
        ಆದರೆ (ಮುಂದಿನ_ir < src0_ಕೊನೆ_ಅಡ್ಡಸಾಲು) {
            ಸ್ಥಿರಾಂಕ uint32_t ಮುಂದಿನ_ಖಂಡ_ಗಾತ್ರ = unary_ಖಂಡ_ಗಾತ್ರ(ಮುಂದಿನ_ir, src0_ಕೊನೆ_ಅಡ್ಡಸಾಲು, ಖಂಡ_2, ಖಂಡ_src0_contig,
                                                              ಖಂಡ_dst_contig, ne01, div_ne01);
            ಸ್ಥಿರಾಂಕ uint32_t pref_ir = ಮುಂದಿನ_ir + ಮುಂದಿನ_ಖಂಡ_ಗಾತ್ರ;
            ಆದರೆ (pref_ir < src0_ಕೊನೆ_ಅಡ್ಡಸಾಲು) {
                ಸ್ಥಿರಾಂಕ uint32_t pref_ಖಂಡ_ಗಾತ್ರ = unary_ಖಂಡ_ಗಾತ್ರ(pref_ir, src0_ಕೊನೆ_ಅಡ್ಡಸಾಲು, ಖಂಡ_2, ಖಂಡ_src0_contig,
                                                                  ಖಂಡ_dst_contig, ne01, div_ne01);
                ಸ್ಥಿರಾಂಕ size_t src0_pref_off = src0_contig ? (pref_ir * nb01) :
                    unary_ಅಡ್ಡಸಾಲು_ಸ್ಥಾನಾಂತರ(pref_ir, ne01, ne02, div_ne01, div_ne02, div_ne012, nb01, nb02, nb03);
                dma_ಸರತಿ_ತಳ್ಳು(dma_q,
                    dma_make_ದತ್ತಾಂಶ(src0_vtcm, ದತ್ತಾಂಶ_src + src0_pref_off),
                    src0_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned, nb01, src0_ದತ್ತಾಂಶ_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ, pref_ಖಂಡ_ಗಾತ್ರ);

                ಆದರೆ (!uctx->ಪ್ರಸಾರ_ತೂಕ) {
                    ಸ್ಥಿರಾಂಕ size_t src1_pref_off = src1_contig ? (pref_ir * nb11) :
                        unary_ಅಡ್ಡಸಾಲು_ಸ್ಥಾನಾಂತರ(pref_ir, ne01, ne02, div_ne01, div_ne02, div_ne012, nb11_bc, nb12_bc,
                                          nb13_bc);
                    dma_ಸರತಿ_ತಳ್ಳು(dma_q,
                        dma_make_ದತ್ತಾಂಶ(src1_vtcm, ದತ್ತಾಂಶ_src1 + src1_pref_off),
                        uctx->src1_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned, nb11, uctx->src1_ದತ್ತಾಂಶ_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ, pref_ಖಂಡ_ಗಾತ್ರ);
                }
            }
        }
        ir += ಖಂಡ_ಗಾತ್ರ_3;
    }

    dma_ಸರತಿ_flush(dma_q);
}

// 3. TRI row-block task with row index ir.
ಸ್ಥಿರ ಶೂನ್ಯ unary_thread_tri_f32(ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ nth, ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ ith, ಶೂನ್ಯ * ದತ್ತಾಂಶ) {
    (ಶೂನ್ಯ) nth;
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_unary_context * uctx = (ಸ್ಥಿರಾಂಕ ರಚನೆ htp_unary_context *) ದತ್ತಾಂಶ;
    ರಚನೆ htp_ops_context * octx = uctx->octx;
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * src = octx->src[0];
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * dst = octx->dst;
    ರಚನೆ htp_thread_trace * tr = &octx->ctx->trace[ith];

    htp_unary_preamble;

    ಸ್ಥಿರಾಂಕ uint32_t src0_nrows_ಪ್ರತಿ_thread = uctx->src0_nrows_ಪ್ರತಿ_thread;
    ಸ್ಥಿರಾಂಕ size_t src0_ದತ್ತಾಂಶ_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ = uctx->src0_ದತ್ತಾಂಶ_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ;
    ಸ್ಥಿರಾಂಕ size_t dst_ದತ್ತಾಂಶ_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ  = uctx->dst_ದತ್ತಾಂಶ_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ;
    ಸ್ಥಿರಾಂಕ size_t src0_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned = uctx->src0_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned;
    ಸ್ಥಿರಾಂಕ size_t dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned  = uctx->dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned;

    ಸ್ಥಿರಾಂಕ uint32_t src0_nrows = uctx->src0_nrows;
    ಸ್ಥಿರಾಂಕ uint32_t src0_ಆರಂಭ_ಅಡ್ಡಸಾಲು = uctx->ಅಡ್ಡಸಾಲು_ಆರಂಭ + src0_nrows_ಪ್ರತಿ_thread * ith;
    ಸ್ಥಿರಾಂಕ uint32_t src0_ಕೊನೆ_ಅಡ್ಡಸಾಲು   = MIN(src0_ಆರಂಭ_ಅಡ್ಡಸಾಲು + src0_nrows_ಪ್ರತಿ_thread, uctx->ಅಡ್ಡಸಾಲು_ಆರಂಭ + src0_nrows);

    ಆದರೆ (src0_ಆರಂಭ_ಅಡ್ಡಸಾಲು >= src0_ಕೊನೆ_ಅಡ್ಡಸಾಲು) {
        ಹಿಂತಿರುಗಿ;
    }

    ಸ್ಥಿರಾಂಕ dma_addr_t ದತ್ತಾಂಶ_src = uctx->ದತ್ತಾಂಶ_src0;
    ಸ್ಥಿರಾಂಕ dma_addr_t ದತ್ತಾಂಶ_dst = uctx->ದತ್ತಾಂಶ_dst;

    uint8_t * src0_vtcm_ದತ್ತಾಂಶ = uctx->vtcm_src0 + (ith * uctx->vtcm_src0_ಗಾತ್ರ_ಪ್ರತಿ_thread);
    uint8_t * dst_vtcm_ದತ್ತಾಂಶ  = uctx->vtcm_dst + (ith * uctx->vtcm_dst_ಗಾತ್ರ_ಪ್ರತಿ_thread);

    ಸ್ಥಿರಾಂಕ size_t src0_vtcm_half_ಗಾತ್ರ = uctx->src0_vtcm_half_ಗಾತ್ರ;
    ಸ್ಥಿರಾಂಕ size_t dst_vtcm_half_ಗಾತ್ರ  = uctx->dst_vtcm_half_ಗಾತ್ರ;

    ಸ್ಥಿರಾಂಕ bool src0_contig = (nb02 == (size_t)ne01 * nb01) &&
                             (nb03 == (size_t)ne02 * nb02);
    ಸ್ಥಿರಾಂಕ bool dst_contig  = (nb2  == (size_t)ne1  * nb1)  &&
                             (nb3  == (size_t)ne2  * nb2);

    ಸ್ಥಿರಾಂಕ ರಚನೆ fastdiv_ಮೌಲ್ಯಗಳು * div_ne01  = &uctx->kparams->div_ne01;
    ಸ್ಥಿರಾಂಕ ರಚನೆ fastdiv_ಮೌಲ್ಯಗಳು * div_ne02  = &uctx->kparams->div_ne02;
    ಸ್ಥಿರಾಂಕ ರಚನೆ fastdiv_ಮೌಲ್ಯಗಳು * div_ne012 = &uctx->kparams->div_ne012;

    ಸ್ಥಿರಾಂಕ uint32_t src0_max_ಖಂಡ = src0_contig ? uctx->ಖಂಡ : MIN((uint32_t)uctx->ಖಂಡ, ne01);
    ಸ್ಥಿರಾಂಕ uint32_t dst_max_ಖಂಡ  = dst_contig  ? uctx->ಖಂಡ : MIN((uint32_t)uctx->ಖಂಡ, ne1);
    ಸ್ಥಿರಾಂಕ uint32_t ಖಂಡ_2 = MIN(src0_max_ಖಂಡ, dst_max_ಖಂಡ);
    ಆದರೆ (ಖಂಡ_2 == 0) {
        FARF(ERROR, "unary-tri : current VTCM reservation %zu is too small, needed at least %zu\n",
             uctx->vtcm_src0_ಗಾತ್ರ_ಪ್ರತಿ_thread, src0_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned);
        ಹಿಂತಿರುಗಿ;
    }

    dma_ಸರತಿ * dma_q = octx->ctx->dma[ith];

    ಗಾಗಿ (uint32_t ir = src0_ಆರಂಭ_ಅಡ್ಡಸಾಲು, vtcm_idx = 0; ir < src0_ಕೊನೆ_ಅಡ್ಡಸಾಲು && vtcm_idx < 2; vtcm_idx++) {
        ಸ್ಥಿರಾಂಕ uint32_t ಖಂಡ_ಗಾತ್ರ_3 = unary_ಖಂಡ_ಗಾತ್ರ(ir, src0_ಕೊನೆ_ಅಡ್ಡಸಾಲು, ಖಂಡ_2, src0_contig, dst_contig,
                                                     ne01, div_ne01);

        dma_ಸರತಿ_ತಳ್ಳು(dma_q,
            dma_make_ದತ್ತಾಂಶ(ದತ್ತಾಂಶ_dst, dst_vtcm_ದತ್ತಾಂಶ + (vtcm_idx * dst_vtcm_half_ಗಾತ್ರ)),
            nb1, dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned, dst_ದತ್ತಾಂಶ_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ, 0);

        ಸ್ಥಿರಾಂಕ size_t src0_off = src0_contig ? (ir * nb01) :
            unary_ಅಡ್ಡಸಾಲು_ಸ್ಥಾನಾಂತರ(ir, ne01, ne02, div_ne01, div_ne02, div_ne012, nb01, nb02, nb03);
        dma_ಸರತಿ_ತಳ್ಳು(dma_q,
            dma_make_ದತ್ತಾಂಶ(src0_vtcm_ದತ್ತಾಂಶ + (vtcm_idx * src0_vtcm_half_ಗಾತ್ರ), ದತ್ತಾಂಶ_src + src0_off),
            src0_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned, nb01, src0_ದತ್ತಾಂಶ_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ, ಖಂಡ_ಗಾತ್ರ_3);

        ir += ಖಂಡ_ಗಾತ್ರ_3;
    }

    unary_tri_compute_fn_t compute = (unary_tri_compute_fn_t) uctx->compute;

    ಗಾಗಿ (uint32_t ir = src0_ಆರಂಭ_ಅಡ್ಡಸಾಲು; ir < src0_ಕೊನೆ_ಅಡ್ಡಸಾಲು; ) {
        ಸ್ಥಿರಾಂಕ uint32_t ಖಂಡ_ಗಾತ್ರ_3 = unary_ಖಂಡ_ಗಾತ್ರ(ir, src0_ಕೊನೆ_ಅಡ್ಡಸಾಲು, ಖಂಡ_2, src0_contig, dst_contig,
                                                     ne01, div_ne01);

        ಶೂನ್ಯ * dst_vtcm  = (ಶೂನ್ಯ *) (uintptr_t) dma_ಸರತಿ_ತೆಗೆ(dma_q).src;
        ಶೂನ್ಯ * src0_vtcm = (ಶೂನ್ಯ *) (uintptr_t) dma_ಸರತಿ_ತೆಗೆ(dma_q).dst;

        htp_trace_event_ಆರಂಭ(tr, HTP_TRACE_EVT_HVX_COMP, ir);
        compute(src0_vtcm, dst_vtcm, ಖಂಡ_ಗಾತ್ರ_3, ir, uctx);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ir);

        ಸ್ಥಿರಾಂಕ size_t dst_off = dst_contig ? (ir * nb1) :
            unary_ಅಡ್ಡಸಾಲು_ಸ್ಥಾನಾಂತರ(ir, ne1, ne2, div_ne01, div_ne02, div_ne012, nb1, nb2, nb3);
        dma_ಸರತಿ_ತಳ್ಳು(dma_q,
            dma_make_ದತ್ತಾಂಶ(ದತ್ತಾಂಶ_dst + dst_off, dst_vtcm),
            nb1, dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned, dst_ದತ್ತಾಂಶ_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ, ಖಂಡ_ಗಾತ್ರ_3);

        ಸ್ಥಿರಾಂಕ uint32_t ಮುಂದಿನ_ir = ir + ಖಂಡ_ಗಾತ್ರ_3;
        ಆದರೆ (ಮುಂದಿನ_ir < src0_ಕೊನೆ_ಅಡ್ಡಸಾಲು) {
            ಸ್ಥಿರಾಂಕ uint32_t ಮುಂದಿನ_ಖಂಡ_ಗಾತ್ರ = unary_ಖಂಡ_ಗಾತ್ರ(ಮುಂದಿನ_ir, src0_ಕೊನೆ_ಅಡ್ಡಸಾಲು, ಖಂಡ_2, src0_contig,
                                                              dst_contig, ne01, div_ne01);
            ಸ್ಥಿರಾಂಕ uint32_t pref_ir = ಮುಂದಿನ_ir + ಮುಂದಿನ_ಖಂಡ_ಗಾತ್ರ;
            ಆದರೆ (pref_ir < src0_ಕೊನೆ_ಅಡ್ಡಸಾಲು) {
                ಸ್ಥಿರಾಂಕ uint32_t pref_ಖಂಡ_ಗಾತ್ರ = unary_ಖಂಡ_ಗಾತ್ರ(pref_ir, src0_ಕೊನೆ_ಅಡ್ಡಸಾಲು, ಖಂಡ_2, src0_contig,
                                                                  dst_contig, ne01, div_ne01);
                ಸ್ಥಿರಾಂಕ size_t src0_pref_off = src0_contig ? (pref_ir * nb01) :
                    unary_ಅಡ್ಡಸಾಲು_ಸ್ಥಾನಾಂತರ(pref_ir, ne01, ne02, div_ne01, div_ne02, div_ne012, nb01, nb02, nb03);
                dma_ಸರತಿ_ತಳ್ಳು(dma_q,
                    dma_make_ದತ್ತಾಂಶ(src0_vtcm, ದತ್ತಾಂಶ_src + src0_pref_off),
                    src0_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned, nb01, src0_ದತ್ತಾಂಶ_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ, pref_ಖಂಡ_ಗಾತ್ರ);
            }
        }
        ir += ಖಂಡ_ಗಾತ್ರ_3;
    }

    dma_ಸರತಿ_flush(dma_q);
}

// 4. Pointwise tiled unary task.
ಸ್ಥಿರ ಶೂನ್ಯ unary_thread_tiled(ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ nth, ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ ith, ಶೂನ್ಯ * ದತ್ತಾಂಶ) {
    (ಶೂನ್ಯ) nth;
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_unary_context * uctx = (ಸ್ಥಿರಾಂಕ ರಚನೆ htp_unary_context *) ದತ್ತಾಂಶ;
    ರಚನೆ htp_ops_context * octx = uctx->octx;
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * src = octx->src[0];
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * dst = octx->dst;
    ರಚನೆ htp_thread_trace * tr = &octx->ctx->trace[ith];

    htp_unary_preamble;

    ಸ್ಥಿರಾಂಕ uint32_t src0_nrows_ಪ್ರತಿ_thread = uctx->src0_nrows_ಪ್ರತಿ_thread;
    ಸ್ಥಿರಾಂಕ uint32_t ಲಂಬಸಾಲು_tile              = uctx->ಲಂಬಸಾಲು_tile;

    ಸ್ಥಿರಾಂಕ uint32_t src0_nrows     = uctx->src0_nrows;
    ಸ್ಥಿರಾಂಕ uint32_t src0_ಆರಂಭ_ಅಡ್ಡಸಾಲು = uctx->ಅಡ್ಡಸಾಲು_ಆರಂಭ + src0_nrows_ಪ್ರತಿ_thread * ith;
    ಸ್ಥಿರಾಂಕ uint32_t src0_ಕೊನೆ_ಅಡ್ಡಸಾಲು   = MIN(src0_ಆರಂಭ_ಅಡ್ಡಸಾಲು + src0_nrows_ಪ್ರತಿ_thread, uctx->ಅಡ್ಡಸಾಲು_ಆರಂಭ + src0_nrows);

    ಆದರೆ (src0_ಆರಂಭ_ಅಡ್ಡಸಾಲು >= src0_ಕೊನೆ_ಅಡ್ಡಸಾಲು) {
        ಹಿಂತಿರುಗಿ;
    }

    ಸ್ಥಿರಾಂಕ dma_addr_t ದತ್ತಾಂಶ_src = uctx->ದತ್ತಾಂಶ_src0;
    ಸ್ಥಿರಾಂಕ dma_addr_t ದತ್ತಾಂಶ_dst = uctx->ದತ್ತಾಂಶ_dst;

    uint8_t * src0_vtcm_ದತ್ತಾಂಶ = uctx->vtcm_src0 + (ith * uctx->vtcm_src0_ಗಾತ್ರ_ಪ್ರತಿ_thread);
    uint8_t * dst_vtcm_ದತ್ತಾಂಶ  = uctx->vtcm_dst + (ith * uctx->vtcm_dst_ಗಾತ್ರ_ಪ್ರತಿ_thread);

    ಸ್ಥಿರಾಂಕ size_t src0_half = uctx->src0_vtcm_half_ಗಾತ್ರ;
    ಸ್ಥಿರಾಂಕ size_t dst_half  = uctx->dst_vtcm_half_ಗಾತ್ರ;

    dma_ಸರತಿ * dma_q = octx->ctx->dma[ith];

    ಸ್ಥಿರಾಂಕ ರಚನೆ fastdiv_ಮೌಲ್ಯಗಳು * div_ne01  = &uctx->kparams->div_ne01;
    ಸ್ಥಿರಾಂಕ ರಚನೆ fastdiv_ಮೌಲ್ಯಗಳು * div_ne02  = &uctx->kparams->div_ne02;
    ಸ್ಥಿರಾಂಕ ರಚನೆ fastdiv_ಮೌಲ್ಯಗಳು * div_ne012 = &uctx->kparams->div_ne012;
    ಸ್ಥಿರಾಂಕ ರಚನೆ fastdiv_ಮೌಲ್ಯಗಳು * div_tpr   = &uctx->kparams->div_tpr;

    ಸ್ಥಿರಾಂಕ uint32_t tiles_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು = (ne0 + ಲಂಬಸಾಲು_tile - 1) / ಲಂಬಸಾಲು_tile;

    ಸ್ಥಿರಾಂಕ bool src0_contig = (nb02 == (size_t)ne01 * nb01) &&
                             (nb03 == (size_t)ne02 * nb02);
    ಸ್ಥಿರಾಂಕ bool dst_contig  = (nb2  == (size_t)ne1  * nb1)  &&
                             (nb3  == (size_t)ne2  * nb2);

    ಸ್ಥಿರಾಂಕ uint32_t ಒಟ್ಟು_tiles = (src0_ಕೊನೆ_ಅಡ್ಡಸಾಲು - src0_ಆರಂಭ_ಅಡ್ಡಸಾಲು) * tiles_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು;

    ಗಾಗಿ (uint32_t t = 0, vtcm_idx = 0; t < ಒಟ್ಟು_tiles && vtcm_idx < 2; t++, vtcm_idx++) {
        ಸ್ಥಿರಾಂಕ uint32_t ಅಡ್ಡಸಾಲು  = src0_ಆರಂಭ_ಅಡ್ಡಸಾಲು + t / tiles_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು;
        ಸ್ಥಿರಾಂಕ uint32_t ಲಂಬಸಾಲು  = (t % tiles_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು) * ಲಂಬಸಾಲು_tile;
        ಸ್ಥಿರಾಂಕ uint32_t tw   = MIN(ಲಂಬಸಾಲು_tile, ne0 - ಲಂಬಸಾಲು);
        ಸ್ಥಿರಾಂಕ size_t   tb   = (size_t) tw * ಗಾತ್ರ(ದಶಮಾಂಶ);
        ಸ್ಥಿರಾಂಕ size_t   soff = (src0_contig ? (ಅಡ್ಡಸಾಲು * nb01) :
                               unary_ಅಡ್ಡಸಾಲು_ಸ್ಥಾನಾಂತರ(ಅಡ್ಡಸಾಲು, ne01, ne02, div_ne01, div_ne02, div_ne012, nb01, nb02, nb03)) +
                               (size_t) ಲಂಬಸಾಲು * ಗಾತ್ರ(ದಶಮಾಂಶ);

        dma_ಸರತಿ_ತಳ್ಳು(dma_q, dma_make_ದತ್ತಾಂಶ(ದತ್ತಾಂಶ_dst, dst_vtcm_ದತ್ತಾಂಶ + (vtcm_idx * dst_half)), 0, 0, 0, 0);
        dma_ಸರತಿ_ತಳ್ಳು(dma_q, dma_make_ದತ್ತಾಂಶ(src0_vtcm_ದತ್ತಾಂಶ + (vtcm_idx * src0_half), ದತ್ತಾಂಶ_src + soff), tb, tb, tb, 1);
    }

    unary_tile_compute_fn_t compute = (unary_tile_compute_fn_t) uctx->compute;

    uint32_t ಅಡ್ಡಸಾಲು = src0_ಆರಂಭ_ಅಡ್ಡಸಾಲು;
    uint32_t ಲಂಬಸಾಲು = 0;
    uint32_t tile_ಒಳ_ಅಡ್ಡಸಾಲು = 0;

    uint32_t prow = src0_ಆರಂಭ_ಅಡ್ಡಸಾಲು + fastdiv(2, div_tpr);
    uint32_t pcol = fastmodulo(2, tiles_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು, div_tpr) * ಲಂಬಸಾಲು_tile;
    uint32_t ptile_ಒಳ_ಅಡ್ಡಸಾಲು = fastmodulo(2, tiles_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು, div_tpr);

    ಗಾಗಿ (uint32_t t = 0; t < ಒಟ್ಟು_tiles; t++) {
        ಶೂನ್ಯ * dst_vtcm = (ಶೂನ್ಯ *) (uintptr_t) dma_ಸರತಿ_ತೆಗೆ(dma_q).src;
        ಶೂನ್ಯ * src_vtcm = (ಶೂನ್ಯ *) (uintptr_t) dma_ಸರತಿ_ತೆಗೆ(dma_q).dst;

        ಸ್ಥಿರಾಂಕ uint32_t tw = MIN(ಲಂಬಸಾಲು_tile, ne0 - ಲಂಬಸಾಲು);

        htp_trace_event_ಆರಂಭ(tr, HTP_TRACE_EVT_HVX_COMP, t);
        compute(dst_vtcm, src_vtcm, tw, uctx);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, t);

        ಸ್ಥಿರಾಂಕ size_t doff = (dst_contig ? (ಅಡ್ಡಸಾಲು * nb1) :
                             unary_ಅಡ್ಡಸಾಲು_ಸ್ಥಾನಾಂತರ(ಅಡ್ಡಸಾಲು, ne1, ne2, div_ne01, div_ne02, div_ne012, nb1, nb2, nb3)) +
                             (size_t) ಲಂಬಸಾಲು * ಗಾತ್ರ(ದಶಮಾಂಶ);
        ಸ್ಥಿರಾಂಕ size_t tb   = (size_t) tw * ಗಾತ್ರ(ದಶಮಾಂಶ);
        dma_ಸರತಿ_ತಳ್ಳು(dma_q, dma_make_ದತ್ತಾಂಶ(ದತ್ತಾಂಶ_dst + doff, dst_vtcm), tb, tb, tb, 1);

        ಸ್ಥಿರಾಂಕ uint32_t pt = t + 2;
        ಆದರೆ (pt < ಒಟ್ಟು_tiles) {
            ಸ್ಥಿರಾಂಕ uint32_t ptw  = MIN(ಲಂಬಸಾಲು_tile, ne0 - pcol);
            ಸ್ಥಿರಾಂಕ size_t   ptb  = (size_t) ptw * ಗಾತ್ರ(ದಶಮಾಂಶ);
            ಸ್ಥಿರಾಂಕ size_t   psoff = (src0_contig ? (prow * nb01) :
                                    unary_ಅಡ್ಡಸಾಲು_ಸ್ಥಾನಾಂತರ(prow, ne01, ne02, div_ne01, div_ne02, div_ne012, nb01, nb02,
                                                     nb03)) +
                                   (size_t) pcol * ಗಾತ್ರ(ದಶಮಾಂಶ);
            dma_ಸರತಿ_ತಳ್ಳು(dma_q, dma_make_ದತ್ತಾಂಶ(src_vtcm, ದತ್ತಾಂಶ_src + psoff), ptb, ptb, ptb, 1);
        }

        tile_ಒಳ_ಅಡ್ಡಸಾಲು++;
        ಲಂಬಸಾಲು += ಲಂಬಸಾಲು_tile;
        ಆದರೆ (tile_ಒಳ_ಅಡ್ಡಸಾಲು == tiles_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು) {
            tile_ಒಳ_ಅಡ್ಡಸಾಲು = 0;
            ಲಂಬಸಾಲು = 0;
            ಅಡ್ಡಸಾಲು++;
        }

        ptile_ಒಳ_ಅಡ್ಡಸಾಲು++;
        pcol += ಲಂಬಸಾಲು_tile;
        ಆದರೆ (ptile_ಒಳ_ಅಡ್ಡಸಾಲು == tiles_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು) {
            ptile_ಒಳ_ಅಡ್ಡಸಾಲು = 0;
            pcol = 0;
            prow++;
        }
    }

    dma_ಸರತಿ_flush(dma_q);
}

// 5. TRI tiled task.
ಸ್ಥಿರ ಶೂನ್ಯ unary_thread_tiled_tri_f32(ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ nth, ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ ith, ಶೂನ್ಯ * ದತ್ತಾಂಶ) {
    (ಶೂನ್ಯ) nth;
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_unary_context * uctx = (ಸ್ಥಿರಾಂಕ ರಚನೆ htp_unary_context *) ದತ್ತಾಂಶ;
    ರಚನೆ htp_ops_context * octx = uctx->octx;
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * src = octx->src[0];
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * dst = octx->dst;
    ರಚನೆ htp_thread_trace * tr = &octx->ctx->trace[ith];

    htp_unary_preamble;

    ಸ್ಥಿರಾಂಕ uint32_t src0_nrows_ಪ್ರತಿ_thread = uctx->src0_nrows_ಪ್ರತಿ_thread;
    ಸ್ಥಿರಾಂಕ int32_t * op_params            = octx->op_params;
    ಸ್ಥಿರಾಂಕ uint32_t ಲಂಬಸಾಲು_tile              = uctx->ಲಂಬಸಾಲು_tile;

    ಸ್ಥಿರಾಂಕ uint32_t src0_nrows     = uctx->src0_nrows;
    ಸ್ಥಿರಾಂಕ uint32_t src0_ಆರಂಭ_ಅಡ್ಡಸಾಲು = uctx->ಅಡ್ಡಸಾಲು_ಆರಂಭ + src0_nrows_ಪ್ರತಿ_thread * ith;
    ಸ್ಥಿರಾಂಕ uint32_t src0_ಕೊನೆ_ಅಡ್ಡಸಾಲು   = MIN(src0_ಆರಂಭ_ಅಡ್ಡಸಾಲು + src0_nrows_ಪ್ರತಿ_thread, uctx->ಅಡ್ಡಸಾಲು_ಆರಂಭ + src0_nrows);

    ಆದರೆ (src0_ಆರಂಭ_ಅಡ್ಡಸಾಲು >= src0_ಕೊನೆ_ಅಡ್ಡಸಾಲು) {
        ಹಿಂತಿರುಗಿ;
    }

    ಸ್ಥಿರಾಂಕ dma_addr_t ದತ್ತಾಂಶ_src = uctx->ದತ್ತಾಂಶ_src0;
    ಸ್ಥಿರಾಂಕ dma_addr_t ದತ್ತಾಂಶ_dst = uctx->ದತ್ತಾಂಶ_dst;

    uint8_t * src0_vtcm_ದತ್ತಾಂಶ = uctx->vtcm_src0 + (ith * uctx->vtcm_src0_ಗಾತ್ರ_ಪ್ರತಿ_thread);
    uint8_t * dst_vtcm_ದತ್ತಾಂಶ  = uctx->vtcm_dst + (ith * uctx->vtcm_dst_ಗಾತ್ರ_ಪ್ರತಿ_thread);

    ಸ್ಥಿರಾಂಕ size_t src0_half = uctx->src0_vtcm_half_ಗಾತ್ರ;
    ಸ್ಥಿರಾಂಕ size_t dst_half  = uctx->dst_vtcm_half_ಗಾತ್ರ;

    dma_ಸರತಿ * dma_q = octx->ctx->dma[ith];

    ಸ್ಥಿರಾಂಕ ರಚನೆ fastdiv_ಮೌಲ್ಯಗಳು * div_ne01  = &uctx->kparams->div_ne01;
    ಸ್ಥಿರಾಂಕ ರಚನೆ fastdiv_ಮೌಲ್ಯಗಳು * div_ne02  = &uctx->kparams->div_ne02;
    ಸ್ಥಿರಾಂಕ ರಚನೆ fastdiv_ಮೌಲ್ಯಗಳು * div_ne012 = &uctx->kparams->div_ne012;
    ಸ್ಥಿರಾಂಕ ರಚನೆ fastdiv_ಮೌಲ್ಯಗಳು * div_tpr   = &uctx->kparams->div_tpr;

    ಸ್ಥಿರಾಂಕ uint32_t tiles_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು = (ne0 + ಲಂಬಸಾಲು_tile - 1) / ಲಂಬಸಾಲು_tile;
    ಸ್ಥಿರಾಂಕ int32_t  tri_ttype     = op_params[0];

    ಸ್ಥಿರಾಂಕ bool src0_contig = (nb02 == (size_t)ne01 * nb01) &&
                             (nb03 == (size_t)ne02 * nb02);
    ಸ್ಥಿರಾಂಕ bool dst_contig  = (nb2  == (size_t)ne1  * nb1)  &&
                             (nb3  == (size_t)ne2  * nb2);

    ಸ್ಥಿರಾಂಕ uint32_t ಒಟ್ಟು_tiles = (src0_ಕೊನೆ_ಅಡ್ಡಸಾಲು - src0_ಆರಂಭ_ಅಡ್ಡಸಾಲು) * tiles_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು;

    ಗಾಗಿ (uint32_t t = 0, vtcm_idx = 0; t < ಒಟ್ಟು_tiles && vtcm_idx < 2; t++, vtcm_idx++) {
        ಸ್ಥಿರಾಂಕ uint32_t ಅಡ್ಡಸಾಲು  = src0_ಆರಂಭ_ಅಡ್ಡಸಾಲು + t / tiles_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು;
        ಸ್ಥಿರಾಂಕ uint32_t ಲಂಬಸಾಲು  = (t % tiles_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು) * ಲಂಬಸಾಲು_tile;
        ಸ್ಥಿರಾಂಕ uint32_t tw   = MIN(ಲಂಬಸಾಲು_tile, ne0 - ಲಂಬಸಾಲು);
        ಸ್ಥಿರಾಂಕ size_t   tb   = (size_t) tw * ಗಾತ್ರ(ದಶಮಾಂಶ);
        ಸ್ಥಿರಾಂಕ size_t   soff = (src0_contig ? (ಅಡ್ಡಸಾಲು * nb01) :
                               unary_ಅಡ್ಡಸಾಲು_ಸ್ಥಾನಾಂತರ(ಅಡ್ಡಸಾಲು, ne01, ne02, div_ne01, div_ne02, div_ne012, nb01, nb02, nb03)) +
                               (size_t) ಲಂಬಸಾಲು * ಗಾತ್ರ(ದಶಮಾಂಶ);

        dma_ಸರತಿ_ತಳ್ಳು(dma_q, dma_make_ದತ್ತಾಂಶ(ದತ್ತಾಂಶ_dst, dst_vtcm_ದತ್ತಾಂಶ + (vtcm_idx * dst_half)), 0, 0, 0, 0);
        dma_ಸರತಿ_ತಳ್ಳು(dma_q, dma_make_ದತ್ತಾಂಶ(src0_vtcm_ದತ್ತಾಂಶ + (vtcm_idx * src0_half), ದತ್ತಾಂಶ_src + soff), tb, tb, tb, 1);
    }

    unary_tiled_tri_compute_fn_t compute = (unary_tiled_tri_compute_fn_t) uctx->compute;

    uint32_t ಅಡ್ಡಸಾಲು = src0_ಆರಂಭ_ಅಡ್ಡಸಾಲು;
    uint32_t ಲಂಬಸಾಲು = 0;
    uint32_t tile_ಒಳ_ಅಡ್ಡಸಾಲು = 0;
    uint32_t i01 = fastmodulo(ಅಡ್ಡಸಾಲು, ne01, div_ne01);

    uint32_t prow = src0_ಆರಂಭ_ಅಡ್ಡಸಾಲು + fastdiv(2, div_tpr);
    uint32_t pcol = fastmodulo(2, tiles_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು, div_tpr) * ಲಂಬಸಾಲು_tile;
    uint32_t ptile_ಒಳ_ಅಡ್ಡಸಾಲು = fastmodulo(2, tiles_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು, div_tpr);

    ಗಾಗಿ (uint32_t t = 0; t < ಒಟ್ಟು_tiles; t++) {
        ಶೂನ್ಯ * dst_vtcm = (ಶೂನ್ಯ *) (uintptr_t) dma_ಸರತಿ_ತೆಗೆ(dma_q).src;
        ಶೂನ್ಯ * src_vtcm = (ಶೂನ್ಯ *) (uintptr_t) dma_ಸರತಿ_ತೆಗೆ(dma_q).dst;

        ಸ್ಥಿರಾಂಕ uint32_t tw = MIN(ಲಂಬಸಾಲು_tile, ne0 - ಲಂಬಸಾಲು);

        htp_trace_event_ಆರಂಭ(tr, HTP_TRACE_EVT_HVX_COMP, t);
        compute(src_vtcm, dst_vtcm, tw, ಲಂಬಸಾಲು, i01, ne0, tri_ttype);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, t);

        ಸ್ಥಿರಾಂಕ size_t doff = (dst_contig ? (ಅಡ್ಡಸಾಲು * nb1) :
                             unary_ಅಡ್ಡಸಾಲು_ಸ್ಥಾನಾಂತರ(ಅಡ್ಡಸಾಲು, ne1, ne2, div_ne01, div_ne02, div_ne012, nb1, nb2, nb3)) +
                             (size_t) ಲಂಬಸಾಲು * ಗಾತ್ರ(ದಶಮಾಂಶ);
        ಸ್ಥಿರಾಂಕ size_t tb   = (size_t) tw * ಗಾತ್ರ(ದಶಮಾಂಶ);
        dma_ಸರತಿ_ತಳ್ಳು(dma_q, dma_make_ದತ್ತಾಂಶ(ದತ್ತಾಂಶ_dst + doff, dst_vtcm), tb, tb, tb, 1);

        ಸ್ಥಿರಾಂಕ uint32_t pt = t + 2;
        ಆದರೆ (pt < ಒಟ್ಟು_tiles) {
            ಸ್ಥಿರಾಂಕ uint32_t ptw  = MIN(ಲಂಬಸಾಲು_tile, ne0 - pcol);
            ಸ್ಥಿರಾಂಕ size_t   ptb  = (size_t) ptw * ಗಾತ್ರ(ದಶಮಾಂಶ);
            ಸ್ಥಿರಾಂಕ size_t   psoff = (src0_contig ? (prow * nb01) :
                                    unary_ಅಡ್ಡಸಾಲು_ಸ್ಥಾನಾಂತರ(prow, ne01, ne02, div_ne01, div_ne02, div_ne012, nb01, nb02,
                                                     nb03)) +
                                   (size_t) pcol * ಗಾತ್ರ(ದಶಮಾಂಶ);
            dma_ಸರತಿ_ತಳ್ಳು(dma_q, dma_make_ದತ್ತಾಂಶ(src_vtcm, ದತ್ತಾಂಶ_src + psoff), ptb, ptb, ptb, 1);
        }

        tile_ಒಳ_ಅಡ್ಡಸಾಲು++;
        ಲಂಬಸಾಲು += ಲಂಬಸಾಲು_tile;
        ಆದರೆ (tile_ಒಳ_ಅಡ್ಡಸಾಲು == tiles_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು) {
            tile_ಒಳ_ಅಡ್ಡಸಾಲು = 0;
            ಲಂಬಸಾಲು = 0;
            ಅಡ್ಡಸಾಲು++;
            i01++;
            ಆದರೆ (i01 == ne01) {
                i01 = 0;
            }
        }

        ptile_ಒಳ_ಅಡ್ಡಸಾಲು++;
        pcol += ಲಂಬಸಾಲು_tile;
        ಆದರೆ (ptile_ಒಳ_ಅಡ್ಡಸಾಲು == tiles_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು) {
            ptile_ಒಳ_ಅಡ್ಡಸಾಲು = 0;
            pcol = 0;
            prow++;
        }
    }

    dma_ಸರತಿ_flush(dma_q);
}

ಸ್ಥಿರ ಪೂರ್ಣಾಂಕ execute_op_unary(ರಚನೆ htp_ops_context * octx) {
    ಪೂರ್ಣಾಂಕ err = HTP_ಸ್ಥಿತಿ_OK;

    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * src0 = octx->src[0];
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * dst  = octx->dst;

    ಸ್ಥಿರಾಂಕ bool is_f16 = (src0->type == HTP_TYPE_F16);

    ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * op_type = ಖಾಲಿ;

    ಆಯ್ಕೆ (octx->op) {
        ಸ್ಥಿತಿ HTP_OP_NORM:            op_type = is_f16 ? "norm-f16"     : "norm-f32";         ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ HTP_OP_RMS_NORM:        op_type = is_f16 ? "rmsnorm-f16"  : "rmsnorm-f32";      ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ HTP_OP_RMS_NORM_MUL:    op_type = "rmsnorm-mul-f32";                            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ HTP_OP_SCALE:           op_type = is_f16 ? "scale-f16"    : "scale-f32";        ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ HTP_OP_CLAMP:           op_type = is_f16 ? "clamp-f16"    : "clamp-f32";        ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ HTP_OP_LEAKY_RELU:      op_type = "leaky-relu-f32";                             ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ HTP_OP_SQR:             op_type = is_f16 ? "sqr-f16"      : "sqr-f32";          ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ HTP_OP_SQRT:            op_type = is_f16 ? "sqrt-f16"     : "sqrt-f32";         ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ HTP_OP_UNARY_NEG:       op_type = "neg-f32";                                    ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ HTP_OP_UNARY_EXP:       op_type = "exp-f32";                                    ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ HTP_OP_UNARY_SIGMOID:   op_type = "sigmoid-f32";                                ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ HTP_OP_UNARY_SILU:      op_type = "silu-f32";                                   ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ HTP_OP_UNARY_GELU:      op_type = "gelu-f32";                                   ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ HTP_OP_UNARY_GELU_ಶೀಘ್ರ: op_type = "gelu-quick-f32";                             ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ HTP_OP_UNARY_GELU_ERF:  op_type = "gelu-erf-f32";                               ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ HTP_OP_UNARY_SOFTPLUS:  op_type = "softplus-f32";                               ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ HTP_OP_UNARY_TANH:      op_type = "tanh-f32";                                   ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ HTP_OP_UNARY_ABS:       op_type = is_f16 ? "abs-f16"      : "abs-f32";          ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ HTP_OP_UNARY_LOG:       op_type = is_f16 ? "log-f16"      : "log-f32";          ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ HTP_OP_UNARY_RELU:      op_type = "relu-f32";                                   ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ HTP_OP_UNARY_ಹೆಜ್ಜೆ:      op_type = is_f16 ? "step-f16"     : "step-f32";         ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ HTP_OP_L2_NORM:         op_type = is_f16 ? "l2norm-f16"   : "l2norm-f32";       ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ HTP_OP_TRI:             op_type = "tri-f32";                                    ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ HTP_OP_HADAMARD:        op_type = is_f16 ? "hadamard-f16" : "hadamard-f32";     ನಿಲ್ಲಿಸು;
        ಪೂರ್ವನಿಯೋಜಿತ:
            FARF(ERROR, "Unsupported unary Op %u\n", octx->op);
            ಹಿಂತಿರುಗಿ HTP_ಸ್ಥಿತಿ_NO_SUPPORT;
    }

    // F16 only has row-block kernels for this subset of ops (see the dispatch switch
    // below) - reject everything else up front, before touching kparams/VTCM.
    ಆದರೆ (is_f16) {
        ಆಯ್ಕೆ (octx->op) {
            ಸ್ಥಿತಿ HTP_OP_NORM:
            ಸ್ಥಿತಿ HTP_OP_RMS_NORM:
            ಸ್ಥಿತಿ HTP_OP_SCALE:
            ಸ್ಥಿತಿ HTP_OP_CLAMP:
            ಸ್ಥಿತಿ HTP_OP_SQR:
            ಸ್ಥಿತಿ HTP_OP_SQRT:
            ಸ್ಥಿತಿ HTP_OP_L2_NORM:
            ಸ್ಥಿತಿ HTP_OP_UNARY_ABS:
            ಸ್ಥಿತಿ HTP_OP_UNARY_LOG:
            ಸ್ಥಿತಿ HTP_OP_UNARY_ಹೆಜ್ಜೆ:
            ಸ್ಥಿತಿ HTP_OP_HADAMARD:
                ನಿಲ್ಲಿಸು;
            ಪೂರ್ವನಿಯೋಜಿತ:
                FARF(ERROR, "unary-%s: not supported for F16\n", op_type);
                ಹಿಂತಿರುಗಿ HTP_ಸ್ಥಿತಿ_NO_SUPPORT;
        }
    }

    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_unary_kernel_params * kparams = (ಸ್ಥಿರಾಂಕ ರಚನೆ htp_unary_kernel_params *) octx->kernel_params;

    ಆದರೆ (!htp_ops_context_ಗಣ_n_ಎಳೆಗಳು(octx, kparams->n_ಎಳೆಗಳು)) {
        ಹಿಂತಿರುಗಿ HTP_ಸ್ಥಿತಿ_INVAL_PARAMS;
    }

    ಸ್ಥಿರಾಂಕ uint32_t src0_nrows = src0->ne[1] * src0->ne[2] * src0->ne[3];
    ಸ್ಥಿರಾಂಕ size_t src0_elem_ಗಾತ್ರ = is_f16 ? ಗಾತ್ರ(_Float16) : ಗಾತ್ರ(ದಶಮಾಂಶ);
    ಸ್ಥಿರಾಂಕ size_t dst_elem_ಗಾತ್ರ  = (dst->type == HTP_TYPE_F16) ? ಗಾತ್ರ(_Float16) : ಗಾತ್ರ(ದಶಮಾಂಶ);
    ಸ್ಥಿರಾಂಕ size_t src0_ದತ್ತಾಂಶ_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ = src0->ne[0] * src0_elem_ಗಾತ್ರ;
    ಸ್ಥಿರಾಂಕ size_t dst_ದತ್ತಾಂಶ_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ  = dst->ne[0]  * dst_elem_ಗಾತ್ರ;

    uint32_t ಅಡ್ಡಸಾಲು_ಆರಂಭ = 0;
    uint32_t nrows     = src0_nrows;

    ಆದರೆ (octx->ctx->mdev.ಎಣಿಕೆ_2 > 1) {
        uint32_t ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_chunk = 0;
        htp_tensor_mdev_ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_chunk(dst, (uint32_t) dst_elem_ಗಾತ್ರ, (uint32_t) dst_ದತ್ತಾಂಶ_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ, &ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_chunk);
        ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor_mdev_range range = htp_tensor_mdev_ವಿಭಜನೆ(src0_nrows, ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_chunk, octx->ctx->mdev.idx, octx->ctx->mdev.ಎಣಿಕೆ_2, &octx->ctx->mdev.ಎಣಿಕೆ_div);
        ಅಡ್ಡಸಾಲು_ಆರಂಭ = range.ಆರಂಭ;
        nrows     = range.ಎಣಿಕೆ_2;
    }

    ಆದರೆ (nrows == 0) {
        ಹಿಂತಿರುಗಿ HTP_ಸ್ಥಿತಿ_OK;
    }

    ಸ್ಥಿರಾಂಕ uint32_t n_ಎಳೆಗಳು = octx->n_ಎಳೆಗಳು;

    ಸ್ಥಿರಾಂಕ size_t src0_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned = kparams->src0_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned;
    ಸ್ಥಿರಾಂಕ size_t dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned  = kparams->dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned;

    // Always 0 for F16 - htp_unary_vtcm_layout_build() keeps F16 on the row-block path,
    // since only F32 has unary_task_f32_tiled_* kernels.
    ಸ್ಥಿರಾಂಕ uint32_t ಲಂಬಸಾಲು_tile = kparams->ಲಂಬಸಾಲು_tile;

    size_t src1_ದತ್ತಾಂಶ_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ = 0;
    size_t src1_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned = kparams->src1_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned;
    bool ಪ್ರಸಾರ_ತೂಕ = kparams->ಪ್ರಸಾರ_ತೂಕ;
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * src1 = ಖಾಲಿ;

    // RMS_NORM_MUL fusion is F32-only (its weight tensor is always F32; see
    // try_fuse_node()'s type guard), so this never triggers when is_f16 is true.
    ಆದರೆ (octx->op == HTP_OP_RMS_NORM_MUL) {
        src1 = octx->src[1];
        src1_ದತ್ತಾಂಶ_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ = src1->ne[0] * ಗಾತ್ರ(ದಶಮಾಂಶ);
    }

    ಆದರೆ (octx->ctx->vtcm_ಗಾತ್ರ < (size_t)kparams->vtcm_ಗಾತ್ರ) {
        FARF(ERROR, "unary-%s : current VTCM reservation %zu is too small, needed %zu\n", op_type, octx->ctx->vtcm_ಗಾತ್ರ, (size_t)kparams->vtcm_ಗಾತ್ರ);
        ಹಿಂತಿರುಗಿ HTP_ಸ್ಥಿತಿ_VTCM_TOO_SMALL;
    }

    octx->src0_spad.src = ಖಾಲಿ;
    octx->src1_spad.src = ಖಾಲಿ;
    octx->dst_spad.src  = ಖಾಲಿ;

    FARF(ಮೇಲಿನ_2, "%s: (%ux%ux%ux%u) -> (%ux%ux%ux%u) : src0-vtcm-size %u src1-vtcm-size %u dst-vtcm-size %u\n", op_type,
         src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3], dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3],
         kparams->vtcm_src0_ಗಾತ್ರ, kparams->vtcm_src1_ಗಾತ್ರ, kparams->vtcm_dst_ಗಾತ್ರ);

    uint8_t * ಸ್ಥಿರಾಂಕ ತಳ = (uint8_t *) octx->ctx->vtcm_ತಳ;
    ರಚನೆ htp_unary_context uctx = {
        .octx                  = octx,
        .kparams               = kparams,
        .src0_nrows_ಪ್ರತಿ_thread = fastdiv(nrows + n_ಎಳೆಗಳು - 1, &octx->n_ಎಳೆಗಳು_div),
        .src0_nrows            = nrows,
        .ಅಡ್ಡಸಾಲು_ಆರಂಭ             = ಅಡ್ಡಸಾಲು_ಆರಂಭ,

        .ದತ್ತಾಂಶ_src0             = src0->ದತ್ತಾಂಶ,
        .ದತ್ತಾಂಶ_src1             = (octx->op == HTP_OP_RMS_NORM_MUL) ? src1->ದತ್ತಾಂಶ : 0,
        .ದತ್ತಾಂಶ_dst              = dst->ದತ್ತಾಂಶ,

        .src0_ದತ್ತಾಂಶ_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ    = src0_ದತ್ತಾಂಶ_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ,
        .src1_ದತ್ತಾಂಶ_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ    = src1_ದತ್ತಾಂಶ_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ,
        .dst_ದತ್ತಾಂಶ_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ     = dst_ದತ್ತಾಂಶ_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ,

        .src0_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned = src0_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned,
        .src1_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned = src1_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned,
        .dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned  = dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned,

        .src0_vtcm_half_ಗಾತ್ರ   = kparams->vtcm_src0_ಗಾತ್ರ_ಪ್ರತಿ_thread / 2,
        .src1_vtcm_half_ಗಾತ್ರ   = (octx->op == HTP_OP_RMS_NORM_MUL) ? (kparams->vtcm_src1_ಗಾತ್ರ_ಪ್ರತಿ_thread / (ಪ್ರಸಾರ_ತೂಕ ? 1 : 2)) : 0,
        .dst_vtcm_half_ಗಾತ್ರ    = kparams->vtcm_dst_ಗಾತ್ರ_ಪ್ರತಿ_thread / 2,

        .ಖಂಡ                 = kparams->ಖಂಡ,
        .nc                    = src0->ne[0],
        .ಲಂಬಸಾಲು_tile              = ಲಂಬಸಾಲು_tile,
        .ಪ್ರಸಾರ_ತೂಕ      = ಪ್ರಸಾರ_ತೂಕ,

        .vtcm_src0             = VTCM_LAYOUT_PTR(uint8_t, ತಳ, 0),
        .vtcm_src1             = VTCM_LAYOUT_PTR_OPTIONAL(uint8_t, ತಳ, kparams->vtcm_src0_ಗಾತ್ರ, kparams->vtcm_src1_ಗಾತ್ರ > 0),
        .vtcm_dst              = VTCM_LAYOUT_PTR(uint8_t, ತಳ, kparams->vtcm_src0_ಗಾತ್ರ + kparams->vtcm_src1_ಗಾತ್ರ),

        .vtcm_src0_ಗಾತ್ರ_ಪ್ರತಿ_thread = kparams->vtcm_src0_ಗಾತ್ರ_ಪ್ರತಿ_thread,
        .vtcm_src1_ಗಾತ್ರ_ಪ್ರತಿ_thread = kparams->vtcm_src1_ಗಾತ್ರ_ಪ್ರತಿ_thread,
        .vtcm_dst_ಗಾತ್ರ_ಪ್ರತಿ_thread  = kparams->vtcm_dst_ಗಾತ್ರ_ಪ್ರತಿ_thread,
    };

    FARF(ಮೇಲಿನ_2, "%s: %s mode (col_tile %u)\n", op_type, ಲಂಬಸಾಲು_tile ? "tiled" : "row-block", ಲಂಬಸಾಲು_tile);

    ಕೆಲಸಗಾರ_callback_t task_func = ಖಾಲಿ;
    ಶೂನ್ಯ * compute_func = ಖಾಲಿ;

    ಆದರೆ (ಲಂಬಸಾಲು_tile) {
        task_func = unary_thread_tiled;
        ಆಯ್ಕೆ (octx->op) {
            ಸ್ಥಿತಿ HTP_OP_SCALE:           compute_func = (ಶೂನ್ಯ *) tile_scale_f32;          ನಿಲ್ಲಿಸು;
            ಸ್ಥಿತಿ HTP_OP_CLAMP:           compute_func = (ಶೂನ್ಯ *) tile_clamp_f32;          ನಿಲ್ಲಿಸು;
            ಸ್ಥಿತಿ HTP_OP_LEAKY_RELU:      compute_func = (ಶೂನ್ಯ *) tile_leaky_relu_f32;     ನಿಲ್ಲಿಸು;
            ಸ್ಥಿತಿ HTP_OP_SQR:             compute_func = (ಶೂನ್ಯ *) tile_sqr_f32;            ನಿಲ್ಲಿಸು;
            ಸ್ಥಿತಿ HTP_OP_SQRT:            compute_func = (ಶೂನ್ಯ *) tile_sqrt_f32;           ನಿಲ್ಲಿಸು;
            ಸ್ಥಿತಿ HTP_OP_UNARY_NEG:       compute_func = (ಶೂನ್ಯ *) tile_neg_f32;            ನಿಲ್ಲಿಸು;
            ಸ್ಥಿತಿ HTP_OP_UNARY_EXP:       compute_func = (ಶೂನ್ಯ *) tile_exp_f32;            ನಿಲ್ಲಿಸು;
            ಸ್ಥಿತಿ HTP_OP_UNARY_SIGMOID:   compute_func = (ಶೂನ್ಯ *) tile_sigmoid_f32;        ನಿಲ್ಲಿಸು;
            ಸ್ಥಿತಿ HTP_OP_UNARY_SILU:      compute_func = (ಶೂನ್ಯ *) tile_silu_f32;           ನಿಲ್ಲಿಸು;
            ಸ್ಥಿತಿ HTP_OP_UNARY_GELU:      compute_func = (ಶೂನ್ಯ *) tile_gelu_f32;           ನಿಲ್ಲಿಸು;
            ಸ್ಥಿತಿ HTP_OP_UNARY_GELU_ಶೀಘ್ರ: compute_func = (ಶೂನ್ಯ *) tile_gelu_ಶೀಘ್ರ_f32;     ನಿಲ್ಲಿಸು;
            ಸ್ಥಿತಿ HTP_OP_UNARY_GELU_ERF:  compute_func = (ಶೂನ್ಯ *) tile_gelu_erf_f32;       ನಿಲ್ಲಿಸು;
            ಸ್ಥಿತಿ HTP_OP_UNARY_SOFTPLUS:  compute_func = (ಶೂನ್ಯ *) tile_softplus_f32;       ನಿಲ್ಲಿಸು;
            ಸ್ಥಿತಿ HTP_OP_UNARY_TANH:      compute_func = (ಶೂನ್ಯ *) tile_tanh_f32;           ನಿಲ್ಲಿಸು;
            ಸ್ಥಿತಿ HTP_OP_UNARY_ABS:       compute_func = (ಶೂನ್ಯ *) tile_abs_f32;            ನಿಲ್ಲಿಸು;
            ಸ್ಥಿತಿ HTP_OP_UNARY_LOG:       compute_func = (ಶೂನ್ಯ *) tile_log_f32;            ನಿಲ್ಲಿಸು;
            ಸ್ಥಿತಿ HTP_OP_UNARY_RELU:      compute_func = (ಶೂನ್ಯ *) tile_relu_f32;           ನಿಲ್ಲಿಸು;
            ಸ್ಥಿತಿ HTP_OP_UNARY_ಹೆಜ್ಜೆ:      compute_func = (ಶೂನ್ಯ *) tile_ಹೆಜ್ಜೆ_f32;           ನಿಲ್ಲಿಸು;
            ಸ್ಥಿತಿ HTP_OP_TRI:
                task_func    = unary_thread_tiled_tri_f32;
                compute_func = (ಶೂನ್ಯ *) tri_apply_tile_f32;
                ನಿಲ್ಲಿಸು;
            ಪೂರ್ವನಿಯೋಜಿತ:                     ನಿಲ್ಲಿಸು;
        }
    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (is_f16) {
        task_func = unary_thread_ಅಡ್ಡಸಾಲು_ಖಂಡ;
        ಆಯ್ಕೆ (octx->op) {
            ಸ್ಥಿತಿ HTP_OP_NORM:            compute_func = (ಶೂನ್ಯ *) norm_f16;                ನಿಲ್ಲಿಸು;
            ಸ್ಥಿತಿ HTP_OP_RMS_NORM:        compute_func = (ಶೂನ್ಯ *) rms_norm_f16;            ನಿಲ್ಲಿಸು;
            ಸ್ಥಿತಿ HTP_OP_SCALE:           compute_func = (ಶೂನ್ಯ *) scale_f16;               ನಿಲ್ಲಿಸು;
            ಸ್ಥಿತಿ HTP_OP_CLAMP:           compute_func = (ಶೂನ್ಯ *) clamp_f16;               ನಿಲ್ಲಿಸು;
            ಸ್ಥಿತಿ HTP_OP_SQR:             compute_func = (ಶೂನ್ಯ *) sqr_f16;                 ನಿಲ್ಲಿಸು;
            ಸ್ಥಿತಿ HTP_OP_SQRT:            compute_func = (ಶೂನ್ಯ *) sqrt_f16;                ನಿಲ್ಲಿಸು;
            ಸ್ಥಿತಿ HTP_OP_L2_NORM:         compute_func = (ಶೂನ್ಯ *) l2_norm_f16;             ನಿಲ್ಲಿಸು;
            ಸ್ಥಿತಿ HTP_OP_UNARY_ABS:       compute_func = (ಶೂನ್ಯ *) abs_f16;                 ನಿಲ್ಲಿಸು;
            ಸ್ಥಿತಿ HTP_OP_UNARY_LOG:       compute_func = (ಶೂನ್ಯ *) log_f16;                 ನಿಲ್ಲಿಸು;
            ಸ್ಥಿತಿ HTP_OP_UNARY_ಹೆಜ್ಜೆ:      compute_func = (ಶೂನ್ಯ *) ಹೆಜ್ಜೆ_f16;                ನಿಲ್ಲಿಸು;
            ಸ್ಥಿತಿ HTP_OP_HADAMARD:        compute_func = (ಶೂನ್ಯ *) hadamard_f16;            ನಿಲ್ಲಿಸು;
            ಪೂರ್ವನಿಯೋಜಿತ:                     ನಿಲ್ಲಿಸು;
        }
    } ಇಲ್ಲದಿದ್ದರೆ {
        task_func = unary_thread_ಅಡ್ಡಸಾಲು_ಖಂಡ;
        ಆಯ್ಕೆ (octx->op) {
            ಸ್ಥಿತಿ HTP_OP_NORM:            compute_func = (ಶೂನ್ಯ *) norm_f32;                ನಿಲ್ಲಿಸು;
            ಸ್ಥಿತಿ HTP_OP_RMS_NORM:        compute_func = (ಶೂನ್ಯ *) rms_norm_f32;            ನಿಲ್ಲಿಸು;
            ಸ್ಥಿತಿ HTP_OP_RMS_NORM_MUL:
                task_func    = unary_thread_rms_norm_mul_f32;
                compute_func = (ಶೂನ್ಯ *) rms_norm_mul_f32;
                ನಿಲ್ಲಿಸು;
            ಸ್ಥಿತಿ HTP_OP_SCALE:           compute_func = (ಶೂನ್ಯ *) scale_f32;               ನಿಲ್ಲಿಸು;
            ಸ್ಥಿತಿ HTP_OP_CLAMP:           compute_func = (ಶೂನ್ಯ *) clamp_f32;               ನಿಲ್ಲಿಸು;
            ಸ್ಥಿತಿ HTP_OP_LEAKY_RELU:      compute_func = (ಶೂನ್ಯ *) leaky_relu_f32;          ನಿಲ್ಲಿಸು;
            ಸ್ಥಿತಿ HTP_OP_SQR:             compute_func = (ಶೂನ್ಯ *) sqr_f32;                 ನಿಲ್ಲಿಸು;
            ಸ್ಥಿತಿ HTP_OP_SQRT:            compute_func = (ಶೂನ್ಯ *) sqrt_f32;                ನಿಲ್ಲಿಸು;
            ಸ್ಥಿತಿ HTP_OP_UNARY_NEG:       compute_func = (ಶೂನ್ಯ *) neg_f32;                 ನಿಲ್ಲಿಸು;
            ಸ್ಥಿತಿ HTP_OP_UNARY_EXP:       compute_func = (ಶೂನ್ಯ *) exp_f32;                 ನಿಲ್ಲಿಸು;
            ಸ್ಥಿತಿ HTP_OP_UNARY_SIGMOID:   compute_func = (ಶೂನ್ಯ *) sigmoid_f32;             ನಿಲ್ಲಿಸು;
            ಸ್ಥಿತಿ HTP_OP_UNARY_SILU:      compute_func = (ಶೂನ್ಯ *) silu_f32;                ನಿಲ್ಲಿಸು;
            ಸ್ಥಿತಿ HTP_OP_UNARY_GELU:      compute_func = (ಶೂನ್ಯ *) gelu_f32;                ನಿಲ್ಲಿಸು;
            ಸ್ಥಿತಿ HTP_OP_UNARY_GELU_ಶೀಘ್ರ: compute_func = (ಶೂನ್ಯ *) gelu_ಶೀಘ್ರ_f32;          ನಿಲ್ಲಿಸು;
            ಸ್ಥಿತಿ HTP_OP_UNARY_GELU_ERF:  compute_func = (ಶೂನ್ಯ *) gelu_erf_f32;            ನಿಲ್ಲಿಸು;
            ಸ್ಥಿತಿ HTP_OP_UNARY_SOFTPLUS:  compute_func = (ಶೂನ್ಯ *) softplus_f32;            ನಿಲ್ಲಿಸು;
            ಸ್ಥಿತಿ HTP_OP_UNARY_TANH:      compute_func = (ಶೂನ್ಯ *) tanh_f32;                ನಿಲ್ಲಿಸು;
            ಸ್ಥಿತಿ HTP_OP_UNARY_ABS:       compute_func = (ಶೂನ್ಯ *) abs_f32;                 ನಿಲ್ಲಿಸು;
            ಸ್ಥಿತಿ HTP_OP_UNARY_LOG:       compute_func = (ಶೂನ್ಯ *) log_f32;                 ನಿಲ್ಲಿಸು;
            ಸ್ಥಿತಿ HTP_OP_UNARY_RELU:      compute_func = (ಶೂನ್ಯ *) relu_f32;                ನಿಲ್ಲಿಸು;
            ಸ್ಥಿತಿ HTP_OP_UNARY_ಹೆಜ್ಜೆ:      compute_func = (ಶೂನ್ಯ *) ಹೆಜ್ಜೆ_f32;                ನಿಲ್ಲಿಸು;
            ಸ್ಥಿತಿ HTP_OP_L2_NORM:         compute_func = (ಶೂನ್ಯ *) l2_norm_f32;             ನಿಲ್ಲಿಸು;
            ಸ್ಥಿತಿ HTP_OP_HADAMARD:        compute_func = (ಶೂನ್ಯ *) hadamard_f32;            ನಿಲ್ಲಿಸು;
            ಸ್ಥಿತಿ HTP_OP_TRI:
                task_func    = unary_thread_tri_f32;
                compute_func = (ಶೂನ್ಯ *) tri_f32;
                ನಿಲ್ಲಿಸು;
            ಪೂರ್ವನಿಯೋಜಿತ:                     ನಿಲ್ಲಿಸು;
        }
    }

    ಆದರೆ (!task_func || !compute_func) {
        FARF(ERROR, "execute_op_unary: task function is NULL for op %d\n", octx->op);
        ಹಿಂತಿರುಗಿ HTP_ಸ್ಥಿತಿ_NO_SUPPORT;
    }

    uctx.compute = compute_func;
    work_ಸರತಿ_ಸತತ(octx->ctx->work_ಸರತಿ, task_func, &uctx, n_ಎಳೆಗಳು);

    ಹಿಂತಿರುಗಿ err;
}

ಪೂರ್ಣಾಂಕ op_unary(ರಚನೆ htp_ops_context * octx) {
    ಆಯ್ಕೆ (octx->src[0]->type) {
        ಸ್ಥಿತಿ HTP_TYPE_F32:
        ಸ್ಥಿತಿ HTP_TYPE_F16:
            ಹಿಂತಿರುಗಿ execute_op_unary(octx);

        ಪೂರ್ವನಿಯೋಜಿತ:
            ಹಿಂತಿರುಗಿ HTP_ಸ್ಥಿತಿ_NO_SUPPORT;
    }
}
