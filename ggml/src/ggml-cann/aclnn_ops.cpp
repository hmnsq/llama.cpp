/*
 * Copyright (c) 2023-2026 The ggml authors
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

#ಅನ್ತರ್ಭಾವ "aclnn_ops.h"

#ಅನ್ತರ್ಭಾವ "ggml-impl.h"
#ಅನ್ತರ್ಭಾವ "ggml.h"


#ಅನ್ತರ್ಭಾವ <aclnnop/aclnn_add.h>
#ಅನ್ತರ್ಭಾವ <aclnnop/aclnn_add_rms_norm.h>
#ಅನ್ತರ್ಭಾವ <aclnnop/aclnn_addcdiv.h>
#ಅನ್ತರ್ಭಾವ <aclnnop/aclnn_argmax.h>
#ಅನ್ತರ್ಭಾವ <aclnnop/aclnn_avgpool2d.h>
#ಅನ್ತರ್ಭಾವ <aclnnop/aclnn_batch_matmul.h>
#ಅನ್ತರ್ಭಾವ <aclnnop/aclnn_cast.h>
#ಅನ್ತರ್ಭಾವ <aclnnop/aclnn_clamp.h>
#ಅನ್ತರ್ಭಾವ <aclnnop/aclnn_constant_pad_nd.h>
#ಅನ್ತರ್ಭಾವ <aclnnop/aclnn_convolution.h>
#ಅನ್ತರ್ಭಾವ <aclnnop/aclnn_copy.h>
#ಅನ್ತರ್ಭಾವ <aclnnop/aclnn_div.h>
#ಅನ್ತರ್ಭಾವ <aclnnop/aclnn_elu.h>
#ಅನ್ತರ್ಭಾವ <aclnnop/aclnn_embedding.h>
#ಅನ್ತರ್ಭಾವ <aclnnop/aclnn_eq_tensor.h>
#ಅನ್ತರ್ಭಾವ <aclnnop/aclnn_exp.h>
#ಅನ್ತರ್ಭಾವ <aclnnop/aclnn_fill_scalar.h>
#ಅನ್ತರ್ಭಾವ <aclnnop/aclnn_fused_infer_attention_score_v2.h>
#ಅನ್ತರ್ಭಾವ <aclnnop/aclnn_ger.h>
#ಅನ್ತರ್ಭಾವ <aclnnop/aclnn_group_norm.h>
#ಅನ್ತರ್ಭಾವ <aclnnop/aclnn_gather_v2.h>
#ಅನ್ತರ್ಭಾವ <aclnnop/aclnn_grouped_matmul_v3.h>
#ಅನ್ತರ್ಭಾವ <aclnnop/aclnn_scatter.h>
#ಅನ್ತರ್ಭಾವ <aclnnop/aclnn_gt_scalar.h>
#ಅನ್ತರ್ಭಾವ <aclnnop/aclnn_im2col.h>
#ಅನ್ತರ್ಭಾವ <aclnnop/aclnn_index_copy.h>
#ಅನ್ತರ್ಭಾವ <aclnnop/aclnn_index_fill_tensor.h>
#ಅನ್ತರ್ಭಾವ <aclnnop/aclnn_index_select.h>
#ಅನ್ತರ್ಭಾವ <aclnnop/aclnn_layer_norm.h>
#ಅನ್ತರ್ಭಾವ <aclnnop/aclnn_log.h>
#ಅನ್ತರ್ಭಾವ <aclnnop/aclnn_matmul.h>
#ಅನ್ತರ್ಭಾವ <aclnnop/aclnn_max_pool.h>
#ಅನ್ತರ್ಭಾವ <aclnnop/aclnn_mean.h>
#ಅನ್ತರ್ಭಾವ <aclnnop/aclnn_mm.h>
#ಅನ್ತರ್ಭಾವ <aclnnop/aclnn_mul.h>
#ಅನ್ತರ್ಭಾವ <aclnnop/aclnn_mv.h>
#ಅನ್ತರ್ಭಾವ <aclnnop/aclnn_permute.h>
#ಅನ್ತರ್ಭಾವ <aclnnop/aclnn_pow.h>
#ಅನ್ತರ್ಭಾವ <aclnnop/aclnn_pow_tensor_tensor.h>
#ಅನ್ತರ್ಭಾವ <aclnnop/aclnn_recurrent_gated_delta_rule.h>
#ಅನ್ತರ್ಭಾವ <aclnnop/aclnn_reduce_sum.h>
#ಅನ್ತರ್ಭಾವ <aclnnop/aclnn_reflection_pad1d.h>
#ಅನ್ತರ್ಭಾವ <aclnnop/aclnn_repeat.h>
#ಅನ್ತರ್ಭಾವ <aclnnop/aclnn_repeat_interleave.h>
#ಅನ್ತರ್ಭಾವ <aclnnop/aclnn_rms_norm.h>
#ಅನ್ತರ್ಭಾವ <aclnnop/aclnn_roll.h>
#ಅನ್ತರ್ಭಾವ <aclnnop/aclnn_softmax.h>
#ಅನ್ತರ್ಭಾವ <aclnnop/aclnn_softmax_cross_entropy_with_logits.h>
#ಅನ್ತರ್ಭಾವ <aclnnop/aclnn_sub.h>
#ಅನ್ತರ್ಭಾವ <aclnnop/aclnn_sum.h>
#ಅನ್ತರ್ಭಾವ <aclnnop/aclnn_threshold.h>
#ಅನ್ತರ್ಭಾವ <aclnnop/aclnn_tril.h>
#ಅನ್ತರ್ಭಾವ <aclnnop/aclnn_triangular_solve.h>
#ಅನ್ತರ್ಭಾವ <aclnnop/aclnn_triu.h>
#ಅನ್ತರ್ಭಾವ <aclnnop/aclnn_logical_not.h>
#ಅನ್ತರ್ಭಾವ <aclnnop/aclnn_masked_fill_scalar.h>
#ಅನ್ತರ್ಭಾವ <aclnnop/aclnn_upsample_nearest_2d.h>
#ಅನ್ತರ್ಭಾವ <aclnnop/aclnn_weight_quant_batch_matmul_v2.h>
#ಅನ್ತರ್ಭಾವ <aclnnop/aclnn_zero.h>
#ಅನ್ತರ್ಭಾವ <float.h>

#ಅನ್ತರ್ಭಾವ <cmath>
#ಅನ್ತರ್ಭಾವ <cstring>
#ಅನ್ತರ್ಭಾವ <exception>
#ಅನ್ತರ್ಭಾವ <vector>

#ವ್ಯಾಖ್ಯಾನಿಸು GGML_COMMON_DECL_C

#ಅನ್ತರ್ಭಾವ "../ggml-common.h"

ಶೂನ್ಯ bcast_shape(ggml_tensor *    src0,
                 ggml_tensor *    src1,
                 ggml_tensor *    dst,
                 acl_tensor_ptr & acl_src0,
                 acl_tensor_ptr & acl_src1,
                 acl_tensor_ptr & acl_dst) {
    GGML_ASSERT(ggml_are_same_shape(src0, dst) && ggml_can_repeat(src1, src0));
    // Need bcast
    ಆದರೆ (!ggml_are_same_shape(src0, src1) && ggml_cann_need_bcast(src0, src1)) {
        BCAST_SHAPE(src0, src1)
        acl_src0 = ggml_cann_create_tensor(src0, BCAST_PARAM(src0));
        acl_src1 = ggml_cann_create_tensor(src1, BCAST_PARAM(src1));
        acl_dst  = ggml_cann_create_tensor(dst, BCAST_PARAM(src0));
    } ಇಲ್ಲದಿದ್ದರೆ {
        acl_src0 = ggml_cann_create_tensor(src0);
        acl_src1 = ggml_cann_create_tensor(src1);
        acl_dst  = ggml_cann_create_tensor(dst);
    }
}

ಶೂನ್ಯ ggml_cann_op_unary(std::function<ಶೂನ್ಯ(ggml_backend_cann_context &, aclTensor *, aclTensor *)> unary_op,
                        ggml_backend_cann_context &                                                ctx,
                        ggml_tensor *                                                              dst) {
    ggml_tensor * src = dst->src[0];

    acl_tensor_ptr acl_src = ggml_cann_create_tensor(src);
    acl_tensor_ptr acl_dst = ggml_cann_create_tensor(dst);

    unary_op(ctx, acl_src.get(), acl_dst.get());
}

ಶೂನ್ಯ ggml_cann_op_unary_gated(std::function<ಶೂನ್ಯ(ggml_backend_cann_context &, aclTensor *, aclTensor *)> unary_op,
                              ggml_backend_cann_context &                                                ctx,
                              ggml_tensor *                                                              dst) {
    ggml_tensor * src0 = dst->src[0];
    ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(ggml_is_contiguous_1(src0));
    GGML_ASSERT(ggml_is_contiguous_1(dst));
    ಸ್ಥಿರಾಂಕ int32_t swapped = ggml_get_op_params_i32(dst, 1);

    acl_tensor_ptr acl_dst = ggml_cann_create_tensor(dst);
    acl_tensor_ptr acl_src0, acl_src1;
    ಆದರೆ (src1) {
        GGML_ASSERT(ggml_is_contiguous_1(src1));
        GGML_ASSERT(src0->type == src1->type);

        acl_src0 = ggml_cann_create_tensor(src0);
        acl_src1 = ggml_cann_create_tensor(src1);
    } ಇಲ್ಲದಿದ್ದರೆ {
        int64_t ne[] = { src0->ne[0] / 2, src0->ne[1], src0->ne[2], src0->ne[3] };
        size_t  nb[] = { src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3] };
        acl_src0     = ggml_cann_create_tensor(src0, ne, nb, GGML_MAX_DIMS, ACL_FORMAT_ND, 0);
        acl_src1 = ggml_cann_create_tensor(src0, ne, nb, GGML_MAX_DIMS, ACL_FORMAT_ND, ne[0] * ggml_element_ಗಾತ್ರ(src0));
        ಆದರೆ (swapped) {
            std::swap(acl_src0, acl_src1);
        }
    }

    unary_op(ctx, acl_src0.get(), acl_dst.get());
    GGML_CANN_CALL_ACLNN_OP(ctx, InplaceMul, acl_dst.get(), acl_src1.get());
}

// Fused SwiGLU using aclnnSwiGlu: splits input along innermost dim, applies
// SiLU to left half, multiplies by right half.
//
// Falls back to the generic two-kernel path when src[1] != nullptr (two
// independent halves) or swapped != 0 (reversed activation order), as
// aclnnSwiGlu only handles the single interleaved tensor in standard order.
//
// CANN tiling for SwiGlu requires (storageShapeDim + viewDims) to be even.
// aclCreateTensor always uses storageShapeDim=1, so viewDims must be odd.
// We use a 3D view (1+3=4, even) to satisfy this constraint while preserving
// correct split semantics along the innermost (ne[0]) dimension.
ಶೂನ್ಯ ggml_cann_swiglu(ggml_backend_cann_context & ctx, ggml_tensor * dst) {
    ಸ್ವಯಂಚಾಲಿತ silu_fn = [](ggml_backend_cann_context & ctx, aclTensor * acl_src, aclTensor * acl_dst) {
        GGML_CANN_CALL_ACLNN_OP(ctx, Silu, acl_src, acl_dst);
    };

    ಸ್ಥಿರಾಂಕ int32_t swapped = ggml_get_op_params_i32(dst, 1);
    ಆದರೆ (dst->src[1] != ಶೂನ್ಯಸೂಚಕ || swapped != 0) {
        ggml_cann_op_unary_gated(silu_fn, ctx, dst);
        ಹಿಂತಿರುಗಿ;
    }

    // aclnnSwiGlu requires the split dim (src->ne[0]) to be even; fall back otherwise.
    ಆದರೆ (dst->src[0]->ne[0] % 2 != 0) {
        ggml_cann_op_unary_gated(silu_fn, ctx, dst);
        ಹಿಂತಿರುಗಿ;
    }

    ggml_tensor * src0 = dst->src[0];
    size_t elem_ಗಾತ್ರ = ggml_element_ಗಾತ್ರ(src0);

    // src0 GGML: [2*ne0, ne1, ne2, ne3] → 3D view [2*ne0, ne1, ne2*ne3]
    // CANN reversed: [ne2*ne3, ne1, 2*ne0], split along CANN dim 2 (last).
    int64_t ne0_x2   = src0->ne[0];
    int64_t ne1      = src0->ne[1];
    int64_t ne23     = src0->ne[2] * src0->ne[3];
    int64_t src3d_ne[] = { ne0_x2, ne1, ne23 };
    size_t  src3d_nb[] = { (size_t)src0->nb[0], (size_t)src0->nb[1], (size_t)src0->nb[2] };
    acl_tensor_ptr acl_src = ggml_cann_create_tensor(src0->ದತ್ತಾಂಶ, ggml_cann_type_mapping(src0->type),
                                                     elem_ಗಾತ್ರ, src3d_ne, src3d_nb, 3);

    // dst GGML: [ne0, ne1, ne2, ne3] → 3D view [ne0, ne1, ne2*ne3]
    int64_t ne0      = dst->ne[0];
    int64_t dst3d_ne[] = { ne0, ne1, ne23 };
    size_t  dst3d_nb[] = { (size_t)dst->nb[0], (size_t)dst->nb[1], (size_t)dst->nb[2] };
    acl_tensor_ptr acl_dst = ggml_cann_create_tensor(dst->ದತ್ತಾಂಶ, ggml_cann_type_mapping(dst->type),
                                                     elem_ಗಾತ್ರ, dst3d_ne, dst3d_nb, 3);

    // CANN tensor [ne23, ne1, 2*ne0]: split along CANN dim 2 (last) = 2*ne0.
    GGML_CANN_CALL_ACLNN_OP(ctx, SwiGlu, acl_src.get(), (int64_t)2, acl_dst.get());
}

ಶೂನ್ಯ ggml_cann_swiglu_clamp(ggml_backend_cann_context & ctx, ggml_tensor * dst) {
    ggml_tensor * src0 = dst->src[0];
    ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(ggml_is_contiguous_1(src0));
    GGML_ASSERT(ggml_is_contiguous_1(dst));

    ಸ್ಥಿರಾಂಕ int32_t  swapped = ggml_get_op_params_i32(dst, 1);
    acl_tensor_ptr acl_gate;
    acl_tensor_ptr acl_ಮೇಲೆ;
    ಆದರೆ (src1) {
        GGML_ASSERT(ggml_is_contiguous_1(src1));
        GGML_ASSERT(src0->type == src1->type);
        acl_gate = ggml_cann_create_tensor(src0);
        acl_ಮೇಲೆ   = ggml_cann_create_tensor(src1);
    } ಇಲ್ಲದಿದ್ದರೆ {
        int64_t ne[] = { src0->ne[0] / 2, src0->ne[1], src0->ne[2], src0->ne[3] };
        size_t  nb[] = { src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3] };
        acl_gate     = ggml_cann_create_tensor(src0, ne, nb, GGML_MAX_DIMS, ACL_FORMAT_ND, 0);
        acl_ಮೇಲೆ = ggml_cann_create_tensor(src0, ne, nb, GGML_MAX_DIMS, ACL_FORMAT_ND, ne[0] * ggml_element_ಗಾತ್ರ(src0));
        ಆದರೆ (swapped) {
            std::swap(acl_gate, acl_ಮೇಲೆ);
        }
    }

    ggml_cann_pool_alloc ತಾತ್ಕಾಲಿಕ_alloc(ctx.pool(), ggml_nbytes(dst));
    acl_tensor_ptr       acl_ತಾತ್ಕಾಲಿಕ = ggml_cann_create_tensor(ತಾತ್ಕಾಲಿಕ_alloc.get(), ggml_cann_type_mapping(dst->type),
                                                            ggml_element_ಗಾತ್ರ(dst), dst->ne, dst->nb, GGML_MAX_DIMS);
    acl_tensor_ptr       acl_dst  = ggml_cann_create_tensor(dst);

    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ    ಎಲ್ಲೆ        = ggml_get_op_params_f32(dst, 3);
    ದಶಮಾಂಶ          min_gate     = -INFINITY;
    ದಶಮಾಂಶ          min_ಮೇಲೆ       = -ಎಲ್ಲೆ;
    ದಶಮಾಂಶ          max_ಮೌಲ್ಯ    = ಎಲ್ಲೆ;
    acl_scalar_ptr acl_min_gate = ggml_cann_create_scalar(&min_gate, ACL_FLOAT);
    acl_scalar_ptr acl_min_ಮೇಲೆ   = ggml_cann_create_scalar(&min_ಮೇಲೆ, ACL_FLOAT);
    acl_scalar_ptr acl_ಎಲ್ಲೆ    = ggml_cann_create_scalar(&max_ಮೌಲ್ಯ, ACL_FLOAT);

    GGML_CANN_CALL_ACLNN_OP(ctx, Clamp, acl_gate.get(), acl_min_gate.get(), acl_ಎಲ್ಲೆ.get(), acl_ತಾತ್ಕಾಲಿಕ.get());
    GGML_CANN_CALL_ACLNN_OP(ctx, Silu, acl_ತಾತ್ಕಾಲಿಕ.get(), acl_dst.get());
    GGML_CANN_CALL_ACLNN_OP(ctx, Clamp, acl_ಮೇಲೆ.get(), acl_min_ಮೇಲೆ.get(), acl_ಎಲ್ಲೆ.get(), acl_ತಾತ್ಕಾಲಿಕ.get());
    GGML_CANN_CALL_ACLNN_OP(ctx, InplaceMul, acl_dst.get(), acl_ತಾತ್ಕಾಲಿಕ.get());
}

// Fused GeGLU using aclnnGeGluV3: splits input along ne[0] (CANN last dim),
// activates the LEFT half with GELU, multiplies by right half.
// approximate: 0=tanh, 1=none(erf). activateLeft=true matches GGML convention.
// outGelu is a required-but-discard output buffer.
//
// Falls back to the generic two-kernel path when src[1] != nullptr (two
// independent halves) or swapped != 0 (reversed activation order), as
// aclnnGeGluV3 only handles the single interleaved tensor in standard order.
ಶೂನ್ಯ ggml_cann_geglu(ggml_backend_cann_context & ctx, ggml_tensor * dst, int64_t approximate) {
    ಸ್ವಯಂಚಾಲಿತ gelu_fn = [](ggml_backend_cann_context & ctx, aclTensor * acl_src, aclTensor * acl_dst) {
        GGML_CANN_CALL_ACLNN_OP(ctx, Gelu, acl_src, acl_dst);
    };

    ಸ್ಥಿರಾಂಕ int32_t swapped = ggml_get_op_params_i32(dst, 1);
    ಆದರೆ (dst->src[1] != ಶೂನ್ಯಸೂಚಕ || swapped != 0) {
        ggml_cann_op_unary_gated(gelu_fn, ctx, dst);
        ಹಿಂತಿರುಗಿ;
    }

    // aclnnGeGluV3 requires the split dim (src->ne[0]) to be even; fall back otherwise.
    ಆದರೆ (dst->src[0]->ne[0] % 2 != 0) {
        ggml_cann_op_unary_gated(gelu_fn, ctx, dst);
        ಹಿಂತಿರುಗಿ;
    }

    ggml_tensor * src0 = dst->src[0];
    acl_tensor_ptr acl_src = ggml_cann_create_tensor(src0);
    acl_tensor_ptr acl_dst = ggml_cann_create_tensor(dst);

    // Allocate a temporary buffer for the required outGelu output (same shape as dst).
    // Build contiguous strides since the pool allocation is a fresh buffer.
    size_t  elem_ಗಾತ್ರ    = ggml_element_ಗಾತ್ರ(dst);
    int64_t ne[GGML_MAX_DIMS] = { dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3] };
    size_t  nb[GGML_MAX_DIMS];
    nb[0] = elem_ಗಾತ್ರ;
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 1; i < GGML_MAX_DIMS; i++) {
        nb[i] = nb[i - 1] * ne[i - 1];
    }
    size_t gelu_ಹೊರಹೋಗು_ಗಾತ್ರ = nb[GGML_MAX_DIMS - 1] * ne[GGML_MAX_DIMS - 1];
    ggml_cann_pool_alloc gelu_ಹೊರಹೋಗು_alloc(ctx.pool(), gelu_ಹೊರಹೋಗು_ಗಾತ್ರ);

    acl_tensor_ptr acl_gelu_ಹೊರಹೋಗು = ggml_cann_create_tensor(
        gelu_ಹೊರಹೋಗು_alloc.get(), ggml_cann_type_mapping(dst->type), elem_ಗಾತ್ರ, ne, nb, GGML_MAX_DIMS);
    // V3 adds activateLeft param; true → Gelu(left)*right, matching GGML convention.
    // GGML dim 0 → CANN last dim (index GGML_MAX_DIMS-1 = 3 for 4D tensor).
    GGML_CANN_CALL_ACLNN_OP(ctx, GeGluV3, acl_src.get(), (int64_t)(GGML_MAX_DIMS - 1), approximate, ನಿಜ,
                             acl_dst.get(), acl_gelu_ಹೊರಹೋಗು.get());
}

/**
 * @brief Repeats elements of a tensor along each dimension according to the
 * specified repeat array.
 *
 * @param ctx The context for the CANN backend operations.
 * @param acl_src The source tensor to be repeated.
 * @param acl_dst The destination tensor after repeating.
 * @param repeat_array The array specifying the number of repetitions along each
 * dimension.
 */
ಸ್ಥಿರ ಶೂನ್ಯ aclnn_repeat(ggml_backend_cann_context & ctx,
                         aclTensor *                 acl_src,
                         aclTensor *                 acl_dst,
                         int64_t *                   repeat_array) {
    // repeat tensor along each dim with repeat_array
    acl_int_array_ptr repeats = ggml_cann_create_int_array(repeat_array, GGML_MAX_DIMS);

    GGML_CANN_CALL_ACLNN_OP(ctx, Repeat, acl_src, repeats.get(), acl_dst);
}

/**
 * @brief Casts the data type of a source tensor to a destination tensor.
 *
 * This function casts the data type of the source tensor `acl_src` to the
 * specified data type `cast_data_type` and stores the result in the destination
 * tensor `acl_dst`.
 *
 * @param ctx The context for the CANN backend operations.
 * @param acl_src The source tensor whose data type will be casted.
 * @param acl_dst The destination tensor where the casted result will be stored.
 * @param cast_data_type The target data type to which the source tensor will be
 * casted.
 */
ಸ್ಥಿರ ಶೂನ್ಯ aclnn_cast(ggml_backend_cann_context & ctx,
                       aclTensor *                 acl_src,
                       aclTensor *                 acl_dst,
                       acl_ದತ್ತಾಂಶ_Type                 cast_ದತ್ತಾಂಶ_type) {
    GGML_CANN_CALL_ACLNN_OP(ctx, Cast, acl_src, cast_ದತ್ತಾಂಶ_type, acl_dst);
}

ಶೂನ್ಯ ggml_cann_repeat(ggml_backend_cann_context & ctx, ggml_tensor * dst) {
    ggml_tensor * src = dst->src[0];
    GGML_ASSERT(ggml_can_repeat(src, dst));

    acl_tensor_ptr acl_src = ggml_cann_create_tensor(src);
    acl_tensor_ptr acl_dst = ggml_cann_create_tensor(dst);

    int64_t repeatsArray[] = { dst->ne[3] / src->ne[3], dst->ne[2] / src->ne[2], dst->ne[1] / src->ne[1],
                               dst->ne[0] / src->ne[0] };

    aclnn_repeat(ctx, acl_src.get(), acl_dst.get(), repeatsArray);
}

ಶೂನ್ಯ aclnn_ಕೂಡಿಸು(ggml_backend_cann_context & ctx, aclTensor * acl_src0, aclTensor * acl_src1, aclTensor * acl_dst) {
    ದಶಮಾಂಶ          alpha_ಮೌಲ್ಯ = 1.0f;
    acl_scalar_ptr alpha      = ggml_cann_create_scalar(&alpha_ಮೌಲ್ಯ, acl_ದತ್ತಾಂಶ_Type::ACL_FLOAT);
    ಆದರೆ (acl_dst != ಶೂನ್ಯಸೂಚಕ) {
        GGML_CANN_CALL_ACLNN_OP(ctx, ಕೂಡಿಸು, acl_src0, acl_src1, alpha.get(), acl_dst);
    } ಇಲ್ಲದಿದ್ದರೆ {
        GGML_CANN_CALL_ACLNN_OP(ctx, Inplace_ಕೂಡಿಸು, acl_src0, acl_src1, alpha.get());
    }
}

ಶೂನ್ಯ aclnn_sub(ggml_backend_cann_context & ctx, aclTensor * acl_src0, aclTensor * acl_src1, aclTensor * acl_dst) {
    ದಶಮಾಂಶ          alpha_ಮೌಲ್ಯ = 1.0f;
    acl_scalar_ptr alpha      = ggml_cann_create_scalar(&alpha_ಮೌಲ್ಯ, acl_ದತ್ತಾಂಶ_Type::ACL_FLOAT);
    ಆದರೆ (acl_dst != ಶೂನ್ಯಸೂಚಕ) {
        GGML_CANN_CALL_ACLNN_OP(ctx, Sub, acl_src0, acl_src1, alpha.get(), acl_dst);
    } ಇಲ್ಲದಿದ್ದರೆ {
        GGML_CANN_CALL_ACLNN_OP(ctx, InplaceSub, acl_src0, acl_src1, alpha.get());
    }
}

ಶೂನ್ಯ aclnn_mul(ggml_backend_cann_context & ctx, aclTensor * acl_src, aclTensor * acl_other, aclTensor * acl_dst) {
    ಆದರೆ (acl_dst != ಶೂನ್ಯಸೂಚಕ) {
        GGML_CANN_CALL_ACLNN_OP(ctx, Mul, acl_src, acl_other, acl_dst);
    } ಇಲ್ಲದಿದ್ದರೆ {
        GGML_CANN_CALL_ACLNN_OP(ctx, InplaceMul, acl_src, acl_other);
    }
}

ಶೂನ್ಯ aclnn_div(ggml_backend_cann_context & ctx, aclTensor * acl_src, aclTensor * acl_other, aclTensor * acl_dst) {
    ಆದರೆ (acl_dst != ಶೂನ್ಯಸೂಚಕ) {
        GGML_CANN_CALL_ACLNN_OP(ctx, Div, acl_src, acl_other, acl_dst);
    } ಇಲ್ಲದಿದ್ದರೆ {
        GGML_CANN_CALL_ACLNN_OP(ctx, InplaceDiv, acl_src, acl_other);
    }
}

/**
 * @brief Multiplies elements of a tensor by a scalar value, optionally
 * in-place.
 *
 * This function multiplies each element of the source tensor `acl_src` by the
 * scalar `scale` and stores the result in the destination tensor `acl_dst`. If
 * `inplace` is true, `acl_dst` will not be used and the operation is performed
 *  in-place on `acl_src`.
 * The operation is defined as:
 * \f[
 *     \text {acl_dst }_i=\text {acl_src }_i \times \text {scale}
 * \f]
 *
 * @param ctx The context for the CANN backend operations.
 * @param acl_src The source tensor whose elements will be multiplied.
 * @param scale The scalar value by which each element of `acl_src` will be
 *  multiplied.
 * @param acl_dst The destination tensor where the result will be stored if
 * `inplace` is false.
 * @param inplace Flag indicating whether to perform the operation in-place on
 * `acl_src`.
 */
ಸ್ಥಿರ ಶೂನ್ಯ aclnn_muls(ggml_backend_cann_context & ctx,
                       aclTensor *                 acl_src,
                       ದಶಮಾಂಶ                       scale,
                       aclTensor *                 acl_dst,
                       ತಾರ್ಕಿಕ                        inplace) {
    acl_scalar_ptr acl_scale = ggml_cann_create_scalar(&scale, acl_ದತ್ತಾಂಶ_Type::ACL_FLOAT);
    ಆದರೆ (inplace) {
        GGML_CANN_CALL_ACLNN_OP(ctx, InplaceMuls, acl_src, acl_scale.get());
    } ಇಲ್ಲದಿದ್ದರೆ {
        GGML_CANN_CALL_ACLNN_OP(ctx, Muls, acl_src, acl_scale.get(), acl_dst);
    }
}

ಶೂನ್ಯ ggml_cann_leaky_relu(ggml_backend_cann_context & ctx, ggml_tensor * dst) {
    ggml_tensor * src = dst->src[0];

    acl_tensor_ptr acl_src = ggml_cann_create_tensor(src);
    acl_tensor_ptr acl_dst = ggml_cann_create_tensor(dst);

    ದಶಮಾಂಶ negative_ಇಳಿಜಾರು;
    ಸ್ಮರಣೆ_ನಕಲು(&negative_ಇಳಿಜಾರು, dst->op_params, ಗಾತ್ರ(ದಶಮಾಂಶ));
    acl_scalar_ptr acl_negative_ಇಳಿಜಾರು = ggml_cann_create_scalar(&negative_ಇಳಿಜಾರು, acl_ದತ್ತಾಂಶ_Type::ACL_FLOAT);

    GGML_CANN_CALL_ACLNN_OP(ctx, LeakyRelu, acl_src.get(), acl_negative_ಇಳಿಜಾರು.get(), acl_dst.get());
}

/**
 * @brief Concatenates a list of tensors along a specified dimension and stores
 * the result in a destination tensor.
 *
 * @param ctx The context for the CANN backend operations.
 * @param tensorList The list of tensors to be concatenated.
 * @param acl_dst The destination tensor where the concatenated result will be
 * stored.
 * @param concat_dim The dimension along which the tensors will be concatenated.
 */
ಸ್ಥಿರ ಶೂನ್ಯ aclnn_concat(ggml_backend_cann_context & ctx,
                         aclTensorList *             tensorList,
                         aclTensor *                 acl_dst,
                         int64_t                     concat_dim) {
    GGML_CANN_CALL_ACLNN_OP(ctx, Cat, tensorList, concat_dim, acl_dst);
}

ಶೂನ್ಯ ggml_cann_concat(ggml_backend_cann_context & ctx, ggml_tensor * dst) {
    ggml_tensor *  src0     = dst->src[0];
    ggml_tensor *  src1     = dst->src[1];
    acl_tensor_ptr acl_src0 = ggml_cann_create_tensor(src0);
    acl_tensor_ptr acl_src1 = ggml_cann_create_tensor(src1);
    acl_tensor_ptr acl_dst  = ggml_cann_create_tensor(dst);

    ಸ್ಥಿರಾಂಕ int32_t dim = ggml_get_op_params_i32(dst, 0);

    GGML_ASSERT(dim >= 0 && dim < 4);
    int32_t acl_dim = 3 - dim;

    acl_tensor_list_ptr tensor_list = ggml_cann_create_tensor_list(acl_src0, acl_src1);
    aclnn_concat(ctx, tensor_list.get(), acl_dst.get(), acl_dim);
}

/**
 * @brief Creates a tensor with values starting from `start`, incremented by
 * `step`, and ending before `stop`.
 *
 * This function performs the operation:
 * \f[
 *    \text {out }_{i+1}=\text {out }_i+\text {step}
 * \f]
 * the range is [start, stop).
 *
 * @param ctx The context for the CANN backend operations.
 * @param acl_dst The destination tensor where the values will be stored.
 * @param start The starting value of the range.
 * @param stop The ending value of the range (exclusive).
 * @param step The step size between consecutive values.
 * @param n_elements The number of elements in the destination tensor.
 */
ಸ್ಥಿರ ಶೂನ್ಯ aclnn_arange(ggml_backend_cann_context & ctx,
                         aclTensor *                 acl_dst,
                         ದಶಮಾಂಶ                       start,
                         ದಶಮಾಂಶ                       stop,
                         ದಶಮಾಂಶ                       ಹೆಜ್ಜೆ,
                         int64_t                     n_elements) {
    int64_t ಹೆಜ್ಜೆಗಳು = (int64_t) std::ceil((stop - start) / ಹೆಜ್ಜೆ);
    GGML_ASSERT(n_elements == ಹೆಜ್ಜೆಗಳು);

    acl_scalar_ptr acl_ಆರಂಭ = ggml_cann_create_scalar(&start, acl_ದತ್ತಾಂಶ_Type::ACL_FLOAT);
    acl_scalar_ptr acl_ಕೊನೆ   = ggml_cann_create_scalar(&stop, acl_ದತ್ತಾಂಶ_Type::ACL_FLOAT);
    acl_scalar_ptr acl_ಹೆಜ್ಜೆ  = ggml_cann_create_scalar(&ಹೆಜ್ಜೆ, acl_ದತ್ತಾಂಶ_Type::ACL_FLOAT);

    GGML_CANN_CALL_ACLNN_OP(ctx, Arange, acl_ಆರಂಭ.get(), acl_ಕೊನೆ.get(), acl_ಹೆಜ್ಜೆ.get(), acl_dst);
}

ಶೂನ್ಯ ggml_cann_arange(ggml_backend_cann_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    acl_tensor_ptr acl_dst = ggml_cann_create_tensor(dst);

    int64_t n_elements = ggml_nelements(dst);
    ದಶಮಾಂಶ   start;
    ದಶಮಾಂಶ   stop;
    ದಶಮಾಂಶ   ಹೆಜ್ಜೆ;
    ಸ್ಮರಣೆ_ನಕಲು(&start, (ದಶಮಾಂಶ *) dst->op_params + 0, ಗಾತ್ರ(ದಶಮಾಂಶ));
    ಸ್ಮರಣೆ_ನಕಲು(&stop, (ದಶಮಾಂಶ *) dst->op_params + 1, ಗಾತ್ರ(ದಶಮಾಂಶ));
    ಸ್ಮರಣೆ_ನಕಲು(&ಹೆಜ್ಜೆ, (ದಶಮಾಂಶ *) dst->op_params + 2, ಗಾತ್ರ(ದಶಮಾಂಶ));

    aclnn_arange(ctx, acl_dst.get(), start, stop, ಹೆಜ್ಜೆ, n_elements);
}

ಶೂನ್ಯ ggml_cann_clamp(ggml_backend_cann_context & ctx, ggml_tensor * dst) {
    ggml_tensor * src = dst->src[0];

    ದಶಮಾಂಶ min;
    ದಶಮಾಂಶ max;
    ಸ್ಮರಣೆ_ನಕಲು(&min, dst->op_params, ಗಾತ್ರ(ದಶಮಾಂಶ));
    ಸ್ಮರಣೆ_ನಕಲು(&max, (ದಶಮಾಂಶ *) dst->op_params + 1, ಗಾತ್ರ(ದಶಮಾಂಶ));

    acl_tensor_ptr acl_src = ggml_cann_create_tensor(src);
    acl_tensor_ptr acl_dst = ggml_cann_create_tensor(dst);

    acl_scalar_ptr acl_min = ggml_cann_create_scalar(&min, acl_ದತ್ತಾಂಶ_Type::ACL_FLOAT);
    acl_scalar_ptr acl_max = ggml_cann_create_scalar(&max, acl_ದತ್ತಾಂಶ_Type::ACL_FLOAT);

    GGML_CANN_CALL_ACLNN_OP(ctx, Clamp, acl_src.get(), acl_min.get(), acl_max.get(), acl_dst.get());
}

ಶೂನ್ಯ ggml_cann_scale(ggml_backend_cann_context & ctx, ggml_tensor * dst) {
    ggml_tensor * src = dst->src[0];

    // scale factor
    ದಶಮಾಂಶ v;
    ಸ್ಮರಣೆ_ನಕಲು(&v, dst->op_params, ಗಾತ್ರ(ದಶಮಾಂಶ));

    acl_scalar_ptr scale   = ggml_cann_create_scalar(&v, acl_ದತ್ತಾಂಶ_Type::ACL_FLOAT);
    acl_tensor_ptr acl_src = ggml_cann_create_tensor(src);
    acl_tensor_ptr acl_dst = ggml_cann_create_tensor(dst);

    GGML_CANN_CALL_ACLNN_OP(ctx, Muls, acl_src.get(), scale.get(), acl_dst.get());
}

ಶೂನ್ಯ ggml_cann_argsort(ggml_backend_cann_context & ctx, ggml_tensor * dst) {
    ggml_tensor *        src   = dst->src[0];
    ಎಣಿಕೆ ggml_ವಿಂಗಡಿಸು_ಕ್ರಮ ಕ್ರಮ = (ಎಣಿಕೆ ggml_ವಿಂಗಡಿಸು_ಕ್ರಮ) dst->op_params[0];

    acl_tensor_ptr       acl_src = ggml_cann_create_tensor(src);
    acl_tensor_ptr       acl_dst = ggml_cann_create_tensor(dst);
    ggml_cann_pool_alloc ತಾತ್ಕಾಲಿಕ_ಬಫರ್_allocator(ctx.pool(), ggml_nelements(dst) * ಗಾತ್ರ(int64_t));
    ಶೂನ್ಯ *               ಬಫರ್ = ತಾತ್ಕಾಲಿಕ_ಬಫರ್_allocator.get();
    acl_tensor_ptr       tmp_tensor =
        ggml_cann_create_tensor(ಬಫರ್, ACL_INT64, ggml_type_ಗಾತ್ರ(dst->type), dst->ne, dst->nb, GGML_MAX_DIMS);
    GGML_CANN_CALL_ACLNN_OP(ctx, Argsort, acl_src.get(), -1, (ಕ್ರಮ == GGML_ವಿಂಗಡಿಸು_ಕ್ರಮ_DESC ? ನಿಜ : ಸುಳ್ಳು),
                            tmp_tensor.get());
    GGML_CANN_CALL_ACLNN_OP(ctx, Cast, tmp_tensor.get(), ggml_cann_type_mapping(dst->type), acl_dst.get());
}

ಶೂನ್ಯ ggml_cann_norm(ggml_backend_cann_context & ctx, ggml_tensor * dst) {
    ggml_tensor * src = dst->src[0];

    acl_tensor_ptr acl_src = ggml_cann_create_tensor(src);
    acl_tensor_ptr acl_dst = ggml_cann_create_tensor(dst);

    ದಶಮಾಂಶ eps;
    ಸ್ಮರಣೆ_ನಕಲು(&eps, dst->op_params, ಗಾತ್ರ(ದಶಮಾಂಶ));

    std::vector<int64_t> norm_ದತ್ತಾಂಶ = { dst->ne[0] };
    acl_int_array_ptr    norm     = ggml_cann_create_int_array(norm_ದತ್ತಾಂಶ.ದತ್ತಾಂಶ(), norm_ದತ್ತಾಂಶ.size());
    GGML_CANN_CALL_ACLNN_OP(ctx, LayerNorm, acl_src.get(), norm.get(), ಶೂನ್ಯಸೂಚಕ, ಶೂನ್ಯಸೂಚಕ, eps, acl_dst.get(), ಶೂನ್ಯಸೂಚಕ,
                            ಶೂನ್ಯಸೂಚಕ);
}

ಶೂನ್ಯ ggml_cann_l2_norm(ggml_backend_cann_context & ctx, ggml_tensor * dst) {
    ggml_tensor * src = dst->src[0];

    ದಶಮಾಂಶ eps;
    ಸ್ಮರಣೆ_ನಕಲು(&eps, dst->op_params, ಗಾತ್ರ(ದಶಮಾಂಶ));

    acl_tensor_ptr acl_src = ggml_cann_create_tensor(src);
    acl_tensor_ptr acl_dst = ggml_cann_create_tensor(dst);

    size_t               type_ಗಾತ್ರ = ggml_type_ಗಾತ್ರ(src->type);
    int64_t              n_ಬೈಟುಗಳು   = src->ne[3] * src->ne[2] * src->ne[1] * type_ಗಾತ್ರ;
    ggml_cann_pool_alloc ತಾತ್ಕಾಲಿಕ_ಬಫರ್_allocator(ctx.pool(), n_ಬೈಟುಗಳು);
    ಶೂನ್ಯ *               ಬಫರ್ = ತಾತ್ಕಾಲಿಕ_ಬಫರ್_allocator.get();

    int64_t norm_ne[] = { 1, src->ne[1], src->ne[2], src->ne[3] };
    size_t  norm_nb[GGML_MAX_DIMS];
    norm_nb[0] = ಗಾತ್ರ(ದಶಮಾಂಶ);
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 1; i < GGML_MAX_DIMS; ++i) {
        norm_nb[i] = norm_nb[i - 1] * norm_ne[i - 1];
    }
    acl_tensor_ptr acl_norm = ggml_cann_create_tensor(ಬಫರ್, ACL_FLOAT, ಗಾತ್ರ(ದಶಮಾಂಶ), norm_ne, norm_nb, GGML_MAX_DIMS);

    std::vector<int64_t> norm_dims  = { 3 };
    acl_int_array_ptr    dims_array = ggml_cann_create_int_array(norm_dims.ದತ್ತಾಂಶ(), norm_dims.size());

    ದಶಮಾಂಶ          p_ಮೌಲ್ಯ  = 2.0f;
    acl_scalar_ptr p_scalar = ggml_cann_create_scalar(&p_ಮೌಲ್ಯ, acl_ದತ್ತಾಂಶ_Type::ACL_FLOAT);
    GGML_CANN_CALL_ACLNN_OP(ctx, Norm, acl_src.get(), p_scalar.get(), dims_array.get(), ನಿಜ, acl_norm.get());

    ggml_cann_pool_alloc clamp_ಬಫರ್_allocator(ctx.pool());
    acl_tensor_ptr       acl_clamped;

    ಆದರೆ (eps > 0.0f) {
        ಶೂನ್ಯ *         clamp_buf  = clamp_ಬಫರ್_allocator.alloc(n_ಬೈಟುಗಳು);
        acl_clamped               = ggml_cann_create_tensor(clamp_buf, ACL_FLOAT, ಗಾತ್ರ(ದಶಮಾಂಶ), norm_ne, norm_nb, GGML_MAX_DIMS);
        acl_scalar_ptr eps_scalar = ggml_cann_create_scalar(&eps, acl_ದತ್ತಾಂಶ_Type::ACL_FLOAT);
        GGML_CANN_CALL_ACLNN_OP(ctx, ClampMin, acl_norm.get(), eps_scalar.get(), acl_clamped.get());
    }

    aclTensor * acl_div_ಒಳಹರಿವು = acl_clamped ? acl_clamped.get() : acl_norm.get();
    GGML_CANN_CALL_ACLNN_OP(ctx, Div, acl_src.get(), acl_div_ಒಳಹರಿವು, acl_dst.get());
}

ಶೂನ್ಯ ggml_cann_cross_entropy_loss(ggml_backend_cann_context & ctx, ggml_tensor * dst) {
    ggml_tensor * src0 = dst->src[0];
    ggml_tensor * src1 = dst->src[1];

    ಸ್ಥಿರಾಂಕ int64_t nc = src0->ne[0];
    ಸ್ಥಿರಾಂಕ int64_t nr = ggml_nrows(src0);

    int64_t logits_ne[] = { nc, nr };
    size_t  logits_nb[2];
    logits_nb[0]              = ggml_type_ಗಾತ್ರ(src0->type);
    logits_nb[1]              = logits_nb[0] * logits_ne[0];
    acl_tensor_ptr acl_logits = ggml_cann_create_tensor(src0->ದತ್ತಾಂಶ, ACL_FLOAT, ಗಾತ್ರ(ದಶಮಾಂಶ), logits_ne, logits_nb, 2);

    int64_t labels_ne[] = { nc, nr };
    size_t  labels_nb[2];
    labels_nb[0]              = ggml_type_ಗಾತ್ರ(src1->type);
    labels_nb[1]              = labels_nb[0] * labels_ne[0];
    acl_tensor_ptr acl_labels = ggml_cann_create_tensor(src1->ದತ್ತಾಂಶ, ACL_FLOAT, ಗಾತ್ರ(ದಶಮಾಂಶ), labels_ne, labels_nb, 2);

    size_t               loss_ಪ್ರತಿ_sample_type_ಗಾತ್ರ = ಗಾತ್ರ(ದಶಮಾಂಶ);
    int64_t              loss_ಪ್ರತಿ_sample_n_ಬೈಟುಗಳು   = nr * loss_ಪ್ರತಿ_sample_type_ಗಾತ್ರ;
    ggml_cann_pool_alloc loss_ಪ್ರತಿ_sample_allocator(ctx.pool(), loss_ಪ್ರತಿ_sample_n_ಬೈಟುಗಳು);
    ಶೂನ್ಯ *               loss_ಪ್ರತಿ_sample_ಬಫರ್ = loss_ಪ್ರತಿ_sample_allocator.get();

    int64_t loss_ಪ್ರತಿ_sample_ne[] = { nr };
    size_t  loss_ಪ್ರತಿ_sample_nb[1];
    loss_ಪ್ರತಿ_sample_nb[0] = loss_ಪ್ರತಿ_sample_type_ಗಾತ್ರ;
    acl_tensor_ptr acl_loss_ಪ್ರತಿ_sample = ggml_cann_create_tensor(
        loss_ಪ್ರತಿ_sample_ಬಫರ್, ACL_FLOAT, loss_ಪ್ರತಿ_sample_type_ಗಾತ್ರ, loss_ಪ್ರತಿ_sample_ne, loss_ಪ್ರತಿ_sample_nb, 1);

    size_t               backprop_n_ಬೈಟುಗಳು = nr * nc * ಗಾತ್ರ(ದಶಮಾಂಶ);
    ggml_cann_pool_alloc backprop_allocator(ctx.pool(), backprop_n_ಬೈಟುಗಳು);
    ಶೂನ್ಯ *               backprop_ಬಫರ್ = backprop_allocator.get();
    acl_tensor_ptr acl_backprop = ggml_cann_create_tensor(backprop_ಬಫರ್, ACL_FLOAT, ಗಾತ್ರ(ದಶಮಾಂಶ), logits_ne, logits_nb, 2);

    GGML_CANN_CALL_ACLNN_OP(ctx, SoftmaxCrossEntropyWithLogits, acl_logits.get(), acl_labels.get(),
                            acl_loss_ಪ್ರತಿ_sample.get(), acl_backprop.get());

    size_t               ಒಟ್ಟು_ಮೊತ್ತ_type_ಗಾತ್ರ = ಗಾತ್ರ(ದಶಮಾಂಶ);
    int64_t              ಒಟ್ಟು_ಮೊತ್ತ_n_ಬೈಟುಗಳು   = 1 * ಒಟ್ಟು_ಮೊತ್ತ_type_ಗಾತ್ರ;
    ggml_cann_pool_alloc ಒಟ್ಟು_ಮೊತ್ತ_allocator(ctx.pool(), ಒಟ್ಟು_ಮೊತ್ತ_n_ಬೈಟುಗಳು);
    ಶೂನ್ಯ *               ಒಟ್ಟು_ಮೊತ್ತ_ಬಫರ್ = ಒಟ್ಟು_ಮೊತ್ತ_allocator.get();

    int64_t ಒಟ್ಟು_ಮೊತ್ತ_ne[] = { 1 };
    size_t  ಒಟ್ಟು_ಮೊತ್ತ_nb[1];
    ಒಟ್ಟು_ಮೊತ್ತ_nb[0] = ಒಟ್ಟು_ಮೊತ್ತ_type_ಗಾತ್ರ;

    acl_tensor_ptr acl_ಒಟ್ಟು_ಮೊತ್ತ =
        ggml_cann_create_tensor(ಒಟ್ಟು_ಮೊತ್ತ_ಬಫರ್, ACL_FLOAT, ಒಟ್ಟು_ಮೊತ್ತ_type_ಗಾತ್ರ, ಒಟ್ಟು_ಮೊತ್ತ_ne, ಒಟ್ಟು_ಮೊತ್ತ_nb, 1);

    std::vector<int64_t> ಒಟ್ಟು_ಮೊತ್ತ_dims    = { 0 };
    acl_int_array_ptr ಒಟ್ಟು_ಮೊತ್ತ_dims_array = ggml_cann_create_int_array(ಒಟ್ಟು_ಮೊತ್ತ_dims.ದತ್ತಾಂಶ(), ಒಟ್ಟು_ಮೊತ್ತ_dims.size());
    ತಾರ್ಕಿಕ              keep_dims            = ಸುಳ್ಳು;

    GGML_CANN_CALL_ACLNN_OP(ctx, Reduce_ಮೊತ್ತ, acl_loss_ಪ್ರತಿ_sample.get(), ಒಟ್ಟು_ಮೊತ್ತ_dims_array.get(), keep_dims, ACL_FLOAT,
                            acl_ಒಟ್ಟು_ಮೊತ್ತ.get());

    ದಶಮಾಂಶ          value        = 1.0f / static_cast<ದಶಮಾಂಶ>(nr);
    acl_scalar_ptr scale_ಗುಣಕ = ggml_cann_create_scalar(&value, acl_ದತ್ತಾಂಶ_Type::ACL_FLOAT);
    acl_tensor_ptr acl_dst =
        ggml_cann_create_tensor(dst->ದತ್ತಾಂಶ, ACL_FLOAT, ಗಾತ್ರ(ದಶಮಾಂಶ), ಒಟ್ಟು_ಮೊತ್ತ_ne, ಒಟ್ಟು_ಮೊತ್ತ_nb, 1);

    GGML_CANN_CALL_ACLNN_OP(ctx, Muls, acl_ಒಟ್ಟು_ಮೊತ್ತ.get(), scale_ಗುಣಕ.get(), acl_dst.get());
}

ಶೂನ್ಯ ggml_cann_ಗುಂಪು_norm(ggml_backend_cann_context & ctx, ggml_tensor * dst) {
    ggml_tensor * src = dst->src[0];

    acl_tensor_ptr acl_src = ggml_cann_create_tensor(src);
    acl_tensor_ptr acl_dst = ggml_cann_create_tensor(dst);

    ಪೂರ್ಣಾಂಕ n_groups = dst->op_params[0];

    ದಶಮಾಂಶ eps;
    ಸ್ಮರಣೆ_ನಕಲು(&eps, dst->op_params + 1, ಗಾತ್ರ(ದಶಮಾಂಶ));

    int64_t N   = src->ne[3];
    int64_t C   = src->ne[2];
    int64_t HxW = src->ne[1] * src->ne[0];

    size_t  type_ಗಾತ್ರ = ggml_type_ಗಾತ್ರ(src->type);
    int64_t ne[]      = { n_groups, N };
    size_t  nb[]      = { type_ಗಾತ್ರ, type_ಗಾತ್ರ * n_groups };
    size_t  n_ಬೈಟುಗಳು   = N * n_groups;

    ggml_cann_pool_alloc ತಾತ್ಕಾಲಿಕ_ಬಫರ್_allocator(ctx.pool(), n_ಬೈಟುಗಳು * 2);
    ಶೂನ್ಯ *               ಬಫರ್       = ತಾತ್ಕಾಲಿಕ_ಬಫರ್_allocator.get();
    acl_tensor_ptr       acl_ಸರಾಸರಿ_ಹೊರಹೋಗು = ggml_cann_create_tensor(ಬಫರ್, ACL_FLOAT, type_ಗಾತ್ರ, ne, nb, ACL_FORMAT_ND);
    acl_tensor_ptr       acl_rstd_ಹೊರಹೋಗು =
        ggml_cann_create_tensor((ಅಕ್ಷರ *) ಬಫರ್ + n_ಬೈಟುಗಳು, ACL_FLOAT, type_ಗಾತ್ರ, ne, nb, ACL_FORMAT_ND);

    GGML_CANN_CALL_ACLNN_OP(ctx, ಗುಂಪು_Norm, acl_src.get(), ಶೂನ್ಯಸೂಚಕ, ಶೂನ್ಯಸೂಚಕ, N, C, HxW, n_groups, eps, acl_dst.get(),
                            acl_ಸರಾಸರಿ_ಹೊರಹೋಗು.get(), acl_rstd_ಹೊರಹೋಗು.get());
}

ಶೂನ್ಯ ggml_cann_ಗಣ(ggml_backend_cann_context & ctx, ggml_tensor * dst) {
    ggml_tensor * src0 = dst->src[0];
    ggml_tensor * src1 = dst->src[1];

    size_t nb1     = ((int32_t *) dst->op_params)[0];
    size_t nb2     = ((int32_t *) dst->op_params)[1];
    size_t nb3     = ((int32_t *) dst->op_params)[2];
    size_t ಸ್ಥಾನಾಂತರ  = ((int32_t *) dst->op_params)[3];
    ತಾರ್ಕಿಕ   inplace = (ತಾರ್ಕಿಕ) ((int32_t *) dst->op_params)[4];

    size_t param_nb[] = { ggml_element_ಗಾತ್ರ(src0), nb1, nb2, nb3 };

    // Create a view of dst at the target offset with src1's dimensions
    acl_tensor_ptr acl_dst  = ggml_cann_create_tensor(dst, src1->ne, param_nb, GGML_MAX_DIMS, ACL_FORMAT_ND, ಸ್ಥಾನಾಂತರ);
    acl_tensor_ptr acl_src1 = ggml_cann_create_tensor(src1);

    ಆದರೆ (!inplace) {
        // First copy src0 to dst entirely
        size_t cpy_ಗಾತ್ರ = ggml_nbytes(dst);
        ACL_CHECK(
            aclrtMemcpyAsync(dst->ದತ್ತಾಂಶ, cpy_ಗಾತ್ರ, src0->ದತ್ತಾಂಶ, cpy_ಗಾತ್ರ, ACL_MEMCPY_DEVICE_ಗಮ್ಯ_DEVICE, ctx.stream()));
    }

    // Copy src1 into the target region of dst
    GGML_CANN_CALL_ACLNN_OP(ctx, InplaceCopy, acl_dst.get(), acl_src1.get());
}

ಶೂನ್ಯ ggml_cann_acc(ggml_backend_cann_context & ctx, ggml_tensor * dst) {
    ggml_tensor * src0 = dst->src[0];
    ggml_tensor * src1 = dst->src[1];

    size_t nb1     = ((int32_t *) dst->op_params)[0];
    size_t nb2     = ((int32_t *) dst->op_params)[1];
    size_t nb3     = ((int32_t *) dst->op_params)[2];
    size_t ಸ್ಥಾನಾಂತರ  = ((int32_t *) dst->op_params)[3];
    ತಾರ್ಕಿಕ   inplace = (ತಾರ್ಕಿಕ) ((int32_t *) dst->op_params)[4];

    size_t param_nb[] = { ggml_element_ಗಾತ್ರ(src0), nb1, nb2, nb3 };

    acl_tensor_ptr acl_dst  = ggml_cann_create_tensor(dst, src1->ne, param_nb, GGML_MAX_DIMS, ACL_FORMAT_ND, ಸ್ಥಾನಾಂತರ);
    acl_tensor_ptr acl_src1 = ggml_cann_create_tensor(src1);

    acl_scalar_ptr alpha      = ಶೂನ್ಯಸೂಚಕ;
    ದಶಮಾಂಶ          alpha_ಮೌಲ್ಯ = 1.0f;
    alpha                     = ggml_cann_create_scalar(&alpha_ಮೌಲ್ಯ, acl_ದತ್ತಾಂಶ_Type::ACL_FLOAT);

    ಆದರೆ (!inplace) {
        size_t cpy_ಗಾತ್ರ = ggml_nbytes(dst);
        ACL_CHECK(
            aclrtMemcpyAsync(dst->ದತ್ತಾಂಶ, cpy_ಗಾತ್ರ, src0->ದತ್ತಾಂಶ, cpy_ಗಾತ್ರ, ACL_MEMCPY_DEVICE_ಗಮ್ಯ_DEVICE, ctx.stream()));
        acl_tensor_ptr acl_src0 =
            ggml_cann_create_tensor(src0, src1->ne, src0->nb, GGML_MAX_DIMS, ACL_FORMAT_ND, ಸ್ಥಾನಾಂತರ);

        GGML_CANN_CALL_ACLNN_OP(ctx, ಕೂಡಿಸು, acl_src0.get(), acl_src1.get(), alpha.get(), acl_dst.get());
    } ಇಲ್ಲದಿದ್ದರೆ {
        GGML_CANN_CALL_ACLNN_OP(ctx, Inplace_ಕೂಡಿಸು, acl_dst.get(), acl_src1.get(), alpha.get());
    }
}

/**
 * @brief Performs sum reduction on a given tensor along specified dimensions.
 *
 * This function reduces the input tensor by summing along the specified dimensions.
 *
 * @param ctx The context for the CANN backend operations.
 * @param dst The destination tensor where the reduced result will be stored.
 * @param dim An array of dimension indices.
 * @param dim_size The number of dimensions.
 */
ಸ್ಥಿರ ಶೂನ್ಯ aclnn_reduce_ಮೊತ್ತ(ggml_backend_cann_context & ctx, ggml_tensor * dst, int64_t * dim, size_t dim_ಗಾತ್ರ) {
    GGML_ASSERT(dst->ne[0] == 1);
    ggml_tensor *     src         = dst->src[0];
    acl_tensor_ptr    acl_src     = ggml_cann_create_tensor(src);
    acl_tensor_ptr    acl_dst     = ggml_cann_create_tensor(dst);
    acl_int_array_ptr reduce_dims = ggml_cann_create_int_array(dim, dim_ಗಾತ್ರ);

    GGML_CANN_CALL_ACLNN_OP(ctx, Reduce_ಮೊತ್ತ, acl_src.get(), reduce_dims.get(), ನಿಜ, ggml_cann_type_mapping(dst->type),
                            acl_dst.get());
}

ಶೂನ್ಯ ggml_cann_ಮೊತ್ತ_ಅಡ್ಡಸಾಲುಗಳು(ggml_backend_cann_context & ctx, ggml_tensor * dst) {
    int64_t reduce_dims[] = { 3 };
    aclnn_reduce_ಮೊತ್ತ(ctx, dst, reduce_dims, 1);
}

ಶೂನ್ಯ ggml_cann_ಮೊತ್ತ(ggml_backend_cann_context & ctx, ggml_tensor * dst) {
    int64_t reduce_dims[] = { 0, 1, 2, 3 };
    aclnn_reduce_ಮೊತ್ತ(ctx, dst, reduce_dims, 4);
}

ಶೂನ್ಯ ggml_cann_cumsum(ggml_backend_cann_context & ctx, ggml_tensor * dst) {
    ggml_tensor * src = dst->src[0];
    acl_tensor_ptr acl_src = ggml_cann_create_tensor(src);
    acl_tensor_ptr acl_dst = ggml_cann_create_tensor(dst);
    // GGML cumsum operates along dim 0 (innermost / ne[0]).
    // ggml_cann_create_tensor reverses dimensions to [ne3,ne2,ne1,ne0],
    // so GGML dim 0 maps to CANN dim 3 (the last dim of the 4-D tensor).
    GGML_CANN_CALL_ACLNN_OP(ctx, Cumsum, acl_src.get(), (int64_t)3,
                            ggml_cann_type_mapping(dst->type), acl_dst.get());
}

ಶೂನ್ಯ ggml_cann_solve_tri(ggml_backend_cann_context & ctx, ggml_tensor * dst) {
    ggml_tensor * src0 = dst->src[0];  // A: [N, N, B2, B3] lower triangular
    ggml_tensor * src1 = dst->src[1];  // B: [K, N, B2, B3]

    acl_tensor_ptr acl_a = ggml_cann_create_tensor(src0);
    acl_tensor_ptr acl_b = ggml_cann_create_tensor(src1);
    acl_tensor_ptr acl_x = ggml_cann_create_tensor(dst);

    // mOut: triangular copy of A (required output), same shape as A.
    ಸ್ಥಿರಾಂಕ size_t a_ಬೈಟುಗಳು = ggml_nbytes(src0);
    ggml_cann_pool_alloc m_alloc(ctx.pool(), a_ಬೈಟುಗಳು);
    acl_tensor_ptr acl_m = ggml_cann_create_tensor(
        m_alloc.get(), ggml_cann_type_mapping(src0->type),
        ggml_type_ಗಾತ್ರ(src0->type), src0->ne, src0->nb, GGML_MAX_DIMS);

    // Solve AX = B: upper=false (lower tri), transpose=false, unitriangular=false.
    GGML_CANN_CALL_ACLNN_OP(ctx, TriangularSolve,
        acl_b.get(), acl_a.get(), ಸುಳ್ಳು, ಸುಳ್ಳು, ಸುಳ್ಳು,
        acl_x.get(), acl_m.get());
}

ಶೂನ್ಯ ggml_cann_diag(ggml_backend_cann_context & ctx, ggml_tensor * dst) {
    ggml_tensor * src = dst->src[0];

    GGML_ASSERT(src->ne[1] == 1);

    ಸ್ಥಿರಾಂಕ int64_t N       = src->ne[0];
    ಸ್ಥಿರಾಂಕ int64_t n_batch = src->ne[2] * src->ne[3];
    ಸ್ಥಿರಾಂಕ size_t  nb_f32  = ಗಾತ್ರ(ದಶಮಾಂಶ);

    // Fill dst with zeros.
    acl_tensor_ptr acl_dst = ggml_cann_create_tensor(dst);
    {
        ದಶಮಾಂಶ          zero = 0.0f;
        acl_scalar_ptr acl_zero = ggml_cann_create_scalar(&zero, ACL_FLOAT);
        GGML_CANN_CALL_ACLNN_OP(ctx, InplaceFillScalar, acl_dst.get(), acl_zero.get());
    }

    // Copy src vector onto the diagonal of dst via strided views.
    // src viewed as [N, n_batch], contiguous strides.
    int64_t ne_vec[2]      = { N, n_batch };
    size_t  nb_src_vec[2]  = { nb_f32, N * nb_f32 };
    // dst diagonal view: stride (N+1)*4 steps along the diagonal.
    size_t  nb_dst_diag[2] = { (N + 1) * nb_f32, N * N * nb_f32 };

    acl_tensor_ptr acl_src_vec  = ggml_cann_create_tensor(src->ದತ್ತಾಂಶ, ACL_FLOAT, nb_f32, ne_vec, nb_src_vec, 2);
    acl_tensor_ptr acl_dst_diag = ggml_cann_create_tensor(dst->ದತ್ತಾಂಶ, ACL_FLOAT, nb_f32, ne_vec, nb_dst_diag, 2);

    GGML_CANN_CALL_ACLNN_OP(ctx, InplaceCopy, acl_dst_diag.get(), acl_src_vec.get());
}

ಶೂನ್ಯ ggml_cann_fill(ggml_backend_cann_context & ctx, ggml_tensor * dst) {
    ದಶಮಾಂಶ c = ggml_get_op_params_f32(dst, 0);

    acl_tensor_ptr acl_dst = ggml_cann_create_tensor(dst);
    acl_scalar_ptr acl_c   = ggml_cann_create_scalar(&c, ACL_FLOAT);
    GGML_CANN_CALL_ACLNN_OP(ctx, InplaceFillScalar, acl_dst.get(), acl_c.get());
}

ಶೂನ್ಯ ggml_cann_tri(ggml_backend_cann_context & ctx, ggml_tensor * dst) {
    ggml_tensor * src = dst->src[0];

    ಸ್ಥಿರಾಂಕ int64_t S       = src->ne[0];
    ಸ್ಥಿರಾಂಕ int64_t n_batch = src->ne[2] * src->ne[3];
    ಸ್ಥಿರಾಂಕ size_t  nb_f32  = ಗಾತ್ರ(ದಶಮಾಂಶ);

    int64_t ne3d[3] = { S, S, n_batch };
    size_t  nb3d[3] = { nb_f32, S * nb_f32, S * S * nb_f32 };

    ಸ್ಥಿರಾಂಕ ggml_tri_type ttype = (ggml_tri_type) ggml_get_op_params_i32(dst, 0);

    acl_tensor_ptr acl_src = ggml_cann_create_tensor(src->ದತ್ತಾಂಶ, ACL_FLOAT, nb_f32, ne3d, nb3d, 3);
    acl_tensor_ptr acl_dst = ggml_cann_create_tensor(dst->ದತ್ತಾಂಶ, ACL_FLOAT, nb_f32, ne3d, nb3d, 3);

    ಆಯ್ಕೆ (ttype) {
        ಸ್ಥಿತಿ GGML_TRI_TYPE_ಅಧೋ:
            // Tril(-1): preserve row > col (strict lower), zero upper + diagonal.
            GGML_CANN_CALL_ACLNN_OP(ctx, Tril, acl_src.get(), (int64_t)-1, acl_dst.get());
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_TRI_TYPE_ಊರ್ಧ್ವ_DIAG:
            // Triu(0): preserve row <= col (upper + diagonal), zero strict lower.
            GGML_CANN_CALL_ACLNN_OP(ctx, Triu, acl_src.get(), (int64_t)0, acl_dst.get());
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_TRI_TYPE_ಊರ್ಧ್ವ:
            // Triu(1): preserve row < col (strict upper), zero lower + diagonal.
            GGML_CANN_CALL_ACLNN_OP(ctx, Triu, acl_src.get(), (int64_t)1, acl_dst.get());
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_TRI_TYPE_ಅಧೋ_DIAG:
            // Tril(0): preserve row >= col (lower + diagonal), zero strict upper.
            GGML_CANN_CALL_ACLNN_OP(ctx, Tril, acl_src.get(), (int64_t)0, acl_dst.get());
            ನಿಲ್ಲಿಸು;
        ಪೂರ್ವನಿಯೋಜಿತ:
            GGML_ABORT("unsupported tri type");
    }
}

ಶೂನ್ಯ ggml_cann_upsample_nearest2d(ggml_backend_cann_context & ctx, ggml_tensor * dst) {
    ggml_tensor *  src     = dst->src[0];
    acl_tensor_ptr acl_src = ggml_cann_create_tensor(src, ಶೂನ್ಯಸೂಚಕ, ಶೂನ್ಯಸೂಚಕ, 0, ACL_FORMAT_NCHW);
    acl_tensor_ptr acl_dst = ggml_cann_create_tensor(dst, ಶೂನ್ಯಸೂಚಕ, ಶೂನ್ಯಸೂಚಕ, 0, ACL_FORMAT_NCHW);

    std::vector<int64_t> output_ಗಾತ್ರ{ dst->ne[1], dst->ne[0] };
    acl_int_array_ptr    output_ಗಾತ್ರ_array = ggml_cann_create_int_array(output_ಗಾತ್ರ.ದತ್ತಾಂಶ(), 2);

    GGML_CANN_CALL_ACLNN_OP(ctx, UpsampleNearest2d, acl_src.get(), output_ಗಾತ್ರ_array.get(), acl_dst.get());
}

/**
 * @brief Pads a tensor with a specified value along each dimension.
 *
 * This function performs padding of the source tensor `acl_src` and stores the
 * result in the destination tensor `acl_dst`. The padding values for each
 * dimension are specified in the `paddings` array.
 *
 * @param ctx The context for the CANN backend operations.
 * @param acl_src The source tensor to be padded.
 * @param acl_dst The destination tensor where the padded result will be stored.
 * @param paddings An array specifying the padding values for each dimension.
 * The size of the array should be twice the number of dimensions of the tensor.
 * @param value The value to be used for padding. The default value is 0.0.
 */
ಸ್ಥಿರ ಶೂನ್ಯ aclnn_pad(ggml_backend_cann_context & ctx,
                      aclTensor *                 acl_src,
                      aclTensor *                 acl_dst,
                      int64_t *                   paddings,
                      ದಶಮಾಂಶ                       value = 0.0f) {
    acl_int_array_ptr acl_pad   = ggml_cann_create_int_array(paddings, GGML_MAX_DIMS * 2);
    acl_scalar_ptr    acl_ಮೌಲ್ಯ = ggml_cann_create_scalar(&value, acl_ದತ್ತಾಂಶ_Type::ACL_FLOAT);

    GGML_CANN_CALL_ACLNN_OP(ctx, ConstantPadNd, acl_src, acl_pad.get(), acl_ಮೌಲ್ಯ.get(), acl_dst);
}

ಶೂನ್ಯ ggml_cann_pad(ggml_backend_cann_context & ctx, ggml_tensor * dst) {
    ggml_tensor *  src     = dst->src[0];
    acl_tensor_ptr acl_src = ggml_cann_create_tensor(src);
    acl_tensor_ptr acl_dst = ggml_cann_create_tensor(dst);

    // padding: value in the array means how much distance will be padding.
    // the position of elements in the array means which dirction to padding,
    // each position means: [dim0.front, dim0.behind, dim1.front, dim1.behind,
    //                       dim2.front, dim2.behind, dim3.front, dim3.behind]
    ಸ್ಥಿರಾಂಕ int32_t lp0 = ggml_get_op_params_i32(dst, 0);
    ಸ್ಥಿರಾಂಕ int32_t rp0 = ggml_get_op_params_i32(dst, 1);
    ಸ್ಥಿರಾಂಕ int32_t lp1 = ggml_get_op_params_i32(dst, 2);
    ಸ್ಥಿರಾಂಕ int32_t rp1 = ggml_get_op_params_i32(dst, 3);
    ಸ್ಥಿರಾಂಕ int32_t lp2 = ggml_get_op_params_i32(dst, 4);
    ಸ್ಥಿರಾಂಕ int32_t rp2 = ggml_get_op_params_i32(dst, 5);
    ಸ್ಥಿರಾಂಕ int32_t lp3 = ggml_get_op_params_i32(dst, 6);
    ಸ್ಥಿರಾಂಕ int32_t rp3 = ggml_get_op_params_i32(dst, 7);

    int64_t paddings[] = { lp0, rp0, lp1, rp1, lp2, rp2, lp3, rp3 };
    aclnn_pad(ctx, acl_src.get(), acl_dst.get(), paddings);
}

/**
 * @brief Performs 2D average pooling on the input tensor and stores the result
 * in the destination tensor.
 *
 * This function performs average pooling on the source tensor and stores the
 * result in the destination tensor. The pooling parameters (kernel size,
 * strides, padding) are specified in the `op_params` of the destination tensor.
 *
 * @param ctx The context for the CANN backend operations.
 * @param dst The destination tensor where the result will be stored. The source
 * tensor is referenced by `dst->src[0]`.
 */
ಸ್ಥಿರ ಶೂನ್ಯ ggml_cann_avg_pool2d(ggml_backend_cann_context & ctx, ggml_tensor * dst) {
    ggml_tensor * src = dst->src[0];
    GGML_ASSERT(src->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    acl_tensor_ptr acl_src = ggml_cann_create_tensor(src, ಶೂನ್ಯಸೂಚಕ, ಶೂನ್ಯಸೂಚಕ, 0, ACL_FORMAT_NCHW);
    acl_tensor_ptr acl_dst = ggml_cann_create_tensor(dst, ಶೂನ್ಯಸೂಚಕ, ಶೂನ್ಯಸೂಚಕ, 0, ACL_FORMAT_NCHW);

    ಸ್ಥಿರಾಂಕ int32_t * opts = (ಸ್ಥಿರಾಂಕ int32_t *) dst->op_params;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ       k0   = opts[1];
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ       k1   = opts[2];
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ       s0   = opts[3];
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ       s1   = opts[4];
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ       p0   = opts[5];
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ       p1   = opts[6];

    std::vector<int64_t> kernel_dims      = { k1, k0 };
    std::vector<int64_t> stride_dims      = { s1, s0 };
    std::vector<int64_t> padding_avg_dims = { p1, p0 };  // (padH, padW)

    acl_int_array_ptr kernel_ಗಾತ್ರ  = ggml_cann_create_int_array(kernel_dims.ದತ್ತಾಂಶ(), 2);
    acl_int_array_ptr strides      = ggml_cann_create_int_array(stride_dims.ದತ್ತಾಂಶ(), 2);
    acl_int_array_ptr paddings_avg = ggml_cann_create_int_array(padding_avg_dims.ದತ್ತಾಂಶ(), 2);

    ತಾರ್ಕಿಕ    ceil_ಬಹುಲಕ         = ಸುಳ್ಳು;
    ತಾರ್ಕಿಕ    ಎಣಿಕೆ_include_pad = ನಿಜ;
    int64_t ಭಾಜಕ_override  = 0;
    int8_t  cube_math_type    = 0;
#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ ASCEND_310P
    cube_math_type = 1;
#ಆದರೆ_ಅಂತ್ಯ

    GGML_CANN_CALL_ACLNN_OP(ctx, AvgPool2d, acl_src.get(), kernel_ಗಾತ್ರ.get(), strides.get(), paddings_avg.get(),
                            ceil_ಬಹುಲಕ, ಎಣಿಕೆ_include_pad, ಭಾಜಕ_override, cube_math_type, acl_dst.get());
}

/**
 * @brief Performs 2D max pooling on the input tensor and stores the result in
 * the destination tensor.
 *
 * This function performs max pooling on the source tensor and stores the result
 * in the destination tensor. The pooling parameters (kernel size, strides,
 * padding) are specified in the `op_params` of the destination tensor.
 *
 * @param ctx The context for the CANN backend operations.
 * @param dst The destination tensor where the result will be stored. The source
 * tensor is referenced by `dst->src[0]`.
 */
ಸ್ಥಿರ ಶೂನ್ಯ ggml_cann_max_pool2d(ggml_backend_cann_context & ctx, ggml_tensor * dst) {
    ggml_tensor * src = dst->src[0];
    GGML_ASSERT(src->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    acl_tensor_ptr acl_src = ggml_cann_create_tensor(src, ಶೂನ್ಯಸೂಚಕ, ಶೂನ್ಯಸೂಚಕ, 0, ACL_FORMAT_NCHW);
    acl_tensor_ptr acl_dst = ggml_cann_create_tensor(dst, ಶೂನ್ಯಸೂಚಕ, ಶೂನ್ಯಸೂಚಕ, 0, ACL_FORMAT_NCHW);

    ಸ್ಥಿರಾಂಕ int32_t * opts = (ಸ್ಥಿರಾಂಕ int32_t *) dst->op_params;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ       k0   = opts[1];
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ       k1   = opts[2];
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ       s0   = opts[3];
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ       s1   = opts[4];
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ       p0   = opts[5];
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ       p1   = opts[6];

    int64_t ತಾತ್ಕಾಲಿಕ_ne[] = { src->ne[0] + p0 * 2, src->ne[1] + p1 * 2, src->ne[2], src->ne[3] };
    size_t  ತಾತ್ಕಾಲಿಕ_nb[GGML_MAX_DIMS];

    ತಾತ್ಕಾಲಿಕ_nb[0] = ggml_element_ಗಾತ್ರ(src);
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 1; i < GGML_MAX_DIMS; i++) {
        ತಾತ್ಕಾಲಿಕ_nb[i] = ತಾತ್ಕಾಲಿಕ_nb[i - 1] * ತಾತ್ಕಾಲಿಕ_ne[i - 1];
    }

    ggml_cann_pool_alloc ತಾತ್ಕಾಲಿಕ_ಬಫರ್_allocator(ctx.pool(), ggml_nbytes(src) + p0 * 2 + p1 * 2 * src->nb[1]);
    ಶೂನ್ಯ *               ಬಫರ್ = ತಾತ್ಕಾಲಿಕ_ಬಫರ್_allocator.get();
    acl_tensor_ptr tmp_tensor   = ggml_cann_create_tensor(ಬಫರ್, ACL_FLOAT, ggml_element_ಗಾತ್ರ(src), ತಾತ್ಕಾಲಿಕ_ne, ತಾತ್ಕಾಲಿಕ_nb,
                                                          GGML_MAX_DIMS, ACL_FORMAT_NCHW);

    // pad: see padding in ggml_cann_pad()
    int64_t paddings[] = { p0, p0, p1, p1, 0, 0, 0, 0 };
    ದಶಮಾಂಶ   value      = -FLT_MAX;
    aclnn_pad(ctx, acl_src.get(), tmp_tensor.get(), paddings, value);

    // max_pool
    std::vector<int64_t> kernel_dims      = { k1, k0 };
    std::vector<int64_t> stride_dims      = { s1, s0 };
    // padding_max_dims: [dim0_start, dim0_end, dim1_start, dim1_end]
    std::vector<int64_t> padding_max_dims = { 0, 0, 0, 0 };
    std::vector<int64_t> dilation_ಗಾತ್ರ    = { 1, 1 };
    acl_int_array_ptr    kernel_ಗಾತ್ರ      = ggml_cann_create_int_array(kernel_dims.ದತ್ತಾಂಶ(), 2);
    acl_int_array_ptr    strides          = ggml_cann_create_int_array(stride_dims.ದತ್ತಾಂಶ(), 2);
    acl_int_array_ptr    paddings_max     = ggml_cann_create_int_array(padding_max_dims.ದತ್ತಾಂಶ(), 4);
    acl_int_array_ptr    dilations        = ggml_cann_create_int_array(dilation_ಗಾತ್ರ.ದತ್ತಾಂಶ(), 2);

    ತಾರ್ಕಿಕ    ceil_ಬಹುಲಕ = ಸುಳ್ಳು;
    int64_t auto_pads = 0;
    GGML_CANN_CALL_ACLNN_OP(ctx, MaxPool, tmp_tensor.get(), kernel_ಗಾತ್ರ.get(), strides.get(), auto_pads,
                            paddings_max.get(), dilations.get(), ceil_ಬಹುಲಕ, acl_dst.get());
}

ಶೂನ್ಯ ggml_cann_pool2d(ggml_backend_cann_context & ctx, ggml_tensor * dst) {
    ಸ್ಥಿರಾಂಕ int32_t *   opts = (ಸ್ಥಿರಾಂಕ int32_t *) dst->op_params;
    ಎಣಿಕೆ ggml_op_pool op   = static_cast<ggml_op_pool>(opts[0]);
    ಆಯ್ಕೆ (op) {
        ಸ್ಥಿತಿ GGML_OP_POOL_AVG:
            ggml_cann_avg_pool2d(ctx, dst);
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_POOL_MAX:
            ggml_cann_max_pool2d(ctx, dst);
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_POOL_ಎಣಿಕೆ:
            GGML_ABORT("fatal error");
            ನಿಲ್ಲಿಸು;
    }
}

/**
 * @brief Copies data from the source tensor to the destination tensor.
 *
 * This function copies data from the source tensor `acl_src` to the destination
 * tensor `acl_dst`.
 *
 * @param ctx The context for the CANN backend operations.
 * @param acl_src The source tensor from which data will be copied.
 * @param acl_dst The destination tensor where the data will be copied to.
 */
ಸ್ಥಿರ ಶೂನ್ಯ cann_copy(ggml_backend_cann_context & ctx, aclTensor * acl_src, aclTensor * acl_dst) {
    GGML_CANN_CALL_ACLNN_OP(ctx, InplaceCopy, acl_dst, acl_src);
}

ಶೂನ್ಯ ggml_cann_dup(ggml_backend_cann_context & ctx, ggml_tensor * dst) {
    ggml_tensor * src0 = dst->src[0];

    ಆದರೆ (ggml_are_same_shape(src0, dst)) {
        acl_tensor_ptr acl_src = ggml_cann_create_tensor(src0);
        acl_tensor_ptr acl_dst = ggml_cann_create_tensor(dst);
        ಆದರೆ (dst->type == src0->type) {
            cann_copy(ctx, acl_src.get(), acl_dst.get());
        } ಇಲ್ಲದಿದ್ದರೆ {
            aclnn_cast(ctx, acl_src.get(), acl_dst.get(), ggml_cann_type_mapping(dst->type));
        }
    } ಇಲ್ಲದಿದ್ದರೆ {
        ಶೂನ್ಯ *               src_trans_ಬಫರ್ = src0->ದತ್ತಾಂಶ;
        ggml_cann_pool_alloc src_ಬಫರ್_allocator;
        ಆದರೆ (!ggml_is_contiguous(src0)) {
            acl_tensor_ptr acl_src = ggml_cann_create_tensor(src0);
            src_ಬಫರ್_allocator.alloc(ctx.pool(), ggml_nelements(src0) * ggml_type_ಗಾತ್ರ(src0->type));
            src_trans_ಬಫರ್ = src_ಬಫರ್_allocator.get();
            size_t src_trans_nb[GGML_MAX_DIMS];
            src_trans_nb[0] = ggml_type_ಗಾತ್ರ(src0->type);
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 1; i < GGML_MAX_DIMS; i++) {
                src_trans_nb[i] = src_trans_nb[i - 1] * src0->ne[i - 1];
            }
            acl_tensor_ptr src_trans_tensor =
                ggml_cann_create_tensor(src_trans_ಬಫರ್, ggml_cann_type_mapping(src0->type),
                                        ggml_type_ಗಾತ್ರ(src0->type), src0->ne, src_trans_nb, GGML_MAX_DIMS);
            cann_copy(ctx, acl_src.get(), src_trans_tensor.get());
        }

        size_t src_reshape_nb[GGML_MAX_DIMS];
        src_reshape_nb[0] = ggml_type_ಗಾತ್ರ(src0->type);
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 1; i < GGML_MAX_DIMS; i++) {
            src_reshape_nb[i] = src_reshape_nb[i - 1] * dst->ne[i - 1];
        }

        acl_tensor_ptr trans_acl_src =
            ggml_cann_create_tensor(src_trans_ಬಫರ್, ggml_cann_type_mapping(src0->type), ggml_type_ಗಾತ್ರ(src0->type),
                                    dst->ne, src_reshape_nb, GGML_MAX_DIMS, ACL_FORMAT_ND);
        acl_tensor_ptr acl_dst = ggml_cann_create_tensor(dst);

        ಆದರೆ (dst->type == src0->type) {
            cann_copy(ctx, trans_acl_src.get(), acl_dst.get());
        } ಇಲ್ಲದಿದ್ದರೆ {
            aclnn_cast(ctx, trans_acl_src.get(), acl_dst.get(), ggml_cann_type_mapping(dst->type));
        }
    }
}

/**
 * @brief Creates an ACL tensor initialized with zeros using a provided buffer.
 *
 * This function initializes a tensor with zeros using the specified buffer and
 * tensor parameters.
 *
 * @param ctx The context for the CANN backend operations.
 * @param buffer The buffer to be used for the tensor data.
 * @param n_bytes The size of the buffer in bytes.
 * @param ne An array specifying the extents (sizes) of each dimension of the
 * tensor.
 * @param dims The number of dimensions of the tensor.
 * @param type The data type of the tensor.
 * @param type_size The size of each element in the tensor data type.
 * @return A tensor smart pointer initialized with zeros.
 */
ಸ್ಥಿರ acl_tensor_ptr aclnn_zero(ggml_backend_cann_context & ctx,
                                 ಶೂನ್ಯ *                      ಬಫರ್,
                                 size_t                      n_ಬೈಟುಗಳು,
                                 int64_t *                   ne,
                                 int64_t                     dims,
                                 acl_ದತ್ತಾಂಶ_Type                 type,
                                 size_t                      type_ಗಾತ್ರ) {
    size_t nb[GGML_MAX_DIMS];
    nb[0] = type_ಗಾತ್ರ;
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 1; i < dims; i++) {
        nb[i] = nb[i - 1] * ne[i - 1];
    }

    acl_tensor_ptr zero = ggml_cann_create_tensor(ಬಫರ್, type, type_ಗಾತ್ರ, ne, nb, dims);
    GGML_CANN_CALL_ACLNN_OP(ctx, InplaceZero, zero.get());
    ಹಿಂತಿರುಗಿ zero;
    GGML_ಬಳಸದ(n_ಬೈಟುಗಳು);
}

/**
 * @brief Creates an ACL tensor initialized with value using a provided buffer.
 *
 * This function initializes a tensor with value using the specified buffer and
 * tensor parameters.
 *
 * @param ctx The context for the CANN backend operations.
 * @param buffer The buffer to be used for the tensor data.
 * @param n_bytes The size of the buffer in bytes.
 * @param ne An array specifying the extents (sizes) of each dimension of the
 * tensor.
 * @param dims The number of dimensions of the tensor.
 * @param type The data type of the tensor.
 * @param type_size The size of each element in the tensor data type.
 * @param value The value to be used for initializing the tensor (default
 * is 1.0).
 * @return A tensor smart pointer initialized with value.
 */
ಸ್ಥಿರ acl_tensor_ptr aclnn_ಮೌಲ್ಯಗಳು(ggml_backend_cann_context & ctx,
                                   ಶೂನ್ಯ *                      ಬಫರ್,
                                   size_t                      n_ಬೈಟುಗಳು,
                                   int64_t *                   ne,
                                   int64_t                     dims,
                                   acl_ದತ್ತಾಂಶ_Type                 type,
                                   size_t                      type_ಗಾತ್ರ,
                                   ದಶಮಾಂಶ                       value = 1.0f) {
    acl_tensor_ptr acl_tensor = aclnn_zero(ctx, ಬಫರ್, n_ಬೈಟುಗಳು, ne, dims, type, type_ಗಾತ್ರ);
    ದಶಮಾಂಶ          alpha_host = 1.0f;
    acl_scalar_ptr alpha      = ggml_cann_create_scalar(&alpha_host, acl_ದತ್ತಾಂಶ_Type::ACL_FLOAT);
    acl_scalar_ptr other      = ggml_cann_create_scalar(&value, acl_ದತ್ತಾಂಶ_Type::ACL_FLOAT);
    GGML_CANN_CALL_ACLNN_OP(ctx, InplaceAdds, acl_tensor.get(), other.get(), alpha.get());
    ಹಿಂತಿರುಗಿ acl_tensor;
}

/**
 * @brief Fills a tensor with a scalar value.
 *
 * This function fills the destination tensor `acl_dst` with the scalar value
 * `scalar`.
 *
 * @param ctx The context for the CANN backend operations.
 * @param scalar The scalar value used to fill the tensor.
 * @param acl_dst The destination tensor to be filled with the scalar value.
 */
ಸ್ಥಿರ ಶೂನ್ಯ aclnn_fill_scalar(ggml_backend_cann_context & ctx, ದಶಮಾಂಶ scalar, aclTensor * acl_dst) {
    acl_scalar_ptr acl_scalar = ggml_cann_create_scalar(&scalar, acl_ದತ್ತಾಂಶ_Type::ACL_FLOAT);
    GGML_CANN_CALL_ACLNN_OP(ctx, InplaceFillScalar, acl_dst, acl_scalar.get());
}

/**
 * @brief Get or expand a cached tensor filled with a scalar value.
 *
 * This function manages cached device memory for tensors. If the current
 * cache size is insufficient for the requested tensor shape, the old memory will
 * be released and new memory will be allocated. The allocated buffer is
 * initialized  with the given scalar value using CANN operations.
 * Finally, an aclTensor object is created from the cached memory and returned.
 *
 * @param ctx           The CANN backend context that manages device memory.
 * @param buffer        A pointer to the cached device buffer (will be allocated
 *                      or reallocated if necessary).
 * @param cache_element The current number of cached elements. This will be
 *                      updated when the cache is expanded.
 * @param ne            The tensor shape array (number of elements in each dimension).
 * @param nb            The stride size for each dimension.
 * @param dtype         Data type of cached tensor.
 * @param dims          The number of tensor dimensions.
 * @param value         The scalar value used to fill the tensor (supports zero
 *                      initialization via memset or arbitrary values via fill_scalar).
 * @return              A tensor smart pointer created from the cached buffer.
 */
ಸ್ಥಿರ acl_tensor_ptr get_cache_acl_tensor(ggml_backend_cann_context & ctx,
                                           ಶೂನ್ಯ **                     ಬಫರ್,
                                           int64_t &                   cache_element,
                                           int64_t *                   ne,
                                           size_t *                    nb,
                                           ggml_type                   dtype,
                                           int64_t                     dims,
                                           ದಶಮಾಂಶ                       value) {
    // Calculate total number of elements
    int64_t n_element = 1;
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < dims; i++) {
        n_element *= ne[i];
    }
    size_t size = n_element * ggml_type_ಗಾತ್ರ(dtype);

    // Allocate or expand cache if needed
    ಆದರೆ (cache_element < n_element) {
        ಆದರೆ (*ಬಫರ್ != ಶೂನ್ಯಸೂಚಕ) {
            aclrtFree(*ಬಫರ್);
            *ಬಫರ್ = ಶೂನ್ಯಸೂಚಕ;
        }

        ACL_CHECK(aclrtMalloc(ಬಫರ್, size, ACL_MEM_MALLOC_HUGE_ಮೊದಲ));
        cache_element = n_element;

        // Initialize cache
        int64_t        pool_ne[1] = { n_element };
        size_t         pool_nb[1] = { ggml_type_ಗಾತ್ರ(dtype) };
        acl_tensor_ptr acl_ಮೌಲ್ಯ =
            ggml_cann_create_tensor(*ಬಫರ್, ggml_cann_type_mapping(dtype), ggml_type_ಗಾತ್ರ(dtype), pool_ne, pool_nb, 1);
        aclnn_fill_scalar(ctx, value, acl_ಮೌಲ್ಯ.get());
    }

    ಹಿಂತಿರುಗಿ ggml_cann_create_tensor(*ಬಫರ್, ggml_cann_type_mapping(dtype), ggml_type_ಗಾತ್ರ(dtype), ne, nb, dims);
}

ಶೂನ್ಯ ggml_cann_rms_norm(ggml_backend_cann_context & ctx, ggml_tensor * dst) {
    ggml_tensor * src = dst->src[0];

    acl_tensor_ptr acl_src = ggml_cann_create_tensor(src);
    acl_tensor_ptr acl_dst = ggml_cann_create_tensor(dst);

    ದಶಮಾಂಶ eps;
    ಸ್ಮರಣೆ_ನಕಲು(&eps, dst->op_params, ಗಾತ್ರ(ದಶಮಾಂಶ));

    // build gamma.
    size_t acl_gamma_nb[GGML_MAX_DIMS];
    // gamma's type is the same with dst.
    acl_gamma_nb[0] = ggml_type_ಗಾತ್ರ(dst->type);
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 1; i < GGML_MAX_DIMS; i++) {
        acl_gamma_nb[i] = acl_gamma_nb[i - 1] * src->ne[i - 1];
    }
    acl_tensor_ptr acl_gamma = get_cache_acl_tensor(
        ctx, &ctx.rms_norm_one_tensor_cache.cache, ctx.rms_norm_one_tensor_cache.size, src->ne, acl_gamma_nb, dst->type,
        1,    // dims
        1.0f  //ಮೌಲ್ಯ
    );

    // build rstd.
    int64_t acl_rstd_ne[] = { src->ne[1], src->ne[2], src->ne[3] };
    size_t  acl_rstd_nb[GGML_MAX_DIMS - 1];
    // rstd will always be F32.
    acl_rstd_nb[0] = ಗಾತ್ರ(ದಶಮಾಂಶ);
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 1; i < GGML_MAX_DIMS - 1; i++) {
        acl_rstd_nb[i] = acl_rstd_nb[i - 1] * acl_rstd_ne[i - 1];
    }
    acl_tensor_ptr acl_rstd =
        get_cache_acl_tensor(ctx, &ctx.rms_norm_zero_tensor_cache.cache, ctx.rms_norm_zero_tensor_cache.size,
                             acl_rstd_ne, acl_rstd_nb, GGML_TYPE_F32, GGML_MAX_DIMS - 1,
                             0.0f  //ಮೌಲ್ಯ
        );

    GGML_CANN_CALL_ACLNN_OP(ctx, RmsNorm, acl_src.get(), acl_gamma.get(), eps, acl_dst.get(), acl_rstd.get());
}

// TODO: performace is low.
ಶೂನ್ಯ ggml_cann_diag_ಮುಖವಾಡ(ggml_backend_cann_context & ctx, ggml_tensor * dst, ದಶಮಾಂಶ value) {
    ggml_tensor * src = dst->src[0];

    acl_tensor_ptr acl_src = ggml_cann_create_tensor(src);
    acl_tensor_ptr acl_dst = ggml_cann_create_tensor(dst);

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ n_past = ((int32_t *) dst->op_params)[0];

    ggml_cann_pool_alloc one_tensor_allocator(ctx.pool(), ggml_nbytes(src));
    ಶೂನ್ಯ *               ಬಫರ್ = one_tensor_allocator.get();

    acl_tensor_ptr ಮುಖವಾಡ_tensor = ggml_cann_create_tensor(ಬಫರ್, ggml_cann_type_mapping(src->type),
                                                         ggml_type_ಗಾತ್ರ(src->type), src->ne, src->nb, GGML_MAX_DIMS);

    aclnn_fill_scalar(ctx, value, ಮುಖವಾಡ_tensor.get());

    ದಶಮಾಂಶ          alpha_ಮೌಲ್ಯ = 1.0f;
    acl_scalar_ptr alpha      = ggml_cann_create_scalar(&alpha_ಮೌಲ್ಯ, acl_ದತ್ತಾಂಶ_Type::ACL_FLOAT);

    GGML_CANN_CALL_ACLNN_OP(ctx, InplaceTriu, ಮುಖವಾಡ_tensor.get(), n_past + 1);
    GGML_CANN_CALL_ACLNN_OP(ctx, Tril, acl_src.get(), n_past + 1, acl_dst.get());
    GGML_CANN_CALL_ACLNN_OP(ctx, Inplace_ಕೂಡಿಸು, acl_dst.get(), ಮುಖವಾಡ_tensor.get(), alpha.get());
}

/**
 * @brief Permutes the dimensions of a tensor according to a specified order.
 *
 * This function permutes the dimensions of the source tensor `acl_src`
 * according to the order specified in the `new_dim` array and stores the result
 * in the destination tensor `acl_dst`.
 *
 * @param ctx The context for the CANN backend operations.
 * @param acl_src The source tensor whose dimensions will be permuted.
 * @param acl_dst The destination tensor where the permuted result will be
 * stored.
 * @param new_dim An array specifying the new order of dimensions for the
 * tensor.
 * @param dims The number of dimensions in the tensor.
 */
ಸ್ಥಿರ ಶೂನ್ಯ aclnn_ಕ್ರಮಬದಲಾವಣೆ(ggml_backend_cann_context & ctx,
                          aclTensor *                 acl_src,
                          aclTensor *                 acl_dst,
                          int64_t *                   new_dim,
                          uint64_t                    dims) {
    acl_int_array_ptr acl_dims = ggml_cann_create_int_array(new_dim, dims);
    GGML_CANN_CALL_ACLNN_OP(ctx, ಕ್ರಮಬದಲಾವಣೆ, acl_src, acl_dims.get(), acl_dst);
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_cann_im2col_2d_post_process(ggml_backend_cann_context & ctx,
                                             ggml_tensor *               dst,
                                             ggml_tensor *               src1,
                                             aclTensor *                 tmp_cast_tensor,
                                             aclTensor *                 tmp_im2col_tensor) {
    // Permute: [N, IC * KH * KW, OW * OH] -> [N, OW * OH, IC * KH * KW]
    int64_t        dst_ne[] = { dst->ne[0], dst->ne[1] * dst->ne[2], dst->ne[3] };
    size_t         dst_nb[] = { dst->nb[0], dst->nb[1], dst->nb[3] };
    acl_tensor_ptr acl_dst  = ggml_cann_create_tensor(dst, dst_ne, dst_nb, GGML_MAX_DIMS - 1);

    int64_t ಕ್ರಮಬದಲಾವಣೆ_dim[] = { 0, 2, 1 };
    ಆದರೆ (src1->type != dst->type) {
        aclnn_ಕ್ರಮಬದಲಾವಣೆ(ctx, tmp_cast_tensor, acl_dst.get(), ಕ್ರಮಬದಲಾವಣೆ_dim, 3);
    } ಇಲ್ಲದಿದ್ದರೆ {
        aclnn_ಕ್ರಮಬದಲಾವಣೆ(ctx, tmp_im2col_tensor, acl_dst.get(), ಕ್ರಮಬದಲಾವಣೆ_dim, 3);
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_cann_im2col_1d_post_process(ggml_backend_cann_context &  ctx,
                                             ggml_tensor *                dst,
                                             ggml_tensor *                src1,
                                             aclTensor *                  tmp_cast_tensor,
                                             aclTensor *                  tmp_im2col_tensor,
                                             ಸ್ಥಿರಾಂಕ std::vector<int64_t> & im2col_op_params) {
    // get params
    ಸ್ಥಿರಾಂಕ int64_t KH             = im2col_op_params[0];
    ಸ್ಥಿರಾಂಕ int64_t KW             = im2col_op_params[1];
    ಸ್ಥಿರಾಂಕ int64_t IW             = im2col_op_params[2];
    ಸ್ಥಿರಾಂಕ int64_t IC             = im2col_op_params[3];
    ಸ್ಥಿರಾಂಕ int64_t N              = im2col_op_params[4];
    ಸ್ಥಿರಾಂಕ int64_t OH             = im2col_op_params[5];
    ಸ್ಥಿರಾಂಕ int64_t OW             = im2col_op_params[6];
    ಸ್ಥಿರಾಂಕ int64_t s0             = im2col_op_params[7];
    ಸ್ಥಿರಾಂಕ int64_t p0             = im2col_op_params[8];
    ಸ್ಥಿರಾಂಕ int64_t d0             = im2col_op_params[9];
    ಸ್ಥಿರಾಂಕ int64_t n_ಬೈಟುಗಳು_ಗುಣಕ = im2col_op_params[10];

    // Permute: [N, IC * KH * KW, OW * OH] ->
    // [N, OW * OH * n_bytes_factor, IC * KH * KW]
    ggml_cann_pool_alloc tmp_ಕ್ರಮಬದಲಾವಣೆ_allocator(ctx.pool());
    tmp_ಕ್ರಮಬದಲಾವಣೆ_allocator.alloc(ggml_nbytes(dst) * n_ಬೈಟುಗಳು_ಗುಣಕ);
    ಶೂನ್ಯ * tmp_ಕ್ರಮಬದಲಾವಣೆ_ಬಫರ್ = tmp_ಕ್ರಮಬದಲಾವಣೆ_allocator.get();

    int64_t tmp_ಕ್ರಮಬದಲಾವಣೆ_ne[] = { IC * KH * KW, OW * OH * n_ಬೈಟುಗಳು_ಗುಣಕ, N };
    size_t  tmp_ಕ್ರಮಬದಲಾವಣೆ_nb[GGML_MAX_DIMS - 1];
    tmp_ಕ್ರಮಬದಲಾವಣೆ_nb[0] = ggml_type_ಗಾತ್ರ(dst->type);
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 1; i < GGML_MAX_DIMS - 1; i++) {
        tmp_ಕ್ರಮಬದಲಾವಣೆ_nb[i] = tmp_ಕ್ರಮಬದಲಾವಣೆ_nb[i - 1] * tmp_ಕ್ರಮಬದಲಾವಣೆ_ne[i - 1];
    }

    acl_tensor_ptr tmp_ಕ್ರಮಬದಲಾವಣೆ_tensor =
        ggml_cann_create_tensor(tmp_ಕ್ರಮಬದಲಾವಣೆ_ಬಫರ್, ggml_cann_type_mapping(dst->type), ggml_type_ಗಾತ್ರ(dst->type),
                                tmp_ಕ್ರಮಬದಲಾವಣೆ_ne, tmp_ಕ್ರಮಬದಲಾವಣೆ_nb, GGML_MAX_DIMS - 1, ACL_FORMAT_ND);

    int64_t ಕ್ರಮಬದಲಾವಣೆ_dim[] = { 0, 2, 1 };
    ಆದರೆ (src1->type != dst->type) {
        aclnn_ಕ್ರಮಬದಲಾವಣೆ(ctx, tmp_cast_tensor, tmp_ಕ್ರಮಬದಲಾವಣೆ_tensor.get(), ಕ್ರಮಬದಲಾವಣೆ_dim, 3);
    } ಇಲ್ಲದಿದ್ದರೆ {
        aclnn_ಕ್ರಮಬದಲಾವಣೆ(ctx, tmp_im2col_tensor, tmp_ಕ್ರಮಬದಲಾವಣೆ_tensor.get(), ಕ್ರಮಬದಲಾವಣೆ_dim, 3);
    }

    // number of times the kernel moves in W dimension
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ n_ಹೆಜ್ಜೆ_w = (IW + 2 * p0 - d0 * (KW - 1) - 1) / s0 + 1;
    size_t    ಸ್ಥಾನಾಂತರ;
    ಶೂನ್ಯ *    cur_dst_ಬಫರ್ = dst->ದತ್ತಾಂಶ, *cur_ಕ್ರಮಬದಲಾವಣೆ_ಬಫರ್ = tmp_ಕ್ರಮಬದಲಾವಣೆ_ಬಫರ್;

    // memory copy with offset to restore 1D im2col from 2d
    ಆದರೆ (IC > 1) {
        ಸ್ಥಾನಾಂತರ          = IC * KH * KW * n_ಹೆಜ್ಜೆ_w * ggml_type_ಗಾತ್ರ(dst->type);
        size_t cpy_ಗಾತ್ರ = KH * KW * ggml_type_ಗಾತ್ರ(dst->type);

        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ c = 0; c < IC; c++) {
            cur_ಕ್ರಮಬದಲಾವಣೆ_ಬಫರ್ = (ಅಕ್ಷರ *) tmp_ಕ್ರಮಬದಲಾವಣೆ_ಬಫರ್ + ಸ್ಥಾನಾಂತರ + KH * KW * c * ggml_type_ಗಾತ್ರ(dst->type);
            cur_dst_ಬಫರ್     = (ಅಕ್ಷರ *) dst->ದತ್ತಾಂಶ + c * KH * KW * n_ಹೆಜ್ಜೆ_w * ggml_type_ಗಾತ್ರ(dst->type);

            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < n_ಹೆಜ್ಜೆ_w; i++) {
                ACL_CHECK(aclrtMemcpyAsync(cur_dst_ಬಫರ್, cpy_ಗಾತ್ರ, cur_ಕ್ರಮಬದಲಾವಣೆ_ಬಫರ್, cpy_ಗಾತ್ರ,
                                           ACL_MEMCPY_DEVICE_ಗಮ್ಯ_DEVICE, ctx.stream()));
                cur_dst_ಬಫರ್     = (ಅಕ್ಷರ *) cur_dst_ಬಫರ್ + KH * KW * ggml_type_ಗಾತ್ರ(dst->type);
                cur_ಕ್ರಮಬದಲಾವಣೆ_ಬಫರ್ = (ಅಕ್ಷರ *) cur_ಕ್ರಮಬದಲಾವಣೆ_ಬಫರ್ + KH * KW * IC * ggml_type_ಗಾತ್ರ(dst->type);
            }
        }
    } ಇಲ್ಲದಿದ್ದರೆ {
        ಸ್ಥಾನಾಂತರ = KH * KW * n_ಹೆಜ್ಜೆ_w * ggml_type_ಗಾತ್ರ(dst->type);  // equal to ggml_nbytes(dst)
        ACL_CHECK(aclrtMemcpyAsync(dst->ದತ್ತಾಂಶ, ಸ್ಥಾನಾಂತರ, (ಅಕ್ಷರ *) tmp_ಕ್ರಮಬದಲಾವಣೆ_ಬಫರ್ + ಸ್ಥಾನಾಂತರ, ಸ್ಥಾನಾಂತರ,
                                   ACL_MEMCPY_DEVICE_ಗಮ್ಯ_DEVICE, ctx.stream()));
    }
}

ಶೂನ್ಯ ggml_cann_im2col(ggml_backend_cann_context & ctx, ggml_tensor * dst) {
    ggml_tensor * src0 = dst->src[0];  // kernel
    ggml_tensor * src1 = dst->src[1];  //ಒಳಹರಿವು

    GGML_TENSOR_ದ್ವಿಮಾನ_OP_LOCALS;

    // aclnnIm2col only works on 2D. set s1, p1, d1 to 1 to perform 2D
    // im2col and do post-processing to restore it to 1D.
    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ    is_2D = ((ಸ್ಥಿರಾಂಕ int32_t *) (dst->op_params))[6] == 1;
    ಸ್ಥಿರಾಂಕ int32_t s0    = ((ಸ್ಥಿರಾಂಕ int32_t *) (dst->op_params))[0];
    ಸ್ಥಿರಾಂಕ int32_t s1    = is_2D ? ((ಸ್ಥಿರಾಂಕ int32_t *) (dst->op_params))[1] : 1;
    ಸ್ಥಿರಾಂಕ int32_t p0    = ((ಸ್ಥಿರಾಂಕ int32_t *) (dst->op_params))[2];
    ಸ್ಥಿರಾಂಕ int32_t p1    = is_2D ? ((ಸ್ಥಿರಾಂಕ int32_t *) (dst->op_params))[3] : 1;
    ಸ್ಥಿರಾಂಕ int32_t d0    = ((ಸ್ಥಿರಾಂಕ int32_t *) (dst->op_params))[4];
    ಸ್ಥಿರಾಂಕ int32_t d1    = is_2D ? ((ಸ್ಥಿರಾಂಕ int32_t *) (dst->op_params))[5] : 1;

    ಸ್ಥಿರಾಂಕ int64_t N  = ne13;
    ಸ್ಥಿರಾಂಕ int64_t IC = ne12;
    ಸ್ಥಿರಾಂಕ int64_t KH = ne01;
    ಸ್ಥಿರಾಂಕ int64_t KW = ne00;
    ಸ್ಥಿರಾಂಕ int64_t IW = ne10;

    ಸ್ಥಿರಾಂಕ int64_t OH = is_2D ? ne2 : 1;
    ಸ್ಥಿರಾಂಕ int64_t OW = ne1;

    // memory allocated increased to 3x when is_2D == false
    ಸ್ಥಿರಾಂಕ int64_t n_ಬೈಟುಗಳು_ಗುಣಕ = is_2D ? 1 : 3;

    // im2col: [N,C,H,W] -> [N, IC * KH * KW, OW * OH * n_bytes_factor]
    acl_tensor_ptr acl_src1        = ggml_cann_create_tensor(src1);
    int64_t        tmp_im2col_ne[] = { OW * OH * n_ಬೈಟುಗಳು_ಗುಣಕ, IC * KH * KW, N };
    size_t         tmp_im2col_nb[GGML_MAX_DIMS - 1];

    tmp_im2col_nb[0] = ggml_type_ಗಾತ್ರ(src1->type);
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 1; i < GGML_MAX_DIMS - 1; i++) {
        tmp_im2col_nb[i] = tmp_im2col_nb[i - 1] * tmp_im2col_ne[i - 1];
    }

    // Calculate im2col.
    // If dst is f16, tmp_buffer is f32, we need alloc src.typesize *
    // dst.elemcount.
    ggml_cann_pool_alloc im2col_allocator(ctx.pool(), ggml_nelements(dst) * ggml_element_ಗಾತ್ರ(src1) * n_ಬೈಟುಗಳು_ಗುಣಕ);
    ಶೂನ್ಯ *               tmp_im2col_ಬಫರ್ = im2col_allocator.get();

    acl_tensor_ptr tmp_im2col_tensor =
        ggml_cann_create_tensor(tmp_im2col_ಬಫರ್, ggml_cann_type_mapping(src1->type), ggml_type_ಗಾತ್ರ(src1->type),
                                tmp_im2col_ne, tmp_im2col_nb, GGML_MAX_DIMS - 1, ACL_FORMAT_ND);

    std::vector<int64_t> kernel_dims   = { KH, KW };
    std::vector<int64_t> dilation_ಗಾತ್ರ = { d1, d0 };
    std::vector<int64_t> padding_dims  = { p1, p0 };
    std::vector<int64_t> stride_dims   = { s1, s0 };
    acl_int_array_ptr    kernel_ಗಾತ್ರ   = ggml_cann_create_int_array(kernel_dims.ದತ್ತಾಂಶ(), 2);
    acl_int_array_ptr    dilations     = ggml_cann_create_int_array(dilation_ಗಾತ್ರ.ದತ್ತಾಂಶ(), 2);
    acl_int_array_ptr    paddings      = ggml_cann_create_int_array(padding_dims.ದತ್ತಾಂಶ(), 2);
    acl_int_array_ptr    strides       = ggml_cann_create_int_array(stride_dims.ದತ್ತಾಂಶ(), 2);
    GGML_CANN_CALL_ACLNN_OP(ctx, Im2col, acl_src1.get(), kernel_ಗಾತ್ರ.get(), dilations.get(), paddings.get(),
                            strides.get(), tmp_im2col_tensor.get());

    // Cast if dst is f16.
    acl_tensor_ptr       tmp_cast_tensor;
    ggml_cann_pool_alloc tmp_cast_allocator(ctx.pool());
    ಶೂನ್ಯ *               tmp_cast_ಬಫರ್ = ಶೂನ್ಯಸೂಚಕ;
    ಆದರೆ (src1->type != dst->type) {
        tmp_cast_allocator.alloc(ggml_nbytes(dst) * n_ಬೈಟುಗಳು_ಗುಣಕ);
        tmp_cast_ಬಫರ್ = tmp_cast_allocator.get();
        size_t ತಾತ್ಕಾಲಿಕ_cast_nb[GGML_MAX_DIMS - 1];
        ತಾತ್ಕಾಲಿಕ_cast_nb[0] = ggml_type_ಗಾತ್ರ(dst->type);
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 1; i < GGML_MAX_DIMS - 1; i++) {
            ತಾತ್ಕಾಲಿಕ_cast_nb[i] = ತಾತ್ಕಾಲಿಕ_cast_nb[i - 1] * tmp_im2col_ne[i - 1];
        }

        tmp_cast_tensor =
            ggml_cann_create_tensor(tmp_cast_ಬಫರ್, ggml_cann_type_mapping(dst->type), ggml_type_ಗಾತ್ರ(dst->type),
                                    tmp_im2col_ne, ತಾತ್ಕಾಲಿಕ_cast_nb, GGML_MAX_DIMS - 1, ACL_FORMAT_ND);
        aclnn_cast(ctx, tmp_im2col_tensor.get(), tmp_cast_tensor.get(), ggml_cann_type_mapping(dst->type));
    }

    // post-processing
    ಆದರೆ (is_2D) {
        ggml_cann_im2col_2d_post_process(ctx, dst, src1, tmp_cast_tensor.get(), tmp_im2col_tensor.get());
    } ಇಲ್ಲದಿದ್ದರೆ {
        std::vector<int64_t> im2col_op_params = { KH, KW, IW, IC, N, OH, OW, s0, p0, d0, n_ಬೈಟುಗಳು_ಗುಣಕ };
        ggml_cann_im2col_1d_post_process(ctx, dst, src1, tmp_cast_tensor.get(), tmp_im2col_tensor.get(),
                                         im2col_op_params);
    }
}

/**
 * @brief Applies element-wise exponential function to the elements of a tensor.
 *
 * This function computes the exponential of each element in the source tensor
 * `acl_src` and stores the result back into the same tensor.
 * The operation is defined as:
 * \f[
 *     \text {acl_src }_i=e^{acl\_src_i}
 * \f]
 *
 * @param ctx The context for the CANN backend operations.
 * @param acl_src The tensor on which the exponential function will be applied.
 */
ಸ್ಥಿರ ಶೂನ್ಯ aclnn_exp(ggml_backend_cann_context & ctx, aclTensor * acl_src) {
    GGML_CANN_CALL_ACLNN_OP(ctx, InplaceExp, acl_src);
}

ಶೂನ್ಯ aclnn_cos(ggml_backend_cann_context & ctx, aclTensor * acl_src, aclTensor * acl_dst) {
    ಆದರೆ (acl_dst == ಶೂನ್ಯಸೂಚಕ) {
        GGML_CANN_CALL_ACLNN_OP(ctx, InplaceCos, acl_src);
    } ಇಲ್ಲದಿದ್ದರೆ {
        GGML_CANN_CALL_ACLNN_OP(ctx, Cos, acl_src, acl_dst);
    }
}

ಶೂನ್ಯ aclnn_sin(ggml_backend_cann_context & ctx, aclTensor * acl_src, aclTensor * acl_dst) {
    ಆದರೆ (acl_dst == ಶೂನ್ಯಸೂಚಕ) {
        GGML_CANN_CALL_ACLNN_OP(ctx, InplaceSin, acl_src);
    } ಇಲ್ಲದಿದ್ದರೆ {
        GGML_CANN_CALL_ACLNN_OP(ctx, Sin, acl_src, acl_dst);
    }
}

ಶೂನ್ಯ ggml_cann_timestep_embedding(ggml_backend_cann_context & ctx, ggml_tensor * dst) {
    ಸ್ಥಿರಾಂಕ ggml_tensor * src = dst->src[0];

    GGML_ASSERT(src->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ dim        = dst->op_params[0];
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ max_period = dst->op_params[1];
    ಪೂರ್ಣಾಂಕ       half       = dim / 2;

    acl_tensor_ptr acl_src = ggml_cann_create_tensor(src);

    // arange: [0, ..., half)
    ದಶಮಾಂಶ   start             = 0;
    ದಶಮಾಂಶ   stop              = half;
    ದಶಮಾಂಶ   ಹೆಜ್ಜೆ              = 1;
    int64_t n_elements_arange = half;
    int64_t tmp_arange_ne[]   = { half };
    size_t  tmp_arange_nb[]   = { ಗಾತ್ರ(dst->type) };

    ggml_cann_pool_alloc arange_allocator(ctx.pool(), half * ಗಾತ್ರ(dst->type));
    ಶೂನ್ಯ *               tmp_arange_ಬಫರ್ = arange_allocator.get();
    acl_tensor_ptr       tmp_arange_tensor =
        ggml_cann_create_tensor(tmp_arange_ಬಫರ್, ggml_cann_type_mapping(dst->type), ggml_type_ಗಾತ್ರ(dst->type),
                                tmp_arange_ne, tmp_arange_nb, GGML_MAX_DIMS - 3, ACL_FORMAT_ND);

    aclnn_arange(ctx, tmp_arange_tensor.get(), start, stop, ಹೆಜ್ಜೆ, n_elements_arange);

    // freq
    ದಶಮಾಂಶ freq_param = -logf(max_period) / half;
    ತಾರ್ಕಿಕ  inplace    = ನಿಜ;
    aclnn_muls(ctx, tmp_arange_tensor.get(), freq_param, ಶೂನ್ಯಸೂಚಕ, inplace);
    aclnn_exp(ctx, tmp_arange_tensor.get());

    // permute: src [0,1,2,3]->[0,1,3,2]
    int64_t tmp_ಕ್ರಮಬದಲಾವಣೆ_ne[] = { src->ne[1], src->ne[0], src->ne[2], src->ne[3] };
    size_t  tmp_ಕ್ರಮಬದಲಾವಣೆ_nb[GGML_MAX_DIMS];
    tmp_ಕ್ರಮಬದಲಾವಣೆ_nb[0] = ggml_type_ಗಾತ್ರ(src->type);
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 1; i < GGML_MAX_DIMS; i++) {
        tmp_ಕ್ರಮಬದಲಾವಣೆ_nb[i] = tmp_ಕ್ರಮಬದಲಾವಣೆ_nb[i - 1] * tmp_ಕ್ರಮಬದಲಾವಣೆ_ne[i - 1];
    }

    ggml_cann_pool_alloc ಕ್ರಮಬದಲಾವಣೆ_allocator(ctx.pool(), ggml_nbytes(src));
    ಶೂನ್ಯ *               tmp_ಕ್ರಮಬದಲಾವಣೆ_ಬಫರ್ = ಕ್ರಮಬದಲಾವಣೆ_allocator.get();
    acl_tensor_ptr       tmp_ಕ್ರಮಬದಲಾವಣೆ_tensor =
        ggml_cann_create_tensor(tmp_ಕ್ರಮಬದಲಾವಣೆ_ಬಫರ್, ggml_cann_type_mapping(src->type), ggml_type_ಗಾತ್ರ(src->type),
                                tmp_ಕ್ರಮಬದಲಾವಣೆ_ne, tmp_ಕ್ರಮಬದಲಾವಣೆ_nb, GGML_MAX_DIMS, ACL_FORMAT_ND);
    int64_t ಕ್ರಮಬದಲಾವಣೆ_dim[] = { 0, 1, 3, 2 };
    int64_t ಅಂಕೆ_dims      = 4;
    aclnn_ಕ್ರಮಬದಲಾವಣೆ(ctx, acl_src.get(), tmp_ಕ್ರಮಬದಲಾವಣೆ_tensor.get(), ಕ್ರಮಬದಲಾವಣೆ_dim, ಅಂಕೆ_dims);

    // timestep * freq
    int64_t tmp_mul_ne[] = { src->ne[1] * half, src->ne[0], src->ne[2], src->ne[3] };
    size_t  tmp_mul_nb[GGML_MAX_DIMS];
    tmp_mul_nb[0] = ggml_type_ಗಾತ್ರ(src->type);
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 1; i < GGML_MAX_DIMS; i++) {
        tmp_mul_nb[i] = tmp_mul_nb[i - 1] * tmp_mul_ne[i - 1];
    }

    ಪೂರ್ಣಾಂಕ mul_nelements = src->ne[1] * half * src->ne[0] * src->ne[2] * src->ne[3];

    ggml_cann_pool_alloc mul_allocator(ctx.pool(), mul_nelements * ggml_type_ಗಾತ್ರ(src->type));
    ಶೂನ್ಯ *               tmp_mul_ಬಫರ್ = mul_allocator.get();
    acl_tensor_ptr       tmp_mul_tensor =
        ggml_cann_create_tensor(tmp_mul_ಬಫರ್, ggml_cann_type_mapping(src->type), ggml_type_ಗಾತ್ರ(src->type),
                                tmp_mul_ne, tmp_mul_nb, GGML_MAX_DIMS, ACL_FORMAT_ND);
    aclnn_mul(ctx, tmp_ಕ್ರಮಬದಲಾವಣೆ_tensor.get(), tmp_arange_tensor.get(), tmp_mul_tensor.get());

    // cos
    ggml_cann_pool_alloc cos_allocator(ctx.pool(), mul_nelements * ggml_type_ಗಾತ್ರ(src->type));
    ಶೂನ್ಯ *               tmp_cos_ಬಫರ್ = cos_allocator.get();
    acl_tensor_ptr       tmp_cos_tensor =
        ggml_cann_create_tensor(tmp_cos_ಬಫರ್, ggml_cann_type_mapping(dst->type), ggml_type_ಗಾತ್ರ(dst->type),
                                tmp_mul_ne, tmp_mul_nb, GGML_MAX_DIMS, ACL_FORMAT_ND);

    aclnn_cos(ctx, tmp_mul_tensor.get(), tmp_cos_tensor.get());

    // sin
    ggml_cann_pool_alloc sin_allocator(ctx.pool(), mul_nelements * ggml_type_ಗಾತ್ರ(src->type));
    ಶೂನ್ಯ *               tmp_sin_ಬಫರ್ = sin_allocator.get();
    acl_tensor_ptr       tmp_sin_tensor =
        ggml_cann_create_tensor(tmp_sin_ಬಫರ್, ggml_cann_type_mapping(dst->type), ggml_type_ಗಾತ್ರ(dst->type),
                                tmp_mul_ne, tmp_mul_nb, GGML_MAX_DIMS, ACL_FORMAT_ND);

    aclnn_sin(ctx, tmp_mul_tensor.get(), tmp_sin_tensor.get());

    // concat
    int64_t             concat_dim  = 3;
    acl_tensor_ptr      acl_dst     = ggml_cann_create_tensor(dst);
    acl_tensor_list_ptr tensor_list = ggml_cann_create_tensor_list(tmp_cos_tensor, tmp_sin_tensor);
    aclnn_concat(ctx, tensor_list.get(), acl_dst.get(), concat_dim);
}

/**
 * @brief Raises each element of a tensor to the power of the corresponding
 * element in another tensor.
 *
 * This function computes the element-wise power of the destination tensor
 * `acl_dst` raised to the power of the exponent tensor `acl_exp`.
 * The operation is defined as:
 * \f[
 *     \text {acl_dst }_i=acl\_dst_i^{\text {acl_exp }_i}
 * \f]
 *
 * @param ctx The context for the CANN backend operations.
 * @param acl_dst The destination tensor, which also serves as the base tensor.
 * @param acl_exp The exponent tensor, each element of which is used to raise
 * the corresponding element in the destination tensor.
 */
ಸ್ಥಿರ ಶೂನ್ಯ aclnn_pow_tensor_tensor(ggml_backend_cann_context & ctx, aclTensor * acl_dst, aclTensor * acl_exp) {
    GGML_CANN_CALL_ACLNN_OP(ctx, InplacePowTensorTensor, acl_dst, acl_exp);
}

/**
 * @brief Generate a range of values and apply a scalar base exponentiation.
 *
 * This function creates an evenly spaced sequence from `start` to `stop` (exclusive),
 * with step size `step`, stores it in a temporary buffer, and then computes:
 *
 * @f[
 * slope[i] = m^{\left( start + i \cdot step \right)}, \quad 0 \le i < size
 * @f]
 *
 * The results are written to the provided @p slope_buffer.
 *
 * @param ctx           CANN backend context for memory allocation and operator execution.
 * @param slope_buffer  Pointer to the output buffer (float array) for the computed slope values.
 * @param m             Scalar base for the exponentiation.
 * @param size          Number of elements in the generated sequence.
 * @param start         Starting exponent offset.
 * @param stop          Stopping exponent offset (exclusive).
 * @param step          Step size for the exponent increment.
 * @param dtype         Data type for slope tensor.
 */
ಸ್ಥಿರ ಶೂನ್ಯ aclnn_get_ಇಳಿಜಾರು_inner(ggml_backend_cann_context & ctx,
                                  ಶೂನ್ಯ *                      ಇಳಿಜಾರು_ಬಫರ್,
                                  ದಶಮಾಂಶ                       m,
                                  int64_t                     size,
                                  ದಶಮಾಂಶ                       start,
                                  ದಶಮಾಂಶ                       stop,
                                  ದಶಮಾಂಶ                       ಹೆಜ್ಜೆ,
                                  ggml_type                   dtype) {
    acl_ದತ್ತಾಂಶ_Type acl_type  = ggml_cann_type_mapping(dtype);
    size_t      type_ಗಾತ್ರ = ggml_type_ಗಾತ್ರ(dtype);

    int64_t ne[] = { size };
    size_t  nb[] = { type_ಗಾತ್ರ };

    ggml_cann_pool_alloc arange_allocator(ctx.pool(), size * type_ಗಾತ್ರ);
    ಶೂನ್ಯ *               arange_ಬಫರ್ = arange_allocator.get();

    acl_tensor_ptr arange_tensor = ggml_cann_create_tensor(arange_ಬಫರ್, acl_type, type_ಗಾತ್ರ, ne, nb, 1);
    aclnn_arange(ctx, arange_tensor.get(), start, stop, ಹೆಜ್ಜೆ, size);

    acl_tensor_ptr ಇಳಿಜಾರು_tensor = ggml_cann_create_tensor(ಇಳಿಜಾರು_ಬಫರ್, acl_type, type_ಗಾತ್ರ, ne, nb, 1);

    acl_scalar_ptr sc = ggml_cann_create_scalar(&m, acl_ದತ್ತಾಂಶ_Type::ACL_FLOAT);

    GGML_CANN_CALL_ACLNN_OP(ctx, PowScalarTensor, sc.get(), arange_tensor.get(), ಇಳಿಜಾರು_tensor.get());
}

/**
 * @brief Compute slope values for multiple attention heads based on ALiBi bias parameters.
 *
 * This function generates slope values for each attention head according to the ALiBi
 * (Attention with Linear Biases) method. It splits the computation into two ranges depending
 * on whether the head index is less than @p n_head_log2 or not, and uses different base values
 * (`m0` and `m1`) for the exponentiation.
 *
 * @f[
 * slope[h] =
 * \begin{cases}
 * m_0^{(h + 1)}, & h < n\_head\_log2 \\
 * m_1^{\left( 2 \cdot (h - n\_head\_log2) + 1 \right)}, & h \geq n\_head\_log2
 * \end{cases}
 * \quad , \quad \text{if } max\_bias > 0
 * @f]
 *
 * If @p max_bias <= 0, all slope values are set to 1.0.
 *
 * @param ctx           CANN backend context for memory allocation and operator execution.
 * @param n_head        Total number of attention heads.
 * @param slope_buffer  Pointer to the output buffer (float array) for storing slopes.
 * @param max_bias      Maximum bias value for slope computation.
 * @param dtype         Data type for slope tensor.
 *
*/
ಸ್ಥಿರ ಶೂನ್ಯ aclnn_get_ಇಳಿಜಾರು(ggml_backend_cann_context & ctx,
                            int64_t                     n_ತಲೆ,
                            ಶೂನ್ಯ *                      ಇಳಿಜಾರು_ಬಫರ್,
                            ದಶಮಾಂಶ                       max_bias,
                            ggml_type                   dtype) {
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ n_ತಲೆ_log2 = 1u << (uint32_t) floor(log2(n_ತಲೆ));

    ದಶಮಾಂಶ m0 = powf(2.0f, -(max_bias) / n_ತಲೆ_log2);
    ದಶಮಾಂಶ m1 = powf(2.0f, -(max_bias / 2.0f) / n_ತಲೆ_log2);

    // const float slope = (max_bias > 0.0f) ?
    //                          h < n_head_log2 ?
    //                              powf(m0, h + 1) :
    //                              powf(m1, 2*(h - n_head_log2) + 1) :
    //                          1.0f;
    // arange1
    ದಶಮಾಂಶ start = 0 + 1;
    ದಶಮಾಂಶ end   = (n_ತಲೆ_log2 - 1) + 1;
    ದಶಮಾಂಶ ಹೆಜ್ಜೆ  = 1;
    ದಶಮಾಂಶ count = n_ತಲೆ_log2;
    // end needs to be +1 because aclnn uses a left-closed, right-open interval.
    aclnn_get_ಇಳಿಜಾರು_inner(ctx, ಇಳಿಜಾರು_ಬಫರ್, m0, count, start, end + 1, ಹೆಜ್ಜೆ, dtype);
    ಆದರೆ (n_ತಲೆ_log2 < n_ತಲೆ) {
        // arange2
        start = 2 * (n_ತಲೆ_log2 - n_ತಲೆ_log2) + 1;
        end   = 2 * ((n_ತಲೆ - 1) - n_ತಲೆ_log2) + 1;
        ಹೆಜ್ಜೆ  = 2;
        count = n_ತಲೆ - n_ತಲೆ_log2;
        aclnn_get_ಇಳಿಜಾರು_inner(ctx, (ಅಕ್ಷರ *) ಇಳಿಜಾರು_ಬಫರ್ + n_ತಲೆ_log2 * ggml_type_ಗಾತ್ರ(dtype), m1, count, start, end + 1,
                              ಹೆಜ್ಜೆ, dtype);
    }
}

/**
 * @brief Add ALiBi (Attention with Linear Biases) positional biases to the attention mask.
 *
 * This function computes the ALiBi slopes for each attention head (if max_bias > 0),
 * multiplies them with the attention mask to produce bias tensors, and adds these biases
 * to the destination tensor (@p dst).
 *
 * The function performs necessary broadcasting of the mask and slope tensors to match
 * the shape of the destination tensor, then applies element-wise multiplication and addition
 * using CANN operators.
 *
 * @param ctx         CANN backend context for memory management and operator execution.
 * @param mask        Input attention mask tensor, assumed to be contiguous.
 * @param dst         Destination tensor to which ALiBi biases will be added.
 * @param dst_ptr     Pointer to the memory of the destination tensor.
 * @param max_bias    Maximum bias value controlling the slope scaling.
 *
 * @note
 * - Write data into dst_ptr using only the shape information of the dst tensor.
 * - `GGML_MAX_DIMS + 2` is used to extend tensor dimensions for broadcasting.
 */
ಸ್ಥಿರ ಶೂನ್ಯ aclnn_ಕೂಡಿಸು_alibi(ggml_backend_cann_context & ctx,
                            ggml_tensor *               ಮುಖವಾಡ,
                            ggml_tensor *               dst,
                            ಶೂನ್ಯ *                      dst_ptr,
                            ದಶಮಾಂಶ                       max_bias) {
    ಶೂನ್ಯ * ಇಳಿಜಾರು_ಬಫರ್ = ಶೂನ್ಯಸೂಚಕ;
    ಶೂನ್ಯ * bias_ಬಫರ್  = ಶೂನ್ಯಸೂಚಕ;

    ಆದರೆ (max_bias > 0.0f) {
        int64_t              n_heads = dst->ne[2];
        ggml_cann_pool_alloc ಇಳಿಜಾರು_allocator(ctx.pool(), n_heads * ಗಾತ್ರ(ದಶಮಾಂಶ));
        ಇಳಿಜಾರು_ಬಫರ್ = ಇಳಿಜಾರು_allocator.get();
        ggml_cann_pool_alloc bias_allocator(ctx.pool(), ggml_nelements(dst) * ggml_element_ಗಾತ್ರ(dst));
        bias_ಬಫರ್ = bias_allocator.get();
        aclnn_get_ಇಳಿಜಾರು(ctx, n_heads, ಇಳಿಜಾರು_ಬಫರ್, max_bias, GGML_TYPE_F32);
    }

    // broadcast for mask, slop and dst;
    int64_t nr2 = dst->ne[2] / ಮುಖವಾಡ->ne[2];
    int64_t nr3 = dst->ne[3] / ಮುಖವಾಡ->ne[3];

    // broadcast the mask across rows
    int64_t ಮುಖವಾಡ_ne[] = { ಮುಖವಾಡ->ne[0], dst->ne[1], ಮುಖವಾಡ->ne[2], 1, ಮುಖವಾಡ->ne[3], 1 };
    size_t  ಮುಖವಾಡ_nb[] = { ಮುಖವಾಡ_nb[0] = ಮುಖವಾಡ->nb[0], ಮುಖವಾಡ_nb[1] = ಮುಖವಾಡ->nb[1], ಮುಖವಾಡ_nb[2] = ಮುಖವಾಡ->nb[2],
                          ಮುಖವಾಡ_nb[3] = ಮುಖವಾಡ->nb[2], ಮುಖವಾಡ_nb[4] = ಮುಖವಾಡ->nb[3], ಮುಖವಾಡ_nb[5] = ಮುಖವಾಡ->nb[3] };

    int64_t dst_ne[] = { dst->ne[0], dst->ne[1], ಮುಖವಾಡ->ne[2], nr2, ಮುಖವಾಡ->ne[3], nr3 };
    size_t  dst_nb[] = { dst_nb[0] = dst->nb[0], dst_nb[1] = dst->nb[1], dst_nb[2] = dst->nb[2],
                         dst_nb[3] = dst->nb[2], dst_nb[4] = dst->nb[3], dst_nb[5] = dst->nb[3] };

    // slope is a 1 dim tensor, slope.ne2 == dst.ne2
    int64_t ಇಳಿಜಾರು_ne[] = { 1, 1, ಮುಖವಾಡ->ne[2], nr2, 1, 1 };
    size_t  ಇಳಿಜಾರು_nb[GGML_MAX_DIMS + 2];
    ಇಳಿಜಾರು_nb[0] = ಗಾತ್ರ(ದಶಮಾಂಶ);
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 1; i < GGML_MAX_DIMS + 2; i++) {
        ಇಳಿಜಾರು_nb[i] = ಇಳಿಜಾರು_nb[i - 1] * ಇಳಿಜಾರು_ne[i - 1];
    }

    acl_tensor_ptr acl_ಇಳಿಜಾರು =
        ggml_cann_create_tensor(ಇಳಿಜಾರು_ಬಫರ್, ACL_FLOAT, ಗಾತ್ರ(ದಶಮಾಂಶ), ಇಳಿಜಾರು_ne, ಇಳಿಜಾರು_nb, GGML_MAX_DIMS + 2);
    acl_tensor_ptr acl_ಮುಖವಾಡ = ggml_cann_create_tensor(ಮುಖವಾಡ, ಮುಖವಾಡ_ne, ಮುಖವಾಡ_nb, GGML_MAX_DIMS + 2);

    // write data into dst_ptr using only the shape information of the dst tensor.
    acl_tensor_ptr acl_dst = ggml_cann_create_tensor(dst_ptr, ggml_cann_type_mapping(dst->type),
                                                     ggml_type_ಗಾತ್ರ(dst->type), dst_ne, dst_nb, GGML_MAX_DIMS + 2);

    ಆದರೆ (max_bias > 0.0f) {
        int64_t bias_ne[] = { ಮುಖವಾಡ->ne[0], dst->ne[1], ಮುಖವಾಡ->ne[2], nr2, ಮುಖವಾಡ->ne[3], 1 };
        size_t  bias_nb[GGML_MAX_DIMS + 2];
        bias_nb[0] = ಗಾತ್ರ(ದಶಮಾಂಶ);
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 1; i < GGML_MAX_DIMS + 2; i++) {
            bias_nb[i] = bias_nb[i - 1] * bias_ne[i - 1];
        }
        acl_tensor_ptr bias_tensor =
            ggml_cann_create_tensor(bias_ಬಫರ್, ACL_FLOAT, ಗಾತ್ರ(ದಶಮಾಂಶ), bias_ne, bias_nb, GGML_MAX_DIMS + 2);

        aclnn_mul(ctx, acl_ಇಳಿಜಾರು.get(), acl_ಮುಖವಾಡ.get(), bias_tensor.get());
        aclnn_ಕೂಡಿಸು(ctx, acl_dst.get(), bias_tensor.get());
    } ಇಲ್ಲದಿದ್ದರೆ {
        aclnn_ಕೂಡಿಸು(ctx, acl_dst.get(), acl_ಮುಖವಾಡ.get());
    }
}

ಶೂನ್ಯ ggml_cann_cpy(ggml_backend_cann_context & ctx, ggml_tensor * dst) {
    ggml_cann_dup(ctx, dst);
}

/**
 * @brief Applies the softmax function to a tensor along a specified dimension.
 *
 * This function computes the softmax of the source tensor `acl_src` along the
 * specified dimension `dim` and stores the result in the destination tensor
 * `acl_dst`.
 *
 * @param ctx The context for the CANN backend operations.
 * @param acl_src The source tensor on which the softmax function will be
 * applied.
 * @param dim The dimension along which the softmax function will be computed.
 * @param acl_dst The destination tensor where the softmax results will be
 * stored.
 */
ಸ್ಥಿರ ಶೂನ್ಯ aclnn_softmax(ggml_backend_cann_context & ctx, aclTensor * acl_src, int64_t dim, aclTensor * acl_dst) {
    GGML_CANN_CALL_ACLNN_OP(ctx, Softmax, acl_src, dim, acl_dst);
}

ಶೂನ್ಯ ggml_cann_softmax(ggml_backend_cann_context & ctx, ggml_tensor * dst) {
    ggml_tensor * src0 = dst->src[0];
    ggml_tensor * src1 = dst->src[1];  //ಮುಖವಾಡ

    acl_tensor_ptr acl_src0 = ggml_cann_create_tensor(src0);
    acl_tensor_ptr acl_dst  = ggml_cann_create_tensor(dst);

    ದಶಮಾಂಶ scale    = 1.0f;
    ದಶಮಾಂಶ max_bias = 0.0f;

    ಸ್ಮರಣೆ_ನಕಲು(&scale, (ದಶಮಾಂಶ *) dst->op_params + 0, ಗಾತ್ರ(ದಶಮಾಂಶ));
    ಸ್ಮರಣೆ_ನಕಲು(&max_bias, (ದಶಮಾಂಶ *) dst->op_params + 1, ಗಾತ್ರ(ದಶಮಾಂಶ));

    // input mul scale
    acl_scalar_ptr       acl_scale = ggml_cann_create_scalar(&scale, acl_ದತ್ತಾಂಶ_Type::ACL_FLOAT);
    ggml_cann_pool_alloc src_tensor_allocator(ctx.pool(), ggml_nbytes(src0));
    ಶೂನ್ಯ *               src_tensor_ಬಫರ್ = src_tensor_allocator.get();
    acl_tensor_ptr       softmax_tensor = ggml_cann_create_tensor(src_tensor_ಬಫರ್, ggml_cann_type_mapping(src0->type),
                                                                  ggml_element_ಗಾತ್ರ(src0), src0->ne, src0->nb, GGML_MAX_DIMS);

    aclnn_muls(ctx, acl_src0.get(), scale, softmax_tensor.get(), ಸುಳ್ಳು);

    //ಮುಖವಾಡ
    ಆದರೆ (src1) {
        aclnn_ಕೂಡಿಸು_alibi(ctx, src1, src0, src_tensor_ಬಫರ್, max_bias);
    }
    // softmax
    aclnn_softmax(ctx, softmax_tensor.get(), 3, acl_dst.get());
}


ಶೂನ್ಯ ggml_cann_get_ಅಡ್ಡಸಾಲುಗಳು(ggml_backend_cann_context & ctx, ggml_tensor * dst) {
    ggml_tensor * src0 = dst->src[0];  //ತೂಕ
    ggml_tensor * src1 = dst->src[1];  //ಸೂಚ್ಯಂಕ

    GGML_ASSERT(dst->type == GGML_TYPE_F32 || dst->type == GGML_TYPE_F16
                || dst->type == GGML_TYPE_BF16);

    // n_idx: number of row indices per (i2, i3) batch slice.
    // ggml guarantees: src0->ne[2] == src1->ne[1], src0->ne[3] == src1->ne[2], src1->ne[3] == 1.
    ಸ್ಥಿರಾಂಕ int64_t n_idx = src1->ne[0];

    // Gather all (i2, i3) batch slices from src into dst.
    // ggml_cann_create_tensor reverses dims, so ACL sees [ne1, ne0].
    // GatherV2 with dim=0 gathers along ACL dim-0 == ggml ne[1] (the vocabulary / row axis).
    // nb: the 4 strides of the source buffer (nb[0..1] for the 2D slice shape,
    //     nb[2..3] for computing per-batch-slice base pointer offsets).
    ಸ್ವಯಂಚಾಲಿತ gather_batched = [&](ಶೂನ್ಯ * src_ತಳ, acl_ದತ್ತಾಂಶ_Type acl_type, size_t type_ಗಾತ್ರ,
                              ಸ್ಥಿರಾಂಕ size_t * nb) {
        int64_t src_ne[2]  = { src0->ne[0], src0->ne[1] };
        size_t  src_nb_2d[2] = { nb[0], nb[1] };
        int64_t dst_ne[2]  = { src0->ne[0], n_idx };
        size_t  dst_nb_2d[2] = { dst->nb[0], dst->nb[1] };
        int64_t idx_ne[1]  = { n_idx };
        size_t  idx_nb[1]  = { (size_t)ggml_element_ಗಾತ್ರ(src1) };

        ಗಾಗಿ (int64_t i3 = 0; i3 < src0->ne[3]; i3++) {
            ಗಾಗಿ (int64_t i2 = 0; i2 < src0->ne[2]; i2++) {
                acl_tensor_ptr acl_src = ggml_cann_create_tensor(
                    (ಅಕ್ಷರ *)src_ತಳ + i3 * nb[3] + i2 * nb[2],
                    acl_type, type_ಗಾತ್ರ, src_ne, src_nb_2d, 2);
                acl_tensor_ptr acl_idx = ggml_cann_create_tensor(
                    (ಅಕ್ಷರ *)src1->ದತ್ತಾಂಶ + i3 * src1->nb[2] + i2 * src1->nb[1],
                    ggml_cann_type_mapping(src1->type), (size_t)ggml_element_ಗಾತ್ರ(src1),
                    idx_ne, idx_nb, 1);
                acl_tensor_ptr acl_dst = ggml_cann_create_tensor(
                    (ಅಕ್ಷರ *)dst->ದತ್ತಾಂಶ + i3 * dst->nb[3] + i2 * dst->nb[2],
                    acl_type, type_ಗಾತ್ರ, dst_ne, dst_nb_2d, 2);
                GGML_CANN_CALL_ACLNN_OP(ctx, GatherV2, acl_src.get(), 0, acl_idx.get(), acl_dst.get());
            }
        }
    };

    ಆಯ್ಕೆ (src0->type) {
        ಸ್ಥಿತಿ GGML_TYPE_BF16:
        ಸ್ಥಿತಿ GGML_TYPE_F16:
        ಸ್ಥಿತಿ GGML_TYPE_F32:
            ಆದರೆ (src0->type == dst->type) {
                gather_batched(src0->ದತ್ತಾಂಶ,
                               ggml_cann_type_mapping(src0->type), ggml_type_ಗಾತ್ರ(src0->type),
                               src0->nb);
            } ಇಲ್ಲದಿದ್ದರೆ {
                // Cast src0 to dst type, then gather.
                ggml_cann_pool_alloc src_cast_allocator(ctx.pool(),
                                                        ggml_nelements(src0) * ggml_element_ಗಾತ್ರ(dst));
                size_t src_cast_nb[GGML_MAX_DIMS];
                src_cast_nb[0] = ggml_type_ಗಾತ್ರ(dst->type);
                ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 1; i < GGML_MAX_DIMS; i++) {
                    src_cast_nb[i] = src_cast_nb[i - 1] * src0->ne[i - 1];
                }
                acl_tensor_ptr acl_src0     = ggml_cann_create_tensor(src0);
                acl_tensor_ptr acl_src_cast = ggml_cann_create_tensor(
                    src_cast_allocator.get(), ggml_cann_type_mapping(dst->type), ggml_type_ಗಾತ್ರ(dst->type),
                    src0->ne, src_cast_nb, GGML_MAX_DIMS);
                aclnn_cast(ctx, acl_src0.get(), acl_src_cast.get(), ggml_cann_type_mapping(dst->type));

                gather_batched(src_cast_allocator.get(),
                               ggml_cann_type_mapping(dst->type), ggml_type_ಗಾತ್ರ(dst->type),
                               src_cast_nb);
            }
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_TYPE_Q8_0:
            {
                // Dequantize Q8_0 to dst type, then gather.
                size_t  ತೂಕ_nb[GGML_MAX_DIMS + 1], scale_nb[GGML_MAX_DIMS + 1], dequant_nb[GGML_MAX_DIMS + 1];
                int64_t ತೂಕ_ne[GGML_MAX_DIMS + 1], scale_ne[GGML_MAX_DIMS + 1], *dequant_ne;
                ತೂಕ_ne[0] = QK8_0;
                ತೂಕ_ne[1] = src0->ne[0] / QK8_0;
                ತೂಕ_nb[0] = ಗಾತ್ರ(int8_t);
                ತೂಕ_nb[1] = ತೂಕ_nb[0] * ತೂಕ_ne[0];
                ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 2; i < GGML_MAX_DIMS + 1; i++) {
                    ತೂಕ_ne[i] = src0->ne[i - 1];
                    ತೂಕ_nb[i] = ತೂಕ_nb[i - 1] * ತೂಕ_ne[i - 1];
                }
                scale_ne[0] = 1;
                scale_ne[1] = src0->ne[0] / QK8_0;
                scale_nb[0] = ಗಾತ್ರ(uint16_t);
                scale_nb[1] = scale_nb[0] * scale_ne[0];
                ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 2; i < GGML_MAX_DIMS + 1; i++) {
                    scale_ne[i] = src0->ne[i - 1];
                    scale_nb[i] = scale_nb[i - 1] * scale_ne[i - 1];
                }
                dequant_ne    = ತೂಕ_ne;
                dequant_nb[0] = ggml_type_ಗಾತ್ರ(dst->type);
                ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 1; i < GGML_MAX_DIMS + 1; i++) {
                    dequant_nb[i] = dequant_nb[i - 1] * dequant_ne[i - 1];
                }
                ಸ್ಥಿರಾಂಕ int64_t scale_ಸ್ಥಾನಾಂತರ = ggml_nelements(src0) * ಗಾತ್ರ(int8_t);
                ggml_cann_pool_alloc dequant_allocator(ctx.pool(),
                                                       ggml_nelements(src0) * ggml_type_ಗಾತ್ರ(dst->type));
                acl_tensor_ptr acl_ತೂಕ = ggml_cann_create_tensor(src0->ದತ್ತಾಂಶ, ACL_INT8, ಗಾತ್ರ(int8_t),
                                                                     ತೂಕ_ne, ತೂಕ_nb, GGML_MAX_DIMS + 1);
                acl_tensor_ptr acl_scale  = ggml_cann_create_tensor(
                    src0->ದತ್ತಾಂಶ, ACL_FLOAT16, ಗಾತ್ರ(uint16_t), scale_ne, scale_nb,
                    GGML_MAX_DIMS + 1, ACL_FORMAT_ND, scale_ಸ್ಥಾನಾಂತರ);
                acl_tensor_ptr acl_dequant = ggml_cann_create_tensor(
                    dequant_allocator.get(), ggml_cann_type_mapping(dst->type),
                    ggml_type_ಗಾತ್ರ(dst->type), dequant_ne, dequant_nb, GGML_MAX_DIMS + 1);
                aclnn_mul(ctx, acl_ತೂಕ.get(), acl_scale.get(), acl_dequant.get());

                // Reinterpret dequant buffer as 4D [src0->ne] with contiguous strides.
                dequant_ne    = src0->ne;
                dequant_nb[0] = ggml_type_ಗಾತ್ರ(dst->type);
                ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 1; i < GGML_MAX_DIMS; i++) {
                    dequant_nb[i] = dequant_nb[i - 1] * src0->ne[i - 1];
                }
                gather_batched(dequant_allocator.get(),
                               ggml_cann_type_mapping(dst->type), ggml_type_ಗಾತ್ರ(dst->type),
                               dequant_nb);
                ನಿಲ್ಲಿಸು;
            }
        ಪೂರ್ವನಿಯೋಜಿತ:
            GGML_ABORT("Unsupported tensor type for GGML_OP_GET_ROWS");
            ನಿಲ್ಲಿಸು;
    }
}

ಶೂನ್ಯ ggml_cann_ಗಣ_ಅಡ್ಡಸಾಲುಗಳು(ggml_backend_cann_context & ctx, ggml_tensor * dst) {
    ggml_tensor * src0 = dst->src[0];  // source values
    ggml_tensor * src1 = dst->src[1];  // row indices

    // n_idx: number of source rows to scatter per batch slice.
    // ggml guarantees: src0->ne[1] == src1->ne[0].
    ಸ್ಥಿರಾಂಕ int64_t n_idx = src1->ne[0];

    // Copy n_idx rows of src [ne0, n_idx] into dst [ne0, ne1] at positions given by a 1D index.
    // ggml_cann_create_tensor reverses dims, so ACL sees [ne1, ne0] for dst.
    // InplaceIndexCopy with dim=0 copies along ACL dim-0 == ggml ne[1] (the row axis).
    // src_nb: the 4 strides of the source buffer (nb[0..1] for the 2D slice shape,
    //         nb[2..3] for computing per-batch-slice base pointer offsets).
    ಸ್ವಯಂಚಾಲಿತ scatter_batched = [&](ಶೂನ್ಯ * src_ತಳ, acl_ದತ್ತಾಂಶ_Type acl_type, size_t type_ಗಾತ್ರ,
                               ಸ್ಥಿರಾಂಕ size_t * src_nb) {
        int64_t d_ne[2]    = { dst->ne[0], dst->ne[1] };
        size_t  d_nb[2]    = { dst->nb[0], dst->nb[1] };
        int64_t s_ne[2]    = { dst->ne[0], n_idx };
        size_t  s_nb_2d[2] = { src_nb[0], src_nb[1] };
        int64_t i_ne[1]    = { n_idx };
        size_t  i_nb[1]    = { (size_t)ggml_element_ಗಾತ್ರ(src1) };

        ಗಾಗಿ (int64_t i3 = 0; i3 < dst->ne[3]; i3++) {
            ಗಾಗಿ (int64_t i2 = 0; i2 < dst->ne[2]; i2++) {
                acl_tensor_ptr acl_dst = ggml_cann_create_tensor(
                    (ಅಕ್ಷರ *)dst->ದತ್ತಾಂಶ + i3 * dst->nb[3] + i2 * dst->nb[2],
                    acl_type, type_ಗಾತ್ರ, d_ne, d_nb, 2);
                acl_tensor_ptr acl_idx = ggml_cann_create_tensor(
                    (ಅಕ್ಷರ *)src1->ದತ್ತಾಂಶ + (i3 % src1->ne[2]) * src1->nb[2] + (i2 % src1->ne[1]) * src1->nb[1],
                    ggml_cann_type_mapping(src1->type), (size_t)ggml_element_ಗಾತ್ರ(src1),
                    i_ne, i_nb, 1);
                acl_tensor_ptr acl_src = ggml_cann_create_tensor(
                    (ಅಕ್ಷರ *)src_ತಳ + i3 * src_nb[3] + i2 * src_nb[2],
                    acl_type, type_ಗಾತ್ರ, s_ne, s_nb_2d, 2);
                GGML_CANN_CALL_ACLNN_OP(ctx, Inplace_ಸೂಚ್ಯಂಕ_Copy, acl_dst.get(), 0, acl_idx.get(), acl_src.get());
            }
        }
    };

    ಆಯ್ಕೆ (dst->type) {
        ಸ್ಥಿತಿ GGML_TYPE_F32:
            scatter_batched(src0->ದತ್ತಾಂಶ,
                            ggml_cann_type_mapping(dst->type), ggml_type_ಗಾತ್ರ(dst->type),
                            src0->nb);
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_TYPE_F16:
        ಸ್ಥಿತಿ GGML_TYPE_BF16:
            {
                // Cast src0 (F32) to dst type first.
                ggml_cann_pool_alloc src_cast_allocator(ctx.pool(),
                                                        ggml_nelements(src0) * ggml_type_ಗಾತ್ರ(dst->type));
                size_t src_cast_nb[GGML_MAX_DIMS];
                src_cast_nb[0] = ggml_type_ಗಾತ್ರ(dst->type);
                ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 1; i < GGML_MAX_DIMS; i++) {
                    src_cast_nb[i] = src_cast_nb[i - 1] * src0->ne[i - 1];
                }
                acl_tensor_ptr acl_src0     = ggml_cann_create_tensor(src0);
                acl_tensor_ptr acl_src_cast = ggml_cann_create_tensor(
                    src_cast_allocator.get(), ggml_cann_type_mapping(dst->type), ggml_type_ಗಾತ್ರ(dst->type),
                    src0->ne, src_cast_nb, GGML_MAX_DIMS);
                aclnn_cast(ctx, acl_src0.get(), acl_src_cast.get(), ggml_cann_type_mapping(dst->type));

                scatter_batched(src_cast_allocator.get(),
                                ggml_cann_type_mapping(dst->type), ggml_type_ಗಾತ್ರ(dst->type),
                                src_cast_nb);
                ನಿಲ್ಲಿಸು;
            }
        ಪೂರ್ವನಿಯೋಜಿತ:
            GGML_ABORT("Unsupported tensor type for GGML_OP_SET_ROWS");
            ನಿಲ್ಲಿಸು;
    }
}

/**
 * @brief Repeats elements of a tensor along a specified dimension.
 *
 * This function repeats each element of the source tensor `acl_src` a specified
 * number of times (`repeats`) along the specified dimension `dim` and stores
 * the result in the destination tensor `acl_dst`.
 *
 * @param ctx The context for the CANN backend operations.
 * @param acl_src The source tensor whose elements will be repeated.
 * @param acl_dst The destination tensor where the repeated elements will be
 * stored.
 * @param dim The dimension along which the elements will be repeated.
 * @param repeats The number of times each element will be repeated.
 * @param output_size The size of the output tensor.
 */
ಸ್ಥಿರ ಶೂನ್ಯ aclnn_repeat_interleave(ggml_backend_cann_context & ctx,
                                    aclTensor *                 acl_src,
                                    aclTensor *                 acl_dst,
                                    int64_t                     dim,
                                    int64_t                     repeats,
                                    int64_t                     output_ಗಾತ್ರ) {
    GGML_CANN_CALL_ACLNN_OP(ctx, RepeatInterleaveIntWithDim, acl_src, repeats, dim, output_ಗಾತ್ರ, acl_dst);
}

/**
 * @brief Performs matrix multiplication with floating-point precision on
 * tensors using the CANN backend.
 *
 * This function performs matrix multiplication of the input tensor and the
 * weight tensor, handling broadcasting and transposing as needed, and stores
 * the result in the destination tensor `dst`.
 *
 * @param ctx The context for the CANN backend operations.
 * @param dst The destination tensor where the result of the matrix
 * multiplication will be stored.
 */
ಸ್ಥಿರ ಶೂನ್ಯ ggml_cann_mat_mul_fp(ggml_backend_cann_context & ctx, ggml_tensor * dst) {
    ggml_tensor * ತೂಕ = dst->src[0];  //ತೂಕ
    ggml_tensor * ಒಳಹರಿವು  = dst->src[1];  //ಒಳಹರಿವು

    // when weight ne2 or ne3 is 1, aclnnMatmulGetWorkspaceSize will auto
    // broadcast, when weight ne2 or ne3 is not 1, weight need repeat.
    BCAST_MUL_MAT_SHAPE(ಒಳಹರಿವು, ತೂಕ, dst);

    int64_t n_dims = bcast_dims;
    ಆದರೆ (bcast_ಒಳಹರಿವು_ne[3] == bcast_ತೂಕ_ne[3] && bcast_ಒಳಹರಿವು_ne[3] == 1) {
        ಆದರೆ (bcast_ಒಳಹರಿವು_ne[2] == 1 && bcast_ತೂಕ_ne[2] == 1) {
            n_dims = 2;
        } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (bcast_ಒಳಹರಿವು_ne[2] == 1) {
            n_dims = 3;
        }
    }

    acl_tensor_ptr acl_ಒಳಹರಿವು_tensor = ggml_cann_create_tensor(ಒಳಹರಿವು, bcast_ಒಳಹರಿವು_ne, bcast_ಒಳಹರಿವು_nb, n_dims);
    int64_t        transpose_ne[]   = { bcast_ತೂಕ_ne[1], bcast_ತೂಕ_ne[0], bcast_ತೂಕ_ne[2],
                                        bcast_ತೂಕ_ne[3], bcast_ತೂಕ_ne[4], bcast_ತೂಕ_ne[5] };
    size_t         transpose_nb[]   = { bcast_ತೂಕ_nb[1], bcast_ತೂಕ_nb[0], bcast_ತೂಕ_nb[2],
                                        bcast_ತೂಕ_nb[3], bcast_ತೂಕ_nb[4], bcast_ತೂಕ_nb[5] };
    acl_tensor_ptr acl_ತೂಕ_tensor;

    // Only check env once.
    ಸ್ಥಿರ ತಾರ್ಕಿಕ ತೂಕ_ಗಮ್ಯ_nz = parse_bool(get_env_as_lowercase("GGML_CANN_WEIGHT_NZ").value_or("on"));
    ಆದರೆ (ತೂಕ_ಗಮ್ಯ_nz && ತೂಕ->type != GGML_TYPE_BF16 && is_matmul_ತೂಕ(ತೂಕ)) {
        acl_ತೂಕ_tensor = ggml_cann_create_tensor(ತೂಕ, transpose_ne, transpose_nb, n_dims, ACL_FORMAT_FRACTAL_NZ);
    } ಇಲ್ಲದಿದ್ದರೆ {
        acl_ತೂಕ_tensor = ggml_cann_create_tensor(ತೂಕ, transpose_ne, transpose_nb, n_dims, ACL_FORMAT_ND);
    }
    acl_tensor_ptr acl_dst = ggml_cann_create_tensor(dst, bcast_dst_ne, bcast_dst_nb, n_dims);

    ಆಯ್ಕೆ (n_dims) {
        ಸ್ಥಿತಿ 2:
            GGML_CANN_CALL_ACLNN_OP(ctx, Mm, acl_ಒಳಹರಿವು_tensor.get(), acl_ತೂಕ_tensor.get(), acl_dst.get(), 2);
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ 3:
            GGML_CANN_CALL_ACLNN_OP(ctx, BatchMatMul, acl_ಒಳಹರಿವು_tensor.get(), acl_ತೂಕ_tensor.get(), acl_dst.get(),
                                    2);
            ನಿಲ್ಲಿಸು;
        ಪೂರ್ವನಿಯೋಜಿತ:
            // ALLOW_FP32_DOWN_PRECISION, when input is
            // fp32, atlas a2 will transpose it to HFLOAT32.
            GGML_CANN_CALL_ACLNN_OP(ctx, Matmul, acl_ಒಳಹರಿವು_tensor.get(), acl_ತೂಕ_tensor.get(), acl_dst.get(), 1);
            ನಿಲ್ಲಿಸು;
    }
}

/**
 * @brief Performs matrix multiplication with quantized weights and
 * floating-point inputs using the CANN backend.
 *
 * This function performs matrix multiplication of the input tensor `src1` and
 * the weight tensor `src0`, handling broadcasting, transposing, and
 * quantization as needed, and stores the result in the destination tensor
 * `dst`.
 *
 * @param ctx The context for the CANN backend operations.
 * @param dst The destination tensor where the result of the matrix
 * multiplication will be stored.
 */
ಸ್ಥಿರ ಶೂನ್ಯ ggml_cann_mul_mat_quant(ggml_backend_cann_context & ctx, ggml_tensor * dst, ಸ್ಥಿರಾಂಕ ಎಣಿಕೆ ggml_type type) {
    ggml_tensor * src0 = dst->src[0];  //ತೂಕ
    ggml_tensor * src1 = dst->src[1];  //ಒಳಹರಿವು

    // The shape of the weight is NCHW.
    // Matrix multiplication uses HW dims.
    // HC is regarded as batch.
    // weight need transpose.
    ದಶಮಾಂಶ ತೂಕ_elem_ಗಾತ್ರ;
    ಆದರೆ (type == GGML_TYPE_Q4_0) {
        ತೂಕ_elem_ಗಾತ್ರ = ದಶಮಾಂಶ(ಗಾತ್ರ(uint8_t)) / 2;
    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (type == GGML_TYPE_Q8_0) {
        ತೂಕ_elem_ಗಾತ್ರ = ದಶಮಾಂಶ(ಗಾತ್ರ(uint8_t));
    } ಇಲ್ಲದಿದ್ದರೆ {
        GGML_ABORT("Only support Q4_0 and Q8_0 MUL_MAT");
    }
    ದಶಮಾಂಶ  ತೂಕ_nb[]   = { src0->ne[0] * ತೂಕ_elem_ಗಾತ್ರ, ತೂಕ_elem_ಗಾತ್ರ };
    size_t ತೂಕ_stride = src0->ne[1] * src0->ne[0] * ತೂಕ_elem_ಗಾತ್ರ;
    size_t ತೂಕ_ಗಾತ್ರ   = ತೂಕ_stride * src0->ne[2] * src0->ne[3];

    // scale stored at the end of weight. Also need transpose.
    size_t scale_elem_ಗಾತ್ರ = ಗಾತ್ರ(uint16_t);
    size_t scale_nb[]      = { src0->ne[0] / QK8_0 * scale_elem_ಗಾತ್ರ, scale_elem_ಗಾತ್ರ };
    size_t scale_stride    = src0->ne[1] * src0->ne[0] / QK8_0 * scale_elem_ಗಾತ್ರ;
    ಅಕ್ಷರ * scale_ಸ್ಥಾನಾಂತರ    = (ಅಕ್ಷರ *) src0->ದತ್ತಾಂಶ + ತೂಕ_ಗಾತ್ರ;

    //ಒಳಹರಿವು
    size_t               ಒಳಹರಿವು_elem_ಗಾತ್ರ = ಗಾತ್ರ(uint16_t);
    int64_t              ಒಳಹರಿವು_ne[]      = { src1->ne[0], src1->ne[1] };
    size_t               ಒಳಹರಿವು_nb[]      = { ಒಳಹರಿವು_elem_ಗಾತ್ರ, ಒಳಹರಿವು_ne[0] * ಒಳಹರಿವು_elem_ಗಾತ್ರ };
    size_t               ಒಳಹರಿವು_stride    = ಒಳಹರಿವು_ne[0] * ಒಳಹರಿವು_ne[1] * ಒಳಹರಿವು_elem_ಗಾತ್ರ;
    ggml_cann_pool_alloc ಒಳಹರಿವು_alloctor(ctx.pool());
    ಶೂನ್ಯ *               ಒಳಹರಿವು_ಬಫರ್ = src1->ದತ್ತಾಂಶ;

    // case in
    ಆದರೆ (src1->type != GGML_TYPE_F16) {
        acl_tensor_ptr acl_src1_tensor = ggml_cann_create_tensor(src1);
        ಒಳಹರಿವು_ಬಫರ್                   = ಒಳಹರಿವು_alloctor.alloc(ggml_nelements(src1) * ಒಳಹರಿವು_elem_ಗಾತ್ರ);

        int64_t * ಒಳಹರಿವು_cast_ne = src1->ne;
        size_t    ಒಳಹರಿವು_cast_nb[GGML_MAX_DIMS];
        ಒಳಹರಿವು_cast_nb[0] = ಗಾತ್ರ(uint16_t);
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 1; i < GGML_MAX_DIMS; i++) {
            ಒಳಹರಿವು_cast_nb[i] = ಒಳಹರಿವು_cast_nb[i - 1] * ಒಳಹರಿವು_cast_ne[i - 1];
        }

        acl_tensor_ptr acl_ಒಳಹರಿವು_tensor = ggml_cann_create_tensor(ಒಳಹರಿವು_ಬಫರ್, ACL_FLOAT16, ಒಳಹರಿವು_elem_ಗಾತ್ರ,
                                                                  ಒಳಹರಿವು_cast_ne, ಒಳಹರಿವು_cast_nb, GGML_MAX_DIMS);
        aclnn_cast(ctx, acl_src1_tensor.get(), acl_ಒಳಹರಿವು_tensor.get(), ACL_FLOAT16);
    }

    // output
    size_t               output_elem_ಗಾತ್ರ = ಗಾತ್ರ(uint16_t);
    size_t               output_nb[]      = { output_elem_ಗಾತ್ರ, dst->ne[0] * output_elem_ಗಾತ್ರ };
    ggml_cann_pool_alloc output_allocator(ctx.pool());
    ಶೂನ್ಯ *               output_ಬಫರ್ = output_allocator.alloc(ggml_nelements(dst) * output_elem_ಗಾತ್ರ);
    size_t               output_stride = dst->ne[0] * dst->ne[1] * output_elem_ಗಾತ್ರ;

    // aclnn
    int64_t              max_elem_ಗಾತ್ರ = 65535;
    int64_t              ವಿಭಜಿಸು_ಗಾತ್ರ    = (src0->ne[1] / max_elem_ಗಾತ್ರ) + 1;
    ggml_cann_pool_alloc workspace_allocator(ctx.pool());
    ಗಾಗಿ (int64_t n1 = 0; n1 < src1->ne[3]; n1++) {
        ಗಾಗಿ (int64_t c1 = 0; c1 < src1->ne[2]; c1++) {
            int64_t n0 = n1 / (src1->ne[3] / src0->ne[3]);
            int64_t c0 = c1 / (src1->ne[2] / src0->ne[2]);

            int64_t batch1 = (n1 * src1->ne[2]) + c1;
            int64_t batch0 = (n0 * src0->ne[2]) + c0;

            acl_tensor_ptr acl_ಒಳಹರಿವು_tensor = ggml_cann_create_tensor(
                (ಅಕ್ಷರ *) ಒಳಹರಿವು_ಬಫರ್ + batch1 * ಒಳಹರಿವು_stride, ACL_FLOAT16, ಒಳಹರಿವು_elem_ಗಾತ್ರ, ಒಳಹರಿವು_ne, ಒಳಹರಿವು_nb, 2);

            // first split
            int64_t ತೂಕ_ne_ಸ್ಥಾನಾಂತರ = 0;
            int64_t ತೂಕ_ne[2]     = { max_elem_ಗಾತ್ರ > src0->ne[1] ? src0->ne[1] : max_elem_ಗಾತ್ರ, src0->ne[0] };
            int64_t scale_ne_ಸ್ಥಾನಾಂತರ  = 0;
            int64_t scale_ne[2]      = { ತೂಕ_ne[0], ತೂಕ_ne[1] / QK8_0 };
            int64_t output_ne_ಸ್ಥಾನಾಂತರ = 0;
            int64_t output_ne[2]     = { ತೂಕ_ne[0], dst->ne[1] };

            acl_tensor_ptr acl_ತೂಕ_tensor =
                ggml_cann_create_tensor((ಅಕ್ಷರ *) src0->ದತ್ತಾಂಶ + batch0 * ತೂಕ_stride, ggml_cann_type_mapping(type),
                                        ತೂಕ_elem_ಗಾತ್ರ, ತೂಕ_ne, ತೂಕ_nb, 2, ACL_FORMAT_ND, ತೂಕ_ne_ಸ್ಥಾನಾಂತರ);
            acl_tensor_ptr acl_scale_tensor =
                ggml_cann_create_tensor(scale_ಸ್ಥಾನಾಂತರ + batch0 * scale_stride, ACL_FLOAT16, scale_elem_ಗಾತ್ರ, scale_ne,
                                        scale_nb, 2, ACL_FORMAT_ND, scale_ne_ಸ್ಥಾನಾಂತರ);
            acl_tensor_ptr acl_output_tensor =
                ggml_cann_create_tensor((ಅಕ್ಷರ *) output_ಬಫರ್ + batch1 * output_stride, ACL_FLOAT16, output_elem_ಗಾತ್ರ,
                                        output_ne, output_nb, 2, ACL_FORMAT_ND, output_ne_ಸ್ಥಾನಾಂತರ);
            int64_t antiquant_ಗುಂಪು_ಗಾತ್ರ = 0;
            ಆದರೆ (src0->ne[0] > QK8_0) {
                antiquant_ಗುಂಪು_ಗಾತ್ರ = QK8_0;
            }
            GGML_CANN_CALL_ACLNN_OP(ctx, ತೂಕ_Quant_Batch_Matmul_V2, acl_ಒಳಹರಿವು_tensor.get(), acl_ತೂಕ_tensor.get(),
                                    acl_scale_tensor.get(), ಶೂನ್ಯಸೂಚಕ, ಶೂನ್ಯಸೂಚಕ, ಶೂನ್ಯಸೂಚಕ, ಶೂನ್ಯಸೂಚಕ, antiquant_ಗುಂಪು_ಗಾತ್ರ,
                                    acl_output_tensor.get());

            // other splits
            ಗಾಗಿ (int64_t ವಿಭಜಿಸು = 1; ವಿಭಜಿಸು < ವಿಭಜಿಸು_ಗಾತ್ರ; ವಿಭಜಿಸು++) {
                ತೂಕ_ne_ಸ್ಥಾನಾಂತರ += ತೂಕ_elem_ಗಾತ್ರ * ತೂಕ_ne[0] * ತೂಕ_ne[1];
                ತೂಕ_ne[0] =
                    max_elem_ಗಾತ್ರ * (ವಿಭಜಿಸು + 1) > src0->ne[1] ? src0->ne[1] - (max_elem_ಗಾತ್ರ * ವಿಭಜಿಸು) : max_elem_ಗಾತ್ರ;
                scale_ne_ಸ್ಥಾನಾಂತರ += scale_elem_ಗಾತ್ರ * scale_ne[0] * scale_ne[1];
                scale_ne[0] = ತೂಕ_ne[0];
                output_ne_ಸ್ಥಾನಾಂತರ += output_elem_ಗಾತ್ರ * output_ne[0] * output_ne[1];
                output_ne[0] = ತೂಕ_ne[0];

                acl_ತೂಕ_tensor =
                    ggml_cann_create_tensor((ಅಕ್ಷರ *) src0->ದತ್ತಾಂಶ + batch0 * ತೂಕ_stride, ggml_cann_type_mapping(type),
                                            ತೂಕ_elem_ಗಾತ್ರ, ತೂಕ_ne, ತೂಕ_nb, 2, ACL_FORMAT_ND, ತೂಕ_ne_ಸ್ಥಾನಾಂತರ);
                acl_scale_tensor =
                    ggml_cann_create_tensor(scale_ಸ್ಥಾನಾಂತರ + batch0 * scale_stride, ACL_FLOAT16, scale_elem_ಗಾತ್ರ,
                                            scale_ne, scale_nb, 2, ACL_FORMAT_ND, scale_ne_ಸ್ಥಾನಾಂತರ);
                acl_output_tensor =
                    ggml_cann_create_tensor((ಅಕ್ಷರ *) output_ಬಫರ್ + batch1 * output_stride, ACL_FLOAT16,
                                            output_elem_ಗಾತ್ರ, output_ne, output_nb, 2, ACL_FORMAT_ND, output_ne_ಸ್ಥಾನಾಂತರ);
                GGML_CANN_CALL_ACLNN_OP(ctx, ತೂಕ_Quant_Batch_Matmul_V2, acl_ಒಳಹರಿವು_tensor.get(), acl_ತೂಕ_tensor.get(),
                                        acl_scale_tensor.get(), ಶೂನ್ಯಸೂಚಕ, ಶೂನ್ಯಸೂಚಕ, ಶೂನ್ಯಸೂಚಕ, ಶೂನ್ಯಸೂಚಕ, antiquant_ಗುಂಪು_ಗಾತ್ರ,
                                        acl_output_tensor.get());
            }
        }
    }

    // cast out
    ಆದರೆ (dst->type != GGML_TYPE_F16) {
        int64_t * output_cast_ne = dst->ne;
        size_t    output_cast_nb[GGML_MAX_DIMS];
        output_cast_nb[0] = ಗಾತ್ರ(uint16_t);
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 1; i < GGML_MAX_DIMS; i++) {
            output_cast_nb[i] = output_cast_nb[i - 1] * output_cast_ne[i - 1];
        }

        acl_tensor_ptr acl_output_tensor = ggml_cann_create_tensor(output_ಬಫರ್, ACL_FLOAT16, output_elem_ಗಾತ್ರ,
                                                                   output_cast_ne, output_cast_nb, GGML_MAX_DIMS);
        acl_tensor_ptr acl_dst_tensor    = ggml_cann_create_tensor(dst);
        aclnn_cast(ctx, acl_output_tensor.get(), acl_dst_tensor.get(), ggml_cann_type_mapping(dst->type));
    }
}

ಶೂನ್ಯ ggml_cann_mul_mat(ggml_backend_cann_context & ctx, ggml_tensor * dst) {
    ಸ್ಥಿರಾಂಕ ಎಣಿಕೆ ggml_type type = dst->src[0]->type;
    ಆಯ್ಕೆ (type) {
        ಸ್ಥಿತಿ GGML_TYPE_F32:
        ಸ್ಥಿತಿ GGML_TYPE_F16:
#ವ್ಯಾಖ್ಯಾನಿತವಾಗದಿದ್ದರೆ ASCEND_310P
        ಸ್ಥಿತಿ GGML_TYPE_BF16:
#ಆದರೆ_ಅಂತ್ಯ
            ggml_cann_mat_mul_fp(ctx, dst);
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_TYPE_Q4_0:
        ಸ್ಥಿತಿ GGML_TYPE_Q8_0:
            ggml_cann_mul_mat_quant(ctx, dst, type);
            ನಿಲ್ಲಿಸು;
        ಪೂರ್ವನಿಯೋಜಿತ:
            GGML_ABORT("Unsupported type for mul_mat");
            ನಿಲ್ಲಿಸು;
    }
}

/**
 * @brief Rolls the elements of a tensor along a specified dimension.
 *
 * This function rolls the elements of the source tensor `acl_src` by the
 * specified shifts `shifts` along the specified dimensions `dims`, and stores
 * the result in the destination tensor `acl_dst`.
 *
 * @param ctx The context for the CANN backend operations.
 * @param acl_src The source tensor whose elements will be rolled.
 * @param acl_dst The destination tensor where the rolled elements will be
 * stored.
 * @param shifts An array specifying the number of positions by which elements
 * are shifted.
 * @param dims An array specifying the dimensions along which elements are
 * shifted.
 */
ಸ್ಥಿರ ಶೂನ್ಯ aclnn_roll(ggml_backend_cann_context & ctx,
                       aclTensor *                 acl_src,
                       aclTensor *                 acl_dst,
                       int64_t *                   shifts,
                       int64_t *                   dims) {
    acl_int_array_ptr acl_shifts = ggml_cann_create_int_array(shifts, 1);
    acl_int_array_ptr acl_dims   = ggml_cann_create_int_array(dims, 1);
    GGML_CANN_CALL_ACLNN_OP(ctx, Roll, acl_src, acl_shifts.get(), acl_dims.get(), acl_dst);
}

/**
 * @brief Fills specified positions of a tensor with a scalar value.
 *
 * This function fills the positions in the source tensor `acl_src` specified by
 * `index` along the dimension `dim` with the scalar value `value`.
 *
 * @param ctx The context for the CANN backend operations.
 * @param acl_src The source tensor where the positions will be filled.
 * @param dim The dimension along which the positions are specified.
 * @param index An array specifying the positions to be filled.
 * @param index_num The number of positions specified in the index array.
 * @param value The scalar value used to fill the specified positions.
 */
ಸ್ಥಿರ ಶೂನ್ಯ aclnn_ಸೂಚ್ಯಂಕ_fill_tensor(ggml_backend_cann_context & ctx,
                                    aclTensor *                 acl_src,
                                    int64_t                     dim,
                                    int64_t *                   ಸೂಚ್ಯಂಕ,
                                    int64_t                     ಸೂಚ್ಯಂಕ_ಅಂಕೆ,
                                    ದಶಮಾಂಶ                       value) {
    acl_int_array_ptr acl_ಸೂಚ್ಯಂಕ = ggml_cann_create_int_array(ಸೂಚ್ಯಂಕ, ಸೂಚ್ಯಂಕ_ಅಂಕೆ);
    acl_scalar_ptr    acl_ಮೌಲ್ಯ = ggml_cann_create_scalar(&value, acl_ದತ್ತಾಂಶ_Type::ACL_FLOAT);
    GGML_CANN_CALL_ACLNN_OP(ctx, Inplace_ಸೂಚ್ಯಂಕ_Fill_Tensor, acl_src, dim, acl_ಸೂಚ್ಯಂಕ.get(), acl_ಮೌಲ್ಯ.get());
}

/**
 * @brief Initializes and caches all intermediate tensors required for RoPE
 *        (Rotary Position Embedding), including support for Yarn, mRoPE,
 *        i-mRoPE, Neox repeat strategy, independent sectors, frequency factors，
 *        and multi-section rotary groups.
 *
 * This function computes and caches the per-dimension θ coefficients used for
 * Q/K rotary embedding. The cache is shared across layers, and recomputed only
 * when any dependent parameter changes.
 *
 * The function now supports:
 *   - Yarn RoPE extrapolation (via @param corr_dims and @param ext_factor)
 *   - Per-dimension independent sector exponent rules (indep_sects + sections[])
 *   - Multi-section RoPE (mRoPE) index mapping (mrope_used + is_imrope)
 *   - Frequency factor division (src2)
 *   - Neox / normal repeat expansion modes
 *
 * @param ctx                CANN backend context, containing memory pool,
 *                           cached buffers, and runtime stream.
 * @param dst                Destination ggml_tensor whose computation
 *                           depends on RoPE (typically Qcur or Kcur).
 * @param corr_dims          [low, high] Yarn correction range.
 * @param ext_factor         Yarn extrapolation strength. 0 = disabled.
 * @param theta_scale        Base multiplier for per-dimension θ exponent.
 * @param freq_scale         Global frequency scaling factor.
 * @param attn_factor        Optional scaling applied to sin/cos (if needed).
 * @param is_neox            Whether to use Neox-style dimension interleave.
 * @param sections           4-way sector sizes for independent-section RoPE
 *                           and multi-section mRoPE (t/h/w/e).
 * @param mrope_used         Whether to enable multi-section rotary embedding.
 * @param is_imrope          Whether to apply interleaved mRoPE rules.
 * @param indep_sects        Whether each dimension runs independent exponent
 *                           resets based on @p sections.
 */
ಸ್ಥಿರ ಶೂನ್ಯ aclnn_rope_cache_init(ggml_backend_cann_context & ctx,
                                  ggml_tensor *               dst,
                                  ದಶಮಾಂಶ *                     corr_dims,
                                  ದಶಮಾಂಶ                       ext_ಗುಣಕ,
                                  ದಶಮಾಂಶ                       theta_scale,
                                  ದಶಮಾಂಶ                       freq_scale,
                                  ದಶಮಾಂಶ                       attn_ಗುಣಕ,
                                  ತಾರ್ಕಿಕ                        is_neox,
                                  ಪೂರ್ಣಾಂಕ                         sections[4],
                                  ತಾರ್ಕಿಕ                        mrope_ಬಳಸಿದ,
                                  ತಾರ್ಕಿಕ                        is_imrope,
                                  ತಾರ್ಕಿಕ                        indep_sects,
                                  int64_t                     rope_dims) {
    ggml_tensor * src1 = dst->src[1];  // position
    ggml_tensor * src2 = dst->src[2];  // freq_factors

    int64_t theta_scale_ಉದ್ದಳತೆ = rope_dims / 2;
    int64_t position_ಉದ್ದಳತೆ    = dst->ne[2];

    // TODO: check theta_scale_length and position_length.
    ಆದರೆ (src2 == ಶೂನ್ಯಸೂಚಕ && ctx.rope_cache.cached &&
        ctx.rope_cache.equal(theta_scale_ಉದ್ದಳತೆ, position_ಉದ್ದಳತೆ, ext_ಗುಣಕ, theta_scale, freq_scale, attn_ಗುಣಕ,
                             is_neox, indep_sects, mrope_ಬಳಸಿದ, is_imrope, sections)) {
        // use cache.
        ಹಿಂತಿರುಗಿ;
    }

    // Step0: calculate tensor shape.
    int64_t theta_scale_ne[] = { theta_scale_ಉದ್ದಳತೆ, 1, 1, 1 };
    size_t  theta_scale_nb[] = { ಗಾತ್ರ(ದಶಮಾಂಶ), theta_scale_ಉದ್ದಳತೆ * ಗಾತ್ರ(ದಶಮಾಂಶ), theta_scale_ಉದ್ದಳತೆ * ಗಾತ್ರ(ದಶಮಾಂಶ),
                                 theta_scale_ಉದ್ದಳತೆ * ಗಾತ್ರ(ದಶಮಾಂಶ) };

    GGML_ASSERT(src1->type == GGML_TYPE_I32);
    int64_t position_ne[] = { 1, 1, position_ಉದ್ದಳತೆ, 1 };
    size_t  position_nb[] = { ಗಾತ್ರ(int32_t), ಗಾತ್ರ(int32_t), ಗಾತ್ರ(int32_t), ಗಾತ್ರ(int32_t) * position_ಉದ್ದಳತೆ };

    int64_t cache_ne[] = { theta_scale_ಉದ್ದಳತೆ, 1, position_ಉದ್ದಳತೆ, 1 };
    size_t  cache_nb[GGML_MAX_DIMS];
    cache_nb[0] = ಗಾತ್ರ(ದಶಮಾಂಶ);
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 1; i < GGML_MAX_DIMS; i++) {
        cache_nb[i] = cache_nb[i - 1] * cache_ne[i - 1];
    }

    // Step1: Compute the coefficient of theta. During the cache_init process, aside from
    // (1) multiplying by the position,
    // (2) dividing by freq_factors,
    // (3) computing the sine and cosine,
    // the other parameters used in the computation generally do not change in most scenarios.
    // Therefore, we can first compute this part of the result and then cache it.

    // Step1.1: prepare theta_scale exponent. if this exponent updated, should update theta_scale_tensor.
    acl_tensor_ptr acl_theta_scale_tensor;
    ತಾರ್ಕಿಕ           theta_scale_updated = ಸುಳ್ಳು;
    ಆದರೆ (ctx.rope_cache.theta_scale_ಉದ್ದಳತೆ != theta_scale_ಉದ್ದಳತೆ || ctx.rope_cache.theta_scale != theta_scale ||
        ctx.rope_cache.indep_sects != indep_sects) {
        theta_scale_updated = ನಿಜ;
        ಆದರೆ (ctx.rope_cache.theta_scale_exp_host != ಶೂನ್ಯಸೂಚಕ) {
            ಬಿಡುಗಡೆ(ctx.rope_cache.theta_scale_exp_host);
        }
        ctx.rope_cache.theta_scale_exp_host = (ದಶಮಾಂಶ *) ಸ್ಮರಣೆ_ಕೊಡು(theta_scale_ಉದ್ದಳತೆ * ಗಾತ್ರ(ದಶಮಾಂಶ));
        GGML_ASSERT(ctx.rope_cache.theta_scale_exp_host != ಶೂನ್ಯಸೂಚಕ);
        ಆದರೆ (!indep_sects) {
            ctx.rope_cache.theta_scale_exp_host[0] = 1;
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 1; i < theta_scale_ಉದ್ದಳತೆ; i++) {
                ctx.rope_cache.theta_scale_exp_host[i] = ctx.rope_cache.theta_scale_exp_host[i - 1] * theta_scale;
            }
        } ಇಲ್ಲದಿದ್ದರೆ {
            ಪೂರ್ಣಾಂಕ sect_dims = sections[0] + sections[1] + sections[2] + sections[3];
            ಪೂರ್ಣಾಂಕ sec_w     = sections[1] + sections[0];
            ಪೂರ್ಣಾಂಕ sec_e     = sections[2] + sec_w;

            ctx.rope_cache.theta_scale_exp_host[0] = 1;
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 1; i < theta_scale_ಉದ್ದಳತೆ; i++) {
                ಪೂರ್ಣಾಂಕ sector = i % sect_dims;
                ಆದರೆ (sector == 0 || sector == sections[0] || sector == sec_w || sector == sec_e) {
                    ctx.rope_cache.theta_scale_exp_host[i] = 1;
                    ಮುಂದುವರಿ;
                }
                ctx.rope_cache.theta_scale_exp_host[i] = ctx.rope_cache.theta_scale_exp_host[i - 1] * theta_scale;
            }
        }

        ಆದರೆ (ctx.rope_cache.theta_scale_cache != ಶೂನ್ಯಸೂಚಕ) {
            ACL_CHECK(aclrtFree(ctx.rope_cache.theta_scale_cache));
        }
        ACL_CHECK(aclrtMalloc(&ctx.rope_cache.theta_scale_cache, theta_scale_ಉದ್ದಳತೆ * ಗಾತ್ರ(ದಶಮಾಂಶ),
                              ACL_MEM_MALLOC_HUGE_ಮೊದಲ));

        ACL_CHECK(aclrtMemcpyAsync(ctx.rope_cache.theta_scale_cache, theta_scale_ಉದ್ದಳತೆ * ಗಾತ್ರ(ದಶಮಾಂಶ),
                                   ctx.rope_cache.theta_scale_exp_host, theta_scale_ಉದ್ದಳತೆ * ಗಾತ್ರ(ದಶಮಾಂಶ),
                                   ACL_MEMCPY_HOST_ಗಮ್ಯ_DEVICE, ctx.stream()));
    }
    acl_theta_scale_tensor = ggml_cann_create_tensor(ctx.rope_cache.theta_scale_cache, ACL_FLOAT, ಗಾತ್ರ(ದಶಮಾಂಶ),
                                                     theta_scale_ne, theta_scale_nb, 1);

    // Step1.2: prepare rope_yarn_ramp, if this part updated, should update theta_scale_tensor.
    // TODO: acl_yarn_ramp_tensor use rope cache.
    ತಾರ್ಕಿಕ           yarn_ramp_tensor_updated = ಸುಳ್ಳು;
    acl_tensor_ptr acl_yarn_ramp_tensor;
    ಆದರೆ (ext_ಗುಣಕ != 0 && (theta_scale_updated || ctx.rope_cache.theta_scale_ಉದ್ದಳತೆ != theta_scale_ಉದ್ದಳತೆ ||
                            ctx.rope_cache.freq_scale != freq_scale)) {
        yarn_ramp_tensor_updated = ನಿಜ;
        ಆದರೆ (ctx.rope_cache.yarn_ramp_cache != ಶೂನ್ಯಸೂಚಕ) {
            ACL_CHECK(aclrtFree(ctx.rope_cache.yarn_ramp_cache));
        }
        ACL_CHECK(aclrtMalloc(&ctx.rope_cache.yarn_ramp_cache, theta_scale_ಉದ್ದಳತೆ * ಗಾತ್ರ(ದಶಮಾಂಶ),
                              ACL_MEM_MALLOC_HUGE_ಮೊದಲ));
        // -rope_yarn_ramp
        // const float y = (i0 / 2 - low) / MAX(0.001f, high - low);
        // return MIN(1, MAX(0, y)) - 1;
        acl_yarn_ramp_tensor      = ggml_cann_create_tensor(ctx.rope_cache.yarn_ramp_cache, ACL_FLOAT, ಗಾತ್ರ(ದಶಮಾಂಶ),
                                                            theta_scale_ne, theta_scale_nb, 1);
        ದಶಮಾಂಶ          zero_ಮೌಲ್ಯ = 0, one_ಮೌಲ್ಯ = 1;
        ದಶಮಾಂಶ          denom_ಸುರಕ್ಷಿತ_ಮೌಲ್ಯ = MAX(0.001f, corr_dims[1] - corr_dims[0]);
        acl_scalar_ptr ಕೆಳಗಿನ              = ggml_cann_create_scalar(&corr_dims[0], acl_ದತ್ತಾಂಶ_Type::ACL_FLOAT);
        acl_scalar_ptr zero             = ggml_cann_create_scalar(&zero_ಮೌಲ್ಯ, acl_ದತ್ತಾಂಶ_Type::ACL_FLOAT);
        acl_scalar_ptr one              = ggml_cann_create_scalar(&one_ಮೌಲ್ಯ, acl_ದತ್ತಾಂಶ_Type::ACL_FLOAT);
        acl_scalar_ptr denom_ಸುರಕ್ಷಿತ       = ggml_cann_create_scalar(&denom_ಸುರಕ್ಷಿತ_ಮೌಲ್ಯ, acl_ದತ್ತಾಂಶ_Type::ACL_FLOAT);
        acl_scalar_ptr ext_ಗುಣಕ_sc    = ggml_cann_create_scalar(&ext_ಗುಣಕ, acl_ದತ್ತಾಂಶ_Type::ACL_FLOAT);

        aclnn_arange(ctx, acl_yarn_ramp_tensor.get(), 0, theta_scale_ಉದ್ದಳತೆ, 1, theta_scale_ಉದ್ದಳತೆ);
        GGML_CANN_CALL_ACLNN_OP(ctx, InplaceSubs, acl_yarn_ramp_tensor.get(), ಕೆಳಗಿನ.get(), one.get());
        GGML_CANN_CALL_ACLNN_OP(ctx, InplaceDivs, acl_yarn_ramp_tensor.get(), denom_ಸುರಕ್ಷಿತ.get());
        GGML_CANN_CALL_ACLNN_OP(ctx, Inplace_ಮಿತಿ, acl_yarn_ramp_tensor.get(), zero.get(), zero.get());
        GGML_CANN_CALL_ACLNN_OP(ctx, InplaceClampMax, acl_yarn_ramp_tensor.get(), one.get());
        GGML_CANN_CALL_ACLNN_OP(ctx, InplaceSubs, acl_yarn_ramp_tensor.get(), one.get(), one.get());
        GGML_CANN_CALL_ACLNN_OP(ctx, InplaceMuls, acl_yarn_ramp_tensor.get(), ext_ಗುಣಕ_sc.get());

        // theta_interp = freq_scale * theta_extrap;
        // theta = theta_interp * (1 - ramp_mix) + theta_extrap * ramp_mix;
        // theta = freq_scale * theta_extrap * (1 - ramp_mix) + theta_extrap * ramp_mix;
        // theta = freq_scale * theta_extrap - freq_scale * theta_extrap * ramp_mix + theta_extrap * ramp_mix;
        // theta = theta_extrap * (freq_scale - freq_scale * ramp_mix + ramp_mix);
        //
        // we cache (freq_scale - freq_scale * ramp_mix + ramp_mix), Considering that the rope_yarn_ramp here is the inverse
        // cache freq_scale + (freq_scale - 1) * ramp_mix
        ದಶಮಾಂಶ          freq_scale_1    = freq_scale - 1;
        acl_scalar_ptr freq_scale_sc   = ggml_cann_create_scalar(&freq_scale, acl_ದತ್ತಾಂಶ_Type::ACL_FLOAT);
        acl_scalar_ptr freq_scale_1_sc = ggml_cann_create_scalar(&freq_scale_1, acl_ದತ್ತಾಂಶ_Type::ACL_FLOAT);
        GGML_CANN_CALL_ACLNN_OP(ctx, InplaceMuls, acl_yarn_ramp_tensor.get(), freq_scale_1_sc.get());
        GGML_CANN_CALL_ACLNN_OP(ctx, InplaceAdds, acl_yarn_ramp_tensor.get(), freq_scale_sc.get(), one.get());
    } ಇಲ್ಲದಿದ್ದರೆ {
        acl_yarn_ramp_tensor = ggml_cann_create_tensor(ctx.rope_cache.yarn_ramp_cache, ACL_FLOAT, ಗಾತ್ರ(ದಶಮಾಂಶ),
                                                       theta_scale_ne, theta_scale_nb, 1);
    }
    // Step 1.3: update theta_scale_tensor according to ext_factor or freq_scale.
    ಆದರೆ (ext_ಗುಣಕ != 0) {
        ಆದರೆ (theta_scale_updated || yarn_ramp_tensor_updated) {
            theta_scale_updated = ನಿಜ;
            aclnn_mul(ctx, acl_theta_scale_tensor.get(), acl_yarn_ramp_tensor.get());
        }
    } ಇಲ್ಲದಿದ್ದರೆ {
        ಆದರೆ (freq_scale != 1 && (ctx.rope_cache.freq_scale != freq_scale || theta_scale_updated)) {
            theta_scale_updated = ನಿಜ;
            aclnn_muls(ctx, acl_theta_scale_tensor.get(), freq_scale, ಶೂನ್ಯಸೂಚಕ, ನಿಜ);
        }
    }

    // Nothing changed, use cache.
    ಆದರೆ (!theta_scale_updated) {
        acl_theta_scale_tensor = ggml_cann_create_tensor(ctx.rope_cache.theta_scale_cache, ACL_FLOAT, ಗಾತ್ರ(ದಶಮಾಂಶ),
                                                         theta_scale_ne, theta_scale_nb, GGML_MAX_DIMS);
    }

    // Step 1.4: prepare select index if mrope
    acl_tensor_ptr position_select_ಸೂಚ್ಯಂಕ_tensor;
    ಆದರೆ (mrope_ಬಳಸಿದ) {
        ಆದರೆ (ctx.rope_cache.sections[0] != sections[0] || ctx.rope_cache.sections[1] != sections[1] ||
            ctx.rope_cache.sections[2] != sections[2] || ctx.rope_cache.sections[3] != sections[3] ||
            ctx.rope_cache.theta_scale_ಉದ್ದಳತೆ != theta_scale_ಉದ್ದಳತೆ || ctx.rope_cache.is_imrope != is_imrope) {
            ಆದರೆ (ctx.rope_cache.position_select_ಸೂಚ್ಯಂಕ_host != ಶೂನ್ಯಸೂಚಕ) {
                ಬಿಡುಗಡೆ(ctx.rope_cache.position_select_ಸೂಚ್ಯಂಕ_host);
            }
            ctx.rope_cache.position_select_ಸೂಚ್ಯಂಕ_host = (ಪೂರ್ಣಾಂಕ *) ಸ್ಮರಣೆ_ಕೊಡು(theta_scale_ಉದ್ದಳತೆ * ಗಾತ್ರ(ಪೂರ್ಣಾಂಕ));
            GGML_ASSERT(ctx.rope_cache.position_select_ಸೂಚ್ಯಂಕ_host != ಶೂನ್ಯಸೂಚಕ);
            ಪೂರ್ಣಾಂಕ sect_dims = sections[0] + sections[1] + sections[2] + sections[3];
            ಪೂರ್ಣಾಂಕ sec_w     = sections[1] + sections[0];
            ಪೂರ್ಣಾಂಕ sec_e     = sections[2] + sec_w;
            // t,h,w,e
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < theta_scale_ಉದ್ದಳತೆ; i++) {
                ಪೂರ್ಣಾಂಕ sector = i % sect_dims;

                ಆದರೆ (is_imrope) {  // qwen3vl apply interleaved mrope
                    ಆದರೆ (sector % 3 == 1 && sector < 3 * sections[1]) {
                        ctx.rope_cache.position_select_ಸೂಚ್ಯಂಕ_host[i] = 1;
                    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (sector % 3 == 2 && sector < 3 * sections[2]) {
                        ctx.rope_cache.position_select_ಸೂಚ್ಯಂಕ_host[i] = 2;
                    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (sector % 3 == 0 && sector < 3 * sections[0]) {
                        ctx.rope_cache.position_select_ಸೂಚ್ಯಂಕ_host[i] = 0;
                    } ಇಲ್ಲದಿದ್ದರೆ {
                        ctx.rope_cache.position_select_ಸೂಚ್ಯಂಕ_host[i] = 3;
                    }
                } ಇಲ್ಲದಿದ್ದರೆ {
                    ಆದರೆ (sector >= sections[0] && sector < sec_w) {
                        ctx.rope_cache.position_select_ಸೂಚ್ಯಂಕ_host[i] = 1;
                    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (sector >= sec_w && sector < sec_e) {
                        ctx.rope_cache.position_select_ಸೂಚ್ಯಂಕ_host[i] = 2;
                    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (sector >= sec_e) {
                        ctx.rope_cache.position_select_ಸೂಚ್ಯಂಕ_host[i] = 3;
                    } ಇಲ್ಲದಿದ್ದರೆ {
                        ctx.rope_cache.position_select_ಸೂಚ್ಯಂಕ_host[i] = 0;
                    }
                }
            }

            ಆದರೆ (ctx.rope_cache.position_select_ಸೂಚ್ಯಂಕ != ಶೂನ್ಯಸೂಚಕ) {
                ACL_CHECK(aclrtFree(ctx.rope_cache.position_select_ಸೂಚ್ಯಂಕ));
            }
            ACL_CHECK(aclrtMalloc(&ctx.rope_cache.position_select_ಸೂಚ್ಯಂಕ, theta_scale_ಉದ್ದಳತೆ * ಗಾತ್ರ(ಪೂರ್ಣಾಂಕ),
                                  ACL_MEM_MALLOC_HUGE_ಮೊದಲ));

            ACL_CHECK(aclrtMemcpyAsync(ctx.rope_cache.position_select_ಸೂಚ್ಯಂಕ, theta_scale_ಉದ್ದಳತೆ * ಗಾತ್ರ(ಪೂರ್ಣಾಂಕ),
                                       ctx.rope_cache.position_select_ಸೂಚ್ಯಂಕ_host, theta_scale_ಉದ್ದಳತೆ * ಗಾತ್ರ(ಪೂರ್ಣಾಂಕ),
                                       ACL_MEMCPY_HOST_ಗಮ್ಯ_DEVICE, ctx.stream()));
        }

        position_select_ಸೂಚ್ಯಂಕ_tensor = ggml_cann_create_tensor(ctx.rope_cache.position_select_ಸೂಚ್ಯಂಕ, ACL_INT32,
                                                               ಗಾತ್ರ(ಪೂರ್ಣಾಂಕ), theta_scale_ne, theta_scale_nb, 1);
    }

    // Step2: divide by freq_factors
    ggml_cann_pool_alloc freq_fac_res_allocator(ctx.pool());
    ಆದರೆ (src2) {
        freq_fac_res_allocator.alloc(theta_scale_ಉದ್ದಳತೆ * ಗಾತ್ರ(ದಶಮಾಂಶ));
        ಶೂನ್ಯ *         freq_fac_res_ptr = freq_fac_res_allocator.get();
        acl_tensor_ptr acl_freq_factors_tensor =
            ggml_cann_create_tensor(src2->ದತ್ತಾಂಶ, ggml_cann_type_mapping(src2->type), ggml_type_ಗಾತ್ರ(src2->type),
                                    theta_scale_ne, theta_scale_nb, GGML_MAX_DIMS);
        acl_tensor_ptr acl_freq_fac_res_tensor = ggml_cann_create_tensor(freq_fac_res_ptr, ACL_FLOAT, ಗಾತ್ರ(ದಶಮಾಂಶ),
                                                                         theta_scale_ne, theta_scale_nb, GGML_MAX_DIMS);
        aclnn_div(ctx, acl_theta_scale_tensor.get(), acl_freq_factors_tensor.get(), acl_freq_fac_res_tensor.get());
        std::swap(acl_theta_scale_tensor, acl_freq_fac_res_tensor);
    }

    // Step3: prepare position_tensor
    acl_tensor_ptr       acl_position_tensor;
    ggml_cann_pool_alloc mrope_position_acllocator(ctx.pool());
    ಆದರೆ (mrope_ಬಳಸಿದ) {
        // Step3.1: select current position;
        // position :
        // pos1: [[0, 1 ,2 ,3 ],
        // pos2:  [4, 5 ,6 ,7 ],
        // pos3:  [8, 9 ,10,11],
        // pos4:  [12,13,14,15] ]
        //
        // select index = [0, 1, 2, 2, 1, 0]
        //
        // selected_tensor:
        // [[0, 1 ,2 ,3 ],
        //  [4, 5 ,6 ,7 ],
        //  [8, 9 ,10,11],
        //  [8, 9 ,10,11],
        //  [4, 5 ,6 ,7 ],
        //  [0, 1 ,2 ,3 ]]
        //
        // transpose, from [seq_len:dims] to [dims:seq_len]
        // [0, 4, 8 ,8 ,4, 0],
        // [1, 5, 9, 9, 5, 1],
        // [2, 6, 10,10,6 ,2],
        // [3, 7, 11,11,7 3 ]]
        //
        // multipy by theta_scale_tensor
        // [theta_scale^0, theta_scale^1, ..., theta_scale ^ n]

        int64_t        mrope_position_ne[] = { position_ಉದ್ದಳತೆ, 4 };
        size_t         mrope_position_nb[] = { ಗಾತ್ರ(ಪೂರ್ಣಾಂಕ), position_ಉದ್ದಳತೆ * ಗಾತ್ರ(ಪೂರ್ಣಾಂಕ) };
        acl_tensor_ptr mrope_position =
            ggml_cann_create_tensor(src1->ದತ್ತಾಂಶ, ggml_cann_type_mapping(src1->type), ggml_type_ಗಾತ್ರ(src1->type),
                                    mrope_position_ne, mrope_position_nb, 2);

        // selected position tensor's shape is a transpose of cache tensor.
        int64_t selected_position_ne[] = { position_ಉದ್ದಳತೆ, theta_scale_ಉದ್ದಳತೆ };
        size_t  selected_position_nb[] = { ಗಾತ್ರ(ದಶಮಾಂಶ), position_ಉದ್ದಳತೆ * ಗಾತ್ರ(ದಶಮಾಂಶ) };
        mrope_position_acllocator.alloc(theta_scale_ಉದ್ದಳತೆ * position_ಉದ್ದಳತೆ * ಗಾತ್ರ(ದಶಮಾಂಶ));
        ಶೂನ್ಯ * mrope_position_ಬಫರ್ = mrope_position_acllocator.get();
        acl_position_tensor =
            ggml_cann_create_tensor(mrope_position_ಬಫರ್, ggml_cann_type_mapping(src1->type),
                                    ggml_type_ಗಾತ್ರ(src1->type), selected_position_ne, selected_position_nb, 2);
        GGML_CANN_CALL_ACLNN_OP(ctx, ಸೂಚ್ಯಂಕ_Select, mrope_position.get(), 0, position_select_ಸೂಚ್ಯಂಕ_tensor.get(),
                                acl_position_tensor.get());

        // transpose
        int64_t transposed_ne[] = { position_ಉದ್ದಳತೆ, 1, theta_scale_ಉದ್ದಳತೆ, 1 };
        size_t  transposed_nb[GGML_MAX_DIMS];
        transposed_nb[0] = ಗಾತ್ರ(ದಶಮಾಂಶ);
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 1; i < GGML_MAX_DIMS; i++) {
            transposed_nb[i] = transposed_nb[i - 1] * transposed_ne[i - 1];
        }

        std::swap(transposed_ne[0], transposed_ne[2]);
        std::swap(transposed_nb[0], transposed_nb[2]);

        acl_position_tensor =
            ggml_cann_create_tensor(mrope_position_ಬಫರ್, ggml_cann_type_mapping(src1->type),
                                    ggml_type_ಗಾತ್ರ(src1->type), transposed_ne, transposed_nb, GGML_MAX_DIMS);

    } ಇಲ್ಲದಿದ್ದರೆ {
        // auto bcast.
        acl_position_tensor =
            ggml_cann_create_tensor(src1->ದತ್ತಾಂಶ, ggml_cann_type_mapping(src1->type), ggml_type_ಗಾತ್ರ(src1->type),
                                    position_ne, position_nb, GGML_MAX_DIMS);
    }

    // Step4: multiply by the position
    int64_t              theta_ಉದ್ದಳತೆ = theta_scale_ಉದ್ದಳತೆ * position_ಉದ್ದಳತೆ;
    ggml_cann_pool_alloc theta_allocator(ctx.pool(), theta_ಉದ್ದಳತೆ * ಗಾತ್ರ(ದಶಮಾಂಶ));
    ಶೂನ್ಯ *               theta_ಬಫರ್ = theta_allocator.get();

    acl_tensor_ptr acl_theta_tensor =
        ggml_cann_create_tensor(theta_ಬಫರ್, ACL_FLOAT, ಗಾತ್ರ(ದಶಮಾಂಶ), cache_ne, cache_nb, GGML_MAX_DIMS);
    aclnn_mul(ctx, acl_position_tensor.get(), acl_theta_scale_tensor.get(), acl_theta_tensor.get());

    // Step5: calculate sin cos.
    // init sin_repeat && cos_repeat, only to accelerate first layer on each device
    ಆದರೆ (position_ಉದ್ದಳತೆ > ctx.rope_cache.position_ಉದ್ದಳತೆ) {
        ctx.rope_cache.position_ಉದ್ದಳತೆ = position_ಉದ್ದಳತೆ;
        ಆದರೆ (ctx.rope_cache.sin_cache != ಶೂನ್ಯಸೂಚಕ) {
            ACL_CHECK(aclrtFree(ctx.rope_cache.sin_cache));
        }
        ಆದರೆ (ctx.rope_cache.cos_cache != ಶೂನ್ಯಸೂಚಕ) {
            ACL_CHECK(aclrtFree(ctx.rope_cache.cos_cache));
        }
        int64_t repeat_theta_ಉದ್ದಳತೆ = theta_scale_ಉದ್ದಳತೆ * position_ಉದ್ದಳತೆ * 2;
        ACL_CHECK(
            aclrtMalloc(&ctx.rope_cache.sin_cache, repeat_theta_ಉದ್ದಳತೆ * ಗಾತ್ರ(ದಶಮಾಂಶ), ACL_MEM_MALLOC_HUGE_ಮೊದಲ));
        ACL_CHECK(
            aclrtMalloc(&ctx.rope_cache.cos_cache, repeat_theta_ಉದ್ದಳತೆ * ಗಾತ್ರ(ದಶಮಾಂಶ), ACL_MEM_MALLOC_HUGE_ಮೊದಲ));
    }

    // sin/cos
    ggml_cann_pool_alloc sin_allocator(ctx.pool(), theta_ಉದ್ದಳತೆ * ಗಾತ್ರ(ದಶಮಾಂಶ));
    ಶೂನ್ಯ *               sin_ಬಫರ್ = sin_allocator.get();
    acl_tensor_ptr       acl_sin_tensor =
        ggml_cann_create_tensor(sin_ಬಫರ್, ACL_FLOAT, ಗಾತ್ರ(ದಶಮಾಂಶ), cache_ne, cache_nb, GGML_MAX_DIMS, ACL_FORMAT_ND);
    aclnn_sin(ctx, acl_theta_tensor.get(), acl_sin_tensor.get());

    ggml_cann_pool_alloc cos_allocator(ctx.pool(), theta_ಉದ್ದಳತೆ * ಗಾತ್ರ(ದಶಮಾಂಶ));
    ಶೂನ್ಯ *               cos_ಬಫರ್ = cos_allocator.get();
    acl_tensor_ptr       acl_cos_tensor =
        ggml_cann_create_tensor(cos_ಬಫರ್, ACL_FLOAT, ಗಾತ್ರ(ದಶಮಾಂಶ), cache_ne, cache_nb, GGML_MAX_DIMS, ACL_FORMAT_ND);
    aclnn_cos(ctx, acl_theta_tensor.get(), acl_cos_tensor.get());

    ಆದರೆ (ext_ಗುಣಕ != 0) {
        attn_ಗುಣಕ *= 1.0f + 0.1f * logf(1.0f / freq_scale);
    }

    // Step 5: multiply by attn_factor
    ಆದರೆ (attn_ಗುಣಕ != 1) {
        aclnn_muls(ctx, acl_sin_tensor.get(), attn_ಗುಣಕ, ಶೂನ್ಯಸೂಚಕ, ನಿಜ);
        aclnn_muls(ctx, acl_cos_tensor.get(), attn_ಗುಣಕ, ಶೂನ್ಯಸೂಚಕ, ನಿಜ);
    }

    int64_t sin_reshape_ne[4] = { rope_dims, 1, dst->ne[2], 1 };
    size_t  sin_reshape_nb[GGML_MAX_DIMS];
    sin_reshape_nb[0] = ಗಾತ್ರ(ದಶಮಾಂಶ);
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 1; i < GGML_MAX_DIMS; i++) {
        sin_reshape_nb[i] = sin_reshape_nb[i - 1] * sin_reshape_ne[i - 1];
    }
    acl_tensor_ptr acl_sin_repeat_tensor = ggml_cann_create_tensor(ctx.rope_cache.sin_cache, ACL_FLOAT, ಗಾತ್ರ(ದಶಮಾಂಶ),
                                                                   sin_reshape_ne, sin_reshape_nb, GGML_MAX_DIMS);
    acl_tensor_ptr acl_cos_repeat_tensor = ggml_cann_create_tensor(ctx.rope_cache.cos_cache, ACL_FLOAT, ಗಾತ್ರ(ದಶಮಾಂಶ),
                                                                   sin_reshape_ne, sin_reshape_nb, GGML_MAX_DIMS);

    // Step 6: repeat
    ಆದರೆ (is_neox) {
        // [sinθ1, sinθ1, sinθ2, sinθ2, ..., sinθn, sinθn]
        int64_t repeatsArray[] = { 1, 1, 1, 2 };
        aclnn_repeat(ctx, acl_sin_tensor.get(), acl_sin_repeat_tensor.get(), repeatsArray);
        aclnn_repeat(ctx, acl_cos_tensor.get(), acl_cos_repeat_tensor.get(), repeatsArray);
    } ಇಲ್ಲದಿದ್ದರೆ {
        int64_t ಅಂಕೆ_repeats = 2;
        int64_t dim         = 3;
        int64_t output_ಗಾತ್ರ = theta_scale_ಉದ್ದಳತೆ * ಅಂಕೆ_repeats;
        // [sinθ1, sinθ2, ..., sinθn, sinθ1, sinθ2, ..., sinθn]
        aclnn_repeat_interleave(ctx, acl_sin_tensor.get(), acl_sin_repeat_tensor.get(), dim, ಅಂಕೆ_repeats, output_ಗಾತ್ರ);
        aclnn_repeat_interleave(ctx, acl_cos_tensor.get(), acl_cos_repeat_tensor.get(), dim, ಅಂಕೆ_repeats, output_ಗಾತ್ರ);
    }

    // Update cached value.
    ctx.rope_cache.cached = ನಿಜ;
    ctx.rope_cache.set(theta_scale_ಉದ್ದಳತೆ, position_ಉದ್ದಳತೆ, ext_ಗುಣಕ, theta_scale, freq_scale, attn_ಗುಣಕ, is_neox,
                       indep_sects, mrope_ಬಳಸಿದ, is_imrope, sections);
}

#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ __cplusplus
ಬಾಹ್ಯ "C" {
#ಆದರೆ_ಅಂತ್ಯ
aclnn_ಸ್ಥಿತಿ aclnn_Rotary_Position_Embedding_Get_Workspace_ಗಾತ್ರ(ಸ್ಥಿರಾಂಕ aclTensor * x,
                                                         ಸ್ಥಿರಾಂಕ aclTensor * ಕೋಜ್ಯಾ,
                                                         ಸ್ಥಿರಾಂಕ aclTensor * ಜ್ಯಾ,
                                                         int64_t           mode,
                                                         ಸ್ಥಿರಾಂಕ aclTensor * y_ಹೊರಹೋಗು,
                                                         uint64_t *        workspace_ಗಾತ್ರ,
                                                         aclOpExecutor **  executor);
aclnn_ಸ್ಥಿತಿ aclnnRotaryPositionEmbedding(ಶೂನ್ಯ *          workspace,
                                         uint64_t        workspace_ಗಾತ್ರ,
                                         aclOpExecutor * executor,
                                         aclrt_ಹರಿವು     stream);
#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ __cplusplus
}
#ಆದರೆ_ಅಂತ್ಯ

ಶೂನ್ಯ ggml_cann_rope(ggml_backend_cann_context & ctx, ggml_tensor * dst) {
    ggml_tensor * src0 = dst->src[0];  //ಒಳಹರಿವು

    // param
    ದಶಮಾಂಶ     freq_ತಳ, freq_scale, ext_ಗುಣಕ, attn_ಗುಣಕ, beta_fast, beta_slow;
    ಪೂರ್ಣಾಂಕ       sections[4];
    // const int n_past     = ((int32_t *) dst->op_params)[0];
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ n_dims     = ((int32_t *) dst->op_params)[1];
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ mode       = ((int32_t *) dst->op_params)[2];
    // const int n_ctx      = ((int32_t *) dst->op_params)[3];
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ n_ctx_orig = ((int32_t *) dst->op_params)[4];

    GGML_TENSOR_UNARY_OP_LOCALS

    ಸ್ಮರಣೆ_ನಕಲು(&freq_ತಳ, (int32_t *) dst->op_params + 5, ಗಾತ್ರ(ದಶಮಾಂಶ));
    ಸ್ಮರಣೆ_ನಕಲು(&freq_scale, (int32_t *) dst->op_params + 6, ಗಾತ್ರ(ದಶಮಾಂಶ));
    ಸ್ಮರಣೆ_ನಕಲು(&ext_ಗುಣಕ, (int32_t *) dst->op_params + 7, ಗಾತ್ರ(ದಶಮಾಂಶ));
    ಸ್ಮರಣೆ_ನಕಲು(&attn_ಗುಣಕ, (int32_t *) dst->op_params + 8, ಗಾತ್ರ(ದಶಮಾಂಶ));
    ಸ್ಮರಣೆ_ನಕಲು(&beta_fast, (int32_t *) dst->op_params + 9, ಗಾತ್ರ(ದಶಮಾಂಶ));
    ಸ್ಮರಣೆ_ನಕಲು(&beta_slow, (int32_t *) dst->op_params + 10, ಗಾತ್ರ(ದಶಮಾಂಶ));
    ಸ್ಮರಣೆ_ನಕಲು(&sections, (int32_t *) dst->op_params + 11, ಗಾತ್ರ(ಪೂರ್ಣಾಂಕ) * 4);

    GGML_ASSERT(n_dims % 2 == 0);
    GGML_ASSERT(n_dims <= ne00);

    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ theta_scale = powf(freq_ತಳ, -2.0f / n_dims);

    ದಶಮಾಂಶ corr_dims[2];
    ggml_rope_yarn_corr_dims(n_dims, n_ctx_orig, freq_ತಳ, beta_fast, beta_slow, corr_dims);

    ತಾರ್ಕಿಕ       is_neox    = mode & GGML_ROPE_TYPE_NEOX;
    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ is_imrope  = mode == GGML_ROPE_TYPE_IMROPE;  // qwen3vl apply interleaved mrope
    // mrope_used means the GGML_ROPE_TYPE_MROPE bit is set.
    // Note: this bit is also set for imrope and some vision modes,
    // so mrope_used does NOT exclusively indicate pure mrope.
    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ mrope_ಬಳಸಿದ = mode & GGML_ROPE_TYPE_MROPE;
    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ is_vision  = mode == GGML_ROPE_TYPE_VISION;

    ಆದರೆ (mrope_ಬಳಸಿದ) {
        GGML_ASSERT(sections[0] > 0 || sections[1] > 0 || sections[2] > 0);
    }

    ಆದರೆ (is_vision) {
        GGML_ASSERT(n_dims == ne0 / 2);
    }

    ಆದರೆ (is_imrope || mrope_ಬಳಸಿದ) {
        is_neox = ನಿಜ;
    }

    int64_t rope_dims = n_dims;

    //Our current RotaryPositionEmbedding does not support the VISION mode,
    //but essentially it only modifies theta_base in mrope,
    //then repeats it at the end in the same way as is_neox.
    //In fact, RoPE is still applied across all dimensions.
    ಆದರೆ (is_vision) {
        rope_dims = src0->ne[0];
    }
    int64_t ಬಾಲ_dims = ne00 - rope_dims;
    ತಾರ್ಕಿಕ    has_ಬಾಲ  = ಬಾಲ_dims > 0;

    // init ctx.rope_cos/rope_sin cache
    aclnn_rope_cache_init(ctx, dst, corr_dims, ext_ಗುಣಕ, theta_scale, freq_scale, attn_ಗುಣಕ, is_neox, sections,
                          mrope_ಬಳಸಿದ, is_imrope, is_vision, rope_dims);

    // Cache is generated with ne00 dimensions, so we use ne00 for reshape
    int64_t sin_reshape_ne[4] = { rope_dims, 1, ne02, 1 };
    size_t  sin_reshape_nb[GGML_MAX_DIMS];
    sin_reshape_nb[0] = ಗಾತ್ರ(ದಶಮಾಂಶ);
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 1; i < GGML_MAX_DIMS; i++) {
        sin_reshape_nb[i] = sin_reshape_nb[i - 1] * sin_reshape_ne[i - 1];
    }
    acl_tensor_ptr acl_sin_reshape_tensor = ggml_cann_create_tensor(ctx.rope_cache.sin_cache, ACL_FLOAT, ಗಾತ್ರ(ದಶಮಾಂಶ),
                                                                    sin_reshape_ne, sin_reshape_nb, GGML_MAX_DIMS);
    acl_tensor_ptr acl_cos_reshape_tensor = ggml_cann_create_tensor(ctx.rope_cache.cos_cache, ACL_FLOAT, ಗಾತ್ರ(ದಶಮಾಂಶ),
                                                                    sin_reshape_ne, sin_reshape_nb, GGML_MAX_DIMS);

    acl_tensor_ptr acl_src = ggml_cann_create_tensor(src0);
    acl_tensor_ptr acl_dst = ggml_cann_create_tensor(dst);
#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ ASCEND_310P
    // Special ROPE operation for 310P

    // roll input
    ಶೂನ್ಯ *               ಒಳಹರಿವು_roll_ಬಫರ್;
    acl_tensor_ptr       acl_minus_one_tensor;
    ಶೂನ್ಯ *               minus_one_scale_ಬಫರ್ = ಶೂನ್ಯಸೂಚಕ;
    ggml_cann_pool_alloc roll_allocator(ctx.pool(), ggml_nbytes(src0));
    ggml_cann_pool_alloc minus_one_scale_allocator(ctx.pool(), ಗಾತ್ರ(ದಶಮಾಂಶ) * src0->ne[0]);
    ಆದರೆ (!is_neox) {
        // roll input: [q0,q1,q2,q3,...] -> [q1,q0,q3,q2,...]
        ಒಳಹರಿವು_roll_ಬಫರ್        = roll_allocator.get();
        int64_t ಒಳಹರಿವು_roll_ne[4] = { 2, src0->ne[1] * (src0->ne[0] / 2), src0->ne[2], src0->ne[3] };
        size_t  ಒಳಹರಿವು_roll_nb[GGML_MAX_DIMS];
        ಒಳಹರಿವು_roll_nb[0] = ggml_type_ಗಾತ್ರ(src0->type);
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 1; i < GGML_MAX_DIMS; i++) {
            ಒಳಹರಿವು_roll_nb[i] = ಒಳಹರಿವು_roll_nb[i - 1] * ಒಳಹರಿವು_roll_ne[i - 1];
        }
        acl_tensor_ptr acl_ಒಳಹರಿವು_roll_tensor =
            ggml_cann_create_tensor(ಒಳಹರಿವು_roll_ಬಫರ್, ggml_cann_type_mapping(src0->type), ggml_type_ಗಾತ್ರ(src0->type),
                                    ಒಳಹರಿವು_roll_ne, ಒಳಹರಿವು_roll_nb, GGML_MAX_DIMS);
        acl_tensor_ptr acl_ಒಳಹರಿವು_tensor =
            ggml_cann_create_tensor(src0->ದತ್ತಾಂಶ, ggml_cann_type_mapping(src0->type), ggml_type_ಗಾತ್ರ(src0->type),
                                    ಒಳಹರಿವು_roll_ne, ಒಳಹರಿವು_roll_nb, GGML_MAX_DIMS);

        int64_t shifts[] = { 1 };
        int64_t dims[]   = { 3 };
        aclnn_roll(ctx, acl_ಒಳಹರಿವು_tensor.get(), acl_ಒಳಹರಿವು_roll_tensor.get(), shifts, dims);

        // init [-1, 1, -1, 1, ...]
        minus_one_scale_ಬಫರ್ = minus_one_scale_allocator.get();

        int64_t minus_one_ne[4] = { src0->ne[0], 1, 1, 1 };
        size_t  minus_one_nb[GGML_MAX_DIMS];
        minus_one_nb[0] = ಗಾತ್ರ(ದಶಮಾಂಶ);
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 1; i < GGML_MAX_DIMS; i++) {
            minus_one_nb[i] = minus_one_nb[i - 1] * minus_one_ne[i - 1];
        }
        acl_minus_one_tensor = aclnn_ಮೌಲ್ಯಗಳು(ctx, minus_one_scale_ಬಫರ್, ಗಾತ್ರ(ದಶಮಾಂಶ) * src0->ne[0], minus_one_ne,
                                            GGML_MAX_DIMS, ACL_FLOAT, ಗಾತ್ರ(ದಶಮಾಂಶ), 1);
        int64_t   dim        = 3;
        int64_t * ಸೂಚ್ಯಂಕ      = ಹೊಸ int64_t[src0->ne[0]];
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < src0->ne[0]; i++) {
            ಸೂಚ್ಯಂಕ[i] = i / 2 * 2;
        }
        int64_t ಸೂಚ್ಯಂಕ_ಅಂಕೆ = src0->ne[0];
        ದಶಮಾಂಶ   value     = -1;
        aclnn_ಸೂಚ್ಯಂಕ_fill_tensor(ctx, acl_minus_one_tensor.get(), dim, ಸೂಚ್ಯಂಕ, ಸೂಚ್ಯಂಕ_ಅಂಕೆ, value);
    } ಇಲ್ಲದಿದ್ದರೆ {
        // roll input: [q0,q1,q2,...] ->
        // [q_half,q_half+1,...,q_end,q0,q1,...q_half-1]
        ಒಳಹರಿವು_roll_ಬಫರ್ = roll_allocator.get();
        acl_tensor_ptr acl_ಒಳಹರಿವು_roll_tensor =
            ggml_cann_create_tensor(ಒಳಹರಿವು_roll_ಬಫರ್, ggml_cann_type_mapping(src0->type), ggml_type_ಗಾತ್ರ(src0->type),
                                    src0->ne, src0->nb, GGML_MAX_DIMS);
        acl_tensor_ptr acl_ಒಳಹರಿವು_tensor = ggml_cann_create_tensor(src0);

        int64_t shifts[] = { src0->ne[0] / 2 };
        int64_t dims[]   = { 3 };
        aclnn_roll(ctx, acl_ಒಳಹರಿವು_tensor.get(), acl_ಒಳಹರಿವು_roll_tensor.get(), shifts, dims);

        // init [-1, -1, -1, 1, 1，1，...]
        minus_one_scale_ಬಫರ್  = minus_one_scale_allocator.get();
        int64_t minus_one_ne[4] = { src0->ne[0], 1, 1, 1 };
        size_t  minus_one_nb[GGML_MAX_DIMS];
        minus_one_nb[0] = ಗಾತ್ರ(ದಶಮಾಂಶ);
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 1; i < GGML_MAX_DIMS; i++) {
            minus_one_nb[i] = minus_one_nb[i - 1] * minus_one_ne[i - 1];
        }
        acl_minus_one_tensor     = aclnn_ಮೌಲ್ಯಗಳು(ctx, minus_one_scale_ಬಫರ್, ಗಾತ್ರ(ದಶಮಾಂಶ) * src0->ne[0], minus_one_ne,
                                                GGML_MAX_DIMS, ACL_FLOAT, ಗಾತ್ರ(ದಶಮಾಂಶ), 1);
        // -1 * first half
        int64_t ಮೊದಲ_half_ne[4] = { src0->ne[0] / 2, 1, 1, 1 };
        size_t  ಮೊದಲ_half_nb[GGML_MAX_DIMS];
        ಮೊದಲ_half_nb[0] = ಗಾತ್ರ(ದಶಮಾಂಶ);
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 1; i < GGML_MAX_DIMS; i++) {
            ಮೊದಲ_half_nb[i] = ಮೊದಲ_half_nb[i - 1] * ಮೊದಲ_half_ne[i - 1];
        }
        acl_tensor_ptr acl_ಮೊದಲ_half_tensor = ggml_cann_create_tensor(minus_one_scale_ಬಫರ್, ACL_FLOAT, ಗಾತ್ರ(ದಶಮಾಂಶ),
                                                                       ಮೊದಲ_half_ne, ಮೊದಲ_half_nb, GGML_MAX_DIMS);
        ತಾರ್ಕಿಕ           inplace               = ನಿಜ;
        ದಶಮಾಂಶ          scale                 = -1;
        aclnn_muls(ctx, acl_ಮೊದಲ_half_tensor.get(), scale, ಶೂನ್ಯಸೂಚಕ, inplace);
    }

    // TODO: n_dims < ne0
    GGML_ASSERT(n_dims == src0->ne[0]);

    // input * scale
    ggml_cann_pool_alloc roll_mul_scale_allocator(ctx.pool(), ggml_nbytes(src0));
    ಶೂನ್ಯ *               ಒಳಹರಿವು_roll_mul_scale_ಬಫರ್ = roll_mul_scale_allocator.get();
    size_t               ಒಳಹರಿವು_nb[GGML_MAX_DIMS];
    ಒಳಹರಿವು_nb[0] = ggml_type_ಗಾತ್ರ(src0->type);
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 1; i < GGML_MAX_DIMS; i++) {
        ಒಳಹರಿವು_nb[i] = ಒಳಹರಿವು_nb[i - 1] * src0->ne[i - 1];
    }
    acl_tensor_ptr acl_ಒಳಹರಿವು_roll_mul_scale_tensor =
        ggml_cann_create_tensor(ಒಳಹರಿವು_roll_mul_scale_ಬಫರ್, ggml_cann_type_mapping(src0->type),
                                ggml_type_ಗಾತ್ರ(src0->type), src0->ne, ಒಳಹರಿವು_nb, GGML_MAX_DIMS);
    acl_tensor_ptr acl_ಒಳಹರಿವು_roll_reshape_tensor =
        ggml_cann_create_tensor(ಒಳಹರಿವು_roll_ಬಫರ್, ggml_cann_type_mapping(src0->type), ggml_type_ಗಾತ್ರ(src0->type),
                                src0->ne, ಒಳಹರಿವು_nb, GGML_MAX_DIMS);

    aclnn_mul(ctx, acl_ಒಳಹರಿವು_roll_reshape_tensor.get(), acl_minus_one_tensor.get(),
              acl_ಒಳಹರಿವು_roll_mul_scale_tensor.get());

    // output
    ಶೂನ್ಯ * output_fp32_ಬಫರ್;
    ಆದರೆ (src0->type == GGML_TYPE_F32) {
        aclnn_mul(ctx, acl_src.get(), acl_cos_reshape_tensor.get());
        aclnn_mul(ctx, acl_ಒಳಹರಿವು_roll_mul_scale_tensor.get(), acl_sin_reshape_tensor.get());
        aclnn_ಕೂಡಿಸು(ctx, acl_src.get(), acl_ಒಳಹರಿವು_roll_mul_scale_tensor.get(), acl_dst.get());
        // TODO: ne0 != n_dims in mode2
    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (src0->type == GGML_TYPE_F16) {
        size_t ಒಳಹರಿವು_fp32_nb[GGML_MAX_DIMS];
        ಒಳಹರಿವು_fp32_nb[0] = ಗಾತ್ರ(ದಶಮಾಂಶ);
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 1; i < GGML_MAX_DIMS; i++) {
            ಒಳಹರಿವು_fp32_nb[i] = ಒಳಹರಿವು_fp32_nb[i - 1] * dst->ne[i - 1];
        }
        ggml_cann_pool_alloc fp32_allocator1(ctx.pool(), ggml_nelements(dst) * ಗಾತ್ರ(ದಶಮಾಂಶ));
        ಶೂನ್ಯ *               ಒಳಹರಿವು_fp32_buffer1 = fp32_allocator1.get();
        acl_tensor_ptr       ಒಳಹರಿವು_fp32_tensor1 = ggml_cann_create_tensor(ಒಳಹರಿವು_fp32_buffer1, ACL_FLOAT, ಗಾತ್ರ(ದಶಮಾಂಶ),
                                                                          dst->ne, ಒಳಹರಿವು_fp32_nb, GGML_MAX_DIMS);
        ggml_cann_pool_alloc fp32_allocator2(ctx.pool(), ggml_nelements(dst) * ಗಾತ್ರ(ದಶಮಾಂಶ));
        ಶೂನ್ಯ *               ಒಳಹರಿವು_fp32_buffer2 = fp32_allocator2.get();
        acl_tensor_ptr       ಒಳಹರಿವು_fp32_tensor2 = ggml_cann_create_tensor(ಒಳಹರಿವು_fp32_buffer2, ACL_FLOAT, ಗಾತ್ರ(ದಶಮಾಂಶ),
                                                                          dst->ne, ಒಳಹರಿವು_fp32_nb, GGML_MAX_DIMS);

        ggml_cann_pool_alloc fp32_allocator(ctx.pool(), ggml_nelements(dst) * ಗಾತ್ರ(ದಶಮಾಂಶ));
        output_fp32_ಬಫರ್                = fp32_allocator.get();
        acl_tensor_ptr output_fp32_tensor = ggml_cann_create_tensor(output_fp32_ಬಫರ್, ACL_FLOAT, ಗಾತ್ರ(ದಶಮಾಂಶ),
                                                                    dst->ne, ಒಳಹರಿವು_fp32_nb, GGML_MAX_DIMS);
        aclnn_mul(ctx, acl_src.get(), acl_cos_reshape_tensor.get(), ಒಳಹರಿವು_fp32_tensor1.get());
        aclnn_mul(ctx, acl_ಒಳಹರಿವು_roll_mul_scale_tensor.get(), acl_sin_reshape_tensor.get(), ಒಳಹರಿವು_fp32_tensor2.get());
        aclnn_ಕೂಡಿಸು(ctx, ಒಳಹರಿವು_fp32_tensor1.get(), ಒಳಹರಿವು_fp32_tensor2.get(), output_fp32_tensor.get());
        aclnn_cast(ctx, output_fp32_tensor.get(), acl_dst.get(), ACL_FLOAT16);
    }
    ಹಿಂತಿರುಗಿ;
#ಆದರೆ_ಅಂತ್ಯ
    int64_t acl_ಬಹುಲಕ = is_neox ? 0 : 1;

    // Pre-define head and tail dimensions for reuse
    int64_t ತಲೆ_ne[GGML_MAX_DIMS] = { rope_dims, ne01, ne02, ne03 };
    int64_t ಬಾಲ_ne[GGML_MAX_DIMS] = { ಬಾಲ_dims, ne01, ne02, ne03 };

    // Step 1: Prepare trans tensors for F16 type conversion to F32 if needed
    ತಾರ್ಕಿಕ                 src_dst_need_trans = ಸುಳ್ಳು;
    ggml_cann_pool_alloc src_trans_allocator(ctx.pool());
    ggml_cann_pool_alloc dst_trans_allocator(ctx.pool());
    acl_tensor_ptr       acl_src_trans_tensor;
    acl_tensor_ptr       acl_dst_trans_tensor;
    ಶೂನ್ಯ *               src_trans_ಬಫರ್ = ಶೂನ್ಯಸೂಚಕ;
    ಶೂನ್ಯ *               dst_trans_ಬಫರ್ = ಶೂನ್ಯಸೂಚಕ;
    size_t               src_dst_trans_nb[GGML_MAX_DIMS];
    ಆದರೆ (src0->type == GGML_TYPE_F16) {
        src_dst_need_trans = ನಿಜ;
        src_trans_ಬಫರ್   = src_trans_allocator.alloc(ggml_nelements(src0) * ಗಾತ್ರ(ದಶಮಾಂಶ));
        dst_trans_ಬಫರ್   = dst_trans_allocator.alloc(ggml_nelements(dst) * ಗಾತ್ರ(ದಶಮಾಂಶ));

        src_dst_trans_nb[0] = ಗಾತ್ರ(ದಶಮಾಂಶ);
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 1; i < GGML_MAX_DIMS; i++) {
            src_dst_trans_nb[i] = src_dst_trans_nb[i - 1] * src0->ne[i - 1];
        }
        acl_src_trans_tensor = ggml_cann_create_tensor(src_trans_ಬಫರ್, ACL_FLOAT, ಗಾತ್ರ(ದಶಮಾಂಶ), src0->ne,
                                                       src_dst_trans_nb, GGML_MAX_DIMS);
        acl_dst_trans_tensor = ggml_cann_create_tensor(dst_trans_ಬಫರ್, ACL_FLOAT, ಗಾತ್ರ(ದಶಮಾಂಶ), dst->ne,
                                                       src_dst_trans_nb, GGML_MAX_DIMS);
        aclnn_cast(ctx, acl_src.get(), acl_src_trans_tensor.get(), ACL_FLOAT);
    }

    // Step 2: Prepare head tensors for tail splitting if needed
    acl_tensor_ptr acl_src_ತಲೆ;
    acl_tensor_ptr acl_dst_ತಲೆ;
    ಆದರೆ (has_ಬಾಲ) {
        // Create head views for RotaryPositionEmbedding (only first rope_dims dimensions)
        // RotaryPositionEmbedding requires contiguous dst tensor, so we use a temporary buffer
        ಆದರೆ (src_dst_need_trans) {
            // Use F32 trans tensor strides
            acl_src_ತಲೆ = ggml_cann_create_tensor((ಅಕ್ಷರ *) src_trans_ಬಫರ್, ACL_FLOAT, ಗಾತ್ರ(ದಶಮಾಂಶ), ತಲೆ_ne,
                                                   src_dst_trans_nb, GGML_MAX_DIMS);
        } ಇಲ್ಲದಿದ್ದರೆ {
            // Use original F32 tensor strides
            acl_src_ತಲೆ = ggml_cann_create_tensor((ಅಕ್ಷರ *) src0->ದತ್ತಾಂಶ, ACL_FLOAT, ಗಾತ್ರ(ದಶಮಾಂಶ), ತಲೆ_ne, src0->nb,
                                                   GGML_MAX_DIMS);
        }

        int64_t              ತಲೆ_elements = rope_dims * ne01 * ne02 * ne03;
        ggml_cann_pool_alloc dst_ತಲೆ_contiguous_allocator(ctx.pool(), ತಲೆ_elements * ಗಾತ್ರ(ದಶಮಾಂಶ));
        ಶೂನ್ಯ *               dst_ತಲೆ_contiguous_ಬಫರ್ = dst_ತಲೆ_contiguous_allocator.get();

        size_t ತಲೆ_contiguous_nb[GGML_MAX_DIMS];
        ತಲೆ_contiguous_nb[0] = ಗಾತ್ರ(ದಶಮಾಂಶ);
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 1; i < GGML_MAX_DIMS; i++) {
            ತಲೆ_contiguous_nb[i] = ತಲೆ_contiguous_nb[i - 1] * ತಲೆ_ne[i - 1];
        }
        acl_dst_ತಲೆ = ggml_cann_create_tensor(dst_ತಲೆ_contiguous_ಬಫರ್, ACL_FLOAT, ಗಾತ್ರ(ದಶಮಾಂಶ), ತಲೆ_ne,
                                               ತಲೆ_contiguous_nb, GGML_MAX_DIMS);
    }

    // Step 3: Execute RotaryPositionEmbedding
    ಆದರೆ (has_ಬಾಲ) {
        // Rotate only the head portion (first rope_dims dimensions)
        GGML_CANN_CALL_ACLNN_OP(ctx, RotaryPositionEmbedding, acl_src_ತಲೆ.get(), acl_cos_reshape_tensor.get(),
                                acl_sin_reshape_tensor.get(), acl_ಬಹುಲಕ, acl_dst_ತಲೆ.get());

        // Copy head result from contiguous buffer back to destination tensor
        ಆದರೆ (src_dst_need_trans) {
            acl_tensor_ptr acl_dst_ತಲೆ_ಗುರಿ = ggml_cann_create_tensor(
                (ಅಕ್ಷರ *) dst_trans_ಬಫರ್, ACL_FLOAT, ಗಾತ್ರ(ದಶಮಾಂಶ), ತಲೆ_ne, src_dst_trans_nb, GGML_MAX_DIMS);
            cann_copy(ctx, acl_dst_ತಲೆ.get(), acl_dst_ತಲೆ_ಗುರಿ.get());
        } ಇಲ್ಲದಿದ್ದರೆ {
            acl_tensor_ptr acl_dst_ತಲೆ_ಗುರಿ =
                ggml_cann_create_tensor((ಅಕ್ಷರ *) dst->ದತ್ತಾಂಶ, ACL_FLOAT, ಗಾತ್ರ(ದಶಮಾಂಶ), ತಲೆ_ne, dst->nb, GGML_MAX_DIMS);
            cann_copy(ctx, acl_dst_ತಲೆ.get(), acl_dst_ತಲೆ_ಗುರಿ.get());
        }
    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (src_dst_need_trans) {
        // Rotate full tensor (no tail), using trans tensors
        GGML_CANN_CALL_ACLNN_OP(ctx, RotaryPositionEmbedding, acl_src_trans_tensor.get(), acl_cos_reshape_tensor.get(),
                                acl_sin_reshape_tensor.get(), acl_ಬಹುಲಕ, acl_dst_trans_tensor.get());
    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (src0->ದತ್ತಾಂಶ == dst->ದತ್ತಾಂಶ && !ggml_is_contiguous(src0)) {
        // In-place on non-contiguous tensor: RotaryPositionEmbedding cannot safely
        // read and write the same non-contiguous buffer. Use contiguous temporaries.
        size_t contiguous_nb[GGML_MAX_DIMS];
        contiguous_nb[0] = ಗಾತ್ರ(ದಶಮಾಂಶ);
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 1; i < GGML_MAX_DIMS; i++) {
            contiguous_nb[i] = contiguous_nb[i - 1] * src0->ne[i - 1];
        }
        int64_t              ಒಟ್ಟು_elements = ggml_nelements(src0);
        ggml_cann_pool_alloc inplace_src_alloc(ctx.pool(), ಒಟ್ಟು_elements * ಗಾತ್ರ(ದಶಮಾಂಶ));
        ggml_cann_pool_alloc inplace_dst_alloc(ctx.pool(), ಒಟ್ಟು_elements * ಗಾತ್ರ(ದಶಮಾಂಶ));

        acl_tensor_ptr acl_src_contig = ggml_cann_create_tensor(inplace_src_alloc.get(), ACL_FLOAT, ಗಾತ್ರ(ದಶಮಾಂಶ),
                                                                src0->ne, contiguous_nb, GGML_MAX_DIMS);
        acl_tensor_ptr acl_dst_contig = ggml_cann_create_tensor(inplace_dst_alloc.get(), ACL_FLOAT, ಗಾತ್ರ(ದಶಮಾಂಶ),
                                                                dst->ne, contiguous_nb, GGML_MAX_DIMS);

        cann_copy(ctx, acl_src.get(), acl_src_contig.get());
        GGML_CANN_CALL_ACLNN_OP(ctx, RotaryPositionEmbedding, acl_src_contig.get(), acl_cos_reshape_tensor.get(),
                                acl_sin_reshape_tensor.get(), acl_ಬಹುಲಕ, acl_dst_contig.get());
        cann_copy(ctx, acl_dst_contig.get(), acl_dst.get());
    } ಇಲ್ಲದಿದ್ದರೆ {
        // Rotate full tensor (no tail), using original tensors
        GGML_CANN_CALL_ACLNN_OP(ctx, RotaryPositionEmbedding, acl_src.get(), acl_cos_reshape_tensor.get(),
                                acl_sin_reshape_tensor.get(), acl_ಬಹುಲಕ, acl_dst.get());
    }

    // Step 4: Copy unrotated tail portion from source to destination
    ಆದರೆ (has_ಬಾಲ) {
        size_t src_ಬಾಲ_ಸ್ಥಾನಾಂತರ;
        size_t dst_ಬಾಲ_ಸ್ಥಾನಾಂತರ;

        ಸ್ವಯಂಚಾಲಿತ copy_ಬಾಲ_device = [&](ಶೂನ್ಯ * src_ptr, ಶೂನ್ಯ * dst_ptr, acl_ದತ್ತಾಂಶ_Type dtype, size_t elem_ಗಾತ್ರ,
                                    size_t * nb_src_arr, size_t * nb_dst_arr) {
            acl_tensor_ptr acl_src_ಬಾಲ =
                ggml_cann_create_tensor(src_ptr, dtype, elem_ಗಾತ್ರ, ಬಾಲ_ne, nb_src_arr, GGML_MAX_DIMS);
            acl_tensor_ptr acl_dst_ಬಾಲ =
                ggml_cann_create_tensor(dst_ptr, dtype, elem_ಗಾತ್ರ, ಬಾಲ_ne, nb_dst_arr, GGML_MAX_DIMS);
            cann_copy(ctx, acl_src_ಬಾಲ.get(), acl_dst_ಬಾಲ.get());
        };

        ಆದರೆ (src_dst_need_trans) {
            // Use F32 trans tensor strides and offsets
            src_ಬಾಲ_ಸ್ಥಾನಾಂತರ = rope_dims * src_dst_trans_nb[0];
            dst_ಬಾಲ_ಸ್ಥಾನಾಂತರ = rope_dims * src_dst_trans_nb[0];
            copy_ಬಾಲ_device((ಅಕ್ಷರ *) src_trans_ಬಫರ್ + src_ಬಾಲ_ಸ್ಥಾನಾಂತರ, (ಅಕ್ಷರ *) dst_trans_ಬಫರ್ + dst_ಬಾಲ_ಸ್ಥಾನಾಂತರ,
                             ACL_FLOAT, ಗಾತ್ರ(ದಶಮಾಂಶ), src_dst_trans_nb, src_dst_trans_nb);
        } ಇಲ್ಲದಿದ್ದರೆ {
            // Use original tensor strides and offsets
            src_ಬಾಲ_ಸ್ಥಾನಾಂತರ = rope_dims * nb00;
            dst_ಬಾಲ_ಸ್ಥಾನಾಂತರ = rope_dims * nb0;
            copy_ಬಾಲ_device((ಅಕ್ಷರ *) src0->ದತ್ತಾಂಶ + src_ಬಾಲ_ಸ್ಥಾನಾಂತರ, (ಅಕ್ಷರ *) dst->ದತ್ತಾಂಶ + dst_ಬಾಲ_ಸ್ಥಾನಾಂತರ,
                             ggml_cann_type_mapping(dst->type), ggml_element_ಗಾತ್ರ(dst), src0->nb, dst->nb);
        }
    }

    // Step 5: Cast back to F16 if needed
    ಆದರೆ (src_dst_need_trans) {
        aclnn_cast(ctx, acl_dst_trans_tensor.get(), acl_dst.get(), ACL_FLOAT16);
    }
}

ಶೂನ್ಯ ggml_cann_rope_cache_preload(ggml_backend_cann_context & ctx, ggml_tensor * dst) {
    ggml_tensor * src0 = dst->src[0];

    ದಶಮಾಂಶ     freq_ತಳ, freq_scale, ext_ಗುಣಕ, attn_ಗುಣಕ, beta_fast, beta_slow;
    ಪೂರ್ಣಾಂಕ       sections[4];
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ n_dims     = ((int32_t *) dst->op_params)[1];
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ mode       = ((int32_t *) dst->op_params)[2];
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ n_ctx_orig = ((int32_t *) dst->op_params)[4];

    GGML_TENSOR_UNARY_OP_LOCALS

    ಸ್ಮರಣೆ_ನಕಲು(&freq_ತಳ, (int32_t *) dst->op_params + 5, ಗಾತ್ರ(ದಶಮಾಂಶ));
    ಸ್ಮರಣೆ_ನಕಲು(&freq_scale, (int32_t *) dst->op_params + 6, ಗಾತ್ರ(ದಶಮಾಂಶ));
    ಸ್ಮರಣೆ_ನಕಲು(&ext_ಗುಣಕ, (int32_t *) dst->op_params + 7, ಗಾತ್ರ(ದಶಮಾಂಶ));
    ಸ್ಮರಣೆ_ನಕಲು(&attn_ಗುಣಕ, (int32_t *) dst->op_params + 8, ಗಾತ್ರ(ದಶಮಾಂಶ));
    ಸ್ಮರಣೆ_ನಕಲು(&beta_fast, (int32_t *) dst->op_params + 9, ಗಾತ್ರ(ದಶಮಾಂಶ));
    ಸ್ಮರಣೆ_ನಕಲು(&beta_slow, (int32_t *) dst->op_params + 10, ಗಾತ್ರ(ದಶಮಾಂಶ));
    ಸ್ಮರಣೆ_ನಕಲು(&sections, (int32_t *) dst->op_params + 11, ಗಾತ್ರ(ಪೂರ್ಣಾಂಕ) * 4);

    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ theta_scale = powf(freq_ತಳ, -2.0f / n_dims);

    ದಶಮಾಂಶ corr_dims[2];
    ggml_rope_yarn_corr_dims(n_dims, n_ctx_orig, freq_ತಳ, beta_fast, beta_slow, corr_dims);

    ತಾರ್ಕಿಕ       is_neox    = mode & GGML_ROPE_TYPE_NEOX;
    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ is_imrope  = mode == GGML_ROPE_TYPE_IMROPE;
    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ mrope_ಬಳಸಿದ = mode & GGML_ROPE_TYPE_MROPE;
    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ is_vision  = mode == GGML_ROPE_TYPE_VISION;

    ಆದರೆ (is_imrope || mrope_ಬಳಸಿದ) {
        is_neox = ನಿಜ;
    }

    int64_t rope_dims = n_dims;
    ಆದರೆ (is_vision) {
        rope_dims = src0->ne[0];
    }

    // Run the full cache init on the non-captured stream.  This performs all
    // host-to-device memcpy, aclrtMalloc/Free, and on-device computations
    // so that the memory pool is warmed up and cache metadata is populated.
    aclnn_rope_cache_init(ctx, dst, corr_dims, ext_ಗುಣಕ, theta_scale, freq_scale, attn_ಗುಣಕ, is_neox, sections,
                          mrope_ಬಳಸಿದ, is_imrope, is_vision, rope_dims);

    // Reset `cached` so that during graph capture the on-device computations
    // (sin/cos, position multiply, repeat, etc.) still execute and get recorded
    // into the captured graph.  The cache metadata (theta_scale_length,
    // theta_scale, sections, position_length, etc.) remains set, which causes
    // all host-to-device copy and malloc/free branches to be skipped.
    ctx.rope_cache.cached = ಸುಳ್ಳು;
}

ಶೂನ್ಯ ggml_cann_argmax(ggml_backend_cann_context & ctx, ggml_tensor * dst) {
    ggml_tensor * src0 = dst->src[0];

    acl_tensor_ptr acl_src = ggml_cann_create_tensor(src0);
    acl_tensor_ptr acl_dst = ggml_cann_create_tensor(dst, dst->ne, dst->nb, 3);

    GGML_CANN_CALL_ACLNN_OP(ctx, ArgMax, acl_src.get(), 3, ಸುಳ್ಳು, acl_dst.get());
}

ಶೂನ್ಯ ggml_cann_conv_transpose_1d(ggml_backend_cann_context & ctx, ggml_tensor * dst) {
    ggml_tensor * src0 = dst->src[0];
    ggml_tensor * src1 = dst->src[1];

    // stride
    int64_t s0 = ((ಸ್ಥಿರಾಂಕ int32_t *) (dst->op_params))[0];

    acl_tensor_ptr acl_ಒಳಹರಿವು  = ggml_cann_create_tensor(src1, src1->ne, src1->nb, 3, ACL_FORMAT_NCL);
    acl_tensor_ptr acl_ತೂಕ = ggml_cann_create_tensor(src0, src0->ne, src0->nb, 3, ACL_FORMAT_NCL);
    acl_tensor_ptr acl_dst    = ggml_cann_create_tensor(dst, dst->ne, dst->nb, 3, ACL_FORMAT_NCL);

    // get base information of input and kernel
    int64_t ಒಳಹರಿವು_len   = *(src1->ne);
    int64_t dst_len     = *(dst->ne);
    int64_t kernel_ಗಾತ್ರ = *(src0->ne);

    // set the max kernel size for each conv
    int64_t max_kernel_ಗಾತ್ರ = 255;

    // compute the partition of kernel
    int64_t part_ಅಂಕೆ = 1;
    part_ಅಂಕೆ         = (kernel_ಗಾತ್ರ + max_kernel_ಗಾತ್ರ - 1) / max_kernel_ಗಾತ್ರ;

    int64_t strideVal[1];
    strideVal[0]                    = s0;
    acl_int_array_ptr stride        = ggml_cann_create_int_array(strideVal, 1);
    int64_t           paddingVal[]  = { 0 };
    acl_int_array_ptr padding       = ggml_cann_create_int_array(paddingVal, 1);
    int64_t           dilationVal[] = { 1 };
    acl_int_array_ptr dilation      = ggml_cann_create_int_array(dilationVal, 1);
    ತಾರ್ಕಿಕ              transposed    = ನಿಜ;
    int64_t           groups        = 1;
    int8_t            cubeMathType  = 0;

#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ ASCEND_310P
    cubeMathType = 1;
#ಆದರೆ_ಅಂತ್ಯ

    ಸ್ವಯಂಚಾಲಿತ ತೂಕ_type = ggml_cann_type_mapping(src0->type);
    ಸ್ವಯಂಚಾಲಿತ dst_type    = ggml_cann_type_mapping(dst->type);

    // slice the kernel to make each conv available
    int64_t ತುಂಡು_dim   = -1;
    int64_t ತುಂಡು_ಆರಂಭ = 0;
    int64_t ತುಂಡು_ಕೊನೆ   = max_kernel_ಗಾತ್ರ;
    int64_t ತುಂಡು_ಹೆಜ್ಜೆ  = 1;
    int64_t interval    = max_kernel_ಗಾತ್ರ;

    int64_t ಎಡ_pad_len  = dilationVal[0] * (max_kernel_ಗಾತ್ರ - 1) + 1 - 2 * paddingVal[0];
    int64_t ಬಲ_pad_len = 0;

    acl_scalar_ptr alpha      = ಶೂನ್ಯಸೂಚಕ;
    ದಶಮಾಂಶ          alpha_ಮೌಲ್ಯ = 1.0;
    alpha                     = ggml_cann_create_scalar(&alpha_ಮೌಲ್ಯ, acl_ದತ್ತಾಂಶ_Type::ACL_FLOAT);

    // set zero to destination
    GGML_CANN_CALL_ACLNN_OP(ctx, InplaceZero, acl_dst.get());

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ k = 0; k < part_ಅಂಕೆ; k++) {
        // create part kernel tensor and slice from big kernel
        ತುಂಡು_ಆರಂಭ = max_kernel_ಗಾತ್ರ * k;
        ಆದರೆ (k == part_ಅಂಕೆ - 1) {
            ತುಂಡು_ಕೊನೆ = kernel_ಗಾತ್ರ;
            interval  = kernel_ಗಾತ್ರ - max_kernel_ಗಾತ್ರ * k;
        } ಇಲ್ಲದಿದ್ದರೆ {
            ತುಂಡು_ಕೊನೆ = max_kernel_ಗಾತ್ರ * (k + 1);
        }

        int64_t part_ne[4];
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < 4; i++) {
            part_ne[i] = *(src0->ne + i);
        }
        part_ne[0] = interval;

        size_t part_nb[4];
        part_nb[0] = ಗಾತ್ರ(ತೂಕ_type);
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 1; i < 4; i++) {
            part_nb[i] = part_nb[i - 1] * part_ne[i - 1];
        }

        ggml_cann_pool_alloc part_kernel_allocator;
        part_kernel_allocator.alloc(ctx.pool(), part_nb[3]);
        ಶೂನ್ಯ * part_kernel_buf = part_kernel_allocator.get();

        acl_tensor_ptr part_kernel = ggml_cann_create_tensor(part_kernel_buf, ತೂಕ_type, ggml_element_ಗಾತ್ರ(src0),
                                                             part_ne, part_nb, 3, ACL_FORMAT_NCL);

        GGML_CANN_CALL_ACLNN_OP(ctx, ತುಂಡು, acl_ತೂಕ.get(), ತುಂಡು_dim, ತುಂಡು_ಆರಂಭ, ತುಂಡು_ಕೊನೆ, ತುಂಡು_ಹೆಜ್ಜೆ,
                                part_kernel.get());

        // create the part conv result tensor
        int64_t part_dst_ne[4];
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < 4; i++) {
            part_dst_ne[i] = *(dst->ne + i);
        }
        part_dst_ne[0] = (ಒಳಹರಿವು_len - 1) * strideVal[0] - 2 * paddingVal[0] + dilationVal[0] * (part_ne[0] - 1) + 1;

        size_t part_dst_nb[4];
        part_dst_nb[0] = ಗಾತ್ರ(ತೂಕ_type);
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 1; i < 4; i++) {
            part_dst_nb[i] = part_dst_nb[i - 1] * part_dst_ne[i - 1];
        }
        ggml_cann_pool_alloc part_dst_allocator;
        part_dst_allocator.alloc(ctx.pool(), part_dst_nb[3]);
        ಶೂನ್ಯ * part_dst_buf = part_dst_allocator.get();

        acl_tensor_ptr acl_part_dst = ggml_cann_create_tensor(part_dst_buf, dst_type, ggml_element_ಗಾತ್ರ(dst),
                                                              part_dst_ne, part_dst_nb, 3, ACL_FORMAT_NCL);
        GGML_CANN_CALL_ACLNN_OP(ctx, InplaceZero, acl_part_dst.get());

        // compute part conv transpose 1d
        GGML_CANN_CALL_ACLNN_OP(ctx, Convolution, acl_ಒಳಹರಿವು.get(), part_kernel.get(), ಶೂನ್ಯಸೂಚಕ, stride.get(),
                                padding.get(), dilation.get(), transposed, padding.get(), groups, acl_part_dst.get(),
                                cubeMathType);

        // compute the position of part result in final result
        int64_t global_ಆರಂಭ = ತುಂಡು_ಆರಂಭ;
        int64_t global_ಕೊನೆ   = std::min((ಒಳಹರಿವು_len - 1) * strideVal[0] + ತುಂಡು_ಕೊನೆ, dst_len);

        ಎಡ_pad_len  = global_ಆರಂಭ;
        ಬಲ_pad_len = dst_len - global_ಕೊನೆ;

        std::vector<int64_t> pad_ದತ್ತಾಂಶ_Val = { ಎಡ_pad_len, ಬಲ_pad_len };
        acl_int_array_ptr    pad_ದತ್ತಾಂಶ    = ggml_cann_create_int_array(pad_ದತ್ತಾಂಶ_Val.ದತ್ತಾಂಶ(), 2);

        acl_scalar_ptr pad_ಮೌಲ್ಯ    = ಶೂನ್ಯಸೂಚಕ;
        ದಶಮಾಂಶ          pad_ಮೌಲ್ಯ_Val = 0.0;
        pad_ಮೌಲ್ಯ                   = ggml_cann_create_scalar(&pad_ಮೌಲ್ಯ_Val, acl_ದತ್ತಾಂಶ_Type::ACL_FLOAT);

        int64_t conv_ಫಲಿತಾಂಶ_ne[4];
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < 4; i++) {
            conv_ಫಲಿತಾಂಶ_ne[i] = *(dst->ne + i);
        }

        size_t conv_ಫಲಿತಾಂಶ_nb[4];
        conv_ಫಲಿತಾಂಶ_nb[0] = ಗಾತ್ರ(ತೂಕ_type);
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 1; i < 4; i++) {
            conv_ಫಲಿತಾಂಶ_nb[i] = conv_ಫಲಿತಾಂಶ_nb[i - 1] * conv_ಫಲಿತಾಂಶ_ne[i - 1];
        }

        ggml_cann_pool_alloc conv_ಫಲಿತಾಂಶ_allocator;
        conv_ಫಲಿತಾಂಶ_allocator.alloc(ctx.pool(), conv_ಫಲಿತಾಂಶ_nb[3]);
        ಶೂನ್ಯ * conv_ಫಲಿತಾಂಶ_buf = conv_ಫಲಿತಾಂಶ_allocator.get();

        acl_tensor_ptr conv_ಫಲಿತಾಂಶ = ggml_cann_create_tensor(conv_ಫಲಿತಾಂಶ_buf, dst_type, ggml_element_ಗಾತ್ರ(dst),
                                                             conv_ಫಲಿತಾಂಶ_ne, conv_ಫಲಿತಾಂಶ_nb, 3, ACL_FORMAT_NCL);

        GGML_CANN_CALL_ACLNN_OP(ctx, InplaceZero, conv_ಫಲಿತಾಂಶ.get());
        GGML_CANN_CALL_ACLNN_OP(ctx, ConstantPadNd, acl_part_dst.get(), pad_ದತ್ತಾಂಶ.get(), pad_ಮೌಲ್ಯ.get(),
                                conv_ಫಲಿತಾಂಶ.get());
        GGML_CANN_CALL_ACLNN_OP(ctx, Inplace_ಕೂಡಿಸು, acl_dst.get(), conv_ಫಲಿತಾಂಶ.get(), alpha.get());
    }
}

ಶೂನ್ಯ ggml_cann_elu(ggml_backend_cann_context & ctx, ggml_tensor * dst) {
    ggml_tensor * src0 = dst->src[0];

    acl_tensor_ptr acl_ಒಳಹರಿವು = ggml_cann_create_tensor(src0);
    acl_tensor_ptr acl_dst   = ggml_cann_create_tensor(dst);

    ದಶಮಾಂಶ          alpha_ಮೌಲ್ಯ = 1.0f;
    acl_scalar_ptr alpha      = ಶೂನ್ಯಸೂಚಕ;
    alpha                     = ggml_cann_create_scalar(&alpha_ಮೌಲ್ಯ, acl_ದತ್ತಾಂಶ_Type::ACL_FLOAT);

    GGML_CANN_CALL_ACLNN_OP(ctx, Elu, acl_ಒಳಹರಿವು.get(), alpha.get(), alpha.get(), alpha.get(), acl_dst.get());
}

ಶೂನ್ಯ ggml_cann_ಸರಾಸರಿ(ggml_backend_cann_context & ctx, ggml_tensor * dst) {
    ggml_tensor * src0 = dst->src[0];

    acl_tensor_ptr acl_src = ggml_cann_create_tensor(src0);
    acl_tensor_ptr acl_dst = ggml_cann_create_tensor(dst);

    int64_t           reduce_Dim_ಮೌಲ್ಯ[] = { 3 };
    acl_int_array_ptr reduceDim        = ggml_cann_create_int_array(reduce_Dim_ಮೌಲ್ಯ, 1);
    ತಾರ್ಕಿಕ              keepDim          = ನಿಜ;

    GGML_CANN_CALL_ACLNN_OP(ctx, ಸರಾಸರಿ, acl_src.get(), reduceDim.get(), keepDim, ACL_FLOAT, acl_dst.get());
}

ಶೂನ್ಯ ggml_cann_pad_reflect_1d(ggml_backend_cann_context & ctx, ggml_tensor * dst) {
    ggml_tensor *     src0             = dst->src[0];
    int32_t *         opts             = (int32_t *) dst->op_params;
    int64_t           paddingsArray[2] = { opts[0], opts[1] };
    acl_int_array_ptr paddings         = ggml_cann_create_int_array(paddingsArray, 2);

    // Collapsing ne[2]*ne[3] into a single batch dimension requires that dim3
    // is contiguous with respect to dim2 in both src and dst.
    GGML_ASSERT(src0->nb[3] == src0->nb[2] * src0->ne[2]);
    GGML_ASSERT(dst->nb[3]  == dst->nb[2]  * dst->ne[2]);

    int64_t src_ne_3d[3] = { src0->ne[0], src0->ne[1], src0->ne[2] * src0->ne[3] };
    int64_t dst_ne_3d[3] = { dst->ne[0],  dst->ne[1],  dst->ne[2]  * dst->ne[3]  };

    acl_tensor_ptr acl_src = ggml_cann_create_tensor(src0->ದತ್ತಾಂಶ, ggml_cann_type_mapping(src0->type),
                                                     ggml_element_ಗಾತ್ರ(src0), src_ne_3d, src0->nb, 3);

    acl_tensor_ptr acl_dst = ggml_cann_create_tensor(dst->ದತ್ತಾಂಶ, ggml_cann_type_mapping(dst->type),
                                                     ggml_element_ಗಾತ್ರ(dst), dst_ne_3d, dst->nb, 3);

    GGML_CANN_CALL_ACLNN_OP(ctx, ReflectionPad1d, acl_src.get(), paddings.get(), acl_dst.get());
}

ಶೂನ್ಯ ggml_cann_ಎಣಿಕೆ_equal(ggml_backend_cann_context & ctx, ggml_tensor * dst) {
    ggml_tensor * src0 = dst->src[0];
    ggml_tensor * src1 = dst->src[1];

    // Write element-wise equality (0 or 1) into a temporary buffer to avoid
    // modifying src0 in-place.  Use the same type as src0 so ReduceSum can
    // consume it directly without a type cast.
    ggml_cann_pool_alloc eq_alloc(ctx.pool(), ggml_nelements(src0) * ggml_element_ಗಾತ್ರ(src0));
    size_t eq_nb[GGML_MAX_DIMS];
    eq_nb[0] = ggml_element_ಗಾತ್ರ(src0);
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 1; i < GGML_MAX_DIMS; i++) {
        eq_nb[i] = eq_nb[i - 1] * src0->ne[i - 1];
    }
    acl_tensor_ptr acl_eq = ggml_cann_create_tensor(
        eq_alloc.get(), ggml_cann_type_mapping(src0->type), ggml_element_ಗಾತ್ರ(src0),
        src0->ne, eq_nb, GGML_MAX_DIMS);

    acl_tensor_ptr acl_self  = ggml_cann_create_tensor(src0);
    acl_tensor_ptr acl_other = ggml_cann_create_tensor(src1);
    GGML_CANN_CALL_ACLNN_OP(ctx, EqTensor, acl_self.get(), acl_other.get(), acl_eq.get());

    // Sum the 0/1 values into dst.
    acl_tensor_ptr    acl_dst    = ggml_cann_create_tensor(dst);
    int64_t           dims[4]    = { 0, 1, 2, 3 };
    acl_int_array_ptr dims_arr   = ggml_cann_create_int_array(dims, 4);
    GGML_CANN_CALL_ACLNN_OP(ctx, Reduce_ಮೊತ್ತ, acl_eq.get(), dims_arr.get(), ನಿಜ,
                            ggml_cann_type_mapping(dst->type), acl_dst.get());
}

ಶೂನ್ಯ ggml_cann_ಹೆಜ್ಜೆ(ggml_backend_cann_context & ctx, ggml_tensor * dst) {
    ggml_tensor * src0 = dst->src[0];

    acl_tensor_ptr acl_src = ggml_cann_create_tensor(src0);
    acl_tensor_ptr acl_dst = ggml_cann_create_tensor(dst);

    ದಶಮಾಂಶ          alpha_ಮೌಲ್ಯ = 0.0f;
    acl_scalar_ptr alpha      = ಶೂನ್ಯಸೂಚಕ;
    alpha                     = ggml_cann_create_scalar(&alpha_ಮೌಲ್ಯ, acl_ದತ್ತಾಂಶ_Type::ACL_FLOAT);

    GGML_CANN_CALL_ACLNN_OP(ctx, GtScalar, acl_src.get(), alpha.get(), acl_dst.get());
}

ಶೂನ್ಯ ggml_cann_softplus(ggml_backend_cann_context & ctx, ggml_tensor * dst) {
    ggml_tensor * src0 = dst->src[0];

    acl_tensor_ptr acl_src = ggml_cann_create_tensor(src0);
    acl_tensor_ptr acl_dst = ggml_cann_create_tensor(dst);

    ದಶಮಾಂಶ          beta_val      = 1.0f;
    ದಶಮಾಂಶ          ಮಿತಿ_val = 20.0f;
    acl_scalar_ptr beta          = ggml_cann_create_scalar(&beta_val,      ACL_FLOAT);
    acl_scalar_ptr ಮಿತಿ     = ggml_cann_create_scalar(&ಮಿತಿ_val, ACL_FLOAT);

    GGML_CANN_CALL_ACLNN_OP(ctx, Softplus, acl_src.get(), beta.get(), ಮಿತಿ.get(), acl_dst.get());
}

ಶೂನ್ಯ ggml_cann_geglu_ಶೀಘ್ರ(ggml_backend_cann_context & ctx, ggml_tensor * dst) {
    ಸ್ವಯಂಚಾಲಿತ gelu_ಶೀಘ್ರ_fn = [](ggml_backend_cann_context & ctx, aclTensor * acl_src, aclTensor * acl_dst) {
        GGML_CANN_CALL_ACLNN_OP(ctx, GeluV2, acl_src, 0, acl_dst);
    };
    ggml_cann_op_unary_gated(gelu_ಶೀಘ್ರ_fn, ctx, dst);
}

/**
 * @brief Performs expert-specific matrix multiplication (MoE) with
 * floating-point precision using the CANN backend.
 *
 * This function executes a matrix multiplication operation tailored for
 * Mixture of Experts (MoE) models, where the input tensor is multiplied
 * with expert-specific weight matrices. It uses the CANN backend for
 * efficient computation and stores the result in the destination tensor `dst`.
 * The operation may leverage identity-based optimizations or routing masks
 * as part of sparse expert selection.
 *
 * @param ctx The context for executing CANN backend operations.
 * @param dst The destination tensor where the MoE multiplication result
 * will be stored.
 *
 * @note This function assumes floating-point data types and is designed for
 * MoE architectures, possibly involving sparse expert routing.
 */
ಸ್ಥಿರ ಶೂನ್ಯ ggml_cann_mul_mat_id_fp(ggml_backend_cann_context & ctx, ggml_tensor * dst) {
    //dst   [M, K, N, 1]
    ggml_tensor * src0 = dst->src[0];  //src0	[D, M, A, 1]  -> [D, M, K, 1]
    ggml_tensor * src1 = dst->src[1];  //src1	[D, B, N, 1], B = K or B = 1 -> [D, 1, K, 1]
    ggml_tensor * ids  = dst->src[2];  //ids	[K, N]

    GGML_ASSERT(src0->ne[3] == 1);
    GGML_ASSERT(src1->ne[3] == 1);
    GGML_ASSERT(dst->ne[3] == 1);

    int64_t batch = src1->ne[2];
    GGML_ASSERT(batch == ids->ne[1]);

    ggml_cann_pool_alloc export_allocator(ctx.pool(), src0->ne[0] * src0->ne[1] * ids->ne[0] * ggml_element_ಗಾತ್ರ(src0));
    ಶೂನ್ಯ *               export_ptr = export_allocator.get();
    ಗಾಗಿ (int64_t i = 0; i < batch; i++) {
        acl_tensor_ptr select_ಸೂಚ್ಯಂಕ  = ggml_cann_create_tensor(ids, ids->ne, ids->nb, 1, ACL_FORMAT_ND, i * ids->nb[1]);
        acl_tensor_ptr export_ತೂಕ = ggml_cann_create_tensor(src0, src0->ne, src0->nb, 3);

        int64_t select_export_ne[] = { src0->ne[0], src0->ne[1], ids->ne[0] };
        size_t  select_export_nb[3];
        select_export_nb[0] = src0->nb[0];
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ k = 1; k < 3; k++) {
            select_export_nb[k] = select_export_nb[k - 1] * select_export_ne[k - 1];
        }

        acl_tensor_ptr select_export =
            ggml_cann_create_tensor(export_ptr, ggml_cann_type_mapping(src0->type), ggml_element_ಗಾತ್ರ(src0),
                                    select_export_ne, select_export_nb, 3);
        GGML_CANN_CALL_ACLNN_OP(ctx, ಸೂಚ್ಯಂಕ_Select, export_ತೂಕ.get(), 0, select_ಸೂಚ್ಯಂಕ.get(), select_export.get());

        int64_t        select_transpose_ne[] = { select_export_ne[1], select_export_ne[0], select_export_ne[2] };
        size_t         select_transpose_nb[] = { select_export_nb[1], select_export_nb[0], select_export_nb[2] };
        acl_tensor_ptr select_export_transpose =
            ggml_cann_create_tensor(export_ptr, ggml_cann_type_mapping(src0->type), ggml_element_ಗಾತ್ರ(src0),
                                    select_transpose_ne, select_transpose_nb, 3);

        int64_t        ಸಕ್ರಿಯ_tensor_ne[] = { src1->ne[0], 1, src1->ne[1] };
        size_t         ಸಕ್ರಿಯ_tensor_nb[] = { src1->nb[0], src1->nb[1], src1->nb[1] };
        acl_tensor_ptr ಸಕ್ರಿಯ_tensor =
            ggml_cann_create_tensor(src1, ಸಕ್ರಿಯ_tensor_ne, ಸಕ್ರಿಯ_tensor_nb, 3, ACL_FORMAT_ND, i * src1->nb[2]);

        int64_t        dst_ne[] = { dst->ne[0], 1, dst->ne[1] };
        size_t         dst_nb[] = { dst->nb[0], dst->nb[1], dst->nb[1] };
        acl_tensor_ptr acl_dst  = ggml_cann_create_tensor(dst, dst_ne, dst_nb, 3, ACL_FORMAT_ND, i * dst->nb[2]);

        GGML_CANN_CALL_ACLNN_OP(ctx, BatchMatMul, ಸಕ್ರಿಯ_tensor.get(), select_export_transpose.get(), acl_dst.get(), 2);
    }
}

/**
 * @brief Performs quantized matrix multiplication for Mixture of Experts (MoE)
 * models using the CANN backend.
 *
 * This function implements MUL_MAT_ID operation for quantized weight matrices
 * (Q4_0 and Q8_0 formats). It selects expert-specific weight matrices based on
 * the provided expert indices, and computes matrix multiplication using CANN's
 * WeightQuantBatchMatmulV2 operator.
 *
 * The function performs the following steps:
 * 1. Converts input/output tensors to F16 format if necessary
 * 2. Uses IndexSelect to extract expert-specific weights and scales based on indices
 * 3. Performs quantized matrix multiplication for each expert using WeightQuantBatchMatmulV2
 * 4. Converts output back to the target type if needed
 *
 * Tensor shapes:
 * - dst:  [M, K, N, 1] - output tensor
 * - src0: [D, M, A, 1] - quantized weight matrices (Q4_0 or Q8_0)
 * - src1: [D, B, N, 1] - input activations (B = K for per-expert input, or B = 1 for broadcast)
 * - ids:  [K, N] - expert indices for routing
 *
 * @param ctx The CANN backend context for operation execution.
 * @param dst The destination tensor where the multiplication result will be stored.
 *
 * @note Only Q4_0 and Q8_0 quantization formats are supported.
 * @note The function handles automatic type conversion to/from F16 as needed by the hardware.
 */
ಸ್ಥಿರ ಶೂನ್ಯ ggml_cann_mul_mat_id_quant(ggml_backend_cann_context & ctx, ggml_tensor * dst) {
    // dst:  [M, K, N, 1]
    // src0: [D, M, A, 1] - quantized weights
    // src1: [D, B, N, 1] - input activations, B = K or B = 1
    // ids:  [K, N] - expert indices
    ggml_tensor * src0 = dst->src[0];
    ggml_tensor * src1 = dst->src[1];
    ggml_tensor * ids  = dst->src[2];

    GGML_ASSERT(src0->ne[3] == 1);
    GGML_ASSERT(src1->ne[3] == 1);
    GGML_ASSERT(dst->ne[3] == 1);
    GGML_ASSERT(src1->ne[2] == ids->ne[1]);

    ಸ್ಥಿರಾಂಕ int64_t        n_batches        = ids->ne[1];
    ಸ್ಥಿರಾಂಕ int64_t        n_select_experts = ids->ne[0];
    ಸ್ಥಿರಾಂಕ ಎಣಿಕೆ ggml_type type             = src0->type;

    ಸ್ಥಿರಾಂಕ int32_t ಗುಂಪು_ಗಾತ್ರ = QK8_0;  // Both Q4_0 and Q8_0 use group size of 32
    GGML_ASSERT(ಗುಂಪು_ಗಾತ್ರ == QK4_0);

    // Calculate element size for quantized weights
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ ತೂಕ_elem_ಗಾತ್ರ =
        (type == GGML_TYPE_Q4_0) ? 0.5f :
        (type == GGML_TYPE_Q8_0) ? 1.0f :
                                   (GGML_ABORT("MUL_MAT_ID only supports Q4_0 and Q8_0"), 0.0f);

    // Calculate scale offset in memory
    ಸ್ಥಿರಾಂಕ size_t ತೂಕ_ಗಾತ್ರ     = src0->ne[0] * src0->ne[1] * src0->ne[2] * ತೂಕ_elem_ಗಾತ್ರ;
    ಸ್ಥಿರಾಂಕ size_t scale_elem_ಗಾತ್ರ = ಗಾತ್ರ(uint16_t);
    ಅಕ್ಷರ *       scale_ದತ್ತಾಂಶ      = (ಅಕ್ಷರ *) src0->ದತ್ತಾಂಶ + ತೂಕ_ಗಾತ್ರ;

    // Allocate buffers for selected expert weights and scales
    ಸ್ಥಿರಾಂಕ size_t         selected_ತೂಕ_ಗಾತ್ರ = src0->ne[0] * src0->ne[1] * n_select_experts * ತೂಕ_elem_ಗಾತ್ರ;
    ggml_cann_pool_alloc selected_ತೂಕ_alloc(ctx.pool(), selected_ತೂಕ_ಗಾತ್ರ);
    ಶೂನ್ಯ *               selected_ತೂಕ_ಬಫರ್ = selected_ತೂಕ_alloc.get();

    ಸ್ಥಿರಾಂಕ size_t selected_scale_ಗಾತ್ರ = (src0->ne[0] / ಗುಂಪು_ಗಾತ್ರ) * src0->ne[1] * n_select_experts * scale_elem_ಗಾತ್ರ;
    ggml_cann_pool_alloc selected_scale_alloc(ctx.pool(), selected_scale_ಗಾತ್ರ);
    ಶೂನ್ಯ *               selected_scale_ಬಫರ್ = selected_scale_alloc.get();

    // Helper lambda to allocate and cast tensor to F16 if needed
    constexpr size_t f16_elem_ಗಾತ್ರ      = ಗಾತ್ರ(uint16_t);
    ಸ್ವಯಂಚಾಲಿತ             prepare_f16_ಬಫರ್ = [&](ggml_tensor * tensor, ggml_cann_pool_alloc & allocator,
                                  ತಾರ್ಕಿಕ need_cast = ಸುಳ್ಳು) -> ಶೂನ್ಯ * {
        ಆದರೆ (tensor->type == GGML_TYPE_F16) {
            ಹಿಂತಿರುಗಿ tensor->ದತ್ತಾಂಶ;
        }

        size_t ಒಟ್ಟು_ಗಾತ್ರ = f16_elem_ಗಾತ್ರ;
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < GGML_MAX_DIMS; i++) {
            ಒಟ್ಟು_ಗಾತ್ರ *= tensor->ne[i];
        }
        ಶೂನ್ಯ * ಬಫರ್ = allocator.alloc(ಒಟ್ಟು_ಗಾತ್ರ);

        ಆದರೆ (need_cast == ಸುಳ್ಳು) {
            ಹಿಂತಿರುಗಿ ಬಫರ್;
        }

        int64_t ne[GGML_MAX_DIMS];
        size_t  nb[GGML_MAX_DIMS] = { f16_elem_ಗಾತ್ರ };
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < GGML_MAX_DIMS; i++) {
            ne[i] = tensor->ne[i];
            ಆದರೆ (i > 0) {
                nb[i] = nb[i - 1] * ne[i - 1];
            }
        }

        acl_tensor_ptr src_tensor = ggml_cann_create_tensor(tensor);
        acl_tensor_ptr f16_tensor = ggml_cann_create_tensor(ಬಫರ್, ACL_FLOAT16, f16_elem_ಗಾತ್ರ, ne, nb, GGML_MAX_DIMS);
        aclnn_cast(ctx, src_tensor.get(), f16_tensor.get(), ACL_FLOAT16);

        ಹಿಂತಿರುಗಿ ಬಫರ್;
    };

    // Prepare input and output buffers
    ggml_cann_pool_alloc ಒಳಹರಿವು_alloc(ctx.pool());
    ಶೂನ್ಯ *               ಒಳಹರಿವು_ಬಫರ್ = prepare_f16_ಬಫರ್(src1, ಒಳಹರಿವು_alloc, ನಿಜ);

    ggml_cann_pool_alloc output_alloc(ctx.pool());
    ಶೂನ್ಯ *               output_ಬಫರ್ = prepare_f16_ಬಫರ್(dst, output_alloc, ಸುಳ್ಳು);

    // Process each batch
    ಗಾಗಿ (int64_t batch_idx = 0; batch_idx < n_batches; batch_idx++) {
        // Create index tensor for current batch
        ಸ್ಥಿರಾಂಕ size_t   ಸೂಚ್ಯಂಕ_ಸ್ಥಾನಾಂತರ  = batch_idx * ids->nb[1];
        acl_tensor_ptr batch_indices = ggml_cann_create_tensor(ids, ids->ne, ids->nb, 1, ACL_FORMAT_ND, ಸೂಚ್ಯಂಕ_ಸ್ಥಾನಾಂತರ);

        // Select quantized weights using expert indices
        // Q4_0 stores 2 values per byte, Q8_0 stores 1 value per byte
        ಸ್ಥಿರಾಂಕ int64_t ತೂಕ_d         = (type == GGML_TYPE_Q4_0) ? src0->ne[0] / 2 : src0->ne[0];
        ಸ್ಥಿರಾಂಕ int64_t ತೂಕ_m         = src0->ne[1];
        ಸ್ಥಿರಾಂಕ int64_t ತೂಕ_n_experts = src0->ne[2];

        int64_t ತೂಕ_ne[3] = { ತೂಕ_d, ತೂಕ_m, ತೂಕ_n_experts };
        size_t  ತೂಕ_nb[3] = { ಗಾತ್ರ(int8_t), ತೂಕ_d * ಗಾತ್ರ(int8_t), ತೂಕ_d * ತೂಕ_m * ಗಾತ್ರ(int8_t) };

        acl_tensor_ptr all_weights =
            ggml_cann_create_tensor(src0->ದತ್ತಾಂಶ, ACL_INT8, ಗಾತ್ರ(int8_t), ತೂಕ_ne, ತೂಕ_nb, 3);

        int64_t selected_ತೂಕ_ne[3] = { ತೂಕ_d, ತೂಕ_m, n_select_experts };
        size_t  selected_ತೂಕ_nb[3] = { ಗಾತ್ರ(int8_t), ತೂಕ_d * ಗಾತ್ರ(int8_t),
                                          ತೂಕ_d * ತೂಕ_m * ಗಾತ್ರ(int8_t) };

        acl_tensor_ptr selected_weights = ggml_cann_create_tensor(selected_ತೂಕ_ಬಫರ್, ACL_INT8, ಗಾತ್ರ(int8_t),
                                                                  selected_ತೂಕ_ne, selected_ತೂಕ_nb, 3);

        GGML_CANN_CALL_ACLNN_OP(ctx, ಸೂಚ್ಯಂಕ_Select, all_weights.get(), 0, batch_indices.get(), selected_weights.get());

        // Select scales using the same expert indices
        ಸ್ಥಿರಾಂಕ int64_t scale_d     = src0->ne[0] / ಗುಂಪು_ಗಾತ್ರ;
        int64_t       scale_ne[3] = { scale_d, ತೂಕ_m, ತೂಕ_n_experts };
        size_t scale_nb[3] = { scale_elem_ಗಾತ್ರ, scale_d * scale_elem_ಗಾತ್ರ, scale_d * ತೂಕ_m * scale_elem_ಗಾತ್ರ };

        acl_tensor_ptr all_scales =
            ggml_cann_create_tensor(scale_ದತ್ತಾಂಶ, ACL_FLOAT16, scale_elem_ಗಾತ್ರ, scale_ne, scale_nb, 3);

        int64_t selected_scale_ne[3] = { scale_d, ತೂಕ_m, n_select_experts };
        size_t  selected_scale_nb[3] = { scale_elem_ಗಾತ್ರ, scale_d * scale_elem_ಗಾತ್ರ,
                                         scale_d * ತೂಕ_m * scale_elem_ಗಾತ್ರ };

        acl_tensor_ptr selected_scales = ggml_cann_create_tensor(selected_scale_ಬಫರ್, ACL_FLOAT16, scale_elem_ಗಾತ್ರ,
                                                                 selected_scale_ne, selected_scale_nb, 3);

        GGML_CANN_CALL_ACLNN_OP(ctx, ಸೂಚ್ಯಂಕ_Select, all_scales.get(), 0, batch_indices.get(), selected_scales.get());

        // Process each expert for current batch
        // IndexSelect output layout: [D, M, K] in contiguous format
        // WeightQuantBatchMatmulV2 expects: [M, D] with row-major stride
        ಗಾಗಿ (int64_t expert_idx = 0; expert_idx < n_select_experts; expert_idx++) {
            // Determine input offset: broadcast if src1->ne[1]==1, otherwise use per-expert input
            ಸ್ಥಿರಾಂಕ size_t ಒಳಹರಿವು_ಸ್ಥಾನಾಂತರ =
                (batch_idx * src1->ne[1] + (src1->ne[1] == 1 ? 0 : expert_idx)) * src1->ne[0] * f16_elem_ಗಾತ್ರ;
            ಸ್ಥಿರಾಂಕ size_t output_ಸ್ಥಾನಾಂತರ = (batch_idx * dst->ne[1] + expert_idx) * dst->ne[0] * f16_elem_ಗಾತ್ರ;

            // Create weight view for current expert: [D, M, K] -> [M, D]
            int64_t      ತೂಕ_view_ne[2]  = { ತೂಕ_m, src0->ne[0] };
            ದಶಮಾಂಶ        ತೂಕ_view_nb[2]  = { src0->ne[0] * ತೂಕ_elem_ಗಾತ್ರ, ತೂಕ_elem_ಗಾತ್ರ };
            ಸ್ಥಿರಾಂಕ size_t ತೂಕ_view_ಸ್ಥಾನಾಂತರ = expert_idx * selected_ತೂಕ_nb[2];

            acl_tensor_ptr ತೂಕ_view =
                ggml_cann_create_tensor(selected_ತೂಕ_ಬಫರ್, ggml_cann_type_mapping(type), ತೂಕ_elem_ಗಾತ್ರ,
                                        ತೂಕ_view_ne, ತೂಕ_view_nb, 2, ACL_FORMAT_ND, ತೂಕ_view_ಸ್ಥಾನಾಂತರ);

            // Create scale view for current expert: [D, M, K] -> [M, D]
            int64_t      scale_view_ne[2]  = { ತೂಕ_m, scale_d };
            size_t       scale_view_nb[2]  = { selected_scale_nb[1], selected_scale_nb[0] };
            ಸ್ಥಿರಾಂಕ size_t scale_view_ಸ್ಥಾನಾಂತರ = expert_idx * selected_scale_nb[2];

            acl_tensor_ptr scale_view =
                ggml_cann_create_tensor(selected_scale_ಬಫರ್, ACL_FLOAT16, scale_elem_ಗಾತ್ರ, scale_view_ne,
                                        scale_view_nb, 2, ACL_FORMAT_ND, scale_view_ಸ್ಥಾನಾಂತರ);

            // Create input activation tensor [D, 1]
            int64_t ಒಳಹರಿವು_ne[2] = { src1->ne[0], 1 };
            size_t  ಒಳಹರಿವು_nb[2] = { f16_elem_ಗಾತ್ರ, src1->ne[0] * f16_elem_ಗಾತ್ರ };

            acl_tensor_ptr ಒಳಹರಿವು_tensor = ggml_cann_create_tensor(ಒಳಹರಿವು_ಬಫರ್, ACL_FLOAT16, f16_elem_ಗಾತ್ರ, ಒಳಹರಿವು_ne,
                                                                  ಒಳಹರಿವು_nb, 2, ACL_FORMAT_ND, ಒಳಹರಿವು_ಸ್ಥಾನಾಂತರ);

            // Create output tensor [M, 1]
            int64_t output_ne[2] = { dst->ne[0], 1 };
            size_t  output_nb[2] = { f16_elem_ಗಾತ್ರ, dst->ne[0] * f16_elem_ಗಾತ್ರ };

            acl_tensor_ptr output_tensor = ggml_cann_create_tensor(output_ಬಫರ್, ACL_FLOAT16, f16_elem_ಗಾತ್ರ, output_ne,
                                                                   output_nb, 2, ACL_FORMAT_ND, output_ಸ್ಥಾನಾಂತರ);

            // Perform quantized matrix multiplication
            GGML_CANN_CALL_ACLNN_OP(ctx, ತೂಕ_Quant_Batch_Matmul_V2, ಒಳಹರಿವು_tensor.get(), ತೂಕ_view.get(),
                                    scale_view.get(), ಶೂನ್ಯಸೂಚಕ, ಶೂನ್ಯಸೂಚಕ, ಶೂನ್ಯಸೂಚಕ, ಶೂನ್ಯಸೂಚಕ, ಗುಂಪು_ಗಾತ್ರ,
                                    output_tensor.get());
        }
    }

    // Cast output back to original type if we used a temporary F16 buffer
    ಆದರೆ (dst->type != GGML_TYPE_F16) {
        int64_t ne[GGML_MAX_DIMS];
        size_t  nb[GGML_MAX_DIMS] = { f16_elem_ಗಾತ್ರ };
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < GGML_MAX_DIMS; i++) {
            ne[i] = dst->ne[i];
            ಆದರೆ (i > 0) {
                nb[i] = nb[i - 1] * ne[i - 1];
            }
        }

        acl_tensor_ptr f16_output =
            ggml_cann_create_tensor(output_ಬಫರ್, ACL_FLOAT16, f16_elem_ಗಾತ್ರ, ne, nb, GGML_MAX_DIMS);
        acl_tensor_ptr dst_tensor = ggml_cann_create_tensor(dst);

        aclnn_cast(ctx, f16_output.get(), dst_tensor.get(), ggml_cann_type_mapping(dst->type));
    }
}

ಶೂನ್ಯ ggml_cann_mul_mat_id(ggml_backend_cann_context & ctx, ggml_tensor * dst) {
    ಸ್ಥಿರಾಂಕ ಎಣಿಕೆ ggml_type type = dst->src[0]->type;
    ಆಯ್ಕೆ (type) {
        ಸ್ಥಿತಿ GGML_TYPE_F32:
        ಸ್ಥಿತಿ GGML_TYPE_F16:
            ggml_cann_mul_mat_id_fp(ctx, dst);
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_TYPE_Q4_0:
        ಸ್ಥಿತಿ GGML_TYPE_Q8_0:
            ggml_cann_mul_mat_id_quant(ctx, dst);
            ನಿಲ್ಲಿಸು;
        ಪೂರ್ವನಿಯೋಜಿತ:
            GGML_ABORT("Unsupported type for mul_mat_id");
            ನಿಲ್ಲಿಸು;
    }
}

ಶೂನ್ಯ ggml_cann_flash_attn_ext(ggml_backend_cann_context & ctx, ggml_tensor * dst) {
    ggml_tensor * src0 = dst->src[0];  // q, fp32 | B, N, S, D (uncont) -> B, S, N, D (cont)
    ggml_tensor * src1 = dst->src[1];  // k, fp16 | B, N, S, D (uncont) -> B, S, N, D (cont)
    ggml_tensor * src2 = dst->src[2];  // v, fp16 | B, N, S, D (uncont) -> B, S, N, D (cont)
    ggml_tensor * src3 = dst->src[3];  // mask, fp16

    // B, N, S, D (uncont) -> B, S, N, D (cont)
    int64_t src0_bsnd_ne[GGML_MAX_DIMS];
    ಸ್ಮರಣೆ_ನಕಲು(src0_bsnd_ne, src0->ne, GGML_MAX_DIMS * ಗಾತ್ರ(int64_t));
    size_t src0_bsnd_nb[GGML_MAX_DIMS];
    ಸ್ಮರಣೆ_ನಕಲು(src0_bsnd_nb, src0->nb, GGML_MAX_DIMS * ಗಾತ್ರ(size_t));
    int64_t src1_bsnd_ne[GGML_MAX_DIMS];
    ಸ್ಮರಣೆ_ನಕಲು(src1_bsnd_ne, src1->ne, GGML_MAX_DIMS * ಗಾತ್ರ(int64_t));
    size_t src1_bsnd_nb[GGML_MAX_DIMS];
    ಸ್ಮರಣೆ_ನಕಲು(src1_bsnd_nb, src1->nb, GGML_MAX_DIMS * ಗಾತ್ರ(size_t));
    int64_t src2_bsnd_ne[GGML_MAX_DIMS];
    ಸ್ಮರಣೆ_ನಕಲು(src2_bsnd_ne, src2->ne, GGML_MAX_DIMS * ಗಾತ್ರ(int64_t));
    size_t src2_bsnd_nb[GGML_MAX_DIMS];
    ಸ್ಮರಣೆ_ನಕಲು(src2_bsnd_nb, src2->nb, GGML_MAX_DIMS * ಗಾತ್ರ(size_t));

    ಸ್ವಯಂಚಾಲಿತ transpose12 = [](int64_t * ne, size_t * nb) {
        int64_t ne_tmp = ne[1];
        size_t  nb_tmp = nb[1];
        ne[1]          = ne[2];
        nb[1]          = nb[2];
        ne[2]          = ne_tmp;
        nb[2]          = nb_tmp;
    };

    transpose12(src0_bsnd_ne, src0_bsnd_nb);
    transpose12(src1_bsnd_ne, src1_bsnd_nb);
    transpose12(src2_bsnd_ne, src2_bsnd_nb);

    ದಶಮಾಂಶ maxBias      = 0.0f;
    ದಶಮಾಂಶ scale_ಮೌಲ್ಯ   = 1.0f;
    ದಶಮಾಂಶ logitSoftcap = 0.0f;
    ಸ್ಮರಣೆ_ನಕಲು(&scale_ಮೌಲ್ಯ, (ದಶಮಾಂಶ *) dst->op_params + 0, ಗಾತ್ರ(ದಶಮಾಂಶ));
    ಸ್ಮರಣೆ_ನಕಲು(&maxBias, (ದಶಮಾಂಶ *) dst->op_params + 1, ಗಾತ್ರ(ದಶಮಾಂಶ));
    ಸ್ಮರಣೆ_ನಕಲು(&logitSoftcap, (ದಶಮಾಂಶ *) dst->op_params + 2, ಗಾತ್ರ(ದಶಮಾಂಶ));

    ಆದರೆ (logitSoftcap == 0.0f) {
        size_t fa_Elem_ಗಾತ್ರ = ಗಾತ್ರ(uint16_t);
        ಸ್ವಯಂಚಾಲಿತ   fa_ದತ್ತಾಂಶ_Type = ACL_FLOAT16;  //ACL_BF16;

        acl_tensor_ptr acl_q_tensor = ಶೂನ್ಯಸೂಚಕ;
        acl_tensor_ptr acl_k_tensor = ಶೂನ್ಯಸೂಚಕ;
        acl_tensor_ptr acl_v_tensor = ಶೂನ್ಯಸೂಚಕ;

        // Step 1: cast the src0 (Query) to fp16 if needed
        ggml_cann_pool_alloc src0_f16_allocator(ctx.pool());
        ಶೂನ್ಯ *               src0_f16_ಬಫರ್ = ಶೂನ್ಯಸೂಚಕ;

        ಆದರೆ (ggml_cann_type_mapping(src0->type) != fa_ದತ್ತಾಂಶ_Type) {
            acl_tensor_ptr acl_src0_f32_tensor =
                ggml_cann_create_tensor(src0, src0_bsnd_ne, src0_bsnd_nb, GGML_MAX_DIMS);
            src0_f16_ಬಫರ್ = src0_f16_allocator.alloc(ggml_nelements(src0) * fa_Elem_ಗಾತ್ರ);

            int64_t * src0_f16_ne = src0_bsnd_ne;
            size_t    src0_f16_nb[GGML_MAX_DIMS];
            src0_f16_nb[0] = ಗಾತ್ರ(uint16_t);
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 1; i < GGML_MAX_DIMS; ++i) {
                src0_f16_nb[i] = src0_f16_nb[i - 1] * src0_f16_ne[i - 1];
            }

            acl_q_tensor = ggml_cann_create_tensor(src0_f16_ಬಫರ್, fa_ದತ್ತಾಂಶ_Type, fa_Elem_ಗಾತ್ರ, src0_f16_ne, src0_f16_nb,
                                                   GGML_MAX_DIMS);
            aclnn_cast(ctx, acl_src0_f32_tensor.get(), acl_q_tensor.get(), fa_ದತ್ತಾಂಶ_Type);
        } ಇಲ್ಲದಿದ್ದರೆ {
            acl_q_tensor = ggml_cann_create_tensor(src0, src0_bsnd_ne, src0_bsnd_nb, GGML_MAX_DIMS);
        }

        // Step 2: create the acl tensors for src1 (Key), src2 (Value),
        //         and the direct output from FusedInferAttention

        acl_k_tensor = ggml_cann_create_tensor(src1, src1_bsnd_ne, src1_bsnd_nb, GGML_MAX_DIMS);
        acl_v_tensor = ggml_cann_create_tensor(src2, src2_bsnd_ne, src2_bsnd_nb, GGML_MAX_DIMS);

        // Step 2.5: Pad Q, K, V along head dimension if D is not a multiple of 16
        //           (required by FusedInferAttentionScoreV2)
        ಸ್ಥಿರಾಂಕ int64_t D         = src0->ne[0];
        ಸ್ಥಿರಾಂಕ int64_t D_padded  = GGML_PAD(D, 16);
        ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ needs_padding = (D != D_padded);

        ggml_cann_pool_alloc q_pad_allocator(ctx.pool());
        ggml_cann_pool_alloc k_pad_allocator(ctx.pool());
        ggml_cann_pool_alloc v_pad_allocator(ctx.pool());

        ಆದರೆ (needs_padding) {
            int64_t paddings[] = { 0, D_padded - D, 0, 0, 0, 0, 0, 0 };

            ಸ್ವಯಂಚಾಲಿತ pad_fa_tensor = [&](acl_tensor_ptr & tensor, ಸ್ಥಿರಾಂಕ int64_t * bsnd_ne,
                                     ggml_cann_pool_alloc & allocator) {
                int64_t pad_ne[GGML_MAX_DIMS] = { D_padded, bsnd_ne[1], bsnd_ne[2], bsnd_ne[3] };
                size_t  pad_nb[GGML_MAX_DIMS];
                pad_nb[0] = fa_Elem_ಗಾತ್ರ;
                ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 1; i < GGML_MAX_DIMS; ++i) {
                    pad_nb[i] = pad_nb[i - 1] * pad_ne[i - 1];
                }
                int64_t nelements = pad_ne[0] * pad_ne[1] * pad_ne[2] * pad_ne[3];
                ಶೂನ್ಯ *  ಬಫರ್    = allocator.alloc(nelements * fa_Elem_ಗಾತ್ರ);
                acl_tensor_ptr padded =
                    ggml_cann_create_tensor(ಬಫರ್, fa_ದತ್ತಾಂಶ_Type, fa_Elem_ಗಾತ್ರ, pad_ne, pad_nb, GGML_MAX_DIMS);
                aclnn_pad(ctx, tensor.get(), padded.get(), paddings);
                tensor = std::move(padded);
            };

            pad_fa_tensor(acl_q_tensor, src0_bsnd_ne, q_pad_allocator);
            pad_fa_tensor(acl_k_tensor, src1_bsnd_ne, k_pad_allocator);
            pad_fa_tensor(acl_v_tensor, src2_bsnd_ne, v_pad_allocator);

            src0_bsnd_ne[0] = D_padded;
            src1_bsnd_ne[0] = D_padded;
            src2_bsnd_ne[0] = D_padded;
        }

        // Step 3: create the PSEShift tensor if needed
        //         this tensor is considered as mask (f16) in the llama.cpp
        acl_tensor_ptr       bcast_pse_tensor;
        ggml_cann_pool_alloc bcast_pse_allocator(ctx.pool());
        ಆದರೆ (src3 != ಶೂನ್ಯಸೂಚಕ) {
            // Construct the truncated pse tensor (common for prefill/decode)
            int64_t trunc_pse_ne[GGML_MAX_DIMS] = {
                src3->ne[0],  // D
                src0->ne[1],  // S (number of Q tokens)
                src3->ne[2],  // mask N
                src3->ne[3]   // B
            };
            size_t * trunc_pse_nb = src3->nb;

            acl_tensor_ptr acl_ಮುಖವಾಡ_f16_trunc_tensor = ggml_cann_create_tensor(
                src3->ದತ್ತಾಂಶ, ACL_FLOAT16, ಗಾತ್ರ(uint16_t), trunc_pse_ne, trunc_pse_nb, GGML_MAX_DIMS);

            int64_t bcast_pse_ne[GGML_MAX_DIMS];
            size_t  bcast_pse_nb[GGML_MAX_DIMS];
            bcast_pse_ne[0] = src3->ne[0];  // D
            bcast_pse_ne[1] = src0->ne[1];  // S
            bcast_pse_ne[2] = src0->ne[2];  // N (num_heads)
            bcast_pse_ne[3] = src3->ne[3];  // B
            ಆದರೆ (maxBias == 0.0f) {
                // When maxBias == 0.0f, use nb = 0 reduce once repeat (Qwen2)
                // Construct the bcast tensor (simulate repeat on the head dimension using stride=0)
                bcast_pse_nb[0] = ಗಾತ್ರ(uint16_t);
                bcast_pse_nb[1] = bcast_pse_nb[0] * bcast_pse_ne[0];
                bcast_pse_nb[2] = 0;  // <---- the head dimension shares the same data
                bcast_pse_nb[3] = src3->nb[3];

                bcast_pse_tensor = ggml_cann_create_tensor(src3->ದತ್ತಾಂಶ, ACL_FLOAT16, ಗಾತ್ರ(uint16_t), bcast_pse_ne,
                                                           bcast_pse_nb, GGML_MAX_DIMS);

            } ಇಲ್ಲದಿದ್ದರೆ {
                bcast_pse_nb[0] = ಗಾತ್ರ(uint16_t);
                ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 1; i < GGML_MAX_DIMS; i++) {
                    bcast_pse_nb[i] = bcast_pse_nb[i - 1] * bcast_pse_ne[i - 1];
                }

                ಶೂನ್ಯ * bcast_pse_ಬಫರ್ =
                    bcast_pse_allocator.alloc(ggml_nelements(src3) * src0->ne[2] * ಗಾತ್ರ(uint16_t));

                bcast_pse_tensor = ggml_cann_create_tensor(bcast_pse_ಬಫರ್, ACL_FLOAT16, ಗಾತ್ರ(uint16_t),
                                                           bcast_pse_ne, bcast_pse_nb, GGML_MAX_DIMS);

                int64_t repeats[] = { 1, src0->ne[2], 1, 1 };
                aclnn_repeat(ctx, acl_ಮುಖವಾಡ_f16_trunc_tensor.get(), bcast_pse_tensor.get(), repeats);

                // alibi
                // Compute the slope if needed. Derived from ggml_cann_softmax().
                ಸ್ಥಿರಾಂಕ int64_t        n_heads = src0->ne[2];
                ggml_cann_pool_alloc ಇಳಿಜಾರು_allocator(ctx.pool(), n_heads * ಗಾತ್ರ(uint16_t));
                ಶೂನ್ಯ *               ಇಳಿಜಾರು_ಬಫರ್ = ಇಳಿಜಾರು_allocator.get();
                aclnn_get_ಇಳಿಜಾರು(ctx, n_heads, ಇಳಿಜಾರು_ಬಫರ್, maxBias, GGML_TYPE_F16);

                int64_t ಇಳಿಜಾರು_ne[] = { 1, 1, n_heads, 1 };
                size_t  ಇಳಿಜಾರು_nb[GGML_MAX_DIMS];
                ಇಳಿಜಾರು_nb[0] = ಗಾತ್ರ(uint16_t);
                ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 1; i < GGML_MAX_DIMS; i++) {
                    ಇಳಿಜಾರು_nb[i] = ಇಳಿಜಾರು_nb[i - 1] * ಇಳಿಜಾರು_ne[0];
                }

                acl_tensor_ptr ಇಳಿಜಾರು_tensor = ggml_cann_create_tensor(ಇಳಿಜಾರು_ಬಫರ್, ACL_FLOAT16, ಗಾತ್ರ(uint16_t),
                                                                      ಇಳಿಜಾರು_ne, ಇಳಿಜಾರು_nb, GGML_MAX_DIMS);
                GGML_CANN_CALL_ACLNN_OP(ctx, InplaceMul, bcast_pse_tensor.get(), ಇಳಿಜಾರು_tensor.get());
            }
        }

        // Step 4: set the inputs for FusedInferAttention.
        acl_tensor_list_ptr acl_k_tensor_list = ggml_cann_create_tensor_list(acl_k_tensor);
        acl_tensor_list_ptr acl_v_tensor_list = ggml_cann_create_tensor_list(acl_v_tensor);

        int64_t ಅಂಕೆ_Heads           = src0->ne[2];  // N
        int64_t ಅಂಕೆ_ಕೀಲಿ_ಮೌಲ್ಯ_Heads   = src1->ne[2];
        // double  scaleValue = 1 / sqrt(src0->ne[0]); // 1/sqrt(d)
        int64_t pre_ಟೋಕನುಗಳು          = 65535;
        int64_t ಮುಂದಿನ_ಟೋಕನುಗಳು         = 65535;
        ಅಕ್ಷರ    layout[5]          = { 'B', 'S', 'N', 'D', 0 };
        int64_t sparse_ಬಹುಲಕ         = 0;
        int64_t innerPrecise       = (src0->ne[1] == 1) ? 0 : 2;
        int64_t ಖಂಡ_ಗಾತ್ರ          = 0;
        int64_t antiquant_ಬಹುಲಕ      = 0;
        ತಾರ್ಕಿಕ    softmaxLseFlag     = ಸುಳ್ಳು;
        int64_t ಕೀಲಿ_Antiquant_ಬಹುಲಕ   = 0;
        int64_t ಮೌಲ್ಯ_Antiquant_ಬಹುಲಕ = 0;

        GGML_ASSERT(dst->type == GGML_TYPE_F32 || dst->type == GGML_TYPE_F16);
        acl_tensor_ptr       fa_dst_tensor;
        ggml_cann_pool_alloc ಹೊರಹೋಗು_f16_allocator(ctx.pool());
        ಆದರೆ (dst->type == GGML_TYPE_F32 || needs_padding) {
            int64_t * ಹೊರಹೋಗು_f16_ne = src0_bsnd_ne;
            size_t    ಹೊರಹೋಗು_f16_nb[GGML_MAX_DIMS];
            ಹೊರಹೋಗು_f16_nb[0] = fa_Elem_ಗಾತ್ರ;
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 1; i < GGML_MAX_DIMS; ++i) {
                ಹೊರಹೋಗು_f16_nb[i] = ಹೊರಹೋಗು_f16_nb[i - 1] * ಹೊರಹೋಗು_f16_ne[i - 1];
            }
            int64_t ಹೊರಹೋಗು_nelements = ಹೊರಹೋಗು_f16_ne[0] * ಹೊರಹೋಗು_f16_ne[1] * ಹೊರಹೋಗು_f16_ne[2] * ಹೊರಹೋಗು_f16_ne[3];
            ಶೂನ್ಯ *  ಹೊರಹೋಗು_f16_ಬಫರ್ = ಹೊರಹೋಗು_f16_allocator.alloc(ಹೊರಹೋಗು_nelements * fa_Elem_ಗಾತ್ರ);

            fa_dst_tensor =
                ggml_cann_create_tensor(ಹೊರಹೋಗು_f16_ಬಫರ್, fa_ದತ್ತಾಂಶ_Type, fa_Elem_ಗಾತ್ರ, ಹೊರಹೋಗು_f16_ne, ಹೊರಹೋಗು_f16_nb, GGML_MAX_DIMS);
        } ಇಲ್ಲದಿದ್ದರೆ {
            fa_dst_tensor = ggml_cann_create_tensor(dst);
        }

        GGML_CANN_CALL_ACLNN_OP(ctx, FusedInferAttentionScoreV2, acl_q_tensor.get(), acl_k_tensor_list.get(),
                                acl_v_tensor_list.get(),               // q, k, v
                                bcast_pse_tensor.get(), ಶೂನ್ಯಸೂಚಕ,       // pse, mask
                                ಶೂನ್ಯಸೂಚಕ, ಶೂನ್ಯಸೂಚಕ,                      // actSeqLen, actSeqLenkv
                                ಶೂನ್ಯಸೂಚಕ, ಶೂನ್ಯಸೂಚಕ,                      // deqScale1, quantScale1
                                ಶೂನ್ಯಸೂಚಕ, ಶೂನ್ಯಸೂಚಕ, ಶೂನ್ಯಸೂಚಕ,             // deqScale2, quantScale2, quantOffset2
                                ಶೂನ್ಯಸೂಚಕ, ಶೂನ್ಯಸೂಚಕ,                      // antiquantScale, antiquantOffset
                                ಶೂನ್ಯಸೂಚಕ,                               // blockTable
                                ಶೂನ್ಯಸೂಚಕ, ಶೂನ್ಯಸೂಚಕ,                      // qPadSize, kvPadSize
                                ಶೂನ್ಯಸೂಚಕ, ಶೂನ್ಯಸೂಚಕ,                      // kAntiquantScale, kAntiQuantOffset
                                ಶೂನ್ಯಸೂಚಕ, ಶೂನ್ಯಸೂಚಕ,                      // vAntiquantScale, vAntiQuantOffset
                                ಶೂನ್ಯಸೂಚಕ, ಶೂನ್ಯಸೂಚಕ, ಶೂನ್ಯಸೂಚಕ,             // kSharedPrefix, vSharedPrefix, actSharedLen
                                ಅಂಕೆ_Heads, scale_ಮೌಲ್ಯ,                  // heads, scaleValue
                                pre_ಟೋಕನುಗಳು, ಮುಂದಿನ_ಟೋಕನುಗಳು,                 // preTokens, nextTokens
                                layout,                                // inputLayout
                                ಅಂಕೆ_ಕೀಲಿ_ಮೌಲ್ಯ_Heads,                      // numKVHeads
                                sparse_ಬಹುಲಕ, innerPrecise,              // sparseMode, innerPrecise
                                ಖಂಡ_ಗಾತ್ರ, antiquant_ಬಹುಲಕ,              // blockSize, antiquantMode
                                softmaxLseFlag,                        // softmaxLseFlag
                                ಕೀಲಿ_Antiquant_ಬಹುಲಕ, ಮೌಲ್ಯ_Antiquant_ಬಹುಲಕ,  // keyAntiqMode, valueAntiqMode
                                fa_dst_tensor.get(),                   // attentionOut
                                ಶೂನ್ಯಸೂಚಕ                                // softmaxLse
        );

        // Step 6: post-processing — slice padded output and/or cast to f32
        ಆದರೆ (needs_padding) {
            ggml_cann_pool_alloc sliced_f16_allocator(ctx.pool());

            ಆದರೆ (dst->type == GGML_TYPE_F32) {
                int64_t sliced_ne[GGML_MAX_DIMS] = { D, src0_bsnd_ne[1], src0_bsnd_ne[2], src0_bsnd_ne[3] };
                size_t  sliced_nb[GGML_MAX_DIMS];
                sliced_nb[0] = fa_Elem_ಗಾತ್ರ;
                ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 1; i < GGML_MAX_DIMS; ++i) {
                    sliced_nb[i] = sliced_nb[i - 1] * sliced_ne[i - 1];
                }
                int64_t sliced_nelements = sliced_ne[0] * sliced_ne[1] * sliced_ne[2] * sliced_ne[3];
                ಶೂನ್ಯ *  sliced_ಬಫರ್    = sliced_f16_allocator.alloc(sliced_nelements * fa_Elem_ಗಾತ್ರ);
                acl_tensor_ptr sliced_f16_tensor = ggml_cann_create_tensor(sliced_ಬಫರ್, fa_ದತ್ತಾಂಶ_Type, fa_Elem_ಗಾತ್ರ,
                                                                           sliced_ne, sliced_nb, GGML_MAX_DIMS);

                GGML_CANN_CALL_ACLNN_OP(ctx, ತುಂಡು, fa_dst_tensor.get(),
                                        (int64_t) -1, (int64_t) 0, D, (int64_t) 1, sliced_f16_tensor.get());

                acl_tensor_ptr acl_dst_tensor = ggml_cann_create_tensor(dst);
                aclnn_cast(ctx, sliced_f16_tensor.get(), acl_dst_tensor.get(), ggml_cann_type_mapping(dst->type));
            } ಇಲ್ಲದಿದ್ದರೆ {
                acl_tensor_ptr acl_dst_tensor = ggml_cann_create_tensor(dst);
                GGML_CANN_CALL_ACLNN_OP(ctx, ತುಂಡು, fa_dst_tensor.get(),
                                        (int64_t) -1, (int64_t) 0, D, (int64_t) 1, acl_dst_tensor.get());
            }
        } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (dst->type == GGML_TYPE_F32) {
            acl_tensor_ptr acl_dst_tensor = ggml_cann_create_tensor(dst);
            aclnn_cast(ctx, fa_dst_tensor.get(), acl_dst_tensor.get(), ggml_cann_type_mapping(dst->type));
        }
    } ಇಲ್ಲದಿದ್ದರೆ {
        GGML_ABORT("Function is not implemented.");
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_cann_ಹೊರಹೋಗು_prod_fp(ggml_backend_cann_context & ctx, ggml_tensor * dst) {
    ggml_tensor * src0 = dst->src[0];  // weight  [ne00=m, ne01=K, ne02, ne03]
    ggml_tensor * src1 = dst->src[1];  // input   [ne10=n, ne11=K, ne12, ne13]
    GGML_TENSOR_ದ್ವಿಮಾನ_OP_LOCALS

    // dst[i,j] = sum_k src0[i,k] * src1[j,k]  i.e. dst = src0 @ src1^T.
    //
    // ggml_cann_create_tensor reverses dimension order, so ACL sees:
    //   acl_src0 slice:   ggml[m,K]  ->  ACL[K,m]
    //   acl_src1 slice:   ggml[n,K]  ->  ACL[K,n]
    //   acl_dst  slice:   ggml[m,n]  ->  ACL[n,m]
    //
    // Build a transposed view of src1 by swapping ne[0]/ne[1]:
    //   src1_t:  ggml[K,n] (swapped strides)  ->  ACL[n,K]
    //
    // Matmul(src1_t [n,K], src0 [K,m]) = [n,m] = acl_dst  ✓
    //
    // The outer batch loop is kept because src0 may have fewer batch slices than
    // dst (ne02 <= ne2, ne03 <= ne3): this is a strided-broadcast not supported
    // by standard CANN Matmul broadcasting.

    ಸ್ಥಿರಾಂಕ acl_ದತ್ತಾಂಶ_Type src0_acl_type = ggml_cann_type_mapping(src0->type);
    ಸ್ಥಿರಾಂಕ acl_ದತ್ತಾಂಶ_Type src1_acl_type = ggml_cann_type_mapping(src1->type);
    ಸ್ಥಿರಾಂಕ acl_ದತ್ತಾಂಶ_Type dst_acl_type  = ggml_cann_type_mapping(dst->type);
    ಸ್ಥಿರಾಂಕ size_t      src0_type_sz  = ggml_type_ಗಾತ್ರ(src0->type);
    ಸ್ಥಿರಾಂಕ size_t      src1_type_sz  = ggml_type_ಗಾತ್ರ(src1->type);
    ಸ್ಥಿರಾಂಕ size_t      dst_type_sz   = ggml_type_ಗಾತ್ರ(dst->type);

    ಸ್ಥಿರಾಂಕ int64_t dps2 = ne2 / ne02;
    ಸ್ಥಿರಾಂಕ int64_t dps3 = ne3 / ne03;

    ಗಾಗಿ (int64_t i3 = 0; i3 < ne3; i3++) {
        ಗಾಗಿ (int64_t i2 = 0; i2 < ne2; i2++) {
            ಸ್ಥಿರಾಂಕ int64_t i02 = i2 / dps2;
            ಸ್ಥಿರಾಂಕ int64_t i03 = i3 / dps3;

            // src0 2D slice at [i02, i03]: ggml [m, K] -> ACL [K, m]
            int64_t src0_ne[2] = { ne00, ne01 };
            size_t  src0_nb[2] = { nb00, nb01 };
            acl_tensor_ptr acl_src0_s = ggml_cann_create_tensor(
                (ಅಕ್ಷರ *) src0->ದತ್ತಾಂಶ + i02 * nb02 + i03 * nb03,
                src0_acl_type, src0_type_sz, src0_ne, src0_nb, 2);

            // src1 transposed 2D slice at [i2, i3]: swap ne/nb -> ggml[K,n] -> ACL[n,K]
            int64_t src1_t_ne[2] = { ne11, ne10 };
            size_t  src1_t_nb[2] = { nb11, nb10 };
            acl_tensor_ptr acl_src1_t = ggml_cann_create_tensor(
                (ಅಕ್ಷರ *) src1->ದತ್ತಾಂಶ + i2 * nb12 + i3 * nb13,
                src1_acl_type, src1_type_sz, src1_t_ne, src1_t_nb, 2);

            // dst 2D slice at [i2, i3]: ggml [m, n] -> ACL [n, m]
            int64_t dst_ne[2] = { ne0, ne1 };
            size_t  dst_nb[2] = { nb0, nb1 };
            acl_tensor_ptr acl_dst_s = ggml_cann_create_tensor(
                (ಅಕ್ಷರ *) dst->ದತ್ತಾಂಶ + i2 * nb2 + i3 * nb3,
                dst_acl_type, dst_type_sz, dst_ne, dst_nb, 2);

            // Matmul(src1_t [n,K], src0 [K,m]) = [n,m] = acl_dst_s  ✓
            GGML_CANN_CALL_ACLNN_OP(ctx, Matmul,
                acl_src1_t.get(), acl_src0_s.get(), acl_dst_s.get(), (int8_t) 1);
        }
    }
}

ಶೂನ್ಯ ggml_cann_ಹೊರಹೋಗು_prod(ggml_backend_cann_context & ctx, ggml_tensor * dst) {
    ggml_tensor * src0 = dst->src[0];

    ಸ್ಥಿರಾಂಕ ಎಣಿಕೆ ggml_type type = src0->type;

    ಆಯ್ಕೆ (type) {
        ಸ್ಥಿತಿ GGML_TYPE_F32:
        ಸ್ಥಿತಿ GGML_TYPE_F16:
            ggml_cann_ಹೊರಹೋಗು_prod_fp(ctx, dst);
            ನಿಲ್ಲಿಸು;
        ಪೂರ್ವನಿಯೋಜಿತ:
            GGML_ABORT("Unsupport type for GGML_OP_OUT_PROD");
            ನಿಲ್ಲಿಸು;
    }
}

ಶೂನ್ಯ ggml_cann_ssm_conv(ggml_backend_cann_context & ctx, ggml_tensor * dst) {
    ggml_tensor * src0 = dst->src[0];  // conv_x
    ggml_tensor * src1 = dst->src[1];  // conv1d.weight

    // This op is currently defined only for F32 in ggml_cpu
    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    // Shapes follow ggml_compute_forward_ssm_conv_f32
    ಸ್ಥಿರಾಂಕ int64_t nc  = src1->ne[0];   // d_conv
    ಸ್ಥಿರಾಂಕ int64_t ncs = src0->ne[0];   // d_conv - 1 + n_t
    ಸ್ಥಿರಾಂಕ int64_t nr  = src0->ne[1];   // d_inner
    ಸ್ಥಿರಾಂಕ int64_t n_s = src0->ne[2];   // n_seqs

    ಸ್ಥಿರಾಂಕ int64_t n_t = dst->ne[1];    // tokens per sequence

    GGML_ASSERT(dst->ne[0] == nr);     // dst: {d_inner, n_t, n_s}
    GGML_ASSERT(src1->ne[1] == nr);    // weight: {d_conv, d_inner}
    GGML_ASSERT(ncs == nc - 1 + n_t);  // conv_x: {d_conv - 1 + n_t, d_inner, n_s}
    GGML_ASSERT(src0->nb[0] == ಗಾತ್ರ(ದಶಮಾಂಶ));
    GGML_ASSERT(src1->nb[0] == ಗಾತ್ರ(ದಶಮಾಂಶ));

    // --- Build CANN tensors ---

    // 1) Input: conv_x as NCL
    //
    // src0->ne = { ncs, nr, n_s, 1 }  // {L_in, C, N}
    // Passing ACL_FORMAT_NCL here means:
    //   reversed dims -> [N, C, L_in] = [n_s, nr, ncs]
    acl_tensor_ptr acl_x = ggml_cann_create_tensor(src0, src0->ne, src0->nb, 3, ACL_FORMAT_NCL);

    // 2) Weights: depthwise conv kernel, view src1 as {K, 1, C}
    //
    // src1 original:   ne = { nc, nr, 1, 1 }  // [K, C, 1, 1]
    // we want a view:  ne_w = { nc, 1, nr }   // [K, 1, C]
    // so that reversed dims -> [C, 1, K] which matches
    //   [out_channels, in_channels/groups, kernel_size]
    int64_t w_ne[GGML_MAX_DIMS] = { nc, 1, nr, 1 };  // [K, 1 input ch. per group, C groups]
    // Layout: src1 data is [K, C] with
    //   offset(k, c) = k*nb0 + c*nb1
    // We want offset_w(k, 0, c) = k*nb0 + c*nb1,
    // so we can reuse nb0 and nb1, and set nb2 = nb1.
    size_t  w_nb[GGML_MAX_DIMS] = { src1->nb[0], src1->nb[1], src1->nb[1], src1->nb[3] };  // same as src1

    acl_tensor_ptr acl_w = ggml_cann_create_tensor(src1->ದತ್ತಾಂಶ, ggml_cann_type_mapping(src1->type),
                                                   ggml_type_ಗಾತ್ರ(src1->type), w_ne, w_nb, 3, ACL_FORMAT_NCL);

    // 3) Output: dst is { d_inner, n_t, n_s } (CLN)
    //
    // We need an NCL view of the same buffer:
    //   desired NCL logical shape: { L_out = n_t, C = nr, N = n_s }
    //
    // Original CLN layout:
    //   dst->ne = { nr, n_t, n_s }
    //   dst->nb[0] = sizeof(float)
    //   dst->nb[1] = nr * sizeof(float)
    //   dst->nb[2] = nr * n_t * sizeof(float)
    //
    // We want offset_new(L, C, N) = offset_orig(C, L, N).
    // Choose:
    //   nb_y[0] = nr * sizeof(float);           // step in L
    //   nb_y[1] = sizeof(float);                // step in C
    //   nb_y[2] = nr * n_t * sizeof(float);     // step in N
    int64_t y_ne[GGML_MAX_DIMS] = { n_t, nr, n_s, 1 };  // [L_out, C, N]
    size_t  y_nb[GGML_MAX_DIMS] = { dst->ne[0] * ಗಾತ್ರ(ದಶಮಾಂಶ), ಗಾತ್ರ(ದಶಮಾಂಶ), dst->ne[0] * dst->ne[1] * ಗಾತ್ರ(ದಶಮಾಂಶ),
                                    dst->nb[3] };       // [nr, 1, nr * n_t]

    acl_tensor_ptr acl_y = ggml_cann_create_tensor(dst->ದತ್ತಾಂಶ, ggml_cann_type_mapping(dst->type),
                                                   ggml_type_ಗಾತ್ರ(dst->type), y_ne, y_nb, 3, ACL_FORMAT_NCL);

    // --- Conv1d parameters: depthwise, stride 1, no padding ("valid") ---
    int64_t strideVal[1]   = { 1 };
    int64_t paddingVal[1]  = { 0 };
    int64_t dilationVal[1] = { 1 };

    acl_int_array_ptr stride   = ggml_cann_create_int_array(strideVal, 1);
    acl_int_array_ptr padding  = ggml_cann_create_int_array(paddingVal, 1);
    acl_int_array_ptr dilation = ggml_cann_create_int_array(dilationVal, 1);

    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ    transposed   = ಸುಳ್ಳು;
    ಸ್ಥಿರಾಂಕ int64_t groups       = nr;  // depthwise: one group per inner dim
    int8_t        cubeMathType = 0;

#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ ASCEND_310P
    cubeMathType = 1;
#ಆದರೆ_ಅಂತ್ಯ

    GGML_CANN_CALL_ACLNN_OP(ctx, Convolution,
                            acl_x.get(),    // input:  N, C, L_in = ncs
                            acl_w.get(),    // weight: [C, 1, K] with groups=nr
                            ಶೂನ್ಯಸೂಚಕ,        // bias
                            stride.get(), padding.get(), dilation.get(), transposed,
                            padding.get(),  // output padding (unused for non-transposed)
                            groups, acl_y.get(), cubeMathType);
}

ಶೂನ್ಯ ggml_cann_op_ಕೂಡಿಸು_rms_norm_fused(ggml_backend_cann_context & ctx,
                                     ggml_tensor *               ಕೂಡಿಸು_ಗಂಟು,
                                     ggml_tensor *               rms_norm_ಗಂಟು) {
    // Get the two input tensors for ADD operation
    ggml_tensor * x1 = ಕೂಡಿಸು_ಗಂಟು->src[0];
    ggml_tensor * x2 = ಕೂಡಿಸು_ಗಂಟು->src[1];

    // Create ACL tensors for the two ADD inputs
    acl_tensor_ptr acl_x1 = ggml_cann_create_tensor(x1);
    acl_tensor_ptr acl_x2 = ggml_cann_create_tensor(x2);

    // Get epsilon parameter from rms_norm_tensor
    ದಶಮಾಂಶ eps;
    ಸ್ಮರಣೆ_ನಕಲು(&eps, rms_norm_ಗಂಟು->op_params, ಗಾತ್ರ(ದಶಮಾಂಶ));

    // Build gamma tensor (RMS normalization scaling factor)
    // Gamma should match the normalized dimensions (last dimension of x1)
    size_t acl_gamma_nb[GGML_MAX_DIMS];
    acl_gamma_nb[0] = ggml_type_ಗಾತ್ರ(rms_norm_ಗಂಟು->type);
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 1; i < GGML_MAX_DIMS; i++) {
        acl_gamma_nb[i] = acl_gamma_nb[i - 1] * x1->ne[i - 1];
    }
    acl_tensor_ptr acl_gamma =
        get_cache_acl_tensor(ctx, &ctx.rms_norm_one_tensor_cache.cache, ctx.rms_norm_one_tensor_cache.size, x1->ne,
                             acl_gamma_nb, rms_norm_ಗಂಟು->type,
                             1,    // dims - only the last dimension
                             1.0f  //ಮೌಲ್ಯ
        );

    // Build rstdOut tensor (output for normalized standard deviation)
    // Shape should be the dimensions that are NOT normalized
    int64_t acl_rstd_ne[] = { 1, x1->ne[1], x1->ne[2], x1->ne[3] };
    size_t  acl_rstd_nb[GGML_MAX_DIMS - 1];
    acl_rstd_nb[0] = ಗಾತ್ರ(ದಶಮಾಂಶ);
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 1; i < GGML_MAX_DIMS - 1; i++) {
        acl_rstd_nb[i] = acl_rstd_nb[i - 1] * acl_rstd_ne[i - 1];
    }
    acl_tensor_ptr acl_rstd =
        get_cache_acl_tensor(ctx, &ctx.rms_norm_zero_tensor_cache.cache, ctx.rms_norm_zero_tensor_cache.size,
                             acl_rstd_ne, acl_rstd_nb, GGML_TYPE_F32, GGML_MAX_DIMS,
                             0.0f  //ಮೌಲ್ಯ
        );

    acl_tensor_ptr acl_xout = ggml_cann_create_tensor(ಕೂಡಿಸು_ಗಂಟು);

    // Create yOut tensor (final output after RMS normalization)
    acl_tensor_ptr acl_yout = ggml_cann_create_tensor(rms_norm_ಗಂಟು);

    // Call fused ADD + RMS_NORM operator
    GGML_CANN_CALL_ACLNN_OP(ctx, ಕೂಡಿಸು_Rms_Norm, acl_x1.get(), acl_x2.get(), acl_gamma.get(),
                            eps,  // double type
                            acl_yout.get(), acl_rstd.get(), acl_xout.get());
}

ಶೂನ್ಯ ggml_cann_gated_ರೇಖೀಯ_attn(ggml_backend_cann_context & ctx, ggml_tensor * dst) {
    ggml_tensor * k = dst->src[0];
    ggml_tensor * v = dst->src[1];
    ggml_tensor * q = dst->src[2];
    ggml_tensor * g = dst->src[3];
    ggml_tensor * s = dst->src[4];

    int64_t B = dst->src[4]->ne[1];
    int64_t T = dst->src[0]->ne[2];
    int64_t H = dst->src[0]->ne[1];
    int64_t C = dst->ne[0];
    int64_t D = C / H;
    int64_t L = T / B;

    int64_t ne_qkg[2] = { 1, D };
    int64_t ne_s[2]   = { D, D };
    int64_t ne_st[2]  = { ne_s[1], ne_s[0] };
    int64_t ne_vo[2]  = { D, 1 };
    int64_t ne_q[1]   = { D };
    size_t  nb_ತಳ   = ggml_type_ಗಾತ್ರ(k->type);
    size_t  nb_qkg[2] = { nb_ತಳ, nb_ತಳ };
    size_t  nb_s[2]   = { nb_ತಳ, D * nb_ತಳ };
    size_t  nb_st[2]  = { nb_s[1], nb_s[0] };
    size_t  nb_vo[2]  = { nb_ತಳ, D * nb_ತಳ };
    size_t  nb_q[1]   = { nb_ತಳ };

    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ scale = ggml_get_op_params_f32(dst, 0);

    acl_tensor_ptr acl_s     = ggml_cann_create_tensor(s, s->ne, s->nb, 2, ACL_FORMAT_ND);
    acl_tensor_ptr new_state = ggml_cann_create_tensor(dst, s->ne, s->nb, 2, ACL_FORMAT_ND, (B * L * H * D) * nb_ತಳ);
    cann_copy(ctx, acl_s.get(), new_state.get());

    ಗಾಗಿ (int64_t b = 0; b < B; b++) {
        ಗಾಗಿ (int64_t h = 0; h < H; h++) {
            size_t         s_ಸ್ಥಾನಾಂತರ = (b * (H * D * D) + h * (D * D)) * nb_ತಳ;
            // D * D
            acl_tensor_ptr acl_s_new =
                ggml_cann_create_tensor(dst, ne_s, nb_s, 2, ACL_FORMAT_ND, (B * L * H * D) * nb_ತಳ + s_ಸ್ಥಾನಾಂತರ);
            acl_tensor_ptr acl_s_new_t =
                ggml_cann_create_tensor(dst, ne_st, nb_st, 2, ACL_FORMAT_ND, (B * L * H * D) * nb_ತಳ + s_ಸ್ಥಾನಾಂತರ);
            ಗಾಗಿ (int64_t l = 0; l < L; l++) {
                size_t               qkvgo_ಸ್ಥಾನಾಂತರ = (b * (L * H * D) + l * (H * D) + h * (D)) * nb_ತಳ;
                // D * 1
                acl_tensor_ptr       acl_k = ggml_cann_create_tensor(k, ne_qkg, nb_qkg, 2, ACL_FORMAT_ND, qkvgo_ಸ್ಥಾನಾಂತರ);
                acl_tensor_ptr       acl_g = ggml_cann_create_tensor(g, ne_qkg, nb_qkg, 2, ACL_FORMAT_ND, qkvgo_ಸ್ಥಾನಾಂತರ);
                // D
                acl_tensor_ptr       acl_q = ggml_cann_create_tensor(q, ne_q, nb_q, 1, ACL_FORMAT_ND, qkvgo_ಸ್ಥಾನಾಂತರ);
                // 1 * D
                acl_tensor_ptr       acl_v = ggml_cann_create_tensor(v, ne_vo, nb_vo, 2, ACL_FORMAT_ND, qkvgo_ಸ್ಥಾನಾಂತರ);
                // D
                acl_tensor_ptr       acl_o = ggml_cann_create_tensor(dst, ne_q, nb_q, 1, ACL_FORMAT_ND, qkvgo_ಸ್ಥಾನಾಂತರ);
                // k ⊗ v
                size_t               buf_ಗಾತ್ರ = D * D * nb_ತಳ;
                ggml_cann_pool_alloc ಬಫರ್_allocator(ctx.pool(), buf_ಗಾತ್ರ);
                acl_tensor_ptr       tmp_tensor = ggml_cann_create_tensor(
                    ಬಫರ್_allocator.get(), ggml_cann_type_mapping(k->type), nb_ತಳ, ne_s, nb_s, 2);
                aclnn_mul(ctx, acl_k.get(), acl_v.get(), tmp_tensor.get());
                //s_new = g ⊗ s_old + k ⊗ v
                aclnn_mul(ctx, acl_s_new.get(), acl_g.get(), ಶೂನ್ಯಸೂಚಕ);
                aclnn_ಕೂಡಿಸು(ctx, acl_s_new.get(), tmp_tensor.get(), ಶೂನ್ಯಸೂಚಕ);
                // compute output
                GGML_CANN_CALL_ACLNN_OP(ctx, Mv, acl_s_new_t.get(), acl_q.get(), acl_o.get(), 1);
                aclnn_muls(ctx, acl_o.get(), scale, ಶೂನ್ಯಸೂಚಕ, ನಿಜ);
            }
        }
    }
}
