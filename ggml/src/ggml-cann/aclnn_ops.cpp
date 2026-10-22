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

#अन्तर्भाव "aclnn_ops.h"

#अन्तर्भाव "ggml-impl.h"
#अन्तर्भाव "ggml.h"


#अन्तर्भाव <aclnnop/aclnn_add.h>
#अन्तर्भाव <aclnnop/aclnn_add_rms_norm.h>
#अन्तर्भाव <aclnnop/aclnn_addcdiv.h>
#अन्तर्भाव <aclnnop/aclnn_argmax.h>
#अन्तर्भाव <aclnnop/aclnn_avgpool2d.h>
#अन्तर्भाव <aclnnop/aclnn_batch_matmul.h>
#अन्तर्भाव <aclnnop/aclnn_cast.h>
#अन्तर्भाव <aclnnop/aclnn_clamp.h>
#अन्तर्भाव <aclnnop/aclnn_constant_pad_nd.h>
#अन्तर्भाव <aclnnop/aclnn_convolution.h>
#अन्तर्भाव <aclnnop/aclnn_copy.h>
#अन्तर्भाव <aclnnop/aclnn_div.h>
#अन्तर्भाव <aclnnop/aclnn_elu.h>
#अन्तर्भाव <aclnnop/aclnn_embedding.h>
#अन्तर्भाव <aclnnop/aclnn_eq_tensor.h>
#अन्तर्भाव <aclnnop/aclnn_exp.h>
#अन्तर्भाव <aclnnop/aclnn_fill_scalar.h>
#अन्तर्भाव <aclnnop/aclnn_fused_infer_attention_score_v2.h>
#अन्तर्भाव <aclnnop/aclnn_ger.h>
#अन्तर्भाव <aclnnop/aclnn_group_norm.h>
#अन्तर्भाव <aclnnop/aclnn_gather_v2.h>
#अन्तर्भाव <aclnnop/aclnn_grouped_matmul_v3.h>
#अन्तर्भाव <aclnnop/aclnn_scatter.h>
#अन्तर्भाव <aclnnop/aclnn_gt_scalar.h>
#अन्तर्भाव <aclnnop/aclnn_im2col.h>
#अन्तर्भाव <aclnnop/aclnn_index_copy.h>
#अन्तर्भाव <aclnnop/aclnn_index_fill_tensor.h>
#अन्तर्भाव <aclnnop/aclnn_index_select.h>
#अन्तर्भाव <aclnnop/aclnn_layer_norm.h>
#अन्तर्भाव <aclnnop/aclnn_log.h>
#अन्तर्भाव <aclnnop/aclnn_matmul.h>
#अन्तर्भाव <aclnnop/aclnn_max_pool.h>
#अन्तर्भाव <aclnnop/aclnn_mean.h>
#अन्तर्भाव <aclnnop/aclnn_mm.h>
#अन्तर्भाव <aclnnop/aclnn_mul.h>
#अन्तर्भाव <aclnnop/aclnn_mv.h>
#अन्तर्भाव <aclnnop/aclnn_permute.h>
#अन्तर्भाव <aclnnop/aclnn_pow.h>
#अन्तर्भाव <aclnnop/aclnn_pow_tensor_tensor.h>
#अन्तर्भाव <aclnnop/aclnn_recurrent_gated_delta_rule.h>
#अन्तर्भाव <aclnnop/aclnn_reduce_sum.h>
#अन्तर्भाव <aclnnop/aclnn_reflection_pad1d.h>
#अन्तर्भाव <aclnnop/aclnn_repeat.h>
#अन्तर्भाव <aclnnop/aclnn_repeat_interleave.h>
#अन्तर्भाव <aclnnop/aclnn_rms_norm.h>
#अन्तर्भाव <aclnnop/aclnn_roll.h>
#अन्तर्भाव <aclnnop/aclnn_softmax.h>
#अन्तर्भाव <aclnnop/aclnn_softmax_cross_entropy_with_logits.h>
#अन्तर्भाव <aclnnop/aclnn_sub.h>
#अन्तर्भाव <aclnnop/aclnn_sum.h>
#अन्तर्भाव <aclnnop/aclnn_threshold.h>
#अन्तर्भाव <aclnnop/aclnn_tril.h>
#अन्तर्भाव <aclnnop/aclnn_triangular_solve.h>
#अन्तर्भाव <aclnnop/aclnn_triu.h>
#अन्तर्भाव <aclnnop/aclnn_logical_not.h>
#अन्तर्भाव <aclnnop/aclnn_masked_fill_scalar.h>
#अन्तर्भाव <aclnnop/aclnn_upsample_nearest_2d.h>
#अन्तर्भाव <aclnnop/aclnn_weight_quant_batch_matmul_v2.h>
#अन्तर्भाव <aclnnop/aclnn_zero.h>
#अन्तर्भाव <float.h>

#अन्तर्भाव <cmath>
#अन्तर्भाव <cstring>
#अन्तर्भाव <exception>
#अन्तर्भाव <vector>

#define GGML_COMMON_DECL_C

#अन्तर्भाव "../ggml-common.h"

शून्य bcast_shape(ggml_tensor *    src0,
                 ggml_tensor *    src1,
                 ggml_tensor *    dst,
                 acl_tensor_ptr & acl_src0,
                 acl_tensor_ptr & acl_src1,
                 acl_tensor_ptr & acl_dst) {
    GGML_ASSERT(ggml_are_same_shape(src0, dst) && ggml_can_repeat(src1, src0));
    // Need bcast
    यदि (!ggml_are_same_shape(src0, src1) && ggml_cann_need_bcast(src0, src1)) {
        BCAST_SHAPE(src0, src1)
        acl_src0 = ggml_cann_create_tensor(src0, BCAST_PARAM(src0));
        acl_src1 = ggml_cann_create_tensor(src1, BCAST_PARAM(src1));
        acl_dst  = ggml_cann_create_tensor(dst, BCAST_PARAM(src0));
    } अन्यथा {
        acl_src0 = ggml_cann_create_tensor(src0);
        acl_src1 = ggml_cann_create_tensor(src1);
        acl_dst  = ggml_cann_create_tensor(dst);
    }
}

शून्य ggml_cann_op_unary(std::function<शून्य(ggml_backend_cann_context &, aclTensor *, aclTensor *)> unary_op,
                        ggml_backend_cann_context &                                                ctx,
                        ggml_tensor *                                                              dst) {
    ggml_tensor * src = dst->src[0];

    acl_tensor_ptr acl_src = ggml_cann_create_tensor(src);
    acl_tensor_ptr acl_dst = ggml_cann_create_tensor(dst);

    unary_op(ctx, acl_src.get(), acl_dst.get());
}

शून्य ggml_cann_op_unary_gated(std::function<शून्य(ggml_backend_cann_context &, aclTensor *, aclTensor *)> unary_op,
                              ggml_backend_cann_context &                                                ctx,
                              ggml_tensor *                                                              dst) {
    ggml_tensor * src0 = dst->src[0];
    ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(ggml_is_contiguous_1(src0));
    GGML_ASSERT(ggml_is_contiguous_1(dst));
    अचर int32_t swapped = ggml_get_op_params_i32(dst, 1);

    acl_tensor_ptr acl_dst = ggml_cann_create_tensor(dst);
    acl_tensor_ptr acl_src0, acl_src1;
    यदि (src1) {
        GGML_ASSERT(ggml_is_contiguous_1(src1));
        GGML_ASSERT(src0->type == src1->type);

        acl_src0 = ggml_cann_create_tensor(src0);
        acl_src1 = ggml_cann_create_tensor(src1);
    } अन्यथा {
        int64_t ne[] = { src0->ne[0] / 2, src0->ne[1], src0->ne[2], src0->ne[3] };
        माप_प्रकार  nb[] = { src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3] };
        acl_src0     = ggml_cann_create_tensor(src0, ne, nb, GGML_MAX_DIMS, ACL_FORMAT_ND, 0);
        acl_src1 = ggml_cann_create_tensor(src0, ne, nb, GGML_MAX_DIMS, ACL_FORMAT_ND, ne[0] * ggml_element_size(src0));
        यदि (swapped) {
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
शून्य ggml_cann_swiglu(ggml_backend_cann_context & ctx, ggml_tensor * dst) {
    स्वतः silu_fn = [](ggml_backend_cann_context & ctx, aclTensor * acl_src, aclTensor * acl_dst) {
        GGML_CANN_CALL_ACLNN_OP(ctx, Silu, acl_src, acl_dst);
    };

    अचर int32_t swapped = ggml_get_op_params_i32(dst, 1);
    यदि (dst->src[1] != nullptr || swapped != 0) {
        ggml_cann_op_unary_gated(silu_fn, ctx, dst);
        फर्काऊ;
    }

    // aclnnSwiGlu requires the split dim (src->ne[0]) to be even; fall back otherwise.
    यदि (dst->src[0]->ne[0] % 2 != 0) {
        ggml_cann_op_unary_gated(silu_fn, ctx, dst);
        फर्काऊ;
    }

    ggml_tensor * src0 = dst->src[0];
    माप_प्रकार elem_size = ggml_element_size(src0);

    // src0 GGML: [2*ne0, ne1, ne2, ne3] → 3D view [2*ne0, ne1, ne2*ne3]
    // CANN reversed: [ne2*ne3, ne1, 2*ne0], split along CANN dim 2 (last).
    int64_t ne0_x2   = src0->ne[0];
    int64_t ne1      = src0->ne[1];
    int64_t ne23     = src0->ne[2] * src0->ne[3];
    int64_t src3d_ne[] = { ne0_x2, ne1, ne23 };
    माप_प्रकार  src3d_nb[] = { (माप_प्रकार)src0->nb[0], (माप_प्रकार)src0->nb[1], (माप_प्रकार)src0->nb[2] };
    acl_tensor_ptr acl_src = ggml_cann_create_tensor(src0->data, ggml_cann_type_mapping(src0->type),
                                                     elem_size, src3d_ne, src3d_nb, 3);

    // dst GGML: [ne0, ne1, ne2, ne3] → 3D view [ne0, ne1, ne2*ne3]
    int64_t ne0      = dst->ne[0];
    int64_t dst3d_ne[] = { ne0, ne1, ne23 };
    माप_प्रकार  dst3d_nb[] = { (माप_प्रकार)dst->nb[0], (माप_प्रकार)dst->nb[1], (माप_प्रकार)dst->nb[2] };
    acl_tensor_ptr acl_dst = ggml_cann_create_tensor(dst->data, ggml_cann_type_mapping(dst->type),
                                                     elem_size, dst3d_ne, dst3d_nb, 3);

    // CANN tensor [ne23, ne1, 2*ne0]: split along CANN dim 2 (last) = 2*ne0.
    GGML_CANN_CALL_ACLNN_OP(ctx, SwiGlu, acl_src.get(), (int64_t)2, acl_dst.get());
}

शून्य ggml_cann_swiglu_clamp(ggml_backend_cann_context & ctx, ggml_tensor * dst) {
    ggml_tensor * src0 = dst->src[0];
    ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(ggml_is_contiguous_1(src0));
    GGML_ASSERT(ggml_is_contiguous_1(dst));

    अचर int32_t  swapped = ggml_get_op_params_i32(dst, 1);
    acl_tensor_ptr acl_gate;
    acl_tensor_ptr acl_up;
    यदि (src1) {
        GGML_ASSERT(ggml_is_contiguous_1(src1));
        GGML_ASSERT(src0->type == src1->type);
        acl_gate = ggml_cann_create_tensor(src0);
        acl_up   = ggml_cann_create_tensor(src1);
    } अन्यथा {
        int64_t ne[] = { src0->ne[0] / 2, src0->ne[1], src0->ne[2], src0->ne[3] };
        माप_प्रकार  nb[] = { src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3] };
        acl_gate     = ggml_cann_create_tensor(src0, ne, nb, GGML_MAX_DIMS, ACL_FORMAT_ND, 0);
        acl_up = ggml_cann_create_tensor(src0, ne, nb, GGML_MAX_DIMS, ACL_FORMAT_ND, ne[0] * ggml_element_size(src0));
        यदि (swapped) {
            std::swap(acl_gate, acl_up);
        }
    }

    ggml_cann_pool_alloc temp_alloc(ctx.pool(), ggml_nbytes(dst));
    acl_tensor_ptr       acl_temp = ggml_cann_create_tensor(temp_alloc.get(), ggml_cann_type_mapping(dst->type),
                                                            ggml_element_size(dst), dst->ne, dst->nb, GGML_MAX_DIMS);
    acl_tensor_ptr       acl_dst  = ggml_cann_create_tensor(dst);

    अचर भिन्न    limit        = ggml_get_op_params_f32(dst, 3);
    भिन्न          min_gate     = -अनन्त;
    भिन्न          min_up       = -limit;
    भिन्न          max_मान    = limit;
    acl_scalar_ptr acl_min_gate = ggml_cann_create_scalar(&min_gate, ACL_FLOAT);
    acl_scalar_ptr acl_min_up   = ggml_cann_create_scalar(&min_up, ACL_FLOAT);
    acl_scalar_ptr acl_limit    = ggml_cann_create_scalar(&max_मान, ACL_FLOAT);

    GGML_CANN_CALL_ACLNN_OP(ctx, Clamp, acl_gate.get(), acl_min_gate.get(), acl_limit.get(), acl_temp.get());
    GGML_CANN_CALL_ACLNN_OP(ctx, Silu, acl_temp.get(), acl_dst.get());
    GGML_CANN_CALL_ACLNN_OP(ctx, Clamp, acl_up.get(), acl_min_up.get(), acl_limit.get(), acl_temp.get());
    GGML_CANN_CALL_ACLNN_OP(ctx, InplaceMul, acl_dst.get(), acl_temp.get());
}

// Fused GeGLU using aclnnGeGluV3: splits input along ne[0] (CANN last dim),
// activates the LEFT half with GELU, multiplies by right half.
// approximate: 0=tanh, 1=none(erf). activateLeft=true matches GGML convention.
// outGelu is a required-but-discard output buffer.
//
// Falls back to the generic two-kernel path when src[1] != nullptr (two
// independent halves) or swapped != 0 (reversed activation order), as
// aclnnGeGluV3 only handles the single interleaved tensor in standard order.
शून्य ggml_cann_geglu(ggml_backend_cann_context & ctx, ggml_tensor * dst, int64_t approximate) {
    स्वतः gelu_fn = [](ggml_backend_cann_context & ctx, aclTensor * acl_src, aclTensor * acl_dst) {
        GGML_CANN_CALL_ACLNN_OP(ctx, Gelu, acl_src, acl_dst);
    };

    अचर int32_t swapped = ggml_get_op_params_i32(dst, 1);
    यदि (dst->src[1] != nullptr || swapped != 0) {
        ggml_cann_op_unary_gated(gelu_fn, ctx, dst);
        फर्काऊ;
    }

    // aclnnGeGluV3 requires the split dim (src->ne[0]) to be even; fall back otherwise.
    यदि (dst->src[0]->ne[0] % 2 != 0) {
        ggml_cann_op_unary_gated(gelu_fn, ctx, dst);
        फर्काऊ;
    }

    ggml_tensor * src0 = dst->src[0];
    acl_tensor_ptr acl_src = ggml_cann_create_tensor(src0);
    acl_tensor_ptr acl_dst = ggml_cann_create_tensor(dst);

    // Allocate a temporary buffer for the required outGelu output (same shape as dst).
    // Build contiguous strides since the pool allocation is a fresh buffer.
    माप_प्रकार  elem_size    = ggml_element_size(dst);
    int64_t ne[GGML_MAX_DIMS] = { dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3] };
    माप_प्रकार  nb[GGML_MAX_DIMS];
    nb[0] = elem_size;
    लागि (पूर्णाङ्क i = 1; i < GGML_MAX_DIMS; i++) {
        nb[i] = nb[i - 1] * ne[i - 1];
    }
    माप_प्रकार gelu_out_size = nb[GGML_MAX_DIMS - 1] * ne[GGML_MAX_DIMS - 1];
    ggml_cann_pool_alloc gelu_out_alloc(ctx.pool(), gelu_out_size);

    acl_tensor_ptr acl_gelu_out = ggml_cann_create_tensor(
        gelu_out_alloc.get(), ggml_cann_type_mapping(dst->type), elem_size, ne, nb, GGML_MAX_DIMS);
    // V3 adds activateLeft param; true → Gelu(left)*right, matching GGML convention.
    // GGML dim 0 → CANN last dim (index GGML_MAX_DIMS-1 = 3 for 4D tensor).
    GGML_CANN_CALL_ACLNN_OP(ctx, GeGluV3, acl_src.get(), (int64_t)(GGML_MAX_DIMS - 1), approximate, सत्य,
                             acl_dst.get(), acl_gelu_out.get());
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
स्थिर शून्य aclnn_repeat(ggml_backend_cann_context & ctx,
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
स्थिर शून्य aclnn_cast(ggml_backend_cann_context & ctx,
                       aclTensor *                 acl_src,
                       aclTensor *                 acl_dst,
                       aclDataType                 cast_data_type) {
    GGML_CANN_CALL_ACLNN_OP(ctx, Cast, acl_src, cast_data_type, acl_dst);
}

शून्य ggml_cann_repeat(ggml_backend_cann_context & ctx, ggml_tensor * dst) {
    ggml_tensor * src = dst->src[0];
    GGML_ASSERT(ggml_can_repeat(src, dst));

    acl_tensor_ptr acl_src = ggml_cann_create_tensor(src);
    acl_tensor_ptr acl_dst = ggml_cann_create_tensor(dst);

    int64_t repeatsArray[] = { dst->ne[3] / src->ne[3], dst->ne[2] / src->ne[2], dst->ne[1] / src->ne[1],
                               dst->ne[0] / src->ne[0] };

    aclnn_repeat(ctx, acl_src.get(), acl_dst.get(), repeatsArray);
}

शून्य aclnn_add(ggml_backend_cann_context & ctx, aclTensor * acl_src0, aclTensor * acl_src1, aclTensor * acl_dst) {
    भिन्न          alpha_मान = 1.0f;
    acl_scalar_ptr alpha      = ggml_cann_create_scalar(&alpha_मान, aclDataType::ACL_FLOAT);
    यदि (acl_dst != nullptr) {
        GGML_CANN_CALL_ACLNN_OP(ctx, Add, acl_src0, acl_src1, alpha.get(), acl_dst);
    } अन्यथा {
        GGML_CANN_CALL_ACLNN_OP(ctx, InplaceAdd, acl_src0, acl_src1, alpha.get());
    }
}

शून्य aclnn_sub(ggml_backend_cann_context & ctx, aclTensor * acl_src0, aclTensor * acl_src1, aclTensor * acl_dst) {
    भिन्न          alpha_मान = 1.0f;
    acl_scalar_ptr alpha      = ggml_cann_create_scalar(&alpha_मान, aclDataType::ACL_FLOAT);
    यदि (acl_dst != nullptr) {
        GGML_CANN_CALL_ACLNN_OP(ctx, Sub, acl_src0, acl_src1, alpha.get(), acl_dst);
    } अन्यथा {
        GGML_CANN_CALL_ACLNN_OP(ctx, InplaceSub, acl_src0, acl_src1, alpha.get());
    }
}

शून्य aclnn_mul(ggml_backend_cann_context & ctx, aclTensor * acl_src, aclTensor * acl_other, aclTensor * acl_dst) {
    यदि (acl_dst != nullptr) {
        GGML_CANN_CALL_ACLNN_OP(ctx, Mul, acl_src, acl_other, acl_dst);
    } अन्यथा {
        GGML_CANN_CALL_ACLNN_OP(ctx, InplaceMul, acl_src, acl_other);
    }
}

शून्य aclnn_div(ggml_backend_cann_context & ctx, aclTensor * acl_src, aclTensor * acl_other, aclTensor * acl_dst) {
    यदि (acl_dst != nullptr) {
        GGML_CANN_CALL_ACLNN_OP(ctx, Div, acl_src, acl_other, acl_dst);
    } अन्यथा {
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
स्थिर शून्य aclnn_muls(ggml_backend_cann_context & ctx,
                       aclTensor *                 acl_src,
                       भिन्न                       scale,
                       aclTensor *                 acl_dst,
                       तार्किक                        inplace) {
    acl_scalar_ptr acl_scale = ggml_cann_create_scalar(&scale, aclDataType::ACL_FLOAT);
    यदि (inplace) {
        GGML_CANN_CALL_ACLNN_OP(ctx, InplaceMuls, acl_src, acl_scale.get());
    } अन्यथा {
        GGML_CANN_CALL_ACLNN_OP(ctx, Muls, acl_src, acl_scale.get(), acl_dst);
    }
}

शून्य ggml_cann_leaky_relu(ggml_backend_cann_context & ctx, ggml_tensor * dst) {
    ggml_tensor * src = dst->src[0];

    acl_tensor_ptr acl_src = ggml_cann_create_tensor(src);
    acl_tensor_ptr acl_dst = ggml_cann_create_tensor(dst);

    भिन्न negative_slope;
    स_नकल(&negative_slope, dst->op_params, परिमाण(भिन्न));
    acl_scalar_ptr acl_negative_slope = ggml_cann_create_scalar(&negative_slope, aclDataType::ACL_FLOAT);

    GGML_CANN_CALL_ACLNN_OP(ctx, LeakyRelu, acl_src.get(), acl_negative_slope.get(), acl_dst.get());
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
स्थिर शून्य aclnn_concat(ggml_backend_cann_context & ctx,
                         aclTensorList *             tensorList,
                         aclTensor *                 acl_dst,
                         int64_t                     concat_dim) {
    GGML_CANN_CALL_ACLNN_OP(ctx, Cat, tensorList, concat_dim, acl_dst);
}

शून्य ggml_cann_concat(ggml_backend_cann_context & ctx, ggml_tensor * dst) {
    ggml_tensor *  src0     = dst->src[0];
    ggml_tensor *  src1     = dst->src[1];
    acl_tensor_ptr acl_src0 = ggml_cann_create_tensor(src0);
    acl_tensor_ptr acl_src1 = ggml_cann_create_tensor(src1);
    acl_tensor_ptr acl_dst  = ggml_cann_create_tensor(dst);

    अचर int32_t dim = ggml_get_op_params_i32(dst, 0);

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
स्थिर शून्य aclnn_arange(ggml_backend_cann_context & ctx,
                         aclTensor *                 acl_dst,
                         भिन्न                       start,
                         भिन्न                       stop,
                         भिन्न                       step,
                         int64_t                     n_elements) {
    int64_t steps = (int64_t) std::उच्चमान((stop - start) / step);
    GGML_ASSERT(n_elements == steps);

    acl_scalar_ptr acl_start = ggml_cann_create_scalar(&start, aclDataType::ACL_FLOAT);
    acl_scalar_ptr acl_end   = ggml_cann_create_scalar(&stop, aclDataType::ACL_FLOAT);
    acl_scalar_ptr acl_step  = ggml_cann_create_scalar(&step, aclDataType::ACL_FLOAT);

    GGML_CANN_CALL_ACLNN_OP(ctx, Arange, acl_start.get(), acl_end.get(), acl_step.get(), acl_dst);
}

शून्य ggml_cann_arange(ggml_backend_cann_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    acl_tensor_ptr acl_dst = ggml_cann_create_tensor(dst);

    int64_t n_elements = ggml_nelements(dst);
    भिन्न   start;
    भिन्न   stop;
    भिन्न   step;
    स_नकल(&start, (भिन्न *) dst->op_params + 0, परिमाण(भिन्न));
    स_नकल(&stop, (भिन्न *) dst->op_params + 1, परिमाण(भिन्न));
    स_नकल(&step, (भिन्न *) dst->op_params + 2, परिमाण(भिन्न));

    aclnn_arange(ctx, acl_dst.get(), start, stop, step, n_elements);
}

शून्य ggml_cann_clamp(ggml_backend_cann_context & ctx, ggml_tensor * dst) {
    ggml_tensor * src = dst->src[0];

    भिन्न min;
    भिन्न max;
    स_नकल(&min, dst->op_params, परिमाण(भिन्न));
    स_नकल(&max, (भिन्न *) dst->op_params + 1, परिमाण(भिन्न));

    acl_tensor_ptr acl_src = ggml_cann_create_tensor(src);
    acl_tensor_ptr acl_dst = ggml_cann_create_tensor(dst);

    acl_scalar_ptr acl_min = ggml_cann_create_scalar(&min, aclDataType::ACL_FLOAT);
    acl_scalar_ptr acl_max = ggml_cann_create_scalar(&max, aclDataType::ACL_FLOAT);

    GGML_CANN_CALL_ACLNN_OP(ctx, Clamp, acl_src.get(), acl_min.get(), acl_max.get(), acl_dst.get());
}

शून्य ggml_cann_scale(ggml_backend_cann_context & ctx, ggml_tensor * dst) {
    ggml_tensor * src = dst->src[0];

    // scale factor
    भिन्न v;
    स_नकल(&v, dst->op_params, परिमाण(भिन्न));

    acl_scalar_ptr scale   = ggml_cann_create_scalar(&v, aclDataType::ACL_FLOAT);
    acl_tensor_ptr acl_src = ggml_cann_create_tensor(src);
    acl_tensor_ptr acl_dst = ggml_cann_create_tensor(dst);

    GGML_CANN_CALL_ACLNN_OP(ctx, Muls, acl_src.get(), scale.get(), acl_dst.get());
}

शून्य ggml_cann_argsort(ggml_backend_cann_context & ctx, ggml_tensor * dst) {
    ggml_tensor *        src   = dst->src[0];
    गणना ggml_sort_order order = (गणना ggml_sort_order) dst->op_params[0];

    acl_tensor_ptr       acl_src = ggml_cann_create_tensor(src);
    acl_tensor_ptr       acl_dst = ggml_cann_create_tensor(dst);
    ggml_cann_pool_alloc temp_buffer_allocator(ctx.pool(), ggml_nelements(dst) * परिमाण(int64_t));
    शून्य *               buffer = temp_buffer_allocator.get();
    acl_tensor_ptr       tmp_tensor =
        ggml_cann_create_tensor(buffer, ACL_INT64, ggml_type_size(dst->type), dst->ne, dst->nb, GGML_MAX_DIMS);
    GGML_CANN_CALL_ACLNN_OP(ctx, Argsort, acl_src.get(), -1, (order == GGML_SORT_ORDER_DESC ? सत्य : असत्य),
                            tmp_tensor.get());
    GGML_CANN_CALL_ACLNN_OP(ctx, Cast, tmp_tensor.get(), ggml_cann_type_mapping(dst->type), acl_dst.get());
}

शून्य ggml_cann_norm(ggml_backend_cann_context & ctx, ggml_tensor * dst) {
    ggml_tensor * src = dst->src[0];

    acl_tensor_ptr acl_src = ggml_cann_create_tensor(src);
    acl_tensor_ptr acl_dst = ggml_cann_create_tensor(dst);

    भिन्न eps;
    स_नकल(&eps, dst->op_params, परिमाण(भिन्न));

    std::vector<int64_t> normData = { dst->ne[0] };
    acl_int_array_ptr    norm     = ggml_cann_create_int_array(normData.data(), normData.size());
    GGML_CANN_CALL_ACLNN_OP(ctx, LayerNorm, acl_src.get(), norm.get(), nullptr, nullptr, eps, acl_dst.get(), nullptr,
                            nullptr);
}

शून्य ggml_cann_l2_norm(ggml_backend_cann_context & ctx, ggml_tensor * dst) {
    ggml_tensor * src = dst->src[0];

    भिन्न eps;
    स_नकल(&eps, dst->op_params, परिमाण(भिन्न));

    acl_tensor_ptr acl_src = ggml_cann_create_tensor(src);
    acl_tensor_ptr acl_dst = ggml_cann_create_tensor(dst);

    माप_प्रकार               type_size = ggml_type_size(src->type);
    int64_t              n_bytes   = src->ne[3] * src->ne[2] * src->ne[1] * type_size;
    ggml_cann_pool_alloc temp_buffer_allocator(ctx.pool(), n_bytes);
    शून्य *               buffer = temp_buffer_allocator.get();

    int64_t norm_ne[] = { 1, src->ne[1], src->ne[2], src->ne[3] };
    माप_प्रकार  norm_nb[GGML_MAX_DIMS];
    norm_nb[0] = परिमाण(भिन्न);
    लागि (पूर्णाङ्क i = 1; i < GGML_MAX_DIMS; ++i) {
        norm_nb[i] = norm_nb[i - 1] * norm_ne[i - 1];
    }
    acl_tensor_ptr acl_norm = ggml_cann_create_tensor(buffer, ACL_FLOAT, परिमाण(भिन्न), norm_ne, norm_nb, GGML_MAX_DIMS);

    std::vector<int64_t> norm_dims  = { 3 };
    acl_int_array_ptr    dims_array = ggml_cann_create_int_array(norm_dims.data(), norm_dims.size());

    भिन्न          p_मान  = 2.0f;
    acl_scalar_ptr p_scalar = ggml_cann_create_scalar(&p_मान, aclDataType::ACL_FLOAT);
    GGML_CANN_CALL_ACLNN_OP(ctx, Norm, acl_src.get(), p_scalar.get(), dims_array.get(), सत्य, acl_norm.get());

    ggml_cann_pool_alloc clamp_buffer_allocator(ctx.pool());
    acl_tensor_ptr       acl_clamped;

    यदि (eps > 0.0f) {
        शून्य *         clamp_buf  = clamp_buffer_allocator.alloc(n_bytes);
        acl_clamped               = ggml_cann_create_tensor(clamp_buf, ACL_FLOAT, परिमाण(भिन्न), norm_ne, norm_nb, GGML_MAX_DIMS);
        acl_scalar_ptr eps_scalar = ggml_cann_create_scalar(&eps, aclDataType::ACL_FLOAT);
        GGML_CANN_CALL_ACLNN_OP(ctx, ClampMin, acl_norm.get(), eps_scalar.get(), acl_clamped.get());
    }

    aclTensor * acl_div_आगत = acl_clamped ? acl_clamped.get() : acl_norm.get();
    GGML_CANN_CALL_ACLNN_OP(ctx, Div, acl_src.get(), acl_div_आगत, acl_dst.get());
}

शून्य ggml_cann_cross_entropy_loss(ggml_backend_cann_context & ctx, ggml_tensor * dst) {
    ggml_tensor * src0 = dst->src[0];
    ggml_tensor * src1 = dst->src[1];

    अचर int64_t nc = src0->ne[0];
    अचर int64_t nr = ggml_nrows(src0);

    int64_t logits_ne[] = { nc, nr };
    माप_प्रकार  logits_nb[2];
    logits_nb[0]              = ggml_type_size(src0->type);
    logits_nb[1]              = logits_nb[0] * logits_ne[0];
    acl_tensor_ptr acl_logits = ggml_cann_create_tensor(src0->data, ACL_FLOAT, परिमाण(भिन्न), logits_ne, logits_nb, 2);

    int64_t labels_ne[] = { nc, nr };
    माप_प्रकार  labels_nb[2];
    labels_nb[0]              = ggml_type_size(src1->type);
    labels_nb[1]              = labels_nb[0] * labels_ne[0];
    acl_tensor_ptr acl_labels = ggml_cann_create_tensor(src1->data, ACL_FLOAT, परिमाण(भिन्न), labels_ne, labels_nb, 2);

    माप_प्रकार               loss_per_sample_type_size = परिमाण(भिन्न);
    int64_t              loss_per_sample_n_bytes   = nr * loss_per_sample_type_size;
    ggml_cann_pool_alloc loss_per_sample_allocator(ctx.pool(), loss_per_sample_n_bytes);
    शून्य *               loss_per_sample_buffer = loss_per_sample_allocator.get();

    int64_t loss_per_sample_ne[] = { nr };
    माप_प्रकार  loss_per_sample_nb[1];
    loss_per_sample_nb[0] = loss_per_sample_type_size;
    acl_tensor_ptr acl_loss_per_sample = ggml_cann_create_tensor(
        loss_per_sample_buffer, ACL_FLOAT, loss_per_sample_type_size, loss_per_sample_ne, loss_per_sample_nb, 1);

    माप_प्रकार               backprop_n_bytes = nr * nc * परिमाण(भिन्न);
    ggml_cann_pool_alloc backprop_allocator(ctx.pool(), backprop_n_bytes);
    शून्य *               backprop_buffer = backprop_allocator.get();
    acl_tensor_ptr acl_backprop = ggml_cann_create_tensor(backprop_buffer, ACL_FLOAT, परिमाण(भिन्न), logits_ne, logits_nb, 2);

    GGML_CANN_CALL_ACLNN_OP(ctx, SoftmaxCrossEntropyWithLogits, acl_logits.get(), acl_labels.get(),
                            acl_loss_per_sample.get(), acl_backprop.get());

    माप_प्रकार               total_sum_type_size = परिमाण(भिन्न);
    int64_t              total_sum_n_bytes   = 1 * total_sum_type_size;
    ggml_cann_pool_alloc total_sum_allocator(ctx.pool(), total_sum_n_bytes);
    शून्य *               total_sum_buffer = total_sum_allocator.get();

    int64_t total_sum_ne[] = { 1 };
    माप_प्रकार  total_sum_nb[1];
    total_sum_nb[0] = total_sum_type_size;

    acl_tensor_ptr acl_total_sum =
        ggml_cann_create_tensor(total_sum_buffer, ACL_FLOAT, total_sum_type_size, total_sum_ne, total_sum_nb, 1);

    std::vector<int64_t> total_sum_dims    = { 0 };
    acl_int_array_ptr total_sum_dims_array = ggml_cann_create_int_array(total_sum_dims.data(), total_sum_dims.size());
    तार्किक              keep_dims            = असत्य;

    GGML_CANN_CALL_ACLNN_OP(ctx, ReduceSum, acl_loss_per_sample.get(), total_sum_dims_array.get(), keep_dims, ACL_FLOAT,
                            acl_total_sum.get());

    भिन्न          value        = 1.0f / static_cast<भिन्न>(nr);
    acl_scalar_ptr scale_factor = ggml_cann_create_scalar(&value, aclDataType::ACL_FLOAT);
    acl_tensor_ptr acl_dst =
        ggml_cann_create_tensor(dst->data, ACL_FLOAT, परिमाण(भिन्न), total_sum_ne, total_sum_nb, 1);

    GGML_CANN_CALL_ACLNN_OP(ctx, Muls, acl_total_sum.get(), scale_factor.get(), acl_dst.get());
}

शून्य ggml_cann_group_norm(ggml_backend_cann_context & ctx, ggml_tensor * dst) {
    ggml_tensor * src = dst->src[0];

    acl_tensor_ptr acl_src = ggml_cann_create_tensor(src);
    acl_tensor_ptr acl_dst = ggml_cann_create_tensor(dst);

    पूर्णाङ्क n_groups = dst->op_params[0];

    भिन्न eps;
    स_नकल(&eps, dst->op_params + 1, परिमाण(भिन्न));

    int64_t N   = src->ne[3];
    int64_t C   = src->ne[2];
    int64_t HxW = src->ne[1] * src->ne[0];

    माप_प्रकार  type_size = ggml_type_size(src->type);
    int64_t ne[]      = { n_groups, N };
    माप_प्रकार  nb[]      = { type_size, type_size * n_groups };
    माप_प्रकार  n_bytes   = N * n_groups;

    ggml_cann_pool_alloc temp_buffer_allocator(ctx.pool(), n_bytes * 2);
    शून्य *               buffer       = temp_buffer_allocator.get();
    acl_tensor_ptr       acl_mean_out = ggml_cann_create_tensor(buffer, ACL_FLOAT, type_size, ne, nb, ACL_FORMAT_ND);
    acl_tensor_ptr       acl_rstd_out =
        ggml_cann_create_tensor((वर्ण *) buffer + n_bytes, ACL_FLOAT, type_size, ne, nb, ACL_FORMAT_ND);

    GGML_CANN_CALL_ACLNN_OP(ctx, GroupNorm, acl_src.get(), nullptr, nullptr, N, C, HxW, n_groups, eps, acl_dst.get(),
                            acl_mean_out.get(), acl_rstd_out.get());
}

शून्य ggml_cann_set(ggml_backend_cann_context & ctx, ggml_tensor * dst) {
    ggml_tensor * src0 = dst->src[0];
    ggml_tensor * src1 = dst->src[1];

    माप_प्रकार nb1     = ((int32_t *) dst->op_params)[0];
    माप_प्रकार nb2     = ((int32_t *) dst->op_params)[1];
    माप_प्रकार nb3     = ((int32_t *) dst->op_params)[2];
    माप_प्रकार offset  = ((int32_t *) dst->op_params)[3];
    तार्किक   inplace = (तार्किक) ((int32_t *) dst->op_params)[4];

    माप_प्रकार param_nb[] = { ggml_element_size(src0), nb1, nb2, nb3 };

    // Create a view of dst at the target offset with src1's dimensions
    acl_tensor_ptr acl_dst  = ggml_cann_create_tensor(dst, src1->ne, param_nb, GGML_MAX_DIMS, ACL_FORMAT_ND, offset);
    acl_tensor_ptr acl_src1 = ggml_cann_create_tensor(src1);

    यदि (!inplace) {
        // First copy src0 to dst entirely
        माप_प्रकार cpy_size = ggml_nbytes(dst);
        ACL_CHECK(
            aclrtMemcpyAsync(dst->data, cpy_size, src0->data, cpy_size, ACL_MEMCPY_DEVICE_TO_DEVICE, ctx.stream()));
    }

    // Copy src1 into the target region of dst
    GGML_CANN_CALL_ACLNN_OP(ctx, InplaceCopy, acl_dst.get(), acl_src1.get());
}

शून्य ggml_cann_acc(ggml_backend_cann_context & ctx, ggml_tensor * dst) {
    ggml_tensor * src0 = dst->src[0];
    ggml_tensor * src1 = dst->src[1];

    माप_प्रकार nb1     = ((int32_t *) dst->op_params)[0];
    माप_प्रकार nb2     = ((int32_t *) dst->op_params)[1];
    माप_प्रकार nb3     = ((int32_t *) dst->op_params)[2];
    माप_प्रकार offset  = ((int32_t *) dst->op_params)[3];
    तार्किक   inplace = (तार्किक) ((int32_t *) dst->op_params)[4];

    माप_प्रकार param_nb[] = { ggml_element_size(src0), nb1, nb2, nb3 };

    acl_tensor_ptr acl_dst  = ggml_cann_create_tensor(dst, src1->ne, param_nb, GGML_MAX_DIMS, ACL_FORMAT_ND, offset);
    acl_tensor_ptr acl_src1 = ggml_cann_create_tensor(src1);

    acl_scalar_ptr alpha      = nullptr;
    भिन्न          alpha_मान = 1.0f;
    alpha                     = ggml_cann_create_scalar(&alpha_मान, aclDataType::ACL_FLOAT);

    यदि (!inplace) {
        माप_प्रकार cpy_size = ggml_nbytes(dst);
        ACL_CHECK(
            aclrtMemcpyAsync(dst->data, cpy_size, src0->data, cpy_size, ACL_MEMCPY_DEVICE_TO_DEVICE, ctx.stream()));
        acl_tensor_ptr acl_src0 =
            ggml_cann_create_tensor(src0, src1->ne, src0->nb, GGML_MAX_DIMS, ACL_FORMAT_ND, offset);

        GGML_CANN_CALL_ACLNN_OP(ctx, Add, acl_src0.get(), acl_src1.get(), alpha.get(), acl_dst.get());
    } अन्यथा {
        GGML_CANN_CALL_ACLNN_OP(ctx, InplaceAdd, acl_dst.get(), acl_src1.get(), alpha.get());
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
स्थिर शून्य aclnn_reduce_sum(ggml_backend_cann_context & ctx, ggml_tensor * dst, int64_t * dim, माप_प्रकार dim_size) {
    GGML_ASSERT(dst->ne[0] == 1);
    ggml_tensor *     src         = dst->src[0];
    acl_tensor_ptr    acl_src     = ggml_cann_create_tensor(src);
    acl_tensor_ptr    acl_dst     = ggml_cann_create_tensor(dst);
    acl_int_array_ptr reduce_dims = ggml_cann_create_int_array(dim, dim_size);

    GGML_CANN_CALL_ACLNN_OP(ctx, ReduceSum, acl_src.get(), reduce_dims.get(), सत्य, ggml_cann_type_mapping(dst->type),
                            acl_dst.get());
}

शून्य ggml_cann_sum_rows(ggml_backend_cann_context & ctx, ggml_tensor * dst) {
    int64_t reduce_dims[] = { 3 };
    aclnn_reduce_sum(ctx, dst, reduce_dims, 1);
}

शून्य ggml_cann_sum(ggml_backend_cann_context & ctx, ggml_tensor * dst) {
    int64_t reduce_dims[] = { 0, 1, 2, 3 };
    aclnn_reduce_sum(ctx, dst, reduce_dims, 4);
}

शून्य ggml_cann_cumsum(ggml_backend_cann_context & ctx, ggml_tensor * dst) {
    ggml_tensor * src = dst->src[0];
    acl_tensor_ptr acl_src = ggml_cann_create_tensor(src);
    acl_tensor_ptr acl_dst = ggml_cann_create_tensor(dst);
    // GGML cumsum operates along dim 0 (innermost / ne[0]).
    // ggml_cann_create_tensor reverses dimensions to [ne3,ne2,ne1,ne0],
    // so GGML dim 0 maps to CANN dim 3 (the last dim of the 4-D tensor).
    GGML_CANN_CALL_ACLNN_OP(ctx, Cumsum, acl_src.get(), (int64_t)3,
                            ggml_cann_type_mapping(dst->type), acl_dst.get());
}

शून्य ggml_cann_solve_tri(ggml_backend_cann_context & ctx, ggml_tensor * dst) {
    ggml_tensor * src0 = dst->src[0];  // A: [N, N, B2, B3] lower triangular
    ggml_tensor * src1 = dst->src[1];  // B: [K, N, B2, B3]

    acl_tensor_ptr acl_a = ggml_cann_create_tensor(src0);
    acl_tensor_ptr acl_b = ggml_cann_create_tensor(src1);
    acl_tensor_ptr acl_x = ggml_cann_create_tensor(dst);

    // mOut: triangular copy of A (required output), same shape as A.
    अचर माप_प्रकार a_bytes = ggml_nbytes(src0);
    ggml_cann_pool_alloc m_alloc(ctx.pool(), a_bytes);
    acl_tensor_ptr acl_m = ggml_cann_create_tensor(
        m_alloc.get(), ggml_cann_type_mapping(src0->type),
        ggml_type_size(src0->type), src0->ne, src0->nb, GGML_MAX_DIMS);

    // Solve AX = B: upper=false (lower tri), transpose=false, unitriangular=false.
    GGML_CANN_CALL_ACLNN_OP(ctx, TriangularSolve,
        acl_b.get(), acl_a.get(), असत्य, असत्य, असत्य,
        acl_x.get(), acl_m.get());
}

शून्य ggml_cann_diag(ggml_backend_cann_context & ctx, ggml_tensor * dst) {
    ggml_tensor * src = dst->src[0];

    GGML_ASSERT(src->ne[1] == 1);

    अचर int64_t N       = src->ne[0];
    अचर int64_t n_batch = src->ne[2] * src->ne[3];
    अचर माप_प्रकार  nb_f32  = परिमाण(भिन्न);

    // Fill dst with zeros.
    acl_tensor_ptr acl_dst = ggml_cann_create_tensor(dst);
    {
        भिन्न          zero = 0.0f;
        acl_scalar_ptr acl_zero = ggml_cann_create_scalar(&zero, ACL_FLOAT);
        GGML_CANN_CALL_ACLNN_OP(ctx, InplaceFillScalar, acl_dst.get(), acl_zero.get());
    }

    // Copy src vector onto the diagonal of dst via strided views.
    // src viewed as [N, n_batch], contiguous strides.
    int64_t ne_vec[2]      = { N, n_batch };
    माप_प्रकार  nb_src_vec[2]  = { nb_f32, N * nb_f32 };
    // dst diagonal view: stride (N+1)*4 steps along the diagonal.
    माप_प्रकार  nb_dst_diag[2] = { (N + 1) * nb_f32, N * N * nb_f32 };

    acl_tensor_ptr acl_src_vec  = ggml_cann_create_tensor(src->data, ACL_FLOAT, nb_f32, ne_vec, nb_src_vec, 2);
    acl_tensor_ptr acl_dst_diag = ggml_cann_create_tensor(dst->data, ACL_FLOAT, nb_f32, ne_vec, nb_dst_diag, 2);

    GGML_CANN_CALL_ACLNN_OP(ctx, InplaceCopy, acl_dst_diag.get(), acl_src_vec.get());
}

शून्य ggml_cann_fill(ggml_backend_cann_context & ctx, ggml_tensor * dst) {
    भिन्न c = ggml_get_op_params_f32(dst, 0);

    acl_tensor_ptr acl_dst = ggml_cann_create_tensor(dst);
    acl_scalar_ptr acl_c   = ggml_cann_create_scalar(&c, ACL_FLOAT);
    GGML_CANN_CALL_ACLNN_OP(ctx, InplaceFillScalar, acl_dst.get(), acl_c.get());
}

शून्य ggml_cann_tri(ggml_backend_cann_context & ctx, ggml_tensor * dst) {
    ggml_tensor * src = dst->src[0];

    अचर int64_t S       = src->ne[0];
    अचर int64_t n_batch = src->ne[2] * src->ne[3];
    अचर माप_प्रकार  nb_f32  = परिमाण(भिन्न);

    int64_t ne3d[3] = { S, S, n_batch };
    माप_प्रकार  nb3d[3] = { nb_f32, S * nb_f32, S * S * nb_f32 };

    अचर ggml_tri_type ttype = (ggml_tri_type) ggml_get_op_params_i32(dst, 0);

    acl_tensor_ptr acl_src = ggml_cann_create_tensor(src->data, ACL_FLOAT, nb_f32, ne3d, nb3d, 3);
    acl_tensor_ptr acl_dst = ggml_cann_create_tensor(dst->data, ACL_FLOAT, nb_f32, ne3d, nb3d, 3);

    विकल्प (ttype) {
        पक्ष GGML_TRI_TYPE_LOWER:
            // Tril(-1): preserve row > col (strict lower), zero upper + diagonal.
            GGML_CANN_CALL_ACLNN_OP(ctx, Tril, acl_src.get(), (int64_t)-1, acl_dst.get());
            रोक;
        पक्ष GGML_TRI_TYPE_UPPER_DIAG:
            // Triu(0): preserve row <= col (upper + diagonal), zero strict lower.
            GGML_CANN_CALL_ACLNN_OP(ctx, Triu, acl_src.get(), (int64_t)0, acl_dst.get());
            रोक;
        पक्ष GGML_TRI_TYPE_UPPER:
            // Triu(1): preserve row < col (strict upper), zero lower + diagonal.
            GGML_CANN_CALL_ACLNN_OP(ctx, Triu, acl_src.get(), (int64_t)1, acl_dst.get());
            रोक;
        पक्ष GGML_TRI_TYPE_LOWER_DIAG:
            // Tril(0): preserve row >= col (lower + diagonal), zero strict upper.
            GGML_CANN_CALL_ACLNN_OP(ctx, Tril, acl_src.get(), (int64_t)0, acl_dst.get());
            रोक;
        स्वभाव:
            GGML_ABORT("unsupported tri type");
    }
}

शून्य ggml_cann_upsample_nearest2d(ggml_backend_cann_context & ctx, ggml_tensor * dst) {
    ggml_tensor *  src     = dst->src[0];
    acl_tensor_ptr acl_src = ggml_cann_create_tensor(src, nullptr, nullptr, 0, ACL_FORMAT_NCHW);
    acl_tensor_ptr acl_dst = ggml_cann_create_tensor(dst, nullptr, nullptr, 0, ACL_FORMAT_NCHW);

    std::vector<int64_t> output_size{ dst->ne[1], dst->ne[0] };
    acl_int_array_ptr    output_size_array = ggml_cann_create_int_array(output_size.data(), 2);

    GGML_CANN_CALL_ACLNN_OP(ctx, UpsampleNearest2d, acl_src.get(), output_size_array.get(), acl_dst.get());
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
स्थिर शून्य aclnn_pad(ggml_backend_cann_context & ctx,
                      aclTensor *                 acl_src,
                      aclTensor *                 acl_dst,
                      int64_t *                   paddings,
                      भिन्न                       value = 0.0f) {
    acl_int_array_ptr acl_pad   = ggml_cann_create_int_array(paddings, GGML_MAX_DIMS * 2);
    acl_scalar_ptr    acl_मान = ggml_cann_create_scalar(&value, aclDataType::ACL_FLOAT);

    GGML_CANN_CALL_ACLNN_OP(ctx, ConstantPadNd, acl_src, acl_pad.get(), acl_मान.get(), acl_dst);
}

शून्य ggml_cann_pad(ggml_backend_cann_context & ctx, ggml_tensor * dst) {
    ggml_tensor *  src     = dst->src[0];
    acl_tensor_ptr acl_src = ggml_cann_create_tensor(src);
    acl_tensor_ptr acl_dst = ggml_cann_create_tensor(dst);

    // padding: value in the array means how much distance will be padding.
    // the position of elements in the array means which dirction to padding,
    // each position means: [dim0.front, dim0.behind, dim1.front, dim1.behind,
    //                       dim2.front, dim2.behind, dim3.front, dim3.behind]
    अचर int32_t lp0 = ggml_get_op_params_i32(dst, 0);
    अचर int32_t rp0 = ggml_get_op_params_i32(dst, 1);
    अचर int32_t lp1 = ggml_get_op_params_i32(dst, 2);
    अचर int32_t rp1 = ggml_get_op_params_i32(dst, 3);
    अचर int32_t lp2 = ggml_get_op_params_i32(dst, 4);
    अचर int32_t rp2 = ggml_get_op_params_i32(dst, 5);
    अचर int32_t lp3 = ggml_get_op_params_i32(dst, 6);
    अचर int32_t rp3 = ggml_get_op_params_i32(dst, 7);

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
स्थिर शून्य ggml_cann_avg_pool2d(ggml_backend_cann_context & ctx, ggml_tensor * dst) {
    ggml_tensor * src = dst->src[0];
    GGML_ASSERT(src->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    acl_tensor_ptr acl_src = ggml_cann_create_tensor(src, nullptr, nullptr, 0, ACL_FORMAT_NCHW);
    acl_tensor_ptr acl_dst = ggml_cann_create_tensor(dst, nullptr, nullptr, 0, ACL_FORMAT_NCHW);

    अचर int32_t * opts = (अचर int32_t *) dst->op_params;
    अचर पूर्णाङ्क       k0   = opts[1];
    अचर पूर्णाङ्क       k1   = opts[2];
    अचर पूर्णाङ्क       s0   = opts[3];
    अचर पूर्णाङ्क       s1   = opts[4];
    अचर पूर्णाङ्क       p0   = opts[5];
    अचर पूर्णाङ्क       p1   = opts[6];

    std::vector<int64_t> kernel_dims      = { k1, k0 };
    std::vector<int64_t> stride_dims      = { s1, s0 };
    std::vector<int64_t> padding_avg_dims = { p1, p0 };  // (padH, padW)

    acl_int_array_ptr kernel_size  = ggml_cann_create_int_array(kernel_dims.data(), 2);
    acl_int_array_ptr strides      = ggml_cann_create_int_array(stride_dims.data(), 2);
    acl_int_array_ptr paddings_avg = ggml_cann_create_int_array(padding_avg_dims.data(), 2);

    तार्किक    ceil_mode         = असत्य;
    तार्किक    गन्ती_include_pad = सत्य;
    int64_t भाजक_override  = 0;
    int8_t  cube_math_type    = 0;
#ifdef ASCEND_310P
    cube_math_type = 1;
#endif

    GGML_CANN_CALL_ACLNN_OP(ctx, AvgPool2d, acl_src.get(), kernel_size.get(), strides.get(), paddings_avg.get(),
                            ceil_mode, गन्ती_include_pad, भाजक_override, cube_math_type, acl_dst.get());
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
स्थिर शून्य ggml_cann_max_pool2d(ggml_backend_cann_context & ctx, ggml_tensor * dst) {
    ggml_tensor * src = dst->src[0];
    GGML_ASSERT(src->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    acl_tensor_ptr acl_src = ggml_cann_create_tensor(src, nullptr, nullptr, 0, ACL_FORMAT_NCHW);
    acl_tensor_ptr acl_dst = ggml_cann_create_tensor(dst, nullptr, nullptr, 0, ACL_FORMAT_NCHW);

    अचर int32_t * opts = (अचर int32_t *) dst->op_params;
    अचर पूर्णाङ्क       k0   = opts[1];
    अचर पूर्णाङ्क       k1   = opts[2];
    अचर पूर्णाङ्क       s0   = opts[3];
    अचर पूर्णाङ्क       s1   = opts[4];
    अचर पूर्णाङ्क       p0   = opts[5];
    अचर पूर्णाङ्क       p1   = opts[6];

    int64_t temp_ne[] = { src->ne[0] + p0 * 2, src->ne[1] + p1 * 2, src->ne[2], src->ne[3] };
    माप_प्रकार  temp_nb[GGML_MAX_DIMS];

    temp_nb[0] = ggml_element_size(src);
    लागि (पूर्णाङ्क i = 1; i < GGML_MAX_DIMS; i++) {
        temp_nb[i] = temp_nb[i - 1] * temp_ne[i - 1];
    }

    ggml_cann_pool_alloc temp_buffer_allocator(ctx.pool(), ggml_nbytes(src) + p0 * 2 + p1 * 2 * src->nb[1]);
    शून्य *               buffer = temp_buffer_allocator.get();
    acl_tensor_ptr tmp_tensor   = ggml_cann_create_tensor(buffer, ACL_FLOAT, ggml_element_size(src), temp_ne, temp_nb,
                                                          GGML_MAX_DIMS, ACL_FORMAT_NCHW);

    // pad: see padding in ggml_cann_pad()
    int64_t paddings[] = { p0, p0, p1, p1, 0, 0, 0, 0 };
    भिन्न   value      = -भग्न_उच्च;
    aclnn_pad(ctx, acl_src.get(), tmp_tensor.get(), paddings, value);

    // max_pool
    std::vector<int64_t> kernel_dims      = { k1, k0 };
    std::vector<int64_t> stride_dims      = { s1, s0 };
    // padding_max_dims: [dim0_start, dim0_end, dim1_start, dim1_end]
    std::vector<int64_t> padding_max_dims = { 0, 0, 0, 0 };
    std::vector<int64_t> dilation_size    = { 1, 1 };
    acl_int_array_ptr    kernel_size      = ggml_cann_create_int_array(kernel_dims.data(), 2);
    acl_int_array_ptr    strides          = ggml_cann_create_int_array(stride_dims.data(), 2);
    acl_int_array_ptr    paddings_max     = ggml_cann_create_int_array(padding_max_dims.data(), 4);
    acl_int_array_ptr    dilations        = ggml_cann_create_int_array(dilation_size.data(), 2);

    तार्किक    ceil_mode = असत्य;
    int64_t auto_pads = 0;
    GGML_CANN_CALL_ACLNN_OP(ctx, MaxPool, tmp_tensor.get(), kernel_size.get(), strides.get(), auto_pads,
                            paddings_max.get(), dilations.get(), ceil_mode, acl_dst.get());
}

शून्य ggml_cann_pool2d(ggml_backend_cann_context & ctx, ggml_tensor * dst) {
    अचर int32_t *   opts = (अचर int32_t *) dst->op_params;
    गणना ggml_op_pool op   = static_cast<ggml_op_pool>(opts[0]);
    विकल्प (op) {
        पक्ष GGML_OP_POOL_AVG:
            ggml_cann_avg_pool2d(ctx, dst);
            रोक;
        पक्ष GGML_OP_POOL_MAX:
            ggml_cann_max_pool2d(ctx, dst);
            रोक;
        पक्ष GGML_OP_POOL_गन्ती:
            GGML_ABORT("fatal error");
            रोक;
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
स्थिर शून्य cann_copy(ggml_backend_cann_context & ctx, aclTensor * acl_src, aclTensor * acl_dst) {
    GGML_CANN_CALL_ACLNN_OP(ctx, InplaceCopy, acl_dst, acl_src);
}

शून्य ggml_cann_dup(ggml_backend_cann_context & ctx, ggml_tensor * dst) {
    ggml_tensor * src0 = dst->src[0];

    यदि (ggml_are_same_shape(src0, dst)) {
        acl_tensor_ptr acl_src = ggml_cann_create_tensor(src0);
        acl_tensor_ptr acl_dst = ggml_cann_create_tensor(dst);
        यदि (dst->type == src0->type) {
            cann_copy(ctx, acl_src.get(), acl_dst.get());
        } अन्यथा {
            aclnn_cast(ctx, acl_src.get(), acl_dst.get(), ggml_cann_type_mapping(dst->type));
        }
    } अन्यथा {
        शून्य *               src_trans_buffer = src0->data;
        ggml_cann_pool_alloc src_buffer_allocator;
        यदि (!ggml_is_contiguous(src0)) {
            acl_tensor_ptr acl_src = ggml_cann_create_tensor(src0);
            src_buffer_allocator.alloc(ctx.pool(), ggml_nelements(src0) * ggml_type_size(src0->type));
            src_trans_buffer = src_buffer_allocator.get();
            माप_प्रकार src_trans_nb[GGML_MAX_DIMS];
            src_trans_nb[0] = ggml_type_size(src0->type);
            लागि (पूर्णाङ्क i = 1; i < GGML_MAX_DIMS; i++) {
                src_trans_nb[i] = src_trans_nb[i - 1] * src0->ne[i - 1];
            }
            acl_tensor_ptr src_trans_tensor =
                ggml_cann_create_tensor(src_trans_buffer, ggml_cann_type_mapping(src0->type),
                                        ggml_type_size(src0->type), src0->ne, src_trans_nb, GGML_MAX_DIMS);
            cann_copy(ctx, acl_src.get(), src_trans_tensor.get());
        }

        माप_प्रकार src_reshape_nb[GGML_MAX_DIMS];
        src_reshape_nb[0] = ggml_type_size(src0->type);
        लागि (पूर्णाङ्क i = 1; i < GGML_MAX_DIMS; i++) {
            src_reshape_nb[i] = src_reshape_nb[i - 1] * dst->ne[i - 1];
        }

        acl_tensor_ptr trans_acl_src =
            ggml_cann_create_tensor(src_trans_buffer, ggml_cann_type_mapping(src0->type), ggml_type_size(src0->type),
                                    dst->ne, src_reshape_nb, GGML_MAX_DIMS, ACL_FORMAT_ND);
        acl_tensor_ptr acl_dst = ggml_cann_create_tensor(dst);

        यदि (dst->type == src0->type) {
            cann_copy(ctx, trans_acl_src.get(), acl_dst.get());
        } अन्यथा {
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
स्थिर acl_tensor_ptr aclnn_zero(ggml_backend_cann_context & ctx,
                                 शून्य *                      buffer,
                                 माप_प्रकार                      n_bytes,
                                 int64_t *                   ne,
                                 int64_t                     dims,
                                 aclDataType                 type,
                                 माप_प्रकार                      type_size) {
    माप_प्रकार nb[GGML_MAX_DIMS];
    nb[0] = type_size;
    लागि (पूर्णाङ्क i = 1; i < dims; i++) {
        nb[i] = nb[i - 1] * ne[i - 1];
    }

    acl_tensor_ptr zero = ggml_cann_create_tensor(buffer, type, type_size, ne, nb, dims);
    GGML_CANN_CALL_ACLNN_OP(ctx, InplaceZero, zero.get());
    फर्काऊ zero;
    GGML_UNUSED(n_bytes);
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
स्थिर acl_tensor_ptr aclnn_मानहरू(ggml_backend_cann_context & ctx,
                                   शून्य *                      buffer,
                                   माप_प्रकार                      n_bytes,
                                   int64_t *                   ne,
                                   int64_t                     dims,
                                   aclDataType                 type,
                                   माप_प्रकार                      type_size,
                                   भिन्न                       value = 1.0f) {
    acl_tensor_ptr acl_tensor = aclnn_zero(ctx, buffer, n_bytes, ne, dims, type, type_size);
    भिन्न          alpha_host = 1.0f;
    acl_scalar_ptr alpha      = ggml_cann_create_scalar(&alpha_host, aclDataType::ACL_FLOAT);
    acl_scalar_ptr other      = ggml_cann_create_scalar(&value, aclDataType::ACL_FLOAT);
    GGML_CANN_CALL_ACLNN_OP(ctx, InplaceAdds, acl_tensor.get(), other.get(), alpha.get());
    फर्काऊ acl_tensor;
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
स्थिर शून्य aclnn_fill_scalar(ggml_backend_cann_context & ctx, भिन्न scalar, aclTensor * acl_dst) {
    acl_scalar_ptr acl_scalar = ggml_cann_create_scalar(&scalar, aclDataType::ACL_FLOAT);
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
स्थिर acl_tensor_ptr get_cache_acl_tensor(ggml_backend_cann_context & ctx,
                                           शून्य **                     buffer,
                                           int64_t &                   cache_element,
                                           int64_t *                   ne,
                                           माप_प्रकार *                    nb,
                                           ggml_type                   dtype,
                                           int64_t                     dims,
                                           भिन्न                       value) {
    // Calculate total number of elements
    int64_t n_element = 1;
    लागि (पूर्णाङ्क i = 0; i < dims; i++) {
        n_element *= ne[i];
    }
    माप_प्रकार size = n_element * ggml_type_size(dtype);

    // Allocate or expand cache if needed
    यदि (cache_element < n_element) {
        यदि (*buffer != nullptr) {
            aclrtFree(*buffer);
            *buffer = nullptr;
        }

        ACL_CHECK(aclrtMalloc(buffer, size, ACL_MEM_MALLOC_HUGE_FIRST));
        cache_element = n_element;

        // Initialize cache
        int64_t        pool_ne[1] = { n_element };
        माप_प्रकार         pool_nb[1] = { ggml_type_size(dtype) };
        acl_tensor_ptr acl_मान =
            ggml_cann_create_tensor(*buffer, ggml_cann_type_mapping(dtype), ggml_type_size(dtype), pool_ne, pool_nb, 1);
        aclnn_fill_scalar(ctx, value, acl_मान.get());
    }

    फर्काऊ ggml_cann_create_tensor(*buffer, ggml_cann_type_mapping(dtype), ggml_type_size(dtype), ne, nb, dims);
}

शून्य ggml_cann_rms_norm(ggml_backend_cann_context & ctx, ggml_tensor * dst) {
    ggml_tensor * src = dst->src[0];

    acl_tensor_ptr acl_src = ggml_cann_create_tensor(src);
    acl_tensor_ptr acl_dst = ggml_cann_create_tensor(dst);

    भिन्न eps;
    स_नकल(&eps, dst->op_params, परिमाण(भिन्न));

    // build gamma.
    माप_प्रकार acl_gamma_nb[GGML_MAX_DIMS];
    // gamma's type is the same with dst.
    acl_gamma_nb[0] = ggml_type_size(dst->type);
    लागि (पूर्णाङ्क i = 1; i < GGML_MAX_DIMS; i++) {
        acl_gamma_nb[i] = acl_gamma_nb[i - 1] * src->ne[i - 1];
    }
    acl_tensor_ptr acl_gamma = get_cache_acl_tensor(
        ctx, &ctx.rms_norm_one_tensor_cache.cache, ctx.rms_norm_one_tensor_cache.size, src->ne, acl_gamma_nb, dst->type,
        1,    // dims
        1.0f  //मान
    );

    // build rstd.
    int64_t acl_rstd_ne[] = { src->ne[1], src->ne[2], src->ne[3] };
    माप_प्रकार  acl_rstd_nb[GGML_MAX_DIMS - 1];
    // rstd will always be F32.
    acl_rstd_nb[0] = परिमाण(भिन्न);
    लागि (पूर्णाङ्क i = 1; i < GGML_MAX_DIMS - 1; i++) {
        acl_rstd_nb[i] = acl_rstd_nb[i - 1] * acl_rstd_ne[i - 1];
    }
    acl_tensor_ptr acl_rstd =
        get_cache_acl_tensor(ctx, &ctx.rms_norm_zero_tensor_cache.cache, ctx.rms_norm_zero_tensor_cache.size,
                             acl_rstd_ne, acl_rstd_nb, GGML_TYPE_F32, GGML_MAX_DIMS - 1,
                             0.0f  //मान
        );

    GGML_CANN_CALL_ACLNN_OP(ctx, RmsNorm, acl_src.get(), acl_gamma.get(), eps, acl_dst.get(), acl_rstd.get());
}

// TODO: performace is low.
शून्य ggml_cann_diag_mask(ggml_backend_cann_context & ctx, ggml_tensor * dst, भिन्न value) {
    ggml_tensor * src = dst->src[0];

    acl_tensor_ptr acl_src = ggml_cann_create_tensor(src);
    acl_tensor_ptr acl_dst = ggml_cann_create_tensor(dst);

    अचर पूर्णाङ्क n_past = ((int32_t *) dst->op_params)[0];

    ggml_cann_pool_alloc one_tensor_allocator(ctx.pool(), ggml_nbytes(src));
    शून्य *               buffer = one_tensor_allocator.get();

    acl_tensor_ptr mask_tensor = ggml_cann_create_tensor(buffer, ggml_cann_type_mapping(src->type),
                                                         ggml_type_size(src->type), src->ne, src->nb, GGML_MAX_DIMS);

    aclnn_fill_scalar(ctx, value, mask_tensor.get());

    भिन्न          alpha_मान = 1.0f;
    acl_scalar_ptr alpha      = ggml_cann_create_scalar(&alpha_मान, aclDataType::ACL_FLOAT);

    GGML_CANN_CALL_ACLNN_OP(ctx, InplaceTriu, mask_tensor.get(), n_past + 1);
    GGML_CANN_CALL_ACLNN_OP(ctx, Tril, acl_src.get(), n_past + 1, acl_dst.get());
    GGML_CANN_CALL_ACLNN_OP(ctx, InplaceAdd, acl_dst.get(), mask_tensor.get(), alpha.get());
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
स्थिर शून्य aclnn_permute(ggml_backend_cann_context & ctx,
                          aclTensor *                 acl_src,
                          aclTensor *                 acl_dst,
                          int64_t *                   new_dim,
                          uint64_t                    dims) {
    acl_int_array_ptr acl_dims = ggml_cann_create_int_array(new_dim, dims);
    GGML_CANN_CALL_ACLNN_OP(ctx, Permute, acl_src, acl_dims.get(), acl_dst);
}

स्थिर शून्य ggml_cann_im2col_2d_post_process(ggml_backend_cann_context & ctx,
                                             ggml_tensor *               dst,
                                             ggml_tensor *               src1,
                                             aclTensor *                 tmp_cast_tensor,
                                             aclTensor *                 tmp_im2col_tensor) {
    // Permute: [N, IC * KH * KW, OW * OH] -> [N, OW * OH, IC * KH * KW]
    int64_t        dst_ne[] = { dst->ne[0], dst->ne[1] * dst->ne[2], dst->ne[3] };
    माप_प्रकार         dst_nb[] = { dst->nb[0], dst->nb[1], dst->nb[3] };
    acl_tensor_ptr acl_dst  = ggml_cann_create_tensor(dst, dst_ne, dst_nb, GGML_MAX_DIMS - 1);

    int64_t permute_dim[] = { 0, 2, 1 };
    यदि (src1->type != dst->type) {
        aclnn_permute(ctx, tmp_cast_tensor, acl_dst.get(), permute_dim, 3);
    } अन्यथा {
        aclnn_permute(ctx, tmp_im2col_tensor, acl_dst.get(), permute_dim, 3);
    }
}

स्थिर शून्य ggml_cann_im2col_1d_post_process(ggml_backend_cann_context &  ctx,
                                             ggml_tensor *                dst,
                                             ggml_tensor *                src1,
                                             aclTensor *                  tmp_cast_tensor,
                                             aclTensor *                  tmp_im2col_tensor,
                                             अचर std::vector<int64_t> & im2col_op_params) {
    // get params
    अचर int64_t KH             = im2col_op_params[0];
    अचर int64_t KW             = im2col_op_params[1];
    अचर int64_t IW             = im2col_op_params[2];
    अचर int64_t IC             = im2col_op_params[3];
    अचर int64_t N              = im2col_op_params[4];
    अचर int64_t OH             = im2col_op_params[5];
    अचर int64_t OW             = im2col_op_params[6];
    अचर int64_t s0             = im2col_op_params[7];
    अचर int64_t p0             = im2col_op_params[8];
    अचर int64_t d0             = im2col_op_params[9];
    अचर int64_t n_bytes_factor = im2col_op_params[10];

    // Permute: [N, IC * KH * KW, OW * OH] ->
    // [N, OW * OH * n_bytes_factor, IC * KH * KW]
    ggml_cann_pool_alloc tmp_permute_allocator(ctx.pool());
    tmp_permute_allocator.alloc(ggml_nbytes(dst) * n_bytes_factor);
    शून्य * tmp_permute_buffer = tmp_permute_allocator.get();

    int64_t tmp_permute_ne[] = { IC * KH * KW, OW * OH * n_bytes_factor, N };
    माप_प्रकार  tmp_permute_nb[GGML_MAX_DIMS - 1];
    tmp_permute_nb[0] = ggml_type_size(dst->type);
    लागि (पूर्णाङ्क i = 1; i < GGML_MAX_DIMS - 1; i++) {
        tmp_permute_nb[i] = tmp_permute_nb[i - 1] * tmp_permute_ne[i - 1];
    }

    acl_tensor_ptr tmp_permute_tensor =
        ggml_cann_create_tensor(tmp_permute_buffer, ggml_cann_type_mapping(dst->type), ggml_type_size(dst->type),
                                tmp_permute_ne, tmp_permute_nb, GGML_MAX_DIMS - 1, ACL_FORMAT_ND);

    int64_t permute_dim[] = { 0, 2, 1 };
    यदि (src1->type != dst->type) {
        aclnn_permute(ctx, tmp_cast_tensor, tmp_permute_tensor.get(), permute_dim, 3);
    } अन्यथा {
        aclnn_permute(ctx, tmp_im2col_tensor, tmp_permute_tensor.get(), permute_dim, 3);
    }

    // number of times the kernel moves in W dimension
    अचर पूर्णाङ्क n_step_w = (IW + 2 * p0 - d0 * (KW - 1) - 1) / s0 + 1;
    माप_प्रकार    offset;
    शून्य *    cur_dst_buffer = dst->data, *cur_permute_buffer = tmp_permute_buffer;

    // memory copy with offset to restore 1D im2col from 2d
    यदि (IC > 1) {
        offset          = IC * KH * KW * n_step_w * ggml_type_size(dst->type);
        माप_प्रकार cpy_size = KH * KW * ggml_type_size(dst->type);

        लागि (पूर्णाङ्क c = 0; c < IC; c++) {
            cur_permute_buffer = (वर्ण *) tmp_permute_buffer + offset + KH * KW * c * ggml_type_size(dst->type);
            cur_dst_buffer     = (वर्ण *) dst->data + c * KH * KW * n_step_w * ggml_type_size(dst->type);

            लागि (पूर्णाङ्क i = 0; i < n_step_w; i++) {
                ACL_CHECK(aclrtMemcpyAsync(cur_dst_buffer, cpy_size, cur_permute_buffer, cpy_size,
                                           ACL_MEMCPY_DEVICE_TO_DEVICE, ctx.stream()));
                cur_dst_buffer     = (वर्ण *) cur_dst_buffer + KH * KW * ggml_type_size(dst->type);
                cur_permute_buffer = (वर्ण *) cur_permute_buffer + KH * KW * IC * ggml_type_size(dst->type);
            }
        }
    } अन्यथा {
        offset = KH * KW * n_step_w * ggml_type_size(dst->type);  // equal to ggml_nbytes(dst)
        ACL_CHECK(aclrtMemcpyAsync(dst->data, offset, (वर्ण *) tmp_permute_buffer + offset, offset,
                                   ACL_MEMCPY_DEVICE_TO_DEVICE, ctx.stream()));
    }
}

शून्य ggml_cann_im2col(ggml_backend_cann_context & ctx, ggml_tensor * dst) {
    ggml_tensor * src0 = dst->src[0];  // kernel
    ggml_tensor * src1 = dst->src[1];  //आगत

    GGML_TENSOR_द्विआधारी_OP_LOCALS;

    // aclnnIm2col only works on 2D. set s1, p1, d1 to 1 to perform 2D
    // im2col and do post-processing to restore it to 1D.
    अचर तार्किक    is_2D = ((अचर int32_t *) (dst->op_params))[6] == 1;
    अचर int32_t s0    = ((अचर int32_t *) (dst->op_params))[0];
    अचर int32_t s1    = is_2D ? ((अचर int32_t *) (dst->op_params))[1] : 1;
    अचर int32_t p0    = ((अचर int32_t *) (dst->op_params))[2];
    अचर int32_t p1    = is_2D ? ((अचर int32_t *) (dst->op_params))[3] : 1;
    अचर int32_t d0    = ((अचर int32_t *) (dst->op_params))[4];
    अचर int32_t d1    = is_2D ? ((अचर int32_t *) (dst->op_params))[5] : 1;

    अचर int64_t N  = ne13;
    अचर int64_t IC = ne12;
    अचर int64_t KH = ne01;
    अचर int64_t KW = ne00;
    अचर int64_t IW = ne10;

    अचर int64_t OH = is_2D ? ne2 : 1;
    अचर int64_t OW = ne1;

    // memory allocated increased to 3x when is_2D == false
    अचर int64_t n_bytes_factor = is_2D ? 1 : 3;

    // im2col: [N,C,H,W] -> [N, IC * KH * KW, OW * OH * n_bytes_factor]
    acl_tensor_ptr acl_src1        = ggml_cann_create_tensor(src1);
    int64_t        tmp_im2col_ne[] = { OW * OH * n_bytes_factor, IC * KH * KW, N };
    माप_प्रकार         tmp_im2col_nb[GGML_MAX_DIMS - 1];

    tmp_im2col_nb[0] = ggml_type_size(src1->type);
    लागि (पूर्णाङ्क i = 1; i < GGML_MAX_DIMS - 1; i++) {
        tmp_im2col_nb[i] = tmp_im2col_nb[i - 1] * tmp_im2col_ne[i - 1];
    }

    // Calculate im2col.
    // If dst is f16, tmp_buffer is f32, we need alloc src.typesize *
    // dst.elemcount.
    ggml_cann_pool_alloc im2col_allocator(ctx.pool(), ggml_nelements(dst) * ggml_element_size(src1) * n_bytes_factor);
    शून्य *               tmp_im2col_buffer = im2col_allocator.get();

    acl_tensor_ptr tmp_im2col_tensor =
        ggml_cann_create_tensor(tmp_im2col_buffer, ggml_cann_type_mapping(src1->type), ggml_type_size(src1->type),
                                tmp_im2col_ne, tmp_im2col_nb, GGML_MAX_DIMS - 1, ACL_FORMAT_ND);

    std::vector<int64_t> kernel_dims   = { KH, KW };
    std::vector<int64_t> dilation_size = { d1, d0 };
    std::vector<int64_t> padding_dims  = { p1, p0 };
    std::vector<int64_t> stride_dims   = { s1, s0 };
    acl_int_array_ptr    kernel_size   = ggml_cann_create_int_array(kernel_dims.data(), 2);
    acl_int_array_ptr    dilations     = ggml_cann_create_int_array(dilation_size.data(), 2);
    acl_int_array_ptr    paddings      = ggml_cann_create_int_array(padding_dims.data(), 2);
    acl_int_array_ptr    strides       = ggml_cann_create_int_array(stride_dims.data(), 2);
    GGML_CANN_CALL_ACLNN_OP(ctx, Im2col, acl_src1.get(), kernel_size.get(), dilations.get(), paddings.get(),
                            strides.get(), tmp_im2col_tensor.get());

    // Cast if dst is f16.
    acl_tensor_ptr       tmp_cast_tensor;
    ggml_cann_pool_alloc tmp_cast_allocator(ctx.pool());
    शून्य *               tmp_cast_buffer = nullptr;
    यदि (src1->type != dst->type) {
        tmp_cast_allocator.alloc(ggml_nbytes(dst) * n_bytes_factor);
        tmp_cast_buffer = tmp_cast_allocator.get();
        माप_प्रकार temp_cast_nb[GGML_MAX_DIMS - 1];
        temp_cast_nb[0] = ggml_type_size(dst->type);
        लागि (पूर्णाङ्क i = 1; i < GGML_MAX_DIMS - 1; i++) {
            temp_cast_nb[i] = temp_cast_nb[i - 1] * tmp_im2col_ne[i - 1];
        }

        tmp_cast_tensor =
            ggml_cann_create_tensor(tmp_cast_buffer, ggml_cann_type_mapping(dst->type), ggml_type_size(dst->type),
                                    tmp_im2col_ne, temp_cast_nb, GGML_MAX_DIMS - 1, ACL_FORMAT_ND);
        aclnn_cast(ctx, tmp_im2col_tensor.get(), tmp_cast_tensor.get(), ggml_cann_type_mapping(dst->type));
    }

    // post-processing
    यदि (is_2D) {
        ggml_cann_im2col_2d_post_process(ctx, dst, src1, tmp_cast_tensor.get(), tmp_im2col_tensor.get());
    } अन्यथा {
        std::vector<int64_t> im2col_op_params = { KH, KW, IW, IC, N, OH, OW, s0, p0, d0, n_bytes_factor };
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
स्थिर शून्य aclnn_exp(ggml_backend_cann_context & ctx, aclTensor * acl_src) {
    GGML_CANN_CALL_ACLNN_OP(ctx, InplaceExp, acl_src);
}

शून्य aclnn_cos(ggml_backend_cann_context & ctx, aclTensor * acl_src, aclTensor * acl_dst) {
    यदि (acl_dst == nullptr) {
        GGML_CANN_CALL_ACLNN_OP(ctx, InplaceCos, acl_src);
    } अन्यथा {
        GGML_CANN_CALL_ACLNN_OP(ctx, Cos, acl_src, acl_dst);
    }
}

शून्य aclnn_sin(ggml_backend_cann_context & ctx, aclTensor * acl_src, aclTensor * acl_dst) {
    यदि (acl_dst == nullptr) {
        GGML_CANN_CALL_ACLNN_OP(ctx, InplaceSin, acl_src);
    } अन्यथा {
        GGML_CANN_CALL_ACLNN_OP(ctx, Sin, acl_src, acl_dst);
    }
}

शून्य ggml_cann_timestep_embedding(ggml_backend_cann_context & ctx, ggml_tensor * dst) {
    अचर ggml_tensor * src = dst->src[0];

    GGML_ASSERT(src->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    अचर पूर्णाङ्क dim        = dst->op_params[0];
    अचर पूर्णाङ्क max_period = dst->op_params[1];
    पूर्णाङ्क       half       = dim / 2;

    acl_tensor_ptr acl_src = ggml_cann_create_tensor(src);

    // arange: [0, ..., half)
    भिन्न   start             = 0;
    भिन्न   stop              = half;
    भिन्न   step              = 1;
    int64_t n_elements_arange = half;
    int64_t tmp_arange_ne[]   = { half };
    माप_प्रकार  tmp_arange_nb[]   = { परिमाण(dst->type) };

    ggml_cann_pool_alloc arange_allocator(ctx.pool(), half * परिमाण(dst->type));
    शून्य *               tmp_arange_buffer = arange_allocator.get();
    acl_tensor_ptr       tmp_arange_tensor =
        ggml_cann_create_tensor(tmp_arange_buffer, ggml_cann_type_mapping(dst->type), ggml_type_size(dst->type),
                                tmp_arange_ne, tmp_arange_nb, GGML_MAX_DIMS - 3, ACL_FORMAT_ND);

    aclnn_arange(ctx, tmp_arange_tensor.get(), start, stop, step, n_elements_arange);

    // freq
    भिन्न freq_param = -logf(max_period) / half;
    तार्किक  inplace    = सत्य;
    aclnn_muls(ctx, tmp_arange_tensor.get(), freq_param, nullptr, inplace);
    aclnn_exp(ctx, tmp_arange_tensor.get());

    // permute: src [0,1,2,3]->[0,1,3,2]
    int64_t tmp_permute_ne[] = { src->ne[1], src->ne[0], src->ne[2], src->ne[3] };
    माप_प्रकार  tmp_permute_nb[GGML_MAX_DIMS];
    tmp_permute_nb[0] = ggml_type_size(src->type);
    लागि (पूर्णाङ्क i = 1; i < GGML_MAX_DIMS; i++) {
        tmp_permute_nb[i] = tmp_permute_nb[i - 1] * tmp_permute_ne[i - 1];
    }

    ggml_cann_pool_alloc permute_allocator(ctx.pool(), ggml_nbytes(src));
    शून्य *               tmp_permute_buffer = permute_allocator.get();
    acl_tensor_ptr       tmp_permute_tensor =
        ggml_cann_create_tensor(tmp_permute_buffer, ggml_cann_type_mapping(src->type), ggml_type_size(src->type),
                                tmp_permute_ne, tmp_permute_nb, GGML_MAX_DIMS, ACL_FORMAT_ND);
    int64_t permute_dim[] = { 0, 1, 3, 2 };
    int64_t अङ्क_dims      = 4;
    aclnn_permute(ctx, acl_src.get(), tmp_permute_tensor.get(), permute_dim, अङ्क_dims);

    // timestep * freq
    int64_t tmp_mul_ne[] = { src->ne[1] * half, src->ne[0], src->ne[2], src->ne[3] };
    माप_प्रकार  tmp_mul_nb[GGML_MAX_DIMS];
    tmp_mul_nb[0] = ggml_type_size(src->type);
    लागि (पूर्णाङ्क i = 1; i < GGML_MAX_DIMS; i++) {
        tmp_mul_nb[i] = tmp_mul_nb[i - 1] * tmp_mul_ne[i - 1];
    }

    पूर्णाङ्क mul_nelements = src->ne[1] * half * src->ne[0] * src->ne[2] * src->ne[3];

    ggml_cann_pool_alloc mul_allocator(ctx.pool(), mul_nelements * ggml_type_size(src->type));
    शून्य *               tmp_mul_buffer = mul_allocator.get();
    acl_tensor_ptr       tmp_mul_tensor =
        ggml_cann_create_tensor(tmp_mul_buffer, ggml_cann_type_mapping(src->type), ggml_type_size(src->type),
                                tmp_mul_ne, tmp_mul_nb, GGML_MAX_DIMS, ACL_FORMAT_ND);
    aclnn_mul(ctx, tmp_permute_tensor.get(), tmp_arange_tensor.get(), tmp_mul_tensor.get());

    // cos
    ggml_cann_pool_alloc cos_allocator(ctx.pool(), mul_nelements * ggml_type_size(src->type));
    शून्य *               tmp_cos_buffer = cos_allocator.get();
    acl_tensor_ptr       tmp_cos_tensor =
        ggml_cann_create_tensor(tmp_cos_buffer, ggml_cann_type_mapping(dst->type), ggml_type_size(dst->type),
                                tmp_mul_ne, tmp_mul_nb, GGML_MAX_DIMS, ACL_FORMAT_ND);

    aclnn_cos(ctx, tmp_mul_tensor.get(), tmp_cos_tensor.get());

    // sin
    ggml_cann_pool_alloc sin_allocator(ctx.pool(), mul_nelements * ggml_type_size(src->type));
    शून्य *               tmp_sin_buffer = sin_allocator.get();
    acl_tensor_ptr       tmp_sin_tensor =
        ggml_cann_create_tensor(tmp_sin_buffer, ggml_cann_type_mapping(dst->type), ggml_type_size(dst->type),
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
स्थिर शून्य aclnn_pow_tensor_tensor(ggml_backend_cann_context & ctx, aclTensor * acl_dst, aclTensor * acl_exp) {
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
स्थिर शून्य aclnn_get_slope_inner(ggml_backend_cann_context & ctx,
                                  शून्य *                      slope_buffer,
                                  भिन्न                       m,
                                  int64_t                     size,
                                  भिन्न                       start,
                                  भिन्न                       stop,
                                  भिन्न                       step,
                                  ggml_type                   dtype) {
    aclDataType acl_type  = ggml_cann_type_mapping(dtype);
    माप_प्रकार      type_size = ggml_type_size(dtype);

    int64_t ne[] = { size };
    माप_प्रकार  nb[] = { type_size };

    ggml_cann_pool_alloc arange_allocator(ctx.pool(), size * type_size);
    शून्य *               arange_buffer = arange_allocator.get();

    acl_tensor_ptr arange_tensor = ggml_cann_create_tensor(arange_buffer, acl_type, type_size, ne, nb, 1);
    aclnn_arange(ctx, arange_tensor.get(), start, stop, step, size);

    acl_tensor_ptr slope_tensor = ggml_cann_create_tensor(slope_buffer, acl_type, type_size, ne, nb, 1);

    acl_scalar_ptr sc = ggml_cann_create_scalar(&m, aclDataType::ACL_FLOAT);

    GGML_CANN_CALL_ACLNN_OP(ctx, PowScalarTensor, sc.get(), arange_tensor.get(), slope_tensor.get());
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
स्थिर शून्य aclnn_get_slope(ggml_backend_cann_context & ctx,
                            int64_t                     n_head,
                            शून्य *                      slope_buffer,
                            भिन्न                       max_bias,
                            ggml_type                   dtype) {
    अचर पूर्णाङ्क n_head_log2 = 1u << (uint32_t) न्यूनमान(log2(n_head));

    भिन्न m0 = powf(2.0f, -(max_bias) / n_head_log2);
    भिन्न m1 = powf(2.0f, -(max_bias / 2.0f) / n_head_log2);

    // const float slope = (max_bias > 0.0f) ?
    //                          h < n_head_log2 ?
    //                              powf(m0, h + 1) :
    //                              powf(m1, 2*(h - n_head_log2) + 1) :
    //                          1.0f;
    // arange1
    भिन्न start = 0 + 1;
    भिन्न end   = (n_head_log2 - 1) + 1;
    भिन्न step  = 1;
    भिन्न count = n_head_log2;
    // end needs to be +1 because aclnn uses a left-closed, right-open interval.
    aclnn_get_slope_inner(ctx, slope_buffer, m0, count, start, end + 1, step, dtype);
    यदि (n_head_log2 < n_head) {
        // arange2
        start = 2 * (n_head_log2 - n_head_log2) + 1;
        end   = 2 * ((n_head - 1) - n_head_log2) + 1;
        step  = 2;
        count = n_head - n_head_log2;
        aclnn_get_slope_inner(ctx, (वर्ण *) slope_buffer + n_head_log2 * ggml_type_size(dtype), m1, count, start, end + 1,
                              step, dtype);
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
स्थिर शून्य aclnn_add_alibi(ggml_backend_cann_context & ctx,
                            ggml_tensor *               mask,
                            ggml_tensor *               dst,
                            शून्य *                      dst_ptr,
                            भिन्न                       max_bias) {
    शून्य * slope_buffer = nullptr;
    शून्य * bias_buffer  = nullptr;

    यदि (max_bias > 0.0f) {
        int64_t              n_heads = dst->ne[2];
        ggml_cann_pool_alloc slope_allocator(ctx.pool(), n_heads * परिमाण(भिन्न));
        slope_buffer = slope_allocator.get();
        ggml_cann_pool_alloc bias_allocator(ctx.pool(), ggml_nelements(dst) * ggml_element_size(dst));
        bias_buffer = bias_allocator.get();
        aclnn_get_slope(ctx, n_heads, slope_buffer, max_bias, GGML_TYPE_F32);
    }

    // broadcast for mask, slop and dst;
    int64_t nr2 = dst->ne[2] / mask->ne[2];
    int64_t nr3 = dst->ne[3] / mask->ne[3];

    // broadcast the mask across rows
    int64_t mask_ne[] = { mask->ne[0], dst->ne[1], mask->ne[2], 1, mask->ne[3], 1 };
    माप_प्रकार  mask_nb[] = { mask_nb[0] = mask->nb[0], mask_nb[1] = mask->nb[1], mask_nb[2] = mask->nb[2],
                          mask_nb[3] = mask->nb[2], mask_nb[4] = mask->nb[3], mask_nb[5] = mask->nb[3] };

    int64_t dst_ne[] = { dst->ne[0], dst->ne[1], mask->ne[2], nr2, mask->ne[3], nr3 };
    माप_प्रकार  dst_nb[] = { dst_nb[0] = dst->nb[0], dst_nb[1] = dst->nb[1], dst_nb[2] = dst->nb[2],
                         dst_nb[3] = dst->nb[2], dst_nb[4] = dst->nb[3], dst_nb[5] = dst->nb[3] };

    // slope is a 1 dim tensor, slope.ne2 == dst.ne2
    int64_t slope_ne[] = { 1, 1, mask->ne[2], nr2, 1, 1 };
    माप_प्रकार  slope_nb[GGML_MAX_DIMS + 2];
    slope_nb[0] = परिमाण(भिन्न);
    लागि (पूर्णाङ्क i = 1; i < GGML_MAX_DIMS + 2; i++) {
        slope_nb[i] = slope_nb[i - 1] * slope_ne[i - 1];
    }

    acl_tensor_ptr acl_slope =
        ggml_cann_create_tensor(slope_buffer, ACL_FLOAT, परिमाण(भिन्न), slope_ne, slope_nb, GGML_MAX_DIMS + 2);
    acl_tensor_ptr acl_mask = ggml_cann_create_tensor(mask, mask_ne, mask_nb, GGML_MAX_DIMS + 2);

    // write data into dst_ptr using only the shape information of the dst tensor.
    acl_tensor_ptr acl_dst = ggml_cann_create_tensor(dst_ptr, ggml_cann_type_mapping(dst->type),
                                                     ggml_type_size(dst->type), dst_ne, dst_nb, GGML_MAX_DIMS + 2);

    यदि (max_bias > 0.0f) {
        int64_t bias_ne[] = { mask->ne[0], dst->ne[1], mask->ne[2], nr2, mask->ne[3], 1 };
        माप_प्रकार  bias_nb[GGML_MAX_DIMS + 2];
        bias_nb[0] = परिमाण(भिन्न);
        लागि (पूर्णाङ्क i = 1; i < GGML_MAX_DIMS + 2; i++) {
            bias_nb[i] = bias_nb[i - 1] * bias_ne[i - 1];
        }
        acl_tensor_ptr bias_tensor =
            ggml_cann_create_tensor(bias_buffer, ACL_FLOAT, परिमाण(भिन्न), bias_ne, bias_nb, GGML_MAX_DIMS + 2);

        aclnn_mul(ctx, acl_slope.get(), acl_mask.get(), bias_tensor.get());
        aclnn_add(ctx, acl_dst.get(), bias_tensor.get());
    } अन्यथा {
        aclnn_add(ctx, acl_dst.get(), acl_mask.get());
    }
}

शून्य ggml_cann_cpy(ggml_backend_cann_context & ctx, ggml_tensor * dst) {
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
स्थिर शून्य aclnn_softmax(ggml_backend_cann_context & ctx, aclTensor * acl_src, int64_t dim, aclTensor * acl_dst) {
    GGML_CANN_CALL_ACLNN_OP(ctx, Softmax, acl_src, dim, acl_dst);
}

शून्य ggml_cann_softmax(ggml_backend_cann_context & ctx, ggml_tensor * dst) {
    ggml_tensor * src0 = dst->src[0];
    ggml_tensor * src1 = dst->src[1];  // mask

    acl_tensor_ptr acl_src0 = ggml_cann_create_tensor(src0);
    acl_tensor_ptr acl_dst  = ggml_cann_create_tensor(dst);

    भिन्न scale    = 1.0f;
    भिन्न max_bias = 0.0f;

    स_नकल(&scale, (भिन्न *) dst->op_params + 0, परिमाण(भिन्न));
    स_नकल(&max_bias, (भिन्न *) dst->op_params + 1, परिमाण(भिन्न));

    // input mul scale
    acl_scalar_ptr       acl_scale = ggml_cann_create_scalar(&scale, aclDataType::ACL_FLOAT);
    ggml_cann_pool_alloc src_tensor_allocator(ctx.pool(), ggml_nbytes(src0));
    शून्य *               src_tensor_buffer = src_tensor_allocator.get();
    acl_tensor_ptr       softmax_tensor = ggml_cann_create_tensor(src_tensor_buffer, ggml_cann_type_mapping(src0->type),
                                                                  ggml_element_size(src0), src0->ne, src0->nb, GGML_MAX_DIMS);

    aclnn_muls(ctx, acl_src0.get(), scale, softmax_tensor.get(), असत्य);

    // mask
    यदि (src1) {
        aclnn_add_alibi(ctx, src1, src0, src_tensor_buffer, max_bias);
    }
    // softmax
    aclnn_softmax(ctx, softmax_tensor.get(), 3, acl_dst.get());
}


शून्य ggml_cann_get_rows(ggml_backend_cann_context & ctx, ggml_tensor * dst) {
    ggml_tensor * src0 = dst->src[0];  // weight
    ggml_tensor * src1 = dst->src[1];  // index

    GGML_ASSERT(dst->type == GGML_TYPE_F32 || dst->type == GGML_TYPE_F16
                || dst->type == GGML_TYPE_BF16);

    // n_idx: number of row indices per (i2, i3) batch slice.
    // ggml guarantees: src0->ne[2] == src1->ne[1], src0->ne[3] == src1->ne[2], src1->ne[3] == 1.
    अचर int64_t n_idx = src1->ne[0];

    // Gather all (i2, i3) batch slices from src into dst.
    // ggml_cann_create_tensor reverses dims, so ACL sees [ne1, ne0].
    // GatherV2 with dim=0 gathers along ACL dim-0 == ggml ne[1] (the vocabulary / row axis).
    // nb: the 4 strides of the source buffer (nb[0..1] for the 2D slice shape,
    //     nb[2..3] for computing per-batch-slice base pointer offsets).
    स्वतः gather_batched = [&](शून्य * src_base, aclDataType acl_type, माप_प्रकार type_size,
                              अचर माप_प्रकार * nb) {
        int64_t src_ne[2]  = { src0->ne[0], src0->ne[1] };
        माप_प्रकार  src_nb_2d[2] = { nb[0], nb[1] };
        int64_t dst_ne[2]  = { src0->ne[0], n_idx };
        माप_प्रकार  dst_nb_2d[2] = { dst->nb[0], dst->nb[1] };
        int64_t idx_ne[1]  = { n_idx };
        माप_प्रकार  idx_nb[1]  = { (माप_प्रकार)ggml_element_size(src1) };

        लागि (int64_t i3 = 0; i3 < src0->ne[3]; i3++) {
            लागि (int64_t i2 = 0; i2 < src0->ne[2]; i2++) {
                acl_tensor_ptr acl_src = ggml_cann_create_tensor(
                    (वर्ण *)src_base + i3 * nb[3] + i2 * nb[2],
                    acl_type, type_size, src_ne, src_nb_2d, 2);
                acl_tensor_ptr acl_idx = ggml_cann_create_tensor(
                    (वर्ण *)src1->data + i3 * src1->nb[2] + i2 * src1->nb[1],
                    ggml_cann_type_mapping(src1->type), (माप_प्रकार)ggml_element_size(src1),
                    idx_ne, idx_nb, 1);
                acl_tensor_ptr acl_dst = ggml_cann_create_tensor(
                    (वर्ण *)dst->data + i3 * dst->nb[3] + i2 * dst->nb[2],
                    acl_type, type_size, dst_ne, dst_nb_2d, 2);
                GGML_CANN_CALL_ACLNN_OP(ctx, GatherV2, acl_src.get(), 0, acl_idx.get(), acl_dst.get());
            }
        }
    };

    विकल्प (src0->type) {
        पक्ष GGML_TYPE_BF16:
        पक्ष GGML_TYPE_F16:
        पक्ष GGML_TYPE_F32:
            यदि (src0->type == dst->type) {
                gather_batched(src0->data,
                               ggml_cann_type_mapping(src0->type), ggml_type_size(src0->type),
                               src0->nb);
            } अन्यथा {
                // Cast src0 to dst type, then gather.
                ggml_cann_pool_alloc src_cast_allocator(ctx.pool(),
                                                        ggml_nelements(src0) * ggml_element_size(dst));
                माप_प्रकार src_cast_nb[GGML_MAX_DIMS];
                src_cast_nb[0] = ggml_type_size(dst->type);
                लागि (पूर्णाङ्क i = 1; i < GGML_MAX_DIMS; i++) {
                    src_cast_nb[i] = src_cast_nb[i - 1] * src0->ne[i - 1];
                }
                acl_tensor_ptr acl_src0     = ggml_cann_create_tensor(src0);
                acl_tensor_ptr acl_src_cast = ggml_cann_create_tensor(
                    src_cast_allocator.get(), ggml_cann_type_mapping(dst->type), ggml_type_size(dst->type),
                    src0->ne, src_cast_nb, GGML_MAX_DIMS);
                aclnn_cast(ctx, acl_src0.get(), acl_src_cast.get(), ggml_cann_type_mapping(dst->type));

                gather_batched(src_cast_allocator.get(),
                               ggml_cann_type_mapping(dst->type), ggml_type_size(dst->type),
                               src_cast_nb);
            }
            रोक;
        पक्ष GGML_TYPE_Q8_0:
            {
                // Dequantize Q8_0 to dst type, then gather.
                माप_प्रकार  weight_nb[GGML_MAX_DIMS + 1], scale_nb[GGML_MAX_DIMS + 1], dequant_nb[GGML_MAX_DIMS + 1];
                int64_t weight_ne[GGML_MAX_DIMS + 1], scale_ne[GGML_MAX_DIMS + 1], *dequant_ne;
                weight_ne[0] = QK8_0;
                weight_ne[1] = src0->ne[0] / QK8_0;
                weight_nb[0] = परिमाण(int8_t);
                weight_nb[1] = weight_nb[0] * weight_ne[0];
                लागि (पूर्णाङ्क i = 2; i < GGML_MAX_DIMS + 1; i++) {
                    weight_ne[i] = src0->ne[i - 1];
                    weight_nb[i] = weight_nb[i - 1] * weight_ne[i - 1];
                }
                scale_ne[0] = 1;
                scale_ne[1] = src0->ne[0] / QK8_0;
                scale_nb[0] = परिमाण(uint16_t);
                scale_nb[1] = scale_nb[0] * scale_ne[0];
                लागि (पूर्णाङ्क i = 2; i < GGML_MAX_DIMS + 1; i++) {
                    scale_ne[i] = src0->ne[i - 1];
                    scale_nb[i] = scale_nb[i - 1] * scale_ne[i - 1];
                }
                dequant_ne    = weight_ne;
                dequant_nb[0] = ggml_type_size(dst->type);
                लागि (पूर्णाङ्क i = 1; i < GGML_MAX_DIMS + 1; i++) {
                    dequant_nb[i] = dequant_nb[i - 1] * dequant_ne[i - 1];
                }
                अचर int64_t scale_offset = ggml_nelements(src0) * परिमाण(int8_t);
                ggml_cann_pool_alloc dequant_allocator(ctx.pool(),
                                                       ggml_nelements(src0) * ggml_type_size(dst->type));
                acl_tensor_ptr acl_weight = ggml_cann_create_tensor(src0->data, ACL_INT8, परिमाण(int8_t),
                                                                     weight_ne, weight_nb, GGML_MAX_DIMS + 1);
                acl_tensor_ptr acl_scale  = ggml_cann_create_tensor(
                    src0->data, ACL_FLOAT16, परिमाण(uint16_t), scale_ne, scale_nb,
                    GGML_MAX_DIMS + 1, ACL_FORMAT_ND, scale_offset);
                acl_tensor_ptr acl_dequant = ggml_cann_create_tensor(
                    dequant_allocator.get(), ggml_cann_type_mapping(dst->type),
                    ggml_type_size(dst->type), dequant_ne, dequant_nb, GGML_MAX_DIMS + 1);
                aclnn_mul(ctx, acl_weight.get(), acl_scale.get(), acl_dequant.get());

                // Reinterpret dequant buffer as 4D [src0->ne] with contiguous strides.
                dequant_ne    = src0->ne;
                dequant_nb[0] = ggml_type_size(dst->type);
                लागि (पूर्णाङ्क i = 1; i < GGML_MAX_DIMS; i++) {
                    dequant_nb[i] = dequant_nb[i - 1] * src0->ne[i - 1];
                }
                gather_batched(dequant_allocator.get(),
                               ggml_cann_type_mapping(dst->type), ggml_type_size(dst->type),
                               dequant_nb);
                रोक;
            }
        स्वभाव:
            GGML_ABORT("Unsupported tensor type for GGML_OP_GET_ROWS");
            रोक;
    }
}

शून्य ggml_cann_set_rows(ggml_backend_cann_context & ctx, ggml_tensor * dst) {
    ggml_tensor * src0 = dst->src[0];  // source values
    ggml_tensor * src1 = dst->src[1];  // row indices

    // n_idx: number of source rows to scatter per batch slice.
    // ggml guarantees: src0->ne[1] == src1->ne[0].
    अचर int64_t n_idx = src1->ne[0];

    // Copy n_idx rows of src [ne0, n_idx] into dst [ne0, ne1] at positions given by a 1D index.
    // ggml_cann_create_tensor reverses dims, so ACL sees [ne1, ne0] for dst.
    // InplaceIndexCopy with dim=0 copies along ACL dim-0 == ggml ne[1] (the row axis).
    // src_nb: the 4 strides of the source buffer (nb[0..1] for the 2D slice shape,
    //         nb[2..3] for computing per-batch-slice base pointer offsets).
    स्वतः scatter_batched = [&](शून्य * src_base, aclDataType acl_type, माप_प्रकार type_size,
                               अचर माप_प्रकार * src_nb) {
        int64_t d_ne[2]    = { dst->ne[0], dst->ne[1] };
        माप_प्रकार  d_nb[2]    = { dst->nb[0], dst->nb[1] };
        int64_t s_ne[2]    = { dst->ne[0], n_idx };
        माप_प्रकार  s_nb_2d[2] = { src_nb[0], src_nb[1] };
        int64_t i_ne[1]    = { n_idx };
        माप_प्रकार  i_nb[1]    = { (माप_प्रकार)ggml_element_size(src1) };

        लागि (int64_t i3 = 0; i3 < dst->ne[3]; i3++) {
            लागि (int64_t i2 = 0; i2 < dst->ne[2]; i2++) {
                acl_tensor_ptr acl_dst = ggml_cann_create_tensor(
                    (वर्ण *)dst->data + i3 * dst->nb[3] + i2 * dst->nb[2],
                    acl_type, type_size, d_ne, d_nb, 2);
                acl_tensor_ptr acl_idx = ggml_cann_create_tensor(
                    (वर्ण *)src1->data + (i3 % src1->ne[2]) * src1->nb[2] + (i2 % src1->ne[1]) * src1->nb[1],
                    ggml_cann_type_mapping(src1->type), (माप_प्रकार)ggml_element_size(src1),
                    i_ne, i_nb, 1);
                acl_tensor_ptr acl_src = ggml_cann_create_tensor(
                    (वर्ण *)src_base + i3 * src_nb[3] + i2 * src_nb[2],
                    acl_type, type_size, s_ne, s_nb_2d, 2);
                GGML_CANN_CALL_ACLNN_OP(ctx, InplaceIndexCopy, acl_dst.get(), 0, acl_idx.get(), acl_src.get());
            }
        }
    };

    विकल्प (dst->type) {
        पक्ष GGML_TYPE_F32:
            scatter_batched(src0->data,
                            ggml_cann_type_mapping(dst->type), ggml_type_size(dst->type),
                            src0->nb);
            रोक;
        पक्ष GGML_TYPE_F16:
        पक्ष GGML_TYPE_BF16:
            {
                // Cast src0 (F32) to dst type first.
                ggml_cann_pool_alloc src_cast_allocator(ctx.pool(),
                                                        ggml_nelements(src0) * ggml_type_size(dst->type));
                माप_प्रकार src_cast_nb[GGML_MAX_DIMS];
                src_cast_nb[0] = ggml_type_size(dst->type);
                लागि (पूर्णाङ्क i = 1; i < GGML_MAX_DIMS; i++) {
                    src_cast_nb[i] = src_cast_nb[i - 1] * src0->ne[i - 1];
                }
                acl_tensor_ptr acl_src0     = ggml_cann_create_tensor(src0);
                acl_tensor_ptr acl_src_cast = ggml_cann_create_tensor(
                    src_cast_allocator.get(), ggml_cann_type_mapping(dst->type), ggml_type_size(dst->type),
                    src0->ne, src_cast_nb, GGML_MAX_DIMS);
                aclnn_cast(ctx, acl_src0.get(), acl_src_cast.get(), ggml_cann_type_mapping(dst->type));

                scatter_batched(src_cast_allocator.get(),
                                ggml_cann_type_mapping(dst->type), ggml_type_size(dst->type),
                                src_cast_nb);
                रोक;
            }
        स्वभाव:
            GGML_ABORT("Unsupported tensor type for GGML_OP_SET_ROWS");
            रोक;
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
स्थिर शून्य aclnn_repeat_interleave(ggml_backend_cann_context & ctx,
                                    aclTensor *                 acl_src,
                                    aclTensor *                 acl_dst,
                                    int64_t                     dim,
                                    int64_t                     repeats,
                                    int64_t                     output_size) {
    GGML_CANN_CALL_ACLNN_OP(ctx, RepeatInterleaveIntWithDim, acl_src, repeats, dim, output_size, acl_dst);
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
स्थिर शून्य ggml_cann_mat_mul_fp(ggml_backend_cann_context & ctx, ggml_tensor * dst) {
    ggml_tensor * weight = dst->src[0];  // weight
    ggml_tensor * आगत  = dst->src[1];  //आगत

    // when weight ne2 or ne3 is 1, aclnnMatmulGetWorkspaceSize will auto
    // broadcast, when weight ne2 or ne3 is not 1, weight need repeat.
    BCAST_MUL_MAT_SHAPE(आगत, weight, dst);

    int64_t n_dims = bcast_dims;
    यदि (bcast_आगत_ne[3] == bcast_weight_ne[3] && bcast_आगत_ne[3] == 1) {
        यदि (bcast_आगत_ne[2] == 1 && bcast_weight_ne[2] == 1) {
            n_dims = 2;
        } अन्यथा यदि (bcast_आगत_ne[2] == 1) {
            n_dims = 3;
        }
    }

    acl_tensor_ptr acl_आगत_tensor = ggml_cann_create_tensor(आगत, bcast_आगत_ne, bcast_आगत_nb, n_dims);
    int64_t        transpose_ne[]   = { bcast_weight_ne[1], bcast_weight_ne[0], bcast_weight_ne[2],
                                        bcast_weight_ne[3], bcast_weight_ne[4], bcast_weight_ne[5] };
    माप_प्रकार         transpose_nb[]   = { bcast_weight_nb[1], bcast_weight_nb[0], bcast_weight_nb[2],
                                        bcast_weight_nb[3], bcast_weight_nb[4], bcast_weight_nb[5] };
    acl_tensor_ptr acl_weight_tensor;

    // Only check env once.
    स्थिर तार्किक weight_to_nz = parse_bool(get_env_as_lowercase("GGML_CANN_WEIGHT_NZ").value_or("on"));
    यदि (weight_to_nz && weight->type != GGML_TYPE_BF16 && is_matmul_weight(weight)) {
        acl_weight_tensor = ggml_cann_create_tensor(weight, transpose_ne, transpose_nb, n_dims, ACL_FORMAT_FRACTAL_NZ);
    } अन्यथा {
        acl_weight_tensor = ggml_cann_create_tensor(weight, transpose_ne, transpose_nb, n_dims, ACL_FORMAT_ND);
    }
    acl_tensor_ptr acl_dst = ggml_cann_create_tensor(dst, bcast_dst_ne, bcast_dst_nb, n_dims);

    विकल्प (n_dims) {
        पक्ष 2:
            GGML_CANN_CALL_ACLNN_OP(ctx, Mm, acl_आगत_tensor.get(), acl_weight_tensor.get(), acl_dst.get(), 2);
            रोक;
        पक्ष 3:
            GGML_CANN_CALL_ACLNN_OP(ctx, BatchMatMul, acl_आगत_tensor.get(), acl_weight_tensor.get(), acl_dst.get(),
                                    2);
            रोक;
        स्वभाव:
            // ALLOW_FP32_DOWN_PRECISION, when input is
            // fp32, atlas a2 will transpose it to HFLOAT32.
            GGML_CANN_CALL_ACLNN_OP(ctx, Matmul, acl_आगत_tensor.get(), acl_weight_tensor.get(), acl_dst.get(), 1);
            रोक;
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
स्थिर शून्य ggml_cann_mul_mat_quant(ggml_backend_cann_context & ctx, ggml_tensor * dst, अचर गणना ggml_type type) {
    ggml_tensor * src0 = dst->src[0];  // weight
    ggml_tensor * src1 = dst->src[1];  //आगत

    // The shape of the weight is NCHW.
    // Matrix multiplication uses HW dims.
    // HC is regarded as batch.
    // weight need transpose.
    भिन्न weight_elem_size;
    यदि (type == GGML_TYPE_Q4_0) {
        weight_elem_size = भिन्न(परिमाण(uint8_t)) / 2;
    } अन्यथा यदि (type == GGML_TYPE_Q8_0) {
        weight_elem_size = भिन्न(परिमाण(uint8_t));
    } अन्यथा {
        GGML_ABORT("Only support Q4_0 and Q8_0 MUL_MAT");
    }
    भिन्न  weight_nb[]   = { src0->ne[0] * weight_elem_size, weight_elem_size };
    माप_प्रकार weight_stride = src0->ne[1] * src0->ne[0] * weight_elem_size;
    माप_प्रकार weight_size   = weight_stride * src0->ne[2] * src0->ne[3];

    // scale stored at the end of weight. Also need transpose.
    माप_प्रकार scale_elem_size = परिमाण(uint16_t);
    माप_प्रकार scale_nb[]      = { src0->ne[0] / QK8_0 * scale_elem_size, scale_elem_size };
    माप_प्रकार scale_stride    = src0->ne[1] * src0->ne[0] / QK8_0 * scale_elem_size;
    वर्ण * scale_offset    = (वर्ण *) src0->data + weight_size;

    //आगत
    माप_प्रकार               आगत_elem_size = परिमाण(uint16_t);
    int64_t              आगत_ne[]      = { src1->ne[0], src1->ne[1] };
    माप_प्रकार               आगत_nb[]      = { आगत_elem_size, आगत_ne[0] * आगत_elem_size };
    माप_प्रकार               आगत_stride    = आगत_ne[0] * आगत_ne[1] * आगत_elem_size;
    ggml_cann_pool_alloc आगत_alloctor(ctx.pool());
    शून्य *               आगत_buffer = src1->data;

    // case in
    यदि (src1->type != GGML_TYPE_F16) {
        acl_tensor_ptr acl_src1_tensor = ggml_cann_create_tensor(src1);
        आगत_buffer                   = आगत_alloctor.alloc(ggml_nelements(src1) * आगत_elem_size);

        int64_t * आगत_cast_ne = src1->ne;
        माप_प्रकार    आगत_cast_nb[GGML_MAX_DIMS];
        आगत_cast_nb[0] = परिमाण(uint16_t);
        लागि (पूर्णाङ्क i = 1; i < GGML_MAX_DIMS; i++) {
            आगत_cast_nb[i] = आगत_cast_nb[i - 1] * आगत_cast_ne[i - 1];
        }

        acl_tensor_ptr acl_आगत_tensor = ggml_cann_create_tensor(आगत_buffer, ACL_FLOAT16, आगत_elem_size,
                                                                  आगत_cast_ne, आगत_cast_nb, GGML_MAX_DIMS);
        aclnn_cast(ctx, acl_src1_tensor.get(), acl_आगत_tensor.get(), ACL_FLOAT16);
    }

    // output
    माप_प्रकार               output_elem_size = परिमाण(uint16_t);
    माप_प्रकार               output_nb[]      = { output_elem_size, dst->ne[0] * output_elem_size };
    ggml_cann_pool_alloc output_allocator(ctx.pool());
    शून्य *               output_buffer = output_allocator.alloc(ggml_nelements(dst) * output_elem_size);
    माप_प्रकार               output_stride = dst->ne[0] * dst->ne[1] * output_elem_size;

    // aclnn
    int64_t              max_elem_size = 65535;
    int64_t              split_size    = (src0->ne[1] / max_elem_size) + 1;
    ggml_cann_pool_alloc workspace_allocator(ctx.pool());
    लागि (int64_t n1 = 0; n1 < src1->ne[3]; n1++) {
        लागि (int64_t c1 = 0; c1 < src1->ne[2]; c1++) {
            int64_t n0 = n1 / (src1->ne[3] / src0->ne[3]);
            int64_t c0 = c1 / (src1->ne[2] / src0->ne[2]);

            int64_t batch1 = (n1 * src1->ne[2]) + c1;
            int64_t batch0 = (n0 * src0->ne[2]) + c0;

            acl_tensor_ptr acl_आगत_tensor = ggml_cann_create_tensor(
                (वर्ण *) आगत_buffer + batch1 * आगत_stride, ACL_FLOAT16, आगत_elem_size, आगत_ne, आगत_nb, 2);

            // first split
            int64_t weight_ne_offset = 0;
            int64_t weight_ne[2]     = { max_elem_size > src0->ne[1] ? src0->ne[1] : max_elem_size, src0->ne[0] };
            int64_t scale_ne_offset  = 0;
            int64_t scale_ne[2]      = { weight_ne[0], weight_ne[1] / QK8_0 };
            int64_t output_ne_offset = 0;
            int64_t output_ne[2]     = { weight_ne[0], dst->ne[1] };

            acl_tensor_ptr acl_weight_tensor =
                ggml_cann_create_tensor((वर्ण *) src0->data + batch0 * weight_stride, ggml_cann_type_mapping(type),
                                        weight_elem_size, weight_ne, weight_nb, 2, ACL_FORMAT_ND, weight_ne_offset);
            acl_tensor_ptr acl_scale_tensor =
                ggml_cann_create_tensor(scale_offset + batch0 * scale_stride, ACL_FLOAT16, scale_elem_size, scale_ne,
                                        scale_nb, 2, ACL_FORMAT_ND, scale_ne_offset);
            acl_tensor_ptr acl_output_tensor =
                ggml_cann_create_tensor((वर्ण *) output_buffer + batch1 * output_stride, ACL_FLOAT16, output_elem_size,
                                        output_ne, output_nb, 2, ACL_FORMAT_ND, output_ne_offset);
            int64_t antiquantGroupSize = 0;
            यदि (src0->ne[0] > QK8_0) {
                antiquantGroupSize = QK8_0;
            }
            GGML_CANN_CALL_ACLNN_OP(ctx, WeightQuantBatchMatmulV2, acl_आगत_tensor.get(), acl_weight_tensor.get(),
                                    acl_scale_tensor.get(), nullptr, nullptr, nullptr, nullptr, antiquantGroupSize,
                                    acl_output_tensor.get());

            // other splits
            लागि (int64_t split = 1; split < split_size; split++) {
                weight_ne_offset += weight_elem_size * weight_ne[0] * weight_ne[1];
                weight_ne[0] =
                    max_elem_size * (split + 1) > src0->ne[1] ? src0->ne[1] - (max_elem_size * split) : max_elem_size;
                scale_ne_offset += scale_elem_size * scale_ne[0] * scale_ne[1];
                scale_ne[0] = weight_ne[0];
                output_ne_offset += output_elem_size * output_ne[0] * output_ne[1];
                output_ne[0] = weight_ne[0];

                acl_weight_tensor =
                    ggml_cann_create_tensor((वर्ण *) src0->data + batch0 * weight_stride, ggml_cann_type_mapping(type),
                                            weight_elem_size, weight_ne, weight_nb, 2, ACL_FORMAT_ND, weight_ne_offset);
                acl_scale_tensor =
                    ggml_cann_create_tensor(scale_offset + batch0 * scale_stride, ACL_FLOAT16, scale_elem_size,
                                            scale_ne, scale_nb, 2, ACL_FORMAT_ND, scale_ne_offset);
                acl_output_tensor =
                    ggml_cann_create_tensor((वर्ण *) output_buffer + batch1 * output_stride, ACL_FLOAT16,
                                            output_elem_size, output_ne, output_nb, 2, ACL_FORMAT_ND, output_ne_offset);
                GGML_CANN_CALL_ACLNN_OP(ctx, WeightQuantBatchMatmulV2, acl_आगत_tensor.get(), acl_weight_tensor.get(),
                                        acl_scale_tensor.get(), nullptr, nullptr, nullptr, nullptr, antiquantGroupSize,
                                        acl_output_tensor.get());
            }
        }
    }

    // cast out
    यदि (dst->type != GGML_TYPE_F16) {
        int64_t * output_cast_ne = dst->ne;
        माप_प्रकार    output_cast_nb[GGML_MAX_DIMS];
        output_cast_nb[0] = परिमाण(uint16_t);
        लागि (पूर्णाङ्क i = 1; i < GGML_MAX_DIMS; i++) {
            output_cast_nb[i] = output_cast_nb[i - 1] * output_cast_ne[i - 1];
        }

        acl_tensor_ptr acl_output_tensor = ggml_cann_create_tensor(output_buffer, ACL_FLOAT16, output_elem_size,
                                                                   output_cast_ne, output_cast_nb, GGML_MAX_DIMS);
        acl_tensor_ptr acl_dst_tensor    = ggml_cann_create_tensor(dst);
        aclnn_cast(ctx, acl_output_tensor.get(), acl_dst_tensor.get(), ggml_cann_type_mapping(dst->type));
    }
}

शून्य ggml_cann_mul_mat(ggml_backend_cann_context & ctx, ggml_tensor * dst) {
    अचर गणना ggml_type type = dst->src[0]->type;
    विकल्प (type) {
        पक्ष GGML_TYPE_F32:
        पक्ष GGML_TYPE_F16:
#ifndef ASCEND_310P
        पक्ष GGML_TYPE_BF16:
#endif
            ggml_cann_mat_mul_fp(ctx, dst);
            रोक;
        पक्ष GGML_TYPE_Q4_0:
        पक्ष GGML_TYPE_Q8_0:
            ggml_cann_mul_mat_quant(ctx, dst, type);
            रोक;
        स्वभाव:
            GGML_ABORT("Unsupported type for mul_mat");
            रोक;
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
स्थिर शून्य aclnn_roll(ggml_backend_cann_context & ctx,
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
स्थिर शून्य aclnn_index_fill_tensor(ggml_backend_cann_context & ctx,
                                    aclTensor *                 acl_src,
                                    int64_t                     dim,
                                    int64_t *                   index,
                                    int64_t                     index_अङ्क,
                                    भिन्न                       value) {
    acl_int_array_ptr acl_index = ggml_cann_create_int_array(index, index_अङ्क);
    acl_scalar_ptr    acl_मान = ggml_cann_create_scalar(&value, aclDataType::ACL_FLOAT);
    GGML_CANN_CALL_ACLNN_OP(ctx, InplaceIndexFillTensor, acl_src, dim, acl_index.get(), acl_मान.get());
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
स्थिर शून्य aclnn_rope_cache_init(ggml_backend_cann_context & ctx,
                                  ggml_tensor *               dst,
                                  भिन्न *                     corr_dims,
                                  भिन्न                       ext_factor,
                                  भिन्न                       theta_scale,
                                  भिन्न                       freq_scale,
                                  भिन्न                       attn_factor,
                                  तार्किक                        is_neox,
                                  पूर्णाङ्क                         sections[4],
                                  तार्किक                        mrope_used,
                                  तार्किक                        is_imrope,
                                  तार्किक                        indep_sects,
                                  int64_t                     rope_dims) {
    ggml_tensor * src1 = dst->src[1];  // position
    ggml_tensor * src2 = dst->src[2];  // freq_factors

    int64_t theta_scale_length = rope_dims / 2;
    int64_t position_length    = dst->ne[2];

    // TODO: check theta_scale_length and position_length.
    यदि (src2 == nullptr && ctx.rope_cache.cached &&
        ctx.rope_cache.equal(theta_scale_length, position_length, ext_factor, theta_scale, freq_scale, attn_factor,
                             is_neox, indep_sects, mrope_used, is_imrope, sections)) {
        // use cache.
        फर्काऊ;
    }

    // Step0: calculate tensor shape.
    int64_t theta_scale_ne[] = { theta_scale_length, 1, 1, 1 };
    माप_प्रकार  theta_scale_nb[] = { परिमाण(भिन्न), theta_scale_length * परिमाण(भिन्न), theta_scale_length * परिमाण(भिन्न),
                                 theta_scale_length * परिमाण(भिन्न) };

    GGML_ASSERT(src1->type == GGML_TYPE_I32);
    int64_t position_ne[] = { 1, 1, position_length, 1 };
    माप_प्रकार  position_nb[] = { परिमाण(int32_t), परिमाण(int32_t), परिमाण(int32_t), परिमाण(int32_t) * position_length };

    int64_t cache_ne[] = { theta_scale_length, 1, position_length, 1 };
    माप_प्रकार  cache_nb[GGML_MAX_DIMS];
    cache_nb[0] = परिमाण(भिन्न);
    लागि (पूर्णाङ्क i = 1; i < GGML_MAX_DIMS; i++) {
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
    तार्किक           theta_scale_updated = असत्य;
    यदि (ctx.rope_cache.theta_scale_length != theta_scale_length || ctx.rope_cache.theta_scale != theta_scale ||
        ctx.rope_cache.indep_sects != indep_sects) {
        theta_scale_updated = सत्य;
        यदि (ctx.rope_cache.theta_scale_exp_host != nullptr) {
            मुक्त(ctx.rope_cache.theta_scale_exp_host);
        }
        ctx.rope_cache.theta_scale_exp_host = (भिन्न *) दो_स्मृति(theta_scale_length * परिमाण(भिन्न));
        GGML_ASSERT(ctx.rope_cache.theta_scale_exp_host != nullptr);
        यदि (!indep_sects) {
            ctx.rope_cache.theta_scale_exp_host[0] = 1;
            लागि (पूर्णाङ्क i = 1; i < theta_scale_length; i++) {
                ctx.rope_cache.theta_scale_exp_host[i] = ctx.rope_cache.theta_scale_exp_host[i - 1] * theta_scale;
            }
        } अन्यथा {
            पूर्णाङ्क sect_dims = sections[0] + sections[1] + sections[2] + sections[3];
            पूर्णाङ्क sec_w     = sections[1] + sections[0];
            पूर्णाङ्क sec_e     = sections[2] + sec_w;

            ctx.rope_cache.theta_scale_exp_host[0] = 1;
            लागि (पूर्णाङ्क i = 1; i < theta_scale_length; i++) {
                पूर्णाङ्क sector = i % sect_dims;
                यदि (sector == 0 || sector == sections[0] || sector == sec_w || sector == sec_e) {
                    ctx.rope_cache.theta_scale_exp_host[i] = 1;
                    जारी;
                }
                ctx.rope_cache.theta_scale_exp_host[i] = ctx.rope_cache.theta_scale_exp_host[i - 1] * theta_scale;
            }
        }

        यदि (ctx.rope_cache.theta_scale_cache != nullptr) {
            ACL_CHECK(aclrtFree(ctx.rope_cache.theta_scale_cache));
        }
        ACL_CHECK(aclrtMalloc(&ctx.rope_cache.theta_scale_cache, theta_scale_length * परिमाण(भिन्न),
                              ACL_MEM_MALLOC_HUGE_FIRST));

        ACL_CHECK(aclrtMemcpyAsync(ctx.rope_cache.theta_scale_cache, theta_scale_length * परिमाण(भिन्न),
                                   ctx.rope_cache.theta_scale_exp_host, theta_scale_length * परिमाण(भिन्न),
                                   ACL_MEMCPY_HOST_TO_DEVICE, ctx.stream()));
    }
    acl_theta_scale_tensor = ggml_cann_create_tensor(ctx.rope_cache.theta_scale_cache, ACL_FLOAT, परिमाण(भिन्न),
                                                     theta_scale_ne, theta_scale_nb, 1);

    // Step1.2: prepare rope_yarn_ramp, if this part updated, should update theta_scale_tensor.
    // TODO: acl_yarn_ramp_tensor use rope cache.
    तार्किक           yarn_ramp_tensor_updated = असत्य;
    acl_tensor_ptr acl_yarn_ramp_tensor;
    यदि (ext_factor != 0 && (theta_scale_updated || ctx.rope_cache.theta_scale_length != theta_scale_length ||
                            ctx.rope_cache.freq_scale != freq_scale)) {
        yarn_ramp_tensor_updated = सत्य;
        यदि (ctx.rope_cache.yarn_ramp_cache != nullptr) {
            ACL_CHECK(aclrtFree(ctx.rope_cache.yarn_ramp_cache));
        }
        ACL_CHECK(aclrtMalloc(&ctx.rope_cache.yarn_ramp_cache, theta_scale_length * परिमाण(भिन्न),
                              ACL_MEM_MALLOC_HUGE_FIRST));
        // -rope_yarn_ramp
        // const float y = (i0 / 2 - low) / MAX(0.001f, high - low);
        // return MIN(1, MAX(0, y)) - 1;
        acl_yarn_ramp_tensor      = ggml_cann_create_tensor(ctx.rope_cache.yarn_ramp_cache, ACL_FLOAT, परिमाण(भिन्न),
                                                            theta_scale_ne, theta_scale_nb, 1);
        भिन्न          zero_मान = 0, one_मान = 1;
        भिन्न          denom_safe_मान = MAX(0.001f, corr_dims[1] - corr_dims[0]);
        acl_scalar_ptr तल्लो              = ggml_cann_create_scalar(&corr_dims[0], aclDataType::ACL_FLOAT);
        acl_scalar_ptr zero             = ggml_cann_create_scalar(&zero_मान, aclDataType::ACL_FLOAT);
        acl_scalar_ptr one              = ggml_cann_create_scalar(&one_मान, aclDataType::ACL_FLOAT);
        acl_scalar_ptr denom_safe       = ggml_cann_create_scalar(&denom_safe_मान, aclDataType::ACL_FLOAT);
        acl_scalar_ptr ext_factor_sc    = ggml_cann_create_scalar(&ext_factor, aclDataType::ACL_FLOAT);

        aclnn_arange(ctx, acl_yarn_ramp_tensor.get(), 0, theta_scale_length, 1, theta_scale_length);
        GGML_CANN_CALL_ACLNN_OP(ctx, InplaceSubs, acl_yarn_ramp_tensor.get(), तल्लो.get(), one.get());
        GGML_CANN_CALL_ACLNN_OP(ctx, InplaceDivs, acl_yarn_ramp_tensor.get(), denom_safe.get());
        GGML_CANN_CALL_ACLNN_OP(ctx, InplaceThreshold, acl_yarn_ramp_tensor.get(), zero.get(), zero.get());
        GGML_CANN_CALL_ACLNN_OP(ctx, InplaceClampMax, acl_yarn_ramp_tensor.get(), one.get());
        GGML_CANN_CALL_ACLNN_OP(ctx, InplaceSubs, acl_yarn_ramp_tensor.get(), one.get(), one.get());
        GGML_CANN_CALL_ACLNN_OP(ctx, InplaceMuls, acl_yarn_ramp_tensor.get(), ext_factor_sc.get());

        // theta_interp = freq_scale * theta_extrap;
        // theta = theta_interp * (1 - ramp_mix) + theta_extrap * ramp_mix;
        // theta = freq_scale * theta_extrap * (1 - ramp_mix) + theta_extrap * ramp_mix;
        // theta = freq_scale * theta_extrap - freq_scale * theta_extrap * ramp_mix + theta_extrap * ramp_mix;
        // theta = theta_extrap * (freq_scale - freq_scale * ramp_mix + ramp_mix);
        //
        // we cache (freq_scale - freq_scale * ramp_mix + ramp_mix), Considering that the rope_yarn_ramp here is the inverse
        // cache freq_scale + (freq_scale - 1) * ramp_mix
        भिन्न          freq_scale_1    = freq_scale - 1;
        acl_scalar_ptr freq_scale_sc   = ggml_cann_create_scalar(&freq_scale, aclDataType::ACL_FLOAT);
        acl_scalar_ptr freq_scale_1_sc = ggml_cann_create_scalar(&freq_scale_1, aclDataType::ACL_FLOAT);
        GGML_CANN_CALL_ACLNN_OP(ctx, InplaceMuls, acl_yarn_ramp_tensor.get(), freq_scale_1_sc.get());
        GGML_CANN_CALL_ACLNN_OP(ctx, InplaceAdds, acl_yarn_ramp_tensor.get(), freq_scale_sc.get(), one.get());
    } अन्यथा {
        acl_yarn_ramp_tensor = ggml_cann_create_tensor(ctx.rope_cache.yarn_ramp_cache, ACL_FLOAT, परिमाण(भिन्न),
                                                       theta_scale_ne, theta_scale_nb, 1);
    }
    // Step 1.3: update theta_scale_tensor according to ext_factor or freq_scale.
    यदि (ext_factor != 0) {
        यदि (theta_scale_updated || yarn_ramp_tensor_updated) {
            theta_scale_updated = सत्य;
            aclnn_mul(ctx, acl_theta_scale_tensor.get(), acl_yarn_ramp_tensor.get());
        }
    } अन्यथा {
        यदि (freq_scale != 1 && (ctx.rope_cache.freq_scale != freq_scale || theta_scale_updated)) {
            theta_scale_updated = सत्य;
            aclnn_muls(ctx, acl_theta_scale_tensor.get(), freq_scale, nullptr, सत्य);
        }
    }

    // Nothing changed, use cache.
    यदि (!theta_scale_updated) {
        acl_theta_scale_tensor = ggml_cann_create_tensor(ctx.rope_cache.theta_scale_cache, ACL_FLOAT, परिमाण(भिन्न),
                                                         theta_scale_ne, theta_scale_nb, GGML_MAX_DIMS);
    }

    // Step 1.4: prepare select index if mrope
    acl_tensor_ptr position_select_index_tensor;
    यदि (mrope_used) {
        यदि (ctx.rope_cache.sections[0] != sections[0] || ctx.rope_cache.sections[1] != sections[1] ||
            ctx.rope_cache.sections[2] != sections[2] || ctx.rope_cache.sections[3] != sections[3] ||
            ctx.rope_cache.theta_scale_length != theta_scale_length || ctx.rope_cache.is_imrope != is_imrope) {
            यदि (ctx.rope_cache.position_select_index_host != nullptr) {
                मुक्त(ctx.rope_cache.position_select_index_host);
            }
            ctx.rope_cache.position_select_index_host = (पूर्णाङ्क *) दो_स्मृति(theta_scale_length * परिमाण(पूर्णाङ्क));
            GGML_ASSERT(ctx.rope_cache.position_select_index_host != nullptr);
            पूर्णाङ्क sect_dims = sections[0] + sections[1] + sections[2] + sections[3];
            पूर्णाङ्क sec_w     = sections[1] + sections[0];
            पूर्णाङ्क sec_e     = sections[2] + sec_w;
            // t,h,w,e
            लागि (पूर्णाङ्क i = 0; i < theta_scale_length; i++) {
                पूर्णाङ्क sector = i % sect_dims;

                यदि (is_imrope) {  // qwen3vl apply interleaved mrope
                    यदि (sector % 3 == 1 && sector < 3 * sections[1]) {
                        ctx.rope_cache.position_select_index_host[i] = 1;
                    } अन्यथा यदि (sector % 3 == 2 && sector < 3 * sections[2]) {
                        ctx.rope_cache.position_select_index_host[i] = 2;
                    } अन्यथा यदि (sector % 3 == 0 && sector < 3 * sections[0]) {
                        ctx.rope_cache.position_select_index_host[i] = 0;
                    } अन्यथा {
                        ctx.rope_cache.position_select_index_host[i] = 3;
                    }
                } अन्यथा {
                    यदि (sector >= sections[0] && sector < sec_w) {
                        ctx.rope_cache.position_select_index_host[i] = 1;
                    } अन्यथा यदि (sector >= sec_w && sector < sec_e) {
                        ctx.rope_cache.position_select_index_host[i] = 2;
                    } अन्यथा यदि (sector >= sec_e) {
                        ctx.rope_cache.position_select_index_host[i] = 3;
                    } अन्यथा {
                        ctx.rope_cache.position_select_index_host[i] = 0;
                    }
                }
            }

            यदि (ctx.rope_cache.position_select_index != nullptr) {
                ACL_CHECK(aclrtFree(ctx.rope_cache.position_select_index));
            }
            ACL_CHECK(aclrtMalloc(&ctx.rope_cache.position_select_index, theta_scale_length * परिमाण(पूर्णाङ्क),
                                  ACL_MEM_MALLOC_HUGE_FIRST));

            ACL_CHECK(aclrtMemcpyAsync(ctx.rope_cache.position_select_index, theta_scale_length * परिमाण(पूर्णाङ्क),
                                       ctx.rope_cache.position_select_index_host, theta_scale_length * परिमाण(पूर्णाङ्क),
                                       ACL_MEMCPY_HOST_TO_DEVICE, ctx.stream()));
        }

        position_select_index_tensor = ggml_cann_create_tensor(ctx.rope_cache.position_select_index, ACL_INT32,
                                                               परिमाण(पूर्णाङ्क), theta_scale_ne, theta_scale_nb, 1);
    }

    // Step2: divide by freq_factors
    ggml_cann_pool_alloc freq_fac_res_allocator(ctx.pool());
    यदि (src2) {
        freq_fac_res_allocator.alloc(theta_scale_length * परिमाण(भिन्न));
        शून्य *         freq_fac_res_ptr = freq_fac_res_allocator.get();
        acl_tensor_ptr acl_freq_factors_tensor =
            ggml_cann_create_tensor(src2->data, ggml_cann_type_mapping(src2->type), ggml_type_size(src2->type),
                                    theta_scale_ne, theta_scale_nb, GGML_MAX_DIMS);
        acl_tensor_ptr acl_freq_fac_res_tensor = ggml_cann_create_tensor(freq_fac_res_ptr, ACL_FLOAT, परिमाण(भिन्न),
                                                                         theta_scale_ne, theta_scale_nb, GGML_MAX_DIMS);
        aclnn_div(ctx, acl_theta_scale_tensor.get(), acl_freq_factors_tensor.get(), acl_freq_fac_res_tensor.get());
        std::swap(acl_theta_scale_tensor, acl_freq_fac_res_tensor);
    }

    // Step3: prepare position_tensor
    acl_tensor_ptr       acl_position_tensor;
    ggml_cann_pool_alloc mrope_position_acllocator(ctx.pool());
    यदि (mrope_used) {
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

        int64_t        mrope_position_ne[] = { position_length, 4 };
        माप_प्रकार         mrope_position_nb[] = { परिमाण(पूर्णाङ्क), position_length * परिमाण(पूर्णाङ्क) };
        acl_tensor_ptr mrope_position =
            ggml_cann_create_tensor(src1->data, ggml_cann_type_mapping(src1->type), ggml_type_size(src1->type),
                                    mrope_position_ne, mrope_position_nb, 2);

        // selected position tensor's shape is a transpose of cache tensor.
        int64_t selected_position_ne[] = { position_length, theta_scale_length };
        माप_प्रकार  selected_position_nb[] = { परिमाण(भिन्न), position_length * परिमाण(भिन्न) };
        mrope_position_acllocator.alloc(theta_scale_length * position_length * परिमाण(भिन्न));
        शून्य * mrope_position_buffer = mrope_position_acllocator.get();
        acl_position_tensor =
            ggml_cann_create_tensor(mrope_position_buffer, ggml_cann_type_mapping(src1->type),
                                    ggml_type_size(src1->type), selected_position_ne, selected_position_nb, 2);
        GGML_CANN_CALL_ACLNN_OP(ctx, IndexSelect, mrope_position.get(), 0, position_select_index_tensor.get(),
                                acl_position_tensor.get());

        // transpose
        int64_t transposed_ne[] = { position_length, 1, theta_scale_length, 1 };
        माप_प्रकार  transposed_nb[GGML_MAX_DIMS];
        transposed_nb[0] = परिमाण(भिन्न);
        लागि (पूर्णाङ्क i = 1; i < GGML_MAX_DIMS; i++) {
            transposed_nb[i] = transposed_nb[i - 1] * transposed_ne[i - 1];
        }

        std::swap(transposed_ne[0], transposed_ne[2]);
        std::swap(transposed_nb[0], transposed_nb[2]);

        acl_position_tensor =
            ggml_cann_create_tensor(mrope_position_buffer, ggml_cann_type_mapping(src1->type),
                                    ggml_type_size(src1->type), transposed_ne, transposed_nb, GGML_MAX_DIMS);

    } अन्यथा {
        // auto bcast.
        acl_position_tensor =
            ggml_cann_create_tensor(src1->data, ggml_cann_type_mapping(src1->type), ggml_type_size(src1->type),
                                    position_ne, position_nb, GGML_MAX_DIMS);
    }

    // Step4: multiply by the position
    int64_t              theta_length = theta_scale_length * position_length;
    ggml_cann_pool_alloc theta_allocator(ctx.pool(), theta_length * परिमाण(भिन्न));
    शून्य *               theta_buffer = theta_allocator.get();

    acl_tensor_ptr acl_theta_tensor =
        ggml_cann_create_tensor(theta_buffer, ACL_FLOAT, परिमाण(भिन्न), cache_ne, cache_nb, GGML_MAX_DIMS);
    aclnn_mul(ctx, acl_position_tensor.get(), acl_theta_scale_tensor.get(), acl_theta_tensor.get());

    // Step5: calculate sin cos.
    // init sin_repeat && cos_repeat, only to accelerate first layer on each device
    यदि (position_length > ctx.rope_cache.position_length) {
        ctx.rope_cache.position_length = position_length;
        यदि (ctx.rope_cache.sin_cache != nullptr) {
            ACL_CHECK(aclrtFree(ctx.rope_cache.sin_cache));
        }
        यदि (ctx.rope_cache.cos_cache != nullptr) {
            ACL_CHECK(aclrtFree(ctx.rope_cache.cos_cache));
        }
        int64_t repeat_theta_length = theta_scale_length * position_length * 2;
        ACL_CHECK(
            aclrtMalloc(&ctx.rope_cache.sin_cache, repeat_theta_length * परिमाण(भिन्न), ACL_MEM_MALLOC_HUGE_FIRST));
        ACL_CHECK(
            aclrtMalloc(&ctx.rope_cache.cos_cache, repeat_theta_length * परिमाण(भिन्न), ACL_MEM_MALLOC_HUGE_FIRST));
    }

    // sin/cos
    ggml_cann_pool_alloc sin_allocator(ctx.pool(), theta_length * परिमाण(भिन्न));
    शून्य *               sin_buffer = sin_allocator.get();
    acl_tensor_ptr       acl_sin_tensor =
        ggml_cann_create_tensor(sin_buffer, ACL_FLOAT, परिमाण(भिन्न), cache_ne, cache_nb, GGML_MAX_DIMS, ACL_FORMAT_ND);
    aclnn_sin(ctx, acl_theta_tensor.get(), acl_sin_tensor.get());

    ggml_cann_pool_alloc cos_allocator(ctx.pool(), theta_length * परिमाण(भिन्न));
    शून्य *               cos_buffer = cos_allocator.get();
    acl_tensor_ptr       acl_cos_tensor =
        ggml_cann_create_tensor(cos_buffer, ACL_FLOAT, परिमाण(भिन्न), cache_ne, cache_nb, GGML_MAX_DIMS, ACL_FORMAT_ND);
    aclnn_cos(ctx, acl_theta_tensor.get(), acl_cos_tensor.get());

    यदि (ext_factor != 0) {
        attn_factor *= 1.0f + 0.1f * logf(1.0f / freq_scale);
    }

    // Step 5: multiply by attn_factor
    यदि (attn_factor != 1) {
        aclnn_muls(ctx, acl_sin_tensor.get(), attn_factor, nullptr, सत्य);
        aclnn_muls(ctx, acl_cos_tensor.get(), attn_factor, nullptr, सत्य);
    }

    int64_t sin_reshape_ne[4] = { rope_dims, 1, dst->ne[2], 1 };
    माप_प्रकार  sin_reshape_nb[GGML_MAX_DIMS];
    sin_reshape_nb[0] = परिमाण(भिन्न);
    लागि (पूर्णाङ्क i = 1; i < GGML_MAX_DIMS; i++) {
        sin_reshape_nb[i] = sin_reshape_nb[i - 1] * sin_reshape_ne[i - 1];
    }
    acl_tensor_ptr acl_sin_repeat_tensor = ggml_cann_create_tensor(ctx.rope_cache.sin_cache, ACL_FLOAT, परिमाण(भिन्न),
                                                                   sin_reshape_ne, sin_reshape_nb, GGML_MAX_DIMS);
    acl_tensor_ptr acl_cos_repeat_tensor = ggml_cann_create_tensor(ctx.rope_cache.cos_cache, ACL_FLOAT, परिमाण(भिन्न),
                                                                   sin_reshape_ne, sin_reshape_nb, GGML_MAX_DIMS);

    // Step 6: repeat
    यदि (is_neox) {
        // [sinθ1, sinθ1, sinθ2, sinθ2, ..., sinθn, sinθn]
        int64_t repeatsArray[] = { 1, 1, 1, 2 };
        aclnn_repeat(ctx, acl_sin_tensor.get(), acl_sin_repeat_tensor.get(), repeatsArray);
        aclnn_repeat(ctx, acl_cos_tensor.get(), acl_cos_repeat_tensor.get(), repeatsArray);
    } अन्यथा {
        int64_t अङ्क_repeats = 2;
        int64_t dim         = 3;
        int64_t output_size = theta_scale_length * अङ्क_repeats;
        // [sinθ1, sinθ2, ..., sinθn, sinθ1, sinθ2, ..., sinθn]
        aclnn_repeat_interleave(ctx, acl_sin_tensor.get(), acl_sin_repeat_tensor.get(), dim, अङ्क_repeats, output_size);
        aclnn_repeat_interleave(ctx, acl_cos_tensor.get(), acl_cos_repeat_tensor.get(), dim, अङ्क_repeats, output_size);
    }

    // Update cached value.
    ctx.rope_cache.cached = सत्य;
    ctx.rope_cache.set(theta_scale_length, position_length, ext_factor, theta_scale, freq_scale, attn_factor, is_neox,
                       indep_sects, mrope_used, is_imrope, sections);
}

#ifdef __cplusplus
बाह्य "C" {
#endif
aclnnStatus aclnnRotaryPositionEmbeddingGetWorkspaceSize(अचर aclTensor * x,
                                                         अचर aclTensor * cos,
                                                         अचर aclTensor * sin,
                                                         int64_t           mode,
                                                         अचर aclTensor * yOut,
                                                         uint64_t *        workspaceSize,
                                                         aclOpExecutor **  executor);
aclnnStatus aclnnRotaryPositionEmbedding(शून्य *          workspace,
                                         uint64_t        workspaceSize,
                                         aclOpExecutor * executor,
                                         aclrtStream     stream);
#ifdef __cplusplus
}
#endif

शून्य ggml_cann_rope(ggml_backend_cann_context & ctx, ggml_tensor * dst) {
    ggml_tensor * src0 = dst->src[0];  //आगत

    // param
    भिन्न     freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow;
    पूर्णाङ्क       sections[4];
    // const int n_past     = ((int32_t *) dst->op_params)[0];
    अचर पूर्णाङ्क n_dims     = ((int32_t *) dst->op_params)[1];
    अचर पूर्णाङ्क mode       = ((int32_t *) dst->op_params)[2];
    // const int n_ctx      = ((int32_t *) dst->op_params)[3];
    अचर पूर्णाङ्क n_ctx_orig = ((int32_t *) dst->op_params)[4];

    GGML_TENSOR_UNARY_OP_LOCALS

    स_नकल(&freq_base, (int32_t *) dst->op_params + 5, परिमाण(भिन्न));
    स_नकल(&freq_scale, (int32_t *) dst->op_params + 6, परिमाण(भिन्न));
    स_नकल(&ext_factor, (int32_t *) dst->op_params + 7, परिमाण(भिन्न));
    स_नकल(&attn_factor, (int32_t *) dst->op_params + 8, परिमाण(भिन्न));
    स_नकल(&beta_fast, (int32_t *) dst->op_params + 9, परिमाण(भिन्न));
    स_नकल(&beta_slow, (int32_t *) dst->op_params + 10, परिमाण(भिन्न));
    स_नकल(&sections, (int32_t *) dst->op_params + 11, परिमाण(पूर्णाङ्क) * 4);

    GGML_ASSERT(n_dims % 2 == 0);
    GGML_ASSERT(n_dims <= ne00);

    अचर भिन्न theta_scale = powf(freq_base, -2.0f / n_dims);

    भिन्न corr_dims[2];
    ggml_rope_yarn_corr_dims(n_dims, n_ctx_orig, freq_base, beta_fast, beta_slow, corr_dims);

    तार्किक       is_neox    = mode & GGML_ROPE_TYPE_NEOX;
    अचर तार्किक is_imrope  = mode == GGML_ROPE_TYPE_IMROPE;  // qwen3vl apply interleaved mrope
    // mrope_used means the GGML_ROPE_TYPE_MROPE bit is set.
    // Note: this bit is also set for imrope and some vision modes,
    // so mrope_used does NOT exclusively indicate pure mrope.
    अचर तार्किक mrope_used = mode & GGML_ROPE_TYPE_MROPE;
    अचर तार्किक is_vision  = mode == GGML_ROPE_TYPE_VISION;

    यदि (mrope_used) {
        GGML_ASSERT(sections[0] > 0 || sections[1] > 0 || sections[2] > 0);
    }

    यदि (is_vision) {
        GGML_ASSERT(n_dims == ne0 / 2);
    }

    यदि (is_imrope || mrope_used) {
        is_neox = सत्य;
    }

    int64_t rope_dims = n_dims;

    //Our current RotaryPositionEmbedding does not support the VISION mode,
    //but essentially it only modifies theta_base in mrope,
    //then repeats it at the end in the same way as is_neox.
    //In fact, RoPE is still applied across all dimensions.
    यदि (is_vision) {
        rope_dims = src0->ne[0];
    }
    int64_t tail_dims = ne00 - rope_dims;
    तार्किक    has_tail  = tail_dims > 0;

    // init ctx.rope_cos/rope_sin cache
    aclnn_rope_cache_init(ctx, dst, corr_dims, ext_factor, theta_scale, freq_scale, attn_factor, is_neox, sections,
                          mrope_used, is_imrope, is_vision, rope_dims);

    // Cache is generated with ne00 dimensions, so we use ne00 for reshape
    int64_t sin_reshape_ne[4] = { rope_dims, 1, ne02, 1 };
    माप_प्रकार  sin_reshape_nb[GGML_MAX_DIMS];
    sin_reshape_nb[0] = परिमाण(भिन्न);
    लागि (पूर्णाङ्क i = 1; i < GGML_MAX_DIMS; i++) {
        sin_reshape_nb[i] = sin_reshape_nb[i - 1] * sin_reshape_ne[i - 1];
    }
    acl_tensor_ptr acl_sin_reshape_tensor = ggml_cann_create_tensor(ctx.rope_cache.sin_cache, ACL_FLOAT, परिमाण(भिन्न),
                                                                    sin_reshape_ne, sin_reshape_nb, GGML_MAX_DIMS);
    acl_tensor_ptr acl_cos_reshape_tensor = ggml_cann_create_tensor(ctx.rope_cache.cos_cache, ACL_FLOAT, परिमाण(भिन्न),
                                                                    sin_reshape_ne, sin_reshape_nb, GGML_MAX_DIMS);

    acl_tensor_ptr acl_src = ggml_cann_create_tensor(src0);
    acl_tensor_ptr acl_dst = ggml_cann_create_tensor(dst);
#ifdef ASCEND_310P
    // Special ROPE operation for 310P

    // roll input
    शून्य *               आगत_roll_buffer;
    acl_tensor_ptr       acl_minus_one_tensor;
    शून्य *               minus_one_scale_buffer = nullptr;
    ggml_cann_pool_alloc roll_allocator(ctx.pool(), ggml_nbytes(src0));
    ggml_cann_pool_alloc minus_one_scale_allocator(ctx.pool(), परिमाण(भिन्न) * src0->ne[0]);
    यदि (!is_neox) {
        // roll input: [q0,q1,q2,q3,...] -> [q1,q0,q3,q2,...]
        आगत_roll_buffer        = roll_allocator.get();
        int64_t आगत_roll_ne[4] = { 2, src0->ne[1] * (src0->ne[0] / 2), src0->ne[2], src0->ne[3] };
        माप_प्रकार  आगत_roll_nb[GGML_MAX_DIMS];
        आगत_roll_nb[0] = ggml_type_size(src0->type);
        लागि (पूर्णाङ्क i = 1; i < GGML_MAX_DIMS; i++) {
            आगत_roll_nb[i] = आगत_roll_nb[i - 1] * आगत_roll_ne[i - 1];
        }
        acl_tensor_ptr acl_आगत_roll_tensor =
            ggml_cann_create_tensor(आगत_roll_buffer, ggml_cann_type_mapping(src0->type), ggml_type_size(src0->type),
                                    आगत_roll_ne, आगत_roll_nb, GGML_MAX_DIMS);
        acl_tensor_ptr acl_आगत_tensor =
            ggml_cann_create_tensor(src0->data, ggml_cann_type_mapping(src0->type), ggml_type_size(src0->type),
                                    आगत_roll_ne, आगत_roll_nb, GGML_MAX_DIMS);

        int64_t shifts[] = { 1 };
        int64_t dims[]   = { 3 };
        aclnn_roll(ctx, acl_आगत_tensor.get(), acl_आगत_roll_tensor.get(), shifts, dims);

        // init [-1, 1, -1, 1, ...]
        minus_one_scale_buffer = minus_one_scale_allocator.get();

        int64_t minus_one_ne[4] = { src0->ne[0], 1, 1, 1 };
        माप_प्रकार  minus_one_nb[GGML_MAX_DIMS];
        minus_one_nb[0] = परिमाण(भिन्न);
        लागि (पूर्णाङ्क i = 1; i < GGML_MAX_DIMS; i++) {
            minus_one_nb[i] = minus_one_nb[i - 1] * minus_one_ne[i - 1];
        }
        acl_minus_one_tensor = aclnn_मानहरू(ctx, minus_one_scale_buffer, परिमाण(भिन्न) * src0->ne[0], minus_one_ne,
                                            GGML_MAX_DIMS, ACL_FLOAT, परिमाण(भिन्न), 1);
        int64_t   dim        = 3;
        int64_t * index      = नूतन int64_t[src0->ne[0]];
        लागि (पूर्णाङ्क i = 0; i < src0->ne[0]; i++) {
            index[i] = i / 2 * 2;
        }
        int64_t index_अङ्क = src0->ne[0];
        भिन्न   value     = -1;
        aclnn_index_fill_tensor(ctx, acl_minus_one_tensor.get(), dim, index, index_अङ्क, value);
    } अन्यथा {
        // roll input: [q0,q1,q2,...] ->
        // [q_half,q_half+1,...,q_end,q0,q1,...q_half-1]
        आगत_roll_buffer = roll_allocator.get();
        acl_tensor_ptr acl_आगत_roll_tensor =
            ggml_cann_create_tensor(आगत_roll_buffer, ggml_cann_type_mapping(src0->type), ggml_type_size(src0->type),
                                    src0->ne, src0->nb, GGML_MAX_DIMS);
        acl_tensor_ptr acl_आगत_tensor = ggml_cann_create_tensor(src0);

        int64_t shifts[] = { src0->ne[0] / 2 };
        int64_t dims[]   = { 3 };
        aclnn_roll(ctx, acl_आगत_tensor.get(), acl_आगत_roll_tensor.get(), shifts, dims);

        // init [-1, -1, -1, 1, 1，1，...]
        minus_one_scale_buffer  = minus_one_scale_allocator.get();
        int64_t minus_one_ne[4] = { src0->ne[0], 1, 1, 1 };
        माप_प्रकार  minus_one_nb[GGML_MAX_DIMS];
        minus_one_nb[0] = परिमाण(भिन्न);
        लागि (पूर्णाङ्क i = 1; i < GGML_MAX_DIMS; i++) {
            minus_one_nb[i] = minus_one_nb[i - 1] * minus_one_ne[i - 1];
        }
        acl_minus_one_tensor     = aclnn_मानहरू(ctx, minus_one_scale_buffer, परिमाण(भिन्न) * src0->ne[0], minus_one_ne,
                                                GGML_MAX_DIMS, ACL_FLOAT, परिमाण(भिन्न), 1);
        // -1 * first half
        int64_t first_half_ne[4] = { src0->ne[0] / 2, 1, 1, 1 };
        माप_प्रकार  first_half_nb[GGML_MAX_DIMS];
        first_half_nb[0] = परिमाण(भिन्न);
        लागि (पूर्णाङ्क i = 1; i < GGML_MAX_DIMS; i++) {
            first_half_nb[i] = first_half_nb[i - 1] * first_half_ne[i - 1];
        }
        acl_tensor_ptr acl_first_half_tensor = ggml_cann_create_tensor(minus_one_scale_buffer, ACL_FLOAT, परिमाण(भिन्न),
                                                                       first_half_ne, first_half_nb, GGML_MAX_DIMS);
        तार्किक           inplace               = सत्य;
        भिन्न          scale                 = -1;
        aclnn_muls(ctx, acl_first_half_tensor.get(), scale, nullptr, inplace);
    }

    // TODO: n_dims < ne0
    GGML_ASSERT(n_dims == src0->ne[0]);

    // input * scale
    ggml_cann_pool_alloc roll_mul_scale_allocator(ctx.pool(), ggml_nbytes(src0));
    शून्य *               आगत_roll_mul_scale_buffer = roll_mul_scale_allocator.get();
    माप_प्रकार               आगत_nb[GGML_MAX_DIMS];
    आगत_nb[0] = ggml_type_size(src0->type);
    लागि (पूर्णाङ्क i = 1; i < GGML_MAX_DIMS; i++) {
        आगत_nb[i] = आगत_nb[i - 1] * src0->ne[i - 1];
    }
    acl_tensor_ptr acl_आगत_roll_mul_scale_tensor =
        ggml_cann_create_tensor(आगत_roll_mul_scale_buffer, ggml_cann_type_mapping(src0->type),
                                ggml_type_size(src0->type), src0->ne, आगत_nb, GGML_MAX_DIMS);
    acl_tensor_ptr acl_आगत_roll_reshape_tensor =
        ggml_cann_create_tensor(आगत_roll_buffer, ggml_cann_type_mapping(src0->type), ggml_type_size(src0->type),
                                src0->ne, आगत_nb, GGML_MAX_DIMS);

    aclnn_mul(ctx, acl_आगत_roll_reshape_tensor.get(), acl_minus_one_tensor.get(),
              acl_आगत_roll_mul_scale_tensor.get());

    // output
    शून्य * output_fp32_buffer;
    यदि (src0->type == GGML_TYPE_F32) {
        aclnn_mul(ctx, acl_src.get(), acl_cos_reshape_tensor.get());
        aclnn_mul(ctx, acl_आगत_roll_mul_scale_tensor.get(), acl_sin_reshape_tensor.get());
        aclnn_add(ctx, acl_src.get(), acl_आगत_roll_mul_scale_tensor.get(), acl_dst.get());
        // TODO: ne0 != n_dims in mode2
    } अन्यथा यदि (src0->type == GGML_TYPE_F16) {
        माप_प्रकार आगत_fp32_nb[GGML_MAX_DIMS];
        आगत_fp32_nb[0] = परिमाण(भिन्न);
        लागि (पूर्णाङ्क i = 1; i < GGML_MAX_DIMS; i++) {
            आगत_fp32_nb[i] = आगत_fp32_nb[i - 1] * dst->ne[i - 1];
        }
        ggml_cann_pool_alloc fp32_allocator1(ctx.pool(), ggml_nelements(dst) * परिमाण(भिन्न));
        शून्य *               आगत_fp32_buffer1 = fp32_allocator1.get();
        acl_tensor_ptr       आगत_fp32_tensor1 = ggml_cann_create_tensor(आगत_fp32_buffer1, ACL_FLOAT, परिमाण(भिन्न),
                                                                          dst->ne, आगत_fp32_nb, GGML_MAX_DIMS);
        ggml_cann_pool_alloc fp32_allocator2(ctx.pool(), ggml_nelements(dst) * परिमाण(भिन्न));
        शून्य *               आगत_fp32_buffer2 = fp32_allocator2.get();
        acl_tensor_ptr       आगत_fp32_tensor2 = ggml_cann_create_tensor(आगत_fp32_buffer2, ACL_FLOAT, परिमाण(भिन्न),
                                                                          dst->ne, आगत_fp32_nb, GGML_MAX_DIMS);

        ggml_cann_pool_alloc fp32_allocator(ctx.pool(), ggml_nelements(dst) * परिमाण(भिन्न));
        output_fp32_buffer                = fp32_allocator.get();
        acl_tensor_ptr output_fp32_tensor = ggml_cann_create_tensor(output_fp32_buffer, ACL_FLOAT, परिमाण(भिन्न),
                                                                    dst->ne, आगत_fp32_nb, GGML_MAX_DIMS);
        aclnn_mul(ctx, acl_src.get(), acl_cos_reshape_tensor.get(), आगत_fp32_tensor1.get());
        aclnn_mul(ctx, acl_आगत_roll_mul_scale_tensor.get(), acl_sin_reshape_tensor.get(), आगत_fp32_tensor2.get());
        aclnn_add(ctx, आगत_fp32_tensor1.get(), आगत_fp32_tensor2.get(), output_fp32_tensor.get());
        aclnn_cast(ctx, output_fp32_tensor.get(), acl_dst.get(), ACL_FLOAT16);
    }
    फर्काऊ;
#endif
    int64_t acl_mode = is_neox ? 0 : 1;

    // Pre-define head and tail dimensions for reuse
    int64_t head_ne[GGML_MAX_DIMS] = { rope_dims, ne01, ne02, ne03 };
    int64_t tail_ne[GGML_MAX_DIMS] = { tail_dims, ne01, ne02, ne03 };

    // Step 1: Prepare trans tensors for F16 type conversion to F32 if needed
    तार्किक                 src_dst_need_trans = असत्य;
    ggml_cann_pool_alloc src_trans_allocator(ctx.pool());
    ggml_cann_pool_alloc dst_trans_allocator(ctx.pool());
    acl_tensor_ptr       acl_src_trans_tensor;
    acl_tensor_ptr       acl_dst_trans_tensor;
    शून्य *               src_trans_buffer = nullptr;
    शून्य *               dst_trans_buffer = nullptr;
    माप_प्रकार               src_dst_trans_nb[GGML_MAX_DIMS];
    यदि (src0->type == GGML_TYPE_F16) {
        src_dst_need_trans = सत्य;
        src_trans_buffer   = src_trans_allocator.alloc(ggml_nelements(src0) * परिमाण(भिन्न));
        dst_trans_buffer   = dst_trans_allocator.alloc(ggml_nelements(dst) * परिमाण(भिन्न));

        src_dst_trans_nb[0] = परिमाण(भिन्न);
        लागि (पूर्णाङ्क i = 1; i < GGML_MAX_DIMS; i++) {
            src_dst_trans_nb[i] = src_dst_trans_nb[i - 1] * src0->ne[i - 1];
        }
        acl_src_trans_tensor = ggml_cann_create_tensor(src_trans_buffer, ACL_FLOAT, परिमाण(भिन्न), src0->ne,
                                                       src_dst_trans_nb, GGML_MAX_DIMS);
        acl_dst_trans_tensor = ggml_cann_create_tensor(dst_trans_buffer, ACL_FLOAT, परिमाण(भिन्न), dst->ne,
                                                       src_dst_trans_nb, GGML_MAX_DIMS);
        aclnn_cast(ctx, acl_src.get(), acl_src_trans_tensor.get(), ACL_FLOAT);
    }

    // Step 2: Prepare head tensors for tail splitting if needed
    acl_tensor_ptr acl_src_head;
    acl_tensor_ptr acl_dst_head;
    यदि (has_tail) {
        // Create head views for RotaryPositionEmbedding (only first rope_dims dimensions)
        // RotaryPositionEmbedding requires contiguous dst tensor, so we use a temporary buffer
        यदि (src_dst_need_trans) {
            // Use F32 trans tensor strides
            acl_src_head = ggml_cann_create_tensor((वर्ण *) src_trans_buffer, ACL_FLOAT, परिमाण(भिन्न), head_ne,
                                                   src_dst_trans_nb, GGML_MAX_DIMS);
        } अन्यथा {
            // Use original F32 tensor strides
            acl_src_head = ggml_cann_create_tensor((वर्ण *) src0->data, ACL_FLOAT, परिमाण(भिन्न), head_ne, src0->nb,
                                                   GGML_MAX_DIMS);
        }

        int64_t              head_elements = rope_dims * ne01 * ne02 * ne03;
        ggml_cann_pool_alloc dst_head_contiguous_allocator(ctx.pool(), head_elements * परिमाण(भिन्न));
        शून्य *               dst_head_contiguous_buffer = dst_head_contiguous_allocator.get();

        माप_प्रकार head_contiguous_nb[GGML_MAX_DIMS];
        head_contiguous_nb[0] = परिमाण(भिन्न);
        लागि (पूर्णाङ्क i = 1; i < GGML_MAX_DIMS; i++) {
            head_contiguous_nb[i] = head_contiguous_nb[i - 1] * head_ne[i - 1];
        }
        acl_dst_head = ggml_cann_create_tensor(dst_head_contiguous_buffer, ACL_FLOAT, परिमाण(भिन्न), head_ne,
                                               head_contiguous_nb, GGML_MAX_DIMS);
    }

    // Step 3: Execute RotaryPositionEmbedding
    यदि (has_tail) {
        // Rotate only the head portion (first rope_dims dimensions)
        GGML_CANN_CALL_ACLNN_OP(ctx, RotaryPositionEmbedding, acl_src_head.get(), acl_cos_reshape_tensor.get(),
                                acl_sin_reshape_tensor.get(), acl_mode, acl_dst_head.get());

        // Copy head result from contiguous buffer back to destination tensor
        यदि (src_dst_need_trans) {
            acl_tensor_ptr acl_dst_head_target = ggml_cann_create_tensor(
                (वर्ण *) dst_trans_buffer, ACL_FLOAT, परिमाण(भिन्न), head_ne, src_dst_trans_nb, GGML_MAX_DIMS);
            cann_copy(ctx, acl_dst_head.get(), acl_dst_head_target.get());
        } अन्यथा {
            acl_tensor_ptr acl_dst_head_target =
                ggml_cann_create_tensor((वर्ण *) dst->data, ACL_FLOAT, परिमाण(भिन्न), head_ne, dst->nb, GGML_MAX_DIMS);
            cann_copy(ctx, acl_dst_head.get(), acl_dst_head_target.get());
        }
    } अन्यथा यदि (src_dst_need_trans) {
        // Rotate full tensor (no tail), using trans tensors
        GGML_CANN_CALL_ACLNN_OP(ctx, RotaryPositionEmbedding, acl_src_trans_tensor.get(), acl_cos_reshape_tensor.get(),
                                acl_sin_reshape_tensor.get(), acl_mode, acl_dst_trans_tensor.get());
    } अन्यथा यदि (src0->data == dst->data && !ggml_is_contiguous(src0)) {
        // In-place on non-contiguous tensor: RotaryPositionEmbedding cannot safely
        // read and write the same non-contiguous buffer. Use contiguous temporaries.
        माप_प्रकार contiguous_nb[GGML_MAX_DIMS];
        contiguous_nb[0] = परिमाण(भिन्न);
        लागि (पूर्णाङ्क i = 1; i < GGML_MAX_DIMS; i++) {
            contiguous_nb[i] = contiguous_nb[i - 1] * src0->ne[i - 1];
        }
        int64_t              total_elements = ggml_nelements(src0);
        ggml_cann_pool_alloc inplace_src_alloc(ctx.pool(), total_elements * परिमाण(भिन्न));
        ggml_cann_pool_alloc inplace_dst_alloc(ctx.pool(), total_elements * परिमाण(भिन्न));

        acl_tensor_ptr acl_src_contig = ggml_cann_create_tensor(inplace_src_alloc.get(), ACL_FLOAT, परिमाण(भिन्न),
                                                                src0->ne, contiguous_nb, GGML_MAX_DIMS);
        acl_tensor_ptr acl_dst_contig = ggml_cann_create_tensor(inplace_dst_alloc.get(), ACL_FLOAT, परिमाण(भिन्न),
                                                                dst->ne, contiguous_nb, GGML_MAX_DIMS);

        cann_copy(ctx, acl_src.get(), acl_src_contig.get());
        GGML_CANN_CALL_ACLNN_OP(ctx, RotaryPositionEmbedding, acl_src_contig.get(), acl_cos_reshape_tensor.get(),
                                acl_sin_reshape_tensor.get(), acl_mode, acl_dst_contig.get());
        cann_copy(ctx, acl_dst_contig.get(), acl_dst.get());
    } अन्यथा {
        // Rotate full tensor (no tail), using original tensors
        GGML_CANN_CALL_ACLNN_OP(ctx, RotaryPositionEmbedding, acl_src.get(), acl_cos_reshape_tensor.get(),
                                acl_sin_reshape_tensor.get(), acl_mode, acl_dst.get());
    }

    // Step 4: Copy unrotated tail portion from source to destination
    यदि (has_tail) {
        माप_प्रकार src_tail_offset;
        माप_प्रकार dst_tail_offset;

        स्वतः copy_tail_device = [&](शून्य * src_ptr, शून्य * dst_ptr, aclDataType dtype, माप_प्रकार elem_size,
                                    माप_प्रकार * nb_src_arr, माप_प्रकार * nb_dst_arr) {
            acl_tensor_ptr acl_src_tail =
                ggml_cann_create_tensor(src_ptr, dtype, elem_size, tail_ne, nb_src_arr, GGML_MAX_DIMS);
            acl_tensor_ptr acl_dst_tail =
                ggml_cann_create_tensor(dst_ptr, dtype, elem_size, tail_ne, nb_dst_arr, GGML_MAX_DIMS);
            cann_copy(ctx, acl_src_tail.get(), acl_dst_tail.get());
        };

        यदि (src_dst_need_trans) {
            // Use F32 trans tensor strides and offsets
            src_tail_offset = rope_dims * src_dst_trans_nb[0];
            dst_tail_offset = rope_dims * src_dst_trans_nb[0];
            copy_tail_device((वर्ण *) src_trans_buffer + src_tail_offset, (वर्ण *) dst_trans_buffer + dst_tail_offset,
                             ACL_FLOAT, परिमाण(भिन्न), src_dst_trans_nb, src_dst_trans_nb);
        } अन्यथा {
            // Use original tensor strides and offsets
            src_tail_offset = rope_dims * nb00;
            dst_tail_offset = rope_dims * nb0;
            copy_tail_device((वर्ण *) src0->data + src_tail_offset, (वर्ण *) dst->data + dst_tail_offset,
                             ggml_cann_type_mapping(dst->type), ggml_element_size(dst), src0->nb, dst->nb);
        }
    }

    // Step 5: Cast back to F16 if needed
    यदि (src_dst_need_trans) {
        aclnn_cast(ctx, acl_dst_trans_tensor.get(), acl_dst.get(), ACL_FLOAT16);
    }
}

शून्य ggml_cann_rope_cache_preload(ggml_backend_cann_context & ctx, ggml_tensor * dst) {
    ggml_tensor * src0 = dst->src[0];

    भिन्न     freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow;
    पूर्णाङ्क       sections[4];
    अचर पूर्णाङ्क n_dims     = ((int32_t *) dst->op_params)[1];
    अचर पूर्णाङ्क mode       = ((int32_t *) dst->op_params)[2];
    अचर पूर्णाङ्क n_ctx_orig = ((int32_t *) dst->op_params)[4];

    GGML_TENSOR_UNARY_OP_LOCALS

    स_नकल(&freq_base, (int32_t *) dst->op_params + 5, परिमाण(भिन्न));
    स_नकल(&freq_scale, (int32_t *) dst->op_params + 6, परिमाण(भिन्न));
    स_नकल(&ext_factor, (int32_t *) dst->op_params + 7, परिमाण(भिन्न));
    स_नकल(&attn_factor, (int32_t *) dst->op_params + 8, परिमाण(भिन्न));
    स_नकल(&beta_fast, (int32_t *) dst->op_params + 9, परिमाण(भिन्न));
    स_नकल(&beta_slow, (int32_t *) dst->op_params + 10, परिमाण(भिन्न));
    स_नकल(&sections, (int32_t *) dst->op_params + 11, परिमाण(पूर्णाङ्क) * 4);

    अचर भिन्न theta_scale = powf(freq_base, -2.0f / n_dims);

    भिन्न corr_dims[2];
    ggml_rope_yarn_corr_dims(n_dims, n_ctx_orig, freq_base, beta_fast, beta_slow, corr_dims);

    तार्किक       is_neox    = mode & GGML_ROPE_TYPE_NEOX;
    अचर तार्किक is_imrope  = mode == GGML_ROPE_TYPE_IMROPE;
    अचर तार्किक mrope_used = mode & GGML_ROPE_TYPE_MROPE;
    अचर तार्किक is_vision  = mode == GGML_ROPE_TYPE_VISION;

    यदि (is_imrope || mrope_used) {
        is_neox = सत्य;
    }

    int64_t rope_dims = n_dims;
    यदि (is_vision) {
        rope_dims = src0->ne[0];
    }

    // Run the full cache init on the non-captured stream.  This performs all
    // host-to-device memcpy, aclrtMalloc/Free, and on-device computations
    // so that the memory pool is warmed up and cache metadata is populated.
    aclnn_rope_cache_init(ctx, dst, corr_dims, ext_factor, theta_scale, freq_scale, attn_factor, is_neox, sections,
                          mrope_used, is_imrope, is_vision, rope_dims);

    // Reset `cached` so that during graph capture the on-device computations
    // (sin/cos, position multiply, repeat, etc.) still execute and get recorded
    // into the captured graph.  The cache metadata (theta_scale_length,
    // theta_scale, sections, position_length, etc.) remains set, which causes
    // all host-to-device copy and malloc/free branches to be skipped.
    ctx.rope_cache.cached = असत्य;
}

शून्य ggml_cann_argmax(ggml_backend_cann_context & ctx, ggml_tensor * dst) {
    ggml_tensor * src0 = dst->src[0];

    acl_tensor_ptr acl_src = ggml_cann_create_tensor(src0);
    acl_tensor_ptr acl_dst = ggml_cann_create_tensor(dst, dst->ne, dst->nb, 3);

    GGML_CANN_CALL_ACLNN_OP(ctx, ArgMax, acl_src.get(), 3, असत्य, acl_dst.get());
}

शून्य ggml_cann_conv_transpose_1d(ggml_backend_cann_context & ctx, ggml_tensor * dst) {
    ggml_tensor * src0 = dst->src[0];
    ggml_tensor * src1 = dst->src[1];

    // stride
    int64_t s0 = ((अचर int32_t *) (dst->op_params))[0];

    acl_tensor_ptr acl_आगत  = ggml_cann_create_tensor(src1, src1->ne, src1->nb, 3, ACL_FORMAT_NCL);
    acl_tensor_ptr acl_weight = ggml_cann_create_tensor(src0, src0->ne, src0->nb, 3, ACL_FORMAT_NCL);
    acl_tensor_ptr acl_dst    = ggml_cann_create_tensor(dst, dst->ne, dst->nb, 3, ACL_FORMAT_NCL);

    // get base information of input and kernel
    int64_t आगत_len   = *(src1->ne);
    int64_t dst_len     = *(dst->ne);
    int64_t kernel_size = *(src0->ne);

    // set the max kernel size for each conv
    int64_t max_kernel_size = 255;

    // compute the partition of kernel
    int64_t part_अङ्क = 1;
    part_अङ्क         = (kernel_size + max_kernel_size - 1) / max_kernel_size;

    int64_t strideVal[1];
    strideVal[0]                    = s0;
    acl_int_array_ptr stride        = ggml_cann_create_int_array(strideVal, 1);
    int64_t           paddingVal[]  = { 0 };
    acl_int_array_ptr padding       = ggml_cann_create_int_array(paddingVal, 1);
    int64_t           dilationVal[] = { 1 };
    acl_int_array_ptr dilation      = ggml_cann_create_int_array(dilationVal, 1);
    तार्किक              transposed    = सत्य;
    int64_t           groups        = 1;
    int8_t            cubeMathType  = 0;

#ifdef ASCEND_310P
    cubeMathType = 1;
#endif

    स्वतः weight_type = ggml_cann_type_mapping(src0->type);
    स्वतः dst_type    = ggml_cann_type_mapping(dst->type);

    // slice the kernel to make each conv available
    int64_t slice_dim   = -1;
    int64_t slice_start = 0;
    int64_t slice_end   = max_kernel_size;
    int64_t slice_step  = 1;
    int64_t interval    = max_kernel_size;

    int64_t बायाँ_pad_len  = dilationVal[0] * (max_kernel_size - 1) + 1 - 2 * paddingVal[0];
    int64_t दायाँ_pad_len = 0;

    acl_scalar_ptr alpha      = nullptr;
    भिन्न          alpha_मान = 1.0;
    alpha                     = ggml_cann_create_scalar(&alpha_मान, aclDataType::ACL_FLOAT);

    // set zero to destination
    GGML_CANN_CALL_ACLNN_OP(ctx, InplaceZero, acl_dst.get());

    लागि (पूर्णाङ्क k = 0; k < part_अङ्क; k++) {
        // create part kernel tensor and slice from big kernel
        slice_start = max_kernel_size * k;
        यदि (k == part_अङ्क - 1) {
            slice_end = kernel_size;
            interval  = kernel_size - max_kernel_size * k;
        } अन्यथा {
            slice_end = max_kernel_size * (k + 1);
        }

        int64_t part_ne[4];
        लागि (पूर्णाङ्क i = 0; i < 4; i++) {
            part_ne[i] = *(src0->ne + i);
        }
        part_ne[0] = interval;

        माप_प्रकार part_nb[4];
        part_nb[0] = परिमाण(weight_type);
        लागि (पूर्णाङ्क i = 1; i < 4; i++) {
            part_nb[i] = part_nb[i - 1] * part_ne[i - 1];
        }

        ggml_cann_pool_alloc part_kernel_allocator;
        part_kernel_allocator.alloc(ctx.pool(), part_nb[3]);
        शून्य * part_kernel_buf = part_kernel_allocator.get();

        acl_tensor_ptr part_kernel = ggml_cann_create_tensor(part_kernel_buf, weight_type, ggml_element_size(src0),
                                                             part_ne, part_nb, 3, ACL_FORMAT_NCL);

        GGML_CANN_CALL_ACLNN_OP(ctx, Slice, acl_weight.get(), slice_dim, slice_start, slice_end, slice_step,
                                part_kernel.get());

        // create the part conv result tensor
        int64_t part_dst_ne[4];
        लागि (पूर्णाङ्क i = 0; i < 4; i++) {
            part_dst_ne[i] = *(dst->ne + i);
        }
        part_dst_ne[0] = (आगत_len - 1) * strideVal[0] - 2 * paddingVal[0] + dilationVal[0] * (part_ne[0] - 1) + 1;

        माप_प्रकार part_dst_nb[4];
        part_dst_nb[0] = परिमाण(weight_type);
        लागि (पूर्णाङ्क i = 1; i < 4; i++) {
            part_dst_nb[i] = part_dst_nb[i - 1] * part_dst_ne[i - 1];
        }
        ggml_cann_pool_alloc part_dst_allocator;
        part_dst_allocator.alloc(ctx.pool(), part_dst_nb[3]);
        शून्य * part_dst_buf = part_dst_allocator.get();

        acl_tensor_ptr acl_part_dst = ggml_cann_create_tensor(part_dst_buf, dst_type, ggml_element_size(dst),
                                                              part_dst_ne, part_dst_nb, 3, ACL_FORMAT_NCL);
        GGML_CANN_CALL_ACLNN_OP(ctx, InplaceZero, acl_part_dst.get());

        // compute part conv transpose 1d
        GGML_CANN_CALL_ACLNN_OP(ctx, Convolution, acl_आगत.get(), part_kernel.get(), nullptr, stride.get(),
                                padding.get(), dilation.get(), transposed, padding.get(), groups, acl_part_dst.get(),
                                cubeMathType);

        // compute the position of part result in final result
        int64_t global_start = slice_start;
        int64_t global_end   = std::min((आगत_len - 1) * strideVal[0] + slice_end, dst_len);

        बायाँ_pad_len  = global_start;
        दायाँ_pad_len = dst_len - global_end;

        std::vector<int64_t> padDataVal = { बायाँ_pad_len, दायाँ_pad_len };
        acl_int_array_ptr    padData    = ggml_cann_create_int_array(padDataVal.data(), 2);

        acl_scalar_ptr pad_मान    = nullptr;
        भिन्न          pad_मान_Val = 0.0;
        pad_मान                   = ggml_cann_create_scalar(&pad_मान_Val, aclDataType::ACL_FLOAT);

        int64_t conv_result_ne[4];
        लागि (पूर्णाङ्क i = 0; i < 4; i++) {
            conv_result_ne[i] = *(dst->ne + i);
        }

        माप_प्रकार conv_result_nb[4];
        conv_result_nb[0] = परिमाण(weight_type);
        लागि (पूर्णाङ्क i = 1; i < 4; i++) {
            conv_result_nb[i] = conv_result_nb[i - 1] * conv_result_ne[i - 1];
        }

        ggml_cann_pool_alloc conv_result_allocator;
        conv_result_allocator.alloc(ctx.pool(), conv_result_nb[3]);
        शून्य * conv_result_buf = conv_result_allocator.get();

        acl_tensor_ptr conv_result = ggml_cann_create_tensor(conv_result_buf, dst_type, ggml_element_size(dst),
                                                             conv_result_ne, conv_result_nb, 3, ACL_FORMAT_NCL);

        GGML_CANN_CALL_ACLNN_OP(ctx, InplaceZero, conv_result.get());
        GGML_CANN_CALL_ACLNN_OP(ctx, ConstantPadNd, acl_part_dst.get(), padData.get(), pad_मान.get(),
                                conv_result.get());
        GGML_CANN_CALL_ACLNN_OP(ctx, InplaceAdd, acl_dst.get(), conv_result.get(), alpha.get());
    }
}

शून्य ggml_cann_elu(ggml_backend_cann_context & ctx, ggml_tensor * dst) {
    ggml_tensor * src0 = dst->src[0];

    acl_tensor_ptr acl_आगत = ggml_cann_create_tensor(src0);
    acl_tensor_ptr acl_dst   = ggml_cann_create_tensor(dst);

    भिन्न          alpha_मान = 1.0f;
    acl_scalar_ptr alpha      = nullptr;
    alpha                     = ggml_cann_create_scalar(&alpha_मान, aclDataType::ACL_FLOAT);

    GGML_CANN_CALL_ACLNN_OP(ctx, Elu, acl_आगत.get(), alpha.get(), alpha.get(), alpha.get(), acl_dst.get());
}

शून्य ggml_cann_mean(ggml_backend_cann_context & ctx, ggml_tensor * dst) {
    ggml_tensor * src0 = dst->src[0];

    acl_tensor_ptr acl_src = ggml_cann_create_tensor(src0);
    acl_tensor_ptr acl_dst = ggml_cann_create_tensor(dst);

    int64_t           reduce_Dim_मान[] = { 3 };
    acl_int_array_ptr reduceDim        = ggml_cann_create_int_array(reduce_Dim_मान, 1);
    तार्किक              keepDim          = सत्य;

    GGML_CANN_CALL_ACLNN_OP(ctx, Mean, acl_src.get(), reduceDim.get(), keepDim, ACL_FLOAT, acl_dst.get());
}

शून्य ggml_cann_pad_reflect_1d(ggml_backend_cann_context & ctx, ggml_tensor * dst) {
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

    acl_tensor_ptr acl_src = ggml_cann_create_tensor(src0->data, ggml_cann_type_mapping(src0->type),
                                                     ggml_element_size(src0), src_ne_3d, src0->nb, 3);

    acl_tensor_ptr acl_dst = ggml_cann_create_tensor(dst->data, ggml_cann_type_mapping(dst->type),
                                                     ggml_element_size(dst), dst_ne_3d, dst->nb, 3);

    GGML_CANN_CALL_ACLNN_OP(ctx, ReflectionPad1d, acl_src.get(), paddings.get(), acl_dst.get());
}

शून्य ggml_cann_गन्ती_equal(ggml_backend_cann_context & ctx, ggml_tensor * dst) {
    ggml_tensor * src0 = dst->src[0];
    ggml_tensor * src1 = dst->src[1];

    // Write element-wise equality (0 or 1) into a temporary buffer to avoid
    // modifying src0 in-place.  Use the same type as src0 so ReduceSum can
    // consume it directly without a type cast.
    ggml_cann_pool_alloc eq_alloc(ctx.pool(), ggml_nelements(src0) * ggml_element_size(src0));
    माप_प्रकार eq_nb[GGML_MAX_DIMS];
    eq_nb[0] = ggml_element_size(src0);
    लागि (पूर्णाङ्क i = 1; i < GGML_MAX_DIMS; i++) {
        eq_nb[i] = eq_nb[i - 1] * src0->ne[i - 1];
    }
    acl_tensor_ptr acl_eq = ggml_cann_create_tensor(
        eq_alloc.get(), ggml_cann_type_mapping(src0->type), ggml_element_size(src0),
        src0->ne, eq_nb, GGML_MAX_DIMS);

    acl_tensor_ptr acl_self  = ggml_cann_create_tensor(src0);
    acl_tensor_ptr acl_other = ggml_cann_create_tensor(src1);
    GGML_CANN_CALL_ACLNN_OP(ctx, EqTensor, acl_self.get(), acl_other.get(), acl_eq.get());

    // Sum the 0/1 values into dst.
    acl_tensor_ptr    acl_dst    = ggml_cann_create_tensor(dst);
    int64_t           dims[4]    = { 0, 1, 2, 3 };
    acl_int_array_ptr dims_arr   = ggml_cann_create_int_array(dims, 4);
    GGML_CANN_CALL_ACLNN_OP(ctx, ReduceSum, acl_eq.get(), dims_arr.get(), सत्य,
                            ggml_cann_type_mapping(dst->type), acl_dst.get());
}

शून्य ggml_cann_step(ggml_backend_cann_context & ctx, ggml_tensor * dst) {
    ggml_tensor * src0 = dst->src[0];

    acl_tensor_ptr acl_src = ggml_cann_create_tensor(src0);
    acl_tensor_ptr acl_dst = ggml_cann_create_tensor(dst);

    भिन्न          alpha_मान = 0.0f;
    acl_scalar_ptr alpha      = nullptr;
    alpha                     = ggml_cann_create_scalar(&alpha_मान, aclDataType::ACL_FLOAT);

    GGML_CANN_CALL_ACLNN_OP(ctx, GtScalar, acl_src.get(), alpha.get(), acl_dst.get());
}

शून्य ggml_cann_softplus(ggml_backend_cann_context & ctx, ggml_tensor * dst) {
    ggml_tensor * src0 = dst->src[0];

    acl_tensor_ptr acl_src = ggml_cann_create_tensor(src0);
    acl_tensor_ptr acl_dst = ggml_cann_create_tensor(dst);

    भिन्न          beta_val      = 1.0f;
    भिन्न          threshold_val = 20.0f;
    acl_scalar_ptr beta          = ggml_cann_create_scalar(&beta_val,      ACL_FLOAT);
    acl_scalar_ptr threshold     = ggml_cann_create_scalar(&threshold_val, ACL_FLOAT);

    GGML_CANN_CALL_ACLNN_OP(ctx, Softplus, acl_src.get(), beta.get(), threshold.get(), acl_dst.get());
}

शून्य ggml_cann_geglu_quick(ggml_backend_cann_context & ctx, ggml_tensor * dst) {
    स्वतः gelu_quick_fn = [](ggml_backend_cann_context & ctx, aclTensor * acl_src, aclTensor * acl_dst) {
        GGML_CANN_CALL_ACLNN_OP(ctx, GeluV2, acl_src, 0, acl_dst);
    };
    ggml_cann_op_unary_gated(gelu_quick_fn, ctx, dst);
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
स्थिर शून्य ggml_cann_mul_mat_id_fp(ggml_backend_cann_context & ctx, ggml_tensor * dst) {
    //dst   [M, K, N, 1]
    ggml_tensor * src0 = dst->src[0];  //src0	[D, M, A, 1]  -> [D, M, K, 1]
    ggml_tensor * src1 = dst->src[1];  //src1	[D, B, N, 1], B = K or B = 1 -> [D, 1, K, 1]
    ggml_tensor * ids  = dst->src[2];  //ids	[K, N]

    GGML_ASSERT(src0->ne[3] == 1);
    GGML_ASSERT(src1->ne[3] == 1);
    GGML_ASSERT(dst->ne[3] == 1);

    int64_t batch = src1->ne[2];
    GGML_ASSERT(batch == ids->ne[1]);

    ggml_cann_pool_alloc export_allocator(ctx.pool(), src0->ne[0] * src0->ne[1] * ids->ne[0] * ggml_element_size(src0));
    शून्य *               export_ptr = export_allocator.get();
    लागि (int64_t i = 0; i < batch; i++) {
        acl_tensor_ptr select_index  = ggml_cann_create_tensor(ids, ids->ne, ids->nb, 1, ACL_FORMAT_ND, i * ids->nb[1]);
        acl_tensor_ptr export_weight = ggml_cann_create_tensor(src0, src0->ne, src0->nb, 3);

        int64_t select_export_ne[] = { src0->ne[0], src0->ne[1], ids->ne[0] };
        माप_प्रकार  select_export_nb[3];
        select_export_nb[0] = src0->nb[0];
        लागि (पूर्णाङ्क k = 1; k < 3; k++) {
            select_export_nb[k] = select_export_nb[k - 1] * select_export_ne[k - 1];
        }

        acl_tensor_ptr select_export =
            ggml_cann_create_tensor(export_ptr, ggml_cann_type_mapping(src0->type), ggml_element_size(src0),
                                    select_export_ne, select_export_nb, 3);
        GGML_CANN_CALL_ACLNN_OP(ctx, IndexSelect, export_weight.get(), 0, select_index.get(), select_export.get());

        int64_t        select_transpose_ne[] = { select_export_ne[1], select_export_ne[0], select_export_ne[2] };
        माप_प्रकार         select_transpose_nb[] = { select_export_nb[1], select_export_nb[0], select_export_nb[2] };
        acl_tensor_ptr select_export_transpose =
            ggml_cann_create_tensor(export_ptr, ggml_cann_type_mapping(src0->type), ggml_element_size(src0),
                                    select_transpose_ne, select_transpose_nb, 3);

        int64_t        active_tensor_ne[] = { src1->ne[0], 1, src1->ne[1] };
        माप_प्रकार         active_tensor_nb[] = { src1->nb[0], src1->nb[1], src1->nb[1] };
        acl_tensor_ptr active_tensor =
            ggml_cann_create_tensor(src1, active_tensor_ne, active_tensor_nb, 3, ACL_FORMAT_ND, i * src1->nb[2]);

        int64_t        dst_ne[] = { dst->ne[0], 1, dst->ne[1] };
        माप_प्रकार         dst_nb[] = { dst->nb[0], dst->nb[1], dst->nb[1] };
        acl_tensor_ptr acl_dst  = ggml_cann_create_tensor(dst, dst_ne, dst_nb, 3, ACL_FORMAT_ND, i * dst->nb[2]);

        GGML_CANN_CALL_ACLNN_OP(ctx, BatchMatMul, active_tensor.get(), select_export_transpose.get(), acl_dst.get(), 2);
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
स्थिर शून्य ggml_cann_mul_mat_id_quant(ggml_backend_cann_context & ctx, ggml_tensor * dst) {
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

    अचर int64_t        n_batches        = ids->ne[1];
    अचर int64_t        n_select_experts = ids->ne[0];
    अचर गणना ggml_type type             = src0->type;

    अचर int32_t group_size = QK8_0;  // Both Q4_0 and Q8_0 use group size of 32
    GGML_ASSERT(group_size == QK4_0);

    // Calculate element size for quantized weights
    अचर भिन्न weight_elem_size =
        (type == GGML_TYPE_Q4_0) ? 0.5f :
        (type == GGML_TYPE_Q8_0) ? 1.0f :
                                   (GGML_ABORT("MUL_MAT_ID only supports Q4_0 and Q8_0"), 0.0f);

    // Calculate scale offset in memory
    अचर माप_प्रकार weight_size     = src0->ne[0] * src0->ne[1] * src0->ne[2] * weight_elem_size;
    अचर माप_प्रकार scale_elem_size = परिमाण(uint16_t);
    वर्ण *       scale_data      = (वर्ण *) src0->data + weight_size;

    // Allocate buffers for selected expert weights and scales
    अचर माप_प्रकार         selected_weight_size = src0->ne[0] * src0->ne[1] * n_select_experts * weight_elem_size;
    ggml_cann_pool_alloc selected_weight_alloc(ctx.pool(), selected_weight_size);
    शून्य *               selected_weight_buffer = selected_weight_alloc.get();

    अचर माप_प्रकार selected_scale_size = (src0->ne[0] / group_size) * src0->ne[1] * n_select_experts * scale_elem_size;
    ggml_cann_pool_alloc selected_scale_alloc(ctx.pool(), selected_scale_size);
    शून्य *               selected_scale_buffer = selected_scale_alloc.get();

    // Helper lambda to allocate and cast tensor to F16 if needed
    constexpr माप_प्रकार f16_elem_size      = परिमाण(uint16_t);
    स्वतः             prepare_f16_buffer = [&](ggml_tensor * tensor, ggml_cann_pool_alloc & allocator,
                                  तार्किक need_cast = असत्य) -> शून्य * {
        यदि (tensor->type == GGML_TYPE_F16) {
            फर्काऊ tensor->data;
        }

        माप_प्रकार total_size = f16_elem_size;
        लागि (पूर्णाङ्क i = 0; i < GGML_MAX_DIMS; i++) {
            total_size *= tensor->ne[i];
        }
        शून्य * buffer = allocator.alloc(total_size);

        यदि (need_cast == असत्य) {
            फर्काऊ buffer;
        }

        int64_t ne[GGML_MAX_DIMS];
        माप_प्रकार  nb[GGML_MAX_DIMS] = { f16_elem_size };
        लागि (पूर्णाङ्क i = 0; i < GGML_MAX_DIMS; i++) {
            ne[i] = tensor->ne[i];
            यदि (i > 0) {
                nb[i] = nb[i - 1] * ne[i - 1];
            }
        }

        acl_tensor_ptr src_tensor = ggml_cann_create_tensor(tensor);
        acl_tensor_ptr f16_tensor = ggml_cann_create_tensor(buffer, ACL_FLOAT16, f16_elem_size, ne, nb, GGML_MAX_DIMS);
        aclnn_cast(ctx, src_tensor.get(), f16_tensor.get(), ACL_FLOAT16);

        फर्काऊ buffer;
    };

    // Prepare input and output buffers
    ggml_cann_pool_alloc आगत_alloc(ctx.pool());
    शून्य *               आगत_buffer = prepare_f16_buffer(src1, आगत_alloc, सत्य);

    ggml_cann_pool_alloc output_alloc(ctx.pool());
    शून्य *               output_buffer = prepare_f16_buffer(dst, output_alloc, असत्य);

    // Process each batch
    लागि (int64_t batch_idx = 0; batch_idx < n_batches; batch_idx++) {
        // Create index tensor for current batch
        अचर माप_प्रकार   index_offset  = batch_idx * ids->nb[1];
        acl_tensor_ptr batch_indices = ggml_cann_create_tensor(ids, ids->ne, ids->nb, 1, ACL_FORMAT_ND, index_offset);

        // Select quantized weights using expert indices
        // Q4_0 stores 2 values per byte, Q8_0 stores 1 value per byte
        अचर int64_t weight_d         = (type == GGML_TYPE_Q4_0) ? src0->ne[0] / 2 : src0->ne[0];
        अचर int64_t weight_m         = src0->ne[1];
        अचर int64_t weight_n_experts = src0->ne[2];

        int64_t weight_ne[3] = { weight_d, weight_m, weight_n_experts };
        माप_प्रकार  weight_nb[3] = { परिमाण(int8_t), weight_d * परिमाण(int8_t), weight_d * weight_m * परिमाण(int8_t) };

        acl_tensor_ptr all_weights =
            ggml_cann_create_tensor(src0->data, ACL_INT8, परिमाण(int8_t), weight_ne, weight_nb, 3);

        int64_t selected_weight_ne[3] = { weight_d, weight_m, n_select_experts };
        माप_प्रकार  selected_weight_nb[3] = { परिमाण(int8_t), weight_d * परिमाण(int8_t),
                                          weight_d * weight_m * परिमाण(int8_t) };

        acl_tensor_ptr selected_weights = ggml_cann_create_tensor(selected_weight_buffer, ACL_INT8, परिमाण(int8_t),
                                                                  selected_weight_ne, selected_weight_nb, 3);

        GGML_CANN_CALL_ACLNN_OP(ctx, IndexSelect, all_weights.get(), 0, batch_indices.get(), selected_weights.get());

        // Select scales using the same expert indices
        अचर int64_t scale_d     = src0->ne[0] / group_size;
        int64_t       scale_ne[3] = { scale_d, weight_m, weight_n_experts };
        माप_प्रकार scale_nb[3] = { scale_elem_size, scale_d * scale_elem_size, scale_d * weight_m * scale_elem_size };

        acl_tensor_ptr all_scales =
            ggml_cann_create_tensor(scale_data, ACL_FLOAT16, scale_elem_size, scale_ne, scale_nb, 3);

        int64_t selected_scale_ne[3] = { scale_d, weight_m, n_select_experts };
        माप_प्रकार  selected_scale_nb[3] = { scale_elem_size, scale_d * scale_elem_size,
                                         scale_d * weight_m * scale_elem_size };

        acl_tensor_ptr selected_scales = ggml_cann_create_tensor(selected_scale_buffer, ACL_FLOAT16, scale_elem_size,
                                                                 selected_scale_ne, selected_scale_nb, 3);

        GGML_CANN_CALL_ACLNN_OP(ctx, IndexSelect, all_scales.get(), 0, batch_indices.get(), selected_scales.get());

        // Process each expert for current batch
        // IndexSelect output layout: [D, M, K] in contiguous format
        // WeightQuantBatchMatmulV2 expects: [M, D] with row-major stride
        लागि (int64_t expert_idx = 0; expert_idx < n_select_experts; expert_idx++) {
            // Determine input offset: broadcast if src1->ne[1]==1, otherwise use per-expert input
            अचर माप_प्रकार आगत_offset =
                (batch_idx * src1->ne[1] + (src1->ne[1] == 1 ? 0 : expert_idx)) * src1->ne[0] * f16_elem_size;
            अचर माप_प्रकार output_offset = (batch_idx * dst->ne[1] + expert_idx) * dst->ne[0] * f16_elem_size;

            // Create weight view for current expert: [D, M, K] -> [M, D]
            int64_t      weight_view_ne[2]  = { weight_m, src0->ne[0] };
            भिन्न        weight_view_nb[2]  = { src0->ne[0] * weight_elem_size, weight_elem_size };
            अचर माप_प्रकार weight_view_offset = expert_idx * selected_weight_nb[2];

            acl_tensor_ptr weight_view =
                ggml_cann_create_tensor(selected_weight_buffer, ggml_cann_type_mapping(type), weight_elem_size,
                                        weight_view_ne, weight_view_nb, 2, ACL_FORMAT_ND, weight_view_offset);

            // Create scale view for current expert: [D, M, K] -> [M, D]
            int64_t      scale_view_ne[2]  = { weight_m, scale_d };
            माप_प्रकार       scale_view_nb[2]  = { selected_scale_nb[1], selected_scale_nb[0] };
            अचर माप_प्रकार scale_view_offset = expert_idx * selected_scale_nb[2];

            acl_tensor_ptr scale_view =
                ggml_cann_create_tensor(selected_scale_buffer, ACL_FLOAT16, scale_elem_size, scale_view_ne,
                                        scale_view_nb, 2, ACL_FORMAT_ND, scale_view_offset);

            // Create input activation tensor [D, 1]
            int64_t आगत_ne[2] = { src1->ne[0], 1 };
            माप_प्रकार  आगत_nb[2] = { f16_elem_size, src1->ne[0] * f16_elem_size };

            acl_tensor_ptr आगत_tensor = ggml_cann_create_tensor(आगत_buffer, ACL_FLOAT16, f16_elem_size, आगत_ne,
                                                                  आगत_nb, 2, ACL_FORMAT_ND, आगत_offset);

            // Create output tensor [M, 1]
            int64_t output_ne[2] = { dst->ne[0], 1 };
            माप_प्रकार  output_nb[2] = { f16_elem_size, dst->ne[0] * f16_elem_size };

            acl_tensor_ptr output_tensor = ggml_cann_create_tensor(output_buffer, ACL_FLOAT16, f16_elem_size, output_ne,
                                                                   output_nb, 2, ACL_FORMAT_ND, output_offset);

            // Perform quantized matrix multiplication
            GGML_CANN_CALL_ACLNN_OP(ctx, WeightQuantBatchMatmulV2, आगत_tensor.get(), weight_view.get(),
                                    scale_view.get(), nullptr, nullptr, nullptr, nullptr, group_size,
                                    output_tensor.get());
        }
    }

    // Cast output back to original type if we used a temporary F16 buffer
    यदि (dst->type != GGML_TYPE_F16) {
        int64_t ne[GGML_MAX_DIMS];
        माप_प्रकार  nb[GGML_MAX_DIMS] = { f16_elem_size };
        लागि (पूर्णाङ्क i = 0; i < GGML_MAX_DIMS; i++) {
            ne[i] = dst->ne[i];
            यदि (i > 0) {
                nb[i] = nb[i - 1] * ne[i - 1];
            }
        }

        acl_tensor_ptr f16_output =
            ggml_cann_create_tensor(output_buffer, ACL_FLOAT16, f16_elem_size, ne, nb, GGML_MAX_DIMS);
        acl_tensor_ptr dst_tensor = ggml_cann_create_tensor(dst);

        aclnn_cast(ctx, f16_output.get(), dst_tensor.get(), ggml_cann_type_mapping(dst->type));
    }
}

शून्य ggml_cann_mul_mat_id(ggml_backend_cann_context & ctx, ggml_tensor * dst) {
    अचर गणना ggml_type type = dst->src[0]->type;
    विकल्प (type) {
        पक्ष GGML_TYPE_F32:
        पक्ष GGML_TYPE_F16:
            ggml_cann_mul_mat_id_fp(ctx, dst);
            रोक;
        पक्ष GGML_TYPE_Q4_0:
        पक्ष GGML_TYPE_Q8_0:
            ggml_cann_mul_mat_id_quant(ctx, dst);
            रोक;
        स्वभाव:
            GGML_ABORT("Unsupported type for mul_mat_id");
            रोक;
    }
}

शून्य ggml_cann_flash_attn_ext(ggml_backend_cann_context & ctx, ggml_tensor * dst) {
    ggml_tensor * src0 = dst->src[0];  // q, fp32 | B, N, S, D (uncont) -> B, S, N, D (cont)
    ggml_tensor * src1 = dst->src[1];  // k, fp16 | B, N, S, D (uncont) -> B, S, N, D (cont)
    ggml_tensor * src2 = dst->src[2];  // v, fp16 | B, N, S, D (uncont) -> B, S, N, D (cont)
    ggml_tensor * src3 = dst->src[3];  // mask, fp16

    // B, N, S, D (uncont) -> B, S, N, D (cont)
    int64_t src0_bsnd_ne[GGML_MAX_DIMS];
    स_नकल(src0_bsnd_ne, src0->ne, GGML_MAX_DIMS * परिमाण(int64_t));
    माप_प्रकार src0_bsnd_nb[GGML_MAX_DIMS];
    स_नकल(src0_bsnd_nb, src0->nb, GGML_MAX_DIMS * परिमाण(माप_प्रकार));
    int64_t src1_bsnd_ne[GGML_MAX_DIMS];
    स_नकल(src1_bsnd_ne, src1->ne, GGML_MAX_DIMS * परिमाण(int64_t));
    माप_प्रकार src1_bsnd_nb[GGML_MAX_DIMS];
    स_नकल(src1_bsnd_nb, src1->nb, GGML_MAX_DIMS * परिमाण(माप_प्रकार));
    int64_t src2_bsnd_ne[GGML_MAX_DIMS];
    स_नकल(src2_bsnd_ne, src2->ne, GGML_MAX_DIMS * परिमाण(int64_t));
    माप_प्रकार src2_bsnd_nb[GGML_MAX_DIMS];
    स_नकल(src2_bsnd_nb, src2->nb, GGML_MAX_DIMS * परिमाण(माप_प्रकार));

    स्वतः transpose12 = [](int64_t * ne, माप_प्रकार * nb) {
        int64_t ne_tmp = ne[1];
        माप_प्रकार  nb_tmp = nb[1];
        ne[1]          = ne[2];
        nb[1]          = nb[2];
        ne[2]          = ne_tmp;
        nb[2]          = nb_tmp;
    };

    transpose12(src0_bsnd_ne, src0_bsnd_nb);
    transpose12(src1_bsnd_ne, src1_bsnd_nb);
    transpose12(src2_bsnd_ne, src2_bsnd_nb);

    भिन्न maxBias      = 0.0f;
    भिन्न scale_मान   = 1.0f;
    भिन्न logitSoftcap = 0.0f;
    स_नकल(&scale_मान, (भिन्न *) dst->op_params + 0, परिमाण(भिन्न));
    स_नकल(&maxBias, (भिन्न *) dst->op_params + 1, परिमाण(भिन्न));
    स_नकल(&logitSoftcap, (भिन्न *) dst->op_params + 2, परिमाण(भिन्न));

    यदि (logitSoftcap == 0.0f) {
        माप_प्रकार faElemSize = परिमाण(uint16_t);
        स्वतः   faDataType = ACL_FLOAT16;  //ACL_BF16;

        acl_tensor_ptr acl_q_tensor = nullptr;
        acl_tensor_ptr acl_k_tensor = nullptr;
        acl_tensor_ptr acl_v_tensor = nullptr;

        // Step 1: cast the src0 (Query) to fp16 if needed
        ggml_cann_pool_alloc src0_f16_allocator(ctx.pool());
        शून्य *               src0_f16_buffer = nullptr;

        यदि (ggml_cann_type_mapping(src0->type) != faDataType) {
            acl_tensor_ptr acl_src0_f32_tensor =
                ggml_cann_create_tensor(src0, src0_bsnd_ne, src0_bsnd_nb, GGML_MAX_DIMS);
            src0_f16_buffer = src0_f16_allocator.alloc(ggml_nelements(src0) * faElemSize);

            int64_t * src0_f16_ne = src0_bsnd_ne;
            माप_प्रकार    src0_f16_nb[GGML_MAX_DIMS];
            src0_f16_nb[0] = परिमाण(uint16_t);
            लागि (पूर्णाङ्क i = 1; i < GGML_MAX_DIMS; ++i) {
                src0_f16_nb[i] = src0_f16_nb[i - 1] * src0_f16_ne[i - 1];
            }

            acl_q_tensor = ggml_cann_create_tensor(src0_f16_buffer, faDataType, faElemSize, src0_f16_ne, src0_f16_nb,
                                                   GGML_MAX_DIMS);
            aclnn_cast(ctx, acl_src0_f32_tensor.get(), acl_q_tensor.get(), faDataType);
        } अन्यथा {
            acl_q_tensor = ggml_cann_create_tensor(src0, src0_bsnd_ne, src0_bsnd_nb, GGML_MAX_DIMS);
        }

        // Step 2: create the acl tensors for src1 (Key), src2 (Value),
        //         and the direct output from FusedInferAttention

        acl_k_tensor = ggml_cann_create_tensor(src1, src1_bsnd_ne, src1_bsnd_nb, GGML_MAX_DIMS);
        acl_v_tensor = ggml_cann_create_tensor(src2, src2_bsnd_ne, src2_bsnd_nb, GGML_MAX_DIMS);

        // Step 2.5: Pad Q, K, V along head dimension if D is not a multiple of 16
        //           (required by FusedInferAttentionScoreV2)
        अचर int64_t D         = src0->ne[0];
        अचर int64_t D_padded  = GGML_PAD(D, 16);
        अचर तार्किक needs_padding = (D != D_padded);

        ggml_cann_pool_alloc q_pad_allocator(ctx.pool());
        ggml_cann_pool_alloc k_pad_allocator(ctx.pool());
        ggml_cann_pool_alloc v_pad_allocator(ctx.pool());

        यदि (needs_padding) {
            int64_t paddings[] = { 0, D_padded - D, 0, 0, 0, 0, 0, 0 };

            स्वतः pad_fa_tensor = [&](acl_tensor_ptr & tensor, अचर int64_t * bsnd_ne,
                                     ggml_cann_pool_alloc & allocator) {
                int64_t pad_ne[GGML_MAX_DIMS] = { D_padded, bsnd_ne[1], bsnd_ne[2], bsnd_ne[3] };
                माप_प्रकार  pad_nb[GGML_MAX_DIMS];
                pad_nb[0] = faElemSize;
                लागि (पूर्णाङ्क i = 1; i < GGML_MAX_DIMS; ++i) {
                    pad_nb[i] = pad_nb[i - 1] * pad_ne[i - 1];
                }
                int64_t nelements = pad_ne[0] * pad_ne[1] * pad_ne[2] * pad_ne[3];
                शून्य *  buffer    = allocator.alloc(nelements * faElemSize);
                acl_tensor_ptr padded =
                    ggml_cann_create_tensor(buffer, faDataType, faElemSize, pad_ne, pad_nb, GGML_MAX_DIMS);
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
        यदि (src3 != nullptr) {
            // Construct the truncated pse tensor (common for prefill/decode)
            int64_t trunc_pse_ne[GGML_MAX_DIMS] = {
                src3->ne[0],  // D
                src0->ne[1],  // S (number of Q tokens)
                src3->ne[2],  // mask N
                src3->ne[3]   // B
            };
            माप_प्रकार * trunc_pse_nb = src3->nb;

            acl_tensor_ptr acl_mask_f16_trunc_tensor = ggml_cann_create_tensor(
                src3->data, ACL_FLOAT16, परिमाण(uint16_t), trunc_pse_ne, trunc_pse_nb, GGML_MAX_DIMS);

            int64_t bcast_pse_ne[GGML_MAX_DIMS];
            माप_प्रकार  bcast_pse_nb[GGML_MAX_DIMS];
            bcast_pse_ne[0] = src3->ne[0];  // D
            bcast_pse_ne[1] = src0->ne[1];  // S
            bcast_pse_ne[2] = src0->ne[2];  // N (num_heads)
            bcast_pse_ne[3] = src3->ne[3];  // B
            यदि (maxBias == 0.0f) {
                // When maxBias == 0.0f, use nb = 0 reduce once repeat (Qwen2)
                // Construct the bcast tensor (simulate repeat on the head dimension using stride=0)
                bcast_pse_nb[0] = परिमाण(uint16_t);
                bcast_pse_nb[1] = bcast_pse_nb[0] * bcast_pse_ne[0];
                bcast_pse_nb[2] = 0;  // <---- the head dimension shares the same data
                bcast_pse_nb[3] = src3->nb[3];

                bcast_pse_tensor = ggml_cann_create_tensor(src3->data, ACL_FLOAT16, परिमाण(uint16_t), bcast_pse_ne,
                                                           bcast_pse_nb, GGML_MAX_DIMS);

            } अन्यथा {
                bcast_pse_nb[0] = परिमाण(uint16_t);
                लागि (पूर्णाङ्क i = 1; i < GGML_MAX_DIMS; i++) {
                    bcast_pse_nb[i] = bcast_pse_nb[i - 1] * bcast_pse_ne[i - 1];
                }

                शून्य * bcast_pse_buffer =
                    bcast_pse_allocator.alloc(ggml_nelements(src3) * src0->ne[2] * परिमाण(uint16_t));

                bcast_pse_tensor = ggml_cann_create_tensor(bcast_pse_buffer, ACL_FLOAT16, परिमाण(uint16_t),
                                                           bcast_pse_ne, bcast_pse_nb, GGML_MAX_DIMS);

                int64_t repeats[] = { 1, src0->ne[2], 1, 1 };
                aclnn_repeat(ctx, acl_mask_f16_trunc_tensor.get(), bcast_pse_tensor.get(), repeats);

                // alibi
                // Compute the slope if needed. Derived from ggml_cann_softmax().
                अचर int64_t        n_heads = src0->ne[2];
                ggml_cann_pool_alloc slope_allocator(ctx.pool(), n_heads * परिमाण(uint16_t));
                शून्य *               slope_buffer = slope_allocator.get();
                aclnn_get_slope(ctx, n_heads, slope_buffer, maxBias, GGML_TYPE_F16);

                int64_t slope_ne[] = { 1, 1, n_heads, 1 };
                माप_प्रकार  slope_nb[GGML_MAX_DIMS];
                slope_nb[0] = परिमाण(uint16_t);
                लागि (पूर्णाङ्क i = 1; i < GGML_MAX_DIMS; i++) {
                    slope_nb[i] = slope_nb[i - 1] * slope_ne[0];
                }

                acl_tensor_ptr slope_tensor = ggml_cann_create_tensor(slope_buffer, ACL_FLOAT16, परिमाण(uint16_t),
                                                                      slope_ne, slope_nb, GGML_MAX_DIMS);
                GGML_CANN_CALL_ACLNN_OP(ctx, InplaceMul, bcast_pse_tensor.get(), slope_tensor.get());
            }
        }

        // Step 4: set the inputs for FusedInferAttention.
        acl_tensor_list_ptr acl_k_tensor_list = ggml_cann_create_tensor_list(acl_k_tensor);
        acl_tensor_list_ptr acl_v_tensor_list = ggml_cann_create_tensor_list(acl_v_tensor);

        int64_t अङ्क_Heads           = src0->ne[2];  // N
        int64_t अङ्क_साँचो_मान_Heads   = src1->ne[2];
        // double  scaleValue = 1 / sqrt(src0->ne[0]); // 1/sqrt(d)
        int64_t preTokens          = 65535;
        int64_t nextTokens         = 65535;
        वर्ण    layout[5]          = { 'B', 'S', 'N', 'D', 0 };
        int64_t sparseMode         = 0;
        int64_t innerPrecise       = (src0->ne[1] == 1) ? 0 : 2;
        int64_t blockSize          = 0;
        int64_t antiquantMode      = 0;
        तार्किक    softmaxLseFlag     = असत्य;
        int64_t साँचो_Antiquant_Mode   = 0;
        int64_t मान_Antiquant_Mode = 0;

        GGML_ASSERT(dst->type == GGML_TYPE_F32 || dst->type == GGML_TYPE_F16);
        acl_tensor_ptr       fa_dst_tensor;
        ggml_cann_pool_alloc out_f16_allocator(ctx.pool());
        यदि (dst->type == GGML_TYPE_F32 || needs_padding) {
            int64_t * out_f16_ne = src0_bsnd_ne;
            माप_प्रकार    out_f16_nb[GGML_MAX_DIMS];
            out_f16_nb[0] = faElemSize;
            लागि (पूर्णाङ्क i = 1; i < GGML_MAX_DIMS; ++i) {
                out_f16_nb[i] = out_f16_nb[i - 1] * out_f16_ne[i - 1];
            }
            int64_t out_nelements = out_f16_ne[0] * out_f16_ne[1] * out_f16_ne[2] * out_f16_ne[3];
            शून्य *  out_f16_buffer = out_f16_allocator.alloc(out_nelements * faElemSize);

            fa_dst_tensor =
                ggml_cann_create_tensor(out_f16_buffer, faDataType, faElemSize, out_f16_ne, out_f16_nb, GGML_MAX_DIMS);
        } अन्यथा {
            fa_dst_tensor = ggml_cann_create_tensor(dst);
        }

        GGML_CANN_CALL_ACLNN_OP(ctx, FusedInferAttentionScoreV2, acl_q_tensor.get(), acl_k_tensor_list.get(),
                                acl_v_tensor_list.get(),               // q, k, v
                                bcast_pse_tensor.get(), nullptr,       // pse, mask
                                nullptr, nullptr,                      // actSeqLen, actSeqLenkv
                                nullptr, nullptr,                      // deqScale1, quantScale1
                                nullptr, nullptr, nullptr,             // deqScale2, quantScale2, quantOffset2
                                nullptr, nullptr,                      // antiquantScale, antiquantOffset
                                nullptr,                               // blockTable
                                nullptr, nullptr,                      // qPadSize, kvPadSize
                                nullptr, nullptr,                      // kAntiquantScale, kAntiQuantOffset
                                nullptr, nullptr,                      // vAntiquantScale, vAntiQuantOffset
                                nullptr, nullptr, nullptr,             // kSharedPrefix, vSharedPrefix, actSharedLen
                                अङ्क_Heads, scale_मान,                  // heads, scaleValue
                                preTokens, nextTokens,                 // preTokens, nextTokens
                                layout,                                // inputLayout
                                अङ्क_साँचो_मान_Heads,                      // numKVHeads
                                sparseMode, innerPrecise,              // sparseMode, innerPrecise
                                blockSize, antiquantMode,              // blockSize, antiquantMode
                                softmaxLseFlag,                        // softmaxLseFlag
                                साँचो_Antiquant_Mode, मान_Antiquant_Mode,  // keyAntiqMode, valueAntiqMode
                                fa_dst_tensor.get(),                   // attentionOut
                                nullptr                                // softmaxLse
        );

        // Step 6: post-processing — slice padded output and/or cast to f32
        यदि (needs_padding) {
            ggml_cann_pool_alloc sliced_f16_allocator(ctx.pool());

            यदि (dst->type == GGML_TYPE_F32) {
                int64_t sliced_ne[GGML_MAX_DIMS] = { D, src0_bsnd_ne[1], src0_bsnd_ne[2], src0_bsnd_ne[3] };
                माप_प्रकार  sliced_nb[GGML_MAX_DIMS];
                sliced_nb[0] = faElemSize;
                लागि (पूर्णाङ्क i = 1; i < GGML_MAX_DIMS; ++i) {
                    sliced_nb[i] = sliced_nb[i - 1] * sliced_ne[i - 1];
                }
                int64_t sliced_nelements = sliced_ne[0] * sliced_ne[1] * sliced_ne[2] * sliced_ne[3];
                शून्य *  sliced_buffer    = sliced_f16_allocator.alloc(sliced_nelements * faElemSize);
                acl_tensor_ptr sliced_f16_tensor = ggml_cann_create_tensor(sliced_buffer, faDataType, faElemSize,
                                                                           sliced_ne, sliced_nb, GGML_MAX_DIMS);

                GGML_CANN_CALL_ACLNN_OP(ctx, Slice, fa_dst_tensor.get(),
                                        (int64_t) -1, (int64_t) 0, D, (int64_t) 1, sliced_f16_tensor.get());

                acl_tensor_ptr acl_dst_tensor = ggml_cann_create_tensor(dst);
                aclnn_cast(ctx, sliced_f16_tensor.get(), acl_dst_tensor.get(), ggml_cann_type_mapping(dst->type));
            } अन्यथा {
                acl_tensor_ptr acl_dst_tensor = ggml_cann_create_tensor(dst);
                GGML_CANN_CALL_ACLNN_OP(ctx, Slice, fa_dst_tensor.get(),
                                        (int64_t) -1, (int64_t) 0, D, (int64_t) 1, acl_dst_tensor.get());
            }
        } अन्यथा यदि (dst->type == GGML_TYPE_F32) {
            acl_tensor_ptr acl_dst_tensor = ggml_cann_create_tensor(dst);
            aclnn_cast(ctx, fa_dst_tensor.get(), acl_dst_tensor.get(), ggml_cann_type_mapping(dst->type));
        }
    } अन्यथा {
        GGML_ABORT("Function is not implemented.");
    }
}

स्थिर शून्य ggml_cann_out_prod_fp(ggml_backend_cann_context & ctx, ggml_tensor * dst) {
    ggml_tensor * src0 = dst->src[0];  // weight  [ne00=m, ne01=K, ne02, ne03]
    ggml_tensor * src1 = dst->src[1];  // input   [ne10=n, ne11=K, ne12, ne13]
    GGML_TENSOR_द्विआधारी_OP_LOCALS

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

    अचर aclDataType src0_acl_type = ggml_cann_type_mapping(src0->type);
    अचर aclDataType src1_acl_type = ggml_cann_type_mapping(src1->type);
    अचर aclDataType dst_acl_type  = ggml_cann_type_mapping(dst->type);
    अचर माप_प्रकार      src0_type_sz  = ggml_type_size(src0->type);
    अचर माप_प्रकार      src1_type_sz  = ggml_type_size(src1->type);
    अचर माप_प्रकार      dst_type_sz   = ggml_type_size(dst->type);

    अचर int64_t dps2 = ne2 / ne02;
    अचर int64_t dps3 = ne3 / ne03;

    लागि (int64_t i3 = 0; i3 < ne3; i3++) {
        लागि (int64_t i2 = 0; i2 < ne2; i2++) {
            अचर int64_t i02 = i2 / dps2;
            अचर int64_t i03 = i3 / dps3;

            // src0 2D slice at [i02, i03]: ggml [m, K] -> ACL [K, m]
            int64_t src0_ne[2] = { ne00, ne01 };
            माप_प्रकार  src0_nb[2] = { nb00, nb01 };
            acl_tensor_ptr acl_src0_s = ggml_cann_create_tensor(
                (वर्ण *) src0->data + i02 * nb02 + i03 * nb03,
                src0_acl_type, src0_type_sz, src0_ne, src0_nb, 2);

            // src1 transposed 2D slice at [i2, i3]: swap ne/nb -> ggml[K,n] -> ACL[n,K]
            int64_t src1_t_ne[2] = { ne11, ne10 };
            माप_प्रकार  src1_t_nb[2] = { nb11, nb10 };
            acl_tensor_ptr acl_src1_t = ggml_cann_create_tensor(
                (वर्ण *) src1->data + i2 * nb12 + i3 * nb13,
                src1_acl_type, src1_type_sz, src1_t_ne, src1_t_nb, 2);

            // dst 2D slice at [i2, i3]: ggml [m, n] -> ACL [n, m]
            int64_t dst_ne[2] = { ne0, ne1 };
            माप_प्रकार  dst_nb[2] = { nb0, nb1 };
            acl_tensor_ptr acl_dst_s = ggml_cann_create_tensor(
                (वर्ण *) dst->data + i2 * nb2 + i3 * nb3,
                dst_acl_type, dst_type_sz, dst_ne, dst_nb, 2);

            // Matmul(src1_t [n,K], src0 [K,m]) = [n,m] = acl_dst_s  ✓
            GGML_CANN_CALL_ACLNN_OP(ctx, Matmul,
                acl_src1_t.get(), acl_src0_s.get(), acl_dst_s.get(), (int8_t) 1);
        }
    }
}

शून्य ggml_cann_out_prod(ggml_backend_cann_context & ctx, ggml_tensor * dst) {
    ggml_tensor * src0 = dst->src[0];

    अचर गणना ggml_type type = src0->type;

    विकल्प (type) {
        पक्ष GGML_TYPE_F32:
        पक्ष GGML_TYPE_F16:
            ggml_cann_out_prod_fp(ctx, dst);
            रोक;
        स्वभाव:
            GGML_ABORT("Unsupport type for GGML_OP_OUT_PROD");
            रोक;
    }
}

शून्य ggml_cann_ssm_conv(ggml_backend_cann_context & ctx, ggml_tensor * dst) {
    ggml_tensor * src0 = dst->src[0];  // conv_x
    ggml_tensor * src1 = dst->src[1];  // conv1d.weight

    // This op is currently defined only for F32 in ggml_cpu
    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    // Shapes follow ggml_compute_forward_ssm_conv_f32
    अचर int64_t nc  = src1->ne[0];   // d_conv
    अचर int64_t ncs = src0->ne[0];   // d_conv - 1 + n_t
    अचर int64_t nr  = src0->ne[1];   // d_inner
    अचर int64_t n_s = src0->ne[2];   // n_seqs

    अचर int64_t n_t = dst->ne[1];    // tokens per sequence

    GGML_ASSERT(dst->ne[0] == nr);     // dst: {d_inner, n_t, n_s}
    GGML_ASSERT(src1->ne[1] == nr);    // weight: {d_conv, d_inner}
    GGML_ASSERT(ncs == nc - 1 + n_t);  // conv_x: {d_conv - 1 + n_t, d_inner, n_s}
    GGML_ASSERT(src0->nb[0] == परिमाण(भिन्न));
    GGML_ASSERT(src1->nb[0] == परिमाण(भिन्न));

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
    माप_प्रकार  w_nb[GGML_MAX_DIMS] = { src1->nb[0], src1->nb[1], src1->nb[1], src1->nb[3] };  // same as src1

    acl_tensor_ptr acl_w = ggml_cann_create_tensor(src1->data, ggml_cann_type_mapping(src1->type),
                                                   ggml_type_size(src1->type), w_ne, w_nb, 3, ACL_FORMAT_NCL);

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
    माप_प्रकार  y_nb[GGML_MAX_DIMS] = { dst->ne[0] * परिमाण(भिन्न), परिमाण(भिन्न), dst->ne[0] * dst->ne[1] * परिमाण(भिन्न),
                                    dst->nb[3] };       // [nr, 1, nr * n_t]

    acl_tensor_ptr acl_y = ggml_cann_create_tensor(dst->data, ggml_cann_type_mapping(dst->type),
                                                   ggml_type_size(dst->type), y_ne, y_nb, 3, ACL_FORMAT_NCL);

    // --- Conv1d parameters: depthwise, stride 1, no padding ("valid") ---
    int64_t strideVal[1]   = { 1 };
    int64_t paddingVal[1]  = { 0 };
    int64_t dilationVal[1] = { 1 };

    acl_int_array_ptr stride   = ggml_cann_create_int_array(strideVal, 1);
    acl_int_array_ptr padding  = ggml_cann_create_int_array(paddingVal, 1);
    acl_int_array_ptr dilation = ggml_cann_create_int_array(dilationVal, 1);

    अचर तार्किक    transposed   = असत्य;
    अचर int64_t groups       = nr;  // depthwise: one group per inner dim
    int8_t        cubeMathType = 0;

#ifdef ASCEND_310P
    cubeMathType = 1;
#endif

    GGML_CANN_CALL_ACLNN_OP(ctx, Convolution,
                            acl_x.get(),    // input:  N, C, L_in = ncs
                            acl_w.get(),    // weight: [C, 1, K] with groups=nr
                            nullptr,        // bias
                            stride.get(), padding.get(), dilation.get(), transposed,
                            padding.get(),  // output padding (unused for non-transposed)
                            groups, acl_y.get(), cubeMathType);
}

शून्य ggml_cann_op_add_rms_norm_fused(ggml_backend_cann_context & ctx,
                                     ggml_tensor *               add_गाँठो,
                                     ggml_tensor *               rms_norm_गाँठो) {
    // Get the two input tensors for ADD operation
    ggml_tensor * x1 = add_गाँठो->src[0];
    ggml_tensor * x2 = add_गाँठो->src[1];

    // Create ACL tensors for the two ADD inputs
    acl_tensor_ptr acl_x1 = ggml_cann_create_tensor(x1);
    acl_tensor_ptr acl_x2 = ggml_cann_create_tensor(x2);

    // Get epsilon parameter from rms_norm_tensor
    भिन्न eps;
    स_नकल(&eps, rms_norm_गाँठो->op_params, परिमाण(भिन्न));

    // Build gamma tensor (RMS normalization scaling factor)
    // Gamma should match the normalized dimensions (last dimension of x1)
    माप_प्रकार acl_gamma_nb[GGML_MAX_DIMS];
    acl_gamma_nb[0] = ggml_type_size(rms_norm_गाँठो->type);
    लागि (पूर्णाङ्क i = 1; i < GGML_MAX_DIMS; i++) {
        acl_gamma_nb[i] = acl_gamma_nb[i - 1] * x1->ne[i - 1];
    }
    acl_tensor_ptr acl_gamma =
        get_cache_acl_tensor(ctx, &ctx.rms_norm_one_tensor_cache.cache, ctx.rms_norm_one_tensor_cache.size, x1->ne,
                             acl_gamma_nb, rms_norm_गाँठो->type,
                             1,    // dims - only the last dimension
                             1.0f  //मान
        );

    // Build rstdOut tensor (output for normalized standard deviation)
    // Shape should be the dimensions that are NOT normalized
    int64_t acl_rstd_ne[] = { 1, x1->ne[1], x1->ne[2], x1->ne[3] };
    माप_प्रकार  acl_rstd_nb[GGML_MAX_DIMS - 1];
    acl_rstd_nb[0] = परिमाण(भिन्न);
    लागि (पूर्णाङ्क i = 1; i < GGML_MAX_DIMS - 1; i++) {
        acl_rstd_nb[i] = acl_rstd_nb[i - 1] * acl_rstd_ne[i - 1];
    }
    acl_tensor_ptr acl_rstd =
        get_cache_acl_tensor(ctx, &ctx.rms_norm_zero_tensor_cache.cache, ctx.rms_norm_zero_tensor_cache.size,
                             acl_rstd_ne, acl_rstd_nb, GGML_TYPE_F32, GGML_MAX_DIMS,
                             0.0f  //मान
        );

    acl_tensor_ptr acl_xout = ggml_cann_create_tensor(add_गाँठो);

    // Create yOut tensor (final output after RMS normalization)
    acl_tensor_ptr acl_yout = ggml_cann_create_tensor(rms_norm_गाँठो);

    // Call fused ADD + RMS_NORM operator
    GGML_CANN_CALL_ACLNN_OP(ctx, AddRmsNorm, acl_x1.get(), acl_x2.get(), acl_gamma.get(),
                            eps,  // double type
                            acl_yout.get(), acl_rstd.get(), acl_xout.get());
}

शून्य ggml_cann_gated_linear_attn(ggml_backend_cann_context & ctx, ggml_tensor * dst) {
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
    माप_प्रकार  nb_base   = ggml_type_size(k->type);
    माप_प्रकार  nb_qkg[2] = { nb_base, nb_base };
    माप_प्रकार  nb_s[2]   = { nb_base, D * nb_base };
    माप_प्रकार  nb_st[2]  = { nb_s[1], nb_s[0] };
    माप_प्रकार  nb_vo[2]  = { nb_base, D * nb_base };
    माप_प्रकार  nb_q[1]   = { nb_base };

    अचर भिन्न scale = ggml_get_op_params_f32(dst, 0);

    acl_tensor_ptr acl_s     = ggml_cann_create_tensor(s, s->ne, s->nb, 2, ACL_FORMAT_ND);
    acl_tensor_ptr new_state = ggml_cann_create_tensor(dst, s->ne, s->nb, 2, ACL_FORMAT_ND, (B * L * H * D) * nb_base);
    cann_copy(ctx, acl_s.get(), new_state.get());

    लागि (int64_t b = 0; b < B; b++) {
        लागि (int64_t h = 0; h < H; h++) {
            माप_प्रकार         s_offset = (b * (H * D * D) + h * (D * D)) * nb_base;
            // D * D
            acl_tensor_ptr acl_s_new =
                ggml_cann_create_tensor(dst, ne_s, nb_s, 2, ACL_FORMAT_ND, (B * L * H * D) * nb_base + s_offset);
            acl_tensor_ptr acl_s_new_t =
                ggml_cann_create_tensor(dst, ne_st, nb_st, 2, ACL_FORMAT_ND, (B * L * H * D) * nb_base + s_offset);
            लागि (int64_t l = 0; l < L; l++) {
                माप_प्रकार               qkvgo_offset = (b * (L * H * D) + l * (H * D) + h * (D)) * nb_base;
                // D * 1
                acl_tensor_ptr       acl_k = ggml_cann_create_tensor(k, ne_qkg, nb_qkg, 2, ACL_FORMAT_ND, qkvgo_offset);
                acl_tensor_ptr       acl_g = ggml_cann_create_tensor(g, ne_qkg, nb_qkg, 2, ACL_FORMAT_ND, qkvgo_offset);
                // D
                acl_tensor_ptr       acl_q = ggml_cann_create_tensor(q, ne_q, nb_q, 1, ACL_FORMAT_ND, qkvgo_offset);
                // 1 * D
                acl_tensor_ptr       acl_v = ggml_cann_create_tensor(v, ne_vo, nb_vo, 2, ACL_FORMAT_ND, qkvgo_offset);
                // D
                acl_tensor_ptr       acl_o = ggml_cann_create_tensor(dst, ne_q, nb_q, 1, ACL_FORMAT_ND, qkvgo_offset);
                // k ⊗ v
                माप_प्रकार               buf_size = D * D * nb_base;
                ggml_cann_pool_alloc buffer_allocator(ctx.pool(), buf_size);
                acl_tensor_ptr       tmp_tensor = ggml_cann_create_tensor(
                    buffer_allocator.get(), ggml_cann_type_mapping(k->type), nb_base, ne_s, nb_s, 2);
                aclnn_mul(ctx, acl_k.get(), acl_v.get(), tmp_tensor.get());
                //s_new = g ⊗ s_old + k ⊗ v
                aclnn_mul(ctx, acl_s_new.get(), acl_g.get(), nullptr);
                aclnn_add(ctx, acl_s_new.get(), tmp_tensor.get(), nullptr);
                // compute output
                GGML_CANN_CALL_ACLNN_OP(ctx, Mv, acl_s_new_t.get(), acl_q.get(), acl_o.get(), 1);
                aclnn_muls(ctx, acl_o.get(), scale, nullptr, सत्य);
            }
        }
    }
}
