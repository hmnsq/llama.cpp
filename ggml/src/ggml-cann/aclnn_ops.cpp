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

#ضمن "aclnn_ops.h"

#ضمن "ggml-impl.h"
#ضمن "ggml.h"


#ضمن <aclnnop/aclnn_add.h>
#ضمن <aclnnop/aclnn_add_rms_norm.h>
#ضمن <aclnnop/aclnn_addcdiv.h>
#ضمن <aclnnop/aclnn_argmax.h>
#ضمن <aclnnop/aclnn_avgpool2d.h>
#ضمن <aclnnop/aclnn_batch_matmul.h>
#ضمن <aclnnop/aclnn_cast.h>
#ضمن <aclnnop/aclnn_clamp.h>
#ضمن <aclnnop/aclnn_constant_pad_nd.h>
#ضمن <aclnnop/aclnn_convolution.h>
#ضمن <aclnnop/aclnn_copy.h>
#ضمن <aclnnop/aclnn_div.h>
#ضمن <aclnnop/aclnn_elu.h>
#ضمن <aclnnop/aclnn_embedding.h>
#ضمن <aclnnop/aclnn_eq_tensor.h>
#ضمن <aclnnop/aclnn_exp.h>
#ضمن <aclnnop/aclnn_fill_scalar.h>
#ضمن <aclnnop/aclnn_fused_infer_attention_score_v2.h>
#ضمن <aclnnop/aclnn_ger.h>
#ضمن <aclnnop/aclnn_group_norm.h>
#ضمن <aclnnop/aclnn_gather_v2.h>
#ضمن <aclnnop/aclnn_grouped_matmul_v3.h>
#ضمن <aclnnop/aclnn_scatter.h>
#ضمن <aclnnop/aclnn_gt_scalar.h>
#ضمن <aclnnop/aclnn_im2col.h>
#ضمن <aclnnop/aclnn_index_copy.h>
#ضمن <aclnnop/aclnn_index_fill_tensor.h>
#ضمن <aclnnop/aclnn_index_select.h>
#ضمن <aclnnop/aclnn_layer_norm.h>
#ضمن <aclnnop/aclnn_log.h>
#ضمن <aclnnop/aclnn_matmul.h>
#ضمن <aclnnop/aclnn_max_pool.h>
#ضمن <aclnnop/aclnn_mean.h>
#ضمن <aclnnop/aclnn_mm.h>
#ضمن <aclnnop/aclnn_mul.h>
#ضمن <aclnnop/aclnn_mv.h>
#ضمن <aclnnop/aclnn_permute.h>
#ضمن <aclnnop/aclnn_pow.h>
#ضمن <aclnnop/aclnn_pow_tensor_tensor.h>
#ضمن <aclnnop/aclnn_recurrent_gated_delta_rule.h>
#ضمن <aclnnop/aclnn_reduce_sum.h>
#ضمن <aclnnop/aclnn_reflection_pad1d.h>
#ضمن <aclnnop/aclnn_repeat.h>
#ضمن <aclnnop/aclnn_repeat_interleave.h>
#ضمن <aclnnop/aclnn_rms_norm.h>
#ضمن <aclnnop/aclnn_roll.h>
#ضمن <aclnnop/aclnn_softmax.h>
#ضمن <aclnnop/aclnn_softmax_cross_entropy_with_logits.h>
#ضمن <aclnnop/aclnn_sub.h>
#ضمن <aclnnop/aclnn_sum.h>
#ضمن <aclnnop/aclnn_threshold.h>
#ضمن <aclnnop/aclnn_tril.h>
#ضمن <aclnnop/aclnn_triangular_solve.h>
#ضمن <aclnnop/aclnn_triu.h>
#ضمن <aclnnop/aclnn_logical_not.h>
#ضمن <aclnnop/aclnn_masked_fill_scalar.h>
#ضمن <aclnnop/aclnn_upsample_nearest_2d.h>
#ضمن <aclnnop/aclnn_weight_quant_batch_matmul_v2.h>
#ضمن <aclnnop/aclnn_zero.h>
#ضمن <float.h>

#ضمن <cmath>
#ضمن <cstring>
#ضمن <exception>
#ضمن <vector>

#وصف_ڪريو GGML_COMMON_DECL_C

#ضمن "../ggml-common.h"

فراغ bcast_shape(ggml_tensor *    src0,
                 ggml_tensor *    src1,
                 ggml_tensor *    dst,
                 acl_tensor_ptr & acl_src0,
                 acl_tensor_ptr & acl_src1,
                 acl_tensor_ptr & acl_dst) {
    GGML_ASSERT(ggml_are_same_shape(src0, dst) && ggml_can_repeat(src1, src0));
    // Need bcast
    إذا (!ggml_are_same_shape(src0, src1) && ggml_cann_need_bcast(src0, src1)) {
        BCAST_SHAPE(src0, src1)
        acl_src0 = ggml_cann_create_tensor(src0, BCAST_PARAM(src0));
        acl_src1 = ggml_cann_create_tensor(src1, BCAST_PARAM(src1));
        acl_dst  = ggml_cann_create_tensor(dst, BCAST_PARAM(src0));
    } وإلا {
        acl_src0 = ggml_cann_create_tensor(src0);
        acl_src1 = ggml_cann_create_tensor(src1);
        acl_dst  = ggml_cann_create_tensor(dst);
    }
}

فراغ ggml_cann_op_unary(std::function<فراغ(ggml_backend_cann_context &, aclTensor *, aclTensor *)> unary_op,
                        ggml_backend_cann_context &                                                ctx,
                        ggml_tensor *                                                              dst) {
    ggml_tensor * src = dst->src[0];

    acl_tensor_ptr acl_src = ggml_cann_create_tensor(src);
    acl_tensor_ptr acl_dst = ggml_cann_create_tensor(dst);

    unary_op(ctx, acl_src.get(), acl_dst.get());
}

فراغ ggml_cann_op_unary_gated(std::function<فراغ(ggml_backend_cann_context &, aclTensor *, aclTensor *)> unary_op,
                              ggml_backend_cann_context &                                                ctx,
                              ggml_tensor *                                                              dst) {
    ggml_tensor * src0 = dst->src[0];
    ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(ggml_is_contiguous_1(src0));
    GGML_ASSERT(ggml_is_contiguous_1(dst));
    ثابت int32_t swapped = ggml_get_op_params_i32(dst, 1);

    acl_tensor_ptr acl_dst = ggml_cann_create_tensor(dst);
    acl_tensor_ptr acl_src0, acl_src1;
    إذا (src1) {
        GGML_ASSERT(ggml_is_contiguous_1(src1));
        GGML_ASSERT(src0->type == src1->type);

        acl_src0 = ggml_cann_create_tensor(src0);
        acl_src1 = ggml_cann_create_tensor(src1);
    } وإلا {
        int64_t ne[] = { src0->ne[0] / 2, src0->ne[1], src0->ne[2], src0->ne[3] };
        size_t  nb[] = { src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3] };
        acl_src0     = ggml_cann_create_tensor(src0, ne, nb, GGML_MAX_DIMS, ACL_FORMAT_ND, 0);
        acl_src1 = ggml_cann_create_tensor(src0, ne, nb, GGML_MAX_DIMS, ACL_FORMAT_ND, ne[0] * ggml_element_سائيز(src0));
        إذا (swapped) {
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
فراغ ggml_cann_swiglu(ggml_backend_cann_context & ctx, ggml_tensor * dst) {
    خودڪار silu_fn = [](ggml_backend_cann_context & ctx, aclTensor * acl_src, aclTensor * acl_dst) {
        GGML_CANN_CALL_ACLNN_OP(ctx, Silu, acl_src, acl_dst);
    };

    ثابت int32_t swapped = ggml_get_op_params_i32(dst, 1);
    إذا (dst->src[1] != خالي_اشارو || swapped != 0) {
        ggml_cann_op_unary_gated(silu_fn, ctx, dst);
        أرجع;
    }

    // aclnnSwiGlu requires the split dim (src->ne[0]) to be even; fall back otherwise.
    إذا (dst->src[0]->ne[0] % 2 != 0) {
        ggml_cann_op_unary_gated(silu_fn, ctx, dst);
        أرجع;
    }

    ggml_tensor * src0 = dst->src[0];
    size_t elem_سائيز = ggml_element_سائيز(src0);

    // src0 GGML: [2*ne0, ne1, ne2, ne3] → 3D view [2*ne0, ne1, ne2*ne3]
    // CANN reversed: [ne2*ne3, ne1, 2*ne0], split along CANN dim 2 (last).
    int64_t ne0_x2   = src0->ne[0];
    int64_t ne1      = src0->ne[1];
    int64_t ne23     = src0->ne[2] * src0->ne[3];
    int64_t src3d_ne[] = { ne0_x2, ne1, ne23 };
    size_t  src3d_nb[] = { (size_t)src0->nb[0], (size_t)src0->nb[1], (size_t)src0->nb[2] };
    acl_tensor_ptr acl_src = ggml_cann_create_tensor(src0->معلومات, ggml_cann_type_mapping(src0->type),
                                                     elem_سائيز, src3d_ne, src3d_nb, 3);

    // dst GGML: [ne0, ne1, ne2, ne3] → 3D view [ne0, ne1, ne2*ne3]
    int64_t ne0      = dst->ne[0];
    int64_t dst3d_ne[] = { ne0, ne1, ne23 };
    size_t  dst3d_nb[] = { (size_t)dst->nb[0], (size_t)dst->nb[1], (size_t)dst->nb[2] };
    acl_tensor_ptr acl_dst = ggml_cann_create_tensor(dst->معلومات, ggml_cann_type_mapping(dst->type),
                                                     elem_سائيز, dst3d_ne, dst3d_nb, 3);

    // CANN tensor [ne23, ne1, 2*ne0]: split along CANN dim 2 (last) = 2*ne0.
    GGML_CANN_CALL_ACLNN_OP(ctx, SwiGlu, acl_src.get(), (int64_t)2, acl_dst.get());
}

فراغ ggml_cann_swiglu_clamp(ggml_backend_cann_context & ctx, ggml_tensor * dst) {
    ggml_tensor * src0 = dst->src[0];
    ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(ggml_is_contiguous_1(src0));
    GGML_ASSERT(ggml_is_contiguous_1(dst));

    ثابت int32_t  swapped = ggml_get_op_params_i32(dst, 1);
    acl_tensor_ptr acl_gate;
    acl_tensor_ptr acl_مٿي;
    إذا (src1) {
        GGML_ASSERT(ggml_is_contiguous_1(src1));
        GGML_ASSERT(src0->type == src1->type);
        acl_gate = ggml_cann_create_tensor(src0);
        acl_مٿي   = ggml_cann_create_tensor(src1);
    } وإلا {
        int64_t ne[] = { src0->ne[0] / 2, src0->ne[1], src0->ne[2], src0->ne[3] };
        size_t  nb[] = { src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3] };
        acl_gate     = ggml_cann_create_tensor(src0, ne, nb, GGML_MAX_DIMS, ACL_FORMAT_ND, 0);
        acl_مٿي = ggml_cann_create_tensor(src0, ne, nb, GGML_MAX_DIMS, ACL_FORMAT_ND, ne[0] * ggml_element_سائيز(src0));
        إذا (swapped) {
            std::swap(acl_gate, acl_مٿي);
        }
    }

    ggml_cann_pool_alloc عارضي_alloc(ctx.pool(), ggml_nbytes(dst));
    acl_tensor_ptr       acl_عارضي = ggml_cann_create_tensor(عارضي_alloc.get(), ggml_cann_type_mapping(dst->type),
                                                            ggml_element_سائيز(dst), dst->ne, dst->nb, GGML_MAX_DIMS);
    acl_tensor_ptr       acl_dst  = ggml_cann_create_tensor(dst);

    ثابت عائم    حد_بندي        = ggml_get_op_params_f32(dst, 3);
    عائم          min_gate     = -INFINITY;
    عائم          min_مٿي       = -حد_بندي;
    عائم          max_قدر    = حد_بندي;
    acl_scalar_ptr acl_min_gate = ggml_cann_create_scalar(&min_gate, ACL_FLOAT);
    acl_scalar_ptr acl_min_مٿي   = ggml_cann_create_scalar(&min_مٿي, ACL_FLOAT);
    acl_scalar_ptr acl_حد_بندي    = ggml_cann_create_scalar(&max_قدر, ACL_FLOAT);

    GGML_CANN_CALL_ACLNN_OP(ctx, Clamp, acl_gate.get(), acl_min_gate.get(), acl_حد_بندي.get(), acl_عارضي.get());
    GGML_CANN_CALL_ACLNN_OP(ctx, Silu, acl_عارضي.get(), acl_dst.get());
    GGML_CANN_CALL_ACLNN_OP(ctx, Clamp, acl_مٿي.get(), acl_min_مٿي.get(), acl_حد_بندي.get(), acl_عارضي.get());
    GGML_CANN_CALL_ACLNN_OP(ctx, InplaceMul, acl_dst.get(), acl_عارضي.get());
}

// Fused GeGLU using aclnnGeGluV3: splits input along ne[0] (CANN last dim),
// activates the LEFT half with GELU, multiplies by right half.
// approximate: 0=tanh, 1=none(erf). activateLeft=true matches GGML convention.
// outGelu is a required-but-discard output buffer.
//
// Falls back to the generic two-kernel path when src[1] != nullptr (two
// independent halves) or swapped != 0 (reversed activation order), as
// aclnnGeGluV3 only handles the single interleaved tensor in standard order.
فراغ ggml_cann_geglu(ggml_backend_cann_context & ctx, ggml_tensor * dst, int64_t approximate) {
    خودڪار gelu_fn = [](ggml_backend_cann_context & ctx, aclTensor * acl_src, aclTensor * acl_dst) {
        GGML_CANN_CALL_ACLNN_OP(ctx, Gelu, acl_src, acl_dst);
    };

    ثابت int32_t swapped = ggml_get_op_params_i32(dst, 1);
    إذا (dst->src[1] != خالي_اشارو || swapped != 0) {
        ggml_cann_op_unary_gated(gelu_fn, ctx, dst);
        أرجع;
    }

    // aclnnGeGluV3 requires the split dim (src->ne[0]) to be even; fall back otherwise.
    إذا (dst->src[0]->ne[0] % 2 != 0) {
        ggml_cann_op_unary_gated(gelu_fn, ctx, dst);
        أرجع;
    }

    ggml_tensor * src0 = dst->src[0];
    acl_tensor_ptr acl_src = ggml_cann_create_tensor(src0);
    acl_tensor_ptr acl_dst = ggml_cann_create_tensor(dst);

    // Allocate a temporary buffer for the required outGelu output (same shape as dst).
    // Build contiguous strides since the pool allocation is a fresh buffer.
    size_t  elem_سائيز    = ggml_element_سائيز(dst);
    int64_t ne[GGML_MAX_DIMS] = { dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3] };
    size_t  nb[GGML_MAX_DIMS];
    nb[0] = elem_سائيز;
    لكل (صحيح i = 1; i < GGML_MAX_DIMS; i++) {
        nb[i] = nb[i - 1] * ne[i - 1];
    }
    size_t gelu_ٻاهريون_سائيز = nb[GGML_MAX_DIMS - 1] * ne[GGML_MAX_DIMS - 1];
    ggml_cann_pool_alloc gelu_ٻاهريون_alloc(ctx.pool(), gelu_ٻاهريون_سائيز);

    acl_tensor_ptr acl_gelu_ٻاهريون = ggml_cann_create_tensor(
        gelu_ٻاهريون_alloc.get(), ggml_cann_type_mapping(dst->type), elem_سائيز, ne, nb, GGML_MAX_DIMS);
    // V3 adds activateLeft param; true → Gelu(left)*right, matching GGML convention.
    // GGML dim 0 → CANN last dim (index GGML_MAX_DIMS-1 = 3 for 4D tensor).
    GGML_CANN_CALL_ACLNN_OP(ctx, GeGluV3, acl_src.get(), (int64_t)(GGML_MAX_DIMS - 1), approximate, سچ,
                             acl_dst.get(), acl_gelu_ٻاهريون.get());
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
ساكن فراغ aclnn_repeat(ggml_backend_cann_context & ctx,
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
ساكن فراغ aclnn_cast(ggml_backend_cann_context & ctx,
                       aclTensor *                 acl_src,
                       aclTensor *                 acl_dst,
                       acl_معلومات_Type                 cast_معلومات_type) {
    GGML_CANN_CALL_ACLNN_OP(ctx, Cast, acl_src, cast_معلومات_type, acl_dst);
}

فراغ ggml_cann_repeat(ggml_backend_cann_context & ctx, ggml_tensor * dst) {
    ggml_tensor * src = dst->src[0];
    GGML_ASSERT(ggml_can_repeat(src, dst));

    acl_tensor_ptr acl_src = ggml_cann_create_tensor(src);
    acl_tensor_ptr acl_dst = ggml_cann_create_tensor(dst);

    int64_t repeatsArray[] = { dst->ne[3] / src->ne[3], dst->ne[2] / src->ne[2], dst->ne[1] / src->ne[1],
                               dst->ne[0] / src->ne[0] };

    aclnn_repeat(ctx, acl_src.get(), acl_dst.get(), repeatsArray);
}

فراغ aclnn_جوڙيو(ggml_backend_cann_context & ctx, aclTensor * acl_src0, aclTensor * acl_src1, aclTensor * acl_dst) {
    عائم          alpha_قدر = 1.0f;
    acl_scalar_ptr alpha      = ggml_cann_create_scalar(&alpha_قدر, acl_معلومات_Type::ACL_FLOAT);
    إذا (acl_dst != خالي_اشارو) {
        GGML_CANN_CALL_ACLNN_OP(ctx, جوڙيو, acl_src0, acl_src1, alpha.get(), acl_dst);
    } وإلا {
        GGML_CANN_CALL_ACLNN_OP(ctx, Inplace_جوڙيو, acl_src0, acl_src1, alpha.get());
    }
}

فراغ aclnn_sub(ggml_backend_cann_context & ctx, aclTensor * acl_src0, aclTensor * acl_src1, aclTensor * acl_dst) {
    عائم          alpha_قدر = 1.0f;
    acl_scalar_ptr alpha      = ggml_cann_create_scalar(&alpha_قدر, acl_معلومات_Type::ACL_FLOAT);
    إذا (acl_dst != خالي_اشارو) {
        GGML_CANN_CALL_ACLNN_OP(ctx, Sub, acl_src0, acl_src1, alpha.get(), acl_dst);
    } وإلا {
        GGML_CANN_CALL_ACLNN_OP(ctx, InplaceSub, acl_src0, acl_src1, alpha.get());
    }
}

فراغ aclnn_mul(ggml_backend_cann_context & ctx, aclTensor * acl_src, aclTensor * acl_other, aclTensor * acl_dst) {
    إذا (acl_dst != خالي_اشارو) {
        GGML_CANN_CALL_ACLNN_OP(ctx, Mul, acl_src, acl_other, acl_dst);
    } وإلا {
        GGML_CANN_CALL_ACLNN_OP(ctx, InplaceMul, acl_src, acl_other);
    }
}

فراغ aclnn_div(ggml_backend_cann_context & ctx, aclTensor * acl_src, aclTensor * acl_other, aclTensor * acl_dst) {
    إذا (acl_dst != خالي_اشارو) {
        GGML_CANN_CALL_ACLNN_OP(ctx, Div, acl_src, acl_other, acl_dst);
    } وإلا {
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
ساكن فراغ aclnn_muls(ggml_backend_cann_context & ctx,
                       aclTensor *                 acl_src,
                       عائم                       scale,
                       aclTensor *                 acl_dst,
                       منطقي                        inplace) {
    acl_scalar_ptr acl_scale = ggml_cann_create_scalar(&scale, acl_معلومات_Type::ACL_FLOAT);
    إذا (inplace) {
        GGML_CANN_CALL_ACLNN_OP(ctx, InplaceMuls, acl_src, acl_scale.get());
    } وإلا {
        GGML_CANN_CALL_ACLNN_OP(ctx, Muls, acl_src, acl_scale.get(), acl_dst);
    }
}

فراغ ggml_cann_leaky_relu(ggml_backend_cann_context & ctx, ggml_tensor * dst) {
    ggml_tensor * src = dst->src[0];

    acl_tensor_ptr acl_src = ggml_cann_create_tensor(src);
    acl_tensor_ptr acl_dst = ggml_cann_create_tensor(dst);

    عائم negative_لاهي;
    يادگيري_نقل(&negative_لاهي, dst->op_params, حجم(عائم));
    acl_scalar_ptr acl_negative_لاهي = ggml_cann_create_scalar(&negative_لاهي, acl_معلومات_Type::ACL_FLOAT);

    GGML_CANN_CALL_ACLNN_OP(ctx, LeakyRelu, acl_src.get(), acl_negative_لاهي.get(), acl_dst.get());
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
ساكن فراغ aclnn_concat(ggml_backend_cann_context & ctx,
                         aclTensorList *             tensorList,
                         aclTensor *                 acl_dst,
                         int64_t                     concat_dim) {
    GGML_CANN_CALL_ACLNN_OP(ctx, Cat, tensorList, concat_dim, acl_dst);
}

فراغ ggml_cann_concat(ggml_backend_cann_context & ctx, ggml_tensor * dst) {
    ggml_tensor *  src0     = dst->src[0];
    ggml_tensor *  src1     = dst->src[1];
    acl_tensor_ptr acl_src0 = ggml_cann_create_tensor(src0);
    acl_tensor_ptr acl_src1 = ggml_cann_create_tensor(src1);
    acl_tensor_ptr acl_dst  = ggml_cann_create_tensor(dst);

    ثابت int32_t dim = ggml_get_op_params_i32(dst, 0);

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
ساكن فراغ aclnn_arange(ggml_backend_cann_context & ctx,
                         aclTensor *                 acl_dst,
                         عائم                       start,
                         عائم                       stop,
                         عائم                       قدم,
                         int64_t                     n_elements) {
    int64_t قدمن = (int64_t) std::ceil((stop - start) / قدم);
    GGML_ASSERT(n_elements == قدمن);

    acl_scalar_ptr acl_شروعات = ggml_cann_create_scalar(&start, acl_معلومات_Type::ACL_FLOAT);
    acl_scalar_ptr acl_پڇاڙي   = ggml_cann_create_scalar(&stop, acl_معلومات_Type::ACL_FLOAT);
    acl_scalar_ptr acl_قدم  = ggml_cann_create_scalar(&قدم, acl_معلومات_Type::ACL_FLOAT);

    GGML_CANN_CALL_ACLNN_OP(ctx, Arange, acl_شروعات.get(), acl_پڇاڙي.get(), acl_قدم.get(), acl_dst);
}

فراغ ggml_cann_arange(ggml_backend_cann_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    acl_tensor_ptr acl_dst = ggml_cann_create_tensor(dst);

    int64_t n_elements = ggml_nelements(dst);
    عائم   start;
    عائم   stop;
    عائم   قدم;
    يادگيري_نقل(&start, (عائم *) dst->op_params + 0, حجم(عائم));
    يادگيري_نقل(&stop, (عائم *) dst->op_params + 1, حجم(عائم));
    يادگيري_نقل(&قدم, (عائم *) dst->op_params + 2, حجم(عائم));

    aclnn_arange(ctx, acl_dst.get(), start, stop, قدم, n_elements);
}

فراغ ggml_cann_clamp(ggml_backend_cann_context & ctx, ggml_tensor * dst) {
    ggml_tensor * src = dst->src[0];

    عائم min;
    عائم max;
    يادگيري_نقل(&min, dst->op_params, حجم(عائم));
    يادگيري_نقل(&max, (عائم *) dst->op_params + 1, حجم(عائم));

    acl_tensor_ptr acl_src = ggml_cann_create_tensor(src);
    acl_tensor_ptr acl_dst = ggml_cann_create_tensor(dst);

    acl_scalar_ptr acl_min = ggml_cann_create_scalar(&min, acl_معلومات_Type::ACL_FLOAT);
    acl_scalar_ptr acl_max = ggml_cann_create_scalar(&max, acl_معلومات_Type::ACL_FLOAT);

    GGML_CANN_CALL_ACLNN_OP(ctx, Clamp, acl_src.get(), acl_min.get(), acl_max.get(), acl_dst.get());
}

فراغ ggml_cann_scale(ggml_backend_cann_context & ctx, ggml_tensor * dst) {
    ggml_tensor * src = dst->src[0];

    // scale factor
    عائم v;
    يادگيري_نقل(&v, dst->op_params, حجم(عائم));

    acl_scalar_ptr scale   = ggml_cann_create_scalar(&v, acl_معلومات_Type::ACL_FLOAT);
    acl_tensor_ptr acl_src = ggml_cann_create_tensor(src);
    acl_tensor_ptr acl_dst = ggml_cann_create_tensor(dst);

    GGML_CANN_CALL_ACLNN_OP(ctx, Muls, acl_src.get(), scale.get(), acl_dst.get());
}

فراغ ggml_cann_argsort(ggml_backend_cann_context & ctx, ggml_tensor * dst) {
    ggml_tensor *        src   = dst->src[0];
    تعداد ggml_ترتيب_ڏيو_ترتيب ترتيب = (تعداد ggml_ترتيب_ڏيو_ترتيب) dst->op_params[0];

    acl_tensor_ptr       acl_src = ggml_cann_create_tensor(src);
    acl_tensor_ptr       acl_dst = ggml_cann_create_tensor(dst);
    ggml_cann_pool_alloc عارضي_بفر_allocator(ctx.pool(), ggml_nelements(dst) * حجم(int64_t));
    فراغ *               بفر = عارضي_بفر_allocator.get();
    acl_tensor_ptr       tmp_tensor =
        ggml_cann_create_tensor(بفر, ACL_INT64, ggml_type_سائيز(dst->type), dst->ne, dst->nb, GGML_MAX_DIMS);
    GGML_CANN_CALL_ACLNN_OP(ctx, Argsort, acl_src.get(), -1, (ترتيب == GGML_ترتيب_ڏيو_ترتيب_DESC ? سچ : خطأ),
                            tmp_tensor.get());
    GGML_CANN_CALL_ACLNN_OP(ctx, Cast, tmp_tensor.get(), ggml_cann_type_mapping(dst->type), acl_dst.get());
}

فراغ ggml_cann_norm(ggml_backend_cann_context & ctx, ggml_tensor * dst) {
    ggml_tensor * src = dst->src[0];

    acl_tensor_ptr acl_src = ggml_cann_create_tensor(src);
    acl_tensor_ptr acl_dst = ggml_cann_create_tensor(dst);

    عائم eps;
    يادگيري_نقل(&eps, dst->op_params, حجم(عائم));

    std::vector<int64_t> norm_معلومات = { dst->ne[0] };
    acl_int_array_ptr    norm     = ggml_cann_create_int_array(norm_معلومات.معلومات(), norm_معلومات.size());
    GGML_CANN_CALL_ACLNN_OP(ctx, LayerNorm, acl_src.get(), norm.get(), خالي_اشارو, خالي_اشارو, eps, acl_dst.get(), خالي_اشارو,
                            خالي_اشارو);
}

فراغ ggml_cann_l2_norm(ggml_backend_cann_context & ctx, ggml_tensor * dst) {
    ggml_tensor * src = dst->src[0];

    عائم eps;
    يادگيري_نقل(&eps, dst->op_params, حجم(عائم));

    acl_tensor_ptr acl_src = ggml_cann_create_tensor(src);
    acl_tensor_ptr acl_dst = ggml_cann_create_tensor(dst);

    size_t               type_سائيز = ggml_type_سائيز(src->type);
    int64_t              n_بائيٽون   = src->ne[3] * src->ne[2] * src->ne[1] * type_سائيز;
    ggml_cann_pool_alloc عارضي_بفر_allocator(ctx.pool(), n_بائيٽون);
    فراغ *               بفر = عارضي_بفر_allocator.get();

    int64_t norm_ne[] = { 1, src->ne[1], src->ne[2], src->ne[3] };
    size_t  norm_nb[GGML_MAX_DIMS];
    norm_nb[0] = حجم(عائم);
    لكل (صحيح i = 1; i < GGML_MAX_DIMS; ++i) {
        norm_nb[i] = norm_nb[i - 1] * norm_ne[i - 1];
    }
    acl_tensor_ptr acl_norm = ggml_cann_create_tensor(بفر, ACL_FLOAT, حجم(عائم), norm_ne, norm_nb, GGML_MAX_DIMS);

    std::vector<int64_t> norm_dims  = { 3 };
    acl_int_array_ptr    dims_array = ggml_cann_create_int_array(norm_dims.معلومات(), norm_dims.size());

    عائم          p_قدر  = 2.0f;
    acl_scalar_ptr p_scalar = ggml_cann_create_scalar(&p_قدر, acl_معلومات_Type::ACL_FLOAT);
    GGML_CANN_CALL_ACLNN_OP(ctx, Norm, acl_src.get(), p_scalar.get(), dims_array.get(), سچ, acl_norm.get());

    ggml_cann_pool_alloc clamp_بفر_allocator(ctx.pool());
    acl_tensor_ptr       acl_clamped;

    إذا (eps > 0.0f) {
        فراغ *         clamp_buf  = clamp_بفر_allocator.alloc(n_بائيٽون);
        acl_clamped               = ggml_cann_create_tensor(clamp_buf, ACL_FLOAT, حجم(عائم), norm_ne, norm_nb, GGML_MAX_DIMS);
        acl_scalar_ptr eps_scalar = ggml_cann_create_scalar(&eps, acl_معلومات_Type::ACL_FLOAT);
        GGML_CANN_CALL_ACLNN_OP(ctx, ClampMin, acl_norm.get(), eps_scalar.get(), acl_clamped.get());
    }

    aclTensor * acl_div_داخلا = acl_clamped ? acl_clamped.get() : acl_norm.get();
    GGML_CANN_CALL_ACLNN_OP(ctx, Div, acl_src.get(), acl_div_داخلا, acl_dst.get());
}

فراغ ggml_cann_cross_entropy_loss(ggml_backend_cann_context & ctx, ggml_tensor * dst) {
    ggml_tensor * src0 = dst->src[0];
    ggml_tensor * src1 = dst->src[1];

    ثابت int64_t nc = src0->ne[0];
    ثابت int64_t nr = ggml_nrows(src0);

    int64_t logits_ne[] = { nc, nr };
    size_t  logits_nb[2];
    logits_nb[0]              = ggml_type_سائيز(src0->type);
    logits_nb[1]              = logits_nb[0] * logits_ne[0];
    acl_tensor_ptr acl_logits = ggml_cann_create_tensor(src0->معلومات, ACL_FLOAT, حجم(عائم), logits_ne, logits_nb, 2);

    int64_t labels_ne[] = { nc, nr };
    size_t  labels_nb[2];
    labels_nb[0]              = ggml_type_سائيز(src1->type);
    labels_nb[1]              = labels_nb[0] * labels_ne[0];
    acl_tensor_ptr acl_labels = ggml_cann_create_tensor(src1->معلومات, ACL_FLOAT, حجم(عائم), labels_ne, labels_nb, 2);

    size_t               loss_في_sample_type_سائيز = حجم(عائم);
    int64_t              loss_في_sample_n_بائيٽون   = nr * loss_في_sample_type_سائيز;
    ggml_cann_pool_alloc loss_في_sample_allocator(ctx.pool(), loss_في_sample_n_بائيٽون);
    فراغ *               loss_في_sample_بفر = loss_في_sample_allocator.get();

    int64_t loss_في_sample_ne[] = { nr };
    size_t  loss_في_sample_nb[1];
    loss_في_sample_nb[0] = loss_في_sample_type_سائيز;
    acl_tensor_ptr acl_loss_في_sample = ggml_cann_create_tensor(
        loss_في_sample_بفر, ACL_FLOAT, loss_في_sample_type_سائيز, loss_في_sample_ne, loss_في_sample_nb, 1);

    size_t               backprop_n_بائيٽون = nr * nc * حجم(عائم);
    ggml_cann_pool_alloc backprop_allocator(ctx.pool(), backprop_n_بائيٽون);
    فراغ *               backprop_بفر = backprop_allocator.get();
    acl_tensor_ptr acl_backprop = ggml_cann_create_tensor(backprop_بفر, ACL_FLOAT, حجم(عائم), logits_ne, logits_nb, 2);

    GGML_CANN_CALL_ACLNN_OP(ctx, SoftmaxCrossEntropyWithLogits, acl_logits.get(), acl_labels.get(),
                            acl_loss_في_sample.get(), acl_backprop.get());

    size_t               ڪل_جوڙ_type_سائيز = حجم(عائم);
    int64_t              ڪل_جوڙ_n_بائيٽون   = 1 * ڪل_جوڙ_type_سائيز;
    ggml_cann_pool_alloc ڪل_جوڙ_allocator(ctx.pool(), ڪل_جوڙ_n_بائيٽون);
    فراغ *               ڪل_جوڙ_بفر = ڪل_جوڙ_allocator.get();

    int64_t ڪل_جوڙ_ne[] = { 1 };
    size_t  ڪل_جوڙ_nb[1];
    ڪل_جوڙ_nb[0] = ڪل_جوڙ_type_سائيز;

    acl_tensor_ptr acl_ڪل_جوڙ =
        ggml_cann_create_tensor(ڪل_جوڙ_بفر, ACL_FLOAT, ڪل_جوڙ_type_سائيز, ڪل_جوڙ_ne, ڪل_جوڙ_nb, 1);

    std::vector<int64_t> ڪل_جوڙ_dims    = { 0 };
    acl_int_array_ptr ڪل_جوڙ_dims_array = ggml_cann_create_int_array(ڪل_جوڙ_dims.معلومات(), ڪل_جوڙ_dims.size());
    منطقي              keep_dims            = خطأ;

    GGML_CANN_CALL_ACLNN_OP(ctx, Reduce_جوڙ, acl_loss_في_sample.get(), ڪل_جوڙ_dims_array.get(), keep_dims, ACL_FLOAT,
                            acl_ڪل_جوڙ.get());

    عائم          value        = 1.0f / static_cast<عائم>(nr);
    acl_scalar_ptr scale_ضرب_ڪندڙ = ggml_cann_create_scalar(&value, acl_معلومات_Type::ACL_FLOAT);
    acl_tensor_ptr acl_dst =
        ggml_cann_create_tensor(dst->معلومات, ACL_FLOAT, حجم(عائم), ڪل_جوڙ_ne, ڪل_جوڙ_nb, 1);

    GGML_CANN_CALL_ACLNN_OP(ctx, Muls, acl_ڪل_جوڙ.get(), scale_ضرب_ڪندڙ.get(), acl_dst.get());
}

فراغ ggml_cann_گروهه_norm(ggml_backend_cann_context & ctx, ggml_tensor * dst) {
    ggml_tensor * src = dst->src[0];

    acl_tensor_ptr acl_src = ggml_cann_create_tensor(src);
    acl_tensor_ptr acl_dst = ggml_cann_create_tensor(dst);

    صحيح n_groups = dst->op_params[0];

    عائم eps;
    يادگيري_نقل(&eps, dst->op_params + 1, حجم(عائم));

    int64_t N   = src->ne[3];
    int64_t C   = src->ne[2];
    int64_t HxW = src->ne[1] * src->ne[0];

    size_t  type_سائيز = ggml_type_سائيز(src->type);
    int64_t ne[]      = { n_groups, N };
    size_t  nb[]      = { type_سائيز, type_سائيز * n_groups };
    size_t  n_بائيٽون   = N * n_groups;

    ggml_cann_pool_alloc عارضي_بفر_allocator(ctx.pool(), n_بائيٽون * 2);
    فراغ *               بفر       = عارضي_بفر_allocator.get();
    acl_tensor_ptr       acl_سراسري_ٻاهريون = ggml_cann_create_tensor(بفر, ACL_FLOAT, type_سائيز, ne, nb, ACL_FORMAT_ND);
    acl_tensor_ptr       acl_rstd_ٻاهريون =
        ggml_cann_create_tensor((حرف *) بفر + n_بائيٽون, ACL_FLOAT, type_سائيز, ne, nb, ACL_FORMAT_ND);

    GGML_CANN_CALL_ACLNN_OP(ctx, گروهه_Norm, acl_src.get(), خالي_اشارو, خالي_اشارو, N, C, HxW, n_groups, eps, acl_dst.get(),
                            acl_سراسري_ٻاهريون.get(), acl_rstd_ٻاهريون.get());
}

فراغ ggml_cann_سيٽ(ggml_backend_cann_context & ctx, ggml_tensor * dst) {
    ggml_tensor * src0 = dst->src[0];
    ggml_tensor * src1 = dst->src[1];

    size_t nb1     = ((int32_t *) dst->op_params)[0];
    size_t nb2     = ((int32_t *) dst->op_params)[1];
    size_t nb3     = ((int32_t *) dst->op_params)[2];
    size_t وٿي  = ((int32_t *) dst->op_params)[3];
    منطقي   inplace = (منطقي) ((int32_t *) dst->op_params)[4];

    size_t param_nb[] = { ggml_element_سائيز(src0), nb1, nb2, nb3 };

    // Create a view of dst at the target offset with src1's dimensions
    acl_tensor_ptr acl_dst  = ggml_cann_create_tensor(dst, src1->ne, param_nb, GGML_MAX_DIMS, ACL_FORMAT_ND, وٿي);
    acl_tensor_ptr acl_src1 = ggml_cann_create_tensor(src1);

    إذا (!inplace) {
        // First copy src0 to dst entirely
        size_t cpy_سائيز = ggml_nbytes(dst);
        ACL_CHECK(
            aclrtMemcpyAsync(dst->معلومات, cpy_سائيز, src0->معلومات, cpy_سائيز, ACL_MEMCPY_DEVICE_منزل_DEVICE, ctx.stream()));
    }

    // Copy src1 into the target region of dst
    GGML_CANN_CALL_ACLNN_OP(ctx, InplaceCopy, acl_dst.get(), acl_src1.get());
}

فراغ ggml_cann_acc(ggml_backend_cann_context & ctx, ggml_tensor * dst) {
    ggml_tensor * src0 = dst->src[0];
    ggml_tensor * src1 = dst->src[1];

    size_t nb1     = ((int32_t *) dst->op_params)[0];
    size_t nb2     = ((int32_t *) dst->op_params)[1];
    size_t nb3     = ((int32_t *) dst->op_params)[2];
    size_t وٿي  = ((int32_t *) dst->op_params)[3];
    منطقي   inplace = (منطقي) ((int32_t *) dst->op_params)[4];

    size_t param_nb[] = { ggml_element_سائيز(src0), nb1, nb2, nb3 };

    acl_tensor_ptr acl_dst  = ggml_cann_create_tensor(dst, src1->ne, param_nb, GGML_MAX_DIMS, ACL_FORMAT_ND, وٿي);
    acl_tensor_ptr acl_src1 = ggml_cann_create_tensor(src1);

    acl_scalar_ptr alpha      = خالي_اشارو;
    عائم          alpha_قدر = 1.0f;
    alpha                     = ggml_cann_create_scalar(&alpha_قدر, acl_معلومات_Type::ACL_FLOAT);

    إذا (!inplace) {
        size_t cpy_سائيز = ggml_nbytes(dst);
        ACL_CHECK(
            aclrtMemcpyAsync(dst->معلومات, cpy_سائيز, src0->معلومات, cpy_سائيز, ACL_MEMCPY_DEVICE_منزل_DEVICE, ctx.stream()));
        acl_tensor_ptr acl_src0 =
            ggml_cann_create_tensor(src0, src1->ne, src0->nb, GGML_MAX_DIMS, ACL_FORMAT_ND, وٿي);

        GGML_CANN_CALL_ACLNN_OP(ctx, جوڙيو, acl_src0.get(), acl_src1.get(), alpha.get(), acl_dst.get());
    } وإلا {
        GGML_CANN_CALL_ACLNN_OP(ctx, Inplace_جوڙيو, acl_dst.get(), acl_src1.get(), alpha.get());
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
ساكن فراغ aclnn_reduce_جوڙ(ggml_backend_cann_context & ctx, ggml_tensor * dst, int64_t * dim, size_t dim_سائيز) {
    GGML_ASSERT(dst->ne[0] == 1);
    ggml_tensor *     src         = dst->src[0];
    acl_tensor_ptr    acl_src     = ggml_cann_create_tensor(src);
    acl_tensor_ptr    acl_dst     = ggml_cann_create_tensor(dst);
    acl_int_array_ptr reduce_dims = ggml_cann_create_int_array(dim, dim_سائيز);

    GGML_CANN_CALL_ACLNN_OP(ctx, Reduce_جوڙ, acl_src.get(), reduce_dims.get(), سچ, ggml_cann_type_mapping(dst->type),
                            acl_dst.get());
}

فراغ ggml_cann_جوڙ_صفون(ggml_backend_cann_context & ctx, ggml_tensor * dst) {
    int64_t reduce_dims[] = { 3 };
    aclnn_reduce_جوڙ(ctx, dst, reduce_dims, 1);
}

فراغ ggml_cann_جوڙ(ggml_backend_cann_context & ctx, ggml_tensor * dst) {
    int64_t reduce_dims[] = { 0, 1, 2, 3 };
    aclnn_reduce_جوڙ(ctx, dst, reduce_dims, 4);
}

فراغ ggml_cann_cumsum(ggml_backend_cann_context & ctx, ggml_tensor * dst) {
    ggml_tensor * src = dst->src[0];
    acl_tensor_ptr acl_src = ggml_cann_create_tensor(src);
    acl_tensor_ptr acl_dst = ggml_cann_create_tensor(dst);
    // GGML cumsum operates along dim 0 (innermost / ne[0]).
    // ggml_cann_create_tensor reverses dimensions to [ne3,ne2,ne1,ne0],
    // so GGML dim 0 maps to CANN dim 3 (the last dim of the 4-D tensor).
    GGML_CANN_CALL_ACLNN_OP(ctx, Cumsum, acl_src.get(), (int64_t)3,
                            ggml_cann_type_mapping(dst->type), acl_dst.get());
}

فراغ ggml_cann_solve_tri(ggml_backend_cann_context & ctx, ggml_tensor * dst) {
    ggml_tensor * src0 = dst->src[0];  // A: [N, N, B2, B3] lower triangular
    ggml_tensor * src1 = dst->src[1];  // B: [K, N, B2, B3]

    acl_tensor_ptr acl_a = ggml_cann_create_tensor(src0);
    acl_tensor_ptr acl_b = ggml_cann_create_tensor(src1);
    acl_tensor_ptr acl_x = ggml_cann_create_tensor(dst);

    // mOut: triangular copy of A (required output), same shape as A.
    ثابت size_t a_بائيٽون = ggml_nbytes(src0);
    ggml_cann_pool_alloc m_alloc(ctx.pool(), a_بائيٽون);
    acl_tensor_ptr acl_m = ggml_cann_create_tensor(
        m_alloc.get(), ggml_cann_type_mapping(src0->type),
        ggml_type_سائيز(src0->type), src0->ne, src0->nb, GGML_MAX_DIMS);

    // Solve AX = B: upper=false (lower tri), transpose=false, unitriangular=false.
    GGML_CANN_CALL_ACLNN_OP(ctx, TriangularSolve,
        acl_b.get(), acl_a.get(), خطأ, خطأ, خطأ,
        acl_x.get(), acl_m.get());
}

فراغ ggml_cann_diag(ggml_backend_cann_context & ctx, ggml_tensor * dst) {
    ggml_tensor * src = dst->src[0];

    GGML_ASSERT(src->ne[1] == 1);

    ثابت int64_t N       = src->ne[0];
    ثابت int64_t n_batch = src->ne[2] * src->ne[3];
    ثابت size_t  nb_f32  = حجم(عائم);

    // Fill dst with zeros.
    acl_tensor_ptr acl_dst = ggml_cann_create_tensor(dst);
    {
        عائم          zero = 0.0f;
        acl_scalar_ptr acl_zero = ggml_cann_create_scalar(&zero, ACL_FLOAT);
        GGML_CANN_CALL_ACLNN_OP(ctx, InplaceFillScalar, acl_dst.get(), acl_zero.get());
    }

    // Copy src vector onto the diagonal of dst via strided views.
    // src viewed as [N, n_batch], contiguous strides.
    int64_t ne_vec[2]      = { N, n_batch };
    size_t  nb_src_vec[2]  = { nb_f32, N * nb_f32 };
    // dst diagonal view: stride (N+1)*4 steps along the diagonal.
    size_t  nb_dst_diag[2] = { (N + 1) * nb_f32, N * N * nb_f32 };

    acl_tensor_ptr acl_src_vec  = ggml_cann_create_tensor(src->معلومات, ACL_FLOAT, nb_f32, ne_vec, nb_src_vec, 2);
    acl_tensor_ptr acl_dst_diag = ggml_cann_create_tensor(dst->معلومات, ACL_FLOAT, nb_f32, ne_vec, nb_dst_diag, 2);

    GGML_CANN_CALL_ACLNN_OP(ctx, InplaceCopy, acl_dst_diag.get(), acl_src_vec.get());
}

فراغ ggml_cann_fill(ggml_backend_cann_context & ctx, ggml_tensor * dst) {
    عائم c = ggml_get_op_params_f32(dst, 0);

    acl_tensor_ptr acl_dst = ggml_cann_create_tensor(dst);
    acl_scalar_ptr acl_c   = ggml_cann_create_scalar(&c, ACL_FLOAT);
    GGML_CANN_CALL_ACLNN_OP(ctx, InplaceFillScalar, acl_dst.get(), acl_c.get());
}

فراغ ggml_cann_tri(ggml_backend_cann_context & ctx, ggml_tensor * dst) {
    ggml_tensor * src = dst->src[0];

    ثابت int64_t S       = src->ne[0];
    ثابت int64_t n_batch = src->ne[2] * src->ne[3];
    ثابت size_t  nb_f32  = حجم(عائم);

    int64_t ne3d[3] = { S, S, n_batch };
    size_t  nb3d[3] = { nb_f32, S * nb_f32, S * S * nb_f32 };

    ثابت ggml_tri_type ttype = (ggml_tri_type) ggml_get_op_params_i32(dst, 0);

    acl_tensor_ptr acl_src = ggml_cann_create_tensor(src->معلومات, ACL_FLOAT, nb_f32, ne3d, nb3d, 3);
    acl_tensor_ptr acl_dst = ggml_cann_create_tensor(dst->معلومات, ACL_FLOAT, nb_f32, ne3d, nb3d, 3);

    بدل (ttype) {
        حالة GGML_TRI_TYPE_هيٺاهون:
            // Tril(-1): preserve row > col (strict lower), zero upper + diagonal.
            GGML_CANN_CALL_ACLNN_OP(ctx, Tril, acl_src.get(), (int64_t)-1, acl_dst.get());
            اكسر;
        حالة GGML_TRI_TYPE_مٿانهون_DIAG:
            // Triu(0): preserve row <= col (upper + diagonal), zero strict lower.
            GGML_CANN_CALL_ACLNN_OP(ctx, Triu, acl_src.get(), (int64_t)0, acl_dst.get());
            اكسر;
        حالة GGML_TRI_TYPE_مٿانهون:
            // Triu(1): preserve row < col (strict upper), zero lower + diagonal.
            GGML_CANN_CALL_ACLNN_OP(ctx, Triu, acl_src.get(), (int64_t)1, acl_dst.get());
            اكسر;
        حالة GGML_TRI_TYPE_هيٺاهون_DIAG:
            // Tril(0): preserve row >= col (lower + diagonal), zero strict upper.
            GGML_CANN_CALL_ACLNN_OP(ctx, Tril, acl_src.get(), (int64_t)0, acl_dst.get());
            اكسر;
        افتراضي:
            GGML_ABORT("unsupported tri type");
    }
}

فراغ ggml_cann_upsample_nearest2d(ggml_backend_cann_context & ctx, ggml_tensor * dst) {
    ggml_tensor *  src     = dst->src[0];
    acl_tensor_ptr acl_src = ggml_cann_create_tensor(src, خالي_اشارو, خالي_اشارو, 0, ACL_FORMAT_NCHW);
    acl_tensor_ptr acl_dst = ggml_cann_create_tensor(dst, خالي_اشارو, خالي_اشارو, 0, ACL_FORMAT_NCHW);

    std::vector<int64_t> output_سائيز{ dst->ne[1], dst->ne[0] };
    acl_int_array_ptr    output_سائيز_array = ggml_cann_create_int_array(output_سائيز.معلومات(), 2);

    GGML_CANN_CALL_ACLNN_OP(ctx, UpsampleNearest2d, acl_src.get(), output_سائيز_array.get(), acl_dst.get());
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
ساكن فراغ aclnn_pad(ggml_backend_cann_context & ctx,
                      aclTensor *                 acl_src,
                      aclTensor *                 acl_dst,
                      int64_t *                   paddings,
                      عائم                       value = 0.0f) {
    acl_int_array_ptr acl_pad   = ggml_cann_create_int_array(paddings, GGML_MAX_DIMS * 2);
    acl_scalar_ptr    acl_قدر = ggml_cann_create_scalar(&value, acl_معلومات_Type::ACL_FLOAT);

    GGML_CANN_CALL_ACLNN_OP(ctx, ConstantPadNd, acl_src, acl_pad.get(), acl_قدر.get(), acl_dst);
}

فراغ ggml_cann_pad(ggml_backend_cann_context & ctx, ggml_tensor * dst) {
    ggml_tensor *  src     = dst->src[0];
    acl_tensor_ptr acl_src = ggml_cann_create_tensor(src);
    acl_tensor_ptr acl_dst = ggml_cann_create_tensor(dst);

    // padding: value in the array means how much distance will be padding.
    // the position of elements in the array means which dirction to padding,
    // each position means: [dim0.front, dim0.behind, dim1.front, dim1.behind,
    //                       dim2.front, dim2.behind, dim3.front, dim3.behind]
    ثابت int32_t lp0 = ggml_get_op_params_i32(dst, 0);
    ثابت int32_t rp0 = ggml_get_op_params_i32(dst, 1);
    ثابت int32_t lp1 = ggml_get_op_params_i32(dst, 2);
    ثابت int32_t rp1 = ggml_get_op_params_i32(dst, 3);
    ثابت int32_t lp2 = ggml_get_op_params_i32(dst, 4);
    ثابت int32_t rp2 = ggml_get_op_params_i32(dst, 5);
    ثابت int32_t lp3 = ggml_get_op_params_i32(dst, 6);
    ثابت int32_t rp3 = ggml_get_op_params_i32(dst, 7);

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
ساكن فراغ ggml_cann_avg_pool2d(ggml_backend_cann_context & ctx, ggml_tensor * dst) {
    ggml_tensor * src = dst->src[0];
    GGML_ASSERT(src->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    acl_tensor_ptr acl_src = ggml_cann_create_tensor(src, خالي_اشارو, خالي_اشارو, 0, ACL_FORMAT_NCHW);
    acl_tensor_ptr acl_dst = ggml_cann_create_tensor(dst, خالي_اشارو, خالي_اشارو, 0, ACL_FORMAT_NCHW);

    ثابت int32_t * opts = (ثابت int32_t *) dst->op_params;
    ثابت صحيح       k0   = opts[1];
    ثابت صحيح       k1   = opts[2];
    ثابت صحيح       s0   = opts[3];
    ثابت صحيح       s1   = opts[4];
    ثابت صحيح       p0   = opts[5];
    ثابت صحيح       p1   = opts[6];

    std::vector<int64_t> kernel_dims      = { k1, k0 };
    std::vector<int64_t> stride_dims      = { s1, s0 };
    std::vector<int64_t> padding_avg_dims = { p1, p0 };  // (padH, padW)

    acl_int_array_ptr kernel_سائيز  = ggml_cann_create_int_array(kernel_dims.معلومات(), 2);
    acl_int_array_ptr strides      = ggml_cann_create_int_array(stride_dims.معلومات(), 2);
    acl_int_array_ptr paddings_avg = ggml_cann_create_int_array(padding_avg_dims.معلومات(), 2);

    منطقي    ceil_گهڻائي         = خطأ;
    منطقي    ڳڻپ_include_pad = سچ;
    int64_t ورهائيندڙ_override  = 0;
    int8_t  cube_math_type    = 0;
#جيڪڏهن_وصف_ٿيل ASCEND_310P
    cube_math_type = 1;
#جيڪڏهن_پورو

    GGML_CANN_CALL_ACLNN_OP(ctx, AvgPool2d, acl_src.get(), kernel_سائيز.get(), strides.get(), paddings_avg.get(),
                            ceil_گهڻائي, ڳڻپ_include_pad, ورهائيندڙ_override, cube_math_type, acl_dst.get());
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
ساكن فراغ ggml_cann_max_pool2d(ggml_backend_cann_context & ctx, ggml_tensor * dst) {
    ggml_tensor * src = dst->src[0];
    GGML_ASSERT(src->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    acl_tensor_ptr acl_src = ggml_cann_create_tensor(src, خالي_اشارو, خالي_اشارو, 0, ACL_FORMAT_NCHW);
    acl_tensor_ptr acl_dst = ggml_cann_create_tensor(dst, خالي_اشارو, خالي_اشارو, 0, ACL_FORMAT_NCHW);

    ثابت int32_t * opts = (ثابت int32_t *) dst->op_params;
    ثابت صحيح       k0   = opts[1];
    ثابت صحيح       k1   = opts[2];
    ثابت صحيح       s0   = opts[3];
    ثابت صحيح       s1   = opts[4];
    ثابت صحيح       p0   = opts[5];
    ثابت صحيح       p1   = opts[6];

    int64_t عارضي_ne[] = { src->ne[0] + p0 * 2, src->ne[1] + p1 * 2, src->ne[2], src->ne[3] };
    size_t  عارضي_nb[GGML_MAX_DIMS];

    عارضي_nb[0] = ggml_element_سائيز(src);
    لكل (صحيح i = 1; i < GGML_MAX_DIMS; i++) {
        عارضي_nb[i] = عارضي_nb[i - 1] * عارضي_ne[i - 1];
    }

    ggml_cann_pool_alloc عارضي_بفر_allocator(ctx.pool(), ggml_nbytes(src) + p0 * 2 + p1 * 2 * src->nb[1]);
    فراغ *               بفر = عارضي_بفر_allocator.get();
    acl_tensor_ptr tmp_tensor   = ggml_cann_create_tensor(بفر, ACL_FLOAT, ggml_element_سائيز(src), عارضي_ne, عارضي_nb,
                                                          GGML_MAX_DIMS, ACL_FORMAT_NCHW);

    // pad: see padding in ggml_cann_pad()
    int64_t paddings[] = { p0, p0, p1, p1, 0, 0, 0, 0 };
    عائم   value      = -FLT_MAX;
    aclnn_pad(ctx, acl_src.get(), tmp_tensor.get(), paddings, value);

    // max_pool
    std::vector<int64_t> kernel_dims      = { k1, k0 };
    std::vector<int64_t> stride_dims      = { s1, s0 };
    // padding_max_dims: [dim0_start, dim0_end, dim1_start, dim1_end]
    std::vector<int64_t> padding_max_dims = { 0, 0, 0, 0 };
    std::vector<int64_t> dilation_سائيز    = { 1, 1 };
    acl_int_array_ptr    kernel_سائيز      = ggml_cann_create_int_array(kernel_dims.معلومات(), 2);
    acl_int_array_ptr    strides          = ggml_cann_create_int_array(stride_dims.معلومات(), 2);
    acl_int_array_ptr    paddings_max     = ggml_cann_create_int_array(padding_max_dims.معلومات(), 4);
    acl_int_array_ptr    dilations        = ggml_cann_create_int_array(dilation_سائيز.معلومات(), 2);

    منطقي    ceil_گهڻائي = خطأ;
    int64_t auto_pads = 0;
    GGML_CANN_CALL_ACLNN_OP(ctx, MaxPool, tmp_tensor.get(), kernel_سائيز.get(), strides.get(), auto_pads,
                            paddings_max.get(), dilations.get(), ceil_گهڻائي, acl_dst.get());
}

فراغ ggml_cann_pool2d(ggml_backend_cann_context & ctx, ggml_tensor * dst) {
    ثابت int32_t *   opts = (ثابت int32_t *) dst->op_params;
    تعداد ggml_op_pool op   = static_cast<ggml_op_pool>(opts[0]);
    بدل (op) {
        حالة GGML_OP_POOL_AVG:
            ggml_cann_avg_pool2d(ctx, dst);
            اكسر;
        حالة GGML_OP_POOL_MAX:
            ggml_cann_max_pool2d(ctx, dst);
            اكسر;
        حالة GGML_OP_POOL_ڳڻپ:
            GGML_ABORT("fatal error");
            اكسر;
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
ساكن فراغ cann_copy(ggml_backend_cann_context & ctx, aclTensor * acl_src, aclTensor * acl_dst) {
    GGML_CANN_CALL_ACLNN_OP(ctx, InplaceCopy, acl_dst, acl_src);
}

فراغ ggml_cann_dup(ggml_backend_cann_context & ctx, ggml_tensor * dst) {
    ggml_tensor * src0 = dst->src[0];

    إذا (ggml_are_same_shape(src0, dst)) {
        acl_tensor_ptr acl_src = ggml_cann_create_tensor(src0);
        acl_tensor_ptr acl_dst = ggml_cann_create_tensor(dst);
        إذا (dst->type == src0->type) {
            cann_copy(ctx, acl_src.get(), acl_dst.get());
        } وإلا {
            aclnn_cast(ctx, acl_src.get(), acl_dst.get(), ggml_cann_type_mapping(dst->type));
        }
    } وإلا {
        فراغ *               src_trans_بفر = src0->معلومات;
        ggml_cann_pool_alloc src_بفر_allocator;
        إذا (!ggml_is_contiguous(src0)) {
            acl_tensor_ptr acl_src = ggml_cann_create_tensor(src0);
            src_بفر_allocator.alloc(ctx.pool(), ggml_nelements(src0) * ggml_type_سائيز(src0->type));
            src_trans_بفر = src_بفر_allocator.get();
            size_t src_trans_nb[GGML_MAX_DIMS];
            src_trans_nb[0] = ggml_type_سائيز(src0->type);
            لكل (صحيح i = 1; i < GGML_MAX_DIMS; i++) {
                src_trans_nb[i] = src_trans_nb[i - 1] * src0->ne[i - 1];
            }
            acl_tensor_ptr src_trans_tensor =
                ggml_cann_create_tensor(src_trans_بفر, ggml_cann_type_mapping(src0->type),
                                        ggml_type_سائيز(src0->type), src0->ne, src_trans_nb, GGML_MAX_DIMS);
            cann_copy(ctx, acl_src.get(), src_trans_tensor.get());
        }

        size_t src_reshape_nb[GGML_MAX_DIMS];
        src_reshape_nb[0] = ggml_type_سائيز(src0->type);
        لكل (صحيح i = 1; i < GGML_MAX_DIMS; i++) {
            src_reshape_nb[i] = src_reshape_nb[i - 1] * dst->ne[i - 1];
        }

        acl_tensor_ptr trans_acl_src =
            ggml_cann_create_tensor(src_trans_بفر, ggml_cann_type_mapping(src0->type), ggml_type_سائيز(src0->type),
                                    dst->ne, src_reshape_nb, GGML_MAX_DIMS, ACL_FORMAT_ND);
        acl_tensor_ptr acl_dst = ggml_cann_create_tensor(dst);

        إذا (dst->type == src0->type) {
            cann_copy(ctx, trans_acl_src.get(), acl_dst.get());
        } وإلا {
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
ساكن acl_tensor_ptr aclnn_zero(ggml_backend_cann_context & ctx,
                                 فراغ *                      بفر,
                                 size_t                      n_بائيٽون,
                                 int64_t *                   ne,
                                 int64_t                     dims,
                                 acl_معلومات_Type                 type,
                                 size_t                      type_سائيز) {
    size_t nb[GGML_MAX_DIMS];
    nb[0] = type_سائيز;
    لكل (صحيح i = 1; i < dims; i++) {
        nb[i] = nb[i - 1] * ne[i - 1];
    }

    acl_tensor_ptr zero = ggml_cann_create_tensor(بفر, type, type_سائيز, ne, nb, dims);
    GGML_CANN_CALL_ACLNN_OP(ctx, InplaceZero, zero.get());
    أرجع zero;
    GGML_اڻ_استعمال_ٿيل(n_بائيٽون);
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
ساكن acl_tensor_ptr aclnn_قدرون(ggml_backend_cann_context & ctx,
                                   فراغ *                      بفر,
                                   size_t                      n_بائيٽون,
                                   int64_t *                   ne,
                                   int64_t                     dims,
                                   acl_معلومات_Type                 type,
                                   size_t                      type_سائيز,
                                   عائم                       value = 1.0f) {
    acl_tensor_ptr acl_tensor = aclnn_zero(ctx, بفر, n_بائيٽون, ne, dims, type, type_سائيز);
    عائم          alpha_host = 1.0f;
    acl_scalar_ptr alpha      = ggml_cann_create_scalar(&alpha_host, acl_معلومات_Type::ACL_FLOAT);
    acl_scalar_ptr other      = ggml_cann_create_scalar(&value, acl_معلومات_Type::ACL_FLOAT);
    GGML_CANN_CALL_ACLNN_OP(ctx, InplaceAdds, acl_tensor.get(), other.get(), alpha.get());
    أرجع acl_tensor;
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
ساكن فراغ aclnn_fill_scalar(ggml_backend_cann_context & ctx, عائم scalar, aclTensor * acl_dst) {
    acl_scalar_ptr acl_scalar = ggml_cann_create_scalar(&scalar, acl_معلومات_Type::ACL_FLOAT);
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
ساكن acl_tensor_ptr get_cache_acl_tensor(ggml_backend_cann_context & ctx,
                                           فراغ **                     بفر,
                                           int64_t &                   cache_element,
                                           int64_t *                   ne,
                                           size_t *                    nb,
                                           ggml_type                   dtype,
                                           int64_t                     dims,
                                           عائم                       value) {
    // Calculate total number of elements
    int64_t n_element = 1;
    لكل (صحيح i = 0; i < dims; i++) {
        n_element *= ne[i];
    }
    size_t size = n_element * ggml_type_سائيز(dtype);

    // Allocate or expand cache if needed
    إذا (cache_element < n_element) {
        إذا (*بفر != خالي_اشارو) {
            aclrtFree(*بفر);
            *بفر = خالي_اشارو;
        }

        ACL_CHECK(aclrtMalloc(بفر, size, ACL_MEM_MALLOC_HUGE_پهريون));
        cache_element = n_element;

        // Initialize cache
        int64_t        pool_ne[1] = { n_element };
        size_t         pool_nb[1] = { ggml_type_سائيز(dtype) };
        acl_tensor_ptr acl_قدر =
            ggml_cann_create_tensor(*بفر, ggml_cann_type_mapping(dtype), ggml_type_سائيز(dtype), pool_ne, pool_nb, 1);
        aclnn_fill_scalar(ctx, value, acl_قدر.get());
    }

    أرجع ggml_cann_create_tensor(*بفر, ggml_cann_type_mapping(dtype), ggml_type_سائيز(dtype), ne, nb, dims);
}

فراغ ggml_cann_rms_norm(ggml_backend_cann_context & ctx, ggml_tensor * dst) {
    ggml_tensor * src = dst->src[0];

    acl_tensor_ptr acl_src = ggml_cann_create_tensor(src);
    acl_tensor_ptr acl_dst = ggml_cann_create_tensor(dst);

    عائم eps;
    يادگيري_نقل(&eps, dst->op_params, حجم(عائم));

    // build gamma.
    size_t acl_gamma_nb[GGML_MAX_DIMS];
    // gamma's type is the same with dst.
    acl_gamma_nb[0] = ggml_type_سائيز(dst->type);
    لكل (صحيح i = 1; i < GGML_MAX_DIMS; i++) {
        acl_gamma_nb[i] = acl_gamma_nb[i - 1] * src->ne[i - 1];
    }
    acl_tensor_ptr acl_gamma = get_cache_acl_tensor(
        ctx, &ctx.rms_norm_one_tensor_cache.cache, ctx.rms_norm_one_tensor_cache.size, src->ne, acl_gamma_nb, dst->type,
        1,    // dims
        1.0f  //قدر
    );

    // build rstd.
    int64_t acl_rstd_ne[] = { src->ne[1], src->ne[2], src->ne[3] };
    size_t  acl_rstd_nb[GGML_MAX_DIMS - 1];
    // rstd will always be F32.
    acl_rstd_nb[0] = حجم(عائم);
    لكل (صحيح i = 1; i < GGML_MAX_DIMS - 1; i++) {
        acl_rstd_nb[i] = acl_rstd_nb[i - 1] * acl_rstd_ne[i - 1];
    }
    acl_tensor_ptr acl_rstd =
        get_cache_acl_tensor(ctx, &ctx.rms_norm_zero_tensor_cache.cache, ctx.rms_norm_zero_tensor_cache.size,
                             acl_rstd_ne, acl_rstd_nb, GGML_TYPE_F32, GGML_MAX_DIMS - 1,
                             0.0f  //قدر
        );

    GGML_CANN_CALL_ACLNN_OP(ctx, RmsNorm, acl_src.get(), acl_gamma.get(), eps, acl_dst.get(), acl_rstd.get());
}

// TODO: performace is low.
فراغ ggml_cann_diag_نقاب(ggml_backend_cann_context & ctx, ggml_tensor * dst, عائم value) {
    ggml_tensor * src = dst->src[0];

    acl_tensor_ptr acl_src = ggml_cann_create_tensor(src);
    acl_tensor_ptr acl_dst = ggml_cann_create_tensor(dst);

    ثابت صحيح n_past = ((int32_t *) dst->op_params)[0];

    ggml_cann_pool_alloc one_tensor_allocator(ctx.pool(), ggml_nbytes(src));
    فراغ *               بفر = one_tensor_allocator.get();

    acl_tensor_ptr نقاب_tensor = ggml_cann_create_tensor(بفر, ggml_cann_type_mapping(src->type),
                                                         ggml_type_سائيز(src->type), src->ne, src->nb, GGML_MAX_DIMS);

    aclnn_fill_scalar(ctx, value, نقاب_tensor.get());

    عائم          alpha_قدر = 1.0f;
    acl_scalar_ptr alpha      = ggml_cann_create_scalar(&alpha_قدر, acl_معلومات_Type::ACL_FLOAT);

    GGML_CANN_CALL_ACLNN_OP(ctx, InplaceTriu, نقاب_tensor.get(), n_past + 1);
    GGML_CANN_CALL_ACLNN_OP(ctx, Tril, acl_src.get(), n_past + 1, acl_dst.get());
    GGML_CANN_CALL_ACLNN_OP(ctx, Inplace_جوڙيو, acl_dst.get(), نقاب_tensor.get(), alpha.get());
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
ساكن فراغ aclnn_ترتيب_بدلايو(ggml_backend_cann_context & ctx,
                          aclTensor *                 acl_src,
                          aclTensor *                 acl_dst,
                          int64_t *                   new_dim,
                          uint64_t                    dims) {
    acl_int_array_ptr acl_dims = ggml_cann_create_int_array(new_dim, dims);
    GGML_CANN_CALL_ACLNN_OP(ctx, ترتيب_بدلايو, acl_src, acl_dims.get(), acl_dst);
}

ساكن فراغ ggml_cann_im2col_2d_post_process(ggml_backend_cann_context & ctx,
                                             ggml_tensor *               dst,
                                             ggml_tensor *               src1,
                                             aclTensor *                 tmp_cast_tensor,
                                             aclTensor *                 tmp_im2col_tensor) {
    // Permute: [N, IC * KH * KW, OW * OH] -> [N, OW * OH, IC * KH * KW]
    int64_t        dst_ne[] = { dst->ne[0], dst->ne[1] * dst->ne[2], dst->ne[3] };
    size_t         dst_nb[] = { dst->nb[0], dst->nb[1], dst->nb[3] };
    acl_tensor_ptr acl_dst  = ggml_cann_create_tensor(dst, dst_ne, dst_nb, GGML_MAX_DIMS - 1);

    int64_t ترتيب_بدلايو_dim[] = { 0, 2, 1 };
    إذا (src1->type != dst->type) {
        aclnn_ترتيب_بدلايو(ctx, tmp_cast_tensor, acl_dst.get(), ترتيب_بدلايو_dim, 3);
    } وإلا {
        aclnn_ترتيب_بدلايو(ctx, tmp_im2col_tensor, acl_dst.get(), ترتيب_بدلايو_dim, 3);
    }
}

ساكن فراغ ggml_cann_im2col_1d_post_process(ggml_backend_cann_context &  ctx,
                                             ggml_tensor *                dst,
                                             ggml_tensor *                src1,
                                             aclTensor *                  tmp_cast_tensor,
                                             aclTensor *                  tmp_im2col_tensor,
                                             ثابت std::vector<int64_t> & im2col_op_params) {
    // get params
    ثابت int64_t KH             = im2col_op_params[0];
    ثابت int64_t KW             = im2col_op_params[1];
    ثابت int64_t IW             = im2col_op_params[2];
    ثابت int64_t IC             = im2col_op_params[3];
    ثابت int64_t N              = im2col_op_params[4];
    ثابت int64_t OH             = im2col_op_params[5];
    ثابت int64_t OW             = im2col_op_params[6];
    ثابت int64_t s0             = im2col_op_params[7];
    ثابت int64_t p0             = im2col_op_params[8];
    ثابت int64_t d0             = im2col_op_params[9];
    ثابت int64_t n_بائيٽون_ضرب_ڪندڙ = im2col_op_params[10];

    // Permute: [N, IC * KH * KW, OW * OH] ->
    // [N, OW * OH * n_bytes_factor, IC * KH * KW]
    ggml_cann_pool_alloc tmp_ترتيب_بدلايو_allocator(ctx.pool());
    tmp_ترتيب_بدلايو_allocator.alloc(ggml_nbytes(dst) * n_بائيٽون_ضرب_ڪندڙ);
    فراغ * tmp_ترتيب_بدلايو_بفر = tmp_ترتيب_بدلايو_allocator.get();

    int64_t tmp_ترتيب_بدلايو_ne[] = { IC * KH * KW, OW * OH * n_بائيٽون_ضرب_ڪندڙ, N };
    size_t  tmp_ترتيب_بدلايو_nb[GGML_MAX_DIMS - 1];
    tmp_ترتيب_بدلايو_nb[0] = ggml_type_سائيز(dst->type);
    لكل (صحيح i = 1; i < GGML_MAX_DIMS - 1; i++) {
        tmp_ترتيب_بدلايو_nb[i] = tmp_ترتيب_بدلايو_nb[i - 1] * tmp_ترتيب_بدلايو_ne[i - 1];
    }

    acl_tensor_ptr tmp_ترتيب_بدلايو_tensor =
        ggml_cann_create_tensor(tmp_ترتيب_بدلايو_بفر, ggml_cann_type_mapping(dst->type), ggml_type_سائيز(dst->type),
                                tmp_ترتيب_بدلايو_ne, tmp_ترتيب_بدلايو_nb, GGML_MAX_DIMS - 1, ACL_FORMAT_ND);

    int64_t ترتيب_بدلايو_dim[] = { 0, 2, 1 };
    إذا (src1->type != dst->type) {
        aclnn_ترتيب_بدلايو(ctx, tmp_cast_tensor, tmp_ترتيب_بدلايو_tensor.get(), ترتيب_بدلايو_dim, 3);
    } وإلا {
        aclnn_ترتيب_بدلايو(ctx, tmp_im2col_tensor, tmp_ترتيب_بدلايو_tensor.get(), ترتيب_بدلايو_dim, 3);
    }

    // number of times the kernel moves in W dimension
    ثابت صحيح n_قدم_w = (IW + 2 * p0 - d0 * (KW - 1) - 1) / s0 + 1;
    size_t    وٿي;
    فراغ *    cur_dst_بفر = dst->معلومات, *cur_ترتيب_بدلايو_بفر = tmp_ترتيب_بدلايو_بفر;

    // memory copy with offset to restore 1D im2col from 2d
    إذا (IC > 1) {
        وٿي          = IC * KH * KW * n_قدم_w * ggml_type_سائيز(dst->type);
        size_t cpy_سائيز = KH * KW * ggml_type_سائيز(dst->type);

        لكل (صحيح c = 0; c < IC; c++) {
            cur_ترتيب_بدلايو_بفر = (حرف *) tmp_ترتيب_بدلايو_بفر + وٿي + KH * KW * c * ggml_type_سائيز(dst->type);
            cur_dst_بفر     = (حرف *) dst->معلومات + c * KH * KW * n_قدم_w * ggml_type_سائيز(dst->type);

            لكل (صحيح i = 0; i < n_قدم_w; i++) {
                ACL_CHECK(aclrtMemcpyAsync(cur_dst_بفر, cpy_سائيز, cur_ترتيب_بدلايو_بفر, cpy_سائيز,
                                           ACL_MEMCPY_DEVICE_منزل_DEVICE, ctx.stream()));
                cur_dst_بفر     = (حرف *) cur_dst_بفر + KH * KW * ggml_type_سائيز(dst->type);
                cur_ترتيب_بدلايو_بفر = (حرف *) cur_ترتيب_بدلايو_بفر + KH * KW * IC * ggml_type_سائيز(dst->type);
            }
        }
    } وإلا {
        وٿي = KH * KW * n_قدم_w * ggml_type_سائيز(dst->type);  // equal to ggml_nbytes(dst)
        ACL_CHECK(aclrtMemcpyAsync(dst->معلومات, وٿي, (حرف *) tmp_ترتيب_بدلايو_بفر + وٿي, وٿي,
                                   ACL_MEMCPY_DEVICE_منزل_DEVICE, ctx.stream()));
    }
}

فراغ ggml_cann_im2col(ggml_backend_cann_context & ctx, ggml_tensor * dst) {
    ggml_tensor * src0 = dst->src[0];  // kernel
    ggml_tensor * src1 = dst->src[1];  //داخلا

    GGML_TENSOR_ٻٽو_OP_LOCALS;

    // aclnnIm2col only works on 2D. set s1, p1, d1 to 1 to perform 2D
    // im2col and do post-processing to restore it to 1D.
    ثابت منطقي    is_2D = ((ثابت int32_t *) (dst->op_params))[6] == 1;
    ثابت int32_t s0    = ((ثابت int32_t *) (dst->op_params))[0];
    ثابت int32_t s1    = is_2D ? ((ثابت int32_t *) (dst->op_params))[1] : 1;
    ثابت int32_t p0    = ((ثابت int32_t *) (dst->op_params))[2];
    ثابت int32_t p1    = is_2D ? ((ثابت int32_t *) (dst->op_params))[3] : 1;
    ثابت int32_t d0    = ((ثابت int32_t *) (dst->op_params))[4];
    ثابت int32_t d1    = is_2D ? ((ثابت int32_t *) (dst->op_params))[5] : 1;

    ثابت int64_t N  = ne13;
    ثابت int64_t IC = ne12;
    ثابت int64_t KH = ne01;
    ثابت int64_t KW = ne00;
    ثابت int64_t IW = ne10;

    ثابت int64_t OH = is_2D ? ne2 : 1;
    ثابت int64_t OW = ne1;

    // memory allocated increased to 3x when is_2D == false
    ثابت int64_t n_بائيٽون_ضرب_ڪندڙ = is_2D ? 1 : 3;

    // im2col: [N,C,H,W] -> [N, IC * KH * KW, OW * OH * n_bytes_factor]
    acl_tensor_ptr acl_src1        = ggml_cann_create_tensor(src1);
    int64_t        tmp_im2col_ne[] = { OW * OH * n_بائيٽون_ضرب_ڪندڙ, IC * KH * KW, N };
    size_t         tmp_im2col_nb[GGML_MAX_DIMS - 1];

    tmp_im2col_nb[0] = ggml_type_سائيز(src1->type);
    لكل (صحيح i = 1; i < GGML_MAX_DIMS - 1; i++) {
        tmp_im2col_nb[i] = tmp_im2col_nb[i - 1] * tmp_im2col_ne[i - 1];
    }

    // Calculate im2col.
    // If dst is f16, tmp_buffer is f32, we need alloc src.typesize *
    // dst.elemcount.
    ggml_cann_pool_alloc im2col_allocator(ctx.pool(), ggml_nelements(dst) * ggml_element_سائيز(src1) * n_بائيٽون_ضرب_ڪندڙ);
    فراغ *               tmp_im2col_بفر = im2col_allocator.get();

    acl_tensor_ptr tmp_im2col_tensor =
        ggml_cann_create_tensor(tmp_im2col_بفر, ggml_cann_type_mapping(src1->type), ggml_type_سائيز(src1->type),
                                tmp_im2col_ne, tmp_im2col_nb, GGML_MAX_DIMS - 1, ACL_FORMAT_ND);

    std::vector<int64_t> kernel_dims   = { KH, KW };
    std::vector<int64_t> dilation_سائيز = { d1, d0 };
    std::vector<int64_t> padding_dims  = { p1, p0 };
    std::vector<int64_t> stride_dims   = { s1, s0 };
    acl_int_array_ptr    kernel_سائيز   = ggml_cann_create_int_array(kernel_dims.معلومات(), 2);
    acl_int_array_ptr    dilations     = ggml_cann_create_int_array(dilation_سائيز.معلومات(), 2);
    acl_int_array_ptr    paddings      = ggml_cann_create_int_array(padding_dims.معلومات(), 2);
    acl_int_array_ptr    strides       = ggml_cann_create_int_array(stride_dims.معلومات(), 2);
    GGML_CANN_CALL_ACLNN_OP(ctx, Im2col, acl_src1.get(), kernel_سائيز.get(), dilations.get(), paddings.get(),
                            strides.get(), tmp_im2col_tensor.get());

    // Cast if dst is f16.
    acl_tensor_ptr       tmp_cast_tensor;
    ggml_cann_pool_alloc tmp_cast_allocator(ctx.pool());
    فراغ *               tmp_cast_بفر = خالي_اشارو;
    إذا (src1->type != dst->type) {
        tmp_cast_allocator.alloc(ggml_nbytes(dst) * n_بائيٽون_ضرب_ڪندڙ);
        tmp_cast_بفر = tmp_cast_allocator.get();
        size_t عارضي_cast_nb[GGML_MAX_DIMS - 1];
        عارضي_cast_nb[0] = ggml_type_سائيز(dst->type);
        لكل (صحيح i = 1; i < GGML_MAX_DIMS - 1; i++) {
            عارضي_cast_nb[i] = عارضي_cast_nb[i - 1] * tmp_im2col_ne[i - 1];
        }

        tmp_cast_tensor =
            ggml_cann_create_tensor(tmp_cast_بفر, ggml_cann_type_mapping(dst->type), ggml_type_سائيز(dst->type),
                                    tmp_im2col_ne, عارضي_cast_nb, GGML_MAX_DIMS - 1, ACL_FORMAT_ND);
        aclnn_cast(ctx, tmp_im2col_tensor.get(), tmp_cast_tensor.get(), ggml_cann_type_mapping(dst->type));
    }

    // post-processing
    إذا (is_2D) {
        ggml_cann_im2col_2d_post_process(ctx, dst, src1, tmp_cast_tensor.get(), tmp_im2col_tensor.get());
    } وإلا {
        std::vector<int64_t> im2col_op_params = { KH, KW, IW, IC, N, OH, OW, s0, p0, d0, n_بائيٽون_ضرب_ڪندڙ };
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
ساكن فراغ aclnn_exp(ggml_backend_cann_context & ctx, aclTensor * acl_src) {
    GGML_CANN_CALL_ACLNN_OP(ctx, InplaceExp, acl_src);
}

فراغ aclnn_cos(ggml_backend_cann_context & ctx, aclTensor * acl_src, aclTensor * acl_dst) {
    إذا (acl_dst == خالي_اشارو) {
        GGML_CANN_CALL_ACLNN_OP(ctx, InplaceCos, acl_src);
    } وإلا {
        GGML_CANN_CALL_ACLNN_OP(ctx, Cos, acl_src, acl_dst);
    }
}

فراغ aclnn_sin(ggml_backend_cann_context & ctx, aclTensor * acl_src, aclTensor * acl_dst) {
    إذا (acl_dst == خالي_اشارو) {
        GGML_CANN_CALL_ACLNN_OP(ctx, InplaceSin, acl_src);
    } وإلا {
        GGML_CANN_CALL_ACLNN_OP(ctx, Sin, acl_src, acl_dst);
    }
}

فراغ ggml_cann_timestep_embedding(ggml_backend_cann_context & ctx, ggml_tensor * dst) {
    ثابت ggml_tensor * src = dst->src[0];

    GGML_ASSERT(src->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    ثابت صحيح dim        = dst->op_params[0];
    ثابت صحيح max_period = dst->op_params[1];
    صحيح       half       = dim / 2;

    acl_tensor_ptr acl_src = ggml_cann_create_tensor(src);

    // arange: [0, ..., half)
    عائم   start             = 0;
    عائم   stop              = half;
    عائم   قدم              = 1;
    int64_t n_elements_arange = half;
    int64_t tmp_arange_ne[]   = { half };
    size_t  tmp_arange_nb[]   = { حجم(dst->type) };

    ggml_cann_pool_alloc arange_allocator(ctx.pool(), half * حجم(dst->type));
    فراغ *               tmp_arange_بفر = arange_allocator.get();
    acl_tensor_ptr       tmp_arange_tensor =
        ggml_cann_create_tensor(tmp_arange_بفر, ggml_cann_type_mapping(dst->type), ggml_type_سائيز(dst->type),
                                tmp_arange_ne, tmp_arange_nb, GGML_MAX_DIMS - 3, ACL_FORMAT_ND);

    aclnn_arange(ctx, tmp_arange_tensor.get(), start, stop, قدم, n_elements_arange);

    // freq
    عائم freq_param = -logf(max_period) / half;
    منطقي  inplace    = سچ;
    aclnn_muls(ctx, tmp_arange_tensor.get(), freq_param, خالي_اشارو, inplace);
    aclnn_exp(ctx, tmp_arange_tensor.get());

    // permute: src [0,1,2,3]->[0,1,3,2]
    int64_t tmp_ترتيب_بدلايو_ne[] = { src->ne[1], src->ne[0], src->ne[2], src->ne[3] };
    size_t  tmp_ترتيب_بدلايو_nb[GGML_MAX_DIMS];
    tmp_ترتيب_بدلايو_nb[0] = ggml_type_سائيز(src->type);
    لكل (صحيح i = 1; i < GGML_MAX_DIMS; i++) {
        tmp_ترتيب_بدلايو_nb[i] = tmp_ترتيب_بدلايو_nb[i - 1] * tmp_ترتيب_بدلايو_ne[i - 1];
    }

    ggml_cann_pool_alloc ترتيب_بدلايو_allocator(ctx.pool(), ggml_nbytes(src));
    فراغ *               tmp_ترتيب_بدلايو_بفر = ترتيب_بدلايو_allocator.get();
    acl_tensor_ptr       tmp_ترتيب_بدلايو_tensor =
        ggml_cann_create_tensor(tmp_ترتيب_بدلايو_بفر, ggml_cann_type_mapping(src->type), ggml_type_سائيز(src->type),
                                tmp_ترتيب_بدلايو_ne, tmp_ترتيب_بدلايو_nb, GGML_MAX_DIMS, ACL_FORMAT_ND);
    int64_t ترتيب_بدلايو_dim[] = { 0, 1, 3, 2 };
    int64_t انگ_dims      = 4;
    aclnn_ترتيب_بدلايو(ctx, acl_src.get(), tmp_ترتيب_بدلايو_tensor.get(), ترتيب_بدلايو_dim, انگ_dims);

    // timestep * freq
    int64_t tmp_mul_ne[] = { src->ne[1] * half, src->ne[0], src->ne[2], src->ne[3] };
    size_t  tmp_mul_nb[GGML_MAX_DIMS];
    tmp_mul_nb[0] = ggml_type_سائيز(src->type);
    لكل (صحيح i = 1; i < GGML_MAX_DIMS; i++) {
        tmp_mul_nb[i] = tmp_mul_nb[i - 1] * tmp_mul_ne[i - 1];
    }

    صحيح mul_nelements = src->ne[1] * half * src->ne[0] * src->ne[2] * src->ne[3];

    ggml_cann_pool_alloc mul_allocator(ctx.pool(), mul_nelements * ggml_type_سائيز(src->type));
    فراغ *               tmp_mul_بفر = mul_allocator.get();
    acl_tensor_ptr       tmp_mul_tensor =
        ggml_cann_create_tensor(tmp_mul_بفر, ggml_cann_type_mapping(src->type), ggml_type_سائيز(src->type),
                                tmp_mul_ne, tmp_mul_nb, GGML_MAX_DIMS, ACL_FORMAT_ND);
    aclnn_mul(ctx, tmp_ترتيب_بدلايو_tensor.get(), tmp_arange_tensor.get(), tmp_mul_tensor.get());

    // cos
    ggml_cann_pool_alloc cos_allocator(ctx.pool(), mul_nelements * ggml_type_سائيز(src->type));
    فراغ *               tmp_cos_بفر = cos_allocator.get();
    acl_tensor_ptr       tmp_cos_tensor =
        ggml_cann_create_tensor(tmp_cos_بفر, ggml_cann_type_mapping(dst->type), ggml_type_سائيز(dst->type),
                                tmp_mul_ne, tmp_mul_nb, GGML_MAX_DIMS, ACL_FORMAT_ND);

    aclnn_cos(ctx, tmp_mul_tensor.get(), tmp_cos_tensor.get());

    // sin
    ggml_cann_pool_alloc sin_allocator(ctx.pool(), mul_nelements * ggml_type_سائيز(src->type));
    فراغ *               tmp_sin_بفر = sin_allocator.get();
    acl_tensor_ptr       tmp_sin_tensor =
        ggml_cann_create_tensor(tmp_sin_بفر, ggml_cann_type_mapping(dst->type), ggml_type_سائيز(dst->type),
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
ساكن فراغ aclnn_pow_tensor_tensor(ggml_backend_cann_context & ctx, aclTensor * acl_dst, aclTensor * acl_exp) {
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
ساكن فراغ aclnn_get_لاهي_inner(ggml_backend_cann_context & ctx,
                                  فراغ *                      لاهي_بفر,
                                  عائم                       m,
                                  int64_t                     size,
                                  عائم                       start,
                                  عائم                       stop,
                                  عائم                       قدم,
                                  ggml_type                   dtype) {
    acl_معلومات_Type acl_type  = ggml_cann_type_mapping(dtype);
    size_t      type_سائيز = ggml_type_سائيز(dtype);

    int64_t ne[] = { size };
    size_t  nb[] = { type_سائيز };

    ggml_cann_pool_alloc arange_allocator(ctx.pool(), size * type_سائيز);
    فراغ *               arange_بفر = arange_allocator.get();

    acl_tensor_ptr arange_tensor = ggml_cann_create_tensor(arange_بفر, acl_type, type_سائيز, ne, nb, 1);
    aclnn_arange(ctx, arange_tensor.get(), start, stop, قدم, size);

    acl_tensor_ptr لاهي_tensor = ggml_cann_create_tensor(لاهي_بفر, acl_type, type_سائيز, ne, nb, 1);

    acl_scalar_ptr sc = ggml_cann_create_scalar(&m, acl_معلومات_Type::ACL_FLOAT);

    GGML_CANN_CALL_ACLNN_OP(ctx, PowScalarTensor, sc.get(), arange_tensor.get(), لاهي_tensor.get());
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
ساكن فراغ aclnn_get_لاهي(ggml_backend_cann_context & ctx,
                            int64_t                     n_سر,
                            فراغ *                      لاهي_بفر,
                            عائم                       max_bias,
                            ggml_type                   dtype) {
    ثابت صحيح n_سر_log2 = 1u << (uint32_t) floor(log2(n_سر));

    عائم m0 = powf(2.0f, -(max_bias) / n_سر_log2);
    عائم m1 = powf(2.0f, -(max_bias / 2.0f) / n_سر_log2);

    // const float slope = (max_bias > 0.0f) ?
    //                          h < n_head_log2 ?
    //                              powf(m0, h + 1) :
    //                              powf(m1, 2*(h - n_head_log2) + 1) :
    //                          1.0f;
    // arange1
    عائم start = 0 + 1;
    عائم end   = (n_سر_log2 - 1) + 1;
    عائم قدم  = 1;
    عائم count = n_سر_log2;
    // end needs to be +1 because aclnn uses a left-closed, right-open interval.
    aclnn_get_لاهي_inner(ctx, لاهي_بفر, m0, count, start, end + 1, قدم, dtype);
    إذا (n_سر_log2 < n_سر) {
        // arange2
        start = 2 * (n_سر_log2 - n_سر_log2) + 1;
        end   = 2 * ((n_سر - 1) - n_سر_log2) + 1;
        قدم  = 2;
        count = n_سر - n_سر_log2;
        aclnn_get_لاهي_inner(ctx, (حرف *) لاهي_بفر + n_سر_log2 * ggml_type_سائيز(dtype), m1, count, start, end + 1,
                              قدم, dtype);
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
ساكن فراغ aclnn_جوڙيو_alibi(ggml_backend_cann_context & ctx,
                            ggml_tensor *               نقاب,
                            ggml_tensor *               dst,
                            فراغ *                      dst_ptr,
                            عائم                       max_bias) {
    فراغ * لاهي_بفر = خالي_اشارو;
    فراغ * bias_بفر  = خالي_اشارو;

    إذا (max_bias > 0.0f) {
        int64_t              n_heads = dst->ne[2];
        ggml_cann_pool_alloc لاهي_allocator(ctx.pool(), n_heads * حجم(عائم));
        لاهي_بفر = لاهي_allocator.get();
        ggml_cann_pool_alloc bias_allocator(ctx.pool(), ggml_nelements(dst) * ggml_element_سائيز(dst));
        bias_بفر = bias_allocator.get();
        aclnn_get_لاهي(ctx, n_heads, لاهي_بفر, max_bias, GGML_TYPE_F32);
    }

    // broadcast for mask, slop and dst;
    int64_t nr2 = dst->ne[2] / نقاب->ne[2];
    int64_t nr3 = dst->ne[3] / نقاب->ne[3];

    // broadcast the mask across rows
    int64_t نقاب_ne[] = { نقاب->ne[0], dst->ne[1], نقاب->ne[2], 1, نقاب->ne[3], 1 };
    size_t  نقاب_nb[] = { نقاب_nb[0] = نقاب->nb[0], نقاب_nb[1] = نقاب->nb[1], نقاب_nb[2] = نقاب->nb[2],
                          نقاب_nb[3] = نقاب->nb[2], نقاب_nb[4] = نقاب->nb[3], نقاب_nb[5] = نقاب->nb[3] };

    int64_t dst_ne[] = { dst->ne[0], dst->ne[1], نقاب->ne[2], nr2, نقاب->ne[3], nr3 };
    size_t  dst_nb[] = { dst_nb[0] = dst->nb[0], dst_nb[1] = dst->nb[1], dst_nb[2] = dst->nb[2],
                         dst_nb[3] = dst->nb[2], dst_nb[4] = dst->nb[3], dst_nb[5] = dst->nb[3] };

    // slope is a 1 dim tensor, slope.ne2 == dst.ne2
    int64_t لاهي_ne[] = { 1, 1, نقاب->ne[2], nr2, 1, 1 };
    size_t  لاهي_nb[GGML_MAX_DIMS + 2];
    لاهي_nb[0] = حجم(عائم);
    لكل (صحيح i = 1; i < GGML_MAX_DIMS + 2; i++) {
        لاهي_nb[i] = لاهي_nb[i - 1] * لاهي_ne[i - 1];
    }

    acl_tensor_ptr acl_لاهي =
        ggml_cann_create_tensor(لاهي_بفر, ACL_FLOAT, حجم(عائم), لاهي_ne, لاهي_nb, GGML_MAX_DIMS + 2);
    acl_tensor_ptr acl_نقاب = ggml_cann_create_tensor(نقاب, نقاب_ne, نقاب_nb, GGML_MAX_DIMS + 2);

    // write data into dst_ptr using only the shape information of the dst tensor.
    acl_tensor_ptr acl_dst = ggml_cann_create_tensor(dst_ptr, ggml_cann_type_mapping(dst->type),
                                                     ggml_type_سائيز(dst->type), dst_ne, dst_nb, GGML_MAX_DIMS + 2);

    إذا (max_bias > 0.0f) {
        int64_t bias_ne[] = { نقاب->ne[0], dst->ne[1], نقاب->ne[2], nr2, نقاب->ne[3], 1 };
        size_t  bias_nb[GGML_MAX_DIMS + 2];
        bias_nb[0] = حجم(عائم);
        لكل (صحيح i = 1; i < GGML_MAX_DIMS + 2; i++) {
            bias_nb[i] = bias_nb[i - 1] * bias_ne[i - 1];
        }
        acl_tensor_ptr bias_tensor =
            ggml_cann_create_tensor(bias_بفر, ACL_FLOAT, حجم(عائم), bias_ne, bias_nb, GGML_MAX_DIMS + 2);

        aclnn_mul(ctx, acl_لاهي.get(), acl_نقاب.get(), bias_tensor.get());
        aclnn_جوڙيو(ctx, acl_dst.get(), bias_tensor.get());
    } وإلا {
        aclnn_جوڙيو(ctx, acl_dst.get(), acl_نقاب.get());
    }
}

فراغ ggml_cann_cpy(ggml_backend_cann_context & ctx, ggml_tensor * dst) {
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
ساكن فراغ aclnn_softmax(ggml_backend_cann_context & ctx, aclTensor * acl_src, int64_t dim, aclTensor * acl_dst) {
    GGML_CANN_CALL_ACLNN_OP(ctx, Softmax, acl_src, dim, acl_dst);
}

فراغ ggml_cann_softmax(ggml_backend_cann_context & ctx, ggml_tensor * dst) {
    ggml_tensor * src0 = dst->src[0];
    ggml_tensor * src1 = dst->src[1];  //نقاب

    acl_tensor_ptr acl_src0 = ggml_cann_create_tensor(src0);
    acl_tensor_ptr acl_dst  = ggml_cann_create_tensor(dst);

    عائم scale    = 1.0f;
    عائم max_bias = 0.0f;

    يادگيري_نقل(&scale, (عائم *) dst->op_params + 0, حجم(عائم));
    يادگيري_نقل(&max_bias, (عائم *) dst->op_params + 1, حجم(عائم));

    // input mul scale
    acl_scalar_ptr       acl_scale = ggml_cann_create_scalar(&scale, acl_معلومات_Type::ACL_FLOAT);
    ggml_cann_pool_alloc src_tensor_allocator(ctx.pool(), ggml_nbytes(src0));
    فراغ *               src_tensor_بفر = src_tensor_allocator.get();
    acl_tensor_ptr       softmax_tensor = ggml_cann_create_tensor(src_tensor_بفر, ggml_cann_type_mapping(src0->type),
                                                                  ggml_element_سائيز(src0), src0->ne, src0->nb, GGML_MAX_DIMS);

    aclnn_muls(ctx, acl_src0.get(), scale, softmax_tensor.get(), خطأ);

    //نقاب
    إذا (src1) {
        aclnn_جوڙيو_alibi(ctx, src1, src0, src_tensor_بفر, max_bias);
    }
    // softmax
    aclnn_softmax(ctx, softmax_tensor.get(), 3, acl_dst.get());
}


فراغ ggml_cann_get_صفون(ggml_backend_cann_context & ctx, ggml_tensor * dst) {
    ggml_tensor * src0 = dst->src[0];  //وزن
    ggml_tensor * src1 = dst->src[1];  //اشاريو

    GGML_ASSERT(dst->type == GGML_TYPE_F32 || dst->type == GGML_TYPE_F16
                || dst->type == GGML_TYPE_BF16);

    // n_idx: number of row indices per (i2, i3) batch slice.
    // ggml guarantees: src0->ne[2] == src1->ne[1], src0->ne[3] == src1->ne[2], src1->ne[3] == 1.
    ثابت int64_t n_idx = src1->ne[0];

    // Gather all (i2, i3) batch slices from src into dst.
    // ggml_cann_create_tensor reverses dims, so ACL sees [ne1, ne0].
    // GatherV2 with dim=0 gathers along ACL dim-0 == ggml ne[1] (the vocabulary / row axis).
    // nb: the 4 strides of the source buffer (nb[0..1] for the 2D slice shape,
    //     nb[2..3] for computing per-batch-slice base pointer offsets).
    خودڪار gather_batched = [&](فراغ * src_بنياد, acl_معلومات_Type acl_type, size_t type_سائيز,
                              ثابت size_t * nb) {
        int64_t src_ne[2]  = { src0->ne[0], src0->ne[1] };
        size_t  src_nb_2d[2] = { nb[0], nb[1] };
        int64_t dst_ne[2]  = { src0->ne[0], n_idx };
        size_t  dst_nb_2d[2] = { dst->nb[0], dst->nb[1] };
        int64_t idx_ne[1]  = { n_idx };
        size_t  idx_nb[1]  = { (size_t)ggml_element_سائيز(src1) };

        لكل (int64_t i3 = 0; i3 < src0->ne[3]; i3++) {
            لكل (int64_t i2 = 0; i2 < src0->ne[2]; i2++) {
                acl_tensor_ptr acl_src = ggml_cann_create_tensor(
                    (حرف *)src_بنياد + i3 * nb[3] + i2 * nb[2],
                    acl_type, type_سائيز, src_ne, src_nb_2d, 2);
                acl_tensor_ptr acl_idx = ggml_cann_create_tensor(
                    (حرف *)src1->معلومات + i3 * src1->nb[2] + i2 * src1->nb[1],
                    ggml_cann_type_mapping(src1->type), (size_t)ggml_element_سائيز(src1),
                    idx_ne, idx_nb, 1);
                acl_tensor_ptr acl_dst = ggml_cann_create_tensor(
                    (حرف *)dst->معلومات + i3 * dst->nb[3] + i2 * dst->nb[2],
                    acl_type, type_سائيز, dst_ne, dst_nb_2d, 2);
                GGML_CANN_CALL_ACLNN_OP(ctx, GatherV2, acl_src.get(), 0, acl_idx.get(), acl_dst.get());
            }
        }
    };

    بدل (src0->type) {
        حالة GGML_TYPE_BF16:
        حالة GGML_TYPE_F16:
        حالة GGML_TYPE_F32:
            إذا (src0->type == dst->type) {
                gather_batched(src0->معلومات,
                               ggml_cann_type_mapping(src0->type), ggml_type_سائيز(src0->type),
                               src0->nb);
            } وإلا {
                // Cast src0 to dst type, then gather.
                ggml_cann_pool_alloc src_cast_allocator(ctx.pool(),
                                                        ggml_nelements(src0) * ggml_element_سائيز(dst));
                size_t src_cast_nb[GGML_MAX_DIMS];
                src_cast_nb[0] = ggml_type_سائيز(dst->type);
                لكل (صحيح i = 1; i < GGML_MAX_DIMS; i++) {
                    src_cast_nb[i] = src_cast_nb[i - 1] * src0->ne[i - 1];
                }
                acl_tensor_ptr acl_src0     = ggml_cann_create_tensor(src0);
                acl_tensor_ptr acl_src_cast = ggml_cann_create_tensor(
                    src_cast_allocator.get(), ggml_cann_type_mapping(dst->type), ggml_type_سائيز(dst->type),
                    src0->ne, src_cast_nb, GGML_MAX_DIMS);
                aclnn_cast(ctx, acl_src0.get(), acl_src_cast.get(), ggml_cann_type_mapping(dst->type));

                gather_batched(src_cast_allocator.get(),
                               ggml_cann_type_mapping(dst->type), ggml_type_سائيز(dst->type),
                               src_cast_nb);
            }
            اكسر;
        حالة GGML_TYPE_Q8_0:
            {
                // Dequantize Q8_0 to dst type, then gather.
                size_t  وزن_nb[GGML_MAX_DIMS + 1], scale_nb[GGML_MAX_DIMS + 1], dequant_nb[GGML_MAX_DIMS + 1];
                int64_t وزن_ne[GGML_MAX_DIMS + 1], scale_ne[GGML_MAX_DIMS + 1], *dequant_ne;
                وزن_ne[0] = QK8_0;
                وزن_ne[1] = src0->ne[0] / QK8_0;
                وزن_nb[0] = حجم(int8_t);
                وزن_nb[1] = وزن_nb[0] * وزن_ne[0];
                لكل (صحيح i = 2; i < GGML_MAX_DIMS + 1; i++) {
                    وزن_ne[i] = src0->ne[i - 1];
                    وزن_nb[i] = وزن_nb[i - 1] * وزن_ne[i - 1];
                }
                scale_ne[0] = 1;
                scale_ne[1] = src0->ne[0] / QK8_0;
                scale_nb[0] = حجم(uint16_t);
                scale_nb[1] = scale_nb[0] * scale_ne[0];
                لكل (صحيح i = 2; i < GGML_MAX_DIMS + 1; i++) {
                    scale_ne[i] = src0->ne[i - 1];
                    scale_nb[i] = scale_nb[i - 1] * scale_ne[i - 1];
                }
                dequant_ne    = وزن_ne;
                dequant_nb[0] = ggml_type_سائيز(dst->type);
                لكل (صحيح i = 1; i < GGML_MAX_DIMS + 1; i++) {
                    dequant_nb[i] = dequant_nb[i - 1] * dequant_ne[i - 1];
                }
                ثابت int64_t scale_وٿي = ggml_nelements(src0) * حجم(int8_t);
                ggml_cann_pool_alloc dequant_allocator(ctx.pool(),
                                                       ggml_nelements(src0) * ggml_type_سائيز(dst->type));
                acl_tensor_ptr acl_وزن = ggml_cann_create_tensor(src0->معلومات, ACL_INT8, حجم(int8_t),
                                                                     وزن_ne, وزن_nb, GGML_MAX_DIMS + 1);
                acl_tensor_ptr acl_scale  = ggml_cann_create_tensor(
                    src0->معلومات, ACL_FLOAT16, حجم(uint16_t), scale_ne, scale_nb,
                    GGML_MAX_DIMS + 1, ACL_FORMAT_ND, scale_وٿي);
                acl_tensor_ptr acl_dequant = ggml_cann_create_tensor(
                    dequant_allocator.get(), ggml_cann_type_mapping(dst->type),
                    ggml_type_سائيز(dst->type), dequant_ne, dequant_nb, GGML_MAX_DIMS + 1);
                aclnn_mul(ctx, acl_وزن.get(), acl_scale.get(), acl_dequant.get());

                // Reinterpret dequant buffer as 4D [src0->ne] with contiguous strides.
                dequant_ne    = src0->ne;
                dequant_nb[0] = ggml_type_سائيز(dst->type);
                لكل (صحيح i = 1; i < GGML_MAX_DIMS; i++) {
                    dequant_nb[i] = dequant_nb[i - 1] * src0->ne[i - 1];
                }
                gather_batched(dequant_allocator.get(),
                               ggml_cann_type_mapping(dst->type), ggml_type_سائيز(dst->type),
                               dequant_nb);
                اكسر;
            }
        افتراضي:
            GGML_ABORT("Unsupported tensor type for GGML_OP_GET_ROWS");
            اكسر;
    }
}

فراغ ggml_cann_سيٽ_صفون(ggml_backend_cann_context & ctx, ggml_tensor * dst) {
    ggml_tensor * src0 = dst->src[0];  // source values
    ggml_tensor * src1 = dst->src[1];  // row indices

    // n_idx: number of source rows to scatter per batch slice.
    // ggml guarantees: src0->ne[1] == src1->ne[0].
    ثابت int64_t n_idx = src1->ne[0];

    // Copy n_idx rows of src [ne0, n_idx] into dst [ne0, ne1] at positions given by a 1D index.
    // ggml_cann_create_tensor reverses dims, so ACL sees [ne1, ne0] for dst.
    // InplaceIndexCopy with dim=0 copies along ACL dim-0 == ggml ne[1] (the row axis).
    // src_nb: the 4 strides of the source buffer (nb[0..1] for the 2D slice shape,
    //         nb[2..3] for computing per-batch-slice base pointer offsets).
    خودڪار scatter_batched = [&](فراغ * src_بنياد, acl_معلومات_Type acl_type, size_t type_سائيز,
                               ثابت size_t * src_nb) {
        int64_t d_ne[2]    = { dst->ne[0], dst->ne[1] };
        size_t  d_nb[2]    = { dst->nb[0], dst->nb[1] };
        int64_t s_ne[2]    = { dst->ne[0], n_idx };
        size_t  s_nb_2d[2] = { src_nb[0], src_nb[1] };
        int64_t i_ne[1]    = { n_idx };
        size_t  i_nb[1]    = { (size_t)ggml_element_سائيز(src1) };

        لكل (int64_t i3 = 0; i3 < dst->ne[3]; i3++) {
            لكل (int64_t i2 = 0; i2 < dst->ne[2]; i2++) {
                acl_tensor_ptr acl_dst = ggml_cann_create_tensor(
                    (حرف *)dst->معلومات + i3 * dst->nb[3] + i2 * dst->nb[2],
                    acl_type, type_سائيز, d_ne, d_nb, 2);
                acl_tensor_ptr acl_idx = ggml_cann_create_tensor(
                    (حرف *)src1->معلومات + (i3 % src1->ne[2]) * src1->nb[2] + (i2 % src1->ne[1]) * src1->nb[1],
                    ggml_cann_type_mapping(src1->type), (size_t)ggml_element_سائيز(src1),
                    i_ne, i_nb, 1);
                acl_tensor_ptr acl_src = ggml_cann_create_tensor(
                    (حرف *)src_بنياد + i3 * src_nb[3] + i2 * src_nb[2],
                    acl_type, type_سائيز, s_ne, s_nb_2d, 2);
                GGML_CANN_CALL_ACLNN_OP(ctx, Inplace_اشاريو_Copy, acl_dst.get(), 0, acl_idx.get(), acl_src.get());
            }
        }
    };

    بدل (dst->type) {
        حالة GGML_TYPE_F32:
            scatter_batched(src0->معلومات,
                            ggml_cann_type_mapping(dst->type), ggml_type_سائيز(dst->type),
                            src0->nb);
            اكسر;
        حالة GGML_TYPE_F16:
        حالة GGML_TYPE_BF16:
            {
                // Cast src0 (F32) to dst type first.
                ggml_cann_pool_alloc src_cast_allocator(ctx.pool(),
                                                        ggml_nelements(src0) * ggml_type_سائيز(dst->type));
                size_t src_cast_nb[GGML_MAX_DIMS];
                src_cast_nb[0] = ggml_type_سائيز(dst->type);
                لكل (صحيح i = 1; i < GGML_MAX_DIMS; i++) {
                    src_cast_nb[i] = src_cast_nb[i - 1] * src0->ne[i - 1];
                }
                acl_tensor_ptr acl_src0     = ggml_cann_create_tensor(src0);
                acl_tensor_ptr acl_src_cast = ggml_cann_create_tensor(
                    src_cast_allocator.get(), ggml_cann_type_mapping(dst->type), ggml_type_سائيز(dst->type),
                    src0->ne, src_cast_nb, GGML_MAX_DIMS);
                aclnn_cast(ctx, acl_src0.get(), acl_src_cast.get(), ggml_cann_type_mapping(dst->type));

                scatter_batched(src_cast_allocator.get(),
                                ggml_cann_type_mapping(dst->type), ggml_type_سائيز(dst->type),
                                src_cast_nb);
                اكسر;
            }
        افتراضي:
            GGML_ABORT("Unsupported tensor type for GGML_OP_SET_ROWS");
            اكسر;
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
ساكن فراغ aclnn_repeat_interleave(ggml_backend_cann_context & ctx,
                                    aclTensor *                 acl_src,
                                    aclTensor *                 acl_dst,
                                    int64_t                     dim,
                                    int64_t                     repeats,
                                    int64_t                     output_سائيز) {
    GGML_CANN_CALL_ACLNN_OP(ctx, RepeatInterleaveIntWithDim, acl_src, repeats, dim, output_سائيز, acl_dst);
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
ساكن فراغ ggml_cann_mat_mul_fp(ggml_backend_cann_context & ctx, ggml_tensor * dst) {
    ggml_tensor * وزن = dst->src[0];  //وزن
    ggml_tensor * داخلا_2  = dst->src[1];  //داخلا

    // when weight ne2 or ne3 is 1, aclnnMatmulGetWorkspaceSize will auto
    // broadcast, when weight ne2 or ne3 is not 1, weight need repeat.
    BCAST_MUL_MAT_SHAPE(داخلا_2, وزن, dst);

    int64_t n_dims = bcast_dims;
    إذا (bcast_داخلا_ne[3] == bcast_وزن_ne[3] && bcast_داخلا_ne[3] == 1) {
        إذا (bcast_داخلا_ne[2] == 1 && bcast_وزن_ne[2] == 1) {
            n_dims = 2;
        } وإلا إذا (bcast_داخلا_ne[2] == 1) {
            n_dims = 3;
        }
    }

    acl_tensor_ptr acl_داخلا_tensor = ggml_cann_create_tensor(داخلا_2, bcast_داخلا_ne, bcast_داخلا_nb, n_dims);
    int64_t        transpose_ne[]   = { bcast_وزن_ne[1], bcast_وزن_ne[0], bcast_وزن_ne[2],
                                        bcast_وزن_ne[3], bcast_وزن_ne[4], bcast_وزن_ne[5] };
    size_t         transpose_nb[]   = { bcast_وزن_nb[1], bcast_وزن_nb[0], bcast_وزن_nb[2],
                                        bcast_وزن_nb[3], bcast_وزن_nb[4], bcast_وزن_nb[5] };
    acl_tensor_ptr acl_وزن_tensor;

    // Only check env once.
    ساكن منطقي وزن_منزل_nz = parse_bool(get_env_as_lowercase("GGML_CANN_WEIGHT_NZ").value_or("on"));
    إذا (وزن_منزل_nz && وزن->type != GGML_TYPE_BF16 && is_matmul_وزن(وزن)) {
        acl_وزن_tensor = ggml_cann_create_tensor(وزن, transpose_ne, transpose_nb, n_dims, ACL_FORMAT_FRACTAL_NZ);
    } وإلا {
        acl_وزن_tensor = ggml_cann_create_tensor(وزن, transpose_ne, transpose_nb, n_dims, ACL_FORMAT_ND);
    }
    acl_tensor_ptr acl_dst = ggml_cann_create_tensor(dst, bcast_dst_ne, bcast_dst_nb, n_dims);

    بدل (n_dims) {
        حالة 2:
            GGML_CANN_CALL_ACLNN_OP(ctx, Mm, acl_داخلا_tensor.get(), acl_وزن_tensor.get(), acl_dst.get(), 2);
            اكسر;
        حالة 3:
            GGML_CANN_CALL_ACLNN_OP(ctx, BatchMatMul, acl_داخلا_tensor.get(), acl_وزن_tensor.get(), acl_dst.get(),
                                    2);
            اكسر;
        افتراضي:
            // ALLOW_FP32_DOWN_PRECISION, when input is
            // fp32, atlas a2 will transpose it to HFLOAT32.
            GGML_CANN_CALL_ACLNN_OP(ctx, Matmul, acl_داخلا_tensor.get(), acl_وزن_tensor.get(), acl_dst.get(), 1);
            اكسر;
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
ساكن فراغ ggml_cann_mul_mat_quant(ggml_backend_cann_context & ctx, ggml_tensor * dst, ثابت تعداد ggml_type type) {
    ggml_tensor * src0 = dst->src[0];  //وزن
    ggml_tensor * src1 = dst->src[1];  //داخلا

    // The shape of the weight is NCHW.
    // Matrix multiplication uses HW dims.
    // HC is regarded as batch.
    // weight need transpose.
    عائم وزن_elem_سائيز;
    إذا (type == GGML_TYPE_Q4_0) {
        وزن_elem_سائيز = عائم(حجم(uint8_t)) / 2;
    } وإلا إذا (type == GGML_TYPE_Q8_0) {
        وزن_elem_سائيز = عائم(حجم(uint8_t));
    } وإلا {
        GGML_ABORT("Only support Q4_0 and Q8_0 MUL_MAT");
    }
    عائم  وزن_nb[]   = { src0->ne[0] * وزن_elem_سائيز, وزن_elem_سائيز };
    size_t وزن_stride = src0->ne[1] * src0->ne[0] * وزن_elem_سائيز;
    size_t وزن_سائيز   = وزن_stride * src0->ne[2] * src0->ne[3];

    // scale stored at the end of weight. Also need transpose.
    size_t scale_elem_سائيز = حجم(uint16_t);
    size_t scale_nb[]      = { src0->ne[0] / QK8_0 * scale_elem_سائيز, scale_elem_سائيز };
    size_t scale_stride    = src0->ne[1] * src0->ne[0] / QK8_0 * scale_elem_سائيز;
    حرف * scale_وٿي    = (حرف *) src0->معلومات + وزن_سائيز;

    //داخلا
    size_t               داخلا_elem_سائيز = حجم(uint16_t);
    int64_t              داخلا_ne[]      = { src1->ne[0], src1->ne[1] };
    size_t               داخلا_nb[]      = { داخلا_elem_سائيز, داخلا_ne[0] * داخلا_elem_سائيز };
    size_t               داخلا_stride    = داخلا_ne[0] * داخلا_ne[1] * داخلا_elem_سائيز;
    ggml_cann_pool_alloc داخلا_alloctor(ctx.pool());
    فراغ *               داخلا_بفر = src1->معلومات;

    // case in
    إذا (src1->type != GGML_TYPE_F16) {
        acl_tensor_ptr acl_src1_tensor = ggml_cann_create_tensor(src1);
        داخلا_بفر                   = داخلا_alloctor.alloc(ggml_nelements(src1) * داخلا_elem_سائيز);

        int64_t * داخلا_cast_ne = src1->ne;
        size_t    داخلا_cast_nb[GGML_MAX_DIMS];
        داخلا_cast_nb[0] = حجم(uint16_t);
        لكل (صحيح i = 1; i < GGML_MAX_DIMS; i++) {
            داخلا_cast_nb[i] = داخلا_cast_nb[i - 1] * داخلا_cast_ne[i - 1];
        }

        acl_tensor_ptr acl_داخلا_tensor = ggml_cann_create_tensor(داخلا_بفر, ACL_FLOAT16, داخلا_elem_سائيز,
                                                                  داخلا_cast_ne, داخلا_cast_nb, GGML_MAX_DIMS);
        aclnn_cast(ctx, acl_src1_tensor.get(), acl_داخلا_tensor.get(), ACL_FLOAT16);
    }

    // output
    size_t               output_elem_سائيز = حجم(uint16_t);
    size_t               output_nb[]      = { output_elem_سائيز, dst->ne[0] * output_elem_سائيز };
    ggml_cann_pool_alloc output_allocator(ctx.pool());
    فراغ *               output_بفر = output_allocator.alloc(ggml_nelements(dst) * output_elem_سائيز);
    size_t               output_stride = dst->ne[0] * dst->ne[1] * output_elem_سائيز;

    // aclnn
    int64_t              max_elem_سائيز = 65535;
    int64_t              ورهايو_سائيز    = (src0->ne[1] / max_elem_سائيز) + 1;
    ggml_cann_pool_alloc workspace_allocator(ctx.pool());
    لكل (int64_t n1 = 0; n1 < src1->ne[3]; n1++) {
        لكل (int64_t c1 = 0; c1 < src1->ne[2]; c1++) {
            int64_t n0 = n1 / (src1->ne[3] / src0->ne[3]);
            int64_t c0 = c1 / (src1->ne[2] / src0->ne[2]);

            int64_t batch1 = (n1 * src1->ne[2]) + c1;
            int64_t batch0 = (n0 * src0->ne[2]) + c0;

            acl_tensor_ptr acl_داخلا_tensor = ggml_cann_create_tensor(
                (حرف *) داخلا_بفر + batch1 * داخلا_stride, ACL_FLOAT16, داخلا_elem_سائيز, داخلا_ne, داخلا_nb, 2);

            // first split
            int64_t وزن_ne_وٿي = 0;
            int64_t وزن_ne[2]     = { max_elem_سائيز > src0->ne[1] ? src0->ne[1] : max_elem_سائيز, src0->ne[0] };
            int64_t scale_ne_وٿي  = 0;
            int64_t scale_ne[2]      = { وزن_ne[0], وزن_ne[1] / QK8_0 };
            int64_t output_ne_وٿي = 0;
            int64_t output_ne[2]     = { وزن_ne[0], dst->ne[1] };

            acl_tensor_ptr acl_وزن_tensor =
                ggml_cann_create_tensor((حرف *) src0->معلومات + batch0 * وزن_stride, ggml_cann_type_mapping(type),
                                        وزن_elem_سائيز, وزن_ne, وزن_nb, 2, ACL_FORMAT_ND, وزن_ne_وٿي);
            acl_tensor_ptr acl_scale_tensor =
                ggml_cann_create_tensor(scale_وٿي + batch0 * scale_stride, ACL_FLOAT16, scale_elem_سائيز, scale_ne,
                                        scale_nb, 2, ACL_FORMAT_ND, scale_ne_وٿي);
            acl_tensor_ptr acl_output_tensor =
                ggml_cann_create_tensor((حرف *) output_بفر + batch1 * output_stride, ACL_FLOAT16, output_elem_سائيز,
                                        output_ne, output_nb, 2, ACL_FORMAT_ND, output_ne_وٿي);
            int64_t antiquant_گروهه_سائيز = 0;
            إذا (src0->ne[0] > QK8_0) {
                antiquant_گروهه_سائيز = QK8_0;
            }
            GGML_CANN_CALL_ACLNN_OP(ctx, وزن_Quant_Batch_Matmul_V2, acl_داخلا_tensor.get(), acl_وزن_tensor.get(),
                                    acl_scale_tensor.get(), خالي_اشارو, خالي_اشارو, خالي_اشارو, خالي_اشارو, antiquant_گروهه_سائيز,
                                    acl_output_tensor.get());

            // other splits
            لكل (int64_t ورهايو = 1; ورهايو < ورهايو_سائيز; ورهايو++) {
                وزن_ne_وٿي += وزن_elem_سائيز * وزن_ne[0] * وزن_ne[1];
                وزن_ne[0] =
                    max_elem_سائيز * (ورهايو + 1) > src0->ne[1] ? src0->ne[1] - (max_elem_سائيز * ورهايو) : max_elem_سائيز;
                scale_ne_وٿي += scale_elem_سائيز * scale_ne[0] * scale_ne[1];
                scale_ne[0] = وزن_ne[0];
                output_ne_وٿي += output_elem_سائيز * output_ne[0] * output_ne[1];
                output_ne[0] = وزن_ne[0];

                acl_وزن_tensor =
                    ggml_cann_create_tensor((حرف *) src0->معلومات + batch0 * وزن_stride, ggml_cann_type_mapping(type),
                                            وزن_elem_سائيز, وزن_ne, وزن_nb, 2, ACL_FORMAT_ND, وزن_ne_وٿي);
                acl_scale_tensor =
                    ggml_cann_create_tensor(scale_وٿي + batch0 * scale_stride, ACL_FLOAT16, scale_elem_سائيز,
                                            scale_ne, scale_nb, 2, ACL_FORMAT_ND, scale_ne_وٿي);
                acl_output_tensor =
                    ggml_cann_create_tensor((حرف *) output_بفر + batch1 * output_stride, ACL_FLOAT16,
                                            output_elem_سائيز, output_ne, output_nb, 2, ACL_FORMAT_ND, output_ne_وٿي);
                GGML_CANN_CALL_ACLNN_OP(ctx, وزن_Quant_Batch_Matmul_V2, acl_داخلا_tensor.get(), acl_وزن_tensor.get(),
                                        acl_scale_tensor.get(), خالي_اشارو, خالي_اشارو, خالي_اشارو, خالي_اشارو, antiquant_گروهه_سائيز,
                                        acl_output_tensor.get());
            }
        }
    }

    // cast out
    إذا (dst->type != GGML_TYPE_F16) {
        int64_t * output_cast_ne = dst->ne;
        size_t    output_cast_nb[GGML_MAX_DIMS];
        output_cast_nb[0] = حجم(uint16_t);
        لكل (صحيح i = 1; i < GGML_MAX_DIMS; i++) {
            output_cast_nb[i] = output_cast_nb[i - 1] * output_cast_ne[i - 1];
        }

        acl_tensor_ptr acl_output_tensor = ggml_cann_create_tensor(output_بفر, ACL_FLOAT16, output_elem_سائيز,
                                                                   output_cast_ne, output_cast_nb, GGML_MAX_DIMS);
        acl_tensor_ptr acl_dst_tensor    = ggml_cann_create_tensor(dst);
        aclnn_cast(ctx, acl_output_tensor.get(), acl_dst_tensor.get(), ggml_cann_type_mapping(dst->type));
    }
}

فراغ ggml_cann_mul_mat(ggml_backend_cann_context & ctx, ggml_tensor * dst) {
    ثابت تعداد ggml_type type = dst->src[0]->type;
    بدل (type) {
        حالة GGML_TYPE_F32:
        حالة GGML_TYPE_F16:
#جيڪڏهن_وصف_نه_ٿيل ASCEND_310P
        حالة GGML_TYPE_BF16:
#جيڪڏهن_پورو
            ggml_cann_mat_mul_fp(ctx, dst);
            اكسر;
        حالة GGML_TYPE_Q4_0:
        حالة GGML_TYPE_Q8_0:
            ggml_cann_mul_mat_quant(ctx, dst, type);
            اكسر;
        افتراضي:
            GGML_ABORT("Unsupported type for mul_mat");
            اكسر;
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
ساكن فراغ aclnn_roll(ggml_backend_cann_context & ctx,
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
ساكن فراغ aclnn_اشاريو_fill_tensor(ggml_backend_cann_context & ctx,
                                    aclTensor *                 acl_src,
                                    int64_t                     dim,
                                    int64_t *                   اشاريو,
                                    int64_t                     اشاريو_انگ,
                                    عائم                       value) {
    acl_int_array_ptr acl_اشاريو = ggml_cann_create_int_array(اشاريو, اشاريو_انگ);
    acl_scalar_ptr    acl_قدر = ggml_cann_create_scalar(&value, acl_معلومات_Type::ACL_FLOAT);
    GGML_CANN_CALL_ACLNN_OP(ctx, Inplace_اشاريو_Fill_Tensor, acl_src, dim, acl_اشاريو.get(), acl_قدر.get());
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
ساكن فراغ aclnn_rope_cache_init(ggml_backend_cann_context & ctx,
                                  ggml_tensor *               dst,
                                  عائم *                     corr_dims,
                                  عائم                       ext_ضرب_ڪندڙ,
                                  عائم                       theta_scale,
                                  عائم                       freq_scale,
                                  عائم                       attn_ضرب_ڪندڙ,
                                  منطقي                        is_neox,
                                  صحيح                         sections[4],
                                  منطقي                        mrope_استعمال_ٿيل,
                                  منطقي                        is_imrope,
                                  منطقي                        indep_sects,
                                  int64_t                     rope_dims) {
    ggml_tensor * src1 = dst->src[1];  // position
    ggml_tensor * src2 = dst->src[2];  // freq_factors

    int64_t theta_scale_ڊيگهائي = rope_dims / 2;
    int64_t position_ڊيگهائي    = dst->ne[2];

    // TODO: check theta_scale_length and position_length.
    إذا (src2 == خالي_اشارو && ctx.rope_cache.cached &&
        ctx.rope_cache.equal(theta_scale_ڊيگهائي, position_ڊيگهائي, ext_ضرب_ڪندڙ, theta_scale, freq_scale, attn_ضرب_ڪندڙ,
                             is_neox, indep_sects, mrope_استعمال_ٿيل, is_imrope, sections)) {
        // use cache.
        أرجع;
    }

    // Step0: calculate tensor shape.
    int64_t theta_scale_ne[] = { theta_scale_ڊيگهائي, 1, 1, 1 };
    size_t  theta_scale_nb[] = { حجم(عائم), theta_scale_ڊيگهائي * حجم(عائم), theta_scale_ڊيگهائي * حجم(عائم),
                                 theta_scale_ڊيگهائي * حجم(عائم) };

    GGML_ASSERT(src1->type == GGML_TYPE_I32);
    int64_t position_ne[] = { 1, 1, position_ڊيگهائي, 1 };
    size_t  position_nb[] = { حجم(int32_t), حجم(int32_t), حجم(int32_t), حجم(int32_t) * position_ڊيگهائي };

    int64_t cache_ne[] = { theta_scale_ڊيگهائي, 1, position_ڊيگهائي, 1 };
    size_t  cache_nb[GGML_MAX_DIMS];
    cache_nb[0] = حجم(عائم);
    لكل (صحيح i = 1; i < GGML_MAX_DIMS; i++) {
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
    منطقي           theta_scale_updated = خطأ;
    إذا (ctx.rope_cache.theta_scale_ڊيگهائي != theta_scale_ڊيگهائي || ctx.rope_cache.theta_scale != theta_scale ||
        ctx.rope_cache.indep_sects != indep_sects) {
        theta_scale_updated = سچ;
        إذا (ctx.rope_cache.theta_scale_exp_host != خالي_اشارو) {
            حرر(ctx.rope_cache.theta_scale_exp_host);
        }
        ctx.rope_cache.theta_scale_exp_host = (عائم *) يادگيري_ڏيو(theta_scale_ڊيگهائي * حجم(عائم));
        GGML_ASSERT(ctx.rope_cache.theta_scale_exp_host != خالي_اشارو);
        إذا (!indep_sects) {
            ctx.rope_cache.theta_scale_exp_host[0] = 1;
            لكل (صحيح i = 1; i < theta_scale_ڊيگهائي; i++) {
                ctx.rope_cache.theta_scale_exp_host[i] = ctx.rope_cache.theta_scale_exp_host[i - 1] * theta_scale;
            }
        } وإلا {
            صحيح sect_dims = sections[0] + sections[1] + sections[2] + sections[3];
            صحيح sec_w     = sections[1] + sections[0];
            صحيح sec_e     = sections[2] + sec_w;

            ctx.rope_cache.theta_scale_exp_host[0] = 1;
            لكل (صحيح i = 1; i < theta_scale_ڊيگهائي; i++) {
                صحيح sector = i % sect_dims;
                إذا (sector == 0 || sector == sections[0] || sector == sec_w || sector == sec_e) {
                    ctx.rope_cache.theta_scale_exp_host[i] = 1;
                    استمر;
                }
                ctx.rope_cache.theta_scale_exp_host[i] = ctx.rope_cache.theta_scale_exp_host[i - 1] * theta_scale;
            }
        }

        إذا (ctx.rope_cache.theta_scale_cache != خالي_اشارو) {
            ACL_CHECK(aclrtFree(ctx.rope_cache.theta_scale_cache));
        }
        ACL_CHECK(aclrtMalloc(&ctx.rope_cache.theta_scale_cache, theta_scale_ڊيگهائي * حجم(عائم),
                              ACL_MEM_MALLOC_HUGE_پهريون));

        ACL_CHECK(aclrtMemcpyAsync(ctx.rope_cache.theta_scale_cache, theta_scale_ڊيگهائي * حجم(عائم),
                                   ctx.rope_cache.theta_scale_exp_host, theta_scale_ڊيگهائي * حجم(عائم),
                                   ACL_MEMCPY_HOST_منزل_DEVICE, ctx.stream()));
    }
    acl_theta_scale_tensor = ggml_cann_create_tensor(ctx.rope_cache.theta_scale_cache, ACL_FLOAT, حجم(عائم),
                                                     theta_scale_ne, theta_scale_nb, 1);

    // Step1.2: prepare rope_yarn_ramp, if this part updated, should update theta_scale_tensor.
    // TODO: acl_yarn_ramp_tensor use rope cache.
    منطقي           yarn_ramp_tensor_updated = خطأ;
    acl_tensor_ptr acl_yarn_ramp_tensor;
    إذا (ext_ضرب_ڪندڙ != 0 && (theta_scale_updated || ctx.rope_cache.theta_scale_ڊيگهائي != theta_scale_ڊيگهائي ||
                            ctx.rope_cache.freq_scale != freq_scale)) {
        yarn_ramp_tensor_updated = سچ;
        إذا (ctx.rope_cache.yarn_ramp_cache != خالي_اشارو) {
            ACL_CHECK(aclrtFree(ctx.rope_cache.yarn_ramp_cache));
        }
        ACL_CHECK(aclrtMalloc(&ctx.rope_cache.yarn_ramp_cache, theta_scale_ڊيگهائي * حجم(عائم),
                              ACL_MEM_MALLOC_HUGE_پهريون));
        // -rope_yarn_ramp
        // const float y = (i0 / 2 - low) / MAX(0.001f, high - low);
        // return MIN(1, MAX(0, y)) - 1;
        acl_yarn_ramp_tensor      = ggml_cann_create_tensor(ctx.rope_cache.yarn_ramp_cache, ACL_FLOAT, حجم(عائم),
                                                            theta_scale_ne, theta_scale_nb, 1);
        عائم          zero_قدر = 0, one_قدر = 1;
        عائم          denom_محفوظ_قدر = MAX(0.001f, corr_dims[1] - corr_dims[0]);
        acl_scalar_ptr هيٺيون              = ggml_cann_create_scalar(&corr_dims[0], acl_معلومات_Type::ACL_FLOAT);
        acl_scalar_ptr zero             = ggml_cann_create_scalar(&zero_قدر, acl_معلومات_Type::ACL_FLOAT);
        acl_scalar_ptr one              = ggml_cann_create_scalar(&one_قدر, acl_معلومات_Type::ACL_FLOAT);
        acl_scalar_ptr denom_محفوظ       = ggml_cann_create_scalar(&denom_محفوظ_قدر, acl_معلومات_Type::ACL_FLOAT);
        acl_scalar_ptr ext_ضرب_ڪندڙ_sc    = ggml_cann_create_scalar(&ext_ضرب_ڪندڙ, acl_معلومات_Type::ACL_FLOAT);

        aclnn_arange(ctx, acl_yarn_ramp_tensor.get(), 0, theta_scale_ڊيگهائي, 1, theta_scale_ڊيگهائي);
        GGML_CANN_CALL_ACLNN_OP(ctx, InplaceSubs, acl_yarn_ramp_tensor.get(), هيٺيون.get(), one.get());
        GGML_CANN_CALL_ACLNN_OP(ctx, InplaceDivs, acl_yarn_ramp_tensor.get(), denom_محفوظ.get());
        GGML_CANN_CALL_ACLNN_OP(ctx, Inplace_حد, acl_yarn_ramp_tensor.get(), zero.get(), zero.get());
        GGML_CANN_CALL_ACLNN_OP(ctx, InplaceClampMax, acl_yarn_ramp_tensor.get(), one.get());
        GGML_CANN_CALL_ACLNN_OP(ctx, InplaceSubs, acl_yarn_ramp_tensor.get(), one.get(), one.get());
        GGML_CANN_CALL_ACLNN_OP(ctx, InplaceMuls, acl_yarn_ramp_tensor.get(), ext_ضرب_ڪندڙ_sc.get());

        // theta_interp = freq_scale * theta_extrap;
        // theta = theta_interp * (1 - ramp_mix) + theta_extrap * ramp_mix;
        // theta = freq_scale * theta_extrap * (1 - ramp_mix) + theta_extrap * ramp_mix;
        // theta = freq_scale * theta_extrap - freq_scale * theta_extrap * ramp_mix + theta_extrap * ramp_mix;
        // theta = theta_extrap * (freq_scale - freq_scale * ramp_mix + ramp_mix);
        //
        // we cache (freq_scale - freq_scale * ramp_mix + ramp_mix), Considering that the rope_yarn_ramp here is the inverse
        // cache freq_scale + (freq_scale - 1) * ramp_mix
        عائم          freq_scale_1    = freq_scale - 1;
        acl_scalar_ptr freq_scale_sc   = ggml_cann_create_scalar(&freq_scale, acl_معلومات_Type::ACL_FLOAT);
        acl_scalar_ptr freq_scale_1_sc = ggml_cann_create_scalar(&freq_scale_1, acl_معلومات_Type::ACL_FLOAT);
        GGML_CANN_CALL_ACLNN_OP(ctx, InplaceMuls, acl_yarn_ramp_tensor.get(), freq_scale_1_sc.get());
        GGML_CANN_CALL_ACLNN_OP(ctx, InplaceAdds, acl_yarn_ramp_tensor.get(), freq_scale_sc.get(), one.get());
    } وإلا {
        acl_yarn_ramp_tensor = ggml_cann_create_tensor(ctx.rope_cache.yarn_ramp_cache, ACL_FLOAT, حجم(عائم),
                                                       theta_scale_ne, theta_scale_nb, 1);
    }
    // Step 1.3: update theta_scale_tensor according to ext_factor or freq_scale.
    إذا (ext_ضرب_ڪندڙ != 0) {
        إذا (theta_scale_updated || yarn_ramp_tensor_updated) {
            theta_scale_updated = سچ;
            aclnn_mul(ctx, acl_theta_scale_tensor.get(), acl_yarn_ramp_tensor.get());
        }
    } وإلا {
        إذا (freq_scale != 1 && (ctx.rope_cache.freq_scale != freq_scale || theta_scale_updated)) {
            theta_scale_updated = سچ;
            aclnn_muls(ctx, acl_theta_scale_tensor.get(), freq_scale, خالي_اشارو, سچ);
        }
    }

    // Nothing changed, use cache.
    إذا (!theta_scale_updated) {
        acl_theta_scale_tensor = ggml_cann_create_tensor(ctx.rope_cache.theta_scale_cache, ACL_FLOAT, حجم(عائم),
                                                         theta_scale_ne, theta_scale_nb, GGML_MAX_DIMS);
    }

    // Step 1.4: prepare select index if mrope
    acl_tensor_ptr position_select_اشاريو_tensor;
    إذا (mrope_استعمال_ٿيل) {
        إذا (ctx.rope_cache.sections[0] != sections[0] || ctx.rope_cache.sections[1] != sections[1] ||
            ctx.rope_cache.sections[2] != sections[2] || ctx.rope_cache.sections[3] != sections[3] ||
            ctx.rope_cache.theta_scale_ڊيگهائي != theta_scale_ڊيگهائي || ctx.rope_cache.is_imrope != is_imrope) {
            إذا (ctx.rope_cache.position_select_اشاريو_host != خالي_اشارو) {
                حرر(ctx.rope_cache.position_select_اشاريو_host);
            }
            ctx.rope_cache.position_select_اشاريو_host = (صحيح *) يادگيري_ڏيو(theta_scale_ڊيگهائي * حجم(صحيح));
            GGML_ASSERT(ctx.rope_cache.position_select_اشاريو_host != خالي_اشارو);
            صحيح sect_dims = sections[0] + sections[1] + sections[2] + sections[3];
            صحيح sec_w     = sections[1] + sections[0];
            صحيح sec_e     = sections[2] + sec_w;
            // t,h,w,e
            لكل (صحيح i = 0; i < theta_scale_ڊيگهائي; i++) {
                صحيح sector = i % sect_dims;

                إذا (is_imrope) {  // qwen3vl apply interleaved mrope
                    إذا (sector % 3 == 1 && sector < 3 * sections[1]) {
                        ctx.rope_cache.position_select_اشاريو_host[i] = 1;
                    } وإلا إذا (sector % 3 == 2 && sector < 3 * sections[2]) {
                        ctx.rope_cache.position_select_اشاريو_host[i] = 2;
                    } وإلا إذا (sector % 3 == 0 && sector < 3 * sections[0]) {
                        ctx.rope_cache.position_select_اشاريو_host[i] = 0;
                    } وإلا {
                        ctx.rope_cache.position_select_اشاريو_host[i] = 3;
                    }
                } وإلا {
                    إذا (sector >= sections[0] && sector < sec_w) {
                        ctx.rope_cache.position_select_اشاريو_host[i] = 1;
                    } وإلا إذا (sector >= sec_w && sector < sec_e) {
                        ctx.rope_cache.position_select_اشاريو_host[i] = 2;
                    } وإلا إذا (sector >= sec_e) {
                        ctx.rope_cache.position_select_اشاريو_host[i] = 3;
                    } وإلا {
                        ctx.rope_cache.position_select_اشاريو_host[i] = 0;
                    }
                }
            }

            إذا (ctx.rope_cache.position_select_اشاريو != خالي_اشارو) {
                ACL_CHECK(aclrtFree(ctx.rope_cache.position_select_اشاريو));
            }
            ACL_CHECK(aclrtMalloc(&ctx.rope_cache.position_select_اشاريو, theta_scale_ڊيگهائي * حجم(صحيح),
                                  ACL_MEM_MALLOC_HUGE_پهريون));

            ACL_CHECK(aclrtMemcpyAsync(ctx.rope_cache.position_select_اشاريو, theta_scale_ڊيگهائي * حجم(صحيح),
                                       ctx.rope_cache.position_select_اشاريو_host, theta_scale_ڊيگهائي * حجم(صحيح),
                                       ACL_MEMCPY_HOST_منزل_DEVICE, ctx.stream()));
        }

        position_select_اشاريو_tensor = ggml_cann_create_tensor(ctx.rope_cache.position_select_اشاريو, ACL_INT32,
                                                               حجم(صحيح), theta_scale_ne, theta_scale_nb, 1);
    }

    // Step2: divide by freq_factors
    ggml_cann_pool_alloc freq_fac_res_allocator(ctx.pool());
    إذا (src2) {
        freq_fac_res_allocator.alloc(theta_scale_ڊيگهائي * حجم(عائم));
        فراغ *         freq_fac_res_ptr = freq_fac_res_allocator.get();
        acl_tensor_ptr acl_freq_factors_tensor =
            ggml_cann_create_tensor(src2->معلومات, ggml_cann_type_mapping(src2->type), ggml_type_سائيز(src2->type),
                                    theta_scale_ne, theta_scale_nb, GGML_MAX_DIMS);
        acl_tensor_ptr acl_freq_fac_res_tensor = ggml_cann_create_tensor(freq_fac_res_ptr, ACL_FLOAT, حجم(عائم),
                                                                         theta_scale_ne, theta_scale_nb, GGML_MAX_DIMS);
        aclnn_div(ctx, acl_theta_scale_tensor.get(), acl_freq_factors_tensor.get(), acl_freq_fac_res_tensor.get());
        std::swap(acl_theta_scale_tensor, acl_freq_fac_res_tensor);
    }

    // Step3: prepare position_tensor
    acl_tensor_ptr       acl_position_tensor;
    ggml_cann_pool_alloc mrope_position_acllocator(ctx.pool());
    إذا (mrope_استعمال_ٿيل) {
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

        int64_t        mrope_position_ne[] = { position_ڊيگهائي, 4 };
        size_t         mrope_position_nb[] = { حجم(صحيح), position_ڊيگهائي * حجم(صحيح) };
        acl_tensor_ptr mrope_position =
            ggml_cann_create_tensor(src1->معلومات, ggml_cann_type_mapping(src1->type), ggml_type_سائيز(src1->type),
                                    mrope_position_ne, mrope_position_nb, 2);

        // selected position tensor's shape is a transpose of cache tensor.
        int64_t selected_position_ne[] = { position_ڊيگهائي, theta_scale_ڊيگهائي };
        size_t  selected_position_nb[] = { حجم(عائم), position_ڊيگهائي * حجم(عائم) };
        mrope_position_acllocator.alloc(theta_scale_ڊيگهائي * position_ڊيگهائي * حجم(عائم));
        فراغ * mrope_position_بفر = mrope_position_acllocator.get();
        acl_position_tensor =
            ggml_cann_create_tensor(mrope_position_بفر, ggml_cann_type_mapping(src1->type),
                                    ggml_type_سائيز(src1->type), selected_position_ne, selected_position_nb, 2);
        GGML_CANN_CALL_ACLNN_OP(ctx, اشاريو_Select, mrope_position.get(), 0, position_select_اشاريو_tensor.get(),
                                acl_position_tensor.get());

        // transpose
        int64_t transposed_ne[] = { position_ڊيگهائي, 1, theta_scale_ڊيگهائي, 1 };
        size_t  transposed_nb[GGML_MAX_DIMS];
        transposed_nb[0] = حجم(عائم);
        لكل (صحيح i = 1; i < GGML_MAX_DIMS; i++) {
            transposed_nb[i] = transposed_nb[i - 1] * transposed_ne[i - 1];
        }

        std::swap(transposed_ne[0], transposed_ne[2]);
        std::swap(transposed_nb[0], transposed_nb[2]);

        acl_position_tensor =
            ggml_cann_create_tensor(mrope_position_بفر, ggml_cann_type_mapping(src1->type),
                                    ggml_type_سائيز(src1->type), transposed_ne, transposed_nb, GGML_MAX_DIMS);

    } وإلا {
        // auto bcast.
        acl_position_tensor =
            ggml_cann_create_tensor(src1->معلومات, ggml_cann_type_mapping(src1->type), ggml_type_سائيز(src1->type),
                                    position_ne, position_nb, GGML_MAX_DIMS);
    }

    // Step4: multiply by the position
    int64_t              theta_ڊيگهائي = theta_scale_ڊيگهائي * position_ڊيگهائي;
    ggml_cann_pool_alloc theta_allocator(ctx.pool(), theta_ڊيگهائي * حجم(عائم));
    فراغ *               theta_بفر = theta_allocator.get();

    acl_tensor_ptr acl_theta_tensor =
        ggml_cann_create_tensor(theta_بفر, ACL_FLOAT, حجم(عائم), cache_ne, cache_nb, GGML_MAX_DIMS);
    aclnn_mul(ctx, acl_position_tensor.get(), acl_theta_scale_tensor.get(), acl_theta_tensor.get());

    // Step5: calculate sin cos.
    // init sin_repeat && cos_repeat, only to accelerate first layer on each device
    إذا (position_ڊيگهائي > ctx.rope_cache.position_ڊيگهائي) {
        ctx.rope_cache.position_ڊيگهائي = position_ڊيگهائي;
        إذا (ctx.rope_cache.sin_cache != خالي_اشارو) {
            ACL_CHECK(aclrtFree(ctx.rope_cache.sin_cache));
        }
        إذا (ctx.rope_cache.cos_cache != خالي_اشارو) {
            ACL_CHECK(aclrtFree(ctx.rope_cache.cos_cache));
        }
        int64_t repeat_theta_ڊيگهائي = theta_scale_ڊيگهائي * position_ڊيگهائي * 2;
        ACL_CHECK(
            aclrtMalloc(&ctx.rope_cache.sin_cache, repeat_theta_ڊيگهائي * حجم(عائم), ACL_MEM_MALLOC_HUGE_پهريون));
        ACL_CHECK(
            aclrtMalloc(&ctx.rope_cache.cos_cache, repeat_theta_ڊيگهائي * حجم(عائم), ACL_MEM_MALLOC_HUGE_پهريون));
    }

    // sin/cos
    ggml_cann_pool_alloc sin_allocator(ctx.pool(), theta_ڊيگهائي * حجم(عائم));
    فراغ *               sin_بفر = sin_allocator.get();
    acl_tensor_ptr       acl_sin_tensor =
        ggml_cann_create_tensor(sin_بفر, ACL_FLOAT, حجم(عائم), cache_ne, cache_nb, GGML_MAX_DIMS, ACL_FORMAT_ND);
    aclnn_sin(ctx, acl_theta_tensor.get(), acl_sin_tensor.get());

    ggml_cann_pool_alloc cos_allocator(ctx.pool(), theta_ڊيگهائي * حجم(عائم));
    فراغ *               cos_بفر = cos_allocator.get();
    acl_tensor_ptr       acl_cos_tensor =
        ggml_cann_create_tensor(cos_بفر, ACL_FLOAT, حجم(عائم), cache_ne, cache_nb, GGML_MAX_DIMS, ACL_FORMAT_ND);
    aclnn_cos(ctx, acl_theta_tensor.get(), acl_cos_tensor.get());

    إذا (ext_ضرب_ڪندڙ != 0) {
        attn_ضرب_ڪندڙ *= 1.0f + 0.1f * logf(1.0f / freq_scale);
    }

    // Step 5: multiply by attn_factor
    إذا (attn_ضرب_ڪندڙ != 1) {
        aclnn_muls(ctx, acl_sin_tensor.get(), attn_ضرب_ڪندڙ, خالي_اشارو, سچ);
        aclnn_muls(ctx, acl_cos_tensor.get(), attn_ضرب_ڪندڙ, خالي_اشارو, سچ);
    }

    int64_t sin_reshape_ne[4] = { rope_dims, 1, dst->ne[2], 1 };
    size_t  sin_reshape_nb[GGML_MAX_DIMS];
    sin_reshape_nb[0] = حجم(عائم);
    لكل (صحيح i = 1; i < GGML_MAX_DIMS; i++) {
        sin_reshape_nb[i] = sin_reshape_nb[i - 1] * sin_reshape_ne[i - 1];
    }
    acl_tensor_ptr acl_sin_repeat_tensor = ggml_cann_create_tensor(ctx.rope_cache.sin_cache, ACL_FLOAT, حجم(عائم),
                                                                   sin_reshape_ne, sin_reshape_nb, GGML_MAX_DIMS);
    acl_tensor_ptr acl_cos_repeat_tensor = ggml_cann_create_tensor(ctx.rope_cache.cos_cache, ACL_FLOAT, حجم(عائم),
                                                                   sin_reshape_ne, sin_reshape_nb, GGML_MAX_DIMS);

    // Step 6: repeat
    إذا (is_neox) {
        // [sinθ1, sinθ1, sinθ2, sinθ2, ..., sinθn, sinθn]
        int64_t repeatsArray[] = { 1, 1, 1, 2 };
        aclnn_repeat(ctx, acl_sin_tensor.get(), acl_sin_repeat_tensor.get(), repeatsArray);
        aclnn_repeat(ctx, acl_cos_tensor.get(), acl_cos_repeat_tensor.get(), repeatsArray);
    } وإلا {
        int64_t انگ_repeats = 2;
        int64_t dim         = 3;
        int64_t output_سائيز = theta_scale_ڊيگهائي * انگ_repeats;
        // [sinθ1, sinθ2, ..., sinθn, sinθ1, sinθ2, ..., sinθn]
        aclnn_repeat_interleave(ctx, acl_sin_tensor.get(), acl_sin_repeat_tensor.get(), dim, انگ_repeats, output_سائيز);
        aclnn_repeat_interleave(ctx, acl_cos_tensor.get(), acl_cos_repeat_tensor.get(), dim, انگ_repeats, output_سائيز);
    }

    // Update cached value.
    ctx.rope_cache.cached = سچ;
    ctx.rope_cache.set(theta_scale_ڊيگهائي, position_ڊيگهائي, ext_ضرب_ڪندڙ, theta_scale, freq_scale, attn_ضرب_ڪندڙ, is_neox,
                       indep_sects, mrope_استعمال_ٿيل, is_imrope, sections);
}

#جيڪڏهن_وصف_ٿيل __cplusplus
خارجي "C" {
#جيڪڏهن_پورو
aclnn_حالت aclnn_Rotary_Position_Embedding_Get_Workspace_سائيز(ثابت aclTensor * x,
                                                         ثابت aclTensor * جيب_تمام,
                                                         ثابت aclTensor * جيب,
                                                         int64_t           mode,
                                                         ثابت aclTensor * y_ٻاهريون,
                                                         uint64_t *        workspace_سائيز,
                                                         aclOpExecutor **  executor);
aclnn_حالت aclnnRotaryPositionEmbedding(فراغ *          workspace,
                                         uint64_t        workspace_سائيز,
                                         aclOpExecutor * executor,
                                         aclrt_وهڪرو     stream);
#جيڪڏهن_وصف_ٿيل __cplusplus
}
#جيڪڏهن_پورو

فراغ ggml_cann_rope(ggml_backend_cann_context & ctx, ggml_tensor * dst) {
    ggml_tensor * src0 = dst->src[0];  //داخلا

    // param
    عائم     freq_بنياد, freq_scale, ext_ضرب_ڪندڙ, attn_ضرب_ڪندڙ, beta_fast, beta_slow;
    صحيح       sections[4];
    // const int n_past     = ((int32_t *) dst->op_params)[0];
    ثابت صحيح n_dims     = ((int32_t *) dst->op_params)[1];
    ثابت صحيح mode       = ((int32_t *) dst->op_params)[2];
    // const int n_ctx      = ((int32_t *) dst->op_params)[3];
    ثابت صحيح n_ctx_orig = ((int32_t *) dst->op_params)[4];

    GGML_TENSOR_UNARY_OP_LOCALS

    يادگيري_نقل(&freq_بنياد, (int32_t *) dst->op_params + 5, حجم(عائم));
    يادگيري_نقل(&freq_scale, (int32_t *) dst->op_params + 6, حجم(عائم));
    يادگيري_نقل(&ext_ضرب_ڪندڙ, (int32_t *) dst->op_params + 7, حجم(عائم));
    يادگيري_نقل(&attn_ضرب_ڪندڙ, (int32_t *) dst->op_params + 8, حجم(عائم));
    يادگيري_نقل(&beta_fast, (int32_t *) dst->op_params + 9, حجم(عائم));
    يادگيري_نقل(&beta_slow, (int32_t *) dst->op_params + 10, حجم(عائم));
    يادگيري_نقل(&sections, (int32_t *) dst->op_params + 11, حجم(صحيح) * 4);

    GGML_ASSERT(n_dims % 2 == 0);
    GGML_ASSERT(n_dims <= ne00);

    ثابت عائم theta_scale = powf(freq_بنياد, -2.0f / n_dims);

    عائم corr_dims[2];
    ggml_rope_yarn_corr_dims(n_dims, n_ctx_orig, freq_بنياد, beta_fast, beta_slow, corr_dims);

    منطقي       is_neox    = mode & GGML_ROPE_TYPE_NEOX;
    ثابت منطقي is_imrope  = mode == GGML_ROPE_TYPE_IMROPE;  // qwen3vl apply interleaved mrope
    // mrope_used means the GGML_ROPE_TYPE_MROPE bit is set.
    // Note: this bit is also set for imrope and some vision modes,
    // so mrope_used does NOT exclusively indicate pure mrope.
    ثابت منطقي mrope_استعمال_ٿيل = mode & GGML_ROPE_TYPE_MROPE;
    ثابت منطقي is_vision  = mode == GGML_ROPE_TYPE_VISION;

    إذا (mrope_استعمال_ٿيل) {
        GGML_ASSERT(sections[0] > 0 || sections[1] > 0 || sections[2] > 0);
    }

    إذا (is_vision) {
        GGML_ASSERT(n_dims == ne0 / 2);
    }

    إذا (is_imrope || mrope_استعمال_ٿيل) {
        is_neox = سچ;
    }

    int64_t rope_dims = n_dims;

    //Our current RotaryPositionEmbedding does not support the VISION mode,
    //but essentially it only modifies theta_base in mrope,
    //then repeats it at the end in the same way as is_neox.
    //In fact, RoPE is still applied across all dimensions.
    إذا (is_vision) {
        rope_dims = src0->ne[0];
    }
    int64_t پڇ_dims = ne00 - rope_dims;
    منطقي    has_پڇ  = پڇ_dims > 0;

    // init ctx.rope_cos/rope_sin cache
    aclnn_rope_cache_init(ctx, dst, corr_dims, ext_ضرب_ڪندڙ, theta_scale, freq_scale, attn_ضرب_ڪندڙ, is_neox, sections,
                          mrope_استعمال_ٿيل, is_imrope, is_vision, rope_dims);

    // Cache is generated with ne00 dimensions, so we use ne00 for reshape
    int64_t sin_reshape_ne[4] = { rope_dims, 1, ne02, 1 };
    size_t  sin_reshape_nb[GGML_MAX_DIMS];
    sin_reshape_nb[0] = حجم(عائم);
    لكل (صحيح i = 1; i < GGML_MAX_DIMS; i++) {
        sin_reshape_nb[i] = sin_reshape_nb[i - 1] * sin_reshape_ne[i - 1];
    }
    acl_tensor_ptr acl_sin_reshape_tensor = ggml_cann_create_tensor(ctx.rope_cache.sin_cache, ACL_FLOAT, حجم(عائم),
                                                                    sin_reshape_ne, sin_reshape_nb, GGML_MAX_DIMS);
    acl_tensor_ptr acl_cos_reshape_tensor = ggml_cann_create_tensor(ctx.rope_cache.cos_cache, ACL_FLOAT, حجم(عائم),
                                                                    sin_reshape_ne, sin_reshape_nb, GGML_MAX_DIMS);

    acl_tensor_ptr acl_src = ggml_cann_create_tensor(src0);
    acl_tensor_ptr acl_dst = ggml_cann_create_tensor(dst);
#جيڪڏهن_وصف_ٿيل ASCEND_310P
    // Special ROPE operation for 310P

    // roll input
    فراغ *               داخلا_roll_بفر;
    acl_tensor_ptr       acl_minus_one_tensor;
    فراغ *               minus_one_scale_بفر = خالي_اشارو;
    ggml_cann_pool_alloc roll_allocator(ctx.pool(), ggml_nbytes(src0));
    ggml_cann_pool_alloc minus_one_scale_allocator(ctx.pool(), حجم(عائم) * src0->ne[0]);
    إذا (!is_neox) {
        // roll input: [q0,q1,q2,q3,...] -> [q1,q0,q3,q2,...]
        داخلا_roll_بفر        = roll_allocator.get();
        int64_t داخلا_roll_ne[4] = { 2, src0->ne[1] * (src0->ne[0] / 2), src0->ne[2], src0->ne[3] };
        size_t  داخلا_roll_nb[GGML_MAX_DIMS];
        داخلا_roll_nb[0] = ggml_type_سائيز(src0->type);
        لكل (صحيح i = 1; i < GGML_MAX_DIMS; i++) {
            داخلا_roll_nb[i] = داخلا_roll_nb[i - 1] * داخلا_roll_ne[i - 1];
        }
        acl_tensor_ptr acl_داخلا_roll_tensor =
            ggml_cann_create_tensor(داخلا_roll_بفر, ggml_cann_type_mapping(src0->type), ggml_type_سائيز(src0->type),
                                    داخلا_roll_ne, داخلا_roll_nb, GGML_MAX_DIMS);
        acl_tensor_ptr acl_داخلا_tensor =
            ggml_cann_create_tensor(src0->معلومات, ggml_cann_type_mapping(src0->type), ggml_type_سائيز(src0->type),
                                    داخلا_roll_ne, داخلا_roll_nb, GGML_MAX_DIMS);

        int64_t shifts[] = { 1 };
        int64_t dims[]   = { 3 };
        aclnn_roll(ctx, acl_داخلا_tensor.get(), acl_داخلا_roll_tensor.get(), shifts, dims);

        // init [-1, 1, -1, 1, ...]
        minus_one_scale_بفر = minus_one_scale_allocator.get();

        int64_t minus_one_ne[4] = { src0->ne[0], 1, 1, 1 };
        size_t  minus_one_nb[GGML_MAX_DIMS];
        minus_one_nb[0] = حجم(عائم);
        لكل (صحيح i = 1; i < GGML_MAX_DIMS; i++) {
            minus_one_nb[i] = minus_one_nb[i - 1] * minus_one_ne[i - 1];
        }
        acl_minus_one_tensor = aclnn_قدرون(ctx, minus_one_scale_بفر, حجم(عائم) * src0->ne[0], minus_one_ne,
                                            GGML_MAX_DIMS, ACL_FLOAT, حجم(عائم), 1);
        int64_t   dim        = 3;
        int64_t * اشاريو      = جديد int64_t[src0->ne[0]];
        لكل (صحيح i = 0; i < src0->ne[0]; i++) {
            اشاريو[i] = i / 2 * 2;
        }
        int64_t اشاريو_انگ = src0->ne[0];
        عائم   value     = -1;
        aclnn_اشاريو_fill_tensor(ctx, acl_minus_one_tensor.get(), dim, اشاريو, اشاريو_انگ, value);
    } وإلا {
        // roll input: [q0,q1,q2,...] ->
        // [q_half,q_half+1,...,q_end,q0,q1,...q_half-1]
        داخلا_roll_بفر = roll_allocator.get();
        acl_tensor_ptr acl_داخلا_roll_tensor =
            ggml_cann_create_tensor(داخلا_roll_بفر, ggml_cann_type_mapping(src0->type), ggml_type_سائيز(src0->type),
                                    src0->ne, src0->nb, GGML_MAX_DIMS);
        acl_tensor_ptr acl_داخلا_tensor = ggml_cann_create_tensor(src0);

        int64_t shifts[] = { src0->ne[0] / 2 };
        int64_t dims[]   = { 3 };
        aclnn_roll(ctx, acl_داخلا_tensor.get(), acl_داخلا_roll_tensor.get(), shifts, dims);

        // init [-1, -1, -1, 1, 1，1，...]
        minus_one_scale_بفر  = minus_one_scale_allocator.get();
        int64_t minus_one_ne[4] = { src0->ne[0], 1, 1, 1 };
        size_t  minus_one_nb[GGML_MAX_DIMS];
        minus_one_nb[0] = حجم(عائم);
        لكل (صحيح i = 1; i < GGML_MAX_DIMS; i++) {
            minus_one_nb[i] = minus_one_nb[i - 1] * minus_one_ne[i - 1];
        }
        acl_minus_one_tensor     = aclnn_قدرون(ctx, minus_one_scale_بفر, حجم(عائم) * src0->ne[0], minus_one_ne,
                                                GGML_MAX_DIMS, ACL_FLOAT, حجم(عائم), 1);
        // -1 * first half
        int64_t پهريون_half_ne[4] = { src0->ne[0] / 2, 1, 1, 1 };
        size_t  پهريون_half_nb[GGML_MAX_DIMS];
        پهريون_half_nb[0] = حجم(عائم);
        لكل (صحيح i = 1; i < GGML_MAX_DIMS; i++) {
            پهريون_half_nb[i] = پهريون_half_nb[i - 1] * پهريون_half_ne[i - 1];
        }
        acl_tensor_ptr acl_پهريون_half_tensor = ggml_cann_create_tensor(minus_one_scale_بفر, ACL_FLOAT, حجم(عائم),
                                                                       پهريون_half_ne, پهريون_half_nb, GGML_MAX_DIMS);
        منطقي           inplace               = سچ;
        عائم          scale                 = -1;
        aclnn_muls(ctx, acl_پهريون_half_tensor.get(), scale, خالي_اشارو, inplace);
    }

    // TODO: n_dims < ne0
    GGML_ASSERT(n_dims == src0->ne[0]);

    // input * scale
    ggml_cann_pool_alloc roll_mul_scale_allocator(ctx.pool(), ggml_nbytes(src0));
    فراغ *               داخلا_roll_mul_scale_بفر = roll_mul_scale_allocator.get();
    size_t               داخلا_nb[GGML_MAX_DIMS];
    داخلا_nb[0] = ggml_type_سائيز(src0->type);
    لكل (صحيح i = 1; i < GGML_MAX_DIMS; i++) {
        داخلا_nb[i] = داخلا_nb[i - 1] * src0->ne[i - 1];
    }
    acl_tensor_ptr acl_داخلا_roll_mul_scale_tensor =
        ggml_cann_create_tensor(داخلا_roll_mul_scale_بفر, ggml_cann_type_mapping(src0->type),
                                ggml_type_سائيز(src0->type), src0->ne, داخلا_nb, GGML_MAX_DIMS);
    acl_tensor_ptr acl_داخلا_roll_reshape_tensor =
        ggml_cann_create_tensor(داخلا_roll_بفر, ggml_cann_type_mapping(src0->type), ggml_type_سائيز(src0->type),
                                src0->ne, داخلا_nb, GGML_MAX_DIMS);

    aclnn_mul(ctx, acl_داخلا_roll_reshape_tensor.get(), acl_minus_one_tensor.get(),
              acl_داخلا_roll_mul_scale_tensor.get());

    // output
    فراغ * output_fp32_بفر;
    إذا (src0->type == GGML_TYPE_F32) {
        aclnn_mul(ctx, acl_src.get(), acl_cos_reshape_tensor.get());
        aclnn_mul(ctx, acl_داخلا_roll_mul_scale_tensor.get(), acl_sin_reshape_tensor.get());
        aclnn_جوڙيو(ctx, acl_src.get(), acl_داخلا_roll_mul_scale_tensor.get(), acl_dst.get());
        // TODO: ne0 != n_dims in mode2
    } وإلا إذا (src0->type == GGML_TYPE_F16) {
        size_t داخلا_fp32_nb[GGML_MAX_DIMS];
        داخلا_fp32_nb[0] = حجم(عائم);
        لكل (صحيح i = 1; i < GGML_MAX_DIMS; i++) {
            داخلا_fp32_nb[i] = داخلا_fp32_nb[i - 1] * dst->ne[i - 1];
        }
        ggml_cann_pool_alloc fp32_allocator1(ctx.pool(), ggml_nelements(dst) * حجم(عائم));
        فراغ *               داخلا_fp32_buffer1 = fp32_allocator1.get();
        acl_tensor_ptr       داخلا_fp32_tensor1 = ggml_cann_create_tensor(داخلا_fp32_buffer1, ACL_FLOAT, حجم(عائم),
                                                                          dst->ne, داخلا_fp32_nb, GGML_MAX_DIMS);
        ggml_cann_pool_alloc fp32_allocator2(ctx.pool(), ggml_nelements(dst) * حجم(عائم));
        فراغ *               داخلا_fp32_buffer2 = fp32_allocator2.get();
        acl_tensor_ptr       داخلا_fp32_tensor2 = ggml_cann_create_tensor(داخلا_fp32_buffer2, ACL_FLOAT, حجم(عائم),
                                                                          dst->ne, داخلا_fp32_nb, GGML_MAX_DIMS);

        ggml_cann_pool_alloc fp32_allocator(ctx.pool(), ggml_nelements(dst) * حجم(عائم));
        output_fp32_بفر                = fp32_allocator.get();
        acl_tensor_ptr output_fp32_tensor = ggml_cann_create_tensor(output_fp32_بفر, ACL_FLOAT, حجم(عائم),
                                                                    dst->ne, داخلا_fp32_nb, GGML_MAX_DIMS);
        aclnn_mul(ctx, acl_src.get(), acl_cos_reshape_tensor.get(), داخلا_fp32_tensor1.get());
        aclnn_mul(ctx, acl_داخلا_roll_mul_scale_tensor.get(), acl_sin_reshape_tensor.get(), داخلا_fp32_tensor2.get());
        aclnn_جوڙيو(ctx, داخلا_fp32_tensor1.get(), داخلا_fp32_tensor2.get(), output_fp32_tensor.get());
        aclnn_cast(ctx, output_fp32_tensor.get(), acl_dst.get(), ACL_FLOAT16);
    }
    أرجع;
#جيڪڏهن_پورو
    int64_t acl_گهڻائي = is_neox ? 0 : 1;

    // Pre-define head and tail dimensions for reuse
    int64_t سر_ne[GGML_MAX_DIMS] = { rope_dims, ne01, ne02, ne03 };
    int64_t پڇ_ne[GGML_MAX_DIMS] = { پڇ_dims, ne01, ne02, ne03 };

    // Step 1: Prepare trans tensors for F16 type conversion to F32 if needed
    منطقي                 src_dst_need_trans = خطأ;
    ggml_cann_pool_alloc src_trans_allocator(ctx.pool());
    ggml_cann_pool_alloc dst_trans_allocator(ctx.pool());
    acl_tensor_ptr       acl_src_trans_tensor;
    acl_tensor_ptr       acl_dst_trans_tensor;
    فراغ *               src_trans_بفر = خالي_اشارو;
    فراغ *               dst_trans_بفر = خالي_اشارو;
    size_t               src_dst_trans_nb[GGML_MAX_DIMS];
    إذا (src0->type == GGML_TYPE_F16) {
        src_dst_need_trans = سچ;
        src_trans_بفر   = src_trans_allocator.alloc(ggml_nelements(src0) * حجم(عائم));
        dst_trans_بفر   = dst_trans_allocator.alloc(ggml_nelements(dst) * حجم(عائم));

        src_dst_trans_nb[0] = حجم(عائم);
        لكل (صحيح i = 1; i < GGML_MAX_DIMS; i++) {
            src_dst_trans_nb[i] = src_dst_trans_nb[i - 1] * src0->ne[i - 1];
        }
        acl_src_trans_tensor = ggml_cann_create_tensor(src_trans_بفر, ACL_FLOAT, حجم(عائم), src0->ne,
                                                       src_dst_trans_nb, GGML_MAX_DIMS);
        acl_dst_trans_tensor = ggml_cann_create_tensor(dst_trans_بفر, ACL_FLOAT, حجم(عائم), dst->ne,
                                                       src_dst_trans_nb, GGML_MAX_DIMS);
        aclnn_cast(ctx, acl_src.get(), acl_src_trans_tensor.get(), ACL_FLOAT);
    }

    // Step 2: Prepare head tensors for tail splitting if needed
    acl_tensor_ptr acl_src_سر;
    acl_tensor_ptr acl_dst_سر;
    إذا (has_پڇ) {
        // Create head views for RotaryPositionEmbedding (only first rope_dims dimensions)
        // RotaryPositionEmbedding requires contiguous dst tensor, so we use a temporary buffer
        إذا (src_dst_need_trans) {
            // Use F32 trans tensor strides
            acl_src_سر = ggml_cann_create_tensor((حرف *) src_trans_بفر, ACL_FLOAT, حجم(عائم), سر_ne,
                                                   src_dst_trans_nb, GGML_MAX_DIMS);
        } وإلا {
            // Use original F32 tensor strides
            acl_src_سر = ggml_cann_create_tensor((حرف *) src0->معلومات, ACL_FLOAT, حجم(عائم), سر_ne, src0->nb,
                                                   GGML_MAX_DIMS);
        }

        int64_t              سر_elements = rope_dims * ne01 * ne02 * ne03;
        ggml_cann_pool_alloc dst_سر_contiguous_allocator(ctx.pool(), سر_elements * حجم(عائم));
        فراغ *               dst_سر_contiguous_بفر = dst_سر_contiguous_allocator.get();

        size_t سر_contiguous_nb[GGML_MAX_DIMS];
        سر_contiguous_nb[0] = حجم(عائم);
        لكل (صحيح i = 1; i < GGML_MAX_DIMS; i++) {
            سر_contiguous_nb[i] = سر_contiguous_nb[i - 1] * سر_ne[i - 1];
        }
        acl_dst_سر = ggml_cann_create_tensor(dst_سر_contiguous_بفر, ACL_FLOAT, حجم(عائم), سر_ne,
                                               سر_contiguous_nb, GGML_MAX_DIMS);
    }

    // Step 3: Execute RotaryPositionEmbedding
    إذا (has_پڇ) {
        // Rotate only the head portion (first rope_dims dimensions)
        GGML_CANN_CALL_ACLNN_OP(ctx, RotaryPositionEmbedding, acl_src_سر.get(), acl_cos_reshape_tensor.get(),
                                acl_sin_reshape_tensor.get(), acl_گهڻائي, acl_dst_سر.get());

        // Copy head result from contiguous buffer back to destination tensor
        إذا (src_dst_need_trans) {
            acl_tensor_ptr acl_dst_سر_هدف = ggml_cann_create_tensor(
                (حرف *) dst_trans_بفر, ACL_FLOAT, حجم(عائم), سر_ne, src_dst_trans_nb, GGML_MAX_DIMS);
            cann_copy(ctx, acl_dst_سر.get(), acl_dst_سر_هدف.get());
        } وإلا {
            acl_tensor_ptr acl_dst_سر_هدف =
                ggml_cann_create_tensor((حرف *) dst->معلومات, ACL_FLOAT, حجم(عائم), سر_ne, dst->nb, GGML_MAX_DIMS);
            cann_copy(ctx, acl_dst_سر.get(), acl_dst_سر_هدف.get());
        }
    } وإلا إذا (src_dst_need_trans) {
        // Rotate full tensor (no tail), using trans tensors
        GGML_CANN_CALL_ACLNN_OP(ctx, RotaryPositionEmbedding, acl_src_trans_tensor.get(), acl_cos_reshape_tensor.get(),
                                acl_sin_reshape_tensor.get(), acl_گهڻائي, acl_dst_trans_tensor.get());
    } وإلا إذا (src0->معلومات == dst->معلومات && !ggml_is_contiguous(src0)) {
        // In-place on non-contiguous tensor: RotaryPositionEmbedding cannot safely
        // read and write the same non-contiguous buffer. Use contiguous temporaries.
        size_t contiguous_nb[GGML_MAX_DIMS];
        contiguous_nb[0] = حجم(عائم);
        لكل (صحيح i = 1; i < GGML_MAX_DIMS; i++) {
            contiguous_nb[i] = contiguous_nb[i - 1] * src0->ne[i - 1];
        }
        int64_t              ڪل_elements = ggml_nelements(src0);
        ggml_cann_pool_alloc inplace_src_alloc(ctx.pool(), ڪل_elements * حجم(عائم));
        ggml_cann_pool_alloc inplace_dst_alloc(ctx.pool(), ڪل_elements * حجم(عائم));

        acl_tensor_ptr acl_src_contig = ggml_cann_create_tensor(inplace_src_alloc.get(), ACL_FLOAT, حجم(عائم),
                                                                src0->ne, contiguous_nb, GGML_MAX_DIMS);
        acl_tensor_ptr acl_dst_contig = ggml_cann_create_tensor(inplace_dst_alloc.get(), ACL_FLOAT, حجم(عائم),
                                                                dst->ne, contiguous_nb, GGML_MAX_DIMS);

        cann_copy(ctx, acl_src.get(), acl_src_contig.get());
        GGML_CANN_CALL_ACLNN_OP(ctx, RotaryPositionEmbedding, acl_src_contig.get(), acl_cos_reshape_tensor.get(),
                                acl_sin_reshape_tensor.get(), acl_گهڻائي, acl_dst_contig.get());
        cann_copy(ctx, acl_dst_contig.get(), acl_dst.get());
    } وإلا {
        // Rotate full tensor (no tail), using original tensors
        GGML_CANN_CALL_ACLNN_OP(ctx, RotaryPositionEmbedding, acl_src.get(), acl_cos_reshape_tensor.get(),
                                acl_sin_reshape_tensor.get(), acl_گهڻائي, acl_dst.get());
    }

    // Step 4: Copy unrotated tail portion from source to destination
    إذا (has_پڇ) {
        size_t src_پڇ_وٿي;
        size_t dst_پڇ_وٿي;

        خودڪار copy_پڇ_device = [&](فراغ * src_ptr, فراغ * dst_ptr, acl_معلومات_Type dtype, size_t elem_سائيز,
                                    size_t * nb_src_arr, size_t * nb_dst_arr) {
            acl_tensor_ptr acl_src_پڇ =
                ggml_cann_create_tensor(src_ptr, dtype, elem_سائيز, پڇ_ne, nb_src_arr, GGML_MAX_DIMS);
            acl_tensor_ptr acl_dst_پڇ =
                ggml_cann_create_tensor(dst_ptr, dtype, elem_سائيز, پڇ_ne, nb_dst_arr, GGML_MAX_DIMS);
            cann_copy(ctx, acl_src_پڇ.get(), acl_dst_پڇ.get());
        };

        إذا (src_dst_need_trans) {
            // Use F32 trans tensor strides and offsets
            src_پڇ_وٿي = rope_dims * src_dst_trans_nb[0];
            dst_پڇ_وٿي = rope_dims * src_dst_trans_nb[0];
            copy_پڇ_device((حرف *) src_trans_بفر + src_پڇ_وٿي, (حرف *) dst_trans_بفر + dst_پڇ_وٿي,
                             ACL_FLOAT, حجم(عائم), src_dst_trans_nb, src_dst_trans_nb);
        } وإلا {
            // Use original tensor strides and offsets
            src_پڇ_وٿي = rope_dims * nb00;
            dst_پڇ_وٿي = rope_dims * nb0;
            copy_پڇ_device((حرف *) src0->معلومات + src_پڇ_وٿي, (حرف *) dst->معلومات + dst_پڇ_وٿي,
                             ggml_cann_type_mapping(dst->type), ggml_element_سائيز(dst), src0->nb, dst->nb);
        }
    }

    // Step 5: Cast back to F16 if needed
    إذا (src_dst_need_trans) {
        aclnn_cast(ctx, acl_dst_trans_tensor.get(), acl_dst.get(), ACL_FLOAT16);
    }
}

فراغ ggml_cann_rope_cache_preload(ggml_backend_cann_context & ctx, ggml_tensor * dst) {
    ggml_tensor * src0 = dst->src[0];

    عائم     freq_بنياد, freq_scale, ext_ضرب_ڪندڙ, attn_ضرب_ڪندڙ, beta_fast, beta_slow;
    صحيح       sections[4];
    ثابت صحيح n_dims     = ((int32_t *) dst->op_params)[1];
    ثابت صحيح mode       = ((int32_t *) dst->op_params)[2];
    ثابت صحيح n_ctx_orig = ((int32_t *) dst->op_params)[4];

    GGML_TENSOR_UNARY_OP_LOCALS

    يادگيري_نقل(&freq_بنياد, (int32_t *) dst->op_params + 5, حجم(عائم));
    يادگيري_نقل(&freq_scale, (int32_t *) dst->op_params + 6, حجم(عائم));
    يادگيري_نقل(&ext_ضرب_ڪندڙ, (int32_t *) dst->op_params + 7, حجم(عائم));
    يادگيري_نقل(&attn_ضرب_ڪندڙ, (int32_t *) dst->op_params + 8, حجم(عائم));
    يادگيري_نقل(&beta_fast, (int32_t *) dst->op_params + 9, حجم(عائم));
    يادگيري_نقل(&beta_slow, (int32_t *) dst->op_params + 10, حجم(عائم));
    يادگيري_نقل(&sections, (int32_t *) dst->op_params + 11, حجم(صحيح) * 4);

    ثابت عائم theta_scale = powf(freq_بنياد, -2.0f / n_dims);

    عائم corr_dims[2];
    ggml_rope_yarn_corr_dims(n_dims, n_ctx_orig, freq_بنياد, beta_fast, beta_slow, corr_dims);

    منطقي       is_neox    = mode & GGML_ROPE_TYPE_NEOX;
    ثابت منطقي is_imrope  = mode == GGML_ROPE_TYPE_IMROPE;
    ثابت منطقي mrope_استعمال_ٿيل = mode & GGML_ROPE_TYPE_MROPE;
    ثابت منطقي is_vision  = mode == GGML_ROPE_TYPE_VISION;

    إذا (is_imrope || mrope_استعمال_ٿيل) {
        is_neox = سچ;
    }

    int64_t rope_dims = n_dims;
    إذا (is_vision) {
        rope_dims = src0->ne[0];
    }

    // Run the full cache init on the non-captured stream.  This performs all
    // host-to-device memcpy, aclrtMalloc/Free, and on-device computations
    // so that the memory pool is warmed up and cache metadata is populated.
    aclnn_rope_cache_init(ctx, dst, corr_dims, ext_ضرب_ڪندڙ, theta_scale, freq_scale, attn_ضرب_ڪندڙ, is_neox, sections,
                          mrope_استعمال_ٿيل, is_imrope, is_vision, rope_dims);

    // Reset `cached` so that during graph capture the on-device computations
    // (sin/cos, position multiply, repeat, etc.) still execute and get recorded
    // into the captured graph.  The cache metadata (theta_scale_length,
    // theta_scale, sections, position_length, etc.) remains set, which causes
    // all host-to-device copy and malloc/free branches to be skipped.
    ctx.rope_cache.cached = خطأ;
}

فراغ ggml_cann_argmax(ggml_backend_cann_context & ctx, ggml_tensor * dst) {
    ggml_tensor * src0 = dst->src[0];

    acl_tensor_ptr acl_src = ggml_cann_create_tensor(src0);
    acl_tensor_ptr acl_dst = ggml_cann_create_tensor(dst, dst->ne, dst->nb, 3);

    GGML_CANN_CALL_ACLNN_OP(ctx, ArgMax, acl_src.get(), 3, خطأ, acl_dst.get());
}

فراغ ggml_cann_conv_transpose_1d(ggml_backend_cann_context & ctx, ggml_tensor * dst) {
    ggml_tensor * src0 = dst->src[0];
    ggml_tensor * src1 = dst->src[1];

    // stride
    int64_t s0 = ((ثابت int32_t *) (dst->op_params))[0];

    acl_tensor_ptr acl_داخلا  = ggml_cann_create_tensor(src1, src1->ne, src1->nb, 3, ACL_FORMAT_NCL);
    acl_tensor_ptr acl_وزن = ggml_cann_create_tensor(src0, src0->ne, src0->nb, 3, ACL_FORMAT_NCL);
    acl_tensor_ptr acl_dst    = ggml_cann_create_tensor(dst, dst->ne, dst->nb, 3, ACL_FORMAT_NCL);

    // get base information of input and kernel
    int64_t داخلا_len   = *(src1->ne);
    int64_t dst_len     = *(dst->ne);
    int64_t kernel_سائيز = *(src0->ne);

    // set the max kernel size for each conv
    int64_t max_kernel_سائيز = 255;

    // compute the partition of kernel
    int64_t part_انگ = 1;
    part_انگ         = (kernel_سائيز + max_kernel_سائيز - 1) / max_kernel_سائيز;

    int64_t strideVal[1];
    strideVal[0]                    = s0;
    acl_int_array_ptr stride        = ggml_cann_create_int_array(strideVal, 1);
    int64_t           paddingVal[]  = { 0 };
    acl_int_array_ptr padding       = ggml_cann_create_int_array(paddingVal, 1);
    int64_t           dilationVal[] = { 1 };
    acl_int_array_ptr dilation      = ggml_cann_create_int_array(dilationVal, 1);
    منطقي              transposed    = سچ;
    int64_t           groups        = 1;
    int8_t            cubeMathType  = 0;

#جيڪڏهن_وصف_ٿيل ASCEND_310P
    cubeMathType = 1;
#جيڪڏهن_پورو

    خودڪار وزن_type = ggml_cann_type_mapping(src0->type);
    خودڪار dst_type    = ggml_cann_type_mapping(dst->type);

    // slice the kernel to make each conv available
    int64_t ٽڪرو_dim   = -1;
    int64_t ٽڪرو_شروعات = 0;
    int64_t ٽڪرو_پڇاڙي   = max_kernel_سائيز;
    int64_t ٽڪرو_قدم  = 1;
    int64_t interval    = max_kernel_سائيز;

    int64_t کاٻو_pad_len  = dilationVal[0] * (max_kernel_سائيز - 1) + 1 - 2 * paddingVal[0];
    int64_t ساڄو_pad_len = 0;

    acl_scalar_ptr alpha      = خالي_اشارو;
    عائم          alpha_قدر = 1.0;
    alpha                     = ggml_cann_create_scalar(&alpha_قدر, acl_معلومات_Type::ACL_FLOAT);

    // set zero to destination
    GGML_CANN_CALL_ACLNN_OP(ctx, InplaceZero, acl_dst.get());

    لكل (صحيح k = 0; k < part_انگ; k++) {
        // create part kernel tensor and slice from big kernel
        ٽڪرو_شروعات = max_kernel_سائيز * k;
        إذا (k == part_انگ - 1) {
            ٽڪرو_پڇاڙي = kernel_سائيز;
            interval  = kernel_سائيز - max_kernel_سائيز * k;
        } وإلا {
            ٽڪرو_پڇاڙي = max_kernel_سائيز * (k + 1);
        }

        int64_t part_ne[4];
        لكل (صحيح i = 0; i < 4; i++) {
            part_ne[i] = *(src0->ne + i);
        }
        part_ne[0] = interval;

        size_t part_nb[4];
        part_nb[0] = حجم(وزن_type);
        لكل (صحيح i = 1; i < 4; i++) {
            part_nb[i] = part_nb[i - 1] * part_ne[i - 1];
        }

        ggml_cann_pool_alloc part_kernel_allocator;
        part_kernel_allocator.alloc(ctx.pool(), part_nb[3]);
        فراغ * part_kernel_buf = part_kernel_allocator.get();

        acl_tensor_ptr part_kernel = ggml_cann_create_tensor(part_kernel_buf, وزن_type, ggml_element_سائيز(src0),
                                                             part_ne, part_nb, 3, ACL_FORMAT_NCL);

        GGML_CANN_CALL_ACLNN_OP(ctx, ٽڪرو, acl_وزن.get(), ٽڪرو_dim, ٽڪرو_شروعات, ٽڪرو_پڇاڙي, ٽڪرو_قدم,
                                part_kernel.get());

        // create the part conv result tensor
        int64_t part_dst_ne[4];
        لكل (صحيح i = 0; i < 4; i++) {
            part_dst_ne[i] = *(dst->ne + i);
        }
        part_dst_ne[0] = (داخلا_len - 1) * strideVal[0] - 2 * paddingVal[0] + dilationVal[0] * (part_ne[0] - 1) + 1;

        size_t part_dst_nb[4];
        part_dst_nb[0] = حجم(وزن_type);
        لكل (صحيح i = 1; i < 4; i++) {
            part_dst_nb[i] = part_dst_nb[i - 1] * part_dst_ne[i - 1];
        }
        ggml_cann_pool_alloc part_dst_allocator;
        part_dst_allocator.alloc(ctx.pool(), part_dst_nb[3]);
        فراغ * part_dst_buf = part_dst_allocator.get();

        acl_tensor_ptr acl_part_dst = ggml_cann_create_tensor(part_dst_buf, dst_type, ggml_element_سائيز(dst),
                                                              part_dst_ne, part_dst_nb, 3, ACL_FORMAT_NCL);
        GGML_CANN_CALL_ACLNN_OP(ctx, InplaceZero, acl_part_dst.get());

        // compute part conv transpose 1d
        GGML_CANN_CALL_ACLNN_OP(ctx, Convolution, acl_داخلا.get(), part_kernel.get(), خالي_اشارو, stride.get(),
                                padding.get(), dilation.get(), transposed, padding.get(), groups, acl_part_dst.get(),
                                cubeMathType);

        // compute the position of part result in final result
        int64_t global_شروعات = ٽڪرو_شروعات;
        int64_t global_پڇاڙي   = std::min((داخلا_len - 1) * strideVal[0] + ٽڪرو_پڇاڙي, dst_len);

        کاٻو_pad_len  = global_شروعات;
        ساڄو_pad_len = dst_len - global_پڇاڙي;

        std::vector<int64_t> pad_معلومات_Val = { کاٻو_pad_len, ساڄو_pad_len };
        acl_int_array_ptr    pad_معلومات    = ggml_cann_create_int_array(pad_معلومات_Val.معلومات(), 2);

        acl_scalar_ptr pad_قدر    = خالي_اشارو;
        عائم          pad_قدر_Val = 0.0;
        pad_قدر                   = ggml_cann_create_scalar(&pad_قدر_Val, acl_معلومات_Type::ACL_FLOAT);

        int64_t conv_نتيجو_ne[4];
        لكل (صحيح i = 0; i < 4; i++) {
            conv_نتيجو_ne[i] = *(dst->ne + i);
        }

        size_t conv_نتيجو_nb[4];
        conv_نتيجو_nb[0] = حجم(وزن_type);
        لكل (صحيح i = 1; i < 4; i++) {
            conv_نتيجو_nb[i] = conv_نتيجو_nb[i - 1] * conv_نتيجو_ne[i - 1];
        }

        ggml_cann_pool_alloc conv_نتيجو_allocator;
        conv_نتيجو_allocator.alloc(ctx.pool(), conv_نتيجو_nb[3]);
        فراغ * conv_نتيجو_buf = conv_نتيجو_allocator.get();

        acl_tensor_ptr conv_نتيجو = ggml_cann_create_tensor(conv_نتيجو_buf, dst_type, ggml_element_سائيز(dst),
                                                             conv_نتيجو_ne, conv_نتيجو_nb, 3, ACL_FORMAT_NCL);

        GGML_CANN_CALL_ACLNN_OP(ctx, InplaceZero, conv_نتيجو.get());
        GGML_CANN_CALL_ACLNN_OP(ctx, ConstantPadNd, acl_part_dst.get(), pad_معلومات.get(), pad_قدر.get(),
                                conv_نتيجو.get());
        GGML_CANN_CALL_ACLNN_OP(ctx, Inplace_جوڙيو, acl_dst.get(), conv_نتيجو.get(), alpha.get());
    }
}

فراغ ggml_cann_elu(ggml_backend_cann_context & ctx, ggml_tensor * dst) {
    ggml_tensor * src0 = dst->src[0];

    acl_tensor_ptr acl_داخلا = ggml_cann_create_tensor(src0);
    acl_tensor_ptr acl_dst   = ggml_cann_create_tensor(dst);

    عائم          alpha_قدر = 1.0f;
    acl_scalar_ptr alpha      = خالي_اشارو;
    alpha                     = ggml_cann_create_scalar(&alpha_قدر, acl_معلومات_Type::ACL_FLOAT);

    GGML_CANN_CALL_ACLNN_OP(ctx, Elu, acl_داخلا.get(), alpha.get(), alpha.get(), alpha.get(), acl_dst.get());
}

فراغ ggml_cann_سراسري(ggml_backend_cann_context & ctx, ggml_tensor * dst) {
    ggml_tensor * src0 = dst->src[0];

    acl_tensor_ptr acl_src = ggml_cann_create_tensor(src0);
    acl_tensor_ptr acl_dst = ggml_cann_create_tensor(dst);

    int64_t           reduce_Dim_قدر[] = { 3 };
    acl_int_array_ptr reduceDim        = ggml_cann_create_int_array(reduce_Dim_قدر, 1);
    منطقي              keepDim          = سچ;

    GGML_CANN_CALL_ACLNN_OP(ctx, سراسري, acl_src.get(), reduceDim.get(), keepDim, ACL_FLOAT, acl_dst.get());
}

فراغ ggml_cann_pad_reflect_1d(ggml_backend_cann_context & ctx, ggml_tensor * dst) {
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

    acl_tensor_ptr acl_src = ggml_cann_create_tensor(src0->معلومات, ggml_cann_type_mapping(src0->type),
                                                     ggml_element_سائيز(src0), src_ne_3d, src0->nb, 3);

    acl_tensor_ptr acl_dst = ggml_cann_create_tensor(dst->معلومات, ggml_cann_type_mapping(dst->type),
                                                     ggml_element_سائيز(dst), dst_ne_3d, dst->nb, 3);

    GGML_CANN_CALL_ACLNN_OP(ctx, ReflectionPad1d, acl_src.get(), paddings.get(), acl_dst.get());
}

فراغ ggml_cann_ڳڻپ_equal(ggml_backend_cann_context & ctx, ggml_tensor * dst) {
    ggml_tensor * src0 = dst->src[0];
    ggml_tensor * src1 = dst->src[1];

    // Write element-wise equality (0 or 1) into a temporary buffer to avoid
    // modifying src0 in-place.  Use the same type as src0 so ReduceSum can
    // consume it directly without a type cast.
    ggml_cann_pool_alloc eq_alloc(ctx.pool(), ggml_nelements(src0) * ggml_element_سائيز(src0));
    size_t eq_nb[GGML_MAX_DIMS];
    eq_nb[0] = ggml_element_سائيز(src0);
    لكل (صحيح i = 1; i < GGML_MAX_DIMS; i++) {
        eq_nb[i] = eq_nb[i - 1] * src0->ne[i - 1];
    }
    acl_tensor_ptr acl_eq = ggml_cann_create_tensor(
        eq_alloc.get(), ggml_cann_type_mapping(src0->type), ggml_element_سائيز(src0),
        src0->ne, eq_nb, GGML_MAX_DIMS);

    acl_tensor_ptr acl_self  = ggml_cann_create_tensor(src0);
    acl_tensor_ptr acl_other = ggml_cann_create_tensor(src1);
    GGML_CANN_CALL_ACLNN_OP(ctx, EqTensor, acl_self.get(), acl_other.get(), acl_eq.get());

    // Sum the 0/1 values into dst.
    acl_tensor_ptr    acl_dst    = ggml_cann_create_tensor(dst);
    int64_t           dims[4]    = { 0, 1, 2, 3 };
    acl_int_array_ptr dims_arr   = ggml_cann_create_int_array(dims, 4);
    GGML_CANN_CALL_ACLNN_OP(ctx, Reduce_جوڙ, acl_eq.get(), dims_arr.get(), سچ,
                            ggml_cann_type_mapping(dst->type), acl_dst.get());
}

فراغ ggml_cann_قدم(ggml_backend_cann_context & ctx, ggml_tensor * dst) {
    ggml_tensor * src0 = dst->src[0];

    acl_tensor_ptr acl_src = ggml_cann_create_tensor(src0);
    acl_tensor_ptr acl_dst = ggml_cann_create_tensor(dst);

    عائم          alpha_قدر = 0.0f;
    acl_scalar_ptr alpha      = خالي_اشارو;
    alpha                     = ggml_cann_create_scalar(&alpha_قدر, acl_معلومات_Type::ACL_FLOAT);

    GGML_CANN_CALL_ACLNN_OP(ctx, GtScalar, acl_src.get(), alpha.get(), acl_dst.get());
}

فراغ ggml_cann_softplus(ggml_backend_cann_context & ctx, ggml_tensor * dst) {
    ggml_tensor * src0 = dst->src[0];

    acl_tensor_ptr acl_src = ggml_cann_create_tensor(src0);
    acl_tensor_ptr acl_dst = ggml_cann_create_tensor(dst);

    عائم          beta_val      = 1.0f;
    عائم          حد_val = 20.0f;
    acl_scalar_ptr beta          = ggml_cann_create_scalar(&beta_val,      ACL_FLOAT);
    acl_scalar_ptr حد     = ggml_cann_create_scalar(&حد_val, ACL_FLOAT);

    GGML_CANN_CALL_ACLNN_OP(ctx, Softplus, acl_src.get(), beta.get(), حد.get(), acl_dst.get());
}

فراغ ggml_cann_geglu_تيز(ggml_backend_cann_context & ctx, ggml_tensor * dst) {
    خودڪار gelu_تيز_fn = [](ggml_backend_cann_context & ctx, aclTensor * acl_src, aclTensor * acl_dst) {
        GGML_CANN_CALL_ACLNN_OP(ctx, GeluV2, acl_src, 0, acl_dst);
    };
    ggml_cann_op_unary_gated(gelu_تيز_fn, ctx, dst);
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
ساكن فراغ ggml_cann_mul_mat_id_fp(ggml_backend_cann_context & ctx, ggml_tensor * dst) {
    //dst   [M, K, N, 1]
    ggml_tensor * src0 = dst->src[0];  //src0	[D, M, A, 1]  -> [D, M, K, 1]
    ggml_tensor * src1 = dst->src[1];  //src1	[D, B, N, 1], B = K or B = 1 -> [D, 1, K, 1]
    ggml_tensor * ids  = dst->src[2];  //ids	[K, N]

    GGML_ASSERT(src0->ne[3] == 1);
    GGML_ASSERT(src1->ne[3] == 1);
    GGML_ASSERT(dst->ne[3] == 1);

    int64_t batch = src1->ne[2];
    GGML_ASSERT(batch == ids->ne[1]);

    ggml_cann_pool_alloc export_allocator(ctx.pool(), src0->ne[0] * src0->ne[1] * ids->ne[0] * ggml_element_سائيز(src0));
    فراغ *               export_ptr = export_allocator.get();
    لكل (int64_t i = 0; i < batch; i++) {
        acl_tensor_ptr select_اشاريو  = ggml_cann_create_tensor(ids, ids->ne, ids->nb, 1, ACL_FORMAT_ND, i * ids->nb[1]);
        acl_tensor_ptr export_وزن = ggml_cann_create_tensor(src0, src0->ne, src0->nb, 3);

        int64_t select_export_ne[] = { src0->ne[0], src0->ne[1], ids->ne[0] };
        size_t  select_export_nb[3];
        select_export_nb[0] = src0->nb[0];
        لكل (صحيح k = 1; k < 3; k++) {
            select_export_nb[k] = select_export_nb[k - 1] * select_export_ne[k - 1];
        }

        acl_tensor_ptr select_export =
            ggml_cann_create_tensor(export_ptr, ggml_cann_type_mapping(src0->type), ggml_element_سائيز(src0),
                                    select_export_ne, select_export_nb, 3);
        GGML_CANN_CALL_ACLNN_OP(ctx, اشاريو_Select, export_وزن.get(), 0, select_اشاريو.get(), select_export.get());

        int64_t        select_transpose_ne[] = { select_export_ne[1], select_export_ne[0], select_export_ne[2] };
        size_t         select_transpose_nb[] = { select_export_nb[1], select_export_nb[0], select_export_nb[2] };
        acl_tensor_ptr select_export_transpose =
            ggml_cann_create_tensor(export_ptr, ggml_cann_type_mapping(src0->type), ggml_element_سائيز(src0),
                                    select_transpose_ne, select_transpose_nb, 3);

        int64_t        سرگرم_tensor_ne[] = { src1->ne[0], 1, src1->ne[1] };
        size_t         سرگرم_tensor_nb[] = { src1->nb[0], src1->nb[1], src1->nb[1] };
        acl_tensor_ptr سرگرم_tensor =
            ggml_cann_create_tensor(src1, سرگرم_tensor_ne, سرگرم_tensor_nb, 3, ACL_FORMAT_ND, i * src1->nb[2]);

        int64_t        dst_ne[] = { dst->ne[0], 1, dst->ne[1] };
        size_t         dst_nb[] = { dst->nb[0], dst->nb[1], dst->nb[1] };
        acl_tensor_ptr acl_dst  = ggml_cann_create_tensor(dst, dst_ne, dst_nb, 3, ACL_FORMAT_ND, i * dst->nb[2]);

        GGML_CANN_CALL_ACLNN_OP(ctx, BatchMatMul, سرگرم_tensor.get(), select_export_transpose.get(), acl_dst.get(), 2);
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
ساكن فراغ ggml_cann_mul_mat_id_quant(ggml_backend_cann_context & ctx, ggml_tensor * dst) {
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

    ثابت int64_t        n_batches        = ids->ne[1];
    ثابت int64_t        n_select_experts = ids->ne[0];
    ثابت تعداد ggml_type type             = src0->type;

    ثابت int32_t گروهه_سائيز = QK8_0;  // Both Q4_0 and Q8_0 use group size of 32
    GGML_ASSERT(گروهه_سائيز == QK4_0);

    // Calculate element size for quantized weights
    ثابت عائم وزن_elem_سائيز =
        (type == GGML_TYPE_Q4_0) ? 0.5f :
        (type == GGML_TYPE_Q8_0) ? 1.0f :
                                   (GGML_ABORT("MUL_MAT_ID only supports Q4_0 and Q8_0"), 0.0f);

    // Calculate scale offset in memory
    ثابت size_t وزن_سائيز     = src0->ne[0] * src0->ne[1] * src0->ne[2] * وزن_elem_سائيز;
    ثابت size_t scale_elem_سائيز = حجم(uint16_t);
    حرف *       scale_معلومات      = (حرف *) src0->معلومات + وزن_سائيز;

    // Allocate buffers for selected expert weights and scales
    ثابت size_t         selected_وزن_سائيز = src0->ne[0] * src0->ne[1] * n_select_experts * وزن_elem_سائيز;
    ggml_cann_pool_alloc selected_وزن_alloc(ctx.pool(), selected_وزن_سائيز);
    فراغ *               selected_وزن_بفر = selected_وزن_alloc.get();

    ثابت size_t selected_scale_سائيز = (src0->ne[0] / گروهه_سائيز) * src0->ne[1] * n_select_experts * scale_elem_سائيز;
    ggml_cann_pool_alloc selected_scale_alloc(ctx.pool(), selected_scale_سائيز);
    فراغ *               selected_scale_بفر = selected_scale_alloc.get();

    // Helper lambda to allocate and cast tensor to F16 if needed
    constexpr size_t f16_elem_سائيز      = حجم(uint16_t);
    خودڪار             prepare_f16_بفر = [&](ggml_tensor * tensor, ggml_cann_pool_alloc & allocator,
                                  منطقي need_cast = خطأ) -> فراغ * {
        إذا (tensor->type == GGML_TYPE_F16) {
            أرجع tensor->معلومات;
        }

        size_t ڪل_سائيز = f16_elem_سائيز;
        لكل (صحيح i = 0; i < GGML_MAX_DIMS; i++) {
            ڪل_سائيز *= tensor->ne[i];
        }
        فراغ * بفر = allocator.alloc(ڪل_سائيز);

        إذا (need_cast == خطأ) {
            أرجع بفر;
        }

        int64_t ne[GGML_MAX_DIMS];
        size_t  nb[GGML_MAX_DIMS] = { f16_elem_سائيز };
        لكل (صحيح i = 0; i < GGML_MAX_DIMS; i++) {
            ne[i] = tensor->ne[i];
            إذا (i > 0) {
                nb[i] = nb[i - 1] * ne[i - 1];
            }
        }

        acl_tensor_ptr src_tensor = ggml_cann_create_tensor(tensor);
        acl_tensor_ptr f16_tensor = ggml_cann_create_tensor(بفر, ACL_FLOAT16, f16_elem_سائيز, ne, nb, GGML_MAX_DIMS);
        aclnn_cast(ctx, src_tensor.get(), f16_tensor.get(), ACL_FLOAT16);

        أرجع بفر;
    };

    // Prepare input and output buffers
    ggml_cann_pool_alloc داخلا_alloc(ctx.pool());
    فراغ *               داخلا_بفر = prepare_f16_بفر(src1, داخلا_alloc, سچ);

    ggml_cann_pool_alloc output_alloc(ctx.pool());
    فراغ *               output_بفر = prepare_f16_بفر(dst, output_alloc, خطأ);

    // Process each batch
    لكل (int64_t batch_idx = 0; batch_idx < n_batches; batch_idx++) {
        // Create index tensor for current batch
        ثابت size_t   اشاريو_وٿي  = batch_idx * ids->nb[1];
        acl_tensor_ptr batch_indices = ggml_cann_create_tensor(ids, ids->ne, ids->nb, 1, ACL_FORMAT_ND, اشاريو_وٿي);

        // Select quantized weights using expert indices
        // Q4_0 stores 2 values per byte, Q8_0 stores 1 value per byte
        ثابت int64_t وزن_d         = (type == GGML_TYPE_Q4_0) ? src0->ne[0] / 2 : src0->ne[0];
        ثابت int64_t وزن_m         = src0->ne[1];
        ثابت int64_t وزن_n_experts = src0->ne[2];

        int64_t وزن_ne[3] = { وزن_d, وزن_m, وزن_n_experts };
        size_t  وزن_nb[3] = { حجم(int8_t), وزن_d * حجم(int8_t), وزن_d * وزن_m * حجم(int8_t) };

        acl_tensor_ptr all_weights =
            ggml_cann_create_tensor(src0->معلومات, ACL_INT8, حجم(int8_t), وزن_ne, وزن_nb, 3);

        int64_t selected_وزن_ne[3] = { وزن_d, وزن_m, n_select_experts };
        size_t  selected_وزن_nb[3] = { حجم(int8_t), وزن_d * حجم(int8_t),
                                          وزن_d * وزن_m * حجم(int8_t) };

        acl_tensor_ptr selected_weights = ggml_cann_create_tensor(selected_وزن_بفر, ACL_INT8, حجم(int8_t),
                                                                  selected_وزن_ne, selected_وزن_nb, 3);

        GGML_CANN_CALL_ACLNN_OP(ctx, اشاريو_Select, all_weights.get(), 0, batch_indices.get(), selected_weights.get());

        // Select scales using the same expert indices
        ثابت int64_t scale_d     = src0->ne[0] / گروهه_سائيز;
        int64_t       scale_ne[3] = { scale_d, وزن_m, وزن_n_experts };
        size_t scale_nb[3] = { scale_elem_سائيز, scale_d * scale_elem_سائيز, scale_d * وزن_m * scale_elem_سائيز };

        acl_tensor_ptr all_scales =
            ggml_cann_create_tensor(scale_معلومات, ACL_FLOAT16, scale_elem_سائيز, scale_ne, scale_nb, 3);

        int64_t selected_scale_ne[3] = { scale_d, وزن_m, n_select_experts };
        size_t  selected_scale_nb[3] = { scale_elem_سائيز, scale_d * scale_elem_سائيز,
                                         scale_d * وزن_m * scale_elem_سائيز };

        acl_tensor_ptr selected_scales = ggml_cann_create_tensor(selected_scale_بفر, ACL_FLOAT16, scale_elem_سائيز,
                                                                 selected_scale_ne, selected_scale_nb, 3);

        GGML_CANN_CALL_ACLNN_OP(ctx, اشاريو_Select, all_scales.get(), 0, batch_indices.get(), selected_scales.get());

        // Process each expert for current batch
        // IndexSelect output layout: [D, M, K] in contiguous format
        // WeightQuantBatchMatmulV2 expects: [M, D] with row-major stride
        لكل (int64_t expert_idx = 0; expert_idx < n_select_experts; expert_idx++) {
            // Determine input offset: broadcast if src1->ne[1]==1, otherwise use per-expert input
            ثابت size_t داخلا_وٿي =
                (batch_idx * src1->ne[1] + (src1->ne[1] == 1 ? 0 : expert_idx)) * src1->ne[0] * f16_elem_سائيز;
            ثابت size_t output_وٿي = (batch_idx * dst->ne[1] + expert_idx) * dst->ne[0] * f16_elem_سائيز;

            // Create weight view for current expert: [D, M, K] -> [M, D]
            int64_t      وزن_view_ne[2]  = { وزن_m, src0->ne[0] };
            عائم        وزن_view_nb[2]  = { src0->ne[0] * وزن_elem_سائيز, وزن_elem_سائيز };
            ثابت size_t وزن_view_وٿي = expert_idx * selected_وزن_nb[2];

            acl_tensor_ptr وزن_view =
                ggml_cann_create_tensor(selected_وزن_بفر, ggml_cann_type_mapping(type), وزن_elem_سائيز,
                                        وزن_view_ne, وزن_view_nb, 2, ACL_FORMAT_ND, وزن_view_وٿي);

            // Create scale view for current expert: [D, M, K] -> [M, D]
            int64_t      scale_view_ne[2]  = { وزن_m, scale_d };
            size_t       scale_view_nb[2]  = { selected_scale_nb[1], selected_scale_nb[0] };
            ثابت size_t scale_view_وٿي = expert_idx * selected_scale_nb[2];

            acl_tensor_ptr scale_view =
                ggml_cann_create_tensor(selected_scale_بفر, ACL_FLOAT16, scale_elem_سائيز, scale_view_ne,
                                        scale_view_nb, 2, ACL_FORMAT_ND, scale_view_وٿي);

            // Create input activation tensor [D, 1]
            int64_t داخلا_ne[2] = { src1->ne[0], 1 };
            size_t  داخلا_nb[2] = { f16_elem_سائيز, src1->ne[0] * f16_elem_سائيز };

            acl_tensor_ptr داخلا_tensor = ggml_cann_create_tensor(داخلا_بفر, ACL_FLOAT16, f16_elem_سائيز, داخلا_ne,
                                                                  داخلا_nb, 2, ACL_FORMAT_ND, داخلا_وٿي);

            // Create output tensor [M, 1]
            int64_t output_ne[2] = { dst->ne[0], 1 };
            size_t  output_nb[2] = { f16_elem_سائيز, dst->ne[0] * f16_elem_سائيز };

            acl_tensor_ptr output_tensor = ggml_cann_create_tensor(output_بفر, ACL_FLOAT16, f16_elem_سائيز, output_ne,
                                                                   output_nb, 2, ACL_FORMAT_ND, output_وٿي);

            // Perform quantized matrix multiplication
            GGML_CANN_CALL_ACLNN_OP(ctx, وزن_Quant_Batch_Matmul_V2, داخلا_tensor.get(), وزن_view.get(),
                                    scale_view.get(), خالي_اشارو, خالي_اشارو, خالي_اشارو, خالي_اشارو, گروهه_سائيز,
                                    output_tensor.get());
        }
    }

    // Cast output back to original type if we used a temporary F16 buffer
    إذا (dst->type != GGML_TYPE_F16) {
        int64_t ne[GGML_MAX_DIMS];
        size_t  nb[GGML_MAX_DIMS] = { f16_elem_سائيز };
        لكل (صحيح i = 0; i < GGML_MAX_DIMS; i++) {
            ne[i] = dst->ne[i];
            إذا (i > 0) {
                nb[i] = nb[i - 1] * ne[i - 1];
            }
        }

        acl_tensor_ptr f16_output =
            ggml_cann_create_tensor(output_بفر, ACL_FLOAT16, f16_elem_سائيز, ne, nb, GGML_MAX_DIMS);
        acl_tensor_ptr dst_tensor = ggml_cann_create_tensor(dst);

        aclnn_cast(ctx, f16_output.get(), dst_tensor.get(), ggml_cann_type_mapping(dst->type));
    }
}

فراغ ggml_cann_mul_mat_id(ggml_backend_cann_context & ctx, ggml_tensor * dst) {
    ثابت تعداد ggml_type type = dst->src[0]->type;
    بدل (type) {
        حالة GGML_TYPE_F32:
        حالة GGML_TYPE_F16:
            ggml_cann_mul_mat_id_fp(ctx, dst);
            اكسر;
        حالة GGML_TYPE_Q4_0:
        حالة GGML_TYPE_Q8_0:
            ggml_cann_mul_mat_id_quant(ctx, dst);
            اكسر;
        افتراضي:
            GGML_ABORT("Unsupported type for mul_mat_id");
            اكسر;
    }
}

فراغ ggml_cann_flash_attn_ext(ggml_backend_cann_context & ctx, ggml_tensor * dst) {
    ggml_tensor * src0 = dst->src[0];  // q, fp32 | B, N, S, D (uncont) -> B, S, N, D (cont)
    ggml_tensor * src1 = dst->src[1];  // k, fp16 | B, N, S, D (uncont) -> B, S, N, D (cont)
    ggml_tensor * src2 = dst->src[2];  // v, fp16 | B, N, S, D (uncont) -> B, S, N, D (cont)
    ggml_tensor * src3 = dst->src[3];  // mask, fp16

    // B, N, S, D (uncont) -> B, S, N, D (cont)
    int64_t src0_bsnd_ne[GGML_MAX_DIMS];
    يادگيري_نقل(src0_bsnd_ne, src0->ne, GGML_MAX_DIMS * حجم(int64_t));
    size_t src0_bsnd_nb[GGML_MAX_DIMS];
    يادگيري_نقل(src0_bsnd_nb, src0->nb, GGML_MAX_DIMS * حجم(size_t));
    int64_t src1_bsnd_ne[GGML_MAX_DIMS];
    يادگيري_نقل(src1_bsnd_ne, src1->ne, GGML_MAX_DIMS * حجم(int64_t));
    size_t src1_bsnd_nb[GGML_MAX_DIMS];
    يادگيري_نقل(src1_bsnd_nb, src1->nb, GGML_MAX_DIMS * حجم(size_t));
    int64_t src2_bsnd_ne[GGML_MAX_DIMS];
    يادگيري_نقل(src2_bsnd_ne, src2->ne, GGML_MAX_DIMS * حجم(int64_t));
    size_t src2_bsnd_nb[GGML_MAX_DIMS];
    يادگيري_نقل(src2_bsnd_nb, src2->nb, GGML_MAX_DIMS * حجم(size_t));

    خودڪار transpose12 = [](int64_t * ne, size_t * nb) {
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

    عائم maxBias      = 0.0f;
    عائم scale_قدر   = 1.0f;
    عائم logitSoftcap = 0.0f;
    يادگيري_نقل(&scale_قدر, (عائم *) dst->op_params + 0, حجم(عائم));
    يادگيري_نقل(&maxBias, (عائم *) dst->op_params + 1, حجم(عائم));
    يادگيري_نقل(&logitSoftcap, (عائم *) dst->op_params + 2, حجم(عائم));

    إذا (logitSoftcap == 0.0f) {
        size_t fa_Elem_سائيز = حجم(uint16_t);
        خودڪار   fa_معلومات_Type = ACL_FLOAT16;  //ACL_BF16;

        acl_tensor_ptr acl_q_tensor = خالي_اشارو;
        acl_tensor_ptr acl_k_tensor = خالي_اشارو;
        acl_tensor_ptr acl_v_tensor = خالي_اشارو;

        // Step 1: cast the src0 (Query) to fp16 if needed
        ggml_cann_pool_alloc src0_f16_allocator(ctx.pool());
        فراغ *               src0_f16_بفر = خالي_اشارو;

        إذا (ggml_cann_type_mapping(src0->type) != fa_معلومات_Type) {
            acl_tensor_ptr acl_src0_f32_tensor =
                ggml_cann_create_tensor(src0, src0_bsnd_ne, src0_bsnd_nb, GGML_MAX_DIMS);
            src0_f16_بفر = src0_f16_allocator.alloc(ggml_nelements(src0) * fa_Elem_سائيز);

            int64_t * src0_f16_ne = src0_bsnd_ne;
            size_t    src0_f16_nb[GGML_MAX_DIMS];
            src0_f16_nb[0] = حجم(uint16_t);
            لكل (صحيح i = 1; i < GGML_MAX_DIMS; ++i) {
                src0_f16_nb[i] = src0_f16_nb[i - 1] * src0_f16_ne[i - 1];
            }

            acl_q_tensor = ggml_cann_create_tensor(src0_f16_بفر, fa_معلومات_Type, fa_Elem_سائيز, src0_f16_ne, src0_f16_nb,
                                                   GGML_MAX_DIMS);
            aclnn_cast(ctx, acl_src0_f32_tensor.get(), acl_q_tensor.get(), fa_معلومات_Type);
        } وإلا {
            acl_q_tensor = ggml_cann_create_tensor(src0, src0_bsnd_ne, src0_bsnd_nb, GGML_MAX_DIMS);
        }

        // Step 2: create the acl tensors for src1 (Key), src2 (Value),
        //         and the direct output from FusedInferAttention

        acl_k_tensor = ggml_cann_create_tensor(src1, src1_bsnd_ne, src1_bsnd_nb, GGML_MAX_DIMS);
        acl_v_tensor = ggml_cann_create_tensor(src2, src2_bsnd_ne, src2_bsnd_nb, GGML_MAX_DIMS);

        // Step 2.5: Pad Q, K, V along head dimension if D is not a multiple of 16
        //           (required by FusedInferAttentionScoreV2)
        ثابت int64_t D         = src0->ne[0];
        ثابت int64_t D_padded  = GGML_PAD(D, 16);
        ثابت منطقي needs_padding = (D != D_padded);

        ggml_cann_pool_alloc q_pad_allocator(ctx.pool());
        ggml_cann_pool_alloc k_pad_allocator(ctx.pool());
        ggml_cann_pool_alloc v_pad_allocator(ctx.pool());

        إذا (needs_padding) {
            int64_t paddings[] = { 0, D_padded - D, 0, 0, 0, 0, 0, 0 };

            خودڪار pad_fa_tensor = [&](acl_tensor_ptr & tensor, ثابت int64_t * bsnd_ne,
                                     ggml_cann_pool_alloc & allocator) {
                int64_t pad_ne[GGML_MAX_DIMS] = { D_padded, bsnd_ne[1], bsnd_ne[2], bsnd_ne[3] };
                size_t  pad_nb[GGML_MAX_DIMS];
                pad_nb[0] = fa_Elem_سائيز;
                لكل (صحيح i = 1; i < GGML_MAX_DIMS; ++i) {
                    pad_nb[i] = pad_nb[i - 1] * pad_ne[i - 1];
                }
                int64_t nelements = pad_ne[0] * pad_ne[1] * pad_ne[2] * pad_ne[3];
                فراغ *  بفر    = allocator.alloc(nelements * fa_Elem_سائيز);
                acl_tensor_ptr padded =
                    ggml_cann_create_tensor(بفر, fa_معلومات_Type, fa_Elem_سائيز, pad_ne, pad_nb, GGML_MAX_DIMS);
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
        إذا (src3 != خالي_اشارو) {
            // Construct the truncated pse tensor (common for prefill/decode)
            int64_t trunc_pse_ne[GGML_MAX_DIMS] = {
                src3->ne[0],  // D
                src0->ne[1],  // S (number of Q tokens)
                src3->ne[2],  // mask N
                src3->ne[3]   // B
            };
            size_t * trunc_pse_nb = src3->nb;

            acl_tensor_ptr acl_نقاب_f16_trunc_tensor = ggml_cann_create_tensor(
                src3->معلومات, ACL_FLOAT16, حجم(uint16_t), trunc_pse_ne, trunc_pse_nb, GGML_MAX_DIMS);

            int64_t bcast_pse_ne[GGML_MAX_DIMS];
            size_t  bcast_pse_nb[GGML_MAX_DIMS];
            bcast_pse_ne[0] = src3->ne[0];  // D
            bcast_pse_ne[1] = src0->ne[1];  // S
            bcast_pse_ne[2] = src0->ne[2];  // N (num_heads)
            bcast_pse_ne[3] = src3->ne[3];  // B
            إذا (maxBias == 0.0f) {
                // When maxBias == 0.0f, use nb = 0 reduce once repeat (Qwen2)
                // Construct the bcast tensor (simulate repeat on the head dimension using stride=0)
                bcast_pse_nb[0] = حجم(uint16_t);
                bcast_pse_nb[1] = bcast_pse_nb[0] * bcast_pse_ne[0];
                bcast_pse_nb[2] = 0;  // <---- the head dimension shares the same data
                bcast_pse_nb[3] = src3->nb[3];

                bcast_pse_tensor = ggml_cann_create_tensor(src3->معلومات, ACL_FLOAT16, حجم(uint16_t), bcast_pse_ne,
                                                           bcast_pse_nb, GGML_MAX_DIMS);

            } وإلا {
                bcast_pse_nb[0] = حجم(uint16_t);
                لكل (صحيح i = 1; i < GGML_MAX_DIMS; i++) {
                    bcast_pse_nb[i] = bcast_pse_nb[i - 1] * bcast_pse_ne[i - 1];
                }

                فراغ * bcast_pse_بفر =
                    bcast_pse_allocator.alloc(ggml_nelements(src3) * src0->ne[2] * حجم(uint16_t));

                bcast_pse_tensor = ggml_cann_create_tensor(bcast_pse_بفر, ACL_FLOAT16, حجم(uint16_t),
                                                           bcast_pse_ne, bcast_pse_nb, GGML_MAX_DIMS);

                int64_t repeats[] = { 1, src0->ne[2], 1, 1 };
                aclnn_repeat(ctx, acl_نقاب_f16_trunc_tensor.get(), bcast_pse_tensor.get(), repeats);

                // alibi
                // Compute the slope if needed. Derived from ggml_cann_softmax().
                ثابت int64_t        n_heads = src0->ne[2];
                ggml_cann_pool_alloc لاهي_allocator(ctx.pool(), n_heads * حجم(uint16_t));
                فراغ *               لاهي_بفر = لاهي_allocator.get();
                aclnn_get_لاهي(ctx, n_heads, لاهي_بفر, maxBias, GGML_TYPE_F16);

                int64_t لاهي_ne[] = { 1, 1, n_heads, 1 };
                size_t  لاهي_nb[GGML_MAX_DIMS];
                لاهي_nb[0] = حجم(uint16_t);
                لكل (صحيح i = 1; i < GGML_MAX_DIMS; i++) {
                    لاهي_nb[i] = لاهي_nb[i - 1] * لاهي_ne[0];
                }

                acl_tensor_ptr لاهي_tensor = ggml_cann_create_tensor(لاهي_بفر, ACL_FLOAT16, حجم(uint16_t),
                                                                      لاهي_ne, لاهي_nb, GGML_MAX_DIMS);
                GGML_CANN_CALL_ACLNN_OP(ctx, InplaceMul, bcast_pse_tensor.get(), لاهي_tensor.get());
            }
        }

        // Step 4: set the inputs for FusedInferAttention.
        acl_tensor_list_ptr acl_k_tensor_list = ggml_cann_create_tensor_list(acl_k_tensor);
        acl_tensor_list_ptr acl_v_tensor_list = ggml_cann_create_tensor_list(acl_v_tensor);

        int64_t انگ_Heads           = src0->ne[2];  // N
        int64_t انگ_ڪنجي_قدر_Heads   = src1->ne[2];
        // double  scaleValue = 1 / sqrt(src0->ne[0]); // 1/sqrt(d)
        int64_t pre_ٽوڪن          = 65535;
        int64_t ايندڙ_ٽوڪن         = 65535;
        حرف    layout[5]          = { 'B', 'S', 'N', 'D', 0 };
        int64_t sparse_گهڻائي         = 0;
        int64_t innerPrecise       = (src0->ne[1] == 1) ? 0 : 2;
        int64_t بلاڪ_سائيز          = 0;
        int64_t antiquant_گهڻائي      = 0;
        منطقي    softmaxLseFlag     = خطأ;
        int64_t ڪنجي_Antiquant_گهڻائي   = 0;
        int64_t قدر_Antiquant_گهڻائي = 0;

        GGML_ASSERT(dst->type == GGML_TYPE_F32 || dst->type == GGML_TYPE_F16);
        acl_tensor_ptr       fa_dst_tensor;
        ggml_cann_pool_alloc ٻاهريون_f16_allocator(ctx.pool());
        إذا (dst->type == GGML_TYPE_F32 || needs_padding) {
            int64_t * ٻاهريون_f16_ne = src0_bsnd_ne;
            size_t    ٻاهريون_f16_nb[GGML_MAX_DIMS];
            ٻاهريون_f16_nb[0] = fa_Elem_سائيز;
            لكل (صحيح i = 1; i < GGML_MAX_DIMS; ++i) {
                ٻاهريون_f16_nb[i] = ٻاهريون_f16_nb[i - 1] * ٻاهريون_f16_ne[i - 1];
            }
            int64_t ٻاهريون_nelements = ٻاهريون_f16_ne[0] * ٻاهريون_f16_ne[1] * ٻاهريون_f16_ne[2] * ٻاهريون_f16_ne[3];
            فراغ *  ٻاهريون_f16_بفر = ٻاهريون_f16_allocator.alloc(ٻاهريون_nelements * fa_Elem_سائيز);

            fa_dst_tensor =
                ggml_cann_create_tensor(ٻاهريون_f16_بفر, fa_معلومات_Type, fa_Elem_سائيز, ٻاهريون_f16_ne, ٻاهريون_f16_nb, GGML_MAX_DIMS);
        } وإلا {
            fa_dst_tensor = ggml_cann_create_tensor(dst);
        }

        GGML_CANN_CALL_ACLNN_OP(ctx, FusedInferAttentionScoreV2, acl_q_tensor.get(), acl_k_tensor_list.get(),
                                acl_v_tensor_list.get(),               // q, k, v
                                bcast_pse_tensor.get(), خالي_اشارو,       // pse, mask
                                خالي_اشارو, خالي_اشارو,                      // actSeqLen, actSeqLenkv
                                خالي_اشارو, خالي_اشارو,                      // deqScale1, quantScale1
                                خالي_اشارو, خالي_اشارو, خالي_اشارو,             // deqScale2, quantScale2, quantOffset2
                                خالي_اشارو, خالي_اشارو,                      // antiquantScale, antiquantOffset
                                خالي_اشارو,                               // blockTable
                                خالي_اشارو, خالي_اشارو,                      // qPadSize, kvPadSize
                                خالي_اشارو, خالي_اشارو,                      // kAntiquantScale, kAntiQuantOffset
                                خالي_اشارو, خالي_اشارو,                      // vAntiquantScale, vAntiQuantOffset
                                خالي_اشارو, خالي_اشارو, خالي_اشارو,             // kSharedPrefix, vSharedPrefix, actSharedLen
                                انگ_Heads, scale_قدر,                  // heads, scaleValue
                                pre_ٽوڪن, ايندڙ_ٽوڪن,                 // preTokens, nextTokens
                                layout,                                // inputLayout
                                انگ_ڪنجي_قدر_Heads,                      // numKVHeads
                                sparse_گهڻائي, innerPrecise,              // sparseMode, innerPrecise
                                بلاڪ_سائيز, antiquant_گهڻائي,              // blockSize, antiquantMode
                                softmaxLseFlag,                        // softmaxLseFlag
                                ڪنجي_Antiquant_گهڻائي, قدر_Antiquant_گهڻائي,  // keyAntiqMode, valueAntiqMode
                                fa_dst_tensor.get(),                   // attentionOut
                                خالي_اشارو                                // softmaxLse
        );

        // Step 6: post-processing — slice padded output and/or cast to f32
        إذا (needs_padding) {
            ggml_cann_pool_alloc sliced_f16_allocator(ctx.pool());

            إذا (dst->type == GGML_TYPE_F32) {
                int64_t sliced_ne[GGML_MAX_DIMS] = { D, src0_bsnd_ne[1], src0_bsnd_ne[2], src0_bsnd_ne[3] };
                size_t  sliced_nb[GGML_MAX_DIMS];
                sliced_nb[0] = fa_Elem_سائيز;
                لكل (صحيح i = 1; i < GGML_MAX_DIMS; ++i) {
                    sliced_nb[i] = sliced_nb[i - 1] * sliced_ne[i - 1];
                }
                int64_t sliced_nelements = sliced_ne[0] * sliced_ne[1] * sliced_ne[2] * sliced_ne[3];
                فراغ *  sliced_بفر    = sliced_f16_allocator.alloc(sliced_nelements * fa_Elem_سائيز);
                acl_tensor_ptr sliced_f16_tensor = ggml_cann_create_tensor(sliced_بفر, fa_معلومات_Type, fa_Elem_سائيز,
                                                                           sliced_ne, sliced_nb, GGML_MAX_DIMS);

                GGML_CANN_CALL_ACLNN_OP(ctx, ٽڪرو, fa_dst_tensor.get(),
                                        (int64_t) -1, (int64_t) 0, D, (int64_t) 1, sliced_f16_tensor.get());

                acl_tensor_ptr acl_dst_tensor = ggml_cann_create_tensor(dst);
                aclnn_cast(ctx, sliced_f16_tensor.get(), acl_dst_tensor.get(), ggml_cann_type_mapping(dst->type));
            } وإلا {
                acl_tensor_ptr acl_dst_tensor = ggml_cann_create_tensor(dst);
                GGML_CANN_CALL_ACLNN_OP(ctx, ٽڪرو, fa_dst_tensor.get(),
                                        (int64_t) -1, (int64_t) 0, D, (int64_t) 1, acl_dst_tensor.get());
            }
        } وإلا إذا (dst->type == GGML_TYPE_F32) {
            acl_tensor_ptr acl_dst_tensor = ggml_cann_create_tensor(dst);
            aclnn_cast(ctx, fa_dst_tensor.get(), acl_dst_tensor.get(), ggml_cann_type_mapping(dst->type));
        }
    } وإلا {
        GGML_ABORT("Function is not implemented.");
    }
}

ساكن فراغ ggml_cann_ٻاهريون_prod_fp(ggml_backend_cann_context & ctx, ggml_tensor * dst) {
    ggml_tensor * src0 = dst->src[0];  // weight  [ne00=m, ne01=K, ne02, ne03]
    ggml_tensor * src1 = dst->src[1];  // input   [ne10=n, ne11=K, ne12, ne13]
    GGML_TENSOR_ٻٽو_OP_LOCALS

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

    ثابت acl_معلومات_Type src0_acl_type = ggml_cann_type_mapping(src0->type);
    ثابت acl_معلومات_Type src1_acl_type = ggml_cann_type_mapping(src1->type);
    ثابت acl_معلومات_Type dst_acl_type  = ggml_cann_type_mapping(dst->type);
    ثابت size_t      src0_type_sz  = ggml_type_سائيز(src0->type);
    ثابت size_t      src1_type_sz  = ggml_type_سائيز(src1->type);
    ثابت size_t      dst_type_sz   = ggml_type_سائيز(dst->type);

    ثابت int64_t dps2 = ne2 / ne02;
    ثابت int64_t dps3 = ne3 / ne03;

    لكل (int64_t i3 = 0; i3 < ne3; i3++) {
        لكل (int64_t i2 = 0; i2 < ne2; i2++) {
            ثابت int64_t i02 = i2 / dps2;
            ثابت int64_t i03 = i3 / dps3;

            // src0 2D slice at [i02, i03]: ggml [m, K] -> ACL [K, m]
            int64_t src0_ne[2] = { ne00, ne01 };
            size_t  src0_nb[2] = { nb00, nb01 };
            acl_tensor_ptr acl_src0_s = ggml_cann_create_tensor(
                (حرف *) src0->معلومات + i02 * nb02 + i03 * nb03,
                src0_acl_type, src0_type_sz, src0_ne, src0_nb, 2);

            // src1 transposed 2D slice at [i2, i3]: swap ne/nb -> ggml[K,n] -> ACL[n,K]
            int64_t src1_t_ne[2] = { ne11, ne10 };
            size_t  src1_t_nb[2] = { nb11, nb10 };
            acl_tensor_ptr acl_src1_t = ggml_cann_create_tensor(
                (حرف *) src1->معلومات + i2 * nb12 + i3 * nb13,
                src1_acl_type, src1_type_sz, src1_t_ne, src1_t_nb, 2);

            // dst 2D slice at [i2, i3]: ggml [m, n] -> ACL [n, m]
            int64_t dst_ne[2] = { ne0, ne1 };
            size_t  dst_nb[2] = { nb0, nb1 };
            acl_tensor_ptr acl_dst_s = ggml_cann_create_tensor(
                (حرف *) dst->معلومات + i2 * nb2 + i3 * nb3,
                dst_acl_type, dst_type_sz, dst_ne, dst_nb, 2);

            // Matmul(src1_t [n,K], src0 [K,m]) = [n,m] = acl_dst_s  ✓
            GGML_CANN_CALL_ACLNN_OP(ctx, Matmul,
                acl_src1_t.get(), acl_src0_s.get(), acl_dst_s.get(), (int8_t) 1);
        }
    }
}

فراغ ggml_cann_ٻاهريون_prod(ggml_backend_cann_context & ctx, ggml_tensor * dst) {
    ggml_tensor * src0 = dst->src[0];

    ثابت تعداد ggml_type type = src0->type;

    بدل (type) {
        حالة GGML_TYPE_F32:
        حالة GGML_TYPE_F16:
            ggml_cann_ٻاهريون_prod_fp(ctx, dst);
            اكسر;
        افتراضي:
            GGML_ABORT("Unsupport type for GGML_OP_OUT_PROD");
            اكسر;
    }
}

فراغ ggml_cann_ssm_conv(ggml_backend_cann_context & ctx, ggml_tensor * dst) {
    ggml_tensor * src0 = dst->src[0];  // conv_x
    ggml_tensor * src1 = dst->src[1];  // conv1d.weight

    // This op is currently defined only for F32 in ggml_cpu
    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    // Shapes follow ggml_compute_forward_ssm_conv_f32
    ثابت int64_t nc  = src1->ne[0];   // d_conv
    ثابت int64_t ncs = src0->ne[0];   // d_conv - 1 + n_t
    ثابت int64_t nr  = src0->ne[1];   // d_inner
    ثابت int64_t n_s = src0->ne[2];   // n_seqs

    ثابت int64_t n_t = dst->ne[1];    // tokens per sequence

    GGML_ASSERT(dst->ne[0] == nr);     // dst: {d_inner, n_t, n_s}
    GGML_ASSERT(src1->ne[1] == nr);    // weight: {d_conv, d_inner}
    GGML_ASSERT(ncs == nc - 1 + n_t);  // conv_x: {d_conv - 1 + n_t, d_inner, n_s}
    GGML_ASSERT(src0->nb[0] == حجم(عائم));
    GGML_ASSERT(src1->nb[0] == حجم(عائم));

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

    acl_tensor_ptr acl_w = ggml_cann_create_tensor(src1->معلومات, ggml_cann_type_mapping(src1->type),
                                                   ggml_type_سائيز(src1->type), w_ne, w_nb, 3, ACL_FORMAT_NCL);

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
    size_t  y_nb[GGML_MAX_DIMS] = { dst->ne[0] * حجم(عائم), حجم(عائم), dst->ne[0] * dst->ne[1] * حجم(عائم),
                                    dst->nb[3] };       // [nr, 1, nr * n_t]

    acl_tensor_ptr acl_y = ggml_cann_create_tensor(dst->معلومات, ggml_cann_type_mapping(dst->type),
                                                   ggml_type_سائيز(dst->type), y_ne, y_nb, 3, ACL_FORMAT_NCL);

    // --- Conv1d parameters: depthwise, stride 1, no padding ("valid") ---
    int64_t strideVal[1]   = { 1 };
    int64_t paddingVal[1]  = { 0 };
    int64_t dilationVal[1] = { 1 };

    acl_int_array_ptr stride   = ggml_cann_create_int_array(strideVal, 1);
    acl_int_array_ptr padding  = ggml_cann_create_int_array(paddingVal, 1);
    acl_int_array_ptr dilation = ggml_cann_create_int_array(dilationVal, 1);

    ثابت منطقي    transposed   = خطأ;
    ثابت int64_t groups       = nr;  // depthwise: one group per inner dim
    int8_t        cubeMathType = 0;

#جيڪڏهن_وصف_ٿيل ASCEND_310P
    cubeMathType = 1;
#جيڪڏهن_پورو

    GGML_CANN_CALL_ACLNN_OP(ctx, Convolution,
                            acl_x.get(),    // input:  N, C, L_in = ncs
                            acl_w.get(),    // weight: [C, 1, K] with groups=nr
                            خالي_اشارو,        // bias
                            stride.get(), padding.get(), dilation.get(), transposed,
                            padding.get(),  // output padding (unused for non-transposed)
                            groups, acl_y.get(), cubeMathType);
}

فراغ ggml_cann_op_جوڙيو_rms_norm_fused(ggml_backend_cann_context & ctx,
                                     ggml_tensor *               جوڙيو_ڳنڍ,
                                     ggml_tensor *               rms_norm_ڳنڍ) {
    // Get the two input tensors for ADD operation
    ggml_tensor * x1 = جوڙيو_ڳنڍ->src[0];
    ggml_tensor * x2 = جوڙيو_ڳنڍ->src[1];

    // Create ACL tensors for the two ADD inputs
    acl_tensor_ptr acl_x1 = ggml_cann_create_tensor(x1);
    acl_tensor_ptr acl_x2 = ggml_cann_create_tensor(x2);

    // Get epsilon parameter from rms_norm_tensor
    عائم eps;
    يادگيري_نقل(&eps, rms_norm_ڳنڍ->op_params, حجم(عائم));

    // Build gamma tensor (RMS normalization scaling factor)
    // Gamma should match the normalized dimensions (last dimension of x1)
    size_t acl_gamma_nb[GGML_MAX_DIMS];
    acl_gamma_nb[0] = ggml_type_سائيز(rms_norm_ڳنڍ->type);
    لكل (صحيح i = 1; i < GGML_MAX_DIMS; i++) {
        acl_gamma_nb[i] = acl_gamma_nb[i - 1] * x1->ne[i - 1];
    }
    acl_tensor_ptr acl_gamma =
        get_cache_acl_tensor(ctx, &ctx.rms_norm_one_tensor_cache.cache, ctx.rms_norm_one_tensor_cache.size, x1->ne,
                             acl_gamma_nb, rms_norm_ڳنڍ->type,
                             1,    // dims - only the last dimension
                             1.0f  //قدر
        );

    // Build rstdOut tensor (output for normalized standard deviation)
    // Shape should be the dimensions that are NOT normalized
    int64_t acl_rstd_ne[] = { 1, x1->ne[1], x1->ne[2], x1->ne[3] };
    size_t  acl_rstd_nb[GGML_MAX_DIMS - 1];
    acl_rstd_nb[0] = حجم(عائم);
    لكل (صحيح i = 1; i < GGML_MAX_DIMS - 1; i++) {
        acl_rstd_nb[i] = acl_rstd_nb[i - 1] * acl_rstd_ne[i - 1];
    }
    acl_tensor_ptr acl_rstd =
        get_cache_acl_tensor(ctx, &ctx.rms_norm_zero_tensor_cache.cache, ctx.rms_norm_zero_tensor_cache.size,
                             acl_rstd_ne, acl_rstd_nb, GGML_TYPE_F32, GGML_MAX_DIMS,
                             0.0f  //قدر
        );

    acl_tensor_ptr acl_xout = ggml_cann_create_tensor(جوڙيو_ڳنڍ);

    // Create yOut tensor (final output after RMS normalization)
    acl_tensor_ptr acl_yout = ggml_cann_create_tensor(rms_norm_ڳنڍ);

    // Call fused ADD + RMS_NORM operator
    GGML_CANN_CALL_ACLNN_OP(ctx, جوڙيو_Rms_Norm, acl_x1.get(), acl_x2.get(), acl_gamma.get(),
                            eps,  // double type
                            acl_yout.get(), acl_rstd.get(), acl_xout.get());
}

فراغ ggml_cann_gated_لڪيري_attn(ggml_backend_cann_context & ctx, ggml_tensor * dst) {
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
    size_t  nb_بنياد   = ggml_type_سائيز(k->type);
    size_t  nb_qkg[2] = { nb_بنياد, nb_بنياد };
    size_t  nb_s[2]   = { nb_بنياد, D * nb_بنياد };
    size_t  nb_st[2]  = { nb_s[1], nb_s[0] };
    size_t  nb_vo[2]  = { nb_بنياد, D * nb_بنياد };
    size_t  nb_q[1]   = { nb_بنياد };

    ثابت عائم scale = ggml_get_op_params_f32(dst, 0);

    acl_tensor_ptr acl_s     = ggml_cann_create_tensor(s, s->ne, s->nb, 2, ACL_FORMAT_ND);
    acl_tensor_ptr new_state = ggml_cann_create_tensor(dst, s->ne, s->nb, 2, ACL_FORMAT_ND, (B * L * H * D) * nb_بنياد);
    cann_copy(ctx, acl_s.get(), new_state.get());

    لكل (int64_t b = 0; b < B; b++) {
        لكل (int64_t h = 0; h < H; h++) {
            size_t         s_وٿي = (b * (H * D * D) + h * (D * D)) * nb_بنياد;
            // D * D
            acl_tensor_ptr acl_s_new =
                ggml_cann_create_tensor(dst, ne_s, nb_s, 2, ACL_FORMAT_ND, (B * L * H * D) * nb_بنياد + s_وٿي);
            acl_tensor_ptr acl_s_new_t =
                ggml_cann_create_tensor(dst, ne_st, nb_st, 2, ACL_FORMAT_ND, (B * L * H * D) * nb_بنياد + s_وٿي);
            لكل (int64_t l = 0; l < L; l++) {
                size_t               qkvgo_وٿي = (b * (L * H * D) + l * (H * D) + h * (D)) * nb_بنياد;
                // D * 1
                acl_tensor_ptr       acl_k = ggml_cann_create_tensor(k, ne_qkg, nb_qkg, 2, ACL_FORMAT_ND, qkvgo_وٿي);
                acl_tensor_ptr       acl_g = ggml_cann_create_tensor(g, ne_qkg, nb_qkg, 2, ACL_FORMAT_ND, qkvgo_وٿي);
                // D
                acl_tensor_ptr       acl_q = ggml_cann_create_tensor(q, ne_q, nb_q, 1, ACL_FORMAT_ND, qkvgo_وٿي);
                // 1 * D
                acl_tensor_ptr       acl_v = ggml_cann_create_tensor(v, ne_vo, nb_vo, 2, ACL_FORMAT_ND, qkvgo_وٿي);
                // D
                acl_tensor_ptr       acl_o = ggml_cann_create_tensor(dst, ne_q, nb_q, 1, ACL_FORMAT_ND, qkvgo_وٿي);
                // k ⊗ v
                size_t               buf_سائيز = D * D * nb_بنياد;
                ggml_cann_pool_alloc بفر_allocator(ctx.pool(), buf_سائيز);
                acl_tensor_ptr       tmp_tensor = ggml_cann_create_tensor(
                    بفر_allocator.get(), ggml_cann_type_mapping(k->type), nb_بنياد, ne_s, nb_s, 2);
                aclnn_mul(ctx, acl_k.get(), acl_v.get(), tmp_tensor.get());
                //s_new = g ⊗ s_old + k ⊗ v
                aclnn_mul(ctx, acl_s_new.get(), acl_g.get(), خالي_اشارو);
                aclnn_جوڙيو(ctx, acl_s_new.get(), tmp_tensor.get(), خالي_اشارو);
                // compute output
                GGML_CANN_CALL_ACLNN_OP(ctx, Mv, acl_s_new_t.get(), acl_q.get(), acl_o.get(), 1);
                aclnn_muls(ctx, acl_o.get(), scale, خالي_اشارو, سچ);
            }
        }
    }
}
