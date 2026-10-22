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

#ਅਨ੍ਤਰ੍ਭਾਵ "aclnn_ops.h"

#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-impl.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml.h"


#ਅਨ੍ਤਰ੍ਭਾਵ <aclnnop/aclnn_add.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <aclnnop/aclnn_add_rms_norm.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <aclnnop/aclnn_addcdiv.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <aclnnop/aclnn_argmax.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <aclnnop/aclnn_avgpool2d.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <aclnnop/aclnn_batch_matmul.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <aclnnop/aclnn_cast.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <aclnnop/aclnn_clamp.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <aclnnop/aclnn_constant_pad_nd.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <aclnnop/aclnn_convolution.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <aclnnop/aclnn_copy.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <aclnnop/aclnn_div.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <aclnnop/aclnn_elu.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <aclnnop/aclnn_embedding.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <aclnnop/aclnn_eq_tensor.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <aclnnop/aclnn_exp.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <aclnnop/aclnn_fill_scalar.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <aclnnop/aclnn_fused_infer_attention_score_v2.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <aclnnop/aclnn_ger.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <aclnnop/aclnn_group_norm.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <aclnnop/aclnn_gather_v2.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <aclnnop/aclnn_grouped_matmul_v3.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <aclnnop/aclnn_scatter.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <aclnnop/aclnn_gt_scalar.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <aclnnop/aclnn_im2col.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <aclnnop/aclnn_index_copy.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <aclnnop/aclnn_index_fill_tensor.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <aclnnop/aclnn_index_select.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <aclnnop/aclnn_layer_norm.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <aclnnop/aclnn_log.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <aclnnop/aclnn_matmul.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <aclnnop/aclnn_max_pool.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <aclnnop/aclnn_mean.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <aclnnop/aclnn_mm.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <aclnnop/aclnn_mul.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <aclnnop/aclnn_mv.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <aclnnop/aclnn_permute.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <aclnnop/aclnn_pow.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <aclnnop/aclnn_pow_tensor_tensor.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <aclnnop/aclnn_recurrent_gated_delta_rule.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <aclnnop/aclnn_reduce_sum.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <aclnnop/aclnn_reflection_pad1d.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <aclnnop/aclnn_repeat.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <aclnnop/aclnn_repeat_interleave.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <aclnnop/aclnn_rms_norm.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <aclnnop/aclnn_roll.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <aclnnop/aclnn_softmax.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <aclnnop/aclnn_softmax_cross_entropy_with_logits.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <aclnnop/aclnn_sub.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <aclnnop/aclnn_sum.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <aclnnop/aclnn_threshold.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <aclnnop/aclnn_tril.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <aclnnop/aclnn_triangular_solve.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <aclnnop/aclnn_triu.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <aclnnop/aclnn_logical_not.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <aclnnop/aclnn_masked_fill_scalar.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <aclnnop/aclnn_upsample_nearest_2d.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <aclnnop/aclnn_weight_quant_batch_matmul_v2.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <aclnnop/aclnn_zero.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <float.h>

#ਅਨ੍ਤਰ੍ਭਾਵ <cmath>
#ਅਨ੍ਤਰ੍ਭਾਵ <cstring>
#ਅਨ੍ਤਰ੍ਭਾਵ <exception>
#ਅਨ੍ਤਰ੍ਭਾਵ <vector>

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_COMMON_DECL_C

#ਅਨ੍ਤਰ੍ਭਾਵ "../ggml-common.h"

ਖਾਲੀ bcast_shape(ggml_tensor *    src0,
                 ggml_tensor *    src1,
                 ggml_tensor *    dst,
                 acl_tensor_ptr & acl_src0,
                 acl_tensor_ptr & acl_src1,
                 acl_tensor_ptr & acl_dst) {
    GGML_ASSERT(ggml_are_same_shape(src0, dst) && ggml_can_repeat(src1, src0));
    // Need bcast
    ਜੇ (!ggml_are_same_shape(src0, src1) && ggml_cann_need_bcast(src0, src1)) {
        BCAST_SHAPE(src0, src1)
        acl_src0 = ggml_cann_create_tensor(src0, BCAST_PARAM(src0));
        acl_src1 = ggml_cann_create_tensor(src1, BCAST_PARAM(src1));
        acl_dst  = ggml_cann_create_tensor(dst, BCAST_PARAM(src0));
    } ਨਹੀਂਤਾਂ {
        acl_src0 = ggml_cann_create_tensor(src0);
        acl_src1 = ggml_cann_create_tensor(src1);
        acl_dst  = ggml_cann_create_tensor(dst);
    }
}

ਖਾਲੀ ggml_cann_op_unary(std::function<ਖਾਲੀ(ggml_backend_cann_context &, aclTensor *, aclTensor *)> unary_op,
                        ggml_backend_cann_context &                                                ctx,
                        ggml_tensor *                                                              dst) {
    ggml_tensor * src = dst->src[0];

    acl_tensor_ptr acl_src = ggml_cann_create_tensor(src);
    acl_tensor_ptr acl_dst = ggml_cann_create_tensor(dst);

    unary_op(ctx, acl_src.get(), acl_dst.get());
}

ਖਾਲੀ ggml_cann_op_unary_gated(std::function<ਖਾਲੀ(ggml_backend_cann_context &, aclTensor *, aclTensor *)> unary_op,
                              ggml_backend_cann_context &                                                ctx,
                              ggml_tensor *                                                              dst) {
    ggml_tensor * src0 = dst->src[0];
    ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(ggml_is_contiguous_1(src0));
    GGML_ASSERT(ggml_is_contiguous_1(dst));
    ਅਚਲ int32_t swapped = ggml_get_op_params_i32(dst, 1);

    acl_tensor_ptr acl_dst = ggml_cann_create_tensor(dst);
    acl_tensor_ptr acl_src0, acl_src1;
    ਜੇ (src1) {
        GGML_ASSERT(ggml_is_contiguous_1(src1));
        GGML_ASSERT(src0->type == src1->type);

        acl_src0 = ggml_cann_create_tensor(src0);
        acl_src1 = ggml_cann_create_tensor(src1);
    } ਨਹੀਂਤਾਂ {
        int64_t ne[] = { src0->ne[0] / 2, src0->ne[1], src0->ne[2], src0->ne[3] };
        size_t  nb[] = { src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3] };
        acl_src0     = ggml_cann_create_tensor(src0, ne, nb, GGML_MAX_DIMS, ACL_FORMAT_ND, 0);
        acl_src1 = ggml_cann_create_tensor(src0, ne, nb, GGML_MAX_DIMS, ACL_FORMAT_ND, ne[0] * ggml_element_ਮਾਪ(src0));
        ਜੇ (swapped) {
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
ਖਾਲੀ ggml_cann_swiglu(ggml_backend_cann_context & ctx, ggml_tensor * dst) {
    ਸਵੈਚਾਲਿਤ silu_fn = [](ggml_backend_cann_context & ctx, aclTensor * acl_src, aclTensor * acl_dst) {
        GGML_CANN_CALL_ACLNN_OP(ctx, Silu, acl_src, acl_dst);
    };

    ਅਚਲ int32_t swapped = ggml_get_op_params_i32(dst, 1);
    ਜੇ (dst->src[1] != ਖਾਲੀ_ਸੂਚਕ || swapped != 0) {
        ggml_cann_op_unary_gated(silu_fn, ctx, dst);
        ਵਾਪਸ;
    }

    // aclnnSwiGlu requires the split dim (src->ne[0]) to be even; fall back otherwise.
    ਜੇ (dst->src[0]->ne[0] % 2 != 0) {
        ggml_cann_op_unary_gated(silu_fn, ctx, dst);
        ਵਾਪਸ;
    }

    ggml_tensor * src0 = dst->src[0];
    size_t elem_ਮਾਪ = ggml_element_ਮਾਪ(src0);

    // src0 GGML: [2*ne0, ne1, ne2, ne3] → 3D view [2*ne0, ne1, ne2*ne3]
    // CANN reversed: [ne2*ne3, ne1, 2*ne0], split along CANN dim 2 (last).
    int64_t ne0_x2   = src0->ne[0];
    int64_t ne1      = src0->ne[1];
    int64_t ne23     = src0->ne[2] * src0->ne[3];
    int64_t src3d_ne[] = { ne0_x2, ne1, ne23 };
    size_t  src3d_nb[] = { (size_t)src0->nb[0], (size_t)src0->nb[1], (size_t)src0->nb[2] };
    acl_tensor_ptr acl_src = ggml_cann_create_tensor(src0->ਅੰਕੜੇ, ggml_cann_type_mapping(src0->type),
                                                     elem_ਮਾਪ, src3d_ne, src3d_nb, 3);

    // dst GGML: [ne0, ne1, ne2, ne3] → 3D view [ne0, ne1, ne2*ne3]
    int64_t ne0      = dst->ne[0];
    int64_t dst3d_ne[] = { ne0, ne1, ne23 };
    size_t  dst3d_nb[] = { (size_t)dst->nb[0], (size_t)dst->nb[1], (size_t)dst->nb[2] };
    acl_tensor_ptr acl_dst = ggml_cann_create_tensor(dst->ਅੰਕੜੇ, ggml_cann_type_mapping(dst->type),
                                                     elem_ਮਾਪ, dst3d_ne, dst3d_nb, 3);

    // CANN tensor [ne23, ne1, 2*ne0]: split along CANN dim 2 (last) = 2*ne0.
    GGML_CANN_CALL_ACLNN_OP(ctx, SwiGlu, acl_src.get(), (int64_t)2, acl_dst.get());
}

ਖਾਲੀ ggml_cann_swiglu_clamp(ggml_backend_cann_context & ctx, ggml_tensor * dst) {
    ggml_tensor * src0 = dst->src[0];
    ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(ggml_is_contiguous_1(src0));
    GGML_ASSERT(ggml_is_contiguous_1(dst));

    ਅਚਲ int32_t  swapped = ggml_get_op_params_i32(dst, 1);
    acl_tensor_ptr acl_gate;
    acl_tensor_ptr acl_ਉੱਪਰ;
    ਜੇ (src1) {
        GGML_ASSERT(ggml_is_contiguous_1(src1));
        GGML_ASSERT(src0->type == src1->type);
        acl_gate = ggml_cann_create_tensor(src0);
        acl_ਉੱਪਰ   = ggml_cann_create_tensor(src1);
    } ਨਹੀਂਤਾਂ {
        int64_t ne[] = { src0->ne[0] / 2, src0->ne[1], src0->ne[2], src0->ne[3] };
        size_t  nb[] = { src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3] };
        acl_gate     = ggml_cann_create_tensor(src0, ne, nb, GGML_MAX_DIMS, ACL_FORMAT_ND, 0);
        acl_ਉੱਪਰ = ggml_cann_create_tensor(src0, ne, nb, GGML_MAX_DIMS, ACL_FORMAT_ND, ne[0] * ggml_element_ਮਾਪ(src0));
        ਜੇ (swapped) {
            std::swap(acl_gate, acl_ਉੱਪਰ);
        }
    }

    ggml_cann_pool_alloc ਆਰਜ਼ੀ_alloc(ctx.pool(), ggml_nbytes(dst));
    acl_tensor_ptr       acl_ਆਰਜ਼ੀ = ggml_cann_create_tensor(ਆਰਜ਼ੀ_alloc.get(), ggml_cann_type_mapping(dst->type),
                                                            ggml_element_ਮਾਪ(dst), dst->ne, dst->nb, GGML_MAX_DIMS);
    acl_tensor_ptr       acl_dst  = ggml_cann_create_tensor(dst);

    ਅਚਲ ਦਸ਼ਮਲਵ    ਹੱਦਬੰਦੀ        = ggml_get_op_params_f32(dst, 3);
    ਦਸ਼ਮਲਵ          min_gate     = -INFINITY;
    ਦਸ਼ਮਲਵ          min_ਉੱਪਰ       = -ਹੱਦਬੰਦੀ;
    ਦਸ਼ਮਲਵ          max_ਮੁੱਲ    = ਹੱਦਬੰਦੀ;
    acl_scalar_ptr acl_min_gate = ggml_cann_create_scalar(&min_gate, ACL_FLOAT);
    acl_scalar_ptr acl_min_ਉੱਪਰ   = ggml_cann_create_scalar(&min_ਉੱਪਰ, ACL_FLOAT);
    acl_scalar_ptr acl_ਹੱਦਬੰਦੀ    = ggml_cann_create_scalar(&max_ਮੁੱਲ, ACL_FLOAT);

    GGML_CANN_CALL_ACLNN_OP(ctx, Clamp, acl_gate.get(), acl_min_gate.get(), acl_ਹੱਦਬੰਦੀ.get(), acl_ਆਰਜ਼ੀ.get());
    GGML_CANN_CALL_ACLNN_OP(ctx, Silu, acl_ਆਰਜ਼ੀ.get(), acl_dst.get());
    GGML_CANN_CALL_ACLNN_OP(ctx, Clamp, acl_ਉੱਪਰ.get(), acl_min_ਉੱਪਰ.get(), acl_ਹੱਦਬੰਦੀ.get(), acl_ਆਰਜ਼ੀ.get());
    GGML_CANN_CALL_ACLNN_OP(ctx, InplaceMul, acl_dst.get(), acl_ਆਰਜ਼ੀ.get());
}

// Fused GeGLU using aclnnGeGluV3: splits input along ne[0] (CANN last dim),
// activates the LEFT half with GELU, multiplies by right half.
// approximate: 0=tanh, 1=none(erf). activateLeft=true matches GGML convention.
// outGelu is a required-but-discard output buffer.
//
// Falls back to the generic two-kernel path when src[1] != nullptr (two
// independent halves) or swapped != 0 (reversed activation order), as
// aclnnGeGluV3 only handles the single interleaved tensor in standard order.
ਖਾਲੀ ggml_cann_geglu(ggml_backend_cann_context & ctx, ggml_tensor * dst, int64_t approximate) {
    ਸਵੈਚਾਲਿਤ gelu_fn = [](ggml_backend_cann_context & ctx, aclTensor * acl_src, aclTensor * acl_dst) {
        GGML_CANN_CALL_ACLNN_OP(ctx, Gelu, acl_src, acl_dst);
    };

    ਅਚਲ int32_t swapped = ggml_get_op_params_i32(dst, 1);
    ਜੇ (dst->src[1] != ਖਾਲੀ_ਸੂਚਕ || swapped != 0) {
        ggml_cann_op_unary_gated(gelu_fn, ctx, dst);
        ਵਾਪਸ;
    }

    // aclnnGeGluV3 requires the split dim (src->ne[0]) to be even; fall back otherwise.
    ਜੇ (dst->src[0]->ne[0] % 2 != 0) {
        ggml_cann_op_unary_gated(gelu_fn, ctx, dst);
        ਵਾਪਸ;
    }

    ggml_tensor * src0 = dst->src[0];
    acl_tensor_ptr acl_src = ggml_cann_create_tensor(src0);
    acl_tensor_ptr acl_dst = ggml_cann_create_tensor(dst);

    // Allocate a temporary buffer for the required outGelu output (same shape as dst).
    // Build contiguous strides since the pool allocation is a fresh buffer.
    size_t  elem_ਮਾਪ    = ggml_element_ਮਾਪ(dst);
    int64_t ne[GGML_MAX_DIMS] = { dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3] };
    size_t  nb[GGML_MAX_DIMS];
    nb[0] = elem_ਮਾਪ;
    ਲਈ (ਪੂਰਨ i = 1; i < GGML_MAX_DIMS; i++) {
        nb[i] = nb[i - 1] * ne[i - 1];
    }
    size_t gelu_ਬਾਹਰਲਾ_ਮਾਪ = nb[GGML_MAX_DIMS - 1] * ne[GGML_MAX_DIMS - 1];
    ggml_cann_pool_alloc gelu_ਬਾਹਰਲਾ_alloc(ctx.pool(), gelu_ਬਾਹਰਲਾ_ਮਾਪ);

    acl_tensor_ptr acl_gelu_ਬਾਹਰਲਾ = ggml_cann_create_tensor(
        gelu_ਬਾਹਰਲਾ_alloc.get(), ggml_cann_type_mapping(dst->type), elem_ਮਾਪ, ne, nb, GGML_MAX_DIMS);
    // V3 adds activateLeft param; true → Gelu(left)*right, matching GGML convention.
    // GGML dim 0 → CANN last dim (index GGML_MAX_DIMS-1 = 3 for 4D tensor).
    GGML_CANN_CALL_ACLNN_OP(ctx, GeGluV3, acl_src.get(), (int64_t)(GGML_MAX_DIMS - 1), approximate, ਸੱਚ,
                             acl_dst.get(), acl_gelu_ਬਾਹਰਲਾ.get());
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
ਸਥਿਰ ਖਾਲੀ aclnn_repeat(ggml_backend_cann_context & ctx,
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
ਸਥਿਰ ਖਾਲੀ aclnn_cast(ggml_backend_cann_context & ctx,
                       aclTensor *                 acl_src,
                       aclTensor *                 acl_dst,
                       acl_ਅੰਕੜੇ_Type                 cast_ਅੰਕੜੇ_type) {
    GGML_CANN_CALL_ACLNN_OP(ctx, Cast, acl_src, cast_ਅੰਕੜੇ_type, acl_dst);
}

ਖਾਲੀ ggml_cann_repeat(ggml_backend_cann_context & ctx, ggml_tensor * dst) {
    ggml_tensor * src = dst->src[0];
    GGML_ASSERT(ggml_can_repeat(src, dst));

    acl_tensor_ptr acl_src = ggml_cann_create_tensor(src);
    acl_tensor_ptr acl_dst = ggml_cann_create_tensor(dst);

    int64_t repeatsArray[] = { dst->ne[3] / src->ne[3], dst->ne[2] / src->ne[2], dst->ne[1] / src->ne[1],
                               dst->ne[0] / src->ne[0] };

    aclnn_repeat(ctx, acl_src.get(), acl_dst.get(), repeatsArray);
}

ਖਾਲੀ aclnn_ਜੋੜੋ(ggml_backend_cann_context & ctx, aclTensor * acl_src0, aclTensor * acl_src1, aclTensor * acl_dst) {
    ਦਸ਼ਮਲਵ          alpha_ਮੁੱਲ = 1.0f;
    acl_scalar_ptr alpha      = ggml_cann_create_scalar(&alpha_ਮੁੱਲ, acl_ਅੰਕੜੇ_Type::ACL_FLOAT);
    ਜੇ (acl_dst != ਖਾਲੀ_ਸੂਚਕ) {
        GGML_CANN_CALL_ACLNN_OP(ctx, ਜੋੜੋ, acl_src0, acl_src1, alpha.get(), acl_dst);
    } ਨਹੀਂਤਾਂ {
        GGML_CANN_CALL_ACLNN_OP(ctx, Inplace_ਜੋੜੋ, acl_src0, acl_src1, alpha.get());
    }
}

ਖਾਲੀ aclnn_sub(ggml_backend_cann_context & ctx, aclTensor * acl_src0, aclTensor * acl_src1, aclTensor * acl_dst) {
    ਦਸ਼ਮਲਵ          alpha_ਮੁੱਲ = 1.0f;
    acl_scalar_ptr alpha      = ggml_cann_create_scalar(&alpha_ਮੁੱਲ, acl_ਅੰਕੜੇ_Type::ACL_FLOAT);
    ਜੇ (acl_dst != ਖਾਲੀ_ਸੂਚਕ) {
        GGML_CANN_CALL_ACLNN_OP(ctx, Sub, acl_src0, acl_src1, alpha.get(), acl_dst);
    } ਨਹੀਂਤਾਂ {
        GGML_CANN_CALL_ACLNN_OP(ctx, InplaceSub, acl_src0, acl_src1, alpha.get());
    }
}

ਖਾਲੀ aclnn_mul(ggml_backend_cann_context & ctx, aclTensor * acl_src, aclTensor * acl_other, aclTensor * acl_dst) {
    ਜੇ (acl_dst != ਖਾਲੀ_ਸੂਚਕ) {
        GGML_CANN_CALL_ACLNN_OP(ctx, Mul, acl_src, acl_other, acl_dst);
    } ਨਹੀਂਤਾਂ {
        GGML_CANN_CALL_ACLNN_OP(ctx, InplaceMul, acl_src, acl_other);
    }
}

ਖਾਲੀ aclnn_div(ggml_backend_cann_context & ctx, aclTensor * acl_src, aclTensor * acl_other, aclTensor * acl_dst) {
    ਜੇ (acl_dst != ਖਾਲੀ_ਸੂਚਕ) {
        GGML_CANN_CALL_ACLNN_OP(ctx, Div, acl_src, acl_other, acl_dst);
    } ਨਹੀਂਤਾਂ {
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
ਸਥਿਰ ਖਾਲੀ aclnn_muls(ggml_backend_cann_context & ctx,
                       aclTensor *                 acl_src,
                       ਦਸ਼ਮਲਵ                       scale,
                       aclTensor *                 acl_dst,
                       ਤਾਰਕਿਕ                        inplace) {
    acl_scalar_ptr acl_scale = ggml_cann_create_scalar(&scale, acl_ਅੰਕੜੇ_Type::ACL_FLOAT);
    ਜੇ (inplace) {
        GGML_CANN_CALL_ACLNN_OP(ctx, InplaceMuls, acl_src, acl_scale.get());
    } ਨਹੀਂਤਾਂ {
        GGML_CANN_CALL_ACLNN_OP(ctx, Muls, acl_src, acl_scale.get(), acl_dst);
    }
}

ਖਾਲੀ ggml_cann_leaky_relu(ggml_backend_cann_context & ctx, ggml_tensor * dst) {
    ggml_tensor * src = dst->src[0];

    acl_tensor_ptr acl_src = ggml_cann_create_tensor(src);
    acl_tensor_ptr acl_dst = ggml_cann_create_tensor(dst);

    ਦਸ਼ਮਲਵ negative_ਢਲਾਨ;
    ਸਮ੍ਰਿਤੀ_ਨਕਲ(&negative_ਢਲਾਨ, dst->op_params, ਆਕਾਰ(ਦਸ਼ਮਲਵ));
    acl_scalar_ptr acl_negative_ਢਲਾਨ = ggml_cann_create_scalar(&negative_ਢਲਾਨ, acl_ਅੰਕੜੇ_Type::ACL_FLOAT);

    GGML_CANN_CALL_ACLNN_OP(ctx, LeakyRelu, acl_src.get(), acl_negative_ਢਲਾਨ.get(), acl_dst.get());
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
ਸਥਿਰ ਖਾਲੀ aclnn_concat(ggml_backend_cann_context & ctx,
                         aclTensorList *             tensorList,
                         aclTensor *                 acl_dst,
                         int64_t                     concat_dim) {
    GGML_CANN_CALL_ACLNN_OP(ctx, Cat, tensorList, concat_dim, acl_dst);
}

ਖਾਲੀ ggml_cann_concat(ggml_backend_cann_context & ctx, ggml_tensor * dst) {
    ggml_tensor *  src0     = dst->src[0];
    ggml_tensor *  src1     = dst->src[1];
    acl_tensor_ptr acl_src0 = ggml_cann_create_tensor(src0);
    acl_tensor_ptr acl_src1 = ggml_cann_create_tensor(src1);
    acl_tensor_ptr acl_dst  = ggml_cann_create_tensor(dst);

    ਅਚਲ int32_t dim = ggml_get_op_params_i32(dst, 0);

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
ਸਥਿਰ ਖਾਲੀ aclnn_arange(ggml_backend_cann_context & ctx,
                         aclTensor *                 acl_dst,
                         ਦਸ਼ਮਲਵ                       start,
                         ਦਸ਼ਮਲਵ                       stop,
                         ਦਸ਼ਮਲਵ                       ਕਦਮ,
                         int64_t                     n_elements) {
    int64_t ਕਦਮਾਂ = (int64_t) std::ceil((stop - start) / ਕਦਮ);
    GGML_ASSERT(n_elements == ਕਦਮਾਂ);

    acl_scalar_ptr acl_ਸ਼ੁਰੂ = ggml_cann_create_scalar(&start, acl_ਅੰਕੜੇ_Type::ACL_FLOAT);
    acl_scalar_ptr acl_ਸਿਰਾ   = ggml_cann_create_scalar(&stop, acl_ਅੰਕੜੇ_Type::ACL_FLOAT);
    acl_scalar_ptr acl_ਕਦਮ  = ggml_cann_create_scalar(&ਕਦਮ, acl_ਅੰਕੜੇ_Type::ACL_FLOAT);

    GGML_CANN_CALL_ACLNN_OP(ctx, Arange, acl_ਸ਼ੁਰੂ.get(), acl_ਸਿਰਾ.get(), acl_ਕਦਮ.get(), acl_dst);
}

ਖਾਲੀ ggml_cann_arange(ggml_backend_cann_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    acl_tensor_ptr acl_dst = ggml_cann_create_tensor(dst);

    int64_t n_elements = ggml_nelements(dst);
    ਦਸ਼ਮਲਵ   start;
    ਦਸ਼ਮਲਵ   stop;
    ਦਸ਼ਮਲਵ   ਕਦਮ;
    ਸਮ੍ਰਿਤੀ_ਨਕਲ(&start, (ਦਸ਼ਮਲਵ *) dst->op_params + 0, ਆਕਾਰ(ਦਸ਼ਮਲਵ));
    ਸਮ੍ਰਿਤੀ_ਨਕਲ(&stop, (ਦਸ਼ਮਲਵ *) dst->op_params + 1, ਆਕਾਰ(ਦਸ਼ਮਲਵ));
    ਸਮ੍ਰਿਤੀ_ਨਕਲ(&ਕਦਮ, (ਦਸ਼ਮਲਵ *) dst->op_params + 2, ਆਕਾਰ(ਦਸ਼ਮਲਵ));

    aclnn_arange(ctx, acl_dst.get(), start, stop, ਕਦਮ, n_elements);
}

ਖਾਲੀ ggml_cann_clamp(ggml_backend_cann_context & ctx, ggml_tensor * dst) {
    ggml_tensor * src = dst->src[0];

    ਦਸ਼ਮਲਵ min;
    ਦਸ਼ਮਲਵ max;
    ਸਮ੍ਰਿਤੀ_ਨਕਲ(&min, dst->op_params, ਆਕਾਰ(ਦਸ਼ਮਲਵ));
    ਸਮ੍ਰਿਤੀ_ਨਕਲ(&max, (ਦਸ਼ਮਲਵ *) dst->op_params + 1, ਆਕਾਰ(ਦਸ਼ਮਲਵ));

    acl_tensor_ptr acl_src = ggml_cann_create_tensor(src);
    acl_tensor_ptr acl_dst = ggml_cann_create_tensor(dst);

    acl_scalar_ptr acl_min = ggml_cann_create_scalar(&min, acl_ਅੰਕੜੇ_Type::ACL_FLOAT);
    acl_scalar_ptr acl_max = ggml_cann_create_scalar(&max, acl_ਅੰਕੜੇ_Type::ACL_FLOAT);

    GGML_CANN_CALL_ACLNN_OP(ctx, Clamp, acl_src.get(), acl_min.get(), acl_max.get(), acl_dst.get());
}

ਖਾਲੀ ggml_cann_scale(ggml_backend_cann_context & ctx, ggml_tensor * dst) {
    ggml_tensor * src = dst->src[0];

    // scale factor
    ਦਸ਼ਮਲਵ v;
    ਸਮ੍ਰਿਤੀ_ਨਕਲ(&v, dst->op_params, ਆਕਾਰ(ਦਸ਼ਮਲਵ));

    acl_scalar_ptr scale   = ggml_cann_create_scalar(&v, acl_ਅੰਕੜੇ_Type::ACL_FLOAT);
    acl_tensor_ptr acl_src = ggml_cann_create_tensor(src);
    acl_tensor_ptr acl_dst = ggml_cann_create_tensor(dst);

    GGML_CANN_CALL_ACLNN_OP(ctx, Muls, acl_src.get(), scale.get(), acl_dst.get());
}

ਖਾਲੀ ggml_cann_argsort(ggml_backend_cann_context & ctx, ggml_tensor * dst) {
    ggml_tensor *        src   = dst->src[0];
    ਗਿਣਤੀ ggml_ਛਾਂਟੋ_ਕ੍ਰਮ ਕ੍ਰਮ = (ਗਿਣਤੀ ggml_ਛਾਂਟੋ_ਕ੍ਰਮ) dst->op_params[0];

    acl_tensor_ptr       acl_src = ggml_cann_create_tensor(src);
    acl_tensor_ptr       acl_dst = ggml_cann_create_tensor(dst);
    ggml_cann_pool_alloc ਆਰਜ਼ੀ_ਬਫਰ_allocator(ctx.pool(), ggml_nelements(dst) * ਆਕਾਰ(int64_t));
    ਖਾਲੀ *               ਬਫਰ = ਆਰਜ਼ੀ_ਬਫਰ_allocator.get();
    acl_tensor_ptr       tmp_tensor =
        ggml_cann_create_tensor(ਬਫਰ, ACL_INT64, ggml_type_ਮਾਪ(dst->type), dst->ne, dst->nb, GGML_MAX_DIMS);
    GGML_CANN_CALL_ACLNN_OP(ctx, Argsort, acl_src.get(), -1, (ਕ੍ਰਮ == GGML_ਛਾਂਟੋ_ਕ੍ਰਮ_DESC ? ਸੱਚ : ਝੂਠ),
                            tmp_tensor.get());
    GGML_CANN_CALL_ACLNN_OP(ctx, Cast, tmp_tensor.get(), ggml_cann_type_mapping(dst->type), acl_dst.get());
}

ਖਾਲੀ ggml_cann_norm(ggml_backend_cann_context & ctx, ggml_tensor * dst) {
    ggml_tensor * src = dst->src[0];

    acl_tensor_ptr acl_src = ggml_cann_create_tensor(src);
    acl_tensor_ptr acl_dst = ggml_cann_create_tensor(dst);

    ਦਸ਼ਮਲਵ eps;
    ਸਮ੍ਰਿਤੀ_ਨਕਲ(&eps, dst->op_params, ਆਕਾਰ(ਦਸ਼ਮਲਵ));

    std::vector<int64_t> norm_ਅੰਕੜੇ = { dst->ne[0] };
    acl_int_array_ptr    norm     = ggml_cann_create_int_array(norm_ਅੰਕੜੇ.ਅੰਕੜੇ(), norm_ਅੰਕੜੇ.size());
    GGML_CANN_CALL_ACLNN_OP(ctx, LayerNorm, acl_src.get(), norm.get(), ਖਾਲੀ_ਸੂਚਕ, ਖਾਲੀ_ਸੂਚਕ, eps, acl_dst.get(), ਖਾਲੀ_ਸੂਚਕ,
                            ਖਾਲੀ_ਸੂਚਕ);
}

ਖਾਲੀ ggml_cann_l2_norm(ggml_backend_cann_context & ctx, ggml_tensor * dst) {
    ggml_tensor * src = dst->src[0];

    ਦਸ਼ਮਲਵ eps;
    ਸਮ੍ਰਿਤੀ_ਨਕਲ(&eps, dst->op_params, ਆਕਾਰ(ਦਸ਼ਮਲਵ));

    acl_tensor_ptr acl_src = ggml_cann_create_tensor(src);
    acl_tensor_ptr acl_dst = ggml_cann_create_tensor(dst);

    size_t               type_ਮਾਪ = ggml_type_ਮਾਪ(src->type);
    int64_t              n_ਬਾਈਟਾਂ   = src->ne[3] * src->ne[2] * src->ne[1] * type_ਮਾਪ;
    ggml_cann_pool_alloc ਆਰਜ਼ੀ_ਬਫਰ_allocator(ctx.pool(), n_ਬਾਈਟਾਂ);
    ਖਾਲੀ *               ਬਫਰ = ਆਰਜ਼ੀ_ਬਫਰ_allocator.get();

    int64_t norm_ne[] = { 1, src->ne[1], src->ne[2], src->ne[3] };
    size_t  norm_nb[GGML_MAX_DIMS];
    norm_nb[0] = ਆਕਾਰ(ਦਸ਼ਮਲਵ);
    ਲਈ (ਪੂਰਨ i = 1; i < GGML_MAX_DIMS; ++i) {
        norm_nb[i] = norm_nb[i - 1] * norm_ne[i - 1];
    }
    acl_tensor_ptr acl_norm = ggml_cann_create_tensor(ਬਫਰ, ACL_FLOAT, ਆਕਾਰ(ਦਸ਼ਮਲਵ), norm_ne, norm_nb, GGML_MAX_DIMS);

    std::vector<int64_t> norm_dims  = { 3 };
    acl_int_array_ptr    dims_array = ggml_cann_create_int_array(norm_dims.ਅੰਕੜੇ(), norm_dims.size());

    ਦਸ਼ਮਲਵ          p_ਮੁੱਲ  = 2.0f;
    acl_scalar_ptr p_scalar = ggml_cann_create_scalar(&p_ਮੁੱਲ, acl_ਅੰਕੜੇ_Type::ACL_FLOAT);
    GGML_CANN_CALL_ACLNN_OP(ctx, Norm, acl_src.get(), p_scalar.get(), dims_array.get(), ਸੱਚ, acl_norm.get());

    ggml_cann_pool_alloc clamp_ਬਫਰ_allocator(ctx.pool());
    acl_tensor_ptr       acl_clamped;

    ਜੇ (eps > 0.0f) {
        ਖਾਲੀ *         clamp_buf  = clamp_ਬਫਰ_allocator.alloc(n_ਬਾਈਟਾਂ);
        acl_clamped               = ggml_cann_create_tensor(clamp_buf, ACL_FLOAT, ਆਕਾਰ(ਦਸ਼ਮਲਵ), norm_ne, norm_nb, GGML_MAX_DIMS);
        acl_scalar_ptr eps_scalar = ggml_cann_create_scalar(&eps, acl_ਅੰਕੜੇ_Type::ACL_FLOAT);
        GGML_CANN_CALL_ACLNN_OP(ctx, ClampMin, acl_norm.get(), eps_scalar.get(), acl_clamped.get());
    }

    aclTensor * acl_div_ਨਿਵੇਸ਼ = acl_clamped ? acl_clamped.get() : acl_norm.get();
    GGML_CANN_CALL_ACLNN_OP(ctx, Div, acl_src.get(), acl_div_ਨਿਵੇਸ਼, acl_dst.get());
}

ਖਾਲੀ ggml_cann_cross_entropy_loss(ggml_backend_cann_context & ctx, ggml_tensor * dst) {
    ggml_tensor * src0 = dst->src[0];
    ggml_tensor * src1 = dst->src[1];

    ਅਚਲ int64_t nc = src0->ne[0];
    ਅਚਲ int64_t nr = ggml_nrows(src0);

    int64_t logits_ne[] = { nc, nr };
    size_t  logits_nb[2];
    logits_nb[0]              = ggml_type_ਮਾਪ(src0->type);
    logits_nb[1]              = logits_nb[0] * logits_ne[0];
    acl_tensor_ptr acl_logits = ggml_cann_create_tensor(src0->ਅੰਕੜੇ, ACL_FLOAT, ਆਕਾਰ(ਦਸ਼ਮਲਵ), logits_ne, logits_nb, 2);

    int64_t labels_ne[] = { nc, nr };
    size_t  labels_nb[2];
    labels_nb[0]              = ggml_type_ਮਾਪ(src1->type);
    labels_nb[1]              = labels_nb[0] * labels_ne[0];
    acl_tensor_ptr acl_labels = ggml_cann_create_tensor(src1->ਅੰਕੜੇ, ACL_FLOAT, ਆਕਾਰ(ਦਸ਼ਮਲਵ), labels_ne, labels_nb, 2);

    size_t               loss_ਪ੍ਰਤੀ_sample_type_ਮਾਪ = ਆਕਾਰ(ਦਸ਼ਮਲਵ);
    int64_t              loss_ਪ੍ਰਤੀ_sample_n_ਬਾਈਟਾਂ   = nr * loss_ਪ੍ਰਤੀ_sample_type_ਮਾਪ;
    ggml_cann_pool_alloc loss_ਪ੍ਰਤੀ_sample_allocator(ctx.pool(), loss_ਪ੍ਰਤੀ_sample_n_ਬਾਈਟਾਂ);
    ਖਾਲੀ *               loss_ਪ੍ਰਤੀ_sample_ਬਫਰ = loss_ਪ੍ਰਤੀ_sample_allocator.get();

    int64_t loss_ਪ੍ਰਤੀ_sample_ne[] = { nr };
    size_t  loss_ਪ੍ਰਤੀ_sample_nb[1];
    loss_ਪ੍ਰਤੀ_sample_nb[0] = loss_ਪ੍ਰਤੀ_sample_type_ਮਾਪ;
    acl_tensor_ptr acl_loss_ਪ੍ਰਤੀ_sample = ggml_cann_create_tensor(
        loss_ਪ੍ਰਤੀ_sample_ਬਫਰ, ACL_FLOAT, loss_ਪ੍ਰਤੀ_sample_type_ਮਾਪ, loss_ਪ੍ਰਤੀ_sample_ne, loss_ਪ੍ਰਤੀ_sample_nb, 1);

    size_t               backprop_n_ਬਾਈਟਾਂ = nr * nc * ਆਕਾਰ(ਦਸ਼ਮਲਵ);
    ggml_cann_pool_alloc backprop_allocator(ctx.pool(), backprop_n_ਬਾਈਟਾਂ);
    ਖਾਲੀ *               backprop_ਬਫਰ = backprop_allocator.get();
    acl_tensor_ptr acl_backprop = ggml_cann_create_tensor(backprop_ਬਫਰ, ACL_FLOAT, ਆਕਾਰ(ਦਸ਼ਮਲਵ), logits_ne, logits_nb, 2);

    GGML_CANN_CALL_ACLNN_OP(ctx, SoftmaxCrossEntropyWithLogits, acl_logits.get(), acl_labels.get(),
                            acl_loss_ਪ੍ਰਤੀ_sample.get(), acl_backprop.get());

    size_t               ਕੁੱਲ_ਜੋੜ_type_ਮਾਪ = ਆਕਾਰ(ਦਸ਼ਮਲਵ);
    int64_t              ਕੁੱਲ_ਜੋੜ_n_ਬਾਈਟਾਂ   = 1 * ਕੁੱਲ_ਜੋੜ_type_ਮਾਪ;
    ggml_cann_pool_alloc ਕੁੱਲ_ਜੋੜ_allocator(ctx.pool(), ਕੁੱਲ_ਜੋੜ_n_ਬਾਈਟਾਂ);
    ਖਾਲੀ *               ਕੁੱਲ_ਜੋੜ_ਬਫਰ = ਕੁੱਲ_ਜੋੜ_allocator.get();

    int64_t ਕੁੱਲ_ਜੋੜ_ne[] = { 1 };
    size_t  ਕੁੱਲ_ਜੋੜ_nb[1];
    ਕੁੱਲ_ਜੋੜ_nb[0] = ਕੁੱਲ_ਜੋੜ_type_ਮਾਪ;

    acl_tensor_ptr acl_ਕੁੱਲ_ਜੋੜ =
        ggml_cann_create_tensor(ਕੁੱਲ_ਜੋੜ_ਬਫਰ, ACL_FLOAT, ਕੁੱਲ_ਜੋੜ_type_ਮਾਪ, ਕੁੱਲ_ਜੋੜ_ne, ਕੁੱਲ_ਜੋੜ_nb, 1);

    std::vector<int64_t> ਕੁੱਲ_ਜੋੜ_dims    = { 0 };
    acl_int_array_ptr ਕੁੱਲ_ਜੋੜ_dims_array = ggml_cann_create_int_array(ਕੁੱਲ_ਜੋੜ_dims.ਅੰਕੜੇ(), ਕੁੱਲ_ਜੋੜ_dims.size());
    ਤਾਰਕਿਕ              keep_dims            = ਝੂਠ;

    GGML_CANN_CALL_ACLNN_OP(ctx, Reduce_ਜੋੜ, acl_loss_ਪ੍ਰਤੀ_sample.get(), ਕੁੱਲ_ਜੋੜ_dims_array.get(), keep_dims, ACL_FLOAT,
                            acl_ਕੁੱਲ_ਜੋੜ.get());

    ਦਸ਼ਮਲਵ          value        = 1.0f / static_cast<ਦਸ਼ਮਲਵ>(nr);
    acl_scalar_ptr scale_ਗੁਣਕ = ggml_cann_create_scalar(&value, acl_ਅੰਕੜੇ_Type::ACL_FLOAT);
    acl_tensor_ptr acl_dst =
        ggml_cann_create_tensor(dst->ਅੰਕੜੇ, ACL_FLOAT, ਆਕਾਰ(ਦਸ਼ਮਲਵ), ਕੁੱਲ_ਜੋੜ_ne, ਕੁੱਲ_ਜੋੜ_nb, 1);

    GGML_CANN_CALL_ACLNN_OP(ctx, Muls, acl_ਕੁੱਲ_ਜੋੜ.get(), scale_ਗੁਣਕ.get(), acl_dst.get());
}

ਖਾਲੀ ggml_cann_ਸਮੂਹ_norm(ggml_backend_cann_context & ctx, ggml_tensor * dst) {
    ggml_tensor * src = dst->src[0];

    acl_tensor_ptr acl_src = ggml_cann_create_tensor(src);
    acl_tensor_ptr acl_dst = ggml_cann_create_tensor(dst);

    ਪੂਰਨ n_groups = dst->op_params[0];

    ਦਸ਼ਮਲਵ eps;
    ਸਮ੍ਰਿਤੀ_ਨਕਲ(&eps, dst->op_params + 1, ਆਕਾਰ(ਦਸ਼ਮਲਵ));

    int64_t N   = src->ne[3];
    int64_t C   = src->ne[2];
    int64_t HxW = src->ne[1] * src->ne[0];

    size_t  type_ਮਾਪ = ggml_type_ਮਾਪ(src->type);
    int64_t ne[]      = { n_groups, N };
    size_t  nb[]      = { type_ਮਾਪ, type_ਮਾਪ * n_groups };
    size_t  n_ਬਾਈਟਾਂ   = N * n_groups;

    ggml_cann_pool_alloc ਆਰਜ਼ੀ_ਬਫਰ_allocator(ctx.pool(), n_ਬਾਈਟਾਂ * 2);
    ਖਾਲੀ *               ਬਫਰ       = ਆਰਜ਼ੀ_ਬਫਰ_allocator.get();
    acl_tensor_ptr       acl_ਔਸਤ_ਬਾਹਰਲਾ = ggml_cann_create_tensor(ਬਫਰ, ACL_FLOAT, type_ਮਾਪ, ne, nb, ACL_FORMAT_ND);
    acl_tensor_ptr       acl_rstd_ਬਾਹਰਲਾ =
        ggml_cann_create_tensor((ਅੱਖਰ *) ਬਫਰ + n_ਬਾਈਟਾਂ, ACL_FLOAT, type_ਮਾਪ, ne, nb, ACL_FORMAT_ND);

    GGML_CANN_CALL_ACLNN_OP(ctx, ਸਮੂਹ_Norm, acl_src.get(), ਖਾਲੀ_ਸੂਚਕ, ਖਾਲੀ_ਸੂਚਕ, N, C, HxW, n_groups, eps, acl_dst.get(),
                            acl_ਔਸਤ_ਬਾਹਰਲਾ.get(), acl_rstd_ਬਾਹਰਲਾ.get());
}

ਖਾਲੀ ggml_cann_ਸੈੱਟ(ggml_backend_cann_context & ctx, ggml_tensor * dst) {
    ggml_tensor * src0 = dst->src[0];
    ggml_tensor * src1 = dst->src[1];

    size_t nb1     = ((int32_t *) dst->op_params)[0];
    size_t nb2     = ((int32_t *) dst->op_params)[1];
    size_t nb3     = ((int32_t *) dst->op_params)[2];
    size_t ਵਿੱਥ  = ((int32_t *) dst->op_params)[3];
    ਤਾਰਕਿਕ   inplace = (ਤਾਰਕਿਕ) ((int32_t *) dst->op_params)[4];

    size_t param_nb[] = { ggml_element_ਮਾਪ(src0), nb1, nb2, nb3 };

    // Create a view of dst at the target offset with src1's dimensions
    acl_tensor_ptr acl_dst  = ggml_cann_create_tensor(dst, src1->ne, param_nb, GGML_MAX_DIMS, ACL_FORMAT_ND, ਵਿੱਥ);
    acl_tensor_ptr acl_src1 = ggml_cann_create_tensor(src1);

    ਜੇ (!inplace) {
        // First copy src0 to dst entirely
        size_t cpy_ਮਾਪ = ggml_nbytes(dst);
        ACL_CHECK(
            aclrtMemcpyAsync(dst->ਅੰਕੜੇ, cpy_ਮਾਪ, src0->ਅੰਕੜੇ, cpy_ਮਾਪ, ACL_MEMCPY_DEVICE_ਮੰਜ਼ਿਲ_DEVICE, ctx.stream()));
    }

    // Copy src1 into the target region of dst
    GGML_CANN_CALL_ACLNN_OP(ctx, InplaceCopy, acl_dst.get(), acl_src1.get());
}

ਖਾਲੀ ggml_cann_acc(ggml_backend_cann_context & ctx, ggml_tensor * dst) {
    ggml_tensor * src0 = dst->src[0];
    ggml_tensor * src1 = dst->src[1];

    size_t nb1     = ((int32_t *) dst->op_params)[0];
    size_t nb2     = ((int32_t *) dst->op_params)[1];
    size_t nb3     = ((int32_t *) dst->op_params)[2];
    size_t ਵਿੱਥ  = ((int32_t *) dst->op_params)[3];
    ਤਾਰਕਿਕ   inplace = (ਤਾਰਕਿਕ) ((int32_t *) dst->op_params)[4];

    size_t param_nb[] = { ggml_element_ਮਾਪ(src0), nb1, nb2, nb3 };

    acl_tensor_ptr acl_dst  = ggml_cann_create_tensor(dst, src1->ne, param_nb, GGML_MAX_DIMS, ACL_FORMAT_ND, ਵਿੱਥ);
    acl_tensor_ptr acl_src1 = ggml_cann_create_tensor(src1);

    acl_scalar_ptr alpha      = ਖਾਲੀ_ਸੂਚਕ;
    ਦਸ਼ਮਲਵ          alpha_ਮੁੱਲ = 1.0f;
    alpha                     = ggml_cann_create_scalar(&alpha_ਮੁੱਲ, acl_ਅੰਕੜੇ_Type::ACL_FLOAT);

    ਜੇ (!inplace) {
        size_t cpy_ਮਾਪ = ggml_nbytes(dst);
        ACL_CHECK(
            aclrtMemcpyAsync(dst->ਅੰਕੜੇ, cpy_ਮਾਪ, src0->ਅੰਕੜੇ, cpy_ਮਾਪ, ACL_MEMCPY_DEVICE_ਮੰਜ਼ਿਲ_DEVICE, ctx.stream()));
        acl_tensor_ptr acl_src0 =
            ggml_cann_create_tensor(src0, src1->ne, src0->nb, GGML_MAX_DIMS, ACL_FORMAT_ND, ਵਿੱਥ);

        GGML_CANN_CALL_ACLNN_OP(ctx, ਜੋੜੋ, acl_src0.get(), acl_src1.get(), alpha.get(), acl_dst.get());
    } ਨਹੀਂਤਾਂ {
        GGML_CANN_CALL_ACLNN_OP(ctx, Inplace_ਜੋੜੋ, acl_dst.get(), acl_src1.get(), alpha.get());
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
ਸਥਿਰ ਖਾਲੀ aclnn_reduce_ਜੋੜ(ggml_backend_cann_context & ctx, ggml_tensor * dst, int64_t * dim, size_t dim_ਮਾਪ) {
    GGML_ASSERT(dst->ne[0] == 1);
    ggml_tensor *     src         = dst->src[0];
    acl_tensor_ptr    acl_src     = ggml_cann_create_tensor(src);
    acl_tensor_ptr    acl_dst     = ggml_cann_create_tensor(dst);
    acl_int_array_ptr reduce_dims = ggml_cann_create_int_array(dim, dim_ਮਾਪ);

    GGML_CANN_CALL_ACLNN_OP(ctx, Reduce_ਜੋੜ, acl_src.get(), reduce_dims.get(), ਸੱਚ, ggml_cann_type_mapping(dst->type),
                            acl_dst.get());
}

ਖਾਲੀ ggml_cann_ਜੋੜ_ਪੰਕਤੀਆਂ(ggml_backend_cann_context & ctx, ggml_tensor * dst) {
    int64_t reduce_dims[] = { 3 };
    aclnn_reduce_ਜੋੜ(ctx, dst, reduce_dims, 1);
}

ਖਾਲੀ ggml_cann_ਜੋੜ(ggml_backend_cann_context & ctx, ggml_tensor * dst) {
    int64_t reduce_dims[] = { 0, 1, 2, 3 };
    aclnn_reduce_ਜੋੜ(ctx, dst, reduce_dims, 4);
}

ਖਾਲੀ ggml_cann_cumsum(ggml_backend_cann_context & ctx, ggml_tensor * dst) {
    ggml_tensor * src = dst->src[0];
    acl_tensor_ptr acl_src = ggml_cann_create_tensor(src);
    acl_tensor_ptr acl_dst = ggml_cann_create_tensor(dst);
    // GGML cumsum operates along dim 0 (innermost / ne[0]).
    // ggml_cann_create_tensor reverses dimensions to [ne3,ne2,ne1,ne0],
    // so GGML dim 0 maps to CANN dim 3 (the last dim of the 4-D tensor).
    GGML_CANN_CALL_ACLNN_OP(ctx, Cumsum, acl_src.get(), (int64_t)3,
                            ggml_cann_type_mapping(dst->type), acl_dst.get());
}

ਖਾਲੀ ggml_cann_solve_tri(ggml_backend_cann_context & ctx, ggml_tensor * dst) {
    ggml_tensor * src0 = dst->src[0];  // A: [N, N, B2, B3] lower triangular
    ggml_tensor * src1 = dst->src[1];  // B: [K, N, B2, B3]

    acl_tensor_ptr acl_a = ggml_cann_create_tensor(src0);
    acl_tensor_ptr acl_b = ggml_cann_create_tensor(src1);
    acl_tensor_ptr acl_x = ggml_cann_create_tensor(dst);

    // mOut: triangular copy of A (required output), same shape as A.
    ਅਚਲ size_t a_ਬਾਈਟਾਂ = ggml_nbytes(src0);
    ggml_cann_pool_alloc m_alloc(ctx.pool(), a_ਬਾਈਟਾਂ);
    acl_tensor_ptr acl_m = ggml_cann_create_tensor(
        m_alloc.get(), ggml_cann_type_mapping(src0->type),
        ggml_type_ਮਾਪ(src0->type), src0->ne, src0->nb, GGML_MAX_DIMS);

    // Solve AX = B: upper=false (lower tri), transpose=false, unitriangular=false.
    GGML_CANN_CALL_ACLNN_OP(ctx, TriangularSolve,
        acl_b.get(), acl_a.get(), ਝੂਠ, ਝੂਠ, ਝੂਠ,
        acl_x.get(), acl_m.get());
}

ਖਾਲੀ ggml_cann_diag(ggml_backend_cann_context & ctx, ggml_tensor * dst) {
    ggml_tensor * src = dst->src[0];

    GGML_ASSERT(src->ne[1] == 1);

    ਅਚਲ int64_t N       = src->ne[0];
    ਅਚਲ int64_t n_batch = src->ne[2] * src->ne[3];
    ਅਚਲ size_t  nb_f32  = ਆਕਾਰ(ਦਸ਼ਮਲਵ);

    // Fill dst with zeros.
    acl_tensor_ptr acl_dst = ggml_cann_create_tensor(dst);
    {
        ਦਸ਼ਮਲਵ          zero = 0.0f;
        acl_scalar_ptr acl_zero = ggml_cann_create_scalar(&zero, ACL_FLOAT);
        GGML_CANN_CALL_ACLNN_OP(ctx, InplaceFillScalar, acl_dst.get(), acl_zero.get());
    }

    // Copy src vector onto the diagonal of dst via strided views.
    // src viewed as [N, n_batch], contiguous strides.
    int64_t ne_vec[2]      = { N, n_batch };
    size_t  nb_src_vec[2]  = { nb_f32, N * nb_f32 };
    // dst diagonal view: stride (N+1)*4 steps along the diagonal.
    size_t  nb_dst_diag[2] = { (N + 1) * nb_f32, N * N * nb_f32 };

    acl_tensor_ptr acl_src_vec  = ggml_cann_create_tensor(src->ਅੰਕੜੇ, ACL_FLOAT, nb_f32, ne_vec, nb_src_vec, 2);
    acl_tensor_ptr acl_dst_diag = ggml_cann_create_tensor(dst->ਅੰਕੜੇ, ACL_FLOAT, nb_f32, ne_vec, nb_dst_diag, 2);

    GGML_CANN_CALL_ACLNN_OP(ctx, InplaceCopy, acl_dst_diag.get(), acl_src_vec.get());
}

ਖਾਲੀ ggml_cann_fill(ggml_backend_cann_context & ctx, ggml_tensor * dst) {
    ਦਸ਼ਮਲਵ c = ggml_get_op_params_f32(dst, 0);

    acl_tensor_ptr acl_dst = ggml_cann_create_tensor(dst);
    acl_scalar_ptr acl_c   = ggml_cann_create_scalar(&c, ACL_FLOAT);
    GGML_CANN_CALL_ACLNN_OP(ctx, InplaceFillScalar, acl_dst.get(), acl_c.get());
}

ਖਾਲੀ ggml_cann_tri(ggml_backend_cann_context & ctx, ggml_tensor * dst) {
    ggml_tensor * src = dst->src[0];

    ਅਚਲ int64_t S       = src->ne[0];
    ਅਚਲ int64_t n_batch = src->ne[2] * src->ne[3];
    ਅਚਲ size_t  nb_f32  = ਆਕਾਰ(ਦਸ਼ਮਲਵ);

    int64_t ne3d[3] = { S, S, n_batch };
    size_t  nb3d[3] = { nb_f32, S * nb_f32, S * S * nb_f32 };

    ਅਚਲ ggml_tri_type ttype = (ggml_tri_type) ggml_get_op_params_i32(dst, 0);

    acl_tensor_ptr acl_src = ggml_cann_create_tensor(src->ਅੰਕੜੇ, ACL_FLOAT, nb_f32, ne3d, nb3d, 3);
    acl_tensor_ptr acl_dst = ggml_cann_create_tensor(dst->ਅੰਕੜੇ, ACL_FLOAT, nb_f32, ne3d, nb3d, 3);

    ਚੋਣ (ttype) {
        ਹਾਲਤ GGML_TRI_TYPE_ਹੇਠਲਾ:
            // Tril(-1): preserve row > col (strict lower), zero upper + diagonal.
            GGML_CANN_CALL_ACLNN_OP(ctx, Tril, acl_src.get(), (int64_t)-1, acl_dst.get());
            ਰੋਕੋ;
        ਹਾਲਤ GGML_TRI_TYPE_ਉੱਤਲਾ_DIAG:
            // Triu(0): preserve row <= col (upper + diagonal), zero strict lower.
            GGML_CANN_CALL_ACLNN_OP(ctx, Triu, acl_src.get(), (int64_t)0, acl_dst.get());
            ਰੋਕੋ;
        ਹਾਲਤ GGML_TRI_TYPE_ਉੱਤਲਾ:
            // Triu(1): preserve row < col (strict upper), zero lower + diagonal.
            GGML_CANN_CALL_ACLNN_OP(ctx, Triu, acl_src.get(), (int64_t)1, acl_dst.get());
            ਰੋਕੋ;
        ਹਾਲਤ GGML_TRI_TYPE_ਹੇਠਲਾ_DIAG:
            // Tril(0): preserve row >= col (lower + diagonal), zero strict upper.
            GGML_CANN_CALL_ACLNN_OP(ctx, Tril, acl_src.get(), (int64_t)0, acl_dst.get());
            ਰੋਕੋ;
        ਮੂਲ:
            GGML_ABORT("unsupported tri type");
    }
}

ਖਾਲੀ ggml_cann_upsample_nearest2d(ggml_backend_cann_context & ctx, ggml_tensor * dst) {
    ggml_tensor *  src     = dst->src[0];
    acl_tensor_ptr acl_src = ggml_cann_create_tensor(src, ਖਾਲੀ_ਸੂਚਕ, ਖਾਲੀ_ਸੂਚਕ, 0, ACL_FORMAT_NCHW);
    acl_tensor_ptr acl_dst = ggml_cann_create_tensor(dst, ਖਾਲੀ_ਸੂਚਕ, ਖਾਲੀ_ਸੂਚਕ, 0, ACL_FORMAT_NCHW);

    std::vector<int64_t> output_ਮਾਪ{ dst->ne[1], dst->ne[0] };
    acl_int_array_ptr    output_ਮਾਪ_array = ggml_cann_create_int_array(output_ਮਾਪ.ਅੰਕੜੇ(), 2);

    GGML_CANN_CALL_ACLNN_OP(ctx, UpsampleNearest2d, acl_src.get(), output_ਮਾਪ_array.get(), acl_dst.get());
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
ਸਥਿਰ ਖਾਲੀ aclnn_pad(ggml_backend_cann_context & ctx,
                      aclTensor *                 acl_src,
                      aclTensor *                 acl_dst,
                      int64_t *                   paddings,
                      ਦਸ਼ਮਲਵ                       value = 0.0f) {
    acl_int_array_ptr acl_pad   = ggml_cann_create_int_array(paddings, GGML_MAX_DIMS * 2);
    acl_scalar_ptr    acl_ਮੁੱਲ = ggml_cann_create_scalar(&value, acl_ਅੰਕੜੇ_Type::ACL_FLOAT);

    GGML_CANN_CALL_ACLNN_OP(ctx, ConstantPadNd, acl_src, acl_pad.get(), acl_ਮੁੱਲ.get(), acl_dst);
}

ਖਾਲੀ ggml_cann_pad(ggml_backend_cann_context & ctx, ggml_tensor * dst) {
    ggml_tensor *  src     = dst->src[0];
    acl_tensor_ptr acl_src = ggml_cann_create_tensor(src);
    acl_tensor_ptr acl_dst = ggml_cann_create_tensor(dst);

    // padding: value in the array means how much distance will be padding.
    // the position of elements in the array means which dirction to padding,
    // each position means: [dim0.front, dim0.behind, dim1.front, dim1.behind,
    //                       dim2.front, dim2.behind, dim3.front, dim3.behind]
    ਅਚਲ int32_t lp0 = ggml_get_op_params_i32(dst, 0);
    ਅਚਲ int32_t rp0 = ggml_get_op_params_i32(dst, 1);
    ਅਚਲ int32_t lp1 = ggml_get_op_params_i32(dst, 2);
    ਅਚਲ int32_t rp1 = ggml_get_op_params_i32(dst, 3);
    ਅਚਲ int32_t lp2 = ggml_get_op_params_i32(dst, 4);
    ਅਚਲ int32_t rp2 = ggml_get_op_params_i32(dst, 5);
    ਅਚਲ int32_t lp3 = ggml_get_op_params_i32(dst, 6);
    ਅਚਲ int32_t rp3 = ggml_get_op_params_i32(dst, 7);

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
ਸਥਿਰ ਖਾਲੀ ggml_cann_avg_pool2d(ggml_backend_cann_context & ctx, ggml_tensor * dst) {
    ggml_tensor * src = dst->src[0];
    GGML_ASSERT(src->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    acl_tensor_ptr acl_src = ggml_cann_create_tensor(src, ਖਾਲੀ_ਸੂਚਕ, ਖਾਲੀ_ਸੂਚਕ, 0, ACL_FORMAT_NCHW);
    acl_tensor_ptr acl_dst = ggml_cann_create_tensor(dst, ਖਾਲੀ_ਸੂਚਕ, ਖਾਲੀ_ਸੂਚਕ, 0, ACL_FORMAT_NCHW);

    ਅਚਲ int32_t * opts = (ਅਚਲ int32_t *) dst->op_params;
    ਅਚਲ ਪੂਰਨ       k0   = opts[1];
    ਅਚਲ ਪੂਰਨ       k1   = opts[2];
    ਅਚਲ ਪੂਰਨ       s0   = opts[3];
    ਅਚਲ ਪੂਰਨ       s1   = opts[4];
    ਅਚਲ ਪੂਰਨ       p0   = opts[5];
    ਅਚਲ ਪੂਰਨ       p1   = opts[6];

    std::vector<int64_t> kernel_dims      = { k1, k0 };
    std::vector<int64_t> stride_dims      = { s1, s0 };
    std::vector<int64_t> padding_avg_dims = { p1, p0 };  // (padH, padW)

    acl_int_array_ptr kernel_ਮਾਪ  = ggml_cann_create_int_array(kernel_dims.ਅੰਕੜੇ(), 2);
    acl_int_array_ptr strides      = ggml_cann_create_int_array(stride_dims.ਅੰਕੜੇ(), 2);
    acl_int_array_ptr paddings_avg = ggml_cann_create_int_array(padding_avg_dims.ਅੰਕੜੇ(), 2);

    ਤਾਰਕਿਕ    ceil_ਬਹੁਲਕ         = ਝੂਠ;
    ਤਾਰਕਿਕ    ਗਿਣਤੀ_include_pad = ਸੱਚ;
    int64_t ਭਾਜਕ_override  = 0;
    int8_t  cube_math_type    = 0;
#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ ASCEND_310P
    cube_math_type = 1;
#ਜੇ_ਅੰਤ

    GGML_CANN_CALL_ACLNN_OP(ctx, AvgPool2d, acl_src.get(), kernel_ਮਾਪ.get(), strides.get(), paddings_avg.get(),
                            ceil_ਬਹੁਲਕ, ਗਿਣਤੀ_include_pad, ਭਾਜਕ_override, cube_math_type, acl_dst.get());
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
ਸਥਿਰ ਖਾਲੀ ggml_cann_max_pool2d(ggml_backend_cann_context & ctx, ggml_tensor * dst) {
    ggml_tensor * src = dst->src[0];
    GGML_ASSERT(src->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    acl_tensor_ptr acl_src = ggml_cann_create_tensor(src, ਖਾਲੀ_ਸੂਚਕ, ਖਾਲੀ_ਸੂਚਕ, 0, ACL_FORMAT_NCHW);
    acl_tensor_ptr acl_dst = ggml_cann_create_tensor(dst, ਖਾਲੀ_ਸੂਚਕ, ਖਾਲੀ_ਸੂਚਕ, 0, ACL_FORMAT_NCHW);

    ਅਚਲ int32_t * opts = (ਅਚਲ int32_t *) dst->op_params;
    ਅਚਲ ਪੂਰਨ       k0   = opts[1];
    ਅਚਲ ਪੂਰਨ       k1   = opts[2];
    ਅਚਲ ਪੂਰਨ       s0   = opts[3];
    ਅਚਲ ਪੂਰਨ       s1   = opts[4];
    ਅਚਲ ਪੂਰਨ       p0   = opts[5];
    ਅਚਲ ਪੂਰਨ       p1   = opts[6];

    int64_t ਆਰਜ਼ੀ_ne[] = { src->ne[0] + p0 * 2, src->ne[1] + p1 * 2, src->ne[2], src->ne[3] };
    size_t  ਆਰਜ਼ੀ_nb[GGML_MAX_DIMS];

    ਆਰਜ਼ੀ_nb[0] = ggml_element_ਮਾਪ(src);
    ਲਈ (ਪੂਰਨ i = 1; i < GGML_MAX_DIMS; i++) {
        ਆਰਜ਼ੀ_nb[i] = ਆਰਜ਼ੀ_nb[i - 1] * ਆਰਜ਼ੀ_ne[i - 1];
    }

    ggml_cann_pool_alloc ਆਰਜ਼ੀ_ਬਫਰ_allocator(ctx.pool(), ggml_nbytes(src) + p0 * 2 + p1 * 2 * src->nb[1]);
    ਖਾਲੀ *               ਬਫਰ = ਆਰਜ਼ੀ_ਬਫਰ_allocator.get();
    acl_tensor_ptr tmp_tensor   = ggml_cann_create_tensor(ਬਫਰ, ACL_FLOAT, ggml_element_ਮਾਪ(src), ਆਰਜ਼ੀ_ne, ਆਰਜ਼ੀ_nb,
                                                          GGML_MAX_DIMS, ACL_FORMAT_NCHW);

    // pad: see padding in ggml_cann_pad()
    int64_t paddings[] = { p0, p0, p1, p1, 0, 0, 0, 0 };
    ਦਸ਼ਮਲਵ   value      = -FLT_MAX;
    aclnn_pad(ctx, acl_src.get(), tmp_tensor.get(), paddings, value);

    // max_pool
    std::vector<int64_t> kernel_dims      = { k1, k0 };
    std::vector<int64_t> stride_dims      = { s1, s0 };
    // padding_max_dims: [dim0_start, dim0_end, dim1_start, dim1_end]
    std::vector<int64_t> padding_max_dims = { 0, 0, 0, 0 };
    std::vector<int64_t> dilation_ਮਾਪ    = { 1, 1 };
    acl_int_array_ptr    kernel_ਮਾਪ      = ggml_cann_create_int_array(kernel_dims.ਅੰਕੜੇ(), 2);
    acl_int_array_ptr    strides          = ggml_cann_create_int_array(stride_dims.ਅੰਕੜੇ(), 2);
    acl_int_array_ptr    paddings_max     = ggml_cann_create_int_array(padding_max_dims.ਅੰਕੜੇ(), 4);
    acl_int_array_ptr    dilations        = ggml_cann_create_int_array(dilation_ਮਾਪ.ਅੰਕੜੇ(), 2);

    ਤਾਰਕਿਕ    ceil_ਬਹੁਲਕ = ਝੂਠ;
    int64_t auto_pads = 0;
    GGML_CANN_CALL_ACLNN_OP(ctx, MaxPool, tmp_tensor.get(), kernel_ਮਾਪ.get(), strides.get(), auto_pads,
                            paddings_max.get(), dilations.get(), ceil_ਬਹੁਲਕ, acl_dst.get());
}

ਖਾਲੀ ggml_cann_pool2d(ggml_backend_cann_context & ctx, ggml_tensor * dst) {
    ਅਚਲ int32_t *   opts = (ਅਚਲ int32_t *) dst->op_params;
    ਗਿਣਤੀ ggml_op_pool op   = static_cast<ggml_op_pool>(opts[0]);
    ਚੋਣ (op) {
        ਹਾਲਤ GGML_OP_POOL_AVG:
            ggml_cann_avg_pool2d(ctx, dst);
            ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_POOL_MAX:
            ggml_cann_max_pool2d(ctx, dst);
            ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_POOL_ਗਿਣਤੀ:
            GGML_ABORT("fatal error");
            ਰੋਕੋ;
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
ਸਥਿਰ ਖਾਲੀ cann_copy(ggml_backend_cann_context & ctx, aclTensor * acl_src, aclTensor * acl_dst) {
    GGML_CANN_CALL_ACLNN_OP(ctx, InplaceCopy, acl_dst, acl_src);
}

ਖਾਲੀ ggml_cann_dup(ggml_backend_cann_context & ctx, ggml_tensor * dst) {
    ggml_tensor * src0 = dst->src[0];

    ਜੇ (ggml_are_same_shape(src0, dst)) {
        acl_tensor_ptr acl_src = ggml_cann_create_tensor(src0);
        acl_tensor_ptr acl_dst = ggml_cann_create_tensor(dst);
        ਜੇ (dst->type == src0->type) {
            cann_copy(ctx, acl_src.get(), acl_dst.get());
        } ਨਹੀਂਤਾਂ {
            aclnn_cast(ctx, acl_src.get(), acl_dst.get(), ggml_cann_type_mapping(dst->type));
        }
    } ਨਹੀਂਤਾਂ {
        ਖਾਲੀ *               src_trans_ਬਫਰ = src0->ਅੰਕੜੇ;
        ggml_cann_pool_alloc src_ਬਫਰ_allocator;
        ਜੇ (!ggml_is_contiguous(src0)) {
            acl_tensor_ptr acl_src = ggml_cann_create_tensor(src0);
            src_ਬਫਰ_allocator.alloc(ctx.pool(), ggml_nelements(src0) * ggml_type_ਮਾਪ(src0->type));
            src_trans_ਬਫਰ = src_ਬਫਰ_allocator.get();
            size_t src_trans_nb[GGML_MAX_DIMS];
            src_trans_nb[0] = ggml_type_ਮਾਪ(src0->type);
            ਲਈ (ਪੂਰਨ i = 1; i < GGML_MAX_DIMS; i++) {
                src_trans_nb[i] = src_trans_nb[i - 1] * src0->ne[i - 1];
            }
            acl_tensor_ptr src_trans_tensor =
                ggml_cann_create_tensor(src_trans_ਬਫਰ, ggml_cann_type_mapping(src0->type),
                                        ggml_type_ਮਾਪ(src0->type), src0->ne, src_trans_nb, GGML_MAX_DIMS);
            cann_copy(ctx, acl_src.get(), src_trans_tensor.get());
        }

        size_t src_reshape_nb[GGML_MAX_DIMS];
        src_reshape_nb[0] = ggml_type_ਮਾਪ(src0->type);
        ਲਈ (ਪੂਰਨ i = 1; i < GGML_MAX_DIMS; i++) {
            src_reshape_nb[i] = src_reshape_nb[i - 1] * dst->ne[i - 1];
        }

        acl_tensor_ptr trans_acl_src =
            ggml_cann_create_tensor(src_trans_ਬਫਰ, ggml_cann_type_mapping(src0->type), ggml_type_ਮਾਪ(src0->type),
                                    dst->ne, src_reshape_nb, GGML_MAX_DIMS, ACL_FORMAT_ND);
        acl_tensor_ptr acl_dst = ggml_cann_create_tensor(dst);

        ਜੇ (dst->type == src0->type) {
            cann_copy(ctx, trans_acl_src.get(), acl_dst.get());
        } ਨਹੀਂਤਾਂ {
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
ਸਥਿਰ acl_tensor_ptr aclnn_zero(ggml_backend_cann_context & ctx,
                                 ਖਾਲੀ *                      ਬਫਰ,
                                 size_t                      n_ਬਾਈਟਾਂ,
                                 int64_t *                   ne,
                                 int64_t                     dims,
                                 acl_ਅੰਕੜੇ_Type                 type,
                                 size_t                      type_ਮਾਪ) {
    size_t nb[GGML_MAX_DIMS];
    nb[0] = type_ਮਾਪ;
    ਲਈ (ਪੂਰਨ i = 1; i < dims; i++) {
        nb[i] = nb[i - 1] * ne[i - 1];
    }

    acl_tensor_ptr zero = ggml_cann_create_tensor(ਬਫਰ, type, type_ਮਾਪ, ne, nb, dims);
    GGML_CANN_CALL_ACLNN_OP(ctx, InplaceZero, zero.get());
    ਵਾਪਸ zero;
    GGML_ਅਣਵਰਤਿਆ(n_ਬਾਈਟਾਂ);
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
ਸਥਿਰ acl_tensor_ptr aclnn_ਮੁੱਲਾਂ(ggml_backend_cann_context & ctx,
                                   ਖਾਲੀ *                      ਬਫਰ,
                                   size_t                      n_ਬਾਈਟਾਂ,
                                   int64_t *                   ne,
                                   int64_t                     dims,
                                   acl_ਅੰਕੜੇ_Type                 type,
                                   size_t                      type_ਮਾਪ,
                                   ਦਸ਼ਮਲਵ                       value = 1.0f) {
    acl_tensor_ptr acl_tensor = aclnn_zero(ctx, ਬਫਰ, n_ਬਾਈਟਾਂ, ne, dims, type, type_ਮਾਪ);
    ਦਸ਼ਮਲਵ          alpha_host = 1.0f;
    acl_scalar_ptr alpha      = ggml_cann_create_scalar(&alpha_host, acl_ਅੰਕੜੇ_Type::ACL_FLOAT);
    acl_scalar_ptr other      = ggml_cann_create_scalar(&value, acl_ਅੰਕੜੇ_Type::ACL_FLOAT);
    GGML_CANN_CALL_ACLNN_OP(ctx, InplaceAdds, acl_tensor.get(), other.get(), alpha.get());
    ਵਾਪਸ acl_tensor;
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
ਸਥਿਰ ਖਾਲੀ aclnn_fill_scalar(ggml_backend_cann_context & ctx, ਦਸ਼ਮਲਵ scalar, aclTensor * acl_dst) {
    acl_scalar_ptr acl_scalar = ggml_cann_create_scalar(&scalar, acl_ਅੰਕੜੇ_Type::ACL_FLOAT);
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
ਸਥਿਰ acl_tensor_ptr get_cache_acl_tensor(ggml_backend_cann_context & ctx,
                                           ਖਾਲੀ **                     ਬਫਰ,
                                           int64_t &                   cache_element,
                                           int64_t *                   ne,
                                           size_t *                    nb,
                                           ggml_type                   dtype,
                                           int64_t                     dims,
                                           ਦਸ਼ਮਲਵ                       value) {
    // Calculate total number of elements
    int64_t n_element = 1;
    ਲਈ (ਪੂਰਨ i = 0; i < dims; i++) {
        n_element *= ne[i];
    }
    size_t size = n_element * ggml_type_ਮਾਪ(dtype);

    // Allocate or expand cache if needed
    ਜੇ (cache_element < n_element) {
        ਜੇ (*ਬਫਰ != ਖਾਲੀ_ਸੂਚਕ) {
            aclrtFree(*ਬਫਰ);
            *ਬਫਰ = ਖਾਲੀ_ਸੂਚਕ;
        }

        ACL_CHECK(aclrtMalloc(ਬਫਰ, size, ACL_MEM_MALLOC_HUGE_ਪਹਿਲਾ));
        cache_element = n_element;

        // Initialize cache
        int64_t        pool_ne[1] = { n_element };
        size_t         pool_nb[1] = { ggml_type_ਮਾਪ(dtype) };
        acl_tensor_ptr acl_ਮੁੱਲ =
            ggml_cann_create_tensor(*ਬਫਰ, ggml_cann_type_mapping(dtype), ggml_type_ਮਾਪ(dtype), pool_ne, pool_nb, 1);
        aclnn_fill_scalar(ctx, value, acl_ਮੁੱਲ.get());
    }

    ਵਾਪਸ ggml_cann_create_tensor(*ਬਫਰ, ggml_cann_type_mapping(dtype), ggml_type_ਮਾਪ(dtype), ne, nb, dims);
}

ਖਾਲੀ ggml_cann_rms_norm(ggml_backend_cann_context & ctx, ggml_tensor * dst) {
    ggml_tensor * src = dst->src[0];

    acl_tensor_ptr acl_src = ggml_cann_create_tensor(src);
    acl_tensor_ptr acl_dst = ggml_cann_create_tensor(dst);

    ਦਸ਼ਮਲਵ eps;
    ਸਮ੍ਰਿਤੀ_ਨਕਲ(&eps, dst->op_params, ਆਕਾਰ(ਦਸ਼ਮਲਵ));

    // build gamma.
    size_t acl_gamma_nb[GGML_MAX_DIMS];
    // gamma's type is the same with dst.
    acl_gamma_nb[0] = ggml_type_ਮਾਪ(dst->type);
    ਲਈ (ਪੂਰਨ i = 1; i < GGML_MAX_DIMS; i++) {
        acl_gamma_nb[i] = acl_gamma_nb[i - 1] * src->ne[i - 1];
    }
    acl_tensor_ptr acl_gamma = get_cache_acl_tensor(
        ctx, &ctx.rms_norm_one_tensor_cache.cache, ctx.rms_norm_one_tensor_cache.size, src->ne, acl_gamma_nb, dst->type,
        1,    // dims
        1.0f  //ਮੁੱਲ
    );

    // build rstd.
    int64_t acl_rstd_ne[] = { src->ne[1], src->ne[2], src->ne[3] };
    size_t  acl_rstd_nb[GGML_MAX_DIMS - 1];
    // rstd will always be F32.
    acl_rstd_nb[0] = ਆਕਾਰ(ਦਸ਼ਮਲਵ);
    ਲਈ (ਪੂਰਨ i = 1; i < GGML_MAX_DIMS - 1; i++) {
        acl_rstd_nb[i] = acl_rstd_nb[i - 1] * acl_rstd_ne[i - 1];
    }
    acl_tensor_ptr acl_rstd =
        get_cache_acl_tensor(ctx, &ctx.rms_norm_zero_tensor_cache.cache, ctx.rms_norm_zero_tensor_cache.size,
                             acl_rstd_ne, acl_rstd_nb, GGML_TYPE_F32, GGML_MAX_DIMS - 1,
                             0.0f  //ਮੁੱਲ
        );

    GGML_CANN_CALL_ACLNN_OP(ctx, RmsNorm, acl_src.get(), acl_gamma.get(), eps, acl_dst.get(), acl_rstd.get());
}

// TODO: performace is low.
ਖਾਲੀ ggml_cann_diag_ਨਕਾਬ(ggml_backend_cann_context & ctx, ggml_tensor * dst, ਦਸ਼ਮਲਵ value) {
    ggml_tensor * src = dst->src[0];

    acl_tensor_ptr acl_src = ggml_cann_create_tensor(src);
    acl_tensor_ptr acl_dst = ggml_cann_create_tensor(dst);

    ਅਚਲ ਪੂਰਨ n_past = ((int32_t *) dst->op_params)[0];

    ggml_cann_pool_alloc one_tensor_allocator(ctx.pool(), ggml_nbytes(src));
    ਖਾਲੀ *               ਬਫਰ = one_tensor_allocator.get();

    acl_tensor_ptr ਨਕਾਬ_tensor = ggml_cann_create_tensor(ਬਫਰ, ggml_cann_type_mapping(src->type),
                                                         ggml_type_ਮਾਪ(src->type), src->ne, src->nb, GGML_MAX_DIMS);

    aclnn_fill_scalar(ctx, value, ਨਕਾਬ_tensor.get());

    ਦਸ਼ਮਲਵ          alpha_ਮੁੱਲ = 1.0f;
    acl_scalar_ptr alpha      = ggml_cann_create_scalar(&alpha_ਮੁੱਲ, acl_ਅੰਕੜੇ_Type::ACL_FLOAT);

    GGML_CANN_CALL_ACLNN_OP(ctx, InplaceTriu, ਨਕਾਬ_tensor.get(), n_past + 1);
    GGML_CANN_CALL_ACLNN_OP(ctx, Tril, acl_src.get(), n_past + 1, acl_dst.get());
    GGML_CANN_CALL_ACLNN_OP(ctx, Inplace_ਜੋੜੋ, acl_dst.get(), ਨਕਾਬ_tensor.get(), alpha.get());
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
ਸਥਿਰ ਖਾਲੀ aclnn_ਕ੍ਰਮ_ਬਦਲੋ(ggml_backend_cann_context & ctx,
                          aclTensor *                 acl_src,
                          aclTensor *                 acl_dst,
                          int64_t *                   new_dim,
                          uint64_t                    dims) {
    acl_int_array_ptr acl_dims = ggml_cann_create_int_array(new_dim, dims);
    GGML_CANN_CALL_ACLNN_OP(ctx, ਕ੍ਰਮ_ਬਦਲੋ, acl_src, acl_dims.get(), acl_dst);
}

ਸਥਿਰ ਖਾਲੀ ggml_cann_im2col_2d_post_process(ggml_backend_cann_context & ctx,
                                             ggml_tensor *               dst,
                                             ggml_tensor *               src1,
                                             aclTensor *                 tmp_cast_tensor,
                                             aclTensor *                 tmp_im2col_tensor) {
    // Permute: [N, IC * KH * KW, OW * OH] -> [N, OW * OH, IC * KH * KW]
    int64_t        dst_ne[] = { dst->ne[0], dst->ne[1] * dst->ne[2], dst->ne[3] };
    size_t         dst_nb[] = { dst->nb[0], dst->nb[1], dst->nb[3] };
    acl_tensor_ptr acl_dst  = ggml_cann_create_tensor(dst, dst_ne, dst_nb, GGML_MAX_DIMS - 1);

    int64_t ਕ੍ਰਮ_ਬਦਲੋ_dim[] = { 0, 2, 1 };
    ਜੇ (src1->type != dst->type) {
        aclnn_ਕ੍ਰਮ_ਬਦਲੋ(ctx, tmp_cast_tensor, acl_dst.get(), ਕ੍ਰਮ_ਬਦਲੋ_dim, 3);
    } ਨਹੀਂਤਾਂ {
        aclnn_ਕ੍ਰਮ_ਬਦਲੋ(ctx, tmp_im2col_tensor, acl_dst.get(), ਕ੍ਰਮ_ਬਦਲੋ_dim, 3);
    }
}

ਸਥਿਰ ਖਾਲੀ ggml_cann_im2col_1d_post_process(ggml_backend_cann_context &  ctx,
                                             ggml_tensor *                dst,
                                             ggml_tensor *                src1,
                                             aclTensor *                  tmp_cast_tensor,
                                             aclTensor *                  tmp_im2col_tensor,
                                             ਅਚਲ std::vector<int64_t> & im2col_op_params) {
    // get params
    ਅਚਲ int64_t KH             = im2col_op_params[0];
    ਅਚਲ int64_t KW             = im2col_op_params[1];
    ਅਚਲ int64_t IW             = im2col_op_params[2];
    ਅਚਲ int64_t IC             = im2col_op_params[3];
    ਅਚਲ int64_t N              = im2col_op_params[4];
    ਅਚਲ int64_t OH             = im2col_op_params[5];
    ਅਚਲ int64_t OW             = im2col_op_params[6];
    ਅਚਲ int64_t s0             = im2col_op_params[7];
    ਅਚਲ int64_t p0             = im2col_op_params[8];
    ਅਚਲ int64_t d0             = im2col_op_params[9];
    ਅਚਲ int64_t n_ਬਾਈਟਾਂ_ਗੁਣਕ = im2col_op_params[10];

    // Permute: [N, IC * KH * KW, OW * OH] ->
    // [N, OW * OH * n_bytes_factor, IC * KH * KW]
    ggml_cann_pool_alloc tmp_ਕ੍ਰਮ_ਬਦਲੋ_allocator(ctx.pool());
    tmp_ਕ੍ਰਮ_ਬਦਲੋ_allocator.alloc(ggml_nbytes(dst) * n_ਬਾਈਟਾਂ_ਗੁਣਕ);
    ਖਾਲੀ * tmp_ਕ੍ਰਮ_ਬਦਲੋ_ਬਫਰ = tmp_ਕ੍ਰਮ_ਬਦਲੋ_allocator.get();

    int64_t tmp_ਕ੍ਰਮ_ਬਦਲੋ_ne[] = { IC * KH * KW, OW * OH * n_ਬਾਈਟਾਂ_ਗੁਣਕ, N };
    size_t  tmp_ਕ੍ਰਮ_ਬਦਲੋ_nb[GGML_MAX_DIMS - 1];
    tmp_ਕ੍ਰਮ_ਬਦਲੋ_nb[0] = ggml_type_ਮਾਪ(dst->type);
    ਲਈ (ਪੂਰਨ i = 1; i < GGML_MAX_DIMS - 1; i++) {
        tmp_ਕ੍ਰਮ_ਬਦਲੋ_nb[i] = tmp_ਕ੍ਰਮ_ਬਦਲੋ_nb[i - 1] * tmp_ਕ੍ਰਮ_ਬਦਲੋ_ne[i - 1];
    }

    acl_tensor_ptr tmp_ਕ੍ਰਮ_ਬਦਲੋ_tensor =
        ggml_cann_create_tensor(tmp_ਕ੍ਰਮ_ਬਦਲੋ_ਬਫਰ, ggml_cann_type_mapping(dst->type), ggml_type_ਮਾਪ(dst->type),
                                tmp_ਕ੍ਰਮ_ਬਦਲੋ_ne, tmp_ਕ੍ਰਮ_ਬਦਲੋ_nb, GGML_MAX_DIMS - 1, ACL_FORMAT_ND);

    int64_t ਕ੍ਰਮ_ਬਦਲੋ_dim[] = { 0, 2, 1 };
    ਜੇ (src1->type != dst->type) {
        aclnn_ਕ੍ਰਮ_ਬਦਲੋ(ctx, tmp_cast_tensor, tmp_ਕ੍ਰਮ_ਬਦਲੋ_tensor.get(), ਕ੍ਰਮ_ਬਦਲੋ_dim, 3);
    } ਨਹੀਂਤਾਂ {
        aclnn_ਕ੍ਰਮ_ਬਦਲੋ(ctx, tmp_im2col_tensor, tmp_ਕ੍ਰਮ_ਬਦਲੋ_tensor.get(), ਕ੍ਰਮ_ਬਦਲੋ_dim, 3);
    }

    // number of times the kernel moves in W dimension
    ਅਚਲ ਪੂਰਨ n_ਕਦਮ_w = (IW + 2 * p0 - d0 * (KW - 1) - 1) / s0 + 1;
    size_t    ਵਿੱਥ;
    ਖਾਲੀ *    cur_dst_ਬਫਰ = dst->ਅੰਕੜੇ, *cur_ਕ੍ਰਮ_ਬਦਲੋ_ਬਫਰ = tmp_ਕ੍ਰਮ_ਬਦਲੋ_ਬਫਰ;

    // memory copy with offset to restore 1D im2col from 2d
    ਜੇ (IC > 1) {
        ਵਿੱਥ          = IC * KH * KW * n_ਕਦਮ_w * ggml_type_ਮਾਪ(dst->type);
        size_t cpy_ਮਾਪ = KH * KW * ggml_type_ਮਾਪ(dst->type);

        ਲਈ (ਪੂਰਨ c = 0; c < IC; c++) {
            cur_ਕ੍ਰਮ_ਬਦਲੋ_ਬਫਰ = (ਅੱਖਰ *) tmp_ਕ੍ਰਮ_ਬਦਲੋ_ਬਫਰ + ਵਿੱਥ + KH * KW * c * ggml_type_ਮਾਪ(dst->type);
            cur_dst_ਬਫਰ     = (ਅੱਖਰ *) dst->ਅੰਕੜੇ + c * KH * KW * n_ਕਦਮ_w * ggml_type_ਮਾਪ(dst->type);

            ਲਈ (ਪੂਰਨ i = 0; i < n_ਕਦਮ_w; i++) {
                ACL_CHECK(aclrtMemcpyAsync(cur_dst_ਬਫਰ, cpy_ਮਾਪ, cur_ਕ੍ਰਮ_ਬਦਲੋ_ਬਫਰ, cpy_ਮਾਪ,
                                           ACL_MEMCPY_DEVICE_ਮੰਜ਼ਿਲ_DEVICE, ctx.stream()));
                cur_dst_ਬਫਰ     = (ਅੱਖਰ *) cur_dst_ਬਫਰ + KH * KW * ggml_type_ਮਾਪ(dst->type);
                cur_ਕ੍ਰਮ_ਬਦਲੋ_ਬਫਰ = (ਅੱਖਰ *) cur_ਕ੍ਰਮ_ਬਦਲੋ_ਬਫਰ + KH * KW * IC * ggml_type_ਮਾਪ(dst->type);
            }
        }
    } ਨਹੀਂਤਾਂ {
        ਵਿੱਥ = KH * KW * n_ਕਦਮ_w * ggml_type_ਮਾਪ(dst->type);  // equal to ggml_nbytes(dst)
        ACL_CHECK(aclrtMemcpyAsync(dst->ਅੰਕੜੇ, ਵਿੱਥ, (ਅੱਖਰ *) tmp_ਕ੍ਰਮ_ਬਦਲੋ_ਬਫਰ + ਵਿੱਥ, ਵਿੱਥ,
                                   ACL_MEMCPY_DEVICE_ਮੰਜ਼ਿਲ_DEVICE, ctx.stream()));
    }
}

ਖਾਲੀ ggml_cann_im2col(ggml_backend_cann_context & ctx, ggml_tensor * dst) {
    ggml_tensor * src0 = dst->src[0];  // kernel
    ggml_tensor * src1 = dst->src[1];  //ਨਿਵੇਸ਼

    GGML_TENSOR_ਦੁਆਧਾਰੀ_OP_LOCALS;

    // aclnnIm2col only works on 2D. set s1, p1, d1 to 1 to perform 2D
    // im2col and do post-processing to restore it to 1D.
    ਅਚਲ ਤਾਰਕਿਕ    is_2D = ((ਅਚਲ int32_t *) (dst->op_params))[6] == 1;
    ਅਚਲ int32_t s0    = ((ਅਚਲ int32_t *) (dst->op_params))[0];
    ਅਚਲ int32_t s1    = is_2D ? ((ਅਚਲ int32_t *) (dst->op_params))[1] : 1;
    ਅਚਲ int32_t p0    = ((ਅਚਲ int32_t *) (dst->op_params))[2];
    ਅਚਲ int32_t p1    = is_2D ? ((ਅਚਲ int32_t *) (dst->op_params))[3] : 1;
    ਅਚਲ int32_t d0    = ((ਅਚਲ int32_t *) (dst->op_params))[4];
    ਅਚਲ int32_t d1    = is_2D ? ((ਅਚਲ int32_t *) (dst->op_params))[5] : 1;

    ਅਚਲ int64_t N  = ne13;
    ਅਚਲ int64_t IC = ne12;
    ਅਚਲ int64_t KH = ne01;
    ਅਚਲ int64_t KW = ne00;
    ਅਚਲ int64_t IW = ne10;

    ਅਚਲ int64_t OH = is_2D ? ne2 : 1;
    ਅਚਲ int64_t OW = ne1;

    // memory allocated increased to 3x when is_2D == false
    ਅਚਲ int64_t n_ਬਾਈਟਾਂ_ਗੁਣਕ = is_2D ? 1 : 3;

    // im2col: [N,C,H,W] -> [N, IC * KH * KW, OW * OH * n_bytes_factor]
    acl_tensor_ptr acl_src1        = ggml_cann_create_tensor(src1);
    int64_t        tmp_im2col_ne[] = { OW * OH * n_ਬਾਈਟਾਂ_ਗੁਣਕ, IC * KH * KW, N };
    size_t         tmp_im2col_nb[GGML_MAX_DIMS - 1];

    tmp_im2col_nb[0] = ggml_type_ਮਾਪ(src1->type);
    ਲਈ (ਪੂਰਨ i = 1; i < GGML_MAX_DIMS - 1; i++) {
        tmp_im2col_nb[i] = tmp_im2col_nb[i - 1] * tmp_im2col_ne[i - 1];
    }

    // Calculate im2col.
    // If dst is f16, tmp_buffer is f32, we need alloc src.typesize *
    // dst.elemcount.
    ggml_cann_pool_alloc im2col_allocator(ctx.pool(), ggml_nelements(dst) * ggml_element_ਮਾਪ(src1) * n_ਬਾਈਟਾਂ_ਗੁਣਕ);
    ਖਾਲੀ *               tmp_im2col_ਬਫਰ = im2col_allocator.get();

    acl_tensor_ptr tmp_im2col_tensor =
        ggml_cann_create_tensor(tmp_im2col_ਬਫਰ, ggml_cann_type_mapping(src1->type), ggml_type_ਮਾਪ(src1->type),
                                tmp_im2col_ne, tmp_im2col_nb, GGML_MAX_DIMS - 1, ACL_FORMAT_ND);

    std::vector<int64_t> kernel_dims   = { KH, KW };
    std::vector<int64_t> dilation_ਮਾਪ = { d1, d0 };
    std::vector<int64_t> padding_dims  = { p1, p0 };
    std::vector<int64_t> stride_dims   = { s1, s0 };
    acl_int_array_ptr    kernel_ਮਾਪ   = ggml_cann_create_int_array(kernel_dims.ਅੰਕੜੇ(), 2);
    acl_int_array_ptr    dilations     = ggml_cann_create_int_array(dilation_ਮਾਪ.ਅੰਕੜੇ(), 2);
    acl_int_array_ptr    paddings      = ggml_cann_create_int_array(padding_dims.ਅੰਕੜੇ(), 2);
    acl_int_array_ptr    strides       = ggml_cann_create_int_array(stride_dims.ਅੰਕੜੇ(), 2);
    GGML_CANN_CALL_ACLNN_OP(ctx, Im2col, acl_src1.get(), kernel_ਮਾਪ.get(), dilations.get(), paddings.get(),
                            strides.get(), tmp_im2col_tensor.get());

    // Cast if dst is f16.
    acl_tensor_ptr       tmp_cast_tensor;
    ggml_cann_pool_alloc tmp_cast_allocator(ctx.pool());
    ਖਾਲੀ *               tmp_cast_ਬਫਰ = ਖਾਲੀ_ਸੂਚਕ;
    ਜੇ (src1->type != dst->type) {
        tmp_cast_allocator.alloc(ggml_nbytes(dst) * n_ਬਾਈਟਾਂ_ਗੁਣਕ);
        tmp_cast_ਬਫਰ = tmp_cast_allocator.get();
        size_t ਆਰਜ਼ੀ_cast_nb[GGML_MAX_DIMS - 1];
        ਆਰਜ਼ੀ_cast_nb[0] = ggml_type_ਮਾਪ(dst->type);
        ਲਈ (ਪੂਰਨ i = 1; i < GGML_MAX_DIMS - 1; i++) {
            ਆਰਜ਼ੀ_cast_nb[i] = ਆਰਜ਼ੀ_cast_nb[i - 1] * tmp_im2col_ne[i - 1];
        }

        tmp_cast_tensor =
            ggml_cann_create_tensor(tmp_cast_ਬਫਰ, ggml_cann_type_mapping(dst->type), ggml_type_ਮਾਪ(dst->type),
                                    tmp_im2col_ne, ਆਰਜ਼ੀ_cast_nb, GGML_MAX_DIMS - 1, ACL_FORMAT_ND);
        aclnn_cast(ctx, tmp_im2col_tensor.get(), tmp_cast_tensor.get(), ggml_cann_type_mapping(dst->type));
    }

    // post-processing
    ਜੇ (is_2D) {
        ggml_cann_im2col_2d_post_process(ctx, dst, src1, tmp_cast_tensor.get(), tmp_im2col_tensor.get());
    } ਨਹੀਂਤਾਂ {
        std::vector<int64_t> im2col_op_params = { KH, KW, IW, IC, N, OH, OW, s0, p0, d0, n_ਬਾਈਟਾਂ_ਗੁਣਕ };
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
ਸਥਿਰ ਖਾਲੀ aclnn_exp(ggml_backend_cann_context & ctx, aclTensor * acl_src) {
    GGML_CANN_CALL_ACLNN_OP(ctx, InplaceExp, acl_src);
}

ਖਾਲੀ aclnn_cos(ggml_backend_cann_context & ctx, aclTensor * acl_src, aclTensor * acl_dst) {
    ਜੇ (acl_dst == ਖਾਲੀ_ਸੂਚਕ) {
        GGML_CANN_CALL_ACLNN_OP(ctx, InplaceCos, acl_src);
    } ਨਹੀਂਤਾਂ {
        GGML_CANN_CALL_ACLNN_OP(ctx, Cos, acl_src, acl_dst);
    }
}

ਖਾਲੀ aclnn_sin(ggml_backend_cann_context & ctx, aclTensor * acl_src, aclTensor * acl_dst) {
    ਜੇ (acl_dst == ਖਾਲੀ_ਸੂਚਕ) {
        GGML_CANN_CALL_ACLNN_OP(ctx, InplaceSin, acl_src);
    } ਨਹੀਂਤਾਂ {
        GGML_CANN_CALL_ACLNN_OP(ctx, Sin, acl_src, acl_dst);
    }
}

ਖਾਲੀ ggml_cann_timestep_embedding(ggml_backend_cann_context & ctx, ggml_tensor * dst) {
    ਅਚਲ ggml_tensor * src = dst->src[0];

    GGML_ASSERT(src->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    ਅਚਲ ਪੂਰਨ dim        = dst->op_params[0];
    ਅਚਲ ਪੂਰਨ max_period = dst->op_params[1];
    ਪੂਰਨ       half       = dim / 2;

    acl_tensor_ptr acl_src = ggml_cann_create_tensor(src);

    // arange: [0, ..., half)
    ਦਸ਼ਮਲਵ   start             = 0;
    ਦਸ਼ਮਲਵ   stop              = half;
    ਦਸ਼ਮਲਵ   ਕਦਮ              = 1;
    int64_t n_elements_arange = half;
    int64_t tmp_arange_ne[]   = { half };
    size_t  tmp_arange_nb[]   = { ਆਕਾਰ(dst->type) };

    ggml_cann_pool_alloc arange_allocator(ctx.pool(), half * ਆਕਾਰ(dst->type));
    ਖਾਲੀ *               tmp_arange_ਬਫਰ = arange_allocator.get();
    acl_tensor_ptr       tmp_arange_tensor =
        ggml_cann_create_tensor(tmp_arange_ਬਫਰ, ggml_cann_type_mapping(dst->type), ggml_type_ਮਾਪ(dst->type),
                                tmp_arange_ne, tmp_arange_nb, GGML_MAX_DIMS - 3, ACL_FORMAT_ND);

    aclnn_arange(ctx, tmp_arange_tensor.get(), start, stop, ਕਦਮ, n_elements_arange);

    // freq
    ਦਸ਼ਮਲਵ freq_param = -logf(max_period) / half;
    ਤਾਰਕਿਕ  inplace    = ਸੱਚ;
    aclnn_muls(ctx, tmp_arange_tensor.get(), freq_param, ਖਾਲੀ_ਸੂਚਕ, inplace);
    aclnn_exp(ctx, tmp_arange_tensor.get());

    // permute: src [0,1,2,3]->[0,1,3,2]
    int64_t tmp_ਕ੍ਰਮ_ਬਦਲੋ_ne[] = { src->ne[1], src->ne[0], src->ne[2], src->ne[3] };
    size_t  tmp_ਕ੍ਰਮ_ਬਦਲੋ_nb[GGML_MAX_DIMS];
    tmp_ਕ੍ਰਮ_ਬਦਲੋ_nb[0] = ggml_type_ਮਾਪ(src->type);
    ਲਈ (ਪੂਰਨ i = 1; i < GGML_MAX_DIMS; i++) {
        tmp_ਕ੍ਰਮ_ਬਦਲੋ_nb[i] = tmp_ਕ੍ਰਮ_ਬਦਲੋ_nb[i - 1] * tmp_ਕ੍ਰਮ_ਬਦਲੋ_ne[i - 1];
    }

    ggml_cann_pool_alloc ਕ੍ਰਮ_ਬਦਲੋ_allocator(ctx.pool(), ggml_nbytes(src));
    ਖਾਲੀ *               tmp_ਕ੍ਰਮ_ਬਦਲੋ_ਬਫਰ = ਕ੍ਰਮ_ਬਦਲੋ_allocator.get();
    acl_tensor_ptr       tmp_ਕ੍ਰਮ_ਬਦਲੋ_tensor =
        ggml_cann_create_tensor(tmp_ਕ੍ਰਮ_ਬਦਲੋ_ਬਫਰ, ggml_cann_type_mapping(src->type), ggml_type_ਮਾਪ(src->type),
                                tmp_ਕ੍ਰਮ_ਬਦਲੋ_ne, tmp_ਕ੍ਰਮ_ਬਦਲੋ_nb, GGML_MAX_DIMS, ACL_FORMAT_ND);
    int64_t ਕ੍ਰਮ_ਬਦਲੋ_dim[] = { 0, 1, 3, 2 };
    int64_t ਅੰਕ_dims      = 4;
    aclnn_ਕ੍ਰਮ_ਬਦਲੋ(ctx, acl_src.get(), tmp_ਕ੍ਰਮ_ਬਦਲੋ_tensor.get(), ਕ੍ਰਮ_ਬਦਲੋ_dim, ਅੰਕ_dims);

    // timestep * freq
    int64_t tmp_mul_ne[] = { src->ne[1] * half, src->ne[0], src->ne[2], src->ne[3] };
    size_t  tmp_mul_nb[GGML_MAX_DIMS];
    tmp_mul_nb[0] = ggml_type_ਮਾਪ(src->type);
    ਲਈ (ਪੂਰਨ i = 1; i < GGML_MAX_DIMS; i++) {
        tmp_mul_nb[i] = tmp_mul_nb[i - 1] * tmp_mul_ne[i - 1];
    }

    ਪੂਰਨ mul_nelements = src->ne[1] * half * src->ne[0] * src->ne[2] * src->ne[3];

    ggml_cann_pool_alloc mul_allocator(ctx.pool(), mul_nelements * ggml_type_ਮਾਪ(src->type));
    ਖਾਲੀ *               tmp_mul_ਬਫਰ = mul_allocator.get();
    acl_tensor_ptr       tmp_mul_tensor =
        ggml_cann_create_tensor(tmp_mul_ਬਫਰ, ggml_cann_type_mapping(src->type), ggml_type_ਮਾਪ(src->type),
                                tmp_mul_ne, tmp_mul_nb, GGML_MAX_DIMS, ACL_FORMAT_ND);
    aclnn_mul(ctx, tmp_ਕ੍ਰਮ_ਬਦਲੋ_tensor.get(), tmp_arange_tensor.get(), tmp_mul_tensor.get());

    // cos
    ggml_cann_pool_alloc cos_allocator(ctx.pool(), mul_nelements * ggml_type_ਮਾਪ(src->type));
    ਖਾਲੀ *               tmp_cos_ਬਫਰ = cos_allocator.get();
    acl_tensor_ptr       tmp_cos_tensor =
        ggml_cann_create_tensor(tmp_cos_ਬਫਰ, ggml_cann_type_mapping(dst->type), ggml_type_ਮਾਪ(dst->type),
                                tmp_mul_ne, tmp_mul_nb, GGML_MAX_DIMS, ACL_FORMAT_ND);

    aclnn_cos(ctx, tmp_mul_tensor.get(), tmp_cos_tensor.get());

    // sin
    ggml_cann_pool_alloc sin_allocator(ctx.pool(), mul_nelements * ggml_type_ਮਾਪ(src->type));
    ਖਾਲੀ *               tmp_sin_ਬਫਰ = sin_allocator.get();
    acl_tensor_ptr       tmp_sin_tensor =
        ggml_cann_create_tensor(tmp_sin_ਬਫਰ, ggml_cann_type_mapping(dst->type), ggml_type_ਮਾਪ(dst->type),
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
ਸਥਿਰ ਖਾਲੀ aclnn_pow_tensor_tensor(ggml_backend_cann_context & ctx, aclTensor * acl_dst, aclTensor * acl_exp) {
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
ਸਥਿਰ ਖਾਲੀ aclnn_get_ਢਲਾਨ_inner(ggml_backend_cann_context & ctx,
                                  ਖਾਲੀ *                      ਢਲਾਨ_ਬਫਰ,
                                  ਦਸ਼ਮਲਵ                       m,
                                  int64_t                     size,
                                  ਦਸ਼ਮਲਵ                       start,
                                  ਦਸ਼ਮਲਵ                       stop,
                                  ਦਸ਼ਮਲਵ                       ਕਦਮ,
                                  ggml_type                   dtype) {
    acl_ਅੰਕੜੇ_Type acl_type  = ggml_cann_type_mapping(dtype);
    size_t      type_ਮਾਪ = ggml_type_ਮਾਪ(dtype);

    int64_t ne[] = { size };
    size_t  nb[] = { type_ਮਾਪ };

    ggml_cann_pool_alloc arange_allocator(ctx.pool(), size * type_ਮਾਪ);
    ਖਾਲੀ *               arange_ਬਫਰ = arange_allocator.get();

    acl_tensor_ptr arange_tensor = ggml_cann_create_tensor(arange_ਬਫਰ, acl_type, type_ਮਾਪ, ne, nb, 1);
    aclnn_arange(ctx, arange_tensor.get(), start, stop, ਕਦਮ, size);

    acl_tensor_ptr ਢਲਾਨ_tensor = ggml_cann_create_tensor(ਢਲਾਨ_ਬਫਰ, acl_type, type_ਮਾਪ, ne, nb, 1);

    acl_scalar_ptr sc = ggml_cann_create_scalar(&m, acl_ਅੰਕੜੇ_Type::ACL_FLOAT);

    GGML_CANN_CALL_ACLNN_OP(ctx, PowScalarTensor, sc.get(), arange_tensor.get(), ਢਲਾਨ_tensor.get());
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
ਸਥਿਰ ਖਾਲੀ aclnn_get_ਢਲਾਨ(ggml_backend_cann_context & ctx,
                            int64_t                     n_ਸਿਰ,
                            ਖਾਲੀ *                      ਢਲਾਨ_ਬਫਰ,
                            ਦਸ਼ਮਲਵ                       max_bias,
                            ggml_type                   dtype) {
    ਅਚਲ ਪੂਰਨ n_ਸਿਰ_log2 = 1u << (uint32_t) floor(log2(n_ਸਿਰ));

    ਦਸ਼ਮਲਵ m0 = powf(2.0f, -(max_bias) / n_ਸਿਰ_log2);
    ਦਸ਼ਮਲਵ m1 = powf(2.0f, -(max_bias / 2.0f) / n_ਸਿਰ_log2);

    // const float slope = (max_bias > 0.0f) ?
    //                          h < n_head_log2 ?
    //                              powf(m0, h + 1) :
    //                              powf(m1, 2*(h - n_head_log2) + 1) :
    //                          1.0f;
    // arange1
    ਦਸ਼ਮਲਵ start = 0 + 1;
    ਦਸ਼ਮਲਵ end   = (n_ਸਿਰ_log2 - 1) + 1;
    ਦਸ਼ਮਲਵ ਕਦਮ  = 1;
    ਦਸ਼ਮਲਵ count = n_ਸਿਰ_log2;
    // end needs to be +1 because aclnn uses a left-closed, right-open interval.
    aclnn_get_ਢਲਾਨ_inner(ctx, ਢਲਾਨ_ਬਫਰ, m0, count, start, end + 1, ਕਦਮ, dtype);
    ਜੇ (n_ਸਿਰ_log2 < n_ਸਿਰ) {
        // arange2
        start = 2 * (n_ਸਿਰ_log2 - n_ਸਿਰ_log2) + 1;
        end   = 2 * ((n_ਸਿਰ - 1) - n_ਸਿਰ_log2) + 1;
        ਕਦਮ  = 2;
        count = n_ਸਿਰ - n_ਸਿਰ_log2;
        aclnn_get_ਢਲਾਨ_inner(ctx, (ਅੱਖਰ *) ਢਲਾਨ_ਬਫਰ + n_ਸਿਰ_log2 * ggml_type_ਮਾਪ(dtype), m1, count, start, end + 1,
                              ਕਦਮ, dtype);
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
ਸਥਿਰ ਖਾਲੀ aclnn_ਜੋੜੋ_alibi(ggml_backend_cann_context & ctx,
                            ggml_tensor *               ਨਕਾਬ,
                            ggml_tensor *               dst,
                            ਖਾਲੀ *                      dst_ptr,
                            ਦਸ਼ਮਲਵ                       max_bias) {
    ਖਾਲੀ * ਢਲਾਨ_ਬਫਰ = ਖਾਲੀ_ਸੂਚਕ;
    ਖਾਲੀ * bias_ਬਫਰ  = ਖਾਲੀ_ਸੂਚਕ;

    ਜੇ (max_bias > 0.0f) {
        int64_t              n_heads = dst->ne[2];
        ggml_cann_pool_alloc ਢਲਾਨ_allocator(ctx.pool(), n_heads * ਆਕਾਰ(ਦਸ਼ਮਲਵ));
        ਢਲਾਨ_ਬਫਰ = ਢਲਾਨ_allocator.get();
        ggml_cann_pool_alloc bias_allocator(ctx.pool(), ggml_nelements(dst) * ggml_element_ਮਾਪ(dst));
        bias_ਬਫਰ = bias_allocator.get();
        aclnn_get_ਢਲਾਨ(ctx, n_heads, ਢਲਾਨ_ਬਫਰ, max_bias, GGML_TYPE_F32);
    }

    // broadcast for mask, slop and dst;
    int64_t nr2 = dst->ne[2] / ਨਕਾਬ->ne[2];
    int64_t nr3 = dst->ne[3] / ਨਕਾਬ->ne[3];

    // broadcast the mask across rows
    int64_t ਨਕਾਬ_ne[] = { ਨਕਾਬ->ne[0], dst->ne[1], ਨਕਾਬ->ne[2], 1, ਨਕਾਬ->ne[3], 1 };
    size_t  ਨਕਾਬ_nb[] = { ਨਕਾਬ_nb[0] = ਨਕਾਬ->nb[0], ਨਕਾਬ_nb[1] = ਨਕਾਬ->nb[1], ਨਕਾਬ_nb[2] = ਨਕਾਬ->nb[2],
                          ਨਕਾਬ_nb[3] = ਨਕਾਬ->nb[2], ਨਕਾਬ_nb[4] = ਨਕਾਬ->nb[3], ਨਕਾਬ_nb[5] = ਨਕਾਬ->nb[3] };

    int64_t dst_ne[] = { dst->ne[0], dst->ne[1], ਨਕਾਬ->ne[2], nr2, ਨਕਾਬ->ne[3], nr3 };
    size_t  dst_nb[] = { dst_nb[0] = dst->nb[0], dst_nb[1] = dst->nb[1], dst_nb[2] = dst->nb[2],
                         dst_nb[3] = dst->nb[2], dst_nb[4] = dst->nb[3], dst_nb[5] = dst->nb[3] };

    // slope is a 1 dim tensor, slope.ne2 == dst.ne2
    int64_t ਢਲਾਨ_ne[] = { 1, 1, ਨਕਾਬ->ne[2], nr2, 1, 1 };
    size_t  ਢਲਾਨ_nb[GGML_MAX_DIMS + 2];
    ਢਲਾਨ_nb[0] = ਆਕਾਰ(ਦਸ਼ਮਲਵ);
    ਲਈ (ਪੂਰਨ i = 1; i < GGML_MAX_DIMS + 2; i++) {
        ਢਲਾਨ_nb[i] = ਢਲਾਨ_nb[i - 1] * ਢਲਾਨ_ne[i - 1];
    }

    acl_tensor_ptr acl_ਢਲਾਨ =
        ggml_cann_create_tensor(ਢਲਾਨ_ਬਫਰ, ACL_FLOAT, ਆਕਾਰ(ਦਸ਼ਮਲਵ), ਢਲਾਨ_ne, ਢਲਾਨ_nb, GGML_MAX_DIMS + 2);
    acl_tensor_ptr acl_ਨਕਾਬ = ggml_cann_create_tensor(ਨਕਾਬ, ਨਕਾਬ_ne, ਨਕਾਬ_nb, GGML_MAX_DIMS + 2);

    // write data into dst_ptr using only the shape information of the dst tensor.
    acl_tensor_ptr acl_dst = ggml_cann_create_tensor(dst_ptr, ggml_cann_type_mapping(dst->type),
                                                     ggml_type_ਮਾਪ(dst->type), dst_ne, dst_nb, GGML_MAX_DIMS + 2);

    ਜੇ (max_bias > 0.0f) {
        int64_t bias_ne[] = { ਨਕਾਬ->ne[0], dst->ne[1], ਨਕਾਬ->ne[2], nr2, ਨਕਾਬ->ne[3], 1 };
        size_t  bias_nb[GGML_MAX_DIMS + 2];
        bias_nb[0] = ਆਕਾਰ(ਦਸ਼ਮਲਵ);
        ਲਈ (ਪੂਰਨ i = 1; i < GGML_MAX_DIMS + 2; i++) {
            bias_nb[i] = bias_nb[i - 1] * bias_ne[i - 1];
        }
        acl_tensor_ptr bias_tensor =
            ggml_cann_create_tensor(bias_ਬਫਰ, ACL_FLOAT, ਆਕਾਰ(ਦਸ਼ਮਲਵ), bias_ne, bias_nb, GGML_MAX_DIMS + 2);

        aclnn_mul(ctx, acl_ਢਲਾਨ.get(), acl_ਨਕਾਬ.get(), bias_tensor.get());
        aclnn_ਜੋੜੋ(ctx, acl_dst.get(), bias_tensor.get());
    } ਨਹੀਂਤਾਂ {
        aclnn_ਜੋੜੋ(ctx, acl_dst.get(), acl_ਨਕਾਬ.get());
    }
}

ਖਾਲੀ ggml_cann_cpy(ggml_backend_cann_context & ctx, ggml_tensor * dst) {
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
ਸਥਿਰ ਖਾਲੀ aclnn_softmax(ggml_backend_cann_context & ctx, aclTensor * acl_src, int64_t dim, aclTensor * acl_dst) {
    GGML_CANN_CALL_ACLNN_OP(ctx, Softmax, acl_src, dim, acl_dst);
}

ਖਾਲੀ ggml_cann_softmax(ggml_backend_cann_context & ctx, ggml_tensor * dst) {
    ggml_tensor * src0 = dst->src[0];
    ggml_tensor * src1 = dst->src[1];  //ਨਕਾਬ

    acl_tensor_ptr acl_src0 = ggml_cann_create_tensor(src0);
    acl_tensor_ptr acl_dst  = ggml_cann_create_tensor(dst);

    ਦਸ਼ਮਲਵ scale    = 1.0f;
    ਦਸ਼ਮਲਵ max_bias = 0.0f;

    ਸਮ੍ਰਿਤੀ_ਨਕਲ(&scale, (ਦਸ਼ਮਲਵ *) dst->op_params + 0, ਆਕਾਰ(ਦਸ਼ਮਲਵ));
    ਸਮ੍ਰਿਤੀ_ਨਕਲ(&max_bias, (ਦਸ਼ਮਲਵ *) dst->op_params + 1, ਆਕਾਰ(ਦਸ਼ਮਲਵ));

    // input mul scale
    acl_scalar_ptr       acl_scale = ggml_cann_create_scalar(&scale, acl_ਅੰਕੜੇ_Type::ACL_FLOAT);
    ggml_cann_pool_alloc src_tensor_allocator(ctx.pool(), ggml_nbytes(src0));
    ਖਾਲੀ *               src_tensor_ਬਫਰ = src_tensor_allocator.get();
    acl_tensor_ptr       softmax_tensor = ggml_cann_create_tensor(src_tensor_ਬਫਰ, ggml_cann_type_mapping(src0->type),
                                                                  ggml_element_ਮਾਪ(src0), src0->ne, src0->nb, GGML_MAX_DIMS);

    aclnn_muls(ctx, acl_src0.get(), scale, softmax_tensor.get(), ਝੂਠ);

    //ਨਕਾਬ
    ਜੇ (src1) {
        aclnn_ਜੋੜੋ_alibi(ctx, src1, src0, src_tensor_ਬਫਰ, max_bias);
    }
    // softmax
    aclnn_softmax(ctx, softmax_tensor.get(), 3, acl_dst.get());
}


ਖਾਲੀ ggml_cann_get_ਪੰਕਤੀਆਂ(ggml_backend_cann_context & ctx, ggml_tensor * dst) {
    ggml_tensor * src0 = dst->src[0];  //ਭਾਰ
    ggml_tensor * src1 = dst->src[1];  //ਸੂਚਕਾਂਕ

    GGML_ASSERT(dst->type == GGML_TYPE_F32 || dst->type == GGML_TYPE_F16
                || dst->type == GGML_TYPE_BF16);

    // n_idx: number of row indices per (i2, i3) batch slice.
    // ggml guarantees: src0->ne[2] == src1->ne[1], src0->ne[3] == src1->ne[2], src1->ne[3] == 1.
    ਅਚਲ int64_t n_idx = src1->ne[0];

    // Gather all (i2, i3) batch slices from src into dst.
    // ggml_cann_create_tensor reverses dims, so ACL sees [ne1, ne0].
    // GatherV2 with dim=0 gathers along ACL dim-0 == ggml ne[1] (the vocabulary / row axis).
    // nb: the 4 strides of the source buffer (nb[0..1] for the 2D slice shape,
    //     nb[2..3] for computing per-batch-slice base pointer offsets).
    ਸਵੈਚਾਲਿਤ gather_batched = [&](ਖਾਲੀ * src_ਅਧਾਰ, acl_ਅੰਕੜੇ_Type acl_type, size_t type_ਮਾਪ,
                              ਅਚਲ size_t * nb) {
        int64_t src_ne[2]  = { src0->ne[0], src0->ne[1] };
        size_t  src_nb_2d[2] = { nb[0], nb[1] };
        int64_t dst_ne[2]  = { src0->ne[0], n_idx };
        size_t  dst_nb_2d[2] = { dst->nb[0], dst->nb[1] };
        int64_t idx_ne[1]  = { n_idx };
        size_t  idx_nb[1]  = { (size_t)ggml_element_ਮਾਪ(src1) };

        ਲਈ (int64_t i3 = 0; i3 < src0->ne[3]; i3++) {
            ਲਈ (int64_t i2 = 0; i2 < src0->ne[2]; i2++) {
                acl_tensor_ptr acl_src = ggml_cann_create_tensor(
                    (ਅੱਖਰ *)src_ਅਧਾਰ + i3 * nb[3] + i2 * nb[2],
                    acl_type, type_ਮਾਪ, src_ne, src_nb_2d, 2);
                acl_tensor_ptr acl_idx = ggml_cann_create_tensor(
                    (ਅੱਖਰ *)src1->ਅੰਕੜੇ + i3 * src1->nb[2] + i2 * src1->nb[1],
                    ggml_cann_type_mapping(src1->type), (size_t)ggml_element_ਮਾਪ(src1),
                    idx_ne, idx_nb, 1);
                acl_tensor_ptr acl_dst = ggml_cann_create_tensor(
                    (ਅੱਖਰ *)dst->ਅੰਕੜੇ + i3 * dst->nb[3] + i2 * dst->nb[2],
                    acl_type, type_ਮਾਪ, dst_ne, dst_nb_2d, 2);
                GGML_CANN_CALL_ACLNN_OP(ctx, GatherV2, acl_src.get(), 0, acl_idx.get(), acl_dst.get());
            }
        }
    };

    ਚੋਣ (src0->type) {
        ਹਾਲਤ GGML_TYPE_BF16:
        ਹਾਲਤ GGML_TYPE_F16:
        ਹਾਲਤ GGML_TYPE_F32:
            ਜੇ (src0->type == dst->type) {
                gather_batched(src0->ਅੰਕੜੇ,
                               ggml_cann_type_mapping(src0->type), ggml_type_ਮਾਪ(src0->type),
                               src0->nb);
            } ਨਹੀਂਤਾਂ {
                // Cast src0 to dst type, then gather.
                ggml_cann_pool_alloc src_cast_allocator(ctx.pool(),
                                                        ggml_nelements(src0) * ggml_element_ਮਾਪ(dst));
                size_t src_cast_nb[GGML_MAX_DIMS];
                src_cast_nb[0] = ggml_type_ਮਾਪ(dst->type);
                ਲਈ (ਪੂਰਨ i = 1; i < GGML_MAX_DIMS; i++) {
                    src_cast_nb[i] = src_cast_nb[i - 1] * src0->ne[i - 1];
                }
                acl_tensor_ptr acl_src0     = ggml_cann_create_tensor(src0);
                acl_tensor_ptr acl_src_cast = ggml_cann_create_tensor(
                    src_cast_allocator.get(), ggml_cann_type_mapping(dst->type), ggml_type_ਮਾਪ(dst->type),
                    src0->ne, src_cast_nb, GGML_MAX_DIMS);
                aclnn_cast(ctx, acl_src0.get(), acl_src_cast.get(), ggml_cann_type_mapping(dst->type));

                gather_batched(src_cast_allocator.get(),
                               ggml_cann_type_mapping(dst->type), ggml_type_ਮਾਪ(dst->type),
                               src_cast_nb);
            }
            ਰੋਕੋ;
        ਹਾਲਤ GGML_TYPE_Q8_0:
            {
                // Dequantize Q8_0 to dst type, then gather.
                size_t  ਭਾਰ_nb[GGML_MAX_DIMS + 1], scale_nb[GGML_MAX_DIMS + 1], dequant_nb[GGML_MAX_DIMS + 1];
                int64_t ਭਾਰ_ne[GGML_MAX_DIMS + 1], scale_ne[GGML_MAX_DIMS + 1], *dequant_ne;
                ਭਾਰ_ne[0] = QK8_0;
                ਭਾਰ_ne[1] = src0->ne[0] / QK8_0;
                ਭਾਰ_nb[0] = ਆਕਾਰ(int8_t);
                ਭਾਰ_nb[1] = ਭਾਰ_nb[0] * ਭਾਰ_ne[0];
                ਲਈ (ਪੂਰਨ i = 2; i < GGML_MAX_DIMS + 1; i++) {
                    ਭਾਰ_ne[i] = src0->ne[i - 1];
                    ਭਾਰ_nb[i] = ਭਾਰ_nb[i - 1] * ਭਾਰ_ne[i - 1];
                }
                scale_ne[0] = 1;
                scale_ne[1] = src0->ne[0] / QK8_0;
                scale_nb[0] = ਆਕਾਰ(uint16_t);
                scale_nb[1] = scale_nb[0] * scale_ne[0];
                ਲਈ (ਪੂਰਨ i = 2; i < GGML_MAX_DIMS + 1; i++) {
                    scale_ne[i] = src0->ne[i - 1];
                    scale_nb[i] = scale_nb[i - 1] * scale_ne[i - 1];
                }
                dequant_ne    = ਭਾਰ_ne;
                dequant_nb[0] = ggml_type_ਮਾਪ(dst->type);
                ਲਈ (ਪੂਰਨ i = 1; i < GGML_MAX_DIMS + 1; i++) {
                    dequant_nb[i] = dequant_nb[i - 1] * dequant_ne[i - 1];
                }
                ਅਚਲ int64_t scale_ਵਿੱਥ = ggml_nelements(src0) * ਆਕਾਰ(int8_t);
                ggml_cann_pool_alloc dequant_allocator(ctx.pool(),
                                                       ggml_nelements(src0) * ggml_type_ਮਾਪ(dst->type));
                acl_tensor_ptr acl_ਭਾਰ = ggml_cann_create_tensor(src0->ਅੰਕੜੇ, ACL_INT8, ਆਕਾਰ(int8_t),
                                                                     ਭਾਰ_ne, ਭਾਰ_nb, GGML_MAX_DIMS + 1);
                acl_tensor_ptr acl_scale  = ggml_cann_create_tensor(
                    src0->ਅੰਕੜੇ, ACL_FLOAT16, ਆਕਾਰ(uint16_t), scale_ne, scale_nb,
                    GGML_MAX_DIMS + 1, ACL_FORMAT_ND, scale_ਵਿੱਥ);
                acl_tensor_ptr acl_dequant = ggml_cann_create_tensor(
                    dequant_allocator.get(), ggml_cann_type_mapping(dst->type),
                    ggml_type_ਮਾਪ(dst->type), dequant_ne, dequant_nb, GGML_MAX_DIMS + 1);
                aclnn_mul(ctx, acl_ਭਾਰ.get(), acl_scale.get(), acl_dequant.get());

                // Reinterpret dequant buffer as 4D [src0->ne] with contiguous strides.
                dequant_ne    = src0->ne;
                dequant_nb[0] = ggml_type_ਮਾਪ(dst->type);
                ਲਈ (ਪੂਰਨ i = 1; i < GGML_MAX_DIMS; i++) {
                    dequant_nb[i] = dequant_nb[i - 1] * src0->ne[i - 1];
                }
                gather_batched(dequant_allocator.get(),
                               ggml_cann_type_mapping(dst->type), ggml_type_ਮਾਪ(dst->type),
                               dequant_nb);
                ਰੋਕੋ;
            }
        ਮੂਲ:
            GGML_ABORT("Unsupported tensor type for GGML_OP_GET_ROWS");
            ਰੋਕੋ;
    }
}

ਖਾਲੀ ggml_cann_ਸੈੱਟ_ਪੰਕਤੀਆਂ(ggml_backend_cann_context & ctx, ggml_tensor * dst) {
    ggml_tensor * src0 = dst->src[0];  // source values
    ggml_tensor * src1 = dst->src[1];  // row indices

    // n_idx: number of source rows to scatter per batch slice.
    // ggml guarantees: src0->ne[1] == src1->ne[0].
    ਅਚਲ int64_t n_idx = src1->ne[0];

    // Copy n_idx rows of src [ne0, n_idx] into dst [ne0, ne1] at positions given by a 1D index.
    // ggml_cann_create_tensor reverses dims, so ACL sees [ne1, ne0] for dst.
    // InplaceIndexCopy with dim=0 copies along ACL dim-0 == ggml ne[1] (the row axis).
    // src_nb: the 4 strides of the source buffer (nb[0..1] for the 2D slice shape,
    //         nb[2..3] for computing per-batch-slice base pointer offsets).
    ਸਵੈਚਾਲਿਤ scatter_batched = [&](ਖਾਲੀ * src_ਅਧਾਰ, acl_ਅੰਕੜੇ_Type acl_type, size_t type_ਮਾਪ,
                               ਅਚਲ size_t * src_nb) {
        int64_t d_ne[2]    = { dst->ne[0], dst->ne[1] };
        size_t  d_nb[2]    = { dst->nb[0], dst->nb[1] };
        int64_t s_ne[2]    = { dst->ne[0], n_idx };
        size_t  s_nb_2d[2] = { src_nb[0], src_nb[1] };
        int64_t i_ne[1]    = { n_idx };
        size_t  i_nb[1]    = { (size_t)ggml_element_ਮਾਪ(src1) };

        ਲਈ (int64_t i3 = 0; i3 < dst->ne[3]; i3++) {
            ਲਈ (int64_t i2 = 0; i2 < dst->ne[2]; i2++) {
                acl_tensor_ptr acl_dst = ggml_cann_create_tensor(
                    (ਅੱਖਰ *)dst->ਅੰਕੜੇ + i3 * dst->nb[3] + i2 * dst->nb[2],
                    acl_type, type_ਮਾਪ, d_ne, d_nb, 2);
                acl_tensor_ptr acl_idx = ggml_cann_create_tensor(
                    (ਅੱਖਰ *)src1->ਅੰਕੜੇ + (i3 % src1->ne[2]) * src1->nb[2] + (i2 % src1->ne[1]) * src1->nb[1],
                    ggml_cann_type_mapping(src1->type), (size_t)ggml_element_ਮਾਪ(src1),
                    i_ne, i_nb, 1);
                acl_tensor_ptr acl_src = ggml_cann_create_tensor(
                    (ਅੱਖਰ *)src_ਅਧਾਰ + i3 * src_nb[3] + i2 * src_nb[2],
                    acl_type, type_ਮਾਪ, s_ne, s_nb_2d, 2);
                GGML_CANN_CALL_ACLNN_OP(ctx, Inplace_ਸੂਚਕਾਂਕ_Copy, acl_dst.get(), 0, acl_idx.get(), acl_src.get());
            }
        }
    };

    ਚੋਣ (dst->type) {
        ਹਾਲਤ GGML_TYPE_F32:
            scatter_batched(src0->ਅੰਕੜੇ,
                            ggml_cann_type_mapping(dst->type), ggml_type_ਮਾਪ(dst->type),
                            src0->nb);
            ਰੋਕੋ;
        ਹਾਲਤ GGML_TYPE_F16:
        ਹਾਲਤ GGML_TYPE_BF16:
            {
                // Cast src0 (F32) to dst type first.
                ggml_cann_pool_alloc src_cast_allocator(ctx.pool(),
                                                        ggml_nelements(src0) * ggml_type_ਮਾਪ(dst->type));
                size_t src_cast_nb[GGML_MAX_DIMS];
                src_cast_nb[0] = ggml_type_ਮਾਪ(dst->type);
                ਲਈ (ਪੂਰਨ i = 1; i < GGML_MAX_DIMS; i++) {
                    src_cast_nb[i] = src_cast_nb[i - 1] * src0->ne[i - 1];
                }
                acl_tensor_ptr acl_src0     = ggml_cann_create_tensor(src0);
                acl_tensor_ptr acl_src_cast = ggml_cann_create_tensor(
                    src_cast_allocator.get(), ggml_cann_type_mapping(dst->type), ggml_type_ਮਾਪ(dst->type),
                    src0->ne, src_cast_nb, GGML_MAX_DIMS);
                aclnn_cast(ctx, acl_src0.get(), acl_src_cast.get(), ggml_cann_type_mapping(dst->type));

                scatter_batched(src_cast_allocator.get(),
                                ggml_cann_type_mapping(dst->type), ggml_type_ਮਾਪ(dst->type),
                                src_cast_nb);
                ਰੋਕੋ;
            }
        ਮੂਲ:
            GGML_ABORT("Unsupported tensor type for GGML_OP_SET_ROWS");
            ਰੋਕੋ;
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
ਸਥਿਰ ਖਾਲੀ aclnn_repeat_interleave(ggml_backend_cann_context & ctx,
                                    aclTensor *                 acl_src,
                                    aclTensor *                 acl_dst,
                                    int64_t                     dim,
                                    int64_t                     repeats,
                                    int64_t                     output_ਮਾਪ) {
    GGML_CANN_CALL_ACLNN_OP(ctx, RepeatInterleaveIntWithDim, acl_src, repeats, dim, output_ਮਾਪ, acl_dst);
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
ਸਥਿਰ ਖਾਲੀ ggml_cann_mat_mul_fp(ggml_backend_cann_context & ctx, ggml_tensor * dst) {
    ggml_tensor * ਭਾਰ = dst->src[0];  //ਭਾਰ
    ggml_tensor * ਨਿਵੇਸ਼  = dst->src[1];  //ਨਿਵੇਸ਼

    // when weight ne2 or ne3 is 1, aclnnMatmulGetWorkspaceSize will auto
    // broadcast, when weight ne2 or ne3 is not 1, weight need repeat.
    BCAST_MUL_MAT_SHAPE(ਨਿਵੇਸ਼, ਭਾਰ, dst);

    int64_t n_dims = bcast_dims;
    ਜੇ (bcast_ਨਿਵੇਸ਼_ne[3] == bcast_ਭਾਰ_ne[3] && bcast_ਨਿਵੇਸ਼_ne[3] == 1) {
        ਜੇ (bcast_ਨਿਵੇਸ਼_ne[2] == 1 && bcast_ਭਾਰ_ne[2] == 1) {
            n_dims = 2;
        } ਨਹੀਂਤਾਂ ਜੇ (bcast_ਨਿਵੇਸ਼_ne[2] == 1) {
            n_dims = 3;
        }
    }

    acl_tensor_ptr acl_ਨਿਵੇਸ਼_tensor = ggml_cann_create_tensor(ਨਿਵੇਸ਼, bcast_ਨਿਵੇਸ਼_ne, bcast_ਨਿਵੇਸ਼_nb, n_dims);
    int64_t        transpose_ne[]   = { bcast_ਭਾਰ_ne[1], bcast_ਭਾਰ_ne[0], bcast_ਭਾਰ_ne[2],
                                        bcast_ਭਾਰ_ne[3], bcast_ਭਾਰ_ne[4], bcast_ਭਾਰ_ne[5] };
    size_t         transpose_nb[]   = { bcast_ਭਾਰ_nb[1], bcast_ਭਾਰ_nb[0], bcast_ਭਾਰ_nb[2],
                                        bcast_ਭਾਰ_nb[3], bcast_ਭਾਰ_nb[4], bcast_ਭਾਰ_nb[5] };
    acl_tensor_ptr acl_ਭਾਰ_tensor;

    // Only check env once.
    ਸਥਿਰ ਤਾਰਕਿਕ ਭਾਰ_ਮੰਜ਼ਿਲ_nz = parse_bool(get_env_as_lowercase("GGML_CANN_WEIGHT_NZ").value_or("on"));
    ਜੇ (ਭਾਰ_ਮੰਜ਼ਿਲ_nz && ਭਾਰ->type != GGML_TYPE_BF16 && is_matmul_ਭਾਰ(ਭਾਰ)) {
        acl_ਭਾਰ_tensor = ggml_cann_create_tensor(ਭਾਰ, transpose_ne, transpose_nb, n_dims, ACL_FORMAT_FRACTAL_NZ);
    } ਨਹੀਂਤਾਂ {
        acl_ਭਾਰ_tensor = ggml_cann_create_tensor(ਭਾਰ, transpose_ne, transpose_nb, n_dims, ACL_FORMAT_ND);
    }
    acl_tensor_ptr acl_dst = ggml_cann_create_tensor(dst, bcast_dst_ne, bcast_dst_nb, n_dims);

    ਚੋਣ (n_dims) {
        ਹਾਲਤ 2:
            GGML_CANN_CALL_ACLNN_OP(ctx, Mm, acl_ਨਿਵੇਸ਼_tensor.get(), acl_ਭਾਰ_tensor.get(), acl_dst.get(), 2);
            ਰੋਕੋ;
        ਹਾਲਤ 3:
            GGML_CANN_CALL_ACLNN_OP(ctx, BatchMatMul, acl_ਨਿਵੇਸ਼_tensor.get(), acl_ਭਾਰ_tensor.get(), acl_dst.get(),
                                    2);
            ਰੋਕੋ;
        ਮੂਲ:
            // ALLOW_FP32_DOWN_PRECISION, when input is
            // fp32, atlas a2 will transpose it to HFLOAT32.
            GGML_CANN_CALL_ACLNN_OP(ctx, Matmul, acl_ਨਿਵੇਸ਼_tensor.get(), acl_ਭਾਰ_tensor.get(), acl_dst.get(), 1);
            ਰੋਕੋ;
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
ਸਥਿਰ ਖਾਲੀ ggml_cann_mul_mat_quant(ggml_backend_cann_context & ctx, ggml_tensor * dst, ਅਚਲ ਗਿਣਤੀ ggml_type type) {
    ggml_tensor * src0 = dst->src[0];  //ਭਾਰ
    ggml_tensor * src1 = dst->src[1];  //ਨਿਵੇਸ਼

    // The shape of the weight is NCHW.
    // Matrix multiplication uses HW dims.
    // HC is regarded as batch.
    // weight need transpose.
    ਦਸ਼ਮਲਵ ਭਾਰ_elem_ਮਾਪ;
    ਜੇ (type == GGML_TYPE_Q4_0) {
        ਭਾਰ_elem_ਮਾਪ = ਦਸ਼ਮਲਵ(ਆਕਾਰ(uint8_t)) / 2;
    } ਨਹੀਂਤਾਂ ਜੇ (type == GGML_TYPE_Q8_0) {
        ਭਾਰ_elem_ਮਾਪ = ਦਸ਼ਮਲਵ(ਆਕਾਰ(uint8_t));
    } ਨਹੀਂਤਾਂ {
        GGML_ABORT("Only support Q4_0 and Q8_0 MUL_MAT");
    }
    ਦਸ਼ਮਲਵ  ਭਾਰ_nb[]   = { src0->ne[0] * ਭਾਰ_elem_ਮਾਪ, ਭਾਰ_elem_ਮਾਪ };
    size_t ਭਾਰ_stride = src0->ne[1] * src0->ne[0] * ਭਾਰ_elem_ਮਾਪ;
    size_t ਭਾਰ_ਮਾਪ   = ਭਾਰ_stride * src0->ne[2] * src0->ne[3];

    // scale stored at the end of weight. Also need transpose.
    size_t scale_elem_ਮਾਪ = ਆਕਾਰ(uint16_t);
    size_t scale_nb[]      = { src0->ne[0] / QK8_0 * scale_elem_ਮਾਪ, scale_elem_ਮਾਪ };
    size_t scale_stride    = src0->ne[1] * src0->ne[0] / QK8_0 * scale_elem_ਮਾਪ;
    ਅੱਖਰ * scale_ਵਿੱਥ    = (ਅੱਖਰ *) src0->ਅੰਕੜੇ + ਭਾਰ_ਮਾਪ;

    //ਨਿਵੇਸ਼
    size_t               ਨਿਵੇਸ਼_elem_ਮਾਪ = ਆਕਾਰ(uint16_t);
    int64_t              ਨਿਵੇਸ਼_ne[]      = { src1->ne[0], src1->ne[1] };
    size_t               ਨਿਵੇਸ਼_nb[]      = { ਨਿਵੇਸ਼_elem_ਮਾਪ, ਨਿਵੇਸ਼_ne[0] * ਨਿਵੇਸ਼_elem_ਮਾਪ };
    size_t               ਨਿਵੇਸ਼_stride    = ਨਿਵੇਸ਼_ne[0] * ਨਿਵੇਸ਼_ne[1] * ਨਿਵੇਸ਼_elem_ਮਾਪ;
    ggml_cann_pool_alloc ਨਿਵੇਸ਼_alloctor(ctx.pool());
    ਖਾਲੀ *               ਨਿਵੇਸ਼_ਬਫਰ = src1->ਅੰਕੜੇ;

    // case in
    ਜੇ (src1->type != GGML_TYPE_F16) {
        acl_tensor_ptr acl_src1_tensor = ggml_cann_create_tensor(src1);
        ਨਿਵੇਸ਼_ਬਫਰ                   = ਨਿਵੇਸ਼_alloctor.alloc(ggml_nelements(src1) * ਨਿਵੇਸ਼_elem_ਮਾਪ);

        int64_t * ਨਿਵੇਸ਼_cast_ne = src1->ne;
        size_t    ਨਿਵੇਸ਼_cast_nb[GGML_MAX_DIMS];
        ਨਿਵੇਸ਼_cast_nb[0] = ਆਕਾਰ(uint16_t);
        ਲਈ (ਪੂਰਨ i = 1; i < GGML_MAX_DIMS; i++) {
            ਨਿਵੇਸ਼_cast_nb[i] = ਨਿਵੇਸ਼_cast_nb[i - 1] * ਨਿਵੇਸ਼_cast_ne[i - 1];
        }

        acl_tensor_ptr acl_ਨਿਵੇਸ਼_tensor = ggml_cann_create_tensor(ਨਿਵੇਸ਼_ਬਫਰ, ACL_FLOAT16, ਨਿਵੇਸ਼_elem_ਮਾਪ,
                                                                  ਨਿਵੇਸ਼_cast_ne, ਨਿਵੇਸ਼_cast_nb, GGML_MAX_DIMS);
        aclnn_cast(ctx, acl_src1_tensor.get(), acl_ਨਿਵੇਸ਼_tensor.get(), ACL_FLOAT16);
    }

    // output
    size_t               output_elem_ਮਾਪ = ਆਕਾਰ(uint16_t);
    size_t               output_nb[]      = { output_elem_ਮਾਪ, dst->ne[0] * output_elem_ਮਾਪ };
    ggml_cann_pool_alloc output_allocator(ctx.pool());
    ਖਾਲੀ *               output_ਬਫਰ = output_allocator.alloc(ggml_nelements(dst) * output_elem_ਮਾਪ);
    size_t               output_stride = dst->ne[0] * dst->ne[1] * output_elem_ਮਾਪ;

    // aclnn
    int64_t              max_elem_ਮਾਪ = 65535;
    int64_t              ਵੰਡੋ_ਮਾਪ    = (src0->ne[1] / max_elem_ਮਾਪ) + 1;
    ggml_cann_pool_alloc workspace_allocator(ctx.pool());
    ਲਈ (int64_t n1 = 0; n1 < src1->ne[3]; n1++) {
        ਲਈ (int64_t c1 = 0; c1 < src1->ne[2]; c1++) {
            int64_t n0 = n1 / (src1->ne[3] / src0->ne[3]);
            int64_t c0 = c1 / (src1->ne[2] / src0->ne[2]);

            int64_t batch1 = (n1 * src1->ne[2]) + c1;
            int64_t batch0 = (n0 * src0->ne[2]) + c0;

            acl_tensor_ptr acl_ਨਿਵੇਸ਼_tensor = ggml_cann_create_tensor(
                (ਅੱਖਰ *) ਨਿਵੇਸ਼_ਬਫਰ + batch1 * ਨਿਵੇਸ਼_stride, ACL_FLOAT16, ਨਿਵੇਸ਼_elem_ਮਾਪ, ਨਿਵੇਸ਼_ne, ਨਿਵੇਸ਼_nb, 2);

            // first split
            int64_t ਭਾਰ_ne_ਵਿੱਥ = 0;
            int64_t ਭਾਰ_ne[2]     = { max_elem_ਮਾਪ > src0->ne[1] ? src0->ne[1] : max_elem_ਮਾਪ, src0->ne[0] };
            int64_t scale_ne_ਵਿੱਥ  = 0;
            int64_t scale_ne[2]      = { ਭਾਰ_ne[0], ਭਾਰ_ne[1] / QK8_0 };
            int64_t output_ne_ਵਿੱਥ = 0;
            int64_t output_ne[2]     = { ਭਾਰ_ne[0], dst->ne[1] };

            acl_tensor_ptr acl_ਭਾਰ_tensor =
                ggml_cann_create_tensor((ਅੱਖਰ *) src0->ਅੰਕੜੇ + batch0 * ਭਾਰ_stride, ggml_cann_type_mapping(type),
                                        ਭਾਰ_elem_ਮਾਪ, ਭਾਰ_ne, ਭਾਰ_nb, 2, ACL_FORMAT_ND, ਭਾਰ_ne_ਵਿੱਥ);
            acl_tensor_ptr acl_scale_tensor =
                ggml_cann_create_tensor(scale_ਵਿੱਥ + batch0 * scale_stride, ACL_FLOAT16, scale_elem_ਮਾਪ, scale_ne,
                                        scale_nb, 2, ACL_FORMAT_ND, scale_ne_ਵਿੱਥ);
            acl_tensor_ptr acl_output_tensor =
                ggml_cann_create_tensor((ਅੱਖਰ *) output_ਬਫਰ + batch1 * output_stride, ACL_FLOAT16, output_elem_ਮਾਪ,
                                        output_ne, output_nb, 2, ACL_FORMAT_ND, output_ne_ਵਿੱਥ);
            int64_t antiquant_ਸਮੂਹ_ਮਾਪ = 0;
            ਜੇ (src0->ne[0] > QK8_0) {
                antiquant_ਸਮੂਹ_ਮਾਪ = QK8_0;
            }
            GGML_CANN_CALL_ACLNN_OP(ctx, ਭਾਰ_Quant_Batch_Matmul_V2, acl_ਨਿਵੇਸ਼_tensor.get(), acl_ਭਾਰ_tensor.get(),
                                    acl_scale_tensor.get(), ਖਾਲੀ_ਸੂਚਕ, ਖਾਲੀ_ਸੂਚਕ, ਖਾਲੀ_ਸੂਚਕ, ਖਾਲੀ_ਸੂਚਕ, antiquant_ਸਮੂਹ_ਮਾਪ,
                                    acl_output_tensor.get());

            // other splits
            ਲਈ (int64_t ਵੰਡੋ = 1; ਵੰਡੋ < ਵੰਡੋ_ਮਾਪ; ਵੰਡੋ++) {
                ਭਾਰ_ne_ਵਿੱਥ += ਭਾਰ_elem_ਮਾਪ * ਭਾਰ_ne[0] * ਭਾਰ_ne[1];
                ਭਾਰ_ne[0] =
                    max_elem_ਮਾਪ * (ਵੰਡੋ + 1) > src0->ne[1] ? src0->ne[1] - (max_elem_ਮਾਪ * ਵੰਡੋ) : max_elem_ਮਾਪ;
                scale_ne_ਵਿੱਥ += scale_elem_ਮਾਪ * scale_ne[0] * scale_ne[1];
                scale_ne[0] = ਭਾਰ_ne[0];
                output_ne_ਵਿੱਥ += output_elem_ਮਾਪ * output_ne[0] * output_ne[1];
                output_ne[0] = ਭਾਰ_ne[0];

                acl_ਭਾਰ_tensor =
                    ggml_cann_create_tensor((ਅੱਖਰ *) src0->ਅੰਕੜੇ + batch0 * ਭਾਰ_stride, ggml_cann_type_mapping(type),
                                            ਭਾਰ_elem_ਮਾਪ, ਭਾਰ_ne, ਭਾਰ_nb, 2, ACL_FORMAT_ND, ਭਾਰ_ne_ਵਿੱਥ);
                acl_scale_tensor =
                    ggml_cann_create_tensor(scale_ਵਿੱਥ + batch0 * scale_stride, ACL_FLOAT16, scale_elem_ਮਾਪ,
                                            scale_ne, scale_nb, 2, ACL_FORMAT_ND, scale_ne_ਵਿੱਥ);
                acl_output_tensor =
                    ggml_cann_create_tensor((ਅੱਖਰ *) output_ਬਫਰ + batch1 * output_stride, ACL_FLOAT16,
                                            output_elem_ਮਾਪ, output_ne, output_nb, 2, ACL_FORMAT_ND, output_ne_ਵਿੱਥ);
                GGML_CANN_CALL_ACLNN_OP(ctx, ਭਾਰ_Quant_Batch_Matmul_V2, acl_ਨਿਵੇਸ਼_tensor.get(), acl_ਭਾਰ_tensor.get(),
                                        acl_scale_tensor.get(), ਖਾਲੀ_ਸੂਚਕ, ਖਾਲੀ_ਸੂਚਕ, ਖਾਲੀ_ਸੂਚਕ, ਖਾਲੀ_ਸੂਚਕ, antiquant_ਸਮੂਹ_ਮਾਪ,
                                        acl_output_tensor.get());
            }
        }
    }

    // cast out
    ਜੇ (dst->type != GGML_TYPE_F16) {
        int64_t * output_cast_ne = dst->ne;
        size_t    output_cast_nb[GGML_MAX_DIMS];
        output_cast_nb[0] = ਆਕਾਰ(uint16_t);
        ਲਈ (ਪੂਰਨ i = 1; i < GGML_MAX_DIMS; i++) {
            output_cast_nb[i] = output_cast_nb[i - 1] * output_cast_ne[i - 1];
        }

        acl_tensor_ptr acl_output_tensor = ggml_cann_create_tensor(output_ਬਫਰ, ACL_FLOAT16, output_elem_ਮਾਪ,
                                                                   output_cast_ne, output_cast_nb, GGML_MAX_DIMS);
        acl_tensor_ptr acl_dst_tensor    = ggml_cann_create_tensor(dst);
        aclnn_cast(ctx, acl_output_tensor.get(), acl_dst_tensor.get(), ggml_cann_type_mapping(dst->type));
    }
}

ਖਾਲੀ ggml_cann_mul_mat(ggml_backend_cann_context & ctx, ggml_tensor * dst) {
    ਅਚਲ ਗਿਣਤੀ ggml_type type = dst->src[0]->type;
    ਚੋਣ (type) {
        ਹਾਲਤ GGML_TYPE_F32:
        ਹਾਲਤ GGML_TYPE_F16:
#ਜੇ_ਅਪਰਿਭਾਸ਼ਿਤ ASCEND_310P
        ਹਾਲਤ GGML_TYPE_BF16:
#ਜੇ_ਅੰਤ
            ggml_cann_mat_mul_fp(ctx, dst);
            ਰੋਕੋ;
        ਹਾਲਤ GGML_TYPE_Q4_0:
        ਹਾਲਤ GGML_TYPE_Q8_0:
            ggml_cann_mul_mat_quant(ctx, dst, type);
            ਰੋਕੋ;
        ਮੂਲ:
            GGML_ABORT("Unsupported type for mul_mat");
            ਰੋਕੋ;
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
ਸਥਿਰ ਖਾਲੀ aclnn_roll(ggml_backend_cann_context & ctx,
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
ਸਥਿਰ ਖਾਲੀ aclnn_ਸੂਚਕਾਂਕ_fill_tensor(ggml_backend_cann_context & ctx,
                                    aclTensor *                 acl_src,
                                    int64_t                     dim,
                                    int64_t *                   ਸੂਚਕਾਂਕ,
                                    int64_t                     ਸੂਚਕਾਂਕ_ਅੰਕ,
                                    ਦਸ਼ਮਲਵ                       value) {
    acl_int_array_ptr acl_ਸੂਚਕਾਂਕ = ggml_cann_create_int_array(ਸੂਚਕਾਂਕ, ਸੂਚਕਾਂਕ_ਅੰਕ);
    acl_scalar_ptr    acl_ਮੁੱਲ = ggml_cann_create_scalar(&value, acl_ਅੰਕੜੇ_Type::ACL_FLOAT);
    GGML_CANN_CALL_ACLNN_OP(ctx, Inplace_ਸੂਚਕਾਂਕ_Fill_Tensor, acl_src, dim, acl_ਸੂਚਕਾਂਕ.get(), acl_ਮੁੱਲ.get());
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
ਸਥਿਰ ਖਾਲੀ aclnn_rope_cache_init(ggml_backend_cann_context & ctx,
                                  ggml_tensor *               dst,
                                  ਦਸ਼ਮਲਵ *                     corr_dims,
                                  ਦਸ਼ਮਲਵ                       ext_ਗੁਣਕ,
                                  ਦਸ਼ਮਲਵ                       theta_scale,
                                  ਦਸ਼ਮਲਵ                       freq_scale,
                                  ਦਸ਼ਮਲਵ                       attn_ਗੁਣਕ,
                                  ਤਾਰਕਿਕ                        is_neox,
                                  ਪੂਰਨ                         sections[4],
                                  ਤਾਰਕਿਕ                        mrope_ਵਰਤਿਆ,
                                  ਤਾਰਕਿਕ                        is_imrope,
                                  ਤਾਰਕਿਕ                        indep_sects,
                                  int64_t                     rope_dims) {
    ggml_tensor * src1 = dst->src[1];  // position
    ggml_tensor * src2 = dst->src[2];  // freq_factors

    int64_t theta_scale_ਲੰਮਾਈ = rope_dims / 2;
    int64_t position_ਲੰਮਾਈ    = dst->ne[2];

    // TODO: check theta_scale_length and position_length.
    ਜੇ (src2 == ਖਾਲੀ_ਸੂਚਕ && ctx.rope_cache.cached &&
        ctx.rope_cache.equal(theta_scale_ਲੰਮਾਈ, position_ਲੰਮਾਈ, ext_ਗੁਣਕ, theta_scale, freq_scale, attn_ਗੁਣਕ,
                             is_neox, indep_sects, mrope_ਵਰਤਿਆ, is_imrope, sections)) {
        // use cache.
        ਵਾਪਸ;
    }

    // Step0: calculate tensor shape.
    int64_t theta_scale_ne[] = { theta_scale_ਲੰਮਾਈ, 1, 1, 1 };
    size_t  theta_scale_nb[] = { ਆਕਾਰ(ਦਸ਼ਮਲਵ), theta_scale_ਲੰਮਾਈ * ਆਕਾਰ(ਦਸ਼ਮਲਵ), theta_scale_ਲੰਮਾਈ * ਆਕਾਰ(ਦਸ਼ਮਲਵ),
                                 theta_scale_ਲੰਮਾਈ * ਆਕਾਰ(ਦਸ਼ਮਲਵ) };

    GGML_ASSERT(src1->type == GGML_TYPE_I32);
    int64_t position_ne[] = { 1, 1, position_ਲੰਮਾਈ, 1 };
    size_t  position_nb[] = { ਆਕਾਰ(int32_t), ਆਕਾਰ(int32_t), ਆਕਾਰ(int32_t), ਆਕਾਰ(int32_t) * position_ਲੰਮਾਈ };

    int64_t cache_ne[] = { theta_scale_ਲੰਮਾਈ, 1, position_ਲੰਮਾਈ, 1 };
    size_t  cache_nb[GGML_MAX_DIMS];
    cache_nb[0] = ਆਕਾਰ(ਦਸ਼ਮਲਵ);
    ਲਈ (ਪੂਰਨ i = 1; i < GGML_MAX_DIMS; i++) {
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
    ਤਾਰਕਿਕ           theta_scale_updated = ਝੂਠ;
    ਜੇ (ctx.rope_cache.theta_scale_ਲੰਮਾਈ != theta_scale_ਲੰਮਾਈ || ctx.rope_cache.theta_scale != theta_scale ||
        ctx.rope_cache.indep_sects != indep_sects) {
        theta_scale_updated = ਸੱਚ;
        ਜੇ (ctx.rope_cache.theta_scale_exp_host != ਖਾਲੀ_ਸੂਚਕ) {
            ਮੁਕਤ(ctx.rope_cache.theta_scale_exp_host);
        }
        ctx.rope_cache.theta_scale_exp_host = (ਦਸ਼ਮਲਵ *) ਸਮ੍ਰਿਤੀ_ਦਿਓ(theta_scale_ਲੰਮਾਈ * ਆਕਾਰ(ਦਸ਼ਮਲਵ));
        GGML_ASSERT(ctx.rope_cache.theta_scale_exp_host != ਖਾਲੀ_ਸੂਚਕ);
        ਜੇ (!indep_sects) {
            ctx.rope_cache.theta_scale_exp_host[0] = 1;
            ਲਈ (ਪੂਰਨ i = 1; i < theta_scale_ਲੰਮਾਈ; i++) {
                ctx.rope_cache.theta_scale_exp_host[i] = ctx.rope_cache.theta_scale_exp_host[i - 1] * theta_scale;
            }
        } ਨਹੀਂਤਾਂ {
            ਪੂਰਨ sect_dims = sections[0] + sections[1] + sections[2] + sections[3];
            ਪੂਰਨ sec_w     = sections[1] + sections[0];
            ਪੂਰਨ sec_e     = sections[2] + sec_w;

            ctx.rope_cache.theta_scale_exp_host[0] = 1;
            ਲਈ (ਪੂਰਨ i = 1; i < theta_scale_ਲੰਮਾਈ; i++) {
                ਪੂਰਨ sector = i % sect_dims;
                ਜੇ (sector == 0 || sector == sections[0] || sector == sec_w || sector == sec_e) {
                    ctx.rope_cache.theta_scale_exp_host[i] = 1;
                    ਜਾਰੀ;
                }
                ctx.rope_cache.theta_scale_exp_host[i] = ctx.rope_cache.theta_scale_exp_host[i - 1] * theta_scale;
            }
        }

        ਜੇ (ctx.rope_cache.theta_scale_cache != ਖਾਲੀ_ਸੂਚਕ) {
            ACL_CHECK(aclrtFree(ctx.rope_cache.theta_scale_cache));
        }
        ACL_CHECK(aclrtMalloc(&ctx.rope_cache.theta_scale_cache, theta_scale_ਲੰਮਾਈ * ਆਕਾਰ(ਦਸ਼ਮਲਵ),
                              ACL_MEM_MALLOC_HUGE_ਪਹਿਲਾ));

        ACL_CHECK(aclrtMemcpyAsync(ctx.rope_cache.theta_scale_cache, theta_scale_ਲੰਮਾਈ * ਆਕਾਰ(ਦਸ਼ਮਲਵ),
                                   ctx.rope_cache.theta_scale_exp_host, theta_scale_ਲੰਮਾਈ * ਆਕਾਰ(ਦਸ਼ਮਲਵ),
                                   ACL_MEMCPY_HOST_ਮੰਜ਼ਿਲ_DEVICE, ctx.stream()));
    }
    acl_theta_scale_tensor = ggml_cann_create_tensor(ctx.rope_cache.theta_scale_cache, ACL_FLOAT, ਆਕਾਰ(ਦਸ਼ਮਲਵ),
                                                     theta_scale_ne, theta_scale_nb, 1);

    // Step1.2: prepare rope_yarn_ramp, if this part updated, should update theta_scale_tensor.
    // TODO: acl_yarn_ramp_tensor use rope cache.
    ਤਾਰਕਿਕ           yarn_ramp_tensor_updated = ਝੂਠ;
    acl_tensor_ptr acl_yarn_ramp_tensor;
    ਜੇ (ext_ਗੁਣਕ != 0 && (theta_scale_updated || ctx.rope_cache.theta_scale_ਲੰਮਾਈ != theta_scale_ਲੰਮਾਈ ||
                            ctx.rope_cache.freq_scale != freq_scale)) {
        yarn_ramp_tensor_updated = ਸੱਚ;
        ਜੇ (ctx.rope_cache.yarn_ramp_cache != ਖਾਲੀ_ਸੂਚਕ) {
            ACL_CHECK(aclrtFree(ctx.rope_cache.yarn_ramp_cache));
        }
        ACL_CHECK(aclrtMalloc(&ctx.rope_cache.yarn_ramp_cache, theta_scale_ਲੰਮਾਈ * ਆਕਾਰ(ਦਸ਼ਮਲਵ),
                              ACL_MEM_MALLOC_HUGE_ਪਹਿਲਾ));
        // -rope_yarn_ramp
        // const float y = (i0 / 2 - low) / MAX(0.001f, high - low);
        // return MIN(1, MAX(0, y)) - 1;
        acl_yarn_ramp_tensor      = ggml_cann_create_tensor(ctx.rope_cache.yarn_ramp_cache, ACL_FLOAT, ਆਕਾਰ(ਦਸ਼ਮਲਵ),
                                                            theta_scale_ne, theta_scale_nb, 1);
        ਦਸ਼ਮਲਵ          zero_ਮੁੱਲ = 0, one_ਮੁੱਲ = 1;
        ਦਸ਼ਮਲਵ          denom_ਸੁਰੱਖਿਅਤ_ਮੁੱਲ = MAX(0.001f, corr_dims[1] - corr_dims[0]);
        acl_scalar_ptr ਨੀਵਾਂ              = ggml_cann_create_scalar(&corr_dims[0], acl_ਅੰਕੜੇ_Type::ACL_FLOAT);
        acl_scalar_ptr zero             = ggml_cann_create_scalar(&zero_ਮੁੱਲ, acl_ਅੰਕੜੇ_Type::ACL_FLOAT);
        acl_scalar_ptr one              = ggml_cann_create_scalar(&one_ਮੁੱਲ, acl_ਅੰਕੜੇ_Type::ACL_FLOAT);
        acl_scalar_ptr denom_ਸੁਰੱਖਿਅਤ       = ggml_cann_create_scalar(&denom_ਸੁਰੱਖਿਅਤ_ਮੁੱਲ, acl_ਅੰਕੜੇ_Type::ACL_FLOAT);
        acl_scalar_ptr ext_ਗੁਣਕ_sc    = ggml_cann_create_scalar(&ext_ਗੁਣਕ, acl_ਅੰਕੜੇ_Type::ACL_FLOAT);

        aclnn_arange(ctx, acl_yarn_ramp_tensor.get(), 0, theta_scale_ਲੰਮਾਈ, 1, theta_scale_ਲੰਮਾਈ);
        GGML_CANN_CALL_ACLNN_OP(ctx, InplaceSubs, acl_yarn_ramp_tensor.get(), ਨੀਵਾਂ.get(), one.get());
        GGML_CANN_CALL_ACLNN_OP(ctx, InplaceDivs, acl_yarn_ramp_tensor.get(), denom_ਸੁਰੱਖਿਅਤ.get());
        GGML_CANN_CALL_ACLNN_OP(ctx, Inplace_ਹੱਦ, acl_yarn_ramp_tensor.get(), zero.get(), zero.get());
        GGML_CANN_CALL_ACLNN_OP(ctx, InplaceClampMax, acl_yarn_ramp_tensor.get(), one.get());
        GGML_CANN_CALL_ACLNN_OP(ctx, InplaceSubs, acl_yarn_ramp_tensor.get(), one.get(), one.get());
        GGML_CANN_CALL_ACLNN_OP(ctx, InplaceMuls, acl_yarn_ramp_tensor.get(), ext_ਗੁਣਕ_sc.get());

        // theta_interp = freq_scale * theta_extrap;
        // theta = theta_interp * (1 - ramp_mix) + theta_extrap * ramp_mix;
        // theta = freq_scale * theta_extrap * (1 - ramp_mix) + theta_extrap * ramp_mix;
        // theta = freq_scale * theta_extrap - freq_scale * theta_extrap * ramp_mix + theta_extrap * ramp_mix;
        // theta = theta_extrap * (freq_scale - freq_scale * ramp_mix + ramp_mix);
        //
        // we cache (freq_scale - freq_scale * ramp_mix + ramp_mix), Considering that the rope_yarn_ramp here is the inverse
        // cache freq_scale + (freq_scale - 1) * ramp_mix
        ਦਸ਼ਮਲਵ          freq_scale_1    = freq_scale - 1;
        acl_scalar_ptr freq_scale_sc   = ggml_cann_create_scalar(&freq_scale, acl_ਅੰਕੜੇ_Type::ACL_FLOAT);
        acl_scalar_ptr freq_scale_1_sc = ggml_cann_create_scalar(&freq_scale_1, acl_ਅੰਕੜੇ_Type::ACL_FLOAT);
        GGML_CANN_CALL_ACLNN_OP(ctx, InplaceMuls, acl_yarn_ramp_tensor.get(), freq_scale_1_sc.get());
        GGML_CANN_CALL_ACLNN_OP(ctx, InplaceAdds, acl_yarn_ramp_tensor.get(), freq_scale_sc.get(), one.get());
    } ਨਹੀਂਤਾਂ {
        acl_yarn_ramp_tensor = ggml_cann_create_tensor(ctx.rope_cache.yarn_ramp_cache, ACL_FLOAT, ਆਕਾਰ(ਦਸ਼ਮਲਵ),
                                                       theta_scale_ne, theta_scale_nb, 1);
    }
    // Step 1.3: update theta_scale_tensor according to ext_factor or freq_scale.
    ਜੇ (ext_ਗੁਣਕ != 0) {
        ਜੇ (theta_scale_updated || yarn_ramp_tensor_updated) {
            theta_scale_updated = ਸੱਚ;
            aclnn_mul(ctx, acl_theta_scale_tensor.get(), acl_yarn_ramp_tensor.get());
        }
    } ਨਹੀਂਤਾਂ {
        ਜੇ (freq_scale != 1 && (ctx.rope_cache.freq_scale != freq_scale || theta_scale_updated)) {
            theta_scale_updated = ਸੱਚ;
            aclnn_muls(ctx, acl_theta_scale_tensor.get(), freq_scale, ਖਾਲੀ_ਸੂਚਕ, ਸੱਚ);
        }
    }

    // Nothing changed, use cache.
    ਜੇ (!theta_scale_updated) {
        acl_theta_scale_tensor = ggml_cann_create_tensor(ctx.rope_cache.theta_scale_cache, ACL_FLOAT, ਆਕਾਰ(ਦਸ਼ਮਲਵ),
                                                         theta_scale_ne, theta_scale_nb, GGML_MAX_DIMS);
    }

    // Step 1.4: prepare select index if mrope
    acl_tensor_ptr position_select_ਸੂਚਕਾਂਕ_tensor;
    ਜੇ (mrope_ਵਰਤਿਆ) {
        ਜੇ (ctx.rope_cache.sections[0] != sections[0] || ctx.rope_cache.sections[1] != sections[1] ||
            ctx.rope_cache.sections[2] != sections[2] || ctx.rope_cache.sections[3] != sections[3] ||
            ctx.rope_cache.theta_scale_ਲੰਮਾਈ != theta_scale_ਲੰਮਾਈ || ctx.rope_cache.is_imrope != is_imrope) {
            ਜੇ (ctx.rope_cache.position_select_ਸੂਚਕਾਂਕ_host != ਖਾਲੀ_ਸੂਚਕ) {
                ਮੁਕਤ(ctx.rope_cache.position_select_ਸੂਚਕਾਂਕ_host);
            }
            ctx.rope_cache.position_select_ਸੂਚਕਾਂਕ_host = (ਪੂਰਨ *) ਸਮ੍ਰਿਤੀ_ਦਿਓ(theta_scale_ਲੰਮਾਈ * ਆਕਾਰ(ਪੂਰਨ));
            GGML_ASSERT(ctx.rope_cache.position_select_ਸੂਚਕਾਂਕ_host != ਖਾਲੀ_ਸੂਚਕ);
            ਪੂਰਨ sect_dims = sections[0] + sections[1] + sections[2] + sections[3];
            ਪੂਰਨ sec_w     = sections[1] + sections[0];
            ਪੂਰਨ sec_e     = sections[2] + sec_w;
            // t,h,w,e
            ਲਈ (ਪੂਰਨ i = 0; i < theta_scale_ਲੰਮਾਈ; i++) {
                ਪੂਰਨ sector = i % sect_dims;

                ਜੇ (is_imrope) {  // qwen3vl apply interleaved mrope
                    ਜੇ (sector % 3 == 1 && sector < 3 * sections[1]) {
                        ctx.rope_cache.position_select_ਸੂਚਕਾਂਕ_host[i] = 1;
                    } ਨਹੀਂਤਾਂ ਜੇ (sector % 3 == 2 && sector < 3 * sections[2]) {
                        ctx.rope_cache.position_select_ਸੂਚਕਾਂਕ_host[i] = 2;
                    } ਨਹੀਂਤਾਂ ਜੇ (sector % 3 == 0 && sector < 3 * sections[0]) {
                        ctx.rope_cache.position_select_ਸੂਚਕਾਂਕ_host[i] = 0;
                    } ਨਹੀਂਤਾਂ {
                        ctx.rope_cache.position_select_ਸੂਚਕਾਂਕ_host[i] = 3;
                    }
                } ਨਹੀਂਤਾਂ {
                    ਜੇ (sector >= sections[0] && sector < sec_w) {
                        ctx.rope_cache.position_select_ਸੂਚਕਾਂਕ_host[i] = 1;
                    } ਨਹੀਂਤਾਂ ਜੇ (sector >= sec_w && sector < sec_e) {
                        ctx.rope_cache.position_select_ਸੂਚਕਾਂਕ_host[i] = 2;
                    } ਨਹੀਂਤਾਂ ਜੇ (sector >= sec_e) {
                        ctx.rope_cache.position_select_ਸੂਚਕਾਂਕ_host[i] = 3;
                    } ਨਹੀਂਤਾਂ {
                        ctx.rope_cache.position_select_ਸੂਚਕਾਂਕ_host[i] = 0;
                    }
                }
            }

            ਜੇ (ctx.rope_cache.position_select_ਸੂਚਕਾਂਕ != ਖਾਲੀ_ਸੂਚਕ) {
                ACL_CHECK(aclrtFree(ctx.rope_cache.position_select_ਸੂਚਕਾਂਕ));
            }
            ACL_CHECK(aclrtMalloc(&ctx.rope_cache.position_select_ਸੂਚਕਾਂਕ, theta_scale_ਲੰਮਾਈ * ਆਕਾਰ(ਪੂਰਨ),
                                  ACL_MEM_MALLOC_HUGE_ਪਹਿਲਾ));

            ACL_CHECK(aclrtMemcpyAsync(ctx.rope_cache.position_select_ਸੂਚਕਾਂਕ, theta_scale_ਲੰਮਾਈ * ਆਕਾਰ(ਪੂਰਨ),
                                       ctx.rope_cache.position_select_ਸੂਚਕਾਂਕ_host, theta_scale_ਲੰਮਾਈ * ਆਕਾਰ(ਪੂਰਨ),
                                       ACL_MEMCPY_HOST_ਮੰਜ਼ਿਲ_DEVICE, ctx.stream()));
        }

        position_select_ਸੂਚਕਾਂਕ_tensor = ggml_cann_create_tensor(ctx.rope_cache.position_select_ਸੂਚਕਾਂਕ, ACL_INT32,
                                                               ਆਕਾਰ(ਪੂਰਨ), theta_scale_ne, theta_scale_nb, 1);
    }

    // Step2: divide by freq_factors
    ggml_cann_pool_alloc freq_fac_res_allocator(ctx.pool());
    ਜੇ (src2) {
        freq_fac_res_allocator.alloc(theta_scale_ਲੰਮਾਈ * ਆਕਾਰ(ਦਸ਼ਮਲਵ));
        ਖਾਲੀ *         freq_fac_res_ptr = freq_fac_res_allocator.get();
        acl_tensor_ptr acl_freq_factors_tensor =
            ggml_cann_create_tensor(src2->ਅੰਕੜੇ, ggml_cann_type_mapping(src2->type), ggml_type_ਮਾਪ(src2->type),
                                    theta_scale_ne, theta_scale_nb, GGML_MAX_DIMS);
        acl_tensor_ptr acl_freq_fac_res_tensor = ggml_cann_create_tensor(freq_fac_res_ptr, ACL_FLOAT, ਆਕਾਰ(ਦਸ਼ਮਲਵ),
                                                                         theta_scale_ne, theta_scale_nb, GGML_MAX_DIMS);
        aclnn_div(ctx, acl_theta_scale_tensor.get(), acl_freq_factors_tensor.get(), acl_freq_fac_res_tensor.get());
        std::swap(acl_theta_scale_tensor, acl_freq_fac_res_tensor);
    }

    // Step3: prepare position_tensor
    acl_tensor_ptr       acl_position_tensor;
    ggml_cann_pool_alloc mrope_position_acllocator(ctx.pool());
    ਜੇ (mrope_ਵਰਤਿਆ) {
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

        int64_t        mrope_position_ne[] = { position_ਲੰਮਾਈ, 4 };
        size_t         mrope_position_nb[] = { ਆਕਾਰ(ਪੂਰਨ), position_ਲੰਮਾਈ * ਆਕਾਰ(ਪੂਰਨ) };
        acl_tensor_ptr mrope_position =
            ggml_cann_create_tensor(src1->ਅੰਕੜੇ, ggml_cann_type_mapping(src1->type), ggml_type_ਮਾਪ(src1->type),
                                    mrope_position_ne, mrope_position_nb, 2);

        // selected position tensor's shape is a transpose of cache tensor.
        int64_t selected_position_ne[] = { position_ਲੰਮਾਈ, theta_scale_ਲੰਮਾਈ };
        size_t  selected_position_nb[] = { ਆਕਾਰ(ਦਸ਼ਮਲਵ), position_ਲੰਮਾਈ * ਆਕਾਰ(ਦਸ਼ਮਲਵ) };
        mrope_position_acllocator.alloc(theta_scale_ਲੰਮਾਈ * position_ਲੰਮਾਈ * ਆਕਾਰ(ਦਸ਼ਮਲਵ));
        ਖਾਲੀ * mrope_position_ਬਫਰ = mrope_position_acllocator.get();
        acl_position_tensor =
            ggml_cann_create_tensor(mrope_position_ਬਫਰ, ggml_cann_type_mapping(src1->type),
                                    ggml_type_ਮਾਪ(src1->type), selected_position_ne, selected_position_nb, 2);
        GGML_CANN_CALL_ACLNN_OP(ctx, ਸੂਚਕਾਂਕ_Select, mrope_position.get(), 0, position_select_ਸੂਚਕਾਂਕ_tensor.get(),
                                acl_position_tensor.get());

        // transpose
        int64_t transposed_ne[] = { position_ਲੰਮਾਈ, 1, theta_scale_ਲੰਮਾਈ, 1 };
        size_t  transposed_nb[GGML_MAX_DIMS];
        transposed_nb[0] = ਆਕਾਰ(ਦਸ਼ਮਲਵ);
        ਲਈ (ਪੂਰਨ i = 1; i < GGML_MAX_DIMS; i++) {
            transposed_nb[i] = transposed_nb[i - 1] * transposed_ne[i - 1];
        }

        std::swap(transposed_ne[0], transposed_ne[2]);
        std::swap(transposed_nb[0], transposed_nb[2]);

        acl_position_tensor =
            ggml_cann_create_tensor(mrope_position_ਬਫਰ, ggml_cann_type_mapping(src1->type),
                                    ggml_type_ਮਾਪ(src1->type), transposed_ne, transposed_nb, GGML_MAX_DIMS);

    } ਨਹੀਂਤਾਂ {
        // auto bcast.
        acl_position_tensor =
            ggml_cann_create_tensor(src1->ਅੰਕੜੇ, ggml_cann_type_mapping(src1->type), ggml_type_ਮਾਪ(src1->type),
                                    position_ne, position_nb, GGML_MAX_DIMS);
    }

    // Step4: multiply by the position
    int64_t              theta_ਲੰਮਾਈ = theta_scale_ਲੰਮਾਈ * position_ਲੰਮਾਈ;
    ggml_cann_pool_alloc theta_allocator(ctx.pool(), theta_ਲੰਮਾਈ * ਆਕਾਰ(ਦਸ਼ਮਲਵ));
    ਖਾਲੀ *               theta_ਬਫਰ = theta_allocator.get();

    acl_tensor_ptr acl_theta_tensor =
        ggml_cann_create_tensor(theta_ਬਫਰ, ACL_FLOAT, ਆਕਾਰ(ਦਸ਼ਮਲਵ), cache_ne, cache_nb, GGML_MAX_DIMS);
    aclnn_mul(ctx, acl_position_tensor.get(), acl_theta_scale_tensor.get(), acl_theta_tensor.get());

    // Step5: calculate sin cos.
    // init sin_repeat && cos_repeat, only to accelerate first layer on each device
    ਜੇ (position_ਲੰਮਾਈ > ctx.rope_cache.position_ਲੰਮਾਈ) {
        ctx.rope_cache.position_ਲੰਮਾਈ = position_ਲੰਮਾਈ;
        ਜੇ (ctx.rope_cache.sin_cache != ਖਾਲੀ_ਸੂਚਕ) {
            ACL_CHECK(aclrtFree(ctx.rope_cache.sin_cache));
        }
        ਜੇ (ctx.rope_cache.cos_cache != ਖਾਲੀ_ਸੂਚਕ) {
            ACL_CHECK(aclrtFree(ctx.rope_cache.cos_cache));
        }
        int64_t repeat_theta_ਲੰਮਾਈ = theta_scale_ਲੰਮਾਈ * position_ਲੰਮਾਈ * 2;
        ACL_CHECK(
            aclrtMalloc(&ctx.rope_cache.sin_cache, repeat_theta_ਲੰਮਾਈ * ਆਕਾਰ(ਦਸ਼ਮਲਵ), ACL_MEM_MALLOC_HUGE_ਪਹਿਲਾ));
        ACL_CHECK(
            aclrtMalloc(&ctx.rope_cache.cos_cache, repeat_theta_ਲੰਮਾਈ * ਆਕਾਰ(ਦਸ਼ਮਲਵ), ACL_MEM_MALLOC_HUGE_ਪਹਿਲਾ));
    }

    // sin/cos
    ggml_cann_pool_alloc sin_allocator(ctx.pool(), theta_ਲੰਮਾਈ * ਆਕਾਰ(ਦਸ਼ਮਲਵ));
    ਖਾਲੀ *               sin_ਬਫਰ = sin_allocator.get();
    acl_tensor_ptr       acl_sin_tensor =
        ggml_cann_create_tensor(sin_ਬਫਰ, ACL_FLOAT, ਆਕਾਰ(ਦਸ਼ਮਲਵ), cache_ne, cache_nb, GGML_MAX_DIMS, ACL_FORMAT_ND);
    aclnn_sin(ctx, acl_theta_tensor.get(), acl_sin_tensor.get());

    ggml_cann_pool_alloc cos_allocator(ctx.pool(), theta_ਲੰਮਾਈ * ਆਕਾਰ(ਦਸ਼ਮਲਵ));
    ਖਾਲੀ *               cos_ਬਫਰ = cos_allocator.get();
    acl_tensor_ptr       acl_cos_tensor =
        ggml_cann_create_tensor(cos_ਬਫਰ, ACL_FLOAT, ਆਕਾਰ(ਦਸ਼ਮਲਵ), cache_ne, cache_nb, GGML_MAX_DIMS, ACL_FORMAT_ND);
    aclnn_cos(ctx, acl_theta_tensor.get(), acl_cos_tensor.get());

    ਜੇ (ext_ਗੁਣਕ != 0) {
        attn_ਗੁਣਕ *= 1.0f + 0.1f * logf(1.0f / freq_scale);
    }

    // Step 5: multiply by attn_factor
    ਜੇ (attn_ਗੁਣਕ != 1) {
        aclnn_muls(ctx, acl_sin_tensor.get(), attn_ਗੁਣਕ, ਖਾਲੀ_ਸੂਚਕ, ਸੱਚ);
        aclnn_muls(ctx, acl_cos_tensor.get(), attn_ਗੁਣਕ, ਖਾਲੀ_ਸੂਚਕ, ਸੱਚ);
    }

    int64_t sin_reshape_ne[4] = { rope_dims, 1, dst->ne[2], 1 };
    size_t  sin_reshape_nb[GGML_MAX_DIMS];
    sin_reshape_nb[0] = ਆਕਾਰ(ਦਸ਼ਮਲਵ);
    ਲਈ (ਪੂਰਨ i = 1; i < GGML_MAX_DIMS; i++) {
        sin_reshape_nb[i] = sin_reshape_nb[i - 1] * sin_reshape_ne[i - 1];
    }
    acl_tensor_ptr acl_sin_repeat_tensor = ggml_cann_create_tensor(ctx.rope_cache.sin_cache, ACL_FLOAT, ਆਕਾਰ(ਦਸ਼ਮਲਵ),
                                                                   sin_reshape_ne, sin_reshape_nb, GGML_MAX_DIMS);
    acl_tensor_ptr acl_cos_repeat_tensor = ggml_cann_create_tensor(ctx.rope_cache.cos_cache, ACL_FLOAT, ਆਕਾਰ(ਦਸ਼ਮਲਵ),
                                                                   sin_reshape_ne, sin_reshape_nb, GGML_MAX_DIMS);

    // Step 6: repeat
    ਜੇ (is_neox) {
        // [sinθ1, sinθ1, sinθ2, sinθ2, ..., sinθn, sinθn]
        int64_t repeatsArray[] = { 1, 1, 1, 2 };
        aclnn_repeat(ctx, acl_sin_tensor.get(), acl_sin_repeat_tensor.get(), repeatsArray);
        aclnn_repeat(ctx, acl_cos_tensor.get(), acl_cos_repeat_tensor.get(), repeatsArray);
    } ਨਹੀਂਤਾਂ {
        int64_t ਅੰਕ_repeats = 2;
        int64_t dim         = 3;
        int64_t output_ਮਾਪ = theta_scale_ਲੰਮਾਈ * ਅੰਕ_repeats;
        // [sinθ1, sinθ2, ..., sinθn, sinθ1, sinθ2, ..., sinθn]
        aclnn_repeat_interleave(ctx, acl_sin_tensor.get(), acl_sin_repeat_tensor.get(), dim, ਅੰਕ_repeats, output_ਮਾਪ);
        aclnn_repeat_interleave(ctx, acl_cos_tensor.get(), acl_cos_repeat_tensor.get(), dim, ਅੰਕ_repeats, output_ਮਾਪ);
    }

    // Update cached value.
    ctx.rope_cache.cached = ਸੱਚ;
    ctx.rope_cache.set(theta_scale_ਲੰਮਾਈ, position_ਲੰਮਾਈ, ext_ਗੁਣਕ, theta_scale, freq_scale, attn_ਗੁਣਕ, is_neox,
                       indep_sects, mrope_ਵਰਤਿਆ, is_imrope, sections);
}

#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ __cplusplus
ਬਾਹਰੀ "C" {
#ਜੇ_ਅੰਤ
aclnn_ਸਥਿਤੀ aclnn_Rotary_Position_Embedding_Get_Workspace_ਮਾਪ(ਅਚਲ aclTensor * x,
                                                         ਅਚਲ aclTensor * ਕੋਜੀਵਾ,
                                                         ਅਚਲ aclTensor * ਜੀਵਾ,
                                                         int64_t           mode,
                                                         ਅਚਲ aclTensor * y_ਬਾਹਰਲਾ,
                                                         uint64_t *        workspace_ਮਾਪ,
                                                         aclOpExecutor **  executor);
aclnn_ਸਥਿਤੀ aclnnRotaryPositionEmbedding(ਖਾਲੀ *          workspace,
                                         uint64_t        workspace_ਮਾਪ,
                                         aclOpExecutor * executor,
                                         aclrt_ਧਾਰਾ     stream);
#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ __cplusplus
}
#ਜੇ_ਅੰਤ

ਖਾਲੀ ggml_cann_rope(ggml_backend_cann_context & ctx, ggml_tensor * dst) {
    ggml_tensor * src0 = dst->src[0];  //ਨਿਵੇਸ਼

    // param
    ਦਸ਼ਮਲਵ     freq_ਅਧਾਰ, freq_scale, ext_ਗੁਣਕ, attn_ਗੁਣਕ, beta_fast, beta_slow;
    ਪੂਰਨ       sections[4];
    // const int n_past     = ((int32_t *) dst->op_params)[0];
    ਅਚਲ ਪੂਰਨ n_dims     = ((int32_t *) dst->op_params)[1];
    ਅਚਲ ਪੂਰਨ mode       = ((int32_t *) dst->op_params)[2];
    // const int n_ctx      = ((int32_t *) dst->op_params)[3];
    ਅਚਲ ਪੂਰਨ n_ctx_orig = ((int32_t *) dst->op_params)[4];

    GGML_TENSOR_UNARY_OP_LOCALS

    ਸਮ੍ਰਿਤੀ_ਨਕਲ(&freq_ਅਧਾਰ, (int32_t *) dst->op_params + 5, ਆਕਾਰ(ਦਸ਼ਮਲਵ));
    ਸਮ੍ਰਿਤੀ_ਨਕਲ(&freq_scale, (int32_t *) dst->op_params + 6, ਆਕਾਰ(ਦਸ਼ਮਲਵ));
    ਸਮ੍ਰਿਤੀ_ਨਕਲ(&ext_ਗੁਣਕ, (int32_t *) dst->op_params + 7, ਆਕਾਰ(ਦਸ਼ਮਲਵ));
    ਸਮ੍ਰਿਤੀ_ਨਕਲ(&attn_ਗੁਣਕ, (int32_t *) dst->op_params + 8, ਆਕਾਰ(ਦਸ਼ਮਲਵ));
    ਸਮ੍ਰਿਤੀ_ਨਕਲ(&beta_fast, (int32_t *) dst->op_params + 9, ਆਕਾਰ(ਦਸ਼ਮਲਵ));
    ਸਮ੍ਰਿਤੀ_ਨਕਲ(&beta_slow, (int32_t *) dst->op_params + 10, ਆਕਾਰ(ਦਸ਼ਮਲਵ));
    ਸਮ੍ਰਿਤੀ_ਨਕਲ(&sections, (int32_t *) dst->op_params + 11, ਆਕਾਰ(ਪੂਰਨ) * 4);

    GGML_ASSERT(n_dims % 2 == 0);
    GGML_ASSERT(n_dims <= ne00);

    ਅਚਲ ਦਸ਼ਮਲਵ theta_scale = powf(freq_ਅਧਾਰ, -2.0f / n_dims);

    ਦਸ਼ਮਲਵ corr_dims[2];
    ggml_rope_yarn_corr_dims(n_dims, n_ctx_orig, freq_ਅਧਾਰ, beta_fast, beta_slow, corr_dims);

    ਤਾਰਕਿਕ       is_neox    = mode & GGML_ROPE_TYPE_NEOX;
    ਅਚਲ ਤਾਰਕਿਕ is_imrope  = mode == GGML_ROPE_TYPE_IMROPE;  // qwen3vl apply interleaved mrope
    // mrope_used means the GGML_ROPE_TYPE_MROPE bit is set.
    // Note: this bit is also set for imrope and some vision modes,
    // so mrope_used does NOT exclusively indicate pure mrope.
    ਅਚਲ ਤਾਰਕਿਕ mrope_ਵਰਤਿਆ = mode & GGML_ROPE_TYPE_MROPE;
    ਅਚਲ ਤਾਰਕਿਕ is_vision  = mode == GGML_ROPE_TYPE_VISION;

    ਜੇ (mrope_ਵਰਤਿਆ) {
        GGML_ASSERT(sections[0] > 0 || sections[1] > 0 || sections[2] > 0);
    }

    ਜੇ (is_vision) {
        GGML_ASSERT(n_dims == ne0 / 2);
    }

    ਜੇ (is_imrope || mrope_ਵਰਤਿਆ) {
        is_neox = ਸੱਚ;
    }

    int64_t rope_dims = n_dims;

    //Our current RotaryPositionEmbedding does not support the VISION mode,
    //but essentially it only modifies theta_base in mrope,
    //then repeats it at the end in the same way as is_neox.
    //In fact, RoPE is still applied across all dimensions.
    ਜੇ (is_vision) {
        rope_dims = src0->ne[0];
    }
    int64_t ਪੂਛ_dims = ne00 - rope_dims;
    ਤਾਰਕਿਕ    has_ਪੂਛ  = ਪੂਛ_dims > 0;

    // init ctx.rope_cos/rope_sin cache
    aclnn_rope_cache_init(ctx, dst, corr_dims, ext_ਗੁਣਕ, theta_scale, freq_scale, attn_ਗੁਣਕ, is_neox, sections,
                          mrope_ਵਰਤਿਆ, is_imrope, is_vision, rope_dims);

    // Cache is generated with ne00 dimensions, so we use ne00 for reshape
    int64_t sin_reshape_ne[4] = { rope_dims, 1, ne02, 1 };
    size_t  sin_reshape_nb[GGML_MAX_DIMS];
    sin_reshape_nb[0] = ਆਕਾਰ(ਦਸ਼ਮਲਵ);
    ਲਈ (ਪੂਰਨ i = 1; i < GGML_MAX_DIMS; i++) {
        sin_reshape_nb[i] = sin_reshape_nb[i - 1] * sin_reshape_ne[i - 1];
    }
    acl_tensor_ptr acl_sin_reshape_tensor = ggml_cann_create_tensor(ctx.rope_cache.sin_cache, ACL_FLOAT, ਆਕਾਰ(ਦਸ਼ਮਲਵ),
                                                                    sin_reshape_ne, sin_reshape_nb, GGML_MAX_DIMS);
    acl_tensor_ptr acl_cos_reshape_tensor = ggml_cann_create_tensor(ctx.rope_cache.cos_cache, ACL_FLOAT, ਆਕਾਰ(ਦਸ਼ਮਲਵ),
                                                                    sin_reshape_ne, sin_reshape_nb, GGML_MAX_DIMS);

    acl_tensor_ptr acl_src = ggml_cann_create_tensor(src0);
    acl_tensor_ptr acl_dst = ggml_cann_create_tensor(dst);
#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ ASCEND_310P
    // Special ROPE operation for 310P

    // roll input
    ਖਾਲੀ *               ਨਿਵੇਸ਼_roll_ਬਫਰ;
    acl_tensor_ptr       acl_minus_one_tensor;
    ਖਾਲੀ *               minus_one_scale_ਬਫਰ = ਖਾਲੀ_ਸੂਚਕ;
    ggml_cann_pool_alloc roll_allocator(ctx.pool(), ggml_nbytes(src0));
    ggml_cann_pool_alloc minus_one_scale_allocator(ctx.pool(), ਆਕਾਰ(ਦਸ਼ਮਲਵ) * src0->ne[0]);
    ਜੇ (!is_neox) {
        // roll input: [q0,q1,q2,q3,...] -> [q1,q0,q3,q2,...]
        ਨਿਵੇਸ਼_roll_ਬਫਰ        = roll_allocator.get();
        int64_t ਨਿਵੇਸ਼_roll_ne[4] = { 2, src0->ne[1] * (src0->ne[0] / 2), src0->ne[2], src0->ne[3] };
        size_t  ਨਿਵੇਸ਼_roll_nb[GGML_MAX_DIMS];
        ਨਿਵੇਸ਼_roll_nb[0] = ggml_type_ਮਾਪ(src0->type);
        ਲਈ (ਪੂਰਨ i = 1; i < GGML_MAX_DIMS; i++) {
            ਨਿਵੇਸ਼_roll_nb[i] = ਨਿਵੇਸ਼_roll_nb[i - 1] * ਨਿਵੇਸ਼_roll_ne[i - 1];
        }
        acl_tensor_ptr acl_ਨਿਵੇਸ਼_roll_tensor =
            ggml_cann_create_tensor(ਨਿਵੇਸ਼_roll_ਬਫਰ, ggml_cann_type_mapping(src0->type), ggml_type_ਮਾਪ(src0->type),
                                    ਨਿਵੇਸ਼_roll_ne, ਨਿਵੇਸ਼_roll_nb, GGML_MAX_DIMS);
        acl_tensor_ptr acl_ਨਿਵੇਸ਼_tensor =
            ggml_cann_create_tensor(src0->ਅੰਕੜੇ, ggml_cann_type_mapping(src0->type), ggml_type_ਮਾਪ(src0->type),
                                    ਨਿਵੇਸ਼_roll_ne, ਨਿਵੇਸ਼_roll_nb, GGML_MAX_DIMS);

        int64_t shifts[] = { 1 };
        int64_t dims[]   = { 3 };
        aclnn_roll(ctx, acl_ਨਿਵੇਸ਼_tensor.get(), acl_ਨਿਵੇਸ਼_roll_tensor.get(), shifts, dims);

        // init [-1, 1, -1, 1, ...]
        minus_one_scale_ਬਫਰ = minus_one_scale_allocator.get();

        int64_t minus_one_ne[4] = { src0->ne[0], 1, 1, 1 };
        size_t  minus_one_nb[GGML_MAX_DIMS];
        minus_one_nb[0] = ਆਕਾਰ(ਦਸ਼ਮਲਵ);
        ਲਈ (ਪੂਰਨ i = 1; i < GGML_MAX_DIMS; i++) {
            minus_one_nb[i] = minus_one_nb[i - 1] * minus_one_ne[i - 1];
        }
        acl_minus_one_tensor = aclnn_ਮੁੱਲਾਂ(ctx, minus_one_scale_ਬਫਰ, ਆਕਾਰ(ਦਸ਼ਮਲਵ) * src0->ne[0], minus_one_ne,
                                            GGML_MAX_DIMS, ACL_FLOAT, ਆਕਾਰ(ਦਸ਼ਮਲਵ), 1);
        int64_t   dim        = 3;
        int64_t * ਸੂਚਕਾਂਕ      = ਨਵਾਂ int64_t[src0->ne[0]];
        ਲਈ (ਪੂਰਨ i = 0; i < src0->ne[0]; i++) {
            ਸੂਚਕਾਂਕ[i] = i / 2 * 2;
        }
        int64_t ਸੂਚਕਾਂਕ_ਅੰਕ = src0->ne[0];
        ਦਸ਼ਮਲਵ   value     = -1;
        aclnn_ਸੂਚਕਾਂਕ_fill_tensor(ctx, acl_minus_one_tensor.get(), dim, ਸੂਚਕਾਂਕ, ਸੂਚਕਾਂਕ_ਅੰਕ, value);
    } ਨਹੀਂਤਾਂ {
        // roll input: [q0,q1,q2,...] ->
        // [q_half,q_half+1,...,q_end,q0,q1,...q_half-1]
        ਨਿਵੇਸ਼_roll_ਬਫਰ = roll_allocator.get();
        acl_tensor_ptr acl_ਨਿਵੇਸ਼_roll_tensor =
            ggml_cann_create_tensor(ਨਿਵੇਸ਼_roll_ਬਫਰ, ggml_cann_type_mapping(src0->type), ggml_type_ਮਾਪ(src0->type),
                                    src0->ne, src0->nb, GGML_MAX_DIMS);
        acl_tensor_ptr acl_ਨਿਵੇਸ਼_tensor = ggml_cann_create_tensor(src0);

        int64_t shifts[] = { src0->ne[0] / 2 };
        int64_t dims[]   = { 3 };
        aclnn_roll(ctx, acl_ਨਿਵੇਸ਼_tensor.get(), acl_ਨਿਵੇਸ਼_roll_tensor.get(), shifts, dims);

        // init [-1, -1, -1, 1, 1，1，...]
        minus_one_scale_ਬਫਰ  = minus_one_scale_allocator.get();
        int64_t minus_one_ne[4] = { src0->ne[0], 1, 1, 1 };
        size_t  minus_one_nb[GGML_MAX_DIMS];
        minus_one_nb[0] = ਆਕਾਰ(ਦਸ਼ਮਲਵ);
        ਲਈ (ਪੂਰਨ i = 1; i < GGML_MAX_DIMS; i++) {
            minus_one_nb[i] = minus_one_nb[i - 1] * minus_one_ne[i - 1];
        }
        acl_minus_one_tensor     = aclnn_ਮੁੱਲਾਂ(ctx, minus_one_scale_ਬਫਰ, ਆਕਾਰ(ਦਸ਼ਮਲਵ) * src0->ne[0], minus_one_ne,
                                                GGML_MAX_DIMS, ACL_FLOAT, ਆਕਾਰ(ਦਸ਼ਮਲਵ), 1);
        // -1 * first half
        int64_t ਪਹਿਲਾ_half_ne[4] = { src0->ne[0] / 2, 1, 1, 1 };
        size_t  ਪਹਿਲਾ_half_nb[GGML_MAX_DIMS];
        ਪਹਿਲਾ_half_nb[0] = ਆਕਾਰ(ਦਸ਼ਮਲਵ);
        ਲਈ (ਪੂਰਨ i = 1; i < GGML_MAX_DIMS; i++) {
            ਪਹਿਲਾ_half_nb[i] = ਪਹਿਲਾ_half_nb[i - 1] * ਪਹਿਲਾ_half_ne[i - 1];
        }
        acl_tensor_ptr acl_ਪਹਿਲਾ_half_tensor = ggml_cann_create_tensor(minus_one_scale_ਬਫਰ, ACL_FLOAT, ਆਕਾਰ(ਦਸ਼ਮਲਵ),
                                                                       ਪਹਿਲਾ_half_ne, ਪਹਿਲਾ_half_nb, GGML_MAX_DIMS);
        ਤਾਰਕਿਕ           inplace               = ਸੱਚ;
        ਦਸ਼ਮਲਵ          scale                 = -1;
        aclnn_muls(ctx, acl_ਪਹਿਲਾ_half_tensor.get(), scale, ਖਾਲੀ_ਸੂਚਕ, inplace);
    }

    // TODO: n_dims < ne0
    GGML_ASSERT(n_dims == src0->ne[0]);

    // input * scale
    ggml_cann_pool_alloc roll_mul_scale_allocator(ctx.pool(), ggml_nbytes(src0));
    ਖਾਲੀ *               ਨਿਵੇਸ਼_roll_mul_scale_ਬਫਰ = roll_mul_scale_allocator.get();
    size_t               ਨਿਵੇਸ਼_nb[GGML_MAX_DIMS];
    ਨਿਵੇਸ਼_nb[0] = ggml_type_ਮਾਪ(src0->type);
    ਲਈ (ਪੂਰਨ i = 1; i < GGML_MAX_DIMS; i++) {
        ਨਿਵੇਸ਼_nb[i] = ਨਿਵੇਸ਼_nb[i - 1] * src0->ne[i - 1];
    }
    acl_tensor_ptr acl_ਨਿਵੇਸ਼_roll_mul_scale_tensor =
        ggml_cann_create_tensor(ਨਿਵੇਸ਼_roll_mul_scale_ਬਫਰ, ggml_cann_type_mapping(src0->type),
                                ggml_type_ਮਾਪ(src0->type), src0->ne, ਨਿਵੇਸ਼_nb, GGML_MAX_DIMS);
    acl_tensor_ptr acl_ਨਿਵੇਸ਼_roll_reshape_tensor =
        ggml_cann_create_tensor(ਨਿਵੇਸ਼_roll_ਬਫਰ, ggml_cann_type_mapping(src0->type), ggml_type_ਮਾਪ(src0->type),
                                src0->ne, ਨਿਵੇਸ਼_nb, GGML_MAX_DIMS);

    aclnn_mul(ctx, acl_ਨਿਵੇਸ਼_roll_reshape_tensor.get(), acl_minus_one_tensor.get(),
              acl_ਨਿਵੇਸ਼_roll_mul_scale_tensor.get());

    // output
    ਖਾਲੀ * output_fp32_ਬਫਰ;
    ਜੇ (src0->type == GGML_TYPE_F32) {
        aclnn_mul(ctx, acl_src.get(), acl_cos_reshape_tensor.get());
        aclnn_mul(ctx, acl_ਨਿਵੇਸ਼_roll_mul_scale_tensor.get(), acl_sin_reshape_tensor.get());
        aclnn_ਜੋੜੋ(ctx, acl_src.get(), acl_ਨਿਵੇਸ਼_roll_mul_scale_tensor.get(), acl_dst.get());
        // TODO: ne0 != n_dims in mode2
    } ਨਹੀਂਤਾਂ ਜੇ (src0->type == GGML_TYPE_F16) {
        size_t ਨਿਵੇਸ਼_fp32_nb[GGML_MAX_DIMS];
        ਨਿਵੇਸ਼_fp32_nb[0] = ਆਕਾਰ(ਦਸ਼ਮਲਵ);
        ਲਈ (ਪੂਰਨ i = 1; i < GGML_MAX_DIMS; i++) {
            ਨਿਵੇਸ਼_fp32_nb[i] = ਨਿਵੇਸ਼_fp32_nb[i - 1] * dst->ne[i - 1];
        }
        ggml_cann_pool_alloc fp32_allocator1(ctx.pool(), ggml_nelements(dst) * ਆਕਾਰ(ਦਸ਼ਮਲਵ));
        ਖਾਲੀ *               ਨਿਵੇਸ਼_fp32_buffer1 = fp32_allocator1.get();
        acl_tensor_ptr       ਨਿਵੇਸ਼_fp32_tensor1 = ggml_cann_create_tensor(ਨਿਵੇਸ਼_fp32_buffer1, ACL_FLOAT, ਆਕਾਰ(ਦਸ਼ਮਲਵ),
                                                                          dst->ne, ਨਿਵੇਸ਼_fp32_nb, GGML_MAX_DIMS);
        ggml_cann_pool_alloc fp32_allocator2(ctx.pool(), ggml_nelements(dst) * ਆਕਾਰ(ਦਸ਼ਮਲਵ));
        ਖਾਲੀ *               ਨਿਵੇਸ਼_fp32_buffer2 = fp32_allocator2.get();
        acl_tensor_ptr       ਨਿਵੇਸ਼_fp32_tensor2 = ggml_cann_create_tensor(ਨਿਵੇਸ਼_fp32_buffer2, ACL_FLOAT, ਆਕਾਰ(ਦਸ਼ਮਲਵ),
                                                                          dst->ne, ਨਿਵੇਸ਼_fp32_nb, GGML_MAX_DIMS);

        ggml_cann_pool_alloc fp32_allocator(ctx.pool(), ggml_nelements(dst) * ਆਕਾਰ(ਦਸ਼ਮਲਵ));
        output_fp32_ਬਫਰ                = fp32_allocator.get();
        acl_tensor_ptr output_fp32_tensor = ggml_cann_create_tensor(output_fp32_ਬਫਰ, ACL_FLOAT, ਆਕਾਰ(ਦਸ਼ਮਲਵ),
                                                                    dst->ne, ਨਿਵੇਸ਼_fp32_nb, GGML_MAX_DIMS);
        aclnn_mul(ctx, acl_src.get(), acl_cos_reshape_tensor.get(), ਨਿਵੇਸ਼_fp32_tensor1.get());
        aclnn_mul(ctx, acl_ਨਿਵੇਸ਼_roll_mul_scale_tensor.get(), acl_sin_reshape_tensor.get(), ਨਿਵੇਸ਼_fp32_tensor2.get());
        aclnn_ਜੋੜੋ(ctx, ਨਿਵੇਸ਼_fp32_tensor1.get(), ਨਿਵੇਸ਼_fp32_tensor2.get(), output_fp32_tensor.get());
        aclnn_cast(ctx, output_fp32_tensor.get(), acl_dst.get(), ACL_FLOAT16);
    }
    ਵਾਪਸ;
#ਜੇ_ਅੰਤ
    int64_t acl_ਬਹੁਲਕ = is_neox ? 0 : 1;

    // Pre-define head and tail dimensions for reuse
    int64_t ਸਿਰ_ne[GGML_MAX_DIMS] = { rope_dims, ne01, ne02, ne03 };
    int64_t ਪੂਛ_ne[GGML_MAX_DIMS] = { ਪੂਛ_dims, ne01, ne02, ne03 };

    // Step 1: Prepare trans tensors for F16 type conversion to F32 if needed
    ਤਾਰਕਿਕ                 src_dst_need_trans = ਝੂਠ;
    ggml_cann_pool_alloc src_trans_allocator(ctx.pool());
    ggml_cann_pool_alloc dst_trans_allocator(ctx.pool());
    acl_tensor_ptr       acl_src_trans_tensor;
    acl_tensor_ptr       acl_dst_trans_tensor;
    ਖਾਲੀ *               src_trans_ਬਫਰ = ਖਾਲੀ_ਸੂਚਕ;
    ਖਾਲੀ *               dst_trans_ਬਫਰ = ਖਾਲੀ_ਸੂਚਕ;
    size_t               src_dst_trans_nb[GGML_MAX_DIMS];
    ਜੇ (src0->type == GGML_TYPE_F16) {
        src_dst_need_trans = ਸੱਚ;
        src_trans_ਬਫਰ   = src_trans_allocator.alloc(ggml_nelements(src0) * ਆਕਾਰ(ਦਸ਼ਮਲਵ));
        dst_trans_ਬਫਰ   = dst_trans_allocator.alloc(ggml_nelements(dst) * ਆਕਾਰ(ਦਸ਼ਮਲਵ));

        src_dst_trans_nb[0] = ਆਕਾਰ(ਦਸ਼ਮਲਵ);
        ਲਈ (ਪੂਰਨ i = 1; i < GGML_MAX_DIMS; i++) {
            src_dst_trans_nb[i] = src_dst_trans_nb[i - 1] * src0->ne[i - 1];
        }
        acl_src_trans_tensor = ggml_cann_create_tensor(src_trans_ਬਫਰ, ACL_FLOAT, ਆਕਾਰ(ਦਸ਼ਮਲਵ), src0->ne,
                                                       src_dst_trans_nb, GGML_MAX_DIMS);
        acl_dst_trans_tensor = ggml_cann_create_tensor(dst_trans_ਬਫਰ, ACL_FLOAT, ਆਕਾਰ(ਦਸ਼ਮਲਵ), dst->ne,
                                                       src_dst_trans_nb, GGML_MAX_DIMS);
        aclnn_cast(ctx, acl_src.get(), acl_src_trans_tensor.get(), ACL_FLOAT);
    }

    // Step 2: Prepare head tensors for tail splitting if needed
    acl_tensor_ptr acl_src_ਸਿਰ;
    acl_tensor_ptr acl_dst_ਸਿਰ;
    ਜੇ (has_ਪੂਛ) {
        // Create head views for RotaryPositionEmbedding (only first rope_dims dimensions)
        // RotaryPositionEmbedding requires contiguous dst tensor, so we use a temporary buffer
        ਜੇ (src_dst_need_trans) {
            // Use F32 trans tensor strides
            acl_src_ਸਿਰ = ggml_cann_create_tensor((ਅੱਖਰ *) src_trans_ਬਫਰ, ACL_FLOAT, ਆਕਾਰ(ਦਸ਼ਮਲਵ), ਸਿਰ_ne,
                                                   src_dst_trans_nb, GGML_MAX_DIMS);
        } ਨਹੀਂਤਾਂ {
            // Use original F32 tensor strides
            acl_src_ਸਿਰ = ggml_cann_create_tensor((ਅੱਖਰ *) src0->ਅੰਕੜੇ, ACL_FLOAT, ਆਕਾਰ(ਦਸ਼ਮਲਵ), ਸਿਰ_ne, src0->nb,
                                                   GGML_MAX_DIMS);
        }

        int64_t              ਸਿਰ_elements = rope_dims * ne01 * ne02 * ne03;
        ggml_cann_pool_alloc dst_ਸਿਰ_contiguous_allocator(ctx.pool(), ਸਿਰ_elements * ਆਕਾਰ(ਦਸ਼ਮਲਵ));
        ਖਾਲੀ *               dst_ਸਿਰ_contiguous_ਬਫਰ = dst_ਸਿਰ_contiguous_allocator.get();

        size_t ਸਿਰ_contiguous_nb[GGML_MAX_DIMS];
        ਸਿਰ_contiguous_nb[0] = ਆਕਾਰ(ਦਸ਼ਮਲਵ);
        ਲਈ (ਪੂਰਨ i = 1; i < GGML_MAX_DIMS; i++) {
            ਸਿਰ_contiguous_nb[i] = ਸਿਰ_contiguous_nb[i - 1] * ਸਿਰ_ne[i - 1];
        }
        acl_dst_ਸਿਰ = ggml_cann_create_tensor(dst_ਸਿਰ_contiguous_ਬਫਰ, ACL_FLOAT, ਆਕਾਰ(ਦਸ਼ਮਲਵ), ਸਿਰ_ne,
                                               ਸਿਰ_contiguous_nb, GGML_MAX_DIMS);
    }

    // Step 3: Execute RotaryPositionEmbedding
    ਜੇ (has_ਪੂਛ) {
        // Rotate only the head portion (first rope_dims dimensions)
        GGML_CANN_CALL_ACLNN_OP(ctx, RotaryPositionEmbedding, acl_src_ਸਿਰ.get(), acl_cos_reshape_tensor.get(),
                                acl_sin_reshape_tensor.get(), acl_ਬਹੁਲਕ, acl_dst_ਸਿਰ.get());

        // Copy head result from contiguous buffer back to destination tensor
        ਜੇ (src_dst_need_trans) {
            acl_tensor_ptr acl_dst_ਸਿਰ_ਨਿਸ਼ਾਨਾ = ggml_cann_create_tensor(
                (ਅੱਖਰ *) dst_trans_ਬਫਰ, ACL_FLOAT, ਆਕਾਰ(ਦਸ਼ਮਲਵ), ਸਿਰ_ne, src_dst_trans_nb, GGML_MAX_DIMS);
            cann_copy(ctx, acl_dst_ਸਿਰ.get(), acl_dst_ਸਿਰ_ਨਿਸ਼ਾਨਾ.get());
        } ਨਹੀਂਤਾਂ {
            acl_tensor_ptr acl_dst_ਸਿਰ_ਨਿਸ਼ਾਨਾ =
                ggml_cann_create_tensor((ਅੱਖਰ *) dst->ਅੰਕੜੇ, ACL_FLOAT, ਆਕਾਰ(ਦਸ਼ਮਲਵ), ਸਿਰ_ne, dst->nb, GGML_MAX_DIMS);
            cann_copy(ctx, acl_dst_ਸਿਰ.get(), acl_dst_ਸਿਰ_ਨਿਸ਼ਾਨਾ.get());
        }
    } ਨਹੀਂਤਾਂ ਜੇ (src_dst_need_trans) {
        // Rotate full tensor (no tail), using trans tensors
        GGML_CANN_CALL_ACLNN_OP(ctx, RotaryPositionEmbedding, acl_src_trans_tensor.get(), acl_cos_reshape_tensor.get(),
                                acl_sin_reshape_tensor.get(), acl_ਬਹੁਲਕ, acl_dst_trans_tensor.get());
    } ਨਹੀਂਤਾਂ ਜੇ (src0->ਅੰਕੜੇ == dst->ਅੰਕੜੇ && !ggml_is_contiguous(src0)) {
        // In-place on non-contiguous tensor: RotaryPositionEmbedding cannot safely
        // read and write the same non-contiguous buffer. Use contiguous temporaries.
        size_t contiguous_nb[GGML_MAX_DIMS];
        contiguous_nb[0] = ਆਕਾਰ(ਦਸ਼ਮਲਵ);
        ਲਈ (ਪੂਰਨ i = 1; i < GGML_MAX_DIMS; i++) {
            contiguous_nb[i] = contiguous_nb[i - 1] * src0->ne[i - 1];
        }
        int64_t              ਕੁੱਲ_elements = ggml_nelements(src0);
        ggml_cann_pool_alloc inplace_src_alloc(ctx.pool(), ਕੁੱਲ_elements * ਆਕਾਰ(ਦਸ਼ਮਲਵ));
        ggml_cann_pool_alloc inplace_dst_alloc(ctx.pool(), ਕੁੱਲ_elements * ਆਕਾਰ(ਦਸ਼ਮਲਵ));

        acl_tensor_ptr acl_src_contig = ggml_cann_create_tensor(inplace_src_alloc.get(), ACL_FLOAT, ਆਕਾਰ(ਦਸ਼ਮਲਵ),
                                                                src0->ne, contiguous_nb, GGML_MAX_DIMS);
        acl_tensor_ptr acl_dst_contig = ggml_cann_create_tensor(inplace_dst_alloc.get(), ACL_FLOAT, ਆਕਾਰ(ਦਸ਼ਮਲਵ),
                                                                dst->ne, contiguous_nb, GGML_MAX_DIMS);

        cann_copy(ctx, acl_src.get(), acl_src_contig.get());
        GGML_CANN_CALL_ACLNN_OP(ctx, RotaryPositionEmbedding, acl_src_contig.get(), acl_cos_reshape_tensor.get(),
                                acl_sin_reshape_tensor.get(), acl_ਬਹੁਲਕ, acl_dst_contig.get());
        cann_copy(ctx, acl_dst_contig.get(), acl_dst.get());
    } ਨਹੀਂਤਾਂ {
        // Rotate full tensor (no tail), using original tensors
        GGML_CANN_CALL_ACLNN_OP(ctx, RotaryPositionEmbedding, acl_src.get(), acl_cos_reshape_tensor.get(),
                                acl_sin_reshape_tensor.get(), acl_ਬਹੁਲਕ, acl_dst.get());
    }

    // Step 4: Copy unrotated tail portion from source to destination
    ਜੇ (has_ਪੂਛ) {
        size_t src_ਪੂਛ_ਵਿੱਥ;
        size_t dst_ਪੂਛ_ਵਿੱਥ;

        ਸਵੈਚਾਲਿਤ copy_ਪੂਛ_device = [&](ਖਾਲੀ * src_ptr, ਖਾਲੀ * dst_ptr, acl_ਅੰਕੜੇ_Type dtype, size_t elem_ਮਾਪ,
                                    size_t * nb_src_arr, size_t * nb_dst_arr) {
            acl_tensor_ptr acl_src_ਪੂਛ =
                ggml_cann_create_tensor(src_ptr, dtype, elem_ਮਾਪ, ਪੂਛ_ne, nb_src_arr, GGML_MAX_DIMS);
            acl_tensor_ptr acl_dst_ਪੂਛ =
                ggml_cann_create_tensor(dst_ptr, dtype, elem_ਮਾਪ, ਪੂਛ_ne, nb_dst_arr, GGML_MAX_DIMS);
            cann_copy(ctx, acl_src_ਪੂਛ.get(), acl_dst_ਪੂਛ.get());
        };

        ਜੇ (src_dst_need_trans) {
            // Use F32 trans tensor strides and offsets
            src_ਪੂਛ_ਵਿੱਥ = rope_dims * src_dst_trans_nb[0];
            dst_ਪੂਛ_ਵਿੱਥ = rope_dims * src_dst_trans_nb[0];
            copy_ਪੂਛ_device((ਅੱਖਰ *) src_trans_ਬਫਰ + src_ਪੂਛ_ਵਿੱਥ, (ਅੱਖਰ *) dst_trans_ਬਫਰ + dst_ਪੂਛ_ਵਿੱਥ,
                             ACL_FLOAT, ਆਕਾਰ(ਦਸ਼ਮਲਵ), src_dst_trans_nb, src_dst_trans_nb);
        } ਨਹੀਂਤਾਂ {
            // Use original tensor strides and offsets
            src_ਪੂਛ_ਵਿੱਥ = rope_dims * nb00;
            dst_ਪੂਛ_ਵਿੱਥ = rope_dims * nb0;
            copy_ਪੂਛ_device((ਅੱਖਰ *) src0->ਅੰਕੜੇ + src_ਪੂਛ_ਵਿੱਥ, (ਅੱਖਰ *) dst->ਅੰਕੜੇ + dst_ਪੂਛ_ਵਿੱਥ,
                             ggml_cann_type_mapping(dst->type), ggml_element_ਮਾਪ(dst), src0->nb, dst->nb);
        }
    }

    // Step 5: Cast back to F16 if needed
    ਜੇ (src_dst_need_trans) {
        aclnn_cast(ctx, acl_dst_trans_tensor.get(), acl_dst.get(), ACL_FLOAT16);
    }
}

ਖਾਲੀ ggml_cann_rope_cache_preload(ggml_backend_cann_context & ctx, ggml_tensor * dst) {
    ggml_tensor * src0 = dst->src[0];

    ਦਸ਼ਮਲਵ     freq_ਅਧਾਰ, freq_scale, ext_ਗੁਣਕ, attn_ਗੁਣਕ, beta_fast, beta_slow;
    ਪੂਰਨ       sections[4];
    ਅਚਲ ਪੂਰਨ n_dims     = ((int32_t *) dst->op_params)[1];
    ਅਚਲ ਪੂਰਨ mode       = ((int32_t *) dst->op_params)[2];
    ਅਚਲ ਪੂਰਨ n_ctx_orig = ((int32_t *) dst->op_params)[4];

    GGML_TENSOR_UNARY_OP_LOCALS

    ਸਮ੍ਰਿਤੀ_ਨਕਲ(&freq_ਅਧਾਰ, (int32_t *) dst->op_params + 5, ਆਕਾਰ(ਦਸ਼ਮਲਵ));
    ਸਮ੍ਰਿਤੀ_ਨਕਲ(&freq_scale, (int32_t *) dst->op_params + 6, ਆਕਾਰ(ਦਸ਼ਮਲਵ));
    ਸਮ੍ਰਿਤੀ_ਨਕਲ(&ext_ਗੁਣਕ, (int32_t *) dst->op_params + 7, ਆਕਾਰ(ਦਸ਼ਮਲਵ));
    ਸਮ੍ਰਿਤੀ_ਨਕਲ(&attn_ਗੁਣਕ, (int32_t *) dst->op_params + 8, ਆਕਾਰ(ਦਸ਼ਮਲਵ));
    ਸਮ੍ਰਿਤੀ_ਨਕਲ(&beta_fast, (int32_t *) dst->op_params + 9, ਆਕਾਰ(ਦਸ਼ਮਲਵ));
    ਸਮ੍ਰਿਤੀ_ਨਕਲ(&beta_slow, (int32_t *) dst->op_params + 10, ਆਕਾਰ(ਦਸ਼ਮਲਵ));
    ਸਮ੍ਰਿਤੀ_ਨਕਲ(&sections, (int32_t *) dst->op_params + 11, ਆਕਾਰ(ਪੂਰਨ) * 4);

    ਅਚਲ ਦਸ਼ਮਲਵ theta_scale = powf(freq_ਅਧਾਰ, -2.0f / n_dims);

    ਦਸ਼ਮਲਵ corr_dims[2];
    ggml_rope_yarn_corr_dims(n_dims, n_ctx_orig, freq_ਅਧਾਰ, beta_fast, beta_slow, corr_dims);

    ਤਾਰਕਿਕ       is_neox    = mode & GGML_ROPE_TYPE_NEOX;
    ਅਚਲ ਤਾਰਕਿਕ is_imrope  = mode == GGML_ROPE_TYPE_IMROPE;
    ਅਚਲ ਤਾਰਕਿਕ mrope_ਵਰਤਿਆ = mode & GGML_ROPE_TYPE_MROPE;
    ਅਚਲ ਤਾਰਕਿਕ is_vision  = mode == GGML_ROPE_TYPE_VISION;

    ਜੇ (is_imrope || mrope_ਵਰਤਿਆ) {
        is_neox = ਸੱਚ;
    }

    int64_t rope_dims = n_dims;
    ਜੇ (is_vision) {
        rope_dims = src0->ne[0];
    }

    // Run the full cache init on the non-captured stream.  This performs all
    // host-to-device memcpy, aclrtMalloc/Free, and on-device computations
    // so that the memory pool is warmed up and cache metadata is populated.
    aclnn_rope_cache_init(ctx, dst, corr_dims, ext_ਗੁਣਕ, theta_scale, freq_scale, attn_ਗੁਣਕ, is_neox, sections,
                          mrope_ਵਰਤਿਆ, is_imrope, is_vision, rope_dims);

    // Reset `cached` so that during graph capture the on-device computations
    // (sin/cos, position multiply, repeat, etc.) still execute and get recorded
    // into the captured graph.  The cache metadata (theta_scale_length,
    // theta_scale, sections, position_length, etc.) remains set, which causes
    // all host-to-device copy and malloc/free branches to be skipped.
    ctx.rope_cache.cached = ਝੂਠ;
}

ਖਾਲੀ ggml_cann_argmax(ggml_backend_cann_context & ctx, ggml_tensor * dst) {
    ggml_tensor * src0 = dst->src[0];

    acl_tensor_ptr acl_src = ggml_cann_create_tensor(src0);
    acl_tensor_ptr acl_dst = ggml_cann_create_tensor(dst, dst->ne, dst->nb, 3);

    GGML_CANN_CALL_ACLNN_OP(ctx, ArgMax, acl_src.get(), 3, ਝੂਠ, acl_dst.get());
}

ਖਾਲੀ ggml_cann_conv_transpose_1d(ggml_backend_cann_context & ctx, ggml_tensor * dst) {
    ggml_tensor * src0 = dst->src[0];
    ggml_tensor * src1 = dst->src[1];

    // stride
    int64_t s0 = ((ਅਚਲ int32_t *) (dst->op_params))[0];

    acl_tensor_ptr acl_ਨਿਵੇਸ਼  = ggml_cann_create_tensor(src1, src1->ne, src1->nb, 3, ACL_FORMAT_NCL);
    acl_tensor_ptr acl_ਭਾਰ = ggml_cann_create_tensor(src0, src0->ne, src0->nb, 3, ACL_FORMAT_NCL);
    acl_tensor_ptr acl_dst    = ggml_cann_create_tensor(dst, dst->ne, dst->nb, 3, ACL_FORMAT_NCL);

    // get base information of input and kernel
    int64_t ਨਿਵੇਸ਼_len   = *(src1->ne);
    int64_t dst_len     = *(dst->ne);
    int64_t kernel_ਮਾਪ = *(src0->ne);

    // set the max kernel size for each conv
    int64_t max_kernel_ਮਾਪ = 255;

    // compute the partition of kernel
    int64_t part_ਅੰਕ = 1;
    part_ਅੰਕ         = (kernel_ਮਾਪ + max_kernel_ਮਾਪ - 1) / max_kernel_ਮਾਪ;

    int64_t strideVal[1];
    strideVal[0]                    = s0;
    acl_int_array_ptr stride        = ggml_cann_create_int_array(strideVal, 1);
    int64_t           paddingVal[]  = { 0 };
    acl_int_array_ptr padding       = ggml_cann_create_int_array(paddingVal, 1);
    int64_t           dilationVal[] = { 1 };
    acl_int_array_ptr dilation      = ggml_cann_create_int_array(dilationVal, 1);
    ਤਾਰਕਿਕ              transposed    = ਸੱਚ;
    int64_t           groups        = 1;
    int8_t            cubeMathType  = 0;

#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ ASCEND_310P
    cubeMathType = 1;
#ਜੇ_ਅੰਤ

    ਸਵੈਚਾਲਿਤ ਭਾਰ_type = ggml_cann_type_mapping(src0->type);
    ਸਵੈਚਾਲਿਤ dst_type    = ggml_cann_type_mapping(dst->type);

    // slice the kernel to make each conv available
    int64_t ਟੁਕੜਾ_dim   = -1;
    int64_t ਟੁਕੜਾ_ਸ਼ੁਰੂ = 0;
    int64_t ਟੁਕੜਾ_ਸਿਰਾ   = max_kernel_ਮਾਪ;
    int64_t ਟੁਕੜਾ_ਕਦਮ  = 1;
    int64_t interval    = max_kernel_ਮਾਪ;

    int64_t ਖੱਬਾ_pad_len  = dilationVal[0] * (max_kernel_ਮਾਪ - 1) + 1 - 2 * paddingVal[0];
    int64_t ਸੱਜਾ_pad_len = 0;

    acl_scalar_ptr alpha      = ਖਾਲੀ_ਸੂਚਕ;
    ਦਸ਼ਮਲਵ          alpha_ਮੁੱਲ = 1.0;
    alpha                     = ggml_cann_create_scalar(&alpha_ਮੁੱਲ, acl_ਅੰਕੜੇ_Type::ACL_FLOAT);

    // set zero to destination
    GGML_CANN_CALL_ACLNN_OP(ctx, InplaceZero, acl_dst.get());

    ਲਈ (ਪੂਰਨ k = 0; k < part_ਅੰਕ; k++) {
        // create part kernel tensor and slice from big kernel
        ਟੁਕੜਾ_ਸ਼ੁਰੂ = max_kernel_ਮਾਪ * k;
        ਜੇ (k == part_ਅੰਕ - 1) {
            ਟੁਕੜਾ_ਸਿਰਾ = kernel_ਮਾਪ;
            interval  = kernel_ਮਾਪ - max_kernel_ਮਾਪ * k;
        } ਨਹੀਂਤਾਂ {
            ਟੁਕੜਾ_ਸਿਰਾ = max_kernel_ਮਾਪ * (k + 1);
        }

        int64_t part_ne[4];
        ਲਈ (ਪੂਰਨ i = 0; i < 4; i++) {
            part_ne[i] = *(src0->ne + i);
        }
        part_ne[0] = interval;

        size_t part_nb[4];
        part_nb[0] = ਆਕਾਰ(ਭਾਰ_type);
        ਲਈ (ਪੂਰਨ i = 1; i < 4; i++) {
            part_nb[i] = part_nb[i - 1] * part_ne[i - 1];
        }

        ggml_cann_pool_alloc part_kernel_allocator;
        part_kernel_allocator.alloc(ctx.pool(), part_nb[3]);
        ਖਾਲੀ * part_kernel_buf = part_kernel_allocator.get();

        acl_tensor_ptr part_kernel = ggml_cann_create_tensor(part_kernel_buf, ਭਾਰ_type, ggml_element_ਮਾਪ(src0),
                                                             part_ne, part_nb, 3, ACL_FORMAT_NCL);

        GGML_CANN_CALL_ACLNN_OP(ctx, ਟੁਕੜਾ, acl_ਭਾਰ.get(), ਟੁਕੜਾ_dim, ਟੁਕੜਾ_ਸ਼ੁਰੂ, ਟੁਕੜਾ_ਸਿਰਾ, ਟੁਕੜਾ_ਕਦਮ,
                                part_kernel.get());

        // create the part conv result tensor
        int64_t part_dst_ne[4];
        ਲਈ (ਪੂਰਨ i = 0; i < 4; i++) {
            part_dst_ne[i] = *(dst->ne + i);
        }
        part_dst_ne[0] = (ਨਿਵੇਸ਼_len - 1) * strideVal[0] - 2 * paddingVal[0] + dilationVal[0] * (part_ne[0] - 1) + 1;

        size_t part_dst_nb[4];
        part_dst_nb[0] = ਆਕਾਰ(ਭਾਰ_type);
        ਲਈ (ਪੂਰਨ i = 1; i < 4; i++) {
            part_dst_nb[i] = part_dst_nb[i - 1] * part_dst_ne[i - 1];
        }
        ggml_cann_pool_alloc part_dst_allocator;
        part_dst_allocator.alloc(ctx.pool(), part_dst_nb[3]);
        ਖਾਲੀ * part_dst_buf = part_dst_allocator.get();

        acl_tensor_ptr acl_part_dst = ggml_cann_create_tensor(part_dst_buf, dst_type, ggml_element_ਮਾਪ(dst),
                                                              part_dst_ne, part_dst_nb, 3, ACL_FORMAT_NCL);
        GGML_CANN_CALL_ACLNN_OP(ctx, InplaceZero, acl_part_dst.get());

        // compute part conv transpose 1d
        GGML_CANN_CALL_ACLNN_OP(ctx, Convolution, acl_ਨਿਵੇਸ਼.get(), part_kernel.get(), ਖਾਲੀ_ਸੂਚਕ, stride.get(),
                                padding.get(), dilation.get(), transposed, padding.get(), groups, acl_part_dst.get(),
                                cubeMathType);

        // compute the position of part result in final result
        int64_t global_ਸ਼ੁਰੂ = ਟੁਕੜਾ_ਸ਼ੁਰੂ;
        int64_t global_ਸਿਰਾ   = std::min((ਨਿਵੇਸ਼_len - 1) * strideVal[0] + ਟੁਕੜਾ_ਸਿਰਾ, dst_len);

        ਖੱਬਾ_pad_len  = global_ਸ਼ੁਰੂ;
        ਸੱਜਾ_pad_len = dst_len - global_ਸਿਰਾ;

        std::vector<int64_t> pad_ਅੰਕੜੇ_Val = { ਖੱਬਾ_pad_len, ਸੱਜਾ_pad_len };
        acl_int_array_ptr    pad_ਅੰਕੜੇ    = ggml_cann_create_int_array(pad_ਅੰਕੜੇ_Val.ਅੰਕੜੇ(), 2);

        acl_scalar_ptr pad_ਮੁੱਲ    = ਖਾਲੀ_ਸੂਚਕ;
        ਦਸ਼ਮਲਵ          pad_ਮੁੱਲ_Val = 0.0;
        pad_ਮੁੱਲ                   = ggml_cann_create_scalar(&pad_ਮੁੱਲ_Val, acl_ਅੰਕੜੇ_Type::ACL_FLOAT);

        int64_t conv_ਨਤੀਜਾ_ne[4];
        ਲਈ (ਪੂਰਨ i = 0; i < 4; i++) {
            conv_ਨਤੀਜਾ_ne[i] = *(dst->ne + i);
        }

        size_t conv_ਨਤੀਜਾ_nb[4];
        conv_ਨਤੀਜਾ_nb[0] = ਆਕਾਰ(ਭਾਰ_type);
        ਲਈ (ਪੂਰਨ i = 1; i < 4; i++) {
            conv_ਨਤੀਜਾ_nb[i] = conv_ਨਤੀਜਾ_nb[i - 1] * conv_ਨਤੀਜਾ_ne[i - 1];
        }

        ggml_cann_pool_alloc conv_ਨਤੀਜਾ_allocator;
        conv_ਨਤੀਜਾ_allocator.alloc(ctx.pool(), conv_ਨਤੀਜਾ_nb[3]);
        ਖਾਲੀ * conv_ਨਤੀਜਾ_buf = conv_ਨਤੀਜਾ_allocator.get();

        acl_tensor_ptr conv_ਨਤੀਜਾ = ggml_cann_create_tensor(conv_ਨਤੀਜਾ_buf, dst_type, ggml_element_ਮਾਪ(dst),
                                                             conv_ਨਤੀਜਾ_ne, conv_ਨਤੀਜਾ_nb, 3, ACL_FORMAT_NCL);

        GGML_CANN_CALL_ACLNN_OP(ctx, InplaceZero, conv_ਨਤੀਜਾ.get());
        GGML_CANN_CALL_ACLNN_OP(ctx, ConstantPadNd, acl_part_dst.get(), pad_ਅੰਕੜੇ.get(), pad_ਮੁੱਲ.get(),
                                conv_ਨਤੀਜਾ.get());
        GGML_CANN_CALL_ACLNN_OP(ctx, Inplace_ਜੋੜੋ, acl_dst.get(), conv_ਨਤੀਜਾ.get(), alpha.get());
    }
}

ਖਾਲੀ ggml_cann_elu(ggml_backend_cann_context & ctx, ggml_tensor * dst) {
    ggml_tensor * src0 = dst->src[0];

    acl_tensor_ptr acl_ਨਿਵੇਸ਼ = ggml_cann_create_tensor(src0);
    acl_tensor_ptr acl_dst   = ggml_cann_create_tensor(dst);

    ਦਸ਼ਮਲਵ          alpha_ਮੁੱਲ = 1.0f;
    acl_scalar_ptr alpha      = ਖਾਲੀ_ਸੂਚਕ;
    alpha                     = ggml_cann_create_scalar(&alpha_ਮੁੱਲ, acl_ਅੰਕੜੇ_Type::ACL_FLOAT);

    GGML_CANN_CALL_ACLNN_OP(ctx, Elu, acl_ਨਿਵੇਸ਼.get(), alpha.get(), alpha.get(), alpha.get(), acl_dst.get());
}

ਖਾਲੀ ggml_cann_ਔਸਤ(ggml_backend_cann_context & ctx, ggml_tensor * dst) {
    ggml_tensor * src0 = dst->src[0];

    acl_tensor_ptr acl_src = ggml_cann_create_tensor(src0);
    acl_tensor_ptr acl_dst = ggml_cann_create_tensor(dst);

    int64_t           reduce_Dim_ਮੁੱਲ[] = { 3 };
    acl_int_array_ptr reduceDim        = ggml_cann_create_int_array(reduce_Dim_ਮੁੱਲ, 1);
    ਤਾਰਕਿਕ              keepDim          = ਸੱਚ;

    GGML_CANN_CALL_ACLNN_OP(ctx, ਔਸਤ, acl_src.get(), reduceDim.get(), keepDim, ACL_FLOAT, acl_dst.get());
}

ਖਾਲੀ ggml_cann_pad_reflect_1d(ggml_backend_cann_context & ctx, ggml_tensor * dst) {
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

    acl_tensor_ptr acl_src = ggml_cann_create_tensor(src0->ਅੰਕੜੇ, ggml_cann_type_mapping(src0->type),
                                                     ggml_element_ਮਾਪ(src0), src_ne_3d, src0->nb, 3);

    acl_tensor_ptr acl_dst = ggml_cann_create_tensor(dst->ਅੰਕੜੇ, ggml_cann_type_mapping(dst->type),
                                                     ggml_element_ਮਾਪ(dst), dst_ne_3d, dst->nb, 3);

    GGML_CANN_CALL_ACLNN_OP(ctx, ReflectionPad1d, acl_src.get(), paddings.get(), acl_dst.get());
}

ਖਾਲੀ ggml_cann_ਗਿਣਤੀ_equal(ggml_backend_cann_context & ctx, ggml_tensor * dst) {
    ggml_tensor * src0 = dst->src[0];
    ggml_tensor * src1 = dst->src[1];

    // Write element-wise equality (0 or 1) into a temporary buffer to avoid
    // modifying src0 in-place.  Use the same type as src0 so ReduceSum can
    // consume it directly without a type cast.
    ggml_cann_pool_alloc eq_alloc(ctx.pool(), ggml_nelements(src0) * ggml_element_ਮਾਪ(src0));
    size_t eq_nb[GGML_MAX_DIMS];
    eq_nb[0] = ggml_element_ਮਾਪ(src0);
    ਲਈ (ਪੂਰਨ i = 1; i < GGML_MAX_DIMS; i++) {
        eq_nb[i] = eq_nb[i - 1] * src0->ne[i - 1];
    }
    acl_tensor_ptr acl_eq = ggml_cann_create_tensor(
        eq_alloc.get(), ggml_cann_type_mapping(src0->type), ggml_element_ਮਾਪ(src0),
        src0->ne, eq_nb, GGML_MAX_DIMS);

    acl_tensor_ptr acl_self  = ggml_cann_create_tensor(src0);
    acl_tensor_ptr acl_other = ggml_cann_create_tensor(src1);
    GGML_CANN_CALL_ACLNN_OP(ctx, EqTensor, acl_self.get(), acl_other.get(), acl_eq.get());

    // Sum the 0/1 values into dst.
    acl_tensor_ptr    acl_dst    = ggml_cann_create_tensor(dst);
    int64_t           dims[4]    = { 0, 1, 2, 3 };
    acl_int_array_ptr dims_arr   = ggml_cann_create_int_array(dims, 4);
    GGML_CANN_CALL_ACLNN_OP(ctx, Reduce_ਜੋੜ, acl_eq.get(), dims_arr.get(), ਸੱਚ,
                            ggml_cann_type_mapping(dst->type), acl_dst.get());
}

ਖਾਲੀ ggml_cann_ਕਦਮ(ggml_backend_cann_context & ctx, ggml_tensor * dst) {
    ggml_tensor * src0 = dst->src[0];

    acl_tensor_ptr acl_src = ggml_cann_create_tensor(src0);
    acl_tensor_ptr acl_dst = ggml_cann_create_tensor(dst);

    ਦਸ਼ਮਲਵ          alpha_ਮੁੱਲ = 0.0f;
    acl_scalar_ptr alpha      = ਖਾਲੀ_ਸੂਚਕ;
    alpha                     = ggml_cann_create_scalar(&alpha_ਮੁੱਲ, acl_ਅੰਕੜੇ_Type::ACL_FLOAT);

    GGML_CANN_CALL_ACLNN_OP(ctx, GtScalar, acl_src.get(), alpha.get(), acl_dst.get());
}

ਖਾਲੀ ggml_cann_softplus(ggml_backend_cann_context & ctx, ggml_tensor * dst) {
    ggml_tensor * src0 = dst->src[0];

    acl_tensor_ptr acl_src = ggml_cann_create_tensor(src0);
    acl_tensor_ptr acl_dst = ggml_cann_create_tensor(dst);

    ਦਸ਼ਮਲਵ          beta_val      = 1.0f;
    ਦਸ਼ਮਲਵ          ਹੱਦ_val = 20.0f;
    acl_scalar_ptr beta          = ggml_cann_create_scalar(&beta_val,      ACL_FLOAT);
    acl_scalar_ptr ਹੱਦ     = ggml_cann_create_scalar(&ਹੱਦ_val, ACL_FLOAT);

    GGML_CANN_CALL_ACLNN_OP(ctx, Softplus, acl_src.get(), beta.get(), ਹੱਦ.get(), acl_dst.get());
}

ਖਾਲੀ ggml_cann_geglu_ਤੇਜ਼(ggml_backend_cann_context & ctx, ggml_tensor * dst) {
    ਸਵੈਚਾਲਿਤ gelu_ਤੇਜ਼_fn = [](ggml_backend_cann_context & ctx, aclTensor * acl_src, aclTensor * acl_dst) {
        GGML_CANN_CALL_ACLNN_OP(ctx, GeluV2, acl_src, 0, acl_dst);
    };
    ggml_cann_op_unary_gated(gelu_ਤੇਜ਼_fn, ctx, dst);
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
ਸਥਿਰ ਖਾਲੀ ggml_cann_mul_mat_id_fp(ggml_backend_cann_context & ctx, ggml_tensor * dst) {
    //dst   [M, K, N, 1]
    ggml_tensor * src0 = dst->src[0];  //src0	[D, M, A, 1]  -> [D, M, K, 1]
    ggml_tensor * src1 = dst->src[1];  //src1	[D, B, N, 1], B = K or B = 1 -> [D, 1, K, 1]
    ggml_tensor * ids  = dst->src[2];  //ids	[K, N]

    GGML_ASSERT(src0->ne[3] == 1);
    GGML_ASSERT(src1->ne[3] == 1);
    GGML_ASSERT(dst->ne[3] == 1);

    int64_t batch = src1->ne[2];
    GGML_ASSERT(batch == ids->ne[1]);

    ggml_cann_pool_alloc export_allocator(ctx.pool(), src0->ne[0] * src0->ne[1] * ids->ne[0] * ggml_element_ਮਾਪ(src0));
    ਖਾਲੀ *               export_ptr = export_allocator.get();
    ਲਈ (int64_t i = 0; i < batch; i++) {
        acl_tensor_ptr select_ਸੂਚਕਾਂਕ  = ggml_cann_create_tensor(ids, ids->ne, ids->nb, 1, ACL_FORMAT_ND, i * ids->nb[1]);
        acl_tensor_ptr export_ਭਾਰ = ggml_cann_create_tensor(src0, src0->ne, src0->nb, 3);

        int64_t select_export_ne[] = { src0->ne[0], src0->ne[1], ids->ne[0] };
        size_t  select_export_nb[3];
        select_export_nb[0] = src0->nb[0];
        ਲਈ (ਪੂਰਨ k = 1; k < 3; k++) {
            select_export_nb[k] = select_export_nb[k - 1] * select_export_ne[k - 1];
        }

        acl_tensor_ptr select_export =
            ggml_cann_create_tensor(export_ptr, ggml_cann_type_mapping(src0->type), ggml_element_ਮਾਪ(src0),
                                    select_export_ne, select_export_nb, 3);
        GGML_CANN_CALL_ACLNN_OP(ctx, ਸੂਚਕਾਂਕ_Select, export_ਭਾਰ.get(), 0, select_ਸੂਚਕਾਂਕ.get(), select_export.get());

        int64_t        select_transpose_ne[] = { select_export_ne[1], select_export_ne[0], select_export_ne[2] };
        size_t         select_transpose_nb[] = { select_export_nb[1], select_export_nb[0], select_export_nb[2] };
        acl_tensor_ptr select_export_transpose =
            ggml_cann_create_tensor(export_ptr, ggml_cann_type_mapping(src0->type), ggml_element_ਮਾਪ(src0),
                                    select_transpose_ne, select_transpose_nb, 3);

        int64_t        ਸਰਗਰਮ_tensor_ne[] = { src1->ne[0], 1, src1->ne[1] };
        size_t         ਸਰਗਰਮ_tensor_nb[] = { src1->nb[0], src1->nb[1], src1->nb[1] };
        acl_tensor_ptr ਸਰਗਰਮ_tensor =
            ggml_cann_create_tensor(src1, ਸਰਗਰਮ_tensor_ne, ਸਰਗਰਮ_tensor_nb, 3, ACL_FORMAT_ND, i * src1->nb[2]);

        int64_t        dst_ne[] = { dst->ne[0], 1, dst->ne[1] };
        size_t         dst_nb[] = { dst->nb[0], dst->nb[1], dst->nb[1] };
        acl_tensor_ptr acl_dst  = ggml_cann_create_tensor(dst, dst_ne, dst_nb, 3, ACL_FORMAT_ND, i * dst->nb[2]);

        GGML_CANN_CALL_ACLNN_OP(ctx, BatchMatMul, ਸਰਗਰਮ_tensor.get(), select_export_transpose.get(), acl_dst.get(), 2);
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
ਸਥਿਰ ਖਾਲੀ ggml_cann_mul_mat_id_quant(ggml_backend_cann_context & ctx, ggml_tensor * dst) {
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

    ਅਚਲ int64_t        n_batches        = ids->ne[1];
    ਅਚਲ int64_t        n_select_experts = ids->ne[0];
    ਅਚਲ ਗਿਣਤੀ ggml_type type             = src0->type;

    ਅਚਲ int32_t ਸਮੂਹ_ਮਾਪ = QK8_0;  // Both Q4_0 and Q8_0 use group size of 32
    GGML_ASSERT(ਸਮੂਹ_ਮਾਪ == QK4_0);

    // Calculate element size for quantized weights
    ਅਚਲ ਦਸ਼ਮਲਵ ਭਾਰ_elem_ਮਾਪ =
        (type == GGML_TYPE_Q4_0) ? 0.5f :
        (type == GGML_TYPE_Q8_0) ? 1.0f :
                                   (GGML_ABORT("MUL_MAT_ID only supports Q4_0 and Q8_0"), 0.0f);

    // Calculate scale offset in memory
    ਅਚਲ size_t ਭਾਰ_ਮਾਪ     = src0->ne[0] * src0->ne[1] * src0->ne[2] * ਭਾਰ_elem_ਮਾਪ;
    ਅਚਲ size_t scale_elem_ਮਾਪ = ਆਕਾਰ(uint16_t);
    ਅੱਖਰ *       scale_ਅੰਕੜੇ      = (ਅੱਖਰ *) src0->ਅੰਕੜੇ + ਭਾਰ_ਮਾਪ;

    // Allocate buffers for selected expert weights and scales
    ਅਚਲ size_t         selected_ਭਾਰ_ਮਾਪ = src0->ne[0] * src0->ne[1] * n_select_experts * ਭਾਰ_elem_ਮਾਪ;
    ggml_cann_pool_alloc selected_ਭਾਰ_alloc(ctx.pool(), selected_ਭਾਰ_ਮਾਪ);
    ਖਾਲੀ *               selected_ਭਾਰ_ਬਫਰ = selected_ਭਾਰ_alloc.get();

    ਅਚਲ size_t selected_scale_ਮਾਪ = (src0->ne[0] / ਸਮੂਹ_ਮਾਪ) * src0->ne[1] * n_select_experts * scale_elem_ਮਾਪ;
    ggml_cann_pool_alloc selected_scale_alloc(ctx.pool(), selected_scale_ਮਾਪ);
    ਖਾਲੀ *               selected_scale_ਬਫਰ = selected_scale_alloc.get();

    // Helper lambda to allocate and cast tensor to F16 if needed
    constexpr size_t f16_elem_ਮਾਪ      = ਆਕਾਰ(uint16_t);
    ਸਵੈਚਾਲਿਤ             prepare_f16_ਬਫਰ = [&](ggml_tensor * tensor, ggml_cann_pool_alloc & allocator,
                                  ਤਾਰਕਿਕ need_cast = ਝੂਠ) -> ਖਾਲੀ * {
        ਜੇ (tensor->type == GGML_TYPE_F16) {
            ਵਾਪਸ tensor->ਅੰਕੜੇ;
        }

        size_t ਕੁੱਲ_ਮਾਪ = f16_elem_ਮਾਪ;
        ਲਈ (ਪੂਰਨ i = 0; i < GGML_MAX_DIMS; i++) {
            ਕੁੱਲ_ਮਾਪ *= tensor->ne[i];
        }
        ਖਾਲੀ * ਬਫਰ = allocator.alloc(ਕੁੱਲ_ਮਾਪ);

        ਜੇ (need_cast == ਝੂਠ) {
            ਵਾਪਸ ਬਫਰ;
        }

        int64_t ne[GGML_MAX_DIMS];
        size_t  nb[GGML_MAX_DIMS] = { f16_elem_ਮਾਪ };
        ਲਈ (ਪੂਰਨ i = 0; i < GGML_MAX_DIMS; i++) {
            ne[i] = tensor->ne[i];
            ਜੇ (i > 0) {
                nb[i] = nb[i - 1] * ne[i - 1];
            }
        }

        acl_tensor_ptr src_tensor = ggml_cann_create_tensor(tensor);
        acl_tensor_ptr f16_tensor = ggml_cann_create_tensor(ਬਫਰ, ACL_FLOAT16, f16_elem_ਮਾਪ, ne, nb, GGML_MAX_DIMS);
        aclnn_cast(ctx, src_tensor.get(), f16_tensor.get(), ACL_FLOAT16);

        ਵਾਪਸ ਬਫਰ;
    };

    // Prepare input and output buffers
    ggml_cann_pool_alloc ਨਿਵੇਸ਼_alloc(ctx.pool());
    ਖਾਲੀ *               ਨਿਵੇਸ਼_ਬਫਰ = prepare_f16_ਬਫਰ(src1, ਨਿਵੇਸ਼_alloc, ਸੱਚ);

    ggml_cann_pool_alloc output_alloc(ctx.pool());
    ਖਾਲੀ *               output_ਬਫਰ = prepare_f16_ਬਫਰ(dst, output_alloc, ਝੂਠ);

    // Process each batch
    ਲਈ (int64_t batch_idx = 0; batch_idx < n_batches; batch_idx++) {
        // Create index tensor for current batch
        ਅਚਲ size_t   ਸੂਚਕਾਂਕ_ਵਿੱਥ  = batch_idx * ids->nb[1];
        acl_tensor_ptr batch_indices = ggml_cann_create_tensor(ids, ids->ne, ids->nb, 1, ACL_FORMAT_ND, ਸੂਚਕਾਂਕ_ਵਿੱਥ);

        // Select quantized weights using expert indices
        // Q4_0 stores 2 values per byte, Q8_0 stores 1 value per byte
        ਅਚਲ int64_t ਭਾਰ_d         = (type == GGML_TYPE_Q4_0) ? src0->ne[0] / 2 : src0->ne[0];
        ਅਚਲ int64_t ਭਾਰ_m         = src0->ne[1];
        ਅਚਲ int64_t ਭਾਰ_n_experts = src0->ne[2];

        int64_t ਭਾਰ_ne[3] = { ਭਾਰ_d, ਭਾਰ_m, ਭਾਰ_n_experts };
        size_t  ਭਾਰ_nb[3] = { ਆਕਾਰ(int8_t), ਭਾਰ_d * ਆਕਾਰ(int8_t), ਭਾਰ_d * ਭਾਰ_m * ਆਕਾਰ(int8_t) };

        acl_tensor_ptr all_weights =
            ggml_cann_create_tensor(src0->ਅੰਕੜੇ, ACL_INT8, ਆਕਾਰ(int8_t), ਭਾਰ_ne, ਭਾਰ_nb, 3);

        int64_t selected_ਭਾਰ_ne[3] = { ਭਾਰ_d, ਭਾਰ_m, n_select_experts };
        size_t  selected_ਭਾਰ_nb[3] = { ਆਕਾਰ(int8_t), ਭਾਰ_d * ਆਕਾਰ(int8_t),
                                          ਭਾਰ_d * ਭਾਰ_m * ਆਕਾਰ(int8_t) };

        acl_tensor_ptr selected_weights = ggml_cann_create_tensor(selected_ਭਾਰ_ਬਫਰ, ACL_INT8, ਆਕਾਰ(int8_t),
                                                                  selected_ਭਾਰ_ne, selected_ਭਾਰ_nb, 3);

        GGML_CANN_CALL_ACLNN_OP(ctx, ਸੂਚਕਾਂਕ_Select, all_weights.get(), 0, batch_indices.get(), selected_weights.get());

        // Select scales using the same expert indices
        ਅਚਲ int64_t scale_d     = src0->ne[0] / ਸਮੂਹ_ਮਾਪ;
        int64_t       scale_ne[3] = { scale_d, ਭਾਰ_m, ਭਾਰ_n_experts };
        size_t scale_nb[3] = { scale_elem_ਮਾਪ, scale_d * scale_elem_ਮਾਪ, scale_d * ਭਾਰ_m * scale_elem_ਮਾਪ };

        acl_tensor_ptr all_scales =
            ggml_cann_create_tensor(scale_ਅੰਕੜੇ, ACL_FLOAT16, scale_elem_ਮਾਪ, scale_ne, scale_nb, 3);

        int64_t selected_scale_ne[3] = { scale_d, ਭਾਰ_m, n_select_experts };
        size_t  selected_scale_nb[3] = { scale_elem_ਮਾਪ, scale_d * scale_elem_ਮਾਪ,
                                         scale_d * ਭਾਰ_m * scale_elem_ਮਾਪ };

        acl_tensor_ptr selected_scales = ggml_cann_create_tensor(selected_scale_ਬਫਰ, ACL_FLOAT16, scale_elem_ਮਾਪ,
                                                                 selected_scale_ne, selected_scale_nb, 3);

        GGML_CANN_CALL_ACLNN_OP(ctx, ਸੂਚਕਾਂਕ_Select, all_scales.get(), 0, batch_indices.get(), selected_scales.get());

        // Process each expert for current batch
        // IndexSelect output layout: [D, M, K] in contiguous format
        // WeightQuantBatchMatmulV2 expects: [M, D] with row-major stride
        ਲਈ (int64_t expert_idx = 0; expert_idx < n_select_experts; expert_idx++) {
            // Determine input offset: broadcast if src1->ne[1]==1, otherwise use per-expert input
            ਅਚਲ size_t ਨਿਵੇਸ਼_ਵਿੱਥ =
                (batch_idx * src1->ne[1] + (src1->ne[1] == 1 ? 0 : expert_idx)) * src1->ne[0] * f16_elem_ਮਾਪ;
            ਅਚਲ size_t output_ਵਿੱਥ = (batch_idx * dst->ne[1] + expert_idx) * dst->ne[0] * f16_elem_ਮਾਪ;

            // Create weight view for current expert: [D, M, K] -> [M, D]
            int64_t      ਭਾਰ_view_ne[2]  = { ਭਾਰ_m, src0->ne[0] };
            ਦਸ਼ਮਲਵ        ਭਾਰ_view_nb[2]  = { src0->ne[0] * ਭਾਰ_elem_ਮਾਪ, ਭਾਰ_elem_ਮਾਪ };
            ਅਚਲ size_t ਭਾਰ_view_ਵਿੱਥ = expert_idx * selected_ਭਾਰ_nb[2];

            acl_tensor_ptr ਭਾਰ_view =
                ggml_cann_create_tensor(selected_ਭਾਰ_ਬਫਰ, ggml_cann_type_mapping(type), ਭਾਰ_elem_ਮਾਪ,
                                        ਭਾਰ_view_ne, ਭਾਰ_view_nb, 2, ACL_FORMAT_ND, ਭਾਰ_view_ਵਿੱਥ);

            // Create scale view for current expert: [D, M, K] -> [M, D]
            int64_t      scale_view_ne[2]  = { ਭਾਰ_m, scale_d };
            size_t       scale_view_nb[2]  = { selected_scale_nb[1], selected_scale_nb[0] };
            ਅਚਲ size_t scale_view_ਵਿੱਥ = expert_idx * selected_scale_nb[2];

            acl_tensor_ptr scale_view =
                ggml_cann_create_tensor(selected_scale_ਬਫਰ, ACL_FLOAT16, scale_elem_ਮਾਪ, scale_view_ne,
                                        scale_view_nb, 2, ACL_FORMAT_ND, scale_view_ਵਿੱਥ);

            // Create input activation tensor [D, 1]
            int64_t ਨਿਵੇਸ਼_ne[2] = { src1->ne[0], 1 };
            size_t  ਨਿਵੇਸ਼_nb[2] = { f16_elem_ਮਾਪ, src1->ne[0] * f16_elem_ਮਾਪ };

            acl_tensor_ptr ਨਿਵੇਸ਼_tensor = ggml_cann_create_tensor(ਨਿਵੇਸ਼_ਬਫਰ, ACL_FLOAT16, f16_elem_ਮਾਪ, ਨਿਵੇਸ਼_ne,
                                                                  ਨਿਵੇਸ਼_nb, 2, ACL_FORMAT_ND, ਨਿਵੇਸ਼_ਵਿੱਥ);

            // Create output tensor [M, 1]
            int64_t output_ne[2] = { dst->ne[0], 1 };
            size_t  output_nb[2] = { f16_elem_ਮਾਪ, dst->ne[0] * f16_elem_ਮਾਪ };

            acl_tensor_ptr output_tensor = ggml_cann_create_tensor(output_ਬਫਰ, ACL_FLOAT16, f16_elem_ਮਾਪ, output_ne,
                                                                   output_nb, 2, ACL_FORMAT_ND, output_ਵਿੱਥ);

            // Perform quantized matrix multiplication
            GGML_CANN_CALL_ACLNN_OP(ctx, ਭਾਰ_Quant_Batch_Matmul_V2, ਨਿਵੇਸ਼_tensor.get(), ਭਾਰ_view.get(),
                                    scale_view.get(), ਖਾਲੀ_ਸੂਚਕ, ਖਾਲੀ_ਸੂਚਕ, ਖਾਲੀ_ਸੂਚਕ, ਖਾਲੀ_ਸੂਚਕ, ਸਮੂਹ_ਮਾਪ,
                                    output_tensor.get());
        }
    }

    // Cast output back to original type if we used a temporary F16 buffer
    ਜੇ (dst->type != GGML_TYPE_F16) {
        int64_t ne[GGML_MAX_DIMS];
        size_t  nb[GGML_MAX_DIMS] = { f16_elem_ਮਾਪ };
        ਲਈ (ਪੂਰਨ i = 0; i < GGML_MAX_DIMS; i++) {
            ne[i] = dst->ne[i];
            ਜੇ (i > 0) {
                nb[i] = nb[i - 1] * ne[i - 1];
            }
        }

        acl_tensor_ptr f16_output =
            ggml_cann_create_tensor(output_ਬਫਰ, ACL_FLOAT16, f16_elem_ਮਾਪ, ne, nb, GGML_MAX_DIMS);
        acl_tensor_ptr dst_tensor = ggml_cann_create_tensor(dst);

        aclnn_cast(ctx, f16_output.get(), dst_tensor.get(), ggml_cann_type_mapping(dst->type));
    }
}

ਖਾਲੀ ggml_cann_mul_mat_id(ggml_backend_cann_context & ctx, ggml_tensor * dst) {
    ਅਚਲ ਗਿਣਤੀ ggml_type type = dst->src[0]->type;
    ਚੋਣ (type) {
        ਹਾਲਤ GGML_TYPE_F32:
        ਹਾਲਤ GGML_TYPE_F16:
            ggml_cann_mul_mat_id_fp(ctx, dst);
            ਰੋਕੋ;
        ਹਾਲਤ GGML_TYPE_Q4_0:
        ਹਾਲਤ GGML_TYPE_Q8_0:
            ggml_cann_mul_mat_id_quant(ctx, dst);
            ਰੋਕੋ;
        ਮੂਲ:
            GGML_ABORT("Unsupported type for mul_mat_id");
            ਰੋਕੋ;
    }
}

ਖਾਲੀ ggml_cann_flash_attn_ext(ggml_backend_cann_context & ctx, ggml_tensor * dst) {
    ggml_tensor * src0 = dst->src[0];  // q, fp32 | B, N, S, D (uncont) -> B, S, N, D (cont)
    ggml_tensor * src1 = dst->src[1];  // k, fp16 | B, N, S, D (uncont) -> B, S, N, D (cont)
    ggml_tensor * src2 = dst->src[2];  // v, fp16 | B, N, S, D (uncont) -> B, S, N, D (cont)
    ggml_tensor * src3 = dst->src[3];  // mask, fp16

    // B, N, S, D (uncont) -> B, S, N, D (cont)
    int64_t src0_bsnd_ne[GGML_MAX_DIMS];
    ਸਮ੍ਰਿਤੀ_ਨਕਲ(src0_bsnd_ne, src0->ne, GGML_MAX_DIMS * ਆਕਾਰ(int64_t));
    size_t src0_bsnd_nb[GGML_MAX_DIMS];
    ਸਮ੍ਰਿਤੀ_ਨਕਲ(src0_bsnd_nb, src0->nb, GGML_MAX_DIMS * ਆਕਾਰ(size_t));
    int64_t src1_bsnd_ne[GGML_MAX_DIMS];
    ਸਮ੍ਰਿਤੀ_ਨਕਲ(src1_bsnd_ne, src1->ne, GGML_MAX_DIMS * ਆਕਾਰ(int64_t));
    size_t src1_bsnd_nb[GGML_MAX_DIMS];
    ਸਮ੍ਰਿਤੀ_ਨਕਲ(src1_bsnd_nb, src1->nb, GGML_MAX_DIMS * ਆਕਾਰ(size_t));
    int64_t src2_bsnd_ne[GGML_MAX_DIMS];
    ਸਮ੍ਰਿਤੀ_ਨਕਲ(src2_bsnd_ne, src2->ne, GGML_MAX_DIMS * ਆਕਾਰ(int64_t));
    size_t src2_bsnd_nb[GGML_MAX_DIMS];
    ਸਮ੍ਰਿਤੀ_ਨਕਲ(src2_bsnd_nb, src2->nb, GGML_MAX_DIMS * ਆਕਾਰ(size_t));

    ਸਵੈਚਾਲਿਤ transpose12 = [](int64_t * ne, size_t * nb) {
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

    ਦਸ਼ਮਲਵ maxBias      = 0.0f;
    ਦਸ਼ਮਲਵ scale_ਮੁੱਲ   = 1.0f;
    ਦਸ਼ਮਲਵ logitSoftcap = 0.0f;
    ਸਮ੍ਰਿਤੀ_ਨਕਲ(&scale_ਮੁੱਲ, (ਦਸ਼ਮਲਵ *) dst->op_params + 0, ਆਕਾਰ(ਦਸ਼ਮਲਵ));
    ਸਮ੍ਰਿਤੀ_ਨਕਲ(&maxBias, (ਦਸ਼ਮਲਵ *) dst->op_params + 1, ਆਕਾਰ(ਦਸ਼ਮਲਵ));
    ਸਮ੍ਰਿਤੀ_ਨਕਲ(&logitSoftcap, (ਦਸ਼ਮਲਵ *) dst->op_params + 2, ਆਕਾਰ(ਦਸ਼ਮਲਵ));

    ਜੇ (logitSoftcap == 0.0f) {
        size_t fa_Elem_ਮਾਪ = ਆਕਾਰ(uint16_t);
        ਸਵੈਚਾਲਿਤ   fa_ਅੰਕੜੇ_Type = ACL_FLOAT16;  //ACL_BF16;

        acl_tensor_ptr acl_q_tensor = ਖਾਲੀ_ਸੂਚਕ;
        acl_tensor_ptr acl_k_tensor = ਖਾਲੀ_ਸੂਚਕ;
        acl_tensor_ptr acl_v_tensor = ਖਾਲੀ_ਸੂਚਕ;

        // Step 1: cast the src0 (Query) to fp16 if needed
        ggml_cann_pool_alloc src0_f16_allocator(ctx.pool());
        ਖਾਲੀ *               src0_f16_ਬਫਰ = ਖਾਲੀ_ਸੂਚਕ;

        ਜੇ (ggml_cann_type_mapping(src0->type) != fa_ਅੰਕੜੇ_Type) {
            acl_tensor_ptr acl_src0_f32_tensor =
                ggml_cann_create_tensor(src0, src0_bsnd_ne, src0_bsnd_nb, GGML_MAX_DIMS);
            src0_f16_ਬਫਰ = src0_f16_allocator.alloc(ggml_nelements(src0) * fa_Elem_ਮਾਪ);

            int64_t * src0_f16_ne = src0_bsnd_ne;
            size_t    src0_f16_nb[GGML_MAX_DIMS];
            src0_f16_nb[0] = ਆਕਾਰ(uint16_t);
            ਲਈ (ਪੂਰਨ i = 1; i < GGML_MAX_DIMS; ++i) {
                src0_f16_nb[i] = src0_f16_nb[i - 1] * src0_f16_ne[i - 1];
            }

            acl_q_tensor = ggml_cann_create_tensor(src0_f16_ਬਫਰ, fa_ਅੰਕੜੇ_Type, fa_Elem_ਮਾਪ, src0_f16_ne, src0_f16_nb,
                                                   GGML_MAX_DIMS);
            aclnn_cast(ctx, acl_src0_f32_tensor.get(), acl_q_tensor.get(), fa_ਅੰਕੜੇ_Type);
        } ਨਹੀਂਤਾਂ {
            acl_q_tensor = ggml_cann_create_tensor(src0, src0_bsnd_ne, src0_bsnd_nb, GGML_MAX_DIMS);
        }

        // Step 2: create the acl tensors for src1 (Key), src2 (Value),
        //         and the direct output from FusedInferAttention

        acl_k_tensor = ggml_cann_create_tensor(src1, src1_bsnd_ne, src1_bsnd_nb, GGML_MAX_DIMS);
        acl_v_tensor = ggml_cann_create_tensor(src2, src2_bsnd_ne, src2_bsnd_nb, GGML_MAX_DIMS);

        // Step 2.5: Pad Q, K, V along head dimension if D is not a multiple of 16
        //           (required by FusedInferAttentionScoreV2)
        ਅਚਲ int64_t D         = src0->ne[0];
        ਅਚਲ int64_t D_padded  = GGML_PAD(D, 16);
        ਅਚਲ ਤਾਰਕਿਕ needs_padding = (D != D_padded);

        ggml_cann_pool_alloc q_pad_allocator(ctx.pool());
        ggml_cann_pool_alloc k_pad_allocator(ctx.pool());
        ggml_cann_pool_alloc v_pad_allocator(ctx.pool());

        ਜੇ (needs_padding) {
            int64_t paddings[] = { 0, D_padded - D, 0, 0, 0, 0, 0, 0 };

            ਸਵੈਚਾਲਿਤ pad_fa_tensor = [&](acl_tensor_ptr & tensor, ਅਚਲ int64_t * bsnd_ne,
                                     ggml_cann_pool_alloc & allocator) {
                int64_t pad_ne[GGML_MAX_DIMS] = { D_padded, bsnd_ne[1], bsnd_ne[2], bsnd_ne[3] };
                size_t  pad_nb[GGML_MAX_DIMS];
                pad_nb[0] = fa_Elem_ਮਾਪ;
                ਲਈ (ਪੂਰਨ i = 1; i < GGML_MAX_DIMS; ++i) {
                    pad_nb[i] = pad_nb[i - 1] * pad_ne[i - 1];
                }
                int64_t nelements = pad_ne[0] * pad_ne[1] * pad_ne[2] * pad_ne[3];
                ਖਾਲੀ *  ਬਫਰ    = allocator.alloc(nelements * fa_Elem_ਮਾਪ);
                acl_tensor_ptr padded =
                    ggml_cann_create_tensor(ਬਫਰ, fa_ਅੰਕੜੇ_Type, fa_Elem_ਮਾਪ, pad_ne, pad_nb, GGML_MAX_DIMS);
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
        ਜੇ (src3 != ਖਾਲੀ_ਸੂਚਕ) {
            // Construct the truncated pse tensor (common for prefill/decode)
            int64_t trunc_pse_ne[GGML_MAX_DIMS] = {
                src3->ne[0],  // D
                src0->ne[1],  // S (number of Q tokens)
                src3->ne[2],  // mask N
                src3->ne[3]   // B
            };
            size_t * trunc_pse_nb = src3->nb;

            acl_tensor_ptr acl_ਨਕਾਬ_f16_trunc_tensor = ggml_cann_create_tensor(
                src3->ਅੰਕੜੇ, ACL_FLOAT16, ਆਕਾਰ(uint16_t), trunc_pse_ne, trunc_pse_nb, GGML_MAX_DIMS);

            int64_t bcast_pse_ne[GGML_MAX_DIMS];
            size_t  bcast_pse_nb[GGML_MAX_DIMS];
            bcast_pse_ne[0] = src3->ne[0];  // D
            bcast_pse_ne[1] = src0->ne[1];  // S
            bcast_pse_ne[2] = src0->ne[2];  // N (num_heads)
            bcast_pse_ne[3] = src3->ne[3];  // B
            ਜੇ (maxBias == 0.0f) {
                // When maxBias == 0.0f, use nb = 0 reduce once repeat (Qwen2)
                // Construct the bcast tensor (simulate repeat on the head dimension using stride=0)
                bcast_pse_nb[0] = ਆਕਾਰ(uint16_t);
                bcast_pse_nb[1] = bcast_pse_nb[0] * bcast_pse_ne[0];
                bcast_pse_nb[2] = 0;  // <---- the head dimension shares the same data
                bcast_pse_nb[3] = src3->nb[3];

                bcast_pse_tensor = ggml_cann_create_tensor(src3->ਅੰਕੜੇ, ACL_FLOAT16, ਆਕਾਰ(uint16_t), bcast_pse_ne,
                                                           bcast_pse_nb, GGML_MAX_DIMS);

            } ਨਹੀਂਤਾਂ {
                bcast_pse_nb[0] = ਆਕਾਰ(uint16_t);
                ਲਈ (ਪੂਰਨ i = 1; i < GGML_MAX_DIMS; i++) {
                    bcast_pse_nb[i] = bcast_pse_nb[i - 1] * bcast_pse_ne[i - 1];
                }

                ਖਾਲੀ * bcast_pse_ਬਫਰ =
                    bcast_pse_allocator.alloc(ggml_nelements(src3) * src0->ne[2] * ਆਕਾਰ(uint16_t));

                bcast_pse_tensor = ggml_cann_create_tensor(bcast_pse_ਬਫਰ, ACL_FLOAT16, ਆਕਾਰ(uint16_t),
                                                           bcast_pse_ne, bcast_pse_nb, GGML_MAX_DIMS);

                int64_t repeats[] = { 1, src0->ne[2], 1, 1 };
                aclnn_repeat(ctx, acl_ਨਕਾਬ_f16_trunc_tensor.get(), bcast_pse_tensor.get(), repeats);

                // alibi
                // Compute the slope if needed. Derived from ggml_cann_softmax().
                ਅਚਲ int64_t        n_heads = src0->ne[2];
                ggml_cann_pool_alloc ਢਲਾਨ_allocator(ctx.pool(), n_heads * ਆਕਾਰ(uint16_t));
                ਖਾਲੀ *               ਢਲਾਨ_ਬਫਰ = ਢਲਾਨ_allocator.get();
                aclnn_get_ਢਲਾਨ(ctx, n_heads, ਢਲਾਨ_ਬਫਰ, maxBias, GGML_TYPE_F16);

                int64_t ਢਲਾਨ_ne[] = { 1, 1, n_heads, 1 };
                size_t  ਢਲਾਨ_nb[GGML_MAX_DIMS];
                ਢਲਾਨ_nb[0] = ਆਕਾਰ(uint16_t);
                ਲਈ (ਪੂਰਨ i = 1; i < GGML_MAX_DIMS; i++) {
                    ਢਲਾਨ_nb[i] = ਢਲਾਨ_nb[i - 1] * ਢਲਾਨ_ne[0];
                }

                acl_tensor_ptr ਢਲਾਨ_tensor = ggml_cann_create_tensor(ਢਲਾਨ_ਬਫਰ, ACL_FLOAT16, ਆਕਾਰ(uint16_t),
                                                                      ਢਲਾਨ_ne, ਢਲਾਨ_nb, GGML_MAX_DIMS);
                GGML_CANN_CALL_ACLNN_OP(ctx, InplaceMul, bcast_pse_tensor.get(), ਢਲਾਨ_tensor.get());
            }
        }

        // Step 4: set the inputs for FusedInferAttention.
        acl_tensor_list_ptr acl_k_tensor_list = ggml_cann_create_tensor_list(acl_k_tensor);
        acl_tensor_list_ptr acl_v_tensor_list = ggml_cann_create_tensor_list(acl_v_tensor);

        int64_t ਅੰਕ_Heads           = src0->ne[2];  // N
        int64_t ਅੰਕ_ਕੁੰਜੀ_ਮੁੱਲ_Heads   = src1->ne[2];
        // double  scaleValue = 1 / sqrt(src0->ne[0]); // 1/sqrt(d)
        int64_t pre_ਟੋਕਨ          = 65535;
        int64_t ਅਗਲਾ_ਟੋਕਨ         = 65535;
        ਅੱਖਰ    layout[5]          = { 'B', 'S', 'N', 'D', 0 };
        int64_t sparse_ਬਹੁਲਕ         = 0;
        int64_t innerPrecise       = (src0->ne[1] == 1) ? 0 : 2;
        int64_t ਖੰਡ_ਮਾਪ          = 0;
        int64_t antiquant_ਬਹੁਲਕ      = 0;
        ਤਾਰਕਿਕ    softmaxLseFlag     = ਝੂਠ;
        int64_t ਕੁੰਜੀ_Antiquant_ਬਹੁਲਕ   = 0;
        int64_t ਮੁੱਲ_Antiquant_ਬਹੁਲਕ = 0;

        GGML_ASSERT(dst->type == GGML_TYPE_F32 || dst->type == GGML_TYPE_F16);
        acl_tensor_ptr       fa_dst_tensor;
        ggml_cann_pool_alloc ਬਾਹਰਲਾ_f16_allocator(ctx.pool());
        ਜੇ (dst->type == GGML_TYPE_F32 || needs_padding) {
            int64_t * ਬਾਹਰਲਾ_f16_ne = src0_bsnd_ne;
            size_t    ਬਾਹਰਲਾ_f16_nb[GGML_MAX_DIMS];
            ਬਾਹਰਲਾ_f16_nb[0] = fa_Elem_ਮਾਪ;
            ਲਈ (ਪੂਰਨ i = 1; i < GGML_MAX_DIMS; ++i) {
                ਬਾਹਰਲਾ_f16_nb[i] = ਬਾਹਰਲਾ_f16_nb[i - 1] * ਬਾਹਰਲਾ_f16_ne[i - 1];
            }
            int64_t ਬਾਹਰਲਾ_nelements = ਬਾਹਰਲਾ_f16_ne[0] * ਬਾਹਰਲਾ_f16_ne[1] * ਬਾਹਰਲਾ_f16_ne[2] * ਬਾਹਰਲਾ_f16_ne[3];
            ਖਾਲੀ *  ਬਾਹਰਲਾ_f16_ਬਫਰ = ਬਾਹਰਲਾ_f16_allocator.alloc(ਬਾਹਰਲਾ_nelements * fa_Elem_ਮਾਪ);

            fa_dst_tensor =
                ggml_cann_create_tensor(ਬਾਹਰਲਾ_f16_ਬਫਰ, fa_ਅੰਕੜੇ_Type, fa_Elem_ਮਾਪ, ਬਾਹਰਲਾ_f16_ne, ਬਾਹਰਲਾ_f16_nb, GGML_MAX_DIMS);
        } ਨਹੀਂਤਾਂ {
            fa_dst_tensor = ggml_cann_create_tensor(dst);
        }

        GGML_CANN_CALL_ACLNN_OP(ctx, FusedInferAttentionScoreV2, acl_q_tensor.get(), acl_k_tensor_list.get(),
                                acl_v_tensor_list.get(),               // q, k, v
                                bcast_pse_tensor.get(), ਖਾਲੀ_ਸੂਚਕ,       // pse, mask
                                ਖਾਲੀ_ਸੂਚਕ, ਖਾਲੀ_ਸੂਚਕ,                      // actSeqLen, actSeqLenkv
                                ਖਾਲੀ_ਸੂਚਕ, ਖਾਲੀ_ਸੂਚਕ,                      // deqScale1, quantScale1
                                ਖਾਲੀ_ਸੂਚਕ, ਖਾਲੀ_ਸੂਚਕ, ਖਾਲੀ_ਸੂਚਕ,             // deqScale2, quantScale2, quantOffset2
                                ਖਾਲੀ_ਸੂਚਕ, ਖਾਲੀ_ਸੂਚਕ,                      // antiquantScale, antiquantOffset
                                ਖਾਲੀ_ਸੂਚਕ,                               // blockTable
                                ਖਾਲੀ_ਸੂਚਕ, ਖਾਲੀ_ਸੂਚਕ,                      // qPadSize, kvPadSize
                                ਖਾਲੀ_ਸੂਚਕ, ਖਾਲੀ_ਸੂਚਕ,                      // kAntiquantScale, kAntiQuantOffset
                                ਖਾਲੀ_ਸੂਚਕ, ਖਾਲੀ_ਸੂਚਕ,                      // vAntiquantScale, vAntiQuantOffset
                                ਖਾਲੀ_ਸੂਚਕ, ਖਾਲੀ_ਸੂਚਕ, ਖਾਲੀ_ਸੂਚਕ,             // kSharedPrefix, vSharedPrefix, actSharedLen
                                ਅੰਕ_Heads, scale_ਮੁੱਲ,                  // heads, scaleValue
                                pre_ਟੋਕਨ, ਅਗਲਾ_ਟੋਕਨ,                 // preTokens, nextTokens
                                layout,                                // inputLayout
                                ਅੰਕ_ਕੁੰਜੀ_ਮੁੱਲ_Heads,                      // numKVHeads
                                sparse_ਬਹੁਲਕ, innerPrecise,              // sparseMode, innerPrecise
                                ਖੰਡ_ਮਾਪ, antiquant_ਬਹੁਲਕ,              // blockSize, antiquantMode
                                softmaxLseFlag,                        // softmaxLseFlag
                                ਕੁੰਜੀ_Antiquant_ਬਹੁਲਕ, ਮੁੱਲ_Antiquant_ਬਹੁਲਕ,  // keyAntiqMode, valueAntiqMode
                                fa_dst_tensor.get(),                   // attentionOut
                                ਖਾਲੀ_ਸੂਚਕ                                // softmaxLse
        );

        // Step 6: post-processing — slice padded output and/or cast to f32
        ਜੇ (needs_padding) {
            ggml_cann_pool_alloc sliced_f16_allocator(ctx.pool());

            ਜੇ (dst->type == GGML_TYPE_F32) {
                int64_t sliced_ne[GGML_MAX_DIMS] = { D, src0_bsnd_ne[1], src0_bsnd_ne[2], src0_bsnd_ne[3] };
                size_t  sliced_nb[GGML_MAX_DIMS];
                sliced_nb[0] = fa_Elem_ਮਾਪ;
                ਲਈ (ਪੂਰਨ i = 1; i < GGML_MAX_DIMS; ++i) {
                    sliced_nb[i] = sliced_nb[i - 1] * sliced_ne[i - 1];
                }
                int64_t sliced_nelements = sliced_ne[0] * sliced_ne[1] * sliced_ne[2] * sliced_ne[3];
                ਖਾਲੀ *  sliced_ਬਫਰ    = sliced_f16_allocator.alloc(sliced_nelements * fa_Elem_ਮਾਪ);
                acl_tensor_ptr sliced_f16_tensor = ggml_cann_create_tensor(sliced_ਬਫਰ, fa_ਅੰਕੜੇ_Type, fa_Elem_ਮਾਪ,
                                                                           sliced_ne, sliced_nb, GGML_MAX_DIMS);

                GGML_CANN_CALL_ACLNN_OP(ctx, ਟੁਕੜਾ, fa_dst_tensor.get(),
                                        (int64_t) -1, (int64_t) 0, D, (int64_t) 1, sliced_f16_tensor.get());

                acl_tensor_ptr acl_dst_tensor = ggml_cann_create_tensor(dst);
                aclnn_cast(ctx, sliced_f16_tensor.get(), acl_dst_tensor.get(), ggml_cann_type_mapping(dst->type));
            } ਨਹੀਂਤਾਂ {
                acl_tensor_ptr acl_dst_tensor = ggml_cann_create_tensor(dst);
                GGML_CANN_CALL_ACLNN_OP(ctx, ਟੁਕੜਾ, fa_dst_tensor.get(),
                                        (int64_t) -1, (int64_t) 0, D, (int64_t) 1, acl_dst_tensor.get());
            }
        } ਨਹੀਂਤਾਂ ਜੇ (dst->type == GGML_TYPE_F32) {
            acl_tensor_ptr acl_dst_tensor = ggml_cann_create_tensor(dst);
            aclnn_cast(ctx, fa_dst_tensor.get(), acl_dst_tensor.get(), ggml_cann_type_mapping(dst->type));
        }
    } ਨਹੀਂਤਾਂ {
        GGML_ABORT("Function is not implemented.");
    }
}

ਸਥਿਰ ਖਾਲੀ ggml_cann_ਬਾਹਰਲਾ_prod_fp(ggml_backend_cann_context & ctx, ggml_tensor * dst) {
    ggml_tensor * src0 = dst->src[0];  // weight  [ne00=m, ne01=K, ne02, ne03]
    ggml_tensor * src1 = dst->src[1];  // input   [ne10=n, ne11=K, ne12, ne13]
    GGML_TENSOR_ਦੁਆਧਾਰੀ_OP_LOCALS

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

    ਅਚਲ acl_ਅੰਕੜੇ_Type src0_acl_type = ggml_cann_type_mapping(src0->type);
    ਅਚਲ acl_ਅੰਕੜੇ_Type src1_acl_type = ggml_cann_type_mapping(src1->type);
    ਅਚਲ acl_ਅੰਕੜੇ_Type dst_acl_type  = ggml_cann_type_mapping(dst->type);
    ਅਚਲ size_t      src0_type_sz  = ggml_type_ਮਾਪ(src0->type);
    ਅਚਲ size_t      src1_type_sz  = ggml_type_ਮਾਪ(src1->type);
    ਅਚਲ size_t      dst_type_sz   = ggml_type_ਮਾਪ(dst->type);

    ਅਚਲ int64_t dps2 = ne2 / ne02;
    ਅਚਲ int64_t dps3 = ne3 / ne03;

    ਲਈ (int64_t i3 = 0; i3 < ne3; i3++) {
        ਲਈ (int64_t i2 = 0; i2 < ne2; i2++) {
            ਅਚਲ int64_t i02 = i2 / dps2;
            ਅਚਲ int64_t i03 = i3 / dps3;

            // src0 2D slice at [i02, i03]: ggml [m, K] -> ACL [K, m]
            int64_t src0_ne[2] = { ne00, ne01 };
            size_t  src0_nb[2] = { nb00, nb01 };
            acl_tensor_ptr acl_src0_s = ggml_cann_create_tensor(
                (ਅੱਖਰ *) src0->ਅੰਕੜੇ + i02 * nb02 + i03 * nb03,
                src0_acl_type, src0_type_sz, src0_ne, src0_nb, 2);

            // src1 transposed 2D slice at [i2, i3]: swap ne/nb -> ggml[K,n] -> ACL[n,K]
            int64_t src1_t_ne[2] = { ne11, ne10 };
            size_t  src1_t_nb[2] = { nb11, nb10 };
            acl_tensor_ptr acl_src1_t = ggml_cann_create_tensor(
                (ਅੱਖਰ *) src1->ਅੰਕੜੇ + i2 * nb12 + i3 * nb13,
                src1_acl_type, src1_type_sz, src1_t_ne, src1_t_nb, 2);

            // dst 2D slice at [i2, i3]: ggml [m, n] -> ACL [n, m]
            int64_t dst_ne[2] = { ne0, ne1 };
            size_t  dst_nb[2] = { nb0, nb1 };
            acl_tensor_ptr acl_dst_s = ggml_cann_create_tensor(
                (ਅੱਖਰ *) dst->ਅੰਕੜੇ + i2 * nb2 + i3 * nb3,
                dst_acl_type, dst_type_sz, dst_ne, dst_nb, 2);

            // Matmul(src1_t [n,K], src0 [K,m]) = [n,m] = acl_dst_s  ✓
            GGML_CANN_CALL_ACLNN_OP(ctx, Matmul,
                acl_src1_t.get(), acl_src0_s.get(), acl_dst_s.get(), (int8_t) 1);
        }
    }
}

ਖਾਲੀ ggml_cann_ਬਾਹਰਲਾ_prod(ggml_backend_cann_context & ctx, ggml_tensor * dst) {
    ggml_tensor * src0 = dst->src[0];

    ਅਚਲ ਗਿਣਤੀ ggml_type type = src0->type;

    ਚੋਣ (type) {
        ਹਾਲਤ GGML_TYPE_F32:
        ਹਾਲਤ GGML_TYPE_F16:
            ggml_cann_ਬਾਹਰਲਾ_prod_fp(ctx, dst);
            ਰੋਕੋ;
        ਮੂਲ:
            GGML_ABORT("Unsupport type for GGML_OP_OUT_PROD");
            ਰੋਕੋ;
    }
}

ਖਾਲੀ ggml_cann_ssm_conv(ggml_backend_cann_context & ctx, ggml_tensor * dst) {
    ggml_tensor * src0 = dst->src[0];  // conv_x
    ggml_tensor * src1 = dst->src[1];  // conv1d.weight

    // This op is currently defined only for F32 in ggml_cpu
    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    // Shapes follow ggml_compute_forward_ssm_conv_f32
    ਅਚਲ int64_t nc  = src1->ne[0];   // d_conv
    ਅਚਲ int64_t ncs = src0->ne[0];   // d_conv - 1 + n_t
    ਅਚਲ int64_t nr  = src0->ne[1];   // d_inner
    ਅਚਲ int64_t n_s = src0->ne[2];   // n_seqs

    ਅਚਲ int64_t n_t = dst->ne[1];    // tokens per sequence

    GGML_ASSERT(dst->ne[0] == nr);     // dst: {d_inner, n_t, n_s}
    GGML_ASSERT(src1->ne[1] == nr);    // weight: {d_conv, d_inner}
    GGML_ASSERT(ncs == nc - 1 + n_t);  // conv_x: {d_conv - 1 + n_t, d_inner, n_s}
    GGML_ASSERT(src0->nb[0] == ਆਕਾਰ(ਦਸ਼ਮਲਵ));
    GGML_ASSERT(src1->nb[0] == ਆਕਾਰ(ਦਸ਼ਮਲਵ));

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

    acl_tensor_ptr acl_w = ggml_cann_create_tensor(src1->ਅੰਕੜੇ, ggml_cann_type_mapping(src1->type),
                                                   ggml_type_ਮਾਪ(src1->type), w_ne, w_nb, 3, ACL_FORMAT_NCL);

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
    size_t  y_nb[GGML_MAX_DIMS] = { dst->ne[0] * ਆਕਾਰ(ਦਸ਼ਮਲਵ), ਆਕਾਰ(ਦਸ਼ਮਲਵ), dst->ne[0] * dst->ne[1] * ਆਕਾਰ(ਦਸ਼ਮਲਵ),
                                    dst->nb[3] };       // [nr, 1, nr * n_t]

    acl_tensor_ptr acl_y = ggml_cann_create_tensor(dst->ਅੰਕੜੇ, ggml_cann_type_mapping(dst->type),
                                                   ggml_type_ਮਾਪ(dst->type), y_ne, y_nb, 3, ACL_FORMAT_NCL);

    // --- Conv1d parameters: depthwise, stride 1, no padding ("valid") ---
    int64_t strideVal[1]   = { 1 };
    int64_t paddingVal[1]  = { 0 };
    int64_t dilationVal[1] = { 1 };

    acl_int_array_ptr stride   = ggml_cann_create_int_array(strideVal, 1);
    acl_int_array_ptr padding  = ggml_cann_create_int_array(paddingVal, 1);
    acl_int_array_ptr dilation = ggml_cann_create_int_array(dilationVal, 1);

    ਅਚਲ ਤਾਰਕਿਕ    transposed   = ਝੂਠ;
    ਅਚਲ int64_t groups       = nr;  // depthwise: one group per inner dim
    int8_t        cubeMathType = 0;

#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ ASCEND_310P
    cubeMathType = 1;
#ਜੇ_ਅੰਤ

    GGML_CANN_CALL_ACLNN_OP(ctx, Convolution,
                            acl_x.get(),    // input:  N, C, L_in = ncs
                            acl_w.get(),    // weight: [C, 1, K] with groups=nr
                            ਖਾਲੀ_ਸੂਚਕ,        // bias
                            stride.get(), padding.get(), dilation.get(), transposed,
                            padding.get(),  // output padding (unused for non-transposed)
                            groups, acl_y.get(), cubeMathType);
}

ਖਾਲੀ ggml_cann_op_ਜੋੜੋ_rms_norm_fused(ggml_backend_cann_context & ctx,
                                     ggml_tensor *               ਜੋੜੋ_ਗੰਢ,
                                     ggml_tensor *               rms_norm_ਗੰਢ) {
    // Get the two input tensors for ADD operation
    ggml_tensor * x1 = ਜੋੜੋ_ਗੰਢ->src[0];
    ggml_tensor * x2 = ਜੋੜੋ_ਗੰਢ->src[1];

    // Create ACL tensors for the two ADD inputs
    acl_tensor_ptr acl_x1 = ggml_cann_create_tensor(x1);
    acl_tensor_ptr acl_x2 = ggml_cann_create_tensor(x2);

    // Get epsilon parameter from rms_norm_tensor
    ਦਸ਼ਮਲਵ eps;
    ਸਮ੍ਰਿਤੀ_ਨਕਲ(&eps, rms_norm_ਗੰਢ->op_params, ਆਕਾਰ(ਦਸ਼ਮਲਵ));

    // Build gamma tensor (RMS normalization scaling factor)
    // Gamma should match the normalized dimensions (last dimension of x1)
    size_t acl_gamma_nb[GGML_MAX_DIMS];
    acl_gamma_nb[0] = ggml_type_ਮਾਪ(rms_norm_ਗੰਢ->type);
    ਲਈ (ਪੂਰਨ i = 1; i < GGML_MAX_DIMS; i++) {
        acl_gamma_nb[i] = acl_gamma_nb[i - 1] * x1->ne[i - 1];
    }
    acl_tensor_ptr acl_gamma =
        get_cache_acl_tensor(ctx, &ctx.rms_norm_one_tensor_cache.cache, ctx.rms_norm_one_tensor_cache.size, x1->ne,
                             acl_gamma_nb, rms_norm_ਗੰਢ->type,
                             1,    // dims - only the last dimension
                             1.0f  //ਮੁੱਲ
        );

    // Build rstdOut tensor (output for normalized standard deviation)
    // Shape should be the dimensions that are NOT normalized
    int64_t acl_rstd_ne[] = { 1, x1->ne[1], x1->ne[2], x1->ne[3] };
    size_t  acl_rstd_nb[GGML_MAX_DIMS - 1];
    acl_rstd_nb[0] = ਆਕਾਰ(ਦਸ਼ਮਲਵ);
    ਲਈ (ਪੂਰਨ i = 1; i < GGML_MAX_DIMS - 1; i++) {
        acl_rstd_nb[i] = acl_rstd_nb[i - 1] * acl_rstd_ne[i - 1];
    }
    acl_tensor_ptr acl_rstd =
        get_cache_acl_tensor(ctx, &ctx.rms_norm_zero_tensor_cache.cache, ctx.rms_norm_zero_tensor_cache.size,
                             acl_rstd_ne, acl_rstd_nb, GGML_TYPE_F32, GGML_MAX_DIMS,
                             0.0f  //ਮੁੱਲ
        );

    acl_tensor_ptr acl_xout = ggml_cann_create_tensor(ਜੋੜੋ_ਗੰਢ);

    // Create yOut tensor (final output after RMS normalization)
    acl_tensor_ptr acl_yout = ggml_cann_create_tensor(rms_norm_ਗੰਢ);

    // Call fused ADD + RMS_NORM operator
    GGML_CANN_CALL_ACLNN_OP(ctx, ਜੋੜੋ_Rms_Norm, acl_x1.get(), acl_x2.get(), acl_gamma.get(),
                            eps,  // double type
                            acl_yout.get(), acl_rstd.get(), acl_xout.get());
}

ਖਾਲੀ ggml_cann_gated_ਰੇਖਿਕ_attn(ggml_backend_cann_context & ctx, ggml_tensor * dst) {
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
    size_t  nb_ਅਧਾਰ   = ggml_type_ਮਾਪ(k->type);
    size_t  nb_qkg[2] = { nb_ਅਧਾਰ, nb_ਅਧਾਰ };
    size_t  nb_s[2]   = { nb_ਅਧਾਰ, D * nb_ਅਧਾਰ };
    size_t  nb_st[2]  = { nb_s[1], nb_s[0] };
    size_t  nb_vo[2]  = { nb_ਅਧਾਰ, D * nb_ਅਧਾਰ };
    size_t  nb_q[1]   = { nb_ਅਧਾਰ };

    ਅਚਲ ਦਸ਼ਮਲਵ scale = ggml_get_op_params_f32(dst, 0);

    acl_tensor_ptr acl_s     = ggml_cann_create_tensor(s, s->ne, s->nb, 2, ACL_FORMAT_ND);
    acl_tensor_ptr new_state = ggml_cann_create_tensor(dst, s->ne, s->nb, 2, ACL_FORMAT_ND, (B * L * H * D) * nb_ਅਧਾਰ);
    cann_copy(ctx, acl_s.get(), new_state.get());

    ਲਈ (int64_t b = 0; b < B; b++) {
        ਲਈ (int64_t h = 0; h < H; h++) {
            size_t         s_ਵਿੱਥ = (b * (H * D * D) + h * (D * D)) * nb_ਅਧਾਰ;
            // D * D
            acl_tensor_ptr acl_s_new =
                ggml_cann_create_tensor(dst, ne_s, nb_s, 2, ACL_FORMAT_ND, (B * L * H * D) * nb_ਅਧਾਰ + s_ਵਿੱਥ);
            acl_tensor_ptr acl_s_new_t =
                ggml_cann_create_tensor(dst, ne_st, nb_st, 2, ACL_FORMAT_ND, (B * L * H * D) * nb_ਅਧਾਰ + s_ਵਿੱਥ);
            ਲਈ (int64_t l = 0; l < L; l++) {
                size_t               qkvgo_ਵਿੱਥ = (b * (L * H * D) + l * (H * D) + h * (D)) * nb_ਅਧਾਰ;
                // D * 1
                acl_tensor_ptr       acl_k = ggml_cann_create_tensor(k, ne_qkg, nb_qkg, 2, ACL_FORMAT_ND, qkvgo_ਵਿੱਥ);
                acl_tensor_ptr       acl_g = ggml_cann_create_tensor(g, ne_qkg, nb_qkg, 2, ACL_FORMAT_ND, qkvgo_ਵਿੱਥ);
                // D
                acl_tensor_ptr       acl_q = ggml_cann_create_tensor(q, ne_q, nb_q, 1, ACL_FORMAT_ND, qkvgo_ਵਿੱਥ);
                // 1 * D
                acl_tensor_ptr       acl_v = ggml_cann_create_tensor(v, ne_vo, nb_vo, 2, ACL_FORMAT_ND, qkvgo_ਵਿੱਥ);
                // D
                acl_tensor_ptr       acl_o = ggml_cann_create_tensor(dst, ne_q, nb_q, 1, ACL_FORMAT_ND, qkvgo_ਵਿੱਥ);
                // k ⊗ v
                size_t               buf_ਮਾਪ = D * D * nb_ਅਧਾਰ;
                ggml_cann_pool_alloc ਬਫਰ_allocator(ctx.pool(), buf_ਮਾਪ);
                acl_tensor_ptr       tmp_tensor = ggml_cann_create_tensor(
                    ਬਫਰ_allocator.get(), ggml_cann_type_mapping(k->type), nb_ਅਧਾਰ, ne_s, nb_s, 2);
                aclnn_mul(ctx, acl_k.get(), acl_v.get(), tmp_tensor.get());
                //s_new = g ⊗ s_old + k ⊗ v
                aclnn_mul(ctx, acl_s_new.get(), acl_g.get(), ਖਾਲੀ_ਸੂਚਕ);
                aclnn_ਜੋੜੋ(ctx, acl_s_new.get(), tmp_tensor.get(), ਖਾਲੀ_ਸੂਚਕ);
                // compute output
                GGML_CANN_CALL_ACLNN_OP(ctx, Mv, acl_s_new_t.get(), acl_q.get(), acl_o.get(), 1);
                aclnn_muls(ctx, acl_o.get(), scale, ਖਾਲੀ_ਸੂਚਕ, ਸੱਚ);
            }
        }
    }
}
