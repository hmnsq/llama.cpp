//
// MIT license
// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: MIT
//

//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

#ਅਨ੍ਤਰ੍ਭਾਵ "conv.hpp"

ਸਥਿਰ  ਖਾਲੀ conv_transpose_1d_kernel(
        ਅਚਲ ਪੂਰਨ s0, ਅਚਲ ਪੂਰਨ output_ਮਾਪ,
        ਅਚਲ ਪੂਰਨ src0_ne0, ਅਚਲ ਪੂਰਨ src0_ne1, ਅਚਲ ਪੂਰਨ src0_ne2,
        ਅਚਲ ਪੂਰਨ src1_ne0, ਅਚਲ ਪੂਰਨ dst_ne0,
        ਅਚਲ ਦਸ਼ਮਲਵ * src0, ਅਚਲ ਦਸ਼ਮਲਵ * src1,  ਦਸ਼ਮਲਵ * dst,
        ਅਚਲ sycl::nd_ਵਸਤੂ<3> &ਵਸਤੂ_ct1) {
    ਪੂਰਨ global_ਸੂਚਕਾਂਕ = ਵਸਤੂ_ct1.get_local_id(2) +
                       ਵਸਤੂ_ct1.get_ਸਮੂਹ(2) * ਵਸਤੂ_ct1.get_local_range(2);
    ਜੇ (global_ਸੂਚਕਾਂਕ >= output_ਮਾਪ) {
        ਵਾਪਸ;
    }

    ਪੂਰਨ ਬਾਹਰਲਾ_ਸੂਚਕਾਂਕ = global_ਸੂਚਕਾਂਕ / dst_ne0;

    ਦਸ਼ਮਲਵ ਸੰਚਾਇਕ = 0;

    ਲਈ (ਪੂਰਨ c = 0; c < src0_ne2; c++) {
        ਪੂਰਨ idx = global_ਸੂਚਕਾਂਕ % dst_ne0;

        ਪੂਰਨ kernel_ਵਿੱਥ = (src0_ne0 * src0_ne1 * c) + (ਬਾਹਰਲਾ_ਸੂਚਕਾਂਕ * src0_ne0);
        ਪੂਰਨ ਨਿਵੇਸ਼_ਵਿੱਥ = src1_ne0 * c;

        ਲਈ (ਪੂਰਨ i = 0; i < src1_ne0; i++) {
            ਜੇ (!(idx >= i*s0 && idx < i*s0 + src0_ne0)) {
                ਜਾਰੀ;
            }
            ਪੂਰਨ ਭਾਰ_idx = idx - i*s0;

            ਦਸ਼ਮਲਵ kernel_ਭਾਰ = src0[kernel_ਵਿੱਥ + ਭਾਰ_idx];
            ਦਸ਼ਮਲਵ ਨਿਵੇਸ਼_ਮੁੱਲ =  src1[ਨਿਵੇਸ਼_ਵਿੱਥ+i];

            ਸੰਚਾਇਕ += kernel_ਭਾਰ * ਨਿਵੇਸ਼_ਮੁੱਲ;
        }
    }
    dst[global_ਸੂਚਕਾਂਕ] = ਸੰਚਾਇਕ;
}

ਸਥਿਰ ਖਾਲੀ conv_transpose_1d_f32_f32_sycl(
    ਅਚਲ ਪੂਰਨ s0, ਅਚਲ ਪੂਰਨ output_ਮਾਪ,
    ਅਚਲ ਪੂਰਨ src0_ne0, ਅਚਲ ਪੂਰਨ src0_ne1, ਅਚਲ ਪੂਰਨ src0_ne2,
    ਅਚਲ ਪੂਰਨ src1_ne0, ਅਚਲ ਪੂਰਨ dst_ne0,
    ਅਚਲ ਦਸ਼ਮਲਵ *src0, ਅਚਲ ਦਸ਼ਮਲਵ *src1, ਦਸ਼ਮਲਵ *dst,
    ਅਚਲ ਕਤਾਰ_ptr& stream) {

    ਅਚਲ ਪੂਰਨ ਅੰਕ_ਖੰਡਾਂ = (output_ਮਾਪ + SYCL_CONV_TRANPOSE_1_D_ਖੰਡ_ਮਾਪ - 1) / SYCL_CONV_TRANPOSE_1_D_ਖੰਡ_ਮਾਪ;
    ਅਚਲ sycl::range<3> ਖੰਡ_dims(1, 1, SYCL_CONV_TRANPOSE_1_D_ਖੰਡ_ਮਾਪ);
    ਅਚਲ sycl::range<3> ਖੰਡ_nums(1, 1, ਅੰਕ_ਖੰਡਾਂ);
    stream->parallel_for(
        sycl::nd_range<3>(
            ਖੰਡ_nums * ਖੰਡ_dims, ਖੰਡ_dims),
        [=](sycl::nd_ਵਸਤੂ<3> ਵਸਤੂ_ct1) {
            conv_transpose_1d_kernel(
                s0, output_ਮਾਪ,
                src0_ne0, src0_ne1, src0_ne2,
                src1_ne0, dst_ne0,
                src0, src1, dst, ਵਸਤੂ_ct1);
        });
}

ਖਾਲੀ ggml_sycl_op_conv_transpose_1d(ggml_backend_sycl_context & ctx, ggml_tensor *dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    ਅਚਲ ggml_tensor *src0 = dst->src[0];
    ਅਚਲ ggml_tensor *src1 = dst->src[1];
    ਅਚਲ ਦਸ਼ਮਲਵ * src0_d = (ਅਚਲ ਦਸ਼ਮਲਵ *)src0->ਅੰਕੜੇ;
    ਅਚਲ ਦਸ਼ਮਲਵ * src1_d = (ਅਚਲ ਦਸ਼ਮਲਵ *)src1->ਅੰਕੜੇ;

    ਦਸ਼ਮਲਵ * dst_d = (ਦਸ਼ਮਲਵ *)dst->ਅੰਕੜੇ;
    dpct::ਕਤਾਰ_ptr stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(ggml_is_contiguous(src1));

    ਅਚਲ int32_t * opts = (ਅਚਲ int32_t *)dst->op_params;

    ਅਚਲ ਪੂਰਨ s0 = opts[0];

    ਅਚਲ int64_t output_ਮਾਪ = ggml_nelements(dst);

    conv_transpose_1d_f32_f32_sycl(s0, output_ਮਾਪ,
        src0->ne[0], src0->ne[1], src0->ne[2],
        src1->ne[0], dst->ne[0],
        src0_d, src1_d, dst_d, stream);
}

