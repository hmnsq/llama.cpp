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

#include "conv.hpp"

static  void conv_transpose_1d_kernel(
        const පූර්ණ_සංඛ්‍යා s0, const පූර්ණ_සංඛ්‍යා output_size,
        const පූර්ණ_සංඛ්‍යා src0_ne0, const පූර්ණ_සංඛ්‍යා src0_ne1, const පූර්ණ_සංඛ්‍යා src0_ne2,
        const පූර්ණ_සංඛ්‍යා src1_ne0, const පූර්ණ_සංඛ්‍යා dst_ne0,
        const float * src0, const float * src1,  float * dst,
        const sycl::nd_item<3> &item_ct1) {
    පූර්ණ_සංඛ්‍යා global_index = item_ct1.get_local_id(2) +
                       item_ct1.get_group(2) * item_ct1.get_local_range(2);
    නම් (global_index >= output_size) {
        ආපසු;
    }

    පූර්ණ_සංඛ්‍යා out_index = global_index / dst_ne0;

    float accumulator = 0;

    සඳහා (පූර්ණ_සංඛ්‍යා c = 0; c < src0_ne2; c++) {
        පූර්ණ_සංඛ්‍යා idx = global_index % dst_ne0;

        පූර්ණ_සංඛ්‍යා kernel_offset = (src0_ne0 * src0_ne1 * c) + (out_index * src0_ne0);
        පූර්ණ_සංඛ්‍යා input_offset = src1_ne0 * c;

        සඳහා (පූර්ණ_සංඛ්‍යා i = 0; i < src1_ne0; i++) {
            නම් (!(idx >= i*s0 && idx < i*s0 + src0_ne0)) {
                continue;
            }
            පූර්ණ_සංඛ්‍යා weight_idx = idx - i*s0;

            float kernel_weight = src0[kernel_offset + weight_idx];
            float input_value =  src1[input_offset+i];

            accumulator += kernel_weight * input_value;
        }
    }
    dst[global_index] = accumulator;
}

static void conv_transpose_1d_f32_f32_sycl(
    const පූර්ණ_සංඛ්‍යා s0, const පූර්ණ_සංඛ්‍යා output_size,
    const පූර්ණ_සංඛ්‍යා src0_ne0, const පූර්ණ_සංඛ්‍යා src0_ne1, const පූර්ණ_සංඛ්‍යා src0_ne2,
    const පූර්ණ_සංඛ්‍යා src1_ne0, const පූර්ණ_සංඛ්‍යා dst_ne0,
    const float *src0, const float *src1, float *dst,
    const queue_ptr& stream) {

    const පූර්ණ_සංඛ්‍යා num_blocks = (output_size + SYCL_CONV_TRANPOSE_1D_BLOCK_SIZE - 1) / SYCL_CONV_TRANPOSE_1D_BLOCK_SIZE;
    const sycl::range<3> block_dims(1, 1, SYCL_CONV_TRANPOSE_1D_BLOCK_SIZE);
    const sycl::range<3> block_nums(1, 1, num_blocks);
    stream->parallel_for(
        sycl::nd_range<3>(
            block_nums * block_dims, block_dims),
        [=](sycl::nd_item<3> item_ct1) {
            conv_transpose_1d_kernel(
                s0, output_size,
                src0_ne0, src0_ne1, src0_ne2,
                src1_ne0, dst_ne0,
                src0, src1, dst, item_ct1);
        });
}

void ggml_sycl_op_conv_transpose_1d(ggml_backend_sycl_context & ctx, ggml_tensor *dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    const ggml_tensor *src0 = dst->src[0];
    const ggml_tensor *src1 = dst->src[1];
    const float * src0_d = (const float *)src0->data;
    const float * src1_d = (const float *)src1->data;

    float * dst_d = (float *)dst->data;
    dpct::queue_ptr stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(ggml_is_contiguous(src1));

    const int32_t * opts = (const int32_t *)dst->op_params;

    const පූර්ණ_සංඛ්‍යා s0 = opts[0];

    const int64_t output_size = ggml_nelements(dst);

    conv_transpose_1d_f32_f32_sycl(s0, output_size,
        src0->ne[0], src0->ne[1], src0->ne[2],
        src1->ne[0], dst->ne[0],
        src0_d, src1_d, dst_d, stream);
}

