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

#অন্তর্ভাৱ "conv.hpp"

স্থির  শূন্য conv_transpose_1d_kernel(
        ধ্রুৱ পূর্ণাঙ্ক s0, ধ্রুৱ পূর্ণাঙ্ক output_size,
        ধ্রুৱ পূর্ণাঙ্ক src0_ne0, ধ্রুৱ পূর্ণাঙ্ক src0_ne1, ধ্রুৱ পূর্ণাঙ্ক src0_ne2,
        ধ্রুৱ পূর্ণাঙ্ক src1_ne0, ধ্রুৱ পূর্ণাঙ্ক dst_ne0,
        ধ্রুৱ ভিন্ন * src0, ধ্রুৱ ভিন্ন * src1,  ভিন্ন * dst,
        ধ্রুৱ sycl::nd_item<3> &item_ct1) {
    পূর্ণাঙ্ক global_index = item_ct1.get_local_id(2) +
                       item_ct1.get_group(2) * item_ct1.get_local_range(2);
    যদি (global_index >= output_size) {
        ঘূৰাই;
    }

    পূর্ণাঙ্ক out_index = global_index / dst_ne0;

    ভিন্ন accumulator = 0;

    বাবে (পূর্ণাঙ্ক c = 0; c < src0_ne2; c++) {
        পূর্ণাঙ্ক idx = global_index % dst_ne0;

        পূর্ণাঙ্ক kernel_offset = (src0_ne0 * src0_ne1 * c) + (out_index * src0_ne0);
        পূর্ণাঙ্ক প্ৰবেশ_offset = src1_ne0 * c;

        বাবে (পূর্ণাঙ্ক i = 0; i < src1_ne0; i++) {
            যদি (!(idx >= i*s0 && idx < i*s0 + src0_ne0)) {
                অনুৱর্ত;
            }
            পূর্ণাঙ্ক weight_idx = idx - i*s0;

            ভিন্ন kernel_weight = src0[kernel_offset + weight_idx];
            ভিন্ন প্ৰবেশ_মান =  src1[প্ৰবেশ_offset+i];

            accumulator += kernel_weight * প্ৰবেশ_মান;
        }
    }
    dst[global_index] = accumulator;
}

স্থির শূন্য conv_transpose_1d_f32_f32_sycl(
    ধ্রুৱ পূর্ণাঙ্ক s0, ধ্রুৱ পূর্ণাঙ্ক output_size,
    ধ্রুৱ পূর্ণাঙ্ক src0_ne0, ধ্রুৱ পূর্ণাঙ্ক src0_ne1, ধ্রুৱ পূর্ণাঙ্ক src0_ne2,
    ধ্রুৱ পূর্ণাঙ্ক src1_ne0, ধ্রুৱ পূর্ণাঙ্ক dst_ne0,
    ধ্রুৱ ভিন্ন *src0, ধ্রুৱ ভিন্ন *src1, ভিন্ন *dst,
    ধ্রুৱ queue_ptr& stream) {

    ধ্রুৱ পূর্ণাঙ্ক অঙ্ক_blocks = (output_size + SYCL_CONV_TRANPOSE_1D_BLOCK_SIZE - 1) / SYCL_CONV_TRANPOSE_1D_BLOCK_SIZE;
    ধ্রুৱ sycl::range<3> block_dims(1, 1, SYCL_CONV_TRANPOSE_1D_BLOCK_SIZE);
    ধ্রুৱ sycl::range<3> block_nums(1, 1, অঙ্ক_blocks);
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

শূন্য ggml_sycl_op_conv_transpose_1d(ggml_backend_sycl_context & ctx, ggml_tensor *dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    ধ্রুৱ ggml_tensor *src0 = dst->src[0];
    ধ্রুৱ ggml_tensor *src1 = dst->src[1];
    ধ্রুৱ ভিন্ন * src0_d = (ধ্রুৱ ভিন্ন *)src0->data;
    ধ্রুৱ ভিন্ন * src1_d = (ধ্রুৱ ভিন্ন *)src1->data;

    ভিন্ন * dst_d = (ভিন্ন *)dst->data;
    dpct::queue_ptr stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(ggml_is_contiguous(src1));

    ধ্রুৱ int32_t * opts = (ধ্রুৱ int32_t *)dst->op_params;

    ধ্রুৱ পূর্ণাঙ্ক s0 = opts[0];

    ধ্রুৱ int64_t output_size = ggml_nelements(dst);

    conv_transpose_1d_f32_f32_sycl(s0, output_size,
        src0->ne[0], src0->ne[1], src0->ne[2],
        src1->ne[0], dst->ne[0],
        src0_d, src1_d, dst_d, stream);
}

