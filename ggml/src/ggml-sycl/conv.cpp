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

#ଅନ୍ତର୍ଭାଵ "conv.hpp"

ସ୍ଥିର  ଶୂନ୍ୟ conv_transpose_1d_kernel(
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା s0, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା output_size,
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା src0_ne0, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା src0_ne1, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା src0_ne2,
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା src1_ne0, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା dst_ne0,
        ଅଚଳ ଦଶମିକ * src0, ଅଚଳ ଦଶମିକ * src1,  ଦଶମିକ * dst,
        ଅଚଳ sycl::nd_item<3> &item_ct1) {
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା global_index = item_ct1.get_local_id(2) +
                       item_ct1.get_group(2) * item_ct1.get_local_range(2);
    ଯଦି (global_index >= output_size) {
        ଫେରନ୍ତୁ;
    }

    ପୂର୍ଣ୍ଣସଂଖ୍ୟା out_index = global_index / dst_ne0;

    ଦଶମିକ accumulator = 0;

    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା c = 0; c < src0_ne2; c++) {
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା idx = global_index % dst_ne0;

        ପୂର୍ଣ୍ଣସଂଖ୍ୟା kernel_offset = (src0_ne0 * src0_ne1 * c) + (out_index * src0_ne0);
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା ନିବେଶ_offset = src1_ne0 * c;

        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < src1_ne0; i++) {
            ଯଦି (!(idx >= i*s0 && idx < i*s0 + src0_ne0)) {
                ଜାରି;
            }
            ପୂର୍ଣ୍ଣସଂଖ୍ୟା weight_idx = idx - i*s0;

            ଦଶମିକ kernel_weight = src0[kernel_offset + weight_idx];
            ଦଶମିକ ନିବେଶ_ମୂଲ୍ୟ =  src1[ନିବେଶ_offset+i];

            accumulator += kernel_weight * ନିବେଶ_ମୂଲ୍ୟ;
        }
    }
    dst[global_index] = accumulator;
}

ସ୍ଥିର ଶୂନ୍ୟ conv_transpose_1d_f32_f32_sycl(
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା s0, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା output_size,
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା src0_ne0, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା src0_ne1, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା src0_ne2,
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା src1_ne0, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା dst_ne0,
    ଅଚଳ ଦଶମିକ *src0, ଅଚଳ ଦଶମିକ *src1, ଦଶମିକ *dst,
    ଅଚଳ queue_ptr& stream) {

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ଅଙ୍କ_blocks = (output_size + SYCL_CONV_TRANPOSE_1D_BLOCK_SIZE - 1) / SYCL_CONV_TRANPOSE_1D_BLOCK_SIZE;
    ଅଚଳ sycl::range<3> block_dims(1, 1, SYCL_CONV_TRANPOSE_1D_BLOCK_SIZE);
    ଅଚଳ sycl::range<3> block_nums(1, 1, ଅଙ୍କ_blocks);
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

ଶୂନ୍ୟ ggml_sycl_op_conv_transpose_1d(ggml_backend_sycl_context & ctx, ggml_tensor *dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    ଅଚଳ ggml_tensor *src0 = dst->src[0];
    ଅଚଳ ggml_tensor *src1 = dst->src[1];
    ଅଚଳ ଦଶମିକ * src0_d = (ଅଚଳ ଦଶମିକ *)src0->data;
    ଅଚଳ ଦଶମିକ * src1_d = (ଅଚଳ ଦଶମିକ *)src1->data;

    ଦଶମିକ * dst_d = (ଦଶମିକ *)dst->data;
    dpct::queue_ptr stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(ggml_is_contiguous(src1));

    ଅଚଳ int32_t * opts = (ଅଚଳ int32_t *)dst->op_params;

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା s0 = opts[0];

    ଅଚଳ int64_t output_size = ggml_nelements(dst);

    conv_transpose_1d_f32_f32_sycl(s0, output_size,
        src0->ne[0], src0->ne[1], src0->ne[2],
        src1->ne[0], dst->ne[0],
        src0_d, src1_d, dst_d, stream);
}

