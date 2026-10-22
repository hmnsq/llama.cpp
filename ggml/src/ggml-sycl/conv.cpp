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

#अन्तर्भाव "conv.hpp"

स्थिर  शून्य conv_transpose_1d_kernel(
        अचर पूर्णाङ्क s0, अचर पूर्णाङ्क output_size,
        अचर पूर्णाङ्क src0_ne0, अचर पूर्णाङ्क src0_ne1, अचर पूर्णाङ्क src0_ne2,
        अचर पूर्णाङ्क src1_ne0, अचर पूर्णाङ्क dst_ne0,
        अचर भिन्न * src0, अचर भिन्न * src1,  भिन्न * dst,
        अचर sycl::nd_item<3> &item_ct1) {
    पूर्णाङ्क global_index = item_ct1.get_local_id(2) +
                       item_ct1.get_group(2) * item_ct1.get_local_range(2);
    यदि (global_index >= output_size) {
        फर्काऊ;
    }

    पूर्णाङ्क out_index = global_index / dst_ne0;

    भिन्न accumulator = 0;

    लागि (पूर्णाङ्क c = 0; c < src0_ne2; c++) {
        पूर्णाङ्क idx = global_index % dst_ne0;

        पूर्णाङ्क kernel_offset = (src0_ne0 * src0_ne1 * c) + (out_index * src0_ne0);
        पूर्णाङ्क आगत_offset = src1_ne0 * c;

        लागि (पूर्णाङ्क i = 0; i < src1_ne0; i++) {
            यदि (!(idx >= i*s0 && idx < i*s0 + src0_ne0)) {
                जारी;
            }
            पूर्णाङ्क weight_idx = idx - i*s0;

            भिन्न kernel_weight = src0[kernel_offset + weight_idx];
            भिन्न आगत_मान =  src1[आगत_offset+i];

            accumulator += kernel_weight * आगत_मान;
        }
    }
    dst[global_index] = accumulator;
}

स्थिर शून्य conv_transpose_1d_f32_f32_sycl(
    अचर पूर्णाङ्क s0, अचर पूर्णाङ्क output_size,
    अचर पूर्णाङ्क src0_ne0, अचर पूर्णाङ्क src0_ne1, अचर पूर्णाङ्क src0_ne2,
    अचर पूर्णाङ्क src1_ne0, अचर पूर्णाङ्क dst_ne0,
    अचर भिन्न *src0, अचर भिन्न *src1, भिन्न *dst,
    अचर queue_ptr& stream) {

    अचर पूर्णाङ्क अङ्क_blocks = (output_size + SYCL_CONV_TRANPOSE_1D_BLOCK_SIZE - 1) / SYCL_CONV_TRANPOSE_1D_BLOCK_SIZE;
    अचर sycl::range<3> block_dims(1, 1, SYCL_CONV_TRANPOSE_1D_BLOCK_SIZE);
    अचर sycl::range<3> block_nums(1, 1, अङ्क_blocks);
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

शून्य ggml_sycl_op_conv_transpose_1d(ggml_backend_sycl_context & ctx, ggml_tensor *dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    अचर ggml_tensor *src0 = dst->src[0];
    अचर ggml_tensor *src1 = dst->src[1];
    अचर भिन्न * src0_d = (अचर भिन्न *)src0->data;
    अचर भिन्न * src1_d = (अचर भिन्न *)src1->data;

    भिन्न * dst_d = (भिन्न *)dst->data;
    dpct::queue_ptr stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(ggml_is_contiguous(src1));

    अचर int32_t * opts = (अचर int32_t *)dst->op_params;

    अचर पूर्णाङ्क s0 = opts[0];

    अचर int64_t output_size = ggml_nelements(dst);

    conv_transpose_1d_f32_f32_sycl(s0, output_size,
        src0->ne[0], src0->ne[1], src0->ne[2],
        src1->ne[0], dst->ne[0],
        src0_d, src1_d, dst_d, stream);
}

