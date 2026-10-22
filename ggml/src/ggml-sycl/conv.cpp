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
        स्थिरांक पूर्णांक s0, स्थिरांक पूर्णांक output_size,
        स्थिरांक पूर्णांक src0_ne0, स्थिरांक पूर्णांक src0_ne1, स्थिरांक पूर्णांक src0_ne2,
        स्थिरांक पूर्णांक src1_ne0, स्थिरांक पूर्णांक dst_ne0,
        स्थिरांक दशांश * src0, स्थिरांक दशांश * src1,  दशांश * dst,
        स्थिरांक sycl::nd_item<3> &item_ct1) {
    पूर्णांक global_index = item_ct1.get_local_id(2) +
                       item_ct1.get_group(2) * item_ct1.get_local_range(2);
    जर (global_index >= output_size) {
        परत;
    }

    पूर्णांक out_index = global_index / dst_ne0;

    दशांश accumulator = 0;

    साठी (पूर्णांक c = 0; c < src0_ne2; c++) {
        पूर्णांक idx = global_index % dst_ne0;

        पूर्णांक kernel_offset = (src0_ne0 * src0_ne1 * c) + (out_index * src0_ne0);
        पूर्णांक आदान_offset = src1_ne0 * c;

        साठी (पूर्णांक i = 0; i < src1_ne0; i++) {
            जर (!(idx >= i*s0 && idx < i*s0 + src0_ne0)) {
                सुरूठेवा;
            }
            पूर्णांक weight_idx = idx - i*s0;

            दशांश kernel_weight = src0[kernel_offset + weight_idx];
            दशांश आदान_मूल्य =  src1[आदान_offset+i];

            accumulator += kernel_weight * आदान_मूल्य;
        }
    }
    dst[global_index] = accumulator;
}

स्थिर शून्य conv_transpose_1d_f32_f32_sycl(
    स्थिरांक पूर्णांक s0, स्थिरांक पूर्णांक output_size,
    स्थिरांक पूर्णांक src0_ne0, स्थिरांक पूर्णांक src0_ne1, स्थिरांक पूर्णांक src0_ne2,
    स्थिरांक पूर्णांक src1_ne0, स्थिरांक पूर्णांक dst_ne0,
    स्थिरांक दशांश *src0, स्थिरांक दशांश *src1, दशांश *dst,
    स्थिरांक queue_ptr& stream) {

    स्थिरांक पूर्णांक अंक_blocks = (output_size + SYCL_CONV_TRANPOSE_1D_BLOCK_SIZE - 1) / SYCL_CONV_TRANPOSE_1D_BLOCK_SIZE;
    स्थिरांक sycl::range<3> block_dims(1, 1, SYCL_CONV_TRANPOSE_1D_BLOCK_SIZE);
    स्थिरांक sycl::range<3> block_nums(1, 1, अंक_blocks);
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
    स्थिरांक ggml_tensor *src0 = dst->src[0];
    स्थिरांक ggml_tensor *src1 = dst->src[1];
    स्थिरांक दशांश * src0_d = (स्थिरांक दशांश *)src0->data;
    स्थिरांक दशांश * src1_d = (स्थिरांक दशांश *)src1->data;

    दशांश * dst_d = (दशांश *)dst->data;
    dpct::queue_ptr stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(ggml_is_contiguous(src1));

    स्थिरांक int32_t * opts = (स्थिरांक int32_t *)dst->op_params;

    स्थिरांक पूर्णांक s0 = opts[0];

    स्थिरांक int64_t output_size = ggml_nelements(dst);

    conv_transpose_1d_f32_f32_sycl(s0, output_size,
        src0->ne[0], src0->ne[1], src0->ne[2],
        src1->ne[0], dst->ne[0],
        src0_d, src1_d, dst_d, stream);
}

