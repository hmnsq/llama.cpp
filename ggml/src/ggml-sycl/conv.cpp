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

#ಅನ್ತರ್ಭಾವ "conv.hpp"

ಸ್ಥಿರ  ಶೂನ್ಯ conv_transpose_1d_kernel(
        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ s0, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ output_size,
        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ src0_ne0, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ src0_ne1, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ src0_ne2,
        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ src1_ne0, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ dst_ne0,
        ಧ್ರುವ ಭಿನ್ನ * src0, ಧ್ರುವ ಭಿನ್ನ * src1,  ಭಿನ್ನ * dst,
        ಧ್ರುವ sycl::nd_item<3> &item_ct1) {
    ಪೂರ್ಣಾಙ್ಕ global_index = item_ct1.get_local_id(2) +
                       item_ct1.get_group(2) * item_ct1.get_local_range(2);
    ಯದಿ (global_index >= output_size) {
        ಪ್ರತ್ಯಾಗಮ;
    }

    ಪೂರ್ಣಾಙ್ಕ out_index = global_index / dst_ne0;

    ಭಿನ್ನ accumulator = 0;

    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ c = 0; c < src0_ne2; c++) {
        ಪೂರ್ಣಾಙ್ಕ idx = global_index % dst_ne0;

        ಪೂರ್ಣಾಙ್ಕ kernel_offset = (src0_ne0 * src0_ne1 * c) + (out_index * src0_ne0);
        ಪೂರ್ಣಾಙ್ಕ ಒಳಹರಿವು_offset = src1_ne0 * c;

        ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < src1_ne0; i++) {
            ಯದಿ (!(idx >= i*s0 && idx < i*s0 + src0_ne0)) {
                ಅನುವರ್ತ;
            }
            ಪೂರ್ಣಾಙ್ಕ weight_idx = idx - i*s0;

            ಭಿನ್ನ kernel_weight = src0[kernel_offset + weight_idx];
            ಭಿನ್ನ ಒಳಹರಿವು_ಮೌಲ್ಯ =  src1[ಒಳಹರಿವು_offset+i];

            accumulator += kernel_weight * ಒಳಹರಿವು_ಮೌಲ್ಯ;
        }
    }
    dst[global_index] = accumulator;
}

ಸ್ಥಿರ ಶೂನ್ಯ conv_transpose_1d_f32_f32_sycl(
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ s0, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ output_size,
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ src0_ne0, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ src0_ne1, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ src0_ne2,
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ src1_ne0, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ dst_ne0,
    ಧ್ರುವ ಭಿನ್ನ *src0, ಧ್ರುವ ಭಿನ್ನ *src1, ಭಿನ್ನ *dst,
    ಧ್ರುವ queue_ptr& stream) {

    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ಅಂಕೆ_blocks = (output_size + SYCL_CONV_TRANPOSE_1D_BLOCK_SIZE - 1) / SYCL_CONV_TRANPOSE_1D_BLOCK_SIZE;
    ಧ್ರುವ sycl::range<3> block_dims(1, 1, SYCL_CONV_TRANPOSE_1D_BLOCK_SIZE);
    ಧ್ರುವ sycl::range<3> block_nums(1, 1, ಅಂಕೆ_blocks);
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

ಶೂನ್ಯ ggml_sycl_op_conv_transpose_1d(ggml_backend_sycl_context & ctx, ggml_tensor *dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    ಧ್ರುವ ggml_tensor *src0 = dst->src[0];
    ಧ್ರುವ ggml_tensor *src1 = dst->src[1];
    ಧ್ರುವ ಭಿನ್ನ * src0_d = (ಧ್ರುವ ಭಿನ್ನ *)src0->data;
    ಧ್ರುವ ಭಿನ್ನ * src1_d = (ಧ್ರುವ ಭಿನ್ನ *)src1->data;

    ಭಿನ್ನ * dst_d = (ಭಿನ್ನ *)dst->data;
    dpct::queue_ptr stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(ggml_is_contiguous(src1));

    ಧ್ರುವ int32_t * opts = (ಧ್ರುವ int32_t *)dst->op_params;

    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ s0 = opts[0];

    ಧ್ರುವ int64_t output_size = ggml_nelements(dst);

    conv_transpose_1d_f32_f32_sycl(s0, output_size,
        src0->ne[0], src0->ne[1], src0->ne[2],
        src1->ne[0], dst->ne[0],
        src0_d, src1_d, dst_d, stream);
}

