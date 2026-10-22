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

#समावेश "conv.hpp"

अटल  व्योम conv_transpose_1d_kernel(
        स्थिर पूर्णांक s0, स्थिर पूर्णांक output_आकार,
        स्थिर पूर्णांक src0_ne0, स्थिर पूर्णांक src0_ne1, स्थिर पूर्णांक src0_ne2,
        स्थिर पूर्णांक src1_ne0, स्थिर पूर्णांक dst_ne0,
        स्थिर भग्न * src0, स्थिर भग्न * src1,  भग्न * dst,
        स्थिर sycl::nd_item<3> &item_ct1) {
    पूर्णांक global_सूचकांक = item_ct1.get_local_id(2) +
                       item_ct1.get_group(2) * item_ct1.get_local_range(2);
    अगर (global_सूचकांक >= output_आकार) {
        वापस;
    }

    पूर्णांक out_सूचकांक = global_सूचकांक / dst_ne0;

    भग्न accumulator = 0;

    क्रम (पूर्णांक c = 0; c < src0_ne2; c++) {
        पूर्णांक idx = global_सूचकांक % dst_ne0;

        पूर्णांक kernel_offset = (src0_ne0 * src0_ne1 * c) + (out_सूचकांक * src0_ne0);
        पूर्णांक निवेश_offset = src1_ne0 * c;

        क्रम (पूर्णांक i = 0; i < src1_ne0; i++) {
            अगर (!(idx >= i*s0 && idx < i*s0 + src0_ne0)) {
                जारी;
            }
            पूर्णांक weight_idx = idx - i*s0;

            भग्न kernel_weight = src0[kernel_offset + weight_idx];
            भग्न निवेश_मूल्य =  src1[निवेश_offset+i];

            accumulator += kernel_weight * निवेश_मूल्य;
        }
    }
    dst[global_सूचकांक] = accumulator;
}

अटल व्योम conv_transpose_1d_f32_f32_sycl(
    स्थिर पूर्णांक s0, स्थिर पूर्णांक output_आकार,
    स्थिर पूर्णांक src0_ne0, स्थिर पूर्णांक src0_ne1, स्थिर पूर्णांक src0_ne2,
    स्थिर पूर्णांक src1_ne0, स्थिर पूर्णांक dst_ne0,
    स्थिर भग्न *src0, स्थिर भग्न *src1, भग्न *dst,
    स्थिर queue_ptr& stream) {

    स्थिर पूर्णांक अंक_blocks = (output_आकार + SYCL_CONV_TRANPOSE_1_D_BLOCK_आकार - 1) / SYCL_CONV_TRANPOSE_1_D_BLOCK_आकार;
    स्थिर sycl::range<3> block_dims(1, 1, SYCL_CONV_TRANPOSE_1_D_BLOCK_आकार);
    स्थिर sycl::range<3> block_nums(1, 1, अंक_blocks);
    stream->parallel_for(
        sycl::nd_range<3>(
            block_nums * block_dims, block_dims),
        [=](sycl::nd_item<3> item_ct1) {
            conv_transpose_1d_kernel(
                s0, output_आकार,
                src0_ne0, src0_ne1, src0_ne2,
                src1_ne0, dst_ne0,
                src0, src1, dst, item_ct1);
        });
}

व्योम ggml_sycl_op_conv_transpose_1d(ggml_backend_sycl_context & ctx, ggml_tensor *dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    स्थिर ggml_tensor *src0 = dst->src[0];
    स्थिर ggml_tensor *src1 = dst->src[1];
    स्थिर भग्न * src0_d = (स्थिर भग्न *)src0->तथ्य;
    स्थिर भग्न * src1_d = (स्थिर भग्न *)src1->तथ्य;

    भग्न * dst_d = (भग्न *)dst->तथ्य;
    dpct::queue_ptr stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(ggml_is_contiguous(src1));

    स्थिर int32_t * opts = (स्थिर int32_t *)dst->op_params;

    स्थिर पूर्णांक s0 = opts[0];

    स्थिर int64_t output_आकार = ggml_nelements(dst);

    conv_transpose_1d_f32_f32_sycl(s0, output_आकार,
        src0->ne[0], src0->ne[1], src0->ne[2],
        src1->ne[0], dst->ne[0],
        src0_d, src1_d, dst_d, stream);
}

