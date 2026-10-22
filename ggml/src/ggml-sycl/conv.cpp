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

#شامل "conv.hpp"

ایستا  تهی conv_transpose_1d_kernel(
        ثابت صحیح s0, ثابت صحیح output_size,
        ثابت صحیح src0_ne0, ثابت صحیح src0_ne1, ثابت صحیح src0_ne2,
        ثابت صحیح src1_ne0, ثابت صحیح dst_ne0,
        ثابت اعشاری * src0, ثابت اعشاری * src1,  اعشاری * dst,
        ثابت sycl::nd_item<3> &item_ct1) {
    صحیح global_index = item_ct1.get_local_id(2) +
                       item_ct1.get_group(2) * item_ct1.get_local_range(2);
    اگر (global_index >= output_size) {
        بازگشت;
    }

    صحیح out_index = global_index / dst_ne0;

    اعشاری accumulator = 0;

    برای (صحیح c = 0; c < src0_ne2; c++) {
        صحیح idx = global_index % dst_ne0;

        صحیح kernel_offset = (src0_ne0 * src0_ne1 * c) + (out_index * src0_ne0);
        صحیح درون_داد_offset = src1_ne0 * c;

        برای (صحیح i = 0; i < src1_ne0; i++) {
            اگر (!(idx >= i*s0 && idx < i*s0 + src0_ne0)) {
                ادامه;
            }
            صحیح weight_idx = idx - i*s0;

            اعشاری kernel_weight = src0[kernel_offset + weight_idx];
            اعشاری درون_داد_مقدار =  src1[درون_داد_offset+i];

            accumulator += kernel_weight * درون_داد_مقدار;
        }
    }
    dst[global_index] = accumulator;
}

ایستا تهی conv_transpose_1d_f32_f32_sycl(
    ثابت صحیح s0, ثابت صحیح output_size,
    ثابت صحیح src0_ne0, ثابت صحیح src0_ne1, ثابت صحیح src0_ne2,
    ثابت صحیح src1_ne0, ثابت صحیح dst_ne0,
    ثابت اعشاری *src0, ثابت اعشاری *src1, اعشاری *dst,
    ثابت queue_ptr& stream) {

    ثابت صحیح رقم_blocks = (output_size + SYCL_CONV_TRANPOSE_1D_BLOCK_SIZE - 1) / SYCL_CONV_TRANPOSE_1D_BLOCK_SIZE;
    ثابت sycl::range<3> block_dims(1, 1, SYCL_CONV_TRANPOSE_1D_BLOCK_SIZE);
    ثابت sycl::range<3> block_nums(1, 1, رقم_blocks);
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

تهی ggml_sycl_op_conv_transpose_1d(ggml_backend_sycl_context & ctx, ggml_tensor *dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    ثابت ggml_tensor *src0 = dst->src[0];
    ثابت ggml_tensor *src1 = dst->src[1];
    ثابت اعشاری * src0_d = (ثابت اعشاری *)src0->data;
    ثابت اعشاری * src1_d = (ثابت اعشاری *)src1->data;

    اعشاری * dst_d = (اعشاری *)dst->data;
    dpct::queue_ptr stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(ggml_is_contiguous(src1));

    ثابت int32_t * opts = (ثابت int32_t *)dst->op_params;

    ثابت صحیح s0 = opts[0];

    ثابت int64_t output_size = ggml_nelements(dst);

    conv_transpose_1d_f32_f32_sycl(s0, output_size,
        src0->ne[0], src0->ne[1], src0->ne[2],
        src1->ne[0], dst->ne[0],
        src0_d, src1_d, dst_d, stream);
}

