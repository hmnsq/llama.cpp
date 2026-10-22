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

#অন্তর্ভাব "conv.hpp"

স্থির  শূন্য conv_transpose_1d_kernel(
        ধ্রুবক পূর্ণসংখ্যা s0, ধ্রুবক পূর্ণসংখ্যা output_আকার,
        ধ্রুবক পূর্ণসংখ্যা src0_ne0, ধ্রুবক পূর্ণসংখ্যা src0_ne1, ধ্রুবক পূর্ণসংখ্যা src0_ne2,
        ধ্রুবক পূর্ণসংখ্যা src1_ne0, ধ্রুবক পূর্ণসংখ্যা dst_ne0,
        ধ্রুবক দশমিক * src0, ধ্রুবক দশমিক * src1,  দশমিক * dst,
        ধ্রুবক sycl::nd_item<3> &item_ct1) {
    পূর্ণসংখ্যা global_সূচক = item_ct1.get_local_id(2) +
                       item_ct1.get_group(2) * item_ct1.get_local_range(2);
    যদি (global_সূচক >= output_আকার) {
        ফেরত;
    }

    পূর্ণসংখ্যা out_সূচক = global_সূচক / dst_ne0;

    দশমিক accumulator = 0;

    জন্য (পূর্ণসংখ্যা c = 0; c < src0_ne2; c++) {
        পূর্ণসংখ্যা idx = global_সূচক % dst_ne0;

        পূর্ণসংখ্যা kernel_offset = (src0_ne0 * src0_ne1 * c) + (out_সূচক * src0_ne0);
        পূর্ণসংখ্যা প্রবেশ_offset = src1_ne0 * c;

        জন্য (পূর্ণসংখ্যা i = 0; i < src1_ne0; i++) {
            যদি (!(idx >= i*s0 && idx < i*s0 + src0_ne0)) {
                চালু;
            }
            পূর্ণসংখ্যা weight_idx = idx - i*s0;

            দশমিক kernel_weight = src0[kernel_offset + weight_idx];
            দশমিক প্রবেশ_মান =  src1[প্রবেশ_offset+i];

            accumulator += kernel_weight * প্রবেশ_মান;
        }
    }
    dst[global_সূচক] = accumulator;
}

স্থির শূন্য conv_transpose_1d_f32_f32_sycl(
    ধ্রুবক পূর্ণসংখ্যা s0, ধ্রুবক পূর্ণসংখ্যা output_আকার,
    ধ্রুবক পূর্ণসংখ্যা src0_ne0, ধ্রুবক পূর্ণসংখ্যা src0_ne1, ধ্রুবক পূর্ণসংখ্যা src0_ne2,
    ধ্রুবক পূর্ণসংখ্যা src1_ne0, ধ্রুবক পূর্ণসংখ্যা dst_ne0,
    ধ্রুবক দশমিক *src0, ধ্রুবক দশমিক *src1, দশমিক *dst,
    ধ্রুবক queue_ptr& stream) {

    ধ্রুবক পূর্ণসংখ্যা অঙ্ক_blocks = (output_আকার + SYCL_CONV_TRANPOSE_1_D_BLOCK_আকার - 1) / SYCL_CONV_TRANPOSE_1_D_BLOCK_আকার;
    ধ্রুবক sycl::range<3> block_dims(1, 1, SYCL_CONV_TRANPOSE_1_D_BLOCK_আকার);
    ধ্রুবক sycl::range<3> block_nums(1, 1, অঙ্ক_blocks);
    stream->parallel_for(
        sycl::nd_range<3>(
            block_nums * block_dims, block_dims),
        [=](sycl::nd_item<3> item_ct1) {
            conv_transpose_1d_kernel(
                s0, output_আকার,
                src0_ne0, src0_ne1, src0_ne2,
                src1_ne0, dst_ne0,
                src0, src1, dst, item_ct1);
        });
}

শূন্য ggml_sycl_op_conv_transpose_1d(ggml_backend_sycl_context & ctx, ggml_tensor *dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    ধ্রুবক ggml_tensor *src0 = dst->src[0];
    ধ্রুবক ggml_tensor *src1 = dst->src[1];
    ধ্রুবক দশমিক * src0_d = (ধ্রুবক দশমিক *)src0->তথ্য;
    ধ্রুবক দশমিক * src1_d = (ধ্রুবক দশমিক *)src1->তথ্য;

    দশমিক * dst_d = (দশমিক *)dst->তথ্য;
    dpct::queue_ptr stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(ggml_is_contiguous(src1));

    ধ্রুবক int32_t * opts = (ধ্রুবক int32_t *)dst->op_params;

    ধ্রুবক পূর্ণসংখ্যা s0 = opts[0];

    ধ্রুবক int64_t output_আকার = ggml_nelements(dst);

    conv_transpose_1d_f32_f32_sycl(s0, output_আকার,
        src0->ne[0], src0->ne[1], src0->ne[2],
        src1->ne[0], dst->ne[0],
        src0_d, src1_d, dst_d, stream);
}

