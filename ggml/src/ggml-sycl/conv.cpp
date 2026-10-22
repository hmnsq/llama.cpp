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

#அந்தர்பாவ "conv.hpp"

நிலையான  வெறுமை conv_transpose_1d_kernel(
        மாறிலி முழு s0, மாறிலி முழு output_size,
        மாறிலி முழு src0_ne0, மாறிலி முழு src0_ne1, மாறிலி முழு src0_ne2,
        மாறிலி முழு src1_ne0, மாறிலி முழு dst_ne0,
        மாறிலி தசம * src0, மாறிலி தசம * src1,  தசம * dst,
        மாறிலி sycl::nd_item<3> &item_ct1) {
    முழு global_index = item_ct1.get_local_id(2) +
                       item_ct1.get_group(2) * item_ct1.get_local_range(2);
    எனில் (global_index >= output_size) {
        திரும்பு;
    }

    முழு out_index = global_index / dst_ne0;

    தசம accumulator = 0;

    க்கு (முழு c = 0; c < src0_ne2; c++) {
        முழு idx = global_index % dst_ne0;

        முழு kernel_offset = (src0_ne0 * src0_ne1 * c) + (out_index * src0_ne0);
        முழு உள்ளீடு_offset = src1_ne0 * c;

        க்கு (முழு i = 0; i < src1_ne0; i++) {
            எனில் (!(idx >= i*s0 && idx < i*s0 + src0_ne0)) {
                தொடர்;
            }
            முழு weight_idx = idx - i*s0;

            தசம kernel_weight = src0[kernel_offset + weight_idx];
            தசம உள்ளீடு_மதிப்பு =  src1[உள்ளீடு_offset+i];

            accumulator += kernel_weight * உள்ளீடு_மதிப்பு;
        }
    }
    dst[global_index] = accumulator;
}

நிலையான வெறுமை conv_transpose_1d_f32_f32_sycl(
    மாறிலி முழு s0, மாறிலி முழு output_size,
    மாறிலி முழு src0_ne0, மாறிலி முழு src0_ne1, மாறிலி முழு src0_ne2,
    மாறிலி முழு src1_ne0, மாறிலி முழு dst_ne0,
    மாறிலி தசம *src0, மாறிலி தசம *src1, தசம *dst,
    மாறிலி queue_ptr& stream) {

    மாறிலி முழு இலக்கம்_blocks = (output_size + SYCL_CONV_TRANPOSE_1D_BLOCK_SIZE - 1) / SYCL_CONV_TRANPOSE_1D_BLOCK_SIZE;
    மாறிலி sycl::range<3> block_dims(1, 1, SYCL_CONV_TRANPOSE_1D_BLOCK_SIZE);
    மாறிலி sycl::range<3> block_nums(1, 1, இலக்கம்_blocks);
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

வெறுமை ggml_sycl_op_conv_transpose_1d(ggml_backend_sycl_context & ctx, ggml_tensor *dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    மாறிலி ggml_tensor *src0 = dst->src[0];
    மாறிலி ggml_tensor *src1 = dst->src[1];
    மாறிலி தசம * src0_d = (மாறிலி தசம *)src0->data;
    மாறிலி தசம * src1_d = (மாறிலி தசம *)src1->data;

    தசம * dst_d = (தசம *)dst->data;
    dpct::queue_ptr stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(ggml_is_contiguous(src1));

    மாறிலி int32_t * opts = (மாறிலி int32_t *)dst->op_params;

    மாறிலி முழு s0 = opts[0];

    மாறிலி int64_t output_size = ggml_nelements(dst);

    conv_transpose_1d_f32_f32_sycl(s0, output_size,
        src0->ne[0], src0->ne[1], src0->ne[2],
        src1->ne[0], dst->ne[0],
        src0_d, src1_d, dst_d, stream);
}

