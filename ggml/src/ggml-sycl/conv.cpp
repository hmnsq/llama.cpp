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

#અન્તર્ભાવ "conv.hpp"

સ્થિર  શૂન્ય conv_transpose_1d_kernel(
        અચલ પૂર્ણાંક s0, અચલ પૂર્ણાંક output_size,
        અચલ પૂર્ણાંક src0_ne0, અચલ પૂર્ણાંક src0_ne1, અચલ પૂર્ણાંક src0_ne2,
        અચલ પૂર્ણાંક src1_ne0, અચલ પૂર્ણાંક dst_ne0,
        અચલ દશાંશ * src0, અચલ દશાંશ * src1,  દશાંશ * dst,
        અચલ sycl::nd_item<3> &item_ct1) {
    પૂર્ણાંક global_index = item_ct1.get_local_id(2) +
                       item_ct1.get_group(2) * item_ct1.get_local_range(2);
    જો (global_index >= output_size) {
        પાછા;
    }

    પૂર્ણાંક out_index = global_index / dst_ne0;

    દશાંશ accumulator = 0;

    માટે (પૂર્ણાંક c = 0; c < src0_ne2; c++) {
        પૂર્ણાંક idx = global_index % dst_ne0;

        પૂર્ણાંક kernel_offset = (src0_ne0 * src0_ne1 * c) + (out_index * src0_ne0);
        પૂર્ણાંક આગત_offset = src1_ne0 * c;

        માટે (પૂર્ણાંક i = 0; i < src1_ne0; i++) {
            જો (!(idx >= i*s0 && idx < i*s0 + src0_ne0)) {
                ચાલુ;
            }
            પૂર્ણાંક weight_idx = idx - i*s0;

            દશાંશ kernel_weight = src0[kernel_offset + weight_idx];
            દશાંશ આગત_કિંમત =  src1[આગત_offset+i];

            accumulator += kernel_weight * આગત_કિંમત;
        }
    }
    dst[global_index] = accumulator;
}

સ્થિર શૂન્ય conv_transpose_1d_f32_f32_sycl(
    અચલ પૂર્ણાંક s0, અચલ પૂર્ણાંક output_size,
    અચલ પૂર્ણાંક src0_ne0, અચલ પૂર્ણાંક src0_ne1, અચલ પૂર્ણાંક src0_ne2,
    અચલ પૂર્ણાંક src1_ne0, અચલ પૂર્ણાંક dst_ne0,
    અચલ દશાંશ *src0, અચલ દશાંશ *src1, દશાંશ *dst,
    અચલ queue_ptr& stream) {

    અચલ પૂર્ણાંક અંક_blocks = (output_size + SYCL_CONV_TRANPOSE_1D_BLOCK_SIZE - 1) / SYCL_CONV_TRANPOSE_1D_BLOCK_SIZE;
    અચલ sycl::range<3> block_dims(1, 1, SYCL_CONV_TRANPOSE_1D_BLOCK_SIZE);
    અચલ sycl::range<3> block_nums(1, 1, અંક_blocks);
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

શૂન્ય ggml_sycl_op_conv_transpose_1d(ggml_backend_sycl_context & ctx, ggml_tensor *dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    અચલ ggml_tensor *src0 = dst->src[0];
    અચલ ggml_tensor *src1 = dst->src[1];
    અચલ દશાંશ * src0_d = (અચલ દશાંશ *)src0->data;
    અચલ દશાંશ * src1_d = (અચલ દશાંશ *)src1->data;

    દશાંશ * dst_d = (દશાંશ *)dst->data;
    dpct::queue_ptr stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(ggml_is_contiguous(src1));

    અચલ int32_t * opts = (અચલ int32_t *)dst->op_params;

    અચલ પૂર્ણાંક s0 = opts[0];

    અચલ int64_t output_size = ggml_nelements(dst);

    conv_transpose_1d_f32_f32_sycl(s0, output_size,
        src0->ne[0], src0->ne[1], src0->ne[2],
        src1->ne[0], dst->ne[0],
        src0_d, src1_d, dst_d, stream);
}

