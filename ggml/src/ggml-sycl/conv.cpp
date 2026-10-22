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

#അന്തര്ഭാവ "conv.hpp"

സ്ഥിതം  ശൂന്യം conv_transpose_1d_kernel(
        സ്ഥിരം പൂർണ്ണസംഖ്യ s0, സ്ഥിരം പൂർണ്ണസംഖ്യ output_size,
        സ്ഥിരം പൂർണ്ണസംഖ്യ src0_ne0, സ്ഥിരം പൂർണ്ണസംഖ്യ src0_ne1, സ്ഥിരം പൂർണ്ണസംഖ്യ src0_ne2,
        സ്ഥിരം പൂർണ്ണസംഖ്യ src1_ne0, സ്ഥിരം പൂർണ്ണസംഖ്യ dst_ne0,
        സ്ഥിരം ദശാംശം * src0, സ്ഥിരം ദശാംശം * src1,  ദശാംശം * dst,
        സ്ഥിരം sycl::nd_item<3> &item_ct1) {
    പൂർണ്ണസംഖ്യ global_index = item_ct1.get_local_id(2) +
                       item_ct1.get_group(2) * item_ct1.get_local_range(2);
    എങ്കിൽ (global_index >= output_size) {
        തിരികെ;
    }

    പൂർണ്ണസംഖ്യ out_index = global_index / dst_ne0;

    ദശാംശം accumulator = 0;

    വേണ്ടി (പൂർണ്ണസംഖ്യ c = 0; c < src0_ne2; c++) {
        പൂർണ്ണസംഖ്യ idx = global_index % dst_ne0;

        പൂർണ്ണസംഖ്യ kernel_offset = (src0_ne0 * src0_ne1 * c) + (out_index * src0_ne0);
        പൂർണ്ണസംഖ്യ നിവേശം_offset = src1_ne0 * c;

        വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < src1_ne0; i++) {
            എങ്കിൽ (!(idx >= i*s0 && idx < i*s0 + src0_ne0)) {
                തുടരുക;
            }
            പൂർണ്ണസംഖ്യ weight_idx = idx - i*s0;

            ദശാംശം kernel_weight = src0[kernel_offset + weight_idx];
            ദശാംശം നിവേശം_മൂല്യം =  src1[നിവേശം_offset+i];

            accumulator += kernel_weight * നിവേശം_മൂല്യം;
        }
    }
    dst[global_index] = accumulator;
}

സ്ഥിതം ശൂന്യം conv_transpose_1d_f32_f32_sycl(
    സ്ഥിരം പൂർണ്ണസംഖ്യ s0, സ്ഥിരം പൂർണ്ണസംഖ്യ output_size,
    സ്ഥിരം പൂർണ്ണസംഖ്യ src0_ne0, സ്ഥിരം പൂർണ്ണസംഖ്യ src0_ne1, സ്ഥിരം പൂർണ്ണസംഖ്യ src0_ne2,
    സ്ഥിരം പൂർണ്ണസംഖ്യ src1_ne0, സ്ഥിരം പൂർണ്ണസംഖ്യ dst_ne0,
    സ്ഥിരം ദശാംശം *src0, സ്ഥിരം ദശാംശം *src1, ദശാംശം *dst,
    സ്ഥിരം queue_ptr& stream) {

    സ്ഥിരം പൂർണ്ണസംഖ്യ അക്കം_blocks = (output_size + SYCL_CONV_TRANPOSE_1D_BLOCK_SIZE - 1) / SYCL_CONV_TRANPOSE_1D_BLOCK_SIZE;
    സ്ഥിരം sycl::range<3> block_dims(1, 1, SYCL_CONV_TRANPOSE_1D_BLOCK_SIZE);
    സ്ഥിരം sycl::range<3> block_nums(1, 1, അക്കം_blocks);
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

ശൂന്യം ggml_sycl_op_conv_transpose_1d(ggml_backend_sycl_context & ctx, ggml_tensor *dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    സ്ഥിരം ggml_tensor *src0 = dst->src[0];
    സ്ഥിരം ggml_tensor *src1 = dst->src[1];
    സ്ഥിരം ദശാംശം * src0_d = (സ്ഥിരം ദശാംശം *)src0->data;
    സ്ഥിരം ദശാംശം * src1_d = (സ്ഥിരം ദശാംശം *)src1->data;

    ദശാംശം * dst_d = (ദശാംശം *)dst->data;
    dpct::queue_ptr stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(ggml_is_contiguous(src1));

    സ്ഥിരം int32_t * opts = (സ്ഥിരം int32_t *)dst->op_params;

    സ്ഥിരം പൂർണ്ണസംഖ്യ s0 = opts[0];

    സ്ഥിരം int64_t output_size = ggml_nelements(dst);

    conv_transpose_1d_f32_f32_sycl(s0, output_size,
        src0->ne[0], src0->ne[1], src0->ne[2],
        src1->ne[0], dst->ne[0],
        src0_d, src1_d, dst_d, stream);
}

