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

#अन्तर्भाव "tsembd.hpp"

स्थिर शून्य timestep_embedding_f32(
        अचर भिन्न * timesteps, भिन्न * dst, अचर पूर्णाङ्क nb1,
        अचर पूर्णाङ्क dim, अचर पूर्णाङ्क max_period, अचर sycl::nd_item<3> &item_ct1) {
    // item_ct1.get_group(1)(blockIDx.y): idx of timesteps->ne[0]
    // item_ct1.get_group(2) (blockIDx.x): idx of ((dim + 1) / 2) / BLOCK_SIZE
    पूर्णाङ्क i = item_ct1.get_group(1);
    पूर्णाङ्क j = item_ct1.get_local_id(2) + item_ct1.get_group(2) * item_ct1.get_local_range(2);
    भिन्न * embed_data = (भिन्न *)((वर्ण *)dst +  i*nb1);

    पूर्णाङ्क half = dim / 2;

    यदि (dim % 2 != 0 && j == half) {
        embed_data[2 * half] = 0.f;
    }

    यदि (j >= half) {
        फर्काऊ;
    }

    भिन्न timestep = timesteps[i];
    भिन्न freq = (भिन्न)sycl::native::exp(-(sycl::log((भिन्न)max_period)) * j / half);
    भिन्न arg = timestep * freq;
    embed_data[j] = sycl::cos(arg);
    embed_data[j + half] = sycl::sin(arg);
}

स्थिर शून्य timestep_embedding_f32_sycl(
        अचर भिन्न * x, भिन्न * dst, अचर पूर्णाङ्क ne00, अचर पूर्णाङ्क nb1,
        अचर पूर्णाङ्क dim, अचर पूर्णाङ्क max_period, अचर queue_ptr& stream) {
    // As the kernel returns when thread.idx is larger than dim/2, the half_ceil does not need to pad
    पूर्णाङ्क half_ceil = dim / 2;
    पूर्णाङ्क अङ्क_blocks = (half_ceil + SYCL_TIMESTEP_EMBEDDING_BLOCK_SIZE - 1) / SYCL_TIMESTEP_EMBEDDING_BLOCK_SIZE;
    sycl::range<3> block_dims(1, 1, SYCL_TIMESTEP_EMBEDDING_BLOCK_SIZE);
    sycl::range<3> gridDim(1, ne00, अङ्क_blocks);
    stream->parallel_for(
        sycl::nd_range<3>(
            gridDim * block_dims, block_dims),
        [=](sycl::nd_item<3> item_ct1) {
            timestep_embedding_f32(
                x, dst, nb1, dim, max_period, item_ct1
            );
        });
}

शून्य ggml_sycl_op_timestep_embedding(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    अचर ggml_tensor *  src0   = dst->src[0];
    अचर भिन्न * src0_d = (अचर भिन्न *)src0->data;
    भिन्न * dst_d = (भिन्न *)dst->data;
    dpct::queue_ptr stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    अचर पूर्णाङ्क dim = dst->op_params[0];
    अचर पूर्णाङ्क max_period = dst->op_params[1];

    timestep_embedding_f32_sycl(src0_d, dst_d, src0->ne[0], dst->nb[1], dim, max_period, stream);
}
