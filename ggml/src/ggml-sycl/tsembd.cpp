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
        स्थिरांक दशांश * timesteps, दशांश * dst, स्थिरांक पूर्णांक nb1,
        स्थिरांक पूर्णांक dim, स्थिरांक पूर्णांक max_period, स्थिरांक sycl::nd_item<3> &item_ct1) {
    // item_ct1.get_group(1)(blockIDx.y): idx of timesteps->ne[0]
    // item_ct1.get_group(2) (blockIDx.x): idx of ((dim + 1) / 2) / BLOCK_SIZE
    पूर्णांक i = item_ct1.get_group(1);
    पूर्णांक j = item_ct1.get_local_id(2) + item_ct1.get_group(2) * item_ct1.get_local_range(2);
    दशांश * embed_data = (दशांश *)((अक्षर *)dst +  i*nb1);

    पूर्णांक half = dim / 2;

    जर (dim % 2 != 0 && j == half) {
        embed_data[2 * half] = 0.f;
    }

    जर (j >= half) {
        परत;
    }

    दशांश timestep = timesteps[i];
    दशांश freq = (दशांश)sycl::native::exp(-(sycl::log((दशांश)max_period)) * j / half);
    दशांश arg = timestep * freq;
    embed_data[j] = sycl::cos(arg);
    embed_data[j + half] = sycl::sin(arg);
}

स्थिर शून्य timestep_embedding_f32_sycl(
        स्थिरांक दशांश * x, दशांश * dst, स्थिरांक पूर्णांक ne00, स्थिरांक पूर्णांक nb1,
        स्थिरांक पूर्णांक dim, स्थिरांक पूर्णांक max_period, स्थिरांक queue_ptr& stream) {
    // As the kernel returns when thread.idx is larger than dim/2, the half_ceil does not need to pad
    पूर्णांक half_ceil = dim / 2;
    पूर्णांक अंक_blocks = (half_ceil + SYCL_TIMESTEP_EMBEDDING_BLOCK_SIZE - 1) / SYCL_TIMESTEP_EMBEDDING_BLOCK_SIZE;
    sycl::range<3> block_dims(1, 1, SYCL_TIMESTEP_EMBEDDING_BLOCK_SIZE);
    sycl::range<3> gridDim(1, ne00, अंक_blocks);
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
    स्थिरांक ggml_tensor *  src0   = dst->src[0];
    स्थिरांक दशांश * src0_d = (स्थिरांक दशांश *)src0->data;
    दशांश * dst_d = (दशांश *)dst->data;
    dpct::queue_ptr stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    स्थिरांक पूर्णांक dim = dst->op_params[0];
    स्थिरांक पूर्णांक max_period = dst->op_params[1];

    timestep_embedding_f32_sycl(src0_d, dst_d, src0->ne[0], dst->nb[1], dim, max_period, stream);
}
