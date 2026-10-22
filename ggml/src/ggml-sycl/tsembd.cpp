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

#அந்தர்பாவ "tsembd.hpp"

நிலையான வெறுமை timestep_embedding_f32(
        மாறிலி தசம * timesteps, தசம * dst, மாறிலி முழு nb1,
        மாறிலி முழு dim, மாறிலி முழு max_period, மாறிலி sycl::nd_item<3> &item_ct1) {
    // item_ct1.get_group(1)(blockIDx.y): idx of timesteps->ne[0]
    // item_ct1.get_group(2) (blockIDx.x): idx of ((dim + 1) / 2) / BLOCK_SIZE
    முழு i = item_ct1.get_group(1);
    முழு j = item_ct1.get_local_id(2) + item_ct1.get_group(2) * item_ct1.get_local_range(2);
    தசம * embed_data = (தசம *)((எழுத்து *)dst +  i*nb1);

    முழு half = dim / 2;

    எனில் (dim % 2 != 0 && j == half) {
        embed_data[2 * half] = 0.f;
    }

    எனில் (j >= half) {
        திரும்பு;
    }

    தசம timestep = timesteps[i];
    தசம freq = (தசம)sycl::native::exp(-(sycl::log((தசம)max_period)) * j / half);
    தசம arg = timestep * freq;
    embed_data[j] = sycl::cos(arg);
    embed_data[j + half] = sycl::sin(arg);
}

நிலையான வெறுமை timestep_embedding_f32_sycl(
        மாறிலி தசம * x, தசம * dst, மாறிலி முழு ne00, மாறிலி முழு nb1,
        மாறிலி முழு dim, மாறிலி முழு max_period, மாறிலி queue_ptr& stream) {
    // As the kernel returns when thread.idx is larger than dim/2, the half_ceil does not need to pad
    முழு half_ceil = dim / 2;
    முழு இலக்கம்_blocks = (half_ceil + SYCL_TIMESTEP_EMBEDDING_BLOCK_SIZE - 1) / SYCL_TIMESTEP_EMBEDDING_BLOCK_SIZE;
    sycl::range<3> block_dims(1, 1, SYCL_TIMESTEP_EMBEDDING_BLOCK_SIZE);
    sycl::range<3> gridDim(1, ne00, இலக்கம்_blocks);
    stream->parallel_for(
        sycl::nd_range<3>(
            gridDim * block_dims, block_dims),
        [=](sycl::nd_item<3> item_ct1) {
            timestep_embedding_f32(
                x, dst, nb1, dim, max_period, item_ct1
            );
        });
}

வெறுமை ggml_sycl_op_timestep_embedding(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    மாறிலி ggml_tensor *  src0   = dst->src[0];
    மாறிலி தசம * src0_d = (மாறிலி தசம *)src0->data;
    தசம * dst_d = (தசம *)dst->data;
    dpct::queue_ptr stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    மாறிலி முழு dim = dst->op_params[0];
    மாறிலி முழு max_period = dst->op_params[1];

    timestep_embedding_f32_sycl(src0_d, dst_d, src0->ne[0], dst->nb[1], dim, max_period, stream);
}
