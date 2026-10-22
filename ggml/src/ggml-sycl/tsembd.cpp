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

#અન્તર્ભાવ "tsembd.hpp"

સ્થિર શૂન્ય timestep_embedding_f32(
        અચલ દશાંશ * timesteps, દશાંશ * dst, અચલ પૂર્ણાંક nb1,
        અચલ પૂર્ણાંક dim, અચલ પૂર્ણાંક max_period, અચલ sycl::nd_item<3> &item_ct1) {
    // item_ct1.get_group(1)(blockIDx.y): idx of timesteps->ne[0]
    // item_ct1.get_group(2) (blockIDx.x): idx of ((dim + 1) / 2) / BLOCK_SIZE
    પૂર્ણાંક i = item_ct1.get_group(1);
    પૂર્ણાંક j = item_ct1.get_local_id(2) + item_ct1.get_group(2) * item_ct1.get_local_range(2);
    દશાંશ * embed_data = (દશાંશ *)((અક્ષર *)dst +  i*nb1);

    પૂર્ણાંક half = dim / 2;

    જો (dim % 2 != 0 && j == half) {
        embed_data[2 * half] = 0.f;
    }

    જો (j >= half) {
        પાછા;
    }

    દશાંશ timestep = timesteps[i];
    દશાંશ freq = (દશાંશ)sycl::native::exp(-(sycl::log((દશાંશ)max_period)) * j / half);
    દશાંશ arg = timestep * freq;
    embed_data[j] = sycl::cos(arg);
    embed_data[j + half] = sycl::sin(arg);
}

સ્થિર શૂન્ય timestep_embedding_f32_sycl(
        અચલ દશાંશ * x, દશાંશ * dst, અચલ પૂર્ણાંક ne00, અચલ પૂર્ણાંક nb1,
        અચલ પૂર્ણાંક dim, અચલ પૂર્ણાંક max_period, અચલ queue_ptr& stream) {
    // As the kernel returns when thread.idx is larger than dim/2, the half_ceil does not need to pad
    પૂર્ણાંક half_ceil = dim / 2;
    પૂર્ણાંક અંક_blocks = (half_ceil + SYCL_TIMESTEP_EMBEDDING_BLOCK_SIZE - 1) / SYCL_TIMESTEP_EMBEDDING_BLOCK_SIZE;
    sycl::range<3> block_dims(1, 1, SYCL_TIMESTEP_EMBEDDING_BLOCK_SIZE);
    sycl::range<3> gridDim(1, ne00, અંક_blocks);
    stream->parallel_for(
        sycl::nd_range<3>(
            gridDim * block_dims, block_dims),
        [=](sycl::nd_item<3> item_ct1) {
            timestep_embedding_f32(
                x, dst, nb1, dim, max_period, item_ct1
            );
        });
}

શૂન્ય ggml_sycl_op_timestep_embedding(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    અચલ ggml_tensor *  src0   = dst->src[0];
    અચલ દશાંશ * src0_d = (અચલ દશાંશ *)src0->data;
    દશાંશ * dst_d = (દશાંશ *)dst->data;
    dpct::queue_ptr stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    અચલ પૂર્ણાંક dim = dst->op_params[0];
    અચલ પૂર્ણાંક max_period = dst->op_params[1];

    timestep_embedding_f32_sycl(src0_d, dst_d, src0->ne[0], dst->nb[1], dim, max_period, stream);
}
