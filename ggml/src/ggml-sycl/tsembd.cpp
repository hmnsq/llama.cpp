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

#ضمن "tsembd.hpp"

ساكن فراغ timestep_embedding_f32(
        ثابت عائم * timesteps, عائم * dst, ثابت صحيح nb1,
        ثابت صحيح dim, ثابت صحيح max_period, ثابت sycl::nd_item<3> &item_ct1) {
    // item_ct1.get_group(1)(blockIDx.y): idx of timesteps->ne[0]
    // item_ct1.get_group(2) (blockIDx.x): idx of ((dim + 1) / 2) / BLOCK_SIZE
    صحيح i = item_ct1.get_group(1);
    صحيح j = item_ct1.get_local_id(2) + item_ct1.get_group(2) * item_ct1.get_local_range(2);
    عائم * embed_data = (عائم *)((حرف *)dst +  i*nb1);

    صحيح half = dim / 2;

    אם (dim % 2 != 0 && j == half) {
        embed_data[2 * half] = 0.f;
    }

    אם (j >= half) {
        החזר;
    }

    عائم timestep = timesteps[i];
    عائم freq = (عائم)sycl::native::exp(-(sycl::log((عائم)max_period)) * j / half);
    عائم arg = timestep * freq;
    embed_data[j] = sycl::cos(arg);
    embed_data[j + half] = sycl::sin(arg);
}

ساكن فراغ timestep_embedding_f32_sycl(
        ثابت عائم * x, عائم * dst, ثابت صحيح ne00, ثابت صحيح nb1,
        ثابت صحيح dim, ثابت صحيح max_period, ثابت queue_ptr& stream) {
    // As the kernel returns when thread.idx is larger than dim/2, the half_ceil does not need to pad
    صحيح half_ceil = dim / 2;
    صحيح num_blocks = (half_ceil + SYCL_TIMESTEP_EMBEDDING_BLOCK_SIZE - 1) / SYCL_TIMESTEP_EMBEDDING_BLOCK_SIZE;
    sycl::range<3> block_dims(1, 1, SYCL_TIMESTEP_EMBEDDING_BLOCK_SIZE);
    sycl::range<3> gridDim(1, ne00, num_blocks);
    stream->parallel_for(
        sycl::nd_range<3>(
            gridDim * block_dims, block_dims),
        [=](sycl::nd_item<3> item_ct1) {
            timestep_embedding_f32(
                x, dst, nb1, dim, max_period, item_ct1
            );
        });
}

فراغ ggml_sycl_op_timestep_embedding(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ثابت ggml_tensor *  src0   = dst->src[0];
    ثابت عائم * src0_d = (ثابت عائم *)src0->data;
    عائم * dst_d = (عائم *)dst->data;
    dpct::queue_ptr stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    ثابت صحيح dim = dst->op_params[0];
    ثابت صحيح max_period = dst->op_params[1];

    timestep_embedding_f32_sycl(src0_d, dst_d, src0->ne[0], dst->nb[1], dim, max_period, stream);
}
