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

#অন্তর্ভাব "tsembd.hpp"

স্থির শূন্য timestep_embedding_f32(
        ধ্রুব ভিন্ন * timesteps, ভিন্ন * dst, ধ্রুব পূর্ণাঙ্ক nb1,
        ধ্রুব পূর্ণাঙ্ক dim, ধ্রুব পূর্ণাঙ্ক max_period, ধ্রুব sycl::nd_item<3> &item_ct1) {
    // item_ct1.get_group(1)(blockIDx.y): idx of timesteps->ne[0]
    // item_ct1.get_group(2) (blockIDx.x): idx of ((dim + 1) / 2) / BLOCK_SIZE
    পূর্ণাঙ্ক i = item_ct1.get_group(1);
    পূর্ণাঙ্ক j = item_ct1.get_local_id(2) + item_ct1.get_group(2) * item_ct1.get_local_range(2);
    ভিন্ন * embed_data = (ভিন্ন *)((বর্ণ *)dst +  i*nb1);

    পূর্ণাঙ্ক half = dim / 2;

    যদি (dim % 2 != 0 && j == half) {
        embed_data[2 * half] = 0.f;
    }

    যদি (j >= half) {
        প্রত্যাগম;
    }

    ভিন্ন timestep = timesteps[i];
    ভিন্ন freq = (ভিন্ন)sycl::native::exp(-(sycl::log((ভিন্ন)max_period)) * j / half);
    ভিন্ন arg = timestep * freq;
    embed_data[j] = sycl::cos(arg);
    embed_data[j + half] = sycl::sin(arg);
}

স্থির শূন্য timestep_embedding_f32_sycl(
        ধ্রুব ভিন্ন * x, ভিন্ন * dst, ধ্রুব পূর্ণাঙ্ক ne00, ধ্রুব পূর্ণাঙ্ক nb1,
        ধ্রুব পূর্ণাঙ্ক dim, ধ্রুব পূর্ণাঙ্ক max_period, ধ্রুব queue_ptr& stream) {
    // As the kernel returns when thread.idx is larger than dim/2, the half_ceil does not need to pad
    পূর্ণাঙ্ক half_ceil = dim / 2;
    পূর্ণাঙ্ক অঙ্ক_blocks = (half_ceil + SYCL_TIMESTEP_EMBEDDING_BLOCK_SIZE - 1) / SYCL_TIMESTEP_EMBEDDING_BLOCK_SIZE;
    sycl::range<3> block_dims(1, 1, SYCL_TIMESTEP_EMBEDDING_BLOCK_SIZE);
    sycl::range<3> gridDim(1, ne00, অঙ্ক_blocks);
    stream->parallel_for(
        sycl::nd_range<3>(
            gridDim * block_dims, block_dims),
        [=](sycl::nd_item<3> item_ct1) {
            timestep_embedding_f32(
                x, dst, nb1, dim, max_period, item_ct1
            );
        });
}

শূন্য ggml_sycl_op_timestep_embedding(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ধ্রুব ggml_tensor *  src0   = dst->src[0];
    ধ্রুব ভিন্ন * src0_d = (ধ্রুব ভিন্ন *)src0->data;
    ভিন্ন * dst_d = (ভিন্ন *)dst->data;
    dpct::queue_ptr stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    ধ্রুব পূর্ণাঙ্ক dim = dst->op_params[0];
    ধ্রুব পূর্ণাঙ্ক max_period = dst->op_params[1];

    timestep_embedding_f32_sycl(src0_d, dst_d, src0->ne[0], dst->nb[1], dim, max_period, stream);
}
