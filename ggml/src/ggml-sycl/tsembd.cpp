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
        ধ্রুবক দশমিক * timesteps, দশমিক * dst, ধ্রুবক পূর্ণসংখ্যা nb1,
        ধ্রুবক পূর্ণসংখ্যা dim, ধ্রুবক পূর্ণসংখ্যা max_period, ধ্রুবক sycl::nd_item<3> &item_ct1) {
    // item_ct1.get_group(1)(blockIDx.y): idx of timesteps->ne[0]
    // item_ct1.get_group(2) (blockIDx.x): idx of ((dim + 1) / 2) / BLOCK_SIZE
    পূর্ণসংখ্যা i = item_ct1.get_group(1);
    পূর্ণসংখ্যা j = item_ct1.get_local_id(2) + item_ct1.get_group(2) * item_ct1.get_local_range(2);
    দশমিক * embed_তথ্য = (দশমিক *)((অক্ষর *)dst +  i*nb1);

    পূর্ণসংখ্যা half = dim / 2;

    যদি (dim % 2 != 0 && j == half) {
        embed_তথ্য[2 * half] = 0.f;
    }

    যদি (j >= half) {
        ফেরত;
    }

    দশমিক timestep = timesteps[i];
    দশমিক freq = (দশমিক)sycl::native::exp(-(sycl::log((দশমিক)max_period)) * j / half);
    দশমিক arg = timestep * freq;
    embed_তথ্য[j] = sycl::cos(arg);
    embed_তথ্য[j + half] = sycl::sin(arg);
}

স্থির শূন্য timestep_embedding_f32_sycl(
        ধ্রুবক দশমিক * x, দশমিক * dst, ধ্রুবক পূর্ণসংখ্যা ne00, ধ্রুবক পূর্ণসংখ্যা nb1,
        ধ্রুবক পূর্ণসংখ্যা dim, ধ্রুবক পূর্ণসংখ্যা max_period, ধ্রুবক queue_ptr& stream) {
    // As the kernel returns when thread.idx is larger than dim/2, the half_ceil does not need to pad
    পূর্ণসংখ্যা half_ceil = dim / 2;
    পূর্ণসংখ্যা অঙ্ক_blocks = (half_ceil + SYCL_TIMESTEP_EMBEDDING_BLOCK_আকার - 1) / SYCL_TIMESTEP_EMBEDDING_BLOCK_আকার;
    sycl::range<3> block_dims(1, 1, SYCL_TIMESTEP_EMBEDDING_BLOCK_আকার);
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
    ধ্রুবক ggml_tensor *  src0   = dst->src[0];
    ধ্রুবক দশমিক * src0_d = (ধ্রুবক দশমিক *)src0->তথ্য;
    দশমিক * dst_d = (দশমিক *)dst->তথ্য;
    dpct::queue_ptr stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    ধ্রুবক পূর্ণসংখ্যা dim = dst->op_params[0];
    ধ্রুবক পূর্ণসংখ্যা max_period = dst->op_params[1];

    timestep_embedding_f32_sycl(src0_d, dst_d, src0->ne[0], dst->nb[1], dim, max_period, stream);
}
