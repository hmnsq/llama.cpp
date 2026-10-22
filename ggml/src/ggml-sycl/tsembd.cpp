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

#شامل "tsembd.hpp"

ایستا تهی timestep_embedding_f32(
        ثابت اعشاری * timesteps, اعشاری * dst, ثابت صحیح nb1,
        ثابت صحیح dim, ثابت صحیح max_period, ثابت sycl::nd_item<3> &item_ct1) {
    // item_ct1.get_group(1)(blockIDx.y): idx of timesteps->ne[0]
    // item_ct1.get_group(2) (blockIDx.x): idx of ((dim + 1) / 2) / BLOCK_SIZE
    صحیح i = item_ct1.get_group(1);
    صحیح j = item_ct1.get_local_id(2) + item_ct1.get_group(2) * item_ct1.get_local_range(2);
    اعشاری * embed_data = (اعشاری *)((نویسه *)dst +  i*nb1);

    صحیح half = dim / 2;

    اگر (dim % 2 != 0 && j == half) {
        embed_data[2 * half] = 0.f;
    }

    اگر (j >= half) {
        بازگشت;
    }

    اعشاری timestep = timesteps[i];
    اعشاری freq = (اعشاری)sycl::native::exp(-(sycl::log((اعشاری)max_period)) * j / half);
    اعشاری arg = timestep * freq;
    embed_data[j] = sycl::cos(arg);
    embed_data[j + half] = sycl::sin(arg);
}

ایستا تهی timestep_embedding_f32_sycl(
        ثابت اعشاری * x, اعشاری * dst, ثابت صحیح ne00, ثابت صحیح nb1,
        ثابت صحیح dim, ثابت صحیح max_period, ثابت queue_ptr& stream) {
    // As the kernel returns when thread.idx is larger than dim/2, the half_ceil does not need to pad
    صحیح half_ceil = dim / 2;
    صحیح رقم_blocks = (half_ceil + SYCL_TIMESTEP_EMBEDDING_BLOCK_SIZE - 1) / SYCL_TIMESTEP_EMBEDDING_BLOCK_SIZE;
    sycl::range<3> block_dims(1, 1, SYCL_TIMESTEP_EMBEDDING_BLOCK_SIZE);
    sycl::range<3> gridDim(1, ne00, رقم_blocks);
    stream->parallel_for(
        sycl::nd_range<3>(
            gridDim * block_dims, block_dims),
        [=](sycl::nd_item<3> item_ct1) {
            timestep_embedding_f32(
                x, dst, nb1, dim, max_period, item_ct1
            );
        });
}

تهی ggml_sycl_op_timestep_embedding(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ثابت ggml_tensor *  src0   = dst->src[0];
    ثابت اعشاری * src0_d = (ثابت اعشاری *)src0->data;
    اعشاری * dst_d = (اعشاری *)dst->data;
    dpct::queue_ptr stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    ثابت صحیح dim = dst->op_params[0];
    ثابت صحیح max_period = dst->op_params[1];

    timestep_embedding_f32_sycl(src0_d, dst_d, src0->ne[0], dst->nb[1], dim, max_period, stream);
}
