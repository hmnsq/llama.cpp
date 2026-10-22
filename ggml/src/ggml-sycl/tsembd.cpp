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

#ಅನ್ತರ್ಭಾವ "tsembd.hpp"

ಸ್ಥಿರ ಶೂನ್ಯ timestep_embedding_f32(
        ಧ್ರುವ ಭಿನ್ನ * timesteps, ಭಿನ್ನ * dst, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb1,
        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ dim, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ max_period, ಧ್ರುವ sycl::nd_item<3> &item_ct1) {
    // item_ct1.get_group(1)(blockIDx.y): idx of timesteps->ne[0]
    // item_ct1.get_group(2) (blockIDx.x): idx of ((dim + 1) / 2) / BLOCK_SIZE
    ಪೂರ್ಣಾಙ್ಕ i = item_ct1.get_group(1);
    ಪೂರ್ಣಾಙ್ಕ j = item_ct1.get_local_id(2) + item_ct1.get_group(2) * item_ct1.get_local_range(2);
    ಭಿನ್ನ * embed_data = (ಭಿನ್ನ *)((ವರ್ಣ *)dst +  i*nb1);

    ಪೂರ್ಣಾಙ್ಕ half = dim / 2;

    ಯದಿ (dim % 2 != 0 && j == half) {
        embed_data[2 * half] = 0.f;
    }

    ಯದಿ (j >= half) {
        ಪ್ರತ್ಯಾಗಮ;
    }

    ಭಿನ್ನ timestep = timesteps[i];
    ಭಿನ್ನ freq = (ಭಿನ್ನ)sycl::native::exp(-(sycl::log((ಭಿನ್ನ)max_period)) * j / half);
    ಭಿನ್ನ arg = timestep * freq;
    embed_data[j] = sycl::cos(arg);
    embed_data[j + half] = sycl::sin(arg);
}

ಸ್ಥಿರ ಶೂನ್ಯ timestep_embedding_f32_sycl(
        ಧ್ರುವ ಭಿನ್ನ * x, ಭಿನ್ನ * dst, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne00, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb1,
        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ dim, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ max_period, ಧ್ರುವ queue_ptr& stream) {
    // As the kernel returns when thread.idx is larger than dim/2, the half_ceil does not need to pad
    ಪೂರ್ಣಾಙ್ಕ half_ceil = dim / 2;
    ಪೂರ್ಣಾಙ್ಕ ಅಂಕೆ_blocks = (half_ceil + SYCL_TIMESTEP_EMBEDDING_BLOCK_SIZE - 1) / SYCL_TIMESTEP_EMBEDDING_BLOCK_SIZE;
    sycl::range<3> block_dims(1, 1, SYCL_TIMESTEP_EMBEDDING_BLOCK_SIZE);
    sycl::range<3> gridDim(1, ne00, ಅಂಕೆ_blocks);
    stream->parallel_for(
        sycl::nd_range<3>(
            gridDim * block_dims, block_dims),
        [=](sycl::nd_item<3> item_ct1) {
            timestep_embedding_f32(
                x, dst, nb1, dim, max_period, item_ct1
            );
        });
}

ಶೂನ್ಯ ggml_sycl_op_timestep_embedding(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ಧ್ರುವ ggml_tensor *  src0   = dst->src[0];
    ಧ್ರುವ ಭಿನ್ನ * src0_d = (ಧ್ರುವ ಭಿನ್ನ *)src0->data;
    ಭಿನ್ನ * dst_d = (ಭಿನ್ನ *)dst->data;
    dpct::queue_ptr stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ dim = dst->op_params[0];
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ max_period = dst->op_params[1];

    timestep_embedding_f32_sycl(src0_d, dst_d, src0->ne[0], dst->nb[1], dim, max_period, stream);
}
