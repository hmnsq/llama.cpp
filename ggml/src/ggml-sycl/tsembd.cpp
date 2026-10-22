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

#അന്തര്ഭാവ "tsembd.hpp"

സ്ഥിതം ശൂന്യം timestep_embedding_f32(
        സ്ഥിരം ദശാംശം * timesteps, ദശാംശം * dst, സ്ഥിരം പൂർണ്ണസംഖ്യ nb1,
        സ്ഥിരം പൂർണ്ണസംഖ്യ dim, സ്ഥിരം പൂർണ്ണസംഖ്യ max_period, സ്ഥിരം sycl::nd_item<3> &item_ct1) {
    // item_ct1.get_group(1)(blockIDx.y): idx of timesteps->ne[0]
    // item_ct1.get_group(2) (blockIDx.x): idx of ((dim + 1) / 2) / BLOCK_SIZE
    പൂർണ്ണസംഖ്യ i = item_ct1.get_group(1);
    പൂർണ്ണസംഖ്യ j = item_ct1.get_local_id(2) + item_ct1.get_group(2) * item_ct1.get_local_range(2);
    ദശാംശം * embed_data = (ദശാംശം *)((അക്ഷരം *)dst +  i*nb1);

    പൂർണ്ണസംഖ്യ half = dim / 2;

    എങ്കിൽ (dim % 2 != 0 && j == half) {
        embed_data[2 * half] = 0.f;
    }

    എങ്കിൽ (j >= half) {
        തിരികെ;
    }

    ദശാംശം timestep = timesteps[i];
    ദശാംശം freq = (ദശാംശം)sycl::native::exp(-(sycl::log((ദശാംശം)max_period)) * j / half);
    ദശാംശം arg = timestep * freq;
    embed_data[j] = sycl::cos(arg);
    embed_data[j + half] = sycl::sin(arg);
}

സ്ഥിതം ശൂന്യം timestep_embedding_f32_sycl(
        സ്ഥിരം ദശാംശം * x, ദശാംശം * dst, സ്ഥിരം പൂർണ്ണസംഖ്യ ne00, സ്ഥിരം പൂർണ്ണസംഖ്യ nb1,
        സ്ഥിരം പൂർണ്ണസംഖ്യ dim, സ്ഥിരം പൂർണ്ണസംഖ്യ max_period, സ്ഥിരം queue_ptr& stream) {
    // As the kernel returns when thread.idx is larger than dim/2, the half_ceil does not need to pad
    പൂർണ്ണസംഖ്യ half_ceil = dim / 2;
    പൂർണ്ണസംഖ്യ അക്കം_blocks = (half_ceil + SYCL_TIMESTEP_EMBEDDING_BLOCK_SIZE - 1) / SYCL_TIMESTEP_EMBEDDING_BLOCK_SIZE;
    sycl::range<3> block_dims(1, 1, SYCL_TIMESTEP_EMBEDDING_BLOCK_SIZE);
    sycl::range<3> gridDim(1, ne00, അക്കം_blocks);
    stream->parallel_for(
        sycl::nd_range<3>(
            gridDim * block_dims, block_dims),
        [=](sycl::nd_item<3> item_ct1) {
            timestep_embedding_f32(
                x, dst, nb1, dim, max_period, item_ct1
            );
        });
}

ശൂന്യം ggml_sycl_op_timestep_embedding(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    സ്ഥിരം ggml_tensor *  src0   = dst->src[0];
    സ്ഥിരം ദശാംശം * src0_d = (സ്ഥിരം ദശാംശം *)src0->data;
    ദശാംശം * dst_d = (ദശാംശം *)dst->data;
    dpct::queue_ptr stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    സ്ഥിരം പൂർണ്ണസംഖ്യ dim = dst->op_params[0];
    സ്ഥിരം പൂർണ്ണസംഖ്യ max_period = dst->op_params[1];

    timestep_embedding_f32_sycl(src0_d, dst_d, src0->ne[0], dst->nb[1], dim, max_period, stream);
}
