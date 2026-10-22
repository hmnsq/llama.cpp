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

#ਅਨ੍ਤਰ੍ਭਾਵ "tsembd.hpp"

ਸਥਿਰ ਖਾਲੀ timestep_embedding_f32(
        ਅਚਲ ਦਸ਼ਮਲਵ * timesteps, ਦਸ਼ਮਲਵ * dst, ਅਚਲ ਪੂਰਨ nb1,
        ਅਚਲ ਪੂਰਨ dim, ਅਚਲ ਪੂਰਨ max_period, ਅਚਲ sycl::nd_ਵਸਤੂ<3> &ਵਸਤੂ_ct1) {
    // item_ct1.get_group(1)(blockIDx.y): idx of timesteps->ne[0]
    // item_ct1.get_group(2) (blockIDx.x): idx of ((dim + 1) / 2) / BLOCK_SIZE
    ਪੂਰਨ i = ਵਸਤੂ_ct1.get_ਸਮੂਹ(1);
    ਪੂਰਨ j = ਵਸਤੂ_ct1.get_local_id(2) + ਵਸਤੂ_ct1.get_ਸਮੂਹ(2) * ਵਸਤੂ_ct1.get_local_range(2);
    ਦਸ਼ਮਲਵ * embed_ਅੰਕੜੇ = (ਦਸ਼ਮਲਵ *)((ਅੱਖਰ *)dst +  i*nb1);

    ਪੂਰਨ half = dim / 2;

    ਜੇ (dim % 2 != 0 && j == half) {
        embed_ਅੰਕੜੇ[2 * half] = 0.f;
    }

    ਜੇ (j >= half) {
        ਵਾਪਸ;
    }

    ਦਸ਼ਮਲਵ timestep = timesteps[i];
    ਦਸ਼ਮਲਵ freq = (ਦਸ਼ਮਲਵ)sycl::native::ਘਾਤਾਂਕ(-(sycl::ਲਘੂਗਣਕ((ਦਸ਼ਮਲਵ)max_period)) * j / half);
    ਦਸ਼ਮਲਵ arg = timestep * freq;
    embed_ਅੰਕੜੇ[j] = sycl::ਕੋਜੀਵਾ(arg);
    embed_ਅੰਕੜੇ[j + half] = sycl::ਜੀਵਾ(arg);
}

ਸਥਿਰ ਖਾਲੀ timestep_embedding_f32_sycl(
        ਅਚਲ ਦਸ਼ਮਲਵ * x, ਦਸ਼ਮਲਵ * dst, ਅਚਲ ਪੂਰਨ ne00, ਅਚਲ ਪੂਰਨ nb1,
        ਅਚਲ ਪੂਰਨ dim, ਅਚਲ ਪੂਰਨ max_period, ਅਚਲ ਕਤਾਰ_ptr& stream) {
    // As the kernel returns when thread.idx is larger than dim/2, the half_ceil does not need to pad
    ਪੂਰਨ half_ceil = dim / 2;
    ਪੂਰਨ ਅੰਕ_ਖੰਡਾਂ = (half_ceil + SYCL_TIMESTEP_EMBEDDING_ਖੰਡ_ਮਾਪ - 1) / SYCL_TIMESTEP_EMBEDDING_ਖੰਡ_ਮਾਪ;
    sycl::range<3> ਖੰਡ_dims(1, 1, SYCL_TIMESTEP_EMBEDDING_ਖੰਡ_ਮਾਪ);
    sycl::range<3> gridDim(1, ne00, ਅੰਕ_ਖੰਡਾਂ);
    stream->parallel_for(
        sycl::nd_range<3>(
            gridDim * ਖੰਡ_dims, ਖੰਡ_dims),
        [=](sycl::nd_ਵਸਤੂ<3> ਵਸਤੂ_ct1) {
            timestep_embedding_f32(
                x, dst, nb1, dim, max_period, ਵਸਤੂ_ct1
            );
        });
}

ਖਾਲੀ ggml_sycl_op_timestep_embedding(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ਅਚਲ ggml_tensor *  src0   = dst->src[0];
    ਅਚਲ ਦਸ਼ਮਲਵ * src0_d = (ਅਚਲ ਦਸ਼ਮਲਵ *)src0->ਅੰਕੜੇ;
    ਦਸ਼ਮਲਵ * dst_d = (ਦਸ਼ਮਲਵ *)dst->ਅੰਕੜੇ;
    dpct::ਕਤਾਰ_ptr stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    ਅਚਲ ਪੂਰਨ dim = dst->op_params[0];
    ਅਚਲ ਪੂਰਨ max_period = dst->op_params[1];

    timestep_embedding_f32_sycl(src0_d, dst_d, src0->ne[0], dst->nb[1], dim, max_period, stream);
}
