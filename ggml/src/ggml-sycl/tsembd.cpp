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
        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * timesteps, ದಶಮಾಂಶ * dst, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nb1,
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ dim, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ max_period, ಸ್ಥಿರಾಂಕ sycl::nd_ಅಂಶ<3> &ಅಂಶ_ct1) {
    // item_ct1.get_group(1)(blockIDx.y): idx of timesteps->ne[0]
    // item_ct1.get_group(2) (blockIDx.x): idx of ((dim + 1) / 2) / BLOCK_SIZE
    ಪೂರ್ಣಾಂಕ i = ಅಂಶ_ct1.get_ಗುಂಪು(1);
    ಪೂರ್ಣಾಂಕ j = ಅಂಶ_ct1.get_local_id(2) + ಅಂಶ_ct1.get_ಗುಂಪು(2) * ಅಂಶ_ct1.get_local_range(2);
    ದಶಮಾಂಶ * embed_ದತ್ತಾಂಶ = (ದಶಮಾಂಶ *)((ಅಕ್ಷರ *)dst +  i*nb1);

    ಪೂರ್ಣಾಂಕ half = dim / 2;

    ಆದರೆ (dim % 2 != 0 && j == half) {
        embed_ದತ್ತಾಂಶ[2 * half] = 0.f;
    }

    ಆದರೆ (j >= half) {
        ಹಿಂತಿರುಗಿ;
    }

    ದಶಮಾಂಶ timestep = timesteps[i];
    ದಶಮಾಂಶ freq = (ದಶಮಾಂಶ)sycl::native::ಘಾತಾಂಕ(-(sycl::ಲಘುಗಣಕ((ದಶಮಾಂಶ)max_period)) * j / half);
    ದಶಮಾಂಶ arg = timestep * freq;
    embed_ದತ್ತಾಂಶ[j] = sycl::ಕೋಜ್ಯಾ(arg);
    embed_ದತ್ತಾಂಶ[j + half] = sycl::ಜ್ಯಾ(arg);
}

ಸ್ಥಿರ ಶೂನ್ಯ timestep_embedding_f32_sycl(
        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * x, ದಶಮಾಂಶ * dst, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ne00, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nb1,
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ dim, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ max_period, ಸ್ಥಿರಾಂಕ ಸರತಿ_ptr& stream) {
    // As the kernel returns when thread.idx is larger than dim/2, the half_ceil does not need to pad
    ಪೂರ್ಣಾಂಕ half_ceil = dim / 2;
    ಪೂರ್ಣಾಂಕ ಅಂಕೆ_ಖಂಡಗಳು = (half_ceil + SYCL_TIMESTEP_EMBEDDING_ಖಂಡ_ಗಾತ್ರ - 1) / SYCL_TIMESTEP_EMBEDDING_ಖಂಡ_ಗಾತ್ರ;
    sycl::range<3> ಖಂಡ_dims(1, 1, SYCL_TIMESTEP_EMBEDDING_ಖಂಡ_ಗಾತ್ರ);
    sycl::range<3> gridDim(1, ne00, ಅಂಕೆ_ಖಂಡಗಳು);
    stream->parallel_for(
        sycl::nd_range<3>(
            gridDim * ಖಂಡ_dims, ಖಂಡ_dims),
        [=](sycl::nd_ಅಂಶ<3> ಅಂಶ_ct1) {
            timestep_embedding_f32(
                x, dst, nb1, dim, max_period, ಅಂಶ_ct1
            );
        });
}

ಶೂನ್ಯ ggml_sycl_op_timestep_embedding(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ಸ್ಥಿರಾಂಕ ggml_tensor *  src0   = dst->src[0];
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * src0_d = (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *)src0->ದತ್ತಾಂಶ;
    ದಶಮಾಂಶ * dst_d = (ದಶಮಾಂಶ *)dst->ದತ್ತಾಂಶ;
    dpct::ಸರತಿ_ptr stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ dim = dst->op_params[0];
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ max_period = dst->op_params[1];

    timestep_embedding_f32_sycl(src0_d, dst_d, src0->ne[0], dst->nb[1], dim, max_period, stream);
}
