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
        ثابت صحيح dim, ثابت صحيح max_period, ثابت sycl::nd_شي<3> &شي_ct1) {
    // item_ct1.get_group(1)(blockIDx.y): idx of timesteps->ne[0]
    // item_ct1.get_group(2) (blockIDx.x): idx of ((dim + 1) / 2) / BLOCK_SIZE
    صحيح i = شي_ct1.get_گروهه(1);
    صحيح j = شي_ct1.get_local_id(2) + شي_ct1.get_گروهه(2) * شي_ct1.get_local_range(2);
    عائم * embed_معلومات = (عائم *)((حرف *)dst +  i*nb1);

    صحيح half = dim / 2;

    إذا (dim % 2 != 0 && j == half) {
        embed_معلومات[2 * half] = 0.f;
    }

    إذا (j >= half) {
        أرجع;
    }

    عائم timestep = timesteps[i];
    عائم freq = (عائم)sycl::native::قوت_نما(-(sycl::لاگرٿم((عائم)max_period)) * j / half);
    عائم arg = timestep * freq;
    embed_معلومات[j] = sycl::جيب_تمام(arg);
    embed_معلومات[j + half] = sycl::جيب(arg);
}

ساكن فراغ timestep_embedding_f32_sycl(
        ثابت عائم * x, عائم * dst, ثابت صحيح ne00, ثابت صحيح nb1,
        ثابت صحيح dim, ثابت صحيح max_period, ثابت قطار_ptr& stream) {
    // As the kernel returns when thread.idx is larger than dim/2, the half_ceil does not need to pad
    صحيح half_ceil = dim / 2;
    صحيح انگ_بلاڪون = (half_ceil + SYCL_TIMESTEP_EMBEDDING_بلاڪ_سائيز - 1) / SYCL_TIMESTEP_EMBEDDING_بلاڪ_سائيز;
    sycl::range<3> بلاڪ_dims(1, 1, SYCL_TIMESTEP_EMBEDDING_بلاڪ_سائيز);
    sycl::range<3> gridDim(1, ne00, انگ_بلاڪون);
    stream->parallel_for(
        sycl::nd_range<3>(
            gridDim * بلاڪ_dims, بلاڪ_dims),
        [=](sycl::nd_شي<3> شي_ct1) {
            timestep_embedding_f32(
                x, dst, nb1, dim, max_period, شي_ct1
            );
        });
}

فراغ ggml_sycl_op_timestep_embedding(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ثابت ggml_tensor *  src0   = dst->src[0];
    ثابت عائم * src0_d = (ثابت عائم *)src0->معلومات;
    عائم * dst_d = (عائم *)dst->معلومات;
    dpct::قطار_ptr stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    ثابت صحيح dim = dst->op_params[0];
    ثابت صحيح max_period = dst->op_params[1];

    timestep_embedding_f32_sycl(src0_d, dst_d, src0->ne[0], dst->nb[1], dim, max_period, stream);
}
