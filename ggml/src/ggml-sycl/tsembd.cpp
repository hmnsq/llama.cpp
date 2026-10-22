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

#అన్తర్భావ "tsembd.hpp"

స్థిర శూన్య timestep_embedding_f32(
        స్థిరరాశి దశాంశం * timesteps, దశాంశం * dst, స్థిరరాశి పూర్ణాంకం nb1,
        స్థిరరాశి పూర్ణాంకం dim, స్థిరరాశి పూర్ణాంకం max_period, స్థిరరాశి sycl::nd_అంశం<3> &అంశం_ct1) {
    // item_ct1.get_group(1)(blockIDx.y): idx of timesteps->ne[0]
    // item_ct1.get_group(2) (blockIDx.x): idx of ((dim + 1) / 2) / BLOCK_SIZE
    పూర్ణాంకం i = అంశం_ct1.get_సమూహం(1);
    పూర్ణాంకం j = అంశం_ct1.get_local_id(2) + అంశం_ct1.get_సమూహం(2) * అంశం_ct1.get_local_range(2);
    దశాంశం * embed_దత్తాంశం = (దశాంశం *)((అక్షరం *)dst +  i*nb1);

    పూర్ణాంకం half = dim / 2;

    ఒకవేళ (dim % 2 != 0 && j == half) {
        embed_దత్తాంశం[2 * half] = 0.f;
    }

    ఒకవేళ (j >= half) {
        తిరిగి;
    }

    దశాంశం timestep = timesteps[i];
    దశాంశం freq = (దశాంశం)sycl::native::ఘాతాంకం(-(sycl::సంవర్గమానం((దశాంశం)max_period)) * j / half);
    దశాంశం arg = timestep * freq;
    embed_దత్తాంశం[j] = sycl::కోజ్యా(arg);
    embed_దత్తాంశం[j + half] = sycl::జ్యా(arg);
}

స్థిర శూన్య timestep_embedding_f32_sycl(
        స్థిరరాశి దశాంశం * x, దశాంశం * dst, స్థిరరాశి పూర్ణాంకం ne00, స్థిరరాశి పూర్ణాంకం nb1,
        స్థిరరాశి పూర్ణాంకం dim, స్థిరరాశి పూర్ణాంకం max_period, స్థిరరాశి వరుస_ptr& stream) {
    // As the kernel returns when thread.idx is larger than dim/2, the half_ceil does not need to pad
    పూర్ణాంకం half_ceil = dim / 2;
    పూర్ణాంకం అంకె_ఖండాలు = (half_ceil + SYCL_TIMESTEP_EMBEDDING_ఖండం_కొలత - 1) / SYCL_TIMESTEP_EMBEDDING_ఖండం_కొలత;
    sycl::range<3> ఖండం_dims(1, 1, SYCL_TIMESTEP_EMBEDDING_ఖండం_కొలత);
    sycl::range<3> gridDim(1, ne00, అంకె_ఖండాలు);
    stream->parallel_for(
        sycl::nd_range<3>(
            gridDim * ఖండం_dims, ఖండం_dims),
        [=](sycl::nd_అంశం<3> అంశం_ct1) {
            timestep_embedding_f32(
                x, dst, nb1, dim, max_period, అంశం_ct1
            );
        });
}

శూన్య ggml_sycl_op_timestep_embedding(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    స్థిరరాశి ggml_tensor *  src0   = dst->src[0];
    స్థిరరాశి దశాంశం * src0_d = (స్థిరరాశి దశాంశం *)src0->దత్తాంశం;
    దశాంశం * dst_d = (దశాంశం *)dst->దత్తాంశం;
    dpct::వరుస_ptr stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    స్థిరరాశి పూర్ణాంకం dim = dst->op_params[0];
    స్థిరరాశి పూర్ణాంకం max_period = dst->op_params[1];

    timestep_embedding_f32_sycl(src0_d, dst_d, src0->ne[0], dst->nb[1], dim, max_period, stream);
}
