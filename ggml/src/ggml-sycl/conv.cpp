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

#అన్తర్భావ "conv.hpp"

స్థిర  శూన్య conv_transpose_1d_kernel(
        స్థిరరాశి పూర్ణాంకం s0, స్థిరరాశి పూర్ణాంకం output_కొలత,
        స్థిరరాశి పూర్ణాంకం src0_ne0, స్థిరరాశి పూర్ణాంకం src0_ne1, స్థిరరాశి పూర్ణాంకం src0_ne2,
        స్థిరరాశి పూర్ణాంకం src1_ne0, స్థిరరాశి పూర్ణాంకం dst_ne0,
        స్థిరరాశి దశాంశం * src0, స్థిరరాశి దశాంశం * src1,  దశాంశం * dst,
        స్థిరరాశి sycl::nd_అంశం<3> &అంశం_ct1) {
    పూర్ణాంకం global_సూచీ = అంశం_ct1.get_local_id(2) +
                       అంశం_ct1.get_సమూహం(2) * అంశం_ct1.get_local_range(2);
    ఒకవేళ (global_సూచీ >= output_కొలత) {
        తిరిగి;
    }

    పూర్ణాంకం బహిర్గతం_సూచీ = global_సూచీ / dst_ne0;

    దశాంశం సంచాయకం = 0;

    కొరకు (పూర్ణాంకం c = 0; c < src0_ne2; c++) {
        పూర్ణాంకం idx = global_సూచీ % dst_ne0;

        పూర్ణాంకం kernel_స్థానభ్రంశం = (src0_ne0 * src0_ne1 * c) + (బహిర్గతం_సూచీ * src0_ne0);
        పూర్ణాంకం ఆదానం_స్థానభ్రంశం = src1_ne0 * c;

        కొరకు (పూర్ణాంకం i = 0; i < src1_ne0; i++) {
            ఒకవేళ (!(idx >= i*s0 && idx < i*s0 + src0_ne0)) {
                కొనసాగు;
            }
            పూర్ణాంకం బరువు_idx = idx - i*s0;

            దశాంశం kernel_బరువు = src0[kernel_స్థానభ్రంశం + బరువు_idx];
            దశాంశం ఆదానం_విలువ =  src1[ఆదానం_స్థానభ్రంశం+i];

            సంచాయకం += kernel_బరువు * ఆదానం_విలువ;
        }
    }
    dst[global_సూచీ] = సంచాయకం;
}

స్థిర శూన్య conv_transpose_1d_f32_f32_sycl(
    స్థిరరాశి పూర్ణాంకం s0, స్థిరరాశి పూర్ణాంకం output_కొలత,
    స్థిరరాశి పూర్ణాంకం src0_ne0, స్థిరరాశి పూర్ణాంకం src0_ne1, స్థిరరాశి పూర్ణాంకం src0_ne2,
    స్థిరరాశి పూర్ణాంకం src1_ne0, స్థిరరాశి పూర్ణాంకం dst_ne0,
    స్థిరరాశి దశాంశం *src0, స్థిరరాశి దశాంశం *src1, దశాంశం *dst,
    స్థిరరాశి వరుస_ptr& stream) {

    స్థిరరాశి పూర్ణాంకం అంకె_ఖండాలు = (output_కొలత + SYCL_CONV_TRANPOSE_1_D_ఖండం_కొలత - 1) / SYCL_CONV_TRANPOSE_1_D_ఖండం_కొలత;
    స్థిరరాశి sycl::range<3> ఖండం_dims(1, 1, SYCL_CONV_TRANPOSE_1_D_ఖండం_కొలత);
    స్థిరరాశి sycl::range<3> ఖండం_nums(1, 1, అంకె_ఖండాలు);
    stream->parallel_for(
        sycl::nd_range<3>(
            ఖండం_nums * ఖండం_dims, ఖండం_dims),
        [=](sycl::nd_అంశం<3> అంశం_ct1) {
            conv_transpose_1d_kernel(
                s0, output_కొలత,
                src0_ne0, src0_ne1, src0_ne2,
                src1_ne0, dst_ne0,
                src0, src1, dst, అంశం_ct1);
        });
}

శూన్య ggml_sycl_op_conv_transpose_1d(ggml_backend_sycl_context & ctx, ggml_tensor *dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    స్థిరరాశి ggml_tensor *src0 = dst->src[0];
    స్థిరరాశి ggml_tensor *src1 = dst->src[1];
    స్థిరరాశి దశాంశం * src0_d = (స్థిరరాశి దశాంశం *)src0->దత్తాంశం;
    స్థిరరాశి దశాంశం * src1_d = (స్థిరరాశి దశాంశం *)src1->దత్తాంశం;

    దశాంశం * dst_d = (దశాంశం *)dst->దత్తాంశం;
    dpct::వరుస_ptr stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(ggml_is_contiguous(src1));

    స్థిరరాశి int32_t * opts = (స్థిరరాశి int32_t *)dst->op_params;

    స్థిరరాశి పూర్ణాంకం s0 = opts[0];

    స్థిరరాశి int64_t output_కొలత = ggml_nelements(dst);

    conv_transpose_1d_f32_f32_sycl(s0, output_కొలత,
        src0->ne[0], src0->ne[1], src0->ne[2],
        src1->ne[0], dst->ne[0],
        src0_d, src1_d, dst_d, stream);
}

