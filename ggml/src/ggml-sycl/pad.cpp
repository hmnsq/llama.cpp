//
// MIT license
// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: MIT
//

//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

//#include "common.hpp"
#అన్తర్భావ "pad.hpp"

స్థిర శూన్య pad_f32(స్థిరరాశి దశాంశం * src, size_t s00, size_t s01, size_t s02, size_t s03,
                    దశాంశం * dst,
                    స్థిరరాశి పూర్ణాంకం lp0, స్థిరరాశి పూర్ణాంకం rp0, స్థిరరాశి పూర్ణాంకం lp1, స్థిరరాశి పూర్ణాంకం rp1,
                    స్థిరరాశి పూర్ణాంకం lp2, స్థిరరాశి పూర్ణాంకం rp2, స్థిరరాశి పూర్ణాంకం lp3, స్థిరరాశి పూర్ణాంకం rp3,
                    స్థిరరాశి పూర్ణాంకం ne0, స్థిరరాశి పూర్ణాంకం ne1, స్థిరరాశి పూర్ణాంకం ne2, స్థిరరాశి పూర్ణాంకం ne3,
                    sycl::nd_అంశం<3> అంశం_ct1) {
    పూర్ణాంకం i0 = అంశం_ct1.get_local_id(2) +
             అంశం_ct1.get_సమూహం(2) * అంశం_ct1.get_local_range(2);
    పూర్ణాంకం i1 = అంశం_ct1.get_సమూహం(1);
    పూర్ణాంకం i2 = అంశం_ct1.get_సమూహం(0) % ne2;
    పూర్ణాంకం i3 = అంశం_ct1.get_సమూహం(0) / ne2;
    ఒకవేళ (i0 >= ne0 || i1 >= ne1 || i2 >= ne2 || i3 >= ne3) {
        తిరిగి;
    }

    స్థిరరాశి int64_t dst_idx = i3*(ne0*ne1*ne2) + i2*(ne0*ne1) + i1*ne0 + i0;
    ఒకవేళ ((i0 >= lp0 && i0 < ne0 - rp0) &&
        (i1 >= lp1 && i1 < ne1 - rp1) &&
        (i2 >= lp2 && i2 < ne2 - rp2) &&
        (i3 >= lp3 && i3 < ne3 - rp3)) {
        స్థిరరాశి int64_t i00 = i0 - lp0;
        స్థిరరాశి int64_t i01 = i1 - lp1;
        స్థిరరాశి int64_t i02 = i2 - lp2;
        స్థిరరాశి int64_t i03 = i3 - lp3;

        స్థిరరాశి int64_t src_idx = i03 * s03 + i02 * s02 + i01 * s01 + i00 * s00;

        dst[dst_idx] = src[src_idx];
    } లేకపోతే {
        dst[dst_idx] = 0.0f;
    }
}

స్థిర శూన్య pad_f32_sycl(స్థిరరాశి దశాంశం * src, size_t s00, size_t s01, size_t s02, size_t s03,
                         దశాంశం * dst, స్థిరరాశి పూర్ణాంకం lp0, స్థిరరాశి పూర్ణాంకం rp0, స్థిరరాశి పూర్ణాంకం lp1, స్థిరరాశి పూర్ణాంకం rp1,
                         స్థిరరాశి పూర్ణాంకం lp2, స్థిరరాశి పూర్ణాంకం rp2, స్థిరరాశి పూర్ణాంకం lp3, స్థిరరాశి పూర్ణాంకం rp3,
                         స్థిరరాశి పూర్ణాంకం ne0, స్థిరరాశి పూర్ణాంకం ne1, స్థిరరాశి పూర్ణాంకం ne2, స్థిరరాశి పూర్ణాంకం ne3,
                         dpct::వరుస_ptr stream) {
    పూర్ణాంకం అంకె_ఖండాలు = (ne0 + SYCL_PAD_ఖండం_కొలత - 1) / SYCL_PAD_ఖండం_కొలత;
    sycl::range<3> grid(ne2 * ne3, ne1, అంకె_ఖండాలు);
    stream->parallel_for(
        sycl::nd_range<3>(grid * sycl::range<3>(1, 1, SYCL_PAD_ఖండం_కొలత),
                          sycl::range<3>(1, 1, SYCL_PAD_ఖండం_కొలత)),
        [=](sycl::nd_అంశం<3> అంశం_ct1) {
            pad_f32(src, s00, s01, s02, s03, dst, lp0, rp0, lp1, rp1, lp2, rp2, lp3, rp3,
                    ne0, ne1, ne2, ne3, అంశం_ct1);
        });
}

శూన్య ggml_sycl_op_pad(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    స్థిరరాశి ggml_tensor * src0 = dst->src[0];
    స్థిరరాశి దశాంశం * src0_d = (స్థిరరాశి దశాంశం *)src0->దత్తాంశం;
    దశాంశం * dst_d = (దశాంశం *)dst->దత్తాంశం;
    dpct::వరుస_ptr stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    స్థిరరాశి size_t ts = ggml_type_కొలత(src0->type);
    స్థిరరాశి size_t s00 = src0->nb[0] / ts;
    స్థిరరాశి size_t s01 = src0->nb[1] / ts;
    స్థిరరాశి size_t s02 = src0->nb[2] / ts;
    స్థిరరాశి size_t s03 = src0->nb[3] / ts;

    స్థిరరాశి int32_t lp0 = ((స్థిరరాశి int32_t *)(dst->op_params))[0];
    స్థిరరాశి int32_t rp0 = ((స్థిరరాశి int32_t *)(dst->op_params))[1];
    స్థిరరాశి int32_t lp1 = ((స్థిరరాశి int32_t *)(dst->op_params))[2];
    స్థిరరాశి int32_t rp1 = ((స్థిరరాశి int32_t *)(dst->op_params))[3];
    స్థిరరాశి int32_t lp2 = ((స్థిరరాశి int32_t *)(dst->op_params))[4];
    స్థిరరాశి int32_t rp2 = ((స్థిరరాశి int32_t *)(dst->op_params))[5];
    స్థిరరాశి int32_t lp3 = ((స్థిరరాశి int32_t *)(dst->op_params))[6];
    స్థిరరాశి int32_t rp3 = ((స్థిరరాశి int32_t *)(dst->op_params))[7];

    pad_f32_sycl(src0_d, s00, s01, s02, s03, dst_d,
                 lp0, rp0, lp1, rp1, lp2, rp2, lp3, rp3,
                 dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3], stream);
}

శూన్య ggml_sycl_pad(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_pad(ctx, dst);
}
