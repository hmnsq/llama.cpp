//
// MIT license
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: MIT
//

//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

#అన్తర్భావ "pool.hpp"
#అన్తర్భావ <float.h>

మూస <రకం_పేరు Ti, రకం_పేరు గమ్యం_2>
స్థిర శూన్య pool2d_nchw_kernel(
        స్థిరరాశి పూర్ణాంకం ih, స్థిరరాశి పూర్ణాంకం iw, స్థిరరాశి పూర్ణాంకం oh, స్థిరరాశి పూర్ణాంకం ow,
        స్థిరరాశి పూర్ణాంకం kh, స్థిరరాశి పూర్ణాంకం kw, స్థిరరాశి పూర్ణాంకం sh, స్థిరరాశి పూర్ణాంకం sw,
        స్థిరరాశి పూర్ణాంకం ph, స్థిరరాశి పూర్ణాంకం pw, స్థిరరాశి పూర్ణాంకం parallel_elements,
        స్థిరరాశి Ti* src, గమ్యం_2* dst, స్థిరరాశి గణన ggml_op_pool op,
        స్థిరరాశి sycl::nd_అంశం<3> &అంశం_ct1) {
        పూర్ణాంకం idx = అంశం_ct1.get_local_id(2) +
                  అంశం_ct1.get_సమూహం(2) * అంశం_ct1.get_local_range(2);
        ఒకవేళ (idx >= parallel_elements) {
            తిరిగి;
        }

        స్థిరరాశి పూర్ణాంకం I_HW = ih * iw;
        స్థిరరాశి పూర్ణాంకం O_HW = oh * ow;
        స్థిరరాశి పూర్ణాంకం nc = idx / O_HW;
        స్థిరరాశి పూర్ణాంకం cur_oh = idx % O_HW / ow;
        స్థిరరాశి పూర్ణాంకం cur_ow = idx % O_HW % ow;
        స్థిరరాశి Ti* i_ptr = src + nc * I_HW;
        గమ్యం_2* o_ptr = dst + nc * O_HW;
        స్థిరరాశి పూర్ణాంకం ప్రారంభం_h = cur_oh * sh - ph;
        స్థిరరాశి పూర్ణాంకం bh = sycl::max(0, ప్రారంభం_h);
        స్థిరరాశి పూర్ణాంకం eh = sycl::min(ih, ప్రారంభం_h + kh);
        స్థిరరాశి పూర్ణాంకం ప్రారంభం_w = cur_ow * sw - pw;
        స్థిరరాశి పూర్ణాంకం bw = sycl::max(0, ప్రారంభం_w);
        స్థిరరాశి పూర్ణాంకం ew = sycl::min(iw, ప్రారంభం_w + kw);

        గమ్యం_2 res = 0;

        ఎంపిక (op) {
            స్థితి GGML_OP_POOL_AVG: res = 0; ఆపు;
            స్థితి GGML_OP_POOL_MAX: res = -FLT_MAX; ఆపు;
            అప్రమేయ:
                res      = (గమ్యం_2) sycl::nan(uint32_t(0));
                ఆపు;
        }

        కొరకు (పూర్ణాంకం i = bh; i < eh; i += 1) {
            కొరకు (పూర్ణాంకం j = bw; j < ew; j += 1) {
                Ti cur = i_ptr[i * iw + j];
                ఎంపిక (op) {
                    స్థితి GGML_OP_POOL_AVG: res += (cur / (kh * kw)); ఆపు;
                    స్థితి GGML_OP_POOL_MAX: res = sycl::max(res, (గమ్యం_2)cur); ఆపు;
                    అప్రమేయ:
                        res = (గమ్యం_2) sycl::nan(uint32_t(0));
                        ఆపు;
                }
            }
        }
        o_ptr[cur_oh * ow + cur_ow] = res;
}

మూస <రకం_పేరు Ti, రకం_పేరు గమ్యం_2>
స్థిర శూన్య pool1d_ncw_kernel(
        స్థిరరాశి పూర్ణాంకం iw, స్థిరరాశి పూర్ణాంకం ow,
        స్థిరరాశి పూర్ణాంకం k, స్థిరరాశి పూర్ణాంకం s,
        స్థిరరాశి పూర్ణాంకం p, స్థిరరాశి పూర్ణాంకం parallel_elements,
        స్థిరరాశి Ti * src, గమ్యం_2 * dst, స్థిరరాశి గణన ggml_op_pool op,
        స్థిరరాశి sycl::nd_అంశం<3> & అంశం_ct1) {
    పూర్ణాంకం idx = అంశం_ct1.get_local_id(2) +
              అంశం_ct1.get_సమూహం(2) * అంశం_ct1.get_local_range(2);
    ఒకవేళ (idx >= parallel_elements) {
        తిరిగి;
    }

    స్థిరరాశి పూర్ణాంకం nc     = idx / ow;
    స్థిరరాశి పూర్ణాంకం cur_ow = idx % ow;
    స్థిరరాశి Ti * i_ptr = src + nc * iw;
    గమ్యం_2 *       o_ptr = dst + nc * ow;
    స్థిరరాశి పూర్ణాంకం  start = cur_ow * s - p;
    స్థిరరాశి పూర్ణాంకం  b     = sycl::max(0, start);
    స్థిరరాశి పూర్ణాంకం  e     = sycl::min(iw, start + k);

    గమ్యం_2 res = 0;
    ఎంపిక (op) {
        స్థితి GGML_OP_POOL_AVG: res = 0;        ఆపు;
        స్థితి GGML_OP_POOL_MAX: res = -FLT_MAX; ఆపు;
        అప్రమేయ:
            res = (గమ్యం_2) sycl::nan(uint32_t(0));
            ఆపు;
    }

    కొరకు (పూర్ణాంకం j = b; j < e; j += 1) {
        Ti cur = i_ptr[j];
        ఎంపిక (op) {
            స్థితి GGML_OP_POOL_AVG: res += cur; ఆపు;
            స్థితి GGML_OP_POOL_MAX: res = sycl::max(res, (గమ్యం_2) cur); ఆపు;
            అప్రమేయ:
                res = (గమ్యం_2) sycl::nan(uint32_t(0));
                ఆపు;
        }
    }

    స్థిరరాశి పూర్ణాంకం count = e - b;
    ఒకవేళ (op == GGML_OP_POOL_AVG) {
        res = (count > 0) ? (res / count) : (గమ్యం_2) 0;
    }
    o_ptr[cur_ow] = res;
}

శూన్య ggml_sycl_op_pool2d(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);
    dpct::వరుస_ptr main_ప్రవాహం = ctx.stream();
    SYCL_CHECK(ggml_sycl_సమితి_device(ctx.device));
    స్థిరరాశి దశాంశం * src0_dd = static_cast<స్థిరరాశి దశాంశం *>(dst->src[0]->దత్తాంశం);
    దశాంశం *       dst_dd  = static_cast<దశాంశం *>(dst->దత్తాంశం);

    స్థిరరాశి int32_t * opts = (స్థిరరాశి int32_t *)dst->op_params;
    గణన ggml_op_pool op = static_cast<ggml_op_pool>(opts[0]);
    స్థిరరాశి పూర్ణాంకం k0 = opts[1];
    స్థిరరాశి పూర్ణాంకం k1 = opts[2];
    స్థిరరాశి పూర్ణాంకం s0 = opts[3];
    స్థిరరాశి పూర్ణాంకం s1 = opts[4];
    స్థిరరాశి పూర్ణాంకం p0 = opts[5];
    స్థిరరాశి పూర్ణాంకం p1 = opts[6];

    స్థిరరాశి int64_t IH = dst->src[0]->ne[1];
    స్థిరరాశి int64_t IW = dst->src[0]->ne[0];

    స్థిరరాశి int64_t N = dst->ne[3];
    స్థిరరాశి int64_t OC = dst->ne[2];
    స్థిరరాశి int64_t OH = dst->ne[1];
    స్థిరరాశి int64_t OW = dst->ne[0];

    స్థిరరాశి పూర్ణాంకం parallel_elements = N * OC * OH * OW;
    స్థిరరాశి పూర్ణాంకం అంకె_ఖండాలు = (parallel_elements + SYCL_POOL2_D_ఖండం_కొలత - 1) / SYCL_POOL2_D_ఖండం_కొలత;
    sycl::range<3> ఖండం_nums(1, 1, అంకె_ఖండాలు);
    main_ప్రవాహం->parallel_for(
        sycl::nd_range<3>(ఖండం_nums *
                              sycl::range<3>(1, 1, SYCL_IM2_నిలువు_ఖండం_కొలత),
                          sycl::range<3>(1, 1, SYCL_IM2_నిలువు_ఖండం_కొలత)),
        [=](sycl::nd_అంశం<3> అంశం_ct1) {
            pool2d_nchw_kernel(IH, IW, OH, OW, k1, k0, s1, s0, p1, p0,
                               parallel_elements, src0_dd, dst_dd, op,
                               అంశం_ct1);
        });
}

శూన్య ggml_sycl_op_pool1d(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);
    dpct::వరుస_ptr main_ప్రవాహం = ctx.stream();
    SYCL_CHECK(ggml_sycl_సమితి_device(ctx.device));
    స్థిరరాశి దశాంశం * src0_dd = static_cast<స్థిరరాశి దశాంశం *>(dst->src[0]->దత్తాంశం);
    దశాంశం *       dst_dd  = static_cast<దశాంశం *>(dst->దత్తాంశం);

    స్థిరరాశి int32_t * opts = (స్థిరరాశి int32_t *)dst->op_params;
    గణన ggml_op_pool op = static_cast<ggml_op_pool>(opts[0]);
    స్థిరరాశి పూర్ణాంకం k0 = opts[1];
    స్థిరరాశి పూర్ణాంకం s0 = opts[2];
    స్థిరరాశి పూర్ణాంకం p0 = opts[3];

    స్థిరరాశి int64_t IW = dst->src[0]->ne[0];
    స్థిరరాశి int64_t OW = dst->ne[0];
    స్థిరరాశి int64_t NC = dst->ne[3] * dst->ne[2] * dst->ne[1];

    స్థిరరాశి పూర్ణాంకం parallel_elements = NC * OW;
    స్థిరరాశి పూర్ణాంకం అంకె_ఖండాలు = (parallel_elements + SYCL_POOL1_D_ఖండం_కొలత - 1) / SYCL_POOL1_D_ఖండం_కొలత;
    sycl::range<3> ఖండం_nums(1, 1, అంకె_ఖండాలు);
    main_ప్రవాహం->parallel_for(
        sycl::nd_range<3>(ఖండం_nums *
                              sycl::range<3>(1, 1, SYCL_POOL1_D_ఖండం_కొలత),
                          sycl::range<3>(1, 1, SYCL_POOL1_D_ఖండం_కొలత)),
        [=](sycl::nd_అంశం<3> అంశం_ct1) {
            pool1d_ncw_kernel(IW, OW, k0, s0, p0,
                              parallel_elements, src0_dd, dst_dd, op,
                              అంశం_ct1);
        });
}
