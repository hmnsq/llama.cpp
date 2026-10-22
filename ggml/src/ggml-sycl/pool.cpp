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

#ضمن "pool.hpp"
#ضمن <float.h>

قالب <نام_نوع Ti, نام_نوع مقصد_2>
ساكن فراغ pool2d_nchw_kernel(
        ثابت صحيح ih, ثابت صحيح iw, ثابت صحيح oh, ثابت صحيح ow,
        ثابت صحيح kh, ثابت صحيح kw, ثابت صحيح sh, ثابت صحيح sw,
        ثابت صحيح ph, ثابت صحيح pw, ثابت صحيح parallel_elements,
        ثابت Ti* src, مقصد_2* dst, ثابت تعداد ggml_op_pool op,
        ثابت sycl::nd_قلم<3> &قلم_ct1) {
        صحيح idx = قلم_ct1.get_local_id(2) +
                  قلم_ct1.get_گروه(2) * قلم_ct1.get_local_range(2);
        إذا (idx >= parallel_elements) {
            أرجع;
        }

        ثابت صحيح I_HW = ih * iw;
        ثابت صحيح O_HW = oh * ow;
        ثابت صحيح nc = idx / O_HW;
        ثابت صحيح cur_oh = idx % O_HW / ow;
        ثابت صحيح cur_ow = idx % O_HW % ow;
        ثابت Ti* i_ptr = src + nc * I_HW;
        مقصد_2* o_ptr = dst + nc * O_HW;
        ثابت صحيح شروع_h = cur_oh * sh - ph;
        ثابت صحيح bh = sycl::max(0, شروع_h);
        ثابت صحيح eh = sycl::min(ih, شروع_h + kh);
        ثابت صحيح شروع_w = cur_ow * sw - pw;
        ثابت صحيح bw = sycl::max(0, شروع_w);
        ثابت صحيح ew = sycl::min(iw, شروع_w + kw);

        مقصد_2 res = 0;

        بدل (op) {
            حالة GGML_OP_POOL_AVG: res = 0; اكسر;
            حالة GGML_OP_POOL_MAX: res = -FLT_MAX; اكسر;
            افتراضي:
                res      = (مقصد_2) sycl::nan(uint32_t(0));
                اكسر;
        }

        لكل (صحيح i = bh; i < eh; i += 1) {
            لكل (صحيح j = bw; j < ew; j += 1) {
                Ti cur = i_ptr[i * iw + j];
                بدل (op) {
                    حالة GGML_OP_POOL_AVG: res += (cur / (kh * kw)); اكسر;
                    حالة GGML_OP_POOL_MAX: res = sycl::max(res, (مقصد_2)cur); اكسر;
                    افتراضي:
                        res = (مقصد_2) sycl::nan(uint32_t(0));
                        اكسر;
                }
            }
        }
        o_ptr[cur_oh * ow + cur_ow] = res;
}

قالب <نام_نوع Ti, نام_نوع مقصد_2>
ساكن فراغ pool1d_ncw_kernel(
        ثابت صحيح iw, ثابت صحيح ow,
        ثابت صحيح k, ثابت صحيح s,
        ثابت صحيح p, ثابت صحيح parallel_elements,
        ثابت Ti * src, مقصد_2 * dst, ثابت تعداد ggml_op_pool op,
        ثابت sycl::nd_قلم<3> & قلم_ct1) {
    صحيح idx = قلم_ct1.get_local_id(2) +
              قلم_ct1.get_گروه(2) * قلم_ct1.get_local_range(2);
    إذا (idx >= parallel_elements) {
        أرجع;
    }

    ثابت صحيح nc     = idx / ow;
    ثابت صحيح cur_ow = idx % ow;
    ثابت Ti * i_ptr = src + nc * iw;
    مقصد_2 *       o_ptr = dst + nc * ow;
    ثابت صحيح  start = cur_ow * s - p;
    ثابت صحيح  b     = sycl::max(0, start);
    ثابت صحيح  e     = sycl::min(iw, start + k);

    مقصد_2 res = 0;
    بدل (op) {
        حالة GGML_OP_POOL_AVG: res = 0;        اكسر;
        حالة GGML_OP_POOL_MAX: res = -FLT_MAX; اكسر;
        افتراضي:
            res = (مقصد_2) sycl::nan(uint32_t(0));
            اكسر;
    }

    لكل (صحيح j = b; j < e; j += 1) {
        Ti cur = i_ptr[j];
        بدل (op) {
            حالة GGML_OP_POOL_AVG: res += cur; اكسر;
            حالة GGML_OP_POOL_MAX: res = sycl::max(res, (مقصد_2) cur); اكسر;
            افتراضي:
                res = (مقصد_2) sycl::nan(uint32_t(0));
                اكسر;
        }
    }

    ثابت صحيح count = e - b;
    إذا (op == GGML_OP_POOL_AVG) {
        res = (count > 0) ? (res / count) : (مقصد_2) 0;
    }
    o_ptr[cur_ow] = res;
}

فراغ ggml_sycl_op_pool2d(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);
    dpct::صف_ptr main_جریان = ctx.stream();
    SYCL_CHECK(ggml_sycl_مجموعه_device(ctx.device));
    ثابت عائم * src0_dd = static_cast<ثابت عائم *>(dst->src[0]->داده);
    عائم *       dst_dd  = static_cast<عائم *>(dst->داده);

    ثابت int32_t * opts = (ثابت int32_t *)dst->op_params;
    تعداد ggml_op_pool op = static_cast<ggml_op_pool>(opts[0]);
    ثابت صحيح k0 = opts[1];
    ثابت صحيح k1 = opts[2];
    ثابت صحيح s0 = opts[3];
    ثابت صحيح s1 = opts[4];
    ثابت صحيح p0 = opts[5];
    ثابت صحيح p1 = opts[6];

    ثابت int64_t IH = dst->src[0]->ne[1];
    ثابت int64_t IW = dst->src[0]->ne[0];

    ثابت int64_t N = dst->ne[3];
    ثابت int64_t OC = dst->ne[2];
    ثابت int64_t OH = dst->ne[1];
    ثابت int64_t OW = dst->ne[0];

    ثابت صحيح parallel_elements = N * OC * OH * OW;
    ثابت صحيح رقم_بلوکها = (parallel_elements + SYCL_POOL2_D_بلوک_اندازه - 1) / SYCL_POOL2_D_بلوک_اندازه;
    sycl::range<3> بلوک_nums(1, 1, رقم_بلوکها);
    main_جریان->parallel_for(
        sycl::nd_range<3>(بلوک_nums *
                              sycl::range<3>(1, 1, SYCL_IM2_ستون_بلوک_اندازه),
                          sycl::range<3>(1, 1, SYCL_IM2_ستون_بلوک_اندازه)),
        [=](sycl::nd_قلم<3> قلم_ct1) {
            pool2d_nchw_kernel(IH, IW, OH, OW, k1, k0, s1, s0, p1, p0,
                               parallel_elements, src0_dd, dst_dd, op,
                               قلم_ct1);
        });
}

فراغ ggml_sycl_op_pool1d(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);
    dpct::صف_ptr main_جریان = ctx.stream();
    SYCL_CHECK(ggml_sycl_مجموعه_device(ctx.device));
    ثابت عائم * src0_dd = static_cast<ثابت عائم *>(dst->src[0]->داده);
    عائم *       dst_dd  = static_cast<عائم *>(dst->داده);

    ثابت int32_t * opts = (ثابت int32_t *)dst->op_params;
    تعداد ggml_op_pool op = static_cast<ggml_op_pool>(opts[0]);
    ثابت صحيح k0 = opts[1];
    ثابت صحيح s0 = opts[2];
    ثابت صحيح p0 = opts[3];

    ثابت int64_t IW = dst->src[0]->ne[0];
    ثابت int64_t OW = dst->ne[0];
    ثابت int64_t NC = dst->ne[3] * dst->ne[2] * dst->ne[1];

    ثابت صحيح parallel_elements = NC * OW;
    ثابت صحيح رقم_بلوکها = (parallel_elements + SYCL_POOL1_D_بلوک_اندازه - 1) / SYCL_POOL1_D_بلوک_اندازه;
    sycl::range<3> بلوک_nums(1, 1, رقم_بلوکها);
    main_جریان->parallel_for(
        sycl::nd_range<3>(بلوک_nums *
                              sycl::range<3>(1, 1, SYCL_POOL1_D_بلوک_اندازه),
                          sycl::range<3>(1, 1, SYCL_POOL1_D_بلوک_اندازه)),
        [=](sycl::nd_قلم<3> قلم_ct1) {
            pool1d_ncw_kernel(IW, OW, k0, s0, p0,
                              parallel_elements, src0_dd, dst_dd, op,
                              قلم_ct1);
        });
}
