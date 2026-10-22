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

سانچو <قسم_نالو Ti, قسم_نالو منزل_2>
ساكن فراغ pool2d_nchw_kernel(
        ثابت صحيح ih, ثابت صحيح iw, ثابت صحيح oh, ثابت صحيح ow,
        ثابت صحيح kh, ثابت صحيح kw, ثابت صحيح sh, ثابت صحيح sw,
        ثابت صحيح ph, ثابت صحيح pw, ثابت صحيح parallel_elements,
        ثابت Ti* src, منزل_2* dst, ثابت تعداد ggml_op_pool op,
        ثابت sycl::nd_شي<3> &شي_ct1) {
        صحيح idx = شي_ct1.get_local_id(2) +
                  شي_ct1.get_گروهه(2) * شي_ct1.get_local_range(2);
        إذا (idx >= parallel_elements) {
            أرجع;
        }

        ثابت صحيح I_HW = ih * iw;
        ثابت صحيح O_HW = oh * ow;
        ثابت صحيح nc = idx / O_HW;
        ثابت صحيح cur_oh = idx % O_HW / ow;
        ثابت صحيح cur_ow = idx % O_HW % ow;
        ثابت Ti* i_ptr = src + nc * I_HW;
        منزل_2* o_ptr = dst + nc * O_HW;
        ثابت صحيح شروعات_h = cur_oh * sh - ph;
        ثابت صحيح bh = sycl::max(0, شروعات_h);
        ثابت صحيح eh = sycl::min(ih, شروعات_h + kh);
        ثابت صحيح شروعات_w = cur_ow * sw - pw;
        ثابت صحيح bw = sycl::max(0, شروعات_w);
        ثابت صحيح ew = sycl::min(iw, شروعات_w + kw);

        منزل_2 res = 0;

        بدل (op) {
            حالة GGML_OP_POOL_AVG: res = 0; اكسر;
            حالة GGML_OP_POOL_MAX: res = -FLT_MAX; اكسر;
            افتراضي:
                res      = (منزل_2) sycl::nan(uint32_t(0));
                اكسر;
        }

        لكل (صحيح i = bh; i < eh; i += 1) {
            لكل (صحيح j = bw; j < ew; j += 1) {
                Ti cur = i_ptr[i * iw + j];
                بدل (op) {
                    حالة GGML_OP_POOL_AVG: res += (cur / (kh * kw)); اكسر;
                    حالة GGML_OP_POOL_MAX: res = sycl::max(res, (منزل_2)cur); اكسر;
                    افتراضي:
                        res = (منزل_2) sycl::nan(uint32_t(0));
                        اكسر;
                }
            }
        }
        o_ptr[cur_oh * ow + cur_ow] = res;
}

سانچو <قسم_نالو Ti, قسم_نالو منزل_2>
ساكن فراغ pool1d_ncw_kernel(
        ثابت صحيح iw, ثابت صحيح ow,
        ثابت صحيح k, ثابت صحيح s,
        ثابت صحيح p, ثابت صحيح parallel_elements,
        ثابت Ti * src, منزل_2 * dst, ثابت تعداد ggml_op_pool op,
        ثابت sycl::nd_شي<3> & شي_ct1) {
    صحيح idx = شي_ct1.get_local_id(2) +
              شي_ct1.get_گروهه(2) * شي_ct1.get_local_range(2);
    إذا (idx >= parallel_elements) {
        أرجع;
    }

    ثابت صحيح nc     = idx / ow;
    ثابت صحيح cur_ow = idx % ow;
    ثابت Ti * i_ptr = src + nc * iw;
    منزل_2 *       o_ptr = dst + nc * ow;
    ثابت صحيح  start = cur_ow * s - p;
    ثابت صحيح  b     = sycl::max(0, start);
    ثابت صحيح  e     = sycl::min(iw, start + k);

    منزل_2 res = 0;
    بدل (op) {
        حالة GGML_OP_POOL_AVG: res = 0;        اكسر;
        حالة GGML_OP_POOL_MAX: res = -FLT_MAX; اكسر;
        افتراضي:
            res = (منزل_2) sycl::nan(uint32_t(0));
            اكسر;
    }

    لكل (صحيح j = b; j < e; j += 1) {
        Ti cur = i_ptr[j];
        بدل (op) {
            حالة GGML_OP_POOL_AVG: res += cur; اكسر;
            حالة GGML_OP_POOL_MAX: res = sycl::max(res, (منزل_2) cur); اكسر;
            افتراضي:
                res = (منزل_2) sycl::nan(uint32_t(0));
                اكسر;
        }
    }

    ثابت صحيح count = e - b;
    إذا (op == GGML_OP_POOL_AVG) {
        res = (count > 0) ? (res / count) : (منزل_2) 0;
    }
    o_ptr[cur_ow] = res;
}

فراغ ggml_sycl_op_pool2d(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);
    dpct::قطار_ptr main_وهڪرو = ctx.stream();
    SYCL_CHECK(ggml_sycl_سيٽ_device(ctx.device));
    ثابت عائم * src0_dd = static_cast<ثابت عائم *>(dst->src[0]->معلومات);
    عائم *       dst_dd  = static_cast<عائم *>(dst->معلومات);

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
    ثابت صحيح انگ_بلاڪون = (parallel_elements + SYCL_POOL2_D_بلاڪ_سائيز - 1) / SYCL_POOL2_D_بلاڪ_سائيز;
    sycl::range<3> بلاڪ_nums(1, 1, انگ_بلاڪون);
    main_وهڪرو->parallel_for(
        sycl::nd_range<3>(بلاڪ_nums *
                              sycl::range<3>(1, 1, SYCL_IM2_کانو_بلاڪ_سائيز),
                          sycl::range<3>(1, 1, SYCL_IM2_کانو_بلاڪ_سائيز)),
        [=](sycl::nd_شي<3> شي_ct1) {
            pool2d_nchw_kernel(IH, IW, OH, OW, k1, k0, s1, s0, p1, p0,
                               parallel_elements, src0_dd, dst_dd, op,
                               شي_ct1);
        });
}

فراغ ggml_sycl_op_pool1d(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);
    dpct::قطار_ptr main_وهڪرو = ctx.stream();
    SYCL_CHECK(ggml_sycl_سيٽ_device(ctx.device));
    ثابت عائم * src0_dd = static_cast<ثابت عائم *>(dst->src[0]->معلومات);
    عائم *       dst_dd  = static_cast<عائم *>(dst->معلومات);

    ثابت int32_t * opts = (ثابت int32_t *)dst->op_params;
    تعداد ggml_op_pool op = static_cast<ggml_op_pool>(opts[0]);
    ثابت صحيح k0 = opts[1];
    ثابت صحيح s0 = opts[2];
    ثابت صحيح p0 = opts[3];

    ثابت int64_t IW = dst->src[0]->ne[0];
    ثابت int64_t OW = dst->ne[0];
    ثابت int64_t NC = dst->ne[3] * dst->ne[2] * dst->ne[1];

    ثابت صحيح parallel_elements = NC * OW;
    ثابت صحيح انگ_بلاڪون = (parallel_elements + SYCL_POOL1_D_بلاڪ_سائيز - 1) / SYCL_POOL1_D_بلاڪ_سائيز;
    sycl::range<3> بلاڪ_nums(1, 1, انگ_بلاڪون);
    main_وهڪرو->parallel_for(
        sycl::nd_range<3>(بلاڪ_nums *
                              sycl::range<3>(1, 1, SYCL_POOL1_D_بلاڪ_سائيز),
                          sycl::range<3>(1, 1, SYCL_POOL1_D_بلاڪ_سائيز)),
        [=](sycl::nd_شي<3> شي_ct1) {
            pool1d_ncw_kernel(IW, OW, k0, s0, p0,
                              parallel_elements, src0_dd, dst_dd, op,
                              شي_ct1);
        });
}
