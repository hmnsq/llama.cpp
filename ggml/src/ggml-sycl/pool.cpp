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

#شامل "pool.hpp"
#شامل <float.h>

سانچہ <قسم_نام Ti, قسم_نام مقصد_2>
ساکن خالی pool2d_nchw_kernel(
        ثابت صحیح ih, ثابت صحیح iw, ثابت صحیح oh, ثابت صحیح ow,
        ثابت صحیح kh, ثابت صحیح kw, ثابت صحیح sh, ثابت صحیح sw,
        ثابت صحیح ph, ثابت صحیح pw, ثابت صحیح parallel_elements,
        ثابت Ti* src, مقصد_2* dst, ثابت شمار ggml_op_pool op,
        ثابت sycl::nd_شے<3> &شے_ct1) {
        صحیح idx = شے_ct1.get_local_id(2) +
                  شے_ct1.get_گروہ(2) * شے_ct1.get_local_range(2);
        اگر (idx >= parallel_elements) {
            واپس;
        }

        ثابت صحیح I_HW = ih * iw;
        ثابت صحیح O_HW = oh * ow;
        ثابت صحیح nc = idx / O_HW;
        ثابت صحیح cur_oh = idx % O_HW / ow;
        ثابت صحیح cur_ow = idx % O_HW % ow;
        ثابت Ti* i_ptr = src + nc * I_HW;
        مقصد_2* o_ptr = dst + nc * O_HW;
        ثابت صحیح آغاز_h = cur_oh * sh - ph;
        ثابت صحیح bh = sycl::max(0, آغاز_h);
        ثابت صحیح eh = sycl::min(ih, آغاز_h + kh);
        ثابت صحیح آغاز_w = cur_ow * sw - pw;
        ثابت صحیح bw = sycl::max(0, آغاز_w);
        ثابت صحیح ew = sycl::min(iw, آغاز_w + kw);

        مقصد_2 res = 0;

        انتخاب (op) {
            حالت GGML_OP_POOL_AVG: res = 0; توڑو;
            حالت GGML_OP_POOL_MAX: res = -FLT_MAX; توڑو;
            طےشدہ:
                res      = (مقصد_2) sycl::nan(uint32_t(0));
                توڑو;
        }

        برائے (صحیح i = bh; i < eh; i += 1) {
            برائے (صحیح j = bw; j < ew; j += 1) {
                Ti cur = i_ptr[i * iw + j];
                انتخاب (op) {
                    حالت GGML_OP_POOL_AVG: res += (cur / (kh * kw)); توڑو;
                    حالت GGML_OP_POOL_MAX: res = sycl::max(res, (مقصد_2)cur); توڑو;
                    طےشدہ:
                        res = (مقصد_2) sycl::nan(uint32_t(0));
                        توڑو;
                }
            }
        }
        o_ptr[cur_oh * ow + cur_ow] = res;
}

سانچہ <قسم_نام Ti, قسم_نام مقصد_2>
ساکن خالی pool1d_ncw_kernel(
        ثابت صحیح iw, ثابت صحیح ow,
        ثابت صحیح k, ثابت صحیح s,
        ثابت صحیح p, ثابت صحیح parallel_elements,
        ثابت Ti * src, مقصد_2 * dst, ثابت شمار ggml_op_pool op,
        ثابت sycl::nd_شے<3> & شے_ct1) {
    صحیح idx = شے_ct1.get_local_id(2) +
              شے_ct1.get_گروہ(2) * شے_ct1.get_local_range(2);
    اگر (idx >= parallel_elements) {
        واپس;
    }

    ثابت صحیح nc     = idx / ow;
    ثابت صحیح cur_ow = idx % ow;
    ثابت Ti * i_ptr = src + nc * iw;
    مقصد_2 *       o_ptr = dst + nc * ow;
    ثابت صحیح  start = cur_ow * s - p;
    ثابت صحیح  b     = sycl::max(0, start);
    ثابت صحیح  e     = sycl::min(iw, start + k);

    مقصد_2 res = 0;
    انتخاب (op) {
        حالت GGML_OP_POOL_AVG: res = 0;        توڑو;
        حالت GGML_OP_POOL_MAX: res = -FLT_MAX; توڑو;
        طےشدہ:
            res = (مقصد_2) sycl::nan(uint32_t(0));
            توڑو;
    }

    برائے (صحیح j = b; j < e; j += 1) {
        Ti cur = i_ptr[j];
        انتخاب (op) {
            حالت GGML_OP_POOL_AVG: res += cur; توڑو;
            حالت GGML_OP_POOL_MAX: res = sycl::max(res, (مقصد_2) cur); توڑو;
            طےشدہ:
                res = (مقصد_2) sycl::nan(uint32_t(0));
                توڑو;
        }
    }

    ثابت صحیح count = e - b;
    اگر (op == GGML_OP_POOL_AVG) {
        res = (count > 0) ? (res / count) : (مقصد_2) 0;
    }
    o_ptr[cur_ow] = res;
}

خالی ggml_sycl_op_pool2d(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);
    dpct::قطار_ptr main_دھارا = ctx.stream();
    SYCL_CHECK(ggml_sycl_سیٹ_device(ctx.device));
    ثابت اعشاری * src0_dd = static_cast<ثابت اعشاری *>(dst->src[0]->ڈیٹا);
    اعشاری *       dst_dd  = static_cast<اعشاری *>(dst->ڈیٹا);

    ثابت int32_t * opts = (ثابت int32_t *)dst->op_params;
    شمار ggml_op_pool op = static_cast<ggml_op_pool>(opts[0]);
    ثابت صحیح k0 = opts[1];
    ثابت صحیح k1 = opts[2];
    ثابت صحیح s0 = opts[3];
    ثابت صحیح s1 = opts[4];
    ثابت صحیح p0 = opts[5];
    ثابت صحیح p1 = opts[6];

    ثابت int64_t IH = dst->src[0]->ne[1];
    ثابت int64_t IW = dst->src[0]->ne[0];

    ثابت int64_t N = dst->ne[3];
    ثابت int64_t OC = dst->ne[2];
    ثابت int64_t OH = dst->ne[1];
    ثابت int64_t OW = dst->ne[0];

    ثابت صحیح parallel_elements = N * OC * OH * OW;
    ثابت صحیح عدد_بلاکیں = (parallel_elements + SYCL_POOL2_D_بلاک_سائز - 1) / SYCL_POOL2_D_بلاک_سائز;
    sycl::range<3> بلاک_nums(1, 1, عدد_بلاکیں);
    main_دھارا->parallel_for(
        sycl::nd_range<3>(بلاک_nums *
                              sycl::range<3>(1, 1, SYCL_IM2_ستون_بلاک_سائز),
                          sycl::range<3>(1, 1, SYCL_IM2_ستون_بلاک_سائز)),
        [=](sycl::nd_شے<3> شے_ct1) {
            pool2d_nchw_kernel(IH, IW, OH, OW, k1, k0, s1, s0, p1, p0,
                               parallel_elements, src0_dd, dst_dd, op,
                               شے_ct1);
        });
}

خالی ggml_sycl_op_pool1d(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);
    dpct::قطار_ptr main_دھارا = ctx.stream();
    SYCL_CHECK(ggml_sycl_سیٹ_device(ctx.device));
    ثابت اعشاری * src0_dd = static_cast<ثابت اعشاری *>(dst->src[0]->ڈیٹا);
    اعشاری *       dst_dd  = static_cast<اعشاری *>(dst->ڈیٹا);

    ثابت int32_t * opts = (ثابت int32_t *)dst->op_params;
    شمار ggml_op_pool op = static_cast<ggml_op_pool>(opts[0]);
    ثابت صحیح k0 = opts[1];
    ثابت صحیح s0 = opts[2];
    ثابت صحیح p0 = opts[3];

    ثابت int64_t IW = dst->src[0]->ne[0];
    ثابت int64_t OW = dst->ne[0];
    ثابت int64_t NC = dst->ne[3] * dst->ne[2] * dst->ne[1];

    ثابت صحیح parallel_elements = NC * OW;
    ثابت صحیح عدد_بلاکیں = (parallel_elements + SYCL_POOL1_D_بلاک_سائز - 1) / SYCL_POOL1_D_بلاک_سائز;
    sycl::range<3> بلاک_nums(1, 1, عدد_بلاکیں);
    main_دھارا->parallel_for(
        sycl::nd_range<3>(بلاک_nums *
                              sycl::range<3>(1, 1, SYCL_POOL1_D_بلاک_سائز),
                          sycl::range<3>(1, 1, SYCL_POOL1_D_بلاک_سائز)),
        [=](sycl::nd_شے<3> شے_ct1) {
            pool1d_ncw_kernel(IW, OW, k0, s0, p0,
                              parallel_elements, src0_dd, dst_dd, op,
                              شے_ct1);
        });
}
