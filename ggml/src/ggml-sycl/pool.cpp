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

#ਅਨ੍ਤਰ੍ਭਾਵ "pool.hpp"
#ਅਨ੍ਤਰ੍ਭਾਵ <float.h>

ਸਾਂਚਾ <ਕਿਸਮ_ਨਾਂ Ti, ਕਿਸਮ_ਨਾਂ ਮੰਜ਼ਿਲ_2>
ਸਥਿਰ ਖਾਲੀ pool2d_nchw_kernel(
        ਅਚਲ ਪੂਰਨ ih, ਅਚਲ ਪੂਰਨ iw, ਅਚਲ ਪੂਰਨ oh, ਅਚਲ ਪੂਰਨ ow,
        ਅਚਲ ਪੂਰਨ kh, ਅਚਲ ਪੂਰਨ kw, ਅਚਲ ਪੂਰਨ sh, ਅਚਲ ਪੂਰਨ sw,
        ਅਚਲ ਪੂਰਨ ph, ਅਚਲ ਪੂਰਨ pw, ਅਚਲ ਪੂਰਨ parallel_elements,
        ਅਚਲ Ti* src, ਮੰਜ਼ਿਲ_2* dst, ਅਚਲ ਗਿਣਤੀ ggml_op_pool op,
        ਅਚਲ sycl::nd_ਵਸਤੂ<3> &ਵਸਤੂ_ct1) {
        ਪੂਰਨ idx = ਵਸਤੂ_ct1.get_local_id(2) +
                  ਵਸਤੂ_ct1.get_ਸਮੂਹ(2) * ਵਸਤੂ_ct1.get_local_range(2);
        ਜੇ (idx >= parallel_elements) {
            ਵਾਪਸ;
        }

        ਅਚਲ ਪੂਰਨ I_HW = ih * iw;
        ਅਚਲ ਪੂਰਨ O_HW = oh * ow;
        ਅਚਲ ਪੂਰਨ nc = idx / O_HW;
        ਅਚਲ ਪੂਰਨ cur_oh = idx % O_HW / ow;
        ਅਚਲ ਪੂਰਨ cur_ow = idx % O_HW % ow;
        ਅਚਲ Ti* i_ptr = src + nc * I_HW;
        ਮੰਜ਼ਿਲ_2* o_ptr = dst + nc * O_HW;
        ਅਚਲ ਪੂਰਨ ਸ਼ੁਰੂ_h = cur_oh * sh - ph;
        ਅਚਲ ਪੂਰਨ bh = sycl::max(0, ਸ਼ੁਰੂ_h);
        ਅਚਲ ਪੂਰਨ eh = sycl::min(ih, ਸ਼ੁਰੂ_h + kh);
        ਅਚਲ ਪੂਰਨ ਸ਼ੁਰੂ_w = cur_ow * sw - pw;
        ਅਚਲ ਪੂਰਨ bw = sycl::max(0, ਸ਼ੁਰੂ_w);
        ਅਚਲ ਪੂਰਨ ew = sycl::min(iw, ਸ਼ੁਰੂ_w + kw);

        ਮੰਜ਼ਿਲ_2 res = 0;

        ਚੋਣ (op) {
            ਹਾਲਤ GGML_OP_POOL_AVG: res = 0; ਰੋਕੋ;
            ਹਾਲਤ GGML_OP_POOL_MAX: res = -FLT_MAX; ਰੋਕੋ;
            ਮੂਲ:
                res      = (ਮੰਜ਼ਿਲ_2) sycl::nan(uint32_t(0));
                ਰੋਕੋ;
        }

        ਲਈ (ਪੂਰਨ i = bh; i < eh; i += 1) {
            ਲਈ (ਪੂਰਨ j = bw; j < ew; j += 1) {
                Ti cur = i_ptr[i * iw + j];
                ਚੋਣ (op) {
                    ਹਾਲਤ GGML_OP_POOL_AVG: res += (cur / (kh * kw)); ਰੋਕੋ;
                    ਹਾਲਤ GGML_OP_POOL_MAX: res = sycl::max(res, (ਮੰਜ਼ਿਲ_2)cur); ਰੋਕੋ;
                    ਮੂਲ:
                        res = (ਮੰਜ਼ਿਲ_2) sycl::nan(uint32_t(0));
                        ਰੋਕੋ;
                }
            }
        }
        o_ptr[cur_oh * ow + cur_ow] = res;
}

ਸਾਂਚਾ <ਕਿਸਮ_ਨਾਂ Ti, ਕਿਸਮ_ਨਾਂ ਮੰਜ਼ਿਲ_2>
ਸਥਿਰ ਖਾਲੀ pool1d_ncw_kernel(
        ਅਚਲ ਪੂਰਨ iw, ਅਚਲ ਪੂਰਨ ow,
        ਅਚਲ ਪੂਰਨ k, ਅਚਲ ਪੂਰਨ s,
        ਅਚਲ ਪੂਰਨ p, ਅਚਲ ਪੂਰਨ parallel_elements,
        ਅਚਲ Ti * src, ਮੰਜ਼ਿਲ_2 * dst, ਅਚਲ ਗਿਣਤੀ ggml_op_pool op,
        ਅਚਲ sycl::nd_ਵਸਤੂ<3> & ਵਸਤੂ_ct1) {
    ਪੂਰਨ idx = ਵਸਤੂ_ct1.get_local_id(2) +
              ਵਸਤੂ_ct1.get_ਸਮੂਹ(2) * ਵਸਤੂ_ct1.get_local_range(2);
    ਜੇ (idx >= parallel_elements) {
        ਵਾਪਸ;
    }

    ਅਚਲ ਪੂਰਨ nc     = idx / ow;
    ਅਚਲ ਪੂਰਨ cur_ow = idx % ow;
    ਅਚਲ Ti * i_ptr = src + nc * iw;
    ਮੰਜ਼ਿਲ_2 *       o_ptr = dst + nc * ow;
    ਅਚਲ ਪੂਰਨ  start = cur_ow * s - p;
    ਅਚਲ ਪੂਰਨ  b     = sycl::max(0, start);
    ਅਚਲ ਪੂਰਨ  e     = sycl::min(iw, start + k);

    ਮੰਜ਼ਿਲ_2 res = 0;
    ਚੋਣ (op) {
        ਹਾਲਤ GGML_OP_POOL_AVG: res = 0;        ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_POOL_MAX: res = -FLT_MAX; ਰੋਕੋ;
        ਮੂਲ:
            res = (ਮੰਜ਼ਿਲ_2) sycl::nan(uint32_t(0));
            ਰੋਕੋ;
    }

    ਲਈ (ਪੂਰਨ j = b; j < e; j += 1) {
        Ti cur = i_ptr[j];
        ਚੋਣ (op) {
            ਹਾਲਤ GGML_OP_POOL_AVG: res += cur; ਰੋਕੋ;
            ਹਾਲਤ GGML_OP_POOL_MAX: res = sycl::max(res, (ਮੰਜ਼ਿਲ_2) cur); ਰੋਕੋ;
            ਮੂਲ:
                res = (ਮੰਜ਼ਿਲ_2) sycl::nan(uint32_t(0));
                ਰੋਕੋ;
        }
    }

    ਅਚਲ ਪੂਰਨ count = e - b;
    ਜੇ (op == GGML_OP_POOL_AVG) {
        res = (count > 0) ? (res / count) : (ਮੰਜ਼ਿਲ_2) 0;
    }
    o_ptr[cur_ow] = res;
}

ਖਾਲੀ ggml_sycl_op_pool2d(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);
    dpct::ਕਤਾਰ_ptr main_ਧਾਰਾ = ctx.stream();
    SYCL_CHECK(ggml_sycl_ਸੈੱਟ_device(ctx.device));
    ਅਚਲ ਦਸ਼ਮਲਵ * src0_dd = static_cast<ਅਚਲ ਦਸ਼ਮਲਵ *>(dst->src[0]->ਅੰਕੜੇ);
    ਦਸ਼ਮਲਵ *       dst_dd  = static_cast<ਦਸ਼ਮਲਵ *>(dst->ਅੰਕੜੇ);

    ਅਚਲ int32_t * opts = (ਅਚਲ int32_t *)dst->op_params;
    ਗਿਣਤੀ ggml_op_pool op = static_cast<ggml_op_pool>(opts[0]);
    ਅਚਲ ਪੂਰਨ k0 = opts[1];
    ਅਚਲ ਪੂਰਨ k1 = opts[2];
    ਅਚਲ ਪੂਰਨ s0 = opts[3];
    ਅਚਲ ਪੂਰਨ s1 = opts[4];
    ਅਚਲ ਪੂਰਨ p0 = opts[5];
    ਅਚਲ ਪੂਰਨ p1 = opts[6];

    ਅਚਲ int64_t IH = dst->src[0]->ne[1];
    ਅਚਲ int64_t IW = dst->src[0]->ne[0];

    ਅਚਲ int64_t N = dst->ne[3];
    ਅਚਲ int64_t OC = dst->ne[2];
    ਅਚਲ int64_t OH = dst->ne[1];
    ਅਚਲ int64_t OW = dst->ne[0];

    ਅਚਲ ਪੂਰਨ parallel_elements = N * OC * OH * OW;
    ਅਚਲ ਪੂਰਨ ਅੰਕ_ਖੰਡਾਂ = (parallel_elements + SYCL_POOL2_D_ਖੰਡ_ਮਾਪ - 1) / SYCL_POOL2_D_ਖੰਡ_ਮਾਪ;
    sycl::range<3> ਖੰਡ_nums(1, 1, ਅੰਕ_ਖੰਡਾਂ);
    main_ਧਾਰਾ->parallel_for(
        sycl::nd_range<3>(ਖੰਡ_nums *
                              sycl::range<3>(1, 1, SYCL_IM2_ਖਾਨਾ_ਖੰਡ_ਮਾਪ),
                          sycl::range<3>(1, 1, SYCL_IM2_ਖਾਨਾ_ਖੰਡ_ਮਾਪ)),
        [=](sycl::nd_ਵਸਤੂ<3> ਵਸਤੂ_ct1) {
            pool2d_nchw_kernel(IH, IW, OH, OW, k1, k0, s1, s0, p1, p0,
                               parallel_elements, src0_dd, dst_dd, op,
                               ਵਸਤੂ_ct1);
        });
}

ਖਾਲੀ ggml_sycl_op_pool1d(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);
    dpct::ਕਤਾਰ_ptr main_ਧਾਰਾ = ctx.stream();
    SYCL_CHECK(ggml_sycl_ਸੈੱਟ_device(ctx.device));
    ਅਚਲ ਦਸ਼ਮਲਵ * src0_dd = static_cast<ਅਚਲ ਦਸ਼ਮਲਵ *>(dst->src[0]->ਅੰਕੜੇ);
    ਦਸ਼ਮਲਵ *       dst_dd  = static_cast<ਦਸ਼ਮਲਵ *>(dst->ਅੰਕੜੇ);

    ਅਚਲ int32_t * opts = (ਅਚਲ int32_t *)dst->op_params;
    ਗਿਣਤੀ ggml_op_pool op = static_cast<ggml_op_pool>(opts[0]);
    ਅਚਲ ਪੂਰਨ k0 = opts[1];
    ਅਚਲ ਪੂਰਨ s0 = opts[2];
    ਅਚਲ ਪੂਰਨ p0 = opts[3];

    ਅਚਲ int64_t IW = dst->src[0]->ne[0];
    ਅਚਲ int64_t OW = dst->ne[0];
    ਅਚਲ int64_t NC = dst->ne[3] * dst->ne[2] * dst->ne[1];

    ਅਚਲ ਪੂਰਨ parallel_elements = NC * OW;
    ਅਚਲ ਪੂਰਨ ਅੰਕ_ਖੰਡਾਂ = (parallel_elements + SYCL_POOL1_D_ਖੰਡ_ਮਾਪ - 1) / SYCL_POOL1_D_ਖੰਡ_ਮਾਪ;
    sycl::range<3> ਖੰਡ_nums(1, 1, ਅੰਕ_ਖੰਡਾਂ);
    main_ਧਾਰਾ->parallel_for(
        sycl::nd_range<3>(ਖੰਡ_nums *
                              sycl::range<3>(1, 1, SYCL_POOL1_D_ਖੰਡ_ਮਾਪ),
                          sycl::range<3>(1, 1, SYCL_POOL1_D_ਖੰਡ_ਮਾਪ)),
        [=](sycl::nd_ਵਸਤੂ<3> ਵਸਤੂ_ct1) {
            pool1d_ncw_kernel(IW, OW, k0, s0, p0,
                              parallel_elements, src0_dd, dst_dd, op,
                              ਵਸਤੂ_ct1);
        });
}
