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

#include "pool.hpp"
#include <float.h>

template <typename Ti, typename To>
static void pool2d_nchw_kernel(
        const цео ih, const цео iw, const цео oh, const цео ow,
        const цео kh, const цео kw, const цео sh, const цео sw,
        const цео ph, const цео pw, const цео parallel_elements,
        const Ti* src, To* dst, const enum ggml_op_pool op,
        const sycl::nd_item<3> &item_ct1) {
        цео idx = item_ct1.get_local_id(2) +
                  item_ct1.get_group(2) * item_ct1.get_local_range(2);
        ако (idx >= parallel_elements) {
            врати;
        }

        const цео I_HW = ih * iw;
        const цео O_HW = oh * ow;
        const цео nc = idx / O_HW;
        const цео cur_oh = idx % O_HW / ow;
        const цео cur_ow = idx % O_HW % ow;
        const Ti* i_ptr = src + nc * I_HW;
        To* o_ptr = dst + nc * O_HW;
        const цео start_h = cur_oh * sh - ph;
        const цео bh = sycl::max(0, start_h);
        const цео eh = sycl::min(ih, start_h + kh);
        const цео start_w = cur_ow * sw - pw;
        const цео bw = sycl::max(0, start_w);
        const цео ew = sycl::min(iw, start_w + kw);

        To res = 0;

        switch (op) {
            case GGML_OP_POOL_AVG: res = 0; break;
            case GGML_OP_POOL_MAX: res = -FLT_MAX; break;
            default:
                res      = (To) sycl::nan(uint32_t(0));
                break;
        }

        за (цео i = bh; i < eh; i += 1) {
            за (цео j = bw; j < ew; j += 1) {
                Ti cur = i_ptr[i * iw + j];
                switch (op) {
                    case GGML_OP_POOL_AVG: res += (cur / (kh * kw)); break;
                    case GGML_OP_POOL_MAX: res = sycl::max(res, (To)cur); break;
                    default:
                        res = (To) sycl::nan(uint32_t(0));
                        break;
                }
            }
        }
        o_ptr[cur_oh * ow + cur_ow] = res;
}

template <typename Ti, typename To>
static void pool1d_ncw_kernel(
        const цео iw, const цео ow,
        const цео k, const цео s,
        const цео p, const цео parallel_elements,
        const Ti * src, To * dst, const enum ggml_op_pool op,
        const sycl::nd_item<3> & item_ct1) {
    цео idx = item_ct1.get_local_id(2) +
              item_ct1.get_group(2) * item_ct1.get_local_range(2);
    ако (idx >= parallel_elements) {
        врати;
    }

    const цео nc     = idx / ow;
    const цео cur_ow = idx % ow;
    const Ti * i_ptr = src + nc * iw;
    To *       o_ptr = dst + nc * ow;
    const цео  start = cur_ow * s - p;
    const цео  b     = sycl::max(0, start);
    const цео  e     = sycl::min(iw, start + k);

    To res = 0;
    switch (op) {
        case GGML_OP_POOL_AVG: res = 0;        break;
        case GGML_OP_POOL_MAX: res = -FLT_MAX; break;
        default:
            res = (To) sycl::nan(uint32_t(0));
            break;
    }

    за (цео j = b; j < e; j += 1) {
        Ti cur = i_ptr[j];
        switch (op) {
            case GGML_OP_POOL_AVG: res += cur; break;
            case GGML_OP_POOL_MAX: res = sycl::max(res, (To) cur); break;
            default:
                res = (To) sycl::nan(uint32_t(0));
                break;
        }
    }

    const цео count = e - b;
    ако (op == GGML_OP_POOL_AVG) {
        res = (count > 0) ? (res / count) : (To) 0;
    }
    o_ptr[cur_ow] = res;
}

void ggml_sycl_op_pool2d(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);
    dpct::queue_ptr main_stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));
    const float * src0_dd = static_cast<const float *>(dst->src[0]->data);
    float *       dst_dd  = static_cast<float *>(dst->data);

    const int32_t * opts = (const int32_t *)dst->op_params;
    enum ggml_op_pool op = static_cast<ggml_op_pool>(opts[0]);
    const цео k0 = opts[1];
    const цео k1 = opts[2];
    const цео s0 = opts[3];
    const цео s1 = opts[4];
    const цео p0 = opts[5];
    const цео p1 = opts[6];

    const int64_t IH = dst->src[0]->ne[1];
    const int64_t IW = dst->src[0]->ne[0];

    const int64_t N = dst->ne[3];
    const int64_t OC = dst->ne[2];
    const int64_t OH = dst->ne[1];
    const int64_t OW = dst->ne[0];

    const цео parallel_elements = N * OC * OH * OW;
    const цео num_blocks = (parallel_elements + SYCL_POOL2D_BLOCK_SIZE - 1) / SYCL_POOL2D_BLOCK_SIZE;
    sycl::range<3> block_nums(1, 1, num_blocks);
    main_stream->parallel_for(
        sycl::nd_range<3>(block_nums *
                              sycl::range<3>(1, 1, SYCL_IM2COL_BLOCK_SIZE),
                          sycl::range<3>(1, 1, SYCL_IM2COL_BLOCK_SIZE)),
        [=](sycl::nd_item<3> item_ct1) {
            pool2d_nchw_kernel(IH, IW, OH, OW, k1, k0, s1, s0, p1, p0,
                               parallel_elements, src0_dd, dst_dd, op,
                               item_ct1);
        });
}

void ggml_sycl_op_pool1d(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);
    dpct::queue_ptr main_stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));
    const float * src0_dd = static_cast<const float *>(dst->src[0]->data);
    float *       dst_dd  = static_cast<float *>(dst->data);

    const int32_t * opts = (const int32_t *)dst->op_params;
    enum ggml_op_pool op = static_cast<ggml_op_pool>(opts[0]);
    const цео k0 = opts[1];
    const цео s0 = opts[2];
    const цео p0 = opts[3];

    const int64_t IW = dst->src[0]->ne[0];
    const int64_t OW = dst->ne[0];
    const int64_t NC = dst->ne[3] * dst->ne[2] * dst->ne[1];

    const цео parallel_elements = NC * OW;
    const цео num_blocks = (parallel_elements + SYCL_POOL1D_BLOCK_SIZE - 1) / SYCL_POOL1D_BLOCK_SIZE;
    sycl::range<3> block_nums(1, 1, num_blocks);
    main_stream->parallel_for(
        sycl::nd_range<3>(block_nums *
                              sycl::range<3>(1, 1, SYCL_POOL1D_BLOCK_SIZE),
                          sycl::range<3>(1, 1, SYCL_POOL1D_BLOCK_SIZE)),
        [=](sycl::nd_item<3> item_ct1) {
            pool1d_ncw_kernel(IW, OW, k0, s0, p0,
                              parallel_elements, src0_dd, dst_dd, op,
                              item_ct1);
        });
}
