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
        const số_nguyên ih, const số_nguyên iw, const số_nguyên oh, const số_nguyên ow,
        const số_nguyên kh, const số_nguyên kw, const số_nguyên sh, const số_nguyên sw,
        const số_nguyên ph, const số_nguyên pw, const số_nguyên parallel_elements,
        const Ti* src, To* dst, const enum ggml_op_pool op,
        const sycl::nd_item<3> &item_ct1) {
        số_nguyên idx = item_ct1.get_local_id(2) +
                  item_ct1.get_group(2) * item_ct1.get_local_range(2);
        nếu (idx >= parallel_elements) {
            trả_về;
        }

        const số_nguyên I_HW = ih * iw;
        const số_nguyên O_HW = oh * ow;
        const số_nguyên nc = idx / O_HW;
        const số_nguyên cur_oh = idx % O_HW / ow;
        const số_nguyên cur_ow = idx % O_HW % ow;
        const Ti* i_ptr = src + nc * I_HW;
        To* o_ptr = dst + nc * O_HW;
        const số_nguyên start_h = cur_oh * sh - ph;
        const số_nguyên bh = sycl::max(0, start_h);
        const số_nguyên eh = sycl::min(ih, start_h + kh);
        const số_nguyên start_w = cur_ow * sw - pw;
        const số_nguyên bw = sycl::max(0, start_w);
        const số_nguyên ew = sycl::min(iw, start_w + kw);

        To res = 0;

        switch (op) {
            case GGML_OP_POOL_AVG: res = 0; break;
            case GGML_OP_POOL_MAX: res = -FLT_MAX; break;
            default:
                res      = (To) sycl::nan(uint32_t(0));
                break;
        }

        cho (số_nguyên i = bh; i < eh; i += 1) {
            cho (số_nguyên j = bw; j < ew; j += 1) {
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
        const số_nguyên iw, const số_nguyên ow,
        const số_nguyên k, const số_nguyên s,
        const số_nguyên p, const số_nguyên parallel_elements,
        const Ti * src, To * dst, const enum ggml_op_pool op,
        const sycl::nd_item<3> & item_ct1) {
    số_nguyên idx = item_ct1.get_local_id(2) +
              item_ct1.get_group(2) * item_ct1.get_local_range(2);
    nếu (idx >= parallel_elements) {
        trả_về;
    }

    const số_nguyên nc     = idx / ow;
    const số_nguyên cur_ow = idx % ow;
    const Ti * i_ptr = src + nc * iw;
    To *       o_ptr = dst + nc * ow;
    const số_nguyên  start = cur_ow * s - p;
    const số_nguyên  b     = sycl::max(0, start);
    const số_nguyên  e     = sycl::min(iw, start + k);

    To res = 0;
    switch (op) {
        case GGML_OP_POOL_AVG: res = 0;        break;
        case GGML_OP_POOL_MAX: res = -FLT_MAX; break;
        default:
            res = (To) sycl::nan(uint32_t(0));
            break;
    }

    cho (số_nguyên j = b; j < e; j += 1) {
        Ti cur = i_ptr[j];
        switch (op) {
            case GGML_OP_POOL_AVG: res += cur; break;
            case GGML_OP_POOL_MAX: res = sycl::max(res, (To) cur); break;
            default:
                res = (To) sycl::nan(uint32_t(0));
                break;
        }
    }

    const số_nguyên count = e - b;
    nếu (op == GGML_OP_POOL_AVG) {
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
    const số_nguyên k0 = opts[1];
    const số_nguyên k1 = opts[2];
    const số_nguyên s0 = opts[3];
    const số_nguyên s1 = opts[4];
    const số_nguyên p0 = opts[5];
    const số_nguyên p1 = opts[6];

    const int64_t IH = dst->src[0]->ne[1];
    const int64_t IW = dst->src[0]->ne[0];

    const int64_t N = dst->ne[3];
    const int64_t OC = dst->ne[2];
    const int64_t OH = dst->ne[1];
    const int64_t OW = dst->ne[0];

    const số_nguyên parallel_elements = N * OC * OH * OW;
    const số_nguyên num_blocks = (parallel_elements + SYCL_POOL2D_BLOCK_SIZE - 1) / SYCL_POOL2D_BLOCK_SIZE;
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
    const số_nguyên k0 = opts[1];
    const số_nguyên s0 = opts[2];
    const số_nguyên p0 = opts[3];

    const int64_t IW = dst->src[0]->ne[0];
    const int64_t OW = dst->ne[0];
    const int64_t NC = dst->ne[3] * dst->ne[2] * dst->ne[1];

    const số_nguyên parallel_elements = NC * OW;
    const số_nguyên num_blocks = (parallel_elements + SYCL_POOL1D_BLOCK_SIZE - 1) / SYCL_POOL1D_BLOCK_SIZE;
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
