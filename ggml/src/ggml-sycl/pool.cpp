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
        const ကိန်းပြည့် ih, const ကိန်းပြည့် iw, const ကိန်းပြည့် oh, const ကိန်းပြည့် ow,
        const ကိန်းပြည့် kh, const ကိန်းပြည့် kw, const ကိန်းပြည့် sh, const ကိန်းပြည့် sw,
        const ကိန်းပြည့် ph, const ကိန်းပြည့် pw, const ကိန်းပြည့် parallel_elements,
        const Ti* src, To* dst, const ga_nanaa ggml_op_pool op,
        const sycl::nd_item<3> &item_ct1) {
        ကိန်းပြည့် idx = item_ct1.get_local_id(2) +
                  item_ct1.get_group(2) * item_ct1.get_local_range(2);
        အကယ်၍ (idx >= parallel_elements) {
            ပြန်ပေး;
        }

        const ကိန်းပြည့် I_HW = ih * iw;
        const ကိန်းပြည့် O_HW = oh * ow;
        const ကိန်းပြည့် nc = idx / O_HW;
        const ကိန်းပြည့် cur_oh = idx % O_HW / ow;
        const ကိန်းပြည့် cur_ow = idx % O_HW % ow;
        const Ti* i_ptr = src + nc * I_HW;
        To* o_ptr = dst + nc * O_HW;
        const ကိန်းပြည့် start_h = cur_oh * sh - ph;
        const ကိန်းပြည့် bh = sycl::max(0, start_h);
        const ကိန်းပြည့် eh = sycl::min(ih, start_h + kh);
        const ကိန်းပြည့် start_w = cur_ow * sw - pw;
        const ကိန်းပြည့် bw = sycl::max(0, start_w);
        const ကိန်းပြည့် ew = sycl::min(iw, start_w + kw);

        To res = 0;

        switch (op) {
            pak_sa GGML_OP_POOL_AVG: res = 0; bha_nga;
            pak_sa GGML_OP_POOL_MAX: res = -FLT_MAX; bha_nga;
            default:
                res      = (To) sycl::nan(uint32_t(0));
                bha_nga;
        }

        အတွက် (ကိန်းပြည့် i = bh; i < eh; i += 1) {
            အတွက် (ကိန်းပြည့် j = bw; j < ew; j += 1) {
                Ti cur = i_ptr[i * iw + j];
                switch (op) {
                    pak_sa GGML_OP_POOL_AVG: res += (cur / (kh * kw)); bha_nga;
                    pak_sa GGML_OP_POOL_MAX: res = sycl::max(res, (To)cur); bha_nga;
                    default:
                        res = (To) sycl::nan(uint32_t(0));
                        bha_nga;
                }
            }
        }
        o_ptr[cur_oh * ow + cur_ow] = res;
}

template <typename Ti, typename To>
static void pool1d_ncw_kernel(
        const ကိန်းပြည့် iw, const ကိန်းပြည့် ow,
        const ကိန်းပြည့် k, const ကိန်းပြည့် s,
        const ကိန်းပြည့် p, const ကိန်းပြည့် parallel_elements,
        const Ti * src, To * dst, const ga_nanaa ggml_op_pool op,
        const sycl::nd_item<3> & item_ct1) {
    ကိန်းပြည့် idx = item_ct1.get_local_id(2) +
              item_ct1.get_group(2) * item_ct1.get_local_range(2);
    အကယ်၍ (idx >= parallel_elements) {
        ပြန်ပေး;
    }

    const ကိန်းပြည့် nc     = idx / ow;
    const ကိန်းပြည့် cur_ow = idx % ow;
    const Ti * i_ptr = src + nc * iw;
    To *       o_ptr = dst + nc * ow;
    const ကိန်းပြည့်  start = cur_ow * s - p;
    const ကိန်းပြည့်  b     = sycl::max(0, start);
    const ကိန်းပြည့်  e     = sycl::min(iw, start + k);

    To res = 0;
    switch (op) {
        pak_sa GGML_OP_POOL_AVG: res = 0;        bha_nga;
        pak_sa GGML_OP_POOL_MAX: res = -FLT_MAX; bha_nga;
        default:
            res = (To) sycl::nan(uint32_t(0));
            bha_nga;
    }

    အတွက် (ကိန်းပြည့် j = b; j < e; j += 1) {
        Ti cur = i_ptr[j];
        switch (op) {
            pak_sa GGML_OP_POOL_AVG: res += cur; bha_nga;
            pak_sa GGML_OP_POOL_MAX: res = sycl::max(res, (To) cur); bha_nga;
            default:
                res = (To) sycl::nan(uint32_t(0));
                bha_nga;
        }
    }

    const ကိန်းပြည့် count = e - b;
    အကယ်၍ (op == GGML_OP_POOL_AVG) {
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
    ga_nanaa ggml_op_pool op = static_cast<ggml_op_pool>(opts[0]);
    const ကိန်းပြည့် k0 = opts[1];
    const ကိန်းပြည့် k1 = opts[2];
    const ကိန်းပြည့် s0 = opts[3];
    const ကိန်းပြည့် s1 = opts[4];
    const ကိန်းပြည့် p0 = opts[5];
    const ကိန်းပြည့် p1 = opts[6];

    const int64_t IH = dst->src[0]->ne[1];
    const int64_t IW = dst->src[0]->ne[0];

    const int64_t N = dst->ne[3];
    const int64_t OC = dst->ne[2];
    const int64_t OH = dst->ne[1];
    const int64_t OW = dst->ne[0];

    const ကိန်းပြည့် parallel_elements = N * OC * OH * OW;
    const ကိန်းပြည့် num_blocks = (parallel_elements + SYCL_POOL2D_BLOCK_SIZE - 1) / SYCL_POOL2D_BLOCK_SIZE;
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
    ga_nanaa ggml_op_pool op = static_cast<ggml_op_pool>(opts[0]);
    const ကိန်းပြည့် k0 = opts[1];
    const ကိန်းပြည့် s0 = opts[2];
    const ကိန်းပြည့် p0 = opts[3];

    const int64_t IW = dst->src[0]->ne[0];
    const int64_t OW = dst->ne[0];
    const int64_t NC = dst->ne[3] * dst->ne[2] * dst->ne[1];

    const ကိန်းပြည့် parallel_elements = NC * OW;
    const ကိန်းပြည့် num_blocks = (parallel_elements + SYCL_POOL1D_BLOCK_SIZE - 1) / SYCL_POOL1D_BLOCK_SIZE;
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
