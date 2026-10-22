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
        const පූර්ණ_සංඛ්‍යා ih, const පූර්ණ_සංඛ්‍යා iw, const පූර්ණ_සංඛ්‍යා oh, const පූර්ණ_සංඛ්‍යා ow,
        const පූර්ණ_සංඛ්‍යා kh, const පූර්ණ_සංඛ්‍යා kw, const පූර්ණ_සංඛ්‍යා sh, const පූර්ණ_සංඛ්‍යා sw,
        const පූර්ණ_සංඛ්‍යා ph, const පූර්ණ_සංඛ්‍යා pw, const පූර්ණ_සංඛ්‍යා parallel_elements,
        const Ti* src, To* dst, const ga_nanaa ggml_op_pool op,
        const sycl::nd_item<3> &item_ct1) {
        පූර්ණ_සංඛ්‍යා idx = item_ct1.get_local_id(2) +
                  item_ct1.get_group(2) * item_ct1.get_local_range(2);
        නම් (idx >= parallel_elements) {
            ආපසු;
        }

        const පූර්ණ_සංඛ්‍යා I_HW = ih * iw;
        const පූර්ණ_සංඛ්‍යා O_HW = oh * ow;
        const පූර්ණ_සංඛ්‍යා nc = idx / O_HW;
        const පූර්ණ_සංඛ්‍යා cur_oh = idx % O_HW / ow;
        const පූර්ණ_සංඛ්‍යා cur_ow = idx % O_HW % ow;
        const Ti* i_ptr = src + nc * I_HW;
        To* o_ptr = dst + nc * O_HW;
        const පූර්ණ_සංඛ්‍යා start_h = cur_oh * sh - ph;
        const පූර්ණ_සංඛ්‍යා bh = sycl::max(0, start_h);
        const පූර්ණ_සංඛ්‍යා eh = sycl::min(ih, start_h + kh);
        const පූර්ණ_සංඛ්‍යා start_w = cur_ow * sw - pw;
        const පූර්ණ_සංඛ්‍යා bw = sycl::max(0, start_w);
        const පූර්ණ_සංඛ්‍යා ew = sycl::min(iw, start_w + kw);

        To res = 0;

        switch (op) {
            pak_sa GGML_OP_POOL_AVG: res = 0; bha_nga;
            pak_sa GGML_OP_POOL_MAX: res = -FLT_MAX; bha_nga;
            default:
                res      = (To) sycl::nan(uint32_t(0));
                bha_nga;
        }

        සඳහා (පූර්ණ_සංඛ්‍යා i = bh; i < eh; i += 1) {
            සඳහා (පූර්ණ_සංඛ්‍යා j = bw; j < ew; j += 1) {
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
        const පූර්ණ_සංඛ්‍යා iw, const පූර්ණ_සංඛ්‍යා ow,
        const පූර්ණ_සංඛ්‍යා k, const පූර්ණ_සංඛ්‍යා s,
        const පූර්ණ_සංඛ්‍යා p, const පූර්ණ_සංඛ්‍යා parallel_elements,
        const Ti * src, To * dst, const ga_nanaa ggml_op_pool op,
        const sycl::nd_item<3> & item_ct1) {
    පූර්ණ_සංඛ්‍යා idx = item_ct1.get_local_id(2) +
              item_ct1.get_group(2) * item_ct1.get_local_range(2);
    නම් (idx >= parallel_elements) {
        ආපසු;
    }

    const පූර්ණ_සංඛ්‍යා nc     = idx / ow;
    const පූර්ණ_සංඛ්‍යා cur_ow = idx % ow;
    const Ti * i_ptr = src + nc * iw;
    To *       o_ptr = dst + nc * ow;
    const පූර්ණ_සංඛ්‍යා  start = cur_ow * s - p;
    const පූර්ණ_සංඛ්‍යා  b     = sycl::max(0, start);
    const පූර්ණ_සංඛ්‍යා  e     = sycl::min(iw, start + k);

    To res = 0;
    switch (op) {
        pak_sa GGML_OP_POOL_AVG: res = 0;        bha_nga;
        pak_sa GGML_OP_POOL_MAX: res = -FLT_MAX; bha_nga;
        default:
            res = (To) sycl::nan(uint32_t(0));
            bha_nga;
    }

    සඳහා (පූර්ණ_සංඛ්‍යා j = b; j < e; j += 1) {
        Ti cur = i_ptr[j];
        switch (op) {
            pak_sa GGML_OP_POOL_AVG: res += cur; bha_nga;
            pak_sa GGML_OP_POOL_MAX: res = sycl::max(res, (To) cur); bha_nga;
            default:
                res = (To) sycl::nan(uint32_t(0));
                bha_nga;
        }
    }

    const පූර්ණ_සංඛ්‍යා count = e - b;
    නම් (op == GGML_OP_POOL_AVG) {
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
    const පූර්ණ_සංඛ්‍යා k0 = opts[1];
    const පූර්ණ_සංඛ්‍යා k1 = opts[2];
    const පූර්ණ_සංඛ්‍යා s0 = opts[3];
    const පූර්ණ_සංඛ්‍යා s1 = opts[4];
    const පූර්ණ_සංඛ්‍යා p0 = opts[5];
    const පූර්ණ_සංඛ්‍යා p1 = opts[6];

    const int64_t IH = dst->src[0]->ne[1];
    const int64_t IW = dst->src[0]->ne[0];

    const int64_t N = dst->ne[3];
    const int64_t OC = dst->ne[2];
    const int64_t OH = dst->ne[1];
    const int64_t OW = dst->ne[0];

    const පූර්ණ_සංඛ්‍යා parallel_elements = N * OC * OH * OW;
    const පූර්ණ_සංඛ්‍යා num_blocks = (parallel_elements + SYCL_POOL2D_BLOCK_SIZE - 1) / SYCL_POOL2D_BLOCK_SIZE;
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
    const පූර්ණ_සංඛ්‍යා k0 = opts[1];
    const පූර්ණ_සංඛ්‍යා s0 = opts[2];
    const පූර්ණ_සංඛ්‍යා p0 = opts[3];

    const int64_t IW = dst->src[0]->ne[0];
    const int64_t OW = dst->ne[0];
    const int64_t NC = dst->ne[3] * dst->ne[2] * dst->ne[1];

    const පූර්ණ_සංඛ්‍යා parallel_elements = NC * OW;
    const පූර්ණ_සංඛ්‍යා num_blocks = (parallel_elements + SYCL_POOL1D_BLOCK_SIZE - 1) / SYCL_POOL1D_BLOCK_SIZE;
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
