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

#অন্তর্ভাৱ "pool.hpp"
#অন্তর্ভাৱ <float.h>

template <typename Ti, typename To>
স্থির শূন্য pool2d_nchw_kernel(
        ধ্রুৱ পূর্ণাঙ্ক ih, ধ্রুৱ পূর্ণাঙ্ক iw, ধ্রুৱ পূর্ণাঙ্ক oh, ধ্রুৱ পূর্ণাঙ্ক ow,
        ধ্রুৱ পূর্ণাঙ্ক kh, ধ্রুৱ পূর্ণাঙ্ক kw, ধ্রুৱ পূর্ণাঙ্ক sh, ধ্রুৱ পূর্ণাঙ্ক sw,
        ধ্রুৱ পূর্ণাঙ্ক ph, ধ্রুৱ পূর্ণাঙ্ক pw, ধ্রুৱ পূর্ণাঙ্ক parallel_elements,
        ধ্রুৱ Ti* src, To* dst, ধ্রুৱ গণনা ggml_op_pool op,
        ধ্রুৱ sycl::nd_item<3> &item_ct1) {
        পূর্ণাঙ্ক idx = item_ct1.get_local_id(2) +
                  item_ct1.get_group(2) * item_ct1.get_local_range(2);
        যদি (idx >= parallel_elements) {
            ঘূৰাই;
        }

        ধ্রুৱ পূর্ণাঙ্ক I_HW = ih * iw;
        ধ্রুৱ পূর্ণাঙ্ক O_HW = oh * ow;
        ধ্রুৱ পূর্ণাঙ্ক nc = idx / O_HW;
        ধ্রুৱ পূর্ণাঙ্ক cur_oh = idx % O_HW / ow;
        ধ্রুৱ পূর্ণাঙ্ক cur_ow = idx % O_HW % ow;
        ধ্রুৱ Ti* i_ptr = src + nc * I_HW;
        To* o_ptr = dst + nc * O_HW;
        ধ্রুৱ পূর্ণাঙ্ক start_h = cur_oh * sh - ph;
        ধ্রুৱ পূর্ণাঙ্ক bh = sycl::max(0, start_h);
        ধ্রুৱ পূর্ণাঙ্ক eh = sycl::min(ih, start_h + kh);
        ধ্রুৱ পূর্ণাঙ্ক start_w = cur_ow * sw - pw;
        ধ্রুৱ পূর্ণাঙ্ক bw = sycl::max(0, start_w);
        ধ্রুৱ পূর্ণাঙ্ক ew = sycl::min(iw, start_w + kw);

        To res = 0;

        ৱিকল্প (op) {
            পক্ষ GGML_OP_POOL_AVG: res = 0; ভঙ্গ;
            পক্ষ GGML_OP_POOL_MAX: res = -FLT_MAX; ভঙ্গ;
            স্ৱভাৱ:
                res      = (To) sycl::nan(uint32_t(0));
                ভঙ্গ;
        }

        বাবে (পূর্ণাঙ্ক i = bh; i < eh; i += 1) {
            বাবে (পূর্ণাঙ্ক j = bw; j < ew; j += 1) {
                Ti cur = i_ptr[i * iw + j];
                ৱিকল্প (op) {
                    পক্ষ GGML_OP_POOL_AVG: res += (cur / (kh * kw)); ভঙ্গ;
                    পক্ষ GGML_OP_POOL_MAX: res = sycl::max(res, (To)cur); ভঙ্গ;
                    স্ৱভাৱ:
                        res = (To) sycl::nan(uint32_t(0));
                        ভঙ্গ;
                }
            }
        }
        o_ptr[cur_oh * ow + cur_ow] = res;
}

template <typename Ti, typename To>
স্থির শূন্য pool1d_ncw_kernel(
        ধ্রুৱ পূর্ণাঙ্ক iw, ধ্রুৱ পূর্ণাঙ্ক ow,
        ধ্রুৱ পূর্ণাঙ্ক k, ধ্রুৱ পূর্ণাঙ্ক s,
        ধ্রুৱ পূর্ণাঙ্ক p, ধ্রুৱ পূর্ণাঙ্ক parallel_elements,
        ধ্রুৱ Ti * src, To * dst, ধ্রুৱ গণনা ggml_op_pool op,
        ধ্রুৱ sycl::nd_item<3> & item_ct1) {
    পূর্ণাঙ্ক idx = item_ct1.get_local_id(2) +
              item_ct1.get_group(2) * item_ct1.get_local_range(2);
    যদি (idx >= parallel_elements) {
        ঘূৰাই;
    }

    ধ্রুৱ পূর্ণাঙ্ক nc     = idx / ow;
    ধ্রুৱ পূর্ণাঙ্ক cur_ow = idx % ow;
    ধ্রুৱ Ti * i_ptr = src + nc * iw;
    To *       o_ptr = dst + nc * ow;
    ধ্রুৱ পূর্ণাঙ্ক  start = cur_ow * s - p;
    ধ্রুৱ পূর্ণাঙ্ক  b     = sycl::max(0, start);
    ধ্রুৱ পূর্ণাঙ্ক  e     = sycl::min(iw, start + k);

    To res = 0;
    ৱিকল্প (op) {
        পক্ষ GGML_OP_POOL_AVG: res = 0;        ভঙ্গ;
        পক্ষ GGML_OP_POOL_MAX: res = -FLT_MAX; ভঙ্গ;
        স্ৱভাৱ:
            res = (To) sycl::nan(uint32_t(0));
            ভঙ্গ;
    }

    বাবে (পূর্ণাঙ্ক j = b; j < e; j += 1) {
        Ti cur = i_ptr[j];
        ৱিকল্প (op) {
            পক্ষ GGML_OP_POOL_AVG: res += cur; ভঙ্গ;
            পক্ষ GGML_OP_POOL_MAX: res = sycl::max(res, (To) cur); ভঙ্গ;
            স্ৱভাৱ:
                res = (To) sycl::nan(uint32_t(0));
                ভঙ্গ;
        }
    }

    ধ্রুৱ পূর্ণাঙ্ক count = e - b;
    যদি (op == GGML_OP_POOL_AVG) {
        res = (count > 0) ? (res / count) : (To) 0;
    }
    o_ptr[cur_ow] = res;
}

শূন্য ggml_sycl_op_pool2d(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);
    dpct::queue_ptr main_stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));
    ধ্রুৱ ভিন্ন * src0_dd = static_cast<ধ্রুৱ ভিন্ন *>(dst->src[0]->data);
    ভিন্ন *       dst_dd  = static_cast<ভিন্ন *>(dst->data);

    ধ্রুৱ int32_t * opts = (ধ্রুৱ int32_t *)dst->op_params;
    গণনা ggml_op_pool op = static_cast<ggml_op_pool>(opts[0]);
    ধ্রুৱ পূর্ণাঙ্ক k0 = opts[1];
    ধ্রুৱ পূর্ণাঙ্ক k1 = opts[2];
    ধ্রুৱ পূর্ণাঙ্ক s0 = opts[3];
    ধ্রুৱ পূর্ণাঙ্ক s1 = opts[4];
    ধ্রুৱ পূর্ণাঙ্ক p0 = opts[5];
    ধ্রুৱ পূর্ণাঙ্ক p1 = opts[6];

    ধ্রুৱ int64_t IH = dst->src[0]->ne[1];
    ধ্রুৱ int64_t IW = dst->src[0]->ne[0];

    ধ্রুৱ int64_t N = dst->ne[3];
    ধ্রুৱ int64_t OC = dst->ne[2];
    ধ্রুৱ int64_t OH = dst->ne[1];
    ধ্রুৱ int64_t OW = dst->ne[0];

    ধ্রুৱ পূর্ণাঙ্ক parallel_elements = N * OC * OH * OW;
    ধ্রুৱ পূর্ণাঙ্ক অঙ্ক_blocks = (parallel_elements + SYCL_POOL2D_BLOCK_SIZE - 1) / SYCL_POOL2D_BLOCK_SIZE;
    sycl::range<3> block_nums(1, 1, অঙ্ক_blocks);
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

শূন্য ggml_sycl_op_pool1d(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);
    dpct::queue_ptr main_stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));
    ধ্রুৱ ভিন্ন * src0_dd = static_cast<ধ্রুৱ ভিন্ন *>(dst->src[0]->data);
    ভিন্ন *       dst_dd  = static_cast<ভিন্ন *>(dst->data);

    ধ্রুৱ int32_t * opts = (ধ্রুৱ int32_t *)dst->op_params;
    গণনা ggml_op_pool op = static_cast<ggml_op_pool>(opts[0]);
    ধ্রুৱ পূর্ণাঙ্ক k0 = opts[1];
    ধ্রুৱ পূর্ণাঙ্ক s0 = opts[2];
    ধ্রুৱ পূর্ণাঙ্ক p0 = opts[3];

    ধ্রুৱ int64_t IW = dst->src[0]->ne[0];
    ধ্রুৱ int64_t OW = dst->ne[0];
    ধ্রুৱ int64_t NC = dst->ne[3] * dst->ne[2] * dst->ne[1];

    ধ্রুৱ পূর্ণাঙ্ক parallel_elements = NC * OW;
    ধ্রুৱ পূর্ণাঙ্ক অঙ্ক_blocks = (parallel_elements + SYCL_POOL1D_BLOCK_SIZE - 1) / SYCL_POOL1D_BLOCK_SIZE;
    sycl::range<3> block_nums(1, 1, অঙ্ক_blocks);
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
