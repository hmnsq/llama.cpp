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

#অন্তর্ভাব "pool.hpp"
#অন্তর্ভাব <float.h>

template <typename Ti, typename To>
স্থির শূন্য pool2d_nchw_kernel(
        ধ্রুবক পূর্ণসংখ্যা ih, ধ্রুবক পূর্ণসংখ্যা iw, ধ্রুবক পূর্ণসংখ্যা oh, ধ্রুবক পূর্ণসংখ্যা ow,
        ধ্রুবক পূর্ণসংখ্যা kh, ধ্রুবক পূর্ণসংখ্যা kw, ধ্রুবক পূর্ণসংখ্যা sh, ধ্রুবক পূর্ণসংখ্যা sw,
        ধ্রুবক পূর্ণসংখ্যা ph, ধ্রুবক পূর্ণসংখ্যা pw, ধ্রুবক পূর্ণসংখ্যা parallel_elements,
        ধ্রুবক Ti* src, To* dst, ধ্রুবক গণনা ggml_op_pool op,
        ধ্রুবক sycl::nd_item<3> &item_ct1) {
        পূর্ণসংখ্যা idx = item_ct1.get_local_id(2) +
                  item_ct1.get_group(2) * item_ct1.get_local_range(2);
        যদি (idx >= parallel_elements) {
            ফেরত;
        }

        ধ্রুবক পূর্ণসংখ্যা I_HW = ih * iw;
        ধ্রুবক পূর্ণসংখ্যা O_HW = oh * ow;
        ধ্রুবক পূর্ণসংখ্যা nc = idx / O_HW;
        ধ্রুবক পূর্ণসংখ্যা cur_oh = idx % O_HW / ow;
        ধ্রুবক পূর্ণসংখ্যা cur_ow = idx % O_HW % ow;
        ধ্রুবক Ti* i_ptr = src + nc * I_HW;
        To* o_ptr = dst + nc * O_HW;
        ধ্রুবক পূর্ণসংখ্যা start_h = cur_oh * sh - ph;
        ধ্রুবক পূর্ণসংখ্যা bh = sycl::max(0, start_h);
        ধ্রুবক পূর্ণসংখ্যা eh = sycl::min(ih, start_h + kh);
        ধ্রুবক পূর্ণসংখ্যা start_w = cur_ow * sw - pw;
        ধ্রুবক পূর্ণসংখ্যা bw = sycl::max(0, start_w);
        ধ্রুবক পূর্ণসংখ্যা ew = sycl::min(iw, start_w + kw);

        To res = 0;

        নির্বাচন (op) {
            ক্ষেত্র GGML_OP_POOL_AVG: res = 0; বিরতি;
            ক্ষেত্র GGML_OP_POOL_MAX: res = -FLT_MAX; বিরতি;
            পূর্বনির্ধারিত:
                res      = (To) sycl::nan(uint32_t(0));
                বিরতি;
        }

        জন্য (পূর্ণসংখ্যা i = bh; i < eh; i += 1) {
            জন্য (পূর্ণসংখ্যা j = bw; j < ew; j += 1) {
                Ti cur = i_ptr[i * iw + j];
                নির্বাচন (op) {
                    ক্ষেত্র GGML_OP_POOL_AVG: res += (cur / (kh * kw)); বিরতি;
                    ক্ষেত্র GGML_OP_POOL_MAX: res = sycl::max(res, (To)cur); বিরতি;
                    পূর্বনির্ধারিত:
                        res = (To) sycl::nan(uint32_t(0));
                        বিরতি;
                }
            }
        }
        o_ptr[cur_oh * ow + cur_ow] = res;
}

template <typename Ti, typename To>
স্থির শূন্য pool1d_ncw_kernel(
        ধ্রুবক পূর্ণসংখ্যা iw, ধ্রুবক পূর্ণসংখ্যা ow,
        ধ্রুবক পূর্ণসংখ্যা k, ধ্রুবক পূর্ণসংখ্যা s,
        ধ্রুবক পূর্ণসংখ্যা p, ধ্রুবক পূর্ণসংখ্যা parallel_elements,
        ধ্রুবক Ti * src, To * dst, ধ্রুবক গণনা ggml_op_pool op,
        ধ্রুবক sycl::nd_item<3> & item_ct1) {
    পূর্ণসংখ্যা idx = item_ct1.get_local_id(2) +
              item_ct1.get_group(2) * item_ct1.get_local_range(2);
    যদি (idx >= parallel_elements) {
        ফেরত;
    }

    ধ্রুবক পূর্ণসংখ্যা nc     = idx / ow;
    ধ্রুবক পূর্ণসংখ্যা cur_ow = idx % ow;
    ধ্রুবক Ti * i_ptr = src + nc * iw;
    To *       o_ptr = dst + nc * ow;
    ধ্রুবক পূর্ণসংখ্যা  start = cur_ow * s - p;
    ধ্রুবক পূর্ণসংখ্যা  b     = sycl::max(0, start);
    ধ্রুবক পূর্ণসংখ্যা  e     = sycl::min(iw, start + k);

    To res = 0;
    নির্বাচন (op) {
        ক্ষেত্র GGML_OP_POOL_AVG: res = 0;        বিরতি;
        ক্ষেত্র GGML_OP_POOL_MAX: res = -FLT_MAX; বিরতি;
        পূর্বনির্ধারিত:
            res = (To) sycl::nan(uint32_t(0));
            বিরতি;
    }

    জন্য (পূর্ণসংখ্যা j = b; j < e; j += 1) {
        Ti cur = i_ptr[j];
        নির্বাচন (op) {
            ক্ষেত্র GGML_OP_POOL_AVG: res += cur; বিরতি;
            ক্ষেত্র GGML_OP_POOL_MAX: res = sycl::max(res, (To) cur); বিরতি;
            পূর্বনির্ধারিত:
                res = (To) sycl::nan(uint32_t(0));
                বিরতি;
        }
    }

    ধ্রুবক পূর্ণসংখ্যা count = e - b;
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
    ধ্রুবক দশমিক * src0_dd = static_cast<ধ্রুবক দশমিক *>(dst->src[0]->তথ্য);
    দশমিক *       dst_dd  = static_cast<দশমিক *>(dst->তথ্য);

    ধ্রুবক int32_t * opts = (ধ্রুবক int32_t *)dst->op_params;
    গণনা ggml_op_pool op = static_cast<ggml_op_pool>(opts[0]);
    ধ্রুবক পূর্ণসংখ্যা k0 = opts[1];
    ধ্রুবক পূর্ণসংখ্যা k1 = opts[2];
    ধ্রুবক পূর্ণসংখ্যা s0 = opts[3];
    ধ্রুবক পূর্ণসংখ্যা s1 = opts[4];
    ধ্রুবক পূর্ণসংখ্যা p0 = opts[5];
    ধ্রুবক পূর্ণসংখ্যা p1 = opts[6];

    ধ্রুবক int64_t IH = dst->src[0]->ne[1];
    ধ্রুবক int64_t IW = dst->src[0]->ne[0];

    ধ্রুবক int64_t N = dst->ne[3];
    ধ্রুবক int64_t OC = dst->ne[2];
    ধ্রুবক int64_t OH = dst->ne[1];
    ধ্রুবক int64_t OW = dst->ne[0];

    ধ্রুবক পূর্ণসংখ্যা parallel_elements = N * OC * OH * OW;
    ধ্রুবক পূর্ণসংখ্যা অঙ্ক_blocks = (parallel_elements + SYCL_POOL2_D_BLOCK_আকার - 1) / SYCL_POOL2_D_BLOCK_আকার;
    sycl::range<3> block_nums(1, 1, অঙ্ক_blocks);
    main_stream->parallel_for(
        sycl::nd_range<3>(block_nums *
                              sycl::range<3>(1, 1, SYCL_IM2_COL_BLOCK_আকার),
                          sycl::range<3>(1, 1, SYCL_IM2_COL_BLOCK_আকার)),
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
    ধ্রুবক দশমিক * src0_dd = static_cast<ধ্রুবক দশমিক *>(dst->src[0]->তথ্য);
    দশমিক *       dst_dd  = static_cast<দশমিক *>(dst->তথ্য);

    ধ্রুবক int32_t * opts = (ধ্রুবক int32_t *)dst->op_params;
    গণনা ggml_op_pool op = static_cast<ggml_op_pool>(opts[0]);
    ধ্রুবক পূর্ণসংখ্যা k0 = opts[1];
    ধ্রুবক পূর্ণসংখ্যা s0 = opts[2];
    ধ্রুবক পূর্ণসংখ্যা p0 = opts[3];

    ধ্রুবক int64_t IW = dst->src[0]->ne[0];
    ধ্রুবক int64_t OW = dst->ne[0];
    ধ্রুবক int64_t NC = dst->ne[3] * dst->ne[2] * dst->ne[1];

    ধ্রুবক পূর্ণসংখ্যা parallel_elements = NC * OW;
    ধ্রুবক পূর্ণসংখ্যা অঙ্ক_blocks = (parallel_elements + SYCL_POOL1_D_BLOCK_আকার - 1) / SYCL_POOL1_D_BLOCK_আকার;
    sycl::range<3> block_nums(1, 1, অঙ্ক_blocks);
    main_stream->parallel_for(
        sycl::nd_range<3>(block_nums *
                              sycl::range<3>(1, 1, SYCL_POOL1_D_BLOCK_আকার),
                          sycl::range<3>(1, 1, SYCL_POOL1_D_BLOCK_আকার)),
        [=](sycl::nd_item<3> item_ct1) {
            pool1d_ncw_kernel(IW, OW, k0, s0, p0,
                              parallel_elements, src0_dd, dst_dd, op,
                              item_ct1);
        });
}
