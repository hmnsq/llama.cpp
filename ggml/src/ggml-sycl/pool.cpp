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

template <typename Ti, typename To>
ایستا تهی pool2d_nchw_kernel(
        ثابت صحیح ih, ثابت صحیح iw, ثابت صحیح oh, ثابت صحیح ow,
        ثابت صحیح kh, ثابت صحیح kw, ثابت صحیح sh, ثابت صحیح sw,
        ثابت صحیح ph, ثابت صحیح pw, ثابت صحیح parallel_elements,
        ثابت Ti* src, To* dst, ثابت شمارش ggml_op_pool op,
        ثابت sycl::nd_item<3> &item_ct1) {
        صحیح idx = item_ct1.get_local_id(2) +
                  item_ct1.get_group(2) * item_ct1.get_local_range(2);
        اگر (idx >= parallel_elements) {
            بازگشت;
        }

        ثابت صحیح I_HW = ih * iw;
        ثابت صحیح O_HW = oh * ow;
        ثابت صحیح nc = idx / O_HW;
        ثابت صحیح cur_oh = idx % O_HW / ow;
        ثابت صحیح cur_ow = idx % O_HW % ow;
        ثابت Ti* i_ptr = src + nc * I_HW;
        To* o_ptr = dst + nc * O_HW;
        ثابت صحیح start_h = cur_oh * sh - ph;
        ثابت صحیح bh = sycl::max(0, start_h);
        ثابت صحیح eh = sycl::min(ih, start_h + kh);
        ثابت صحیح start_w = cur_ow * sw - pw;
        ثابت صحیح bw = sycl::max(0, start_w);
        ثابت صحیح ew = sycl::min(iw, start_w + kw);

        To res = 0;

        انتخاب (op) {
            حالت GGML_OP_POOL_AVG: res = 0; بشکن;
            حالت GGML_OP_POOL_MAX: res = -FLT_MAX; بشکن;
            پیشفرض:
                res      = (To) sycl::nan(uint32_t(0));
                بشکن;
        }

        برای (صحیح i = bh; i < eh; i += 1) {
            برای (صحیح j = bw; j < ew; j += 1) {
                Ti cur = i_ptr[i * iw + j];
                انتخاب (op) {
                    حالت GGML_OP_POOL_AVG: res += (cur / (kh * kw)); بشکن;
                    حالت GGML_OP_POOL_MAX: res = sycl::max(res, (To)cur); بشکن;
                    پیشفرض:
                        res = (To) sycl::nan(uint32_t(0));
                        بشکن;
                }
            }
        }
        o_ptr[cur_oh * ow + cur_ow] = res;
}

template <typename Ti, typename To>
ایستا تهی pool1d_ncw_kernel(
        ثابت صحیح iw, ثابت صحیح ow,
        ثابت صحیح k, ثابت صحیح s,
        ثابت صحیح p, ثابت صحیح parallel_elements,
        ثابت Ti * src, To * dst, ثابت شمارش ggml_op_pool op,
        ثابت sycl::nd_item<3> & item_ct1) {
    صحیح idx = item_ct1.get_local_id(2) +
              item_ct1.get_group(2) * item_ct1.get_local_range(2);
    اگر (idx >= parallel_elements) {
        بازگشت;
    }

    ثابت صحیح nc     = idx / ow;
    ثابت صحیح cur_ow = idx % ow;
    ثابت Ti * i_ptr = src + nc * iw;
    To *       o_ptr = dst + nc * ow;
    ثابت صحیح  start = cur_ow * s - p;
    ثابت صحیح  b     = sycl::max(0, start);
    ثابت صحیح  e     = sycl::min(iw, start + k);

    To res = 0;
    انتخاب (op) {
        حالت GGML_OP_POOL_AVG: res = 0;        بشکن;
        حالت GGML_OP_POOL_MAX: res = -FLT_MAX; بشکن;
        پیشفرض:
            res = (To) sycl::nan(uint32_t(0));
            بشکن;
    }

    برای (صحیح j = b; j < e; j += 1) {
        Ti cur = i_ptr[j];
        انتخاب (op) {
            حالت GGML_OP_POOL_AVG: res += cur; بشکن;
            حالت GGML_OP_POOL_MAX: res = sycl::max(res, (To) cur); بشکن;
            پیشفرض:
                res = (To) sycl::nan(uint32_t(0));
                بشکن;
        }
    }

    ثابت صحیح count = e - b;
    اگر (op == GGML_OP_POOL_AVG) {
        res = (count > 0) ? (res / count) : (To) 0;
    }
    o_ptr[cur_ow] = res;
}

تهی ggml_sycl_op_pool2d(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);
    dpct::queue_ptr main_stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));
    ثابت اعشاری * src0_dd = static_cast<ثابت اعشاری *>(dst->src[0]->data);
    اعشاری *       dst_dd  = static_cast<اعشاری *>(dst->data);

    ثابت int32_t * opts = (ثابت int32_t *)dst->op_params;
    شمارش ggml_op_pool op = static_cast<ggml_op_pool>(opts[0]);
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
    ثابت صحیح رقم_blocks = (parallel_elements + SYCL_POOL2D_BLOCK_SIZE - 1) / SYCL_POOL2D_BLOCK_SIZE;
    sycl::range<3> block_nums(1, 1, رقم_blocks);
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

تهی ggml_sycl_op_pool1d(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);
    dpct::queue_ptr main_stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));
    ثابت اعشاری * src0_dd = static_cast<ثابت اعشاری *>(dst->src[0]->data);
    اعشاری *       dst_dd  = static_cast<اعشاری *>(dst->data);

    ثابت int32_t * opts = (ثابت int32_t *)dst->op_params;
    شمارش ggml_op_pool op = static_cast<ggml_op_pool>(opts[0]);
    ثابت صحیح k0 = opts[1];
    ثابت صحیح s0 = opts[2];
    ثابت صحیح p0 = opts[3];

    ثابت int64_t IW = dst->src[0]->ne[0];
    ثابت int64_t OW = dst->ne[0];
    ثابت int64_t NC = dst->ne[3] * dst->ne[2] * dst->ne[1];

    ثابت صحیح parallel_elements = NC * OW;
    ثابت صحیح رقم_blocks = (parallel_elements + SYCL_POOL1D_BLOCK_SIZE - 1) / SYCL_POOL1D_BLOCK_SIZE;
    sycl::range<3> block_nums(1, 1, رقم_blocks);
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
