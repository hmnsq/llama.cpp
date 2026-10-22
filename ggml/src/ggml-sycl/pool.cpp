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

#अन्तर्भाव "pool.hpp"
#अन्तर्भाव <float.h>

ढाँचा <typename Ti, typename To>
स्थिर शून्य pool2d_nchw_kernel(
        अचर पूर्णाङ्क ih, अचर पूर्णाङ्क iw, अचर पूर्णाङ्क oh, अचर पूर्णाङ्क ow,
        अचर पूर्णाङ्क kh, अचर पूर्णाङ्क kw, अचर पूर्णाङ्क sh, अचर पूर्णाङ्क sw,
        अचर पूर्णाङ्क ph, अचर पूर्णाङ्क pw, अचर पूर्णाङ्क parallel_elements,
        अचर Ti* src, To* dst, अचर गणना ggml_op_pool op,
        अचर sycl::nd_item<3> &item_ct1) {
        पूर्णाङ्क idx = item_ct1.get_local_id(2) +
                  item_ct1.get_group(2) * item_ct1.get_local_range(2);
        यदि (idx >= parallel_elements) {
            फर्काऊ;
        }

        अचर पूर्णाङ्क I_HW = ih * iw;
        अचर पूर्णाङ्क O_HW = oh * ow;
        अचर पूर्णाङ्क nc = idx / O_HW;
        अचर पूर्णाङ्क cur_oh = idx % O_HW / ow;
        अचर पूर्णाङ्क cur_ow = idx % O_HW % ow;
        अचर Ti* i_ptr = src + nc * I_HW;
        To* o_ptr = dst + nc * O_HW;
        अचर पूर्णाङ्क start_h = cur_oh * sh - ph;
        अचर पूर्णाङ्क bh = sycl::max(0, start_h);
        अचर पूर्णाङ्क eh = sycl::min(ih, start_h + kh);
        अचर पूर्णाङ्क start_w = cur_ow * sw - pw;
        अचर पूर्णाङ्क bw = sycl::max(0, start_w);
        अचर पूर्णाङ्क ew = sycl::min(iw, start_w + kw);

        To res = 0;

        विकल्प (op) {
            पक्ष GGML_OP_POOL_AVG: res = 0; रोक;
            पक्ष GGML_OP_POOL_MAX: res = -भग्न_उच्च; रोक;
            स्वभाव:
                res      = (To) sycl::nan(uint32_t(0));
                रोक;
        }

        लागि (पूर्णाङ्क i = bh; i < eh; i += 1) {
            लागि (पूर्णाङ्क j = bw; j < ew; j += 1) {
                Ti cur = i_ptr[i * iw + j];
                विकल्प (op) {
                    पक्ष GGML_OP_POOL_AVG: res += (cur / (kh * kw)); रोक;
                    पक्ष GGML_OP_POOL_MAX: res = sycl::max(res, (To)cur); रोक;
                    स्वभाव:
                        res = (To) sycl::nan(uint32_t(0));
                        रोक;
                }
            }
        }
        o_ptr[cur_oh * ow + cur_ow] = res;
}

ढाँचा <typename Ti, typename To>
स्थिर शून्य pool1d_ncw_kernel(
        अचर पूर्णाङ्क iw, अचर पूर्णाङ्क ow,
        अचर पूर्णाङ्क k, अचर पूर्णाङ्क s,
        अचर पूर्णाङ्क p, अचर पूर्णाङ्क parallel_elements,
        अचर Ti * src, To * dst, अचर गणना ggml_op_pool op,
        अचर sycl::nd_item<3> & item_ct1) {
    पूर्णाङ्क idx = item_ct1.get_local_id(2) +
              item_ct1.get_group(2) * item_ct1.get_local_range(2);
    यदि (idx >= parallel_elements) {
        फर्काऊ;
    }

    अचर पूर्णाङ्क nc     = idx / ow;
    अचर पूर्णाङ्क cur_ow = idx % ow;
    अचर Ti * i_ptr = src + nc * iw;
    To *       o_ptr = dst + nc * ow;
    अचर पूर्णाङ्क  start = cur_ow * s - p;
    अचर पूर्णाङ्क  b     = sycl::max(0, start);
    अचर पूर्णाङ्क  e     = sycl::min(iw, start + k);

    To res = 0;
    विकल्प (op) {
        पक्ष GGML_OP_POOL_AVG: res = 0;        रोक;
        पक्ष GGML_OP_POOL_MAX: res = -भग्न_उच्च; रोक;
        स्वभाव:
            res = (To) sycl::nan(uint32_t(0));
            रोक;
    }

    लागि (पूर्णाङ्क j = b; j < e; j += 1) {
        Ti cur = i_ptr[j];
        विकल्प (op) {
            पक्ष GGML_OP_POOL_AVG: res += cur; रोक;
            पक्ष GGML_OP_POOL_MAX: res = sycl::max(res, (To) cur); रोक;
            स्वभाव:
                res = (To) sycl::nan(uint32_t(0));
                रोक;
        }
    }

    अचर पूर्णाङ्क count = e - b;
    यदि (op == GGML_OP_POOL_AVG) {
        res = (count > 0) ? (res / count) : (To) 0;
    }
    o_ptr[cur_ow] = res;
}

शून्य ggml_sycl_op_pool2d(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);
    dpct::queue_ptr main_stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));
    अचर भिन्न * src0_dd = static_cast<अचर भिन्न *>(dst->src[0]->data);
    भिन्न *       dst_dd  = static_cast<भिन्न *>(dst->data);

    अचर int32_t * opts = (अचर int32_t *)dst->op_params;
    गणना ggml_op_pool op = static_cast<ggml_op_pool>(opts[0]);
    अचर पूर्णाङ्क k0 = opts[1];
    अचर पूर्णाङ्क k1 = opts[2];
    अचर पूर्णाङ्क s0 = opts[3];
    अचर पूर्णाङ्क s1 = opts[4];
    अचर पूर्णाङ्क p0 = opts[5];
    अचर पूर्णाङ्क p1 = opts[6];

    अचर int64_t IH = dst->src[0]->ne[1];
    अचर int64_t IW = dst->src[0]->ne[0];

    अचर int64_t N = dst->ne[3];
    अचर int64_t OC = dst->ne[2];
    अचर int64_t OH = dst->ne[1];
    अचर int64_t OW = dst->ne[0];

    अचर पूर्णाङ्क parallel_elements = N * OC * OH * OW;
    अचर पूर्णाङ्क अङ्क_blocks = (parallel_elements + SYCL_POOL2D_BLOCK_SIZE - 1) / SYCL_POOL2D_BLOCK_SIZE;
    sycl::range<3> block_nums(1, 1, अङ्क_blocks);
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

शून्य ggml_sycl_op_pool1d(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);
    dpct::queue_ptr main_stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));
    अचर भिन्न * src0_dd = static_cast<अचर भिन्न *>(dst->src[0]->data);
    भिन्न *       dst_dd  = static_cast<भिन्न *>(dst->data);

    अचर int32_t * opts = (अचर int32_t *)dst->op_params;
    गणना ggml_op_pool op = static_cast<ggml_op_pool>(opts[0]);
    अचर पूर्णाङ्क k0 = opts[1];
    अचर पूर्णाङ्क s0 = opts[2];
    अचर पूर्णाङ्क p0 = opts[3];

    अचर int64_t IW = dst->src[0]->ne[0];
    अचर int64_t OW = dst->ne[0];
    अचर int64_t NC = dst->ne[3] * dst->ne[2] * dst->ne[1];

    अचर पूर्णाङ्क parallel_elements = NC * OW;
    अचर पूर्णाङ्क अङ्क_blocks = (parallel_elements + SYCL_POOL1D_BLOCK_SIZE - 1) / SYCL_POOL1D_BLOCK_SIZE;
    sycl::range<3> block_nums(1, 1, अङ्क_blocks);
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
