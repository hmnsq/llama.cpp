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

#समावेश "pool.hpp"
#समावेश <float.h>

ढाँचा <typename Ti, typename To>
अटल व्योम pool2d_nchw_kernel(
        स्थिर पूर्णांक ih, स्थिर पूर्णांक iw, स्थिर पूर्णांक oh, स्थिर पूर्णांक ow,
        स्थिर पूर्णांक kh, स्थिर पूर्णांक kw, स्थिर पूर्णांक sh, स्थिर पूर्णांक sw,
        स्थिर पूर्णांक ph, स्थिर पूर्णांक pw, स्थिर पूर्णांक parallel_elements,
        स्थिर Ti* src, To* dst, स्थिर क्रमागत ggml_op_pool op,
        स्थिर sycl::nd_item<3> &item_ct1) {
        पूर्णांक idx = item_ct1.get_local_id(2) +
                  item_ct1.get_group(2) * item_ct1.get_local_range(2);
        अगर (idx >= parallel_elements) {
            वापस;
        }

        स्थिर पूर्णांक I_HW = ih * iw;
        स्थिर पूर्णांक O_HW = oh * ow;
        स्थिर पूर्णांक nc = idx / O_HW;
        स्थिर पूर्णांक cur_oh = idx % O_HW / ow;
        स्थिर पूर्णांक cur_ow = idx % O_HW % ow;
        स्थिर Ti* i_ptr = src + nc * I_HW;
        To* o_ptr = dst + nc * O_HW;
        स्थिर पूर्णांक start_h = cur_oh * sh - ph;
        स्थिर पूर्णांक bh = sycl::max(0, start_h);
        स्थिर पूर्णांक eh = sycl::min(ih, start_h + kh);
        स्थिर पूर्णांक start_w = cur_ow * sw - pw;
        स्थिर पूर्णांक bw = sycl::max(0, start_w);
        स्थिर पूर्णांक ew = sycl::min(iw, start_w + kw);

        To res = 0;

        चयन (op) {
            हाल GGML_OP_POOL_AVG: res = 0; अवरोध;
            हाल GGML_OP_POOL_MAX: res = -भग्न_उच्च; अवरोध;
            शेष:
                res      = (To) sycl::nan(uint32_t(0));
                अवरोध;
        }

        क्रम (पूर्णांक i = bh; i < eh; i += 1) {
            क्रम (पूर्णांक j = bw; j < ew; j += 1) {
                Ti cur = i_ptr[i * iw + j];
                चयन (op) {
                    हाल GGML_OP_POOL_AVG: res += (cur / (kh * kw)); अवरोध;
                    हाल GGML_OP_POOL_MAX: res = sycl::max(res, (To)cur); अवरोध;
                    शेष:
                        res = (To) sycl::nan(uint32_t(0));
                        अवरोध;
                }
            }
        }
        o_ptr[cur_oh * ow + cur_ow] = res;
}

ढाँचा <typename Ti, typename To>
अटल व्योम pool1d_ncw_kernel(
        स्थिर पूर्णांक iw, स्थिर पूर्णांक ow,
        स्थिर पूर्णांक k, स्थिर पूर्णांक s,
        स्थिर पूर्णांक p, स्थिर पूर्णांक parallel_elements,
        स्थिर Ti * src, To * dst, स्थिर क्रमागत ggml_op_pool op,
        स्थिर sycl::nd_item<3> & item_ct1) {
    पूर्णांक idx = item_ct1.get_local_id(2) +
              item_ct1.get_group(2) * item_ct1.get_local_range(2);
    अगर (idx >= parallel_elements) {
        वापस;
    }

    स्थिर पूर्णांक nc     = idx / ow;
    स्थिर पूर्णांक cur_ow = idx % ow;
    स्थिर Ti * i_ptr = src + nc * iw;
    To *       o_ptr = dst + nc * ow;
    स्थिर पूर्णांक  start = cur_ow * s - p;
    स्थिर पूर्णांक  b     = sycl::max(0, start);
    स्थिर पूर्णांक  e     = sycl::min(iw, start + k);

    To res = 0;
    चयन (op) {
        हाल GGML_OP_POOL_AVG: res = 0;        अवरोध;
        हाल GGML_OP_POOL_MAX: res = -भग्न_उच्च; अवरोध;
        शेष:
            res = (To) sycl::nan(uint32_t(0));
            अवरोध;
    }

    क्रम (पूर्णांक j = b; j < e; j += 1) {
        Ti cur = i_ptr[j];
        चयन (op) {
            हाल GGML_OP_POOL_AVG: res += cur; अवरोध;
            हाल GGML_OP_POOL_MAX: res = sycl::max(res, (To) cur); अवरोध;
            शेष:
                res = (To) sycl::nan(uint32_t(0));
                अवरोध;
        }
    }

    स्थिर पूर्णांक count = e - b;
    अगर (op == GGML_OP_POOL_AVG) {
        res = (count > 0) ? (res / count) : (To) 0;
    }
    o_ptr[cur_ow] = res;
}

व्योम ggml_sycl_op_pool2d(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);
    dpct::queue_ptr main_stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));
    स्थिर भग्न * src0_dd = static_cast<स्थिर भग्न *>(dst->src[0]->तथ्य);
    भग्न *       dst_dd  = static_cast<भग्न *>(dst->तथ्य);

    स्थिर int32_t * opts = (स्थिर int32_t *)dst->op_params;
    क्रमागत ggml_op_pool op = static_cast<ggml_op_pool>(opts[0]);
    स्थिर पूर्णांक k0 = opts[1];
    स्थिर पूर्णांक k1 = opts[2];
    स्थिर पूर्णांक s0 = opts[3];
    स्थिर पूर्णांक s1 = opts[4];
    स्थिर पूर्णांक p0 = opts[5];
    स्थिर पूर्णांक p1 = opts[6];

    स्थिर int64_t IH = dst->src[0]->ne[1];
    स्थिर int64_t IW = dst->src[0]->ne[0];

    स्थिर int64_t N = dst->ne[3];
    स्थिर int64_t OC = dst->ne[2];
    स्थिर int64_t OH = dst->ne[1];
    स्थिर int64_t OW = dst->ne[0];

    स्थिर पूर्णांक parallel_elements = N * OC * OH * OW;
    स्थिर पूर्णांक अंक_blocks = (parallel_elements + SYCL_POOL2_D_BLOCK_आकार - 1) / SYCL_POOL2_D_BLOCK_आकार;
    sycl::range<3> block_nums(1, 1, अंक_blocks);
    main_stream->parallel_for(
        sycl::nd_range<3>(block_nums *
                              sycl::range<3>(1, 1, SYCL_IM2_COL_BLOCK_आकार),
                          sycl::range<3>(1, 1, SYCL_IM2_COL_BLOCK_आकार)),
        [=](sycl::nd_item<3> item_ct1) {
            pool2d_nchw_kernel(IH, IW, OH, OW, k1, k0, s1, s0, p1, p0,
                               parallel_elements, src0_dd, dst_dd, op,
                               item_ct1);
        });
}

व्योम ggml_sycl_op_pool1d(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);
    dpct::queue_ptr main_stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));
    स्थिर भग्न * src0_dd = static_cast<स्थिर भग्न *>(dst->src[0]->तथ्य);
    भग्न *       dst_dd  = static_cast<भग्न *>(dst->तथ्य);

    स्थिर int32_t * opts = (स्थिर int32_t *)dst->op_params;
    क्रमागत ggml_op_pool op = static_cast<ggml_op_pool>(opts[0]);
    स्थिर पूर्णांक k0 = opts[1];
    स्थिर पूर्णांक s0 = opts[2];
    स्थिर पूर्णांक p0 = opts[3];

    स्थिर int64_t IW = dst->src[0]->ne[0];
    स्थिर int64_t OW = dst->ne[0];
    स्थिर int64_t NC = dst->ne[3] * dst->ne[2] * dst->ne[1];

    स्थिर पूर्णांक parallel_elements = NC * OW;
    स्थिर पूर्णांक अंक_blocks = (parallel_elements + SYCL_POOL1_D_BLOCK_आकार - 1) / SYCL_POOL1_D_BLOCK_आकार;
    sycl::range<3> block_nums(1, 1, अंक_blocks);
    main_stream->parallel_for(
        sycl::nd_range<3>(block_nums *
                              sycl::range<3>(1, 1, SYCL_POOL1_D_BLOCK_आकार),
                          sycl::range<3>(1, 1, SYCL_POOL1_D_BLOCK_आकार)),
        [=](sycl::nd_item<3> item_ct1) {
            pool1d_ncw_kernel(IW, OW, k0, s0, p0,
                              parallel_elements, src0_dd, dst_dd, op,
                              item_ct1);
        });
}
