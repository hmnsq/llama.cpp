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

#ଅନ୍ତର୍ଭାଵ "pool.hpp"
#ଅନ୍ତର୍ଭାଵ <float.h>

template <typename Ti, typename To>
ସ୍ଥିର ଶୂନ୍ୟ pool2d_nchw_kernel(
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ih, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା iw, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା oh, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ow,
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା kh, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା kw, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା sh, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା sw,
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ph, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା pw, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା parallel_elements,
        ଅଚଳ Ti* src, To* dst, ଅଚଳ ଗଣନା ggml_op_pool op,
        ଅଚଳ sycl::nd_item<3> &item_ct1) {
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା idx = item_ct1.get_local_id(2) +
                  item_ct1.get_group(2) * item_ct1.get_local_range(2);
        ଯଦି (idx >= parallel_elements) {
            ଫେରନ୍ତୁ;
        }

        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା I_HW = ih * iw;
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା O_HW = oh * ow;
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nc = idx / O_HW;
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା cur_oh = idx % O_HW / ow;
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା cur_ow = idx % O_HW % ow;
        ଅଚଳ Ti* i_ptr = src + nc * I_HW;
        To* o_ptr = dst + nc * O_HW;
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା start_h = cur_oh * sh - ph;
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା bh = sycl::max(0, start_h);
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା eh = sycl::min(ih, start_h + kh);
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା start_w = cur_ow * sw - pw;
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା bw = sycl::max(0, start_w);
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ew = sycl::min(iw, start_w + kw);

        To res = 0;

        ବାଛ (op) {
            ସ୍ଥିତି GGML_OP_POOL_AVG: res = 0; ଭାଙ୍ଗ;
            ସ୍ଥିତି GGML_OP_POOL_MAX: res = -FLT_MAX; ଭାଙ୍ଗ;
            ପୂର୍ବନିର୍ଦ୍ଧାରିତ:
                res      = (To) sycl::nan(uint32_t(0));
                ଭାଙ୍ଗ;
        }

        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = bh; i < eh; i += 1) {
            ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା j = bw; j < ew; j += 1) {
                Ti cur = i_ptr[i * iw + j];
                ବାଛ (op) {
                    ସ୍ଥିତି GGML_OP_POOL_AVG: res += (cur / (kh * kw)); ଭାଙ୍ଗ;
                    ସ୍ଥିତି GGML_OP_POOL_MAX: res = sycl::max(res, (To)cur); ଭାଙ୍ଗ;
                    ପୂର୍ବନିର୍ଦ୍ଧାରିତ:
                        res = (To) sycl::nan(uint32_t(0));
                        ଭାଙ୍ଗ;
                }
            }
        }
        o_ptr[cur_oh * ow + cur_ow] = res;
}

template <typename Ti, typename To>
ସ୍ଥିର ଶୂନ୍ୟ pool1d_ncw_kernel(
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା iw, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ow,
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା k, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା s,
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା p, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା parallel_elements,
        ଅଚଳ Ti * src, To * dst, ଅଚଳ ଗଣନା ggml_op_pool op,
        ଅଚଳ sycl::nd_item<3> & item_ct1) {
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା idx = item_ct1.get_local_id(2) +
              item_ct1.get_group(2) * item_ct1.get_local_range(2);
    ଯଦି (idx >= parallel_elements) {
        ଫେରନ୍ତୁ;
    }

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nc     = idx / ow;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା cur_ow = idx % ow;
    ଅଚଳ Ti * i_ptr = src + nc * iw;
    To *       o_ptr = dst + nc * ow;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା  start = cur_ow * s - p;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା  b     = sycl::max(0, start);
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା  e     = sycl::min(iw, start + k);

    To res = 0;
    ବାଛ (op) {
        ସ୍ଥିତି GGML_OP_POOL_AVG: res = 0;        ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_OP_POOL_MAX: res = -FLT_MAX; ଭାଙ୍ଗ;
        ପୂର୍ବନିର୍ଦ୍ଧାରିତ:
            res = (To) sycl::nan(uint32_t(0));
            ଭାଙ୍ଗ;
    }

    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା j = b; j < e; j += 1) {
        Ti cur = i_ptr[j];
        ବାଛ (op) {
            ସ୍ଥିତି GGML_OP_POOL_AVG: res += cur; ଭାଙ୍ଗ;
            ସ୍ଥିତି GGML_OP_POOL_MAX: res = sycl::max(res, (To) cur); ଭାଙ୍ଗ;
            ପୂର୍ବନିର୍ଦ୍ଧାରିତ:
                res = (To) sycl::nan(uint32_t(0));
                ଭାଙ୍ଗ;
        }
    }

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା count = e - b;
    ଯଦି (op == GGML_OP_POOL_AVG) {
        res = (count > 0) ? (res / count) : (To) 0;
    }
    o_ptr[cur_ow] = res;
}

ଶୂନ୍ୟ ggml_sycl_op_pool2d(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);
    dpct::queue_ptr main_stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));
    ଅଚଳ ଦଶମିକ * src0_dd = static_cast<ଅଚଳ ଦଶମିକ *>(dst->src[0]->data);
    ଦଶମିକ *       dst_dd  = static_cast<ଦଶମିକ *>(dst->data);

    ଅଚଳ int32_t * opts = (ଅଚଳ int32_t *)dst->op_params;
    ଗଣନା ggml_op_pool op = static_cast<ggml_op_pool>(opts[0]);
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା k0 = opts[1];
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା k1 = opts[2];
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା s0 = opts[3];
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା s1 = opts[4];
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା p0 = opts[5];
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା p1 = opts[6];

    ଅଚଳ int64_t IH = dst->src[0]->ne[1];
    ଅଚଳ int64_t IW = dst->src[0]->ne[0];

    ଅଚଳ int64_t N = dst->ne[3];
    ଅଚଳ int64_t OC = dst->ne[2];
    ଅଚଳ int64_t OH = dst->ne[1];
    ଅଚଳ int64_t OW = dst->ne[0];

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା parallel_elements = N * OC * OH * OW;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ଅଙ୍କ_blocks = (parallel_elements + SYCL_POOL2D_BLOCK_SIZE - 1) / SYCL_POOL2D_BLOCK_SIZE;
    sycl::range<3> block_nums(1, 1, ଅଙ୍କ_blocks);
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

ଶୂନ୍ୟ ggml_sycl_op_pool1d(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);
    dpct::queue_ptr main_stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));
    ଅଚଳ ଦଶମିକ * src0_dd = static_cast<ଅଚଳ ଦଶମିକ *>(dst->src[0]->data);
    ଦଶମିକ *       dst_dd  = static_cast<ଦଶମିକ *>(dst->data);

    ଅଚଳ int32_t * opts = (ଅଚଳ int32_t *)dst->op_params;
    ଗଣନା ggml_op_pool op = static_cast<ggml_op_pool>(opts[0]);
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା k0 = opts[1];
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା s0 = opts[2];
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା p0 = opts[3];

    ଅଚଳ int64_t IW = dst->src[0]->ne[0];
    ଅଚଳ int64_t OW = dst->ne[0];
    ଅଚଳ int64_t NC = dst->ne[3] * dst->ne[2] * dst->ne[1];

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା parallel_elements = NC * OW;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ଅଙ୍କ_blocks = (parallel_elements + SYCL_POOL1D_BLOCK_SIZE - 1) / SYCL_POOL1D_BLOCK_SIZE;
    sycl::range<3> block_nums(1, 1, ଅଙ୍କ_blocks);
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
