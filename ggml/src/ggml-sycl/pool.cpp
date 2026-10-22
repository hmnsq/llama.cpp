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

#அந்தர்பாவ "pool.hpp"
#அந்தர்பாவ <float.h>

template <typename Ti, typename To>
நிலையான வெறுமை pool2d_nchw_kernel(
        மாறிலி முழு ih, மாறிலி முழு iw, மாறிலி முழு oh, மாறிலி முழு ow,
        மாறிலி முழு kh, மாறிலி முழு kw, மாறிலி முழு sh, மாறிலி முழு sw,
        மாறிலி முழு ph, மாறிலி முழு pw, மாறிலி முழு parallel_elements,
        மாறிலி Ti* src, To* dst, மாறிலி எண்ணிக்கை ggml_op_pool op,
        மாறிலி sycl::nd_item<3> &item_ct1) {
        முழு idx = item_ct1.get_local_id(2) +
                  item_ct1.get_group(2) * item_ct1.get_local_range(2);
        எனில் (idx >= parallel_elements) {
            திரும்பு;
        }

        மாறிலி முழு I_HW = ih * iw;
        மாறிலி முழு O_HW = oh * ow;
        மாறிலி முழு nc = idx / O_HW;
        மாறிலி முழு cur_oh = idx % O_HW / ow;
        மாறிலி முழு cur_ow = idx % O_HW % ow;
        மாறிலி Ti* i_ptr = src + nc * I_HW;
        To* o_ptr = dst + nc * O_HW;
        மாறிலி முழு start_h = cur_oh * sh - ph;
        மாறிலி முழு bh = sycl::max(0, start_h);
        மாறிலி முழு eh = sycl::min(ih, start_h + kh);
        மாறிலி முழு start_w = cur_ow * sw - pw;
        மாறிலி முழு bw = sycl::max(0, start_w);
        மாறிலி முழு ew = sycl::min(iw, start_w + kw);

        To res = 0;

        தேர்வு (op) {
            நிலை GGML_OP_POOL_AVG: res = 0; நிறுத்து;
            நிலை GGML_OP_POOL_MAX: res = -FLT_MAX; நிறுத்து;
            இயல்பு:
                res      = (To) sycl::nan(uint32_t(0));
                நிறுத்து;
        }

        க்கு (முழு i = bh; i < eh; i += 1) {
            க்கு (முழு j = bw; j < ew; j += 1) {
                Ti cur = i_ptr[i * iw + j];
                தேர்வு (op) {
                    நிலை GGML_OP_POOL_AVG: res += (cur / (kh * kw)); நிறுத்து;
                    நிலை GGML_OP_POOL_MAX: res = sycl::max(res, (To)cur); நிறுத்து;
                    இயல்பு:
                        res = (To) sycl::nan(uint32_t(0));
                        நிறுத்து;
                }
            }
        }
        o_ptr[cur_oh * ow + cur_ow] = res;
}

template <typename Ti, typename To>
நிலையான வெறுமை pool1d_ncw_kernel(
        மாறிலி முழு iw, மாறிலி முழு ow,
        மாறிலி முழு k, மாறிலி முழு s,
        மாறிலி முழு p, மாறிலி முழு parallel_elements,
        மாறிலி Ti * src, To * dst, மாறிலி எண்ணிக்கை ggml_op_pool op,
        மாறிலி sycl::nd_item<3> & item_ct1) {
    முழு idx = item_ct1.get_local_id(2) +
              item_ct1.get_group(2) * item_ct1.get_local_range(2);
    எனில் (idx >= parallel_elements) {
        திரும்பு;
    }

    மாறிலி முழு nc     = idx / ow;
    மாறிலி முழு cur_ow = idx % ow;
    மாறிலி Ti * i_ptr = src + nc * iw;
    To *       o_ptr = dst + nc * ow;
    மாறிலி முழு  start = cur_ow * s - p;
    மாறிலி முழு  b     = sycl::max(0, start);
    மாறிலி முழு  e     = sycl::min(iw, start + k);

    To res = 0;
    தேர்வு (op) {
        நிலை GGML_OP_POOL_AVG: res = 0;        நிறுத்து;
        நிலை GGML_OP_POOL_MAX: res = -FLT_MAX; நிறுத்து;
        இயல்பு:
            res = (To) sycl::nan(uint32_t(0));
            நிறுத்து;
    }

    க்கு (முழு j = b; j < e; j += 1) {
        Ti cur = i_ptr[j];
        தேர்வு (op) {
            நிலை GGML_OP_POOL_AVG: res += cur; நிறுத்து;
            நிலை GGML_OP_POOL_MAX: res = sycl::max(res, (To) cur); நிறுத்து;
            இயல்பு:
                res = (To) sycl::nan(uint32_t(0));
                நிறுத்து;
        }
    }

    மாறிலி முழு count = e - b;
    எனில் (op == GGML_OP_POOL_AVG) {
        res = (count > 0) ? (res / count) : (To) 0;
    }
    o_ptr[cur_ow] = res;
}

வெறுமை ggml_sycl_op_pool2d(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);
    dpct::queue_ptr main_stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));
    மாறிலி தசம * src0_dd = static_cast<மாறிலி தசம *>(dst->src[0]->data);
    தசம *       dst_dd  = static_cast<தசம *>(dst->data);

    மாறிலி int32_t * opts = (மாறிலி int32_t *)dst->op_params;
    எண்ணிக்கை ggml_op_pool op = static_cast<ggml_op_pool>(opts[0]);
    மாறிலி முழு k0 = opts[1];
    மாறிலி முழு k1 = opts[2];
    மாறிலி முழு s0 = opts[3];
    மாறிலி முழு s1 = opts[4];
    மாறிலி முழு p0 = opts[5];
    மாறிலி முழு p1 = opts[6];

    மாறிலி int64_t IH = dst->src[0]->ne[1];
    மாறிலி int64_t IW = dst->src[0]->ne[0];

    மாறிலி int64_t N = dst->ne[3];
    மாறிலி int64_t OC = dst->ne[2];
    மாறிலி int64_t OH = dst->ne[1];
    மாறிலி int64_t OW = dst->ne[0];

    மாறிலி முழு parallel_elements = N * OC * OH * OW;
    மாறிலி முழு இலக்கம்_blocks = (parallel_elements + SYCL_POOL2D_BLOCK_SIZE - 1) / SYCL_POOL2D_BLOCK_SIZE;
    sycl::range<3> block_nums(1, 1, இலக்கம்_blocks);
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

வெறுமை ggml_sycl_op_pool1d(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);
    dpct::queue_ptr main_stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));
    மாறிலி தசம * src0_dd = static_cast<மாறிலி தசம *>(dst->src[0]->data);
    தசம *       dst_dd  = static_cast<தசம *>(dst->data);

    மாறிலி int32_t * opts = (மாறிலி int32_t *)dst->op_params;
    எண்ணிக்கை ggml_op_pool op = static_cast<ggml_op_pool>(opts[0]);
    மாறிலி முழு k0 = opts[1];
    மாறிலி முழு s0 = opts[2];
    மாறிலி முழு p0 = opts[3];

    மாறிலி int64_t IW = dst->src[0]->ne[0];
    மாறிலி int64_t OW = dst->ne[0];
    மாறிலி int64_t NC = dst->ne[3] * dst->ne[2] * dst->ne[1];

    மாறிலி முழு parallel_elements = NC * OW;
    மாறிலி முழு இலக்கம்_blocks = (parallel_elements + SYCL_POOL1D_BLOCK_SIZE - 1) / SYCL_POOL1D_BLOCK_SIZE;
    sycl::range<3> block_nums(1, 1, இலக்கம்_blocks);
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
