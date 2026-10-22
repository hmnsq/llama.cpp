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

#ಅನ್ತರ್ಭಾವ "pool.hpp"
#ಅನ್ತರ್ಭಾವ <float.h>

template <typename Ti, typename To>
ಸ್ಥಿರ ಶೂನ್ಯ pool2d_nchw_kernel(
        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ih, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ iw, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ oh, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ow,
        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ kh, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ kw, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ sh, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ sw,
        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ph, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ pw, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ parallel_elements,
        ಧ್ರುವ Ti* src, To* dst, ಧ್ರುವ ಗಣನಾ ggml_op_pool op,
        ಧ್ರುವ sycl::nd_item<3> &item_ct1) {
        ಪೂರ್ಣಾಙ್ಕ idx = item_ct1.get_local_id(2) +
                  item_ct1.get_group(2) * item_ct1.get_local_range(2);
        ಯದಿ (idx >= parallel_elements) {
            ಪ್ರತ್ಯಾಗಮ;
        }

        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ I_HW = ih * iw;
        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ O_HW = oh * ow;
        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nc = idx / O_HW;
        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ cur_oh = idx % O_HW / ow;
        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ cur_ow = idx % O_HW % ow;
        ಧ್ರುವ Ti* i_ptr = src + nc * I_HW;
        To* o_ptr = dst + nc * O_HW;
        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ start_h = cur_oh * sh - ph;
        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ bh = sycl::max(0, start_h);
        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ eh = sycl::min(ih, start_h + kh);
        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ start_w = cur_ow * sw - pw;
        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ bw = sycl::max(0, start_w);
        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ew = sycl::min(iw, start_w + kw);

        To res = 0;

        ವಿಕಲ್ಪ (op) {
            ಪಕ್ಷ GGML_OP_POOL_AVG: res = 0; ಭಙ್ಗ;
            ಪಕ್ಷ GGML_OP_POOL_MAX: res = -FLT_MAX; ಭಙ್ಗ;
            ಸ್ವಭಾವ:
                res      = (To) sycl::nan(uint32_t(0));
                ಭಙ್ಗ;
        }

        ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = bh; i < eh; i += 1) {
            ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ j = bw; j < ew; j += 1) {
                Ti cur = i_ptr[i * iw + j];
                ವಿಕಲ್ಪ (op) {
                    ಪಕ್ಷ GGML_OP_POOL_AVG: res += (cur / (kh * kw)); ಭಙ್ಗ;
                    ಪಕ್ಷ GGML_OP_POOL_MAX: res = sycl::max(res, (To)cur); ಭಙ್ಗ;
                    ಸ್ವಭಾವ:
                        res = (To) sycl::nan(uint32_t(0));
                        ಭಙ್ಗ;
                }
            }
        }
        o_ptr[cur_oh * ow + cur_ow] = res;
}

template <typename Ti, typename To>
ಸ್ಥಿರ ಶೂನ್ಯ pool1d_ncw_kernel(
        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ iw, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ow,
        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ k, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ s,
        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ p, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ parallel_elements,
        ಧ್ರುವ Ti * src, To * dst, ಧ್ರುವ ಗಣನಾ ggml_op_pool op,
        ಧ್ರುವ sycl::nd_item<3> & item_ct1) {
    ಪೂರ್ಣಾಙ್ಕ idx = item_ct1.get_local_id(2) +
              item_ct1.get_group(2) * item_ct1.get_local_range(2);
    ಯದಿ (idx >= parallel_elements) {
        ಪ್ರತ್ಯಾಗಮ;
    }

    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nc     = idx / ow;
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ cur_ow = idx % ow;
    ಧ್ರುವ Ti * i_ptr = src + nc * iw;
    To *       o_ptr = dst + nc * ow;
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ  start = cur_ow * s - p;
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ  b     = sycl::max(0, start);
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ  e     = sycl::min(iw, start + k);

    To res = 0;
    ವಿಕಲ್ಪ (op) {
        ಪಕ್ಷ GGML_OP_POOL_AVG: res = 0;        ಭಙ್ಗ;
        ಪಕ್ಷ GGML_OP_POOL_MAX: res = -FLT_MAX; ಭಙ್ಗ;
        ಸ್ವಭಾವ:
            res = (To) sycl::nan(uint32_t(0));
            ಭಙ್ಗ;
    }

    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ j = b; j < e; j += 1) {
        Ti cur = i_ptr[j];
        ವಿಕಲ್ಪ (op) {
            ಪಕ್ಷ GGML_OP_POOL_AVG: res += cur; ಭಙ್ಗ;
            ಪಕ್ಷ GGML_OP_POOL_MAX: res = sycl::max(res, (To) cur); ಭಙ್ಗ;
            ಸ್ವಭಾವ:
                res = (To) sycl::nan(uint32_t(0));
                ಭಙ್ಗ;
        }
    }

    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ count = e - b;
    ಯದಿ (op == GGML_OP_POOL_AVG) {
        res = (count > 0) ? (res / count) : (To) 0;
    }
    o_ptr[cur_ow] = res;
}

ಶೂನ್ಯ ggml_sycl_op_pool2d(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);
    dpct::queue_ptr main_stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));
    ಧ್ರುವ ಭಿನ್ನ * src0_dd = static_cast<ಧ್ರುವ ಭಿನ್ನ *>(dst->src[0]->data);
    ಭಿನ್ನ *       dst_dd  = static_cast<ಭಿನ್ನ *>(dst->data);

    ಧ್ರುವ int32_t * opts = (ಧ್ರುವ int32_t *)dst->op_params;
    ಗಣನಾ ggml_op_pool op = static_cast<ggml_op_pool>(opts[0]);
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ k0 = opts[1];
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ k1 = opts[2];
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ s0 = opts[3];
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ s1 = opts[4];
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ p0 = opts[5];
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ p1 = opts[6];

    ಧ್ರುವ int64_t IH = dst->src[0]->ne[1];
    ಧ್ರುವ int64_t IW = dst->src[0]->ne[0];

    ಧ್ರುವ int64_t N = dst->ne[3];
    ಧ್ರುವ int64_t OC = dst->ne[2];
    ಧ್ರುವ int64_t OH = dst->ne[1];
    ಧ್ರುವ int64_t OW = dst->ne[0];

    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ parallel_elements = N * OC * OH * OW;
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ಅಂಕೆ_blocks = (parallel_elements + SYCL_POOL2D_BLOCK_SIZE - 1) / SYCL_POOL2D_BLOCK_SIZE;
    sycl::range<3> block_nums(1, 1, ಅಂಕೆ_blocks);
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

ಶೂನ್ಯ ggml_sycl_op_pool1d(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);
    dpct::queue_ptr main_stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));
    ಧ್ರುವ ಭಿನ್ನ * src0_dd = static_cast<ಧ್ರುವ ಭಿನ್ನ *>(dst->src[0]->data);
    ಭಿನ್ನ *       dst_dd  = static_cast<ಭಿನ್ನ *>(dst->data);

    ಧ್ರುವ int32_t * opts = (ಧ್ರುವ int32_t *)dst->op_params;
    ಗಣನಾ ggml_op_pool op = static_cast<ggml_op_pool>(opts[0]);
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ k0 = opts[1];
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ s0 = opts[2];
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ p0 = opts[3];

    ಧ್ರುವ int64_t IW = dst->src[0]->ne[0];
    ಧ್ರುವ int64_t OW = dst->ne[0];
    ಧ್ರುವ int64_t NC = dst->ne[3] * dst->ne[2] * dst->ne[1];

    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ parallel_elements = NC * OW;
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ಅಂಕೆ_blocks = (parallel_elements + SYCL_POOL1D_BLOCK_SIZE - 1) / SYCL_POOL1D_BLOCK_SIZE;
    sycl::range<3> block_nums(1, 1, ಅಂಕೆ_blocks);
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
