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

ಮಾದರಿ <ಬಗೆ_ಹೆಸರು Ti, ಬಗೆ_ಹೆಸರು ಗಮ್ಯ_2>
ಸ್ಥಿರ ಶೂನ್ಯ pool2d_nchw_kernel(
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ih, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ iw, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ oh, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ow,
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ kh, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ kw, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ sh, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ sw,
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ph, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ pw, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ parallel_elements,
        ಸ್ಥಿರಾಂಕ Ti* src, ಗಮ್ಯ_2* dst, ಸ್ಥಿರಾಂಕ ಎಣಿಕೆ ggml_op_pool op,
        ಸ್ಥಿರಾಂಕ sycl::nd_ಅಂಶ<3> &ಅಂಶ_ct1) {
        ಪೂರ್ಣಾಂಕ idx = ಅಂಶ_ct1.get_local_id(2) +
                  ಅಂಶ_ct1.get_ಗುಂಪು(2) * ಅಂಶ_ct1.get_local_range(2);
        ಆದರೆ (idx >= parallel_elements) {
            ಹಿಂತಿರುಗಿ;
        }

        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ I_HW = ih * iw;
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ O_HW = oh * ow;
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nc = idx / O_HW;
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ cur_oh = idx % O_HW / ow;
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ cur_ow = idx % O_HW % ow;
        ಸ್ಥಿರಾಂಕ Ti* i_ptr = src + nc * I_HW;
        ಗಮ್ಯ_2* o_ptr = dst + nc * O_HW;
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಆರಂಭ_h = cur_oh * sh - ph;
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ bh = sycl::max(0, ಆರಂಭ_h);
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ eh = sycl::min(ih, ಆರಂಭ_h + kh);
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಆರಂಭ_w = cur_ow * sw - pw;
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ bw = sycl::max(0, ಆರಂಭ_w);
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ew = sycl::min(iw, ಆರಂಭ_w + kw);

        ಗಮ್ಯ_2 res = 0;

        ಆಯ್ಕೆ (op) {
            ಸ್ಥಿತಿ GGML_OP_POOL_AVG: res = 0; ನಿಲ್ಲಿಸು;
            ಸ್ಥಿತಿ GGML_OP_POOL_MAX: res = -FLT_MAX; ನಿಲ್ಲಿಸು;
            ಪೂರ್ವನಿಯೋಜಿತ:
                res      = (ಗಮ್ಯ_2) sycl::nan(uint32_t(0));
                ನಿಲ್ಲಿಸು;
        }

        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = bh; i < eh; i += 1) {
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = bw; j < ew; j += 1) {
                Ti cur = i_ptr[i * iw + j];
                ಆಯ್ಕೆ (op) {
                    ಸ್ಥಿತಿ GGML_OP_POOL_AVG: res += (cur / (kh * kw)); ನಿಲ್ಲಿಸು;
                    ಸ್ಥಿತಿ GGML_OP_POOL_MAX: res = sycl::max(res, (ಗಮ್ಯ_2)cur); ನಿಲ್ಲಿಸು;
                    ಪೂರ್ವನಿಯೋಜಿತ:
                        res = (ಗಮ್ಯ_2) sycl::nan(uint32_t(0));
                        ನಿಲ್ಲಿಸು;
                }
            }
        }
        o_ptr[cur_oh * ow + cur_ow] = res;
}

ಮಾದರಿ <ಬಗೆ_ಹೆಸರು Ti, ಬಗೆ_ಹೆಸರು ಗಮ್ಯ_2>
ಸ್ಥಿರ ಶೂನ್ಯ pool1d_ncw_kernel(
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ iw, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ow,
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ k, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ s,
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ p, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ parallel_elements,
        ಸ್ಥಿರಾಂಕ Ti * src, ಗಮ್ಯ_2 * dst, ಸ್ಥಿರಾಂಕ ಎಣಿಕೆ ggml_op_pool op,
        ಸ್ಥಿರಾಂಕ sycl::nd_ಅಂಶ<3> & ಅಂಶ_ct1) {
    ಪೂರ್ಣಾಂಕ idx = ಅಂಶ_ct1.get_local_id(2) +
              ಅಂಶ_ct1.get_ಗುಂಪು(2) * ಅಂಶ_ct1.get_local_range(2);
    ಆದರೆ (idx >= parallel_elements) {
        ಹಿಂತಿರುಗಿ;
    }

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nc     = idx / ow;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ cur_ow = idx % ow;
    ಸ್ಥಿರಾಂಕ Ti * i_ptr = src + nc * iw;
    ಗಮ್ಯ_2 *       o_ptr = dst + nc * ow;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ  start = cur_ow * s - p;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ  b     = sycl::max(0, start);
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ  e     = sycl::min(iw, start + k);

    ಗಮ್ಯ_2 res = 0;
    ಆಯ್ಕೆ (op) {
        ಸ್ಥಿತಿ GGML_OP_POOL_AVG: res = 0;        ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_POOL_MAX: res = -FLT_MAX; ನಿಲ್ಲಿಸು;
        ಪೂರ್ವನಿಯೋಜಿತ:
            res = (ಗಮ್ಯ_2) sycl::nan(uint32_t(0));
            ನಿಲ್ಲಿಸು;
    }

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = b; j < e; j += 1) {
        Ti cur = i_ptr[j];
        ಆಯ್ಕೆ (op) {
            ಸ್ಥಿತಿ GGML_OP_POOL_AVG: res += cur; ನಿಲ್ಲಿಸು;
            ಸ್ಥಿತಿ GGML_OP_POOL_MAX: res = sycl::max(res, (ಗಮ್ಯ_2) cur); ನಿಲ್ಲಿಸು;
            ಪೂರ್ವನಿಯೋಜಿತ:
                res = (ಗಮ್ಯ_2) sycl::nan(uint32_t(0));
                ನಿಲ್ಲಿಸು;
        }
    }

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ count = e - b;
    ಆದರೆ (op == GGML_OP_POOL_AVG) {
        res = (count > 0) ? (res / count) : (ಗಮ್ಯ_2) 0;
    }
    o_ptr[cur_ow] = res;
}

ಶೂನ್ಯ ggml_sycl_op_pool2d(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);
    dpct::ಸರತಿ_ptr main_ಹರಿವು = ctx.stream();
    SYCL_CHECK(ggml_sycl_ಗಣ_device(ctx.device));
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * src0_dd = static_cast<ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *>(dst->src[0]->ದತ್ತಾಂಶ);
    ದಶಮಾಂಶ *       dst_dd  = static_cast<ದಶಮಾಂಶ *>(dst->ದತ್ತಾಂಶ);

    ಸ್ಥಿರಾಂಕ int32_t * opts = (ಸ್ಥಿರಾಂಕ int32_t *)dst->op_params;
    ಎಣಿಕೆ ggml_op_pool op = static_cast<ggml_op_pool>(opts[0]);
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ k0 = opts[1];
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ k1 = opts[2];
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ s0 = opts[3];
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ s1 = opts[4];
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ p0 = opts[5];
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ p1 = opts[6];

    ಸ್ಥಿರಾಂಕ int64_t IH = dst->src[0]->ne[1];
    ಸ್ಥಿರಾಂಕ int64_t IW = dst->src[0]->ne[0];

    ಸ್ಥಿರಾಂಕ int64_t N = dst->ne[3];
    ಸ್ಥಿರಾಂಕ int64_t OC = dst->ne[2];
    ಸ್ಥಿರಾಂಕ int64_t OH = dst->ne[1];
    ಸ್ಥಿರಾಂಕ int64_t OW = dst->ne[0];

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ parallel_elements = N * OC * OH * OW;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಅಂಕೆ_ಖಂಡಗಳು = (parallel_elements + SYCL_POOL2_D_ಖಂಡ_ಗಾತ್ರ - 1) / SYCL_POOL2_D_ಖಂಡ_ಗಾತ್ರ;
    sycl::range<3> ಖಂಡ_nums(1, 1, ಅಂಕೆ_ಖಂಡಗಳು);
    main_ಹರಿವು->parallel_for(
        sycl::nd_range<3>(ಖಂಡ_nums *
                              sycl::range<3>(1, 1, SYCL_IM2_ಲಂಬಸಾಲು_ಖಂಡ_ಗಾತ್ರ),
                          sycl::range<3>(1, 1, SYCL_IM2_ಲಂಬಸಾಲು_ಖಂಡ_ಗಾತ್ರ)),
        [=](sycl::nd_ಅಂಶ<3> ಅಂಶ_ct1) {
            pool2d_nchw_kernel(IH, IW, OH, OW, k1, k0, s1, s0, p1, p0,
                               parallel_elements, src0_dd, dst_dd, op,
                               ಅಂಶ_ct1);
        });
}

ಶೂನ್ಯ ggml_sycl_op_pool1d(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);
    dpct::ಸರತಿ_ptr main_ಹರಿವು = ctx.stream();
    SYCL_CHECK(ggml_sycl_ಗಣ_device(ctx.device));
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * src0_dd = static_cast<ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *>(dst->src[0]->ದತ್ತಾಂಶ);
    ದಶಮಾಂಶ *       dst_dd  = static_cast<ದಶಮಾಂಶ *>(dst->ದತ್ತಾಂಶ);

    ಸ್ಥಿರಾಂಕ int32_t * opts = (ಸ್ಥಿರಾಂಕ int32_t *)dst->op_params;
    ಎಣಿಕೆ ggml_op_pool op = static_cast<ggml_op_pool>(opts[0]);
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ k0 = opts[1];
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ s0 = opts[2];
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ p0 = opts[3];

    ಸ್ಥಿರಾಂಕ int64_t IW = dst->src[0]->ne[0];
    ಸ್ಥಿರಾಂಕ int64_t OW = dst->ne[0];
    ಸ್ಥಿರಾಂಕ int64_t NC = dst->ne[3] * dst->ne[2] * dst->ne[1];

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ parallel_elements = NC * OW;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಅಂಕೆ_ಖಂಡಗಳು = (parallel_elements + SYCL_POOL1_D_ಖಂಡ_ಗಾತ್ರ - 1) / SYCL_POOL1_D_ಖಂಡ_ಗಾತ್ರ;
    sycl::range<3> ಖಂಡ_nums(1, 1, ಅಂಕೆ_ಖಂಡಗಳು);
    main_ಹರಿವು->parallel_for(
        sycl::nd_range<3>(ಖಂಡ_nums *
                              sycl::range<3>(1, 1, SYCL_POOL1_D_ಖಂಡ_ಗಾತ್ರ),
                          sycl::range<3>(1, 1, SYCL_POOL1_D_ಖಂಡ_ಗಾತ್ರ)),
        [=](sycl::nd_ಅಂಶ<3> ಅಂಶ_ct1) {
            pool1d_ncw_kernel(IW, OW, k0, s0, p0,
                              parallel_elements, src0_dd, dst_dd, op,
                              ಅಂಶ_ct1);
        });
}
