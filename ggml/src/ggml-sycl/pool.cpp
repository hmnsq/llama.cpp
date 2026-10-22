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

#અન્તર્ભાવ "pool.hpp"
#અન્તર્ભાવ <float.h>

template <typename Ti, typename To>
સ્થિર શૂન્ય pool2d_nchw_kernel(
        અચલ પૂર્ણાંક ih, અચલ પૂર્ણાંક iw, અચલ પૂર્ણાંક oh, અચલ પૂર્ણાંક ow,
        અચલ પૂર્ણાંક kh, અચલ પૂર્ણાંક kw, અચલ પૂર્ણાંક sh, અચલ પૂર્ણાંક sw,
        અચલ પૂર્ણાંક ph, અચલ પૂર્ણાંક pw, અચલ પૂર્ણાંક parallel_elements,
        અચલ Ti* src, To* dst, અચલ ગણના ggml_op_pool op,
        અચલ sycl::nd_item<3> &item_ct1) {
        પૂર્ણાંક idx = item_ct1.get_local_id(2) +
                  item_ct1.get_group(2) * item_ct1.get_local_range(2);
        જો (idx >= parallel_elements) {
            પાછા;
        }

        અચલ પૂર્ણાંક I_HW = ih * iw;
        અચલ પૂર્ણાંક O_HW = oh * ow;
        અચલ પૂર્ણાંક nc = idx / O_HW;
        અચલ પૂર્ણાંક cur_oh = idx % O_HW / ow;
        અચલ પૂર્ણાંક cur_ow = idx % O_HW % ow;
        અચલ Ti* i_ptr = src + nc * I_HW;
        To* o_ptr = dst + nc * O_HW;
        અચલ પૂર્ણાંક start_h = cur_oh * sh - ph;
        અચલ પૂર્ણાંક bh = sycl::max(0, start_h);
        અચલ પૂર્ણાંક eh = sycl::min(ih, start_h + kh);
        અચલ પૂર્ણાંક start_w = cur_ow * sw - pw;
        અચલ પૂર્ણાંક bw = sycl::max(0, start_w);
        અચલ પૂર્ણાંક ew = sycl::min(iw, start_w + kw);

        To res = 0;

        પસંદગી (op) {
            સ્થિતિ GGML_OP_POOL_AVG: res = 0; તોડો;
            સ્થિતિ GGML_OP_POOL_MAX: res = -FLT_MAX; તોડો;
            મૂળભૂત:
                res      = (To) sycl::nan(uint32_t(0));
                તોડો;
        }

        માટે (પૂર્ણાંક i = bh; i < eh; i += 1) {
            માટે (પૂર્ણાંક j = bw; j < ew; j += 1) {
                Ti cur = i_ptr[i * iw + j];
                પસંદગી (op) {
                    સ્થિતિ GGML_OP_POOL_AVG: res += (cur / (kh * kw)); તોડો;
                    સ્થિતિ GGML_OP_POOL_MAX: res = sycl::max(res, (To)cur); તોડો;
                    મૂળભૂત:
                        res = (To) sycl::nan(uint32_t(0));
                        તોડો;
                }
            }
        }
        o_ptr[cur_oh * ow + cur_ow] = res;
}

template <typename Ti, typename To>
સ્થિર શૂન્ય pool1d_ncw_kernel(
        અચલ પૂર્ણાંક iw, અચલ પૂર્ણાંક ow,
        અચલ પૂર્ણાંક k, અચલ પૂર્ણાંક s,
        અચલ પૂર્ણાંક p, અચલ પૂર્ણાંક parallel_elements,
        અચલ Ti * src, To * dst, અચલ ગણના ggml_op_pool op,
        અચલ sycl::nd_item<3> & item_ct1) {
    પૂર્ણાંક idx = item_ct1.get_local_id(2) +
              item_ct1.get_group(2) * item_ct1.get_local_range(2);
    જો (idx >= parallel_elements) {
        પાછા;
    }

    અચલ પૂર્ણાંક nc     = idx / ow;
    અચલ પૂર્ણાંક cur_ow = idx % ow;
    અચલ Ti * i_ptr = src + nc * iw;
    To *       o_ptr = dst + nc * ow;
    અચલ પૂર્ણાંક  start = cur_ow * s - p;
    અચલ પૂર્ણાંક  b     = sycl::max(0, start);
    અચલ પૂર્ણાંક  e     = sycl::min(iw, start + k);

    To res = 0;
    પસંદગી (op) {
        સ્થિતિ GGML_OP_POOL_AVG: res = 0;        તોડો;
        સ્થિતિ GGML_OP_POOL_MAX: res = -FLT_MAX; તોડો;
        મૂળભૂત:
            res = (To) sycl::nan(uint32_t(0));
            તોડો;
    }

    માટે (પૂર્ણાંક j = b; j < e; j += 1) {
        Ti cur = i_ptr[j];
        પસંદગી (op) {
            સ્થિતિ GGML_OP_POOL_AVG: res += cur; તોડો;
            સ્થિતિ GGML_OP_POOL_MAX: res = sycl::max(res, (To) cur); તોડો;
            મૂળભૂત:
                res = (To) sycl::nan(uint32_t(0));
                તોડો;
        }
    }

    અચલ પૂર્ણાંક count = e - b;
    જો (op == GGML_OP_POOL_AVG) {
        res = (count > 0) ? (res / count) : (To) 0;
    }
    o_ptr[cur_ow] = res;
}

શૂન્ય ggml_sycl_op_pool2d(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);
    dpct::queue_ptr main_stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));
    અચલ દશાંશ * src0_dd = static_cast<અચલ દશાંશ *>(dst->src[0]->data);
    દશાંશ *       dst_dd  = static_cast<દશાંશ *>(dst->data);

    અચલ int32_t * opts = (અચલ int32_t *)dst->op_params;
    ગણના ggml_op_pool op = static_cast<ggml_op_pool>(opts[0]);
    અચલ પૂર્ણાંક k0 = opts[1];
    અચલ પૂર્ણાંક k1 = opts[2];
    અચલ પૂર્ણાંક s0 = opts[3];
    અચલ પૂર્ણાંક s1 = opts[4];
    અચલ પૂર્ણાંક p0 = opts[5];
    અચલ પૂર્ણાંક p1 = opts[6];

    અચલ int64_t IH = dst->src[0]->ne[1];
    અચલ int64_t IW = dst->src[0]->ne[0];

    અચલ int64_t N = dst->ne[3];
    અચલ int64_t OC = dst->ne[2];
    અચલ int64_t OH = dst->ne[1];
    અચલ int64_t OW = dst->ne[0];

    અચલ પૂર્ણાંક parallel_elements = N * OC * OH * OW;
    અચલ પૂર્ણાંક અંક_blocks = (parallel_elements + SYCL_POOL2D_BLOCK_SIZE - 1) / SYCL_POOL2D_BLOCK_SIZE;
    sycl::range<3> block_nums(1, 1, અંક_blocks);
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

શૂન્ય ggml_sycl_op_pool1d(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);
    dpct::queue_ptr main_stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));
    અચલ દશાંશ * src0_dd = static_cast<અચલ દશાંશ *>(dst->src[0]->data);
    દશાંશ *       dst_dd  = static_cast<દશાંશ *>(dst->data);

    અચલ int32_t * opts = (અચલ int32_t *)dst->op_params;
    ગણના ggml_op_pool op = static_cast<ggml_op_pool>(opts[0]);
    અચલ પૂર્ણાંક k0 = opts[1];
    અચલ પૂર્ણાંક s0 = opts[2];
    અચલ પૂર્ણાંક p0 = opts[3];

    અચલ int64_t IW = dst->src[0]->ne[0];
    અચલ int64_t OW = dst->ne[0];
    અચલ int64_t NC = dst->ne[3] * dst->ne[2] * dst->ne[1];

    અચલ પૂર્ણાંક parallel_elements = NC * OW;
    અચલ પૂર્ણાંક અંક_blocks = (parallel_elements + SYCL_POOL1D_BLOCK_SIZE - 1) / SYCL_POOL1D_BLOCK_SIZE;
    sycl::range<3> block_nums(1, 1, અંક_blocks);
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
