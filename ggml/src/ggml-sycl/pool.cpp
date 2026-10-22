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

#അന്തര്ഭാവ "pool.hpp"
#അന്തര്ഭാവ <float.h>

template <typename Ti, typename To>
സ്ഥിതം ശൂന്യം pool2d_nchw_kernel(
        സ്ഥിരം പൂർണ്ണസംഖ്യ ih, സ്ഥിരം പൂർണ്ണസംഖ്യ iw, സ്ഥിരം പൂർണ്ണസംഖ്യ oh, സ്ഥിരം പൂർണ്ണസംഖ്യ ow,
        സ്ഥിരം പൂർണ്ണസംഖ്യ kh, സ്ഥിരം പൂർണ്ണസംഖ്യ kw, സ്ഥിരം പൂർണ്ണസംഖ്യ sh, സ്ഥിരം പൂർണ്ണസംഖ്യ sw,
        സ്ഥിരം പൂർണ്ണസംഖ്യ ph, സ്ഥിരം പൂർണ്ണസംഖ്യ pw, സ്ഥിരം പൂർണ്ണസംഖ്യ parallel_elements,
        സ്ഥിരം Ti* src, To* dst, സ്ഥിരം എണ്ണൽ ggml_op_pool op,
        സ്ഥിരം sycl::nd_item<3> &item_ct1) {
        പൂർണ്ണസംഖ്യ idx = item_ct1.get_local_id(2) +
                  item_ct1.get_group(2) * item_ct1.get_local_range(2);
        എങ്കിൽ (idx >= parallel_elements) {
            തിരികെ;
        }

        സ്ഥിരം പൂർണ്ണസംഖ്യ I_HW = ih * iw;
        സ്ഥിരം പൂർണ്ണസംഖ്യ O_HW = oh * ow;
        സ്ഥിരം പൂർണ്ണസംഖ്യ nc = idx / O_HW;
        സ്ഥിരം പൂർണ്ണസംഖ്യ cur_oh = idx % O_HW / ow;
        സ്ഥിരം പൂർണ്ണസംഖ്യ cur_ow = idx % O_HW % ow;
        സ്ഥിരം Ti* i_ptr = src + nc * I_HW;
        To* o_ptr = dst + nc * O_HW;
        സ്ഥിരം പൂർണ്ണസംഖ്യ start_h = cur_oh * sh - ph;
        സ്ഥിരം പൂർണ്ണസംഖ്യ bh = sycl::max(0, start_h);
        സ്ഥിരം പൂർണ്ണസംഖ്യ eh = sycl::min(ih, start_h + kh);
        സ്ഥിരം പൂർണ്ണസംഖ്യ start_w = cur_ow * sw - pw;
        സ്ഥിരം പൂർണ്ണസംഖ്യ bw = sycl::max(0, start_w);
        സ്ഥിരം പൂർണ്ണസംഖ്യ ew = sycl::min(iw, start_w + kw);

        To res = 0;

        തിരഞ്ഞെടുപ്പ് (op) {
            അവസ്ഥ GGML_OP_POOL_AVG: res = 0; നിർത്തുക;
            അവസ്ഥ GGML_OP_POOL_MAX: res = -FLT_MAX; നിർത്തുക;
            സ്ഥിരസ്ഥിതി:
                res      = (To) sycl::nan(uint32_t(0));
                നിർത്തുക;
        }

        വേണ്ടി (പൂർണ്ണസംഖ്യ i = bh; i < eh; i += 1) {
            വേണ്ടി (പൂർണ്ണസംഖ്യ j = bw; j < ew; j += 1) {
                Ti cur = i_ptr[i * iw + j];
                തിരഞ്ഞെടുപ്പ് (op) {
                    അവസ്ഥ GGML_OP_POOL_AVG: res += (cur / (kh * kw)); നിർത്തുക;
                    അവസ്ഥ GGML_OP_POOL_MAX: res = sycl::max(res, (To)cur); നിർത്തുക;
                    സ്ഥിരസ്ഥിതി:
                        res = (To) sycl::nan(uint32_t(0));
                        നിർത്തുക;
                }
            }
        }
        o_ptr[cur_oh * ow + cur_ow] = res;
}

template <typename Ti, typename To>
സ്ഥിതം ശൂന്യം pool1d_ncw_kernel(
        സ്ഥിരം പൂർണ്ണസംഖ്യ iw, സ്ഥിരം പൂർണ്ണസംഖ്യ ow,
        സ്ഥിരം പൂർണ്ണസംഖ്യ k, സ്ഥിരം പൂർണ്ണസംഖ്യ s,
        സ്ഥിരം പൂർണ്ണസംഖ്യ p, സ്ഥിരം പൂർണ്ണസംഖ്യ parallel_elements,
        സ്ഥിരം Ti * src, To * dst, സ്ഥിരം എണ്ണൽ ggml_op_pool op,
        സ്ഥിരം sycl::nd_item<3> & item_ct1) {
    പൂർണ്ണസംഖ്യ idx = item_ct1.get_local_id(2) +
              item_ct1.get_group(2) * item_ct1.get_local_range(2);
    എങ്കിൽ (idx >= parallel_elements) {
        തിരികെ;
    }

    സ്ഥിരം പൂർണ്ണസംഖ്യ nc     = idx / ow;
    സ്ഥിരം പൂർണ്ണസംഖ്യ cur_ow = idx % ow;
    സ്ഥിരം Ti * i_ptr = src + nc * iw;
    To *       o_ptr = dst + nc * ow;
    സ്ഥിരം പൂർണ്ണസംഖ്യ  start = cur_ow * s - p;
    സ്ഥിരം പൂർണ്ണസംഖ്യ  b     = sycl::max(0, start);
    സ്ഥിരം പൂർണ്ണസംഖ്യ  e     = sycl::min(iw, start + k);

    To res = 0;
    തിരഞ്ഞെടുപ്പ് (op) {
        അവസ്ഥ GGML_OP_POOL_AVG: res = 0;        നിർത്തുക;
        അവസ്ഥ GGML_OP_POOL_MAX: res = -FLT_MAX; നിർത്തുക;
        സ്ഥിരസ്ഥിതി:
            res = (To) sycl::nan(uint32_t(0));
            നിർത്തുക;
    }

    വേണ്ടി (പൂർണ്ണസംഖ്യ j = b; j < e; j += 1) {
        Ti cur = i_ptr[j];
        തിരഞ്ഞെടുപ്പ് (op) {
            അവസ്ഥ GGML_OP_POOL_AVG: res += cur; നിർത്തുക;
            അവസ്ഥ GGML_OP_POOL_MAX: res = sycl::max(res, (To) cur); നിർത്തുക;
            സ്ഥിരസ്ഥിതി:
                res = (To) sycl::nan(uint32_t(0));
                നിർത്തുക;
        }
    }

    സ്ഥിരം പൂർണ്ണസംഖ്യ count = e - b;
    എങ്കിൽ (op == GGML_OP_POOL_AVG) {
        res = (count > 0) ? (res / count) : (To) 0;
    }
    o_ptr[cur_ow] = res;
}

ശൂന്യം ggml_sycl_op_pool2d(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);
    dpct::queue_ptr main_stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));
    സ്ഥിരം ദശാംശം * src0_dd = static_cast<സ്ഥിരം ദശാംശം *>(dst->src[0]->data);
    ദശാംശം *       dst_dd  = static_cast<ദശാംശം *>(dst->data);

    സ്ഥിരം int32_t * opts = (സ്ഥിരം int32_t *)dst->op_params;
    എണ്ണൽ ggml_op_pool op = static_cast<ggml_op_pool>(opts[0]);
    സ്ഥിരം പൂർണ്ണസംഖ്യ k0 = opts[1];
    സ്ഥിരം പൂർണ്ണസംഖ്യ k1 = opts[2];
    സ്ഥിരം പൂർണ്ണസംഖ്യ s0 = opts[3];
    സ്ഥിരം പൂർണ്ണസംഖ്യ s1 = opts[4];
    സ്ഥിരം പൂർണ്ണസംഖ്യ p0 = opts[5];
    സ്ഥിരം പൂർണ്ണസംഖ്യ p1 = opts[6];

    സ്ഥിരം int64_t IH = dst->src[0]->ne[1];
    സ്ഥിരം int64_t IW = dst->src[0]->ne[0];

    സ്ഥിരം int64_t N = dst->ne[3];
    സ്ഥിരം int64_t OC = dst->ne[2];
    സ്ഥിരം int64_t OH = dst->ne[1];
    സ്ഥിരം int64_t OW = dst->ne[0];

    സ്ഥിരം പൂർണ്ണസംഖ്യ parallel_elements = N * OC * OH * OW;
    സ്ഥിരം പൂർണ്ണസംഖ്യ അക്കം_blocks = (parallel_elements + SYCL_POOL2D_BLOCK_SIZE - 1) / SYCL_POOL2D_BLOCK_SIZE;
    sycl::range<3> block_nums(1, 1, അക്കം_blocks);
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

ശൂന്യം ggml_sycl_op_pool1d(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);
    dpct::queue_ptr main_stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));
    സ്ഥിരം ദശാംശം * src0_dd = static_cast<സ്ഥിരം ദശാംശം *>(dst->src[0]->data);
    ദശാംശം *       dst_dd  = static_cast<ദശാംശം *>(dst->data);

    സ്ഥിരം int32_t * opts = (സ്ഥിരം int32_t *)dst->op_params;
    എണ്ണൽ ggml_op_pool op = static_cast<ggml_op_pool>(opts[0]);
    സ്ഥിരം പൂർണ്ണസംഖ്യ k0 = opts[1];
    സ്ഥിരം പൂർണ്ണസംഖ്യ s0 = opts[2];
    സ്ഥിരം പൂർണ്ണസംഖ്യ p0 = opts[3];

    സ്ഥിരം int64_t IW = dst->src[0]->ne[0];
    സ്ഥിരം int64_t OW = dst->ne[0];
    സ്ഥിരം int64_t NC = dst->ne[3] * dst->ne[2] * dst->ne[1];

    സ്ഥിരം പൂർണ്ണസംഖ്യ parallel_elements = NC * OW;
    സ്ഥിരം പൂർണ്ണസംഖ്യ അക്കം_blocks = (parallel_elements + SYCL_POOL1D_BLOCK_SIZE - 1) / SYCL_POOL1D_BLOCK_SIZE;
    sycl::range<3> block_nums(1, 1, അക്കം_blocks);
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
