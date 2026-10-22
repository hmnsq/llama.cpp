//
// MIT license
// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: MIT
//

//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

#که_نه_تعریف_شوی GGML_SYCL_PRESETS_HPP
#تعریف_کړه GGML_SYCL_PRESETS_HPP

#تعریف_کړه GGML_SYCL_MAX_STREAMS       8
#تعریف_کړه GGML_SYCL_MAX_BUFFERS       256

#تعریف_کړه WARP_اندازه GGML_SYCL_WARP_اندازه
#تعریف_کړه ماتریکس_لیکه_PADDING 512 // last row of quant. matrices is a multiple of this to avoid out-of-bounds memory accesses

#تعریف_کړه SYCL_COL2_IM_1_D_بلاک_اندازه 256
#تعریف_کړه SYCL_TANH_بلاک_اندازه 256
#تعریف_کړه SYCL_RELU_بلاک_اندازه 256
#تعریف_کړه SYCL_HARDSIGMOID_بلاک_اندازه 256
#تعریف_کړه SYCL_HARDSWISH_بلاک_اندازه 256
#تعریف_کړه SYCL_EXP_بلاک_اندازه 256
#تعریف_کړه SYCL_NEG_بلاک_اندازه 256
#تعریف_کړه SYCL_SIGMOID_بلاک_اندازه 256
#تعریف_کړه SYCL_SQRT_بلاک_اندازه 256
#تعریف_کړه SYCL_SIN_بلاک_اندازه 256
#تعریف_کړه SYCL_SQR_بلاک_اندازه 256
#تعریف_کړه SYCL_سېټ_بلاک_اندازه 256
#تعریف_کړه SYCL_CPY_بلاک_اندازه 32
#تعریف_کړه SYCL_SCALE_بلاک_اندازه 256
#تعریف_کړه SYCL_CLAMP_بلاک_اندازه 256
#تعریف_کړه SYCL_ROPE_بلاک_اندازه 256
#تعریف_کړه SYCL_ALIBI_بلاک_اندازه 32
#تعریف_کړه SYCL_DIAG_ماسک_INF_بلاک_اندازه 32
#تعریف_کړه SYCL_QUANTIZE_بلاک_اندازه 256
#تعریف_کړه SYCL_DEQUANTIZE_بلاک_اندازه 256
#تعریف_کړه SYCL_GET_لیکې_بلاک_اندازه 256
#تعریف_کړه SYCL_UPSCALE_بلاک_اندازه 256
#تعریف_کړه SYCL_CONCAT_بلاک_اندازه 256
#تعریف_کړه SYCL_PAD_بلاک_اندازه 256
#تعریف_کړه SYCL_ACC_بلاک_اندازه 256
#تعریف_کړه SYCL_IM2_ستون_بلاک_اندازه 256
#تعریف_کړه SYCL_POOL1_D_بلاک_اندازه 256
#تعریف_کړه SYCL_POOL2_D_بلاک_اندازه 256
#تعریف_کړه SYCL_ARGMAX_بلاک_اندازه 256
#تعریف_کړه SYCL_CONV_TRANPOSE_1_D_بلاک_اندازه 256
#تعریف_کړه SYCL_TIMESTEP_EMBEDDING_بلاک_اندازه 256
#تعریف_کړه SYCL_ARANGE_بلاک_اندازه 256

// dmmv = dequantize_mul_mat_vec
#که_نه_تعریف_شوی GGML_SYCL_DMMV_X
#تعریف_کړه GGML_SYCL_DMMV_X 32
#د_که_پای
#که_نه_تعریف_شوی GGML_SYCL_MMV_Y
#تعریف_کړه GGML_SYCL_MMV_Y 1
#د_که_پای

#که_نه_تعریف_شوی K_QUANTS_فی_ITERATION
#تعریف_کړه K_QUANTS_فی_ITERATION 1
#وإلا
static_assert(K_QUANTS_فی_ITERATION == 1 || K_QUANTS_فی_ITERATION == 2, "K_QUANTS_PER_ITERATION must be 1 or 2");
#د_که_پای

#که_نه_تعریف_شوی GGML_SYCL_PEER_MAX_BATCH_اندازه
#تعریف_کړه GGML_SYCL_PEER_MAX_BATCH_اندازه 128
#د_که_پای // GGML_SYCL_PEER_MAX_BATCH_SIZE

#تعریف_کړه MUL_MAT_SRC1_ستون_STRIDE 128

#تعریف_کړه QK_WARP_اندازه 32
#تعریف_کړه WARP_32_اندازه 32
#تعریف_کړه WARP_16_اندازه 16

#د_که_پای // GGML_SYCL_PRESETS_HPP
