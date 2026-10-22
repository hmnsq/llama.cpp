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

#اگر_تعریف_نشده GGML_SYCL_PRESETS_HPP
#تعریف_کن GGML_SYCL_PRESETS_HPP

#تعریف_کن GGML_SYCL_MAX_STREAMS       8
#تعریف_کن GGML_SYCL_MAX_BUFFERS       256

#تعریف_کن WARP_اندازه GGML_SYCL_WARP_اندازه
#تعریف_کن ماتریس_ردیف_PADDING 512 // last row of quant. matrices is a multiple of this to avoid out-of-bounds memory accesses

#تعریف_کن SYCL_COL2_IM_1_D_بلوک_اندازه 256
#تعریف_کن SYCL_TANH_بلوک_اندازه 256
#تعریف_کن SYCL_RELU_بلوک_اندازه 256
#تعریف_کن SYCL_HARDSIGMOID_بلوک_اندازه 256
#تعریف_کن SYCL_HARDSWISH_بلوک_اندازه 256
#تعریف_کن SYCL_EXP_بلوک_اندازه 256
#تعریف_کن SYCL_NEG_بلوک_اندازه 256
#تعریف_کن SYCL_SIGMOID_بلوک_اندازه 256
#تعریف_کن SYCL_SQRT_بلوک_اندازه 256
#تعریف_کن SYCL_SIN_بلوک_اندازه 256
#تعریف_کن SYCL_SQR_بلوک_اندازه 256
#تعریف_کن SYCL_مجموعه_بلوک_اندازه 256
#تعریف_کن SYCL_CPY_بلوک_اندازه 32
#تعریف_کن SYCL_SCALE_بلوک_اندازه 256
#تعریف_کن SYCL_CLAMP_بلوک_اندازه 256
#تعریف_کن SYCL_ROPE_بلوک_اندازه 256
#تعریف_کن SYCL_ALIBI_بلوک_اندازه 32
#تعریف_کن SYCL_DIAG_ماسک_INF_بلوک_اندازه 32
#تعریف_کن SYCL_QUANTIZE_بلوک_اندازه 256
#تعریف_کن SYCL_DEQUANTIZE_بلوک_اندازه 256
#تعریف_کن SYCL_GET_ردیفها_بلوک_اندازه 256
#تعریف_کن SYCL_UPSCALE_بلوک_اندازه 256
#تعریف_کن SYCL_CONCAT_بلوک_اندازه 256
#تعریف_کن SYCL_PAD_بلوک_اندازه 256
#تعریف_کن SYCL_ACC_بلوک_اندازه 256
#تعریف_کن SYCL_IM2_ستون_بلوک_اندازه 256
#تعریف_کن SYCL_POOL1_D_بلوک_اندازه 256
#تعریف_کن SYCL_POOL2_D_بلوک_اندازه 256
#تعریف_کن SYCL_ARGMAX_بلوک_اندازه 256
#تعریف_کن SYCL_CONV_TRANPOSE_1_D_بلوک_اندازه 256
#تعریف_کن SYCL_TIMESTEP_EMBEDDING_بلوک_اندازه 256
#تعریف_کن SYCL_ARANGE_بلوک_اندازه 256

// dmmv = dequantize_mul_mat_vec
#اگر_تعریف_نشده GGML_SYCL_DMMV_X
#تعریف_کن GGML_SYCL_DMMV_X 32
#پایان_اگر
#اگر_تعریف_نشده GGML_SYCL_MMV_Y
#تعریف_کن GGML_SYCL_MMV_Y 1
#پایان_اگر

#اگر_تعریف_نشده K_QUANTS_در_هر_ITERATION
#تعریف_کن K_QUANTS_در_هر_ITERATION 1
#وإلا
static_assert(K_QUANTS_در_هر_ITERATION == 1 || K_QUANTS_در_هر_ITERATION == 2, "K_QUANTS_PER_ITERATION must be 1 or 2");
#پایان_اگر

#اگر_تعریف_نشده GGML_SYCL_PEER_MAX_BATCH_اندازه
#تعریف_کن GGML_SYCL_PEER_MAX_BATCH_اندازه 128
#پایان_اگر // GGML_SYCL_PEER_MAX_BATCH_SIZE

#تعریف_کن MUL_MAT_SRC1_ستون_STRIDE 128

#تعریف_کن QK_WARP_اندازه 32
#تعریف_کن WARP_32_اندازه 32
#تعریف_کن WARP_16_اندازه 16

#پایان_اگر // GGML_SYCL_PRESETS_HPP
