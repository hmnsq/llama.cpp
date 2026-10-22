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

#اگر_غیر_تعریف_شدہ GGML_SYCL_PRESETS_HPP
#تعریف_کرو GGML_SYCL_PRESETS_HPP

#تعریف_کرو GGML_SYCL_MAX_STREAMS       8
#تعریف_کرو GGML_SYCL_MAX_BUFFERS       256

#تعریف_کرو WARP_سائز GGML_SYCL_WARP_سائز
#تعریف_کرو میٹرکس_صف_PADDING 512 // last row of quant. matrices is a multiple of this to avoid out-of-bounds memory accesses

#تعریف_کرو SYCL_COL2_IM_1_D_بلاک_سائز 256
#تعریف_کرو SYCL_TANH_بلاک_سائز 256
#تعریف_کرو SYCL_RELU_بلاک_سائز 256
#تعریف_کرو SYCL_HARDSIGMOID_بلاک_سائز 256
#تعریف_کرو SYCL_HARDSWISH_بلاک_سائز 256
#تعریف_کرو SYCL_EXP_بلاک_سائز 256
#تعریف_کرو SYCL_NEG_بلاک_سائز 256
#تعریف_کرو SYCL_SIGMOID_بلاک_سائز 256
#تعریف_کرو SYCL_SQRT_بلاک_سائز 256
#تعریف_کرو SYCL_SIN_بلاک_سائز 256
#تعریف_کرو SYCL_SQR_بلاک_سائز 256
#تعریف_کرو SYCL_سیٹ_بلاک_سائز 256
#تعریف_کرو SYCL_CPY_بلاک_سائز 32
#تعریف_کرو SYCL_SCALE_بلاک_سائز 256
#تعریف_کرو SYCL_CLAMP_بلاک_سائز 256
#تعریف_کرو SYCL_ROPE_بلاک_سائز 256
#تعریف_کرو SYCL_ALIBI_بلاک_سائز 32
#تعریف_کرو SYCL_DIAG_نقاب_INF_بلاک_سائز 32
#تعریف_کرو SYCL_QUANTIZE_بلاک_سائز 256
#تعریف_کرو SYCL_DEQUANTIZE_بلاک_سائز 256
#تعریف_کرو SYCL_GET_صفیں_بلاک_سائز 256
#تعریف_کرو SYCL_UPSCALE_بلاک_سائز 256
#تعریف_کرو SYCL_CONCAT_بلاک_سائز 256
#تعریف_کرو SYCL_PAD_بلاک_سائز 256
#تعریف_کرو SYCL_ACC_بلاک_سائز 256
#تعریف_کرو SYCL_IM2_ستون_بلاک_سائز 256
#تعریف_کرو SYCL_POOL1_D_بلاک_سائز 256
#تعریف_کرو SYCL_POOL2_D_بلاک_سائز 256
#تعریف_کرو SYCL_ARGMAX_بلاک_سائز 256
#تعریف_کرو SYCL_CONV_TRANPOSE_1_D_بلاک_سائز 256
#تعریف_کرو SYCL_TIMESTEP_EMBEDDING_بلاک_سائز 256
#تعریف_کرو SYCL_ARANGE_بلاک_سائز 256

// dmmv = dequantize_mul_mat_vec
#اگر_غیر_تعریف_شدہ GGML_SYCL_DMMV_X
#تعریف_کرو GGML_SYCL_DMMV_X 32
#اگر_ختم
#اگر_غیر_تعریف_شدہ GGML_SYCL_MMV_Y
#تعریف_کرو GGML_SYCL_MMV_Y 1
#اگر_ختم

#اگر_غیر_تعریف_شدہ K_QUANTS_فی_ITERATION
#تعریف_کرو K_QUANTS_فی_ITERATION 1
#ورنہ
static_assert(K_QUANTS_فی_ITERATION == 1 || K_QUANTS_فی_ITERATION == 2, "K_QUANTS_PER_ITERATION must be 1 or 2");
#اگر_ختم

#اگر_غیر_تعریف_شدہ GGML_SYCL_PEER_MAX_BATCH_سائز
#تعریف_کرو GGML_SYCL_PEER_MAX_BATCH_سائز 128
#اگر_ختم // GGML_SYCL_PEER_MAX_BATCH_SIZE

#تعریف_کرو MUL_MAT_SRC1_ستون_STRIDE 128

#تعریف_کرو QK_WARP_سائز 32
#تعریف_کرو WARP_32_سائز 32
#تعریف_کرو WARP_16_سائز 16

#اگر_ختم // GGML_SYCL_PRESETS_HPP
