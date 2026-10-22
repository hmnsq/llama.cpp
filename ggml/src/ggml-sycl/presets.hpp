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

#جيڪڏهن_وصف_نه_ٿيل GGML_SYCL_PRESETS_HPP
#وصف_ڪريو GGML_SYCL_PRESETS_HPP

#وصف_ڪريو GGML_SYCL_MAX_STREAMS       8
#وصف_ڪريو GGML_SYCL_MAX_BUFFERS       256

#وصف_ڪريو WARP_سائيز GGML_SYCL_WARP_سائيز
#وصف_ڪريو ميٽرڪس_صف_PADDING 512 // last row of quant. matrices is a multiple of this to avoid out-of-bounds memory accesses

#وصف_ڪريو SYCL_COL2_IM_1_D_بلاڪ_سائيز 256
#وصف_ڪريو SYCL_TANH_بلاڪ_سائيز 256
#وصف_ڪريو SYCL_RELU_بلاڪ_سائيز 256
#وصف_ڪريو SYCL_HARDSIGMOID_بلاڪ_سائيز 256
#وصف_ڪريو SYCL_HARDSWISH_بلاڪ_سائيز 256
#وصف_ڪريو SYCL_EXP_بلاڪ_سائيز 256
#وصف_ڪريو SYCL_NEG_بلاڪ_سائيز 256
#وصف_ڪريو SYCL_SIGMOID_بلاڪ_سائيز 256
#وصف_ڪريو SYCL_SQRT_بلاڪ_سائيز 256
#وصف_ڪريو SYCL_SIN_بلاڪ_سائيز 256
#وصف_ڪريو SYCL_SQR_بلاڪ_سائيز 256
#وصف_ڪريو SYCL_سيٽ_بلاڪ_سائيز 256
#وصف_ڪريو SYCL_CPY_بلاڪ_سائيز 32
#وصف_ڪريو SYCL_SCALE_بلاڪ_سائيز 256
#وصف_ڪريو SYCL_CLAMP_بلاڪ_سائيز 256
#وصف_ڪريو SYCL_ROPE_بلاڪ_سائيز 256
#وصف_ڪريو SYCL_ALIBI_بلاڪ_سائيز 32
#وصف_ڪريو SYCL_DIAG_نقاب_INF_بلاڪ_سائيز 32
#وصف_ڪريو SYCL_QUANTIZE_بلاڪ_سائيز 256
#وصف_ڪريو SYCL_DEQUANTIZE_بلاڪ_سائيز 256
#وصف_ڪريو SYCL_GET_صفون_بلاڪ_سائيز 256
#وصف_ڪريو SYCL_UPSCALE_بلاڪ_سائيز 256
#وصف_ڪريو SYCL_CONCAT_بلاڪ_سائيز 256
#وصف_ڪريو SYCL_PAD_بلاڪ_سائيز 256
#وصف_ڪريو SYCL_ACC_بلاڪ_سائيز 256
#وصف_ڪريو SYCL_IM2_کانو_بلاڪ_سائيز 256
#وصف_ڪريو SYCL_POOL1_D_بلاڪ_سائيز 256
#وصف_ڪريو SYCL_POOL2_D_بلاڪ_سائيز 256
#وصف_ڪريو SYCL_ARGMAX_بلاڪ_سائيز 256
#وصف_ڪريو SYCL_CONV_TRANPOSE_1_D_بلاڪ_سائيز 256
#وصف_ڪريو SYCL_TIMESTEP_EMBEDDING_بلاڪ_سائيز 256
#وصف_ڪريو SYCL_ARANGE_بلاڪ_سائيز 256

// dmmv = dequantize_mul_mat_vec
#جيڪڏهن_وصف_نه_ٿيل GGML_SYCL_DMMV_X
#وصف_ڪريو GGML_SYCL_DMMV_X 32
#جيڪڏهن_پورو
#جيڪڏهن_وصف_نه_ٿيل GGML_SYCL_MMV_Y
#وصف_ڪريو GGML_SYCL_MMV_Y 1
#جيڪڏهن_پورو

#جيڪڏهن_وصف_نه_ٿيل K_QUANTS_في_ITERATION
#وصف_ڪريو K_QUANTS_في_ITERATION 1
#وإلا
static_assert(K_QUANTS_في_ITERATION == 1 || K_QUANTS_في_ITERATION == 2, "K_QUANTS_PER_ITERATION must be 1 or 2");
#جيڪڏهن_پورو

#جيڪڏهن_وصف_نه_ٿيل GGML_SYCL_PEER_MAX_BATCH_سائيز
#وصف_ڪريو GGML_SYCL_PEER_MAX_BATCH_سائيز 128
#جيڪڏهن_پورو // GGML_SYCL_PEER_MAX_BATCH_SIZE

#وصف_ڪريو MUL_MAT_SRC1_کانو_STRIDE 128

#وصف_ڪريو QK_WARP_سائيز 32
#وصف_ڪريو WARP_32_سائيز 32
#وصف_ڪريو WARP_16_سائيز 16

#جيڪڏهن_پورو // GGML_SYCL_PRESETS_HPP
