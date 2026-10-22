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

#ਜੇ_ਅਪਰਿਭਾਸ਼ਿਤ GGML_SYCL_PRESETS_HPP
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_SYCL_PRESETS_HPP

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_SYCL_MAX_STREAMS       8
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_SYCL_MAX_BUFFERS       256

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ WARP_ਮਾਪ GGML_SYCL_WARP_ਮਾਪ
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ ਮੈਟ੍ਰਿਕਸ_ਪੰਕਤੀ_PADDING 512 // last row of quant. matrices is a multiple of this to avoid out-of-bounds memory accesses

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ SYCL_COL2_IM_1_D_ਖੰਡ_ਮਾਪ 256
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ SYCL_TANH_ਖੰਡ_ਮਾਪ 256
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ SYCL_RELU_ਖੰਡ_ਮਾਪ 256
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ SYCL_HARDSIGMOID_ਖੰਡ_ਮਾਪ 256
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ SYCL_HARDSWISH_ਖੰਡ_ਮਾਪ 256
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ SYCL_EXP_ਖੰਡ_ਮਾਪ 256
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ SYCL_NEG_ਖੰਡ_ਮਾਪ 256
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ SYCL_SIGMOID_ਖੰਡ_ਮਾਪ 256
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ SYCL_SQRT_ਖੰਡ_ਮਾਪ 256
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ SYCL_SIN_ਖੰਡ_ਮਾਪ 256
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ SYCL_SQR_ਖੰਡ_ਮਾਪ 256
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ SYCL_ਸੈੱਟ_ਖੰਡ_ਮਾਪ 256
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ SYCL_CPY_ਖੰਡ_ਮਾਪ 32
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ SYCL_SCALE_ਖੰਡ_ਮਾਪ 256
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ SYCL_CLAMP_ਖੰਡ_ਮਾਪ 256
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ SYCL_ROPE_ਖੰਡ_ਮਾਪ 256
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ SYCL_ALIBI_ਖੰਡ_ਮਾਪ 32
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ SYCL_DIAG_ਨਕਾਬ_INF_ਖੰਡ_ਮਾਪ 32
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ SYCL_QUANTIZE_ਖੰਡ_ਮਾਪ 256
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ SYCL_DEQUANTIZE_ਖੰਡ_ਮਾਪ 256
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ SYCL_GET_ਪੰਕਤੀਆਂ_ਖੰਡ_ਮਾਪ 256
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ SYCL_UPSCALE_ਖੰਡ_ਮਾਪ 256
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ SYCL_CONCAT_ਖੰਡ_ਮਾਪ 256
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ SYCL_PAD_ਖੰਡ_ਮਾਪ 256
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ SYCL_ACC_ਖੰਡ_ਮਾਪ 256
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ SYCL_IM2_ਖਾਨਾ_ਖੰਡ_ਮਾਪ 256
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ SYCL_POOL1_D_ਖੰਡ_ਮਾਪ 256
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ SYCL_POOL2_D_ਖੰਡ_ਮਾਪ 256
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ SYCL_ARGMAX_ਖੰਡ_ਮਾਪ 256
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ SYCL_CONV_TRANPOSE_1_D_ਖੰਡ_ਮਾਪ 256
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ SYCL_TIMESTEP_EMBEDDING_ਖੰਡ_ਮਾਪ 256
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ SYCL_ARANGE_ਖੰਡ_ਮਾਪ 256

// dmmv = dequantize_mul_mat_vec
#ਜੇ_ਅਪਰਿਭਾਸ਼ਿਤ GGML_SYCL_DMMV_X
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_SYCL_DMMV_X 32
#ਜੇ_ਅੰਤ
#ਜੇ_ਅਪਰਿਭਾਸ਼ਿਤ GGML_SYCL_MMV_Y
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_SYCL_MMV_Y 1
#ਜੇ_ਅੰਤ

#ਜੇ_ਅਪਰਿਭਾਸ਼ਿਤ K_QUANTS_ਪ੍ਰਤੀ_ITERATION
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ K_QUANTS_ਪ੍ਰਤੀ_ITERATION 1
#ਨਹੀਂਤਾਂ
static_assert(K_QUANTS_ਪ੍ਰਤੀ_ITERATION == 1 || K_QUANTS_ਪ੍ਰਤੀ_ITERATION == 2, "K_QUANTS_PER_ITERATION must be 1 or 2");
#ਜੇ_ਅੰਤ

#ਜੇ_ਅਪਰਿਭਾਸ਼ਿਤ GGML_SYCL_PEER_MAX_BATCH_ਮਾਪ
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_SYCL_PEER_MAX_BATCH_ਮਾਪ 128
#ਜੇ_ਅੰਤ // GGML_SYCL_PEER_MAX_BATCH_SIZE

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ MUL_MAT_SRC1_ਖਾਨਾ_STRIDE 128

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ QK_WARP_ਮਾਪ 32
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ WARP_32_ਮਾਪ 32
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ WARP_16_ਮਾਪ 16

#ਜੇ_ਅੰਤ // GGML_SYCL_PRESETS_HPP
