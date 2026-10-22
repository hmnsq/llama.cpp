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

#నిర్వచితం_కాకపోతే GGML_SYCL_PRESETS_HPP
#నిర్వచించు GGML_SYCL_PRESETS_HPP

#నిర్వచించు GGML_SYCL_MAX_STREAMS       8
#నిర్వచించు GGML_SYCL_MAX_BUFFERS       256

#నిర్వచించు WARP_కొలత GGML_SYCL_WARP_కొలత
#నిర్వచించు మాతృక_అడ్డువరుస_PADDING 512 // last row of quant. matrices is a multiple of this to avoid out-of-bounds memory accesses

#నిర్వచించు SYCL_COL2_IM_1_D_ఖండం_కొలత 256
#నిర్వచించు SYCL_TANH_ఖండం_కొలత 256
#నిర్వచించు SYCL_RELU_ఖండం_కొలత 256
#నిర్వచించు SYCL_HARDSIGMOID_ఖండం_కొలత 256
#నిర్వచించు SYCL_HARDSWISH_ఖండం_కొలత 256
#నిర్వచించు SYCL_EXP_ఖండం_కొలత 256
#నిర్వచించు SYCL_NEG_ఖండం_కొలత 256
#నిర్వచించు SYCL_SIGMOID_ఖండం_కొలత 256
#నిర్వచించు SYCL_SQRT_ఖండం_కొలత 256
#నిర్వచించు SYCL_SIN_ఖండం_కొలత 256
#నిర్వచించు SYCL_SQR_ఖండం_కొలత 256
#నిర్వచించు SYCL_సమితి_ఖండం_కొలత 256
#నిర్వచించు SYCL_CPY_ఖండం_కొలత 32
#నిర్వచించు SYCL_SCALE_ఖండం_కొలత 256
#నిర్వచించు SYCL_CLAMP_ఖండం_కొలత 256
#నిర్వచించు SYCL_ROPE_ఖండం_కొలత 256
#నిర్వచించు SYCL_ALIBI_ఖండం_కొలత 32
#నిర్వచించు SYCL_DIAG_ముసుగు_INF_ఖండం_కొలత 32
#నిర్వచించు SYCL_QUANTIZE_ఖండం_కొలత 256
#నిర్వచించు SYCL_DEQUANTIZE_ఖండం_కొలత 256
#నిర్వచించు SYCL_GET_అడ్డువరుసలు_ఖండం_కొలత 256
#నిర్వచించు SYCL_UPSCALE_ఖండం_కొలత 256
#నిర్వచించు SYCL_CONCAT_ఖండం_కొలత 256
#నిర్వచించు SYCL_PAD_ఖండం_కొలత 256
#నిర్వచించు SYCL_ACC_ఖండం_కొలత 256
#నిర్వచించు SYCL_IM2_నిలువు_ఖండం_కొలత 256
#నిర్వచించు SYCL_POOL1_D_ఖండం_కొలత 256
#నిర్వచించు SYCL_POOL2_D_ఖండం_కొలత 256
#నిర్వచించు SYCL_ARGMAX_ఖండం_కొలత 256
#నిర్వచించు SYCL_CONV_TRANPOSE_1_D_ఖండం_కొలత 256
#నిర్వచించు SYCL_TIMESTEP_EMBEDDING_ఖండం_కొలత 256
#నిర్వచించు SYCL_ARANGE_ఖండం_కొలత 256

// dmmv = dequantize_mul_mat_vec
#నిర్వచితం_కాకపోతే GGML_SYCL_DMMV_X
#నిర్వచించు GGML_SYCL_DMMV_X 32
#అయితే_ముగింపు
#నిర్వచితం_కాకపోతే GGML_SYCL_MMV_Y
#నిర్వచించు GGML_SYCL_MMV_Y 1
#అయితే_ముగింపు

#నిర్వచితం_కాకపోతే K_QUANTS_చొప్పున_ITERATION
#నిర్వచించు K_QUANTS_చొప్పున_ITERATION 1
#లేకపోతే
static_assert(K_QUANTS_చొప్పున_ITERATION == 1 || K_QUANTS_చొప్పున_ITERATION == 2, "K_QUANTS_PER_ITERATION must be 1 or 2");
#అయితే_ముగింపు

#నిర్వచితం_కాకపోతే GGML_SYCL_PEER_MAX_BATCH_కొలత
#నిర్వచించు GGML_SYCL_PEER_MAX_BATCH_కొలత 128
#అయితే_ముగింపు // GGML_SYCL_PEER_MAX_BATCH_SIZE

#నిర్వచించు MUL_MAT_SRC1_నిలువు_STRIDE 128

#నిర్వచించు QK_WARP_కొలత 32
#నిర్వచించు WARP_32_కొలత 32
#నిర్వచించు WARP_16_కొలత 16

#అయితే_ముగింపు // GGML_SYCL_PRESETS_HPP
