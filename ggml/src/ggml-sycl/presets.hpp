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

#ವ್ಯಾಖ್ಯಾನಿತವಾಗದಿದ್ದರೆ GGML_SYCL_PRESETS_HPP
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_SYCL_PRESETS_HPP

#ವ್ಯಾಖ್ಯಾನಿಸು GGML_SYCL_MAX_STREAMS       8
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_SYCL_MAX_BUFFERS       256

#ವ್ಯಾಖ್ಯಾನಿಸು WARP_ಗಾತ್ರ GGML_SYCL_WARP_ಗಾತ್ರ
#ವ್ಯಾಖ್ಯಾನಿಸು ಮಾತೃಕೆ_ಅಡ್ಡಸಾಲು_PADDING 512 // last row of quant. matrices is a multiple of this to avoid out-of-bounds memory accesses

#ವ್ಯಾಖ್ಯಾನಿಸು SYCL_COL2_IM_1_D_ಖಂಡ_ಗಾತ್ರ 256
#ವ್ಯಾಖ್ಯಾನಿಸು SYCL_TANH_ಖಂಡ_ಗಾತ್ರ 256
#ವ್ಯಾಖ್ಯಾನಿಸು SYCL_RELU_ಖಂಡ_ಗಾತ್ರ 256
#ವ್ಯಾಖ್ಯಾನಿಸು SYCL_HARDSIGMOID_ಖಂಡ_ಗಾತ್ರ 256
#ವ್ಯಾಖ್ಯಾನಿಸು SYCL_HARDSWISH_ಖಂಡ_ಗಾತ್ರ 256
#ವ್ಯಾಖ್ಯಾನಿಸು SYCL_EXP_ಖಂಡ_ಗಾತ್ರ 256
#ವ್ಯಾಖ್ಯಾನಿಸು SYCL_NEG_ಖಂಡ_ಗಾತ್ರ 256
#ವ್ಯಾಖ್ಯಾನಿಸು SYCL_SIGMOID_ಖಂಡ_ಗಾತ್ರ 256
#ವ್ಯಾಖ್ಯಾನಿಸು SYCL_SQRT_ಖಂಡ_ಗಾತ್ರ 256
#ವ್ಯಾಖ್ಯಾನಿಸು SYCL_SIN_ಖಂಡ_ಗಾತ್ರ 256
#ವ್ಯಾಖ್ಯಾನಿಸು SYCL_SQR_ಖಂಡ_ಗಾತ್ರ 256
#ವ್ಯಾಖ್ಯಾನಿಸು SYCL_ಗಣ_ಖಂಡ_ಗಾತ್ರ 256
#ವ್ಯಾಖ್ಯಾನಿಸು SYCL_CPY_ಖಂಡ_ಗಾತ್ರ 32
#ವ್ಯಾಖ್ಯಾನಿಸು SYCL_SCALE_ಖಂಡ_ಗಾತ್ರ 256
#ವ್ಯಾಖ್ಯಾನಿಸು SYCL_CLAMP_ಖಂಡ_ಗಾತ್ರ 256
#ವ್ಯಾಖ್ಯಾನಿಸು SYCL_ROPE_ಖಂಡ_ಗಾತ್ರ 256
#ವ್ಯಾಖ್ಯಾನಿಸು SYCL_ALIBI_ಖಂಡ_ಗಾತ್ರ 32
#ವ್ಯಾಖ್ಯಾನಿಸು SYCL_DIAG_ಮುಖವಾಡ_INF_ಖಂಡ_ಗಾತ್ರ 32
#ವ್ಯಾಖ್ಯಾನಿಸು SYCL_QUANTIZE_ಖಂಡ_ಗಾತ್ರ 256
#ವ್ಯಾಖ್ಯಾನಿಸು SYCL_DEQUANTIZE_ಖಂಡ_ಗಾತ್ರ 256
#ವ್ಯಾಖ್ಯಾನಿಸು SYCL_GET_ಅಡ್ಡಸಾಲುಗಳು_ಖಂಡ_ಗಾತ್ರ 256
#ವ್ಯಾಖ್ಯಾನಿಸು SYCL_UPSCALE_ಖಂಡ_ಗಾತ್ರ 256
#ವ್ಯಾಖ್ಯಾನಿಸು SYCL_CONCAT_ಖಂಡ_ಗಾತ್ರ 256
#ವ್ಯಾಖ್ಯಾನಿಸು SYCL_PAD_ಖಂಡ_ಗಾತ್ರ 256
#ವ್ಯಾಖ್ಯಾನಿಸು SYCL_ACC_ಖಂಡ_ಗಾತ್ರ 256
#ವ್ಯಾಖ್ಯಾನಿಸು SYCL_IM2_ಲಂಬಸಾಲು_ಖಂಡ_ಗಾತ್ರ 256
#ವ್ಯಾಖ್ಯಾನಿಸು SYCL_POOL1_D_ಖಂಡ_ಗಾತ್ರ 256
#ವ್ಯಾಖ್ಯಾನಿಸು SYCL_POOL2_D_ಖಂಡ_ಗಾತ್ರ 256
#ವ್ಯಾಖ್ಯಾನಿಸು SYCL_ARGMAX_ಖಂಡ_ಗಾತ್ರ 256
#ವ್ಯಾಖ್ಯಾನಿಸು SYCL_CONV_TRANPOSE_1_D_ಖಂಡ_ಗಾತ್ರ 256
#ವ್ಯಾಖ್ಯಾನಿಸು SYCL_TIMESTEP_EMBEDDING_ಖಂಡ_ಗಾತ್ರ 256
#ವ್ಯಾಖ್ಯಾನಿಸು SYCL_ARANGE_ಖಂಡ_ಗಾತ್ರ 256

// dmmv = dequantize_mul_mat_vec
#ವ್ಯಾಖ್ಯಾನಿತವಾಗದಿದ್ದರೆ GGML_SYCL_DMMV_X
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_SYCL_DMMV_X 32
#ಆದರೆ_ಅಂತ್ಯ
#ವ್ಯಾಖ್ಯಾನಿತವಾಗದಿದ್ದರೆ GGML_SYCL_MMV_Y
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_SYCL_MMV_Y 1
#ಆದರೆ_ಅಂತ್ಯ

#ವ್ಯಾಖ್ಯಾನಿತವಾಗದಿದ್ದರೆ K_QUANTS_ಪ್ರತಿ_ITERATION
#ವ್ಯಾಖ್ಯಾನಿಸು K_QUANTS_ಪ್ರತಿ_ITERATION 1
#ಇಲ್ಲದಿದ್ದರೆ
static_assert(K_QUANTS_ಪ್ರತಿ_ITERATION == 1 || K_QUANTS_ಪ್ರತಿ_ITERATION == 2, "K_QUANTS_PER_ITERATION must be 1 or 2");
#ಆದರೆ_ಅಂತ್ಯ

#ವ್ಯಾಖ್ಯಾನಿತವಾಗದಿದ್ದರೆ GGML_SYCL_PEER_MAX_BATCH_ಗಾತ್ರ
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_SYCL_PEER_MAX_BATCH_ಗಾತ್ರ 128
#ಆದರೆ_ಅಂತ್ಯ // GGML_SYCL_PEER_MAX_BATCH_SIZE

#ವ್ಯಾಖ್ಯಾನಿಸು MUL_MAT_SRC1_ಲಂಬಸಾಲು_STRIDE 128

#ವ್ಯಾಖ್ಯಾನಿಸು QK_WARP_ಗಾತ್ರ 32
#ವ್ಯಾಖ್ಯಾನಿಸು WARP_32_ಗಾತ್ರ 32
#ವ್ಯಾಖ್ಯಾನಿಸು WARP_16_ಗಾತ್ರ 16

#ಆದರೆ_ಅಂತ್ಯ // GGML_SYCL_PRESETS_HPP
