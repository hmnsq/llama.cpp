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

#यदि_अपरिभाषितम् GGML_SYCL_PRESETS_HPP
#परिभाषय GGML_SYCL_PRESETS_HPP

#परिभाषय GGML_SYCL_MAX_STREAMS       8
#परिभाषय GGML_SYCL_MAX_BUFFERS       256

#परिभाषय WARP_आकार GGML_SYCL_WARP_आकार
#परिभाषय MATRIX_ROW_PADDING 512 // last row of quant. matrices is a multiple of this to avoid out-of-bounds memory accesses

#परिभाषय SYCL_COL2_IM_1_D_खण्डम्_आकार 256
#परिभाषय SYCL_TANH_खण्डम्_आकार 256
#परिभाषय SYCL_RELU_खण्डम्_आकार 256
#परिभाषय SYCL_HARDSIGMOID_खण्डम्_आकार 256
#परिभाषय SYCL_HARDSWISH_खण्डम्_आकार 256
#परिभाषय SYCL_EXP_खण्डम्_आकार 256
#परिभाषय SYCL_NEG_खण्डम्_आकार 256
#परिभाषय SYCL_SIGMOID_खण्डम्_आकार 256
#परिभाषय SYCL_SQRT_खण्डम्_आकार 256
#परिभाषय SYCL_SIN_खण्डम्_आकार 256
#परिभाषय SYCL_SQR_खण्डम्_आकार 256
#परिभाषय SYCL_समुच्चयः_खण्डम्_आकार 256
#परिभाषय SYCL_CPY_खण्डम्_आकार 32
#परिभाषय SYCL_SCALE_खण्डम्_आकार 256
#परिभाषय SYCL_CLAMP_खण्डम्_आकार 256
#परिभाषय SYCL_ROPE_खण्डम्_आकार 256
#परिभाषय SYCL_ALIBI_खण्डम्_आकार 32
#परिभाषय SYCL_DIAG_आवरणम्_INF_खण्डम्_आकार 32
#परिभाषय SYCL_QUANTIZE_खण्डम्_आकार 256
#परिभाषय SYCL_DEQUANTIZE_खण्डम्_आकार 256
#परिभाषय SYCL_GET_पङ्क्तिकाः_खण्डम्_आकार 256
#परिभाषय SYCL_UPSCALE_खण्डम्_आकार 256
#परिभाषय SYCL_CONCAT_खण्डम्_आकार 256
#परिभाषय SYCL_PAD_खण्डम्_आकार 256
#परिभाषय SYCL_ACC_खण्डम्_आकार 256
#परिभाषय SYCL_IM2_COL_खण्डम्_आकार 256
#परिभाषय SYCL_POOL1_D_खण्डम्_आकार 256
#परिभाषय SYCL_POOL2_D_खण्डम्_आकार 256
#परिभाषय SYCL_ARGMAX_खण्डम्_आकार 256
#परिभाषय SYCL_CONV_TRANPOSE_1_D_खण्डम्_आकार 256
#परिभाषय SYCL_TIMESTEP_EMBEDDING_खण्डम्_आकार 256
#परिभाषय SYCL_ARANGE_खण्डम्_आकार 256

// dmmv = dequantize_mul_mat_vec
#यदि_अपरिभाषितम् GGML_SYCL_DMMV_X
#परिभाषय GGML_SYCL_DMMV_X 32
#यदि_अन्तः
#यदि_अपरिभाषितम् GGML_SYCL_MMV_Y
#परिभाषय GGML_SYCL_MMV_Y 1
#यदि_अन्तः

#यदि_अपरिभाषितम् K_QUANTS_प्रत्येकम्_ITERATION
#परिभाषय K_QUANTS_प्रत्येकम्_ITERATION 1
#अन्यथा
static_assert(K_QUANTS_प्रत्येकम्_ITERATION == 1 || K_QUANTS_प्रत्येकम्_ITERATION == 2, "K_QUANTS_PER_ITERATION must be 1 or 2");
#यदि_अन्तः

#यदि_अपरिभाषितम् GGML_SYCL_PEER_MAX_BATCH_आकार
#परिभाषय GGML_SYCL_PEER_MAX_BATCH_आकार 128
#यदि_अन्तः // GGML_SYCL_PEER_MAX_BATCH_SIZE

#परिभाषय MUL_MAT_SRC1_COL_STRIDE 128

#परिभाषय QK_WARP_आकार 32
#परिभाषय WARP_32_आकार 32
#परिभाषय WARP_16_आकार 16

#यदि_अन्तः // GGML_SYCL_PRESETS_HPP
