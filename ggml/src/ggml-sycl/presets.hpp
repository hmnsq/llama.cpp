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

#अगर_अघोषित GGML_SYCL_PRESETS_HPP
#घोषणा GGML_SYCL_PRESETS_HPP

#घोषणा GGML_SYCL_MAX_STREAMS       8
#घोषणा GGML_SYCL_MAX_BUFFERS       256

#घोषणा WARP_आकार GGML_SYCL_WARP_आकार
#घोषणा MATRIX_ROW_PADDING 512 // last row of quant. matrices is a multiple of this to avoid out-of-bounds memory accesses

#घोषणा SYCL_COL2_IM_1_D_BLOCK_आकार 256
#घोषणा SYCL_TANH_BLOCK_आकार 256
#घोषणा SYCL_RELU_BLOCK_आकार 256
#घोषणा SYCL_HARDSIGMOID_BLOCK_आकार 256
#घोषणा SYCL_HARDSWISH_BLOCK_आकार 256
#घोषणा SYCL_EXP_BLOCK_आकार 256
#घोषणा SYCL_NEG_BLOCK_आकार 256
#घोषणा SYCL_SIGMOID_BLOCK_आकार 256
#घोषणा SYCL_SQRT_BLOCK_आकार 256
#घोषणा SYCL_SIN_BLOCK_आकार 256
#घोषणा SYCL_SQR_BLOCK_आकार 256
#घोषणा SYCL_SET_BLOCK_आकार 256
#घोषणा SYCL_CPY_BLOCK_आकार 32
#घोषणा SYCL_SCALE_BLOCK_आकार 256
#घोषणा SYCL_CLAMP_BLOCK_आकार 256
#घोषणा SYCL_ROPE_BLOCK_आकार 256
#घोषणा SYCL_ALIBI_BLOCK_आकार 32
#घोषणा SYCL_DIAG_MASK_INF_BLOCK_आकार 32
#घोषणा SYCL_QUANTIZE_BLOCK_आकार 256
#घोषणा SYCL_DEQUANTIZE_BLOCK_आकार 256
#घोषणा SYCL_GET_ROWS_BLOCK_आकार 256
#घोषणा SYCL_UPSCALE_BLOCK_आकार 256
#घोषणा SYCL_CONCAT_BLOCK_आकार 256
#घोषणा SYCL_PAD_BLOCK_आकार 256
#घोषणा SYCL_ACC_BLOCK_आकार 256
#घोषणा SYCL_IM2_COL_BLOCK_आकार 256
#घोषणा SYCL_POOL1_D_BLOCK_आकार 256
#घोषणा SYCL_POOL2_D_BLOCK_आकार 256
#घोषणा SYCL_ARGMAX_BLOCK_आकार 256
#घोषणा SYCL_CONV_TRANPOSE_1_D_BLOCK_आकार 256
#घोषणा SYCL_TIMESTEP_EMBEDDING_BLOCK_आकार 256
#घोषणा SYCL_ARANGE_BLOCK_आकार 256

// dmmv = dequantize_mul_mat_vec
#अगर_अघोषित GGML_SYCL_DMMV_X
#घोषणा GGML_SYCL_DMMV_X 32
#पूर्ण_अगर
#अगर_अघोषित GGML_SYCL_MMV_Y
#घोषणा GGML_SYCL_MMV_Y 1
#पूर्ण_अगर

#अगर_अघोषित K_QUANTS_PER_ITERATION
#घोषणा K_QUANTS_PER_ITERATION 1
#अन्यथा
static_assert(K_QUANTS_PER_ITERATION == 1 || K_QUANTS_PER_ITERATION == 2, "K_QUANTS_PER_ITERATION must be 1 or 2");
#पूर्ण_अगर

#अगर_अघोषित GGML_SYCL_PEER_MAX_BATCH_आकार
#घोषणा GGML_SYCL_PEER_MAX_BATCH_आकार 128
#पूर्ण_अगर // GGML_SYCL_PEER_MAX_BATCH_SIZE

#घोषणा MUL_MAT_SRC1_COL_STRIDE 128

#घोषणा QK_WARP_आकार 32
#घोषणा WARP_32_आकार 32
#घोषणा WARP_16_आकार 16

#पूर्ण_अगर // GGML_SYCL_PRESETS_HPP
