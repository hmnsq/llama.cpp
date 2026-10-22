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

#ifndef GGML_SYCL_DMMV_HPP
#define GGML_SYCL_DMMV_HPP

#ಅನ್ತರ್ಭಾವ "common.hpp"


ಶೂನ್ಯ ggml_sycl_op_dequantize_mul_mat_vec(
    ggml_backend_sycl_context & ctx,
    ಧ್ರುವ ggml_tensor *src0, ಧ್ರುವ ggml_tensor *src1, ggml_tensor *dst,
    ಧ್ರುವ ವರ್ಣ *src0_dd_i, ಧ್ರುವ ಭಿನ್ನ *src1_ddf_i, ಧ್ರುವ ವರ್ಣ *src1_ddq_i,
    ಭಿನ್ನ *dst_dd_i, ಧ್ರುವ int64_t row_ಕೆಳಗಿನ, ಧ್ರುವ int64_t row_ಮೇಲಿನ,
    ಧ್ರುವ int64_t src1_ncols, ಧ್ರುವ int64_t src1_padded_row_size,
    ಧ್ರುವ dpct::queue_ptr &stream);

// Fused dense-FFN mat-vec: writes glu(gate . y, up . y) for same-type reordered
// weights (Q2_K..Q6_K, Q8_0) and a single F32 activation column, using the ESIMD DMMV kernels.
// vx is the up weight, vgate the gate weight. Only defined when GGML_SYCL_DMMV_HAS_ESIMD;
// returns false if the type is unhandled.
ತಾರ್ಕಿಕ ggml_sycl_dequantize_mul_mat_vec_glu_reorder_esimd(
    ಗಣನಾ ggml_type src0_type, ಗಣನಾ ggml_glu_op glu_op,
    ಧ್ರುವ ಶೂನ್ಯ * vx, ಧ್ರುವ ಶೂನ್ಯ * vgate, ಧ್ರುವ ಭಿನ್ನ * y,
    ಭಿನ್ನ * dst, ಪೂರ್ಣಾಙ್ಕ ncols, ಪೂರ್ಣಾಙ್ಕ nrows, dpct::queue_ptr stream);

#endif // GGML_SYCL_DMMV_HPP
