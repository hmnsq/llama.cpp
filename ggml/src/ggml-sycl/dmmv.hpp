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

#ವ್ಯಾಖ್ಯಾನಿತವಾಗದಿದ್ದರೆ GGML_SYCL_DMMV_HPP
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_SYCL_DMMV_HPP

#ಅನ್ತರ್ಭಾವ "common.hpp"


ಶೂನ್ಯ ggml_sycl_op_dequantize_mul_mat_vec(
    ggml_backend_sycl_context & ctx,
    ಸ್ಥಿರಾಂಕ ggml_tensor *src0, ಸ್ಥಿರಾಂಕ ggml_tensor *src1, ggml_tensor *dst,
    ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *src0_dd_i, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *src1_ddf_i, ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *src1_ddq_i,
    ದಶಮಾಂಶ *dst_dd_i, ಸ್ಥಿರಾಂಕ int64_t ಅಡ್ಡಸಾಲು_ಕೆಳಗಿನ, ಸ್ಥಿರಾಂಕ int64_t ಅಡ್ಡಸಾಲು_ಮೇಲಿನ,
    ಸ್ಥಿರಾಂಕ int64_t src1_ncols, ಸ್ಥಿರಾಂಕ int64_t src1_padded_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ,
    ಸ್ಥಿರಾಂಕ dpct::ಸರತಿ_ptr &stream);

// Fused dense-FFN mat-vec: writes glu(gate . y, up . y) for same-type reordered
// weights (Q2_K..Q6_K, Q8_0) and a single F32 activation column, using the ESIMD DMMV kernels.
// vx is the up weight, vgate the gate weight. Only defined when GGML_SYCL_DMMV_HAS_ESIMD;
// returns false if the type is unhandled.
ತಾರ್ಕಿಕ ggml_sycl_dequantize_mul_mat_vec_glu_reorder_esimd(
    ಎಣಿಕೆ ggml_type src0_type, ಎಣಿಕೆ ggml_glu_op glu_op,
    ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * vx, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * vgate, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * y,
    ದಶಮಾಂಶ * dst, ಪೂರ್ಣಾಂಕ ncols, ಪೂರ್ಣಾಂಕ nrows, dpct::ಸರತಿ_ptr stream);

#ಆದರೆ_ಅಂತ್ಯ // GGML_SYCL_DMMV_HPP
