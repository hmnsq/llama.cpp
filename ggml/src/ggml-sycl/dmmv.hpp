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

#ଅନ୍ତର୍ଭାଵ "common.hpp"


ଶୂନ୍ୟ ggml_sycl_op_dequantize_mul_mat_vec(
    ggml_backend_sycl_context & ctx,
    ଅଚଳ ggml_tensor *src0, ଅଚଳ ggml_tensor *src1, ggml_tensor *dst,
    ଅଚଳ ଅକ୍ଷର *src0_dd_i, ଅଚଳ ଦଶମିକ *src1_ddf_i, ଅଚଳ ଅକ୍ଷର *src1_ddq_i,
    ଦଶମିକ *dst_dd_i, ଅଚଳ int64_t row_ତଳ, ଅଚଳ int64_t row_ଉପର,
    ଅଚଳ int64_t src1_ncols, ଅଚଳ int64_t src1_padded_row_size,
    ଅଚଳ dpct::queue_ptr &stream);

// Fused dense-FFN mat-vec: writes glu(gate . y, up . y) for same-type reordered
// weights (Q2_K..Q6_K, Q8_0) and a single F32 activation column, using the ESIMD DMMV kernels.
// vx is the up weight, vgate the gate weight. Only defined when GGML_SYCL_DMMV_HAS_ESIMD;
// returns false if the type is unhandled.
ତାର୍କିକ ggml_sycl_dequantize_mul_mat_vec_glu_reorder_esimd(
    ଗଣନା ggml_type src0_type, ଗଣନା ggml_glu_op glu_op,
    ଅଚଳ ଶୂନ୍ୟ * vx, ଅଚଳ ଶୂନ୍ୟ * vgate, ଅଚଳ ଦଶମିକ * y,
    ଦଶମିକ * dst, ପୂର୍ଣ୍ଣସଂଖ୍ୟା ncols, ପୂର୍ଣ୍ଣସଂଖ୍ୟା nrows, dpct::queue_ptr stream);

#endif // GGML_SYCL_DMMV_HPP
