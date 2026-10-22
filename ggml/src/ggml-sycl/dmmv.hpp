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

#অন্তর্ভাব "common.hpp"


শূন্য ggml_sycl_op_dequantize_mul_mat_vec(
    ggml_backend_sycl_context & ctx,
    ধ্রুব ggml_tensor *src0, ধ্রুব ggml_tensor *src1, ggml_tensor *dst,
    ধ্রুব বর্ণ *src0_dd_i, ধ্রুব ভিন্ন *src1_ddf_i, ধ্রুব বর্ণ *src1_ddq_i,
    ভিন্ন *dst_dd_i, ধ্রুব int64_t row_নিচের, ধ্রুব int64_t row_উপরের,
    ধ্রুব int64_t src1_ncols, ধ্রুব int64_t src1_padded_row_size,
    ধ্রুব dpct::queue_ptr &stream);

// Fused dense-FFN mat-vec: writes glu(gate . y, up . y) for same-type reordered
// weights (Q2_K..Q6_K, Q8_0) and a single F32 activation column, using the ESIMD DMMV kernels.
// vx is the up weight, vgate the gate weight. Only defined when GGML_SYCL_DMMV_HAS_ESIMD;
// returns false if the type is unhandled.
তার্কিক ggml_sycl_dequantize_mul_mat_vec_glu_reorder_esimd(
    গণনা ggml_type src0_type, গণনা ggml_glu_op glu_op,
    ধ্রুব শূন্য * vx, ধ্রুব শূন্য * vgate, ধ্রুব ভিন্ন * y,
    ভিন্ন * dst, পূর্ণাঙ্ক ncols, পূর্ণাঙ্ক nrows, dpct::queue_ptr stream);

#endif // GGML_SYCL_DMMV_HPP
