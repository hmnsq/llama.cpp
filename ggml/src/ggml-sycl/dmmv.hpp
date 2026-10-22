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

#شامل "common.hpp"


تهی ggml_sycl_op_dequantize_mul_mat_vec(
    ggml_backend_sycl_context & ctx,
    ثابت ggml_tensor *src0, ثابت ggml_tensor *src1, ggml_tensor *dst,
    ثابت نویسه *src0_dd_i, ثابت اعشاری *src1_ddf_i, ثابت نویسه *src1_ddq_i,
    اعشاری *dst_dd_i, ثابت int64_t row_کم, ثابت int64_t row_بالا,
    ثابت int64_t src1_ncols, ثابت int64_t src1_padded_row_size,
    ثابت dpct::queue_ptr &stream);

// Fused dense-FFN mat-vec: writes glu(gate . y, up . y) for same-type reordered
// weights (Q2_K..Q6_K, Q8_0) and a single F32 activation column, using the ESIMD DMMV kernels.
// vx is the up weight, vgate the gate weight. Only defined when GGML_SYCL_DMMV_HAS_ESIMD;
// returns false if the type is unhandled.
منطقی ggml_sycl_dequantize_mul_mat_vec_glu_reorder_esimd(
    شمارش ggml_type src0_type, شمارش ggml_glu_op glu_op,
    ثابت تهی * vx, ثابت تهی * vgate, ثابت اعشاری * y,
    اعشاری * dst, صحیح ncols, صحیح nrows, dpct::queue_ptr stream);

#endif // GGML_SYCL_DMMV_HPP
