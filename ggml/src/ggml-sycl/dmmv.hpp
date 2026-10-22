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

#اگر_غیر_تعریف_شدہ GGML_SYCL_DMMV_HPP
#تعریف_کرو GGML_SYCL_DMMV_HPP

#شامل "common.hpp"


خالی ggml_sycl_op_dequantize_mul_mat_vec(
    ggml_backend_sycl_context & ctx,
    ثابت ggml_tensor *src0, ثابت ggml_tensor *src1, ggml_tensor *dst,
    ثابت حرف *src0_dd_i, ثابت اعشاری *src1_ddf_i, ثابت حرف *src1_ddq_i,
    اعشاری *dst_dd_i, ثابت int64_t صف_کم, ثابت int64_t صف_بلند,
    ثابت int64_t src1_ncols, ثابت int64_t src1_padded_صف_سائز,
    ثابت dpct::قطار_ptr &stream);

// Fused dense-FFN mat-vec: writes glu(gate . y, up . y) for same-type reordered
// weights (Q2_K..Q6_K, Q8_0) and a single F32 activation column, using the ESIMD DMMV kernels.
// vx is the up weight, vgate the gate weight. Only defined when GGML_SYCL_DMMV_HAS_ESIMD;
// returns false if the type is unhandled.
منطقی ggml_sycl_dequantize_mul_mat_vec_glu_reorder_esimd(
    شمار ggml_type src0_type, شمار ggml_glu_op glu_op,
    ثابت خالی * vx, ثابت خالی * vgate, ثابت اعشاری * y,
    اعشاری * dst, صحیح ncols, صحیح nrows, dpct::قطار_ptr stream);

#اگر_ختم // GGML_SYCL_DMMV_HPP
