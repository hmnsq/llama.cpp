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

#اگر_تعریف_نشده GGML_SYCL_DMMV_HPP
#تعریف_کن GGML_SYCL_DMMV_HPP

#ضمن "common.hpp"


فراغ ggml_sycl_op_dequantize_mul_mat_vec(
    ggml_backend_sycl_context & ctx,
    ثابت ggml_tensor *src0, ثابت ggml_tensor *src1, ggml_tensor *dst,
    ثابت حرف *src0_dd_i, ثابت عائم *src1_ddf_i, ثابت حرف *src1_ddq_i,
    عائم *dst_dd_i, ثابت int64_t ردیف_کم, ثابت int64_t ردیف_بالا,
    ثابت int64_t src1_ncols, ثابت int64_t src1_padded_ردیف_اندازه,
    ثابت dpct::صف_ptr &stream);

// Fused dense-FFN mat-vec: writes glu(gate . y, up . y) for same-type reordered
// weights (Q2_K..Q6_K, Q8_0) and a single F32 activation column, using the ESIMD DMMV kernels.
// vx is the up weight, vgate the gate weight. Only defined when GGML_SYCL_DMMV_HAS_ESIMD;
// returns false if the type is unhandled.
منطقي ggml_sycl_dequantize_mul_mat_vec_glu_reorder_esimd(
    تعداد ggml_type src0_type, تعداد ggml_glu_op glu_op,
    ثابت فراغ * vx, ثابت فراغ * vgate, ثابت عائم * y,
    عائم * dst, صحيح ncols, صحيح nrows, dpct::صف_ptr stream);

#پایان_اگر // GGML_SYCL_DMMV_HPP
