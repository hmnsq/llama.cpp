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

#अन्तर्भाव "common.hpp"


शून्य ggml_sycl_op_dequantize_mul_mat_vec(
    ggml_backend_sycl_context & ctx,
    स्थिरांक ggml_tensor *src0, स्थिरांक ggml_tensor *src1, ggml_tensor *dst,
    स्थिरांक अक्षर *src0_dd_i, स्थिरांक दशांश *src1_ddf_i, स्थिरांक अक्षर *src1_ddq_i,
    दशांश *dst_dd_i, स्थिरांक int64_t row_खालचा, स्थिरांक int64_t row_वरचा,
    स्थिरांक int64_t src1_ncols, स्थिरांक int64_t src1_padded_row_size,
    स्थिरांक dpct::queue_ptr &stream);

// Fused dense-FFN mat-vec: writes glu(gate . y, up . y) for same-type reordered
// weights (Q2_K..Q6_K, Q8_0) and a single F32 activation column, using the ESIMD DMMV kernels.
// vx is the up weight, vgate the gate weight. Only defined when GGML_SYCL_DMMV_HAS_ESIMD;
// returns false if the type is unhandled.
तार्किक ggml_sycl_dequantize_mul_mat_vec_glu_reorder_esimd(
    गणना ggml_type src0_type, गणना ggml_glu_op glu_op,
    स्थिरांक शून्य * vx, स्थिरांक शून्य * vgate, स्थिरांक दशांश * y,
    दशांश * dst, पूर्णांक ncols, पूर्णांक nrows, dpct::queue_ptr stream);

#endif // GGML_SYCL_DMMV_HPP
