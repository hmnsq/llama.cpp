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

#अगर_अघोषित GGML_SYCL_DMMV_HPP
#घोषणा GGML_SYCL_DMMV_HPP

#समावेश "common.hpp"


व्योम ggml_sycl_op_dequantize_mul_mat_vec(
    ggml_backend_sycl_context & ctx,
    स्थिर ggml_tensor *src0, स्थिर ggml_tensor *src1, ggml_tensor *dst,
    स्थिर अक्षर *src0_dd_i, स्थिर भग्न *src1_ddf_i, स्थिर अक्षर *src1_ddq_i,
    भग्न *dst_dd_i, स्थिर int64_t row_निचला, स्थिर int64_t row_ऊपरी,
    स्थिर int64_t src1_ncols, स्थिर int64_t src1_padded_row_आकार,
    स्थिर dpct::queue_ptr &stream);

// Fused dense-FFN mat-vec: writes glu(gate . y, up . y) for same-type reordered
// weights (Q2_K..Q6_K, Q8_0) and a single F32 activation column, using the ESIMD DMMV kernels.
// vx is the up weight, vgate the gate weight. Only defined when GGML_SYCL_DMMV_HAS_ESIMD;
// returns false if the type is unhandled.
bool ggml_sycl_dequantize_mul_mat_vec_glu_reorder_esimd(
    क्रमागत ggml_type src0_type, क्रमागत ggml_glu_op glu_op,
    स्थिर व्योम * vx, स्थिर व्योम * vgate, स्थिर भग्न * y,
    भग्न * dst, पूर्णांक ncols, पूर्णांक nrows, dpct::queue_ptr stream);

#पूर्ण_अगर // GGML_SYCL_DMMV_HPP
