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
    ध्रुव ggml_tensor *src0, ध्रुव ggml_tensor *src1, ggml_tensor *dst,
    ध्रुव वर्ण *src0_dd_i, ध्रुव भिन्न *src1_ddf_i, ध्रुव वर्ण *src1_ddq_i,
    भिन्न *dst_dd_i, ध्रुव int64_t row_खालचा, ध्रुव int64_t row_वरचा,
    ध्रुव int64_t src1_ncols, ध्रुव int64_t src1_padded_row_size,
    ध्रुव dpct::queue_ptr &stream);

// Fused dense-FFN mat-vec: writes glu(gate . y, up . y) for same-type reordered
// weights (Q2_K..Q6_K, Q8_0) and a single F32 activation column, using the ESIMD DMMV kernels.
// vx is the up weight, vgate the gate weight. Only defined when GGML_SYCL_DMMV_HAS_ESIMD;
// returns false if the type is unhandled.
तार्किक ggml_sycl_dequantize_mul_mat_vec_glu_reorder_esimd(
    गणना ggml_type src0_type, गणना ggml_glu_op glu_op,
    ध्रुव शून्य * vx, ध्रुव शून्य * vgate, ध्रुव भिन्न * y,
    भिन्न * dst, पूर्णाङ्क ncols, पूर्णाङ्क nrows, dpct::queue_ptr stream);

#endif // GGML_SYCL_DMMV_HPP
