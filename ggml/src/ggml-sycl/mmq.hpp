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

#ifndef GGML_SYCL_MMQ_HPP
#define GGML_SYCL_MMQ_HPP

#ضمن "common.hpp"

فراغ ggml_sycl_op_mul_mat_q(
    ggml_backend_sycl_context & ctx,
    ثابت ggml_tensor* src0,
    ثابت ggml_tensor* src1,
    ggml_tensor* dst,
    ثابت حرف* src0_dd_i,
    ثابت عائم* src1_ddf_i,
    ثابت حرف* src1_ddq_i,
    عائم* dst_dd_i,
    ثابت int64_t row_نزم,
    ثابت int64_t row_بەرز,
    ثابت int64_t src1_ncols,
    ثابت int64_t src1_padded_row_size,
    ثابت dpct::queue_ptr& stream);

#endif // GGML_SYCL_MMQ_HPP
