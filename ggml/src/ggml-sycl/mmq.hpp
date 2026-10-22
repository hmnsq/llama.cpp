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

#ಅನ್ತರ್ಭಾವ "common.hpp"

ಶೂನ್ಯ ggml_sycl_op_mul_mat_q(
    ggml_backend_sycl_context & ctx,
    ಧ್ರುವ ggml_tensor* src0,
    ಧ್ರುವ ggml_tensor* src1,
    ggml_tensor* dst,
    ಧ್ರುವ ವರ್ಣ* src0_dd_i,
    ಧ್ರುವ ಭಿನ್ನ* src1_ddf_i,
    ಧ್ರುವ ವರ್ಣ* src1_ddq_i,
    ಭಿನ್ನ* dst_dd_i,
    ಧ್ರುವ int64_t row_ಕೆಳಗಿನ,
    ಧ್ರುವ int64_t row_ಮೇಲಿನ,
    ಧ್ರುವ int64_t src1_ncols,
    ಧ್ರುವ int64_t src1_padded_row_size,
    ಧ್ರುವ dpct::queue_ptr& stream);

#endif // GGML_SYCL_MMQ_HPP
