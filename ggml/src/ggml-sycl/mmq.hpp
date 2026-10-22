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

#অন্তর্ভাব "common.hpp"

শূন্য ggml_sycl_op_mul_mat_q(
    ggml_backend_sycl_context & ctx,
    ধ্রুবক ggml_tensor* src0,
    ধ্রুবক ggml_tensor* src1,
    ggml_tensor* dst,
    ধ্রুবক অক্ষর* src0_dd_i,
    ধ্রুবক দশমিক* src1_ddf_i,
    ধ্রুবক অক্ষর* src1_ddq_i,
    দশমিক* dst_dd_i,
    ধ্রুবক int64_t row_নিচের,
    ধ্রুবক int64_t row_উপরের,
    ধ্রুবক int64_t src1_ncols,
    ধ্রুবক int64_t src1_padded_row_আকার,
    ধ্রুবক dpct::queue_ptr& stream);

#endif // GGML_SYCL_MMQ_HPP
