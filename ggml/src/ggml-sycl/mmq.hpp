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

#ଅନ୍ତର୍ଭାଵ "common.hpp"

ଶୂନ୍ୟ ggml_sycl_op_mul_mat_q(
    ggml_backend_sycl_context & ctx,
    ଅଚଳ ggml_tensor* src0,
    ଅଚଳ ggml_tensor* src1,
    ggml_tensor* dst,
    ଅଚଳ ଅକ୍ଷର* src0_dd_i,
    ଅଚଳ ଦଶମିକ* src1_ddf_i,
    ଅଚଳ ଅକ୍ଷର* src1_ddq_i,
    ଦଶମିକ* dst_dd_i,
    ଅଚଳ int64_t row_ତଳ,
    ଅଚଳ int64_t row_ଉପର,
    ଅଚଳ int64_t src1_ncols,
    ଅଚଳ int64_t src1_padded_row_size,
    ଅଚଳ dpct::queue_ptr& stream);

#endif // GGML_SYCL_MMQ_HPP
