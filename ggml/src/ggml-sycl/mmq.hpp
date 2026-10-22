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

#অন্তর্ভাৱ "common.hpp"

শূন্য ggml_sycl_op_mul_mat_q(
    ggml_backend_sycl_context & ctx,
    ধ্রুৱ ggml_tensor* src0,
    ধ্রুৱ ggml_tensor* src1,
    ggml_tensor* dst,
    ধ্রুৱ ৱর্ণ* src0_dd_i,
    ধ্রুৱ ভিন্ন* src1_ddf_i,
    ধ্রুৱ ৱর্ণ* src1_ddq_i,
    ভিন্ন* dst_dd_i,
    ধ্রুৱ int64_t row_নিচেৰ,
    ধ্রুৱ int64_t row_উপৰেৰ,
    ধ্রুৱ int64_t src1_ncols,
    ধ্রুৱ int64_t src1_padded_row_size,
    ধ্রুৱ dpct::queue_ptr& stream);

#endif // GGML_SYCL_MMQ_HPP
