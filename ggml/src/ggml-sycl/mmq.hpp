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

#अन्तर्भाव "common.hpp"

शून्य ggml_sycl_op_mul_mat_q(
    ggml_backend_sycl_context & ctx,
    स्थिरांक ggml_tensor* src0,
    स्थिरांक ggml_tensor* src1,
    ggml_tensor* dst,
    स्थिरांक अक्षर* src0_dd_i,
    स्थिरांक दशांश* src1_ddf_i,
    स्थिरांक अक्षर* src1_ddq_i,
    दशांश* dst_dd_i,
    स्थिरांक int64_t row_खालचा,
    स्थिरांक int64_t row_वरचा,
    स्थिरांक int64_t src1_ncols,
    स्थिरांक int64_t src1_padded_row_size,
    स्थिरांक dpct::queue_ptr& stream);

#endif // GGML_SYCL_MMQ_HPP
