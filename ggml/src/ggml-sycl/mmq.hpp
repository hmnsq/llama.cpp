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
    अचर ggml_tensor* src0,
    अचर ggml_tensor* src1,
    ggml_tensor* dst,
    अचर वर्ण* src0_dd_i,
    अचर भिन्न* src1_ddf_i,
    अचर वर्ण* src1_ddq_i,
    भिन्न* dst_dd_i,
    अचर int64_t row_तल्लो,
    अचर int64_t row_माथिल्लो,
    अचर int64_t src1_ncols,
    अचर int64_t src1_padded_row_size,
    अचर dpct::queue_ptr& stream);

#endif // GGML_SYCL_MMQ_HPP
