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

#அந்தர்பாவ "common.hpp"

வெறுமை ggml_sycl_op_mul_mat_q(
    ggml_backend_sycl_context & ctx,
    மாறிலி ggml_tensor* src0,
    மாறிலி ggml_tensor* src1,
    ggml_tensor* dst,
    மாறிலி எழுத்து* src0_dd_i,
    மாறிலி தசம* src1_ddf_i,
    மாறிலி எழுத்து* src1_ddq_i,
    தசம* dst_dd_i,
    மாறிலி int64_t row_கீழ்,
    மாறிலி int64_t row_உயர்வு,
    மாறிலி int64_t src1_ncols,
    மாறிலி int64_t src1_padded_row_size,
    மாறிலி dpct::queue_ptr& stream);

#endif // GGML_SYCL_MMQ_HPP
