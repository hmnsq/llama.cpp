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

#അന്തര്ഭാവ "common.hpp"

ശൂന്യം ggml_sycl_op_mul_mat_q(
    ggml_backend_sycl_context & ctx,
    സ്ഥിരം ggml_tensor* src0,
    സ്ഥിരം ggml_tensor* src1,
    ggml_tensor* dst,
    സ്ഥിരം അക്ഷരം* src0_dd_i,
    സ്ഥിരം ദശാംശം* src1_ddf_i,
    സ്ഥിരം അക്ഷരം* src1_ddq_i,
    ദശാംശം* dst_dd_i,
    സ്ഥിരം int64_t row_താഴെയുള്ള,
    സ്ഥിരം int64_t row_മുകളിലെ,
    സ്ഥിരം int64_t src1_ncols,
    സ്ഥിരം int64_t src1_padded_row_size,
    സ്ഥിരം dpct::queue_ptr& stream);

#endif // GGML_SYCL_MMQ_HPP
