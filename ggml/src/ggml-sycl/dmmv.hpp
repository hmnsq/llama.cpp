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

#અન્તર્ભાવ "common.hpp"


શૂન્ય ggml_sycl_op_dequantize_mul_mat_vec(
    ggml_backend_sycl_context & ctx,
    અચલ ggml_tensor *src0, અચલ ggml_tensor *src1, ggml_tensor *dst,
    અચલ અક્ષર *src0_dd_i, અચલ દશાંશ *src1_ddf_i, અચલ અક્ષર *src1_ddq_i,
    દશાંશ *dst_dd_i, અચલ int64_t row_નીચલું, અચલ int64_t row_ઉપલું,
    અચલ int64_t src1_ncols, અચલ int64_t src1_padded_row_size,
    અચલ dpct::queue_ptr &stream);

// Fused dense-FFN mat-vec: writes glu(gate . y, up . y) for same-type reordered
// weights (Q2_K..Q6_K, Q8_0) and a single F32 activation column, using the ESIMD DMMV kernels.
// vx is the up weight, vgate the gate weight. Only defined when GGML_SYCL_DMMV_HAS_ESIMD;
// returns false if the type is unhandled.
તાર્કિક ggml_sycl_dequantize_mul_mat_vec_glu_reorder_esimd(
    ગણના ggml_type src0_type, ગણના ggml_glu_op glu_op,
    અચલ શૂન્ય * vx, અચલ શૂન્ય * vgate, અચલ દશાંશ * y,
    દશાંશ * dst, પૂર્ણાંક ncols, પૂર્ણાંક nrows, dpct::queue_ptr stream);

#endif // GGML_SYCL_DMMV_HPP
