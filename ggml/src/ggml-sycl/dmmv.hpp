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

#അന്തര്ഭാവ "common.hpp"


ശൂന്യം ggml_sycl_op_dequantize_mul_mat_vec(
    ggml_backend_sycl_context & ctx,
    സ്ഥിരം ggml_tensor *src0, സ്ഥിരം ggml_tensor *src1, ggml_tensor *dst,
    സ്ഥിരം അക്ഷരം *src0_dd_i, സ്ഥിരം ദശാംശം *src1_ddf_i, സ്ഥിരം അക്ഷരം *src1_ddq_i,
    ദശാംശം *dst_dd_i, സ്ഥിരം int64_t row_താഴെയുള്ള, സ്ഥിരം int64_t row_മുകളിലെ,
    സ്ഥിരം int64_t src1_ncols, സ്ഥിരം int64_t src1_padded_row_size,
    സ്ഥിരം dpct::queue_ptr &stream);

// Fused dense-FFN mat-vec: writes glu(gate . y, up . y) for same-type reordered
// weights (Q2_K..Q6_K, Q8_0) and a single F32 activation column, using the ESIMD DMMV kernels.
// vx is the up weight, vgate the gate weight. Only defined when GGML_SYCL_DMMV_HAS_ESIMD;
// returns false if the type is unhandled.
തർക്കം ggml_sycl_dequantize_mul_mat_vec_glu_reorder_esimd(
    എണ്ണൽ ggml_type src0_type, എണ്ണൽ ggml_glu_op glu_op,
    സ്ഥിരം ശൂന്യം * vx, സ്ഥിരം ശൂന്യം * vgate, സ്ഥിരം ദശാംശം * y,
    ദശാംശം * dst, പൂർണ്ണസംഖ്യ ncols, പൂർണ്ണസംഖ്യ nrows, dpct::queue_ptr stream);

#endif // GGML_SYCL_DMMV_HPP
