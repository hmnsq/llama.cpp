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

#ਜੇ_ਅਪਰਿਭਾਸ਼ਿਤ GGML_SYCL_DMMV_HPP
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_SYCL_DMMV_HPP

#ਅਨ੍ਤਰ੍ਭਾਵ "common.hpp"


ਖਾਲੀ ggml_sycl_op_dequantize_mul_mat_vec(
    ggml_backend_sycl_context & ctx,
    ਅਚਲ ggml_tensor *src0, ਅਚਲ ggml_tensor *src1, ggml_tensor *dst,
    ਅਚਲ ਅੱਖਰ *src0_dd_i, ਅਚਲ ਦਸ਼ਮਲਵ *src1_ddf_i, ਅਚਲ ਅੱਖਰ *src1_ddq_i,
    ਦਸ਼ਮਲਵ *dst_dd_i, ਅਚਲ int64_t ਪੰਕਤੀ_ਨੀਵਾਂ, ਅਚਲ int64_t ਪੰਕਤੀ_ਉੱਚਾ,
    ਅਚਲ int64_t src1_ncols, ਅਚਲ int64_t src1_padded_ਪੰਕਤੀ_ਮਾਪ,
    ਅਚਲ dpct::ਕਤਾਰ_ptr &stream);

// Fused dense-FFN mat-vec: writes glu(gate . y, up . y) for same-type reordered
// weights (Q2_K..Q6_K, Q8_0) and a single F32 activation column, using the ESIMD DMMV kernels.
// vx is the up weight, vgate the gate weight. Only defined when GGML_SYCL_DMMV_HAS_ESIMD;
// returns false if the type is unhandled.
ਤਾਰਕਿਕ ggml_sycl_dequantize_mul_mat_vec_glu_reorder_esimd(
    ਗਿਣਤੀ ggml_type src0_type, ਗਿਣਤੀ ggml_glu_op glu_op,
    ਅਚਲ ਖਾਲੀ * vx, ਅਚਲ ਖਾਲੀ * vgate, ਅਚਲ ਦਸ਼ਮਲਵ * y,
    ਦਸ਼ਮਲਵ * dst, ਪੂਰਨ ncols, ਪੂਰਨ nrows, dpct::ਕਤਾਰ_ptr stream);

#ਜੇ_ਅੰਤ // GGML_SYCL_DMMV_HPP
