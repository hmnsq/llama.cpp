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

#నిర్వచితం_కాకపోతే GGML_SYCL_DMMV_HPP
#నిర్వచించు GGML_SYCL_DMMV_HPP

#అన్తర్భావ "common.hpp"


శూన్య ggml_sycl_op_dequantize_mul_mat_vec(
    ggml_backend_sycl_context & ctx,
    స్థిరరాశి ggml_tensor *src0, స్థిరరాశి ggml_tensor *src1, ggml_tensor *dst,
    స్థిరరాశి అక్షరం *src0_dd_i, స్థిరరాశి దశాంశం *src1_ddf_i, స్థిరరాశి అక్షరం *src1_ddq_i,
    దశాంశం *dst_dd_i, స్థిరరాశి int64_t అడ్డువరుస_దిగువ, స్థిరరాశి int64_t అడ్డువరుస_ఎగువ,
    స్థిరరాశి int64_t src1_ncols, స్థిరరాశి int64_t src1_padded_అడ్డువరుస_కొలత,
    స్థిరరాశి dpct::వరుస_ptr &stream);

// Fused dense-FFN mat-vec: writes glu(gate . y, up . y) for same-type reordered
// weights (Q2_K..Q6_K, Q8_0) and a single F32 activation column, using the ESIMD DMMV kernels.
// vx is the up weight, vgate the gate weight. Only defined when GGML_SYCL_DMMV_HAS_ESIMD;
// returns false if the type is unhandled.
తార్కిక ggml_sycl_dequantize_mul_mat_vec_glu_reorder_esimd(
    గణన ggml_type src0_type, గణన ggml_glu_op glu_op,
    స్థిరరాశి శూన్య * vx, స్థిరరాశి శూన్య * vgate, స్థిరరాశి దశాంశం * y,
    దశాంశం * dst, పూర్ణాంకం ncols, పూర్ణాంకం nrows, dpct::వరుస_ptr stream);

#అయితే_ముగింపు // GGML_SYCL_DMMV_HPP
