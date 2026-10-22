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

#నిర్వచితం_కాకపోతే GGML_SYCL_MMQ_HPP
#నిర్వచించు GGML_SYCL_MMQ_HPP

#అన్తర్భావ "common.hpp"

శూన్య ggml_sycl_op_mul_mat_q(
    ggml_backend_sycl_context & ctx,
    స్థిరరాశి ggml_tensor* src0,
    స్థిరరాశి ggml_tensor* src1,
    ggml_tensor* dst,
    స్థిరరాశి అక్షరం* src0_dd_i,
    స్థిరరాశి దశాంశం* src1_ddf_i,
    స్థిరరాశి అక్షరం* src1_ddq_i,
    దశాంశం* dst_dd_i,
    స్థిరరాశి int64_t అడ్డువరుస_దిగువ,
    స్థిరరాశి int64_t అడ్డువరుస_ఎగువ,
    స్థిరరాశి int64_t src1_ncols,
    స్థిరరాశి int64_t src1_padded_అడ్డువరుస_కొలత,
    స్థిరరాశి dpct::వరుస_ptr& stream);

#అయితే_ముగింపు // GGML_SYCL_MMQ_HPP
