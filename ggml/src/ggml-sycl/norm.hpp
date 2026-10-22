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

#నిర్వచితం_కాకపోతే GGML_SYCL_NORM_HPP
#నిర్వచించు GGML_SYCL_NORM_HPP

#అన్తర్భావ "common.hpp"

శూన్య ggml_sycl_op_norm(ggml_backend_sycl_context& ctx, ggml_tensor* dst);

శూన్య ggml_sycl_op_rms_norm(ggml_backend_sycl_context& ctx, ggml_tensor* dst);

శూన్య ggml_sycl_op_rms_norm_fused(ggml_backend_sycl_context& ctx, ggml_tensor* dst, ggml_tensor* mul);

శూన్య ggml_sycl_op_rms_norm_scale_fused(ggml_backend_sycl_context& ctx, ggml_tensor* dst, ggml_tensor* scale_tensor);

శూన్య ggml_sycl_op_rms_norm_fused_కూడు(ggml_backend_sycl_context& ctx, ggml_tensor* dst, ggml_tensor* mul_tensor, ggml_tensor* కూడు_tensor);

శూన్య ggml_sycl_op_rms_norm_వెనుక(ggml_backend_sycl_context& ctx, ggml_tensor* dst);

శూన్య ggml_sycl_op_సమూహం_norm(ggml_backend_sycl_context& ctx, ggml_tensor* dst);

శూన్య ggml_sycl_op_l2_norm(ggml_backend_sycl_context& ctx, ggml_tensor* dst);

#నిర్వచించు GGML_SYCL_L2_BATCH_MAX 8
శూన్య ggml_sycl_l2_norm_batch(ggml_backend_sycl_context & ctx, ggml_tensor ** nodes, పూర్ణాంకం count);

#అయితే_ముగింపు // GGML_SYCL_NORM_HPP
