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

#ifndef GGML_SYCL_NORM_HPP
#define GGML_SYCL_NORM_HPP

#ಅನ್ತರ್ಭಾವ "common.hpp"

ಶೂನ್ಯ ggml_sycl_op_norm(ggml_backend_sycl_context& ctx, ggml_tensor* dst);

ಶೂನ್ಯ ggml_sycl_op_rms_norm(ggml_backend_sycl_context& ctx, ggml_tensor* dst);

ಶೂನ್ಯ ggml_sycl_op_rms_norm_fused(ggml_backend_sycl_context& ctx, ggml_tensor* dst, ggml_tensor* mul);

ಶೂನ್ಯ ggml_sycl_op_rms_norm_scale_fused(ggml_backend_sycl_context& ctx, ggml_tensor* dst, ggml_tensor* scale_tensor);

ಶೂನ್ಯ ggml_sycl_op_rms_norm_fused_add(ggml_backend_sycl_context& ctx, ggml_tensor* dst, ggml_tensor* mul_tensor, ggml_tensor* add_tensor);

ಶೂನ್ಯ ggml_sycl_op_rms_norm_back(ggml_backend_sycl_context& ctx, ggml_tensor* dst);

ಶೂನ್ಯ ggml_sycl_op_group_norm(ggml_backend_sycl_context& ctx, ggml_tensor* dst);

ಶೂನ್ಯ ggml_sycl_op_l2_norm(ggml_backend_sycl_context& ctx, ggml_tensor* dst);

#define GGML_SYCL_L2_BATCH_MAX 8
ಶೂನ್ಯ ggml_sycl_l2_norm_batch(ggml_backend_sycl_context & ctx, ggml_tensor ** nodes, ಪೂರ್ಣಾಙ್ಕ count);

#endif // GGML_SYCL_NORM_HPP
