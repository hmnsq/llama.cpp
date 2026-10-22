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

#ವ್ಯಾಖ್ಯಾನಿತವಾಗದಿದ್ದರೆ GGML_SYCL_NORM_HPP
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_SYCL_NORM_HPP

#ಅನ್ತರ್ಭಾವ "common.hpp"

ಶೂನ್ಯ ggml_sycl_op_norm(ggml_backend_sycl_context& ctx, ggml_tensor* dst);

ಶೂನ್ಯ ggml_sycl_op_rms_norm(ggml_backend_sycl_context& ctx, ggml_tensor* dst);

ಶೂನ್ಯ ggml_sycl_op_rms_norm_fused(ggml_backend_sycl_context& ctx, ggml_tensor* dst, ggml_tensor* mul);

ಶೂನ್ಯ ggml_sycl_op_rms_norm_scale_fused(ggml_backend_sycl_context& ctx, ggml_tensor* dst, ggml_tensor* scale_tensor);

ಶೂನ್ಯ ggml_sycl_op_rms_norm_fused_ಕೂಡಿಸು(ggml_backend_sycl_context& ctx, ggml_tensor* dst, ggml_tensor* mul_tensor, ggml_tensor* ಕೂಡಿಸು_tensor);

ಶೂನ್ಯ ggml_sycl_op_rms_norm_ಹಿಂಭಾಗ(ggml_backend_sycl_context& ctx, ggml_tensor* dst);

ಶೂನ್ಯ ggml_sycl_op_ಗುಂಪು_norm(ggml_backend_sycl_context& ctx, ggml_tensor* dst);

ಶೂನ್ಯ ggml_sycl_op_l2_norm(ggml_backend_sycl_context& ctx, ggml_tensor* dst);

#ವ್ಯಾಖ್ಯಾನಿಸು GGML_SYCL_L2_BATCH_MAX 8
ಶೂನ್ಯ ggml_sycl_l2_norm_batch(ggml_backend_sycl_context & ctx, ggml_tensor ** nodes, ಪೂರ್ಣಾಂಕ count);

#ಆದರೆ_ಅಂತ್ಯ // GGML_SYCL_NORM_HPP
