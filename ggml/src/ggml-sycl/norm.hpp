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

#ਜੇ_ਅਪਰਿਭਾਸ਼ਿਤ GGML_SYCL_NORM_HPP
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_SYCL_NORM_HPP

#ਅਨ੍ਤਰ੍ਭਾਵ "common.hpp"

ਖਾਲੀ ggml_sycl_op_norm(ggml_backend_sycl_context& ctx, ggml_tensor* dst);

ਖਾਲੀ ggml_sycl_op_rms_norm(ggml_backend_sycl_context& ctx, ggml_tensor* dst);

ਖਾਲੀ ggml_sycl_op_rms_norm_fused(ggml_backend_sycl_context& ctx, ggml_tensor* dst, ggml_tensor* mul);

ਖਾਲੀ ggml_sycl_op_rms_norm_scale_fused(ggml_backend_sycl_context& ctx, ggml_tensor* dst, ggml_tensor* scale_tensor);

ਖਾਲੀ ggml_sycl_op_rms_norm_fused_ਜੋੜੋ(ggml_backend_sycl_context& ctx, ggml_tensor* dst, ggml_tensor* mul_tensor, ggml_tensor* ਜੋੜੋ_tensor);

ਖਾਲੀ ggml_sycl_op_rms_norm_ਪਿੱਠ(ggml_backend_sycl_context& ctx, ggml_tensor* dst);

ਖਾਲੀ ggml_sycl_op_ਸਮੂਹ_norm(ggml_backend_sycl_context& ctx, ggml_tensor* dst);

ਖਾਲੀ ggml_sycl_op_l2_norm(ggml_backend_sycl_context& ctx, ggml_tensor* dst);

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_SYCL_L2_BATCH_MAX 8
ਖਾਲੀ ggml_sycl_l2_norm_batch(ggml_backend_sycl_context & ctx, ggml_tensor ** nodes, ਪੂਰਨ count);

#ਜੇ_ਅੰਤ // GGML_SYCL_NORM_HPP
