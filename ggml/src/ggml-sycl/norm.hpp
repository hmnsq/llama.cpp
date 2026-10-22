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

#اگر_غیر_تعریف_شدہ GGML_SYCL_NORM_HPP
#تعریف_کرو GGML_SYCL_NORM_HPP

#شامل "common.hpp"

خالی ggml_sycl_op_norm(ggml_backend_sycl_context& ctx, ggml_tensor* dst);

خالی ggml_sycl_op_rms_norm(ggml_backend_sycl_context& ctx, ggml_tensor* dst);

خالی ggml_sycl_op_rms_norm_fused(ggml_backend_sycl_context& ctx, ggml_tensor* dst, ggml_tensor* mul);

خالی ggml_sycl_op_rms_norm_scale_fused(ggml_backend_sycl_context& ctx, ggml_tensor* dst, ggml_tensor* scale_tensor);

خالی ggml_sycl_op_rms_norm_fused_جوڑو(ggml_backend_sycl_context& ctx, ggml_tensor* dst, ggml_tensor* mul_tensor, ggml_tensor* جوڑو_tensor);

خالی ggml_sycl_op_rms_norm_عقب(ggml_backend_sycl_context& ctx, ggml_tensor* dst);

خالی ggml_sycl_op_گروہ_norm(ggml_backend_sycl_context& ctx, ggml_tensor* dst);

خالی ggml_sycl_op_l2_norm(ggml_backend_sycl_context& ctx, ggml_tensor* dst);

#تعریف_کرو GGML_SYCL_L2_BATCH_MAX 8
خالی ggml_sycl_l2_norm_batch(ggml_backend_sycl_context & ctx, ggml_tensor ** nodes, صحیح count);

#اگر_ختم // GGML_SYCL_NORM_HPP
