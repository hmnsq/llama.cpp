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

#جے_بنا_تعریف GGML_SYCL_NORM_HPP
#تعریف_کرو GGML_SYCL_NORM_HPP

#ضمن "common.hpp"

فراغ ggml_sycl_op_norm(ggml_backend_sycl_context& ctx, ggml_tensor* dst);

فراغ ggml_sycl_op_rms_norm(ggml_backend_sycl_context& ctx, ggml_tensor* dst);

فراغ ggml_sycl_op_rms_norm_fused(ggml_backend_sycl_context& ctx, ggml_tensor* dst, ggml_tensor* mul);

فراغ ggml_sycl_op_rms_norm_scale_fused(ggml_backend_sycl_context& ctx, ggml_tensor* dst, ggml_tensor* scale_tensor);

فراغ ggml_sycl_op_rms_norm_fused_جوڑو(ggml_backend_sycl_context& ctx, ggml_tensor* dst, ggml_tensor* mul_tensor, ggml_tensor* جوڑو_tensor);

فراغ ggml_sycl_op_rms_norm_پٹھ(ggml_backend_sycl_context& ctx, ggml_tensor* dst);

فراغ ggml_sycl_op_گروہ_norm(ggml_backend_sycl_context& ctx, ggml_tensor* dst);

فراغ ggml_sycl_op_l2_norm(ggml_backend_sycl_context& ctx, ggml_tensor* dst);

#تعریف_کرو GGML_SYCL_L2_BATCH_MAX 8
فراغ ggml_sycl_l2_norm_batch(ggml_backend_sycl_context & ctx, ggml_tensor ** nodes, صحيح count);

#جے_انت // GGML_SYCL_NORM_HPP
