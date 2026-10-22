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

#جيڪڏهن_وصف_نه_ٿيل GGML_SYCL_NORM_HPP
#وصف_ڪريو GGML_SYCL_NORM_HPP

#ضمن "common.hpp"

فراغ ggml_sycl_op_norm(ggml_backend_sycl_context& ctx, ggml_tensor* dst);

فراغ ggml_sycl_op_rms_norm(ggml_backend_sycl_context& ctx, ggml_tensor* dst);

فراغ ggml_sycl_op_rms_norm_fused(ggml_backend_sycl_context& ctx, ggml_tensor* dst, ggml_tensor* mul);

فراغ ggml_sycl_op_rms_norm_scale_fused(ggml_backend_sycl_context& ctx, ggml_tensor* dst, ggml_tensor* scale_tensor);

فراغ ggml_sycl_op_rms_norm_fused_جوڙيو(ggml_backend_sycl_context& ctx, ggml_tensor* dst, ggml_tensor* mul_tensor, ggml_tensor* جوڙيو_tensor);

فراغ ggml_sycl_op_rms_norm_پٺ(ggml_backend_sycl_context& ctx, ggml_tensor* dst);

فراغ ggml_sycl_op_گروهه_norm(ggml_backend_sycl_context& ctx, ggml_tensor* dst);

فراغ ggml_sycl_op_l2_norm(ggml_backend_sycl_context& ctx, ggml_tensor* dst);

#وصف_ڪريو GGML_SYCL_L2_BATCH_MAX 8
فراغ ggml_sycl_l2_norm_batch(ggml_backend_sycl_context & ctx, ggml_tensor ** nodes, صحيح count);

#جيڪڏهن_پورو // GGML_SYCL_NORM_HPP
