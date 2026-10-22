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

#यदि_अपरिभाषितम् GGML_SYCL_NORM_HPP
#परिभाषय GGML_SYCL_NORM_HPP

#अन्तर्भाव "common.hpp"

शून्य ggml_sycl_op_norm(ggml_backend_sycl_context& ctx, ggml_tensor* dst);

शून्य ggml_sycl_op_rms_norm(ggml_backend_sycl_context& ctx, ggml_tensor* dst);

शून्य ggml_sycl_op_rms_norm_fused(ggml_backend_sycl_context& ctx, ggml_tensor* dst, ggml_tensor* mul);

शून्य ggml_sycl_op_rms_norm_scale_fused(ggml_backend_sycl_context& ctx, ggml_tensor* dst, ggml_tensor* scale_tensor);

शून्य ggml_sycl_op_rms_norm_fused_सङ्कलय(ggml_backend_sycl_context& ctx, ggml_tensor* dst, ggml_tensor* mul_tensor, ggml_tensor* सङ्कलय_tensor);

शून्य ggml_sycl_op_rms_norm_पृष्ठभागः(ggml_backend_sycl_context& ctx, ggml_tensor* dst);

शून्य ggml_sycl_op_गणः_norm(ggml_backend_sycl_context& ctx, ggml_tensor* dst);

शून्य ggml_sycl_op_l2_norm(ggml_backend_sycl_context& ctx, ggml_tensor* dst);

#परिभाषय GGML_SYCL_L2_BATCH_MAX 8
शून्य ggml_sycl_l2_norm_batch(ggml_backend_sycl_context & ctx, ggml_tensor ** nodes, पूर्णाङ्क count);

#यदि_अन्तः // GGML_SYCL_NORM_HPP
