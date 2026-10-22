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

#अगर_अघोषित GGML_SYCL_NORM_HPP
#घोषणा GGML_SYCL_NORM_HPP

#समावेश "common.hpp"

व्योम ggml_sycl_op_norm(ggml_backend_sycl_context& ctx, ggml_tensor* dst);

व्योम ggml_sycl_op_rms_norm(ggml_backend_sycl_context& ctx, ggml_tensor* dst);

व्योम ggml_sycl_op_rms_norm_fused(ggml_backend_sycl_context& ctx, ggml_tensor* dst, ggml_tensor* mul);

व्योम ggml_sycl_op_rms_norm_scale_fused(ggml_backend_sycl_context& ctx, ggml_tensor* dst, ggml_tensor* scale_tensor);

व्योम ggml_sycl_op_rms_norm_fused_add(ggml_backend_sycl_context& ctx, ggml_tensor* dst, ggml_tensor* mul_tensor, ggml_tensor* add_tensor);

व्योम ggml_sycl_op_rms_norm_back(ggml_backend_sycl_context& ctx, ggml_tensor* dst);

व्योम ggml_sycl_op_group_norm(ggml_backend_sycl_context& ctx, ggml_tensor* dst);

व्योम ggml_sycl_op_l2_norm(ggml_backend_sycl_context& ctx, ggml_tensor* dst);

#घोषणा GGML_SYCL_L2_BATCH_MAX 8
व्योम ggml_sycl_l2_norm_batch(ggml_backend_sycl_context & ctx, ggml_tensor ** nodes, पूर्णांक count);

#पूर्ण_अगर // GGML_SYCL_NORM_HPP
