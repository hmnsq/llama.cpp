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

#جے_بنا_تعریف GGML_SYCL_SOFTMAX_HPP
#تعریف_کرو GGML_SYCL_SOFTMAX_HPP

#ضمن "common.hpp"

#تعریف_کرو SYCL_SOFT_MAX_بلاک_ناپ 1024

فراغ ggml_sycl_op_soft_max(ggml_backend_sycl_context &ctx, ggml_tensor *dst);

فراغ ggml_sycl_op_soft_max_پٹھ(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

#جے_انت // GGML_SYCL_SOFTMAX_HPP
