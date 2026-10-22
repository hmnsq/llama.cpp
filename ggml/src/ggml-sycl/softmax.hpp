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

#اگر_تعریف_نشده GGML_SYCL_SOFTMAX_HPP
#تعریف_کن GGML_SYCL_SOFTMAX_HPP

#ضمن "common.hpp"

#تعریف_کن SYCL_SOFT_MAX_بلوک_اندازه 1024

فراغ ggml_sycl_op_soft_max(ggml_backend_sycl_context &ctx, ggml_tensor *dst);

فراغ ggml_sycl_op_soft_max_پشت(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

#پایان_اگر // GGML_SYCL_SOFTMAX_HPP
