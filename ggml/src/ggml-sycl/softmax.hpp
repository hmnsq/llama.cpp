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

#अगर_अघोषित GGML_SYCL_SOFTMAX_HPP
#घोषणा GGML_SYCL_SOFTMAX_HPP

#समावेश "common.hpp"

#घोषणा SYCL_SOFT_MAX_BLOCK_आकार 1024

व्योम ggml_sycl_op_soft_max(ggml_backend_sycl_context &ctx, ggml_tensor *dst);

व्योम ggml_sycl_op_soft_max_back(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

#पूर्ण_अगर // GGML_SYCL_SOFTMAX_HPP
