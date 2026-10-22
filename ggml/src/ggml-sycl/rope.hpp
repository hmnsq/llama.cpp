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

#यदि_अपरिभाषितम् GGML_SYCL_ROPE_HPP
#परिभाषय GGML_SYCL_ROPE_HPP

#अन्तर्भाव "common.hpp"

शून्य ggml_sycl_rope(ggml_backend_sycl_context & ctx, ggml_tensor *dst);

शून्य ggml_sycl_rope_पृष्ठभागः(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

शून्य ggml_sycl_rope_fused(ggml_backend_sycl_context & ctx, ggml_tensor * dst, ggml_tensor * समुच्चयः_पङ्क्तिकाः);

#यदि_अन्तः // GGML_SYCL_ROPE_HPP
