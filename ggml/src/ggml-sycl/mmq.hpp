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

#ਜੇ_ਅਪਰਿਭਾਸ਼ਿਤ GGML_SYCL_MMQ_HPP
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_SYCL_MMQ_HPP

#ਅਨ੍ਤਰ੍ਭਾਵ "common.hpp"

ਖਾਲੀ ggml_sycl_op_mul_mat_q(
    ggml_backend_sycl_context & ctx,
    ਅਚਲ ggml_tensor* src0,
    ਅਚਲ ggml_tensor* src1,
    ggml_tensor* dst,
    ਅਚਲ ਅੱਖਰ* src0_dd_i,
    ਅਚਲ ਦਸ਼ਮਲਵ* src1_ddf_i,
    ਅਚਲ ਅੱਖਰ* src1_ddq_i,
    ਦਸ਼ਮਲਵ* dst_dd_i,
    ਅਚਲ int64_t ਪੰਕਤੀ_ਨੀਵਾਂ,
    ਅਚਲ int64_t ਪੰਕਤੀ_ਉੱਚਾ,
    ਅਚਲ int64_t src1_ncols,
    ਅਚਲ int64_t src1_padded_ਪੰਕਤੀ_ਮਾਪ,
    ਅਚਲ dpct::ਕਤਾਰ_ptr& stream);

#ਜੇ_ਅੰਤ // GGML_SYCL_MMQ_HPP
