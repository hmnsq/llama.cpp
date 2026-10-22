//
// MIT license
// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: MIT
//

//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

#ifndef GGML_SYCL_CONVERT_HPP
#define GGML_SYCL_CONVERT_HPP

#अन्तर्भाव "common.hpp"

template <typename T>
using to_t_sycl_t = शून्य (*)(ध्रुव शून्य * __restrict__ x, T * __restrict__ y, int64_t k, dpct::queue_ptr stream);
प्रकारनिर्देश to_t_sycl_t<भिन्न>      to_fp32_sycl_t;
प्रकारनिर्देश to_t_sycl_t<sycl::half> to_fp16_sycl_t;

to_fp16_sycl_t ggml_get_to_fp16_sycl(ggml_type type, ggml_tensor * dst);
to_fp32_sycl_t ggml_get_to_fp32_sycl(ggml_type type, ggml_tensor * dst);

#ifdef GGML_SYCL_HAS_BF16
प्रकारनिर्देश to_t_sycl_t<sycl::ext::oneapi::bfloat16> to_bf16_sycl_t;
to_bf16_sycl_t ggml_get_to_bf16_sycl(ggml_type type, ggml_tensor * dst);
#endif

// Nc = Non-contiguous
template <typename T>
using to_t_nc_sycl_t = शून्य (*)(ध्रुव शून्य * x, T * y, int64_t ne00, int64_t ne01, int64_t ne02, int64_t ne03,
                                   int64_t s01, int64_t s02, int64_t s03, dpct::queue_ptr queue);

प्रकारनिर्देश to_t_nc_sycl_t<sycl::half> to_fp16_nc_sycl_t;
to_fp16_nc_sycl_t ggml_get_to_fp16_nc_sycl(ggml_type type);

template<typename dst_t, typename src_t>
 अन्तःस्थ dst_t ggml_sycl_cast(src_t x) {
    सचे constexpr (std::is_same_v<dst_t, src_t>) {
        पटिनिवत्त x;
#ifdef GGML_SYCL_HAS_BF16
    } अञ्ञथा सचे constexpr (std::is_same_v<dst_t, sycl::ext::oneapi::bfloat16>) {
        पटिनिवत्त sycl::ext::oneapi::bfloat16(भिन्न(x));
    } अञ्ञथा सचे constexpr (std::is_same_v<src_t, sycl::ext::oneapi::bfloat16>) {
        पटिनिवत्त static_cast<भिन्न>(x);
#endif
    } अञ्ञथा सचे constexpr (std::is_same_v<src_t, sycl::float2> && std::is_same_v<dst_t, sycl::half2>) {
        पटिनिवत्त x.template convert<sycl::half, sycl::rounding_mode::rte>();
#ifdef GGML_SYCL_HAS_BF16
    } अञ्ञथा सचे constexpr (std::is_same_v<src_t, sycl::float2> &&
                         std::is_same_v<dst_t, sycl::vec<sycl::ext::oneapi::bfloat16, 2>>) {
        पटिनिवत्त {x.x, x.y};
#endif
    } अञ्ञथा सचे constexpr(std::is_same_v<dst_t, int32_t>) {
        पटिनिवत्त int32_t(x);
    } अञ्ञथा {
        पटिनिवत्त भिन्न(x);
    }
}


#endif  // GGML_SYCL_CONVERT_HPP
