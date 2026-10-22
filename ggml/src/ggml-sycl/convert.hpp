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

#شامل "common.hpp"

template <typename T>
using to_t_sycl_t = تهی (*)(ثابت تهی * __restrict__ x, T * __restrict__ y, int64_t k, dpct::queue_ptr stream);
تعریفنوع to_t_sycl_t<اعشاری>      to_fp32_sycl_t;
تعریفنوع to_t_sycl_t<sycl::half> to_fp16_sycl_t;

to_fp16_sycl_t ggml_get_to_fp16_sycl(ggml_type type, ggml_tensor * dst);
to_fp32_sycl_t ggml_get_to_fp32_sycl(ggml_type type, ggml_tensor * dst);

#ifdef GGML_SYCL_HAS_BF16
تعریفنوع to_t_sycl_t<sycl::ext::oneapi::bfloat16> to_bf16_sycl_t;
to_bf16_sycl_t ggml_get_to_bf16_sycl(ggml_type type, ggml_tensor * dst);
#endif

// Nc = Non-contiguous
template <typename T>
using to_t_nc_sycl_t = تهی (*)(ثابت تهی * x, T * y, int64_t ne00, int64_t ne01, int64_t ne02, int64_t ne03,
                                   int64_t s01, int64_t s02, int64_t s03, dpct::queue_ptr queue);

تعریفنوع to_t_nc_sycl_t<sycl::half> to_fp16_nc_sycl_t;
to_fp16_nc_sycl_t ggml_get_to_fp16_nc_sycl(ggml_type type);

template<typename dst_t, typename src_t>
 درونخطی dst_t ggml_sycl_cast(src_t x) {
    اگر constexpr (std::is_same_v<dst_t, src_t>) {
        بازگشت x;
#ifdef GGML_SYCL_HAS_BF16
    } وگرنه اگر constexpr (std::is_same_v<dst_t, sycl::ext::oneapi::bfloat16>) {
        بازگشت sycl::ext::oneapi::bfloat16(اعشاری(x));
    } وگرنه اگر constexpr (std::is_same_v<src_t, sycl::ext::oneapi::bfloat16>) {
        بازگشت static_cast<اعشاری>(x);
#endif
    } وگرنه اگر constexpr (std::is_same_v<src_t, sycl::float2> && std::is_same_v<dst_t, sycl::half2>) {
        بازگشت x.template convert<sycl::half, sycl::rounding_mode::rte>();
#ifdef GGML_SYCL_HAS_BF16
    } وگرنه اگر constexpr (std::is_same_v<src_t, sycl::float2> &&
                         std::is_same_v<dst_t, sycl::vec<sycl::ext::oneapi::bfloat16, 2>>) {
        بازگشت {x.x, x.y};
#endif
    } وگرنه اگر constexpr(std::is_same_v<dst_t, int32_t>) {
        بازگشت int32_t(x);
    } وگرنه {
        بازگشت اعشاری(x);
    }
}


#endif  // GGML_SYCL_CONVERT_HPP
