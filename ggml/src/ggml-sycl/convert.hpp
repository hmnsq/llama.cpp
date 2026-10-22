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

#اگر_تعریف_نشده GGML_SYCL_CONVERT_HPP
#تعریف_کن GGML_SYCL_CONVERT_HPP

#ضمن "common.hpp"

قالب <نام_نوع T>
استفاده مقصد_t_sycl_t = فراغ (*)(ثابت فراغ * __restrict__ x, T * __restrict__ y, int64_t k, dpct::صف_ptr stream);
تعريفنوع مقصد_t_sycl_t<عائم>      مقصد_fp32_sycl_t;
تعريفنوع مقصد_t_sycl_t<sycl::half> مقصد_fp16_sycl_t;

مقصد_fp16_sycl_t ggml_get_مقصد_fp16_sycl(ggml_type type, ggml_tensor * dst);
مقصد_fp32_sycl_t ggml_get_مقصد_fp32_sycl(ggml_type type, ggml_tensor * dst);

#اگر_تعریف_شده GGML_SYCL_HAS_BF16
تعريفنوع مقصد_t_sycl_t<sycl::ext::oneapi::bfloat16> مقصد_bf16_sycl_t;
مقصد_bf16_sycl_t ggml_get_مقصد_bf16_sycl(ggml_type type, ggml_tensor * dst);
#پایان_اگر

// Nc = Non-contiguous
قالب <نام_نوع T>
استفاده مقصد_t_nc_sycl_t = فراغ (*)(ثابت فراغ * x, T * y, int64_t ne00, int64_t ne01, int64_t ne02, int64_t ne03,
                                   int64_t s01, int64_t s02, int64_t s03, dpct::صف_ptr queue);

تعريفنوع مقصد_t_nc_sycl_t<sycl::half> مقصد_fp16_nc_sycl_t;
مقصد_fp16_nc_sycl_t ggml_get_مقصد_fp16_nc_sycl(ggml_type type);

قالب<نام_نوع dst_t, نام_نوع src_t>
 مضمن dst_t ggml_sycl_cast(src_t x) {
    إذا constexpr (std::is_same_v<dst_t, src_t>) {
        أرجع x;
#اگر_تعریف_شده GGML_SYCL_HAS_BF16
    } وإلا إذا constexpr (std::is_same_v<dst_t, sycl::ext::oneapi::bfloat16>) {
        أرجع sycl::ext::oneapi::bfloat16(عائم(x));
    } وإلا إذا constexpr (std::is_same_v<src_t, sycl::ext::oneapi::bfloat16>) {
        أرجع static_cast<عائم>(x);
#پایان_اگر
    } وإلا إذا constexpr (std::is_same_v<src_t, sycl::float2> && std::is_same_v<dst_t, sycl::half2>) {
        أرجع x.قالب convert<sycl::half, sycl::rounding_نما::rte>();
#اگر_تعریف_شده GGML_SYCL_HAS_BF16
    } وإلا إذا constexpr (std::is_same_v<src_t, sycl::float2> &&
                         std::is_same_v<dst_t, sycl::vec<sycl::ext::oneapi::bfloat16, 2>>) {
        أرجع {x.x, x.y};
#پایان_اگر
    } وإلا إذا constexpr(std::is_same_v<dst_t, int32_t>) {
        أرجع int32_t(x);
    } وإلا {
        أرجع عائم(x);
    }
}


#پایان_اگر  // GGML_SYCL_CONVERT_HPP
