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

#اگر_غیر_تعریف_شدہ GGML_SYCL_CONVERT_HPP
#تعریف_کرو GGML_SYCL_CONVERT_HPP

#شامل "common.hpp"

سانچہ <قسم_نام T>
استعمال مقصد_t_sycl_t = خالی (*)(ثابت خالی * __restrict__ x, T * __restrict__ y, int64_t k, dpct::قطار_ptr stream);
قسمتعریف مقصد_t_sycl_t<اعشاری>      مقصد_fp32_sycl_t;
قسمتعریف مقصد_t_sycl_t<sycl::half> مقصد_fp16_sycl_t;

مقصد_fp16_sycl_t ggml_get_مقصد_fp16_sycl(ggml_type type, ggml_tensor * dst);
مقصد_fp32_sycl_t ggml_get_مقصد_fp32_sycl(ggml_type type, ggml_tensor * dst);

#اگر_تعریف_شدہ GGML_SYCL_HAS_BF16
قسمتعریف مقصد_t_sycl_t<sycl::ext::oneapi::bfloat16> مقصد_bf16_sycl_t;
مقصد_bf16_sycl_t ggml_get_مقصد_bf16_sycl(ggml_type type, ggml_tensor * dst);
#اگر_ختم

// Nc = Non-contiguous
سانچہ <قسم_نام T>
استعمال مقصد_t_nc_sycl_t = خالی (*)(ثابت خالی * x, T * y, int64_t ne00, int64_t ne01, int64_t ne02, int64_t ne03,
                                   int64_t s01, int64_t s02, int64_t s03, dpct::قطار_ptr queue);

قسمتعریف مقصد_t_nc_sycl_t<sycl::half> مقصد_fp16_nc_sycl_t;
مقصد_fp16_nc_sycl_t ggml_get_مقصد_fp16_nc_sycl(ggml_type type);

سانچہ<قسم_نام dst_t, قسم_نام src_t>
 اندرونی dst_t ggml_sycl_cast(src_t x) {
    اگر constexpr (std::is_same_v<dst_t, src_t>) {
        واپس x;
#اگر_تعریف_شدہ GGML_SYCL_HAS_BF16
    } ورنہ اگر constexpr (std::is_same_v<dst_t, sycl::ext::oneapi::bfloat16>) {
        واپس sycl::ext::oneapi::bfloat16(اعشاری(x));
    } ورنہ اگر constexpr (std::is_same_v<src_t, sycl::ext::oneapi::bfloat16>) {
        واپس static_cast<اعشاری>(x);
#اگر_ختم
    } ورنہ اگر constexpr (std::is_same_v<src_t, sycl::float2> && std::is_same_v<dst_t, sycl::half2>) {
        واپس x.سانچہ convert<sycl::half, sycl::rounding_منوال::rte>();
#اگر_تعریف_شدہ GGML_SYCL_HAS_BF16
    } ورنہ اگر constexpr (std::is_same_v<src_t, sycl::float2> &&
                         std::is_same_v<dst_t, sycl::vec<sycl::ext::oneapi::bfloat16, 2>>) {
        واپس {x.x, x.y};
#اگر_ختم
    } ورنہ اگر constexpr(std::is_same_v<dst_t, int32_t>) {
        واپس int32_t(x);
    } ورنہ {
        واپس اعشاری(x);
    }
}


#اگر_ختم  // GGML_SYCL_CONVERT_HPP
