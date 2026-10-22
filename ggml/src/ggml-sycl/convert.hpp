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

#که_نه_تعریف_شوی GGML_SYCL_CONVERT_HPP
#تعریف_کړه GGML_SYCL_CONVERT_HPP

#ضمن "common.hpp"

کينډۍ <ډول_نوم T>
کارول منزل_t_sycl_t = فراغ (*)(ثابت فراغ * __restrict__ x, T * __restrict__ y, int64_t k, dpct::کتار_ptr stream);
تعريفنوع منزل_t_sycl_t<عائم>      منزل_fp32_sycl_t;
تعريفنوع منزل_t_sycl_t<sycl::half> منزل_fp16_sycl_t;

منزل_fp16_sycl_t ggml_get_منزل_fp16_sycl(ggml_type type, ggml_tensor * dst);
منزل_fp32_sycl_t ggml_get_منزل_fp32_sycl(ggml_type type, ggml_tensor * dst);

#که_تعریف_شوی GGML_SYCL_HAS_BF16
تعريفنوع منزل_t_sycl_t<sycl::ext::oneapi::bfloat16> منزل_bf16_sycl_t;
منزل_bf16_sycl_t ggml_get_منزل_bf16_sycl(ggml_type type, ggml_tensor * dst);
#د_که_پای

// Nc = Non-contiguous
کينډۍ <ډول_نوم T>
کارول منزل_t_nc_sycl_t = فراغ (*)(ثابت فراغ * x, T * y, int64_t ne00, int64_t ne01, int64_t ne02, int64_t ne03,
                                   int64_t s01, int64_t s02, int64_t s03, dpct::کتار_ptr queue);

تعريفنوع منزل_t_nc_sycl_t<sycl::half> منزل_fp16_nc_sycl_t;
منزل_fp16_nc_sycl_t ggml_get_منزل_fp16_nc_sycl(ggml_type type);

کينډۍ<ډول_نوم dst_t, ډول_نوم src_t>
 مضمن dst_t ggml_sycl_cast(src_t x) {
    إذا constexpr (std::is_same_v<dst_t, src_t>) {
        أرجع x;
#که_تعریف_شوی GGML_SYCL_HAS_BF16
    } وإلا إذا constexpr (std::is_same_v<dst_t, sycl::ext::oneapi::bfloat16>) {
        أرجع sycl::ext::oneapi::bfloat16(عائم(x));
    } وإلا إذا constexpr (std::is_same_v<src_t, sycl::ext::oneapi::bfloat16>) {
        أرجع static_cast<عائم>(x);
#د_که_پای
    } وإلا إذا constexpr (std::is_same_v<src_t, sycl::float2> && std::is_same_v<dst_t, sycl::half2>) {
        أرجع x.کينډۍ convert<sycl::half, sycl::rounding_ډېرپېښ::rte>();
#که_تعریف_شوی GGML_SYCL_HAS_BF16
    } وإلا إذا constexpr (std::is_same_v<src_t, sycl::float2> &&
                         std::is_same_v<dst_t, sycl::vec<sycl::ext::oneapi::bfloat16, 2>>) {
        أرجع {x.x, x.y};
#د_که_پای
    } وإلا إذا constexpr(std::is_same_v<dst_t, int32_t>) {
        أرجع int32_t(x);
    } وإلا {
        أرجع عائم(x);
    }
}


#د_که_پای  // GGML_SYCL_CONVERT_HPP
