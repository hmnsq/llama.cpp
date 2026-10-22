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

#ਜੇ_ਅਪਰਿਭਾਸ਼ਿਤ GGML_SYCL_CONVERT_HPP
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_SYCL_CONVERT_HPP

#ਅਨ੍ਤਰ੍ਭਾਵ "common.hpp"

ਸਾਂਚਾ <ਕਿਸਮ_ਨਾਂ T>
ਵਰਤੋਂ ਮੰਜ਼ਿਲ_t_sycl_t = ਖਾਲੀ (*)(ਅਚਲ ਖਾਲੀ * __restrict__ x, T * __restrict__ y, int64_t k, dpct::ਕਤਾਰ_ptr stream);
ਕਿਸਮਪਰਿਭਾਸ਼ਾ ਮੰਜ਼ਿਲ_t_sycl_t<ਦਸ਼ਮਲਵ>      ਮੰਜ਼ਿਲ_fp32_sycl_t;
ਕਿਸਮਪਰਿਭਾਸ਼ਾ ਮੰਜ਼ਿਲ_t_sycl_t<sycl::half> ਮੰਜ਼ਿਲ_fp16_sycl_t;

ਮੰਜ਼ਿਲ_fp16_sycl_t ggml_get_ਮੰਜ਼ਿਲ_fp16_sycl(ggml_type type, ggml_tensor * dst);
ਮੰਜ਼ਿਲ_fp32_sycl_t ggml_get_ਮੰਜ਼ਿਲ_fp32_sycl(ggml_type type, ggml_tensor * dst);

#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ GGML_SYCL_HAS_BF16
ਕਿਸਮਪਰਿਭਾਸ਼ਾ ਮੰਜ਼ਿਲ_t_sycl_t<sycl::ext::oneapi::bfloat16> ਮੰਜ਼ਿਲ_bf16_sycl_t;
ਮੰਜ਼ਿਲ_bf16_sycl_t ggml_get_ਮੰਜ਼ਿਲ_bf16_sycl(ggml_type type, ggml_tensor * dst);
#ਜੇ_ਅੰਤ

// Nc = Non-contiguous
ਸਾਂਚਾ <ਕਿਸਮ_ਨਾਂ T>
ਵਰਤੋਂ ਮੰਜ਼ਿਲ_t_nc_sycl_t = ਖਾਲੀ (*)(ਅਚਲ ਖਾਲੀ * x, T * y, int64_t ne00, int64_t ne01, int64_t ne02, int64_t ne03,
                                   int64_t s01, int64_t s02, int64_t s03, dpct::ਕਤਾਰ_ptr queue);

ਕਿਸਮਪਰਿਭਾਸ਼ਾ ਮੰਜ਼ਿਲ_t_nc_sycl_t<sycl::half> ਮੰਜ਼ਿਲ_fp16_nc_sycl_t;
ਮੰਜ਼ਿਲ_fp16_nc_sycl_t ggml_get_ਮੰਜ਼ਿਲ_fp16_nc_sycl(ggml_type type);

ਸਾਂਚਾ<ਕਿਸਮ_ਨਾਂ dst_t, ਕਿਸਮ_ਨਾਂ src_t>
 ਸਤਰਅੰਦਰ dst_t ggml_sycl_cast(src_t x) {
    ਜੇ constexpr (std::is_same_v<dst_t, src_t>) {
        ਵਾਪਸ x;
#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ GGML_SYCL_HAS_BF16
    } ਨਹੀਂਤਾਂ ਜੇ constexpr (std::is_same_v<dst_t, sycl::ext::oneapi::bfloat16>) {
        ਵਾਪਸ sycl::ext::oneapi::bfloat16(ਦਸ਼ਮਲਵ(x));
    } ਨਹੀਂਤਾਂ ਜੇ constexpr (std::is_same_v<src_t, sycl::ext::oneapi::bfloat16>) {
        ਵਾਪਸ static_cast<ਦਸ਼ਮਲਵ>(x);
#ਜੇ_ਅੰਤ
    } ਨਹੀਂਤਾਂ ਜੇ constexpr (std::is_same_v<src_t, sycl::float2> && std::is_same_v<dst_t, sycl::half2>) {
        ਵਾਪਸ x.ਸਾਂਚਾ convert<sycl::half, sycl::rounding_ਬਹੁਲਕ::rte>();
#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ GGML_SYCL_HAS_BF16
    } ਨਹੀਂਤਾਂ ਜੇ constexpr (std::is_same_v<src_t, sycl::float2> &&
                         std::is_same_v<dst_t, sycl::vec<sycl::ext::oneapi::bfloat16, 2>>) {
        ਵਾਪਸ {x.x, x.y};
#ਜੇ_ਅੰਤ
    } ਨਹੀਂਤਾਂ ਜੇ constexpr(std::is_same_v<dst_t, int32_t>) {
        ਵਾਪਸ int32_t(x);
    } ਨਹੀਂਤਾਂ {
        ਵਾਪਸ ਦਸ਼ਮਲਵ(x);
    }
}


#ਜੇ_ਅੰਤ  // GGML_SYCL_CONVERT_HPP
