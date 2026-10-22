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

#ଅନ୍ତର୍ଭାଵ "common.hpp"

template <typename T>
using to_t_sycl_t = ଶୂନ୍ୟ (*)(ଅଚଳ ଶୂନ୍ୟ * __restrict__ x, T * __restrict__ y, int64_t k, dpct::queue_ptr stream);
ପ୍ରକାରବ୍ୟାଖ୍ୟା to_t_sycl_t<ଦଶମିକ>      to_fp32_sycl_t;
ପ୍ରକାରବ୍ୟାଖ୍ୟା to_t_sycl_t<sycl::half> to_fp16_sycl_t;

to_fp16_sycl_t ggml_get_to_fp16_sycl(ggml_type type, ggml_tensor * dst);
to_fp32_sycl_t ggml_get_to_fp32_sycl(ggml_type type, ggml_tensor * dst);

#ifdef GGML_SYCL_HAS_BF16
ପ୍ରକାରବ୍ୟାଖ୍ୟା to_t_sycl_t<sycl::ext::oneapi::bfloat16> to_bf16_sycl_t;
to_bf16_sycl_t ggml_get_to_bf16_sycl(ggml_type type, ggml_tensor * dst);
#endif

// Nc = Non-contiguous
template <typename T>
using to_t_nc_sycl_t = ଶୂନ୍ୟ (*)(ଅଚଳ ଶୂନ୍ୟ * x, T * y, int64_t ne00, int64_t ne01, int64_t ne02, int64_t ne03,
                                   int64_t s01, int64_t s02, int64_t s03, dpct::queue_ptr queue);

ପ୍ରକାରବ୍ୟାଖ୍ୟା to_t_nc_sycl_t<sycl::half> to_fp16_nc_sycl_t;
to_fp16_nc_sycl_t ggml_get_to_fp16_nc_sycl(ggml_type type);

template<typename dst_t, typename src_t>
 ଧାଡ଼ିଭିତରେ dst_t ggml_sycl_cast(src_t x) {
    ଯଦି constexpr (std::is_same_v<dst_t, src_t>) {
        ଫେରନ୍ତୁ x;
#ifdef GGML_SYCL_HAS_BF16
    } ଅନ୍ୟଥା ଯଦି constexpr (std::is_same_v<dst_t, sycl::ext::oneapi::bfloat16>) {
        ଫେରନ୍ତୁ sycl::ext::oneapi::bfloat16(ଦଶମିକ(x));
    } ଅନ୍ୟଥା ଯଦି constexpr (std::is_same_v<src_t, sycl::ext::oneapi::bfloat16>) {
        ଫେରନ୍ତୁ static_cast<ଦଶମିକ>(x);
#endif
    } ଅନ୍ୟଥା ଯଦି constexpr (std::is_same_v<src_t, sycl::float2> && std::is_same_v<dst_t, sycl::half2>) {
        ଫେରନ୍ତୁ x.template convert<sycl::half, sycl::rounding_mode::rte>();
#ifdef GGML_SYCL_HAS_BF16
    } ଅନ୍ୟଥା ଯଦି constexpr (std::is_same_v<src_t, sycl::float2> &&
                         std::is_same_v<dst_t, sycl::vec<sycl::ext::oneapi::bfloat16, 2>>) {
        ଫେରନ୍ତୁ {x.x, x.y};
#endif
    } ଅନ୍ୟଥା ଯଦି constexpr(std::is_same_v<dst_t, int32_t>) {
        ଫେରନ୍ତୁ int32_t(x);
    } ଅନ୍ୟଥା {
        ଫେରନ୍ତୁ ଦଶମିକ(x);
    }
}


#endif  // GGML_SYCL_CONVERT_HPP
