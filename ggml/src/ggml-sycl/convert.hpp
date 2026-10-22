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

#അന്തര്ഭാവ "common.hpp"

template <typename T>
using to_t_sycl_t = ശൂന്യം (*)(സ്ഥിരം ശൂന്യം * __restrict__ x, T * __restrict__ y, int64_t k, dpct::queue_ptr stream);
തരനിർവചനം to_t_sycl_t<ദശാംശം>      to_fp32_sycl_t;
തരനിർവചനം to_t_sycl_t<sycl::half> to_fp16_sycl_t;

to_fp16_sycl_t ggml_get_to_fp16_sycl(ggml_type type, ggml_tensor * dst);
to_fp32_sycl_t ggml_get_to_fp32_sycl(ggml_type type, ggml_tensor * dst);

#ifdef GGML_SYCL_HAS_BF16
തരനിർവചനം to_t_sycl_t<sycl::ext::oneapi::bfloat16> to_bf16_sycl_t;
to_bf16_sycl_t ggml_get_to_bf16_sycl(ggml_type type, ggml_tensor * dst);
#endif

// Nc = Non-contiguous
template <typename T>
using to_t_nc_sycl_t = ശൂന്യം (*)(സ്ഥിരം ശൂന്യം * x, T * y, int64_t ne00, int64_t ne01, int64_t ne02, int64_t ne03,
                                   int64_t s01, int64_t s02, int64_t s03, dpct::queue_ptr queue);

തരനിർവചനം to_t_nc_sycl_t<sycl::half> to_fp16_nc_sycl_t;
to_fp16_nc_sycl_t ggml_get_to_fp16_nc_sycl(ggml_type type);

template<typename dst_t, typename src_t>
 വരിയിൽ dst_t ggml_sycl_cast(src_t x) {
    എങ്കിൽ constexpr (std::is_same_v<dst_t, src_t>) {
        തിരികെ x;
#ifdef GGML_SYCL_HAS_BF16
    } അല്ലെങ്കിൽ എങ്കിൽ constexpr (std::is_same_v<dst_t, sycl::ext::oneapi::bfloat16>) {
        തിരികെ sycl::ext::oneapi::bfloat16(ദശാംശം(x));
    } അല്ലെങ്കിൽ എങ്കിൽ constexpr (std::is_same_v<src_t, sycl::ext::oneapi::bfloat16>) {
        തിരികെ static_cast<ദശാംശം>(x);
#endif
    } അല്ലെങ്കിൽ എങ്കിൽ constexpr (std::is_same_v<src_t, sycl::float2> && std::is_same_v<dst_t, sycl::half2>) {
        തിരികെ x.template convert<sycl::half, sycl::rounding_mode::rte>();
#ifdef GGML_SYCL_HAS_BF16
    } അല്ലെങ്കിൽ എങ്കിൽ constexpr (std::is_same_v<src_t, sycl::float2> &&
                         std::is_same_v<dst_t, sycl::vec<sycl::ext::oneapi::bfloat16, 2>>) {
        തിരികെ {x.x, x.y};
#endif
    } അല്ലെങ്കിൽ എങ്കിൽ constexpr(std::is_same_v<dst_t, int32_t>) {
        തിരികെ int32_t(x);
    } അല്ലെങ്കിൽ {
        തിരികെ ദശാംശം(x);
    }
}


#endif  // GGML_SYCL_CONVERT_HPP
