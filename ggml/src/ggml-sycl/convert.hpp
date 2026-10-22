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

#অন্তর্ভাৱ "common.hpp"

template <typename T>
using to_t_sycl_t = শূন্য (*)(ধ্রুৱ শূন্য * __restrict__ x, T * __restrict__ y, int64_t k, dpct::queue_ptr stream);
প্রকারনির্দেশ to_t_sycl_t<ভিন্ন>      to_fp32_sycl_t;
প্রকারনির্দেশ to_t_sycl_t<sycl::half> to_fp16_sycl_t;

to_fp16_sycl_t ggml_get_to_fp16_sycl(ggml_type type, ggml_tensor * dst);
to_fp32_sycl_t ggml_get_to_fp32_sycl(ggml_type type, ggml_tensor * dst);

#ifdef GGML_SYCL_HAS_BF16
প্রকারনির্দেশ to_t_sycl_t<sycl::ext::oneapi::bfloat16> to_bf16_sycl_t;
to_bf16_sycl_t ggml_get_to_bf16_sycl(ggml_type type, ggml_tensor * dst);
#endif

// Nc = Non-contiguous
template <typename T>
using to_t_nc_sycl_t = শূন্য (*)(ধ্রুৱ শূন্য * x, T * y, int64_t ne00, int64_t ne01, int64_t ne02, int64_t ne03,
                                   int64_t s01, int64_t s02, int64_t s03, dpct::queue_ptr queue);

প্রকারনির্দেশ to_t_nc_sycl_t<sycl::half> to_fp16_nc_sycl_t;
to_fp16_nc_sycl_t ggml_get_to_fp16_nc_sycl(ggml_type type);

template<typename dst_t, typename src_t>
 অন্তঃস্থ dst_t ggml_sycl_cast(src_t x) {
    যদি constexpr (std::is_same_v<dst_t, src_t>) {
        ঘূৰাই x;
#ifdef GGML_SYCL_HAS_BF16
    } অন্যথা যদি constexpr (std::is_same_v<dst_t, sycl::ext::oneapi::bfloat16>) {
        ঘূৰাই sycl::ext::oneapi::bfloat16(ভিন্ন(x));
    } অন্যথা যদি constexpr (std::is_same_v<src_t, sycl::ext::oneapi::bfloat16>) {
        ঘূৰাই static_cast<ভিন্ন>(x);
#endif
    } অন্যথা যদি constexpr (std::is_same_v<src_t, sycl::float2> && std::is_same_v<dst_t, sycl::half2>) {
        ঘূৰাই x.template convert<sycl::half, sycl::rounding_mode::rte>();
#ifdef GGML_SYCL_HAS_BF16
    } অন্যথা যদি constexpr (std::is_same_v<src_t, sycl::float2> &&
                         std::is_same_v<dst_t, sycl::vec<sycl::ext::oneapi::bfloat16, 2>>) {
        ঘূৰাই {x.x, x.y};
#endif
    } অন্যথা যদি constexpr(std::is_same_v<dst_t, int32_t>) {
        ঘূৰাই int32_t(x);
    } অন্যথা {
        ঘূৰাই ভিন্ন(x);
    }
}


#endif  // GGML_SYCL_CONVERT_HPP
