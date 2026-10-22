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

#यदि_अपरिभाषितम् GGML_SYCL_CONVERT_HPP
#परिभाषय GGML_SYCL_CONVERT_HPP

#अन्तर्भाव "common.hpp"

प्रतिरूपम् <प्रकारनाम T>
उपयुज्य गन्तव्यम्_t_sycl_t = शून्य (*)(ध्रुव शून्य * __restrict__ x, T * __restrict__ y, int64_t k, dpct::queue_ptr stream);
प्रकारनिर्देश गन्तव्यम्_t_sycl_t<भिन्न>      गन्तव्यम्_fp32_sycl_t;
प्रकारनिर्देश गन्तव्यम्_t_sycl_t<sycl::half> गन्तव्यम्_fp16_sycl_t;

गन्तव्यम्_fp16_sycl_t ggml_get_गन्तव्यम्_fp16_sycl(ggml_type type, ggml_tensor * dst);
गन्तव्यम्_fp32_sycl_t ggml_get_गन्तव्यम्_fp32_sycl(ggml_type type, ggml_tensor * dst);

#यदि_परिभाषितम् GGML_SYCL_HAS_BF16
प्रकारनिर्देश गन्तव्यम्_t_sycl_t<sycl::ext::oneapi::bfloat16> गन्तव्यम्_bf16_sycl_t;
गन्तव्यम्_bf16_sycl_t ggml_get_गन्तव्यम्_bf16_sycl(ggml_type type, ggml_tensor * dst);
#यदि_अन्तः

// Nc = Non-contiguous
प्रतिरूपम् <प्रकारनाम T>
उपयुज्य गन्तव्यम्_t_nc_sycl_t = शून्य (*)(ध्रुव शून्य * x, T * y, int64_t ne00, int64_t ne01, int64_t ne02, int64_t ne03,
                                   int64_t s01, int64_t s02, int64_t s03, dpct::queue_ptr queue);

प्रकारनिर्देश गन्तव्यम्_t_nc_sycl_t<sycl::half> गन्तव्यम्_fp16_nc_sycl_t;
गन्तव्यम्_fp16_nc_sycl_t ggml_get_गन्तव्यम्_fp16_nc_sycl(ggml_type type);

प्रतिरूपम्<प्रकारनाम dst_t, प्रकारनाम src_t>
 अन्तःस्थ dst_t ggml_sycl_cast(src_t x) {
    यदि constexpr (std::is_same_v<dst_t, src_t>) {
        प्रत्यागम x;
#यदि_परिभाषितम् GGML_SYCL_HAS_BF16
    } अन्यथा यदि constexpr (std::is_same_v<dst_t, sycl::ext::oneapi::bfloat16>) {
        प्रत्यागम sycl::ext::oneapi::bfloat16(भिन्न(x));
    } अन्यथा यदि constexpr (std::is_same_v<src_t, sycl::ext::oneapi::bfloat16>) {
        प्रत्यागम static_cast<भिन्न>(x);
#यदि_अन्तः
    } अन्यथा यदि constexpr (std::is_same_v<src_t, sycl::float2> && std::is_same_v<dst_t, sycl::half2>) {
        प्रत्यागम x.प्रतिरूपम् convert<sycl::half, sycl::rounding_बहुलकम्::rte>();
#यदि_परिभाषितम् GGML_SYCL_HAS_BF16
    } अन्यथा यदि constexpr (std::is_same_v<src_t, sycl::float2> &&
                         std::is_same_v<dst_t, sycl::vec<sycl::ext::oneapi::bfloat16, 2>>) {
        प्रत्यागम {x.x, x.y};
#यदि_अन्तः
    } अन्यथा यदि constexpr(std::is_same_v<dst_t, int32_t>) {
        प्रत्यागम int32_t(x);
    } अन्यथा {
        प्रत्यागम भिन्न(x);
    }
}


#यदि_अन्तः  // GGML_SYCL_CONVERT_HPP
