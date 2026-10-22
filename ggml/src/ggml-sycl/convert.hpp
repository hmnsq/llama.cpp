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

#अगर_अघोषित GGML_SYCL_CONVERT_HPP
#घोषणा GGML_SYCL_CONVERT_HPP

#समावेश "common.hpp"

ढाँचा <typename T>
using to_t_sycl_t = व्योम (*)(स्थिर व्योम * __restrict__ x, T * __restrict__ y, int64_t k, dpct::queue_ptr stream);
प्रकार to_t_sycl_t<भग्न>      to_fp32_sycl_t;
प्रकार to_t_sycl_t<sycl::half> to_fp16_sycl_t;

to_fp16_sycl_t ggml_get_to_fp16_sycl(ggml_type type, ggml_tensor * dst);
to_fp32_sycl_t ggml_get_to_fp32_sycl(ggml_type type, ggml_tensor * dst);

#अगर_घोषित GGML_SYCL_HAS_BF16
प्रकार to_t_sycl_t<sycl::ext::oneapi::bfloat16> to_bf16_sycl_t;
to_bf16_sycl_t ggml_get_to_bf16_sycl(ggml_type type, ggml_tensor * dst);
#पूर्ण_अगर

// Nc = Non-contiguous
ढाँचा <typename T>
using to_t_nc_sycl_t = व्योम (*)(स्थिर व्योम * x, T * y, int64_t ne00, int64_t ne01, int64_t ne02, int64_t ne03,
                                   int64_t s01, int64_t s02, int64_t s03, dpct::queue_ptr queue);

प्रकार to_t_nc_sycl_t<sycl::half> to_fp16_nc_sycl_t;
to_fp16_nc_sycl_t ggml_get_to_fp16_nc_sycl(ggml_type type);

ढाँचा<typename dst_t, typename src_t>
 अंतरभूत dst_t ggml_sycl_cast(src_t x) {
    अगर constexpr (std::is_same_v<dst_t, src_t>) {
        वापस x;
#अगर_घोषित GGML_SYCL_HAS_BF16
    } अन्यथा अगर constexpr (std::is_same_v<dst_t, sycl::ext::oneapi::bfloat16>) {
        वापस sycl::ext::oneapi::bfloat16(भग्न(x));
    } अन्यथा अगर constexpr (std::is_same_v<src_t, sycl::ext::oneapi::bfloat16>) {
        वापस static_cast<भग्न>(x);
#पूर्ण_अगर
    } अन्यथा अगर constexpr (std::is_same_v<src_t, sycl::float2> && std::is_same_v<dst_t, sycl::half2>) {
        वापस x.ढाँचा convert<sycl::half, sycl::rounding_mode::rte>();
#अगर_घोषित GGML_SYCL_HAS_BF16
    } अन्यथा अगर constexpr (std::is_same_v<src_t, sycl::float2> &&
                         std::is_same_v<dst_t, sycl::vec<sycl::ext::oneapi::bfloat16, 2>>) {
        वापस {x.x, x.y};
#पूर्ण_अगर
    } अन्यथा अगर constexpr(std::is_same_v<dst_t, int32_t>) {
        वापस int32_t(x);
    } अन्यथा {
        वापस भग्न(x);
    }
}


#पूर्ण_अगर  // GGML_SYCL_CONVERT_HPP
