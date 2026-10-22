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

#ವ್ಯಾಖ್ಯಾನಿತವಾಗದಿದ್ದರೆ GGML_SYCL_CONVERT_HPP
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_SYCL_CONVERT_HPP

#ಅನ್ತರ್ಭಾವ "common.hpp"

ಮಾದರಿ <ಬಗೆ_ಹೆಸರು T>
ಬಳಸಿ ಗಮ್ಯ_t_sycl_t = ಶೂನ್ಯ (*)(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * __restrict__ x, T * __restrict__ y, int64_t k, dpct::ಸರತಿ_ptr stream);
ಪ್ರಕಾರವ್ಯಾಖ್ಯೆ ಗಮ್ಯ_t_sycl_t<ದಶಮಾಂಶ>      ಗಮ್ಯ_fp32_sycl_t;
ಪ್ರಕಾರವ್ಯಾಖ್ಯೆ ಗಮ್ಯ_t_sycl_t<sycl::half> ಗಮ್ಯ_fp16_sycl_t;

ಗಮ್ಯ_fp16_sycl_t ggml_get_ಗಮ್ಯ_fp16_sycl(ggml_type type, ggml_tensor * dst);
ಗಮ್ಯ_fp32_sycl_t ggml_get_ಗಮ್ಯ_fp32_sycl(ggml_type type, ggml_tensor * dst);

#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ GGML_SYCL_HAS_BF16
ಪ್ರಕಾರವ್ಯಾಖ್ಯೆ ಗಮ್ಯ_t_sycl_t<sycl::ext::oneapi::bfloat16> ಗಮ್ಯ_bf16_sycl_t;
ಗಮ್ಯ_bf16_sycl_t ggml_get_ಗಮ್ಯ_bf16_sycl(ggml_type type, ggml_tensor * dst);
#ಆದರೆ_ಅಂತ್ಯ

// Nc = Non-contiguous
ಮಾದರಿ <ಬಗೆ_ಹೆಸರು T>
ಬಳಸಿ ಗಮ್ಯ_t_nc_sycl_t = ಶೂನ್ಯ (*)(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * x, T * y, int64_t ne00, int64_t ne01, int64_t ne02, int64_t ne03,
                                   int64_t s01, int64_t s02, int64_t s03, dpct::ಸರತಿ_ptr queue);

ಪ್ರಕಾರವ್ಯಾಖ್ಯೆ ಗಮ್ಯ_t_nc_sycl_t<sycl::half> ಗಮ್ಯ_fp16_nc_sycl_t;
ಗಮ್ಯ_fp16_nc_sycl_t ggml_get_ಗಮ್ಯ_fp16_nc_sycl(ggml_type type);

ಮಾದರಿ<ಬಗೆ_ಹೆಸರು dst_t, ಬಗೆ_ಹೆಸರು src_t>
 ಸಾಲೊಳಗೆ dst_t ggml_sycl_cast(src_t x) {
    ಆದರೆ constexpr (std::is_same_v<dst_t, src_t>) {
        ಹಿಂತಿರುಗಿ x;
#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ GGML_SYCL_HAS_BF16
    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ constexpr (std::is_same_v<dst_t, sycl::ext::oneapi::bfloat16>) {
        ಹಿಂತಿರುಗಿ sycl::ext::oneapi::bfloat16(ದಶಮಾಂಶ(x));
    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ constexpr (std::is_same_v<src_t, sycl::ext::oneapi::bfloat16>) {
        ಹಿಂತಿರುಗಿ static_cast<ದಶಮಾಂಶ>(x);
#ಆದರೆ_ಅಂತ್ಯ
    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ constexpr (std::is_same_v<src_t, sycl::float2> && std::is_same_v<dst_t, sycl::half2>) {
        ಹಿಂತಿರುಗಿ x.ಮಾದರಿ convert<sycl::half, sycl::rounding_ಬಹುಲಕ::rte>();
#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ GGML_SYCL_HAS_BF16
    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ constexpr (std::is_same_v<src_t, sycl::float2> &&
                         std::is_same_v<dst_t, sycl::vec<sycl::ext::oneapi::bfloat16, 2>>) {
        ಹಿಂತಿರುಗಿ {x.x, x.y};
#ಆದರೆ_ಅಂತ್ಯ
    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ constexpr(std::is_same_v<dst_t, int32_t>) {
        ಹಿಂತಿರುಗಿ int32_t(x);
    } ಇಲ್ಲದಿದ್ದರೆ {
        ಹಿಂತಿರುಗಿ ದಶಮಾಂಶ(x);
    }
}


#ಆದರೆ_ಅಂತ್ಯ  // GGML_SYCL_CONVERT_HPP
