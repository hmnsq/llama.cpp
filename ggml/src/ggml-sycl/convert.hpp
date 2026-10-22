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

#నిర్వచితం_కాకపోతే GGML_SYCL_CONVERT_HPP
#నిర్వచించు GGML_SYCL_CONVERT_HPP

#అన్తర్భావ "common.hpp"

మూస <రకం_పేరు T>
ఉపయోగించి గమ్యం_t_sycl_t = శూన్య (*)(స్థిరరాశి శూన్య * __restrict__ x, T * __restrict__ y, int64_t k, dpct::వరుస_ptr stream);
రకనిర్వచనం గమ్యం_t_sycl_t<దశాంశం>      గమ్యం_fp32_sycl_t;
రకనిర్వచనం గమ్యం_t_sycl_t<sycl::half> గమ్యం_fp16_sycl_t;

గమ్యం_fp16_sycl_t ggml_get_గమ్యం_fp16_sycl(ggml_type type, ggml_tensor * dst);
గమ్యం_fp32_sycl_t ggml_get_గమ్యం_fp32_sycl(ggml_type type, ggml_tensor * dst);

#నిర్వచితమైతే GGML_SYCL_HAS_BF16
రకనిర్వచనం గమ్యం_t_sycl_t<sycl::ext::oneapi::bfloat16> గమ్యం_bf16_sycl_t;
గమ్యం_bf16_sycl_t ggml_get_గమ్యం_bf16_sycl(ggml_type type, ggml_tensor * dst);
#అయితే_ముగింపు

// Nc = Non-contiguous
మూస <రకం_పేరు T>
ఉపయోగించి గమ్యం_t_nc_sycl_t = శూన్య (*)(స్థిరరాశి శూన్య * x, T * y, int64_t ne00, int64_t ne01, int64_t ne02, int64_t ne03,
                                   int64_t s01, int64_t s02, int64_t s03, dpct::వరుస_ptr queue);

రకనిర్వచనం గమ్యం_t_nc_sycl_t<sycl::half> గమ్యం_fp16_nc_sycl_t;
గమ్యం_fp16_nc_sycl_t ggml_get_గమ్యం_fp16_nc_sycl(ggml_type type);

మూస<రకం_పేరు dst_t, రకం_పేరు src_t>
 అంతఃవరుస dst_t ggml_sycl_cast(src_t x) {
    ఒకవేళ constexpr (std::is_same_v<dst_t, src_t>) {
        తిరిగి x;
#నిర్వచితమైతే GGML_SYCL_HAS_BF16
    } లేకపోతే ఒకవేళ constexpr (std::is_same_v<dst_t, sycl::ext::oneapi::bfloat16>) {
        తిరిగి sycl::ext::oneapi::bfloat16(దశాంశం(x));
    } లేకపోతే ఒకవేళ constexpr (std::is_same_v<src_t, sycl::ext::oneapi::bfloat16>) {
        తిరిగి static_cast<దశాంశం>(x);
#అయితే_ముగింపు
    } లేకపోతే ఒకవేళ constexpr (std::is_same_v<src_t, sycl::float2> && std::is_same_v<dst_t, sycl::half2>) {
        తిరిగి x.మూస convert<sycl::half, sycl::rounding_బహుళకం::rte>();
#నిర్వచితమైతే GGML_SYCL_HAS_BF16
    } లేకపోతే ఒకవేళ constexpr (std::is_same_v<src_t, sycl::float2> &&
                         std::is_same_v<dst_t, sycl::vec<sycl::ext::oneapi::bfloat16, 2>>) {
        తిరిగి {x.x, x.y};
#అయితే_ముగింపు
    } లేకపోతే ఒకవేళ constexpr(std::is_same_v<dst_t, int32_t>) {
        తిరిగి int32_t(x);
    } లేకపోతే {
        తిరిగి దశాంశం(x);
    }
}


#అయితే_ముగింపు  // GGML_SYCL_CONVERT_HPP
