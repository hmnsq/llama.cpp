//
// MIT license
// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: MIT
//

//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

#నిర్వచితం_కాకపోతే GGML_SYCL_GEMM_HPP
#నిర్వచించు GGML_SYCL_GEMM_HPP

#అన్తర్భావ "ggml-sycl.h"

#ఒకవేళ GGML_SYCL_DNNL

#అన్తర్భావ "dnnl.hpp"
#అన్తర్భావ "dnnl_sycl.hpp"

తరగతి DnnlGemmWrapper {
బహిరంగ:
    ఉపయోగించి dt = dnnl::జ్ఞాపకం::దత్తాంశం_type;
    ఉపయోగించి ట్యాగ్ = dnnl::జ్ఞాపకం::format_ట్యాగ్;

    మూస<రకం_పేరు T>
    స్థిర constexpr dt గమ్యం_dt() {
        ఒకవేళ constexpr (std::is_same_v<T, దశాంశం>) తిరిగి dt::f32;
        లేకపోతే ఒకవేళ constexpr (std::is_same_v<T, sycl::half>) తిరిగి dt::f16;
#నిర్వచితమైతే GGML_SYCL_HAS_BF16
        లేకపోతే ఒకవేళ constexpr (std::is_same_v<T, sycl::ext::oneapi::bfloat16>) తిరిగి dt::bf16;
#అయితే_ముగింపు
        లేకపోతే static_assert(0);
    }

    స్థిర శూన్య gemm(ggml_backend_sycl_context & ctx, పూర్ణాంకం m, పూర్ణాంకం n, పూర్ణాంకం k,
        స్థిరరాశి శూన్య * a, dt at, dnnl_dim_t stra0, dnnl_dim_t stra1, dnnl_dim_t stra2,
        స్థిరరాశి శూన్య * b, dt bt, dnnl_dim_t strb0, dnnl_dim_t strb1, dnnl_dim_t strb2,
        శూన్య * c, dt ct, స్థిరరాశి వరుస_ptr & q, dnnl_dim_t batches_a, dnnl_dim_t batches_b) {

        స్వయంచాలక stream = ctx.ప్రవాహం_dnnl(q);
        స్వయంచాలక eng = ctx.engine_dnnl(q);

        dnnl::జ్ఞాపకం::dims a_dims = {batches_a, m, k };
        dnnl::జ్ఞాపకం::dims a_strides = {stra2, stra1, stra0};
        స్థిరరాశి స్వయంచాలక a_లోపల_md = dnnl::జ్ఞాపకం::desc(a_dims, at, a_strides);

        dnnl::జ్ఞాపకం::dims b_dims = {batches_b, k, n };
        dnnl::జ్ఞాపకం::dims b_strides = {strb2, strb0, strb1};
        స్థిరరాశి స్వయంచాలక b_లోపల_md = dnnl::జ్ఞాపకం::desc(b_dims, bt, b_strides);

        dnnl::జ్ఞాపకం::dims c_dims = { std::max(batches_a, batches_b), m, n};
        dnnl::జ్ఞాపకం::dims c_strides = {m*n, 1,  m };
        స్థిరరాశి స్వయంచాలక c_md    = dnnl::జ్ఞాపకం::desc(c_dims, ct, c_strides);
        dnnl::primitive_attr primitive_attr;
        primitive_attr.సమితి_scratchpad_బహుళకం(dnnl::scratchpad_బహుళకం::user);

#నిర్వచితమైతే GGML_SYCL_F16
        primitive_attr.సమితి_fpmath_బహుళకం(dnnl::fpmath_బహుళకం::f16);
#అయితే_ముగింపు

        స్వయంచాలక a_mem = dnnl::జ్ఞాపకం(a_లోపల_md, eng, const_cast<శూన్య*>(a));
        స్వయంచాలక b_mem = dnnl::జ్ఞాపకం(b_లోపల_md, eng, const_cast<శూన్య*>(b));
        స్వయంచాలక matmul_pd = dnnl::matmul::primitive_desc(eng, a_లోపల_md, b_లోపల_md, c_md, primitive_attr);
        స్వయంచాలక c_mem = dnnl::జ్ఞాపకం(matmul_pd.dst_desc(), eng, c);

        స్థిరరాశి స్వయంచాలక scratchpad_md = matmul_pd.scratchpad_desc();
        ggml_sycl_pool_alloc<uint8_t> scratchpad(ctx.pool());
        శూన్య * scratchpad_ptr = scratchpad_md.get_కొలత() > 0 ? scratchpad.alloc(scratchpad_md.get_కొలత()) : శూన్యసూచిక;
        స్వయంచాలక scratchpad_mem = dnnl::జ్ఞాపకం(scratchpad_md, eng, scratchpad_ptr);

        స్వయంచాలక matmul_prim = dnnl::matmul(matmul_pd);

        std::unordered_map<పూర్ణాంకం, dnnl::జ్ఞాపకం> matmul_వాదనలు;
        matmul_వాదనలు.insert({ DNNL_ARG_SRC, a_mem });
        matmul_వాదనలు.insert({ DNNL_ARG_WEIGHTS, b_mem });

        matmul_వాదనలు.insert({ DNNL_ARG_DST, c_mem });
        matmul_వాదనలు.insert({ DNNL_ARG_SCRATCHPAD, scratchpad_mem });

        matmul_prim.execute(stream, matmul_వాదనలు);
    }

    స్థిర శూన్య అడ్డువరుస_gemm(ggml_backend_sycl_context & ctx, పూర్ణాంకం m, పూర్ణాంకం n, పూర్ణాంకం k,
        స్థిరరాశి శూన్య * a, dt at, స్థిరరాశి శూన్య * b, dt bt, శూన్య * c, dt ct, స్థిరరాశి వరుస_ptr & q) {

        gemm(ctx, m, n, k, a, at, 1, k, k * m, b, bt, 1, k, n * k, c, ct, q, 1, 1);
    }
};

#అయితే_ముగింపు

#అయితే_ముగింపు // GGML_SYCL_GEMM_HPP
