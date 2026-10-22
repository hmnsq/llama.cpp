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

#यदि_अपरिभाषितम् GGML_SYCL_GEMM_HPP
#परिभाषय GGML_SYCL_GEMM_HPP

#अन्तर्भाव "ggml-sycl.h"

#यदि GGML_SYCL_DNNL

#अन्तर्भाव "dnnl.hpp"
#अन्तर्भाव "dnnl_sycl.hpp"

वर्ग DnnlGemmWrapper {
सार्वजनिक:
    उपयुज्य dt = dnnl::स्मृतिः::दत्त_type;
    उपयुज्य चिह्नकम् = dnnl::स्मृतिः::format_चिह्नकम्;

    प्रतिरूपम्<प्रकारनाम T>
    स्थिर constexpr dt गन्तव्यम्_dt() {
        यदि constexpr (std::is_same_v<T, भिन्न>) प्रत्यागम dt::f32;
        अन्यथा यदि constexpr (std::is_same_v<T, sycl::half>) प्रत्यागम dt::f16;
#यदि_परिभाषितम् GGML_SYCL_HAS_BF16
        अन्यथा यदि constexpr (std::is_same_v<T, sycl::ext::oneapi::bfloat16>) प्रत्यागम dt::bf16;
#यदि_अन्तः
        अन्यथा static_assert(0);
    }

    स्थिर शून्य gemm(ggml_backend_sycl_context & ctx, पूर्णाङ्क m, पूर्णाङ्क n, पूर्णाङ्क k,
        ध्रुव शून्य * a, dt at, dnnl_dim_t stra0, dnnl_dim_t stra1, dnnl_dim_t stra2,
        ध्रुव शून्य * b, dt bt, dnnl_dim_t strb0, dnnl_dim_t strb1, dnnl_dim_t strb2,
        शून्य * c, dt ct, ध्रुव queue_ptr & q, dnnl_dim_t batches_a, dnnl_dim_t batches_b) {

        स्वचालित stream = ctx.प्रवाहः_dnnl(q);
        स्वचालित eng = ctx.engine_dnnl(q);

        dnnl::स्मृतिः::dims a_dims = {batches_a, m, k };
        dnnl::स्मृतिः::dims a_strides = {stra2, stra1, stra0};
        ध्रुव स्वचालित a_अन्तर्_md = dnnl::स्मृतिः::desc(a_dims, at, a_strides);

        dnnl::स्मृतिः::dims b_dims = {batches_b, k, n };
        dnnl::स्मृतिः::dims b_strides = {strb2, strb0, strb1};
        ध्रुव स्वचालित b_अन्तर्_md = dnnl::स्मृतिः::desc(b_dims, bt, b_strides);

        dnnl::स्मृतिः::dims c_dims = { std::max(batches_a, batches_b), m, n};
        dnnl::स्मृतिः::dims c_strides = {m*n, 1,  m };
        ध्रुव स्वचालित c_md    = dnnl::स्मृतिः::desc(c_dims, ct, c_strides);
        dnnl::primitive_attr primitive_attr;
        primitive_attr.समुच्चयः_scratchpad_बहुलकम्(dnnl::scratchpad_बहुलकम्::user);

#यदि_परिभाषितम् GGML_SYCL_F16
        primitive_attr.समुच्चयः_fpmath_बहुलकम्(dnnl::fpmath_बहुलकम्::f16);
#यदि_अन्तः

        स्वचालित a_mem = dnnl::स्मृतिः(a_अन्तर्_md, eng, const_cast<शून्य*>(a));
        स्वचालित b_mem = dnnl::स्मृतिः(b_अन्तर्_md, eng, const_cast<शून्य*>(b));
        स्वचालित matmul_pd = dnnl::matmul::primitive_desc(eng, a_अन्तर्_md, b_अन्तर्_md, c_md, primitive_attr);
        स्वचालित c_mem = dnnl::स्मृतिः(matmul_pd.dst_desc(), eng, c);

        ध्रुव स्वचालित scratchpad_md = matmul_pd.scratchpad_desc();
        ggml_sycl_pool_alloc<uint8_t> scratchpad(ctx.pool());
        शून्य * scratchpad_ptr = scratchpad_md.get_आकार() > 0 ? scratchpad.alloc(scratchpad_md.get_आकार()) : शून्यसूचकः;
        स्वचालित scratchpad_mem = dnnl::स्मृतिः(scratchpad_md, eng, scratchpad_ptr);

        स्वचालित matmul_prim = dnnl::matmul(matmul_pd);

        std::unordered_map<पूर्णाङ्क, dnnl::स्मृतिः> matmul_तर्काः;
        matmul_तर्काः.insert({ DNNL_ARG_SRC, a_mem });
        matmul_तर्काः.insert({ DNNL_ARG_WEIGHTS, b_mem });

        matmul_तर्काः.insert({ DNNL_ARG_DST, c_mem });
        matmul_तर्काः.insert({ DNNL_ARG_SCRATCHPAD, scratchpad_mem });

        matmul_prim.execute(stream, matmul_तर्काः);
    }

    स्थिर शून्य row_gemm(ggml_backend_sycl_context & ctx, पूर्णाङ्क m, पूर्णाङ्क n, पूर्णाङ्क k,
        ध्रुव शून्य * a, dt at, ध्रुव शून्य * b, dt bt, शून्य * c, dt ct, ध्रुव queue_ptr & q) {

        gemm(ctx, m, n, k, a, at, 1, k, k * m, b, bt, 1, k, n * k, c, ct, q, 1, 1);
    }
};

#यदि_अन्तः

#यदि_अन्तः // GGML_SYCL_GEMM_HPP
