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

#अगर_अघोषित GGML_SYCL_GEMM_HPP
#घोषणा GGML_SYCL_GEMM_HPP

#समावेश "ggml-sycl.h"

#अगर GGML_SYCL_DNNL

#समावेश "dnnl.hpp"
#समावेश "dnnl_sycl.hpp"

श्रेणी DnnlGemmWrapper {
खुला:
    using dt = dnnl::memory::तथ्य_type;
    using tag = dnnl::memory::format_tag;

    ढाँचा<typename T>
    अटल constexpr dt to_dt() {
        अगर constexpr (std::is_same_v<T, भग्न>) वापस dt::f32;
        अन्यथा अगर constexpr (std::is_same_v<T, sycl::half>) वापस dt::f16;
#अगर_घोषित GGML_SYCL_HAS_BF16
        अन्यथा अगर constexpr (std::is_same_v<T, sycl::ext::oneapi::bfloat16>) वापस dt::bf16;
#पूर्ण_अगर
        अन्यथा static_assert(0);
    }

    अटल व्योम gemm(ggml_backend_sycl_context & ctx, पूर्णांक m, पूर्णांक n, पूर्णांक k,
        स्थिर व्योम * a, dt at, dnnl_dim_t stra0, dnnl_dim_t stra1, dnnl_dim_t stra2,
        स्थिर व्योम * b, dt bt, dnnl_dim_t strb0, dnnl_dim_t strb1, dnnl_dim_t strb2,
        व्योम * c, dt ct, स्थिर queue_ptr & q, dnnl_dim_t batches_a, dnnl_dim_t batches_b) {

        स्वतः stream = ctx.stream_dnnl(q);
        स्वतः eng = ctx.engine_dnnl(q);

        dnnl::memory::dims a_dims = {batches_a, m, k };
        dnnl::memory::dims a_strides = {stra2, stra1, stra0};
        स्थिर स्वतः a_in_md = dnnl::memory::desc(a_dims, at, a_strides);

        dnnl::memory::dims b_dims = {batches_b, k, n };
        dnnl::memory::dims b_strides = {strb2, strb0, strb1};
        स्थिर स्वतः b_in_md = dnnl::memory::desc(b_dims, bt, b_strides);

        dnnl::memory::dims c_dims = { std::max(batches_a, batches_b), m, n};
        dnnl::memory::dims c_strides = {m*n, 1,  m };
        स्थिर स्वतः c_md    = dnnl::memory::desc(c_dims, ct, c_strides);
        dnnl::primitive_attr primitive_attr;
        primitive_attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);

#अगर_घोषित GGML_SYCL_F16
        primitive_attr.set_fpmath_mode(dnnl::fpmath_mode::f16);
#पूर्ण_अगर

        स्वतः a_mem = dnnl::memory(a_in_md, eng, const_cast<व्योम*>(a));
        स्वतः b_mem = dnnl::memory(b_in_md, eng, const_cast<व्योम*>(b));
        स्वतः matmul_pd = dnnl::matmul::primitive_desc(eng, a_in_md, b_in_md, c_md, primitive_attr);
        स्वतः c_mem = dnnl::memory(matmul_pd.dst_desc(), eng, c);

        स्थिर स्वतः scratchpad_md = matmul_pd.scratchpad_desc();
        ggml_sycl_pool_alloc<uint8_t> scratchpad(ctx.pool());
        व्योम * scratchpad_ptr = scratchpad_md.get_आकार() > 0 ? scratchpad.alloc(scratchpad_md.get_आकार()) : nullptr;
        स्वतः scratchpad_mem = dnnl::memory(scratchpad_md, eng, scratchpad_ptr);

        स्वतः matmul_prim = dnnl::matmul(matmul_pd);

        std::unordered_map<पूर्णांक, dnnl::memory> matmul_args;
        matmul_args.insert({ DNNL_ARG_SRC, a_mem });
        matmul_args.insert({ DNNL_ARG_WEIGHTS, b_mem });

        matmul_args.insert({ DNNL_ARG_DST, c_mem });
        matmul_args.insert({ DNNL_ARG_SCRATCHPAD, scratchpad_mem });

        matmul_prim.execute(stream, matmul_args);
    }

    अटल व्योम row_gemm(ggml_backend_sycl_context & ctx, पूर्णांक m, पूर्णांक n, पूर्णांक k,
        स्थिर व्योम * a, dt at, स्थिर व्योम * b, dt bt, व्योम * c, dt ct, स्थिर queue_ptr & q) {

        gemm(ctx, m, n, k, a, at, 1, k, k * m, b, bt, 1, k, n * k, c, ct, q, 1, 1);
    }
};

#पूर्ण_अगर

#पूर्ण_अगर // GGML_SYCL_GEMM_HPP
