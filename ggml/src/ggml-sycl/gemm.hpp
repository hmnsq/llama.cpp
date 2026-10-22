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

#ifndef GGML_SYCL_GEMM_HPP
#define GGML_SYCL_GEMM_HPP

#ضمن "ggml-sycl.h"

#إذا GGML_SYCL_DNNL

#ضمن "dnnl.hpp"
#ضمن "dnnl_sycl.hpp"

صنف DnnlGemmWrapper {
عام:
    using dt = dnnl::memory::data_type;
    using tag = dnnl::memory::format_tag;

    template<typename T>
    ساكن constexpr dt to_dt() {
        إذا constexpr (std::is_same_v<T, عائم>) أرجع dt::f32;
        وإلا إذا constexpr (std::is_same_v<T, sycl::half>) أرجع dt::f16;
#ifdef GGML_SYCL_HAS_BF16
        وإلا إذا constexpr (std::is_same_v<T, sycl::ext::oneapi::bfloat16>) أرجع dt::bf16;
#endif
        وإلا static_assert(0);
    }

    ساكن فراغ gemm(ggml_backend_sycl_context & ctx, صحيح m, صحيح n, صحيح k,
        ثابت فراغ * a, dt at, dnnl_dim_t stra0, dnnl_dim_t stra1, dnnl_dim_t stra2,
        ثابت فراغ * b, dt bt, dnnl_dim_t strb0, dnnl_dim_t strb1, dnnl_dim_t strb2,
        فراغ * c, dt ct, ثابت queue_ptr & q, dnnl_dim_t batches_a, dnnl_dim_t batches_b) {

        auto stream = ctx.stream_dnnl(q);
        auto eng = ctx.engine_dnnl(q);

        dnnl::memory::dims a_dims = {batches_a, m, k };
        dnnl::memory::dims a_strides = {stra2, stra1, stra0};
        ثابت auto a_in_md = dnnl::memory::desc(a_dims, at, a_strides);

        dnnl::memory::dims b_dims = {batches_b, k, n };
        dnnl::memory::dims b_strides = {strb2, strb0, strb1};
        ثابت auto b_in_md = dnnl::memory::desc(b_dims, bt, b_strides);

        dnnl::memory::dims c_dims = { std::max(batches_a, batches_b), m, n};
        dnnl::memory::dims c_strides = {m*n, 1,  m };
        ثابت auto c_md    = dnnl::memory::desc(c_dims, ct, c_strides);
        dnnl::primitive_attr primitive_attr;
        primitive_attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);

#ifdef GGML_SYCL_F16
        primitive_attr.set_fpmath_mode(dnnl::fpmath_mode::f16);
#endif

        auto a_mem = dnnl::memory(a_in_md, eng, const_cast<فراغ*>(a));
        auto b_mem = dnnl::memory(b_in_md, eng, const_cast<فراغ*>(b));
        auto matmul_pd = dnnl::matmul::primitive_desc(eng, a_in_md, b_in_md, c_md, primitive_attr);
        auto c_mem = dnnl::memory(matmul_pd.dst_desc(), eng, c);

        ثابت auto scratchpad_md = matmul_pd.scratchpad_desc();
        ggml_sycl_pool_alloc<uint8_t> scratchpad(ctx.pool());
        فراغ * scratchpad_ptr = scratchpad_md.get_size() > 0 ? scratchpad.alloc(scratchpad_md.get_size()) : nullptr;
        auto scratchpad_mem = dnnl::memory(scratchpad_md, eng, scratchpad_ptr);

        auto matmul_prim = dnnl::matmul(matmul_pd);

        std::unordered_map<صحيح, dnnl::memory> matmul_حجوجي;
        matmul_حجوجي.insert({ DNNL_ARG_SRC, a_mem });
        matmul_حجوجي.insert({ DNNL_ARG_WEIGHTS, b_mem });

        matmul_حجوجي.insert({ DNNL_ARG_DST, c_mem });
        matmul_حجوجي.insert({ DNNL_ARG_SCRATCHPAD, scratchpad_mem });

        matmul_prim.execute(stream, matmul_حجوجي);
    }

    ساكن فراغ row_gemm(ggml_backend_sycl_context & ctx, صحيح m, صحيح n, صحيح k,
        ثابت فراغ * a, dt at, ثابت فراغ * b, dt bt, فراغ * c, dt ct, ثابت queue_ptr & q) {

        gemm(ctx, m, n, k, a, at, 1, k, k * m, b, bt, 1, k, n * k, c, ct, q, 1, 1);
    }
};

#endif

#endif // GGML_SYCL_GEMM_HPP
