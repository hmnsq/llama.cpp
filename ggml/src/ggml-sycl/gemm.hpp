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

#ਜੇ_ਅਪਰਿਭਾਸ਼ਿਤ GGML_SYCL_GEMM_HPP
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_SYCL_GEMM_HPP

#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-sycl.h"

#ਜੇ GGML_SYCL_DNNL

#ਅਨ੍ਤਰ੍ਭਾਵ "dnnl.hpp"
#ਅਨ੍ਤਰ੍ਭਾਵ "dnnl_sycl.hpp"

ਵਰਗ DnnlGemmWrapper {
ਜਨਤਕ:
    ਵਰਤੋਂ dt = dnnl::ਸਮ੍ਰਿਤੀ::ਅੰਕੜੇ_type;
    ਵਰਤੋਂ ਟੈਗ = dnnl::ਸਮ੍ਰਿਤੀ::format_ਟੈਗ;

    ਸਾਂਚਾ<ਕਿਸਮ_ਨਾਂ T>
    ਸਥਿਰ constexpr dt ਮੰਜ਼ਿਲ_dt() {
        ਜੇ constexpr (std::is_same_v<T, ਦਸ਼ਮਲਵ>) ਵਾਪਸ dt::f32;
        ਨਹੀਂਤਾਂ ਜੇ constexpr (std::is_same_v<T, sycl::half>) ਵਾਪਸ dt::f16;
#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ GGML_SYCL_HAS_BF16
        ਨਹੀਂਤਾਂ ਜੇ constexpr (std::is_same_v<T, sycl::ext::oneapi::bfloat16>) ਵਾਪਸ dt::bf16;
#ਜੇ_ਅੰਤ
        ਨਹੀਂਤਾਂ static_assert(0);
    }

    ਸਥਿਰ ਖਾਲੀ gemm(ggml_backend_sycl_context & ctx, ਪੂਰਨ m, ਪੂਰਨ n, ਪੂਰਨ k,
        ਅਚਲ ਖਾਲੀ * a, dt at, dnnl_dim_t stra0, dnnl_dim_t stra1, dnnl_dim_t stra2,
        ਅਚਲ ਖਾਲੀ * b, dt bt, dnnl_dim_t strb0, dnnl_dim_t strb1, dnnl_dim_t strb2,
        ਖਾਲੀ * c, dt ct, ਅਚਲ ਕਤਾਰ_ptr & q, dnnl_dim_t batches_a, dnnl_dim_t batches_b) {

        ਸਵੈਚਾਲਿਤ stream = ctx.ਧਾਰਾ_dnnl(q);
        ਸਵੈਚਾਲਿਤ eng = ctx.engine_dnnl(q);

        dnnl::ਸਮ੍ਰਿਤੀ::dims a_dims = {batches_a, m, k };
        dnnl::ਸਮ੍ਰਿਤੀ::dims a_strides = {stra2, stra1, stra0};
        ਅਚਲ ਸਵੈਚਾਲਿਤ a_ਵਿੱਚ_md = dnnl::ਸਮ੍ਰਿਤੀ::desc(a_dims, at, a_strides);

        dnnl::ਸਮ੍ਰਿਤੀ::dims b_dims = {batches_b, k, n };
        dnnl::ਸਮ੍ਰਿਤੀ::dims b_strides = {strb2, strb0, strb1};
        ਅਚਲ ਸਵੈਚਾਲਿਤ b_ਵਿੱਚ_md = dnnl::ਸਮ੍ਰਿਤੀ::desc(b_dims, bt, b_strides);

        dnnl::ਸਮ੍ਰਿਤੀ::dims c_dims = { std::max(batches_a, batches_b), m, n};
        dnnl::ਸਮ੍ਰਿਤੀ::dims c_strides = {m*n, 1,  m };
        ਅਚਲ ਸਵੈਚਾਲਿਤ c_md    = dnnl::ਸਮ੍ਰਿਤੀ::desc(c_dims, ct, c_strides);
        dnnl::primitive_attr primitive_attr;
        primitive_attr.ਸੈੱਟ_scratchpad_ਬਹੁਲਕ(dnnl::scratchpad_ਬਹੁਲਕ::user);

#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ GGML_SYCL_F16
        primitive_attr.ਸੈੱਟ_fpmath_ਬਹੁਲਕ(dnnl::fpmath_ਬਹੁਲਕ::f16);
#ਜੇ_ਅੰਤ

        ਸਵੈਚਾਲਿਤ a_mem = dnnl::ਸਮ੍ਰਿਤੀ(a_ਵਿੱਚ_md, eng, const_cast<ਖਾਲੀ*>(a));
        ਸਵੈਚਾਲਿਤ b_mem = dnnl::ਸਮ੍ਰਿਤੀ(b_ਵਿੱਚ_md, eng, const_cast<ਖਾਲੀ*>(b));
        ਸਵੈਚਾਲਿਤ matmul_pd = dnnl::matmul::primitive_desc(eng, a_ਵਿੱਚ_md, b_ਵਿੱਚ_md, c_md, primitive_attr);
        ਸਵੈਚਾਲਿਤ c_mem = dnnl::ਸਮ੍ਰਿਤੀ(matmul_pd.dst_desc(), eng, c);

        ਅਚਲ ਸਵੈਚਾਲਿਤ scratchpad_md = matmul_pd.scratchpad_desc();
        ggml_sycl_pool_alloc<uint8_t> scratchpad(ctx.pool());
        ਖਾਲੀ * scratchpad_ptr = scratchpad_md.get_ਮਾਪ() > 0 ? scratchpad.alloc(scratchpad_md.get_ਮਾਪ()) : ਖਾਲੀ_ਸੂਚਕ;
        ਸਵੈਚਾਲਿਤ scratchpad_mem = dnnl::ਸਮ੍ਰਿਤੀ(scratchpad_md, eng, scratchpad_ptr);

        ਸਵੈਚਾਲਿਤ matmul_prim = dnnl::matmul(matmul_pd);

        std::unordered_map<ਪੂਰਨ, dnnl::ਸਮ੍ਰਿਤੀ> matmul_ਦਲੀਲਾਂ;
        matmul_ਦਲੀਲਾਂ.insert({ DNNL_ARG_SRC, a_mem });
        matmul_ਦਲੀਲਾਂ.insert({ DNNL_ARG_WEIGHTS, b_mem });

        matmul_ਦਲੀਲਾਂ.insert({ DNNL_ARG_DST, c_mem });
        matmul_ਦਲੀਲਾਂ.insert({ DNNL_ARG_SCRATCHPAD, scratchpad_mem });

        matmul_prim.execute(stream, matmul_ਦਲੀਲਾਂ);
    }

    ਸਥਿਰ ਖਾਲੀ ਪੰਕਤੀ_gemm(ggml_backend_sycl_context & ctx, ਪੂਰਨ m, ਪੂਰਨ n, ਪੂਰਨ k,
        ਅਚਲ ਖਾਲੀ * a, dt at, ਅਚਲ ਖਾਲੀ * b, dt bt, ਖਾਲੀ * c, dt ct, ਅਚਲ ਕਤਾਰ_ptr & q) {

        gemm(ctx, m, n, k, a, at, 1, k, k * m, b, bt, 1, k, n * k, c, ct, q, 1, 1);
    }
};

#ਜੇ_ਅੰਤ

#ਜੇ_ਅੰਤ // GGML_SYCL_GEMM_HPP
