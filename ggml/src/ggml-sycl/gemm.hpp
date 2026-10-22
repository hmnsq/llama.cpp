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

#اگر_غیر_تعریف_شدہ GGML_SYCL_GEMM_HPP
#تعریف_کرو GGML_SYCL_GEMM_HPP

#شامل "ggml-sycl.h"

#اگر GGML_SYCL_DNNL

#شامل "dnnl.hpp"
#شامل "dnnl_sycl.hpp"

جماعت DnnlGemmWrapper {
عوامی:
    استعمال dt = dnnl::حافظہ::ڈیٹا_type;
    استعمال لیبل = dnnl::حافظہ::format_لیبل;

    سانچہ<قسم_نام T>
    ساکن constexpr dt مقصد_dt() {
        اگر constexpr (std::is_same_v<T, اعشاری>) واپس dt::f32;
        ورنہ اگر constexpr (std::is_same_v<T, sycl::half>) واپس dt::f16;
#اگر_تعریف_شدہ GGML_SYCL_HAS_BF16
        ورنہ اگر constexpr (std::is_same_v<T, sycl::ext::oneapi::bfloat16>) واپس dt::bf16;
#اگر_ختم
        ورنہ static_assert(0);
    }

    ساکن خالی gemm(ggml_backend_sycl_context & ctx, صحیح m, صحیح n, صحیح k,
        ثابت خالی * a, dt at, dnnl_dim_t stra0, dnnl_dim_t stra1, dnnl_dim_t stra2,
        ثابت خالی * b, dt bt, dnnl_dim_t strb0, dnnl_dim_t strb1, dnnl_dim_t strb2,
        خالی * c, dt ct, ثابت قطار_ptr & q, dnnl_dim_t batches_a, dnnl_dim_t batches_b) {

        خودکار stream = ctx.دھارا_dnnl(q);
        خودکار eng = ctx.engine_dnnl(q);

        dnnl::حافظہ::dims a_dims = {batches_a, m, k };
        dnnl::حافظہ::dims a_strides = {stra2, stra1, stra0};
        ثابت خودکار a_اندر_md = dnnl::حافظہ::desc(a_dims, at, a_strides);

        dnnl::حافظہ::dims b_dims = {batches_b, k, n };
        dnnl::حافظہ::dims b_strides = {strb2, strb0, strb1};
        ثابت خودکار b_اندر_md = dnnl::حافظہ::desc(b_dims, bt, b_strides);

        dnnl::حافظہ::dims c_dims = { std::max(batches_a, batches_b), m, n};
        dnnl::حافظہ::dims c_strides = {m*n, 1,  m };
        ثابت خودکار c_md    = dnnl::حافظہ::desc(c_dims, ct, c_strides);
        dnnl::primitive_attr primitive_attr;
        primitive_attr.سیٹ_scratchpad_منوال(dnnl::scratchpad_منوال::user);

#اگر_تعریف_شدہ GGML_SYCL_F16
        primitive_attr.سیٹ_fpmath_منوال(dnnl::fpmath_منوال::f16);
#اگر_ختم

        خودکار a_mem = dnnl::حافظہ(a_اندر_md, eng, const_cast<خالی*>(a));
        خودکار b_mem = dnnl::حافظہ(b_اندر_md, eng, const_cast<خالی*>(b));
        خودکار matmul_pd = dnnl::matmul::primitive_desc(eng, a_اندر_md, b_اندر_md, c_md, primitive_attr);
        خودکار c_mem = dnnl::حافظہ(matmul_pd.dst_desc(), eng, c);

        ثابت خودکار scratchpad_md = matmul_pd.scratchpad_desc();
        ggml_sycl_pool_alloc<uint8_t> scratchpad(ctx.pool());
        خالی * scratchpad_ptr = scratchpad_md.get_سائز() > 0 ? scratchpad.alloc(scratchpad_md.get_سائز()) : خالی_اشاریہ;
        خودکار scratchpad_mem = dnnl::حافظہ(scratchpad_md, eng, scratchpad_ptr);

        خودکار matmul_prim = dnnl::matmul(matmul_pd);

        std::unordered_map<صحیح, dnnl::حافظہ> matmul_دلائل;
        matmul_دلائل.insert({ DNNL_ARG_SRC, a_mem });
        matmul_دلائل.insert({ DNNL_ARG_WEIGHTS, b_mem });

        matmul_دلائل.insert({ DNNL_ARG_DST, c_mem });
        matmul_دلائل.insert({ DNNL_ARG_SCRATCHPAD, scratchpad_mem });

        matmul_prim.execute(stream, matmul_دلائل);
    }

    ساکن خالی صف_gemm(ggml_backend_sycl_context & ctx, صحیح m, صحیح n, صحیح k,
        ثابت خالی * a, dt at, ثابت خالی * b, dt bt, خالی * c, dt ct, ثابت قطار_ptr & q) {

        gemm(ctx, m, n, k, a, at, 1, k, k * m, b, bt, 1, k, n * k, c, ct, q, 1, 1);
    }
};

#اگر_ختم

#اگر_ختم // GGML_SYCL_GEMM_HPP
