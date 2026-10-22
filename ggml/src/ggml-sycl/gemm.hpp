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

#جيڪڏهن_وصف_نه_ٿيل GGML_SYCL_GEMM_HPP
#وصف_ڪريو GGML_SYCL_GEMM_HPP

#ضمن "ggml-sycl.h"

#إذا GGML_SYCL_DNNL

#ضمن "dnnl.hpp"
#ضمن "dnnl_sycl.hpp"

صنف DnnlGemmWrapper {
عام:
    استعمال dt = dnnl::يادگيري::معلومات_type;
    استعمال ٽيگ = dnnl::يادگيري::format_ٽيگ;

    سانچو<قسم_نالو T>
    ساكن constexpr dt منزل_dt() {
        إذا constexpr (std::is_same_v<T, عائم>) أرجع dt::f32;
        وإلا إذا constexpr (std::is_same_v<T, sycl::half>) أرجع dt::f16;
#جيڪڏهن_وصف_ٿيل GGML_SYCL_HAS_BF16
        وإلا إذا constexpr (std::is_same_v<T, sycl::ext::oneapi::bfloat16>) أرجع dt::bf16;
#جيڪڏهن_پورو
        وإلا static_assert(0);
    }

    ساكن فراغ gemm(ggml_backend_sycl_context & ctx, صحيح m, صحيح n, صحيح k,
        ثابت فراغ * a, dt at, dnnl_dim_t stra0, dnnl_dim_t stra1, dnnl_dim_t stra2,
        ثابت فراغ * b, dt bt, dnnl_dim_t strb0, dnnl_dim_t strb1, dnnl_dim_t strb2,
        فراغ * c, dt ct, ثابت قطار_ptr & q, dnnl_dim_t batches_a, dnnl_dim_t batches_b) {

        خودڪار stream = ctx.وهڪرو_dnnl(q);
        خودڪار eng = ctx.engine_dnnl(q);

        dnnl::يادگيري::dims a_dims = {batches_a, m, k };
        dnnl::يادگيري::dims a_strides = {stra2, stra1, stra0};
        ثابت خودڪار a_اندريون_md = dnnl::يادگيري::desc(a_dims, at, a_strides);

        dnnl::يادگيري::dims b_dims = {batches_b, k, n };
        dnnl::يادگيري::dims b_strides = {strb2, strb0, strb1};
        ثابت خودڪار b_اندريون_md = dnnl::يادگيري::desc(b_dims, bt, b_strides);

        dnnl::يادگيري::dims c_dims = { std::max(batches_a, batches_b), m, n};
        dnnl::يادگيري::dims c_strides = {m*n, 1,  m };
        ثابت خودڪار c_md    = dnnl::يادگيري::desc(c_dims, ct, c_strides);
        dnnl::primitive_attr primitive_attr;
        primitive_attr.سيٽ_scratchpad_گهڻائي(dnnl::scratchpad_گهڻائي::user);

#جيڪڏهن_وصف_ٿيل GGML_SYCL_F16
        primitive_attr.سيٽ_fpmath_گهڻائي(dnnl::fpmath_گهڻائي::f16);
#جيڪڏهن_پورو

        خودڪار a_mem = dnnl::يادگيري(a_اندريون_md, eng, const_cast<فراغ*>(a));
        خودڪار b_mem = dnnl::يادگيري(b_اندريون_md, eng, const_cast<فراغ*>(b));
        خودڪار matmul_pd = dnnl::matmul::primitive_desc(eng, a_اندريون_md, b_اندريون_md, c_md, primitive_attr);
        خودڪار c_mem = dnnl::يادگيري(matmul_pd.dst_desc(), eng, c);

        ثابت خودڪار scratchpad_md = matmul_pd.scratchpad_desc();
        ggml_sycl_pool_alloc<uint8_t> scratchpad(ctx.pool());
        فراغ * scratchpad_ptr = scratchpad_md.get_سائيز() > 0 ? scratchpad.alloc(scratchpad_md.get_سائيز()) : خالي_اشارو;
        خودڪار scratchpad_mem = dnnl::يادگيري(scratchpad_md, eng, scratchpad_ptr);

        خودڪار matmul_prim = dnnl::matmul(matmul_pd);

        std::unordered_map<صحيح, dnnl::يادگيري> matmul_دليلون;
        matmul_دليلون.insert({ DNNL_ARG_SRC, a_mem });
        matmul_دليلون.insert({ DNNL_ARG_WEIGHTS, b_mem });

        matmul_دليلون.insert({ DNNL_ARG_DST, c_mem });
        matmul_دليلون.insert({ DNNL_ARG_SCRATCHPAD, scratchpad_mem });

        matmul_prim.execute(stream, matmul_دليلون);
    }

    ساكن فراغ صف_gemm(ggml_backend_sycl_context & ctx, صحيح m, صحيح n, صحيح k,
        ثابت فراغ * a, dt at, ثابت فراغ * b, dt bt, فراغ * c, dt ct, ثابت قطار_ptr & q) {

        gemm(ctx, m, n, k, a, at, 1, k, k * m, b, bt, 1, k, n * k, c, ct, q, 1, 1);
    }
};

#جيڪڏهن_پورو

#جيڪڏهن_پورو // GGML_SYCL_GEMM_HPP
