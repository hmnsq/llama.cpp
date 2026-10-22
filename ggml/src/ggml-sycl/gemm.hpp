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

#اگر_تعریف_نشده GGML_SYCL_GEMM_HPP
#تعریف_کن GGML_SYCL_GEMM_HPP

#ضمن "ggml-sycl.h"

#إذا GGML_SYCL_DNNL

#ضمن "dnnl.hpp"
#ضمن "dnnl_sycl.hpp"

صنف DnnlGemmWrapper {
عام:
    استفاده dt = dnnl::حافظه::داده_type;
    استفاده برچسب = dnnl::حافظه::format_برچسب;

    قالب<نام_نوع T>
    ساكن constexpr dt مقصد_dt() {
        إذا constexpr (std::is_same_v<T, عائم>) أرجع dt::f32;
        وإلا إذا constexpr (std::is_same_v<T, sycl::half>) أرجع dt::f16;
#اگر_تعریف_شده GGML_SYCL_HAS_BF16
        وإلا إذا constexpr (std::is_same_v<T, sycl::ext::oneapi::bfloat16>) أرجع dt::bf16;
#پایان_اگر
        وإلا static_assert(0);
    }

    ساكن فراغ gemm(ggml_backend_sycl_context & ctx, صحيح m, صحيح n, صحيح k,
        ثابت فراغ * a, dt at, dnnl_dim_t stra0, dnnl_dim_t stra1, dnnl_dim_t stra2,
        ثابت فراغ * b, dt bt, dnnl_dim_t strb0, dnnl_dim_t strb1, dnnl_dim_t strb2,
        فراغ * c, dt ct, ثابت صف_ptr & q, dnnl_dim_t batches_a, dnnl_dim_t batches_b) {

        خودکار stream = ctx.جریان_dnnl(q);
        خودکار eng = ctx.engine_dnnl(q);

        dnnl::حافظه::dims a_dims = {batches_a, m, k };
        dnnl::حافظه::dims a_strides = {stra2, stra1, stra0};
        ثابت خودکار a_درون_md = dnnl::حافظه::desc(a_dims, at, a_strides);

        dnnl::حافظه::dims b_dims = {batches_b, k, n };
        dnnl::حافظه::dims b_strides = {strb2, strb0, strb1};
        ثابت خودکار b_درون_md = dnnl::حافظه::desc(b_dims, bt, b_strides);

        dnnl::حافظه::dims c_dims = { std::max(batches_a, batches_b), m, n};
        dnnl::حافظه::dims c_strides = {m*n, 1,  m };
        ثابت خودکار c_md    = dnnl::حافظه::desc(c_dims, ct, c_strides);
        dnnl::primitive_attr primitive_attr;
        primitive_attr.مجموعه_scratchpad_نما(dnnl::scratchpad_نما::user);

#اگر_تعریف_شده GGML_SYCL_F16
        primitive_attr.مجموعه_fpmath_نما(dnnl::fpmath_نما::f16);
#پایان_اگر

        خودکار a_mem = dnnl::حافظه(a_درون_md, eng, const_cast<فراغ*>(a));
        خودکار b_mem = dnnl::حافظه(b_درون_md, eng, const_cast<فراغ*>(b));
        خودکار matmul_pd = dnnl::matmul::primitive_desc(eng, a_درون_md, b_درون_md, c_md, primitive_attr);
        خودکار c_mem = dnnl::حافظه(matmul_pd.dst_desc(), eng, c);

        ثابت خودکار scratchpad_md = matmul_pd.scratchpad_desc();
        ggml_sycl_pool_alloc<uint8_t> scratchpad(ctx.pool());
        فراغ * scratchpad_ptr = scratchpad_md.get_اندازه() > 0 ? scratchpad.alloc(scratchpad_md.get_اندازه()) : اشاره_تهی;
        خودکار scratchpad_mem = dnnl::حافظه(scratchpad_md, eng, scratchpad_ptr);

        خودکار matmul_prim = dnnl::matmul(matmul_pd);

        std::unordered_map<صحيح, dnnl::حافظه> matmul_آرگومانها;
        matmul_آرگومانها.insert({ DNNL_ARG_SRC, a_mem });
        matmul_آرگومانها.insert({ DNNL_ARG_WEIGHTS, b_mem });

        matmul_آرگومانها.insert({ DNNL_ARG_DST, c_mem });
        matmul_آرگومانها.insert({ DNNL_ARG_SCRATCHPAD, scratchpad_mem });

        matmul_prim.execute(stream, matmul_آرگومانها);
    }

    ساكن فراغ ردیف_gemm(ggml_backend_sycl_context & ctx, صحيح m, صحيح n, صحيح k,
        ثابت فراغ * a, dt at, ثابت فراغ * b, dt bt, فراغ * c, dt ct, ثابت صف_ptr & q) {

        gemm(ctx, m, n, k, a, at, 1, k, k * m, b, bt, 1, k, n * k, c, ct, q, 1, 1);
    }
};

#پایان_اگر

#پایان_اگر // GGML_SYCL_GEMM_HPP
