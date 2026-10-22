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

#ವ್ಯಾಖ್ಯಾನಿತವಾಗದಿದ್ದರೆ GGML_SYCL_GEMM_HPP
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_SYCL_GEMM_HPP

#ಅನ್ತರ್ಭಾವ "ggml-sycl.h"

#ಆದರೆ GGML_SYCL_DNNL

#ಅನ್ತರ್ಭಾವ "dnnl.hpp"
#ಅನ್ತರ್ಭಾವ "dnnl_sycl.hpp"

ವರ್ಗ DnnlGemmWrapper {
ಸಾರ್ವಜನಿಕ:
    ಬಳಸಿ dt = dnnl::ಸ್ಮರಣೆ::ದತ್ತಾಂಶ_type;
    ಬಳಸಿ ಟ್ಯಾಗ್ = dnnl::ಸ್ಮರಣೆ::format_ಟ್ಯಾಗ್;

    ಮಾದರಿ<ಬಗೆ_ಹೆಸರು T>
    ಸ್ಥಿರ constexpr dt ಗಮ್ಯ_dt() {
        ಆದರೆ constexpr (std::is_same_v<T, ದಶಮಾಂಶ>) ಹಿಂತಿರುಗಿ dt::f32;
        ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ constexpr (std::is_same_v<T, sycl::half>) ಹಿಂತಿರುಗಿ dt::f16;
#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ GGML_SYCL_HAS_BF16
        ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ constexpr (std::is_same_v<T, sycl::ext::oneapi::bfloat16>) ಹಿಂತಿರುಗಿ dt::bf16;
#ಆದರೆ_ಅಂತ್ಯ
        ಇಲ್ಲದಿದ್ದರೆ static_assert(0);
    }

    ಸ್ಥಿರ ಶೂನ್ಯ gemm(ggml_backend_sycl_context & ctx, ಪೂರ್ಣಾಂಕ m, ಪೂರ್ಣಾಂಕ n, ಪೂರ್ಣಾಂಕ k,
        ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * a, dt at, dnnl_dim_t stra0, dnnl_dim_t stra1, dnnl_dim_t stra2,
        ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * b, dt bt, dnnl_dim_t strb0, dnnl_dim_t strb1, dnnl_dim_t strb2,
        ಶೂನ್ಯ * c, dt ct, ಸ್ಥಿರಾಂಕ ಸರತಿ_ptr & q, dnnl_dim_t batches_a, dnnl_dim_t batches_b) {

        ಸ್ವಯಂಚಾಲಿತ stream = ctx.ಹರಿವು_dnnl(q);
        ಸ್ವಯಂಚಾಲಿತ eng = ctx.engine_dnnl(q);

        dnnl::ಸ್ಮರಣೆ::dims a_dims = {batches_a, m, k };
        dnnl::ಸ್ಮರಣೆ::dims a_strides = {stra2, stra1, stra0};
        ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ a_ಒಳ_md = dnnl::ಸ್ಮರಣೆ::desc(a_dims, at, a_strides);

        dnnl::ಸ್ಮರಣೆ::dims b_dims = {batches_b, k, n };
        dnnl::ಸ್ಮರಣೆ::dims b_strides = {strb2, strb0, strb1};
        ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ b_ಒಳ_md = dnnl::ಸ್ಮರಣೆ::desc(b_dims, bt, b_strides);

        dnnl::ಸ್ಮರಣೆ::dims c_dims = { std::max(batches_a, batches_b), m, n};
        dnnl::ಸ್ಮರಣೆ::dims c_strides = {m*n, 1,  m };
        ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ c_md    = dnnl::ಸ್ಮರಣೆ::desc(c_dims, ct, c_strides);
        dnnl::primitive_attr primitive_attr;
        primitive_attr.ಗಣ_scratchpad_ಬಹುಲಕ(dnnl::scratchpad_ಬಹುಲಕ::user);

#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ GGML_SYCL_F16
        primitive_attr.ಗಣ_fpmath_ಬಹುಲಕ(dnnl::fpmath_ಬಹುಲಕ::f16);
#ಆದರೆ_ಅಂತ್ಯ

        ಸ್ವಯಂಚಾಲಿತ a_mem = dnnl::ಸ್ಮರಣೆ(a_ಒಳ_md, eng, const_cast<ಶೂನ್ಯ*>(a));
        ಸ್ವಯಂಚಾಲಿತ b_mem = dnnl::ಸ್ಮರಣೆ(b_ಒಳ_md, eng, const_cast<ಶೂನ್ಯ*>(b));
        ಸ್ವಯಂಚಾಲಿತ matmul_pd = dnnl::matmul::primitive_desc(eng, a_ಒಳ_md, b_ಒಳ_md, c_md, primitive_attr);
        ಸ್ವಯಂಚಾಲಿತ c_mem = dnnl::ಸ್ಮರಣೆ(matmul_pd.dst_desc(), eng, c);

        ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ scratchpad_md = matmul_pd.scratchpad_desc();
        ggml_sycl_pool_alloc<uint8_t> scratchpad(ctx.pool());
        ಶೂನ್ಯ * scratchpad_ptr = scratchpad_md.get_ಗಾತ್ರ() > 0 ? scratchpad.alloc(scratchpad_md.get_ಗಾತ್ರ()) : ಶೂನ್ಯಸೂಚಕ;
        ಸ್ವಯಂಚಾಲಿತ scratchpad_mem = dnnl::ಸ್ಮರಣೆ(scratchpad_md, eng, scratchpad_ptr);

        ಸ್ವಯಂಚಾಲಿತ matmul_prim = dnnl::matmul(matmul_pd);

        std::unordered_map<ಪೂರ್ಣಾಂಕ, dnnl::ಸ್ಮರಣೆ> matmul_ವಾದಗಳು;
        matmul_ವಾದಗಳು.insert({ DNNL_ARG_SRC, a_mem });
        matmul_ವಾದಗಳು.insert({ DNNL_ARG_WEIGHTS, b_mem });

        matmul_ವಾದಗಳು.insert({ DNNL_ARG_DST, c_mem });
        matmul_ವಾದಗಳು.insert({ DNNL_ARG_SCRATCHPAD, scratchpad_mem });

        matmul_prim.execute(stream, matmul_ವಾದಗಳು);
    }

    ಸ್ಥಿರ ಶೂನ್ಯ ಅಡ್ಡಸಾಲು_gemm(ggml_backend_sycl_context & ctx, ಪೂರ್ಣಾಂಕ m, ಪೂರ್ಣಾಂಕ n, ಪೂರ್ಣಾಂಕ k,
        ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * a, dt at, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * b, dt bt, ಶೂನ್ಯ * c, dt ct, ಸ್ಥಿರಾಂಕ ಸರತಿ_ptr & q) {

        gemm(ctx, m, n, k, a, at, 1, k, k * m, b, bt, 1, k, n * k, c, ct, q, 1, 1);
    }
};

#ಆದರೆ_ಅಂತ್ಯ

#ಆದರೆ_ಅಂತ್ಯ // GGML_SYCL_GEMM_HPP
