// SPDX-FileCopyrightText: Copyright 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
// SPDX-License-Identifier: MIT
//

#ਹਦਾਇਤ once

#ਅਨ੍ਤਰ੍ਭਾਵ "ggml.h"

ਗਿਣਤੀ cpu_feature {
    CPU_FEATURE_NONE    = 0,
    CPU_FEATURE_DOTPROD = 1,
    CPU_FEATURE_I8MM    = 2,
    CPU_FEATURE_SVE     = 4,
    CPU_FEATURE_SME     = 8,
    CPU_FEATURE_SME2    = 16,
    CPU_FEATURE_FP16    = 32,
    CPU_FEATURE_SVEI8MM = 64
};

ਸਤਰਅੰਦਰ cpu_feature& ਸੰਚਾਲਕ|=(cpu_feature& lhs, cpu_feature rhs) {
    lhs = static_cast<cpu_feature>(lhs | rhs);
    ਵਾਪਸ lhs;
}
ਸਤਰਅੰਦਰ cpu_feature ਸੰਚਾਲਕ|(cpu_feature lhs, cpu_feature rhs) {
    ਵਾਪਸ static_cast<cpu_feature>(static_cast<ਪੂਰਨ>(lhs) | static_cast<ਪੂਰਨ>(rhs));
}

ਬਣਤਰ kernel_info {
    size_t (*get_m_ਕਦਮ)(ਖਾਲੀ);
    size_t (*get_n_ਕਦਮ)(ਖਾਲੀ);
    size_t (*get_mr)(ਖਾਲੀ);
    size_t (*get_nr)(ਖਾਲੀ);
    size_t (*get_kr)(ਖਾਲੀ);
    size_t (*get_sr)(ਖਾਲੀ);

    size_t (*get_dst_ਵਿੱਥ)(size_t m_idx, size_t n_idx, size_t stride);

    size_t (*get_lhs_ਵਿੱਥ_ex)(size_t m_idx, size_t k, size_t bl);

    size_t (*get_rhs_packed_ਵਿੱਥ_ex)(size_t n_idx, size_t k, size_t bl);

    ਖਾਲੀ (*ਲੜੀਵਾਰ_kernel_ex)(
        size_t m, size_t n, size_t k, size_t bl,
        ਅਚਲ ਖਾਲੀ* lhs_packed, ਅਚਲ ਖਾਲੀ* rhs_packed,
        ਖਾਲੀ* dst, size_t dst_stride_ਪੰਕਤੀ, size_t dst_stride_ਖਾਨਾ,
        ਦਸ਼ਮਲਵ clamp_min, ਦਸ਼ਮਲਵ clamp_max);
};

ਬਣਤਰ lhs_packing_info {
    size_t (*get_ਵਿੱਥ)(size_t m_idx, size_t lhs_stride);

    size_t (*get_packed_ਵਿੱਥ_ex)(size_t m_idx, size_t k, size_t bl, size_t mr, size_t kr, size_t sr);

    size_t (*packed_ਮਾਪ_ex)(size_t m, size_t k, size_t bl, size_t mr, size_t kr, size_t sr);

    ਖਾਲੀ (*pack_func_ex)(size_t m, size_t k, size_t bl, size_t mr, size_t kr, size_t sr,
        size_t m_idx_ਸ਼ੁਰੂ, ਅਚਲ ਖਾਲੀ * lhs, size_t lhs_stride, ਖਾਲੀ * lhs_packed);
};

ਗਿਣਤੀ rhs_repack_ਬਹੁਲਕ {
    RHS_REPACK_ਪ੍ਰਤੀ_KERNEL,
    RHS_REPACK_SINGLE_ONLY,
};

ਬਣਤਰ rhs_packing_info {
    size_t (*packed_ਮਾਪ_ex)(size_t n, size_t k, size_t nr, size_t kr, size_t bl);

    ਖਾਲੀ (*pack_func_ex)(size_t ਅੰਕ_groups, size_t n, size_t k, size_t nr, size_t kr, size_t sr, size_t bl,
        size_t rhs_stride, ਅਚਲ ਖਾਲੀ * rhs, ਅਚਲ ਖਾਲੀ * bias, ਅਚਲ ਖਾਲੀ * scale, ਖਾਲੀ * rhs_packed, size_t extra_ਬਾਈਟਾਂ, ਅਚਲ ਖਾਲੀ * params);

    rhs_repack_ਬਹੁਲਕ repack_ਬਹੁਲਕ = RHS_REPACK_ਪ੍ਰਤੀ_KERNEL;
};

ਬਣਤਰ ggml_kleidiai_kernels {
    kernel_info      gemm;
    lhs_packing_info gemm_lhs_info;

    kernel_info      gemv;
    lhs_packing_info gemv_lhs_info;

    rhs_packing_info rhs_info;

    cpu_feature required_cpu;
    ggml_type lhs_type;
    ggml_type rhs_type;
    ggml_type op_type;
};

ggml_kleidiai_kernels * ggml_kleidiai_select_kernels(cpu_feature cpu_features, ਅਚਲ ggml_tensor * tensor);
ggml_kleidiai_kernels * ggml_kleidiai_select_kernels_q4_0(cpu_feature features);
ggml_kleidiai_kernels * ggml_kleidiai_select_kernels_q4_k(cpu_feature features);
ggml_kleidiai_kernels * ggml_kleidiai_select_kernels_q8_0(cpu_feature features);
ggml_kleidiai_kernels * ggml_kleidiai_select_kernels_f32(cpu_feature features);
