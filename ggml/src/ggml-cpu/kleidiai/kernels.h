// SPDX-FileCopyrightText: Copyright 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
// SPDX-License-Identifier: MIT
//

#ಸೂಚನೆ once

#ಅನ್ತರ್ಭಾವ "ggml.h"

ಎಣಿಕೆ cpu_feature {
    CPU_FEATURE_NONE    = 0,
    CPU_FEATURE_DOTPROD = 1,
    CPU_FEATURE_I8MM    = 2,
    CPU_FEATURE_SVE     = 4,
    CPU_FEATURE_SME     = 8,
    CPU_FEATURE_SME2    = 16,
    CPU_FEATURE_FP16    = 32,
    CPU_FEATURE_SVEI8MM = 64
};

ಸಾಲೊಳಗೆ cpu_feature& ಕ್ರಿಯಾಕಾರಕ|=(cpu_feature& lhs, cpu_feature rhs) {
    lhs = static_cast<cpu_feature>(lhs | rhs);
    ಹಿಂತಿರುಗಿ lhs;
}
ಸಾಲೊಳಗೆ cpu_feature ಕ್ರಿಯಾಕಾರಕ|(cpu_feature lhs, cpu_feature rhs) {
    ಹಿಂತಿರುಗಿ static_cast<cpu_feature>(static_cast<ಪೂರ್ಣಾಂಕ>(lhs) | static_cast<ಪೂರ್ಣಾಂಕ>(rhs));
}

ರಚನೆ kernel_info {
    size_t (*get_m_ಹೆಜ್ಜೆ)(ಶೂನ್ಯ);
    size_t (*get_n_ಹೆಜ್ಜೆ)(ಶೂನ್ಯ);
    size_t (*get_mr)(ಶೂನ್ಯ);
    size_t (*get_nr)(ಶೂನ್ಯ);
    size_t (*get_kr)(ಶೂನ್ಯ);
    size_t (*get_sr)(ಶೂನ್ಯ);

    size_t (*get_dst_ಸ್ಥಾನಾಂತರ)(size_t m_idx, size_t n_idx, size_t stride);

    size_t (*get_lhs_ಸ್ಥಾನಾಂತರ_ex)(size_t m_idx, size_t k, size_t bl);

    size_t (*get_rhs_packed_ಸ್ಥಾನಾಂತರ_ex)(size_t n_idx, size_t k, size_t bl);

    ಶೂನ್ಯ (*ಸತತ_kernel_ex)(
        size_t m, size_t n, size_t k, size_t bl,
        ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ* lhs_packed, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ* rhs_packed,
        ಶೂನ್ಯ* dst, size_t dst_stride_ಅಡ್ಡಸಾಲು, size_t dst_stride_ಲಂಬಸಾಲು,
        ದಶಮಾಂಶ clamp_min, ದಶಮಾಂಶ clamp_max);
};

ರಚನೆ lhs_packing_info {
    size_t (*get_ಸ್ಥಾನಾಂತರ)(size_t m_idx, size_t lhs_stride);

    size_t (*get_packed_ಸ್ಥಾನಾಂತರ_ex)(size_t m_idx, size_t k, size_t bl, size_t mr, size_t kr, size_t sr);

    size_t (*packed_ಗಾತ್ರ_ex)(size_t m, size_t k, size_t bl, size_t mr, size_t kr, size_t sr);

    ಶೂನ್ಯ (*pack_func_ex)(size_t m, size_t k, size_t bl, size_t mr, size_t kr, size_t sr,
        size_t m_idx_ಆರಂಭ, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * lhs, size_t lhs_stride, ಶೂನ್ಯ * lhs_packed);
};

ಎಣಿಕೆ rhs_repack_ಬಹುಲಕ {
    RHS_REPACK_ಪ್ರತಿ_KERNEL,
    RHS_REPACK_SINGLE_ONLY,
};

ರಚನೆ rhs_packing_info {
    size_t (*packed_ಗಾತ್ರ_ex)(size_t n, size_t k, size_t nr, size_t kr, size_t bl);

    ಶೂನ್ಯ (*pack_func_ex)(size_t ಅಂಕೆ_groups, size_t n, size_t k, size_t nr, size_t kr, size_t sr, size_t bl,
        size_t rhs_stride, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * rhs, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * bias, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * scale, ಶೂನ್ಯ * rhs_packed, size_t extra_ಬೈಟುಗಳು, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * params);

    rhs_repack_ಬಹುಲಕ repack_ಬಹುಲಕ = RHS_REPACK_ಪ್ರತಿ_KERNEL;
};

ರಚನೆ ggml_kleidiai_kernels {
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

ggml_kleidiai_kernels * ggml_kleidiai_select_kernels(cpu_feature cpu_features, ಸ್ಥಿರಾಂಕ ggml_tensor * tensor);
ggml_kleidiai_kernels * ggml_kleidiai_select_kernels_q4_0(cpu_feature features);
ggml_kleidiai_kernels * ggml_kleidiai_select_kernels_q4_k(cpu_feature features);
ggml_kleidiai_kernels * ggml_kleidiai_select_kernels_q8_0(cpu_feature features);
ggml_kleidiai_kernels * ggml_kleidiai_select_kernels_f32(cpu_feature features);
