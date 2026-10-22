// SPDX-FileCopyrightText: Copyright 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
// SPDX-License-Identifier: MIT
//

#pragma once

#ಅನ್ತರ್ಭಾವ "ggml.h"

ಗಣನಾ cpu_feature {
    CPU_FEATURE_NONE    = 0,
    CPU_FEATURE_DOTPROD = 1,
    CPU_FEATURE_I8MM    = 2,
    CPU_FEATURE_SVE     = 4,
    CPU_FEATURE_SME     = 8,
    CPU_FEATURE_SME2    = 16,
    CPU_FEATURE_FP16    = 32,
    CPU_FEATURE_SVEI8MM = 64
};

ಅನ್ತಃಸ್ಥ cpu_feature& operator|=(cpu_feature& lhs, cpu_feature rhs) {
    lhs = static_cast<cpu_feature>(lhs | rhs);
    ಪ್ರತ್ಯಾಗಮ lhs;
}
ಅನ್ತಃಸ್ಥ cpu_feature operator|(cpu_feature lhs, cpu_feature rhs) {
    ಪ್ರತ್ಯಾಗಮ static_cast<cpu_feature>(static_cast<ಪೂರ್ಣಾಙ್ಕ>(lhs) | static_cast<ಪೂರ್ಣಾಙ್ಕ>(rhs));
}

ಸಂರಚನಾ kernel_info {
    size_t (*get_m_step)(ಶೂನ್ಯ);
    size_t (*get_n_step)(ಶೂನ್ಯ);
    size_t (*get_mr)(ಶೂನ್ಯ);
    size_t (*get_nr)(ಶೂನ್ಯ);
    size_t (*get_kr)(ಶೂನ್ಯ);
    size_t (*get_sr)(ಶೂನ್ಯ);

    size_t (*get_dst_offset)(size_t m_idx, size_t n_idx, size_t stride);

    size_t (*get_lhs_offset_ex)(size_t m_idx, size_t k, size_t bl);

    size_t (*get_rhs_packed_offset_ex)(size_t n_idx, size_t k, size_t bl);

    ಶೂನ್ಯ (*run_kernel_ex)(
        size_t m, size_t n, size_t k, size_t bl,
        ಧ್ರುವ ಶೂನ್ಯ* lhs_packed, ಧ್ರುವ ಶೂನ್ಯ* rhs_packed,
        ಶೂನ್ಯ* dst, size_t dst_stride_row, size_t dst_stride_col,
        ಭಿನ್ನ clamp_min, ಭಿನ್ನ clamp_max);
};

ಸಂರಚನಾ lhs_packing_info {
    size_t (*get_offset)(size_t m_idx, size_t lhs_stride);

    size_t (*get_packed_offset_ex)(size_t m_idx, size_t k, size_t bl, size_t mr, size_t kr, size_t sr);

    size_t (*packed_size_ex)(size_t m, size_t k, size_t bl, size_t mr, size_t kr, size_t sr);

    ಶೂನ್ಯ (*pack_func_ex)(size_t m, size_t k, size_t bl, size_t mr, size_t kr, size_t sr,
        size_t m_idx_start, ಧ್ರುವ ಶೂನ್ಯ * lhs, size_t lhs_stride, ಶೂನ್ಯ * lhs_packed);
};

ಗಣನಾ rhs_repack_mode {
    RHS_REPACK_PER_KERNEL,
    RHS_REPACK_SINGLE_ONLY,
};

ಸಂರಚನಾ rhs_packing_info {
    size_t (*packed_size_ex)(size_t n, size_t k, size_t nr, size_t kr, size_t bl);

    ಶೂನ್ಯ (*pack_func_ex)(size_t ಅಂಕೆ_groups, size_t n, size_t k, size_t nr, size_t kr, size_t sr, size_t bl,
        size_t rhs_stride, ಧ್ರುವ ಶೂನ್ಯ * rhs, ಧ್ರುವ ಶೂನ್ಯ * bias, ಧ್ರುವ ಶೂನ್ಯ * scale, ಶೂನ್ಯ * rhs_packed, size_t extra_bytes, ಧ್ರುವ ಶೂನ್ಯ * params);

    rhs_repack_mode repack_mode = RHS_REPACK_PER_KERNEL;
};

ಸಂರಚನಾ ggml_kleidiai_kernels {
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

ggml_kleidiai_kernels * ggml_kleidiai_select_kernels(cpu_feature cpu_features, ಧ್ರುವ ggml_tensor * tensor);
ggml_kleidiai_kernels * ggml_kleidiai_select_kernels_q4_0(cpu_feature features);
ggml_kleidiai_kernels * ggml_kleidiai_select_kernels_q4_k(cpu_feature features);
ggml_kleidiai_kernels * ggml_kleidiai_select_kernels_q8_0(cpu_feature features);
ggml_kleidiai_kernels * ggml_kleidiai_select_kernels_f32(cpu_feature features);
