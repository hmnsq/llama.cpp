// SPDX-FileCopyrightText: Copyright 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
// SPDX-License-Identifier: MIT
//

#pragma once

#अन्तर्भाव "ggml.h"

गणना cpu_feature {
    CPU_FEATURE_NONE    = 0,
    CPU_FEATURE_DOTPROD = 1,
    CPU_FEATURE_I8MM    = 2,
    CPU_FEATURE_SVE     = 4,
    CPU_FEATURE_SME     = 8,
    CPU_FEATURE_SME2    = 16,
    CPU_FEATURE_FP16    = 32,
    CPU_FEATURE_SVEI8MM = 64
};

अन्तःस्थ cpu_feature& operator|=(cpu_feature& lhs, cpu_feature rhs) {
    lhs = static_cast<cpu_feature>(lhs | rhs);
    प्रत्यागम lhs;
}
अन्तःस्थ cpu_feature operator|(cpu_feature lhs, cpu_feature rhs) {
    प्रत्यागम static_cast<cpu_feature>(static_cast<पूर्णाङ्क>(lhs) | static_cast<पूर्णाङ्क>(rhs));
}

संरचना kernel_info {
    size_t (*get_m_step)(शून्य);
    size_t (*get_n_step)(शून्य);
    size_t (*get_mr)(शून्य);
    size_t (*get_nr)(शून्य);
    size_t (*get_kr)(शून्य);
    size_t (*get_sr)(शून्य);

    size_t (*get_dst_offset)(size_t m_idx, size_t n_idx, size_t stride);

    size_t (*get_lhs_offset_ex)(size_t m_idx, size_t k, size_t bl);

    size_t (*get_rhs_packed_offset_ex)(size_t n_idx, size_t k, size_t bl);

    शून्य (*run_kernel_ex)(
        size_t m, size_t n, size_t k, size_t bl,
        ध्रुव शून्य* lhs_packed, ध्रुव शून्य* rhs_packed,
        शून्य* dst, size_t dst_stride_row, size_t dst_stride_col,
        भिन्न clamp_min, भिन्न clamp_max);
};

संरचना lhs_packing_info {
    size_t (*get_offset)(size_t m_idx, size_t lhs_stride);

    size_t (*get_packed_offset_ex)(size_t m_idx, size_t k, size_t bl, size_t mr, size_t kr, size_t sr);

    size_t (*packed_size_ex)(size_t m, size_t k, size_t bl, size_t mr, size_t kr, size_t sr);

    शून्य (*pack_func_ex)(size_t m, size_t k, size_t bl, size_t mr, size_t kr, size_t sr,
        size_t m_idx_start, ध्रुव शून्य * lhs, size_t lhs_stride, शून्य * lhs_packed);
};

गणना rhs_repack_mode {
    RHS_REPACK_PER_KERNEL,
    RHS_REPACK_SINGLE_ONLY,
};

संरचना rhs_packing_info {
    size_t (*packed_size_ex)(size_t n, size_t k, size_t nr, size_t kr, size_t bl);

    शून्य (*pack_func_ex)(size_t अङ्कः_groups, size_t n, size_t k, size_t nr, size_t kr, size_t sr, size_t bl,
        size_t rhs_stride, ध्रुव शून्य * rhs, ध्रुव शून्य * bias, ध्रुव शून्य * scale, शून्य * rhs_packed, size_t extra_bytes, ध्रुव शून्य * params);

    rhs_repack_mode repack_mode = RHS_REPACK_PER_KERNEL;
};

संरचना ggml_kleidiai_kernels {
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

ggml_kleidiai_kernels * ggml_kleidiai_select_kernels(cpu_feature cpu_features, ध्रुव ggml_tensor * tensor);
ggml_kleidiai_kernels * ggml_kleidiai_select_kernels_q4_0(cpu_feature features);
ggml_kleidiai_kernels * ggml_kleidiai_select_kernels_q4_k(cpu_feature features);
ggml_kleidiai_kernels * ggml_kleidiai_select_kernels_q8_0(cpu_feature features);
ggml_kleidiai_kernels * ggml_kleidiai_select_kernels_f32(cpu_feature features);
