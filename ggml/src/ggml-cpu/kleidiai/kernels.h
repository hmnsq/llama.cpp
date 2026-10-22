// SPDX-FileCopyrightText: Copyright 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
// SPDX-License-Identifier: MIT
//

#pragma once

#অন্তর্ভাব "ggml.h"

গণনা cpu_feature {
    CPU_FEATURE_NONE    = 0,
    CPU_FEATURE_DOTPROD = 1,
    CPU_FEATURE_I8MM    = 2,
    CPU_FEATURE_SVE     = 4,
    CPU_FEATURE_SME     = 8,
    CPU_FEATURE_SME2    = 16,
    CPU_FEATURE_FP16    = 32,
    CPU_FEATURE_SVEI8MM = 64
};

অন্তঃস্থ cpu_feature& operator|=(cpu_feature& lhs, cpu_feature rhs) {
    lhs = static_cast<cpu_feature>(lhs | rhs);
    প্রত্যাগম lhs;
}
অন্তঃস্থ cpu_feature operator|(cpu_feature lhs, cpu_feature rhs) {
    প্রত্যাগম static_cast<cpu_feature>(static_cast<পূর্ণাঙ্ক>(lhs) | static_cast<পূর্ণাঙ্ক>(rhs));
}

সংরচনা kernel_info {
    size_t (*get_m_step)(শূন্য);
    size_t (*get_n_step)(শূন্য);
    size_t (*get_mr)(শূন্য);
    size_t (*get_nr)(শূন্য);
    size_t (*get_kr)(শূন্য);
    size_t (*get_sr)(শূন্য);

    size_t (*get_dst_offset)(size_t m_idx, size_t n_idx, size_t stride);

    size_t (*get_lhs_offset_ex)(size_t m_idx, size_t k, size_t bl);

    size_t (*get_rhs_packed_offset_ex)(size_t n_idx, size_t k, size_t bl);

    শূন্য (*run_kernel_ex)(
        size_t m, size_t n, size_t k, size_t bl,
        ধ্রুব শূন্য* lhs_packed, ধ্রুব শূন্য* rhs_packed,
        শূন্য* dst, size_t dst_stride_row, size_t dst_stride_col,
        ভিন্ন clamp_min, ভিন্ন clamp_max);
};

সংরচনা lhs_packing_info {
    size_t (*get_offset)(size_t m_idx, size_t lhs_stride);

    size_t (*get_packed_offset_ex)(size_t m_idx, size_t k, size_t bl, size_t mr, size_t kr, size_t sr);

    size_t (*packed_size_ex)(size_t m, size_t k, size_t bl, size_t mr, size_t kr, size_t sr);

    শূন্য (*pack_func_ex)(size_t m, size_t k, size_t bl, size_t mr, size_t kr, size_t sr,
        size_t m_idx_start, ধ্রুব শূন্য * lhs, size_t lhs_stride, শূন্য * lhs_packed);
};

গণনা rhs_repack_mode {
    RHS_REPACK_PER_KERNEL,
    RHS_REPACK_SINGLE_ONLY,
};

সংরচনা rhs_packing_info {
    size_t (*packed_size_ex)(size_t n, size_t k, size_t nr, size_t kr, size_t bl);

    শূন্য (*pack_func_ex)(size_t অঙ্ক_groups, size_t n, size_t k, size_t nr, size_t kr, size_t sr, size_t bl,
        size_t rhs_stride, ধ্রুব শূন্য * rhs, ধ্রুব শূন্য * bias, ধ্রুব শূন্য * scale, শূন্য * rhs_packed, size_t extra_bytes, ধ্রুব শূন্য * params);

    rhs_repack_mode repack_mode = RHS_REPACK_PER_KERNEL;
};

সংরচনা ggml_kleidiai_kernels {
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

ggml_kleidiai_kernels * ggml_kleidiai_select_kernels(cpu_feature cpu_features, ধ্রুব ggml_tensor * tensor);
ggml_kleidiai_kernels * ggml_kleidiai_select_kernels_q4_0(cpu_feature features);
ggml_kleidiai_kernels * ggml_kleidiai_select_kernels_q4_k(cpu_feature features);
ggml_kleidiai_kernels * ggml_kleidiai_select_kernels_q8_0(cpu_feature features);
ggml_kleidiai_kernels * ggml_kleidiai_select_kernels_f32(cpu_feature features);
