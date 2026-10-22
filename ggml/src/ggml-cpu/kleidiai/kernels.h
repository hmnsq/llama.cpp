// SPDX-FileCopyrightText: Copyright 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
// SPDX-License-Identifier: MIT
//

#pragma once

#ضمن "ggml.h"

تعداد cpu_feature {
    CPU_FEATURE_NONE    = 0,
    CPU_FEATURE_DOTPROD = 1,
    CPU_FEATURE_I8MM    = 2,
    CPU_FEATURE_SVE     = 4,
    CPU_FEATURE_SME     = 8,
    CPU_FEATURE_SME2    = 16,
    CPU_FEATURE_FP16    = 32,
    CPU_FEATURE_SVEI8MM = 64
};

مضمن cpu_feature& operator|=(cpu_feature& lhs, cpu_feature rhs) {
    lhs = static_cast<cpu_feature>(lhs | rhs);
    أرجع lhs;
}
مضمن cpu_feature operator|(cpu_feature lhs, cpu_feature rhs) {
    أرجع static_cast<cpu_feature>(static_cast<صحيح>(lhs) | static_cast<صحيح>(rhs));
}

بنية kernel_info {
    size_t (*get_m_step)(فراغ);
    size_t (*get_n_step)(فراغ);
    size_t (*get_mr)(فراغ);
    size_t (*get_nr)(فراغ);
    size_t (*get_kr)(فراغ);
    size_t (*get_sr)(فراغ);

    size_t (*get_dst_offset)(size_t m_idx, size_t n_idx, size_t stride);

    size_t (*get_lhs_offset_ex)(size_t m_idx, size_t k, size_t bl);

    size_t (*get_rhs_packed_offset_ex)(size_t n_idx, size_t k, size_t bl);

    فراغ (*run_kernel_ex)(
        size_t m, size_t n, size_t k, size_t bl,
        ثابت فراغ* lhs_packed, ثابت فراغ* rhs_packed,
        فراغ* dst, size_t dst_stride_row, size_t dst_stride_col,
        عائم clamp_min, عائم clamp_max);
};

بنية lhs_packing_info {
    size_t (*get_offset)(size_t m_idx, size_t lhs_stride);

    size_t (*get_packed_offset_ex)(size_t m_idx, size_t k, size_t bl, size_t mr, size_t kr, size_t sr);

    size_t (*packed_size_ex)(size_t m, size_t k, size_t bl, size_t mr, size_t kr, size_t sr);

    فراغ (*pack_func_ex)(size_t m, size_t k, size_t bl, size_t mr, size_t kr, size_t sr,
        size_t m_idx_start, ثابت فراغ * lhs, size_t lhs_stride, فراغ * lhs_packed);
};

تعداد rhs_repack_mode {
    RHS_REPACK_PER_KERNEL,
    RHS_REPACK_SINGLE_ONLY,
};

بنية rhs_packing_info {
    size_t (*packed_size_ex)(size_t n, size_t k, size_t nr, size_t kr, size_t bl);

    فراغ (*pack_func_ex)(size_t num_groups, size_t n, size_t k, size_t nr, size_t kr, size_t sr, size_t bl,
        size_t rhs_stride, ثابت فراغ * rhs, ثابت فراغ * bias, ثابت فراغ * scale, فراغ * rhs_packed, size_t extra_bytes, ثابت فراغ * params);

    rhs_repack_mode repack_mode = RHS_REPACK_PER_KERNEL;
};

بنية ggml_kleidiai_kernels {
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

ggml_kleidiai_kernels * ggml_kleidiai_select_kernels(cpu_feature cpu_features, ثابت ggml_tensor * tensor);
ggml_kleidiai_kernels * ggml_kleidiai_select_kernels_q4_0(cpu_feature features);
ggml_kleidiai_kernels * ggml_kleidiai_select_kernels_q4_k(cpu_feature features);
ggml_kleidiai_kernels * ggml_kleidiai_select_kernels_q8_0(cpu_feature features);
ggml_kleidiai_kernels * ggml_kleidiai_select_kernels_f32(cpu_feature features);
