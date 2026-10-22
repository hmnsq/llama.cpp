// SPDX-FileCopyrightText: Copyright 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
// SPDX-License-Identifier: MIT
//

#ہدایت once

#شامل "ggml.h"

شمار cpu_feature {
    CPU_FEATURE_NONE    = 0,
    CPU_FEATURE_DOTPROD = 1,
    CPU_FEATURE_I8MM    = 2,
    CPU_FEATURE_SVE     = 4,
    CPU_FEATURE_SME     = 8,
    CPU_FEATURE_SME2    = 16,
    CPU_FEATURE_FP16    = 32,
    CPU_FEATURE_SVEI8MM = 64
};

اندرونی cpu_feature& عامل|=(cpu_feature& lhs, cpu_feature rhs) {
    lhs = static_cast<cpu_feature>(lhs | rhs);
    واپس lhs;
}
اندرونی cpu_feature عامل|(cpu_feature lhs, cpu_feature rhs) {
    واپس static_cast<cpu_feature>(static_cast<صحیح>(lhs) | static_cast<صحیح>(rhs));
}

ساخت kernel_info {
    size_t (*get_m_قدم)(خالی);
    size_t (*get_n_قدم)(خالی);
    size_t (*get_mr)(خالی);
    size_t (*get_nr)(خالی);
    size_t (*get_kr)(خالی);
    size_t (*get_sr)(خالی);

    size_t (*get_dst_انحراف)(size_t m_idx, size_t n_idx, size_t stride);

    size_t (*get_lhs_انحراف_ex)(size_t m_idx, size_t k, size_t bl);

    size_t (*get_rhs_packed_انحراف_ex)(size_t n_idx, size_t k, size_t bl);

    خالی (*تسلسل_kernel_ex)(
        size_t m, size_t n, size_t k, size_t bl,
        ثابت خالی* lhs_packed, ثابت خالی* rhs_packed,
        خالی* dst, size_t dst_stride_صف, size_t dst_stride_ستون,
        اعشاری clamp_min, اعشاری clamp_max);
};

ساخت lhs_packing_info {
    size_t (*get_انحراف)(size_t m_idx, size_t lhs_stride);

    size_t (*get_packed_انحراف_ex)(size_t m_idx, size_t k, size_t bl, size_t mr, size_t kr, size_t sr);

    size_t (*packed_سائز_ex)(size_t m, size_t k, size_t bl, size_t mr, size_t kr, size_t sr);

    خالی (*pack_func_ex)(size_t m, size_t k, size_t bl, size_t mr, size_t kr, size_t sr,
        size_t m_idx_آغاز, ثابت خالی * lhs, size_t lhs_stride, خالی * lhs_packed);
};

شمار rhs_repack_منوال {
    RHS_REPACK_فی_KERNEL,
    RHS_REPACK_SINGLE_ONLY,
};

ساخت rhs_packing_info {
    size_t (*packed_سائز_ex)(size_t n, size_t k, size_t nr, size_t kr, size_t bl);

    خالی (*pack_func_ex)(size_t عدد_groups, size_t n, size_t k, size_t nr, size_t kr, size_t sr, size_t bl,
        size_t rhs_stride, ثابت خالی * rhs, ثابت خالی * bias, ثابت خالی * scale, خالی * rhs_packed, size_t extra_بائٹیں, ثابت خالی * params);

    rhs_repack_منوال repack_منوال = RHS_REPACK_فی_KERNEL;
};

ساخت ggml_kleidiai_kernels {
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
