// SPDX-FileCopyrightText: Copyright 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
// SPDX-License-Identifier: MIT
//

#आशय once

#समावेश "ggml.h"

क्रमागत cpu_feature {
    CPU_FEATURE_NONE    = 0,
    CPU_FEATURE_DOTPROD = 1,
    CPU_FEATURE_I8MM    = 2,
    CPU_FEATURE_SVE     = 4,
    CPU_FEATURE_SME     = 8,
    CPU_FEATURE_SME2    = 16,
    CPU_FEATURE_FP16    = 32,
    CPU_FEATURE_SVEI8MM = 64
};

अंतरभूत cpu_feature& चालक|=(cpu_feature& lhs, cpu_feature rhs) {
    lhs = static_cast<cpu_feature>(lhs | rhs);
    वापस lhs;
}
अंतरभूत cpu_feature चालक|(cpu_feature lhs, cpu_feature rhs) {
    वापस static_cast<cpu_feature>(static_cast<पूर्णांक>(lhs) | static_cast<पूर्णांक>(rhs));
}

काष्ठा kernel_info {
    माप_प्रकार (*get_m_step)(व्योम);
    माप_प्रकार (*get_n_step)(व्योम);
    माप_प्रकार (*get_mr)(व्योम);
    माप_प्रकार (*get_nr)(व्योम);
    माप_प्रकार (*get_kr)(व्योम);
    माप_प्रकार (*get_sr)(व्योम);

    माप_प्रकार (*get_dst_offset)(माप_प्रकार m_idx, माप_प्रकार n_idx, माप_प्रकार stride);

    माप_प्रकार (*get_lhs_offset_ex)(माप_प्रकार m_idx, माप_प्रकार k, माप_प्रकार bl);

    माप_प्रकार (*get_rhs_packed_offset_ex)(माप_प्रकार n_idx, माप_प्रकार k, माप_प्रकार bl);

    व्योम (*run_kernel_ex)(
        माप_प्रकार m, माप_प्रकार n, माप_प्रकार k, माप_प्रकार bl,
        स्थिर व्योम* lhs_packed, स्थिर व्योम* rhs_packed,
        व्योम* dst, माप_प्रकार dst_stride_row, माप_प्रकार dst_stride_col,
        भग्न clamp_min, भग्न clamp_max);
};

काष्ठा lhs_packing_info {
    माप_प्रकार (*get_offset)(माप_प्रकार m_idx, माप_प्रकार lhs_stride);

    माप_प्रकार (*get_packed_offset_ex)(माप_प्रकार m_idx, माप_प्रकार k, माप_प्रकार bl, माप_प्रकार mr, माप_प्रकार kr, माप_प्रकार sr);

    माप_प्रकार (*packed_आकार_ex)(माप_प्रकार m, माप_प्रकार k, माप_प्रकार bl, माप_प्रकार mr, माप_प्रकार kr, माप_प्रकार sr);

    व्योम (*pack_func_ex)(माप_प्रकार m, माप_प्रकार k, माप_प्रकार bl, माप_प्रकार mr, माप_प्रकार kr, माप_प्रकार sr,
        माप_प्रकार m_idx_start, स्थिर व्योम * lhs, माप_प्रकार lhs_stride, व्योम * lhs_packed);
};

क्रमागत rhs_repack_mode {
    RHS_REPACK_PER_KERNEL,
    RHS_REPACK_SINGLE_ONLY,
};

काष्ठा rhs_packing_info {
    माप_प्रकार (*packed_आकार_ex)(माप_प्रकार n, माप_प्रकार k, माप_प्रकार nr, माप_प्रकार kr, माप_प्रकार bl);

    व्योम (*pack_func_ex)(माप_प्रकार अंक_groups, माप_प्रकार n, माप_प्रकार k, माप_प्रकार nr, माप_प्रकार kr, माप_प्रकार sr, माप_प्रकार bl,
        माप_प्रकार rhs_stride, स्थिर व्योम * rhs, स्थिर व्योम * bias, स्थिर व्योम * scale, व्योम * rhs_packed, माप_प्रकार extra_bytes, स्थिर व्योम * params);

    rhs_repack_mode repack_mode = RHS_REPACK_PER_KERNEL;
};

काष्ठा ggml_kleidiai_kernels {
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

ggml_kleidiai_kernels * ggml_kleidiai_select_kernels(cpu_feature cpu_features, स्थिर ggml_tensor * tensor);
ggml_kleidiai_kernels * ggml_kleidiai_select_kernels_q4_0(cpu_feature features);
ggml_kleidiai_kernels * ggml_kleidiai_select_kernels_q4_k(cpu_feature features);
ggml_kleidiai_kernels * ggml_kleidiai_select_kernels_q8_0(cpu_feature features);
ggml_kleidiai_kernels * ggml_kleidiai_select_kernels_f32(cpu_feature features);
