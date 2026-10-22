// SPDX-FileCopyrightText: Copyright 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
// SPDX-License-Identifier: MIT
//

#pragma once

#அந்தர்பாவ "ggml.h"

எண்ணிக்கை cpu_feature {
    CPU_FEATURE_NONE    = 0,
    CPU_FEATURE_DOTPROD = 1,
    CPU_FEATURE_I8MM    = 2,
    CPU_FEATURE_SVE     = 4,
    CPU_FEATURE_SME     = 8,
    CPU_FEATURE_SME2    = 16,
    CPU_FEATURE_FP16    = 32,
    CPU_FEATURE_SVEI8MM = 64
};

உள்வரி cpu_feature& operator|=(cpu_feature& lhs, cpu_feature rhs) {
    lhs = static_cast<cpu_feature>(lhs | rhs);
    திரும்பு lhs;
}
உள்வரி cpu_feature operator|(cpu_feature lhs, cpu_feature rhs) {
    திரும்பு static_cast<cpu_feature>(static_cast<முழு>(lhs) | static_cast<முழு>(rhs));
}

கட்டமைப்பு kernel_info {
    size_t (*get_m_step)(வெறுமை);
    size_t (*get_n_step)(வெறுமை);
    size_t (*get_mr)(வெறுமை);
    size_t (*get_nr)(வெறுமை);
    size_t (*get_kr)(வெறுமை);
    size_t (*get_sr)(வெறுமை);

    size_t (*get_dst_offset)(size_t m_idx, size_t n_idx, size_t stride);

    size_t (*get_lhs_offset_ex)(size_t m_idx, size_t k, size_t bl);

    size_t (*get_rhs_packed_offset_ex)(size_t n_idx, size_t k, size_t bl);

    வெறுமை (*run_kernel_ex)(
        size_t m, size_t n, size_t k, size_t bl,
        மாறிலி வெறுமை* lhs_packed, மாறிலி வெறுமை* rhs_packed,
        வெறுமை* dst, size_t dst_stride_row, size_t dst_stride_col,
        தசம clamp_min, தசம clamp_max);
};

கட்டமைப்பு lhs_packing_info {
    size_t (*get_offset)(size_t m_idx, size_t lhs_stride);

    size_t (*get_packed_offset_ex)(size_t m_idx, size_t k, size_t bl, size_t mr, size_t kr, size_t sr);

    size_t (*packed_size_ex)(size_t m, size_t k, size_t bl, size_t mr, size_t kr, size_t sr);

    வெறுமை (*pack_func_ex)(size_t m, size_t k, size_t bl, size_t mr, size_t kr, size_t sr,
        size_t m_idx_start, மாறிலி வெறுமை * lhs, size_t lhs_stride, வெறுமை * lhs_packed);
};

எண்ணிக்கை rhs_repack_mode {
    RHS_REPACK_PER_KERNEL,
    RHS_REPACK_SINGLE_ONLY,
};

கட்டமைப்பு rhs_packing_info {
    size_t (*packed_size_ex)(size_t n, size_t k, size_t nr, size_t kr, size_t bl);

    வெறுமை (*pack_func_ex)(size_t இலக்கம்_groups, size_t n, size_t k, size_t nr, size_t kr, size_t sr, size_t bl,
        size_t rhs_stride, மாறிலி வெறுமை * rhs, மாறிலி வெறுமை * bias, மாறிலி வெறுமை * scale, வெறுமை * rhs_packed, size_t extra_bytes, மாறிலி வெறுமை * params);

    rhs_repack_mode repack_mode = RHS_REPACK_PER_KERNEL;
};

கட்டமைப்பு ggml_kleidiai_kernels {
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

ggml_kleidiai_kernels * ggml_kleidiai_select_kernels(cpu_feature cpu_features, மாறிலி ggml_tensor * tensor);
ggml_kleidiai_kernels * ggml_kleidiai_select_kernels_q4_0(cpu_feature features);
ggml_kleidiai_kernels * ggml_kleidiai_select_kernels_q4_k(cpu_feature features);
ggml_kleidiai_kernels * ggml_kleidiai_select_kernels_q8_0(cpu_feature features);
ggml_kleidiai_kernels * ggml_kleidiai_select_kernels_f32(cpu_feature features);
