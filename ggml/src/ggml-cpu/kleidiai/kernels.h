// SPDX-FileCopyrightText: Copyright 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
// SPDX-License-Identifier: MIT
//

#pragma once

#ଅନ୍ତର୍ଭାଵ "ggml.h"

ଗଣନା cpu_feature {
    CPU_FEATURE_NONE    = 0,
    CPU_FEATURE_DOTPROD = 1,
    CPU_FEATURE_I8MM    = 2,
    CPU_FEATURE_SVE     = 4,
    CPU_FEATURE_SME     = 8,
    CPU_FEATURE_SME2    = 16,
    CPU_FEATURE_FP16    = 32,
    CPU_FEATURE_SVEI8MM = 64
};

ଧାଡ଼ିଭିତରେ cpu_feature& operator|=(cpu_feature& lhs, cpu_feature rhs) {
    lhs = static_cast<cpu_feature>(lhs | rhs);
    ଫେରନ୍ତୁ lhs;
}
ଧାଡ଼ିଭିତରେ cpu_feature operator|(cpu_feature lhs, cpu_feature rhs) {
    ଫେରନ୍ତୁ static_cast<cpu_feature>(static_cast<ପୂର୍ଣ୍ଣସଂଖ୍ୟା>(lhs) | static_cast<ପୂର୍ଣ୍ଣସଂଖ୍ୟା>(rhs));
}

ସଂରଚନା kernel_info {
    size_t (*get_m_step)(ଶୂନ୍ୟ);
    size_t (*get_n_step)(ଶୂନ୍ୟ);
    size_t (*get_mr)(ଶୂନ୍ୟ);
    size_t (*get_nr)(ଶୂନ୍ୟ);
    size_t (*get_kr)(ଶୂନ୍ୟ);
    size_t (*get_sr)(ଶୂନ୍ୟ);

    size_t (*get_dst_offset)(size_t m_idx, size_t n_idx, size_t stride);

    size_t (*get_lhs_offset_ex)(size_t m_idx, size_t k, size_t bl);

    size_t (*get_rhs_packed_offset_ex)(size_t n_idx, size_t k, size_t bl);

    ଶୂନ୍ୟ (*run_kernel_ex)(
        size_t m, size_t n, size_t k, size_t bl,
        ଅଚଳ ଶୂନ୍ୟ* lhs_packed, ଅଚଳ ଶୂନ୍ୟ* rhs_packed,
        ଶୂନ୍ୟ* dst, size_t dst_stride_row, size_t dst_stride_col,
        ଦଶମିକ clamp_min, ଦଶମିକ clamp_max);
};

ସଂରଚନା lhs_packing_info {
    size_t (*get_offset)(size_t m_idx, size_t lhs_stride);

    size_t (*get_packed_offset_ex)(size_t m_idx, size_t k, size_t bl, size_t mr, size_t kr, size_t sr);

    size_t (*packed_size_ex)(size_t m, size_t k, size_t bl, size_t mr, size_t kr, size_t sr);

    ଶୂନ୍ୟ (*pack_func_ex)(size_t m, size_t k, size_t bl, size_t mr, size_t kr, size_t sr,
        size_t m_idx_start, ଅଚଳ ଶୂନ୍ୟ * lhs, size_t lhs_stride, ଶୂନ୍ୟ * lhs_packed);
};

ଗଣନା rhs_repack_mode {
    RHS_REPACK_PER_KERNEL,
    RHS_REPACK_SINGLE_ONLY,
};

ସଂରଚନା rhs_packing_info {
    size_t (*packed_size_ex)(size_t n, size_t k, size_t nr, size_t kr, size_t bl);

    ଶୂନ୍ୟ (*pack_func_ex)(size_t ଅଙ୍କ_groups, size_t n, size_t k, size_t nr, size_t kr, size_t sr, size_t bl,
        size_t rhs_stride, ଅଚଳ ଶୂନ୍ୟ * rhs, ଅଚଳ ଶୂନ୍ୟ * bias, ଅଚଳ ଶୂନ୍ୟ * scale, ଶୂନ୍ୟ * rhs_packed, size_t extra_bytes, ଅଚଳ ଶୂନ୍ୟ * params);

    rhs_repack_mode repack_mode = RHS_REPACK_PER_KERNEL;
};

ସଂରଚନା ggml_kleidiai_kernels {
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

ggml_kleidiai_kernels * ggml_kleidiai_select_kernels(cpu_feature cpu_features, ଅଚଳ ggml_tensor * tensor);
ggml_kleidiai_kernels * ggml_kleidiai_select_kernels_q4_0(cpu_feature features);
ggml_kleidiai_kernels * ggml_kleidiai_select_kernels_q4_k(cpu_feature features);
ggml_kleidiai_kernels * ggml_kleidiai_select_kernels_q8_0(cpu_feature features);
ggml_kleidiai_kernels * ggml_kleidiai_select_kernels_f32(cpu_feature features);
