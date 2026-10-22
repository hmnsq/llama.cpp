// SPDX-FileCopyrightText: Copyright 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
// SPDX-License-Identifier: MIT
//

#సూచన once

#అన్తర్భావ "ggml.h"

గణన cpu_feature {
    CPU_FEATURE_NONE    = 0,
    CPU_FEATURE_DOTPROD = 1,
    CPU_FEATURE_I8MM    = 2,
    CPU_FEATURE_SVE     = 4,
    CPU_FEATURE_SME     = 8,
    CPU_FEATURE_SME2    = 16,
    CPU_FEATURE_FP16    = 32,
    CPU_FEATURE_SVEI8MM = 64
};

అంతఃవరుస cpu_feature& క్రియాకారకం|=(cpu_feature& lhs, cpu_feature rhs) {
    lhs = static_cast<cpu_feature>(lhs | rhs);
    తిరిగి lhs;
}
అంతఃవరుస cpu_feature క్రియాకారకం|(cpu_feature lhs, cpu_feature rhs) {
    తిరిగి static_cast<cpu_feature>(static_cast<పూర్ణాంకం>(lhs) | static_cast<పూర్ణాంకం>(rhs));
}

నిర్మాణం kernel_info {
    size_t (*get_m_అడుగు)(శూన్య);
    size_t (*get_n_అడుగు)(శూన్య);
    size_t (*get_mr)(శూన్య);
    size_t (*get_nr)(శూన్య);
    size_t (*get_kr)(శూన్య);
    size_t (*get_sr)(శూన్య);

    size_t (*get_dst_స్థానభ్రంశం)(size_t m_idx, size_t n_idx, size_t stride);

    size_t (*get_lhs_స్థానభ్రంశం_ex)(size_t m_idx, size_t k, size_t bl);

    size_t (*get_rhs_packed_స్థానభ్రంశం_ex)(size_t n_idx, size_t k, size_t bl);

    శూన్య (*నిరంతరం_kernel_ex)(
        size_t m, size_t n, size_t k, size_t bl,
        స్థిరరాశి శూన్య* lhs_packed, స్థిరరాశి శూన్య* rhs_packed,
        శూన్య* dst, size_t dst_stride_అడ్డువరుస, size_t dst_stride_నిలువు,
        దశాంశం clamp_min, దశాంశం clamp_max);
};

నిర్మాణం lhs_packing_info {
    size_t (*get_స్థానభ్రంశం)(size_t m_idx, size_t lhs_stride);

    size_t (*get_packed_స్థానభ్రంశం_ex)(size_t m_idx, size_t k, size_t bl, size_t mr, size_t kr, size_t sr);

    size_t (*packed_కొలత_ex)(size_t m, size_t k, size_t bl, size_t mr, size_t kr, size_t sr);

    శూన్య (*pack_func_ex)(size_t m, size_t k, size_t bl, size_t mr, size_t kr, size_t sr,
        size_t m_idx_ప్రారంభం, స్థిరరాశి శూన్య * lhs, size_t lhs_stride, శూన్య * lhs_packed);
};

గణన rhs_repack_బహుళకం {
    RHS_REPACK_చొప్పున_KERNEL,
    RHS_REPACK_SINGLE_ONLY,
};

నిర్మాణం rhs_packing_info {
    size_t (*packed_కొలత_ex)(size_t n, size_t k, size_t nr, size_t kr, size_t bl);

    శూన్య (*pack_func_ex)(size_t అంకె_groups, size_t n, size_t k, size_t nr, size_t kr, size_t sr, size_t bl,
        size_t rhs_stride, స్థిరరాశి శూన్య * rhs, స్థిరరాశి శూన్య * bias, స్థిరరాశి శూన్య * scale, శూన్య * rhs_packed, size_t extra_బైట్లు, స్థిరరాశి శూన్య * params);

    rhs_repack_బహుళకం repack_బహుళకం = RHS_REPACK_చొప్పున_KERNEL;
};

నిర్మాణం ggml_kleidiai_kernels {
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

ggml_kleidiai_kernels * ggml_kleidiai_select_kernels(cpu_feature cpu_features, స్థిరరాశి ggml_tensor * tensor);
ggml_kleidiai_kernels * ggml_kleidiai_select_kernels_q4_0(cpu_feature features);
ggml_kleidiai_kernels * ggml_kleidiai_select_kernels_q4_k(cpu_feature features);
ggml_kleidiai_kernels * ggml_kleidiai_select_kernels_q8_0(cpu_feature features);
ggml_kleidiai_kernels * ggml_kleidiai_select_kernels_f32(cpu_feature features);
