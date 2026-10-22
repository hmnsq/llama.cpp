//
// MIT license
// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: MIT
//

//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

#ವ್ಯಾಖ್ಯಾನಿತವಾಗದಿದ್ದರೆ GGML_SYCL_MMVQ_HPP
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_SYCL_MMVQ_HPP

#ಅನ್ತರ್ಭಾವ "common.hpp"

// q5_K multi-column MMVQ shares weights across columns, pairs rows and fuses gate/up in the reorder
// layout: faster on Xe2 (BMG), so untested archs keep the per-column kernel
ಸಾಲೊಳಗೆ ತಾರ್ಕಿಕ ggml_sycl_q5_k_mmvq_reuse(ಪೂರ್ಣಾಂಕ device) {
    ಸ್ಥಿರಾಂಕ gpu_arch arch = ggml_sycl_info().devices[device].hw_info.arch;
    ಹಿಂತಿರುಗಿ arch == gpu_arch::intel_gpu_bmg_g21 || arch == gpu_arch::intel_gpu_bmg_g31;
}

ಶೂನ್ಯ ggml_sycl_op_mul_mat_vec_q(
    ggml_backend_sycl_context & ctx,
    ಸ್ಥಿರಾಂಕ ggml_tensor *src0, ಸ್ಥಿರಾಂಕ ggml_tensor *src1, ggml_tensor *dst,
    ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *src0_dd_i, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *src1_ddf_i, ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *src1_ddq_i,
    ದಶಮಾಂಶ *dst_dd_i, ಸ್ಥಿರಾಂಕ int64_t ಅಡ್ಡಸಾಲು_ಕೆಳಗಿನ, ಸ್ಥಿರಾಂಕ int64_t ಅಡ್ಡಸಾಲು_ಮೇಲಿನ,
    ಸ್ಥಿರಾಂಕ int64_t src1_ncols, ಸ್ಥಿರಾಂಕ int64_t src1_padded_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ,
    ಸ್ಥಿರಾಂಕ dpct::ಸರತಿ_ptr &stream);

// Requires standard (non-reorder) block layout for src0.
// Returns false if src0_type isn't handled; caller should fall back.
ತಾರ್ಕಿಕ ggml_sycl_mul_mat_vec_q_id(
    ಎಣಿಕೆ ggml_type     src0_type,
    ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *       vx_ತಳ,             // start of stacked expert weights
    ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *       vy,                  // pre-quantized src1 (Q8_1)
    ಸ್ಥಿರಾಂಕ int32_t *    ids_dev,             // device-side int32, length n_experts_used
    ದಶಮಾಂಶ *            dst_ತಳ,
    ಪೂರ್ಣಾಂಕ                ncols,
    ಪೂರ್ಣಾಂಕ                nrows,
    ಪೂರ್ಣಾಂಕ                n_experts_ಬಳಸಿದ,
    size_t             expert_ತೂಕ_stride, // bytes between experts in vx_base
    size_t             dst_ಅಡ್ಡಸಾಲು_stride,       // bytes between dst rows
    size_t             src1_ಅಡ್ಡಸಾಲು_stride,      // 0 = shared src1, else per-expert stride in bytes
    dpct::ಸರತಿ_ptr    stream);

// Reorder (SoA) variant of the fused MoE expert GEMV.
// vx_base: each expert slice (stride expert_weight_stride == src0->nb[2]) is a self-contained reorder/SoA layout.
// vy: src1 quantized with quantize_and_reorder_q8_1_soa (per-row SoA). Returns false if src0_type isn't handled.
ತಾರ್ಕಿಕ ggml_sycl_mul_mat_vec_q_id_reorder(
    ಎಣಿಕೆ ggml_type     src0_type,
    ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *       vx_ತಳ,
    ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *       vy,
    ಸ್ಥಿರಾಂಕ int32_t *    ids_dev,
    ದಶಮಾಂಶ *            dst_ತಳ,
    ಪೂರ್ಣಾಂಕ                ncols,
    ಪೂರ್ಣಾಂಕ                nrows,
    ಪೂರ್ಣಾಂಕ                n_experts_ಬಳಸಿದ,
    size_t             expert_ತೂಕ_stride,
    size_t             dst_ಅಡ್ಡಸಾಲು_stride,
    size_t             src1_ಅಡ್ಡಸಾಲು_stride,
    dpct::ಸರತಿ_ptr    stream);

// Fused dense-FFN GEMV: writes glu(gate . y, up . y) instead of the two mat-vec results.
// vx / vgate must share shape, stride and reorder layout. Returns false if unhandled.
ತಾರ್ಕಿಕ ggml_sycl_mul_mat_vec_q_glu_reorder(
    ಎಣಿಕೆ ggml_type     src0_type,
    ಎಣಿಕೆ ggml_glu_op   glu_op,
    ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *       vx,
    ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *       vgate,
    ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *       vy,
    ದಶಮಾಂಶ *            dst,
    ಪೂರ್ಣಾಂಕ                ncols,                // K, shared by both weights
    ಪೂರ್ಣಾಂಕ                nrows,                // output rows, i.e. weight ne[1]
    ಪೂರ್ಣಾಂಕ                ncols_dst,            // activation columns, 1..MMVQ_MAX_BATCH_SIZE
    ಪೂರ್ಣಾಂಕ                stride_ಲಂಬಸಾಲು_y_ಬೈಟುಗಳು,   // bytes between activation columns in vy
    ಪೂರ್ಣಾಂಕ                stride_ಲಂಬಸಾಲು_dst,       // floats between output columns in dst
    dpct::ಸರತಿ_ptr    stream);


// Fused dense-FFN GEMV + GLU over the standard (non-reorder) layout; the gate and up
// weights may carry different block types (q5_K / iq4_xs, mixed included).
// vy: src1 quantized with plain quantize_q8_1 (padded rows). stride_col_y is in
// block_q8_1 units. Returns false if the pair or batch is unhandled; caller falls back.
ತಾರ್ಕಿಕ ggml_sycl_mul_mat_vec_q_glu_plain(
    ಎಣಿಕೆ ggml_type     gate_type,
    ಎಣಿಕೆ ggml_type     ಮೇಲೆ_type,
    ಎಣಿಕೆ ggml_glu_op   glu_op,
    ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *       vgate,
    ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *       vup,
    ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *       vy,
    ದಶಮಾಂಶ *            dst,
    ಪೂರ್ಣಾಂಕ                ncols,                // K, shared by both weights
    ಪೂರ್ಣಾಂಕ                nrows,                // output rows, i.e. weight ne[1]
    ಪೂರ್ಣಾಂಕ                ncols_dst,            // activation columns, 1..MMVQ_MAX_BATCH_SIZE
    ಪೂರ್ಣಾಂಕ                stride_ಲಂಬಸಾಲು_y,         // block_q8_1 units between activation columns
    ಪೂರ್ಣಾಂಕ                stride_ಲಂಬಸಾಲು_dst,       // floats between output columns in dst
    dpct::ಸರತಿ_ptr    stream);

#ಆದರೆ_ಅಂತ್ಯ // GGML_SYCL_MMVQ_HPP
