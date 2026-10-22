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

#ifndef GGML_SYCL_MMVQ_HPP
#define GGML_SYCL_MMVQ_HPP

#ಅನ್ತರ್ಭಾವ "common.hpp"

// q5_K multi-column MMVQ shares weights across columns, pairs rows and fuses gate/up in the reorder
// layout: faster on Xe2 (BMG), so untested archs keep the per-column kernel
ಅನ್ತಃಸ್ಥ ತಾರ್ಕಿಕ ggml_sycl_q5_k_mmvq_reuse(ಪೂರ್ಣಾಙ್ಕ device) {
    ಧ್ರುವ gpu_arch arch = ggml_sycl_info().devices[device].hw_info.arch;
    ಪ್ರತ್ಯಾಗಮ arch == gpu_arch::intel_gpu_bmg_g21 || arch == gpu_arch::intel_gpu_bmg_g31;
}

ಶೂನ್ಯ ggml_sycl_op_mul_mat_vec_q(
    ggml_backend_sycl_context & ctx,
    ಧ್ರುವ ggml_tensor *src0, ಧ್ರುವ ggml_tensor *src1, ggml_tensor *dst,
    ಧ್ರುವ ವರ್ಣ *src0_dd_i, ಧ್ರುವ ಭಿನ್ನ *src1_ddf_i, ಧ್ರುವ ವರ್ಣ *src1_ddq_i,
    ಭಿನ್ನ *dst_dd_i, ಧ್ರುವ int64_t row_ಕೆಳಗಿನ, ಧ್ರುವ int64_t row_ಮೇಲಿನ,
    ಧ್ರುವ int64_t src1_ncols, ಧ್ರುವ int64_t src1_padded_row_size,
    ಧ್ರುವ dpct::queue_ptr &stream);

// Requires standard (non-reorder) block layout for src0.
// Returns false if src0_type isn't handled; caller should fall back.
ತಾರ್ಕಿಕ ggml_sycl_mul_mat_vec_q_id(
    ಗಣನಾ ggml_type     src0_type,
    ಧ್ರುವ ಶೂನ್ಯ *       vx_base,             // start of stacked expert weights
    ಧ್ರುವ ಶೂನ್ಯ *       vy,                  // pre-quantized src1 (Q8_1)
    ಧ್ರುವ int32_t *    ids_dev,             // device-side int32, length n_experts_used
    ಭಿನ್ನ *            dst_base,
    ಪೂರ್ಣಾಙ್ಕ                ncols,
    ಪೂರ್ಣಾಙ್ಕ                nrows,
    ಪೂರ್ಣಾಙ್ಕ                n_experts_used,
    size_t             expert_weight_stride, // bytes between experts in vx_base
    size_t             dst_row_stride,       // bytes between dst rows
    size_t             src1_row_stride,      // 0 = shared src1, else per-expert stride in bytes
    dpct::queue_ptr    stream);

// Reorder (SoA) variant of the fused MoE expert GEMV.
// vx_base: each expert slice (stride expert_weight_stride == src0->nb[2]) is a self-contained reorder/SoA layout.
// vy: src1 quantized with quantize_and_reorder_q8_1_soa (per-row SoA). Returns false if src0_type isn't handled.
ತಾರ್ಕಿಕ ggml_sycl_mul_mat_vec_q_id_reorder(
    ಗಣನಾ ggml_type     src0_type,
    ಧ್ರುವ ಶೂನ್ಯ *       vx_base,
    ಧ್ರುವ ಶೂನ್ಯ *       vy,
    ಧ್ರುವ int32_t *    ids_dev,
    ಭಿನ್ನ *            dst_base,
    ಪೂರ್ಣಾಙ್ಕ                ncols,
    ಪೂರ್ಣಾಙ್ಕ                nrows,
    ಪೂರ್ಣಾಙ್ಕ                n_experts_used,
    size_t             expert_weight_stride,
    size_t             dst_row_stride,
    size_t             src1_row_stride,
    dpct::queue_ptr    stream);

// Fused dense-FFN GEMV: writes glu(gate . y, up . y) instead of the two mat-vec results.
// vx / vgate must share shape, stride and reorder layout. Returns false if unhandled.
ತಾರ್ಕಿಕ ggml_sycl_mul_mat_vec_q_glu_reorder(
    ಗಣನಾ ggml_type     src0_type,
    ಗಣನಾ ggml_glu_op   glu_op,
    ಧ್ರುವ ಶೂನ್ಯ *       vx,
    ಧ್ರುವ ಶೂನ್ಯ *       vgate,
    ಧ್ರುವ ಶೂನ್ಯ *       vy,
    ಭಿನ್ನ *            dst,
    ಪೂರ್ಣಾಙ್ಕ                ncols,                // K, shared by both weights
    ಪೂರ್ಣಾಙ್ಕ                nrows,                // output rows, i.e. weight ne[1]
    ಪೂರ್ಣಾಙ್ಕ                ncols_dst,            // activation columns, 1..MMVQ_MAX_BATCH_SIZE
    ಪೂರ್ಣಾಙ್ಕ                stride_col_y_bytes,   // bytes between activation columns in vy
    ಪೂರ್ಣಾಙ್ಕ                stride_col_dst,       // floats between output columns in dst
    dpct::queue_ptr    stream);


// Fused dense-FFN GEMV + GLU over the standard (non-reorder) layout; the gate and up
// weights may carry different block types (q5_K / iq4_xs, mixed included).
// vy: src1 quantized with plain quantize_q8_1 (padded rows). stride_col_y is in
// block_q8_1 units. Returns false if the pair or batch is unhandled; caller falls back.
ತಾರ್ಕಿಕ ggml_sycl_mul_mat_vec_q_glu_plain(
    ಗಣನಾ ggml_type     gate_type,
    ಗಣನಾ ggml_type     up_type,
    ಗಣನಾ ggml_glu_op   glu_op,
    ಧ್ರುವ ಶೂನ್ಯ *       vgate,
    ಧ್ರುವ ಶೂನ್ಯ *       vup,
    ಧ್ರುವ ಶೂನ್ಯ *       vy,
    ಭಿನ್ನ *            dst,
    ಪೂರ್ಣಾಙ್ಕ                ncols,                // K, shared by both weights
    ಪೂರ್ಣಾಙ್ಕ                nrows,                // output rows, i.e. weight ne[1]
    ಪೂರ್ಣಾಙ್ಕ                ncols_dst,            // activation columns, 1..MMVQ_MAX_BATCH_SIZE
    ಪೂರ್ಣಾಙ್ಕ                stride_col_y,         // block_q8_1 units between activation columns
    ಪೂರ್ಣಾಙ್ಕ                stride_col_dst,       // floats between output columns in dst
    dpct::queue_ptr    stream);

#endif // GGML_SYCL_MMVQ_HPP
