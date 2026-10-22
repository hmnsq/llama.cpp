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

#অন্তর্ভাব "common.hpp"

// q5_K multi-column MMVQ shares weights across columns, pairs rows and fuses gate/up in the reorder
// layout: faster on Xe2 (BMG), so untested archs keep the per-column kernel
অন্তঃরেখ যৌক্তিক ggml_sycl_q5_k_mmvq_reuse(পূর্ণসংখ্যা device) {
    ধ্রুবক gpu_arch arch = ggml_sycl_info().devices[device].hw_info.arch;
    ফেরত arch == gpu_arch::intel_gpu_bmg_g21 || arch == gpu_arch::intel_gpu_bmg_g31;
}

শূন্য ggml_sycl_op_mul_mat_vec_q(
    ggml_backend_sycl_context & ctx,
    ধ্রুবক ggml_tensor *src0, ধ্রুবক ggml_tensor *src1, ggml_tensor *dst,
    ধ্রুবক অক্ষর *src0_dd_i, ধ্রুবক দশমিক *src1_ddf_i, ধ্রুবক অক্ষর *src1_ddq_i,
    দশমিক *dst_dd_i, ধ্রুবক int64_t row_নিচের, ধ্রুবক int64_t row_উপরের,
    ধ্রুবক int64_t src1_ncols, ধ্রুবক int64_t src1_padded_row_আকার,
    ধ্রুবক dpct::queue_ptr &stream);

// Requires standard (non-reorder) block layout for src0.
// Returns false if src0_type isn't handled; caller should fall back.
যৌক্তিক ggml_sycl_mul_mat_vec_q_id(
    গণনা ggml_type     src0_type,
    ধ্রুবক শূন্য *       vx_base,             // start of stacked expert weights
    ধ্রুবক শূন্য *       vy,                  // pre-quantized src1 (Q8_1)
    ধ্রুবক int32_t *    ids_dev,             // device-side int32, length n_experts_used
    দশমিক *            dst_base,
    পূর্ণসংখ্যা                ncols,
    পূর্ণসংখ্যা                nrows,
    পূর্ণসংখ্যা                n_experts_used,
    size_t             expert_weight_stride, // bytes between experts in vx_base
    size_t             dst_row_stride,       // bytes between dst rows
    size_t             src1_row_stride,      // 0 = shared src1, else per-expert stride in bytes
    dpct::queue_ptr    stream);

// Reorder (SoA) variant of the fused MoE expert GEMV.
// vx_base: each expert slice (stride expert_weight_stride == src0->nb[2]) is a self-contained reorder/SoA layout.
// vy: src1 quantized with quantize_and_reorder_q8_1_soa (per-row SoA). Returns false if src0_type isn't handled.
যৌক্তিক ggml_sycl_mul_mat_vec_q_id_reorder(
    গণনা ggml_type     src0_type,
    ধ্রুবক শূন্য *       vx_base,
    ধ্রুবক শূন্য *       vy,
    ধ্রুবক int32_t *    ids_dev,
    দশমিক *            dst_base,
    পূর্ণসংখ্যা                ncols,
    পূর্ণসংখ্যা                nrows,
    পূর্ণসংখ্যা                n_experts_used,
    size_t             expert_weight_stride,
    size_t             dst_row_stride,
    size_t             src1_row_stride,
    dpct::queue_ptr    stream);

// Fused dense-FFN GEMV: writes glu(gate . y, up . y) instead of the two mat-vec results.
// vx / vgate must share shape, stride and reorder layout. Returns false if unhandled.
যৌক্তিক ggml_sycl_mul_mat_vec_q_glu_reorder(
    গণনা ggml_type     src0_type,
    গণনা ggml_glu_op   glu_op,
    ধ্রুবক শূন্য *       vx,
    ধ্রুবক শূন্য *       vgate,
    ধ্রুবক শূন্য *       vy,
    দশমিক *            dst,
    পূর্ণসংখ্যা                ncols,                // K, shared by both weights
    পূর্ণসংখ্যা                nrows,                // output rows, i.e. weight ne[1]
    পূর্ণসংখ্যা                ncols_dst,            // activation columns, 1..MMVQ_MAX_BATCH_SIZE
    পূর্ণসংখ্যা                stride_col_y_bytes,   // bytes between activation columns in vy
    পূর্ণসংখ্যা                stride_col_dst,       // floats between output columns in dst
    dpct::queue_ptr    stream);


// Fused dense-FFN GEMV + GLU over the standard (non-reorder) layout; the gate and up
// weights may carry different block types (q5_K / iq4_xs, mixed included).
// vy: src1 quantized with plain quantize_q8_1 (padded rows). stride_col_y is in
// block_q8_1 units. Returns false if the pair or batch is unhandled; caller falls back.
যৌক্তিক ggml_sycl_mul_mat_vec_q_glu_plain(
    গণনা ggml_type     gate_type,
    গণনা ggml_type     up_type,
    গণনা ggml_glu_op   glu_op,
    ধ্রুবক শূন্য *       vgate,
    ধ্রুবক শূন্য *       vup,
    ধ্রুবক শূন্য *       vy,
    দশমিক *            dst,
    পূর্ণসংখ্যা                ncols,                // K, shared by both weights
    পূর্ণসংখ্যা                nrows,                // output rows, i.e. weight ne[1]
    পূর্ণসংখ্যা                ncols_dst,            // activation columns, 1..MMVQ_MAX_BATCH_SIZE
    পূর্ণসংখ্যা                stride_col_y,         // block_q8_1 units between activation columns
    পূর্ণসংখ্যা                stride_col_dst,       // floats between output columns in dst
    dpct::queue_ptr    stream);

#endif // GGML_SYCL_MMVQ_HPP
