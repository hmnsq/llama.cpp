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

#ଅନ୍ତର୍ଭାଵ "common.hpp"

// q5_K multi-column MMVQ shares weights across columns, pairs rows and fuses gate/up in the reorder
// layout: faster on Xe2 (BMG), so untested archs keep the per-column kernel
ଧାଡ଼ିଭିତରେ ତାର୍କିକ ggml_sycl_q5_k_mmvq_reuse(ପୂର୍ଣ୍ଣସଂଖ୍ୟା device) {
    ଅଚଳ gpu_arch arch = ggml_sycl_info().devices[device].hw_info.arch;
    ଫେରନ୍ତୁ arch == gpu_arch::intel_gpu_bmg_g21 || arch == gpu_arch::intel_gpu_bmg_g31;
}

ଶୂନ୍ୟ ggml_sycl_op_mul_mat_vec_q(
    ggml_backend_sycl_context & ctx,
    ଅଚଳ ggml_tensor *src0, ଅଚଳ ggml_tensor *src1, ggml_tensor *dst,
    ଅଚଳ ଅକ୍ଷର *src0_dd_i, ଅଚଳ ଦଶମିକ *src1_ddf_i, ଅଚଳ ଅକ୍ଷର *src1_ddq_i,
    ଦଶମିକ *dst_dd_i, ଅଚଳ int64_t row_ତଳ, ଅଚଳ int64_t row_ଉପର,
    ଅଚଳ int64_t src1_ncols, ଅଚଳ int64_t src1_padded_row_size,
    ଅଚଳ dpct::queue_ptr &stream);

// Requires standard (non-reorder) block layout for src0.
// Returns false if src0_type isn't handled; caller should fall back.
ତାର୍କିକ ggml_sycl_mul_mat_vec_q_id(
    ଗଣନା ggml_type     src0_type,
    ଅଚଳ ଶୂନ୍ୟ *       vx_base,             // start of stacked expert weights
    ଅଚଳ ଶୂନ୍ୟ *       vy,                  // pre-quantized src1 (Q8_1)
    ଅଚଳ int32_t *    ids_dev,             // device-side int32, length n_experts_used
    ଦଶମିକ *            dst_base,
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା                ncols,
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା                nrows,
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା                n_experts_used,
    size_t             expert_weight_stride, // bytes between experts in vx_base
    size_t             dst_row_stride,       // bytes between dst rows
    size_t             src1_row_stride,      // 0 = shared src1, else per-expert stride in bytes
    dpct::queue_ptr    stream);

// Reorder (SoA) variant of the fused MoE expert GEMV.
// vx_base: each expert slice (stride expert_weight_stride == src0->nb[2]) is a self-contained reorder/SoA layout.
// vy: src1 quantized with quantize_and_reorder_q8_1_soa (per-row SoA). Returns false if src0_type isn't handled.
ତାର୍କିକ ggml_sycl_mul_mat_vec_q_id_reorder(
    ଗଣନା ggml_type     src0_type,
    ଅଚଳ ଶୂନ୍ୟ *       vx_base,
    ଅଚଳ ଶୂନ୍ୟ *       vy,
    ଅଚଳ int32_t *    ids_dev,
    ଦଶମିକ *            dst_base,
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା                ncols,
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା                nrows,
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା                n_experts_used,
    size_t             expert_weight_stride,
    size_t             dst_row_stride,
    size_t             src1_row_stride,
    dpct::queue_ptr    stream);

// Fused dense-FFN GEMV: writes glu(gate . y, up . y) instead of the two mat-vec results.
// vx / vgate must share shape, stride and reorder layout. Returns false if unhandled.
ତାର୍କିକ ggml_sycl_mul_mat_vec_q_glu_reorder(
    ଗଣନା ggml_type     src0_type,
    ଗଣନା ggml_glu_op   glu_op,
    ଅଚଳ ଶୂନ୍ୟ *       vx,
    ଅଚଳ ଶୂନ୍ୟ *       vgate,
    ଅଚଳ ଶୂନ୍ୟ *       vy,
    ଦଶମିକ *            dst,
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା                ncols,                // K, shared by both weights
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା                nrows,                // output rows, i.e. weight ne[1]
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା                ncols_dst,            // activation columns, 1..MMVQ_MAX_BATCH_SIZE
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା                stride_col_y_bytes,   // bytes between activation columns in vy
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା                stride_col_dst,       // floats between output columns in dst
    dpct::queue_ptr    stream);


// Fused dense-FFN GEMV + GLU over the standard (non-reorder) layout; the gate and up
// weights may carry different block types (q5_K / iq4_xs, mixed included).
// vy: src1 quantized with plain quantize_q8_1 (padded rows). stride_col_y is in
// block_q8_1 units. Returns false if the pair or batch is unhandled; caller falls back.
ତାର୍କିକ ggml_sycl_mul_mat_vec_q_glu_plain(
    ଗଣନା ggml_type     gate_type,
    ଗଣନା ggml_type     up_type,
    ଗଣନା ggml_glu_op   glu_op,
    ଅଚଳ ଶୂନ୍ୟ *       vgate,
    ଅଚଳ ଶୂନ୍ୟ *       vup,
    ଅଚଳ ଶୂନ୍ୟ *       vy,
    ଦଶମିକ *            dst,
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା                ncols,                // K, shared by both weights
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା                nrows,                // output rows, i.e. weight ne[1]
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା                ncols_dst,            // activation columns, 1..MMVQ_MAX_BATCH_SIZE
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା                stride_col_y,         // block_q8_1 units between activation columns
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା                stride_col_dst,       // floats between output columns in dst
    dpct::queue_ptr    stream);

#endif // GGML_SYCL_MMVQ_HPP
