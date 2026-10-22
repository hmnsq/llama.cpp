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

#അന്തര്ഭാവ "common.hpp"

// q5_K multi-column MMVQ shares weights across columns, pairs rows and fuses gate/up in the reorder
// layout: faster on Xe2 (BMG), so untested archs keep the per-column kernel
വരിയിൽ തർക്കം ggml_sycl_q5_k_mmvq_reuse(പൂർണ്ണസംഖ്യ device) {
    സ്ഥിരം gpu_arch arch = ggml_sycl_info().devices[device].hw_info.arch;
    തിരികെ arch == gpu_arch::intel_gpu_bmg_g21 || arch == gpu_arch::intel_gpu_bmg_g31;
}

ശൂന്യം ggml_sycl_op_mul_mat_vec_q(
    ggml_backend_sycl_context & ctx,
    സ്ഥിരം ggml_tensor *src0, സ്ഥിരം ggml_tensor *src1, ggml_tensor *dst,
    സ്ഥിരം അക്ഷരം *src0_dd_i, സ്ഥിരം ദശാംശം *src1_ddf_i, സ്ഥിരം അക്ഷരം *src1_ddq_i,
    ദശാംശം *dst_dd_i, സ്ഥിരം int64_t row_താഴെയുള്ള, സ്ഥിരം int64_t row_മുകളിലെ,
    സ്ഥിരം int64_t src1_ncols, സ്ഥിരം int64_t src1_padded_row_size,
    സ്ഥിരം dpct::queue_ptr &stream);

// Requires standard (non-reorder) block layout for src0.
// Returns false if src0_type isn't handled; caller should fall back.
തർക്കം ggml_sycl_mul_mat_vec_q_id(
    എണ്ണൽ ggml_type     src0_type,
    സ്ഥിരം ശൂന്യം *       vx_base,             // start of stacked expert weights
    സ്ഥിരം ശൂന്യം *       vy,                  // pre-quantized src1 (Q8_1)
    സ്ഥിരം int32_t *    ids_dev,             // device-side int32, length n_experts_used
    ദശാംശം *            dst_base,
    പൂർണ്ണസംഖ്യ                ncols,
    പൂർണ്ണസംഖ്യ                nrows,
    പൂർണ്ണസംഖ്യ                n_experts_used,
    size_t             expert_weight_stride, // bytes between experts in vx_base
    size_t             dst_row_stride,       // bytes between dst rows
    size_t             src1_row_stride,      // 0 = shared src1, else per-expert stride in bytes
    dpct::queue_ptr    stream);

// Reorder (SoA) variant of the fused MoE expert GEMV.
// vx_base: each expert slice (stride expert_weight_stride == src0->nb[2]) is a self-contained reorder/SoA layout.
// vy: src1 quantized with quantize_and_reorder_q8_1_soa (per-row SoA). Returns false if src0_type isn't handled.
തർക്കം ggml_sycl_mul_mat_vec_q_id_reorder(
    എണ്ണൽ ggml_type     src0_type,
    സ്ഥിരം ശൂന്യം *       vx_base,
    സ്ഥിരം ശൂന്യം *       vy,
    സ്ഥിരം int32_t *    ids_dev,
    ദശാംശം *            dst_base,
    പൂർണ്ണസംഖ്യ                ncols,
    പൂർണ്ണസംഖ്യ                nrows,
    പൂർണ്ണസംഖ്യ                n_experts_used,
    size_t             expert_weight_stride,
    size_t             dst_row_stride,
    size_t             src1_row_stride,
    dpct::queue_ptr    stream);

// Fused dense-FFN GEMV: writes glu(gate . y, up . y) instead of the two mat-vec results.
// vx / vgate must share shape, stride and reorder layout. Returns false if unhandled.
തർക്കം ggml_sycl_mul_mat_vec_q_glu_reorder(
    എണ്ണൽ ggml_type     src0_type,
    എണ്ണൽ ggml_glu_op   glu_op,
    സ്ഥിരം ശൂന്യം *       vx,
    സ്ഥിരം ശൂന്യം *       vgate,
    സ്ഥിരം ശൂന്യം *       vy,
    ദശാംശം *            dst,
    പൂർണ്ണസംഖ്യ                ncols,                // K, shared by both weights
    പൂർണ്ണസംഖ്യ                nrows,                // output rows, i.e. weight ne[1]
    പൂർണ്ണസംഖ്യ                ncols_dst,            // activation columns, 1..MMVQ_MAX_BATCH_SIZE
    പൂർണ്ണസംഖ്യ                stride_col_y_bytes,   // bytes between activation columns in vy
    പൂർണ്ണസംഖ്യ                stride_col_dst,       // floats between output columns in dst
    dpct::queue_ptr    stream);


// Fused dense-FFN GEMV + GLU over the standard (non-reorder) layout; the gate and up
// weights may carry different block types (q5_K / iq4_xs, mixed included).
// vy: src1 quantized with plain quantize_q8_1 (padded rows). stride_col_y is in
// block_q8_1 units. Returns false if the pair or batch is unhandled; caller falls back.
തർക്കം ggml_sycl_mul_mat_vec_q_glu_plain(
    എണ്ണൽ ggml_type     gate_type,
    എണ്ണൽ ggml_type     up_type,
    എണ്ണൽ ggml_glu_op   glu_op,
    സ്ഥിരം ശൂന്യം *       vgate,
    സ്ഥിരം ശൂന്യം *       vup,
    സ്ഥിരം ശൂന്യം *       vy,
    ദശാംശം *            dst,
    പൂർണ്ണസംഖ്യ                ncols,                // K, shared by both weights
    പൂർണ്ണസംഖ്യ                nrows,                // output rows, i.e. weight ne[1]
    പൂർണ്ണസംഖ്യ                ncols_dst,            // activation columns, 1..MMVQ_MAX_BATCH_SIZE
    പൂർണ്ണസംഖ്യ                stride_col_y,         // block_q8_1 units between activation columns
    പൂർണ്ണസംഖ്യ                stride_col_dst,       // floats between output columns in dst
    dpct::queue_ptr    stream);

#endif // GGML_SYCL_MMVQ_HPP
