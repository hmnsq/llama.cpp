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

#నిర్వచితం_కాకపోతే GGML_SYCL_MMVQ_HPP
#నిర్వచించు GGML_SYCL_MMVQ_HPP

#అన్తర్భావ "common.hpp"

// q5_K multi-column MMVQ shares weights across columns, pairs rows and fuses gate/up in the reorder
// layout: faster on Xe2 (BMG), so untested archs keep the per-column kernel
అంతఃవరుస తార్కిక ggml_sycl_q5_k_mmvq_reuse(పూర్ణాంకం device) {
    స్థిరరాశి gpu_arch arch = ggml_sycl_info().devices[device].hw_info.arch;
    తిరిగి arch == gpu_arch::intel_gpu_bmg_g21 || arch == gpu_arch::intel_gpu_bmg_g31;
}

శూన్య ggml_sycl_op_mul_mat_vec_q(
    ggml_backend_sycl_context & ctx,
    స్థిరరాశి ggml_tensor *src0, స్థిరరాశి ggml_tensor *src1, ggml_tensor *dst,
    స్థిరరాశి అక్షరం *src0_dd_i, స్థిరరాశి దశాంశం *src1_ddf_i, స్థిరరాశి అక్షరం *src1_ddq_i,
    దశాంశం *dst_dd_i, స్థిరరాశి int64_t అడ్డువరుస_దిగువ, స్థిరరాశి int64_t అడ్డువరుస_ఎగువ,
    స్థిరరాశి int64_t src1_ncols, స్థిరరాశి int64_t src1_padded_అడ్డువరుస_కొలత,
    స్థిరరాశి dpct::వరుస_ptr &stream);

// Requires standard (non-reorder) block layout for src0.
// Returns false if src0_type isn't handled; caller should fall back.
తార్కిక ggml_sycl_mul_mat_vec_q_id(
    గణన ggml_type     src0_type,
    స్థిరరాశి శూన్య *       vx_పునాది,             // start of stacked expert weights
    స్థిరరాశి శూన్య *       vy,                  // pre-quantized src1 (Q8_1)
    స్థిరరాశి int32_t *    ids_dev,             // device-side int32, length n_experts_used
    దశాంశం *            dst_పునాది,
    పూర్ణాంకం                ncols,
    పూర్ణాంకం                nrows,
    పూర్ణాంకం                n_experts_వాడిన,
    size_t             expert_బరువు_stride, // bytes between experts in vx_base
    size_t             dst_అడ్డువరుస_stride,       // bytes between dst rows
    size_t             src1_అడ్డువరుస_stride,      // 0 = shared src1, else per-expert stride in bytes
    dpct::వరుస_ptr    stream);

// Reorder (SoA) variant of the fused MoE expert GEMV.
// vx_base: each expert slice (stride expert_weight_stride == src0->nb[2]) is a self-contained reorder/SoA layout.
// vy: src1 quantized with quantize_and_reorder_q8_1_soa (per-row SoA). Returns false if src0_type isn't handled.
తార్కిక ggml_sycl_mul_mat_vec_q_id_reorder(
    గణన ggml_type     src0_type,
    స్థిరరాశి శూన్య *       vx_పునాది,
    స్థిరరాశి శూన్య *       vy,
    స్థిరరాశి int32_t *    ids_dev,
    దశాంశం *            dst_పునాది,
    పూర్ణాంకం                ncols,
    పూర్ణాంకం                nrows,
    పూర్ణాంకం                n_experts_వాడిన,
    size_t             expert_బరువు_stride,
    size_t             dst_అడ్డువరుస_stride,
    size_t             src1_అడ్డువరుస_stride,
    dpct::వరుస_ptr    stream);

// Fused dense-FFN GEMV: writes glu(gate . y, up . y) instead of the two mat-vec results.
// vx / vgate must share shape, stride and reorder layout. Returns false if unhandled.
తార్కిక ggml_sycl_mul_mat_vec_q_glu_reorder(
    గణన ggml_type     src0_type,
    గణన ggml_glu_op   glu_op,
    స్థిరరాశి శూన్య *       vx,
    స్థిరరాశి శూన్య *       vgate,
    స్థిరరాశి శూన్య *       vy,
    దశాంశం *            dst,
    పూర్ణాంకం                ncols,                // K, shared by both weights
    పూర్ణాంకం                nrows,                // output rows, i.e. weight ne[1]
    పూర్ణాంకం                ncols_dst,            // activation columns, 1..MMVQ_MAX_BATCH_SIZE
    పూర్ణాంకం                stride_నిలువు_y_బైట్లు,   // bytes between activation columns in vy
    పూర్ణాంకం                stride_నిలువు_dst,       // floats between output columns in dst
    dpct::వరుస_ptr    stream);


// Fused dense-FFN GEMV + GLU over the standard (non-reorder) layout; the gate and up
// weights may carry different block types (q5_K / iq4_xs, mixed included).
// vy: src1 quantized with plain quantize_q8_1 (padded rows). stride_col_y is in
// block_q8_1 units. Returns false if the pair or batch is unhandled; caller falls back.
తార్కిక ggml_sycl_mul_mat_vec_q_glu_plain(
    గణన ggml_type     gate_type,
    గణన ggml_type     పైకి_type,
    గణన ggml_glu_op   glu_op,
    స్థిరరాశి శూన్య *       vgate,
    స్థిరరాశి శూన్య *       vup,
    స్థిరరాశి శూన్య *       vy,
    దశాంశం *            dst,
    పూర్ణాంకం                ncols,                // K, shared by both weights
    పూర్ణాంకం                nrows,                // output rows, i.e. weight ne[1]
    పూర్ణాంకం                ncols_dst,            // activation columns, 1..MMVQ_MAX_BATCH_SIZE
    పూర్ణాంకం                stride_నిలువు_y,         // block_q8_1 units between activation columns
    పూర్ణాంకం                stride_నిలువు_dst,       // floats between output columns in dst
    dpct::వరుస_ptr    stream);

#అయితే_ముగింపు // GGML_SYCL_MMVQ_HPP
