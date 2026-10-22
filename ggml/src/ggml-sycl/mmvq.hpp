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

#ਜੇ_ਅਪਰਿਭਾਸ਼ਿਤ GGML_SYCL_MMVQ_HPP
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_SYCL_MMVQ_HPP

#ਅਨ੍ਤਰ੍ਭਾਵ "common.hpp"

// q5_K multi-column MMVQ shares weights across columns, pairs rows and fuses gate/up in the reorder
// layout: faster on Xe2 (BMG), so untested archs keep the per-column kernel
ਸਤਰਅੰਦਰ ਤਾਰਕਿਕ ggml_sycl_q5_k_mmvq_reuse(ਪੂਰਨ device) {
    ਅਚਲ gpu_arch arch = ggml_sycl_info().devices[device].hw_info.arch;
    ਵਾਪਸ arch == gpu_arch::intel_gpu_bmg_g21 || arch == gpu_arch::intel_gpu_bmg_g31;
}

ਖਾਲੀ ggml_sycl_op_mul_mat_vec_q(
    ggml_backend_sycl_context & ctx,
    ਅਚਲ ggml_tensor *src0, ਅਚਲ ggml_tensor *src1, ggml_tensor *dst,
    ਅਚਲ ਅੱਖਰ *src0_dd_i, ਅਚਲ ਦਸ਼ਮਲਵ *src1_ddf_i, ਅਚਲ ਅੱਖਰ *src1_ddq_i,
    ਦਸ਼ਮਲਵ *dst_dd_i, ਅਚਲ int64_t ਪੰਕਤੀ_ਨੀਵਾਂ, ਅਚਲ int64_t ਪੰਕਤੀ_ਉੱਚਾ,
    ਅਚਲ int64_t src1_ncols, ਅਚਲ int64_t src1_padded_ਪੰਕਤੀ_ਮਾਪ,
    ਅਚਲ dpct::ਕਤਾਰ_ptr &stream);

// Requires standard (non-reorder) block layout for src0.
// Returns false if src0_type isn't handled; caller should fall back.
ਤਾਰਕਿਕ ggml_sycl_mul_mat_vec_q_id(
    ਗਿਣਤੀ ggml_type     src0_type,
    ਅਚਲ ਖਾਲੀ *       vx_ਅਧਾਰ,             // start of stacked expert weights
    ਅਚਲ ਖਾਲੀ *       vy,                  // pre-quantized src1 (Q8_1)
    ਅਚਲ int32_t *    ids_dev,             // device-side int32, length n_experts_used
    ਦਸ਼ਮਲਵ *            dst_ਅਧਾਰ,
    ਪੂਰਨ                ncols,
    ਪੂਰਨ                nrows,
    ਪੂਰਨ                n_experts_ਵਰਤਿਆ,
    size_t             expert_ਭਾਰ_stride, // bytes between experts in vx_base
    size_t             dst_ਪੰਕਤੀ_stride,       // bytes between dst rows
    size_t             src1_ਪੰਕਤੀ_stride,      // 0 = shared src1, else per-expert stride in bytes
    dpct::ਕਤਾਰ_ptr    stream);

// Reorder (SoA) variant of the fused MoE expert GEMV.
// vx_base: each expert slice (stride expert_weight_stride == src0->nb[2]) is a self-contained reorder/SoA layout.
// vy: src1 quantized with quantize_and_reorder_q8_1_soa (per-row SoA). Returns false if src0_type isn't handled.
ਤਾਰਕਿਕ ggml_sycl_mul_mat_vec_q_id_reorder(
    ਗਿਣਤੀ ggml_type     src0_type,
    ਅਚਲ ਖਾਲੀ *       vx_ਅਧਾਰ,
    ਅਚਲ ਖਾਲੀ *       vy,
    ਅਚਲ int32_t *    ids_dev,
    ਦਸ਼ਮਲਵ *            dst_ਅਧਾਰ,
    ਪੂਰਨ                ncols,
    ਪੂਰਨ                nrows,
    ਪੂਰਨ                n_experts_ਵਰਤਿਆ,
    size_t             expert_ਭਾਰ_stride,
    size_t             dst_ਪੰਕਤੀ_stride,
    size_t             src1_ਪੰਕਤੀ_stride,
    dpct::ਕਤਾਰ_ptr    stream);

// Fused dense-FFN GEMV: writes glu(gate . y, up . y) instead of the two mat-vec results.
// vx / vgate must share shape, stride and reorder layout. Returns false if unhandled.
ਤਾਰਕਿਕ ggml_sycl_mul_mat_vec_q_glu_reorder(
    ਗਿਣਤੀ ggml_type     src0_type,
    ਗਿਣਤੀ ggml_glu_op   glu_op,
    ਅਚਲ ਖਾਲੀ *       vx,
    ਅਚਲ ਖਾਲੀ *       vgate,
    ਅਚਲ ਖਾਲੀ *       vy,
    ਦਸ਼ਮਲਵ *            dst,
    ਪੂਰਨ                ncols,                // K, shared by both weights
    ਪੂਰਨ                nrows,                // output rows, i.e. weight ne[1]
    ਪੂਰਨ                ncols_dst,            // activation columns, 1..MMVQ_MAX_BATCH_SIZE
    ਪੂਰਨ                stride_ਖਾਨਾ_y_ਬਾਈਟਾਂ,   // bytes between activation columns in vy
    ਪੂਰਨ                stride_ਖਾਨਾ_dst,       // floats between output columns in dst
    dpct::ਕਤਾਰ_ptr    stream);


// Fused dense-FFN GEMV + GLU over the standard (non-reorder) layout; the gate and up
// weights may carry different block types (q5_K / iq4_xs, mixed included).
// vy: src1 quantized with plain quantize_q8_1 (padded rows). stride_col_y is in
// block_q8_1 units. Returns false if the pair or batch is unhandled; caller falls back.
ਤਾਰਕਿਕ ggml_sycl_mul_mat_vec_q_glu_plain(
    ਗਿਣਤੀ ggml_type     gate_type,
    ਗਿਣਤੀ ggml_type     ਉੱਪਰ_type,
    ਗਿਣਤੀ ggml_glu_op   glu_op,
    ਅਚਲ ਖਾਲੀ *       vgate,
    ਅਚਲ ਖਾਲੀ *       vup,
    ਅਚਲ ਖਾਲੀ *       vy,
    ਦਸ਼ਮਲਵ *            dst,
    ਪੂਰਨ                ncols,                // K, shared by both weights
    ਪੂਰਨ                nrows,                // output rows, i.e. weight ne[1]
    ਪੂਰਨ                ncols_dst,            // activation columns, 1..MMVQ_MAX_BATCH_SIZE
    ਪੂਰਨ                stride_ਖਾਨਾ_y,         // block_q8_1 units between activation columns
    ਪੂਰਨ                stride_ਖਾਨਾ_dst,       // floats between output columns in dst
    dpct::ਕਤਾਰ_ptr    stream);

#ਜੇ_ਅੰਤ // GGML_SYCL_MMVQ_HPP
