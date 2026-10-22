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

#اگر_غیر_تعریف_شدہ GGML_SYCL_MMVQ_HPP
#تعریف_کرو GGML_SYCL_MMVQ_HPP

#شامل "common.hpp"

// q5_K multi-column MMVQ shares weights across columns, pairs rows and fuses gate/up in the reorder
// layout: faster on Xe2 (BMG), so untested archs keep the per-column kernel
اندرونی منطقی ggml_sycl_q5_k_mmvq_reuse(صحیح device) {
    ثابت gpu_arch arch = ggml_sycl_info().devices[device].hw_info.arch;
    واپس arch == gpu_arch::intel_gpu_bmg_g21 || arch == gpu_arch::intel_gpu_bmg_g31;
}

خالی ggml_sycl_op_mul_mat_vec_q(
    ggml_backend_sycl_context & ctx,
    ثابت ggml_tensor *src0, ثابت ggml_tensor *src1, ggml_tensor *dst,
    ثابت حرف *src0_dd_i, ثابت اعشاری *src1_ddf_i, ثابت حرف *src1_ddq_i,
    اعشاری *dst_dd_i, ثابت int64_t صف_کم, ثابت int64_t صف_بلند,
    ثابت int64_t src1_ncols, ثابت int64_t src1_padded_صف_سائز,
    ثابت dpct::قطار_ptr &stream);

// Requires standard (non-reorder) block layout for src0.
// Returns false if src0_type isn't handled; caller should fall back.
منطقی ggml_sycl_mul_mat_vec_q_id(
    شمار ggml_type     src0_type,
    ثابت خالی *       vx_بنیاد,             // start of stacked expert weights
    ثابت خالی *       vy,                  // pre-quantized src1 (Q8_1)
    ثابت int32_t *    ids_dev,             // device-side int32, length n_experts_used
    اعشاری *            dst_بنیاد,
    صحیح                ncols,
    صحیح                nrows,
    صحیح                n_experts_مستعمل,
    size_t             expert_وزن_stride, // bytes between experts in vx_base
    size_t             dst_صف_stride,       // bytes between dst rows
    size_t             src1_صف_stride,      // 0 = shared src1, else per-expert stride in bytes
    dpct::قطار_ptr    stream);

// Reorder (SoA) variant of the fused MoE expert GEMV.
// vx_base: each expert slice (stride expert_weight_stride == src0->nb[2]) is a self-contained reorder/SoA layout.
// vy: src1 quantized with quantize_and_reorder_q8_1_soa (per-row SoA). Returns false if src0_type isn't handled.
منطقی ggml_sycl_mul_mat_vec_q_id_reorder(
    شمار ggml_type     src0_type,
    ثابت خالی *       vx_بنیاد,
    ثابت خالی *       vy,
    ثابت int32_t *    ids_dev,
    اعشاری *            dst_بنیاد,
    صحیح                ncols,
    صحیح                nrows,
    صحیح                n_experts_مستعمل,
    size_t             expert_وزن_stride,
    size_t             dst_صف_stride,
    size_t             src1_صف_stride,
    dpct::قطار_ptr    stream);

// Fused dense-FFN GEMV: writes glu(gate . y, up . y) instead of the two mat-vec results.
// vx / vgate must share shape, stride and reorder layout. Returns false if unhandled.
منطقی ggml_sycl_mul_mat_vec_q_glu_reorder(
    شمار ggml_type     src0_type,
    شمار ggml_glu_op   glu_op,
    ثابت خالی *       vx,
    ثابت خالی *       vgate,
    ثابت خالی *       vy,
    اعشاری *            dst,
    صحیح                ncols,                // K, shared by both weights
    صحیح                nrows,                // output rows, i.e. weight ne[1]
    صحیح                ncols_dst,            // activation columns, 1..MMVQ_MAX_BATCH_SIZE
    صحیح                stride_ستون_y_بائٹیں,   // bytes between activation columns in vy
    صحیح                stride_ستون_dst,       // floats between output columns in dst
    dpct::قطار_ptr    stream);


// Fused dense-FFN GEMV + GLU over the standard (non-reorder) layout; the gate and up
// weights may carry different block types (q5_K / iq4_xs, mixed included).
// vy: src1 quantized with plain quantize_q8_1 (padded rows). stride_col_y is in
// block_q8_1 units. Returns false if the pair or batch is unhandled; caller falls back.
منطقی ggml_sycl_mul_mat_vec_q_glu_plain(
    شمار ggml_type     gate_type,
    شمار ggml_type     اوپر_type,
    شمار ggml_glu_op   glu_op,
    ثابت خالی *       vgate,
    ثابت خالی *       vup,
    ثابت خالی *       vy,
    اعشاری *            dst,
    صحیح                ncols,                // K, shared by both weights
    صحیح                nrows,                // output rows, i.e. weight ne[1]
    صحیح                ncols_dst,            // activation columns, 1..MMVQ_MAX_BATCH_SIZE
    صحیح                stride_ستون_y,         // block_q8_1 units between activation columns
    صحیح                stride_ستون_dst,       // floats between output columns in dst
    dpct::قطار_ptr    stream);

#اگر_ختم // GGML_SYCL_MMVQ_HPP
