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

#اگر_تعریف_نشده GGML_SYCL_MMVQ_HPP
#تعریف_کن GGML_SYCL_MMVQ_HPP

#ضمن "common.hpp"

// q5_K multi-column MMVQ shares weights across columns, pairs rows and fuses gate/up in the reorder
// layout: faster on Xe2 (BMG), so untested archs keep the per-column kernel
مضمن منطقي ggml_sycl_q5_k_mmvq_reuse(صحيح device) {
    ثابت gpu_arch arch = ggml_sycl_info().devices[device].hw_info.arch;
    أرجع arch == gpu_arch::intel_gpu_bmg_g21 || arch == gpu_arch::intel_gpu_bmg_g31;
}

فراغ ggml_sycl_op_mul_mat_vec_q(
    ggml_backend_sycl_context & ctx,
    ثابت ggml_tensor *src0, ثابت ggml_tensor *src1, ggml_tensor *dst,
    ثابت حرف *src0_dd_i, ثابت عائم *src1_ddf_i, ثابت حرف *src1_ddq_i,
    عائم *dst_dd_i, ثابت int64_t ردیف_کم, ثابت int64_t ردیف_بالا,
    ثابت int64_t src1_ncols, ثابت int64_t src1_padded_ردیف_اندازه,
    ثابت dpct::صف_ptr &stream);

// Requires standard (non-reorder) block layout for src0.
// Returns false if src0_type isn't handled; caller should fall back.
منطقي ggml_sycl_mul_mat_vec_q_id(
    تعداد ggml_type     src0_type,
    ثابت فراغ *       vx_پایه,             // start of stacked expert weights
    ثابت فراغ *       vy,                  // pre-quantized src1 (Q8_1)
    ثابت int32_t *    ids_dev,             // device-side int32, length n_experts_used
    عائم *            dst_پایه,
    صحيح                ncols,
    صحيح                nrows,
    صحيح                n_experts_استفاده_شده,
    size_t             expert_وزن_stride, // bytes between experts in vx_base
    size_t             dst_ردیف_stride,       // bytes between dst rows
    size_t             src1_ردیف_stride,      // 0 = shared src1, else per-expert stride in bytes
    dpct::صف_ptr    stream);

// Reorder (SoA) variant of the fused MoE expert GEMV.
// vx_base: each expert slice (stride expert_weight_stride == src0->nb[2]) is a self-contained reorder/SoA layout.
// vy: src1 quantized with quantize_and_reorder_q8_1_soa (per-row SoA). Returns false if src0_type isn't handled.
منطقي ggml_sycl_mul_mat_vec_q_id_reorder(
    تعداد ggml_type     src0_type,
    ثابت فراغ *       vx_پایه,
    ثابت فراغ *       vy,
    ثابت int32_t *    ids_dev,
    عائم *            dst_پایه,
    صحيح                ncols,
    صحيح                nrows,
    صحيح                n_experts_استفاده_شده,
    size_t             expert_وزن_stride,
    size_t             dst_ردیف_stride,
    size_t             src1_ردیف_stride,
    dpct::صف_ptr    stream);

// Fused dense-FFN GEMV: writes glu(gate . y, up . y) instead of the two mat-vec results.
// vx / vgate must share shape, stride and reorder layout. Returns false if unhandled.
منطقي ggml_sycl_mul_mat_vec_q_glu_reorder(
    تعداد ggml_type     src0_type,
    تعداد ggml_glu_op   glu_op,
    ثابت فراغ *       vx,
    ثابت فراغ *       vgate,
    ثابت فراغ *       vy,
    عائم *            dst,
    صحيح                ncols,                // K, shared by both weights
    صحيح                nrows,                // output rows, i.e. weight ne[1]
    صحيح                ncols_dst,            // activation columns, 1..MMVQ_MAX_BATCH_SIZE
    صحيح                stride_ستون_y_بایتها,   // bytes between activation columns in vy
    صحيح                stride_ستون_dst,       // floats between output columns in dst
    dpct::صف_ptr    stream);


// Fused dense-FFN GEMV + GLU over the standard (non-reorder) layout; the gate and up
// weights may carry different block types (q5_K / iq4_xs, mixed included).
// vy: src1 quantized with plain quantize_q8_1 (padded rows). stride_col_y is in
// block_q8_1 units. Returns false if the pair or batch is unhandled; caller falls back.
منطقي ggml_sycl_mul_mat_vec_q_glu_plain(
    تعداد ggml_type     gate_type,
    تعداد ggml_type     به_بالا_type,
    تعداد ggml_glu_op   glu_op,
    ثابت فراغ *       vgate,
    ثابت فراغ *       vup,
    ثابت فراغ *       vy,
    عائم *            dst,
    صحيح                ncols,                // K, shared by both weights
    صحيح                nrows,                // output rows, i.e. weight ne[1]
    صحيح                ncols_dst,            // activation columns, 1..MMVQ_MAX_BATCH_SIZE
    صحيح                stride_ستون_y,         // block_q8_1 units between activation columns
    صحيح                stride_ستون_dst,       // floats between output columns in dst
    dpct::صف_ptr    stream);

#پایان_اگر // GGML_SYCL_MMVQ_HPP
