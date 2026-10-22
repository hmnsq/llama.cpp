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

#अगर_अघोषित GGML_SYCL_MMVQ_HPP
#घोषणा GGML_SYCL_MMVQ_HPP

#समावेश "common.hpp"

// q5_K multi-column MMVQ shares weights across columns, pairs rows and fuses gate/up in the reorder
// layout: faster on Xe2 (BMG), so untested archs keep the per-column kernel
अंतरभूत bool ggml_sycl_q5_k_mmvq_reuse(पूर्णांक device) {
    स्थिर gpu_arch arch = ggml_sycl_info().devices[device].hw_info.arch;
    वापस arch == gpu_arch::intel_gpu_bmg_g21 || arch == gpu_arch::intel_gpu_bmg_g31;
}

व्योम ggml_sycl_op_mul_mat_vec_q(
    ggml_backend_sycl_context & ctx,
    स्थिर ggml_tensor *src0, स्थिर ggml_tensor *src1, ggml_tensor *dst,
    स्थिर अक्षर *src0_dd_i, स्थिर भग्न *src1_ddf_i, स्थिर अक्षर *src1_ddq_i,
    भग्न *dst_dd_i, स्थिर int64_t row_निचला, स्थिर int64_t row_ऊपरी,
    स्थिर int64_t src1_ncols, स्थिर int64_t src1_padded_row_आकार,
    स्थिर dpct::queue_ptr &stream);

// Requires standard (non-reorder) block layout for src0.
// Returns false if src0_type isn't handled; caller should fall back.
bool ggml_sycl_mul_mat_vec_q_id(
    क्रमागत ggml_type     src0_type,
    स्थिर व्योम *       vx_base,             // start of stacked expert weights
    स्थिर व्योम *       vy,                  // pre-quantized src1 (Q8_1)
    स्थिर int32_t *    ids_dev,             // device-side int32, length n_experts_used
    भग्न *            dst_base,
    पूर्णांक                ncols,
    पूर्णांक                nrows,
    पूर्णांक                n_experts_used,
    माप_प्रकार             expert_weight_stride, // bytes between experts in vx_base
    माप_प्रकार             dst_row_stride,       // bytes between dst rows
    माप_प्रकार             src1_row_stride,      // 0 = shared src1, else per-expert stride in bytes
    dpct::queue_ptr    stream);

// Reorder (SoA) variant of the fused MoE expert GEMV.
// vx_base: each expert slice (stride expert_weight_stride == src0->nb[2]) is a self-contained reorder/SoA layout.
// vy: src1 quantized with quantize_and_reorder_q8_1_soa (per-row SoA). Returns false if src0_type isn't handled.
bool ggml_sycl_mul_mat_vec_q_id_reorder(
    क्रमागत ggml_type     src0_type,
    स्थिर व्योम *       vx_base,
    स्थिर व्योम *       vy,
    स्थिर int32_t *    ids_dev,
    भग्न *            dst_base,
    पूर्णांक                ncols,
    पूर्णांक                nrows,
    पूर्णांक                n_experts_used,
    माप_प्रकार             expert_weight_stride,
    माप_प्रकार             dst_row_stride,
    माप_प्रकार             src1_row_stride,
    dpct::queue_ptr    stream);

// Fused dense-FFN GEMV: writes glu(gate . y, up . y) instead of the two mat-vec results.
// vx / vgate must share shape, stride and reorder layout. Returns false if unhandled.
bool ggml_sycl_mul_mat_vec_q_glu_reorder(
    क्रमागत ggml_type     src0_type,
    क्रमागत ggml_glu_op   glu_op,
    स्थिर व्योम *       vx,
    स्थिर व्योम *       vgate,
    स्थिर व्योम *       vy,
    भग्न *            dst,
    पूर्णांक                ncols,                // K, shared by both weights
    पूर्णांक                nrows,                // output rows, i.e. weight ne[1]
    पूर्णांक                ncols_dst,            // activation columns, 1..MMVQ_MAX_BATCH_SIZE
    पूर्णांक                stride_col_y_bytes,   // bytes between activation columns in vy
    पूर्णांक                stride_col_dst,       // floats between output columns in dst
    dpct::queue_ptr    stream);


// Fused dense-FFN GEMV + GLU over the standard (non-reorder) layout; the gate and up
// weights may carry different block types (q5_K / iq4_xs, mixed included).
// vy: src1 quantized with plain quantize_q8_1 (padded rows). stride_col_y is in
// block_q8_1 units. Returns false if the pair or batch is unhandled; caller falls back.
bool ggml_sycl_mul_mat_vec_q_glu_plain(
    क्रमागत ggml_type     gate_type,
    क्रमागत ggml_type     up_type,
    क्रमागत ggml_glu_op   glu_op,
    स्थिर व्योम *       vgate,
    स्थिर व्योम *       vup,
    स्थिर व्योम *       vy,
    भग्न *            dst,
    पूर्णांक                ncols,                // K, shared by both weights
    पूर्णांक                nrows,                // output rows, i.e. weight ne[1]
    पूर्णांक                ncols_dst,            // activation columns, 1..MMVQ_MAX_BATCH_SIZE
    पूर्णांक                stride_col_y,         // block_q8_1 units between activation columns
    पूर्णांक                stride_col_dst,       // floats between output columns in dst
    dpct::queue_ptr    stream);

#पूर्ण_अगर // GGML_SYCL_MMVQ_HPP
