// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// Local mirror of OpenVINO's internal ov::op::internal::GatherMatmul op.
//
// The op class body (validate_and_infer_types / clone_with_new_inputs) is
// provided by the linked libopenvino.so; only the declaration is needed here so
// the backend can construct the node directly (same approach as GatedDeltaNet).
// The class layout must stay in sync with
//   openvino/src/common/transformations/include/ov_ops/gather_matmul.hpp
//
// \note GatherMatmul op class is under development and subject to change.

#pragma once

#অন্তর্ভাব "openvino/op/op.hpp"

নামাকাশ ov::op::internal {

বর্গ OPENVINO_API GatherMatmul : সার্বজনিক ov::op::Op {
সার্বজনিক:
    OPENVINO_OP("GatherMatmul")

    GatherMatmul() = স্বভাব;

    GatherMatmul(ধ্রুব ov::Output<Node>& A,
                 ধ্রুব ov::Output<Node>& B,
                 ধ্রুব ov::Output<Node>& indices,
                 ধ্রুব ov::Output<Node>& bias);

    GatherMatmul(ধ্রুব ov::Output<Node>& A, ধ্রুব ov::Output<Node>& B, ধ্রুব ov::Output<Node>& indices);

    std::shared_ptr<Node> clone_with_new_inputs(ধ্রুব ov::OutputVector& new_args) ধ্রুব override;

    শূন্য validate_and_infer_types() override;

গুপ্ত:
    // the weights matrix B is expected to have the transposed form [group, N, K]
    স্থির constexpr তার্কিক transp_a = অসত্য;
    স্থির constexpr তার্কিক transp_b = সত্য;
};

}  // namespace ov::op::internal
