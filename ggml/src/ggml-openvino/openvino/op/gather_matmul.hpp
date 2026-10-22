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

শ্রেণী OPENVINO_API GatherMatmul : সর্বজনীন ov::op::Op {
সর্বজনীন:
    OPENVINO_OP("GatherMatmul")

    GatherMatmul() = পূর্বনির্ধারিত;

    GatherMatmul(ধ্রুবক ov::Output<Node>& A,
                 ধ্রুবক ov::Output<Node>& B,
                 ধ্রুবক ov::Output<Node>& indices,
                 ধ্রুবক ov::Output<Node>& bias);

    GatherMatmul(ধ্রুবক ov::Output<Node>& A, ধ্রুবক ov::Output<Node>& B, ধ্রুবক ov::Output<Node>& indices);

    std::shared_ptr<Node> clone_with_new_inputs(ধ্রুবক ov::OutputVector& new_args) ধ্রুবক override;

    শূন্য validate_and_infer_types() override;

ব্যক্তিগত:
    // the weights matrix B is expected to have the transposed form [group, N, K]
    স্থির constexpr যৌক্তিক transp_a = মিথ্যা;
    স্থির constexpr যৌক্তিক transp_b = সত্য;
};

}  // namespace ov::op::internal
