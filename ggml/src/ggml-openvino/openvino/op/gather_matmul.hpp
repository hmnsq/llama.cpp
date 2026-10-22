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

#অন্তর্ভাৱ "openvino/op/op.hpp"

নামাকাশ ov::op::internal {

শ্ৰেণী OPENVINO_API GatherMatmul : সার্ৱজনিক ov::op::Op {
সার্ৱজনিক:
    OPENVINO_OP("GatherMatmul")

    GatherMatmul() = স্ৱভাৱ;

    GatherMatmul(ধ্রুৱ ov::Output<Node>& A,
                 ধ্রুৱ ov::Output<Node>& B,
                 ধ্রুৱ ov::Output<Node>& indices,
                 ধ্রুৱ ov::Output<Node>& bias);

    GatherMatmul(ধ্রুৱ ov::Output<Node>& A, ধ্রুৱ ov::Output<Node>& B, ধ্রুৱ ov::Output<Node>& indices);

    std::shared_ptr<Node> clone_with_new_inputs(ধ্রুৱ ov::OutputVector& new_args) ধ্রুৱ override;

    শূন্য validate_and_infer_types() override;

গুপ্ত:
    // the weights matrix B is expected to have the transposed form [group, N, K]
    স্থির constexpr তার্কিক transp_a = মিছা;
    স্থির constexpr তার্কিক transp_b = সঁচা;
};

}  // namespace ov::op::internal
