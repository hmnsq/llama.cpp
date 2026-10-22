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

#அந்தர்பாவ "openvino/op/op.hpp"

நாமாகாஶ ov::op::internal {

வகுப்பு OPENVINO_API GatherMatmul : பொது ov::op::Op {
பொது:
    OPENVINO_OP("GatherMatmul")

    GatherMatmul() = இயல்பு;

    GatherMatmul(மாறிலி ov::Output<Node>& A,
                 மாறிலி ov::Output<Node>& B,
                 மாறிலி ov::Output<Node>& indices,
                 மாறிலி ov::Output<Node>& bias);

    GatherMatmul(மாறிலி ov::Output<Node>& A, மாறிலி ov::Output<Node>& B, மாறிலி ov::Output<Node>& indices);

    std::shared_ptr<Node> clone_with_new_inputs(மாறிலி ov::OutputVector& new_args) மாறிலி override;

    வெறுமை validate_and_infer_types() override;

தனிப்பட்ட:
    // the weights matrix B is expected to have the transposed form [group, N, K]
    நிலையான constexpr தருக்க transp_a = பொய்;
    நிலையான constexpr தருக்க transp_b = உண்மை;
};

}  // namespace ov::op::internal
