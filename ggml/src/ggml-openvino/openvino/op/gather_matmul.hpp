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

#ضمن "openvino/op/op.hpp"

نطاق ov::op::internal {

صنف OPENVINO_API GatherMatmul : عام ov::op::Op {
عام:
    OPENVINO_OP("GatherMatmul")

    GatherMatmul() = افتراضي;

    GatherMatmul(ثابت ov::Output<Node>& A,
                 ثابت ov::Output<Node>& B,
                 ثابت ov::Output<Node>& indices,
                 ثابت ov::Output<Node>& bias);

    GatherMatmul(ثابت ov::Output<Node>& A, ثابت ov::Output<Node>& B, ثابت ov::Output<Node>& indices);

    std::shared_ptr<Node> clone_with_new_inputs(ثابت ov::OutputVector& new_args) ثابت override;

    فراغ validate_and_infer_types() override;

خاص:
    // the weights matrix B is expected to have the transposed form [group, N, K]
    ساكن constexpr منطقي transp_a = ܕܓܠܐ;
    ساكن constexpr منطقي transp_b = ܫܪܪܐ;
};

}  // namespace ov::op::internal
