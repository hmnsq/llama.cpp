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

#ಅನ್ತರ್ಭಾವ "openvino/op/op.hpp"

ನಾಮಾಕಾಶ ov::op::internal {

ವರ್ಗ OPENVINO_API GatherMatmul : ಸಾರ್ವಜನಿಕ ov::op::Op {
ಸಾರ್ವಜನಿಕ:
    OPENVINO_OP("GatherMatmul")

    GatherMatmul() = ಸ್ವಭಾವ;

    GatherMatmul(ಧ್ರುವ ov::Output<Node>& A,
                 ಧ್ರುವ ov::Output<Node>& B,
                 ಧ್ರುವ ov::Output<Node>& indices,
                 ಧ್ರುವ ov::Output<Node>& bias);

    GatherMatmul(ಧ್ರುವ ov::Output<Node>& A, ಧ್ರುವ ov::Output<Node>& B, ಧ್ರುವ ov::Output<Node>& indices);

    std::shared_ptr<Node> clone_with_new_inputs(ಧ್ರುವ ov::OutputVector& new_args) ಧ್ರುವ override;

    ಶೂನ್ಯ validate_and_infer_types() override;

ಗುಪ್ತ:
    // the weights matrix B is expected to have the transposed form [group, N, K]
    ಸ್ಥಿರ constexpr ತಾರ್ಕಿಕ transp_a = ಅಸತ್ಯ;
    ಸ್ಥಿರ constexpr ತಾರ್ಕಿಕ transp_b = ಸತ್ಯ;
};

}  // namespace ov::op::internal
