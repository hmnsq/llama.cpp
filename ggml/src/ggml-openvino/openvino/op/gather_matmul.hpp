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

#ಸೂಚನೆ once

#ಅನ್ತರ್ಭಾವ "openvino/op/op.hpp"

ನಾಮಾಕಾಶ ov::op::internal {

ವರ್ಗ OPENVINO_API GatherMatmul : ಸಾರ್ವಜನಿಕ ov::op::Op {
ಸಾರ್ವಜನಿಕ:
    OPENVINO_OP("GatherMatmul")

    GatherMatmul() = ಪೂರ್ವನಿಯೋಜಿತ;

    GatherMatmul(ಸ್ಥಿರಾಂಕ ov::Output<Node>& A,
                 ಸ್ಥಿರಾಂಕ ov::Output<Node>& B,
                 ಸ್ಥಿರಾಂಕ ov::Output<Node>& indices,
                 ಸ್ಥಿರಾಂಕ ov::Output<Node>& bias);

    GatherMatmul(ಸ್ಥಿರಾಂಕ ov::Output<Node>& A, ಸ್ಥಿರಾಂಕ ov::Output<Node>& B, ಸ್ಥಿರಾಂಕ ov::Output<Node>& indices);

    std::shared_ptr<Node> clone_with_new_inputs(ಸ್ಥಿರಾಂಕ ov::Output_ಸದಿಶ& new_ವಾದಗಳು) ಸ್ಥಿರಾಂಕ override;

    ಶೂನ್ಯ validate_and_infer_types() override;

ಖಾಸಗಿ:
    // the weights matrix B is expected to have the transposed form [group, N, K]
    ಸ್ಥಿರ constexpr ತಾರ್ಕಿಕ transp_a = ಸುಳ್ಳು;
    ಸ್ಥಿರ constexpr ತಾರ್ಕಿಕ transp_b = ನಿಜ;
};

}  // namespace ov::op::internal
