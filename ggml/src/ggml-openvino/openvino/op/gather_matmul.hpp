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

#आशय once

#समावेश "openvino/op/op.hpp"

namespace ov::op::internal {

श्रेणी OPENVINO_API GatherMatmul : खुला ov::op::Op {
खुला:
    OPENVINO_OP("GatherMatmul")

    GatherMatmul() = शेष;

    GatherMatmul(स्थिर ov::Output<Node>& A,
                 स्थिर ov::Output<Node>& B,
                 स्थिर ov::Output<Node>& indices,
                 स्थिर ov::Output<Node>& bias);

    GatherMatmul(स्थिर ov::Output<Node>& A, स्थिर ov::Output<Node>& B, स्थिर ov::Output<Node>& indices);

    std::shared_ptr<Node> clone_with_new_inputs(स्थिर ov::OutputVector& new_args) स्थिर override;

    व्योम validate_and_infer_types() override;

निजी:
    // the weights matrix B is expected to have the transposed form [group, N, K]
    अटल constexpr bool transp_a = false;
    अटल constexpr bool transp_b = true;
};

}  // namespace ov::op::internal
