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

#ਹਦਾਇਤ once

#ਅਨ੍ਤਰ੍ਭਾਵ "openvino/op/op.hpp"

ਨਾਮਾਕਾਸ਼ ov::op::internal {

ਵਰਗ OPENVINO_API GatherMatmul : ਜਨਤਕ ov::op::Op {
ਜਨਤਕ:
    OPENVINO_OP("GatherMatmul")

    GatherMatmul() = ਮੂਲ;

    GatherMatmul(ਅਚਲ ov::Output<Node>& A,
                 ਅਚਲ ov::Output<Node>& B,
                 ਅਚਲ ov::Output<Node>& indices,
                 ਅਚਲ ov::Output<Node>& bias);

    GatherMatmul(ਅਚਲ ov::Output<Node>& A, ਅਚਲ ov::Output<Node>& B, ਅਚਲ ov::Output<Node>& indices);

    std::shared_ptr<Node> clone_with_new_inputs(ਅਚਲ ov::Output_ਸਦਿਸ਼& new_ਦਲੀਲਾਂ) ਅਚਲ override;

    ਖਾਲੀ validate_and_infer_types() override;

ਨਿਜੀ:
    // the weights matrix B is expected to have the transposed form [group, N, K]
    ਸਥਿਰ constexpr ਤਾਰਕਿਕ transp_a = ਝੂਠ;
    ਸਥਿਰ constexpr ਤਾਰਕਿਕ transp_b = ਸੱਚ;
};

}  // namespace ov::op::internal
