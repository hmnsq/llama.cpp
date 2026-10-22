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

#અન્તર્ભાવ "openvino/op/op.hpp"

નામાકાશ ov::op::internal {

વર્ગ OPENVINO_API GatherMatmul : જાહેર ov::op::Op {
જાહેર:
    OPENVINO_OP("GatherMatmul")

    GatherMatmul() = મૂળભૂત;

    GatherMatmul(અચલ ov::Output<Node>& A,
                 અચલ ov::Output<Node>& B,
                 અચલ ov::Output<Node>& indices,
                 અચલ ov::Output<Node>& bias);

    GatherMatmul(અચલ ov::Output<Node>& A, અચલ ov::Output<Node>& B, અચલ ov::Output<Node>& indices);

    std::shared_ptr<Node> clone_with_new_inputs(અચલ ov::OutputVector& new_args) અચલ override;

    શૂન્ય validate_and_infer_types() override;

ખાનગી:
    // the weights matrix B is expected to have the transposed form [group, N, K]
    સ્થિર constexpr તાર્કિક transp_a = ખોટું;
    સ્થિર constexpr તાર્કિક transp_b = સાચું;
};

}  // namespace ov::op::internal
