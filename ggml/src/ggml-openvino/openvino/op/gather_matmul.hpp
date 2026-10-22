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

#അന്തര്ഭാവ "openvino/op/op.hpp"

നാമാകാശ ov::op::internal {

വർഗ്ഗം OPENVINO_API GatherMatmul : പൊതു ov::op::Op {
പൊതു:
    OPENVINO_OP("GatherMatmul")

    GatherMatmul() = സ്ഥിരസ്ഥിതി;

    GatherMatmul(സ്ഥിരം ov::Output<Node>& A,
                 സ്ഥിരം ov::Output<Node>& B,
                 സ്ഥിരം ov::Output<Node>& indices,
                 സ്ഥിരം ov::Output<Node>& bias);

    GatherMatmul(സ്ഥിരം ov::Output<Node>& A, സ്ഥിരം ov::Output<Node>& B, സ്ഥിരം ov::Output<Node>& indices);

    std::shared_ptr<Node> clone_with_new_inputs(സ്ഥിരം ov::OutputVector& new_args) സ്ഥിരം override;

    ശൂന്യം validate_and_infer_types() override;

സ്വകാര്യ:
    // the weights matrix B is expected to have the transposed form [group, N, K]
    സ്ഥിതം constexpr തർക്കം transp_a = അസത്യം;
    സ്ഥിതം constexpr തർക്കം transp_b = സത്യം;
};

}  // namespace ov::op::internal
