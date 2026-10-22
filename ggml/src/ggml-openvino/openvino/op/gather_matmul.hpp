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

#సూచన once

#అన్తర్భావ "openvino/op/op.hpp"

నామాకాశ ov::op::internal {

తరగతి OPENVINO_API GatherMatmul : బహిరంగ ov::op::Op {
బహిరంగ:
    OPENVINO_OP("GatherMatmul")

    GatherMatmul() = అప్రమేయ;

    GatherMatmul(స్థిరరాశి ov::Output<Node>& A,
                 స్థిరరాశి ov::Output<Node>& B,
                 స్థిరరాశి ov::Output<Node>& indices,
                 స్థిరరాశి ov::Output<Node>& bias);

    GatherMatmul(స్థిరరాశి ov::Output<Node>& A, స్థిరరాశి ov::Output<Node>& B, స్థిరరాశి ov::Output<Node>& indices);

    std::shared_ptr<Node> clone_with_new_inputs(స్థిరరాశి ov::Output_సదిశ& new_వాదనలు) స్థిరరాశి override;

    శూన్య validate_and_infer_types() override;

అంతరంగ:
    // the weights matrix B is expected to have the transposed form [group, N, K]
    స్థిర constexpr తార్కిక transp_a = అబద్ధం;
    స్థిర constexpr తార్కిక transp_b = నిజం;
};

}  // namespace ov::op::internal
