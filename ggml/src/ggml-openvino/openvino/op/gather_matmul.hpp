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

#ଅନ୍ତର୍ଭାଵ "openvino/op/op.hpp"

ନାମାକାଶ ov::op::internal {

ଶ୍ରେଣୀ OPENVINO_API GatherMatmul : ସର୍ବସାଧାରଣ ov::op::Op {
ସର୍ବସାଧାରଣ:
    OPENVINO_OP("GatherMatmul")

    GatherMatmul() = ପୂର୍ବନିର୍ଦ୍ଧାରିତ;

    GatherMatmul(ଅଚଳ ov::Output<Node>& A,
                 ଅଚଳ ov::Output<Node>& B,
                 ଅଚଳ ov::Output<Node>& indices,
                 ଅଚଳ ov::Output<Node>& bias);

    GatherMatmul(ଅଚଳ ov::Output<Node>& A, ଅଚଳ ov::Output<Node>& B, ଅଚଳ ov::Output<Node>& indices);

    std::shared_ptr<Node> clone_with_new_inputs(ଅଚଳ ov::OutputVector& new_args) ଅଚଳ override;

    ଶୂନ୍ୟ validate_and_infer_types() override;

ବ୍ୟକ୍ତିଗତ:
    // the weights matrix B is expected to have the transposed form [group, N, K]
    ସ୍ଥିର constexpr ତାର୍କିକ transp_a = ମିଥ୍ୟା;
    ସ୍ଥିର constexpr ତାର୍କିକ transp_b = ସତ୍ୟ;
};

}  // namespace ov::op::internal
