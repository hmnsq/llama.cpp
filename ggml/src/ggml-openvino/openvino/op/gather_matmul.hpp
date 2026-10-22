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

#अन्तर्भाव "openvino/op/op.hpp"

नामाकाश ov::op::internal {

वर्ग OPENVINO_API GatherMatmul : सार्वजनिक ov::op::Op {
सार्वजनिक:
    OPENVINO_OP("GatherMatmul")

    GatherMatmul() = पूर्वनिर्धारित;

    GatherMatmul(स्थिरांक ov::Output<Node>& A,
                 स्थिरांक ov::Output<Node>& B,
                 स्थिरांक ov::Output<Node>& indices,
                 स्थिरांक ov::Output<Node>& bias);

    GatherMatmul(स्थिरांक ov::Output<Node>& A, स्थिरांक ov::Output<Node>& B, स्थिरांक ov::Output<Node>& indices);

    std::shared_ptr<Node> clone_with_new_inputs(स्थिरांक ov::OutputVector& new_args) स्थिरांक override;

    शून्य validate_and_infer_types() override;

खाजगी:
    // the weights matrix B is expected to have the transposed form [group, N, K]
    स्थिर constexpr तार्किक transp_a = असत्य;
    स्थिर constexpr तार्किक transp_b = सत्य;
};

}  // namespace ov::op::internal
