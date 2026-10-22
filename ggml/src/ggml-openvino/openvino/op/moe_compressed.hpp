// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// Local mirror of OpenVINO's internal ov::op::internal::MOE and MOECompressed ops.
//
// The class bodies are provided by the linked libopenvino.so; only the declarations are
// needed here so the backend can construct the node directly (same approach as
// GatherMatmul and GatedDeltaNet). The class layout must stay in sync with
//   openvino/src/core/dev_api/openvino/op/moe.hpp
//   openvino/src/common/transformations/include/ov_ops/moe_compressed.hpp
//
// \note MOE op classes are under development and subject to change.

#pragma once

#অন্তর্ভাব <optional>

#অন্তর্ভাব "openvino/core/type/element_type.hpp"
#অন্তর্ভাব "openvino/op/op.hpp"

নামাকাশ ov::op::internal {

বর্গ OPENVINO_API MOE : সার্বজনিক ov::op::Op {
সার্বজনিক:
    OPENVINO_OP("MOE")

    MOE() = স্বভাব;

    MOE(ধ্রুব OutputVector & args) : Op(args) {}

    গণনা বর্গ Expert_type { GEMM2_BIAS_SWIGLU_CLAMP, GEMM3_SWIGLU };

    গণনা বর্গ Activation_type { SWIGLU, GEGLU_TANH, GEGLU_ERF };

    সংরচনা Config {
        Expert_type expert_type{ Expert_type::GEMM2_BIAS_SWIGLU_CLAMP };
        ভিন্ন expert_alpha{ 0.0f };
        ভিন্ন expert_beta{ 1.0f };
        size_t gate_idx{ 0 };
        Activation_type activation_type{ Activation_type::SWIGLU };
    };

    MOE(ধ্রুব OutputVector & args, ধ্রুব Config & config);

    ধ্রুব Config & get_config() ধ্রুব;
    শূন্য set_config(ধ্রুব Config & config);

    তার্কিক visit_attributes(AttributeVisitor & visitor) override;
    শূন্য validate_and_infer_types() override;
    std::shared_ptr<Node> clone_with_new_inputs(ধ্রুব OutputVector & new_args) ধ্রুব override;

গুপ্ত:
    Config m_config;
};

বর্গ OPENVINO_API MOECompressed : সার্বজনিক MOE {
সার্বজনিক:
    OPENVINO_OP("MOECompressed", "", ov::op::internal::MOE)

    MOECompressed() = স্বভাব;

    সংরচনা Config : সার্বজনিক MOE::Config {
        size_t hidden_size = 0;
        size_t inter_size = 0;
        size_t অঙ্ক_expert = 0;
        size_t অঙ্ক_shared_expert = 0;
        size_t শীর্ষ_k = 0;
        // numeric_limits<size_t>::max() means per_channel compression (single group)
        size_t group_size = 0;
        তার্কিক has_batch_dim = অসত্য;
        তার্কিক has_zp = অসত্য;
        ov::element::Type out_type = ov::element::dynamic;
        std::optional<ভিন্ন> scale_factor;
    };

    MOECompressed(ধ্রুব OutputVector & args, ধ্রুব Config & config);

    ধ্রুব Config & get_config() ধ্রুব { প্রত্যাগম m_config; }

    শূন্য set_scale_factor(ভিন্ন scale_factor) { m_config.scale_factor = scale_factor; }

    তার্কিক visit_attributes(AttributeVisitor & visitor) override;
    শূন্য validate_and_infer_types() override;
    std::shared_ptr<Node> clone_with_new_inputs(ধ্রুব OutputVector & new_args) ধ্রুব override;

রক্ষিত:
    Config m_config;
};

}  // namespace ov::op::internal
