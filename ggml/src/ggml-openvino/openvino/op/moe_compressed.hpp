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

শ্রেণী OPENVINO_API MOE : সর্বজনীন ov::op::Op {
সর্বজনীন:
    OPENVINO_OP("MOE")

    MOE() = পূর্বনির্ধারিত;

    MOE(ধ্রুবক OutputVector & args) : Op(args) {}

    গণনা শ্রেণী Expert_type { GEMM2_BIAS_SWIGLU_CLAMP, GEMM3_SWIGLU };

    গণনা শ্রেণী Activation_type { SWIGLU, GEGLU_TANH, GEGLU_ERF };

    কাঠামো Config {
        Expert_type expert_type{ Expert_type::GEMM2_BIAS_SWIGLU_CLAMP };
        দশমিক expert_alpha{ 0.0f };
        দশমিক expert_beta{ 1.0f };
        size_t gate_idx{ 0 };
        Activation_type activation_type{ Activation_type::SWIGLU };
    };

    MOE(ধ্রুবক OutputVector & args, ধ্রুবক Config & config);

    ধ্রুবক Config & get_config() ধ্রুবক;
    শূন্য set_config(ধ্রুবক Config & config);

    যৌক্তিক visit_attributes(AttributeVisitor & visitor) override;
    শূন্য validate_and_infer_types() override;
    std::shared_ptr<Node> clone_with_new_inputs(ধ্রুবক OutputVector & new_args) ধ্রুবক override;

ব্যক্তিগত:
    Config m_config;
};

শ্রেণী OPENVINO_API MOECompressed : সর্বজনীন MOE {
সর্বজনীন:
    OPENVINO_OP("MOECompressed", "", ov::op::internal::MOE)

    MOECompressed() = পূর্বনির্ধারিত;

    কাঠামো Config : সর্বজনীন MOE::Config {
        size_t hidden_আকার = 0;
        size_t inter_আকার = 0;
        size_t অঙ্ক_expert = 0;
        size_t অঙ্ক_shared_expert = 0;
        size_t শীর্ষ_k = 0;
        // numeric_limits<size_t>::max() means per_channel compression (single group)
        size_t group_আকার = 0;
        যৌক্তিক has_batch_dim = মিথ্যা;
        যৌক্তিক has_zp = মিথ্যা;
        ov::element::Type out_type = ov::element::dynamic;
        std::optional<দশমিক> scale_factor;
    };

    MOECompressed(ধ্রুবক OutputVector & args, ধ্রুবক Config & config);

    ধ্রুবক Config & get_config() ধ্রুবক { ফেরত m_config; }

    শূন্য set_scale_factor(দশমিক scale_factor) { m_config.scale_factor = scale_factor; }

    যৌক্তিক visit_attributes(AttributeVisitor & visitor) override;
    শূন্য validate_and_infer_types() override;
    std::shared_ptr<Node> clone_with_new_inputs(ধ্রুবক OutputVector & new_args) ধ্রুবক override;

সুরক্ষিত:
    Config m_config;
};

}  // namespace ov::op::internal
