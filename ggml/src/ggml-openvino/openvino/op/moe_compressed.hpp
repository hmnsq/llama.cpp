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

#अन्तर्भाव <optional>

#अन्तर्भाव "openvino/core/type/element_type.hpp"
#अन्तर्भाव "openvino/op/op.hpp"

नामाकाश ov::op::internal {

वग्ग OPENVINO_API MOE : सार्वजनिक ov::op::Op {
सार्वजनिक:
    OPENVINO_OP("MOE")

    MOE() = स्वभाव;

    MOE(ध्रुव OutputVector & args) : Op(args) {}

    गणना वग्ग Expert_type { GEMM2_BIAS_SWIGLU_CLAMP, GEMM3_SWIGLU };

    गणना वग्ग Activation_type { SWIGLU, GEGLU_TANH, GEGLU_ERF };

    संरचना Config {
        Expert_type expert_type{ Expert_type::GEMM2_BIAS_SWIGLU_CLAMP };
        भिन्न expert_alpha{ 0.0f };
        भिन्न expert_beta{ 1.0f };
        size_t gate_idx{ 0 };
        Activation_type activation_type{ Activation_type::SWIGLU };
    };

    MOE(ध्रुव OutputVector & args, ध्रुव Config & config);

    ध्रुव Config & get_config() ध्रुव;
    शून्य set_config(ध्रुव Config & config);

    तार्किक visit_attributes(AttributeVisitor & visitor) override;
    शून्य validate_and_infer_types() override;
    std::shared_ptr<Node> clone_with_new_inputs(ध्रुव OutputVector & new_args) ध्रुव override;

गुप्त:
    Config m_config;
};

वग्ग OPENVINO_API MOECompressed : सार्वजनिक MOE {
सार्वजनिक:
    OPENVINO_OP("MOECompressed", "", ov::op::internal::MOE)

    MOECompressed() = स्वभाव;

    संरचना Config : सार्वजनिक MOE::Config {
        size_t hidden_size = 0;
        size_t inter_size = 0;
        size_t अङ्कः_expert = 0;
        size_t अङ्कः_shared_expert = 0;
        size_t शिखरम्_k = 0;
        // numeric_limits<size_t>::max() means per_channel compression (single group)
        size_t group_size = 0;
        तार्किक has_batch_dim = मिच्छा;
        तार्किक has_zp = मिच्छा;
        ov::element::Type out_type = ov::element::dynamic;
        std::optional<भिन्न> scale_factor;
    };

    MOECompressed(ध्रुव OutputVector & args, ध्रुव Config & config);

    ध्रुव Config & get_config() ध्रुव { पटिनिवत्त m_config; }

    शून्य set_scale_factor(भिन्न scale_factor) { m_config.scale_factor = scale_factor; }

    तार्किक visit_attributes(AttributeVisitor & visitor) override;
    शून्य validate_and_infer_types() override;
    std::shared_ptr<Node> clone_with_new_inputs(ध्रुव OutputVector & new_args) ध्रुव override;

रक्षित:
    Config m_config;
};

}  // namespace ov::op::internal
