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

#அந்தர்பாவ <optional>

#அந்தர்பாவ "openvino/core/type/element_type.hpp"
#அந்தர்பாவ "openvino/op/op.hpp"

நாமாகாஶ ov::op::internal {

வகுப்பு OPENVINO_API MOE : பொது ov::op::Op {
பொது:
    OPENVINO_OP("MOE")

    MOE() = இயல்பு;

    MOE(மாறிலி OutputVector & args) : Op(args) {}

    எண்ணிக்கை வகுப்பு Expert_type { GEMM2_BIAS_SWIGLU_CLAMP, GEMM3_SWIGLU };

    எண்ணிக்கை வகுப்பு Activation_type { SWIGLU, GEGLU_TANH, GEGLU_ERF };

    கட்டமைப்பு Config {
        Expert_type expert_type{ Expert_type::GEMM2_BIAS_SWIGLU_CLAMP };
        தசம expert_alpha{ 0.0f };
        தசம expert_beta{ 1.0f };
        size_t gate_idx{ 0 };
        Activation_type activation_type{ Activation_type::SWIGLU };
    };

    MOE(மாறிலி OutputVector & args, மாறிலி Config & config);

    மாறிலி Config & get_config() மாறிலி;
    வெறுமை set_config(மாறிலி Config & config);

    தருக்க visit_attributes(AttributeVisitor & visitor) override;
    வெறுமை validate_and_infer_types() override;
    std::shared_ptr<Node> clone_with_new_inputs(மாறிலி OutputVector & new_args) மாறிலி override;

தனிப்பட்ட:
    Config m_config;
};

வகுப்பு OPENVINO_API MOECompressed : பொது MOE {
பொது:
    OPENVINO_OP("MOECompressed", "", ov::op::internal::MOE)

    MOECompressed() = இயல்பு;

    கட்டமைப்பு Config : பொது MOE::Config {
        size_t hidden_size = 0;
        size_t inter_size = 0;
        size_t இலக்கம்_expert = 0;
        size_t இலக்கம்_shared_expert = 0;
        size_t உச்சி_k = 0;
        // numeric_limits<size_t>::max() means per_channel compression (single group)
        size_t group_size = 0;
        தருக்க has_batch_dim = பொய்;
        தருக்க has_zp = பொய்;
        ov::element::Type out_type = ov::element::dynamic;
        std::optional<தசம> scale_factor;
    };

    MOECompressed(மாறிலி OutputVector & args, மாறிலி Config & config);

    மாறிலி Config & get_config() மாறிலி { திரும்பு m_config; }

    வெறுமை set_scale_factor(தசம scale_factor) { m_config.scale_factor = scale_factor; }

    தருக்க visit_attributes(AttributeVisitor & visitor) override;
    வெறுமை validate_and_infer_types() override;
    std::shared_ptr<Node> clone_with_new_inputs(மாறிலி OutputVector & new_args) மாறிலி override;

பாதுகாக்கப்பட்ட:
    Config m_config;
};

}  // namespace ov::op::internal
