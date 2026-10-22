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

#ಅನ್ತರ್ಭಾವ <optional>

#ಅನ್ತರ್ಭಾವ "openvino/core/type/element_type.hpp"
#ಅನ್ತರ್ಭಾವ "openvino/op/op.hpp"

ನಾಮಾಕಾಶ ov::op::internal {

ವರ್ಗ OPENVINO_API MOE : ಸಾರ್ವಜನಿಕ ov::op::Op {
ಸಾರ್ವಜನಿಕ:
    OPENVINO_OP("MOE")

    MOE() = ಸ್ವಭಾವ;

    MOE(ಧ್ರುವ OutputVector & args) : Op(args) {}

    ಗಣನಾ ವರ್ಗ Expert_type { GEMM2_BIAS_SWIGLU_CLAMP, GEMM3_SWIGLU };

    ಗಣನಾ ವರ್ಗ Activation_type { SWIGLU, GEGLU_TANH, GEGLU_ERF };

    ಸಂರಚನಾ Config {
        Expert_type expert_type{ Expert_type::GEMM2_BIAS_SWIGLU_CLAMP };
        ಭಿನ್ನ expert_alpha{ 0.0f };
        ಭಿನ್ನ expert_beta{ 1.0f };
        size_t gate_idx{ 0 };
        Activation_type activation_type{ Activation_type::SWIGLU };
    };

    MOE(ಧ್ರುವ OutputVector & args, ಧ್ರುವ Config & config);

    ಧ್ರುವ Config & get_config() ಧ್ರುವ;
    ಶೂನ್ಯ set_config(ಧ್ರುವ Config & config);

    ತಾರ್ಕಿಕ visit_attributes(AttributeVisitor & visitor) override;
    ಶೂನ್ಯ validate_and_infer_types() override;
    std::shared_ptr<Node> clone_with_new_inputs(ಧ್ರುವ OutputVector & new_args) ಧ್ರುವ override;

ಗುಪ್ತ:
    Config m_config;
};

ವರ್ಗ OPENVINO_API MOECompressed : ಸಾರ್ವಜನಿಕ MOE {
ಸಾರ್ವಜನಿಕ:
    OPENVINO_OP("MOECompressed", "", ov::op::internal::MOE)

    MOECompressed() = ಸ್ವಭಾವ;

    ಸಂರಚನಾ Config : ಸಾರ್ವಜನಿಕ MOE::Config {
        size_t hidden_size = 0;
        size_t inter_size = 0;
        size_t ಅಂಕೆ_expert = 0;
        size_t ಅಂಕೆ_shared_expert = 0;
        size_t ತುದಿ_k = 0;
        // numeric_limits<size_t>::max() means per_channel compression (single group)
        size_t group_size = 0;
        ತಾರ್ಕಿಕ has_batch_dim = ಅಸತ್ಯ;
        ತಾರ್ಕಿಕ has_zp = ಅಸತ್ಯ;
        ov::element::Type out_type = ov::element::dynamic;
        std::optional<ಭಿನ್ನ> scale_factor;
    };

    MOECompressed(ಧ್ರುವ OutputVector & args, ಧ್ರುವ Config & config);

    ಧ್ರುವ Config & get_config() ಧ್ರುವ { ಪ್ರತ್ಯಾಗಮ m_config; }

    ಶೂನ್ಯ set_scale_factor(ಭಿನ್ನ scale_factor) { m_config.scale_factor = scale_factor; }

    ತಾರ್ಕಿಕ visit_attributes(AttributeVisitor & visitor) override;
    ಶೂನ್ಯ validate_and_infer_types() override;
    std::shared_ptr<Node> clone_with_new_inputs(ಧ್ರುವ OutputVector & new_args) ಧ್ರುವ override;

ರಕ್ಷಿತ:
    Config m_config;
};

}  // namespace ov::op::internal
