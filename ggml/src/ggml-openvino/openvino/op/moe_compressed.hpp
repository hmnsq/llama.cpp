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

#ಸೂಚನೆ once

#ಅನ್ತರ್ಭಾವ <optional>

#ಅನ್ತರ್ಭಾವ "openvino/core/type/element_type.hpp"
#ಅನ್ತರ್ಭಾವ "openvino/op/op.hpp"

ನಾಮಾಕಾಶ ov::op::internal {

ವರ್ಗ OPENVINO_API MOE : ಸಾರ್ವಜನಿಕ ov::op::Op {
ಸಾರ್ವಜನಿಕ:
    OPENVINO_OP("MOE")

    MOE() = ಪೂರ್ವನಿಯೋಜಿತ;

    MOE(ಸ್ಥಿರಾಂಕ Output_ಸದಿಶ & ವಾದಗಳು_2) : Op(ವಾದಗಳು_2) {}

    ಎಣಿಕೆ ವರ್ಗ Expert_type { GEMM2_BIAS_SWIGLU_CLAMP, GEMM3_SWIGLU };

    ಎಣಿಕೆ ವರ್ಗ Activation_type { SWIGLU, GEGLU_TANH, GEGLU_ERF };

    ರಚನೆ Config {
        Expert_type expert_type{ Expert_type::GEMM2_BIAS_SWIGLU_CLAMP };
        ದಶಮಾಂಶ expert_alpha{ 0.0f };
        ದಶಮಾಂಶ expert_beta{ 1.0f };
        size_t gate_idx{ 0 };
        Activation_type activation_type{ Activation_type::SWIGLU };
    };

    MOE(ಸ್ಥಿರಾಂಕ Output_ಸದಿಶ & ವಾದಗಳು_2, ಸ್ಥಿರಾಂಕ Config & config);

    ಸ್ಥಿರಾಂಕ Config & get_config() ಸ್ಥಿರಾಂಕ;
    ಶೂನ್ಯ ಗಣ_config(ಸ್ಥಿರಾಂಕ Config & config);

    ತಾರ್ಕಿಕ ಭೇಟಿ_ಗುಣಲಕ್ಷಣಗಳು(AttributeVisitor & visitor) override;
    ಶೂನ್ಯ validate_and_infer_types() override;
    std::shared_ptr<Node> clone_with_new_inputs(ಸ್ಥಿರಾಂಕ Output_ಸದಿಶ & new_ವಾದಗಳು) ಸ್ಥಿರಾಂಕ override;

ಖಾಸಗಿ:
    Config m_config;
};

ವರ್ಗ OPENVINO_API MOECompressed : ಸಾರ್ವಜನಿಕ MOE {
ಸಾರ್ವಜನಿಕ:
    OPENVINO_OP("MOECompressed", "", ov::op::internal::MOE)

    MOECompressed() = ಪೂರ್ವನಿಯೋಜಿತ;

    ರಚನೆ Config : ಸಾರ್ವಜನಿಕ MOE::Config {
        size_t hidden_ಗಾತ್ರ = 0;
        size_t inter_ಗಾತ್ರ = 0;
        size_t ಅಂಕೆ_expert = 0;
        size_t ಅಂಕೆ_shared_expert = 0;
        size_t ತುದಿ_k = 0;
        // numeric_limits<size_t>::max() means per_channel compression (single group)
        size_t ಗುಂಪು_ಗಾತ್ರ = 0;
        ತಾರ್ಕಿಕ has_batch_dim = ಸುಳ್ಳು;
        ತಾರ್ಕಿಕ has_zp = ಸುಳ್ಳು;
        ov::element::Type ಹೊರಹೋಗು_type = ov::element::dynamic;
        std::optional<ದಶಮಾಂಶ> scale_ಗುಣಕ;
    };

    MOECompressed(ಸ್ಥಿರಾಂಕ Output_ಸದಿಶ & ವಾದಗಳು_2, ಸ್ಥಿರಾಂಕ Config & config);

    ಸ್ಥಿರಾಂಕ Config & get_config() ಸ್ಥಿರಾಂಕ { ಹಿಂತಿರುಗಿ m_config; }

    ಶೂನ್ಯ ಗಣ_scale_ಗುಣಕ(ದಶಮಾಂಶ scale_ಗುಣಕ) { m_config.scale_ಗುಣಕ = scale_ಗುಣಕ; }

    ತಾರ್ಕಿಕ ಭೇಟಿ_ಗುಣಲಕ್ಷಣಗಳು(AttributeVisitor & visitor) override;
    ಶೂನ್ಯ validate_and_infer_types() override;
    std::shared_ptr<Node> clone_with_new_inputs(ಸ್ಥಿರಾಂಕ Output_ಸದಿಶ & new_ವಾದಗಳು) ಸ್ಥಿರಾಂಕ override;

ಸಂರಕ್ಷಿತ:
    Config m_config;
};

}  // namespace ov::op::internal
