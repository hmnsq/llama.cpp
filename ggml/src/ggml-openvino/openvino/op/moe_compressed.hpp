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

#അന്തര്ഭാവ <optional>

#അന്തര്ഭാവ "openvino/core/type/element_type.hpp"
#അന്തര്ഭാവ "openvino/op/op.hpp"

നാമാകാശ ov::op::internal {

വർഗ്ഗം OPENVINO_API MOE : പൊതു ov::op::Op {
പൊതു:
    OPENVINO_OP("MOE")

    MOE() = സ്ഥിരസ്ഥിതി;

    MOE(സ്ഥിരം OutputVector & args) : Op(args) {}

    എണ്ണൽ വർഗ്ഗം Expert_type { GEMM2_BIAS_SWIGLU_CLAMP, GEMM3_SWIGLU };

    എണ്ണൽ വർഗ്ഗം Activation_type { SWIGLU, GEGLU_TANH, GEGLU_ERF };

    ഘടന Config {
        Expert_type expert_type{ Expert_type::GEMM2_BIAS_SWIGLU_CLAMP };
        ദശാംശം expert_alpha{ 0.0f };
        ദശാംശം expert_beta{ 1.0f };
        size_t gate_idx{ 0 };
        Activation_type activation_type{ Activation_type::SWIGLU };
    };

    MOE(സ്ഥിരം OutputVector & args, സ്ഥിരം Config & config);

    സ്ഥിരം Config & get_config() സ്ഥിരം;
    ശൂന്യം set_config(സ്ഥിരം Config & config);

    തർക്കം visit_attributes(AttributeVisitor & visitor) override;
    ശൂന്യം validate_and_infer_types() override;
    std::shared_ptr<Node> clone_with_new_inputs(സ്ഥിരം OutputVector & new_args) സ്ഥിരം override;

സ്വകാര്യ:
    Config m_config;
};

വർഗ്ഗം OPENVINO_API MOECompressed : പൊതു MOE {
പൊതു:
    OPENVINO_OP("MOECompressed", "", ov::op::internal::MOE)

    MOECompressed() = സ്ഥിരസ്ഥിതി;

    ഘടന Config : പൊതു MOE::Config {
        size_t hidden_size = 0;
        size_t inter_size = 0;
        size_t അക്കം_expert = 0;
        size_t അക്കം_shared_expert = 0;
        size_t മുകൾ_k = 0;
        // numeric_limits<size_t>::max() means per_channel compression (single group)
        size_t group_size = 0;
        തർക്കം has_batch_dim = അസത്യം;
        തർക്കം has_zp = അസത്യം;
        ov::element::Type out_type = ov::element::dynamic;
        std::optional<ദശാംശം> scale_factor;
    };

    MOECompressed(സ്ഥിരം OutputVector & args, സ്ഥിരം Config & config);

    സ്ഥിരം Config & get_config() സ്ഥിരം { തിരികെ m_config; }

    ശൂന്യം set_scale_factor(ദശാംശം scale_factor) { m_config.scale_factor = scale_factor; }

    തർക്കം visit_attributes(AttributeVisitor & visitor) override;
    ശൂന്യം validate_and_infer_types() override;
    std::shared_ptr<Node> clone_with_new_inputs(സ്ഥിരം OutputVector & new_args) സ്ഥിരം override;

സംരക്ഷിത:
    Config m_config;
};

}  // namespace ov::op::internal
