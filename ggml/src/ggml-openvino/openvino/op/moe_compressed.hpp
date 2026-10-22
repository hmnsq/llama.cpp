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

#અન્તર્ભાવ <optional>

#અન્તર્ભાવ "openvino/core/type/element_type.hpp"
#અન્તર્ભાવ "openvino/op/op.hpp"

નામાકાશ ov::op::internal {

વર્ગ OPENVINO_API MOE : જાહેર ov::op::Op {
જાહેર:
    OPENVINO_OP("MOE")

    MOE() = મૂળભૂત;

    MOE(અચલ OutputVector & args) : Op(args) {}

    ગણના વર્ગ Expert_type { GEMM2_BIAS_SWIGLU_CLAMP, GEMM3_SWIGLU };

    ગણના વર્ગ Activation_type { SWIGLU, GEGLU_TANH, GEGLU_ERF };

    સંરચના Config {
        Expert_type expert_type{ Expert_type::GEMM2_BIAS_SWIGLU_CLAMP };
        દશાંશ expert_alpha{ 0.0f };
        દશાંશ expert_beta{ 1.0f };
        size_t gate_idx{ 0 };
        Activation_type activation_type{ Activation_type::SWIGLU };
    };

    MOE(અચલ OutputVector & args, અચલ Config & config);

    અચલ Config & get_config() અચલ;
    શૂન્ય set_config(અચલ Config & config);

    તાર્કિક visit_attributes(AttributeVisitor & visitor) override;
    શૂન્ય validate_and_infer_types() override;
    std::shared_ptr<Node> clone_with_new_inputs(અચલ OutputVector & new_args) અચલ override;

ખાનગી:
    Config m_config;
};

વર્ગ OPENVINO_API MOECompressed : જાહેર MOE {
જાહેર:
    OPENVINO_OP("MOECompressed", "", ov::op::internal::MOE)

    MOECompressed() = મૂળભૂત;

    સંરચના Config : જાહેર MOE::Config {
        size_t hidden_size = 0;
        size_t inter_size = 0;
        size_t અંક_expert = 0;
        size_t અંક_shared_expert = 0;
        size_t ટોચ_k = 0;
        // numeric_limits<size_t>::max() means per_channel compression (single group)
        size_t group_size = 0;
        તાર્કિક has_batch_dim = ખોટું;
        તાર્કિક has_zp = ખોટું;
        ov::element::Type out_type = ov::element::dynamic;
        std::optional<દશાંશ> scale_factor;
    };

    MOECompressed(અચલ OutputVector & args, અચલ Config & config);

    અચલ Config & get_config() અચલ { પાછા m_config; }

    શૂન્ય set_scale_factor(દશાંશ scale_factor) { m_config.scale_factor = scale_factor; }

    તાર્કિક visit_attributes(AttributeVisitor & visitor) override;
    શૂન્ય validate_and_infer_types() override;
    std::shared_ptr<Node> clone_with_new_inputs(અચલ OutputVector & new_args) અચલ override;

સુરક્ષિત:
    Config m_config;
};

}  // namespace ov::op::internal
