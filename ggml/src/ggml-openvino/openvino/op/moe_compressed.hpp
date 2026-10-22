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

#आशय once

#समावेश <optional>

#समावेश "openvino/core/type/element_type.hpp"
#समावेश "openvino/op/op.hpp"

namespace ov::op::internal {

श्रेणी OPENVINO_API MOE : खुला ov::op::Op {
खुला:
    OPENVINO_OP("MOE")

    MOE() = शेष;

    MOE(स्थिर OutputVector & args) : Op(args) {}

    क्रमागत श्रेणी Expert_type { GEMM2_BIAS_SWIGLU_CLAMP, GEMM3_SWIGLU };

    क्रमागत श्रेणी Activation_type { SWIGLU, GEGLU_TANH, GEGLU_ERF };

    काष्ठा Config {
        Expert_type expert_type{ Expert_type::GEMM2_BIAS_SWIGLU_CLAMP };
        भग्न expert_alpha{ 0.0f };
        भग्न expert_beta{ 1.0f };
        माप_प्रकार gate_idx{ 0 };
        Activation_type activation_type{ Activation_type::SWIGLU };
    };

    MOE(स्थिर OutputVector & args, स्थिर Config & config);

    स्थिर Config & get_config() स्थिर;
    व्योम set_config(स्थिर Config & config);

    bool visit_attributes(AttributeVisitor & visitor) override;
    व्योम validate_and_infer_types() override;
    std::shared_ptr<Node> clone_with_new_inputs(स्थिर OutputVector & new_args) स्थिर override;

निजी:
    Config m_config;
};

श्रेणी OPENVINO_API MOECompressed : खुला MOE {
खुला:
    OPENVINO_OP("MOECompressed", "", ov::op::internal::MOE)

    MOECompressed() = शेष;

    काष्ठा Config : खुला MOE::Config {
        माप_प्रकार hidden_आकार = 0;
        माप_प्रकार inter_आकार = 0;
        माप_प्रकार अंक_expert = 0;
        माप_प्रकार अंक_shared_expert = 0;
        माप_प्रकार शीर्ष_k = 0;
        // numeric_limits<size_t>::max() means per_channel compression (single group)
        माप_प्रकार group_आकार = 0;
        bool has_batch_dim = false;
        bool has_zp = false;
        ov::element::Type out_type = ov::element::dynamic;
        std::optional<भग्न> scale_factor;
    };

    MOECompressed(स्थिर OutputVector & args, स्थिर Config & config);

    स्थिर Config & get_config() स्थिर { वापस m_config; }

    व्योम set_scale_factor(भग्न scale_factor) { m_config.scale_factor = scale_factor; }

    bool visit_attributes(AttributeVisitor & visitor) override;
    व्योम validate_and_infer_types() override;
    std::shared_ptr<Node> clone_with_new_inputs(स्थिर OutputVector & new_args) स्थिर override;

रक्षित:
    Config m_config;
};

}  // namespace ov::op::internal
