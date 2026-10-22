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

#ہدایت once

#ضمن <optional>

#ضمن "openvino/core/type/element_type.hpp"
#ضمن "openvino/op/op.hpp"

نطاق ov::op::internal {

صنف OPENVINO_API MOE : عام ov::op::Op {
عام:
    OPENVINO_OP("MOE")

    MOE() = افتراضي;

    MOE(ثابت Output_سمتیہ & دلیلاں_2) : Op(دلیلاں_2) {}

    تعداد صنف Expert_type { GEMM2_BIAS_SWIGLU_CLAMP, GEMM3_SWIGLU };

    تعداد صنف Activation_type { SWIGLU, GEGLU_TANH, GEGLU_ERF };

    بنية Config {
        Expert_type expert_type{ Expert_type::GEMM2_BIAS_SWIGLU_CLAMP };
        عائم expert_alpha{ 0.0f };
        عائم expert_beta{ 1.0f };
        size_t gate_idx{ 0 };
        Activation_type activation_type{ Activation_type::SWIGLU };
    };

    MOE(ثابت Output_سمتیہ & دلیلاں_2, ثابت Config & config);

    ثابت Config & get_config() ثابت;
    فراغ سیٹ_config(ثابت Config & config);

    منطقي ویکھو_گن(AttributeVisitor & visitor) override;
    فراغ validate_and_infer_types() override;
    std::shared_ptr<Node> clone_with_new_inputs(ثابت Output_سمتیہ & new_دلیلاں) ثابت override;

خاص:
    Config m_config;
};

صنف OPENVINO_API MOECompressed : عام MOE {
عام:
    OPENVINO_OP("MOECompressed", "", ov::op::internal::MOE)

    MOECompressed() = افتراضي;

    بنية Config : عام MOE::Config {
        size_t hidden_ناپ = 0;
        size_t inter_ناپ = 0;
        size_t عدد_expert = 0;
        size_t عدد_shared_expert = 0;
        size_t چوٹی_k = 0;
        // numeric_limits<size_t>::max() means per_channel compression (single group)
        size_t گروہ_ناپ = 0;
        منطقي has_batch_dim = خطأ;
        منطقي has_zp = خطأ;
        ov::element::Type باہرلا_type = ov::element::dynamic;
        std::optional<عائم> scale_ضارب;
    };

    MOECompressed(ثابت Output_سمتیہ & دلیلاں_2, ثابت Config & config);

    ثابت Config & get_config() ثابت { أرجع m_config; }

    فراغ سیٹ_scale_ضارب(عائم scale_ضارب) { m_config.scale_ضارب = scale_ضارب; }

    منطقي ویکھو_گن(AttributeVisitor & visitor) override;
    فراغ validate_and_infer_types() override;
    std::shared_ptr<Node> clone_with_new_inputs(ثابت Output_سمتیہ & new_دلیلاں) ثابت override;

محمي:
    Config m_config;
};

}  // namespace ov::op::internal
