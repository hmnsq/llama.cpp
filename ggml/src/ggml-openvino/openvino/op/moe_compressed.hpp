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

#شامل <optional>

#شامل "openvino/core/type/element_type.hpp"
#شامل "openvino/op/op.hpp"

نامخلا ov::op::internal {

جماعت OPENVINO_API MOE : عوامی ov::op::Op {
عوامی:
    OPENVINO_OP("MOE")

    MOE() = طےشدہ;

    MOE(ثابت Output_سمتیہ & دلائل_2) : Op(دلائل_2) {}

    شمار جماعت Expert_type { GEMM2_BIAS_SWIGLU_CLAMP, GEMM3_SWIGLU };

    شمار جماعت Activation_type { SWIGLU, GEGLU_TANH, GEGLU_ERF };

    ساخت Config {
        Expert_type expert_type{ Expert_type::GEMM2_BIAS_SWIGLU_CLAMP };
        اعشاری expert_alpha{ 0.0f };
        اعشاری expert_beta{ 1.0f };
        size_t gate_idx{ 0 };
        Activation_type activation_type{ Activation_type::SWIGLU };
    };

    MOE(ثابت Output_سمتیہ & دلائل_2, ثابت Config & config);

    ثابت Config & get_config() ثابت;
    خالی سیٹ_config(ثابت Config & config);

    منطقی دورہ_اوصاف(AttributeVisitor & visitor) override;
    خالی validate_and_infer_types() override;
    std::shared_ptr<Node> clone_with_new_inputs(ثابت Output_سمتیہ & new_دلائل) ثابت override;

نجی:
    Config m_config;
};

جماعت OPENVINO_API MOECompressed : عوامی MOE {
عوامی:
    OPENVINO_OP("MOECompressed", "", ov::op::internal::MOE)

    MOECompressed() = طےشدہ;

    ساخت Config : عوامی MOE::Config {
        size_t hidden_سائز = 0;
        size_t inter_سائز = 0;
        size_t عدد_expert = 0;
        size_t عدد_shared_expert = 0;
        size_t چوٹی_k = 0;
        // numeric_limits<size_t>::max() means per_channel compression (single group)
        size_t گروہ_سائز = 0;
        منطقی has_batch_dim = جھوٹ;
        منطقی has_zp = جھوٹ;
        ov::element::Type برآمد_type = ov::element::dynamic;
        std::optional<اعشاری> scale_ضربی_عدد;
    };

    MOECompressed(ثابت Output_سمتیہ & دلائل_2, ثابت Config & config);

    ثابت Config & get_config() ثابت { واپس m_config; }

    خالی سیٹ_scale_ضربی_عدد(اعشاری scale_ضربی_عدد) { m_config.scale_ضربی_عدد = scale_ضربی_عدد; }

    منطقی دورہ_اوصاف(AttributeVisitor & visitor) override;
    خالی validate_and_infer_types() override;
    std::shared_ptr<Node> clone_with_new_inputs(ثابت Output_سمتیہ & new_دلائل) ثابت override;

محفوظ:
    Config m_config;
};

}  // namespace ov::op::internal
