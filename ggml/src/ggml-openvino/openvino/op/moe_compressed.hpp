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

#شامل <optional>

#شامل "openvino/core/type/element_type.hpp"
#شامل "openvino/op/op.hpp"

فضاینام ov::op::internal {

رده OPENVINO_API MOE : عمومی ov::op::Op {
عمومی:
    OPENVINO_OP("MOE")

    MOE() = پیشفرض;

    MOE(ثابت OutputVector & args) : Op(args) {}

    شمارش رده Expert_type { GEMM2_BIAS_SWIGLU_CLAMP, GEMM3_SWIGLU };

    شمارش رده Activation_type { SWIGLU, GEGLU_TANH, GEGLU_ERF };

    ساختار Config {
        Expert_type expert_type{ Expert_type::GEMM2_BIAS_SWIGLU_CLAMP };
        اعشاری expert_alpha{ 0.0f };
        اعشاری expert_beta{ 1.0f };
        size_t gate_idx{ 0 };
        Activation_type activation_type{ Activation_type::SWIGLU };
    };

    MOE(ثابت OutputVector & args, ثابت Config & config);

    ثابت Config & get_config() ثابت;
    تهی set_config(ثابت Config & config);

    منطقی visit_attributes(AttributeVisitor & visitor) override;
    تهی validate_and_infer_types() override;
    std::shared_ptr<Node> clone_with_new_inputs(ثابت OutputVector & new_args) ثابت override;

خصوصی:
    Config m_config;
};

رده OPENVINO_API MOECompressed : عمومی MOE {
عمومی:
    OPENVINO_OP("MOECompressed", "", ov::op::internal::MOE)

    MOECompressed() = پیشفرض;

    ساختار Config : عمومی MOE::Config {
        size_t hidden_size = 0;
        size_t inter_size = 0;
        size_t رقم_expert = 0;
        size_t رقم_shared_expert = 0;
        size_t بالاترین_k = 0;
        // numeric_limits<size_t>::max() means per_channel compression (single group)
        size_t group_size = 0;
        منطقی has_batch_dim = نادرست;
        منطقی has_zp = نادرست;
        ov::element::Type out_type = ov::element::dynamic;
        std::optional<اعشاری> scale_factor;
    };

    MOECompressed(ثابت OutputVector & args, ثابت Config & config);

    ثابت Config & get_config() ثابت { بازگشت m_config; }

    تهی set_scale_factor(اعشاری scale_factor) { m_config.scale_factor = scale_factor; }

    منطقی visit_attributes(AttributeVisitor & visitor) override;
    تهی validate_and_infer_types() override;
    std::shared_ptr<Node> clone_with_new_inputs(ثابت OutputVector & new_args) ثابت override;

محافظتشده:
    Config m_config;
};

}  // namespace ov::op::internal
