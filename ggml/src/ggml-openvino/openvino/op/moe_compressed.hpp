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

#సూచన once

#అన్తర్భావ <optional>

#అన్తర్భావ "openvino/core/type/element_type.hpp"
#అన్తర్భావ "openvino/op/op.hpp"

నామాకాశ ov::op::internal {

తరగతి OPENVINO_API MOE : బహిరంగ ov::op::Op {
బహిరంగ:
    OPENVINO_OP("MOE")

    MOE() = అప్రమేయ;

    MOE(స్థిరరాశి Output_సదిశ & వాదనలు_2) : Op(వాదనలు_2) {}

    గణన తరగతి Expert_type { GEMM2_BIAS_SWIGLU_CLAMP, GEMM3_SWIGLU };

    గణన తరగతి Activation_type { SWIGLU, GEGLU_TANH, GEGLU_ERF };

    నిర్మాణం Config {
        Expert_type expert_type{ Expert_type::GEMM2_BIAS_SWIGLU_CLAMP };
        దశాంశం expert_alpha{ 0.0f };
        దశాంశం expert_beta{ 1.0f };
        size_t gate_idx{ 0 };
        Activation_type activation_type{ Activation_type::SWIGLU };
    };

    MOE(స్థిరరాశి Output_సదిశ & వాదనలు_2, స్థిరరాశి Config & config);

    స్థిరరాశి Config & get_config() స్థిరరాశి;
    శూన్య సమితి_config(స్థిరరాశి Config & config);

    తార్కిక సందర్శించు_లక్షణాలు(AttributeVisitor & visitor) override;
    శూన్య validate_and_infer_types() override;
    std::shared_ptr<Node> clone_with_new_inputs(స్థిరరాశి Output_సదిశ & new_వాదనలు) స్థిరరాశి override;

అంతరంగ:
    Config m_config;
};

తరగతి OPENVINO_API MOECompressed : బహిరంగ MOE {
బహిరంగ:
    OPENVINO_OP("MOECompressed", "", ov::op::internal::MOE)

    MOECompressed() = అప్రమేయ;

    నిర్మాణం Config : బహిరంగ MOE::Config {
        size_t hidden_కొలత = 0;
        size_t inter_కొలత = 0;
        size_t అంకె_expert = 0;
        size_t అంకె_shared_expert = 0;
        size_t శిఖరం_k = 0;
        // numeric_limits<size_t>::max() means per_channel compression (single group)
        size_t సమూహం_కొలత = 0;
        తార్కిక has_batch_dim = అబద్ధం;
        తార్కిక has_zp = అబద్ధం;
        ov::element::Type బహిర్గతం_type = ov::element::dynamic;
        std::optional<దశాంశం> scale_గుణకం;
    };

    MOECompressed(స్థిరరాశి Output_సదిశ & వాదనలు_2, స్థిరరాశి Config & config);

    స్థిరరాశి Config & get_config() స్థిరరాశి { తిరిగి m_config; }

    శూన్య సమితి_scale_గుణకం(దశాంశం scale_గుణకం) { m_config.scale_గుణకం = scale_గుణకం; }

    తార్కిక సందర్శించు_లక్షణాలు(AttributeVisitor & visitor) override;
    శూన్య validate_and_infer_types() override;
    std::shared_ptr<Node> clone_with_new_inputs(స్థిరరాశి Output_సదిశ & new_వాదనలు) స్థిరరాశి override;

రక్షిత:
    Config m_config;
};

}  // namespace ov::op::internal
