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

#ଅନ୍ତର୍ଭାଵ <optional>

#ଅନ୍ତର୍ଭାଵ "openvino/core/type/element_type.hpp"
#ଅନ୍ତର୍ଭାଵ "openvino/op/op.hpp"

ନାମାକାଶ ov::op::internal {

ଶ୍ରେଣୀ OPENVINO_API MOE : ସର୍ବସାଧାରଣ ov::op::Op {
ସର୍ବସାଧାରଣ:
    OPENVINO_OP("MOE")

    MOE() = ପୂର୍ବନିର୍ଦ୍ଧାରିତ;

    MOE(ଅଚଳ OutputVector & args) : Op(args) {}

    ଗଣନା ଶ୍ରେଣୀ Expert_type { GEMM2_BIAS_SWIGLU_CLAMP, GEMM3_SWIGLU };

    ଗଣନା ଶ୍ରେଣୀ Activation_type { SWIGLU, GEGLU_TANH, GEGLU_ERF };

    ସଂରଚନା Config {
        Expert_type expert_type{ Expert_type::GEMM2_BIAS_SWIGLU_CLAMP };
        ଦଶମିକ expert_alpha{ 0.0f };
        ଦଶମିକ expert_beta{ 1.0f };
        size_t gate_idx{ 0 };
        Activation_type activation_type{ Activation_type::SWIGLU };
    };

    MOE(ଅଚଳ OutputVector & args, ଅଚଳ Config & config);

    ଅଚଳ Config & get_config() ଅଚଳ;
    ଶୂନ୍ୟ set_config(ଅଚଳ Config & config);

    ତାର୍କିକ visit_attributes(AttributeVisitor & visitor) override;
    ଶୂନ୍ୟ validate_and_infer_types() override;
    std::shared_ptr<Node> clone_with_new_inputs(ଅଚଳ OutputVector & new_args) ଅଚଳ override;

ବ୍ୟକ୍ତିଗତ:
    Config m_config;
};

ଶ୍ରେଣୀ OPENVINO_API MOECompressed : ସର୍ବସାଧାରଣ MOE {
ସର୍ବସାଧାରଣ:
    OPENVINO_OP("MOECompressed", "", ov::op::internal::MOE)

    MOECompressed() = ପୂର୍ବନିର୍ଦ୍ଧାରିତ;

    ସଂରଚନା Config : ସର୍ବସାଧାରଣ MOE::Config {
        size_t hidden_size = 0;
        size_t inter_size = 0;
        size_t ଅଙ୍କ_expert = 0;
        size_t ଅଙ୍କ_shared_expert = 0;
        size_t ଶୀର୍ଷ_k = 0;
        // numeric_limits<size_t>::max() means per_channel compression (single group)
        size_t group_size = 0;
        ତାର୍କିକ has_batch_dim = ମିଥ୍ୟା;
        ତାର୍କିକ has_zp = ମିଥ୍ୟା;
        ov::element::Type out_type = ov::element::dynamic;
        std::optional<ଦଶମିକ> scale_factor;
    };

    MOECompressed(ଅଚଳ OutputVector & args, ଅଚଳ Config & config);

    ଅଚଳ Config & get_config() ଅଚଳ { ଫେରନ୍ତୁ m_config; }

    ଶୂନ୍ୟ set_scale_factor(ଦଶମିକ scale_factor) { m_config.scale_factor = scale_factor; }

    ତାର୍କିକ visit_attributes(AttributeVisitor & visitor) override;
    ଶୂନ୍ୟ validate_and_infer_types() override;
    std::shared_ptr<Node> clone_with_new_inputs(ଅଚଳ OutputVector & new_args) ଅଚଳ override;

ସୁରକ୍ଷିତ:
    Config m_config;
};

}  // namespace ov::op::internal
