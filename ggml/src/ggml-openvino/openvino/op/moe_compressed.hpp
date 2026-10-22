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

#ਹਦਾਇਤ once

#ਅਨ੍ਤਰ੍ਭਾਵ <optional>

#ਅਨ੍ਤਰ੍ਭਾਵ "openvino/core/type/element_type.hpp"
#ਅਨ੍ਤਰ੍ਭਾਵ "openvino/op/op.hpp"

ਨਾਮਾਕਾਸ਼ ov::op::internal {

ਵਰਗ OPENVINO_API MOE : ਜਨਤਕ ov::op::Op {
ਜਨਤਕ:
    OPENVINO_OP("MOE")

    MOE() = ਮੂਲ;

    MOE(ਅਚਲ Output_ਸਦਿਸ਼ & ਦਲੀਲਾਂ_2) : Op(ਦਲੀਲਾਂ_2) {}

    ਗਿਣਤੀ ਵਰਗ Expert_type { GEMM2_BIAS_SWIGLU_CLAMP, GEMM3_SWIGLU };

    ਗਿਣਤੀ ਵਰਗ Activation_type { SWIGLU, GEGLU_TANH, GEGLU_ERF };

    ਬਣਤਰ Config {
        Expert_type expert_type{ Expert_type::GEMM2_BIAS_SWIGLU_CLAMP };
        ਦਸ਼ਮਲਵ expert_alpha{ 0.0f };
        ਦਸ਼ਮਲਵ expert_beta{ 1.0f };
        size_t gate_idx{ 0 };
        Activation_type activation_type{ Activation_type::SWIGLU };
    };

    MOE(ਅਚਲ Output_ਸਦਿਸ਼ & ਦਲੀਲਾਂ_2, ਅਚਲ Config & config);

    ਅਚਲ Config & get_config() ਅਚਲ;
    ਖਾਲੀ ਸੈੱਟ_config(ਅਚਲ Config & config);

    ਤਾਰਕਿਕ ਦੇਖੋ_ਗੁਣ(AttributeVisitor & visitor) override;
    ਖਾਲੀ validate_and_infer_types() override;
    std::shared_ptr<Node> clone_with_new_inputs(ਅਚਲ Output_ਸਦਿਸ਼ & new_ਦਲੀਲਾਂ) ਅਚਲ override;

ਨਿਜੀ:
    Config m_config;
};

ਵਰਗ OPENVINO_API MOECompressed : ਜਨਤਕ MOE {
ਜਨਤਕ:
    OPENVINO_OP("MOECompressed", "", ov::op::internal::MOE)

    MOECompressed() = ਮੂਲ;

    ਬਣਤਰ Config : ਜਨਤਕ MOE::Config {
        size_t hidden_ਮਾਪ = 0;
        size_t inter_ਮਾਪ = 0;
        size_t ਅੰਕ_expert = 0;
        size_t ਅੰਕ_shared_expert = 0;
        size_t ਸਿਖਰ_k = 0;
        // numeric_limits<size_t>::max() means per_channel compression (single group)
        size_t ਸਮੂਹ_ਮਾਪ = 0;
        ਤਾਰਕਿਕ has_batch_dim = ਝੂਠ;
        ਤਾਰਕਿਕ has_zp = ਝੂਠ;
        ov::element::Type ਬਾਹਰਲਾ_type = ov::element::dynamic;
        std::optional<ਦਸ਼ਮਲਵ> scale_ਗੁਣਕ;
    };

    MOECompressed(ਅਚਲ Output_ਸਦਿਸ਼ & ਦਲੀਲਾਂ_2, ਅਚਲ Config & config);

    ਅਚਲ Config & get_config() ਅਚਲ { ਵਾਪਸ m_config; }

    ਖਾਲੀ ਸੈੱਟ_scale_ਗੁਣਕ(ਦਸ਼ਮਲਵ scale_ਗੁਣਕ) { m_config.scale_ਗੁਣਕ = scale_ਗੁਣਕ; }

    ਤਾਰਕਿਕ ਦੇਖੋ_ਗੁਣ(AttributeVisitor & visitor) override;
    ਖਾਲੀ validate_and_infer_types() override;
    std::shared_ptr<Node> clone_with_new_inputs(ਅਚਲ Output_ਸਦਿਸ਼ & new_ਦਲੀਲਾਂ) ਅਚਲ override;

ਸੁਰੱਖਿਅਤ:
    Config m_config;
};

}  // namespace ov::op::internal
