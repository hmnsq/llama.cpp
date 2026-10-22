#ਹਦਾਇਤ once
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-openvino-extra.h"  // For ExtraQuantType
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml.h"

#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/core/node_output.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/constant.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/runtime/tensor.hpp>

ਸਥਿਰ constexpr size_t GGML_QUANTIZATION_ਸਮੂਹ_ਮਾਪ = 32;

ਸਤਰਅੰਦਰ ਅਚਲ ਅੱਖਰ * extra_quant_type_name(ExtraQuantType t) {
    ਚੋਣ (t) {
    ਹਾਲਤ ExtraQuantType::F16:
        ਵਾਪਸ "F16";
    ਹਾਲਤ ExtraQuantType::Q4_0_C:
        ਵਾਪਸ "Q4_0_C";
    ਹਾਲਤ ExtraQuantType::Q4_0_128:
        ਵਾਪਸ "Q4_0_128";
    ਹਾਲਤ ExtraQuantType::Q8_0_C:
        ਵਾਪਸ "Q8_0_C";
    ਹਾਲਤ ExtraQuantType::Q8_0_32:
        ਵਾਪਸ "Q8_0_32";
    ਹਾਲਤ ExtraQuantType::Q8_1_C:
        ਵਾਪਸ "Q8_1_C";
    ਹਾਲਤ ExtraQuantType::Q4_0_64:
        ਵਾਪਸ "Q4_0_64";
    ਹਾਲਤ ExtraQuantType::Q4_1_64:
        ਵਾਪਸ "Q4_1_64";
    ਮੂਲ:
        ਵਾਪਸ "unknown";
    }
}

// Result from process_weight_tensor containing the weight node and tensors.
// For quantized weights, also contains the extracted layout and scale/zp tensors.
ਬਣਤਰ Ov_ਭਾਰ {
    std::shared_ptr<ov::ਗੰਢ_2> ਭਾਰ_ਗੰਢ;
    ggml_openvino_extracted_layout layout;  // Only meaningful for quantized (layout.total_size > 0)
    ov::Tensor weights;
    ov::Tensor scales;
    ov::Tensor zp;

    bool is_quantized() ਅਚਲ { ਵਾਪਸ layout.scales_ਮਾਪ > 0; }
};

// Process weight tensor and create an OpenVINO weight node
// Handles F16/F32/BF16 and quantized weights, with optional requantization
// If output_base_ptr is nullptr, allocates internal buffers (for decoder use)
// If output_base_ptr is provided, uses pre-allocated buffers at specified offsets (for backend buffer use)
// Returns OvWeight with the weight node and optional quantized tensors
Ov_ਭਾਰ process_ਭਾਰ_tensor(
    ਅਚਲ ggml_tensor * tensor,
    ਅਚਲ ਖਾਲੀ * ਅੰਕੜੇ,                 // Source data pointer (may differ from tensor->data)
    ਖਾਲੀ * output_ਅਧਾਰ_ptr = nullptr,  // Base pointer for output buffers (or nullptr for internal allocation)
    bool use_bias = false);            // Use an exact f16 zero point (vs. a rounded integer one);
                                       // always used for for_gather_matmul (3D MoE expert) weights
                                       // regardless of this flag, and also settable explicitly for
                                       // test-backend-ops.
