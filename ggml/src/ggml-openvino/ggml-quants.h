#ಸೂಚನೆ once
#ಅನ್ತರ್ಭಾವ "ggml-openvino-extra.h"  // For ExtraQuantType
#ಅನ್ತರ್ಭಾವ "ggml.h"

#ಅನ್ತರ್ಭಾವ <openvino/core/node_output.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/constant.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/runtime/tensor.hpp>

ಸ್ಥಿರ constexpr size_t GGML_QUANTIZATION_ಗುಂಪು_ಗಾತ್ರ = 32;

ಸಾಲೊಳಗೆ ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * extra_quant_type_name(ExtraQuantType t) {
    ಆಯ್ಕೆ (t) {
    ಸ್ಥಿತಿ ExtraQuantType::F16:
        ಹಿಂತಿರುಗಿ "F16";
    ಸ್ಥಿತಿ ExtraQuantType::Q4_0_C:
        ಹಿಂತಿರುಗಿ "Q4_0_C";
    ಸ್ಥಿತಿ ExtraQuantType::Q4_0_128:
        ಹಿಂತಿರುಗಿ "Q4_0_128";
    ಸ್ಥಿತಿ ExtraQuantType::Q8_0_C:
        ಹಿಂತಿರುಗಿ "Q8_0_C";
    ಸ್ಥಿತಿ ExtraQuantType::Q8_0_32:
        ಹಿಂತಿರುಗಿ "Q8_0_32";
    ಸ್ಥಿತಿ ExtraQuantType::Q8_1_C:
        ಹಿಂತಿರುಗಿ "Q8_1_C";
    ಸ್ಥಿತಿ ExtraQuantType::Q4_0_64:
        ಹಿಂತಿರುಗಿ "Q4_0_64";
    ಸ್ಥಿತಿ ExtraQuantType::Q4_1_64:
        ಹಿಂತಿರುಗಿ "Q4_1_64";
    ಪೂರ್ವನಿಯೋಜಿತ:
        ಹಿಂತಿರುಗಿ "unknown";
    }
}

// Result from process_weight_tensor containing the weight node and tensors.
// For quantized weights, also contains the extracted layout and scale/zp tensors.
ರಚನೆ Ov_ತೂಕ {
    std::shared_ptr<ov::ಗಂಟು_2> ತೂಕ_ಗಂಟು;
    ggml_openvino_extracted_layout layout;  // Only meaningful for quantized (layout.total_size > 0)
    ov::Tensor weights;
    ov::Tensor scales;
    ov::Tensor zp;

    bool is_quantized() ಸ್ಥಿರಾಂಕ { ಹಿಂತಿರುಗಿ layout.scales_ಗಾತ್ರ > 0; }
};

// Process weight tensor and create an OpenVINO weight node
// Handles F16/F32/BF16 and quantized weights, with optional requantization
// If output_base_ptr is nullptr, allocates internal buffers (for decoder use)
// If output_base_ptr is provided, uses pre-allocated buffers at specified offsets (for backend buffer use)
// Returns OvWeight with the weight node and optional quantized tensors
Ov_ತೂಕ process_ತೂಕ_tensor(
    ಸ್ಥಿರಾಂಕ ggml_tensor * tensor,
    ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * ದತ್ತಾಂಶ,                 // Source data pointer (may differ from tensor->data)
    ಶೂನ್ಯ * output_ತಳ_ptr = nullptr,  // Base pointer for output buffers (or nullptr for internal allocation)
    bool use_bias = false);            // Use an exact f16 zero point (vs. a rounded integer one);
                                       // always used for for_gather_matmul (3D MoE expert) weights
                                       // regardless of this flag, and also settable explicitly for
                                       // test-backend-ops.
