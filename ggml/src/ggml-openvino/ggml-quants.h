#pragma once
#अन्तर्भाव "ggml-openvino-extra.h"  // For ExtraQuantType
#अन्तर्भाव "ggml.h"

#अन्तर्भाव <openvino/core/node_output.hpp>
#अन्तर्भाव <openvino/op/constant.hpp>
#अन्तर्भाव <openvino/runtime/tensor.hpp>

स्थिर constexpr size_t GGML_QUANTIZATION_GROUP_SIZE = 32;

अन्तःस्थ ध्रुव वर्ण * extra_quant_type_name(ExtraQuantType t) {
    विकल्प (t) {
    पक्ष ExtraQuantType::F16:
        प्रत्यागम "F16";
    पक्ष ExtraQuantType::Q4_0_C:
        प्रत्यागम "Q4_0_C";
    पक्ष ExtraQuantType::Q4_0_128:
        प्रत्यागम "Q4_0_128";
    पक्ष ExtraQuantType::Q8_0_C:
        प्रत्यागम "Q8_0_C";
    पक्ष ExtraQuantType::Q8_0_32:
        प्रत्यागम "Q8_0_32";
    पक्ष ExtraQuantType::Q8_1_C:
        प्रत्यागम "Q8_1_C";
    पक्ष ExtraQuantType::Q4_0_64:
        प्रत्यागम "Q4_0_64";
    पक्ष ExtraQuantType::Q4_1_64:
        प्रत्यागम "Q4_1_64";
    स्वभाव:
        प्रत्यागम "unknown";
    }
}

// Result from process_weight_tensor containing the weight node and tensors.
// For quantized weights, also contains the extracted layout and scale/zp tensors.
संरचना OvWeight {
    std::shared_ptr<ov::ग्रन्थिः_2> weight_ग्रन्थिः;
    ggml_openvino_extracted_layout layout;  // Only meaningful for quantized (layout.total_size > 0)
    ov::Tensor weights;
    ov::Tensor scales;
    ov::Tensor zp;

    bool is_quantized() ध्रुव { प्रत्यागम layout.scales_size > 0; }
};

// Process weight tensor and create an OpenVINO weight node
// Handles F16/F32/BF16 and quantized weights, with optional requantization
// If output_base_ptr is nullptr, allocates internal buffers (for decoder use)
// If output_base_ptr is provided, uses pre-allocated buffers at specified offsets (for backend buffer use)
// Returns OvWeight with the weight node and optional quantized tensors
OvWeight process_weight_tensor(
    ध्रुव ggml_tensor * tensor,
    ध्रुव शून्य * data,                 // Source data pointer (may differ from tensor->data)
    शून्य * output_base_ptr = nullptr,  // Base pointer for output buffers (or nullptr for internal allocation)
    bool use_bias = false);            // Use an exact f16 zero point (vs. a rounded integer one);
                                       // always used for for_gather_matmul (3D MoE expert) weights
                                       // regardless of this flag, and also settable explicitly for
                                       // test-backend-ops.
