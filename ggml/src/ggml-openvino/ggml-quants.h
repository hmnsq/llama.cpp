#pragma once
#അന്തര്ഭാവ "ggml-openvino-extra.h"  // For ExtraQuantType
#അന്തര്ഭാവ "ggml.h"

#അന്തര്ഭാവ <openvino/core/node_output.hpp>
#അന്തര്ഭാവ <openvino/op/constant.hpp>
#അന്തര്ഭാവ <openvino/runtime/tensor.hpp>

സ്ഥിതം constexpr size_t GGML_QUANTIZATION_GROUP_SIZE = 32;

വരിയിൽ സ്ഥിരം അക്ഷരം * extra_quant_type_name(ExtraQuantType t) {
    തിരഞ്ഞെടുപ്പ് (t) {
    അവസ്ഥ ExtraQuantType::F16:
        തിരികെ "F16";
    അവസ്ഥ ExtraQuantType::Q4_0_C:
        തിരികെ "Q4_0_C";
    അവസ്ഥ ExtraQuantType::Q4_0_128:
        തിരികെ "Q4_0_128";
    അവസ്ഥ ExtraQuantType::Q8_0_C:
        തിരികെ "Q8_0_C";
    അവസ്ഥ ExtraQuantType::Q8_0_32:
        തിരികെ "Q8_0_32";
    അവസ്ഥ ExtraQuantType::Q8_1_C:
        തിരികെ "Q8_1_C";
    അവസ്ഥ ExtraQuantType::Q4_0_64:
        തിരികെ "Q4_0_64";
    അവസ്ഥ ExtraQuantType::Q4_1_64:
        തിരികെ "Q4_1_64";
    സ്ഥിരസ്ഥിതി:
        തിരികെ "unknown";
    }
}

// Result from process_weight_tensor containing the weight node and tensors.
// For quantized weights, also contains the extracted layout and scale/zp tensors.
ഘടന OvWeight {
    std::shared_ptr<ov::കണ്ണി_2> weight_കണ്ണി;
    ggml_openvino_extracted_layout layout;  // Only meaningful for quantized (layout.total_size > 0)
    ov::Tensor weights;
    ov::Tensor scales;
    ov::Tensor zp;

    bool is_quantized() സ്ഥിരം { തിരികെ layout.scales_size > 0; }
};

// Process weight tensor and create an OpenVINO weight node
// Handles F16/F32/BF16 and quantized weights, with optional requantization
// If output_base_ptr is nullptr, allocates internal buffers (for decoder use)
// If output_base_ptr is provided, uses pre-allocated buffers at specified offsets (for backend buffer use)
// Returns OvWeight with the weight node and optional quantized tensors
OvWeight process_weight_tensor(
    സ്ഥിരം ggml_tensor * tensor,
    സ്ഥിരം ശൂന്യം * data,                 // Source data pointer (may differ from tensor->data)
    ശൂന്യം * output_base_ptr = nullptr,  // Base pointer for output buffers (or nullptr for internal allocation)
    bool use_bias = false);            // Use an exact f16 zero point (vs. a rounded integer one);
                                       // always used for for_gather_matmul (3D MoE expert) weights
                                       // regardless of this flag, and also settable explicitly for
                                       // test-backend-ops.
