#pragma once
#अन्तर्भाव "ggml-openvino-extra.h"  // For ExtraQuantType
#अन्तर्भाव "ggml.h"

#अन्तर्भाव <openvino/core/node_output.hpp>
#अन्तर्भाव <openvino/op/constant.hpp>
#अन्तर्भाव <openvino/runtime/tensor.hpp>

स्थिर constexpr माप_प्रकार GGML_QUANTIZATION_GROUP_SIZE = 32;

ओळीत स्थिरांक अक्षर * extra_quant_type_name(ExtraQuantType t) {
    निवड (t) {
    स्थिती ExtraQuantType::F16:
        परत "F16";
    स्थिती ExtraQuantType::Q4_0_C:
        परत "Q4_0_C";
    स्थिती ExtraQuantType::Q4_0_128:
        परत "Q4_0_128";
    स्थिती ExtraQuantType::Q8_0_C:
        परत "Q8_0_C";
    स्थिती ExtraQuantType::Q8_0_32:
        परत "Q8_0_32";
    स्थिती ExtraQuantType::Q8_1_C:
        परत "Q8_1_C";
    स्थिती ExtraQuantType::Q4_0_64:
        परत "Q4_0_64";
    स्थिती ExtraQuantType::Q4_1_64:
        परत "Q4_1_64";
    पूर्वनिर्धारित:
        परत "unknown";
    }
}

// Result from process_weight_tensor containing the weight node and tensors.
// For quantized weights, also contains the extracted layout and scale/zp tensors.
संरचना OvWeight {
    std::shared_ptr<ov::गाठ_2> weight_गाठ;
    ggml_openvino_extracted_layout layout;  // Only meaningful for quantized (layout.total_size > 0)
    ov::Tensor weights;
    ov::Tensor scales;
    ov::Tensor zp;

    bool is_quantized() स्थिरांक { परत layout.scales_size > 0; }
};

// Process weight tensor and create an OpenVINO weight node
// Handles F16/F32/BF16 and quantized weights, with optional requantization
// If output_base_ptr is nullptr, allocates internal buffers (for decoder use)
// If output_base_ptr is provided, uses pre-allocated buffers at specified offsets (for backend buffer use)
// Returns OvWeight with the weight node and optional quantized tensors
OvWeight process_weight_tensor(
    स्थिरांक ggml_tensor * tensor,
    स्थिरांक शून्य * data,                 // Source data pointer (may differ from tensor->data)
    शून्य * output_base_ptr = nullptr,  // Base pointer for output buffers (or nullptr for internal allocation)
    bool use_bias = false);            // Use an exact f16 zero point (vs. a rounded integer one);
                                       // always used for for_gather_matmul (3D MoE expert) weights
                                       // regardless of this flag, and also settable explicitly for
                                       // test-backend-ops.
