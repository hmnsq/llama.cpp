#आशय once
#समावेश "ggml-openvino-extra.h"  // For ExtraQuantType
#समावेश "ggml.h"

#समावेश <openvino/core/node_output.hpp>
#समावेश <openvino/op/constant.hpp>
#समावेश <openvino/runtime/tensor.hpp>

अटल constexpr माप_प्रकार GGML_QUANTIZATION_GROUP_आकार = 32;

अंतरभूत स्थिर अक्षर * extra_quant_type_name(ExtraQuantType t) {
    चयन (t) {
    हाल ExtraQuantType::F16:
        वापस "F16";
    हाल ExtraQuantType::Q4_0_C:
        वापस "Q4_0_C";
    हाल ExtraQuantType::Q4_0_128:
        वापस "Q4_0_128";
    हाल ExtraQuantType::Q8_0_C:
        वापस "Q8_0_C";
    हाल ExtraQuantType::Q8_0_32:
        वापस "Q8_0_32";
    हाल ExtraQuantType::Q8_1_C:
        वापस "Q8_1_C";
    हाल ExtraQuantType::Q4_0_64:
        वापस "Q4_0_64";
    हाल ExtraQuantType::Q4_1_64:
        वापस "Q4_1_64";
    शेष:
        वापस "unknown";
    }
}

// Result from process_weight_tensor containing the weight node and tensors.
// For quantized weights, also contains the extracted layout and scale/zp tensors.
काष्ठा OvWeight {
    std::shared_ptr<ov::गाँठ_2> weight_गाँठ;
    ggml_openvino_extracted_layout layout;  // Only meaningful for quantized (layout.total_size > 0)
    ov::Tensor weights;
    ov::Tensor scales;
    ov::Tensor zp;

    bool is_quantized() स्थिर { वापस layout.scales_आकार > 0; }
};

// Process weight tensor and create an OpenVINO weight node
// Handles F16/F32/BF16 and quantized weights, with optional requantization
// If output_base_ptr is nullptr, allocates internal buffers (for decoder use)
// If output_base_ptr is provided, uses pre-allocated buffers at specified offsets (for backend buffer use)
// Returns OvWeight with the weight node and optional quantized tensors
OvWeight process_weight_tensor(
    स्थिर ggml_tensor * tensor,
    स्थिर व्योम * तथ्य,                 // Source data pointer (may differ from tensor->data)
    व्योम * output_base_ptr = nullptr,  // Base pointer for output buffers (or nullptr for internal allocation)
    bool use_bias = false);            // Use an exact f16 zero point (vs. a rounded integer one);
                                       // always used for for_gather_matmul (3D MoE expert) weights
                                       // regardless of this flag, and also settable explicitly for
                                       // test-backend-ops.
