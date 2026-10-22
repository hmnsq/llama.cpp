#pragma once
#அந்தர்பாவ "ggml-openvino-extra.h"  // For ExtraQuantType
#அந்தர்பாவ "ggml.h"

#அந்தர்பாவ <openvino/core/node_output.hpp>
#அந்தர்பாவ <openvino/op/constant.hpp>
#அந்தர்பாவ <openvino/runtime/tensor.hpp>

நிலையான constexpr size_t GGML_QUANTIZATION_GROUP_SIZE = 32;

உள்வரி மாறிலி எழுத்து * extra_quant_type_name(ExtraQuantType t) {
    தேர்வு (t) {
    நிலை ExtraQuantType::F16:
        திரும்பு "F16";
    நிலை ExtraQuantType::Q4_0_C:
        திரும்பு "Q4_0_C";
    நிலை ExtraQuantType::Q4_0_128:
        திரும்பு "Q4_0_128";
    நிலை ExtraQuantType::Q8_0_C:
        திரும்பு "Q8_0_C";
    நிலை ExtraQuantType::Q8_0_32:
        திரும்பு "Q8_0_32";
    நிலை ExtraQuantType::Q8_1_C:
        திரும்பு "Q8_1_C";
    நிலை ExtraQuantType::Q4_0_64:
        திரும்பு "Q4_0_64";
    நிலை ExtraQuantType::Q4_1_64:
        திரும்பு "Q4_1_64";
    இயல்பு:
        திரும்பு "unknown";
    }
}

// Result from process_weight_tensor containing the weight node and tensors.
// For quantized weights, also contains the extracted layout and scale/zp tensors.
கட்டமைப்பு OvWeight {
    std::shared_ptr<ov::முனை_2> weight_முனை;
    ggml_openvino_extracted_layout layout;  // Only meaningful for quantized (layout.total_size > 0)
    ov::Tensor weights;
    ov::Tensor scales;
    ov::Tensor zp;

    bool is_quantized() மாறிலி { திரும்பு layout.scales_size > 0; }
};

// Process weight tensor and create an OpenVINO weight node
// Handles F16/F32/BF16 and quantized weights, with optional requantization
// If output_base_ptr is nullptr, allocates internal buffers (for decoder use)
// If output_base_ptr is provided, uses pre-allocated buffers at specified offsets (for backend buffer use)
// Returns OvWeight with the weight node and optional quantized tensors
OvWeight process_weight_tensor(
    மாறிலி ggml_tensor * tensor,
    மாறிலி வெறுமை * data,                 // Source data pointer (may differ from tensor->data)
    வெறுமை * output_base_ptr = nullptr,  // Base pointer for output buffers (or nullptr for internal allocation)
    bool use_bias = false);            // Use an exact f16 zero point (vs. a rounded integer one);
                                       // always used for for_gather_matmul (3D MoE expert) weights
                                       // regardless of this flag, and also settable explicitly for
                                       // test-backend-ops.
