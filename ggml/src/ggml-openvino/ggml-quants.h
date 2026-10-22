#pragma once
#ಅನ್ತರ್ಭಾವ "ggml-openvino-extra.h"  // For ExtraQuantType
#ಅನ್ತರ್ಭಾವ "ggml.h"

#ಅನ್ತರ್ಭಾವ <openvino/core/node_output.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/constant.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/runtime/tensor.hpp>

ಸ್ಥಿರ constexpr size_t GGML_QUANTIZATION_GROUP_SIZE = 32;

ಅನ್ತಃಸ್ಥ ಧ್ರುವ ವರ್ಣ * extra_quant_type_name(ExtraQuantType t) {
    ವಿಕಲ್ಪ (t) {
    ಪಕ್ಷ ExtraQuantType::F16:
        ಪ್ರತ್ಯಾಗಮ "F16";
    ಪಕ್ಷ ExtraQuantType::Q4_0_C:
        ಪ್ರತ್ಯಾಗಮ "Q4_0_C";
    ಪಕ್ಷ ExtraQuantType::Q4_0_128:
        ಪ್ರತ್ಯಾಗಮ "Q4_0_128";
    ಪಕ್ಷ ExtraQuantType::Q8_0_C:
        ಪ್ರತ್ಯಾಗಮ "Q8_0_C";
    ಪಕ್ಷ ExtraQuantType::Q8_0_32:
        ಪ್ರತ್ಯಾಗಮ "Q8_0_32";
    ಪಕ್ಷ ExtraQuantType::Q8_1_C:
        ಪ್ರತ್ಯಾಗಮ "Q8_1_C";
    ಪಕ್ಷ ExtraQuantType::Q4_0_64:
        ಪ್ರತ್ಯಾಗಮ "Q4_0_64";
    ಪಕ್ಷ ExtraQuantType::Q4_1_64:
        ಪ್ರತ್ಯಾಗಮ "Q4_1_64";
    ಸ್ವಭಾವ:
        ಪ್ರತ್ಯಾಗಮ "unknown";
    }
}

// Result from process_weight_tensor containing the weight node and tensors.
// For quantized weights, also contains the extracted layout and scale/zp tensors.
ಸಂರಚನಾ OvWeight {
    std::shared_ptr<ov::ಗಂಟು_2> weight_ಗಂಟು;
    ggml_openvino_extracted_layout layout;  // Only meaningful for quantized (layout.total_size > 0)
    ov::Tensor weights;
    ov::Tensor scales;
    ov::Tensor zp;

    bool is_quantized() ಧ್ರುವ { ಪ್ರತ್ಯಾಗಮ layout.scales_size > 0; }
};

// Process weight tensor and create an OpenVINO weight node
// Handles F16/F32/BF16 and quantized weights, with optional requantization
// If output_base_ptr is nullptr, allocates internal buffers (for decoder use)
// If output_base_ptr is provided, uses pre-allocated buffers at specified offsets (for backend buffer use)
// Returns OvWeight with the weight node and optional quantized tensors
OvWeight process_weight_tensor(
    ಧ್ರುವ ggml_tensor * tensor,
    ಧ್ರುವ ಶೂನ್ಯ * data,                 // Source data pointer (may differ from tensor->data)
    ಶೂನ್ಯ * output_base_ptr = nullptr,  // Base pointer for output buffers (or nullptr for internal allocation)
    bool use_bias = false);            // Use an exact f16 zero point (vs. a rounded integer one);
                                       // always used for for_gather_matmul (3D MoE expert) weights
                                       // regardless of this flag, and also settable explicitly for
                                       // test-backend-ops.
