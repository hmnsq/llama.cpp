#pragma once
#অন্তর্ভাৱ "ggml-openvino-extra.h"  // For ExtraQuantType
#অন্তর্ভাৱ "ggml.h"

#অন্তর্ভাৱ <openvino/core/node_output.hpp>
#অন্তর্ভাৱ <openvino/op/constant.hpp>
#অন্তর্ভাৱ <openvino/runtime/tensor.hpp>

স্থির constexpr size_t GGML_QUANTIZATION_GROUP_SIZE = 32;

অন্তঃস্থ ধ্রুৱ ৱর্ণ * extra_quant_type_name(ExtraQuantType t) {
    ৱিকল্প (t) {
    পক্ষ ExtraQuantType::F16:
        ঘূৰাই "F16";
    পক্ষ ExtraQuantType::Q4_0_C:
        ঘূৰাই "Q4_0_C";
    পক্ষ ExtraQuantType::Q4_0_128:
        ঘূৰাই "Q4_0_128";
    পক্ষ ExtraQuantType::Q8_0_C:
        ঘূৰাই "Q8_0_C";
    পক্ষ ExtraQuantType::Q8_0_32:
        ঘূৰাই "Q8_0_32";
    পক্ষ ExtraQuantType::Q8_1_C:
        ঘূৰাই "Q8_1_C";
    পক্ষ ExtraQuantType::Q4_0_64:
        ঘূৰাই "Q4_0_64";
    পক্ষ ExtraQuantType::Q4_1_64:
        ঘূৰাই "Q4_1_64";
    স্ৱভাৱ:
        ঘূৰাই "unknown";
    }
}

// Result from process_weight_tensor containing the weight node and tensors.
// For quantized weights, also contains the extracted layout and scale/zp tensors.
সংরচনা OvWeight {
    std::shared_ptr<ov::গ্ৰন্থি_2> weight_গ্ৰন্থি;
    ggml_openvino_extracted_layout layout;  // Only meaningful for quantized (layout.total_size > 0)
    ov::Tensor weights;
    ov::Tensor scales;
    ov::Tensor zp;

    bool is_quantized() ধ্রুৱ { ঘূৰাই layout.scales_size > 0; }
};

// Process weight tensor and create an OpenVINO weight node
// Handles F16/F32/BF16 and quantized weights, with optional requantization
// If output_base_ptr is nullptr, allocates internal buffers (for decoder use)
// If output_base_ptr is provided, uses pre-allocated buffers at specified offsets (for backend buffer use)
// Returns OvWeight with the weight node and optional quantized tensors
OvWeight process_weight_tensor(
    ধ্রুৱ ggml_tensor * tensor,
    ধ্রুৱ শূন্য * data,                 // Source data pointer (may differ from tensor->data)
    শূন্য * output_base_ptr = nullptr,  // Base pointer for output buffers (or nullptr for internal allocation)
    bool use_bias = false);            // Use an exact f16 zero point (vs. a rounded integer one);
                                       // always used for for_gather_matmul (3D MoE expert) weights
                                       // regardless of this flag, and also settable explicitly for
                                       // test-backend-ops.
