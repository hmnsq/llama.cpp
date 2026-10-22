#pragma once
#شامل "ggml-openvino-extra.h"  // For ExtraQuantType
#شامل "ggml.h"

#شامل <openvino/core/node_output.hpp>
#شامل <openvino/op/constant.hpp>
#شامل <openvino/runtime/tensor.hpp>

ایستا constexpr size_t GGML_QUANTIZATION_GROUP_SIZE = 32;

درونخطی ثابت نویسه * extra_quant_type_name(ExtraQuantType t) {
    انتخاب (t) {
    حالت ExtraQuantType::F16:
        بازگشت "F16";
    حالت ExtraQuantType::Q4_0_C:
        بازگشت "Q4_0_C";
    حالت ExtraQuantType::Q4_0_128:
        بازگشت "Q4_0_128";
    حالت ExtraQuantType::Q8_0_C:
        بازگشت "Q8_0_C";
    حالت ExtraQuantType::Q8_0_32:
        بازگشت "Q8_0_32";
    حالت ExtraQuantType::Q8_1_C:
        بازگشت "Q8_1_C";
    حالت ExtraQuantType::Q4_0_64:
        بازگشت "Q4_0_64";
    حالت ExtraQuantType::Q4_1_64:
        بازگشت "Q4_1_64";
    پیشفرض:
        بازگشت "unknown";
    }
}

// Result from process_weight_tensor containing the weight node and tensors.
// For quantized weights, also contains the extracted layout and scale/zp tensors.
ساختار OvWeight {
    std::shared_ptr<ov::گره_2> weight_گره;
    ggml_openvino_extracted_layout layout;  // Only meaningful for quantized (layout.total_size > 0)
    ov::Tensor weights;
    ov::Tensor scales;
    ov::Tensor zp;

    bool is_quantized() ثابت { بازگشت layout.scales_size > 0; }
};

// Process weight tensor and create an OpenVINO weight node
// Handles F16/F32/BF16 and quantized weights, with optional requantization
// If output_base_ptr is nullptr, allocates internal buffers (for decoder use)
// If output_base_ptr is provided, uses pre-allocated buffers at specified offsets (for backend buffer use)
// Returns OvWeight with the weight node and optional quantized tensors
OvWeight process_weight_tensor(
    ثابت ggml_tensor * tensor,
    ثابت تهی * data,                 // Source data pointer (may differ from tensor->data)
    تهی * output_base_ptr = nullptr,  // Base pointer for output buffers (or nullptr for internal allocation)
    bool use_bias = false);            // Use an exact f16 zero point (vs. a rounded integer one);
                                       // always used for for_gather_matmul (3D MoE expert) weights
                                       // regardless of this flag, and also settable explicitly for
                                       // test-backend-ops.
