#pragma once
#અન્તર્ભાવ "ggml-openvino-extra.h"  // For ExtraQuantType
#અન્તર્ભાવ "ggml.h"

#અન્તર્ભાવ <openvino/core/node_output.hpp>
#અન્તર્ભાવ <openvino/op/constant.hpp>
#અન્તર્ભાવ <openvino/runtime/tensor.hpp>

સ્થિર constexpr size_t GGML_QUANTIZATION_GROUP_SIZE = 32;

અંતઃરેખ અચલ અક્ષર * extra_quant_type_name(ExtraQuantType t) {
    પસંદગી (t) {
    સ્થિતિ ExtraQuantType::F16:
        પાછા "F16";
    સ્થિતિ ExtraQuantType::Q4_0_C:
        પાછા "Q4_0_C";
    સ્થિતિ ExtraQuantType::Q4_0_128:
        પાછા "Q4_0_128";
    સ્થિતિ ExtraQuantType::Q8_0_C:
        પાછા "Q8_0_C";
    સ્થિતિ ExtraQuantType::Q8_0_32:
        પાછા "Q8_0_32";
    સ્થિતિ ExtraQuantType::Q8_1_C:
        પાછા "Q8_1_C";
    સ્થિતિ ExtraQuantType::Q4_0_64:
        પાછા "Q4_0_64";
    સ્થિતિ ExtraQuantType::Q4_1_64:
        પાછા "Q4_1_64";
    મૂળભૂત:
        પાછા "unknown";
    }
}

// Result from process_weight_tensor containing the weight node and tensors.
// For quantized weights, also contains the extracted layout and scale/zp tensors.
સંરચના OvWeight {
    std::shared_ptr<ov::ગાંઠ_2> weight_ગાંઠ;
    ggml_openvino_extracted_layout layout;  // Only meaningful for quantized (layout.total_size > 0)
    ov::Tensor weights;
    ov::Tensor scales;
    ov::Tensor zp;

    bool is_quantized() અચલ { પાછા layout.scales_size > 0; }
};

// Process weight tensor and create an OpenVINO weight node
// Handles F16/F32/BF16 and quantized weights, with optional requantization
// If output_base_ptr is nullptr, allocates internal buffers (for decoder use)
// If output_base_ptr is provided, uses pre-allocated buffers at specified offsets (for backend buffer use)
// Returns OvWeight with the weight node and optional quantized tensors
OvWeight process_weight_tensor(
    અચલ ggml_tensor * tensor,
    અચલ શૂન્ય * data,                 // Source data pointer (may differ from tensor->data)
    શૂન્ય * output_base_ptr = nullptr,  // Base pointer for output buffers (or nullptr for internal allocation)
    bool use_bias = false);            // Use an exact f16 zero point (vs. a rounded integer one);
                                       // always used for for_gather_matmul (3D MoE expert) weights
                                       // regardless of this flag, and also settable explicitly for
                                       // test-backend-ops.
