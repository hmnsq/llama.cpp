#pragma once
#ଅନ୍ତର୍ଭାଵ "ggml-openvino-extra.h"  // For ExtraQuantType
#ଅନ୍ତର୍ଭାଵ "ggml.h"

#ଅନ୍ତର୍ଭାଵ <openvino/core/node_output.hpp>
#ଅନ୍ତର୍ଭାଵ <openvino/op/constant.hpp>
#ଅନ୍ତର୍ଭାଵ <openvino/runtime/tensor.hpp>

ସ୍ଥିର constexpr size_t GGML_QUANTIZATION_GROUP_SIZE = 32;

ଧାଡ଼ିଭିତରେ ଅଚଳ ଅକ୍ଷର * extra_quant_type_name(ExtraQuantType t) {
    ବାଛ (t) {
    ସ୍ଥିତି ExtraQuantType::F16:
        ଫେରନ୍ତୁ "F16";
    ସ୍ଥିତି ExtraQuantType::Q4_0_C:
        ଫେରନ୍ତୁ "Q4_0_C";
    ସ୍ଥିତି ExtraQuantType::Q4_0_128:
        ଫେରନ୍ତୁ "Q4_0_128";
    ସ୍ଥିତି ExtraQuantType::Q8_0_C:
        ଫେରନ୍ତୁ "Q8_0_C";
    ସ୍ଥିତି ExtraQuantType::Q8_0_32:
        ଫେରନ୍ତୁ "Q8_0_32";
    ସ୍ଥିତି ExtraQuantType::Q8_1_C:
        ଫେରନ୍ତୁ "Q8_1_C";
    ସ୍ଥିତି ExtraQuantType::Q4_0_64:
        ଫେରନ୍ତୁ "Q4_0_64";
    ସ୍ଥିତି ExtraQuantType::Q4_1_64:
        ଫେରନ୍ତୁ "Q4_1_64";
    ପୂର୍ବନିର୍ଦ୍ଧାରିତ:
        ଫେରନ୍ତୁ "unknown";
    }
}

// Result from process_weight_tensor containing the weight node and tensors.
// For quantized weights, also contains the extracted layout and scale/zp tensors.
ସଂରଚନା OvWeight {
    std::shared_ptr<ov::ଗଣ୍ଠି_2> weight_ଗଣ୍ଠି;
    ggml_openvino_extracted_layout layout;  // Only meaningful for quantized (layout.total_size > 0)
    ov::Tensor weights;
    ov::Tensor scales;
    ov::Tensor zp;

    bool is_quantized() ଅଚଳ { ଫେରନ୍ତୁ layout.scales_size > 0; }
};

// Process weight tensor and create an OpenVINO weight node
// Handles F16/F32/BF16 and quantized weights, with optional requantization
// If output_base_ptr is nullptr, allocates internal buffers (for decoder use)
// If output_base_ptr is provided, uses pre-allocated buffers at specified offsets (for backend buffer use)
// Returns OvWeight with the weight node and optional quantized tensors
OvWeight process_weight_tensor(
    ଅଚଳ ggml_tensor * tensor,
    ଅଚଳ ଶୂନ୍ୟ * data,                 // Source data pointer (may differ from tensor->data)
    ଶୂନ୍ୟ * output_base_ptr = nullptr,  // Base pointer for output buffers (or nullptr for internal allocation)
    bool use_bias = false);            // Use an exact f16 zero point (vs. a rounded integer one);
                                       // always used for for_gather_matmul (3D MoE expert) weights
                                       // regardless of this flag, and also settable explicitly for
                                       // test-backend-ops.
