#ہدایت once
#شامل "ggml-openvino-extra.h"  // For ExtraQuantType
#شامل "ggml.h"

#شامل <openvino/core/node_output.hpp>
#شامل <openvino/op/constant.hpp>
#شامل <openvino/runtime/tensor.hpp>

ساکن constexpr size_t GGML_QUANTIZATION_گروہ_سائز = 32;

اندرونی ثابت حرف * extra_quant_type_name(ExtraQuantType t) {
    انتخاب (t) {
    حالت ExtraQuantType::F16:
        واپس "F16";
    حالت ExtraQuantType::Q4_0_C:
        واپس "Q4_0_C";
    حالت ExtraQuantType::Q4_0_128:
        واپس "Q4_0_128";
    حالت ExtraQuantType::Q8_0_C:
        واپس "Q8_0_C";
    حالت ExtraQuantType::Q8_0_32:
        واپس "Q8_0_32";
    حالت ExtraQuantType::Q8_1_C:
        واپس "Q8_1_C";
    حالت ExtraQuantType::Q4_0_64:
        واپس "Q4_0_64";
    حالت ExtraQuantType::Q4_1_64:
        واپس "Q4_1_64";
    طےشدہ:
        واپس "unknown";
    }
}

// Result from process_weight_tensor containing the weight node and tensors.
// For quantized weights, also contains the extracted layout and scale/zp tensors.
ساخت Ov_وزن {
    std::shared_ptr<ov::گرہ_2> وزن_گرہ;
    ggml_openvino_extracted_layout layout;  // Only meaningful for quantized (layout.total_size > 0)
    ov::Tensor weights;
    ov::Tensor scales;
    ov::Tensor zp;

    bool is_quantized() ثابت { واپس layout.scales_سائز > 0; }
};

// Process weight tensor and create an OpenVINO weight node
// Handles F16/F32/BF16 and quantized weights, with optional requantization
// If output_base_ptr is nullptr, allocates internal buffers (for decoder use)
// If output_base_ptr is provided, uses pre-allocated buffers at specified offsets (for backend buffer use)
// Returns OvWeight with the weight node and optional quantized tensors
Ov_وزن process_وزن_tensor(
    ثابت ggml_tensor * tensor,
    ثابت خالی * ڈیٹا,                 // Source data pointer (may differ from tensor->data)
    خالی * output_بنیاد_ptr = nullptr,  // Base pointer for output buffers (or nullptr for internal allocation)
    bool use_bias = false);            // Use an exact f16 zero point (vs. a rounded integer one);
                                       // always used for for_gather_matmul (3D MoE expert) weights
                                       // regardless of this flag, and also settable explicitly for
                                       // test-backend-ops.
