#సూచన once
#అన్తర్భావ "ggml-openvino-extra.h"  // For ExtraQuantType
#అన్తర్భావ "ggml.h"

#అన్తర్భావ <openvino/core/node_output.hpp>
#అన్తర్భావ <openvino/op/constant.hpp>
#అన్తర్భావ <openvino/runtime/tensor.hpp>

స్థిర constexpr size_t GGML_QUANTIZATION_సమూహం_కొలత = 32;

అంతఃవరుస స్థిరరాశి అక్షరం * extra_quant_type_name(ExtraQuantType t) {
    ఎంపిక (t) {
    స్థితి ExtraQuantType::F16:
        తిరిగి "F16";
    స్థితి ExtraQuantType::Q4_0_C:
        తిరిగి "Q4_0_C";
    స్థితి ExtraQuantType::Q4_0_128:
        తిరిగి "Q4_0_128";
    స్థితి ExtraQuantType::Q8_0_C:
        తిరిగి "Q8_0_C";
    స్థితి ExtraQuantType::Q8_0_32:
        తిరిగి "Q8_0_32";
    స్థితి ExtraQuantType::Q8_1_C:
        తిరిగి "Q8_1_C";
    స్థితి ExtraQuantType::Q4_0_64:
        తిరిగి "Q4_0_64";
    స్థితి ExtraQuantType::Q4_1_64:
        తిరిగి "Q4_1_64";
    అప్రమేయ:
        తిరిగి "unknown";
    }
}

// Result from process_weight_tensor containing the weight node and tensors.
// For quantized weights, also contains the extracted layout and scale/zp tensors.
నిర్మాణం Ov_బరువు {
    std::shared_ptr<ov::కణుపు_2> బరువు_కణుపు;
    ggml_openvino_extracted_layout layout;  // Only meaningful for quantized (layout.total_size > 0)
    ov::Tensor weights;
    ov::Tensor scales;
    ov::Tensor zp;

    bool is_quantized() స్థిరరాశి { తిరిగి layout.scales_కొలత > 0; }
};

// Process weight tensor and create an OpenVINO weight node
// Handles F16/F32/BF16 and quantized weights, with optional requantization
// If output_base_ptr is nullptr, allocates internal buffers (for decoder use)
// If output_base_ptr is provided, uses pre-allocated buffers at specified offsets (for backend buffer use)
// Returns OvWeight with the weight node and optional quantized tensors
Ov_బరువు process_బరువు_tensor(
    స్థిరరాశి ggml_tensor * tensor,
    స్థిరరాశి శూన్య * దత్తాంశం,                 // Source data pointer (may differ from tensor->data)
    శూన్య * output_పునాది_ptr = nullptr,  // Base pointer for output buffers (or nullptr for internal allocation)
    bool use_bias = false);            // Use an exact f16 zero point (vs. a rounded integer one);
                                       // always used for for_gather_matmul (3D MoE expert) weights
                                       // regardless of this flag, and also settable explicitly for
                                       // test-backend-ops.
