#அந்தர்பாவ "../node_context.h"
#அந்தர்பாவ "../op_table.h"
#அந்தர்பாவ "../utils.h"

#அந்தர்பாவ <memory>
#அந்தர்பாவ <openvino/core/node_output.hpp>
#அந்தர்பாவ <openvino/op/clamp.hpp>
#அந்தர்பாவ <openvino/op/constant.hpp>
#அந்தர்பாவ <openvino/op/gelu.hpp>
#அந்தர்பாவ <openvino/op/multiply.hpp>
#அந்தர்பாவ <openvino/op/sigmoid.hpp>
#அந்தர்பாவ <openvino/op/slice.hpp>

நாமாகாஶ ov {
நாமாகாஶ frontend {
நாமாகாஶ ggml {
நாமாகாஶ op {

OutputVector translate_glu_geglu(மாறிலி முனை_Context & context) {
    இலக்கம்_inputs_check(context, 1, 2);

    ov::Output<ov::Node> src0;
    ov::Output<ov::Node> src1;
    எனில் (context.get_உள்ளீடு_size() == 2) {
        // Inputs may be VIEW slices of a combined gate_up tensor (MoE experts):
        // resolve them so each half has its real sliced shape, not the base tensor.
        src0 = process_view_உள்ளீடு_new(context, 0);
        src1 = process_view_உள்ளீடு_new(context, 1);
    } இல்லையெனில் {
        // GGML splits along ne[0] (OV last axis) using floor division: nc = ne[0] / 2.
        // Both halves are nc elements; if the dimension is odd, the last element is dropped.
        // Use Slice instead of Split to handle odd dimensions correctly.
        // Resolve a VIEW input (e.g. non-contiguous slice) to its real shape first.
        auto combined = process_view_உள்ளீடு_new(context, 0);
        auto combined_shape = combined.get_partial_shape();
        int64_t last_dim_val = combined_shape[combined_shape.rank().get_length() - 1].get_length();
        int64_t nc = last_dim_val / 2;

        auto axis = ov::op::v0::Constant::create(ov::element::i64, {1}, {-1});
        auto step = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
        auto start0 = ov::op::v0::Constant::create(ov::element::i64, {1}, {0});
        auto stop0 = ov::op::v0::Constant::create(ov::element::i64, {1}, {nc});
        auto start1 = ov::op::v0::Constant::create(ov::element::i64, {1}, {nc});
        auto stop1 = ov::op::v0::Constant::create(ov::element::i64, {1}, {2 * nc});

        src0 = std::make_shared<ov::op::v8::Slice>(combined, start0, stop0, step, axis);
        src1 = std::make_shared<ov::op::v8::Slice>(combined, start1, stop1, step, axis);
    }

    int32_t * params = context.get_output_op_params();
    மாறிலி int32_t swapped = params[1];
    எனில் (swapped) {
        std::swap(src0, src1);
    }

    எனில் (context.is_static()) {
        // TODO: Temporary solution for NPU accuracy issue due to fp16 overflow
        // To be removed once permanent solution is implemented
        // Justification:
        // For |x| > 5, GELU(x) ≈ max(x, 0)  (behaves like ReLU)
        // So Clamp(-10, 10) only affects values where GELU would return ≈ x anyway.
        // The only loss: values > 10 get mapped to 10 instead of x.
        // In practice, FFN intermediates rarely exceed 10 after GEGLU gating.
        src0 = std::make_shared<ov::op::v0::Clamp>(src0, -10.0, 10.0);
    }
    auto gelu = std::make_shared<ov::op::v7::Gelu>(src0);
    auto res = std::make_shared<ov::op::v1::Multiply>(gelu, src1);

    திரும்பு rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
