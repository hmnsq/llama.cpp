#அந்தர்பாவ "../node_context.h"
#அந்தர்பாவ "../op_table.h"
#அந்தர்பாவ "../utils.h"

#அந்தர்பாவ <memory>
#அந்தர்பாவ <openvino/core/node_output.hpp>
#அந்தர்பாவ <openvino/op/constant.hpp>
#அந்தர்பாவ <openvino/op/multiply.hpp>
#அந்தர்பாவ <openvino/op/slice.hpp>
#அந்தர்பாவ <openvino/op/swish.hpp>

நாமாகாஶ ov {
நாமாகாஶ frontend {
நாமாகாஶ ggml {
நாமாகாஶ op {

OutputVector translate_glu_geglu_quick(மாறிலி முனை_Context & context) {
    இலக்கம்_inputs_check(context, 1, 2);

    ov::Output<ov::Node> src0;
    ov::Output<ov::Node> src1;
    எனில் (context.get_உள்ளீடு_size() == 2) {
        src0 = process_view_உள்ளீடு_new(context, 0);
        src1 = process_view_உள்ளீடு_new(context, 1);
    } இல்லையெனில் {
        // split along last axis, nc = ne[0] / 2
        auto combined = process_view_உள்ளீடு_new(context, 0);
        auto combined_shape = combined.get_partial_shape();
        int64_t last_dim_val = combined_shape[combined_shape.rank().get_length() - 1].get_length();
        int64_t nc = last_dim_val / 2;

        auto axis  = ov::op::v0::Constant::create(ov::element::i64, {1}, {-1});
        auto step  = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
        auto start0 = ov::op::v0::Constant::create(ov::element::i64, {1}, {0});
        auto stop0  = ov::op::v0::Constant::create(ov::element::i64, {1}, {nc});
        auto start1 = ov::op::v0::Constant::create(ov::element::i64, {1}, {nc});
        auto stop1  = ov::op::v0::Constant::create(ov::element::i64, {1}, {2 * nc});

        src0 = std::make_shared<ov::op::v8::Slice>(combined, start0, stop0, step, axis);
        src1 = std::make_shared<ov::op::v8::Slice>(combined, start1, stop1, step, axis);
    }

    int32_t * params = context.get_output_op_params();
    மாறிலி int32_t swapped = params[1];
    எனில் (swapped) {
        std::swap(src0, src1);
    }

    // GELU_QUICK(x) = x * sigmoid(1.702 * x)
    // Create the constant in the same type as src0 to avoid f16/f32 mismatch.
    auto உள்ளீடு_type = src0.get_element_type();
    auto coef = ov::op::v0::Constant::create(உள்ளீடு_type, ov::Shape{}, {1.702f});
    auto gated   = std::make_shared<ov::op::v4::Swish>(src0, coef);
    auto res     = std::make_shared<ov::op::v1::Multiply>(gated, src1);

    திரும்பு rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
