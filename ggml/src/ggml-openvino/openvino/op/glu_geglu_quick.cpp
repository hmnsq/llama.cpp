#અન્તર્ભાવ "../node_context.h"
#અન્તર્ભાવ "../op_table.h"
#અન્તર્ભાવ "../utils.h"

#અન્તર્ભાવ <memory>
#અન્તર્ભાવ <openvino/core/node_output.hpp>
#અન્તર્ભાવ <openvino/op/constant.hpp>
#અન્તર્ભાવ <openvino/op/multiply.hpp>
#અન્તર્ભાવ <openvino/op/slice.hpp>
#અન્તર્ભાવ <openvino/op/swish.hpp>

નામાકાશ ov {
નામાકાશ frontend {
નામાકાશ ggml {
નામાકાશ op {

OutputVector translate_glu_geglu_quick(અચલ ગાંઠ_Context & context) {
    અંક_inputs_check(context, 1, 2);

    ov::Output<ov::Node> src0;
    ov::Output<ov::Node> src1;
    જો (context.get_આગત_size() == 2) {
        src0 = process_view_આગત_new(context, 0);
        src1 = process_view_આગત_new(context, 1);
    } નહીંતર {
        // split along last axis, nc = ne[0] / 2
        auto combined = process_view_આગત_new(context, 0);
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
    અચલ int32_t swapped = params[1];
    જો (swapped) {
        std::swap(src0, src1);
    }

    // GELU_QUICK(x) = x * sigmoid(1.702 * x)
    // Create the constant in the same type as src0 to avoid f16/f32 mismatch.
    auto આગત_type = src0.get_element_type();
    auto coef = ov::op::v0::Constant::create(આગત_type, ov::Shape{}, {1.702f});
    auto gated   = std::make_shared<ov::op::v4::Swish>(src0, coef);
    auto res     = std::make_shared<ov::op::v1::Multiply>(gated, src1);

    પાછા rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
