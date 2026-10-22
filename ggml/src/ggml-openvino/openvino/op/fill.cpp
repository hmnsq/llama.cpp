#அந்தர்பாவ "../node_context.h"
#அந்தர்பாவ "../op_table.h"
#அந்தர்பாவ "../utils.h"

#அந்தர்பாவ <openvino/op/broadcast.hpp>
#அந்தர்பாவ <openvino/op/constant.hpp>

நாமாகாஶ ov {
நாமாகாஶ frontend {
நாமாகாஶ ggml {
நாமாகாஶ op {

// GGML FILL sets all elements of a tensor to a constant value.
// The constant is stored as a float in op_params[0].
OutputVector translate_fill(மாறிலி முனை_Context & context) {
    இலக்கம்_inputs_check(context, 1, 1);

    தசம c;
    memcpy(&c, context.get_output_op_params(), அளவு(தசம));

    auto shape = context.get_உள்ளீடு_shape(0).to_shape();

    auto val = ov::op::v0::Constant::create(context.get_output_type(), {}, {c});
    auto target_shape = ov::op::v0::Constant::create(ov::element::i64, {shape.size()},
        std::vector<int64_t>(shape.begin(), shape.end()));
    auto res = std::make_shared<ov::op::v3::Broadcast>(val, target_shape);

    திரும்பு rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
