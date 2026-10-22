#شامل "../node_context.h"
#شامل "../op_table.h"
#شامل "../utils.h"

#شامل <openvino/op/broadcast.hpp>
#شامل <openvino/op/constant.hpp>

فضاینام ov {
فضاینام frontend {
فضاینام ggml {
فضاینام op {

// GGML FILL sets all elements of a tensor to a constant value.
// The constant is stored as a float in op_params[0].
OutputVector translate_fill(ثابت گره_Context & context) {
    رقم_inputs_check(context, 1, 1);

    اعشاری c;
    memcpy(&c, context.get_output_op_params(), اندازه(اعشاری));

    auto shape = context.get_درون_داد_shape(0).to_shape();

    auto val = ov::op::v0::Constant::create(context.get_output_type(), {}, {c});
    auto target_shape = ov::op::v0::Constant::create(ov::element::i64, {shape.size()},
        std::vector<int64_t>(shape.begin(), shape.end()));
    auto res = std::make_shared<ov::op::v3::Broadcast>(val, target_shape);

    بازگشت rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
