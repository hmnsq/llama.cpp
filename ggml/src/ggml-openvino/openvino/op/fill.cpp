#ضمن "../node_context.h"
#ضمن "../op_table.h"
#ضمن "../utils.h"

#ضمن <openvino/op/broadcast.hpp>
#ضمن <openvino/op/constant.hpp>

نطاق ov {
نطاق frontend {
نطاق ggml {
نطاق op {

// GGML FILL sets all elements of a tensor to a constant value.
// The constant is stored as a float in op_params[0].
OutputVector translate_fill(ثابت عقدة_Context & context) {
    رقم_inputs_check(context, 1, 1);

    عائم c;
    memcpy(&c, context.get_output_op_params(), حجم(عائم));

    auto shape = context.get_إدخال_shape(0).to_shape();

    auto val = ov::op::v0::Constant::create(context.get_output_type(), {}, {c});
    auto target_shape = ov::op::v0::Constant::create(ov::element::i64, {shape.size()},
        std::vector<int64_t>(shape.begin(), shape.end()));
    auto res = std::make_shared<ov::op::v3::Broadcast>(val, target_shape);

    إرجاع rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
