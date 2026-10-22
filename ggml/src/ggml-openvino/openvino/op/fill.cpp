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
Output_سمتیہ translate_fill(ثابت گرہ_Context & context) {
    عدد_inputs_check(context, 1, 1);

    عائم c;
    حافظہ_نقل(&c, context.get_output_op_params(), حجم(عائم));

    خودکار shape = context.get_داخلہ_shape(0).منزل_shape();

    خودکار val = ov::op::v0::Constant::create(context.get_output_type(), {}, {c});
    خودکار نشانہ_shape = ov::op::v0::Constant::create(ov::element::i64, {shape.size()},
        std::vector<int64_t>(shape.begin(), shape.end()));
    خودکار res = std::make_shared<ov::op::v3::نشر>(val, نشانہ_shape);

    أرجع rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
