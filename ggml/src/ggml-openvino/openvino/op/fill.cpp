#شامل "../node_context.h"
#شامل "../op_table.h"
#شامل "../utils.h"

#شامل <openvino/op/broadcast.hpp>
#شامل <openvino/op/constant.hpp>

نامخلا ov {
نامخلا frontend {
نامخلا ggml {
نامخلا op {

// GGML FILL sets all elements of a tensor to a constant value.
// The constant is stored as a float in op_params[0].
Output_سمتیہ translate_fill(ثابت گرہ_Context & context) {
    عدد_inputs_check(context, 1, 1);

    اعشاری c;
    حافظہ_نقل(&c, context.get_output_op_params(), حجم(اعشاری));

    خودکار shape = context.get_داخلہ_shape(0).مقصد_shape();

    خودکار val = ov::op::v0::Constant::create(context.get_output_type(), {}, {c});
    خودکار ہدف_shape = ov::op::v0::Constant::create(ov::element::i64, {shape.size()},
        std::vector<int64_t>(shape.begin(), shape.end()));
    خودکار res = std::make_shared<ov::op::v3::نشر>(val, ہدف_shape);

    واپس rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
