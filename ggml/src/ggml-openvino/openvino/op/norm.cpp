#ضمن "../node_context.h"
#ضمن "../op_table.h"
#ضمن "../utils.h"

#ضمن <cstring>
#ضمن <memory>
#ضمن <openvino/op/constant.hpp>
#ضمن <openvino/op/mvn.hpp>

نطاق ov {
نطاق frontend {
نطاق ggml {
نطاق op {

OutputVector translate_norm(ثابت نود_Context & context) {
    اڠک_inputs_check(context, 1, 1);

    auto ماسوقن_نود = process_view_ماسوقن_new(context, 0);
    عائم eps;
    memcpy(&eps, context.get_output_op_params(), حجم(عائم));

    auto axes = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {-1});
    auto res = std::make_shared<ov::op::v6::MVN>(ماسوقن_نود, axes, true, eps, ov::op::MVNEpsMode::INSIDE_SQRT);

    أرجع rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
