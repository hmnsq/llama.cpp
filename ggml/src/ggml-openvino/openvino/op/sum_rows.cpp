#ضمن "../node_context.h"
#ضمن "../op_table.h"
#ضمن "../utils.h"

#ضمن <memory>
#ضمن <openvino/op/constant.hpp>
#ضمن <openvino/op/reduce_sum.hpp>

نطاق ov {
نطاق frontend {
نطاق ggml {
نطاق op {

OutputVector translate_sum_rows(ثابت گرہ_Context & context) {
    عدد_inputs_check(context, 1, 1);

    auto داخلہ = process_view_داخلہ_new(context, 0);
    auto res = std::make_shared<ov::op::v1::ReduceSum>(
        داخلہ, ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {-1}), true);

    أرجع rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
