#ضمن "../node_context.h"
#ضمن "../op_table.h"
#ضمن "../utils.h"

#ضمن <memory>
#ضمن <openvino/op/multiply.hpp>
#ضمن <openvino/op/sqrt.hpp>

نطاق ov {
نطاق frontend {
نطاق ggml {
نطاق op {

OutputVector translate_sqr(ثابت تۈگۈن_Context & context) {
    رەقەم_inputs_check(context, 1, 1);

    auto كىرگۈزۈش = process_view_كىرگۈزۈش_new(context, 0);
    auto res = std::make_shared<ov::op::v1::Multiply>(كىرگۈزۈش, كىرگۈزۈش);

    أرجع rename_outputs_with_suffix({res}, context.get_name());
}

OutputVector translate_sqrt(ثابت تۈگۈن_Context & context) {
    رەقەم_inputs_check(context, 1, 1);

    auto كىرگۈزۈش = process_view_كىرگۈزۈش_new(context, 0);
    auto res = std::make_shared<ov::op::v0::Sqrt>(كىرگۈزۈش);

    أرجع rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
