#ضمن "../node_context.h"
#ضمن "../op_table.h"
#ضمن "../utils.h"

#ضمن <memory>
#ضمن <openvino/op/constant.hpp>
#ضمن <openvino/op/range.hpp>
#ضمن <openvino/op/reduce_sum.hpp>
#ضمن <openvino/op/reshape.hpp>
#ضمن <openvino/op/shape_of.hpp>

نطاق ov {
نطاق frontend {
نطاق ggml {
نطاق op {

OutputVector translate_sum(ثابت عقدة_Context & context) {
    رقم_inputs_check(context, 1, 1);

    auto إدخال = process_view_إدخال_new(context, 0);
    auto axes = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{4}, {0, 1, 2, 3});
    auto res = std::make_shared<ov::op::v1::ReduceSum>(إدخال, axes, true);

    إرجاع rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
