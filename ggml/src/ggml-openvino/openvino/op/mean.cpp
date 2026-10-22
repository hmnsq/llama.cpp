#ضمن "../node_context.h"
#ضمن "../op_table.h"
#ضمن "../utils.h"

#ضمن <memory>
#ضمن <openvino/op/constant.hpp>
#ضمن <openvino/op/reduce_mean.hpp>

نطاق ov {
نطاق frontend {
نطاق ggml {
نطاق op {

OutputVector translate_mean(ثابت گرێ_Context & context) {
    ژمارە_inputs_check(context, 1, 1);

    auto تێخستن = process_view_تێخستن_new(context, 0);
    auto axis = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {-1});
    auto res = std::make_shared<ov::op::v1::ReduceMean>(تێخستن, axis, true);

    أرجع rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
