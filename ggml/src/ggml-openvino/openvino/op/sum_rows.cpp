#شامل "../node_context.h"
#شامل "../op_table.h"
#شامل "../utils.h"

#شامل <memory>
#شامل <openvino/op/constant.hpp>
#شامل <openvino/op/reduce_sum.hpp>

فضاینام ov {
فضاینام frontend {
فضاینام ggml {
فضاینام op {

OutputVector translate_sum_rows(ثابت گره_Context & context) {
    رقم_inputs_check(context, 1, 1);

    auto درون_داد = process_view_درون_داد_new(context, 0);
    auto res = std::make_shared<ov::op::v1::ReduceSum>(
        درون_داد, ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {-1}), درست);

    بازگشت rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
