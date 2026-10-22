#شامل "../node_context.h"
#شامل "../op_table.h"
#شامل "../utils.h"

#شامل <memory>
#شامل <openvino/op/constant.hpp>
#شامل <openvino/op/range.hpp>
#شامل <openvino/op/reduce_sum.hpp>
#شامل <openvino/op/reshape.hpp>
#شامل <openvino/op/shape_of.hpp>

فضاینام ov {
فضاینام frontend {
فضاینام ggml {
فضاینام op {

OutputVector translate_sum(ثابت گره_Context & context) {
    رقم_inputs_check(context, 1, 1);

    auto درون_داد = process_view_درون_داد_new(context, 0);
    auto axes = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{4}, {0, 1, 2, 3});
    auto res = std::make_shared<ov::op::v1::ReduceSum>(درون_داد, axes, درست);

    بازگشت rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
