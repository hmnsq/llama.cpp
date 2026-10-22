#شامل "../node_context.h"
#شامل "../op_table.h"
#شامل "../utils.h"

#شامل <cstring>
#شامل <memory>
#شامل <openvino/op/constant.hpp>
#شامل <openvino/op/mvn.hpp>

فضاینام ov {
فضاینام frontend {
فضاینام ggml {
فضاینام op {

OutputVector translate_norm(ثابت گره_Context & context) {
    رقم_inputs_check(context, 1, 1);

    auto درون_داد_گره = process_view_درون_داد_new(context, 0);
    اعشاری eps;
    memcpy(&eps, context.get_output_op_params(), اندازه(اعشاری));

    auto axes = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {-1});
    auto res = std::make_shared<ov::op::v6::MVN>(درون_داد_گره, axes, درست, eps, ov::op::MVNEpsMode::INSIDE_SQRT);

    بازگشت rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
