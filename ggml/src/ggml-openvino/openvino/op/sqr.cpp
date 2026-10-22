#شامل "../node_context.h"
#شامل "../op_table.h"
#شامل "../utils.h"

#شامل <memory>
#شامل <openvino/op/multiply.hpp>
#شامل <openvino/op/sqrt.hpp>

فضاینام ov {
فضاینام frontend {
فضاینام ggml {
فضاینام op {

OutputVector translate_sqr(ثابت گره_Context & context) {
    رقم_inputs_check(context, 1, 1);

    auto درون_داد = process_view_درون_داد_new(context, 0);
    auto res = std::make_shared<ov::op::v1::Multiply>(درون_داد, درون_داد);

    بازگشت rename_outputs_with_suffix({res}, context.get_name());
}

OutputVector translate_sqrt(ثابت گره_Context & context) {
    رقم_inputs_check(context, 1, 1);

    auto درون_داد = process_view_درون_داد_new(context, 0);
    auto res = std::make_shared<ov::op::v0::Sqrt>(درون_داد);

    بازگشت rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
