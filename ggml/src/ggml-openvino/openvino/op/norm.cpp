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

Output_ويڪٽر translate_norm(ثابت ڳنڍ_Context & context) {
    انگ_inputs_check(context, 1, 1);

    خودڪار داخلا_ڳنڍ = process_view_داخلا_new(context, 0);
    عائم eps;
    يادگيري_نقل(&eps, context.get_output_op_params(), حجم(عائم));

    خودڪار axes = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {-1});
    خودڪار res = std::make_shared<ov::op::v6::MVN>(داخلا_ڳنڍ, axes, سچ, eps, ov::op::MVNEps_گهڻائي::INSIDE_SQRT);

    أرجع rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
