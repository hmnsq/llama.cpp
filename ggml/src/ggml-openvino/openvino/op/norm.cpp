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

Output_وکتور translate_norm(ثابت گره_Context & context) {
    رقم_inputs_check(context, 1, 1);

    خودکار درون_داد_گره = process_view_درون_داد_new(context, 0);
    عائم eps;
    کپی_حافظه(&eps, context.get_output_op_params(), حجم(عائم));

    خودکار axes = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {-1});
    خودکار res = std::make_shared<ov::op::v6::MVN>(درون_داد_گره, axes, درست, eps, ov::op::MVNEps_نما::INSIDE_SQRT);

    أرجع rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
