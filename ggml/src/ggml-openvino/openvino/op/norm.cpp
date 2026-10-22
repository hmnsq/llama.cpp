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

Output_ویکټور translate_norm(ثابت غوټه_Context & context) {
    عدد_inputs_check(context, 1, 1);

    خپلکاری ننوت_غوټه = process_view_ننوت_new(context, 0);
    عائم eps;
    حافظه_نقل(&eps, context.get_output_op_params(), حجم(عائم));

    خپلکاری axes = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {-1});
    خپلکاری res = std::make_shared<ov::op::v6::MVN>(ننوت_غوټه, axes, ریښتیا, eps, ov::op::MVNEps_ډېرپېښ::INSIDE_SQRT);

    أرجع rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
