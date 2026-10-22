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

Output_ویکټور translate_sqr(ثابت غوټه_Context & context) {
    عدد_inputs_check(context, 1, 1);

    خپلکاری ننوت = process_view_ننوت_new(context, 0);
    خپلکاری res = std::make_shared<ov::op::v1::Multiply>(ننوت, ننوت);

    أرجع rename_outputs_with_suffix({res}, context.get_name());
}

Output_ویکټور translate_sqrt(ثابت غوټه_Context & context) {
    عدد_inputs_check(context, 1, 1);

    خپلکاری ننوت = process_view_ننوت_new(context, 0);
    خپلکاری res = std::make_shared<ov::op::v0::Sqrt>(ننوت);

    أرجع rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
