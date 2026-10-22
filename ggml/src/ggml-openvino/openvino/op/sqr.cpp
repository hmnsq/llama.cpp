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

Output_وکتور translate_sqr(ثابت گره_Context & context) {
    رقم_inputs_check(context, 1, 1);

    خودکار درون_داد = process_view_درون_داد_new(context, 0);
    خودکار res = std::make_shared<ov::op::v1::Multiply>(درون_داد, درون_داد);

    أرجع rename_outputs_with_suffix({res}, context.get_name());
}

Output_وکتور translate_sqrt(ثابت گره_Context & context) {
    رقم_inputs_check(context, 1, 1);

    خودکار درون_داد = process_view_درون_داد_new(context, 0);
    خودکار res = std::make_shared<ov::op::v0::Sqrt>(درون_داد);

    أرجع rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
