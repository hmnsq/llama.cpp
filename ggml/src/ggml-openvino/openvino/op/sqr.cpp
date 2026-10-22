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

Output_ويڪٽر translate_sqr(ثابت ڳنڍ_Context & context) {
    انگ_inputs_check(context, 1, 1);

    خودڪار داخلا_2 = process_view_داخلا_new(context, 0);
    خودڪار res = std::make_shared<ov::op::v1::Multiply>(داخلا_2, داخلا_2);

    أرجع rename_outputs_with_suffix({res}, context.get_name());
}

Output_ويڪٽر translate_sqrt(ثابت ڳنڍ_Context & context) {
    انگ_inputs_check(context, 1, 1);

    خودڪار داخلا_2 = process_view_داخلا_new(context, 0);
    خودڪار res = std::make_shared<ov::op::v0::Sqrt>(داخلا_2);

    أرجع rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
