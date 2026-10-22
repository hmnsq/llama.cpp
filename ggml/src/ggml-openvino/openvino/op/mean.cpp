#ضمن "../node_context.h"
#ضمن "../op_table.h"
#ضمن "../utils.h"

#ضمن <memory>
#ضمن <openvino/op/constant.hpp>
#ضمن <openvino/op/reduce_mean.hpp>

نطاق ov {
نطاق frontend {
نطاق ggml {
نطاق op {

Output_ويڪٽر translate_سراسري(ثابت ڳنڍ_Context & context) {
    انگ_inputs_check(context, 1, 1);

    خودڪار داخلا_2 = process_view_داخلا_new(context, 0);
    خودڪار axis = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {-1});
    خودڪار res = std::make_shared<ov::op::v1::Reduce_سراسري>(داخلا_2, axis, سچ);

    أرجع rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
