#ضمن "../node_context.h"
#ضمن "../op_table.h"
#ضمن "../utils.h"

#ضمن <memory>
#ضمن <openvino/op/constant.hpp>
#ضمن <openvino/op/reduce_sum.hpp>

نطاق ov {
نطاق frontend {
نطاق ggml {
نطاق op {

Output_وکتور translate_جمع_ردیفها(ثابت گره_Context & context) {
    رقم_inputs_check(context, 1, 1);

    خودکار درون_داد = process_view_درون_داد_new(context, 0);
    خودکار res = std::make_shared<ov::op::v1::Reduce_جمع>(
        درون_داد, ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {-1}), درست);

    أرجع rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
