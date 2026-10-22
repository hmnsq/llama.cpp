
#ضمن "../node_context.h"
#ضمن "../op_table.h"
#ضمن "../utils.h"

#ضمن <memory>
#ضمن <openvino/op/reshape.hpp>
#ضمن <openvino/op/slice.hpp>

نطاق ov {
نطاق frontend {
نطاق ggml {
نطاق op {

Output_ويڪٽر translate_cont(ثابت ڳنڍ_Context & context) {
    انگ_inputs_check(context, 1, 1);

    خودڪار src_shape = context.get_داخلا_shape(0).منزل_shape();
    خودڪار dst_shape = context.get_output_shape().منزل_shape();

    إذا (context.get_op_dynamic_dim() != -1) {
        dst_shape[3 - context.get_op_dynamic_dim()] = -1;
    }

    خودڪار داخلا_2 = process_view_داخلا_new(context, 0);

    ov::Output<Node> res;
    res = std::make_shared<ov::op::v1::Reshape>(
        داخلا_2, ov::op::v0::Constant::create(ov::element::i64, {dst_shape.size()}, dst_shape), خطأ);

    أرجع rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
