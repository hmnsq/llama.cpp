
#شامل "../node_context.h"
#شامل "../op_table.h"
#شامل "../utils.h"

#شامل <memory>
#شامل <openvino/op/reshape.hpp>
#شامل <openvino/op/slice.hpp>

نامخلا ov {
نامخلا frontend {
نامخلا ggml {
نامخلا op {

Output_سمتیہ translate_cont(ثابت گرہ_Context & context) {
    عدد_inputs_check(context, 1, 1);

    خودکار src_shape = context.get_داخلہ_shape(0).مقصد_shape();
    خودکار dst_shape = context.get_output_shape().مقصد_shape();

    اگر (context.get_op_dynamic_dim() != -1) {
        dst_shape[3 - context.get_op_dynamic_dim()] = -1;
    }

    خودکار داخلہ = process_view_داخلہ_new(context, 0);

    ov::Output<Node> res;
    res = std::make_shared<ov::op::v1::Reshape>(
        داخلہ, ov::op::v0::Constant::create(ov::element::i64, {dst_shape.size()}, dst_shape), جھوٹ);

    واپس rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
