#شامل "../node_context.h"
#شامل "../op_table.h"
#شامل "../utils.h"

#شامل <memory>
#شامل <openvino/op/constant.hpp>
#شامل <openvino/op/range.hpp>
#شامل <openvino/op/reduce_sum.hpp>
#شامل <openvino/op/reshape.hpp>
#شامل <openvino/op/shape_of.hpp>

نامخلا ov {
نامخلا frontend {
نامخلا ggml {
نامخلا op {

Output_سمتیہ translate_مجموع(ثابت گرہ_Context & context) {
    عدد_inputs_check(context, 1, 1);

    خودکار داخلہ = process_view_داخلہ_new(context, 0);
    خودکار axes = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{4}, {0, 1, 2, 3});
    خودکار res = std::make_shared<ov::op::v1::Reduce_مجموع>(داخلہ, axes, سچ);

    واپس rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
