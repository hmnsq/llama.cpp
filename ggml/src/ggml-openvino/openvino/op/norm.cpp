#شامل "../node_context.h"
#شامل "../op_table.h"
#شامل "../utils.h"

#شامل <cstring>
#شامل <memory>
#شامل <openvino/op/constant.hpp>
#شامل <openvino/op/mvn.hpp>

نامخلا ov {
نامخلا frontend {
نامخلا ggml {
نامخلا op {

Output_سمتیہ translate_norm(ثابت گرہ_Context & context) {
    عدد_inputs_check(context, 1, 1);

    خودکار داخلہ_گرہ = process_view_داخلہ_new(context, 0);
    اعشاری eps;
    حافظہ_نقل(&eps, context.get_output_op_params(), حجم(اعشاری));

    خودکار axes = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {-1});
    خودکار res = std::make_shared<ov::op::v6::MVN>(داخلہ_گرہ, axes, سچ, eps, ov::op::MVNEps_منوال::INSIDE_SQRT);

    واپس rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
