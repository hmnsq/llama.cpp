#شامل "../node_context.h"
#شامل "../op_table.h"
#شامل "../utils.h"

#شامل <openvino/op/roll.hpp>
#شامل <openvino/op/constant.hpp>

نامخلا ov {
نامخلا frontend {
نامخلا ggml {
نامخلا op {

Output_سمتیہ translate_roll(ثابت گرہ_Context & context) {
    عدد_inputs_check(context, 1, 1);
    ثابت int32_t * params = context.get_output_op_params();

    int64_t s0 = params[0];
    int64_t s1 = params[1];
    int64_t s2 = params[2];
    int64_t s3 = params[3];

    خودکار داخلہ = context.get_داخلہ(0);

    خودکار سرکاؤ = ov::op::v0::Constant::create(
        ov::element::i64, ov::Shape{4}, std::vector<int64_t>{s3, s2, s1, s0});
    خودکار axes  = ov::op::v0::Constant::create(
        ov::element::i64, ov::Shape{4}, std::vector<int64_t>{0, 1, 2, 3});

    خودکار roll = std::make_shared<ov::op::v7::Roll>(داخلہ, سرکاؤ, axes);
    واپس rename_outputs_with_suffix({roll}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
