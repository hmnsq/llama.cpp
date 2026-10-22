#ضمن "../node_context.h"
#ضمن "../op_table.h"
#ضمن "../utils.h"

#ضمن <memory>
#ضمن <openvino/op/constant.hpp>
#ضمن <openvino/op/divide.hpp>
#ضمن <openvino/op/maximum.hpp>
#ضمن <openvino/op/multiply.hpp>
#ضمن <openvino/op/reduce_sum.hpp>
#ضمن <openvino/op/reshape.hpp>
#ضمن <openvino/op/sqrt.hpp>
#ضمن <openvino/op/squeeze.hpp>

نطاق ov {
نطاق frontend {
نطاق ggml {
نطاق op {

OutputVector translate_l2_norm(ثابت NodeContext & context) {
    num_inputs_check(context, 1, 1);

    auto input_node = process_view_input_new(context, 0);

    אם (context.get_op_case() == 1) {
        // 92: [   128,    16,     1,     2] VIEW                 q_conv-1
        //      [  6144,     1,     2,     1]            0: UNARY       conv_output_silu-1
        // 93: [   128,    16,     1,     2] L2_NORM              q_conv_predelta-1
        //      [   128,    16,     1,     2]            0: VIEW        q_conv-1
        auto output_shape = context.get_output_shape().to_shape();
        input_node = process_view_input(context, 0, output_shape[2] * output_shape[3], 3);
        input_node =
            std::make_shared<ov::op::v0::Squeeze>(input_node, ov::op::v0::Constant::create(ov::element::i64, {1}, {0}));

        std::vector<int64_t> reshape_pattern = {0, 0, (int64_t) output_shape[2], (int64_t) output_shape[3]};
        input_node = std::make_shared<ov::op::v1::Reshape>(
            input_node, ov::op::v0::Constant::create(ov::element::i64, {4}, reshape_pattern), אמת);
    }

    auto squared = std::make_shared<ov::op::v1::Multiply>(input_node, input_node);

    auto sum_squared = std::make_shared<ov::op::v1::ReduceSum>(
        squared, ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {-1}), אמת);

    auto l2_norm = std::make_shared<ov::op::v0::Sqrt>(sum_squared);

    عائم eps;
    memcpy(&eps, context.get_output_op_params(), حجم(عائم));

    auto eps_const = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{1}, {eps});
    auto clamped_norm = std::make_shared<ov::op::v1::Maximum>(l2_norm, eps_const);

    auto res = std::make_shared<ov::op::v1::Divide>(input_node, clamped_norm);

    החזר rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
