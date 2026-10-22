#شامل "../node_context.h"
#شامل "../op_table.h"
#شامل "../utils.h"

#شامل <memory>
#شامل <openvino/op/constant.hpp>
#شامل <openvino/op/divide.hpp>
#شامل <openvino/op/maximum.hpp>
#شامل <openvino/op/multiply.hpp>
#شامل <openvino/op/reduce_sum.hpp>
#شامل <openvino/op/reshape.hpp>
#شامل <openvino/op/sqrt.hpp>
#شامل <openvino/op/squeeze.hpp>

فضاینام ov {
فضاینام frontend {
فضاینام ggml {
فضاینام op {

OutputVector translate_l2_norm(ثابت گره_Context & context) {
    رقم_inputs_check(context, 1, 1);

    auto درون_داد_گره = process_view_درون_داد_new(context, 0);

    اگر (context.get_op_case() == 1) {
        // 92: [   128,    16,     1,     2] VIEW                 q_conv-1
        //      [  6144,     1,     2,     1]            0: UNARY       conv_output_silu-1
        // 93: [   128,    16,     1,     2] L2_NORM              q_conv_predelta-1
        //      [   128,    16,     1,     2]            0: VIEW        q_conv-1
        auto output_shape = context.get_output_shape().to_shape();
        درون_داد_گره = process_view_درون_داد(context, 0, output_shape[2] * output_shape[3], 3);
        درون_داد_گره =
            std::make_shared<ov::op::v0::Squeeze>(درون_داد_گره, ov::op::v0::Constant::create(ov::element::i64, {1}, {0}));

        std::vector<int64_t> reshape_pattern = {0, 0, (int64_t) output_shape[2], (int64_t) output_shape[3]};
        درون_داد_گره = std::make_shared<ov::op::v1::Reshape>(
            درون_داد_گره, ov::op::v0::Constant::create(ov::element::i64, {4}, reshape_pattern), درست);
    }

    auto squared = std::make_shared<ov::op::v1::Multiply>(درون_داد_گره, درون_داد_گره);

    auto sum_squared = std::make_shared<ov::op::v1::ReduceSum>(
        squared, ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {-1}), درست);

    auto l2_norm = std::make_shared<ov::op::v0::Sqrt>(sum_squared);

    اعشاری eps;
    memcpy(&eps, context.get_output_op_params(), اندازه(اعشاری));

    auto eps_const = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{1}, {eps});
    auto clamped_norm = std::make_shared<ov::op::v1::Maximum>(l2_norm, eps_const);

    auto res = std::make_shared<ov::op::v1::Divide>(درون_داد_گره, clamped_norm);

    بازگشت rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
