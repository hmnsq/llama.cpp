#અન્તર્ભાવ "../node_context.h"
#અન્તર્ભાવ "../op_table.h"
#અન્તર્ભાવ "../utils.h"

#અન્તર્ભાવ <memory>
#અન્તર્ભાવ <openvino/op/constant.hpp>
#અન્તર્ભાવ <openvino/op/divide.hpp>
#અન્તર્ભાવ <openvino/op/maximum.hpp>
#અન્તર્ભાવ <openvino/op/multiply.hpp>
#અન્તર્ભાવ <openvino/op/reduce_sum.hpp>
#અન્તર્ભાવ <openvino/op/reshape.hpp>
#અન્તર્ભાવ <openvino/op/sqrt.hpp>
#અન્તર્ભાવ <openvino/op/squeeze.hpp>

નામાકાશ ov {
નામાકાશ frontend {
નામાકાશ ggml {
નામાકાશ op {

OutputVector translate_l2_norm(અચલ ગાંઠ_Context & context) {
    અંક_inputs_check(context, 1, 1);

    auto આગત_ગાંઠ = process_view_આગત_new(context, 0);

    જો (context.get_op_case() == 1) {
        // 92: [   128,    16,     1,     2] VIEW                 q_conv-1
        //      [  6144,     1,     2,     1]            0: UNARY       conv_output_silu-1
        // 93: [   128,    16,     1,     2] L2_NORM              q_conv_predelta-1
        //      [   128,    16,     1,     2]            0: VIEW        q_conv-1
        auto output_shape = context.get_output_shape().to_shape();
        આગત_ગાંઠ = process_view_આગત(context, 0, output_shape[2] * output_shape[3], 3);
        આગત_ગાંઠ =
            std::make_shared<ov::op::v0::Squeeze>(આગત_ગાંઠ, ov::op::v0::Constant::create(ov::element::i64, {1}, {0}));

        std::vector<int64_t> reshape_pattern = {0, 0, (int64_t) output_shape[2], (int64_t) output_shape[3]};
        આગત_ગાંઠ = std::make_shared<ov::op::v1::Reshape>(
            આગત_ગાંઠ, ov::op::v0::Constant::create(ov::element::i64, {4}, reshape_pattern), સાચું);
    }

    auto squared = std::make_shared<ov::op::v1::Multiply>(આગત_ગાંઠ, આગત_ગાંઠ);

    auto sum_squared = std::make_shared<ov::op::v1::ReduceSum>(
        squared, ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {-1}), સાચું);

    auto l2_norm = std::make_shared<ov::op::v0::Sqrt>(sum_squared);

    દશાંશ eps;
    memcpy(&eps, context.get_output_op_params(), કદ(દશાંશ));

    auto eps_const = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{1}, {eps});
    auto clamped_norm = std::make_shared<ov::op::v1::Maximum>(l2_norm, eps_const);

    auto res = std::make_shared<ov::op::v1::Divide>(આગત_ગાંઠ, clamped_norm);

    પાછા rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
