#ଅନ୍ତର୍ଭାଵ "../node_context.h"
#ଅନ୍ତର୍ଭାଵ "../op_table.h"
#ଅନ୍ତର୍ଭାଵ "../utils.h"

#ଅନ୍ତର୍ଭାଵ <memory>
#ଅନ୍ତର୍ଭାଵ <openvino/op/constant.hpp>
#ଅନ୍ତର୍ଭାଵ <openvino/op/divide.hpp>
#ଅନ୍ତର୍ଭାଵ <openvino/op/maximum.hpp>
#ଅନ୍ତର୍ଭାଵ <openvino/op/multiply.hpp>
#ଅନ୍ତର୍ଭାଵ <openvino/op/reduce_sum.hpp>
#ଅନ୍ତର୍ଭାଵ <openvino/op/reshape.hpp>
#ଅନ୍ତର୍ଭାଵ <openvino/op/sqrt.hpp>
#ଅନ୍ତର୍ଭାଵ <openvino/op/squeeze.hpp>

ନାମାକାଶ ov {
ନାମାକାଶ frontend {
ନାମାକାଶ ggml {
ନାମାକାଶ op {

OutputVector translate_l2_norm(ଅଚଳ ଗଣ୍ଠି_Context & context) {
    ଅଙ୍କ_inputs_check(context, 1, 1);

    auto ନିବେଶ_ଗଣ୍ଠି = process_view_ନିବେଶ_new(context, 0);

    ଯଦି (context.get_op_case() == 1) {
        // 92: [   128,    16,     1,     2] VIEW                 q_conv-1
        //      [  6144,     1,     2,     1]            0: UNARY       conv_output_silu-1
        // 93: [   128,    16,     1,     2] L2_NORM              q_conv_predelta-1
        //      [   128,    16,     1,     2]            0: VIEW        q_conv-1
        auto output_shape = context.get_output_shape().to_shape();
        ନିବେଶ_ଗଣ୍ଠି = process_view_ନିବେଶ(context, 0, output_shape[2] * output_shape[3], 3);
        ନିବେଶ_ଗଣ୍ଠି =
            std::make_shared<ov::op::v0::Squeeze>(ନିବେଶ_ଗଣ୍ଠି, ov::op::v0::Constant::create(ov::element::i64, {1}, {0}));

        std::vector<int64_t> reshape_pattern = {0, 0, (int64_t) output_shape[2], (int64_t) output_shape[3]};
        ନିବେଶ_ଗଣ୍ଠି = std::make_shared<ov::op::v1::Reshape>(
            ନିବେଶ_ଗଣ୍ଠି, ov::op::v0::Constant::create(ov::element::i64, {4}, reshape_pattern), ସତ୍ୟ);
    }

    auto squared = std::make_shared<ov::op::v1::Multiply>(ନିବେଶ_ଗଣ୍ଠି, ନିବେଶ_ଗଣ୍ଠି);

    auto sum_squared = std::make_shared<ov::op::v1::ReduceSum>(
        squared, ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {-1}), ସତ୍ୟ);

    auto l2_norm = std::make_shared<ov::op::v0::Sqrt>(sum_squared);

    ଦଶମିକ eps;
    memcpy(&eps, context.get_output_op_params(), ଆକାର(ଦଶମିକ));

    auto eps_const = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{1}, {eps});
    auto clamped_norm = std::make_shared<ov::op::v1::Maximum>(l2_norm, eps_const);

    auto res = std::make_shared<ov::op::v1::Divide>(ନିବେଶ_ଗଣ୍ଠି, clamped_norm);

    ଫେରନ୍ତୁ rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
