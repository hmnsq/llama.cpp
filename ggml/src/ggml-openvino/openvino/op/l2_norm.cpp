#অন্তর্ভাব "../node_context.h"
#অন্তর্ভাব "../op_table.h"
#অন্তর্ভাব "../utils.h"

#অন্তর্ভাব <memory>
#অন্তর্ভাব <openvino/op/constant.hpp>
#অন্তর্ভাব <openvino/op/divide.hpp>
#অন্তর্ভাব <openvino/op/maximum.hpp>
#অন্তর্ভাব <openvino/op/multiply.hpp>
#অন্তর্ভাব <openvino/op/reduce_sum.hpp>
#অন্তর্ভাব <openvino/op/reshape.hpp>
#অন্তর্ভাব <openvino/op/sqrt.hpp>
#অন্তর্ভাব <openvino/op/squeeze.hpp>

নামাকাশ ov {
নামাকাশ frontend {
নামাকাশ ggml {
নামাকাশ op {

OutputVector translate_l2_norm(ধ্রুব গ্রন্থি_Context & context) {
    অঙ্ক_inputs_check(context, 1, 1);

    auto প্রবেশ_গ্রন্থি = process_view_প্রবেশ_new(context, 0);

    যদি (context.get_op_case() == 1) {
        // 92: [   128,    16,     1,     2] VIEW                 q_conv-1
        //      [  6144,     1,     2,     1]            0: UNARY       conv_output_silu-1
        // 93: [   128,    16,     1,     2] L2_NORM              q_conv_predelta-1
        //      [   128,    16,     1,     2]            0: VIEW        q_conv-1
        auto output_shape = context.get_output_shape().to_shape();
        প্রবেশ_গ্রন্থি = process_view_প্রবেশ(context, 0, output_shape[2] * output_shape[3], 3);
        প্রবেশ_গ্রন্থি =
            std::make_shared<ov::op::v0::Squeeze>(প্রবেশ_গ্রন্থি, ov::op::v0::Constant::create(ov::element::i64, {1}, {0}));

        std::vector<int64_t> reshape_pattern = {0, 0, (int64_t) output_shape[2], (int64_t) output_shape[3]};
        প্রবেশ_গ্রন্থি = std::make_shared<ov::op::v1::Reshape>(
            প্রবেশ_গ্রন্থি, ov::op::v0::Constant::create(ov::element::i64, {4}, reshape_pattern), সত্য);
    }

    auto squared = std::make_shared<ov::op::v1::Multiply>(প্রবেশ_গ্রন্থি, প্রবেশ_গ্রন্থি);

    auto sum_squared = std::make_shared<ov::op::v1::ReduceSum>(
        squared, ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {-1}), সত্য);

    auto l2_norm = std::make_shared<ov::op::v0::Sqrt>(sum_squared);

    ভিন্ন eps;
    memcpy(&eps, context.get_output_op_params(), পরিমাণ(ভিন্ন));

    auto eps_const = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{1}, {eps});
    auto clamped_norm = std::make_shared<ov::op::v1::Maximum>(l2_norm, eps_const);

    auto res = std::make_shared<ov::op::v1::Divide>(প্রবেশ_গ্রন্থি, clamped_norm);

    প্রত্যাগম rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
