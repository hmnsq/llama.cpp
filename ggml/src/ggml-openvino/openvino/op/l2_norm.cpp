#அந்தர்பாவ "../node_context.h"
#அந்தர்பாவ "../op_table.h"
#அந்தர்பாவ "../utils.h"

#அந்தர்பாவ <memory>
#அந்தர்பாவ <openvino/op/constant.hpp>
#அந்தர்பாவ <openvino/op/divide.hpp>
#அந்தர்பாவ <openvino/op/maximum.hpp>
#அந்தர்பாவ <openvino/op/multiply.hpp>
#அந்தர்பாவ <openvino/op/reduce_sum.hpp>
#அந்தர்பாவ <openvino/op/reshape.hpp>
#அந்தர்பாவ <openvino/op/sqrt.hpp>
#அந்தர்பாவ <openvino/op/squeeze.hpp>

நாமாகாஶ ov {
நாமாகாஶ frontend {
நாமாகாஶ ggml {
நாமாகாஶ op {

OutputVector translate_l2_norm(மாறிலி முனை_Context & context) {
    இலக்கம்_inputs_check(context, 1, 1);

    auto உள்ளீடு_முனை = process_view_உள்ளீடு_new(context, 0);

    எனில் (context.get_op_case() == 1) {
        // 92: [   128,    16,     1,     2] VIEW                 q_conv-1
        //      [  6144,     1,     2,     1]            0: UNARY       conv_output_silu-1
        // 93: [   128,    16,     1,     2] L2_NORM              q_conv_predelta-1
        //      [   128,    16,     1,     2]            0: VIEW        q_conv-1
        auto output_shape = context.get_output_shape().to_shape();
        உள்ளீடு_முனை = process_view_உள்ளீடு(context, 0, output_shape[2] * output_shape[3], 3);
        உள்ளீடு_முனை =
            std::make_shared<ov::op::v0::Squeeze>(உள்ளீடு_முனை, ov::op::v0::Constant::create(ov::element::i64, {1}, {0}));

        std::vector<int64_t> reshape_pattern = {0, 0, (int64_t) output_shape[2], (int64_t) output_shape[3]};
        உள்ளீடு_முனை = std::make_shared<ov::op::v1::Reshape>(
            உள்ளீடு_முனை, ov::op::v0::Constant::create(ov::element::i64, {4}, reshape_pattern), உண்மை);
    }

    auto squared = std::make_shared<ov::op::v1::Multiply>(உள்ளீடு_முனை, உள்ளீடு_முனை);

    auto sum_squared = std::make_shared<ov::op::v1::ReduceSum>(
        squared, ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {-1}), உண்மை);

    auto l2_norm = std::make_shared<ov::op::v0::Sqrt>(sum_squared);

    தசம eps;
    memcpy(&eps, context.get_output_op_params(), அளவு(தசம));

    auto eps_const = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{1}, {eps});
    auto clamped_norm = std::make_shared<ov::op::v1::Maximum>(l2_norm, eps_const);

    auto res = std::make_shared<ov::op::v1::Divide>(உள்ளீடு_முனை, clamped_norm);

    திரும்பு rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
