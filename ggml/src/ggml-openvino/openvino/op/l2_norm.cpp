#ಅನ್ತರ್ಭಾವ "../node_context.h"
#ಅನ್ತರ್ಭಾವ "../op_table.h"
#ಅನ್ತರ್ಭಾವ "../utils.h"

#ಅನ್ತರ್ಭಾವ <memory>
#ಅನ್ತರ್ಭಾವ <openvino/op/constant.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/divide.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/maximum.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/multiply.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/reduce_sum.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/reshape.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/sqrt.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/squeeze.hpp>

ನಾಮಾಕಾಶ ov {
ನಾಮಾಕಾಶ frontend {
ನಾಮಾಕಾಶ ggml {
ನಾಮಾಕಾಶ op {

OutputVector translate_l2_norm(ಧ್ರುವ ಗಂಟು_Context & context) {
    ಅಂಕೆ_inputs_check(context, 1, 1);

    auto ಒಳಹರಿವು_ಗಂಟು = process_view_ಒಳಹರಿವು_new(context, 0);

    ಯದಿ (context.get_op_case() == 1) {
        // 92: [   128,    16,     1,     2] VIEW                 q_conv-1
        //      [  6144,     1,     2,     1]            0: UNARY       conv_output_silu-1
        // 93: [   128,    16,     1,     2] L2_NORM              q_conv_predelta-1
        //      [   128,    16,     1,     2]            0: VIEW        q_conv-1
        auto output_shape = context.get_output_shape().to_shape();
        ಒಳಹರಿವು_ಗಂಟು = process_view_ಒಳಹರಿವು(context, 0, output_shape[2] * output_shape[3], 3);
        ಒಳಹರಿವು_ಗಂಟು =
            std::make_shared<ov::op::v0::Squeeze>(ಒಳಹರಿವು_ಗಂಟು, ov::op::v0::Constant::create(ov::element::i64, {1}, {0}));

        std::vector<int64_t> reshape_pattern = {0, 0, (int64_t) output_shape[2], (int64_t) output_shape[3]};
        ಒಳಹರಿವು_ಗಂಟು = std::make_shared<ov::op::v1::Reshape>(
            ಒಳಹರಿವು_ಗಂಟು, ov::op::v0::Constant::create(ov::element::i64, {4}, reshape_pattern), ಸತ್ಯ);
    }

    auto squared = std::make_shared<ov::op::v1::Multiply>(ಒಳಹರಿವು_ಗಂಟು, ಒಳಹರಿವು_ಗಂಟು);

    auto sum_squared = std::make_shared<ov::op::v1::ReduceSum>(
        squared, ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {-1}), ಸತ್ಯ);

    auto l2_norm = std::make_shared<ov::op::v0::Sqrt>(sum_squared);

    ಭಿನ್ನ eps;
    memcpy(&eps, context.get_output_op_params(), ಪರಿಮಾಣ(ಭಿನ್ನ));

    auto eps_const = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{1}, {eps});
    auto clamped_norm = std::make_shared<ov::op::v1::Maximum>(l2_norm, eps_const);

    auto res = std::make_shared<ov::op::v1::Divide>(ಒಳಹರಿವು_ಗಂಟು, clamped_norm);

    ಪ್ರತ್ಯಾಗಮ rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
