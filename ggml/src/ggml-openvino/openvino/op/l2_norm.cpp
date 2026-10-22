#अन्तर्भाव "../node_context.h"
#अन्तर्भाव "../op_table.h"
#अन्तर्भाव "../utils.h"

#अन्तर्भाव <memory>
#अन्तर्भाव <openvino/op/constant.hpp>
#अन्तर्भाव <openvino/op/divide.hpp>
#अन्तर्भाव <openvino/op/maximum.hpp>
#अन्तर्भाव <openvino/op/multiply.hpp>
#अन्तर्भाव <openvino/op/reduce_sum.hpp>
#अन्तर्भाव <openvino/op/reshape.hpp>
#अन्तर्भाव <openvino/op/sqrt.hpp>
#अन्तर्भाव <openvino/op/squeeze.hpp>

नामाकाश ov {
नामाकाश frontend {
नामाकाश ggml {
नामाकाश op {

Output_सदिशम् translate_l2_norm(ध्रुव ग्रन्थिः_Context & context) {
    अङ्कः_inputs_check(context, 1, 1);

    स्वचालित निवेशः_ग्रन्थिः = process_view_निवेशः_new(context, 0);

    यदि (context.get_op_case() == 1) {
        // 92: [   128,    16,     1,     2] VIEW                 q_conv-1
        //      [  6144,     1,     2,     1]            0: UNARY       conv_output_silu-1
        // 93: [   128,    16,     1,     2] L2_NORM              q_conv_predelta-1
        //      [   128,    16,     1,     2]            0: VIEW        q_conv-1
        स्वचालित output_shape = context.get_output_shape().गन्तव्यम्_shape();
        निवेशः_ग्रन्थिः = process_view_निवेशः(context, 0, output_shape[2] * output_shape[3], 3);
        निवेशः_ग्रन्थिः =
            std::make_shared<ov::op::v0::Squeeze>(निवेशः_ग्रन्थिः, ov::op::v0::Constant::create(ov::element::i64, {1}, {0}));

        std::vector<int64_t> reshape_आदर्शः = {0, 0, (int64_t) output_shape[2], (int64_t) output_shape[3]};
        निवेशः_ग्रन्थिः = std::make_shared<ov::op::v1::Reshape>(
            निवेशः_ग्रन्थिः, ov::op::v0::Constant::create(ov::element::i64, {4}, reshape_आदर्शः), सत्य);
    }

    स्वचालित squared = std::make_shared<ov::op::v1::Multiply>(निवेशः_ग्रन्थिः, निवेशः_ग्रन्थिः);

    स्वचालित योग_squared = std::make_shared<ov::op::v1::Reduce_योग>(
        squared, ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {-1}), सत्य);

    स्वचालित l2_norm = std::make_shared<ov::op::v0::Sqrt>(योग_squared);

    भिन्न eps;
    स्मृतिं_प्रतिलिख(&eps, context.get_output_op_params(), परिमाण(भिन्न));

    स्वचालित eps_const = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{1}, {eps});
    स्वचालित clamped_norm = std::make_shared<ov::op::v1::Maximum>(l2_norm, eps_const);

    स्वचालित res = std::make_shared<ov::op::v1::Divide>(निवेशः_ग्रन्थिः, clamped_norm);

    प्रत्यागम rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
