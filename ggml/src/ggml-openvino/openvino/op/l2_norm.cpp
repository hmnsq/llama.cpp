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

OutputVector translate_l2_norm(अचर गाँठो_Context & context) {
    अङ्क_inputs_check(context, 1, 1);

    स्वतः आगत_गाँठो = process_view_आगत_new(context, 0);

    यदि (context.get_op_case() == 1) {
        // 92: [   128,    16,     1,     2] VIEW                 q_conv-1
        //      [  6144,     1,     2,     1]            0: UNARY       conv_output_silu-1
        // 93: [   128,    16,     1,     2] L2_NORM              q_conv_predelta-1
        //      [   128,    16,     1,     2]            0: VIEW        q_conv-1
        स्वतः output_shape = context.get_output_shape().to_shape();
        आगत_गाँठो = process_view_आगत(context, 0, output_shape[2] * output_shape[3], 3);
        आगत_गाँठो =
            std::make_shared<ov::op::v0::Squeeze>(आगत_गाँठो, ov::op::v0::Constant::create(ov::element::i64, {1}, {0}));

        std::vector<int64_t> reshape_pattern = {0, 0, (int64_t) output_shape[2], (int64_t) output_shape[3]};
        आगत_गाँठो = std::make_shared<ov::op::v1::Reshape>(
            आगत_गाँठो, ov::op::v0::Constant::create(ov::element::i64, {4}, reshape_pattern), सत्य);
    }

    स्वतः squared = std::make_shared<ov::op::v1::Multiply>(आगत_गाँठो, आगत_गाँठो);

    स्वतः sum_squared = std::make_shared<ov::op::v1::ReduceSum>(
        squared, ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {-1}), सत्य);

    स्वतः l2_norm = std::make_shared<ov::op::v0::Sqrt>(sum_squared);

    भिन्न eps;
    स_नकल(&eps, context.get_output_op_params(), परिमाण(भिन्न));

    स्वतः eps_const = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{1}, {eps});
    स्वतः clamped_norm = std::make_shared<ov::op::v1::Maximum>(l2_norm, eps_const);

    स्वतः res = std::make_shared<ov::op::v1::Divide>(आगत_गाँठो, clamped_norm);

    फर्काऊ rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
