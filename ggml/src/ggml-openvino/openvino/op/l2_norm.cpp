#അന്തര്ഭാവ "../node_context.h"
#അന്തര്ഭാവ "../op_table.h"
#അന്തര്ഭാവ "../utils.h"

#അന്തര്ഭാവ <memory>
#അന്തര്ഭാവ <openvino/op/constant.hpp>
#അന്തര്ഭാവ <openvino/op/divide.hpp>
#അന്തര്ഭാവ <openvino/op/maximum.hpp>
#അന്തര്ഭാവ <openvino/op/multiply.hpp>
#അന്തര്ഭാവ <openvino/op/reduce_sum.hpp>
#അന്തര്ഭാവ <openvino/op/reshape.hpp>
#അന്തര്ഭാവ <openvino/op/sqrt.hpp>
#അന്തര്ഭാവ <openvino/op/squeeze.hpp>

നാമാകാശ ov {
നാമാകാശ frontend {
നാമാകാശ ggml {
നാമാകാശ op {

OutputVector translate_l2_norm(സ്ഥിരം കണ്ണി_Context & context) {
    അക്കം_inputs_check(context, 1, 1);

    auto നിവേശം_കണ്ണി = process_view_നിവേശം_new(context, 0);

    എങ്കിൽ (context.get_op_case() == 1) {
        // 92: [   128,    16,     1,     2] VIEW                 q_conv-1
        //      [  6144,     1,     2,     1]            0: UNARY       conv_output_silu-1
        // 93: [   128,    16,     1,     2] L2_NORM              q_conv_predelta-1
        //      [   128,    16,     1,     2]            0: VIEW        q_conv-1
        auto output_shape = context.get_output_shape().to_shape();
        നിവേശം_കണ്ണി = process_view_നിവേശം(context, 0, output_shape[2] * output_shape[3], 3);
        നിവേശം_കണ്ണി =
            std::make_shared<ov::op::v0::Squeeze>(നിവേശം_കണ്ണി, ov::op::v0::Constant::create(ov::element::i64, {1}, {0}));

        std::vector<int64_t> reshape_pattern = {0, 0, (int64_t) output_shape[2], (int64_t) output_shape[3]};
        നിവേശം_കണ്ണി = std::make_shared<ov::op::v1::Reshape>(
            നിവേശം_കണ്ണി, ov::op::v0::Constant::create(ov::element::i64, {4}, reshape_pattern), സത്യം);
    }

    auto squared = std::make_shared<ov::op::v1::Multiply>(നിവേശം_കണ്ണി, നിവേശം_കണ്ണി);

    auto sum_squared = std::make_shared<ov::op::v1::ReduceSum>(
        squared, ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {-1}), സത്യം);

    auto l2_norm = std::make_shared<ov::op::v0::Sqrt>(sum_squared);

    ദശാംശം eps;
    memcpy(&eps, context.get_output_op_params(), വലുപ്പം(ദശാംശം));

    auto eps_const = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{1}, {eps});
    auto clamped_norm = std::make_shared<ov::op::v1::Maximum>(l2_norm, eps_const);

    auto res = std::make_shared<ov::op::v1::Divide>(നിവേശം_കണ്ണി, clamped_norm);

    തിരികെ rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
