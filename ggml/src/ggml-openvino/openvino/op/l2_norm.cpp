#समावेश "../node_context.h"
#समावेश "../op_table.h"
#समावेश "../utils.h"

#समावेश <memory>
#समावेश <openvino/op/constant.hpp>
#समावेश <openvino/op/divide.hpp>
#समावेश <openvino/op/maximum.hpp>
#समावेश <openvino/op/multiply.hpp>
#समावेश <openvino/op/reduce_sum.hpp>
#समावेश <openvino/op/reshape.hpp>
#समावेश <openvino/op/sqrt.hpp>
#समावेश <openvino/op/squeeze.hpp>

namespace ov {
namespace frontend {
namespace ggml {
namespace op {

OutputVector translate_l2_norm(स्थिर गाँठ_Context & context) {
    अंक_inputs_check(context, 1, 1);

    स्वतः निवेश_गाँठ = process_view_निवेश_new(context, 0);

    अगर (context.get_op_case() == 1) {
        // 92: [   128,    16,     1,     2] VIEW                 q_conv-1
        //      [  6144,     1,     2,     1]            0: UNARY       conv_output_silu-1
        // 93: [   128,    16,     1,     2] L2_NORM              q_conv_predelta-1
        //      [   128,    16,     1,     2]            0: VIEW        q_conv-1
        स्वतः output_shape = context.get_output_shape().to_shape();
        निवेश_गाँठ = process_view_निवेश(context, 0, output_shape[2] * output_shape[3], 3);
        निवेश_गाँठ =
            std::make_shared<ov::op::v0::Squeeze>(निवेश_गाँठ, ov::op::v0::Constant::create(ov::element::i64, {1}, {0}));

        std::vector<int64_t> reshape_pattern = {0, 0, (int64_t) output_shape[2], (int64_t) output_shape[3]};
        निवेश_गाँठ = std::make_shared<ov::op::v1::Reshape>(
            निवेश_गाँठ, ov::op::v0::Constant::create(ov::element::i64, {4}, reshape_pattern), true);
    }

    स्वतः squared = std::make_shared<ov::op::v1::Multiply>(निवेश_गाँठ, निवेश_गाँठ);

    स्वतः योग_squared = std::make_shared<ov::op::v1::Reduce_योग>(
        squared, ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {-1}), true);

    स्वतः l2_norm = std::make_shared<ov::op::v0::Sqrt>(योग_squared);

    भग्न eps;
    स_नकल(&eps, context.get_output_op_params(), माप(भग्न));

    स्वतः eps_const = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{1}, {eps});
    स्वतः clamped_norm = std::make_shared<ov::op::v1::Maximum>(l2_norm, eps_const);

    स्वतः res = std::make_shared<ov::op::v1::Divide>(निवेश_गाँठ, clamped_norm);

    वापस rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
