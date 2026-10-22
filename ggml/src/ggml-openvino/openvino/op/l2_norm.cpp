#ਅਨ੍ਤਰ੍ਭਾਵ "../node_context.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "../op_table.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "../utils.h"

#ਅਨ੍ਤਰ੍ਭਾਵ <memory>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/constant.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/divide.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/maximum.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/multiply.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/reduce_sum.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/reshape.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/sqrt.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/squeeze.hpp>

ਨਾਮਾਕਾਸ਼ ov {
ਨਾਮਾਕਾਸ਼ frontend {
ਨਾਮਾਕਾਸ਼ ggml {
ਨਾਮਾਕਾਸ਼ op {

Output_ਸਦਿਸ਼ translate_l2_norm(ਅਚਲ ਗੰਢ_Context & context) {
    ਅੰਕ_inputs_check(context, 1, 1);

    ਸਵੈਚਾਲਿਤ ਨਿਵੇਸ਼_ਗੰਢ = process_view_ਨਿਵੇਸ਼_new(context, 0);

    ਜੇ (context.get_op_case() == 1) {
        // 92: [   128,    16,     1,     2] VIEW                 q_conv-1
        //      [  6144,     1,     2,     1]            0: UNARY       conv_output_silu-1
        // 93: [   128,    16,     1,     2] L2_NORM              q_conv_predelta-1
        //      [   128,    16,     1,     2]            0: VIEW        q_conv-1
        ਸਵੈਚਾਲਿਤ output_shape = context.get_output_shape().ਮੰਜ਼ਿਲ_shape();
        ਨਿਵੇਸ਼_ਗੰਢ = process_view_ਨਿਵੇਸ਼(context, 0, output_shape[2] * output_shape[3], 3);
        ਨਿਵੇਸ਼_ਗੰਢ =
            std::make_shared<ov::op::v0::Squeeze>(ਨਿਵੇਸ਼_ਗੰਢ, ov::op::v0::Constant::create(ov::element::i64, {1}, {0}));

        std::vector<int64_t> reshape_ਨਮੂਨਾ = {0, 0, (int64_t) output_shape[2], (int64_t) output_shape[3]};
        ਨਿਵੇਸ਼_ਗੰਢ = std::make_shared<ov::op::v1::Reshape>(
            ਨਿਵੇਸ਼_ਗੰਢ, ov::op::v0::Constant::create(ov::element::i64, {4}, reshape_ਨਮੂਨਾ), ਸੱਚ);
    }

    ਸਵੈਚਾਲਿਤ squared = std::make_shared<ov::op::v1::Multiply>(ਨਿਵੇਸ਼_ਗੰਢ, ਨਿਵੇਸ਼_ਗੰਢ);

    ਸਵੈਚਾਲਿਤ ਜੋੜ_squared = std::make_shared<ov::op::v1::Reduce_ਜੋੜ>(
        squared, ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {-1}), ਸੱਚ);

    ਸਵੈਚਾਲਿਤ l2_norm = std::make_shared<ov::op::v0::Sqrt>(ਜੋੜ_squared);

    ਦਸ਼ਮਲਵ eps;
    ਸਮ੍ਰਿਤੀ_ਨਕਲ(&eps, context.get_output_op_params(), ਆਕਾਰ(ਦਸ਼ਮਲਵ));

    ਸਵੈਚਾਲਿਤ eps_const = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{1}, {eps});
    ਸਵੈਚਾਲਿਤ clamped_norm = std::make_shared<ov::op::v1::Maximum>(l2_norm, eps_const);

    ਸਵੈਚਾਲਿਤ res = std::make_shared<ov::op::v1::Divide>(ਨਿਵੇਸ਼_ਗੰਢ, clamped_norm);

    ਵਾਪਸ rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
