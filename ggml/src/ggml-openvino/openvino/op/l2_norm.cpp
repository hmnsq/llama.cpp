#ضمن "../node_context.h"
#ضمن "../op_table.h"
#ضمن "../utils.h"

#ضمن <memory>
#ضمن <openvino/op/constant.hpp>
#ضمن <openvino/op/divide.hpp>
#ضمن <openvino/op/maximum.hpp>
#ضمن <openvino/op/multiply.hpp>
#ضمن <openvino/op/reduce_sum.hpp>
#ضمن <openvino/op/reshape.hpp>
#ضمن <openvino/op/sqrt.hpp>
#ضمن <openvino/op/squeeze.hpp>

نطاق ov {
نطاق frontend {
نطاق ggml {
نطاق op {

Output_ويڪٽر translate_l2_norm(ثابت ڳنڍ_Context & context) {
    انگ_inputs_check(context, 1, 1);

    خودڪار داخلا_ڳنڍ = process_view_داخلا_new(context, 0);

    إذا (context.get_op_case() == 1) {
        // 92: [   128,    16,     1,     2] VIEW                 q_conv-1
        //      [  6144,     1,     2,     1]            0: UNARY       conv_output_silu-1
        // 93: [   128,    16,     1,     2] L2_NORM              q_conv_predelta-1
        //      [   128,    16,     1,     2]            0: VIEW        q_conv-1
        خودڪار output_shape = context.get_output_shape().منزل_shape();
        داخلا_ڳنڍ = process_view_داخلا(context, 0, output_shape[2] * output_shape[3], 3);
        داخلا_ڳنڍ =
            std::make_shared<ov::op::v0::Squeeze>(داخلا_ڳنڍ, ov::op::v0::Constant::create(ov::element::i64, {1}, {0}));

        std::vector<int64_t> reshape_نمونو = {0, 0, (int64_t) output_shape[2], (int64_t) output_shape[3]};
        داخلا_ڳنڍ = std::make_shared<ov::op::v1::Reshape>(
            داخلا_ڳنڍ, ov::op::v0::Constant::create(ov::element::i64, {4}, reshape_نمونو), سچ);
    }

    خودڪار squared = std::make_shared<ov::op::v1::Multiply>(داخلا_ڳنڍ, داخلا_ڳنڍ);

    خودڪار جوڙ_squared = std::make_shared<ov::op::v1::Reduce_جوڙ>(
        squared, ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {-1}), سچ);

    خودڪار l2_norm = std::make_shared<ov::op::v0::Sqrt>(جوڙ_squared);

    عائم eps;
    يادگيري_نقل(&eps, context.get_output_op_params(), حجم(عائم));

    خودڪار eps_const = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{1}, {eps});
    خودڪار clamped_norm = std::make_shared<ov::op::v1::Maximum>(l2_norm, eps_const);

    خودڪار res = std::make_shared<ov::op::v1::Divide>(داخلا_ڳنڍ, clamped_norm);

    أرجع rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
