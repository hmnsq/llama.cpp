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

Output_وکتور translate_l2_norm(ثابت گره_Context & context) {
    رقم_inputs_check(context, 1, 1);

    خودکار درون_داد_گره = process_view_درون_داد_new(context, 0);

    إذا (context.get_op_case() == 1) {
        // 92: [   128,    16,     1,     2] VIEW                 q_conv-1
        //      [  6144,     1,     2,     1]            0: UNARY       conv_output_silu-1
        // 93: [   128,    16,     1,     2] L2_NORM              q_conv_predelta-1
        //      [   128,    16,     1,     2]            0: VIEW        q_conv-1
        خودکار output_shape = context.get_output_shape().مقصد_shape();
        درون_داد_گره = process_view_درون_داد(context, 0, output_shape[2] * output_shape[3], 3);
        درون_داد_گره =
            std::make_shared<ov::op::v0::Squeeze>(درون_داد_گره, ov::op::v0::Constant::create(ov::element::i64, {1}, {0}));

        std::vector<int64_t> reshape_الگو = {0, 0, (int64_t) output_shape[2], (int64_t) output_shape[3]};
        درون_داد_گره = std::make_shared<ov::op::v1::Reshape>(
            درون_داد_گره, ov::op::v0::Constant::create(ov::element::i64, {4}, reshape_الگو), درست);
    }

    خودکار squared = std::make_shared<ov::op::v1::Multiply>(درون_داد_گره, درون_داد_گره);

    خودکار جمع_squared = std::make_shared<ov::op::v1::Reduce_جمع>(
        squared, ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {-1}), درست);

    خودکار l2_norm = std::make_shared<ov::op::v0::Sqrt>(جمع_squared);

    عائم eps;
    کپی_حافظه(&eps, context.get_output_op_params(), حجم(عائم));

    خودکار eps_const = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{1}, {eps});
    خودکار clamped_norm = std::make_shared<ov::op::v1::Maximum>(l2_norm, eps_const);

    خودکار res = std::make_shared<ov::op::v1::Divide>(درون_داد_گره, clamped_norm);

    أرجع rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
