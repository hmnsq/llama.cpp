#అన్తర్భావ "../node_context.h"
#అన్తర్భావ "../op_table.h"
#అన్తర్భావ "../utils.h"

#అన్తర్భావ <memory>
#అన్తర్భావ <openvino/op/constant.hpp>
#అన్తర్భావ <openvino/op/divide.hpp>
#అన్తర్భావ <openvino/op/maximum.hpp>
#అన్తర్భావ <openvino/op/multiply.hpp>
#అన్తర్భావ <openvino/op/reduce_sum.hpp>
#అన్తర్భావ <openvino/op/reshape.hpp>
#అన్తర్భావ <openvino/op/sqrt.hpp>
#అన్తర్భావ <openvino/op/squeeze.hpp>

నామాకాశ ov {
నామాకాశ frontend {
నామాకాశ ggml {
నామాకాశ op {

Output_సదిశ translate_l2_norm(స్థిరరాశి కణుపు_Context & context) {
    అంకె_inputs_check(context, 1, 1);

    స్వయంచాలక ఆదానం_కణుపు = process_view_ఆదానం_new(context, 0);

    ఒకవేళ (context.get_op_case() == 1) {
        // 92: [   128,    16,     1,     2] VIEW                 q_conv-1
        //      [  6144,     1,     2,     1]            0: UNARY       conv_output_silu-1
        // 93: [   128,    16,     1,     2] L2_NORM              q_conv_predelta-1
        //      [   128,    16,     1,     2]            0: VIEW        q_conv-1
        స్వయంచాలక output_shape = context.get_output_shape().గమ్యం_shape();
        ఆదానం_కణుపు = process_view_ఆదానం(context, 0, output_shape[2] * output_shape[3], 3);
        ఆదానం_కణుపు =
            std::make_shared<ov::op::v0::Squeeze>(ఆదానం_కణుపు, ov::op::v0::Constant::create(ov::element::i64, {1}, {0}));

        std::vector<int64_t> reshape_నమూనా = {0, 0, (int64_t) output_shape[2], (int64_t) output_shape[3]};
        ఆదానం_కణుపు = std::make_shared<ov::op::v1::Reshape>(
            ఆదానం_కణుపు, ov::op::v0::Constant::create(ov::element::i64, {4}, reshape_నమూనా), నిజం);
    }

    స్వయంచాలక squared = std::make_shared<ov::op::v1::Multiply>(ఆదానం_కణుపు, ఆదానం_కణుపు);

    స్వయంచాలక కూడిక_squared = std::make_shared<ov::op::v1::Reduce_కూడిక>(
        squared, ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {-1}), నిజం);

    స్వయంచాలక l2_norm = std::make_shared<ov::op::v0::Sqrt>(కూడిక_squared);

    దశాంశం eps;
    జ్ఞాపకం_నకలు(&eps, context.get_output_op_params(), పరిమాణం(దశాంశం));

    స్వయంచాలక eps_const = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{1}, {eps});
    స్వయంచాలక clamped_norm = std::make_shared<ov::op::v1::Maximum>(l2_norm, eps_const);

    స్వయంచాలక res = std::make_shared<ov::op::v1::Divide>(ఆదానం_కణుపు, clamped_norm);

    తిరిగి rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
