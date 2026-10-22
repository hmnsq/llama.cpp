#అన్తర్భావ "../node_context.h"
#అన్తర్భావ "../op_table.h"
#అన్తర్భావ "../utils.h"

#అన్తర్భావ <cstring>
#అన్తర్భావ <memory>
#అన్తర్భావ <openvino/op/constant.hpp>
#అన్తర్భావ <openvino/op/mvn.hpp>

నామాకాశ ov {
నామాకాశ frontend {
నామాకాశ ggml {
నామాకాశ op {

Output_సదిశ translate_norm(స్థిరరాశి కణుపు_Context & context) {
    అంకె_inputs_check(context, 1, 1);

    స్వయంచాలక ఆదానం_కణుపు = process_view_ఆదానం_new(context, 0);
    దశాంశం eps;
    జ్ఞాపకం_నకలు(&eps, context.get_output_op_params(), పరిమాణం(దశాంశం));

    స్వయంచాలక axes = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {-1});
    స్వయంచాలక res = std::make_shared<ov::op::v6::MVN>(ఆదానం_కణుపు, axes, నిజం, eps, ov::op::MVNEps_బహుళకం::INSIDE_SQRT);

    తిరిగి rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
