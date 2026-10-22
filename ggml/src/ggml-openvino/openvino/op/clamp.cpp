#అన్తర్భావ "../node_context.h"
#అన్తర్భావ "../op_table.h"
#అన్తర్భావ "../utils.h"

#అన్తర్భావ <cstring>
#అన్తర్భావ <openvino/op/clamp.hpp>

నామాకాశ ov {
నామాకాశ frontend {
నామాకాశ ggml {
నామాకాశ op {

Output_సదిశ translate_clamp(స్థిరరాశి కణుపు_Context & context) {
    అంకె_inputs_check(context, 1, 1);

    స్వయంచాలక ఆదానం = process_view_ఆదానం_new(context, 0);

    స్థిరరాశి int32_t * op_params = context.get_output_op_params();
    FRONT_చివర_CHECK_IMPLEMENTED(op_params != శూన్యసూచిక, "CLAMP requires output op params");

    దశాంశం min;
    దశాంశం max;
    std::జ్ఞాపకం_నకలు(&min, reinterpret_cast<స్థిరరాశి దశాంశం *>(op_params) + 0, పరిమాణం(దశాంశం));
    std::జ్ఞాపకం_నకలు(&max, reinterpret_cast<స్థిరరాశి దశాంశం *>(op_params) + 1, పరిమాణం(దశాంశం));

    స్వయంచాలక res = std::make_shared<ov::op::v0::Clamp>(ఆదానం, min, max);
    తిరిగి rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
