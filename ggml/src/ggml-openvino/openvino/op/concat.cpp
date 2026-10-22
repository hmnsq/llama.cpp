#అన్తర్భావ "../node_context.h"
#అన్తర్భావ "../op_table.h"
#అన్తర్భావ "../utils.h"

#అన్తర్భావ <memory>
#అన్తర్భావ <openvino/frontend/exception.hpp>
#అన్తర్భావ <openvino/op/concat.hpp>
#అన్తర్భావ <openvino/op/convert.hpp>

నామాకాశ ov {
నామాకాశ frontend {
నామాకాశ ggml {
నామాకాశ op {

Output_సదిశ translate_concat(స్థిరరాశి కణుపు_Context & context) {
    అంకె_inputs_check(context, 2, 2);

    స్థిరరాశి int32_t * op_params = context.get_output_op_params();
    FRONT_చివర_CHECK_IMPLEMENTED(op_params != శూన్యసూచిక, "CONCAT requires output op params");

    స్థిరరాశి స్వయంచాలక output_shape = context.get_output_shape();
    FRONT_చివర_CHECK_IMPLEMENTED(output_shape.rank().is_static(), "CONCAT requires static output rank");

    స్థిరరాశి స్వయంచాలక rank = output_shape.rank().get_నిడివి();
    స్థిరరాశి int32_t ggml_dim = op_params[0];
    FRONT_చివర_CHECK_IMPLEMENTED(ggml_dim >= 0 && ggml_dim < rank, "CONCAT axis is out of range");

    స్వయంచాలక ఆదానం_0 = process_view_ఆదానం_new(context, 0);
    స్వయంచాలక ఆదానం_1 = process_view_ఆదానం_new(context, 1);
    స్థిరరాశి స్వయంచాలక output_type = context.get_output_type();

    ఒకవేళ (ఆదానం_0.get_element_type() != output_type) {
        ఆదానం_0 = std::make_shared<ov::op::v0::Convert>(ఆదానం_0, output_type);
    }
    ఒకవేళ (ఆదానం_1.get_element_type() != output_type) {
        ఆదానం_1 = std::make_shared<ov::op::v0::Convert>(ఆదానం_1, output_type);
    }

    స్థిరరాశి స్వయంచాలక axis = static_cast<int64_t>(rank - 1 - ggml_dim);
    స్వయంచాలక res = std::make_shared<ov::op::v0::Concat>(Output_సదిశ{ఆదానం_0, ఆదానం_1}, axis);

    తిరిగి rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
