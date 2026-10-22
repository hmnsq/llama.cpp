#అన్తర్భావ "../node_context.h"
#అన్తర్భావ "../op_table.h"
#అన్తర్భావ "../utils.h"

#అన్తర్భావ <openvino/op/broadcast.hpp>
#అన్తర్భావ <openvino/op/constant.hpp>

నామాకాశ ov {
నామాకాశ frontend {
నామాకాశ ggml {
నామాకాశ op {

// GGML FILL sets all elements of a tensor to a constant value.
// The constant is stored as a float in op_params[0].
Output_సదిశ translate_fill(స్థిరరాశి కణుపు_Context & context) {
    అంకె_inputs_check(context, 1, 1);

    దశాంశం c;
    జ్ఞాపకం_నకలు(&c, context.get_output_op_params(), పరిమాణం(దశాంశం));

    స్వయంచాలక shape = context.get_ఆదానం_shape(0).గమ్యం_shape();

    స్వయంచాలక val = ov::op::v0::Constant::create(context.get_output_type(), {}, {c});
    స్వయంచాలక లక్ష్యం_shape = ov::op::v0::Constant::create(ov::element::i64, {shape.size()},
        std::vector<int64_t>(shape.begin(), shape.end()));
    స్వయంచాలక res = std::make_shared<ov::op::v3::ప్రసారం>(val, లక్ష్యం_shape);

    తిరిగి rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
