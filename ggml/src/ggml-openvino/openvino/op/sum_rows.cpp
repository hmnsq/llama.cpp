#అన్తర్భావ "../node_context.h"
#అన్తర్భావ "../op_table.h"
#అన్తర్భావ "../utils.h"

#అన్తర్భావ <memory>
#అన్తర్భావ <openvino/op/constant.hpp>
#అన్తర్భావ <openvino/op/reduce_sum.hpp>

నామాకాశ ov {
నామాకాశ frontend {
నామాకాశ ggml {
నామాకాశ op {

Output_సదిశ translate_కూడిక_అడ్డువరుసలు(స్థిరరాశి కణుపు_Context & context) {
    అంకె_inputs_check(context, 1, 1);

    స్వయంచాలక ఆదానం = process_view_ఆదానం_new(context, 0);
    స్వయంచాలక res = std::make_shared<ov::op::v1::Reduce_కూడిక>(
        ఆదానం, ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {-1}), నిజం);

    తిరిగి rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
