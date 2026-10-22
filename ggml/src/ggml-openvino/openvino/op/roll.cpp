#అన్తర్భావ "../node_context.h"
#అన్తర్భావ "../op_table.h"
#అన్తర్భావ "../utils.h"

#అన్తర్భావ <openvino/op/roll.hpp>
#అన్తర్భావ <openvino/op/constant.hpp>

నామాకాశ ov {
నామాకాశ frontend {
నామాకాశ ggml {
నామాకాశ op {

Output_సదిశ translate_roll(స్థిరరాశి కణుపు_Context & context) {
    అంకె_inputs_check(context, 1, 1);
    స్థిరరాశి int32_t * params = context.get_output_op_params();

    int64_t s0 = params[0];
    int64_t s1 = params[1];
    int64_t s2 = params[2];
    int64_t s3 = params[3];

    స్వయంచాలక ఆదానం = context.get_ఆదానం(0);

    స్వయంచాలక జరుపు = ov::op::v0::Constant::create(
        ov::element::i64, ov::Shape{4}, std::vector<int64_t>{s3, s2, s1, s0});
    స్వయంచాలక axes  = ov::op::v0::Constant::create(
        ov::element::i64, ov::Shape{4}, std::vector<int64_t>{0, 1, 2, 3});

    స్వయంచాలక roll = std::make_shared<ov::op::v7::Roll>(ఆదానం, జరుపు, axes);
    తిరిగి rename_outputs_with_suffix({roll}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
