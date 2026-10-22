#అన్తర్భావ "../node_context.h"
#అన్తర్భావ "../op_table.h"
#అన్తర్భావ "../utils.h"

#అన్తర్భావ <openvino/op/avg_pool.hpp>
#అన్తర్భావ <openvino/op/max_pool.hpp>
#అన్తర్భావ <openvino/op/convert.hpp>

నామాకాశ ov {
నామాకాశ frontend {
నామాకాశ ggml {
నామాకాశ op {

Output_సదిశ translate_pool_2d(స్థిరరాశి కణుపు_Context & context) {
    అంకె_inputs_check(context, 1, 1);
    స్థిరరాశి int32_t * params = context.get_output_op_params();

    స్థిరరాశి పూర్ణాంకం k0 = params[1];
    స్థిరరాశి పూర్ణాంకం k1 = params[2];
    స్థిరరాశి పూర్ణాంకం s0 = params[3];
    స్థిరరాశి పూర్ణాంకం s1 = params[4];
    స్థిరరాశి పూర్ణాంకం p0 = params[5];
    స్థిరరాశి పూర్ణాంకం p1 = params[6];

    స్థిరరాశి పూర్ణాంకం op_case = context.get_op_case();
    ov::Output<Node> ఆదానం = context.get_ఆదానం(0);
    ov::Strides strides{static_cast<size_t>(s1), static_cast<size_t>(s0)};
    ov::Shape pads_begin{static_cast<size_t>(p1), static_cast<size_t>(p0)};
    ov::Shape pads_చివర{static_cast<size_t>(p1), static_cast<size_t>(p0)};
    ov::Shape kernel{static_cast<size_t>(k1), static_cast<size_t>(k0)};
    ov::Output<Node> res;

    ఎంపిక (op_case) {
    స్థితి 1:  // GGML_OP_POOL_MAX
    {
        res = std::make_shared<ov::op::v1::MaxPool>(ఆదానం, strides, pads_begin, pads_చివర, kernel);
        ఆపు;
    }
    స్థితి 2:  // GGML_OP_POOL_AVG
    {
        res = std::make_shared<ov::op::v1::AvgPool>(ఆదానం, strides, pads_begin, pads_చివర, kernel, అబద్ధం);
        ఆపు;
    }
    అప్రమేయ:
        ఆపు;
    }
    తిరిగి rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
