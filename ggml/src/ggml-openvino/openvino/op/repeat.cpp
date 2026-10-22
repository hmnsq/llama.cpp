#అన్తర్భావ "../node_context.h"
#అన్తర్భావ "../op_table.h"
#అన్తర్భావ "../utils.h"

#అన్తర్భావ <memory>
#అన్తర్భావ <openvino/op/broadcast.hpp>
#అన్తర్భావ <openvino/op/constant.hpp>
#అన్తర్భావ <openvino/op/divide.hpp>
#అన్తర్భావ <openvino/op/shape_of.hpp>
#అన్తర్భావ <openvino/op/tile.hpp>
#అన్తర్భావ <vector>

నామాకాశ ov {
నామాకాశ frontend {
నామాకాశ ggml {
నామాకాశ op {

// GGML_OP_REPEAT tiles src[0] to fill the destination shape. Every destination
// dimension is an integer multiple of the corresponding source dimension.
Output_సదిశ translate_repeat(స్థిరరాశి కణుపు_Context & context) {
    అంకె_inputs_check(context, 1, 2);

    స్వయంచాలక ఆదానం = process_view_ఆదానం_new(context, 0);

    స్థిరరాశి స్వయంచాలక ఆదానం_shape = context.get_ఆదానం_shape(0).గమ్యం_shape();
    స్థిరరాశి స్వయంచాలక output_shape = context.get_output_shape().గమ్యం_shape();

    std::vector<int64_t> repeats(4, 1);
    కొరకు (size_t axis = 0; axis < 4; ++axis) {
        స్థిరరాశి int64_t ఆదానం_dim = ఆదానం_shape[axis];
        స్థిరరాశి int64_t output_dim = output_shape[axis];

        FRONT_చివర_OP_CONVERSION_CHECK(ఆదానం_dim > 0 && output_dim > 0 && output_dim % ఆదానం_dim == 0,
                                      "REPEAT input shape ", ఆదానం_shape, " cannot tile to match ", output_shape);

        repeats[axis] = output_dim / ఆదానం_dim;
    }

    స్వయంచాలక repeats_కణుపు = ov::op::v0::Constant::create(ov::element::i64, {repeats.size()}, repeats);
    ov::Output<ov::Node> res = std::make_shared<ov::op::v0::Tile>(ఆదానం, repeats_కణుపు);
    తిరిగి rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
