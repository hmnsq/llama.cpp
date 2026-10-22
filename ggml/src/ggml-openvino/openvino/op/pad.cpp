#అన్తర్భావ "../op_table.h"
#అన్తర్భావ "../utils.h"

#అన్తర్భావ <array>
#అన్తర్భావ <cstdint>
#అన్తర్భావ <openvino/op/constant.hpp>
#అన్తర్భావ <openvino/op/gather.hpp>
#అన్తర్భావ <openvino/op/pad.hpp>
#అన్తర్భావ <openvino/op/reshape.hpp>
#అన్తర్భావ <openvino/op/shape_of.hpp>
#అన్తర్భావ <utility>
#అన్తర్భావ <vector>

నామాకాశ ov {
నామాకాశ frontend {
నామాకాశ ggml {
నామాకాశ op {

నామాకాశ {

ov::Output<ov::Node> translate_circular_pad(ov::Output<ov::Node> ఆదానం,
                                            స్థిరరాశి std::array<int32_t, 8> & pads,
                                            స్థిరరాశి ov::Shape & ఆదానం_shape) {
    ov::Output<ov::Node> ఫలితం = std::move(ఆదానం);

    స్థిరరాశి std::array<int32_t, 4> pads_begin = {pads[6], pads[4], pads[2], pads[0]};
    స్థిరరాశి std::array<int32_t, 4> pads_చివర = {pads[7], pads[5], pads[3], pads[1]};

    కొరకు (size_t axis = 0; axis < ఆదానం_shape.size(); ++axis) {
        స్థిరరాశి int64_t ఆదానం_dim = static_cast<int64_t>(ఆదానం_shape[axis]);
        స్థిరరాశి int64_t pad_begin = pads_begin[axis];
        స్థిరరాశి int64_t pad_చివర = pads_చివర[axis];

        ఒకవేళ (pad_begin == 0 && pad_చివర == 0) {
            కొనసాగు;
        }

        FRONT_చివర_CHECK_IMPLEMENTED(ఆదానం_dim > 0, "Circular PAD requires static non-zero input dimensions");

        std::vector<int64_t> indices(static_cast<size_t>(ఆదానం_dim + pad_begin + pad_చివర));
        కొరకు (int64_t సూచీ = 0; సూచీ < static_cast<int64_t>(indices.size()); ++సూచీ) {
            int64_t wrapped = (సూచీ - pad_begin) % ఆదానం_dim;
            ఒకవేళ (wrapped < 0) {
                wrapped += ఆదానం_dim;
            }
            indices[static_cast<size_t>(సూచీ)] = wrapped;
        }

        స్వయంచాలక gather_indices = ov::op::v0::Constant::create(ov::element::i64, {indices.size()}, indices);
        స్వయంచాలక gather_axis = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {axis});
        ఫలితం = std::make_shared<ov::op::v8::Gather>(ఫలితం, gather_indices, gather_axis);
    }

    తిరిగి ఫలితం;
}

}  // namespace

Output_సదిశ translate_pad(స్థిరరాశి కణుపు_Context & context) {
    అంకె_inputs_check(context, 1, 1);

    స్వయంచాలక ఆదానం = process_view_ఆదానం_new(context, 0);
    ఒకవేళ (context.get_ఆదానం_shape(0) == context.get_output_shape()) {
        తిరిగి rename_outputs_with_suffix({ఆదానం}, context.get_name());
    }

    స్థిరరాశి int32_t * op_params = context.get_output_op_params();
    FRONT_చివర_CHECK_IMPLEMENTED(op_params != శూన్యసూచిక, "PAD requires output op params");

    స్థిరరాశి std::array<int32_t, 8> pads = {op_params[0], op_params[1], op_params[2], op_params[3],
                                         op_params[4], op_params[5], op_params[6], op_params[7]};
    స్థిరరాశి తార్కిక circular = op_params[8] != 0;

    ఒకవేళ (circular) {
        స్వయంచాలక res = translate_circular_pad(ఆదానం, pads, context.get_ఆదానం_shape(0).గమ్యం_shape());
        తిరిగి rename_outputs_with_suffix({res}, context.get_name());
    }

    స్థిరరాశి std::vector<int64_t> pads_begin = {pads[6], pads[4], pads[2], pads[0]};
    స్థిరరాశి std::vector<int64_t> pads_చివర = {pads[7], pads[5], pads[3], pads[1]};

    స్వయంచాలక pads_begin_కణుపు = ov::op::v0::Constant::create(ov::element::i64, {pads_begin.size()}, pads_begin);
    స్వయంచాలక pads_చివర_కణుపు = ov::op::v0::Constant::create(ov::element::i64, {pads_చివర.size()}, pads_చివర);
    స్వయంచాలక pad_విలువ = ov::op::v0::Constant::create(context.get_ఆదానం_type(0), ov::Shape{}, {0});
    స్వయంచాలక res =
        std::make_shared<ov::op::v1::Pad>(ఆదానం, pads_begin_కణుపు, pads_చివర_కణుపు, pad_విలువ, ov::op::Pad_బహుళకం::CONSTANT);

    తిరిగి rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
