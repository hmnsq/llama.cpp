#అన్తర్భావ "../node_context.h"
#అన్తర్భావ "../op_table.h"
#అన్తర్భావ "../utils.h"

#అన్తర్భావ <memory>
#అన్తర్భావ <openvino/core/node.hpp>
#అన్తర్భావ <openvino/core/node_output.hpp>
#అన్తర్భావ <openvino/op/add.hpp>
#అన్తర్భావ <openvino/op/constant.hpp>
#అన్తర్భావ <openvino/op/convert.hpp>
#అన్తర్భావ <openvino/op/gather.hpp>
#అన్తర్భావ <openvino/op/reshape.hpp>
#అన్తర్భావ <openvino/op/shape_of.hpp>

నామాకాశ ov {
నామాకాశ frontend {
నామాకాశ ggml {
నామాకాశ op {

స్థిర ov::Output<ov::Node> reshape_కూడు_id_ఆదానం_గమ్యం_2d(స్థిరరాశి ov::Output<ov::Node> & ఆదానం,
                                                       స్థిరరాశి ov::PartialShape & ఆదానం_shape,
                                                       స్థిరరాశి std::vector<పూర్ణాంకం> & dims) {
    స్థిరరాశి స్వయంచాలక & actual_shape = ఆదానం.get_partial_shape();
    ఒకవేళ (actual_shape.rank().is_static() && actual_shape.rank().get_నిడివి() == 2) {
        తిరిగి ఆదానం;
    }

    ఒకవేళ (ఆదానం_shape.rank().is_static() && ఆదానం_shape.rank().get_నిడివి() == 2) {
        తిరిగి ఆదానం;
    }

    స్వయంచాలక shape = std::make_shared<ov::op::v3::Shape_యొక్క>(ఆదానం, ov::element::i64);
    తిరిగి std::make_shared<ov::op::v1::Reshape>(ఆదానం, get_dimensions(shape, dims), అబద్ధం);
}

Output_సదిశ translate_కూడు_id(స్థిరరాశి కణుపు_Context & context) {
    అంకె_inputs_check(context, 3, 3);

    స్వయంచాలక ఆదానం = process_view_ఆదానం_new(context, 0);
    స్వయంచాలక bias = process_view_ఆదానం_new(context, 1);
    స్వయంచాలక ids = process_view_ఆదానం_new(context, 2);

    // OpenVINO uses reversed GGML dimensions:
    //   input: [1, n_token, n_used, n_embd]
    //   bias:  [1, 1, n_expert, n_embd]
    //   ids:   [1, 1, n_token, n_used]
    // Model bias constants may already be stored as [n_expert, n_embd].
    bias = reshape_కూడు_id_ఆదానం_గమ్యం_2d(bias, context.get_ఆదానం_shape(1), {2, 3});
    ids = reshape_కూడు_id_ఆదానం_గమ్యం_2d(ids, context.get_ఆదానం_shape(2), {2, 3});

    ఒకవేళ (ids.get_element_type() != ov::element::i32 && ids.get_element_type() != ov::element::i64) {
        ids = std::make_shared<ov::op::v0::Convert>(ids, ov::element::i32);
    }

    స్వయంచాలక gather_axis = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{}, {0});
    ov::Output<ov::Node> selected_bias = std::make_shared<ov::op::v8::Gather>(bias, ids, gather_axis);
    selected_bias = std::make_shared<ov::op::v1::Reshape>(
        selected_bias, std::make_shared<ov::op::v3::Shape_యొక్క>(ఆదానం, ov::element::i64), అబద్ధం);

    ఒకవేళ (selected_bias.get_element_type() != ఆదానం.get_element_type()) {
        selected_bias = std::make_shared<ov::op::v0::Convert>(selected_bias, ఆదానం.get_element_type());
    }

    ov::Output<ov::Node> res = std::make_shared<ov::op::v1::కూడు>(ఆదానం, selected_bias);
    స్థిరరాశి స్వయంచాలక output_type = context.get_output_type();
    ఒకవేళ (res.get_element_type() != output_type) {
        res = std::make_shared<ov::op::v0::Convert>(res, output_type);
    }

    తిరిగి rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
