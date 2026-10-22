#અન્તર્ભાવ "../node_context.h"
#અન્તર્ભાવ "../op_table.h"
#અન્તર્ભાવ "../utils.h"

#અન્તર્ભાવ <memory>
#અન્તર્ભાવ <openvino/core/node.hpp>
#અન્તર્ભાવ <openvino/core/node_output.hpp>
#અન્તર્ભાવ <openvino/op/add.hpp>
#અન્તર્ભાવ <openvino/op/constant.hpp>
#અન્તર્ભાવ <openvino/op/convert.hpp>
#અન્તર્ભાવ <openvino/op/gather.hpp>
#અન્તર્ભાવ <openvino/op/reshape.hpp>
#અન્તર્ભાવ <openvino/op/shape_of.hpp>

નામાકાશ ov {
નામાકાશ frontend {
નામાકાશ ggml {
નામાકાશ op {

સ્થિર ov::Output<ov::Node> reshape_add_id_આગત_to_2d(અચલ ov::Output<ov::Node> & આગત,
                                                       અચલ ov::PartialShape & આગત_shape,
                                                       અચલ std::vector<પૂર્ણાંક> & dims) {
    અચલ auto & actual_shape = આગત.get_partial_shape();
    જો (actual_shape.rank().is_static() && actual_shape.rank().get_length() == 2) {
        પાછા આગત;
    }

    જો (આગત_shape.rank().is_static() && આગત_shape.rank().get_length() == 2) {
        પાછા આગત;
    }

    auto shape = std::make_shared<ov::op::v3::ShapeOf>(આગત, ov::element::i64);
    પાછા std::make_shared<ov::op::v1::Reshape>(આગત, get_dimensions(shape, dims), ખોટું);
}

OutputVector translate_add_id(અચલ ગાંઠ_Context & context) {
    અંક_inputs_check(context, 3, 3);

    auto આગત = process_view_આગત_new(context, 0);
    auto bias = process_view_આગત_new(context, 1);
    auto ids = process_view_આગત_new(context, 2);

    // OpenVINO uses reversed GGML dimensions:
    //   input: [1, n_token, n_used, n_embd]
    //   bias:  [1, 1, n_expert, n_embd]
    //   ids:   [1, 1, n_token, n_used]
    // Model bias constants may already be stored as [n_expert, n_embd].
    bias = reshape_add_id_આગત_to_2d(bias, context.get_આગત_shape(1), {2, 3});
    ids = reshape_add_id_આગત_to_2d(ids, context.get_આગત_shape(2), {2, 3});

    જો (ids.get_element_type() != ov::element::i32 && ids.get_element_type() != ov::element::i64) {
        ids = std::make_shared<ov::op::v0::Convert>(ids, ov::element::i32);
    }

    auto gather_axis = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{}, {0});
    ov::Output<ov::Node> selected_bias = std::make_shared<ov::op::v8::Gather>(bias, ids, gather_axis);
    selected_bias = std::make_shared<ov::op::v1::Reshape>(
        selected_bias, std::make_shared<ov::op::v3::ShapeOf>(આગત, ov::element::i64), ખોટું);

    જો (selected_bias.get_element_type() != આગત.get_element_type()) {
        selected_bias = std::make_shared<ov::op::v0::Convert>(selected_bias, આગત.get_element_type());
    }

    ov::Output<ov::Node> res = std::make_shared<ov::op::v1::Add>(આગત, selected_bias);
    અચલ auto output_type = context.get_output_type();
    જો (res.get_element_type() != output_type) {
        res = std::make_shared<ov::op::v0::Convert>(res, output_type);
    }

    પાછા rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
