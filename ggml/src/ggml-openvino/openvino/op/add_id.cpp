#அந்தர்பாவ "../node_context.h"
#அந்தர்பாவ "../op_table.h"
#அந்தர்பாவ "../utils.h"

#அந்தர்பாவ <memory>
#அந்தர்பாவ <openvino/core/node.hpp>
#அந்தர்பாவ <openvino/core/node_output.hpp>
#அந்தர்பாவ <openvino/op/add.hpp>
#அந்தர்பாவ <openvino/op/constant.hpp>
#அந்தர்பாவ <openvino/op/convert.hpp>
#அந்தர்பாவ <openvino/op/gather.hpp>
#அந்தர்பாவ <openvino/op/reshape.hpp>
#அந்தர்பாவ <openvino/op/shape_of.hpp>

நாமாகாஶ ov {
நாமாகாஶ frontend {
நாமாகாஶ ggml {
நாமாகாஶ op {

நிலையான ov::Output<ov::Node> reshape_add_id_உள்ளீடு_to_2d(மாறிலி ov::Output<ov::Node> & உள்ளீடு,
                                                       மாறிலி ov::PartialShape & உள்ளீடு_shape,
                                                       மாறிலி std::vector<முழு> & dims) {
    மாறிலி auto & actual_shape = உள்ளீடு.get_partial_shape();
    எனில் (actual_shape.rank().is_static() && actual_shape.rank().get_length() == 2) {
        திரும்பு உள்ளீடு;
    }

    எனில் (உள்ளீடு_shape.rank().is_static() && உள்ளீடு_shape.rank().get_length() == 2) {
        திரும்பு உள்ளீடு;
    }

    auto shape = std::make_shared<ov::op::v3::ShapeOf>(உள்ளீடு, ov::element::i64);
    திரும்பு std::make_shared<ov::op::v1::Reshape>(உள்ளீடு, get_dimensions(shape, dims), பொய்);
}

OutputVector translate_add_id(மாறிலி முனை_Context & context) {
    இலக்கம்_inputs_check(context, 3, 3);

    auto உள்ளீடு = process_view_உள்ளீடு_new(context, 0);
    auto bias = process_view_உள்ளீடு_new(context, 1);
    auto ids = process_view_உள்ளீடு_new(context, 2);

    // OpenVINO uses reversed GGML dimensions:
    //   input: [1, n_token, n_used, n_embd]
    //   bias:  [1, 1, n_expert, n_embd]
    //   ids:   [1, 1, n_token, n_used]
    // Model bias constants may already be stored as [n_expert, n_embd].
    bias = reshape_add_id_உள்ளீடு_to_2d(bias, context.get_உள்ளீடு_shape(1), {2, 3});
    ids = reshape_add_id_உள்ளீடு_to_2d(ids, context.get_உள்ளீடு_shape(2), {2, 3});

    எனில் (ids.get_element_type() != ov::element::i32 && ids.get_element_type() != ov::element::i64) {
        ids = std::make_shared<ov::op::v0::Convert>(ids, ov::element::i32);
    }

    auto gather_axis = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{}, {0});
    ov::Output<ov::Node> selected_bias = std::make_shared<ov::op::v8::Gather>(bias, ids, gather_axis);
    selected_bias = std::make_shared<ov::op::v1::Reshape>(
        selected_bias, std::make_shared<ov::op::v3::ShapeOf>(உள்ளீடு, ov::element::i64), பொய்);

    எனில் (selected_bias.get_element_type() != உள்ளீடு.get_element_type()) {
        selected_bias = std::make_shared<ov::op::v0::Convert>(selected_bias, உள்ளீடு.get_element_type());
    }

    ov::Output<ov::Node> res = std::make_shared<ov::op::v1::Add>(உள்ளீடு, selected_bias);
    மாறிலி auto output_type = context.get_output_type();
    எனில் (res.get_element_type() != output_type) {
        res = std::make_shared<ov::op::v0::Convert>(res, output_type);
    }

    திரும்பு rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
