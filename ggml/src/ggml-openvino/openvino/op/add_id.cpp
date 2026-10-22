#شامل "../node_context.h"
#شامل "../op_table.h"
#شامل "../utils.h"

#شامل <memory>
#شامل <openvino/core/node.hpp>
#شامل <openvino/core/node_output.hpp>
#شامل <openvino/op/add.hpp>
#شامل <openvino/op/constant.hpp>
#شامل <openvino/op/convert.hpp>
#شامل <openvino/op/gather.hpp>
#شامل <openvino/op/reshape.hpp>
#شامل <openvino/op/shape_of.hpp>

فضاینام ov {
فضاینام frontend {
فضاینام ggml {
فضاینام op {

ایستا ov::Output<ov::Node> reshape_add_id_درون_داد_to_2d(ثابت ov::Output<ov::Node> & درون_داد,
                                                       ثابت ov::PartialShape & درون_داد_shape,
                                                       ثابت std::vector<صحیح> & dims) {
    ثابت auto & actual_shape = درون_داد.get_partial_shape();
    اگر (actual_shape.rank().is_static() && actual_shape.rank().get_length() == 2) {
        بازگشت درون_داد;
    }

    اگر (درون_داد_shape.rank().is_static() && درون_داد_shape.rank().get_length() == 2) {
        بازگشت درون_داد;
    }

    auto shape = std::make_shared<ov::op::v3::ShapeOf>(درون_داد, ov::element::i64);
    بازگشت std::make_shared<ov::op::v1::Reshape>(درون_داد, get_dimensions(shape, dims), نادرست);
}

OutputVector translate_add_id(ثابت گره_Context & context) {
    رقم_inputs_check(context, 3, 3);

    auto درون_داد = process_view_درون_داد_new(context, 0);
    auto bias = process_view_درون_داد_new(context, 1);
    auto ids = process_view_درون_داد_new(context, 2);

    // OpenVINO uses reversed GGML dimensions:
    //   input: [1, n_token, n_used, n_embd]
    //   bias:  [1, 1, n_expert, n_embd]
    //   ids:   [1, 1, n_token, n_used]
    // Model bias constants may already be stored as [n_expert, n_embd].
    bias = reshape_add_id_درون_داد_to_2d(bias, context.get_درون_داد_shape(1), {2, 3});
    ids = reshape_add_id_درون_داد_to_2d(ids, context.get_درون_داد_shape(2), {2, 3});

    اگر (ids.get_element_type() != ov::element::i32 && ids.get_element_type() != ov::element::i64) {
        ids = std::make_shared<ov::op::v0::Convert>(ids, ov::element::i32);
    }

    auto gather_axis = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{}, {0});
    ov::Output<ov::Node> selected_bias = std::make_shared<ov::op::v8::Gather>(bias, ids, gather_axis);
    selected_bias = std::make_shared<ov::op::v1::Reshape>(
        selected_bias, std::make_shared<ov::op::v3::ShapeOf>(درون_داد, ov::element::i64), نادرست);

    اگر (selected_bias.get_element_type() != درون_داد.get_element_type()) {
        selected_bias = std::make_shared<ov::op::v0::Convert>(selected_bias, درون_داد.get_element_type());
    }

    ov::Output<ov::Node> res = std::make_shared<ov::op::v1::Add>(درون_داد, selected_bias);
    ثابت auto output_type = context.get_output_type();
    اگر (res.get_element_type() != output_type) {
        res = std::make_shared<ov::op::v0::Convert>(res, output_type);
    }

    بازگشت rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
