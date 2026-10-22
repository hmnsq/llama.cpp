#ضمن "../node_context.h"
#ضمن "../op_table.h"
#ضمن "../utils.h"

#ضمن <memory>
#ضمن <openvino/core/node.hpp>
#ضمن <openvino/core/node_output.hpp>
#ضمن <openvino/op/add.hpp>
#ضمن <openvino/op/constant.hpp>
#ضمن <openvino/op/convert.hpp>
#ضمن <openvino/op/gather.hpp>
#ضمن <openvino/op/reshape.hpp>
#ضمن <openvino/op/shape_of.hpp>

نطاق ov {
نطاق frontend {
نطاق ggml {
نطاق op {

ساكن ov::Output<ov::Node> reshape_add_id_داخلہ_to_2d(ثابت ov::Output<ov::Node> & داخلہ,
                                                       ثابت ov::PartialShape & داخلہ_shape,
                                                       ثابت std::vector<صحيح> & dims) {
    ثابت auto & actual_shape = داخلہ.get_partial_shape();
    إذا (actual_shape.rank().is_static() && actual_shape.rank().get_length() == 2) {
        أرجع داخلہ;
    }

    إذا (داخلہ_shape.rank().is_static() && داخلہ_shape.rank().get_length() == 2) {
        أرجع داخلہ;
    }

    auto shape = std::make_shared<ov::op::v3::ShapeOf>(داخلہ, ov::element::i64);
    أرجع std::make_shared<ov::op::v1::Reshape>(داخلہ, get_dimensions(shape, dims), خطأ);
}

OutputVector translate_add_id(ثابت گرہ_Context & context) {
    عدد_inputs_check(context, 3, 3);

    auto داخلہ = process_view_داخلہ_new(context, 0);
    auto bias = process_view_داخلہ_new(context, 1);
    auto ids = process_view_داخلہ_new(context, 2);

    // OpenVINO uses reversed GGML dimensions:
    //   input: [1, n_token, n_used, n_embd]
    //   bias:  [1, 1, n_expert, n_embd]
    //   ids:   [1, 1, n_token, n_used]
    // Model bias constants may already be stored as [n_expert, n_embd].
    bias = reshape_add_id_داخلہ_to_2d(bias, context.get_داخلہ_shape(1), {2, 3});
    ids = reshape_add_id_داخلہ_to_2d(ids, context.get_داخلہ_shape(2), {2, 3});

    إذا (ids.get_element_type() != ov::element::i32 && ids.get_element_type() != ov::element::i64) {
        ids = std::make_shared<ov::op::v0::Convert>(ids, ov::element::i32);
    }

    auto gather_axis = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{}, {0});
    ov::Output<ov::Node> selected_bias = std::make_shared<ov::op::v8::Gather>(bias, ids, gather_axis);
    selected_bias = std::make_shared<ov::op::v1::Reshape>(
        selected_bias, std::make_shared<ov::op::v3::ShapeOf>(داخلہ, ov::element::i64), خطأ);

    إذا (selected_bias.get_element_type() != داخلہ.get_element_type()) {
        selected_bias = std::make_shared<ov::op::v0::Convert>(selected_bias, داخلہ.get_element_type());
    }

    ov::Output<ov::Node> res = std::make_shared<ov::op::v1::Add>(داخلہ, selected_bias);
    ثابت auto output_type = context.get_output_type();
    إذا (res.get_element_type() != output_type) {
        res = std::make_shared<ov::op::v0::Convert>(res, output_type);
    }

    أرجع rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
