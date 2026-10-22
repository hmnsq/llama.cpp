#অন্তর্ভাব "../node_context.h"
#অন্তর্ভাব "../op_table.h"
#অন্তর্ভাব "../utils.h"

#অন্তর্ভাব <memory>
#অন্তর্ভাব <openvino/core/node.hpp>
#অন্তর্ভাব <openvino/core/node_output.hpp>
#অন্তর্ভাব <openvino/op/add.hpp>
#অন্তর্ভাব <openvino/op/constant.hpp>
#অন্তর্ভাব <openvino/op/convert.hpp>
#অন্তর্ভাব <openvino/op/gather.hpp>
#অন্তর্ভাব <openvino/op/reshape.hpp>
#অন্তর্ভাব <openvino/op/shape_of.hpp>

নামাকাশ ov {
নামাকাশ frontend {
নামাকাশ ggml {
নামাকাশ op {

স্থির ov::Output<ov::Node> reshape_add_id_প্রবেশ_to_2d(ধ্রুব ov::Output<ov::Node> & প্রবেশ,
                                                       ধ্রুব ov::PartialShape & প্রবেশ_shape,
                                                       ধ্রুব std::vector<পূর্ণাঙ্ক> & dims) {
    ধ্রুব auto & actual_shape = প্রবেশ.get_partial_shape();
    যদি (actual_shape.rank().is_static() && actual_shape.rank().get_length() == 2) {
        প্রত্যাগম প্রবেশ;
    }

    যদি (প্রবেশ_shape.rank().is_static() && প্রবেশ_shape.rank().get_length() == 2) {
        প্রত্যাগম প্রবেশ;
    }

    auto shape = std::make_shared<ov::op::v3::ShapeOf>(প্রবেশ, ov::element::i64);
    প্রত্যাগম std::make_shared<ov::op::v1::Reshape>(প্রবেশ, get_dimensions(shape, dims), অসত্য);
}

OutputVector translate_add_id(ধ্রুব গ্রন্থি_Context & context) {
    অঙ্ক_inputs_check(context, 3, 3);

    auto প্রবেশ = process_view_প্রবেশ_new(context, 0);
    auto bias = process_view_প্রবেশ_new(context, 1);
    auto ids = process_view_প্রবেশ_new(context, 2);

    // OpenVINO uses reversed GGML dimensions:
    //   input: [1, n_token, n_used, n_embd]
    //   bias:  [1, 1, n_expert, n_embd]
    //   ids:   [1, 1, n_token, n_used]
    // Model bias constants may already be stored as [n_expert, n_embd].
    bias = reshape_add_id_প্রবেশ_to_2d(bias, context.get_প্রবেশ_shape(1), {2, 3});
    ids = reshape_add_id_প্রবেশ_to_2d(ids, context.get_প্রবেশ_shape(2), {2, 3});

    যদি (ids.get_element_type() != ov::element::i32 && ids.get_element_type() != ov::element::i64) {
        ids = std::make_shared<ov::op::v0::Convert>(ids, ov::element::i32);
    }

    auto gather_axis = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{}, {0});
    ov::Output<ov::Node> selected_bias = std::make_shared<ov::op::v8::Gather>(bias, ids, gather_axis);
    selected_bias = std::make_shared<ov::op::v1::Reshape>(
        selected_bias, std::make_shared<ov::op::v3::ShapeOf>(প্রবেশ, ov::element::i64), অসত্য);

    যদি (selected_bias.get_element_type() != প্রবেশ.get_element_type()) {
        selected_bias = std::make_shared<ov::op::v0::Convert>(selected_bias, প্রবেশ.get_element_type());
    }

    ov::Output<ov::Node> res = std::make_shared<ov::op::v1::Add>(প্রবেশ, selected_bias);
    ধ্রুব auto output_type = context.get_output_type();
    যদি (res.get_element_type() != output_type) {
        res = std::make_shared<ov::op::v0::Convert>(res, output_type);
    }

    প্রত্যাগম rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
