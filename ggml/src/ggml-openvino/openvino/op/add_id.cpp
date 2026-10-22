#अन्तर्भाव "../node_context.h"
#अन्तर्भाव "../op_table.h"
#अन्तर्भाव "../utils.h"

#अन्तर्भाव <memory>
#अन्तर्भाव <openvino/core/node.hpp>
#अन्तर्भाव <openvino/core/node_output.hpp>
#अन्तर्भाव <openvino/op/add.hpp>
#अन्तर्भाव <openvino/op/constant.hpp>
#अन्तर्भाव <openvino/op/convert.hpp>
#अन्तर्भाव <openvino/op/gather.hpp>
#अन्तर्भाव <openvino/op/reshape.hpp>
#अन्तर्भाव <openvino/op/shape_of.hpp>

नामाकाश ov {
नामाकाश frontend {
नामाकाश ggml {
नामाकाश op {

स्थिर ov::Output<ov::Node> reshape_add_id_आदान_to_2d(ध्रुव ov::Output<ov::Node> & आदान,
                                                       ध्रुव ov::PartialShape & आदान_shape,
                                                       ध्रुव std::vector<पूर्णाङ्क> & dims) {
    ध्रुव auto & actual_shape = आदान.get_partial_shape();
    यदि (actual_shape.rank().is_static() && actual_shape.rank().get_length() == 2) {
        प्रत्यागम आदान;
    }

    यदि (आदान_shape.rank().is_static() && आदान_shape.rank().get_length() == 2) {
        प्रत्यागम आदान;
    }

    auto shape = std::make_shared<ov::op::v3::ShapeOf>(आदान, ov::element::i64);
    प्रत्यागम std::make_shared<ov::op::v1::Reshape>(आदान, get_dimensions(shape, dims), असत्य);
}

OutputVector translate_add_id(ध्रुव गाठ_Context & context) {
    अंक_inputs_check(context, 3, 3);

    auto आदान = process_view_आदान_new(context, 0);
    auto bias = process_view_आदान_new(context, 1);
    auto ids = process_view_आदान_new(context, 2);

    // OpenVINO uses reversed GGML dimensions:
    //   input: [1, n_token, n_used, n_embd]
    //   bias:  [1, 1, n_expert, n_embd]
    //   ids:   [1, 1, n_token, n_used]
    // Model bias constants may already be stored as [n_expert, n_embd].
    bias = reshape_add_id_आदान_to_2d(bias, context.get_आदान_shape(1), {2, 3});
    ids = reshape_add_id_आदान_to_2d(ids, context.get_आदान_shape(2), {2, 3});

    यदि (ids.get_element_type() != ov::element::i32 && ids.get_element_type() != ov::element::i64) {
        ids = std::make_shared<ov::op::v0::Convert>(ids, ov::element::i32);
    }

    auto gather_axis = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{}, {0});
    ov::Output<ov::Node> selected_bias = std::make_shared<ov::op::v8::Gather>(bias, ids, gather_axis);
    selected_bias = std::make_shared<ov::op::v1::Reshape>(
        selected_bias, std::make_shared<ov::op::v3::ShapeOf>(आदान, ov::element::i64), असत्य);

    यदि (selected_bias.get_element_type() != आदान.get_element_type()) {
        selected_bias = std::make_shared<ov::op::v0::Convert>(selected_bias, आदान.get_element_type());
    }

    ov::Output<ov::Node> res = std::make_shared<ov::op::v1::Add>(आदान, selected_bias);
    ध्रुव auto output_type = context.get_output_type();
    यदि (res.get_element_type() != output_type) {
        res = std::make_shared<ov::op::v0::Convert>(res, output_type);
    }

    प्रत्यागम rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
