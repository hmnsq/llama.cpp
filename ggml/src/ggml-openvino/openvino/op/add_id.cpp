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

स्थिर ov::Output<ov::Node> reshape_add_id_आदान_to_2d(स्थिरांक ov::Output<ov::Node> & आदान,
                                                       स्थिरांक ov::PartialShape & आदान_shape,
                                                       स्थिरांक std::vector<पूर्णांक> & dims) {
    स्थिरांक स्वतः & actual_shape = आदान.get_partial_shape();
    जर (actual_shape.rank().is_static() && actual_shape.rank().get_length() == 2) {
        परत आदान;
    }

    जर (आदान_shape.rank().is_static() && आदान_shape.rank().get_length() == 2) {
        परत आदान;
    }

    स्वतः shape = std::make_shared<ov::op::v3::ShapeOf>(आदान, ov::element::i64);
    परत std::make_shared<ov::op::v1::Reshape>(आदान, get_dimensions(shape, dims), असत्य);
}

OutputVector translate_add_id(स्थिरांक गाठ_Context & context) {
    अंक_inputs_check(context, 3, 3);

    स्वतः आदान = process_view_आदान_new(context, 0);
    स्वतः bias = process_view_आदान_new(context, 1);
    स्वतः ids = process_view_आदान_new(context, 2);

    // OpenVINO uses reversed GGML dimensions:
    //   input: [1, n_token, n_used, n_embd]
    //   bias:  [1, 1, n_expert, n_embd]
    //   ids:   [1, 1, n_token, n_used]
    // Model bias constants may already be stored as [n_expert, n_embd].
    bias = reshape_add_id_आदान_to_2d(bias, context.get_आदान_shape(1), {2, 3});
    ids = reshape_add_id_आदान_to_2d(ids, context.get_आदान_shape(2), {2, 3});

    जर (ids.get_element_type() != ov::element::i32 && ids.get_element_type() != ov::element::i64) {
        ids = std::make_shared<ov::op::v0::Convert>(ids, ov::element::i32);
    }

    स्वतः gather_axis = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{}, {0});
    ov::Output<ov::Node> selected_bias = std::make_shared<ov::op::v8::Gather>(bias, ids, gather_axis);
    selected_bias = std::make_shared<ov::op::v1::Reshape>(
        selected_bias, std::make_shared<ov::op::v3::ShapeOf>(आदान, ov::element::i64), असत्य);

    जर (selected_bias.get_element_type() != आदान.get_element_type()) {
        selected_bias = std::make_shared<ov::op::v0::Convert>(selected_bias, आदान.get_element_type());
    }

    ov::Output<ov::Node> res = std::make_shared<ov::op::v1::Add>(आदान, selected_bias);
    स्थिरांक स्वतः output_type = context.get_output_type();
    जर (res.get_element_type() != output_type) {
        res = std::make_shared<ov::op::v0::Convert>(res, output_type);
    }

    परत rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
