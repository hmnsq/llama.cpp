#അന്തര്ഭാവ "../node_context.h"
#അന്തര്ഭാവ "../op_table.h"
#അന്തര്ഭാവ "../utils.h"

#അന്തര്ഭാവ <memory>
#അന്തര്ഭാവ <openvino/core/node.hpp>
#അന്തര്ഭാവ <openvino/core/node_output.hpp>
#അന്തര്ഭാവ <openvino/op/add.hpp>
#അന്തര്ഭാവ <openvino/op/constant.hpp>
#അന്തര്ഭാവ <openvino/op/convert.hpp>
#അന്തര്ഭാവ <openvino/op/gather.hpp>
#അന്തര്ഭാവ <openvino/op/reshape.hpp>
#അന്തര്ഭാവ <openvino/op/shape_of.hpp>

നാമാകാശ ov {
നാമാകാശ frontend {
നാമാകാശ ggml {
നാമാകാശ op {

സ്ഥിതം ov::Output<ov::Node> reshape_add_id_നിവേശം_to_2d(സ്ഥിരം ov::Output<ov::Node> & നിവേശം,
                                                       സ്ഥിരം ov::PartialShape & നിവേശം_shape,
                                                       സ്ഥിരം std::vector<പൂർണ്ണസംഖ്യ> & dims) {
    സ്ഥിരം auto & actual_shape = നിവേശം.get_partial_shape();
    എങ്കിൽ (actual_shape.rank().is_static() && actual_shape.rank().get_length() == 2) {
        തിരികെ നിവേശം;
    }

    എങ്കിൽ (നിവേശം_shape.rank().is_static() && നിവേശം_shape.rank().get_length() == 2) {
        തിരികെ നിവേശം;
    }

    auto shape = std::make_shared<ov::op::v3::ShapeOf>(നിവേശം, ov::element::i64);
    തിരികെ std::make_shared<ov::op::v1::Reshape>(നിവേശം, get_dimensions(shape, dims), അസത്യം);
}

OutputVector translate_add_id(സ്ഥിരം കണ്ണി_Context & context) {
    അക്കം_inputs_check(context, 3, 3);

    auto നിവേശം = process_view_നിവേശം_new(context, 0);
    auto bias = process_view_നിവേശം_new(context, 1);
    auto ids = process_view_നിവേശം_new(context, 2);

    // OpenVINO uses reversed GGML dimensions:
    //   input: [1, n_token, n_used, n_embd]
    //   bias:  [1, 1, n_expert, n_embd]
    //   ids:   [1, 1, n_token, n_used]
    // Model bias constants may already be stored as [n_expert, n_embd].
    bias = reshape_add_id_നിവേശം_to_2d(bias, context.get_നിവേശം_shape(1), {2, 3});
    ids = reshape_add_id_നിവേശം_to_2d(ids, context.get_നിവേശം_shape(2), {2, 3});

    എങ്കിൽ (ids.get_element_type() != ov::element::i32 && ids.get_element_type() != ov::element::i64) {
        ids = std::make_shared<ov::op::v0::Convert>(ids, ov::element::i32);
    }

    auto gather_axis = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{}, {0});
    ov::Output<ov::Node> selected_bias = std::make_shared<ov::op::v8::Gather>(bias, ids, gather_axis);
    selected_bias = std::make_shared<ov::op::v1::Reshape>(
        selected_bias, std::make_shared<ov::op::v3::ShapeOf>(നിവേശം, ov::element::i64), അസത്യം);

    എങ്കിൽ (selected_bias.get_element_type() != നിവേശം.get_element_type()) {
        selected_bias = std::make_shared<ov::op::v0::Convert>(selected_bias, നിവേശം.get_element_type());
    }

    ov::Output<ov::Node> res = std::make_shared<ov::op::v1::Add>(നിവേശം, selected_bias);
    സ്ഥിരം auto output_type = context.get_output_type();
    എങ്കിൽ (res.get_element_type() != output_type) {
        res = std::make_shared<ov::op::v0::Convert>(res, output_type);
    }

    തിരികെ rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
