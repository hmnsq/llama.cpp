#അന്തര്ഭാവ "../node_context.h"
#അന്തര്ഭാവ "../op_table.h"
#അന്തര്ഭാവ "../utils.h"

#അന്തര്ഭാവ <memory>
#അന്തര്ഭാവ <openvino/frontend/exception.hpp>
#അന്തര്ഭാവ <openvino/op/concat.hpp>
#അന്തര്ഭാവ <openvino/op/convert.hpp>

നാമാകാശ ov {
നാമാകാശ frontend {
നാമാകാശ ggml {
നാമാകാശ op {

OutputVector translate_concat(സ്ഥിരം കണ്ണി_Context & context) {
    അക്കം_inputs_check(context, 2, 2);

    സ്ഥിരം int32_t * op_params = context.get_output_op_params();
    FRONT_END_CHECK_IMPLEMENTED(op_params != nullptr, "CONCAT requires output op params");

    സ്ഥിരം auto output_shape = context.get_output_shape();
    FRONT_END_CHECK_IMPLEMENTED(output_shape.rank().is_static(), "CONCAT requires static output rank");

    സ്ഥിരം auto rank = output_shape.rank().get_length();
    സ്ഥിരം int32_t ggml_dim = op_params[0];
    FRONT_END_CHECK_IMPLEMENTED(ggml_dim >= 0 && ggml_dim < rank, "CONCAT axis is out of range");

    auto നിവേശം_0 = process_view_നിവേശം_new(context, 0);
    auto നിവേശം_1 = process_view_നിവേശം_new(context, 1);
    സ്ഥിരം auto output_type = context.get_output_type();

    എങ്കിൽ (നിവേശം_0.get_element_type() != output_type) {
        നിവേശം_0 = std::make_shared<ov::op::v0::Convert>(നിവേശം_0, output_type);
    }
    എങ്കിൽ (നിവേശം_1.get_element_type() != output_type) {
        നിവേശം_1 = std::make_shared<ov::op::v0::Convert>(നിവേശം_1, output_type);
    }

    സ്ഥിരം auto axis = static_cast<int64_t>(rank - 1 - ggml_dim);
    auto res = std::make_shared<ov::op::v0::Concat>(OutputVector{നിവേശം_0, നിവേശം_1}, axis);

    തിരികെ rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
