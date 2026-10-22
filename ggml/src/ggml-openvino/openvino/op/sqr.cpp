#അന്തര്ഭാവ "../node_context.h"
#അന്തര്ഭാവ "../op_table.h"
#അന്തര്ഭാവ "../utils.h"

#അന്തര്ഭാവ <memory>
#അന്തര്ഭാവ <openvino/op/multiply.hpp>
#അന്തര്ഭാവ <openvino/op/sqrt.hpp>

നാമാകാശ ov {
നാമാകാശ frontend {
നാമാകാശ ggml {
നാമാകാശ op {

OutputVector translate_sqr(സ്ഥിരം കണ്ണി_Context & context) {
    അക്കം_inputs_check(context, 1, 1);

    auto നിവേശം = process_view_നിവേശം_new(context, 0);
    auto res = std::make_shared<ov::op::v1::Multiply>(നിവേശം, നിവേശം);

    തിരികെ rename_outputs_with_suffix({res}, context.get_name());
}

OutputVector translate_sqrt(സ്ഥിരം കണ്ണി_Context & context) {
    അക്കം_inputs_check(context, 1, 1);

    auto നിവേശം = process_view_നിവേശം_new(context, 0);
    auto res = std::make_shared<ov::op::v0::Sqrt>(നിവേശം);

    തിരികെ rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
