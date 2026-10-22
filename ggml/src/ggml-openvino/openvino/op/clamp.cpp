#അന്തര്ഭാവ "../node_context.h"
#അന്തര്ഭാവ "../op_table.h"
#അന്തര്ഭാവ "../utils.h"

#അന്തര്ഭാവ <cstring>
#അന്തര്ഭാവ <openvino/op/clamp.hpp>

നാമാകാശ ov {
നാമാകാശ frontend {
നാമാകാശ ggml {
നാമാകാശ op {

OutputVector translate_clamp(സ്ഥിരം കണ്ണി_Context & context) {
    അക്കം_inputs_check(context, 1, 1);

    auto നിവേശം = process_view_നിവേശം_new(context, 0);

    സ്ഥിരം int32_t * op_params = context.get_output_op_params();
    FRONT_END_CHECK_IMPLEMENTED(op_params != nullptr, "CLAMP requires output op params");

    ദശാംശം min;
    ദശാംശം max;
    std::memcpy(&min, reinterpret_cast<സ്ഥിരം ദശാംശം *>(op_params) + 0, വലുപ്പം(ദശാംശം));
    std::memcpy(&max, reinterpret_cast<സ്ഥിരം ദശാംശം *>(op_params) + 1, വലുപ്പം(ദശാംശം));

    auto res = std::make_shared<ov::op::v0::Clamp>(നിവേശം, min, max);
    തിരികെ rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
