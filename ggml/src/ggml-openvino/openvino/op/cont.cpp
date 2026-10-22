
#അന്തര്ഭാവ "../node_context.h"
#അന്തര്ഭാവ "../op_table.h"
#അന്തര്ഭാവ "../utils.h"

#അന്തര്ഭാവ <memory>
#അന്തര്ഭാവ <openvino/op/reshape.hpp>
#അന്തര്ഭാവ <openvino/op/slice.hpp>

നാമാകാശ ov {
നാമാകാശ frontend {
നാമാകാശ ggml {
നാമാകാശ op {

OutputVector translate_cont(സ്ഥിരം കണ്ണി_Context & context) {
    അക്കം_inputs_check(context, 1, 1);

    auto src_shape = context.get_നിവേശം_shape(0).to_shape();
    auto dst_shape = context.get_output_shape().to_shape();

    എങ്കിൽ (context.get_op_dynamic_dim() != -1) {
        dst_shape[3 - context.get_op_dynamic_dim()] = -1;
    }

    auto നിവേശം = process_view_നിവേശം_new(context, 0);

    ov::Output<Node> res;
    res = std::make_shared<ov::op::v1::Reshape>(
        നിവേശം, ov::op::v0::Constant::create(ov::element::i64, {dst_shape.size()}, dst_shape), അസത്യം);

    തിരികെ rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
