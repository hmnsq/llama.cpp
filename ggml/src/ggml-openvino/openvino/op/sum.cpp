#അന്തര്ഭാവ "../node_context.h"
#അന്തര്ഭാവ "../op_table.h"
#അന്തര്ഭാവ "../utils.h"

#അന്തര്ഭാവ <memory>
#അന്തര്ഭാവ <openvino/op/constant.hpp>
#അന്തര്ഭാവ <openvino/op/range.hpp>
#അന്തര്ഭാവ <openvino/op/reduce_sum.hpp>
#അന്തര്ഭാവ <openvino/op/reshape.hpp>
#അന്തര്ഭാവ <openvino/op/shape_of.hpp>

നാമാകാശ ov {
നാമാകാശ frontend {
നാമാകാശ ggml {
നാമാകാശ op {

OutputVector translate_sum(സ്ഥിരം കണ്ണി_Context & context) {
    അക്കം_inputs_check(context, 1, 1);

    auto നിവേശം = process_view_നിവേശം_new(context, 0);
    auto axes = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{4}, {0, 1, 2, 3});
    auto res = std::make_shared<ov::op::v1::ReduceSum>(നിവേശം, axes, സത്യം);

    തിരികെ rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
