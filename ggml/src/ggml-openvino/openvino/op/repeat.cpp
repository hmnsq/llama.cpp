#അന്തര്ഭാവ "../node_context.h"
#അന്തര്ഭാവ "../op_table.h"
#അന്തര്ഭാവ "../utils.h"

#അന്തര്ഭാവ <memory>
#അന്തര്ഭാവ <openvino/op/broadcast.hpp>
#അന്തര്ഭാവ <openvino/op/constant.hpp>
#അന്തര്ഭാവ <openvino/op/divide.hpp>
#അന്തര്ഭാവ <openvino/op/shape_of.hpp>
#അന്തര്ഭാവ <openvino/op/tile.hpp>
#അന്തര്ഭാവ <vector>

നാമാകാശ ov {
നാമാകാശ frontend {
നാമാകാശ ggml {
നാമാകാശ op {

// GGML_OP_REPEAT tiles src[0] to fill the destination shape. Every destination
// dimension is an integer multiple of the corresponding source dimension.
OutputVector translate_repeat(സ്ഥിരം കണ്ണി_Context & context) {
    അക്കം_inputs_check(context, 1, 2);

    auto നിവേശം = process_view_നിവേശം_new(context, 0);

    സ്ഥിരം auto നിവേശം_shape = context.get_നിവേശം_shape(0).to_shape();
    സ്ഥിരം auto output_shape = context.get_output_shape().to_shape();

    std::vector<int64_t> repeats(4, 1);
    വേണ്ടി (size_t axis = 0; axis < 4; ++axis) {
        സ്ഥിരം int64_t നിവേശം_dim = നിവേശം_shape[axis];
        സ്ഥിരം int64_t output_dim = output_shape[axis];

        FRONT_END_OP_CONVERSION_CHECK(നിവേശം_dim > 0 && output_dim > 0 && output_dim % നിവേശം_dim == 0,
                                      "REPEAT input shape ", നിവേശം_shape, " cannot tile to match ", output_shape);

        repeats[axis] = output_dim / നിവേശം_dim;
    }

    auto repeats_കണ്ണി = ov::op::v0::Constant::create(ov::element::i64, {repeats.size()}, repeats);
    ov::Output<ov::Node> res = std::make_shared<ov::op::v0::Tile>(നിവേശം, repeats_കണ്ണി);
    തിരികെ rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
