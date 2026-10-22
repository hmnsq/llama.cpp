#अन्तर्भाव "../node_context.h"
#अन्तर्भाव "../op_table.h"
#अन्तर्भाव "../utils.h"

#अन्तर्भाव <memory>
#अन्तर्भाव <openvino/op/broadcast.hpp>
#अन्तर्भाव <openvino/op/constant.hpp>
#अन्तर्भाव <openvino/op/divide.hpp>
#अन्तर्भाव <openvino/op/shape_of.hpp>
#अन्तर्भाव <openvino/op/tile.hpp>
#अन्तर्भाव <vector>

नामाकाश ov {
नामाकाश frontend {
नामाकाश ggml {
नामाकाश op {

// GGML_OP_REPEAT tiles src[0] to fill the destination shape. Every destination
// dimension is an integer multiple of the corresponding source dimension.
OutputVector translate_repeat(अचर गाँठो_Context & context) {
    अङ्क_inputs_check(context, 1, 2);

    स्वतः आगत = process_view_आगत_new(context, 0);

    अचर स्वतः आगत_shape = context.get_आगत_shape(0).to_shape();
    अचर स्वतः output_shape = context.get_output_shape().to_shape();

    std::vector<int64_t> repeats(4, 1);
    लागि (माप_प्रकार axis = 0; axis < 4; ++axis) {
        अचर int64_t आगत_dim = आगत_shape[axis];
        अचर int64_t output_dim = output_shape[axis];

        FRONT_END_OP_CONVERSION_CHECK(आगत_dim > 0 && output_dim > 0 && output_dim % आगत_dim == 0,
                                      "REPEAT input shape ", आगत_shape, " cannot tile to match ", output_shape);

        repeats[axis] = output_dim / आगत_dim;
    }

    स्वतः repeats_गाँठो = ov::op::v0::Constant::create(ov::element::i64, {repeats.size()}, repeats);
    ov::Output<ov::Node> res = std::make_shared<ov::op::v0::Tile>(आगत, repeats_गाँठो);
    फर्काऊ rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
