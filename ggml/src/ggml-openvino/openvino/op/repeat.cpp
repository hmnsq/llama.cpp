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
OutputVector translate_repeat(ध्रुव गाठ_Context & context) {
    अंक_inputs_check(context, 1, 2);

    auto आदान = process_view_आदान_new(context, 0);

    ध्रुव auto आदान_shape = context.get_आदान_shape(0).to_shape();
    ध्रुव auto output_shape = context.get_output_shape().to_shape();

    std::vector<int64_t> repeats(4, 1);
    प्रति (size_t axis = 0; axis < 4; ++axis) {
        ध्रुव int64_t आदान_dim = आदान_shape[axis];
        ध्रुव int64_t output_dim = output_shape[axis];

        FRONT_END_OP_CONVERSION_CHECK(आदान_dim > 0 && output_dim > 0 && output_dim % आदान_dim == 0,
                                      "REPEAT input shape ", आदान_shape, " cannot tile to match ", output_shape);

        repeats[axis] = output_dim / आदान_dim;
    }

    auto repeats_गाठ = ov::op::v0::Constant::create(ov::element::i64, {repeats.size()}, repeats);
    ov::Output<ov::Node> res = std::make_shared<ov::op::v0::Tile>(आदान, repeats_गाठ);
    प्रत्यागम rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
