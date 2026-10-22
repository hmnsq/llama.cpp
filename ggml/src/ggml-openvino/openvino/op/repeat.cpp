#ضمن "../node_context.h"
#ضمن "../op_table.h"
#ضمن "../utils.h"

#ضمن <memory>
#ضمن <openvino/op/broadcast.hpp>
#ضمن <openvino/op/constant.hpp>
#ضمن <openvino/op/divide.hpp>
#ضمن <openvino/op/shape_of.hpp>
#ضمن <openvino/op/tile.hpp>
#ضمن <vector>

نطاق ov {
نطاق frontend {
نطاق ggml {
نطاق op {

// GGML_OP_REPEAT tiles src[0] to fill the destination shape. Every destination
// dimension is an integer multiple of the corresponding source dimension.
OutputVector translate_repeat(ثابت NodeContext & context) {
    num_inputs_check(context, 1, 2);

    auto input = process_view_input_new(context, 0);

    ثابت auto input_shape = context.get_input_shape(0).to_shape();
    ثابت auto output_shape = context.get_output_shape().to_shape();

    std::vector<int64_t> repeats(4, 1);
    עבור (size_t axis = 0; axis < 4; ++axis) {
        ثابت int64_t input_dim = input_shape[axis];
        ثابت int64_t output_dim = output_shape[axis];

        FRONT_END_OP_CONVERSION_CHECK(input_dim > 0 && output_dim > 0 && output_dim % input_dim == 0,
                                      "REPEAT input shape ", input_shape, " cannot tile to match ", output_shape);

        repeats[axis] = output_dim / input_dim;
    }

    auto repeats_node = ov::op::v0::Constant::create(ov::element::i64, {repeats.size()}, repeats);
    ov::Output<ov::Node> res = std::make_shared<ov::op::v0::Tile>(input, repeats_node);
    החזר rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
