#અન્તર્ભાવ "../node_context.h"
#અન્તર્ભાવ "../op_table.h"
#અન્તર્ભાવ "../utils.h"

#અન્તર્ભાવ <memory>
#અન્તર્ભાવ <openvino/op/broadcast.hpp>
#અન્તર્ભાવ <openvino/op/constant.hpp>
#અન્તર્ભાવ <openvino/op/divide.hpp>
#અન્તર્ભાવ <openvino/op/shape_of.hpp>
#અન્તર્ભાવ <openvino/op/tile.hpp>
#અન્તર્ભાવ <vector>

નામાકાશ ov {
નામાકાશ frontend {
નામાકાશ ggml {
નામાકાશ op {

// GGML_OP_REPEAT tiles src[0] to fill the destination shape. Every destination
// dimension is an integer multiple of the corresponding source dimension.
OutputVector translate_repeat(અચલ ગાંઠ_Context & context) {
    અંક_inputs_check(context, 1, 2);

    auto આગત = process_view_આગત_new(context, 0);

    અચલ auto આગત_shape = context.get_આગત_shape(0).to_shape();
    અચલ auto output_shape = context.get_output_shape().to_shape();

    std::vector<int64_t> repeats(4, 1);
    માટે (size_t axis = 0; axis < 4; ++axis) {
        અચલ int64_t આગત_dim = આગત_shape[axis];
        અચલ int64_t output_dim = output_shape[axis];

        FRONT_END_OP_CONVERSION_CHECK(આગત_dim > 0 && output_dim > 0 && output_dim % આગત_dim == 0,
                                      "REPEAT input shape ", આગત_shape, " cannot tile to match ", output_shape);

        repeats[axis] = output_dim / આગત_dim;
    }

    auto repeats_ગાંઠ = ov::op::v0::Constant::create(ov::element::i64, {repeats.size()}, repeats);
    ov::Output<ov::Node> res = std::make_shared<ov::op::v0::Tile>(આગત, repeats_ગાંઠ);
    પાછા rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
