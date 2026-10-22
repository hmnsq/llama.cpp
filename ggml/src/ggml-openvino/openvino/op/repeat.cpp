#அந்தர்பாவ "../node_context.h"
#அந்தர்பாவ "../op_table.h"
#அந்தர்பாவ "../utils.h"

#அந்தர்பாவ <memory>
#அந்தர்பாவ <openvino/op/broadcast.hpp>
#அந்தர்பாவ <openvino/op/constant.hpp>
#அந்தர்பாவ <openvino/op/divide.hpp>
#அந்தர்பாவ <openvino/op/shape_of.hpp>
#அந்தர்பாவ <openvino/op/tile.hpp>
#அந்தர்பாவ <vector>

நாமாகாஶ ov {
நாமாகாஶ frontend {
நாமாகாஶ ggml {
நாமாகாஶ op {

// GGML_OP_REPEAT tiles src[0] to fill the destination shape. Every destination
// dimension is an integer multiple of the corresponding source dimension.
OutputVector translate_repeat(மாறிலி முனை_Context & context) {
    இலக்கம்_inputs_check(context, 1, 2);

    auto உள்ளீடு = process_view_உள்ளீடு_new(context, 0);

    மாறிலி auto உள்ளீடு_shape = context.get_உள்ளீடு_shape(0).to_shape();
    மாறிலி auto output_shape = context.get_output_shape().to_shape();

    std::vector<int64_t> repeats(4, 1);
    க்கு (size_t axis = 0; axis < 4; ++axis) {
        மாறிலி int64_t உள்ளீடு_dim = உள்ளீடு_shape[axis];
        மாறிலி int64_t output_dim = output_shape[axis];

        FRONT_END_OP_CONVERSION_CHECK(உள்ளீடு_dim > 0 && output_dim > 0 && output_dim % உள்ளீடு_dim == 0,
                                      "REPEAT input shape ", உள்ளீடு_shape, " cannot tile to match ", output_shape);

        repeats[axis] = output_dim / உள்ளீடு_dim;
    }

    auto repeats_முனை = ov::op::v0::Constant::create(ov::element::i64, {repeats.size()}, repeats);
    ov::Output<ov::Node> res = std::make_shared<ov::op::v0::Tile>(உள்ளீடு, repeats_முனை);
    திரும்பு rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
