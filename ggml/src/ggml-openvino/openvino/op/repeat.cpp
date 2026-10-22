#অন্তর্ভাব "../node_context.h"
#অন্তর্ভাব "../op_table.h"
#অন্তর্ভাব "../utils.h"

#অন্তর্ভাব <memory>
#অন্তর্ভাব <openvino/op/broadcast.hpp>
#অন্তর্ভাব <openvino/op/constant.hpp>
#অন্তর্ভাব <openvino/op/divide.hpp>
#অন্তর্ভাব <openvino/op/shape_of.hpp>
#অন্তর্ভাব <openvino/op/tile.hpp>
#অন্তর্ভাব <vector>

নামাকাশ ov {
নামাকাশ frontend {
নামাকাশ ggml {
নামাকাশ op {

// GGML_OP_REPEAT tiles src[0] to fill the destination shape. Every destination
// dimension is an integer multiple of the corresponding source dimension.
OutputVector translate_repeat(ধ্রুবক গ্রন্থি_Context & context) {
    অঙ্ক_inputs_check(context, 1, 2);

    auto প্রবেশ = process_view_প্রবেশ_new(context, 0);

    ধ্রুবক auto প্রবেশ_shape = context.get_প্রবেশ_shape(0).to_shape();
    ধ্রুবক auto output_shape = context.get_output_shape().to_shape();

    std::vector<int64_t> repeats(4, 1);
    জন্য (size_t axis = 0; axis < 4; ++axis) {
        ধ্রুবক int64_t প্রবেশ_dim = প্রবেশ_shape[axis];
        ধ্রুবক int64_t output_dim = output_shape[axis];

        FRONT_END_OP_CONVERSION_CHECK(প্রবেশ_dim > 0 && output_dim > 0 && output_dim % প্রবেশ_dim == 0,
                                      "REPEAT input shape ", প্রবেশ_shape, " cannot tile to match ", output_shape);

        repeats[axis] = output_dim / প্রবেশ_dim;
    }

    auto repeats_গ্রন্থি = ov::op::v0::Constant::create(ov::element::i64, {repeats.size()}, repeats);
    ov::Output<ov::Node> res = std::make_shared<ov::op::v0::Tile>(প্রবেশ, repeats_গ্রন্থি);
    ফেরত rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
