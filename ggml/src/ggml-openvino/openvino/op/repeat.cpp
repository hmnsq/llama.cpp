#অন্তর্ভাৱ "../node_context.h"
#অন্তর্ভাৱ "../op_table.h"
#অন্তর্ভাৱ "../utils.h"

#অন্তর্ভাৱ <memory>
#অন্তর্ভাৱ <openvino/op/broadcast.hpp>
#অন্তর্ভাৱ <openvino/op/constant.hpp>
#অন্তর্ভাৱ <openvino/op/divide.hpp>
#অন্তর্ভাৱ <openvino/op/shape_of.hpp>
#অন্তর্ভাৱ <openvino/op/tile.hpp>
#অন্তর্ভাৱ <vector>

নামাকাশ ov {
নামাকাশ frontend {
নামাকাশ ggml {
নামাকাশ op {

// GGML_OP_REPEAT tiles src[0] to fill the destination shape. Every destination
// dimension is an integer multiple of the corresponding source dimension.
OutputVector translate_repeat(ধ্রুৱ গ্ৰন্থি_Context & context) {
    অঙ্ক_inputs_check(context, 1, 2);

    auto প্ৰবেশ = process_view_প্ৰবেশ_new(context, 0);

    ধ্রুৱ auto প্ৰবেশ_shape = context.get_প্ৰবেশ_shape(0).to_shape();
    ধ্রুৱ auto output_shape = context.get_output_shape().to_shape();

    std::vector<int64_t> repeats(4, 1);
    বাবে (size_t axis = 0; axis < 4; ++axis) {
        ধ্রুৱ int64_t প্ৰবেশ_dim = প্ৰবেশ_shape[axis];
        ধ্রুৱ int64_t output_dim = output_shape[axis];

        FRONT_END_OP_CONVERSION_CHECK(প্ৰবেশ_dim > 0 && output_dim > 0 && output_dim % প্ৰবেশ_dim == 0,
                                      "REPEAT input shape ", প্ৰবেশ_shape, " cannot tile to match ", output_shape);

        repeats[axis] = output_dim / প্ৰবেশ_dim;
    }

    auto repeats_গ্ৰন্থি = ov::op::v0::Constant::create(ov::element::i64, {repeats.size()}, repeats);
    ov::Output<ov::Node> res = std::make_shared<ov::op::v0::Tile>(প্ৰবেশ, repeats_গ্ৰন্থি);
    ঘূৰাই rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
