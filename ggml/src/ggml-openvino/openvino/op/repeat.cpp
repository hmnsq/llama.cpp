#ଅନ୍ତର୍ଭାଵ "../node_context.h"
#ଅନ୍ତର୍ଭାଵ "../op_table.h"
#ଅନ୍ତର୍ଭାଵ "../utils.h"

#ଅନ୍ତର୍ଭାଵ <memory>
#ଅନ୍ତର୍ଭାଵ <openvino/op/broadcast.hpp>
#ଅନ୍ତର୍ଭାଵ <openvino/op/constant.hpp>
#ଅନ୍ତର୍ଭାଵ <openvino/op/divide.hpp>
#ଅନ୍ତର୍ଭାଵ <openvino/op/shape_of.hpp>
#ଅନ୍ତର୍ଭାଵ <openvino/op/tile.hpp>
#ଅନ୍ତର୍ଭାଵ <vector>

ନାମାକାଶ ov {
ନାମାକାଶ frontend {
ନାମାକାଶ ggml {
ନାମାକାଶ op {

// GGML_OP_REPEAT tiles src[0] to fill the destination shape. Every destination
// dimension is an integer multiple of the corresponding source dimension.
OutputVector translate_repeat(ଅଚଳ ଗଣ୍ଠି_Context & context) {
    ଅଙ୍କ_inputs_check(context, 1, 2);

    auto ନିବେଶ = process_view_ନିବେଶ_new(context, 0);

    ଅଚଳ auto ନିବେଶ_shape = context.get_ନିବେଶ_shape(0).to_shape();
    ଅଚଳ auto output_shape = context.get_output_shape().to_shape();

    std::vector<int64_t> repeats(4, 1);
    ପାଇଁ (size_t axis = 0; axis < 4; ++axis) {
        ଅଚଳ int64_t ନିବେଶ_dim = ନିବେଶ_shape[axis];
        ଅଚଳ int64_t output_dim = output_shape[axis];

        FRONT_END_OP_CONVERSION_CHECK(ନିବେଶ_dim > 0 && output_dim > 0 && output_dim % ନିବେଶ_dim == 0,
                                      "REPEAT input shape ", ନିବେଶ_shape, " cannot tile to match ", output_shape);

        repeats[axis] = output_dim / ନିବେଶ_dim;
    }

    auto repeats_ଗଣ୍ଠି = ov::op::v0::Constant::create(ov::element::i64, {repeats.size()}, repeats);
    ov::Output<ov::Node> res = std::make_shared<ov::op::v0::Tile>(ନିବେଶ, repeats_ଗଣ୍ଠି);
    ଫେରନ୍ତୁ rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
