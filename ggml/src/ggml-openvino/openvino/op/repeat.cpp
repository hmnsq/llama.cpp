#ਅਨ੍ਤਰ੍ਭਾਵ "../node_context.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "../op_table.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "../utils.h"

#ਅਨ੍ਤਰ੍ਭਾਵ <memory>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/broadcast.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/constant.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/divide.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/shape_of.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/tile.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <vector>

ਨਾਮਾਕਾਸ਼ ov {
ਨਾਮਾਕਾਸ਼ frontend {
ਨਾਮਾਕਾਸ਼ ggml {
ਨਾਮਾਕਾਸ਼ op {

// GGML_OP_REPEAT tiles src[0] to fill the destination shape. Every destination
// dimension is an integer multiple of the corresponding source dimension.
Output_ਸਦਿਸ਼ translate_repeat(ਅਚਲ ਗੰਢ_Context & context) {
    ਅੰਕ_inputs_check(context, 1, 2);

    ਸਵੈਚਾਲਿਤ ਨਿਵੇਸ਼ = process_view_ਨਿਵੇਸ਼_new(context, 0);

    ਅਚਲ ਸਵੈਚਾਲਿਤ ਨਿਵੇਸ਼_shape = context.get_ਨਿਵੇਸ਼_shape(0).ਮੰਜ਼ਿਲ_shape();
    ਅਚਲ ਸਵੈਚਾਲਿਤ output_shape = context.get_output_shape().ਮੰਜ਼ਿਲ_shape();

    std::vector<int64_t> repeats(4, 1);
    ਲਈ (size_t axis = 0; axis < 4; ++axis) {
        ਅਚਲ int64_t ਨਿਵੇਸ਼_dim = ਨਿਵੇਸ਼_shape[axis];
        ਅਚਲ int64_t output_dim = output_shape[axis];

        FRONT_ਸਿਰਾ_OP_CONVERSION_CHECK(ਨਿਵੇਸ਼_dim > 0 && output_dim > 0 && output_dim % ਨਿਵੇਸ਼_dim == 0,
                                      "REPEAT input shape ", ਨਿਵੇਸ਼_shape, " cannot tile to match ", output_shape);

        repeats[axis] = output_dim / ਨਿਵੇਸ਼_dim;
    }

    ਸਵੈਚਾਲਿਤ repeats_ਗੰਢ = ov::op::v0::Constant::create(ov::element::i64, {repeats.size()}, repeats);
    ov::Output<ov::Node> res = std::make_shared<ov::op::v0::Tile>(ਨਿਵੇਸ਼, repeats_ਗੰਢ);
    ਵਾਪਸ rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
