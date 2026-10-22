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
Output_وکتور translate_repeat(ثابت گره_Context & context) {
    رقم_inputs_check(context, 1, 2);

    خودکار درون_داد = process_view_درون_داد_new(context, 0);

    ثابت خودکار درون_داد_shape = context.get_درون_داد_shape(0).مقصد_shape();
    ثابت خودکار output_shape = context.get_output_shape().مقصد_shape();

    std::vector<int64_t> repeats(4, 1);
    لكل (size_t axis = 0; axis < 4; ++axis) {
        ثابت int64_t درون_داد_dim = درون_داد_shape[axis];
        ثابت int64_t output_dim = output_shape[axis];

        FRONT_انتها_OP_CONVERSION_CHECK(درون_داد_dim > 0 && output_dim > 0 && output_dim % درون_داد_dim == 0,
                                      "REPEAT input shape ", درون_داد_shape, " cannot tile to match ", output_shape);

        repeats[axis] = output_dim / درون_داد_dim;
    }

    خودکار repeats_گره = ov::op::v0::Constant::create(ov::element::i64, {repeats.size()}, repeats);
    ov::Output<ov::Node> res = std::make_shared<ov::op::v0::Tile>(درون_داد, repeats_گره);
    أرجع rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
