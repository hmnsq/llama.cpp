#شامل "../node_context.h"
#شامل "../op_table.h"
#شامل "../utils.h"

#شامل <memory>
#شامل <openvino/op/broadcast.hpp>
#شامل <openvino/op/constant.hpp>
#شامل <openvino/op/divide.hpp>
#شامل <openvino/op/shape_of.hpp>
#شامل <openvino/op/tile.hpp>
#شامل <vector>

نامخلا ov {
نامخلا frontend {
نامخلا ggml {
نامخلا op {

// GGML_OP_REPEAT tiles src[0] to fill the destination shape. Every destination
// dimension is an integer multiple of the corresponding source dimension.
Output_سمتیہ translate_repeat(ثابت گرہ_Context & context) {
    عدد_inputs_check(context, 1, 2);

    خودکار داخلہ = process_view_داخلہ_new(context, 0);

    ثابت خودکار داخلہ_shape = context.get_داخلہ_shape(0).مقصد_shape();
    ثابت خودکار output_shape = context.get_output_shape().مقصد_shape();

    std::vector<int64_t> repeats(4, 1);
    برائے (size_t axis = 0; axis < 4; ++axis) {
        ثابت int64_t داخلہ_dim = داخلہ_shape[axis];
        ثابت int64_t output_dim = output_shape[axis];

        FRONT_انتہا_OP_CONVERSION_CHECK(داخلہ_dim > 0 && output_dim > 0 && output_dim % داخلہ_dim == 0,
                                      "REPEAT input shape ", داخلہ_shape, " cannot tile to match ", output_shape);

        repeats[axis] = output_dim / داخلہ_dim;
    }

    خودکار repeats_گرہ = ov::op::v0::Constant::create(ov::element::i64, {repeats.size()}, repeats);
    ov::Output<ov::Node> res = std::make_shared<ov::op::v0::Tile>(داخلہ, repeats_گرہ);
    واپس rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
