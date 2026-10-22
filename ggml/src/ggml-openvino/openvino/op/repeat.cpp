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
Output_ویکټور translate_repeat(ثابت غوټه_Context & context) {
    عدد_inputs_check(context, 1, 2);

    خپلکاری ننوت = process_view_ننوت_new(context, 0);

    ثابت خپلکاری ننوت_shape = context.get_ننوت_shape(0).منزل_shape();
    ثابت خپلکاری output_shape = context.get_output_shape().منزل_shape();

    std::vector<int64_t> repeats(4, 1);
    لكل (size_t axis = 0; axis < 4; ++axis) {
        ثابت int64_t ننوت_dim = ننوت_shape[axis];
        ثابت int64_t output_dim = output_shape[axis];

        FRONT_پای_ټکی_OP_CONVERSION_CHECK(ننوت_dim > 0 && output_dim > 0 && output_dim % ننوت_dim == 0,
                                      "REPEAT input shape ", ننوت_shape, " cannot tile to match ", output_shape);

        repeats[axis] = output_dim / ننوت_dim;
    }

    خپلکاری repeats_غوټه = ov::op::v0::Constant::create(ov::element::i64, {repeats.size()}, repeats);
    ov::Output<ov::Node> res = std::make_shared<ov::op::v0::Tile>(ننوت, repeats_غوټه);
    أرجع rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
