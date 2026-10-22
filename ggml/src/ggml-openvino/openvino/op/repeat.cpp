#समावेश "../node_context.h"
#समावेश "../op_table.h"
#समावेश "../utils.h"

#समावेश <memory>
#समावेश <openvino/op/broadcast.hpp>
#समावेश <openvino/op/constant.hpp>
#समावेश <openvino/op/divide.hpp>
#समावेश <openvino/op/shape_of.hpp>
#समावेश <openvino/op/tile.hpp>
#समावेश <vector>

namespace ov {
namespace frontend {
namespace ggml {
namespace op {

// GGML_OP_REPEAT tiles src[0] to fill the destination shape. Every destination
// dimension is an integer multiple of the corresponding source dimension.
OutputVector translate_repeat(स्थिर गाँठ_Context & context) {
    अंक_inputs_check(context, 1, 2);

    स्वतः निवेश = process_view_निवेश_new(context, 0);

    स्थिर स्वतः निवेश_shape = context.get_निवेश_shape(0).to_shape();
    स्थिर स्वतः output_shape = context.get_output_shape().to_shape();

    std::vector<int64_t> repeats(4, 1);
    क्रम (माप_प्रकार axis = 0; axis < 4; ++axis) {
        स्थिर int64_t निवेश_dim = निवेश_shape[axis];
        स्थिर int64_t output_dim = output_shape[axis];

        FRONT_END_OP_CONVERSION_CHECK(निवेश_dim > 0 && output_dim > 0 && output_dim % निवेश_dim == 0,
                                      "REPEAT input shape ", निवेश_shape, " cannot tile to match ", output_shape);

        repeats[axis] = output_dim / निवेश_dim;
    }

    स्वतः repeats_गाँठ = ov::op::v0::Constant::create(ov::element::i64, {repeats.size()}, repeats);
    ov::Output<ov::Node> res = std::make_shared<ov::op::v0::Tile>(निवेश, repeats_गाँठ);
    वापस rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
