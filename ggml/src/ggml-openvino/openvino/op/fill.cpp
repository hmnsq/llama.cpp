#समावेश "../node_context.h"
#समावेश "../op_table.h"
#समावेश "../utils.h"

#समावेश <openvino/op/broadcast.hpp>
#समावेश <openvino/op/constant.hpp>

namespace ov {
namespace frontend {
namespace ggml {
namespace op {

// GGML FILL sets all elements of a tensor to a constant value.
// The constant is stored as a float in op_params[0].
OutputVector translate_fill(स्थिर गाँठ_Context & context) {
    अंक_inputs_check(context, 1, 1);

    भग्न c;
    स_नकल(&c, context.get_output_op_params(), माप(भग्न));

    स्वतः shape = context.get_निवेश_shape(0).to_shape();

    स्वतः val = ov::op::v0::Constant::create(context.get_output_type(), {}, {c});
    स्वतः target_shape = ov::op::v0::Constant::create(ov::element::i64, {shape.size()},
        std::vector<int64_t>(shape.begin(), shape.end()));
    स्वतः res = std::make_shared<ov::op::v3::Broadcast>(val, target_shape);

    वापस rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
