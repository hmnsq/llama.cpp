#समावेश "../node_context.h"
#समावेश "../op_table.h"
#समावेश "../utils.h"

#समावेश <cstring>
#समावेश <memory>
#समावेश <openvino/op/constant.hpp>
#समावेश <openvino/op/mvn.hpp>

namespace ov {
namespace frontend {
namespace ggml {
namespace op {

OutputVector translate_norm(स्थिर गाँठ_Context & context) {
    अंक_inputs_check(context, 1, 1);

    स्वतः निवेश_गाँठ = process_view_निवेश_new(context, 0);
    भग्न eps;
    स_नकल(&eps, context.get_output_op_params(), माप(भग्न));

    स्वतः axes = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {-1});
    स्वतः res = std::make_shared<ov::op::v6::MVN>(निवेश_गाँठ, axes, true, eps, ov::op::MVNEpsMode::INSIDE_SQRT);

    वापस rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
