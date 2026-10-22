#समावेश "../node_context.h"
#समावेश "../op_table.h"
#समावेश "../utils.h"

#समावेश <memory>
#समावेश <openvino/op/constant.hpp>
#समावेश <openvino/op/reduce_sum.hpp>

namespace ov {
namespace frontend {
namespace ggml {
namespace op {

OutputVector translate_योग_rows(स्थिर गाँठ_Context & context) {
    अंक_inputs_check(context, 1, 1);

    स्वतः निवेश = process_view_निवेश_new(context, 0);
    स्वतः res = std::make_shared<ov::op::v1::Reduce_योग>(
        निवेश, ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {-1}), true);

    वापस rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
