#समावेश "../node_context.h"
#समावेश "../op_table.h"
#समावेश "../utils.h"

#समावेश <memory>
#समावेश <openvino/op/multiply.hpp>
#समावेश <openvino/op/sqrt.hpp>

namespace ov {
namespace frontend {
namespace ggml {
namespace op {

OutputVector translate_sqr(स्थिर गाँठ_Context & context) {
    अंक_inputs_check(context, 1, 1);

    स्वतः निवेश = process_view_निवेश_new(context, 0);
    स्वतः res = std::make_shared<ov::op::v1::Multiply>(निवेश, निवेश);

    वापस rename_outputs_with_suffix({res}, context.get_name());
}

OutputVector translate_sqrt(स्थिर गाँठ_Context & context) {
    अंक_inputs_check(context, 1, 1);

    स्वतः निवेश = process_view_निवेश_new(context, 0);
    स्वतः res = std::make_shared<ov::op::v0::Sqrt>(निवेश);

    वापस rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
