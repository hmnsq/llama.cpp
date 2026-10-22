#समावेश "../node_context.h"
#समावेश "../op_table.h"
#समावेश "../utils.h"

#समावेश <cstring>
#समावेश <openvino/op/clamp.hpp>

namespace ov {
namespace frontend {
namespace ggml {
namespace op {

OutputVector translate_clamp(स्थिर गाँठ_Context & context) {
    अंक_inputs_check(context, 1, 1);

    स्वतः निवेश = process_view_निवेश_new(context, 0);

    स्थिर int32_t * op_params = context.get_output_op_params();
    FRONT_END_CHECK_IMPLEMENTED(op_params != nullptr, "CLAMP requires output op params");

    भग्न min;
    भग्न max;
    std::स_नकल(&min, reinterpret_cast<स्थिर भग्न *>(op_params) + 0, माप(भग्न));
    std::स_नकल(&max, reinterpret_cast<स्थिर भग्न *>(op_params) + 1, माप(भग्न));

    स्वतः res = std::make_shared<ov::op::v0::Clamp>(निवेश, min, max);
    वापस rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
