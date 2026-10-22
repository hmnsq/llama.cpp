#समावेश "../node_context.h"
#समावेश "../op_table.h"
#समावेश "../utils.h"

#समावेश <openvino/op/constant.hpp>
#समावेश <openvino/op/cum_sum.hpp>

namespace ov {
namespace frontend {
namespace ggml {
namespace op {

// GGML cumsum computes prefix sum along dim 0 (the innermost/fastest dimension).
// In OV layout the dims are reversed: ggml [ne0, ne1, ne2, ne3] → OV [ne3, ne2, ne1, ne0],
// so ggml dim 0 maps to OV axis 3 (last axis).
OutputVector translate_cumsum(स्थिर गाँठ_Context & context) {
    अंक_inputs_check(context, 1, 1);

    स्वतः x    = context.get_निवेश(0);
    स्वतः axis = ov::op::v0::Constant::create(ov::element::i64, {}, {3});
    स्वतः res  = std::make_shared<ov::op::v0::Cum_योग>(x, axis);

    वापस rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
