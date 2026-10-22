#ضمن "../node_context.h"
#ضمن "../op_table.h"
#ضمن "../utils.h"

#ضمن <openvino/op/constant.hpp>
#ضمن <openvino/op/cum_sum.hpp>

نطاق ov {
نطاق frontend {
نطاق ggml {
نطاق op {

// GGML cumsum computes prefix sum along dim 0 (the innermost/fastest dimension).
// In OV layout the dims are reversed: ggml [ne0, ne1, ne2, ne3] → OV [ne3, ne2, ne1, ne0],
// so ggml dim 0 maps to OV axis 3 (last axis).
Output_وکتور translate_cumsum(ثابت گره_Context & context) {
    رقم_inputs_check(context, 1, 1);

    خودکار x    = context.get_درون_داد(0);
    خودکار axis = ov::op::v0::Constant::create(ov::element::i64, {}, {3});
    خودکار res  = std::make_shared<ov::op::v0::Cum_جمع>(x, axis);

    أرجع rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
