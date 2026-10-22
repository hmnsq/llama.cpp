#شامل "../node_context.h"
#شامل "../op_table.h"
#شامل "../utils.h"

#شامل <openvino/op/constant.hpp>
#شامل <openvino/op/cum_sum.hpp>

نامخلا ov {
نامخلا frontend {
نامخلا ggml {
نامخلا op {

// GGML cumsum computes prefix sum along dim 0 (the innermost/fastest dimension).
// In OV layout the dims are reversed: ggml [ne0, ne1, ne2, ne3] → OV [ne3, ne2, ne1, ne0],
// so ggml dim 0 maps to OV axis 3 (last axis).
Output_سمتیہ translate_cumsum(ثابت گرہ_Context & context) {
    عدد_inputs_check(context, 1, 1);

    خودکار x    = context.get_داخلہ(0);
    خودکار axis = ov::op::v0::Constant::create(ov::element::i64, {}, {3});
    خودکار res  = std::make_shared<ov::op::v0::Cum_مجموع>(x, axis);

    واپس rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
