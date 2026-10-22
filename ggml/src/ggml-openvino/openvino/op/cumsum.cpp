#ਅਨ੍ਤਰ੍ਭਾਵ "../node_context.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "../op_table.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "../utils.h"

#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/constant.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/cum_sum.hpp>

ਨਾਮਾਕਾਸ਼ ov {
ਨਾਮਾਕਾਸ਼ frontend {
ਨਾਮਾਕਾਸ਼ ggml {
ਨਾਮਾਕਾਸ਼ op {

// GGML cumsum computes prefix sum along dim 0 (the innermost/fastest dimension).
// In OV layout the dims are reversed: ggml [ne0, ne1, ne2, ne3] → OV [ne3, ne2, ne1, ne0],
// so ggml dim 0 maps to OV axis 3 (last axis).
Output_ਸਦਿਸ਼ translate_cumsum(ਅਚਲ ਗੰਢ_Context & context) {
    ਅੰਕ_inputs_check(context, 1, 1);

    ਸਵੈਚਾਲਿਤ x    = context.get_ਨਿਵੇਸ਼(0);
    ਸਵੈਚਾਲਿਤ axis = ov::op::v0::Constant::create(ov::element::i64, {}, {3});
    ਸਵੈਚਾਲਿਤ res  = std::make_shared<ov::op::v0::Cum_ਜੋੜ>(x, axis);

    ਵਾਪਸ rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
