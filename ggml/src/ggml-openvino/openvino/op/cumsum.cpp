#અન્તર્ભાવ "../node_context.h"
#અન્તર્ભાવ "../op_table.h"
#અન્તર્ભાવ "../utils.h"

#અન્તર્ભાવ <openvino/op/constant.hpp>
#અન્તર્ભાવ <openvino/op/cum_sum.hpp>

નામાકાશ ov {
નામાકાશ frontend {
નામાકાશ ggml {
નામાકાશ op {

// GGML cumsum computes prefix sum along dim 0 (the innermost/fastest dimension).
// In OV layout the dims are reversed: ggml [ne0, ne1, ne2, ne3] → OV [ne3, ne2, ne1, ne0],
// so ggml dim 0 maps to OV axis 3 (last axis).
OutputVector translate_cumsum(અચલ ગાંઠ_Context & context) {
    અંક_inputs_check(context, 1, 1);

    auto x    = context.get_આગત(0);
    auto axis = ov::op::v0::Constant::create(ov::element::i64, {}, {3});
    auto res  = std::make_shared<ov::op::v0::CumSum>(x, axis);

    પાછા rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
