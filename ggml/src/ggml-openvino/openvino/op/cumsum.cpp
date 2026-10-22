#অন্তর্ভাৱ "../node_context.h"
#অন্তর্ভাৱ "../op_table.h"
#অন্তর্ভাৱ "../utils.h"

#অন্তর্ভাৱ <openvino/op/constant.hpp>
#অন্তর্ভাৱ <openvino/op/cum_sum.hpp>

নামাকাশ ov {
নামাকাশ frontend {
নামাকাশ ggml {
নামাকাশ op {

// GGML cumsum computes prefix sum along dim 0 (the innermost/fastest dimension).
// In OV layout the dims are reversed: ggml [ne0, ne1, ne2, ne3] → OV [ne3, ne2, ne1, ne0],
// so ggml dim 0 maps to OV axis 3 (last axis).
OutputVector translate_cumsum(ধ্রুৱ গ্ৰন্থি_Context & context) {
    অঙ্ক_inputs_check(context, 1, 1);

    auto x    = context.get_প্ৰবেশ(0);
    auto axis = ov::op::v0::Constant::create(ov::element::i64, {}, {3});
    auto res  = std::make_shared<ov::op::v0::CumSum>(x, axis);

    ঘূৰাই rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
