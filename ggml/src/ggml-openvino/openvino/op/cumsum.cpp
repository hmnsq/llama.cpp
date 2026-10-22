#अन्तर्भाव "../node_context.h"
#अन्तर्भाव "../op_table.h"
#अन्तर्भाव "../utils.h"

#अन्तर्भाव <openvino/op/constant.hpp>
#अन्तर्भाव <openvino/op/cum_sum.hpp>

नामाकाश ov {
नामाकाश frontend {
नामाकाश ggml {
नामाकाश op {

// GGML cumsum computes prefix sum along dim 0 (the innermost/fastest dimension).
// In OV layout the dims are reversed: ggml [ne0, ne1, ne2, ne3] → OV [ne3, ne2, ne1, ne0],
// so ggml dim 0 maps to OV axis 3 (last axis).
OutputVector translate_cumsum(अचर गाँठो_Context & context) {
    अङ्क_inputs_check(context, 1, 1);

    स्वतः x    = context.get_आगत(0);
    स्वतः axis = ov::op::v0::Constant::create(ov::element::i64, {}, {3});
    स्वतः res  = std::make_shared<ov::op::v0::CumSum>(x, axis);

    फर्काऊ rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
