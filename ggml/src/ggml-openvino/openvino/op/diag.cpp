#অন্তর্ভাব "../node_context.h"
#অন্তর্ভাব "../op_table.h"
#অন্তর্ভাব "../utils.h"

#অন্তর্ভাব <openvino/op/constant.hpp>
#অন্তর্ভাব <openvino/op/eye.hpp>
#অন্তর্ভাব <openvino/op/multiply.hpp>

নামাকাশ ov {
নামাকাশ frontend {
নামাকাশ ggml {
নামাকাশ op {

// GGML DIAG takes a 1D vector (ne0, 1, ne2, ne3) and produces a diagonal matrix
// of shape (ne0, ne0, ne2, ne3).
// In OV layout (ggml [ne0, ne1, ne2, ne3] → OV [ne3, ne2, ne1, ne0]):
//   input:  [ne3, ne2, 1, ne0]
//   output: [ne3, ne2, ne0, ne0]
// The diagonal: output[..., i, j] = input[..., 0, j] if i == j, else 0.
OutputVector translate_diag(ধ্রুব গ্রন্থি_Context & context) {
    অঙ্ক_inputs_check(context, 1, 1);

    auto x = process_view_প্রবেশ_new(context, 0);  // OV shape: [ne3, ne2, 1, ne0]

    auto n = get_dimensions(x.get_গ্রন্থি_shared_ptr(), {3});
    auto zero_diag = ov::op::v0::Constant::create(ov::element::i64, {}, {0});

    auto eye = std::make_shared<ov::op::v9::Eye>(n, n, zero_diag, x.get_element_type());
    auto res = std::make_shared<ov::op::v1::Multiply>(x, eye);

    প্রত্যাগম rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
