#ضمن "../node_context.h"
#ضمن "../op_table.h"
#ضمن "../utils.h"

#ضمن <openvino/op/constant.hpp>
#ضمن <openvino/op/eye.hpp>
#ضمن <openvino/op/multiply.hpp>

نطاق ov {
نطاق frontend {
نطاق ggml {
نطاق op {

// GGML DIAG takes a 1D vector (ne0, 1, ne2, ne3) and produces a diagonal matrix
// of shape (ne0, ne0, ne2, ne3).
// In OV layout (ggml [ne0, ne1, ne2, ne3] → OV [ne3, ne2, ne1, ne0]):
//   input:  [ne3, ne2, 1, ne0]
//   output: [ne3, ne2, ne0, ne0]
// The diagonal: output[..., i, j] = input[..., 0, j] if i == j, else 0.
OutputVector translate_diag(ثابت نود_Context & context) {
    اڠک_inputs_check(context, 1, 1);

    auto x = process_view_ماسوقن_new(context, 0);  // OV shape: [ne3, ne2, 1, ne0]

    auto n = get_dimensions(x.get_نود_shared_ptr(), {3});
    auto zero_diag = ov::op::v0::Constant::create(ov::element::i64, {}, {0});

    auto eye = std::make_shared<ov::op::v9::Eye>(n, n, zero_diag, x.get_element_type());
    auto res = std::make_shared<ov::op::v1::Multiply>(x, eye);

    أرجع rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
