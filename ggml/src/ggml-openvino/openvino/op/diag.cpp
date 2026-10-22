#ਅਨ੍ਤਰ੍ਭਾਵ "../node_context.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "../op_table.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "../utils.h"

#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/constant.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/eye.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/multiply.hpp>

ਨਾਮਾਕਾਸ਼ ov {
ਨਾਮਾਕਾਸ਼ frontend {
ਨਾਮਾਕਾਸ਼ ggml {
ਨਾਮਾਕਾਸ਼ op {

// GGML DIAG takes a 1D vector (ne0, 1, ne2, ne3) and produces a diagonal matrix
// of shape (ne0, ne0, ne2, ne3).
// In OV layout (ggml [ne0, ne1, ne2, ne3] → OV [ne3, ne2, ne1, ne0]):
//   input:  [ne3, ne2, 1, ne0]
//   output: [ne3, ne2, ne0, ne0]
// The diagonal: output[..., i, j] = input[..., 0, j] if i == j, else 0.
Output_ਸਦਿਸ਼ translate_diag(ਅਚਲ ਗੰਢ_Context & context) {
    ਅੰਕ_inputs_check(context, 1, 1);

    ਸਵੈਚਾਲਿਤ x = process_view_ਨਿਵੇਸ਼_new(context, 0);  // OV shape: [ne3, ne2, 1, ne0]

    ਸਵੈਚਾਲਿਤ n = get_dimensions(x.get_ਗੰਢ_shared_ptr(), {3});
    ਸਵੈਚਾਲਿਤ zero_diag = ov::op::v0::Constant::create(ov::element::i64, {}, {0});

    ਸਵੈਚਾਲਿਤ eye = std::make_shared<ov::op::v9::Eye>(n, n, zero_diag, x.get_element_type());
    ਸਵੈਚਾਲਿਤ res = std::make_shared<ov::op::v1::Multiply>(x, eye);

    ਵਾਪਸ rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
