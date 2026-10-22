#ಅನ್ತರ್ಭಾವ "../node_context.h"
#ಅನ್ತರ್ಭಾವ "../op_table.h"
#ಅನ್ತರ್ಭಾವ "../utils.h"

#ಅನ್ತರ್ಭಾವ <openvino/op/constant.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/eye.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/multiply.hpp>

ನಾಮಾಕಾಶ ov {
ನಾಮಾಕಾಶ frontend {
ನಾಮಾಕಾಶ ggml {
ನಾಮಾಕಾಶ op {

// GGML DIAG takes a 1D vector (ne0, 1, ne2, ne3) and produces a diagonal matrix
// of shape (ne0, ne0, ne2, ne3).
// In OV layout (ggml [ne0, ne1, ne2, ne3] → OV [ne3, ne2, ne1, ne0]):
//   input:  [ne3, ne2, 1, ne0]
//   output: [ne3, ne2, ne0, ne0]
// The diagonal: output[..., i, j] = input[..., 0, j] if i == j, else 0.
Output_ಸದಿಶ translate_diag(ಸ್ಥಿರಾಂಕ ಗಂಟು_Context & context) {
    ಅಂಕೆ_inputs_check(context, 1, 1);

    ಸ್ವಯಂಚಾಲಿತ x = process_view_ಒಳಹರಿವು_new(context, 0);  // OV shape: [ne3, ne2, 1, ne0]

    ಸ್ವಯಂಚಾಲಿತ n = get_dimensions(x.get_ಗಂಟು_shared_ptr(), {3});
    ಸ್ವಯಂಚಾಲಿತ zero_diag = ov::op::v0::Constant::create(ov::element::i64, {}, {0});

    ಸ್ವಯಂಚಾಲಿತ eye = std::make_shared<ov::op::v9::Eye>(n, n, zero_diag, x.get_element_type());
    ಸ್ವಯಂಚಾಲಿತ res = std::make_shared<ov::op::v1::Multiply>(x, eye);

    ಹಿಂತಿರುಗಿ rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
