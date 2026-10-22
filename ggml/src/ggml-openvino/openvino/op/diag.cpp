#شامل "../node_context.h"
#شامل "../op_table.h"
#شامل "../utils.h"

#شامل <openvino/op/constant.hpp>
#شامل <openvino/op/eye.hpp>
#شامل <openvino/op/multiply.hpp>

نامخلا ov {
نامخلا frontend {
نامخلا ggml {
نامخلا op {

// GGML DIAG takes a 1D vector (ne0, 1, ne2, ne3) and produces a diagonal matrix
// of shape (ne0, ne0, ne2, ne3).
// In OV layout (ggml [ne0, ne1, ne2, ne3] → OV [ne3, ne2, ne1, ne0]):
//   input:  [ne3, ne2, 1, ne0]
//   output: [ne3, ne2, ne0, ne0]
// The diagonal: output[..., i, j] = input[..., 0, j] if i == j, else 0.
Output_سمتیہ translate_diag(ثابت گرہ_Context & context) {
    عدد_inputs_check(context, 1, 1);

    خودکار x = process_view_داخلہ_new(context, 0);  // OV shape: [ne3, ne2, 1, ne0]

    خودکار n = get_dimensions(x.get_گرہ_shared_ptr(), {3});
    خودکار zero_diag = ov::op::v0::Constant::create(ov::element::i64, {}, {0});

    خودکار eye = std::make_shared<ov::op::v9::Eye>(n, n, zero_diag, x.get_element_type());
    خودکار res = std::make_shared<ov::op::v1::Multiply>(x, eye);

    واپس rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
