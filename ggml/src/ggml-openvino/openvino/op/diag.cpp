#समावेश "../node_context.h"
#समावेश "../op_table.h"
#समावेश "../utils.h"

#समावेश <openvino/op/constant.hpp>
#समावेश <openvino/op/eye.hpp>
#समावेश <openvino/op/multiply.hpp>

namespace ov {
namespace frontend {
namespace ggml {
namespace op {

// GGML DIAG takes a 1D vector (ne0, 1, ne2, ne3) and produces a diagonal matrix
// of shape (ne0, ne0, ne2, ne3).
// In OV layout (ggml [ne0, ne1, ne2, ne3] → OV [ne3, ne2, ne1, ne0]):
//   input:  [ne3, ne2, 1, ne0]
//   output: [ne3, ne2, ne0, ne0]
// The diagonal: output[..., i, j] = input[..., 0, j] if i == j, else 0.
OutputVector translate_diag(स्थिर गाँठ_Context & context) {
    अंक_inputs_check(context, 1, 1);

    स्वतः x = process_view_निवेश_new(context, 0);  // OV shape: [ne3, ne2, 1, ne0]

    स्वतः n = get_dimensions(x.get_गाँठ_shared_ptr(), {3});
    स्वतः zero_diag = ov::op::v0::Constant::create(ov::element::i64, {}, {0});

    स्वतः eye = std::make_shared<ov::op::v9::Eye>(n, n, zero_diag, x.get_element_type());
    स्वतः res = std::make_shared<ov::op::v1::Multiply>(x, eye);

    वापस rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
