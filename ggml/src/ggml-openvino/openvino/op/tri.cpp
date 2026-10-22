#ಅನ್ತರ್ಭಾವ "../node_context.h"
#ಅನ್ತರ್ಭಾವ "../op_table.h"
#ಅನ್ತರ್ಭಾವ "../utils.h"

#ಅನ್ತರ್ಭಾವ <openvino/op/constant.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/greater.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/greater_eq.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/less.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/less_eq.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/range.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/reshape.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/select.hpp>

ನಾಮಾಕಾಶ ov {
ನಾಮಾಕಾಶ frontend {
ನಾಮಾಕಾಶ ggml {
ನಾಮಾಕಾಶ op {

// GGML TRI zeroes out elements outside a triangular region of a square matrix.
// The type param (stored in op_params[0]) maps to ggml_tri_type:
//   0 = UPPER_DIAG : keep where col >= row
//   1 = UPPER      : keep where col >  row
//   2 = LOWER_DIAG : keep where col <= row
//   3 = LOWER      : keep where col <  row
//
// In OV layout (ggml [ne0, ne1, ne2, ne3] → OV [ne3, ne2, ne1, ne0]):
//   ggml dim 0 (ne0, cols) → OV axis 3
//   ggml dim 1 (ne1, rows) → OV axis 2
// The matrix is square so ne0 == ne1.
Output_ಸದಿಶ translate_tri(ಸ್ಥಿರಾಂಕ ಗಂಟು_Context & context) {
    ಅಂಕೆ_inputs_check(context, 1, 1);

    ಸ್ವಯಂಚಾಲಿತ x = context.get_ಒಳಹರಿವು(0);  // OV shape: [ne3, ne2, ne1, ne0]

    int32_t tri_type = context.get_output_op_params()[0];

    ಸ್ವಯಂಚಾಲಿತ shape = context.get_ಒಳಹರಿವು_shape(0).ಗಮ್ಯ_shape();
    int64_t n = static_cast<int64_t>(shape[3]);  // ne0 == ne1

    // Build index range [0, 1, ..., n-1]
    ಸ್ವಯಂಚಾಲಿತ start = ov::op::v0::Constant::create(ov::element::i64, {}, {int64_t(0)});
    ಸ್ವಯಂಚಾಲಿತ stop  = ov::op::v0::Constant::create(ov::element::i64, {}, {n});
    ಸ್ವಯಂಚಾಲಿತ ಹೆಜ್ಜೆ  = ov::op::v0::Constant::create(ov::element::i64, {}, {int64_t(1)});
    ಸ್ವಯಂಚಾಲಿತ range = std::make_shared<ov::op::v4::Range>(start, stop, ಹೆಜ್ಜೆ, ov::element::i64);

    // col_idx shape [1, 1, 1, n] — broadcasts over batch and row dims
    ಸ್ವಯಂಚಾಲಿತ ಲಂಬಸಾಲು_shape = ov::op::v0::Constant::create(ov::element::i64, {4}, std::vector<int64_t>{1, 1, 1, n});
    ಸ್ವಯಂಚಾಲಿತ ಲಂಬಸಾಲು_idx   = std::make_shared<ov::op::v1::Reshape>(range, ಲಂಬಸಾಲು_shape, ಸುಳ್ಳು);

    // row_idx shape [1, 1, n, 1] — broadcasts over batch and col dims
    ಸ್ವಯಂಚಾಲಿತ ಅಡ್ಡಸಾಲು_shape = ov::op::v0::Constant::create(ov::element::i64, {4}, std::vector<int64_t>{1, 1, n, 1});
    ಸ್ವಯಂಚಾಲಿತ ಅಡ್ಡಸಾಲು_idx   = std::make_shared<ov::op::v1::Reshape>(range, ಅಡ್ಡಸಾಲು_shape, ಸುಳ್ಳು);

    // Build boolean mask: true where element should be kept
    std::shared_ptr<ov::Node> ಮುಖವಾಡ;
    ಆಯ್ಕೆ (tri_type) {
        ಸ್ಥಿತಿ 0:  // UPPER_DIAG: col >= row
            ಮುಖವಾಡ = std::make_shared<ov::op::v1::GreaterEqual>(ಲಂಬಸಾಲು_idx, ಅಡ್ಡಸಾಲು_idx);
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ 1:  // UPPER: col > row
            ಮುಖವಾಡ = std::make_shared<ov::op::v1::Greater>(ಲಂಬಸಾಲು_idx, ಅಡ್ಡಸಾಲು_idx);
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ 2:  // LOWER_DIAG: col <= row
            ಮುಖವಾಡ = std::make_shared<ov::op::v1::LessEqual>(ಲಂಬಸಾಲು_idx, ಅಡ್ಡಸಾಲು_idx);
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ 3:  // LOWER: col < row
            ಮುಖವಾಡ = std::make_shared<ov::op::v1::Less>(ಲಂಬಸಾಲು_idx, ಅಡ್ಡಸಾಲು_idx);
            ನಿಲ್ಲಿಸು;
        ಪೂರ್ವನಿಯೋಜಿತ:
            ಎಸೆ std::runtime_error("translate_tri: invalid tri_type " + std::to_string(tri_type));
    }

    ಸ್ವಯಂಚಾಲಿತ zero = ov::op::v0::Constant::create(ov::element::f32, {}, {0.0f});
    ಸ್ವಯಂಚಾಲಿತ res  = std::make_shared<ov::op::v1::Select>(ಮುಖವಾಡ, x, zero);

    ಹಿಂತಿರುಗಿ rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
