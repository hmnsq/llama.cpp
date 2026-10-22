#அந்தர்பாவ "../node_context.h"
#அந்தர்பாவ "../op_table.h"
#அந்தர்பாவ "../utils.h"

#அந்தர்பாவ <openvino/op/constant.hpp>
#அந்தர்பாவ <openvino/op/greater.hpp>
#அந்தர்பாவ <openvino/op/greater_eq.hpp>
#அந்தர்பாவ <openvino/op/less.hpp>
#அந்தர்பாவ <openvino/op/less_eq.hpp>
#அந்தர்பாவ <openvino/op/range.hpp>
#அந்தர்பாவ <openvino/op/reshape.hpp>
#அந்தர்பாவ <openvino/op/select.hpp>

நாமாகாஶ ov {
நாமாகாஶ frontend {
நாமாகாஶ ggml {
நாமாகாஶ op {

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
OutputVector translate_tri(மாறிலி முனை_Context & context) {
    இலக்கம்_inputs_check(context, 1, 1);

    auto x = context.get_உள்ளீடு(0);  // OV shape: [ne3, ne2, ne1, ne0]

    int32_t tri_type = context.get_output_op_params()[0];

    auto shape = context.get_உள்ளீடு_shape(0).to_shape();
    int64_t n = static_cast<int64_t>(shape[3]);  // ne0 == ne1

    // Build index range [0, 1, ..., n-1]
    auto start = ov::op::v0::Constant::create(ov::element::i64, {}, {int64_t(0)});
    auto stop  = ov::op::v0::Constant::create(ov::element::i64, {}, {n});
    auto step  = ov::op::v0::Constant::create(ov::element::i64, {}, {int64_t(1)});
    auto range = std::make_shared<ov::op::v4::Range>(start, stop, step, ov::element::i64);

    // col_idx shape [1, 1, 1, n] — broadcasts over batch and row dims
    auto col_shape = ov::op::v0::Constant::create(ov::element::i64, {4}, std::vector<int64_t>{1, 1, 1, n});
    auto col_idx   = std::make_shared<ov::op::v1::Reshape>(range, col_shape, பொய்);

    // row_idx shape [1, 1, n, 1] — broadcasts over batch and col dims
    auto row_shape = ov::op::v0::Constant::create(ov::element::i64, {4}, std::vector<int64_t>{1, 1, n, 1});
    auto row_idx   = std::make_shared<ov::op::v1::Reshape>(range, row_shape, பொய்);

    // Build boolean mask: true where element should be kept
    std::shared_ptr<ov::Node> mask;
    தேர்வு (tri_type) {
        நிலை 0:  // UPPER_DIAG: col >= row
            mask = std::make_shared<ov::op::v1::GreaterEqual>(col_idx, row_idx);
            நிறுத்து;
        நிலை 1:  // UPPER: col > row
            mask = std::make_shared<ov::op::v1::Greater>(col_idx, row_idx);
            நிறுத்து;
        நிலை 2:  // LOWER_DIAG: col <= row
            mask = std::make_shared<ov::op::v1::LessEqual>(col_idx, row_idx);
            நிறுத்து;
        நிலை 3:  // LOWER: col < row
            mask = std::make_shared<ov::op::v1::Less>(col_idx, row_idx);
            நிறுத்து;
        இயல்பு:
            எறி std::runtime_error("translate_tri: invalid tri_type " + std::to_string(tri_type));
    }

    auto zero = ov::op::v0::Constant::create(ov::element::f32, {}, {0.0f});
    auto res  = std::make_shared<ov::op::v1::Select>(mask, x, zero);

    திரும்பு rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
