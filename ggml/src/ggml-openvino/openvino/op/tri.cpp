#شامل "../node_context.h"
#شامل "../op_table.h"
#شامل "../utils.h"

#شامل <openvino/op/constant.hpp>
#شامل <openvino/op/greater.hpp>
#شامل <openvino/op/greater_eq.hpp>
#شامل <openvino/op/less.hpp>
#شامل <openvino/op/less_eq.hpp>
#شامل <openvino/op/range.hpp>
#شامل <openvino/op/reshape.hpp>
#شامل <openvino/op/select.hpp>

نامخلا ov {
نامخلا frontend {
نامخلا ggml {
نامخلا op {

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
Output_سمتیہ translate_tri(ثابت گرہ_Context & context) {
    عدد_inputs_check(context, 1, 1);

    خودکار x = context.get_داخلہ(0);  // OV shape: [ne3, ne2, ne1, ne0]

    int32_t tri_type = context.get_output_op_params()[0];

    خودکار shape = context.get_داخلہ_shape(0).مقصد_shape();
    int64_t n = static_cast<int64_t>(shape[3]);  // ne0 == ne1

    // Build index range [0, 1, ..., n-1]
    خودکار start = ov::op::v0::Constant::create(ov::element::i64, {}, {int64_t(0)});
    خودکار stop  = ov::op::v0::Constant::create(ov::element::i64, {}, {n});
    خودکار قدم  = ov::op::v0::Constant::create(ov::element::i64, {}, {int64_t(1)});
    خودکار range = std::make_shared<ov::op::v4::Range>(start, stop, قدم, ov::element::i64);

    // col_idx shape [1, 1, 1, n] — broadcasts over batch and row dims
    خودکار ستون_shape = ov::op::v0::Constant::create(ov::element::i64, {4}, std::vector<int64_t>{1, 1, 1, n});
    خودکار ستون_idx   = std::make_shared<ov::op::v1::Reshape>(range, ستون_shape, جھوٹ);

    // row_idx shape [1, 1, n, 1] — broadcasts over batch and col dims
    خودکار صف_shape = ov::op::v0::Constant::create(ov::element::i64, {4}, std::vector<int64_t>{1, 1, n, 1});
    خودکار صف_idx   = std::make_shared<ov::op::v1::Reshape>(range, صف_shape, جھوٹ);

    // Build boolean mask: true where element should be kept
    std::shared_ptr<ov::Node> نقاب;
    انتخاب (tri_type) {
        حالت 0:  // UPPER_DIAG: col >= row
            نقاب = std::make_shared<ov::op::v1::GreaterEqual>(ستون_idx, صف_idx);
            توڑو;
        حالت 1:  // UPPER: col > row
            نقاب = std::make_shared<ov::op::v1::Greater>(ستون_idx, صف_idx);
            توڑو;
        حالت 2:  // LOWER_DIAG: col <= row
            نقاب = std::make_shared<ov::op::v1::LessEqual>(ستون_idx, صف_idx);
            توڑو;
        حالت 3:  // LOWER: col < row
            نقاب = std::make_shared<ov::op::v1::Less>(ستون_idx, صف_idx);
            توڑو;
        طےشدہ:
            پھینکو std::runtime_error("translate_tri: invalid tri_type " + std::to_string(tri_type));
    }

    خودکار zero = ov::op::v0::Constant::create(ov::element::f32, {}, {0.0f});
    خودکار res  = std::make_shared<ov::op::v1::Select>(نقاب, x, zero);

    واپس rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
