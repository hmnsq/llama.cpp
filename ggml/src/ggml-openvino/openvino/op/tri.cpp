#అన్తర్భావ "../node_context.h"
#అన్తర్భావ "../op_table.h"
#అన్తర్భావ "../utils.h"

#అన్తర్భావ <openvino/op/constant.hpp>
#అన్తర్భావ <openvino/op/greater.hpp>
#అన్తర్భావ <openvino/op/greater_eq.hpp>
#అన్తర్భావ <openvino/op/less.hpp>
#అన్తర్భావ <openvino/op/less_eq.hpp>
#అన్తర్భావ <openvino/op/range.hpp>
#అన్తర్భావ <openvino/op/reshape.hpp>
#అన్తర్భావ <openvino/op/select.hpp>

నామాకాశ ov {
నామాకాశ frontend {
నామాకాశ ggml {
నామాకాశ op {

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
Output_సదిశ translate_tri(స్థిరరాశి కణుపు_Context & context) {
    అంకె_inputs_check(context, 1, 1);

    స్వయంచాలక x = context.get_ఆదానం(0);  // OV shape: [ne3, ne2, ne1, ne0]

    int32_t tri_type = context.get_output_op_params()[0];

    స్వయంచాలక shape = context.get_ఆదానం_shape(0).గమ్యం_shape();
    int64_t n = static_cast<int64_t>(shape[3]);  // ne0 == ne1

    // Build index range [0, 1, ..., n-1]
    స్వయంచాలక start = ov::op::v0::Constant::create(ov::element::i64, {}, {int64_t(0)});
    స్వయంచాలక stop  = ov::op::v0::Constant::create(ov::element::i64, {}, {n});
    స్వయంచాలక అడుగు  = ov::op::v0::Constant::create(ov::element::i64, {}, {int64_t(1)});
    స్వయంచాలక range = std::make_shared<ov::op::v4::Range>(start, stop, అడుగు, ov::element::i64);

    // col_idx shape [1, 1, 1, n] — broadcasts over batch and row dims
    స్వయంచాలక నిలువు_shape = ov::op::v0::Constant::create(ov::element::i64, {4}, std::vector<int64_t>{1, 1, 1, n});
    స్వయంచాలక నిలువు_idx   = std::make_shared<ov::op::v1::Reshape>(range, నిలువు_shape, అబద్ధం);

    // row_idx shape [1, 1, n, 1] — broadcasts over batch and col dims
    స్వయంచాలక అడ్డువరుస_shape = ov::op::v0::Constant::create(ov::element::i64, {4}, std::vector<int64_t>{1, 1, n, 1});
    స్వయంచాలక అడ్డువరుస_idx   = std::make_shared<ov::op::v1::Reshape>(range, అడ్డువరుస_shape, అబద్ధం);

    // Build boolean mask: true where element should be kept
    std::shared_ptr<ov::Node> ముసుగు;
    ఎంపిక (tri_type) {
        స్థితి 0:  // UPPER_DIAG: col >= row
            ముసుగు = std::make_shared<ov::op::v1::GreaterEqual>(నిలువు_idx, అడ్డువరుస_idx);
            ఆపు;
        స్థితి 1:  // UPPER: col > row
            ముసుగు = std::make_shared<ov::op::v1::Greater>(నిలువు_idx, అడ్డువరుస_idx);
            ఆపు;
        స్థితి 2:  // LOWER_DIAG: col <= row
            ముసుగు = std::make_shared<ov::op::v1::LessEqual>(నిలువు_idx, అడ్డువరుస_idx);
            ఆపు;
        స్థితి 3:  // LOWER: col < row
            ముసుగు = std::make_shared<ov::op::v1::Less>(నిలువు_idx, అడ్డువరుస_idx);
            ఆపు;
        అప్రమేయ:
            విసురు std::runtime_error("translate_tri: invalid tri_type " + std::to_string(tri_type));
    }

    స్వయంచాలక zero = ov::op::v0::Constant::create(ov::element::f32, {}, {0.0f});
    స్వయంచాలక res  = std::make_shared<ov::op::v1::Select>(ముసుగు, x, zero);

    తిరిగి rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
