#అన్తర్భావ "../node_context.h"
#అన్తర్భావ "../op_table.h"
#అన్తర్భావ "../utils.h"

#అన్తర్భావ <openvino/op/constant.hpp>
#అన్తర్భావ <openvino/op/eye.hpp>
#అన్తర్భావ <openvino/op/multiply.hpp>

నామాకాశ ov {
నామాకాశ frontend {
నామాకాశ ggml {
నామాకాశ op {

// GGML DIAG takes a 1D vector (ne0, 1, ne2, ne3) and produces a diagonal matrix
// of shape (ne0, ne0, ne2, ne3).
// In OV layout (ggml [ne0, ne1, ne2, ne3] → OV [ne3, ne2, ne1, ne0]):
//   input:  [ne3, ne2, 1, ne0]
//   output: [ne3, ne2, ne0, ne0]
// The diagonal: output[..., i, j] = input[..., 0, j] if i == j, else 0.
Output_సదిశ translate_diag(స్థిరరాశి కణుపు_Context & context) {
    అంకె_inputs_check(context, 1, 1);

    స్వయంచాలక x = process_view_ఆదానం_new(context, 0);  // OV shape: [ne3, ne2, 1, ne0]

    స్వయంచాలక n = get_dimensions(x.get_కణుపు_shared_ptr(), {3});
    స్వయంచాలక zero_diag = ov::op::v0::Constant::create(ov::element::i64, {}, {0});

    స్వయంచాలక eye = std::make_shared<ov::op::v9::Eye>(n, n, zero_diag, x.get_element_type());
    స్వయంచాలక res = std::make_shared<ov::op::v1::Multiply>(x, eye);

    తిరిగి rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
