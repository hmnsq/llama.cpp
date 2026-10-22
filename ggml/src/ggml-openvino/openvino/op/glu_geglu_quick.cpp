#అన్తర్భావ "../node_context.h"
#అన్తర్భావ "../op_table.h"
#అన్తర్భావ "../utils.h"

#అన్తర్భావ <memory>
#అన్తర్భావ <openvino/core/node_output.hpp>
#అన్తర్భావ <openvino/op/constant.hpp>
#అన్తర్భావ <openvino/op/multiply.hpp>
#అన్తర్భావ <openvino/op/slice.hpp>
#అన్తర్భావ <openvino/op/swish.hpp>

నామాకాశ ov {
నామాకాశ frontend {
నామాకాశ ggml {
నామాకాశ op {

Output_సదిశ translate_glu_geglu_శీఘ్ర(స్థిరరాశి కణుపు_Context & context) {
    అంకె_inputs_check(context, 1, 2);

    ov::Output<ov::Node> src0;
    ov::Output<ov::Node> src1;
    ఒకవేళ (context.get_ఆదానం_కొలత() == 2) {
        src0 = process_view_ఆదానం_new(context, 0);
        src1 = process_view_ఆదానం_new(context, 1);
    } లేకపోతే {
        // split along last axis, nc = ne[0] / 2
        స్వయంచాలక combined = process_view_ఆదానం_new(context, 0);
        స్వయంచాలక combined_shape = combined.get_partial_shape();
        int64_t చివరి_dim_val = combined_shape[combined_shape.rank().get_నిడివి() - 1].get_నిడివి();
        int64_t nc = చివరి_dim_val / 2;

        స్వయంచాలక axis  = ov::op::v0::Constant::create(ov::element::i64, {1}, {-1});
        స్వయంచాలక అడుగు  = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
        స్వయంచాలక start0 = ov::op::v0::Constant::create(ov::element::i64, {1}, {0});
        స్వయంచాలక stop0  = ov::op::v0::Constant::create(ov::element::i64, {1}, {nc});
        స్వయంచాలక start1 = ov::op::v0::Constant::create(ov::element::i64, {1}, {nc});
        స్వయంచాలక stop1  = ov::op::v0::Constant::create(ov::element::i64, {1}, {2 * nc});

        src0 = std::make_shared<ov::op::v8::ముక్క>(combined, start0, stop0, అడుగు, axis);
        src1 = std::make_shared<ov::op::v8::ముక్క>(combined, start1, stop1, అడుగు, axis);
    }

    int32_t * params = context.get_output_op_params();
    స్థిరరాశి int32_t swapped = params[1];
    ఒకవేళ (swapped) {
        std::swap(src0, src1);
    }

    // GELU_QUICK(x) = x * sigmoid(1.702 * x)
    // Create the constant in the same type as src0 to avoid f16/f32 mismatch.
    స్వయంచాలక ఆదానం_type = src0.get_element_type();
    స్వయంచాలక coef = ov::op::v0::Constant::create(ఆదానం_type, ov::Shape{}, {1.702f});
    స్వయంచాలక gated   = std::make_shared<ov::op::v4::Swish>(src0, coef);
    స్వయంచాలక res     = std::make_shared<ov::op::v1::Multiply>(gated, src1);

    తిరిగి rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
