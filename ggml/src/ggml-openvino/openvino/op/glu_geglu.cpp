#అన్తర్భావ "../node_context.h"
#అన్తర్భావ "../op_table.h"
#అన్తర్భావ "../utils.h"

#అన్తర్భావ <memory>
#అన్తర్భావ <openvino/core/node_output.hpp>
#అన్తర్భావ <openvino/op/clamp.hpp>
#అన్తర్భావ <openvino/op/constant.hpp>
#అన్తర్భావ <openvino/op/gelu.hpp>
#అన్తర్భావ <openvino/op/multiply.hpp>
#అన్తర్భావ <openvino/op/sigmoid.hpp>
#అన్తర్భావ <openvino/op/slice.hpp>

నామాకాశ ov {
నామాకాశ frontend {
నామాకాశ ggml {
నామాకాశ op {

Output_సదిశ translate_glu_geglu(స్థిరరాశి కణుపు_Context & context) {
    అంకె_inputs_check(context, 1, 2);

    ov::Output<ov::Node> src0;
    ov::Output<ov::Node> src1;
    ఒకవేళ (context.get_ఆదానం_కొలత() == 2) {
        // Inputs may be VIEW slices of a combined gate_up tensor (MoE experts):
        // resolve them so each half has its real sliced shape, not the base tensor.
        src0 = process_view_ఆదానం_new(context, 0);
        src1 = process_view_ఆదానం_new(context, 1);
    } లేకపోతే {
        // GGML splits along ne[0] (OV last axis) using floor division: nc = ne[0] / 2.
        // Both halves are nc elements; if the dimension is odd, the last element is dropped.
        // Use Slice instead of Split to handle odd dimensions correctly.
        // Resolve a VIEW input (e.g. non-contiguous slice) to its real shape first.
        స్వయంచాలక combined = process_view_ఆదానం_new(context, 0);
        స్వయంచాలక combined_shape = combined.get_partial_shape();
        int64_t చివరి_dim_val = combined_shape[combined_shape.rank().get_నిడివి() - 1].get_నిడివి();
        int64_t nc = చివరి_dim_val / 2;

        స్వయంచాలక axis = ov::op::v0::Constant::create(ov::element::i64, {1}, {-1});
        స్వయంచాలక అడుగు = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
        స్వయంచాలక start0 = ov::op::v0::Constant::create(ov::element::i64, {1}, {0});
        స్వయంచాలక stop0 = ov::op::v0::Constant::create(ov::element::i64, {1}, {nc});
        స్వయంచాలక start1 = ov::op::v0::Constant::create(ov::element::i64, {1}, {nc});
        స్వయంచాలక stop1 = ov::op::v0::Constant::create(ov::element::i64, {1}, {2 * nc});

        src0 = std::make_shared<ov::op::v8::ముక్క>(combined, start0, stop0, అడుగు, axis);
        src1 = std::make_shared<ov::op::v8::ముక్క>(combined, start1, stop1, అడుగు, axis);
    }

    int32_t * params = context.get_output_op_params();
    స్థిరరాశి int32_t swapped = params[1];
    ఒకవేళ (swapped) {
        std::swap(src0, src1);
    }

    ఒకవేళ (context.is_static()) {
        // TODO: Temporary solution for NPU accuracy issue due to fp16 overflow
        // To be removed once permanent solution is implemented
        // Justification:
        // For |x| > 5, GELU(x) ≈ max(x, 0)  (behaves like ReLU)
        // So Clamp(-10, 10) only affects values where GELU would return ≈ x anyway.
        // The only loss: values > 10 get mapped to 10 instead of x.
        // In practice, FFN intermediates rarely exceed 10 after GEGLU gating.
        src0 = std::make_shared<ov::op::v0::Clamp>(src0, -10.0, 10.0);
    }
    స్వయంచాలక gelu = std::make_shared<ov::op::v7::Gelu>(src0);
    స్వయంచాలక res = std::make_shared<ov::op::v1::Multiply>(gelu, src1);

    తిరిగి rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
