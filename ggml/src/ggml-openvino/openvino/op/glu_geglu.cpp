#شامل "../node_context.h"
#شامل "../op_table.h"
#شامل "../utils.h"

#شامل <memory>
#شامل <openvino/core/node_output.hpp>
#شامل <openvino/op/clamp.hpp>
#شامل <openvino/op/constant.hpp>
#شامل <openvino/op/gelu.hpp>
#شامل <openvino/op/multiply.hpp>
#شامل <openvino/op/sigmoid.hpp>
#شامل <openvino/op/slice.hpp>

نامخلا ov {
نامخلا frontend {
نامخلا ggml {
نامخلا op {

Output_سمتیہ translate_glu_geglu(ثابت گرہ_Context & context) {
    عدد_inputs_check(context, 1, 2);

    ov::Output<ov::Node> src0;
    ov::Output<ov::Node> src1;
    اگر (context.get_داخلہ_سائز() == 2) {
        // Inputs may be VIEW slices of a combined gate_up tensor (MoE experts):
        // resolve them so each half has its real sliced shape, not the base tensor.
        src0 = process_view_داخلہ_new(context, 0);
        src1 = process_view_داخلہ_new(context, 1);
    } ورنہ {
        // GGML splits along ne[0] (OV last axis) using floor division: nc = ne[0] / 2.
        // Both halves are nc elements; if the dimension is odd, the last element is dropped.
        // Use Slice instead of Split to handle odd dimensions correctly.
        // Resolve a VIEW input (e.g. non-contiguous slice) to its real shape first.
        خودکار combined = process_view_داخلہ_new(context, 0);
        خودکار combined_shape = combined.get_partial_shape();
        int64_t آخری_dim_val = combined_shape[combined_shape.rank().get_لمبائی() - 1].get_لمبائی();
        int64_t nc = آخری_dim_val / 2;

        خودکار axis = ov::op::v0::Constant::create(ov::element::i64, {1}, {-1});
        خودکار قدم = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
        خودکار start0 = ov::op::v0::Constant::create(ov::element::i64, {1}, {0});
        خودکار stop0 = ov::op::v0::Constant::create(ov::element::i64, {1}, {nc});
        خودکار start1 = ov::op::v0::Constant::create(ov::element::i64, {1}, {nc});
        خودکار stop1 = ov::op::v0::Constant::create(ov::element::i64, {1}, {2 * nc});

        src0 = std::make_shared<ov::op::v8::قاش>(combined, start0, stop0, قدم, axis);
        src1 = std::make_shared<ov::op::v8::قاش>(combined, start1, stop1, قدم, axis);
    }

    int32_t * params = context.get_output_op_params();
    ثابت int32_t swapped = params[1];
    اگر (swapped) {
        std::swap(src0, src1);
    }

    اگر (context.is_static()) {
        // TODO: Temporary solution for NPU accuracy issue due to fp16 overflow
        // To be removed once permanent solution is implemented
        // Justification:
        // For |x| > 5, GELU(x) ≈ max(x, 0)  (behaves like ReLU)
        // So Clamp(-10, 10) only affects values where GELU would return ≈ x anyway.
        // The only loss: values > 10 get mapped to 10 instead of x.
        // In practice, FFN intermediates rarely exceed 10 after GEGLU gating.
        src0 = std::make_shared<ov::op::v0::Clamp>(src0, -10.0, 10.0);
    }
    خودکار gelu = std::make_shared<ov::op::v7::Gelu>(src0);
    خودکار res = std::make_shared<ov::op::v1::Multiply>(gelu, src1);

    واپس rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
