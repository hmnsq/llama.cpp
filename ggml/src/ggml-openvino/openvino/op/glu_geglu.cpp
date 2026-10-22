#ضمن "../node_context.h"
#ضمن "../op_table.h"
#ضمن "../utils.h"

#ضمن <memory>
#ضمن <openvino/core/node_output.hpp>
#ضمن <openvino/op/clamp.hpp>
#ضمن <openvino/op/constant.hpp>
#ضمن <openvino/op/gelu.hpp>
#ضمن <openvino/op/multiply.hpp>
#ضمن <openvino/op/sigmoid.hpp>
#ضمن <openvino/op/slice.hpp>

نطاق ov {
نطاق frontend {
نطاق ggml {
نطاق op {

Output_ویکټور translate_glu_geglu(ثابت غوټه_Context & context) {
    عدد_inputs_check(context, 1, 2);

    ov::Output<ov::Node> src0;
    ov::Output<ov::Node> src1;
    إذا (context.get_ننوت_اندازه() == 2) {
        // Inputs may be VIEW slices of a combined gate_up tensor (MoE experts):
        // resolve them so each half has its real sliced shape, not the base tensor.
        src0 = process_view_ننوت_new(context, 0);
        src1 = process_view_ننوت_new(context, 1);
    } وإلا {
        // GGML splits along ne[0] (OV last axis) using floor division: nc = ne[0] / 2.
        // Both halves are nc elements; if the dimension is odd, the last element is dropped.
        // Use Slice instead of Split to handle odd dimensions correctly.
        // Resolve a VIEW input (e.g. non-contiguous slice) to its real shape first.
        خپلکاری combined = process_view_ننوت_new(context, 0);
        خپلکاری combined_shape = combined.get_partial_shape();
        int64_t وروستی_dim_val = combined_shape[combined_shape.rank().get_اوږدتیا() - 1].get_اوږدتیا();
        int64_t nc = وروستی_dim_val / 2;

        خپلکاری axis = ov::op::v0::Constant::create(ov::element::i64, {1}, {-1});
        خپلکاری ګام = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
        خپلکاری start0 = ov::op::v0::Constant::create(ov::element::i64, {1}, {0});
        خپلکاری stop0 = ov::op::v0::Constant::create(ov::element::i64, {1}, {nc});
        خپلکاری start1 = ov::op::v0::Constant::create(ov::element::i64, {1}, {nc});
        خپلکاری stop1 = ov::op::v0::Constant::create(ov::element::i64, {1}, {2 * nc});

        src0 = std::make_shared<ov::op::v8::ټوټه>(combined, start0, stop0, ګام, axis);
        src1 = std::make_shared<ov::op::v8::ټوټه>(combined, start1, stop1, ګام, axis);
    }

    int32_t * params = context.get_output_op_params();
    ثابت int32_t swapped = params[1];
    إذا (swapped) {
        std::swap(src0, src1);
    }

    إذا (context.is_static()) {
        // TODO: Temporary solution for NPU accuracy issue due to fp16 overflow
        // To be removed once permanent solution is implemented
        // Justification:
        // For |x| > 5, GELU(x) ≈ max(x, 0)  (behaves like ReLU)
        // So Clamp(-10, 10) only affects values where GELU would return ≈ x anyway.
        // The only loss: values > 10 get mapped to 10 instead of x.
        // In practice, FFN intermediates rarely exceed 10 after GEGLU gating.
        src0 = std::make_shared<ov::op::v0::Clamp>(src0, -10.0, 10.0);
    }
    خپلکاری gelu = std::make_shared<ov::op::v7::Gelu>(src0);
    خپلکاری res = std::make_shared<ov::op::v1::Multiply>(gelu, src1);

    أرجع rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
