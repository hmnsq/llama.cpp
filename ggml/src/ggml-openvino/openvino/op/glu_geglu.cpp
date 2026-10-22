#ਅਨ੍ਤਰ੍ਭਾਵ "../node_context.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "../op_table.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "../utils.h"

#ਅਨ੍ਤਰ੍ਭਾਵ <memory>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/core/node_output.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/clamp.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/constant.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/gelu.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/multiply.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/sigmoid.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/slice.hpp>

ਨਾਮਾਕਾਸ਼ ov {
ਨਾਮਾਕਾਸ਼ frontend {
ਨਾਮਾਕਾਸ਼ ggml {
ਨਾਮਾਕਾਸ਼ op {

Output_ਸਦਿਸ਼ translate_glu_geglu(ਅਚਲ ਗੰਢ_Context & context) {
    ਅੰਕ_inputs_check(context, 1, 2);

    ov::Output<ov::Node> src0;
    ov::Output<ov::Node> src1;
    ਜੇ (context.get_ਨਿਵੇਸ਼_ਮਾਪ() == 2) {
        // Inputs may be VIEW slices of a combined gate_up tensor (MoE experts):
        // resolve them so each half has its real sliced shape, not the base tensor.
        src0 = process_view_ਨਿਵੇਸ਼_new(context, 0);
        src1 = process_view_ਨਿਵੇਸ਼_new(context, 1);
    } ਨਹੀਂਤਾਂ {
        // GGML splits along ne[0] (OV last axis) using floor division: nc = ne[0] / 2.
        // Both halves are nc elements; if the dimension is odd, the last element is dropped.
        // Use Slice instead of Split to handle odd dimensions correctly.
        // Resolve a VIEW input (e.g. non-contiguous slice) to its real shape first.
        ਸਵੈਚਾਲਿਤ combined = process_view_ਨਿਵੇਸ਼_new(context, 0);
        ਸਵੈਚਾਲਿਤ combined_shape = combined.get_partial_shape();
        int64_t ਆਖਰੀ_dim_val = combined_shape[combined_shape.rank().get_ਲੰਮਾਈ() - 1].get_ਲੰਮਾਈ();
        int64_t nc = ਆਖਰੀ_dim_val / 2;

        ਸਵੈਚਾਲਿਤ axis = ov::op::v0::Constant::create(ov::element::i64, {1}, {-1});
        ਸਵੈਚਾਲਿਤ ਕਦਮ = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
        ਸਵੈਚਾਲਿਤ start0 = ov::op::v0::Constant::create(ov::element::i64, {1}, {0});
        ਸਵੈਚਾਲਿਤ stop0 = ov::op::v0::Constant::create(ov::element::i64, {1}, {nc});
        ਸਵੈਚਾਲਿਤ start1 = ov::op::v0::Constant::create(ov::element::i64, {1}, {nc});
        ਸਵੈਚਾਲਿਤ stop1 = ov::op::v0::Constant::create(ov::element::i64, {1}, {2 * nc});

        src0 = std::make_shared<ov::op::v8::ਟੁਕੜਾ>(combined, start0, stop0, ਕਦਮ, axis);
        src1 = std::make_shared<ov::op::v8::ਟੁਕੜਾ>(combined, start1, stop1, ਕਦਮ, axis);
    }

    int32_t * params = context.get_output_op_params();
    ਅਚਲ int32_t swapped = params[1];
    ਜੇ (swapped) {
        std::swap(src0, src1);
    }

    ਜੇ (context.is_static()) {
        // TODO: Temporary solution for NPU accuracy issue due to fp16 overflow
        // To be removed once permanent solution is implemented
        // Justification:
        // For |x| > 5, GELU(x) ≈ max(x, 0)  (behaves like ReLU)
        // So Clamp(-10, 10) only affects values where GELU would return ≈ x anyway.
        // The only loss: values > 10 get mapped to 10 instead of x.
        // In practice, FFN intermediates rarely exceed 10 after GEGLU gating.
        src0 = std::make_shared<ov::op::v0::Clamp>(src0, -10.0, 10.0);
    }
    ਸਵੈਚਾਲਿਤ gelu = std::make_shared<ov::op::v7::Gelu>(src0);
    ਸਵੈਚਾਲਿਤ res = std::make_shared<ov::op::v1::Multiply>(gelu, src1);

    ਵਾਪਸ rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
