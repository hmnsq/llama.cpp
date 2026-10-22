#अन्तर्भाव "../node_context.h"
#अन्तर्भाव "../op_table.h"
#अन्तर्भाव "../utils.h"

#अन्तर्भाव <memory>
#अन्तर्भाव <openvino/core/node_output.hpp>
#अन्तर्भाव <openvino/op/clamp.hpp>
#अन्तर्भाव <openvino/op/constant.hpp>
#अन्तर्भाव <openvino/op/gelu.hpp>
#अन्तर्भाव <openvino/op/multiply.hpp>
#अन्तर्भाव <openvino/op/sigmoid.hpp>
#अन्तर्भाव <openvino/op/slice.hpp>

नामाकाश ov {
नामाकाश frontend {
नामाकाश ggml {
नामाकाश op {

Output_सदिशम् translate_glu_geglu(ध्रुव ग्रन्थिः_Context & context) {
    अङ्कः_inputs_check(context, 1, 2);

    ov::Output<ov::Node> src0;
    ov::Output<ov::Node> src1;
    यदि (context.get_निवेशः_आकार() == 2) {
        // Inputs may be VIEW slices of a combined gate_up tensor (MoE experts):
        // resolve them so each half has its real sliced shape, not the base tensor.
        src0 = process_view_निवेशः_new(context, 0);
        src1 = process_view_निवेशः_new(context, 1);
    } अन्यथा {
        // GGML splits along ne[0] (OV last axis) using floor division: nc = ne[0] / 2.
        // Both halves are nc elements; if the dimension is odd, the last element is dropped.
        // Use Slice instead of Split to handle odd dimensions correctly.
        // Resolve a VIEW input (e.g. non-contiguous slice) to its real shape first.
        स्वचालित combined = process_view_निवेशः_new(context, 0);
        स्वचालित combined_shape = combined.get_partial_shape();
        int64_t चरमः_dim_val = combined_shape[combined_shape.rank().get_दीर्घता() - 1].get_दीर्घता();
        int64_t nc = चरमः_dim_val / 2;

        स्वचालित axis = ov::op::v0::Constant::create(ov::element::i64, {1}, {-1});
        स्वचालित चरणः = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
        स्वचालित start0 = ov::op::v0::Constant::create(ov::element::i64, {1}, {0});
        स्वचालित stop0 = ov::op::v0::Constant::create(ov::element::i64, {1}, {nc});
        स्वचालित start1 = ov::op::v0::Constant::create(ov::element::i64, {1}, {nc});
        स्वचालित stop1 = ov::op::v0::Constant::create(ov::element::i64, {1}, {2 * nc});

        src0 = std::make_shared<ov::op::v8::खण्डिका>(combined, start0, stop0, चरणः, axis);
        src1 = std::make_shared<ov::op::v8::खण्डिका>(combined, start1, stop1, चरणः, axis);
    }

    int32_t * params = context.get_output_op_params();
    ध्रुव int32_t swapped = params[1];
    यदि (swapped) {
        std::swap(src0, src1);
    }

    यदि (context.is_static()) {
        // TODO: Temporary solution for NPU accuracy issue due to fp16 overflow
        // To be removed once permanent solution is implemented
        // Justification:
        // For |x| > 5, GELU(x) ≈ max(x, 0)  (behaves like ReLU)
        // So Clamp(-10, 10) only affects values where GELU would return ≈ x anyway.
        // The only loss: values > 10 get mapped to 10 instead of x.
        // In practice, FFN intermediates rarely exceed 10 after GEGLU gating.
        src0 = std::make_shared<ov::op::v0::Clamp>(src0, -10.0, 10.0);
    }
    स्वचालित gelu = std::make_shared<ov::op::v7::Gelu>(src0);
    स्वचालित res = std::make_shared<ov::op::v1::Multiply>(gelu, src1);

    प्रत्यागम rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
