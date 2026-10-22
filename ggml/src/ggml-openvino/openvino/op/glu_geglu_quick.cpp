#अन्तर्भाव "../node_context.h"
#अन्तर्भाव "../op_table.h"
#अन्तर्भाव "../utils.h"

#अन्तर्भाव <memory>
#अन्तर्भाव <openvino/core/node_output.hpp>
#अन्तर्भाव <openvino/op/constant.hpp>
#अन्तर्भाव <openvino/op/multiply.hpp>
#अन्तर्भाव <openvino/op/slice.hpp>
#अन्तर्भाव <openvino/op/swish.hpp>

नामाकाश ov {
नामाकाश frontend {
नामाकाश ggml {
नामाकाश op {

Output_सदिशम् translate_glu_geglu_शीघ्र(ध्रुव ग्रन्थिः_Context & context) {
    अङ्कः_inputs_check(context, 1, 2);

    ov::Output<ov::Node> src0;
    ov::Output<ov::Node> src1;
    यदि (context.get_निवेशः_आकार() == 2) {
        src0 = process_view_निवेशः_new(context, 0);
        src1 = process_view_निवेशः_new(context, 1);
    } अन्यथा {
        // split along last axis, nc = ne[0] / 2
        स्वचालित combined = process_view_निवेशः_new(context, 0);
        स्वचालित combined_shape = combined.get_partial_shape();
        int64_t चरमः_dim_val = combined_shape[combined_shape.rank().get_दीर्घता() - 1].get_दीर्घता();
        int64_t nc = चरमः_dim_val / 2;

        स्वचालित axis  = ov::op::v0::Constant::create(ov::element::i64, {1}, {-1});
        स्वचालित चरणः  = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
        स्वचालित start0 = ov::op::v0::Constant::create(ov::element::i64, {1}, {0});
        स्वचालित stop0  = ov::op::v0::Constant::create(ov::element::i64, {1}, {nc});
        स्वचालित start1 = ov::op::v0::Constant::create(ov::element::i64, {1}, {nc});
        स्वचालित stop1  = ov::op::v0::Constant::create(ov::element::i64, {1}, {2 * nc});

        src0 = std::make_shared<ov::op::v8::खण्डिका>(combined, start0, stop0, चरणः, axis);
        src1 = std::make_shared<ov::op::v8::खण्डिका>(combined, start1, stop1, चरणः, axis);
    }

    int32_t * params = context.get_output_op_params();
    ध्रुव int32_t swapped = params[1];
    यदि (swapped) {
        std::swap(src0, src1);
    }

    // GELU_QUICK(x) = x * sigmoid(1.702 * x)
    // Create the constant in the same type as src0 to avoid f16/f32 mismatch.
    स्वचालित निवेशः_type = src0.get_element_type();
    स्वचालित coef = ov::op::v0::Constant::create(निवेशः_type, ov::Shape{}, {1.702f});
    स्वचालित gated   = std::make_shared<ov::op::v4::Swish>(src0, coef);
    स्वचालित res     = std::make_shared<ov::op::v1::Multiply>(gated, src1);

    प्रत्यागम rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
