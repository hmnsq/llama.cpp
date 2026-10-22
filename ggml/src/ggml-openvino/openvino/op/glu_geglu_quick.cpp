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

OutputVector translate_glu_geglu_quick(ध्रुव गाँठ_Context & context) {
    अंक_inputs_check(context, 1, 2);

    ov::Output<ov::Node> src0;
    ov::Output<ov::Node> src1;
    यदि (context.get_निवेश_size() == 2) {
        src0 = process_view_निवेश_new(context, 0);
        src1 = process_view_निवेश_new(context, 1);
    } अन्यथा {
        // split along last axis, nc = ne[0] / 2
        स्वतः combined = process_view_निवेश_new(context, 0);
        स्वतः combined_shape = combined.get_partial_shape();
        int64_t last_dim_val = combined_shape[combined_shape.rank().get_length() - 1].get_length();
        int64_t nc = last_dim_val / 2;

        स्वतः axis  = ov::op::v0::Constant::create(ov::element::i64, {1}, {-1});
        स्वतः step  = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
        स्वतः start0 = ov::op::v0::Constant::create(ov::element::i64, {1}, {0});
        स्वतः stop0  = ov::op::v0::Constant::create(ov::element::i64, {1}, {nc});
        स्वतः start1 = ov::op::v0::Constant::create(ov::element::i64, {1}, {nc});
        स्वतः stop1  = ov::op::v0::Constant::create(ov::element::i64, {1}, {2 * nc});

        src0 = std::make_shared<ov::op::v8::Slice>(combined, start0, stop0, step, axis);
        src1 = std::make_shared<ov::op::v8::Slice>(combined, start1, stop1, step, axis);
    }

    int32_t * params = context.get_output_op_params();
    ध्रुव int32_t swapped = params[1];
    यदि (swapped) {
        std::swap(src0, src1);
    }

    // GELU_QUICK(x) = x * sigmoid(1.702 * x)
    // Create the constant in the same type as src0 to avoid f16/f32 mismatch.
    स्वतः निवेश_type = src0.get_element_type();
    स्वतः coef = ov::op::v0::Constant::create(निवेश_type, ov::Shape{}, {1.702f});
    स्वतः gated   = std::make_shared<ov::op::v4::Swish>(src0, coef);
    स्वतः res     = std::make_shared<ov::op::v1::Multiply>(gated, src1);

    प्रत्यागम rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
