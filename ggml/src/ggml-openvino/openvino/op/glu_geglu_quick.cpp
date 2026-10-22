#ਅਨ੍ਤਰ੍ਭਾਵ "../node_context.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "../op_table.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "../utils.h"

#ਅਨ੍ਤਰ੍ਭਾਵ <memory>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/core/node_output.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/constant.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/multiply.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/slice.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/swish.hpp>

ਨਾਮਾਕਾਸ਼ ov {
ਨਾਮਾਕਾਸ਼ frontend {
ਨਾਮਾਕਾਸ਼ ggml {
ਨਾਮਾਕਾਸ਼ op {

Output_ਸਦਿਸ਼ translate_glu_geglu_ਤੇਜ਼(ਅਚਲ ਗੰਢ_Context & context) {
    ਅੰਕ_inputs_check(context, 1, 2);

    ov::Output<ov::Node> src0;
    ov::Output<ov::Node> src1;
    ਜੇ (context.get_ਨਿਵੇਸ਼_ਮਾਪ() == 2) {
        src0 = process_view_ਨਿਵੇਸ਼_new(context, 0);
        src1 = process_view_ਨਿਵੇਸ਼_new(context, 1);
    } ਨਹੀਂਤਾਂ {
        // split along last axis, nc = ne[0] / 2
        ਸਵੈਚਾਲਿਤ combined = process_view_ਨਿਵੇਸ਼_new(context, 0);
        ਸਵੈਚਾਲਿਤ combined_shape = combined.get_partial_shape();
        int64_t ਆਖਰੀ_dim_val = combined_shape[combined_shape.rank().get_ਲੰਮਾਈ() - 1].get_ਲੰਮਾਈ();
        int64_t nc = ਆਖਰੀ_dim_val / 2;

        ਸਵੈਚਾਲਿਤ axis  = ov::op::v0::Constant::create(ov::element::i64, {1}, {-1});
        ਸਵੈਚਾਲਿਤ ਕਦਮ  = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
        ਸਵੈਚਾਲਿਤ start0 = ov::op::v0::Constant::create(ov::element::i64, {1}, {0});
        ਸਵੈਚਾਲਿਤ stop0  = ov::op::v0::Constant::create(ov::element::i64, {1}, {nc});
        ਸਵੈਚਾਲਿਤ start1 = ov::op::v0::Constant::create(ov::element::i64, {1}, {nc});
        ਸਵੈਚਾਲਿਤ stop1  = ov::op::v0::Constant::create(ov::element::i64, {1}, {2 * nc});

        src0 = std::make_shared<ov::op::v8::ਟੁਕੜਾ>(combined, start0, stop0, ਕਦਮ, axis);
        src1 = std::make_shared<ov::op::v8::ਟੁਕੜਾ>(combined, start1, stop1, ਕਦਮ, axis);
    }

    int32_t * params = context.get_output_op_params();
    ਅਚਲ int32_t swapped = params[1];
    ਜੇ (swapped) {
        std::swap(src0, src1);
    }

    // GELU_QUICK(x) = x * sigmoid(1.702 * x)
    // Create the constant in the same type as src0 to avoid f16/f32 mismatch.
    ਸਵੈਚਾਲਿਤ ਨਿਵੇਸ਼_type = src0.get_element_type();
    ਸਵੈਚਾਲਿਤ coef = ov::op::v0::Constant::create(ਨਿਵੇਸ਼_type, ov::Shape{}, {1.702f});
    ਸਵੈਚਾਲਿਤ gated   = std::make_shared<ov::op::v4::Swish>(src0, coef);
    ਸਵੈਚਾਲਿਤ res     = std::make_shared<ov::op::v1::Multiply>(gated, src1);

    ਵਾਪਸ rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
