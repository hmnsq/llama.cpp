#شامل "../node_context.h"
#شامل "../op_table.h"
#شامل "../utils.h"

#شامل <memory>
#شامل <openvino/core/node_output.hpp>
#شامل <openvino/op/constant.hpp>
#شامل <openvino/op/multiply.hpp>
#شامل <openvino/op/slice.hpp>
#شامل <openvino/op/swish.hpp>

نامخلا ov {
نامخلا frontend {
نامخلا ggml {
نامخلا op {

Output_سمتیہ translate_glu_geglu_تیز(ثابت گرہ_Context & context) {
    عدد_inputs_check(context, 1, 2);

    ov::Output<ov::Node> src0;
    ov::Output<ov::Node> src1;
    اگر (context.get_داخلہ_سائز() == 2) {
        src0 = process_view_داخلہ_new(context, 0);
        src1 = process_view_داخلہ_new(context, 1);
    } ورنہ {
        // split along last axis, nc = ne[0] / 2
        خودکار combined = process_view_داخلہ_new(context, 0);
        خودکار combined_shape = combined.get_partial_shape();
        int64_t آخری_dim_val = combined_shape[combined_shape.rank().get_لمبائی() - 1].get_لمبائی();
        int64_t nc = آخری_dim_val / 2;

        خودکار axis  = ov::op::v0::Constant::create(ov::element::i64, {1}, {-1});
        خودکار قدم  = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
        خودکار start0 = ov::op::v0::Constant::create(ov::element::i64, {1}, {0});
        خودکار stop0  = ov::op::v0::Constant::create(ov::element::i64, {1}, {nc});
        خودکار start1 = ov::op::v0::Constant::create(ov::element::i64, {1}, {nc});
        خودکار stop1  = ov::op::v0::Constant::create(ov::element::i64, {1}, {2 * nc});

        src0 = std::make_shared<ov::op::v8::قاش>(combined, start0, stop0, قدم, axis);
        src1 = std::make_shared<ov::op::v8::قاش>(combined, start1, stop1, قدم, axis);
    }

    int32_t * params = context.get_output_op_params();
    ثابت int32_t swapped = params[1];
    اگر (swapped) {
        std::swap(src0, src1);
    }

    // GELU_QUICK(x) = x * sigmoid(1.702 * x)
    // Create the constant in the same type as src0 to avoid f16/f32 mismatch.
    خودکار داخلہ_type = src0.get_element_type();
    خودکار coef = ov::op::v0::Constant::create(داخلہ_type, ov::Shape{}, {1.702f});
    خودکار gated   = std::make_shared<ov::op::v4::Swish>(src0, coef);
    خودکار res     = std::make_shared<ov::op::v1::Multiply>(gated, src1);

    واپس rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
