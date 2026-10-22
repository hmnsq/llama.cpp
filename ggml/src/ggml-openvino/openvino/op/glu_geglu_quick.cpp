#ضمن "../node_context.h"
#ضمن "../op_table.h"
#ضمن "../utils.h"

#ضمن <memory>
#ضمن <openvino/core/node_output.hpp>
#ضمن <openvino/op/constant.hpp>
#ضمن <openvino/op/multiply.hpp>
#ضمن <openvino/op/slice.hpp>
#ضمن <openvino/op/swish.hpp>

نطاق ov {
نطاق frontend {
نطاق ggml {
نطاق op {

Output_ويڪٽر translate_glu_geglu_تيز(ثابت ڳنڍ_Context & context) {
    انگ_inputs_check(context, 1, 2);

    ov::Output<ov::Node> src0;
    ov::Output<ov::Node> src1;
    إذا (context.get_داخلا_سائيز() == 2) {
        src0 = process_view_داخلا_new(context, 0);
        src1 = process_view_داخلا_new(context, 1);
    } وإلا {
        // split along last axis, nc = ne[0] / 2
        خودڪار combined = process_view_داخلا_new(context, 0);
        خودڪار combined_shape = combined.get_partial_shape();
        int64_t آخري_dim_val = combined_shape[combined_shape.rank().get_ڊيگهائي() - 1].get_ڊيگهائي();
        int64_t nc = آخري_dim_val / 2;

        خودڪار axis  = ov::op::v0::Constant::create(ov::element::i64, {1}, {-1});
        خودڪار قدم  = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
        خودڪار start0 = ov::op::v0::Constant::create(ov::element::i64, {1}, {0});
        خودڪار stop0  = ov::op::v0::Constant::create(ov::element::i64, {1}, {nc});
        خودڪار start1 = ov::op::v0::Constant::create(ov::element::i64, {1}, {nc});
        خودڪار stop1  = ov::op::v0::Constant::create(ov::element::i64, {1}, {2 * nc});

        src0 = std::make_shared<ov::op::v8::ٽڪرو>(combined, start0, stop0, قدم, axis);
        src1 = std::make_shared<ov::op::v8::ٽڪرو>(combined, start1, stop1, قدم, axis);
    }

    int32_t * params = context.get_output_op_params();
    ثابت int32_t swapped = params[1];
    إذا (swapped) {
        std::swap(src0, src1);
    }

    // GELU_QUICK(x) = x * sigmoid(1.702 * x)
    // Create the constant in the same type as src0 to avoid f16/f32 mismatch.
    خودڪار داخلا_type = src0.get_element_type();
    خودڪار coef = ov::op::v0::Constant::create(داخلا_type, ov::Shape{}, {1.702f});
    خودڪار gated   = std::make_shared<ov::op::v4::Swish>(src0, coef);
    خودڪار res     = std::make_shared<ov::op::v1::Multiply>(gated, src1);

    أرجع rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
