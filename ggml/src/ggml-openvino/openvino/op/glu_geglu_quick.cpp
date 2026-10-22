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

Output_ویکټور translate_glu_geglu_چټک(ثابت غوټه_Context & context) {
    عدد_inputs_check(context, 1, 2);

    ov::Output<ov::Node> src0;
    ov::Output<ov::Node> src1;
    إذا (context.get_ننوت_اندازه() == 2) {
        src0 = process_view_ننوت_new(context, 0);
        src1 = process_view_ننوت_new(context, 1);
    } وإلا {
        // split along last axis, nc = ne[0] / 2
        خپلکاری combined = process_view_ننوت_new(context, 0);
        خپلکاری combined_shape = combined.get_partial_shape();
        int64_t وروستی_dim_val = combined_shape[combined_shape.rank().get_اوږدتیا() - 1].get_اوږدتیا();
        int64_t nc = وروستی_dim_val / 2;

        خپلکاری axis  = ov::op::v0::Constant::create(ov::element::i64, {1}, {-1});
        خپلکاری ګام  = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
        خپلکاری start0 = ov::op::v0::Constant::create(ov::element::i64, {1}, {0});
        خپلکاری stop0  = ov::op::v0::Constant::create(ov::element::i64, {1}, {nc});
        خپلکاری start1 = ov::op::v0::Constant::create(ov::element::i64, {1}, {nc});
        خپلکاری stop1  = ov::op::v0::Constant::create(ov::element::i64, {1}, {2 * nc});

        src0 = std::make_shared<ov::op::v8::ټوټه>(combined, start0, stop0, ګام, axis);
        src1 = std::make_shared<ov::op::v8::ټوټه>(combined, start1, stop1, ګام, axis);
    }

    int32_t * params = context.get_output_op_params();
    ثابت int32_t swapped = params[1];
    إذا (swapped) {
        std::swap(src0, src1);
    }

    // GELU_QUICK(x) = x * sigmoid(1.702 * x)
    // Create the constant in the same type as src0 to avoid f16/f32 mismatch.
    خپلکاری ننوت_type = src0.get_element_type();
    خپلکاری coef = ov::op::v0::Constant::create(ننوت_type, ov::Shape{}, {1.702f});
    خپلکاری gated   = std::make_shared<ov::op::v4::Swish>(src0, coef);
    خپلکاری res     = std::make_shared<ov::op::v1::Multiply>(gated, src1);

    أرجع rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
