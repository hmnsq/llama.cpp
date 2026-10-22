#ಅನ್ತರ್ಭಾವ "../node_context.h"
#ಅನ್ತರ್ಭಾವ "../op_table.h"
#ಅನ್ತರ್ಭಾವ "../utils.h"

#ಅನ್ತರ್ಭಾವ <memory>
#ಅನ್ತರ್ಭಾವ <openvino/core/node_output.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/constant.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/multiply.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/slice.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/swish.hpp>

ನಾಮಾಕಾಶ ov {
ನಾಮಾಕಾಶ frontend {
ನಾಮಾಕಾಶ ggml {
ನಾಮಾಕಾಶ op {

Output_ಸದಿಶ translate_glu_geglu_ಶೀಘ್ರ(ಸ್ಥಿರಾಂಕ ಗಂಟು_Context & context) {
    ಅಂಕೆ_inputs_check(context, 1, 2);

    ov::Output<ov::Node> src0;
    ov::Output<ov::Node> src1;
    ಆದರೆ (context.get_ಒಳಹರಿವು_ಗಾತ್ರ() == 2) {
        src0 = process_view_ಒಳಹರಿವು_new(context, 0);
        src1 = process_view_ಒಳಹರಿವು_new(context, 1);
    } ಇಲ್ಲದಿದ್ದರೆ {
        // split along last axis, nc = ne[0] / 2
        ಸ್ವಯಂಚಾಲಿತ combined = process_view_ಒಳಹರಿವು_new(context, 0);
        ಸ್ವಯಂಚಾಲಿತ combined_shape = combined.get_partial_shape();
        int64_t ಕೊನೆಯ_dim_val = combined_shape[combined_shape.rank().get_ಉದ್ದಳತೆ() - 1].get_ಉದ್ದಳತೆ();
        int64_t nc = ಕೊನೆಯ_dim_val / 2;

        ಸ್ವಯಂಚಾಲಿತ axis  = ov::op::v0::Constant::create(ov::element::i64, {1}, {-1});
        ಸ್ವಯಂಚಾಲಿತ ಹೆಜ್ಜೆ  = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
        ಸ್ವಯಂಚಾಲಿತ start0 = ov::op::v0::Constant::create(ov::element::i64, {1}, {0});
        ಸ್ವಯಂಚಾಲಿತ stop0  = ov::op::v0::Constant::create(ov::element::i64, {1}, {nc});
        ಸ್ವಯಂಚಾಲಿತ start1 = ov::op::v0::Constant::create(ov::element::i64, {1}, {nc});
        ಸ್ವಯಂಚಾಲಿತ stop1  = ov::op::v0::Constant::create(ov::element::i64, {1}, {2 * nc});

        src0 = std::make_shared<ov::op::v8::ತುಂಡು>(combined, start0, stop0, ಹೆಜ್ಜೆ, axis);
        src1 = std::make_shared<ov::op::v8::ತುಂಡು>(combined, start1, stop1, ಹೆಜ್ಜೆ, axis);
    }

    int32_t * params = context.get_output_op_params();
    ಸ್ಥಿರಾಂಕ int32_t swapped = params[1];
    ಆದರೆ (swapped) {
        std::swap(src0, src1);
    }

    // GELU_QUICK(x) = x * sigmoid(1.702 * x)
    // Create the constant in the same type as src0 to avoid f16/f32 mismatch.
    ಸ್ವಯಂಚಾಲಿತ ಒಳಹರಿವು_type = src0.get_element_type();
    ಸ್ವಯಂಚಾಲಿತ coef = ov::op::v0::Constant::create(ಒಳಹರಿವು_type, ov::Shape{}, {1.702f});
    ಸ್ವಯಂಚಾಲಿತ gated   = std::make_shared<ov::op::v4::Swish>(src0, coef);
    ಸ್ವಯಂಚಾಲಿತ res     = std::make_shared<ov::op::v1::Multiply>(gated, src1);

    ಹಿಂತಿರುಗಿ rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
