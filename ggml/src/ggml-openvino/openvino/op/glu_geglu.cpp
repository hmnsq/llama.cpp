#ಅನ್ತರ್ಭಾವ "../node_context.h"
#ಅನ್ತರ್ಭಾವ "../op_table.h"
#ಅನ್ತರ್ಭಾವ "../utils.h"

#ಅನ್ತರ್ಭಾವ <memory>
#ಅನ್ತರ್ಭಾವ <openvino/core/node_output.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/clamp.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/constant.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/gelu.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/multiply.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/sigmoid.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/slice.hpp>

ನಾಮಾಕಾಶ ov {
ನಾಮಾಕಾಶ frontend {
ನಾಮಾಕಾಶ ggml {
ನಾಮಾಕಾಶ op {

Output_ಸದಿಶ translate_glu_geglu(ಸ್ಥಿರಾಂಕ ಗಂಟು_Context & context) {
    ಅಂಕೆ_inputs_check(context, 1, 2);

    ov::Output<ov::Node> src0;
    ov::Output<ov::Node> src1;
    ಆದರೆ (context.get_ಒಳಹರಿವು_ಗಾತ್ರ() == 2) {
        // Inputs may be VIEW slices of a combined gate_up tensor (MoE experts):
        // resolve them so each half has its real sliced shape, not the base tensor.
        src0 = process_view_ಒಳಹರಿವು_new(context, 0);
        src1 = process_view_ಒಳಹರಿವು_new(context, 1);
    } ಇಲ್ಲದಿದ್ದರೆ {
        // GGML splits along ne[0] (OV last axis) using floor division: nc = ne[0] / 2.
        // Both halves are nc elements; if the dimension is odd, the last element is dropped.
        // Use Slice instead of Split to handle odd dimensions correctly.
        // Resolve a VIEW input (e.g. non-contiguous slice) to its real shape first.
        ಸ್ವಯಂಚಾಲಿತ combined = process_view_ಒಳಹರಿವು_new(context, 0);
        ಸ್ವಯಂಚಾಲಿತ combined_shape = combined.get_partial_shape();
        int64_t ಕೊನೆಯ_dim_val = combined_shape[combined_shape.rank().get_ಉದ್ದಳತೆ() - 1].get_ಉದ್ದಳತೆ();
        int64_t nc = ಕೊನೆಯ_dim_val / 2;

        ಸ್ವಯಂಚಾಲಿತ axis = ov::op::v0::Constant::create(ov::element::i64, {1}, {-1});
        ಸ್ವಯಂಚಾಲಿತ ಹೆಜ್ಜೆ = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
        ಸ್ವಯಂಚಾಲಿತ start0 = ov::op::v0::Constant::create(ov::element::i64, {1}, {0});
        ಸ್ವಯಂಚಾಲಿತ stop0 = ov::op::v0::Constant::create(ov::element::i64, {1}, {nc});
        ಸ್ವಯಂಚಾಲಿತ start1 = ov::op::v0::Constant::create(ov::element::i64, {1}, {nc});
        ಸ್ವಯಂಚಾಲಿತ stop1 = ov::op::v0::Constant::create(ov::element::i64, {1}, {2 * nc});

        src0 = std::make_shared<ov::op::v8::ತುಂಡು>(combined, start0, stop0, ಹೆಜ್ಜೆ, axis);
        src1 = std::make_shared<ov::op::v8::ತುಂಡು>(combined, start1, stop1, ಹೆಜ್ಜೆ, axis);
    }

    int32_t * params = context.get_output_op_params();
    ಸ್ಥಿರಾಂಕ int32_t swapped = params[1];
    ಆದರೆ (swapped) {
        std::swap(src0, src1);
    }

    ಆದರೆ (context.is_static()) {
        // TODO: Temporary solution for NPU accuracy issue due to fp16 overflow
        // To be removed once permanent solution is implemented
        // Justification:
        // For |x| > 5, GELU(x) ≈ max(x, 0)  (behaves like ReLU)
        // So Clamp(-10, 10) only affects values where GELU would return ≈ x anyway.
        // The only loss: values > 10 get mapped to 10 instead of x.
        // In practice, FFN intermediates rarely exceed 10 after GEGLU gating.
        src0 = std::make_shared<ov::op::v0::Clamp>(src0, -10.0, 10.0);
    }
    ಸ್ವಯಂಚಾಲಿತ gelu = std::make_shared<ov::op::v7::Gelu>(src0);
    ಸ್ವಯಂಚಾಲಿತ res = std::make_shared<ov::op::v1::Multiply>(gelu, src1);

    ಹಿಂತಿರುಗಿ rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
