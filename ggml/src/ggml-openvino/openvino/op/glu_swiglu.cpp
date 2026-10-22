#अन्तर्भाव "../node_context.h"
#अन्तर्भाव "../op_table.h"
#अन्तर्भाव "../utils.h"

#अन्तर्भाव <cstdint>
#अन्तर्भाव <limits>
#अन्तर्भाव <memory>
#अन्तर्भाव <openvino/core/node_output.hpp>
#अन्तर्भाव <openvino/op/add.hpp>
#अन्तर्भाव <openvino/op/clamp.hpp>
#अन्तर्भाव <openvino/op/constant.hpp>
#अन्तर्भाव <openvino/op/convert.hpp>
#अन्तर्भाव <openvino/op/multiply.hpp>
#अन्तर्भाव <openvino/op/slice.hpp>
#अन्तर्भाव <openvino/op/swish.hpp>

नामाकाश ov {
नामाकाश frontend {
नामाकाश ggml {
नामाकाश op {

स्थिर std::pair<ov::Output<ov::Node>, ov::Output<ov::Node>> get_glu_inputs(ध्रुव ग्रन्थिः_Context & context) {
    अङ्कः_inputs_check(context, 1, 2);

    ov::Output<ov::Node> src0;
    ov::Output<ov::Node> src1;
    सचे (context.get_निवेशः_size() == 2) {
        // Inputs may be VIEW slices of a combined gate_up tensor (MoE experts):
        // resolve them so each half has its real sliced shape, not the base tensor.
        src0 = process_view_निवेशः_new(context, 0);
        src1 = process_view_निवेशः_new(context, 1);
    } अञ्ञथा {
        // GGML splits along ne[0] (OV last axis) using floor division: nc = ne[0] / 2.
        // Both halves are nc elements; if the dimension is odd, the last element is dropped.
        // Use Slice instead of Split to handle odd dimensions correctly.
        // Resolve a VIEW input (e.g. non-contiguous slice) to its real shape first.
        auto combined = process_view_निवेशः_new(context, 0);
        auto combined_shape = combined.get_partial_shape();
        int64_t last_dim_val = combined_shape[combined_shape.rank().get_length() - 1].get_length();
        int64_t nc = last_dim_val / 2;

        auto axis = ov::op::v0::Constant::create(ov::element::i64, {1}, {-1});
        auto step = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
        auto start0 = ov::op::v0::Constant::create(ov::element::i64, {1}, {0});
        auto stop0 = ov::op::v0::Constant::create(ov::element::i64, {1}, {nc});
        auto start1 = ov::op::v0::Constant::create(ov::element::i64, {1}, {nc});
        auto stop1 = ov::op::v0::Constant::create(ov::element::i64, {1}, {2 * nc});

        src0 = std::make_shared<ov::op::v8::Slice>(combined, start0, stop0, step, axis);
        src1 = std::make_shared<ov::op::v8::Slice>(combined, start1, stop1, step, axis);
    }

    int32_t * params = context.get_output_op_params();
    ध्रुव int32_t swapped = params[1];
    सचे (swapped) {
        std::swap(src0, src1);
    }

    पटिनिवत्त {src0, src1};
}

OutputVector translate_glu_swiglu(ध्रुव ग्रन्थिः_Context & context) {
    auto [src0, src1] = get_glu_inputs(context);

    auto silu = std::make_shared<ov::op::v4::Swish>(src0);
    auto res = std::make_shared<ov::op::v1::Multiply>(silu, src1);

    पटिनिवत्त rename_outputs_with_suffix({res}, context.get_name());
}

OutputVector translate_glu_swiglu_oai(ध्रुव ग्रन्थिः_Context & context) {
    auto [src0, src1] = get_glu_inputs(context);

    ध्रुव int32_t * params = context.get_output_op_params();
    ध्रुव भिन्न alpha = reinterpret_cast<ध्रुव भिन्न *>(params)[2];
    ध्रुव भिन्न limit = reinterpret_cast<ध्रुव भिन्न *>(params)[3];

    // Compute in f32: alpha_const/one are f32, and Swish/Add need matching input types.
    auto output_type = context.get_output_type();
    सचे (src0.get_element_type() != ov::element::f32) {
        src0 = std::make_shared<ov::op::v0::Convert>(src0, ov::element::f32);
    }
    सचे (src1.get_element_type() != ov::element::f32) {
        src1 = std::make_shared<ov::op::v0::Convert>(src1, ov::element::f32);
    }

    auto gate = std::make_shared<ov::op::v0::Clamp>(src0, -std::numeric_limits<भिन्न>::infinity(), limit);
    auto alpha_const = ov::op::v0::Constant::create(ov::element::f32, {}, {alpha});
    auto out_glu = std::make_shared<ov::op::v4::Swish>(gate, alpha_const);

    auto up = std::make_shared<ov::op::v0::Clamp>(src1, -limit, limit);
    auto one = ov::op::v0::Constant::create(ov::element::f32, {}, {1.0f});
    auto up_plus_one = std::make_shared<ov::op::v1::Add>(up, one);
    ov::Output<ov::Node> res = std::make_shared<ov::op::v1::Multiply>(out_glu, up_plus_one);
    सचे (res.get_element_type() != output_type) {
        res = std::make_shared<ov::op::v0::Convert>(res, output_type);
    }

    पटिनिवत्त rename_outputs_with_suffix({res}, context.get_name());
}

OutputVector translate_glu_swiglu_clamp(ध्रुव ग्रन्थिः_Context & context) {
    auto [src0, src1] = get_glu_inputs(context);

    ध्रुव int32_t * params = context.get_output_op_params();
    ध्रुव भिन्न limit = reinterpret_cast<ध्रुव भिन्न *>(params)[3];

    // Compute in f32: f16 Swish/Clamp rounding drifts past the 1e-7 test tolerance.
    auto output_type = context.get_output_type();
    सचे (src0.get_element_type() != ov::element::f32) {
        src0 = std::make_shared<ov::op::v0::Convert>(src0, ov::element::f32);
    }
    सचे (src1.get_element_type() != ov::element::f32) {
        src1 = std::make_shared<ov::op::v0::Convert>(src1, ov::element::f32);
    }

    auto gate = std::make_shared<ov::op::v0::Clamp>(src0, -std::numeric_limits<भिन्न>::infinity(), limit);
    auto silu = std::make_shared<ov::op::v4::Swish>(gate);
    auto up = std::make_shared<ov::op::v0::Clamp>(src1, -limit, limit);
    ov::Output<ov::Node> res = std::make_shared<ov::op::v1::Multiply>(silu, up);
    सचे (res.get_element_type() != output_type) {
        res = std::make_shared<ov::op::v0::Convert>(res, output_type);
    }

    पटिनिवत्त rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
