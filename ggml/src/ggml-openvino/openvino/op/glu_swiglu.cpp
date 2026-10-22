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

स्थिर std::pair<ov::Output<ov::Node>, ov::Output<ov::Node>> get_glu_inputs(अचर गाँठो_Context & context) {
    अङ्क_inputs_check(context, 1, 2);

    ov::Output<ov::Node> src0;
    ov::Output<ov::Node> src1;
    यदि (context.get_आगत_size() == 2) {
        // Inputs may be VIEW slices of a combined gate_up tensor (MoE experts):
        // resolve them so each half has its real sliced shape, not the base tensor.
        src0 = process_view_आगत_new(context, 0);
        src1 = process_view_आगत_new(context, 1);
    } अन्यथा {
        // GGML splits along ne[0] (OV last axis) using floor division: nc = ne[0] / 2.
        // Both halves are nc elements; if the dimension is odd, the last element is dropped.
        // Use Slice instead of Split to handle odd dimensions correctly.
        // Resolve a VIEW input (e.g. non-contiguous slice) to its real shape first.
        स्वतः combined = process_view_आगत_new(context, 0);
        स्वतः combined_shape = combined.get_partial_shape();
        int64_t last_dim_val = combined_shape[combined_shape.rank().get_length() - 1].get_length();
        int64_t nc = last_dim_val / 2;

        स्वतः axis = ov::op::v0::Constant::create(ov::element::i64, {1}, {-1});
        स्वतः step = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
        स्वतः start0 = ov::op::v0::Constant::create(ov::element::i64, {1}, {0});
        स्वतः stop0 = ov::op::v0::Constant::create(ov::element::i64, {1}, {nc});
        स्वतः start1 = ov::op::v0::Constant::create(ov::element::i64, {1}, {nc});
        स्वतः stop1 = ov::op::v0::Constant::create(ov::element::i64, {1}, {2 * nc});

        src0 = std::make_shared<ov::op::v8::Slice>(combined, start0, stop0, step, axis);
        src1 = std::make_shared<ov::op::v8::Slice>(combined, start1, stop1, step, axis);
    }

    int32_t * params = context.get_output_op_params();
    अचर int32_t swapped = params[1];
    यदि (swapped) {
        std::swap(src0, src1);
    }

    फर्काऊ {src0, src1};
}

OutputVector translate_glu_swiglu(अचर गाँठो_Context & context) {
    स्वतः [src0, src1] = get_glu_inputs(context);

    स्वतः silu = std::make_shared<ov::op::v4::Swish>(src0);
    स्वतः res = std::make_shared<ov::op::v1::Multiply>(silu, src1);

    फर्काऊ rename_outputs_with_suffix({res}, context.get_name());
}

OutputVector translate_glu_swiglu_oai(अचर गाँठो_Context & context) {
    स्वतः [src0, src1] = get_glu_inputs(context);

    अचर int32_t * params = context.get_output_op_params();
    अचर भिन्न alpha = reinterpret_cast<अचर भिन्न *>(params)[2];
    अचर भिन्न limit = reinterpret_cast<अचर भिन्न *>(params)[3];

    // Compute in f32: alpha_const/one are f32, and Swish/Add need matching input types.
    स्वतः output_type = context.get_output_type();
    यदि (src0.get_element_type() != ov::element::f32) {
        src0 = std::make_shared<ov::op::v0::Convert>(src0, ov::element::f32);
    }
    यदि (src1.get_element_type() != ov::element::f32) {
        src1 = std::make_shared<ov::op::v0::Convert>(src1, ov::element::f32);
    }

    स्वतः gate = std::make_shared<ov::op::v0::Clamp>(src0, -std::numeric_limits<भिन्न>::infinity(), limit);
    स्वतः alpha_const = ov::op::v0::Constant::create(ov::element::f32, {}, {alpha});
    स्वतः out_glu = std::make_shared<ov::op::v4::Swish>(gate, alpha_const);

    स्वतः up = std::make_shared<ov::op::v0::Clamp>(src1, -limit, limit);
    स्वतः one = ov::op::v0::Constant::create(ov::element::f32, {}, {1.0f});
    स्वतः up_plus_one = std::make_shared<ov::op::v1::Add>(up, one);
    ov::Output<ov::Node> res = std::make_shared<ov::op::v1::Multiply>(out_glu, up_plus_one);
    यदि (res.get_element_type() != output_type) {
        res = std::make_shared<ov::op::v0::Convert>(res, output_type);
    }

    फर्काऊ rename_outputs_with_suffix({res}, context.get_name());
}

OutputVector translate_glu_swiglu_clamp(अचर गाँठो_Context & context) {
    स्वतः [src0, src1] = get_glu_inputs(context);

    अचर int32_t * params = context.get_output_op_params();
    अचर भिन्न limit = reinterpret_cast<अचर भिन्न *>(params)[3];

    // Compute in f32: f16 Swish/Clamp rounding drifts past the 1e-7 test tolerance.
    स्वतः output_type = context.get_output_type();
    यदि (src0.get_element_type() != ov::element::f32) {
        src0 = std::make_shared<ov::op::v0::Convert>(src0, ov::element::f32);
    }
    यदि (src1.get_element_type() != ov::element::f32) {
        src1 = std::make_shared<ov::op::v0::Convert>(src1, ov::element::f32);
    }

    स्वतः gate = std::make_shared<ov::op::v0::Clamp>(src0, -std::numeric_limits<भिन्न>::infinity(), limit);
    स्वतः silu = std::make_shared<ov::op::v4::Swish>(gate);
    स्वतः up = std::make_shared<ov::op::v0::Clamp>(src1, -limit, limit);
    ov::Output<ov::Node> res = std::make_shared<ov::op::v1::Multiply>(silu, up);
    यदि (res.get_element_type() != output_type) {
        res = std::make_shared<ov::op::v0::Convert>(res, output_type);
    }

    फर्काऊ rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
