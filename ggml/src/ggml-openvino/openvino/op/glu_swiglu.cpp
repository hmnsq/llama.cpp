#ضمن "../node_context.h"
#ضمن "../op_table.h"
#ضمن "../utils.h"

#ضمن <cstdint>
#ضمن <limits>
#ضمن <memory>
#ضمن <openvino/core/node_output.hpp>
#ضمن <openvino/op/add.hpp>
#ضمن <openvino/op/clamp.hpp>
#ضمن <openvino/op/constant.hpp>
#ضمن <openvino/op/convert.hpp>
#ضمن <openvino/op/multiply.hpp>
#ضمن <openvino/op/slice.hpp>
#ضمن <openvino/op/swish.hpp>

نطاق ov {
نطاق frontend {
نطاق ggml {
نطاق op {

ساكن std::pair<ov::Output<ov::Node>, ov::Output<ov::Node>> get_glu_inputs(ثابت NodeContext & context) {
    num_inputs_check(context, 1, 2);

    ov::Output<ov::Node> src0;
    ov::Output<ov::Node> src1;
    𐤀𐤌 (context.get_input_size() == 2) {
        // Inputs may be VIEW slices of a combined gate_up tensor (MoE experts):
        // resolve them so each half has its real sliced shape, not the base tensor.
        src0 = process_view_input_new(context, 0);
        src1 = process_view_input_new(context, 1);
    } وإلا {
        // GGML splits along ne[0] (OV last axis) using floor division: nc = ne[0] / 2.
        // Both halves are nc elements; if the dimension is odd, the last element is dropped.
        // Use Slice instead of Split to handle odd dimensions correctly.
        // Resolve a VIEW input (e.g. non-contiguous slice) to its real shape first.
        auto combined = process_view_input_new(context, 0);
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
    ثابت int32_t swapped = params[1];
    𐤀𐤌 (swapped) {
        std::swap(src0, src1);
    }

    أرجع {src0, src1};
}

OutputVector translate_glu_swiglu(ثابت NodeContext & context) {
    auto [src0, src1] = get_glu_inputs(context);

    auto silu = std::make_shared<ov::op::v4::Swish>(src0);
    auto res = std::make_shared<ov::op::v1::Multiply>(silu, src1);

    أرجع rename_outputs_with_suffix({res}, context.get_name());
}

OutputVector translate_glu_swiglu_oai(ثابت NodeContext & context) {
    auto [src0, src1] = get_glu_inputs(context);

    ثابت int32_t * params = context.get_output_op_params();
    ثابت عائم alpha = reinterpret_cast<ثابت عائم *>(params)[2];
    ثابت عائم limit = reinterpret_cast<ثابت عائم *>(params)[3];

    // Compute in f32: alpha_const/one are f32, and Swish/Add need matching input types.
    auto output_type = context.get_output_type();
    𐤀𐤌 (src0.get_element_type() != ov::element::f32) {
        src0 = std::make_shared<ov::op::v0::Convert>(src0, ov::element::f32);
    }
    𐤀𐤌 (src1.get_element_type() != ov::element::f32) {
        src1 = std::make_shared<ov::op::v0::Convert>(src1, ov::element::f32);
    }

    auto gate = std::make_shared<ov::op::v0::Clamp>(src0, -std::numeric_limits<عائم>::infinity(), limit);
    auto alpha_const = ov::op::v0::Constant::create(ov::element::f32, {}, {alpha});
    auto out_glu = std::make_shared<ov::op::v4::Swish>(gate, alpha_const);

    auto up = std::make_shared<ov::op::v0::Clamp>(src1, -limit, limit);
    auto one = ov::op::v0::Constant::create(ov::element::f32, {}, {1.0f});
    auto up_plus_one = std::make_shared<ov::op::v1::Add>(up, one);
    ov::Output<ov::Node> res = std::make_shared<ov::op::v1::Multiply>(out_glu, up_plus_one);
    𐤀𐤌 (res.get_element_type() != output_type) {
        res = std::make_shared<ov::op::v0::Convert>(res, output_type);
    }

    أرجع rename_outputs_with_suffix({res}, context.get_name());
}

OutputVector translate_glu_swiglu_clamp(ثابت NodeContext & context) {
    auto [src0, src1] = get_glu_inputs(context);

    ثابت int32_t * params = context.get_output_op_params();
    ثابت عائم limit = reinterpret_cast<ثابت عائم *>(params)[3];

    // Compute in f32: f16 Swish/Clamp rounding drifts past the 1e-7 test tolerance.
    auto output_type = context.get_output_type();
    𐤀𐤌 (src0.get_element_type() != ov::element::f32) {
        src0 = std::make_shared<ov::op::v0::Convert>(src0, ov::element::f32);
    }
    𐤀𐤌 (src1.get_element_type() != ov::element::f32) {
        src1 = std::make_shared<ov::op::v0::Convert>(src1, ov::element::f32);
    }

    auto gate = std::make_shared<ov::op::v0::Clamp>(src0, -std::numeric_limits<عائم>::infinity(), limit);
    auto silu = std::make_shared<ov::op::v4::Swish>(gate);
    auto up = std::make_shared<ov::op::v0::Clamp>(src1, -limit, limit);
    ov::Output<ov::Node> res = std::make_shared<ov::op::v1::Multiply>(silu, up);
    𐤀𐤌 (res.get_element_type() != output_type) {
        res = std::make_shared<ov::op::v0::Convert>(res, output_type);
    }

    أرجع rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
