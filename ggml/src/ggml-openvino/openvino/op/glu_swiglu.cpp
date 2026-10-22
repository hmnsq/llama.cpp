#समावेश "../node_context.h"
#समावेश "../op_table.h"
#समावेश "../utils.h"

#समावेश <cstdint>
#समावेश <limits>
#समावेश <memory>
#समावेश <openvino/core/node_output.hpp>
#समावेश <openvino/op/add.hpp>
#समावेश <openvino/op/clamp.hpp>
#समावेश <openvino/op/constant.hpp>
#समावेश <openvino/op/convert.hpp>
#समावेश <openvino/op/multiply.hpp>
#समावेश <openvino/op/slice.hpp>
#समावेश <openvino/op/swish.hpp>

namespace ov {
namespace frontend {
namespace ggml {
namespace op {

अटल std::pair<ov::Output<ov::Node>, ov::Output<ov::Node>> get_glu_inputs(स्थिर गाँठ_Context & context) {
    अंक_inputs_check(context, 1, 2);

    ov::Output<ov::Node> src0;
    ov::Output<ov::Node> src1;
    अगर (context.get_निवेश_आकार() == 2) {
        // Inputs may be VIEW slices of a combined gate_up tensor (MoE experts):
        // resolve them so each half has its real sliced shape, not the base tensor.
        src0 = process_view_निवेश_new(context, 0);
        src1 = process_view_निवेश_new(context, 1);
    } अन्यथा {
        // GGML splits along ne[0] (OV last axis) using floor division: nc = ne[0] / 2.
        // Both halves are nc elements; if the dimension is odd, the last element is dropped.
        // Use Slice instead of Split to handle odd dimensions correctly.
        // Resolve a VIEW input (e.g. non-contiguous slice) to its real shape first.
        स्वतः combined = process_view_निवेश_new(context, 0);
        स्वतः combined_shape = combined.get_partial_shape();
        int64_t last_dim_val = combined_shape[combined_shape.rank().get_लंबाई() - 1].get_लंबाई();
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
    स्थिर int32_t swapped = params[1];
    अगर (swapped) {
        std::swap(src0, src1);
    }

    वापस {src0, src1};
}

OutputVector translate_glu_swiglu(स्थिर गाँठ_Context & context) {
    स्वतः [src0, src1] = get_glu_inputs(context);

    स्वतः silu = std::make_shared<ov::op::v4::Swish>(src0);
    स्वतः res = std::make_shared<ov::op::v1::Multiply>(silu, src1);

    वापस rename_outputs_with_suffix({res}, context.get_name());
}

OutputVector translate_glu_swiglu_oai(स्थिर गाँठ_Context & context) {
    स्वतः [src0, src1] = get_glu_inputs(context);

    स्थिर int32_t * params = context.get_output_op_params();
    स्थिर भग्न alpha = reinterpret_cast<स्थिर भग्न *>(params)[2];
    स्थिर भग्न limit = reinterpret_cast<स्थिर भग्न *>(params)[3];

    // Compute in f32: alpha_const/one are f32, and Swish/Add need matching input types.
    स्वतः output_type = context.get_output_type();
    अगर (src0.get_element_type() != ov::element::f32) {
        src0 = std::make_shared<ov::op::v0::Convert>(src0, ov::element::f32);
    }
    अगर (src1.get_element_type() != ov::element::f32) {
        src1 = std::make_shared<ov::op::v0::Convert>(src1, ov::element::f32);
    }

    स्वतः gate = std::make_shared<ov::op::v0::Clamp>(src0, -std::numeric_limits<भग्न>::infinity(), limit);
    स्वतः alpha_const = ov::op::v0::Constant::create(ov::element::f32, {}, {alpha});
    स्वतः out_glu = std::make_shared<ov::op::v4::Swish>(gate, alpha_const);

    स्वतः up = std::make_shared<ov::op::v0::Clamp>(src1, -limit, limit);
    स्वतः one = ov::op::v0::Constant::create(ov::element::f32, {}, {1.0f});
    स्वतः up_plus_one = std::make_shared<ov::op::v1::Add>(up, one);
    ov::Output<ov::Node> res = std::make_shared<ov::op::v1::Multiply>(out_glu, up_plus_one);
    अगर (res.get_element_type() != output_type) {
        res = std::make_shared<ov::op::v0::Convert>(res, output_type);
    }

    वापस rename_outputs_with_suffix({res}, context.get_name());
}

OutputVector translate_glu_swiglu_clamp(स्थिर गाँठ_Context & context) {
    स्वतः [src0, src1] = get_glu_inputs(context);

    स्थिर int32_t * params = context.get_output_op_params();
    स्थिर भग्न limit = reinterpret_cast<स्थिर भग्न *>(params)[3];

    // Compute in f32: f16 Swish/Clamp rounding drifts past the 1e-7 test tolerance.
    स्वतः output_type = context.get_output_type();
    अगर (src0.get_element_type() != ov::element::f32) {
        src0 = std::make_shared<ov::op::v0::Convert>(src0, ov::element::f32);
    }
    अगर (src1.get_element_type() != ov::element::f32) {
        src1 = std::make_shared<ov::op::v0::Convert>(src1, ov::element::f32);
    }

    स्वतः gate = std::make_shared<ov::op::v0::Clamp>(src0, -std::numeric_limits<भग्न>::infinity(), limit);
    स्वतः silu = std::make_shared<ov::op::v4::Swish>(gate);
    स्वतः up = std::make_shared<ov::op::v0::Clamp>(src1, -limit, limit);
    ov::Output<ov::Node> res = std::make_shared<ov::op::v1::Multiply>(silu, up);
    अगर (res.get_element_type() != output_type) {
        res = std::make_shared<ov::op::v0::Convert>(res, output_type);
    }

    वापस rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
