#समावेश "../node_context.h"
#समावेश "../op_table.h"
#समावेश "../utils.h"

#समावेश <memory>
#समावेश <openvino/core/node_output.hpp>
#समावेश <openvino/op/clamp.hpp>
#समावेश <openvino/op/constant.hpp>
#समावेश <openvino/op/gelu.hpp>
#समावेश <openvino/op/multiply.hpp>
#समावेश <openvino/op/sigmoid.hpp>
#समावेश <openvino/op/slice.hpp>

namespace ov {
namespace frontend {
namespace ggml {
namespace op {

OutputVector translate_glu_geglu(स्थिर गाँठ_Context & context) {
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

    अगर (context.is_static()) {
        // TODO: Temporary solution for NPU accuracy issue due to fp16 overflow
        // To be removed once permanent solution is implemented
        // Justification:
        // For |x| > 5, GELU(x) ≈ max(x, 0)  (behaves like ReLU)
        // So Clamp(-10, 10) only affects values where GELU would return ≈ x anyway.
        // The only loss: values > 10 get mapped to 10 instead of x.
        // In practice, FFN intermediates rarely exceed 10 after GEGLU gating.
        src0 = std::make_shared<ov::op::v0::Clamp>(src0, -10.0, 10.0);
    }
    स्वतः gelu = std::make_shared<ov::op::v7::Gelu>(src0);
    स्वतः res = std::make_shared<ov::op::v1::Multiply>(gelu, src1);

    वापस rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
