#شامل "../node_context.h"
#شامل "../op_table.h"
#شامل "../utils.h"

#شامل <cstdint>
#شامل <limits>
#شامل <memory>
#شامل <openvino/core/node_output.hpp>
#شامل <openvino/op/add.hpp>
#شامل <openvino/op/clamp.hpp>
#شامل <openvino/op/constant.hpp>
#شامل <openvino/op/convert.hpp>
#شامل <openvino/op/multiply.hpp>
#شامل <openvino/op/slice.hpp>
#شامل <openvino/op/swish.hpp>

نامخلا ov {
نامخلا frontend {
نامخلا ggml {
نامخلا op {

ساکن std::pair<ov::Output<ov::Node>, ov::Output<ov::Node>> get_glu_inputs(ثابت گرہ_Context & context) {
    عدد_inputs_check(context, 1, 2);

    ov::Output<ov::Node> src0;
    ov::Output<ov::Node> src1;
    اگر (context.get_داخلہ_سائز() == 2) {
        // Inputs may be VIEW slices of a combined gate_up tensor (MoE experts):
        // resolve them so each half has its real sliced shape, not the base tensor.
        src0 = process_view_داخلہ_new(context, 0);
        src1 = process_view_داخلہ_new(context, 1);
    } ورنہ {
        // GGML splits along ne[0] (OV last axis) using floor division: nc = ne[0] / 2.
        // Both halves are nc elements; if the dimension is odd, the last element is dropped.
        // Use Slice instead of Split to handle odd dimensions correctly.
        // Resolve a VIEW input (e.g. non-contiguous slice) to its real shape first.
        خودکار combined = process_view_داخلہ_new(context, 0);
        خودکار combined_shape = combined.get_partial_shape();
        int64_t آخری_dim_val = combined_shape[combined_shape.rank().get_لمبائی() - 1].get_لمبائی();
        int64_t nc = آخری_dim_val / 2;

        خودکار axis = ov::op::v0::Constant::create(ov::element::i64, {1}, {-1});
        خودکار قدم = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
        خودکار start0 = ov::op::v0::Constant::create(ov::element::i64, {1}, {0});
        خودکار stop0 = ov::op::v0::Constant::create(ov::element::i64, {1}, {nc});
        خودکار start1 = ov::op::v0::Constant::create(ov::element::i64, {1}, {nc});
        خودکار stop1 = ov::op::v0::Constant::create(ov::element::i64, {1}, {2 * nc});

        src0 = std::make_shared<ov::op::v8::قاش>(combined, start0, stop0, قدم, axis);
        src1 = std::make_shared<ov::op::v8::قاش>(combined, start1, stop1, قدم, axis);
    }

    int32_t * params = context.get_output_op_params();
    ثابت int32_t swapped = params[1];
    اگر (swapped) {
        std::swap(src0, src1);
    }

    واپس {src0, src1};
}

Output_سمتیہ translate_glu_swiglu(ثابت گرہ_Context & context) {
    خودکار [src0, src1] = get_glu_inputs(context);

    خودکار silu = std::make_shared<ov::op::v4::Swish>(src0);
    خودکار res = std::make_shared<ov::op::v1::Multiply>(silu, src1);

    واپس rename_outputs_with_suffix({res}, context.get_name());
}

Output_سمتیہ translate_glu_swiglu_oai(ثابت گرہ_Context & context) {
    خودکار [src0, src1] = get_glu_inputs(context);

    ثابت int32_t * params = context.get_output_op_params();
    ثابت اعشاری alpha = reinterpret_cast<ثابت اعشاری *>(params)[2];
    ثابت اعشاری حد_بندی = reinterpret_cast<ثابت اعشاری *>(params)[3];

    // Compute in f32: alpha_const/one are f32, and Swish/Add need matching input types.
    خودکار output_type = context.get_output_type();
    اگر (src0.get_element_type() != ov::element::f32) {
        src0 = std::make_shared<ov::op::v0::Convert>(src0, ov::element::f32);
    }
    اگر (src1.get_element_type() != ov::element::f32) {
        src1 = std::make_shared<ov::op::v0::Convert>(src1, ov::element::f32);
    }

    خودکار gate = std::make_shared<ov::op::v0::Clamp>(src0, -std::numeric_limits<اعشاری>::infinity(), حد_بندی);
    خودکار alpha_const = ov::op::v0::Constant::create(ov::element::f32, {}, {alpha});
    خودکار برآمد_glu = std::make_shared<ov::op::v4::Swish>(gate, alpha_const);

    خودکار اوپر = std::make_shared<ov::op::v0::Clamp>(src1, -حد_بندی, حد_بندی);
    خودکار one = ov::op::v0::Constant::create(ov::element::f32, {}, {1.0f});
    خودکار اوپر_plus_one = std::make_shared<ov::op::v1::جوڑو>(اوپر, one);
    ov::Output<ov::Node> res = std::make_shared<ov::op::v1::Multiply>(برآمد_glu, اوپر_plus_one);
    اگر (res.get_element_type() != output_type) {
        res = std::make_shared<ov::op::v0::Convert>(res, output_type);
    }

    واپس rename_outputs_with_suffix({res}, context.get_name());
}

Output_سمتیہ translate_glu_swiglu_clamp(ثابت گرہ_Context & context) {
    خودکار [src0, src1] = get_glu_inputs(context);

    ثابت int32_t * params = context.get_output_op_params();
    ثابت اعشاری حد_بندی = reinterpret_cast<ثابت اعشاری *>(params)[3];

    // Compute in f32: f16 Swish/Clamp rounding drifts past the 1e-7 test tolerance.
    خودکار output_type = context.get_output_type();
    اگر (src0.get_element_type() != ov::element::f32) {
        src0 = std::make_shared<ov::op::v0::Convert>(src0, ov::element::f32);
    }
    اگر (src1.get_element_type() != ov::element::f32) {
        src1 = std::make_shared<ov::op::v0::Convert>(src1, ov::element::f32);
    }

    خودکار gate = std::make_shared<ov::op::v0::Clamp>(src0, -std::numeric_limits<اعشاری>::infinity(), حد_بندی);
    خودکار silu = std::make_shared<ov::op::v4::Swish>(gate);
    خودکار اوپر = std::make_shared<ov::op::v0::Clamp>(src1, -حد_بندی, حد_بندی);
    ov::Output<ov::Node> res = std::make_shared<ov::op::v1::Multiply>(silu, اوپر);
    اگر (res.get_element_type() != output_type) {
        res = std::make_shared<ov::op::v0::Convert>(res, output_type);
    }

    واپس rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
