#ਅਨ੍ਤਰ੍ਭਾਵ "../node_context.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "../op_table.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "../utils.h"

#ਅਨ੍ਤਰ੍ਭਾਵ <cstdint>
#ਅਨ੍ਤਰ੍ਭਾਵ <limits>
#ਅਨ੍ਤਰ੍ਭਾਵ <memory>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/core/node_output.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/add.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/clamp.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/constant.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/convert.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/multiply.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/slice.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/swish.hpp>

ਨਾਮਾਕਾਸ਼ ov {
ਨਾਮਾਕਾਸ਼ frontend {
ਨਾਮਾਕਾਸ਼ ggml {
ਨਾਮਾਕਾਸ਼ op {

ਸਥਿਰ std::pair<ov::Output<ov::Node>, ov::Output<ov::Node>> get_glu_inputs(ਅਚਲ ਗੰਢ_Context & context) {
    ਅੰਕ_inputs_check(context, 1, 2);

    ov::Output<ov::Node> src0;
    ov::Output<ov::Node> src1;
    ਜੇ (context.get_ਨਿਵੇਸ਼_ਮਾਪ() == 2) {
        // Inputs may be VIEW slices of a combined gate_up tensor (MoE experts):
        // resolve them so each half has its real sliced shape, not the base tensor.
        src0 = process_view_ਨਿਵੇਸ਼_new(context, 0);
        src1 = process_view_ਨਿਵੇਸ਼_new(context, 1);
    } ਨਹੀਂਤਾਂ {
        // GGML splits along ne[0] (OV last axis) using floor division: nc = ne[0] / 2.
        // Both halves are nc elements; if the dimension is odd, the last element is dropped.
        // Use Slice instead of Split to handle odd dimensions correctly.
        // Resolve a VIEW input (e.g. non-contiguous slice) to its real shape first.
        ਸਵੈਚਾਲਿਤ combined = process_view_ਨਿਵੇਸ਼_new(context, 0);
        ਸਵੈਚਾਲਿਤ combined_shape = combined.get_partial_shape();
        int64_t ਆਖਰੀ_dim_val = combined_shape[combined_shape.rank().get_ਲੰਮਾਈ() - 1].get_ਲੰਮਾਈ();
        int64_t nc = ਆਖਰੀ_dim_val / 2;

        ਸਵੈਚਾਲਿਤ axis = ov::op::v0::Constant::create(ov::element::i64, {1}, {-1});
        ਸਵੈਚਾਲਿਤ ਕਦਮ = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
        ਸਵੈਚਾਲਿਤ start0 = ov::op::v0::Constant::create(ov::element::i64, {1}, {0});
        ਸਵੈਚਾਲਿਤ stop0 = ov::op::v0::Constant::create(ov::element::i64, {1}, {nc});
        ਸਵੈਚਾਲਿਤ start1 = ov::op::v0::Constant::create(ov::element::i64, {1}, {nc});
        ਸਵੈਚਾਲਿਤ stop1 = ov::op::v0::Constant::create(ov::element::i64, {1}, {2 * nc});

        src0 = std::make_shared<ov::op::v8::ਟੁਕੜਾ>(combined, start0, stop0, ਕਦਮ, axis);
        src1 = std::make_shared<ov::op::v8::ਟੁਕੜਾ>(combined, start1, stop1, ਕਦਮ, axis);
    }

    int32_t * params = context.get_output_op_params();
    ਅਚਲ int32_t swapped = params[1];
    ਜੇ (swapped) {
        std::swap(src0, src1);
    }

    ਵਾਪਸ {src0, src1};
}

Output_ਸਦਿਸ਼ translate_glu_swiglu(ਅਚਲ ਗੰਢ_Context & context) {
    ਸਵੈਚਾਲਿਤ [src0, src1] = get_glu_inputs(context);

    ਸਵੈਚਾਲਿਤ silu = std::make_shared<ov::op::v4::Swish>(src0);
    ਸਵੈਚਾਲਿਤ res = std::make_shared<ov::op::v1::Multiply>(silu, src1);

    ਵਾਪਸ rename_outputs_with_suffix({res}, context.get_name());
}

Output_ਸਦਿਸ਼ translate_glu_swiglu_oai(ਅਚਲ ਗੰਢ_Context & context) {
    ਸਵੈਚਾਲਿਤ [src0, src1] = get_glu_inputs(context);

    ਅਚਲ int32_t * params = context.get_output_op_params();
    ਅਚਲ ਦਸ਼ਮਲਵ alpha = reinterpret_cast<ਅਚਲ ਦਸ਼ਮਲਵ *>(params)[2];
    ਅਚਲ ਦਸ਼ਮਲਵ ਹੱਦਬੰਦੀ = reinterpret_cast<ਅਚਲ ਦਸ਼ਮਲਵ *>(params)[3];

    // Compute in f32: alpha_const/one are f32, and Swish/Add need matching input types.
    ਸਵੈਚਾਲਿਤ output_type = context.get_output_type();
    ਜੇ (src0.get_element_type() != ov::element::f32) {
        src0 = std::make_shared<ov::op::v0::Convert>(src0, ov::element::f32);
    }
    ਜੇ (src1.get_element_type() != ov::element::f32) {
        src1 = std::make_shared<ov::op::v0::Convert>(src1, ov::element::f32);
    }

    ਸਵੈਚਾਲਿਤ gate = std::make_shared<ov::op::v0::Clamp>(src0, -std::numeric_limits<ਦਸ਼ਮਲਵ>::infinity(), ਹੱਦਬੰਦੀ);
    ਸਵੈਚਾਲਿਤ alpha_const = ov::op::v0::Constant::create(ov::element::f32, {}, {alpha});
    ਸਵੈਚਾਲਿਤ ਬਾਹਰਲਾ_glu = std::make_shared<ov::op::v4::Swish>(gate, alpha_const);

    ਸਵੈਚਾਲਿਤ ਉੱਪਰ = std::make_shared<ov::op::v0::Clamp>(src1, -ਹੱਦਬੰਦੀ, ਹੱਦਬੰਦੀ);
    ਸਵੈਚਾਲਿਤ one = ov::op::v0::Constant::create(ov::element::f32, {}, {1.0f});
    ਸਵੈਚਾਲਿਤ ਉੱਪਰ_plus_one = std::make_shared<ov::op::v1::ਜੋੜੋ>(ਉੱਪਰ, one);
    ov::Output<ov::Node> res = std::make_shared<ov::op::v1::Multiply>(ਬਾਹਰਲਾ_glu, ਉੱਪਰ_plus_one);
    ਜੇ (res.get_element_type() != output_type) {
        res = std::make_shared<ov::op::v0::Convert>(res, output_type);
    }

    ਵਾਪਸ rename_outputs_with_suffix({res}, context.get_name());
}

Output_ਸਦਿਸ਼ translate_glu_swiglu_clamp(ਅਚਲ ਗੰਢ_Context & context) {
    ਸਵੈਚਾਲਿਤ [src0, src1] = get_glu_inputs(context);

    ਅਚਲ int32_t * params = context.get_output_op_params();
    ਅਚਲ ਦਸ਼ਮਲਵ ਹੱਦਬੰਦੀ = reinterpret_cast<ਅਚਲ ਦਸ਼ਮਲਵ *>(params)[3];

    // Compute in f32: f16 Swish/Clamp rounding drifts past the 1e-7 test tolerance.
    ਸਵੈਚਾਲਿਤ output_type = context.get_output_type();
    ਜੇ (src0.get_element_type() != ov::element::f32) {
        src0 = std::make_shared<ov::op::v0::Convert>(src0, ov::element::f32);
    }
    ਜੇ (src1.get_element_type() != ov::element::f32) {
        src1 = std::make_shared<ov::op::v0::Convert>(src1, ov::element::f32);
    }

    ਸਵੈਚਾਲਿਤ gate = std::make_shared<ov::op::v0::Clamp>(src0, -std::numeric_limits<ਦਸ਼ਮਲਵ>::infinity(), ਹੱਦਬੰਦੀ);
    ਸਵੈਚਾਲਿਤ silu = std::make_shared<ov::op::v4::Swish>(gate);
    ਸਵੈਚਾਲਿਤ ਉੱਪਰ = std::make_shared<ov::op::v0::Clamp>(src1, -ਹੱਦਬੰਦੀ, ਹੱਦਬੰਦੀ);
    ov::Output<ov::Node> res = std::make_shared<ov::op::v1::Multiply>(silu, ਉੱਪਰ);
    ਜੇ (res.get_element_type() != output_type) {
        res = std::make_shared<ov::op::v0::Convert>(res, output_type);
    }

    ਵਾਪਸ rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
