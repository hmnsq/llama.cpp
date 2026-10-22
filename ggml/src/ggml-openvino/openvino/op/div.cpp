#ਅਨ੍ਤਰ੍ਭਾਵ "../node_context.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "../op_table.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "../utils.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml.h"

#ਅਨ੍ਤਰ੍ਭਾਵ <memory>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/core/type.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/constant.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/convert.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/divide.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/multiply.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/shape_of.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/sigmoid.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/swish.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/tile.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/util/precision_sensitive_attribute.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <vector>

ਨਾਮਾਕਾਸ਼ ov {
ਨਾਮਾਕਾਸ਼ frontend {
ਨਾਮਾਕਾਸ਼ ggml {
ਨਾਮਾਕਾਸ਼ op {

ਨਾਮਾਕਾਸ਼ {

ਤਾਰਕਿਕ is_silu_div_ਨਮੂਨਾ(ਅਚਲ ov::Output<ov::Node> & numerator,
                         ਅਚਲ ov::Output<ov::Node> & denominator,
                         ਅਚਲ ਗੰਢ_Context & context) {
    ਜੇ (context.get_ਨਿਵੇਸ਼_ਮਾਪ() != 2) {
        ਵਾਪਸ ਝੂਠ;
    }

    ਅਚਲ ਸਵੈਚਾਲਿਤ * unary_op = reinterpret_cast<ਅਚਲ ggml_unary_op *>(context.get_ਨਿਵੇਸ਼_op_params(0));
    ਜੇ (unary_op == ਖਾਲੀ_ਸੂਚਕ || *unary_op != GGML_UNARY_OP_SILU) {
        ਵਾਪਸ ਝੂਠ;
    }

    ਅਚਲ ਸਵੈਚਾਲਿਤ denom_ਗੰਢ = denominator.get_ਗੰਢ_shared_ptr();

    ਜੇ (ਸਵੈਚਾਲਿਤ swish = ov::as_type_ptr<ov::op::v4::Swish>(numerator.get_ਗੰਢ_shared_ptr())) {
        ਵਾਪਸ swish->ਨਿਵੇਸ਼_ਮੁੱਲ(0).get_ਗੰਢ_shared_ptr() == denom_ਗੰਢ;
    }
    ਵਾਪਸ ਝੂਠ;
}

ov::Output<ov::Node> repeat_ਨਿਵੇਸ਼_ਮੰਜ਼ਿਲ_ਮੇਲ(ਅਚਲ ਗੰਢ_Context & context,
                                           ਅਚਲ ov::Output<ov::Node> & ਨਿਵੇਸ਼,
                                           ਅਚਲ ov::Output<ov::Node> & ਨਿਸ਼ਾਨਾ,
                                           size_t ਨਿਵੇਸ਼_ਸੂਚਕਾਂਕ) {
    ਅਚਲ ਸਵੈਚਾਲਿਤ ਨਿਵੇਸ਼_shape = context.get_ਨਿਵੇਸ਼_shape(ਨਿਵੇਸ਼_ਸੂਚਕਾਂਕ);
    ਅਚਲ ਸਵੈਚਾਲਿਤ ਨਿਸ਼ਾਨਾ_shape = context.get_ਨਿਵੇਸ਼_shape(0);

    ਜੇ (ਨਿਵੇਸ਼_shape == ਨਿਸ਼ਾਨਾ_shape) {
        ਵਾਪਸ ਨਿਵੇਸ਼;
    }

    ਜੇ (ਨਿਵੇਸ਼_shape.rank().is_static() && ਨਿਸ਼ਾਨਾ_shape.rank().is_static()) {
        ਅਚਲ ਸਵੈਚਾਲਿਤ rank = static_cast<size_t>(ਨਿਵੇਸ਼_shape.rank().get_ਲੰਮਾਈ());
        std::vector<int64_t> repeats(rank, 1);
        ਤਾਰਕਿਕ needs_repeat = ਝੂਠ;

        ਲਈ (size_t axis = 0; axis < rank; ++axis) {
            FRONT_ਸਿਰਾ_OP_CONVERSION_CHECK(ਨਿਵੇਸ਼_shape[axis].is_static() && ਨਿਸ਼ਾਨਾ_shape[axis].is_static(),
                                          "DIV repeat requires static dimensions on both inputs");

            ਅਚਲ int64_t ਨਿਵੇਸ਼_dim = ਨਿਵੇਸ਼_shape[axis].get_ਲੰਮਾਈ();
            ਅਚਲ int64_t ਨਿਸ਼ਾਨਾ_dim = ਨਿਸ਼ਾਨਾ_shape[axis].get_ਲੰਮਾਈ();

            FRONT_ਸਿਰਾ_OP_CONVERSION_CHECK(ਨਿਵੇਸ਼_dim > 0 && ਨਿਸ਼ਾਨਾ_dim > 0 && ਨਿਸ਼ਾਨਾ_dim % ਨਿਵੇਸ਼_dim == 0,
                                          "DIV input shape ", ਨਿਵੇਸ਼_shape, " cannot repeat to match ", ਨਿਸ਼ਾਨਾ_shape);

            repeats[axis] = ਨਿਸ਼ਾਨਾ_dim / ਨਿਵੇਸ਼_dim;
            needs_repeat = needs_repeat || repeats[axis] != 1;
        }

        ਜੇ (!needs_repeat) {
            ਵਾਪਸ ਨਿਵੇਸ਼;
        }

        ਸਵੈਚਾਲਿਤ repeats_ਗੰਢ = ov::op::v0::Constant::create(ov::element::i64, {repeats.size()}, repeats);
        ਵਾਪਸ std::make_shared<ov::op::v0::Tile>(ਨਿਵੇਸ਼, repeats_ਗੰਢ);
    }

    ਸਵੈਚਾਲਿਤ ਨਿਵੇਸ਼_shape_ਗੰਢ = std::make_shared<ov::op::v3::Shape_ਦਾ>(ਨਿਵੇਸ਼, ov::element::i64);
    ਸਵੈਚਾਲਿਤ ਨਿਸ਼ਾਨਾ_shape_ਗੰਢ = std::make_shared<ov::op::v3::Shape_ਦਾ>(ਨਿਸ਼ਾਨਾ, ov::element::i64);
    ਸਵੈਚਾਲਿਤ repeats_ਗੰਢ = std::make_shared<ov::op::v1::Divide>(ਨਿਸ਼ਾਨਾ_shape_ਗੰਢ, ਨਿਵੇਸ਼_shape_ਗੰਢ);
    ਵਾਪਸ std::make_shared<ov::op::v0::Tile>(ਨਿਵੇਸ਼, repeats_ਗੰਢ);
}

}  // namespace

Output_ਸਦਿਸ਼ translate_div(ਅਚਲ ਗੰਢ_Context & context) {
    ਅੰਕ_inputs_check(context, 2, 2);

    ਸਵੈਚਾਲਿਤ ਨਿਵੇਸ਼_0 = process_view_ਨਿਵੇਸ਼_new(context, 0);
    ਸਵੈਚਾਲਿਤ ਨਿਵੇਸ਼_1 = process_view_ਨਿਵੇਸ਼_new(context, 1);

    ਜੇ (is_silu_div_ਨਮੂਨਾ(ਨਿਵੇਸ਼_0, ਨਿਵੇਸ਼_1, context)) {
        ov::Output<ov::Node> res = std::make_shared<ov::op::v0::Sigmoid>(ਨਿਵੇਸ਼_1);
        ਜੇ (res.get_element_type() != context.get_output_type()) {
            res = std::make_shared<ov::op::v0::Convert>(res, context.get_output_type());
        }
        ਵਾਪਸ rename_outputs_with_suffix({res}, context.get_name());
    }

    ਨਿਵੇਸ਼_1 = repeat_ਨਿਵੇਸ਼_ਮੰਜ਼ਿਲ_ਮੇਲ(context, ਨਿਵੇਸ਼_1, ਨਿਵੇਸ਼_0, 1);

    ਅਚਲ ਸਵੈਚਾਲਿਤ output_type = context.get_output_type();
    ਅਚਲ ਤਾਰਕਿਕ use_f32_compute = ਨਿਵੇਸ਼_0.get_element_type() != ov::element::f32 ||
                                 ਨਿਵੇਸ਼_1.get_element_type() != ov::element::f32 || output_type != ov::element::f32;

    ਜੇ (use_f32_compute) {
        ਨਿਵੇਸ਼_0 = std::make_shared<ov::op::v0::Convert>(ਨਿਵੇਸ਼_0, ov::element::f32);
        ਨਿਵੇਸ਼_1 = std::make_shared<ov::op::v0::Convert>(ਨਿਵੇਸ਼_1, ov::element::f32);
    }

    ov::Output<ov::Node> res = std::make_shared<ov::op::v1::Divide>(ਨਿਵੇਸ਼_0, ਨਿਵੇਸ਼_1);
    ਜੇ (use_f32_compute) {
        // Keep the reciprocal/divide path in FP32. Without this hint, the GPU
        // plugin can still compress the subgraph back to FP16 and overflow on
        // small shexp gate values (e.g. silu(x) / x in qwen2moe).
        ov::mark_as_precision_sensitive(res.get_ਗੰਢ_shared_ptr()->ਨਿਵੇਸ਼(0));
        ov::mark_as_precision_sensitive(res.get_ਗੰਢ_shared_ptr()->ਨਿਵੇਸ਼(1));
    }
    ਜੇ (res.get_element_type() != output_type) {
        ਸਵੈਚਾਲਿਤ output_convert = std::make_shared<ov::op::v0::Convert>(res, output_type);
        ਜੇ (use_f32_compute) {
            ov::mark_as_precision_sensitive(output_convert->ਨਿਵੇਸ਼(0));
        }
        res = output_convert;
    }
    ਵਾਪਸ rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
