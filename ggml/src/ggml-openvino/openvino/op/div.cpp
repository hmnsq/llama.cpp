#ضمن "../node_context.h"
#ضمن "../op_table.h"
#ضمن "../utils.h"
#ضمن "ggml.h"

#ضمن <memory>
#ضمن <openvino/core/type.hpp>
#ضمن <openvino/op/constant.hpp>
#ضمن <openvino/op/convert.hpp>
#ضمن <openvino/op/divide.hpp>
#ضمن <openvino/op/multiply.hpp>
#ضمن <openvino/op/shape_of.hpp>
#ضمن <openvino/op/sigmoid.hpp>
#ضمن <openvino/op/swish.hpp>
#ضمن <openvino/op/tile.hpp>
#ضمن <openvino/op/util/precision_sensitive_attribute.hpp>
#ضمن <vector>

نطاق ov {
نطاق frontend {
نطاق ggml {
نطاق op {

نطاق {

منطقي is_silu_div_pattern(ثابت ov::Output<ov::Node> & numerator,
                         ثابت ov::Output<ov::Node> & denominator,
                         ثابت گرێ_Context & context) {
    إذا (context.get_تێخستن_size() != 2) {
        أرجع خطأ;
    }

    ثابت auto * unary_op = reinterpret_cast<ثابت ggml_unary_op *>(context.get_تێخستن_op_params(0));
    إذا (unary_op == nullptr || *unary_op != GGML_UNARY_OP_SILU) {
        أرجع خطأ;
    }

    ثابت auto denom_گرێ = denominator.get_گرێ_shared_ptr();

    إذا (auto swish = ov::as_type_ptr<ov::op::v4::Swish>(numerator.get_گرێ_shared_ptr())) {
        أرجع swish->تێخستن_بەها(0).get_گرێ_shared_ptr() == denom_گرێ;
    }
    أرجع خطأ;
}

ov::Output<ov::Node> repeat_تێخستن_to_match(ثابت گرێ_Context & context,
                                           ثابت ov::Output<ov::Node> & تێخستن,
                                           ثابت ov::Output<ov::Node> & target,
                                           size_t تێخستن_index) {
    ثابت auto تێخستن_shape = context.get_تێخستن_shape(تێخستن_index);
    ثابت auto target_shape = context.get_تێخستن_shape(0);

    إذا (تێخستن_shape == target_shape) {
        أرجع تێخستن;
    }

    إذا (تێخستن_shape.rank().is_static() && target_shape.rank().is_static()) {
        ثابت auto rank = static_cast<size_t>(تێخستن_shape.rank().get_length());
        std::vector<int64_t> repeats(rank, 1);
        منطقي needs_repeat = خطأ;

        لكل (size_t axis = 0; axis < rank; ++axis) {
            FRONT_END_OP_CONVERSION_CHECK(تێخستن_shape[axis].is_static() && target_shape[axis].is_static(),
                                          "DIV repeat requires static dimensions on both inputs");

            ثابت int64_t تێخستن_dim = تێخستن_shape[axis].get_length();
            ثابت int64_t target_dim = target_shape[axis].get_length();

            FRONT_END_OP_CONVERSION_CHECK(تێخستن_dim > 0 && target_dim > 0 && target_dim % تێخستن_dim == 0,
                                          "DIV input shape ", تێخستن_shape, " cannot repeat to match ", target_shape);

            repeats[axis] = target_dim / تێخستن_dim;
            needs_repeat = needs_repeat || repeats[axis] != 1;
        }

        إذا (!needs_repeat) {
            أرجع تێخستن;
        }

        auto repeats_گرێ = ov::op::v0::Constant::create(ov::element::i64, {repeats.size()}, repeats);
        أرجع std::make_shared<ov::op::v0::Tile>(تێخستن, repeats_گرێ);
    }

    auto تێخستن_shape_گرێ = std::make_shared<ov::op::v3::ShapeOf>(تێخستن, ov::element::i64);
    auto target_shape_گرێ = std::make_shared<ov::op::v3::ShapeOf>(target, ov::element::i64);
    auto repeats_گرێ = std::make_shared<ov::op::v1::Divide>(target_shape_گرێ, تێخستن_shape_گرێ);
    أرجع std::make_shared<ov::op::v0::Tile>(تێخستن, repeats_گرێ);
}

}  // namespace

OutputVector translate_div(ثابت گرێ_Context & context) {
    ژمارە_inputs_check(context, 2, 2);

    auto تێخستن_0 = process_view_تێخستن_new(context, 0);
    auto تێخستن_1 = process_view_تێخستن_new(context, 1);

    إذا (is_silu_div_pattern(تێخستن_0, تێخستن_1, context)) {
        ov::Output<ov::Node> res = std::make_shared<ov::op::v0::Sigmoid>(تێخستن_1);
        إذا (res.get_element_type() != context.get_output_type()) {
            res = std::make_shared<ov::op::v0::Convert>(res, context.get_output_type());
        }
        أرجع rename_outputs_with_suffix({res}, context.get_name());
    }

    تێخستن_1 = repeat_تێخستن_to_match(context, تێخستن_1, تێخستن_0, 1);

    ثابت auto output_type = context.get_output_type();
    ثابت منطقي use_f32_compute = تێخستن_0.get_element_type() != ov::element::f32 ||
                                 تێخستن_1.get_element_type() != ov::element::f32 || output_type != ov::element::f32;

    إذا (use_f32_compute) {
        تێخستن_0 = std::make_shared<ov::op::v0::Convert>(تێخستن_0, ov::element::f32);
        تێخستن_1 = std::make_shared<ov::op::v0::Convert>(تێخستن_1, ov::element::f32);
    }

    ov::Output<ov::Node> res = std::make_shared<ov::op::v1::Divide>(تێخستن_0, تێخستن_1);
    إذا (use_f32_compute) {
        // Keep the reciprocal/divide path in FP32. Without this hint, the GPU
        // plugin can still compress the subgraph back to FP16 and overflow on
        // small shexp gate values (e.g. silu(x) / x in qwen2moe).
        ov::mark_as_precision_sensitive(res.get_گرێ_shared_ptr()->تێخستن(0));
        ov::mark_as_precision_sensitive(res.get_گرێ_shared_ptr()->تێخستن(1));
    }
    إذا (res.get_element_type() != output_type) {
        auto output_convert = std::make_shared<ov::op::v0::Convert>(res, output_type);
        إذا (use_f32_compute) {
            ov::mark_as_precision_sensitive(output_convert->تێخستن(0));
        }
        res = output_convert;
    }
    أرجع rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
