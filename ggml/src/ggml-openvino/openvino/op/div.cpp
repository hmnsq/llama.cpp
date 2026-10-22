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
                         ثابت كلي_Context & context) {
    إذا (context.get_شغرو_size() != 2) {
        أرجع خطأ;
    }

    ثابت auto * unary_op = reinterpret_cast<ثابت ggml_unary_op *>(context.get_شغرو_op_params(0));
    إذا (unary_op == nullptr || *unary_op != GGML_UNARY_OP_SILU) {
        أرجع خطأ;
    }

    ثابت auto denom_كلي = denominator.get_كلي_shared_ptr();

    إذا (auto swish = ov::as_type_ptr<ov::op::v4::Swish>(numerator.get_كلي_shared_ptr())) {
        أرجع swish->شغرو_درج(0).get_كلي_shared_ptr() == denom_كلي;
    }
    أرجع خطأ;
}

ov::Output<ov::Node> repeat_شغرو_to_match(ثابت كلي_Context & context,
                                           ثابت ov::Output<ov::Node> & شغرو,
                                           ثابت ov::Output<ov::Node> & target,
                                           size_t شغرو_index) {
    ثابت auto شغرو_shape = context.get_شغرو_shape(شغرو_index);
    ثابت auto target_shape = context.get_شغرو_shape(0);

    إذا (شغرو_shape == target_shape) {
        أرجع شغرو;
    }

    إذا (شغرو_shape.rank().is_static() && target_shape.rank().is_static()) {
        ثابت auto rank = static_cast<size_t>(شغرو_shape.rank().get_length());
        std::vector<int64_t> repeats(rank, 1);
        منطقي needs_repeat = خطأ;

        لكل (size_t axis = 0; axis < rank; ++axis) {
            FRONT_END_OP_CONVERSION_CHECK(شغرو_shape[axis].is_static() && target_shape[axis].is_static(),
                                          "DIV repeat requires static dimensions on both inputs");

            ثابت int64_t شغرو_dim = شغرو_shape[axis].get_length();
            ثابت int64_t target_dim = target_shape[axis].get_length();

            FRONT_END_OP_CONVERSION_CHECK(شغرو_dim > 0 && target_dim > 0 && target_dim % شغرو_dim == 0,
                                          "DIV input shape ", شغرو_shape, " cannot repeat to match ", target_shape);

            repeats[axis] = target_dim / شغرو_dim;
            needs_repeat = needs_repeat || repeats[axis] != 1;
        }

        إذا (!needs_repeat) {
            أرجع شغرو;
        }

        auto repeats_كلي = ov::op::v0::Constant::create(ov::element::i64, {repeats.size()}, repeats);
        أرجع std::make_shared<ov::op::v0::Tile>(شغرو, repeats_كلي);
    }

    auto شغرو_shape_كلي = std::make_shared<ov::op::v3::ShapeOf>(شغرو, ov::element::i64);
    auto target_shape_كلي = std::make_shared<ov::op::v3::ShapeOf>(target, ov::element::i64);
    auto repeats_كلي = std::make_shared<ov::op::v1::Divide>(target_shape_كلي, شغرو_shape_كلي);
    أرجع std::make_shared<ov::op::v0::Tile>(شغرو, repeats_كلي);
}

}  // namespace

OutputVector translate_div(ثابت كلي_Context & context) {
    لمب_inputs_check(context, 2, 2);

    auto شغرو_0 = process_view_شغرو_new(context, 0);
    auto شغرو_1 = process_view_شغرو_new(context, 1);

    إذا (is_silu_div_pattern(شغرو_0, شغرو_1, context)) {
        ov::Output<ov::Node> res = std::make_shared<ov::op::v0::Sigmoid>(شغرو_1);
        إذا (res.get_element_type() != context.get_output_type()) {
            res = std::make_shared<ov::op::v0::Convert>(res, context.get_output_type());
        }
        أرجع rename_outputs_with_suffix({res}, context.get_name());
    }

    شغرو_1 = repeat_شغرو_to_match(context, شغرو_1, شغرو_0, 1);

    ثابت auto output_type = context.get_output_type();
    ثابت منطقي use_f32_compute = شغرو_0.get_element_type() != ov::element::f32 ||
                                 شغرو_1.get_element_type() != ov::element::f32 || output_type != ov::element::f32;

    إذا (use_f32_compute) {
        شغرو_0 = std::make_shared<ov::op::v0::Convert>(شغرو_0, ov::element::f32);
        شغرو_1 = std::make_shared<ov::op::v0::Convert>(شغرو_1, ov::element::f32);
    }

    ov::Output<ov::Node> res = std::make_shared<ov::op::v1::Divide>(شغرو_0, شغرو_1);
    إذا (use_f32_compute) {
        // Keep the reciprocal/divide path in FP32. Without this hint, the GPU
        // plugin can still compress the subgraph back to FP16 and overflow on
        // small shexp gate values (e.g. silu(x) / x in qwen2moe).
        ov::mark_as_precision_sensitive(res.get_كلي_shared_ptr()->شغرو(0));
        ov::mark_as_precision_sensitive(res.get_كلي_shared_ptr()->شغرو(1));
    }
    إذا (res.get_element_type() != output_type) {
        auto output_convert = std::make_shared<ov::op::v0::Convert>(res, output_type);
        إذا (use_f32_compute) {
            ov::mark_as_precision_sensitive(output_convert->شغرو(0));
        }
        res = output_convert;
    }
    أرجع rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
