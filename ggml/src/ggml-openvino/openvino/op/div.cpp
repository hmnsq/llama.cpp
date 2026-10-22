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
                         ثابت نود_Context & context) {
    إذا (context.get_ماسوقن_size() != 2) {
        أرجع خطأ;
    }

    ثابت auto * unary_op = reinterpret_cast<ثابت ggml_unary_op *>(context.get_ماسوقن_op_params(0));
    إذا (unary_op == nullptr || *unary_op != GGML_UNARY_OP_SILU) {
        أرجع خطأ;
    }

    ثابت auto denom_نود = denominator.get_نود_shared_ptr();

    إذا (auto swish = ov::as_type_ptr<ov::op::v4::Swish>(numerator.get_نود_shared_ptr())) {
        أرجع swish->ماسوقن_نيلاي(0).get_نود_shared_ptr() == denom_نود;
    }
    أرجع خطأ;
}

ov::Output<ov::Node> repeat_ماسوقن_to_match(ثابت نود_Context & context,
                                           ثابت ov::Output<ov::Node> & ماسوقن,
                                           ثابت ov::Output<ov::Node> & target,
                                           size_t ماسوقن_index) {
    ثابت auto ماسوقن_shape = context.get_ماسوقن_shape(ماسوقن_index);
    ثابت auto target_shape = context.get_ماسوقن_shape(0);

    إذا (ماسوقن_shape == target_shape) {
        أرجع ماسوقن;
    }

    إذا (ماسوقن_shape.rank().is_static() && target_shape.rank().is_static()) {
        ثابت auto rank = static_cast<size_t>(ماسوقن_shape.rank().get_length());
        std::vector<int64_t> repeats(rank, 1);
        منطقي needs_repeat = خطأ;

        لكل (size_t axis = 0; axis < rank; ++axis) {
            FRONT_END_OP_CONVERSION_CHECK(ماسوقن_shape[axis].is_static() && target_shape[axis].is_static(),
                                          "DIV repeat requires static dimensions on both inputs");

            ثابت int64_t ماسوقن_dim = ماسوقن_shape[axis].get_length();
            ثابت int64_t target_dim = target_shape[axis].get_length();

            FRONT_END_OP_CONVERSION_CHECK(ماسوقن_dim > 0 && target_dim > 0 && target_dim % ماسوقن_dim == 0,
                                          "DIV input shape ", ماسوقن_shape, " cannot repeat to match ", target_shape);

            repeats[axis] = target_dim / ماسوقن_dim;
            needs_repeat = needs_repeat || repeats[axis] != 1;
        }

        إذا (!needs_repeat) {
            أرجع ماسوقن;
        }

        auto repeats_نود = ov::op::v0::Constant::create(ov::element::i64, {repeats.size()}, repeats);
        أرجع std::make_shared<ov::op::v0::Tile>(ماسوقن, repeats_نود);
    }

    auto ماسوقن_shape_نود = std::make_shared<ov::op::v3::ShapeOf>(ماسوقن, ov::element::i64);
    auto target_shape_نود = std::make_shared<ov::op::v3::ShapeOf>(target, ov::element::i64);
    auto repeats_نود = std::make_shared<ov::op::v1::Divide>(target_shape_نود, ماسوقن_shape_نود);
    أرجع std::make_shared<ov::op::v0::Tile>(ماسوقن, repeats_نود);
}

}  // namespace

OutputVector translate_div(ثابت نود_Context & context) {
    اڠک_inputs_check(context, 2, 2);

    auto ماسوقن_0 = process_view_ماسوقن_new(context, 0);
    auto ماسوقن_1 = process_view_ماسوقن_new(context, 1);

    إذا (is_silu_div_pattern(ماسوقن_0, ماسوقن_1, context)) {
        ov::Output<ov::Node> res = std::make_shared<ov::op::v0::Sigmoid>(ماسوقن_1);
        إذا (res.get_element_type() != context.get_output_type()) {
            res = std::make_shared<ov::op::v0::Convert>(res, context.get_output_type());
        }
        أرجع rename_outputs_with_suffix({res}, context.get_name());
    }

    ماسوقن_1 = repeat_ماسوقن_to_match(context, ماسوقن_1, ماسوقن_0, 1);

    ثابت auto output_type = context.get_output_type();
    ثابت منطقي use_f32_compute = ماسوقن_0.get_element_type() != ov::element::f32 ||
                                 ماسوقن_1.get_element_type() != ov::element::f32 || output_type != ov::element::f32;

    إذا (use_f32_compute) {
        ماسوقن_0 = std::make_shared<ov::op::v0::Convert>(ماسوقن_0, ov::element::f32);
        ماسوقن_1 = std::make_shared<ov::op::v0::Convert>(ماسوقن_1, ov::element::f32);
    }

    ov::Output<ov::Node> res = std::make_shared<ov::op::v1::Divide>(ماسوقن_0, ماسوقن_1);
    إذا (use_f32_compute) {
        // Keep the reciprocal/divide path in FP32. Without this hint, the GPU
        // plugin can still compress the subgraph back to FP16 and overflow on
        // small shexp gate values (e.g. silu(x) / x in qwen2moe).
        ov::mark_as_precision_sensitive(res.get_نود_shared_ptr()->ماسوقن(0));
        ov::mark_as_precision_sensitive(res.get_نود_shared_ptr()->ماسوقن(1));
    }
    إذا (res.get_element_type() != output_type) {
        auto output_convert = std::make_shared<ov::op::v0::Convert>(res, output_type);
        إذا (use_f32_compute) {
            ov::mark_as_precision_sensitive(output_convert->ماسوقن(0));
        }
        res = output_convert;
    }
    أرجع rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
