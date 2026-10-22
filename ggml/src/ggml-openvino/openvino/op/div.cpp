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
                         ثابت عقدة_Context & context) {
    إذا (context.get_إدخال_size() != 2) {
        إرجاع خطأ;
    }

    ثابت auto * unary_op = reinterpret_cast<ثابت ggml_unary_op *>(context.get_إدخال_op_params(0));
    إذا (unary_op == nullptr || *unary_op != GGML_UNARY_OP_SILU) {
        إرجاع خطأ;
    }

    ثابت auto denom_عقدة = denominator.get_عقدة_shared_ptr();

    إذا (auto swish = ov::as_type_ptr<ov::op::v4::Swish>(numerator.get_عقدة_shared_ptr())) {
        إرجاع swish->إدخال_قيمة(0).get_عقدة_shared_ptr() == denom_عقدة;
    }
    إرجاع خطأ;
}

ov::Output<ov::Node> repeat_إدخال_to_match(ثابت عقدة_Context & context,
                                           ثابت ov::Output<ov::Node> & إدخال,
                                           ثابت ov::Output<ov::Node> & target,
                                           size_t إدخال_index) {
    ثابت auto إدخال_shape = context.get_إدخال_shape(إدخال_index);
    ثابت auto target_shape = context.get_إدخال_shape(0);

    إذا (إدخال_shape == target_shape) {
        إرجاع إدخال;
    }

    إذا (إدخال_shape.rank().is_static() && target_shape.rank().is_static()) {
        ثابت auto rank = static_cast<size_t>(إدخال_shape.rank().get_length());
        std::vector<int64_t> repeats(rank, 1);
        منطقي needs_repeat = خطأ;

        لكل (size_t axis = 0; axis < rank; ++axis) {
            FRONT_END_OP_CONVERSION_CHECK(إدخال_shape[axis].is_static() && target_shape[axis].is_static(),
                                          "DIV repeat requires static dimensions on both inputs");

            ثابت int64_t إدخال_dim = إدخال_shape[axis].get_length();
            ثابت int64_t target_dim = target_shape[axis].get_length();

            FRONT_END_OP_CONVERSION_CHECK(إدخال_dim > 0 && target_dim > 0 && target_dim % إدخال_dim == 0,
                                          "DIV input shape ", إدخال_shape, " cannot repeat to match ", target_shape);

            repeats[axis] = target_dim / إدخال_dim;
            needs_repeat = needs_repeat || repeats[axis] != 1;
        }

        إذا (!needs_repeat) {
            إرجاع إدخال;
        }

        auto repeats_عقدة = ov::op::v0::Constant::create(ov::element::i64, {repeats.size()}, repeats);
        إرجاع std::make_shared<ov::op::v0::Tile>(إدخال, repeats_عقدة);
    }

    auto إدخال_shape_عقدة = std::make_shared<ov::op::v3::ShapeOf>(إدخال, ov::element::i64);
    auto target_shape_عقدة = std::make_shared<ov::op::v3::ShapeOf>(target, ov::element::i64);
    auto repeats_عقدة = std::make_shared<ov::op::v1::Divide>(target_shape_عقدة, إدخال_shape_عقدة);
    إرجاع std::make_shared<ov::op::v0::Tile>(إدخال, repeats_عقدة);
}

}  // namespace

OutputVector translate_div(ثابت عقدة_Context & context) {
    رقم_inputs_check(context, 2, 2);

    auto إدخال_0 = process_view_إدخال_new(context, 0);
    auto إدخال_1 = process_view_إدخال_new(context, 1);

    إذا (is_silu_div_pattern(إدخال_0, إدخال_1, context)) {
        ov::Output<ov::Node> res = std::make_shared<ov::op::v0::Sigmoid>(إدخال_1);
        إذا (res.get_element_type() != context.get_output_type()) {
            res = std::make_shared<ov::op::v0::Convert>(res, context.get_output_type());
        }
        إرجاع rename_outputs_with_suffix({res}, context.get_name());
    }

    إدخال_1 = repeat_إدخال_to_match(context, إدخال_1, إدخال_0, 1);

    ثابت auto output_type = context.get_output_type();
    ثابت منطقي use_f32_compute = إدخال_0.get_element_type() != ov::element::f32 ||
                                 إدخال_1.get_element_type() != ov::element::f32 || output_type != ov::element::f32;

    إذا (use_f32_compute) {
        إدخال_0 = std::make_shared<ov::op::v0::Convert>(إدخال_0, ov::element::f32);
        إدخال_1 = std::make_shared<ov::op::v0::Convert>(إدخال_1, ov::element::f32);
    }

    ov::Output<ov::Node> res = std::make_shared<ov::op::v1::Divide>(إدخال_0, إدخال_1);
    إذا (use_f32_compute) {
        // Keep the reciprocal/divide path in FP32. Without this hint, the GPU
        // plugin can still compress the subgraph back to FP16 and overflow on
        // small shexp gate values (e.g. silu(x) / x in qwen2moe).
        ov::mark_as_precision_sensitive(res.get_عقدة_shared_ptr()->إدخال(0));
        ov::mark_as_precision_sensitive(res.get_عقدة_shared_ptr()->إدخال(1));
    }
    إذا (res.get_element_type() != output_type) {
        auto output_convert = std::make_shared<ov::op::v0::Convert>(res, output_type);
        إذا (use_f32_compute) {
            ov::mark_as_precision_sensitive(output_convert->إدخال(0));
        }
        res = output_convert;
    }
    إرجاع rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
