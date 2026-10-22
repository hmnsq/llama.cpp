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

منطقي is_silu_div_الگو(ثابت ov::Output<ov::Node> & numerator,
                         ثابت ov::Output<ov::Node> & denominator,
                         ثابت گره_Context & context) {
    إذا (context.get_درون_داد_اندازه() != 2) {
        أرجع خطأ;
    }

    ثابت خودکار * unary_op = reinterpret_cast<ثابت ggml_unary_op *>(context.get_درون_داد_op_params(0));
    إذا (unary_op == اشاره_تهی || *unary_op != GGML_UNARY_OP_SILU) {
        أرجع خطأ;
    }

    ثابت خودکار denom_گره = denominator.get_گره_shared_ptr();

    إذا (خودکار swish = ov::as_type_ptr<ov::op::v4::Swish>(numerator.get_گره_shared_ptr())) {
        أرجع swish->درون_داد_مقدار(0).get_گره_shared_ptr() == denom_گره;
    }
    أرجع خطأ;
}

ov::Output<ov::Node> repeat_درون_داد_مقصد_تطابق(ثابت گره_Context & context,
                                           ثابت ov::Output<ov::Node> & درون_داد,
                                           ثابت ov::Output<ov::Node> & هدف,
                                           size_t درون_داد_شاخص) {
    ثابت خودکار درون_داد_shape = context.get_درون_داد_shape(درون_داد_شاخص);
    ثابت خودکار هدف_shape = context.get_درون_داد_shape(0);

    إذا (درون_داد_shape == هدف_shape) {
        أرجع درون_داد;
    }

    إذا (درون_داد_shape.rank().is_static() && هدف_shape.rank().is_static()) {
        ثابت خودکار rank = static_cast<size_t>(درون_داد_shape.rank().get_درازی());
        std::vector<int64_t> repeats(rank, 1);
        منطقي needs_repeat = خطأ;

        لكل (size_t axis = 0; axis < rank; ++axis) {
            FRONT_انتها_OP_CONVERSION_CHECK(درون_داد_shape[axis].is_static() && هدف_shape[axis].is_static(),
                                          "DIV repeat requires static dimensions on both inputs");

            ثابت int64_t درون_داد_dim = درون_داد_shape[axis].get_درازی();
            ثابت int64_t هدف_dim = هدف_shape[axis].get_درازی();

            FRONT_انتها_OP_CONVERSION_CHECK(درون_داد_dim > 0 && هدف_dim > 0 && هدف_dim % درون_داد_dim == 0,
                                          "DIV input shape ", درون_داد_shape, " cannot repeat to match ", هدف_shape);

            repeats[axis] = هدف_dim / درون_داد_dim;
            needs_repeat = needs_repeat || repeats[axis] != 1;
        }

        إذا (!needs_repeat) {
            أرجع درون_داد;
        }

        خودکار repeats_گره = ov::op::v0::Constant::create(ov::element::i64, {repeats.size()}, repeats);
        أرجع std::make_shared<ov::op::v0::Tile>(درون_داد, repeats_گره);
    }

    خودکار درون_داد_shape_گره = std::make_shared<ov::op::v3::Shape_از>(درون_داد, ov::element::i64);
    خودکار هدف_shape_گره = std::make_shared<ov::op::v3::Shape_از>(هدف, ov::element::i64);
    خودکار repeats_گره = std::make_shared<ov::op::v1::Divide>(هدف_shape_گره, درون_داد_shape_گره);
    أرجع std::make_shared<ov::op::v0::Tile>(درون_داد, repeats_گره);
}

}  // namespace

Output_وکتور translate_div(ثابت گره_Context & context) {
    رقم_inputs_check(context, 2, 2);

    خودکار درون_داد_0 = process_view_درون_داد_new(context, 0);
    خودکار درون_داد_1 = process_view_درون_داد_new(context, 1);

    إذا (is_silu_div_الگو(درون_داد_0, درون_داد_1, context)) {
        ov::Output<ov::Node> res = std::make_shared<ov::op::v0::Sigmoid>(درون_داد_1);
        إذا (res.get_element_type() != context.get_output_type()) {
            res = std::make_shared<ov::op::v0::Convert>(res, context.get_output_type());
        }
        أرجع rename_outputs_with_suffix({res}, context.get_name());
    }

    درون_داد_1 = repeat_درون_داد_مقصد_تطابق(context, درون_داد_1, درون_داد_0, 1);

    ثابت خودکار output_type = context.get_output_type();
    ثابت منطقي use_f32_compute = درون_داد_0.get_element_type() != ov::element::f32 ||
                                 درون_داد_1.get_element_type() != ov::element::f32 || output_type != ov::element::f32;

    إذا (use_f32_compute) {
        درون_داد_0 = std::make_shared<ov::op::v0::Convert>(درون_داد_0, ov::element::f32);
        درون_داد_1 = std::make_shared<ov::op::v0::Convert>(درون_داد_1, ov::element::f32);
    }

    ov::Output<ov::Node> res = std::make_shared<ov::op::v1::Divide>(درون_داد_0, درون_داد_1);
    إذا (use_f32_compute) {
        // Keep the reciprocal/divide path in FP32. Without this hint, the GPU
        // plugin can still compress the subgraph back to FP16 and overflow on
        // small shexp gate values (e.g. silu(x) / x in qwen2moe).
        ov::mark_as_precision_sensitive(res.get_گره_shared_ptr()->درون_داد(0));
        ov::mark_as_precision_sensitive(res.get_گره_shared_ptr()->درون_داد(1));
    }
    إذا (res.get_element_type() != output_type) {
        خودکار output_convert = std::make_shared<ov::op::v0::Convert>(res, output_type);
        إذا (use_f32_compute) {
            ov::mark_as_precision_sensitive(output_convert->درون_داد(0));
        }
        res = output_convert;
    }
    أرجع rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
