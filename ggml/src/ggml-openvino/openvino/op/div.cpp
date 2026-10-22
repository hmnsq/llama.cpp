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

منطقي is_silu_div_نمونه(ثابت ov::Output<ov::Node> & numerator,
                         ثابت ov::Output<ov::Node> & denominator,
                         ثابت غوټه_Context & context) {
    إذا (context.get_ننوت_اندازه() != 2) {
        أرجع خطأ;
    }

    ثابت خپلکاری * unary_op = reinterpret_cast<ثابت ggml_unary_op *>(context.get_ننوت_op_params(0));
    إذا (unary_op == تش_اشاره || *unary_op != GGML_UNARY_OP_SILU) {
        أرجع خطأ;
    }

    ثابت خپلکاری denom_غوټه = denominator.get_غوټه_shared_ptr();

    إذا (خپلکاری swish = ov::as_type_ptr<ov::op::v4::Swish>(numerator.get_غوټه_shared_ptr())) {
        أرجع swish->ننوت_ارزښت(0).get_غوټه_shared_ptr() == denom_غوټه;
    }
    أرجع خطأ;
}

ov::Output<ov::Node> repeat_ننوت_منزل_سمون(ثابت غوټه_Context & context,
                                           ثابت ov::Output<ov::Node> & ننوت,
                                           ثابت ov::Output<ov::Node> & موخه,
                                           size_t ننوت_شاخص) {
    ثابت خپلکاری ننوت_shape = context.get_ننوت_shape(ننوت_شاخص);
    ثابت خپلکاری موخه_shape = context.get_ننوت_shape(0);

    إذا (ننوت_shape == موخه_shape) {
        أرجع ننوت;
    }

    إذا (ننوت_shape.rank().is_static() && موخه_shape.rank().is_static()) {
        ثابت خپلکاری rank = static_cast<size_t>(ننوت_shape.rank().get_اوږدتیا());
        std::vector<int64_t> repeats(rank, 1);
        منطقي needs_repeat = خطأ;

        لكل (size_t axis = 0; axis < rank; ++axis) {
            FRONT_پای_ټکی_OP_CONVERSION_CHECK(ننوت_shape[axis].is_static() && موخه_shape[axis].is_static(),
                                          "DIV repeat requires static dimensions on both inputs");

            ثابت int64_t ننوت_dim = ننوت_shape[axis].get_اوږدتیا();
            ثابت int64_t موخه_dim = موخه_shape[axis].get_اوږدتیا();

            FRONT_پای_ټکی_OP_CONVERSION_CHECK(ننوت_dim > 0 && موخه_dim > 0 && موخه_dim % ننوت_dim == 0,
                                          "DIV input shape ", ننوت_shape, " cannot repeat to match ", موخه_shape);

            repeats[axis] = موخه_dim / ننوت_dim;
            needs_repeat = needs_repeat || repeats[axis] != 1;
        }

        إذا (!needs_repeat) {
            أرجع ننوت;
        }

        خپلکاری repeats_غوټه = ov::op::v0::Constant::create(ov::element::i64, {repeats.size()}, repeats);
        أرجع std::make_shared<ov::op::v0::Tile>(ننوت, repeats_غوټه);
    }

    خپلکاری ننوت_shape_غوټه = std::make_shared<ov::op::v3::Shape_د>(ننوت, ov::element::i64);
    خپلکاری موخه_shape_غوټه = std::make_shared<ov::op::v3::Shape_د>(موخه, ov::element::i64);
    خپلکاری repeats_غوټه = std::make_shared<ov::op::v1::Divide>(موخه_shape_غوټه, ننوت_shape_غوټه);
    أرجع std::make_shared<ov::op::v0::Tile>(ننوت, repeats_غوټه);
}

}  // namespace

Output_ویکټور translate_div(ثابت غوټه_Context & context) {
    عدد_inputs_check(context, 2, 2);

    خپلکاری ننوت_0 = process_view_ننوت_new(context, 0);
    خپلکاری ننوت_1 = process_view_ننوت_new(context, 1);

    إذا (is_silu_div_نمونه(ننوت_0, ننوت_1, context)) {
        ov::Output<ov::Node> res = std::make_shared<ov::op::v0::Sigmoid>(ننوت_1);
        إذا (res.get_element_type() != context.get_output_type()) {
            res = std::make_shared<ov::op::v0::Convert>(res, context.get_output_type());
        }
        أرجع rename_outputs_with_suffix({res}, context.get_name());
    }

    ننوت_1 = repeat_ننوت_منزل_سمون(context, ننوت_1, ننوت_0, 1);

    ثابت خپلکاری output_type = context.get_output_type();
    ثابت منطقي use_f32_compute = ننوت_0.get_element_type() != ov::element::f32 ||
                                 ننوت_1.get_element_type() != ov::element::f32 || output_type != ov::element::f32;

    إذا (use_f32_compute) {
        ننوت_0 = std::make_shared<ov::op::v0::Convert>(ننوت_0, ov::element::f32);
        ننوت_1 = std::make_shared<ov::op::v0::Convert>(ننوت_1, ov::element::f32);
    }

    ov::Output<ov::Node> res = std::make_shared<ov::op::v1::Divide>(ننوت_0, ننوت_1);
    إذا (use_f32_compute) {
        // Keep the reciprocal/divide path in FP32. Without this hint, the GPU
        // plugin can still compress the subgraph back to FP16 and overflow on
        // small shexp gate values (e.g. silu(x) / x in qwen2moe).
        ov::mark_as_precision_sensitive(res.get_غوټه_shared_ptr()->ننوت(0));
        ov::mark_as_precision_sensitive(res.get_غوټه_shared_ptr()->ننوت(1));
    }
    إذا (res.get_element_type() != output_type) {
        خپلکاری output_convert = std::make_shared<ov::op::v0::Convert>(res, output_type);
        إذا (use_f32_compute) {
            ov::mark_as_precision_sensitive(output_convert->ننوت(0));
        }
        res = output_convert;
    }
    أرجع rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
