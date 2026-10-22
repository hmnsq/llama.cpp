#ضمن "../node_context.h"
#ضمن "../op_table.h"
#ضمن "../utils.h"
#ضمن "ggml-openvino/ggml-openvino-extra.h"

#ضمن <memory>
#ضمن <openvino/op/abs.hpp>
#ضمن <openvino/op/add.hpp>
#ضمن <openvino/op/clamp.hpp>
#ضمن <openvino/op/constant.hpp>
#ضمن <openvino/op/convert.hpp>
#ضمن <openvino/op/elu.hpp>
#ضمن <openvino/op/exp.hpp>
#ضمن <openvino/op/gelu.hpp>
#ضمن <openvino/op/greater.hpp>
#ضمن <openvino/op/hard_sigmoid.hpp>
#ضمن <openvino/op/log.hpp>
#ضمن <openvino/op/multiply.hpp>
#ضمن <openvino/op/negative.hpp>
#ضمن <openvino/op/relu.hpp>
#ضمن <openvino/op/round.hpp>
#ضمن <openvino/op/sigmoid.hpp>
#ضمن <openvino/op/softplus.hpp>
#ضمن <openvino/op/subtract.hpp>

نطاق ov {
نطاق frontend {
نطاق ggml {
نطاق op {

Output_ویکټور translate_unary_gelu(ثابت غوټه_Context & context) {
    عدد_inputs_check(context, 1, 1);
    خپلکاری ننوت = process_view_ننوت_new(context, 0);
    خپلکاری res = std::make_shared<ov::op::v7::Gelu>(ننوت, ov::op::Gelu_Approximation_ډېرپېښ::TANH);
    أرجع rename_outputs_with_suffix({res}, context.get_name());
}

Output_ویکټور translate_unary_gelu_erf(ثابت غوټه_Context & context) {
    عدد_inputs_check(context, 1, 1);
    خپلکاری ننوت = process_view_ننوت_new(context, 0);
    خپلکاری res = std::make_shared<ov::op::v7::Gelu>(ننوت, ov::op::Gelu_Approximation_ډېرپېښ::ERF);
    أرجع rename_outputs_with_suffix({res}, context.get_name());
}

Output_ویکټور translate_unary_gelu_چټک(ثابت غوټه_Context & context) {
    عدد_inputs_check(context, 1, 1);
    خپلکاری ننوت = process_view_ننوت_new(context, 0);
    خپلکاری scale = ov::op::v0::Constant::create(ننوت.get_element_type(), ov::Shape{}, {1.702f});
    خپلکاری mul = std::make_shared<ov::op::v1::Multiply>(ننوت, scale);
    خپلکاری sig = std::make_shared<ov::op::v0::Sigmoid>(mul);
    خپلکاری res = std::make_shared<ov::op::v1::Multiply>(ننوت, sig);
    أرجع rename_outputs_with_suffix({res}, context.get_name());
}

Output_ویکټور translate_unary_elu(ثابت غوټه_Context & context) {
    عدد_inputs_check(context, 1, 1);
    خپلکاری ننوت = process_view_ننوت_new(context, 0);
    خپلکاری res = std::make_shared<ov::op::v0::Elu>(ننوت, 1.0);
    أرجع rename_outputs_with_suffix({res}, context.get_name());
}

Output_ویکټور translate_unary_hardsigmoid(ثابت غوټه_Context & context) {
    عدد_inputs_check(context, 1, 1);
    // compute in f32 like the ggml reference: 1/6 is not exact in f16/bf16 (NPU cannot take the f32 path)
    خپلکاری ننوت = process_view_ننوت_new(context, 0);
    ثابت خپلکاری type = ggml_openvino_is_npu() ? ننوت.get_element_type() : ov::element::f32;
    ov::Output<ov::Node> x = ننوت;
    إذا (type != ننوت.get_element_type()) {
        x = std::make_shared<ov::op::v0::Convert>(ننوت, type);
    }
    خپلکاری alpha = ov::op::v0::Constant::create(type, ov::Shape{}, {1.0f / 6.0f});
    خپلکاری beta = ov::op::v0::Constant::create(type, ov::Shape{}, {0.5f});
    ov::Output<ov::Node> res = std::make_shared<ov::op::v0::HardSigmoid>(x, alpha, beta);
    إذا (type != ننوت.get_element_type()) {
        res = std::make_shared<ov::op::v0::Convert>(res, ننوت.get_element_type());
    }
    أرجع rename_outputs_with_suffix({res}, context.get_name());
}

Output_ویکټور translate_unary_ګام(ثابت غوټه_Context & context) {
    عدد_inputs_check(context, 1, 1);
    خپلکاری ننوت = process_view_ننوت_new(context, 0);
    خپلکاری zero = ov::op::v0::Constant::create(ننوت.get_element_type(), ov::Shape{}, {0.0f});
    خپلکاری cond = std::make_shared<ov::op::v1::Greater>(ننوت, zero);
    خپلکاری res = std::make_shared<ov::op::v0::Convert>(cond, ننوت.get_element_type());
    أرجع rename_outputs_with_suffix({res}, context.get_name());
}

Output_ویکټور translate_unary_پړاو(ثابت غوټه_Context & context) {
    عدد_inputs_check(context, 1, 1);
    خپلکاری ننوت = process_view_ننوت_new(context, 0);
    خپلکاری res = std::make_shared<ov::op::v5::پړاو>(ننوت, ov::op::v5::پړاو::پړاو_ډېرپېښ::HALF_AWAY_FROM_ZERO);
    أرجع rename_outputs_with_suffix({res}, context.get_name());
}

Output_ویکټور translate_unary_expm1(ثابت غوټه_Context & context) {
    عدد_inputs_check(context, 1, 1);
    // compute in f32 like the ggml reference: exp(x) - 1 in f16 loses the small-x digits (NPU cannot take the f32 path)
    خپلکاری ننوت = process_view_ننوت_new(context, 0);
    ثابت خپلکاری type = ggml_openvino_is_npu() ? ننوت.get_element_type() : ov::element::f32;
    ov::Output<ov::Node> x = ننوت;
    إذا (type != ننوت.get_element_type()) {
        x = std::make_shared<ov::op::v0::Convert>(ننوت, type);
    }
    خپلکاری توانیز_تابع = std::make_shared<ov::op::v0::Exp>(x);
    خپلکاری one = ov::op::v0::Constant::create(type, ov::Shape{}, {1.0f});
    ov::Output<ov::Node> res = std::make_shared<ov::op::v1::کمول>(توانیز_تابع, one);
    إذا (type != ننوت.get_element_type()) {
        res = std::make_shared<ov::op::v0::Convert>(res, ننوت.get_element_type());
    }
    أرجع rename_outputs_with_suffix({res}, context.get_name());
}

Output_ویکټور translate_unary_softplus(ثابت غوټه_Context & context) {
    عدد_inputs_check(context, 1, 1);

    إذا (ggml_openvino_getenv_int("GGML_OPENVINO_NATIVE_SOFTPLUS") != 0) {
        أرجع translate_1to1_سمون_1_ننوت<ov::op::v4::SoftPlus>(context);
    }

    خپلکاری ننوت = process_view_ننوت_new(context, 0);
    ثابت خپلکاری element_type = ننوت.get_element_type();
    خپلکاری one = ov::op::v0::Constant::create(element_type, ov::Shape{}, {1.0f});

    خپلکاری مثبت = std::make_shared<ov::op::v0::Relu>(ننوت);
    خپلکاری مطلق_قیمت = std::make_shared<ov::op::v0::Abs>(ننوت);
    خپلکاری neg_abs = std::make_shared<ov::op::v0::Negative>(مطلق_قیمت);
    خپلکاری exp_neg_abs = std::make_shared<ov::op::v0::Exp>(neg_abs);
    خپلکاری log_غړی = std::make_shared<ov::op::v0::Log>(std::make_shared<ov::op::v1::جمع_کړه>(one, exp_neg_abs));
    خپلکاری res = std::make_shared<ov::op::v1::جمع_کړه>(مثبت, log_غړی);

    أرجع rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
