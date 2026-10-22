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

Output_وکتور translate_unary_gelu(ثابت گره_Context & context) {
    رقم_inputs_check(context, 1, 1);
    خودکار درون_داد = process_view_درون_داد_new(context, 0);
    خودکار res = std::make_shared<ov::op::v7::Gelu>(درون_داد, ov::op::Gelu_Approximation_نما::TANH);
    أرجع rename_outputs_with_suffix({res}, context.get_name());
}

Output_وکتور translate_unary_gelu_erf(ثابت گره_Context & context) {
    رقم_inputs_check(context, 1, 1);
    خودکار درون_داد = process_view_درون_داد_new(context, 0);
    خودکار res = std::make_shared<ov::op::v7::Gelu>(درون_داد, ov::op::Gelu_Approximation_نما::ERF);
    أرجع rename_outputs_with_suffix({res}, context.get_name());
}

Output_وکتور translate_unary_gelu_سریع(ثابت گره_Context & context) {
    رقم_inputs_check(context, 1, 1);
    خودکار درون_داد = process_view_درون_داد_new(context, 0);
    خودکار scale = ov::op::v0::Constant::create(درون_داد.get_element_type(), ov::Shape{}, {1.702f});
    خودکار mul = std::make_shared<ov::op::v1::Multiply>(درون_داد, scale);
    خودکار sig = std::make_shared<ov::op::v0::Sigmoid>(mul);
    خودکار res = std::make_shared<ov::op::v1::Multiply>(درون_داد, sig);
    أرجع rename_outputs_with_suffix({res}, context.get_name());
}

Output_وکتور translate_unary_elu(ثابت گره_Context & context) {
    رقم_inputs_check(context, 1, 1);
    خودکار درون_داد = process_view_درون_داد_new(context, 0);
    خودکار res = std::make_shared<ov::op::v0::Elu>(درون_داد, 1.0);
    أرجع rename_outputs_with_suffix({res}, context.get_name());
}

Output_وکتور translate_unary_hardsigmoid(ثابت گره_Context & context) {
    رقم_inputs_check(context, 1, 1);
    // compute in f32 like the ggml reference: 1/6 is not exact in f16/bf16 (NPU cannot take the f32 path)
    خودکار درون_داد = process_view_درون_داد_new(context, 0);
    ثابت خودکار type = ggml_openvino_is_npu() ? درون_داد.get_element_type() : ov::element::f32;
    ov::Output<ov::Node> x = درون_داد;
    إذا (type != درون_داد.get_element_type()) {
        x = std::make_shared<ov::op::v0::Convert>(درون_داد, type);
    }
    خودکار alpha = ov::op::v0::Constant::create(type, ov::Shape{}, {1.0f / 6.0f});
    خودکار beta = ov::op::v0::Constant::create(type, ov::Shape{}, {0.5f});
    ov::Output<ov::Node> res = std::make_shared<ov::op::v0::HardSigmoid>(x, alpha, beta);
    إذا (type != درون_داد.get_element_type()) {
        res = std::make_shared<ov::op::v0::Convert>(res, درون_داد.get_element_type());
    }
    أرجع rename_outputs_with_suffix({res}, context.get_name());
}

Output_وکتور translate_unary_گام(ثابت گره_Context & context) {
    رقم_inputs_check(context, 1, 1);
    خودکار درون_داد = process_view_درون_داد_new(context, 0);
    خودکار zero = ov::op::v0::Constant::create(درون_داد.get_element_type(), ov::Shape{}, {0.0f});
    خودکار cond = std::make_shared<ov::op::v1::Greater>(درون_داد, zero);
    خودکار res = std::make_shared<ov::op::v0::Convert>(cond, درون_داد.get_element_type());
    أرجع rename_outputs_with_suffix({res}, context.get_name());
}

Output_وکتور translate_unary_دور(ثابت گره_Context & context) {
    رقم_inputs_check(context, 1, 1);
    خودکار درون_داد = process_view_درون_داد_new(context, 0);
    خودکار res = std::make_shared<ov::op::v5::دور>(درون_داد, ov::op::v5::دور::دور_نما::HALF_AWAY_FROM_ZERO);
    أرجع rename_outputs_with_suffix({res}, context.get_name());
}

Output_وکتور translate_unary_expm1(ثابت گره_Context & context) {
    رقم_inputs_check(context, 1, 1);
    // compute in f32 like the ggml reference: exp(x) - 1 in f16 loses the small-x digits (NPU cannot take the f32 path)
    خودکار درون_داد = process_view_درون_داد_new(context, 0);
    ثابت خودکار type = ggml_openvino_is_npu() ? درون_داد.get_element_type() : ov::element::f32;
    ov::Output<ov::Node> x = درون_داد;
    إذا (type != درون_داد.get_element_type()) {
        x = std::make_shared<ov::op::v0::Convert>(درون_داد, type);
    }
    خودکار نمایی_تابع = std::make_shared<ov::op::v0::Exp>(x);
    خودکار one = ov::op::v0::Constant::create(type, ov::Shape{}, {1.0f});
    ov::Output<ov::Node> res = std::make_shared<ov::op::v1::تفریق>(نمایی_تابع, one);
    إذا (type != درون_داد.get_element_type()) {
        res = std::make_shared<ov::op::v0::Convert>(res, درون_داد.get_element_type());
    }
    أرجع rename_outputs_with_suffix({res}, context.get_name());
}

Output_وکتور translate_unary_softplus(ثابت گره_Context & context) {
    رقم_inputs_check(context, 1, 1);

    إذا (ggml_openvino_getenv_int("GGML_OPENVINO_NATIVE_SOFTPLUS") != 0) {
        أرجع translate_1to1_تطابق_1_درون_داد<ov::op::v4::SoftPlus>(context);
    }

    خودکار درون_داد = process_view_درون_داد_new(context, 0);
    ثابت خودکار element_type = درون_داد.get_element_type();
    خودکار one = ov::op::v0::Constant::create(element_type, ov::Shape{}, {1.0f});

    خودکار مثبت = std::make_shared<ov::op::v0::Relu>(درون_داد);
    خودکار قدر_مطلق = std::make_shared<ov::op::v0::Abs>(درون_داد);
    خودکار neg_abs = std::make_shared<ov::op::v0::Negative>(قدر_مطلق);
    خودکار exp_neg_abs = std::make_shared<ov::op::v0::Exp>(neg_abs);
    خودکار log_جمله = std::make_shared<ov::op::v0::Log>(std::make_shared<ov::op::v1::جمع_کن>(one, exp_neg_abs));
    خودکار res = std::make_shared<ov::op::v1::جمع_کن>(مثبت, log_جمله);

    أرجع rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
