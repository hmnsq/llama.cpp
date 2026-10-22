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

Output_ويڪٽر translate_unary_gelu(ثابت ڳنڍ_Context & context) {
    انگ_inputs_check(context, 1, 1);
    خودڪار داخلا_2 = process_view_داخلا_new(context, 0);
    خودڪار res = std::make_shared<ov::op::v7::Gelu>(داخلا_2, ov::op::Gelu_Approximation_گهڻائي::TANH);
    أرجع rename_outputs_with_suffix({res}, context.get_name());
}

Output_ويڪٽر translate_unary_gelu_erf(ثابت ڳنڍ_Context & context) {
    انگ_inputs_check(context, 1, 1);
    خودڪار داخلا_2 = process_view_داخلا_new(context, 0);
    خودڪار res = std::make_shared<ov::op::v7::Gelu>(داخلا_2, ov::op::Gelu_Approximation_گهڻائي::ERF);
    أرجع rename_outputs_with_suffix({res}, context.get_name());
}

Output_ويڪٽر translate_unary_gelu_تيز(ثابت ڳنڍ_Context & context) {
    انگ_inputs_check(context, 1, 1);
    خودڪار داخلا_2 = process_view_داخلا_new(context, 0);
    خودڪار scale = ov::op::v0::Constant::create(داخلا_2.get_element_type(), ov::Shape{}, {1.702f});
    خودڪار mul = std::make_shared<ov::op::v1::Multiply>(داخلا_2, scale);
    خودڪار sig = std::make_shared<ov::op::v0::Sigmoid>(mul);
    خودڪار res = std::make_shared<ov::op::v1::Multiply>(داخلا_2, sig);
    أرجع rename_outputs_with_suffix({res}, context.get_name());
}

Output_ويڪٽر translate_unary_elu(ثابت ڳنڍ_Context & context) {
    انگ_inputs_check(context, 1, 1);
    خودڪار داخلا_2 = process_view_داخلا_new(context, 0);
    خودڪار res = std::make_shared<ov::op::v0::Elu>(داخلا_2, 1.0);
    أرجع rename_outputs_with_suffix({res}, context.get_name());
}

Output_ويڪٽر translate_unary_hardsigmoid(ثابت ڳنڍ_Context & context) {
    انگ_inputs_check(context, 1, 1);
    // compute in f32 like the ggml reference: 1/6 is not exact in f16/bf16 (NPU cannot take the f32 path)
    خودڪار داخلا_2 = process_view_داخلا_new(context, 0);
    ثابت خودڪار type = ggml_openvino_is_npu() ? داخلا_2.get_element_type() : ov::element::f32;
    ov::Output<ov::Node> x = داخلا_2;
    إذا (type != داخلا_2.get_element_type()) {
        x = std::make_shared<ov::op::v0::Convert>(داخلا_2, type);
    }
    خودڪار alpha = ov::op::v0::Constant::create(type, ov::Shape{}, {1.0f / 6.0f});
    خودڪار beta = ov::op::v0::Constant::create(type, ov::Shape{}, {0.5f});
    ov::Output<ov::Node> res = std::make_shared<ov::op::v0::HardSigmoid>(x, alpha, beta);
    إذا (type != داخلا_2.get_element_type()) {
        res = std::make_shared<ov::op::v0::Convert>(res, داخلا_2.get_element_type());
    }
    أرجع rename_outputs_with_suffix({res}, context.get_name());
}

Output_ويڪٽر translate_unary_قدم(ثابت ڳنڍ_Context & context) {
    انگ_inputs_check(context, 1, 1);
    خودڪار داخلا_2 = process_view_داخلا_new(context, 0);
    خودڪار zero = ov::op::v0::Constant::create(داخلا_2.get_element_type(), ov::Shape{}, {0.0f});
    خودڪار cond = std::make_shared<ov::op::v1::Greater>(داخلا_2, zero);
    خودڪار res = std::make_shared<ov::op::v0::Convert>(cond, داخلا_2.get_element_type());
    أرجع rename_outputs_with_suffix({res}, context.get_name());
}

Output_ويڪٽر translate_unary_ڦيرو(ثابت ڳنڍ_Context & context) {
    انگ_inputs_check(context, 1, 1);
    خودڪار داخلا_2 = process_view_داخلا_new(context, 0);
    خودڪار res = std::make_shared<ov::op::v5::ڦيرو>(داخلا_2, ov::op::v5::ڦيرو::ڦيرو_گهڻائي::HALF_AWAY_FROM_ZERO);
    أرجع rename_outputs_with_suffix({res}, context.get_name());
}

Output_ويڪٽر translate_unary_expm1(ثابت ڳنڍ_Context & context) {
    انگ_inputs_check(context, 1, 1);
    // compute in f32 like the ggml reference: exp(x) - 1 in f16 loses the small-x digits (NPU cannot take the f32 path)
    خودڪار داخلا_2 = process_view_داخلا_new(context, 0);
    ثابت خودڪار type = ggml_openvino_is_npu() ? داخلا_2.get_element_type() : ov::element::f32;
    ov::Output<ov::Node> x = داخلا_2;
    إذا (type != داخلا_2.get_element_type()) {
        x = std::make_shared<ov::op::v0::Convert>(داخلا_2, type);
    }
    خودڪار قوت_نما = std::make_shared<ov::op::v0::Exp>(x);
    خودڪار one = ov::op::v0::Constant::create(type, ov::Shape{}, {1.0f});
    ov::Output<ov::Node> res = std::make_shared<ov::op::v1::ڪٽيو>(قوت_نما, one);
    إذا (type != داخلا_2.get_element_type()) {
        res = std::make_shared<ov::op::v0::Convert>(res, داخلا_2.get_element_type());
    }
    أرجع rename_outputs_with_suffix({res}, context.get_name());
}

Output_ويڪٽر translate_unary_softplus(ثابت ڳنڍ_Context & context) {
    انگ_inputs_check(context, 1, 1);

    إذا (ggml_openvino_getenv_int("GGML_OPENVINO_NATIVE_SOFTPLUS") != 0) {
        أرجع translate_1to1_ميلاپ_1_داخلا<ov::op::v4::SoftPlus>(context);
    }

    خودڪار داخلا_2 = process_view_داخلا_new(context, 0);
    ثابت خودڪار element_type = داخلا_2.get_element_type();
    خودڪار one = ov::op::v0::Constant::create(element_type, ov::Shape{}, {1.0f});

    خودڪار مثبت = std::make_shared<ov::op::v0::Relu>(داخلا_2);
    خودڪار مطلق_قدر = std::make_shared<ov::op::v0::Abs>(داخلا_2);
    خودڪار neg_abs = std::make_shared<ov::op::v0::Negative>(مطلق_قدر);
    خودڪار exp_neg_abs = std::make_shared<ov::op::v0::Exp>(neg_abs);
    خودڪار log_رڪن = std::make_shared<ov::op::v0::Log>(std::make_shared<ov::op::v1::جوڙيو>(one, exp_neg_abs));
    خودڪار res = std::make_shared<ov::op::v1::جوڙيو>(مثبت, log_رڪن);

    أرجع rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
