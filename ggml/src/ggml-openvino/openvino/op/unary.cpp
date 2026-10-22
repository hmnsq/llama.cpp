#شامل "../node_context.h"
#شامل "../op_table.h"
#شامل "../utils.h"
#شامل "ggml-openvino/ggml-openvino-extra.h"

#شامل <memory>
#شامل <openvino/op/abs.hpp>
#شامل <openvino/op/add.hpp>
#شامل <openvino/op/clamp.hpp>
#شامل <openvino/op/constant.hpp>
#شامل <openvino/op/convert.hpp>
#شامل <openvino/op/elu.hpp>
#شامل <openvino/op/exp.hpp>
#شامل <openvino/op/gelu.hpp>
#شامل <openvino/op/greater.hpp>
#شامل <openvino/op/hard_sigmoid.hpp>
#شامل <openvino/op/log.hpp>
#شامل <openvino/op/multiply.hpp>
#شامل <openvino/op/negative.hpp>
#شامل <openvino/op/relu.hpp>
#شامل <openvino/op/round.hpp>
#شامل <openvino/op/sigmoid.hpp>
#شامل <openvino/op/softplus.hpp>
#شامل <openvino/op/subtract.hpp>

فضاینام ov {
فضاینام frontend {
فضاینام ggml {
فضاینام op {

OutputVector translate_unary_gelu(ثابت گره_Context & context) {
    رقم_inputs_check(context, 1, 1);
    auto درون_داد = process_view_درون_داد_new(context, 0);
    auto res = std::make_shared<ov::op::v7::Gelu>(درون_داد, ov::op::GeluApproximationMode::TANH);
    بازگشت rename_outputs_with_suffix({res}, context.get_name());
}

OutputVector translate_unary_gelu_erf(ثابت گره_Context & context) {
    رقم_inputs_check(context, 1, 1);
    auto درون_داد = process_view_درون_داد_new(context, 0);
    auto res = std::make_shared<ov::op::v7::Gelu>(درون_داد, ov::op::GeluApproximationMode::ERF);
    بازگشت rename_outputs_with_suffix({res}, context.get_name());
}

OutputVector translate_unary_gelu_quick(ثابت گره_Context & context) {
    رقم_inputs_check(context, 1, 1);
    auto درون_داد = process_view_درون_داد_new(context, 0);
    auto scale = ov::op::v0::Constant::create(درون_داد.get_element_type(), ov::Shape{}, {1.702f});
    auto mul = std::make_shared<ov::op::v1::Multiply>(درون_داد, scale);
    auto sig = std::make_shared<ov::op::v0::Sigmoid>(mul);
    auto res = std::make_shared<ov::op::v1::Multiply>(درون_داد, sig);
    بازگشت rename_outputs_with_suffix({res}, context.get_name());
}

OutputVector translate_unary_elu(ثابت گره_Context & context) {
    رقم_inputs_check(context, 1, 1);
    auto درون_داد = process_view_درون_داد_new(context, 0);
    auto res = std::make_shared<ov::op::v0::Elu>(درون_داد, 1.0);
    بازگشت rename_outputs_with_suffix({res}, context.get_name());
}

OutputVector translate_unary_hardsigmoid(ثابت گره_Context & context) {
    رقم_inputs_check(context, 1, 1);
    // compute in f32 like the ggml reference: 1/6 is not exact in f16/bf16 (NPU cannot take the f32 path)
    auto درون_داد = process_view_درون_داد_new(context, 0);
    ثابت auto type = ggml_openvino_is_npu() ? درون_داد.get_element_type() : ov::element::f32;
    ov::Output<ov::Node> x = درون_داد;
    اگر (type != درون_داد.get_element_type()) {
        x = std::make_shared<ov::op::v0::Convert>(درون_داد, type);
    }
    auto alpha = ov::op::v0::Constant::create(type, ov::Shape{}, {1.0f / 6.0f});
    auto beta = ov::op::v0::Constant::create(type, ov::Shape{}, {0.5f});
    ov::Output<ov::Node> res = std::make_shared<ov::op::v0::HardSigmoid>(x, alpha, beta);
    اگر (type != درون_داد.get_element_type()) {
        res = std::make_shared<ov::op::v0::Convert>(res, درون_داد.get_element_type());
    }
    بازگشت rename_outputs_with_suffix({res}, context.get_name());
}

OutputVector translate_unary_step(ثابت گره_Context & context) {
    رقم_inputs_check(context, 1, 1);
    auto درون_داد = process_view_درون_داد_new(context, 0);
    auto zero = ov::op::v0::Constant::create(درون_داد.get_element_type(), ov::Shape{}, {0.0f});
    auto cond = std::make_shared<ov::op::v1::Greater>(درون_داد, zero);
    auto res = std::make_shared<ov::op::v0::Convert>(cond, درون_داد.get_element_type());
    بازگشت rename_outputs_with_suffix({res}, context.get_name());
}

OutputVector translate_unary_round(ثابت گره_Context & context) {
    رقم_inputs_check(context, 1, 1);
    auto درون_داد = process_view_درون_داد_new(context, 0);
    auto res = std::make_shared<ov::op::v5::Round>(درون_داد, ov::op::v5::Round::RoundMode::HALF_AWAY_FROM_ZERO);
    بازگشت rename_outputs_with_suffix({res}, context.get_name());
}

OutputVector translate_unary_expm1(ثابت گره_Context & context) {
    رقم_inputs_check(context, 1, 1);
    // compute in f32 like the ggml reference: exp(x) - 1 in f16 loses the small-x digits (NPU cannot take the f32 path)
    auto درون_داد = process_view_درون_داد_new(context, 0);
    ثابت auto type = ggml_openvino_is_npu() ? درون_داد.get_element_type() : ov::element::f32;
    ov::Output<ov::Node> x = درون_داد;
    اگر (type != درون_داد.get_element_type()) {
        x = std::make_shared<ov::op::v0::Convert>(درون_داد, type);
    }
    auto exp = std::make_shared<ov::op::v0::Exp>(x);
    auto one = ov::op::v0::Constant::create(type, ov::Shape{}, {1.0f});
    ov::Output<ov::Node> res = std::make_shared<ov::op::v1::Subtract>(exp, one);
    اگر (type != درون_داد.get_element_type()) {
        res = std::make_shared<ov::op::v0::Convert>(res, درون_داد.get_element_type());
    }
    بازگشت rename_outputs_with_suffix({res}, context.get_name());
}

OutputVector translate_unary_softplus(ثابت گره_Context & context) {
    رقم_inputs_check(context, 1, 1);

    اگر (ggml_openvino_getenv_int("GGML_OPENVINO_NATIVE_SOFTPLUS") != 0) {
        بازگشت translate_1to1_match_1_درون_داد<ov::op::v4::SoftPlus>(context);
    }

    auto درون_داد = process_view_درون_داد_new(context, 0);
    ثابت auto element_type = درون_داد.get_element_type();
    auto one = ov::op::v0::Constant::create(element_type, ov::Shape{}, {1.0f});

    auto positive = std::make_shared<ov::op::v0::Relu>(درون_داد);
    auto abs = std::make_shared<ov::op::v0::Abs>(درون_داد);
    auto neg_abs = std::make_shared<ov::op::v0::Negative>(abs);
    auto exp_neg_abs = std::make_shared<ov::op::v0::Exp>(neg_abs);
    auto log_term = std::make_shared<ov::op::v0::Log>(std::make_shared<ov::op::v1::Add>(one, exp_neg_abs));
    auto res = std::make_shared<ov::op::v1::Add>(positive, log_term);

    بازگشت rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
