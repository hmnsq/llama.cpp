#અન્તર્ભાવ "../node_context.h"
#અન્તર્ભાવ "../op_table.h"
#અન્તર્ભાવ "../utils.h"
#અન્તર્ભાવ "ggml-openvino/ggml-openvino-extra.h"

#અન્તર્ભાવ <memory>
#અન્તર્ભાવ <openvino/op/abs.hpp>
#અન્તર્ભાવ <openvino/op/add.hpp>
#અન્તર્ભાવ <openvino/op/clamp.hpp>
#અન્તર્ભાવ <openvino/op/constant.hpp>
#અન્તર્ભાવ <openvino/op/convert.hpp>
#અન્તર્ભાવ <openvino/op/elu.hpp>
#અન્તર્ભાવ <openvino/op/exp.hpp>
#અન્તર્ભાવ <openvino/op/gelu.hpp>
#અન્તર્ભાવ <openvino/op/greater.hpp>
#અન્તર્ભાવ <openvino/op/hard_sigmoid.hpp>
#અન્તર્ભાવ <openvino/op/log.hpp>
#અન્તર્ભાવ <openvino/op/multiply.hpp>
#અન્તર્ભાવ <openvino/op/negative.hpp>
#અન્તર્ભાવ <openvino/op/relu.hpp>
#અન્તર્ભાવ <openvino/op/round.hpp>
#અન્તર્ભાવ <openvino/op/sigmoid.hpp>
#અન્તર્ભાવ <openvino/op/softplus.hpp>
#અન્તર્ભાવ <openvino/op/subtract.hpp>

નામાકાશ ov {
નામાકાશ frontend {
નામાકાશ ggml {
નામાકાશ op {

OutputVector translate_unary_gelu(અચલ ગાંઠ_Context & context) {
    અંક_inputs_check(context, 1, 1);
    auto આગત = process_view_આગત_new(context, 0);
    auto res = std::make_shared<ov::op::v7::Gelu>(આગત, ov::op::GeluApproximationMode::TANH);
    પાછા rename_outputs_with_suffix({res}, context.get_name());
}

OutputVector translate_unary_gelu_erf(અચલ ગાંઠ_Context & context) {
    અંક_inputs_check(context, 1, 1);
    auto આગત = process_view_આગત_new(context, 0);
    auto res = std::make_shared<ov::op::v7::Gelu>(આગત, ov::op::GeluApproximationMode::ERF);
    પાછા rename_outputs_with_suffix({res}, context.get_name());
}

OutputVector translate_unary_gelu_quick(અચલ ગાંઠ_Context & context) {
    અંક_inputs_check(context, 1, 1);
    auto આગત = process_view_આગત_new(context, 0);
    auto scale = ov::op::v0::Constant::create(આગત.get_element_type(), ov::Shape{}, {1.702f});
    auto mul = std::make_shared<ov::op::v1::Multiply>(આગત, scale);
    auto sig = std::make_shared<ov::op::v0::Sigmoid>(mul);
    auto res = std::make_shared<ov::op::v1::Multiply>(આગત, sig);
    પાછા rename_outputs_with_suffix({res}, context.get_name());
}

OutputVector translate_unary_elu(અચલ ગાંઠ_Context & context) {
    અંક_inputs_check(context, 1, 1);
    auto આગત = process_view_આગત_new(context, 0);
    auto res = std::make_shared<ov::op::v0::Elu>(આગત, 1.0);
    પાછા rename_outputs_with_suffix({res}, context.get_name());
}

OutputVector translate_unary_hardsigmoid(અચલ ગાંઠ_Context & context) {
    અંક_inputs_check(context, 1, 1);
    // compute in f32 like the ggml reference: 1/6 is not exact in f16/bf16 (NPU cannot take the f32 path)
    auto આગત = process_view_આગત_new(context, 0);
    અચલ auto type = ggml_openvino_is_npu() ? આગત.get_element_type() : ov::element::f32;
    ov::Output<ov::Node> x = આગત;
    જો (type != આગત.get_element_type()) {
        x = std::make_shared<ov::op::v0::Convert>(આગત, type);
    }
    auto alpha = ov::op::v0::Constant::create(type, ov::Shape{}, {1.0f / 6.0f});
    auto beta = ov::op::v0::Constant::create(type, ov::Shape{}, {0.5f});
    ov::Output<ov::Node> res = std::make_shared<ov::op::v0::HardSigmoid>(x, alpha, beta);
    જો (type != આગત.get_element_type()) {
        res = std::make_shared<ov::op::v0::Convert>(res, આગત.get_element_type());
    }
    પાછા rename_outputs_with_suffix({res}, context.get_name());
}

OutputVector translate_unary_step(અચલ ગાંઠ_Context & context) {
    અંક_inputs_check(context, 1, 1);
    auto આગત = process_view_આગત_new(context, 0);
    auto zero = ov::op::v0::Constant::create(આગત.get_element_type(), ov::Shape{}, {0.0f});
    auto cond = std::make_shared<ov::op::v1::Greater>(આગત, zero);
    auto res = std::make_shared<ov::op::v0::Convert>(cond, આગત.get_element_type());
    પાછા rename_outputs_with_suffix({res}, context.get_name());
}

OutputVector translate_unary_round(અચલ ગાંઠ_Context & context) {
    અંક_inputs_check(context, 1, 1);
    auto આગત = process_view_આગત_new(context, 0);
    auto res = std::make_shared<ov::op::v5::Round>(આગત, ov::op::v5::Round::RoundMode::HALF_AWAY_FROM_ZERO);
    પાછા rename_outputs_with_suffix({res}, context.get_name());
}

OutputVector translate_unary_expm1(અચલ ગાંઠ_Context & context) {
    અંક_inputs_check(context, 1, 1);
    // compute in f32 like the ggml reference: exp(x) - 1 in f16 loses the small-x digits (NPU cannot take the f32 path)
    auto આગત = process_view_આગત_new(context, 0);
    અચલ auto type = ggml_openvino_is_npu() ? આગત.get_element_type() : ov::element::f32;
    ov::Output<ov::Node> x = આગત;
    જો (type != આગત.get_element_type()) {
        x = std::make_shared<ov::op::v0::Convert>(આગત, type);
    }
    auto exp = std::make_shared<ov::op::v0::Exp>(x);
    auto one = ov::op::v0::Constant::create(type, ov::Shape{}, {1.0f});
    ov::Output<ov::Node> res = std::make_shared<ov::op::v1::Subtract>(exp, one);
    જો (type != આગત.get_element_type()) {
        res = std::make_shared<ov::op::v0::Convert>(res, આગત.get_element_type());
    }
    પાછા rename_outputs_with_suffix({res}, context.get_name());
}

OutputVector translate_unary_softplus(અચલ ગાંઠ_Context & context) {
    અંક_inputs_check(context, 1, 1);

    જો (ggml_openvino_getenv_int("GGML_OPENVINO_NATIVE_SOFTPLUS") != 0) {
        પાછા translate_1to1_match_1_આગત<ov::op::v4::SoftPlus>(context);
    }

    auto આગત = process_view_આગત_new(context, 0);
    અચલ auto element_type = આગત.get_element_type();
    auto one = ov::op::v0::Constant::create(element_type, ov::Shape{}, {1.0f});

    auto positive = std::make_shared<ov::op::v0::Relu>(આગત);
    auto abs = std::make_shared<ov::op::v0::Abs>(આગત);
    auto neg_abs = std::make_shared<ov::op::v0::Negative>(abs);
    auto exp_neg_abs = std::make_shared<ov::op::v0::Exp>(neg_abs);
    auto log_term = std::make_shared<ov::op::v0::Log>(std::make_shared<ov::op::v1::Add>(one, exp_neg_abs));
    auto res = std::make_shared<ov::op::v1::Add>(positive, log_term);

    પાછા rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
