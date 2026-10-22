#ଅନ୍ତର୍ଭାଵ "../node_context.h"
#ଅନ୍ତର୍ଭାଵ "../op_table.h"
#ଅନ୍ତର୍ଭାଵ "../utils.h"
#ଅନ୍ତର୍ଭାଵ "ggml-openvino/ggml-openvino-extra.h"

#ଅନ୍ତର୍ଭାଵ <memory>
#ଅନ୍ତର୍ଭାଵ <openvino/op/abs.hpp>
#ଅନ୍ତର୍ଭାଵ <openvino/op/add.hpp>
#ଅନ୍ତର୍ଭାଵ <openvino/op/clamp.hpp>
#ଅନ୍ତର୍ଭାଵ <openvino/op/constant.hpp>
#ଅନ୍ତର୍ଭାଵ <openvino/op/convert.hpp>
#ଅନ୍ତର୍ଭାଵ <openvino/op/elu.hpp>
#ଅନ୍ତର୍ଭାଵ <openvino/op/exp.hpp>
#ଅନ୍ତର୍ଭାଵ <openvino/op/gelu.hpp>
#ଅନ୍ତର୍ଭାଵ <openvino/op/greater.hpp>
#ଅନ୍ତର୍ଭାଵ <openvino/op/hard_sigmoid.hpp>
#ଅନ୍ତର୍ଭାଵ <openvino/op/log.hpp>
#ଅନ୍ତର୍ଭାଵ <openvino/op/multiply.hpp>
#ଅନ୍ତର୍ଭାଵ <openvino/op/negative.hpp>
#ଅନ୍ତର୍ଭାଵ <openvino/op/relu.hpp>
#ଅନ୍ତର୍ଭାଵ <openvino/op/round.hpp>
#ଅନ୍ତର୍ଭାଵ <openvino/op/sigmoid.hpp>
#ଅନ୍ତର୍ଭାଵ <openvino/op/softplus.hpp>
#ଅନ୍ତର୍ଭାଵ <openvino/op/subtract.hpp>

ନାମାକାଶ ov {
ନାମାକାଶ frontend {
ନାମାକାଶ ggml {
ନାମାକାଶ op {

OutputVector translate_unary_gelu(ଅଚଳ ଗଣ୍ଠି_Context & context) {
    ଅଙ୍କ_inputs_check(context, 1, 1);
    auto ନିବେଶ = process_view_ନିବେଶ_new(context, 0);
    auto res = std::make_shared<ov::op::v7::Gelu>(ନିବେଶ, ov::op::GeluApproximationMode::TANH);
    ଫେରନ୍ତୁ rename_outputs_with_suffix({res}, context.get_name());
}

OutputVector translate_unary_gelu_erf(ଅଚଳ ଗଣ୍ଠି_Context & context) {
    ଅଙ୍କ_inputs_check(context, 1, 1);
    auto ନିବେଶ = process_view_ନିବେଶ_new(context, 0);
    auto res = std::make_shared<ov::op::v7::Gelu>(ନିବେଶ, ov::op::GeluApproximationMode::ERF);
    ଫେରନ୍ତୁ rename_outputs_with_suffix({res}, context.get_name());
}

OutputVector translate_unary_gelu_quick(ଅଚଳ ଗଣ୍ଠି_Context & context) {
    ଅଙ୍କ_inputs_check(context, 1, 1);
    auto ନିବେଶ = process_view_ନିବେଶ_new(context, 0);
    auto scale = ov::op::v0::Constant::create(ନିବେଶ.get_element_type(), ov::Shape{}, {1.702f});
    auto mul = std::make_shared<ov::op::v1::Multiply>(ନିବେଶ, scale);
    auto sig = std::make_shared<ov::op::v0::Sigmoid>(mul);
    auto res = std::make_shared<ov::op::v1::Multiply>(ନିବେଶ, sig);
    ଫେରନ୍ତୁ rename_outputs_with_suffix({res}, context.get_name());
}

OutputVector translate_unary_elu(ଅଚଳ ଗଣ୍ଠି_Context & context) {
    ଅଙ୍କ_inputs_check(context, 1, 1);
    auto ନିବେଶ = process_view_ନିବେଶ_new(context, 0);
    auto res = std::make_shared<ov::op::v0::Elu>(ନିବେଶ, 1.0);
    ଫେରନ୍ତୁ rename_outputs_with_suffix({res}, context.get_name());
}

OutputVector translate_unary_hardsigmoid(ଅଚଳ ଗଣ୍ଠି_Context & context) {
    ଅଙ୍କ_inputs_check(context, 1, 1);
    // compute in f32 like the ggml reference: 1/6 is not exact in f16/bf16 (NPU cannot take the f32 path)
    auto ନିବେଶ = process_view_ନିବେଶ_new(context, 0);
    ଅଚଳ auto type = ggml_openvino_is_npu() ? ନିବେଶ.get_element_type() : ov::element::f32;
    ov::Output<ov::Node> x = ନିବେଶ;
    ଯଦି (type != ନିବେଶ.get_element_type()) {
        x = std::make_shared<ov::op::v0::Convert>(ନିବେଶ, type);
    }
    auto alpha = ov::op::v0::Constant::create(type, ov::Shape{}, {1.0f / 6.0f});
    auto beta = ov::op::v0::Constant::create(type, ov::Shape{}, {0.5f});
    ov::Output<ov::Node> res = std::make_shared<ov::op::v0::HardSigmoid>(x, alpha, beta);
    ଯଦି (type != ନିବେଶ.get_element_type()) {
        res = std::make_shared<ov::op::v0::Convert>(res, ନିବେଶ.get_element_type());
    }
    ଫେରନ୍ତୁ rename_outputs_with_suffix({res}, context.get_name());
}

OutputVector translate_unary_step(ଅଚଳ ଗଣ୍ଠି_Context & context) {
    ଅଙ୍କ_inputs_check(context, 1, 1);
    auto ନିବେଶ = process_view_ନିବେଶ_new(context, 0);
    auto zero = ov::op::v0::Constant::create(ନିବେଶ.get_element_type(), ov::Shape{}, {0.0f});
    auto cond = std::make_shared<ov::op::v1::Greater>(ନିବେଶ, zero);
    auto res = std::make_shared<ov::op::v0::Convert>(cond, ନିବେଶ.get_element_type());
    ଫେରନ୍ତୁ rename_outputs_with_suffix({res}, context.get_name());
}

OutputVector translate_unary_round(ଅଚଳ ଗଣ୍ଠି_Context & context) {
    ଅଙ୍କ_inputs_check(context, 1, 1);
    auto ନିବେଶ = process_view_ନିବେଶ_new(context, 0);
    auto res = std::make_shared<ov::op::v5::Round>(ନିବେଶ, ov::op::v5::Round::RoundMode::HALF_AWAY_FROM_ZERO);
    ଫେରନ୍ତୁ rename_outputs_with_suffix({res}, context.get_name());
}

OutputVector translate_unary_expm1(ଅଚଳ ଗଣ୍ଠି_Context & context) {
    ଅଙ୍କ_inputs_check(context, 1, 1);
    // compute in f32 like the ggml reference: exp(x) - 1 in f16 loses the small-x digits (NPU cannot take the f32 path)
    auto ନିବେଶ = process_view_ନିବେଶ_new(context, 0);
    ଅଚଳ auto type = ggml_openvino_is_npu() ? ନିବେଶ.get_element_type() : ov::element::f32;
    ov::Output<ov::Node> x = ନିବେଶ;
    ଯଦି (type != ନିବେଶ.get_element_type()) {
        x = std::make_shared<ov::op::v0::Convert>(ନିବେଶ, type);
    }
    auto exp = std::make_shared<ov::op::v0::Exp>(x);
    auto one = ov::op::v0::Constant::create(type, ov::Shape{}, {1.0f});
    ov::Output<ov::Node> res = std::make_shared<ov::op::v1::Subtract>(exp, one);
    ଯଦି (type != ନିବେଶ.get_element_type()) {
        res = std::make_shared<ov::op::v0::Convert>(res, ନିବେଶ.get_element_type());
    }
    ଫେରନ୍ତୁ rename_outputs_with_suffix({res}, context.get_name());
}

OutputVector translate_unary_softplus(ଅଚଳ ଗଣ୍ଠି_Context & context) {
    ଅଙ୍କ_inputs_check(context, 1, 1);

    ଯଦି (ggml_openvino_getenv_int("GGML_OPENVINO_NATIVE_SOFTPLUS") != 0) {
        ଫେରନ୍ତୁ translate_1to1_match_1_ନିବେଶ<ov::op::v4::SoftPlus>(context);
    }

    auto ନିବେଶ = process_view_ନିବେଶ_new(context, 0);
    ଅଚଳ auto element_type = ନିବେଶ.get_element_type();
    auto one = ov::op::v0::Constant::create(element_type, ov::Shape{}, {1.0f});

    auto positive = std::make_shared<ov::op::v0::Relu>(ନିବେଶ);
    auto abs = std::make_shared<ov::op::v0::Abs>(ନିବେଶ);
    auto neg_abs = std::make_shared<ov::op::v0::Negative>(abs);
    auto exp_neg_abs = std::make_shared<ov::op::v0::Exp>(neg_abs);
    auto log_term = std::make_shared<ov::op::v0::Log>(std::make_shared<ov::op::v1::Add>(one, exp_neg_abs));
    auto res = std::make_shared<ov::op::v1::Add>(positive, log_term);

    ଫେରନ୍ତୁ rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
