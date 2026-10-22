#অন্তর্ভাব "../node_context.h"
#অন্তর্ভাব "../op_table.h"
#অন্তর্ভাব "../utils.h"
#অন্তর্ভাব "ggml-openvino/ggml-openvino-extra.h"

#অন্তর্ভাব <memory>
#অন্তর্ভাব <openvino/op/abs.hpp>
#অন্তর্ভাব <openvino/op/add.hpp>
#অন্তর্ভাব <openvino/op/clamp.hpp>
#অন্তর্ভাব <openvino/op/constant.hpp>
#অন্তর্ভাব <openvino/op/convert.hpp>
#অন্তর্ভাব <openvino/op/elu.hpp>
#অন্তর্ভাব <openvino/op/exp.hpp>
#অন্তর্ভাব <openvino/op/gelu.hpp>
#অন্তর্ভাব <openvino/op/greater.hpp>
#অন্তর্ভাব <openvino/op/hard_sigmoid.hpp>
#অন্তর্ভাব <openvino/op/log.hpp>
#অন্তর্ভাব <openvino/op/multiply.hpp>
#অন্তর্ভাব <openvino/op/negative.hpp>
#অন্তর্ভাব <openvino/op/relu.hpp>
#অন্তর্ভাব <openvino/op/round.hpp>
#অন্তর্ভাব <openvino/op/sigmoid.hpp>
#অন্তর্ভাব <openvino/op/softplus.hpp>
#অন্তর্ভাব <openvino/op/subtract.hpp>

নামাকাশ ov {
নামাকাশ frontend {
নামাকাশ ggml {
নামাকাশ op {

OutputVector translate_unary_gelu(ধ্রুবক গ্রন্থি_Context & context) {
    অঙ্ক_inputs_check(context, 1, 1);
    auto প্রবেশ = process_view_প্রবেশ_new(context, 0);
    auto res = std::make_shared<ov::op::v7::Gelu>(প্রবেশ, ov::op::GeluApproximationMode::TANH);
    ফেরত rename_outputs_with_suffix({res}, context.get_name());
}

OutputVector translate_unary_gelu_erf(ধ্রুবক গ্রন্থি_Context & context) {
    অঙ্ক_inputs_check(context, 1, 1);
    auto প্রবেশ = process_view_প্রবেশ_new(context, 0);
    auto res = std::make_shared<ov::op::v7::Gelu>(প্রবেশ, ov::op::GeluApproximationMode::ERF);
    ফেরত rename_outputs_with_suffix({res}, context.get_name());
}

OutputVector translate_unary_gelu_quick(ধ্রুবক গ্রন্থি_Context & context) {
    অঙ্ক_inputs_check(context, 1, 1);
    auto প্রবেশ = process_view_প্রবেশ_new(context, 0);
    auto scale = ov::op::v0::Constant::create(প্রবেশ.get_element_type(), ov::Shape{}, {1.702f});
    auto mul = std::make_shared<ov::op::v1::Multiply>(প্রবেশ, scale);
    auto sig = std::make_shared<ov::op::v0::Sigmoid>(mul);
    auto res = std::make_shared<ov::op::v1::Multiply>(প্রবেশ, sig);
    ফেরত rename_outputs_with_suffix({res}, context.get_name());
}

OutputVector translate_unary_elu(ধ্রুবক গ্রন্থি_Context & context) {
    অঙ্ক_inputs_check(context, 1, 1);
    auto প্রবেশ = process_view_প্রবেশ_new(context, 0);
    auto res = std::make_shared<ov::op::v0::Elu>(প্রবেশ, 1.0);
    ফেরত rename_outputs_with_suffix({res}, context.get_name());
}

OutputVector translate_unary_hardsigmoid(ধ্রুবক গ্রন্থি_Context & context) {
    অঙ্ক_inputs_check(context, 1, 1);
    // compute in f32 like the ggml reference: 1/6 is not exact in f16/bf16 (NPU cannot take the f32 path)
    auto প্রবেশ = process_view_প্রবেশ_new(context, 0);
    ধ্রুবক auto type = ggml_openvino_is_npu() ? প্রবেশ.get_element_type() : ov::element::f32;
    ov::Output<ov::Node> x = প্রবেশ;
    যদি (type != প্রবেশ.get_element_type()) {
        x = std::make_shared<ov::op::v0::Convert>(প্রবেশ, type);
    }
    auto alpha = ov::op::v0::Constant::create(type, ov::Shape{}, {1.0f / 6.0f});
    auto beta = ov::op::v0::Constant::create(type, ov::Shape{}, {0.5f});
    ov::Output<ov::Node> res = std::make_shared<ov::op::v0::HardSigmoid>(x, alpha, beta);
    যদি (type != প্রবেশ.get_element_type()) {
        res = std::make_shared<ov::op::v0::Convert>(res, প্রবেশ.get_element_type());
    }
    ফেরত rename_outputs_with_suffix({res}, context.get_name());
}

OutputVector translate_unary_step(ধ্রুবক গ্রন্থি_Context & context) {
    অঙ্ক_inputs_check(context, 1, 1);
    auto প্রবেশ = process_view_প্রবেশ_new(context, 0);
    auto zero = ov::op::v0::Constant::create(প্রবেশ.get_element_type(), ov::Shape{}, {0.0f});
    auto cond = std::make_shared<ov::op::v1::Greater>(প্রবেশ, zero);
    auto res = std::make_shared<ov::op::v0::Convert>(cond, প্রবেশ.get_element_type());
    ফেরত rename_outputs_with_suffix({res}, context.get_name());
}

OutputVector translate_unary_round(ধ্রুবক গ্রন্থি_Context & context) {
    অঙ্ক_inputs_check(context, 1, 1);
    auto প্রবেশ = process_view_প্রবেশ_new(context, 0);
    auto res = std::make_shared<ov::op::v5::Round>(প্রবেশ, ov::op::v5::Round::RoundMode::HALF_AWAY_FROM_ZERO);
    ফেরত rename_outputs_with_suffix({res}, context.get_name());
}

OutputVector translate_unary_expm1(ধ্রুবক গ্রন্থি_Context & context) {
    অঙ্ক_inputs_check(context, 1, 1);
    // compute in f32 like the ggml reference: exp(x) - 1 in f16 loses the small-x digits (NPU cannot take the f32 path)
    auto প্রবেশ = process_view_প্রবেশ_new(context, 0);
    ধ্রুবক auto type = ggml_openvino_is_npu() ? প্রবেশ.get_element_type() : ov::element::f32;
    ov::Output<ov::Node> x = প্রবেশ;
    যদি (type != প্রবেশ.get_element_type()) {
        x = std::make_shared<ov::op::v0::Convert>(প্রবেশ, type);
    }
    auto exp = std::make_shared<ov::op::v0::Exp>(x);
    auto one = ov::op::v0::Constant::create(type, ov::Shape{}, {1.0f});
    ov::Output<ov::Node> res = std::make_shared<ov::op::v1::Subtract>(exp, one);
    যদি (type != প্রবেশ.get_element_type()) {
        res = std::make_shared<ov::op::v0::Convert>(res, প্রবেশ.get_element_type());
    }
    ফেরত rename_outputs_with_suffix({res}, context.get_name());
}

OutputVector translate_unary_softplus(ধ্রুবক গ্রন্থি_Context & context) {
    অঙ্ক_inputs_check(context, 1, 1);

    যদি (ggml_openvino_getenv_int("GGML_OPENVINO_NATIVE_SOFTPLUS") != 0) {
        ফেরত translate_1to1_match_1_প্রবেশ<ov::op::v4::SoftPlus>(context);
    }

    auto প্রবেশ = process_view_প্রবেশ_new(context, 0);
    ধ্রুবক auto element_type = প্রবেশ.get_element_type();
    auto one = ov::op::v0::Constant::create(element_type, ov::Shape{}, {1.0f});

    auto positive = std::make_shared<ov::op::v0::Relu>(প্রবেশ);
    auto abs = std::make_shared<ov::op::v0::Abs>(প্রবেশ);
    auto neg_abs = std::make_shared<ov::op::v0::Negative>(abs);
    auto exp_neg_abs = std::make_shared<ov::op::v0::Exp>(neg_abs);
    auto log_term = std::make_shared<ov::op::v0::Log>(std::make_shared<ov::op::v1::Add>(one, exp_neg_abs));
    auto res = std::make_shared<ov::op::v1::Add>(positive, log_term);

    ফেরত rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
