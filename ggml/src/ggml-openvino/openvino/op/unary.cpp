#അന്തര്ഭാവ "../node_context.h"
#അന്തര്ഭാവ "../op_table.h"
#അന്തര്ഭാവ "../utils.h"
#അന്തര്ഭാവ "ggml-openvino/ggml-openvino-extra.h"

#അന്തര്ഭാവ <memory>
#അന്തര്ഭാവ <openvino/op/abs.hpp>
#അന്തര്ഭാവ <openvino/op/add.hpp>
#അന്തര്ഭാവ <openvino/op/clamp.hpp>
#അന്തര്ഭാവ <openvino/op/constant.hpp>
#അന്തര്ഭാവ <openvino/op/convert.hpp>
#അന്തര്ഭാവ <openvino/op/elu.hpp>
#അന്തര്ഭാവ <openvino/op/exp.hpp>
#അന്തര്ഭാവ <openvino/op/gelu.hpp>
#അന്തര്ഭാവ <openvino/op/greater.hpp>
#അന്തര്ഭാവ <openvino/op/hard_sigmoid.hpp>
#അന്തര്ഭാവ <openvino/op/log.hpp>
#അന്തര്ഭാവ <openvino/op/multiply.hpp>
#അന്തര്ഭാവ <openvino/op/negative.hpp>
#അന്തര്ഭാവ <openvino/op/relu.hpp>
#അന്തര്ഭാവ <openvino/op/round.hpp>
#അന്തര്ഭാവ <openvino/op/sigmoid.hpp>
#അന്തര്ഭാവ <openvino/op/softplus.hpp>
#അന്തര്ഭാവ <openvino/op/subtract.hpp>

നാമാകാശ ov {
നാമാകാശ frontend {
നാമാകാശ ggml {
നാമാകാശ op {

OutputVector translate_unary_gelu(സ്ഥിരം കണ്ണി_Context & context) {
    അക്കം_inputs_check(context, 1, 1);
    auto നിവേശം = process_view_നിവേശം_new(context, 0);
    auto res = std::make_shared<ov::op::v7::Gelu>(നിവേശം, ov::op::GeluApproximationMode::TANH);
    തിരികെ rename_outputs_with_suffix({res}, context.get_name());
}

OutputVector translate_unary_gelu_erf(സ്ഥിരം കണ്ണി_Context & context) {
    അക്കം_inputs_check(context, 1, 1);
    auto നിവേശം = process_view_നിവേശം_new(context, 0);
    auto res = std::make_shared<ov::op::v7::Gelu>(നിവേശം, ov::op::GeluApproximationMode::ERF);
    തിരികെ rename_outputs_with_suffix({res}, context.get_name());
}

OutputVector translate_unary_gelu_quick(സ്ഥിരം കണ്ണി_Context & context) {
    അക്കം_inputs_check(context, 1, 1);
    auto നിവേശം = process_view_നിവേശം_new(context, 0);
    auto scale = ov::op::v0::Constant::create(നിവേശം.get_element_type(), ov::Shape{}, {1.702f});
    auto mul = std::make_shared<ov::op::v1::Multiply>(നിവേശം, scale);
    auto sig = std::make_shared<ov::op::v0::Sigmoid>(mul);
    auto res = std::make_shared<ov::op::v1::Multiply>(നിവേശം, sig);
    തിരികെ rename_outputs_with_suffix({res}, context.get_name());
}

OutputVector translate_unary_elu(സ്ഥിരം കണ്ണി_Context & context) {
    അക്കം_inputs_check(context, 1, 1);
    auto നിവേശം = process_view_നിവേശം_new(context, 0);
    auto res = std::make_shared<ov::op::v0::Elu>(നിവേശം, 1.0);
    തിരികെ rename_outputs_with_suffix({res}, context.get_name());
}

OutputVector translate_unary_hardsigmoid(സ്ഥിരം കണ്ണി_Context & context) {
    അക്കം_inputs_check(context, 1, 1);
    // compute in f32 like the ggml reference: 1/6 is not exact in f16/bf16 (NPU cannot take the f32 path)
    auto നിവേശം = process_view_നിവേശം_new(context, 0);
    സ്ഥിരം auto type = ggml_openvino_is_npu() ? നിവേശം.get_element_type() : ov::element::f32;
    ov::Output<ov::Node> x = നിവേശം;
    എങ്കിൽ (type != നിവേശം.get_element_type()) {
        x = std::make_shared<ov::op::v0::Convert>(നിവേശം, type);
    }
    auto alpha = ov::op::v0::Constant::create(type, ov::Shape{}, {1.0f / 6.0f});
    auto beta = ov::op::v0::Constant::create(type, ov::Shape{}, {0.5f});
    ov::Output<ov::Node> res = std::make_shared<ov::op::v0::HardSigmoid>(x, alpha, beta);
    എങ്കിൽ (type != നിവേശം.get_element_type()) {
        res = std::make_shared<ov::op::v0::Convert>(res, നിവേശം.get_element_type());
    }
    തിരികെ rename_outputs_with_suffix({res}, context.get_name());
}

OutputVector translate_unary_step(സ്ഥിരം കണ്ണി_Context & context) {
    അക്കം_inputs_check(context, 1, 1);
    auto നിവേശം = process_view_നിവേശം_new(context, 0);
    auto zero = ov::op::v0::Constant::create(നിവേശം.get_element_type(), ov::Shape{}, {0.0f});
    auto cond = std::make_shared<ov::op::v1::Greater>(നിവേശം, zero);
    auto res = std::make_shared<ov::op::v0::Convert>(cond, നിവേശം.get_element_type());
    തിരികെ rename_outputs_with_suffix({res}, context.get_name());
}

OutputVector translate_unary_round(സ്ഥിരം കണ്ണി_Context & context) {
    അക്കം_inputs_check(context, 1, 1);
    auto നിവേശം = process_view_നിവേശം_new(context, 0);
    auto res = std::make_shared<ov::op::v5::Round>(നിവേശം, ov::op::v5::Round::RoundMode::HALF_AWAY_FROM_ZERO);
    തിരികെ rename_outputs_with_suffix({res}, context.get_name());
}

OutputVector translate_unary_expm1(സ്ഥിരം കണ്ണി_Context & context) {
    അക്കം_inputs_check(context, 1, 1);
    // compute in f32 like the ggml reference: exp(x) - 1 in f16 loses the small-x digits (NPU cannot take the f32 path)
    auto നിവേശം = process_view_നിവേശം_new(context, 0);
    സ്ഥിരം auto type = ggml_openvino_is_npu() ? നിവേശം.get_element_type() : ov::element::f32;
    ov::Output<ov::Node> x = നിവേശം;
    എങ്കിൽ (type != നിവേശം.get_element_type()) {
        x = std::make_shared<ov::op::v0::Convert>(നിവേശം, type);
    }
    auto exp = std::make_shared<ov::op::v0::Exp>(x);
    auto one = ov::op::v0::Constant::create(type, ov::Shape{}, {1.0f});
    ov::Output<ov::Node> res = std::make_shared<ov::op::v1::Subtract>(exp, one);
    എങ്കിൽ (type != നിവേശം.get_element_type()) {
        res = std::make_shared<ov::op::v0::Convert>(res, നിവേശം.get_element_type());
    }
    തിരികെ rename_outputs_with_suffix({res}, context.get_name());
}

OutputVector translate_unary_softplus(സ്ഥിരം കണ്ണി_Context & context) {
    അക്കം_inputs_check(context, 1, 1);

    എങ്കിൽ (ggml_openvino_getenv_int("GGML_OPENVINO_NATIVE_SOFTPLUS") != 0) {
        തിരികെ translate_1to1_match_1_നിവേശം<ov::op::v4::SoftPlus>(context);
    }

    auto നിവേശം = process_view_നിവേശം_new(context, 0);
    സ്ഥിരം auto element_type = നിവേശം.get_element_type();
    auto one = ov::op::v0::Constant::create(element_type, ov::Shape{}, {1.0f});

    auto positive = std::make_shared<ov::op::v0::Relu>(നിവേശം);
    auto abs = std::make_shared<ov::op::v0::Abs>(നിവേശം);
    auto neg_abs = std::make_shared<ov::op::v0::Negative>(abs);
    auto exp_neg_abs = std::make_shared<ov::op::v0::Exp>(neg_abs);
    auto log_term = std::make_shared<ov::op::v0::Log>(std::make_shared<ov::op::v1::Add>(one, exp_neg_abs));
    auto res = std::make_shared<ov::op::v1::Add>(positive, log_term);

    തിരികെ rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
