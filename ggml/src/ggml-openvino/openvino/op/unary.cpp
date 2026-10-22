#अन्तर्भाव "../node_context.h"
#अन्तर्भाव "../op_table.h"
#अन्तर्भाव "../utils.h"
#अन्तर्भाव "ggml-openvino/ggml-openvino-extra.h"

#अन्तर्भाव <memory>
#अन्तर्भाव <openvino/op/abs.hpp>
#अन्तर्भाव <openvino/op/add.hpp>
#अन्तर्भाव <openvino/op/clamp.hpp>
#अन्तर्भाव <openvino/op/constant.hpp>
#अन्तर्भाव <openvino/op/convert.hpp>
#अन्तर्भाव <openvino/op/elu.hpp>
#अन्तर्भाव <openvino/op/exp.hpp>
#अन्तर्भाव <openvino/op/gelu.hpp>
#अन्तर्भाव <openvino/op/greater.hpp>
#अन्तर्भाव <openvino/op/hard_sigmoid.hpp>
#अन्तर्भाव <openvino/op/log.hpp>
#अन्तर्भाव <openvino/op/multiply.hpp>
#अन्तर्भाव <openvino/op/negative.hpp>
#अन्तर्भाव <openvino/op/relu.hpp>
#अन्तर्भाव <openvino/op/round.hpp>
#अन्तर्भाव <openvino/op/sigmoid.hpp>
#अन्तर्भाव <openvino/op/softplus.hpp>
#अन्तर्भाव <openvino/op/subtract.hpp>

नामाकाश ov {
नामाकाश frontend {
नामाकाश ggml {
नामाकाश op {

OutputVector translate_unary_gelu(ध्रुव गाँठ_Context & context) {
    अंक_inputs_check(context, 1, 1);
    auto निवेश = process_view_निवेश_new(context, 0);
    auto res = std::make_shared<ov::op::v7::Gelu>(निवेश, ov::op::GeluApproximationMode::TANH);
    प्रत्यागम rename_outputs_with_suffix({res}, context.get_name());
}

OutputVector translate_unary_gelu_erf(ध्रुव गाँठ_Context & context) {
    अंक_inputs_check(context, 1, 1);
    auto निवेश = process_view_निवेश_new(context, 0);
    auto res = std::make_shared<ov::op::v7::Gelu>(निवेश, ov::op::GeluApproximationMode::ERF);
    प्रत्यागम rename_outputs_with_suffix({res}, context.get_name());
}

OutputVector translate_unary_gelu_quick(ध्रुव गाँठ_Context & context) {
    अंक_inputs_check(context, 1, 1);
    auto निवेश = process_view_निवेश_new(context, 0);
    auto scale = ov::op::v0::Constant::create(निवेश.get_element_type(), ov::Shape{}, {1.702f});
    auto mul = std::make_shared<ov::op::v1::Multiply>(निवेश, scale);
    auto sig = std::make_shared<ov::op::v0::Sigmoid>(mul);
    auto res = std::make_shared<ov::op::v1::Multiply>(निवेश, sig);
    प्रत्यागम rename_outputs_with_suffix({res}, context.get_name());
}

OutputVector translate_unary_elu(ध्रुव गाँठ_Context & context) {
    अंक_inputs_check(context, 1, 1);
    auto निवेश = process_view_निवेश_new(context, 0);
    auto res = std::make_shared<ov::op::v0::Elu>(निवेश, 1.0);
    प्रत्यागम rename_outputs_with_suffix({res}, context.get_name());
}

OutputVector translate_unary_hardsigmoid(ध्रुव गाँठ_Context & context) {
    अंक_inputs_check(context, 1, 1);
    // compute in f32 like the ggml reference: 1/6 is not exact in f16/bf16 (NPU cannot take the f32 path)
    auto निवेश = process_view_निवेश_new(context, 0);
    ध्रुव auto type = ggml_openvino_is_npu() ? निवेश.get_element_type() : ov::element::f32;
    ov::Output<ov::Node> x = निवेश;
    यदि (type != निवेश.get_element_type()) {
        x = std::make_shared<ov::op::v0::Convert>(निवेश, type);
    }
    auto alpha = ov::op::v0::Constant::create(type, ov::Shape{}, {1.0f / 6.0f});
    auto beta = ov::op::v0::Constant::create(type, ov::Shape{}, {0.5f});
    ov::Output<ov::Node> res = std::make_shared<ov::op::v0::HardSigmoid>(x, alpha, beta);
    यदि (type != निवेश.get_element_type()) {
        res = std::make_shared<ov::op::v0::Convert>(res, निवेश.get_element_type());
    }
    प्रत्यागम rename_outputs_with_suffix({res}, context.get_name());
}

OutputVector translate_unary_step(ध्रुव गाँठ_Context & context) {
    अंक_inputs_check(context, 1, 1);
    auto निवेश = process_view_निवेश_new(context, 0);
    auto zero = ov::op::v0::Constant::create(निवेश.get_element_type(), ov::Shape{}, {0.0f});
    auto cond = std::make_shared<ov::op::v1::Greater>(निवेश, zero);
    auto res = std::make_shared<ov::op::v0::Convert>(cond, निवेश.get_element_type());
    प्रत्यागम rename_outputs_with_suffix({res}, context.get_name());
}

OutputVector translate_unary_round(ध्रुव गाँठ_Context & context) {
    अंक_inputs_check(context, 1, 1);
    auto निवेश = process_view_निवेश_new(context, 0);
    auto res = std::make_shared<ov::op::v5::Round>(निवेश, ov::op::v5::Round::RoundMode::HALF_AWAY_FROM_ZERO);
    प्रत्यागम rename_outputs_with_suffix({res}, context.get_name());
}

OutputVector translate_unary_expm1(ध्रुव गाँठ_Context & context) {
    अंक_inputs_check(context, 1, 1);
    // compute in f32 like the ggml reference: exp(x) - 1 in f16 loses the small-x digits (NPU cannot take the f32 path)
    auto निवेश = process_view_निवेश_new(context, 0);
    ध्रुव auto type = ggml_openvino_is_npu() ? निवेश.get_element_type() : ov::element::f32;
    ov::Output<ov::Node> x = निवेश;
    यदि (type != निवेश.get_element_type()) {
        x = std::make_shared<ov::op::v0::Convert>(निवेश, type);
    }
    auto exp = std::make_shared<ov::op::v0::Exp>(x);
    auto one = ov::op::v0::Constant::create(type, ov::Shape{}, {1.0f});
    ov::Output<ov::Node> res = std::make_shared<ov::op::v1::Subtract>(exp, one);
    यदि (type != निवेश.get_element_type()) {
        res = std::make_shared<ov::op::v0::Convert>(res, निवेश.get_element_type());
    }
    प्रत्यागम rename_outputs_with_suffix({res}, context.get_name());
}

OutputVector translate_unary_softplus(ध्रुव गाँठ_Context & context) {
    अंक_inputs_check(context, 1, 1);

    यदि (ggml_openvino_getenv_int("GGML_OPENVINO_NATIVE_SOFTPLUS") != 0) {
        प्रत्यागम translate_1to1_match_1_निवेश<ov::op::v4::SoftPlus>(context);
    }

    auto निवेश = process_view_निवेश_new(context, 0);
    ध्रुव auto element_type = निवेश.get_element_type();
    auto one = ov::op::v0::Constant::create(element_type, ov::Shape{}, {1.0f});

    auto positive = std::make_shared<ov::op::v0::Relu>(निवेश);
    auto abs = std::make_shared<ov::op::v0::Abs>(निवेश);
    auto neg_abs = std::make_shared<ov::op::v0::Negative>(abs);
    auto exp_neg_abs = std::make_shared<ov::op::v0::Exp>(neg_abs);
    auto log_term = std::make_shared<ov::op::v0::Log>(std::make_shared<ov::op::v1::Add>(one, exp_neg_abs));
    auto res = std::make_shared<ov::op::v1::Add>(positive, log_term);

    प्रत्यागम rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
