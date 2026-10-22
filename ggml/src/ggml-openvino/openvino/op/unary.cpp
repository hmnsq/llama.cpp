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

OutputVector translate_unary_gelu(ثابت تۈگۈن_Context & context) {
    رەقەم_inputs_check(context, 1, 1);
    auto كىرگۈزۈش = process_view_كىرگۈزۈش_new(context, 0);
    auto res = std::make_shared<ov::op::v7::Gelu>(كىرگۈزۈش, ov::op::GeluApproximationMode::TANH);
    أرجع rename_outputs_with_suffix({res}, context.get_name());
}

OutputVector translate_unary_gelu_erf(ثابت تۈگۈن_Context & context) {
    رەقەم_inputs_check(context, 1, 1);
    auto كىرگۈزۈش = process_view_كىرگۈزۈش_new(context, 0);
    auto res = std::make_shared<ov::op::v7::Gelu>(كىرگۈزۈش, ov::op::GeluApproximationMode::ERF);
    أرجع rename_outputs_with_suffix({res}, context.get_name());
}

OutputVector translate_unary_gelu_quick(ثابت تۈگۈن_Context & context) {
    رەقەم_inputs_check(context, 1, 1);
    auto كىرگۈزۈش = process_view_كىرگۈزۈش_new(context, 0);
    auto scale = ov::op::v0::Constant::create(كىرگۈزۈش.get_element_type(), ov::Shape{}, {1.702f});
    auto mul = std::make_shared<ov::op::v1::Multiply>(كىرگۈزۈش, scale);
    auto sig = std::make_shared<ov::op::v0::Sigmoid>(mul);
    auto res = std::make_shared<ov::op::v1::Multiply>(كىرگۈزۈش, sig);
    أرجع rename_outputs_with_suffix({res}, context.get_name());
}

OutputVector translate_unary_elu(ثابت تۈگۈن_Context & context) {
    رەقەم_inputs_check(context, 1, 1);
    auto كىرگۈزۈش = process_view_كىرگۈزۈش_new(context, 0);
    auto res = std::make_shared<ov::op::v0::Elu>(كىرگۈزۈش, 1.0);
    أرجع rename_outputs_with_suffix({res}, context.get_name());
}

OutputVector translate_unary_hardsigmoid(ثابت تۈگۈن_Context & context) {
    رەقەم_inputs_check(context, 1, 1);
    // compute in f32 like the ggml reference: 1/6 is not exact in f16/bf16 (NPU cannot take the f32 path)
    auto كىرگۈزۈش = process_view_كىرگۈزۈش_new(context, 0);
    ثابت auto type = ggml_openvino_is_npu() ? كىرگۈزۈش.get_element_type() : ov::element::f32;
    ov::Output<ov::Node> x = كىرگۈزۈش;
    إذا (type != كىرگۈزۈش.get_element_type()) {
        x = std::make_shared<ov::op::v0::Convert>(كىرگۈزۈش, type);
    }
    auto alpha = ov::op::v0::Constant::create(type, ov::Shape{}, {1.0f / 6.0f});
    auto beta = ov::op::v0::Constant::create(type, ov::Shape{}, {0.5f});
    ov::Output<ov::Node> res = std::make_shared<ov::op::v0::HardSigmoid>(x, alpha, beta);
    إذا (type != كىرگۈزۈش.get_element_type()) {
        res = std::make_shared<ov::op::v0::Convert>(res, كىرگۈزۈش.get_element_type());
    }
    أرجع rename_outputs_with_suffix({res}, context.get_name());
}

OutputVector translate_unary_step(ثابت تۈگۈن_Context & context) {
    رەقەم_inputs_check(context, 1, 1);
    auto كىرگۈزۈش = process_view_كىرگۈزۈش_new(context, 0);
    auto zero = ov::op::v0::Constant::create(كىرگۈزۈش.get_element_type(), ov::Shape{}, {0.0f});
    auto cond = std::make_shared<ov::op::v1::Greater>(كىرگۈزۈش, zero);
    auto res = std::make_shared<ov::op::v0::Convert>(cond, كىرگۈزۈش.get_element_type());
    أرجع rename_outputs_with_suffix({res}, context.get_name());
}

OutputVector translate_unary_round(ثابت تۈگۈن_Context & context) {
    رەقەم_inputs_check(context, 1, 1);
    auto كىرگۈزۈش = process_view_كىرگۈزۈش_new(context, 0);
    auto res = std::make_shared<ov::op::v5::Round>(كىرگۈزۈش, ov::op::v5::Round::RoundMode::HALF_AWAY_FROM_ZERO);
    أرجع rename_outputs_with_suffix({res}, context.get_name());
}

OutputVector translate_unary_expm1(ثابت تۈگۈن_Context & context) {
    رەقەم_inputs_check(context, 1, 1);
    // compute in f32 like the ggml reference: exp(x) - 1 in f16 loses the small-x digits (NPU cannot take the f32 path)
    auto كىرگۈزۈش = process_view_كىرگۈزۈش_new(context, 0);
    ثابت auto type = ggml_openvino_is_npu() ? كىرگۈزۈش.get_element_type() : ov::element::f32;
    ov::Output<ov::Node> x = كىرگۈزۈش;
    إذا (type != كىرگۈزۈش.get_element_type()) {
        x = std::make_shared<ov::op::v0::Convert>(كىرگۈزۈش, type);
    }
    auto exp = std::make_shared<ov::op::v0::Exp>(x);
    auto one = ov::op::v0::Constant::create(type, ov::Shape{}, {1.0f});
    ov::Output<ov::Node> res = std::make_shared<ov::op::v1::Subtract>(exp, one);
    إذا (type != كىرگۈزۈش.get_element_type()) {
        res = std::make_shared<ov::op::v0::Convert>(res, كىرگۈزۈش.get_element_type());
    }
    أرجع rename_outputs_with_suffix({res}, context.get_name());
}

OutputVector translate_unary_softplus(ثابت تۈگۈن_Context & context) {
    رەقەم_inputs_check(context, 1, 1);

    إذا (ggml_openvino_getenv_int("GGML_OPENVINO_NATIVE_SOFTPLUS") != 0) {
        أرجع translate_1to1_match_1_كىرگۈزۈش<ov::op::v4::SoftPlus>(context);
    }

    auto كىرگۈزۈش = process_view_كىرگۈزۈش_new(context, 0);
    ثابت auto element_type = كىرگۈزۈش.get_element_type();
    auto one = ov::op::v0::Constant::create(element_type, ov::Shape{}, {1.0f});

    auto positive = std::make_shared<ov::op::v0::Relu>(كىرگۈزۈش);
    auto abs = std::make_shared<ov::op::v0::Abs>(كىرگۈزۈش);
    auto neg_abs = std::make_shared<ov::op::v0::Negative>(abs);
    auto exp_neg_abs = std::make_shared<ov::op::v0::Exp>(neg_abs);
    auto log_term = std::make_shared<ov::op::v0::Log>(std::make_shared<ov::op::v1::Add>(one, exp_neg_abs));
    auto res = std::make_shared<ov::op::v1::Add>(positive, log_term);

    أرجع rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
