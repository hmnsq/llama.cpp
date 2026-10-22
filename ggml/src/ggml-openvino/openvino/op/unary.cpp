#ਅਨ੍ਤਰ੍ਭਾਵ "../node_context.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "../op_table.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "../utils.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-openvino/ggml-openvino-extra.h"

#ਅਨ੍ਤਰ੍ਭਾਵ <memory>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/abs.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/add.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/clamp.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/constant.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/convert.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/elu.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/exp.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/gelu.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/greater.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/hard_sigmoid.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/log.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/multiply.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/negative.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/relu.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/round.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/sigmoid.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/softplus.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/subtract.hpp>

ਨਾਮਾਕਾਸ਼ ov {
ਨਾਮਾਕਾਸ਼ frontend {
ਨਾਮਾਕਾਸ਼ ggml {
ਨਾਮਾਕਾਸ਼ op {

Output_ਸਦਿਸ਼ translate_unary_gelu(ਅਚਲ ਗੰਢ_Context & context) {
    ਅੰਕ_inputs_check(context, 1, 1);
    ਸਵੈਚਾਲਿਤ ਨਿਵੇਸ਼ = process_view_ਨਿਵੇਸ਼_new(context, 0);
    ਸਵੈਚਾਲਿਤ res = std::make_shared<ov::op::v7::Gelu>(ਨਿਵੇਸ਼, ov::op::Gelu_Approximation_ਬਹੁਲਕ::TANH);
    ਵਾਪਸ rename_outputs_with_suffix({res}, context.get_name());
}

Output_ਸਦਿਸ਼ translate_unary_gelu_erf(ਅਚਲ ਗੰਢ_Context & context) {
    ਅੰਕ_inputs_check(context, 1, 1);
    ਸਵੈਚਾਲਿਤ ਨਿਵੇਸ਼ = process_view_ਨਿਵੇਸ਼_new(context, 0);
    ਸਵੈਚਾਲਿਤ res = std::make_shared<ov::op::v7::Gelu>(ਨਿਵੇਸ਼, ov::op::Gelu_Approximation_ਬਹੁਲਕ::ERF);
    ਵਾਪਸ rename_outputs_with_suffix({res}, context.get_name());
}

Output_ਸਦਿਸ਼ translate_unary_gelu_ਤੇਜ਼(ਅਚਲ ਗੰਢ_Context & context) {
    ਅੰਕ_inputs_check(context, 1, 1);
    ਸਵੈਚਾਲਿਤ ਨਿਵੇਸ਼ = process_view_ਨਿਵੇਸ਼_new(context, 0);
    ਸਵੈਚਾਲਿਤ scale = ov::op::v0::Constant::create(ਨਿਵੇਸ਼.get_element_type(), ov::Shape{}, {1.702f});
    ਸਵੈਚਾਲਿਤ mul = std::make_shared<ov::op::v1::Multiply>(ਨਿਵੇਸ਼, scale);
    ਸਵੈਚਾਲਿਤ sig = std::make_shared<ov::op::v0::Sigmoid>(mul);
    ਸਵੈਚਾਲਿਤ res = std::make_shared<ov::op::v1::Multiply>(ਨਿਵੇਸ਼, sig);
    ਵਾਪਸ rename_outputs_with_suffix({res}, context.get_name());
}

Output_ਸਦਿਸ਼ translate_unary_elu(ਅਚਲ ਗੰਢ_Context & context) {
    ਅੰਕ_inputs_check(context, 1, 1);
    ਸਵੈਚਾਲਿਤ ਨਿਵੇਸ਼ = process_view_ਨਿਵੇਸ਼_new(context, 0);
    ਸਵੈਚਾਲਿਤ res = std::make_shared<ov::op::v0::Elu>(ਨਿਵੇਸ਼, 1.0);
    ਵਾਪਸ rename_outputs_with_suffix({res}, context.get_name());
}

Output_ਸਦਿਸ਼ translate_unary_hardsigmoid(ਅਚਲ ਗੰਢ_Context & context) {
    ਅੰਕ_inputs_check(context, 1, 1);
    // compute in f32 like the ggml reference: 1/6 is not exact in f16/bf16 (NPU cannot take the f32 path)
    ਸਵੈਚਾਲਿਤ ਨਿਵੇਸ਼ = process_view_ਨਿਵੇਸ਼_new(context, 0);
    ਅਚਲ ਸਵੈਚਾਲਿਤ type = ggml_openvino_is_npu() ? ਨਿਵੇਸ਼.get_element_type() : ov::element::f32;
    ov::Output<ov::Node> x = ਨਿਵੇਸ਼;
    ਜੇ (type != ਨਿਵੇਸ਼.get_element_type()) {
        x = std::make_shared<ov::op::v0::Convert>(ਨਿਵੇਸ਼, type);
    }
    ਸਵੈਚਾਲਿਤ alpha = ov::op::v0::Constant::create(type, ov::Shape{}, {1.0f / 6.0f});
    ਸਵੈਚਾਲਿਤ beta = ov::op::v0::Constant::create(type, ov::Shape{}, {0.5f});
    ov::Output<ov::Node> res = std::make_shared<ov::op::v0::HardSigmoid>(x, alpha, beta);
    ਜੇ (type != ਨਿਵੇਸ਼.get_element_type()) {
        res = std::make_shared<ov::op::v0::Convert>(res, ਨਿਵੇਸ਼.get_element_type());
    }
    ਵਾਪਸ rename_outputs_with_suffix({res}, context.get_name());
}

Output_ਸਦਿਸ਼ translate_unary_ਕਦਮ(ਅਚਲ ਗੰਢ_Context & context) {
    ਅੰਕ_inputs_check(context, 1, 1);
    ਸਵੈਚਾਲਿਤ ਨਿਵੇਸ਼ = process_view_ਨਿਵੇਸ਼_new(context, 0);
    ਸਵੈਚਾਲਿਤ zero = ov::op::v0::Constant::create(ਨਿਵੇਸ਼.get_element_type(), ov::Shape{}, {0.0f});
    ਸਵੈਚਾਲਿਤ cond = std::make_shared<ov::op::v1::Greater>(ਨਿਵੇਸ਼, zero);
    ਸਵੈਚਾਲਿਤ res = std::make_shared<ov::op::v0::Convert>(cond, ਨਿਵੇਸ਼.get_element_type());
    ਵਾਪਸ rename_outputs_with_suffix({res}, context.get_name());
}

Output_ਸਦਿਸ਼ translate_unary_ਗੇੜਾ(ਅਚਲ ਗੰਢ_Context & context) {
    ਅੰਕ_inputs_check(context, 1, 1);
    ਸਵੈਚਾਲਿਤ ਨਿਵੇਸ਼ = process_view_ਨਿਵੇਸ਼_new(context, 0);
    ਸਵੈਚਾਲਿਤ res = std::make_shared<ov::op::v5::ਗੇੜਾ>(ਨਿਵੇਸ਼, ov::op::v5::ਗੇੜਾ::ਗੇੜਾ_ਬਹੁਲਕ::HALF_AWAY_FROM_ZERO);
    ਵਾਪਸ rename_outputs_with_suffix({res}, context.get_name());
}

Output_ਸਦਿਸ਼ translate_unary_expm1(ਅਚਲ ਗੰਢ_Context & context) {
    ਅੰਕ_inputs_check(context, 1, 1);
    // compute in f32 like the ggml reference: exp(x) - 1 in f16 loses the small-x digits (NPU cannot take the f32 path)
    ਸਵੈਚਾਲਿਤ ਨਿਵੇਸ਼ = process_view_ਨਿਵੇਸ਼_new(context, 0);
    ਅਚਲ ਸਵੈਚਾਲਿਤ type = ggml_openvino_is_npu() ? ਨਿਵੇਸ਼.get_element_type() : ov::element::f32;
    ov::Output<ov::Node> x = ਨਿਵੇਸ਼;
    ਜੇ (type != ਨਿਵੇਸ਼.get_element_type()) {
        x = std::make_shared<ov::op::v0::Convert>(ਨਿਵੇਸ਼, type);
    }
    ਸਵੈਚਾਲਿਤ ਘਾਤਾਂਕ = std::make_shared<ov::op::v0::Exp>(x);
    ਸਵੈਚਾਲਿਤ one = ov::op::v0::Constant::create(type, ov::Shape{}, {1.0f});
    ov::Output<ov::Node> res = std::make_shared<ov::op::v1::ਘਟਾਓ>(ਘਾਤਾਂਕ, one);
    ਜੇ (type != ਨਿਵੇਸ਼.get_element_type()) {
        res = std::make_shared<ov::op::v0::Convert>(res, ਨਿਵੇਸ਼.get_element_type());
    }
    ਵਾਪਸ rename_outputs_with_suffix({res}, context.get_name());
}

Output_ਸਦਿਸ਼ translate_unary_softplus(ਅਚਲ ਗੰਢ_Context & context) {
    ਅੰਕ_inputs_check(context, 1, 1);

    ਜੇ (ggml_openvino_getenv_int("GGML_OPENVINO_NATIVE_SOFTPLUS") != 0) {
        ਵਾਪਸ translate_1to1_ਮੇਲ_1_ਨਿਵੇਸ਼<ov::op::v4::SoftPlus>(context);
    }

    ਸਵੈਚਾਲਿਤ ਨਿਵੇਸ਼ = process_view_ਨਿਵੇਸ਼_new(context, 0);
    ਅਚਲ ਸਵੈਚਾਲਿਤ element_type = ਨਿਵੇਸ਼.get_element_type();
    ਸਵੈਚਾਲਿਤ one = ov::op::v0::Constant::create(element_type, ov::Shape{}, {1.0f});

    ਸਵੈਚਾਲਿਤ ਸਕਾਰਾਤਮਕ = std::make_shared<ov::op::v0::Relu>(ਨਿਵੇਸ਼);
    ਸਵੈਚਾਲਿਤ ਨਿਰਪੇਖ = std::make_shared<ov::op::v0::Abs>(ਨਿਵੇਸ਼);
    ਸਵੈਚਾਲਿਤ neg_abs = std::make_shared<ov::op::v0::Negative>(ਨਿਰਪੇਖ);
    ਸਵੈਚਾਲਿਤ exp_neg_abs = std::make_shared<ov::op::v0::Exp>(neg_abs);
    ਸਵੈਚਾਲਿਤ log_ਪਦ = std::make_shared<ov::op::v0::Log>(std::make_shared<ov::op::v1::ਜੋੜੋ>(one, exp_neg_abs));
    ਸਵੈਚਾਲਿਤ res = std::make_shared<ov::op::v1::ਜੋੜੋ>(ਸਕਾਰਾਤਮਕ, log_ਪਦ);

    ਵਾਪਸ rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
