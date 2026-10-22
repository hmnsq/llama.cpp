#அந்தர்பாவ "../node_context.h"
#அந்தர்பாவ "../op_table.h"
#அந்தர்பாவ "../utils.h"
#அந்தர்பாவ "ggml-openvino/ggml-openvino-extra.h"

#அந்தர்பாவ <memory>
#அந்தர்பாவ <openvino/op/abs.hpp>
#அந்தர்பாவ <openvino/op/add.hpp>
#அந்தர்பாவ <openvino/op/clamp.hpp>
#அந்தர்பாவ <openvino/op/constant.hpp>
#அந்தர்பாவ <openvino/op/convert.hpp>
#அந்தர்பாவ <openvino/op/elu.hpp>
#அந்தர்பாவ <openvino/op/exp.hpp>
#அந்தர்பாவ <openvino/op/gelu.hpp>
#அந்தர்பாவ <openvino/op/greater.hpp>
#அந்தர்பாவ <openvino/op/hard_sigmoid.hpp>
#அந்தர்பாவ <openvino/op/log.hpp>
#அந்தர்பாவ <openvino/op/multiply.hpp>
#அந்தர்பாவ <openvino/op/negative.hpp>
#அந்தர்பாவ <openvino/op/relu.hpp>
#அந்தர்பாவ <openvino/op/round.hpp>
#அந்தர்பாவ <openvino/op/sigmoid.hpp>
#அந்தர்பாவ <openvino/op/softplus.hpp>
#அந்தர்பாவ <openvino/op/subtract.hpp>

நாமாகாஶ ov {
நாமாகாஶ frontend {
நாமாகாஶ ggml {
நாமாகாஶ op {

OutputVector translate_unary_gelu(மாறிலி முனை_Context & context) {
    இலக்கம்_inputs_check(context, 1, 1);
    auto உள்ளீடு = process_view_உள்ளீடு_new(context, 0);
    auto res = std::make_shared<ov::op::v7::Gelu>(உள்ளீடு, ov::op::GeluApproximationMode::TANH);
    திரும்பு rename_outputs_with_suffix({res}, context.get_name());
}

OutputVector translate_unary_gelu_erf(மாறிலி முனை_Context & context) {
    இலக்கம்_inputs_check(context, 1, 1);
    auto உள்ளீடு = process_view_உள்ளீடு_new(context, 0);
    auto res = std::make_shared<ov::op::v7::Gelu>(உள்ளீடு, ov::op::GeluApproximationMode::ERF);
    திரும்பு rename_outputs_with_suffix({res}, context.get_name());
}

OutputVector translate_unary_gelu_quick(மாறிலி முனை_Context & context) {
    இலக்கம்_inputs_check(context, 1, 1);
    auto உள்ளீடு = process_view_உள்ளீடு_new(context, 0);
    auto scale = ov::op::v0::Constant::create(உள்ளீடு.get_element_type(), ov::Shape{}, {1.702f});
    auto mul = std::make_shared<ov::op::v1::Multiply>(உள்ளீடு, scale);
    auto sig = std::make_shared<ov::op::v0::Sigmoid>(mul);
    auto res = std::make_shared<ov::op::v1::Multiply>(உள்ளீடு, sig);
    திரும்பு rename_outputs_with_suffix({res}, context.get_name());
}

OutputVector translate_unary_elu(மாறிலி முனை_Context & context) {
    இலக்கம்_inputs_check(context, 1, 1);
    auto உள்ளீடு = process_view_உள்ளீடு_new(context, 0);
    auto res = std::make_shared<ov::op::v0::Elu>(உள்ளீடு, 1.0);
    திரும்பு rename_outputs_with_suffix({res}, context.get_name());
}

OutputVector translate_unary_hardsigmoid(மாறிலி முனை_Context & context) {
    இலக்கம்_inputs_check(context, 1, 1);
    // compute in f32 like the ggml reference: 1/6 is not exact in f16/bf16 (NPU cannot take the f32 path)
    auto உள்ளீடு = process_view_உள்ளீடு_new(context, 0);
    மாறிலி auto type = ggml_openvino_is_npu() ? உள்ளீடு.get_element_type() : ov::element::f32;
    ov::Output<ov::Node> x = உள்ளீடு;
    எனில் (type != உள்ளீடு.get_element_type()) {
        x = std::make_shared<ov::op::v0::Convert>(உள்ளீடு, type);
    }
    auto alpha = ov::op::v0::Constant::create(type, ov::Shape{}, {1.0f / 6.0f});
    auto beta = ov::op::v0::Constant::create(type, ov::Shape{}, {0.5f});
    ov::Output<ov::Node> res = std::make_shared<ov::op::v0::HardSigmoid>(x, alpha, beta);
    எனில் (type != உள்ளீடு.get_element_type()) {
        res = std::make_shared<ov::op::v0::Convert>(res, உள்ளீடு.get_element_type());
    }
    திரும்பு rename_outputs_with_suffix({res}, context.get_name());
}

OutputVector translate_unary_step(மாறிலி முனை_Context & context) {
    இலக்கம்_inputs_check(context, 1, 1);
    auto உள்ளீடு = process_view_உள்ளீடு_new(context, 0);
    auto zero = ov::op::v0::Constant::create(உள்ளீடு.get_element_type(), ov::Shape{}, {0.0f});
    auto cond = std::make_shared<ov::op::v1::Greater>(உள்ளீடு, zero);
    auto res = std::make_shared<ov::op::v0::Convert>(cond, உள்ளீடு.get_element_type());
    திரும்பு rename_outputs_with_suffix({res}, context.get_name());
}

OutputVector translate_unary_round(மாறிலி முனை_Context & context) {
    இலக்கம்_inputs_check(context, 1, 1);
    auto உள்ளீடு = process_view_உள்ளீடு_new(context, 0);
    auto res = std::make_shared<ov::op::v5::Round>(உள்ளீடு, ov::op::v5::Round::RoundMode::HALF_AWAY_FROM_ZERO);
    திரும்பு rename_outputs_with_suffix({res}, context.get_name());
}

OutputVector translate_unary_expm1(மாறிலி முனை_Context & context) {
    இலக்கம்_inputs_check(context, 1, 1);
    // compute in f32 like the ggml reference: exp(x) - 1 in f16 loses the small-x digits (NPU cannot take the f32 path)
    auto உள்ளீடு = process_view_உள்ளீடு_new(context, 0);
    மாறிலி auto type = ggml_openvino_is_npu() ? உள்ளீடு.get_element_type() : ov::element::f32;
    ov::Output<ov::Node> x = உள்ளீடு;
    எனில் (type != உள்ளீடு.get_element_type()) {
        x = std::make_shared<ov::op::v0::Convert>(உள்ளீடு, type);
    }
    auto exp = std::make_shared<ov::op::v0::Exp>(x);
    auto one = ov::op::v0::Constant::create(type, ov::Shape{}, {1.0f});
    ov::Output<ov::Node> res = std::make_shared<ov::op::v1::Subtract>(exp, one);
    எனில் (type != உள்ளீடு.get_element_type()) {
        res = std::make_shared<ov::op::v0::Convert>(res, உள்ளீடு.get_element_type());
    }
    திரும்பு rename_outputs_with_suffix({res}, context.get_name());
}

OutputVector translate_unary_softplus(மாறிலி முனை_Context & context) {
    இலக்கம்_inputs_check(context, 1, 1);

    எனில் (ggml_openvino_getenv_int("GGML_OPENVINO_NATIVE_SOFTPLUS") != 0) {
        திரும்பு translate_1to1_match_1_உள்ளீடு<ov::op::v4::SoftPlus>(context);
    }

    auto உள்ளீடு = process_view_உள்ளீடு_new(context, 0);
    மாறிலி auto element_type = உள்ளீடு.get_element_type();
    auto one = ov::op::v0::Constant::create(element_type, ov::Shape{}, {1.0f});

    auto positive = std::make_shared<ov::op::v0::Relu>(உள்ளீடு);
    auto abs = std::make_shared<ov::op::v0::Abs>(உள்ளீடு);
    auto neg_abs = std::make_shared<ov::op::v0::Negative>(abs);
    auto exp_neg_abs = std::make_shared<ov::op::v0::Exp>(neg_abs);
    auto log_term = std::make_shared<ov::op::v0::Log>(std::make_shared<ov::op::v1::Add>(one, exp_neg_abs));
    auto res = std::make_shared<ov::op::v1::Add>(positive, log_term);

    திரும்பு rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
