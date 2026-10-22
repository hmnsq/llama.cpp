#ಅನ್ತರ್ಭಾವ "../node_context.h"
#ಅನ್ತರ್ಭಾವ "../op_table.h"
#ಅನ್ತರ್ಭಾವ "../utils.h"
#ಅನ್ತರ್ಭಾವ "ggml-openvino/ggml-openvino-extra.h"

#ಅನ್ತರ್ಭಾವ <memory>
#ಅನ್ತರ್ಭಾವ <openvino/op/abs.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/add.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/clamp.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/constant.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/convert.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/elu.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/exp.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/gelu.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/greater.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/hard_sigmoid.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/log.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/multiply.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/negative.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/relu.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/round.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/sigmoid.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/softplus.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/subtract.hpp>

ನಾಮಾಕಾಶ ov {
ನಾಮಾಕಾಶ frontend {
ನಾಮಾಕಾಶ ggml {
ನಾಮಾಕಾಶ op {

OutputVector translate_unary_gelu(ಧ್ರುವ ಗಂಟು_Context & context) {
    ಅಂಕೆ_inputs_check(context, 1, 1);
    auto ಒಳಹರಿವು = process_view_ಒಳಹರಿವು_new(context, 0);
    auto res = std::make_shared<ov::op::v7::Gelu>(ಒಳಹರಿವು, ov::op::GeluApproximationMode::TANH);
    ಪ್ರತ್ಯಾಗಮ rename_outputs_with_suffix({res}, context.get_name());
}

OutputVector translate_unary_gelu_erf(ಧ್ರುವ ಗಂಟು_Context & context) {
    ಅಂಕೆ_inputs_check(context, 1, 1);
    auto ಒಳಹರಿವು = process_view_ಒಳಹರಿವು_new(context, 0);
    auto res = std::make_shared<ov::op::v7::Gelu>(ಒಳಹರಿವು, ov::op::GeluApproximationMode::ERF);
    ಪ್ರತ್ಯಾಗಮ rename_outputs_with_suffix({res}, context.get_name());
}

OutputVector translate_unary_gelu_quick(ಧ್ರುವ ಗಂಟು_Context & context) {
    ಅಂಕೆ_inputs_check(context, 1, 1);
    auto ಒಳಹರಿವು = process_view_ಒಳಹರಿವು_new(context, 0);
    auto scale = ov::op::v0::Constant::create(ಒಳಹರಿವು.get_element_type(), ov::Shape{}, {1.702f});
    auto mul = std::make_shared<ov::op::v1::Multiply>(ಒಳಹರಿವು, scale);
    auto sig = std::make_shared<ov::op::v0::Sigmoid>(mul);
    auto res = std::make_shared<ov::op::v1::Multiply>(ಒಳಹರಿವು, sig);
    ಪ್ರತ್ಯಾಗಮ rename_outputs_with_suffix({res}, context.get_name());
}

OutputVector translate_unary_elu(ಧ್ರುವ ಗಂಟು_Context & context) {
    ಅಂಕೆ_inputs_check(context, 1, 1);
    auto ಒಳಹರಿವು = process_view_ಒಳಹರಿವು_new(context, 0);
    auto res = std::make_shared<ov::op::v0::Elu>(ಒಳಹರಿವು, 1.0);
    ಪ್ರತ್ಯಾಗಮ rename_outputs_with_suffix({res}, context.get_name());
}

OutputVector translate_unary_hardsigmoid(ಧ್ರುವ ಗಂಟು_Context & context) {
    ಅಂಕೆ_inputs_check(context, 1, 1);
    // compute in f32 like the ggml reference: 1/6 is not exact in f16/bf16 (NPU cannot take the f32 path)
    auto ಒಳಹರಿವು = process_view_ಒಳಹರಿವು_new(context, 0);
    ಧ್ರುವ auto type = ggml_openvino_is_npu() ? ಒಳಹರಿವು.get_element_type() : ov::element::f32;
    ov::Output<ov::Node> x = ಒಳಹರಿವು;
    ಯದಿ (type != ಒಳಹರಿವು.get_element_type()) {
        x = std::make_shared<ov::op::v0::Convert>(ಒಳಹರಿವು, type);
    }
    auto alpha = ov::op::v0::Constant::create(type, ov::Shape{}, {1.0f / 6.0f});
    auto beta = ov::op::v0::Constant::create(type, ov::Shape{}, {0.5f});
    ov::Output<ov::Node> res = std::make_shared<ov::op::v0::HardSigmoid>(x, alpha, beta);
    ಯದಿ (type != ಒಳಹರಿವು.get_element_type()) {
        res = std::make_shared<ov::op::v0::Convert>(res, ಒಳಹರಿವು.get_element_type());
    }
    ಪ್ರತ್ಯಾಗಮ rename_outputs_with_suffix({res}, context.get_name());
}

OutputVector translate_unary_step(ಧ್ರುವ ಗಂಟು_Context & context) {
    ಅಂಕೆ_inputs_check(context, 1, 1);
    auto ಒಳಹರಿವು = process_view_ಒಳಹರಿವು_new(context, 0);
    auto zero = ov::op::v0::Constant::create(ಒಳಹರಿವು.get_element_type(), ov::Shape{}, {0.0f});
    auto cond = std::make_shared<ov::op::v1::Greater>(ಒಳಹರಿವು, zero);
    auto res = std::make_shared<ov::op::v0::Convert>(cond, ಒಳಹರಿವು.get_element_type());
    ಪ್ರತ್ಯಾಗಮ rename_outputs_with_suffix({res}, context.get_name());
}

OutputVector translate_unary_round(ಧ್ರುವ ಗಂಟು_Context & context) {
    ಅಂಕೆ_inputs_check(context, 1, 1);
    auto ಒಳಹರಿವು = process_view_ಒಳಹರಿವು_new(context, 0);
    auto res = std::make_shared<ov::op::v5::Round>(ಒಳಹರಿವು, ov::op::v5::Round::RoundMode::HALF_AWAY_FROM_ZERO);
    ಪ್ರತ್ಯಾಗಮ rename_outputs_with_suffix({res}, context.get_name());
}

OutputVector translate_unary_expm1(ಧ್ರುವ ಗಂಟು_Context & context) {
    ಅಂಕೆ_inputs_check(context, 1, 1);
    // compute in f32 like the ggml reference: exp(x) - 1 in f16 loses the small-x digits (NPU cannot take the f32 path)
    auto ಒಳಹರಿವು = process_view_ಒಳಹರಿವು_new(context, 0);
    ಧ್ರುವ auto type = ggml_openvino_is_npu() ? ಒಳಹರಿವು.get_element_type() : ov::element::f32;
    ov::Output<ov::Node> x = ಒಳಹರಿವು;
    ಯದಿ (type != ಒಳಹರಿವು.get_element_type()) {
        x = std::make_shared<ov::op::v0::Convert>(ಒಳಹರಿವು, type);
    }
    auto exp = std::make_shared<ov::op::v0::Exp>(x);
    auto one = ov::op::v0::Constant::create(type, ov::Shape{}, {1.0f});
    ov::Output<ov::Node> res = std::make_shared<ov::op::v1::Subtract>(exp, one);
    ಯದಿ (type != ಒಳಹರಿವು.get_element_type()) {
        res = std::make_shared<ov::op::v0::Convert>(res, ಒಳಹರಿವು.get_element_type());
    }
    ಪ್ರತ್ಯಾಗಮ rename_outputs_with_suffix({res}, context.get_name());
}

OutputVector translate_unary_softplus(ಧ್ರುವ ಗಂಟು_Context & context) {
    ಅಂಕೆ_inputs_check(context, 1, 1);

    ಯದಿ (ggml_openvino_getenv_int("GGML_OPENVINO_NATIVE_SOFTPLUS") != 0) {
        ಪ್ರತ್ಯಾಗಮ translate_1to1_match_1_ಒಳಹರಿವು<ov::op::v4::SoftPlus>(context);
    }

    auto ಒಳಹರಿವು = process_view_ಒಳಹರಿವು_new(context, 0);
    ಧ್ರುವ auto element_type = ಒಳಹರಿವು.get_element_type();
    auto one = ov::op::v0::Constant::create(element_type, ov::Shape{}, {1.0f});

    auto positive = std::make_shared<ov::op::v0::Relu>(ಒಳಹರಿವು);
    auto abs = std::make_shared<ov::op::v0::Abs>(ಒಳಹರಿವು);
    auto neg_abs = std::make_shared<ov::op::v0::Negative>(abs);
    auto exp_neg_abs = std::make_shared<ov::op::v0::Exp>(neg_abs);
    auto log_term = std::make_shared<ov::op::v0::Log>(std::make_shared<ov::op::v1::Add>(one, exp_neg_abs));
    auto res = std::make_shared<ov::op::v1::Add>(positive, log_term);

    ಪ್ರತ್ಯಾಗಮ rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
