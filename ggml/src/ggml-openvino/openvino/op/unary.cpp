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

Output_ಸದಿಶ translate_unary_gelu(ಸ್ಥಿರಾಂಕ ಗಂಟು_Context & context) {
    ಅಂಕೆ_inputs_check(context, 1, 1);
    ಸ್ವಯಂಚಾಲಿತ ಒಳಹರಿವು = process_view_ಒಳಹರಿವು_new(context, 0);
    ಸ್ವಯಂಚಾಲಿತ res = std::make_shared<ov::op::v7::Gelu>(ಒಳಹರಿವು, ov::op::Gelu_Approximation_ಬಹುಲಕ::TANH);
    ಹಿಂತಿರುಗಿ rename_outputs_with_suffix({res}, context.get_name());
}

Output_ಸದಿಶ translate_unary_gelu_erf(ಸ್ಥಿರಾಂಕ ಗಂಟು_Context & context) {
    ಅಂಕೆ_inputs_check(context, 1, 1);
    ಸ್ವಯಂಚಾಲಿತ ಒಳಹರಿವು = process_view_ಒಳಹರಿವು_new(context, 0);
    ಸ್ವಯಂಚಾಲಿತ res = std::make_shared<ov::op::v7::Gelu>(ಒಳಹರಿವು, ov::op::Gelu_Approximation_ಬಹುಲಕ::ERF);
    ಹಿಂತಿರುಗಿ rename_outputs_with_suffix({res}, context.get_name());
}

Output_ಸದಿಶ translate_unary_gelu_ಶೀಘ್ರ(ಸ್ಥಿರಾಂಕ ಗಂಟು_Context & context) {
    ಅಂಕೆ_inputs_check(context, 1, 1);
    ಸ್ವಯಂಚಾಲಿತ ಒಳಹರಿವು = process_view_ಒಳಹರಿವು_new(context, 0);
    ಸ್ವಯಂಚಾಲಿತ scale = ov::op::v0::Constant::create(ಒಳಹರಿವು.get_element_type(), ov::Shape{}, {1.702f});
    ಸ್ವಯಂಚಾಲಿತ mul = std::make_shared<ov::op::v1::Multiply>(ಒಳಹರಿವು, scale);
    ಸ್ವಯಂಚಾಲಿತ sig = std::make_shared<ov::op::v0::Sigmoid>(mul);
    ಸ್ವಯಂಚಾಲಿತ res = std::make_shared<ov::op::v1::Multiply>(ಒಳಹರಿವು, sig);
    ಹಿಂತಿರುಗಿ rename_outputs_with_suffix({res}, context.get_name());
}

Output_ಸದಿಶ translate_unary_elu(ಸ್ಥಿರಾಂಕ ಗಂಟು_Context & context) {
    ಅಂಕೆ_inputs_check(context, 1, 1);
    ಸ್ವಯಂಚಾಲಿತ ಒಳಹರಿವು = process_view_ಒಳಹರಿವು_new(context, 0);
    ಸ್ವಯಂಚಾಲಿತ res = std::make_shared<ov::op::v0::Elu>(ಒಳಹರಿವು, 1.0);
    ಹಿಂತಿರುಗಿ rename_outputs_with_suffix({res}, context.get_name());
}

Output_ಸದಿಶ translate_unary_hardsigmoid(ಸ್ಥಿರಾಂಕ ಗಂಟು_Context & context) {
    ಅಂಕೆ_inputs_check(context, 1, 1);
    // compute in f32 like the ggml reference: 1/6 is not exact in f16/bf16 (NPU cannot take the f32 path)
    ಸ್ವಯಂಚಾಲಿತ ಒಳಹರಿವು = process_view_ಒಳಹರಿವು_new(context, 0);
    ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ type = ggml_openvino_is_npu() ? ಒಳಹರಿವು.get_element_type() : ov::element::f32;
    ov::Output<ov::Node> x = ಒಳಹರಿವು;
    ಆದರೆ (type != ಒಳಹರಿವು.get_element_type()) {
        x = std::make_shared<ov::op::v0::Convert>(ಒಳಹರಿವು, type);
    }
    ಸ್ವಯಂಚಾಲಿತ alpha = ov::op::v0::Constant::create(type, ov::Shape{}, {1.0f / 6.0f});
    ಸ್ವಯಂಚಾಲಿತ beta = ov::op::v0::Constant::create(type, ov::Shape{}, {0.5f});
    ov::Output<ov::Node> res = std::make_shared<ov::op::v0::HardSigmoid>(x, alpha, beta);
    ಆದರೆ (type != ಒಳಹರಿವು.get_element_type()) {
        res = std::make_shared<ov::op::v0::Convert>(res, ಒಳಹರಿವು.get_element_type());
    }
    ಹಿಂತಿರುಗಿ rename_outputs_with_suffix({res}, context.get_name());
}

Output_ಸದಿಶ translate_unary_ಹೆಜ್ಜೆ(ಸ್ಥಿರಾಂಕ ಗಂಟು_Context & context) {
    ಅಂಕೆ_inputs_check(context, 1, 1);
    ಸ್ವಯಂಚಾಲಿತ ಒಳಹರಿವು = process_view_ಒಳಹರಿವು_new(context, 0);
    ಸ್ವಯಂಚಾಲಿತ zero = ov::op::v0::Constant::create(ಒಳಹರಿವು.get_element_type(), ov::Shape{}, {0.0f});
    ಸ್ವಯಂಚಾಲಿತ cond = std::make_shared<ov::op::v1::Greater>(ಒಳಹರಿವು, zero);
    ಸ್ವಯಂಚಾಲಿತ res = std::make_shared<ov::op::v0::Convert>(cond, ಒಳಹರಿವು.get_element_type());
    ಹಿಂತಿರುಗಿ rename_outputs_with_suffix({res}, context.get_name());
}

Output_ಸದಿಶ translate_unary_ಸುತ್ತು(ಸ್ಥಿರಾಂಕ ಗಂಟು_Context & context) {
    ಅಂಕೆ_inputs_check(context, 1, 1);
    ಸ್ವಯಂಚಾಲಿತ ಒಳಹರಿವು = process_view_ಒಳಹರಿವು_new(context, 0);
    ಸ್ವಯಂಚಾಲಿತ res = std::make_shared<ov::op::v5::ಸುತ್ತು>(ಒಳಹರಿವು, ov::op::v5::ಸುತ್ತು::ಸುತ್ತು_ಬಹುಲಕ::HALF_AWAY_FROM_ZERO);
    ಹಿಂತಿರುಗಿ rename_outputs_with_suffix({res}, context.get_name());
}

Output_ಸದಿಶ translate_unary_expm1(ಸ್ಥಿರಾಂಕ ಗಂಟು_Context & context) {
    ಅಂಕೆ_inputs_check(context, 1, 1);
    // compute in f32 like the ggml reference: exp(x) - 1 in f16 loses the small-x digits (NPU cannot take the f32 path)
    ಸ್ವಯಂಚಾಲಿತ ಒಳಹರಿವು = process_view_ಒಳಹರಿವು_new(context, 0);
    ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ type = ggml_openvino_is_npu() ? ಒಳಹರಿವು.get_element_type() : ov::element::f32;
    ov::Output<ov::Node> x = ಒಳಹರಿವು;
    ಆದರೆ (type != ಒಳಹರಿವು.get_element_type()) {
        x = std::make_shared<ov::op::v0::Convert>(ಒಳಹರಿವು, type);
    }
    ಸ್ವಯಂಚಾಲಿತ ಘಾತಾಂಕ = std::make_shared<ov::op::v0::Exp>(x);
    ಸ್ವಯಂಚಾಲಿತ one = ov::op::v0::Constant::create(type, ov::Shape{}, {1.0f});
    ov::Output<ov::Node> res = std::make_shared<ov::op::v1::ಕಳೆ>(ಘಾತಾಂಕ, one);
    ಆದರೆ (type != ಒಳಹರಿವು.get_element_type()) {
        res = std::make_shared<ov::op::v0::Convert>(res, ಒಳಹರಿವು.get_element_type());
    }
    ಹಿಂತಿರುಗಿ rename_outputs_with_suffix({res}, context.get_name());
}

Output_ಸದಿಶ translate_unary_softplus(ಸ್ಥಿರಾಂಕ ಗಂಟು_Context & context) {
    ಅಂಕೆ_inputs_check(context, 1, 1);

    ಆದರೆ (ggml_openvino_getenv_int("GGML_OPENVINO_NATIVE_SOFTPLUS") != 0) {
        ಹಿಂತಿರುಗಿ translate_1to1_ಹೊಂದಿಕೆ_1_ಒಳಹರಿವು<ov::op::v4::SoftPlus>(context);
    }

    ಸ್ವಯಂಚಾಲಿತ ಒಳಹರಿವು = process_view_ಒಳಹರಿವು_new(context, 0);
    ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ element_type = ಒಳಹರಿವು.get_element_type();
    ಸ್ವಯಂಚಾಲಿತ one = ov::op::v0::Constant::create(element_type, ov::Shape{}, {1.0f});

    ಸ್ವಯಂಚಾಲಿತ ಧನಾತ್ಮಕ = std::make_shared<ov::op::v0::Relu>(ಒಳಹರಿವು);
    ಸ್ವಯಂಚಾಲಿತ ನಿರಪೇಕ್ಷ = std::make_shared<ov::op::v0::Abs>(ಒಳಹರಿವು);
    ಸ್ವಯಂಚಾಲಿತ neg_abs = std::make_shared<ov::op::v0::Negative>(ನಿರಪೇಕ್ಷ);
    ಸ್ವಯಂಚಾಲಿತ exp_neg_abs = std::make_shared<ov::op::v0::Exp>(neg_abs);
    ಸ್ವಯಂಚಾಲಿತ log_ಪದ = std::make_shared<ov::op::v0::Log>(std::make_shared<ov::op::v1::ಕೂಡಿಸು>(one, exp_neg_abs));
    ಸ್ವಯಂಚಾಲಿತ res = std::make_shared<ov::op::v1::ಕೂಡಿಸು>(ಧನಾತ್ಮಕ, log_ಪದ);

    ಹಿಂತಿರುಗಿ rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
