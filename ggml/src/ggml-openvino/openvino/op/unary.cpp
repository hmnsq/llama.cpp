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

OutputVector translate_unary_gelu(स्थिरांक गाठ_Context & context) {
    अंक_inputs_check(context, 1, 1);
    स्वतः आदान = process_view_आदान_new(context, 0);
    स्वतः res = std::make_shared<ov::op::v7::Gelu>(आदान, ov::op::GeluApproximationMode::TANH);
    परत rename_outputs_with_suffix({res}, context.get_name());
}

OutputVector translate_unary_gelu_erf(स्थिरांक गाठ_Context & context) {
    अंक_inputs_check(context, 1, 1);
    स्वतः आदान = process_view_आदान_new(context, 0);
    स्वतः res = std::make_shared<ov::op::v7::Gelu>(आदान, ov::op::GeluApproximationMode::ERF);
    परत rename_outputs_with_suffix({res}, context.get_name());
}

OutputVector translate_unary_gelu_quick(स्थिरांक गाठ_Context & context) {
    अंक_inputs_check(context, 1, 1);
    स्वतः आदान = process_view_आदान_new(context, 0);
    स्वतः scale = ov::op::v0::Constant::create(आदान.get_element_type(), ov::Shape{}, {1.702f});
    स्वतः mul = std::make_shared<ov::op::v1::Multiply>(आदान, scale);
    स्वतः sig = std::make_shared<ov::op::v0::Sigmoid>(mul);
    स्वतः res = std::make_shared<ov::op::v1::Multiply>(आदान, sig);
    परत rename_outputs_with_suffix({res}, context.get_name());
}

OutputVector translate_unary_elu(स्थिरांक गाठ_Context & context) {
    अंक_inputs_check(context, 1, 1);
    स्वतः आदान = process_view_आदान_new(context, 0);
    स्वतः res = std::make_shared<ov::op::v0::Elu>(आदान, 1.0);
    परत rename_outputs_with_suffix({res}, context.get_name());
}

OutputVector translate_unary_hardsigmoid(स्थिरांक गाठ_Context & context) {
    अंक_inputs_check(context, 1, 1);
    // compute in f32 like the ggml reference: 1/6 is not exact in f16/bf16 (NPU cannot take the f32 path)
    स्वतः आदान = process_view_आदान_new(context, 0);
    स्थिरांक स्वतः type = ggml_openvino_is_npu() ? आदान.get_element_type() : ov::element::f32;
    ov::Output<ov::Node> x = आदान;
    जर (type != आदान.get_element_type()) {
        x = std::make_shared<ov::op::v0::Convert>(आदान, type);
    }
    स्वतः alpha = ov::op::v0::Constant::create(type, ov::Shape{}, {1.0f / 6.0f});
    स्वतः beta = ov::op::v0::Constant::create(type, ov::Shape{}, {0.5f});
    ov::Output<ov::Node> res = std::make_shared<ov::op::v0::HardSigmoid>(x, alpha, beta);
    जर (type != आदान.get_element_type()) {
        res = std::make_shared<ov::op::v0::Convert>(res, आदान.get_element_type());
    }
    परत rename_outputs_with_suffix({res}, context.get_name());
}

OutputVector translate_unary_step(स्थिरांक गाठ_Context & context) {
    अंक_inputs_check(context, 1, 1);
    स्वतः आदान = process_view_आदान_new(context, 0);
    स्वतः zero = ov::op::v0::Constant::create(आदान.get_element_type(), ov::Shape{}, {0.0f});
    स्वतः cond = std::make_shared<ov::op::v1::Greater>(आदान, zero);
    स्वतः res = std::make_shared<ov::op::v0::Convert>(cond, आदान.get_element_type());
    परत rename_outputs_with_suffix({res}, context.get_name());
}

OutputVector translate_unary_round(स्थिरांक गाठ_Context & context) {
    अंक_inputs_check(context, 1, 1);
    स्वतः आदान = process_view_आदान_new(context, 0);
    स्वतः res = std::make_shared<ov::op::v5::Round>(आदान, ov::op::v5::Round::RoundMode::HALF_AWAY_FROM_ZERO);
    परत rename_outputs_with_suffix({res}, context.get_name());
}

OutputVector translate_unary_expm1(स्थिरांक गाठ_Context & context) {
    अंक_inputs_check(context, 1, 1);
    // compute in f32 like the ggml reference: exp(x) - 1 in f16 loses the small-x digits (NPU cannot take the f32 path)
    स्वतः आदान = process_view_आदान_new(context, 0);
    स्थिरांक स्वतः type = ggml_openvino_is_npu() ? आदान.get_element_type() : ov::element::f32;
    ov::Output<ov::Node> x = आदान;
    जर (type != आदान.get_element_type()) {
        x = std::make_shared<ov::op::v0::Convert>(आदान, type);
    }
    स्वतः exp = std::make_shared<ov::op::v0::Exp>(x);
    स्वतः one = ov::op::v0::Constant::create(type, ov::Shape{}, {1.0f});
    ov::Output<ov::Node> res = std::make_shared<ov::op::v1::Subtract>(exp, one);
    जर (type != आदान.get_element_type()) {
        res = std::make_shared<ov::op::v0::Convert>(res, आदान.get_element_type());
    }
    परत rename_outputs_with_suffix({res}, context.get_name());
}

OutputVector translate_unary_softplus(स्थिरांक गाठ_Context & context) {
    अंक_inputs_check(context, 1, 1);

    जर (ggml_openvino_getenv_int("GGML_OPENVINO_NATIVE_SOFTPLUS") != 0) {
        परत translate_1to1_match_1_आदान<ov::op::v4::SoftPlus>(context);
    }

    स्वतः आदान = process_view_आदान_new(context, 0);
    स्थिरांक स्वतः element_type = आदान.get_element_type();
    स्वतः one = ov::op::v0::Constant::create(element_type, ov::Shape{}, {1.0f});

    स्वतः positive = std::make_shared<ov::op::v0::Relu>(आदान);
    स्वतः असल = std::make_shared<ov::op::v0::Abs>(आदान);
    स्वतः neg_abs = std::make_shared<ov::op::v0::Negative>(असल);
    स्वतः exp_neg_abs = std::make_shared<ov::op::v0::Exp>(neg_abs);
    स्वतः log_term = std::make_shared<ov::op::v0::Log>(std::make_shared<ov::op::v1::Add>(one, exp_neg_abs));
    स्वतः res = std::make_shared<ov::op::v1::Add>(positive, log_term);

    परत rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
