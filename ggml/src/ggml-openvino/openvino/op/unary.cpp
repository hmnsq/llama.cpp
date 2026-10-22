#समावेश "../node_context.h"
#समावेश "../op_table.h"
#समावेश "../utils.h"
#समावेश "ggml-openvino/ggml-openvino-extra.h"

#समावेश <memory>
#समावेश <openvino/op/abs.hpp>
#समावेश <openvino/op/add.hpp>
#समावेश <openvino/op/clamp.hpp>
#समावेश <openvino/op/constant.hpp>
#समावेश <openvino/op/convert.hpp>
#समावेश <openvino/op/elu.hpp>
#समावेश <openvino/op/exp.hpp>
#समावेश <openvino/op/gelu.hpp>
#समावेश <openvino/op/greater.hpp>
#समावेश <openvino/op/hard_sigmoid.hpp>
#समावेश <openvino/op/log.hpp>
#समावेश <openvino/op/multiply.hpp>
#समावेश <openvino/op/negative.hpp>
#समावेश <openvino/op/relu.hpp>
#समावेश <openvino/op/round.hpp>
#समावेश <openvino/op/sigmoid.hpp>
#समावेश <openvino/op/softplus.hpp>
#समावेश <openvino/op/subtract.hpp>

namespace ov {
namespace frontend {
namespace ggml {
namespace op {

OutputVector translate_unary_gelu(स्थिर गाँठ_Context & context) {
    अंक_inputs_check(context, 1, 1);
    स्वतः निवेश = process_view_निवेश_new(context, 0);
    स्वतः res = std::make_shared<ov::op::v7::Gelu>(निवेश, ov::op::GeluApproximationMode::TANH);
    वापस rename_outputs_with_suffix({res}, context.get_name());
}

OutputVector translate_unary_gelu_erf(स्थिर गाँठ_Context & context) {
    अंक_inputs_check(context, 1, 1);
    स्वतः निवेश = process_view_निवेश_new(context, 0);
    स्वतः res = std::make_shared<ov::op::v7::Gelu>(निवेश, ov::op::GeluApproximationMode::ERF);
    वापस rename_outputs_with_suffix({res}, context.get_name());
}

OutputVector translate_unary_gelu_quick(स्थिर गाँठ_Context & context) {
    अंक_inputs_check(context, 1, 1);
    स्वतः निवेश = process_view_निवेश_new(context, 0);
    स्वतः scale = ov::op::v0::Constant::create(निवेश.get_element_type(), ov::Shape{}, {1.702f});
    स्वतः mul = std::make_shared<ov::op::v1::Multiply>(निवेश, scale);
    स्वतः sig = std::make_shared<ov::op::v0::Sigmoid>(mul);
    स्वतः res = std::make_shared<ov::op::v1::Multiply>(निवेश, sig);
    वापस rename_outputs_with_suffix({res}, context.get_name());
}

OutputVector translate_unary_elu(स्थिर गाँठ_Context & context) {
    अंक_inputs_check(context, 1, 1);
    स्वतः निवेश = process_view_निवेश_new(context, 0);
    स्वतः res = std::make_shared<ov::op::v0::Elu>(निवेश, 1.0);
    वापस rename_outputs_with_suffix({res}, context.get_name());
}

OutputVector translate_unary_hardsigmoid(स्थिर गाँठ_Context & context) {
    अंक_inputs_check(context, 1, 1);
    // compute in f32 like the ggml reference: 1/6 is not exact in f16/bf16 (NPU cannot take the f32 path)
    स्वतः निवेश = process_view_निवेश_new(context, 0);
    स्थिर स्वतः type = ggml_openvino_is_npu() ? निवेश.get_element_type() : ov::element::f32;
    ov::Output<ov::Node> x = निवेश;
    अगर (type != निवेश.get_element_type()) {
        x = std::make_shared<ov::op::v0::Convert>(निवेश, type);
    }
    स्वतः alpha = ov::op::v0::Constant::create(type, ov::Shape{}, {1.0f / 6.0f});
    स्वतः beta = ov::op::v0::Constant::create(type, ov::Shape{}, {0.5f});
    ov::Output<ov::Node> res = std::make_shared<ov::op::v0::HardSigmoid>(x, alpha, beta);
    अगर (type != निवेश.get_element_type()) {
        res = std::make_shared<ov::op::v0::Convert>(res, निवेश.get_element_type());
    }
    वापस rename_outputs_with_suffix({res}, context.get_name());
}

OutputVector translate_unary_step(स्थिर गाँठ_Context & context) {
    अंक_inputs_check(context, 1, 1);
    स्वतः निवेश = process_view_निवेश_new(context, 0);
    स्वतः zero = ov::op::v0::Constant::create(निवेश.get_element_type(), ov::Shape{}, {0.0f});
    स्वतः cond = std::make_shared<ov::op::v1::Greater>(निवेश, zero);
    स्वतः res = std::make_shared<ov::op::v0::Convert>(cond, निवेश.get_element_type());
    वापस rename_outputs_with_suffix({res}, context.get_name());
}

OutputVector translate_unary_round(स्थिर गाँठ_Context & context) {
    अंक_inputs_check(context, 1, 1);
    स्वतः निवेश = process_view_निवेश_new(context, 0);
    स्वतः res = std::make_shared<ov::op::v5::Round>(निवेश, ov::op::v5::Round::RoundMode::HALF_AWAY_FROM_ZERO);
    वापस rename_outputs_with_suffix({res}, context.get_name());
}

OutputVector translate_unary_expm1(स्थिर गाँठ_Context & context) {
    अंक_inputs_check(context, 1, 1);
    // compute in f32 like the ggml reference: exp(x) - 1 in f16 loses the small-x digits (NPU cannot take the f32 path)
    स्वतः निवेश = process_view_निवेश_new(context, 0);
    स्थिर स्वतः type = ggml_openvino_is_npu() ? निवेश.get_element_type() : ov::element::f32;
    ov::Output<ov::Node> x = निवेश;
    अगर (type != निवेश.get_element_type()) {
        x = std::make_shared<ov::op::v0::Convert>(निवेश, type);
    }
    स्वतः exp = std::make_shared<ov::op::v0::Exp>(x);
    स्वतः one = ov::op::v0::Constant::create(type, ov::Shape{}, {1.0f});
    ov::Output<ov::Node> res = std::make_shared<ov::op::v1::Subtract>(exp, one);
    अगर (type != निवेश.get_element_type()) {
        res = std::make_shared<ov::op::v0::Convert>(res, निवेश.get_element_type());
    }
    वापस rename_outputs_with_suffix({res}, context.get_name());
}

OutputVector translate_unary_softplus(स्थिर गाँठ_Context & context) {
    अंक_inputs_check(context, 1, 1);

    अगर (ggml_openvino_getenv_int("GGML_OPENVINO_NATIVE_SOFTPLUS") != 0) {
        वापस translate_1to1_match_1_निवेश<ov::op::v4::SoftPlus>(context);
    }

    स्वतः निवेश = process_view_निवेश_new(context, 0);
    स्थिर स्वतः element_type = निवेश.get_element_type();
    स्वतः one = ov::op::v0::Constant::create(element_type, ov::Shape{}, {1.0f});

    स्वतः positive = std::make_shared<ov::op::v0::Relu>(निवेश);
    स्वतः असल = std::make_shared<ov::op::v0::Abs>(निवेश);
    स्वतः neg_abs = std::make_shared<ov::op::v0::Negative>(असल);
    स्वतः exp_neg_abs = std::make_shared<ov::op::v0::Exp>(neg_abs);
    स्वतः log_term = std::make_shared<ov::op::v0::Log>(std::make_shared<ov::op::v1::Add>(one, exp_neg_abs));
    स्वतः res = std::make_shared<ov::op::v1::Add>(positive, log_term);

    वापस rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
