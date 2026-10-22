#అన్తర్భావ "../node_context.h"
#అన్తర్భావ "../op_table.h"
#అన్తర్భావ "../utils.h"
#అన్తర్భావ "ggml-openvino/ggml-openvino-extra.h"

#అన్తర్భావ <memory>
#అన్తర్భావ <openvino/op/abs.hpp>
#అన్తర్భావ <openvino/op/add.hpp>
#అన్తర్భావ <openvino/op/clamp.hpp>
#అన్తర్భావ <openvino/op/constant.hpp>
#అన్తర్భావ <openvino/op/convert.hpp>
#అన్తర్భావ <openvino/op/elu.hpp>
#అన్తర్భావ <openvino/op/exp.hpp>
#అన్తర్భావ <openvino/op/gelu.hpp>
#అన్తర్భావ <openvino/op/greater.hpp>
#అన్తర్భావ <openvino/op/hard_sigmoid.hpp>
#అన్తర్భావ <openvino/op/log.hpp>
#అన్తర్భావ <openvino/op/multiply.hpp>
#అన్తర్భావ <openvino/op/negative.hpp>
#అన్తర్భావ <openvino/op/relu.hpp>
#అన్తర్భావ <openvino/op/round.hpp>
#అన్తర్భావ <openvino/op/sigmoid.hpp>
#అన్తర్భావ <openvino/op/softplus.hpp>
#అన్తర్భావ <openvino/op/subtract.hpp>

నామాకాశ ov {
నామాకాశ frontend {
నామాకాశ ggml {
నామాకాశ op {

Output_సదిశ translate_unary_gelu(స్థిరరాశి కణుపు_Context & context) {
    అంకె_inputs_check(context, 1, 1);
    స్వయంచాలక ఆదానం = process_view_ఆదానం_new(context, 0);
    స్వయంచాలక res = std::make_shared<ov::op::v7::Gelu>(ఆదానం, ov::op::Gelu_Approximation_బహుళకం::TANH);
    తిరిగి rename_outputs_with_suffix({res}, context.get_name());
}

Output_సదిశ translate_unary_gelu_erf(స్థిరరాశి కణుపు_Context & context) {
    అంకె_inputs_check(context, 1, 1);
    స్వయంచాలక ఆదానం = process_view_ఆదానం_new(context, 0);
    స్వయంచాలక res = std::make_shared<ov::op::v7::Gelu>(ఆదానం, ov::op::Gelu_Approximation_బహుళకం::ERF);
    తిరిగి rename_outputs_with_suffix({res}, context.get_name());
}

Output_సదిశ translate_unary_gelu_శీఘ్ర(స్థిరరాశి కణుపు_Context & context) {
    అంకె_inputs_check(context, 1, 1);
    స్వయంచాలక ఆదానం = process_view_ఆదానం_new(context, 0);
    స్వయంచాలక scale = ov::op::v0::Constant::create(ఆదానం.get_element_type(), ov::Shape{}, {1.702f});
    స్వయంచాలక mul = std::make_shared<ov::op::v1::Multiply>(ఆదానం, scale);
    స్వయంచాలక sig = std::make_shared<ov::op::v0::Sigmoid>(mul);
    స్వయంచాలక res = std::make_shared<ov::op::v1::Multiply>(ఆదానం, sig);
    తిరిగి rename_outputs_with_suffix({res}, context.get_name());
}

Output_సదిశ translate_unary_elu(స్థిరరాశి కణుపు_Context & context) {
    అంకె_inputs_check(context, 1, 1);
    స్వయంచాలక ఆదానం = process_view_ఆదానం_new(context, 0);
    స్వయంచాలక res = std::make_shared<ov::op::v0::Elu>(ఆదానం, 1.0);
    తిరిగి rename_outputs_with_suffix({res}, context.get_name());
}

Output_సదిశ translate_unary_hardsigmoid(స్థిరరాశి కణుపు_Context & context) {
    అంకె_inputs_check(context, 1, 1);
    // compute in f32 like the ggml reference: 1/6 is not exact in f16/bf16 (NPU cannot take the f32 path)
    స్వయంచాలక ఆదానం = process_view_ఆదానం_new(context, 0);
    స్థిరరాశి స్వయంచాలక type = ggml_openvino_is_npu() ? ఆదానం.get_element_type() : ov::element::f32;
    ov::Output<ov::Node> x = ఆదానం;
    ఒకవేళ (type != ఆదానం.get_element_type()) {
        x = std::make_shared<ov::op::v0::Convert>(ఆదానం, type);
    }
    స్వయంచాలక alpha = ov::op::v0::Constant::create(type, ov::Shape{}, {1.0f / 6.0f});
    స్వయంచాలక beta = ov::op::v0::Constant::create(type, ov::Shape{}, {0.5f});
    ov::Output<ov::Node> res = std::make_shared<ov::op::v0::HardSigmoid>(x, alpha, beta);
    ఒకవేళ (type != ఆదానం.get_element_type()) {
        res = std::make_shared<ov::op::v0::Convert>(res, ఆదానం.get_element_type());
    }
    తిరిగి rename_outputs_with_suffix({res}, context.get_name());
}

Output_సదిశ translate_unary_అడుగు(స్థిరరాశి కణుపు_Context & context) {
    అంకె_inputs_check(context, 1, 1);
    స్వయంచాలక ఆదానం = process_view_ఆదానం_new(context, 0);
    స్వయంచాలక zero = ov::op::v0::Constant::create(ఆదానం.get_element_type(), ov::Shape{}, {0.0f});
    స్వయంచాలక cond = std::make_shared<ov::op::v1::Greater>(ఆదానం, zero);
    స్వయంచాలక res = std::make_shared<ov::op::v0::Convert>(cond, ఆదానం.get_element_type());
    తిరిగి rename_outputs_with_suffix({res}, context.get_name());
}

Output_సదిశ translate_unary_విడత(స్థిరరాశి కణుపు_Context & context) {
    అంకె_inputs_check(context, 1, 1);
    స్వయంచాలక ఆదానం = process_view_ఆదానం_new(context, 0);
    స్వయంచాలక res = std::make_shared<ov::op::v5::విడత>(ఆదానం, ov::op::v5::విడత::విడత_బహుళకం::HALF_AWAY_FROM_ZERO);
    తిరిగి rename_outputs_with_suffix({res}, context.get_name());
}

Output_సదిశ translate_unary_expm1(స్థిరరాశి కణుపు_Context & context) {
    అంకె_inputs_check(context, 1, 1);
    // compute in f32 like the ggml reference: exp(x) - 1 in f16 loses the small-x digits (NPU cannot take the f32 path)
    స్వయంచాలక ఆదానం = process_view_ఆదానం_new(context, 0);
    స్థిరరాశి స్వయంచాలక type = ggml_openvino_is_npu() ? ఆదానం.get_element_type() : ov::element::f32;
    ov::Output<ov::Node> x = ఆదానం;
    ఒకవేళ (type != ఆదానం.get_element_type()) {
        x = std::make_shared<ov::op::v0::Convert>(ఆదానం, type);
    }
    స్వయంచాలక ఘాతాంకం = std::make_shared<ov::op::v0::Exp>(x);
    స్వయంచాలక one = ov::op::v0::Constant::create(type, ov::Shape{}, {1.0f});
    ov::Output<ov::Node> res = std::make_shared<ov::op::v1::వ్యవకలనం>(ఘాతాంకం, one);
    ఒకవేళ (type != ఆదానం.get_element_type()) {
        res = std::make_shared<ov::op::v0::Convert>(res, ఆదానం.get_element_type());
    }
    తిరిగి rename_outputs_with_suffix({res}, context.get_name());
}

Output_సదిశ translate_unary_softplus(స్థిరరాశి కణుపు_Context & context) {
    అంకె_inputs_check(context, 1, 1);

    ఒకవేళ (ggml_openvino_getenv_int("GGML_OPENVINO_NATIVE_SOFTPLUS") != 0) {
        తిరిగి translate_1to1_సరిపోలిక_1_ఆదానం<ov::op::v4::SoftPlus>(context);
    }

    స్వయంచాలక ఆదానం = process_view_ఆదానం_new(context, 0);
    స్థిరరాశి స్వయంచాలక element_type = ఆదానం.get_element_type();
    స్వయంచాలక one = ov::op::v0::Constant::create(element_type, ov::Shape{}, {1.0f});

    స్వయంచాలక ధనాత్మక = std::make_shared<ov::op::v0::Relu>(ఆదానం);
    స్వయంచాలక నిరపేక్ష = std::make_shared<ov::op::v0::Abs>(ఆదానం);
    స్వయంచాలక neg_abs = std::make_shared<ov::op::v0::Negative>(నిరపేక్ష);
    స్వయంచాలక exp_neg_abs = std::make_shared<ov::op::v0::Exp>(neg_abs);
    స్వయంచాలక log_పదం = std::make_shared<ov::op::v0::Log>(std::make_shared<ov::op::v1::కూడు>(one, exp_neg_abs));
    స్వయంచాలక res = std::make_shared<ov::op::v1::కూడు>(ధనాత్మక, log_పదం);

    తిరిగి rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
