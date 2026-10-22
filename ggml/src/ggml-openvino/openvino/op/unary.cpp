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

Output_सदिशम् translate_unary_gelu(ध्रुव ग्रन्थिः_Context & context) {
    अङ्कः_inputs_check(context, 1, 1);
    स्वचालित निवेशः = process_view_निवेशः_new(context, 0);
    स्वचालित res = std::make_shared<ov::op::v7::Gelu>(निवेशः, ov::op::Gelu_Approximation_बहुलकम्::TANH);
    प्रत्यागम rename_outputs_with_suffix({res}, context.get_name());
}

Output_सदिशम् translate_unary_gelu_erf(ध्रुव ग्रन्थिः_Context & context) {
    अङ्कः_inputs_check(context, 1, 1);
    स्वचालित निवेशः = process_view_निवेशः_new(context, 0);
    स्वचालित res = std::make_shared<ov::op::v7::Gelu>(निवेशः, ov::op::Gelu_Approximation_बहुलकम्::ERF);
    प्रत्यागम rename_outputs_with_suffix({res}, context.get_name());
}

Output_सदिशम् translate_unary_gelu_शीघ्र(ध्रुव ग्रन्थिः_Context & context) {
    अङ्कः_inputs_check(context, 1, 1);
    स्वचालित निवेशः = process_view_निवेशः_new(context, 0);
    स्वचालित scale = ov::op::v0::Constant::create(निवेशः.get_element_type(), ov::Shape{}, {1.702f});
    स्वचालित mul = std::make_shared<ov::op::v1::Multiply>(निवेशः, scale);
    स्वचालित sig = std::make_shared<ov::op::v0::Sigmoid>(mul);
    स्वचालित res = std::make_shared<ov::op::v1::Multiply>(निवेशः, sig);
    प्रत्यागम rename_outputs_with_suffix({res}, context.get_name());
}

Output_सदिशम् translate_unary_elu(ध्रुव ग्रन्थिः_Context & context) {
    अङ्कः_inputs_check(context, 1, 1);
    स्वचालित निवेशः = process_view_निवेशः_new(context, 0);
    स्वचालित res = std::make_shared<ov::op::v0::Elu>(निवेशः, 1.0);
    प्रत्यागम rename_outputs_with_suffix({res}, context.get_name());
}

Output_सदिशम् translate_unary_hardsigmoid(ध्रुव ग्रन्थिः_Context & context) {
    अङ्कः_inputs_check(context, 1, 1);
    // compute in f32 like the ggml reference: 1/6 is not exact in f16/bf16 (NPU cannot take the f32 path)
    स्वचालित निवेशः = process_view_निवेशः_new(context, 0);
    ध्रुव स्वचालित type = ggml_openvino_is_npu() ? निवेशः.get_element_type() : ov::element::f32;
    ov::Output<ov::Node> x = निवेशः;
    यदि (type != निवेशः.get_element_type()) {
        x = std::make_shared<ov::op::v0::Convert>(निवेशः, type);
    }
    स्वचालित alpha = ov::op::v0::Constant::create(type, ov::Shape{}, {1.0f / 6.0f});
    स्वचालित beta = ov::op::v0::Constant::create(type, ov::Shape{}, {0.5f});
    ov::Output<ov::Node> res = std::make_shared<ov::op::v0::HardSigmoid>(x, alpha, beta);
    यदि (type != निवेशः.get_element_type()) {
        res = std::make_shared<ov::op::v0::Convert>(res, निवेशः.get_element_type());
    }
    प्रत्यागम rename_outputs_with_suffix({res}, context.get_name());
}

Output_सदिशम् translate_unary_चरणः(ध्रुव ग्रन्थिः_Context & context) {
    अङ्कः_inputs_check(context, 1, 1);
    स्वचालित निवेशः = process_view_निवेशः_new(context, 0);
    स्वचालित zero = ov::op::v0::Constant::create(निवेशः.get_element_type(), ov::Shape{}, {0.0f});
    स्वचालित cond = std::make_shared<ov::op::v1::Greater>(निवेशः, zero);
    स्वचालित res = std::make_shared<ov::op::v0::Convert>(cond, निवेशः.get_element_type());
    प्रत्यागम rename_outputs_with_suffix({res}, context.get_name());
}

Output_सदिशम् translate_unary_आवर्तनम्(ध्रुव ग्रन्थिः_Context & context) {
    अङ्कः_inputs_check(context, 1, 1);
    स्वचालित निवेशः = process_view_निवेशः_new(context, 0);
    स्वचालित res = std::make_shared<ov::op::v5::आवर्तनम्>(निवेशः, ov::op::v5::आवर्तनम्::आवर्तनम्_बहुलकम्::HALF_AWAY_FROM_ZERO);
    प्रत्यागम rename_outputs_with_suffix({res}, context.get_name());
}

Output_सदिशम् translate_unary_expm1(ध्रुव ग्रन्थिः_Context & context) {
    अङ्कः_inputs_check(context, 1, 1);
    // compute in f32 like the ggml reference: exp(x) - 1 in f16 loses the small-x digits (NPU cannot take the f32 path)
    स्वचालित निवेशः = process_view_निवेशः_new(context, 0);
    ध्रुव स्वचालित type = ggml_openvino_is_npu() ? निवेशः.get_element_type() : ov::element::f32;
    ov::Output<ov::Node> x = निवेशः;
    यदि (type != निवेशः.get_element_type()) {
        x = std::make_shared<ov::op::v0::Convert>(निवेशः, type);
    }
    स्वचालित घातांकः = std::make_shared<ov::op::v0::Exp>(x);
    स्वचालित one = ov::op::v0::Constant::create(type, ov::Shape{}, {1.0f});
    ov::Output<ov::Node> res = std::make_shared<ov::op::v1::व्यवकलय>(घातांकः, one);
    यदि (type != निवेशः.get_element_type()) {
        res = std::make_shared<ov::op::v0::Convert>(res, निवेशः.get_element_type());
    }
    प्रत्यागम rename_outputs_with_suffix({res}, context.get_name());
}

Output_सदिशम् translate_unary_softplus(ध्रुव ग्रन्थिः_Context & context) {
    अङ्कः_inputs_check(context, 1, 1);

    यदि (ggml_openvino_getenv_int("GGML_OPENVINO_NATIVE_SOFTPLUS") != 0) {
        प्रत्यागम translate_1to1_मेलनम्_1_निवेशः<ov::op::v4::SoftPlus>(context);
    }

    स्वचालित निवेशः = process_view_निवेशः_new(context, 0);
    ध्रुव स्वचालित element_type = निवेशः.get_element_type();
    स्वचालित one = ov::op::v0::Constant::create(element_type, ov::Shape{}, {1.0f});

    स्वचालित धनात्मकम् = std::make_shared<ov::op::v0::Relu>(निवेशः);
    स्वचालित निरपेक्षम् = std::make_shared<ov::op::v0::Abs>(निवेशः);
    स्वचालित neg_abs = std::make_shared<ov::op::v0::Negative>(निरपेक्षम्);
    स्वचालित exp_neg_abs = std::make_shared<ov::op::v0::Exp>(neg_abs);
    स्वचालित log_पदांशः = std::make_shared<ov::op::v0::Log>(std::make_shared<ov::op::v1::सङ्कलय>(one, exp_neg_abs));
    स्वचालित res = std::make_shared<ov::op::v1::सङ्कलय>(धनात्मकम्, log_पदांशः);

    प्रत्यागम rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
