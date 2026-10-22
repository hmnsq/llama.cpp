#شامل "../node_context.h"
#شامل "../op_table.h"
#شامل "../utils.h"
#شامل "ggml-openvino/ggml-openvino-extra.h"

#شامل <memory>
#شامل <openvino/op/abs.hpp>
#شامل <openvino/op/add.hpp>
#شامل <openvino/op/clamp.hpp>
#شامل <openvino/op/constant.hpp>
#شامل <openvino/op/convert.hpp>
#شامل <openvino/op/elu.hpp>
#شامل <openvino/op/exp.hpp>
#شامل <openvino/op/gelu.hpp>
#شامل <openvino/op/greater.hpp>
#شامل <openvino/op/hard_sigmoid.hpp>
#شامل <openvino/op/log.hpp>
#شامل <openvino/op/multiply.hpp>
#شامل <openvino/op/negative.hpp>
#شامل <openvino/op/relu.hpp>
#شامل <openvino/op/round.hpp>
#شامل <openvino/op/sigmoid.hpp>
#شامل <openvino/op/softplus.hpp>
#شامل <openvino/op/subtract.hpp>

نامخلا ov {
نامخلا frontend {
نامخلا ggml {
نامخلا op {

Output_سمتیہ translate_unary_gelu(ثابت گرہ_Context & context) {
    عدد_inputs_check(context, 1, 1);
    خودکار داخلہ = process_view_داخلہ_new(context, 0);
    خودکار res = std::make_shared<ov::op::v7::Gelu>(داخلہ, ov::op::Gelu_Approximation_منوال::TANH);
    واپس rename_outputs_with_suffix({res}, context.get_name());
}

Output_سمتیہ translate_unary_gelu_erf(ثابت گرہ_Context & context) {
    عدد_inputs_check(context, 1, 1);
    خودکار داخلہ = process_view_داخلہ_new(context, 0);
    خودکار res = std::make_shared<ov::op::v7::Gelu>(داخلہ, ov::op::Gelu_Approximation_منوال::ERF);
    واپس rename_outputs_with_suffix({res}, context.get_name());
}

Output_سمتیہ translate_unary_gelu_تیز(ثابت گرہ_Context & context) {
    عدد_inputs_check(context, 1, 1);
    خودکار داخلہ = process_view_داخلہ_new(context, 0);
    خودکار scale = ov::op::v0::Constant::create(داخلہ.get_element_type(), ov::Shape{}, {1.702f});
    خودکار mul = std::make_shared<ov::op::v1::Multiply>(داخلہ, scale);
    خودکار sig = std::make_shared<ov::op::v0::Sigmoid>(mul);
    خودکار res = std::make_shared<ov::op::v1::Multiply>(داخلہ, sig);
    واپس rename_outputs_with_suffix({res}, context.get_name());
}

Output_سمتیہ translate_unary_elu(ثابت گرہ_Context & context) {
    عدد_inputs_check(context, 1, 1);
    خودکار داخلہ = process_view_داخلہ_new(context, 0);
    خودکار res = std::make_shared<ov::op::v0::Elu>(داخلہ, 1.0);
    واپس rename_outputs_with_suffix({res}, context.get_name());
}

Output_سمتیہ translate_unary_hardsigmoid(ثابت گرہ_Context & context) {
    عدد_inputs_check(context, 1, 1);
    // compute in f32 like the ggml reference: 1/6 is not exact in f16/bf16 (NPU cannot take the f32 path)
    خودکار داخلہ = process_view_داخلہ_new(context, 0);
    ثابت خودکار type = ggml_openvino_is_npu() ? داخلہ.get_element_type() : ov::element::f32;
    ov::Output<ov::Node> x = داخلہ;
    اگر (type != داخلہ.get_element_type()) {
        x = std::make_shared<ov::op::v0::Convert>(داخلہ, type);
    }
    خودکار alpha = ov::op::v0::Constant::create(type, ov::Shape{}, {1.0f / 6.0f});
    خودکار beta = ov::op::v0::Constant::create(type, ov::Shape{}, {0.5f});
    ov::Output<ov::Node> res = std::make_shared<ov::op::v0::HardSigmoid>(x, alpha, beta);
    اگر (type != داخلہ.get_element_type()) {
        res = std::make_shared<ov::op::v0::Convert>(res, داخلہ.get_element_type());
    }
    واپس rename_outputs_with_suffix({res}, context.get_name());
}

Output_سمتیہ translate_unary_قدم(ثابت گرہ_Context & context) {
    عدد_inputs_check(context, 1, 1);
    خودکار داخلہ = process_view_داخلہ_new(context, 0);
    خودکار zero = ov::op::v0::Constant::create(داخلہ.get_element_type(), ov::Shape{}, {0.0f});
    خودکار cond = std::make_shared<ov::op::v1::Greater>(داخلہ, zero);
    خودکار res = std::make_shared<ov::op::v0::Convert>(cond, داخلہ.get_element_type());
    واپس rename_outputs_with_suffix({res}, context.get_name());
}

Output_سمتیہ translate_unary_باری(ثابت گرہ_Context & context) {
    عدد_inputs_check(context, 1, 1);
    خودکار داخلہ = process_view_داخلہ_new(context, 0);
    خودکار res = std::make_shared<ov::op::v5::باری>(داخلہ, ov::op::v5::باری::باری_منوال::HALF_AWAY_FROM_ZERO);
    واپس rename_outputs_with_suffix({res}, context.get_name());
}

Output_سمتیہ translate_unary_expm1(ثابت گرہ_Context & context) {
    عدد_inputs_check(context, 1, 1);
    // compute in f32 like the ggml reference: exp(x) - 1 in f16 loses the small-x digits (NPU cannot take the f32 path)
    خودکار داخلہ = process_view_داخلہ_new(context, 0);
    ثابت خودکار type = ggml_openvino_is_npu() ? داخلہ.get_element_type() : ov::element::f32;
    ov::Output<ov::Node> x = داخلہ;
    اگر (type != داخلہ.get_element_type()) {
        x = std::make_shared<ov::op::v0::Convert>(داخلہ, type);
    }
    خودکار قوت_نما = std::make_shared<ov::op::v0::Exp>(x);
    خودکار one = ov::op::v0::Constant::create(type, ov::Shape{}, {1.0f});
    ov::Output<ov::Node> res = std::make_shared<ov::op::v1::تفریق>(قوت_نما, one);
    اگر (type != داخلہ.get_element_type()) {
        res = std::make_shared<ov::op::v0::Convert>(res, داخلہ.get_element_type());
    }
    واپس rename_outputs_with_suffix({res}, context.get_name());
}

Output_سمتیہ translate_unary_softplus(ثابت گرہ_Context & context) {
    عدد_inputs_check(context, 1, 1);

    اگر (ggml_openvino_getenv_int("GGML_OPENVINO_NATIVE_SOFTPLUS") != 0) {
        واپس translate_1to1_مطابقت_1_داخلہ<ov::op::v4::SoftPlus>(context);
    }

    خودکار داخلہ = process_view_داخلہ_new(context, 0);
    ثابت خودکار element_type = داخلہ.get_element_type();
    خودکار one = ov::op::v0::Constant::create(element_type, ov::Shape{}, {1.0f});

    خودکار مثبت = std::make_shared<ov::op::v0::Relu>(داخلہ);
    خودکار مطلق_قدر = std::make_shared<ov::op::v0::Abs>(داخلہ);
    خودکار neg_abs = std::make_shared<ov::op::v0::Negative>(مطلق_قدر);
    خودکار exp_neg_abs = std::make_shared<ov::op::v0::Exp>(neg_abs);
    خودکار log_رکن = std::make_shared<ov::op::v0::Log>(std::make_shared<ov::op::v1::جوڑو>(one, exp_neg_abs));
    خودکار res = std::make_shared<ov::op::v1::جوڑو>(مثبت, log_رکن);

    واپس rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
