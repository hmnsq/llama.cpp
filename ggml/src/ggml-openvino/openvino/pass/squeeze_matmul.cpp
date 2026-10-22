#ضمن "squeeze_matmul.h"

#ضمن <openvino/core/graph_util.hpp>
#ضمن <openvino/core/rt_info.hpp>
#ضمن <openvino/core/type.hpp>
#ضمن <openvino/op/constant.hpp>
#ضمن <openvino/op/matmul.hpp>
#ضمن <openvino/op/squeeze.hpp>
#ضمن <openvino/op/unsqueeze.hpp>
#ضمن <openvino/pass/pattern/op/label.hpp>
#ضمن <openvino/pass/pattern/op/pattern.hpp>
#ضمن <openvino/pass/pattern/op/wrap_type.hpp>

نطاق opp = ov::pass::pattern;

نطاق ov {
نطاق frontend {
نطاق ggml {
نطاق pass {

// For quantized models, NPUW expects the activation to be 3d in DQ(DynamicQuantization) opt, e.g. DQMatMulGQ2i
SqueezeMatmul::SqueezeMatmul() {
    خودڪار m_act = opp::any_داخلا();
    خودڪار m_wei = opp::any_داخلا();
    خودڪار m_matmul = opp::wrap_type<ov::op::v0::MatMul>({m_act, m_wei});

    ثابت خودڪار callback = [=](ov::pass::pattern::Matcher & m) {
        ثابت خودڪار & نمونو_map = m.get_نمونو_قدر_map();
        خودڪار matmul_ڳنڍ =
            ov::as_type_ptr<ov::op::v0::MatMul>(نمونو_map.at(m_matmul).get_ڳنڍ_shared_ptr());
        خودڪار act = نمونو_map.at(m_act);
        خودڪار wei = نمونو_map.at(m_wei);
        خودڪار act_shape = act.get_partial_shape();
        خودڪار wei_shape = wei.get_partial_shape();
        إذا (act_shape.rank().is_dynamic() || wei_shape.rank().is_dynamic()) {
            أرجع خطأ;
        }
        إذا (act_shape.rank().get_ڊيگهائي() == 4 && wei_shape.rank().get_ڊيگهائي() == 2) {
            خودڪار axis = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {0});
            خودڪار squeezed_act = std::make_shared<ov::op::v0::Squeeze>(act, axis);
            خودڪار new_matmul = std::make_shared<ov::op::v0::MatMul>(squeezed_act, wei, matmul_ڳنڍ->get_transpose_a(),
                                                                   matmul_ڳنڍ->get_transpose_b());
            خودڪار unsqueezed_output = std::make_shared<ov::op::v0::Unsqueeze>(new_matmul, axis);
            unsqueezed_output->سيٽ_friendly_name(matmul_ڳنڍ->get_friendly_name());
            ov::copy_runtime_info(matmul_ڳنڍ, {squeezed_act, new_matmul, unsqueezed_output});
            ov::replace_ڳنڍ(matmul_ڳنڍ, unsqueezed_output);
            أرجع سچ;
        }
        أرجع خطأ;
    };

    رجسٽر_matcher(std::make_shared<ov::pass::pattern::Matcher>(m_matmul, "ov::frontend::ggml::pass::SqueezeMatmul"),
                     callback);
}

}  // namespace pass
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
