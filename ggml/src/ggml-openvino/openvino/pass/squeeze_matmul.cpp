#ਅਨ੍ਤਰ੍ਭਾਵ "squeeze_matmul.h"

#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/core/graph_util.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/core/rt_info.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/core/type.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/constant.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/matmul.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/squeeze.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/unsqueeze.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/pass/pattern/op/label.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/pass/pattern/op/pattern.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/pass/pattern/op/wrap_type.hpp>

ਨਾਮਾਕਾਸ਼ opp = ov::pass::pattern;

ਨਾਮਾਕਾਸ਼ ov {
ਨਾਮਾਕਾਸ਼ frontend {
ਨਾਮਾਕਾਸ਼ ggml {
ਨਾਮਾਕਾਸ਼ pass {

// For quantized models, NPUW expects the activation to be 3d in DQ(DynamicQuantization) opt, e.g. DQMatMulGQ2i
SqueezeMatmul::SqueezeMatmul() {
    ਸਵੈਚਾਲਿਤ m_act = opp::any_ਨਿਵੇਸ਼();
    ਸਵੈਚਾਲਿਤ m_wei = opp::any_ਨਿਵੇਸ਼();
    ਸਵੈਚਾਲਿਤ m_matmul = opp::wrap_type<ov::op::v0::MatMul>({m_act, m_wei});

    ਅਚਲ ਸਵੈਚਾਲਿਤ callback = [=](ov::pass::pattern::Matcher & m) {
        ਅਚਲ ਸਵੈਚਾਲਿਤ & ਨਮੂਨਾ_map = m.get_ਨਮੂਨਾ_ਮੁੱਲ_map();
        ਸਵੈਚਾਲਿਤ matmul_ਗੰਢ =
            ov::as_type_ptr<ov::op::v0::MatMul>(ਨਮੂਨਾ_map.at(m_matmul).get_ਗੰਢ_shared_ptr());
        ਸਵੈਚਾਲਿਤ act = ਨਮੂਨਾ_map.at(m_act);
        ਸਵੈਚਾਲਿਤ wei = ਨਮੂਨਾ_map.at(m_wei);
        ਸਵੈਚਾਲਿਤ act_shape = act.get_partial_shape();
        ਸਵੈਚਾਲਿਤ wei_shape = wei.get_partial_shape();
        ਜੇ (act_shape.rank().is_dynamic() || wei_shape.rank().is_dynamic()) {
            ਵਾਪਸ ਝੂਠ;
        }
        ਜੇ (act_shape.rank().get_ਲੰਮਾਈ() == 4 && wei_shape.rank().get_ਲੰਮਾਈ() == 2) {
            ਸਵੈਚਾਲਿਤ axis = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {0});
            ਸਵੈਚਾਲਿਤ squeezed_act = std::make_shared<ov::op::v0::Squeeze>(act, axis);
            ਸਵੈਚਾਲਿਤ new_matmul = std::make_shared<ov::op::v0::MatMul>(squeezed_act, wei, matmul_ਗੰਢ->get_transpose_a(),
                                                                   matmul_ਗੰਢ->get_transpose_b());
            ਸਵੈਚਾਲਿਤ unsqueezed_output = std::make_shared<ov::op::v0::Unsqueeze>(new_matmul, axis);
            unsqueezed_output->ਸੈੱਟ_friendly_name(matmul_ਗੰਢ->get_friendly_name());
            ov::copy_runtime_info(matmul_ਗੰਢ, {squeezed_act, new_matmul, unsqueezed_output});
            ov::replace_ਗੰਢ(matmul_ਗੰਢ, unsqueezed_output);
            ਵਾਪਸ ਸੱਚ;
        }
        ਵਾਪਸ ਝੂਠ;
    };

    ਰਜਿਸਟਰ_matcher(std::make_shared<ov::pass::pattern::Matcher>(m_matmul, "ov::frontend::ggml::pass::SqueezeMatmul"),
                     callback);
}

}  // namespace pass
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
