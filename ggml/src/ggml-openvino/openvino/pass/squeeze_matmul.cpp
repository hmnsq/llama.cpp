#ಅನ್ತರ್ಭಾವ "squeeze_matmul.h"

#ಅನ್ತರ್ಭಾವ <openvino/core/graph_util.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/core/rt_info.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/core/type.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/constant.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/matmul.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/squeeze.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/unsqueeze.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/pass/pattern/op/label.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/pass/pattern/op/pattern.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/pass/pattern/op/wrap_type.hpp>

ನಾಮಾಕಾಶ opp = ov::pass::pattern;

ನಾಮಾಕಾಶ ov {
ನಾಮಾಕಾಶ frontend {
ನಾಮಾಕಾಶ ggml {
ನಾಮಾಕಾಶ pass {

// For quantized models, NPUW expects the activation to be 3d in DQ(DynamicQuantization) opt, e.g. DQMatMulGQ2i
SqueezeMatmul::SqueezeMatmul() {
    ಸ್ವಯಂಚಾಲಿತ m_act = opp::any_ಒಳಹರಿವು();
    ಸ್ವಯಂಚಾಲಿತ m_wei = opp::any_ಒಳಹರಿವು();
    ಸ್ವಯಂಚಾಲಿತ m_matmul = opp::wrap_type<ov::op::v0::MatMul>({m_act, m_wei});

    ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ callback = [=](ov::pass::pattern::Matcher & m) {
        ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ & ನಮೂನೆ_map = m.get_ನಮೂನೆ_ಮೌಲ್ಯ_map();
        ಸ್ವಯಂಚಾಲಿತ matmul_ಗಂಟು =
            ov::as_type_ptr<ov::op::v0::MatMul>(ನಮೂನೆ_map.at(m_matmul).get_ಗಂಟು_shared_ptr());
        ಸ್ವಯಂಚಾಲಿತ act = ನಮೂನೆ_map.at(m_act);
        ಸ್ವಯಂಚಾಲಿತ wei = ನಮೂನೆ_map.at(m_wei);
        ಸ್ವಯಂಚಾಲಿತ act_shape = act.get_partial_shape();
        ಸ್ವಯಂಚಾಲಿತ wei_shape = wei.get_partial_shape();
        ಆದರೆ (act_shape.rank().is_dynamic() || wei_shape.rank().is_dynamic()) {
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }
        ಆದರೆ (act_shape.rank().get_ಉದ್ದಳತೆ() == 4 && wei_shape.rank().get_ಉದ್ದಳತೆ() == 2) {
            ಸ್ವಯಂಚಾಲಿತ axis = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {0});
            ಸ್ವಯಂಚಾಲಿತ squeezed_act = std::make_shared<ov::op::v0::Squeeze>(act, axis);
            ಸ್ವಯಂಚಾಲಿತ new_matmul = std::make_shared<ov::op::v0::MatMul>(squeezed_act, wei, matmul_ಗಂಟು->get_transpose_a(),
                                                                   matmul_ಗಂಟು->get_transpose_b());
            ಸ್ವಯಂಚಾಲಿತ unsqueezed_output = std::make_shared<ov::op::v0::Unsqueeze>(new_matmul, axis);
            unsqueezed_output->ಗಣ_friendly_name(matmul_ಗಂಟು->get_friendly_name());
            ov::copy_runtime_info(matmul_ಗಂಟು, {squeezed_act, new_matmul, unsqueezed_output});
            ov::replace_ಗಂಟು(matmul_ಗಂಟು, unsqueezed_output);
            ಹಿಂತಿರುಗಿ ನಿಜ;
        }
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    };

    ರಿಜಿಸ್ಟರ್_matcher(std::make_shared<ov::pass::pattern::Matcher>(m_matmul, "ov::frontend::ggml::pass::SqueezeMatmul"),
                     callback);
}

}  // namespace pass
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
