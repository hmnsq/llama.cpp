#अन्तर्भाव "squeeze_matmul.h"

#अन्तर्भाव <openvino/core/graph_util.hpp>
#अन्तर्भाव <openvino/core/rt_info.hpp>
#अन्तर्भाव <openvino/core/type.hpp>
#अन्तर्भाव <openvino/op/constant.hpp>
#अन्तर्भाव <openvino/op/matmul.hpp>
#अन्तर्भाव <openvino/op/squeeze.hpp>
#अन्तर्भाव <openvino/op/unsqueeze.hpp>
#अन्तर्भाव <openvino/pass/pattern/op/label.hpp>
#अन्तर्भाव <openvino/pass/pattern/op/pattern.hpp>
#अन्तर्भाव <openvino/pass/pattern/op/wrap_type.hpp>

नामाकाश opp = ov::pass::pattern;

नामाकाश ov {
नामाकाश frontend {
नामाकाश ggml {
नामाकाश pass {

// For quantized models, NPUW expects the activation to be 3d in DQ(DynamicQuantization) opt, e.g. DQMatMulGQ2i
SqueezeMatmul::SqueezeMatmul() {
    स्वतः m_act = opp::any_आदान();
    स्वतः m_wei = opp::any_आदान();
    स्वतः m_matmul = opp::wrap_type<ov::op::v0::MatMul>({m_act, m_wei});

    स्थिरांक स्वतः callback = [=](ov::pass::pattern::Matcher & m) {
        स्थिरांक स्वतः & pattern_map = m.get_pattern_मूल्य_map();
        स्वतः matmul_गाठ =
            ov::as_type_ptr<ov::op::v0::MatMul>(pattern_map.at(m_matmul).get_गाठ_shared_ptr());
        स्वतः act = pattern_map.at(m_act);
        स्वतः wei = pattern_map.at(m_wei);
        स्वतः act_shape = act.get_partial_shape();
        स्वतः wei_shape = wei.get_partial_shape();
        जर (act_shape.rank().is_dynamic() || wei_shape.rank().is_dynamic()) {
            परत असत्य;
        }
        जर (act_shape.rank().get_length() == 4 && wei_shape.rank().get_length() == 2) {
            स्वतः axis = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {0});
            स्वतः squeezed_act = std::make_shared<ov::op::v0::Squeeze>(act, axis);
            स्वतः new_matmul = std::make_shared<ov::op::v0::MatMul>(squeezed_act, wei, matmul_गाठ->get_transpose_a(),
                                                                   matmul_गाठ->get_transpose_b());
            स्वतः unsqueezed_output = std::make_shared<ov::op::v0::Unsqueeze>(new_matmul, axis);
            unsqueezed_output->set_friendly_name(matmul_गाठ->get_friendly_name());
            ov::copy_runtime_info(matmul_गाठ, {squeezed_act, new_matmul, unsqueezed_output});
            ov::replace_गाठ(matmul_गाठ, unsqueezed_output);
            परत सत्य;
        }
        परत असत्य;
    };

    register_matcher(std::make_shared<ov::pass::pattern::Matcher>(m_matmul, "ov::frontend::ggml::pass::SqueezeMatmul"),
                     callback);
}

}  // namespace pass
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
