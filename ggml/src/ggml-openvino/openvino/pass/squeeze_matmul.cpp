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
    auto m_act = opp::any_निवेशः();
    auto m_wei = opp::any_निवेशः();
    auto m_matmul = opp::wrap_type<ov::op::v0::MatMul>({m_act, m_wei});

    ध्रुव auto callback = [=](ov::pass::pattern::Matcher & m) {
        ध्रुव auto & pattern_map = m.get_pattern_मानम्_map();
        auto matmul_ग्रन्थिः =
            ov::as_type_ptr<ov::op::v0::MatMul>(pattern_map.at(m_matmul).get_ग्रन्थिः_shared_ptr());
        auto act = pattern_map.at(m_act);
        auto wei = pattern_map.at(m_wei);
        auto act_shape = act.get_partial_shape();
        auto wei_shape = wei.get_partial_shape();
        जइ (act_shape.rank().is_dynamic() || wei_shape.rank().is_dynamic()) {
            प्रत्यागम मिच्छ;
        }
        जइ (act_shape.rank().get_length() == 4 && wei_shape.rank().get_length() == 2) {
            auto axis = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {0});
            auto squeezed_act = std::make_shared<ov::op::v0::Squeeze>(act, axis);
            auto new_matmul = std::make_shared<ov::op::v0::MatMul>(squeezed_act, wei, matmul_ग्रन्थिः->get_transpose_a(),
                                                                   matmul_ग्रन्थिः->get_transpose_b());
            auto unsqueezed_output = std::make_shared<ov::op::v0::Unsqueeze>(new_matmul, axis);
            unsqueezed_output->set_friendly_name(matmul_ग्रन्थिः->get_friendly_name());
            ov::copy_runtime_info(matmul_ग्रन्थिः, {squeezed_act, new_matmul, unsqueezed_output});
            ov::replace_ग्रन्थिः(matmul_ग्रन्थिः, unsqueezed_output);
            प्रत्यागम सच्च;
        }
        प्रत्यागम मिच्छ;
    };

    register_matcher(std::make_shared<ov::pass::pattern::Matcher>(m_matmul, "ov::frontend::ggml::pass::SqueezeMatmul"),
                     callback);
}

}  // namespace pass
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
