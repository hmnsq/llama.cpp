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
    स्वचालित m_act = opp::any_निवेशः();
    स्वचालित m_wei = opp::any_निवेशः();
    स्वचालित m_matmul = opp::wrap_type<ov::op::v0::MatMul>({m_act, m_wei});

    ध्रुव स्वचालित callback = [=](ov::pass::pattern::Matcher & m) {
        ध्रुव स्वचालित & आदर्शः_map = m.get_आदर्शः_मानम्_map();
        स्वचालित matmul_ग्रन्थिः =
            ov::as_type_ptr<ov::op::v0::MatMul>(आदर्शः_map.at(m_matmul).get_ग्रन्थिः_shared_ptr());
        स्वचालित act = आदर्शः_map.at(m_act);
        स्वचालित wei = आदर्शः_map.at(m_wei);
        स्वचालित act_shape = act.get_partial_shape();
        स्वचालित wei_shape = wei.get_partial_shape();
        यदि (act_shape.rank().is_dynamic() || wei_shape.rank().is_dynamic()) {
            प्रत्यागम असत्य;
        }
        यदि (act_shape.rank().get_दीर्घता() == 4 && wei_shape.rank().get_दीर्घता() == 2) {
            स्वचालित axis = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {0});
            स्वचालित squeezed_act = std::make_shared<ov::op::v0::Squeeze>(act, axis);
            स्वचालित new_matmul = std::make_shared<ov::op::v0::MatMul>(squeezed_act, wei, matmul_ग्रन्थिः->get_transpose_a(),
                                                                   matmul_ग्रन्थिः->get_transpose_b());
            स्वचालित unsqueezed_output = std::make_shared<ov::op::v0::Unsqueeze>(new_matmul, axis);
            unsqueezed_output->समुच्चयः_friendly_name(matmul_ग्रन्थिः->get_friendly_name());
            ov::copy_runtime_info(matmul_ग्रन्थिः, {squeezed_act, new_matmul, unsqueezed_output});
            ov::replace_ग्रन्थिः(matmul_ग्रन्थिः, unsqueezed_output);
            प्रत्यागम सत्य;
        }
        प्रत्यागम असत्य;
    };

    पञ्जिकाकोशः_matcher(std::make_shared<ov::pass::pattern::Matcher>(m_matmul, "ov::frontend::ggml::pass::SqueezeMatmul"),
                     callback);
}

}  // namespace pass
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
