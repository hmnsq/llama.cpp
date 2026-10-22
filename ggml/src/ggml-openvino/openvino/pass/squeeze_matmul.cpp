#അന്തര്ഭാവ "squeeze_matmul.h"

#അന്തര്ഭാവ <openvino/core/graph_util.hpp>
#അന്തര്ഭാവ <openvino/core/rt_info.hpp>
#അന്തര്ഭാവ <openvino/core/type.hpp>
#അന്തര്ഭാവ <openvino/op/constant.hpp>
#അന്തര്ഭാവ <openvino/op/matmul.hpp>
#അന്തര്ഭാവ <openvino/op/squeeze.hpp>
#അന്തര്ഭാവ <openvino/op/unsqueeze.hpp>
#അന്തര്ഭാവ <openvino/pass/pattern/op/label.hpp>
#അന്തര്ഭാവ <openvino/pass/pattern/op/pattern.hpp>
#അന്തര്ഭാവ <openvino/pass/pattern/op/wrap_type.hpp>

നാമാകാശ opp = ov::pass::pattern;

നാമാകാശ ov {
നാമാകാശ frontend {
നാമാകാശ ggml {
നാമാകാശ pass {

// For quantized models, NPUW expects the activation to be 3d in DQ(DynamicQuantization) opt, e.g. DQMatMulGQ2i
SqueezeMatmul::SqueezeMatmul() {
    auto m_act = opp::any_നിവേശം();
    auto m_wei = opp::any_നിവേശം();
    auto m_matmul = opp::wrap_type<ov::op::v0::MatMul>({m_act, m_wei});

    സ്ഥിരം auto callback = [=](ov::pass::pattern::Matcher & m) {
        സ്ഥിരം auto & pattern_map = m.get_pattern_മൂല്യം_map();
        auto matmul_കണ്ണി =
            ov::as_type_ptr<ov::op::v0::MatMul>(pattern_map.at(m_matmul).get_കണ്ണി_shared_ptr());
        auto act = pattern_map.at(m_act);
        auto wei = pattern_map.at(m_wei);
        auto act_shape = act.get_partial_shape();
        auto wei_shape = wei.get_partial_shape();
        എങ്കിൽ (act_shape.rank().is_dynamic() || wei_shape.rank().is_dynamic()) {
            തിരികെ അസത്യം;
        }
        എങ്കിൽ (act_shape.rank().get_length() == 4 && wei_shape.rank().get_length() == 2) {
            auto axis = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {0});
            auto squeezed_act = std::make_shared<ov::op::v0::Squeeze>(act, axis);
            auto new_matmul = std::make_shared<ov::op::v0::MatMul>(squeezed_act, wei, matmul_കണ്ണി->get_transpose_a(),
                                                                   matmul_കണ്ണി->get_transpose_b());
            auto unsqueezed_output = std::make_shared<ov::op::v0::Unsqueeze>(new_matmul, axis);
            unsqueezed_output->set_friendly_name(matmul_കണ്ണി->get_friendly_name());
            ov::copy_runtime_info(matmul_കണ്ണി, {squeezed_act, new_matmul, unsqueezed_output});
            ov::replace_കണ്ണി(matmul_കണ്ണി, unsqueezed_output);
            തിരികെ സത്യം;
        }
        തിരികെ അസത്യം;
    };

    register_matcher(std::make_shared<ov::pass::pattern::Matcher>(m_matmul, "ov::frontend::ggml::pass::SqueezeMatmul"),
                     callback);
}

}  // namespace pass
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
