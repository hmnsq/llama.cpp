#ଅନ୍ତର୍ଭାଵ "squeeze_matmul.h"

#ଅନ୍ତର୍ଭାଵ <openvino/core/graph_util.hpp>
#ଅନ୍ତର୍ଭାଵ <openvino/core/rt_info.hpp>
#ଅନ୍ତର୍ଭାଵ <openvino/core/type.hpp>
#ଅନ୍ତର୍ଭାଵ <openvino/op/constant.hpp>
#ଅନ୍ତର୍ଭାଵ <openvino/op/matmul.hpp>
#ଅନ୍ତର୍ଭାଵ <openvino/op/squeeze.hpp>
#ଅନ୍ତର୍ଭାଵ <openvino/op/unsqueeze.hpp>
#ଅନ୍ତର୍ଭାଵ <openvino/pass/pattern/op/label.hpp>
#ଅନ୍ତର୍ଭାଵ <openvino/pass/pattern/op/pattern.hpp>
#ଅନ୍ତର୍ଭାଵ <openvino/pass/pattern/op/wrap_type.hpp>

ନାମାକାଶ opp = ov::pass::pattern;

ନାମାକାଶ ov {
ନାମାକାଶ frontend {
ନାମାକାଶ ggml {
ନାମାକାଶ pass {

// For quantized models, NPUW expects the activation to be 3d in DQ(DynamicQuantization) opt, e.g. DQMatMulGQ2i
SqueezeMatmul::SqueezeMatmul() {
    auto m_act = opp::any_ନିବେଶ();
    auto m_wei = opp::any_ନିବେଶ();
    auto m_matmul = opp::wrap_type<ov::op::v0::MatMul>({m_act, m_wei});

    ଅଚଳ auto callback = [=](ov::pass::pattern::Matcher & m) {
        ଅଚଳ auto & pattern_map = m.get_pattern_ମୂଲ୍ୟ_map();
        auto matmul_ଗଣ୍ଠି =
            ov::as_type_ptr<ov::op::v0::MatMul>(pattern_map.at(m_matmul).get_ଗଣ୍ଠି_shared_ptr());
        auto act = pattern_map.at(m_act);
        auto wei = pattern_map.at(m_wei);
        auto act_shape = act.get_partial_shape();
        auto wei_shape = wei.get_partial_shape();
        ଯଦି (act_shape.rank().is_dynamic() || wei_shape.rank().is_dynamic()) {
            ଫେରନ୍ତୁ ମିଥ୍ୟା;
        }
        ଯଦି (act_shape.rank().get_length() == 4 && wei_shape.rank().get_length() == 2) {
            auto axis = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {0});
            auto squeezed_act = std::make_shared<ov::op::v0::Squeeze>(act, axis);
            auto new_matmul = std::make_shared<ov::op::v0::MatMul>(squeezed_act, wei, matmul_ଗଣ୍ଠି->get_transpose_a(),
                                                                   matmul_ଗଣ୍ଠି->get_transpose_b());
            auto unsqueezed_output = std::make_shared<ov::op::v0::Unsqueeze>(new_matmul, axis);
            unsqueezed_output->set_friendly_name(matmul_ଗଣ୍ଠି->get_friendly_name());
            ov::copy_runtime_info(matmul_ଗଣ୍ଠି, {squeezed_act, new_matmul, unsqueezed_output});
            ov::replace_ଗଣ୍ଠି(matmul_ଗଣ୍ଠି, unsqueezed_output);
            ଫେରନ୍ତୁ ସତ୍ୟ;
        }
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    };

    register_matcher(std::make_shared<ov::pass::pattern::Matcher>(m_matmul, "ov::frontend::ggml::pass::SqueezeMatmul"),
                     callback);
}

}  // namespace pass
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
