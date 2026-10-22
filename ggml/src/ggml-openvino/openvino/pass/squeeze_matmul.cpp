#અન્તર્ભાવ "squeeze_matmul.h"

#અન્તર્ભાવ <openvino/core/graph_util.hpp>
#અન્તર્ભાવ <openvino/core/rt_info.hpp>
#અન્તર્ભાવ <openvino/core/type.hpp>
#અન્તર્ભાવ <openvino/op/constant.hpp>
#અન્તર્ભાવ <openvino/op/matmul.hpp>
#અન્તર્ભાવ <openvino/op/squeeze.hpp>
#અન્તર્ભાવ <openvino/op/unsqueeze.hpp>
#અન્તર્ભાવ <openvino/pass/pattern/op/label.hpp>
#અન્તર્ભાવ <openvino/pass/pattern/op/pattern.hpp>
#અન્તર્ભાવ <openvino/pass/pattern/op/wrap_type.hpp>

નામાકાશ opp = ov::pass::pattern;

નામાકાશ ov {
નામાકાશ frontend {
નામાકાશ ggml {
નામાકાશ pass {

// For quantized models, NPUW expects the activation to be 3d in DQ(DynamicQuantization) opt, e.g. DQMatMulGQ2i
SqueezeMatmul::SqueezeMatmul() {
    auto m_act = opp::any_આગત();
    auto m_wei = opp::any_આગત();
    auto m_matmul = opp::wrap_type<ov::op::v0::MatMul>({m_act, m_wei});

    અચલ auto callback = [=](ov::pass::pattern::Matcher & m) {
        અચલ auto & pattern_map = m.get_pattern_કિંમત_map();
        auto matmul_ગાંઠ =
            ov::as_type_ptr<ov::op::v0::MatMul>(pattern_map.at(m_matmul).get_ગાંઠ_shared_ptr());
        auto act = pattern_map.at(m_act);
        auto wei = pattern_map.at(m_wei);
        auto act_shape = act.get_partial_shape();
        auto wei_shape = wei.get_partial_shape();
        જો (act_shape.rank().is_dynamic() || wei_shape.rank().is_dynamic()) {
            પાછા ખોટું;
        }
        જો (act_shape.rank().get_length() == 4 && wei_shape.rank().get_length() == 2) {
            auto axis = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {0});
            auto squeezed_act = std::make_shared<ov::op::v0::Squeeze>(act, axis);
            auto new_matmul = std::make_shared<ov::op::v0::MatMul>(squeezed_act, wei, matmul_ગાંઠ->get_transpose_a(),
                                                                   matmul_ગાંઠ->get_transpose_b());
            auto unsqueezed_output = std::make_shared<ov::op::v0::Unsqueeze>(new_matmul, axis);
            unsqueezed_output->set_friendly_name(matmul_ગાંઠ->get_friendly_name());
            ov::copy_runtime_info(matmul_ગાંઠ, {squeezed_act, new_matmul, unsqueezed_output});
            ov::replace_ગાંઠ(matmul_ગાંઠ, unsqueezed_output);
            પાછા સાચું;
        }
        પાછા ખોટું;
    };

    register_matcher(std::make_shared<ov::pass::pattern::Matcher>(m_matmul, "ov::frontend::ggml::pass::SqueezeMatmul"),
                     callback);
}

}  // namespace pass
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
