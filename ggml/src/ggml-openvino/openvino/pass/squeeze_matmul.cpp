#அந்தர்பாவ "squeeze_matmul.h"

#அந்தர்பாவ <openvino/core/graph_util.hpp>
#அந்தர்பாவ <openvino/core/rt_info.hpp>
#அந்தர்பாவ <openvino/core/type.hpp>
#அந்தர்பாவ <openvino/op/constant.hpp>
#அந்தர்பாவ <openvino/op/matmul.hpp>
#அந்தர்பாவ <openvino/op/squeeze.hpp>
#அந்தர்பாவ <openvino/op/unsqueeze.hpp>
#அந்தர்பாவ <openvino/pass/pattern/op/label.hpp>
#அந்தர்பாவ <openvino/pass/pattern/op/pattern.hpp>
#அந்தர்பாவ <openvino/pass/pattern/op/wrap_type.hpp>

நாமாகாஶ opp = ov::pass::pattern;

நாமாகாஶ ov {
நாமாகாஶ frontend {
நாமாகாஶ ggml {
நாமாகாஶ pass {

// For quantized models, NPUW expects the activation to be 3d in DQ(DynamicQuantization) opt, e.g. DQMatMulGQ2i
SqueezeMatmul::SqueezeMatmul() {
    auto m_act = opp::any_உள்ளீடு();
    auto m_wei = opp::any_உள்ளீடு();
    auto m_matmul = opp::wrap_type<ov::op::v0::MatMul>({m_act, m_wei});

    மாறிலி auto callback = [=](ov::pass::pattern::Matcher & m) {
        மாறிலி auto & pattern_map = m.get_pattern_மதிப்பு_map();
        auto matmul_முனை =
            ov::as_type_ptr<ov::op::v0::MatMul>(pattern_map.at(m_matmul).get_முனை_shared_ptr());
        auto act = pattern_map.at(m_act);
        auto wei = pattern_map.at(m_wei);
        auto act_shape = act.get_partial_shape();
        auto wei_shape = wei.get_partial_shape();
        எனில் (act_shape.rank().is_dynamic() || wei_shape.rank().is_dynamic()) {
            திரும்பு பொய்;
        }
        எனில் (act_shape.rank().get_length() == 4 && wei_shape.rank().get_length() == 2) {
            auto axis = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {0});
            auto squeezed_act = std::make_shared<ov::op::v0::Squeeze>(act, axis);
            auto new_matmul = std::make_shared<ov::op::v0::MatMul>(squeezed_act, wei, matmul_முனை->get_transpose_a(),
                                                                   matmul_முனை->get_transpose_b());
            auto unsqueezed_output = std::make_shared<ov::op::v0::Unsqueeze>(new_matmul, axis);
            unsqueezed_output->set_friendly_name(matmul_முனை->get_friendly_name());
            ov::copy_runtime_info(matmul_முனை, {squeezed_act, new_matmul, unsqueezed_output});
            ov::replace_முனை(matmul_முனை, unsqueezed_output);
            திரும்பு உண்மை;
        }
        திரும்பு பொய்;
    };

    register_matcher(std::make_shared<ov::pass::pattern::Matcher>(m_matmul, "ov::frontend::ggml::pass::SqueezeMatmul"),
                     callback);
}

}  // namespace pass
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
