#شامل "squeeze_matmul.h"

#شامل <openvino/core/graph_util.hpp>
#شامل <openvino/core/rt_info.hpp>
#شامل <openvino/core/type.hpp>
#شامل <openvino/op/constant.hpp>
#شامل <openvino/op/matmul.hpp>
#شامل <openvino/op/squeeze.hpp>
#شامل <openvino/op/unsqueeze.hpp>
#شامل <openvino/pass/pattern/op/label.hpp>
#شامل <openvino/pass/pattern/op/pattern.hpp>
#شامل <openvino/pass/pattern/op/wrap_type.hpp>

فضاینام opp = ov::pass::pattern;

فضاینام ov {
فضاینام frontend {
فضاینام ggml {
فضاینام pass {

// For quantized models, NPUW expects the activation to be 3d in DQ(DynamicQuantization) opt, e.g. DQMatMulGQ2i
SqueezeMatmul::SqueezeMatmul() {
    auto m_act = opp::any_درون_داد();
    auto m_wei = opp::any_درون_داد();
    auto m_matmul = opp::wrap_type<ov::op::v0::MatMul>({m_act, m_wei});

    ثابت auto callback = [=](ov::pass::pattern::Matcher & m) {
        ثابت auto & pattern_map = m.get_pattern_مقدار_map();
        auto matmul_گره =
            ov::as_type_ptr<ov::op::v0::MatMul>(pattern_map.at(m_matmul).get_گره_shared_ptr());
        auto act = pattern_map.at(m_act);
        auto wei = pattern_map.at(m_wei);
        auto act_shape = act.get_partial_shape();
        auto wei_shape = wei.get_partial_shape();
        اگر (act_shape.rank().is_dynamic() || wei_shape.rank().is_dynamic()) {
            بازگشت نادرست;
        }
        اگر (act_shape.rank().get_length() == 4 && wei_shape.rank().get_length() == 2) {
            auto axis = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {0});
            auto squeezed_act = std::make_shared<ov::op::v0::Squeeze>(act, axis);
            auto new_matmul = std::make_shared<ov::op::v0::MatMul>(squeezed_act, wei, matmul_گره->get_transpose_a(),
                                                                   matmul_گره->get_transpose_b());
            auto unsqueezed_output = std::make_shared<ov::op::v0::Unsqueeze>(new_matmul, axis);
            unsqueezed_output->set_friendly_name(matmul_گره->get_friendly_name());
            ov::copy_runtime_info(matmul_گره, {squeezed_act, new_matmul, unsqueezed_output});
            ov::replace_گره(matmul_گره, unsqueezed_output);
            بازگشت درست;
        }
        بازگشت نادرست;
    };

    register_matcher(std::make_shared<ov::pass::pattern::Matcher>(m_matmul, "ov::frontend::ggml::pass::SqueezeMatmul"),
                     callback);
}

}  // namespace pass
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
