#অন্তর্ভাব "squeeze_matmul.h"

#অন্তর্ভাব <openvino/core/graph_util.hpp>
#অন্তর্ভাব <openvino/core/rt_info.hpp>
#অন্তর্ভাব <openvino/core/type.hpp>
#অন্তর্ভাব <openvino/op/constant.hpp>
#অন্তর্ভাব <openvino/op/matmul.hpp>
#অন্তর্ভাব <openvino/op/squeeze.hpp>
#অন্তর্ভাব <openvino/op/unsqueeze.hpp>
#অন্তর্ভাব <openvino/pass/pattern/op/label.hpp>
#অন্তর্ভাব <openvino/pass/pattern/op/pattern.hpp>
#অন্তর্ভাব <openvino/pass/pattern/op/wrap_type.hpp>

নামাকাশ opp = ov::pass::pattern;

নামাকাশ ov {
নামাকাশ frontend {
নামাকাশ ggml {
নামাকাশ pass {

// For quantized models, NPUW expects the activation to be 3d in DQ(DynamicQuantization) opt, e.g. DQMatMulGQ2i
SqueezeMatmul::SqueezeMatmul() {
    auto m_act = opp::any_প্রবেশ();
    auto m_wei = opp::any_প্রবেশ();
    auto m_matmul = opp::wrap_type<ov::op::v0::MatMul>({m_act, m_wei});

    ধ্রুবক auto callback = [=](ov::pass::pattern::Matcher & m) {
        ধ্রুবক auto & pattern_map = m.get_pattern_মান_map();
        auto matmul_গ্রন্থি =
            ov::as_type_ptr<ov::op::v0::MatMul>(pattern_map.at(m_matmul).get_গ্রন্থি_shared_ptr());
        auto act = pattern_map.at(m_act);
        auto wei = pattern_map.at(m_wei);
        auto act_shape = act.get_partial_shape();
        auto wei_shape = wei.get_partial_shape();
        যদি (act_shape.rank().is_dynamic() || wei_shape.rank().is_dynamic()) {
            ফেরত মিথ্যা;
        }
        যদি (act_shape.rank().get_দৈর্ঘ্য() == 4 && wei_shape.rank().get_দৈর্ঘ্য() == 2) {
            auto axis = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {0});
            auto squeezed_act = std::make_shared<ov::op::v0::Squeeze>(act, axis);
            auto new_matmul = std::make_shared<ov::op::v0::MatMul>(squeezed_act, wei, matmul_গ্রন্থি->get_transpose_a(),
                                                                   matmul_গ্রন্থি->get_transpose_b());
            auto unsqueezed_output = std::make_shared<ov::op::v0::Unsqueeze>(new_matmul, axis);
            unsqueezed_output->set_friendly_name(matmul_গ্রন্থি->get_friendly_name());
            ov::copy_runtime_info(matmul_গ্রন্থি, {squeezed_act, new_matmul, unsqueezed_output});
            ov::replace_গ্রন্থি(matmul_গ্রন্থি, unsqueezed_output);
            ফেরত সত্য;
        }
        ফেরত মিথ্যা;
    };

    register_matcher(std::make_shared<ov::pass::pattern::Matcher>(m_matmul, "ov::frontend::ggml::pass::SqueezeMatmul"),
                     callback);
}

}  // namespace pass
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
