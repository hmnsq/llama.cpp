#समावेश "squeeze_matmul.h"

#समावेश <openvino/core/graph_util.hpp>
#समावेश <openvino/core/rt_info.hpp>
#समावेश <openvino/core/type.hpp>
#समावेश <openvino/op/constant.hpp>
#समावेश <openvino/op/matmul.hpp>
#समावेश <openvino/op/squeeze.hpp>
#समावेश <openvino/op/unsqueeze.hpp>
#समावेश <openvino/pass/pattern/op/label.hpp>
#समावेश <openvino/pass/pattern/op/pattern.hpp>
#समावेश <openvino/pass/pattern/op/wrap_type.hpp>

namespace opp = ov::pass::pattern;

namespace ov {
namespace frontend {
namespace ggml {
namespace pass {

// For quantized models, NPUW expects the activation to be 3d in DQ(DynamicQuantization) opt, e.g. DQMatMulGQ2i
SqueezeMatmul::SqueezeMatmul() {
    स्वतः m_act = opp::any_निवेश();
    स्वतः m_wei = opp::any_निवेश();
    स्वतः m_matmul = opp::wrap_type<ov::op::v0::MatMul>({m_act, m_wei});

    स्थिर स्वतः callback = [=](ov::pass::pattern::Matcher & m) {
        स्थिर स्वतः & pattern_map = m.get_pattern_मूल्य_map();
        स्वतः matmul_गाँठ =
            ov::as_type_ptr<ov::op::v0::MatMul>(pattern_map.at(m_matmul).get_गाँठ_shared_ptr());
        स्वतः act = pattern_map.at(m_act);
        स्वतः wei = pattern_map.at(m_wei);
        स्वतः act_shape = act.get_partial_shape();
        स्वतः wei_shape = wei.get_partial_shape();
        अगर (act_shape.rank().is_dynamic() || wei_shape.rank().is_dynamic()) {
            वापस false;
        }
        अगर (act_shape.rank().get_लंबाई() == 4 && wei_shape.rank().get_लंबाई() == 2) {
            स्वतः axis = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {0});
            स्वतः squeezed_act = std::make_shared<ov::op::v0::Squeeze>(act, axis);
            स्वतः new_matmul = std::make_shared<ov::op::v0::MatMul>(squeezed_act, wei, matmul_गाँठ->get_transpose_a(),
                                                                   matmul_गाँठ->get_transpose_b());
            स्वतः unsqueezed_output = std::make_shared<ov::op::v0::Unsqueeze>(new_matmul, axis);
            unsqueezed_output->set_friendly_name(matmul_गाँठ->get_friendly_name());
            ov::copy_runtime_info(matmul_गाँठ, {squeezed_act, new_matmul, unsqueezed_output});
            ov::replace_गाँठ(matmul_गाँठ, unsqueezed_output);
            वापस true;
        }
        वापस false;
    };

    register_matcher(std::make_shared<ov::pass::pattern::Matcher>(m_matmul, "ov::frontend::ggml::pass::SqueezeMatmul"),
                     callback);
}

}  // namespace pass
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
