#అన్తర్భావ "squeeze_matmul.h"

#అన్తర్భావ <openvino/core/graph_util.hpp>
#అన్తర్భావ <openvino/core/rt_info.hpp>
#అన్తర్భావ <openvino/core/type.hpp>
#అన్తర్భావ <openvino/op/constant.hpp>
#అన్తర్భావ <openvino/op/matmul.hpp>
#అన్తర్భావ <openvino/op/squeeze.hpp>
#అన్తర్భావ <openvino/op/unsqueeze.hpp>
#అన్తర్భావ <openvino/pass/pattern/op/label.hpp>
#అన్తర్భావ <openvino/pass/pattern/op/pattern.hpp>
#అన్తర్భావ <openvino/pass/pattern/op/wrap_type.hpp>

నామాకాశ opp = ov::pass::pattern;

నామాకాశ ov {
నామాకాశ frontend {
నామాకాశ ggml {
నామాకాశ pass {

// For quantized models, NPUW expects the activation to be 3d in DQ(DynamicQuantization) opt, e.g. DQMatMulGQ2i
SqueezeMatmul::SqueezeMatmul() {
    స్వయంచాలక m_act = opp::any_ఆదానం();
    స్వయంచాలక m_wei = opp::any_ఆదానం();
    స్వయంచాలక m_matmul = opp::wrap_type<ov::op::v0::MatMul>({m_act, m_wei});

    స్థిరరాశి స్వయంచాలక callback = [=](ov::pass::pattern::Matcher & m) {
        స్థిరరాశి స్వయంచాలక & నమూనా_map = m.get_నమూనా_విలువ_map();
        స్వయంచాలక matmul_కణుపు =
            ov::as_type_ptr<ov::op::v0::MatMul>(నమూనా_map.at(m_matmul).get_కణుపు_shared_ptr());
        స్వయంచాలక act = నమూనా_map.at(m_act);
        స్వయంచాలక wei = నమూనా_map.at(m_wei);
        స్వయంచాలక act_shape = act.get_partial_shape();
        స్వయంచాలక wei_shape = wei.get_partial_shape();
        ఒకవేళ (act_shape.rank().is_dynamic() || wei_shape.rank().is_dynamic()) {
            తిరిగి అబద్ధం;
        }
        ఒకవేళ (act_shape.rank().get_నిడివి() == 4 && wei_shape.rank().get_నిడివి() == 2) {
            స్వయంచాలక axis = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {0});
            స్వయంచాలక squeezed_act = std::make_shared<ov::op::v0::Squeeze>(act, axis);
            స్వయంచాలక new_matmul = std::make_shared<ov::op::v0::MatMul>(squeezed_act, wei, matmul_కణుపు->get_transpose_a(),
                                                                   matmul_కణుపు->get_transpose_b());
            స్వయంచాలక unsqueezed_output = std::make_shared<ov::op::v0::Unsqueeze>(new_matmul, axis);
            unsqueezed_output->సమితి_friendly_name(matmul_కణుపు->get_friendly_name());
            ov::copy_runtime_info(matmul_కణుపు, {squeezed_act, new_matmul, unsqueezed_output});
            ov::replace_కణుపు(matmul_కణుపు, unsqueezed_output);
            తిరిగి నిజం;
        }
        తిరిగి అబద్ధం;
    };

    రిజిస్టరు_matcher(std::make_shared<ov::pass::pattern::Matcher>(m_matmul, "ov::frontend::ggml::pass::SqueezeMatmul"),
                     callback);
}

}  // namespace pass
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
