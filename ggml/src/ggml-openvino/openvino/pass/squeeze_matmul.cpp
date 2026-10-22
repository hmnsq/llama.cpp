#ضمن "squeeze_matmul.h"

#ضمن <openvino/core/graph_util.hpp>
#ضمن <openvino/core/rt_info.hpp>
#ضمن <openvino/core/type.hpp>
#ضمن <openvino/op/constant.hpp>
#ضمن <openvino/op/matmul.hpp>
#ضمن <openvino/op/squeeze.hpp>
#ضمن <openvino/op/unsqueeze.hpp>
#ضمن <openvino/pass/pattern/op/label.hpp>
#ضمن <openvino/pass/pattern/op/pattern.hpp>
#ضمن <openvino/pass/pattern/op/wrap_type.hpp>

نطاق opp = ov::pass::pattern;

نطاق ov {
نطاق frontend {
نطاق ggml {
نطاق pass {

// For quantized models, NPUW expects the activation to be 3d in DQ(DynamicQuantization) opt, e.g. DQMatMulGQ2i
SqueezeMatmul::SqueezeMatmul() {
    خودکار m_act = opp::any_درون_داد();
    خودکار m_wei = opp::any_درون_داد();
    خودکار m_matmul = opp::wrap_type<ov::op::v0::MatMul>({m_act, m_wei});

    ثابت خودکار callback = [=](ov::pass::pattern::Matcher & m) {
        ثابت خودکار & الگو_map = m.get_الگو_مقدار_map();
        خودکار matmul_گره =
            ov::as_type_ptr<ov::op::v0::MatMul>(الگو_map.at(m_matmul).get_گره_shared_ptr());
        خودکار act = الگو_map.at(m_act);
        خودکار wei = الگو_map.at(m_wei);
        خودکار act_shape = act.get_partial_shape();
        خودکار wei_shape = wei.get_partial_shape();
        إذا (act_shape.rank().is_dynamic() || wei_shape.rank().is_dynamic()) {
            أرجع خطأ;
        }
        إذا (act_shape.rank().get_درازی() == 4 && wei_shape.rank().get_درازی() == 2) {
            خودکار axis = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {0});
            خودکار squeezed_act = std::make_shared<ov::op::v0::Squeeze>(act, axis);
            خودکار new_matmul = std::make_shared<ov::op::v0::MatMul>(squeezed_act, wei, matmul_گره->get_transpose_a(),
                                                                   matmul_گره->get_transpose_b());
            خودکار unsqueezed_output = std::make_shared<ov::op::v0::Unsqueeze>(new_matmul, axis);
            unsqueezed_output->مجموعه_friendly_name(matmul_گره->get_friendly_name());
            ov::copy_runtime_info(matmul_گره, {squeezed_act, new_matmul, unsqueezed_output});
            ov::replace_گره(matmul_گره, unsqueezed_output);
            أرجع درست;
        }
        أرجع خطأ;
    };

    ثبات_matcher(std::make_shared<ov::pass::pattern::Matcher>(m_matmul, "ov::frontend::ggml::pass::SqueezeMatmul"),
                     callback);
}

}  // namespace pass
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
