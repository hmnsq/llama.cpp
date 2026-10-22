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
    خپلکاری m_act = opp::any_ننوت();
    خپلکاری m_wei = opp::any_ننوت();
    خپلکاری m_matmul = opp::wrap_type<ov::op::v0::MatMul>({m_act, m_wei});

    ثابت خپلکاری callback = [=](ov::pass::pattern::Matcher & m) {
        ثابت خپلکاری & نمونه_map = m.get_نمونه_ارزښت_map();
        خپلکاری matmul_غوټه =
            ov::as_type_ptr<ov::op::v0::MatMul>(نمونه_map.at(m_matmul).get_غوټه_shared_ptr());
        خپلکاری act = نمونه_map.at(m_act);
        خپلکاری wei = نمونه_map.at(m_wei);
        خپلکاری act_shape = act.get_partial_shape();
        خپلکاری wei_shape = wei.get_partial_shape();
        إذا (act_shape.rank().is_dynamic() || wei_shape.rank().is_dynamic()) {
            أرجع خطأ;
        }
        إذا (act_shape.rank().get_اوږدتیا() == 4 && wei_shape.rank().get_اوږدتیا() == 2) {
            خپلکاری axis = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {0});
            خپلکاری squeezed_act = std::make_shared<ov::op::v0::Squeeze>(act, axis);
            خپلکاری new_matmul = std::make_shared<ov::op::v0::MatMul>(squeezed_act, wei, matmul_غوټه->get_transpose_a(),
                                                                   matmul_غوټه->get_transpose_b());
            خپلکاری unsqueezed_output = std::make_shared<ov::op::v0::Unsqueeze>(new_matmul, axis);
            unsqueezed_output->سېټ_friendly_name(matmul_غوټه->get_friendly_name());
            ov::copy_runtime_info(matmul_غوټه, {squeezed_act, new_matmul, unsqueezed_output});
            ov::replace_غوټه(matmul_غوټه, unsqueezed_output);
            أرجع ریښتیا;
        }
        أرجع خطأ;
    };

    ثبتوونکی_matcher(std::make_shared<ov::pass::pattern::Matcher>(m_matmul, "ov::frontend::ggml::pass::SqueezeMatmul"),
                     callback);
}

}  // namespace pass
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
