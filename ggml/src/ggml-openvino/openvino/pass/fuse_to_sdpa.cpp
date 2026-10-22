#ضمن "fuse_to_sdpa.h"

#ضمن <openvino/core/graph_util.hpp>
#ضمن <openvino/core/rt_info.hpp>
#ضمن <openvino/op/add.hpp>
#ضمن <openvino/op/convert.hpp>
#ضمن <openvino/op/matmul.hpp>
#ضمن <openvino/op/multiply.hpp>
#ضمن <openvino/op/scaled_dot_product_attention.hpp>
#ضمن <openvino/op/softmax.hpp>
#ضمن <openvino/op/transpose.hpp>
#ضمن <openvino/pass/pattern/op/label.hpp>
#ضمن <openvino/pass/pattern/op/pattern.hpp>
#ضمن <openvino/pass/pattern/op/wrap_type.hpp>

نطاق ov {
نطاق frontend {
نطاق ggml {
نطاق pass {

Fuse_منزل_SDPA::Fuse_منزل_SDPA() {
    // Not maintained since FLASH_ATTN_EXT has replaced this pattern
    ثابت خپلکاری m_k = ov::pass::pattern::any_ننوت();
    ثابت خپلکاری m_q = ov::pass::pattern::any_ننوت();
    ثابت خپلکاری m_qk = ov::pass::pattern::wrap_type<ov::op::v0::MatMul>({m_q, m_k});
    ثابت خپلکاری m_qk_f32 = ov::pass::pattern::wrap_type<ov::op::v0::Convert>({m_qk});
    ثابت خپلکاری m_scale = ov::pass::pattern::any_ننوت();
    ثابت خپلکاری m_scaled_qk = ov::pass::pattern::wrap_type<ov::op::v1::Multiply>({m_qk_f32, m_scale});
    ثابت خپلکاری m_ماسک = ov::pass::pattern::any_ننوت();
    ثابت خپلکاری m_masked_qk = ov::pass::pattern::wrap_type<ov::op::v1::جمع_کړه>({m_scaled_qk, m_ماسک});
    ثابت خپلکاری m_softmax_qk = ov::pass::pattern::wrap_type<ov::op::v8::Softmax>({m_masked_qk});
    ثابت خپلکاری m_softmax_qk_f16 = ov::pass::pattern::wrap_type<ov::op::v0::Convert>({m_softmax_qk});
    ثابت خپلکاری m_v = ov::pass::pattern::any_ننوت();
    ثابت خپلکاری m_qkv = ov::pass::pattern::wrap_type<ov::op::v0::MatMul>({m_softmax_qk_f16, m_v});

    ثابت خپلکاری callback = [=](ov::pass::pattern::Matcher & m) {
        خپلکاری & نمونه_منزل_output = m.get_نمونه_ارزښت_map();
        خپلکاری k = نمونه_منزل_output[m_k];
        خپلکاری q = نمونه_منزل_output[m_q];
        خپلکاری v = نمونه_منزل_output[m_v];
        خپلکاری ماسک = نمونه_منزل_output[m_ماسک];
        خپلکاری scale = نمونه_منزل_output[m_scale];

        خپلکاری ماسک_f16 = ثبتوونکی_new_غوټه<ov::op::v0::Convert>(ماسک, ov::element::f16);
        خپلکاری scale_f16 = ثبتوونکی_new_غوټه<ov::op::v0::Convert>(scale, ov::element::f16);
        خپلکاری sdpa = std::make_shared<ov::op::v13::Scaled_Dot_حاصل_ضرب_Attention>(q, k, v, ماسک_f16, scale_f16, خطأ);

        ov::replace_غوټه(m.get_سمون_ریښه(), sdpa);
        ov::copy_runtime_info(m.get_matched_nodes(), sdpa);

        أرجع ریښتیا;
    };
    ثبتوونکی_matcher(std::make_shared<ov::pass::pattern::Matcher>(m_qkv, "ov::frontend::ggml::pass::FuseToSDPA"),
                     callback);
}

}  // namespace pass
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
