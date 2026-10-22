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

FuseToSDPA::FuseToSDPA() {
    // Not maintained since FLASH_ATTN_EXT has replaced this pattern
    ثابت auto m_k = ov::pass::pattern::any_ماسوقن();
    ثابت auto m_q = ov::pass::pattern::any_ماسوقن();
    ثابت auto m_qk = ov::pass::pattern::wrap_type<ov::op::v0::MatMul>({m_q, m_k});
    ثابت auto m_qk_f32 = ov::pass::pattern::wrap_type<ov::op::v0::Convert>({m_qk});
    ثابت auto m_scale = ov::pass::pattern::any_ماسوقن();
    ثابت auto m_scaled_qk = ov::pass::pattern::wrap_type<ov::op::v1::Multiply>({m_qk_f32, m_scale});
    ثابت auto m_mask = ov::pass::pattern::any_ماسوقن();
    ثابت auto m_masked_qk = ov::pass::pattern::wrap_type<ov::op::v1::Add>({m_scaled_qk, m_mask});
    ثابت auto m_softmax_qk = ov::pass::pattern::wrap_type<ov::op::v8::Softmax>({m_masked_qk});
    ثابت auto m_softmax_qk_f16 = ov::pass::pattern::wrap_type<ov::op::v0::Convert>({m_softmax_qk});
    ثابت auto m_v = ov::pass::pattern::any_ماسوقن();
    ثابت auto m_qkv = ov::pass::pattern::wrap_type<ov::op::v0::MatMul>({m_softmax_qk_f16, m_v});

    ثابت auto callback = [=](ov::pass::pattern::Matcher & m) {
        auto & pattern_to_output = m.get_pattern_نيلاي_map();
        auto k = pattern_to_output[m_k];
        auto q = pattern_to_output[m_q];
        auto v = pattern_to_output[m_v];
        auto mask = pattern_to_output[m_mask];
        auto scale = pattern_to_output[m_scale];

        auto mask_f16 = register_new_نود<ov::op::v0::Convert>(mask, ov::element::f16);
        auto scale_f16 = register_new_نود<ov::op::v0::Convert>(scale, ov::element::f16);
        auto sdpa = std::make_shared<ov::op::v13::ScaledDotProductAttention>(q, k, v, mask_f16, scale_f16, خطأ);

        ov::replace_نود(m.get_match_اکر(), sdpa);
        ov::copy_runtime_info(m.get_matched_nodes(), sdpa);

        أرجع true;
    };
    register_matcher(std::make_shared<ov::pass::pattern::Matcher>(m_qkv, "ov::frontend::ggml::pass::FuseToSDPA"),
                     callback);
}

}  // namespace pass
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
