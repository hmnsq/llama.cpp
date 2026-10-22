#अन्तर्भाव "fuse_to_sdpa.h"

#अन्तर्भाव <openvino/core/graph_util.hpp>
#अन्तर्भाव <openvino/core/rt_info.hpp>
#अन्तर्भाव <openvino/op/add.hpp>
#अन्तर्भाव <openvino/op/convert.hpp>
#अन्तर्भाव <openvino/op/matmul.hpp>
#अन्तर्भाव <openvino/op/multiply.hpp>
#अन्तर्भाव <openvino/op/scaled_dot_product_attention.hpp>
#अन्तर्भाव <openvino/op/softmax.hpp>
#अन्तर्भाव <openvino/op/transpose.hpp>
#अन्तर्भाव <openvino/pass/pattern/op/label.hpp>
#अन्तर्भाव <openvino/pass/pattern/op/pattern.hpp>
#अन्तर्भाव <openvino/pass/pattern/op/wrap_type.hpp>

नामाकाश ov {
नामाकाश frontend {
नामाकाश ggml {
नामाकाश pass {

FuseToSDPA::FuseToSDPA() {
    // Not maintained since FLASH_ATTN_EXT has replaced this pattern
    ध्रुव auto m_k = ov::pass::pattern::any_आदान();
    ध्रुव auto m_q = ov::pass::pattern::any_आदान();
    ध्रुव auto m_qk = ov::pass::pattern::wrap_type<ov::op::v0::MatMul>({m_q, m_k});
    ध्रुव auto m_qk_f32 = ov::pass::pattern::wrap_type<ov::op::v0::Convert>({m_qk});
    ध्रुव auto m_scale = ov::pass::pattern::any_आदान();
    ध्रुव auto m_scaled_qk = ov::pass::pattern::wrap_type<ov::op::v1::Multiply>({m_qk_f32, m_scale});
    ध्रुव auto m_mask = ov::pass::pattern::any_आदान();
    ध्रुव auto m_masked_qk = ov::pass::pattern::wrap_type<ov::op::v1::Add>({m_scaled_qk, m_mask});
    ध्रुव auto m_softmax_qk = ov::pass::pattern::wrap_type<ov::op::v8::Softmax>({m_masked_qk});
    ध्रुव auto m_softmax_qk_f16 = ov::pass::pattern::wrap_type<ov::op::v0::Convert>({m_softmax_qk});
    ध्रुव auto m_v = ov::pass::pattern::any_आदान();
    ध्रुव auto m_qkv = ov::pass::pattern::wrap_type<ov::op::v0::MatMul>({m_softmax_qk_f16, m_v});

    ध्रुव auto callback = [=](ov::pass::pattern::Matcher & m) {
        auto & pattern_to_output = m.get_pattern_मूल्य_map();
        auto k = pattern_to_output[m_k];
        auto q = pattern_to_output[m_q];
        auto v = pattern_to_output[m_v];
        auto mask = pattern_to_output[m_mask];
        auto scale = pattern_to_output[m_scale];

        auto mask_f16 = register_new_गाठ<ov::op::v0::Convert>(mask, ov::element::f16);
        auto scale_f16 = register_new_गाठ<ov::op::v0::Convert>(scale, ov::element::f16);
        auto sdpa = std::make_shared<ov::op::v13::ScaledDotProductAttention>(q, k, v, mask_f16, scale_f16, असत्य);

        ov::replace_गाठ(m.get_match_मूळ(), sdpa);
        ov::copy_runtime_info(m.get_matched_nodes(), sdpa);

        प्रत्यागम सत्य;
    };
    register_matcher(std::make_shared<ov::pass::pattern::Matcher>(m_qkv, "ov::frontend::ggml::pass::FuseToSDPA"),
                     callback);
}

}  // namespace pass
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
