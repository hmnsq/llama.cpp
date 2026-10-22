#ಅನ್ತರ್ಭಾವ "fuse_to_sdpa.h"

#ಅನ್ತರ್ಭಾವ <openvino/core/graph_util.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/core/rt_info.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/add.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/convert.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/matmul.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/multiply.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/scaled_dot_product_attention.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/softmax.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/transpose.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/pass/pattern/op/label.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/pass/pattern/op/pattern.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/pass/pattern/op/wrap_type.hpp>

ನಾಮಾಕಾಶ ov {
ನಾಮಾಕಾಶ frontend {
ನಾಮಾಕಾಶ ggml {
ನಾಮಾಕಾಶ pass {

FuseToSDPA::FuseToSDPA() {
    // Not maintained since FLASH_ATTN_EXT has replaced this pattern
    ಧ್ರುವ auto m_k = ov::pass::pattern::any_ಒಳಹರಿವು();
    ಧ್ರುವ auto m_q = ov::pass::pattern::any_ಒಳಹರಿವು();
    ಧ್ರುವ auto m_qk = ov::pass::pattern::wrap_type<ov::op::v0::MatMul>({m_q, m_k});
    ಧ್ರುವ auto m_qk_f32 = ov::pass::pattern::wrap_type<ov::op::v0::Convert>({m_qk});
    ಧ್ರುವ auto m_scale = ov::pass::pattern::any_ಒಳಹರಿವು();
    ಧ್ರುವ auto m_scaled_qk = ov::pass::pattern::wrap_type<ov::op::v1::Multiply>({m_qk_f32, m_scale});
    ಧ್ರುವ auto m_mask = ov::pass::pattern::any_ಒಳಹರಿವು();
    ಧ್ರುವ auto m_masked_qk = ov::pass::pattern::wrap_type<ov::op::v1::Add>({m_scaled_qk, m_mask});
    ಧ್ರುವ auto m_softmax_qk = ov::pass::pattern::wrap_type<ov::op::v8::Softmax>({m_masked_qk});
    ಧ್ರುವ auto m_softmax_qk_f16 = ov::pass::pattern::wrap_type<ov::op::v0::Convert>({m_softmax_qk});
    ಧ್ರುವ auto m_v = ov::pass::pattern::any_ಒಳಹರಿವು();
    ಧ್ರುವ auto m_qkv = ov::pass::pattern::wrap_type<ov::op::v0::MatMul>({m_softmax_qk_f16, m_v});

    ಧ್ರುವ auto callback = [=](ov::pass::pattern::Matcher & m) {
        auto & pattern_to_output = m.get_pattern_ಮೌಲ್ಯ_map();
        auto k = pattern_to_output[m_k];
        auto q = pattern_to_output[m_q];
        auto v = pattern_to_output[m_v];
        auto mask = pattern_to_output[m_mask];
        auto scale = pattern_to_output[m_scale];

        auto mask_f16 = register_new_ಗಂಟು<ov::op::v0::Convert>(mask, ov::element::f16);
        auto scale_f16 = register_new_ಗಂಟು<ov::op::v0::Convert>(scale, ov::element::f16);
        auto sdpa = std::make_shared<ov::op::v13::ScaledDotProductAttention>(q, k, v, mask_f16, scale_f16, ಅಸತ್ಯ);

        ov::replace_ಗಂಟು(m.get_match_ಬೇರು(), sdpa);
        ov::copy_runtime_info(m.get_matched_nodes(), sdpa);

        ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
    };
    register_matcher(std::make_shared<ov::pass::pattern::Matcher>(m_qkv, "ov::frontend::ggml::pass::FuseToSDPA"),
                     callback);
}

}  // namespace pass
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
