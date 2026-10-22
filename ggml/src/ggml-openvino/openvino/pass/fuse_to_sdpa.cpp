#ଅନ୍ତର୍ଭାଵ "fuse_to_sdpa.h"

#ଅନ୍ତର୍ଭାଵ <openvino/core/graph_util.hpp>
#ଅନ୍ତର୍ଭାଵ <openvino/core/rt_info.hpp>
#ଅନ୍ତର୍ଭାଵ <openvino/op/add.hpp>
#ଅନ୍ତର୍ଭାଵ <openvino/op/convert.hpp>
#ଅନ୍ତର୍ଭାଵ <openvino/op/matmul.hpp>
#ଅନ୍ତର୍ଭାଵ <openvino/op/multiply.hpp>
#ଅନ୍ତର୍ଭାଵ <openvino/op/scaled_dot_product_attention.hpp>
#ଅନ୍ତର୍ଭାଵ <openvino/op/softmax.hpp>
#ଅନ୍ତର୍ଭାଵ <openvino/op/transpose.hpp>
#ଅନ୍ତର୍ଭାଵ <openvino/pass/pattern/op/label.hpp>
#ଅନ୍ତର୍ଭାଵ <openvino/pass/pattern/op/pattern.hpp>
#ଅନ୍ତର୍ଭାଵ <openvino/pass/pattern/op/wrap_type.hpp>

ନାମାକାଶ ov {
ନାମାକାଶ frontend {
ନାମାକାଶ ggml {
ନାମାକାଶ pass {

FuseToSDPA::FuseToSDPA() {
    // Not maintained since FLASH_ATTN_EXT has replaced this pattern
    ଅଚଳ auto m_k = ov::pass::pattern::any_ନିବେଶ();
    ଅଚଳ auto m_q = ov::pass::pattern::any_ନିବେଶ();
    ଅଚଳ auto m_qk = ov::pass::pattern::wrap_type<ov::op::v0::MatMul>({m_q, m_k});
    ଅଚଳ auto m_qk_f32 = ov::pass::pattern::wrap_type<ov::op::v0::Convert>({m_qk});
    ଅଚଳ auto m_scale = ov::pass::pattern::any_ନିବେଶ();
    ଅଚଳ auto m_scaled_qk = ov::pass::pattern::wrap_type<ov::op::v1::Multiply>({m_qk_f32, m_scale});
    ଅଚଳ auto m_mask = ov::pass::pattern::any_ନିବେଶ();
    ଅଚଳ auto m_masked_qk = ov::pass::pattern::wrap_type<ov::op::v1::Add>({m_scaled_qk, m_mask});
    ଅଚଳ auto m_softmax_qk = ov::pass::pattern::wrap_type<ov::op::v8::Softmax>({m_masked_qk});
    ଅଚଳ auto m_softmax_qk_f16 = ov::pass::pattern::wrap_type<ov::op::v0::Convert>({m_softmax_qk});
    ଅଚଳ auto m_v = ov::pass::pattern::any_ନିବେଶ();
    ଅଚଳ auto m_qkv = ov::pass::pattern::wrap_type<ov::op::v0::MatMul>({m_softmax_qk_f16, m_v});

    ଅଚଳ auto callback = [=](ov::pass::pattern::Matcher & m) {
        auto & pattern_to_output = m.get_pattern_ମୂଲ୍ୟ_map();
        auto k = pattern_to_output[m_k];
        auto q = pattern_to_output[m_q];
        auto v = pattern_to_output[m_v];
        auto mask = pattern_to_output[m_mask];
        auto scale = pattern_to_output[m_scale];

        auto mask_f16 = register_new_ଗଣ୍ଠି<ov::op::v0::Convert>(mask, ov::element::f16);
        auto scale_f16 = register_new_ଗଣ୍ଠି<ov::op::v0::Convert>(scale, ov::element::f16);
        auto sdpa = std::make_shared<ov::op::v13::ScaledDotProductAttention>(q, k, v, mask_f16, scale_f16, ମିଥ୍ୟା);

        ov::replace_ଗଣ୍ଠି(m.get_match_ମୂଳ(), sdpa);
        ov::copy_runtime_info(m.get_matched_nodes(), sdpa);

        ଫେରନ୍ତୁ ସତ୍ୟ;
    };
    register_matcher(std::make_shared<ov::pass::pattern::Matcher>(m_qkv, "ov::frontend::ggml::pass::FuseToSDPA"),
                     callback);
}

}  // namespace pass
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
