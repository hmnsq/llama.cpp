#அந்தர்பாவ "fuse_to_sdpa.h"

#அந்தர்பாவ <openvino/core/graph_util.hpp>
#அந்தர்பாவ <openvino/core/rt_info.hpp>
#அந்தர்பாவ <openvino/op/add.hpp>
#அந்தர்பாவ <openvino/op/convert.hpp>
#அந்தர்பாவ <openvino/op/matmul.hpp>
#அந்தர்பாவ <openvino/op/multiply.hpp>
#அந்தர்பாவ <openvino/op/scaled_dot_product_attention.hpp>
#அந்தர்பாவ <openvino/op/softmax.hpp>
#அந்தர்பாவ <openvino/op/transpose.hpp>
#அந்தர்பாவ <openvino/pass/pattern/op/label.hpp>
#அந்தர்பாவ <openvino/pass/pattern/op/pattern.hpp>
#அந்தர்பாவ <openvino/pass/pattern/op/wrap_type.hpp>

நாமாகாஶ ov {
நாமாகாஶ frontend {
நாமாகாஶ ggml {
நாமாகாஶ pass {

FuseToSDPA::FuseToSDPA() {
    // Not maintained since FLASH_ATTN_EXT has replaced this pattern
    மாறிலி auto m_k = ov::pass::pattern::any_உள்ளீடு();
    மாறிலி auto m_q = ov::pass::pattern::any_உள்ளீடு();
    மாறிலி auto m_qk = ov::pass::pattern::wrap_type<ov::op::v0::MatMul>({m_q, m_k});
    மாறிலி auto m_qk_f32 = ov::pass::pattern::wrap_type<ov::op::v0::Convert>({m_qk});
    மாறிலி auto m_scale = ov::pass::pattern::any_உள்ளீடு();
    மாறிலி auto m_scaled_qk = ov::pass::pattern::wrap_type<ov::op::v1::Multiply>({m_qk_f32, m_scale});
    மாறிலி auto m_mask = ov::pass::pattern::any_உள்ளீடு();
    மாறிலி auto m_masked_qk = ov::pass::pattern::wrap_type<ov::op::v1::Add>({m_scaled_qk, m_mask});
    மாறிலி auto m_softmax_qk = ov::pass::pattern::wrap_type<ov::op::v8::Softmax>({m_masked_qk});
    மாறிலி auto m_softmax_qk_f16 = ov::pass::pattern::wrap_type<ov::op::v0::Convert>({m_softmax_qk});
    மாறிலி auto m_v = ov::pass::pattern::any_உள்ளீடு();
    மாறிலி auto m_qkv = ov::pass::pattern::wrap_type<ov::op::v0::MatMul>({m_softmax_qk_f16, m_v});

    மாறிலி auto callback = [=](ov::pass::pattern::Matcher & m) {
        auto & pattern_to_output = m.get_pattern_மதிப்பு_map();
        auto k = pattern_to_output[m_k];
        auto q = pattern_to_output[m_q];
        auto v = pattern_to_output[m_v];
        auto mask = pattern_to_output[m_mask];
        auto scale = pattern_to_output[m_scale];

        auto mask_f16 = register_new_முனை<ov::op::v0::Convert>(mask, ov::element::f16);
        auto scale_f16 = register_new_முனை<ov::op::v0::Convert>(scale, ov::element::f16);
        auto sdpa = std::make_shared<ov::op::v13::ScaledDotProductAttention>(q, k, v, mask_f16, scale_f16, பொய்);

        ov::replace_முனை(m.get_match_வேர்(), sdpa);
        ov::copy_runtime_info(m.get_matched_nodes(), sdpa);

        திரும்பு உண்மை;
    };
    register_matcher(std::make_shared<ov::pass::pattern::Matcher>(m_qkv, "ov::frontend::ggml::pass::FuseToSDPA"),
                     callback);
}

}  // namespace pass
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
