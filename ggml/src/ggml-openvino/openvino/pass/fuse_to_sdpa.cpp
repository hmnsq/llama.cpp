#شامل "fuse_to_sdpa.h"

#شامل <openvino/core/graph_util.hpp>
#شامل <openvino/core/rt_info.hpp>
#شامل <openvino/op/add.hpp>
#شامل <openvino/op/convert.hpp>
#شامل <openvino/op/matmul.hpp>
#شامل <openvino/op/multiply.hpp>
#شامل <openvino/op/scaled_dot_product_attention.hpp>
#شامل <openvino/op/softmax.hpp>
#شامل <openvino/op/transpose.hpp>
#شامل <openvino/pass/pattern/op/label.hpp>
#شامل <openvino/pass/pattern/op/pattern.hpp>
#شامل <openvino/pass/pattern/op/wrap_type.hpp>

فضاینام ov {
فضاینام frontend {
فضاینام ggml {
فضاینام pass {

FuseToSDPA::FuseToSDPA() {
    // Not maintained since FLASH_ATTN_EXT has replaced this pattern
    ثابت auto m_k = ov::pass::pattern::any_درون_داد();
    ثابت auto m_q = ov::pass::pattern::any_درون_داد();
    ثابت auto m_qk = ov::pass::pattern::wrap_type<ov::op::v0::MatMul>({m_q, m_k});
    ثابت auto m_qk_f32 = ov::pass::pattern::wrap_type<ov::op::v0::Convert>({m_qk});
    ثابت auto m_scale = ov::pass::pattern::any_درون_داد();
    ثابت auto m_scaled_qk = ov::pass::pattern::wrap_type<ov::op::v1::Multiply>({m_qk_f32, m_scale});
    ثابت auto m_mask = ov::pass::pattern::any_درون_داد();
    ثابت auto m_masked_qk = ov::pass::pattern::wrap_type<ov::op::v1::Add>({m_scaled_qk, m_mask});
    ثابت auto m_softmax_qk = ov::pass::pattern::wrap_type<ov::op::v8::Softmax>({m_masked_qk});
    ثابت auto m_softmax_qk_f16 = ov::pass::pattern::wrap_type<ov::op::v0::Convert>({m_softmax_qk});
    ثابت auto m_v = ov::pass::pattern::any_درون_داد();
    ثابت auto m_qkv = ov::pass::pattern::wrap_type<ov::op::v0::MatMul>({m_softmax_qk_f16, m_v});

    ثابت auto callback = [=](ov::pass::pattern::Matcher & m) {
        auto & pattern_to_output = m.get_pattern_مقدار_map();
        auto k = pattern_to_output[m_k];
        auto q = pattern_to_output[m_q];
        auto v = pattern_to_output[m_v];
        auto mask = pattern_to_output[m_mask];
        auto scale = pattern_to_output[m_scale];

        auto mask_f16 = register_new_گره<ov::op::v0::Convert>(mask, ov::element::f16);
        auto scale_f16 = register_new_گره<ov::op::v0::Convert>(scale, ov::element::f16);
        auto sdpa = std::make_shared<ov::op::v13::ScaledDotProductAttention>(q, k, v, mask_f16, scale_f16, نادرست);

        ov::replace_گره(m.get_match_ریشه(), sdpa);
        ov::copy_runtime_info(m.get_matched_nodes(), sdpa);

        بازگشت درست;
    };
    register_matcher(std::make_shared<ov::pass::pattern::Matcher>(m_qkv, "ov::frontend::ggml::pass::FuseToSDPA"),
                     callback);
}

}  // namespace pass
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
