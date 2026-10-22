#ਅਨ੍ਤਰ੍ਭਾਵ "fuse_to_sdpa.h"

#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/core/graph_util.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/core/rt_info.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/add.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/convert.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/matmul.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/multiply.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/scaled_dot_product_attention.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/softmax.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/transpose.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/pass/pattern/op/label.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/pass/pattern/op/pattern.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/pass/pattern/op/wrap_type.hpp>

ਨਾਮਾਕਾਸ਼ ov {
ਨਾਮਾਕਾਸ਼ frontend {
ਨਾਮਾਕਾਸ਼ ggml {
ਨਾਮਾਕਾਸ਼ pass {

Fuse_ਮੰਜ਼ਿਲ_SDPA::Fuse_ਮੰਜ਼ਿਲ_SDPA() {
    // Not maintained since FLASH_ATTN_EXT has replaced this pattern
    ਅਚਲ ਸਵੈਚਾਲਿਤ m_k = ov::pass::pattern::any_ਨਿਵੇਸ਼();
    ਅਚਲ ਸਵੈਚਾਲਿਤ m_q = ov::pass::pattern::any_ਨਿਵੇਸ਼();
    ਅਚਲ ਸਵੈਚਾਲਿਤ m_qk = ov::pass::pattern::wrap_type<ov::op::v0::MatMul>({m_q, m_k});
    ਅਚਲ ਸਵੈਚਾਲਿਤ m_qk_f32 = ov::pass::pattern::wrap_type<ov::op::v0::Convert>({m_qk});
    ਅਚਲ ਸਵੈਚਾਲਿਤ m_scale = ov::pass::pattern::any_ਨਿਵੇਸ਼();
    ਅਚਲ ਸਵੈਚਾਲਿਤ m_scaled_qk = ov::pass::pattern::wrap_type<ov::op::v1::Multiply>({m_qk_f32, m_scale});
    ਅਚਲ ਸਵੈਚਾਲਿਤ m_ਨਕਾਬ = ov::pass::pattern::any_ਨਿਵੇਸ਼();
    ਅਚਲ ਸਵੈਚਾਲਿਤ m_masked_qk = ov::pass::pattern::wrap_type<ov::op::v1::ਜੋੜੋ>({m_scaled_qk, m_ਨਕਾਬ});
    ਅਚਲ ਸਵੈਚਾਲਿਤ m_softmax_qk = ov::pass::pattern::wrap_type<ov::op::v8::Softmax>({m_masked_qk});
    ਅਚਲ ਸਵੈਚਾਲਿਤ m_softmax_qk_f16 = ov::pass::pattern::wrap_type<ov::op::v0::Convert>({m_softmax_qk});
    ਅਚਲ ਸਵੈਚਾਲਿਤ m_v = ov::pass::pattern::any_ਨਿਵੇਸ਼();
    ਅਚਲ ਸਵੈਚਾਲਿਤ m_qkv = ov::pass::pattern::wrap_type<ov::op::v0::MatMul>({m_softmax_qk_f16, m_v});

    ਅਚਲ ਸਵੈਚਾਲਿਤ callback = [=](ov::pass::pattern::Matcher & m) {
        ਸਵੈਚਾਲਿਤ & ਨਮੂਨਾ_ਮੰਜ਼ਿਲ_output = m.get_ਨਮੂਨਾ_ਮੁੱਲ_map();
        ਸਵੈਚਾਲਿਤ k = ਨਮੂਨਾ_ਮੰਜ਼ਿਲ_output[m_k];
        ਸਵੈਚਾਲਿਤ q = ਨਮੂਨਾ_ਮੰਜ਼ਿਲ_output[m_q];
        ਸਵੈਚਾਲਿਤ v = ਨਮੂਨਾ_ਮੰਜ਼ਿਲ_output[m_v];
        ਸਵੈਚਾਲਿਤ ਨਕਾਬ = ਨਮੂਨਾ_ਮੰਜ਼ਿਲ_output[m_ਨਕਾਬ];
        ਸਵੈਚਾਲਿਤ scale = ਨਮੂਨਾ_ਮੰਜ਼ਿਲ_output[m_scale];

        ਸਵੈਚਾਲਿਤ ਨਕਾਬ_f16 = ਰਜਿਸਟਰ_new_ਗੰਢ<ov::op::v0::Convert>(ਨਕਾਬ, ov::element::f16);
        ਸਵੈਚਾਲਿਤ scale_f16 = ਰਜਿਸਟਰ_new_ਗੰਢ<ov::op::v0::Convert>(scale, ov::element::f16);
        ਸਵੈਚਾਲਿਤ sdpa = std::make_shared<ov::op::v13::Scaled_Dot_ਗੁਣਨਫਲ_Attention>(q, k, v, ਨਕਾਬ_f16, scale_f16, ਝੂਠ);

        ov::replace_ਗੰਢ(m.get_ਮੇਲ_ਜੜ੍ਹ(), sdpa);
        ov::copy_runtime_info(m.get_matched_nodes(), sdpa);

        ਵਾਪਸ ਸੱਚ;
    };
    ਰਜਿਸਟਰ_matcher(std::make_shared<ov::pass::pattern::Matcher>(m_qkv, "ov::frontend::ggml::pass::FuseToSDPA"),
                     callback);
}

}  // namespace pass
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
