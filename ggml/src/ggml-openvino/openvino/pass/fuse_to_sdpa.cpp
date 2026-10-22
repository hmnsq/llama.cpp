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

Fuse_ಗಮ್ಯ_SDPA::Fuse_ಗಮ್ಯ_SDPA() {
    // Not maintained since FLASH_ATTN_EXT has replaced this pattern
    ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ m_k = ov::pass::pattern::any_ಒಳಹರಿವು();
    ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ m_q = ov::pass::pattern::any_ಒಳಹರಿವು();
    ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ m_qk = ov::pass::pattern::wrap_type<ov::op::v0::MatMul>({m_q, m_k});
    ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ m_qk_f32 = ov::pass::pattern::wrap_type<ov::op::v0::Convert>({m_qk});
    ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ m_scale = ov::pass::pattern::any_ಒಳಹರಿವು();
    ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ m_scaled_qk = ov::pass::pattern::wrap_type<ov::op::v1::Multiply>({m_qk_f32, m_scale});
    ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ m_ಮುಖವಾಡ = ov::pass::pattern::any_ಒಳಹರಿವು();
    ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ m_masked_qk = ov::pass::pattern::wrap_type<ov::op::v1::ಕೂಡಿಸು>({m_scaled_qk, m_ಮುಖವಾಡ});
    ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ m_softmax_qk = ov::pass::pattern::wrap_type<ov::op::v8::Softmax>({m_masked_qk});
    ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ m_softmax_qk_f16 = ov::pass::pattern::wrap_type<ov::op::v0::Convert>({m_softmax_qk});
    ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ m_v = ov::pass::pattern::any_ಒಳಹರಿವು();
    ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ m_qkv = ov::pass::pattern::wrap_type<ov::op::v0::MatMul>({m_softmax_qk_f16, m_v});

    ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ callback = [=](ov::pass::pattern::Matcher & m) {
        ಸ್ವಯಂಚಾಲಿತ & ನಮೂನೆ_ಗಮ್ಯ_output = m.get_ನಮೂನೆ_ಮೌಲ್ಯ_map();
        ಸ್ವಯಂಚಾಲಿತ k = ನಮೂನೆ_ಗಮ್ಯ_output[m_k];
        ಸ್ವಯಂಚಾಲಿತ q = ನಮೂನೆ_ಗಮ್ಯ_output[m_q];
        ಸ್ವಯಂಚಾಲಿತ v = ನಮೂನೆ_ಗಮ್ಯ_output[m_v];
        ಸ್ವಯಂಚಾಲಿತ ಮುಖವಾಡ = ನಮೂನೆ_ಗಮ್ಯ_output[m_ಮುಖವಾಡ];
        ಸ್ವಯಂಚಾಲಿತ scale = ನಮೂನೆ_ಗಮ್ಯ_output[m_scale];

        ಸ್ವಯಂಚಾಲಿತ ಮುಖವಾಡ_f16 = ರಿಜಿಸ್ಟರ್_new_ಗಂಟು<ov::op::v0::Convert>(ಮುಖವಾಡ, ov::element::f16);
        ಸ್ವಯಂಚಾಲಿತ scale_f16 = ರಿಜಿಸ್ಟರ್_new_ಗಂಟು<ov::op::v0::Convert>(scale, ov::element::f16);
        ಸ್ವಯಂಚಾಲಿತ sdpa = std::make_shared<ov::op::v13::Scaled_Dot_ಗುಣಲಬ್ಧ_Attention>(q, k, v, ಮುಖವಾಡ_f16, scale_f16, ಸುಳ್ಳು);

        ov::replace_ಗಂಟು(m.get_ಹೊಂದಿಕೆ_ಬೇರು(), sdpa);
        ov::copy_runtime_info(m.get_matched_nodes(), sdpa);

        ಹಿಂತಿರುಗಿ ನಿಜ;
    };
    ರಿಜಿಸ್ಟರ್_matcher(std::make_shared<ov::pass::pattern::Matcher>(m_qkv, "ov::frontend::ggml::pass::FuseToSDPA"),
                     callback);
}

}  // namespace pass
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
