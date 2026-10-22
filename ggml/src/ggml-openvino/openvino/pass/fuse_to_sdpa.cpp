#അന്തര്ഭാവ "fuse_to_sdpa.h"

#അന്തര്ഭാവ <openvino/core/graph_util.hpp>
#അന്തര്ഭാവ <openvino/core/rt_info.hpp>
#അന്തര്ഭാവ <openvino/op/add.hpp>
#അന്തര്ഭാവ <openvino/op/convert.hpp>
#അന്തര്ഭാവ <openvino/op/matmul.hpp>
#അന്തര്ഭാവ <openvino/op/multiply.hpp>
#അന്തര്ഭാവ <openvino/op/scaled_dot_product_attention.hpp>
#അന്തര്ഭാവ <openvino/op/softmax.hpp>
#അന്തര്ഭാവ <openvino/op/transpose.hpp>
#അന്തര്ഭാവ <openvino/pass/pattern/op/label.hpp>
#അന്തര്ഭാവ <openvino/pass/pattern/op/pattern.hpp>
#അന്തര്ഭാവ <openvino/pass/pattern/op/wrap_type.hpp>

നാമാകാശ ov {
നാമാകാശ frontend {
നാമാകാശ ggml {
നാമാകാശ pass {

FuseToSDPA::FuseToSDPA() {
    // Not maintained since FLASH_ATTN_EXT has replaced this pattern
    സ്ഥിരം auto m_k = ov::pass::pattern::any_നിവേശം();
    സ്ഥിരം auto m_q = ov::pass::pattern::any_നിവേശം();
    സ്ഥിരം auto m_qk = ov::pass::pattern::wrap_type<ov::op::v0::MatMul>({m_q, m_k});
    സ്ഥിരം auto m_qk_f32 = ov::pass::pattern::wrap_type<ov::op::v0::Convert>({m_qk});
    സ്ഥിരം auto m_scale = ov::pass::pattern::any_നിവേശം();
    സ്ഥിരം auto m_scaled_qk = ov::pass::pattern::wrap_type<ov::op::v1::Multiply>({m_qk_f32, m_scale});
    സ്ഥിരം auto m_mask = ov::pass::pattern::any_നിവേശം();
    സ്ഥിരം auto m_masked_qk = ov::pass::pattern::wrap_type<ov::op::v1::Add>({m_scaled_qk, m_mask});
    സ്ഥിരം auto m_softmax_qk = ov::pass::pattern::wrap_type<ov::op::v8::Softmax>({m_masked_qk});
    സ്ഥിരം auto m_softmax_qk_f16 = ov::pass::pattern::wrap_type<ov::op::v0::Convert>({m_softmax_qk});
    സ്ഥിരം auto m_v = ov::pass::pattern::any_നിവേശം();
    സ്ഥിരം auto m_qkv = ov::pass::pattern::wrap_type<ov::op::v0::MatMul>({m_softmax_qk_f16, m_v});

    സ്ഥിരം auto callback = [=](ov::pass::pattern::Matcher & m) {
        auto & pattern_to_output = m.get_pattern_മൂല്യം_map();
        auto k = pattern_to_output[m_k];
        auto q = pattern_to_output[m_q];
        auto v = pattern_to_output[m_v];
        auto mask = pattern_to_output[m_mask];
        auto scale = pattern_to_output[m_scale];

        auto mask_f16 = register_new_കണ്ണി<ov::op::v0::Convert>(mask, ov::element::f16);
        auto scale_f16 = register_new_കണ്ണി<ov::op::v0::Convert>(scale, ov::element::f16);
        auto sdpa = std::make_shared<ov::op::v13::ScaledDotProductAttention>(q, k, v, mask_f16, scale_f16, അസത്യം);

        ov::replace_കണ്ണി(m.get_match_വേര്(), sdpa);
        ov::copy_runtime_info(m.get_matched_nodes(), sdpa);

        തിരികെ സത്യം;
    };
    register_matcher(std::make_shared<ov::pass::pattern::Matcher>(m_qkv, "ov::frontend::ggml::pass::FuseToSDPA"),
                     callback);
}

}  // namespace pass
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
