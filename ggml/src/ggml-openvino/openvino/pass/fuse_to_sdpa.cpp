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
    ثابت خودڪار m_k = ov::pass::pattern::any_داخلا();
    ثابت خودڪار m_q = ov::pass::pattern::any_داخلا();
    ثابت خودڪار m_qk = ov::pass::pattern::wrap_type<ov::op::v0::MatMul>({m_q, m_k});
    ثابت خودڪار m_qk_f32 = ov::pass::pattern::wrap_type<ov::op::v0::Convert>({m_qk});
    ثابت خودڪار m_scale = ov::pass::pattern::any_داخلا();
    ثابت خودڪار m_scaled_qk = ov::pass::pattern::wrap_type<ov::op::v1::Multiply>({m_qk_f32, m_scale});
    ثابت خودڪار m_نقاب = ov::pass::pattern::any_داخلا();
    ثابت خودڪار m_masked_qk = ov::pass::pattern::wrap_type<ov::op::v1::جوڙيو>({m_scaled_qk, m_نقاب});
    ثابت خودڪار m_softmax_qk = ov::pass::pattern::wrap_type<ov::op::v8::Softmax>({m_masked_qk});
    ثابت خودڪار m_softmax_qk_f16 = ov::pass::pattern::wrap_type<ov::op::v0::Convert>({m_softmax_qk});
    ثابت خودڪار m_v = ov::pass::pattern::any_داخلا();
    ثابت خودڪار m_qkv = ov::pass::pattern::wrap_type<ov::op::v0::MatMul>({m_softmax_qk_f16, m_v});

    ثابت خودڪار callback = [=](ov::pass::pattern::Matcher & m) {
        خودڪار & نمونو_منزل_output = m.get_نمونو_قدر_map();
        خودڪار k = نمونو_منزل_output[m_k];
        خودڪار q = نمونو_منزل_output[m_q];
        خودڪار v = نمونو_منزل_output[m_v];
        خودڪار نقاب = نمونو_منزل_output[m_نقاب];
        خودڪار scale = نمونو_منزل_output[m_scale];

        خودڪار نقاب_f16 = رجسٽر_new_ڳنڍ<ov::op::v0::Convert>(نقاب, ov::element::f16);
        خودڪار scale_f16 = رجسٽر_new_ڳنڍ<ov::op::v0::Convert>(scale, ov::element::f16);
        خودڪار sdpa = std::make_shared<ov::op::v13::Scaled_Dot_حاصل_ضرب_Attention>(q, k, v, نقاب_f16, scale_f16, خطأ);

        ov::replace_ڳنڍ(m.get_ميلاپ_پاڙ(), sdpa);
        ov::copy_runtime_info(m.get_matched_nodes(), sdpa);

        أرجع سچ;
    };
    رجسٽر_matcher(std::make_shared<ov::pass::pattern::Matcher>(m_qkv, "ov::frontend::ggml::pass::FuseToSDPA"),
                     callback);
}

}  // namespace pass
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
