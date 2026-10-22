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

نامخلا ov {
نامخلا frontend {
نامخلا ggml {
نامخلا pass {

Fuse_مقصد_SDPA::Fuse_مقصد_SDPA() {
    // Not maintained since FLASH_ATTN_EXT has replaced this pattern
    ثابت خودکار m_k = ov::pass::pattern::any_داخلہ();
    ثابت خودکار m_q = ov::pass::pattern::any_داخلہ();
    ثابت خودکار m_qk = ov::pass::pattern::wrap_type<ov::op::v0::MatMul>({m_q, m_k});
    ثابت خودکار m_qk_f32 = ov::pass::pattern::wrap_type<ov::op::v0::Convert>({m_qk});
    ثابت خودکار m_scale = ov::pass::pattern::any_داخلہ();
    ثابت خودکار m_scaled_qk = ov::pass::pattern::wrap_type<ov::op::v1::Multiply>({m_qk_f32, m_scale});
    ثابت خودکار m_نقاب = ov::pass::pattern::any_داخلہ();
    ثابت خودکار m_masked_qk = ov::pass::pattern::wrap_type<ov::op::v1::جوڑو>({m_scaled_qk, m_نقاب});
    ثابت خودکار m_softmax_qk = ov::pass::pattern::wrap_type<ov::op::v8::Softmax>({m_masked_qk});
    ثابت خودکار m_softmax_qk_f16 = ov::pass::pattern::wrap_type<ov::op::v0::Convert>({m_softmax_qk});
    ثابت خودکار m_v = ov::pass::pattern::any_داخلہ();
    ثابت خودکار m_qkv = ov::pass::pattern::wrap_type<ov::op::v0::MatMul>({m_softmax_qk_f16, m_v});

    ثابت خودکار callback = [=](ov::pass::pattern::Matcher & m) {
        خودکار & نمونہ_مقصد_output = m.get_نمونہ_قدر_map();
        خودکار k = نمونہ_مقصد_output[m_k];
        خودکار q = نمونہ_مقصد_output[m_q];
        خودکار v = نمونہ_مقصد_output[m_v];
        خودکار نقاب = نمونہ_مقصد_output[m_نقاب];
        خودکار scale = نمونہ_مقصد_output[m_scale];

        خودکار نقاب_f16 = رجسٹر_new_گرہ<ov::op::v0::Convert>(نقاب, ov::element::f16);
        خودکار scale_f16 = رجسٹر_new_گرہ<ov::op::v0::Convert>(scale, ov::element::f16);
        خودکار sdpa = std::make_shared<ov::op::v13::Scaled_Dot_حاصل_ضرب_Attention>(q, k, v, نقاب_f16, scale_f16, جھوٹ);

        ov::replace_گرہ(m.get_مطابقت_جڑ(), sdpa);
        ov::copy_runtime_info(m.get_matched_nodes(), sdpa);

        واپس سچ;
    };
    رجسٹر_matcher(std::make_shared<ov::pass::pattern::Matcher>(m_qkv, "ov::frontend::ggml::pass::FuseToSDPA"),
                     callback);
}

}  // namespace pass
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
