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

Fuse_مقصد_SDPA::Fuse_مقصد_SDPA() {
    // Not maintained since FLASH_ATTN_EXT has replaced this pattern
    ثابت خودکار m_k = ov::pass::pattern::any_درون_داد();
    ثابت خودکار m_q = ov::pass::pattern::any_درون_داد();
    ثابت خودکار m_qk = ov::pass::pattern::wrap_type<ov::op::v0::MatMul>({m_q, m_k});
    ثابت خودکار m_qk_f32 = ov::pass::pattern::wrap_type<ov::op::v0::Convert>({m_qk});
    ثابت خودکار m_scale = ov::pass::pattern::any_درون_داد();
    ثابت خودکار m_scaled_qk = ov::pass::pattern::wrap_type<ov::op::v1::Multiply>({m_qk_f32, m_scale});
    ثابت خودکار m_ماسک = ov::pass::pattern::any_درون_داد();
    ثابت خودکار m_masked_qk = ov::pass::pattern::wrap_type<ov::op::v1::جمع_کن>({m_scaled_qk, m_ماسک});
    ثابت خودکار m_softmax_qk = ov::pass::pattern::wrap_type<ov::op::v8::Softmax>({m_masked_qk});
    ثابت خودکار m_softmax_qk_f16 = ov::pass::pattern::wrap_type<ov::op::v0::Convert>({m_softmax_qk});
    ثابت خودکار m_v = ov::pass::pattern::any_درون_داد();
    ثابت خودکار m_qkv = ov::pass::pattern::wrap_type<ov::op::v0::MatMul>({m_softmax_qk_f16, m_v});

    ثابت خودکار callback = [=](ov::pass::pattern::Matcher & m) {
        خودکار & الگو_مقصد_output = m.get_الگو_مقدار_map();
        خودکار k = الگو_مقصد_output[m_k];
        خودکار q = الگو_مقصد_output[m_q];
        خودکار v = الگو_مقصد_output[m_v];
        خودکار ماسک = الگو_مقصد_output[m_ماسک];
        خودکار scale = الگو_مقصد_output[m_scale];

        خودکار ماسک_f16 = ثبات_new_گره<ov::op::v0::Convert>(ماسک, ov::element::f16);
        خودکار scale_f16 = ثبات_new_گره<ov::op::v0::Convert>(scale, ov::element::f16);
        خودکار sdpa = std::make_shared<ov::op::v13::Scaled_Dot_حاصلضرب_Attention>(q, k, v, ماسک_f16, scale_f16, خطأ);

        ov::replace_گره(m.get_تطابق_ریشه(), sdpa);
        ov::copy_runtime_info(m.get_matched_nodes(), sdpa);

        أرجع درست;
    };
    ثبات_matcher(std::make_shared<ov::pass::pattern::Matcher>(m_qkv, "ov::frontend::ggml::pass::FuseToSDPA"),
                     callback);
}

}  // namespace pass
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
