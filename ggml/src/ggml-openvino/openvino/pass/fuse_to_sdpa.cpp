#అన్తర్భావ "fuse_to_sdpa.h"

#అన్తర్భావ <openvino/core/graph_util.hpp>
#అన్తర్భావ <openvino/core/rt_info.hpp>
#అన్తర్భావ <openvino/op/add.hpp>
#అన్తర్భావ <openvino/op/convert.hpp>
#అన్తర్భావ <openvino/op/matmul.hpp>
#అన్తర్భావ <openvino/op/multiply.hpp>
#అన్తర్భావ <openvino/op/scaled_dot_product_attention.hpp>
#అన్తర్భావ <openvino/op/softmax.hpp>
#అన్తర్భావ <openvino/op/transpose.hpp>
#అన్తర్భావ <openvino/pass/pattern/op/label.hpp>
#అన్తర్భావ <openvino/pass/pattern/op/pattern.hpp>
#అన్తర్భావ <openvino/pass/pattern/op/wrap_type.hpp>

నామాకాశ ov {
నామాకాశ frontend {
నామాకాశ ggml {
నామాకాశ pass {

Fuse_గమ్యం_SDPA::Fuse_గమ్యం_SDPA() {
    // Not maintained since FLASH_ATTN_EXT has replaced this pattern
    స్థిరరాశి స్వయంచాలక m_k = ov::pass::pattern::any_ఆదానం();
    స్థిరరాశి స్వయంచాలక m_q = ov::pass::pattern::any_ఆదానం();
    స్థిరరాశి స్వయంచాలక m_qk = ov::pass::pattern::wrap_type<ov::op::v0::MatMul>({m_q, m_k});
    స్థిరరాశి స్వయంచాలక m_qk_f32 = ov::pass::pattern::wrap_type<ov::op::v0::Convert>({m_qk});
    స్థిరరాశి స్వయంచాలక m_scale = ov::pass::pattern::any_ఆదానం();
    స్థిరరాశి స్వయంచాలక m_scaled_qk = ov::pass::pattern::wrap_type<ov::op::v1::Multiply>({m_qk_f32, m_scale});
    స్థిరరాశి స్వయంచాలక m_ముసుగు = ov::pass::pattern::any_ఆదానం();
    స్థిరరాశి స్వయంచాలక m_masked_qk = ov::pass::pattern::wrap_type<ov::op::v1::కూడు>({m_scaled_qk, m_ముసుగు});
    స్థిరరాశి స్వయంచాలక m_softmax_qk = ov::pass::pattern::wrap_type<ov::op::v8::Softmax>({m_masked_qk});
    స్థిరరాశి స్వయంచాలక m_softmax_qk_f16 = ov::pass::pattern::wrap_type<ov::op::v0::Convert>({m_softmax_qk});
    స్థిరరాశి స్వయంచాలక m_v = ov::pass::pattern::any_ఆదానం();
    స్థిరరాశి స్వయంచాలక m_qkv = ov::pass::pattern::wrap_type<ov::op::v0::MatMul>({m_softmax_qk_f16, m_v});

    స్థిరరాశి స్వయంచాలక callback = [=](ov::pass::pattern::Matcher & m) {
        స్వయంచాలక & నమూనా_గమ్యం_output = m.get_నమూనా_విలువ_map();
        స్వయంచాలక k = నమూనా_గమ్యం_output[m_k];
        స్వయంచాలక q = నమూనా_గమ్యం_output[m_q];
        స్వయంచాలక v = నమూనా_గమ్యం_output[m_v];
        స్వయంచాలక ముసుగు = నమూనా_గమ్యం_output[m_ముసుగు];
        స్వయంచాలక scale = నమూనా_గమ్యం_output[m_scale];

        స్వయంచాలక ముసుగు_f16 = రిజిస్టరు_new_కణుపు<ov::op::v0::Convert>(ముసుగు, ov::element::f16);
        స్వయంచాలక scale_f16 = రిజిస్టరు_new_కణుపు<ov::op::v0::Convert>(scale, ov::element::f16);
        స్వయంచాలక sdpa = std::make_shared<ov::op::v13::Scaled_Dot_లబ్ధం_Attention>(q, k, v, ముసుగు_f16, scale_f16, అబద్ధం);

        ov::replace_కణుపు(m.get_సరిపోలిక_మూలం(), sdpa);
        ov::copy_runtime_info(m.get_matched_nodes(), sdpa);

        తిరిగి నిజం;
    };
    రిజిస్టరు_matcher(std::make_shared<ov::pass::pattern::Matcher>(m_qkv, "ov::frontend::ggml::pass::FuseToSDPA"),
                     callback);
}

}  // namespace pass
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
