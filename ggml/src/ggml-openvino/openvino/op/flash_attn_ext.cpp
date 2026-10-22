#అన్తర్భావ "../node_context.h"
#అన్తర్భావ "../op_table.h"
#అన్తర్భావ "../utils.h"
#అన్తర్భావ "ggml-openvino/ggml-openvino-extra.h"

#అన్తర్భావ <cstddef>
#అన్తర్భావ <cstdint>
#అన్తర్భావ <memory>
#అన్తర్భావ <openvino/op/add.hpp>
#అన్తర్భావ <openvino/op/broadcast.hpp>
#అన్తర్భావ <openvino/op/concat.hpp>
#అన్తర్భావ <openvino/op/constant.hpp>
#అన్తర్భావ <openvino/op/convert.hpp>
#అన్తర్భావ <openvino/op/matmul.hpp>
#అన్తర్భావ <openvino/op/multiply.hpp>
#అన్తర్భావ <openvino/op/reshape.hpp>
#అన్తర్భావ <openvino/op/scaled_dot_product_attention.hpp>
#అన్తర్భావ <openvino/op/slice.hpp>
#అన్తర్భావ <openvino/op/softmax.hpp>
#అన్తర్భావ <openvino/op/transpose.hpp>
#అన్తర్భావ <openvino/op/unsqueeze.hpp>
#అన్తర్భావ <string>

నామాకాశ ov {
నామాకాశ frontend {
నామాకాశ ggml {
నామాకాశ op {
స్థిర ov::Output<ov::Node> reshape_flat_kv(స్థిరరాశి ov::Output<ov::Node> & kv_flat,
                                            size_t view_స్థానభ్రంశం_బైట్లు,
                                            size_t nb1_బైట్లు,
                                            int64_t n_తల,
                                            int64_t తల_కొలత,
                                            స్థిరరాశి ov::Output<ov::Node> & attention_కొలత) {
    int64_t n_state = n_తల * తల_కొలత;
    int64_t layer_ప్రారంభం_elem = (int64_t) (view_స్థానభ్రంశం_బైట్లు / (nb1_బైట్లు / n_state));
    // Dynamic slice: [layer_start_elem, layer_start_elem + n_kv * n_state)
    స్వయంచాలక ప్రారంభం_c = ov::op::v0::Constant::create(ov::element::i64, {1}, {layer_ప్రారంభం_elem});
    స్వయంచాలక n_state_c = ov::op::v0::Constant::create(ov::element::i64, {1}, {n_state});
    // end = start + attention_size * n_state  (both static + dynamic)
    స్వయంచాలక kv_len_elems = std::make_shared<ov::op::v1::Multiply>(attention_కొలత, n_state_c);
    స్వయంచాలక చివర_c = std::make_shared<ov::op::v1::కూడు>(ప్రారంభం_c, kv_len_elems);
    స్వయంచాలక అడుగు_c = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
    స్వయంచాలక axis_c = ov::op::v0::Constant::create(ov::element::i64, {1}, {3});
    స్వయంచాలక sliced = std::make_shared<ov::op::v8::ముక్క>(kv_flat, ప్రారంభం_c, చివర_c, అడుగు_c, axis_c);

    // KV cache is laid out as {n_kv, n_head, head_size} in memory
    // Reshape to {1, n_kv, n_head, head_size}, then transpose to {1, n_head, n_kv, head_size}
    // as required by SDPA.
    స్వయంచాలక one_c = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
    స్వయంచాలక n_తల_c = ov::op::v0::Constant::create(ov::element::i64, {1}, {n_తల});
    స్వయంచాలక తల_కొలత_c = ov::op::v0::Constant::create(ov::element::i64, {1}, {తల_కొలత});
    // reshape: {n_kv*n_state} -> {1, n_kv, n_head, head_size}
    స్వయంచాలక new_shape =
        std::make_shared<ov::op::v0::Concat>(ov::Output_సదిశ{one_c, attention_కొలత, n_తల_c, తల_కొలత_c}, 0);
    స్వయంచాలక reshaped = std::make_shared<ov::op::v1::Reshape>(sliced, new_shape, అబద్ధం);
    // transpose: {1, n_kv, n_head, head_size} -> {1, n_head, n_kv, head_size}
    స్వయంచాలక perm = ov::op::v0::Constant::create(ov::element::i64, {4}, {0, 2, 1, 3});
    స్వయంచాలక ret = std::make_shared<ov::op::v1::Transpose>(reshaped, perm);
    తిరిగి ret;
}

Output_సదిశ translate_flash_attn_ext(స్థిరరాశి కణుపు_Context & context) {
    అంకె_inputs_check(context, 3, 4);
    స్థిరరాశి తార్కిక has_ముసుగు = context.get_ఆదానం_కొలత() == 4;
    స్వయంచాలక q_f32 = context.get_ఆదానం(0);
    స్వయంచాలక k = context.get_ఆదానం(1);
    స్వయంచాలక v = context.get_ఆదానం(2);
    స్థిరరాశి పూర్ణాంకం op_case = context.get_op_case();

    ఒకవేళ (op_case == 1 || op_case == 2) {
        int64_t n_state_తల = (int64_t) context.get_view_ఆదానం_ggml_shape(1, 0)[3];
        int64_t n_తల = (int64_t) context.get_view_ఆదానం_ggml_shape(1, 0)[1];
        size_t nb1 = context.get_view_ఆదానం_stride(1, 0)[2];
        size_t స్థానభ్రంశం = context.get_view_ఆదానం_స్థానభ్రంశం(1, 0);
        ov::Output<ov::Node> attention_కొలత;
        ఒకవేళ (op_case == 1) {
            attention_కొలత = context.get_ఆదానం("attention_size");
        } లేకపోతే {
            attention_కొలత = context.get_ఆదానం("attention_size_static");
        }
        k = reshape_flat_kv(k, స్థానభ్రంశం, nb1, n_తల, n_state_తల, attention_కొలత);
        v = reshape_flat_kv(v, స్థానభ్రంశం, nb1, n_తల, n_state_తల, attention_కొలత);
    }

    దశాంశం * params = reinterpret_cast<దశాంశం *>(context.get_output_op_params());
    దశాంశం scale = params[0];
    // float max_bias      = params[1];
    // float logit_softcap = params[2];

    స్వయంచాలక q = std::make_shared<ov::op::v0::Convert>(q_f32, ov::element::f16);
    స్వయంచాలక scale_కణుపు = std::make_shared<ov::op::v0::Constant>(ov::element::f16, ov::Shape{}, std::vector<దశాంశం>{scale});

    ov::Output<ov::Node> res;

    // For stateful
    ov::Output<ov::Node> ముసుగు;
    ఒకవేళ (has_ముసుగు) {
        ముసుగు = context.get_ఆదానం(3);
        std::string ముసుగు_name = "KQ_mask_sliced";
        ఒకవేళ (context.get_ఆదానం_పేర్లు()[3].find("swa") != std::string::npos) {
            ముసుగు_name = "KQ_mask_swa_sliced";
        }
        ఒకవేళ (context.has_ఆదానం(ముసుగు_name)) {
            ముసుగు = context.get_ఆదానం(ముసుగు_name);
        }
        ఒకవేళ (ముసుగు.get_element_type() != ov::element::f16) {
            ముసుగు = std::make_shared<ov::op::v0::Convert>(ముసుగు, ov::element::f16);
        }
    }

    //auto tile_kv = [&](int64_t num_heads, int64_t num_heads_kv, int64_t head_size, ov::Output<Node> kv) {
    //    int64_t factor = num_heads / num_heads_kv;
    //    if (factor > 1 && num_heads_kv > 1) {
    స్వయంచాలక q_shape = context.get_ఆదానం_shape(0).గమ్యం_shape();
    స్వయంచాలక k_shape = context.get_ఆదానం_shape(1).గమ్యం_shape();
    స్థిరరాశి int64_t అంకె_heads = q_shape[1];
    స్థిరరాశి int64_t అంకె_heads_kv = k_shape[1];
    స్థిరరాశి int64_t తల_కొలత = q_shape[3];
    స్థిరరాశి int64_t గుణకం = అంకె_heads / అంకె_heads_kv;

    // Manual GQA attention: enabled by default on GPU in stateless mode.
    // Set GGML_OPENVINO_MANUAL_GQA_ATTN to a positive value (e.g. 1) to force-enable,
    // or to 0 to force-disable. Unset falls back to the device-based default.
    స్థిర స్థిరరాశి తార్కిక manual_gqa_enabled = []() {
        స్థిరరాశి అక్షరం * env = ggml_openvino_getenv_str("GGML_OPENVINO_MANUAL_GQA_ATTN");
        ఒకవేళ (env != శూన్యసూచిక) {
            తిరిగి ggml_openvino_getenv_int("GGML_OPENVINO_MANUAL_GQA_ATTN") > 0;
        }
        తిరిగి ggml_openvino_is_gpu();
    }();
    స్థిరరాశి తార్కిక use_manual_gqa_attention =
        manual_gqa_enabled && గుణకం > 1 && అంకె_heads_kv > 1 && !context.is_stateful();

    ఒకవేళ (use_manual_gqa_attention) {
        // Q, K, V arrive as [B, n_heads(_kv), S, head_size], where B is the active
        // batch (n_seq_active) and may be > 1 (llama-perplexity, llama-server -np > 1)
        // or dynamic. Reshape to
        //   K_r: [B, num_heads_kv, 1, S, head_size]
        //   Q_r: [B, num_heads_kv, factor, S_q, head_size]
        // and let MatMul broadcast across the factor dim without materialising
        // an expanded K/V. The leading 0 + special_zero=true copies B at runtime,
        // so this is correct for B == 1, B > 1, and dynamic B alike. Only the head
        // dims and head_size are baked in as literals; the sequence dim stays -1.
        స్వయంచాలక k_5d_shape = ov::op::v0::Constant::create(ov::element::i64, {5},
                                                       std::vector<int64_t>{0, అంకె_heads_kv, 1, -1, తల_కొలత});
        స్వయంచాలక v_5d_shape = ov::op::v0::Constant::create(ov::element::i64, {5},
                                                       std::vector<int64_t>{0, అంకె_heads_kv, 1, -1, తల_కొలత});
        స్వయంచాలక q_5d_shape = ov::op::v0::Constant::create(ov::element::i64, {5},
                                                       std::vector<int64_t>{0, అంకె_heads_kv, గుణకం, -1, తల_కొలత});

        స్వయంచాలక k_r = std::make_shared<ov::op::v1::Reshape>(k, k_5d_shape, నిజం);
        స్వయంచాలక v_r = std::make_shared<ov::op::v1::Reshape>(v, v_5d_shape, నిజం);
        స్వయంచాలక q_r = std::make_shared<ov::op::v1::Reshape>(q, q_5d_shape, నిజం);

        // QK^T → [B, num_heads_kv, factor, S_q, S_k]
        స్వయంచాలక qk = std::make_shared<ov::op::v0::MatMul>(q_r, k_r, /*tA=*/అబద్ధం, /*tB=*/నిజం);
        స్వయంచాలక qk_scaled = std::make_shared<ov::op::v1::Multiply>(qk, scale_కణుపు);

        // Mask arrives as [B, 1, S_q, S_k]. Unsqueeze a factor axis at position 2 to
        // get [B, 1, 1, S_q, S_k], which NUMPY-broadcasts cleanly against the
        // [B, num_heads_kv, factor, S_q, S_k] scores: B==B, then 1→num_heads_kv and
        // 1→factor on the head dims.
        ov::Output<ov::Node> qk_masked;
        ఒకవేళ (has_ముసుగు) {
            స్వయంచాలక ముసుగు_unsq1 =
                std::make_shared<ov::op::v0::Unsqueeze>(ముసుగు, ov::op::v0::Constant::create(ov::element::i64, {1}, {2}));
            qk_masked = std::make_shared<ov::op::v1::కూడు>(qk_scaled, ముసుగు_unsq1);
        } లేకపోతే {
            qk_masked = qk_scaled;
        }

        స్వయంచాలక softmax = std::make_shared<ov::op::v8::Softmax>(qk_masked, /*axis=*/-1);

        // softmax @ V → [B, num_heads_kv, factor, S_q, head_size]
        స్వయంచాలక attn = std::make_shared<ov::op::v0::MatMul>(softmax, v_r);

        // Reshape back to [B, num_heads, S_q, head_size] (combine num_heads_kv * factor).
        // Leading 0 + special_zero=true copies B at runtime.
        స్వయంచాలక బహిర్గతం_4d_shape =
            ov::op::v0::Constant::create(ov::element::i64, {4}, std::vector<int64_t>{0, అంకె_heads, -1, తల_కొలత});
        స్వయంచాలక బహిర్గతం_4d = std::make_shared<ov::op::v1::Reshape>(attn, బహిర్గతం_4d_shape, నిజం);

        // The standard SDPA path's downstream is Transpose(0,2,1,3) → Convert(f32).
        // Replicate it here so callers see the same output layout/dtype.
        res = std::make_shared<ov::op::v1::Transpose>(
            బహిర్గతం_4d, ov::op::v0::Constant::create(ov::element::i64, {4}, {0, 2, 1, 3}));
        res = std::make_shared<ov::op::v0::Convert>(res, ov::element::f32);
        తిరిగి rename_outputs_with_suffix({res}, context.get_name());
    }

    // Default path: explicit Broadcast → SDPA. Kept as the fallback because
    // (a) it goes through the GPU plugin's micro-SDPA fast path (FlashAttention
    // tiles via DPAS), and (b) the manual path above is still being validated.
    స్వయంచాలక tile_kv = [&](int64_t n_heads, int64_t n_heads_kv, int64_t hs, ov::Output<Node> kv) {
        int64_t f = n_heads / n_heads_kv;
        ఒకవేళ (f > 1 && n_heads_kv > 1) {
            ov::Output<ov::Node> kv_ప్రసారం_shape;
            ov::Output<ov::Node> kv_unsqueezed;
            ov::Output<ov::Node> new_kv_shape;
            స్వయంచాలక unsqueeze_axes = ov::op::v0::Constant::create(ov::element::i64, Shape{}, {2});
            kv_unsqueezed = std::make_shared<ov::op::v0::Unsqueeze>(kv, unsqueeze_axes);

            kv_ప్రసారం_shape = ov::op::v0::Constant::create(ov::element::i64, {5},
                                                              {(int64_t) 1, (int64_t) 1, f, (int64_t) 1, (int64_t) 1});
            new_kv_shape =
                ov::op::v0::Constant::create(ov::element::i64, {4}, {(int64_t) 0, n_heads, (int64_t) -1, hs});
            //    ov::element::i64, {5}, {(int64_t) 1, (int64_t) 1, factor, (int64_t) 1, (int64_t) 1});
            //new_kv_shape =
            //    ov::op::v0::Constant::create(ov::element::i64, {4}, {(int64_t) 0, num_heads, (int64_t) -1, head_size});

            kv = std::make_shared<ov::op::v3::ప్రసారం>(kv_unsqueezed, kv_ప్రసారం_shape,
                                                         ov::op::ప్రసారం_Type::BIDIRECTIONAL);
            kv = std::make_shared<ov::op::v1::Reshape>(kv, new_kv_shape, నిజం);
        }
        తిరిగి kv;
    };

    //auto q_shape = context.get_input_shape(0).to_shape();
    //auto k_shape = context.get_input_shape(1).to_shape();
    //k = tile_kv(q_shape[1], k_shape[1], q_shape[3], k);
    //v = tile_kv(q_shape[1], k_shape[1], q_shape[3], v);
    k = tile_kv(అంకె_heads, అంకె_heads_kv, తల_కొలత, k);
    v = tile_kv(అంకె_heads, అంకె_heads_kv, తల_కొలత, v);

    constexpr స్వయంచాలక causal = అబద్ధం;
    ఒకవేళ (has_ముసుగు) {
        స్వయంచాలక sdpa = std::make_shared<ov::op::v13::Scaled_Dot_లబ్ధం_Attention>(q, k, v, ముసుగు, scale_కణుపు, causal);
        res = std::make_shared<ov::op::v1::Transpose>(
            sdpa, ov::op::v0::Constant::create(ov::element::i64, {4}, {0, 2, 1, 3}));
    } లేకపోతే {
        స్వయంచాలక sdpa = std::make_shared<ov::op::v13::Scaled_Dot_లబ్ధం_Attention>(q, k, v, scale_కణుపు, causal);
        res = std::make_shared<ov::op::v1::Transpose>(
            sdpa, ov::op::v0::Constant::create(ov::element::i64, {4}, {0, 2, 1, 3}));
    }
    res = std::make_shared<ov::op::v0::Convert>(res, ov::element::f32);
    తిరిగి rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
