#अन्तर्भाव "../node_context.h"
#अन्तर्भाव "../op_table.h"
#अन्तर्भाव "../utils.h"
#अन्तर्भाव "ggml-openvino/ggml-openvino-extra.h"

#अन्तर्भाव <cstddef>
#अन्तर्भाव <cstdint>
#अन्तर्भाव <memory>
#अन्तर्भाव <openvino/op/add.hpp>
#अन्तर्भाव <openvino/op/broadcast.hpp>
#अन्तर्भाव <openvino/op/concat.hpp>
#अन्तर्भाव <openvino/op/constant.hpp>
#अन्तर्भाव <openvino/op/convert.hpp>
#अन्तर्भाव <openvino/op/matmul.hpp>
#अन्तर्भाव <openvino/op/multiply.hpp>
#अन्तर्भाव <openvino/op/reshape.hpp>
#अन्तर्भाव <openvino/op/scaled_dot_product_attention.hpp>
#अन्तर्भाव <openvino/op/slice.hpp>
#अन्तर्भाव <openvino/op/softmax.hpp>
#अन्तर्भाव <openvino/op/transpose.hpp>
#अन्तर्भाव <openvino/op/unsqueeze.hpp>
#अन्तर्भाव <string>

नामाकाश ov {
नामाकाश frontend {
नामाकाश ggml {
नामाकाश op {
स्थिर ov::Output<ov::Node> reshape_flat_kv(ध्रुव ov::Output<ov::Node> & kv_flat,
                                            size_t view_विस्थापनम्_अष्टकानि,
                                            size_t nb1_अष्टकानि,
                                            int64_t n_शीर्ष,
                                            int64_t शीर्ष_आकार,
                                            ध्रुव ov::Output<ov::Node> & attention_आकार) {
    int64_t n_state = n_शीर्ष * शीर्ष_आकार;
    int64_t layer_start_elem = (int64_t) (view_विस्थापनम्_अष्टकानि / (nb1_अष्टकानि / n_state));
    // Dynamic slice: [layer_start_elem, layer_start_elem + n_kv * n_state)
    स्वचालित start_c = ov::op::v0::Constant::create(ov::element::i64, {1}, {layer_start_elem});
    स्वचालित n_state_c = ov::op::v0::Constant::create(ov::element::i64, {1}, {n_state});
    // end = start + attention_size * n_state  (both static + dynamic)
    स्वचालित kv_len_elems = std::make_shared<ov::op::v1::Multiply>(attention_आकार, n_state_c);
    स्वचालित अन्तः_c = std::make_shared<ov::op::v1::सङ्कलय>(start_c, kv_len_elems);
    स्वचालित चरणः_c = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
    स्वचालित axis_c = ov::op::v0::Constant::create(ov::element::i64, {1}, {3});
    स्वचालित sliced = std::make_shared<ov::op::v8::खण्डिका>(kv_flat, start_c, अन्तः_c, चरणः_c, axis_c);

    // KV cache is laid out as {n_kv, n_head, head_size} in memory
    // Reshape to {1, n_kv, n_head, head_size}, then transpose to {1, n_head, n_kv, head_size}
    // as required by SDPA.
    स्वचालित one_c = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
    स्वचालित n_शीर्ष_c = ov::op::v0::Constant::create(ov::element::i64, {1}, {n_शीर्ष});
    स्वचालित शीर्ष_आकार_c = ov::op::v0::Constant::create(ov::element::i64, {1}, {शीर्ष_आकार});
    // reshape: {n_kv*n_state} -> {1, n_kv, n_head, head_size}
    स्वचालित new_shape =
        std::make_shared<ov::op::v0::Concat>(ov::Output_सदिशम्{one_c, attention_आकार, n_शीर्ष_c, शीर्ष_आकार_c}, 0);
    स्वचालित reshaped = std::make_shared<ov::op::v1::Reshape>(sliced, new_shape, असत्य);
    // transpose: {1, n_kv, n_head, head_size} -> {1, n_head, n_kv, head_size}
    स्वचालित perm = ov::op::v0::Constant::create(ov::element::i64, {4}, {0, 2, 1, 3});
    स्वचालित ret = std::make_shared<ov::op::v1::Transpose>(reshaped, perm);
    प्रत्यागम ret;
}

Output_सदिशम् translate_flash_attn_ext(ध्रुव ग्रन्थिः_Context & context) {
    अङ्कः_inputs_check(context, 3, 4);
    ध्रुव तार्किक has_आवरणम् = context.get_निवेशः_आकार() == 4;
    स्वचालित q_f32 = context.get_निवेशः(0);
    स्वचालित k = context.get_निवेशः(1);
    स्वचालित v = context.get_निवेशः(2);
    ध्रुव पूर्णाङ्क op_case = context.get_op_case();

    यदि (op_case == 1 || op_case == 2) {
        int64_t n_state_शीर्ष = (int64_t) context.get_view_निवेशः_ggml_shape(1, 0)[3];
        int64_t n_शीर्ष = (int64_t) context.get_view_निवेशः_ggml_shape(1, 0)[1];
        size_t nb1 = context.get_view_निवेशः_stride(1, 0)[2];
        size_t विस्थापनम् = context.get_view_निवेशः_विस्थापनम्(1, 0);
        ov::Output<ov::Node> attention_आकार;
        यदि (op_case == 1) {
            attention_आकार = context.get_निवेशः("attention_size");
        } अन्यथा {
            attention_आकार = context.get_निवेशः("attention_size_static");
        }
        k = reshape_flat_kv(k, विस्थापनम्, nb1, n_शीर्ष, n_state_शीर्ष, attention_आकार);
        v = reshape_flat_kv(v, विस्थापनम्, nb1, n_शीर्ष, n_state_शीर्ष, attention_आकार);
    }

    भिन्न * params = reinterpret_cast<भिन्न *>(context.get_output_op_params());
    भिन्न scale = params[0];
    // float max_bias      = params[1];
    // float logit_softcap = params[2];

    स्वचालित q = std::make_shared<ov::op::v0::Convert>(q_f32, ov::element::f16);
    स्वचालित scale_ग्रन्थिः = std::make_shared<ov::op::v0::Constant>(ov::element::f16, ov::Shape{}, std::vector<भिन्न>{scale});

    ov::Output<ov::Node> res;

    // For stateful
    ov::Output<ov::Node> आवरणम्;
    यदि (has_आवरणम्) {
        आवरणम् = context.get_निवेशः(3);
        std::string आवरणम्_name = "KQ_mask_sliced";
        यदि (context.get_निवेशः_नामानि()[3].find("swa") != std::string::npos) {
            आवरणम्_name = "KQ_mask_swa_sliced";
        }
        यदि (context.has_निवेशः(आवरणम्_name)) {
            आवरणम् = context.get_निवेशः(आवरणम्_name);
        }
        यदि (आवरणम्.get_element_type() != ov::element::f16) {
            आवरणम् = std::make_shared<ov::op::v0::Convert>(आवरणम्, ov::element::f16);
        }
    }

    //auto tile_kv = [&](int64_t num_heads, int64_t num_heads_kv, int64_t head_size, ov::Output<Node> kv) {
    //    int64_t factor = num_heads / num_heads_kv;
    //    if (factor > 1 && num_heads_kv > 1) {
    स्वचालित q_shape = context.get_निवेशः_shape(0).गन्तव्यम्_shape();
    स्वचालित k_shape = context.get_निवेशः_shape(1).गन्तव्यम्_shape();
    ध्रुव int64_t अङ्कः_heads = q_shape[1];
    ध्रुव int64_t अङ्कः_heads_kv = k_shape[1];
    ध्रुव int64_t शीर्ष_आकार = q_shape[3];
    ध्रुव int64_t गुणकः = अङ्कः_heads / अङ्कः_heads_kv;

    // Manual GQA attention: enabled by default on GPU in stateless mode.
    // Set GGML_OPENVINO_MANUAL_GQA_ATTN to a positive value (e.g. 1) to force-enable,
    // or to 0 to force-disable. Unset falls back to the device-based default.
    स्थिर ध्रुव तार्किक manual_gqa_enabled = []() {
        ध्रुव वर्ण * env = ggml_openvino_getenv_str("GGML_OPENVINO_MANUAL_GQA_ATTN");
        यदि (env != शून्यसूचकः) {
            प्रत्यागम ggml_openvino_getenv_int("GGML_OPENVINO_MANUAL_GQA_ATTN") > 0;
        }
        प्रत्यागम ggml_openvino_is_gpu();
    }();
    ध्रुव तार्किक use_manual_gqa_attention =
        manual_gqa_enabled && गुणकः > 1 && अङ्कः_heads_kv > 1 && !context.is_stateful();

    यदि (use_manual_gqa_attention) {
        // Q, K, V arrive as [B, n_heads(_kv), S, head_size], where B is the active
        // batch (n_seq_active) and may be > 1 (llama-perplexity, llama-server -np > 1)
        // or dynamic. Reshape to
        //   K_r: [B, num_heads_kv, 1, S, head_size]
        //   Q_r: [B, num_heads_kv, factor, S_q, head_size]
        // and let MatMul broadcast across the factor dim without materialising
        // an expanded K/V. The leading 0 + special_zero=true copies B at runtime,
        // so this is correct for B == 1, B > 1, and dynamic B alike. Only the head
        // dims and head_size are baked in as literals; the sequence dim stays -1.
        स्वचालित k_5d_shape = ov::op::v0::Constant::create(ov::element::i64, {5},
                                                       std::vector<int64_t>{0, अङ्कः_heads_kv, 1, -1, शीर्ष_आकार});
        स्वचालित v_5d_shape = ov::op::v0::Constant::create(ov::element::i64, {5},
                                                       std::vector<int64_t>{0, अङ्कः_heads_kv, 1, -1, शीर्ष_आकार});
        स्वचालित q_5d_shape = ov::op::v0::Constant::create(ov::element::i64, {5},
                                                       std::vector<int64_t>{0, अङ्कः_heads_kv, गुणकः, -1, शीर्ष_आकार});

        स्वचालित k_r = std::make_shared<ov::op::v1::Reshape>(k, k_5d_shape, सत्य);
        स्वचालित v_r = std::make_shared<ov::op::v1::Reshape>(v, v_5d_shape, सत्य);
        स्वचालित q_r = std::make_shared<ov::op::v1::Reshape>(q, q_5d_shape, सत्य);

        // QK^T → [B, num_heads_kv, factor, S_q, S_k]
        स्वचालित qk = std::make_shared<ov::op::v0::MatMul>(q_r, k_r, /*tA=*/असत्य, /*tB=*/सत्य);
        स्वचालित qk_scaled = std::make_shared<ov::op::v1::Multiply>(qk, scale_ग्रन्थिः);

        // Mask arrives as [B, 1, S_q, S_k]. Unsqueeze a factor axis at position 2 to
        // get [B, 1, 1, S_q, S_k], which NUMPY-broadcasts cleanly against the
        // [B, num_heads_kv, factor, S_q, S_k] scores: B==B, then 1→num_heads_kv and
        // 1→factor on the head dims.
        ov::Output<ov::Node> qk_masked;
        यदि (has_आवरणम्) {
            स्वचालित आवरणम्_unsq1 =
                std::make_shared<ov::op::v0::Unsqueeze>(आवरणम्, ov::op::v0::Constant::create(ov::element::i64, {1}, {2}));
            qk_masked = std::make_shared<ov::op::v1::सङ्कलय>(qk_scaled, आवरणम्_unsq1);
        } अन्यथा {
            qk_masked = qk_scaled;
        }

        स्वचालित softmax = std::make_shared<ov::op::v8::Softmax>(qk_masked, /*axis=*/-1);

        // softmax @ V → [B, num_heads_kv, factor, S_q, head_size]
        स्वचालित attn = std::make_shared<ov::op::v0::MatMul>(softmax, v_r);

        // Reshape back to [B, num_heads, S_q, head_size] (combine num_heads_kv * factor).
        // Leading 0 + special_zero=true copies B at runtime.
        स्वचालित बहिर्गमनम्_4d_shape =
            ov::op::v0::Constant::create(ov::element::i64, {4}, std::vector<int64_t>{0, अङ्कः_heads, -1, शीर्ष_आकार});
        स्वचालित बहिर्गमनम्_4d = std::make_shared<ov::op::v1::Reshape>(attn, बहिर्गमनम्_4d_shape, सत्य);

        // The standard SDPA path's downstream is Transpose(0,2,1,3) → Convert(f32).
        // Replicate it here so callers see the same output layout/dtype.
        res = std::make_shared<ov::op::v1::Transpose>(
            बहिर्गमनम्_4d, ov::op::v0::Constant::create(ov::element::i64, {4}, {0, 2, 1, 3}));
        res = std::make_shared<ov::op::v0::Convert>(res, ov::element::f32);
        प्रत्यागम rename_outputs_with_suffix({res}, context.get_name());
    }

    // Default path: explicit Broadcast → SDPA. Kept as the fallback because
    // (a) it goes through the GPU plugin's micro-SDPA fast path (FlashAttention
    // tiles via DPAS), and (b) the manual path above is still being validated.
    स्वचालित tile_kv = [&](int64_t n_heads, int64_t n_heads_kv, int64_t hs, ov::Output<Node> kv) {
        int64_t f = n_heads / n_heads_kv;
        यदि (f > 1 && n_heads_kv > 1) {
            ov::Output<ov::Node> kv_प्रसारणम्_shape;
            ov::Output<ov::Node> kv_unsqueezed;
            ov::Output<ov::Node> new_kv_shape;
            स्वचालित unsqueeze_axes = ov::op::v0::Constant::create(ov::element::i64, Shape{}, {2});
            kv_unsqueezed = std::make_shared<ov::op::v0::Unsqueeze>(kv, unsqueeze_axes);

            kv_प्रसारणम्_shape = ov::op::v0::Constant::create(ov::element::i64, {5},
                                                              {(int64_t) 1, (int64_t) 1, f, (int64_t) 1, (int64_t) 1});
            new_kv_shape =
                ov::op::v0::Constant::create(ov::element::i64, {4}, {(int64_t) 0, n_heads, (int64_t) -1, hs});
            //    ov::element::i64, {5}, {(int64_t) 1, (int64_t) 1, factor, (int64_t) 1, (int64_t) 1});
            //new_kv_shape =
            //    ov::op::v0::Constant::create(ov::element::i64, {4}, {(int64_t) 0, num_heads, (int64_t) -1, head_size});

            kv = std::make_shared<ov::op::v3::प्रसारणम्>(kv_unsqueezed, kv_प्रसारणम्_shape,
                                                         ov::op::प्रसारणम्_Type::BIDIRECTIONAL);
            kv = std::make_shared<ov::op::v1::Reshape>(kv, new_kv_shape, सत्य);
        }
        प्रत्यागम kv;
    };

    //auto q_shape = context.get_input_shape(0).to_shape();
    //auto k_shape = context.get_input_shape(1).to_shape();
    //k = tile_kv(q_shape[1], k_shape[1], q_shape[3], k);
    //v = tile_kv(q_shape[1], k_shape[1], q_shape[3], v);
    k = tile_kv(अङ्कः_heads, अङ्कः_heads_kv, शीर्ष_आकार, k);
    v = tile_kv(अङ्कः_heads, अङ्कः_heads_kv, शीर्ष_आकार, v);

    constexpr स्वचालित causal = असत्य;
    यदि (has_आवरणम्) {
        स्वचालित sdpa = std::make_shared<ov::op::v13::Scaled_Dot_गुणनफलम्_Attention>(q, k, v, आवरणम्, scale_ग्रन्थिः, causal);
        res = std::make_shared<ov::op::v1::Transpose>(
            sdpa, ov::op::v0::Constant::create(ov::element::i64, {4}, {0, 2, 1, 3}));
    } अन्यथा {
        स्वचालित sdpa = std::make_shared<ov::op::v13::Scaled_Dot_गुणनफलम्_Attention>(q, k, v, scale_ग्रन्थिः, causal);
        res = std::make_shared<ov::op::v1::Transpose>(
            sdpa, ov::op::v0::Constant::create(ov::element::i64, {4}, {0, 2, 1, 3}));
    }
    res = std::make_shared<ov::op::v0::Convert>(res, ov::element::f32);
    प्रत्यागम rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
