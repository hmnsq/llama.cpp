#ಅನ್ತರ್ಭಾವ "../node_context.h"
#ಅನ್ತರ್ಭಾವ "../op_table.h"
#ಅನ್ತರ್ಭಾವ "../utils.h"
#ಅನ್ತರ್ಭಾವ "ggml-openvino/ggml-openvino-extra.h"

#ಅನ್ತರ್ಭಾವ <cstddef>
#ಅನ್ತರ್ಭಾವ <cstdint>
#ಅನ್ತರ್ಭಾವ <memory>
#ಅನ್ತರ್ಭಾವ <openvino/op/add.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/broadcast.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/concat.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/constant.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/convert.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/matmul.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/multiply.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/reshape.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/scaled_dot_product_attention.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/slice.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/softmax.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/transpose.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/unsqueeze.hpp>
#ಅನ್ತರ್ಭಾವ <string>

ನಾಮಾಕಾಶ ov {
ನಾಮಾಕಾಶ frontend {
ನಾಮಾಕಾಶ ggml {
ನಾಮಾಕಾಶ op {
ಸ್ಥಿರ ov::Output<ov::Node> reshape_flat_kv(ಸ್ಥಿರಾಂಕ ov::Output<ov::Node> & kv_flat,
                                            size_t view_ಸ್ಥಾನಾಂತರ_ಬೈಟುಗಳು,
                                            size_t nb1_ಬೈಟುಗಳು,
                                            int64_t n_ತಲೆ,
                                            int64_t ತಲೆ_ಗಾತ್ರ,
                                            ಸ್ಥಿರಾಂಕ ov::Output<ov::Node> & attention_ಗಾತ್ರ) {
    int64_t n_state = n_ತಲೆ * ತಲೆ_ಗಾತ್ರ;
    int64_t layer_ಆರಂಭ_elem = (int64_t) (view_ಸ್ಥಾನಾಂತರ_ಬೈಟುಗಳು / (nb1_ಬೈಟುಗಳು / n_state));
    // Dynamic slice: [layer_start_elem, layer_start_elem + n_kv * n_state)
    ಸ್ವಯಂಚಾಲಿತ ಆರಂಭ_c = ov::op::v0::Constant::create(ov::element::i64, {1}, {layer_ಆರಂಭ_elem});
    ಸ್ವಯಂಚಾಲಿತ n_state_c = ov::op::v0::Constant::create(ov::element::i64, {1}, {n_state});
    // end = start + attention_size * n_state  (both static + dynamic)
    ಸ್ವಯಂಚಾಲಿತ kv_len_elems = std::make_shared<ov::op::v1::Multiply>(attention_ಗಾತ್ರ, n_state_c);
    ಸ್ವಯಂಚಾಲಿತ ಕೊನೆ_c = std::make_shared<ov::op::v1::ಕೂಡಿಸು>(ಆರಂಭ_c, kv_len_elems);
    ಸ್ವಯಂಚಾಲಿತ ಹೆಜ್ಜೆ_c = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
    ಸ್ವಯಂಚಾಲಿತ axis_c = ov::op::v0::Constant::create(ov::element::i64, {1}, {3});
    ಸ್ವಯಂಚಾಲಿತ sliced = std::make_shared<ov::op::v8::ತುಂಡು>(kv_flat, ಆರಂಭ_c, ಕೊನೆ_c, ಹೆಜ್ಜೆ_c, axis_c);

    // KV cache is laid out as {n_kv, n_head, head_size} in memory
    // Reshape to {1, n_kv, n_head, head_size}, then transpose to {1, n_head, n_kv, head_size}
    // as required by SDPA.
    ಸ್ವಯಂಚಾಲಿತ one_c = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
    ಸ್ವಯಂಚಾಲಿತ n_ತಲೆ_c = ov::op::v0::Constant::create(ov::element::i64, {1}, {n_ತಲೆ});
    ಸ್ವಯಂಚಾಲಿತ ತಲೆ_ಗಾತ್ರ_c = ov::op::v0::Constant::create(ov::element::i64, {1}, {ತಲೆ_ಗಾತ್ರ});
    // reshape: {n_kv*n_state} -> {1, n_kv, n_head, head_size}
    ಸ್ವಯಂಚಾಲಿತ new_shape =
        std::make_shared<ov::op::v0::Concat>(ov::Output_ಸದಿಶ{one_c, attention_ಗಾತ್ರ, n_ತಲೆ_c, ತಲೆ_ಗಾತ್ರ_c}, 0);
    ಸ್ವಯಂಚಾಲಿತ reshaped = std::make_shared<ov::op::v1::Reshape>(sliced, new_shape, ಸುಳ್ಳು);
    // transpose: {1, n_kv, n_head, head_size} -> {1, n_head, n_kv, head_size}
    ಸ್ವಯಂಚಾಲಿತ perm = ov::op::v0::Constant::create(ov::element::i64, {4}, {0, 2, 1, 3});
    ಸ್ವಯಂಚಾಲಿತ ret = std::make_shared<ov::op::v1::Transpose>(reshaped, perm);
    ಹಿಂತಿರುಗಿ ret;
}

Output_ಸದಿಶ translate_flash_attn_ext(ಸ್ಥಿರಾಂಕ ಗಂಟು_Context & context) {
    ಅಂಕೆ_inputs_check(context, 3, 4);
    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ has_ಮುಖವಾಡ = context.get_ಒಳಹರಿವು_ಗಾತ್ರ() == 4;
    ಸ್ವಯಂಚಾಲಿತ q_f32 = context.get_ಒಳಹರಿವು(0);
    ಸ್ವಯಂಚಾಲಿತ k = context.get_ಒಳಹರಿವು(1);
    ಸ್ವಯಂಚಾಲಿತ v = context.get_ಒಳಹರಿವು(2);
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ op_case = context.get_op_case();

    ಆದರೆ (op_case == 1 || op_case == 2) {
        int64_t n_state_ತಲೆ = (int64_t) context.get_view_ಒಳಹರಿವು_ggml_shape(1, 0)[3];
        int64_t n_ತಲೆ = (int64_t) context.get_view_ಒಳಹರಿವು_ggml_shape(1, 0)[1];
        size_t nb1 = context.get_view_ಒಳಹರಿವು_stride(1, 0)[2];
        size_t ಸ್ಥಾನಾಂತರ = context.get_view_ಒಳಹರಿವು_ಸ್ಥಾನಾಂತರ(1, 0);
        ov::Output<ov::Node> attention_ಗಾತ್ರ;
        ಆದರೆ (op_case == 1) {
            attention_ಗಾತ್ರ = context.get_ಒಳಹರಿವು("attention_size");
        } ಇಲ್ಲದಿದ್ದರೆ {
            attention_ಗಾತ್ರ = context.get_ಒಳಹರಿವು("attention_size_static");
        }
        k = reshape_flat_kv(k, ಸ್ಥಾನಾಂತರ, nb1, n_ತಲೆ, n_state_ತಲೆ, attention_ಗಾತ್ರ);
        v = reshape_flat_kv(v, ಸ್ಥಾನಾಂತರ, nb1, n_ತಲೆ, n_state_ತಲೆ, attention_ಗಾತ್ರ);
    }

    ದಶಮಾಂಶ * params = reinterpret_cast<ದಶಮಾಂಶ *>(context.get_output_op_params());
    ದಶಮಾಂಶ scale = params[0];
    // float max_bias      = params[1];
    // float logit_softcap = params[2];

    ಸ್ವಯಂಚಾಲಿತ q = std::make_shared<ov::op::v0::Convert>(q_f32, ov::element::f16);
    ಸ್ವಯಂಚಾಲಿತ scale_ಗಂಟು = std::make_shared<ov::op::v0::Constant>(ov::element::f16, ov::Shape{}, std::vector<ದಶಮಾಂಶ>{scale});

    ov::Output<ov::Node> res;

    // For stateful
    ov::Output<ov::Node> ಮುಖವಾಡ;
    ಆದರೆ (has_ಮುಖವಾಡ) {
        ಮುಖವಾಡ = context.get_ಒಳಹರಿವು(3);
        std::string ಮುಖವಾಡ_name = "KQ_mask_sliced";
        ಆದರೆ (context.get_ಒಳಹರಿವು_ಹೆಸರುಗಳು()[3].find("swa") != std::string::npos) {
            ಮುಖವಾಡ_name = "KQ_mask_swa_sliced";
        }
        ಆದರೆ (context.has_ಒಳಹರಿವು(ಮುಖವಾಡ_name)) {
            ಮುಖವಾಡ = context.get_ಒಳಹರಿವು(ಮುಖವಾಡ_name);
        }
        ಆದರೆ (ಮುಖವಾಡ.get_element_type() != ov::element::f16) {
            ಮುಖವಾಡ = std::make_shared<ov::op::v0::Convert>(ಮುಖವಾಡ, ov::element::f16);
        }
    }

    //auto tile_kv = [&](int64_t num_heads, int64_t num_heads_kv, int64_t head_size, ov::Output<Node> kv) {
    //    int64_t factor = num_heads / num_heads_kv;
    //    if (factor > 1 && num_heads_kv > 1) {
    ಸ್ವಯಂಚಾಲಿತ q_shape = context.get_ಒಳಹರಿವು_shape(0).ಗಮ್ಯ_shape();
    ಸ್ವಯಂಚಾಲಿತ k_shape = context.get_ಒಳಹರಿವು_shape(1).ಗಮ್ಯ_shape();
    ಸ್ಥಿರಾಂಕ int64_t ಅಂಕೆ_heads = q_shape[1];
    ಸ್ಥಿರಾಂಕ int64_t ಅಂಕೆ_heads_kv = k_shape[1];
    ಸ್ಥಿರಾಂಕ int64_t ತಲೆ_ಗಾತ್ರ = q_shape[3];
    ಸ್ಥಿರಾಂಕ int64_t ಗುಣಕ = ಅಂಕೆ_heads / ಅಂಕೆ_heads_kv;

    // Manual GQA attention: enabled by default on GPU in stateless mode.
    // Set GGML_OPENVINO_MANUAL_GQA_ATTN to a positive value (e.g. 1) to force-enable,
    // or to 0 to force-disable. Unset falls back to the device-based default.
    ಸ್ಥಿರ ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ manual_gqa_enabled = []() {
        ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * env = ggml_openvino_getenv_str("GGML_OPENVINO_MANUAL_GQA_ATTN");
        ಆದರೆ (env != ಶೂನ್ಯಸೂಚಕ) {
            ಹಿಂತಿರುಗಿ ggml_openvino_getenv_int("GGML_OPENVINO_MANUAL_GQA_ATTN") > 0;
        }
        ಹಿಂತಿರುಗಿ ggml_openvino_is_gpu();
    }();
    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ use_manual_gqa_attention =
        manual_gqa_enabled && ಗುಣಕ > 1 && ಅಂಕೆ_heads_kv > 1 && !context.is_stateful();

    ಆದರೆ (use_manual_gqa_attention) {
        // Q, K, V arrive as [B, n_heads(_kv), S, head_size], where B is the active
        // batch (n_seq_active) and may be > 1 (llama-perplexity, llama-server -np > 1)
        // or dynamic. Reshape to
        //   K_r: [B, num_heads_kv, 1, S, head_size]
        //   Q_r: [B, num_heads_kv, factor, S_q, head_size]
        // and let MatMul broadcast across the factor dim without materialising
        // an expanded K/V. The leading 0 + special_zero=true copies B at runtime,
        // so this is correct for B == 1, B > 1, and dynamic B alike. Only the head
        // dims and head_size are baked in as literals; the sequence dim stays -1.
        ಸ್ವಯಂಚಾಲಿತ k_5d_shape = ov::op::v0::Constant::create(ov::element::i64, {5},
                                                       std::vector<int64_t>{0, ಅಂಕೆ_heads_kv, 1, -1, ತಲೆ_ಗಾತ್ರ});
        ಸ್ವಯಂಚಾಲಿತ v_5d_shape = ov::op::v0::Constant::create(ov::element::i64, {5},
                                                       std::vector<int64_t>{0, ಅಂಕೆ_heads_kv, 1, -1, ತಲೆ_ಗಾತ್ರ});
        ಸ್ವಯಂಚಾಲಿತ q_5d_shape = ov::op::v0::Constant::create(ov::element::i64, {5},
                                                       std::vector<int64_t>{0, ಅಂಕೆ_heads_kv, ಗುಣಕ, -1, ತಲೆ_ಗಾತ್ರ});

        ಸ್ವಯಂಚಾಲಿತ k_r = std::make_shared<ov::op::v1::Reshape>(k, k_5d_shape, ನಿಜ);
        ಸ್ವಯಂಚಾಲಿತ v_r = std::make_shared<ov::op::v1::Reshape>(v, v_5d_shape, ನಿಜ);
        ಸ್ವಯಂಚಾಲಿತ q_r = std::make_shared<ov::op::v1::Reshape>(q, q_5d_shape, ನಿಜ);

        // QK^T → [B, num_heads_kv, factor, S_q, S_k]
        ಸ್ವಯಂಚಾಲಿತ qk = std::make_shared<ov::op::v0::MatMul>(q_r, k_r, /*tA=*/ಸುಳ್ಳು, /*tB=*/ನಿಜ);
        ಸ್ವಯಂಚಾಲಿತ qk_scaled = std::make_shared<ov::op::v1::Multiply>(qk, scale_ಗಂಟು);

        // Mask arrives as [B, 1, S_q, S_k]. Unsqueeze a factor axis at position 2 to
        // get [B, 1, 1, S_q, S_k], which NUMPY-broadcasts cleanly against the
        // [B, num_heads_kv, factor, S_q, S_k] scores: B==B, then 1→num_heads_kv and
        // 1→factor on the head dims.
        ov::Output<ov::Node> qk_masked;
        ಆದರೆ (has_ಮುಖವಾಡ) {
            ಸ್ವಯಂಚಾಲಿತ ಮುಖವಾಡ_unsq1 =
                std::make_shared<ov::op::v0::Unsqueeze>(ಮುಖವಾಡ, ov::op::v0::Constant::create(ov::element::i64, {1}, {2}));
            qk_masked = std::make_shared<ov::op::v1::ಕೂಡಿಸು>(qk_scaled, ಮುಖವಾಡ_unsq1);
        } ಇಲ್ಲದಿದ್ದರೆ {
            qk_masked = qk_scaled;
        }

        ಸ್ವಯಂಚಾಲಿತ softmax = std::make_shared<ov::op::v8::Softmax>(qk_masked, /*axis=*/-1);

        // softmax @ V → [B, num_heads_kv, factor, S_q, head_size]
        ಸ್ವಯಂಚಾಲಿತ attn = std::make_shared<ov::op::v0::MatMul>(softmax, v_r);

        // Reshape back to [B, num_heads, S_q, head_size] (combine num_heads_kv * factor).
        // Leading 0 + special_zero=true copies B at runtime.
        ಸ್ವಯಂಚಾಲಿತ ಹೊರಹೋಗು_4d_shape =
            ov::op::v0::Constant::create(ov::element::i64, {4}, std::vector<int64_t>{0, ಅಂಕೆ_heads, -1, ತಲೆ_ಗಾತ್ರ});
        ಸ್ವಯಂಚಾಲಿತ ಹೊರಹೋಗು_4d = std::make_shared<ov::op::v1::Reshape>(attn, ಹೊರಹೋಗು_4d_shape, ನಿಜ);

        // The standard SDPA path's downstream is Transpose(0,2,1,3) → Convert(f32).
        // Replicate it here so callers see the same output layout/dtype.
        res = std::make_shared<ov::op::v1::Transpose>(
            ಹೊರಹೋಗು_4d, ov::op::v0::Constant::create(ov::element::i64, {4}, {0, 2, 1, 3}));
        res = std::make_shared<ov::op::v0::Convert>(res, ov::element::f32);
        ಹಿಂತಿರುಗಿ rename_outputs_with_suffix({res}, context.get_name());
    }

    // Default path: explicit Broadcast → SDPA. Kept as the fallback because
    // (a) it goes through the GPU plugin's micro-SDPA fast path (FlashAttention
    // tiles via DPAS), and (b) the manual path above is still being validated.
    ಸ್ವಯಂಚಾಲಿತ tile_kv = [&](int64_t n_heads, int64_t n_heads_kv, int64_t hs, ov::Output<Node> kv) {
        int64_t f = n_heads / n_heads_kv;
        ಆದರೆ (f > 1 && n_heads_kv > 1) {
            ov::Output<ov::Node> kv_ಪ್ರಸಾರ_shape;
            ov::Output<ov::Node> kv_unsqueezed;
            ov::Output<ov::Node> new_kv_shape;
            ಸ್ವಯಂಚಾಲಿತ unsqueeze_axes = ov::op::v0::Constant::create(ov::element::i64, Shape{}, {2});
            kv_unsqueezed = std::make_shared<ov::op::v0::Unsqueeze>(kv, unsqueeze_axes);

            kv_ಪ್ರಸಾರ_shape = ov::op::v0::Constant::create(ov::element::i64, {5},
                                                              {(int64_t) 1, (int64_t) 1, f, (int64_t) 1, (int64_t) 1});
            new_kv_shape =
                ov::op::v0::Constant::create(ov::element::i64, {4}, {(int64_t) 0, n_heads, (int64_t) -1, hs});
            //    ov::element::i64, {5}, {(int64_t) 1, (int64_t) 1, factor, (int64_t) 1, (int64_t) 1});
            //new_kv_shape =
            //    ov::op::v0::Constant::create(ov::element::i64, {4}, {(int64_t) 0, num_heads, (int64_t) -1, head_size});

            kv = std::make_shared<ov::op::v3::ಪ್ರಸಾರ>(kv_unsqueezed, kv_ಪ್ರಸಾರ_shape,
                                                         ov::op::ಪ್ರಸಾರ_Type::BIDIRECTIONAL);
            kv = std::make_shared<ov::op::v1::Reshape>(kv, new_kv_shape, ನಿಜ);
        }
        ಹಿಂತಿರುಗಿ kv;
    };

    //auto q_shape = context.get_input_shape(0).to_shape();
    //auto k_shape = context.get_input_shape(1).to_shape();
    //k = tile_kv(q_shape[1], k_shape[1], q_shape[3], k);
    //v = tile_kv(q_shape[1], k_shape[1], q_shape[3], v);
    k = tile_kv(ಅಂಕೆ_heads, ಅಂಕೆ_heads_kv, ತಲೆ_ಗಾತ್ರ, k);
    v = tile_kv(ಅಂಕೆ_heads, ಅಂಕೆ_heads_kv, ತಲೆ_ಗಾತ್ರ, v);

    constexpr ಸ್ವಯಂಚಾಲಿತ causal = ಸುಳ್ಳು;
    ಆದರೆ (has_ಮುಖವಾಡ) {
        ಸ್ವಯಂಚಾಲಿತ sdpa = std::make_shared<ov::op::v13::Scaled_Dot_ಗುಣಲಬ್ಧ_Attention>(q, k, v, ಮುಖವಾಡ, scale_ಗಂಟು, causal);
        res = std::make_shared<ov::op::v1::Transpose>(
            sdpa, ov::op::v0::Constant::create(ov::element::i64, {4}, {0, 2, 1, 3}));
    } ಇಲ್ಲದಿದ್ದರೆ {
        ಸ್ವಯಂಚಾಲಿತ sdpa = std::make_shared<ov::op::v13::Scaled_Dot_ಗುಣಲಬ್ಧ_Attention>(q, k, v, scale_ಗಂಟು, causal);
        res = std::make_shared<ov::op::v1::Transpose>(
            sdpa, ov::op::v0::Constant::create(ov::element::i64, {4}, {0, 2, 1, 3}));
    }
    res = std::make_shared<ov::op::v0::Convert>(res, ov::element::f32);
    ಹಿಂತಿರುಗಿ rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
