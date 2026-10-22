#ਅਨ੍ਤਰ੍ਭਾਵ "../node_context.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "../op_table.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "../utils.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-openvino/ggml-openvino-extra.h"

#ਅਨ੍ਤਰ੍ਭਾਵ <cstddef>
#ਅਨ੍ਤਰ੍ਭਾਵ <cstdint>
#ਅਨ੍ਤਰ੍ਭਾਵ <memory>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/add.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/broadcast.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/concat.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/constant.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/convert.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/matmul.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/multiply.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/reshape.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/scaled_dot_product_attention.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/slice.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/softmax.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/transpose.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/unsqueeze.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <string>

ਨਾਮਾਕਾਸ਼ ov {
ਨਾਮਾਕਾਸ਼ frontend {
ਨਾਮਾਕਾਸ਼ ggml {
ਨਾਮਾਕਾਸ਼ op {
ਸਥਿਰ ov::Output<ov::Node> reshape_flat_kv(ਅਚਲ ov::Output<ov::Node> & kv_flat,
                                            size_t view_ਵਿੱਥ_ਬਾਈਟਾਂ,
                                            size_t nb1_ਬਾਈਟਾਂ,
                                            int64_t n_ਸਿਰ,
                                            int64_t ਸਿਰ_ਮਾਪ,
                                            ਅਚਲ ov::Output<ov::Node> & attention_ਮਾਪ) {
    int64_t n_state = n_ਸਿਰ * ਸਿਰ_ਮਾਪ;
    int64_t layer_ਸ਼ੁਰੂ_elem = (int64_t) (view_ਵਿੱਥ_ਬਾਈਟਾਂ / (nb1_ਬਾਈਟਾਂ / n_state));
    // Dynamic slice: [layer_start_elem, layer_start_elem + n_kv * n_state)
    ਸਵੈਚਾਲਿਤ ਸ਼ੁਰੂ_c = ov::op::v0::Constant::create(ov::element::i64, {1}, {layer_ਸ਼ੁਰੂ_elem});
    ਸਵੈਚਾਲਿਤ n_state_c = ov::op::v0::Constant::create(ov::element::i64, {1}, {n_state});
    // end = start + attention_size * n_state  (both static + dynamic)
    ਸਵੈਚਾਲਿਤ kv_len_elems = std::make_shared<ov::op::v1::Multiply>(attention_ਮਾਪ, n_state_c);
    ਸਵੈਚਾਲਿਤ ਸਿਰਾ_c = std::make_shared<ov::op::v1::ਜੋੜੋ>(ਸ਼ੁਰੂ_c, kv_len_elems);
    ਸਵੈਚਾਲਿਤ ਕਦਮ_c = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
    ਸਵੈਚਾਲਿਤ axis_c = ov::op::v0::Constant::create(ov::element::i64, {1}, {3});
    ਸਵੈਚਾਲਿਤ sliced = std::make_shared<ov::op::v8::ਟੁਕੜਾ>(kv_flat, ਸ਼ੁਰੂ_c, ਸਿਰਾ_c, ਕਦਮ_c, axis_c);

    // KV cache is laid out as {n_kv, n_head, head_size} in memory
    // Reshape to {1, n_kv, n_head, head_size}, then transpose to {1, n_head, n_kv, head_size}
    // as required by SDPA.
    ਸਵੈਚਾਲਿਤ one_c = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
    ਸਵੈਚਾਲਿਤ n_ਸਿਰ_c = ov::op::v0::Constant::create(ov::element::i64, {1}, {n_ਸਿਰ});
    ਸਵੈਚਾਲਿਤ ਸਿਰ_ਮਾਪ_c = ov::op::v0::Constant::create(ov::element::i64, {1}, {ਸਿਰ_ਮਾਪ});
    // reshape: {n_kv*n_state} -> {1, n_kv, n_head, head_size}
    ਸਵੈਚਾਲਿਤ new_shape =
        std::make_shared<ov::op::v0::Concat>(ov::Output_ਸਦਿਸ਼{one_c, attention_ਮਾਪ, n_ਸਿਰ_c, ਸਿਰ_ਮਾਪ_c}, 0);
    ਸਵੈਚਾਲਿਤ reshaped = std::make_shared<ov::op::v1::Reshape>(sliced, new_shape, ਝੂਠ);
    // transpose: {1, n_kv, n_head, head_size} -> {1, n_head, n_kv, head_size}
    ਸਵੈਚਾਲਿਤ perm = ov::op::v0::Constant::create(ov::element::i64, {4}, {0, 2, 1, 3});
    ਸਵੈਚਾਲਿਤ ret = std::make_shared<ov::op::v1::Transpose>(reshaped, perm);
    ਵਾਪਸ ret;
}

Output_ਸਦਿਸ਼ translate_flash_attn_ext(ਅਚਲ ਗੰਢ_Context & context) {
    ਅੰਕ_inputs_check(context, 3, 4);
    ਅਚਲ ਤਾਰਕਿਕ has_ਨਕਾਬ = context.get_ਨਿਵੇਸ਼_ਮਾਪ() == 4;
    ਸਵੈਚਾਲਿਤ q_f32 = context.get_ਨਿਵੇਸ਼(0);
    ਸਵੈਚਾਲਿਤ k = context.get_ਨਿਵੇਸ਼(1);
    ਸਵੈਚਾਲਿਤ v = context.get_ਨਿਵੇਸ਼(2);
    ਅਚਲ ਪੂਰਨ op_case = context.get_op_case();

    ਜੇ (op_case == 1 || op_case == 2) {
        int64_t n_state_ਸਿਰ = (int64_t) context.get_view_ਨਿਵੇਸ਼_ggml_shape(1, 0)[3];
        int64_t n_ਸਿਰ = (int64_t) context.get_view_ਨਿਵੇਸ਼_ggml_shape(1, 0)[1];
        size_t nb1 = context.get_view_ਨਿਵੇਸ਼_stride(1, 0)[2];
        size_t ਵਿੱਥ = context.get_view_ਨਿਵੇਸ਼_ਵਿੱਥ(1, 0);
        ov::Output<ov::Node> attention_ਮਾਪ;
        ਜੇ (op_case == 1) {
            attention_ਮਾਪ = context.get_ਨਿਵੇਸ਼("attention_size");
        } ਨਹੀਂਤਾਂ {
            attention_ਮਾਪ = context.get_ਨਿਵੇਸ਼("attention_size_static");
        }
        k = reshape_flat_kv(k, ਵਿੱਥ, nb1, n_ਸਿਰ, n_state_ਸਿਰ, attention_ਮਾਪ);
        v = reshape_flat_kv(v, ਵਿੱਥ, nb1, n_ਸਿਰ, n_state_ਸਿਰ, attention_ਮਾਪ);
    }

    ਦਸ਼ਮਲਵ * params = reinterpret_cast<ਦਸ਼ਮਲਵ *>(context.get_output_op_params());
    ਦਸ਼ਮਲਵ scale = params[0];
    // float max_bias      = params[1];
    // float logit_softcap = params[2];

    ਸਵੈਚਾਲਿਤ q = std::make_shared<ov::op::v0::Convert>(q_f32, ov::element::f16);
    ਸਵੈਚਾਲਿਤ scale_ਗੰਢ = std::make_shared<ov::op::v0::Constant>(ov::element::f16, ov::Shape{}, std::vector<ਦਸ਼ਮਲਵ>{scale});

    ov::Output<ov::Node> res;

    // For stateful
    ov::Output<ov::Node> ਨਕਾਬ;
    ਜੇ (has_ਨਕਾਬ) {
        ਨਕਾਬ = context.get_ਨਿਵੇਸ਼(3);
        std::string ਨਕਾਬ_name = "KQ_mask_sliced";
        ਜੇ (context.get_ਨਿਵੇਸ਼_ਨਾਂ()[3].find("swa") != std::string::npos) {
            ਨਕਾਬ_name = "KQ_mask_swa_sliced";
        }
        ਜੇ (context.has_ਨਿਵੇਸ਼(ਨਕਾਬ_name)) {
            ਨਕਾਬ = context.get_ਨਿਵੇਸ਼(ਨਕਾਬ_name);
        }
        ਜੇ (ਨਕਾਬ.get_element_type() != ov::element::f16) {
            ਨਕਾਬ = std::make_shared<ov::op::v0::Convert>(ਨਕਾਬ, ov::element::f16);
        }
    }

    //auto tile_kv = [&](int64_t num_heads, int64_t num_heads_kv, int64_t head_size, ov::Output<Node> kv) {
    //    int64_t factor = num_heads / num_heads_kv;
    //    if (factor > 1 && num_heads_kv > 1) {
    ਸਵੈਚਾਲਿਤ q_shape = context.get_ਨਿਵੇਸ਼_shape(0).ਮੰਜ਼ਿਲ_shape();
    ਸਵੈਚਾਲਿਤ k_shape = context.get_ਨਿਵੇਸ਼_shape(1).ਮੰਜ਼ਿਲ_shape();
    ਅਚਲ int64_t ਅੰਕ_heads = q_shape[1];
    ਅਚਲ int64_t ਅੰਕ_heads_kv = k_shape[1];
    ਅਚਲ int64_t ਸਿਰ_ਮਾਪ = q_shape[3];
    ਅਚਲ int64_t ਗੁਣਕ = ਅੰਕ_heads / ਅੰਕ_heads_kv;

    // Manual GQA attention: enabled by default on GPU in stateless mode.
    // Set GGML_OPENVINO_MANUAL_GQA_ATTN to a positive value (e.g. 1) to force-enable,
    // or to 0 to force-disable. Unset falls back to the device-based default.
    ਸਥਿਰ ਅਚਲ ਤਾਰਕਿਕ manual_gqa_enabled = []() {
        ਅਚਲ ਅੱਖਰ * env = ggml_openvino_getenv_str("GGML_OPENVINO_MANUAL_GQA_ATTN");
        ਜੇ (env != ਖਾਲੀ_ਸੂਚਕ) {
            ਵਾਪਸ ggml_openvino_getenv_int("GGML_OPENVINO_MANUAL_GQA_ATTN") > 0;
        }
        ਵਾਪਸ ggml_openvino_is_gpu();
    }();
    ਅਚਲ ਤਾਰਕਿਕ use_manual_gqa_attention =
        manual_gqa_enabled && ਗੁਣਕ > 1 && ਅੰਕ_heads_kv > 1 && !context.is_stateful();

    ਜੇ (use_manual_gqa_attention) {
        // Q, K, V arrive as [B, n_heads(_kv), S, head_size], where B is the active
        // batch (n_seq_active) and may be > 1 (llama-perplexity, llama-server -np > 1)
        // or dynamic. Reshape to
        //   K_r: [B, num_heads_kv, 1, S, head_size]
        //   Q_r: [B, num_heads_kv, factor, S_q, head_size]
        // and let MatMul broadcast across the factor dim without materialising
        // an expanded K/V. The leading 0 + special_zero=true copies B at runtime,
        // so this is correct for B == 1, B > 1, and dynamic B alike. Only the head
        // dims and head_size are baked in as literals; the sequence dim stays -1.
        ਸਵੈਚਾਲਿਤ k_5d_shape = ov::op::v0::Constant::create(ov::element::i64, {5},
                                                       std::vector<int64_t>{0, ਅੰਕ_heads_kv, 1, -1, ਸਿਰ_ਮਾਪ});
        ਸਵੈਚਾਲਿਤ v_5d_shape = ov::op::v0::Constant::create(ov::element::i64, {5},
                                                       std::vector<int64_t>{0, ਅੰਕ_heads_kv, 1, -1, ਸਿਰ_ਮਾਪ});
        ਸਵੈਚਾਲਿਤ q_5d_shape = ov::op::v0::Constant::create(ov::element::i64, {5},
                                                       std::vector<int64_t>{0, ਅੰਕ_heads_kv, ਗੁਣਕ, -1, ਸਿਰ_ਮਾਪ});

        ਸਵੈਚਾਲਿਤ k_r = std::make_shared<ov::op::v1::Reshape>(k, k_5d_shape, ਸੱਚ);
        ਸਵੈਚਾਲਿਤ v_r = std::make_shared<ov::op::v1::Reshape>(v, v_5d_shape, ਸੱਚ);
        ਸਵੈਚਾਲਿਤ q_r = std::make_shared<ov::op::v1::Reshape>(q, q_5d_shape, ਸੱਚ);

        // QK^T → [B, num_heads_kv, factor, S_q, S_k]
        ਸਵੈਚਾਲਿਤ qk = std::make_shared<ov::op::v0::MatMul>(q_r, k_r, /*tA=*/ਝੂਠ, /*tB=*/ਸੱਚ);
        ਸਵੈਚਾਲਿਤ qk_scaled = std::make_shared<ov::op::v1::Multiply>(qk, scale_ਗੰਢ);

        // Mask arrives as [B, 1, S_q, S_k]. Unsqueeze a factor axis at position 2 to
        // get [B, 1, 1, S_q, S_k], which NUMPY-broadcasts cleanly against the
        // [B, num_heads_kv, factor, S_q, S_k] scores: B==B, then 1→num_heads_kv and
        // 1→factor on the head dims.
        ov::Output<ov::Node> qk_masked;
        ਜੇ (has_ਨਕਾਬ) {
            ਸਵੈਚਾਲਿਤ ਨਕਾਬ_unsq1 =
                std::make_shared<ov::op::v0::Unsqueeze>(ਨਕਾਬ, ov::op::v0::Constant::create(ov::element::i64, {1}, {2}));
            qk_masked = std::make_shared<ov::op::v1::ਜੋੜੋ>(qk_scaled, ਨਕਾਬ_unsq1);
        } ਨਹੀਂਤਾਂ {
            qk_masked = qk_scaled;
        }

        ਸਵੈਚਾਲਿਤ softmax = std::make_shared<ov::op::v8::Softmax>(qk_masked, /*axis=*/-1);

        // softmax @ V → [B, num_heads_kv, factor, S_q, head_size]
        ਸਵੈਚਾਲਿਤ attn = std::make_shared<ov::op::v0::MatMul>(softmax, v_r);

        // Reshape back to [B, num_heads, S_q, head_size] (combine num_heads_kv * factor).
        // Leading 0 + special_zero=true copies B at runtime.
        ਸਵੈਚਾਲਿਤ ਬਾਹਰਲਾ_4d_shape =
            ov::op::v0::Constant::create(ov::element::i64, {4}, std::vector<int64_t>{0, ਅੰਕ_heads, -1, ਸਿਰ_ਮਾਪ});
        ਸਵੈਚਾਲਿਤ ਬਾਹਰਲਾ_4d = std::make_shared<ov::op::v1::Reshape>(attn, ਬਾਹਰਲਾ_4d_shape, ਸੱਚ);

        // The standard SDPA path's downstream is Transpose(0,2,1,3) → Convert(f32).
        // Replicate it here so callers see the same output layout/dtype.
        res = std::make_shared<ov::op::v1::Transpose>(
            ਬਾਹਰਲਾ_4d, ov::op::v0::Constant::create(ov::element::i64, {4}, {0, 2, 1, 3}));
        res = std::make_shared<ov::op::v0::Convert>(res, ov::element::f32);
        ਵਾਪਸ rename_outputs_with_suffix({res}, context.get_name());
    }

    // Default path: explicit Broadcast → SDPA. Kept as the fallback because
    // (a) it goes through the GPU plugin's micro-SDPA fast path (FlashAttention
    // tiles via DPAS), and (b) the manual path above is still being validated.
    ਸਵੈਚਾਲਿਤ tile_kv = [&](int64_t n_heads, int64_t n_heads_kv, int64_t hs, ov::Output<Node> kv) {
        int64_t f = n_heads / n_heads_kv;
        ਜੇ (f > 1 && n_heads_kv > 1) {
            ov::Output<ov::Node> kv_ਪ੍ਰਸਾਰਣ_shape;
            ov::Output<ov::Node> kv_unsqueezed;
            ov::Output<ov::Node> new_kv_shape;
            ਸਵੈਚਾਲਿਤ unsqueeze_axes = ov::op::v0::Constant::create(ov::element::i64, Shape{}, {2});
            kv_unsqueezed = std::make_shared<ov::op::v0::Unsqueeze>(kv, unsqueeze_axes);

            kv_ਪ੍ਰਸਾਰਣ_shape = ov::op::v0::Constant::create(ov::element::i64, {5},
                                                              {(int64_t) 1, (int64_t) 1, f, (int64_t) 1, (int64_t) 1});
            new_kv_shape =
                ov::op::v0::Constant::create(ov::element::i64, {4}, {(int64_t) 0, n_heads, (int64_t) -1, hs});
            //    ov::element::i64, {5}, {(int64_t) 1, (int64_t) 1, factor, (int64_t) 1, (int64_t) 1});
            //new_kv_shape =
            //    ov::op::v0::Constant::create(ov::element::i64, {4}, {(int64_t) 0, num_heads, (int64_t) -1, head_size});

            kv = std::make_shared<ov::op::v3::ਪ੍ਰਸਾਰਣ>(kv_unsqueezed, kv_ਪ੍ਰਸਾਰਣ_shape,
                                                         ov::op::ਪ੍ਰਸਾਰਣ_Type::BIDIRECTIONAL);
            kv = std::make_shared<ov::op::v1::Reshape>(kv, new_kv_shape, ਸੱਚ);
        }
        ਵਾਪਸ kv;
    };

    //auto q_shape = context.get_input_shape(0).to_shape();
    //auto k_shape = context.get_input_shape(1).to_shape();
    //k = tile_kv(q_shape[1], k_shape[1], q_shape[3], k);
    //v = tile_kv(q_shape[1], k_shape[1], q_shape[3], v);
    k = tile_kv(ਅੰਕ_heads, ਅੰਕ_heads_kv, ਸਿਰ_ਮਾਪ, k);
    v = tile_kv(ਅੰਕ_heads, ਅੰਕ_heads_kv, ਸਿਰ_ਮਾਪ, v);

    constexpr ਸਵੈਚਾਲਿਤ causal = ਝੂਠ;
    ਜੇ (has_ਨਕਾਬ) {
        ਸਵੈਚਾਲਿਤ sdpa = std::make_shared<ov::op::v13::Scaled_Dot_ਗੁਣਨਫਲ_Attention>(q, k, v, ਨਕਾਬ, scale_ਗੰਢ, causal);
        res = std::make_shared<ov::op::v1::Transpose>(
            sdpa, ov::op::v0::Constant::create(ov::element::i64, {4}, {0, 2, 1, 3}));
    } ਨਹੀਂਤਾਂ {
        ਸਵੈਚਾਲਿਤ sdpa = std::make_shared<ov::op::v13::Scaled_Dot_ਗੁਣਨਫਲ_Attention>(q, k, v, scale_ਗੰਢ, causal);
        res = std::make_shared<ov::op::v1::Transpose>(
            sdpa, ov::op::v0::Constant::create(ov::element::i64, {4}, {0, 2, 1, 3}));
    }
    res = std::make_shared<ov::op::v0::Convert>(res, ov::element::f32);
    ਵਾਪਸ rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
