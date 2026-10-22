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
स्थिर ov::Output<ov::Node> reshape_flat_kv(अचर ov::Output<ov::Node> & kv_flat,
                                            माप_प्रकार view_offset_bytes,
                                            माप_प्रकार nb1_bytes,
                                            int64_t n_head,
                                            int64_t head_size,
                                            अचर ov::Output<ov::Node> & attention_size) {
    int64_t n_state = n_head * head_size;
    int64_t layer_start_elem = (int64_t) (view_offset_bytes / (nb1_bytes / n_state));
    // Dynamic slice: [layer_start_elem, layer_start_elem + n_kv * n_state)
    स्वतः start_c = ov::op::v0::Constant::create(ov::element::i64, {1}, {layer_start_elem});
    स्वतः n_state_c = ov::op::v0::Constant::create(ov::element::i64, {1}, {n_state});
    // end = start + attention_size * n_state  (both static + dynamic)
    स्वतः kv_len_elems = std::make_shared<ov::op::v1::Multiply>(attention_size, n_state_c);
    स्वतः end_c = std::make_shared<ov::op::v1::Add>(start_c, kv_len_elems);
    स्वतः step_c = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
    स्वतः axis_c = ov::op::v0::Constant::create(ov::element::i64, {1}, {3});
    स्वतः sliced = std::make_shared<ov::op::v8::Slice>(kv_flat, start_c, end_c, step_c, axis_c);

    // KV cache is laid out as {n_kv, n_head, head_size} in memory
    // Reshape to {1, n_kv, n_head, head_size}, then transpose to {1, n_head, n_kv, head_size}
    // as required by SDPA.
    स्वतः one_c = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
    स्वतः n_head_c = ov::op::v0::Constant::create(ov::element::i64, {1}, {n_head});
    स्वतः head_size_c = ov::op::v0::Constant::create(ov::element::i64, {1}, {head_size});
    // reshape: {n_kv*n_state} -> {1, n_kv, n_head, head_size}
    स्वतः new_shape =
        std::make_shared<ov::op::v0::Concat>(ov::OutputVector{one_c, attention_size, n_head_c, head_size_c}, 0);
    स्वतः reshaped = std::make_shared<ov::op::v1::Reshape>(sliced, new_shape, असत्य);
    // transpose: {1, n_kv, n_head, head_size} -> {1, n_head, n_kv, head_size}
    स्वतः perm = ov::op::v0::Constant::create(ov::element::i64, {4}, {0, 2, 1, 3});
    स्वतः ret = std::make_shared<ov::op::v1::Transpose>(reshaped, perm);
    फर्काऊ ret;
}

OutputVector translate_flash_attn_ext(अचर गाँठो_Context & context) {
    अङ्क_inputs_check(context, 3, 4);
    अचर तार्किक has_mask = context.get_आगत_size() == 4;
    स्वतः q_f32 = context.get_आगत(0);
    स्वतः k = context.get_आगत(1);
    स्वतः v = context.get_आगत(2);
    अचर पूर्णाङ्क op_case = context.get_op_case();

    यदि (op_case == 1 || op_case == 2) {
        int64_t n_state_head = (int64_t) context.get_view_आगत_ggml_shape(1, 0)[3];
        int64_t n_head = (int64_t) context.get_view_आगत_ggml_shape(1, 0)[1];
        माप_प्रकार nb1 = context.get_view_आगत_stride(1, 0)[2];
        माप_प्रकार offset = context.get_view_आगत_offset(1, 0);
        ov::Output<ov::Node> attention_size;
        यदि (op_case == 1) {
            attention_size = context.get_आगत("attention_size");
        } अन्यथा {
            attention_size = context.get_आगत("attention_size_static");
        }
        k = reshape_flat_kv(k, offset, nb1, n_head, n_state_head, attention_size);
        v = reshape_flat_kv(v, offset, nb1, n_head, n_state_head, attention_size);
    }

    भिन्न * params = reinterpret_cast<भिन्न *>(context.get_output_op_params());
    भिन्न scale = params[0];
    // float max_bias      = params[1];
    // float logit_softcap = params[2];

    स्वतः q = std::make_shared<ov::op::v0::Convert>(q_f32, ov::element::f16);
    स्वतः scale_गाँठो = std::make_shared<ov::op::v0::Constant>(ov::element::f16, ov::Shape{}, std::vector<भिन्न>{scale});

    ov::Output<ov::Node> res;

    // For stateful
    ov::Output<ov::Node> mask;
    यदि (has_mask) {
        mask = context.get_आगत(3);
        std::string mask_name = "KQ_mask_sliced";
        यदि (context.get_आगत_names()[3].find("swa") != std::string::npos) {
            mask_name = "KQ_mask_swa_sliced";
        }
        यदि (context.has_आगत(mask_name)) {
            mask = context.get_आगत(mask_name);
        }
        यदि (mask.get_element_type() != ov::element::f16) {
            mask = std::make_shared<ov::op::v0::Convert>(mask, ov::element::f16);
        }
    }

    //auto tile_kv = [&](int64_t num_heads, int64_t num_heads_kv, int64_t head_size, ov::Output<Node> kv) {
    //    int64_t factor = num_heads / num_heads_kv;
    //    if (factor > 1 && num_heads_kv > 1) {
    स्वतः q_shape = context.get_आगत_shape(0).to_shape();
    स्वतः k_shape = context.get_आगत_shape(1).to_shape();
    अचर int64_t अङ्क_heads = q_shape[1];
    अचर int64_t अङ्क_heads_kv = k_shape[1];
    अचर int64_t head_size = q_shape[3];
    अचर int64_t factor = अङ्क_heads / अङ्क_heads_kv;

    // Manual GQA attention: enabled by default on GPU in stateless mode.
    // Set GGML_OPENVINO_MANUAL_GQA_ATTN to a positive value (e.g. 1) to force-enable,
    // or to 0 to force-disable. Unset falls back to the device-based default.
    स्थिर अचर तार्किक manual_gqa_enabled = []() {
        अचर वर्ण * env = ggml_openvino_getenv_str("GGML_OPENVINO_MANUAL_GQA_ATTN");
        यदि (env != nullptr) {
            फर्काऊ ggml_openvino_getenv_int("GGML_OPENVINO_MANUAL_GQA_ATTN") > 0;
        }
        फर्काऊ ggml_openvino_is_gpu();
    }();
    अचर तार्किक use_manual_gqa_attention =
        manual_gqa_enabled && factor > 1 && अङ्क_heads_kv > 1 && !context.is_stateful();

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
        स्वतः k_5d_shape = ov::op::v0::Constant::create(ov::element::i64, {5},
                                                       std::vector<int64_t>{0, अङ्क_heads_kv, 1, -1, head_size});
        स्वतः v_5d_shape = ov::op::v0::Constant::create(ov::element::i64, {5},
                                                       std::vector<int64_t>{0, अङ्क_heads_kv, 1, -1, head_size});
        स्वतः q_5d_shape = ov::op::v0::Constant::create(ov::element::i64, {5},
                                                       std::vector<int64_t>{0, अङ्क_heads_kv, factor, -1, head_size});

        स्वतः k_r = std::make_shared<ov::op::v1::Reshape>(k, k_5d_shape, सत्य);
        स्वतः v_r = std::make_shared<ov::op::v1::Reshape>(v, v_5d_shape, सत्य);
        स्वतः q_r = std::make_shared<ov::op::v1::Reshape>(q, q_5d_shape, सत्य);

        // QK^T → [B, num_heads_kv, factor, S_q, S_k]
        स्वतः qk = std::make_shared<ov::op::v0::MatMul>(q_r, k_r, /*tA=*/असत्य, /*tB=*/सत्य);
        स्वतः qk_scaled = std::make_shared<ov::op::v1::Multiply>(qk, scale_गाँठो);

        // Mask arrives as [B, 1, S_q, S_k]. Unsqueeze a factor axis at position 2 to
        // get [B, 1, 1, S_q, S_k], which NUMPY-broadcasts cleanly against the
        // [B, num_heads_kv, factor, S_q, S_k] scores: B==B, then 1→num_heads_kv and
        // 1→factor on the head dims.
        ov::Output<ov::Node> qk_masked;
        यदि (has_mask) {
            स्वतः mask_unsq1 =
                std::make_shared<ov::op::v0::Unsqueeze>(mask, ov::op::v0::Constant::create(ov::element::i64, {1}, {2}));
            qk_masked = std::make_shared<ov::op::v1::Add>(qk_scaled, mask_unsq1);
        } अन्यथा {
            qk_masked = qk_scaled;
        }

        स्वतः softmax = std::make_shared<ov::op::v8::Softmax>(qk_masked, /*axis=*/-1);

        // softmax @ V → [B, num_heads_kv, factor, S_q, head_size]
        स्वतः attn = std::make_shared<ov::op::v0::MatMul>(softmax, v_r);

        // Reshape back to [B, num_heads, S_q, head_size] (combine num_heads_kv * factor).
        // Leading 0 + special_zero=true copies B at runtime.
        स्वतः out_4d_shape =
            ov::op::v0::Constant::create(ov::element::i64, {4}, std::vector<int64_t>{0, अङ्क_heads, -1, head_size});
        स्वतः out_4d = std::make_shared<ov::op::v1::Reshape>(attn, out_4d_shape, सत्य);

        // The standard SDPA path's downstream is Transpose(0,2,1,3) → Convert(f32).
        // Replicate it here so callers see the same output layout/dtype.
        res = std::make_shared<ov::op::v1::Transpose>(
            out_4d, ov::op::v0::Constant::create(ov::element::i64, {4}, {0, 2, 1, 3}));
        res = std::make_shared<ov::op::v0::Convert>(res, ov::element::f32);
        फर्काऊ rename_outputs_with_suffix({res}, context.get_name());
    }

    // Default path: explicit Broadcast → SDPA. Kept as the fallback because
    // (a) it goes through the GPU plugin's micro-SDPA fast path (FlashAttention
    // tiles via DPAS), and (b) the manual path above is still being validated.
    स्वतः tile_kv = [&](int64_t n_heads, int64_t n_heads_kv, int64_t hs, ov::Output<Node> kv) {
        int64_t f = n_heads / n_heads_kv;
        यदि (f > 1 && n_heads_kv > 1) {
            ov::Output<ov::Node> kv_broadcast_shape;
            ov::Output<ov::Node> kv_unsqueezed;
            ov::Output<ov::Node> new_kv_shape;
            स्वतः unsqueeze_axes = ov::op::v0::Constant::create(ov::element::i64, Shape{}, {2});
            kv_unsqueezed = std::make_shared<ov::op::v0::Unsqueeze>(kv, unsqueeze_axes);

            kv_broadcast_shape = ov::op::v0::Constant::create(ov::element::i64, {5},
                                                              {(int64_t) 1, (int64_t) 1, f, (int64_t) 1, (int64_t) 1});
            new_kv_shape =
                ov::op::v0::Constant::create(ov::element::i64, {4}, {(int64_t) 0, n_heads, (int64_t) -1, hs});
            //    ov::element::i64, {5}, {(int64_t) 1, (int64_t) 1, factor, (int64_t) 1, (int64_t) 1});
            //new_kv_shape =
            //    ov::op::v0::Constant::create(ov::element::i64, {4}, {(int64_t) 0, num_heads, (int64_t) -1, head_size});

            kv = std::make_shared<ov::op::v3::Broadcast>(kv_unsqueezed, kv_broadcast_shape,
                                                         ov::op::BroadcastType::BIDIRECTIONAL);
            kv = std::make_shared<ov::op::v1::Reshape>(kv, new_kv_shape, सत्य);
        }
        फर्काऊ kv;
    };

    //auto q_shape = context.get_input_shape(0).to_shape();
    //auto k_shape = context.get_input_shape(1).to_shape();
    //k = tile_kv(q_shape[1], k_shape[1], q_shape[3], k);
    //v = tile_kv(q_shape[1], k_shape[1], q_shape[3], v);
    k = tile_kv(अङ्क_heads, अङ्क_heads_kv, head_size, k);
    v = tile_kv(अङ्क_heads, अङ्क_heads_kv, head_size, v);

    constexpr स्वतः causal = असत्य;
    यदि (has_mask) {
        स्वतः sdpa = std::make_shared<ov::op::v13::ScaledDotProductAttention>(q, k, v, mask, scale_गाँठो, causal);
        res = std::make_shared<ov::op::v1::Transpose>(
            sdpa, ov::op::v0::Constant::create(ov::element::i64, {4}, {0, 2, 1, 3}));
    } अन्यथा {
        स्वतः sdpa = std::make_shared<ov::op::v13::ScaledDotProductAttention>(q, k, v, scale_गाँठो, causal);
        res = std::make_shared<ov::op::v1::Transpose>(
            sdpa, ov::op::v0::Constant::create(ov::element::i64, {4}, {0, 2, 1, 3}));
    }
    res = std::make_shared<ov::op::v0::Convert>(res, ov::element::f32);
    फर्काऊ rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
