#ضمن "../node_context.h"
#ضمن "../op_table.h"
#ضمن "../utils.h"
#ضمن "ggml-openvino/ggml-openvino-extra.h"

#ضمن <cstddef>
#ضمن <cstdint>
#ضمن <memory>
#ضمن <openvino/op/add.hpp>
#ضمن <openvino/op/broadcast.hpp>
#ضمن <openvino/op/concat.hpp>
#ضمن <openvino/op/constant.hpp>
#ضمن <openvino/op/convert.hpp>
#ضمن <openvino/op/matmul.hpp>
#ضمن <openvino/op/multiply.hpp>
#ضمن <openvino/op/reshape.hpp>
#ضمن <openvino/op/scaled_dot_product_attention.hpp>
#ضمن <openvino/op/slice.hpp>
#ضمن <openvino/op/softmax.hpp>
#ضمن <openvino/op/transpose.hpp>
#ضمن <openvino/op/unsqueeze.hpp>
#ضمن <string>

نطاق ov {
نطاق frontend {
نطاق ggml {
نطاق op {
ساكن ov::Output<ov::Node> reshape_flat_kv(ثابت ov::Output<ov::Node> & kv_flat,
                                            size_t view_وٿي_بائيٽون,
                                            size_t nb1_بائيٽون,
                                            int64_t n_سر,
                                            int64_t سر_سائيز,
                                            ثابت ov::Output<ov::Node> & attention_سائيز) {
    int64_t n_state = n_سر * سر_سائيز;
    int64_t layer_شروعات_elem = (int64_t) (view_وٿي_بائيٽون / (nb1_بائيٽون / n_state));
    // Dynamic slice: [layer_start_elem, layer_start_elem + n_kv * n_state)
    خودڪار شروعات_c = ov::op::v0::Constant::create(ov::element::i64, {1}, {layer_شروعات_elem});
    خودڪار n_state_c = ov::op::v0::Constant::create(ov::element::i64, {1}, {n_state});
    // end = start + attention_size * n_state  (both static + dynamic)
    خودڪار kv_len_elems = std::make_shared<ov::op::v1::Multiply>(attention_سائيز, n_state_c);
    خودڪار پڇاڙي_c = std::make_shared<ov::op::v1::جوڙيو>(شروعات_c, kv_len_elems);
    خودڪار قدم_c = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
    خودڪار axis_c = ov::op::v0::Constant::create(ov::element::i64, {1}, {3});
    خودڪار sliced = std::make_shared<ov::op::v8::ٽڪرو>(kv_flat, شروعات_c, پڇاڙي_c, قدم_c, axis_c);

    // KV cache is laid out as {n_kv, n_head, head_size} in memory
    // Reshape to {1, n_kv, n_head, head_size}, then transpose to {1, n_head, n_kv, head_size}
    // as required by SDPA.
    خودڪار one_c = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
    خودڪار n_سر_c = ov::op::v0::Constant::create(ov::element::i64, {1}, {n_سر});
    خودڪار سر_سائيز_c = ov::op::v0::Constant::create(ov::element::i64, {1}, {سر_سائيز});
    // reshape: {n_kv*n_state} -> {1, n_kv, n_head, head_size}
    خودڪار new_shape =
        std::make_shared<ov::op::v0::Concat>(ov::Output_ويڪٽر{one_c, attention_سائيز, n_سر_c, سر_سائيز_c}, 0);
    خودڪار reshaped = std::make_shared<ov::op::v1::Reshape>(sliced, new_shape, خطأ);
    // transpose: {1, n_kv, n_head, head_size} -> {1, n_head, n_kv, head_size}
    خودڪار perm = ov::op::v0::Constant::create(ov::element::i64, {4}, {0, 2, 1, 3});
    خودڪار ret = std::make_shared<ov::op::v1::Transpose>(reshaped, perm);
    أرجع ret;
}

Output_ويڪٽر translate_flash_attn_ext(ثابت ڳنڍ_Context & context) {
    انگ_inputs_check(context, 3, 4);
    ثابت منطقي has_نقاب = context.get_داخلا_سائيز() == 4;
    خودڪار q_f32 = context.get_داخلا(0);
    خودڪار k = context.get_داخلا(1);
    خودڪار v = context.get_داخلا(2);
    ثابت صحيح op_case = context.get_op_case();

    إذا (op_case == 1 || op_case == 2) {
        int64_t n_state_سر = (int64_t) context.get_view_داخلا_ggml_shape(1, 0)[3];
        int64_t n_سر = (int64_t) context.get_view_داخلا_ggml_shape(1, 0)[1];
        size_t nb1 = context.get_view_داخلا_stride(1, 0)[2];
        size_t وٿي = context.get_view_داخلا_وٿي(1, 0);
        ov::Output<ov::Node> attention_سائيز;
        إذا (op_case == 1) {
            attention_سائيز = context.get_داخلا("attention_size");
        } وإلا {
            attention_سائيز = context.get_داخلا("attention_size_static");
        }
        k = reshape_flat_kv(k, وٿي, nb1, n_سر, n_state_سر, attention_سائيز);
        v = reshape_flat_kv(v, وٿي, nb1, n_سر, n_state_سر, attention_سائيز);
    }

    عائم * params = reinterpret_cast<عائم *>(context.get_output_op_params());
    عائم scale = params[0];
    // float max_bias      = params[1];
    // float logit_softcap = params[2];

    خودڪار q = std::make_shared<ov::op::v0::Convert>(q_f32, ov::element::f16);
    خودڪار scale_ڳنڍ = std::make_shared<ov::op::v0::Constant>(ov::element::f16, ov::Shape{}, std::vector<عائم>{scale});

    ov::Output<ov::Node> res;

    // For stateful
    ov::Output<ov::Node> نقاب;
    إذا (has_نقاب) {
        نقاب = context.get_داخلا(3);
        std::string نقاب_name = "KQ_mask_sliced";
        إذا (context.get_داخلا_نالا()[3].find("swa") != std::string::npos) {
            نقاب_name = "KQ_mask_swa_sliced";
        }
        إذا (context.has_داخلا(نقاب_name)) {
            نقاب = context.get_داخلا(نقاب_name);
        }
        إذا (نقاب.get_element_type() != ov::element::f16) {
            نقاب = std::make_shared<ov::op::v0::Convert>(نقاب, ov::element::f16);
        }
    }

    //auto tile_kv = [&](int64_t num_heads, int64_t num_heads_kv, int64_t head_size, ov::Output<Node> kv) {
    //    int64_t factor = num_heads / num_heads_kv;
    //    if (factor > 1 && num_heads_kv > 1) {
    خودڪار q_shape = context.get_داخلا_shape(0).منزل_shape();
    خودڪار k_shape = context.get_داخلا_shape(1).منزل_shape();
    ثابت int64_t انگ_heads = q_shape[1];
    ثابت int64_t انگ_heads_kv = k_shape[1];
    ثابت int64_t سر_سائيز = q_shape[3];
    ثابت int64_t ضرب_ڪندڙ = انگ_heads / انگ_heads_kv;

    // Manual GQA attention: enabled by default on GPU in stateless mode.
    // Set GGML_OPENVINO_MANUAL_GQA_ATTN to a positive value (e.g. 1) to force-enable,
    // or to 0 to force-disable. Unset falls back to the device-based default.
    ساكن ثابت منطقي manual_gqa_enabled = []() {
        ثابت حرف * env = ggml_openvino_getenv_str("GGML_OPENVINO_MANUAL_GQA_ATTN");
        إذا (env != خالي_اشارو) {
            أرجع ggml_openvino_getenv_int("GGML_OPENVINO_MANUAL_GQA_ATTN") > 0;
        }
        أرجع ggml_openvino_is_gpu();
    }();
    ثابت منطقي use_manual_gqa_attention =
        manual_gqa_enabled && ضرب_ڪندڙ > 1 && انگ_heads_kv > 1 && !context.is_stateful();

    إذا (use_manual_gqa_attention) {
        // Q, K, V arrive as [B, n_heads(_kv), S, head_size], where B is the active
        // batch (n_seq_active) and may be > 1 (llama-perplexity, llama-server -np > 1)
        // or dynamic. Reshape to
        //   K_r: [B, num_heads_kv, 1, S, head_size]
        //   Q_r: [B, num_heads_kv, factor, S_q, head_size]
        // and let MatMul broadcast across the factor dim without materialising
        // an expanded K/V. The leading 0 + special_zero=true copies B at runtime,
        // so this is correct for B == 1, B > 1, and dynamic B alike. Only the head
        // dims and head_size are baked in as literals; the sequence dim stays -1.
        خودڪار k_5d_shape = ov::op::v0::Constant::create(ov::element::i64, {5},
                                                       std::vector<int64_t>{0, انگ_heads_kv, 1, -1, سر_سائيز});
        خودڪار v_5d_shape = ov::op::v0::Constant::create(ov::element::i64, {5},
                                                       std::vector<int64_t>{0, انگ_heads_kv, 1, -1, سر_سائيز});
        خودڪار q_5d_shape = ov::op::v0::Constant::create(ov::element::i64, {5},
                                                       std::vector<int64_t>{0, انگ_heads_kv, ضرب_ڪندڙ, -1, سر_سائيز});

        خودڪار k_r = std::make_shared<ov::op::v1::Reshape>(k, k_5d_shape, سچ);
        خودڪار v_r = std::make_shared<ov::op::v1::Reshape>(v, v_5d_shape, سچ);
        خودڪار q_r = std::make_shared<ov::op::v1::Reshape>(q, q_5d_shape, سچ);

        // QK^T → [B, num_heads_kv, factor, S_q, S_k]
        خودڪار qk = std::make_shared<ov::op::v0::MatMul>(q_r, k_r, /*tA=*/خطأ, /*tB=*/سچ);
        خودڪار qk_scaled = std::make_shared<ov::op::v1::Multiply>(qk, scale_ڳنڍ);

        // Mask arrives as [B, 1, S_q, S_k]. Unsqueeze a factor axis at position 2 to
        // get [B, 1, 1, S_q, S_k], which NUMPY-broadcasts cleanly against the
        // [B, num_heads_kv, factor, S_q, S_k] scores: B==B, then 1→num_heads_kv and
        // 1→factor on the head dims.
        ov::Output<ov::Node> qk_masked;
        إذا (has_نقاب) {
            خودڪار نقاب_unsq1 =
                std::make_shared<ov::op::v0::Unsqueeze>(نقاب, ov::op::v0::Constant::create(ov::element::i64, {1}, {2}));
            qk_masked = std::make_shared<ov::op::v1::جوڙيو>(qk_scaled, نقاب_unsq1);
        } وإلا {
            qk_masked = qk_scaled;
        }

        خودڪار softmax = std::make_shared<ov::op::v8::Softmax>(qk_masked, /*axis=*/-1);

        // softmax @ V → [B, num_heads_kv, factor, S_q, head_size]
        خودڪار attn = std::make_shared<ov::op::v0::MatMul>(softmax, v_r);

        // Reshape back to [B, num_heads, S_q, head_size] (combine num_heads_kv * factor).
        // Leading 0 + special_zero=true copies B at runtime.
        خودڪار ٻاهريون_4d_shape =
            ov::op::v0::Constant::create(ov::element::i64, {4}, std::vector<int64_t>{0, انگ_heads, -1, سر_سائيز});
        خودڪار ٻاهريون_4d = std::make_shared<ov::op::v1::Reshape>(attn, ٻاهريون_4d_shape, سچ);

        // The standard SDPA path's downstream is Transpose(0,2,1,3) → Convert(f32).
        // Replicate it here so callers see the same output layout/dtype.
        res = std::make_shared<ov::op::v1::Transpose>(
            ٻاهريون_4d, ov::op::v0::Constant::create(ov::element::i64, {4}, {0, 2, 1, 3}));
        res = std::make_shared<ov::op::v0::Convert>(res, ov::element::f32);
        أرجع rename_outputs_with_suffix({res}, context.get_name());
    }

    // Default path: explicit Broadcast → SDPA. Kept as the fallback because
    // (a) it goes through the GPU plugin's micro-SDPA fast path (FlashAttention
    // tiles via DPAS), and (b) the manual path above is still being validated.
    خودڪار tile_kv = [&](int64_t n_heads, int64_t n_heads_kv, int64_t hs, ov::Output<Node> kv) {
        int64_t f = n_heads / n_heads_kv;
        إذا (f > 1 && n_heads_kv > 1) {
            ov::Output<ov::Node> kv_نشر_shape;
            ov::Output<ov::Node> kv_unsqueezed;
            ov::Output<ov::Node> new_kv_shape;
            خودڪار unsqueeze_axes = ov::op::v0::Constant::create(ov::element::i64, Shape{}, {2});
            kv_unsqueezed = std::make_shared<ov::op::v0::Unsqueeze>(kv, unsqueeze_axes);

            kv_نشر_shape = ov::op::v0::Constant::create(ov::element::i64, {5},
                                                              {(int64_t) 1, (int64_t) 1, f, (int64_t) 1, (int64_t) 1});
            new_kv_shape =
                ov::op::v0::Constant::create(ov::element::i64, {4}, {(int64_t) 0, n_heads, (int64_t) -1, hs});
            //    ov::element::i64, {5}, {(int64_t) 1, (int64_t) 1, factor, (int64_t) 1, (int64_t) 1});
            //new_kv_shape =
            //    ov::op::v0::Constant::create(ov::element::i64, {4}, {(int64_t) 0, num_heads, (int64_t) -1, head_size});

            kv = std::make_shared<ov::op::v3::نشر>(kv_unsqueezed, kv_نشر_shape,
                                                         ov::op::نشر_Type::BIDIRECTIONAL);
            kv = std::make_shared<ov::op::v1::Reshape>(kv, new_kv_shape, سچ);
        }
        أرجع kv;
    };

    //auto q_shape = context.get_input_shape(0).to_shape();
    //auto k_shape = context.get_input_shape(1).to_shape();
    //k = tile_kv(q_shape[1], k_shape[1], q_shape[3], k);
    //v = tile_kv(q_shape[1], k_shape[1], q_shape[3], v);
    k = tile_kv(انگ_heads, انگ_heads_kv, سر_سائيز, k);
    v = tile_kv(انگ_heads, انگ_heads_kv, سر_سائيز, v);

    constexpr خودڪار causal = خطأ;
    إذا (has_نقاب) {
        خودڪار sdpa = std::make_shared<ov::op::v13::Scaled_Dot_حاصل_ضرب_Attention>(q, k, v, نقاب, scale_ڳنڍ, causal);
        res = std::make_shared<ov::op::v1::Transpose>(
            sdpa, ov::op::v0::Constant::create(ov::element::i64, {4}, {0, 2, 1, 3}));
    } وإلا {
        خودڪار sdpa = std::make_shared<ov::op::v13::Scaled_Dot_حاصل_ضرب_Attention>(q, k, v, scale_ڳنڍ, causal);
        res = std::make_shared<ov::op::v1::Transpose>(
            sdpa, ov::op::v0::Constant::create(ov::element::i64, {4}, {0, 2, 1, 3}));
    }
    res = std::make_shared<ov::op::v0::Convert>(res, ov::element::f32);
    أرجع rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
