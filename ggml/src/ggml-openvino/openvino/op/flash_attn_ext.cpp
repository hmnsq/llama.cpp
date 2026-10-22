#समावेश "../node_context.h"
#समावेश "../op_table.h"
#समावेश "../utils.h"
#समावेश "ggml-openvino/ggml-openvino-extra.h"

#समावेश <cstddef>
#समावेश <cstdint>
#समावेश <memory>
#समावेश <openvino/op/add.hpp>
#समावेश <openvino/op/broadcast.hpp>
#समावेश <openvino/op/concat.hpp>
#समावेश <openvino/op/constant.hpp>
#समावेश <openvino/op/convert.hpp>
#समावेश <openvino/op/matmul.hpp>
#समावेश <openvino/op/multiply.hpp>
#समावेश <openvino/op/reshape.hpp>
#समावेश <openvino/op/scaled_dot_product_attention.hpp>
#समावेश <openvino/op/slice.hpp>
#समावेश <openvino/op/softmax.hpp>
#समावेश <openvino/op/transpose.hpp>
#समावेश <openvino/op/unsqueeze.hpp>
#समावेश <string>

namespace ov {
namespace frontend {
namespace ggml {
namespace op {
अटल ov::Output<ov::Node> reshape_flat_kv(स्थिर ov::Output<ov::Node> & kv_flat,
                                            माप_प्रकार view_offset_bytes,
                                            माप_प्रकार nb1_bytes,
                                            int64_t n_शीर्ष,
                                            int64_t शीर्ष_आकार,
                                            स्थिर ov::Output<ov::Node> & attention_आकार) {
    int64_t n_state = n_शीर्ष * शीर्ष_आकार;
    int64_t layer_start_elem = (int64_t) (view_offset_bytes / (nb1_bytes / n_state));
    // Dynamic slice: [layer_start_elem, layer_start_elem + n_kv * n_state)
    स्वतः start_c = ov::op::v0::Constant::create(ov::element::i64, {1}, {layer_start_elem});
    स्वतः n_state_c = ov::op::v0::Constant::create(ov::element::i64, {1}, {n_state});
    // end = start + attention_size * n_state  (both static + dynamic)
    स्वतः kv_len_elems = std::make_shared<ov::op::v1::Multiply>(attention_आकार, n_state_c);
    स्वतः end_c = std::make_shared<ov::op::v1::Add>(start_c, kv_len_elems);
    स्वतः step_c = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
    स्वतः axis_c = ov::op::v0::Constant::create(ov::element::i64, {1}, {3});
    स्वतः sliced = std::make_shared<ov::op::v8::Slice>(kv_flat, start_c, end_c, step_c, axis_c);

    // KV cache is laid out as {n_kv, n_head, head_size} in memory
    // Reshape to {1, n_kv, n_head, head_size}, then transpose to {1, n_head, n_kv, head_size}
    // as required by SDPA.
    स्वतः one_c = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
    स्वतः n_शीर्ष_c = ov::op::v0::Constant::create(ov::element::i64, {1}, {n_शीर्ष});
    स्वतः शीर्ष_आकार_c = ov::op::v0::Constant::create(ov::element::i64, {1}, {शीर्ष_आकार});
    // reshape: {n_kv*n_state} -> {1, n_kv, n_head, head_size}
    स्वतः new_shape =
        std::make_shared<ov::op::v0::Concat>(ov::OutputVector{one_c, attention_आकार, n_शीर्ष_c, शीर्ष_आकार_c}, 0);
    स्वतः reshaped = std::make_shared<ov::op::v1::Reshape>(sliced, new_shape, false);
    // transpose: {1, n_kv, n_head, head_size} -> {1, n_head, n_kv, head_size}
    स्वतः perm = ov::op::v0::Constant::create(ov::element::i64, {4}, {0, 2, 1, 3});
    स्वतः ret = std::make_shared<ov::op::v1::Transpose>(reshaped, perm);
    वापस ret;
}

OutputVector translate_flash_attn_ext(स्थिर गाँठ_Context & context) {
    अंक_inputs_check(context, 3, 4);
    स्थिर bool has_mask = context.get_निवेश_आकार() == 4;
    स्वतः q_f32 = context.get_निवेश(0);
    स्वतः k = context.get_निवेश(1);
    स्वतः v = context.get_निवेश(2);
    स्थिर पूर्णांक op_case = context.get_op_case();

    अगर (op_case == 1 || op_case == 2) {
        int64_t n_state_शीर्ष = (int64_t) context.get_view_निवेश_ggml_shape(1, 0)[3];
        int64_t n_शीर्ष = (int64_t) context.get_view_निवेश_ggml_shape(1, 0)[1];
        माप_प्रकार nb1 = context.get_view_निवेश_stride(1, 0)[2];
        माप_प्रकार offset = context.get_view_निवेश_offset(1, 0);
        ov::Output<ov::Node> attention_आकार;
        अगर (op_case == 1) {
            attention_आकार = context.get_निवेश("attention_size");
        } अन्यथा {
            attention_आकार = context.get_निवेश("attention_size_static");
        }
        k = reshape_flat_kv(k, offset, nb1, n_शीर्ष, n_state_शीर्ष, attention_आकार);
        v = reshape_flat_kv(v, offset, nb1, n_शीर्ष, n_state_शीर्ष, attention_आकार);
    }

    भग्न * params = reinterpret_cast<भग्न *>(context.get_output_op_params());
    भग्न scale = params[0];
    // float max_bias      = params[1];
    // float logit_softcap = params[2];

    स्वतः q = std::make_shared<ov::op::v0::Convert>(q_f32, ov::element::f16);
    स्वतः scale_गाँठ = std::make_shared<ov::op::v0::Constant>(ov::element::f16, ov::Shape{}, std::vector<भग्न>{scale});

    ov::Output<ov::Node> res;

    // For stateful
    ov::Output<ov::Node> mask;
    अगर (has_mask) {
        mask = context.get_निवेश(3);
        std::string mask_name = "KQ_mask_sliced";
        अगर (context.get_निवेश_names()[3].find("swa") != std::string::npos) {
            mask_name = "KQ_mask_swa_sliced";
        }
        अगर (context.has_निवेश(mask_name)) {
            mask = context.get_निवेश(mask_name);
        }
        अगर (mask.get_element_type() != ov::element::f16) {
            mask = std::make_shared<ov::op::v0::Convert>(mask, ov::element::f16);
        }
    }

    //auto tile_kv = [&](int64_t num_heads, int64_t num_heads_kv, int64_t head_size, ov::Output<Node> kv) {
    //    int64_t factor = num_heads / num_heads_kv;
    //    if (factor > 1 && num_heads_kv > 1) {
    स्वतः q_shape = context.get_निवेश_shape(0).to_shape();
    स्वतः k_shape = context.get_निवेश_shape(1).to_shape();
    स्थिर int64_t अंक_heads = q_shape[1];
    स्थिर int64_t अंक_heads_kv = k_shape[1];
    स्थिर int64_t शीर्ष_आकार = q_shape[3];
    स्थिर int64_t factor = अंक_heads / अंक_heads_kv;

    // Manual GQA attention: enabled by default on GPU in stateless mode.
    // Set GGML_OPENVINO_MANUAL_GQA_ATTN to a positive value (e.g. 1) to force-enable,
    // or to 0 to force-disable. Unset falls back to the device-based default.
    अटल स्थिर bool manual_gqa_enabled = []() {
        स्थिर अक्षर * env = ggml_openvino_getenv_str("GGML_OPENVINO_MANUAL_GQA_ATTN");
        अगर (env != nullptr) {
            वापस ggml_openvino_getenv_int("GGML_OPENVINO_MANUAL_GQA_ATTN") > 0;
        }
        वापस ggml_openvino_is_gpu();
    }();
    स्थिर bool use_manual_gqa_attention =
        manual_gqa_enabled && factor > 1 && अंक_heads_kv > 1 && !context.is_stateful();

    अगर (use_manual_gqa_attention) {
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
                                                       std::vector<int64_t>{0, अंक_heads_kv, 1, -1, शीर्ष_आकार});
        स्वतः v_5d_shape = ov::op::v0::Constant::create(ov::element::i64, {5},
                                                       std::vector<int64_t>{0, अंक_heads_kv, 1, -1, शीर्ष_आकार});
        स्वतः q_5d_shape = ov::op::v0::Constant::create(ov::element::i64, {5},
                                                       std::vector<int64_t>{0, अंक_heads_kv, factor, -1, शीर्ष_आकार});

        स्वतः k_r = std::make_shared<ov::op::v1::Reshape>(k, k_5d_shape, true);
        स्वतः v_r = std::make_shared<ov::op::v1::Reshape>(v, v_5d_shape, true);
        स्वतः q_r = std::make_shared<ov::op::v1::Reshape>(q, q_5d_shape, true);

        // QK^T → [B, num_heads_kv, factor, S_q, S_k]
        स्वतः qk = std::make_shared<ov::op::v0::MatMul>(q_r, k_r, /*tA=*/false, /*tB=*/true);
        स्वतः qk_scaled = std::make_shared<ov::op::v1::Multiply>(qk, scale_गाँठ);

        // Mask arrives as [B, 1, S_q, S_k]. Unsqueeze a factor axis at position 2 to
        // get [B, 1, 1, S_q, S_k], which NUMPY-broadcasts cleanly against the
        // [B, num_heads_kv, factor, S_q, S_k] scores: B==B, then 1→num_heads_kv and
        // 1→factor on the head dims.
        ov::Output<ov::Node> qk_masked;
        अगर (has_mask) {
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
            ov::op::v0::Constant::create(ov::element::i64, {4}, std::vector<int64_t>{0, अंक_heads, -1, शीर्ष_आकार});
        स्वतः out_4d = std::make_shared<ov::op::v1::Reshape>(attn, out_4d_shape, true);

        // The standard SDPA path's downstream is Transpose(0,2,1,3) → Convert(f32).
        // Replicate it here so callers see the same output layout/dtype.
        res = std::make_shared<ov::op::v1::Transpose>(
            out_4d, ov::op::v0::Constant::create(ov::element::i64, {4}, {0, 2, 1, 3}));
        res = std::make_shared<ov::op::v0::Convert>(res, ov::element::f32);
        वापस rename_outputs_with_suffix({res}, context.get_name());
    }

    // Default path: explicit Broadcast → SDPA. Kept as the fallback because
    // (a) it goes through the GPU plugin's micro-SDPA fast path (FlashAttention
    // tiles via DPAS), and (b) the manual path above is still being validated.
    स्वतः tile_kv = [&](int64_t n_heads, int64_t n_heads_kv, int64_t hs, ov::Output<Node> kv) {
        int64_t f = n_heads / n_heads_kv;
        अगर (f > 1 && n_heads_kv > 1) {
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
            kv = std::make_shared<ov::op::v1::Reshape>(kv, new_kv_shape, true);
        }
        वापस kv;
    };

    //auto q_shape = context.get_input_shape(0).to_shape();
    //auto k_shape = context.get_input_shape(1).to_shape();
    //k = tile_kv(q_shape[1], k_shape[1], q_shape[3], k);
    //v = tile_kv(q_shape[1], k_shape[1], q_shape[3], v);
    k = tile_kv(अंक_heads, अंक_heads_kv, शीर्ष_आकार, k);
    v = tile_kv(अंक_heads, अंक_heads_kv, शीर्ष_आकार, v);

    constexpr स्वतः causal = false;
    अगर (has_mask) {
        स्वतः sdpa = std::make_shared<ov::op::v13::ScaledDotProductAttention>(q, k, v, mask, scale_गाँठ, causal);
        res = std::make_shared<ov::op::v1::Transpose>(
            sdpa, ov::op::v0::Constant::create(ov::element::i64, {4}, {0, 2, 1, 3}));
    } अन्यथा {
        स्वतः sdpa = std::make_shared<ov::op::v13::ScaledDotProductAttention>(q, k, v, scale_गाँठ, causal);
        res = std::make_shared<ov::op::v1::Transpose>(
            sdpa, ov::op::v0::Constant::create(ov::element::i64, {4}, {0, 2, 1, 3}));
    }
    res = std::make_shared<ov::op::v0::Convert>(res, ov::element::f32);
    वापस rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
