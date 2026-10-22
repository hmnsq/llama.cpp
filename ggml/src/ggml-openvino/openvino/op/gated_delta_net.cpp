#அந்தர்பாவ "gated_delta_net.hpp"

#அந்தர்பாவ "../node_context.h"
#அந்தர்பாவ "../op_table.h"
#அந்தர்பாவ "../utils.h"
#அந்தர்பாவ "ggml-openvino/ggml-openvino-extra.h"

#அந்தர்பாவ <cmath>
#அந்தர்பாவ <cstdint>
#அந்தர்பாவ <memory>
#அந்தர்பாவ <numeric>
#அந்தர்பாவ <openvino/op/add.hpp>
#அந்தர்பாவ <openvino/op/broadcast.hpp>
#அந்தர்பாவ <openvino/op/concat.hpp>
#அந்தர்பாவ <openvino/op/constant.hpp>
#அந்தர்பாவ <openvino/op/convert.hpp>
#அந்தர்பாவ <openvino/op/divide.hpp>
#அந்தர்பாவ <openvino/op/exp.hpp>
#அந்தர்பாவ <openvino/op/gather.hpp>
#அந்தர்பாவ <openvino/op/less.hpp>
#அந்தர்பாவ <openvino/op/loop.hpp>
#அந்தர்பாவ <openvino/op/matmul.hpp>
#அந்தர்பாவ <openvino/op/multiply.hpp>
#அந்தர்பாவ <openvino/op/reduce_mean.hpp>
#அந்தர்பாவ <openvino/op/reshape.hpp>
#அந்தர்பாவ <openvino/op/squeeze.hpp>
#அந்தர்பாவ <openvino/op/sqrt.hpp>
#அந்தர்பாவ <openvino/op/subtract.hpp>
#அந்தர்பாவ <openvino/op/tile.hpp>
#அந்தர்பாவ <openvino/op/transpose.hpp>
#அந்தர்பாவ <openvino/op/unsqueeze.hpp>
#அந்தர்பாவ <vector>

நாமாகாஶ ov {
நாமாகாஶ frontend {
நாமாகாஶ ggml {
நாமாகாஶ op {

நிலையான OutputVector translate_gated_delta_net_ref(மாறிலி முனை_Context & context);

நிலையான தருக்க match_gdn_l2_norm(மாறிலி Output<Node> & normalized, Output<Node> & உள்ளீடு, தசம & eps) {
    // Match the RMSNorm decomposition emitted by translate_rms_norm, followed by GGML SCALE.
    மாறிலி auto scale = ov::as_type_ptr<ov::op::v1::Multiply>(normalized.get_முனை_shared_ptr());
    எனில் (!scale) {
        திரும்பு பொய்;
    }
    மாறிலி auto factor = ov::as_type_ptr<ov::op::v0::Constant>(scale->get_உள்ளீடு_முனை_shared_ptr(1));
    மாறிலி auto rms = ov::as_type_ptr<ov::op::v1::Multiply>(scale->get_உள்ளீடு_முனை_shared_ptr(0));
    எனில் (!factor || ov::shape_size(factor->get_shape()) != 1 || !rms) {
        திரும்பு பொய்;
    }
    மாறிலி auto x = rms->உள்ளீடு_மதிப்பு(0);
    மாறிலி auto & shape = x.get_partial_shape();
    எனில் (x.get_element_type() != ov::element::f32 || shape.rank() != 4 || shape[3].is_dynamic() ||
        shape[3].get_length() <= 0 || normalized.get_partial_shape() != shape) {
        திரும்பு பொய்;
    }
    மாறிலி தசம dim = static_cast<தசம>(shape[3].get_length());
    எனில் (factor->cast_vector<தசம>()[0] != 1.0f / std::sqrt(dim)) {
        திரும்பு பொய்;
    }
    மாறிலி auto reciprocal = ov::as_type_ptr<ov::op::v1::Divide>(rms->get_உள்ளீடு_முனை_shared_ptr(1));
    எனில் (!reciprocal) {
        திரும்பு பொய்;
    }
    மாறிலி auto one = ov::as_type_ptr<ov::op::v0::Constant>(reciprocal->get_உள்ளீடு_முனை_shared_ptr(0));
    மாறிலி auto வேர் = ov::as_type_ptr<ov::op::v0::Sqrt>(reciprocal->get_உள்ளீடு_முனை_shared_ptr(1));
    எனில் (!one || ov::shape_size(one->get_shape()) != 1 || one->cast_vector<தசம>()[0] != 1.0f || !வேர்) {
        திரும்பு பொய்;
    }
    மாறிலி auto add = ov::as_type_ptr<ov::op::v1::Add>(வேர்->get_உள்ளீடு_முனை_shared_ptr(0));
    எனில் (!add) {
        திரும்பு பொய்;
    }
    மாறிலி auto mean = ov::as_type_ptr<ov::op::v1::ReduceMean>(add->get_உள்ளீடு_முனை_shared_ptr(0));
    மாறிலி auto rms_eps = ov::as_type_ptr<ov::op::v0::Constant>(add->get_உள்ளீடு_முனை_shared_ptr(1));
    எனில் (!mean || !mean->get_keep_dims() || !rms_eps || ov::shape_size(rms_eps->get_shape()) != 1) {
        திரும்பு பொய்;
    }
    மாறிலி auto axes = ov::as_type_ptr<ov::op::v0::Constant>(mean->get_உள்ளீடு_முனை_shared_ptr(1));
    மாறிலி auto square = ov::as_type_ptr<ov::op::v1::Multiply>(mean->get_உள்ளீடு_முனை_shared_ptr(0));
    எனில் (!axes || axes->cast_vector<int64_t>() != std::vector<int64_t>{-1} || !square ||
        square->உள்ளீடு_மதிப்பு(0) != x || square->உள்ளீடு_மதிப்பு(1) != x) {
        திரும்பு பொய்;
    }
    // RMSNorm(x, rms_eps) / sqrt(D) = x / sqrt(sum(x*x) + D*rms_eps).
    eps = dim * rms_eps->cast_vector<தசம>()[0];
    எனில் (!std::isfinite(eps) || eps <= 0.0f) {
        திரும்பு பொய்;
    }
    உள்ளீடு = x;
    திரும்பு உண்மை;
}

OutputVector translate_gated_delta_net(மாறிலி முனை_Context & context) {
    auto v_shape = context.get_உள்ளீடு_shape(2).to_shape();  // [B, T, H_v, S_v]
    auto q_shape = context.get_உள்ளீடு_shape(0).to_shape();  // [B, T, H_k, S_k]

    // Fused GatedDeltaNet op only supports scalar gate (kda=0).
    // Fall back to reference implementation for per-key-dimension gating.
    // if (kda) {
    //     return translate_gated_delta_net_ref(context);
    // }

    // const int64_t B = v_shape[0];
    // const int64_t T = v_shape[1];
    மாறிலி int64_t H_v = v_shape[2];
    மாறிலி int64_t S_v = v_shape[3];
    மாறிலி int64_t H_k = q_shape[2];
    // const int64_t S_k = q_shape[3];

    auto q = context.get_உள்ளீடு(0);
    auto k = context.get_உள்ளீடு(1);
    auto v = process_view_உள்ளீடு(context, 2, H_v * S_v, 3);
    auto g = context.get_உள்ளீடு(3);
    auto beta = context.get_உள்ளீடு(4);
    auto state = context.get_உள்ளீடு(5);

    Output<Node> raw_q, raw_k;
    தசம q_eps = 1e-6f, k_eps = 1e-6f;
    மாறிலி தருக்க fuse_qk_l2norm = ggml_openvino_is_gpu() &&
        match_gdn_l2_norm(q, raw_q, q_eps) && match_gdn_l2_norm(k, raw_k, k_eps);
    எனில் (fuse_qk_l2norm) {
        // Keep head tiling below; GDN applies normalization per head and keeps its attention scale.
        q = raw_q;
        k = raw_k;
    }

    // ggml maps GQA heads in tiled order, while OV GDN maps repeated heads in grouped order.
    எனில் (H_v != H_k) {
        மாறிலி int64_t repeat = H_v / H_k;
        auto repeats = ov::op::v0::Constant::create(ov::element::i64, {4}, std::vector<int64_t>{1, 1, repeat, 1});
        q = std::make_shared<ov::op::v0::Tile>(q, repeats);
        k = std::make_shared<ov::op::v0::Tile>(k, repeats);
    }

    எனில் (context.get_view_உள்ளீடு_size(2)) {
        // Same as l2_norm case 1
        v = std::make_shared<ov::op::v0::Squeeze>(v, ov::op::v0::Constant::create(ov::element::i64, {1}, {0}));
        auto v_shape = context.get_உள்ளீடு_shape(2).to_shape();
        std::vector<int64_t> reshape_pattern = {0, 0, (int64_t) v_shape[2], (int64_t) v_shape[3]};
        v = std::make_shared<ov::op::v1::Reshape>(
            v, ov::op::v0::Constant::create(ov::element::i64, {4}, reshape_pattern), உண்மை);
    }

    // ggml state layout (OV notation): [B, H_v, value_dim, key_dim]
    // GatedDeltaNet op expects: [B, H_v, key_dim, value_dim]
    auto state_perm = ov::op::v0::Constant::create(ov::element::i64, {4}, std::vector<int64_t>{0, 1, 3, 2});
    state = std::make_shared<ov::op::v1::Transpose>(state, state_perm);

    g = std::make_shared<ov::op::v0::Squeeze>(g, ov::op::v0::Constant::create(ov::element::i64, {1}, {3}));
    beta = std::make_shared<ov::op::v0::Squeeze>(beta, ov::op::v0::Constant::create(ov::element::i64, {1}, {3}));

    எனில் (context.has_உள்ளீடு("chunk_valid_len")) {
        // The last prefill chunk is padded with fabricated tokens. The recurrence is
        //   S_t = S_{t-1} * exp(g_t) + k_t (x) ((v_t - S_{t-1}^T k_t) * beta_t)
        // so forcing g = 0 and beta = 0 makes a padded step an exact identity and keeps the final
        // state equal to the state after the last real token. Attention output at those positions
        // is garbage but never read.
        மாறிலி auto & g_ps = g.get_partial_shape();
        FRONT_END_OP_CONVERSION_CHECK(g_ps.rank().is_static() && g_ps.rank().get_length() == 3 && g_ps[1].is_static(),
                                      "GATED_DELTA_NET pad masking requires a static token dimension");
        மாறிலி int64_t n_tokens = g_ps[1].get_length();
        std::vector<int64_t> positions(n_tokens);
        std::iota(positions.begin(), positions.end(), 0);
        auto valid = std::make_shared<ov::op::v1::Less>(
            ov::op::v0::Constant::create(ov::element::i64, {(size_t) n_tokens}, positions),
            context.get_உள்ளீடு("chunk_valid_len"));
        auto mask = std::make_shared<ov::op::v0::Unsqueeze>(
            std::make_shared<ov::op::v0::Convert>(valid, g.get_element_type()),
            ov::op::v0::Constant::create(ov::element::i64, {2}, std::vector<int64_t>{0, 2}));
        g = std::make_shared<ov::op::v1::Multiply>(g, mask);
        beta = std::make_shared<ov::op::v1::Multiply>(beta, mask);
    }

    // std::cout << "GatedDeltaNet input shapes: q=" << q.get_partial_shape() << ", k=" << k.get_partial_shape()
    //           << ", v=" << v.get_partial_shape() << ", g=" << g.get_partial_shape()
    //           << ", beta=" << beta.get_partial_shape() << ", state=" << state.get_partial_shape() << std::endl;

    auto gdn = std::make_shared<ov::op::internal::GatedDeltaNet>(q, k, v, state, g, beta, fuse_qk_l2norm, q_eps, k_eps);
    auto attn_4d = gdn->output(0);
    auto state_4d = gdn->output(1);  // [B, H_v, key_dim, value_dim]

    // std::cout << "GatedDeltaNet output shapes: attn=" << gdn->output(0).get_partial_shape()
    //           << ", new_state=" << gdn->output(1).get_partial_shape() << std::endl;

    // Transpose output state back to ggml layout [B, H_v, value_dim, key_dim]
    auto state_transposed = std::make_shared<ov::op::v1::Transpose>(state_4d, state_perm);
    எனில் (context.get_output_names().size() == 2) {
        // The canonical graph consumes the packed GGML result only through separate attention
        // and state VIEWs. Publish the native outputs under those VIEW names to avoid
        // flatten -> concat -> reshape -> slice -> reshape chains. This also works for B > 1.
        திரும்பு rename_outputs_with_suffix({attn_4d, state_transposed}, context.get_name());
    }

    auto flat_shape_1d = ov::op::v0::Constant::create(ov::element::i64, {1}, {-1});
    auto attn = std::make_shared<ov::op::v1::Reshape>(attn_4d, flat_shape_1d, பொய்);
    auto new_state = std::make_shared<ov::op::v1::Reshape>(state_transposed, flat_shape_1d, பொய்);
    auto packed = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{attn, new_state}, 0);
    auto out_shape = ov::op::v0::Constant::create(ov::element::i64, {4},
                                                  std::vector<int64_t>{1, 1, -1 /*T * B + S_v * B*/, S_v * H_v});
    auto res = std::make_shared<ov::op::v1::Reshape>(packed, out_shape, பொய்);

    திரும்பு rename_outputs_with_suffix({res}, context.get_name());
}

நிலையான OutputVector translate_gated_delta_net_ref(மாறிலி முனை_Context & context) {
    இலக்கம்_inputs_check(context, 6, 6);

    // Inputs (OV shapes are reversed from ggml):
    // ggml: q[S_k, H_k, T, B], k[S_k, H_k, T, B], v[S_v, H_v, T, B]
    // OV:   q[B, T, H_k, S_k], k[B, T, H_k, S_k], v[B, T, H_v, S_v]
    // ggml: g[1 or S_v, H_v, T, B], beta[1, H_v, T, B]
    // OV:   g[B, T, H_v, 1 or S_v], beta[B, T, H_v, 1]
    // ggml: state[S_v, S_v, H_v, B]
    // OV:   state[B, H_v, S_v, S_v]
    auto q = process_view_உள்ளீடு_new(context, 0);
    auto k = process_view_உள்ளீடு_new(context, 1);
    auto v = process_view_உள்ளீடு_new(context, 2);
    auto g = process_view_உள்ளீடு_new(context, 3);
    auto beta = process_view_உள்ளீடு_new(context, 4);
    auto state = process_view_உள்ளீடு_new(context, 5);

    auto v_shape = context.get_உள்ளீடு_shape(2).to_shape();  // [B, T, H_v, S_v]
    auto q_shape = context.get_உள்ளீடு_shape(0).to_shape();  // [B, T, H_k, S_k]
    auto g_shape = context.get_உள்ளீடு_shape(3).to_shape();  // [B, T, H_v, 1 or S_v]

    மாறிலி int64_t B = v_shape[0];
    மாறிலி int64_t T = v_shape[1];
    மாறிலி int64_t H_v = v_shape[2];
    மாறிலி int64_t S_v = v_shape[3];
    மாறிலி int64_t H_k = q_shape[2];
    மாறிலி தருக்க kda = (g_shape[3] == (size_t) S_v);

    மாறிலி int64_t rq1 = H_v / H_k;  // head repeat factor
    மாறிலி தசம scale = 1.0f / std::sqrt((தசம) S_v);

    auto axis_1 = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
    auto axis_2 = ov::op::v0::Constant::create(ov::element::i64, {1}, {2});

    // Transpose inputs from [B, T, H, S] to [B, H, T, S] for easier per-head processing
    auto perm_0213 = ov::op::v0::Constant::create(ov::element::i64, {4}, std::vector<int64_t>{0, 2, 1, 3});
    auto q_t = std::make_shared<ov::op::v1::Transpose>(q, perm_0213);        // [B, H_k, T, S_k]
    auto k_t = std::make_shared<ov::op::v1::Transpose>(k, perm_0213);        // [B, H_k, T, S_k]
    auto v_t = std::make_shared<ov::op::v1::Transpose>(v, perm_0213);        // [B, H_v, T, S_v]
    auto g_t = std::make_shared<ov::op::v1::Transpose>(g, perm_0213);        // [B, H_v, T, 1 or S_v]
    auto beta_t = std::make_shared<ov::op::v1::Transpose>(beta, perm_0213);  // [B, H_v, T, 1]

    // Broadcast Q, K heads to match V heads if GQA is used (H_v > H_k)
    ov::Output<ov::Node> q_bh = q_t;
    ov::Output<ov::Node> k_bh = k_t;
    எனில் (rq1 > 1) {
        auto q_unsq = std::make_shared<ov::op::v0::Unsqueeze>(q_t, axis_2);  // [B, H_k, 1, T, S]
        auto k_unsq = std::make_shared<ov::op::v0::Unsqueeze>(k_t, axis_2);  // [B, H_k, 1, T, S]

        auto bcast_shape = ov::op::v0::Constant::create(ov::element::i64, {5}, std::vector<int64_t>{1, 1, rq1, 1, 1});
        auto q_bcast =
            std::make_shared<ov::op::v3::Broadcast>(q_unsq, bcast_shape, ov::op::BroadcastType::BIDIRECTIONAL);
        auto k_bcast =
            std::make_shared<ov::op::v3::Broadcast>(k_unsq, bcast_shape, ov::op::BroadcastType::BIDIRECTIONAL);

        // Transpose [B, H_k, rq1, T, S] -> [B, rq1, H_k, T, S] so that reshape merges
        // as [rq1, H_k] giving repeat-blocks pattern matching CPU: iq1 = iv1 % H_k
        auto perm_5d = ov::op::v0::Constant::create(ov::element::i64, {5}, std::vector<int64_t>{0, 2, 1, 3, 4});
        auto q_transposed = std::make_shared<ov::op::v1::Transpose>(q_bcast, perm_5d);
        auto k_transposed = std::make_shared<ov::op::v1::Transpose>(k_bcast, perm_5d);

        auto new_shape = ov::op::v0::Constant::create(ov::element::i64, {4}, std::vector<int64_t>{B, H_v, T, S_v});
        q_bh = std::make_shared<ov::op::v1::Reshape>(q_transposed, new_shape, பொய்);
        k_bh = std::make_shared<ov::op::v1::Reshape>(k_transposed, new_shape, பொய்);
    }

    // Merge batch and head dims: [B*H_v, T, S_v]
    auto merge_bh = [&](மாறிலி ov::Output<ov::Node> & x, int64_t last_dim) {
        auto shape = ov::op::v0::Constant::create(ov::element::i64, {3}, std::vector<int64_t>{B * H_v, T, last_dim});
        திரும்பு std::make_shared<ov::op::v1::Reshape>(x, shape, பொய்);
    };

    auto q_m = merge_bh(q_bh, S_v);           // [B*H_v, T, S_v]
    auto k_m = merge_bh(k_bh, S_v);           // [B*H_v, T, S_v]
    auto v_m = merge_bh(v_t, S_v);            // [B*H_v, T, S_v]
    auto g_m = merge_bh(g_t, kda ? S_v : 1);  // [B*H_v, T, 1 or S_v]
    auto beta_m = merge_bh(beta_t, 1);        // [B*H_v, T, 1]

    // State: [B, H_v, S_v, S_v] -> [B*H_v, S_v, S_v]
    auto state_shape = ov::op::v0::Constant::create(ov::element::i64, {3}, std::vector<int64_t>{B * H_v, S_v, S_v});
    auto state_m = std::make_shared<ov::op::v1::Reshape>(state, state_shape, பொய்);

    auto scale_const = ov::op::v0::Constant::create(ov::element::f32, {}, std::vector<தசம>{scale});

    // --- Build Loop body ---
    // Body parameters (no iteration counter needed, use -1 in special ports)
    auto body_state = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic());
    auto body_q = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic());
    auto body_k = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic());
    auto body_v = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic());
    auto body_g = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic());
    auto body_beta = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic());
    auto body_iter = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::Shape{1});

    // Condition output (always true - we rely on trip_count for termination)
    auto body_cond_out = ov::op::v0::Constant::create(ov::element::boolean, ov::Shape{1}, std::vector<தருக்க>{உண்மை});

    // Gather current token from invariant inputs using iteration counter
    auto q_t_cur = std::make_shared<ov::op::v8::Gather>(body_q, body_iter, axis_1);     // [B*H_v, 1, S_v]
    auto k_t_cur = std::make_shared<ov::op::v8::Gather>(body_k, body_iter, axis_1);     // [B*H_v, 1, S_v]
    auto v_t_cur = std::make_shared<ov::op::v8::Gather>(body_v, body_iter, axis_1);     // [B*H_v, 1, S_v]
    auto g_t_cur = std::make_shared<ov::op::v8::Gather>(body_g, body_iter, axis_1);     // [B*H_v, 1, 1 or S_v]
    auto b_t_cur = std::make_shared<ov::op::v8::Gather>(body_beta, body_iter, axis_1);  // [B*H_v, 1, 1]

    // Squeeze token dim
    auto q_cur = std::make_shared<ov::op::v0::Squeeze>(q_t_cur, axis_1);  // [B*H_v, S_v]
    auto k_cur = std::make_shared<ov::op::v0::Squeeze>(k_t_cur, axis_1);  // [B*H_v, S_v]
    auto v_cur = std::make_shared<ov::op::v0::Squeeze>(v_t_cur, axis_1);  // [B*H_v, S_v]
    auto g_cur = std::make_shared<ov::op::v0::Squeeze>(g_t_cur, axis_1);  // [B*H_v, 1 or S_v]
    auto b_cur = std::make_shared<ov::op::v0::Squeeze>(b_t_cur, axis_1);  // [B*H_v, 1]

    // Step 1: Apply decay gate to state
    auto exp_g = std::make_shared<ov::op::v0::Exp>(g_cur);                                // [B*H_v, 1 or S_v]
    auto exp_g_unsq = std::make_shared<ov::op::v0::Unsqueeze>(exp_g, axis_1);             // [B*H_v, 1, 1 or S_v]
    auto state_decayed = std::make_shared<ov::op::v1::Multiply>(body_state, exp_g_unsq);  // [B*H_v, S_v, S_v]

    // Step 2: delta = (v - S @ k) * beta
    auto k_col = std::make_shared<ov::op::v0::Unsqueeze>(k_cur, axis_2);                 // [B*H_v, S_v, 1]
    auto sk = std::make_shared<ov::op::v0::MatMul>(state_decayed, k_col, பொய், பொய்);  // [B*H_v, S_v, 1]
    auto sk_sq = std::make_shared<ov::op::v0::Squeeze>(sk, axis_2);                      // [B*H_v, S_v]
    auto v_minus_sk = std::make_shared<ov::op::v1::Subtract>(v_cur, sk_sq);              // [B*H_v, S_v]
    auto delta = std::make_shared<ov::op::v1::Multiply>(v_minus_sk, b_cur);              // [B*H_v, S_v]

    // Step 3: state += outer(delta, k)
    auto delta_col = std::make_shared<ov::op::v0::Unsqueeze>(delta, axis_2);                 // [B*H_v, S_v, 1]
    auto k_row = std::make_shared<ov::op::v0::Unsqueeze>(k_cur, axis_1);                     // [B*H_v, 1, S_v]
    auto outer_prod = std::make_shared<ov::op::v0::MatMul>(delta_col, k_row, பொய், பொய்);  // [B*H_v, S_v, S_v]
    auto state_updated = std::make_shared<ov::op::v1::Add>(state_decayed, outer_prod);       // [B*H_v, S_v, S_v]

    // Step 4: attn_out = S @ q * scale
    auto q_col = std::make_shared<ov::op::v0::Unsqueeze>(q_cur, axis_2);                 // [B*H_v, S_v, 1]
    auto sq = std::make_shared<ov::op::v0::MatMul>(state_updated, q_col, பொய், பொய்);  // [B*H_v, S_v, 1]
    auto sq_squeezed = std::make_shared<ov::op::v0::Squeeze>(sq, axis_2);                // [B*H_v, S_v]
    auto attn_out = std::make_shared<ov::op::v1::Multiply>(sq_squeezed, scale_const);    // [B*H_v, S_v]

    // Unsqueeze attn_out to [B*H_v, 1, S_v] for scan output concatenation
    auto attn_out_unsq = std::make_shared<ov::op::v0::Unsqueeze>(attn_out, axis_1);  // [B*H_v, 1, S_v]

    // --- Assemble Loop ---
    // Body: results = [condition, state_updated, attn_out_unsq]
    auto body = std::make_shared<ov::Model>(
        ov::OutputVector{body_cond_out, state_updated, attn_out_unsq},
        ov::ParameterVector{body_iter, body_state, body_q, body_k, body_v, body_g, body_beta});

    auto trip_மொத்தம் = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{T});
    auto exec_cond = ov::op::v0::Constant::create(ov::element::boolean, ov::Shape{1}, std::vector<தருக்க>{உண்மை});

    auto loop = std::make_shared<ov::op::v5::Loop>(trip_மொத்தம், exec_cond);
    loop->set_function(body);
    loop->set_special_body_ports(ov::op::v5::Loop::SpecialBodyPorts{0, 0});

    // Carried state: feeds back from body output 1 to body_state param
    loop->set_merged_உள்ளீடு(body_state, state_m, state_updated);
    // Invariant inputs: passed through unchanged each iteration
    loop->set_invariant_உள்ளீடு(body_q, q_m);
    loop->set_invariant_உள்ளீடு(body_k, k_m);
    loop->set_invariant_உள்ளீடு(body_v, v_m);
    loop->set_invariant_உள்ளீடு(body_g, g_m);
    loop->set_invariant_உள்ளீடு(body_beta, beta_m);

    // Loop outputs:
    // 1) Final state (last iteration value of state_updated)
    auto final_state_out = loop->get_iter_மதிப்பு(state_updated, -1);  // [B*H_v, S_v, S_v]
    // 2) Concatenated attention outputs across all iterations along axis 1
    auto attn_concat_out = loop->get_concatenated_slices(attn_out_unsq, 0, 1, 1, -1, 1);  // [B*H_v, T, S_v]

    // --- Pack outputs to match ggml layout ---
    // ggml output ne = {S_v*H, T*B + S_v*B, 1, 1} -> OV [1, 1, T*B+S_v*B, S_v*H_v]
    // attn: [B, T, H_v, S_v] row-major, state: [B, H_v, S_v, S_v] row-major

    // attn: [B*H_v, T, S_v] -> [B, H_v, T, S_v] -> transpose to [B, T, H_v, S_v] -> flatten
    auto attn_4d_shape = ov::op::v0::Constant::create(ov::element::i64, {4}, std::vector<int64_t>{B, H_v, T, S_v});
    auto attn_4d = std::make_shared<ov::op::v1::Reshape>(attn_concat_out, attn_4d_shape, பொய்);
    auto attn_perm = std::make_shared<ov::op::v1::Transpose>(attn_4d, perm_0213);  // [B, T, H_v, S_v]

    auto flat_shape_1d = ov::op::v0::Constant::create(ov::element::i64, {1}, std::vector<int64_t>{-1});
    auto attn_1d = std::make_shared<ov::op::v1::Reshape>(attn_perm, flat_shape_1d, பொய்);

    // state: [B*H_v, S_v, S_v] -> [B, H_v, S_v, S_v] -> flatten
    auto state_4d_shape = ov::op::v0::Constant::create(ov::element::i64, {4}, std::vector<int64_t>{B, H_v, S_v, S_v});
    auto state_4d = std::make_shared<ov::op::v1::Reshape>(final_state_out, state_4d_shape, பொய்);
    எனில் (context.get_output_names().size() == 2) {
        // Match the fused translator's direct attention/state contract.
        திரும்பு rename_outputs_with_suffix({attn_perm, state_4d}, context.get_name());
    }

    auto state_1d = std::make_shared<ov::op::v1::Reshape>(state_4d, flat_shape_1d, பொய்);

    // Concat [attn | state] and reshape to final output
    auto packed = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{attn_1d, state_1d}, 0);
    auto out_shape =
        ov::op::v0::Constant::create(ov::element::i64, {4}, std::vector<int64_t>{1, 1, T * B + S_v * B, S_v * H_v});
    auto res = std::make_shared<ov::op::v1::Reshape>(packed, out_shape, பொய்);

    திரும்பு rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
