#अन्तर्भाव "gated_delta_net.hpp"

#अन्तर्भाव "../node_context.h"
#अन्तर्भाव "../op_table.h"
#अन्तर्भाव "../utils.h"
#अन्तर्भाव "ggml-openvino/ggml-openvino-extra.h"

#अन्तर्भाव <cmath>
#अन्तर्भाव <cstdint>
#अन्तर्भाव <memory>
#अन्तर्भाव <numeric>
#अन्तर्भाव <openvino/op/add.hpp>
#अन्तर्भाव <openvino/op/broadcast.hpp>
#अन्तर्भाव <openvino/op/concat.hpp>
#अन्तर्भाव <openvino/op/constant.hpp>
#अन्तर्भाव <openvino/op/convert.hpp>
#अन्तर्भाव <openvino/op/divide.hpp>
#अन्तर्भाव <openvino/op/exp.hpp>
#अन्तर्भाव <openvino/op/gather.hpp>
#अन्तर्भाव <openvino/op/less.hpp>
#अन्तर्भाव <openvino/op/loop.hpp>
#अन्तर्भाव <openvino/op/matmul.hpp>
#अन्तर्भाव <openvino/op/multiply.hpp>
#अन्तर्भाव <openvino/op/reduce_mean.hpp>
#अन्तर्भाव <openvino/op/reshape.hpp>
#अन्तर्भाव <openvino/op/squeeze.hpp>
#अन्तर्भाव <openvino/op/sqrt.hpp>
#अन्तर्भाव <openvino/op/subtract.hpp>
#अन्तर्भाव <openvino/op/tile.hpp>
#अन्तर्भाव <openvino/op/transpose.hpp>
#अन्तर्भाव <openvino/op/unsqueeze.hpp>
#अन्तर्भाव <vector>

नामाकाश ov {
नामाकाश frontend {
नामाकाश ggml {
नामाकाश op {

स्थिर OutputVector translate_gated_delta_net_ref(ध्रुव गाँठ_Context & context);

स्थिर तार्किक match_gdn_l2_norm(ध्रुव Output<Node> & normalized, Output<Node> & निवेश, भिन्न & eps) {
    // Match the RMSNorm decomposition emitted by translate_rms_norm, followed by GGML SCALE.
    ध्रुव स्वतः scale = ov::as_type_ptr<ov::op::v1::Multiply>(normalized.get_गाँठ_shared_ptr());
    यदि (!scale) {
        प्रत्यागम असत्य;
    }
    ध्रुव स्वतः factor = ov::as_type_ptr<ov::op::v0::Constant>(scale->get_निवेश_गाँठ_shared_ptr(1));
    ध्रुव स्वतः rms = ov::as_type_ptr<ov::op::v1::Multiply>(scale->get_निवेश_गाँठ_shared_ptr(0));
    यदि (!factor || ov::shape_size(factor->get_shape()) != 1 || !rms) {
        प्रत्यागम असत्य;
    }
    ध्रुव स्वतः x = rms->निवेश_मूल्य(0);
    ध्रुव स्वतः & shape = x.get_partial_shape();
    यदि (x.get_element_type() != ov::element::f32 || shape.rank() != 4 || shape[3].is_dynamic() ||
        shape[3].get_length() <= 0 || normalized.get_partial_shape() != shape) {
        प्रत्यागम असत्य;
    }
    ध्रुव भिन्न dim = static_cast<भिन्न>(shape[3].get_length());
    यदि (factor->cast_vector<भिन्न>()[0] != 1.0f / std::वर्ग_मूल(dim)) {
        प्रत्यागम असत्य;
    }
    ध्रुव स्वतः reciprocal = ov::as_type_ptr<ov::op::v1::Divide>(rms->get_निवेश_गाँठ_shared_ptr(1));
    यदि (!reciprocal) {
        प्रत्यागम असत्य;
    }
    ध्रुव स्वतः one = ov::as_type_ptr<ov::op::v0::Constant>(reciprocal->get_निवेश_गाँठ_shared_ptr(0));
    ध्रुव स्वतः मूल = ov::as_type_ptr<ov::op::v0::Sqrt>(reciprocal->get_निवेश_गाँठ_shared_ptr(1));
    यदि (!one || ov::shape_size(one->get_shape()) != 1 || one->cast_vector<भिन्न>()[0] != 1.0f || !मूल) {
        प्रत्यागम असत्य;
    }
    ध्रुव स्वतः add = ov::as_type_ptr<ov::op::v1::Add>(मूल->get_निवेश_गाँठ_shared_ptr(0));
    यदि (!add) {
        प्रत्यागम असत्य;
    }
    ध्रुव स्वतः mean = ov::as_type_ptr<ov::op::v1::ReduceMean>(add->get_निवेश_गाँठ_shared_ptr(0));
    ध्रुव स्वतः rms_eps = ov::as_type_ptr<ov::op::v0::Constant>(add->get_निवेश_गाँठ_shared_ptr(1));
    यदि (!mean || !mean->get_keep_dims() || !rms_eps || ov::shape_size(rms_eps->get_shape()) != 1) {
        प्रत्यागम असत्य;
    }
    ध्रुव स्वतः axes = ov::as_type_ptr<ov::op::v0::Constant>(mean->get_निवेश_गाँठ_shared_ptr(1));
    ध्रुव स्वतः square = ov::as_type_ptr<ov::op::v1::Multiply>(mean->get_निवेश_गाँठ_shared_ptr(0));
    यदि (!axes || axes->cast_vector<int64_t>() != std::vector<int64_t>{-1} || !square ||
        square->निवेश_मूल्य(0) != x || square->निवेश_मूल्य(1) != x) {
        प्रत्यागम असत्य;
    }
    // RMSNorm(x, rms_eps) / sqrt(D) = x / sqrt(sum(x*x) + D*rms_eps).
    eps = dim * rms_eps->cast_vector<भिन्न>()[0];
    यदि (!std::है_सीमित(eps) || eps <= 0.0f) {
        प्रत्यागम असत्य;
    }
    निवेश = x;
    प्रत्यागम सत्य;
}

OutputVector translate_gated_delta_net(ध्रुव गाँठ_Context & context) {
    स्वतः v_shape = context.get_निवेश_shape(2).to_shape();  // [B, T, H_v, S_v]
    स्वतः q_shape = context.get_निवेश_shape(0).to_shape();  // [B, T, H_k, S_k]

    // Fused GatedDeltaNet op only supports scalar gate (kda=0).
    // Fall back to reference implementation for per-key-dimension gating.
    // if (kda) {
    //     return translate_gated_delta_net_ref(context);
    // }

    // const int64_t B = v_shape[0];
    // const int64_t T = v_shape[1];
    ध्रुव int64_t H_v = v_shape[2];
    ध्रुव int64_t S_v = v_shape[3];
    ध्रुव int64_t H_k = q_shape[2];
    // const int64_t S_k = q_shape[3];

    स्वतः q = context.get_निवेश(0);
    स्वतः k = context.get_निवेश(1);
    स्वतः v = process_view_निवेश(context, 2, H_v * S_v, 3);
    स्वतः g = context.get_निवेश(3);
    स्वतः beta = context.get_निवेश(4);
    स्वतः state = context.get_निवेश(5);

    Output<Node> raw_q, raw_k;
    भिन्न q_eps = 1e-6f, k_eps = 1e-6f;
    ध्रुव तार्किक fuse_qk_l2norm = ggml_openvino_is_gpu() &&
        match_gdn_l2_norm(q, raw_q, q_eps) && match_gdn_l2_norm(k, raw_k, k_eps);
    यदि (fuse_qk_l2norm) {
        // Keep head tiling below; GDN applies normalization per head and keeps its attention scale.
        q = raw_q;
        k = raw_k;
    }

    // ggml maps GQA heads in tiled order, while OV GDN maps repeated heads in grouped order.
    यदि (H_v != H_k) {
        ध्रुव int64_t repeat = H_v / H_k;
        स्वतः repeats = ov::op::v0::Constant::create(ov::element::i64, {4}, std::vector<int64_t>{1, 1, repeat, 1});
        q = std::make_shared<ov::op::v0::Tile>(q, repeats);
        k = std::make_shared<ov::op::v0::Tile>(k, repeats);
    }

    यदि (context.get_view_निवेश_size(2)) {
        // Same as l2_norm case 1
        v = std::make_shared<ov::op::v0::Squeeze>(v, ov::op::v0::Constant::create(ov::element::i64, {1}, {0}));
        स्वतः v_shape = context.get_निवेश_shape(2).to_shape();
        std::vector<int64_t> reshape_pattern = {0, 0, (int64_t) v_shape[2], (int64_t) v_shape[3]};
        v = std::make_shared<ov::op::v1::Reshape>(
            v, ov::op::v0::Constant::create(ov::element::i64, {4}, reshape_pattern), सत्य);
    }

    // ggml state layout (OV notation): [B, H_v, value_dim, key_dim]
    // GatedDeltaNet op expects: [B, H_v, key_dim, value_dim]
    स्वतः state_perm = ov::op::v0::Constant::create(ov::element::i64, {4}, std::vector<int64_t>{0, 1, 3, 2});
    state = std::make_shared<ov::op::v1::Transpose>(state, state_perm);

    g = std::make_shared<ov::op::v0::Squeeze>(g, ov::op::v0::Constant::create(ov::element::i64, {1}, {3}));
    beta = std::make_shared<ov::op::v0::Squeeze>(beta, ov::op::v0::Constant::create(ov::element::i64, {1}, {3}));

    यदि (context.has_निवेश("chunk_valid_len")) {
        // The last prefill chunk is padded with fabricated tokens. The recurrence is
        //   S_t = S_{t-1} * exp(g_t) + k_t (x) ((v_t - S_{t-1}^T k_t) * beta_t)
        // so forcing g = 0 and beta = 0 makes a padded step an exact identity and keeps the final
        // state equal to the state after the last real token. Attention output at those positions
        // is garbage but never read.
        ध्रुव स्वतः & g_ps = g.get_partial_shape();
        FRONT_END_OP_CONVERSION_CHECK(g_ps.rank().is_static() && g_ps.rank().get_length() == 3 && g_ps[1].is_static(),
                                      "GATED_DELTA_NET pad masking requires a static token dimension");
        ध्रुव int64_t n_tokens = g_ps[1].get_length();
        std::vector<int64_t> positions(n_tokens);
        std::iota(positions.begin(), positions.end(), 0);
        स्वतः valid = std::make_shared<ov::op::v1::Less>(
            ov::op::v0::Constant::create(ov::element::i64, {(माप_प्रकार) n_tokens}, positions),
            context.get_निवेश("chunk_valid_len"));
        स्वतः mask = std::make_shared<ov::op::v0::Unsqueeze>(
            std::make_shared<ov::op::v0::Convert>(valid, g.get_element_type()),
            ov::op::v0::Constant::create(ov::element::i64, {2}, std::vector<int64_t>{0, 2}));
        g = std::make_shared<ov::op::v1::Multiply>(g, mask);
        beta = std::make_shared<ov::op::v1::Multiply>(beta, mask);
    }

    // std::cout << "GatedDeltaNet input shapes: q=" << q.get_partial_shape() << ", k=" << k.get_partial_shape()
    //           << ", v=" << v.get_partial_shape() << ", g=" << g.get_partial_shape()
    //           << ", beta=" << beta.get_partial_shape() << ", state=" << state.get_partial_shape() << std::endl;

    स्वतः gdn = std::make_shared<ov::op::internal::GatedDeltaNet>(q, k, v, state, g, beta, fuse_qk_l2norm, q_eps, k_eps);
    स्वतः attn_4d = gdn->output(0);
    स्वतः state_4d = gdn->output(1);  // [B, H_v, key_dim, value_dim]

    // std::cout << "GatedDeltaNet output shapes: attn=" << gdn->output(0).get_partial_shape()
    //           << ", new_state=" << gdn->output(1).get_partial_shape() << std::endl;

    // Transpose output state back to ggml layout [B, H_v, value_dim, key_dim]
    स्वतः state_transposed = std::make_shared<ov::op::v1::Transpose>(state_4d, state_perm);
    यदि (context.get_output_names().size() == 2) {
        // The canonical graph consumes the packed GGML result only through separate attention
        // and state VIEWs. Publish the native outputs under those VIEW names to avoid
        // flatten -> concat -> reshape -> slice -> reshape chains. This also works for B > 1.
        प्रत्यागम rename_outputs_with_suffix({attn_4d, state_transposed}, context.get_name());
    }

    स्वतः flat_shape_1d = ov::op::v0::Constant::create(ov::element::i64, {1}, {-1});
    स्वतः attn = std::make_shared<ov::op::v1::Reshape>(attn_4d, flat_shape_1d, असत्य);
    स्वतः new_state = std::make_shared<ov::op::v1::Reshape>(state_transposed, flat_shape_1d, असत्य);
    स्वतः packed = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{attn, new_state}, 0);
    स्वतः out_shape = ov::op::v0::Constant::create(ov::element::i64, {4},
                                                  std::vector<int64_t>{1, 1, -1 /*T * B + S_v * B*/, S_v * H_v});
    स्वतः res = std::make_shared<ov::op::v1::Reshape>(packed, out_shape, असत्य);

    प्रत्यागम rename_outputs_with_suffix({res}, context.get_name());
}

स्थिर OutputVector translate_gated_delta_net_ref(ध्रुव गाँठ_Context & context) {
    अंक_inputs_check(context, 6, 6);

    // Inputs (OV shapes are reversed from ggml):
    // ggml: q[S_k, H_k, T, B], k[S_k, H_k, T, B], v[S_v, H_v, T, B]
    // OV:   q[B, T, H_k, S_k], k[B, T, H_k, S_k], v[B, T, H_v, S_v]
    // ggml: g[1 or S_v, H_v, T, B], beta[1, H_v, T, B]
    // OV:   g[B, T, H_v, 1 or S_v], beta[B, T, H_v, 1]
    // ggml: state[S_v, S_v, H_v, B]
    // OV:   state[B, H_v, S_v, S_v]
    स्वतः q = process_view_निवेश_new(context, 0);
    स्वतः k = process_view_निवेश_new(context, 1);
    स्वतः v = process_view_निवेश_new(context, 2);
    स्वतः g = process_view_निवेश_new(context, 3);
    स्वतः beta = process_view_निवेश_new(context, 4);
    स्वतः state = process_view_निवेश_new(context, 5);

    स्वतः v_shape = context.get_निवेश_shape(2).to_shape();  // [B, T, H_v, S_v]
    स्वतः q_shape = context.get_निवेश_shape(0).to_shape();  // [B, T, H_k, S_k]
    स्वतः g_shape = context.get_निवेश_shape(3).to_shape();  // [B, T, H_v, 1 or S_v]

    ध्रुव int64_t B = v_shape[0];
    ध्रुव int64_t T = v_shape[1];
    ध्रुव int64_t H_v = v_shape[2];
    ध्रुव int64_t S_v = v_shape[3];
    ध्रुव int64_t H_k = q_shape[2];
    ध्रुव तार्किक kda = (g_shape[3] == (माप_प्रकार) S_v);

    ध्रुव int64_t rq1 = H_v / H_k;  // head repeat factor
    ध्रुव भिन्न scale = 1.0f / std::वर्ग_मूल((भिन्न) S_v);

    स्वतः axis_1 = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
    स्वतः axis_2 = ov::op::v0::Constant::create(ov::element::i64, {1}, {2});

    // Transpose inputs from [B, T, H, S] to [B, H, T, S] for easier per-head processing
    स्वतः perm_0213 = ov::op::v0::Constant::create(ov::element::i64, {4}, std::vector<int64_t>{0, 2, 1, 3});
    स्वतः q_t = std::make_shared<ov::op::v1::Transpose>(q, perm_0213);        // [B, H_k, T, S_k]
    स्वतः k_t = std::make_shared<ov::op::v1::Transpose>(k, perm_0213);        // [B, H_k, T, S_k]
    स्वतः v_t = std::make_shared<ov::op::v1::Transpose>(v, perm_0213);        // [B, H_v, T, S_v]
    स्वतः g_t = std::make_shared<ov::op::v1::Transpose>(g, perm_0213);        // [B, H_v, T, 1 or S_v]
    स्वतः beta_t = std::make_shared<ov::op::v1::Transpose>(beta, perm_0213);  // [B, H_v, T, 1]

    // Broadcast Q, K heads to match V heads if GQA is used (H_v > H_k)
    ov::Output<ov::Node> q_bh = q_t;
    ov::Output<ov::Node> k_bh = k_t;
    यदि (rq1 > 1) {
        स्वतः q_unsq = std::make_shared<ov::op::v0::Unsqueeze>(q_t, axis_2);  // [B, H_k, 1, T, S]
        स्वतः k_unsq = std::make_shared<ov::op::v0::Unsqueeze>(k_t, axis_2);  // [B, H_k, 1, T, S]

        स्वतः bcast_shape = ov::op::v0::Constant::create(ov::element::i64, {5}, std::vector<int64_t>{1, 1, rq1, 1, 1});
        स्वतः q_bcast =
            std::make_shared<ov::op::v3::Broadcast>(q_unsq, bcast_shape, ov::op::BroadcastType::BIDIRECTIONAL);
        स्वतः k_bcast =
            std::make_shared<ov::op::v3::Broadcast>(k_unsq, bcast_shape, ov::op::BroadcastType::BIDIRECTIONAL);

        // Transpose [B, H_k, rq1, T, S] -> [B, rq1, H_k, T, S] so that reshape merges
        // as [rq1, H_k] giving repeat-blocks pattern matching CPU: iq1 = iv1 % H_k
        स्वतः perm_5d = ov::op::v0::Constant::create(ov::element::i64, {5}, std::vector<int64_t>{0, 2, 1, 3, 4});
        स्वतः q_transposed = std::make_shared<ov::op::v1::Transpose>(q_bcast, perm_5d);
        स्वतः k_transposed = std::make_shared<ov::op::v1::Transpose>(k_bcast, perm_5d);

        स्वतः new_shape = ov::op::v0::Constant::create(ov::element::i64, {4}, std::vector<int64_t>{B, H_v, T, S_v});
        q_bh = std::make_shared<ov::op::v1::Reshape>(q_transposed, new_shape, असत्य);
        k_bh = std::make_shared<ov::op::v1::Reshape>(k_transposed, new_shape, असत्य);
    }

    // Merge batch and head dims: [B*H_v, T, S_v]
    स्वतः merge_bh = [&](ध्रुव ov::Output<ov::Node> & x, int64_t last_dim) {
        स्वतः shape = ov::op::v0::Constant::create(ov::element::i64, {3}, std::vector<int64_t>{B * H_v, T, last_dim});
        प्रत्यागम std::make_shared<ov::op::v1::Reshape>(x, shape, असत्य);
    };

    स्वतः q_m = merge_bh(q_bh, S_v);           // [B*H_v, T, S_v]
    स्वतः k_m = merge_bh(k_bh, S_v);           // [B*H_v, T, S_v]
    स्वतः v_m = merge_bh(v_t, S_v);            // [B*H_v, T, S_v]
    स्वतः g_m = merge_bh(g_t, kda ? S_v : 1);  // [B*H_v, T, 1 or S_v]
    स्वतः beta_m = merge_bh(beta_t, 1);        // [B*H_v, T, 1]

    // State: [B, H_v, S_v, S_v] -> [B*H_v, S_v, S_v]
    स्वतः state_shape = ov::op::v0::Constant::create(ov::element::i64, {3}, std::vector<int64_t>{B * H_v, S_v, S_v});
    स्वतः state_m = std::make_shared<ov::op::v1::Reshape>(state, state_shape, असत्य);

    स्वतः scale_const = ov::op::v0::Constant::create(ov::element::f32, {}, std::vector<भिन्न>{scale});

    // --- Build Loop body ---
    // Body parameters (no iteration counter needed, use -1 in special ports)
    स्वतः body_state = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic());
    स्वतः body_q = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic());
    स्वतः body_k = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic());
    स्वतः body_v = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic());
    स्वतः body_g = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic());
    स्वतः body_beta = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic());
    स्वतः body_iter = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::Shape{1});

    // Condition output (always true - we rely on trip_count for termination)
    स्वतः body_cond_out = ov::op::v0::Constant::create(ov::element::boolean, ov::Shape{1}, std::vector<तार्किक>{सत्य});

    // Gather current token from invariant inputs using iteration counter
    स्वतः q_t_cur = std::make_shared<ov::op::v8::Gather>(body_q, body_iter, axis_1);     // [B*H_v, 1, S_v]
    स्वतः k_t_cur = std::make_shared<ov::op::v8::Gather>(body_k, body_iter, axis_1);     // [B*H_v, 1, S_v]
    स्वतः v_t_cur = std::make_shared<ov::op::v8::Gather>(body_v, body_iter, axis_1);     // [B*H_v, 1, S_v]
    स्वतः g_t_cur = std::make_shared<ov::op::v8::Gather>(body_g, body_iter, axis_1);     // [B*H_v, 1, 1 or S_v]
    स्वतः b_t_cur = std::make_shared<ov::op::v8::Gather>(body_beta, body_iter, axis_1);  // [B*H_v, 1, 1]

    // Squeeze token dim
    स्वतः q_cur = std::make_shared<ov::op::v0::Squeeze>(q_t_cur, axis_1);  // [B*H_v, S_v]
    स्वतः k_cur = std::make_shared<ov::op::v0::Squeeze>(k_t_cur, axis_1);  // [B*H_v, S_v]
    स्वतः v_cur = std::make_shared<ov::op::v0::Squeeze>(v_t_cur, axis_1);  // [B*H_v, S_v]
    स्वतः g_cur = std::make_shared<ov::op::v0::Squeeze>(g_t_cur, axis_1);  // [B*H_v, 1 or S_v]
    स्वतः b_cur = std::make_shared<ov::op::v0::Squeeze>(b_t_cur, axis_1);  // [B*H_v, 1]

    // Step 1: Apply decay gate to state
    स्वतः exp_g = std::make_shared<ov::op::v0::Exp>(g_cur);                                // [B*H_v, 1 or S_v]
    स्वतः exp_g_unsq = std::make_shared<ov::op::v0::Unsqueeze>(exp_g, axis_1);             // [B*H_v, 1, 1 or S_v]
    स्वतः state_decayed = std::make_shared<ov::op::v1::Multiply>(body_state, exp_g_unsq);  // [B*H_v, S_v, S_v]

    // Step 2: delta = (v - S @ k) * beta
    स्वतः k_col = std::make_shared<ov::op::v0::Unsqueeze>(k_cur, axis_2);                 // [B*H_v, S_v, 1]
    स्वतः sk = std::make_shared<ov::op::v0::MatMul>(state_decayed, k_col, असत्य, असत्य);  // [B*H_v, S_v, 1]
    स्वतः sk_sq = std::make_shared<ov::op::v0::Squeeze>(sk, axis_2);                      // [B*H_v, S_v]
    स्वतः v_minus_sk = std::make_shared<ov::op::v1::Subtract>(v_cur, sk_sq);              // [B*H_v, S_v]
    स्वतः delta = std::make_shared<ov::op::v1::Multiply>(v_minus_sk, b_cur);              // [B*H_v, S_v]

    // Step 3: state += outer(delta, k)
    स्वतः delta_col = std::make_shared<ov::op::v0::Unsqueeze>(delta, axis_2);                 // [B*H_v, S_v, 1]
    स्वतः k_row = std::make_shared<ov::op::v0::Unsqueeze>(k_cur, axis_1);                     // [B*H_v, 1, S_v]
    स्वतः outer_prod = std::make_shared<ov::op::v0::MatMul>(delta_col, k_row, असत्य, असत्य);  // [B*H_v, S_v, S_v]
    स्वतः state_updated = std::make_shared<ov::op::v1::Add>(state_decayed, outer_prod);       // [B*H_v, S_v, S_v]

    // Step 4: attn_out = S @ q * scale
    स्वतः q_col = std::make_shared<ov::op::v0::Unsqueeze>(q_cur, axis_2);                 // [B*H_v, S_v, 1]
    स्वतः sq = std::make_shared<ov::op::v0::MatMul>(state_updated, q_col, असत्य, असत्य);  // [B*H_v, S_v, 1]
    स्वतः sq_squeezed = std::make_shared<ov::op::v0::Squeeze>(sq, axis_2);                // [B*H_v, S_v]
    स्वतः attn_out = std::make_shared<ov::op::v1::Multiply>(sq_squeezed, scale_const);    // [B*H_v, S_v]

    // Unsqueeze attn_out to [B*H_v, 1, S_v] for scan output concatenation
    स्वतः attn_out_unsq = std::make_shared<ov::op::v0::Unsqueeze>(attn_out, axis_1);  // [B*H_v, 1, S_v]

    // --- Assemble Loop ---
    // Body: results = [condition, state_updated, attn_out_unsq]
    स्वतः body = std::make_shared<ov::Model>(
        ov::OutputVector{body_cond_out, state_updated, attn_out_unsq},
        ov::ParameterVector{body_iter, body_state, body_q, body_k, body_v, body_g, body_beta});

    स्वतः trip_गिनती = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{T});
    स्वतः exec_cond = ov::op::v0::Constant::create(ov::element::boolean, ov::Shape{1}, std::vector<तार्किक>{सत्य});

    स्वतः loop = std::make_shared<ov::op::v5::Loop>(trip_गिनती, exec_cond);
    loop->set_function(body);
    loop->set_special_body_ports(ov::op::v5::Loop::SpecialBodyPorts{0, 0});

    // Carried state: feeds back from body output 1 to body_state param
    loop->set_merged_निवेश(body_state, state_m, state_updated);
    // Invariant inputs: passed through unchanged each iteration
    loop->set_invariant_निवेश(body_q, q_m);
    loop->set_invariant_निवेश(body_k, k_m);
    loop->set_invariant_निवेश(body_v, v_m);
    loop->set_invariant_निवेश(body_g, g_m);
    loop->set_invariant_निवेश(body_beta, beta_m);

    // Loop outputs:
    // 1) Final state (last iteration value of state_updated)
    स्वतः final_state_out = loop->get_iter_मूल्य(state_updated, -1);  // [B*H_v, S_v, S_v]
    // 2) Concatenated attention outputs across all iterations along axis 1
    स्वतः attn_concat_out = loop->get_concatenated_slices(attn_out_unsq, 0, 1, 1, -1, 1);  // [B*H_v, T, S_v]

    // --- Pack outputs to match ggml layout ---
    // ggml output ne = {S_v*H, T*B + S_v*B, 1, 1} -> OV [1, 1, T*B+S_v*B, S_v*H_v]
    // attn: [B, T, H_v, S_v] row-major, state: [B, H_v, S_v, S_v] row-major

    // attn: [B*H_v, T, S_v] -> [B, H_v, T, S_v] -> transpose to [B, T, H_v, S_v] -> flatten
    स्वतः attn_4d_shape = ov::op::v0::Constant::create(ov::element::i64, {4}, std::vector<int64_t>{B, H_v, T, S_v});
    स्वतः attn_4d = std::make_shared<ov::op::v1::Reshape>(attn_concat_out, attn_4d_shape, असत्य);
    स्वतः attn_perm = std::make_shared<ov::op::v1::Transpose>(attn_4d, perm_0213);  // [B, T, H_v, S_v]

    स्वतः flat_shape_1d = ov::op::v0::Constant::create(ov::element::i64, {1}, std::vector<int64_t>{-1});
    स्वतः attn_1d = std::make_shared<ov::op::v1::Reshape>(attn_perm, flat_shape_1d, असत्य);

    // state: [B*H_v, S_v, S_v] -> [B, H_v, S_v, S_v] -> flatten
    स्वतः state_4d_shape = ov::op::v0::Constant::create(ov::element::i64, {4}, std::vector<int64_t>{B, H_v, S_v, S_v});
    स्वतः state_4d = std::make_shared<ov::op::v1::Reshape>(final_state_out, state_4d_shape, असत्य);
    यदि (context.get_output_names().size() == 2) {
        // Match the fused translator's direct attention/state contract.
        प्रत्यागम rename_outputs_with_suffix({attn_perm, state_4d}, context.get_name());
    }

    स्वतः state_1d = std::make_shared<ov::op::v1::Reshape>(state_4d, flat_shape_1d, असत्य);

    // Concat [attn | state] and reshape to final output
    स्वतः packed = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{attn_1d, state_1d}, 0);
    स्वतः out_shape =
        ov::op::v0::Constant::create(ov::element::i64, {4}, std::vector<int64_t>{1, 1, T * B + S_v * B, S_v * H_v});
    स्वतः res = std::make_shared<ov::op::v1::Reshape>(packed, out_shape, असत्य);

    प्रत्यागम rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
