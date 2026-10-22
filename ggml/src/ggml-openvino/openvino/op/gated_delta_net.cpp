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

स्थिर Output_सदिशम् translate_gated_delta_net_ref(ध्रुव ग्रन्थिः_Context & context);

स्थिर तार्किक मेलनम्_gdn_l2_norm(ध्रुव Output<Node> & normalized, Output<Node> & निवेशः, भिन्न & eps) {
    // Match the RMSNorm decomposition emitted by translate_rms_norm, followed by GGML SCALE.
    ध्रुव स्वचालित scale = ov::as_type_ptr<ov::op::v1::Multiply>(normalized.get_ग्रन्थिः_shared_ptr());
    यदि (!scale) {
        प्रत्यागम असत्य;
    }
    ध्रुव स्वचालित गुणकः = ov::as_type_ptr<ov::op::v0::Constant>(scale->get_निवेशः_ग्रन्थिः_shared_ptr(1));
    ध्रुव स्वचालित rms = ov::as_type_ptr<ov::op::v1::Multiply>(scale->get_निवेशः_ग्रन्थिः_shared_ptr(0));
    यदि (!गुणकः || ov::shape_आकार(गुणकः->get_shape()) != 1 || !rms) {
        प्रत्यागम असत्य;
    }
    ध्रुव स्वचालित x = rms->निवेशः_मानम्(0);
    ध्रुव स्वचालित & shape = x.get_partial_shape();
    यदि (x.get_element_type() != ov::element::f32 || shape.rank() != 4 || shape[3].is_dynamic() ||
        shape[3].get_दीर्घता() <= 0 || normalized.get_partial_shape() != shape) {
        प्रत्यागम असत्य;
    }
    ध्रुव भिन्न dim = static_cast<भिन्न>(shape[3].get_दीर्घता());
    यदि (गुणकः->cast_सदिशम्<भिन्न>()[0] != 1.0f / std::वर्गमूलम्(dim)) {
        प्रत्यागम असत्य;
    }
    ध्रुव स्वचालित reciprocal = ov::as_type_ptr<ov::op::v1::Divide>(rms->get_निवेशः_ग्रन्थिः_shared_ptr(1));
    यदि (!reciprocal) {
        प्रत्यागम असत्य;
    }
    ध्रुव स्वचालित one = ov::as_type_ptr<ov::op::v0::Constant>(reciprocal->get_निवेशः_ग्रन्थिः_shared_ptr(0));
    ध्रुव स्वचालित मूलम् = ov::as_type_ptr<ov::op::v0::Sqrt>(reciprocal->get_निवेशः_ग्रन्थिः_shared_ptr(1));
    यदि (!one || ov::shape_आकार(one->get_shape()) != 1 || one->cast_सदिशम्<भिन्न>()[0] != 1.0f || !मूलम्) {
        प्रत्यागम असत्य;
    }
    ध्रुव स्वचालित सङ्कलय_2 = ov::as_type_ptr<ov::op::v1::सङ्कलय>(मूलम्->get_निवेशः_ग्रन्थिः_shared_ptr(0));
    यदि (!सङ्कलय_2) {
        प्रत्यागम असत्य;
    }
    ध्रुव स्वचालित माध्यम्_2 = ov::as_type_ptr<ov::op::v1::Reduce_माध्यम्>(सङ्कलय_2->get_निवेशः_ग्रन्थिः_shared_ptr(0));
    ध्रुव स्वचालित rms_eps = ov::as_type_ptr<ov::op::v0::Constant>(सङ्कलय_2->get_निवेशः_ग्रन्थिः_shared_ptr(1));
    यदि (!माध्यम्_2 || !माध्यम्_2->get_keep_dims() || !rms_eps || ov::shape_आकार(rms_eps->get_shape()) != 1) {
        प्रत्यागम असत्य;
    }
    ध्रुव स्वचालित axes = ov::as_type_ptr<ov::op::v0::Constant>(माध्यम्_2->get_निवेशः_ग्रन्थिः_shared_ptr(1));
    ध्रुव स्वचालित वर्गफलम् = ov::as_type_ptr<ov::op::v1::Multiply>(माध्यम्_2->get_निवेशः_ग्रन्थिः_shared_ptr(0));
    यदि (!axes || axes->cast_सदिशम्<int64_t>() != std::vector<int64_t>{-1} || !वर्गफलम् ||
        वर्गफलम्->निवेशः_मानम्(0) != x || वर्गफलम्->निवेशः_मानम्(1) != x) {
        प्रत्यागम असत्य;
    }
    // RMSNorm(x, rms_eps) / sqrt(D) = x / sqrt(sum(x*x) + D*rms_eps).
    eps = dim * rms_eps->cast_सदिशम्<भिन्न>()[0];
    यदि (!std::isfinite(eps) || eps <= 0.0f) {
        प्रत्यागम असत्य;
    }
    निवेशः = x;
    प्रत्यागम सत्य;
}

Output_सदिशम् translate_gated_delta_net(ध्रुव ग्रन्थिः_Context & context) {
    स्वचालित v_shape = context.get_निवेशः_shape(2).गन्तव्यम्_shape();  // [B, T, H_v, S_v]
    स्वचालित q_shape = context.get_निवेशः_shape(0).गन्तव्यम्_shape();  // [B, T, H_k, S_k]

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

    स्वचालित q = context.get_निवेशः(0);
    स्वचालित k = context.get_निवेशः(1);
    स्वचालित v = process_view_निवेशः(context, 2, H_v * S_v, 3);
    स्वचालित g = context.get_निवेशः(3);
    स्वचालित beta = context.get_निवेशः(4);
    स्वचालित state = context.get_निवेशः(5);

    Output<Node> raw_q, raw_k;
    भिन्न q_eps = 1e-6f, k_eps = 1e-6f;
    ध्रुव तार्किक fuse_qk_l2norm = ggml_openvino_is_gpu() &&
        मेलनम्_gdn_l2_norm(q, raw_q, q_eps) && मेलनम्_gdn_l2_norm(k, raw_k, k_eps);
    यदि (fuse_qk_l2norm) {
        // Keep head tiling below; GDN applies normalization per head and keeps its attention scale.
        q = raw_q;
        k = raw_k;
    }

    // ggml maps GQA heads in tiled order, while OV GDN maps repeated heads in grouped order.
    यदि (H_v != H_k) {
        ध्रुव int64_t repeat = H_v / H_k;
        स्वचालित repeats = ov::op::v0::Constant::create(ov::element::i64, {4}, std::vector<int64_t>{1, 1, repeat, 1});
        q = std::make_shared<ov::op::v0::Tile>(q, repeats);
        k = std::make_shared<ov::op::v0::Tile>(k, repeats);
    }

    यदि (context.get_view_निवेशः_आकार(2)) {
        // Same as l2_norm case 1
        v = std::make_shared<ov::op::v0::Squeeze>(v, ov::op::v0::Constant::create(ov::element::i64, {1}, {0}));
        स्वचालित v_shape = context.get_निवेशः_shape(2).गन्तव्यम्_shape();
        std::vector<int64_t> reshape_आदर्शः = {0, 0, (int64_t) v_shape[2], (int64_t) v_shape[3]};
        v = std::make_shared<ov::op::v1::Reshape>(
            v, ov::op::v0::Constant::create(ov::element::i64, {4}, reshape_आदर्शः), सत्य);
    }

    // ggml state layout (OV notation): [B, H_v, value_dim, key_dim]
    // GatedDeltaNet op expects: [B, H_v, key_dim, value_dim]
    स्वचालित state_perm = ov::op::v0::Constant::create(ov::element::i64, {4}, std::vector<int64_t>{0, 1, 3, 2});
    state = std::make_shared<ov::op::v1::Transpose>(state, state_perm);

    g = std::make_shared<ov::op::v0::Squeeze>(g, ov::op::v0::Constant::create(ov::element::i64, {1}, {3}));
    beta = std::make_shared<ov::op::v0::Squeeze>(beta, ov::op::v0::Constant::create(ov::element::i64, {1}, {3}));

    यदि (context.has_निवेशः("chunk_valid_len")) {
        // The last prefill chunk is padded with fabricated tokens. The recurrence is
        //   S_t = S_{t-1} * exp(g_t) + k_t (x) ((v_t - S_{t-1}^T k_t) * beta_t)
        // so forcing g = 0 and beta = 0 makes a padded step an exact identity and keeps the final
        // state equal to the state after the last real token. Attention output at those positions
        // is garbage but never read.
        ध्रुव स्वचालित & g_ps = g.get_partial_shape();
        FRONT_अन्तः_OP_CONVERSION_CHECK(g_ps.rank().is_static() && g_ps.rank().get_दीर्घता() == 3 && g_ps[1].is_static(),
                                      "GATED_DELTA_NET pad masking requires a static token dimension");
        ध्रुव int64_t n_tokens = g_ps[1].get_दीर्घता();
        std::vector<int64_t> positions(n_tokens);
        std::iota(positions.begin(), positions.end(), 0);
        स्वचालित valid = std::make_shared<ov::op::v1::Less>(
            ov::op::v0::Constant::create(ov::element::i64, {(size_t) n_tokens}, positions),
            context.get_निवेशः("chunk_valid_len"));
        स्वचालित आवरणम् = std::make_shared<ov::op::v0::Unsqueeze>(
            std::make_shared<ov::op::v0::Convert>(valid, g.get_element_type()),
            ov::op::v0::Constant::create(ov::element::i64, {2}, std::vector<int64_t>{0, 2}));
        g = std::make_shared<ov::op::v1::Multiply>(g, आवरणम्);
        beta = std::make_shared<ov::op::v1::Multiply>(beta, आवरणम्);
    }

    // std::cout << "GatedDeltaNet input shapes: q=" << q.get_partial_shape() << ", k=" << k.get_partial_shape()
    //           << ", v=" << v.get_partial_shape() << ", g=" << g.get_partial_shape()
    //           << ", beta=" << beta.get_partial_shape() << ", state=" << state.get_partial_shape() << std::endl;

    स्वचालित gdn = std::make_shared<ov::op::internal::GatedDeltaNet>(q, k, v, state, g, beta, fuse_qk_l2norm, q_eps, k_eps);
    स्वचालित attn_4d = gdn->output(0);
    स्वचालित state_4d = gdn->output(1);  // [B, H_v, key_dim, value_dim]

    // std::cout << "GatedDeltaNet output shapes: attn=" << gdn->output(0).get_partial_shape()
    //           << ", new_state=" << gdn->output(1).get_partial_shape() << std::endl;

    // Transpose output state back to ggml layout [B, H_v, value_dim, key_dim]
    स्वचालित state_transposed = std::make_shared<ov::op::v1::Transpose>(state_4d, state_perm);
    यदि (context.get_output_नामानि().size() == 2) {
        // The canonical graph consumes the packed GGML result only through separate attention
        // and state VIEWs. Publish the native outputs under those VIEW names to avoid
        // flatten -> concat -> reshape -> slice -> reshape chains. This also works for B > 1.
        प्रत्यागम rename_outputs_with_suffix({attn_4d, state_transposed}, context.get_name());
    }

    स्वचालित flat_shape_1d = ov::op::v0::Constant::create(ov::element::i64, {1}, {-1});
    स्वचालित attn = std::make_shared<ov::op::v1::Reshape>(attn_4d, flat_shape_1d, असत्य);
    स्वचालित new_state = std::make_shared<ov::op::v1::Reshape>(state_transposed, flat_shape_1d, असत्य);
    स्वचालित packed = std::make_shared<ov::op::v0::Concat>(ov::Output_सदिशम्{attn, new_state}, 0);
    स्वचालित बहिर्गमनम्_shape = ov::op::v0::Constant::create(ov::element::i64, {4},
                                                  std::vector<int64_t>{1, 1, -1 /*T * B + S_v * B*/, S_v * H_v});
    स्वचालित res = std::make_shared<ov::op::v1::Reshape>(packed, बहिर्गमनम्_shape, असत्य);

    प्रत्यागम rename_outputs_with_suffix({res}, context.get_name());
}

स्थिर Output_सदिशम् translate_gated_delta_net_ref(ध्रुव ग्रन्थिः_Context & context) {
    अङ्कः_inputs_check(context, 6, 6);

    // Inputs (OV shapes are reversed from ggml):
    // ggml: q[S_k, H_k, T, B], k[S_k, H_k, T, B], v[S_v, H_v, T, B]
    // OV:   q[B, T, H_k, S_k], k[B, T, H_k, S_k], v[B, T, H_v, S_v]
    // ggml: g[1 or S_v, H_v, T, B], beta[1, H_v, T, B]
    // OV:   g[B, T, H_v, 1 or S_v], beta[B, T, H_v, 1]
    // ggml: state[S_v, S_v, H_v, B]
    // OV:   state[B, H_v, S_v, S_v]
    स्वचालित q = process_view_निवेशः_new(context, 0);
    स्वचालित k = process_view_निवेशः_new(context, 1);
    स्वचालित v = process_view_निवेशः_new(context, 2);
    स्वचालित g = process_view_निवेशः_new(context, 3);
    स्वचालित beta = process_view_निवेशः_new(context, 4);
    स्वचालित state = process_view_निवेशः_new(context, 5);

    स्वचालित v_shape = context.get_निवेशः_shape(2).गन्तव्यम्_shape();  // [B, T, H_v, S_v]
    स्वचालित q_shape = context.get_निवेशः_shape(0).गन्तव्यम्_shape();  // [B, T, H_k, S_k]
    स्वचालित g_shape = context.get_निवेशः_shape(3).गन्तव्यम्_shape();  // [B, T, H_v, 1 or S_v]

    ध्रुव int64_t B = v_shape[0];
    ध्रुव int64_t T = v_shape[1];
    ध्रुव int64_t H_v = v_shape[2];
    ध्रुव int64_t S_v = v_shape[3];
    ध्रुव int64_t H_k = q_shape[2];
    ध्रुव तार्किक kda = (g_shape[3] == (size_t) S_v);

    ध्रुव int64_t rq1 = H_v / H_k;  // head repeat factor
    ध्रुव भिन्न scale = 1.0f / std::वर्गमूलम्((भिन्न) S_v);

    स्वचालित axis_1 = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
    स्वचालित axis_2 = ov::op::v0::Constant::create(ov::element::i64, {1}, {2});

    // Transpose inputs from [B, T, H, S] to [B, H, T, S] for easier per-head processing
    स्वचालित perm_0213 = ov::op::v0::Constant::create(ov::element::i64, {4}, std::vector<int64_t>{0, 2, 1, 3});
    स्वचालित q_t = std::make_shared<ov::op::v1::Transpose>(q, perm_0213);        // [B, H_k, T, S_k]
    स्वचालित k_t = std::make_shared<ov::op::v1::Transpose>(k, perm_0213);        // [B, H_k, T, S_k]
    स्वचालित v_t = std::make_shared<ov::op::v1::Transpose>(v, perm_0213);        // [B, H_v, T, S_v]
    स्वचालित g_t = std::make_shared<ov::op::v1::Transpose>(g, perm_0213);        // [B, H_v, T, 1 or S_v]
    स्वचालित beta_t = std::make_shared<ov::op::v1::Transpose>(beta, perm_0213);  // [B, H_v, T, 1]

    // Broadcast Q, K heads to match V heads if GQA is used (H_v > H_k)
    ov::Output<ov::Node> q_bh = q_t;
    ov::Output<ov::Node> k_bh = k_t;
    यदि (rq1 > 1) {
        स्वचालित q_unsq = std::make_shared<ov::op::v0::Unsqueeze>(q_t, axis_2);  // [B, H_k, 1, T, S]
        स्वचालित k_unsq = std::make_shared<ov::op::v0::Unsqueeze>(k_t, axis_2);  // [B, H_k, 1, T, S]

        स्वचालित bcast_shape = ov::op::v0::Constant::create(ov::element::i64, {5}, std::vector<int64_t>{1, 1, rq1, 1, 1});
        स्वचालित q_bcast =
            std::make_shared<ov::op::v3::प्रसारणम्>(q_unsq, bcast_shape, ov::op::प्रसारणम्_Type::BIDIRECTIONAL);
        स्वचालित k_bcast =
            std::make_shared<ov::op::v3::प्रसारणम्>(k_unsq, bcast_shape, ov::op::प्रसारणम्_Type::BIDIRECTIONAL);

        // Transpose [B, H_k, rq1, T, S] -> [B, rq1, H_k, T, S] so that reshape merges
        // as [rq1, H_k] giving repeat-blocks pattern matching CPU: iq1 = iv1 % H_k
        स्वचालित perm_5d = ov::op::v0::Constant::create(ov::element::i64, {5}, std::vector<int64_t>{0, 2, 1, 3, 4});
        स्वचालित q_transposed = std::make_shared<ov::op::v1::Transpose>(q_bcast, perm_5d);
        स्वचालित k_transposed = std::make_shared<ov::op::v1::Transpose>(k_bcast, perm_5d);

        स्वचालित new_shape = ov::op::v0::Constant::create(ov::element::i64, {4}, std::vector<int64_t>{B, H_v, T, S_v});
        q_bh = std::make_shared<ov::op::v1::Reshape>(q_transposed, new_shape, असत्य);
        k_bh = std::make_shared<ov::op::v1::Reshape>(k_transposed, new_shape, असत्य);
    }

    // Merge batch and head dims: [B*H_v, T, S_v]
    स्वचालित merge_bh = [&](ध्रुव ov::Output<ov::Node> & x, int64_t चरमः_dim) {
        स्वचालित shape = ov::op::v0::Constant::create(ov::element::i64, {3}, std::vector<int64_t>{B * H_v, T, चरमः_dim});
        प्रत्यागम std::make_shared<ov::op::v1::Reshape>(x, shape, असत्य);
    };

    स्वचालित q_m = merge_bh(q_bh, S_v);           // [B*H_v, T, S_v]
    स्वचालित k_m = merge_bh(k_bh, S_v);           // [B*H_v, T, S_v]
    स्वचालित v_m = merge_bh(v_t, S_v);            // [B*H_v, T, S_v]
    स्वचालित g_m = merge_bh(g_t, kda ? S_v : 1);  // [B*H_v, T, 1 or S_v]
    स्वचालित beta_m = merge_bh(beta_t, 1);        // [B*H_v, T, 1]

    // State: [B, H_v, S_v, S_v] -> [B*H_v, S_v, S_v]
    स्वचालित state_shape = ov::op::v0::Constant::create(ov::element::i64, {3}, std::vector<int64_t>{B * H_v, S_v, S_v});
    स्वचालित state_m = std::make_shared<ov::op::v1::Reshape>(state, state_shape, असत्य);

    स्वचालित scale_const = ov::op::v0::Constant::create(ov::element::f32, {}, std::vector<भिन्न>{scale});

    // --- Build Loop body ---
    // Body parameters (no iteration counter needed, use -1 in special ports)
    स्वचालित body_state = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic());
    स्वचालित body_q = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic());
    स्वचालित body_k = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic());
    स्वचालित body_v = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic());
    स्वचालित body_g = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic());
    स्वचालित body_beta = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic());
    स्वचालित body_iter = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::Shape{1});

    // Condition output (always true - we rely on trip_count for termination)
    स्वचालित body_cond_बहिर्गमनम् = ov::op::v0::Constant::create(ov::element::boolean, ov::Shape{1}, std::vector<तार्किक>{सत्य});

    // Gather current token from invariant inputs using iteration counter
    स्वचालित q_t_cur = std::make_shared<ov::op::v8::Gather>(body_q, body_iter, axis_1);     // [B*H_v, 1, S_v]
    स्वचालित k_t_cur = std::make_shared<ov::op::v8::Gather>(body_k, body_iter, axis_1);     // [B*H_v, 1, S_v]
    स्वचालित v_t_cur = std::make_shared<ov::op::v8::Gather>(body_v, body_iter, axis_1);     // [B*H_v, 1, S_v]
    स्वचालित g_t_cur = std::make_shared<ov::op::v8::Gather>(body_g, body_iter, axis_1);     // [B*H_v, 1, 1 or S_v]
    स्वचालित b_t_cur = std::make_shared<ov::op::v8::Gather>(body_beta, body_iter, axis_1);  // [B*H_v, 1, 1]

    // Squeeze token dim
    स्वचालित q_cur = std::make_shared<ov::op::v0::Squeeze>(q_t_cur, axis_1);  // [B*H_v, S_v]
    स्वचालित k_cur = std::make_shared<ov::op::v0::Squeeze>(k_t_cur, axis_1);  // [B*H_v, S_v]
    स्वचालित v_cur = std::make_shared<ov::op::v0::Squeeze>(v_t_cur, axis_1);  // [B*H_v, S_v]
    स्वचालित g_cur = std::make_shared<ov::op::v0::Squeeze>(g_t_cur, axis_1);  // [B*H_v, 1 or S_v]
    स्वचालित b_cur = std::make_shared<ov::op::v0::Squeeze>(b_t_cur, axis_1);  // [B*H_v, 1]

    // Step 1: Apply decay gate to state
    स्वचालित exp_g = std::make_shared<ov::op::v0::Exp>(g_cur);                                // [B*H_v, 1 or S_v]
    स्वचालित exp_g_unsq = std::make_shared<ov::op::v0::Unsqueeze>(exp_g, axis_1);             // [B*H_v, 1, 1 or S_v]
    स्वचालित state_decayed = std::make_shared<ov::op::v1::Multiply>(body_state, exp_g_unsq);  // [B*H_v, S_v, S_v]

    // Step 2: delta = (v - S @ k) * beta
    स्वचालित k_col = std::make_shared<ov::op::v0::Unsqueeze>(k_cur, axis_2);                 // [B*H_v, S_v, 1]
    स्वचालित sk = std::make_shared<ov::op::v0::MatMul>(state_decayed, k_col, असत्य, असत्य);  // [B*H_v, S_v, 1]
    स्वचालित sk_sq = std::make_shared<ov::op::v0::Squeeze>(sk, axis_2);                      // [B*H_v, S_v]
    स्वचालित v_minus_sk = std::make_shared<ov::op::v1::व्यवकलय>(v_cur, sk_sq);              // [B*H_v, S_v]
    स्वचालित delta = std::make_shared<ov::op::v1::Multiply>(v_minus_sk, b_cur);              // [B*H_v, S_v]

    // Step 3: state += outer(delta, k)
    स्वचालित delta_col = std::make_shared<ov::op::v0::Unsqueeze>(delta, axis_2);                 // [B*H_v, S_v, 1]
    स्वचालित k_row = std::make_shared<ov::op::v0::Unsqueeze>(k_cur, axis_1);                     // [B*H_v, 1, S_v]
    स्वचालित outer_prod = std::make_shared<ov::op::v0::MatMul>(delta_col, k_row, असत्य, असत्य);  // [B*H_v, S_v, S_v]
    स्वचालित state_updated = std::make_shared<ov::op::v1::सङ्कलय>(state_decayed, outer_prod);       // [B*H_v, S_v, S_v]

    // Step 4: attn_out = S @ q * scale
    स्वचालित q_col = std::make_shared<ov::op::v0::Unsqueeze>(q_cur, axis_2);                 // [B*H_v, S_v, 1]
    स्वचालित sq = std::make_shared<ov::op::v0::MatMul>(state_updated, q_col, असत्य, असत्य);  // [B*H_v, S_v, 1]
    स्वचालित sq_squeezed = std::make_shared<ov::op::v0::Squeeze>(sq, axis_2);                // [B*H_v, S_v]
    स्वचालित attn_बहिर्गमनम् = std::make_shared<ov::op::v1::Multiply>(sq_squeezed, scale_const);    // [B*H_v, S_v]

    // Unsqueeze attn_out to [B*H_v, 1, S_v] for scan output concatenation
    स्वचालित attn_बहिर्गमनम्_unsq = std::make_shared<ov::op::v0::Unsqueeze>(attn_बहिर्गमनम्, axis_1);  // [B*H_v, 1, S_v]

    // --- Assemble Loop ---
    // Body: results = [condition, state_updated, attn_out_unsq]
    स्वचालित body = std::make_shared<ov::Model>(
        ov::Output_सदिशम्{body_cond_बहिर्गमनम्, state_updated, attn_बहिर्गमनम्_unsq},
        ov::Parameter_सदिशम्{body_iter, body_state, body_q, body_k, body_v, body_g, body_beta});

    स्वचालित trip_गणनम् = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{T});
    स्वचालित exec_cond = ov::op::v0::Constant::create(ov::element::boolean, ov::Shape{1}, std::vector<तार्किक>{सत्य});

    स्वचालित loop = std::make_shared<ov::op::v5::Loop>(trip_गणनम्, exec_cond);
    loop->समुच्चयः_function(body);
    loop->समुच्चयः_special_body_ports(ov::op::v5::Loop::SpecialBodyPorts{0, 0});

    // Carried state: feeds back from body output 1 to body_state param
    loop->समुच्चयः_merged_निवेशः(body_state, state_m, state_updated);
    // Invariant inputs: passed through unchanged each iteration
    loop->समुच्चयः_invariant_निवेशः(body_q, q_m);
    loop->समुच्चयः_invariant_निवेशः(body_k, k_m);
    loop->समुच्चयः_invariant_निवेशः(body_v, v_m);
    loop->समुच्चयः_invariant_निवेशः(body_g, g_m);
    loop->समुच्चयः_invariant_निवेशः(body_beta, beta_m);

    // Loop outputs:
    // 1) Final state (last iteration value of state_updated)
    स्वचालित final_state_बहिर्गमनम् = loop->get_iter_मानम्(state_updated, -1);  // [B*H_v, S_v, S_v]
    // 2) Concatenated attention outputs across all iterations along axis 1
    स्वचालित attn_concat_बहिर्गमनम् = loop->get_concatenated_slices(attn_बहिर्गमनम्_unsq, 0, 1, 1, -1, 1);  // [B*H_v, T, S_v]

    // --- Pack outputs to match ggml layout ---
    // ggml output ne = {S_v*H, T*B + S_v*B, 1, 1} -> OV [1, 1, T*B+S_v*B, S_v*H_v]
    // attn: [B, T, H_v, S_v] row-major, state: [B, H_v, S_v, S_v] row-major

    // attn: [B*H_v, T, S_v] -> [B, H_v, T, S_v] -> transpose to [B, T, H_v, S_v] -> flatten
    स्वचालित attn_4d_shape = ov::op::v0::Constant::create(ov::element::i64, {4}, std::vector<int64_t>{B, H_v, T, S_v});
    स्वचालित attn_4d = std::make_shared<ov::op::v1::Reshape>(attn_concat_बहिर्गमनम्, attn_4d_shape, असत्य);
    स्वचालित attn_perm = std::make_shared<ov::op::v1::Transpose>(attn_4d, perm_0213);  // [B, T, H_v, S_v]

    स्वचालित flat_shape_1d = ov::op::v0::Constant::create(ov::element::i64, {1}, std::vector<int64_t>{-1});
    स्वचालित attn_1d = std::make_shared<ov::op::v1::Reshape>(attn_perm, flat_shape_1d, असत्य);

    // state: [B*H_v, S_v, S_v] -> [B, H_v, S_v, S_v] -> flatten
    स्वचालित state_4d_shape = ov::op::v0::Constant::create(ov::element::i64, {4}, std::vector<int64_t>{B, H_v, S_v, S_v});
    स्वचालित state_4d = std::make_shared<ov::op::v1::Reshape>(final_state_बहिर्गमनम्, state_4d_shape, असत्य);
    यदि (context.get_output_नामानि().size() == 2) {
        // Match the fused translator's direct attention/state contract.
        प्रत्यागम rename_outputs_with_suffix({attn_perm, state_4d}, context.get_name());
    }

    स्वचालित state_1d = std::make_shared<ov::op::v1::Reshape>(state_4d, flat_shape_1d, असत्य);

    // Concat [attn | state] and reshape to final output
    स्वचालित packed = std::make_shared<ov::op::v0::Concat>(ov::Output_सदिशम्{attn_1d, state_1d}, 0);
    स्वचालित बहिर्गमनम्_shape =
        ov::op::v0::Constant::create(ov::element::i64, {4}, std::vector<int64_t>{1, 1, T * B + S_v * B, S_v * H_v});
    स्वचालित res = std::make_shared<ov::op::v1::Reshape>(packed, बहिर्गमनम्_shape, असत्य);

    प्रत्यागम rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
