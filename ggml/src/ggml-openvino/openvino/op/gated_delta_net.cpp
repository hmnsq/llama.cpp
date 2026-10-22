#అన్తర్భావ "gated_delta_net.hpp"

#అన్తర్భావ "../node_context.h"
#అన్తర్భావ "../op_table.h"
#అన్తర్భావ "../utils.h"
#అన్తర్భావ "ggml-openvino/ggml-openvino-extra.h"

#అన్తర్భావ <cmath>
#అన్తర్భావ <cstdint>
#అన్తర్భావ <memory>
#అన్తర్భావ <numeric>
#అన్తర్భావ <openvino/op/add.hpp>
#అన్తర్భావ <openvino/op/broadcast.hpp>
#అన్తర్భావ <openvino/op/concat.hpp>
#అన్తర్భావ <openvino/op/constant.hpp>
#అన్తర్భావ <openvino/op/convert.hpp>
#అన్తర్భావ <openvino/op/divide.hpp>
#అన్తర్భావ <openvino/op/exp.hpp>
#అన్తర్భావ <openvino/op/gather.hpp>
#అన్తర్భావ <openvino/op/less.hpp>
#అన్తర్భావ <openvino/op/loop.hpp>
#అన్తర్భావ <openvino/op/matmul.hpp>
#అన్తర్భావ <openvino/op/multiply.hpp>
#అన్తర్భావ <openvino/op/reduce_mean.hpp>
#అన్తర్భావ <openvino/op/reshape.hpp>
#అన్తర్భావ <openvino/op/squeeze.hpp>
#అన్తర్భావ <openvino/op/sqrt.hpp>
#అన్తర్భావ <openvino/op/subtract.hpp>
#అన్తర్భావ <openvino/op/tile.hpp>
#అన్తర్భావ <openvino/op/transpose.hpp>
#అన్తర్భావ <openvino/op/unsqueeze.hpp>
#అన్తర్భావ <vector>

నామాకాశ ov {
నామాకాశ frontend {
నామాకాశ ggml {
నామాకాశ op {

స్థిర Output_సదిశ translate_gated_delta_net_ref(స్థిరరాశి కణుపు_Context & context);

స్థిర తార్కిక సరిపోలిక_gdn_l2_norm(స్థిరరాశి Output<Node> & normalized, Output<Node> & ఆదానం, దశాంశం & eps) {
    // Match the RMSNorm decomposition emitted by translate_rms_norm, followed by GGML SCALE.
    స్థిరరాశి స్వయంచాలక scale = ov::as_type_ptr<ov::op::v1::Multiply>(normalized.get_కణుపు_shared_ptr());
    ఒకవేళ (!scale) {
        తిరిగి అబద్ధం;
    }
    స్థిరరాశి స్వయంచాలక గుణకం = ov::as_type_ptr<ov::op::v0::Constant>(scale->get_ఆదానం_కణుపు_shared_ptr(1));
    స్థిరరాశి స్వయంచాలక rms = ov::as_type_ptr<ov::op::v1::Multiply>(scale->get_ఆదానం_కణుపు_shared_ptr(0));
    ఒకవేళ (!గుణకం || ov::shape_కొలత(గుణకం->get_shape()) != 1 || !rms) {
        తిరిగి అబద్ధం;
    }
    స్థిరరాశి స్వయంచాలక x = rms->ఆదానం_విలువ(0);
    స్థిరరాశి స్వయంచాలక & shape = x.get_partial_shape();
    ఒకవేళ (x.get_element_type() != ov::element::f32 || shape.rank() != 4 || shape[3].is_dynamic() ||
        shape[3].get_నిడివి() <= 0 || normalized.get_partial_shape() != shape) {
        తిరిగి అబద్ధం;
    }
    స్థిరరాశి దశాంశం dim = static_cast<దశాంశం>(shape[3].get_నిడివి());
    ఒకవేళ (గుణకం->cast_సదిశ<దశాంశం>()[0] != 1.0f / std::వర్గమూలం(dim)) {
        తిరిగి అబద్ధం;
    }
    స్థిరరాశి స్వయంచాలక reciprocal = ov::as_type_ptr<ov::op::v1::Divide>(rms->get_ఆదానం_కణుపు_shared_ptr(1));
    ఒకవేళ (!reciprocal) {
        తిరిగి అబద్ధం;
    }
    స్థిరరాశి స్వయంచాలక one = ov::as_type_ptr<ov::op::v0::Constant>(reciprocal->get_ఆదానం_కణుపు_shared_ptr(0));
    స్థిరరాశి స్వయంచాలక మూలం = ov::as_type_ptr<ov::op::v0::Sqrt>(reciprocal->get_ఆదానం_కణుపు_shared_ptr(1));
    ఒకవేళ (!one || ov::shape_కొలత(one->get_shape()) != 1 || one->cast_సదిశ<దశాంశం>()[0] != 1.0f || !మూలం) {
        తిరిగి అబద్ధం;
    }
    స్థిరరాశి స్వయంచాలక కూడు_2 = ov::as_type_ptr<ov::op::v1::కూడు>(మూలం->get_ఆదానం_కణుపు_shared_ptr(0));
    ఒకవేళ (!కూడు_2) {
        తిరిగి అబద్ధం;
    }
    స్థిరరాశి స్వయంచాలక సగటు_2 = ov::as_type_ptr<ov::op::v1::Reduce_సగటు>(కూడు_2->get_ఆదానం_కణుపు_shared_ptr(0));
    స్థిరరాశి స్వయంచాలక rms_eps = ov::as_type_ptr<ov::op::v0::Constant>(కూడు_2->get_ఆదానం_కణుపు_shared_ptr(1));
    ఒకవేళ (!సగటు_2 || !సగటు_2->get_keep_dims() || !rms_eps || ov::shape_కొలత(rms_eps->get_shape()) != 1) {
        తిరిగి అబద్ధం;
    }
    స్థిరరాశి స్వయంచాలక axes = ov::as_type_ptr<ov::op::v0::Constant>(సగటు_2->get_ఆదానం_కణుపు_shared_ptr(1));
    స్థిరరాశి స్వయంచాలక వర్గం = ov::as_type_ptr<ov::op::v1::Multiply>(సగటు_2->get_ఆదానం_కణుపు_shared_ptr(0));
    ఒకవేళ (!axes || axes->cast_సదిశ<int64_t>() != std::vector<int64_t>{-1} || !వర్గం ||
        వర్గం->ఆదానం_విలువ(0) != x || వర్గం->ఆదానం_విలువ(1) != x) {
        తిరిగి అబద్ధం;
    }
    // RMSNorm(x, rms_eps) / sqrt(D) = x / sqrt(sum(x*x) + D*rms_eps).
    eps = dim * rms_eps->cast_సదిశ<దశాంశం>()[0];
    ఒకవేళ (!std::isfinite(eps) || eps <= 0.0f) {
        తిరిగి అబద్ధం;
    }
    ఆదానం = x;
    తిరిగి నిజం;
}

Output_సదిశ translate_gated_delta_net(స్థిరరాశి కణుపు_Context & context) {
    స్వయంచాలక v_shape = context.get_ఆదానం_shape(2).గమ్యం_shape();  // [B, T, H_v, S_v]
    స్వయంచాలక q_shape = context.get_ఆదానం_shape(0).గమ్యం_shape();  // [B, T, H_k, S_k]

    // Fused GatedDeltaNet op only supports scalar gate (kda=0).
    // Fall back to reference implementation for per-key-dimension gating.
    // if (kda) {
    //     return translate_gated_delta_net_ref(context);
    // }

    // const int64_t B = v_shape[0];
    // const int64_t T = v_shape[1];
    స్థిరరాశి int64_t H_v = v_shape[2];
    స్థిరరాశి int64_t S_v = v_shape[3];
    స్థిరరాశి int64_t H_k = q_shape[2];
    // const int64_t S_k = q_shape[3];

    స్వయంచాలక q = context.get_ఆదానం(0);
    స్వయంచాలక k = context.get_ఆదానం(1);
    స్వయంచాలక v = process_view_ఆదానం(context, 2, H_v * S_v, 3);
    స్వయంచాలక g = context.get_ఆదానం(3);
    స్వయంచాలక beta = context.get_ఆదానం(4);
    స్వయంచాలక state = context.get_ఆదానం(5);

    Output<Node> raw_q, raw_k;
    దశాంశం q_eps = 1e-6f, k_eps = 1e-6f;
    స్థిరరాశి తార్కిక fuse_qk_l2norm = ggml_openvino_is_gpu() &&
        సరిపోలిక_gdn_l2_norm(q, raw_q, q_eps) && సరిపోలిక_gdn_l2_norm(k, raw_k, k_eps);
    ఒకవేళ (fuse_qk_l2norm) {
        // Keep head tiling below; GDN applies normalization per head and keeps its attention scale.
        q = raw_q;
        k = raw_k;
    }

    // ggml maps GQA heads in tiled order, while OV GDN maps repeated heads in grouped order.
    ఒకవేళ (H_v != H_k) {
        స్థిరరాశి int64_t repeat = H_v / H_k;
        స్వయంచాలక repeats = ov::op::v0::Constant::create(ov::element::i64, {4}, std::vector<int64_t>{1, 1, repeat, 1});
        q = std::make_shared<ov::op::v0::Tile>(q, repeats);
        k = std::make_shared<ov::op::v0::Tile>(k, repeats);
    }

    ఒకవేళ (context.get_view_ఆదానం_కొలత(2)) {
        // Same as l2_norm case 1
        v = std::make_shared<ov::op::v0::Squeeze>(v, ov::op::v0::Constant::create(ov::element::i64, {1}, {0}));
        స్వయంచాలక v_shape = context.get_ఆదానం_shape(2).గమ్యం_shape();
        std::vector<int64_t> reshape_నమూనా = {0, 0, (int64_t) v_shape[2], (int64_t) v_shape[3]};
        v = std::make_shared<ov::op::v1::Reshape>(
            v, ov::op::v0::Constant::create(ov::element::i64, {4}, reshape_నమూనా), నిజం);
    }

    // ggml state layout (OV notation): [B, H_v, value_dim, key_dim]
    // GatedDeltaNet op expects: [B, H_v, key_dim, value_dim]
    స్వయంచాలక state_perm = ov::op::v0::Constant::create(ov::element::i64, {4}, std::vector<int64_t>{0, 1, 3, 2});
    state = std::make_shared<ov::op::v1::Transpose>(state, state_perm);

    g = std::make_shared<ov::op::v0::Squeeze>(g, ov::op::v0::Constant::create(ov::element::i64, {1}, {3}));
    beta = std::make_shared<ov::op::v0::Squeeze>(beta, ov::op::v0::Constant::create(ov::element::i64, {1}, {3}));

    ఒకవేళ (context.has_ఆదానం("chunk_valid_len")) {
        // The last prefill chunk is padded with fabricated tokens. The recurrence is
        //   S_t = S_{t-1} * exp(g_t) + k_t (x) ((v_t - S_{t-1}^T k_t) * beta_t)
        // so forcing g = 0 and beta = 0 makes a padded step an exact identity and keeps the final
        // state equal to the state after the last real token. Attention output at those positions
        // is garbage but never read.
        స్థిరరాశి స్వయంచాలక & g_ps = g.get_partial_shape();
        FRONT_చివర_OP_CONVERSION_CHECK(g_ps.rank().is_static() && g_ps.rank().get_నిడివి() == 3 && g_ps[1].is_static(),
                                      "GATED_DELTA_NET pad masking requires a static token dimension");
        స్థిరరాశి int64_t n_టోకెన్లు = g_ps[1].get_నిడివి();
        std::vector<int64_t> positions(n_టోకెన్లు);
        std::iota(positions.begin(), positions.end(), 0);
        స్వయంచాలక valid = std::make_shared<ov::op::v1::Less>(
            ov::op::v0::Constant::create(ov::element::i64, {(size_t) n_టోకెన్లు}, positions),
            context.get_ఆదానం("chunk_valid_len"));
        స్వయంచాలక ముసుగు = std::make_shared<ov::op::v0::Unsqueeze>(
            std::make_shared<ov::op::v0::Convert>(valid, g.get_element_type()),
            ov::op::v0::Constant::create(ov::element::i64, {2}, std::vector<int64_t>{0, 2}));
        g = std::make_shared<ov::op::v1::Multiply>(g, ముసుగు);
        beta = std::make_shared<ov::op::v1::Multiply>(beta, ముసుగు);
    }

    // std::cout << "GatedDeltaNet input shapes: q=" << q.get_partial_shape() << ", k=" << k.get_partial_shape()
    //           << ", v=" << v.get_partial_shape() << ", g=" << g.get_partial_shape()
    //           << ", beta=" << beta.get_partial_shape() << ", state=" << state.get_partial_shape() << std::endl;

    స్వయంచాలక gdn = std::make_shared<ov::op::internal::GatedDeltaNet>(q, k, v, state, g, beta, fuse_qk_l2norm, q_eps, k_eps);
    స్వయంచాలక attn_4d = gdn->output(0);
    స్వయంచాలక state_4d = gdn->output(1);  // [B, H_v, key_dim, value_dim]

    // std::cout << "GatedDeltaNet output shapes: attn=" << gdn->output(0).get_partial_shape()
    //           << ", new_state=" << gdn->output(1).get_partial_shape() << std::endl;

    // Transpose output state back to ggml layout [B, H_v, value_dim, key_dim]
    స్వయంచాలక state_transposed = std::make_shared<ov::op::v1::Transpose>(state_4d, state_perm);
    ఒకవేళ (context.get_output_పేర్లు().size() == 2) {
        // The canonical graph consumes the packed GGML result only through separate attention
        // and state VIEWs. Publish the native outputs under those VIEW names to avoid
        // flatten -> concat -> reshape -> slice -> reshape chains. This also works for B > 1.
        తిరిగి rename_outputs_with_suffix({attn_4d, state_transposed}, context.get_name());
    }

    స్వయంచాలక flat_shape_1d = ov::op::v0::Constant::create(ov::element::i64, {1}, {-1});
    స్వయంచాలక attn = std::make_shared<ov::op::v1::Reshape>(attn_4d, flat_shape_1d, అబద్ధం);
    స్వయంచాలక new_state = std::make_shared<ov::op::v1::Reshape>(state_transposed, flat_shape_1d, అబద్ధం);
    స్వయంచాలక packed = std::make_shared<ov::op::v0::Concat>(ov::Output_సదిశ{attn, new_state}, 0);
    స్వయంచాలక బహిర్గతం_shape = ov::op::v0::Constant::create(ov::element::i64, {4},
                                                  std::vector<int64_t>{1, 1, -1 /*T * B + S_v * B*/, S_v * H_v});
    స్వయంచాలక res = std::make_shared<ov::op::v1::Reshape>(packed, బహిర్గతం_shape, అబద్ధం);

    తిరిగి rename_outputs_with_suffix({res}, context.get_name());
}

స్థిర Output_సదిశ translate_gated_delta_net_ref(స్థిరరాశి కణుపు_Context & context) {
    అంకె_inputs_check(context, 6, 6);

    // Inputs (OV shapes are reversed from ggml):
    // ggml: q[S_k, H_k, T, B], k[S_k, H_k, T, B], v[S_v, H_v, T, B]
    // OV:   q[B, T, H_k, S_k], k[B, T, H_k, S_k], v[B, T, H_v, S_v]
    // ggml: g[1 or S_v, H_v, T, B], beta[1, H_v, T, B]
    // OV:   g[B, T, H_v, 1 or S_v], beta[B, T, H_v, 1]
    // ggml: state[S_v, S_v, H_v, B]
    // OV:   state[B, H_v, S_v, S_v]
    స్వయంచాలక q = process_view_ఆదానం_new(context, 0);
    స్వయంచాలక k = process_view_ఆదానం_new(context, 1);
    స్వయంచాలక v = process_view_ఆదానం_new(context, 2);
    స్వయంచాలక g = process_view_ఆదానం_new(context, 3);
    స్వయంచాలక beta = process_view_ఆదానం_new(context, 4);
    స్వయంచాలక state = process_view_ఆదానం_new(context, 5);

    స్వయంచాలక v_shape = context.get_ఆదానం_shape(2).గమ్యం_shape();  // [B, T, H_v, S_v]
    స్వయంచాలక q_shape = context.get_ఆదానం_shape(0).గమ్యం_shape();  // [B, T, H_k, S_k]
    స్వయంచాలక g_shape = context.get_ఆదానం_shape(3).గమ్యం_shape();  // [B, T, H_v, 1 or S_v]

    స్థిరరాశి int64_t B = v_shape[0];
    స్థిరరాశి int64_t T = v_shape[1];
    స్థిరరాశి int64_t H_v = v_shape[2];
    స్థిరరాశి int64_t S_v = v_shape[3];
    స్థిరరాశి int64_t H_k = q_shape[2];
    స్థిరరాశి తార్కిక kda = (g_shape[3] == (size_t) S_v);

    స్థిరరాశి int64_t rq1 = H_v / H_k;  // head repeat factor
    స్థిరరాశి దశాంశం scale = 1.0f / std::వర్గమూలం((దశాంశం) S_v);

    స్వయంచాలక axis_1 = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
    స్వయంచాలక axis_2 = ov::op::v0::Constant::create(ov::element::i64, {1}, {2});

    // Transpose inputs from [B, T, H, S] to [B, H, T, S] for easier per-head processing
    స్వయంచాలక perm_0213 = ov::op::v0::Constant::create(ov::element::i64, {4}, std::vector<int64_t>{0, 2, 1, 3});
    స్వయంచాలక q_t = std::make_shared<ov::op::v1::Transpose>(q, perm_0213);        // [B, H_k, T, S_k]
    స్వయంచాలక k_t = std::make_shared<ov::op::v1::Transpose>(k, perm_0213);        // [B, H_k, T, S_k]
    స్వయంచాలక v_t = std::make_shared<ov::op::v1::Transpose>(v, perm_0213);        // [B, H_v, T, S_v]
    స్వయంచాలక g_t = std::make_shared<ov::op::v1::Transpose>(g, perm_0213);        // [B, H_v, T, 1 or S_v]
    స్వయంచాలక beta_t = std::make_shared<ov::op::v1::Transpose>(beta, perm_0213);  // [B, H_v, T, 1]

    // Broadcast Q, K heads to match V heads if GQA is used (H_v > H_k)
    ov::Output<ov::Node> q_bh = q_t;
    ov::Output<ov::Node> k_bh = k_t;
    ఒకవేళ (rq1 > 1) {
        స్వయంచాలక q_unsq = std::make_shared<ov::op::v0::Unsqueeze>(q_t, axis_2);  // [B, H_k, 1, T, S]
        స్వయంచాలక k_unsq = std::make_shared<ov::op::v0::Unsqueeze>(k_t, axis_2);  // [B, H_k, 1, T, S]

        స్వయంచాలక bcast_shape = ov::op::v0::Constant::create(ov::element::i64, {5}, std::vector<int64_t>{1, 1, rq1, 1, 1});
        స్వయంచాలక q_bcast =
            std::make_shared<ov::op::v3::ప్రసారం>(q_unsq, bcast_shape, ov::op::ప్రసారం_Type::BIDIRECTIONAL);
        స్వయంచాలక k_bcast =
            std::make_shared<ov::op::v3::ప్రసారం>(k_unsq, bcast_shape, ov::op::ప్రసారం_Type::BIDIRECTIONAL);

        // Transpose [B, H_k, rq1, T, S] -> [B, rq1, H_k, T, S] so that reshape merges
        // as [rq1, H_k] giving repeat-blocks pattern matching CPU: iq1 = iv1 % H_k
        స్వయంచాలక perm_5d = ov::op::v0::Constant::create(ov::element::i64, {5}, std::vector<int64_t>{0, 2, 1, 3, 4});
        స్వయంచాలక q_transposed = std::make_shared<ov::op::v1::Transpose>(q_bcast, perm_5d);
        స్వయంచాలక k_transposed = std::make_shared<ov::op::v1::Transpose>(k_bcast, perm_5d);

        స్వయంచాలక new_shape = ov::op::v0::Constant::create(ov::element::i64, {4}, std::vector<int64_t>{B, H_v, T, S_v});
        q_bh = std::make_shared<ov::op::v1::Reshape>(q_transposed, new_shape, అబద్ధం);
        k_bh = std::make_shared<ov::op::v1::Reshape>(k_transposed, new_shape, అబద్ధం);
    }

    // Merge batch and head dims: [B*H_v, T, S_v]
    స్వయంచాలక విలీనం_bh = [&](స్థిరరాశి ov::Output<ov::Node> & x, int64_t చివరి_dim) {
        స్వయంచాలక shape = ov::op::v0::Constant::create(ov::element::i64, {3}, std::vector<int64_t>{B * H_v, T, చివరి_dim});
        తిరిగి std::make_shared<ov::op::v1::Reshape>(x, shape, అబద్ధం);
    };

    స్వయంచాలక q_m = విలీనం_bh(q_bh, S_v);           // [B*H_v, T, S_v]
    స్వయంచాలక k_m = విలీనం_bh(k_bh, S_v);           // [B*H_v, T, S_v]
    స్వయంచాలక v_m = విలీనం_bh(v_t, S_v);            // [B*H_v, T, S_v]
    స్వయంచాలక g_m = విలీనం_bh(g_t, kda ? S_v : 1);  // [B*H_v, T, 1 or S_v]
    స్వయంచాలక beta_m = విలీనం_bh(beta_t, 1);        // [B*H_v, T, 1]

    // State: [B, H_v, S_v, S_v] -> [B*H_v, S_v, S_v]
    స్వయంచాలక state_shape = ov::op::v0::Constant::create(ov::element::i64, {3}, std::vector<int64_t>{B * H_v, S_v, S_v});
    స్వయంచాలక state_m = std::make_shared<ov::op::v1::Reshape>(state, state_shape, అబద్ధం);

    స్వయంచాలక scale_const = ov::op::v0::Constant::create(ov::element::f32, {}, std::vector<దశాంశం>{scale});

    // --- Build Loop body ---
    // Body parameters (no iteration counter needed, use -1 in special ports)
    స్వయంచాలక body_state = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic());
    స్వయంచాలక body_q = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic());
    స్వయంచాలక body_k = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic());
    స్వయంచాలక body_v = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic());
    స్వయంచాలక body_g = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic());
    స్వయంచాలక body_beta = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic());
    స్వయంచాలక body_iter = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::Shape{1});

    // Condition output (always true - we rely on trip_count for termination)
    స్వయంచాలక body_cond_బహిర్గతం = ov::op::v0::Constant::create(ov::element::boolean, ov::Shape{1}, std::vector<తార్కిక>{నిజం});

    // Gather current token from invariant inputs using iteration counter
    స్వయంచాలక q_t_cur = std::make_shared<ov::op::v8::Gather>(body_q, body_iter, axis_1);     // [B*H_v, 1, S_v]
    స్వయంచాలక k_t_cur = std::make_shared<ov::op::v8::Gather>(body_k, body_iter, axis_1);     // [B*H_v, 1, S_v]
    స్వయంచాలక v_t_cur = std::make_shared<ov::op::v8::Gather>(body_v, body_iter, axis_1);     // [B*H_v, 1, S_v]
    స్వయంచాలక g_t_cur = std::make_shared<ov::op::v8::Gather>(body_g, body_iter, axis_1);     // [B*H_v, 1, 1 or S_v]
    స్వయంచాలక b_t_cur = std::make_shared<ov::op::v8::Gather>(body_beta, body_iter, axis_1);  // [B*H_v, 1, 1]

    // Squeeze token dim
    స్వయంచాలక q_cur = std::make_shared<ov::op::v0::Squeeze>(q_t_cur, axis_1);  // [B*H_v, S_v]
    స్వయంచాలక k_cur = std::make_shared<ov::op::v0::Squeeze>(k_t_cur, axis_1);  // [B*H_v, S_v]
    స్వయంచాలక v_cur = std::make_shared<ov::op::v0::Squeeze>(v_t_cur, axis_1);  // [B*H_v, S_v]
    స్వయంచాలక g_cur = std::make_shared<ov::op::v0::Squeeze>(g_t_cur, axis_1);  // [B*H_v, 1 or S_v]
    స్వయంచాలక b_cur = std::make_shared<ov::op::v0::Squeeze>(b_t_cur, axis_1);  // [B*H_v, 1]

    // Step 1: Apply decay gate to state
    స్వయంచాలక exp_g = std::make_shared<ov::op::v0::Exp>(g_cur);                                // [B*H_v, 1 or S_v]
    స్వయంచాలక exp_g_unsq = std::make_shared<ov::op::v0::Unsqueeze>(exp_g, axis_1);             // [B*H_v, 1, 1 or S_v]
    స్వయంచాలక state_decayed = std::make_shared<ov::op::v1::Multiply>(body_state, exp_g_unsq);  // [B*H_v, S_v, S_v]

    // Step 2: delta = (v - S @ k) * beta
    స్వయంచాలక k_నిలువు = std::make_shared<ov::op::v0::Unsqueeze>(k_cur, axis_2);                 // [B*H_v, S_v, 1]
    స్వయంచాలక sk = std::make_shared<ov::op::v0::MatMul>(state_decayed, k_నిలువు, అబద్ధం, అబద్ధం);  // [B*H_v, S_v, 1]
    స్వయంచాలక sk_sq = std::make_shared<ov::op::v0::Squeeze>(sk, axis_2);                      // [B*H_v, S_v]
    స్వయంచాలక v_minus_sk = std::make_shared<ov::op::v1::వ్యవకలనం>(v_cur, sk_sq);              // [B*H_v, S_v]
    స్వయంచాలక delta = std::make_shared<ov::op::v1::Multiply>(v_minus_sk, b_cur);              // [B*H_v, S_v]

    // Step 3: state += outer(delta, k)
    స్వయంచాలక delta_నిలువు = std::make_shared<ov::op::v0::Unsqueeze>(delta, axis_2);                 // [B*H_v, S_v, 1]
    స్వయంచాలక k_అడ్డువరుస = std::make_shared<ov::op::v0::Unsqueeze>(k_cur, axis_1);                     // [B*H_v, 1, S_v]
    స్వయంచాలక outer_prod = std::make_shared<ov::op::v0::MatMul>(delta_నిలువు, k_అడ్డువరుస, అబద్ధం, అబద్ధం);  // [B*H_v, S_v, S_v]
    స్వయంచాలక state_updated = std::make_shared<ov::op::v1::కూడు>(state_decayed, outer_prod);       // [B*H_v, S_v, S_v]

    // Step 4: attn_out = S @ q * scale
    స్వయంచాలక q_నిలువు = std::make_shared<ov::op::v0::Unsqueeze>(q_cur, axis_2);                 // [B*H_v, S_v, 1]
    స్వయంచాలక sq = std::make_shared<ov::op::v0::MatMul>(state_updated, q_నిలువు, అబద్ధం, అబద్ధం);  // [B*H_v, S_v, 1]
    స్వయంచాలక sq_squeezed = std::make_shared<ov::op::v0::Squeeze>(sq, axis_2);                // [B*H_v, S_v]
    స్వయంచాలక attn_బహిర్గతం = std::make_shared<ov::op::v1::Multiply>(sq_squeezed, scale_const);    // [B*H_v, S_v]

    // Unsqueeze attn_out to [B*H_v, 1, S_v] for scan output concatenation
    స్వయంచాలక attn_బహిర్గతం_unsq = std::make_shared<ov::op::v0::Unsqueeze>(attn_బహిర్గతం, axis_1);  // [B*H_v, 1, S_v]

    // --- Assemble Loop ---
    // Body: results = [condition, state_updated, attn_out_unsq]
    స్వయంచాలక body = std::make_shared<ov::Model>(
        ov::Output_సదిశ{body_cond_బహిర్గతం, state_updated, attn_బహిర్గతం_unsq},
        ov::Parameter_సదిశ{body_iter, body_state, body_q, body_k, body_v, body_g, body_beta});

    స్వయంచాలక trip_లెక్క = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{T});
    స్వయంచాలక exec_cond = ov::op::v0::Constant::create(ov::element::boolean, ov::Shape{1}, std::vector<తార్కిక>{నిజం});

    స్వయంచాలక loop = std::make_shared<ov::op::v5::Loop>(trip_లెక్క, exec_cond);
    loop->సమితి_function(body);
    loop->సమితి_special_body_ports(ov::op::v5::Loop::SpecialBodyPorts{0, 0});

    // Carried state: feeds back from body output 1 to body_state param
    loop->సమితి_merged_ఆదానం(body_state, state_m, state_updated);
    // Invariant inputs: passed through unchanged each iteration
    loop->సమితి_invariant_ఆదానం(body_q, q_m);
    loop->సమితి_invariant_ఆదానం(body_k, k_m);
    loop->సమితి_invariant_ఆదానం(body_v, v_m);
    loop->సమితి_invariant_ఆదానం(body_g, g_m);
    loop->సమితి_invariant_ఆదానం(body_beta, beta_m);

    // Loop outputs:
    // 1) Final state (last iteration value of state_updated)
    స్వయంచాలక final_state_బహిర్గతం = loop->get_iter_విలువ(state_updated, -1);  // [B*H_v, S_v, S_v]
    // 2) Concatenated attention outputs across all iterations along axis 1
    స్వయంచాలక attn_concat_బహిర్గతం = loop->get_concatenated_slices(attn_బహిర్గతం_unsq, 0, 1, 1, -1, 1);  // [B*H_v, T, S_v]

    // --- Pack outputs to match ggml layout ---
    // ggml output ne = {S_v*H, T*B + S_v*B, 1, 1} -> OV [1, 1, T*B+S_v*B, S_v*H_v]
    // attn: [B, T, H_v, S_v] row-major, state: [B, H_v, S_v, S_v] row-major

    // attn: [B*H_v, T, S_v] -> [B, H_v, T, S_v] -> transpose to [B, T, H_v, S_v] -> flatten
    స్వయంచాలక attn_4d_shape = ov::op::v0::Constant::create(ov::element::i64, {4}, std::vector<int64_t>{B, H_v, T, S_v});
    స్వయంచాలక attn_4d = std::make_shared<ov::op::v1::Reshape>(attn_concat_బహిర్గతం, attn_4d_shape, అబద్ధం);
    స్వయంచాలక attn_perm = std::make_shared<ov::op::v1::Transpose>(attn_4d, perm_0213);  // [B, T, H_v, S_v]

    స్వయంచాలక flat_shape_1d = ov::op::v0::Constant::create(ov::element::i64, {1}, std::vector<int64_t>{-1});
    స్వయంచాలక attn_1d = std::make_shared<ov::op::v1::Reshape>(attn_perm, flat_shape_1d, అబద్ధం);

    // state: [B*H_v, S_v, S_v] -> [B, H_v, S_v, S_v] -> flatten
    స్వయంచాలక state_4d_shape = ov::op::v0::Constant::create(ov::element::i64, {4}, std::vector<int64_t>{B, H_v, S_v, S_v});
    స్వయంచాలక state_4d = std::make_shared<ov::op::v1::Reshape>(final_state_బహిర్గతం, state_4d_shape, అబద్ధం);
    ఒకవేళ (context.get_output_పేర్లు().size() == 2) {
        // Match the fused translator's direct attention/state contract.
        తిరిగి rename_outputs_with_suffix({attn_perm, state_4d}, context.get_name());
    }

    స్వయంచాలక state_1d = std::make_shared<ov::op::v1::Reshape>(state_4d, flat_shape_1d, అబద్ధం);

    // Concat [attn | state] and reshape to final output
    స్వయంచాలక packed = std::make_shared<ov::op::v0::Concat>(ov::Output_సదిశ{attn_1d, state_1d}, 0);
    స్వయంచాలక బహిర్గతం_shape =
        ov::op::v0::Constant::create(ov::element::i64, {4}, std::vector<int64_t>{1, 1, T * B + S_v * B, S_v * H_v});
    స్వయంచాలక res = std::make_shared<ov::op::v1::Reshape>(packed, బహిర్గతం_shape, అబద్ధం);

    తిరిగి rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
