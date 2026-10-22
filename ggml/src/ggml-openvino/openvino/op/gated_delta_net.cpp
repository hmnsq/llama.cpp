#ضمن "gated_delta_net.hpp"

#ضمن "../node_context.h"
#ضمن "../op_table.h"
#ضمن "../utils.h"
#ضمن "ggml-openvino/ggml-openvino-extra.h"

#ضمن <cmath>
#ضمن <cstdint>
#ضمن <memory>
#ضمن <numeric>
#ضمن <openvino/op/add.hpp>
#ضمن <openvino/op/broadcast.hpp>
#ضمن <openvino/op/concat.hpp>
#ضمن <openvino/op/constant.hpp>
#ضمن <openvino/op/convert.hpp>
#ضمن <openvino/op/divide.hpp>
#ضمن <openvino/op/exp.hpp>
#ضمن <openvino/op/gather.hpp>
#ضمن <openvino/op/less.hpp>
#ضمن <openvino/op/loop.hpp>
#ضمن <openvino/op/matmul.hpp>
#ضمن <openvino/op/multiply.hpp>
#ضمن <openvino/op/reduce_mean.hpp>
#ضمن <openvino/op/reshape.hpp>
#ضمن <openvino/op/squeeze.hpp>
#ضمن <openvino/op/sqrt.hpp>
#ضمن <openvino/op/subtract.hpp>
#ضمن <openvino/op/tile.hpp>
#ضمن <openvino/op/transpose.hpp>
#ضمن <openvino/op/unsqueeze.hpp>
#ضمن <vector>

نطاق ov {
نطاق frontend {
نطاق ggml {
نطاق op {

ساكن Output_ويڪٽر translate_gated_delta_net_ref(ثابت ڳنڍ_Context & context);

ساكن منطقي ميلاپ_gdn_l2_norm(ثابت Output<Node> & normalized, Output<Node> & داخلا_2, عائم & eps) {
    // Match the RMSNorm decomposition emitted by translate_rms_norm, followed by GGML SCALE.
    ثابت خودڪار scale = ov::as_type_ptr<ov::op::v1::Multiply>(normalized.get_ڳنڍ_shared_ptr());
    إذا (!scale) {
        أرجع خطأ;
    }
    ثابت خودڪار ضرب_ڪندڙ = ov::as_type_ptr<ov::op::v0::Constant>(scale->get_داخلا_ڳنڍ_shared_ptr(1));
    ثابت خودڪار rms = ov::as_type_ptr<ov::op::v1::Multiply>(scale->get_داخلا_ڳنڍ_shared_ptr(0));
    إذا (!ضرب_ڪندڙ || ov::shape_سائيز(ضرب_ڪندڙ->get_shape()) != 1 || !rms) {
        أرجع خطأ;
    }
    ثابت خودڪار x = rms->داخلا_قدر(0);
    ثابت خودڪار & shape = x.get_partial_shape();
    إذا (x.get_element_type() != ov::element::f32 || shape.rank() != 4 || shape[3].is_dynamic() ||
        shape[3].get_ڊيگهائي() <= 0 || normalized.get_partial_shape() != shape) {
        أرجع خطأ;
    }
    ثابت عائم dim = static_cast<عائم>(shape[3].get_ڊيگهائي());
    إذا (ضرب_ڪندڙ->cast_ويڪٽر<عائم>()[0] != 1.0f / std::جذر(dim)) {
        أرجع خطأ;
    }
    ثابت خودڪار reciprocal = ov::as_type_ptr<ov::op::v1::Divide>(rms->get_داخلا_ڳنڍ_shared_ptr(1));
    إذا (!reciprocal) {
        أرجع خطأ;
    }
    ثابت خودڪار one = ov::as_type_ptr<ov::op::v0::Constant>(reciprocal->get_داخلا_ڳنڍ_shared_ptr(0));
    ثابت خودڪار پاڙ = ov::as_type_ptr<ov::op::v0::Sqrt>(reciprocal->get_داخلا_ڳنڍ_shared_ptr(1));
    إذا (!one || ov::shape_سائيز(one->get_shape()) != 1 || one->cast_ويڪٽر<عائم>()[0] != 1.0f || !پاڙ) {
        أرجع خطأ;
    }
    ثابت خودڪار جوڙيو_2 = ov::as_type_ptr<ov::op::v1::جوڙيو>(پاڙ->get_داخلا_ڳنڍ_shared_ptr(0));
    إذا (!جوڙيو_2) {
        أرجع خطأ;
    }
    ثابت خودڪار سراسري_2 = ov::as_type_ptr<ov::op::v1::Reduce_سراسري>(جوڙيو_2->get_داخلا_ڳنڍ_shared_ptr(0));
    ثابت خودڪار rms_eps = ov::as_type_ptr<ov::op::v0::Constant>(جوڙيو_2->get_داخلا_ڳنڍ_shared_ptr(1));
    إذا (!سراسري_2 || !سراسري_2->get_keep_dims() || !rms_eps || ov::shape_سائيز(rms_eps->get_shape()) != 1) {
        أرجع خطأ;
    }
    ثابت خودڪار axes = ov::as_type_ptr<ov::op::v0::Constant>(سراسري_2->get_داخلا_ڳنڍ_shared_ptr(1));
    ثابت خودڪار چورس = ov::as_type_ptr<ov::op::v1::Multiply>(سراسري_2->get_داخلا_ڳنڍ_shared_ptr(0));
    إذا (!axes || axes->cast_ويڪٽر<int64_t>() != std::vector<int64_t>{-1} || !چورس ||
        چورس->داخلا_قدر(0) != x || چورس->داخلا_قدر(1) != x) {
        أرجع خطأ;
    }
    // RMSNorm(x, rms_eps) / sqrt(D) = x / sqrt(sum(x*x) + D*rms_eps).
    eps = dim * rms_eps->cast_ويڪٽر<عائم>()[0];
    إذا (!std::isfinite(eps) || eps <= 0.0f) {
        أرجع خطأ;
    }
    داخلا_2 = x;
    أرجع سچ;
}

Output_ويڪٽر translate_gated_delta_net(ثابت ڳنڍ_Context & context) {
    خودڪار v_shape = context.get_داخلا_shape(2).منزل_shape();  // [B, T, H_v, S_v]
    خودڪار q_shape = context.get_داخلا_shape(0).منزل_shape();  // [B, T, H_k, S_k]

    // Fused GatedDeltaNet op only supports scalar gate (kda=0).
    // Fall back to reference implementation for per-key-dimension gating.
    // if (kda) {
    //     return translate_gated_delta_net_ref(context);
    // }

    // const int64_t B = v_shape[0];
    // const int64_t T = v_shape[1];
    ثابت int64_t H_v = v_shape[2];
    ثابت int64_t S_v = v_shape[3];
    ثابت int64_t H_k = q_shape[2];
    // const int64_t S_k = q_shape[3];

    خودڪار q = context.get_داخلا(0);
    خودڪار k = context.get_داخلا(1);
    خودڪار v = process_view_داخلا(context, 2, H_v * S_v, 3);
    خودڪار g = context.get_داخلا(3);
    خودڪار beta = context.get_داخلا(4);
    خودڪار state = context.get_داخلا(5);

    Output<Node> raw_q, raw_k;
    عائم q_eps = 1e-6f, k_eps = 1e-6f;
    ثابت منطقي fuse_qk_l2norm = ggml_openvino_is_gpu() &&
        ميلاپ_gdn_l2_norm(q, raw_q, q_eps) && ميلاپ_gdn_l2_norm(k, raw_k, k_eps);
    إذا (fuse_qk_l2norm) {
        // Keep head tiling below; GDN applies normalization per head and keeps its attention scale.
        q = raw_q;
        k = raw_k;
    }

    // ggml maps GQA heads in tiled order, while OV GDN maps repeated heads in grouped order.
    إذا (H_v != H_k) {
        ثابت int64_t repeat = H_v / H_k;
        خودڪار repeats = ov::op::v0::Constant::create(ov::element::i64, {4}, std::vector<int64_t>{1, 1, repeat, 1});
        q = std::make_shared<ov::op::v0::Tile>(q, repeats);
        k = std::make_shared<ov::op::v0::Tile>(k, repeats);
    }

    إذا (context.get_view_داخلا_سائيز(2)) {
        // Same as l2_norm case 1
        v = std::make_shared<ov::op::v0::Squeeze>(v, ov::op::v0::Constant::create(ov::element::i64, {1}, {0}));
        خودڪار v_shape = context.get_داخلا_shape(2).منزل_shape();
        std::vector<int64_t> reshape_نمونو = {0, 0, (int64_t) v_shape[2], (int64_t) v_shape[3]};
        v = std::make_shared<ov::op::v1::Reshape>(
            v, ov::op::v0::Constant::create(ov::element::i64, {4}, reshape_نمونو), سچ);
    }

    // ggml state layout (OV notation): [B, H_v, value_dim, key_dim]
    // GatedDeltaNet op expects: [B, H_v, key_dim, value_dim]
    خودڪار state_perm = ov::op::v0::Constant::create(ov::element::i64, {4}, std::vector<int64_t>{0, 1, 3, 2});
    state = std::make_shared<ov::op::v1::Transpose>(state, state_perm);

    g = std::make_shared<ov::op::v0::Squeeze>(g, ov::op::v0::Constant::create(ov::element::i64, {1}, {3}));
    beta = std::make_shared<ov::op::v0::Squeeze>(beta, ov::op::v0::Constant::create(ov::element::i64, {1}, {3}));

    إذا (context.has_داخلا("chunk_valid_len")) {
        // The last prefill chunk is padded with fabricated tokens. The recurrence is
        //   S_t = S_{t-1} * exp(g_t) + k_t (x) ((v_t - S_{t-1}^T k_t) * beta_t)
        // so forcing g = 0 and beta = 0 makes a padded step an exact identity and keeps the final
        // state equal to the state after the last real token. Attention output at those positions
        // is garbage but never read.
        ثابت خودڪار & g_ps = g.get_partial_shape();
        FRONT_پڇاڙي_OP_CONVERSION_CHECK(g_ps.rank().is_static() && g_ps.rank().get_ڊيگهائي() == 3 && g_ps[1].is_static(),
                                      "GATED_DELTA_NET pad masking requires a static token dimension");
        ثابت int64_t n_ٽوڪن = g_ps[1].get_ڊيگهائي();
        std::vector<int64_t> positions(n_ٽوڪن);
        std::iota(positions.begin(), positions.end(), 0);
        خودڪار valid = std::make_shared<ov::op::v1::Less>(
            ov::op::v0::Constant::create(ov::element::i64, {(size_t) n_ٽوڪن}, positions),
            context.get_داخلا("chunk_valid_len"));
        خودڪار نقاب = std::make_shared<ov::op::v0::Unsqueeze>(
            std::make_shared<ov::op::v0::Convert>(valid, g.get_element_type()),
            ov::op::v0::Constant::create(ov::element::i64, {2}, std::vector<int64_t>{0, 2}));
        g = std::make_shared<ov::op::v1::Multiply>(g, نقاب);
        beta = std::make_shared<ov::op::v1::Multiply>(beta, نقاب);
    }

    // std::cout << "GatedDeltaNet input shapes: q=" << q.get_partial_shape() << ", k=" << k.get_partial_shape()
    //           << ", v=" << v.get_partial_shape() << ", g=" << g.get_partial_shape()
    //           << ", beta=" << beta.get_partial_shape() << ", state=" << state.get_partial_shape() << std::endl;

    خودڪار gdn = std::make_shared<ov::op::internal::GatedDeltaNet>(q, k, v, state, g, beta, fuse_qk_l2norm, q_eps, k_eps);
    خودڪار attn_4d = gdn->output(0);
    خودڪار state_4d = gdn->output(1);  // [B, H_v, key_dim, value_dim]

    // std::cout << "GatedDeltaNet output shapes: attn=" << gdn->output(0).get_partial_shape()
    //           << ", new_state=" << gdn->output(1).get_partial_shape() << std::endl;

    // Transpose output state back to ggml layout [B, H_v, value_dim, key_dim]
    خودڪار state_transposed = std::make_shared<ov::op::v1::Transpose>(state_4d, state_perm);
    إذا (context.get_output_نالا().size() == 2) {
        // The canonical graph consumes the packed GGML result only through separate attention
        // and state VIEWs. Publish the native outputs under those VIEW names to avoid
        // flatten -> concat -> reshape -> slice -> reshape chains. This also works for B > 1.
        أرجع rename_outputs_with_suffix({attn_4d, state_transposed}, context.get_name());
    }

    خودڪار flat_shape_1d = ov::op::v0::Constant::create(ov::element::i64, {1}, {-1});
    خودڪار attn = std::make_shared<ov::op::v1::Reshape>(attn_4d, flat_shape_1d, خطأ);
    خودڪار new_state = std::make_shared<ov::op::v1::Reshape>(state_transposed, flat_shape_1d, خطأ);
    خودڪار packed = std::make_shared<ov::op::v0::Concat>(ov::Output_ويڪٽر{attn, new_state}, 0);
    خودڪار ٻاهريون_shape = ov::op::v0::Constant::create(ov::element::i64, {4},
                                                  std::vector<int64_t>{1, 1, -1 /*T * B + S_v * B*/, S_v * H_v});
    خودڪار res = std::make_shared<ov::op::v1::Reshape>(packed, ٻاهريون_shape, خطأ);

    أرجع rename_outputs_with_suffix({res}, context.get_name());
}

ساكن Output_ويڪٽر translate_gated_delta_net_ref(ثابت ڳنڍ_Context & context) {
    انگ_inputs_check(context, 6, 6);

    // Inputs (OV shapes are reversed from ggml):
    // ggml: q[S_k, H_k, T, B], k[S_k, H_k, T, B], v[S_v, H_v, T, B]
    // OV:   q[B, T, H_k, S_k], k[B, T, H_k, S_k], v[B, T, H_v, S_v]
    // ggml: g[1 or S_v, H_v, T, B], beta[1, H_v, T, B]
    // OV:   g[B, T, H_v, 1 or S_v], beta[B, T, H_v, 1]
    // ggml: state[S_v, S_v, H_v, B]
    // OV:   state[B, H_v, S_v, S_v]
    خودڪار q = process_view_داخلا_new(context, 0);
    خودڪار k = process_view_داخلا_new(context, 1);
    خودڪار v = process_view_داخلا_new(context, 2);
    خودڪار g = process_view_داخلا_new(context, 3);
    خودڪار beta = process_view_داخلا_new(context, 4);
    خودڪار state = process_view_داخلا_new(context, 5);

    خودڪار v_shape = context.get_داخلا_shape(2).منزل_shape();  // [B, T, H_v, S_v]
    خودڪار q_shape = context.get_داخلا_shape(0).منزل_shape();  // [B, T, H_k, S_k]
    خودڪار g_shape = context.get_داخلا_shape(3).منزل_shape();  // [B, T, H_v, 1 or S_v]

    ثابت int64_t B = v_shape[0];
    ثابت int64_t T = v_shape[1];
    ثابت int64_t H_v = v_shape[2];
    ثابت int64_t S_v = v_shape[3];
    ثابت int64_t H_k = q_shape[2];
    ثابت منطقي kda = (g_shape[3] == (size_t) S_v);

    ثابت int64_t rq1 = H_v / H_k;  // head repeat factor
    ثابت عائم scale = 1.0f / std::جذر((عائم) S_v);

    خودڪار axis_1 = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
    خودڪار axis_2 = ov::op::v0::Constant::create(ov::element::i64, {1}, {2});

    // Transpose inputs from [B, T, H, S] to [B, H, T, S] for easier per-head processing
    خودڪار perm_0213 = ov::op::v0::Constant::create(ov::element::i64, {4}, std::vector<int64_t>{0, 2, 1, 3});
    خودڪار q_t = std::make_shared<ov::op::v1::Transpose>(q, perm_0213);        // [B, H_k, T, S_k]
    خودڪار k_t = std::make_shared<ov::op::v1::Transpose>(k, perm_0213);        // [B, H_k, T, S_k]
    خودڪار v_t = std::make_shared<ov::op::v1::Transpose>(v, perm_0213);        // [B, H_v, T, S_v]
    خودڪار g_t = std::make_shared<ov::op::v1::Transpose>(g, perm_0213);        // [B, H_v, T, 1 or S_v]
    خودڪار beta_t = std::make_shared<ov::op::v1::Transpose>(beta, perm_0213);  // [B, H_v, T, 1]

    // Broadcast Q, K heads to match V heads if GQA is used (H_v > H_k)
    ov::Output<ov::Node> q_bh = q_t;
    ov::Output<ov::Node> k_bh = k_t;
    إذا (rq1 > 1) {
        خودڪار q_unsq = std::make_shared<ov::op::v0::Unsqueeze>(q_t, axis_2);  // [B, H_k, 1, T, S]
        خودڪار k_unsq = std::make_shared<ov::op::v0::Unsqueeze>(k_t, axis_2);  // [B, H_k, 1, T, S]

        خودڪار bcast_shape = ov::op::v0::Constant::create(ov::element::i64, {5}, std::vector<int64_t>{1, 1, rq1, 1, 1});
        خودڪار q_bcast =
            std::make_shared<ov::op::v3::نشر>(q_unsq, bcast_shape, ov::op::نشر_Type::BIDIRECTIONAL);
        خودڪار k_bcast =
            std::make_shared<ov::op::v3::نشر>(k_unsq, bcast_shape, ov::op::نشر_Type::BIDIRECTIONAL);

        // Transpose [B, H_k, rq1, T, S] -> [B, rq1, H_k, T, S] so that reshape merges
        // as [rq1, H_k] giving repeat-blocks pattern matching CPU: iq1 = iv1 % H_k
        خودڪار perm_5d = ov::op::v0::Constant::create(ov::element::i64, {5}, std::vector<int64_t>{0, 2, 1, 3, 4});
        خودڪار q_transposed = std::make_shared<ov::op::v1::Transpose>(q_bcast, perm_5d);
        خودڪار k_transposed = std::make_shared<ov::op::v1::Transpose>(k_bcast, perm_5d);

        خودڪار new_shape = ov::op::v0::Constant::create(ov::element::i64, {4}, std::vector<int64_t>{B, H_v, T, S_v});
        q_bh = std::make_shared<ov::op::v1::Reshape>(q_transposed, new_shape, خطأ);
        k_bh = std::make_shared<ov::op::v1::Reshape>(k_transposed, new_shape, خطأ);
    }

    // Merge batch and head dims: [B*H_v, T, S_v]
    خودڪار ملايو_bh = [&](ثابت ov::Output<ov::Node> & x, int64_t آخري_dim) {
        خودڪار shape = ov::op::v0::Constant::create(ov::element::i64, {3}, std::vector<int64_t>{B * H_v, T, آخري_dim});
        أرجع std::make_shared<ov::op::v1::Reshape>(x, shape, خطأ);
    };

    خودڪار q_m = ملايو_bh(q_bh, S_v);           // [B*H_v, T, S_v]
    خودڪار k_m = ملايو_bh(k_bh, S_v);           // [B*H_v, T, S_v]
    خودڪار v_m = ملايو_bh(v_t, S_v);            // [B*H_v, T, S_v]
    خودڪار g_m = ملايو_bh(g_t, kda ? S_v : 1);  // [B*H_v, T, 1 or S_v]
    خودڪار beta_m = ملايو_bh(beta_t, 1);        // [B*H_v, T, 1]

    // State: [B, H_v, S_v, S_v] -> [B*H_v, S_v, S_v]
    خودڪار state_shape = ov::op::v0::Constant::create(ov::element::i64, {3}, std::vector<int64_t>{B * H_v, S_v, S_v});
    خودڪار state_m = std::make_shared<ov::op::v1::Reshape>(state, state_shape, خطأ);

    خودڪار scale_const = ov::op::v0::Constant::create(ov::element::f32, {}, std::vector<عائم>{scale});

    // --- Build Loop body ---
    // Body parameters (no iteration counter needed, use -1 in special ports)
    خودڪار body_state = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic());
    خودڪار body_q = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic());
    خودڪار body_k = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic());
    خودڪار body_v = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic());
    خودڪار body_g = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic());
    خودڪار body_beta = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic());
    خودڪار body_iter = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::Shape{1});

    // Condition output (always true - we rely on trip_count for termination)
    خودڪار body_cond_ٻاهريون = ov::op::v0::Constant::create(ov::element::boolean, ov::Shape{1}, std::vector<منطقي>{سچ});

    // Gather current token from invariant inputs using iteration counter
    خودڪار q_t_cur = std::make_shared<ov::op::v8::Gather>(body_q, body_iter, axis_1);     // [B*H_v, 1, S_v]
    خودڪار k_t_cur = std::make_shared<ov::op::v8::Gather>(body_k, body_iter, axis_1);     // [B*H_v, 1, S_v]
    خودڪار v_t_cur = std::make_shared<ov::op::v8::Gather>(body_v, body_iter, axis_1);     // [B*H_v, 1, S_v]
    خودڪار g_t_cur = std::make_shared<ov::op::v8::Gather>(body_g, body_iter, axis_1);     // [B*H_v, 1, 1 or S_v]
    خودڪار b_t_cur = std::make_shared<ov::op::v8::Gather>(body_beta, body_iter, axis_1);  // [B*H_v, 1, 1]

    // Squeeze token dim
    خودڪار q_cur = std::make_shared<ov::op::v0::Squeeze>(q_t_cur, axis_1);  // [B*H_v, S_v]
    خودڪار k_cur = std::make_shared<ov::op::v0::Squeeze>(k_t_cur, axis_1);  // [B*H_v, S_v]
    خودڪار v_cur = std::make_shared<ov::op::v0::Squeeze>(v_t_cur, axis_1);  // [B*H_v, S_v]
    خودڪار g_cur = std::make_shared<ov::op::v0::Squeeze>(g_t_cur, axis_1);  // [B*H_v, 1 or S_v]
    خودڪار b_cur = std::make_shared<ov::op::v0::Squeeze>(b_t_cur, axis_1);  // [B*H_v, 1]

    // Step 1: Apply decay gate to state
    خودڪار exp_g = std::make_shared<ov::op::v0::Exp>(g_cur);                                // [B*H_v, 1 or S_v]
    خودڪار exp_g_unsq = std::make_shared<ov::op::v0::Unsqueeze>(exp_g, axis_1);             // [B*H_v, 1, 1 or S_v]
    خودڪار state_decayed = std::make_shared<ov::op::v1::Multiply>(body_state, exp_g_unsq);  // [B*H_v, S_v, S_v]

    // Step 2: delta = (v - S @ k) * beta
    خودڪار k_کانو = std::make_shared<ov::op::v0::Unsqueeze>(k_cur, axis_2);                 // [B*H_v, S_v, 1]
    خودڪار sk = std::make_shared<ov::op::v0::MatMul>(state_decayed, k_کانو, خطأ, خطأ);  // [B*H_v, S_v, 1]
    خودڪار sk_sq = std::make_shared<ov::op::v0::Squeeze>(sk, axis_2);                      // [B*H_v, S_v]
    خودڪار v_minus_sk = std::make_shared<ov::op::v1::ڪٽيو>(v_cur, sk_sq);              // [B*H_v, S_v]
    خودڪار delta = std::make_shared<ov::op::v1::Multiply>(v_minus_sk, b_cur);              // [B*H_v, S_v]

    // Step 3: state += outer(delta, k)
    خودڪار delta_کانو = std::make_shared<ov::op::v0::Unsqueeze>(delta, axis_2);                 // [B*H_v, S_v, 1]
    خودڪار k_صف = std::make_shared<ov::op::v0::Unsqueeze>(k_cur, axis_1);                     // [B*H_v, 1, S_v]
    خودڪار outer_prod = std::make_shared<ov::op::v0::MatMul>(delta_کانو, k_صف, خطأ, خطأ);  // [B*H_v, S_v, S_v]
    خودڪار state_updated = std::make_shared<ov::op::v1::جوڙيو>(state_decayed, outer_prod);       // [B*H_v, S_v, S_v]

    // Step 4: attn_out = S @ q * scale
    خودڪار q_کانو = std::make_shared<ov::op::v0::Unsqueeze>(q_cur, axis_2);                 // [B*H_v, S_v, 1]
    خودڪار sq = std::make_shared<ov::op::v0::MatMul>(state_updated, q_کانو, خطأ, خطأ);  // [B*H_v, S_v, 1]
    خودڪار sq_squeezed = std::make_shared<ov::op::v0::Squeeze>(sq, axis_2);                // [B*H_v, S_v]
    خودڪار attn_ٻاهريون = std::make_shared<ov::op::v1::Multiply>(sq_squeezed, scale_const);    // [B*H_v, S_v]

    // Unsqueeze attn_out to [B*H_v, 1, S_v] for scan output concatenation
    خودڪار attn_ٻاهريون_unsq = std::make_shared<ov::op::v0::Unsqueeze>(attn_ٻاهريون, axis_1);  // [B*H_v, 1, S_v]

    // --- Assemble Loop ---
    // Body: results = [condition, state_updated, attn_out_unsq]
    خودڪار body = std::make_shared<ov::Model>(
        ov::Output_ويڪٽر{body_cond_ٻاهريون, state_updated, attn_ٻاهريون_unsq},
        ov::Parameter_ويڪٽر{body_iter, body_state, body_q, body_k, body_v, body_g, body_beta});

    خودڪار trip_ڳڻپ = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{T});
    خودڪار exec_cond = ov::op::v0::Constant::create(ov::element::boolean, ov::Shape{1}, std::vector<منطقي>{سچ});

    خودڪار loop = std::make_shared<ov::op::v5::Loop>(trip_ڳڻپ, exec_cond);
    loop->سيٽ_function(body);
    loop->سيٽ_special_body_ports(ov::op::v5::Loop::SpecialBodyPorts{0, 0});

    // Carried state: feeds back from body output 1 to body_state param
    loop->سيٽ_merged_داخلا(body_state, state_m, state_updated);
    // Invariant inputs: passed through unchanged each iteration
    loop->سيٽ_invariant_داخلا(body_q, q_m);
    loop->سيٽ_invariant_داخلا(body_k, k_m);
    loop->سيٽ_invariant_داخلا(body_v, v_m);
    loop->سيٽ_invariant_داخلا(body_g, g_m);
    loop->سيٽ_invariant_داخلا(body_beta, beta_m);

    // Loop outputs:
    // 1) Final state (last iteration value of state_updated)
    خودڪار final_state_ٻاهريون = loop->get_iter_قدر(state_updated, -1);  // [B*H_v, S_v, S_v]
    // 2) Concatenated attention outputs across all iterations along axis 1
    خودڪار attn_concat_ٻاهريون = loop->get_concatenated_slices(attn_ٻاهريون_unsq, 0, 1, 1, -1, 1);  // [B*H_v, T, S_v]

    // --- Pack outputs to match ggml layout ---
    // ggml output ne = {S_v*H, T*B + S_v*B, 1, 1} -> OV [1, 1, T*B+S_v*B, S_v*H_v]
    // attn: [B, T, H_v, S_v] row-major, state: [B, H_v, S_v, S_v] row-major

    // attn: [B*H_v, T, S_v] -> [B, H_v, T, S_v] -> transpose to [B, T, H_v, S_v] -> flatten
    خودڪار attn_4d_shape = ov::op::v0::Constant::create(ov::element::i64, {4}, std::vector<int64_t>{B, H_v, T, S_v});
    خودڪار attn_4d = std::make_shared<ov::op::v1::Reshape>(attn_concat_ٻاهريون, attn_4d_shape, خطأ);
    خودڪار attn_perm = std::make_shared<ov::op::v1::Transpose>(attn_4d, perm_0213);  // [B, T, H_v, S_v]

    خودڪار flat_shape_1d = ov::op::v0::Constant::create(ov::element::i64, {1}, std::vector<int64_t>{-1});
    خودڪار attn_1d = std::make_shared<ov::op::v1::Reshape>(attn_perm, flat_shape_1d, خطأ);

    // state: [B*H_v, S_v, S_v] -> [B, H_v, S_v, S_v] -> flatten
    خودڪار state_4d_shape = ov::op::v0::Constant::create(ov::element::i64, {4}, std::vector<int64_t>{B, H_v, S_v, S_v});
    خودڪار state_4d = std::make_shared<ov::op::v1::Reshape>(final_state_ٻاهريون, state_4d_shape, خطأ);
    إذا (context.get_output_نالا().size() == 2) {
        // Match the fused translator's direct attention/state contract.
        أرجع rename_outputs_with_suffix({attn_perm, state_4d}, context.get_name());
    }

    خودڪار state_1d = std::make_shared<ov::op::v1::Reshape>(state_4d, flat_shape_1d, خطأ);

    // Concat [attn | state] and reshape to final output
    خودڪار packed = std::make_shared<ov::op::v0::Concat>(ov::Output_ويڪٽر{attn_1d, state_1d}, 0);
    خودڪار ٻاهريون_shape =
        ov::op::v0::Constant::create(ov::element::i64, {4}, std::vector<int64_t>{1, 1, T * B + S_v * B, S_v * H_v});
    خودڪار res = std::make_shared<ov::op::v1::Reshape>(packed, ٻاهريون_shape, خطأ);

    أرجع rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
