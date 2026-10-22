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

ساكن Output_سمتیہ translate_gated_delta_net_ref(ثابت گرہ_Context & context);

ساكن منطقي میل_gdn_l2_norm(ثابت Output<Node> & normalized, Output<Node> & داخلہ, عائم & eps) {
    // Match the RMSNorm decomposition emitted by translate_rms_norm, followed by GGML SCALE.
    ثابت خودکار scale = ov::as_type_ptr<ov::op::v1::Multiply>(normalized.get_گرہ_shared_ptr());
    إذا (!scale) {
        أرجع خطأ;
    }
    ثابت خودکار ضارب = ov::as_type_ptr<ov::op::v0::Constant>(scale->get_داخلہ_گرہ_shared_ptr(1));
    ثابت خودکار rms = ov::as_type_ptr<ov::op::v1::Multiply>(scale->get_داخلہ_گرہ_shared_ptr(0));
    إذا (!ضارب || ov::shape_ناپ(ضارب->get_shape()) != 1 || !rms) {
        أرجع خطأ;
    }
    ثابت خودکار x = rms->داخلہ_قدر(0);
    ثابت خودکار & shape = x.get_partial_shape();
    إذا (x.get_element_type() != ov::element::f32 || shape.rank() != 4 || shape[3].is_dynamic() ||
        shape[3].get_لمائی() <= 0 || normalized.get_partial_shape() != shape) {
        أرجع خطأ;
    }
    ثابت عائم dim = static_cast<عائم>(shape[3].get_لمائی());
    إذا (ضارب->cast_سمتیہ<عائم>()[0] != 1.0f / std::جذر(dim)) {
        أرجع خطأ;
    }
    ثابت خودکار reciprocal = ov::as_type_ptr<ov::op::v1::Divide>(rms->get_داخلہ_گرہ_shared_ptr(1));
    إذا (!reciprocal) {
        أرجع خطأ;
    }
    ثابت خودکار one = ov::as_type_ptr<ov::op::v0::Constant>(reciprocal->get_داخلہ_گرہ_shared_ptr(0));
    ثابت خودکار جڑ = ov::as_type_ptr<ov::op::v0::Sqrt>(reciprocal->get_داخلہ_گرہ_shared_ptr(1));
    إذا (!one || ov::shape_ناپ(one->get_shape()) != 1 || one->cast_سمتیہ<عائم>()[0] != 1.0f || !جڑ) {
        أرجع خطأ;
    }
    ثابت خودکار جوڑو_2 = ov::as_type_ptr<ov::op::v1::جوڑو>(جڑ->get_داخلہ_گرہ_shared_ptr(0));
    إذا (!جوڑو_2) {
        أرجع خطأ;
    }
    ثابت خودکار اوسط_2 = ov::as_type_ptr<ov::op::v1::Reduce_اوسط>(جوڑو_2->get_داخلہ_گرہ_shared_ptr(0));
    ثابت خودکار rms_eps = ov::as_type_ptr<ov::op::v0::Constant>(جوڑو_2->get_داخلہ_گرہ_shared_ptr(1));
    إذا (!اوسط_2 || !اوسط_2->get_keep_dims() || !rms_eps || ov::shape_ناپ(rms_eps->get_shape()) != 1) {
        أرجع خطأ;
    }
    ثابت خودکار axes = ov::as_type_ptr<ov::op::v0::Constant>(اوسط_2->get_داخلہ_گرہ_shared_ptr(1));
    ثابت خودکار مربع = ov::as_type_ptr<ov::op::v1::Multiply>(اوسط_2->get_داخلہ_گرہ_shared_ptr(0));
    إذا (!axes || axes->cast_سمتیہ<int64_t>() != std::vector<int64_t>{-1} || !مربع ||
        مربع->داخلہ_قدر(0) != x || مربع->داخلہ_قدر(1) != x) {
        أرجع خطأ;
    }
    // RMSNorm(x, rms_eps) / sqrt(D) = x / sqrt(sum(x*x) + D*rms_eps).
    eps = dim * rms_eps->cast_سمتیہ<عائم>()[0];
    إذا (!std::isfinite(eps) || eps <= 0.0f) {
        أرجع خطأ;
    }
    داخلہ = x;
    أرجع سچ;
}

Output_سمتیہ translate_gated_delta_net(ثابت گرہ_Context & context) {
    خودکار v_shape = context.get_داخلہ_shape(2).منزل_shape();  // [B, T, H_v, S_v]
    خودکار q_shape = context.get_داخلہ_shape(0).منزل_shape();  // [B, T, H_k, S_k]

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

    خودکار q = context.get_داخلہ(0);
    خودکار k = context.get_داخلہ(1);
    خودکار v = process_view_داخلہ(context, 2, H_v * S_v, 3);
    خودکار g = context.get_داخلہ(3);
    خودکار beta = context.get_داخلہ(4);
    خودکار state = context.get_داخلہ(5);

    Output<Node> raw_q, raw_k;
    عائم q_eps = 1e-6f, k_eps = 1e-6f;
    ثابت منطقي fuse_qk_l2norm = ggml_openvino_is_gpu() &&
        میل_gdn_l2_norm(q, raw_q, q_eps) && میل_gdn_l2_norm(k, raw_k, k_eps);
    إذا (fuse_qk_l2norm) {
        // Keep head tiling below; GDN applies normalization per head and keeps its attention scale.
        q = raw_q;
        k = raw_k;
    }

    // ggml maps GQA heads in tiled order, while OV GDN maps repeated heads in grouped order.
    إذا (H_v != H_k) {
        ثابت int64_t repeat = H_v / H_k;
        خودکار repeats = ov::op::v0::Constant::create(ov::element::i64, {4}, std::vector<int64_t>{1, 1, repeat, 1});
        q = std::make_shared<ov::op::v0::Tile>(q, repeats);
        k = std::make_shared<ov::op::v0::Tile>(k, repeats);
    }

    إذا (context.get_view_داخلہ_ناپ(2)) {
        // Same as l2_norm case 1
        v = std::make_shared<ov::op::v0::Squeeze>(v, ov::op::v0::Constant::create(ov::element::i64, {1}, {0}));
        خودکار v_shape = context.get_داخلہ_shape(2).منزل_shape();
        std::vector<int64_t> reshape_نمونہ = {0, 0, (int64_t) v_shape[2], (int64_t) v_shape[3]};
        v = std::make_shared<ov::op::v1::Reshape>(
            v, ov::op::v0::Constant::create(ov::element::i64, {4}, reshape_نمونہ), سچ);
    }

    // ggml state layout (OV notation): [B, H_v, value_dim, key_dim]
    // GatedDeltaNet op expects: [B, H_v, key_dim, value_dim]
    خودکار state_perm = ov::op::v0::Constant::create(ov::element::i64, {4}, std::vector<int64_t>{0, 1, 3, 2});
    state = std::make_shared<ov::op::v1::Transpose>(state, state_perm);

    g = std::make_shared<ov::op::v0::Squeeze>(g, ov::op::v0::Constant::create(ov::element::i64, {1}, {3}));
    beta = std::make_shared<ov::op::v0::Squeeze>(beta, ov::op::v0::Constant::create(ov::element::i64, {1}, {3}));

    إذا (context.has_داخلہ("chunk_valid_len")) {
        // The last prefill chunk is padded with fabricated tokens. The recurrence is
        //   S_t = S_{t-1} * exp(g_t) + k_t (x) ((v_t - S_{t-1}^T k_t) * beta_t)
        // so forcing g = 0 and beta = 0 makes a padded step an exact identity and keeps the final
        // state equal to the state after the last real token. Attention output at those positions
        // is garbage but never read.
        ثابت خودکار & g_ps = g.get_partial_shape();
        FRONT_سرا_OP_CONVERSION_CHECK(g_ps.rank().is_static() && g_ps.rank().get_لمائی() == 3 && g_ps[1].is_static(),
                                      "GATED_DELTA_NET pad masking requires a static token dimension");
        ثابت int64_t n_ٹوکن = g_ps[1].get_لمائی();
        std::vector<int64_t> positions(n_ٹوکن);
        std::iota(positions.begin(), positions.end(), 0);
        خودکار valid = std::make_shared<ov::op::v1::Less>(
            ov::op::v0::Constant::create(ov::element::i64, {(size_t) n_ٹوکن}, positions),
            context.get_داخلہ("chunk_valid_len"));
        خودکار نقاب = std::make_shared<ov::op::v0::Unsqueeze>(
            std::make_shared<ov::op::v0::Convert>(valid, g.get_element_type()),
            ov::op::v0::Constant::create(ov::element::i64, {2}, std::vector<int64_t>{0, 2}));
        g = std::make_shared<ov::op::v1::Multiply>(g, نقاب);
        beta = std::make_shared<ov::op::v1::Multiply>(beta, نقاب);
    }

    // std::cout << "GatedDeltaNet input shapes: q=" << q.get_partial_shape() << ", k=" << k.get_partial_shape()
    //           << ", v=" << v.get_partial_shape() << ", g=" << g.get_partial_shape()
    //           << ", beta=" << beta.get_partial_shape() << ", state=" << state.get_partial_shape() << std::endl;

    خودکار gdn = std::make_shared<ov::op::internal::GatedDeltaNet>(q, k, v, state, g, beta, fuse_qk_l2norm, q_eps, k_eps);
    خودکار attn_4d = gdn->output(0);
    خودکار state_4d = gdn->output(1);  // [B, H_v, key_dim, value_dim]

    // std::cout << "GatedDeltaNet output shapes: attn=" << gdn->output(0).get_partial_shape()
    //           << ", new_state=" << gdn->output(1).get_partial_shape() << std::endl;

    // Transpose output state back to ggml layout [B, H_v, value_dim, key_dim]
    خودکار state_transposed = std::make_shared<ov::op::v1::Transpose>(state_4d, state_perm);
    إذا (context.get_output_ناں().size() == 2) {
        // The canonical graph consumes the packed GGML result only through separate attention
        // and state VIEWs. Publish the native outputs under those VIEW names to avoid
        // flatten -> concat -> reshape -> slice -> reshape chains. This also works for B > 1.
        أرجع rename_outputs_with_suffix({attn_4d, state_transposed}, context.get_name());
    }

    خودکار flat_shape_1d = ov::op::v0::Constant::create(ov::element::i64, {1}, {-1});
    خودکار attn = std::make_shared<ov::op::v1::Reshape>(attn_4d, flat_shape_1d, خطأ);
    خودکار new_state = std::make_shared<ov::op::v1::Reshape>(state_transposed, flat_shape_1d, خطأ);
    خودکار packed = std::make_shared<ov::op::v0::Concat>(ov::Output_سمتیہ{attn, new_state}, 0);
    خودکار باہرلا_shape = ov::op::v0::Constant::create(ov::element::i64, {4},
                                                  std::vector<int64_t>{1, 1, -1 /*T * B + S_v * B*/, S_v * H_v});
    خودکار res = std::make_shared<ov::op::v1::Reshape>(packed, باہرلا_shape, خطأ);

    أرجع rename_outputs_with_suffix({res}, context.get_name());
}

ساكن Output_سمتیہ translate_gated_delta_net_ref(ثابت گرہ_Context & context) {
    عدد_inputs_check(context, 6, 6);

    // Inputs (OV shapes are reversed from ggml):
    // ggml: q[S_k, H_k, T, B], k[S_k, H_k, T, B], v[S_v, H_v, T, B]
    // OV:   q[B, T, H_k, S_k], k[B, T, H_k, S_k], v[B, T, H_v, S_v]
    // ggml: g[1 or S_v, H_v, T, B], beta[1, H_v, T, B]
    // OV:   g[B, T, H_v, 1 or S_v], beta[B, T, H_v, 1]
    // ggml: state[S_v, S_v, H_v, B]
    // OV:   state[B, H_v, S_v, S_v]
    خودکار q = process_view_داخلہ_new(context, 0);
    خودکار k = process_view_داخلہ_new(context, 1);
    خودکار v = process_view_داخلہ_new(context, 2);
    خودکار g = process_view_داخلہ_new(context, 3);
    خودکار beta = process_view_داخلہ_new(context, 4);
    خودکار state = process_view_داخلہ_new(context, 5);

    خودکار v_shape = context.get_داخلہ_shape(2).منزل_shape();  // [B, T, H_v, S_v]
    خودکار q_shape = context.get_داخلہ_shape(0).منزل_shape();  // [B, T, H_k, S_k]
    خودکار g_shape = context.get_داخلہ_shape(3).منزل_shape();  // [B, T, H_v, 1 or S_v]

    ثابت int64_t B = v_shape[0];
    ثابت int64_t T = v_shape[1];
    ثابت int64_t H_v = v_shape[2];
    ثابت int64_t S_v = v_shape[3];
    ثابت int64_t H_k = q_shape[2];
    ثابت منطقي kda = (g_shape[3] == (size_t) S_v);

    ثابت int64_t rq1 = H_v / H_k;  // head repeat factor
    ثابت عائم scale = 1.0f / std::جذر((عائم) S_v);

    خودکار axis_1 = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
    خودکار axis_2 = ov::op::v0::Constant::create(ov::element::i64, {1}, {2});

    // Transpose inputs from [B, T, H, S] to [B, H, T, S] for easier per-head processing
    خودکار perm_0213 = ov::op::v0::Constant::create(ov::element::i64, {4}, std::vector<int64_t>{0, 2, 1, 3});
    خودکار q_t = std::make_shared<ov::op::v1::Transpose>(q, perm_0213);        // [B, H_k, T, S_k]
    خودکار k_t = std::make_shared<ov::op::v1::Transpose>(k, perm_0213);        // [B, H_k, T, S_k]
    خودکار v_t = std::make_shared<ov::op::v1::Transpose>(v, perm_0213);        // [B, H_v, T, S_v]
    خودکار g_t = std::make_shared<ov::op::v1::Transpose>(g, perm_0213);        // [B, H_v, T, 1 or S_v]
    خودکار beta_t = std::make_shared<ov::op::v1::Transpose>(beta, perm_0213);  // [B, H_v, T, 1]

    // Broadcast Q, K heads to match V heads if GQA is used (H_v > H_k)
    ov::Output<ov::Node> q_bh = q_t;
    ov::Output<ov::Node> k_bh = k_t;
    إذا (rq1 > 1) {
        خودکار q_unsq = std::make_shared<ov::op::v0::Unsqueeze>(q_t, axis_2);  // [B, H_k, 1, T, S]
        خودکار k_unsq = std::make_shared<ov::op::v0::Unsqueeze>(k_t, axis_2);  // [B, H_k, 1, T, S]

        خودکار bcast_shape = ov::op::v0::Constant::create(ov::element::i64, {5}, std::vector<int64_t>{1, 1, rq1, 1, 1});
        خودکار q_bcast =
            std::make_shared<ov::op::v3::نشر>(q_unsq, bcast_shape, ov::op::نشر_Type::BIDIRECTIONAL);
        خودکار k_bcast =
            std::make_shared<ov::op::v3::نشر>(k_unsq, bcast_shape, ov::op::نشر_Type::BIDIRECTIONAL);

        // Transpose [B, H_k, rq1, T, S] -> [B, rq1, H_k, T, S] so that reshape merges
        // as [rq1, H_k] giving repeat-blocks pattern matching CPU: iq1 = iv1 % H_k
        خودکار perm_5d = ov::op::v0::Constant::create(ov::element::i64, {5}, std::vector<int64_t>{0, 2, 1, 3, 4});
        خودکار q_transposed = std::make_shared<ov::op::v1::Transpose>(q_bcast, perm_5d);
        خودکار k_transposed = std::make_shared<ov::op::v1::Transpose>(k_bcast, perm_5d);

        خودکار new_shape = ov::op::v0::Constant::create(ov::element::i64, {4}, std::vector<int64_t>{B, H_v, T, S_v});
        q_bh = std::make_shared<ov::op::v1::Reshape>(q_transposed, new_shape, خطأ);
        k_bh = std::make_shared<ov::op::v1::Reshape>(k_transposed, new_shape, خطأ);
    }

    // Merge batch and head dims: [B*H_v, T, S_v]
    خودکار ملاؤ_bh = [&](ثابت ov::Output<ov::Node> & x, int64_t آخری_dim) {
        خودکار shape = ov::op::v0::Constant::create(ov::element::i64, {3}, std::vector<int64_t>{B * H_v, T, آخری_dim});
        أرجع std::make_shared<ov::op::v1::Reshape>(x, shape, خطأ);
    };

    خودکار q_m = ملاؤ_bh(q_bh, S_v);           // [B*H_v, T, S_v]
    خودکار k_m = ملاؤ_bh(k_bh, S_v);           // [B*H_v, T, S_v]
    خودکار v_m = ملاؤ_bh(v_t, S_v);            // [B*H_v, T, S_v]
    خودکار g_m = ملاؤ_bh(g_t, kda ? S_v : 1);  // [B*H_v, T, 1 or S_v]
    خودکار beta_m = ملاؤ_bh(beta_t, 1);        // [B*H_v, T, 1]

    // State: [B, H_v, S_v, S_v] -> [B*H_v, S_v, S_v]
    خودکار state_shape = ov::op::v0::Constant::create(ov::element::i64, {3}, std::vector<int64_t>{B * H_v, S_v, S_v});
    خودکار state_m = std::make_shared<ov::op::v1::Reshape>(state, state_shape, خطأ);

    خودکار scale_const = ov::op::v0::Constant::create(ov::element::f32, {}, std::vector<عائم>{scale});

    // --- Build Loop body ---
    // Body parameters (no iteration counter needed, use -1 in special ports)
    خودکار body_state = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic());
    خودکار body_q = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic());
    خودکار body_k = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic());
    خودکار body_v = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic());
    خودکار body_g = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic());
    خودکار body_beta = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic());
    خودکار body_iter = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::Shape{1});

    // Condition output (always true - we rely on trip_count for termination)
    خودکار body_cond_باہرلا = ov::op::v0::Constant::create(ov::element::boolean, ov::Shape{1}, std::vector<منطقي>{سچ});

    // Gather current token from invariant inputs using iteration counter
    خودکار q_t_cur = std::make_shared<ov::op::v8::Gather>(body_q, body_iter, axis_1);     // [B*H_v, 1, S_v]
    خودکار k_t_cur = std::make_shared<ov::op::v8::Gather>(body_k, body_iter, axis_1);     // [B*H_v, 1, S_v]
    خودکار v_t_cur = std::make_shared<ov::op::v8::Gather>(body_v, body_iter, axis_1);     // [B*H_v, 1, S_v]
    خودکار g_t_cur = std::make_shared<ov::op::v8::Gather>(body_g, body_iter, axis_1);     // [B*H_v, 1, 1 or S_v]
    خودکار b_t_cur = std::make_shared<ov::op::v8::Gather>(body_beta, body_iter, axis_1);  // [B*H_v, 1, 1]

    // Squeeze token dim
    خودکار q_cur = std::make_shared<ov::op::v0::Squeeze>(q_t_cur, axis_1);  // [B*H_v, S_v]
    خودکار k_cur = std::make_shared<ov::op::v0::Squeeze>(k_t_cur, axis_1);  // [B*H_v, S_v]
    خودکار v_cur = std::make_shared<ov::op::v0::Squeeze>(v_t_cur, axis_1);  // [B*H_v, S_v]
    خودکار g_cur = std::make_shared<ov::op::v0::Squeeze>(g_t_cur, axis_1);  // [B*H_v, 1 or S_v]
    خودکار b_cur = std::make_shared<ov::op::v0::Squeeze>(b_t_cur, axis_1);  // [B*H_v, 1]

    // Step 1: Apply decay gate to state
    خودکار exp_g = std::make_shared<ov::op::v0::Exp>(g_cur);                                // [B*H_v, 1 or S_v]
    خودکار exp_g_unsq = std::make_shared<ov::op::v0::Unsqueeze>(exp_g, axis_1);             // [B*H_v, 1, 1 or S_v]
    خودکار state_decayed = std::make_shared<ov::op::v1::Multiply>(body_state, exp_g_unsq);  // [B*H_v, S_v, S_v]

    // Step 2: delta = (v - S @ k) * beta
    خودکار k_خانہ = std::make_shared<ov::op::v0::Unsqueeze>(k_cur, axis_2);                 // [B*H_v, S_v, 1]
    خودکار sk = std::make_shared<ov::op::v0::MatMul>(state_decayed, k_خانہ, خطأ, خطأ);  // [B*H_v, S_v, 1]
    خودکار sk_sq = std::make_shared<ov::op::v0::Squeeze>(sk, axis_2);                      // [B*H_v, S_v]
    خودکار v_minus_sk = std::make_shared<ov::op::v1::گھٹاؤ>(v_cur, sk_sq);              // [B*H_v, S_v]
    خودکار delta = std::make_shared<ov::op::v1::Multiply>(v_minus_sk, b_cur);              // [B*H_v, S_v]

    // Step 3: state += outer(delta, k)
    خودکار delta_خانہ = std::make_shared<ov::op::v0::Unsqueeze>(delta, axis_2);                 // [B*H_v, S_v, 1]
    خودکار k_صف = std::make_shared<ov::op::v0::Unsqueeze>(k_cur, axis_1);                     // [B*H_v, 1, S_v]
    خودکار outer_prod = std::make_shared<ov::op::v0::MatMul>(delta_خانہ, k_صف, خطأ, خطأ);  // [B*H_v, S_v, S_v]
    خودکار state_updated = std::make_shared<ov::op::v1::جوڑو>(state_decayed, outer_prod);       // [B*H_v, S_v, S_v]

    // Step 4: attn_out = S @ q * scale
    خودکار q_خانہ = std::make_shared<ov::op::v0::Unsqueeze>(q_cur, axis_2);                 // [B*H_v, S_v, 1]
    خودکار sq = std::make_shared<ov::op::v0::MatMul>(state_updated, q_خانہ, خطأ, خطأ);  // [B*H_v, S_v, 1]
    خودکار sq_squeezed = std::make_shared<ov::op::v0::Squeeze>(sq, axis_2);                // [B*H_v, S_v]
    خودکار attn_باہرلا = std::make_shared<ov::op::v1::Multiply>(sq_squeezed, scale_const);    // [B*H_v, S_v]

    // Unsqueeze attn_out to [B*H_v, 1, S_v] for scan output concatenation
    خودکار attn_باہرلا_unsq = std::make_shared<ov::op::v0::Unsqueeze>(attn_باہرلا, axis_1);  // [B*H_v, 1, S_v]

    // --- Assemble Loop ---
    // Body: results = [condition, state_updated, attn_out_unsq]
    خودکار body = std::make_shared<ov::Model>(
        ov::Output_سمتیہ{body_cond_باہرلا, state_updated, attn_باہرلا_unsq},
        ov::Parameter_سمتیہ{body_iter, body_state, body_q, body_k, body_v, body_g, body_beta});

    خودکار trip_گنتی = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{T});
    خودکار exec_cond = ov::op::v0::Constant::create(ov::element::boolean, ov::Shape{1}, std::vector<منطقي>{سچ});

    خودکار loop = std::make_shared<ov::op::v5::Loop>(trip_گنتی, exec_cond);
    loop->سیٹ_function(body);
    loop->سیٹ_special_body_ports(ov::op::v5::Loop::SpecialBodyPorts{0, 0});

    // Carried state: feeds back from body output 1 to body_state param
    loop->سیٹ_merged_داخلہ(body_state, state_m, state_updated);
    // Invariant inputs: passed through unchanged each iteration
    loop->سیٹ_invariant_داخلہ(body_q, q_m);
    loop->سیٹ_invariant_داخلہ(body_k, k_m);
    loop->سیٹ_invariant_داخلہ(body_v, v_m);
    loop->سیٹ_invariant_داخلہ(body_g, g_m);
    loop->سیٹ_invariant_داخلہ(body_beta, beta_m);

    // Loop outputs:
    // 1) Final state (last iteration value of state_updated)
    خودکار final_state_باہرلا = loop->get_iter_قدر(state_updated, -1);  // [B*H_v, S_v, S_v]
    // 2) Concatenated attention outputs across all iterations along axis 1
    خودکار attn_concat_باہرلا = loop->get_concatenated_slices(attn_باہرلا_unsq, 0, 1, 1, -1, 1);  // [B*H_v, T, S_v]

    // --- Pack outputs to match ggml layout ---
    // ggml output ne = {S_v*H, T*B + S_v*B, 1, 1} -> OV [1, 1, T*B+S_v*B, S_v*H_v]
    // attn: [B, T, H_v, S_v] row-major, state: [B, H_v, S_v, S_v] row-major

    // attn: [B*H_v, T, S_v] -> [B, H_v, T, S_v] -> transpose to [B, T, H_v, S_v] -> flatten
    خودکار attn_4d_shape = ov::op::v0::Constant::create(ov::element::i64, {4}, std::vector<int64_t>{B, H_v, T, S_v});
    خودکار attn_4d = std::make_shared<ov::op::v1::Reshape>(attn_concat_باہرلا, attn_4d_shape, خطأ);
    خودکار attn_perm = std::make_shared<ov::op::v1::Transpose>(attn_4d, perm_0213);  // [B, T, H_v, S_v]

    خودکار flat_shape_1d = ov::op::v0::Constant::create(ov::element::i64, {1}, std::vector<int64_t>{-1});
    خودکار attn_1d = std::make_shared<ov::op::v1::Reshape>(attn_perm, flat_shape_1d, خطأ);

    // state: [B*H_v, S_v, S_v] -> [B, H_v, S_v, S_v] -> flatten
    خودکار state_4d_shape = ov::op::v0::Constant::create(ov::element::i64, {4}, std::vector<int64_t>{B, H_v, S_v, S_v});
    خودکار state_4d = std::make_shared<ov::op::v1::Reshape>(final_state_باہرلا, state_4d_shape, خطأ);
    إذا (context.get_output_ناں().size() == 2) {
        // Match the fused translator's direct attention/state contract.
        أرجع rename_outputs_with_suffix({attn_perm, state_4d}, context.get_name());
    }

    خودکار state_1d = std::make_shared<ov::op::v1::Reshape>(state_4d, flat_shape_1d, خطأ);

    // Concat [attn | state] and reshape to final output
    خودکار packed = std::make_shared<ov::op::v0::Concat>(ov::Output_سمتیہ{attn_1d, state_1d}, 0);
    خودکار باہرلا_shape =
        ov::op::v0::Constant::create(ov::element::i64, {4}, std::vector<int64_t>{1, 1, T * B + S_v * B, S_v * H_v});
    خودکار res = std::make_shared<ov::op::v1::Reshape>(packed, باہرلا_shape, خطأ);

    أرجع rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
