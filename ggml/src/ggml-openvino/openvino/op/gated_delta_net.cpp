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

ساكن Output_ویکټور translate_gated_delta_net_ref(ثابت غوټه_Context & context);

ساكن منطقي سمون_gdn_l2_norm(ثابت Output<Node> & normalized, Output<Node> & ننوت, عائم & eps) {
    // Match the RMSNorm decomposition emitted by translate_rms_norm, followed by GGML SCALE.
    ثابت خپلکاری scale = ov::as_type_ptr<ov::op::v1::Multiply>(normalized.get_غوټه_shared_ptr());
    إذا (!scale) {
        أرجع خطأ;
    }
    ثابت خپلکاری ضریب = ov::as_type_ptr<ov::op::v0::Constant>(scale->get_ننوت_غوټه_shared_ptr(1));
    ثابت خپلکاری rms = ov::as_type_ptr<ov::op::v1::Multiply>(scale->get_ننوت_غوټه_shared_ptr(0));
    إذا (!ضریب || ov::shape_اندازه(ضریب->get_shape()) != 1 || !rms) {
        أرجع خطأ;
    }
    ثابت خپلکاری x = rms->ننوت_ارزښت(0);
    ثابت خپلکاری & shape = x.get_partial_shape();
    إذا (x.get_element_type() != ov::element::f32 || shape.rank() != 4 || shape[3].is_dynamic() ||
        shape[3].get_اوږدتیا() <= 0 || normalized.get_partial_shape() != shape) {
        أرجع خطأ;
    }
    ثابت عائم dim = static_cast<عائم>(shape[3].get_اوږدتیا());
    إذا (ضریب->cast_ویکټور<عائم>()[0] != 1.0f / std::جذر(dim)) {
        أرجع خطأ;
    }
    ثابت خپلکاری reciprocal = ov::as_type_ptr<ov::op::v1::Divide>(rms->get_ننوت_غوټه_shared_ptr(1));
    إذا (!reciprocal) {
        أرجع خطأ;
    }
    ثابت خپلکاری one = ov::as_type_ptr<ov::op::v0::Constant>(reciprocal->get_ننوت_غوټه_shared_ptr(0));
    ثابت خپلکاری ریښه = ov::as_type_ptr<ov::op::v0::Sqrt>(reciprocal->get_ننوت_غوټه_shared_ptr(1));
    إذا (!one || ov::shape_اندازه(one->get_shape()) != 1 || one->cast_ویکټور<عائم>()[0] != 1.0f || !ریښه) {
        أرجع خطأ;
    }
    ثابت خپلکاری جمع_کړه_2 = ov::as_type_ptr<ov::op::v1::جمع_کړه>(ریښه->get_ننوت_غوټه_shared_ptr(0));
    إذا (!جمع_کړه_2) {
        أرجع خطأ;
    }
    ثابت خپلکاری منځنۍ_2 = ov::as_type_ptr<ov::op::v1::Reduce_منځنۍ>(جمع_کړه_2->get_ننوت_غوټه_shared_ptr(0));
    ثابت خپلکاری rms_eps = ov::as_type_ptr<ov::op::v0::Constant>(جمع_کړه_2->get_ننوت_غوټه_shared_ptr(1));
    إذا (!منځنۍ_2 || !منځنۍ_2->get_keep_dims() || !rms_eps || ov::shape_اندازه(rms_eps->get_shape()) != 1) {
        أرجع خطأ;
    }
    ثابت خپلکاری axes = ov::as_type_ptr<ov::op::v0::Constant>(منځنۍ_2->get_ننوت_غوټه_shared_ptr(1));
    ثابت خپلکاری مربع = ov::as_type_ptr<ov::op::v1::Multiply>(منځنۍ_2->get_ننوت_غوټه_shared_ptr(0));
    إذا (!axes || axes->cast_ویکټور<int64_t>() != std::vector<int64_t>{-1} || !مربع ||
        مربع->ننوت_ارزښت(0) != x || مربع->ننوت_ارزښت(1) != x) {
        أرجع خطأ;
    }
    // RMSNorm(x, rms_eps) / sqrt(D) = x / sqrt(sum(x*x) + D*rms_eps).
    eps = dim * rms_eps->cast_ویکټور<عائم>()[0];
    إذا (!std::isfinite(eps) || eps <= 0.0f) {
        أرجع خطأ;
    }
    ننوت = x;
    أرجع ریښتیا;
}

Output_ویکټور translate_gated_delta_net(ثابت غوټه_Context & context) {
    خپلکاری v_shape = context.get_ننوت_shape(2).منزل_shape();  // [B, T, H_v, S_v]
    خپلکاری q_shape = context.get_ننوت_shape(0).منزل_shape();  // [B, T, H_k, S_k]

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

    خپلکاری q = context.get_ننوت(0);
    خپلکاری k = context.get_ننوت(1);
    خپلکاری v = process_view_ننوت(context, 2, H_v * S_v, 3);
    خپلکاری g = context.get_ننوت(3);
    خپلکاری beta = context.get_ننوت(4);
    خپلکاری state = context.get_ننوت(5);

    Output<Node> raw_q, raw_k;
    عائم q_eps = 1e-6f, k_eps = 1e-6f;
    ثابت منطقي fuse_qk_l2norm = ggml_openvino_is_gpu() &&
        سمون_gdn_l2_norm(q, raw_q, q_eps) && سمون_gdn_l2_norm(k, raw_k, k_eps);
    إذا (fuse_qk_l2norm) {
        // Keep head tiling below; GDN applies normalization per head and keeps its attention scale.
        q = raw_q;
        k = raw_k;
    }

    // ggml maps GQA heads in tiled order, while OV GDN maps repeated heads in grouped order.
    إذا (H_v != H_k) {
        ثابت int64_t repeat = H_v / H_k;
        خپلکاری repeats = ov::op::v0::Constant::create(ov::element::i64, {4}, std::vector<int64_t>{1, 1, repeat, 1});
        q = std::make_shared<ov::op::v0::Tile>(q, repeats);
        k = std::make_shared<ov::op::v0::Tile>(k, repeats);
    }

    إذا (context.get_view_ننوت_اندازه(2)) {
        // Same as l2_norm case 1
        v = std::make_shared<ov::op::v0::Squeeze>(v, ov::op::v0::Constant::create(ov::element::i64, {1}, {0}));
        خپلکاری v_shape = context.get_ننوت_shape(2).منزل_shape();
        std::vector<int64_t> reshape_نمونه = {0, 0, (int64_t) v_shape[2], (int64_t) v_shape[3]};
        v = std::make_shared<ov::op::v1::Reshape>(
            v, ov::op::v0::Constant::create(ov::element::i64, {4}, reshape_نمونه), ریښتیا);
    }

    // ggml state layout (OV notation): [B, H_v, value_dim, key_dim]
    // GatedDeltaNet op expects: [B, H_v, key_dim, value_dim]
    خپلکاری state_perm = ov::op::v0::Constant::create(ov::element::i64, {4}, std::vector<int64_t>{0, 1, 3, 2});
    state = std::make_shared<ov::op::v1::Transpose>(state, state_perm);

    g = std::make_shared<ov::op::v0::Squeeze>(g, ov::op::v0::Constant::create(ov::element::i64, {1}, {3}));
    beta = std::make_shared<ov::op::v0::Squeeze>(beta, ov::op::v0::Constant::create(ov::element::i64, {1}, {3}));

    إذا (context.has_ننوت("chunk_valid_len")) {
        // The last prefill chunk is padded with fabricated tokens. The recurrence is
        //   S_t = S_{t-1} * exp(g_t) + k_t (x) ((v_t - S_{t-1}^T k_t) * beta_t)
        // so forcing g = 0 and beta = 0 makes a padded step an exact identity and keeps the final
        // state equal to the state after the last real token. Attention output at those positions
        // is garbage but never read.
        ثابت خپلکاری & g_ps = g.get_partial_shape();
        FRONT_پای_ټکی_OP_CONVERSION_CHECK(g_ps.rank().is_static() && g_ps.rank().get_اوږدتیا() == 3 && g_ps[1].is_static(),
                                      "GATED_DELTA_NET pad masking requires a static token dimension");
        ثابت int64_t n_نښې = g_ps[1].get_اوږدتیا();
        std::vector<int64_t> positions(n_نښې);
        std::iota(positions.begin(), positions.end(), 0);
        خپلکاری valid = std::make_shared<ov::op::v1::Less>(
            ov::op::v0::Constant::create(ov::element::i64, {(size_t) n_نښې}, positions),
            context.get_ننوت("chunk_valid_len"));
        خپلکاری ماسک = std::make_shared<ov::op::v0::Unsqueeze>(
            std::make_shared<ov::op::v0::Convert>(valid, g.get_element_type()),
            ov::op::v0::Constant::create(ov::element::i64, {2}, std::vector<int64_t>{0, 2}));
        g = std::make_shared<ov::op::v1::Multiply>(g, ماسک);
        beta = std::make_shared<ov::op::v1::Multiply>(beta, ماسک);
    }

    // std::cout << "GatedDeltaNet input shapes: q=" << q.get_partial_shape() << ", k=" << k.get_partial_shape()
    //           << ", v=" << v.get_partial_shape() << ", g=" << g.get_partial_shape()
    //           << ", beta=" << beta.get_partial_shape() << ", state=" << state.get_partial_shape() << std::endl;

    خپلکاری gdn = std::make_shared<ov::op::internal::GatedDeltaNet>(q, k, v, state, g, beta, fuse_qk_l2norm, q_eps, k_eps);
    خپلکاری attn_4d = gdn->output(0);
    خپلکاری state_4d = gdn->output(1);  // [B, H_v, key_dim, value_dim]

    // std::cout << "GatedDeltaNet output shapes: attn=" << gdn->output(0).get_partial_shape()
    //           << ", new_state=" << gdn->output(1).get_partial_shape() << std::endl;

    // Transpose output state back to ggml layout [B, H_v, value_dim, key_dim]
    خپلکاری state_transposed = std::make_shared<ov::op::v1::Transpose>(state_4d, state_perm);
    إذا (context.get_output_نومونه().size() == 2) {
        // The canonical graph consumes the packed GGML result only through separate attention
        // and state VIEWs. Publish the native outputs under those VIEW names to avoid
        // flatten -> concat -> reshape -> slice -> reshape chains. This also works for B > 1.
        أرجع rename_outputs_with_suffix({attn_4d, state_transposed}, context.get_name());
    }

    خپلکاری flat_shape_1d = ov::op::v0::Constant::create(ov::element::i64, {1}, {-1});
    خپلکاری attn = std::make_shared<ov::op::v1::Reshape>(attn_4d, flat_shape_1d, خطأ);
    خپلکاری new_state = std::make_shared<ov::op::v1::Reshape>(state_transposed, flat_shape_1d, خطأ);
    خپلکاری packed = std::make_shared<ov::op::v0::Concat>(ov::Output_ویکټور{attn, new_state}, 0);
    خپلکاری بهر_shape = ov::op::v0::Constant::create(ov::element::i64, {4},
                                                  std::vector<int64_t>{1, 1, -1 /*T * B + S_v * B*/, S_v * H_v});
    خپلکاری res = std::make_shared<ov::op::v1::Reshape>(packed, بهر_shape, خطأ);

    أرجع rename_outputs_with_suffix({res}, context.get_name());
}

ساكن Output_ویکټور translate_gated_delta_net_ref(ثابت غوټه_Context & context) {
    عدد_inputs_check(context, 6, 6);

    // Inputs (OV shapes are reversed from ggml):
    // ggml: q[S_k, H_k, T, B], k[S_k, H_k, T, B], v[S_v, H_v, T, B]
    // OV:   q[B, T, H_k, S_k], k[B, T, H_k, S_k], v[B, T, H_v, S_v]
    // ggml: g[1 or S_v, H_v, T, B], beta[1, H_v, T, B]
    // OV:   g[B, T, H_v, 1 or S_v], beta[B, T, H_v, 1]
    // ggml: state[S_v, S_v, H_v, B]
    // OV:   state[B, H_v, S_v, S_v]
    خپلکاری q = process_view_ننوت_new(context, 0);
    خپلکاری k = process_view_ننوت_new(context, 1);
    خپلکاری v = process_view_ننوت_new(context, 2);
    خپلکاری g = process_view_ننوت_new(context, 3);
    خپلکاری beta = process_view_ننوت_new(context, 4);
    خپلکاری state = process_view_ننوت_new(context, 5);

    خپلکاری v_shape = context.get_ننوت_shape(2).منزل_shape();  // [B, T, H_v, S_v]
    خپلکاری q_shape = context.get_ننوت_shape(0).منزل_shape();  // [B, T, H_k, S_k]
    خپلکاری g_shape = context.get_ننوت_shape(3).منزل_shape();  // [B, T, H_v, 1 or S_v]

    ثابت int64_t B = v_shape[0];
    ثابت int64_t T = v_shape[1];
    ثابت int64_t H_v = v_shape[2];
    ثابت int64_t S_v = v_shape[3];
    ثابت int64_t H_k = q_shape[2];
    ثابت منطقي kda = (g_shape[3] == (size_t) S_v);

    ثابت int64_t rq1 = H_v / H_k;  // head repeat factor
    ثابت عائم scale = 1.0f / std::جذر((عائم) S_v);

    خپلکاری axis_1 = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
    خپلکاری axis_2 = ov::op::v0::Constant::create(ov::element::i64, {1}, {2});

    // Transpose inputs from [B, T, H, S] to [B, H, T, S] for easier per-head processing
    خپلکاری perm_0213 = ov::op::v0::Constant::create(ov::element::i64, {4}, std::vector<int64_t>{0, 2, 1, 3});
    خپلکاری q_t = std::make_shared<ov::op::v1::Transpose>(q, perm_0213);        // [B, H_k, T, S_k]
    خپلکاری k_t = std::make_shared<ov::op::v1::Transpose>(k, perm_0213);        // [B, H_k, T, S_k]
    خپلکاری v_t = std::make_shared<ov::op::v1::Transpose>(v, perm_0213);        // [B, H_v, T, S_v]
    خپلکاری g_t = std::make_shared<ov::op::v1::Transpose>(g, perm_0213);        // [B, H_v, T, 1 or S_v]
    خپلکاری beta_t = std::make_shared<ov::op::v1::Transpose>(beta, perm_0213);  // [B, H_v, T, 1]

    // Broadcast Q, K heads to match V heads if GQA is used (H_v > H_k)
    ov::Output<ov::Node> q_bh = q_t;
    ov::Output<ov::Node> k_bh = k_t;
    إذا (rq1 > 1) {
        خپلکاری q_unsq = std::make_shared<ov::op::v0::Unsqueeze>(q_t, axis_2);  // [B, H_k, 1, T, S]
        خپلکاری k_unsq = std::make_shared<ov::op::v0::Unsqueeze>(k_t, axis_2);  // [B, H_k, 1, T, S]

        خپلکاری bcast_shape = ov::op::v0::Constant::create(ov::element::i64, {5}, std::vector<int64_t>{1, 1, rq1, 1, 1});
        خپلکاری q_bcast =
            std::make_shared<ov::op::v3::خپرونه>(q_unsq, bcast_shape, ov::op::خپرونه_Type::BIDIRECTIONAL);
        خپلکاری k_bcast =
            std::make_shared<ov::op::v3::خپرونه>(k_unsq, bcast_shape, ov::op::خپرونه_Type::BIDIRECTIONAL);

        // Transpose [B, H_k, rq1, T, S] -> [B, rq1, H_k, T, S] so that reshape merges
        // as [rq1, H_k] giving repeat-blocks pattern matching CPU: iq1 = iv1 % H_k
        خپلکاری perm_5d = ov::op::v0::Constant::create(ov::element::i64, {5}, std::vector<int64_t>{0, 2, 1, 3, 4});
        خپلکاری q_transposed = std::make_shared<ov::op::v1::Transpose>(q_bcast, perm_5d);
        خپلکاری k_transposed = std::make_shared<ov::op::v1::Transpose>(k_bcast, perm_5d);

        خپلکاری new_shape = ov::op::v0::Constant::create(ov::element::i64, {4}, std::vector<int64_t>{B, H_v, T, S_v});
        q_bh = std::make_shared<ov::op::v1::Reshape>(q_transposed, new_shape, خطأ);
        k_bh = std::make_shared<ov::op::v1::Reshape>(k_transposed, new_shape, خطأ);
    }

    // Merge batch and head dims: [B*H_v, T, S_v]
    خپلکاری یوځای_bh = [&](ثابت ov::Output<ov::Node> & x, int64_t وروستی_dim) {
        خپلکاری shape = ov::op::v0::Constant::create(ov::element::i64, {3}, std::vector<int64_t>{B * H_v, T, وروستی_dim});
        أرجع std::make_shared<ov::op::v1::Reshape>(x, shape, خطأ);
    };

    خپلکاری q_m = یوځای_bh(q_bh, S_v);           // [B*H_v, T, S_v]
    خپلکاری k_m = یوځای_bh(k_bh, S_v);           // [B*H_v, T, S_v]
    خپلکاری v_m = یوځای_bh(v_t, S_v);            // [B*H_v, T, S_v]
    خپلکاری g_m = یوځای_bh(g_t, kda ? S_v : 1);  // [B*H_v, T, 1 or S_v]
    خپلکاری beta_m = یوځای_bh(beta_t, 1);        // [B*H_v, T, 1]

    // State: [B, H_v, S_v, S_v] -> [B*H_v, S_v, S_v]
    خپلکاری state_shape = ov::op::v0::Constant::create(ov::element::i64, {3}, std::vector<int64_t>{B * H_v, S_v, S_v});
    خپلکاری state_m = std::make_shared<ov::op::v1::Reshape>(state, state_shape, خطأ);

    خپلکاری scale_const = ov::op::v0::Constant::create(ov::element::f32, {}, std::vector<عائم>{scale});

    // --- Build Loop body ---
    // Body parameters (no iteration counter needed, use -1 in special ports)
    خپلکاری body_state = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic());
    خپلکاری body_q = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic());
    خپلکاری body_k = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic());
    خپلکاری body_v = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic());
    خپلکاری body_g = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic());
    خپلکاری body_beta = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic());
    خپلکاری body_iter = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::Shape{1});

    // Condition output (always true - we rely on trip_count for termination)
    خپلکاری body_cond_بهر = ov::op::v0::Constant::create(ov::element::boolean, ov::Shape{1}, std::vector<منطقي>{ریښتیا});

    // Gather current token from invariant inputs using iteration counter
    خپلکاری q_t_cur = std::make_shared<ov::op::v8::Gather>(body_q, body_iter, axis_1);     // [B*H_v, 1, S_v]
    خپلکاری k_t_cur = std::make_shared<ov::op::v8::Gather>(body_k, body_iter, axis_1);     // [B*H_v, 1, S_v]
    خپلکاری v_t_cur = std::make_shared<ov::op::v8::Gather>(body_v, body_iter, axis_1);     // [B*H_v, 1, S_v]
    خپلکاری g_t_cur = std::make_shared<ov::op::v8::Gather>(body_g, body_iter, axis_1);     // [B*H_v, 1, 1 or S_v]
    خپلکاری b_t_cur = std::make_shared<ov::op::v8::Gather>(body_beta, body_iter, axis_1);  // [B*H_v, 1, 1]

    // Squeeze token dim
    خپلکاری q_cur = std::make_shared<ov::op::v0::Squeeze>(q_t_cur, axis_1);  // [B*H_v, S_v]
    خپلکاری k_cur = std::make_shared<ov::op::v0::Squeeze>(k_t_cur, axis_1);  // [B*H_v, S_v]
    خپلکاری v_cur = std::make_shared<ov::op::v0::Squeeze>(v_t_cur, axis_1);  // [B*H_v, S_v]
    خپلکاری g_cur = std::make_shared<ov::op::v0::Squeeze>(g_t_cur, axis_1);  // [B*H_v, 1 or S_v]
    خپلکاری b_cur = std::make_shared<ov::op::v0::Squeeze>(b_t_cur, axis_1);  // [B*H_v, 1]

    // Step 1: Apply decay gate to state
    خپلکاری exp_g = std::make_shared<ov::op::v0::Exp>(g_cur);                                // [B*H_v, 1 or S_v]
    خپلکاری exp_g_unsq = std::make_shared<ov::op::v0::Unsqueeze>(exp_g, axis_1);             // [B*H_v, 1, 1 or S_v]
    خپلکاری state_decayed = std::make_shared<ov::op::v1::Multiply>(body_state, exp_g_unsq);  // [B*H_v, S_v, S_v]

    // Step 2: delta = (v - S @ k) * beta
    خپلکاری k_ستون = std::make_shared<ov::op::v0::Unsqueeze>(k_cur, axis_2);                 // [B*H_v, S_v, 1]
    خپلکاری sk = std::make_shared<ov::op::v0::MatMul>(state_decayed, k_ستون, خطأ, خطأ);  // [B*H_v, S_v, 1]
    خپلکاری sk_sq = std::make_shared<ov::op::v0::Squeeze>(sk, axis_2);                      // [B*H_v, S_v]
    خپلکاری v_minus_sk = std::make_shared<ov::op::v1::کمول>(v_cur, sk_sq);              // [B*H_v, S_v]
    خپلکاری delta = std::make_shared<ov::op::v1::Multiply>(v_minus_sk, b_cur);              // [B*H_v, S_v]

    // Step 3: state += outer(delta, k)
    خپلکاری delta_ستون = std::make_shared<ov::op::v0::Unsqueeze>(delta, axis_2);                 // [B*H_v, S_v, 1]
    خپلکاری k_لیکه = std::make_shared<ov::op::v0::Unsqueeze>(k_cur, axis_1);                     // [B*H_v, 1, S_v]
    خپلکاری outer_prod = std::make_shared<ov::op::v0::MatMul>(delta_ستون, k_لیکه, خطأ, خطأ);  // [B*H_v, S_v, S_v]
    خپلکاری state_updated = std::make_shared<ov::op::v1::جمع_کړه>(state_decayed, outer_prod);       // [B*H_v, S_v, S_v]

    // Step 4: attn_out = S @ q * scale
    خپلکاری q_ستون = std::make_shared<ov::op::v0::Unsqueeze>(q_cur, axis_2);                 // [B*H_v, S_v, 1]
    خپلکاری sq = std::make_shared<ov::op::v0::MatMul>(state_updated, q_ستون, خطأ, خطأ);  // [B*H_v, S_v, 1]
    خپلکاری sq_squeezed = std::make_shared<ov::op::v0::Squeeze>(sq, axis_2);                // [B*H_v, S_v]
    خپلکاری attn_بهر = std::make_shared<ov::op::v1::Multiply>(sq_squeezed, scale_const);    // [B*H_v, S_v]

    // Unsqueeze attn_out to [B*H_v, 1, S_v] for scan output concatenation
    خپلکاری attn_بهر_unsq = std::make_shared<ov::op::v0::Unsqueeze>(attn_بهر, axis_1);  // [B*H_v, 1, S_v]

    // --- Assemble Loop ---
    // Body: results = [condition, state_updated, attn_out_unsq]
    خپلکاری body = std::make_shared<ov::Model>(
        ov::Output_ویکټور{body_cond_بهر, state_updated, attn_بهر_unsq},
        ov::Parameter_ویکټور{body_iter, body_state, body_q, body_k, body_v, body_g, body_beta});

    خپلکاری trip_شمېر = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{T});
    خپلکاری exec_cond = ov::op::v0::Constant::create(ov::element::boolean, ov::Shape{1}, std::vector<منطقي>{ریښتیا});

    خپلکاری loop = std::make_shared<ov::op::v5::Loop>(trip_شمېر, exec_cond);
    loop->سېټ_function(body);
    loop->سېټ_special_body_ports(ov::op::v5::Loop::SpecialBodyPorts{0, 0});

    // Carried state: feeds back from body output 1 to body_state param
    loop->سېټ_merged_ننوت(body_state, state_m, state_updated);
    // Invariant inputs: passed through unchanged each iteration
    loop->سېټ_invariant_ننوت(body_q, q_m);
    loop->سېټ_invariant_ننوت(body_k, k_m);
    loop->سېټ_invariant_ننوت(body_v, v_m);
    loop->سېټ_invariant_ننوت(body_g, g_m);
    loop->سېټ_invariant_ننوت(body_beta, beta_m);

    // Loop outputs:
    // 1) Final state (last iteration value of state_updated)
    خپلکاری final_state_بهر = loop->get_iter_ارزښت(state_updated, -1);  // [B*H_v, S_v, S_v]
    // 2) Concatenated attention outputs across all iterations along axis 1
    خپلکاری attn_concat_بهر = loop->get_concatenated_slices(attn_بهر_unsq, 0, 1, 1, -1, 1);  // [B*H_v, T, S_v]

    // --- Pack outputs to match ggml layout ---
    // ggml output ne = {S_v*H, T*B + S_v*B, 1, 1} -> OV [1, 1, T*B+S_v*B, S_v*H_v]
    // attn: [B, T, H_v, S_v] row-major, state: [B, H_v, S_v, S_v] row-major

    // attn: [B*H_v, T, S_v] -> [B, H_v, T, S_v] -> transpose to [B, T, H_v, S_v] -> flatten
    خپلکاری attn_4d_shape = ov::op::v0::Constant::create(ov::element::i64, {4}, std::vector<int64_t>{B, H_v, T, S_v});
    خپلکاری attn_4d = std::make_shared<ov::op::v1::Reshape>(attn_concat_بهر, attn_4d_shape, خطأ);
    خپلکاری attn_perm = std::make_shared<ov::op::v1::Transpose>(attn_4d, perm_0213);  // [B, T, H_v, S_v]

    خپلکاری flat_shape_1d = ov::op::v0::Constant::create(ov::element::i64, {1}, std::vector<int64_t>{-1});
    خپلکاری attn_1d = std::make_shared<ov::op::v1::Reshape>(attn_perm, flat_shape_1d, خطأ);

    // state: [B*H_v, S_v, S_v] -> [B, H_v, S_v, S_v] -> flatten
    خپلکاری state_4d_shape = ov::op::v0::Constant::create(ov::element::i64, {4}, std::vector<int64_t>{B, H_v, S_v, S_v});
    خپلکاری state_4d = std::make_shared<ov::op::v1::Reshape>(final_state_بهر, state_4d_shape, خطأ);
    إذا (context.get_output_نومونه().size() == 2) {
        // Match the fused translator's direct attention/state contract.
        أرجع rename_outputs_with_suffix({attn_perm, state_4d}, context.get_name());
    }

    خپلکاری state_1d = std::make_shared<ov::op::v1::Reshape>(state_4d, flat_shape_1d, خطأ);

    // Concat [attn | state] and reshape to final output
    خپلکاری packed = std::make_shared<ov::op::v0::Concat>(ov::Output_ویکټور{attn_1d, state_1d}, 0);
    خپلکاری بهر_shape =
        ov::op::v0::Constant::create(ov::element::i64, {4}, std::vector<int64_t>{1, 1, T * B + S_v * B, S_v * H_v});
    خپلکاری res = std::make_shared<ov::op::v1::Reshape>(packed, بهر_shape, خطأ);

    أرجع rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
