#ਅਨ੍ਤਰ੍ਭਾਵ "gated_delta_net.hpp"

#ਅਨ੍ਤਰ੍ਭਾਵ "../node_context.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "../op_table.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "../utils.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-openvino/ggml-openvino-extra.h"

#ਅਨ੍ਤਰ੍ਭਾਵ <cmath>
#ਅਨ੍ਤਰ੍ਭਾਵ <cstdint>
#ਅਨ੍ਤਰ੍ਭਾਵ <memory>
#ਅਨ੍ਤਰ੍ਭਾਵ <numeric>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/add.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/broadcast.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/concat.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/constant.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/convert.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/divide.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/exp.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/gather.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/less.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/loop.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/matmul.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/multiply.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/reduce_mean.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/reshape.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/squeeze.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/sqrt.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/subtract.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/tile.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/transpose.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/unsqueeze.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <vector>

ਨਾਮਾਕਾਸ਼ ov {
ਨਾਮਾਕਾਸ਼ frontend {
ਨਾਮਾਕਾਸ਼ ggml {
ਨਾਮਾਕਾਸ਼ op {

ਸਥਿਰ Output_ਸਦਿਸ਼ translate_gated_delta_net_ref(ਅਚਲ ਗੰਢ_Context & context);

ਸਥਿਰ ਤਾਰਕਿਕ ਮੇਲ_gdn_l2_norm(ਅਚਲ Output<Node> & normalized, Output<Node> & ਨਿਵੇਸ਼, ਦਸ਼ਮਲਵ & eps) {
    // Match the RMSNorm decomposition emitted by translate_rms_norm, followed by GGML SCALE.
    ਅਚਲ ਸਵੈਚਾਲਿਤ scale = ov::as_type_ptr<ov::op::v1::Multiply>(normalized.get_ਗੰਢ_shared_ptr());
    ਜੇ (!scale) {
        ਵਾਪਸ ਝੂਠ;
    }
    ਅਚਲ ਸਵੈਚਾਲਿਤ ਗੁਣਕ = ov::as_type_ptr<ov::op::v0::Constant>(scale->get_ਨਿਵੇਸ਼_ਗੰਢ_shared_ptr(1));
    ਅਚਲ ਸਵੈਚਾਲਿਤ rms = ov::as_type_ptr<ov::op::v1::Multiply>(scale->get_ਨਿਵੇਸ਼_ਗੰਢ_shared_ptr(0));
    ਜੇ (!ਗੁਣਕ || ov::shape_ਮਾਪ(ਗੁਣਕ->get_shape()) != 1 || !rms) {
        ਵਾਪਸ ਝੂਠ;
    }
    ਅਚਲ ਸਵੈਚਾਲਿਤ x = rms->ਨਿਵੇਸ਼_ਮੁੱਲ(0);
    ਅਚਲ ਸਵੈਚਾਲਿਤ & shape = x.get_partial_shape();
    ਜੇ (x.get_element_type() != ov::element::f32 || shape.rank() != 4 || shape[3].is_dynamic() ||
        shape[3].get_ਲੰਮਾਈ() <= 0 || normalized.get_partial_shape() != shape) {
        ਵਾਪਸ ਝੂਠ;
    }
    ਅਚਲ ਦਸ਼ਮਲਵ dim = static_cast<ਦਸ਼ਮਲਵ>(shape[3].get_ਲੰਮਾਈ());
    ਜੇ (ਗੁਣਕ->cast_ਸਦਿਸ਼<ਦਸ਼ਮਲਵ>()[0] != 1.0f / std::ਵਰਗਮੂਲ(dim)) {
        ਵਾਪਸ ਝੂਠ;
    }
    ਅਚਲ ਸਵੈਚਾਲਿਤ reciprocal = ov::as_type_ptr<ov::op::v1::Divide>(rms->get_ਨਿਵੇਸ਼_ਗੰਢ_shared_ptr(1));
    ਜੇ (!reciprocal) {
        ਵਾਪਸ ਝੂਠ;
    }
    ਅਚਲ ਸਵੈਚਾਲਿਤ one = ov::as_type_ptr<ov::op::v0::Constant>(reciprocal->get_ਨਿਵੇਸ਼_ਗੰਢ_shared_ptr(0));
    ਅਚਲ ਸਵੈਚਾਲਿਤ ਜੜ੍ਹ = ov::as_type_ptr<ov::op::v0::Sqrt>(reciprocal->get_ਨਿਵੇਸ਼_ਗੰਢ_shared_ptr(1));
    ਜੇ (!one || ov::shape_ਮਾਪ(one->get_shape()) != 1 || one->cast_ਸਦਿਸ਼<ਦਸ਼ਮਲਵ>()[0] != 1.0f || !ਜੜ੍ਹ) {
        ਵਾਪਸ ਝੂਠ;
    }
    ਅਚਲ ਸਵੈਚਾਲਿਤ ਜੋੜੋ_2 = ov::as_type_ptr<ov::op::v1::ਜੋੜੋ>(ਜੜ੍ਹ->get_ਨਿਵੇਸ਼_ਗੰਢ_shared_ptr(0));
    ਜੇ (!ਜੋੜੋ_2) {
        ਵਾਪਸ ਝੂਠ;
    }
    ਅਚਲ ਸਵੈਚਾਲਿਤ ਔਸਤ_2 = ov::as_type_ptr<ov::op::v1::Reduce_ਔਸਤ>(ਜੋੜੋ_2->get_ਨਿਵੇਸ਼_ਗੰਢ_shared_ptr(0));
    ਅਚਲ ਸਵੈਚਾਲਿਤ rms_eps = ov::as_type_ptr<ov::op::v0::Constant>(ਜੋੜੋ_2->get_ਨਿਵੇਸ਼_ਗੰਢ_shared_ptr(1));
    ਜੇ (!ਔਸਤ_2 || !ਔਸਤ_2->get_keep_dims() || !rms_eps || ov::shape_ਮਾਪ(rms_eps->get_shape()) != 1) {
        ਵਾਪਸ ਝੂਠ;
    }
    ਅਚਲ ਸਵੈਚਾਲਿਤ axes = ov::as_type_ptr<ov::op::v0::Constant>(ਔਸਤ_2->get_ਨਿਵੇਸ਼_ਗੰਢ_shared_ptr(1));
    ਅਚਲ ਸਵੈਚਾਲਿਤ ਵਰਗਫਲ = ov::as_type_ptr<ov::op::v1::Multiply>(ਔਸਤ_2->get_ਨਿਵੇਸ਼_ਗੰਢ_shared_ptr(0));
    ਜੇ (!axes || axes->cast_ਸਦਿਸ਼<int64_t>() != std::vector<int64_t>{-1} || !ਵਰਗਫਲ ||
        ਵਰਗਫਲ->ਨਿਵੇਸ਼_ਮੁੱਲ(0) != x || ਵਰਗਫਲ->ਨਿਵੇਸ਼_ਮੁੱਲ(1) != x) {
        ਵਾਪਸ ਝੂਠ;
    }
    // RMSNorm(x, rms_eps) / sqrt(D) = x / sqrt(sum(x*x) + D*rms_eps).
    eps = dim * rms_eps->cast_ਸਦਿਸ਼<ਦਸ਼ਮਲਵ>()[0];
    ਜੇ (!std::isfinite(eps) || eps <= 0.0f) {
        ਵਾਪਸ ਝੂਠ;
    }
    ਨਿਵੇਸ਼ = x;
    ਵਾਪਸ ਸੱਚ;
}

Output_ਸਦਿਸ਼ translate_gated_delta_net(ਅਚਲ ਗੰਢ_Context & context) {
    ਸਵੈਚਾਲਿਤ v_shape = context.get_ਨਿਵੇਸ਼_shape(2).ਮੰਜ਼ਿਲ_shape();  // [B, T, H_v, S_v]
    ਸਵੈਚਾਲਿਤ q_shape = context.get_ਨਿਵੇਸ਼_shape(0).ਮੰਜ਼ਿਲ_shape();  // [B, T, H_k, S_k]

    // Fused GatedDeltaNet op only supports scalar gate (kda=0).
    // Fall back to reference implementation for per-key-dimension gating.
    // if (kda) {
    //     return translate_gated_delta_net_ref(context);
    // }

    // const int64_t B = v_shape[0];
    // const int64_t T = v_shape[1];
    ਅਚਲ int64_t H_v = v_shape[2];
    ਅਚਲ int64_t S_v = v_shape[3];
    ਅਚਲ int64_t H_k = q_shape[2];
    // const int64_t S_k = q_shape[3];

    ਸਵੈਚਾਲਿਤ q = context.get_ਨਿਵੇਸ਼(0);
    ਸਵੈਚਾਲਿਤ k = context.get_ਨਿਵੇਸ਼(1);
    ਸਵੈਚਾਲਿਤ v = process_view_ਨਿਵੇਸ਼(context, 2, H_v * S_v, 3);
    ਸਵੈਚਾਲਿਤ g = context.get_ਨਿਵੇਸ਼(3);
    ਸਵੈਚਾਲਿਤ beta = context.get_ਨਿਵੇਸ਼(4);
    ਸਵੈਚਾਲਿਤ state = context.get_ਨਿਵੇਸ਼(5);

    Output<Node> raw_q, raw_k;
    ਦਸ਼ਮਲਵ q_eps = 1e-6f, k_eps = 1e-6f;
    ਅਚਲ ਤਾਰਕਿਕ fuse_qk_l2norm = ggml_openvino_is_gpu() &&
        ਮੇਲ_gdn_l2_norm(q, raw_q, q_eps) && ਮੇਲ_gdn_l2_norm(k, raw_k, k_eps);
    ਜੇ (fuse_qk_l2norm) {
        // Keep head tiling below; GDN applies normalization per head and keeps its attention scale.
        q = raw_q;
        k = raw_k;
    }

    // ggml maps GQA heads in tiled order, while OV GDN maps repeated heads in grouped order.
    ਜੇ (H_v != H_k) {
        ਅਚਲ int64_t repeat = H_v / H_k;
        ਸਵੈਚਾਲਿਤ repeats = ov::op::v0::Constant::create(ov::element::i64, {4}, std::vector<int64_t>{1, 1, repeat, 1});
        q = std::make_shared<ov::op::v0::Tile>(q, repeats);
        k = std::make_shared<ov::op::v0::Tile>(k, repeats);
    }

    ਜੇ (context.get_view_ਨਿਵੇਸ਼_ਮਾਪ(2)) {
        // Same as l2_norm case 1
        v = std::make_shared<ov::op::v0::Squeeze>(v, ov::op::v0::Constant::create(ov::element::i64, {1}, {0}));
        ਸਵੈਚਾਲਿਤ v_shape = context.get_ਨਿਵੇਸ਼_shape(2).ਮੰਜ਼ਿਲ_shape();
        std::vector<int64_t> reshape_ਨਮੂਨਾ = {0, 0, (int64_t) v_shape[2], (int64_t) v_shape[3]};
        v = std::make_shared<ov::op::v1::Reshape>(
            v, ov::op::v0::Constant::create(ov::element::i64, {4}, reshape_ਨਮੂਨਾ), ਸੱਚ);
    }

    // ggml state layout (OV notation): [B, H_v, value_dim, key_dim]
    // GatedDeltaNet op expects: [B, H_v, key_dim, value_dim]
    ਸਵੈਚਾਲਿਤ state_perm = ov::op::v0::Constant::create(ov::element::i64, {4}, std::vector<int64_t>{0, 1, 3, 2});
    state = std::make_shared<ov::op::v1::Transpose>(state, state_perm);

    g = std::make_shared<ov::op::v0::Squeeze>(g, ov::op::v0::Constant::create(ov::element::i64, {1}, {3}));
    beta = std::make_shared<ov::op::v0::Squeeze>(beta, ov::op::v0::Constant::create(ov::element::i64, {1}, {3}));

    ਜੇ (context.has_ਨਿਵੇਸ਼("chunk_valid_len")) {
        // The last prefill chunk is padded with fabricated tokens. The recurrence is
        //   S_t = S_{t-1} * exp(g_t) + k_t (x) ((v_t - S_{t-1}^T k_t) * beta_t)
        // so forcing g = 0 and beta = 0 makes a padded step an exact identity and keeps the final
        // state equal to the state after the last real token. Attention output at those positions
        // is garbage but never read.
        ਅਚਲ ਸਵੈਚਾਲਿਤ & g_ps = g.get_partial_shape();
        FRONT_ਸਿਰਾ_OP_CONVERSION_CHECK(g_ps.rank().is_static() && g_ps.rank().get_ਲੰਮਾਈ() == 3 && g_ps[1].is_static(),
                                      "GATED_DELTA_NET pad masking requires a static token dimension");
        ਅਚਲ int64_t n_ਟੋਕਨ = g_ps[1].get_ਲੰਮਾਈ();
        std::vector<int64_t> positions(n_ਟੋਕਨ);
        std::iota(positions.begin(), positions.end(), 0);
        ਸਵੈਚਾਲਿਤ valid = std::make_shared<ov::op::v1::Less>(
            ov::op::v0::Constant::create(ov::element::i64, {(size_t) n_ਟੋਕਨ}, positions),
            context.get_ਨਿਵੇਸ਼("chunk_valid_len"));
        ਸਵੈਚਾਲਿਤ ਨਕਾਬ = std::make_shared<ov::op::v0::Unsqueeze>(
            std::make_shared<ov::op::v0::Convert>(valid, g.get_element_type()),
            ov::op::v0::Constant::create(ov::element::i64, {2}, std::vector<int64_t>{0, 2}));
        g = std::make_shared<ov::op::v1::Multiply>(g, ਨਕਾਬ);
        beta = std::make_shared<ov::op::v1::Multiply>(beta, ਨਕਾਬ);
    }

    // std::cout << "GatedDeltaNet input shapes: q=" << q.get_partial_shape() << ", k=" << k.get_partial_shape()
    //           << ", v=" << v.get_partial_shape() << ", g=" << g.get_partial_shape()
    //           << ", beta=" << beta.get_partial_shape() << ", state=" << state.get_partial_shape() << std::endl;

    ਸਵੈਚਾਲਿਤ gdn = std::make_shared<ov::op::internal::GatedDeltaNet>(q, k, v, state, g, beta, fuse_qk_l2norm, q_eps, k_eps);
    ਸਵੈਚਾਲਿਤ attn_4d = gdn->output(0);
    ਸਵੈਚਾਲਿਤ state_4d = gdn->output(1);  // [B, H_v, key_dim, value_dim]

    // std::cout << "GatedDeltaNet output shapes: attn=" << gdn->output(0).get_partial_shape()
    //           << ", new_state=" << gdn->output(1).get_partial_shape() << std::endl;

    // Transpose output state back to ggml layout [B, H_v, value_dim, key_dim]
    ਸਵੈਚਾਲਿਤ state_transposed = std::make_shared<ov::op::v1::Transpose>(state_4d, state_perm);
    ਜੇ (context.get_output_ਨਾਂ().size() == 2) {
        // The canonical graph consumes the packed GGML result only through separate attention
        // and state VIEWs. Publish the native outputs under those VIEW names to avoid
        // flatten -> concat -> reshape -> slice -> reshape chains. This also works for B > 1.
        ਵਾਪਸ rename_outputs_with_suffix({attn_4d, state_transposed}, context.get_name());
    }

    ਸਵੈਚਾਲਿਤ flat_shape_1d = ov::op::v0::Constant::create(ov::element::i64, {1}, {-1});
    ਸਵੈਚਾਲਿਤ attn = std::make_shared<ov::op::v1::Reshape>(attn_4d, flat_shape_1d, ਝੂਠ);
    ਸਵੈਚਾਲਿਤ new_state = std::make_shared<ov::op::v1::Reshape>(state_transposed, flat_shape_1d, ਝੂਠ);
    ਸਵੈਚਾਲਿਤ packed = std::make_shared<ov::op::v0::Concat>(ov::Output_ਸਦਿਸ਼{attn, new_state}, 0);
    ਸਵੈਚਾਲਿਤ ਬਾਹਰਲਾ_shape = ov::op::v0::Constant::create(ov::element::i64, {4},
                                                  std::vector<int64_t>{1, 1, -1 /*T * B + S_v * B*/, S_v * H_v});
    ਸਵੈਚਾਲਿਤ res = std::make_shared<ov::op::v1::Reshape>(packed, ਬਾਹਰਲਾ_shape, ਝੂਠ);

    ਵਾਪਸ rename_outputs_with_suffix({res}, context.get_name());
}

ਸਥਿਰ Output_ਸਦਿਸ਼ translate_gated_delta_net_ref(ਅਚਲ ਗੰਢ_Context & context) {
    ਅੰਕ_inputs_check(context, 6, 6);

    // Inputs (OV shapes are reversed from ggml):
    // ggml: q[S_k, H_k, T, B], k[S_k, H_k, T, B], v[S_v, H_v, T, B]
    // OV:   q[B, T, H_k, S_k], k[B, T, H_k, S_k], v[B, T, H_v, S_v]
    // ggml: g[1 or S_v, H_v, T, B], beta[1, H_v, T, B]
    // OV:   g[B, T, H_v, 1 or S_v], beta[B, T, H_v, 1]
    // ggml: state[S_v, S_v, H_v, B]
    // OV:   state[B, H_v, S_v, S_v]
    ਸਵੈਚਾਲਿਤ q = process_view_ਨਿਵੇਸ਼_new(context, 0);
    ਸਵੈਚਾਲਿਤ k = process_view_ਨਿਵੇਸ਼_new(context, 1);
    ਸਵੈਚਾਲਿਤ v = process_view_ਨਿਵੇਸ਼_new(context, 2);
    ਸਵੈਚਾਲਿਤ g = process_view_ਨਿਵੇਸ਼_new(context, 3);
    ਸਵੈਚਾਲਿਤ beta = process_view_ਨਿਵੇਸ਼_new(context, 4);
    ਸਵੈਚਾਲਿਤ state = process_view_ਨਿਵੇਸ਼_new(context, 5);

    ਸਵੈਚਾਲਿਤ v_shape = context.get_ਨਿਵੇਸ਼_shape(2).ਮੰਜ਼ਿਲ_shape();  // [B, T, H_v, S_v]
    ਸਵੈਚਾਲਿਤ q_shape = context.get_ਨਿਵੇਸ਼_shape(0).ਮੰਜ਼ਿਲ_shape();  // [B, T, H_k, S_k]
    ਸਵੈਚਾਲਿਤ g_shape = context.get_ਨਿਵੇਸ਼_shape(3).ਮੰਜ਼ਿਲ_shape();  // [B, T, H_v, 1 or S_v]

    ਅਚਲ int64_t B = v_shape[0];
    ਅਚਲ int64_t T = v_shape[1];
    ਅਚਲ int64_t H_v = v_shape[2];
    ਅਚਲ int64_t S_v = v_shape[3];
    ਅਚਲ int64_t H_k = q_shape[2];
    ਅਚਲ ਤਾਰਕਿਕ kda = (g_shape[3] == (size_t) S_v);

    ਅਚਲ int64_t rq1 = H_v / H_k;  // head repeat factor
    ਅਚਲ ਦਸ਼ਮਲਵ scale = 1.0f / std::ਵਰਗਮੂਲ((ਦਸ਼ਮਲਵ) S_v);

    ਸਵੈਚਾਲਿਤ axis_1 = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
    ਸਵੈਚਾਲਿਤ axis_2 = ov::op::v0::Constant::create(ov::element::i64, {1}, {2});

    // Transpose inputs from [B, T, H, S] to [B, H, T, S] for easier per-head processing
    ਸਵੈਚਾਲਿਤ perm_0213 = ov::op::v0::Constant::create(ov::element::i64, {4}, std::vector<int64_t>{0, 2, 1, 3});
    ਸਵੈਚਾਲਿਤ q_t = std::make_shared<ov::op::v1::Transpose>(q, perm_0213);        // [B, H_k, T, S_k]
    ਸਵੈਚਾਲਿਤ k_t = std::make_shared<ov::op::v1::Transpose>(k, perm_0213);        // [B, H_k, T, S_k]
    ਸਵੈਚਾਲਿਤ v_t = std::make_shared<ov::op::v1::Transpose>(v, perm_0213);        // [B, H_v, T, S_v]
    ਸਵੈਚਾਲਿਤ g_t = std::make_shared<ov::op::v1::Transpose>(g, perm_0213);        // [B, H_v, T, 1 or S_v]
    ਸਵੈਚਾਲਿਤ beta_t = std::make_shared<ov::op::v1::Transpose>(beta, perm_0213);  // [B, H_v, T, 1]

    // Broadcast Q, K heads to match V heads if GQA is used (H_v > H_k)
    ov::Output<ov::Node> q_bh = q_t;
    ov::Output<ov::Node> k_bh = k_t;
    ਜੇ (rq1 > 1) {
        ਸਵੈਚਾਲਿਤ q_unsq = std::make_shared<ov::op::v0::Unsqueeze>(q_t, axis_2);  // [B, H_k, 1, T, S]
        ਸਵੈਚਾਲਿਤ k_unsq = std::make_shared<ov::op::v0::Unsqueeze>(k_t, axis_2);  // [B, H_k, 1, T, S]

        ਸਵੈਚਾਲਿਤ bcast_shape = ov::op::v0::Constant::create(ov::element::i64, {5}, std::vector<int64_t>{1, 1, rq1, 1, 1});
        ਸਵੈਚਾਲਿਤ q_bcast =
            std::make_shared<ov::op::v3::ਪ੍ਰਸਾਰਣ>(q_unsq, bcast_shape, ov::op::ਪ੍ਰਸਾਰਣ_Type::BIDIRECTIONAL);
        ਸਵੈਚਾਲਿਤ k_bcast =
            std::make_shared<ov::op::v3::ਪ੍ਰਸਾਰਣ>(k_unsq, bcast_shape, ov::op::ਪ੍ਰਸਾਰਣ_Type::BIDIRECTIONAL);

        // Transpose [B, H_k, rq1, T, S] -> [B, rq1, H_k, T, S] so that reshape merges
        // as [rq1, H_k] giving repeat-blocks pattern matching CPU: iq1 = iv1 % H_k
        ਸਵੈਚਾਲਿਤ perm_5d = ov::op::v0::Constant::create(ov::element::i64, {5}, std::vector<int64_t>{0, 2, 1, 3, 4});
        ਸਵੈਚਾਲਿਤ q_transposed = std::make_shared<ov::op::v1::Transpose>(q_bcast, perm_5d);
        ਸਵੈਚਾਲਿਤ k_transposed = std::make_shared<ov::op::v1::Transpose>(k_bcast, perm_5d);

        ਸਵੈਚਾਲਿਤ new_shape = ov::op::v0::Constant::create(ov::element::i64, {4}, std::vector<int64_t>{B, H_v, T, S_v});
        q_bh = std::make_shared<ov::op::v1::Reshape>(q_transposed, new_shape, ਝੂਠ);
        k_bh = std::make_shared<ov::op::v1::Reshape>(k_transposed, new_shape, ਝੂਠ);
    }

    // Merge batch and head dims: [B*H_v, T, S_v]
    ਸਵੈਚਾਲਿਤ ਮਿਲਾਓ_bh = [&](ਅਚਲ ov::Output<ov::Node> & x, int64_t ਆਖਰੀ_dim) {
        ਸਵੈਚਾਲਿਤ shape = ov::op::v0::Constant::create(ov::element::i64, {3}, std::vector<int64_t>{B * H_v, T, ਆਖਰੀ_dim});
        ਵਾਪਸ std::make_shared<ov::op::v1::Reshape>(x, shape, ਝੂਠ);
    };

    ਸਵੈਚਾਲਿਤ q_m = ਮਿਲਾਓ_bh(q_bh, S_v);           // [B*H_v, T, S_v]
    ਸਵੈਚਾਲਿਤ k_m = ਮਿਲਾਓ_bh(k_bh, S_v);           // [B*H_v, T, S_v]
    ਸਵੈਚਾਲਿਤ v_m = ਮਿਲਾਓ_bh(v_t, S_v);            // [B*H_v, T, S_v]
    ਸਵੈਚਾਲਿਤ g_m = ਮਿਲਾਓ_bh(g_t, kda ? S_v : 1);  // [B*H_v, T, 1 or S_v]
    ਸਵੈਚਾਲਿਤ beta_m = ਮਿਲਾਓ_bh(beta_t, 1);        // [B*H_v, T, 1]

    // State: [B, H_v, S_v, S_v] -> [B*H_v, S_v, S_v]
    ਸਵੈਚਾਲਿਤ state_shape = ov::op::v0::Constant::create(ov::element::i64, {3}, std::vector<int64_t>{B * H_v, S_v, S_v});
    ਸਵੈਚਾਲਿਤ state_m = std::make_shared<ov::op::v1::Reshape>(state, state_shape, ਝੂਠ);

    ਸਵੈਚਾਲਿਤ scale_const = ov::op::v0::Constant::create(ov::element::f32, {}, std::vector<ਦਸ਼ਮਲਵ>{scale});

    // --- Build Loop body ---
    // Body parameters (no iteration counter needed, use -1 in special ports)
    ਸਵੈਚਾਲਿਤ body_state = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic());
    ਸਵੈਚਾਲਿਤ body_q = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic());
    ਸਵੈਚਾਲਿਤ body_k = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic());
    ਸਵੈਚਾਲਿਤ body_v = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic());
    ਸਵੈਚਾਲਿਤ body_g = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic());
    ਸਵੈਚਾਲਿਤ body_beta = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic());
    ਸਵੈਚਾਲਿਤ body_iter = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::Shape{1});

    // Condition output (always true - we rely on trip_count for termination)
    ਸਵੈਚਾਲਿਤ body_cond_ਬਾਹਰਲਾ = ov::op::v0::Constant::create(ov::element::boolean, ov::Shape{1}, std::vector<ਤਾਰਕਿਕ>{ਸੱਚ});

    // Gather current token from invariant inputs using iteration counter
    ਸਵੈਚਾਲਿਤ q_t_cur = std::make_shared<ov::op::v8::Gather>(body_q, body_iter, axis_1);     // [B*H_v, 1, S_v]
    ਸਵੈਚਾਲਿਤ k_t_cur = std::make_shared<ov::op::v8::Gather>(body_k, body_iter, axis_1);     // [B*H_v, 1, S_v]
    ਸਵੈਚਾਲਿਤ v_t_cur = std::make_shared<ov::op::v8::Gather>(body_v, body_iter, axis_1);     // [B*H_v, 1, S_v]
    ਸਵੈਚਾਲਿਤ g_t_cur = std::make_shared<ov::op::v8::Gather>(body_g, body_iter, axis_1);     // [B*H_v, 1, 1 or S_v]
    ਸਵੈਚਾਲਿਤ b_t_cur = std::make_shared<ov::op::v8::Gather>(body_beta, body_iter, axis_1);  // [B*H_v, 1, 1]

    // Squeeze token dim
    ਸਵੈਚਾਲਿਤ q_cur = std::make_shared<ov::op::v0::Squeeze>(q_t_cur, axis_1);  // [B*H_v, S_v]
    ਸਵੈਚਾਲਿਤ k_cur = std::make_shared<ov::op::v0::Squeeze>(k_t_cur, axis_1);  // [B*H_v, S_v]
    ਸਵੈਚਾਲਿਤ v_cur = std::make_shared<ov::op::v0::Squeeze>(v_t_cur, axis_1);  // [B*H_v, S_v]
    ਸਵੈਚਾਲਿਤ g_cur = std::make_shared<ov::op::v0::Squeeze>(g_t_cur, axis_1);  // [B*H_v, 1 or S_v]
    ਸਵੈਚਾਲਿਤ b_cur = std::make_shared<ov::op::v0::Squeeze>(b_t_cur, axis_1);  // [B*H_v, 1]

    // Step 1: Apply decay gate to state
    ਸਵੈਚਾਲਿਤ exp_g = std::make_shared<ov::op::v0::Exp>(g_cur);                                // [B*H_v, 1 or S_v]
    ਸਵੈਚਾਲਿਤ exp_g_unsq = std::make_shared<ov::op::v0::Unsqueeze>(exp_g, axis_1);             // [B*H_v, 1, 1 or S_v]
    ਸਵੈਚਾਲਿਤ state_decayed = std::make_shared<ov::op::v1::Multiply>(body_state, exp_g_unsq);  // [B*H_v, S_v, S_v]

    // Step 2: delta = (v - S @ k) * beta
    ਸਵੈਚਾਲਿਤ k_ਖਾਨਾ = std::make_shared<ov::op::v0::Unsqueeze>(k_cur, axis_2);                 // [B*H_v, S_v, 1]
    ਸਵੈਚਾਲਿਤ sk = std::make_shared<ov::op::v0::MatMul>(state_decayed, k_ਖਾਨਾ, ਝੂਠ, ਝੂਠ);  // [B*H_v, S_v, 1]
    ਸਵੈਚਾਲਿਤ sk_sq = std::make_shared<ov::op::v0::Squeeze>(sk, axis_2);                      // [B*H_v, S_v]
    ਸਵੈਚਾਲਿਤ v_minus_sk = std::make_shared<ov::op::v1::ਘਟਾਓ>(v_cur, sk_sq);              // [B*H_v, S_v]
    ਸਵੈਚਾਲਿਤ delta = std::make_shared<ov::op::v1::Multiply>(v_minus_sk, b_cur);              // [B*H_v, S_v]

    // Step 3: state += outer(delta, k)
    ਸਵੈਚਾਲਿਤ delta_ਖਾਨਾ = std::make_shared<ov::op::v0::Unsqueeze>(delta, axis_2);                 // [B*H_v, S_v, 1]
    ਸਵੈਚਾਲਿਤ k_ਪੰਕਤੀ = std::make_shared<ov::op::v0::Unsqueeze>(k_cur, axis_1);                     // [B*H_v, 1, S_v]
    ਸਵੈਚਾਲਿਤ outer_prod = std::make_shared<ov::op::v0::MatMul>(delta_ਖਾਨਾ, k_ਪੰਕਤੀ, ਝੂਠ, ਝੂਠ);  // [B*H_v, S_v, S_v]
    ਸਵੈਚਾਲਿਤ state_updated = std::make_shared<ov::op::v1::ਜੋੜੋ>(state_decayed, outer_prod);       // [B*H_v, S_v, S_v]

    // Step 4: attn_out = S @ q * scale
    ਸਵੈਚਾਲਿਤ q_ਖਾਨਾ = std::make_shared<ov::op::v0::Unsqueeze>(q_cur, axis_2);                 // [B*H_v, S_v, 1]
    ਸਵੈਚਾਲਿਤ sq = std::make_shared<ov::op::v0::MatMul>(state_updated, q_ਖਾਨਾ, ਝੂਠ, ਝੂਠ);  // [B*H_v, S_v, 1]
    ਸਵੈਚਾਲਿਤ sq_squeezed = std::make_shared<ov::op::v0::Squeeze>(sq, axis_2);                // [B*H_v, S_v]
    ਸਵੈਚਾਲਿਤ attn_ਬਾਹਰਲਾ = std::make_shared<ov::op::v1::Multiply>(sq_squeezed, scale_const);    // [B*H_v, S_v]

    // Unsqueeze attn_out to [B*H_v, 1, S_v] for scan output concatenation
    ਸਵੈਚਾਲਿਤ attn_ਬਾਹਰਲਾ_unsq = std::make_shared<ov::op::v0::Unsqueeze>(attn_ਬਾਹਰਲਾ, axis_1);  // [B*H_v, 1, S_v]

    // --- Assemble Loop ---
    // Body: results = [condition, state_updated, attn_out_unsq]
    ਸਵੈਚਾਲਿਤ body = std::make_shared<ov::Model>(
        ov::Output_ਸਦਿਸ਼{body_cond_ਬਾਹਰਲਾ, state_updated, attn_ਬਾਹਰਲਾ_unsq},
        ov::Parameter_ਸਦਿਸ਼{body_iter, body_state, body_q, body_k, body_v, body_g, body_beta});

    ਸਵੈਚਾਲਿਤ trip_ਗਿਣਤੀ = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{T});
    ਸਵੈਚਾਲਿਤ exec_cond = ov::op::v0::Constant::create(ov::element::boolean, ov::Shape{1}, std::vector<ਤਾਰਕਿਕ>{ਸੱਚ});

    ਸਵੈਚਾਲਿਤ loop = std::make_shared<ov::op::v5::Loop>(trip_ਗਿਣਤੀ, exec_cond);
    loop->ਸੈੱਟ_function(body);
    loop->ਸੈੱਟ_special_body_ports(ov::op::v5::Loop::SpecialBodyPorts{0, 0});

    // Carried state: feeds back from body output 1 to body_state param
    loop->ਸੈੱਟ_merged_ਨਿਵੇਸ਼(body_state, state_m, state_updated);
    // Invariant inputs: passed through unchanged each iteration
    loop->ਸੈੱਟ_invariant_ਨਿਵੇਸ਼(body_q, q_m);
    loop->ਸੈੱਟ_invariant_ਨਿਵੇਸ਼(body_k, k_m);
    loop->ਸੈੱਟ_invariant_ਨਿਵੇਸ਼(body_v, v_m);
    loop->ਸੈੱਟ_invariant_ਨਿਵੇਸ਼(body_g, g_m);
    loop->ਸੈੱਟ_invariant_ਨਿਵੇਸ਼(body_beta, beta_m);

    // Loop outputs:
    // 1) Final state (last iteration value of state_updated)
    ਸਵੈਚਾਲਿਤ final_state_ਬਾਹਰਲਾ = loop->get_iter_ਮੁੱਲ(state_updated, -1);  // [B*H_v, S_v, S_v]
    // 2) Concatenated attention outputs across all iterations along axis 1
    ਸਵੈਚਾਲਿਤ attn_concat_ਬਾਹਰਲਾ = loop->get_concatenated_slices(attn_ਬਾਹਰਲਾ_unsq, 0, 1, 1, -1, 1);  // [B*H_v, T, S_v]

    // --- Pack outputs to match ggml layout ---
    // ggml output ne = {S_v*H, T*B + S_v*B, 1, 1} -> OV [1, 1, T*B+S_v*B, S_v*H_v]
    // attn: [B, T, H_v, S_v] row-major, state: [B, H_v, S_v, S_v] row-major

    // attn: [B*H_v, T, S_v] -> [B, H_v, T, S_v] -> transpose to [B, T, H_v, S_v] -> flatten
    ਸਵੈਚਾਲਿਤ attn_4d_shape = ov::op::v0::Constant::create(ov::element::i64, {4}, std::vector<int64_t>{B, H_v, T, S_v});
    ਸਵੈਚਾਲਿਤ attn_4d = std::make_shared<ov::op::v1::Reshape>(attn_concat_ਬਾਹਰਲਾ, attn_4d_shape, ਝੂਠ);
    ਸਵੈਚਾਲਿਤ attn_perm = std::make_shared<ov::op::v1::Transpose>(attn_4d, perm_0213);  // [B, T, H_v, S_v]

    ਸਵੈਚਾਲਿਤ flat_shape_1d = ov::op::v0::Constant::create(ov::element::i64, {1}, std::vector<int64_t>{-1});
    ਸਵੈਚਾਲਿਤ attn_1d = std::make_shared<ov::op::v1::Reshape>(attn_perm, flat_shape_1d, ਝੂਠ);

    // state: [B*H_v, S_v, S_v] -> [B, H_v, S_v, S_v] -> flatten
    ਸਵੈਚਾਲਿਤ state_4d_shape = ov::op::v0::Constant::create(ov::element::i64, {4}, std::vector<int64_t>{B, H_v, S_v, S_v});
    ਸਵੈਚਾਲਿਤ state_4d = std::make_shared<ov::op::v1::Reshape>(final_state_ਬਾਹਰਲਾ, state_4d_shape, ਝੂਠ);
    ਜੇ (context.get_output_ਨਾਂ().size() == 2) {
        // Match the fused translator's direct attention/state contract.
        ਵਾਪਸ rename_outputs_with_suffix({attn_perm, state_4d}, context.get_name());
    }

    ਸਵੈਚਾਲਿਤ state_1d = std::make_shared<ov::op::v1::Reshape>(state_4d, flat_shape_1d, ਝੂਠ);

    // Concat [attn | state] and reshape to final output
    ਸਵੈਚਾਲਿਤ packed = std::make_shared<ov::op::v0::Concat>(ov::Output_ਸਦਿਸ਼{attn_1d, state_1d}, 0);
    ਸਵੈਚਾਲਿਤ ਬਾਹਰਲਾ_shape =
        ov::op::v0::Constant::create(ov::element::i64, {4}, std::vector<int64_t>{1, 1, T * B + S_v * B, S_v * H_v});
    ਸਵੈਚਾਲਿਤ res = std::make_shared<ov::op::v1::Reshape>(packed, ਬਾਹਰਲਾ_shape, ਝੂਠ);

    ਵਾਪਸ rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
