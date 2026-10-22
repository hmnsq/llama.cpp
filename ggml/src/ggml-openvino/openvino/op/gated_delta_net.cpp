#ಅನ್ತರ್ಭಾವ "gated_delta_net.hpp"

#ಅನ್ತರ್ಭಾವ "../node_context.h"
#ಅನ್ತರ್ಭಾವ "../op_table.h"
#ಅನ್ತರ್ಭಾವ "../utils.h"
#ಅನ್ತರ್ಭಾವ "ggml-openvino/ggml-openvino-extra.h"

#ಅನ್ತರ್ಭಾವ <cmath>
#ಅನ್ತರ್ಭಾವ <cstdint>
#ಅನ್ತರ್ಭಾವ <memory>
#ಅನ್ತರ್ಭಾವ <numeric>
#ಅನ್ತರ್ಭಾವ <openvino/op/add.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/broadcast.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/concat.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/constant.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/convert.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/divide.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/exp.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/gather.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/less.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/loop.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/matmul.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/multiply.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/reduce_mean.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/reshape.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/squeeze.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/sqrt.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/subtract.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/tile.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/transpose.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/unsqueeze.hpp>
#ಅನ್ತರ್ಭಾವ <vector>

ನಾಮಾಕಾಶ ov {
ನಾಮಾಕಾಶ frontend {
ನಾಮಾಕಾಶ ggml {
ನಾಮಾಕಾಶ op {

ಸ್ಥಿರ Output_ಸದಿಶ translate_gated_delta_net_ref(ಸ್ಥಿರಾಂಕ ಗಂಟು_Context & context);

ಸ್ಥಿರ ತಾರ್ಕಿಕ ಹೊಂದಿಕೆ_gdn_l2_norm(ಸ್ಥಿರಾಂಕ Output<Node> & normalized, Output<Node> & ಒಳಹರಿವು, ದಶಮಾಂಶ & eps) {
    // Match the RMSNorm decomposition emitted by translate_rms_norm, followed by GGML SCALE.
    ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ scale = ov::as_type_ptr<ov::op::v1::Multiply>(normalized.get_ಗಂಟು_shared_ptr());
    ಆದರೆ (!scale) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }
    ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ ಗುಣಕ = ov::as_type_ptr<ov::op::v0::Constant>(scale->get_ಒಳಹರಿವು_ಗಂಟು_shared_ptr(1));
    ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ rms = ov::as_type_ptr<ov::op::v1::Multiply>(scale->get_ಒಳಹರಿವು_ಗಂಟು_shared_ptr(0));
    ಆದರೆ (!ಗುಣಕ || ov::shape_ಗಾತ್ರ(ಗುಣಕ->get_shape()) != 1 || !rms) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }
    ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ x = rms->ಒಳಹರಿವು_ಮೌಲ್ಯ(0);
    ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ & shape = x.get_partial_shape();
    ಆದರೆ (x.get_element_type() != ov::element::f32 || shape.rank() != 4 || shape[3].is_dynamic() ||
        shape[3].get_ಉದ್ದಳತೆ() <= 0 || normalized.get_partial_shape() != shape) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ dim = static_cast<ದಶಮಾಂಶ>(shape[3].get_ಉದ್ದಳತೆ());
    ಆದರೆ (ಗುಣಕ->cast_ಸದಿಶ<ದಶಮಾಂಶ>()[0] != 1.0f / std::ವರ್ಗಮೂಲ(dim)) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }
    ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ reciprocal = ov::as_type_ptr<ov::op::v1::Divide>(rms->get_ಒಳಹರಿವು_ಗಂಟು_shared_ptr(1));
    ಆದರೆ (!reciprocal) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }
    ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ one = ov::as_type_ptr<ov::op::v0::Constant>(reciprocal->get_ಒಳಹರಿವು_ಗಂಟು_shared_ptr(0));
    ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ ಬೇರು = ov::as_type_ptr<ov::op::v0::Sqrt>(reciprocal->get_ಒಳಹರಿವು_ಗಂಟು_shared_ptr(1));
    ಆದರೆ (!one || ov::shape_ಗಾತ್ರ(one->get_shape()) != 1 || one->cast_ಸದಿಶ<ದಶಮಾಂಶ>()[0] != 1.0f || !ಬೇರು) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }
    ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ ಕೂಡಿಸು_2 = ov::as_type_ptr<ov::op::v1::ಕೂಡಿಸು>(ಬೇರು->get_ಒಳಹರಿವು_ಗಂಟು_shared_ptr(0));
    ಆದರೆ (!ಕೂಡಿಸು_2) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }
    ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ ಸರಾಸರಿ_2 = ov::as_type_ptr<ov::op::v1::Reduce_ಸರಾಸರಿ>(ಕೂಡಿಸು_2->get_ಒಳಹರಿವು_ಗಂಟು_shared_ptr(0));
    ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ rms_eps = ov::as_type_ptr<ov::op::v0::Constant>(ಕೂಡಿಸು_2->get_ಒಳಹರಿವು_ಗಂಟು_shared_ptr(1));
    ಆದರೆ (!ಸರಾಸರಿ_2 || !ಸರಾಸರಿ_2->get_keep_dims() || !rms_eps || ov::shape_ಗಾತ್ರ(rms_eps->get_shape()) != 1) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }
    ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ axes = ov::as_type_ptr<ov::op::v0::Constant>(ಸರಾಸರಿ_2->get_ಒಳಹರಿವು_ಗಂಟು_shared_ptr(1));
    ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ ವರ್ಗಫಲ = ov::as_type_ptr<ov::op::v1::Multiply>(ಸರಾಸರಿ_2->get_ಒಳಹರಿವು_ಗಂಟು_shared_ptr(0));
    ಆದರೆ (!axes || axes->cast_ಸದಿಶ<int64_t>() != std::vector<int64_t>{-1} || !ವರ್ಗಫಲ ||
        ವರ್ಗಫಲ->ಒಳಹರಿವು_ಮೌಲ್ಯ(0) != x || ವರ್ಗಫಲ->ಒಳಹರಿವು_ಮೌಲ್ಯ(1) != x) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }
    // RMSNorm(x, rms_eps) / sqrt(D) = x / sqrt(sum(x*x) + D*rms_eps).
    eps = dim * rms_eps->cast_ಸದಿಶ<ದಶಮಾಂಶ>()[0];
    ಆದರೆ (!std::isfinite(eps) || eps <= 0.0f) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }
    ಒಳಹರಿವು = x;
    ಹಿಂತಿರುಗಿ ನಿಜ;
}

Output_ಸದಿಶ translate_gated_delta_net(ಸ್ಥಿರಾಂಕ ಗಂಟು_Context & context) {
    ಸ್ವಯಂಚಾಲಿತ v_shape = context.get_ಒಳಹರಿವು_shape(2).ಗಮ್ಯ_shape();  // [B, T, H_v, S_v]
    ಸ್ವಯಂಚಾಲಿತ q_shape = context.get_ಒಳಹರಿವು_shape(0).ಗಮ್ಯ_shape();  // [B, T, H_k, S_k]

    // Fused GatedDeltaNet op only supports scalar gate (kda=0).
    // Fall back to reference implementation for per-key-dimension gating.
    // if (kda) {
    //     return translate_gated_delta_net_ref(context);
    // }

    // const int64_t B = v_shape[0];
    // const int64_t T = v_shape[1];
    ಸ್ಥಿರಾಂಕ int64_t H_v = v_shape[2];
    ಸ್ಥಿರಾಂಕ int64_t S_v = v_shape[3];
    ಸ್ಥಿರಾಂಕ int64_t H_k = q_shape[2];
    // const int64_t S_k = q_shape[3];

    ಸ್ವಯಂಚಾಲಿತ q = context.get_ಒಳಹರಿವು(0);
    ಸ್ವಯಂಚಾಲಿತ k = context.get_ಒಳಹರಿವು(1);
    ಸ್ವಯಂಚಾಲಿತ v = process_view_ಒಳಹರಿವು(context, 2, H_v * S_v, 3);
    ಸ್ವಯಂಚಾಲಿತ g = context.get_ಒಳಹರಿವು(3);
    ಸ್ವಯಂಚಾಲಿತ beta = context.get_ಒಳಹರಿವು(4);
    ಸ್ವಯಂಚಾಲಿತ state = context.get_ಒಳಹರಿವು(5);

    Output<Node> raw_q, raw_k;
    ದಶಮಾಂಶ q_eps = 1e-6f, k_eps = 1e-6f;
    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ fuse_qk_l2norm = ggml_openvino_is_gpu() &&
        ಹೊಂದಿಕೆ_gdn_l2_norm(q, raw_q, q_eps) && ಹೊಂದಿಕೆ_gdn_l2_norm(k, raw_k, k_eps);
    ಆದರೆ (fuse_qk_l2norm) {
        // Keep head tiling below; GDN applies normalization per head and keeps its attention scale.
        q = raw_q;
        k = raw_k;
    }

    // ggml maps GQA heads in tiled order, while OV GDN maps repeated heads in grouped order.
    ಆದರೆ (H_v != H_k) {
        ಸ್ಥಿರಾಂಕ int64_t repeat = H_v / H_k;
        ಸ್ವಯಂಚಾಲಿತ repeats = ov::op::v0::Constant::create(ov::element::i64, {4}, std::vector<int64_t>{1, 1, repeat, 1});
        q = std::make_shared<ov::op::v0::Tile>(q, repeats);
        k = std::make_shared<ov::op::v0::Tile>(k, repeats);
    }

    ಆದರೆ (context.get_view_ಒಳಹರಿವು_ಗಾತ್ರ(2)) {
        // Same as l2_norm case 1
        v = std::make_shared<ov::op::v0::Squeeze>(v, ov::op::v0::Constant::create(ov::element::i64, {1}, {0}));
        ಸ್ವಯಂಚಾಲಿತ v_shape = context.get_ಒಳಹರಿವು_shape(2).ಗಮ್ಯ_shape();
        std::vector<int64_t> reshape_ನಮೂನೆ = {0, 0, (int64_t) v_shape[2], (int64_t) v_shape[3]};
        v = std::make_shared<ov::op::v1::Reshape>(
            v, ov::op::v0::Constant::create(ov::element::i64, {4}, reshape_ನಮೂನೆ), ನಿಜ);
    }

    // ggml state layout (OV notation): [B, H_v, value_dim, key_dim]
    // GatedDeltaNet op expects: [B, H_v, key_dim, value_dim]
    ಸ್ವಯಂಚಾಲಿತ state_perm = ov::op::v0::Constant::create(ov::element::i64, {4}, std::vector<int64_t>{0, 1, 3, 2});
    state = std::make_shared<ov::op::v1::Transpose>(state, state_perm);

    g = std::make_shared<ov::op::v0::Squeeze>(g, ov::op::v0::Constant::create(ov::element::i64, {1}, {3}));
    beta = std::make_shared<ov::op::v0::Squeeze>(beta, ov::op::v0::Constant::create(ov::element::i64, {1}, {3}));

    ಆದರೆ (context.has_ಒಳಹರಿವು("chunk_valid_len")) {
        // The last prefill chunk is padded with fabricated tokens. The recurrence is
        //   S_t = S_{t-1} * exp(g_t) + k_t (x) ((v_t - S_{t-1}^T k_t) * beta_t)
        // so forcing g = 0 and beta = 0 makes a padded step an exact identity and keeps the final
        // state equal to the state after the last real token. Attention output at those positions
        // is garbage but never read.
        ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ & g_ps = g.get_partial_shape();
        FRONT_ಕೊನೆ_OP_CONVERSION_CHECK(g_ps.rank().is_static() && g_ps.rank().get_ಉದ್ದಳತೆ() == 3 && g_ps[1].is_static(),
                                      "GATED_DELTA_NET pad masking requires a static token dimension");
        ಸ್ಥಿರಾಂಕ int64_t n_ಟೋಕನುಗಳು = g_ps[1].get_ಉದ್ದಳತೆ();
        std::vector<int64_t> positions(n_ಟೋಕನುಗಳು);
        std::iota(positions.begin(), positions.end(), 0);
        ಸ್ವಯಂಚಾಲಿತ valid = std::make_shared<ov::op::v1::Less>(
            ov::op::v0::Constant::create(ov::element::i64, {(size_t) n_ಟೋಕನುಗಳು}, positions),
            context.get_ಒಳಹರಿವು("chunk_valid_len"));
        ಸ್ವಯಂಚಾಲಿತ ಮುಖವಾಡ = std::make_shared<ov::op::v0::Unsqueeze>(
            std::make_shared<ov::op::v0::Convert>(valid, g.get_element_type()),
            ov::op::v0::Constant::create(ov::element::i64, {2}, std::vector<int64_t>{0, 2}));
        g = std::make_shared<ov::op::v1::Multiply>(g, ಮುಖವಾಡ);
        beta = std::make_shared<ov::op::v1::Multiply>(beta, ಮುಖವಾಡ);
    }

    // std::cout << "GatedDeltaNet input shapes: q=" << q.get_partial_shape() << ", k=" << k.get_partial_shape()
    //           << ", v=" << v.get_partial_shape() << ", g=" << g.get_partial_shape()
    //           << ", beta=" << beta.get_partial_shape() << ", state=" << state.get_partial_shape() << std::endl;

    ಸ್ವಯಂಚಾಲಿತ gdn = std::make_shared<ov::op::internal::GatedDeltaNet>(q, k, v, state, g, beta, fuse_qk_l2norm, q_eps, k_eps);
    ಸ್ವಯಂಚಾಲಿತ attn_4d = gdn->output(0);
    ಸ್ವಯಂಚಾಲಿತ state_4d = gdn->output(1);  // [B, H_v, key_dim, value_dim]

    // std::cout << "GatedDeltaNet output shapes: attn=" << gdn->output(0).get_partial_shape()
    //           << ", new_state=" << gdn->output(1).get_partial_shape() << std::endl;

    // Transpose output state back to ggml layout [B, H_v, value_dim, key_dim]
    ಸ್ವಯಂಚಾಲಿತ state_transposed = std::make_shared<ov::op::v1::Transpose>(state_4d, state_perm);
    ಆದರೆ (context.get_output_ಹೆಸರುಗಳು().size() == 2) {
        // The canonical graph consumes the packed GGML result only through separate attention
        // and state VIEWs. Publish the native outputs under those VIEW names to avoid
        // flatten -> concat -> reshape -> slice -> reshape chains. This also works for B > 1.
        ಹಿಂತಿರುಗಿ rename_outputs_with_suffix({attn_4d, state_transposed}, context.get_name());
    }

    ಸ್ವಯಂಚಾಲಿತ flat_shape_1d = ov::op::v0::Constant::create(ov::element::i64, {1}, {-1});
    ಸ್ವಯಂಚಾಲಿತ attn = std::make_shared<ov::op::v1::Reshape>(attn_4d, flat_shape_1d, ಸುಳ್ಳು);
    ಸ್ವಯಂಚಾಲಿತ new_state = std::make_shared<ov::op::v1::Reshape>(state_transposed, flat_shape_1d, ಸುಳ್ಳು);
    ಸ್ವಯಂಚಾಲಿತ packed = std::make_shared<ov::op::v0::Concat>(ov::Output_ಸದಿಶ{attn, new_state}, 0);
    ಸ್ವಯಂಚಾಲಿತ ಹೊರಹೋಗು_shape = ov::op::v0::Constant::create(ov::element::i64, {4},
                                                  std::vector<int64_t>{1, 1, -1 /*T * B + S_v * B*/, S_v * H_v});
    ಸ್ವಯಂಚಾಲಿತ res = std::make_shared<ov::op::v1::Reshape>(packed, ಹೊರಹೋಗು_shape, ಸುಳ್ಳು);

    ಹಿಂತಿರುಗಿ rename_outputs_with_suffix({res}, context.get_name());
}

ಸ್ಥಿರ Output_ಸದಿಶ translate_gated_delta_net_ref(ಸ್ಥಿರಾಂಕ ಗಂಟು_Context & context) {
    ಅಂಕೆ_inputs_check(context, 6, 6);

    // Inputs (OV shapes are reversed from ggml):
    // ggml: q[S_k, H_k, T, B], k[S_k, H_k, T, B], v[S_v, H_v, T, B]
    // OV:   q[B, T, H_k, S_k], k[B, T, H_k, S_k], v[B, T, H_v, S_v]
    // ggml: g[1 or S_v, H_v, T, B], beta[1, H_v, T, B]
    // OV:   g[B, T, H_v, 1 or S_v], beta[B, T, H_v, 1]
    // ggml: state[S_v, S_v, H_v, B]
    // OV:   state[B, H_v, S_v, S_v]
    ಸ್ವಯಂಚಾಲಿತ q = process_view_ಒಳಹರಿವು_new(context, 0);
    ಸ್ವಯಂಚಾಲಿತ k = process_view_ಒಳಹರಿವು_new(context, 1);
    ಸ್ವಯಂಚಾಲಿತ v = process_view_ಒಳಹರಿವು_new(context, 2);
    ಸ್ವಯಂಚಾಲಿತ g = process_view_ಒಳಹರಿವು_new(context, 3);
    ಸ್ವಯಂಚಾಲಿತ beta = process_view_ಒಳಹರಿವು_new(context, 4);
    ಸ್ವಯಂಚಾಲಿತ state = process_view_ಒಳಹರಿವು_new(context, 5);

    ಸ್ವಯಂಚಾಲಿತ v_shape = context.get_ಒಳಹರಿವು_shape(2).ಗಮ್ಯ_shape();  // [B, T, H_v, S_v]
    ಸ್ವಯಂಚಾಲಿತ q_shape = context.get_ಒಳಹರಿವು_shape(0).ಗಮ್ಯ_shape();  // [B, T, H_k, S_k]
    ಸ್ವಯಂಚಾಲಿತ g_shape = context.get_ಒಳಹರಿವು_shape(3).ಗಮ್ಯ_shape();  // [B, T, H_v, 1 or S_v]

    ಸ್ಥಿರಾಂಕ int64_t B = v_shape[0];
    ಸ್ಥಿರಾಂಕ int64_t T = v_shape[1];
    ಸ್ಥಿರಾಂಕ int64_t H_v = v_shape[2];
    ಸ್ಥಿರಾಂಕ int64_t S_v = v_shape[3];
    ಸ್ಥಿರಾಂಕ int64_t H_k = q_shape[2];
    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ kda = (g_shape[3] == (size_t) S_v);

    ಸ್ಥಿರಾಂಕ int64_t rq1 = H_v / H_k;  // head repeat factor
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ scale = 1.0f / std::ವರ್ಗಮೂಲ((ದಶಮಾಂಶ) S_v);

    ಸ್ವಯಂಚಾಲಿತ axis_1 = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
    ಸ್ವಯಂಚಾಲಿತ axis_2 = ov::op::v0::Constant::create(ov::element::i64, {1}, {2});

    // Transpose inputs from [B, T, H, S] to [B, H, T, S] for easier per-head processing
    ಸ್ವಯಂಚಾಲಿತ perm_0213 = ov::op::v0::Constant::create(ov::element::i64, {4}, std::vector<int64_t>{0, 2, 1, 3});
    ಸ್ವಯಂಚಾಲಿತ q_t = std::make_shared<ov::op::v1::Transpose>(q, perm_0213);        // [B, H_k, T, S_k]
    ಸ್ವಯಂಚಾಲಿತ k_t = std::make_shared<ov::op::v1::Transpose>(k, perm_0213);        // [B, H_k, T, S_k]
    ಸ್ವಯಂಚಾಲಿತ v_t = std::make_shared<ov::op::v1::Transpose>(v, perm_0213);        // [B, H_v, T, S_v]
    ಸ್ವಯಂಚಾಲಿತ g_t = std::make_shared<ov::op::v1::Transpose>(g, perm_0213);        // [B, H_v, T, 1 or S_v]
    ಸ್ವಯಂಚಾಲಿತ beta_t = std::make_shared<ov::op::v1::Transpose>(beta, perm_0213);  // [B, H_v, T, 1]

    // Broadcast Q, K heads to match V heads if GQA is used (H_v > H_k)
    ov::Output<ov::Node> q_bh = q_t;
    ov::Output<ov::Node> k_bh = k_t;
    ಆದರೆ (rq1 > 1) {
        ಸ್ವಯಂಚಾಲಿತ q_unsq = std::make_shared<ov::op::v0::Unsqueeze>(q_t, axis_2);  // [B, H_k, 1, T, S]
        ಸ್ವಯಂಚಾಲಿತ k_unsq = std::make_shared<ov::op::v0::Unsqueeze>(k_t, axis_2);  // [B, H_k, 1, T, S]

        ಸ್ವಯಂಚಾಲಿತ bcast_shape = ov::op::v0::Constant::create(ov::element::i64, {5}, std::vector<int64_t>{1, 1, rq1, 1, 1});
        ಸ್ವಯಂಚಾಲಿತ q_bcast =
            std::make_shared<ov::op::v3::ಪ್ರಸಾರ>(q_unsq, bcast_shape, ov::op::ಪ್ರಸಾರ_Type::BIDIRECTIONAL);
        ಸ್ವಯಂಚಾಲಿತ k_bcast =
            std::make_shared<ov::op::v3::ಪ್ರಸಾರ>(k_unsq, bcast_shape, ov::op::ಪ್ರಸಾರ_Type::BIDIRECTIONAL);

        // Transpose [B, H_k, rq1, T, S] -> [B, rq1, H_k, T, S] so that reshape merges
        // as [rq1, H_k] giving repeat-blocks pattern matching CPU: iq1 = iv1 % H_k
        ಸ್ವಯಂಚಾಲಿತ perm_5d = ov::op::v0::Constant::create(ov::element::i64, {5}, std::vector<int64_t>{0, 2, 1, 3, 4});
        ಸ್ವಯಂಚಾಲಿತ q_transposed = std::make_shared<ov::op::v1::Transpose>(q_bcast, perm_5d);
        ಸ್ವಯಂಚಾಲಿತ k_transposed = std::make_shared<ov::op::v1::Transpose>(k_bcast, perm_5d);

        ಸ್ವಯಂಚಾಲಿತ new_shape = ov::op::v0::Constant::create(ov::element::i64, {4}, std::vector<int64_t>{B, H_v, T, S_v});
        q_bh = std::make_shared<ov::op::v1::Reshape>(q_transposed, new_shape, ಸುಳ್ಳು);
        k_bh = std::make_shared<ov::op::v1::Reshape>(k_transposed, new_shape, ಸುಳ್ಳು);
    }

    // Merge batch and head dims: [B*H_v, T, S_v]
    ಸ್ವಯಂಚಾಲಿತ ವಿಲೀನ_bh = [&](ಸ್ಥಿರಾಂಕ ov::Output<ov::Node> & x, int64_t ಕೊನೆಯ_dim) {
        ಸ್ವಯಂಚಾಲಿತ shape = ov::op::v0::Constant::create(ov::element::i64, {3}, std::vector<int64_t>{B * H_v, T, ಕೊನೆಯ_dim});
        ಹಿಂತಿರುಗಿ std::make_shared<ov::op::v1::Reshape>(x, shape, ಸುಳ್ಳು);
    };

    ಸ್ವಯಂಚಾಲಿತ q_m = ವಿಲೀನ_bh(q_bh, S_v);           // [B*H_v, T, S_v]
    ಸ್ವಯಂಚಾಲಿತ k_m = ವಿಲೀನ_bh(k_bh, S_v);           // [B*H_v, T, S_v]
    ಸ್ವಯಂಚಾಲಿತ v_m = ವಿಲೀನ_bh(v_t, S_v);            // [B*H_v, T, S_v]
    ಸ್ವಯಂಚಾಲಿತ g_m = ವಿಲೀನ_bh(g_t, kda ? S_v : 1);  // [B*H_v, T, 1 or S_v]
    ಸ್ವಯಂಚಾಲಿತ beta_m = ವಿಲೀನ_bh(beta_t, 1);        // [B*H_v, T, 1]

    // State: [B, H_v, S_v, S_v] -> [B*H_v, S_v, S_v]
    ಸ್ವಯಂಚಾಲಿತ state_shape = ov::op::v0::Constant::create(ov::element::i64, {3}, std::vector<int64_t>{B * H_v, S_v, S_v});
    ಸ್ವಯಂಚಾಲಿತ state_m = std::make_shared<ov::op::v1::Reshape>(state, state_shape, ಸುಳ್ಳು);

    ಸ್ವಯಂಚಾಲಿತ scale_const = ov::op::v0::Constant::create(ov::element::f32, {}, std::vector<ದಶಮಾಂಶ>{scale});

    // --- Build Loop body ---
    // Body parameters (no iteration counter needed, use -1 in special ports)
    ಸ್ವಯಂಚಾಲಿತ body_state = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic());
    ಸ್ವಯಂಚಾಲಿತ body_q = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic());
    ಸ್ವಯಂಚಾಲಿತ body_k = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic());
    ಸ್ವಯಂಚಾಲಿತ body_v = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic());
    ಸ್ವಯಂಚಾಲಿತ body_g = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic());
    ಸ್ವಯಂಚಾಲಿತ body_beta = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic());
    ಸ್ವಯಂಚಾಲಿತ body_iter = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::Shape{1});

    // Condition output (always true - we rely on trip_count for termination)
    ಸ್ವಯಂಚಾಲಿತ body_cond_ಹೊರಹೋಗು = ov::op::v0::Constant::create(ov::element::boolean, ov::Shape{1}, std::vector<ತಾರ್ಕಿಕ>{ನಿಜ});

    // Gather current token from invariant inputs using iteration counter
    ಸ್ವಯಂಚಾಲಿತ q_t_cur = std::make_shared<ov::op::v8::Gather>(body_q, body_iter, axis_1);     // [B*H_v, 1, S_v]
    ಸ್ವಯಂಚಾಲಿತ k_t_cur = std::make_shared<ov::op::v8::Gather>(body_k, body_iter, axis_1);     // [B*H_v, 1, S_v]
    ಸ್ವಯಂಚಾಲಿತ v_t_cur = std::make_shared<ov::op::v8::Gather>(body_v, body_iter, axis_1);     // [B*H_v, 1, S_v]
    ಸ್ವಯಂಚಾಲಿತ g_t_cur = std::make_shared<ov::op::v8::Gather>(body_g, body_iter, axis_1);     // [B*H_v, 1, 1 or S_v]
    ಸ್ವಯಂಚಾಲಿತ b_t_cur = std::make_shared<ov::op::v8::Gather>(body_beta, body_iter, axis_1);  // [B*H_v, 1, 1]

    // Squeeze token dim
    ಸ್ವಯಂಚಾಲಿತ q_cur = std::make_shared<ov::op::v0::Squeeze>(q_t_cur, axis_1);  // [B*H_v, S_v]
    ಸ್ವಯಂಚಾಲಿತ k_cur = std::make_shared<ov::op::v0::Squeeze>(k_t_cur, axis_1);  // [B*H_v, S_v]
    ಸ್ವಯಂಚಾಲಿತ v_cur = std::make_shared<ov::op::v0::Squeeze>(v_t_cur, axis_1);  // [B*H_v, S_v]
    ಸ್ವಯಂಚಾಲಿತ g_cur = std::make_shared<ov::op::v0::Squeeze>(g_t_cur, axis_1);  // [B*H_v, 1 or S_v]
    ಸ್ವಯಂಚಾಲಿತ b_cur = std::make_shared<ov::op::v0::Squeeze>(b_t_cur, axis_1);  // [B*H_v, 1]

    // Step 1: Apply decay gate to state
    ಸ್ವಯಂಚಾಲಿತ exp_g = std::make_shared<ov::op::v0::Exp>(g_cur);                                // [B*H_v, 1 or S_v]
    ಸ್ವಯಂಚಾಲಿತ exp_g_unsq = std::make_shared<ov::op::v0::Unsqueeze>(exp_g, axis_1);             // [B*H_v, 1, 1 or S_v]
    ಸ್ವಯಂಚಾಲಿತ state_decayed = std::make_shared<ov::op::v1::Multiply>(body_state, exp_g_unsq);  // [B*H_v, S_v, S_v]

    // Step 2: delta = (v - S @ k) * beta
    ಸ್ವಯಂಚಾಲಿತ k_ಲಂಬಸಾಲು = std::make_shared<ov::op::v0::Unsqueeze>(k_cur, axis_2);                 // [B*H_v, S_v, 1]
    ಸ್ವಯಂಚಾಲಿತ sk = std::make_shared<ov::op::v0::MatMul>(state_decayed, k_ಲಂಬಸಾಲು, ಸುಳ್ಳು, ಸುಳ್ಳು);  // [B*H_v, S_v, 1]
    ಸ್ವಯಂಚಾಲಿತ sk_sq = std::make_shared<ov::op::v0::Squeeze>(sk, axis_2);                      // [B*H_v, S_v]
    ಸ್ವಯಂಚಾಲಿತ v_minus_sk = std::make_shared<ov::op::v1::ಕಳೆ>(v_cur, sk_sq);              // [B*H_v, S_v]
    ಸ್ವಯಂಚಾಲಿತ delta = std::make_shared<ov::op::v1::Multiply>(v_minus_sk, b_cur);              // [B*H_v, S_v]

    // Step 3: state += outer(delta, k)
    ಸ್ವಯಂಚಾಲಿತ delta_ಲಂಬಸಾಲು = std::make_shared<ov::op::v0::Unsqueeze>(delta, axis_2);                 // [B*H_v, S_v, 1]
    ಸ್ವಯಂಚಾಲಿತ k_ಅಡ್ಡಸಾಲು = std::make_shared<ov::op::v0::Unsqueeze>(k_cur, axis_1);                     // [B*H_v, 1, S_v]
    ಸ್ವಯಂಚಾಲಿತ outer_prod = std::make_shared<ov::op::v0::MatMul>(delta_ಲಂಬಸಾಲು, k_ಅಡ್ಡಸಾಲು, ಸುಳ್ಳು, ಸುಳ್ಳು);  // [B*H_v, S_v, S_v]
    ಸ್ವಯಂಚಾಲಿತ state_updated = std::make_shared<ov::op::v1::ಕೂಡಿಸು>(state_decayed, outer_prod);       // [B*H_v, S_v, S_v]

    // Step 4: attn_out = S @ q * scale
    ಸ್ವಯಂಚಾಲಿತ q_ಲಂಬಸಾಲು = std::make_shared<ov::op::v0::Unsqueeze>(q_cur, axis_2);                 // [B*H_v, S_v, 1]
    ಸ್ವಯಂಚಾಲಿತ sq = std::make_shared<ov::op::v0::MatMul>(state_updated, q_ಲಂಬಸಾಲು, ಸುಳ್ಳು, ಸುಳ್ಳು);  // [B*H_v, S_v, 1]
    ಸ್ವಯಂಚಾಲಿತ sq_squeezed = std::make_shared<ov::op::v0::Squeeze>(sq, axis_2);                // [B*H_v, S_v]
    ಸ್ವಯಂಚಾಲಿತ attn_ಹೊರಹೋಗು = std::make_shared<ov::op::v1::Multiply>(sq_squeezed, scale_const);    // [B*H_v, S_v]

    // Unsqueeze attn_out to [B*H_v, 1, S_v] for scan output concatenation
    ಸ್ವಯಂಚಾಲಿತ attn_ಹೊರಹೋಗು_unsq = std::make_shared<ov::op::v0::Unsqueeze>(attn_ಹೊರಹೋಗು, axis_1);  // [B*H_v, 1, S_v]

    // --- Assemble Loop ---
    // Body: results = [condition, state_updated, attn_out_unsq]
    ಸ್ವಯಂಚಾಲಿತ body = std::make_shared<ov::Model>(
        ov::Output_ಸದಿಶ{body_cond_ಹೊರಹೋಗು, state_updated, attn_ಹೊರಹೋಗು_unsq},
        ov::Parameter_ಸದಿಶ{body_iter, body_state, body_q, body_k, body_v, body_g, body_beta});

    ಸ್ವಯಂಚಾಲಿತ trip_ಎಣಿಕೆ = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{T});
    ಸ್ವಯಂಚಾಲಿತ exec_cond = ov::op::v0::Constant::create(ov::element::boolean, ov::Shape{1}, std::vector<ತಾರ್ಕಿಕ>{ನಿಜ});

    ಸ್ವಯಂಚಾಲಿತ loop = std::make_shared<ov::op::v5::Loop>(trip_ಎಣಿಕೆ, exec_cond);
    loop->ಗಣ_function(body);
    loop->ಗಣ_special_body_ports(ov::op::v5::Loop::SpecialBodyPorts{0, 0});

    // Carried state: feeds back from body output 1 to body_state param
    loop->ಗಣ_merged_ಒಳಹರಿವು(body_state, state_m, state_updated);
    // Invariant inputs: passed through unchanged each iteration
    loop->ಗಣ_invariant_ಒಳಹರಿವು(body_q, q_m);
    loop->ಗಣ_invariant_ಒಳಹರಿವು(body_k, k_m);
    loop->ಗಣ_invariant_ಒಳಹರಿವು(body_v, v_m);
    loop->ಗಣ_invariant_ಒಳಹರಿವು(body_g, g_m);
    loop->ಗಣ_invariant_ಒಳಹರಿವು(body_beta, beta_m);

    // Loop outputs:
    // 1) Final state (last iteration value of state_updated)
    ಸ್ವಯಂಚಾಲಿತ final_state_ಹೊರಹೋಗು = loop->get_iter_ಮೌಲ್ಯ(state_updated, -1);  // [B*H_v, S_v, S_v]
    // 2) Concatenated attention outputs across all iterations along axis 1
    ಸ್ವಯಂಚಾಲಿತ attn_concat_ಹೊರಹೋಗು = loop->get_concatenated_slices(attn_ಹೊರಹೋಗು_unsq, 0, 1, 1, -1, 1);  // [B*H_v, T, S_v]

    // --- Pack outputs to match ggml layout ---
    // ggml output ne = {S_v*H, T*B + S_v*B, 1, 1} -> OV [1, 1, T*B+S_v*B, S_v*H_v]
    // attn: [B, T, H_v, S_v] row-major, state: [B, H_v, S_v, S_v] row-major

    // attn: [B*H_v, T, S_v] -> [B, H_v, T, S_v] -> transpose to [B, T, H_v, S_v] -> flatten
    ಸ್ವಯಂಚಾಲಿತ attn_4d_shape = ov::op::v0::Constant::create(ov::element::i64, {4}, std::vector<int64_t>{B, H_v, T, S_v});
    ಸ್ವಯಂಚಾಲಿತ attn_4d = std::make_shared<ov::op::v1::Reshape>(attn_concat_ಹೊರಹೋಗು, attn_4d_shape, ಸುಳ್ಳು);
    ಸ್ವಯಂಚಾಲಿತ attn_perm = std::make_shared<ov::op::v1::Transpose>(attn_4d, perm_0213);  // [B, T, H_v, S_v]

    ಸ್ವಯಂಚಾಲಿತ flat_shape_1d = ov::op::v0::Constant::create(ov::element::i64, {1}, std::vector<int64_t>{-1});
    ಸ್ವಯಂಚಾಲಿತ attn_1d = std::make_shared<ov::op::v1::Reshape>(attn_perm, flat_shape_1d, ಸುಳ್ಳು);

    // state: [B*H_v, S_v, S_v] -> [B, H_v, S_v, S_v] -> flatten
    ಸ್ವಯಂಚಾಲಿತ state_4d_shape = ov::op::v0::Constant::create(ov::element::i64, {4}, std::vector<int64_t>{B, H_v, S_v, S_v});
    ಸ್ವಯಂಚಾಲಿತ state_4d = std::make_shared<ov::op::v1::Reshape>(final_state_ಹೊರಹೋಗು, state_4d_shape, ಸುಳ್ಳು);
    ಆದರೆ (context.get_output_ಹೆಸರುಗಳು().size() == 2) {
        // Match the fused translator's direct attention/state contract.
        ಹಿಂತಿರುಗಿ rename_outputs_with_suffix({attn_perm, state_4d}, context.get_name());
    }

    ಸ್ವಯಂಚಾಲಿತ state_1d = std::make_shared<ov::op::v1::Reshape>(state_4d, flat_shape_1d, ಸುಳ್ಳು);

    // Concat [attn | state] and reshape to final output
    ಸ್ವಯಂಚಾಲಿತ packed = std::make_shared<ov::op::v0::Concat>(ov::Output_ಸದಿಶ{attn_1d, state_1d}, 0);
    ಸ್ವಯಂಚಾಲಿತ ಹೊರಹೋಗು_shape =
        ov::op::v0::Constant::create(ov::element::i64, {4}, std::vector<int64_t>{1, 1, T * B + S_v * B, S_v * H_v});
    ಸ್ವಯಂಚಾಲಿತ res = std::make_shared<ov::op::v1::Reshape>(packed, ಹೊರಹೋಗು_shape, ಸುಳ್ಳು);

    ಹಿಂತಿರುಗಿ rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
