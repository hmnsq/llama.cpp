#ಅನ್ತರ್ಭಾವ <sycl/sycl.hpp>
#ಅನ್ತರ್ಭಾವ "dpct/helper.hpp"
#ಅನ್ತರ್ಭಾವ "common.hpp"
#ಅನ್ತರ್ಭಾವ "ggml.h"
#ಅನ್ತರ್ಭಾವ "gated_delta_net.hpp"
#ಅನ್ತರ್ಭಾವ <cmath>


ಮಾದರಿ <ಪೂರ್ಣಾಂಕ S_v, ತಾರ್ಕಿಕ KDA, ತಾರ್ಕಿಕ keep_rs_t>
ಶೂನ್ಯ gated_delta_net_sycl(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *     q,
                          ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *     k,
                          ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *     v,
                          ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *     g,
                          ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *     beta,
                          ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *     curr_state,
                          ದಶಮಾಂಶ *           dst,
                          ದಶಮಾಂಶ *           state,
                          int64_t           H,
                          int64_t           n_ಟೋಕನುಗಳು,
                          int64_t           sq1,
                          int64_t           sq2,
                          int64_t           sq3,
                          int64_t           sv1,
                          int64_t           sv2,
                          int64_t           sv3,
                          int64_t           sb1,
                          int64_t           sb2,
                          int64_t           sb3,
                          ಸ್ಥಿರಾಂಕ sycl::uint3 neqk1_magic,
                          ಸ್ಥಿರಾಂಕ sycl::uint3 rq3_magic,
                          ದಶಮಾಂಶ             scale,
                          int64_t           state_ಸ್ಥಾನ_stride,
                          ಪೂರ್ಣಾಂಕ               K) {
    ಸ್ವಯಂಚಾಲಿತ           ಅಂಶ_ct1 = sycl::ext::oneapi::this_work_ಅಂಶ::get_nd_ಅಂಶ<3>();
    ಸ್ಥಿರಾಂಕ uint32_t h_idx    = ಅಂಶ_ct1.get_ಗುಂಪು(2);
    ಸ್ಥಿರಾಂಕ uint32_t sequence = ಅಂಶ_ct1.get_ಗುಂಪು(1);
    // each warp owns one column, using warp-level primitives to reduce across rows
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ      lane     = ಅಂಶ_ct1.get_local_id(2);
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ      ಲಂಬಸಾಲು      = ಅಂಶ_ct1.get_ಗುಂಪು(0) * ಅಂಶ_ct1.get_local_range(1) + ಅಂಶ_ct1.get_local_id(1);

    ಸ್ಥಿರಾಂಕ uint32_t iq1 = fastmodulo(h_idx, neqk1_magic);
    ಸ್ಥಿರಾಂಕ uint32_t iq3 = fastdiv(sequence, rq3_magic);

    ದಶಮಾಂಶ *       attn_ದತ್ತಾಂಶ        = dst;

    // input state holds s0 only [S_v, S_v, H, n_seqs] — seq stride is D = H * S_v * S_v.
    // output state layout (per-slot D * n_seqs) — same per-(seq,head) offset as before.
    ಸ್ಥಿರಾಂಕ int64_t state_ಒಳ_ಸ್ಥಾನಾಂತರ      = sequence * H * S_v * S_v + h_idx * S_v * S_v;
    ಸ್ಥಿರಾಂಕ int64_t state_ಹೊರಹೋಗು_ಸ್ಥಾನಾಂತರ     = (sequence * H + h_idx) * S_v * S_v;
    state += state_ಹೊರಹೋಗು_ಸ್ಥಾನಾಂತರ;
    curr_state += state_ಒಳ_ಸ್ಥಾನಾಂತರ + ಲಂಬಸಾಲು * S_v;
    attn_ದತ್ತಾಂಶ += (sequence * n_ಟೋಕನುಗಳು * H + h_idx) * S_v;

    constexpr ಪೂರ್ಣಾಂಕ warp_ಗಾತ್ರ = ggml_sycl_get_physical_warp_ಗಾತ್ರ() < S_v ? ggml_sycl_get_physical_warp_ಗಾತ್ರ() : S_v;
    static_assert(S_v % warp_ಗಾತ್ರ == 0, "S_v must be a multiple of warp_size");
    constexpr ಪೂರ್ಣಾಂಕ ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_lane = (S_v + warp_ಗಾತ್ರ - 1) / warp_ಗಾತ್ರ;
    ದಶಮಾಂಶ         s_shard[ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_lane];
#ಸೂಚನೆ unroll
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ r = 0; r < ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_lane; r++) {
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ i = r * warp_ಗಾತ್ರ + lane;
        s_shard[r]  = curr_state[i];
    }

    // snapshot slot mapping: slot 0 = most recent state, slot s = s tokens back.
    // When n_tokens < K only slots 0..n_tokens-1 are written; older slots are caller-owned.

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ t = 0; t < n_ಟೋಕನುಗಳು; t++) {
        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * q_t = q + iq3 * sq3 + t * sq2 + iq1 * sq1;
        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * k_t = k + iq3 * sq3 + t * sq2 + iq1 * sq1;
        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * v_t = v + sequence * sv3 + t * sv2 + h_idx * sv1;

        ಸ್ಥಿರಾಂಕ int64_t gb_ಸ್ಥಾನಾಂತರ = sequence * sb3 + t * sb2 + h_idx * sb1;
        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * beta_t = beta + gb_ಸ್ಥಾನಾಂತರ;
        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * g_t    = g    + gb_ಸ್ಥಾನಾಂತರ * (KDA ? S_v : 1);

        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ beta_val = *beta_t;

        ಆದರೆ constexpr (!KDA) {
            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ g_val = sycl::native::ಘಾತಾಂಕ(*g_t);

            // kv[col] = (S^T @ k)[col] = sum_i S[i][col] * k[i]
            ದಶಮಾಂಶ kv_shard = 0.0f;
#ಸೂಚನೆ unroll
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ r = 0; r < ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_lane; r++) {
                ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ i = r * warp_ಗಾತ್ರ + lane;
                kv_shard += s_shard[r] * k_t[i];
            }
            ದಶಮಾಂಶ kv_ಲಂಬಸಾಲು = warp_reduce_ಮೊತ್ತ<warp_ಗಾತ್ರ>(kv_shard);

            // delta[col] = (v[col] - g * kv[col]) * beta
            ದಶಮಾಂಶ delta_ಲಂಬಸಾಲು = (v_t[ಲಂಬಸಾಲು] - g_val * kv_ಲಂಬಸಾಲು) * beta_val;

            // fused: S[i][col] = g * S[i][col] + k[i] * delta[col]
            // attn[col] = (S^T @ q)[col] = sum_i S[i][col] * q[i]
            ದಶಮಾಂಶ attn_partial = 0.0f;
#ಸೂಚನೆ unroll
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ r = 0; r < ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_lane; r++) {
                ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ i = r * warp_ಗಾತ್ರ + lane;
                s_shard[r]  = g_val * s_shard[r] + k_t[i] * delta_ಲಂಬಸಾಲು;
                attn_partial += s_shard[r] * q_t[i];
            }

            ದಶಮಾಂಶ attn_ಲಂಬಸಾಲು = warp_reduce_ಮೊತ್ತ<warp_ಗಾತ್ರ>(attn_partial);

            ಆದರೆ (lane == 0) {
                attn_ದತ್ತಾಂಶ[ಲಂಬಸಾಲು] = attn_ಲಂಬಸಾಲು * scale;
            }
        } ಇಲ್ಲದಿದ್ದರೆ {
            // kv[col] = sum_i g[i] * S[i][col] * k[i]
            ದಶಮಾಂಶ kv_shard = 0.0f;
#ಸೂಚನೆ unroll
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ r = 0; r < ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_lane; r++) {
                ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ i = r * warp_ಗಾತ್ರ + lane;
                kv_shard += sycl::native::ಘಾತಾಂಕ(g_t[i]) * s_shard[r] * k_t[i];
            }

            ದಶಮಾಂಶ kv_ಲಂಬಸಾಲು = warp_reduce_ಮೊತ್ತ<warp_ಗಾತ್ರ>(kv_shard);

            // delta[col] = (v[col] - kv[col]) * beta
            ದಶಮಾಂಶ delta_ಲಂಬಸಾಲು = (v_t[ಲಂಬಸಾಲು] - kv_ಲಂಬಸಾಲು) * beta_val;

            // fused: S[i][col] = g[i] * S[i][col] + k[i] * delta[col]
            // attn[col] = (S^T @ q)[col] = sum_i S[i][col] * q[i]
            ದಶಮಾಂಶ attn_partial = 0.0f;
#ಸೂಚನೆ unroll
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ r = 0; r < ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_lane; r++) {
                ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ i = r * warp_ಗಾತ್ರ + lane;
                s_shard[r]  = sycl::native::ಘಾತಾಂಕ(g_t[i]) * s_shard[r] + k_t[i] * delta_ಲಂಬಸಾಲು;
                attn_partial += s_shard[r] * q_t[i];
            }

            ದಶಮಾಂಶ attn_ಲಂಬಸಾಲು = warp_reduce_ಮೊತ್ತ<warp_ಗಾತ್ರ>(attn_partial);

            ಆದರೆ (lane == 0) {
                attn_ದತ್ತಾಂಶ[ಲಂಬಸಾಲು] = attn_ಲಂಬಸಾಲು * scale;
            }
        }

        attn_ದತ್ತಾಂಶ += S_v * H;


    // Write state back to global memory
        ಆದರೆ constexpr (keep_rs_t) {
            ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಗುರಿ_ಸ್ಥಾನ = (ಪೂರ್ಣಾಂಕ) n_ಟೋಕನುಗಳು - 1 - t;
            ಆದರೆ (ಗುರಿ_ಸ್ಥಾನ >= 0 && ಗುರಿ_ಸ್ಥಾನ < K) {
                ದಶಮಾಂಶ * curr_state = state + ಗುರಿ_ಸ್ಥಾನ * state_ಸ್ಥಾನ_stride;
#ಸೂಚನೆ unroll
                ಗಾಗಿ (ಪೂರ್ಣಾಂಕ r = 0; r < ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_lane; r++) {
                    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ i = r * warp_ಗಾತ್ರ + lane;
                    curr_state[ಲಂಬಸಾಲು * S_v + i] = s_shard[r];
                }
            }
        }
    }

    ಆದರೆ constexpr (!keep_rs_t) {
#ಸೂಚನೆ unroll
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ r = 0; r < ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_lane; r++) {
            ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ i          = r * warp_ಗಾತ್ರ + lane;
            state[ಲಂಬಸಾಲು * S_v + i] = s_shard[r];
        }
    }
}

ಮಾದರಿ <ತಾರ್ಕಿಕ KDA, ತಾರ್ಕಿಕ keep_rs_t>
ಸ್ಥಿರ ಶೂನ್ಯ launch_gated_delta_net(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *   q_d,
                                   ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *   k_d,
                                   ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *   v_d,
                                   ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *   g_d,
                                   ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *   b_d,
                                   ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *   s_d,
                                   ದಶಮಾಂಶ *         dst_d,
                                   ದಶಮಾಂಶ *         state_d,
                                   int64_t         S_v,
                                   int64_t         H,
                                   int64_t         n_ಟೋಕನುಗಳು,
                                   int64_t         n_seqs,
                                   int64_t         sq1,
                                   int64_t         sq2,
                                   int64_t         sq3,
                                   int64_t         sv1,
                                   int64_t         sv2,
                                   int64_t         sv3,
                                   int64_t         sb1,
                                   int64_t         sb2,
                                   int64_t         sb3,
                                   int64_t         neqk1,
                                   int64_t         rq3,
                                   ದಶಮಾಂಶ           scale,
                                   int64_t         state_ಸ್ಥಾನ_stride,
                                   ಪೂರ್ಣಾಂಕ             K,
                                   dpct::ಸರತಿ_ptr stream) {
    //TODO: Add chunked kernel for even faster pre-fill
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ warp_ಗಾತ್ರ = ggml_sycl_info().devices[ggml_sycl_get_device()].warp_ಗಾತ್ರ;

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಅಂಕೆ_warps = 4;
    dpct::dim3 grid_dims(H, n_seqs, (S_v + ಅಂಕೆ_warps - 1) / ಅಂಕೆ_warps);
    dpct::dim3 ಖಂಡ_dims(warp_ಗಾತ್ರ <= S_v ? warp_ಗಾತ್ರ : S_v, ಅಂಕೆ_warps, 1);

    ಸ್ಥಿರಾಂಕ sycl::uint3 neqk1_magic = init_fastdiv_ಮೌಲ್ಯಗಳು(neqk1);
    ಸ್ಥಿರಾಂಕ sycl::uint3 rq3_magic   = init_fastdiv_ಮೌಲ್ಯಗಳು(rq3);

    ಆಯ್ಕೆ (S_v) {
        ಸ್ಥಿತಿ 16:
            {
                constexpr ಪೂರ್ಣಾಂಕ sv = 16;
                stream->parallel_for(sycl::nd_range<3>(grid_dims * ಖಂಡ_dims, ಖಂಡ_dims),
                                     [=](sycl::nd_ಅಂಶ<3> /*item_ct1*/) [[sycl::reqd_sub_ಗುಂಪು_ಗಾತ್ರ(WARP_ಗಾತ್ರ)]] {
                                         gated_delta_net_sycl<sv, KDA, keep_rs_t>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H, n_ಟೋಕನುಗಳು,
                                                                       sq1, sq2, sq3, sv1, sv2, sv3, sb1, sb2,
                                                                       sb3, neqk1_magic, rq3_magic, scale, state_ಸ್ಥಾನ_stride, K);
                                     });
            }
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ 32:
            {
                constexpr ಪೂರ್ಣಾಂಕ sv = 32;
                stream->parallel_for(sycl::nd_range<3>(grid_dims * ಖಂಡ_dims, ಖಂಡ_dims),
                                     [=](sycl::nd_ಅಂಶ<3> /*item_ct1*/) [[sycl::reqd_sub_ಗುಂಪು_ಗಾತ್ರ(WARP_ಗಾತ್ರ)]] {
                                         gated_delta_net_sycl<sv, KDA, keep_rs_t>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H, n_ಟೋಕನುಗಳು,
                                                                       sq1, sq2, sq3, sv1, sv2, sv3, sb1, sb2,
                                                                       sb3, neqk1_magic, rq3_magic, scale, state_ಸ್ಥಾನ_stride, K);
                                     });
            }
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ 64: {
            {
                constexpr ಪೂರ್ಣಾಂಕ sv = 64;
                stream->parallel_for(sycl::nd_range<3>(grid_dims * ಖಂಡ_dims, ಖಂಡ_dims),
                                        [=](sycl::nd_ಅಂಶ<3> /*item_ct1*/) [[sycl::reqd_sub_ಗುಂಪು_ಗಾತ್ರ(WARP_ಗಾತ್ರ)]] {
                                            gated_delta_net_sycl<sv, KDA, keep_rs_t>(
                                                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H, n_ಟೋಕನುಗಳು, sq1, sq2,
                                                sq3, sv1, sv2, sv3, sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_ಸ್ಥಾನ_stride, K);
                                        });
            }
            ನಿಲ್ಲಿಸು;
        }
        ಸ್ಥಿತಿ 128: {
            {
                constexpr ಪೂರ್ಣಾಂಕ sv = 128;
                stream->parallel_for(sycl::nd_range<3>(grid_dims * ಖಂಡ_dims, ಖಂಡ_dims),
                                        [=](sycl::nd_ಅಂಶ<3> /*item_ct1*/) [[sycl::reqd_sub_ಗುಂಪು_ಗಾತ್ರ(WARP_ಗಾತ್ರ)]] {
                                            gated_delta_net_sycl<sv, KDA, keep_rs_t>(
                                                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H, n_ಟೋಕನುಗಳು, sq1, sq2,
                                                sq3, sv1, sv2, sv3, sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_ಸ್ಥಾನ_stride, K);
                                        });
            }
            ನಿಲ್ಲಿಸು;
        }
        ಪೂರ್ವನಿಯೋಜಿತ:
            GGML_ABORT("fatal error");
            ನಿಲ್ಲಿಸು;
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_sycl_op_gated_delta_net_impl(ggml_backend_sycl_context & ctx, ggml_tensor * dst,
                                              ಸ್ಥಿರಾಂಕ ggml_sycl_gated_delta_net_fused_cache * cache) {
    ggml_tensor * src_q     = dst->src[0];
    ggml_tensor * src_k     = dst->src[1];
    ggml_tensor * src_v     = dst->src[2];
    ggml_tensor * src_g     = dst->src[3];
    ggml_tensor * src_beta  = dst->src[4];
    ggml_tensor * src_state = dst->src[5];

    GGML_TENSOR_LOCALS(int64_t, neq, src_q, ne);
    GGML_TENSOR_LOCALS(size_t , nbq, src_q, nb);
    GGML_TENSOR_LOCALS(int64_t, nek, src_k, ne);
    GGML_TENSOR_LOCALS(size_t , nbk, src_k, nb);
    GGML_TENSOR_LOCALS(int64_t, nev, src_v, ne);
    GGML_TENSOR_LOCALS(size_t,  nbv, src_v, nb);
    GGML_TENSOR_LOCALS(size_t,  nbb, src_beta, nb);

    ಸ್ಥಿರಾಂಕ int64_t S_v      = nev0;
    ಸ್ಥಿರಾಂಕ int64_t H        = nev1;
    ಸ್ಥಿರಾಂಕ int64_t n_ಟೋಕನುಗಳು = nev2;
    ಸ್ಥಿರಾಂಕ int64_t n_seqs   = nev3;

    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ kda = (src_g->ne[0] == S_v);

    GGML_ASSERT(neq1 == nek1);
    ಸ್ಥಿರಾಂಕ int64_t neqk1 = neq1;

    ಸ್ಥಿರಾಂಕ int64_t rq3 = nev3 / neq3;

    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * q_d = (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) src_q->ದತ್ತಾಂಶ;
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * k_d = (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) src_k->ದತ್ತಾಂಶ;
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * v_d = (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) src_v->ದತ್ತಾಂಶ;
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * g_d = (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) src_g->ದತ್ತಾಂಶ;
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * b_d = (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) src_beta->ದತ್ತಾಂಶ;

    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * s_d   = (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) src_state->ದತ್ತಾಂಶ;
    ದಶಮಾಂಶ *       dst_d = (ದಶಮಾಂಶ *) dst->ದತ್ತಾಂಶ;

    GGML_ASSERT(ggml_is_contiguous_ಅಡ್ಡಸಾಲುಗಳು(src_q));
    GGML_ASSERT(ggml_is_contiguous_ಅಡ್ಡಸಾಲುಗಳು(src_k));
    GGML_ASSERT(ggml_is_contiguous_ಅಡ್ಡಸಾಲುಗಳು(src_v));
    GGML_ASSERT(ggml_are_same_stride(src_q, src_k));
    GGML_ASSERT(src_g->ne[0] == 1 || kda);
    GGML_ASSERT(ggml_is_contiguous(src_g));
    GGML_ASSERT(ggml_is_contiguous(src_beta));
    GGML_ASSERT(ggml_is_contiguous(src_state));

    // strides in floats (beta strides used for both g and beta offset computation)
    ಸ್ಥಿರಾಂಕ int64_t sq1 = nbq1 / ಗಾತ್ರ(ದಶಮಾಂಶ);
    ಸ್ಥಿರಾಂಕ int64_t sq2 = nbq2 / ಗಾತ್ರ(ದಶಮಾಂಶ);
    ಸ್ಥಿರಾಂಕ int64_t sq3 = nbq3 / ಗಾತ್ರ(ದಶಮಾಂಶ);
    ಸ್ಥಿರಾಂಕ int64_t sv1 = nbv1 / ಗಾತ್ರ(ದಶಮಾಂಶ);
    ಸ್ಥಿರಾಂಕ int64_t sv2 = nbv2 / ಗಾತ್ರ(ದಶಮಾಂಶ);
    ಸ್ಥಿರಾಂಕ int64_t sv3 = nbv3 / ಗಾತ್ರ(ದಶಮಾಂಶ);
    ಸ್ಥಿರಾಂಕ int64_t sb1 = nbb1 / ಗಾತ್ರ(ದಶಮಾಂಶ);
    ಸ್ಥಿರಾಂಕ int64_t sb2 = nbb2 / ಗಾತ್ರ(ದಶಮಾಂಶ);
    ಸ್ಥಿರಾಂಕ int64_t sb3 = nbb3 / ಗಾತ್ರ(ದಶಮಾಂಶ);

    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ scale = 1.0f / sqrtf((ದಶಮಾಂಶ) S_v);

    dpct::ಸರತಿ_ptr stream = ctx.stream();

    // K (snapshot slot count) is an op param; state holds s0 only [S_v, S_v, H, n_seqs].
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ K = ggml_get_op_params_i32(dst, 0);
    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ keep_rs = K > 1;

    // recurrent state -> dst tail (after attention scores), or the cache when fusing
    ದಶಮಾಂಶ * state_d           = dst_d + S_v * H * n_ಟೋಕನುಗಳು * n_seqs;
    int64_t state_ಸ್ಥಾನ_stride = S_v * S_v * H * n_seqs;
    ಆದರೆ (cache != ಶೂನ್ಯಸೂಚಕ) {
        state_d           = cache->ದತ್ತಾಂಶ;
        state_ಸ್ಥಾನ_stride = cache->ಸ್ಥಾನ_stride;
    }

    ಆದರೆ (kda) {
        ಆದರೆ (keep_rs) {
            launch_gated_delta_net<ನಿಜ, ನಿಜ>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_ಟೋಕನುಗಳು, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_ಸ್ಥಾನ_stride, K, stream);
        } ಇಲ್ಲದಿದ್ದರೆ {
            launch_gated_delta_net<ನಿಜ, ಸುಳ್ಳು>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_ಟೋಕನುಗಳು, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_ಸ್ಥಾನ_stride, K, stream);
        }
    } ಇಲ್ಲದಿದ್ದರೆ {
        ಆದರೆ (keep_rs) {
            launch_gated_delta_net<ಸುಳ್ಳು, ನಿಜ>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_ಟೋಕನುಗಳು, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_ಸ್ಥಾನ_stride, K, stream);
        } ಇಲ್ಲದಿದ್ದರೆ {
            launch_gated_delta_net<ಸುಳ್ಳು, ಸುಳ್ಳು>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_ಟೋಕನುಗಳು, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_ಸ್ಥಾನ_stride, K, stream);
        }
    }
}

ಶೂನ್ಯ ggml_sycl_op_gated_delta_net(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_op_gated_delta_net_impl(ctx, dst, ಶೂನ್ಯಸೂಚಕ);
}

ಶೂನ್ಯ ggml_sycl_gated_delta_net(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/6);
    ggml_sycl_op_gated_delta_net(ctx, dst);
}

ಶೂನ್ಯ ggml_sycl_op_gated_delta_net_fused_cache(ggml_backend_sycl_context & ctx, ggml_tensor * dst,
                                              ggml_sycl_gated_delta_net_fused_cache cache) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/6);
    ggml_sycl_op_gated_delta_net_impl(ctx, dst, &cache);
}
