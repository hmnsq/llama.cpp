#ਅਨ੍ਤਰ੍ਭਾਵ <sycl/sycl.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ "dpct/helper.hpp"
#ਅਨ੍ਤਰ੍ਭਾਵ "common.hpp"
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "gated_delta_net.hpp"
#ਅਨ੍ਤਰ੍ਭਾਵ <cmath>


ਸਾਂਚਾ <ਪੂਰਨ S_v, ਤਾਰਕਿਕ KDA, ਤਾਰਕਿਕ keep_rs_t>
ਖਾਲੀ gated_delta_net_sycl(ਅਚਲ ਦਸ਼ਮਲਵ *     q,
                          ਅਚਲ ਦਸ਼ਮਲਵ *     k,
                          ਅਚਲ ਦਸ਼ਮਲਵ *     v,
                          ਅਚਲ ਦਸ਼ਮਲਵ *     g,
                          ਅਚਲ ਦਸ਼ਮਲਵ *     beta,
                          ਅਚਲ ਦਸ਼ਮਲਵ *     curr_state,
                          ਦਸ਼ਮਲਵ *           dst,
                          ਦਸ਼ਮਲਵ *           state,
                          int64_t           H,
                          int64_t           n_ਟੋਕਨ,
                          int64_t           sq1,
                          int64_t           sq2,
                          int64_t           sq3,
                          int64_t           sv1,
                          int64_t           sv2,
                          int64_t           sv3,
                          int64_t           sb1,
                          int64_t           sb2,
                          int64_t           sb3,
                          ਅਚਲ sycl::uint3 neqk1_magic,
                          ਅਚਲ sycl::uint3 rq3_magic,
                          ਦਸ਼ਮਲਵ             scale,
                          int64_t           state_ਥਾਂ_stride,
                          ਪੂਰਨ               K) {
    ਸਵੈਚਾਲਿਤ           ਵਸਤੂ_ct1 = sycl::ext::oneapi::this_work_ਵਸਤੂ::get_nd_ਵਸਤੂ<3>();
    ਅਚਲ uint32_t h_idx    = ਵਸਤੂ_ct1.get_ਸਮੂਹ(2);
    ਅਚਲ uint32_t sequence = ਵਸਤੂ_ct1.get_ਸਮੂਹ(1);
    // each warp owns one column, using warp-level primitives to reduce across rows
    ਅਚਲ ਪੂਰਨ      lane     = ਵਸਤੂ_ct1.get_local_id(2);
    ਅਚਲ ਪੂਰਨ      ਖਾਨਾ      = ਵਸਤੂ_ct1.get_ਸਮੂਹ(0) * ਵਸਤੂ_ct1.get_local_range(1) + ਵਸਤੂ_ct1.get_local_id(1);

    ਅਚਲ uint32_t iq1 = fastmodulo(h_idx, neqk1_magic);
    ਅਚਲ uint32_t iq3 = fastdiv(sequence, rq3_magic);

    ਦਸ਼ਮਲਵ *       attn_ਅੰਕੜੇ        = dst;

    // input state holds s0 only [S_v, S_v, H, n_seqs] — seq stride is D = H * S_v * S_v.
    // output state layout (per-slot D * n_seqs) — same per-(seq,head) offset as before.
    ਅਚਲ int64_t state_ਵਿੱਚ_ਵਿੱਥ      = sequence * H * S_v * S_v + h_idx * S_v * S_v;
    ਅਚਲ int64_t state_ਬਾਹਰਲਾ_ਵਿੱਥ     = (sequence * H + h_idx) * S_v * S_v;
    state += state_ਬਾਹਰਲਾ_ਵਿੱਥ;
    curr_state += state_ਵਿੱਚ_ਵਿੱਥ + ਖਾਨਾ * S_v;
    attn_ਅੰਕੜੇ += (sequence * n_ਟੋਕਨ * H + h_idx) * S_v;

    constexpr ਪੂਰਨ warp_ਮਾਪ = ggml_sycl_get_physical_warp_ਮਾਪ() < S_v ? ggml_sycl_get_physical_warp_ਮਾਪ() : S_v;
    static_assert(S_v % warp_ਮਾਪ == 0, "S_v must be a multiple of warp_size");
    constexpr ਪੂਰਨ ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_lane = (S_v + warp_ਮਾਪ - 1) / warp_ਮਾਪ;
    ਦਸ਼ਮਲਵ         s_shard[ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_lane];
#ਹਦਾਇਤ unroll
    ਲਈ (ਪੂਰਨ r = 0; r < ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_lane; r++) {
        ਅਚਲ ਪੂਰਨ i = r * warp_ਮਾਪ + lane;
        s_shard[r]  = curr_state[i];
    }

    // snapshot slot mapping: slot 0 = most recent state, slot s = s tokens back.
    // When n_tokens < K only slots 0..n_tokens-1 are written; older slots are caller-owned.

    ਲਈ (ਪੂਰਨ t = 0; t < n_ਟੋਕਨ; t++) {
        ਅਚਲ ਦਸ਼ਮਲਵ * q_t = q + iq3 * sq3 + t * sq2 + iq1 * sq1;
        ਅਚਲ ਦਸ਼ਮਲਵ * k_t = k + iq3 * sq3 + t * sq2 + iq1 * sq1;
        ਅਚਲ ਦਸ਼ਮਲਵ * v_t = v + sequence * sv3 + t * sv2 + h_idx * sv1;

        ਅਚਲ int64_t gb_ਵਿੱਥ = sequence * sb3 + t * sb2 + h_idx * sb1;
        ਅਚਲ ਦਸ਼ਮਲਵ * beta_t = beta + gb_ਵਿੱਥ;
        ਅਚਲ ਦਸ਼ਮਲਵ * g_t    = g    + gb_ਵਿੱਥ * (KDA ? S_v : 1);

        ਅਚਲ ਦਸ਼ਮਲਵ beta_val = *beta_t;

        ਜੇ constexpr (!KDA) {
            ਅਚਲ ਦਸ਼ਮਲਵ g_val = sycl::native::ਘਾਤਾਂਕ(*g_t);

            // kv[col] = (S^T @ k)[col] = sum_i S[i][col] * k[i]
            ਦਸ਼ਮਲਵ kv_shard = 0.0f;
#ਹਦਾਇਤ unroll
            ਲਈ (ਪੂਰਨ r = 0; r < ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_lane; r++) {
                ਅਚਲ ਪੂਰਨ i = r * warp_ਮਾਪ + lane;
                kv_shard += s_shard[r] * k_t[i];
            }
            ਦਸ਼ਮਲਵ kv_ਖਾਨਾ = warp_reduce_ਜੋੜ<warp_ਮਾਪ>(kv_shard);

            // delta[col] = (v[col] - g * kv[col]) * beta
            ਦਸ਼ਮਲਵ delta_ਖਾਨਾ = (v_t[ਖਾਨਾ] - g_val * kv_ਖਾਨਾ) * beta_val;

            // fused: S[i][col] = g * S[i][col] + k[i] * delta[col]
            // attn[col] = (S^T @ q)[col] = sum_i S[i][col] * q[i]
            ਦਸ਼ਮਲਵ attn_partial = 0.0f;
#ਹਦਾਇਤ unroll
            ਲਈ (ਪੂਰਨ r = 0; r < ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_lane; r++) {
                ਅਚਲ ਪੂਰਨ i = r * warp_ਮਾਪ + lane;
                s_shard[r]  = g_val * s_shard[r] + k_t[i] * delta_ਖਾਨਾ;
                attn_partial += s_shard[r] * q_t[i];
            }

            ਦਸ਼ਮਲਵ attn_ਖਾਨਾ = warp_reduce_ਜੋੜ<warp_ਮਾਪ>(attn_partial);

            ਜੇ (lane == 0) {
                attn_ਅੰਕੜੇ[ਖਾਨਾ] = attn_ਖਾਨਾ * scale;
            }
        } ਨਹੀਂਤਾਂ {
            // kv[col] = sum_i g[i] * S[i][col] * k[i]
            ਦਸ਼ਮਲਵ kv_shard = 0.0f;
#ਹਦਾਇਤ unroll
            ਲਈ (ਪੂਰਨ r = 0; r < ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_lane; r++) {
                ਅਚਲ ਪੂਰਨ i = r * warp_ਮਾਪ + lane;
                kv_shard += sycl::native::ਘਾਤਾਂਕ(g_t[i]) * s_shard[r] * k_t[i];
            }

            ਦਸ਼ਮਲਵ kv_ਖਾਨਾ = warp_reduce_ਜੋੜ<warp_ਮਾਪ>(kv_shard);

            // delta[col] = (v[col] - kv[col]) * beta
            ਦਸ਼ਮਲਵ delta_ਖਾਨਾ = (v_t[ਖਾਨਾ] - kv_ਖਾਨਾ) * beta_val;

            // fused: S[i][col] = g[i] * S[i][col] + k[i] * delta[col]
            // attn[col] = (S^T @ q)[col] = sum_i S[i][col] * q[i]
            ਦਸ਼ਮਲਵ attn_partial = 0.0f;
#ਹਦਾਇਤ unroll
            ਲਈ (ਪੂਰਨ r = 0; r < ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_lane; r++) {
                ਅਚਲ ਪੂਰਨ i = r * warp_ਮਾਪ + lane;
                s_shard[r]  = sycl::native::ਘਾਤਾਂਕ(g_t[i]) * s_shard[r] + k_t[i] * delta_ਖਾਨਾ;
                attn_partial += s_shard[r] * q_t[i];
            }

            ਦਸ਼ਮਲਵ attn_ਖਾਨਾ = warp_reduce_ਜੋੜ<warp_ਮਾਪ>(attn_partial);

            ਜੇ (lane == 0) {
                attn_ਅੰਕੜੇ[ਖਾਨਾ] = attn_ਖਾਨਾ * scale;
            }
        }

        attn_ਅੰਕੜੇ += S_v * H;


    // Write state back to global memory
        ਜੇ constexpr (keep_rs_t) {
            ਅਚਲ ਪੂਰਨ ਨਿਸ਼ਾਨਾ_ਥਾਂ = (ਪੂਰਨ) n_ਟੋਕਨ - 1 - t;
            ਜੇ (ਨਿਸ਼ਾਨਾ_ਥਾਂ >= 0 && ਨਿਸ਼ਾਨਾ_ਥਾਂ < K) {
                ਦਸ਼ਮਲਵ * curr_state = state + ਨਿਸ਼ਾਨਾ_ਥਾਂ * state_ਥਾਂ_stride;
#ਹਦਾਇਤ unroll
                ਲਈ (ਪੂਰਨ r = 0; r < ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_lane; r++) {
                    ਅਚਲ ਪੂਰਨ i = r * warp_ਮਾਪ + lane;
                    curr_state[ਖਾਨਾ * S_v + i] = s_shard[r];
                }
            }
        }
    }

    ਜੇ constexpr (!keep_rs_t) {
#ਹਦਾਇਤ unroll
        ਲਈ (ਪੂਰਨ r = 0; r < ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_lane; r++) {
            ਅਚਲ ਪੂਰਨ i          = r * warp_ਮਾਪ + lane;
            state[ਖਾਨਾ * S_v + i] = s_shard[r];
        }
    }
}

ਸਾਂਚਾ <ਤਾਰਕਿਕ KDA, ਤਾਰਕਿਕ keep_rs_t>
ਸਥਿਰ ਖਾਲੀ launch_gated_delta_net(ਅਚਲ ਦਸ਼ਮਲਵ *   q_d,
                                   ਅਚਲ ਦਸ਼ਮਲਵ *   k_d,
                                   ਅਚਲ ਦਸ਼ਮਲਵ *   v_d,
                                   ਅਚਲ ਦਸ਼ਮਲਵ *   g_d,
                                   ਅਚਲ ਦਸ਼ਮਲਵ *   b_d,
                                   ਅਚਲ ਦਸ਼ਮਲਵ *   s_d,
                                   ਦਸ਼ਮਲਵ *         dst_d,
                                   ਦਸ਼ਮਲਵ *         state_d,
                                   int64_t         S_v,
                                   int64_t         H,
                                   int64_t         n_ਟੋਕਨ,
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
                                   ਦਸ਼ਮਲਵ           scale,
                                   int64_t         state_ਥਾਂ_stride,
                                   ਪੂਰਨ             K,
                                   dpct::ਕਤਾਰ_ptr stream) {
    //TODO: Add chunked kernel for even faster pre-fill
    ਅਚਲ ਪੂਰਨ warp_ਮਾਪ = ggml_sycl_info().devices[ggml_sycl_get_device()].warp_ਮਾਪ;

    ਅਚਲ ਪੂਰਨ ਅੰਕ_warps = 4;
    dpct::dim3 grid_dims(H, n_seqs, (S_v + ਅੰਕ_warps - 1) / ਅੰਕ_warps);
    dpct::dim3 ਖੰਡ_dims(warp_ਮਾਪ <= S_v ? warp_ਮਾਪ : S_v, ਅੰਕ_warps, 1);

    ਅਚਲ sycl::uint3 neqk1_magic = init_fastdiv_ਮੁੱਲਾਂ(neqk1);
    ਅਚਲ sycl::uint3 rq3_magic   = init_fastdiv_ਮੁੱਲਾਂ(rq3);

    ਚੋਣ (S_v) {
        ਹਾਲਤ 16:
            {
                constexpr ਪੂਰਨ sv = 16;
                stream->parallel_for(sycl::nd_range<3>(grid_dims * ਖੰਡ_dims, ਖੰਡ_dims),
                                     [=](sycl::nd_ਵਸਤੂ<3> /*item_ct1*/) [[sycl::reqd_sub_ਸਮੂਹ_ਮਾਪ(WARP_ਮਾਪ)]] {
                                         gated_delta_net_sycl<sv, KDA, keep_rs_t>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H, n_ਟੋਕਨ,
                                                                       sq1, sq2, sq3, sv1, sv2, sv3, sb1, sb2,
                                                                       sb3, neqk1_magic, rq3_magic, scale, state_ਥਾਂ_stride, K);
                                     });
            }
            ਰੋਕੋ;
        ਹਾਲਤ 32:
            {
                constexpr ਪੂਰਨ sv = 32;
                stream->parallel_for(sycl::nd_range<3>(grid_dims * ਖੰਡ_dims, ਖੰਡ_dims),
                                     [=](sycl::nd_ਵਸਤੂ<3> /*item_ct1*/) [[sycl::reqd_sub_ਸਮੂਹ_ਮਾਪ(WARP_ਮਾਪ)]] {
                                         gated_delta_net_sycl<sv, KDA, keep_rs_t>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H, n_ਟੋਕਨ,
                                                                       sq1, sq2, sq3, sv1, sv2, sv3, sb1, sb2,
                                                                       sb3, neqk1_magic, rq3_magic, scale, state_ਥਾਂ_stride, K);
                                     });
            }
            ਰੋਕੋ;
        ਹਾਲਤ 64: {
            {
                constexpr ਪੂਰਨ sv = 64;
                stream->parallel_for(sycl::nd_range<3>(grid_dims * ਖੰਡ_dims, ਖੰਡ_dims),
                                        [=](sycl::nd_ਵਸਤੂ<3> /*item_ct1*/) [[sycl::reqd_sub_ਸਮੂਹ_ਮਾਪ(WARP_ਮਾਪ)]] {
                                            gated_delta_net_sycl<sv, KDA, keep_rs_t>(
                                                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H, n_ਟੋਕਨ, sq1, sq2,
                                                sq3, sv1, sv2, sv3, sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_ਥਾਂ_stride, K);
                                        });
            }
            ਰੋਕੋ;
        }
        ਹਾਲਤ 128: {
            {
                constexpr ਪੂਰਨ sv = 128;
                stream->parallel_for(sycl::nd_range<3>(grid_dims * ਖੰਡ_dims, ਖੰਡ_dims),
                                        [=](sycl::nd_ਵਸਤੂ<3> /*item_ct1*/) [[sycl::reqd_sub_ਸਮੂਹ_ਮਾਪ(WARP_ਮਾਪ)]] {
                                            gated_delta_net_sycl<sv, KDA, keep_rs_t>(
                                                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H, n_ਟੋਕਨ, sq1, sq2,
                                                sq3, sv1, sv2, sv3, sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_ਥਾਂ_stride, K);
                                        });
            }
            ਰੋਕੋ;
        }
        ਮੂਲ:
            GGML_ABORT("fatal error");
            ਰੋਕੋ;
    }
}

ਸਥਿਰ ਖਾਲੀ ggml_sycl_op_gated_delta_net_impl(ggml_backend_sycl_context & ctx, ggml_tensor * dst,
                                              ਅਚਲ ggml_sycl_gated_delta_net_fused_cache * cache) {
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

    ਅਚਲ int64_t S_v      = nev0;
    ਅਚਲ int64_t H        = nev1;
    ਅਚਲ int64_t n_ਟੋਕਨ = nev2;
    ਅਚਲ int64_t n_seqs   = nev3;

    ਅਚਲ ਤਾਰਕਿਕ kda = (src_g->ne[0] == S_v);

    GGML_ASSERT(neq1 == nek1);
    ਅਚਲ int64_t neqk1 = neq1;

    ਅਚਲ int64_t rq3 = nev3 / neq3;

    ਅਚਲ ਦਸ਼ਮਲਵ * q_d = (ਅਚਲ ਦਸ਼ਮਲਵ *) src_q->ਅੰਕੜੇ;
    ਅਚਲ ਦਸ਼ਮਲਵ * k_d = (ਅਚਲ ਦਸ਼ਮਲਵ *) src_k->ਅੰਕੜੇ;
    ਅਚਲ ਦਸ਼ਮਲਵ * v_d = (ਅਚਲ ਦਸ਼ਮਲਵ *) src_v->ਅੰਕੜੇ;
    ਅਚਲ ਦਸ਼ਮਲਵ * g_d = (ਅਚਲ ਦਸ਼ਮਲਵ *) src_g->ਅੰਕੜੇ;
    ਅਚਲ ਦਸ਼ਮਲਵ * b_d = (ਅਚਲ ਦਸ਼ਮਲਵ *) src_beta->ਅੰਕੜੇ;

    ਅਚਲ ਦਸ਼ਮਲਵ * s_d   = (ਅਚਲ ਦਸ਼ਮਲਵ *) src_state->ਅੰਕੜੇ;
    ਦਸ਼ਮਲਵ *       dst_d = (ਦਸ਼ਮਲਵ *) dst->ਅੰਕੜੇ;

    GGML_ASSERT(ggml_is_contiguous_ਪੰਕਤੀਆਂ(src_q));
    GGML_ASSERT(ggml_is_contiguous_ਪੰਕਤੀਆਂ(src_k));
    GGML_ASSERT(ggml_is_contiguous_ਪੰਕਤੀਆਂ(src_v));
    GGML_ASSERT(ggml_are_same_stride(src_q, src_k));
    GGML_ASSERT(src_g->ne[0] == 1 || kda);
    GGML_ASSERT(ggml_is_contiguous(src_g));
    GGML_ASSERT(ggml_is_contiguous(src_beta));
    GGML_ASSERT(ggml_is_contiguous(src_state));

    // strides in floats (beta strides used for both g and beta offset computation)
    ਅਚਲ int64_t sq1 = nbq1 / ਆਕਾਰ(ਦਸ਼ਮਲਵ);
    ਅਚਲ int64_t sq2 = nbq2 / ਆਕਾਰ(ਦਸ਼ਮਲਵ);
    ਅਚਲ int64_t sq3 = nbq3 / ਆਕਾਰ(ਦਸ਼ਮਲਵ);
    ਅਚਲ int64_t sv1 = nbv1 / ਆਕਾਰ(ਦਸ਼ਮਲਵ);
    ਅਚਲ int64_t sv2 = nbv2 / ਆਕਾਰ(ਦਸ਼ਮਲਵ);
    ਅਚਲ int64_t sv3 = nbv3 / ਆਕਾਰ(ਦਸ਼ਮਲਵ);
    ਅਚਲ int64_t sb1 = nbb1 / ਆਕਾਰ(ਦਸ਼ਮਲਵ);
    ਅਚਲ int64_t sb2 = nbb2 / ਆਕਾਰ(ਦਸ਼ਮਲਵ);
    ਅਚਲ int64_t sb3 = nbb3 / ਆਕਾਰ(ਦਸ਼ਮਲਵ);

    ਅਚਲ ਦਸ਼ਮਲਵ scale = 1.0f / sqrtf((ਦਸ਼ਮਲਵ) S_v);

    dpct::ਕਤਾਰ_ptr stream = ctx.stream();

    // K (snapshot slot count) is an op param; state holds s0 only [S_v, S_v, H, n_seqs].
    ਅਚਲ ਪੂਰਨ K = ggml_get_op_params_i32(dst, 0);
    ਅਚਲ ਤਾਰਕਿਕ keep_rs = K > 1;

    // recurrent state -> dst tail (after attention scores), or the cache when fusing
    ਦਸ਼ਮਲਵ * state_d           = dst_d + S_v * H * n_ਟੋਕਨ * n_seqs;
    int64_t state_ਥਾਂ_stride = S_v * S_v * H * n_seqs;
    ਜੇ (cache != ਖਾਲੀ_ਸੂਚਕ) {
        state_d           = cache->ਅੰਕੜੇ;
        state_ਥਾਂ_stride = cache->ਥਾਂ_stride;
    }

    ਜੇ (kda) {
        ਜੇ (keep_rs) {
            launch_gated_delta_net<ਸੱਚ, ਸੱਚ>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_ਟੋਕਨ, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_ਥਾਂ_stride, K, stream);
        } ਨਹੀਂਤਾਂ {
            launch_gated_delta_net<ਸੱਚ, ਝੂਠ>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_ਟੋਕਨ, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_ਥਾਂ_stride, K, stream);
        }
    } ਨਹੀਂਤਾਂ {
        ਜੇ (keep_rs) {
            launch_gated_delta_net<ਝੂਠ, ਸੱਚ>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_ਟੋਕਨ, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_ਥਾਂ_stride, K, stream);
        } ਨਹੀਂਤਾਂ {
            launch_gated_delta_net<ਝੂਠ, ਝੂਠ>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_ਟੋਕਨ, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_ਥਾਂ_stride, K, stream);
        }
    }
}

ਖਾਲੀ ggml_sycl_op_gated_delta_net(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_op_gated_delta_net_impl(ctx, dst, ਖਾਲੀ_ਸੂਚਕ);
}

ਖਾਲੀ ggml_sycl_gated_delta_net(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/6);
    ggml_sycl_op_gated_delta_net(ctx, dst);
}

ਖਾਲੀ ggml_sycl_op_gated_delta_net_fused_cache(ggml_backend_sycl_context & ctx, ggml_tensor * dst,
                                              ggml_sycl_gated_delta_net_fused_cache cache) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/6);
    ggml_sycl_op_gated_delta_net_impl(ctx, dst, &cache);
}
