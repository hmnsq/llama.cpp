#ಅನ್ತರ್ಭಾವ <sycl/sycl.hpp>
#ಅನ್ತರ್ಭಾವ "dpct/helper.hpp"
#ಅನ್ತರ್ಭಾವ "common.hpp"
#ಅನ್ತರ್ಭಾವ "ggml.h"
#ಅನ್ತರ್ಭಾವ "gated_delta_net.hpp"
#ಅನ್ತರ್ಭಾವ <cmath>


template <ಪೂರ್ಣಾಙ್ಕ S_v, ತಾರ್ಕಿಕ KDA, ತಾರ್ಕಿಕ keep_rs_t>
ಶೂನ್ಯ gated_delta_net_sycl(ಧ್ರುವ ಭಿನ್ನ *     q,
                          ಧ್ರುವ ಭಿನ್ನ *     k,
                          ಧ್ರುವ ಭಿನ್ನ *     v,
                          ಧ್ರುವ ಭಿನ್ನ *     g,
                          ಧ್ರುವ ಭಿನ್ನ *     beta,
                          ಧ್ರುವ ಭಿನ್ನ *     curr_state,
                          ಭಿನ್ನ *           dst,
                          ಭಿನ್ನ *           state,
                          int64_t           H,
                          int64_t           n_tokens,
                          int64_t           sq1,
                          int64_t           sq2,
                          int64_t           sq3,
                          int64_t           sv1,
                          int64_t           sv2,
                          int64_t           sv3,
                          int64_t           sb1,
                          int64_t           sb2,
                          int64_t           sb3,
                          ಧ್ರುವ sycl::uint3 neqk1_magic,
                          ಧ್ರುವ sycl::uint3 rq3_magic,
                          ಭಿನ್ನ             scale,
                          int64_t           state_slot_stride,
                          ಪೂರ್ಣಾಙ್ಕ               K) {
    auto           item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    ಧ್ರುವ uint32_t h_idx    = item_ct1.get_group(2);
    ಧ್ರುವ uint32_t sequence = item_ct1.get_group(1);
    // each warp owns one column, using warp-level primitives to reduce across rows
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ      lane     = item_ct1.get_local_id(2);
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ      col      = item_ct1.get_group(0) * item_ct1.get_local_range(1) + item_ct1.get_local_id(1);

    ಧ್ರುವ uint32_t iq1 = fastmodulo(h_idx, neqk1_magic);
    ಧ್ರುವ uint32_t iq3 = fastdiv(sequence, rq3_magic);

    ಭಿನ್ನ *       attn_data        = dst;

    // input state holds s0 only [S_v, S_v, H, n_seqs] — seq stride is D = H * S_v * S_v.
    // output state layout (per-slot D * n_seqs) — same per-(seq,head) offset as before.
    ಧ್ರುವ int64_t state_in_offset      = sequence * H * S_v * S_v + h_idx * S_v * S_v;
    ಧ್ರುವ int64_t state_out_offset     = (sequence * H + h_idx) * S_v * S_v;
    state += state_out_offset;
    curr_state += state_in_offset + col * S_v;
    attn_data += (sequence * n_tokens * H + h_idx) * S_v;

    constexpr ಪೂರ್ಣಾಙ್ಕ warp_size = ggml_sycl_get_physical_warp_size() < S_v ? ggml_sycl_get_physical_warp_size() : S_v;
    static_assert(S_v % warp_size == 0, "S_v must be a multiple of warp_size");
    constexpr ಪೂರ್ಣಾಙ್ಕ rows_per_lane = (S_v + warp_size - 1) / warp_size;
    ಭಿನ್ನ         s_shard[rows_per_lane];
#pragma unroll
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ r = 0; r < rows_per_lane; r++) {
        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ i = r * warp_size + lane;
        s_shard[r]  = curr_state[i];
    }

    // snapshot slot mapping: slot 0 = most recent state, slot s = s tokens back.
    // When n_tokens < K only slots 0..n_tokens-1 are written; older slots are caller-owned.

    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ t = 0; t < n_tokens; t++) {
        ಧ್ರುವ ಭಿನ್ನ * q_t = q + iq3 * sq3 + t * sq2 + iq1 * sq1;
        ಧ್ರುವ ಭಿನ್ನ * k_t = k + iq3 * sq3 + t * sq2 + iq1 * sq1;
        ಧ್ರುವ ಭಿನ್ನ * v_t = v + sequence * sv3 + t * sv2 + h_idx * sv1;

        ಧ್ರುವ int64_t gb_offset = sequence * sb3 + t * sb2 + h_idx * sb1;
        ಧ್ರುವ ಭಿನ್ನ * beta_t = beta + gb_offset;
        ಧ್ರುವ ಭಿನ್ನ * g_t    = g    + gb_offset * (KDA ? S_v : 1);

        ಧ್ರುವ ಭಿನ್ನ beta_val = *beta_t;

        ಯದಿ constexpr (!KDA) {
            ಧ್ರುವ ಭಿನ್ನ g_val = sycl::native::exp(*g_t);

            // kv[col] = (S^T @ k)[col] = sum_i S[i][col] * k[i]
            ಭಿನ್ನ kv_shard = 0.0f;
#pragma unroll
            ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ r = 0; r < rows_per_lane; r++) {
                ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ i = r * warp_size + lane;
                kv_shard += s_shard[r] * k_t[i];
            }
            ಭಿನ್ನ kv_col = warp_reduce_sum<warp_size>(kv_shard);

            // delta[col] = (v[col] - g * kv[col]) * beta
            ಭಿನ್ನ delta_col = (v_t[col] - g_val * kv_col) * beta_val;

            // fused: S[i][col] = g * S[i][col] + k[i] * delta[col]
            // attn[col] = (S^T @ q)[col] = sum_i S[i][col] * q[i]
            ಭಿನ್ನ attn_partial = 0.0f;
#pragma unroll
            ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ r = 0; r < rows_per_lane; r++) {
                ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ i = r * warp_size + lane;
                s_shard[r]  = g_val * s_shard[r] + k_t[i] * delta_col;
                attn_partial += s_shard[r] * q_t[i];
            }

            ಭಿನ್ನ attn_col = warp_reduce_sum<warp_size>(attn_partial);

            ಯದಿ (lane == 0) {
                attn_data[col] = attn_col * scale;
            }
        } ಅನ್ಯಥಾ {
            // kv[col] = sum_i g[i] * S[i][col] * k[i]
            ಭಿನ್ನ kv_shard = 0.0f;
#pragma unroll
            ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ r = 0; r < rows_per_lane; r++) {
                ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ i = r * warp_size + lane;
                kv_shard += sycl::native::exp(g_t[i]) * s_shard[r] * k_t[i];
            }

            ಭಿನ್ನ kv_col = warp_reduce_sum<warp_size>(kv_shard);

            // delta[col] = (v[col] - kv[col]) * beta
            ಭಿನ್ನ delta_col = (v_t[col] - kv_col) * beta_val;

            // fused: S[i][col] = g[i] * S[i][col] + k[i] * delta[col]
            // attn[col] = (S^T @ q)[col] = sum_i S[i][col] * q[i]
            ಭಿನ್ನ attn_partial = 0.0f;
#pragma unroll
            ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ r = 0; r < rows_per_lane; r++) {
                ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ i = r * warp_size + lane;
                s_shard[r]  = sycl::native::exp(g_t[i]) * s_shard[r] + k_t[i] * delta_col;
                attn_partial += s_shard[r] * q_t[i];
            }

            ಭಿನ್ನ attn_col = warp_reduce_sum<warp_size>(attn_partial);

            ಯದಿ (lane == 0) {
                attn_data[col] = attn_col * scale;
            }
        }

        attn_data += S_v * H;


    // Write state back to global memory
        ಯದಿ constexpr (keep_rs_t) {
            ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ target_slot = (ಪೂರ್ಣಾಙ್ಕ) n_tokens - 1 - t;
            ಯದಿ (target_slot >= 0 && target_slot < K) {
                ಭಿನ್ನ * curr_state = state + target_slot * state_slot_stride;
#pragma unroll
                ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ r = 0; r < rows_per_lane; r++) {
                    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ i = r * warp_size + lane;
                    curr_state[col * S_v + i] = s_shard[r];
                }
            }
        }
    }

    ಯದಿ constexpr (!keep_rs_t) {
#pragma unroll
        ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ r = 0; r < rows_per_lane; r++) {
            ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ i          = r * warp_size + lane;
            state[col * S_v + i] = s_shard[r];
        }
    }
}

template <ತಾರ್ಕಿಕ KDA, ತಾರ್ಕಿಕ keep_rs_t>
ಸ್ಥಿರ ಶೂನ್ಯ launch_gated_delta_net(ಧ್ರುವ ಭಿನ್ನ *   q_d,
                                   ಧ್ರುವ ಭಿನ್ನ *   k_d,
                                   ಧ್ರುವ ಭಿನ್ನ *   v_d,
                                   ಧ್ರುವ ಭಿನ್ನ *   g_d,
                                   ಧ್ರುವ ಭಿನ್ನ *   b_d,
                                   ಧ್ರುವ ಭಿನ್ನ *   s_d,
                                   ಭಿನ್ನ *         dst_d,
                                   ಭಿನ್ನ *         state_d,
                                   int64_t         S_v,
                                   int64_t         H,
                                   int64_t         n_tokens,
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
                                   ಭಿನ್ನ           scale,
                                   int64_t         state_slot_stride,
                                   ಪೂರ್ಣಾಙ್ಕ             K,
                                   dpct::queue_ptr stream) {
    //TODO: Add chunked kernel for even faster pre-fill
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ warp_size = ggml_sycl_info().devices[ggml_sycl_get_device()].warp_size;

    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ಅಂಕೆ_warps = 4;
    dpct::dim3 grid_dims(H, n_seqs, (S_v + ಅಂಕೆ_warps - 1) / ಅಂಕೆ_warps);
    dpct::dim3 block_dims(warp_size <= S_v ? warp_size : S_v, ಅಂಕೆ_warps, 1);

    ಧ್ರುವ sycl::uint3 neqk1_magic = init_fastdiv_ಮೌಲ್ಯಗಳು(neqk1);
    ಧ್ರುವ sycl::uint3 rq3_magic   = init_fastdiv_ಮೌಲ್ಯಗಳು(rq3);

    ವಿಕಲ್ಪ (S_v) {
        ಪಕ್ಷ 16:
            {
                constexpr ಪೂರ್ಣಾಙ್ಕ sv = 16;
                stream->parallel_for(sycl::nd_range<3>(grid_dims * block_dims, block_dims),
                                     [=](sycl::nd_item<3> /*item_ct1*/) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                                         gated_delta_net_sycl<sv, KDA, keep_rs_t>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H, n_tokens,
                                                                       sq1, sq2, sq3, sv1, sv2, sv3, sb1, sb2,
                                                                       sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K);
                                     });
            }
            ಭಙ್ಗ;
        ಪಕ್ಷ 32:
            {
                constexpr ಪೂರ್ಣಾಙ್ಕ sv = 32;
                stream->parallel_for(sycl::nd_range<3>(grid_dims * block_dims, block_dims),
                                     [=](sycl::nd_item<3> /*item_ct1*/) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                                         gated_delta_net_sycl<sv, KDA, keep_rs_t>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H, n_tokens,
                                                                       sq1, sq2, sq3, sv1, sv2, sv3, sb1, sb2,
                                                                       sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K);
                                     });
            }
            ಭಙ್ಗ;
        ಪಕ್ಷ 64: {
            {
                constexpr ಪೂರ್ಣಾಙ್ಕ sv = 64;
                stream->parallel_for(sycl::nd_range<3>(grid_dims * block_dims, block_dims),
                                        [=](sycl::nd_item<3> /*item_ct1*/) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                                            gated_delta_net_sycl<sv, KDA, keep_rs_t>(
                                                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H, n_tokens, sq1, sq2,
                                                sq3, sv1, sv2, sv3, sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K);
                                        });
            }
            ಭಙ್ಗ;
        }
        ಪಕ್ಷ 128: {
            {
                constexpr ಪೂರ್ಣಾಙ್ಕ sv = 128;
                stream->parallel_for(sycl::nd_range<3>(grid_dims * block_dims, block_dims),
                                        [=](sycl::nd_item<3> /*item_ct1*/) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                                            gated_delta_net_sycl<sv, KDA, keep_rs_t>(
                                                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H, n_tokens, sq1, sq2,
                                                sq3, sv1, sv2, sv3, sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K);
                                        });
            }
            ಭಙ್ಗ;
        }
        ಸ್ವಭಾವ:
            GGML_ABORT("fatal error");
            ಭಙ್ಗ;
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_sycl_op_gated_delta_net_impl(ggml_backend_sycl_context & ctx, ggml_tensor * dst,
                                              ಧ್ರುವ ggml_sycl_gated_delta_net_fused_cache * cache) {
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

    ಧ್ರುವ int64_t S_v      = nev0;
    ಧ್ರುವ int64_t H        = nev1;
    ಧ್ರುವ int64_t n_tokens = nev2;
    ಧ್ರುವ int64_t n_seqs   = nev3;

    ಧ್ರುವ ತಾರ್ಕಿಕ kda = (src_g->ne[0] == S_v);

    GGML_ASSERT(neq1 == nek1);
    ಧ್ರುವ int64_t neqk1 = neq1;

    ಧ್ರುವ int64_t rq3 = nev3 / neq3;

    ಧ್ರುವ ಭಿನ್ನ * q_d = (ಧ್ರುವ ಭಿನ್ನ *) src_q->data;
    ಧ್ರುವ ಭಿನ್ನ * k_d = (ಧ್ರುವ ಭಿನ್ನ *) src_k->data;
    ಧ್ರುವ ಭಿನ್ನ * v_d = (ಧ್ರುವ ಭಿನ್ನ *) src_v->data;
    ಧ್ರುವ ಭಿನ್ನ * g_d = (ಧ್ರುವ ಭಿನ್ನ *) src_g->data;
    ಧ್ರುವ ಭಿನ್ನ * b_d = (ಧ್ರುವ ಭಿನ್ನ *) src_beta->data;

    ಧ್ರುವ ಭಿನ್ನ * s_d   = (ಧ್ರುವ ಭಿನ್ನ *) src_state->data;
    ಭಿನ್ನ *       dst_d = (ಭಿನ್ನ *) dst->data;

    GGML_ASSERT(ggml_is_contiguous_rows(src_q));
    GGML_ASSERT(ggml_is_contiguous_rows(src_k));
    GGML_ASSERT(ggml_is_contiguous_rows(src_v));
    GGML_ASSERT(ggml_are_same_stride(src_q, src_k));
    GGML_ASSERT(src_g->ne[0] == 1 || kda);
    GGML_ASSERT(ggml_is_contiguous(src_g));
    GGML_ASSERT(ggml_is_contiguous(src_beta));
    GGML_ASSERT(ggml_is_contiguous(src_state));

    // strides in floats (beta strides used for both g and beta offset computation)
    ಧ್ರುವ int64_t sq1 = nbq1 / ಪರಿಮಾಣ(ಭಿನ್ನ);
    ಧ್ರುವ int64_t sq2 = nbq2 / ಪರಿಮಾಣ(ಭಿನ್ನ);
    ಧ್ರುವ int64_t sq3 = nbq3 / ಪರಿಮಾಣ(ಭಿನ್ನ);
    ಧ್ರುವ int64_t sv1 = nbv1 / ಪರಿಮಾಣ(ಭಿನ್ನ);
    ಧ್ರುವ int64_t sv2 = nbv2 / ಪರಿಮಾಣ(ಭಿನ್ನ);
    ಧ್ರುವ int64_t sv3 = nbv3 / ಪರಿಮಾಣ(ಭಿನ್ನ);
    ಧ್ರುವ int64_t sb1 = nbb1 / ಪರಿಮಾಣ(ಭಿನ್ನ);
    ಧ್ರುವ int64_t sb2 = nbb2 / ಪರಿಮಾಣ(ಭಿನ್ನ);
    ಧ್ರುವ int64_t sb3 = nbb3 / ಪರಿಮಾಣ(ಭಿನ್ನ);

    ಧ್ರುವ ಭಿನ್ನ scale = 1.0f / sqrtf((ಭಿನ್ನ) S_v);

    dpct::queue_ptr stream = ctx.stream();

    // K (snapshot slot count) is an op param; state holds s0 only [S_v, S_v, H, n_seqs].
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ K = ggml_get_op_params_i32(dst, 0);
    ಧ್ರುವ ತಾರ್ಕಿಕ keep_rs = K > 1;

    // recurrent state -> dst tail (after attention scores), or the cache when fusing
    ಭಿನ್ನ * state_d           = dst_d + S_v * H * n_tokens * n_seqs;
    int64_t state_slot_stride = S_v * S_v * H * n_seqs;
    ಯದಿ (cache != nullptr) {
        state_d           = cache->data;
        state_slot_stride = cache->slot_stride;
    }

    ಯದಿ (kda) {
        ಯದಿ (keep_rs) {
            launch_gated_delta_net<ಸತ್ಯ, ಸತ್ಯ>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, stream);
        } ಅನ್ಯಥಾ {
            launch_gated_delta_net<ಸತ್ಯ, ಅಸತ್ಯ>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, stream);
        }
    } ಅನ್ಯಥಾ {
        ಯದಿ (keep_rs) {
            launch_gated_delta_net<ಅಸತ್ಯ, ಸತ್ಯ>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, stream);
        } ಅನ್ಯಥಾ {
            launch_gated_delta_net<ಅಸತ್ಯ, ಅಸತ್ಯ>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, stream);
        }
    }
}

ಶೂನ್ಯ ggml_sycl_op_gated_delta_net(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_op_gated_delta_net_impl(ctx, dst, nullptr);
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
