#ଅନ୍ତର୍ଭାଵ <sycl/sycl.hpp>
#ଅନ୍ତର୍ଭାଵ "dpct/helper.hpp"
#ଅନ୍ତର୍ଭାଵ "common.hpp"
#ଅନ୍ତର୍ଭାଵ "ggml.h"
#ଅନ୍ତର୍ଭାଵ "gated_delta_net.hpp"
#ଅନ୍ତର୍ଭାଵ <cmath>


template <ପୂର୍ଣ୍ଣସଂଖ୍ୟା S_v, ତାର୍କିକ KDA, ତାର୍କିକ keep_rs_t>
ଶୂନ୍ୟ gated_delta_net_sycl(ଅଚଳ ଦଶମିକ *     q,
                          ଅଚଳ ଦଶମିକ *     k,
                          ଅଚଳ ଦଶମିକ *     v,
                          ଅଚଳ ଦଶମିକ *     g,
                          ଅଚଳ ଦଶମିକ *     beta,
                          ଅଚଳ ଦଶମିକ *     curr_state,
                          ଦଶମିକ *           dst,
                          ଦଶମିକ *           state,
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
                          ଅଚଳ sycl::uint3 neqk1_magic,
                          ଅଚଳ sycl::uint3 rq3_magic,
                          ଦଶମିକ             scale,
                          int64_t           state_slot_stride,
                          ପୂର୍ଣ୍ଣସଂଖ୍ୟା               K) {
    auto           item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    ଅଚଳ uint32_t h_idx    = item_ct1.get_group(2);
    ଅଚଳ uint32_t sequence = item_ct1.get_group(1);
    // each warp owns one column, using warp-level primitives to reduce across rows
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା      lane     = item_ct1.get_local_id(2);
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା      col      = item_ct1.get_group(0) * item_ct1.get_local_range(1) + item_ct1.get_local_id(1);

    ଅଚଳ uint32_t iq1 = fastmodulo(h_idx, neqk1_magic);
    ଅଚଳ uint32_t iq3 = fastdiv(sequence, rq3_magic);

    ଦଶମିକ *       attn_data        = dst;

    // input state holds s0 only [S_v, S_v, H, n_seqs] — seq stride is D = H * S_v * S_v.
    // output state layout (per-slot D * n_seqs) — same per-(seq,head) offset as before.
    ଅଚଳ int64_t state_in_offset      = sequence * H * S_v * S_v + h_idx * S_v * S_v;
    ଅଚଳ int64_t state_out_offset     = (sequence * H + h_idx) * S_v * S_v;
    state += state_out_offset;
    curr_state += state_in_offset + col * S_v;
    attn_data += (sequence * n_tokens * H + h_idx) * S_v;

    constexpr ପୂର୍ଣ୍ଣସଂଖ୍ୟା warp_size = ggml_sycl_get_physical_warp_size() < S_v ? ggml_sycl_get_physical_warp_size() : S_v;
    static_assert(S_v % warp_size == 0, "S_v must be a multiple of warp_size");
    constexpr ପୂର୍ଣ୍ଣସଂଖ୍ୟା rows_per_lane = (S_v + warp_size - 1) / warp_size;
    ଦଶମିକ         s_shard[rows_per_lane];
#pragma unroll
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା r = 0; r < rows_per_lane; r++) {
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = r * warp_size + lane;
        s_shard[r]  = curr_state[i];
    }

    // snapshot slot mapping: slot 0 = most recent state, slot s = s tokens back.
    // When n_tokens < K only slots 0..n_tokens-1 are written; older slots are caller-owned.

    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା t = 0; t < n_tokens; t++) {
        ଅଚଳ ଦଶମିକ * q_t = q + iq3 * sq3 + t * sq2 + iq1 * sq1;
        ଅଚଳ ଦଶମିକ * k_t = k + iq3 * sq3 + t * sq2 + iq1 * sq1;
        ଅଚଳ ଦଶମିକ * v_t = v + sequence * sv3 + t * sv2 + h_idx * sv1;

        ଅଚଳ int64_t gb_offset = sequence * sb3 + t * sb2 + h_idx * sb1;
        ଅଚଳ ଦଶମିକ * beta_t = beta + gb_offset;
        ଅଚଳ ଦଶମିକ * g_t    = g    + gb_offset * (KDA ? S_v : 1);

        ଅଚଳ ଦଶମିକ beta_val = *beta_t;

        ଯଦି constexpr (!KDA) {
            ଅଚଳ ଦଶମିକ g_val = sycl::native::exp(*g_t);

            // kv[col] = (S^T @ k)[col] = sum_i S[i][col] * k[i]
            ଦଶମିକ kv_shard = 0.0f;
#pragma unroll
            ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା r = 0; r < rows_per_lane; r++) {
                ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = r * warp_size + lane;
                kv_shard += s_shard[r] * k_t[i];
            }
            ଦଶମିକ kv_col = warp_reduce_sum<warp_size>(kv_shard);

            // delta[col] = (v[col] - g * kv[col]) * beta
            ଦଶମିକ delta_col = (v_t[col] - g_val * kv_col) * beta_val;

            // fused: S[i][col] = g * S[i][col] + k[i] * delta[col]
            // attn[col] = (S^T @ q)[col] = sum_i S[i][col] * q[i]
            ଦଶମିକ attn_partial = 0.0f;
#pragma unroll
            ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା r = 0; r < rows_per_lane; r++) {
                ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = r * warp_size + lane;
                s_shard[r]  = g_val * s_shard[r] + k_t[i] * delta_col;
                attn_partial += s_shard[r] * q_t[i];
            }

            ଦଶମିକ attn_col = warp_reduce_sum<warp_size>(attn_partial);

            ଯଦି (lane == 0) {
                attn_data[col] = attn_col * scale;
            }
        } ଅନ୍ୟଥା {
            // kv[col] = sum_i g[i] * S[i][col] * k[i]
            ଦଶମିକ kv_shard = 0.0f;
#pragma unroll
            ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା r = 0; r < rows_per_lane; r++) {
                ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = r * warp_size + lane;
                kv_shard += sycl::native::exp(g_t[i]) * s_shard[r] * k_t[i];
            }

            ଦଶମିକ kv_col = warp_reduce_sum<warp_size>(kv_shard);

            // delta[col] = (v[col] - kv[col]) * beta
            ଦଶମିକ delta_col = (v_t[col] - kv_col) * beta_val;

            // fused: S[i][col] = g[i] * S[i][col] + k[i] * delta[col]
            // attn[col] = (S^T @ q)[col] = sum_i S[i][col] * q[i]
            ଦଶମିକ attn_partial = 0.0f;
#pragma unroll
            ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା r = 0; r < rows_per_lane; r++) {
                ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = r * warp_size + lane;
                s_shard[r]  = sycl::native::exp(g_t[i]) * s_shard[r] + k_t[i] * delta_col;
                attn_partial += s_shard[r] * q_t[i];
            }

            ଦଶମିକ attn_col = warp_reduce_sum<warp_size>(attn_partial);

            ଯଦି (lane == 0) {
                attn_data[col] = attn_col * scale;
            }
        }

        attn_data += S_v * H;


    // Write state back to global memory
        ଯଦି constexpr (keep_rs_t) {
            ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା target_slot = (ପୂର୍ଣ୍ଣସଂଖ୍ୟା) n_tokens - 1 - t;
            ଯଦି (target_slot >= 0 && target_slot < K) {
                ଦଶମିକ * curr_state = state + target_slot * state_slot_stride;
#pragma unroll
                ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା r = 0; r < rows_per_lane; r++) {
                    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = r * warp_size + lane;
                    curr_state[col * S_v + i] = s_shard[r];
                }
            }
        }
    }

    ଯଦି constexpr (!keep_rs_t) {
#pragma unroll
        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା r = 0; r < rows_per_lane; r++) {
            ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା i          = r * warp_size + lane;
            state[col * S_v + i] = s_shard[r];
        }
    }
}

template <ତାର୍କିକ KDA, ତାର୍କିକ keep_rs_t>
ସ୍ଥିର ଶୂନ୍ୟ launch_gated_delta_net(ଅଚଳ ଦଶମିକ *   q_d,
                                   ଅଚଳ ଦଶମିକ *   k_d,
                                   ଅଚଳ ଦଶମିକ *   v_d,
                                   ଅଚଳ ଦଶମିକ *   g_d,
                                   ଅଚଳ ଦଶମିକ *   b_d,
                                   ଅଚଳ ଦଶମିକ *   s_d,
                                   ଦଶମିକ *         dst_d,
                                   ଦଶମିକ *         state_d,
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
                                   ଦଶମିକ           scale,
                                   int64_t         state_slot_stride,
                                   ପୂର୍ଣ୍ଣସଂଖ୍ୟା             K,
                                   dpct::queue_ptr stream) {
    //TODO: Add chunked kernel for even faster pre-fill
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା warp_size = ggml_sycl_info().devices[ggml_sycl_get_device()].warp_size;

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ଅଙ୍କ_warps = 4;
    dpct::dim3 grid_dims(H, n_seqs, (S_v + ଅଙ୍କ_warps - 1) / ଅଙ୍କ_warps);
    dpct::dim3 block_dims(warp_size <= S_v ? warp_size : S_v, ଅଙ୍କ_warps, 1);

    ଅଚଳ sycl::uint3 neqk1_magic = init_fastdiv_ମୂଲ୍ୟଗୁଡ଼ିକ(neqk1);
    ଅଚଳ sycl::uint3 rq3_magic   = init_fastdiv_ମୂଲ୍ୟଗୁଡ଼ିକ(rq3);

    ବାଛ (S_v) {
        ସ୍ଥିତି 16:
            {
                constexpr ପୂର୍ଣ୍ଣସଂଖ୍ୟା sv = 16;
                stream->parallel_for(sycl::nd_range<3>(grid_dims * block_dims, block_dims),
                                     [=](sycl::nd_item<3> /*item_ct1*/) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                                         gated_delta_net_sycl<sv, KDA, keep_rs_t>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H, n_tokens,
                                                                       sq1, sq2, sq3, sv1, sv2, sv3, sb1, sb2,
                                                                       sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K);
                                     });
            }
            ଭାଙ୍ଗ;
        ସ୍ଥିତି 32:
            {
                constexpr ପୂର୍ଣ୍ଣସଂଖ୍ୟା sv = 32;
                stream->parallel_for(sycl::nd_range<3>(grid_dims * block_dims, block_dims),
                                     [=](sycl::nd_item<3> /*item_ct1*/) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                                         gated_delta_net_sycl<sv, KDA, keep_rs_t>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H, n_tokens,
                                                                       sq1, sq2, sq3, sv1, sv2, sv3, sb1, sb2,
                                                                       sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K);
                                     });
            }
            ଭାଙ୍ଗ;
        ସ୍ଥିତି 64: {
            {
                constexpr ପୂର୍ଣ୍ଣସଂଖ୍ୟା sv = 64;
                stream->parallel_for(sycl::nd_range<3>(grid_dims * block_dims, block_dims),
                                        [=](sycl::nd_item<3> /*item_ct1*/) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                                            gated_delta_net_sycl<sv, KDA, keep_rs_t>(
                                                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H, n_tokens, sq1, sq2,
                                                sq3, sv1, sv2, sv3, sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K);
                                        });
            }
            ଭାଙ୍ଗ;
        }
        ସ୍ଥିତି 128: {
            {
                constexpr ପୂର୍ଣ୍ଣସଂଖ୍ୟା sv = 128;
                stream->parallel_for(sycl::nd_range<3>(grid_dims * block_dims, block_dims),
                                        [=](sycl::nd_item<3> /*item_ct1*/) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                                            gated_delta_net_sycl<sv, KDA, keep_rs_t>(
                                                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H, n_tokens, sq1, sq2,
                                                sq3, sv1, sv2, sv3, sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K);
                                        });
            }
            ଭାଙ୍ଗ;
        }
        ପୂର୍ବନିର୍ଦ୍ଧାରିତ:
            GGML_ABORT("fatal error");
            ଭାଙ୍ଗ;
    }
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_sycl_op_gated_delta_net_impl(ggml_backend_sycl_context & ctx, ggml_tensor * dst,
                                              ଅଚଳ ggml_sycl_gated_delta_net_fused_cache * cache) {
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

    ଅଚଳ int64_t S_v      = nev0;
    ଅଚଳ int64_t H        = nev1;
    ଅଚଳ int64_t n_tokens = nev2;
    ଅଚଳ int64_t n_seqs   = nev3;

    ଅଚଳ ତାର୍କିକ kda = (src_g->ne[0] == S_v);

    GGML_ASSERT(neq1 == nek1);
    ଅଚଳ int64_t neqk1 = neq1;

    ଅଚଳ int64_t rq3 = nev3 / neq3;

    ଅଚଳ ଦଶମିକ * q_d = (ଅଚଳ ଦଶମିକ *) src_q->data;
    ଅଚଳ ଦଶମିକ * k_d = (ଅଚଳ ଦଶମିକ *) src_k->data;
    ଅଚଳ ଦଶମିକ * v_d = (ଅଚଳ ଦଶମିକ *) src_v->data;
    ଅଚଳ ଦଶମିକ * g_d = (ଅଚଳ ଦଶମିକ *) src_g->data;
    ଅଚଳ ଦଶମିକ * b_d = (ଅଚଳ ଦଶମିକ *) src_beta->data;

    ଅଚଳ ଦଶମିକ * s_d   = (ଅଚଳ ଦଶମିକ *) src_state->data;
    ଦଶମିକ *       dst_d = (ଦଶମିକ *) dst->data;

    GGML_ASSERT(ggml_is_contiguous_rows(src_q));
    GGML_ASSERT(ggml_is_contiguous_rows(src_k));
    GGML_ASSERT(ggml_is_contiguous_rows(src_v));
    GGML_ASSERT(ggml_are_same_stride(src_q, src_k));
    GGML_ASSERT(src_g->ne[0] == 1 || kda);
    GGML_ASSERT(ggml_is_contiguous(src_g));
    GGML_ASSERT(ggml_is_contiguous(src_beta));
    GGML_ASSERT(ggml_is_contiguous(src_state));

    // strides in floats (beta strides used for both g and beta offset computation)
    ଅଚଳ int64_t sq1 = nbq1 / ଆକାର(ଦଶମିକ);
    ଅଚଳ int64_t sq2 = nbq2 / ଆକାର(ଦଶମିକ);
    ଅଚଳ int64_t sq3 = nbq3 / ଆକାର(ଦଶମିକ);
    ଅଚଳ int64_t sv1 = nbv1 / ଆକାର(ଦଶମିକ);
    ଅଚଳ int64_t sv2 = nbv2 / ଆକାର(ଦଶମିକ);
    ଅଚଳ int64_t sv3 = nbv3 / ଆକାର(ଦଶମିକ);
    ଅଚଳ int64_t sb1 = nbb1 / ଆକାର(ଦଶମିକ);
    ଅଚଳ int64_t sb2 = nbb2 / ଆକାର(ଦଶମିକ);
    ଅଚଳ int64_t sb3 = nbb3 / ଆକାର(ଦଶମିକ);

    ଅଚଳ ଦଶମିକ scale = 1.0f / sqrtf((ଦଶମିକ) S_v);

    dpct::queue_ptr stream = ctx.stream();

    // K (snapshot slot count) is an op param; state holds s0 only [S_v, S_v, H, n_seqs].
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା K = ggml_get_op_params_i32(dst, 0);
    ଅଚଳ ତାର୍କିକ keep_rs = K > 1;

    // recurrent state -> dst tail (after attention scores), or the cache when fusing
    ଦଶମିକ * state_d           = dst_d + S_v * H * n_tokens * n_seqs;
    int64_t state_slot_stride = S_v * S_v * H * n_seqs;
    ଯଦି (cache != nullptr) {
        state_d           = cache->data;
        state_slot_stride = cache->slot_stride;
    }

    ଯଦି (kda) {
        ଯଦି (keep_rs) {
            launch_gated_delta_net<ସତ୍ୟ, ସତ୍ୟ>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, stream);
        } ଅନ୍ୟଥା {
            launch_gated_delta_net<ସତ୍ୟ, ମିଥ୍ୟା>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, stream);
        }
    } ଅନ୍ୟଥା {
        ଯଦି (keep_rs) {
            launch_gated_delta_net<ମିଥ୍ୟା, ସତ୍ୟ>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, stream);
        } ଅନ୍ୟଥା {
            launch_gated_delta_net<ମିଥ୍ୟା, ମିଥ୍ୟା>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, stream);
        }
    }
}

ଶୂନ୍ୟ ggml_sycl_op_gated_delta_net(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_op_gated_delta_net_impl(ctx, dst, nullptr);
}

ଶୂନ୍ୟ ggml_sycl_gated_delta_net(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/6);
    ggml_sycl_op_gated_delta_net(ctx, dst);
}

ଶୂନ୍ୟ ggml_sycl_op_gated_delta_net_fused_cache(ggml_backend_sycl_context & ctx, ggml_tensor * dst,
                                              ggml_sycl_gated_delta_net_fused_cache cache) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/6);
    ggml_sycl_op_gated_delta_net_impl(ctx, dst, &cache);
}
