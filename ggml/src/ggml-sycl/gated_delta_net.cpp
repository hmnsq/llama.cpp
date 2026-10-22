#അന്തര്ഭാവ <sycl/sycl.hpp>
#അന്തര്ഭാവ "dpct/helper.hpp"
#അന്തര്ഭാവ "common.hpp"
#അന്തര്ഭാവ "ggml.h"
#അന്തര്ഭാവ "gated_delta_net.hpp"
#അന്തര്ഭാവ <cmath>


template <പൂർണ്ണസംഖ്യ S_v, തർക്കം KDA, തർക്കം keep_rs_t>
ശൂന്യം gated_delta_net_sycl(സ്ഥിരം ദശാംശം *     q,
                          സ്ഥിരം ദശാംശം *     k,
                          സ്ഥിരം ദശാംശം *     v,
                          സ്ഥിരം ദശാംശം *     g,
                          സ്ഥിരം ദശാംശം *     beta,
                          സ്ഥിരം ദശാംശം *     curr_state,
                          ദശാംശം *           dst,
                          ദശാംശം *           state,
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
                          സ്ഥിരം sycl::uint3 neqk1_magic,
                          സ്ഥിരം sycl::uint3 rq3_magic,
                          ദശാംശം             scale,
                          int64_t           state_slot_stride,
                          പൂർണ്ണസംഖ്യ               K) {
    auto           item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    സ്ഥിരം uint32_t h_idx    = item_ct1.get_group(2);
    സ്ഥിരം uint32_t sequence = item_ct1.get_group(1);
    // each warp owns one column, using warp-level primitives to reduce across rows
    സ്ഥിരം പൂർണ്ണസംഖ്യ      lane     = item_ct1.get_local_id(2);
    സ്ഥിരം പൂർണ്ണസംഖ്യ      col      = item_ct1.get_group(0) * item_ct1.get_local_range(1) + item_ct1.get_local_id(1);

    സ്ഥിരം uint32_t iq1 = fastmodulo(h_idx, neqk1_magic);
    സ്ഥിരം uint32_t iq3 = fastdiv(sequence, rq3_magic);

    ദശാംശം *       attn_data        = dst;

    // input state holds s0 only [S_v, S_v, H, n_seqs] — seq stride is D = H * S_v * S_v.
    // output state layout (per-slot D * n_seqs) — same per-(seq,head) offset as before.
    സ്ഥിരം int64_t state_in_offset      = sequence * H * S_v * S_v + h_idx * S_v * S_v;
    സ്ഥിരം int64_t state_out_offset     = (sequence * H + h_idx) * S_v * S_v;
    state += state_out_offset;
    curr_state += state_in_offset + col * S_v;
    attn_data += (sequence * n_tokens * H + h_idx) * S_v;

    constexpr പൂർണ്ണസംഖ്യ warp_size = ggml_sycl_get_physical_warp_size() < S_v ? ggml_sycl_get_physical_warp_size() : S_v;
    static_assert(S_v % warp_size == 0, "S_v must be a multiple of warp_size");
    constexpr പൂർണ്ണസംഖ്യ rows_per_lane = (S_v + warp_size - 1) / warp_size;
    ദശാംശം         s_shard[rows_per_lane];
#pragma unroll
    വേണ്ടി (പൂർണ്ണസംഖ്യ r = 0; r < rows_per_lane; r++) {
        സ്ഥിരം പൂർണ്ണസംഖ്യ i = r * warp_size + lane;
        s_shard[r]  = curr_state[i];
    }

    // snapshot slot mapping: slot 0 = most recent state, slot s = s tokens back.
    // When n_tokens < K only slots 0..n_tokens-1 are written; older slots are caller-owned.

    വേണ്ടി (പൂർണ്ണസംഖ്യ t = 0; t < n_tokens; t++) {
        സ്ഥിരം ദശാംശം * q_t = q + iq3 * sq3 + t * sq2 + iq1 * sq1;
        സ്ഥിരം ദശാംശം * k_t = k + iq3 * sq3 + t * sq2 + iq1 * sq1;
        സ്ഥിരം ദശാംശം * v_t = v + sequence * sv3 + t * sv2 + h_idx * sv1;

        സ്ഥിരം int64_t gb_offset = sequence * sb3 + t * sb2 + h_idx * sb1;
        സ്ഥിരം ദശാംശം * beta_t = beta + gb_offset;
        സ്ഥിരം ദശാംശം * g_t    = g    + gb_offset * (KDA ? S_v : 1);

        സ്ഥിരം ദശാംശം beta_val = *beta_t;

        എങ്കിൽ constexpr (!KDA) {
            സ്ഥിരം ദശാംശം g_val = sycl::native::exp(*g_t);

            // kv[col] = (S^T @ k)[col] = sum_i S[i][col] * k[i]
            ദശാംശം kv_shard = 0.0f;
#pragma unroll
            വേണ്ടി (പൂർണ്ണസംഖ്യ r = 0; r < rows_per_lane; r++) {
                സ്ഥിരം പൂർണ്ണസംഖ്യ i = r * warp_size + lane;
                kv_shard += s_shard[r] * k_t[i];
            }
            ദശാംശം kv_col = warp_reduce_sum<warp_size>(kv_shard);

            // delta[col] = (v[col] - g * kv[col]) * beta
            ദശാംശം delta_col = (v_t[col] - g_val * kv_col) * beta_val;

            // fused: S[i][col] = g * S[i][col] + k[i] * delta[col]
            // attn[col] = (S^T @ q)[col] = sum_i S[i][col] * q[i]
            ദശാംശം attn_partial = 0.0f;
#pragma unroll
            വേണ്ടി (പൂർണ്ണസംഖ്യ r = 0; r < rows_per_lane; r++) {
                സ്ഥിരം പൂർണ്ണസംഖ്യ i = r * warp_size + lane;
                s_shard[r]  = g_val * s_shard[r] + k_t[i] * delta_col;
                attn_partial += s_shard[r] * q_t[i];
            }

            ദശാംശം attn_col = warp_reduce_sum<warp_size>(attn_partial);

            എങ്കിൽ (lane == 0) {
                attn_data[col] = attn_col * scale;
            }
        } അല്ലെങ്കിൽ {
            // kv[col] = sum_i g[i] * S[i][col] * k[i]
            ദശാംശം kv_shard = 0.0f;
#pragma unroll
            വേണ്ടി (പൂർണ്ണസംഖ്യ r = 0; r < rows_per_lane; r++) {
                സ്ഥിരം പൂർണ്ണസംഖ്യ i = r * warp_size + lane;
                kv_shard += sycl::native::exp(g_t[i]) * s_shard[r] * k_t[i];
            }

            ദശാംശം kv_col = warp_reduce_sum<warp_size>(kv_shard);

            // delta[col] = (v[col] - kv[col]) * beta
            ദശാംശം delta_col = (v_t[col] - kv_col) * beta_val;

            // fused: S[i][col] = g[i] * S[i][col] + k[i] * delta[col]
            // attn[col] = (S^T @ q)[col] = sum_i S[i][col] * q[i]
            ദശാംശം attn_partial = 0.0f;
#pragma unroll
            വേണ്ടി (പൂർണ്ണസംഖ്യ r = 0; r < rows_per_lane; r++) {
                സ്ഥിരം പൂർണ്ണസംഖ്യ i = r * warp_size + lane;
                s_shard[r]  = sycl::native::exp(g_t[i]) * s_shard[r] + k_t[i] * delta_col;
                attn_partial += s_shard[r] * q_t[i];
            }

            ദശാംശം attn_col = warp_reduce_sum<warp_size>(attn_partial);

            എങ്കിൽ (lane == 0) {
                attn_data[col] = attn_col * scale;
            }
        }

        attn_data += S_v * H;


    // Write state back to global memory
        എങ്കിൽ constexpr (keep_rs_t) {
            സ്ഥിരം പൂർണ്ണസംഖ്യ target_slot = (പൂർണ്ണസംഖ്യ) n_tokens - 1 - t;
            എങ്കിൽ (target_slot >= 0 && target_slot < K) {
                ദശാംശം * curr_state = state + target_slot * state_slot_stride;
#pragma unroll
                വേണ്ടി (പൂർണ്ണസംഖ്യ r = 0; r < rows_per_lane; r++) {
                    സ്ഥിരം പൂർണ്ണസംഖ്യ i = r * warp_size + lane;
                    curr_state[col * S_v + i] = s_shard[r];
                }
            }
        }
    }

    എങ്കിൽ constexpr (!keep_rs_t) {
#pragma unroll
        വേണ്ടി (പൂർണ്ണസംഖ്യ r = 0; r < rows_per_lane; r++) {
            സ്ഥിരം പൂർണ്ണസംഖ്യ i          = r * warp_size + lane;
            state[col * S_v + i] = s_shard[r];
        }
    }
}

template <തർക്കം KDA, തർക്കം keep_rs_t>
സ്ഥിതം ശൂന്യം launch_gated_delta_net(സ്ഥിരം ദശാംശം *   q_d,
                                   സ്ഥിരം ദശാംശം *   k_d,
                                   സ്ഥിരം ദശാംശം *   v_d,
                                   സ്ഥിരം ദശാംശം *   g_d,
                                   സ്ഥിരം ദശാംശം *   b_d,
                                   സ്ഥിരം ദശാംശം *   s_d,
                                   ദശാംശം *         dst_d,
                                   ദശാംശം *         state_d,
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
                                   ദശാംശം           scale,
                                   int64_t         state_slot_stride,
                                   പൂർണ്ണസംഖ്യ             K,
                                   dpct::queue_ptr stream) {
    //TODO: Add chunked kernel for even faster pre-fill
    സ്ഥിരം പൂർണ്ണസംഖ്യ warp_size = ggml_sycl_info().devices[ggml_sycl_get_device()].warp_size;

    സ്ഥിരം പൂർണ്ണസംഖ്യ അക്കം_warps = 4;
    dpct::dim3 grid_dims(H, n_seqs, (S_v + അക്കം_warps - 1) / അക്കം_warps);
    dpct::dim3 block_dims(warp_size <= S_v ? warp_size : S_v, അക്കം_warps, 1);

    സ്ഥിരം sycl::uint3 neqk1_magic = init_fastdiv_മൂല്യങ്ങൾ(neqk1);
    സ്ഥിരം sycl::uint3 rq3_magic   = init_fastdiv_മൂല്യങ്ങൾ(rq3);

    തിരഞ്ഞെടുപ്പ് (S_v) {
        അവസ്ഥ 16:
            {
                constexpr പൂർണ്ണസംഖ്യ sv = 16;
                stream->parallel_for(sycl::nd_range<3>(grid_dims * block_dims, block_dims),
                                     [=](sycl::nd_item<3> /*item_ct1*/) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                                         gated_delta_net_sycl<sv, KDA, keep_rs_t>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H, n_tokens,
                                                                       sq1, sq2, sq3, sv1, sv2, sv3, sb1, sb2,
                                                                       sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K);
                                     });
            }
            നിർത്തുക;
        അവസ്ഥ 32:
            {
                constexpr പൂർണ്ണസംഖ്യ sv = 32;
                stream->parallel_for(sycl::nd_range<3>(grid_dims * block_dims, block_dims),
                                     [=](sycl::nd_item<3> /*item_ct1*/) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                                         gated_delta_net_sycl<sv, KDA, keep_rs_t>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H, n_tokens,
                                                                       sq1, sq2, sq3, sv1, sv2, sv3, sb1, sb2,
                                                                       sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K);
                                     });
            }
            നിർത്തുക;
        അവസ്ഥ 64: {
            {
                constexpr പൂർണ്ണസംഖ്യ sv = 64;
                stream->parallel_for(sycl::nd_range<3>(grid_dims * block_dims, block_dims),
                                        [=](sycl::nd_item<3> /*item_ct1*/) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                                            gated_delta_net_sycl<sv, KDA, keep_rs_t>(
                                                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H, n_tokens, sq1, sq2,
                                                sq3, sv1, sv2, sv3, sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K);
                                        });
            }
            നിർത്തുക;
        }
        അവസ്ഥ 128: {
            {
                constexpr പൂർണ്ണസംഖ്യ sv = 128;
                stream->parallel_for(sycl::nd_range<3>(grid_dims * block_dims, block_dims),
                                        [=](sycl::nd_item<3> /*item_ct1*/) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                                            gated_delta_net_sycl<sv, KDA, keep_rs_t>(
                                                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H, n_tokens, sq1, sq2,
                                                sq3, sv1, sv2, sv3, sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K);
                                        });
            }
            നിർത്തുക;
        }
        സ്ഥിരസ്ഥിതി:
            GGML_ABORT("fatal error");
            നിർത്തുക;
    }
}

സ്ഥിതം ശൂന്യം ggml_sycl_op_gated_delta_net_impl(ggml_backend_sycl_context & ctx, ggml_tensor * dst,
                                              സ്ഥിരം ggml_sycl_gated_delta_net_fused_cache * cache) {
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

    സ്ഥിരം int64_t S_v      = nev0;
    സ്ഥിരം int64_t H        = nev1;
    സ്ഥിരം int64_t n_tokens = nev2;
    സ്ഥിരം int64_t n_seqs   = nev3;

    സ്ഥിരം തർക്കം kda = (src_g->ne[0] == S_v);

    GGML_ASSERT(neq1 == nek1);
    സ്ഥിരം int64_t neqk1 = neq1;

    സ്ഥിരം int64_t rq3 = nev3 / neq3;

    സ്ഥിരം ദശാംശം * q_d = (സ്ഥിരം ദശാംശം *) src_q->data;
    സ്ഥിരം ദശാംശം * k_d = (സ്ഥിരം ദശാംശം *) src_k->data;
    സ്ഥിരം ദശാംശം * v_d = (സ്ഥിരം ദശാംശം *) src_v->data;
    സ്ഥിരം ദശാംശം * g_d = (സ്ഥിരം ദശാംശം *) src_g->data;
    സ്ഥിരം ദശാംശം * b_d = (സ്ഥിരം ദശാംശം *) src_beta->data;

    സ്ഥിരം ദശാംശം * s_d   = (സ്ഥിരം ദശാംശം *) src_state->data;
    ദശാംശം *       dst_d = (ദശാംശം *) dst->data;

    GGML_ASSERT(ggml_is_contiguous_rows(src_q));
    GGML_ASSERT(ggml_is_contiguous_rows(src_k));
    GGML_ASSERT(ggml_is_contiguous_rows(src_v));
    GGML_ASSERT(ggml_are_same_stride(src_q, src_k));
    GGML_ASSERT(src_g->ne[0] == 1 || kda);
    GGML_ASSERT(ggml_is_contiguous(src_g));
    GGML_ASSERT(ggml_is_contiguous(src_beta));
    GGML_ASSERT(ggml_is_contiguous(src_state));

    // strides in floats (beta strides used for both g and beta offset computation)
    സ്ഥിരം int64_t sq1 = nbq1 / വലുപ്പം(ദശാംശം);
    സ്ഥിരം int64_t sq2 = nbq2 / വലുപ്പം(ദശാംശം);
    സ്ഥിരം int64_t sq3 = nbq3 / വലുപ്പം(ദശാംശം);
    സ്ഥിരം int64_t sv1 = nbv1 / വലുപ്പം(ദശാംശം);
    സ്ഥിരം int64_t sv2 = nbv2 / വലുപ്പം(ദശാംശം);
    സ്ഥിരം int64_t sv3 = nbv3 / വലുപ്പം(ദശാംശം);
    സ്ഥിരം int64_t sb1 = nbb1 / വലുപ്പം(ദശാംശം);
    സ്ഥിരം int64_t sb2 = nbb2 / വലുപ്പം(ദശാംശം);
    സ്ഥിരം int64_t sb3 = nbb3 / വലുപ്പം(ദശാംശം);

    സ്ഥിരം ദശാംശം scale = 1.0f / sqrtf((ദശാംശം) S_v);

    dpct::queue_ptr stream = ctx.stream();

    // K (snapshot slot count) is an op param; state holds s0 only [S_v, S_v, H, n_seqs].
    സ്ഥിരം പൂർണ്ണസംഖ്യ K = ggml_get_op_params_i32(dst, 0);
    സ്ഥിരം തർക്കം keep_rs = K > 1;

    // recurrent state -> dst tail (after attention scores), or the cache when fusing
    ദശാംശം * state_d           = dst_d + S_v * H * n_tokens * n_seqs;
    int64_t state_slot_stride = S_v * S_v * H * n_seqs;
    എങ്കിൽ (cache != nullptr) {
        state_d           = cache->data;
        state_slot_stride = cache->slot_stride;
    }

    എങ്കിൽ (kda) {
        എങ്കിൽ (keep_rs) {
            launch_gated_delta_net<സത്യം, സത്യം>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, stream);
        } അല്ലെങ്കിൽ {
            launch_gated_delta_net<സത്യം, അസത്യം>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, stream);
        }
    } അല്ലെങ്കിൽ {
        എങ്കിൽ (keep_rs) {
            launch_gated_delta_net<അസത്യം, സത്യം>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, stream);
        } അല്ലെങ്കിൽ {
            launch_gated_delta_net<അസത്യം, അസത്യം>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, stream);
        }
    }
}

ശൂന്യം ggml_sycl_op_gated_delta_net(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_op_gated_delta_net_impl(ctx, dst, nullptr);
}

ശൂന്യം ggml_sycl_gated_delta_net(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/6);
    ggml_sycl_op_gated_delta_net(ctx, dst);
}

ശൂന്യം ggml_sycl_op_gated_delta_net_fused_cache(ggml_backend_sycl_context & ctx, ggml_tensor * dst,
                                              ggml_sycl_gated_delta_net_fused_cache cache) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/6);
    ggml_sycl_op_gated_delta_net_impl(ctx, dst, &cache);
}
