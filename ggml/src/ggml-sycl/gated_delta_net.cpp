#અન્તર્ભાવ <sycl/sycl.hpp>
#અન્તર્ભાવ "dpct/helper.hpp"
#અન્તર્ભાવ "common.hpp"
#અન્તર્ભાવ "ggml.h"
#અન્તર્ભાવ "gated_delta_net.hpp"
#અન્તર્ભાવ <cmath>


template <પૂર્ણાંક S_v, તાર્કિક KDA, તાર્કિક keep_rs_t>
શૂન્ય gated_delta_net_sycl(અચલ દશાંશ *     q,
                          અચલ દશાંશ *     k,
                          અચલ દશાંશ *     v,
                          અચલ દશાંશ *     g,
                          અચલ દશાંશ *     beta,
                          અચલ દશાંશ *     curr_state,
                          દશાંશ *           dst,
                          દશાંશ *           state,
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
                          અચલ sycl::uint3 neqk1_magic,
                          અચલ sycl::uint3 rq3_magic,
                          દશાંશ             scale,
                          int64_t           state_slot_stride,
                          પૂર્ણાંક               K) {
    auto           item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    અચલ uint32_t h_idx    = item_ct1.get_group(2);
    અચલ uint32_t sequence = item_ct1.get_group(1);
    // each warp owns one column, using warp-level primitives to reduce across rows
    અચલ પૂર્ણાંક      lane     = item_ct1.get_local_id(2);
    અચલ પૂર્ણાંક      col      = item_ct1.get_group(0) * item_ct1.get_local_range(1) + item_ct1.get_local_id(1);

    અચલ uint32_t iq1 = fastmodulo(h_idx, neqk1_magic);
    અચલ uint32_t iq3 = fastdiv(sequence, rq3_magic);

    દશાંશ *       attn_data        = dst;

    // input state holds s0 only [S_v, S_v, H, n_seqs] — seq stride is D = H * S_v * S_v.
    // output state layout (per-slot D * n_seqs) — same per-(seq,head) offset as before.
    અચલ int64_t state_in_offset      = sequence * H * S_v * S_v + h_idx * S_v * S_v;
    અચલ int64_t state_out_offset     = (sequence * H + h_idx) * S_v * S_v;
    state += state_out_offset;
    curr_state += state_in_offset + col * S_v;
    attn_data += (sequence * n_tokens * H + h_idx) * S_v;

    constexpr પૂર્ણાંક warp_size = ggml_sycl_get_physical_warp_size() < S_v ? ggml_sycl_get_physical_warp_size() : S_v;
    static_assert(S_v % warp_size == 0, "S_v must be a multiple of warp_size");
    constexpr પૂર્ણાંક rows_per_lane = (S_v + warp_size - 1) / warp_size;
    દશાંશ         s_shard[rows_per_lane];
#pragma unroll
    માટે (પૂર્ણાંક r = 0; r < rows_per_lane; r++) {
        અચલ પૂર્ણાંક i = r * warp_size + lane;
        s_shard[r]  = curr_state[i];
    }

    // snapshot slot mapping: slot 0 = most recent state, slot s = s tokens back.
    // When n_tokens < K only slots 0..n_tokens-1 are written; older slots are caller-owned.

    માટે (પૂર્ણાંક t = 0; t < n_tokens; t++) {
        અચલ દશાંશ * q_t = q + iq3 * sq3 + t * sq2 + iq1 * sq1;
        અચલ દશાંશ * k_t = k + iq3 * sq3 + t * sq2 + iq1 * sq1;
        અચલ દશાંશ * v_t = v + sequence * sv3 + t * sv2 + h_idx * sv1;

        અચલ int64_t gb_offset = sequence * sb3 + t * sb2 + h_idx * sb1;
        અચલ દશાંશ * beta_t = beta + gb_offset;
        અચલ દશાંશ * g_t    = g    + gb_offset * (KDA ? S_v : 1);

        અચલ દશાંશ beta_val = *beta_t;

        જો constexpr (!KDA) {
            અચલ દશાંશ g_val = sycl::native::exp(*g_t);

            // kv[col] = (S^T @ k)[col] = sum_i S[i][col] * k[i]
            દશાંશ kv_shard = 0.0f;
#pragma unroll
            માટે (પૂર્ણાંક r = 0; r < rows_per_lane; r++) {
                અચલ પૂર્ણાંક i = r * warp_size + lane;
                kv_shard += s_shard[r] * k_t[i];
            }
            દશાંશ kv_col = warp_reduce_sum<warp_size>(kv_shard);

            // delta[col] = (v[col] - g * kv[col]) * beta
            દશાંશ delta_col = (v_t[col] - g_val * kv_col) * beta_val;

            // fused: S[i][col] = g * S[i][col] + k[i] * delta[col]
            // attn[col] = (S^T @ q)[col] = sum_i S[i][col] * q[i]
            દશાંશ attn_partial = 0.0f;
#pragma unroll
            માટે (પૂર્ણાંક r = 0; r < rows_per_lane; r++) {
                અચલ પૂર્ણાંક i = r * warp_size + lane;
                s_shard[r]  = g_val * s_shard[r] + k_t[i] * delta_col;
                attn_partial += s_shard[r] * q_t[i];
            }

            દશાંશ attn_col = warp_reduce_sum<warp_size>(attn_partial);

            જો (lane == 0) {
                attn_data[col] = attn_col * scale;
            }
        } નહીંતર {
            // kv[col] = sum_i g[i] * S[i][col] * k[i]
            દશાંશ kv_shard = 0.0f;
#pragma unroll
            માટે (પૂર્ણાંક r = 0; r < rows_per_lane; r++) {
                અચલ પૂર્ણાંક i = r * warp_size + lane;
                kv_shard += sycl::native::exp(g_t[i]) * s_shard[r] * k_t[i];
            }

            દશાંશ kv_col = warp_reduce_sum<warp_size>(kv_shard);

            // delta[col] = (v[col] - kv[col]) * beta
            દશાંશ delta_col = (v_t[col] - kv_col) * beta_val;

            // fused: S[i][col] = g[i] * S[i][col] + k[i] * delta[col]
            // attn[col] = (S^T @ q)[col] = sum_i S[i][col] * q[i]
            દશાંશ attn_partial = 0.0f;
#pragma unroll
            માટે (પૂર્ણાંક r = 0; r < rows_per_lane; r++) {
                અચલ પૂર્ણાંક i = r * warp_size + lane;
                s_shard[r]  = sycl::native::exp(g_t[i]) * s_shard[r] + k_t[i] * delta_col;
                attn_partial += s_shard[r] * q_t[i];
            }

            દશાંશ attn_col = warp_reduce_sum<warp_size>(attn_partial);

            જો (lane == 0) {
                attn_data[col] = attn_col * scale;
            }
        }

        attn_data += S_v * H;


    // Write state back to global memory
        જો constexpr (keep_rs_t) {
            અચલ પૂર્ણાંક target_slot = (પૂર્ણાંક) n_tokens - 1 - t;
            જો (target_slot >= 0 && target_slot < K) {
                દશાંશ * curr_state = state + target_slot * state_slot_stride;
#pragma unroll
                માટે (પૂર્ણાંક r = 0; r < rows_per_lane; r++) {
                    અચલ પૂર્ણાંક i = r * warp_size + lane;
                    curr_state[col * S_v + i] = s_shard[r];
                }
            }
        }
    }

    જો constexpr (!keep_rs_t) {
#pragma unroll
        માટે (પૂર્ણાંક r = 0; r < rows_per_lane; r++) {
            અચલ પૂર્ણાંક i          = r * warp_size + lane;
            state[col * S_v + i] = s_shard[r];
        }
    }
}

template <તાર્કિક KDA, તાર્કિક keep_rs_t>
સ્થિર શૂન્ય launch_gated_delta_net(અચલ દશાંશ *   q_d,
                                   અચલ દશાંશ *   k_d,
                                   અચલ દશાંશ *   v_d,
                                   અચલ દશાંશ *   g_d,
                                   અચલ દશાંશ *   b_d,
                                   અચલ દશાંશ *   s_d,
                                   દશાંશ *         dst_d,
                                   દશાંશ *         state_d,
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
                                   દશાંશ           scale,
                                   int64_t         state_slot_stride,
                                   પૂર્ણાંક             K,
                                   dpct::queue_ptr stream) {
    //TODO: Add chunked kernel for even faster pre-fill
    અચલ પૂર્ણાંક warp_size = ggml_sycl_info().devices[ggml_sycl_get_device()].warp_size;

    અચલ પૂર્ણાંક અંક_warps = 4;
    dpct::dim3 grid_dims(H, n_seqs, (S_v + અંક_warps - 1) / અંક_warps);
    dpct::dim3 block_dims(warp_size <= S_v ? warp_size : S_v, અંક_warps, 1);

    અચલ sycl::uint3 neqk1_magic = init_fastdiv_કિંમતો(neqk1);
    અચલ sycl::uint3 rq3_magic   = init_fastdiv_કિંમતો(rq3);

    પસંદગી (S_v) {
        સ્થિતિ 16:
            {
                constexpr પૂર્ણાંક sv = 16;
                stream->parallel_for(sycl::nd_range<3>(grid_dims * block_dims, block_dims),
                                     [=](sycl::nd_item<3> /*item_ct1*/) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                                         gated_delta_net_sycl<sv, KDA, keep_rs_t>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H, n_tokens,
                                                                       sq1, sq2, sq3, sv1, sv2, sv3, sb1, sb2,
                                                                       sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K);
                                     });
            }
            તોડો;
        સ્થિતિ 32:
            {
                constexpr પૂર્ણાંક sv = 32;
                stream->parallel_for(sycl::nd_range<3>(grid_dims * block_dims, block_dims),
                                     [=](sycl::nd_item<3> /*item_ct1*/) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                                         gated_delta_net_sycl<sv, KDA, keep_rs_t>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H, n_tokens,
                                                                       sq1, sq2, sq3, sv1, sv2, sv3, sb1, sb2,
                                                                       sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K);
                                     });
            }
            તોડો;
        સ્થિતિ 64: {
            {
                constexpr પૂર્ણાંક sv = 64;
                stream->parallel_for(sycl::nd_range<3>(grid_dims * block_dims, block_dims),
                                        [=](sycl::nd_item<3> /*item_ct1*/) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                                            gated_delta_net_sycl<sv, KDA, keep_rs_t>(
                                                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H, n_tokens, sq1, sq2,
                                                sq3, sv1, sv2, sv3, sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K);
                                        });
            }
            તોડો;
        }
        સ્થિતિ 128: {
            {
                constexpr પૂર્ણાંક sv = 128;
                stream->parallel_for(sycl::nd_range<3>(grid_dims * block_dims, block_dims),
                                        [=](sycl::nd_item<3> /*item_ct1*/) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                                            gated_delta_net_sycl<sv, KDA, keep_rs_t>(
                                                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H, n_tokens, sq1, sq2,
                                                sq3, sv1, sv2, sv3, sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K);
                                        });
            }
            તોડો;
        }
        મૂળભૂત:
            GGML_ABORT("fatal error");
            તોડો;
    }
}

સ્થિર શૂન્ય ggml_sycl_op_gated_delta_net_impl(ggml_backend_sycl_context & ctx, ggml_tensor * dst,
                                              અચલ ggml_sycl_gated_delta_net_fused_cache * cache) {
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

    અચલ int64_t S_v      = nev0;
    અચલ int64_t H        = nev1;
    અચલ int64_t n_tokens = nev2;
    અચલ int64_t n_seqs   = nev3;

    અચલ તાર્કિક kda = (src_g->ne[0] == S_v);

    GGML_ASSERT(neq1 == nek1);
    અચલ int64_t neqk1 = neq1;

    અચલ int64_t rq3 = nev3 / neq3;

    અચલ દશાંશ * q_d = (અચલ દશાંશ *) src_q->data;
    અચલ દશાંશ * k_d = (અચલ દશાંશ *) src_k->data;
    અચલ દશાંશ * v_d = (અચલ દશાંશ *) src_v->data;
    અચલ દશાંશ * g_d = (અચલ દશાંશ *) src_g->data;
    અચલ દશાંશ * b_d = (અચલ દશાંશ *) src_beta->data;

    અચલ દશાંશ * s_d   = (અચલ દશાંશ *) src_state->data;
    દશાંશ *       dst_d = (દશાંશ *) dst->data;

    GGML_ASSERT(ggml_is_contiguous_rows(src_q));
    GGML_ASSERT(ggml_is_contiguous_rows(src_k));
    GGML_ASSERT(ggml_is_contiguous_rows(src_v));
    GGML_ASSERT(ggml_are_same_stride(src_q, src_k));
    GGML_ASSERT(src_g->ne[0] == 1 || kda);
    GGML_ASSERT(ggml_is_contiguous(src_g));
    GGML_ASSERT(ggml_is_contiguous(src_beta));
    GGML_ASSERT(ggml_is_contiguous(src_state));

    // strides in floats (beta strides used for both g and beta offset computation)
    અચલ int64_t sq1 = nbq1 / કદ(દશાંશ);
    અચલ int64_t sq2 = nbq2 / કદ(દશાંશ);
    અચલ int64_t sq3 = nbq3 / કદ(દશાંશ);
    અચલ int64_t sv1 = nbv1 / કદ(દશાંશ);
    અચલ int64_t sv2 = nbv2 / કદ(દશાંશ);
    અચલ int64_t sv3 = nbv3 / કદ(દશાંશ);
    અચલ int64_t sb1 = nbb1 / કદ(દશાંશ);
    અચલ int64_t sb2 = nbb2 / કદ(દશાંશ);
    અચલ int64_t sb3 = nbb3 / કદ(દશાંશ);

    અચલ દશાંશ scale = 1.0f / sqrtf((દશાંશ) S_v);

    dpct::queue_ptr stream = ctx.stream();

    // K (snapshot slot count) is an op param; state holds s0 only [S_v, S_v, H, n_seqs].
    અચલ પૂર્ણાંક K = ggml_get_op_params_i32(dst, 0);
    અચલ તાર્કિક keep_rs = K > 1;

    // recurrent state -> dst tail (after attention scores), or the cache when fusing
    દશાંશ * state_d           = dst_d + S_v * H * n_tokens * n_seqs;
    int64_t state_slot_stride = S_v * S_v * H * n_seqs;
    જો (cache != nullptr) {
        state_d           = cache->data;
        state_slot_stride = cache->slot_stride;
    }

    જો (kda) {
        જો (keep_rs) {
            launch_gated_delta_net<સાચું, સાચું>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, stream);
        } નહીંતર {
            launch_gated_delta_net<સાચું, ખોટું>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, stream);
        }
    } નહીંતર {
        જો (keep_rs) {
            launch_gated_delta_net<ખોટું, સાચું>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, stream);
        } નહીંતર {
            launch_gated_delta_net<ખોટું, ખોટું>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, stream);
        }
    }
}

શૂન્ય ggml_sycl_op_gated_delta_net(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_op_gated_delta_net_impl(ctx, dst, nullptr);
}

શૂન્ય ggml_sycl_gated_delta_net(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/6);
    ggml_sycl_op_gated_delta_net(ctx, dst);
}

શૂન્ય ggml_sycl_op_gated_delta_net_fused_cache(ggml_backend_sycl_context & ctx, ggml_tensor * dst,
                                              ggml_sycl_gated_delta_net_fused_cache cache) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/6);
    ggml_sycl_op_gated_delta_net_impl(ctx, dst, &cache);
}
