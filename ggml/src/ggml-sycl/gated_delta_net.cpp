#অন্তর্ভাৱ <sycl/sycl.hpp>
#অন্তর্ভাৱ "dpct/helper.hpp"
#অন্তর্ভাৱ "common.hpp"
#অন্তর্ভাৱ "ggml.h"
#অন্তর্ভাৱ "gated_delta_net.hpp"
#অন্তর্ভাৱ <cmath>


template <পূর্ণাঙ্ক S_v, তার্কিক KDA, তার্কিক keep_rs_t>
শূন্য gated_delta_net_sycl(ধ্রুৱ ভিন্ন *     q,
                          ধ্রুৱ ভিন্ন *     k,
                          ধ্রুৱ ভিন্ন *     v,
                          ধ্রুৱ ভিন্ন *     g,
                          ধ্রুৱ ভিন্ন *     beta,
                          ধ্রুৱ ভিন্ন *     curr_state,
                          ভিন্ন *           dst,
                          ভিন্ন *           state,
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
                          ধ্রুৱ sycl::uint3 neqk1_magic,
                          ধ্রুৱ sycl::uint3 rq3_magic,
                          ভিন্ন             scale,
                          int64_t           state_slot_stride,
                          পূর্ণাঙ্ক               K) {
    auto           item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    ধ্রুৱ uint32_t h_idx    = item_ct1.get_group(2);
    ধ্রুৱ uint32_t sequence = item_ct1.get_group(1);
    // each warp owns one column, using warp-level primitives to reduce across rows
    ধ্রুৱ পূর্ণাঙ্ক      lane     = item_ct1.get_local_id(2);
    ধ্রুৱ পূর্ণাঙ্ক      col      = item_ct1.get_group(0) * item_ct1.get_local_range(1) + item_ct1.get_local_id(1);

    ধ্রুৱ uint32_t iq1 = fastmodulo(h_idx, neqk1_magic);
    ধ্রুৱ uint32_t iq3 = fastdiv(sequence, rq3_magic);

    ভিন্ন *       attn_data        = dst;

    // input state holds s0 only [S_v, S_v, H, n_seqs] — seq stride is D = H * S_v * S_v.
    // output state layout (per-slot D * n_seqs) — same per-(seq,head) offset as before.
    ধ্রুৱ int64_t state_in_offset      = sequence * H * S_v * S_v + h_idx * S_v * S_v;
    ধ্রুৱ int64_t state_out_offset     = (sequence * H + h_idx) * S_v * S_v;
    state += state_out_offset;
    curr_state += state_in_offset + col * S_v;
    attn_data += (sequence * n_tokens * H + h_idx) * S_v;

    constexpr পূর্ণাঙ্ক warp_size = ggml_sycl_get_physical_warp_size() < S_v ? ggml_sycl_get_physical_warp_size() : S_v;
    static_assert(S_v % warp_size == 0, "S_v must be a multiple of warp_size");
    constexpr পূর্ণাঙ্ক rows_per_lane = (S_v + warp_size - 1) / warp_size;
    ভিন্ন         s_shard[rows_per_lane];
#pragma unroll
    বাবে (পূর্ণাঙ্ক r = 0; r < rows_per_lane; r++) {
        ধ্রুৱ পূর্ণাঙ্ক i = r * warp_size + lane;
        s_shard[r]  = curr_state[i];
    }

    // snapshot slot mapping: slot 0 = most recent state, slot s = s tokens back.
    // When n_tokens < K only slots 0..n_tokens-1 are written; older slots are caller-owned.

    বাবে (পূর্ণাঙ্ক t = 0; t < n_tokens; t++) {
        ধ্রুৱ ভিন্ন * q_t = q + iq3 * sq3 + t * sq2 + iq1 * sq1;
        ধ্রুৱ ভিন্ন * k_t = k + iq3 * sq3 + t * sq2 + iq1 * sq1;
        ধ্রুৱ ভিন্ন * v_t = v + sequence * sv3 + t * sv2 + h_idx * sv1;

        ধ্রুৱ int64_t gb_offset = sequence * sb3 + t * sb2 + h_idx * sb1;
        ধ্রুৱ ভিন্ন * beta_t = beta + gb_offset;
        ধ্রুৱ ভিন্ন * g_t    = g    + gb_offset * (KDA ? S_v : 1);

        ধ্রুৱ ভিন্ন beta_val = *beta_t;

        যদি constexpr (!KDA) {
            ধ্রুৱ ভিন্ন g_val = sycl::native::exp(*g_t);

            // kv[col] = (S^T @ k)[col] = sum_i S[i][col] * k[i]
            ভিন্ন kv_shard = 0.0f;
#pragma unroll
            বাবে (পূর্ণাঙ্ক r = 0; r < rows_per_lane; r++) {
                ধ্রুৱ পূর্ণাঙ্ক i = r * warp_size + lane;
                kv_shard += s_shard[r] * k_t[i];
            }
            ভিন্ন kv_col = warp_reduce_sum<warp_size>(kv_shard);

            // delta[col] = (v[col] - g * kv[col]) * beta
            ভিন্ন delta_col = (v_t[col] - g_val * kv_col) * beta_val;

            // fused: S[i][col] = g * S[i][col] + k[i] * delta[col]
            // attn[col] = (S^T @ q)[col] = sum_i S[i][col] * q[i]
            ভিন্ন attn_partial = 0.0f;
#pragma unroll
            বাবে (পূর্ণাঙ্ক r = 0; r < rows_per_lane; r++) {
                ধ্রুৱ পূর্ণাঙ্ক i = r * warp_size + lane;
                s_shard[r]  = g_val * s_shard[r] + k_t[i] * delta_col;
                attn_partial += s_shard[r] * q_t[i];
            }

            ভিন্ন attn_col = warp_reduce_sum<warp_size>(attn_partial);

            যদি (lane == 0) {
                attn_data[col] = attn_col * scale;
            }
        } অন্যথা {
            // kv[col] = sum_i g[i] * S[i][col] * k[i]
            ভিন্ন kv_shard = 0.0f;
#pragma unroll
            বাবে (পূর্ণাঙ্ক r = 0; r < rows_per_lane; r++) {
                ধ্রুৱ পূর্ণাঙ্ক i = r * warp_size + lane;
                kv_shard += sycl::native::exp(g_t[i]) * s_shard[r] * k_t[i];
            }

            ভিন্ন kv_col = warp_reduce_sum<warp_size>(kv_shard);

            // delta[col] = (v[col] - kv[col]) * beta
            ভিন্ন delta_col = (v_t[col] - kv_col) * beta_val;

            // fused: S[i][col] = g[i] * S[i][col] + k[i] * delta[col]
            // attn[col] = (S^T @ q)[col] = sum_i S[i][col] * q[i]
            ভিন্ন attn_partial = 0.0f;
#pragma unroll
            বাবে (পূর্ণাঙ্ক r = 0; r < rows_per_lane; r++) {
                ধ্রুৱ পূর্ণাঙ্ক i = r * warp_size + lane;
                s_shard[r]  = sycl::native::exp(g_t[i]) * s_shard[r] + k_t[i] * delta_col;
                attn_partial += s_shard[r] * q_t[i];
            }

            ভিন্ন attn_col = warp_reduce_sum<warp_size>(attn_partial);

            যদি (lane == 0) {
                attn_data[col] = attn_col * scale;
            }
        }

        attn_data += S_v * H;


    // Write state back to global memory
        যদি constexpr (keep_rs_t) {
            ধ্রুৱ পূর্ণাঙ্ক target_slot = (পূর্ণাঙ্ক) n_tokens - 1 - t;
            যদি (target_slot >= 0 && target_slot < K) {
                ভিন্ন * curr_state = state + target_slot * state_slot_stride;
#pragma unroll
                বাবে (পূর্ণাঙ্ক r = 0; r < rows_per_lane; r++) {
                    ধ্রুৱ পূর্ণাঙ্ক i = r * warp_size + lane;
                    curr_state[col * S_v + i] = s_shard[r];
                }
            }
        }
    }

    যদি constexpr (!keep_rs_t) {
#pragma unroll
        বাবে (পূর্ণাঙ্ক r = 0; r < rows_per_lane; r++) {
            ধ্রুৱ পূর্ণাঙ্ক i          = r * warp_size + lane;
            state[col * S_v + i] = s_shard[r];
        }
    }
}

template <তার্কিক KDA, তার্কিক keep_rs_t>
স্থির শূন্য launch_gated_delta_net(ধ্রুৱ ভিন্ন *   q_d,
                                   ধ্রুৱ ভিন্ন *   k_d,
                                   ধ্রুৱ ভিন্ন *   v_d,
                                   ধ্রুৱ ভিন্ন *   g_d,
                                   ধ্রুৱ ভিন্ন *   b_d,
                                   ধ্রুৱ ভিন্ন *   s_d,
                                   ভিন্ন *         dst_d,
                                   ভিন্ন *         state_d,
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
                                   ভিন্ন           scale,
                                   int64_t         state_slot_stride,
                                   পূর্ণাঙ্ক             K,
                                   dpct::queue_ptr stream) {
    //TODO: Add chunked kernel for even faster pre-fill
    ধ্রুৱ পূর্ণাঙ্ক warp_size = ggml_sycl_info().devices[ggml_sycl_get_device()].warp_size;

    ধ্রুৱ পূর্ণাঙ্ক অঙ্ক_warps = 4;
    dpct::dim3 grid_dims(H, n_seqs, (S_v + অঙ্ক_warps - 1) / অঙ্ক_warps);
    dpct::dim3 block_dims(warp_size <= S_v ? warp_size : S_v, অঙ্ক_warps, 1);

    ধ্রুৱ sycl::uint3 neqk1_magic = init_fastdiv_মানসমূহ(neqk1);
    ধ্রুৱ sycl::uint3 rq3_magic   = init_fastdiv_মানসমূহ(rq3);

    ৱিকল্প (S_v) {
        পক্ষ 16:
            {
                constexpr পূর্ণাঙ্ক sv = 16;
                stream->parallel_for(sycl::nd_range<3>(grid_dims * block_dims, block_dims),
                                     [=](sycl::nd_item<3> /*item_ct1*/) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                                         gated_delta_net_sycl<sv, KDA, keep_rs_t>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H, n_tokens,
                                                                       sq1, sq2, sq3, sv1, sv2, sv3, sb1, sb2,
                                                                       sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K);
                                     });
            }
            ভঙ্গ;
        পক্ষ 32:
            {
                constexpr পূর্ণাঙ্ক sv = 32;
                stream->parallel_for(sycl::nd_range<3>(grid_dims * block_dims, block_dims),
                                     [=](sycl::nd_item<3> /*item_ct1*/) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                                         gated_delta_net_sycl<sv, KDA, keep_rs_t>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H, n_tokens,
                                                                       sq1, sq2, sq3, sv1, sv2, sv3, sb1, sb2,
                                                                       sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K);
                                     });
            }
            ভঙ্গ;
        পক্ষ 64: {
            {
                constexpr পূর্ণাঙ্ক sv = 64;
                stream->parallel_for(sycl::nd_range<3>(grid_dims * block_dims, block_dims),
                                        [=](sycl::nd_item<3> /*item_ct1*/) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                                            gated_delta_net_sycl<sv, KDA, keep_rs_t>(
                                                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H, n_tokens, sq1, sq2,
                                                sq3, sv1, sv2, sv3, sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K);
                                        });
            }
            ভঙ্গ;
        }
        পক্ষ 128: {
            {
                constexpr পূর্ণাঙ্ক sv = 128;
                stream->parallel_for(sycl::nd_range<3>(grid_dims * block_dims, block_dims),
                                        [=](sycl::nd_item<3> /*item_ct1*/) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                                            gated_delta_net_sycl<sv, KDA, keep_rs_t>(
                                                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H, n_tokens, sq1, sq2,
                                                sq3, sv1, sv2, sv3, sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K);
                                        });
            }
            ভঙ্গ;
        }
        স্ৱভাৱ:
            GGML_ABORT("fatal error");
            ভঙ্গ;
    }
}

স্থির শূন্য ggml_sycl_op_gated_delta_net_impl(ggml_backend_sycl_context & ctx, ggml_tensor * dst,
                                              ধ্রুৱ ggml_sycl_gated_delta_net_fused_cache * cache) {
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

    ধ্রুৱ int64_t S_v      = nev0;
    ধ্রুৱ int64_t H        = nev1;
    ধ্রুৱ int64_t n_tokens = nev2;
    ধ্রুৱ int64_t n_seqs   = nev3;

    ধ্রুৱ তার্কিক kda = (src_g->ne[0] == S_v);

    GGML_ASSERT(neq1 == nek1);
    ধ্রুৱ int64_t neqk1 = neq1;

    ধ্রুৱ int64_t rq3 = nev3 / neq3;

    ধ্রুৱ ভিন্ন * q_d = (ধ্রুৱ ভিন্ন *) src_q->data;
    ধ্রুৱ ভিন্ন * k_d = (ধ্রুৱ ভিন্ন *) src_k->data;
    ধ্রুৱ ভিন্ন * v_d = (ধ্রুৱ ভিন্ন *) src_v->data;
    ধ্রুৱ ভিন্ন * g_d = (ধ্রুৱ ভিন্ন *) src_g->data;
    ধ্রুৱ ভিন্ন * b_d = (ধ্রুৱ ভিন্ন *) src_beta->data;

    ধ্রুৱ ভিন্ন * s_d   = (ধ্রুৱ ভিন্ন *) src_state->data;
    ভিন্ন *       dst_d = (ভিন্ন *) dst->data;

    GGML_ASSERT(ggml_is_contiguous_rows(src_q));
    GGML_ASSERT(ggml_is_contiguous_rows(src_k));
    GGML_ASSERT(ggml_is_contiguous_rows(src_v));
    GGML_ASSERT(ggml_are_same_stride(src_q, src_k));
    GGML_ASSERT(src_g->ne[0] == 1 || kda);
    GGML_ASSERT(ggml_is_contiguous(src_g));
    GGML_ASSERT(ggml_is_contiguous(src_beta));
    GGML_ASSERT(ggml_is_contiguous(src_state));

    // strides in floats (beta strides used for both g and beta offset computation)
    ধ্রুৱ int64_t sq1 = nbq1 / পরিমাণ(ভিন্ন);
    ধ্রুৱ int64_t sq2 = nbq2 / পরিমাণ(ভিন্ন);
    ধ্রুৱ int64_t sq3 = nbq3 / পরিমাণ(ভিন্ন);
    ধ্রুৱ int64_t sv1 = nbv1 / পরিমাণ(ভিন্ন);
    ধ্রুৱ int64_t sv2 = nbv2 / পরিমাণ(ভিন্ন);
    ধ্রুৱ int64_t sv3 = nbv3 / পরিমাণ(ভিন্ন);
    ধ্রুৱ int64_t sb1 = nbb1 / পরিমাণ(ভিন্ন);
    ধ্রুৱ int64_t sb2 = nbb2 / পরিমাণ(ভিন্ন);
    ধ্রুৱ int64_t sb3 = nbb3 / পরিমাণ(ভিন্ন);

    ধ্রুৱ ভিন্ন scale = 1.0f / sqrtf((ভিন্ন) S_v);

    dpct::queue_ptr stream = ctx.stream();

    // K (snapshot slot count) is an op param; state holds s0 only [S_v, S_v, H, n_seqs].
    ধ্রুৱ পূর্ণাঙ্ক K = ggml_get_op_params_i32(dst, 0);
    ধ্রুৱ তার্কিক keep_rs = K > 1;

    // recurrent state -> dst tail (after attention scores), or the cache when fusing
    ভিন্ন * state_d           = dst_d + S_v * H * n_tokens * n_seqs;
    int64_t state_slot_stride = S_v * S_v * H * n_seqs;
    যদি (cache != nullptr) {
        state_d           = cache->data;
        state_slot_stride = cache->slot_stride;
    }

    যদি (kda) {
        যদি (keep_rs) {
            launch_gated_delta_net<সঁচা, সঁচা>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, stream);
        } অন্যথা {
            launch_gated_delta_net<সঁচা, মিছা>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, stream);
        }
    } অন্যথা {
        যদি (keep_rs) {
            launch_gated_delta_net<মিছা, সঁচা>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, stream);
        } অন্যথা {
            launch_gated_delta_net<মিছা, মিছা>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, stream);
        }
    }
}

শূন্য ggml_sycl_op_gated_delta_net(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_op_gated_delta_net_impl(ctx, dst, nullptr);
}

শূন্য ggml_sycl_gated_delta_net(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/6);
    ggml_sycl_op_gated_delta_net(ctx, dst);
}

শূন্য ggml_sycl_op_gated_delta_net_fused_cache(ggml_backend_sycl_context & ctx, ggml_tensor * dst,
                                              ggml_sycl_gated_delta_net_fused_cache cache) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/6);
    ggml_sycl_op_gated_delta_net_impl(ctx, dst, &cache);
}
