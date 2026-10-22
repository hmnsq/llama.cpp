#समावेश <sycl/sycl.hpp>
#समावेश "dpct/helper.hpp"
#समावेश "common.hpp"
#समावेश "ggml.h"
#समावेश "gated_delta_net.hpp"
#समावेश <cmath>


ढाँचा <पूर्णांक S_v, bool KDA, bool keep_rs_t>
व्योम gated_delta_net_sycl(स्थिर भग्न *     q,
                          स्थिर भग्न *     k,
                          स्थिर भग्न *     v,
                          स्थिर भग्न *     g,
                          स्थिर भग्न *     beta,
                          स्थिर भग्न *     curr_state,
                          भग्न *           dst,
                          भग्न *           state,
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
                          स्थिर sycl::uint3 neqk1_magic,
                          स्थिर sycl::uint3 rq3_magic,
                          भग्न             scale,
                          int64_t           state_slot_stride,
                          पूर्णांक               K) {
    स्वतः           item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    स्थिर uint32_t h_idx    = item_ct1.get_group(2);
    स्थिर uint32_t sequence = item_ct1.get_group(1);
    // each warp owns one column, using warp-level primitives to reduce across rows
    स्थिर पूर्णांक      lane     = item_ct1.get_local_id(2);
    स्थिर पूर्णांक      col      = item_ct1.get_group(0) * item_ct1.get_local_range(1) + item_ct1.get_local_id(1);

    स्थिर uint32_t iq1 = fastmodulo(h_idx, neqk1_magic);
    स्थिर uint32_t iq3 = fastdiv(sequence, rq3_magic);

    भग्न *       attn_तथ्य        = dst;

    // input state holds s0 only [S_v, S_v, H, n_seqs] — seq stride is D = H * S_v * S_v.
    // output state layout (per-slot D * n_seqs) — same per-(seq,head) offset as before.
    स्थिर int64_t state_in_offset      = sequence * H * S_v * S_v + h_idx * S_v * S_v;
    स्थिर int64_t state_out_offset     = (sequence * H + h_idx) * S_v * S_v;
    state += state_out_offset;
    curr_state += state_in_offset + col * S_v;
    attn_तथ्य += (sequence * n_tokens * H + h_idx) * S_v;

    constexpr पूर्णांक warp_आकार = ggml_sycl_get_physical_warp_आकार() < S_v ? ggml_sycl_get_physical_warp_आकार() : S_v;
    static_assert(S_v % warp_आकार == 0, "S_v must be a multiple of warp_size");
    constexpr पूर्णांक rows_per_lane = (S_v + warp_आकार - 1) / warp_आकार;
    भग्न         s_shard[rows_per_lane];
#आशय unroll
    क्रम (पूर्णांक r = 0; r < rows_per_lane; r++) {
        स्थिर पूर्णांक i = r * warp_आकार + lane;
        s_shard[r]  = curr_state[i];
    }

    // snapshot slot mapping: slot 0 = most recent state, slot s = s tokens back.
    // When n_tokens < K only slots 0..n_tokens-1 are written; older slots are caller-owned.

    क्रम (पूर्णांक t = 0; t < n_tokens; t++) {
        स्थिर भग्न * q_t = q + iq3 * sq3 + t * sq2 + iq1 * sq1;
        स्थिर भग्न * k_t = k + iq3 * sq3 + t * sq2 + iq1 * sq1;
        स्थिर भग्न * v_t = v + sequence * sv3 + t * sv2 + h_idx * sv1;

        स्थिर int64_t gb_offset = sequence * sb3 + t * sb2 + h_idx * sb1;
        स्थिर भग्न * beta_t = beta + gb_offset;
        स्थिर भग्न * g_t    = g    + gb_offset * (KDA ? S_v : 1);

        स्थिर भग्न beta_val = *beta_t;

        अगर constexpr (!KDA) {
            स्थिर भग्न g_val = sycl::native::exp(*g_t);

            // kv[col] = (S^T @ k)[col] = sum_i S[i][col] * k[i]
            भग्न kv_shard = 0.0f;
#आशय unroll
            क्रम (पूर्णांक r = 0; r < rows_per_lane; r++) {
                स्थिर पूर्णांक i = r * warp_आकार + lane;
                kv_shard += s_shard[r] * k_t[i];
            }
            भग्न kv_col = warp_reduce_योग<warp_आकार>(kv_shard);

            // delta[col] = (v[col] - g * kv[col]) * beta
            भग्न delta_col = (v_t[col] - g_val * kv_col) * beta_val;

            // fused: S[i][col] = g * S[i][col] + k[i] * delta[col]
            // attn[col] = (S^T @ q)[col] = sum_i S[i][col] * q[i]
            भग्न attn_partial = 0.0f;
#आशय unroll
            क्रम (पूर्णांक r = 0; r < rows_per_lane; r++) {
                स्थिर पूर्णांक i = r * warp_आकार + lane;
                s_shard[r]  = g_val * s_shard[r] + k_t[i] * delta_col;
                attn_partial += s_shard[r] * q_t[i];
            }

            भग्न attn_col = warp_reduce_योग<warp_आकार>(attn_partial);

            अगर (lane == 0) {
                attn_तथ्य[col] = attn_col * scale;
            }
        } अन्यथा {
            // kv[col] = sum_i g[i] * S[i][col] * k[i]
            भग्न kv_shard = 0.0f;
#आशय unroll
            क्रम (पूर्णांक r = 0; r < rows_per_lane; r++) {
                स्थिर पूर्णांक i = r * warp_आकार + lane;
                kv_shard += sycl::native::exp(g_t[i]) * s_shard[r] * k_t[i];
            }

            भग्न kv_col = warp_reduce_योग<warp_आकार>(kv_shard);

            // delta[col] = (v[col] - kv[col]) * beta
            भग्न delta_col = (v_t[col] - kv_col) * beta_val;

            // fused: S[i][col] = g[i] * S[i][col] + k[i] * delta[col]
            // attn[col] = (S^T @ q)[col] = sum_i S[i][col] * q[i]
            भग्न attn_partial = 0.0f;
#आशय unroll
            क्रम (पूर्णांक r = 0; r < rows_per_lane; r++) {
                स्थिर पूर्णांक i = r * warp_आकार + lane;
                s_shard[r]  = sycl::native::exp(g_t[i]) * s_shard[r] + k_t[i] * delta_col;
                attn_partial += s_shard[r] * q_t[i];
            }

            भग्न attn_col = warp_reduce_योग<warp_आकार>(attn_partial);

            अगर (lane == 0) {
                attn_तथ्य[col] = attn_col * scale;
            }
        }

        attn_तथ्य += S_v * H;


    // Write state back to global memory
        अगर constexpr (keep_rs_t) {
            स्थिर पूर्णांक target_slot = (पूर्णांक) n_tokens - 1 - t;
            अगर (target_slot >= 0 && target_slot < K) {
                भग्न * curr_state = state + target_slot * state_slot_stride;
#आशय unroll
                क्रम (पूर्णांक r = 0; r < rows_per_lane; r++) {
                    स्थिर पूर्णांक i = r * warp_आकार + lane;
                    curr_state[col * S_v + i] = s_shard[r];
                }
            }
        }
    }

    अगर constexpr (!keep_rs_t) {
#आशय unroll
        क्रम (पूर्णांक r = 0; r < rows_per_lane; r++) {
            स्थिर पूर्णांक i          = r * warp_आकार + lane;
            state[col * S_v + i] = s_shard[r];
        }
    }
}

ढाँचा <bool KDA, bool keep_rs_t>
अटल व्योम launch_gated_delta_net(स्थिर भग्न *   q_d,
                                   स्थिर भग्न *   k_d,
                                   स्थिर भग्न *   v_d,
                                   स्थिर भग्न *   g_d,
                                   स्थिर भग्न *   b_d,
                                   स्थिर भग्न *   s_d,
                                   भग्न *         dst_d,
                                   भग्न *         state_d,
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
                                   भग्न           scale,
                                   int64_t         state_slot_stride,
                                   पूर्णांक             K,
                                   dpct::queue_ptr stream) {
    //TODO: Add chunked kernel for even faster pre-fill
    स्थिर पूर्णांक warp_आकार = ggml_sycl_info().devices[ggml_sycl_get_device()].warp_आकार;

    स्थिर पूर्णांक अंक_warps = 4;
    dpct::dim3 grid_dims(H, n_seqs, (S_v + अंक_warps - 1) / अंक_warps);
    dpct::dim3 block_dims(warp_आकार <= S_v ? warp_आकार : S_v, अंक_warps, 1);

    स्थिर sycl::uint3 neqk1_magic = init_fastdiv_मूल्यावली(neqk1);
    स्थिर sycl::uint3 rq3_magic   = init_fastdiv_मूल्यावली(rq3);

    चयन (S_v) {
        हाल 16:
            {
                constexpr पूर्णांक sv = 16;
                stream->parallel_for(sycl::nd_range<3>(grid_dims * block_dims, block_dims),
                                     [=](sycl::nd_item<3> /*item_ct1*/) [[sycl::reqd_sub_group_आकार(WARP_आकार)]] {
                                         gated_delta_net_sycl<sv, KDA, keep_rs_t>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H, n_tokens,
                                                                       sq1, sq2, sq3, sv1, sv2, sv3, sb1, sb2,
                                                                       sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K);
                                     });
            }
            अवरोध;
        हाल 32:
            {
                constexpr पूर्णांक sv = 32;
                stream->parallel_for(sycl::nd_range<3>(grid_dims * block_dims, block_dims),
                                     [=](sycl::nd_item<3> /*item_ct1*/) [[sycl::reqd_sub_group_आकार(WARP_आकार)]] {
                                         gated_delta_net_sycl<sv, KDA, keep_rs_t>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H, n_tokens,
                                                                       sq1, sq2, sq3, sv1, sv2, sv3, sb1, sb2,
                                                                       sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K);
                                     });
            }
            अवरोध;
        हाल 64: {
            {
                constexpr पूर्णांक sv = 64;
                stream->parallel_for(sycl::nd_range<3>(grid_dims * block_dims, block_dims),
                                        [=](sycl::nd_item<3> /*item_ct1*/) [[sycl::reqd_sub_group_आकार(WARP_आकार)]] {
                                            gated_delta_net_sycl<sv, KDA, keep_rs_t>(
                                                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H, n_tokens, sq1, sq2,
                                                sq3, sv1, sv2, sv3, sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K);
                                        });
            }
            अवरोध;
        }
        हाल 128: {
            {
                constexpr पूर्णांक sv = 128;
                stream->parallel_for(sycl::nd_range<3>(grid_dims * block_dims, block_dims),
                                        [=](sycl::nd_item<3> /*item_ct1*/) [[sycl::reqd_sub_group_आकार(WARP_आकार)]] {
                                            gated_delta_net_sycl<sv, KDA, keep_rs_t>(
                                                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H, n_tokens, sq1, sq2,
                                                sq3, sv1, sv2, sv3, sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K);
                                        });
            }
            अवरोध;
        }
        शेष:
            GGML_ABORT("fatal error");
            अवरोध;
    }
}

अटल व्योम ggml_sycl_op_gated_delta_net_impl(ggml_backend_sycl_context & ctx, ggml_tensor * dst,
                                              स्थिर ggml_sycl_gated_delta_net_fused_cache * cache) {
    ggml_tensor * src_q     = dst->src[0];
    ggml_tensor * src_k     = dst->src[1];
    ggml_tensor * src_v     = dst->src[2];
    ggml_tensor * src_g     = dst->src[3];
    ggml_tensor * src_beta  = dst->src[4];
    ggml_tensor * src_state = dst->src[5];

    GGML_TENSOR_LOCALS(int64_t, neq, src_q, ne);
    GGML_TENSOR_LOCALS(माप_प्रकार , nbq, src_q, nb);
    GGML_TENSOR_LOCALS(int64_t, nek, src_k, ne);
    GGML_TENSOR_LOCALS(माप_प्रकार , nbk, src_k, nb);
    GGML_TENSOR_LOCALS(int64_t, nev, src_v, ne);
    GGML_TENSOR_LOCALS(माप_प्रकार,  nbv, src_v, nb);
    GGML_TENSOR_LOCALS(माप_प्रकार,  nbb, src_beta, nb);

    स्थिर int64_t S_v      = nev0;
    स्थिर int64_t H        = nev1;
    स्थिर int64_t n_tokens = nev2;
    स्थिर int64_t n_seqs   = nev3;

    स्थिर bool kda = (src_g->ne[0] == S_v);

    GGML_ASSERT(neq1 == nek1);
    स्थिर int64_t neqk1 = neq1;

    स्थिर int64_t rq3 = nev3 / neq3;

    स्थिर भग्न * q_d = (स्थिर भग्न *) src_q->तथ्य;
    स्थिर भग्न * k_d = (स्थिर भग्न *) src_k->तथ्य;
    स्थिर भग्न * v_d = (स्थिर भग्न *) src_v->तथ्य;
    स्थिर भग्न * g_d = (स्थिर भग्न *) src_g->तथ्य;
    स्थिर भग्न * b_d = (स्थिर भग्न *) src_beta->तथ्य;

    स्थिर भग्न * s_d   = (स्थिर भग्न *) src_state->तथ्य;
    भग्न *       dst_d = (भग्न *) dst->तथ्य;

    GGML_ASSERT(ggml_is_contiguous_rows(src_q));
    GGML_ASSERT(ggml_is_contiguous_rows(src_k));
    GGML_ASSERT(ggml_is_contiguous_rows(src_v));
    GGML_ASSERT(ggml_are_same_stride(src_q, src_k));
    GGML_ASSERT(src_g->ne[0] == 1 || kda);
    GGML_ASSERT(ggml_is_contiguous(src_g));
    GGML_ASSERT(ggml_is_contiguous(src_beta));
    GGML_ASSERT(ggml_is_contiguous(src_state));

    // strides in floats (beta strides used for both g and beta offset computation)
    स्थिर int64_t sq1 = nbq1 / माप(भग्न);
    स्थिर int64_t sq2 = nbq2 / माप(भग्न);
    स्थिर int64_t sq3 = nbq3 / माप(भग्न);
    स्थिर int64_t sv1 = nbv1 / माप(भग्न);
    स्थिर int64_t sv2 = nbv2 / माप(भग्न);
    स्थिर int64_t sv3 = nbv3 / माप(भग्न);
    स्थिर int64_t sb1 = nbb1 / माप(भग्न);
    स्थिर int64_t sb2 = nbb2 / माप(भग्न);
    स्थिर int64_t sb3 = nbb3 / माप(भग्न);

    स्थिर भग्न scale = 1.0f / sqrtf((भग्न) S_v);

    dpct::queue_ptr stream = ctx.stream();

    // K (snapshot slot count) is an op param; state holds s0 only [S_v, S_v, H, n_seqs].
    स्थिर पूर्णांक K = ggml_get_op_params_i32(dst, 0);
    स्थिर bool keep_rs = K > 1;

    // recurrent state -> dst tail (after attention scores), or the cache when fusing
    भग्न * state_d           = dst_d + S_v * H * n_tokens * n_seqs;
    int64_t state_slot_stride = S_v * S_v * H * n_seqs;
    अगर (cache != nullptr) {
        state_d           = cache->तथ्य;
        state_slot_stride = cache->slot_stride;
    }

    अगर (kda) {
        अगर (keep_rs) {
            launch_gated_delta_net<true, true>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, stream);
        } अन्यथा {
            launch_gated_delta_net<true, false>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, stream);
        }
    } अन्यथा {
        अगर (keep_rs) {
            launch_gated_delta_net<false, true>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, stream);
        } अन्यथा {
            launch_gated_delta_net<false, false>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, stream);
        }
    }
}

व्योम ggml_sycl_op_gated_delta_net(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_op_gated_delta_net_impl(ctx, dst, nullptr);
}

व्योम ggml_sycl_gated_delta_net(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/6);
    ggml_sycl_op_gated_delta_net(ctx, dst);
}

व्योम ggml_sycl_op_gated_delta_net_fused_cache(ggml_backend_sycl_context & ctx, ggml_tensor * dst,
                                              ggml_sycl_gated_delta_net_fused_cache cache) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/6);
    ggml_sycl_op_gated_delta_net_impl(ctx, dst, &cache);
}
