#অন্তর্ভাব <sycl/sycl.hpp>
#অন্তর্ভাব "dpct/helper.hpp"
#অন্তর্ভাব "common.hpp"
#অন্তর্ভাব "ggml.h"
#অন্তর্ভাব "gated_delta_net.hpp"
#অন্তর্ভাব <cmath>


template <পূর্ণসংখ্যা S_v, যৌক্তিক KDA, যৌক্তিক keep_rs_t>
শূন্য gated_delta_net_sycl(ধ্রুবক দশমিক *     q,
                          ধ্রুবক দশমিক *     k,
                          ধ্রুবক দশমিক *     v,
                          ধ্রুবক দশমিক *     g,
                          ধ্রুবক দশমিক *     beta,
                          ধ্রুবক দশমিক *     curr_state,
                          দশমিক *           dst,
                          দশমিক *           state,
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
                          ধ্রুবক sycl::uint3 neqk1_magic,
                          ধ্রুবক sycl::uint3 rq3_magic,
                          দশমিক             scale,
                          int64_t           state_slot_stride,
                          পূর্ণসংখ্যা               K) {
    auto           item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    ধ্রুবক uint32_t h_idx    = item_ct1.get_group(2);
    ধ্রুবক uint32_t sequence = item_ct1.get_group(1);
    // each warp owns one column, using warp-level primitives to reduce across rows
    ধ্রুবক পূর্ণসংখ্যা      lane     = item_ct1.get_local_id(2);
    ধ্রুবক পূর্ণসংখ্যা      col      = item_ct1.get_group(0) * item_ct1.get_local_range(1) + item_ct1.get_local_id(1);

    ধ্রুবক uint32_t iq1 = fastmodulo(h_idx, neqk1_magic);
    ধ্রুবক uint32_t iq3 = fastdiv(sequence, rq3_magic);

    দশমিক *       attn_তথ্য        = dst;

    // input state holds s0 only [S_v, S_v, H, n_seqs] — seq stride is D = H * S_v * S_v.
    // output state layout (per-slot D * n_seqs) — same per-(seq,head) offset as before.
    ধ্রুবক int64_t state_in_offset      = sequence * H * S_v * S_v + h_idx * S_v * S_v;
    ধ্রুবক int64_t state_out_offset     = (sequence * H + h_idx) * S_v * S_v;
    state += state_out_offset;
    curr_state += state_in_offset + col * S_v;
    attn_তথ্য += (sequence * n_tokens * H + h_idx) * S_v;

    constexpr পূর্ণসংখ্যা warp_আকার = ggml_sycl_get_physical_warp_আকার() < S_v ? ggml_sycl_get_physical_warp_আকার() : S_v;
    static_assert(S_v % warp_আকার == 0, "S_v must be a multiple of warp_size");
    constexpr পূর্ণসংখ্যা rows_per_lane = (S_v + warp_আকার - 1) / warp_আকার;
    দশমিক         s_shard[rows_per_lane];
#pragma unroll
    জন্য (পূর্ণসংখ্যা r = 0; r < rows_per_lane; r++) {
        ধ্রুবক পূর্ণসংখ্যা i = r * warp_আকার + lane;
        s_shard[r]  = curr_state[i];
    }

    // snapshot slot mapping: slot 0 = most recent state, slot s = s tokens back.
    // When n_tokens < K only slots 0..n_tokens-1 are written; older slots are caller-owned.

    জন্য (পূর্ণসংখ্যা t = 0; t < n_tokens; t++) {
        ধ্রুবক দশমিক * q_t = q + iq3 * sq3 + t * sq2 + iq1 * sq1;
        ধ্রুবক দশমিক * k_t = k + iq3 * sq3 + t * sq2 + iq1 * sq1;
        ধ্রুবক দশমিক * v_t = v + sequence * sv3 + t * sv2 + h_idx * sv1;

        ধ্রুবক int64_t gb_offset = sequence * sb3 + t * sb2 + h_idx * sb1;
        ধ্রুবক দশমিক * beta_t = beta + gb_offset;
        ধ্রুবক দশমিক * g_t    = g    + gb_offset * (KDA ? S_v : 1);

        ধ্রুবক দশমিক beta_val = *beta_t;

        যদি constexpr (!KDA) {
            ধ্রুবক দশমিক g_val = sycl::native::exp(*g_t);

            // kv[col] = (S^T @ k)[col] = sum_i S[i][col] * k[i]
            দশমিক kv_shard = 0.0f;
#pragma unroll
            জন্য (পূর্ণসংখ্যা r = 0; r < rows_per_lane; r++) {
                ধ্রুবক পূর্ণসংখ্যা i = r * warp_আকার + lane;
                kv_shard += s_shard[r] * k_t[i];
            }
            দশমিক kv_col = warp_reduce_যোগ<warp_আকার>(kv_shard);

            // delta[col] = (v[col] - g * kv[col]) * beta
            দশমিক delta_col = (v_t[col] - g_val * kv_col) * beta_val;

            // fused: S[i][col] = g * S[i][col] + k[i] * delta[col]
            // attn[col] = (S^T @ q)[col] = sum_i S[i][col] * q[i]
            দশমিক attn_partial = 0.0f;
#pragma unroll
            জন্য (পূর্ণসংখ্যা r = 0; r < rows_per_lane; r++) {
                ধ্রুবক পূর্ণসংখ্যা i = r * warp_আকার + lane;
                s_shard[r]  = g_val * s_shard[r] + k_t[i] * delta_col;
                attn_partial += s_shard[r] * q_t[i];
            }

            দশমিক attn_col = warp_reduce_যোগ<warp_আকার>(attn_partial);

            যদি (lane == 0) {
                attn_তথ্য[col] = attn_col * scale;
            }
        } অন্যথা {
            // kv[col] = sum_i g[i] * S[i][col] * k[i]
            দশমিক kv_shard = 0.0f;
#pragma unroll
            জন্য (পূর্ণসংখ্যা r = 0; r < rows_per_lane; r++) {
                ধ্রুবক পূর্ণসংখ্যা i = r * warp_আকার + lane;
                kv_shard += sycl::native::exp(g_t[i]) * s_shard[r] * k_t[i];
            }

            দশমিক kv_col = warp_reduce_যোগ<warp_আকার>(kv_shard);

            // delta[col] = (v[col] - kv[col]) * beta
            দশমিক delta_col = (v_t[col] - kv_col) * beta_val;

            // fused: S[i][col] = g[i] * S[i][col] + k[i] * delta[col]
            // attn[col] = (S^T @ q)[col] = sum_i S[i][col] * q[i]
            দশমিক attn_partial = 0.0f;
#pragma unroll
            জন্য (পূর্ণসংখ্যা r = 0; r < rows_per_lane; r++) {
                ধ্রুবক পূর্ণসংখ্যা i = r * warp_আকার + lane;
                s_shard[r]  = sycl::native::exp(g_t[i]) * s_shard[r] + k_t[i] * delta_col;
                attn_partial += s_shard[r] * q_t[i];
            }

            দশমিক attn_col = warp_reduce_যোগ<warp_আকার>(attn_partial);

            যদি (lane == 0) {
                attn_তথ্য[col] = attn_col * scale;
            }
        }

        attn_তথ্য += S_v * H;


    // Write state back to global memory
        যদি constexpr (keep_rs_t) {
            ধ্রুবক পূর্ণসংখ্যা target_slot = (পূর্ণসংখ্যা) n_tokens - 1 - t;
            যদি (target_slot >= 0 && target_slot < K) {
                দশমিক * curr_state = state + target_slot * state_slot_stride;
#pragma unroll
                জন্য (পূর্ণসংখ্যা r = 0; r < rows_per_lane; r++) {
                    ধ্রুবক পূর্ণসংখ্যা i = r * warp_আকার + lane;
                    curr_state[col * S_v + i] = s_shard[r];
                }
            }
        }
    }

    যদি constexpr (!keep_rs_t) {
#pragma unroll
        জন্য (পূর্ণসংখ্যা r = 0; r < rows_per_lane; r++) {
            ধ্রুবক পূর্ণসংখ্যা i          = r * warp_আকার + lane;
            state[col * S_v + i] = s_shard[r];
        }
    }
}

template <যৌক্তিক KDA, যৌক্তিক keep_rs_t>
স্থির শূন্য launch_gated_delta_net(ধ্রুবক দশমিক *   q_d,
                                   ধ্রুবক দশমিক *   k_d,
                                   ধ্রুবক দশমিক *   v_d,
                                   ধ্রুবক দশমিক *   g_d,
                                   ধ্রুবক দশমিক *   b_d,
                                   ধ্রুবক দশমিক *   s_d,
                                   দশমিক *         dst_d,
                                   দশমিক *         state_d,
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
                                   দশমিক           scale,
                                   int64_t         state_slot_stride,
                                   পূর্ণসংখ্যা             K,
                                   dpct::queue_ptr stream) {
    //TODO: Add chunked kernel for even faster pre-fill
    ধ্রুবক পূর্ণসংখ্যা warp_আকার = ggml_sycl_info().devices[ggml_sycl_get_device()].warp_আকার;

    ধ্রুবক পূর্ণসংখ্যা অঙ্ক_warps = 4;
    dpct::dim3 grid_dims(H, n_seqs, (S_v + অঙ্ক_warps - 1) / অঙ্ক_warps);
    dpct::dim3 block_dims(warp_আকার <= S_v ? warp_আকার : S_v, অঙ্ক_warps, 1);

    ধ্রুবক sycl::uint3 neqk1_magic = init_fastdiv_মানসমূহ(neqk1);
    ধ্রুবক sycl::uint3 rq3_magic   = init_fastdiv_মানসমূহ(rq3);

    নির্বাচন (S_v) {
        ক্ষেত্র 16:
            {
                constexpr পূর্ণসংখ্যা sv = 16;
                stream->parallel_for(sycl::nd_range<3>(grid_dims * block_dims, block_dims),
                                     [=](sycl::nd_item<3> /*item_ct1*/) [[sycl::reqd_sub_group_আকার(WARP_আকার)]] {
                                         gated_delta_net_sycl<sv, KDA, keep_rs_t>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H, n_tokens,
                                                                       sq1, sq2, sq3, sv1, sv2, sv3, sb1, sb2,
                                                                       sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K);
                                     });
            }
            বিরতি;
        ক্ষেত্র 32:
            {
                constexpr পূর্ণসংখ্যা sv = 32;
                stream->parallel_for(sycl::nd_range<3>(grid_dims * block_dims, block_dims),
                                     [=](sycl::nd_item<3> /*item_ct1*/) [[sycl::reqd_sub_group_আকার(WARP_আকার)]] {
                                         gated_delta_net_sycl<sv, KDA, keep_rs_t>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H, n_tokens,
                                                                       sq1, sq2, sq3, sv1, sv2, sv3, sb1, sb2,
                                                                       sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K);
                                     });
            }
            বিরতি;
        ক্ষেত্র 64: {
            {
                constexpr পূর্ণসংখ্যা sv = 64;
                stream->parallel_for(sycl::nd_range<3>(grid_dims * block_dims, block_dims),
                                        [=](sycl::nd_item<3> /*item_ct1*/) [[sycl::reqd_sub_group_আকার(WARP_আকার)]] {
                                            gated_delta_net_sycl<sv, KDA, keep_rs_t>(
                                                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H, n_tokens, sq1, sq2,
                                                sq3, sv1, sv2, sv3, sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K);
                                        });
            }
            বিরতি;
        }
        ক্ষেত্র 128: {
            {
                constexpr পূর্ণসংখ্যা sv = 128;
                stream->parallel_for(sycl::nd_range<3>(grid_dims * block_dims, block_dims),
                                        [=](sycl::nd_item<3> /*item_ct1*/) [[sycl::reqd_sub_group_আকার(WARP_আকার)]] {
                                            gated_delta_net_sycl<sv, KDA, keep_rs_t>(
                                                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H, n_tokens, sq1, sq2,
                                                sq3, sv1, sv2, sv3, sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K);
                                        });
            }
            বিরতি;
        }
        পূর্বনির্ধারিত:
            GGML_ABORT("fatal error");
            বিরতি;
    }
}

স্থির শূন্য ggml_sycl_op_gated_delta_net_impl(ggml_backend_sycl_context & ctx, ggml_tensor * dst,
                                              ধ্রুবক ggml_sycl_gated_delta_net_fused_cache * cache) {
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

    ধ্রুবক int64_t S_v      = nev0;
    ধ্রুবক int64_t H        = nev1;
    ধ্রুবক int64_t n_tokens = nev2;
    ধ্রুবক int64_t n_seqs   = nev3;

    ধ্রুবক যৌক্তিক kda = (src_g->ne[0] == S_v);

    GGML_ASSERT(neq1 == nek1);
    ধ্রুবক int64_t neqk1 = neq1;

    ধ্রুবক int64_t rq3 = nev3 / neq3;

    ধ্রুবক দশমিক * q_d = (ধ্রুবক দশমিক *) src_q->তথ্য;
    ধ্রুবক দশমিক * k_d = (ধ্রুবক দশমিক *) src_k->তথ্য;
    ধ্রুবক দশমিক * v_d = (ধ্রুবক দশমিক *) src_v->তথ্য;
    ধ্রুবক দশমিক * g_d = (ধ্রুবক দশমিক *) src_g->তথ্য;
    ধ্রুবক দশমিক * b_d = (ধ্রুবক দশমিক *) src_beta->তথ্য;

    ধ্রুবক দশমিক * s_d   = (ধ্রুবক দশমিক *) src_state->তথ্য;
    দশমিক *       dst_d = (দশমিক *) dst->তথ্য;

    GGML_ASSERT(ggml_is_contiguous_rows(src_q));
    GGML_ASSERT(ggml_is_contiguous_rows(src_k));
    GGML_ASSERT(ggml_is_contiguous_rows(src_v));
    GGML_ASSERT(ggml_are_same_stride(src_q, src_k));
    GGML_ASSERT(src_g->ne[0] == 1 || kda);
    GGML_ASSERT(ggml_is_contiguous(src_g));
    GGML_ASSERT(ggml_is_contiguous(src_beta));
    GGML_ASSERT(ggml_is_contiguous(src_state));

    // strides in floats (beta strides used for both g and beta offset computation)
    ধ্রুবক int64_t sq1 = nbq1 / আকার(দশমিক);
    ধ্রুবক int64_t sq2 = nbq2 / আকার(দশমিক);
    ধ্রুবক int64_t sq3 = nbq3 / আকার(দশমিক);
    ধ্রুবক int64_t sv1 = nbv1 / আকার(দশমিক);
    ধ্রুবক int64_t sv2 = nbv2 / আকার(দশমিক);
    ধ্রুবক int64_t sv3 = nbv3 / আকার(দশমিক);
    ধ্রুবক int64_t sb1 = nbb1 / আকার(দশমিক);
    ধ্রুবক int64_t sb2 = nbb2 / আকার(দশমিক);
    ধ্রুবক int64_t sb3 = nbb3 / আকার(দশমিক);

    ধ্রুবক দশমিক scale = 1.0f / sqrtf((দশমিক) S_v);

    dpct::queue_ptr stream = ctx.stream();

    // K (snapshot slot count) is an op param; state holds s0 only [S_v, S_v, H, n_seqs].
    ধ্রুবক পূর্ণসংখ্যা K = ggml_get_op_params_i32(dst, 0);
    ধ্রুবক যৌক্তিক keep_rs = K > 1;

    // recurrent state -> dst tail (after attention scores), or the cache when fusing
    দশমিক * state_d           = dst_d + S_v * H * n_tokens * n_seqs;
    int64_t state_slot_stride = S_v * S_v * H * n_seqs;
    যদি (cache != nullptr) {
        state_d           = cache->তথ্য;
        state_slot_stride = cache->slot_stride;
    }

    যদি (kda) {
        যদি (keep_rs) {
            launch_gated_delta_net<সত্য, সত্য>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, stream);
        } অন্যথা {
            launch_gated_delta_net<সত্য, মিথ্যা>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, stream);
        }
    } অন্যথা {
        যদি (keep_rs) {
            launch_gated_delta_net<মিথ্যা, সত্য>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, stream);
        } অন্যথা {
            launch_gated_delta_net<মিথ্যা, মিথ্যা>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
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
