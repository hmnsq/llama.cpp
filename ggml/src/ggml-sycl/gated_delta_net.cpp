#अन्तर्भाव <sycl/sycl.hpp>
#अन्तर्भाव "dpct/helper.hpp"
#अन्तर्भाव "common.hpp"
#अन्तर्भाव "ggml.h"
#अन्तर्भाव "gated_delta_net.hpp"
#अन्तर्भाव <cmath>


ढाँचा <पूर्णांक S_v, तार्किक KDA, तार्किक keep_rs_t>
शून्य gated_delta_net_sycl(स्थिरांक दशांश *     q,
                          स्थिरांक दशांश *     k,
                          स्थिरांक दशांश *     v,
                          स्थिरांक दशांश *     g,
                          स्थिरांक दशांश *     beta,
                          स्थिरांक दशांश *     curr_state,
                          दशांश *           dst,
                          दशांश *           state,
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
                          स्थिरांक sycl::uint3 neqk1_magic,
                          स्थिरांक sycl::uint3 rq3_magic,
                          दशांश             scale,
                          int64_t           state_slot_stride,
                          पूर्णांक               K) {
    स्वतः           item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    स्थिरांक uint32_t h_idx    = item_ct1.get_group(2);
    स्थिरांक uint32_t sequence = item_ct1.get_group(1);
    // each warp owns one column, using warp-level primitives to reduce across rows
    स्थिरांक पूर्णांक      lane     = item_ct1.get_local_id(2);
    स्थिरांक पूर्णांक      col      = item_ct1.get_group(0) * item_ct1.get_local_range(1) + item_ct1.get_local_id(1);

    स्थिरांक uint32_t iq1 = fastmodulo(h_idx, neqk1_magic);
    स्थिरांक uint32_t iq3 = fastdiv(sequence, rq3_magic);

    दशांश *       attn_data        = dst;

    // input state holds s0 only [S_v, S_v, H, n_seqs] — seq stride is D = H * S_v * S_v.
    // output state layout (per-slot D * n_seqs) — same per-(seq,head) offset as before.
    स्थिरांक int64_t state_in_offset      = sequence * H * S_v * S_v + h_idx * S_v * S_v;
    स्थिरांक int64_t state_out_offset     = (sequence * H + h_idx) * S_v * S_v;
    state += state_out_offset;
    curr_state += state_in_offset + col * S_v;
    attn_data += (sequence * n_tokens * H + h_idx) * S_v;

    constexpr पूर्णांक warp_size = ggml_sycl_get_physical_warp_size() < S_v ? ggml_sycl_get_physical_warp_size() : S_v;
    static_assert(S_v % warp_size == 0, "S_v must be a multiple of warp_size");
    constexpr पूर्णांक rows_per_lane = (S_v + warp_size - 1) / warp_size;
    दशांश         s_shard[rows_per_lane];
#pragma unroll
    साठी (पूर्णांक r = 0; r < rows_per_lane; r++) {
        स्थिरांक पूर्णांक i = r * warp_size + lane;
        s_shard[r]  = curr_state[i];
    }

    // snapshot slot mapping: slot 0 = most recent state, slot s = s tokens back.
    // When n_tokens < K only slots 0..n_tokens-1 are written; older slots are caller-owned.

    साठी (पूर्णांक t = 0; t < n_tokens; t++) {
        स्थिरांक दशांश * q_t = q + iq3 * sq3 + t * sq2 + iq1 * sq1;
        स्थिरांक दशांश * k_t = k + iq3 * sq3 + t * sq2 + iq1 * sq1;
        स्थिरांक दशांश * v_t = v + sequence * sv3 + t * sv2 + h_idx * sv1;

        स्थिरांक int64_t gb_offset = sequence * sb3 + t * sb2 + h_idx * sb1;
        स्थिरांक दशांश * beta_t = beta + gb_offset;
        स्थिरांक दशांश * g_t    = g    + gb_offset * (KDA ? S_v : 1);

        स्थिरांक दशांश beta_val = *beta_t;

        जर constexpr (!KDA) {
            स्थिरांक दशांश g_val = sycl::native::exp(*g_t);

            // kv[col] = (S^T @ k)[col] = sum_i S[i][col] * k[i]
            दशांश kv_shard = 0.0f;
#pragma unroll
            साठी (पूर्णांक r = 0; r < rows_per_lane; r++) {
                स्थिरांक पूर्णांक i = r * warp_size + lane;
                kv_shard += s_shard[r] * k_t[i];
            }
            दशांश kv_col = warp_reduce_sum<warp_size>(kv_shard);

            // delta[col] = (v[col] - g * kv[col]) * beta
            दशांश delta_col = (v_t[col] - g_val * kv_col) * beta_val;

            // fused: S[i][col] = g * S[i][col] + k[i] * delta[col]
            // attn[col] = (S^T @ q)[col] = sum_i S[i][col] * q[i]
            दशांश attn_partial = 0.0f;
#pragma unroll
            साठी (पूर्णांक r = 0; r < rows_per_lane; r++) {
                स्थिरांक पूर्णांक i = r * warp_size + lane;
                s_shard[r]  = g_val * s_shard[r] + k_t[i] * delta_col;
                attn_partial += s_shard[r] * q_t[i];
            }

            दशांश attn_col = warp_reduce_sum<warp_size>(attn_partial);

            जर (lane == 0) {
                attn_data[col] = attn_col * scale;
            }
        } अन्यथा {
            // kv[col] = sum_i g[i] * S[i][col] * k[i]
            दशांश kv_shard = 0.0f;
#pragma unroll
            साठी (पूर्णांक r = 0; r < rows_per_lane; r++) {
                स्थिरांक पूर्णांक i = r * warp_size + lane;
                kv_shard += sycl::native::exp(g_t[i]) * s_shard[r] * k_t[i];
            }

            दशांश kv_col = warp_reduce_sum<warp_size>(kv_shard);

            // delta[col] = (v[col] - kv[col]) * beta
            दशांश delta_col = (v_t[col] - kv_col) * beta_val;

            // fused: S[i][col] = g[i] * S[i][col] + k[i] * delta[col]
            // attn[col] = (S^T @ q)[col] = sum_i S[i][col] * q[i]
            दशांश attn_partial = 0.0f;
#pragma unroll
            साठी (पूर्णांक r = 0; r < rows_per_lane; r++) {
                स्थिरांक पूर्णांक i = r * warp_size + lane;
                s_shard[r]  = sycl::native::exp(g_t[i]) * s_shard[r] + k_t[i] * delta_col;
                attn_partial += s_shard[r] * q_t[i];
            }

            दशांश attn_col = warp_reduce_sum<warp_size>(attn_partial);

            जर (lane == 0) {
                attn_data[col] = attn_col * scale;
            }
        }

        attn_data += S_v * H;


    // Write state back to global memory
        जर constexpr (keep_rs_t) {
            स्थिरांक पूर्णांक target_slot = (पूर्णांक) n_tokens - 1 - t;
            जर (target_slot >= 0 && target_slot < K) {
                दशांश * curr_state = state + target_slot * state_slot_stride;
#pragma unroll
                साठी (पूर्णांक r = 0; r < rows_per_lane; r++) {
                    स्थिरांक पूर्णांक i = r * warp_size + lane;
                    curr_state[col * S_v + i] = s_shard[r];
                }
            }
        }
    }

    जर constexpr (!keep_rs_t) {
#pragma unroll
        साठी (पूर्णांक r = 0; r < rows_per_lane; r++) {
            स्थिरांक पूर्णांक i          = r * warp_size + lane;
            state[col * S_v + i] = s_shard[r];
        }
    }
}

ढाँचा <तार्किक KDA, तार्किक keep_rs_t>
स्थिर शून्य launch_gated_delta_net(स्थिरांक दशांश *   q_d,
                                   स्थिरांक दशांश *   k_d,
                                   स्थिरांक दशांश *   v_d,
                                   स्थिरांक दशांश *   g_d,
                                   स्थिरांक दशांश *   b_d,
                                   स्थिरांक दशांश *   s_d,
                                   दशांश *         dst_d,
                                   दशांश *         state_d,
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
                                   दशांश           scale,
                                   int64_t         state_slot_stride,
                                   पूर्णांक             K,
                                   dpct::queue_ptr stream) {
    //TODO: Add chunked kernel for even faster pre-fill
    स्थिरांक पूर्णांक warp_size = ggml_sycl_info().devices[ggml_sycl_get_device()].warp_size;

    स्थिरांक पूर्णांक अंक_warps = 4;
    dpct::dim3 grid_dims(H, n_seqs, (S_v + अंक_warps - 1) / अंक_warps);
    dpct::dim3 block_dims(warp_size <= S_v ? warp_size : S_v, अंक_warps, 1);

    स्थिरांक sycl::uint3 neqk1_magic = init_fastdiv_मूल्ये(neqk1);
    स्थिरांक sycl::uint3 rq3_magic   = init_fastdiv_मूल्ये(rq3);

    निवड (S_v) {
        स्थिती 16:
            {
                constexpr पूर्णांक sv = 16;
                stream->parallel_for(sycl::nd_range<3>(grid_dims * block_dims, block_dims),
                                     [=](sycl::nd_item<3> /*item_ct1*/) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                                         gated_delta_net_sycl<sv, KDA, keep_rs_t>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H, n_tokens,
                                                                       sq1, sq2, sq3, sv1, sv2, sv3, sb1, sb2,
                                                                       sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K);
                                     });
            }
            थांबा;
        स्थिती 32:
            {
                constexpr पूर्णांक sv = 32;
                stream->parallel_for(sycl::nd_range<3>(grid_dims * block_dims, block_dims),
                                     [=](sycl::nd_item<3> /*item_ct1*/) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                                         gated_delta_net_sycl<sv, KDA, keep_rs_t>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H, n_tokens,
                                                                       sq1, sq2, sq3, sv1, sv2, sv3, sb1, sb2,
                                                                       sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K);
                                     });
            }
            थांबा;
        स्थिती 64: {
            {
                constexpr पूर्णांक sv = 64;
                stream->parallel_for(sycl::nd_range<3>(grid_dims * block_dims, block_dims),
                                        [=](sycl::nd_item<3> /*item_ct1*/) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                                            gated_delta_net_sycl<sv, KDA, keep_rs_t>(
                                                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H, n_tokens, sq1, sq2,
                                                sq3, sv1, sv2, sv3, sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K);
                                        });
            }
            थांबा;
        }
        स्थिती 128: {
            {
                constexpr पूर्णांक sv = 128;
                stream->parallel_for(sycl::nd_range<3>(grid_dims * block_dims, block_dims),
                                        [=](sycl::nd_item<3> /*item_ct1*/) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                                            gated_delta_net_sycl<sv, KDA, keep_rs_t>(
                                                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H, n_tokens, sq1, sq2,
                                                sq3, sv1, sv2, sv3, sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K);
                                        });
            }
            थांबा;
        }
        पूर्वनिर्धारित:
            GGML_ABORT("fatal error");
            थांबा;
    }
}

स्थिर शून्य ggml_sycl_op_gated_delta_net_impl(ggml_backend_sycl_context & ctx, ggml_tensor * dst,
                                              स्थिरांक ggml_sycl_gated_delta_net_fused_cache * cache) {
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

    स्थिरांक int64_t S_v      = nev0;
    स्थिरांक int64_t H        = nev1;
    स्थिरांक int64_t n_tokens = nev2;
    स्थिरांक int64_t n_seqs   = nev3;

    स्थिरांक तार्किक kda = (src_g->ne[0] == S_v);

    GGML_ASSERT(neq1 == nek1);
    स्थिरांक int64_t neqk1 = neq1;

    स्थिरांक int64_t rq3 = nev3 / neq3;

    स्थिरांक दशांश * q_d = (स्थिरांक दशांश *) src_q->data;
    स्थिरांक दशांश * k_d = (स्थिरांक दशांश *) src_k->data;
    स्थिरांक दशांश * v_d = (स्थिरांक दशांश *) src_v->data;
    स्थिरांक दशांश * g_d = (स्थिरांक दशांश *) src_g->data;
    स्थिरांक दशांश * b_d = (स्थिरांक दशांश *) src_beta->data;

    स्थिरांक दशांश * s_d   = (स्थिरांक दशांश *) src_state->data;
    दशांश *       dst_d = (दशांश *) dst->data;

    GGML_ASSERT(ggml_is_contiguous_rows(src_q));
    GGML_ASSERT(ggml_is_contiguous_rows(src_k));
    GGML_ASSERT(ggml_is_contiguous_rows(src_v));
    GGML_ASSERT(ggml_are_same_stride(src_q, src_k));
    GGML_ASSERT(src_g->ne[0] == 1 || kda);
    GGML_ASSERT(ggml_is_contiguous(src_g));
    GGML_ASSERT(ggml_is_contiguous(src_beta));
    GGML_ASSERT(ggml_is_contiguous(src_state));

    // strides in floats (beta strides used for both g and beta offset computation)
    स्थिरांक int64_t sq1 = nbq1 / आकार(दशांश);
    स्थिरांक int64_t sq2 = nbq2 / आकार(दशांश);
    स्थिरांक int64_t sq3 = nbq3 / आकार(दशांश);
    स्थिरांक int64_t sv1 = nbv1 / आकार(दशांश);
    स्थिरांक int64_t sv2 = nbv2 / आकार(दशांश);
    स्थिरांक int64_t sv3 = nbv3 / आकार(दशांश);
    स्थिरांक int64_t sb1 = nbb1 / आकार(दशांश);
    स्थिरांक int64_t sb2 = nbb2 / आकार(दशांश);
    स्थिरांक int64_t sb3 = nbb3 / आकार(दशांश);

    स्थिरांक दशांश scale = 1.0f / sqrtf((दशांश) S_v);

    dpct::queue_ptr stream = ctx.stream();

    // K (snapshot slot count) is an op param; state holds s0 only [S_v, S_v, H, n_seqs].
    स्थिरांक पूर्णांक K = ggml_get_op_params_i32(dst, 0);
    स्थिरांक तार्किक keep_rs = K > 1;

    // recurrent state -> dst tail (after attention scores), or the cache when fusing
    दशांश * state_d           = dst_d + S_v * H * n_tokens * n_seqs;
    int64_t state_slot_stride = S_v * S_v * H * n_seqs;
    जर (cache != nullptr) {
        state_d           = cache->data;
        state_slot_stride = cache->slot_stride;
    }

    जर (kda) {
        जर (keep_rs) {
            launch_gated_delta_net<सत्य, सत्य>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, stream);
        } अन्यथा {
            launch_gated_delta_net<सत्य, असत्य>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, stream);
        }
    } अन्यथा {
        जर (keep_rs) {
            launch_gated_delta_net<असत्य, सत्य>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, stream);
        } अन्यथा {
            launch_gated_delta_net<असत्य, असत्य>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, stream);
        }
    }
}

शून्य ggml_sycl_op_gated_delta_net(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_op_gated_delta_net_impl(ctx, dst, nullptr);
}

शून्य ggml_sycl_gated_delta_net(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/6);
    ggml_sycl_op_gated_delta_net(ctx, dst);
}

शून्य ggml_sycl_op_gated_delta_net_fused_cache(ggml_backend_sycl_context & ctx, ggml_tensor * dst,
                                              ggml_sycl_gated_delta_net_fused_cache cache) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/6);
    ggml_sycl_op_gated_delta_net_impl(ctx, dst, &cache);
}
