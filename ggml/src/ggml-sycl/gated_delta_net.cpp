#अन्तर्भाव <sycl/sycl.hpp>
#अन्तर्भाव "dpct/helper.hpp"
#अन्तर्भाव "common.hpp"
#अन्तर्भाव "ggml.h"
#अन्तर्भाव "gated_delta_net.hpp"
#अन्तर्भाव <cmath>


प्रतिरूपम् <पूर्णाङ्क S_v, तार्किक KDA, तार्किक keep_rs_t>
शून्य gated_delta_net_sycl(ध्रुव भिन्न *     q,
                          ध्रुव भिन्न *     k,
                          ध्रुव भिन्न *     v,
                          ध्रुव भिन्न *     g,
                          ध्रुव भिन्न *     beta,
                          ध्रुव भिन्न *     curr_state,
                          भिन्न *           dst,
                          भिन्न *           state,
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
                          ध्रुव sycl::uint3 neqk1_magic,
                          ध्रुव sycl::uint3 rq3_magic,
                          भिन्न             scale,
                          int64_t           state_slot_stride,
                          पूर्णाङ्क               K) {
    स्वचालित           item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    ध्रुव uint32_t h_idx    = item_ct1.get_गणः(2);
    ध्रुव uint32_t sequence = item_ct1.get_गणः(1);
    // each warp owns one column, using warp-level primitives to reduce across rows
    ध्रुव पूर्णाङ्क      lane     = item_ct1.get_local_id(2);
    ध्रुव पूर्णाङ्क      col      = item_ct1.get_गणः(0) * item_ct1.get_local_range(1) + item_ct1.get_local_id(1);

    ध्रुव uint32_t iq1 = fastmodulo(h_idx, neqk1_magic);
    ध्रुव uint32_t iq3 = fastdiv(sequence, rq3_magic);

    भिन्न *       attn_दत्त        = dst;

    // input state holds s0 only [S_v, S_v, H, n_seqs] — seq stride is D = H * S_v * S_v.
    // output state layout (per-slot D * n_seqs) — same per-(seq,head) offset as before.
    ध्रुव int64_t state_अन्तर्_विस्थापनम्      = sequence * H * S_v * S_v + h_idx * S_v * S_v;
    ध्रुव int64_t state_बहिर्गमनम्_विस्थापनम्     = (sequence * H + h_idx) * S_v * S_v;
    state += state_बहिर्गमनम्_विस्थापनम्;
    curr_state += state_अन्तर्_विस्थापनम् + col * S_v;
    attn_दत्त += (sequence * n_tokens * H + h_idx) * S_v;

    constexpr पूर्णाङ्क warp_आकार = ggml_sycl_get_physical_warp_आकार() < S_v ? ggml_sycl_get_physical_warp_आकार() : S_v;
    static_assert(S_v % warp_आकार == 0, "S_v must be a multiple of warp_size");
    constexpr पूर्णाङ्क पङ्क्तिकाः_प्रत्येकम्_lane = (S_v + warp_आकार - 1) / warp_आकार;
    भिन्न         s_shard[पङ्क्तिकाः_प्रत्येकम्_lane];
#निर्देशः unroll
    प्रति (पूर्णाङ्क r = 0; r < पङ्क्तिकाः_प्रत्येकम्_lane; r++) {
        ध्रुव पूर्णाङ्क i = r * warp_आकार + lane;
        s_shard[r]  = curr_state[i];
    }

    // snapshot slot mapping: slot 0 = most recent state, slot s = s tokens back.
    // When n_tokens < K only slots 0..n_tokens-1 are written; older slots are caller-owned.

    प्रति (पूर्णाङ्क t = 0; t < n_tokens; t++) {
        ध्रुव भिन्न * q_t = q + iq3 * sq3 + t * sq2 + iq1 * sq1;
        ध्रुव भिन्न * k_t = k + iq3 * sq3 + t * sq2 + iq1 * sq1;
        ध्रुव भिन्न * v_t = v + sequence * sv3 + t * sv2 + h_idx * sv1;

        ध्रुव int64_t gb_विस्थापनम् = sequence * sb3 + t * sb2 + h_idx * sb1;
        ध्रुव भिन्न * beta_t = beta + gb_विस्थापनम्;
        ध्रुव भिन्न * g_t    = g    + gb_विस्थापनम् * (KDA ? S_v : 1);

        ध्रुव भिन्न beta_val = *beta_t;

        यदि constexpr (!KDA) {
            ध्रुव भिन्न g_val = sycl::native::घातांकः(*g_t);

            // kv[col] = (S^T @ k)[col] = sum_i S[i][col] * k[i]
            भिन्न kv_shard = 0.0f;
#निर्देशः unroll
            प्रति (पूर्णाङ्क r = 0; r < पङ्क्तिकाः_प्रत्येकम्_lane; r++) {
                ध्रुव पूर्णाङ्क i = r * warp_आकार + lane;
                kv_shard += s_shard[r] * k_t[i];
            }
            भिन्न kv_col = warp_reduce_योग<warp_आकार>(kv_shard);

            // delta[col] = (v[col] - g * kv[col]) * beta
            भिन्न delta_col = (v_t[col] - g_val * kv_col) * beta_val;

            // fused: S[i][col] = g * S[i][col] + k[i] * delta[col]
            // attn[col] = (S^T @ q)[col] = sum_i S[i][col] * q[i]
            भिन्न attn_partial = 0.0f;
#निर्देशः unroll
            प्रति (पूर्णाङ्क r = 0; r < पङ्क्तिकाः_प्रत्येकम्_lane; r++) {
                ध्रुव पूर्णाङ्क i = r * warp_आकार + lane;
                s_shard[r]  = g_val * s_shard[r] + k_t[i] * delta_col;
                attn_partial += s_shard[r] * q_t[i];
            }

            भिन्न attn_col = warp_reduce_योग<warp_आकार>(attn_partial);

            यदि (lane == 0) {
                attn_दत्त[col] = attn_col * scale;
            }
        } अन्यथा {
            // kv[col] = sum_i g[i] * S[i][col] * k[i]
            भिन्न kv_shard = 0.0f;
#निर्देशः unroll
            प्रति (पूर्णाङ्क r = 0; r < पङ्क्तिकाः_प्रत्येकम्_lane; r++) {
                ध्रुव पूर्णाङ्क i = r * warp_आकार + lane;
                kv_shard += sycl::native::घातांकः(g_t[i]) * s_shard[r] * k_t[i];
            }

            भिन्न kv_col = warp_reduce_योग<warp_आकार>(kv_shard);

            // delta[col] = (v[col] - kv[col]) * beta
            भिन्न delta_col = (v_t[col] - kv_col) * beta_val;

            // fused: S[i][col] = g[i] * S[i][col] + k[i] * delta[col]
            // attn[col] = (S^T @ q)[col] = sum_i S[i][col] * q[i]
            भिन्न attn_partial = 0.0f;
#निर्देशः unroll
            प्रति (पूर्णाङ्क r = 0; r < पङ्क्तिकाः_प्रत्येकम्_lane; r++) {
                ध्रुव पूर्णाङ्क i = r * warp_आकार + lane;
                s_shard[r]  = sycl::native::घातांकः(g_t[i]) * s_shard[r] + k_t[i] * delta_col;
                attn_partial += s_shard[r] * q_t[i];
            }

            भिन्न attn_col = warp_reduce_योग<warp_आकार>(attn_partial);

            यदि (lane == 0) {
                attn_दत्त[col] = attn_col * scale;
            }
        }

        attn_दत्त += S_v * H;


    // Write state back to global memory
        यदि constexpr (keep_rs_t) {
            ध्रुव पूर्णाङ्क target_slot = (पूर्णाङ्क) n_tokens - 1 - t;
            यदि (target_slot >= 0 && target_slot < K) {
                भिन्न * curr_state = state + target_slot * state_slot_stride;
#निर्देशः unroll
                प्रति (पूर्णाङ्क r = 0; r < पङ्क्तिकाः_प्रत्येकम्_lane; r++) {
                    ध्रुव पूर्णाङ्क i = r * warp_आकार + lane;
                    curr_state[col * S_v + i] = s_shard[r];
                }
            }
        }
    }

    यदि constexpr (!keep_rs_t) {
#निर्देशः unroll
        प्रति (पूर्णाङ्क r = 0; r < पङ्क्तिकाः_प्रत्येकम्_lane; r++) {
            ध्रुव पूर्णाङ्क i          = r * warp_आकार + lane;
            state[col * S_v + i] = s_shard[r];
        }
    }
}

प्रतिरूपम् <तार्किक KDA, तार्किक keep_rs_t>
स्थिर शून्य launch_gated_delta_net(ध्रुव भिन्न *   q_d,
                                   ध्रुव भिन्न *   k_d,
                                   ध्रुव भिन्न *   v_d,
                                   ध्रुव भिन्न *   g_d,
                                   ध्रुव भिन्न *   b_d,
                                   ध्रुव भिन्न *   s_d,
                                   भिन्न *         dst_d,
                                   भिन्न *         state_d,
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
                                   भिन्न           scale,
                                   int64_t         state_slot_stride,
                                   पूर्णाङ्क             K,
                                   dpct::queue_ptr stream) {
    //TODO: Add chunked kernel for even faster pre-fill
    ध्रुव पूर्णाङ्क warp_आकार = ggml_sycl_info().devices[ggml_sycl_get_device()].warp_आकार;

    ध्रुव पूर्णाङ्क अङ्कः_warps = 4;
    dpct::dim3 grid_dims(H, n_seqs, (S_v + अङ्कः_warps - 1) / अङ्कः_warps);
    dpct::dim3 खण्डम्_dims(warp_आकार <= S_v ? warp_आकार : S_v, अङ्कः_warps, 1);

    ध्रुव sycl::uint3 neqk1_magic = init_fastdiv_मानानि(neqk1);
    ध्रुव sycl::uint3 rq3_magic   = init_fastdiv_मानानि(rq3);

    विकल्प (S_v) {
        पक्ष 16:
            {
                constexpr पूर्णाङ्क sv = 16;
                stream->parallel_for(sycl::nd_range<3>(grid_dims * खण्डम्_dims, खण्डम्_dims),
                                     [=](sycl::nd_item<3> /*item_ct1*/) [[sycl::reqd_sub_गणः_आकार(WARP_आकार)]] {
                                         gated_delta_net_sycl<sv, KDA, keep_rs_t>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H, n_tokens,
                                                                       sq1, sq2, sq3, sv1, sv2, sv3, sb1, sb2,
                                                                       sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K);
                                     });
            }
            भङ्ग;
        पक्ष 32:
            {
                constexpr पूर्णाङ्क sv = 32;
                stream->parallel_for(sycl::nd_range<3>(grid_dims * खण्डम्_dims, खण्डम्_dims),
                                     [=](sycl::nd_item<3> /*item_ct1*/) [[sycl::reqd_sub_गणः_आकार(WARP_आकार)]] {
                                         gated_delta_net_sycl<sv, KDA, keep_rs_t>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H, n_tokens,
                                                                       sq1, sq2, sq3, sv1, sv2, sv3, sb1, sb2,
                                                                       sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K);
                                     });
            }
            भङ्ग;
        पक्ष 64: {
            {
                constexpr पूर्णाङ्क sv = 64;
                stream->parallel_for(sycl::nd_range<3>(grid_dims * खण्डम्_dims, खण्डम्_dims),
                                        [=](sycl::nd_item<3> /*item_ct1*/) [[sycl::reqd_sub_गणः_आकार(WARP_आकार)]] {
                                            gated_delta_net_sycl<sv, KDA, keep_rs_t>(
                                                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H, n_tokens, sq1, sq2,
                                                sq3, sv1, sv2, sv3, sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K);
                                        });
            }
            भङ्ग;
        }
        पक्ष 128: {
            {
                constexpr पूर्णाङ्क sv = 128;
                stream->parallel_for(sycl::nd_range<3>(grid_dims * खण्डम्_dims, खण्डम्_dims),
                                        [=](sycl::nd_item<3> /*item_ct1*/) [[sycl::reqd_sub_गणः_आकार(WARP_आकार)]] {
                                            gated_delta_net_sycl<sv, KDA, keep_rs_t>(
                                                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H, n_tokens, sq1, sq2,
                                                sq3, sv1, sv2, sv3, sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K);
                                        });
            }
            भङ्ग;
        }
        स्वभाव:
            GGML_ABORT("fatal error");
            भङ्ग;
    }
}

स्थिर शून्य ggml_sycl_op_gated_delta_net_impl(ggml_backend_sycl_context & ctx, ggml_tensor * dst,
                                              ध्रुव ggml_sycl_gated_delta_net_fused_cache * cache) {
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

    ध्रुव int64_t S_v      = nev0;
    ध्रुव int64_t H        = nev1;
    ध्रुव int64_t n_tokens = nev2;
    ध्रुव int64_t n_seqs   = nev3;

    ध्रुव तार्किक kda = (src_g->ne[0] == S_v);

    GGML_ASSERT(neq1 == nek1);
    ध्रुव int64_t neqk1 = neq1;

    ध्रुव int64_t rq3 = nev3 / neq3;

    ध्रुव भिन्न * q_d = (ध्रुव भिन्न *) src_q->दत्त;
    ध्रुव भिन्न * k_d = (ध्रुव भिन्न *) src_k->दत्त;
    ध्रुव भिन्न * v_d = (ध्रुव भिन्न *) src_v->दत्त;
    ध्रुव भिन्न * g_d = (ध्रुव भिन्न *) src_g->दत्त;
    ध्रुव भिन्न * b_d = (ध्रुव भिन्न *) src_beta->दत्त;

    ध्रुव भिन्न * s_d   = (ध्रुव भिन्न *) src_state->दत्त;
    भिन्न *       dst_d = (भिन्न *) dst->दत्त;

    GGML_ASSERT(ggml_is_contiguous_पङ्क्तिकाः(src_q));
    GGML_ASSERT(ggml_is_contiguous_पङ्क्तिकाः(src_k));
    GGML_ASSERT(ggml_is_contiguous_पङ्क्तिकाः(src_v));
    GGML_ASSERT(ggml_are_same_stride(src_q, src_k));
    GGML_ASSERT(src_g->ne[0] == 1 || kda);
    GGML_ASSERT(ggml_is_contiguous(src_g));
    GGML_ASSERT(ggml_is_contiguous(src_beta));
    GGML_ASSERT(ggml_is_contiguous(src_state));

    // strides in floats (beta strides used for both g and beta offset computation)
    ध्रुव int64_t sq1 = nbq1 / परिमाण(भिन्न);
    ध्रुव int64_t sq2 = nbq2 / परिमाण(भिन्न);
    ध्रुव int64_t sq3 = nbq3 / परिमाण(भिन्न);
    ध्रुव int64_t sv1 = nbv1 / परिमाण(भिन्न);
    ध्रुव int64_t sv2 = nbv2 / परिमाण(भिन्न);
    ध्रुव int64_t sv3 = nbv3 / परिमाण(भिन्न);
    ध्रुव int64_t sb1 = nbb1 / परिमाण(भिन्न);
    ध्रुव int64_t sb2 = nbb2 / परिमाण(भिन्न);
    ध्रुव int64_t sb3 = nbb3 / परिमाण(भिन्न);

    ध्रुव भिन्न scale = 1.0f / sqrtf((भिन्न) S_v);

    dpct::queue_ptr stream = ctx.stream();

    // K (snapshot slot count) is an op param; state holds s0 only [S_v, S_v, H, n_seqs].
    ध्रुव पूर्णाङ्क K = ggml_get_op_params_i32(dst, 0);
    ध्रुव तार्किक keep_rs = K > 1;

    // recurrent state -> dst tail (after attention scores), or the cache when fusing
    भिन्न * state_d           = dst_d + S_v * H * n_tokens * n_seqs;
    int64_t state_slot_stride = S_v * S_v * H * n_seqs;
    यदि (cache != शून्यसूचकः) {
        state_d           = cache->दत्त;
        state_slot_stride = cache->slot_stride;
    }

    यदि (kda) {
        यदि (keep_rs) {
            launch_gated_delta_net<सत्य, सत्य>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, stream);
        } अन्यथा {
            launch_gated_delta_net<सत्य, असत्य>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, stream);
        }
    } अन्यथा {
        यदि (keep_rs) {
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
    ggml_sycl_op_gated_delta_net_impl(ctx, dst, शून्यसूचकः);
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
