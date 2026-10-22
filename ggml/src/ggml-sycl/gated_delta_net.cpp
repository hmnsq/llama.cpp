#شامل <sycl/sycl.hpp>
#شامل "dpct/helper.hpp"
#شامل "common.hpp"
#شامل "ggml.h"
#شامل "gated_delta_net.hpp"
#شامل <cmath>


سانچہ <صحیح S_v, منطقی KDA, منطقی keep_rs_t>
خالی gated_delta_net_sycl(ثابت اعشاری *     q,
                          ثابت اعشاری *     k,
                          ثابت اعشاری *     v,
                          ثابت اعشاری *     g,
                          ثابت اعشاری *     beta,
                          ثابت اعشاری *     curr_state,
                          اعشاری *           dst,
                          اعشاری *           state,
                          int64_t           H,
                          int64_t           n_علامات,
                          int64_t           sq1,
                          int64_t           sq2,
                          int64_t           sq3,
                          int64_t           sv1,
                          int64_t           sv2,
                          int64_t           sv3,
                          int64_t           sb1,
                          int64_t           sb2,
                          int64_t           sb3,
                          ثابت sycl::uint3 neqk1_magic,
                          ثابت sycl::uint3 rq3_magic,
                          اعشاری             scale,
                          int64_t           state_خانہ_stride,
                          صحیح               K) {
    خودکار           شے_ct1 = sycl::ext::oneapi::this_work_شے::get_nd_شے<3>();
    ثابت uint32_t h_idx    = شے_ct1.get_گروہ(2);
    ثابت uint32_t sequence = شے_ct1.get_گروہ(1);
    // each warp owns one column, using warp-level primitives to reduce across rows
    ثابت صحیح      lane     = شے_ct1.get_local_id(2);
    ثابت صحیح      ستون      = شے_ct1.get_گروہ(0) * شے_ct1.get_local_range(1) + شے_ct1.get_local_id(1);

    ثابت uint32_t iq1 = fastmodulo(h_idx, neqk1_magic);
    ثابت uint32_t iq3 = fastdiv(sequence, rq3_magic);

    اعشاری *       attn_ڈیٹا        = dst;

    // input state holds s0 only [S_v, S_v, H, n_seqs] — seq stride is D = H * S_v * S_v.
    // output state layout (per-slot D * n_seqs) — same per-(seq,head) offset as before.
    ثابت int64_t state_اندر_انحراف      = sequence * H * S_v * S_v + h_idx * S_v * S_v;
    ثابت int64_t state_برآمد_انحراف     = (sequence * H + h_idx) * S_v * S_v;
    state += state_برآمد_انحراف;
    curr_state += state_اندر_انحراف + ستون * S_v;
    attn_ڈیٹا += (sequence * n_علامات * H + h_idx) * S_v;

    constexpr صحیح warp_سائز = ggml_sycl_get_physical_warp_سائز() < S_v ? ggml_sycl_get_physical_warp_سائز() : S_v;
    static_assert(S_v % warp_سائز == 0, "S_v must be a multiple of warp_size");
    constexpr صحیح صفیں_فی_lane = (S_v + warp_سائز - 1) / warp_سائز;
    اعشاری         s_shard[صفیں_فی_lane];
#ہدایت unroll
    برائے (صحیح r = 0; r < صفیں_فی_lane; r++) {
        ثابت صحیح i = r * warp_سائز + lane;
        s_shard[r]  = curr_state[i];
    }

    // snapshot slot mapping: slot 0 = most recent state, slot s = s tokens back.
    // When n_tokens < K only slots 0..n_tokens-1 are written; older slots are caller-owned.

    برائے (صحیح t = 0; t < n_علامات; t++) {
        ثابت اعشاری * q_t = q + iq3 * sq3 + t * sq2 + iq1 * sq1;
        ثابت اعشاری * k_t = k + iq3 * sq3 + t * sq2 + iq1 * sq1;
        ثابت اعشاری * v_t = v + sequence * sv3 + t * sv2 + h_idx * sv1;

        ثابت int64_t gb_انحراف = sequence * sb3 + t * sb2 + h_idx * sb1;
        ثابت اعشاری * beta_t = beta + gb_انحراف;
        ثابت اعشاری * g_t    = g    + gb_انحراف * (KDA ? S_v : 1);

        ثابت اعشاری beta_val = *beta_t;

        اگر constexpr (!KDA) {
            ثابت اعشاری g_val = sycl::native::قوت_نما(*g_t);

            // kv[col] = (S^T @ k)[col] = sum_i S[i][col] * k[i]
            اعشاری kv_shard = 0.0f;
#ہدایت unroll
            برائے (صحیح r = 0; r < صفیں_فی_lane; r++) {
                ثابت صحیح i = r * warp_سائز + lane;
                kv_shard += s_shard[r] * k_t[i];
            }
            اعشاری kv_ستون = warp_reduce_مجموع<warp_سائز>(kv_shard);

            // delta[col] = (v[col] - g * kv[col]) * beta
            اعشاری delta_ستون = (v_t[ستون] - g_val * kv_ستون) * beta_val;

            // fused: S[i][col] = g * S[i][col] + k[i] * delta[col]
            // attn[col] = (S^T @ q)[col] = sum_i S[i][col] * q[i]
            اعشاری attn_partial = 0.0f;
#ہدایت unroll
            برائے (صحیح r = 0; r < صفیں_فی_lane; r++) {
                ثابت صحیح i = r * warp_سائز + lane;
                s_shard[r]  = g_val * s_shard[r] + k_t[i] * delta_ستون;
                attn_partial += s_shard[r] * q_t[i];
            }

            اعشاری attn_ستون = warp_reduce_مجموع<warp_سائز>(attn_partial);

            اگر (lane == 0) {
                attn_ڈیٹا[ستون] = attn_ستون * scale;
            }
        } ورنہ {
            // kv[col] = sum_i g[i] * S[i][col] * k[i]
            اعشاری kv_shard = 0.0f;
#ہدایت unroll
            برائے (صحیح r = 0; r < صفیں_فی_lane; r++) {
                ثابت صحیح i = r * warp_سائز + lane;
                kv_shard += sycl::native::قوت_نما(g_t[i]) * s_shard[r] * k_t[i];
            }

            اعشاری kv_ستون = warp_reduce_مجموع<warp_سائز>(kv_shard);

            // delta[col] = (v[col] - kv[col]) * beta
            اعشاری delta_ستون = (v_t[ستون] - kv_ستون) * beta_val;

            // fused: S[i][col] = g[i] * S[i][col] + k[i] * delta[col]
            // attn[col] = (S^T @ q)[col] = sum_i S[i][col] * q[i]
            اعشاری attn_partial = 0.0f;
#ہدایت unroll
            برائے (صحیح r = 0; r < صفیں_فی_lane; r++) {
                ثابت صحیح i = r * warp_سائز + lane;
                s_shard[r]  = sycl::native::قوت_نما(g_t[i]) * s_shard[r] + k_t[i] * delta_ستون;
                attn_partial += s_shard[r] * q_t[i];
            }

            اعشاری attn_ستون = warp_reduce_مجموع<warp_سائز>(attn_partial);

            اگر (lane == 0) {
                attn_ڈیٹا[ستون] = attn_ستون * scale;
            }
        }

        attn_ڈیٹا += S_v * H;


    // Write state back to global memory
        اگر constexpr (keep_rs_t) {
            ثابت صحیح ہدف_خانہ = (صحیح) n_علامات - 1 - t;
            اگر (ہدف_خانہ >= 0 && ہدف_خانہ < K) {
                اعشاری * curr_state = state + ہدف_خانہ * state_خانہ_stride;
#ہدایت unroll
                برائے (صحیح r = 0; r < صفیں_فی_lane; r++) {
                    ثابت صحیح i = r * warp_سائز + lane;
                    curr_state[ستون * S_v + i] = s_shard[r];
                }
            }
        }
    }

    اگر constexpr (!keep_rs_t) {
#ہدایت unroll
        برائے (صحیح r = 0; r < صفیں_فی_lane; r++) {
            ثابت صحیح i          = r * warp_سائز + lane;
            state[ستون * S_v + i] = s_shard[r];
        }
    }
}

سانچہ <منطقی KDA, منطقی keep_rs_t>
ساکن خالی launch_gated_delta_net(ثابت اعشاری *   q_d,
                                   ثابت اعشاری *   k_d,
                                   ثابت اعشاری *   v_d,
                                   ثابت اعشاری *   g_d,
                                   ثابت اعشاری *   b_d,
                                   ثابت اعشاری *   s_d,
                                   اعشاری *         dst_d,
                                   اعشاری *         state_d,
                                   int64_t         S_v,
                                   int64_t         H,
                                   int64_t         n_علامات,
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
                                   اعشاری           scale,
                                   int64_t         state_خانہ_stride,
                                   صحیح             K,
                                   dpct::قطار_ptr stream) {
    //TODO: Add chunked kernel for even faster pre-fill
    ثابت صحیح warp_سائز = ggml_sycl_info().devices[ggml_sycl_get_device()].warp_سائز;

    ثابت صحیح عدد_warps = 4;
    dpct::dim3 grid_dims(H, n_seqs, (S_v + عدد_warps - 1) / عدد_warps);
    dpct::dim3 بلاک_dims(warp_سائز <= S_v ? warp_سائز : S_v, عدد_warps, 1);

    ثابت sycl::uint3 neqk1_magic = init_fastdiv_اقدار(neqk1);
    ثابت sycl::uint3 rq3_magic   = init_fastdiv_اقدار(rq3);

    انتخاب (S_v) {
        حالت 16:
            {
                constexpr صحیح sv = 16;
                stream->parallel_for(sycl::nd_range<3>(grid_dims * بلاک_dims, بلاک_dims),
                                     [=](sycl::nd_شے<3> /*item_ct1*/) [[sycl::reqd_sub_گروہ_سائز(WARP_سائز)]] {
                                         gated_delta_net_sycl<sv, KDA, keep_rs_t>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H, n_علامات,
                                                                       sq1, sq2, sq3, sv1, sv2, sv3, sb1, sb2,
                                                                       sb3, neqk1_magic, rq3_magic, scale, state_خانہ_stride, K);
                                     });
            }
            توڑو;
        حالت 32:
            {
                constexpr صحیح sv = 32;
                stream->parallel_for(sycl::nd_range<3>(grid_dims * بلاک_dims, بلاک_dims),
                                     [=](sycl::nd_شے<3> /*item_ct1*/) [[sycl::reqd_sub_گروہ_سائز(WARP_سائز)]] {
                                         gated_delta_net_sycl<sv, KDA, keep_rs_t>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H, n_علامات,
                                                                       sq1, sq2, sq3, sv1, sv2, sv3, sb1, sb2,
                                                                       sb3, neqk1_magic, rq3_magic, scale, state_خانہ_stride, K);
                                     });
            }
            توڑو;
        حالت 64: {
            {
                constexpr صحیح sv = 64;
                stream->parallel_for(sycl::nd_range<3>(grid_dims * بلاک_dims, بلاک_dims),
                                        [=](sycl::nd_شے<3> /*item_ct1*/) [[sycl::reqd_sub_گروہ_سائز(WARP_سائز)]] {
                                            gated_delta_net_sycl<sv, KDA, keep_rs_t>(
                                                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H, n_علامات, sq1, sq2,
                                                sq3, sv1, sv2, sv3, sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_خانہ_stride, K);
                                        });
            }
            توڑو;
        }
        حالت 128: {
            {
                constexpr صحیح sv = 128;
                stream->parallel_for(sycl::nd_range<3>(grid_dims * بلاک_dims, بلاک_dims),
                                        [=](sycl::nd_شے<3> /*item_ct1*/) [[sycl::reqd_sub_گروہ_سائز(WARP_سائز)]] {
                                            gated_delta_net_sycl<sv, KDA, keep_rs_t>(
                                                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H, n_علامات, sq1, sq2,
                                                sq3, sv1, sv2, sv3, sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_خانہ_stride, K);
                                        });
            }
            توڑو;
        }
        طےشدہ:
            GGML_ABORT("fatal error");
            توڑو;
    }
}

ساکن خالی ggml_sycl_op_gated_delta_net_impl(ggml_backend_sycl_context & ctx, ggml_tensor * dst,
                                              ثابت ggml_sycl_gated_delta_net_fused_cache * cache) {
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

    ثابت int64_t S_v      = nev0;
    ثابت int64_t H        = nev1;
    ثابت int64_t n_علامات = nev2;
    ثابت int64_t n_seqs   = nev3;

    ثابت منطقی kda = (src_g->ne[0] == S_v);

    GGML_ASSERT(neq1 == nek1);
    ثابت int64_t neqk1 = neq1;

    ثابت int64_t rq3 = nev3 / neq3;

    ثابت اعشاری * q_d = (ثابت اعشاری *) src_q->ڈیٹا;
    ثابت اعشاری * k_d = (ثابت اعشاری *) src_k->ڈیٹا;
    ثابت اعشاری * v_d = (ثابت اعشاری *) src_v->ڈیٹا;
    ثابت اعشاری * g_d = (ثابت اعشاری *) src_g->ڈیٹا;
    ثابت اعشاری * b_d = (ثابت اعشاری *) src_beta->ڈیٹا;

    ثابت اعشاری * s_d   = (ثابت اعشاری *) src_state->ڈیٹا;
    اعشاری *       dst_d = (اعشاری *) dst->ڈیٹا;

    GGML_ASSERT(ggml_is_contiguous_صفیں(src_q));
    GGML_ASSERT(ggml_is_contiguous_صفیں(src_k));
    GGML_ASSERT(ggml_is_contiguous_صفیں(src_v));
    GGML_ASSERT(ggml_are_same_stride(src_q, src_k));
    GGML_ASSERT(src_g->ne[0] == 1 || kda);
    GGML_ASSERT(ggml_is_contiguous(src_g));
    GGML_ASSERT(ggml_is_contiguous(src_beta));
    GGML_ASSERT(ggml_is_contiguous(src_state));

    // strides in floats (beta strides used for both g and beta offset computation)
    ثابت int64_t sq1 = nbq1 / حجم(اعشاری);
    ثابت int64_t sq2 = nbq2 / حجم(اعشاری);
    ثابت int64_t sq3 = nbq3 / حجم(اعشاری);
    ثابت int64_t sv1 = nbv1 / حجم(اعشاری);
    ثابت int64_t sv2 = nbv2 / حجم(اعشاری);
    ثابت int64_t sv3 = nbv3 / حجم(اعشاری);
    ثابت int64_t sb1 = nbb1 / حجم(اعشاری);
    ثابت int64_t sb2 = nbb2 / حجم(اعشاری);
    ثابت int64_t sb3 = nbb3 / حجم(اعشاری);

    ثابت اعشاری scale = 1.0f / sqrtf((اعشاری) S_v);

    dpct::قطار_ptr stream = ctx.stream();

    // K (snapshot slot count) is an op param; state holds s0 only [S_v, S_v, H, n_seqs].
    ثابت صحیح K = ggml_get_op_params_i32(dst, 0);
    ثابت منطقی keep_rs = K > 1;

    // recurrent state -> dst tail (after attention scores), or the cache when fusing
    اعشاری * state_d           = dst_d + S_v * H * n_علامات * n_seqs;
    int64_t state_خانہ_stride = S_v * S_v * H * n_seqs;
    اگر (cache != خالی_اشاریہ) {
        state_d           = cache->ڈیٹا;
        state_خانہ_stride = cache->خانہ_stride;
    }

    اگر (kda) {
        اگر (keep_rs) {
            launch_gated_delta_net<سچ, سچ>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_علامات, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_خانہ_stride, K, stream);
        } ورنہ {
            launch_gated_delta_net<سچ, جھوٹ>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_علامات, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_خانہ_stride, K, stream);
        }
    } ورنہ {
        اگر (keep_rs) {
            launch_gated_delta_net<جھوٹ, سچ>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_علامات, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_خانہ_stride, K, stream);
        } ورنہ {
            launch_gated_delta_net<جھوٹ, جھوٹ>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_علامات, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_خانہ_stride, K, stream);
        }
    }
}

خالی ggml_sycl_op_gated_delta_net(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_op_gated_delta_net_impl(ctx, dst, خالی_اشاریہ);
}

خالی ggml_sycl_gated_delta_net(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/6);
    ggml_sycl_op_gated_delta_net(ctx, dst);
}

خالی ggml_sycl_op_gated_delta_net_fused_cache(ggml_backend_sycl_context & ctx, ggml_tensor * dst,
                                              ggml_sycl_gated_delta_net_fused_cache cache) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/6);
    ggml_sycl_op_gated_delta_net_impl(ctx, dst, &cache);
}
