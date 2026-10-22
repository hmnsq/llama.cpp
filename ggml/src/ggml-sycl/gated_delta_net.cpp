#ضمن <sycl/sycl.hpp>
#ضمن "dpct/helper.hpp"
#ضمن "common.hpp"
#ضمن "ggml.h"
#ضمن "gated_delta_net.hpp"
#ضمن <cmath>


سانچو <صحيح S_v, منطقي KDA, منطقي keep_rs_t>
فراغ gated_delta_net_sycl(ثابت عائم *     q,
                          ثابت عائم *     k,
                          ثابت عائم *     v,
                          ثابت عائم *     g,
                          ثابت عائم *     beta,
                          ثابت عائم *     curr_state,
                          عائم *           dst,
                          عائم *           state,
                          int64_t           H,
                          int64_t           n_ٽوڪن,
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
                          عائم             scale,
                          int64_t           state_جڳهه_stride,
                          صحيح               K) {
    خودڪار           شي_ct1 = sycl::ext::oneapi::this_work_شي::get_nd_شي<3>();
    ثابت uint32_t h_idx    = شي_ct1.get_گروهه(2);
    ثابت uint32_t sequence = شي_ct1.get_گروهه(1);
    // each warp owns one column, using warp-level primitives to reduce across rows
    ثابت صحيح      lane     = شي_ct1.get_local_id(2);
    ثابت صحيح      کانو      = شي_ct1.get_گروهه(0) * شي_ct1.get_local_range(1) + شي_ct1.get_local_id(1);

    ثابت uint32_t iq1 = fastmodulo(h_idx, neqk1_magic);
    ثابت uint32_t iq3 = fastdiv(sequence, rq3_magic);

    عائم *       attn_معلومات        = dst;

    // input state holds s0 only [S_v, S_v, H, n_seqs] — seq stride is D = H * S_v * S_v.
    // output state layout (per-slot D * n_seqs) — same per-(seq,head) offset as before.
    ثابت int64_t state_اندريون_وٿي      = sequence * H * S_v * S_v + h_idx * S_v * S_v;
    ثابت int64_t state_ٻاهريون_وٿي     = (sequence * H + h_idx) * S_v * S_v;
    state += state_ٻاهريون_وٿي;
    curr_state += state_اندريون_وٿي + کانو * S_v;
    attn_معلومات += (sequence * n_ٽوڪن * H + h_idx) * S_v;

    constexpr صحيح warp_سائيز = ggml_sycl_get_physical_warp_سائيز() < S_v ? ggml_sycl_get_physical_warp_سائيز() : S_v;
    static_assert(S_v % warp_سائيز == 0, "S_v must be a multiple of warp_size");
    constexpr صحيح صفون_في_lane = (S_v + warp_سائيز - 1) / warp_سائيز;
    عائم         s_shard[صفون_في_lane];
#هدايت unroll
    لكل (صحيح r = 0; r < صفون_في_lane; r++) {
        ثابت صحيح i = r * warp_سائيز + lane;
        s_shard[r]  = curr_state[i];
    }

    // snapshot slot mapping: slot 0 = most recent state, slot s = s tokens back.
    // When n_tokens < K only slots 0..n_tokens-1 are written; older slots are caller-owned.

    لكل (صحيح t = 0; t < n_ٽوڪن; t++) {
        ثابت عائم * q_t = q + iq3 * sq3 + t * sq2 + iq1 * sq1;
        ثابت عائم * k_t = k + iq3 * sq3 + t * sq2 + iq1 * sq1;
        ثابت عائم * v_t = v + sequence * sv3 + t * sv2 + h_idx * sv1;

        ثابت int64_t gb_وٿي = sequence * sb3 + t * sb2 + h_idx * sb1;
        ثابت عائم * beta_t = beta + gb_وٿي;
        ثابت عائم * g_t    = g    + gb_وٿي * (KDA ? S_v : 1);

        ثابت عائم beta_val = *beta_t;

        إذا constexpr (!KDA) {
            ثابت عائم g_val = sycl::native::قوت_نما(*g_t);

            // kv[col] = (S^T @ k)[col] = sum_i S[i][col] * k[i]
            عائم kv_shard = 0.0f;
#هدايت unroll
            لكل (صحيح r = 0; r < صفون_في_lane; r++) {
                ثابت صحيح i = r * warp_سائيز + lane;
                kv_shard += s_shard[r] * k_t[i];
            }
            عائم kv_کانو = warp_reduce_جوڙ<warp_سائيز>(kv_shard);

            // delta[col] = (v[col] - g * kv[col]) * beta
            عائم delta_کانو = (v_t[کانو] - g_val * kv_کانو) * beta_val;

            // fused: S[i][col] = g * S[i][col] + k[i] * delta[col]
            // attn[col] = (S^T @ q)[col] = sum_i S[i][col] * q[i]
            عائم attn_partial = 0.0f;
#هدايت unroll
            لكل (صحيح r = 0; r < صفون_في_lane; r++) {
                ثابت صحيح i = r * warp_سائيز + lane;
                s_shard[r]  = g_val * s_shard[r] + k_t[i] * delta_کانو;
                attn_partial += s_shard[r] * q_t[i];
            }

            عائم attn_کانو = warp_reduce_جوڙ<warp_سائيز>(attn_partial);

            إذا (lane == 0) {
                attn_معلومات[کانو] = attn_کانو * scale;
            }
        } وإلا {
            // kv[col] = sum_i g[i] * S[i][col] * k[i]
            عائم kv_shard = 0.0f;
#هدايت unroll
            لكل (صحيح r = 0; r < صفون_في_lane; r++) {
                ثابت صحيح i = r * warp_سائيز + lane;
                kv_shard += sycl::native::قوت_نما(g_t[i]) * s_shard[r] * k_t[i];
            }

            عائم kv_کانو = warp_reduce_جوڙ<warp_سائيز>(kv_shard);

            // delta[col] = (v[col] - kv[col]) * beta
            عائم delta_کانو = (v_t[کانو] - kv_کانو) * beta_val;

            // fused: S[i][col] = g[i] * S[i][col] + k[i] * delta[col]
            // attn[col] = (S^T @ q)[col] = sum_i S[i][col] * q[i]
            عائم attn_partial = 0.0f;
#هدايت unroll
            لكل (صحيح r = 0; r < صفون_في_lane; r++) {
                ثابت صحيح i = r * warp_سائيز + lane;
                s_shard[r]  = sycl::native::قوت_نما(g_t[i]) * s_shard[r] + k_t[i] * delta_کانو;
                attn_partial += s_shard[r] * q_t[i];
            }

            عائم attn_کانو = warp_reduce_جوڙ<warp_سائيز>(attn_partial);

            إذا (lane == 0) {
                attn_معلومات[کانو] = attn_کانو * scale;
            }
        }

        attn_معلومات += S_v * H;


    // Write state back to global memory
        إذا constexpr (keep_rs_t) {
            ثابت صحيح هدف_جڳهه = (صحيح) n_ٽوڪن - 1 - t;
            إذا (هدف_جڳهه >= 0 && هدف_جڳهه < K) {
                عائم * curr_state = state + هدف_جڳهه * state_جڳهه_stride;
#هدايت unroll
                لكل (صحيح r = 0; r < صفون_في_lane; r++) {
                    ثابت صحيح i = r * warp_سائيز + lane;
                    curr_state[کانو * S_v + i] = s_shard[r];
                }
            }
        }
    }

    إذا constexpr (!keep_rs_t) {
#هدايت unroll
        لكل (صحيح r = 0; r < صفون_في_lane; r++) {
            ثابت صحيح i          = r * warp_سائيز + lane;
            state[کانو * S_v + i] = s_shard[r];
        }
    }
}

سانچو <منطقي KDA, منطقي keep_rs_t>
ساكن فراغ launch_gated_delta_net(ثابت عائم *   q_d,
                                   ثابت عائم *   k_d,
                                   ثابت عائم *   v_d,
                                   ثابت عائم *   g_d,
                                   ثابت عائم *   b_d,
                                   ثابت عائم *   s_d,
                                   عائم *         dst_d,
                                   عائم *         state_d,
                                   int64_t         S_v,
                                   int64_t         H,
                                   int64_t         n_ٽوڪن,
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
                                   عائم           scale,
                                   int64_t         state_جڳهه_stride,
                                   صحيح             K,
                                   dpct::قطار_ptr stream) {
    //TODO: Add chunked kernel for even faster pre-fill
    ثابت صحيح warp_سائيز = ggml_sycl_info().devices[ggml_sycl_get_device()].warp_سائيز;

    ثابت صحيح انگ_warps = 4;
    dpct::dim3 grid_dims(H, n_seqs, (S_v + انگ_warps - 1) / انگ_warps);
    dpct::dim3 بلاڪ_dims(warp_سائيز <= S_v ? warp_سائيز : S_v, انگ_warps, 1);

    ثابت sycl::uint3 neqk1_magic = init_fastdiv_قدرون(neqk1);
    ثابت sycl::uint3 rq3_magic   = init_fastdiv_قدرون(rq3);

    بدل (S_v) {
        حالة 16:
            {
                constexpr صحيح sv = 16;
                stream->parallel_for(sycl::nd_range<3>(grid_dims * بلاڪ_dims, بلاڪ_dims),
                                     [=](sycl::nd_شي<3> /*item_ct1*/) [[sycl::reqd_sub_گروهه_سائيز(WARP_سائيز)]] {
                                         gated_delta_net_sycl<sv, KDA, keep_rs_t>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H, n_ٽوڪن,
                                                                       sq1, sq2, sq3, sv1, sv2, sv3, sb1, sb2,
                                                                       sb3, neqk1_magic, rq3_magic, scale, state_جڳهه_stride, K);
                                     });
            }
            اكسر;
        حالة 32:
            {
                constexpr صحيح sv = 32;
                stream->parallel_for(sycl::nd_range<3>(grid_dims * بلاڪ_dims, بلاڪ_dims),
                                     [=](sycl::nd_شي<3> /*item_ct1*/) [[sycl::reqd_sub_گروهه_سائيز(WARP_سائيز)]] {
                                         gated_delta_net_sycl<sv, KDA, keep_rs_t>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H, n_ٽوڪن,
                                                                       sq1, sq2, sq3, sv1, sv2, sv3, sb1, sb2,
                                                                       sb3, neqk1_magic, rq3_magic, scale, state_جڳهه_stride, K);
                                     });
            }
            اكسر;
        حالة 64: {
            {
                constexpr صحيح sv = 64;
                stream->parallel_for(sycl::nd_range<3>(grid_dims * بلاڪ_dims, بلاڪ_dims),
                                        [=](sycl::nd_شي<3> /*item_ct1*/) [[sycl::reqd_sub_گروهه_سائيز(WARP_سائيز)]] {
                                            gated_delta_net_sycl<sv, KDA, keep_rs_t>(
                                                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H, n_ٽوڪن, sq1, sq2,
                                                sq3, sv1, sv2, sv3, sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_جڳهه_stride, K);
                                        });
            }
            اكسر;
        }
        حالة 128: {
            {
                constexpr صحيح sv = 128;
                stream->parallel_for(sycl::nd_range<3>(grid_dims * بلاڪ_dims, بلاڪ_dims),
                                        [=](sycl::nd_شي<3> /*item_ct1*/) [[sycl::reqd_sub_گروهه_سائيز(WARP_سائيز)]] {
                                            gated_delta_net_sycl<sv, KDA, keep_rs_t>(
                                                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H, n_ٽوڪن, sq1, sq2,
                                                sq3, sv1, sv2, sv3, sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_جڳهه_stride, K);
                                        });
            }
            اكسر;
        }
        افتراضي:
            GGML_ABORT("fatal error");
            اكسر;
    }
}

ساكن فراغ ggml_sycl_op_gated_delta_net_impl(ggml_backend_sycl_context & ctx, ggml_tensor * dst,
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
    ثابت int64_t n_ٽوڪن = nev2;
    ثابت int64_t n_seqs   = nev3;

    ثابت منطقي kda = (src_g->ne[0] == S_v);

    GGML_ASSERT(neq1 == nek1);
    ثابت int64_t neqk1 = neq1;

    ثابت int64_t rq3 = nev3 / neq3;

    ثابت عائم * q_d = (ثابت عائم *) src_q->معلومات;
    ثابت عائم * k_d = (ثابت عائم *) src_k->معلومات;
    ثابت عائم * v_d = (ثابت عائم *) src_v->معلومات;
    ثابت عائم * g_d = (ثابت عائم *) src_g->معلومات;
    ثابت عائم * b_d = (ثابت عائم *) src_beta->معلومات;

    ثابت عائم * s_d   = (ثابت عائم *) src_state->معلومات;
    عائم *       dst_d = (عائم *) dst->معلومات;

    GGML_ASSERT(ggml_is_contiguous_صفون(src_q));
    GGML_ASSERT(ggml_is_contiguous_صفون(src_k));
    GGML_ASSERT(ggml_is_contiguous_صفون(src_v));
    GGML_ASSERT(ggml_are_same_stride(src_q, src_k));
    GGML_ASSERT(src_g->ne[0] == 1 || kda);
    GGML_ASSERT(ggml_is_contiguous(src_g));
    GGML_ASSERT(ggml_is_contiguous(src_beta));
    GGML_ASSERT(ggml_is_contiguous(src_state));

    // strides in floats (beta strides used for both g and beta offset computation)
    ثابت int64_t sq1 = nbq1 / حجم(عائم);
    ثابت int64_t sq2 = nbq2 / حجم(عائم);
    ثابت int64_t sq3 = nbq3 / حجم(عائم);
    ثابت int64_t sv1 = nbv1 / حجم(عائم);
    ثابت int64_t sv2 = nbv2 / حجم(عائم);
    ثابت int64_t sv3 = nbv3 / حجم(عائم);
    ثابت int64_t sb1 = nbb1 / حجم(عائم);
    ثابت int64_t sb2 = nbb2 / حجم(عائم);
    ثابت int64_t sb3 = nbb3 / حجم(عائم);

    ثابت عائم scale = 1.0f / sqrtf((عائم) S_v);

    dpct::قطار_ptr stream = ctx.stream();

    // K (snapshot slot count) is an op param; state holds s0 only [S_v, S_v, H, n_seqs].
    ثابت صحيح K = ggml_get_op_params_i32(dst, 0);
    ثابت منطقي keep_rs = K > 1;

    // recurrent state -> dst tail (after attention scores), or the cache when fusing
    عائم * state_d           = dst_d + S_v * H * n_ٽوڪن * n_seqs;
    int64_t state_جڳهه_stride = S_v * S_v * H * n_seqs;
    إذا (cache != خالي_اشارو) {
        state_d           = cache->معلومات;
        state_جڳهه_stride = cache->جڳهه_stride;
    }

    إذا (kda) {
        إذا (keep_rs) {
            launch_gated_delta_net<سچ, سچ>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_ٽوڪن, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_جڳهه_stride, K, stream);
        } وإلا {
            launch_gated_delta_net<سچ, خطأ>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_ٽوڪن, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_جڳهه_stride, K, stream);
        }
    } وإلا {
        إذا (keep_rs) {
            launch_gated_delta_net<خطأ, سچ>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_ٽوڪن, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_جڳهه_stride, K, stream);
        } وإلا {
            launch_gated_delta_net<خطأ, خطأ>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_ٽوڪن, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_جڳهه_stride, K, stream);
        }
    }
}

فراغ ggml_sycl_op_gated_delta_net(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_op_gated_delta_net_impl(ctx, dst, خالي_اشارو);
}

فراغ ggml_sycl_gated_delta_net(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/6);
    ggml_sycl_op_gated_delta_net(ctx, dst);
}

فراغ ggml_sycl_op_gated_delta_net_fused_cache(ggml_backend_sycl_context & ctx, ggml_tensor * dst,
                                              ggml_sycl_gated_delta_net_fused_cache cache) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/6);
    ggml_sycl_op_gated_delta_net_impl(ctx, dst, &cache);
}
