#అన్తర్భావ <sycl/sycl.hpp>
#అన్తర్భావ "dpct/helper.hpp"
#అన్తర్భావ "common.hpp"
#అన్తర్భావ "ggml.h"
#అన్తర్భావ "gated_delta_net.hpp"
#అన్తర్భావ <cmath>


మూస <పూర్ణాంకం S_v, తార్కిక KDA, తార్కిక keep_rs_t>
శూన్య gated_delta_net_sycl(స్థిరరాశి దశాంశం *     q,
                          స్థిరరాశి దశాంశం *     k,
                          స్థిరరాశి దశాంశం *     v,
                          స్థిరరాశి దశాంశం *     g,
                          స్థిరరాశి దశాంశం *     beta,
                          స్థిరరాశి దశాంశం *     curr_state,
                          దశాంశం *           dst,
                          దశాంశం *           state,
                          int64_t           H,
                          int64_t           n_టోకెన్లు,
                          int64_t           sq1,
                          int64_t           sq2,
                          int64_t           sq3,
                          int64_t           sv1,
                          int64_t           sv2,
                          int64_t           sv3,
                          int64_t           sb1,
                          int64_t           sb2,
                          int64_t           sb3,
                          స్థిరరాశి sycl::uint3 neqk1_magic,
                          స్థిరరాశి sycl::uint3 rq3_magic,
                          దశాంశం             scale,
                          int64_t           state_స్థానం_stride,
                          పూర్ణాంకం               K) {
    స్వయంచాలక           అంశం_ct1 = sycl::ext::oneapi::this_work_అంశం::get_nd_అంశం<3>();
    స్థిరరాశి uint32_t h_idx    = అంశం_ct1.get_సమూహం(2);
    స్థిరరాశి uint32_t sequence = అంశం_ct1.get_సమూహం(1);
    // each warp owns one column, using warp-level primitives to reduce across rows
    స్థిరరాశి పూర్ణాంకం      lane     = అంశం_ct1.get_local_id(2);
    స్థిరరాశి పూర్ణాంకం      నిలువు      = అంశం_ct1.get_సమూహం(0) * అంశం_ct1.get_local_range(1) + అంశం_ct1.get_local_id(1);

    స్థిరరాశి uint32_t iq1 = fastmodulo(h_idx, neqk1_magic);
    స్థిరరాశి uint32_t iq3 = fastdiv(sequence, rq3_magic);

    దశాంశం *       attn_దత్తాంశం        = dst;

    // input state holds s0 only [S_v, S_v, H, n_seqs] — seq stride is D = H * S_v * S_v.
    // output state layout (per-slot D * n_seqs) — same per-(seq,head) offset as before.
    స్థిరరాశి int64_t state_లోపల_స్థానభ్రంశం      = sequence * H * S_v * S_v + h_idx * S_v * S_v;
    స్థిరరాశి int64_t state_బహిర్గతం_స్థానభ్రంశం     = (sequence * H + h_idx) * S_v * S_v;
    state += state_బహిర్గతం_స్థానభ్రంశం;
    curr_state += state_లోపల_స్థానభ్రంశం + నిలువు * S_v;
    attn_దత్తాంశం += (sequence * n_టోకెన్లు * H + h_idx) * S_v;

    constexpr పూర్ణాంకం warp_కొలత = ggml_sycl_get_physical_warp_కొలత() < S_v ? ggml_sycl_get_physical_warp_కొలత() : S_v;
    static_assert(S_v % warp_కొలత == 0, "S_v must be a multiple of warp_size");
    constexpr పూర్ణాంకం అడ్డువరుసలు_చొప్పున_lane = (S_v + warp_కొలత - 1) / warp_కొలత;
    దశాంశం         s_shard[అడ్డువరుసలు_చొప్పున_lane];
#సూచన unroll
    కొరకు (పూర్ణాంకం r = 0; r < అడ్డువరుసలు_చొప్పున_lane; r++) {
        స్థిరరాశి పూర్ణాంకం i = r * warp_కొలత + lane;
        s_shard[r]  = curr_state[i];
    }

    // snapshot slot mapping: slot 0 = most recent state, slot s = s tokens back.
    // When n_tokens < K only slots 0..n_tokens-1 are written; older slots are caller-owned.

    కొరకు (పూర్ణాంకం t = 0; t < n_టోకెన్లు; t++) {
        స్థిరరాశి దశాంశం * q_t = q + iq3 * sq3 + t * sq2 + iq1 * sq1;
        స్థిరరాశి దశాంశం * k_t = k + iq3 * sq3 + t * sq2 + iq1 * sq1;
        స్థిరరాశి దశాంశం * v_t = v + sequence * sv3 + t * sv2 + h_idx * sv1;

        స్థిరరాశి int64_t gb_స్థానభ్రంశం = sequence * sb3 + t * sb2 + h_idx * sb1;
        స్థిరరాశి దశాంశం * beta_t = beta + gb_స్థానభ్రంశం;
        స్థిరరాశి దశాంశం * g_t    = g    + gb_స్థానభ్రంశం * (KDA ? S_v : 1);

        స్థిరరాశి దశాంశం beta_val = *beta_t;

        ఒకవేళ constexpr (!KDA) {
            స్థిరరాశి దశాంశం g_val = sycl::native::ఘాతాంకం(*g_t);

            // kv[col] = (S^T @ k)[col] = sum_i S[i][col] * k[i]
            దశాంశం kv_shard = 0.0f;
#సూచన unroll
            కొరకు (పూర్ణాంకం r = 0; r < అడ్డువరుసలు_చొప్పున_lane; r++) {
                స్థిరరాశి పూర్ణాంకం i = r * warp_కొలత + lane;
                kv_shard += s_shard[r] * k_t[i];
            }
            దశాంశం kv_నిలువు = warp_reduce_కూడిక<warp_కొలత>(kv_shard);

            // delta[col] = (v[col] - g * kv[col]) * beta
            దశాంశం delta_నిలువు = (v_t[నిలువు] - g_val * kv_నిలువు) * beta_val;

            // fused: S[i][col] = g * S[i][col] + k[i] * delta[col]
            // attn[col] = (S^T @ q)[col] = sum_i S[i][col] * q[i]
            దశాంశం attn_partial = 0.0f;
#సూచన unroll
            కొరకు (పూర్ణాంకం r = 0; r < అడ్డువరుసలు_చొప్పున_lane; r++) {
                స్థిరరాశి పూర్ణాంకం i = r * warp_కొలత + lane;
                s_shard[r]  = g_val * s_shard[r] + k_t[i] * delta_నిలువు;
                attn_partial += s_shard[r] * q_t[i];
            }

            దశాంశం attn_నిలువు = warp_reduce_కూడిక<warp_కొలత>(attn_partial);

            ఒకవేళ (lane == 0) {
                attn_దత్తాంశం[నిలువు] = attn_నిలువు * scale;
            }
        } లేకపోతే {
            // kv[col] = sum_i g[i] * S[i][col] * k[i]
            దశాంశం kv_shard = 0.0f;
#సూచన unroll
            కొరకు (పూర్ణాంకం r = 0; r < అడ్డువరుసలు_చొప్పున_lane; r++) {
                స్థిరరాశి పూర్ణాంకం i = r * warp_కొలత + lane;
                kv_shard += sycl::native::ఘాతాంకం(g_t[i]) * s_shard[r] * k_t[i];
            }

            దశాంశం kv_నిలువు = warp_reduce_కూడిక<warp_కొలత>(kv_shard);

            // delta[col] = (v[col] - kv[col]) * beta
            దశాంశం delta_నిలువు = (v_t[నిలువు] - kv_నిలువు) * beta_val;

            // fused: S[i][col] = g[i] * S[i][col] + k[i] * delta[col]
            // attn[col] = (S^T @ q)[col] = sum_i S[i][col] * q[i]
            దశాంశం attn_partial = 0.0f;
#సూచన unroll
            కొరకు (పూర్ణాంకం r = 0; r < అడ్డువరుసలు_చొప్పున_lane; r++) {
                స్థిరరాశి పూర్ణాంకం i = r * warp_కొలత + lane;
                s_shard[r]  = sycl::native::ఘాతాంకం(g_t[i]) * s_shard[r] + k_t[i] * delta_నిలువు;
                attn_partial += s_shard[r] * q_t[i];
            }

            దశాంశం attn_నిలువు = warp_reduce_కూడిక<warp_కొలత>(attn_partial);

            ఒకవేళ (lane == 0) {
                attn_దత్తాంశం[నిలువు] = attn_నిలువు * scale;
            }
        }

        attn_దత్తాంశం += S_v * H;


    // Write state back to global memory
        ఒకవేళ constexpr (keep_rs_t) {
            స్థిరరాశి పూర్ణాంకం లక్ష్యం_స్థానం = (పూర్ణాంకం) n_టోకెన్లు - 1 - t;
            ఒకవేళ (లక్ష్యం_స్థానం >= 0 && లక్ష్యం_స్థానం < K) {
                దశాంశం * curr_state = state + లక్ష్యం_స్థానం * state_స్థానం_stride;
#సూచన unroll
                కొరకు (పూర్ణాంకం r = 0; r < అడ్డువరుసలు_చొప్పున_lane; r++) {
                    స్థిరరాశి పూర్ణాంకం i = r * warp_కొలత + lane;
                    curr_state[నిలువు * S_v + i] = s_shard[r];
                }
            }
        }
    }

    ఒకవేళ constexpr (!keep_rs_t) {
#సూచన unroll
        కొరకు (పూర్ణాంకం r = 0; r < అడ్డువరుసలు_చొప్పున_lane; r++) {
            స్థిరరాశి పూర్ణాంకం i          = r * warp_కొలత + lane;
            state[నిలువు * S_v + i] = s_shard[r];
        }
    }
}

మూస <తార్కిక KDA, తార్కిక keep_rs_t>
స్థిర శూన్య launch_gated_delta_net(స్థిరరాశి దశాంశం *   q_d,
                                   స్థిరరాశి దశాంశం *   k_d,
                                   స్థిరరాశి దశాంశం *   v_d,
                                   స్థిరరాశి దశాంశం *   g_d,
                                   స్థిరరాశి దశాంశం *   b_d,
                                   స్థిరరాశి దశాంశం *   s_d,
                                   దశాంశం *         dst_d,
                                   దశాంశం *         state_d,
                                   int64_t         S_v,
                                   int64_t         H,
                                   int64_t         n_టోకెన్లు,
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
                                   దశాంశం           scale,
                                   int64_t         state_స్థానం_stride,
                                   పూర్ణాంకం             K,
                                   dpct::వరుస_ptr stream) {
    //TODO: Add chunked kernel for even faster pre-fill
    స్థిరరాశి పూర్ణాంకం warp_కొలత = ggml_sycl_info().devices[ggml_sycl_get_device()].warp_కొలత;

    స్థిరరాశి పూర్ణాంకం అంకె_warps = 4;
    dpct::dim3 grid_dims(H, n_seqs, (S_v + అంకె_warps - 1) / అంకె_warps);
    dpct::dim3 ఖండం_dims(warp_కొలత <= S_v ? warp_కొలత : S_v, అంకె_warps, 1);

    స్థిరరాశి sycl::uint3 neqk1_magic = init_fastdiv_విలువలు(neqk1);
    స్థిరరాశి sycl::uint3 rq3_magic   = init_fastdiv_విలువలు(rq3);

    ఎంపిక (S_v) {
        స్థితి 16:
            {
                constexpr పూర్ణాంకం sv = 16;
                stream->parallel_for(sycl::nd_range<3>(grid_dims * ఖండం_dims, ఖండం_dims),
                                     [=](sycl::nd_అంశం<3> /*item_ct1*/) [[sycl::reqd_sub_సమూహం_కొలత(WARP_కొలత)]] {
                                         gated_delta_net_sycl<sv, KDA, keep_rs_t>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H, n_టోకెన్లు,
                                                                       sq1, sq2, sq3, sv1, sv2, sv3, sb1, sb2,
                                                                       sb3, neqk1_magic, rq3_magic, scale, state_స్థానం_stride, K);
                                     });
            }
            ఆపు;
        స్థితి 32:
            {
                constexpr పూర్ణాంకం sv = 32;
                stream->parallel_for(sycl::nd_range<3>(grid_dims * ఖండం_dims, ఖండం_dims),
                                     [=](sycl::nd_అంశం<3> /*item_ct1*/) [[sycl::reqd_sub_సమూహం_కొలత(WARP_కొలత)]] {
                                         gated_delta_net_sycl<sv, KDA, keep_rs_t>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H, n_టోకెన్లు,
                                                                       sq1, sq2, sq3, sv1, sv2, sv3, sb1, sb2,
                                                                       sb3, neqk1_magic, rq3_magic, scale, state_స్థానం_stride, K);
                                     });
            }
            ఆపు;
        స్థితి 64: {
            {
                constexpr పూర్ణాంకం sv = 64;
                stream->parallel_for(sycl::nd_range<3>(grid_dims * ఖండం_dims, ఖండం_dims),
                                        [=](sycl::nd_అంశం<3> /*item_ct1*/) [[sycl::reqd_sub_సమూహం_కొలత(WARP_కొలత)]] {
                                            gated_delta_net_sycl<sv, KDA, keep_rs_t>(
                                                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H, n_టోకెన్లు, sq1, sq2,
                                                sq3, sv1, sv2, sv3, sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_స్థానం_stride, K);
                                        });
            }
            ఆపు;
        }
        స్థితి 128: {
            {
                constexpr పూర్ణాంకం sv = 128;
                stream->parallel_for(sycl::nd_range<3>(grid_dims * ఖండం_dims, ఖండం_dims),
                                        [=](sycl::nd_అంశం<3> /*item_ct1*/) [[sycl::reqd_sub_సమూహం_కొలత(WARP_కొలత)]] {
                                            gated_delta_net_sycl<sv, KDA, keep_rs_t>(
                                                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H, n_టోకెన్లు, sq1, sq2,
                                                sq3, sv1, sv2, sv3, sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_స్థానం_stride, K);
                                        });
            }
            ఆపు;
        }
        అప్రమేయ:
            GGML_ABORT("fatal error");
            ఆపు;
    }
}

స్థిర శూన్య ggml_sycl_op_gated_delta_net_impl(ggml_backend_sycl_context & ctx, ggml_tensor * dst,
                                              స్థిరరాశి ggml_sycl_gated_delta_net_fused_cache * cache) {
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

    స్థిరరాశి int64_t S_v      = nev0;
    స్థిరరాశి int64_t H        = nev1;
    స్థిరరాశి int64_t n_టోకెన్లు = nev2;
    స్థిరరాశి int64_t n_seqs   = nev3;

    స్థిరరాశి తార్కిక kda = (src_g->ne[0] == S_v);

    GGML_ASSERT(neq1 == nek1);
    స్థిరరాశి int64_t neqk1 = neq1;

    స్థిరరాశి int64_t rq3 = nev3 / neq3;

    స్థిరరాశి దశాంశం * q_d = (స్థిరరాశి దశాంశం *) src_q->దత్తాంశం;
    స్థిరరాశి దశాంశం * k_d = (స్థిరరాశి దశాంశం *) src_k->దత్తాంశం;
    స్థిరరాశి దశాంశం * v_d = (స్థిరరాశి దశాంశం *) src_v->దత్తాంశం;
    స్థిరరాశి దశాంశం * g_d = (స్థిరరాశి దశాంశం *) src_g->దత్తాంశం;
    స్థిరరాశి దశాంశం * b_d = (స్థిరరాశి దశాంశం *) src_beta->దత్తాంశం;

    స్థిరరాశి దశాంశం * s_d   = (స్థిరరాశి దశాంశం *) src_state->దత్తాంశం;
    దశాంశం *       dst_d = (దశాంశం *) dst->దత్తాంశం;

    GGML_ASSERT(ggml_is_contiguous_అడ్డువరుసలు(src_q));
    GGML_ASSERT(ggml_is_contiguous_అడ్డువరుసలు(src_k));
    GGML_ASSERT(ggml_is_contiguous_అడ్డువరుసలు(src_v));
    GGML_ASSERT(ggml_are_same_stride(src_q, src_k));
    GGML_ASSERT(src_g->ne[0] == 1 || kda);
    GGML_ASSERT(ggml_is_contiguous(src_g));
    GGML_ASSERT(ggml_is_contiguous(src_beta));
    GGML_ASSERT(ggml_is_contiguous(src_state));

    // strides in floats (beta strides used for both g and beta offset computation)
    స్థిరరాశి int64_t sq1 = nbq1 / పరిమాణం(దశాంశం);
    స్థిరరాశి int64_t sq2 = nbq2 / పరిమాణం(దశాంశం);
    స్థిరరాశి int64_t sq3 = nbq3 / పరిమాణం(దశాంశం);
    స్థిరరాశి int64_t sv1 = nbv1 / పరిమాణం(దశాంశం);
    స్థిరరాశి int64_t sv2 = nbv2 / పరిమాణం(దశాంశం);
    స్థిరరాశి int64_t sv3 = nbv3 / పరిమాణం(దశాంశం);
    స్థిరరాశి int64_t sb1 = nbb1 / పరిమాణం(దశాంశం);
    స్థిరరాశి int64_t sb2 = nbb2 / పరిమాణం(దశాంశం);
    స్థిరరాశి int64_t sb3 = nbb3 / పరిమాణం(దశాంశం);

    స్థిరరాశి దశాంశం scale = 1.0f / sqrtf((దశాంశం) S_v);

    dpct::వరుస_ptr stream = ctx.stream();

    // K (snapshot slot count) is an op param; state holds s0 only [S_v, S_v, H, n_seqs].
    స్థిరరాశి పూర్ణాంకం K = ggml_get_op_params_i32(dst, 0);
    స్థిరరాశి తార్కిక keep_rs = K > 1;

    // recurrent state -> dst tail (after attention scores), or the cache when fusing
    దశాంశం * state_d           = dst_d + S_v * H * n_టోకెన్లు * n_seqs;
    int64_t state_స్థానం_stride = S_v * S_v * H * n_seqs;
    ఒకవేళ (cache != శూన్యసూచిక) {
        state_d           = cache->దత్తాంశం;
        state_స్థానం_stride = cache->స్థానం_stride;
    }

    ఒకవేళ (kda) {
        ఒకవేళ (keep_rs) {
            launch_gated_delta_net<నిజం, నిజం>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_టోకెన్లు, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_స్థానం_stride, K, stream);
        } లేకపోతే {
            launch_gated_delta_net<నిజం, అబద్ధం>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_టోకెన్లు, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_స్థానం_stride, K, stream);
        }
    } లేకపోతే {
        ఒకవేళ (keep_rs) {
            launch_gated_delta_net<అబద్ధం, నిజం>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_టోకెన్లు, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_స్థానం_stride, K, stream);
        } లేకపోతే {
            launch_gated_delta_net<అబద్ధం, అబద్ధం>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_టోకెన్లు, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_స్థానం_stride, K, stream);
        }
    }
}

శూన్య ggml_sycl_op_gated_delta_net(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_op_gated_delta_net_impl(ctx, dst, శూన్యసూచిక);
}

శూన్య ggml_sycl_gated_delta_net(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/6);
    ggml_sycl_op_gated_delta_net(ctx, dst);
}

శూన్య ggml_sycl_op_gated_delta_net_fused_cache(ggml_backend_sycl_context & ctx, ggml_tensor * dst,
                                              ggml_sycl_gated_delta_net_fused_cache cache) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/6);
    ggml_sycl_op_gated_delta_net_impl(ctx, dst, &cache);
}
