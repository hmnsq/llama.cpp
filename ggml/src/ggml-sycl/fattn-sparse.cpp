#అన్తర్భావ "fattn.hpp"
#అన్తర్భావ "fattn-sparse.hpp"

#అన్తర్భావ <cstdint>
#అన్తర్భావ <cstdio>
#అన్తర్భావ <cstring>

స్థిర constexpr int64_t SPARSE_FA_PAD       = 256;
స్థిర constexpr int64_t SPARSE_FA_MIN_నిష్పత్తి = 2;

బాహ్య పూర్ణాంకం g_ggml_sycl_enable_sparse_fa;
బాహ్య పూర్ణాంకం g_ggml_sycl_debug_sparse_fa;
బాహ్య పూర్ణాంకం g_ggml_sycl_sparse_fa_margin;

స్థిర పూర్ణాంకం sparse_fa_enabled(శూన్య) {
    తిరిగి g_ggml_sycl_enable_sparse_fa;
}

స్థిర పూర్ణాంకం sparse_fa_debug(శూన్య) {
    తిరిగి g_ggml_sycl_debug_sparse_fa;
}

// slack above n_kv_max; callers may exceed the hint by a few always-attended positions
స్థిర పూర్ణాంకం sparse_fa_margin(శూన్య) {
    తిరిగి g_ggml_sycl_sparse_fa_margin;
}

// Unordered output is fine: softmax over the selected set is permutation invariant.
స్థిర శూన్య sparse_fa_compact_ముసుగు(sycl::queue * stream,
                                   స్థిరరాశి sycl::half * __restrict__ ముసుగు,
                                   int32_t * __restrict__ indices,
                                   int32_t * __restrict__ count,
                                   స్థిరరాశి int64_t n_kv,
                                   స్థిరరాశి int64_t n_kv_g) {
    constexpr size_t WG = 256;
    స్థిరరాశి size_t global = (size_t) GGML_PAD(n_kv, (int64_t) WG);

    stream->parallel_for(
        sycl::nd_range<1>(sycl::range<1>(global), sycl::range<1>(WG)),
        [=](sycl::nd_అంశం<1> అంశం) {
            స్థిరరాశి int64_t i = (int64_t) అంశం.get_global_id(0);
            ఒకవేళ (i >= n_kv || !sycl::isfinite((దశాంశం) ముసుగు[i])) {
                తిరిగి;
            }

            sycl::atomic_ref<int32_t,
                             sycl::memory_order::relaxed,
                             sycl::జ్ఞాపకం_scope::device,
                             sycl::access::చిరునామా_space::global_space> ctr(*count);

            స్థిరరాశి int32_t pos = ctr.fetch_add(1);
            ఒకవేళ (pos < (int32_t) n_kv_g) {
                indices[pos] = (int32_t) i;
            }
        });
}

// Rows along ne[0] are contiguous for every type used as a KV cache, so this is
// a plain byte copy and needs no per-type code. Padding slots are zeroed.
స్థిర శూన్య sparse_fa_gather_అడ్డువరుసలు(sycl::queue * stream,
                                  స్థిరరాశి uint8_t * __restrict__ src,
                                  uint8_t * __restrict__ dst,
                                  స్థిరరాశి int32_t * __restrict__ indices,
                                  స్థిరరాశి int32_t * __restrict__ count,
                                  స్థిరరాశి size_t అడ్డువరుస_కొలత,
                                  స్థిరరాశి size_t src_nb1,
                                  స్థిరరాశి size_t src_nb2,
                                  స్థిరరాశి int64_t n_kv_g,
                                  స్థిరరాశి int64_t n_తల) {
    GGML_ASSERT(అడ్డువరుస_కొలత % పరిమాణం(uint32_t) == 0);
    స్థిరరాశి size_t పదాలు = అడ్డువరుస_కొలత / పరిమాణం(uint32_t);

    stream->parallel_for(
        sycl::range<3>((size_t) n_తల, (size_t) n_kv_g, పదాలు),
        [=](sycl::id<3> id) {
            స్థిరరాశి int64_t h    = (int64_t) id[0];
            స్థిరరాశి int64_t స్థానం = (int64_t) id[1];
            స్థిరరాశి size_t  w    = id[2];

            uint32_t * dst_అడ్డువరుస =
                (uint32_t *) (dst + ((size_t) (h * n_kv_g + స్థానం)) * అడ్డువరుస_కొలత);

            ఒకవేళ (స్థానం >= (int64_t) *count) {
                dst_అడ్డువరుస[w] = 0;
                తిరిగి;
            }

            స్థిరరాశి uint32_t * src_అడ్డువరుస =
                (స్థిరరాశి uint32_t *) (src + (size_t) indices[స్థానం] * src_nb1 +
                                    (size_t) h * src_nb2);
            dst_అడ్డువరుస[w] = src_అడ్డువరుస[w];
        });
}

స్థిర శూన్య sparse_fa_gather_ముసుగు(sycl::queue * stream,
                                  స్థిరరాశి sycl::half * __restrict__ ముసుగు,
                                  sycl::half * __restrict__ ముసుగు_g,
                                  స్థిరరాశి int32_t * __restrict__ indices,
                                  స్థిరరాశి int32_t * __restrict__ count,
                                  స్థిరరాశి int64_t n_kv_g,
                                  స్థిరరాశి int64_t n_అడ్డువరుసలు,
                                  స్థిరరాశి size_t ముసుగు_s1) {
    stream->parallel_for(
        sycl::range<2>((size_t) n_అడ్డువరుసలు, (size_t) n_kv_g),
        [=](sycl::id<2> id) {
            స్థిరరాశి int64_t r    = (int64_t) id[0];
            స్థిరరాశి int64_t స్థానం = (int64_t) id[1];

            sycl::half v = sycl::half(-INFINITY);
            ఒకవేళ (స్థానం < (int64_t) *count) {
                v = ముసుగు[(size_t) r * ముసుగు_s1 + (size_t) indices[స్థానం]];
            }
            ముసుగు_g[(size_t) r * n_kv_g + స్థానం] = v;
        });
}

స్థిర తార్కిక sparse_fa_applicable(స్థిరరాశి ggml_tensor * dst, int64_t & n_kv_g_బహిర్గతం) {
    స్థిరరాశి ggml_tensor * Q    = dst->src[0];
    స్థిరరాశి ggml_tensor * K    = dst->src[1];
    స్థిరరాశి ggml_tensor * V    = dst->src[2];
    స్థిరరాశి ggml_tensor * ముసుగు = dst->src[3];

    ఒకవేళ (!Q || !K || !V || !ముసుగు) {
        తిరిగి అబద్ధం;
    }

    స్థిరరాశి int32_t n_kv_max = ggml_get_op_params_i32(dst, 4);
    ఒకవేళ (n_kv_max <= 0) {
        తిరిగి అబద్ధం;
    }

    దశాంశం max_bias      = 0.0f;
    దశాంశం logit_softcap = 0.0f;
    జ్ఞాపకం_నకలు(&max_bias,      (స్థిరరాశి దశాంశం *) dst->op_params + 1, పరిమాణం(దశాంశం));
    జ్ఞాపకం_నకలు(&logit_softcap, (స్థిరరాశి దశాంశం *) dst->op_params + 2, పరిమాణం(దశాంశం));
    ఒకవేళ (max_bias != 0.0f || logit_softcap != 0.0f) {
        తిరిగి అబద్ధం;
    }

    // single-token decode only; prefill amortises the scan already
    ఒకవేళ (Q->ne[1] != 1) {
        తిరిగి అబద్ధం;
    }
    ఒకవేళ (K->ne[3] != 1 || V->ne[3] != 1 || ముసుగు->ne[2] != 1 || ముసుగు->ne[3] != 1) {
        తిరిగి అబద్ధం;
    }
    ఒకవేళ (ముసుగు->type != GGML_TYPE_F16 || ముసుగు->ne[0] < K->ne[1]) {
        తిరిగి అబద్ధం;
    }
    ఒకవేళ (K->ne[2] != V->ne[2]) {
        తిరిగి అబద్ధం;
    }

    // nb[1] may stride over heads (interleaved cache); only ne[0] must be contiguous
    ఒకవేళ (K->nb[0] != ggml_type_కొలత(K->type) || V->nb[0] != ggml_type_కొలత(V->type)) {
        తిరిగి అబద్ధం;
    }

    స్థిరరాశి size_t k_అడ్డువరుస = ggml_అడ్డువరుస_కొలత(K->type, K->ne[0]);
    స్థిరరాశి size_t v_అడ్డువరుస = ggml_అడ్డువరుస_కొలత(V->type, V->ne[0]);
    ఒకవేళ (k_అడ్డువరుస % పరిమాణం(uint32_t) || v_అడ్డువరుస % పరిమాణం(uint32_t)) {
        తిరిగి అబద్ధం;
    }

    స్థిరరాశి int64_t n_kv_g = GGML_PAD((int64_t) n_kv_max + sparse_fa_margin(), SPARSE_FA_PAD);
    ఒకవేళ (n_kv_g * SPARSE_FA_MIN_నిష్పత్తి > K->ne[1]) {
        తిరిగి అబద్ధం;
    }

    n_kv_g_బహిర్గతం = n_kv_g;
    తిరిగి నిజం;
}

తార్కిక ggml_sycl_flash_attn_ext_sparse(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    int64_t n_kv_g = 0;
    ఒకవేళ (!sparse_fa_enabled() || !sparse_fa_applicable(dst, n_kv_g)) {
        తిరిగి అబద్ధం;
    }

    ggml_tensor * K    = dst->src[1];
    ggml_tensor * V    = dst->src[2];
    ggml_tensor * ముసుగు = dst->src[3];

    స్థిరరాశి int64_t n_kv     = K->ne[1];
    స్థిరరాశి int64_t n_తల_k = K->ne[2];
    స్థిరరాశి int64_t n_అడ్డువరుసలు_m = ముసుగు->ne[1];

    స్థిరరాశి size_t k_అడ్డువరుస = ggml_అడ్డువరుస_కొలత(K->type, K->ne[0]);
    స్థిరరాశి size_t v_అడ్డువరుస = ggml_అడ్డువరుస_కొలత(V->type, V->ne[0]);

    dpct::వరుస_ptr stream = ctx.stream();

    ggml_sycl_pool_alloc<int32_t>    idx_alloc(ctx.pool(), (size_t) n_kv_g);
    ggml_sycl_pool_alloc<int32_t>    cnt_alloc(ctx.pool(), 1);
    ggml_sycl_pool_alloc<uint8_t>    k_alloc(ctx.pool(), (size_t) n_తల_k * n_kv_g * k_అడ్డువరుస);
    ggml_sycl_pool_alloc<uint8_t>    v_alloc(ctx.pool(), (size_t) n_తల_k * n_kv_g * v_అడ్డువరుస);
    ggml_sycl_pool_alloc<sycl::half> m_alloc(ctx.pool(), (size_t) n_అడ్డువరుసలు_m * n_kv_g);

    int32_t *    d_idx  = idx_alloc.get();
    int32_t *    d_cnt  = cnt_alloc.get();
    uint8_t *    d_K    = k_alloc.get();
    uint8_t *    d_V    = v_alloc.get();
    sycl::half * d_ముసుగు = m_alloc.get();

    // reset with a kernel: on Level Zero a queue memset is not always ordered before the next kernel
    stream->single_task([=]() { *d_cnt = 0; });

    sparse_fa_compact_ముసుగు(stream, (స్థిరరాశి sycl::half *) ముసుగు->దత్తాంశం,
                           d_idx, d_cnt, n_kv, n_kv_g);

    sparse_fa_gather_అడ్డువరుసలు(stream, (స్థిరరాశి uint8_t *) K->దత్తాంశం, d_K, d_idx, d_cnt,
                          k_అడ్డువరుస, K->nb[1], K->nb[2], n_kv_g, n_తల_k);

    sparse_fa_gather_అడ్డువరుసలు(stream, (స్థిరరాశి uint8_t *) V->దత్తాంశం, d_V, d_idx, d_cnt,
                          v_అడ్డువరుస, V->nb[1], V->nb[2], n_kv_g, n_తల_k);

    sparse_fa_gather_ముసుగు(stream, (స్థిరరాశి sycl::half *) ముసుగు->దత్తాంశం, d_ముసుగు,
                          d_idx, d_cnt, n_kv_g, n_అడ్డువరుసలు_m,
                          ముసుగు->nb[1] / పరిమాణం(sycl::half));

    ఒకవేళ (sparse_fa_debug()) {
        int32_t h_cnt = 0;
        SYCL_CHECK(CHECK_TRY_ERROR(stream->జ్ఞాపకం_నకలు(&h_cnt, d_cnt, పరిమాణం(int32_t))));
        SYCL_CHECK(CHECK_TRY_ERROR(stream->wait()));
        fprintf(stderr, "[FA-SPARSE] n_kv=%lld n_kv_max=%d n_kv_g=%lld finite=%d%s\n",
                (దీర్ఘ దీర్ఘ) n_kv, ggml_get_op_params_i32(dst, 4),
                (దీర్ఘ దీర్ఘ) n_kv_g, (పూర్ణాంకం) h_cnt,
                h_cnt > (int32_t) n_kv_g ? "  OVERFLOW" : "");
    }

    // shallow copies retargeted at the gathered buffers; kernels are unchanged
    ggml_tensor K_g = *K;
    K_g.దత్తాంశం      = d_K;
    K_g.ne[1]     = n_kv_g;
    K_g.nb[1]     = k_అడ్డువరుస;
    K_g.nb[2]     = (size_t) n_kv_g * k_అడ్డువరుస;
    K_g.nb[3]     = (size_t) n_తల_k * n_kv_g * k_అడ్డువరుస;
    K_g.view_src  = శూన్యసూచిక;
    K_g.view_offs = 0;

    ggml_tensor V_g = *V;
    V_g.దత్తాంశం      = d_V;
    V_g.ne[1]     = n_kv_g;
    V_g.nb[1]     = v_అడ్డువరుస;
    V_g.nb[2]     = (size_t) n_kv_g * v_అడ్డువరుస;
    V_g.nb[3]     = (size_t) V->ne[2] * n_kv_g * v_అడ్డువరుస;
    V_g.view_src  = శూన్యసూచిక;
    V_g.view_offs = 0;

    ggml_tensor M_g = *ముసుగు;
    M_g.దత్తాంశం      = d_ముసుగు;
    M_g.ne[0]     = n_kv_g;
    M_g.nb[1]     = (size_t) n_kv_g * పరిమాణం(sycl::half);
    M_g.nb[2]     = M_g.nb[1] * ముసుగు->ne[1];
    M_g.nb[3]     = M_g.nb[2];
    M_g.view_src  = శూన్యసూచిక;
    M_g.view_offs = 0;

    ggml_tensor dst_g = *dst;
    dst_g.src[1] = &K_g;
    dst_g.src[2] = &V_g;
    dst_g.src[3] = &M_g;
    dst_g.op_params[4] = 0;   // avoid re-entering this path

    ggml_sycl_flash_attn_ext(ctx, &dst_g);

    తిరిగి నిజం;
}
