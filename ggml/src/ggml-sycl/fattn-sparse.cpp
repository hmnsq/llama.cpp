#ਅਨ੍ਤਰ੍ਭਾਵ "fattn.hpp"
#ਅਨ੍ਤਰ੍ਭਾਵ "fattn-sparse.hpp"

#ਅਨ੍ਤਰ੍ਭਾਵ <cstdint>
#ਅਨ੍ਤਰ੍ਭਾਵ <cstdio>
#ਅਨ੍ਤਰ੍ਭਾਵ <cstring>

ਸਥਿਰ constexpr int64_t SPARSE_FA_PAD       = 256;
ਸਥਿਰ constexpr int64_t SPARSE_FA_MIN_ਅਨੁਪਾਤ = 2;

ਬਾਹਰੀ ਪੂਰਨ g_ggml_sycl_enable_sparse_fa;
ਬਾਹਰੀ ਪੂਰਨ g_ggml_sycl_debug_sparse_fa;
ਬਾਹਰੀ ਪੂਰਨ g_ggml_sycl_sparse_fa_margin;

ਸਥਿਰ ਪੂਰਨ sparse_fa_enabled(ਖਾਲੀ) {
    ਵਾਪਸ g_ggml_sycl_enable_sparse_fa;
}

ਸਥਿਰ ਪੂਰਨ sparse_fa_debug(ਖਾਲੀ) {
    ਵਾਪਸ g_ggml_sycl_debug_sparse_fa;
}

// slack above n_kv_max; callers may exceed the hint by a few always-attended positions
ਸਥਿਰ ਪੂਰਨ sparse_fa_margin(ਖਾਲੀ) {
    ਵਾਪਸ g_ggml_sycl_sparse_fa_margin;
}

// Unordered output is fine: softmax over the selected set is permutation invariant.
ਸਥਿਰ ਖਾਲੀ sparse_fa_compact_ਨਕਾਬ(sycl::queue * stream,
                                   ਅਚਲ sycl::half * __restrict__ ਨਕਾਬ,
                                   int32_t * __restrict__ indices,
                                   int32_t * __restrict__ count,
                                   ਅਚਲ int64_t n_kv,
                                   ਅਚਲ int64_t n_kv_g) {
    constexpr size_t WG = 256;
    ਅਚਲ size_t global = (size_t) GGML_PAD(n_kv, (int64_t) WG);

    stream->parallel_for(
        sycl::nd_range<1>(sycl::range<1>(global), sycl::range<1>(WG)),
        [=](sycl::nd_ਵਸਤੂ<1> ਵਸਤੂ) {
            ਅਚਲ int64_t i = (int64_t) ਵਸਤੂ.get_global_id(0);
            ਜੇ (i >= n_kv || !sycl::isfinite((ਦਸ਼ਮਲਵ) ਨਕਾਬ[i])) {
                ਵਾਪਸ;
            }

            sycl::atomic_ref<int32_t,
                             sycl::memory_order::relaxed,
                             sycl::ਸਮ੍ਰਿਤੀ_scope::device,
                             sycl::access::ਪਤਾ_space::global_space> ctr(*count);

            ਅਚਲ int32_t pos = ctr.fetch_add(1);
            ਜੇ (pos < (int32_t) n_kv_g) {
                indices[pos] = (int32_t) i;
            }
        });
}

// Rows along ne[0] are contiguous for every type used as a KV cache, so this is
// a plain byte copy and needs no per-type code. Padding slots are zeroed.
ਸਥਿਰ ਖਾਲੀ sparse_fa_gather_ਪੰਕਤੀਆਂ(sycl::queue * stream,
                                  ਅਚਲ uint8_t * __restrict__ src,
                                  uint8_t * __restrict__ dst,
                                  ਅਚਲ int32_t * __restrict__ indices,
                                  ਅਚਲ int32_t * __restrict__ count,
                                  ਅਚਲ size_t ਪੰਕਤੀ_ਮਾਪ,
                                  ਅਚਲ size_t src_nb1,
                                  ਅਚਲ size_t src_nb2,
                                  ਅਚਲ int64_t n_kv_g,
                                  ਅਚਲ int64_t n_ਸਿਰ) {
    GGML_ASSERT(ਪੰਕਤੀ_ਮਾਪ % ਆਕਾਰ(uint32_t) == 0);
    ਅਚਲ size_t ਸ਼ਬਦ = ਪੰਕਤੀ_ਮਾਪ / ਆਕਾਰ(uint32_t);

    stream->parallel_for(
        sycl::range<3>((size_t) n_ਸਿਰ, (size_t) n_kv_g, ਸ਼ਬਦ),
        [=](sycl::id<3> id) {
            ਅਚਲ int64_t h    = (int64_t) id[0];
            ਅਚਲ int64_t ਥਾਂ = (int64_t) id[1];
            ਅਚਲ size_t  w    = id[2];

            uint32_t * dst_ਪੰਕਤੀ =
                (uint32_t *) (dst + ((size_t) (h * n_kv_g + ਥਾਂ)) * ਪੰਕਤੀ_ਮਾਪ);

            ਜੇ (ਥਾਂ >= (int64_t) *count) {
                dst_ਪੰਕਤੀ[w] = 0;
                ਵਾਪਸ;
            }

            ਅਚਲ uint32_t * src_ਪੰਕਤੀ =
                (ਅਚਲ uint32_t *) (src + (size_t) indices[ਥਾਂ] * src_nb1 +
                                    (size_t) h * src_nb2);
            dst_ਪੰਕਤੀ[w] = src_ਪੰਕਤੀ[w];
        });
}

ਸਥਿਰ ਖਾਲੀ sparse_fa_gather_ਨਕਾਬ(sycl::queue * stream,
                                  ਅਚਲ sycl::half * __restrict__ ਨਕਾਬ,
                                  sycl::half * __restrict__ ਨਕਾਬ_g,
                                  ਅਚਲ int32_t * __restrict__ indices,
                                  ਅਚਲ int32_t * __restrict__ count,
                                  ਅਚਲ int64_t n_kv_g,
                                  ਅਚਲ int64_t n_ਪੰਕਤੀਆਂ,
                                  ਅਚਲ size_t ਨਕਾਬ_s1) {
    stream->parallel_for(
        sycl::range<2>((size_t) n_ਪੰਕਤੀਆਂ, (size_t) n_kv_g),
        [=](sycl::id<2> id) {
            ਅਚਲ int64_t r    = (int64_t) id[0];
            ਅਚਲ int64_t ਥਾਂ = (int64_t) id[1];

            sycl::half v = sycl::half(-INFINITY);
            ਜੇ (ਥਾਂ < (int64_t) *count) {
                v = ਨਕਾਬ[(size_t) r * ਨਕਾਬ_s1 + (size_t) indices[ਥਾਂ]];
            }
            ਨਕਾਬ_g[(size_t) r * n_kv_g + ਥਾਂ] = v;
        });
}

ਸਥਿਰ ਤਾਰਕਿਕ sparse_fa_applicable(ਅਚਲ ggml_tensor * dst, int64_t & n_kv_g_ਬਾਹਰਲਾ) {
    ਅਚਲ ggml_tensor * Q    = dst->src[0];
    ਅਚਲ ggml_tensor * K    = dst->src[1];
    ਅਚਲ ggml_tensor * V    = dst->src[2];
    ਅਚਲ ggml_tensor * ਨਕਾਬ = dst->src[3];

    ਜੇ (!Q || !K || !V || !ਨਕਾਬ) {
        ਵਾਪਸ ਝੂਠ;
    }

    ਅਚਲ int32_t n_kv_max = ggml_get_op_params_i32(dst, 4);
    ਜੇ (n_kv_max <= 0) {
        ਵਾਪਸ ਝੂਠ;
    }

    ਦਸ਼ਮਲਵ max_bias      = 0.0f;
    ਦਸ਼ਮਲਵ logit_softcap = 0.0f;
    ਸਮ੍ਰਿਤੀ_ਨਕਲ(&max_bias,      (ਅਚਲ ਦਸ਼ਮਲਵ *) dst->op_params + 1, ਆਕਾਰ(ਦਸ਼ਮਲਵ));
    ਸਮ੍ਰਿਤੀ_ਨਕਲ(&logit_softcap, (ਅਚਲ ਦਸ਼ਮਲਵ *) dst->op_params + 2, ਆਕਾਰ(ਦਸ਼ਮਲਵ));
    ਜੇ (max_bias != 0.0f || logit_softcap != 0.0f) {
        ਵਾਪਸ ਝੂਠ;
    }

    // single-token decode only; prefill amortises the scan already
    ਜੇ (Q->ne[1] != 1) {
        ਵਾਪਸ ਝੂਠ;
    }
    ਜੇ (K->ne[3] != 1 || V->ne[3] != 1 || ਨਕਾਬ->ne[2] != 1 || ਨਕਾਬ->ne[3] != 1) {
        ਵਾਪਸ ਝੂਠ;
    }
    ਜੇ (ਨਕਾਬ->type != GGML_TYPE_F16 || ਨਕਾਬ->ne[0] < K->ne[1]) {
        ਵਾਪਸ ਝੂਠ;
    }
    ਜੇ (K->ne[2] != V->ne[2]) {
        ਵਾਪਸ ਝੂਠ;
    }

    // nb[1] may stride over heads (interleaved cache); only ne[0] must be contiguous
    ਜੇ (K->nb[0] != ggml_type_ਮਾਪ(K->type) || V->nb[0] != ggml_type_ਮਾਪ(V->type)) {
        ਵਾਪਸ ਝੂਠ;
    }

    ਅਚਲ size_t k_ਪੰਕਤੀ = ggml_ਪੰਕਤੀ_ਮਾਪ(K->type, K->ne[0]);
    ਅਚਲ size_t v_ਪੰਕਤੀ = ggml_ਪੰਕਤੀ_ਮਾਪ(V->type, V->ne[0]);
    ਜੇ (k_ਪੰਕਤੀ % ਆਕਾਰ(uint32_t) || v_ਪੰਕਤੀ % ਆਕਾਰ(uint32_t)) {
        ਵਾਪਸ ਝੂਠ;
    }

    ਅਚਲ int64_t n_kv_g = GGML_PAD((int64_t) n_kv_max + sparse_fa_margin(), SPARSE_FA_PAD);
    ਜੇ (n_kv_g * SPARSE_FA_MIN_ਅਨੁਪਾਤ > K->ne[1]) {
        ਵਾਪਸ ਝੂਠ;
    }

    n_kv_g_ਬਾਹਰਲਾ = n_kv_g;
    ਵਾਪਸ ਸੱਚ;
}

ਤਾਰਕਿਕ ggml_sycl_flash_attn_ext_sparse(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    int64_t n_kv_g = 0;
    ਜੇ (!sparse_fa_enabled() || !sparse_fa_applicable(dst, n_kv_g)) {
        ਵਾਪਸ ਝੂਠ;
    }

    ggml_tensor * K    = dst->src[1];
    ggml_tensor * V    = dst->src[2];
    ggml_tensor * ਨਕਾਬ = dst->src[3];

    ਅਚਲ int64_t n_kv     = K->ne[1];
    ਅਚਲ int64_t n_ਸਿਰ_k = K->ne[2];
    ਅਚਲ int64_t n_ਪੰਕਤੀਆਂ_m = ਨਕਾਬ->ne[1];

    ਅਚਲ size_t k_ਪੰਕਤੀ = ggml_ਪੰਕਤੀ_ਮਾਪ(K->type, K->ne[0]);
    ਅਚਲ size_t v_ਪੰਕਤੀ = ggml_ਪੰਕਤੀ_ਮਾਪ(V->type, V->ne[0]);

    dpct::ਕਤਾਰ_ptr stream = ctx.stream();

    ggml_sycl_pool_alloc<int32_t>    idx_alloc(ctx.pool(), (size_t) n_kv_g);
    ggml_sycl_pool_alloc<int32_t>    cnt_alloc(ctx.pool(), 1);
    ggml_sycl_pool_alloc<uint8_t>    k_alloc(ctx.pool(), (size_t) n_ਸਿਰ_k * n_kv_g * k_ਪੰਕਤੀ);
    ggml_sycl_pool_alloc<uint8_t>    v_alloc(ctx.pool(), (size_t) n_ਸਿਰ_k * n_kv_g * v_ਪੰਕਤੀ);
    ggml_sycl_pool_alloc<sycl::half> m_alloc(ctx.pool(), (size_t) n_ਪੰਕਤੀਆਂ_m * n_kv_g);

    int32_t *    d_idx  = idx_alloc.get();
    int32_t *    d_cnt  = cnt_alloc.get();
    uint8_t *    d_K    = k_alloc.get();
    uint8_t *    d_V    = v_alloc.get();
    sycl::half * d_ਨਕਾਬ = m_alloc.get();

    // reset with a kernel: on Level Zero a queue memset is not always ordered before the next kernel
    stream->single_task([=]() { *d_cnt = 0; });

    sparse_fa_compact_ਨਕਾਬ(stream, (ਅਚਲ sycl::half *) ਨਕਾਬ->ਅੰਕੜੇ,
                           d_idx, d_cnt, n_kv, n_kv_g);

    sparse_fa_gather_ਪੰਕਤੀਆਂ(stream, (ਅਚਲ uint8_t *) K->ਅੰਕੜੇ, d_K, d_idx, d_cnt,
                          k_ਪੰਕਤੀ, K->nb[1], K->nb[2], n_kv_g, n_ਸਿਰ_k);

    sparse_fa_gather_ਪੰਕਤੀਆਂ(stream, (ਅਚਲ uint8_t *) V->ਅੰਕੜੇ, d_V, d_idx, d_cnt,
                          v_ਪੰਕਤੀ, V->nb[1], V->nb[2], n_kv_g, n_ਸਿਰ_k);

    sparse_fa_gather_ਨਕਾਬ(stream, (ਅਚਲ sycl::half *) ਨਕਾਬ->ਅੰਕੜੇ, d_ਨਕਾਬ,
                          d_idx, d_cnt, n_kv_g, n_ਪੰਕਤੀਆਂ_m,
                          ਨਕਾਬ->nb[1] / ਆਕਾਰ(sycl::half));

    ਜੇ (sparse_fa_debug()) {
        int32_t h_cnt = 0;
        SYCL_CHECK(CHECK_TRY_ERROR(stream->ਸਮ੍ਰਿਤੀ_ਨਕਲ(&h_cnt, d_cnt, ਆਕਾਰ(int32_t))));
        SYCL_CHECK(CHECK_TRY_ERROR(stream->wait()));
        fprintf(stderr, "[FA-SPARSE] n_kv=%lld n_kv_max=%d n_kv_g=%lld finite=%d%s\n",
                (ਲੰਮਾ ਲੰਮਾ) n_kv, ggml_get_op_params_i32(dst, 4),
                (ਲੰਮਾ ਲੰਮਾ) n_kv_g, (ਪੂਰਨ) h_cnt,
                h_cnt > (int32_t) n_kv_g ? "  OVERFLOW" : "");
    }

    // shallow copies retargeted at the gathered buffers; kernels are unchanged
    ggml_tensor K_g = *K;
    K_g.ਅੰਕੜੇ      = d_K;
    K_g.ne[1]     = n_kv_g;
    K_g.nb[1]     = k_ਪੰਕਤੀ;
    K_g.nb[2]     = (size_t) n_kv_g * k_ਪੰਕਤੀ;
    K_g.nb[3]     = (size_t) n_ਸਿਰ_k * n_kv_g * k_ਪੰਕਤੀ;
    K_g.view_src  = ਖਾਲੀ_ਸੂਚਕ;
    K_g.view_offs = 0;

    ggml_tensor V_g = *V;
    V_g.ਅੰਕੜੇ      = d_V;
    V_g.ne[1]     = n_kv_g;
    V_g.nb[1]     = v_ਪੰਕਤੀ;
    V_g.nb[2]     = (size_t) n_kv_g * v_ਪੰਕਤੀ;
    V_g.nb[3]     = (size_t) V->ne[2] * n_kv_g * v_ਪੰਕਤੀ;
    V_g.view_src  = ਖਾਲੀ_ਸੂਚਕ;
    V_g.view_offs = 0;

    ggml_tensor M_g = *ਨਕਾਬ;
    M_g.ਅੰਕੜੇ      = d_ਨਕਾਬ;
    M_g.ne[0]     = n_kv_g;
    M_g.nb[1]     = (size_t) n_kv_g * ਆਕਾਰ(sycl::half);
    M_g.nb[2]     = M_g.nb[1] * ਨਕਾਬ->ne[1];
    M_g.nb[3]     = M_g.nb[2];
    M_g.view_src  = ਖਾਲੀ_ਸੂਚਕ;
    M_g.view_offs = 0;

    ggml_tensor dst_g = *dst;
    dst_g.src[1] = &K_g;
    dst_g.src[2] = &V_g;
    dst_g.src[3] = &M_g;
    dst_g.op_params[4] = 0;   // avoid re-entering this path

    ggml_sycl_flash_attn_ext(ctx, &dst_g);

    ਵਾਪਸ ਸੱਚ;
}
