#ಅನ್ತರ್ಭಾವ "fattn.hpp"
#ಅನ್ತರ್ಭಾವ "fattn-sparse.hpp"

#ಅನ್ತರ್ಭಾವ <cstdint>
#ಅನ್ತರ್ಭಾವ <cstdio>
#ಅನ್ತರ್ಭಾವ <cstring>

ಸ್ಥಿರ constexpr int64_t SPARSE_FA_PAD       = 256;
ಸ್ಥಿರ constexpr int64_t SPARSE_FA_MIN_ಅನುಪಾತ = 2;

ಬಾಹ್ಯ ಪೂರ್ಣಾಂಕ g_ggml_sycl_enable_sparse_fa;
ಬಾಹ್ಯ ಪೂರ್ಣಾಂಕ g_ggml_sycl_debug_sparse_fa;
ಬಾಹ್ಯ ಪೂರ್ಣಾಂಕ g_ggml_sycl_sparse_fa_margin;

ಸ್ಥಿರ ಪೂರ್ಣಾಂಕ sparse_fa_enabled(ಶೂನ್ಯ) {
    ಹಿಂತಿರುಗಿ g_ggml_sycl_enable_sparse_fa;
}

ಸ್ಥಿರ ಪೂರ್ಣಾಂಕ sparse_fa_debug(ಶೂನ್ಯ) {
    ಹಿಂತಿರುಗಿ g_ggml_sycl_debug_sparse_fa;
}

// slack above n_kv_max; callers may exceed the hint by a few always-attended positions
ಸ್ಥಿರ ಪೂರ್ಣಾಂಕ sparse_fa_margin(ಶೂನ್ಯ) {
    ಹಿಂತಿರುಗಿ g_ggml_sycl_sparse_fa_margin;
}

// Unordered output is fine: softmax over the selected set is permutation invariant.
ಸ್ಥಿರ ಶೂನ್ಯ sparse_fa_compact_ಮುಖವಾಡ(sycl::queue * stream,
                                   ಸ್ಥಿರಾಂಕ sycl::half * __restrict__ ಮುಖವಾಡ,
                                   int32_t * __restrict__ indices,
                                   int32_t * __restrict__ count,
                                   ಸ್ಥಿರಾಂಕ int64_t n_kv,
                                   ಸ್ಥಿರಾಂಕ int64_t n_kv_g) {
    constexpr size_t WG = 256;
    ಸ್ಥಿರಾಂಕ size_t global = (size_t) GGML_PAD(n_kv, (int64_t) WG);

    stream->parallel_for(
        sycl::nd_range<1>(sycl::range<1>(global), sycl::range<1>(WG)),
        [=](sycl::nd_ಅಂಶ<1> ಅಂಶ) {
            ಸ್ಥಿರಾಂಕ int64_t i = (int64_t) ಅಂಶ.get_global_id(0);
            ಆದರೆ (i >= n_kv || !sycl::isfinite((ದಶಮಾಂಶ) ಮುಖವಾಡ[i])) {
                ಹಿಂತಿರುಗಿ;
            }

            sycl::atomic_ref<int32_t,
                             sycl::memory_order::relaxed,
                             sycl::ಸ್ಮರಣೆ_scope::device,
                             sycl::access::ವಿಳಾಸ_space::global_space> ctr(*count);

            ಸ್ಥಿರಾಂಕ int32_t pos = ctr.fetch_add(1);
            ಆದರೆ (pos < (int32_t) n_kv_g) {
                indices[pos] = (int32_t) i;
            }
        });
}

// Rows along ne[0] are contiguous for every type used as a KV cache, so this is
// a plain byte copy and needs no per-type code. Padding slots are zeroed.
ಸ್ಥಿರ ಶೂನ್ಯ sparse_fa_gather_ಅಡ್ಡಸಾಲುಗಳು(sycl::queue * stream,
                                  ಸ್ಥಿರಾಂಕ uint8_t * __restrict__ src,
                                  uint8_t * __restrict__ dst,
                                  ಸ್ಥಿರಾಂಕ int32_t * __restrict__ indices,
                                  ಸ್ಥಿರಾಂಕ int32_t * __restrict__ count,
                                  ಸ್ಥಿರಾಂಕ size_t ಅಡ್ಡಸಾಲು_ಗಾತ್ರ,
                                  ಸ್ಥಿರಾಂಕ size_t src_nb1,
                                  ಸ್ಥಿರಾಂಕ size_t src_nb2,
                                  ಸ್ಥಿರಾಂಕ int64_t n_kv_g,
                                  ಸ್ಥಿರಾಂಕ int64_t n_ತಲೆ) {
    GGML_ASSERT(ಅಡ್ಡಸಾಲು_ಗಾತ್ರ % ಗಾತ್ರ(uint32_t) == 0);
    ಸ್ಥಿರಾಂಕ size_t ಪದಗಳು = ಅಡ್ಡಸಾಲು_ಗಾತ್ರ / ಗಾತ್ರ(uint32_t);

    stream->parallel_for(
        sycl::range<3>((size_t) n_ತಲೆ, (size_t) n_kv_g, ಪದಗಳು),
        [=](sycl::id<3> id) {
            ಸ್ಥಿರಾಂಕ int64_t h    = (int64_t) id[0];
            ಸ್ಥಿರಾಂಕ int64_t ಸ್ಥಾನ = (int64_t) id[1];
            ಸ್ಥಿರಾಂಕ size_t  w    = id[2];

            uint32_t * dst_ಅಡ್ಡಸಾಲು =
                (uint32_t *) (dst + ((size_t) (h * n_kv_g + ಸ್ಥಾನ)) * ಅಡ್ಡಸಾಲು_ಗಾತ್ರ);

            ಆದರೆ (ಸ್ಥಾನ >= (int64_t) *count) {
                dst_ಅಡ್ಡಸಾಲು[w] = 0;
                ಹಿಂತಿರುಗಿ;
            }

            ಸ್ಥಿರಾಂಕ uint32_t * src_ಅಡ್ಡಸಾಲು =
                (ಸ್ಥಿರಾಂಕ uint32_t *) (src + (size_t) indices[ಸ್ಥಾನ] * src_nb1 +
                                    (size_t) h * src_nb2);
            dst_ಅಡ್ಡಸಾಲು[w] = src_ಅಡ್ಡಸಾಲು[w];
        });
}

ಸ್ಥಿರ ಶೂನ್ಯ sparse_fa_gather_ಮುಖವಾಡ(sycl::queue * stream,
                                  ಸ್ಥಿರಾಂಕ sycl::half * __restrict__ ಮುಖವಾಡ,
                                  sycl::half * __restrict__ ಮುಖವಾಡ_g,
                                  ಸ್ಥಿರಾಂಕ int32_t * __restrict__ indices,
                                  ಸ್ಥಿರಾಂಕ int32_t * __restrict__ count,
                                  ಸ್ಥಿರಾಂಕ int64_t n_kv_g,
                                  ಸ್ಥಿರಾಂಕ int64_t n_ಅಡ್ಡಸಾಲುಗಳು,
                                  ಸ್ಥಿರಾಂಕ size_t ಮುಖವಾಡ_s1) {
    stream->parallel_for(
        sycl::range<2>((size_t) n_ಅಡ್ಡಸಾಲುಗಳು, (size_t) n_kv_g),
        [=](sycl::id<2> id) {
            ಸ್ಥಿರಾಂಕ int64_t r    = (int64_t) id[0];
            ಸ್ಥಿರಾಂಕ int64_t ಸ್ಥಾನ = (int64_t) id[1];

            sycl::half v = sycl::half(-INFINITY);
            ಆದರೆ (ಸ್ಥಾನ < (int64_t) *count) {
                v = ಮುಖವಾಡ[(size_t) r * ಮುಖವಾಡ_s1 + (size_t) indices[ಸ್ಥಾನ]];
            }
            ಮುಖವಾಡ_g[(size_t) r * n_kv_g + ಸ್ಥಾನ] = v;
        });
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ sparse_fa_applicable(ಸ್ಥಿರಾಂಕ ggml_tensor * dst, int64_t & n_kv_g_ಹೊರಹೋಗು) {
    ಸ್ಥಿರಾಂಕ ggml_tensor * Q    = dst->src[0];
    ಸ್ಥಿರಾಂಕ ggml_tensor * K    = dst->src[1];
    ಸ್ಥಿರಾಂಕ ggml_tensor * V    = dst->src[2];
    ಸ್ಥಿರಾಂಕ ggml_tensor * ಮುಖವಾಡ = dst->src[3];

    ಆದರೆ (!Q || !K || !V || !ಮುಖವಾಡ) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ಸ್ಥಿರಾಂಕ int32_t n_kv_max = ggml_get_op_params_i32(dst, 4);
    ಆದರೆ (n_kv_max <= 0) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ದಶಮಾಂಶ max_bias      = 0.0f;
    ದಶಮಾಂಶ logit_softcap = 0.0f;
    ಸ್ಮರಣೆ_ನಕಲು(&max_bias,      (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) dst->op_params + 1, ಗಾತ್ರ(ದಶಮಾಂಶ));
    ಸ್ಮರಣೆ_ನಕಲು(&logit_softcap, (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) dst->op_params + 2, ಗಾತ್ರ(ದಶಮಾಂಶ));
    ಆದರೆ (max_bias != 0.0f || logit_softcap != 0.0f) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    // single-token decode only; prefill amortises the scan already
    ಆದರೆ (Q->ne[1] != 1) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }
    ಆದರೆ (K->ne[3] != 1 || V->ne[3] != 1 || ಮುಖವಾಡ->ne[2] != 1 || ಮುಖವಾಡ->ne[3] != 1) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }
    ಆದರೆ (ಮುಖವಾಡ->type != GGML_TYPE_F16 || ಮುಖವಾಡ->ne[0] < K->ne[1]) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }
    ಆದರೆ (K->ne[2] != V->ne[2]) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    // nb[1] may stride over heads (interleaved cache); only ne[0] must be contiguous
    ಆದರೆ (K->nb[0] != ggml_type_ಗಾತ್ರ(K->type) || V->nb[0] != ggml_type_ಗಾತ್ರ(V->type)) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ಸ್ಥಿರಾಂಕ size_t k_ಅಡ್ಡಸಾಲು = ggml_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ(K->type, K->ne[0]);
    ಸ್ಥಿರಾಂಕ size_t v_ಅಡ್ಡಸಾಲು = ggml_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ(V->type, V->ne[0]);
    ಆದರೆ (k_ಅಡ್ಡಸಾಲು % ಗಾತ್ರ(uint32_t) || v_ಅಡ್ಡಸಾಲು % ಗಾತ್ರ(uint32_t)) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ಸ್ಥಿರಾಂಕ int64_t n_kv_g = GGML_PAD((int64_t) n_kv_max + sparse_fa_margin(), SPARSE_FA_PAD);
    ಆದರೆ (n_kv_g * SPARSE_FA_MIN_ಅನುಪಾತ > K->ne[1]) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    n_kv_g_ಹೊರಹೋಗು = n_kv_g;
    ಹಿಂತಿರುಗಿ ನಿಜ;
}

ತಾರ್ಕಿಕ ggml_sycl_flash_attn_ext_sparse(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    int64_t n_kv_g = 0;
    ಆದರೆ (!sparse_fa_enabled() || !sparse_fa_applicable(dst, n_kv_g)) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ggml_tensor * K    = dst->src[1];
    ggml_tensor * V    = dst->src[2];
    ggml_tensor * ಮುಖವಾಡ = dst->src[3];

    ಸ್ಥಿರಾಂಕ int64_t n_kv     = K->ne[1];
    ಸ್ಥಿರಾಂಕ int64_t n_ತಲೆ_k = K->ne[2];
    ಸ್ಥಿರಾಂಕ int64_t n_ಅಡ್ಡಸಾಲುಗಳು_m = ಮುಖವಾಡ->ne[1];

    ಸ್ಥಿರಾಂಕ size_t k_ಅಡ್ಡಸಾಲು = ggml_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ(K->type, K->ne[0]);
    ಸ್ಥಿರಾಂಕ size_t v_ಅಡ್ಡಸಾಲು = ggml_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ(V->type, V->ne[0]);

    dpct::ಸರತಿ_ptr stream = ctx.stream();

    ggml_sycl_pool_alloc<int32_t>    idx_alloc(ctx.pool(), (size_t) n_kv_g);
    ggml_sycl_pool_alloc<int32_t>    cnt_alloc(ctx.pool(), 1);
    ggml_sycl_pool_alloc<uint8_t>    k_alloc(ctx.pool(), (size_t) n_ತಲೆ_k * n_kv_g * k_ಅಡ್ಡಸಾಲು);
    ggml_sycl_pool_alloc<uint8_t>    v_alloc(ctx.pool(), (size_t) n_ತಲೆ_k * n_kv_g * v_ಅಡ್ಡಸಾಲು);
    ggml_sycl_pool_alloc<sycl::half> m_alloc(ctx.pool(), (size_t) n_ಅಡ್ಡಸಾಲುಗಳು_m * n_kv_g);

    int32_t *    d_idx  = idx_alloc.get();
    int32_t *    d_cnt  = cnt_alloc.get();
    uint8_t *    d_K    = k_alloc.get();
    uint8_t *    d_V    = v_alloc.get();
    sycl::half * d_ಮುಖವಾಡ = m_alloc.get();

    // reset with a kernel: on Level Zero a queue memset is not always ordered before the next kernel
    stream->single_task([=]() { *d_cnt = 0; });

    sparse_fa_compact_ಮುಖವಾಡ(stream, (ಸ್ಥಿರಾಂಕ sycl::half *) ಮುಖವಾಡ->ದತ್ತಾಂಶ,
                           d_idx, d_cnt, n_kv, n_kv_g);

    sparse_fa_gather_ಅಡ್ಡಸಾಲುಗಳು(stream, (ಸ್ಥಿರಾಂಕ uint8_t *) K->ದತ್ತಾಂಶ, d_K, d_idx, d_cnt,
                          k_ಅಡ್ಡಸಾಲು, K->nb[1], K->nb[2], n_kv_g, n_ತಲೆ_k);

    sparse_fa_gather_ಅಡ್ಡಸಾಲುಗಳು(stream, (ಸ್ಥಿರಾಂಕ uint8_t *) V->ದತ್ತಾಂಶ, d_V, d_idx, d_cnt,
                          v_ಅಡ್ಡಸಾಲು, V->nb[1], V->nb[2], n_kv_g, n_ತಲೆ_k);

    sparse_fa_gather_ಮುಖವಾಡ(stream, (ಸ್ಥಿರಾಂಕ sycl::half *) ಮುಖವಾಡ->ದತ್ತಾಂಶ, d_ಮುಖವಾಡ,
                          d_idx, d_cnt, n_kv_g, n_ಅಡ್ಡಸಾಲುಗಳು_m,
                          ಮುಖವಾಡ->nb[1] / ಗಾತ್ರ(sycl::half));

    ಆದರೆ (sparse_fa_debug()) {
        int32_t h_cnt = 0;
        SYCL_CHECK(CHECK_TRY_ERROR(stream->ಸ್ಮರಣೆ_ನಕಲು(&h_cnt, d_cnt, ಗಾತ್ರ(int32_t))));
        SYCL_CHECK(CHECK_TRY_ERROR(stream->wait()));
        fprintf(stderr, "[FA-SPARSE] n_kv=%lld n_kv_max=%d n_kv_g=%lld finite=%d%s\n",
                (ದೀರ್ಘ ದೀರ್ಘ) n_kv, ggml_get_op_params_i32(dst, 4),
                (ದೀರ್ಘ ದೀರ್ಘ) n_kv_g, (ಪೂರ್ಣಾಂಕ) h_cnt,
                h_cnt > (int32_t) n_kv_g ? "  OVERFLOW" : "");
    }

    // shallow copies retargeted at the gathered buffers; kernels are unchanged
    ggml_tensor K_g = *K;
    K_g.ದತ್ತಾಂಶ      = d_K;
    K_g.ne[1]     = n_kv_g;
    K_g.nb[1]     = k_ಅಡ್ಡಸಾಲು;
    K_g.nb[2]     = (size_t) n_kv_g * k_ಅಡ್ಡಸಾಲು;
    K_g.nb[3]     = (size_t) n_ತಲೆ_k * n_kv_g * k_ಅಡ್ಡಸಾಲು;
    K_g.view_src  = ಶೂನ್ಯಸೂಚಕ;
    K_g.view_offs = 0;

    ggml_tensor V_g = *V;
    V_g.ದತ್ತಾಂಶ      = d_V;
    V_g.ne[1]     = n_kv_g;
    V_g.nb[1]     = v_ಅಡ್ಡಸಾಲು;
    V_g.nb[2]     = (size_t) n_kv_g * v_ಅಡ್ಡಸಾಲು;
    V_g.nb[3]     = (size_t) V->ne[2] * n_kv_g * v_ಅಡ್ಡಸಾಲು;
    V_g.view_src  = ಶೂನ್ಯಸೂಚಕ;
    V_g.view_offs = 0;

    ggml_tensor M_g = *ಮುಖವಾಡ;
    M_g.ದತ್ತಾಂಶ      = d_ಮುಖವಾಡ;
    M_g.ne[0]     = n_kv_g;
    M_g.nb[1]     = (size_t) n_kv_g * ಗಾತ್ರ(sycl::half);
    M_g.nb[2]     = M_g.nb[1] * ಮುಖವಾಡ->ne[1];
    M_g.nb[3]     = M_g.nb[2];
    M_g.view_src  = ಶೂನ್ಯಸೂಚಕ;
    M_g.view_offs = 0;

    ggml_tensor dst_g = *dst;
    dst_g.src[1] = &K_g;
    dst_g.src[2] = &V_g;
    dst_g.src[3] = &M_g;
    dst_g.op_params[4] = 0;   // avoid re-entering this path

    ggml_sycl_flash_attn_ext(ctx, &dst_g);

    ಹಿಂತಿರುಗಿ ನಿಜ;
}
