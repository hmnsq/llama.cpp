#अन्तर्भाव "fattn.hpp"
#अन्तर्भाव "fattn-sparse.hpp"

#अन्तर्भाव <cstdint>
#अन्तर्भाव <cstdio>
#अन्तर्भाव <cstring>

स्थिर constexpr int64_t SPARSE_FA_PAD       = 256;
स्थिर constexpr int64_t SPARSE_FA_MIN_अनुपातः = 2;

बाह्य पूर्णाङ्क g_ggml_sycl_enable_sparse_fa;
बाह्य पूर्णाङ्क g_ggml_sycl_debug_sparse_fa;
बाह्य पूर्णाङ्क g_ggml_sycl_sparse_fa_margin;

स्थिर पूर्णाङ्क sparse_fa_enabled(शून्य) {
    प्रत्यागम g_ggml_sycl_enable_sparse_fa;
}

स्थिर पूर्णाङ्क sparse_fa_debug(शून्य) {
    प्रत्यागम g_ggml_sycl_debug_sparse_fa;
}

// slack above n_kv_max; callers may exceed the hint by a few always-attended positions
स्थिर पूर्णाङ्क sparse_fa_margin(शून्य) {
    प्रत्यागम g_ggml_sycl_sparse_fa_margin;
}

// Unordered output is fine: softmax over the selected set is permutation invariant.
स्थिर शून्य sparse_fa_compact_आवरणम्(sycl::queue * stream,
                                   ध्रुव sycl::half * __restrict__ आवरणम्,
                                   int32_t * __restrict__ indices,
                                   int32_t * __restrict__ count,
                                   ध्रुव int64_t n_kv,
                                   ध्रुव int64_t n_kv_g) {
    constexpr size_t WG = 256;
    ध्रुव size_t global = (size_t) GGML_PAD(n_kv, (int64_t) WG);

    stream->parallel_for(
        sycl::nd_range<1>(sycl::range<1>(global), sycl::range<1>(WG)),
        [=](sycl::nd_item<1> item) {
            ध्रुव int64_t i = (int64_t) item.get_global_id(0);
            यदि (i >= n_kv || !sycl::isfinite((भिन्न) आवरणम्[i])) {
                प्रत्यागम;
            }

            sycl::atomic_ref<int32_t,
                             sycl::memory_order::relaxed,
                             sycl::स्मृतिः_scope::device,
                             sycl::access::स्थानाङ्कः_space::global_space> ctr(*count);

            ध्रुव int32_t pos = ctr.fetch_add(1);
            यदि (pos < (int32_t) n_kv_g) {
                indices[pos] = (int32_t) i;
            }
        });
}

// Rows along ne[0] are contiguous for every type used as a KV cache, so this is
// a plain byte copy and needs no per-type code. Padding slots are zeroed.
स्थिर शून्य sparse_fa_gather_पङ्क्तिकाः(sycl::queue * stream,
                                  ध्रुव uint8_t * __restrict__ src,
                                  uint8_t * __restrict__ dst,
                                  ध्रुव int32_t * __restrict__ indices,
                                  ध्रुव int32_t * __restrict__ count,
                                  ध्रुव size_t row_आकार,
                                  ध्रुव size_t src_nb1,
                                  ध्रुव size_t src_nb2,
                                  ध्रुव int64_t n_kv_g,
                                  ध्रुव int64_t n_शीर्ष) {
    GGML_ASSERT(row_आकार % परिमाण(uint32_t) == 0);
    ध्रुव size_t शब्दाः = row_आकार / परिमाण(uint32_t);

    stream->parallel_for(
        sycl::range<3>((size_t) n_शीर्ष, (size_t) n_kv_g, शब्दाः),
        [=](sycl::id<3> id) {
            ध्रुव int64_t h    = (int64_t) id[0];
            ध्रुव int64_t slot = (int64_t) id[1];
            ध्रुव size_t  w    = id[2];

            uint32_t * dst_row =
                (uint32_t *) (dst + ((size_t) (h * n_kv_g + slot)) * row_आकार);

            यदि (slot >= (int64_t) *count) {
                dst_row[w] = 0;
                प्रत्यागम;
            }

            ध्रुव uint32_t * src_row =
                (ध्रुव uint32_t *) (src + (size_t) indices[slot] * src_nb1 +
                                    (size_t) h * src_nb2);
            dst_row[w] = src_row[w];
        });
}

स्थिर शून्य sparse_fa_gather_आवरणम्(sycl::queue * stream,
                                  ध्रुव sycl::half * __restrict__ आवरणम्,
                                  sycl::half * __restrict__ आवरणम्_g,
                                  ध्रुव int32_t * __restrict__ indices,
                                  ध्रुव int32_t * __restrict__ count,
                                  ध्रुव int64_t n_kv_g,
                                  ध्रुव int64_t n_पङ्क्तिकाः,
                                  ध्रुव size_t आवरणम्_s1) {
    stream->parallel_for(
        sycl::range<2>((size_t) n_पङ्क्तिकाः, (size_t) n_kv_g),
        [=](sycl::id<2> id) {
            ध्रुव int64_t r    = (int64_t) id[0];
            ध्रुव int64_t slot = (int64_t) id[1];

            sycl::half v = sycl::half(-INFINITY);
            यदि (slot < (int64_t) *count) {
                v = आवरणम्[(size_t) r * आवरणम्_s1 + (size_t) indices[slot]];
            }
            आवरणम्_g[(size_t) r * n_kv_g + slot] = v;
        });
}

स्थिर तार्किक sparse_fa_applicable(ध्रुव ggml_tensor * dst, int64_t & n_kv_g_बहिर्गमनम्) {
    ध्रुव ggml_tensor * Q    = dst->src[0];
    ध्रुव ggml_tensor * K    = dst->src[1];
    ध्रुव ggml_tensor * V    = dst->src[2];
    ध्रुव ggml_tensor * आवरणम् = dst->src[3];

    यदि (!Q || !K || !V || !आवरणम्) {
        प्रत्यागम असत्य;
    }

    ध्रुव int32_t n_kv_max = ggml_get_op_params_i32(dst, 4);
    यदि (n_kv_max <= 0) {
        प्रत्यागम असत्य;
    }

    भिन्न max_bias      = 0.0f;
    भिन्न logit_softcap = 0.0f;
    स्मृतिं_प्रतिलिख(&max_bias,      (ध्रुव भिन्न *) dst->op_params + 1, परिमाण(भिन्न));
    स्मृतिं_प्रतिलिख(&logit_softcap, (ध्रुव भिन्न *) dst->op_params + 2, परिमाण(भिन्न));
    यदि (max_bias != 0.0f || logit_softcap != 0.0f) {
        प्रत्यागम असत्य;
    }

    // single-token decode only; prefill amortises the scan already
    यदि (Q->ne[1] != 1) {
        प्रत्यागम असत्य;
    }
    यदि (K->ne[3] != 1 || V->ne[3] != 1 || आवरणम्->ne[2] != 1 || आवरणम्->ne[3] != 1) {
        प्रत्यागम असत्य;
    }
    यदि (आवरणम्->type != GGML_TYPE_F16 || आवरणम्->ne[0] < K->ne[1]) {
        प्रत्यागम असत्य;
    }
    यदि (K->ne[2] != V->ne[2]) {
        प्रत्यागम असत्य;
    }

    // nb[1] may stride over heads (interleaved cache); only ne[0] must be contiguous
    यदि (K->nb[0] != ggml_type_आकार(K->type) || V->nb[0] != ggml_type_आकार(V->type)) {
        प्रत्यागम असत्य;
    }

    ध्रुव size_t k_row = ggml_row_आकार(K->type, K->ne[0]);
    ध्रुव size_t v_row = ggml_row_आकार(V->type, V->ne[0]);
    यदि (k_row % परिमाण(uint32_t) || v_row % परिमाण(uint32_t)) {
        प्रत्यागम असत्य;
    }

    ध्रुव int64_t n_kv_g = GGML_PAD((int64_t) n_kv_max + sparse_fa_margin(), SPARSE_FA_PAD);
    यदि (n_kv_g * SPARSE_FA_MIN_अनुपातः > K->ne[1]) {
        प्रत्यागम असत्य;
    }

    n_kv_g_बहिर्गमनम् = n_kv_g;
    प्रत्यागम सत्य;
}

तार्किक ggml_sycl_flash_attn_ext_sparse(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    int64_t n_kv_g = 0;
    यदि (!sparse_fa_enabled() || !sparse_fa_applicable(dst, n_kv_g)) {
        प्रत्यागम असत्य;
    }

    ggml_tensor * K    = dst->src[1];
    ggml_tensor * V    = dst->src[2];
    ggml_tensor * आवरणम् = dst->src[3];

    ध्रुव int64_t n_kv     = K->ne[1];
    ध्रुव int64_t n_शीर्ष_k = K->ne[2];
    ध्रुव int64_t n_पङ्क्तिकाः_m = आवरणम्->ne[1];

    ध्रुव size_t k_row = ggml_row_आकार(K->type, K->ne[0]);
    ध्रुव size_t v_row = ggml_row_आकार(V->type, V->ne[0]);

    dpct::queue_ptr stream = ctx.stream();

    ggml_sycl_pool_alloc<int32_t>    idx_alloc(ctx.pool(), (size_t) n_kv_g);
    ggml_sycl_pool_alloc<int32_t>    cnt_alloc(ctx.pool(), 1);
    ggml_sycl_pool_alloc<uint8_t>    k_alloc(ctx.pool(), (size_t) n_शीर्ष_k * n_kv_g * k_row);
    ggml_sycl_pool_alloc<uint8_t>    v_alloc(ctx.pool(), (size_t) n_शीर्ष_k * n_kv_g * v_row);
    ggml_sycl_pool_alloc<sycl::half> m_alloc(ctx.pool(), (size_t) n_पङ्क्तिकाः_m * n_kv_g);

    int32_t *    d_idx  = idx_alloc.get();
    int32_t *    d_cnt  = cnt_alloc.get();
    uint8_t *    d_K    = k_alloc.get();
    uint8_t *    d_V    = v_alloc.get();
    sycl::half * d_आवरणम् = m_alloc.get();

    // reset with a kernel: on Level Zero a queue memset is not always ordered before the next kernel
    stream->single_task([=]() { *d_cnt = 0; });

    sparse_fa_compact_आवरणम्(stream, (ध्रुव sycl::half *) आवरणम्->दत्त,
                           d_idx, d_cnt, n_kv, n_kv_g);

    sparse_fa_gather_पङ्क्तिकाः(stream, (ध्रुव uint8_t *) K->दत्त, d_K, d_idx, d_cnt,
                          k_row, K->nb[1], K->nb[2], n_kv_g, n_शीर्ष_k);

    sparse_fa_gather_पङ्क्तिकाः(stream, (ध्रुव uint8_t *) V->दत्त, d_V, d_idx, d_cnt,
                          v_row, V->nb[1], V->nb[2], n_kv_g, n_शीर्ष_k);

    sparse_fa_gather_आवरणम्(stream, (ध्रुव sycl::half *) आवरणम्->दत्त, d_आवरणम्,
                          d_idx, d_cnt, n_kv_g, n_पङ्क्तिकाः_m,
                          आवरणम्->nb[1] / परिमाण(sycl::half));

    यदि (sparse_fa_debug()) {
        int32_t h_cnt = 0;
        SYCL_CHECK(CHECK_TRY_ERROR(stream->स्मृतिं_प्रतिलिख(&h_cnt, d_cnt, परिमाण(int32_t))));
        SYCL_CHECK(CHECK_TRY_ERROR(stream->प्रतीक्षस्व()));
        fprintf(stderr, "[FA-SPARSE] n_kv=%lld n_kv_max=%d n_kv_g=%lld finite=%d%s\n",
                (दीर्घ दीर्घ) n_kv, ggml_get_op_params_i32(dst, 4),
                (दीर्घ दीर्घ) n_kv_g, (पूर्णाङ्क) h_cnt,
                h_cnt > (int32_t) n_kv_g ? "  OVERFLOW" : "");
    }

    // shallow copies retargeted at the gathered buffers; kernels are unchanged
    ggml_tensor K_g = *K;
    K_g.दत्त      = d_K;
    K_g.ne[1]     = n_kv_g;
    K_g.nb[1]     = k_row;
    K_g.nb[2]     = (size_t) n_kv_g * k_row;
    K_g.nb[3]     = (size_t) n_शीर्ष_k * n_kv_g * k_row;
    K_g.view_src  = शून्यसूचकः;
    K_g.view_offs = 0;

    ggml_tensor V_g = *V;
    V_g.दत्त      = d_V;
    V_g.ne[1]     = n_kv_g;
    V_g.nb[1]     = v_row;
    V_g.nb[2]     = (size_t) n_kv_g * v_row;
    V_g.nb[3]     = (size_t) V->ne[2] * n_kv_g * v_row;
    V_g.view_src  = शून्यसूचकः;
    V_g.view_offs = 0;

    ggml_tensor M_g = *आवरणम्;
    M_g.दत्त      = d_आवरणम्;
    M_g.ne[0]     = n_kv_g;
    M_g.nb[1]     = (size_t) n_kv_g * परिमाण(sycl::half);
    M_g.nb[2]     = M_g.nb[1] * आवरणम्->ne[1];
    M_g.nb[3]     = M_g.nb[2];
    M_g.view_src  = शून्यसूचकः;
    M_g.view_offs = 0;

    ggml_tensor dst_g = *dst;
    dst_g.src[1] = &K_g;
    dst_g.src[2] = &V_g;
    dst_g.src[3] = &M_g;
    dst_g.op_params[4] = 0;   // avoid re-entering this path

    ggml_sycl_flash_attn_ext(ctx, &dst_g);

    प्रत्यागम सत्य;
}
