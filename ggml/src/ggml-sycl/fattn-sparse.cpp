#अन्तर्भाव "fattn.hpp"
#अन्तर्भाव "fattn-sparse.hpp"

#अन्तर्भाव <cstdint>
#अन्तर्भाव <cstdio>
#अन्तर्भाव <cstring>

स्थिर constexpr int64_t SPARSE_FA_PAD       = 256;
स्थिर constexpr int64_t SPARSE_FA_MIN_RATIO = 2;

बाह्य पूर्णाङ्क g_ggml_sycl_enable_sparse_fa;
बाह्य पूर्णाङ्क g_ggml_sycl_debug_sparse_fa;
बाह्य पूर्णाङ्क g_ggml_sycl_sparse_fa_margin;

स्थिर पूर्णाङ्क sparse_fa_enabled(शून्य) {
    फर्काऊ g_ggml_sycl_enable_sparse_fa;
}

स्थिर पूर्णाङ्क sparse_fa_debug(शून्य) {
    फर्काऊ g_ggml_sycl_debug_sparse_fa;
}

// slack above n_kv_max; callers may exceed the hint by a few always-attended positions
स्थिर पूर्णाङ्क sparse_fa_margin(शून्य) {
    फर्काऊ g_ggml_sycl_sparse_fa_margin;
}

// Unordered output is fine: softmax over the selected set is permutation invariant.
स्थिर शून्य sparse_fa_compact_mask(sycl::queue * stream,
                                   अचर sycl::half * __restrict__ mask,
                                   int32_t * __restrict__ indices,
                                   int32_t * __restrict__ count,
                                   अचर int64_t n_kv,
                                   अचर int64_t n_kv_g) {
    constexpr माप_प्रकार WG = 256;
    अचर माप_प्रकार global = (माप_प्रकार) GGML_PAD(n_kv, (int64_t) WG);

    stream->parallel_for(
        sycl::nd_range<1>(sycl::range<1>(global), sycl::range<1>(WG)),
        [=](sycl::nd_item<1> item) {
            अचर int64_t i = (int64_t) item.get_global_id(0);
            यदि (i >= n_kv || !sycl::है_सीमित((भिन्न) mask[i])) {
                फर्काऊ;
            }

            sycl::atomic_ref<int32_t,
                             sycl::memory_order::relaxed,
                             sycl::memory_scope::device,
                             sycl::access::address_space::global_space> ctr(*count);

            अचर int32_t pos = ctr.fetch_add(1);
            यदि (pos < (int32_t) n_kv_g) {
                indices[pos] = (int32_t) i;
            }
        });
}

// Rows along ne[0] are contiguous for every type used as a KV cache, so this is
// a plain byte copy and needs no per-type code. Padding slots are zeroed.
स्थिर शून्य sparse_fa_gather_rows(sycl::queue * stream,
                                  अचर uint8_t * __restrict__ src,
                                  uint8_t * __restrict__ dst,
                                  अचर int32_t * __restrict__ indices,
                                  अचर int32_t * __restrict__ count,
                                  अचर माप_प्रकार row_size,
                                  अचर माप_प्रकार src_nb1,
                                  अचर माप_प्रकार src_nb2,
                                  अचर int64_t n_kv_g,
                                  अचर int64_t n_head) {
    GGML_ASSERT(row_size % परिमाण(uint32_t) == 0);
    अचर माप_प्रकार शब्दहरू = row_size / परिमाण(uint32_t);

    stream->parallel_for(
        sycl::range<3>((माप_प्रकार) n_head, (माप_प्रकार) n_kv_g, शब्दहरू),
        [=](sycl::id<3> id) {
            अचर int64_t h    = (int64_t) id[0];
            अचर int64_t slot = (int64_t) id[1];
            अचर माप_प्रकार  w    = id[2];

            uint32_t * dst_row =
                (uint32_t *) (dst + ((माप_प्रकार) (h * n_kv_g + slot)) * row_size);

            यदि (slot >= (int64_t) *count) {
                dst_row[w] = 0;
                फर्काऊ;
            }

            अचर uint32_t * src_row =
                (अचर uint32_t *) (src + (माप_प्रकार) indices[slot] * src_nb1 +
                                    (माप_प्रकार) h * src_nb2);
            dst_row[w] = src_row[w];
        });
}

स्थिर शून्य sparse_fa_gather_mask(sycl::queue * stream,
                                  अचर sycl::half * __restrict__ mask,
                                  sycl::half * __restrict__ mask_g,
                                  अचर int32_t * __restrict__ indices,
                                  अचर int32_t * __restrict__ count,
                                  अचर int64_t n_kv_g,
                                  अचर int64_t n_rows,
                                  अचर माप_प्रकार mask_s1) {
    stream->parallel_for(
        sycl::range<2>((माप_प्रकार) n_rows, (माप_प्रकार) n_kv_g),
        [=](sycl::id<2> id) {
            अचर int64_t r    = (int64_t) id[0];
            अचर int64_t slot = (int64_t) id[1];

            sycl::half v = sycl::half(-अनन्त);
            यदि (slot < (int64_t) *count) {
                v = mask[(माप_प्रकार) r * mask_s1 + (माप_प्रकार) indices[slot]];
            }
            mask_g[(माप_प्रकार) r * n_kv_g + slot] = v;
        });
}

स्थिर तार्किक sparse_fa_applicable(अचर ggml_tensor * dst, int64_t & n_kv_g_out) {
    अचर ggml_tensor * Q    = dst->src[0];
    अचर ggml_tensor * K    = dst->src[1];
    अचर ggml_tensor * V    = dst->src[2];
    अचर ggml_tensor * mask = dst->src[3];

    यदि (!Q || !K || !V || !mask) {
        फर्काऊ असत्य;
    }

    अचर int32_t n_kv_max = ggml_get_op_params_i32(dst, 4);
    यदि (n_kv_max <= 0) {
        फर्काऊ असत्य;
    }

    भिन्न max_bias      = 0.0f;
    भिन्न logit_softcap = 0.0f;
    स_नकल(&max_bias,      (अचर भिन्न *) dst->op_params + 1, परिमाण(भिन्न));
    स_नकल(&logit_softcap, (अचर भिन्न *) dst->op_params + 2, परिमाण(भिन्न));
    यदि (max_bias != 0.0f || logit_softcap != 0.0f) {
        फर्काऊ असत्य;
    }

    // single-token decode only; prefill amortises the scan already
    यदि (Q->ne[1] != 1) {
        फर्काऊ असत्य;
    }
    यदि (K->ne[3] != 1 || V->ne[3] != 1 || mask->ne[2] != 1 || mask->ne[3] != 1) {
        फर्काऊ असत्य;
    }
    यदि (mask->type != GGML_TYPE_F16 || mask->ne[0] < K->ne[1]) {
        फर्काऊ असत्य;
    }
    यदि (K->ne[2] != V->ne[2]) {
        फर्काऊ असत्य;
    }

    // nb[1] may stride over heads (interleaved cache); only ne[0] must be contiguous
    यदि (K->nb[0] != ggml_type_size(K->type) || V->nb[0] != ggml_type_size(V->type)) {
        फर्काऊ असत्य;
    }

    अचर माप_प्रकार k_row = ggml_row_size(K->type, K->ne[0]);
    अचर माप_प्रकार v_row = ggml_row_size(V->type, V->ne[0]);
    यदि (k_row % परिमाण(uint32_t) || v_row % परिमाण(uint32_t)) {
        फर्काऊ असत्य;
    }

    अचर int64_t n_kv_g = GGML_PAD((int64_t) n_kv_max + sparse_fa_margin(), SPARSE_FA_PAD);
    यदि (n_kv_g * SPARSE_FA_MIN_RATIO > K->ne[1]) {
        फर्काऊ असत्य;
    }

    n_kv_g_out = n_kv_g;
    फर्काऊ सत्य;
}

तार्किक ggml_sycl_flash_attn_ext_sparse(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    int64_t n_kv_g = 0;
    यदि (!sparse_fa_enabled() || !sparse_fa_applicable(dst, n_kv_g)) {
        फर्काऊ असत्य;
    }

    ggml_tensor * K    = dst->src[1];
    ggml_tensor * V    = dst->src[2];
    ggml_tensor * mask = dst->src[3];

    अचर int64_t n_kv     = K->ne[1];
    अचर int64_t n_head_k = K->ne[2];
    अचर int64_t n_rows_m = mask->ne[1];

    अचर माप_प्रकार k_row = ggml_row_size(K->type, K->ne[0]);
    अचर माप_प्रकार v_row = ggml_row_size(V->type, V->ne[0]);

    dpct::queue_ptr stream = ctx.stream();

    ggml_sycl_pool_alloc<int32_t>    idx_alloc(ctx.pool(), (माप_प्रकार) n_kv_g);
    ggml_sycl_pool_alloc<int32_t>    cnt_alloc(ctx.pool(), 1);
    ggml_sycl_pool_alloc<uint8_t>    k_alloc(ctx.pool(), (माप_प्रकार) n_head_k * n_kv_g * k_row);
    ggml_sycl_pool_alloc<uint8_t>    v_alloc(ctx.pool(), (माप_प्रकार) n_head_k * n_kv_g * v_row);
    ggml_sycl_pool_alloc<sycl::half> m_alloc(ctx.pool(), (माप_प्रकार) n_rows_m * n_kv_g);

    int32_t *    d_idx  = idx_alloc.get();
    int32_t *    d_cnt  = cnt_alloc.get();
    uint8_t *    d_K    = k_alloc.get();
    uint8_t *    d_V    = v_alloc.get();
    sycl::half * d_mask = m_alloc.get();

    // reset with a kernel: on Level Zero a queue memset is not always ordered before the next kernel
    stream->single_task([=]() { *d_cnt = 0; });

    sparse_fa_compact_mask(stream, (अचर sycl::half *) mask->data,
                           d_idx, d_cnt, n_kv, n_kv_g);

    sparse_fa_gather_rows(stream, (अचर uint8_t *) K->data, d_K, d_idx, d_cnt,
                          k_row, K->nb[1], K->nb[2], n_kv_g, n_head_k);

    sparse_fa_gather_rows(stream, (अचर uint8_t *) V->data, d_V, d_idx, d_cnt,
                          v_row, V->nb[1], V->nb[2], n_kv_g, n_head_k);

    sparse_fa_gather_mask(stream, (अचर sycl::half *) mask->data, d_mask,
                          d_idx, d_cnt, n_kv_g, n_rows_m,
                          mask->nb[1] / परिमाण(sycl::half));

    यदि (sparse_fa_debug()) {
        int32_t h_cnt = 0;
        SYCL_CHECK(CHECK_TRY_ERROR(stream->स_नकल(&h_cnt, d_cnt, परिमाण(int32_t))));
        SYCL_CHECK(CHECK_TRY_ERROR(stream->रुको()));
        ख_लिखो(मानक_त्रुटि, "[FA-SPARSE] n_kv=%lld n_kv_max=%d n_kv_g=%lld finite=%d%s\n",
                (दीर्घ दीर्घ) n_kv, ggml_get_op_params_i32(dst, 4),
                (दीर्घ दीर्घ) n_kv_g, (पूर्णाङ्क) h_cnt,
                h_cnt > (int32_t) n_kv_g ? "  OVERFLOW" : "");
    }

    // shallow copies retargeted at the gathered buffers; kernels are unchanged
    ggml_tensor K_g = *K;
    K_g.data      = d_K;
    K_g.ne[1]     = n_kv_g;
    K_g.nb[1]     = k_row;
    K_g.nb[2]     = (माप_प्रकार) n_kv_g * k_row;
    K_g.nb[3]     = (माप_प्रकार) n_head_k * n_kv_g * k_row;
    K_g.view_src  = nullptr;
    K_g.view_offs = 0;

    ggml_tensor V_g = *V;
    V_g.data      = d_V;
    V_g.ne[1]     = n_kv_g;
    V_g.nb[1]     = v_row;
    V_g.nb[2]     = (माप_प्रकार) n_kv_g * v_row;
    V_g.nb[3]     = (माप_प्रकार) V->ne[2] * n_kv_g * v_row;
    V_g.view_src  = nullptr;
    V_g.view_offs = 0;

    ggml_tensor M_g = *mask;
    M_g.data      = d_mask;
    M_g.ne[0]     = n_kv_g;
    M_g.nb[1]     = (माप_प्रकार) n_kv_g * परिमाण(sycl::half);
    M_g.nb[2]     = M_g.nb[1] * mask->ne[1];
    M_g.nb[3]     = M_g.nb[2];
    M_g.view_src  = nullptr;
    M_g.view_offs = 0;

    ggml_tensor dst_g = *dst;
    dst_g.src[1] = &K_g;
    dst_g.src[2] = &V_g;
    dst_g.src[3] = &M_g;
    dst_g.op_params[4] = 0;   // avoid re-entering this path

    ggml_sycl_flash_attn_ext(ctx, &dst_g);

    फर्काऊ सत्य;
}
