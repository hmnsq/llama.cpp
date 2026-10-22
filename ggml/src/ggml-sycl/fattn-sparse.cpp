#ಅನ್ತರ್ಭಾವ "fattn.hpp"
#ಅನ್ತರ್ಭಾವ "fattn-sparse.hpp"

#ಅನ್ತರ್ಭಾವ <cstdint>
#ಅನ್ತರ್ಭಾವ <cstdio>
#ಅನ್ತರ್ಭಾವ <cstring>

ಸ್ಥಿರ constexpr int64_t SPARSE_FA_PAD       = 256;
ಸ್ಥಿರ constexpr int64_t SPARSE_FA_MIN_RATIO = 2;

ಬಾಹ್ಯ ಪೂರ್ಣಾಙ್ಕ g_ggml_sycl_enable_sparse_fa;
ಬಾಹ್ಯ ಪೂರ್ಣಾಙ್ಕ g_ggml_sycl_debug_sparse_fa;
ಬಾಹ್ಯ ಪೂರ್ಣಾಙ್ಕ g_ggml_sycl_sparse_fa_margin;

ಸ್ಥಿರ ಪೂರ್ಣಾಙ್ಕ sparse_fa_enabled(ಶೂನ್ಯ) {
    ಪ್ರತ್ಯಾಗಮ g_ggml_sycl_enable_sparse_fa;
}

ಸ್ಥಿರ ಪೂರ್ಣಾಙ್ಕ sparse_fa_debug(ಶೂನ್ಯ) {
    ಪ್ರತ್ಯಾಗಮ g_ggml_sycl_debug_sparse_fa;
}

// slack above n_kv_max; callers may exceed the hint by a few always-attended positions
ಸ್ಥಿರ ಪೂರ್ಣಾಙ್ಕ sparse_fa_margin(ಶೂನ್ಯ) {
    ಪ್ರತ್ಯಾಗಮ g_ggml_sycl_sparse_fa_margin;
}

// Unordered output is fine: softmax over the selected set is permutation invariant.
ಸ್ಥಿರ ಶೂನ್ಯ sparse_fa_compact_mask(sycl::queue * stream,
                                   ಧ್ರುವ sycl::half * __restrict__ mask,
                                   int32_t * __restrict__ indices,
                                   int32_t * __restrict__ count,
                                   ಧ್ರುವ int64_t n_kv,
                                   ಧ್ರುವ int64_t n_kv_g) {
    constexpr size_t WG = 256;
    ಧ್ರುವ size_t global = (size_t) GGML_PAD(n_kv, (int64_t) WG);

    stream->parallel_for(
        sycl::nd_range<1>(sycl::range<1>(global), sycl::range<1>(WG)),
        [=](sycl::nd_item<1> item) {
            ಧ್ರುವ int64_t i = (int64_t) item.get_global_id(0);
            ಯದಿ (i >= n_kv || !sycl::isfinite((ಭಿನ್ನ) mask[i])) {
                ಪ್ರತ್ಯಾಗಮ;
            }

            sycl::atomic_ref<int32_t,
                             sycl::memory_order::relaxed,
                             sycl::memory_scope::device,
                             sycl::access::address_space::global_space> ctr(*count);

            ಧ್ರುವ int32_t pos = ctr.fetch_add(1);
            ಯದಿ (pos < (int32_t) n_kv_g) {
                indices[pos] = (int32_t) i;
            }
        });
}

// Rows along ne[0] are contiguous for every type used as a KV cache, so this is
// a plain byte copy and needs no per-type code. Padding slots are zeroed.
ಸ್ಥಿರ ಶೂನ್ಯ sparse_fa_gather_rows(sycl::queue * stream,
                                  ಧ್ರುವ uint8_t * __restrict__ src,
                                  uint8_t * __restrict__ dst,
                                  ಧ್ರುವ int32_t * __restrict__ indices,
                                  ಧ್ರುವ int32_t * __restrict__ count,
                                  ಧ್ರುವ size_t row_size,
                                  ಧ್ರುವ size_t src_nb1,
                                  ಧ್ರುವ size_t src_nb2,
                                  ಧ್ರುವ int64_t n_kv_g,
                                  ಧ್ರುವ int64_t n_head) {
    GGML_ASSERT(row_size % ಪರಿಮಾಣ(uint32_t) == 0);
    ಧ್ರುವ size_t ಪದಗಳು = row_size / ಪರಿಮಾಣ(uint32_t);

    stream->parallel_for(
        sycl::range<3>((size_t) n_head, (size_t) n_kv_g, ಪದಗಳು),
        [=](sycl::id<3> id) {
            ಧ್ರುವ int64_t h    = (int64_t) id[0];
            ಧ್ರುವ int64_t slot = (int64_t) id[1];
            ಧ್ರುವ size_t  w    = id[2];

            uint32_t * dst_row =
                (uint32_t *) (dst + ((size_t) (h * n_kv_g + slot)) * row_size);

            ಯದಿ (slot >= (int64_t) *count) {
                dst_row[w] = 0;
                ಪ್ರತ್ಯಾಗಮ;
            }

            ಧ್ರುವ uint32_t * src_row =
                (ಧ್ರುವ uint32_t *) (src + (size_t) indices[slot] * src_nb1 +
                                    (size_t) h * src_nb2);
            dst_row[w] = src_row[w];
        });
}

ಸ್ಥಿರ ಶೂನ್ಯ sparse_fa_gather_mask(sycl::queue * stream,
                                  ಧ್ರುವ sycl::half * __restrict__ mask,
                                  sycl::half * __restrict__ mask_g,
                                  ಧ್ರುವ int32_t * __restrict__ indices,
                                  ಧ್ರುವ int32_t * __restrict__ count,
                                  ಧ್ರುವ int64_t n_kv_g,
                                  ಧ್ರುವ int64_t n_rows,
                                  ಧ್ರುವ size_t mask_s1) {
    stream->parallel_for(
        sycl::range<2>((size_t) n_rows, (size_t) n_kv_g),
        [=](sycl::id<2> id) {
            ಧ್ರುವ int64_t r    = (int64_t) id[0];
            ಧ್ರುವ int64_t slot = (int64_t) id[1];

            sycl::half v = sycl::half(-INFINITY);
            ಯದಿ (slot < (int64_t) *count) {
                v = mask[(size_t) r * mask_s1 + (size_t) indices[slot]];
            }
            mask_g[(size_t) r * n_kv_g + slot] = v;
        });
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ sparse_fa_applicable(ಧ್ರುವ ggml_tensor * dst, int64_t & n_kv_g_out) {
    ಧ್ರುವ ggml_tensor * Q    = dst->src[0];
    ಧ್ರುವ ggml_tensor * K    = dst->src[1];
    ಧ್ರುವ ggml_tensor * V    = dst->src[2];
    ಧ್ರುವ ggml_tensor * mask = dst->src[3];

    ಯದಿ (!Q || !K || !V || !mask) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }

    ಧ್ರುವ int32_t n_kv_max = ggml_get_op_params_i32(dst, 4);
    ಯದಿ (n_kv_max <= 0) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }

    ಭಿನ್ನ max_bias      = 0.0f;
    ಭಿನ್ನ logit_softcap = 0.0f;
    memcpy(&max_bias,      (ಧ್ರುವ ಭಿನ್ನ *) dst->op_params + 1, ಪರಿಮಾಣ(ಭಿನ್ನ));
    memcpy(&logit_softcap, (ಧ್ರುವ ಭಿನ್ನ *) dst->op_params + 2, ಪರಿಮಾಣ(ಭಿನ್ನ));
    ಯದಿ (max_bias != 0.0f || logit_softcap != 0.0f) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }

    // single-token decode only; prefill amortises the scan already
    ಯದಿ (Q->ne[1] != 1) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }
    ಯದಿ (K->ne[3] != 1 || V->ne[3] != 1 || mask->ne[2] != 1 || mask->ne[3] != 1) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }
    ಯದಿ (mask->type != GGML_TYPE_F16 || mask->ne[0] < K->ne[1]) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }
    ಯದಿ (K->ne[2] != V->ne[2]) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }

    // nb[1] may stride over heads (interleaved cache); only ne[0] must be contiguous
    ಯದಿ (K->nb[0] != ggml_type_size(K->type) || V->nb[0] != ggml_type_size(V->type)) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }

    ಧ್ರುವ size_t k_row = ggml_row_size(K->type, K->ne[0]);
    ಧ್ರುವ size_t v_row = ggml_row_size(V->type, V->ne[0]);
    ಯದಿ (k_row % ಪರಿಮಾಣ(uint32_t) || v_row % ಪರಿಮಾಣ(uint32_t)) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }

    ಧ್ರುವ int64_t n_kv_g = GGML_PAD((int64_t) n_kv_max + sparse_fa_margin(), SPARSE_FA_PAD);
    ಯದಿ (n_kv_g * SPARSE_FA_MIN_RATIO > K->ne[1]) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }

    n_kv_g_out = n_kv_g;
    ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
}

ತಾರ್ಕಿಕ ggml_sycl_flash_attn_ext_sparse(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    int64_t n_kv_g = 0;
    ಯದಿ (!sparse_fa_enabled() || !sparse_fa_applicable(dst, n_kv_g)) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }

    ggml_tensor * K    = dst->src[1];
    ggml_tensor * V    = dst->src[2];
    ggml_tensor * mask = dst->src[3];

    ಧ್ರುವ int64_t n_kv     = K->ne[1];
    ಧ್ರುವ int64_t n_head_k = K->ne[2];
    ಧ್ರುವ int64_t n_rows_m = mask->ne[1];

    ಧ್ರುವ size_t k_row = ggml_row_size(K->type, K->ne[0]);
    ಧ್ರುವ size_t v_row = ggml_row_size(V->type, V->ne[0]);

    dpct::queue_ptr stream = ctx.stream();

    ggml_sycl_pool_alloc<int32_t>    idx_alloc(ctx.pool(), (size_t) n_kv_g);
    ggml_sycl_pool_alloc<int32_t>    cnt_alloc(ctx.pool(), 1);
    ggml_sycl_pool_alloc<uint8_t>    k_alloc(ctx.pool(), (size_t) n_head_k * n_kv_g * k_row);
    ggml_sycl_pool_alloc<uint8_t>    v_alloc(ctx.pool(), (size_t) n_head_k * n_kv_g * v_row);
    ggml_sycl_pool_alloc<sycl::half> m_alloc(ctx.pool(), (size_t) n_rows_m * n_kv_g);

    int32_t *    d_idx  = idx_alloc.get();
    int32_t *    d_cnt  = cnt_alloc.get();
    uint8_t *    d_K    = k_alloc.get();
    uint8_t *    d_V    = v_alloc.get();
    sycl::half * d_mask = m_alloc.get();

    // reset with a kernel: on Level Zero a queue memset is not always ordered before the next kernel
    stream->single_task([=]() { *d_cnt = 0; });

    sparse_fa_compact_mask(stream, (ಧ್ರುವ sycl::half *) mask->data,
                           d_idx, d_cnt, n_kv, n_kv_g);

    sparse_fa_gather_rows(stream, (ಧ್ರುವ uint8_t *) K->data, d_K, d_idx, d_cnt,
                          k_row, K->nb[1], K->nb[2], n_kv_g, n_head_k);

    sparse_fa_gather_rows(stream, (ಧ್ರುವ uint8_t *) V->data, d_V, d_idx, d_cnt,
                          v_row, V->nb[1], V->nb[2], n_kv_g, n_head_k);

    sparse_fa_gather_mask(stream, (ಧ್ರುವ sycl::half *) mask->data, d_mask,
                          d_idx, d_cnt, n_kv_g, n_rows_m,
                          mask->nb[1] / ಪರಿಮಾಣ(sycl::half));

    ಯದಿ (sparse_fa_debug()) {
        int32_t h_cnt = 0;
        SYCL_CHECK(CHECK_TRY_ERROR(stream->memcpy(&h_cnt, d_cnt, ಪರಿಮಾಣ(int32_t))));
        SYCL_CHECK(CHECK_TRY_ERROR(stream->wait()));
        fprintf(stderr, "[FA-SPARSE] n_kv=%lld n_kv_max=%d n_kv_g=%lld finite=%d%s\n",
                (long long) n_kv, ggml_get_op_params_i32(dst, 4),
                (long long) n_kv_g, (ಪೂರ್ಣಾಙ್ಕ) h_cnt,
                h_cnt > (int32_t) n_kv_g ? "  OVERFLOW" : "");
    }

    // shallow copies retargeted at the gathered buffers; kernels are unchanged
    ggml_tensor K_g = *K;
    K_g.data      = d_K;
    K_g.ne[1]     = n_kv_g;
    K_g.nb[1]     = k_row;
    K_g.nb[2]     = (size_t) n_kv_g * k_row;
    K_g.nb[3]     = (size_t) n_head_k * n_kv_g * k_row;
    K_g.view_src  = nullptr;
    K_g.view_offs = 0;

    ggml_tensor V_g = *V;
    V_g.data      = d_V;
    V_g.ne[1]     = n_kv_g;
    V_g.nb[1]     = v_row;
    V_g.nb[2]     = (size_t) n_kv_g * v_row;
    V_g.nb[3]     = (size_t) V->ne[2] * n_kv_g * v_row;
    V_g.view_src  = nullptr;
    V_g.view_offs = 0;

    ggml_tensor M_g = *mask;
    M_g.data      = d_mask;
    M_g.ne[0]     = n_kv_g;
    M_g.nb[1]     = (size_t) n_kv_g * ಪರಿಮಾಣ(sycl::half);
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

    ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
}
