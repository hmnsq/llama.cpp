#અન્તર્ભાવ "fattn.hpp"
#અન્તર્ભાવ "fattn-sparse.hpp"

#અન્તર્ભાવ <cstdint>
#અન્તર્ભાવ <cstdio>
#અન્તર્ભાવ <cstring>

સ્થિર constexpr int64_t SPARSE_FA_PAD       = 256;
સ્થિર constexpr int64_t SPARSE_FA_MIN_RATIO = 2;

બાહ્ય પૂર્ણાંક g_ggml_sycl_enable_sparse_fa;
બાહ્ય પૂર્ણાંક g_ggml_sycl_debug_sparse_fa;
બાહ્ય પૂર્ણાંક g_ggml_sycl_sparse_fa_margin;

સ્થિર પૂર્ણાંક sparse_fa_enabled(શૂન્ય) {
    પાછા g_ggml_sycl_enable_sparse_fa;
}

સ્થિર પૂર્ણાંક sparse_fa_debug(શૂન્ય) {
    પાછા g_ggml_sycl_debug_sparse_fa;
}

// slack above n_kv_max; callers may exceed the hint by a few always-attended positions
સ્થિર પૂર્ણાંક sparse_fa_margin(શૂન્ય) {
    પાછા g_ggml_sycl_sparse_fa_margin;
}

// Unordered output is fine: softmax over the selected set is permutation invariant.
સ્થિર શૂન્ય sparse_fa_compact_mask(sycl::queue * stream,
                                   અચલ sycl::half * __restrict__ mask,
                                   int32_t * __restrict__ indices,
                                   int32_t * __restrict__ count,
                                   અચલ int64_t n_kv,
                                   અચલ int64_t n_kv_g) {
    constexpr size_t WG = 256;
    અચલ size_t global = (size_t) GGML_PAD(n_kv, (int64_t) WG);

    stream->parallel_for(
        sycl::nd_range<1>(sycl::range<1>(global), sycl::range<1>(WG)),
        [=](sycl::nd_item<1> item) {
            અચલ int64_t i = (int64_t) item.get_global_id(0);
            જો (i >= n_kv || !sycl::isfinite((દશાંશ) mask[i])) {
                પાછા;
            }

            sycl::atomic_ref<int32_t,
                             sycl::memory_order::relaxed,
                             sycl::memory_scope::device,
                             sycl::access::address_space::global_space> ctr(*count);

            અચલ int32_t pos = ctr.fetch_add(1);
            જો (pos < (int32_t) n_kv_g) {
                indices[pos] = (int32_t) i;
            }
        });
}

// Rows along ne[0] are contiguous for every type used as a KV cache, so this is
// a plain byte copy and needs no per-type code. Padding slots are zeroed.
સ્થિર શૂન્ય sparse_fa_gather_rows(sycl::queue * stream,
                                  અચલ uint8_t * __restrict__ src,
                                  uint8_t * __restrict__ dst,
                                  અચલ int32_t * __restrict__ indices,
                                  અચલ int32_t * __restrict__ count,
                                  અચલ size_t row_size,
                                  અચલ size_t src_nb1,
                                  અચલ size_t src_nb2,
                                  અચલ int64_t n_kv_g,
                                  અચલ int64_t n_head) {
    GGML_ASSERT(row_size % કદ(uint32_t) == 0);
    અચલ size_t શબ્દો = row_size / કદ(uint32_t);

    stream->parallel_for(
        sycl::range<3>((size_t) n_head, (size_t) n_kv_g, શબ્દો),
        [=](sycl::id<3> id) {
            અચલ int64_t h    = (int64_t) id[0];
            અચલ int64_t slot = (int64_t) id[1];
            અચલ size_t  w    = id[2];

            uint32_t * dst_row =
                (uint32_t *) (dst + ((size_t) (h * n_kv_g + slot)) * row_size);

            જો (slot >= (int64_t) *count) {
                dst_row[w] = 0;
                પાછા;
            }

            અચલ uint32_t * src_row =
                (અચલ uint32_t *) (src + (size_t) indices[slot] * src_nb1 +
                                    (size_t) h * src_nb2);
            dst_row[w] = src_row[w];
        });
}

સ્થિર શૂન્ય sparse_fa_gather_mask(sycl::queue * stream,
                                  અચલ sycl::half * __restrict__ mask,
                                  sycl::half * __restrict__ mask_g,
                                  અચલ int32_t * __restrict__ indices,
                                  અચલ int32_t * __restrict__ count,
                                  અચલ int64_t n_kv_g,
                                  અચલ int64_t n_rows,
                                  અચલ size_t mask_s1) {
    stream->parallel_for(
        sycl::range<2>((size_t) n_rows, (size_t) n_kv_g),
        [=](sycl::id<2> id) {
            અચલ int64_t r    = (int64_t) id[0];
            અચલ int64_t slot = (int64_t) id[1];

            sycl::half v = sycl::half(-INFINITY);
            જો (slot < (int64_t) *count) {
                v = mask[(size_t) r * mask_s1 + (size_t) indices[slot]];
            }
            mask_g[(size_t) r * n_kv_g + slot] = v;
        });
}

સ્થિર તાર્કિક sparse_fa_applicable(અચલ ggml_tensor * dst, int64_t & n_kv_g_out) {
    અચલ ggml_tensor * Q    = dst->src[0];
    અચલ ggml_tensor * K    = dst->src[1];
    અચલ ggml_tensor * V    = dst->src[2];
    અચલ ggml_tensor * mask = dst->src[3];

    જો (!Q || !K || !V || !mask) {
        પાછા ખોટું;
    }

    અચલ int32_t n_kv_max = ggml_get_op_params_i32(dst, 4);
    જો (n_kv_max <= 0) {
        પાછા ખોટું;
    }

    દશાંશ max_bias      = 0.0f;
    દશાંશ logit_softcap = 0.0f;
    memcpy(&max_bias,      (અચલ દશાંશ *) dst->op_params + 1, કદ(દશાંશ));
    memcpy(&logit_softcap, (અચલ દશાંશ *) dst->op_params + 2, કદ(દશાંશ));
    જો (max_bias != 0.0f || logit_softcap != 0.0f) {
        પાછા ખોટું;
    }

    // single-token decode only; prefill amortises the scan already
    જો (Q->ne[1] != 1) {
        પાછા ખોટું;
    }
    જો (K->ne[3] != 1 || V->ne[3] != 1 || mask->ne[2] != 1 || mask->ne[3] != 1) {
        પાછા ખોટું;
    }
    જો (mask->type != GGML_TYPE_F16 || mask->ne[0] < K->ne[1]) {
        પાછા ખોટું;
    }
    જો (K->ne[2] != V->ne[2]) {
        પાછા ખોટું;
    }

    // nb[1] may stride over heads (interleaved cache); only ne[0] must be contiguous
    જો (K->nb[0] != ggml_type_size(K->type) || V->nb[0] != ggml_type_size(V->type)) {
        પાછા ખોટું;
    }

    અચલ size_t k_row = ggml_row_size(K->type, K->ne[0]);
    અચલ size_t v_row = ggml_row_size(V->type, V->ne[0]);
    જો (k_row % કદ(uint32_t) || v_row % કદ(uint32_t)) {
        પાછા ખોટું;
    }

    અચલ int64_t n_kv_g = GGML_PAD((int64_t) n_kv_max + sparse_fa_margin(), SPARSE_FA_PAD);
    જો (n_kv_g * SPARSE_FA_MIN_RATIO > K->ne[1]) {
        પાછા ખોટું;
    }

    n_kv_g_out = n_kv_g;
    પાછા સાચું;
}

તાર્કિક ggml_sycl_flash_attn_ext_sparse(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    int64_t n_kv_g = 0;
    જો (!sparse_fa_enabled() || !sparse_fa_applicable(dst, n_kv_g)) {
        પાછા ખોટું;
    }

    ggml_tensor * K    = dst->src[1];
    ggml_tensor * V    = dst->src[2];
    ggml_tensor * mask = dst->src[3];

    અચલ int64_t n_kv     = K->ne[1];
    અચલ int64_t n_head_k = K->ne[2];
    અચલ int64_t n_rows_m = mask->ne[1];

    અચલ size_t k_row = ggml_row_size(K->type, K->ne[0]);
    અચલ size_t v_row = ggml_row_size(V->type, V->ne[0]);

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

    sparse_fa_compact_mask(stream, (અચલ sycl::half *) mask->data,
                           d_idx, d_cnt, n_kv, n_kv_g);

    sparse_fa_gather_rows(stream, (અચલ uint8_t *) K->data, d_K, d_idx, d_cnt,
                          k_row, K->nb[1], K->nb[2], n_kv_g, n_head_k);

    sparse_fa_gather_rows(stream, (અચલ uint8_t *) V->data, d_V, d_idx, d_cnt,
                          v_row, V->nb[1], V->nb[2], n_kv_g, n_head_k);

    sparse_fa_gather_mask(stream, (અચલ sycl::half *) mask->data, d_mask,
                          d_idx, d_cnt, n_kv_g, n_rows_m,
                          mask->nb[1] / કદ(sycl::half));

    જો (sparse_fa_debug()) {
        int32_t h_cnt = 0;
        SYCL_CHECK(CHECK_TRY_ERROR(stream->memcpy(&h_cnt, d_cnt, કદ(int32_t))));
        SYCL_CHECK(CHECK_TRY_ERROR(stream->wait()));
        fprintf(stderr, "[FA-SPARSE] n_kv=%lld n_kv_max=%d n_kv_g=%lld finite=%d%s\n",
                (long long) n_kv, ggml_get_op_params_i32(dst, 4),
                (long long) n_kv_g, (પૂર્ણાંક) h_cnt,
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
    M_g.nb[1]     = (size_t) n_kv_g * કદ(sycl::half);
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

    પાછા સાચું;
}
