#അന്തര്ഭാവ "fattn.hpp"
#അന്തര്ഭാവ "fattn-sparse.hpp"

#അന്തര്ഭാവ <cstdint>
#അന്തര്ഭാവ <cstdio>
#അന്തര്ഭാവ <cstring>

സ്ഥിതം constexpr int64_t SPARSE_FA_PAD       = 256;
സ്ഥിതം constexpr int64_t SPARSE_FA_MIN_RATIO = 2;

ബാഹ്യ പൂർണ്ണസംഖ്യ g_ggml_sycl_enable_sparse_fa;
ബാഹ്യ പൂർണ്ണസംഖ്യ g_ggml_sycl_debug_sparse_fa;
ബാഹ്യ പൂർണ്ണസംഖ്യ g_ggml_sycl_sparse_fa_margin;

സ്ഥിതം പൂർണ്ണസംഖ്യ sparse_fa_enabled(ശൂന്യം) {
    തിരികെ g_ggml_sycl_enable_sparse_fa;
}

സ്ഥിതം പൂർണ്ണസംഖ്യ sparse_fa_debug(ശൂന്യം) {
    തിരികെ g_ggml_sycl_debug_sparse_fa;
}

// slack above n_kv_max; callers may exceed the hint by a few always-attended positions
സ്ഥിതം പൂർണ്ണസംഖ്യ sparse_fa_margin(ശൂന്യം) {
    തിരികെ g_ggml_sycl_sparse_fa_margin;
}

// Unordered output is fine: softmax over the selected set is permutation invariant.
സ്ഥിതം ശൂന്യം sparse_fa_compact_mask(sycl::queue * stream,
                                   സ്ഥിരം sycl::half * __restrict__ mask,
                                   int32_t * __restrict__ indices,
                                   int32_t * __restrict__ count,
                                   സ്ഥിരം int64_t n_kv,
                                   സ്ഥിരം int64_t n_kv_g) {
    constexpr size_t WG = 256;
    സ്ഥിരം size_t global = (size_t) GGML_PAD(n_kv, (int64_t) WG);

    stream->parallel_for(
        sycl::nd_range<1>(sycl::range<1>(global), sycl::range<1>(WG)),
        [=](sycl::nd_item<1> item) {
            സ്ഥിരം int64_t i = (int64_t) item.get_global_id(0);
            എങ്കിൽ (i >= n_kv || !sycl::isfinite((ദശാംശം) mask[i])) {
                തിരികെ;
            }

            sycl::atomic_ref<int32_t,
                             sycl::memory_order::relaxed,
                             sycl::memory_scope::device,
                             sycl::access::address_space::global_space> ctr(*count);

            സ്ഥിരം int32_t pos = ctr.fetch_add(1);
            എങ്കിൽ (pos < (int32_t) n_kv_g) {
                indices[pos] = (int32_t) i;
            }
        });
}

// Rows along ne[0] are contiguous for every type used as a KV cache, so this is
// a plain byte copy and needs no per-type code. Padding slots are zeroed.
സ്ഥിതം ശൂന്യം sparse_fa_gather_rows(sycl::queue * stream,
                                  സ്ഥിരം uint8_t * __restrict__ src,
                                  uint8_t * __restrict__ dst,
                                  സ്ഥിരം int32_t * __restrict__ indices,
                                  സ്ഥിരം int32_t * __restrict__ count,
                                  സ്ഥിരം size_t row_size,
                                  സ്ഥിരം size_t src_nb1,
                                  സ്ഥിരം size_t src_nb2,
                                  സ്ഥിരം int64_t n_kv_g,
                                  സ്ഥിരം int64_t n_head) {
    GGML_ASSERT(row_size % വലുപ്പം(uint32_t) == 0);
    സ്ഥിരം size_t വാക്കുകൾ = row_size / വലുപ്പം(uint32_t);

    stream->parallel_for(
        sycl::range<3>((size_t) n_head, (size_t) n_kv_g, വാക്കുകൾ),
        [=](sycl::id<3> id) {
            സ്ഥിരം int64_t h    = (int64_t) id[0];
            സ്ഥിരം int64_t slot = (int64_t) id[1];
            സ്ഥിരം size_t  w    = id[2];

            uint32_t * dst_row =
                (uint32_t *) (dst + ((size_t) (h * n_kv_g + slot)) * row_size);

            എങ്കിൽ (slot >= (int64_t) *count) {
                dst_row[w] = 0;
                തിരികെ;
            }

            സ്ഥിരം uint32_t * src_row =
                (സ്ഥിരം uint32_t *) (src + (size_t) indices[slot] * src_nb1 +
                                    (size_t) h * src_nb2);
            dst_row[w] = src_row[w];
        });
}

സ്ഥിതം ശൂന്യം sparse_fa_gather_mask(sycl::queue * stream,
                                  സ്ഥിരം sycl::half * __restrict__ mask,
                                  sycl::half * __restrict__ mask_g,
                                  സ്ഥിരം int32_t * __restrict__ indices,
                                  സ്ഥിരം int32_t * __restrict__ count,
                                  സ്ഥിരം int64_t n_kv_g,
                                  സ്ഥിരം int64_t n_rows,
                                  സ്ഥിരം size_t mask_s1) {
    stream->parallel_for(
        sycl::range<2>((size_t) n_rows, (size_t) n_kv_g),
        [=](sycl::id<2> id) {
            സ്ഥിരം int64_t r    = (int64_t) id[0];
            സ്ഥിരം int64_t slot = (int64_t) id[1];

            sycl::half v = sycl::half(-INFINITY);
            എങ്കിൽ (slot < (int64_t) *count) {
                v = mask[(size_t) r * mask_s1 + (size_t) indices[slot]];
            }
            mask_g[(size_t) r * n_kv_g + slot] = v;
        });
}

സ്ഥിതം തർക്കം sparse_fa_applicable(സ്ഥിരം ggml_tensor * dst, int64_t & n_kv_g_out) {
    സ്ഥിരം ggml_tensor * Q    = dst->src[0];
    സ്ഥിരം ggml_tensor * K    = dst->src[1];
    സ്ഥിരം ggml_tensor * V    = dst->src[2];
    സ്ഥിരം ggml_tensor * mask = dst->src[3];

    എങ്കിൽ (!Q || !K || !V || !mask) {
        തിരികെ അസത്യം;
    }

    സ്ഥിരം int32_t n_kv_max = ggml_get_op_params_i32(dst, 4);
    എങ്കിൽ (n_kv_max <= 0) {
        തിരികെ അസത്യം;
    }

    ദശാംശം max_bias      = 0.0f;
    ദശാംശം logit_softcap = 0.0f;
    memcpy(&max_bias,      (സ്ഥിരം ദശാംശം *) dst->op_params + 1, വലുപ്പം(ദശാംശം));
    memcpy(&logit_softcap, (സ്ഥിരം ദശാംശം *) dst->op_params + 2, വലുപ്പം(ദശാംശം));
    എങ്കിൽ (max_bias != 0.0f || logit_softcap != 0.0f) {
        തിരികെ അസത്യം;
    }

    // single-token decode only; prefill amortises the scan already
    എങ്കിൽ (Q->ne[1] != 1) {
        തിരികെ അസത്യം;
    }
    എങ്കിൽ (K->ne[3] != 1 || V->ne[3] != 1 || mask->ne[2] != 1 || mask->ne[3] != 1) {
        തിരികെ അസത്യം;
    }
    എങ്കിൽ (mask->type != GGML_TYPE_F16 || mask->ne[0] < K->ne[1]) {
        തിരികെ അസത്യം;
    }
    എങ്കിൽ (K->ne[2] != V->ne[2]) {
        തിരികെ അസത്യം;
    }

    // nb[1] may stride over heads (interleaved cache); only ne[0] must be contiguous
    എങ്കിൽ (K->nb[0] != ggml_type_size(K->type) || V->nb[0] != ggml_type_size(V->type)) {
        തിരികെ അസത്യം;
    }

    സ്ഥിരം size_t k_row = ggml_row_size(K->type, K->ne[0]);
    സ്ഥിരം size_t v_row = ggml_row_size(V->type, V->ne[0]);
    എങ്കിൽ (k_row % വലുപ്പം(uint32_t) || v_row % വലുപ്പം(uint32_t)) {
        തിരികെ അസത്യം;
    }

    സ്ഥിരം int64_t n_kv_g = GGML_PAD((int64_t) n_kv_max + sparse_fa_margin(), SPARSE_FA_PAD);
    എങ്കിൽ (n_kv_g * SPARSE_FA_MIN_RATIO > K->ne[1]) {
        തിരികെ അസത്യം;
    }

    n_kv_g_out = n_kv_g;
    തിരികെ സത്യം;
}

തർക്കം ggml_sycl_flash_attn_ext_sparse(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    int64_t n_kv_g = 0;
    എങ്കിൽ (!sparse_fa_enabled() || !sparse_fa_applicable(dst, n_kv_g)) {
        തിരികെ അസത്യം;
    }

    ggml_tensor * K    = dst->src[1];
    ggml_tensor * V    = dst->src[2];
    ggml_tensor * mask = dst->src[3];

    സ്ഥിരം int64_t n_kv     = K->ne[1];
    സ്ഥിരം int64_t n_head_k = K->ne[2];
    സ്ഥിരം int64_t n_rows_m = mask->ne[1];

    സ്ഥിരം size_t k_row = ggml_row_size(K->type, K->ne[0]);
    സ്ഥിരം size_t v_row = ggml_row_size(V->type, V->ne[0]);

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

    sparse_fa_compact_mask(stream, (സ്ഥിരം sycl::half *) mask->data,
                           d_idx, d_cnt, n_kv, n_kv_g);

    sparse_fa_gather_rows(stream, (സ്ഥിരം uint8_t *) K->data, d_K, d_idx, d_cnt,
                          k_row, K->nb[1], K->nb[2], n_kv_g, n_head_k);

    sparse_fa_gather_rows(stream, (സ്ഥിരം uint8_t *) V->data, d_V, d_idx, d_cnt,
                          v_row, V->nb[1], V->nb[2], n_kv_g, n_head_k);

    sparse_fa_gather_mask(stream, (സ്ഥിരം sycl::half *) mask->data, d_mask,
                          d_idx, d_cnt, n_kv_g, n_rows_m,
                          mask->nb[1] / വലുപ്പം(sycl::half));

    എങ്കിൽ (sparse_fa_debug()) {
        int32_t h_cnt = 0;
        SYCL_CHECK(CHECK_TRY_ERROR(stream->memcpy(&h_cnt, d_cnt, വലുപ്പം(int32_t))));
        SYCL_CHECK(CHECK_TRY_ERROR(stream->wait()));
        fprintf(stderr, "[FA-SPARSE] n_kv=%lld n_kv_max=%d n_kv_g=%lld finite=%d%s\n",
                (long long) n_kv, ggml_get_op_params_i32(dst, 4),
                (long long) n_kv_g, (പൂർണ്ണസംഖ്യ) h_cnt,
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
    M_g.nb[1]     = (size_t) n_kv_g * വലുപ്പം(sycl::half);
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

    തിരികെ സത്യം;
}
