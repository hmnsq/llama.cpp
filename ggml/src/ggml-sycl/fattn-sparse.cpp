#ضمن "fattn.hpp"
#ضمن "fattn-sparse.hpp"

#ضمن <cstdint>
#ضمن <cstdio>
#ضمن <cstring>

ساكن constexpr int64_t SPARSE_FA_PAD       = 256;
ساكن constexpr int64_t SPARSE_FA_MIN_نسبت = 2;

خارجي صحيح g_ggml_sycl_enable_sparse_fa;
خارجي صحيح g_ggml_sycl_debug_sparse_fa;
خارجي صحيح g_ggml_sycl_sparse_fa_margin;

ساكن صحيح sparse_fa_enabled(فراغ) {
    أرجع g_ggml_sycl_enable_sparse_fa;
}

ساكن صحيح sparse_fa_debug(فراغ) {
    أرجع g_ggml_sycl_debug_sparse_fa;
}

// slack above n_kv_max; callers may exceed the hint by a few always-attended positions
ساكن صحيح sparse_fa_margin(فراغ) {
    أرجع g_ggml_sycl_sparse_fa_margin;
}

// Unordered output is fine: softmax over the selected set is permutation invariant.
ساكن فراغ sparse_fa_compact_ماسک(sycl::queue * stream,
                                   ثابت sycl::half * __restrict__ ماسک,
                                   int32_t * __restrict__ indices,
                                   int32_t * __restrict__ count,
                                   ثابت int64_t n_kv,
                                   ثابت int64_t n_kv_g) {
    constexpr size_t WG = 256;
    ثابت size_t global = (size_t) GGML_PAD(n_kv, (int64_t) WG);

    stream->parallel_for(
        sycl::nd_range<1>(sycl::range<1>(global), sycl::range<1>(WG)),
        [=](sycl::nd_قلم<1> قلم) {
            ثابت int64_t i = (int64_t) قلم.get_global_id(0);
            إذا (i >= n_kv || !sycl::isfinite((عائم) ماسک[i])) {
                أرجع;
            }

            sycl::atomic_ref<int32_t,
                             sycl::memory_order::relaxed,
                             sycl::حافظه_scope::device,
                             sycl::access::آدرس_space::global_space> ctr(*count);

            ثابت int32_t pos = ctr.fetch_add(1);
            إذا (pos < (int32_t) n_kv_g) {
                indices[pos] = (int32_t) i;
            }
        });
}

// Rows along ne[0] are contiguous for every type used as a KV cache, so this is
// a plain byte copy and needs no per-type code. Padding slots are zeroed.
ساكن فراغ sparse_fa_gather_ردیفها(sycl::queue * stream,
                                  ثابت uint8_t * __restrict__ src,
                                  uint8_t * __restrict__ dst,
                                  ثابت int32_t * __restrict__ indices,
                                  ثابت int32_t * __restrict__ count,
                                  ثابت size_t ردیف_اندازه,
                                  ثابت size_t src_nb1,
                                  ثابت size_t src_nb2,
                                  ثابت int64_t n_kv_g,
                                  ثابت int64_t n_سر) {
    GGML_ASSERT(ردیف_اندازه % حجم(uint32_t) == 0);
    ثابت size_t واژگان = ردیف_اندازه / حجم(uint32_t);

    stream->parallel_for(
        sycl::range<3>((size_t) n_سر, (size_t) n_kv_g, واژگان),
        [=](sycl::id<3> id) {
            ثابت int64_t h    = (int64_t) id[0];
            ثابت int64_t جایگاه = (int64_t) id[1];
            ثابت size_t  w    = id[2];

            uint32_t * dst_ردیف =
                (uint32_t *) (dst + ((size_t) (h * n_kv_g + جایگاه)) * ردیف_اندازه);

            إذا (جایگاه >= (int64_t) *count) {
                dst_ردیف[w] = 0;
                أرجع;
            }

            ثابت uint32_t * src_ردیف =
                (ثابت uint32_t *) (src + (size_t) indices[جایگاه] * src_nb1 +
                                    (size_t) h * src_nb2);
            dst_ردیف[w] = src_ردیف[w];
        });
}

ساكن فراغ sparse_fa_gather_ماسک(sycl::queue * stream,
                                  ثابت sycl::half * __restrict__ ماسک,
                                  sycl::half * __restrict__ ماسک_g,
                                  ثابت int32_t * __restrict__ indices,
                                  ثابت int32_t * __restrict__ count,
                                  ثابت int64_t n_kv_g,
                                  ثابت int64_t n_ردیفها,
                                  ثابت size_t ماسک_s1) {
    stream->parallel_for(
        sycl::range<2>((size_t) n_ردیفها, (size_t) n_kv_g),
        [=](sycl::id<2> id) {
            ثابت int64_t r    = (int64_t) id[0];
            ثابت int64_t جایگاه = (int64_t) id[1];

            sycl::half v = sycl::half(-INFINITY);
            إذا (جایگاه < (int64_t) *count) {
                v = ماسک[(size_t) r * ماسک_s1 + (size_t) indices[جایگاه]];
            }
            ماسک_g[(size_t) r * n_kv_g + جایگاه] = v;
        });
}

ساكن منطقي sparse_fa_applicable(ثابت ggml_tensor * dst, int64_t & n_kv_g_بیرون) {
    ثابت ggml_tensor * Q    = dst->src[0];
    ثابت ggml_tensor * K    = dst->src[1];
    ثابت ggml_tensor * V    = dst->src[2];
    ثابت ggml_tensor * ماسک = dst->src[3];

    إذا (!Q || !K || !V || !ماسک) {
        أرجع خطأ;
    }

    ثابت int32_t n_kv_max = ggml_get_op_params_i32(dst, 4);
    إذا (n_kv_max <= 0) {
        أرجع خطأ;
    }

    عائم max_bias      = 0.0f;
    عائم logit_softcap = 0.0f;
    کپی_حافظه(&max_bias,      (ثابت عائم *) dst->op_params + 1, حجم(عائم));
    کپی_حافظه(&logit_softcap, (ثابت عائم *) dst->op_params + 2, حجم(عائم));
    إذا (max_bias != 0.0f || logit_softcap != 0.0f) {
        أرجع خطأ;
    }

    // single-token decode only; prefill amortises the scan already
    إذا (Q->ne[1] != 1) {
        أرجع خطأ;
    }
    إذا (K->ne[3] != 1 || V->ne[3] != 1 || ماسک->ne[2] != 1 || ماسک->ne[3] != 1) {
        أرجع خطأ;
    }
    إذا (ماسک->type != GGML_TYPE_F16 || ماسک->ne[0] < K->ne[1]) {
        أرجع خطأ;
    }
    إذا (K->ne[2] != V->ne[2]) {
        أرجع خطأ;
    }

    // nb[1] may stride over heads (interleaved cache); only ne[0] must be contiguous
    إذا (K->nb[0] != ggml_type_اندازه(K->type) || V->nb[0] != ggml_type_اندازه(V->type)) {
        أرجع خطأ;
    }

    ثابت size_t k_ردیف = ggml_ردیف_اندازه(K->type, K->ne[0]);
    ثابت size_t v_ردیف = ggml_ردیف_اندازه(V->type, V->ne[0]);
    إذا (k_ردیف % حجم(uint32_t) || v_ردیف % حجم(uint32_t)) {
        أرجع خطأ;
    }

    ثابت int64_t n_kv_g = GGML_PAD((int64_t) n_kv_max + sparse_fa_margin(), SPARSE_FA_PAD);
    إذا (n_kv_g * SPARSE_FA_MIN_نسبت > K->ne[1]) {
        أرجع خطأ;
    }

    n_kv_g_بیرون = n_kv_g;
    أرجع درست;
}

منطقي ggml_sycl_flash_attn_ext_sparse(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    int64_t n_kv_g = 0;
    إذا (!sparse_fa_enabled() || !sparse_fa_applicable(dst, n_kv_g)) {
        أرجع خطأ;
    }

    ggml_tensor * K    = dst->src[1];
    ggml_tensor * V    = dst->src[2];
    ggml_tensor * ماسک = dst->src[3];

    ثابت int64_t n_kv     = K->ne[1];
    ثابت int64_t n_سر_k = K->ne[2];
    ثابت int64_t n_ردیفها_m = ماسک->ne[1];

    ثابت size_t k_ردیف = ggml_ردیف_اندازه(K->type, K->ne[0]);
    ثابت size_t v_ردیف = ggml_ردیف_اندازه(V->type, V->ne[0]);

    dpct::صف_ptr stream = ctx.stream();

    ggml_sycl_pool_alloc<int32_t>    idx_alloc(ctx.pool(), (size_t) n_kv_g);
    ggml_sycl_pool_alloc<int32_t>    cnt_alloc(ctx.pool(), 1);
    ggml_sycl_pool_alloc<uint8_t>    k_alloc(ctx.pool(), (size_t) n_سر_k * n_kv_g * k_ردیف);
    ggml_sycl_pool_alloc<uint8_t>    v_alloc(ctx.pool(), (size_t) n_سر_k * n_kv_g * v_ردیف);
    ggml_sycl_pool_alloc<sycl::half> m_alloc(ctx.pool(), (size_t) n_ردیفها_m * n_kv_g);

    int32_t *    d_idx  = idx_alloc.get();
    int32_t *    d_cnt  = cnt_alloc.get();
    uint8_t *    d_K    = k_alloc.get();
    uint8_t *    d_V    = v_alloc.get();
    sycl::half * d_ماسک = m_alloc.get();

    // reset with a kernel: on Level Zero a queue memset is not always ordered before the next kernel
    stream->single_task([=]() { *d_cnt = 0; });

    sparse_fa_compact_ماسک(stream, (ثابت sycl::half *) ماسک->داده,
                           d_idx, d_cnt, n_kv, n_kv_g);

    sparse_fa_gather_ردیفها(stream, (ثابت uint8_t *) K->داده, d_K, d_idx, d_cnt,
                          k_ردیف, K->nb[1], K->nb[2], n_kv_g, n_سر_k);

    sparse_fa_gather_ردیفها(stream, (ثابت uint8_t *) V->داده, d_V, d_idx, d_cnt,
                          v_ردیف, V->nb[1], V->nb[2], n_kv_g, n_سر_k);

    sparse_fa_gather_ماسک(stream, (ثابت sycl::half *) ماسک->داده, d_ماسک,
                          d_idx, d_cnt, n_kv_g, n_ردیفها_m,
                          ماسک->nb[1] / حجم(sycl::half));

    إذا (sparse_fa_debug()) {
        int32_t h_cnt = 0;
        SYCL_CHECK(CHECK_TRY_ERROR(stream->کپی_حافظه(&h_cnt, d_cnt, حجم(int32_t))));
        SYCL_CHECK(CHECK_TRY_ERROR(stream->wait()));
        fprintf(stderr, "[FA-SPARSE] n_kv=%lld n_kv_max=%d n_kv_g=%lld finite=%d%s\n",
                (دراز دراز) n_kv, ggml_get_op_params_i32(dst, 4),
                (دراز دراز) n_kv_g, (صحيح) h_cnt,
                h_cnt > (int32_t) n_kv_g ? "  OVERFLOW" : "");
    }

    // shallow copies retargeted at the gathered buffers; kernels are unchanged
    ggml_tensor K_g = *K;
    K_g.داده      = d_K;
    K_g.ne[1]     = n_kv_g;
    K_g.nb[1]     = k_ردیف;
    K_g.nb[2]     = (size_t) n_kv_g * k_ردیف;
    K_g.nb[3]     = (size_t) n_سر_k * n_kv_g * k_ردیف;
    K_g.view_src  = اشاره_تهی;
    K_g.view_offs = 0;

    ggml_tensor V_g = *V;
    V_g.داده      = d_V;
    V_g.ne[1]     = n_kv_g;
    V_g.nb[1]     = v_ردیف;
    V_g.nb[2]     = (size_t) n_kv_g * v_ردیف;
    V_g.nb[3]     = (size_t) V->ne[2] * n_kv_g * v_ردیف;
    V_g.view_src  = اشاره_تهی;
    V_g.view_offs = 0;

    ggml_tensor M_g = *ماسک;
    M_g.داده      = d_ماسک;
    M_g.ne[0]     = n_kv_g;
    M_g.nb[1]     = (size_t) n_kv_g * حجم(sycl::half);
    M_g.nb[2]     = M_g.nb[1] * ماسک->ne[1];
    M_g.nb[3]     = M_g.nb[2];
    M_g.view_src  = اشاره_تهی;
    M_g.view_offs = 0;

    ggml_tensor dst_g = *dst;
    dst_g.src[1] = &K_g;
    dst_g.src[2] = &V_g;
    dst_g.src[3] = &M_g;
    dst_g.op_params[4] = 0;   // avoid re-entering this path

    ggml_sycl_flash_attn_ext(ctx, &dst_g);

    أرجع درست;
}
