#شامل "fattn.hpp"
#شامل "fattn-sparse.hpp"

#شامل <cstdint>
#شامل <cstdio>
#شامل <cstring>

ساکن constexpr int64_t SPARSE_FA_PAD       = 256;
ساکن constexpr int64_t SPARSE_FA_MIN_تناسب = 2;

خارجی صحیح g_ggml_sycl_enable_sparse_fa;
خارجی صحیح g_ggml_sycl_debug_sparse_fa;
خارجی صحیح g_ggml_sycl_sparse_fa_margin;

ساکن صحیح sparse_fa_enabled(خالی) {
    واپس g_ggml_sycl_enable_sparse_fa;
}

ساکن صحیح sparse_fa_debug(خالی) {
    واپس g_ggml_sycl_debug_sparse_fa;
}

// slack above n_kv_max; callers may exceed the hint by a few always-attended positions
ساکن صحیح sparse_fa_margin(خالی) {
    واپس g_ggml_sycl_sparse_fa_margin;
}

// Unordered output is fine: softmax over the selected set is permutation invariant.
ساکن خالی sparse_fa_compact_نقاب(sycl::queue * stream,
                                   ثابت sycl::half * __restrict__ نقاب,
                                   int32_t * __restrict__ indices,
                                   int32_t * __restrict__ count,
                                   ثابت int64_t n_kv,
                                   ثابت int64_t n_kv_g) {
    constexpr size_t WG = 256;
    ثابت size_t global = (size_t) GGML_PAD(n_kv, (int64_t) WG);

    stream->parallel_for(
        sycl::nd_range<1>(sycl::range<1>(global), sycl::range<1>(WG)),
        [=](sycl::nd_شے<1> شے) {
            ثابت int64_t i = (int64_t) شے.get_global_id(0);
            اگر (i >= n_kv || !sycl::isfinite((اعشاری) نقاب[i])) {
                واپس;
            }

            sycl::atomic_ref<int32_t,
                             sycl::memory_order::relaxed,
                             sycl::حافظہ_scope::device,
                             sycl::access::پتہ_space::global_space> ctr(*count);

            ثابت int32_t pos = ctr.fetch_add(1);
            اگر (pos < (int32_t) n_kv_g) {
                indices[pos] = (int32_t) i;
            }
        });
}

// Rows along ne[0] are contiguous for every type used as a KV cache, so this is
// a plain byte copy and needs no per-type code. Padding slots are zeroed.
ساکن خالی sparse_fa_gather_صفیں(sycl::queue * stream,
                                  ثابت uint8_t * __restrict__ src,
                                  uint8_t * __restrict__ dst,
                                  ثابت int32_t * __restrict__ indices,
                                  ثابت int32_t * __restrict__ count,
                                  ثابت size_t صف_سائز,
                                  ثابت size_t src_nb1,
                                  ثابت size_t src_nb2,
                                  ثابت int64_t n_kv_g,
                                  ثابت int64_t n_سر) {
    GGML_ASSERT(صف_سائز % حجم(uint32_t) == 0);
    ثابت size_t الفاظ = صف_سائز / حجم(uint32_t);

    stream->parallel_for(
        sycl::range<3>((size_t) n_سر, (size_t) n_kv_g, الفاظ),
        [=](sycl::id<3> id) {
            ثابت int64_t h    = (int64_t) id[0];
            ثابت int64_t خانہ = (int64_t) id[1];
            ثابت size_t  w    = id[2];

            uint32_t * dst_صف =
                (uint32_t *) (dst + ((size_t) (h * n_kv_g + خانہ)) * صف_سائز);

            اگر (خانہ >= (int64_t) *count) {
                dst_صف[w] = 0;
                واپس;
            }

            ثابت uint32_t * src_صف =
                (ثابت uint32_t *) (src + (size_t) indices[خانہ] * src_nb1 +
                                    (size_t) h * src_nb2);
            dst_صف[w] = src_صف[w];
        });
}

ساکن خالی sparse_fa_gather_نقاب(sycl::queue * stream,
                                  ثابت sycl::half * __restrict__ نقاب,
                                  sycl::half * __restrict__ نقاب_g,
                                  ثابت int32_t * __restrict__ indices,
                                  ثابت int32_t * __restrict__ count,
                                  ثابت int64_t n_kv_g,
                                  ثابت int64_t n_صفیں,
                                  ثابت size_t نقاب_s1) {
    stream->parallel_for(
        sycl::range<2>((size_t) n_صفیں, (size_t) n_kv_g),
        [=](sycl::id<2> id) {
            ثابت int64_t r    = (int64_t) id[0];
            ثابت int64_t خانہ = (int64_t) id[1];

            sycl::half v = sycl::half(-INFINITY);
            اگر (خانہ < (int64_t) *count) {
                v = نقاب[(size_t) r * نقاب_s1 + (size_t) indices[خانہ]];
            }
            نقاب_g[(size_t) r * n_kv_g + خانہ] = v;
        });
}

ساکن منطقی sparse_fa_applicable(ثابت ggml_tensor * dst, int64_t & n_kv_g_برآمد) {
    ثابت ggml_tensor * Q    = dst->src[0];
    ثابت ggml_tensor * K    = dst->src[1];
    ثابت ggml_tensor * V    = dst->src[2];
    ثابت ggml_tensor * نقاب = dst->src[3];

    اگر (!Q || !K || !V || !نقاب) {
        واپس جھوٹ;
    }

    ثابت int32_t n_kv_max = ggml_get_op_params_i32(dst, 4);
    اگر (n_kv_max <= 0) {
        واپس جھوٹ;
    }

    اعشاری max_bias      = 0.0f;
    اعشاری logit_softcap = 0.0f;
    حافظہ_نقل(&max_bias,      (ثابت اعشاری *) dst->op_params + 1, حجم(اعشاری));
    حافظہ_نقل(&logit_softcap, (ثابت اعشاری *) dst->op_params + 2, حجم(اعشاری));
    اگر (max_bias != 0.0f || logit_softcap != 0.0f) {
        واپس جھوٹ;
    }

    // single-token decode only; prefill amortises the scan already
    اگر (Q->ne[1] != 1) {
        واپس جھوٹ;
    }
    اگر (K->ne[3] != 1 || V->ne[3] != 1 || نقاب->ne[2] != 1 || نقاب->ne[3] != 1) {
        واپس جھوٹ;
    }
    اگر (نقاب->type != GGML_TYPE_F16 || نقاب->ne[0] < K->ne[1]) {
        واپس جھوٹ;
    }
    اگر (K->ne[2] != V->ne[2]) {
        واپس جھوٹ;
    }

    // nb[1] may stride over heads (interleaved cache); only ne[0] must be contiguous
    اگر (K->nb[0] != ggml_type_سائز(K->type) || V->nb[0] != ggml_type_سائز(V->type)) {
        واپس جھوٹ;
    }

    ثابت size_t k_صف = ggml_صف_سائز(K->type, K->ne[0]);
    ثابت size_t v_صف = ggml_صف_سائز(V->type, V->ne[0]);
    اگر (k_صف % حجم(uint32_t) || v_صف % حجم(uint32_t)) {
        واپس جھوٹ;
    }

    ثابت int64_t n_kv_g = GGML_PAD((int64_t) n_kv_max + sparse_fa_margin(), SPARSE_FA_PAD);
    اگر (n_kv_g * SPARSE_FA_MIN_تناسب > K->ne[1]) {
        واپس جھوٹ;
    }

    n_kv_g_برآمد = n_kv_g;
    واپس سچ;
}

منطقی ggml_sycl_flash_attn_ext_sparse(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    int64_t n_kv_g = 0;
    اگر (!sparse_fa_enabled() || !sparse_fa_applicable(dst, n_kv_g)) {
        واپس جھوٹ;
    }

    ggml_tensor * K    = dst->src[1];
    ggml_tensor * V    = dst->src[2];
    ggml_tensor * نقاب = dst->src[3];

    ثابت int64_t n_kv     = K->ne[1];
    ثابت int64_t n_سر_k = K->ne[2];
    ثابت int64_t n_صفیں_m = نقاب->ne[1];

    ثابت size_t k_صف = ggml_صف_سائز(K->type, K->ne[0]);
    ثابت size_t v_صف = ggml_صف_سائز(V->type, V->ne[0]);

    dpct::قطار_ptr stream = ctx.stream();

    ggml_sycl_pool_alloc<int32_t>    idx_alloc(ctx.pool(), (size_t) n_kv_g);
    ggml_sycl_pool_alloc<int32_t>    cnt_alloc(ctx.pool(), 1);
    ggml_sycl_pool_alloc<uint8_t>    k_alloc(ctx.pool(), (size_t) n_سر_k * n_kv_g * k_صف);
    ggml_sycl_pool_alloc<uint8_t>    v_alloc(ctx.pool(), (size_t) n_سر_k * n_kv_g * v_صف);
    ggml_sycl_pool_alloc<sycl::half> m_alloc(ctx.pool(), (size_t) n_صفیں_m * n_kv_g);

    int32_t *    d_idx  = idx_alloc.get();
    int32_t *    d_cnt  = cnt_alloc.get();
    uint8_t *    d_K    = k_alloc.get();
    uint8_t *    d_V    = v_alloc.get();
    sycl::half * d_نقاب = m_alloc.get();

    // reset with a kernel: on Level Zero a queue memset is not always ordered before the next kernel
    stream->single_task([=]() { *d_cnt = 0; });

    sparse_fa_compact_نقاب(stream, (ثابت sycl::half *) نقاب->ڈیٹا,
                           d_idx, d_cnt, n_kv, n_kv_g);

    sparse_fa_gather_صفیں(stream, (ثابت uint8_t *) K->ڈیٹا, d_K, d_idx, d_cnt,
                          k_صف, K->nb[1], K->nb[2], n_kv_g, n_سر_k);

    sparse_fa_gather_صفیں(stream, (ثابت uint8_t *) V->ڈیٹا, d_V, d_idx, d_cnt,
                          v_صف, V->nb[1], V->nb[2], n_kv_g, n_سر_k);

    sparse_fa_gather_نقاب(stream, (ثابت sycl::half *) نقاب->ڈیٹا, d_نقاب,
                          d_idx, d_cnt, n_kv_g, n_صفیں_m,
                          نقاب->nb[1] / حجم(sycl::half));

    اگر (sparse_fa_debug()) {
        int32_t h_cnt = 0;
        SYCL_CHECK(CHECK_TRY_ERROR(stream->حافظہ_نقل(&h_cnt, d_cnt, حجم(int32_t))));
        SYCL_CHECK(CHECK_TRY_ERROR(stream->wait()));
        fprintf(stderr, "[FA-SPARSE] n_kv=%lld n_kv_max=%d n_kv_g=%lld finite=%d%s\n",
                (طویل طویل) n_kv, ggml_get_op_params_i32(dst, 4),
                (طویل طویل) n_kv_g, (صحیح) h_cnt,
                h_cnt > (int32_t) n_kv_g ? "  OVERFLOW" : "");
    }

    // shallow copies retargeted at the gathered buffers; kernels are unchanged
    ggml_tensor K_g = *K;
    K_g.ڈیٹا      = d_K;
    K_g.ne[1]     = n_kv_g;
    K_g.nb[1]     = k_صف;
    K_g.nb[2]     = (size_t) n_kv_g * k_صف;
    K_g.nb[3]     = (size_t) n_سر_k * n_kv_g * k_صف;
    K_g.view_src  = خالی_اشاریہ;
    K_g.view_offs = 0;

    ggml_tensor V_g = *V;
    V_g.ڈیٹا      = d_V;
    V_g.ne[1]     = n_kv_g;
    V_g.nb[1]     = v_صف;
    V_g.nb[2]     = (size_t) n_kv_g * v_صف;
    V_g.nb[3]     = (size_t) V->ne[2] * n_kv_g * v_صف;
    V_g.view_src  = خالی_اشاریہ;
    V_g.view_offs = 0;

    ggml_tensor M_g = *نقاب;
    M_g.ڈیٹا      = d_نقاب;
    M_g.ne[0]     = n_kv_g;
    M_g.nb[1]     = (size_t) n_kv_g * حجم(sycl::half);
    M_g.nb[2]     = M_g.nb[1] * نقاب->ne[1];
    M_g.nb[3]     = M_g.nb[2];
    M_g.view_src  = خالی_اشاریہ;
    M_g.view_offs = 0;

    ggml_tensor dst_g = *dst;
    dst_g.src[1] = &K_g;
    dst_g.src[2] = &V_g;
    dst_g.src[3] = &M_g;
    dst_g.op_params[4] = 0;   // avoid re-entering this path

    ggml_sycl_flash_attn_ext(ctx, &dst_g);

    واپس سچ;
}
