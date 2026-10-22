// Flash attention via oneMKL GEMM (XMX-accelerated).
// Uses column_major::gemm for Q*K^T and S*V matmuls
// with an online softmax SYCL kernel.
//
// All GQA query heads sharing a KV head are batched into single
// GEMM calls, amortizing MKL launch overhead across K and V reuse.
//

#ਅਨ੍ਤਰ੍ਭਾਵ "common.hpp"
#ਅਨ੍ਤਰ੍ਭਾਵ "fattn-common.hpp"
#ਅਨ੍ਤਰ੍ਭਾਵ "convert.hpp"
#ਅਨ੍ਤਰ੍ਭਾਵ "fattn.hpp"

#ਅਨ੍ਤਰ੍ਭਾਵ <oneapi/mkl.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <cstdio>
#ਅਨ੍ਤਰ੍ਭਾਵ <chrono>

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ MKL_FA_CHUNK_ਮਾਪ_KV 8192

// Number of query rows processed per tile. The score buffers (KQ_f32, S_f16)
// are sized q_tile_rows * chunk_size, so this bounds their footprint
// regardless of batch size (n_query_rows = n_queries * gqa_ratio). A typical
// single-ubatch prefill (e.g. ubatch 1024 * gqa 8 = 8192 rows) is exactly one
// tile, so it runs with no extra iterations. Larger batches tile and stay
// bounded. Override with GGML_SYCL_MKL_FA_Q_TILE.
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ MKL_FA_Q_TILE 8192

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ MKL_FA_WG_ਮਾਪ 256

ਵਰਤੋਂ oneapi::mkl::transpose;
ਵਰਤੋਂ oneapi::mkl::blas::ਕਾਲਮ_major::gemm;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// Pack all GQA Q heads for one KV head into fp16, applying q_scale.
// Launches one kernel per GQA group — each kernel copies exactly
// n_queries * DKQ elements using the per-group dst offset and
// per-head source stride.
ਸਥਿਰ ਖਾਲੀ mkl_fa_pack_q_fp16(
    dpct::ਕਤਾਰ_ptr stream,
    sycl::half * __restrict dst,
    ਅਚਲ ਦਸ਼ਮਲਵ * __restrict q_src,
    ਪੂਰਨ n_queries, ਪੂਰਨ DKQ,
    ਪੂਰਨ gqa_ਅਨੁਪਾਤ, ਪੂਰਨ kvh_ਅਧਾਰ_ਸਿਰ,
    ਦਸ਼ਮਲਵ q_scale, int64_t q_ਪੰਕਤੀ_stride, int64_t q_ਸਿਰ_stride,
    int64_t wg_ਮਾਪ) {

    ਲਈ (ਪੂਰਨ iqg = 0; iqg < gqa_ਅਨੁਪਾਤ; iqg++) {
        ਪੂਰਨ     iqh       = kvh_ਅਧਾਰ_ਸਿਰ + iqg;
        sycl::half * dst_g = dst + (int64_t)iqg * n_queries * DKQ;

        ਅਚਲ int64_t n_elem = (int64_t)n_queries * DKQ;
        ਅਚਲ int64_t wg = ((n_elem + wg_ਮਾਪ - 1) / wg_ਮਾਪ) * wg_ਮਾਪ;

        stream->submit([&](sycl::handler & cgh) {
            cgh.parallel_for(sycl::nd_range<1>(wg, wg_ਮਾਪ),
                [=](sycl::nd_ਵਸਤੂ<1> ਵਸਤੂ) {
                    int64_t e = ਵਸਤੂ.get_global_id(0);
                    ਜੇ (e >= n_elem) ਵਾਪਸ;

                    int64_t q = e / DKQ;
                    int64_t d = e - q * DKQ;

                    // Stride-aware source offset: handles permuted,
                    // sliced, or contiguous Q tensor layouts.
                    int64_t src_off = d
                        + q * q_ਪੰਕਤੀ_stride
                        + (int64_t)iqh * q_ਸਿਰ_stride;

                    dst_g[e] = sycl::half(
                        q_src[src_off] * q_scale);
                });
        });
    }
}

// Zero-initialize the online softmax state arrays.
// KQ_max → -inf, KQ_sum → 0, VKQ_accum → 0.
// Merged into one kernel to avoid per-array launch overhead.
ਸਥਿਰ ਖਾਲੀ mkl_fa_init_softmax_state(
    dpct::ਕਤਾਰ_ptr stream,
    ਦਸ਼ਮਲਵ * kmax, ਦਸ਼ਮਲਵ * ksum, ਦਸ਼ਮਲਵ * vacc,
    ਪੂਰਨ n_query_ਪੰਕਤੀਆਂ, ਪੂਰਨ DV, int64_t wg_ਮਾਪ) {

    ਅਚਲ ਦਸ਼ਮਲਵ    neg_inf   = -1e30f;
    ਅਚਲ int64_t  n_maxsum  = n_query_ਪੰਕਤੀਆਂ;
    ਅਚਲ int64_t  n_vacc    = (int64_t)n_query_ਪੰਕਤੀਆਂ * DV;
    ਅਚਲ int64_t  ਕੁੱਲ     = (n_vacc > n_maxsum) ? n_vacc : n_maxsum;
    ਅਚਲ int64_t  wg = ((ਕੁੱਲ + wg_ਮਾਪ - 1) / wg_ਮਾਪ) * wg_ਮਾਪ;

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<1>(wg, wg_ਮਾਪ),
            [=](sycl::nd_ਵਸਤੂ<1> ਵਸਤੂ) {
                int64_t i = ਵਸਤੂ.get_global_id(0);
                ਜੇ (i < n_maxsum) {
                    kmax[i] = neg_inf;
                    ksum[i] = 0.0f;
                }
                ਜੇ (i < n_vacc) {
                    vacc[i] = 0.0f;
                }
            });
    });
}

// Online softmax over one KV chunk for a tile of GQA query rows.
// The tile spans absolute rows [q0, q0 + q_rows). Score buffers
// (KQ_f32/S_f16) are indexed RELATIVE to the tile; the persistent state
// (VKQ_accum/KQ_max/KQ_sum) and mask are indexed by ABSOLUTE row.
// One WORK-GROUP per query row (local size = wg_size): work-items stride
// over the chunk so adjacent items touch adjacent elements (coalesced),
// the row max/sum come from group reductions, and the DV-long VKQ
// rescale is spread across the items. Item 0 is the sole writer of
// KQ_max/KQ_sum; its writes are ordered after every other item's reads
// by the second group reduction (a collective). Per-element math is
// identical to the original one-item-per-row kernel: softcap before
// mask, native::exp, -1e30 sentinel, half-precision S. Only the float
// summation order differs (tree vs serial), i.e. last-ulp level.
ਸਥਿਰ ਖਾਲੀ mkl_fa_online_softmax_chunk(
    dpct::ਕਤਾਰ_ptr stream,
    ਦਸ਼ਮਲਵ * __restrict KQ_f32,
    sycl::half * __restrict S_f16,
    ਦਸ਼ਮਲਵ * __restrict KQ_max,
    ਦਸ਼ਮਲਵ * __restrict KQ_ਜੋੜ,
    ਦਸ਼ਮਲਵ * __restrict VKQ_accum,
    ਪੂਰਨ q0, ਪੂਰਨ q_ਪੰਕਤੀਆਂ, ਪੂਰਨ n_queries, ਪੂਰਨ DV,
    ਪੂਰਨ chunk_ਮਾਪ, ਪੂਰਨ chunk_ਸ਼ੁਰੂ,
    ਪੂਰਨ kvh_ਸਿਰ,
    ਅਚਲ sycl::half * ਨਕਾਬ_ਅੰਕੜੇ, int64_t ਨਕਾਬ_ਸਿਰ_stride,
    int64_t ਨਕਾਬ_ਪੰਕਤੀ_stride, ਪੂਰਨ ਨਕਾਬ_n_heads,
    ਦਸ਼ਮਲਵ logit_softcap, int64_t wg_ਮਾਪ) {

    // One work-group per query row: exactly q_rows groups of wg_size
    // items. q_rows * wg_size is already a multiple of wg_size, so unlike
    // the one-item-per-row kernels there is no round-up / tail guard.
    ਅਚਲ int64_t wg         = q_ਪੰਕਤੀਆਂ * wg_ਮਾਪ;
    ਅਚਲ ਪੂਰਨ     local_ਮਾਪ = (ਪੂਰਨ) wg_ਮਾਪ;  // stride in the loops below
    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<1>(wg, wg_ਮਾਪ),
            [=](sycl::nd_ਵਸਤੂ<1> ਵਸਤੂ) {
                ਅਚਲ ਪੂਰਨ local_id = (ਪੂਰਨ)ਵਸਤੂ.get_local_id(0);
                ਅਚਲ ਪੂਰਨ ਪੰਕਤੀ = (ਪੂਰਨ)ਵਸਤੂ.get_ਸਮੂਹ(0); // tile-relative
                ਅਚਲ ਪੂਰਨ jc_abs    = q0 + ਪੰਕਤੀ;
                ਅਚਲ ਪੂਰਨ gqa_ਸਮੂਹ = jc_abs / n_queries;
                ਅਚਲ ਪੂਰਨ q_ਪੰਕਤੀ     = jc_abs % n_queries;
                // Score buffers are tile-local (relative index).
                ਅਚਲ ਦਸ਼ਮਲਵ * __restrict KQ_ਪੰਕਤੀ = KQ_f32
                    + ਪੰਕਤੀ * (int64_t)chunk_ਮਾਪ;
                sycl::half * __restrict S_ਪੰਕਤੀ = S_f16
                    + ਪੰਕਤੀ * (int64_t)chunk_ਮਾਪ;
                // Persistent accumulator is full-sized (absolute index).
                ਦਸ਼ਮਲਵ * __restrict vkq = VKQ_accum
                    + jc_abs * (int64_t)DV;
                ਅਚਲ sycl::half * ਨਕਾਬ_h = ਖਾਲੀ_ਸੂਚਕ;
                int64_t m_stride = 0;
                ਜੇ (ਨਕਾਬ_ਅੰਕੜੇ) {
                    ਪੂਰਨ m_ਸਿਰ = (ਨਕਾਬ_n_heads > 1)
                        ? (kvh_ਸਿਰ + gqa_ਸਮੂਹ) : 0;
                    ਨਕਾਬ_h   = ਨਕਾਬ_ਅੰਕੜੇ + (int64_t)m_ਸਿਰ * ਨਕਾਬ_ਸਿਰ_stride;
                    m_stride = ਨਕਾਬ_ਪੰਕਤੀ_stride;
                }
                // Score at chunk offset i — original per-element math.
                ਸਵੈਚਾਲਿਤ score = [&](ਪੂਰਨ i) {
                    ਦਸ਼ਮਲਵ s = KQ_ਪੰਕਤੀ[i];
                    ਜੇ (logit_softcap != 0.0f) {
                        s = logit_softcap * sycl::tanh(s);
                    }
                    ਜੇ (ਨਕਾਬ_h) {
                        s += (ਦਸ਼ਮਲਵ)ਨਕਾਬ_h[q_ਪੰਕਤੀ * m_stride
                            + (chunk_ਸ਼ੁਰੂ + i)];
                    }
                    ਵਾਪਸ s;
                };
                // Pass 1: strided (coalesced) row-wise local maximum.
                ਦਸ਼ਮਲਵ local_max = -1e30f;
                ਲਈ (ਪੂਰਨ i = local_id; i < chunk_ਮਾਪ; i += local_ਮਾਪ) {
                    ਦਸ਼ਮਲਵ s = score(i);
                    ਜੇ (s > local_max) local_max = s;
                }
                ਅਚਲ ਦਸ਼ਮਲਵ final_local_max = sycl::reduce_over_ਸਮੂਹ(
                    ਵਸਤੂ.get_ਸਮੂਹ(), local_max, sycl::ਵੱਧ_ਤੋਂ_ਵੱਧ<ਦਸ਼ਮਲਵ>());
                // Rescale previous accumulator by exp(old_max - new_max)
                ਦਸ਼ਮਲਵ old_max = KQ_max[jc_abs];
                ਦਸ਼ਮਲਵ new_max = (old_max > final_local_max) ? old_max : final_local_max;
                ਦਸ਼ਮਲਵ rescale = (old_max < -1e29f) ? 1.0f
                    : sycl::native::ਘਾਤਾਂਕ(old_max - new_max);
                ਲਈ (ਪੂਰਨ v = local_id; v < DV; v += local_ਮਾਪ) {
                    vkq[v] *= rescale;
                }
                // Pass 2: softmax numerators, strided; S row written once.
                ਦਸ਼ਮਲਵ local_ਜੋੜ = 0.0f;
                ਲਈ (ਪੂਰਨ i = local_id; i < chunk_ਮਾਪ; i += local_ਮਾਪ) {
                    ਦਸ਼ਮਲਵ s = score(i);
                    ਦਸ਼ਮਲਵ val = sycl::native::ਘਾਤਾਂਕ(s - new_max);
                    S_ਪੰਕਤੀ[i] = sycl::half(val);
                    local_ਜੋੜ += val;
                }
                ਅਚਲ ਦਸ਼ਮਲਵ ਕੁੱਲ_ਜੋੜ = sycl::reduce_over_ਸਮੂਹ(
                    ਵਸਤੂ.get_ਸਮੂਹ(), local_ਜੋੜ, sycl::plus<ਦਸ਼ਮਲਵ>());
                ਜੇ (local_id == 0) {
                    KQ_ਜੋੜ[jc_abs] = KQ_ਜੋੜ[jc_abs] * rescale + ਕੁੱਲ_ਜੋੜ;
                    KQ_max[jc_abs] = new_max;
                }
            });
    });
}

// Write one GQA group's normalized output to its destination head.
ਸਥਿਰ ਖਾਲੀ mkl_fa_normalize_ਸਿਰ(
    dpct::ਕਤਾਰ_ptr stream,
    ਦਸ਼ਮਲਵ * __restrict dst_batch,
    ਅਚਲ ਦਸ਼ਮਲਵ * __restrict VKQ_accum,
    ਅਚਲ ਦਸ਼ਮਲਵ * __restrict KQ_ਜੋੜ,
    ਪੂਰਨ iqh, ਪੂਰਨ n_queries, ਪੂਰਨ DV, ਪੂਰਨ n_q_heads,
    int64_t src_ਵਿੱਥ, int64_t wg_ਮਾਪ) {

    ਅਚਲ int64_t wg = ((n_queries + wg_ਮਾਪ - 1) / wg_ਮਾਪ) * wg_ਮਾਪ;

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<1>(wg, wg_ਮਾਪ),
            [=](sycl::nd_ਵਸਤੂ<1> ਵਸਤੂ) {
                ਪੂਰਨ jc = ਵਸਤੂ.get_global_id(0);
                ਜੇ (jc >= n_queries) ਵਾਪਸ;

                ਪੂਰਨ     ksum_idx = (ਪੂਰਨ)(src_ਵਿੱਥ / DV) + jc;
                ਦਸ਼ਮਲਵ   inv_ਜੋੜ  = 1.0f / KQ_ਜੋੜ[ksum_idx];
                ਅਚਲ ਦਸ਼ਮਲਵ * __restrict src = VKQ_accum
                    + src_ਵਿੱਥ + jc * (int64_t)DV;
                // Interleaved dst layout (matching TILE):
                // rows alternate between heads, then increment query.
                // offset = (query * n_q_heads + head) * DV
                ਦਸ਼ਮਲਵ * __restrict dst_ਪੰਕਤੀ = dst_batch
                    + ((int64_t)jc * n_q_heads + iqh) * (int64_t)DV;

                ਲਈ (ਪੂਰਨ v = 0; v < DV; v++) {
                    dst_ਪੰਕਤੀ[v] = src[v] * inv_ਜੋੜ;
                }
            });
    });
}

// ---------------------------------------------------------------------------
// Per-chunk dequant
//
// Rather than dequantizing all of K/V up front (footprint scales with
// context), we dequant one KV-head chunk at a time into a dense
// [this_chunk x D] fp16 buffer (row-major, lda = D). The source address of
// element (head=ikvh, row=chunk_start+r, col=c) decomposes into independent
// linear terms head_off(ikvh) + row_off(chunk_start) + (r,c), so slicing a
// chunk is a clean pointer offset in every layout case. The true-Gemma-
// interleave vs padded-seq-view distinction is resolved once when the
// descriptor is built; slicing does not reintroduce it.
// ---------------------------------------------------------------------------
ਗਿਣਤੀ mkl_fa_kv_desc_ਬਹੁਲਕ {
    MKL_FA_KV_ਬਹੁਲਕ_F16_DENSE       = 0,
    MKL_FA_KV_ਬਹੁਲਕ_F16_INTERLEAVED = 1,
    MKL_FA_KV_ਬਹੁਲਕ_QUANT_CONTIG    = 2,
    MKL_FA_KV_ਬਹੁਲਕ_QUANT_NC        = 3,
};

ਬਣਤਰ mkl_fa_kv_desc {
    ਅਚਲ ਅੱਖਰ *         ਅੰਕੜੇ = ਖਾਲੀ_ਸੂਚਕ;
    ggml_type            type = GGML_TYPE_F16;
    int64_t              D    = 0;      // ne[0]
    int64_t              nb1  = 0;      // byte stride, seq dim
    int64_t              nb2  = 0;      // byte stride, head dim
    mkl_fa_kv_desc_ਬਹੁਲਕ  mode = MKL_FA_KV_ਬਹੁਲਕ_F16_DENSE;
    int64_t              ts   = 0;      // type size (mode 3 base offset)
    int64_t              s01  = 0;      // nc row stride in blocks (mode 3)
    int64_t              s02  = 0;      // nc head stride in blocks (mode 3)
};

ਸਥਿਰ mkl_fa_kv_desc mkl_fa_make_desc(ਅਚਲ ggml_tensor * T, ਤਾਰਕਿਕ interleaved, ਪੂਰਨ n_kv_heads) {
    mkl_fa_kv_desc d;
    d.ਅੰਕੜੇ = (ਅਚਲ ਅੱਖਰ *)T->ਅੰਕੜੇ;
    d.type = T->type;
    d.D    = T->ne[0];
    d.nb1  = (int64_t)T->nb[1];
    d.nb2  = (int64_t)T->nb[2];
    d.ts   = (int64_t)ggml_type_ਮਾਪ(T->type);

    ਜੇ (T->type == GGML_TYPE_F16) {
        // MLA's V cache is a 512-wide view of 576-wide K rows. Treat any
        // padded row stride as strided even when there is only one KV head.
        d.mode = interleaved || d.nb1 != d.D * (int64_t)ਆਕਾਰ(sycl::half)
            ? MKL_FA_KV_ਬਹੁਲਕ_F16_INTERLEAVED : MKL_FA_KV_ਬਹੁਲਕ_F16_DENSE;
    } ਨਹੀਂਤਾਂ ਜੇ (ggml_is_contiguously_allocated(T) && !interleaved) {
        d.mode = MKL_FA_KV_ਬਹੁਲਕ_QUANT_CONTIG;
    } ਨਹੀਂਤਾਂ {
        d.mode = MKL_FA_KV_ਬਹੁਲਕ_QUANT_NC;
        ਅਚਲ int64_t bs          = (int64_t)ggml_blck_ਮਾਪ(T->type);
        ਅਚਲ int64_t blk_ਪ੍ਰਤੀ_ਪੰਕਤੀ = T->ne[0] / bs;
        // True Gemma interleave packs heads within a row (nb[2] < ne[1]*nb[1])
        // → reconstruct physical strides. Padded seq-views (nb[2] > ne[1]*nb[1])
        // already have correct physical strides.
        ਅਚਲ ਤਾਰਕਿਕ gemma = interleaved &&
            ((int64_t)T->nb[2] < (int64_t)T->ne[1] * (int64_t)T->nb[1]);
        ਜੇ (gemma) {
            d.s01 = (int64_t)n_kv_heads * blk_ਪ੍ਰਤੀ_ਪੰਕਤੀ;
            d.s02 = blk_ਪ੍ਰਤੀ_ਪੰਕਤੀ;
        } ਨਹੀਂਤਾਂ {
            d.s01 = d.nb1 / d.ts;
            d.s02 = d.nb2 / d.ts;
        }
    }
    ਵਾਪਸ d;
}

// Dequant one KV-head chunk into a dense [this_chunk x D] fp16 buffer.
ਸਥਿਰ ਖਾਲੀ mkl_fa_dequant_chunk(
    dpct::ਕਤਾਰ_ptr stream, ਅਚਲ mkl_fa_kv_desc & d, ggml_tensor * dst_ctx,
    sycl::half * ਬਾਹਰਲਾ, ਪੂਰਨ ikvh, ਪੂਰਨ chunk_ਸ਼ੁਰੂ, ਪੂਰਨ this_chunk) {

    ਅਚਲ int64_t D = d.D;
    ਚੋਣ (d.mode) {
        ਹਾਲਤ MKL_FA_KV_ਬਹੁਲਕ_F16_DENSE: {
            ਅਚਲ ਅੱਖਰ * ਅਧਾਰ = d.ਅੰਕੜੇ + (int64_t)ikvh * d.nb2
                + (int64_t)chunk_ਸ਼ੁਰੂ * d.nb1;
            stream->ਸਮ੍ਰਿਤੀ_ਨਕਲ(ਬਾਹਰਲਾ, ਅਧਾਰ, (size_t)this_chunk * D * ਆਕਾਰ(sycl::half));
            ਰੋਕੋ;
        }
        ਹਾਲਤ MKL_FA_KV_ਬਹੁਲਕ_F16_INTERLEAVED: {
            ਅਚਲ ਅੱਖਰ * ਅਧਾਰ = d.ਅੰਕੜੇ + (int64_t)ikvh * d.nb2
                + (int64_t)chunk_ਸ਼ੁਰੂ * d.nb1;
            ਅਚਲ int64_t ਪੰਕਤੀ_halfs = d.nb1 / (int64_t)ਆਕਾਰ(sycl::half);
            ਅਚਲ sycl::half * src = (ਅਚਲ sycl::half *)ਅਧਾਰ;
            stream->parallel_for(
                sycl::range<2>((size_t)this_chunk, (size_t)D),
                [=](sycl::ਵਸਤੂ<2> it) {
                    int64_t r = it.get_id(0);
                    int64_t c = it.get_id(1);
                    ਬਾਹਰਲਾ[r * D + c] = src[r * ਪੰਕਤੀ_halfs + c];
                });
            ਰੋਕੋ;
        }
        ਹਾਲਤ MKL_FA_KV_ਬਹੁਲਕ_QUANT_CONTIG: {
            ਅਚਲ ਅੱਖਰ * ਅਧਾਰ = d.ਅੰਕੜੇ + (int64_t)ikvh * d.nb2
                + (int64_t)chunk_ਸ਼ੁਰੂ * d.nb1;
            ਮੰਜ਼ਿਲ_fp16_sycl_t ਮੰਜ਼ਿਲ_fp16 = ggml_get_ਮੰਜ਼ਿਲ_fp16_sycl(d.type, dst_ctx);
            ਮੰਜ਼ਿਲ_fp16(ਅਧਾਰ, ਬਾਹਰਲਾ, (int64_t)this_chunk * D, stream);
            ਰੋਕੋ;
        }
        ਮੂਲ: {  // MKL_FA_KV_MODE_QUANT_NC
            ਮੰਜ਼ਿਲ_fp16_nc_sycl_t ਮੰਜ਼ਿਲ_fp16 = ggml_get_ਮੰਜ਼ਿਲ_fp16_nc_sycl(d.type);
            ਅਚਲ int64_t ਅਧਾਰ_ਖੰਡਾਂ = (int64_t)ikvh * d.s02
                + (int64_t)chunk_ਸ਼ੁਰੂ * d.s01;
            ਅਚਲ ਅੱਖਰ * ਅਧਾਰ = d.ਅੰਕੜੇ + ਅਧਾਰ_ਖੰਡਾਂ * d.ts;
            // ne02 = ne03 = 1 → s02/s03 inert; head+chunk offset carried by base.
            ਮੰਜ਼ਿਲ_fp16(ਅਧਾਰ, ਬਾਹਰਲਾ, D, this_chunk, 1, 1, d.s01, d.s02, d.s02, stream);
            ਰੋਕੋ;
        }
    }
}

// ---------------------------------------------------------------------------
// MKL Flash Attention orchestrator
//
// Pipeline: dequantize K/V → for each KV head:
//   pack GQA Q heads → MKL GEMM KQ → online softmax →
//   MKL GEMM VKQ → accumulate → normalize → scatter to dst
// ---------------------------------------------------------------------------
ਖਾਲੀ ggml_sycl_flash_attn_ext_mkl(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {

    ਅਚਲ ggml_tensor * Q    = dst->src[0];
    ਅਚਲ ggml_tensor * K    = dst->src[1];
    ਅਚਲ ggml_tensor * V    = dst->src[2];
    ਅਚਲ ggml_tensor * ਨਕਾਬ = dst->src[3];
    ggml_tensor * KQV = dst;

    GGML_ASSERT(Q->type == GGML_TYPE_F32);
    GGML_ASSERT(KQV->type == GGML_TYPE_F32);

    // --- Op params ---
    ਦਸ਼ਮਲਵ scale = 1.0f, max_bias = 0.0f, logit_softcap = 0.0f;
    ਸਮ੍ਰਿਤੀ_ਨਕਲ(&scale,         (ਅਚਲ ਦਸ਼ਮਲਵ *)KQV->op_params + 0, ਆਕਾਰ(ਦਸ਼ਮਲਵ));
    ਸਮ੍ਰਿਤੀ_ਨਕਲ(&max_bias,      (ਅਚਲ ਦਸ਼ਮਲਵ *)KQV->op_params + 1, ਆਕਾਰ(ਦਸ਼ਮਲਵ));
    ਸਮ੍ਰਿਤੀ_ਨਕਲ(&logit_softcap, (ਅਚਲ ਦਸ਼ਮਲਵ *)KQV->op_params + 2, ਆਕਾਰ(ਦਸ਼ਮਲਵ));

    ਅਚਲ ਦਸ਼ਮਲਵ q_scale = scale;

    // --- Dimensions ---
    ਅਚਲ ਪੂਰਨ DKQ        = (ਪੂਰਨ)K->ne[0];
    ਅਚਲ ਪੂਰਨ DV         = (ਪੂਰਨ)V->ne[0];
    ਅਚਲ ਪੂਰਨ n_queries  = (ਪੂਰਨ)Q->ne[1];
    ਅਚਲ ਪੂਰਨ n_q_heads  = (ਪੂਰਨ)Q->ne[2];
    ਅਚਲ ਪੂਰਨ n_kv_heads = (ਪੂਰਨ)K->ne[2];
    ਅਚਲ ਪੂਰਨ n_batch    = (ਪੂਰਨ)Q->ne[3];
    ਅਚਲ ਪੂਰਨ n_kv       = (ਪੂਰਨ)K->ne[1];
    ਅਚਲ ਪੂਰਨ gqa_ਅਨੁਪਾਤ  = n_q_heads / n_kv_heads;
    ਅਚਲ ਪੂਰਨ n_query_ਪੰਕਤੀਆਂ = n_queries * gqa_ਅਨੁਪਾਤ;

    GGML_ASSERT(n_q_heads % n_kv_heads == 0);
    GGML_ASSERT(max_bias == 0.0f);  // ALiBi not supported
    GGML_ASSERT(Q->ne[3] == K->ne[3] || K->ne[3] == 1);

    ਅਚਲ ਪੂਰਨ chunk_ਮਾਪ = std::min(MKL_FA_CHUNK_ਮਾਪ_KV, n_kv);

    // Query rows are processed in tiles of q_tile_rows so the score buffers
    // (KQ_f32/S_f16 = q_tile_rows * chunk_size) stay bounded regardless of
    // batch size. n_query_rows <= Q_TILE is a single tile (no extra work).
    ਸਥਿਰ ਪੂਰਨ q_tile_env = ggml_sycl_get_env("GGML_SYCL_MKL_FA_Q_TILE", MKL_FA_Q_TILE);
    ਅਚਲ ਪੂਰਨ q_tile_ਪੰਕਤੀਆਂ = std::max(1, std::min(q_tile_env, n_query_ਪੰਕਤੀਆਂ));

    ਅਚਲ int64_t wg_ਮਾਪ = MKL_FA_WG_ਮਾਪ;

    // --- Debug output (gated by GGML_SYCL_MKL_FA_DEBUG=1) ---
    ਸਥਿਰ ਪੂਰਨ mkl_call_ਗਿਣਤੀ = 0;
    mkl_call_ਗਿਣਤੀ++;
    ਸਥਿਰ ਪੂਰਨ mkl_debug = ggml_sycl_get_env("GGML_SYCL_MKL_FA_DEBUG", 0);
    ਅਚਲ ਤਾਰਕਿਕ do_print = (mkl_debug == 1);

    ਅਚਲ int64_t q_ਪੰਕਤੀ_stride  = Q->nb[1] / ਆਕਾਰ(ਦਸ਼ਮਲਵ);
    ਅਚਲ int64_t q_ਸਿਰ_stride = Q->nb[2] / ਆਕਾਰ(ਦਸ਼ਮਲਵ);

    // Alias the dequantized buffers only when K and V expose the same values.
    // MLA V is a narrower view of K and needs its own strided dequantization.
    ਅਚਲ ਤਾਰਕਿਕ V_is_K_view = V->ne[0] == K->ne[0] && V->view_src
        && (V->view_src == K || (V->view_src == K->view_src
            && V->view_offs == K->view_offs));

    // Early interleaved detection for debug output.
    // True interleaved detection happens after dequant (nb12_fp16 == nb11_fp16),
    // but we can pre-detect on the original tensor strides.
    ਅਚਲ ਤਾਰਕਿਕ k_early_interleaved =
        ((int64_t)K->ne[1] * K->nb[1] != K->nb[2]);
    ਅਚਲ ਤਾਰਕਿਕ v_early_interleaved =
        !V_is_K_view && ((int64_t)V->ne[1] * V->nb[1] != V->nb[2]);

    ਜੇ (do_print) {
        GGML_LOG_INFO("[MKL-FA] #%d D=%d DV=%d n_q=%d n_kv=%d "
                "n_qh=%d n_kvh=%d gqa=%d batch=%d K=%s V=%s "
                "chunk=%d buf=%.1fMB%s%s\n",
                mkl_call_ਗਿਣਤੀ, DKQ, DV, n_queries, n_kv,
                n_q_heads, n_kv_heads, gqa_ਅਨੁਪਾਤ, n_batch,
                ggml_type_name(K->type), ggml_type_name(V->type),
                chunk_ਮਾਪ,
                (ਦੁੱਗਣਾ)((int64_t)n_query_ਪੰਕਤੀਆਂ * chunk_ਮਾਪ * ਆਕਾਰ(ਦਸ਼ਮਲਵ))
                    / (1024.0 * 1024.0),
                k_early_interleaved ? " K_ILV" : "",
                v_early_interleaved ? " V_ILV" : "");
        GGML_LOG_INFO("[MKL-FA] #%d Q-nb1=%lld Q-nb2=%lld "
                "q_rs=%lld q_hs=%lld dst_rs=%lld dst_hs=%lld\n",
                mkl_call_ਗਿਣਤੀ,
                (ਲੰਮਾ ਲੰਮਾ)Q->nb[1], (ਲੰਮਾ ਲੰਮਾ)Q->nb[2],
                (ਲੰਮਾ ਲੰਮਾ)q_ਪੰਕਤੀ_stride, (ਲੰਮਾ ਲੰਮਾ)q_ਸਿਰ_stride,
                (ਲੰਮਾ ਲੰਮਾ)(KQV->nb[1] / ਆਕਾਰ(ਦਸ਼ਮਲਵ)),
                (ਲੰਮਾ ਲੰਮਾ)(KQV->nb[2] / ਆਕਾਰ(ਦਸ਼ਮਲਵ)));
    }

    // --- Stream and allocators ---
    dpct::ਕਤਾਰ_ptr stream = ctx.stream();

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ MKL_TAKE_ਸਮਾਂ(t0)  ਸਵੈਚਾਲਿਤ t0 = std::chrono::steady_clock::now()
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ MKL_ACCUM(acc, t0) ਕਰੋ { ਜੇ (do_print) { \
    acc += (int64_t)std::chrono::duration_cast \
    <std::chrono::microseconds>(std::chrono::steady_clock::now() - (t0)).count(); \
} } ਜਦੋਂ(0)

    int64_t gemm_kq_ਸਮਾਂ_us  = 0;
    int64_t gemm_vkq_ਸਮਾਂ_us = 0;
    int64_t softmax_ਸਮਾਂ_us  = 0;
    int64_t dequant_ਸਮਾਂ_us  = 0;

    MKL_TAKE_ਸਮਾਂ(t_deq);

    // --- K/V dequant descriptors ---
    // Dequant is done per-chunk inside the KV loop (footprint independent of
    // context). Output is always dense row-major fp16 [this_chunk x D], lda=D.
    // Interleaved detection: ne[1]*nb[1] != nb[2] means heads are interleaved.
    ਅਚਲ ਤਾਰਕਿਕ k_interleaved =
        ((int64_t)K->ne[1] * K->nb[1] != K->nb[2]) && K->ne[2] > 1;
    ਅਚਲ ਤਾਰਕਿਕ v_interleaved =
        ((int64_t)V->ne[1] * V->nb[1] != V->nb[2]) && V->ne[2] > 1;

    ਅਚਲ mkl_fa_kv_desc K_desc = mkl_fa_make_desc(K, k_interleaved, n_kv_heads);
    ਅਚਲ mkl_fa_kv_desc V_desc = V_is_K_view
        ? K_desc : mkl_fa_make_desc(V, v_interleaved, n_kv_heads);

    MKL_ACCUM(dequant_ਸਮਾਂ_us, t_deq);

    // --- Resolve mask pointers ---
    int64_t ਨਕਾਬ_ਸਿਰ_stride = 0;
    int64_t ਨਕਾਬ_ਪੰਕਤੀ_stride  = 0;
    ਪੂਰਨ     ਨਕਾਬ_n_heads     = 0;

    ਜੇ (ਨਕਾਬ) {
        // Use actual fp16 device size (2 bytes), NOT sizeof(sycl::half)
        // which may be 4 on the host in oneAPI.
        ਨਕਾਬ_ਸਿਰ_stride = ਨਕਾਬ->nb[2] / 2;
        ਨਕਾਬ_ਪੰਕਤੀ_stride  = ਨਕਾਬ->nb[1] / 2;
        ਨਕਾਬ_n_heads     = (ਪੂਰਨ)ਨਕਾਬ->ne[2];
    }

    // --- Allocate intermediates from pool ---
    ggml_sycl_pool & pool = ctx.pool();

    ggml_sycl_pool_alloc<ਦਸ਼ਮਲਵ>      KQ_f32(pool);      // [q_tile_rows x chunk]
    ggml_sycl_pool_alloc<sycl::half> S_f16(pool);       // [q_tile_rows x chunk]
    ggml_sycl_pool_alloc<ਦਸ਼ਮਲਵ>      VKQ_chunk(pool);   // [q_tile_rows x DV]
    ggml_sycl_pool_alloc<ਦਸ਼ਮਲਵ>      VKQ_accum(pool);   // [n_query_rows x DV] (full)
    ggml_sycl_pool_alloc<ਦਸ਼ਮਲਵ>      KQ_max(pool);      // [n_query_rows] (full)
    ggml_sycl_pool_alloc<ਦਸ਼ਮਲਵ>      KQ_ਜੋੜ(pool);      // [n_query_rows] (full)
    ggml_sycl_pool_alloc<sycl::half> Q_ਸਿਰ_f16(pool);  // [n_query_rows x DKQ] (full)
    ggml_sycl_pool_alloc<sycl::half> K_chunk_f16(pool); // [chunk x DKQ] (per-chunk dequant)
    ggml_sycl_pool_alloc<sycl::half> V_chunk_f16(pool); // [chunk x DV] (per-chunk dequant)

    KQ_f32.alloc((size_t)q_tile_ਪੰਕਤੀਆਂ * chunk_ਮਾਪ);
    S_f16.alloc((size_t)q_tile_ਪੰਕਤੀਆਂ * chunk_ਮਾਪ);
    VKQ_chunk.alloc((size_t)q_tile_ਪੰਕਤੀਆਂ * DV);
    VKQ_accum.alloc((size_t)n_query_ਪੰਕਤੀਆਂ * DV);
    KQ_max.alloc(n_query_ਪੰਕਤੀਆਂ);
    KQ_ਜੋੜ.alloc(n_query_ਪੰਕਤੀਆਂ);
    Q_ਸਿਰ_f16.alloc((size_t)n_query_ਪੰਕਤੀਆਂ * DKQ);
    K_chunk_f16.alloc((size_t)chunk_ਮਾਪ * DKQ);

    sycl::half * V_chunk_f16_ptr;
    ਜੇ (V_is_K_view) {
        V_chunk_f16_ptr = K_chunk_f16.ptr;   // V aliases K (DV == DKQ)
    } ਨਹੀਂਤਾਂ {
        V_chunk_f16.alloc((size_t)chunk_ਮਾਪ * DV);
        V_chunk_f16_ptr = V_chunk_f16.ptr;
    }

    sycl::half * Q_ਸਿਰ_f16_ptr  = Q_ਸਿਰ_f16.ptr;
    ਦਸ਼ਮਲਵ      * KQ_f32_ptr      = KQ_f32.ptr;
    sycl::half * S_f16_ptr       = S_f16.ptr;
    ਦਸ਼ਮਲਵ      * VKQ_chunk_ptr   = VKQ_chunk.ptr;
    ਦਸ਼ਮਲਵ      * VKQ_accum_ptr   = VKQ_accum.ptr;
    ਦਸ਼ਮਲਵ      * KQ_max_ptr      = KQ_max.ptr;
    ਦਸ਼ਮਲਵ      * KQ_ਜੋੜ_ptr      = KQ_ਜੋੜ.ptr;
    sycl::half * K_chunk_f16_ptr = K_chunk_f16.ptr;

    ਅਚਲ ਦਸ਼ਮਲਵ alpha = 1.0f;
    ਅਚਲ ਦਸ਼ਮਲਵ beta  = 0.0f;

    ਲਈ (ਪੂਰਨ ib = 0; ib < n_batch; ib++) {
        ਅਚਲ ਦਸ਼ਮਲਵ * Q_batch = (ਅਚਲ ਦਸ਼ਮਲਵ *)Q->ਅੰਕੜੇ
            + ib * (Q->nb[3] / ਆਕਾਰ(ਦਸ਼ਮਲਵ));
        ਦਸ਼ਮਲਵ * dst_batch = (ਦਸ਼ਮਲਵ *)KQV->ਅੰਕੜੇ
            + ib * (KQV->nb[3] / ਆਕਾਰ(ਦਸ਼ਮਲਵ));

        ਅਚਲ sycl::half * ਨਕਾਬ_batch = ਖਾਲੀ_ਸੂਚਕ;
        ਜੇ (ਨਕਾਬ) {
            ਪੂਰਨ m_batch = (ਨਕਾਬ->ne[3] > 1) ? ib : 0;
            ਨਕਾਬ_batch = (ਅਚਲ sycl::half *)ਨਕਾਬ->ਅੰਕੜੇ
                + m_batch * (ਨਕਾਬ->nb[3] / 2);  // 2 = actual fp16 device size
        }

        ਲਈ (ਪੂਰਨ ikvh = 0; ikvh < n_kv_heads; ikvh++) {
            ਪੂਰਨ kvh_ਅਧਾਰ_ਸਿਰ = ikvh * gqa_ਅਨੁਪਾਤ;

            // 1. Pack all GQA Q heads into fp16 (full n_query_rows)
            mkl_fa_pack_q_fp16(stream,
                Q_ਸਿਰ_f16_ptr, Q_batch,
                n_queries, DKQ,
                gqa_ਅਨੁਪਾਤ, kvh_ਅਧਾਰ_ਸਿਰ,
                q_scale, q_ਪੰਕਤੀ_stride, q_ਸਿਰ_stride, wg_ਮਾਪ);

            // 2. Initialize softmax state (full n_query_rows)
            mkl_fa_init_softmax_state(stream,
                KQ_max_ptr, KQ_ਜੋੜ_ptr, VKQ_accum_ptr,
                n_query_ਪੰਕਤੀਆਂ, DV, wg_ਮਾਪ);

            // Sync before MKL GEMM (MKL may use an internal queue)
            stream->wait();

            // 3. KV chunk loop (OUTER): dequant each chunk once, then tile queries.
            ਲਈ (ਪੂਰਨ chunk_ਸ਼ੁਰੂ = 0; chunk_ਸ਼ੁਰੂ < n_kv; chunk_ਸ਼ੁਰੂ += chunk_ਮਾਪ) {
                ਪੂਰਨ this_chunk = std::min(chunk_ਮਾਪ, n_kv - chunk_ਸ਼ੁਰੂ);

                // 3a. Dequant this KV chunk to dense fp16 (once per chunk)
                {
                    MKL_TAKE_ਸਮਾਂ(t0);
                    mkl_fa_dequant_chunk(stream, K_desc, KQV,
                        K_chunk_f16_ptr, ikvh, chunk_ਸ਼ੁਰੂ, this_chunk);
                    ਜੇ (!V_is_K_view) {
                        mkl_fa_dequant_chunk(stream, V_desc, KQV,
                            V_chunk_f16_ptr, ikvh, chunk_ਸ਼ੁਰੂ, this_chunk);
                    }
                    stream->wait();  // dequant must be ready before MKL GEMM
                    MKL_ACCUM(dequant_ਸਮਾਂ_us, t0);
                }

                // 3b. Query tile loop (INNER) — bounds KQ_f32/S_f16 footprint.
                ਲਈ (ਪੂਰਨ q0 = 0; q0 < n_query_ਪੰਕਤੀਆਂ; q0 += q_tile_ਪੰਕਤੀਆਂ) {
                    ਪੂਰਨ q_ਪੰਕਤੀਆਂ = std::min(q_tile_ਪੰਕਤੀਆਂ, n_query_ਪੰਕਤੀਆਂ - q0);

                    // GEMM: KQ = Q_tile × K_chunk^T
                    {
                        MKL_TAKE_ਸਮਾਂ(t0);
                        sycl::event ev = gemm(*stream,
                            transpose::trans, transpose::nontrans,
                            this_chunk, q_ਪੰਕਤੀਆਂ, DKQ,
                            alpha,
                            K_chunk_f16_ptr, DKQ,
                            Q_ਸਿਰ_f16_ptr + (int64_t)q0 * DKQ, DKQ,
                            beta,
                            KQ_f32_ptr, this_chunk);
                        ਕੋਸ਼ਿਸ਼ { ev.wait_and_throw(); } ਫੜੋ (sycl::exception & e) {
                            GGML_LOG_INFO("[MKL-FA] GEMM KQ: %s\n", e.what());
                            GGML_ABORT("MKL GEMM KQ failed");
                        }
                        MKL_ACCUM(gemm_kq_ਸਮਾਂ_us, t0);
                    }
                    // Online softmax over this chunk for this query tile
                    {
                        MKL_TAKE_ਸਮਾਂ(t0);
                        mkl_fa_online_softmax_chunk(stream,
                            KQ_f32_ptr, S_f16_ptr,
                            KQ_max_ptr, KQ_ਜੋੜ_ptr, VKQ_accum_ptr,
                            q0, q_ਪੰਕਤੀਆਂ, n_queries, DV,
                            this_chunk, chunk_ਸ਼ੁਰੂ,
                            kvh_ਅਧਾਰ_ਸਿਰ,
                            ਨਕਾਬ_batch, ਨਕਾਬ_ਸਿਰ_stride,
                            ਨਕਾਬ_ਪੰਕਤੀ_stride, ਨਕਾਬ_n_heads,
                            logit_softcap, wg_ਮਾਪ);
                        stream->wait();  // S_f16 must be ready for GEMM
                        MKL_ACCUM(softmax_ਸਮਾਂ_us, t0);
                    }

                    // GEMM: VKQ_chunk = S × V_chunk
                    {
                        MKL_TAKE_ਸਮਾਂ(t0);
                        sycl::event ev = gemm(*stream,
                            transpose::nontrans, transpose::nontrans,
                            DV, q_ਪੰਕਤੀਆਂ, this_chunk,
                            alpha,
                            V_chunk_f16_ptr, DV,
                            S_f16_ptr, this_chunk,
                            beta,
                            VKQ_chunk_ptr, DV);
                        ਕੋਸ਼ਿਸ਼ { ev.wait_and_throw(); } ਫੜੋ (sycl::exception & e) {
                            GGML_LOG_INFO("[MKL-FA] GEMM VKQ: %s\n", e.what());
                            GGML_ABORT("MKL GEMM VKQ failed");
                        }
                        MKL_ACCUM(gemm_vkq_ਸਮਾਂ_us, t0);
                    }
                    // VKQ_accum[q0..] += VKQ_chunk
                    {
                        ਅਚਲ int64_t n_ਕੁੱਲ = (int64_t)q_ਪੰਕਤੀਆਂ * DV;
                        ਅਚਲ int64_t wg = ((n_ਕੁੱਲ + wg_ਮਾਪ - 1) / wg_ਮਾਪ)
                            * wg_ਮਾਪ;
                        ਦਸ਼ਮਲਵ * accum = VKQ_accum_ptr + (int64_t)q0 * DV;
                        stream->submit([&](sycl::handler & cgh) {
                            cgh.parallel_for(sycl::nd_range<1>(wg, wg_ਮਾਪ),
                                [=](sycl::nd_ਵਸਤੂ<1> ਵਸਤੂ) {
                                    int64_t i = ਵਸਤੂ.get_global_id(0);
                                    ਜੇ (i < n_ਕੁੱਲ) {
                                        accum[i] += VKQ_chunk_ptr[i];
                                    }
                                });
                        });
                    }
                }
            }

            // 4. Normalize and scatter each GQA head to dst
            ਲਈ (ਪੂਰਨ iqg = 0; iqg < gqa_ਅਨੁਪਾਤ; iqg++) {
                ਪੂਰਨ     iqh        = kvh_ਅਧਾਰ_ਸਿਰ + iqg;
                int64_t src_ਵਿੱਥ = (int64_t)iqg * n_queries * DV;
                mkl_fa_normalize_ਸਿਰ(stream,
                    dst_batch, VKQ_accum_ptr, KQ_ਜੋੜ_ptr,
                    iqh, n_queries, DV, n_q_heads,
                    src_ਵਿੱਥ, wg_ਮਾਪ);
            }
        }
    }

#ਪਰਿਭਾਸ਼ਾ_ਰੱਦ MKL_TAKE_ਸਮਾਂ
#ਪਰਿਭਾਸ਼ਾ_ਰੱਦ MKL_ACCUM

    ਜੇ (do_print) {
        ਅਚਲ int64_t v_chunk_elems = V_is_K_view ? 0 : (int64_t)chunk_ਮਾਪ * DV;
        ਦੁੱਗਣਾ ਕੁੱਲ_mb = (ਦੁੱਗਣਾ)(
            (int64_t)q_tile_ਪੰਕਤੀਆਂ * chunk_ਮਾਪ * ਆਕਾਰ(ਦਸ਼ਮਲਵ)      // KQ_f32
          + (int64_t)q_tile_ਪੰਕਤੀਆਂ * chunk_ਮਾਪ * ਆਕਾਰ(sycl::half) // S_f16
          + (int64_t)q_tile_ਪੰਕਤੀਆਂ * DV * ਆਕਾਰ(ਦਸ਼ਮਲਵ)              // VKQ_chunk
          + (int64_t)n_query_ਪੰਕਤੀਆਂ * DV * ਆਕਾਰ(ਦਸ਼ਮਲਵ)             // VKQ_accum
          + (int64_t)n_query_ਪੰਕਤੀਆਂ * ਆਕਾਰ(ਦਸ਼ਮਲਵ)                  // KQ_max
          + (int64_t)n_query_ਪੰਕਤੀਆਂ * ਆਕਾਰ(ਦਸ਼ਮਲਵ)                  // KQ_sum
          + (int64_t)n_query_ਪੰਕਤੀਆਂ * DKQ * ਆਕਾਰ(sycl::half)       // Q_head_f16
          + (int64_t)chunk_ਮਾਪ * DKQ * ਆਕਾਰ(sycl::half)         // K_chunk_f16
          + v_chunk_elems * (int64_t)ਆਕਾਰ(sycl::half)            // V_chunk_f16
        ) / (1024.0 * 1024.0);
        GGML_LOG_INFO("[MKL-FA] #%d n_kv=%d n_q=%d q_tile=%d time_us: "
                "dequant=%lld GEMM_KQ=%lld softmax=%lld GEMM_VKQ=%lld "
                "buf_mb=%.1f\n",
                mkl_call_ਗਿਣਤੀ, n_kv, n_queries, q_tile_ਪੰਕਤੀਆਂ,
                (ਲੰਮਾ ਲੰਮਾ)dequant_ਸਮਾਂ_us,
                (ਲੰਮਾ ਲੰਮਾ)gemm_kq_ਸਮਾਂ_us,
                (ਲੰਮਾ ਲੰਮਾ)softmax_ਸਮਾਂ_us,
                (ਲੰਮਾ ਲੰਮਾ)gemm_vkq_ਸਮਾਂ_us,
                ਕੁੱਲ_mb);
    }
}
